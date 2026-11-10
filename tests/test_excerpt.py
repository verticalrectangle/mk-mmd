"""[[excerpt]]: spans of the clip written as videos of their own (docs/design.md: Post, Excerpts)."""
import shutil
import subprocess

import numpy as np
import pytest

from mkmmd.cli import post as P
from mkmmd.core import excerpt as EX

OUTS = ["16x9", "1x1"]


@pytest.mark.parametrize("spec, frag", [
    ({"from": 1.0, "to": 2.0}, "`name` is required"),
    ({"name": "a b", "from": 1.0, "to": 2.0}, "letters, digits"),
    ({"name": "loop", "from": 2.0, "to": 1.0}, "no span inside the clip"),
    ({"name": "loop", "from": 1.0, "to": 9.0}, "no span inside the clip"),
    ({"name": "loop", "from": 1.0, "to": 2.0, "outputs": ["4x3"]}, "one or a list of 16x9, 1x1"),
    ({"name": "loop", "from": 1.0, "to": 2.0, "fade": 1}, "unknown keys ['fade']"),
])
def test_bad_excerpts_say_what_is_wrong(spec, frag):
    with pytest.raises(EX.ExcerptError) as e:
        EX.normalize([spec], 4.0, 30, OUTS)
    assert frag in str(e.value)


def test_an_excerpt_holds_the_frames_a_shot_over_its_span_holds():
    spec = {"name": "loop", "from": 33.5311, "to": 40.9803, "outputs": "1x1", "loop": True}
    e, = EX.normalize([spec], 48.4203, 30, OUTS)
    assert (e["first"], e["frames"], e["outputs"]) == (1006, 223, ["1x1"])     # round(from x fps) up to round(to x fps)
    assert EX.normalize([{"name": "all", "from": 0.0, "to": 4.0}], 4.0, 30, OUTS)[0]["outputs"] == OUTS
    assert "afade" in EX.filters({**e, "loop": False}, 30)[1] and "afade" not in EX.filters(e, 30)[1]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_encoder_writes_an_excerpts_frames_to_a_file_of_its_own(tmp_path):
    w, h, n = 32, 16, 12
    e = {**EX.normalize([{"name": "mid", "from": 0.1, "to": 0.3}], 0.4, 30, ["t"])[0], "out": tmp_path / "mid.mp4"}
    ff = P.encode(range(n), (w, h), 30, tmp_path / "all.mp4", crf=1, preset="ultrafast", excerpts=[e])
    for k in range(n):                                              # frame k is a flat grey, lighter by 20 a frame
        ff.stdin.write(np.full((h, w, 3), 20 * k, np.uint8).tobytes())
    ff.stdin.close()
    assert ff.wait() == 0

    def levels(path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, h * w).mean(1)
    whole, cut = levels(tmp_path / "all.mp4"), levels(tmp_path / "mid.mp4")
    assert len(whole) == n and np.all(np.diff(whole) > 10)
    assert cut == pytest.approx(whole[3:9], abs=2.0)                # frames 3 to 8 of the clip, and no other


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("rate, encoded", [(96000, 48000), (44100, 44100)])
def test_the_song_is_encoded_at_a_rate_players_take_in_the_video_and_in_its_excerpt(tmp_path, rate, encoded):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration=1",
                    "-c:a", "pcm_s24le", str(tmp_path / "song.wav")], check=True)
    e = {**EX.normalize([{"name": "mid", "from": 0.1, "to": 0.3}], 0.4, 30, ["t"])[0], "out": tmp_path / "mid.mp4"}
    ff = P.encode(range(12), (32, 16), 30, tmp_path / "all.mp4", crf=30, preset="ultrafast", excerpts=[e],
                  audio={"file": str(tmp_path / "song.wav"), "start": 0.0, "duration": 0.4})
    ff.stdin.write(bytes(12 * 32 * 16 * 3))
    ff.stdin.close()
    assert ff.wait() == 0
    for name in ("all.mp4", "mid.mp4"):
        got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                              "-of", "csv=p=0", str(tmp_path / name)], capture_output=True, text=True, check=True)
        assert int(got.stdout.strip()) == encoded
