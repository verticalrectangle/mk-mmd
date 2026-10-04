"""`mk post` lays the screen layers (`screen/<frame>.png`, docs/design.md: Text, Screen type) over the cut's frames, after the cut
effects and before the grade; a frame without a layer is untouched and a missing frame repeats the previous one with its layer.
Tiny synthetic frames; ffmpeg decodes what was encoded."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from mkmmd import post as P
from mkmmd.cli import main as MAIN
from mkmmd.core import screentype as SR

W, H = 32, 18
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def frame(path, rgb):
    Image.new("RGB", (W, H), rgb).save(path)


def layer(path, box, rgba):
    """A straight RGBA layer: transparent but for `box` (x0, y0, x1, y1) in `rgba`."""
    a = np.zeros((H, W, 4), np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = rgba
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, "RGBA").save(path)


def decode(video, n):
    """The first n frames of an encoded video as (n, H, W, 3) uint8."""
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt",
                        "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(n, H, W, 3)


def project(tmp_path):
    (tmp_path / "mk.toml").write_text(
        '[project]\nname = "t"\nfps = 10\nframe0 = 5\nduration = 0.4\n[[output]]\nname = "wide"\nsize = [32, 18]\n', encoding="utf-8")
    d = tmp_path / "renders" / "draft" / "wide"
    d.mkdir(parents=True)
    return d


def run_post(tmp_path, capsys, *extra):
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--no-audio", "--crf", "1", "--project", str(tmp_path), *extra])
    out = capsys.readouterr().out
    return e.value.code, json.loads(out[out.index("{"):])


def test_layer_rel_names_the_file_by_frame_and_read_rgba_gives_straight_floats(tmp_path):
    assert SR.layer_rel(7) == "screen/00007.png" and SR.layer_rel(12345) == "screen/12345.png"
    layer(tmp_path / "l.png", (2, 3, 6, 5), (200, 100, 50, 128))
    a = P.read_rgba(tmp_path / "l.png")
    assert a.shape == (H, W, 4) and a.dtype == np.float32
    assert a[3, 2].tolist() == pytest.approx([200 / 255, 100 / 255, 50 / 255, 128 / 255], abs=1e-6)       # straight, not premultiplied
    assert a[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert P.read_rgba(tmp_path / "nope.png") is None
    assert P.read_rgba(tmp_path / "l.png", (W - 1, H - 1)).shape == (H - 1, W - 1, 4)                       # a pixel too big: cut


def test_post_lays_the_layer_over_the_frame_and_leaves_frames_without_one_alone(tmp_path, capsys):
    d = project(tmp_path)
    for f in range(5, 9):
        frame(d / f"{f:05d}.png", (40, 40, 40))
    layer(d / SR.layer_rel(6), (4, 4, 12, 10), (255, 0, 0, 255))
    layer(d / SR.layer_rel(7), (4, 4, 12, 10), (0, 0, 255, 255))
    code, rep = run_post(tmp_path, capsys)
    assert code == 0 and rep["outputs"]["wide"]["screen_type_frames"] == 2
    v = decode(tmp_path / "out" / "t_wide_draft.mp4", 4)
    assert v[1, 6, 8].tolist() == pytest.approx([255, 0, 0], abs=40) and v[2, 6, 8].tolist()[2] > 200       # red, then blue
    assert abs(int(v[0, 6, 8, 0]) - 40) < 12 and abs(int(v[3, 6, 8, 2]) - 40) < 12                          # no layer: the frame
    assert abs(int(v[1, 1, 1, 0]) - 40) < 12                                                                # outside the box too


def test_a_half_transparent_layer_blends_in_display_space(tmp_path, capsys):
    d = project(tmp_path)
    for f in range(5, 9):
        frame(d / f"{f:05d}.png", (100, 100, 100))
    layer(d / SR.layer_rel(5), (0, 0, W, H), (200, 0, 0, 128))
    run_post(tmp_path, capsys)
    v = decode(tmp_path / "out" / "t_wide_draft.mp4", 1)
    want = 100 + (200 - 100) * 128 / 255
    assert float(v[0, 9, 16, 0]) == pytest.approx(want, abs=14) and float(v[0, 9, 16, 1]) < 100


def test_a_missing_frame_repeats_the_previous_one_with_its_layer(tmp_path, capsys):
    d = project(tmp_path)
    for f in (5, 6, 8):
        frame(d / f"{f:05d}.png", (40, 40, 40))
    layer(d / SR.layer_rel(6), (4, 4, 12, 10), (255, 0, 0, 255))
    code, rep = run_post(tmp_path, capsys, "--allow-gaps")
    v = decode(tmp_path / "out" / "t_wide_draft.mp4", 4)
    assert code == 0 and rep["outputs"]["wide"]["missing"] == 1
    assert v[1, 6, 8, 0] > 200 and v[2, 6, 8, 0] > 200                       # frame 7 is frame 6 again, red box and all
    assert abs(int(v[3, 6, 8, 0]) - 40) < 12                                  # frame 8 has none
