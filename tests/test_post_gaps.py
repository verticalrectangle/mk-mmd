"""`mk post --allow-gaps` (docs/design.md: Post): a missing frame repeats the one before it, and a gap at the very start of
the clip (a render of a later stretch only) takes the first frame that was rendered instead of failing. Tiny synthetic
frames, ffmpeg, no Blender."""
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from mkmmd.cli import main as MAIN

W, H = 40, 20
BLUE = (30, 60, 200)


def clip(tmp_path, rendered):
    """A clip of four frames (5..8) at 10 fps of which only `rendered` are on disk, each a flat blue."""
    (tmp_path / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = 10\nframe0 = 5\nduration = 0.4\n[[output]]\nname = "sq"\nsize = [{W}, {H}]\n'
        f'[post]\nfloor = false\nvignette = 0.0\ngrain = {{ amount = 0.0 }}\n', encoding="utf-8")
    d = tmp_path / "renders" / "draft" / "sq"
    d.mkdir(parents=True)
    for f in rendered:
        Image.new("RGB", (W, H), BLUE).save(d / f"{f:05d}.png")


def run_post(tmp_path, capsys, *extra):
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--project", str(tmp_path), "--no-audio", *extra])
    out = capsys.readouterr().out
    return e.value.code, json.loads(out[out.index("{"):])


def frames_of(video):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def test_a_gap_at_the_start_takes_the_first_rendered_frame(tmp_path, capsys):
    clip(tmp_path, [7, 8])
    code, rep = run_post(tmp_path, capsys, "--allow-gaps")
    assert code == 0, rep
    o = rep["outputs"]["sq"]
    assert o["frames"] == 4 and o["missing"] == 2
    v = frames_of(o["video"]).astype(int)
    assert len(v) == 4 and v[..., 2].mean() > 150                                       # the blue picture, not black
    assert all(np.abs(v[k] - v[2]).max() <= 2 for k in (0, 1))                           # the lead-in is frame 7's picture


def test_gaps_without_the_flag_are_still_an_error(tmp_path, capsys):
    clip(tmp_path, [7, 8])
    code, rep = run_post(tmp_path, capsys)
    assert code == 2 and "frames missing" in rep["error"]
