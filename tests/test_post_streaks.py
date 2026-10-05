"""Lens streaks ([post] streaks, mkmmd.post.Streaks): the picture's brightest lights smear sideways into thin tinted lines,
alike in every output; dim light and the screen type never streak."""
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from mkmmd import post as P
from mkmmd.cli import main as MAIN
from mkmmd.core import palette as PAL
from mkmmd.core import screentype as SR

MOON = PAL.get("rose-pine-moon")


def _frame(w, h, spots):
    img = np.full((h, w, 3), 0.12, np.float32)
    for (x, y), v in spots:
        img[y - 2:y + 2, x - 2:x + 2] = v
    return img


def test_a_lamp_draws_a_thin_tinted_line_sideways_and_dim_light_draws_none():
    off_white = PAL.srgb(PAL.resolve("text", MOON))                         # the lyrics' colour
    img = _frame(960, 540, [((480, 270), 1.0), ((200, 100), 0.6), ((700, 420), off_white)])
    d = P.Streaks({}, MOON, (960, 540))(img) - img
    assert d[270, 580].mean() > 0.03 and d[270, 700].mean() > 0.01          # far along its row, both ways
    assert d[270, 380].mean() == pytest.approx(d[270, 580].mean(), rel=0.05)
    assert np.abs(d[255, 480]).max() < 1e-4 and np.abs(d[285, 480]).max() < 1e-4   # nothing above or below it
    assert np.abs(d[100, 230]).max() < 1e-6 and np.abs(d[420, 730]).max() < 1e-6   # a dim spot, the off-white: nothing
    r, g, b = d[270, 580]
    assert g > r and b > r                                                  # toward foam, a cool cyan


def test_a_wide_and_a_tall_output_streak_alike():
    wide = P.Streaks({}, MOON, (960, 540))(_frame(960, 540, [((480, 270), 1.0)]))
    tall = P.Streaks({}, MOON, (540, 960))(_frame(540, 960, [((270, 480), 1.0)]))
    for dx in (40, 150):                                                    # the same falloff in pixels
        assert wide[270, 480 + dx].mean() == pytest.approx(tall[480, 270 + dx].mean(), rel=0.1)


def test_an_unknown_key_is_an_error():
    with pytest.raises(ValueError, match="unknown key"):
        P.Streaks({"lenght": 0.2}, MOON, (64, 36))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_post_streaks_the_lights_in_the_picture_but_never_the_screen_type(tmp_path, capsys):
    W, H = 192, 108
    (tmp_path / "mk.toml").write_text('[project]\nname = "t"\nfps = 10\nframe0 = 5\nduration = 0.2\n[[output]]\nname = "wide"\n'
                                      'size = [192, 108]\n[post]\nstreaks = { strength = 0.8 }\n', encoding="utf-8")
    d = tmp_path / "renders" / "draft" / "wide"
    d.mkdir(parents=True)
    for f in (5, 6):
        a = np.full((H, W, 3), 40, np.uint8)
        a[28:32, 94:98] = 255                                               # a lamp in the picture, row 30
        Image.fromarray(a, "RGB").save(d / f"{f:05d}.png")
    lay = np.zeros((H, W, 4), np.uint8)
    lay[78:82, 94:98] = 255                                                 # white type on the screen layer, row 80
    (d / "screen").mkdir()
    Image.fromarray(lay, "RGBA").save(d / SR.layer_rel(5))
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--no-audio", "--crf", "1", "--project", str(tmp_path)])
    assert e.value.code == 0, capsys.readouterr().out
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(tmp_path / "out" / "t_wide_draft.mp4"), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    v = np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3).astype(int)
    assert v[30, 120].mean() - v[15, 120].mean() > 12                     # beside the lamp, on its row: a streak
    assert abs(v[80, 120].mean() - v[65, 120].mean()) < 6                  # beside the type, on its row: nothing
