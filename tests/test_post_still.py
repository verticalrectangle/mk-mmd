"""`mk post` for a picture (docs/design.md: Post): a clip of one frame is a still, written as `<name>_<output>[_<preset>].png` at
its own size with no sound and no ffmpeg; `[post] screen = "over"` lays the screen type over the finished, graded picture
(its colour exactly as written, no grain on it: type set on a photograph) instead of under the grade. Tiny synthetic frames,
no Blender."""
import json

import numpy as np
import pytest
from PIL import Image

from mkmmd.cli import main as MAIN
from mkmmd.core import screentype as SR

W, H = 41, 21                                     # odd: a video would be cut to 40 x 20
GREEN = (0, 200, 0)


def still(tmp_path, post=""):
    (tmp_path / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = 10\nframe0 = 5\nduration = 0.1\n[[output]]\nname = "sq"\nsize = [{W}, {H}]\n'
        f'[audio]\nfile = "{tmp_path / "missing.wav"}"\n[post]\n{post}', encoding="utf-8")
    d = tmp_path / "renders" / "draft" / "sq"
    (d / "screen").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(60, 200, (H, W, 3), dtype=np.uint8), "RGB").save(d / "00005.png")
    a = np.zeros((H, W, 4), np.uint8)
    a[5:15, 5:20] = (*GREEN, 255)
    Image.fromarray(a, "RGBA").save(d / SR.layer_rel(5))
    return d


def run_post(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--project", str(tmp_path)])
    out = capsys.readouterr().out
    return e.value.code, json.loads(out[out.index("{"):])


def test_a_clip_of_one_frame_is_a_png_at_its_own_size_without_sound(tmp_path, capsys):
    still(tmp_path, "grain = { amount = 0.0 }\n")
    code, rep = run_post(tmp_path, capsys)
    assert code == 0, rep
    o = rep["outputs"]["sq"]
    assert rep["audio"] is False and o["image"].endswith("t_sq_draft.png") and o["size"] == [W, H]
    assert Image.open(o["image"]).size == (W, H) and not list((tmp_path / "out").glob("*.mp4"))


def test_screen_over_lays_the_type_on_the_graded_picture_in_its_own_colour(tmp_path, capsys):
    """Black and white with heavy grain: over, the type keeps its green exactly; graded (the default), it is grey and
    grained like the picture under it."""
    grade = "saturation = 0.0\ncontrast = 1.2\nfloor = false\ngrain = { amount = 0.1 }\n"
    still(tmp_path, 'screen = "over"\n' + grade)
    assert run_post(tmp_path, capsys)[0] == 0
    v = np.asarray(Image.open(tmp_path / "out" / "t_sq_draft.png").convert("RGB"), int)
    assert (v[5:15, 5:20] == GREEN).all()
    assert np.ptp(v[0:5], axis=-1).max() <= 1                                      # the picture: grey
    still(tmp_path, grade)
    assert run_post(tmp_path, capsys)[0] == 0
    g = np.asarray(Image.open(tmp_path / "out" / "t_sq_draft.png").convert("RGB"), int)[5:15, 5:20]
    assert np.ptp(g, axis=-1).max() <= 1 and g.std() > 3                           # grey and grained


def test_an_unknown_screen_value_is_refused(tmp_path, capsys):
    still(tmp_path, 'screen = "above"\n')
    code, rep = run_post(tmp_path, capsys)
    assert code == 2 and "screen = 'above'" in rep["error"] and '"over"' in rep["error"]
