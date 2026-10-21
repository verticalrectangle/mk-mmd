"""`[post] keep` (docs/design.md: Post): a black-and-white grade that keeps a cast member's eyes in colour. `mk post` saturates
each frame by the mask `mk render` draws beside it (`keep/<frame>.png`): the kept saturation where the mask is 1, the grade's
own where it is 0, between them by its share; screen type laid over a kept spot under the grade is graded as type. Tiny
synthetic stills (a clip of one frame: the graded frame is written as a PNG), no Blender."""
import json
import re

import numpy as np
import pytest
from PIL import Image

from mkmmd.cli import main as MAIN
from mkmmd.core import keep as KP
from mkmmd.core import screentype as SR

W, H = 40, 20
RED = (200, 40, 40)
PLAIN = 'saturation = 0.0\ncontrast = 1.0\nvignette = 0.0\nfloor = false\ngrain = { amount = 0.0 }\n'


def still(tmp_path, post, cast="eve"):
    """A one-frame project with a cast member `cast` and the [post] table `post`; returns the frame folder."""
    (tmp_path / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = 10\nframe0 = 5\nduration = 0.1\n[[output]]\nname = "sq"\nsize = [{W}, {H}]\n'
        f'[[cast]]\nname = "{cast}"\npmx = "eve.pmx"\n[post]\n{post}', encoding="utf-8")
    d = tmp_path / "renders" / "draft" / "sq"
    (d / "keep").mkdir(parents=True)
    Image.new("RGB", (W, H), RED).save(d / "00005.png")
    return d


def mask(path, cols):
    """The keep mask: `cols` {(x0, x1): value 0..1} over the full height, 0 elsewhere."""
    m = np.zeros((H, W), np.uint8)
    for (x0, x1), v in cols.items():
        m[:, x0:x1] = round(v * 255)
    Image.fromarray(m, "L").save(path)


def run_post(tmp_path, capsys, *extra):
    with pytest.raises(SystemExit) as e:
        MAIN.main(["post", "--preset", "draft", "--project", str(tmp_path), *extra])
    out = capsys.readouterr().out
    return e.value.code, json.loads(out[out.index("{"):])


def picture(tmp_path):
    return np.asarray(Image.open(tmp_path / "out" / "t_sq_draft.png").convert("RGB"), int)


def test_the_kept_material_keeps_its_colour_and_the_rest_is_grey(tmp_path, capsys):
    """Left of x 10 the mask is 1 (her iris whole), 10-20 a half (its antialiased edge), right of 20 none: the left keeps
    the frame's red, the right is its grey, the half between."""
    d = still(tmp_path, PLAIN + 'keep = { cast = "eve", materials = ["瞳"] }\n')
    mask(d / KP.rel(5), {(0, 10): 1.0, (10, 20): 0.5})
    code, rep = run_post(tmp_path, capsys)
    assert code == 0, rep
    assert rep["outputs"]["sq"]["keep_frames"] == 1
    v = picture(tmp_path)
    grey = round(0.2126 * RED[0] + 0.7152 * RED[1] + 0.0722 * RED[2])
    assert np.abs(v[:, :10] - RED).max() <= 1                                     # kept: the frame's own colour
    assert np.abs(v[:, 20:] - grey).max() <= 1                                    # the rest: grey
    half = v[:, 10:20]
    assert np.abs(half[..., 0] - (grey + RED[0]) / 2).max() <= 2 and np.abs(half[..., 1] - (grey + RED[1]) / 2).max() <= 2


def test_keep_saturation_strengthens_the_kept_colour(tmp_path, capsys):
    d = still(tmp_path, PLAIN + 'keep = { cast = "eve", materials = ["瞳"], saturation = 1.5 }\n')
    mask(d / KP.rel(5), {(0, W): 1.0})
    assert run_post(tmp_path, capsys)[0] == 0
    v = picture(tmp_path)
    y = 0.2126 * RED[0] + 0.7152 * RED[1] + 0.0722 * RED[2]
    want = np.clip([y + (c - y) * 1.5 for c in RED], 0, 255)
    assert np.abs(v - want).max() <= 1


def test_screen_type_over_a_kept_spot_is_graded_as_type(tmp_path, capsys):
    """Green type laid on the kept left half under the grade (`screen = "graded"`, the default) turns grey like the rest
    of the type: the mask is cut where the type covers the picture."""
    d = still(tmp_path, PLAIN + 'keep = { cast = "eve", materials = ["瞳"] }\n')
    mask(d / KP.rel(5), {(0, W): 1.0})
    a = np.zeros((H, W, 4), np.uint8)
    a[5:15, 0:10] = (0, 200, 0, 255)
    (d / "screen").mkdir()
    Image.fromarray(a, "RGBA").save(d / SR.layer_rel(5))
    assert run_post(tmp_path, capsys)[0] == 0
    v = picture(tmp_path)
    assert np.ptp(v[5:15, 0:10], axis=-1).max() <= 1                             # the type: grey
    assert np.abs(v[0:5, 0:10] - RED).max() <= 1                                  # the kept picture beside it: red


def test_a_missing_mask_is_an_error_unless_gaps_are_allowed_then_the_frame_is_graded_whole(tmp_path, capsys):
    still(tmp_path, PLAIN + 'keep = { cast = "eve", materials = ["瞳"] }\n')
    code, rep = run_post(tmp_path, capsys)
    assert code == 2 and "1 keep masks missing" in rep["error"] and "mk render" in rep["error"]
    code, rep = run_post(tmp_path, capsys, "--allow-gaps")
    assert code == 0 and rep["outputs"]["sq"]["keep_frames"] == 0
    assert np.ptp(picture(tmp_path), axis=-1).max() <= 1                          # all grey


def test_a_keep_table_that_names_no_cast_member_or_no_materials_is_refused():
    cast = [{"name": "eve"}, {"name": "rin"}]
    assert KP.spec({"keep": {"cast": "eve", "materials": "瞳"}}, cast) == {"cast": "eve", "materials": ["瞳"],
                                                                          "saturation": 1.0}
    assert KP.spec({}, cast) is None
    for keep, says in (({"cast": "len", "materials": ["瞳"]}, "have eve, rin"), ({"cast": "eve"}, "materials = [...]"),
                       ({"cast": "eve", "materials": ["瞳"], "satur": 1}, "unknown key satur"),
                       ({"cast": "eve", "materials": ["瞳"], "saturation": -1}, "a number >= 0"), ("eve", "is a table")):
        with pytest.raises(ValueError, match=re.escape(says)):
            KP.spec({"keep": keep}, cast)
