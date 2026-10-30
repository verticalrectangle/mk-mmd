"""The paper looks of the vector style (mkmmd.core.crayon, mkmmd.core.cutout, mkmmd.core.shotstyle): a sketch in crayon
(waxy lines on the paper's tooth, each part's in its own crayon, coloured in with `fill`) and a look cut out of
construction paper (`cutout`: no lines, rough edges, a white rim, shadows), checked and drawn from made-up passes."""
import numpy as np
import pytest

from mkmmd.core import crayon as CR
from mkmmd.core import cutout as CU
from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS

DAWN = PAL.get("rose-pine-dawn")
VEC = {"colors": {"background": "#F0E8D8", "line": "#202020"},
       "tones": {"red": "#D03030", "blue": {"lit": "#3050D0", "shade": "#203090"}},
       "materials": [{"match": "left", "tone": "red"}, {"match": "right", "tone": "blue"}]}


def _vec(shot):
    return SS.normalize({"name": "v", "style": "vector", **shot}, DAWN, VEC)["vector"]


def _passes(size=240, near=None):
    """Two materials side by side on a background: id 1 (left, red) and 2 (right, blue), depth 3 m (`near`: id 1 at that
    depth instead), all lit."""
    ids = np.zeros((size, size), np.int32)
    ids[40:200, 40:120], ids[40:200, 120:200] = 1, 2
    depth = np.full(ids.shape, 3.0)
    if near is not None:
        depth[ids == 1] = near
    return {"id": ids, "depth": depth, "shade": np.full(ids.shape, 0.4, np.float32)}


def _draw(shot, t=1.0, **kw):
    spec = _vec(shot)
    return SS.compose_vector(_passes(**kw), spec, SS.vector_table(["left", "right"], spec), scale=1, t=t)


# ---------------------------------------------------------------- crayon textures
def test_the_paper_tooth_is_fixed_by_its_seed_and_spread_about_the_middle():
    a, b = CR.tooth((64, 80), 1.0, 3), CR.tooth((64, 80), 1.0, 3)
    assert a is b and not a.flags.writeable                                  # kept: the sheet never moves
    assert not np.array_equal(a, CR.tooth((64, 80), 1.0, 4))
    assert 0.4 < float(a.mean()) < 0.6 and 0.0 <= float(a.min()) and float(a.max()) <= 1.0


def test_wax_takes_more_of_the_tooth_the_harder_the_crayon_is_pressed():
    grain = CR.tooth((64, 64), 1.0, 1)
    shares = [float(CR.wax(np.float32(p), grain).mean()) for p in (0.2, 0.5, 0.8)]
    assert shares[0] < shares[1] < shares[2]


def test_a_stroke_covers_its_line_grainy_and_nothing_far_from_it():
    m = np.zeros((60, 120), bool)
    m[27:33, 10:110] = True
    a = CR.stroke(m, 6.0, 1.0, np.random.default_rng(1), CR.tooth(m.shape, 1.0, 1), 0.8)
    assert a[:15].max() == 0.0 and a[45:].max() == 0.0                       # the line, not the page
    core = a[29:31, 20:100]
    assert 0.4 < float(core.mean()) < 0.98 and float(core.min()) < 0.5       # waxy: the tooth shows through


def test_a_scribble_colours_its_region_in_strokes_and_stays_in_it():
    region = np.zeros((80, 80), bool)
    region[10:70, 10:70] = True
    a = CR.scribble(region, 1.0, np.random.default_rng(2), CR.tooth(region.shape, 1.0, 1), 30.0, 0.9)
    assert a[~region].max() == 0.0
    assert 0.2 < float(a[region].mean()) < 0.95


def test_the_box_blur_keeps_the_mean_of_a_flat_picture():
    x = np.full((20, 30), 0.25, np.float32)
    assert np.allclose(CR.blur(x, 3), 0.25, atol=1e-5)


# ---------------------------------------------------------------- the crayon sketch
@pytest.mark.parametrize("sketch, frag", [
    ({"pen": "brush"}, "known: pencil, crayon"),
    ({"inks": ["love"]}, "a crayon's"),
    ({"fill": 0.5}, "a crayon's"),
    ({"pen": "crayon", "fill": 2}, "a share in 0..1"),
    ({"pen": "crayon", "inks": []}, "tones"),
])
def test_bad_crayons_say_what_is_wrong(sketch, frag):
    with pytest.raises(SS.StyleError, match=frag):
        _vec({"sketch": sketch})


def test_a_crayon_is_waxier_than_a_pencil_by_default():
    assert _vec({"sketch": {"pen": "crayon"}})["sketch"]["grain"] == SS.CRAYON_GRAIN
    assert _vec({"sketch": True})["sketch"]["grain"] == SS.SKETCH["grain"]
    assert _vec({"sketch": {"pen": "crayon", "grain": 0.1}})["sketch"]["grain"] == 0.1


def test_a_crayon_draws_each_parts_lines_in_its_own_crayon_on_paper():
    out = _draw({"lines": {"outline": 8, "inner": 6},
                 "sketch": {"pen": "crayon", "paper": "#FFFFFF", "ink": "#000000", "inks": ["#FF0000", "#0000FF"],
                            "strokes": 1, "boil": 0.0}})
    assert np.allclose(out[120, 80], 1.0) and np.allclose(out[10, 10], 1.0)  # fills and the background: paper
    left, right = out[60:180, 36:44].reshape(-1, 3), out[60:180, 196:204].reshape(-1, 3)
    redness = lambda px: float(((px[:, 0] - px[:, 2]) > 0.3).mean())       # noqa: E731
    assert redness(left) > 0.3 and redness(right) < 0.05                     # the left part's outline in red
    blueness = float(((right[:, 2] - right[:, 0]) > 0.3).mean())
    assert blueness > 0.3                                                    # the right part's in blue


def test_crayon_lines_in_their_tones_take_each_parts_colour_toward_the_ink():
    out = _draw({"lines": {"outline": 8, "inner": 6},
                 "sketch": {"pen": "crayon", "paper": "#FFFFFF", "ink": "#000000", "inks": "tones", "strokes": 1,
                            "boil": 0.0}})
    left = out[60:180, 36:44].reshape(-1, 3)
    dark = left[left.sum(1) < 2.5]
    assert len(dark) and float(np.median(dark[:, 0] - dark[:, 2])) > 0.15     # red (toward black), not grey


def test_a_crayon_fill_colours_the_parts_in_and_needs_their_tones():
    spec = _vec({"sketch": {"pen": "crayon", "paper": "#FFFFFF", "ink": "#000000", "fill": 0.9, "boil": 0.0}})
    assert SS.draws_fills(spec) and not SS.draws_fills(_vec({"sketch": True}))
    out = _draw({"sketch": {"pen": "crayon", "paper": "#FFFFFF", "ink": "#000000", "fill": 0.9, "boil": 0.0}})
    inside = out[70:170, 60:100].reshape(-1, 3)
    assert float((inside[:, 0] - inside[:, 1] > 0.2).mean()) > 0.3           # red wax in the left part
    assert float((inside.min(1) > 0.95).mean()) > 0.02                       # and the paper between the strokes


def test_a_crayon_is_drawn_again_fps_times_a_second():
    shot = {"sketch": {"pen": "crayon", "fps": 12, "boil": 2.0}}
    a = _draw(shot, t=1.0)
    assert np.array_equal(_draw(shot, t=1.04), a) and not np.array_equal(_draw(shot, t=1.09), a)


# ---------------------------------------------------------------- the cut-out
@pytest.mark.parametrize("cut, frag", [
    ({"rim": "white"}, "rim"),
    ({"shadow": {"offset": 3}}, "offset"),
    ({"grain": 2}, "out of range"),
    ({"seed": -1}, "whole number"),
    ({"scissors": 1}, "unknown keys"),
])
def test_bad_cutouts_say_what_is_wrong(cut, frag):
    with pytest.raises(SS.StyleError, match=frag):
        _vec({"cutout": cut})


def test_a_look_is_a_sketch_or_a_cutout_not_both_and_a_cutout_needs_the_vector_style():
    with pytest.raises(SS.StyleError, match="not both"):
        _vec({"sketch": True, "cutout": True})
    with pytest.raises(SS.StyleError, match="belong to the vector style"):
        SS.normalize({"name": "s", "style": "silhouette", "cutout": True}, DAWN)
    spec = _vec({"cutout": {"rim": False, "shadow": False}})
    assert spec["cutout"]["rim"] is None and spec["cutout"]["shadow"] is None and SS.draws_fills(spec)


def test_a_cutout_lays_its_pieces_in_their_colours_on_a_white_rim_without_lines():
    out = _draw({"cutout": {"rough": 0.0, "lift": 0.0, "grain": 0.0, "shadow": False}})
    assert np.allclose(out[120, 80], SS.resolve_colour("#D03030", DAWN), atol=0.02)    # the red piece
    assert out[120, 160][2] > out[120, 160][0] + 0.2                                    # the blue piece
    assert np.allclose(out[120, 39], 1.0, atol=0.02)                                   # the white rim round them
    assert np.allclose(out[5, 5], SS.resolve_colour("#F0E8D8", DAWN), atol=0.02)       # the paper behind
    assert out[120, 118:123].min() > 0.15                                              # no ink line between


def test_a_cutout_casts_its_shadow_down_and_right_and_the_near_piece_on_the_far_one():
    shot = {"cutout": {"rough": 0.0, "lift": 0.0, "grain": 0.0, "rim": False,
                       "shadow": {"offset": [0.04, 0.04], "blur": 1.0, "alpha": 0.6}}}
    out = _draw(shot, near=2.0)
    bg = np.asarray(SS.resolve_colour("#F0E8D8", DAWN))
    assert out[205, 205].sum() < bg.sum() - 0.3                              # the shadow below and to the right
    assert np.allclose(out[20, 20], bg, atol=0.02)                           # none up and to the left
    far = np.asarray(SS.resolve_colour("#3050D0", DAWN))
    assert out[150, 123].sum() < far.sum() - 0.05                            # the near red piece shades the blue one
    assert np.allclose(out[150, 190], far, atol=0.02)                        # away from its edge, the blue as it is


def test_a_cutouts_scissors_cut_again_fps_times_a_second():
    shot = {"cutout": {"fps": 8, "rough": 3.0}}
    a = _draw(shot, t=1.0)
    assert np.array_equal(_draw(shot, t=1.1), a) and not np.array_equal(_draw(shot, t=1.13), a)


def test_a_cutout_without_a_figure_is_the_paper_alone():
    spec = _vec({"cutout": True})
    p = _passes()
    out = SS.compose_vector(p, spec, SS.vector_table(["left", "right"], spec), scale=1, subject=False, t=0.0)
    assert out.shape == (240, 240, 3)
    assert np.allclose(out.mean((0, 1)), SS.resolve_colour("#F0E8D8", DAWN), atol=0.03)


def test_cutout_normalize_fills_in_its_defaults():
    o = CU.normalize(True, lambda v, what: [0.0, 0.0, 0.0])
    assert o["rim"]["width"] == CU.RIM["width"] and o["shadow"]["alpha"] == CU.SHADOW["alpha"]
    assert CU.normalize(None, None) is None and CU.normalize(False, None) is None
