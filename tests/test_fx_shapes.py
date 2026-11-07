"""The picture as flat shapes (mkmmd.core.shapes: the entries checked, the cell on the hits, the deal; mkmmd.shapes: a
figure come apart into shapes on an empty ground, inks by shape, shapes that never leave their cells) on synthetic
frames."""
import math

import cv2
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import shapes as SHF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import shapes as SH
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 240, 160
CREAM = (0.98, 0.96, 0.93)
PLUM = (0.34, 0.32, 0.47)


def entries(*specs):
    proj = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(specs)}
    return TR.plan(proj, FPS, F0, DAWN)["fx"]


def shapes(**kw):
    return {"kind": "shapes", "from": 0.0, "to": 4.0, "back": list(CREAM), "pop": 0.0, **kw}


def disc():
    img = np.full((H, W, 3), CREAM, np.float32)
    cv2.circle(img, (W // 2, H // 2), 50, PLUM, -1)
    return img


def run(spec, img, frame=F0 + 30):
    fx = entries(spec)
    return FF.FrameFx(fx, (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None).frame(frame, img)


def test_a_figure_on_a_flat_ground_comes_apart_into_shapes_and_the_ground_stays_empty():
    img = disc()
    out = run(shapes(cell=0.1), img)
    far = np.ones((H, W), bool)
    cv2.circle(far.view(np.uint8), (W // 2, H // 2), 62, 0, -1)
    assert np.array_equal(out[far], img[far])                         # the ground: not a pixel changed
    inside = out[H // 2 - 40:H // 2 + 40, W // 2 - 40:W // 2 + 40]
    on = np.all(np.abs(inside - PLUM) < 1e-3, axis=-1).mean()
    assert 0.3 < on < 0.9                                             # the disc is shapes in its colour, gaps between
    mix = (inside - CREAM) / (np.array(PLUM) - CREAM)                 # and nothing but its colour and the ground's
    assert np.allclose(mix, mix[..., :1], atol=1e-3)


def test_over_the_picture_a_shape_wears_the_colour_under_its_cell_and_the_picture_shows_between():
    img = np.zeros((H, W, 3), np.float32)
    img[:, : W // 2] = (0.9, 0.1, 0.1)
    img[:, W // 2:] = (0.1, 0.1, 0.9)
    out = run(shapes(cell=0.15, back="picture"), img)
    cell = 0.15 * H
    far = np.abs(np.arange(W) - W / 2) > 1.5 * cell
    assert np.array_equal(out[:, far], img[:, far])                  # a shape the colour of its ground, the ground
    near = out[:, ~far]
    assert np.any(np.all(np.abs(near - (0.5, 0.1, 0.5)) < 0.25, axis=-1))   # shapes on the line: the two mixed

def test_inks_by_shape_give_each_kind_of_shape_its_own_colour():
    red, blue = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)
    out = run(shapes(cell=0.08, shape=["circle", "square"], inks=[list(red), list(blue)], pick="shape"), disc())
    seen = {c for c in (red, blue) if np.any(np.all(np.abs(out - c) < 1e-3, axis=-1))}
    assert seen == {red, blue}
    one = run(shapes(cell=0.08, shape="circle", inks=[list(red), list(blue)], pick="shape"), disc())
    assert not np.any(np.all(np.abs(one - blue) < 1e-3, axis=-1))


def test_with_hits_the_cell_changes_only_on_them():
    fx = entries(shapes(cell=[[0.0, 0.4], [4.0, 0.02]], times=[1.5, 3.0]))[0]
    a, since_a, step_a = SH.cell_at(fx, 1.2, [1.5, 3.0])
    b, since_b, step_b = SH.cell_at(fx, 1.4, [1.5, 3.0])
    c, since_c, step_c = SH.cell_at(fx, 1.6, [1.5, 3.0])
    assert (a, since_a, step_a) == (b, since_b, step_b) == (pytest.approx(0.4), 0.0, 0)
    assert c == pytest.approx(0.4 + (0.02 - 0.4) * 1.5 / 4.0) and (since_c, step_c) == (1.5, 1)
    img = disc()
    assert np.array_equal(run(shapes(cell=[[0.0, 0.4], [4.0, 0.02]], times=[1.5, 3.0]), img, F0 + 36),
                          run(shapes(cell=[[0.0, 0.4], [4.0, 0.02]], times=[1.5, 3.0]), img, F0 + 42))


def test_on_a_hit_each_cell_keeps_its_old_shape_until_its_new_one_pops_so_the_picture_never_empties():
    img = np.full((H, W, 3), PLUM, np.float32)
    spec = shapes(cell=[[0.0, 0.2], [4.0, 0.05]], times=[1.0], pop=0.3, stagger=0.2)
    cover = lambda out: float(np.mean(np.abs(out[..., 0] - CREAM[0]) > 0.02))
    steady, after = cover(run(spec, img, F0 + 25)), cover(run(spec, img, F0 + 31))    # before the hit, a frame after
    assert steady > 0.3 and after > 0.7 * steady


def test_a_shape_popping_past_its_size_stops_at_its_cell_and_stays_round():
    img = np.full((H, W, 3), PLUM, np.float32)
    cell = 0.25
    dx, dy = SH.lattice((W, H), cell, "hex")
    peak = float(np.linspace(0, 1, 2001)[np.argmax(SH.pop(np.linspace(0, 1, 2001)))])
    out = SHF.picture(img, cell, "hex", ["circle"], 1.0, 0.0, 0.0, None, None, tuple(CREAM), 0.06, 0.3, 0.0, 1, 0,
                      peak * 0.3, 0.0)
    cov = np.abs(out[..., 0] - CREAM[0]) / abs(PLUM[0] - CREAM[0])
    n, lab = cv2.connectedComponents((cov > 0.5).astype(np.uint8))
    mid = lab[H // 2, W // 2]
    ys, xs = np.nonzero(lab == mid)
    r = 0.5 * min(dx, dy)
    assert xs.max() - xs.min() == pytest.approx(ys.max() - ys.min(), abs=1)       # round, not cut flat by its cell
    assert cov[ys.min() - 2:ys.max() + 3, xs.min() - 2:xs.max() + 3].sum() == pytest.approx(math.pi * r * r, rel=0.03)


@pytest.mark.parametrize("bad, says", [
    ({"back": "picture", "skip": 0.1}, "give the `back` colour"),
    ({"pick": "shape"}, "give the inks"),
    ({"shape": ["circle", "blob"]}, "expected one of"),
    ({"cell": 2.0}, "cell"),
])
def test_a_shapes_entry_that_cannot_be_drawn_is_refused_with_a_reason(bad, says):
    with pytest.raises(TR.TransitionError, match=says):
        entries(shapes(**bad))
