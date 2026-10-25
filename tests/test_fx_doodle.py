"""Doodles ([[fx]] kind "doodle", mkmmd.core.picturefx and mkmmd.picturefx): their keys, the scene points their marks
ask mk render for (projected through the shot in the cut), the pop's spring, the boil, the shapes, and a mark drawn as a
coloured pen line with a white rim where its point lands, sized by its depth with `size_m`."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import picturefx as PC
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 160, 90
EYE = {"point": 'bone("eye.L").head'}


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


@pytest.mark.parametrize("marks, frag", [
    ([], "`marks` is required"),
    ([{"shape": "moon", "at": [0.5, 0.5]}], "none of star"),
    ([{"shape": "x"}], "`at` is required"),
    ([{"shape": "x", "at": EYE, "size": 0.1, "size_m": 0.03}], "not both"),
    ([{"shape": "x", "at": [0.5, 0.5], "size_m": 0.03}], "scene point's depth"),
    ([{"shape": "x", "at": [0.5, 0.5], "t": 3.5}], "outside the span"),
    ([{"shape": "x", "at": [0.5, 0.5], "colour": "love"}], "unknown keys"),
    ([{"shape": "x", "at": {"point": ""}}], "expected {point"),
])
def test_bad_marks_say_what_is_wrong(marks, frag):
    with pytest.raises(FX.FxError, match=frag.replace("{", r"\{").replace("(", r"\(")):
        FX.normalize({"kind": "doodle", "from": 1.0, "to": 3.0, "marks": marks}, DAWN)


def test_marks_at_points_ask_for_their_points_through_the_shot_in_the_cut():
    pl = plan({"kind": "zoom", "from": 0, "to": 1, "times": [0.5]},
              {"kind": "doodle", "from": 1.5, "to": 2.5, "marks": [
                  {"shape": "x", "at": EYE, "size_m": 0.03}, {"shape": "star", "at": [0.2, 0.2]},
                  {"shape": "x", "at": {"point": 'bone("eye.R").head'}}]})
    fx = pl["fx"][1]
    assert FX.points(fx) == [("x1m0", 'bone("eye.L").head'), ("x1m2", 'bone("eye.R").head')]
    dem = TR.demands(pl)
    f_a, f_b = FX.frame_of(1.6, FPS, F0), FX.frame_of(2.2, FPS, F0)
    assert {"kind": "point", "key": "x1m0", "expr": 'bone("eye.L").head', "shot": "a"} in dem[f_a]
    assert {"kind": "point", "key": "x1m2", "expr": 'bone("eye.R").head', "shot": "b"} in dem[f_b]
    assert FX.frame_of(2.5, FPS, F0) not in dem


def test_the_pop_springs_in_and_settles():
    a = np.linspace(0.0, 1.0, 401)
    s = np.array([PC.pop(x) for x in a])
    assert PC.pop(0.0) == 0.0 and PC.pop(-0.1) == 0.0
    peak = a[s.argmax()]
    assert 1.08 < s.max() < 1.18 and 0.1 < peak < 0.2                      # a small overshoot near half the spring
    assert abs(PC.pop(0.6) - 1.0) < 0.01


@pytest.mark.parametrize("name", PC.SHAPES)
def test_every_shape_fits_its_box(name):
    strokes = PC.shape(name)
    pts = np.concatenate(strokes)
    assert all(len(p) >= 2 for p in strokes)
    assert np.abs(pts).max() <= 0.52 and np.ptp(pts[:, 0]) > 0.3 and np.ptp(pts[:, 1]) > 0.3


def test_the_boil_redraws_the_same_way_for_a_step():
    s = PC.shape("heart")
    one = PC.boil(s, np.random.default_rng([1, 2, 3]), 0.02)
    two = PC.boil(s, np.random.default_rng([1, 2, 3]), 0.02)
    other = PC.boil(s, np.random.default_rng([1, 2, 4]), 0.02)
    assert all(np.array_equal(a, b) for a, b in zip(one, two))
    gap = np.abs(one[0] - other[0]).max()
    assert 0.005 < gap < 0.15                                               # a hand's wobble, still the heart
    fx = FX.normalize({"kind": "doodle", "from": 1.0, "to": 3.0, "fps": 8, "marks": [{"shape": "x", "at": [0, 0]}]}, DAWN)
    assert [PC.boil_step(fx, 1.0 + i / 30) for i in range(9)] == [0, 0, 0, 0, 1, 1, 1, 1, 2]


def test_a_mark_is_drawn_where_its_point_lands_after_it_pops_in():
    w, h = 640, 360                                                         # a third of 1080: px at 1080 x 1/3
    pl = TR.plan({"project": {"duration": 4.0}, "output": [{"name": "t", "size": [w, h]}],
                  "shot": [{"name": "a", "from": 0.0, "to": 4.0}],
                  "fx": [{"kind": "doodle", "from": 1.0, "to": 2.0, "width": 18, "outline": 9, "marks": [
                      {"shape": "x", "at": EYE, "size_m": 0.4, "color": "pine", "t": 1.2}]}]}, FPS, F0, DAWN)
    pine = np.asarray(FX.normalize({"kind": "tint", "from": 0, "to": 1, "color": "pine"}, DAWN)["color"], np.float32)
    point = {}
    ffx = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=lambda g: None,
                     point_of=lambda key, g: point.setdefault((key, g), ((320.0, 180.0), 0.5)))
    img = np.full((h, w, 3), 0.1, np.float32)
    before = ffx.frame(FX.frame_of(1.15, FPS, F0), img)
    assert np.array_equal(before, img)                                     # not yet
    f = FX.frame_of(1.8, FPS, F0)
    out = ffx.frame(f, img)
    assert ("x0m0", f) in point
    ink = np.all(np.abs(out - pine) < 0.02, axis=2)
    rim = np.all(out > 0.98, axis=2)
    ys, xs = np.nonzero(ink)
    assert ink.sum() > 400 and rim.sum() > 400
    assert abs(xs.mean() - 320) < 6 and abs(ys.mean() - 180) < 6          # centred on the point
    span = 0.4 * 0.5 * h                                                     # 0.4 m at half a frame height per metre
    assert 0.6 * span < np.ptp(xs) < 1.1 * span


def test_strokes_are_drawn_inside_the_frame_only():
    img = np.zeros((20, 30, 3), np.float32)
    PF.draw_strokes(img, [np.array([[-50.0, -50.0], [10.0, 10.0], [100.0, 5.0]])], (1.0, 0.0, 0.0), 3.0, 2.0)
    assert img[..., 0].max() == pytest.approx(1.0) and np.isfinite(img).all()
    assert PF.draw_strokes(img, [np.array([[-90.0, -90.0], [-80.0, -85.0]])], (0, 1, 0), 3, 2) is img
