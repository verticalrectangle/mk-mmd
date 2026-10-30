"""The liquid light show ([[fx]] kind "liquid", mkmmd.core.fields and mkmmd.fields): its keys, the inks a function of the
clip second and the seed (the same frames in any order, the same picture about the centre in every output), smooth from
frame to frame away from the hits and swelling on them, laid by its blend and opacity, kept behind her or on her by a
mask, and fast enough for post."""
import re
import time

import numpy as np
import pytest

from mkmmd import fields as FL
from mkmmd import framefx as FF
from mkmmd.core import fields as CF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 160, 90


def plan(*fx, size=(W, H)):
    data = {"project": {"duration": 6.0}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 6.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def runner(pl, figure=None, size=(W, H)):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=(lambda key, g: figure) if figure is not None else None)


def liquid(**kw):
    return {"kind": "liquid", "from": 0.0, "to": 5.0, **kw}


def ground(size=(W, H)):
    return np.full((size[1], size[0], 3), 0.5, np.float32)


@pytest.mark.parametrize("spec, frag", [
    (liquid(inks=[]), "inks = []: expected a list of 1 to 6 colours"),
    (liquid(inks="love"), "expected a list of 1 to 6 colours"),
    (liquid(inks=["love"] * 7), "1 to 6 colours"),
    (liquid(inks=["love", "nope"]), "colour 'nope'"),
    (liquid(blend="add"), "blend 'add' is none of over, screen, multiply, soft-light"),
    (liquid(opacity=-0.1), "opacity = -0.1: expected 0..1"),
    (liquid(scale=0.1), "scale = 0.1: expected 0.2..5"),
    (liquid(pulse=2), "pulse = 2: expected 0..1"),
    (liquid(bubbles=-1), "bubbles = -1: expected 0..40"),
    (liquid(cells=1.5), "cells = 1.5: expected 0..1"),
    (liquid(mask="figure"), "needs the figure's `objects`"),
    (liquid(objects="Reisen*"), "give both or neither"),
    (liquid(lane="kick", times=[1.0]), "one of them"),
    (liquid(spin=3), "unknown keys ['spin']"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, DAWN)


def test_the_same_frames_in_any_order_and_another_seed_another_show():
    pl = plan(liquid())
    a, b = runner(pl), runner(pl)
    forward = [a.frame(F0 + 40 + j, ground()) for j in range(3)]
    assert np.array_equal(b.frame(F0 + 42, ground()), forward[2])
    assert np.array_equal(b.frame(F0 + 40, ground()), forward[0])
    other = runner(plan(liquid(seed=2))).frame(F0 + 40, ground())
    assert float(np.abs(other - forward[0]).mean()) > 0.05


def test_it_flows_smoothly_and_swells_on_a_hit():
    pl = plan(liquid(times=[2.0], pulse=0.3))
    r = runner(pl)
    calm = [r.frame(F0 + 30 + j, ground()) for j in range(8)]               # 1.0 .. 1.23 s: no hit near
    steps = [float(np.abs(x - y).mean()) for x, y in zip(calm, calm[1:])]
    assert max(steps) < 0.012 and min(steps) > 0.0                         # it moves, a little at a time
    near = [float(np.quantile(np.abs(x - y), 0.99)) for x, y in zip(calm, calm[1:])]
    assert max(near) < 0.25                                                 # nothing pops in or out
    before, hit = r.frame(F0 + 59, ground()), r.frame(F0 + 62, ground())   # 1.97 s, then 2.07 s
    assert float(np.abs(hit - before).mean()) > 3.0 * max(steps)            # the hit swells it


def test_the_swell_rises_on_a_hit_and_dies_away():
    assert CF.swell(0.99, [1.0], 0.4) == 0.0
    assert CF.swell(1.0 + CF.ATTACK, [1.0], 0.4) == pytest.approx(1.0)
    assert CF.swell(1.0 + CF.ATTACK + 0.4, [1.0], 0.4) == pytest.approx(np.exp(-1.0))
    assert CF.swell(2.0, [1.0, 1.9], 0.4) > CF.swell(2.0, [1.0], 0.4)       # the latest counts


def test_every_output_shows_the_same_show_about_the_centre():
    wide, square = (192, 108), (108, 108)
    fx = dict(FX.normalize(liquid(bubbles=0), DAWN), index=0, f0=F0, f1=F0 + 150)
    a = FL.liquid(fx, wide, 1.5, (wide[0] / 2 - 0.5, wide[1] / 2 - 0.5), 0.0, FL._Memo())
    b = FL.liquid(fx, square, 1.5, (square[0] / 2 - 0.5, square[1] / 2 - 0.5), 0.0, FL._Memo())
    mid = a[:, 42:150]
    assert float(np.abs(mid - b).mean()) < 0.02                             # the square is the wide one's middle
    tall = FL.liquid(fx, (108, 192), 1.5, (53.5, 95.5), 0.0, FL._Memo())
    assert float(np.abs(tall[42:150] - b).mean()) < 0.02


def test_it_is_painted_in_its_inks_on_its_light():
    fx = dict(FX.normalize(liquid(inks=["love", "foam"], cells=0, bubbles=0), DAWN), index=0, f0=F0, f1=F0 + 150)
    pic = FL.liquid(fx, (W, H), 2.0, (W / 2 - 0.5, H / 2 - 0.5), 0.0, FL._Memo())
    light = np.asarray(PAL.srgb(DAWN["base"]), np.float32)
    assert np.all((pic >= 0.0) & (pic <= 1.0))
    bare = np.abs(pic - light).max(-1) < 0.03
    assert 0.02 < bare.mean() < 0.9                                         # clear water shows the lamp
    rg = pic[..., 0] - pic[..., 1]
    assert rg.max() > 0.12 and rg.min() < -0.05                             # love's rose and foam's teal both


@pytest.mark.parametrize("mask", ["figure", "ground"])
def test_a_mask_keeps_it_on_her_or_behind_her(mask):
    m = np.zeros((H, W), np.float32)
    m[20:70, 60:100] = 1.0
    img = np.random.default_rng(3).random((H, W, 3)).astype(np.float32)
    out = runner(plan(liquid(mask=mask, objects="Reisen*")), figure=m).frame(F0 + 20, img)
    kept = (m < 0.5) if mask == "figure" else (m > 0.5)
    assert np.array_equal(out[kept], img[kept])
    assert not np.allclose(out[~kept], img[~kept])


def test_blend_and_opacity_lay_it_on_the_picture():
    img = ground() * 0.6
    show = runner(plan(liquid())).frame(F0 + 20, img)
    half = runner(plan(liquid(opacity=0.5))).frame(F0 + 20, img)
    assert np.allclose(half, img * 0.5 + show * 0.5, atol=1e-5)
    mult = runner(plan(liquid(blend="multiply"))).frame(F0 + 20, img)
    assert np.allclose(mult, img * show, atol=1e-5)


def test_fast_enough_for_post():
    fx = dict(FX.normalize(liquid(), DAWN), index=0, f0=F0, f1=F0 + 150)
    memo = FL._Memo()
    FL.liquid(fx, (1920, 1080), 1.0, (959.5, 539.5), 0.0, memo)
    t0 = time.perf_counter()
    FL.liquid(fx, (1920, 1080), 1.0 + 1 / 30, (959.5, 539.5), 0.0, memo)
    assert time.perf_counter() - t0 < 0.6                                   # about 0.15 s; slack for a busy machine
