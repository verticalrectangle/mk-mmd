"""Festival lasers ([[fx]] kind "lasers", mkmmd.core.lasers and mkmmd.lasers): the keys and what they refuse, the fans'
geometry (spread, sweep, mirror), their swing and breathing without hits and on them, the hits paced to three a second,
beams from frame points and projected scene points, light only added (or screened), kept off her by a ground mask, the
same frame in any order, and the GPU leaving the kind to the CPU."""
import math
import re

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import lasers as LZF
from mkmmd import postgpu as PG
from mkmmd.core import framefx as FX
from mkmmd.core import lasers as LZ
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 160, 90
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lasers(**kw):
    return {"kind": "lasers", "from": 0.0, "to": 3.0, **kw}


def plan(*fx, size=(W, H)):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def runner(pl, figure=None, point=None, size=(W, H)):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None,
                      point_of=(lambda key, g: (point(g), 0.01)) if point is not None else None,
                      figure_of=(lambda key, g: figure) if figure is not None else None)


def dark():
    return np.full((H, W, 3), 0.05, np.float32)


@pytest.mark.parametrize("spec, frag", [
    (lasers(count=0), "count = 0: expected 1..64"),
    (lasers(spread=400), "spread = 400: expected 0..360"),
    (lasers(by="ring"), "by 'ring' is none of fan, beam"),
    (lasers(blend="over"), "blend 'over' is none of add, screen"),
    (lasers(mirror="yes"), "mirror is true or false"),
    (lasers(origin="left"), "origin = 'left': expected [x, y]"),
    (lasers(origin=[[0.5, 1.0]] * 17), "a list of 1 to 16 of them"),
    (lasers(origin={"point": ""}), "expected {point = \"<expression>\"} or [x, y]"),
    (lasers(mask="ground"), "needs the figure's `objects`"),
    (lasers(times=[1.0], lane="chug"), "one of them"),
    (lasers(colors=[]), "expected a colour, or a list of them"),
    (lasers(beams=3), "unknown keys ['beams']"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, MOON)


def test_origins_colours_and_points():
    one = FX.normalize(lasers(), MOON)
    assert one["origin"] == [{"mode": "frame", "at": [0.5, 1.0]}] and len(one["colors"]) == 2
    assert one["colors"][0] == [round(c, 6) for c in PAL.srgb(MOON["love"])]
    two = FX.normalize(lasers(origin=[{"point": "obj('car_headlamp_L').loc"}, [0.8, 1.1]], colors="gold"), MOON)
    assert [o["mode"] for o in two["origin"]] == ["point", "frame"] and len(two["colors"]) == 1
    pl = plan(lasers(origin=[{"point": "obj('a').loc"}, {"point": "obj('b').loc"}]))
    assert FX.points(pl["fx"][0]) == [("x0o0", "obj('a').loc"), ("x0o1", "obj('b').loc")]
    assert FX.points(plan(lasers())["fx"][0]) == []
    assert not FX.uses_figure(pl["fx"][0]) and FX.uses_figure(plan(lasers(mask="ground", objects="R*"))["fx"][0])


def test_hits_are_paced_to_three_a_second():
    assert LZ.paced([0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 0.8]) == [0.0, 0.4, 0.75]
    assert LZ.paced([0.9, 0.0, 1.0 / 3.0]) == [0.0, 1.0 / 3.0, 0.9]
    assert LZ.paced([]) == []


def test_a_fan_spreads_sweeps_and_mirrors():
    fx = FX.normalize(lasers(count=5, spread=40, aim=80, sweep=20, fan=0.5), MOON)
    beams = LZ.fan(fx, 0, (0, 1.0, 1.0, 1.0))
    assert [round(a, 6) for a, _ in beams] == [80.0, 90.0, 100.0, 110.0, 120.0]   # aim + sweep, 40 degrees wide
    closed = LZ.fan(fx, 0, (0, 0.0, 0.0, 1.0))
    assert [round(a, 6) for a, _ in closed] == [70.0, 75.0, 80.0, 85.0, 90.0]     # closed by half its spread
    mirrored = LZ.fan(fx, 1, (0, 1.0, 1.0, 1.0))
    assert sorted(round(180.0 - a, 6) for a, _ in mirrored) == [80.0, 90.0, 100.0, 110.0, 120.0]
    assert [c for _, c in beams] == [0] * 5 and [c for _, c in mirrored] == [1] * 5      # a colour per fan
    by_beam = FX.normalize(lasers(count=3, by="beam", mirror=False), MOON)
    assert [c for _, c in LZ.fan(by_beam, 1, (2, 0.0, 1.0, 1.0))] == [2, 3, 4]        # a colour per beam, stepped
    assert [round(a, 6) for a, _ in LZ.fan(by_beam, 1, (2, 0.0, 1.0, 1.0))] == [65.0, 90.0, 115.0]


def test_without_hits_the_fans_swing_steadily_and_on_hits_they_swing_side_to_side():
    fx = FX.normalize(lasers(rate=0.5), MOON)
    k, side, opened, light = LZ.state(fx, 0.5, [])                # a quarter of a swing: at its end, closed
    assert (k, light) == (0, 1.0) and side == pytest.approx(1.0) and opened == pytest.approx(0.0, abs=1e-9)
    assert LZ.state(fx, 1.0, [])[1] == pytest.approx(0.0, abs=1e-9) and LZ.state(fx, 1.0, [])[2] == pytest.approx(1.0)
    hit = FX.normalize(lasers(pulse=0.4, snap=0.2, decay=0.3, times=[1.0, 2.0]), MOON)
    before = LZ.state(hit, 0.5, [1.0, 2.0])
    assert before == (0, 1.0, 0.0, pytest.approx(0.6))           # resting on its first side, closed, dimmed
    on = LZ.state(hit, 1.0 + LZ.ATTACK, [1.0, 2.0])
    assert on[0] == 1 and on[2] == pytest.approx(1.0) and on[3] == pytest.approx(1.0)   # thrown open, full light
    mid = LZ.state(hit, 1.1, [1.0, 2.0])
    assert -1.0 < mid[1] < 1.0                                   # on its way over
    after = LZ.state(hit, 1.5, [1.0, 2.0])
    assert after[1] == -1.0 and after[2] < 0.3 and 0.6 <= after[3] < 0.75             # over, closing, dimming
    assert LZ.state(hit, 2.5, [1.0, 2.0])[1] == 1.0              # and back on the next


def test_a_beam_is_drawn_where_it_points_and_only_adds_light():
    pl = plan(lasers(origin=[0.5, 1.0], count=1, spread=0, sweep=0, aim=90, glow=0, haze=0, fade=0,
                     colors="text"))
    img = dark()
    out = runner(pl).frame(F0 + 10, img)
    col = out[:, 79:81].max(1) @ LUMA                            # the column straight up from the bottom middle
    assert (col[5:-5] > 0.5).all()
    assert np.allclose(out[:, :60], img[:, :60]) and np.allclose(out[:, 100:], img[:, 100:])
    assert (out >= img - 1e-6).all() and out.max() <= 1.0
    screen = runner(plan(lasers(blend="screen", colors="text"))).frame(F0 + 10, img)
    assert (screen >= img - 1e-6).all() and screen.max() <= 1.0 and screen.max() > 0.5


def test_a_fan_leaves_a_scene_point_and_rides_it():
    spec = lasers(origin={"point": "obj('lamp').loc"}, count=1, spread=0, sweep=0, aim=0, glow=0, haze=0, fade=0,
                  colors="text")
    r = runner(plan(spec), point=lambda g: (20.0 + (g - F0), 30.0))  # the lamp moves right a px a frame
    for g in (F0 + 5, F0 + 25):
        out = r.frame(g, dark())
        x0 = 20 + (g - F0)
        assert out[30, x0 + 3:] .max(1).min() > 0.5             # a beam to the right from the lamp, on its row
        assert np.allclose(out[30, :x0 - 3], 0.05) and np.allclose(out[60:], 0.05)


def test_a_ground_mask_keeps_her_and_the_frames_come_out_the_same_in_any_order():
    m = np.zeros((H, W), np.float32)
    m[20:70, 60:100] = 1.0
    pl = plan(lasers(mask="ground", objects="Reisen*", times=[0.5, 1.0]))
    img = dark()
    r = runner(pl, figure=m)
    seq = [r.frame(F0 + g, img) for g in (10, 20, 30, 40)]
    assert all(np.array_equal(o[20:70, 60:100], img[20:70, 60:100]) for o in seq)
    assert any(not np.allclose(o, img) for o in seq)
    fresh = runner(pl, figure=m)
    assert np.array_equal(fresh.frame(F0 + 40, img), seq[3]) and np.array_equal(fresh.frame(F0 + 10, img), seq[0])


def test_hits_faster_than_three_a_second_flash_three_times_a_second_at_most():
    times = [0.1 * j for j in range(30)]                         # ten hits a second
    pl = plan(lasers(times=times, pulse=1.0, decay=0.08, snap=0.0, fan=0.0, sweep=0.0))
    r = runner(pl)
    means = np.array([float((r.frame(F0 + g, dark()) @ LUMA).mean()) for g in range(90)])
    rises = [g for g in range(1, 89) if means[g] > means[g - 1] + 1e-4 and means[g] >= means[g + 1]]
    for s in range(3):                                           # in every second of the span
        assert len([g for g in rises if s * FPS <= g < (s + 1) * FPS]) <= 3


def test_the_gpu_leaves_lasers_to_the_cpu():
    pl = plan(lasers())
    r = runner(pl)
    assert PG.ops(r, r.fxs[0], F0 + 3) is None
    assert LZF.clip_segment((-10.0, 5.0), (20.0, 5.0), (0.0, 0.0, 9.0, 9.0)) == ((0.0, 5.0), (9.0, 5.0))
    assert LZF.clip_segment((-10.0, -5.0), (-1.0, -5.0), (0.0, 0.0, 9.0, 9.0)) is None
    mask = LZF.beam_mask((40, 20), (-1000.0, 10.0), [0.0], 5000.0, 1, 4.0)   # from far off the frame: clipped, drawn
    assert mask[10].min() > 200 and mask[0].max() == 0
    assert math.isclose(LZ.PACE, 1.0 / 3.0)
