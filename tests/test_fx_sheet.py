"""Sticker sheets (mkmmd.core.sheet: the entries checked, the doodles' places, the slap, the jumps, hops and peels, a
fling's flight; mkmmd.sheet: the live sticker and the hollow outline drawn over the paper) on a synthetic figure."""
import math

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import sheet as ST
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 240, 320
MAGENTA = (1.0, 0.0, 1.0)
FLAT = {"mottle": 0, "fibres": 0, "tooth": 0, "lamp": 0}


def plan(fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def sheet(**kw):
    return {"kind": "sheet", "from": 1.0, "to": 3.0, "objects": "Rin*", "drift": False, **kw}


def figure(g):
    """A figure 60 x 100 px, a step right on every frame from 1 s on: the frame of 0.5 s is the one a hollow keeps."""
    m = np.zeros((H, W), np.float32)
    x = 40 + (0 if g < F0 + FPS else 10)
    m[60:160, x:x + 60] = 1.0
    return m


def frames(fx):
    pl = plan(fx)
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: np.full((H, W, 3), MAGENTA, np.float32),
                      point_of=lambda key, g: ((W / 2, H / 2), None), figure_of=lambda key, g: figure(g)), pl


def test_doodles_keep_clear_of_boxes_and_each_other_and_are_seeded():
    radii = [0.04] * 12
    clear = [[0.3, 0.2, 0.7, 0.9], [0.0, 0.78, 1.0, 1.0]]
    area = [0.03, 0.09, 0.86, 0.97]
    a = ST.scatter(radii, 9 / 16, area, clear, 0.01, 3)
    assert a == ST.scatter(radii, 9 / 16, area, clear, 0.01, 3)
    assert a != ST.scatter(radii, 9 / 16, area, clear, 0.01, 4)
    got = [(x * 9 / 16, y) for x, y in (p for p in a if p is not None)]
    assert len(got) >= 6
    for i, (x, y) in enumerate(got):
        assert area[0] * 9 / 16 + 0.04 <= x <= area[2] * 9 / 16 - 0.04 and area[1] + 0.04 <= y <= area[3] - 0.04
        for x0, y0, x1, y1 in clear:
            dx = max(x0 * 9 / 16 - x, x - x1 * 9 / 16, 0.0)
            dy = max(y0 - y, y - y1, 0.0)
            assert math.hypot(dx, dy) >= 0.04 + 0.01 - 1e-9
        for X, Y in got[i + 1:]:
            assert math.hypot(X - x, Y - y) >= 0.08 + 0.01 - 1e-9


def test_a_doodle_with_no_room_is_left_out():
    assert ST.scatter([0.2], 1.0, [0.0, 0.0, 1.0, 1.0], [[0.0, 0.0, 1.0, 0.9]], 0.0, 1) == [None]


def test_doodles_keep_clear_of_the_figure_shape_not_its_box():
    """A figure shaped like an L (a body and an arm out to the right) leaves room under the arm that its box does not."""
    def away(p):                                   # distance (frame heights) from two rectangles, x in heights too
        d = np.full(len(p), np.inf)
        for x0, y0, x1, y1 in ((0.2, 0.1, 0.4, 0.9), (0.4, 0.1, 0.9, 0.25)):
            dx = np.maximum(np.maximum(x0 - p[:, 0], p[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - p[:, 1], p[:, 1] - y1), 0.0)
            d = np.minimum(d, np.hypot(dx, dy))
        return d
    got = [p for p in ST.scatter([0.05] * 8, 1.0, [0.4, 0.25, 1.0, 1.0], [], 0.01, 2, away) if p is not None]
    assert len(got) >= 4
    assert all(away(np.array([p]))[0] >= 0.06 - 1e-9 for p in got)


def test_slap_lands_big_squashes_and_settles_in_its_time():
    s = [{"t": 1.0, "size": 1.3, "drop": 0.08, "squash": 0.2, "settle": 0.25, "dust": 0}]
    assert ST.slap_at(s, 0.5) == {"scale": 1.3, "sx": 1.0, "sy": 1.0, "lift": 1.0}
    mid = ST.slap_at(s, 0.96)
    assert 1.0 < mid["scale"] < 1.3 and 0 < mid["lift"] < 1
    deep = min(ST.slap_at(s, 1.0 + k / 600)["sy"] for k in range(1, 60))
    assert deep == pytest.approx(0.8, abs=0.005)
    us = [k / 600 for k in range(1, 150)]
    first = min(us, key=lambda u: ST.slap_at(s, 1.0 + u)["sy"])
    assert 0.02 < first < 0.05                                            # the squash peaks within three frames
    top = max(ST.slap_at(s, 1.0 + u)["sy"] for u in us)
    assert 1.0 < top < 1.03                                               # bounces back past rest, never thin
    for u in (0.25, 0.3, 1.0):
        assert ST.slap_at(s, 1.0 + u) == {"scale": 1.0, "sx": 1.0, "sy": 1.0, "lift": 0.0}
    q = [ST.slap_at(s, 1.0 + k / 600)["sy"] for k in range(0, 160)]
    assert max(abs(a - b) for a, b in zip(q, q[1:])) < 0.03                # no jump from one sample to the next


def test_jump_hop_and_peel_of_a_doodle():
    d = {"angle": 10.0, "spin": 0.5, "twist": 1.0, "corner": "top-left"}
    jump = {"t": 1.0, "dur": 0.5, "stagger": 0.1, "height": 0.8, "spin": 120.0}
    hop = {"height": 0.25, "dur": 0.24, "ripple": 0.1}
    at, origin = (0.8, 0.3), (0.5, 0.5)

    def at_t(t, peel=None):
        return ST.doodle_at(t, d, at, 1.0, jump, [2.0], hop, peel, 0.45, origin)
    assert not at_t(1.05)["shown"]                                         # the far one leaves 0.1 s late
    s = at_t(1.1 + 0.2)
    assert s["shown"] and s["lift"] > 0.5 and s["scale"] > 1.2
    assert origin[0] < s["at"][0] < at[0]
    land = at_t(1.1 + 0.5 + 0.02)
    assert land["at"] == at and land["lift"] == 0.0 and land["sy"] < 1.0
    rest = at_t(1.8)
    assert (rest["at"], rest["angle"], rest["lift"], rest["sy"]) == (at, 10.0, 0.0, 1.0)
    up = at_t(2.1 - 0.12)                                                  # off the paper half a hop before its beat
    assert up["lift"] == pytest.approx(0.25) and up["angle"] != 10.0
    assert at_t(2.1 - 0.01)["lift"] > 0.0 and at_t(2.1 + 0.001)["lift"] == 0.0   # and lands on it (+ its ripple)
    assert at_t(2.1 + 0.03)["sy"] < 1.0 and at_t(2.1 + 0.03)["angle"] != 10.0     # squash and wobble on landing
    assert at_t(2.6)["lift"] == 0.0
    assert at_t(3.0, peel=3.0)["peel"] == 0.0 and at_t(3.2, peel=3.0)["peel"] == pytest.approx(0.2 / 0.45)
    assert not at_t(3.5, peel=3.0)["shown"]


def test_hops_on_the_timelines_beats_and_downbeats():
    beats = [0.08, 0.58, 1.08, 1.59, 2.08, 2.58, 3.08, 3.59, 4.09, 4.59]
    assert ST.beat_times(beats, [0.58, 2.58, 4.59], 1) == beats
    assert ST.beat_times(beats, [0.58, 2.58, 4.59], 4) == [0.58, 2.58, 4.59]


def test_the_peeled_grip_rises_and_leaves():
    peel = {"t": 1.0, "dur": 0.8, "corner": "top-right", "radius": 0.05}
    grip, size = (700.0, 400.0), (1080, 1920)
    c0 = ST.peel_grip(peel, 0.5, grip, size)
    assert c0["z"] == 0.0 and c0["p"] == pytest.approx([700 / 1080, 400 / 1920], abs=1e-3)
    zs = [ST.peel_grip(peel, 1.0 + 0.8 * k / 10, grip, size)["z"] for k in range(1, 11)]
    assert zs[0] > 0.0 and max(zs) > 0.05                                   # it lifts from the peel's first moment
    end = ST.peel_grip(peel, 1.8, grip, size)["p"]
    assert not (0.0 <= end[0] <= 1.0 and 0.0 <= end[1] <= 1.0)              # it has left the frame


def test_peel_order_and_ripple():
    dd = {"seed": 1, "peel": {"times": [2.0, 3.0], "order": "near"}}
    places = [(0.9, 0.9), None, (0.55, 0.5), (0.2, 0.2)]
    assert ST.peel_order(dd, places, (0.5, 0.5), 1) == {2: 2.0, 3: 3.0}
    w = ST.ripple(places, (0.5, 0.5))
    assert w[2] == 0.0 and w[0] == 1.0 and 0 < w[3] < 1


def test_fling_grows_toward_the_lens_and_leaves_the_frame():
    fx = FX.normalize({"kind": "fling", "from": 0.0, "to": 3.0, "center": [0.5, 0.6], "times": [1.0, 2.0]}, DAWN)
    items = ST.fling_items(fx, [1.0, 2.0])
    assert len(items) == 2 * fx["count"] and items == ST.fling_items(fx, [1.0, 2.0])
    for it in items:
        assert ST.fling_at(fx, it, it["t0"] - 0.01) is None and ST.fling_at(fx, it, it["t0"] + fx["life"] + 0.01) is None
        sizes = [ST.fling_at(fx, it, it["t0"] + fx["life"] * k / 20)["scale"] for k in range(1, 20)]
        assert all(b > a for a, b in zip(sizes, sizes[1:]))
        end = ST.fling_at(fx, it, it["t0"] + fx["life"] * (1 - 1e-9))
        assert end["scale"] == pytest.approx(fx["size"] * it["size"] * fx["grow"])
        dx, dy = end["off"]
        assert math.hypot(dx, dy) > 0.5 + end["scale"] / 2                  # out past any frame's edge by the end


def test_fling_thrown_away_comes_in_large_and_falls_into_the_point():
    base = {"kind": "fling", "from": 0.0, "to": 3.0, "center": [0.5, 0.6], "times": [1.0], "grow": 10.0, "life": 1.2}
    near, fx = FX.normalize(base, DAWN), FX.normalize({**base, "away": True}, DAWN)
    assert near["away"] is False
    items = ST.fling_items(fx, [1.0])
    assert items == ST.fling_items(near, [1.0])                   # the same stickers, their flight played backwards
    for it in items:
        start = ST.fling_at(fx, it, it["t0"])
        assert start["scale"] == pytest.approx(fx["size"] * it["size"] * fx["grow"]) and start["depth"] < 0.11
        assert math.hypot(*start["off"]) > 0.5 + start["scale"] / 2          # in from past any frame's edge
        ages = [fx["life"] * k / 20 for k in range(1, 20)]
        for a in ages:
            s, m = ST.fling_at(fx, it, it["t0"] + a), ST.fling_at(near, it, it["t0"] + fx["life"] - a)
            assert s["off"] == pytest.approx(m["off"]) and s["depth"] == pytest.approx(m["depth"])
        sizes = [ST.fling_at(fx, it, it["t0"] + a)["scale"] for a in ages]
        assert all(b < a for a, b in zip(sizes, sizes[1:]))                  # shrinking all the way in
        end = ST.fling_at(fx, it, it["t0"] + fx["life"] * (1 - 1e-9))
        assert end["scale"] < 1e-6 and math.hypot(*end["off"]) < 1e-6          # gone at the point
        assert ST.fling_at(fx, it, it["t0"] + fx["life"] + 0.01) is None
    with pytest.raises(FX.FxError, match="away"):
        FX.normalize({**base, "away": "yes"}, DAWN)


def test_sheet_reads_its_first_frame_and_the_frames_it_freezes():
    fx = plan([sheet(hollows=[{"t": 0.5, "at": [0.75, 0.6]}], peel={"t": 2.5})])["fx"][0]
    f = FX.frame_of(2.0, FPS, F0)
    assert sorted(FX.sources(fx, f, F0)) == sorted([fx["f0"], FX.frame_of(0.5, FPS, F0), FX.frame_of(2.5, FPS, F0)])
    assert FX.uses_figure(fx)
    lay = TR.demands(plan([sheet(hollows=[{"t": 0.5}])]), frames=[FX.frame_of(0.5, FPS, F0)])
    assert any(it["kind"] == "figure" for its in lay.values() for it in its)
    with pytest.raises(TR.TransitionError, match="give its `objects`"):
        plan([{"kind": "sheet", "from": 1.0, "to": 3.0, "hollows": [{"t": 0.5}]}])


def test_live_sticker_and_hollow_outline_on_the_paper():
    ffx, pl = frames([sheet(paper=FLAT, shadow=False, hollows=[{"t": 0.5, "at": [0.75, 0.6], "sheen": 0.0}])])
    out = ffx.frame(FX.frame_of(2.0, FPS, F0), np.full((H, W, 3), MAGENTA, np.float32))
    paper = np.asarray(FX._colour("base", DAWN, "paper"), np.float32)
    assert np.allclose(out[110, 80], MAGENTA, atol=1e-3)                    # the live sticker where its frame shows it
    assert np.allclose(out[110, 50 - 2], 1.0, atol=0.02)                    # its white rim just outside it
    hx, hy = int(0.75 * W), int(0.6 * H)                                    # the hollow: its pivot (the feet) there
    assert np.allclose(out[hy - 40, hx], paper, atol=0.02)                  # nothing inside but the paper
    assert np.allclose(out[hy - 40, hx - 30 - 2], 1.0, atol=0.02)           # its rim, white, round the figure's side
    assert np.abs(out[:, 120:] - np.asarray(MAGENTA, np.float32)).sum(2).min() > 0.5   # no figure in it


def test_a_hollow_s_blank_sticker_lies_in_it_fades_and_is_peeled_off():
    blank = {"t": 2.0, "dur": 0.6, "color": "iris", "fade": [1.0, 2.0]}
    ffx, pl = frames([sheet(paper=FLAT, shadow=False, hollows=[{"t": 0.5, "at": [0.75, 0.6], "sheen": 0.0,
                                                               "peel": blank}])])
    img = np.full((H, W, 3), MAGENTA, np.float32)
    paper, iris, light = (np.asarray(FX._colour(c, DAWN, "c"), np.float32) for c in ("base", "iris", "surface"))
    hx, hy = int(0.75 * W), int(0.6 * H)
    at = lambda t: ffx.frame(FX.frame_of(t, FPS, F0), img)[hy - 40, hx]                  # noqa: E731
    assert np.allclose(at(1.0), iris, atol=0.02)                            # the sticker still there, its flat colour
    assert np.allclose(at(1.5), iris + (light - iris) * ST.BLANK_FADE * 0.5, atol=0.02)   # half way through its fade
    assert np.allclose(at(2.8), paper, atol=0.02)                           # peeled off: nothing but the paper
    with pytest.raises(TR.TransitionError, match="needs `t`"):
        plan([sheet(hollows=[{"t": 0.5, "peel": {"color": "iris"}}])])


def test_a_peel_starts_at_the_sticker_s_own_corner_and_a_hand_rides_it():
    def notched(key, g):                       # the figure's top right quarter cut away: that corner of its box is empty,
        m = figure(F0)                         # and the sticker's own corner (69, 60) lies 30 px in from it
        m[60:110, 70:100] = 0.0
        return m
    peel = {"t": 1.2, "dur": 1.6, "corner": "top-right"}
    hand = {"kind": "cutout", "from": 1.0, "to": 3.0, "clip": "hand.mp4", "at": {"sticker": 1.2}}
    of = dict(fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=lambda key, g: ((W / 2, H / 2), None),
              figure_of=notched)
    img = np.full((H, W, 3), MAGENTA, np.float32)
    alone = FF.FrameFx(plan([sheet(paper=FLAT, shadow=False, edge=False, peel=peel)])["fx"], (W, H), **of)
    assert np.allclose(alone.frame(FX.frame_of(1.1, FPS, F0), img)[64, 65], MAGENTA, atol=0.02)
    lifted = FX.frame_of(1.5, FPS, F0)         # the fold has gone 23 px: from the box's corner it would only be arriving
    assert not np.allclose(alone.frame(lifted, img)[64, 65], MAGENTA, atol=0.2)
    both = FF.FrameFx(plan([sheet(paper=FLAT, shadow=False, edge=False, peel=peel), hand])["fx"], (W, H), **of)
    assert both._cutout_at(both.fxs[1], FX.frame_of(1.1, FPS, F0)) == pytest.approx((69.0, 60.0))
    x, y = both._cutout_at(both.fxs[1], lifted)
    assert x < 66.0 and y > 63.0               # carried back over the sticker, away from the pinched corner
    with pytest.raises(TR.TransitionError, match="rides the sticker peeled from 2 s; the sheets' peels start at"):
        plan([sheet(peel=peel), {**hand, "at": {"sticker": 2.0}}])


def test_no_doodle_is_laid_in_a_hollow_even_one_the_frame_cuts():
    hollow = {"t": 0.5, "at": [0.95, 0.9], "height": 0.6}             # the figure's shape, large, half past the right edge:
    ffx, pl = frames([sheet(hollows=[hollow], doodles={"count": 60, "size": 0.04, "gap": 0.0})])   # its rim is no closed ring
    places = [p for p in ffx._sheet_state(ffx.fxs[0])["doodles"]["places"] if p is not None]
    assert len(places) >= 20
    x0, y0, y1 = 0.95 * W - 57.6, 0.9 * H - 192.0, 0.9 * H            # its box: 192 px tall, from x0 to the frame's edge
    assert not [p for p in places if p[0] * W > x0 and y0 < p[1] * H < y1]
