"""The physical cuts (`[[transition]] kind = "peel" | "shatter" | "tear"`; mkmmd.core.peel, .shatter, .tear, .sheet3d and
their drawers): their specs and windows, a hand's timing, the peel's fold line and the corner it publishes, the shards
covering the frame exactly once and all leaving it by the cut, the two halves of a tear partitioning the frame and
swinging out of it, and the composites on small synthetic frames. Fixtures use made-up shot names."""
import math

import numpy as np
import pytest

from mkmmd import peel as PEEL
from mkmmd import shatter as SHATTER
from mkmmd import tear as TEAR
from mkmmd.core import palette as PAL
from mkmmd.core import peel as PL
from mkmmd.core import shatter as SH
from mkmmd.core import sheet3d as S3
from mkmmd.core import tear as TE
from mkmmd.core import transition as TR

FPS, F0 = 60, 1
DAWN = PAL.get("rose-pine-dawn")
SIZES = [(108, 192), (108, 108), (192, 108)]


def plan(kind, **kw):
    data = {"output": [{"name": "o"}], "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}],
            "transition": [{"at": 2.0, "kind": kind, **kw}]}
    return TR.plan(data, FPS, F0, DAWN)


def tr_of(kind, **kw):
    return plan(kind, **kw)["transitions"][0]


# ---------------------------------------------------------------- specs and windows
@pytest.mark.parametrize("kind, dur", [("peel", 1.0), ("shatter", 0.5), ("tear", 0.7)])
def test_each_window_ends_at_the_cut_and_needs_only_the_incoming_plates(kind, dur):
    t = tr_of(kind)
    assert (t["dur"], t["n"], t["last"], t["plate"], t["matte"]) == (dur, round(dur * FPS), t["cut"] - 1, "b", None)
    need = TR.demands(plan(kind))
    assert sorted(need) == list(range(t["first"], t["cut"]))
    assert all(items == [{"kind": "plate", "shot": "b"}] for items in need.values())


def test_a_shatter_at_a_scene_point_reads_it_on_its_first_frame_only():
    pl = plan("shatter", center={"point": 'obj("x").loc'})
    t = pl["transitions"][0]
    need = TR.demands(pl)
    assert {"kind": "point", "key": "t0", "expr": 'obj("x").loc', "shot": "a"} in need[t["first"]]
    assert all(i["kind"] == "plate" for f, items in need.items() if f != t["first"] for i in items)
    assert {"kind": "point", "key": "t0", "expr": 'obj("x").loc', "shot": "a"} in TR.demands(pl, [t["last"]])[t["first"]]


@pytest.mark.parametrize("kind, kw, frag", [
    ("peel", {"from": "middle"}, "from = 'middle'"),
    ("peel", {"angle": 50}, "angle = 50.0"),
    ("peel", {"ease": "in"}, "unknown keys ['ease']"),
    ("peel", {"edge": {"width": 4, "colour": "red"}}, "unknown keys ['colour']"),
    ("shatter", {"material": "wood"}, "material = 'wood'"),
    ("shatter", {"count": 3}, "count = 3"),
    ("shatter", {"center": {"pt": "x"}}, "center"),
    ("tear", {"run": 0.7}, "run = 0.7"),
    ("tear", {"from": "left"}, "from = 'left'"),
    ("tear", {"jag": 0.5}, "jag = 0.5"),
])
def test_bad_cuts_say_what_is_wrong(kind, kw, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")):
        tr_of(kind, **kw)


def test_a_hands_pull_pinches_holds_and_ends_in_a_flick():
    us = np.linspace(0.0, 1.0, 401)
    p = np.array([S3.hand(u) for u in us])
    assert p[0] == 0.0 and p[-1] == pytest.approx(1.0)
    assert np.all(np.diff(p) >= -1e-12)                                    # never back
    assert S3.hand(0.1) < 0.07 and S3.hand(0.2) == pytest.approx(0.1)    # the pinch (a fifth): lifts, then holds
    assert S3.hand(0.03) > 0.01                                            # at once: a corner up on the first frames
    v = np.diff(p)
    assert v[-1] == v.max() and v[-1] > 2.0 * 0.9 / (0.8 * 400)           # fastest at the end: a flick
    assert np.max(np.abs(np.diff(v))) < 0.02 * v.max()                     # no jolt where one key hands on


# ---------------------------------------------------------------- the peel
@pytest.mark.parametrize("corner", list(PL.CORNERS))
def test_the_fold_starts_at_the_pinched_corner_and_the_sheet_has_left_by_the_cut(corner):
    for size in SIZES:
        t = tr_of("peel", **{"from": corner})
        n = PL.normal(t)
        cx, cy = PL.corner_px(t, size)
        lo, hi = PL.reach(t, size)
        assert cx * n[0] + cy * n[1] == pytest.approx(hi)                  # the normal points into that corner
        cs = [PL.line(t, k, size) for k in range(t["n"])]
        assert cs[0] < hi and all(a > b for a, b in zip(cs, cs[1:]))       # lifted on the first frame, on every frame on
        # on the cut every point of the sheet lies past the frame's far side: the trailing (last stuck) edge is out
        c1 = PL.travel(t, size)[1]
        x, _ = PL.lift(lo - c1, c1, PL.roll(t, size), math.radians(PL.PHI[2]))
        assert x < lo


def test_the_corner_is_bent_back_over_the_picture_from_the_first_frame_and_laid_back_by_the_pinchs_end():
    t = tr_of("peel")
    phis = [PL.phi(t, k) for k in range(t["n"])]
    assert math.pi / 2 < phis[0] < max(phis) and phis[int(0.2 * t["n"])] >= math.radians(149.0)
    assert phis[-1] == pytest.approx(math.radians(PL.PHI[2]), abs=0.02)


def test_the_published_corner_is_where_the_drawn_sheet_has_it():
    w, h = 108, 192
    t = tr_of("peel", **{"from": "top-right"})
    main = np.full((h, w, 3), 0.8, np.float32)
    seen_off = False
    for k in range(4, t["n"], 4):
        p = PL.corner(t, t["first"] + k, (w, h))
        x, y = p["p"][0] * w - 0.5, p["p"][1] * h - 0.5
        sh = PEEL.sheet(main, np.ones((h, w), np.float32), (1, 1, 1), PL.normal(t), PL.line(t, k, (w, h)),
                        PL.roll(t, (w, h)), PL.phi(t, k))
        # the drawn sheet is there: fully covered pixels right by it (a corner pixel itself is only partly covered)
        ix, iy = int(round(x)), int(round(y))
        if 3 <= ix < w - 3 and 3 <= iy < h - 3:
            assert sh.up_a[iy - 2:iy + 3, ix - 2:ix + 3].max() > 0.9, (k, p)
        elif not (0 <= x < w and 0 <= y < h):
            seen_off = True
    assert seen_off                                                       # it leaves the frame
    before = PL.corner(t, t["first"] - 1, (w, h))
    assert before["p"] == pytest.approx([1.0, 0.0]) and before["scale"] == 1.0


def test_a_peel_uncovers_more_of_the_incoming_picture_every_frame_and_its_back_is_paper():
    w, h = 108, 192
    t = tr_of("peel", shadow=0.0)
    main = np.zeros((h, w, 3), np.float32)
    main[..., 0] = 1.0                                                     # red
    inc = np.zeros((h, w, 3), np.float32)
    inc[..., 2] = 1.0                                                      # blue
    shares, paper = [], 0
    for k in range(t["n"]):
        out = PEEL.frame(t, t["first"] + k, main, inc)
        assert out.max() <= 1.0 + 1e-6
        shares.append(float(((out[..., 2] > 0.9) & (out[..., 0] < 0.1)).mean()))
        paper = max(paper, int(((out.min(axis=2) > 0.6)).sum()))           # the white back, never the picture's colour
    assert shares[0] < 0.02 and shares[-1] > 0.9
    assert all(b >= a - 0.002 for a, b in zip(shares, shares[1:]))
    assert paper > 0.1 * w * h


def test_the_paper_back_shows_at_the_pinched_corner_within_the_first_tenth_of_the_window():
    w, h = 216, 384
    t = tr_of("peel", shadow=0.0)
    main = np.zeros((h, w, 3), np.float32)
    main[..., 0] = 1.0                                                     # red
    inc = np.zeros((h, w, 3), np.float32)
    inc[..., 2] = 1.0                                                      # blue
    r = int(0.15 * w)                                                      # by the bottom-right corner
    for k in range(int(0.1 * t["n"])):
        out = PEEL.frame(t, t["first"] + k, main, inc)[h - r:, w - r:]
        if (out.min(axis=2) > 0.6).any():
            break
    else:
        pytest.fail("no paper at the pinched corner in the first tenth of the window")


@pytest.mark.parametrize("R, f, end", [(12.0, math.pi, 5.0), (12.0, math.radians(150), 40.0), (119.0, math.pi, 2300.0),
                                       (12.0, math.radians(170), 900.0)])
def test_the_roll_is_one_unbroken_sheet_that_winds_inside_itself_and_never_reaches_the_frame(R, f, end):
    us, xs, zs, _ = PL.profile(R, f, end, step=0.25)
    r0 = PL.spiral(R, f, end)[0]
    for i in range(1, len(us)):
        assert math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]) <= us[i] - us[i - 1] + 1e-6   # no jumps, no stretch
        assert zs[i] > 0.0                                                 # off the frame past the fold
        if us[i] > 2.0 * math.pi * r0:                                     # a turn on: inside the first
            assert math.hypot(xs[i], zs[i] - r0) < r0
    assert PL.lift(us[-1], 3.0, R, f, end) == pytest.approx((3.0 + xs[-1], zs[-1]))


# ---------------------------------------------------------------- the shatter
def _inside(poly, x, y):
    """Whether (x, y) is inside the convex polygon (either winding), with a hair's tolerance."""
    s = None
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        c = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(c) < 1e-9:
            continue
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


@pytest.mark.parametrize("material", list(SH.MATERIALS))
def test_the_shards_cover_the_frame_exactly_once(material):
    rng = np.random.default_rng(3)
    for size in SIZES:
        t = tr_of("shatter", material=material, count=60, seed=4)
        w, h = size
        for impact in ((w * 0.5, h * 0.45), (3.0, h - 2.0)):
            cells = SH.cells(t, size, impact)
            assert len(cells) >= 20
            assert sum(abs(SH.area(c["poly"])) for c in cells) == pytest.approx(w * h, rel=1e-9)
            for x, y in rng.uniform((-0.5, -0.5), (w - 0.5, h - 0.5), (400, 2)):
                assert sum(_inside(c["poly"], x, y) for c in cells) == 1
            # smaller near the impact than far from it
            near = min(cells, key=lambda c: math.dist(SH.centroid(c["poly"]), impact))
            assert abs(SH.area(near["poly"])) < np.median([abs(SH.area(c["poly"])) for c in cells])


def test_every_shard_starts_in_place_and_has_left_the_frame_by_the_cut():
    for size in SIZES:
        for material in SH.MATERIALS:
            t = tr_of("shatter", material=material)
            impact = (size[0] * 0.3, size[1] * 0.6)
            cells = SH.cells(t, size, impact)
            flights = SH.flights(t, size, impact, cells)
            for c, f in zip(cells, flights):
                H = np.float64(SH.homography(f, 0.0, size))
                assert H / H[2, 2] == pytest.approx(np.eye(3), abs=1e-9)
                assert SH._gone(c["poly"], f, size)
                assert SH.clock(t, f, t["n"] - 1, FPS) == 1.0
                assert SH.clock(t, f, SH.crack_frames(t, FPS) - 1, FPS) == 0.0


def test_a_shatter_cracks_then_breaks_onto_the_incoming_picture_and_is_the_same_every_time():
    w, h = 108, 192
    t = tr_of("shatter", shadow=0.0)
    yy, xx = np.mgrid[0:h, 0:w]
    main = np.stack([xx / w, yy / h, np.full((h, w), 0.3)], axis=2).astype(np.float32)
    inc = np.zeros((h, w, 3), np.float32)
    first = SHATTER.frame(t, t["first"], main, inc, (54.0, 86.0), FPS)
    assert np.median(np.abs(first - main)) < 0.06                          # whole, cracked
    nc = SH.crack_frames(t, FPS)
    shares = [float((SHATTER.frame(t, t["first"] + k, main, inc, (54.0, 86.0), FPS).max(axis=2) < 0.02).mean())
              for k in range(nc, t["n"])]
    assert shares[-1] > 0.85 and shares[-1] > shares[0]
    again = SHATTER.frame(t, t["first"] + 10, main, inc, (54.0, 86.0), FPS)
    assert np.array_equal(again, SHATTER.frame(t, t["first"] + 10, main, inc, (54.0, 86.0), FPS))
    assert again.max() <= 1.0


# ---------------------------------------------------------------- the tear
def test_the_two_halves_of_a_tear_partition_the_frame_and_their_paper_lies_under_the_other_print():
    for size in SIZES:
        t = tr_of("tear", angle=-20, center=[0.4, 0.5], jag=0.05)
        lay = TEAR._layout(t["angle"], t["from"], t["jag"], t["seed"], tuple(t["center"]), size)
        assert lay["print_a"] + lay["print_b"] == pytest.approx(np.ones(size[::-1]), abs=1e-6)
        both = lay["print_a"] * lay["print_b"] + 1e-6                     # only the antialiased edge pixel is shared
        assert np.all(lay["paper_a"] * lay["print_a"] <= both) and np.all(lay["paper_b"] * lay["print_b"] <= both)
        assert 0.2 < float(lay["print_a"].mean()) < 0.8
        o, d, e = TE.axes(t, size)
        line = TE.rip(t, size)
        ys, xs = np.mgrid[0:size[1]:7, 0:size[0]:7]
        for x, y in zip(xs.ravel(), ys.ravel()):
            want = TE.half_of(line, o, d, e, float(x), float(y))
            assert (lay["print_a"][y, x] > 0.5) == (want == "a") or abs(lay["print_a"][y, x] - 0.5) < 0.5


def test_a_tear_rips_from_its_start_then_the_halves_leave_the_frame_by_the_cut():
    w, h = 108, 192
    t = tr_of("tear", shadow=0.0)
    main = np.full((h, w, 3), 0.9, np.float32)
    main[..., 2] = 0.1                                                     # yellow
    inc = np.zeros((h, w, 3), np.float32)
    inc[..., 2] = 1.0                                                      # blue
    nr = TE.run_frames(t, FPS)
    mid = TEAR.frame(t, t["first"] + nr - 1, main, inc, FPS)              # the rip is through, the halves bent apart
    blue = (mid[..., 2] > 0.9) & (mid[..., 0] < 0.1)
    rows = blue.any(axis=1)
    assert rows[: h // 8].any() and not rows[-h // 8:].any()               # open at the top, whole at the bottom
    last = TEAR.frame(t, t["last"], main, inc, FPS)
    assert float(((last[..., 2] > 0.9) & (last[..., 0] < 0.1)).mean()) > 0.995
    for size in SIZES:
        t2 = dict(t)
        sw = TE.swing(t2, size)
        assert all(math.radians(TE.SWING[0]) <= sw[k] < math.radians(TE.SWING[1]) for k in "ab")   # found, not given up


def test_the_grips_ride_the_halves_apart():
    size = (108, 192)
    t = tr_of("tear")
    g = [TE.grips(t, t["first"] + k, FPS, size) for k in range(-1, t["n"])]
    assert g[0]["a"][0] < g[0]["b"][0]                                     # a on the left of a downward tear
    gaps = [p["b"][0] - p["a"][0] for p in g]
    inside = [gap for p, gap in zip(g, gaps) if 0.0 <= p["a"][0] and p["b"][0] <= 1.0]
    assert len(inside) > TE.run_frames(t, FPS) and all(b >= a - 1e-9 for a, b in zip(inside, inside[1:]))
    assert g[-1]["a"][0] < 0.0 and g[-1]["b"][0] > 1.0
