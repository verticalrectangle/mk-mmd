"""library:rail_track without Blender: the plan (stations, ties, rail spans, the curve), the laying front's schedule
(nothing past `laid`, clean rests at every tie spacing, rails sliding only over landed ties, spikes only on landed rails),
the driver expressions against their numpy twins, the meshes and the card (mkmmd/core/rail_track.py,
mkmmd/blender/library/props/rail_track_layout.py)."""
import itertools
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import rail_track_layout as LAY
from mkmmd.core import rail_track as RT

BLENDER = {"clamp": lambda x, lo, hi: max(lo, min(hi, x)), "min": min, "max": max, "pow": pow, "abs": abs,
           "sin": math.sin, "cos": math.cos}


def ev(expr, laid):
    """A driver expression at `laid`, as Blender's simple-expression evaluator reads it (Python syntax, its functions)."""
    assert len(expr) <= RT.MAX_EXPR and "**" not in expr
    return eval(compile(expr, "<driver>", "eval"), {"__builtins__": {}}, dict(BLENDER, l=float(laid)))


PLANS = {"straight": RT.Plan(20.0, 0.0), "left": RT.Plan(20.0, 60.0), "right": RT.Plan(33.0, -250.0),
         "short": RT.Plan(2.0, 0.0)}


# ======================================================================================================== the plan
def test_the_plan_rounds_the_length_to_tie_spacings_and_cuts_rails_in_30_ft_lengths():
    p = PLANS["straight"]
    assert p.n == 33 and p.length == pytest.approx(33 * RT.SPACING) and RT.RAIL == pytest.approx(9.144)
    assert p.ties[0] == pytest.approx(RT.SPACING / 2) and np.allclose(np.diff(p.ties), RT.SPACING)
    assert p.spans == [pytest.approx((0.0, 9.144)), pytest.approx((9.144, 18.288)), pytest.approx((18.288, p.length))]
    assert p.done == pytest.approx(p.length + 2 * RT.SPACING)
    assert [p.span_of(i) for i in (0, 14, 15, 29, 30, 32)] == [0, 0, 1, 1, 2, 2]
    short = PLANS["short"]
    assert short.n == 3 and short.spans == [pytest.approx((0.0, 3 * RT.SPACING))]


@pytest.mark.parametrize("bad", [{"length": 0.1}, {"length": "far"}, {"radius": 20.0}, {"radius": -49.0},
                                 {"length": 400.0, "radius": 60.0}, {"radius": float("nan")}])
def test_a_track_that_cannot_be_laid_is_refused(bad):
    with pytest.raises(RT.TrackError):
        RT.Plan(**{"length": 20.0, "radius": 0.0, **bad})


@pytest.mark.parametrize("name", ["straight", "left", "right"])
def test_the_centre_line_is_unit_speed_with_the_curvature_asked_for(name):
    p = PLANS[name]
    s = np.linspace(0.0, p.length, 2001)
    x, y, h = p.pose(s)
    assert (x[0], y[0], h[0]) == (0.0, 0.0, 0.0)
    ds = s[1] - s[0]
    speed = np.hypot(np.diff(x), np.diff(y)) / ds
    assert np.allclose(speed, 1.0, atol=1e-5)
    assert np.allclose(np.diff(h) / ds, p.k)
    side = p.point(s[:5], y=1.0) - p.point(s[:5])                 # the offset is the unit normal, to the left
    assert np.allclose(np.linalg.norm(side, axis=1), 1.0)
    assert np.allclose(side[:, 0] * np.cos(h[:5]) + side[:, 1] * np.sin(h[:5]), 0.0)
    assert np.all(np.cos(h[:5]) * side[:, 1] - np.sin(h[:5]) * side[:, 0] > 0)
    if p.k:
        c = np.array(p.centre)
        assert np.allclose(np.hypot(x - c[0], y - c[1]), abs(p.radius))


# ======================================================================================================== the schedule
def states(p, laid):
    """Everything the laying front decides at `laid`, from the numpy twins."""
    bed = [RT.bed_scale(p, k, laid) for k in range(p.n)]
    ties = [RT.tie_state(p, i, laid) for i in range(p.n)]
    rails = [RT.rail_state(p, j, laid) for j in range(len(p.spans))]
    spikes = [RT.spike_scale(p, i, laid) for i in range(p.n)]
    return bed, ties, rails, spikes


@pytest.mark.parametrize("name", list(PLANS))
def test_nothing_exists_past_the_front(name):
    p = PLANS[name]
    for laid in np.linspace(-0.5, p.done + 0.5, 1501):
        bed, ties, rails, _spikes = states(p, laid)
        for k, (sx, syz) in enumerate(bed):
            if syz > RT.MIN_SCALE:
                assert k * RT.SPACING + sx * RT.SPACING <= laid + 1e-9          # the toe of the bed's front ramp
        for i, (_, _, shown) in enumerate(ties):
            if shown:
                assert p.ties[i] - RT.NEAR <= laid + 1e-9                         # a tie from its near edge on
        for j, (start, _, shown) in enumerate(rails):
            if shown:
                r, e = p.spans[j]
                assert start + (e - r) <= laid - RT.LAG + 1e-9                    # a rail's nose LAG behind
                assert start >= -1e-9                                             # never behind the start


@pytest.mark.parametrize("name", list(PLANS))
def test_rails_slide_only_over_landed_ties_and_after_the_last_rail_landed(name):
    p = PLANS[name]
    for laid in np.linspace(0.0, p.done + 0.3, 2001):
        bed, ties, rails, spikes = states(p, laid)
        for j, (start, lift, shown) in enumerate(rails):
            if not shown:
                continue
            r, e = p.spans[j]
            if j > 0:
                assert rails[j - 1][1] == pytest.approx(0.0, abs=1e-12)          # the last rail is down
                assert rails[j - 1][0] == pytest.approx(p.spans[j - 1][0])
            under = (p.ties + RT.TIE_W / 2 > start) & (p.ties - RT.TIE_W / 2 < start + (e - r))
            for i in np.nonzero(under)[0]:
                h, tip, there = ties[i]
                assert there and h == pytest.approx(0.0, abs=1e-12) and tip == pytest.approx(0.0, abs=1e-12)
            toe = max(k * RT.SPACING + sx * RT.SPACING for k, (sx, syz) in enumerate(bed) if syz > RT.MIN_SCALE)
            assert toe >= start + (e - r)                                         # and over the bed
        for i, sz in enumerate(spikes):
            if sz > RT.MIN_SCALE:
                start, lift, shown = rails[p.span_of(i)]
                assert shown and lift == pytest.approx(0.0, abs=1e-12) and start == pytest.approx(p.spans[p.span_of(i)][0])


def near(v, *targets, tol=1e-12):
    """v is within `tol` of one of the targets (float noise at a window's end is not a half-way state)."""
    return any(abs(float(v) - t) <= tol for t in targets)


@pytest.mark.parametrize("name", list(PLANS))
def test_every_tie_spacing_is_a_clean_rest(name):
    """`laid` keyed in steps of a tie spacing (or of a rail) never leaves a tie in the air, a rail half landed, a spike
    half popped or a bed piece half grown."""
    p = PLANS[name]
    for k in range(p.n + 3):
        laid = k * RT.SPACING
        bed, ties, rails, spikes = states(p, laid)
        assert all(near(sx, RT.MIN_SCALE, 1.0) for sx, _ in bed), laid
        assert all(not shown or (near(h, 0.0) and near(tip, 0.0)) for h, tip, shown in ties), laid
        assert all(not shown or near(lift, 0.0, RT.LIFT) for _, lift, shown in rails), laid
        assert all(near(sz, RT.MIN_SCALE, 1.0) for sz in spikes), laid
    shown = [sum(bool(t[2]) for t in states(p, k * RT.SPACING)[1]) for k in range(p.n + 1)]
    assert shown == list(range(p.n + 1))                                          # one more tie per spacing


@pytest.mark.parametrize("name", list(PLANS))
def test_the_track_is_whole_from_done_and_gone_at_zero(name):
    p = PLANS[name]
    bed, ties, rails, spikes = states(p, p.done)
    assert all(sx == 1.0 and syz == 1.0 for sx, syz in bed)
    assert all(shown and near(h, 0.0) and near(tip, 0.0) for h, tip, shown in ties)
    assert all(shown and near(lift, 0.0) and start == pytest.approx(r) for (start, lift, shown), (r, _) in
               zip(rails, p.spans))
    assert all(near(sz, 1.0) for sz in spikes)
    bed, ties, rails, spikes = states(p, 0.0)
    assert all(syz == RT.MIN_SCALE for _, syz in bed) and not any(t[2] for t in ties) and not any(r[2] for r in rails)
    assert all(sz == RT.MIN_SCALE for sz in spikes)


def test_a_tie_drops_and_bounces_and_a_rail_slides_with_the_front_then_lands():
    p = PLANS["straight"]
    a = p.ties[3] - RT.NEAR
    h = [float(RT.tie_state(p, 3, a + q * RT.DROP_W)[0]) for q in np.linspace(0.0, 1.0, 101)]
    assert h[0] == pytest.approx(RT.DROP_H) and h[60] == pytest.approx(0.0, abs=1e-9) and near(h[-1], 0.0)
    assert max(h[61:]) == pytest.approx(RT.BOUNCE * RT.DROP_H, rel=1e-3)                    # the small bounce
    assert all(b <= a_ + 1e-12 for a_, b in itertools.pairwise(h[:61]))                     # falling all the way down
    r, e = p.spans[1]
    for laid in np.linspace(p.shown_from(1) + 2 * RT.EDGE, e + RT.LAG, 7):
        start, lift, shown = RT.rail_state(p, 1, laid)
        assert shown and lift == pytest.approx(RT.LIFT) and start + (e - r) == pytest.approx(laid - RT.LAG)
    lifts = [float(RT.rail_state(p, 1, x)[1]) for x in np.linspace(e + RT.LAG, e + RT.SPACING, 41)]
    assert lifts[0] == pytest.approx(RT.LIFT) and near(lifts[-1], 0.0) and max(lifts[31:]) > 0.5 * RT.LAND_BOUNCE
    assert RT.rail_state(p, 0, p.shown_from(0))[0] == 0.0                                    # the first drops in place


# ======================================================================================================== drivers
@pytest.mark.parametrize("name", list(PLANS))
def test_the_drivers_compute_the_numpy_schedule(name):
    p = PLANS[name]
    z0, seat = 0.358, 0.372
    laids = np.concatenate([np.linspace(-0.3, p.done + 0.3, 157), [k * RT.SPACING for k in range(p.n + 3)]])
    for k in sorted({0, 1, p.n // 2, p.n - 1}):
        ex = RT.bed_expr(p, k)
        for laid in laids:
            sx, syz = RT.bed_scale(p, k, laid)
            assert ev(ex["sx"], laid) == pytest.approx(float(sx))
            if abs(laid - k * RT.SPACING - RT.EDGE) > 1e-9:            # (a switch exactly on its edge is either)
                assert ev(ex["syz"], laid) == pytest.approx(float(syz))
    for i in sorted({0, 1, 14, 15, p.n // 2, p.n - 1} & set(range(p.n))):
        ex, tilt = RT.tie_expr(p, i, z0, -4.5), -4.5
        sz = RT.spike_expr(p, i)["sz"]
        for laid in laids:
            h, tip, shown = RT.tie_state(p, i, laid, tilt)
            assert ev(ex["z"], laid) == pytest.approx(z0 + float(h), abs=1e-9)
            assert ev(ex["rx"], laid) == pytest.approx(float(tip), abs=1e-9)
            if abs(laid - (p.ties[i] - RT.NEAR)) > 1e-9:
                assert ev(ex["s"], laid) == (1.0 if shown else RT.MIN_SCALE)
            assert ev(sz, laid) == pytest.approx(float(RT.spike_scale(p, i, laid)), abs=1e-9)
    for j in range(len(p.spans)):
        ex = RT.rail_expr(p, j, seat)
        for laid in laids:
            start, lift, shown = RT.rail_state(p, j, laid)
            want = float(start) if p.k == 0.0 else p.k * float(start)
            assert ev(ex["slide"], laid) == pytest.approx(want, abs=1e-9)
            assert ev(ex["z"], laid) == pytest.approx(seat + float(lift), abs=1e-9)
            if abs(laid - p.shown_from(j) - RT.EDGE) > 1e-9:
                assert ev(ex["s"], laid) == (1.0 if shown else RT.MIN_SCALE)
    fx = RT.front_expr(p)
    for laid in laids:
        x, y, h = (float(v) for v in p.pose(min(max(laid, 0.0), p.length)))
        assert ev(fx["x"], laid) == pytest.approx(x, abs=1e-6)
        if p.k:
            assert ev(fx["y"], laid) == pytest.approx(y, abs=1e-6) and ev(fx["rz"], laid) == pytest.approx(h, abs=1e-8)
        else:
            assert set(fx) == {"x"}


def test_a_curved_rail_span_slides_along_the_arc_by_turning_about_the_centre():
    """Turning a span's mesh (built at span 0's place, about the centre) by its `slide` angle puts it on its own span."""
    p = PLANS["left"]
    m = LAY.rail_mesh(p, p.spans[1][1] - p.spans[1][0], joint=False)
    a = p.k * p.spans[1][0]
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    V = m.V[:, :2] @ R.T + np.array(p.centre)
    c = np.array(p.centre)
    ang = np.arctan2(V[:, 0] - c[0], c[1] - V[:, 1])                         # station angle about the centre
    assert ang.min() == pytest.approx(p.k * p.spans[1][0], abs=1e-6)
    assert ang.max() == pytest.approx(p.k * p.spans[1][1], abs=1e-6)


# ======================================================================================================== meshes
@pytest.mark.parametrize("plates", [False, True])
def test_the_ties_are_hewn_closed_and_carry_the_rails_at_their_top(plates):
    tops = []
    for v in range(LAY.VARIANTS):
        m = LAY.tie_mesh(v, seed=3, plates=plates)
        assert m.is_closed() and m.volume() > 0
        _lo, hi = m.bbox()
        tie = m.Qm == LAY.MATS.index("tie")
        Vt = m.V[np.unique(m.Q[tie])]
        tops.append(Vt[:, 2].max())
        assert 2.55 < Vt[:, 1].max() - Vt[:, 1].min() < 2.63                       # 8.5 ft, give or take
        assert 0.17 < -Vt[:, 2].min() < 0.19 and 0.21 < Vt[:, 0].max() - Vt[:, 0].min() < 0.28
        if plates:
            assert hi[2] == pytest.approx(LAY.PLATE_T + 0.007 - 0.0005, abs=1e-6)        # the shoulders' tops
    assert np.allclose(tops, 0.0, atol=1e-9)                                           # the rails' seat
    assert len({round(float(LAY.tie_mesh(v).volume()), 6) for v in range(LAY.VARIANTS)}) == LAY.VARIANTS
    assert LAY.tie_mesh(2).V.tolist() == LAY.tie_mesh(2).V.tolist()                     # deterministic


def test_the_rail_section_is_a_t_rail_and_the_rails_lie_at_standard_gauge():
    sec = LAY.rail_section()
    from mkmmd.core import shell as S
    assert S.polygon_area(sec) > 0
    assert sec[:, 0].max() - sec[:, 0].min() == pytest.approx(RT.FOOT_W, abs=1e-6)
    assert sec[:, 1].min() == pytest.approx(0.0) and sec[:, 1].max() == pytest.approx(RT.RAIL_H, abs=1e-6)
    for name in ("straight", "left"):
        p = PLANS[name]
        m = LAY.rail_mesh(p, RT.RAIL)
        assert m.is_closed() and m.volume() > 0
        V = m.V + ([p.centre[0], p.centre[1], 0.0] if p.k else 0.0)
        s = 0.0                                                              # a ring: the heads' inner faces 1435 apart
        x, y, h = (float(v) for v in p.pose(s))
        n = np.array([-math.sin(h), math.cos(h)])
        t = np.array([math.cos(h), math.sin(h)])
        d = V[:, :2] - [x, y]
        near = (np.abs(d @ t) < 0.01) & (V[:, 2] > RT.RAIL_H - 0.03) & (V[:, 2] < RT.RAIL_H - 0.008)
        lat = d[near] @ n
        assert lat[lat > 0].min() + (-lat[lat < 0]).min() == pytest.approx(RT.GAUGE, abs=0.004)
    heads = m.Qm == LAY.MATS.index("railhead")
    assert heads.any() and (m.V[m.Q[heads]][..., 2] > RT.RAIL_H - 0.005).all()


def test_the_joints_and_spikes_sit_on_the_rails():
    p = PLANS["straight"]
    j = LAY.rail_mesh(p, RT.RAIL, joint=True)
    fit = j.Qm == LAY.MATS.index("plate")
    extra = j.V[np.unique(np.concatenate([j.Q[fit].ravel(), j.T[j.Tm == LAY.MATS.index("plate")].ravel()]))]
    assert extra[:, 0].min() == pytest.approx(-LAY.BAR_L / 2, abs=0.02) and extra[:, 0].max() < LAY.BAR_L / 2 + 0.02
    assert extra[:, 2].min() > LAY.FOOT_EDGE and extra[:, 2].max() < RT.RAIL_H - LAY.HEAD_D + 0.01
    assert np.abs(extra[:, 1]).min() > LAY.RAIL_Y - 0.06 and np.abs(extra[:, 1]).max() < LAY.RAIL_Y + 0.06
    sp = LAY.spikes_mesh()
    assert sp.is_closed()
    heads = sp.V[:, 2] > LAY.SPIKE_DROP + LAY.FOOT_EDGE + 0.001
    lat = np.abs(sp.V[heads, 1])
    assert lat.min() > LAY.RAIL_Y - RT.FOOT_W / 2 - LAY.SHANK - 0.001                  # the heads reach over the foot
    assert lat.min() < LAY.RAIL_Y - RT.FOOT_W / 2 and lat.max() > LAY.RAIL_Y + RT.FOOT_W / 2
    shanks = np.abs(sp.V[:, 1])
    assert not np.any((shanks > LAY.RAIL_Y - RT.FOOT_W / 2 + 0.001) & (shanks < LAY.RAIL_Y + RT.FOOT_W / 2 - 0.001)
                      & (sp.V[:, 2] < LAY.SPIKE_DROP + LAY.FOOT_EDGE - 0.001))       # no shank through the foot


@pytest.mark.parametrize("k", [0.0, 1 / 60.0, -1 / 300.0])
def test_the_bed_pieces_join_into_one_bed_with_ramps_at_the_front_and_the_start(k):
    """A piece's start ring lies on the last piece's end ring (one surface, its shading carried on), its toe one tie
    spacing on and its top a bed's height short of it (the front ramp); the first piece's start ramps up the other way."""
    first, piece = LAY.bed_mesh(k, first=True, rings=5), LAY.bed_mesh(k, rings=5)
    m = len(LAY.bed_profile())
    for mesh in (first, piece):
        nq, _ = mesh.face_vectors()                                # the open-bottomed shell faces out: its top up
        cq, _ = mesh.face_centres()
        assert (nq[cq[:, 2] > RT.BED_H - 0.01][:, 2] > 0).all()
        assert mesh.V[:, 2].max() == pytest.approx(RT.BED_H + 0.004, abs=1e-6)
    p = RT.Plan(10.0, 0.0 if k == 0.0 else 1.0 / k)

    def placed(mesh, j):                                       # piece j's vertices in the track's frame
        x, y, h = (float(v) for v in p.pose(j * RT.SPACING))
        R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        return np.column_stack([mesh.V[:, :2] @ R.T + [x, y], mesh.V[:, 2]])
    end0 = placed(first, 0)[4 * m:5 * m]
    start1 = placed(piece, 1)[:m]
    assert np.abs(end0 - start1).max() < 1e-9
    if k == 0.0:
        side, side0 = piece.V[:5 * m], first.V[:5 * m]          # the rings (the caps' centres come after)
        top = np.abs(side[:, 1]) < 1e-9                          # the crown's line
        assert side[:, 0].max() == pytest.approx(RT.SPACING)                            # the toe
        assert side[top, 0].max() == pytest.approx(RT.SPACING - RT.BED_H)               # the top: the front ramp
        assert side[top, 0].min() == pytest.approx(-RT.BED_H) and side0[:, 0].min() == pytest.approx(0.0)
        assert side0[np.abs(side0[:, 1]) < 1e-9, 0].min() == pytest.approx(RT.BED_H)   # the start ramps up


def test_options_and_the_card():
    with pytest.raises(LAY.LayoutError):
        LAY.options({"plates": "yes"})
    with pytest.raises(LAY.LayoutError):
        LAY.options({"radius": 10})
    with pytest.raises(LAY.LayoutError):
        LAY.options({"laid": "all"})
    opts, plan = LAY.options({"length": 12, "radius": -200, "laid": 0, "gold": "#ffffff"})
    assert opts["laid"] == 0.0 and plan.n == 20 and plan.k == pytest.approx(-1 / 200)
    assert LAY.options({})[0]["laid"] == pytest.approx(LAY.options({})[1].done)          # whole by default
    c = LAY.card("tr", plan, opts, {"tie": "#123456"})
    assert c["params"] == {"laid": 0.0} and c["origin"] == "start"
    assert c["track"]["spacing"] == RT.SPACING and c["track"]["rail"] == pytest.approx(RT.RAIL)
    assert c["track"]["done"] == pytest.approx(plan.done) and c["track"]["ties"] == 20
    anchors = {a["name"]: a for a in c["use"]["anchor"]}
    assert anchors["front"]["object"] == "tr_front"
    assert anchors["last_tie"]["point"][2] == pytest.approx(LAY.TIE_TOP)
    lo, hi = np.array(c["bounds"]["min"]), np.array(c["bounds"]["max"])
    P = plan.point(np.linspace(0, plan.length, 50), y=LAY.BED_TOE_Y)
    assert np.all(P[:, :2] >= lo[:2] - 1e-6) and np.all(P[:, :2] <= hi[:2] + 1e-6)


def test_the_ties_vary_deterministically_and_keep_their_line_under_tie_plates():
    p = PLANS["straight"]
    a, b = RT.tie_variety(p, seed=4), RT.tie_variety(p, seed=4)
    assert a == b and a != RT.tie_variety(p, seed=5)
    assert len({v for v, *_ in a}) > 3 and all(abs(dy) <= 0.03 and abs(yaw) <= 0.8 for _, _, dy, yaw, _ in a)
    assert all(dy == 0.0 and yaw == 0.0 for _, _, dy, yaw, _ in RT.tie_variety(p, seed=4, plates=True))
