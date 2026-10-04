"""The hearts prop's maths (mkmmd/core/hearts.py), without Blender: the outline of a heart, its pillow mesh, the layout of a field
and the rise / sway / pop of every heart, and the driver expressions Blender evaluates (checked against the numpy reference by
evaluating them in Python with Blender's function names)."""
import ast
import math

import numpy as np
import pytest

from mkmmd.core import hearts as HR
from mkmmd.core import shell as S


# ===================================================================================================================
# the outline
# ===================================================================================================================
@pytest.fixture(scope="module")
def outline():
    return HR.heart_outline(96)


def _curvature(P):
    """Signed curvature at each vertex of a closed counter-clockwise polygon (positive: convex), by the circumscribed circle."""
    a, b, c = np.roll(P, 1, axis=0), P, np.roll(P, -1, axis=0)
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    la, lb, lc = (np.linalg.norm(u - v, axis=1) for u, v in ((b, a), (c, b), (c, a)))
    return 2.0 * cr / (la * lb * lc)


def test_outline_is_a_closed_counter_clockwise_unit_width_loop(outline):
    P = outline
    assert P.shape == (96, 2)
    assert HR.signed_area(P) > 0                                       # counter-clockwise from the front (x right, z up)
    assert P[:, 0].max() - P[:, 0].min() == pytest.approx(1.0, abs=1e-9)
    assert HR.area_centroid(P) == pytest.approx([0.0, 0.0], abs=1e-9)
    steps = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    assert steps.max() / steps.min() < 1.15                            # evenly spaced by arc length


def test_outline_is_mirror_symmetric_from_the_tip_to_the_notch(outline):
    P = outline
    assert P[0, 1] == P[:, 1].min() and P[0, 0] == 0.0                 # point 0 is the tip, at the bottom
    assert P[48, 0] == 0.0                                             # point n / 2 is the bottom of the notch
    for j in range(1, 48):
        assert P[96 - j] == pytest.approx(P[j] * (-1.0, 1.0), abs=1e-12)


def test_outline_reads_as_a_heart(outline):
    P = outline
    height = P[:, 1].max() - P[:, 1].min()
    assert 0.8 < height < 0.95                                         # a little wider than tall, as a classic heart
    notch_depth = (P[:, 1].max() - P[48, 1]) / height
    assert 0.10 < notch_depth < 0.30                                   # two lobes with a real cleft, not a dimple
    assert P[48, 1] > P[:, 1].min() + 0.5 * height                     # the cleft is in the upper half
    k = _curvature(P)
    concave = np.nonzero(k < 0)[0]
    notch = np.abs(concave - 48) <= 12
    assert notch.sum() >= 3 and k[48] < -10                            # the notch curves in sharply ...
    assert np.all(k[concave[~notch]] > -1.5)                           # ... the sides only a little (nearly straight to the tip)
    assert 0.02 < 1.0 / k[0] < 0.15                                    # the tip is rounded, not a needle
    assert 1.0 / abs(k[48]) > 0.02                                     # the notch is rounded as well


def test_outline_is_star_shaped_about_its_centre(outline):
    ang = np.unwrap(np.arctan2(outline[:, 1], outline[:, 0]))
    d = np.diff(np.concatenate([ang, ang[:1] + 2.0 * math.pi]))
    assert np.all(d > 0)                                               # every ray from the centre meets the outline once
    assert ang[-1] - ang[0] < 2.0 * math.pi


def test_rounding_softens_the_tip_and_the_notch():
    sharp, soft = HR.heart_outline(96, 14.0), HR.heart_outline(96, 7.0)
    assert 1.0 / _curvature(soft)[0] > 1.0 / _curvature(sharp)[0]
    assert soft[48, 1] > sharp[48, 1]                                  # the cleft fills in


def test_outline_needs_an_even_point_count():
    with pytest.raises(ValueError):
        HR.heart_outline(95)
    with pytest.raises(ValueError):
        HR.heart_outline(8)


# ===================================================================================================================
# the mesh
# ===================================================================================================================
@pytest.fixture(scope="module")
def mesh():
    return HR.heart_mesh()


def test_mesh_is_a_closed_outward_pillow(mesh):
    assert mesh.is_closed() and mesh.volume() > 0
    lo, hi = mesh.bbox()
    assert hi[0] - lo[0] == pytest.approx(1.0, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(HR.PUFF, abs=1e-9)           # thickness along y, centred
    assert lo[1] == pytest.approx(-hi[1], abs=1e-12)
    n = mesh.vertex_normals()
    out = np.einsum("ij,ij->i", n, mesh.V - mesh.V.mean(0))
    assert (out > -1e-9).mean() > 0.97                                 # normals point away from the middle (but at the notch)
    nq, nt = mesh.face_vectors()
    assert np.linalg.norm(nq, axis=1).min() > 0 and np.linalg.norm(nt, axis=1).min() > 0


def test_mesh_rim_is_the_outline(mesh):
    O = HR.heart_outline(96)
    eq = mesh.V[np.isclose(mesh.V[:, 1], 0.0, atol=1e-9)]               # the equator ring: the rim of the pillow
    assert len(eq) == 96
    for p in eq[:, [0, 2]]:
        assert np.linalg.norm(O - p, axis=1).min() < 1e-9


def test_mesh_is_inside_its_outline_seen_from_the_front(mesh):
    """Seen from the front the silhouette of the pillow is the heart: every vertex projects inside the outline."""
    O = HR.heart_outline(96)
    x, z = mesh.V[:, 0], mesh.V[:, 2]
    inside = np.zeros(len(x), bool)
    j = len(O) - 1
    for i in range(len(O)):                                            # even-odd ray crossing, vectorised over the points
        xi, zi, xj, zj = O[i, 0], O[i, 1], O[j, 0], O[j, 1]
        cross = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / (zj - zi + 1e-300) + xi)
        inside ^= cross
        j = i
    rim = np.isclose(mesh.V[:, 1], 0.0, atol=1e-9)
    assert inside[~rim].all()


def test_mesh_front_pole_faces_the_viewer(mesh):
    i = int(np.argmin(mesh.V[:, 1]))
    assert mesh.V[i] == pytest.approx([0.0, -HR.PUFF / 2, 0.0], abs=1e-12)          # the front pole looks along -Y


def test_mesh_has_a_valley_between_the_lobes(mesh):
    """At the same distance from the centre the pillow is thicker over a lobe than over the notch: a valley runs from the middle
    into the cleft, as on a real puffy heart."""
    n, rings = 96, 20
    O = HR.heart_outline(n)
    ang = np.degrees(np.arctan2(O[:, 1], O[:, 0]))
    notch, lobe = 48, int(np.argmin(np.abs(ang - 60.0)))
    assert np.hypot(*O[lobe]) > 1.5 * np.hypot(*O[notch])
    r = 0.16

    def thickness(j):
        ring = np.arange(1, rings // 2 + 1)                              # front half: pole .. equator
        pts = mesh.V[(ring - 1) * n + j]
        return float(np.interp(r, np.hypot(pts[:, 0], pts[:, 2]), -pts[:, 1]))     # depth in front of the equator plane
    assert thickness(lobe) > 1.3 * thickness(notch) > 0


def test_mesh_arguments_are_checked():
    with pytest.raises(ValueError):
        HR.heart_mesh(rings=7)
    assert HR.heart_mesh(0.8).bbox()[1][1] == pytest.approx(0.4)


def test_halfwidth_grows_with_the_turn():
    base = HR.halfwidth(0.0, 0.0)
    assert base == pytest.approx(0.5 * HR.OVERSHOOT, abs=1e-6)
    assert HR.halfwidth(25.0, 10.0) > base
    assert HR.halfwidth(40.0, 20.0) > HR.halfwidth(25.0, 10.0)
    assert HR.halfwidth(25.0, 10.0) < 0.62                              # the room a field leaves is under 62 % of a heart's size


# ===================================================================================================================
# the options
# ===================================================================================================================
def test_options_defaults_and_overrides():
    count, seed, P, puff, colour, glow = HR.options({})
    assert (count, seed, colour) == (16, 1, "love") and P == HR.DEFAULTS and puff == HR.PUFF and glow == 0.3
    count, seed, P, puff, colour, glow = HR.options({"count": 12, "size": [0.07, 0.15], "clear": 0.3, "heart": "rose", "rise": "0.4"})
    assert count == 12 and (P["size_min"], P["size_max"]) == (0.07, 0.15) and P["clear"] == 0.3 and P["rise"] == 0.4


@pytest.mark.parametrize("bad", [{"count": 0}, {"size": [0.2, 0.1]}, {"size": 0.1}, {"rise": 0}, {"clear": -1}, {"fan": 2},
                                 {"puff": 3}, {"spread": "wide"}, {"pop": 0}])
def test_options_reject_what_cannot_work(bad):
    with pytest.raises(ValueError):
        HR.options(bad)


# ===================================================================================================================
# the layout
# ===================================================================================================================
def test_layout_is_deterministic_and_seeded():
    a, b, c = HR.Layout(16, 7), HR.Layout(16, 7), HR.Layout(16, 8)
    for f in HR.Layout.FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert not np.array_equal(a.o, c.o)
    with pytest.raises(ValueError):
        HR.Layout(0)


@pytest.mark.parametrize("count", [1, 2, 5, 12, 16, 20, 33])
def test_layout_covers_phases_sides_and_sizes_evenly(count):
    L = HR.Layout(count, 3)
    for f in HR.Layout.FIELDS:
        assert getattr(L, f).shape == (count,) and np.all(np.isfinite(getattr(L, f)))
    assert np.all((L.o >= 0) & (L.o < 1)) and np.all((L.d >= 0) & (L.d <= 1)) and np.all((L.r >= 0) & (L.r <= 1))
    assert set(L.v) <= {0.97, 1.03} and np.all(np.abs(L.e) <= 1)
    assert abs(L.side.sum()) <= 1                                       # as many hearts left as right
    for side in (1.0, -1.0):                                            # within a plume the births are evenly spaced
        o = np.sort(L.o[L.side == side])
        m = len(o)
        if m < 3:
            continue
        gaps = np.diff(np.concatenate([o, [o[0] + 1.0]]))
        assert gaps.min() > 0.7 / m and gaps.max() < 1.3 / m, (side, gaps * m)
        steps = np.abs(np.diff(L.d[L.side == side]))                    # neighbours differ in anchor and in size
        assert steps.min() > 0.05 or m < 4


@pytest.mark.parametrize("count", [10, 14, 18, 20])
@pytest.mark.parametrize("P", [dict(HR.DEFAULTS, clear=0.3, spread=0.8, fan=0.6), dict(HR.DEFAULTS, clear=0.25, spread=0.7, fan=0.0),
                               dict(HR.DEFAULTS, clear=0.3, spread=0.7, fan=0.6, height=1.9, clear_top=0.4)])
def test_hearts_do_not_run_into_each_other(count, P):
    """Seen from the front no two hearts touch, at any time of a long clip (the layout gives each plume even births and one
    speed), so a flat silhouette keeps every heart a heart."""
    for seed in (1, 2, 3, 4):
        L = HR.Layout(count, seed)
        q = HR.pose(_frames(40, 15), P, L, _hw(P))
        x, z, s = q["x"], q["z"], q["s"]
        for i in range(count):
            for j in range(i + 1, count):
                gap = np.hypot(x[:, i] - x[:, j], z[:, i] - z[:, j]) - 0.5 * (s[:, i] + s[:, j]) * 0.95
                grown = (s[:, i] > 0.4 * HR.size(P, L)[i]) & (s[:, j] > 0.4 * HR.size(P, L)[j])
                assert gap[grown].min() > 0 if grown.any() else True, (count, seed, i, j)


# ===================================================================================================================
# the field
# ===================================================================================================================
PARAM_SETS = [
    dict(HR.DEFAULTS),
    dict(HR.DEFAULTS, clear=0.3, spread=0.65, fan=0.0),
    dict(HR.DEFAULTS, clear=0.25, spread=0.7, fan=1.0, sway=0.1, rise=0.4, height=2.0, size_min=0.08, size_max=0.2),
    dict(HR.DEFAULTS, clear=0.3, spread=0.7, fan=0.6, height=1.9, clear_top=0.5),
]


def _hw(P):
    return HR.halfwidth(P["spin"], P["tilt"])


def _frames(seconds, fps=30):
    return np.arange(int(seconds * fps)) / fps


@pytest.mark.parametrize("P", PARAM_SETS)
def test_hearts_climb_steadily_through_the_column_and_wrap_at_the_ends(P):
    L = HR.Layout(16, 1)
    t = _frames(40)
    q = HR.pose(t, P, L, _hw(P))
    z = q["z"]
    assert z.min() >= -0.5 * P["height"] - 1e-9 and z.max() <= 0.5 * P["height"] + 1e-9
    dz = np.diff(z, axis=0)
    up = dz > 0
    wraps = ~up
    assert wraps.sum(axis=0).min() >= 1                                 # every heart is reborn within 40 s
    assert np.all(dz[up] == pytest.approx(P["rise"] * L.v[np.nonzero(up)[1]] / 30.0, rel=1e-6))     # constant speed
    # each heart's cycle is its life: the time between births
    lf = HR.life(P, L)
    for i in range(L.count):
        births = t[1:][wraps[:, i]]
        if len(births) > 1:
            assert np.diff(births) == pytest.approx(lf[i], abs=1.0 / 30.0)


@pytest.mark.parametrize("P", PARAM_SETS)
def test_nothing_pops_but_the_deliberate_pop_in(P):
    """Between two frames a heart's scale, position and turn change by a little; the only jump is the rebirth, at scale 0."""
    L = HR.Layout(16, 2)
    t = _frames(30)
    q = HR.pose(t, P, L, _hw(P))
    wrap = np.diff(q["age"], axis=0) < 0
    for key, limit in (("x", 0.02), ("y", 0.02), ("z", 0.03), ("rx", 0.02), ("ry", 0.02), ("rz", 0.05)):
        step = np.abs(np.diff(q[key], axis=0))
        assert step[~wrap].max() < limit, key
    sz = HR.size(P, L)
    ds = np.abs(np.diff(q["s"], axis=0)) / (sz * P["amount"])
    birth = q["age"][:-1] < 1.05 * P["pop"]
    assert ds[birth & ~wrap].max() <= 4.8 / (P["pop"] * 30.0)           # the deliberate pop: the ease starts 4.7 x its mean slope
    assert ds[~birth & ~wrap].max() <= 1.6 / (P["fade"] * 30.0)         # the shrink at the top: a smoothstep, 1.5 x its mean
    # at the rebirth the heart is gone: the frame before it has shrunk to a pin, the frame after is the first of its pop-in
    i, j = np.nonzero(wrap)
    assert np.all(q["s"][i, j] <= 0.02 * sz[j]) and np.all(q["s"][i + 1, j] <= 0.4 * sz[j])
    assert HR.pop_in(0.0) == pytest.approx(0.0, abs=1e-12) and HR.pop_in(1.0) == pytest.approx(1.0)       # born from nothing


@pytest.mark.parametrize("P", PARAM_SETS)
def test_pop_overshoots_ten_percent_and_settles(P):
    L = HR.Layout(8, 4)
    q = HR.pose(_frames(60), P, L, _hw(P))
    ratio = q["s"] / (HR.size(P, L) * P["amount"])
    assert ratio.max() == pytest.approx(HR.OVERSHOOT, abs=0.01)
    mid = (q["age"] > 1.2 * P["pop"]) & (q["age"] < HR.life(P, L) - 1.2 * P["fade"])
    assert np.allclose(ratio[mid], 1.0)                                  # fully grown, no wobble in size between pop and fade


def test_amount_scales_the_field_and_zero_hides_it():
    L = HR.Layout(16, 1)
    full = HR.pose(_frames(5), dict(HR.DEFAULTS), L, 0.57)["s"]
    half = HR.pose(_frames(5), dict(HR.DEFAULTS, amount=0.5), L, 0.57)["s"]
    grown = full > 0.01
    assert half[grown] == pytest.approx(0.5 * full[grown], rel=1e-9)
    gone = HR.pose(_frames(5), dict(HR.DEFAULTS, amount=0.0), L, 0.57)["s"]
    assert gone.max() == HR.MIN_SCALE                                    # never a singular matrix


@pytest.mark.parametrize("P", PARAM_SETS[:2] + [dict(HR.DEFAULTS, clear=0.3, spread=0.7, fan=0.6, spin=30.0, tilt=15.0, sway=0.08)])
def test_hearts_never_enter_the_clear_column(P):
    """The real mesh at the real turn, scale and place: its nearest point to the axis stays beyond `clear`, on its own side,
    at every instant of a long clip and for several seeds (the figure stands in that column; the camera is in front)."""
    P = dict(P, clear=max(P["clear"], 0.25), spread=max(P["spread"], 0.7))
    worst = _nearest_to_axis(P, seeds=(1, 5, 9))
    assert worst >= -1e-9


def _nearest_to_axis(P, seeds, below=None, seconds=24):
    """The smallest (nearest point of the heart to the axis - clear), over hearts, instants and seeds, while the heart's centre is
    lower than `below` (metres above the root; None: always)."""
    m = HR.heart_mesh(n=48, rings=12)
    hw = _hw(P)
    worst = np.inf
    for seed in seeds:
        L = HR.Layout(16, seed)
        t = _frames(seconds, 15)
        q = HR.pose(t, P, L, hw)
        for i in range(L.count):
            for k in range(len(t)):
                if below is not None and q["z"][k, i] > below:
                    continue
                R = S.rot_matrix(math.degrees(q["rx"][k, i]), math.degrees(q["ry"][k, i]), math.degrees(q["rz"][k, i]))
                x = (m.V @ R.T)[:, 0] * q["s"][k, i] + q["x"][k, i]
                near = np.abs(x).min() if np.all(np.sign(x) == L.side[i]) else 0.0
                worst = min(worst, near - P["clear"])
    return worst


def test_clear_column_closes_above_clear_top_and_hearts_arch_over_the_head():
    P = dict(HR.DEFAULTS, clear=0.3, spread=0.7, fan=0.6, height=1.9, clear_top=0.4)
    assert _nearest_to_axis(P, seeds=(1, 5), below=P["clear_top"]) >= -1e-9            # below the top: still outside the column
    L = HR.Layout(16, 1)
    q = HR.pose(_frames(60), P, L, _hw(P))
    inside = np.abs(q["x"]) < P["clear"]
    assert not inside[q["z"] <= P["clear_top"]].any()                                   # (the centres, anywhere below the top)
    assert inside[q["z"] > P["clear_top"] + HR.CLEAR_SOFT].any()                        # above it they do come over the head
    assert np.allclose(q["cl"][q["z"] > P["clear_top"] + HR.CLEAR_SOFT], 0.0)
    assert np.allclose(q["cl"][q["z"] <= P["clear_top"]], P["clear"])
    # and the closing is smooth: no heart's centre moves sideways by more than a few millimetres a frame because of it
    dx = np.abs(np.diff(q["x"], axis=0))
    wrap = np.diff(q["age"], axis=0) < 0
    assert dx[~wrap].max() < 0.02


def test_hearts_stay_within_the_spread_and_the_depth():
    P = dict(HR.DEFAULTS, clear=0.3, spread=0.65, depth=0.2)
    L = HR.Layout(20, 1)
    q = HR.pose(_frames(40), P, L, _hw(P))
    # the centre is within `spread` plus the sway; the depth within `depth` plus the sway's y share
    assert np.abs(q["x"]).max() <= P["spread"] + P["sway"] + 1e-9
    assert np.abs(q["x"]).min() >= P["clear"] + 0.5 * HR.OVERSHOOT * P["size_min"] - 1e-9
    assert np.abs(q["y"]).max() <= P["depth"] + 0.5 * P["sway"] + 1e-9


def test_fan_opens_the_plumes_as_they_rise():
    L = HR.Layout(16, 1)
    P0, P1 = dict(HR.DEFAULTS, clear=0.3, fan=0.0), dict(HR.DEFAULTS, clear=0.3, fan=1.0)
    t = _frames(30)
    a0, a1 = HR.pose(t, P0, L, 0.57), HR.pose(t, P1, L, 0.57)
    low, high = a1["u"] < 0.15, a1["u"] > 0.85
    wide = lambda q, sel: np.abs(q["x"])[sel].mean()          # noqa: E731
    assert wide(a1, low) < wide(a1, high)                                 # the fan: nearer the column at the bottom
    assert wide(a1, low) < wide(a0, low)


def test_turns_are_bounded_by_spin_and_tilt():
    P = dict(HR.DEFAULTS, spin=30.0, tilt=12.0)
    q = HR.pose(_frames(60), P, HR.Layout(20, 6), 0.57)
    assert np.degrees(np.abs(q["rz"]).max()) <= 30.0 + 1e-9 and np.degrees(np.abs(q["ry"]).max()) <= 12.0 + 1e-9
    assert np.degrees(np.abs(q["rx"]).max()) <= 6.0 + 1e-9
    assert np.degrees(np.abs(q["rz"]).max()) > 15.0                       # they do turn
    still = HR.pose(_frames(5), dict(HR.DEFAULTS, spin=0.0, tilt=0.0, sway=0.0), HR.Layout(8, 1), 0.5)
    assert np.abs(still["rz"]).max() == 0 and np.abs(still["ry"]).max() == 0 and np.abs(still["rx"]).max() == 0


# ===================================================================================================================
# the driver expressions
# ===================================================================================================================
def _clamp(x, lo, hi):
    return min(max(x, lo), hi)


def _smoothstep(a, b, x):
    t = _clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


BLENDER = {"fmod": math.fmod, "min": min, "max": max, "clamp": _clamp, "smoothstep": _smoothstep, "sin": math.sin,
           "cos": math.cos, "abs": abs, "radians": math.radians, "sqrt": math.sqrt}


def _evaluate(expr, env):
    """Evaluate a driver expression the way Blender's simple-expression evaluator does (Python syntax, its functions)."""
    return eval(compile(expr, "<driver>", "eval"), {"__builtins__": {}}, dict(BLENDER, **env))   # noqa: S307


def _run(expr_set, P, frame):
    env = dict(P, frame=float(frame))
    for h in HR.HELPERS:
        env[h] = _evaluate(expr_set[h], env)
    return {k: _evaluate(v, env) for k, v in expr_set.items() if k not in HR.HELPERS}, env


@pytest.mark.parametrize("P", PARAM_SETS)
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_expressions_equal_the_numpy_reference(P, fps):
    L = HR.Layout(16, 11)
    hw = _hw(P)
    frames = [0, 1, 17, 40, 61, 77, 123, 250, 999]
    ref = HR.pose(np.array(frames) / fps, P, L, hw)
    for i in range(L.count):
        ex = HR.expressions(L, i, fps, hw)
        for n, f in enumerate(frames):
            got, env = _run(ex, P, f)
            assert env["age"] == pytest.approx(ref["age"][n, i], abs=1e-6)
            assert env["u"] == pytest.approx(ref["u"][n, i], abs=1e-9)
            assert env["cl"] == pytest.approx(ref["cl"][n, i], abs=1e-9)
            for k in ("x", "y", "z", "rx", "ry", "rz", "s"):
                assert got[k] == pytest.approx(ref[k][n, i], abs=1e-6), (k, i, f)


def test_expressions_are_blender_simple_expressions():
    """At most 255 characters (what Blender stores), only Blender's native functions, no operator or name that would need
    Python (`%`, `**`, `//`, attribute access, calls of anything else), every name a root property, a helper or `frame`."""
    ok_names = set(HR.PARAMS) | set(HR.HELPERS) | HR.FUNCS | {"frame"}
    for count, seed in ((16, 1), (20, 7), (12, 3), (64, 9)):
        L = HR.Layout(count, seed)
        for P in PARAM_SETS:
            for i in range(L.count):
                ex = HR.expressions(L, i, 30, _hw(P))
                assert list(ex) == list(HR.CHANNELS) and HR.CHANNELS[: len(HR.HELPERS)] == HR.HELPERS
                for ch, text in ex.items():
                    assert len(text) <= HR.MAX_EXPR, (ch, len(text))
                    tree = ast.parse(text, mode="eval")
                    for node in ast.walk(tree):
                        assert isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
                                                 ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub)), (ch, text)
                        if isinstance(node, ast.Call):
                            assert isinstance(node.func, ast.Name) and node.func.id in HR.FUNCS, (ch, text)
                        if isinstance(node, ast.Name):
                            assert node.id in ok_names, (ch, node.id)
                    assert "e-" not in text and "e+" not in text          # plain decimals only


def test_helper_properties_are_read_only_after_they_are_defined():
    """`age` reads no helper; `u` and `pin` read `age`; `cl`, `z` read `u`; `x` reads `cl` and `u`; `s` reads `age` and `pin`: no
    cycle, the order of CHANNELS is a valid evaluation order."""
    L = HR.Layout(4, 1)
    ex = HR.expressions(L, 0, 30, 0.57)
    names = {k: set(ast.walk(ast.parse(v, mode="eval"))) for k, v in ex.items()}
    used = {k: {n.id for n in nodes if isinstance(n, ast.Name)} & set(HR.HELPERS) for k, nodes in names.items()}
    done = set()
    for ch in HR.CHANNELS:
        assert used[ch] <= done, (ch, used[ch] - done)
        if ch in HR.HELPERS:
            done.add(ch)
    assert used["age"] == set() and used["u"] == {"age"} and used["cl"] == {"u"} and used["z"] == {"u"}
    assert used["x"] == {"cl", "u"} and used["pin"] == {"age"} and used["s"] == {"age", "pin"}


def test_expressions_survive_extreme_parameters():
    """Zero pop / fade / rise cannot divide by zero (Blender would return 0 and flag the driver), and a heart is never exactly
    scale 0."""
    L = HR.Layout(8, 5)
    P = dict(HR.DEFAULTS, pop=0.0, fade=0.0, rise=0.0)
    for i in range(L.count):
        ex = HR.expressions(L, i, 30, 0.57)
        for f in (0, 10, 100):
            got, env = _run(ex, P, f)
            assert all(math.isfinite(v) for v in got.values()) and got["s"] >= HR.MIN_SCALE


def test_negative_frames_are_fine():
    """fmod keeps the sign of the dividend: a clock that starts before frame 0 must not give a negative age."""
    L = HR.Layout(8, 5)
    for i in range(L.count):
        got, env = _run(HR.expressions(L, i, 30, 0.57), HR.DEFAULTS, -300)
        assert 0.0 <= env["age"] < HR.life(HR.DEFAULTS, L)[i] and 0.0 <= env["u"] < 1.0
