"""The ribbons' maths (mkmmd/core/ribbon.py, docs/design.md: Ribbons), no Blender: the entries checked, the paths an orbit,
a spiral, a swoop, a nozzle's lead-in, a fall and a pile make, the even resample with the faces square to it, the
draw-on clock and its keys, the beat pulses, and the push that keeps a centre line clear of a body."""
import math

import numpy as np
import pytest

from mkmmd.core import ribbon as RB


def entry(**kw):
    base = {"name": "r", "orbit": {"around": "cast:a.upper_body"}, "draw": [1.0, 2.0]}
    base.update(kw)
    return base


# ---------------------------------------------------------------- checking
def test_an_orbit_entry_takes_the_style_and_path_defaults():
    r = RB.check(entry())
    assert r["style"] == "crayon" and r["path"]["kind"] == "orbit" and r["face"] == "out"
    assert r["path"]["radius"] == (0.45, 0.45) and r["path"]["turns"] == 1.0 and r["path"]["tilt"] == (0.0, 0.0)
    assert r["at"] == 1.0 and r["ease"] == "linear" and r["clear"] == {"margin": RB.MARGIN, "cast": None}
    assert r["width"] is None and RB.sizes(r, 0.05) == (0.05, pytest.approx(0.0225), 0.01)
    swoop = RB.check({"name": "s", "style": "neon", "swoop": {"points": [[0, 0, 1], [1, 0, 1]]}, "draw": [0, 1]})
    assert swoop["face"] == "up" and swoop["halo"] == RB.HALO and swoop["glow"] == RB.GLOW


@pytest.mark.parametrize("bad, names", [
    ({"colour": "love"}, ["colour"]),
    ({"style": "pastel"}, ["pastel"]),
    ({"swoop": {"points": [[0, 0, 0], [1, 0, 0]]}}, ["orbit", "swoop"]),
    ({"draw": [2.0, 1.0]}, ["draw"]),
    ({"retract": [3, 4], "chase": [3, 4]}, ["retract", "chase"]),
    ({"retract": [1.5, 3.0]}, ["retract", "2"]),
    ({"stripes": ["love", "gold"]}, ["stripes"]),
    ({"style": "toothpaste", "thickness": 0.01}, ["thickness"]),
    ({"style": "toothpaste", "color": "love"}, ["color", "stripes"]),
    ({"glow": 3.0}, ["glow", "neon"]),
    ({"from": "tube.nozzle"}, ["from", "tube.nozzle"]),
    ({"pulse": {"gain": 0.2}}, ["lane", "hits"]),
    ({"pulse": {"lane": "kick", "hits": [1.0]}}, ["lane", "hits"]),
    ({"orbit": {"around": "cast:a", "turns": 0}}, ["turns"]),
    ({"orbit": {"around": "cast:a", "spin": 2}}, ["spin"]),
    ({"ride": "her"}, ["ride", "her"]),
    ({"squeeze": True}, ["squeeze", "from"]),
])
def test_a_bad_entry_is_refused_naming_what_is_wrong(bad, names):
    with pytest.raises(RB.RibbonError) as e:
        RB.check(entry(**bad))
    assert "ribbon 'r'" in str(e.value) and all(n in str(e.value) for n in names), str(e.value)


def test_a_from_alone_falls_and_piles_and_a_fall_needs_a_from():
    r = RB.check({"name": "p", "style": "toothpaste", "from": "prop:tube.nozzle", "draw": [0, 1]})
    assert r["path"] == {"kind": "fall", "reach": 0.25, "floor": 0.0} and r["from"] == {"prop": "tube", "use": "nozzle"}
    assert r["pile"] == {"turns": 2.5, "radius": None, "shrink": 0.6, "step": None}
    assert r["stripes"] == list(RB.STRIPES)
    for bad in ({"name": "p", "fall": {}, "draw": [0, 1]}, {"name": "p", "draw": [0, 1]}):
        with pytest.raises(RB.RibbonError) as e:
            RB.check(bad)
        assert "from" in str(e.value)


# ---------------------------------------------------------------- paths
def test_an_orbit_runs_round_its_centre_from_her_front_toward_her_left_at_its_radius():
    o = RB.check(entry(orbit={"around": [0, 0, 0], "radius": 0.5, "turns": 0.25}))["path"]
    o["start"] = 0.0
    P, out = RB.orbit([1.0, 2.0, 1.0], np.eye(3), o)
    assert np.linalg.norm(P[:, :2] - [1.0, 2.0], axis=1) == pytest.approx(0.5, abs=1e-9)
    assert P[0] == pytest.approx([1.0, 1.5, 1.0]) and P[-1] == pytest.approx([1.5, 2.0, 1.0])  # front, then her left
    assert np.einsum("ij,ij->i", out, P - [1.0, 2.0, 1.0]) == pytest.approx(0.5, abs=1e-9)    # out is away from the axis
    o = RB.check(entry(orbit={"around": [0, 0, 0], "radius": [0.5, 0.3], "turns": -1.0, "rise": 0.2, "height": 0.1,
                              "tilt": 30}))["path"]
    o["start"] = 0.0
    P, _ = RB.orbit([0.0, 0.0, 0.0], np.eye(3), o)
    r = np.linalg.norm(P, axis=1)
    assert r[0] == pytest.approx(math.hypot(0.5, 0.1), abs=1e-6) and r[-1] == pytest.approx(math.hypot(0.3, 0.3), abs=1e-6)
    assert P[0, 2] > 0.1 + 0.5 * math.sin(math.radians(30)) - 0.02       # the front raised by the tilt
    quarter = P[len(P) // 4]
    assert quarter[0] < -0.2                                              # negative turns: toward her right first


def test_an_orbit_in_a_members_frame_starts_in_front_of_her_whatever_her_yaw():
    yaw = math.radians(90.0)                                              # she faces +X
    axes = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    o = RB.check(entry(orbit={"around": [0, 0, 0], "radius": 0.4, "turns": 0.5}))["path"]
    o["start"] = 0.0
    P, _ = RB.orbit([0.0, 0.0, 1.0], axes, o)
    assert P[0] == pytest.approx([0.4, 0.0, 1.0], abs=1e-9)
    assert RB.orbit_start([0.0, 0.0, 1.0], axes, o, [0.0, 0.7, 1.0]) == pytest.approx(90.0)   # her left


def test_a_spiral_climbs_from_its_bottom_to_its_top_round_the_axis():
    o = RB.check({"name": "s", "spiral": {"around": "cast:a", "radius": [0.4, 0.3], "turns": 2.0, "bottom": 0.2},
                  "draw": [0, 1]})["path"]
    P, out = RB.spiral([1.0, 0.0, 0.5], np.eye(3), o, top=1.8)
    assert P[0, 2] == pytest.approx(0.7) and P[-1, 2] == pytest.approx(2.3)
    assert np.linalg.norm(P[0, :2] - [1.0, 0.0]) == pytest.approx(0.4) and np.linalg.norm(P[-1, :2] - [1.0, 0.0]) == \
        pytest.approx(0.3)
    ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0] - 1.0))
    assert ang[-1] - ang[0] == pytest.approx(4.0 * math.pi, abs=1e-6)
    with pytest.raises(RB.RibbonError, match="not above its bottom"):
        RB.spiral([0, 0, 0], np.eye(3), o, top=0.1)


def test_a_swoop_passes_through_its_points_and_a_closed_one_comes_back():
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.5, 1.5], [2.0, 0.0, 1.0], [3.0, 1.0, 1.2]])
    P = RB.catmull(pts)
    for p in pts:
        assert np.linalg.norm(P - p, axis=1).min() < 1e-9
    assert np.linalg.norm(np.diff(P, axis=0), axis=1).max() < 2.0 * RB.RAW_STEP
    Q = RB.catmull(pts[:3], closed=True)
    assert Q[0] == pytest.approx(Q[-1])
    with pytest.raises(RB.RibbonError, match="same place"):
        RB.catmull(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float))


def test_a_lead_in_starts_inside_the_nozzle_and_leaves_it_along_its_direction():
    P = np.stack([np.linspace(0.5, 1.5, 50), np.full(50, 0.3), np.full(50, 1.0)], 1)
    Q, n = RB.lead_in([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], P, inset=0.06)
    assert Q[0] == pytest.approx([0.0, 0.0, 0.94]) and n < len(Q) and Q[n] == pytest.approx(P[0])
    mouth = int(np.argmin(np.linalg.norm(Q[:n] - [0.0, 0.0, 1.0], axis=1)))
    assert np.allclose(Q[:mouth + 1, :2], 0.0, atol=1e-12)                 # straight out along +Z to the mouth
    T = RB.tangents(Q)
    assert T[n - 1] @ T[n + 1] > 0.99                                      # into the path along its own tangent


def test_a_fall_lands_on_the_floor_ahead_and_a_pile_climbs_on_itself_shrinking():
    f = {"reach": 0.3, "floor": 0.1}
    P = RB.fall([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], f, width=0.04)
    assert P[-1] == pytest.approx([0.3, 0.0, 0.12], abs=1e-9) and np.all(np.diff(P[len(P) // 2:, 2]) < 1e-12)
    with pytest.raises(RB.RibbonError, match="not above the floor"):
        RB.fall([0.0, 0.0, 0.1], [1.0, 0.0, 0.0], f, width=0.04)
    coil, out = RB.pile(P, 0.04, {"turns": 2.0, "radius": 0.08, "shrink": 0.5, "step": 0.03})
    c = P[-1] + np.array([0.0, 0.08, 0.0])                                 # to the left of the heading (+X)
    r = np.linalg.norm(coil[:, :2] - c[:2], axis=1)
    assert r[0] == pytest.approx(0.08, rel=0.01) and r[-1] == pytest.approx(0.04, rel=1e-6)
    assert coil[-1, 2] - P[-1, 2] == pytest.approx(0.06) and np.all(np.diff(coil[:, 2]) > 0)
    assert np.einsum("ij,ij->i", out[:, :2], coil[:, :2] - c[:2]) == pytest.approx(r, rel=1e-9)


def test_a_centre_line_is_even_its_faces_square_and_its_nozzle_part_pinned():
    r = RB.check({"name": "p", "style": "toothpaste", "from": "prop:t.nozzle",
                  "orbit": {"around": [0, 0, 1], "radius": 0.5, "turns": 1.0}, "draw": [0, 1], "pile": True})
    w, _, ds = RB.sizes(r, 0.04)
    c = RB.centre_line(r, {"center": np.array([0.0, 0.0, 1.0]), "axes": np.eye(3),
                           "from": (np.array([0.8, -0.2, 1.1]), np.array([-1.0, 0.0, 0.0]))}, w, ds)
    P, F, s = c["P"], c["F"], c["s"]
    assert np.diff(s) == pytest.approx(np.full(len(s) - 1, np.diff(s).mean()), rel=1e-6) and np.diff(s).mean() <= ds
    assert np.abs(np.einsum("ij,ij->i", F, RB.tangents(P))).max() < 1e-9 and np.linalg.norm(F, axis=1) == \
        pytest.approx(1.0)
    assert P[0] == pytest.approx([0.8 + 1.5 * w, -0.2, 1.1], abs=1e-9)    # inside the nozzle, 1.5 widths back
    assert s[c["pinned"] - 1] <= c["lead_m"] + w < s[c["pinned"]]
    assert np.linalg.norm(P[-1, :2]) < 0.5 and P[-1, 2] > 1.0              # it ends coiled up on itself
    crayon = RB.check(entry(orbit={"around": [0, 0, 0], "radius": 0.4}, seed=3))
    a = RB.centre_line(crayon, {"center": np.zeros(3), "axes": np.eye(3)}, 0.05, 0.01)
    b = RB.centre_line(crayon, {"center": np.zeros(3), "axes": np.eye(3)}, 0.05, 0.01)
    assert np.array_equal(a["P"], b["P"])                                    # the hand's wobble is seeded
    off = np.abs(np.linalg.norm(a["P"][:, :2], axis=1) - 0.4)
    assert 0.003 < off.max() <= RB.WOBBLE * 0.05 * 1.2 and 0.88 <= a["wscale"].min() and a["wscale"].max() <= 1.12


def test_faces_follow_the_wanted_direction_and_carry_on_where_the_path_runs_along_it():
    t = np.linspace(0.0, 1.0, 200)
    P = np.stack([np.zeros(200), np.zeros(200), t], 1)                     # straight up: "up" runs along it
    F = RB.faces(P, "up")
    assert np.abs(np.einsum("ij,ij->i", F, RB.tangents(P))).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", F[1:], F[:-1])).min() > 0.999   # no flips
    Q = np.stack([t, np.zeros(200), np.zeros(200)], 1)
    assert RB.faces(Q, [0.0, -1.0, 0.3]) == pytest.approx(np.tile([0.0, -0.957826, 0.287348], (200, 1)), abs=1e-6)


# ---------------------------------------------------------------- timing
def test_the_clock_draws_on_eased_then_retracts_or_is_chased():
    r = RB.check(entry(draw=[1.0, 3.0], ease="smooth", retract=[5.0, 6.0]))
    assert [RB.clock(r, t) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 7.0)] == [
        (0.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.0, 0.0)]
    assert RB.clock(r, 1.5)[0] == pytest.approx(RB.ease("smooth", 0.25)) and RB.ease("smooth", 0.25) < 0.25
    c = RB.check(entry(draw=[1.0, 2.0], chase=[3.0, 4.0], ease="out"))
    assert RB.clock(c, 3.5) == (1.0, pytest.approx(0.75)) and RB.clock(c, 9.0) == (1.0, 1.0)
    keys = RB.clock_keys(c, 30.0, 31)
    frames = [f for f, _, _ in keys]
    assert frames == list(range(61, 92)) + list(range(121, 152))
    for f, g, tl in keys:
        assert (g, tl) == RB.clock(c, (f - 31) / 30.0)
    assert RB.ease("in", 0.5) == 0.25 and RB.ease("out", 0.5) == 0.75 and RB.ease("linear", 2.0) == 1.0


def test_pulses_peak_on_the_hits_and_overlapping_ones_take_the_larger_exactly():
    hits = [1.0, 1.2, 2.0]
    keys = RB.pulse_keys(hits, gain=0.5, attack=0.05, decay=0.3)

    def exact(t):
        v = 0.0
        for h in hits:
            if h - 0.05 <= t < h:
                v = max(v, (t - (h - 0.05)) / 0.05)
            elif h <= t <= h + 0.3:
                v = max(v, 1.0 - (t - h) / 0.3)
        return 1.0 + 0.5 * v

    ts = np.linspace(0.5, 2.6, 2101)
    assert [RB.value_at(keys, t) for t in ts] == pytest.approx([exact(t) for t in ts], abs=1e-9)
    assert RB.value_at(keys, 1.2) == 1.5 and RB.value_at(keys, 0.0) == 1.0 and RB.value_at(keys, 3.0) == pytest.approx(1.0)
    assert RB.pulse_keys([], 0.5, 0.05, 0.3) == [] and RB.value_at([], 1.0) == 1.0


def test_a_ribbon_shows_the_points_between_its_tail_and_its_front():
    s = np.linspace(0.0, 2.0, 21)
    assert np.flatnonzero(RB.visible(s, 2.0, 0.5, 0.0)).tolist() == list(range(11))
    assert np.flatnonzero(RB.visible(s, 2.0, 1.0, 0.75)).tolist() == list(range(15, 21))
    assert not RB.visible(s, 2.0, 0.0, 0.0).any() and not RB.visible(s, 2.0, 1.0, 1.0).any()


def test_the_squeeze_follows_the_draw_and_a_retract_lets_go():
    r = RB.check(entry(draw=[1.0, 2.0], retract=[3.0, 4.0]))
    assert RB.squeeze_keys(r, 0.0, 0.8) == [(1.0, 0.0), (2.0, 0.8), (3.0, 0.8), (4.0, 0.0)]


# ---------------------------------------------------------------- keeping clear
def sphere(c, R, rad):
    def measure(P):
        d = P - c
        dist = np.linalg.norm(d, axis=1)
        return R + rad - dist, d / np.maximum(dist, 1e-12)[:, None]
    return measure


def test_a_line_through_a_body_bends_round_it_smoothly_and_its_pinned_start_stays():
    n = 401
    P = np.stack([np.linspace(-1.0, 1.0, n), np.full(n, 0.05), np.zeros(n)], 1)
    Q, rep = RB.push_clear(P, sphere(np.zeros(3), 0.3, 0.04), sigma=6.0)
    gap = np.linalg.norm(Q, axis=1) - 0.34
    assert gap.min() > -1e-3 and rep["need_m"] <= 5e-4 and rep["points"] > 0 and 0.2 < rep["moved_m"] < 0.35
    bend = np.linalg.norm(np.diff(Q, 2, axis=0), axis=1)
    assert bend.max() < 0.02                                               # no kink: it bends round
    assert np.allclose(Q[:50], P[:50]) and np.allclose(Q[-50:], P[-50:])  # far from the body nothing moves
    R, rep = RB.push_clear(P, sphere(np.array([-0.6, 0.0, 0.0]), 0.3, 0.04), pinned=200, sigma=6.0)
    assert np.array_equal(R[:200], P[:200]) and rep["need_m"] > 0.1     # a body on the pinned part: reported, not moved
    Q, rep = RB.push_clear(P, sphere(np.array([0.0, 5.0, 0.0]), 0.3, 0.04))
    assert np.array_equal(Q, P) and rep == {"rounds": 0, "moved_m": 0.0, "points": 0, "need_m": 0.0}


def test_the_envelope_is_the_fattest_the_surface_gets():
    crayon = RB.check(entry())
    assert RB.envelope(crayon, 0.05, np.array([1.0, 1.1]), 1.3) == pytest.approx([0.0325, 0.03575])
    neon = RB.check({"name": "n", "style": "neon", "swoop": {"points": [[0, 0, 1], [1, 0, 1]]}, "draw": [0, 1]})
    assert RB.envelope(neon, 0.03, np.ones(2)) == pytest.approx(np.full(2, RB.HALO_SEEN * 4.0 * 0.015))


def test_a_pushed_line_is_respaced_evenly_with_its_width_scales_following():
    P = np.stack([np.r_[np.linspace(0, 1, 30), np.linspace(1.02, 2, 5)], np.zeros(35), np.zeros(35)], 1)
    F = np.tile([0.0, 0.0, 1.0], (35, 1))
    w = np.linspace(1.0, 2.0, 35) ** 0
    w[-5:] = 1.1
    Q, G, w2, s = RB.respace(P, F, w, 0.05)
    assert np.diff(s) == pytest.approx(np.full(len(s) - 1, 0.05), rel=1e-6) and s[-1] == pytest.approx(2.0)
    assert w2[0] == 1.0 and w2[-1] == pytest.approx(1.1) and G == pytest.approx(np.tile([0, 0, 1.0], (len(Q), 1)))
