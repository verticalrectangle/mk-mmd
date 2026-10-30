"""A swarm of butterflies, bpy-free (mkmmd.core.butterflies): the seeded layout, the formations (rings, orbits, drift), the
burst and the bloom, the wing beats (variety, unison, a hit's ripple), the blend, the instance matrices against the flap
contract of mkmmd.core.butterfly, the options. The geometry-nodes tree computing the same is checked in
tests/test_butterflies_build.py."""
import math

import numpy as np
import pytest

from mkmmd.core import butterflies as BS
from mkmmd.core import butterfly as BF


def swarm(**slots):
    opts, P = BS.options(slots)
    return opts, P, BS.layout(opts)


def at(L, P, t=0.0, beats=0.0, turns=0.0, **live):
    """The numpy pose at clip second t with the clocks and any live property overridden."""
    return BS.pose(L, BS.inputs(dict(P, **live), t, beats, turns))


def smoothstep(a, b, x):
    f = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return f * f * (3.0 - 2.0 * f)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# ================================================================================================= layout
def test_the_layout_is_the_seed_s_and_the_seed_moves_only_what_is_random():
    _, _, A = swarm(seed=3)
    _, _, B = swarm(seed=3)
    _, _, C = swarm(seed=4)
    for k in BS.ATTRS:
        assert np.array_equal(A.attrs[k], B.attrs[k]), k
    assert A.style == B.style and np.array_equal(A.way, B.way)
    for k in ("rad", "depth", "theta0", "rdir", "ring_f", "sgn"):             # the rings are geometry, not chance
        assert np.array_equal(A.attrs[k], C.attrs[k]), k
    for k in ("poff", "rate", "span", "ophi0", "orho", "anchor", "jit", "eps", "speed"):
        assert not np.allclose(A.attrs[k], C.attrs[k]), k


@pytest.mark.parametrize("count, want", [(48, [8, 16, 24]), (100, [8, 12, 20, 28, 32]), (42, [7, 14, 21]),
                                         (1, [1]), (9, [3, 6])])
def test_a_count_makes_rings_that_grow_outward(count, want):
    assert BS.counts_for(count) == want


def test_every_count_is_kept_and_no_ring_is_empty():
    for count in range(1, BS.MAX_COUNT + 1):
        c = BS.counts_for(count)
        assert sum(c) == count and min(c) >= 1 and c == sorted(c), (count, c)


def test_rings_have_their_counts_radii_cup_and_interleave():
    _, P, L = swarm(ring_counts=[6, 6, 6], inner=0.4, ring_gap=0.2, cup=0.3)
    p = at(L, P)["pos"]
    ys, angles = [], []
    for k in range(3):
        m = L.ring == k
        assert m.sum() == 6
        assert np.hypot(p[m, 0], p[m, 2]) == pytest.approx(np.full(6, 0.4 + 0.2 * k), abs=1e-6)
        ys.append(p[m, 1])
        angles.append(np.sort(np.mod(np.arctan2(p[m, 2], p[m, 0]), 2 * np.pi)))
        assert np.diff(angles[-1]) == pytest.approx(np.full(5, np.pi / 3), abs=1e-6)       # evenly round the ring
    assert ys[0] == pytest.approx(np.zeros(6), abs=1e-6) and ys[2] == pytest.approx(np.full(6, 0.3), abs=1e-6)
    assert 0.0 < ys[1][0] < 0.3                                                # a bowl opening toward the camera (-Y)
    for k in (1, 2):                                                           # half a slot off the ring inside it
        off = np.mod(angles[k][0] - angles[k - 1][0], np.pi / 3)
        assert off == pytest.approx(np.pi / 6, abs=1e-6), k
    assert np.abs(p[L.ring == 0, 2]).max() == pytest.approx(0.4, abs=1e-6)   # a butterfly on top: the mandala stands upright


def test_the_rings_turn_in_alternate_directions_by_turns_the_inner_ones_faster():
    _, P, L = swarm(inner=0.5, ring_gap=0.3)
    a0, a1 = (np.arctan2(at(L, P, turns=tn)["pos"][:, 2], at(L, P, turns=tn)["pos"][:, 0]) for tn in (0.0, 0.01))
    d = wrap(a1 - a0)
    speed = []
    for k in range(3):
        m = L.ring == k
        want = (-1) ** k * 2 * np.pi * 0.01 * (0.5 / (0.5 + 0.3 * k)) ** BS.RING_SPEED
        assert d[m] == pytest.approx(np.full(m.sum(), want), abs=1e-6), k
        speed.append(abs(want))
    assert speed[0] > speed[1] > speed[2]


# ================================================================================================= frames
def _outward(p):
    o = np.stack([p[:, 0], np.zeros(len(p)), p[:, 2]], 1)
    return o / np.linalg.norm(o, axis=1, keepdims=True)


def test_frames_are_rotations_with_the_head_along_minus_y_and_the_back_along_z():
    _, P, L = swarm()
    R = at(L, P, t=0.7, beats=1.3, turns=0.2, rings=0.5, orbit=0.3, drift=0.4, radial=0.3, burst=0.1)
    rot = R["rot"]
    assert np.abs(np.einsum("nij,nik->njk", rot, rot) - np.eye(3)).max() < 1e-9
    assert np.linalg.det(rot) == pytest.approx(np.ones(L.N), abs=1e-9)
    h = R["head"] / np.linalg.norm(R["head"], axis=1, keepdims=True)
    assert -rot[:, :, 1] == pytest.approx(h, abs=1e-9)                        # the head exact
    b = R["back"] - np.einsum("ni,ni->n", R["back"], h)[:, None] * h
    assert rot[:, :, 2] == pytest.approx(b / np.linalg.norm(b, axis=1, keepdims=True), abs=1e-9)   # the back, made square


def test_petals_point_outward_with_their_backs_to_the_camera():
    _, P, L = swarm(heading=0.0)
    R = at(L, P, turns=0.13)
    assert R["head"] == pytest.approx(_outward(R["pos"]), abs=1e-6)
    assert R["rot"][:, :, 2] == pytest.approx(np.tile([0.0, -1.0, 0.0], (L.N, 1)), abs=1e-6)


def test_heading_90_flies_along_each_ring_its_own_way():
    _, P, L = swarm(heading=90.0)
    R0, R1 = at(L, P, turns=0.2), at(L, P, turns=0.2001)
    motion = R1["pos"] - R0["pos"]
    motion /= np.linalg.norm(motion, axis=1, keepdims=True)
    assert R0["head"] == pytest.approx(motion, abs=1e-3)


@pytest.mark.parametrize("radial", [0.5, 1.0])
def test_radial_turns_each_butterfly_about_its_head(radial):
    _, P, L = swarm()
    flat, R = at(L, P, turns=0.3), at(L, P, turns=0.3, radial=radial)
    assert R["head"] == pytest.approx(flat["head"], abs=1e-9)                 # the head axis stays
    tilt = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", R["rot"][:, :, 2], flat["rot"][:, :, 2]), -1, 1)))
    assert tilt == pytest.approx(np.full(L.N, 90.0 * radial), abs=1e-6)
    if radial == 1.0:                       # wings radial: the wing plane holds the mandala's axis (Y)
        assert np.abs(R["rot"][:, 1, 2]).max() < 1e-6 and np.abs(np.abs(R["rot"][:, 1, 0]) - 1).max() < 1e-6


# ================================================================================================= orbit, drift
def test_orbits_keep_their_radius_tilt_and_sense_and_bank_toward_the_centre():
    centre, tilt = np.array([0.1, 0.2, 0.3]), 20.0
    _, P, L = swarm(formation="orbit", orbit_at=list(centre), orbit_tilt=tilt, count=60)
    n = L.attrs["onrm"]
    assert np.degrees(np.arccos(n[:, 2])).max() <= tilt + 1e-4
    assert np.degrees(np.arccos(n[:, 2])).max() > 0.8 * tilt                  # the inclinations do use the range
    for turns in (0.0, 0.37, 1.9):
        R = at(L, P, turns=turns, orbit_r=0.4)
        d = R["pos"] - centre
        r = np.linalg.norm(d, axis=1)
        assert r == pytest.approx(0.4 * L.attrs["orho"], rel=1e-6)
        assert r.min() >= 0.4 * (1 - BS.ORBIT_VAR) - 1e-6 and r.max() <= 0.4 * (1 + BS.ORBIT_VAR) + 1e-6
        assert np.abs(np.einsum("ni,ni->n", d, n)).max() < 1e-6                # in its own plane
        h = R["head"]
        assert np.abs(np.einsum("ni,ni->n", h, d)).max() < 1e-6                # along the orbit
        assert (np.einsum("ni,ni->n", np.cross(d, h), n) > 0).all()             # all the same way round
        back = R["rot"][:, :, 2]
        bank = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", back, n), -1, 1)))
        assert bank == pytest.approx(np.full(L.N, BS.BANK), abs=1e-4)
        assert (np.einsum("ni,ni->n", back, d) < 0).all()                        # banked toward the centre
    nodes = np.sort(np.mod(np.arctan2(L.attrs["oe1"][:, 1], L.attrs["oe1"][:, 0]), 2 * np.pi))
    assert np.diff(np.concatenate([nodes, [nodes[0] + 2 * np.pi]])).max() < 3 * 2 * np.pi / L.N   # spread round


def test_the_flock_wanders_inside_its_ellipsoid_heads_along_its_way_backs_up():
    _, P, L = swarm(formation="drift", drift_r=0.8, count=80)
    axes = 0.8 * np.array(BS.DRIFT_AXES)
    hz = L.attrs["dw1"] / (2 * np.pi)
    assert hz.min() >= BS.DRIFT_HZ[0] - 1e-6 and hz.max() <= BS.DRIFT_HZ[1] + 1e-6
    far = 0.0
    for t in np.linspace(0.0, 120.0, 241):
        R = at(L, P, t=t)
        q = np.linalg.norm(R["pos"] / axes, axis=1)
        assert q.max() <= 1.0 + 1e-6, t
        far = max(far, q.max())
        if t < 30:
            ahead = at(L, P, t=t + 1e-4)["pos"] - at(L, P, t=t - 1e-4)["pos"]
            ahead /= np.linalg.norm(ahead, axis=1, keepdims=True)
            assert R["head"] == pytest.approx(ahead, abs=1e-4)
            assert (R["rot"][:, 2, 2] > 0).all()                               # the back up
    assert far > 0.6                                                           # and it does wander out


# ================================================================================================= burst, bloom
def _way(R0, L):
    o = R0["pos"] + L.attrs["eps"]
    o = o / np.linalg.norm(o, axis=1, keepdims=True) + L.attrs["jit"] + [0.0, 0.0, BS.BURST_RISE]
    return o / np.linalg.norm(o, axis=1, keepdims=True)


@pytest.mark.parametrize("formation", ["rings", "orbit", "drift"])
def test_a_burst_flies_them_out_along_their_ways_and_back_and_is_gone_at_1(formation):
    _, P, L = swarm(formation=formation)
    kw = dict(t=1.3, beats=2.0, turns=0.3, burst_reach=2.5)
    R0 = at(L, P, **kw)
    assert at(L, P, burst=0.0, **kw)["pos"] == pytest.approx(R0["pos"], abs=1e-12)
    way = _way(R0, L)
    for s in (0.1, 0.5, 0.8):
        R = at(L, P, burst=s, **kw)
        d = R["pos"] - R0["pos"]
        assert d == pytest.approx(way * (2.5 * L.attrs["speed"] * s * s)[:, None], abs=1e-9)   # straight out, s^2
        assert R["phase"] - R0["phase"] == pytest.approx(np.full(L.N, P["burst_flaps"] * s), abs=1e-9)
        if s >= BS.BURST_TURN:
            assert R["head"] == pytest.approx(way, abs=1e-9)                   # turned to their ways
    assert at(L, P, burst=0.85, **kw)["span"] == pytest.approx(R0["span"], abs=1e-12)
    fade = 1.0 - smoothstep(*BS.BURST_FADE, 0.95)
    assert at(L, P, burst=0.95, **kw)["span"] == pytest.approx(np.maximum(fade * R0["span"], BS.MIN_SCALE), abs=1e-12)
    assert fade < 0.3                                                           # nearly gone by then
    assert at(L, P, burst=1.0, **kw)["span"] == pytest.approx(np.full(L.N, BS.MIN_SCALE))    # gone, never scale 0


def test_bloom_0_hides_them_all_at_the_root_and_the_rings_grow_in_inner_first():
    _, P, L = swarm()
    R = at(L, P, bloom=0.0)
    assert R["span"] == pytest.approx(np.full(L.N, BS.MIN_SCALE)) and np.abs(R["pos"]).max() < 1e-12
    full = at(L, P, bloom=1.0)
    assert full["span"] == pytest.approx(P["size"] * L.attrs["span"])
    reach = np.linalg.norm(full["pos"], axis=1)
    last = None
    for b in np.linspace(0.0, 1.0, 41):
        R = at(L, P, bloom=b)
        g = np.linalg.norm(R["pos"], axis=1) / reach                           # how far out, as a share
        assert R["span"] == pytest.approx(np.maximum(g * full["span"], BS.MIN_SCALE), abs=1e-12)   # and as big
        per = [g[L.ring == k].max() for k in range(L.K)]
        assert all(g[L.ring == k].min() == pytest.approx(per[k], abs=1e-12) for k in range(L.K))   # a ring as one
        assert per == sorted(per, reverse=True), b                             # inner rings first
        if last is not None:
            assert all(p >= q - 1e-12 for p, q in zip(per, last))
        last = per
    assert last == pytest.approx([1.0] * L.K)
    mid = at(L, P, bloom=0.5)
    grow = smoothstep(L.attrs["blo"], L.attrs["blo"] + BS.BLOOM_W, 0.5)
    a_full = np.arctan2(at(L, P)["pos"][:, 2], at(L, P)["pos"][:, 0])
    a_mid = np.arctan2(mid["pos"][:, 2], mid["pos"][:, 0])
    m = grow > 0.05
    assert wrap(a_mid - a_full)[m] == pytest.approx((-BS.SPIRAL * L.attrs["sgn"] * (1 - grow))[m], abs=1e-6)  # a spiral


# ================================================================================================= wing beats
def test_phases_and_rates_spread_and_spread_0_flaps_in_unison():
    _, P, L = swarm(count=200)
    off = np.sort(L.attrs["poff"])
    assert np.diff(np.concatenate([off, [off[0] + 1]])).max() < 0.08          # all over the beat
    assert abs(np.exp(2j * np.pi * off).mean()) < 0.15
    style_rate = np.array([BF.RATE_FACTOR[s] for s in L.style])
    k = L.attrs["rate"] / style_rate
    assert k.min() >= 1 - BS.RATE_VAR - 1e-6 and k.max() <= 1 + BS.RATE_VAR + 1e-6 and k.std() > 0.05
    _, P0, U = swarm(count=200, phase_spread=0.0)
    for beats, t in ((0.0, 0.0), (7.3, 2.4)):
        ph = at(U, P0, t=t, beats=beats)["phase"]
        assert np.ptp(ph) < 1e-9                                               # unison
    ph = at(L, P, t=2.4, beats=7.3)["phase"]
    assert np.ptp(np.mod(ph, 1.0)) > 0.9


def test_a_hit_s_extra_beat_ripples_out_ring_by_ring():
    _, P, L = swarm(ring_counts=[8, 8, 8], ripple=0.3, hit_time=0.2)
    hit = 1.0
    for t in np.linspace(0.9, 1.7, 33):
        quiet = at(L, P, t=t, beats=3 * t)["phase"]
        R = at(L, P, t=t, beats=3 * t, hit=hit)["phase"]
        extra = R - quiet + 1.0                                                # the default hit (-1000) has long added its beat
        for k in range(3):
            want = smoothstep(0.0, 1.0, (t - hit - 0.3 * k / 2) / 0.2)
            assert extra[L.ring == k] == pytest.approx(np.full(8, want), abs=1e-9), (t, k)
    starts = [hit + 0.3 * k / 2 for k in range(3)]
    assert starts == sorted(starts) and starts[0] == hit                       # the inner ring first, on the hit


# ================================================================================================= blend
def test_weights_are_normalised_continuous_and_nothing_means_rings():
    _, P, L = swarm()
    kw = dict(t=1.1, turns=0.4)
    rings, orbit = at(L, P, rings=1.0, **kw), at(L, P, rings=0.0, orbit=1.0, **kw)
    assert at(L, P, rings=3.0, **kw)["pos"] == pytest.approx(rings["pos"], abs=1e-12)
    half = at(L, P, rings=2.0, orbit=2.0, **kw)
    assert half["pos"] == pytest.approx(0.5 * (rings["pos"] + orbit["pos"]), abs=1e-12)
    assert at(L, P, rings=0.0, orbit=0.0, drift=0.0, **kw)["pos"] == pytest.approx(rings["pos"], abs=1e-12)
    assert at(L, P, rings=0.0, orbit=0.0, drift=1e-7, **kw)["pos"] == pytest.approx(rings["pos"], abs=1e-3)
    steps = [at(L, P, rings=1.0 - w, orbit=w, **kw)["pos"] for w in np.linspace(0, 1, 201)]
    assert max(np.abs(b - a).max() for a, b in zip(steps, steps[1:])) < 0.03   # no jump on the way


# ================================================================================================= the flap contract
def test_the_wings_turn_about_their_hinges_by_the_flap_and_the_body_bobs():
    _, P, L = swarm()
    R = at(L, P, t=0.4, beats=1.37, turns=0.1, flap_amp=90.0, flap_mid=10.0)
    assert R["angle"] == pytest.approx(BF.wing_angle(R["phase"], 90.0, 10.0), abs=1e-12)
    assert R["bob"] == pytest.approx(BF.body_bob(R["phase"]), abs=1e-12)
    span, rot = R["span"], R["rot"]
    assert R["body"][:, :3, 3] == pytest.approx(R["pos"] + rot[:, :, 2] * (span * R["bob"])[:, None], abs=1e-12)
    for part, side, sign in (("wing_L", "L", -1.0), ("wing_R", "R", 1.0)):
        rel = np.linalg.inv(R["body"]) @ R[part]
        c, s = np.cos(sign * R["angle"]), np.sin(sign * R["angle"])
        assert rel[:, :3, 3] == pytest.approx(np.tile(BF.hinge(side), (L.N, 1)), abs=1e-9)
        assert rel[:, 0, 0] == pytest.approx(c, abs=1e-9) and rel[:, 0, 2] == pytest.approx(s, abs=1e-9)
        tip = R[part] @ np.array([0.5 if side == "L" else -0.5, 0.0, 0.0, 1.0])
        up = np.einsum("ni,ni->n", tip[:, :3] - R[part][:, :3, 3], rot[:, :, 2]) / span
        assert up == pytest.approx(0.5 * np.sin(R["angle"]), abs=1e-9)        # a positive angle lifts both tips


# ================================================================================================= styles
def test_styles_follow_the_mix_and_repeat_round_a_symmetric_mandala():
    _, _, L = swarm(styles={"monarch": 2.0, "blue": 1.0})
    assert L.sym == 8                                                          # one sector: 1 + 2 + 3 slots
    n = {s: L.style.count(s) for s in set(L.style)}
    assert n == {"monarch": 32, "blue": 16}
    for k, c in enumerate(L.counts):
        idx = np.flatnonzero(L.ring == k)
        step = c // L.sym
        for j in range(c - step):
            a, b = idx[j], idx[j + step]
            assert L.style[a] == L.style[b] and L.way[a] == L.way[b]
    assert all(0 <= w < len(BF.WAYS[s]) for s, w in zip(L.style, L.way))
    assert [L.combos[c] for c in L.combo] == [(s, int(w)) for s, w in zip(L.style, L.way)]
    assert len(set(L.combos)) == len(L.combos)


# ================================================================================================= numpy semantics
def test_the_numpy_ops_divide_and_normalise_as_blender_does():
    o = BS.NP
    assert o.div(np.array([1.0, 2.0]), np.array([0.0, 4.0])).tolist() == [0.0, 0.5]
    assert o.normalize(o.vec(np.zeros(2), np.zeros(2), np.zeros(2))).tolist() == [[0.0, 0.0]] * 3
    assert o.smoothstep(0.3, 0.3, np.array([0.1, 0.5])).tolist() == [0.0, 0.0]
    assert o.sqrt(np.array([-1.0, 4.0])).tolist() == [0.0, 2.0]


def test_bounds_hold_every_formation_but_not_the_burst():
    opts, P, L = swarm(orbit_at=[0.2, 0.0, 0.5], cup=-0.2)
    lo, hi = np.array(BS.bounds(opts, P, L)[0]), np.array(BS.bounds(opts, P, L)[1])
    pad = 0.5 * P["size"] * L.attrs["span"].max()
    for f in BS.FORMATIONS:
        live = {g: float(g == f) for g in BS.FORMATIONS}
        for t in np.linspace(0, 40, 21):
            p = at(L, P, t=t, turns=0.05 * t, **live)["pos"]
            assert (p >= lo + pad - 1e-9).all() and (p <= hi - pad + 1e-9).all(), (f, t)
    assert (at(L, P, burst=0.7)["pos"] > hi).any()


# ================================================================================================= options
@pytest.mark.parametrize("slots, frag", [
    ({"ring_counts": [8, 0]}, "ring_counts is a list"),
    ({"ring_counts": [8] * 13}, "ring_counts is a list"),
    ({"count": 0}, "count is a whole number"),
    ({"count": 48, "ring_counts": [8, 16]}, "disagrees"),
    ({"inner": 0}, "inner is"),
    ({"formation": "swirl"}, "formation is one of"),
    ({"styles": {"moth": 1}}, "styles is"),
    ({"styles": {"monarch": 0}}, "styles is"),
    ({"seed": 1.5}, "seed is"),
    ({"size_var": 1.0}, "size_var is"),
    ({"phase_spread": 2}, "phase_spread is"),
    ({"orbit_at": [0, 0]}, "orbit_at is"),
    ({"orbit_tilt": 120}, "orbit_tilt is"),
    ({"drift_r": -1}, "drift_r is"),
    ({"level": "yes"}, "level is"),
    ({"burst": 1.5}, "burst is"),
    ({"rings": -1}, "rings is a weight"),
    ({"size": 0}, "size is"),
    ({"hit_time": 0}, "hit_time is"),
    ({"flap": "fast"}, "flap is a number"),
])
def test_a_bad_option_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        BS.options(slots)


def test_the_formation_sets_the_starting_weights_and_slots_override_them():
    assert {k: BS.options({"formation": "drift"})[1][k] for k in BS.FORMATIONS} == {"rings": 0, "orbit": 0, "drift": 1}
    assert BS.options({"formation": "orbit", "rings": 0.5})[1]["rings"] == 0.5
    opts, _ = BS.options({"count": 100, "level": True, "rose": "#ffffff"})          # a palette slot is not ours
    assert opts["ring_counts"] == [8, 12, 20, 28, 32] and opts["level"] is True
    assert math.isclose(BS.options({})[1]["flap"], BF.FLAP_PARAMS["flap"][0])
