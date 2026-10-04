import argparse
import functools
import json
import math

import numpy as np
import pytest

from mkmmd.cli import grip as cli
from mkmmd.core import gripframe as GF
from mkmmd.solvers import geom
from mkmmd.solvers import grip as G


# ---------------------------------------------------------------- a synthetic hand (no Blender)
# Canonical right hand: wrist at the origin, fingers along +Y, palm facing -Z, thumb on the -X side; the left hand is
# the mirror image in X; both are then rotated and moved so nothing is axis-aligned. Capsule fingers, a boxy palm and
# a static cuff.
SEG = {"thumb": (0.020, 0.020, 0.036), "index": (0.025, 0.018, 0.033), "middle": (0.028, 0.022, 0.035),
       "ring": (0.026, 0.019, 0.033), "little": (0.018, 0.017, 0.030)}
RAD = {"thumb": 0.0085, "index": 0.0075, "middle": 0.0078, "ring": 0.0072, "little": 0.0065}
KNUCKLE_X = {"index": -0.027, "middle": -0.009, "ring": 0.009, "little": 0.027}
PALM = 0.076


def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _placement():
    T = np.eye(4)
    T[:3, :3] = _rot([0, 0, 1], 70.0) @ _rot([0, 1, 0], -20.0) @ _rot([1, 0, 0], 30.0)
    T[:3, 3] = (-0.4, 0.05, 1.0)
    return T


def _frame_from_y(d):
    y = d / np.linalg.norm(d)
    ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, ref)
    x /= np.linalg.norm(x)
    return np.stack([x, y, np.cross(x, y)], 1)


def synthetic_hand(side="R", ring_pts=10, rings=7):
    T = _placement()
    mirror = np.diag([1.0, 1.0, 1.0]) if side == "R" else np.diag([-1.0, 1.0, 1.0])
    heads, tails, parents, dirs, radius = [np.zeros(3)], [np.array([0, 0.03, 0])], [-1], [np.array([0, 1.0, 0])], [0.0]
    chain_of = {}
    for f in G.FINGERS:
        if f == "thumb":
            base, d = np.array([-0.018, 0.016, -0.004]), np.array([-0.55, 0.75, -0.35])
        else:
            base, d = np.array([KNUCKLE_X[f], PALM, 0.0]), np.array([0.0, 1.0, 0.0])
        d = d / np.linalg.norm(d)
        idx, p, prev = [], base.copy(), 0
        for j in range(3):
            heads.append(p.copy())
            tails.append(p + d * SEG[f][j])
            dirs.append(d.copy())
            radius.append(RAD[f])
            parents.append(prev)
            prev = len(heads) - 1
            idx.append(prev)
            p = p + d * SEG[f][j]
        heads.append(p.copy())                                  # the tip bone: head at the fingertip
        tails.append(p + d * 0.01)
        dirs.append(d.copy())
        radius.append(RAD[f] * 0.9)
        parents.append(prev)
        idx.append(len(heads) - 1)
        chain_of[f] = idx
    heads, tails, dirs = np.array(heads), np.array(tails), np.array(dirs)
    n = len(heads)
    verts, weights = [], []

    def add(pts, w):
        for q in pts:
            verts.append(q)
            row = np.zeros(n + 1)
            for c, v in w.items():
                row[c] += v
            weights.append(row)

    for i in range(1, n):                                       # finger skin: rings round each bone
        if i % 4 == 0:                                          # tip bones carry no skin of their own
            continue
        a, b = heads[i], heads[i + 1]
        ax = (b - a) / np.linalg.norm(b - a)
        e1 = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        is_last = (i % 4 == 3)
        for r in range(rings):
            s = r / (rings - 1)
            rad = radius[i] * (0.8 if is_last and s > 0.9 else 1.0)
            ring = [np.cos(2 * np.pi * k / ring_pts) * e1 + np.sin(2 * np.pi * k / ring_pts) * e2
                    for k in range(ring_pts)]
            pts = [a + (b - a) * s + rad * c for c in ring]
            blend = 0.5 if (not is_last and s > 0.85) else 0.0
            parent_blend = 0.5 if (s < 0.15 and parents[i] > 0) else 0.0
            w = {i: 1.0 - blend - parent_blend}
            if blend:
                w[i + 1] = blend
            if parent_blend:
                w[parents[i]] = parent_blend
            add(pts, w)
    for z in (-0.012, 0.0, 0.012):                              # palm sheets, blended into the finger bases
        for y in np.linspace(0.0, PALM, 10):
            for x in np.linspace(-0.037, 0.037, 12):
                if z == 0.0 and abs(x) < 0.034:
                    continue
                w = {0: 1.0}
                if y > PALM - 0.006:
                    fb = min(G.FOUR, key=lambda f: abs(KNUCKLE_X[f] - x))
                    w = {0: 0.5, chain_of[fb][0]: 0.5}
                add([np.array([x, y, z])], w)
    for y in np.linspace(-0.05, -0.005, 5):                     # forearm cuff: static
        for k in range(14):
            a = 2 * np.pi * k / 14
            add([np.array([0.026 * np.cos(a), y, 0.022 * np.sin(a)])], {n: 1.0})
    V, W = np.array(verts), np.array(weights)
    heads, tails, V, dirs = heads @ mirror.T, tails @ mirror.T, V @ mirror.T, dirs @ mirror.T
    rest = np.array([_frame_from_y(d) for d in dirs])
    R0, t0 = T[:3, :3], T[:3, 3]
    heads, tails, V = heads @ R0.T + t0, tails @ R0.T + t0, V @ R0.T + t0
    rest = np.einsum("ij,njk->nik", R0, rest)
    names = [f"bone{i}.{side}" for i in range(n)]
    names[0] = f"wrist.{side}"
    arm = np.array([[0, -0.30, 0.05], [0, -0.14, 0.0], [0, 0, 0]], float) @ mirror.T @ R0.T + t0
    return G.HandModel(side, names, parents, heads, tails, rest, V, W, arm_chain=arm)


HANDS = {s: synthetic_hand(s) for s in "RL"}


def moved(hand, turns=((0.0, 0.0, 1.0, 70.0), (0.0, 1.0, 0.0, -25.0), (1.0, 0.0, 0.0, 40.0)), shift=(1.5, -2.0, 0.4)):
    """The same hand model rigidly turned and moved somewhere else in the world."""
    R = np.eye(3)
    for x, y, z, deg in turns:
        R = R @ G._rotm([x, y, z], deg)
    a = G._hand_args(hand)
    for key in ("heads", "tails", "V", "arm_chain"):
        a[key] = np.asarray(a[key]) @ R.T + np.asarray(shift)
    a["rest_rot"] = np.einsum("ij,njk->nik", R, a["rest_rot"])
    return G.HandModel(**a)


@pytest.fixture(scope="module", params=["R", "L"])
def hand(request):
    return synthetic_hand(request.param)


def _angles(**kw):
    deg = np.zeros(21)
    for k, v in kw.items():
        deg[G.DOF.index(k)] = v
    return deg


# ---------------------------------------------------------------- the hand model: kinematics and skinning
def test_rest_pose_leaves_bones_and_skin_in_place(hand):
    Q, D, H, P = hand.pose(np.zeros(21))
    assert np.allclose(P, hand.V, atol=1e-12) and np.allclose(H, hand.h, atol=1e-12)


def test_links_stay_rigid_and_skin_follows_its_bone(hand):
    rng = np.random.default_rng(3)
    Q, D, H, P = hand.pose(rng.uniform(-30, 60, 21))
    for i, p in enumerate(hand.parents):
        if p >= 0:
            assert np.linalg.norm(H[i] - H[p]) == pytest.approx(np.linalg.norm(hand.h[i] - hand.h[p]), abs=1e-12)
    for b in range(1, hand.nb):
        rows = np.where(hand.W[:, b] == 1.0)[0]               # skin weighted fully to one bone moves rigidly with it
        if len(rows):
            assert np.allclose(P[rows], H[b] + (hand.V[rows] - hand.h[b]) @ D[b].T, atol=1e-12)


def test_blended_vertices_average_their_bones_transforms(hand):
    Q, D, H, P = hand.pose(_angles(index_mcp=50.0, index_pip=40.0))
    v = next(r for r in range(len(hand.V)) if hand.W[r, hand.F["index"][0]] == 0.5 and hand.W[r, 0] == 0.5)
    a = hand.V[v]
    parts = [H[b] + D[b] @ (a - hand.h[b]) for b in (0, hand.F["index"][0])]
    assert np.allclose(P[v], 0.5 * parts[0] + 0.5 * parts[1], atol=1e-12)


def test_positive_angles_curl_toward_the_palm_and_spread_toward_the_thumb(hand):
    tip = hand.F["index"][3]
    _, _, H0, _ = hand.pose(np.zeros(21))
    for f in G.FOUR:
        tip = hand.F[f][3]
        _, _, H, _ = hand.pose(_angles(**{f + "_mcp": 40.0}))
        assert (H[tip] - H0[tip]) @ hand.ventral > 0.015           # the tip moved toward the palm side
        _, _, H, _ = hand.pose(_angles(**{f + "_spr": 15.0}))
        assert (H[tip] - H0[tip]) @ hand.radial > 0.005            # and spreading swings it toward the thumb
    assert hand.ventral @ (hand.h[hand.F["thumb"][3]] - hand.palm_c) > 0        # the thumb lies on the palm side


def test_local_rotations_are_relative_to_the_bones_own_axes(hand):
    i = hand.F["index"][1]
    R = hand.rest_rot[i]
    for col, expect in ((1, [math.cos(math.radians(15)), 0, math.sin(math.radians(15)), 0]),
                        (0, [math.cos(math.radians(15)), math.sin(math.radians(15)), 0, 0])):
        Q = np.broadcast_to(np.eye(3), (hand.nb, 3, 3)).copy()
        Q[i] = G._rotm(R[:, col], 30.0)                        # 30 deg about the bone's own x or y axis
        assert np.allclose(hand.local_quats(Q)[hand.names[i]], expect, atol=1e-9)
    Q = hand.rotations(np.random.default_rng(5).uniform(-20, 60, 21))
    assert np.allclose(hand.rotations_from_local(hand.local_quats(Q)), Q, atol=1e-9)
    assert set(hand.local_quats(Q)) == {hand.names[b] for f in G.FINGERS for b in hand.F[f][:3]}


def test_frame_in_wrist_is_the_frame_in_the_wrist_bones_rest_frame(hand):
    Bw, h0 = hand.rest_rot[0], hand.h[0]
    assert np.allclose(hand.frame_in_wrist(Bw, h0), np.eye(4), atol=1e-12)
    R = G._rotm([1, 2, 3], 40.0) @ Bw
    p = h0 + Bw @ np.array([0.01, -0.02, 0.03])
    T = hand.frame_in_wrist(R, p)
    assert np.allclose(T[:3, 3], [0.01, -0.02, 0.03], atol=1e-12)
    assert np.allclose(Bw @ T[:3, :3], R, atol=1e-12) and np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_hand_model_file_roundtrip(hand, tmp_path):
    path = tmp_path / "hand.npz"
    np.savez(path, side=np.array(hand.side), names=np.array(hand.names), sem=np.array(hand.sem), parents=hand.parents,
             heads=hand.h, tails=hand.tails, rest_rot=hand.rest_rot, V=hand.V, W=hand.W, arm_world=hand.arm_world,
             arm_chain=hand.arm_chain)
    again = G.HandModel.from_npz(path)
    assert again.digest() == hand.digest() and again.side == hand.side and np.allclose(again.arm_chain, hand.arm_chain)


# ---------------------------------------------------------------- shared helpers
def test_radius_profile_of_a_pen():
    r = G._radius_fn(0.006, tip=0.0155)
    assert r(0.0) == pytest.approx(0.15 * 0.006) and r(0.0155) == pytest.approx(0.006)
    assert r(0.1) == pytest.approx(0.006)
    assert G._radius_fn(0.006)(0.009) == pytest.approx(0.5 * 0.006)          # default tip = 3 radius
    prof = G._radius_fn([[0.0, 0.001], [0.01, 0.005], [0.05, 0.005]])
    assert prof(0.005) == pytest.approx(0.003) and prof(0.2) == pytest.approx(0.005)


def test_nib_offset_forms():
    assert np.array_equal(G._nib_offset(None), np.zeros(3))
    assert np.array_equal(G._nib_offset(0.01), [0.0, 0.0, 0.01])
    assert np.array_equal(G._nib_offset([0.001, 0.002, 0.003]), [0.001, 0.002, 0.003])


def test_run_seeds_in_processes_matches_in_process_and_restores_the_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    assert G._run_seeds(math.sqrt, [4.0, 9.0, 16.0], workers=2) == [2.0, 3.0, 4.0]
    assert G._run_seeds(math.sqrt, [4.0, 9.0, 16.0], workers=1) == [2.0, 3.0, 4.0]
    import os
    assert os.environ["OMP_NUM_THREADS"] == "7" and "OPENBLAS_NUM_THREADS" not in os.environ


@pytest.mark.parametrize("style,prop,kw", [
    ("pen", {"length": 0.14, "radius": 0.0068, "tip": 0.0155}, {"nfev": (6, 6, 6, 0)}),
    ("wheel", {"radius": 0.19, "tube": 0.015}, {"nfev": (8, 4)}),
    ("pinch", {"width": 0.011}, {"nfev": (25, 25, 15)}),
    ("rest", {"surface": "plane"}, {"nfev": (10, 10, 10)})])
def test_every_style_gives_the_same_answer_in_processes_as_in_this_process(style, prop, kw):
    one = G.solve(style, HANDS["R"], prop, seeds=2, workers=1, **kw)
    two = G.solve(style, HANDS["R"], prop, seeds=2, workers=2, **kw)
    assert one["solver"]["cost"] == pytest.approx(two["solver"]["cost"], rel=1e-9)
    assert one["bones"].keys() == two["bones"].keys()
    for k, q in one["bones"].items():
        assert np.allclose(q, two["bones"][k], atol=1e-9)
    assert np.allclose(one["target_in_wrist"], two["target_in_wrist"], atol=1e-9)


def test_unknown_style_lists_the_styles(hand):
    with pytest.raises(ValueError, match="pen, wheel, pinch, rest"):
        G.solve("spoon", hand, {})


# ---------------------------------------------------------------- pen: writing posture and mirror symmetry
def _posture_hand_state(hand, outward_deg=0.0):
    """A pen along the horizontal direction to the shoulder, turned `outward_deg` toward the hand's outward side."""
    post = G._Posture(hand, {"nib": [0.0, -0.3, 0.74], "shoulder": [0.0, 0.0, 0.93], "pole": [-0.5, 0.2, 0.8],
                             "upper": 0.21, "fore": 0.17})
    to_shoulder = np.array([0.0, 1.0, 0.0])
    to_pen = math.cos(math.radians(outward_deg)) * to_shoulder + math.sin(math.radians(outward_deg)) * post.outward
    u = math.cos(math.radians(50)) * to_pen + math.sin(math.radians(50)) * post.up
    A = {"p": np.array([0.0, -0.3, 0.74]), "u": u, "P": hand.V}
    return post, A


def test_writing_posture_reads_elevation_and_azimuth_outward_for_either_hand(hand):
    for out in (-20.0, 0.0, 25.0):
        post, A = _posture_hand_state(hand, out)
        wr = post.writing(A, np.eye(3), hand, np.arange(len(hand.V)))
        assert wr["elevation"] == pytest.approx(50.0, abs=1e-9)
        assert wr["azimuth"] == pytest.approx(out, abs=1e-9)         # outward is positive, mirrored for the left hand


def test_writing_posture_tilt_is_the_back_of_the_hand_turned_outward(hand):
    post, A = _posture_hand_state(hand)
    for tilt in (-15.0, 0.0, 30.0):
        want = math.cos(math.radians(tilt)) * post.up + math.sin(math.radians(tilt)) * post.outward
        G_ = geom.min_rot(hand.dorsal[None], want[None])[0]
        wr = post.writing(A, G_, hand, np.arange(len(hand.V)))
        assert wr["tilt"] == pytest.approx(tilt, abs=1e-6)


def test_writing_posture_needs_a_nib_and_a_shoulder(hand):
    with pytest.raises(ValueError, match="nib"):
        G._Posture(hand, {"shoulder": [0, 0, 1.0]})
    bare = synthetic_hand(hand.side)
    bare.arm_chain = np.full((3, 3), np.nan)
    with pytest.raises(ValueError, match="shoulder"):
        G._Posture(bare, {"nib": [0, 0, 0.7]})
    with pytest.raises(ValueError, match="unknown posture target"):
        G._Posture(hand, {"nib": [0, 0, 0.7], "shoulder": [0, 0, 1.0], "pole": [1, 0, 1.0], "upper": .2, "fore": .2,
                          "target": {"wobble": (1, 1)}})


def test_mirrored_hands_pose_alike():
    """The same joint angles bend a left hand into the mirror image of the right hand's pose (thumb roll included)."""
    right, left = synthetic_hand("R"), synthetic_hand("L")
    rng = np.random.default_rng(11)
    for _ in range(3):
        deg = rng.uniform(-15, 55, 21)
        pr, pl = right.pose(deg)[3], left.pose(deg)[3]
        for ref in (0, 400, 900):
            assert np.allclose(np.linalg.norm(pr - pr[ref], axis=1), np.linalg.norm(pl - pl[ref], axis=1), atol=1e-12)


def test_pen_problem_is_the_same_for_mirrored_hands():
    """Every residual of the pen grip (pads against their slots, web, priors, clashes) agrees for mirrored hands."""
    prop = dict(length=0.14, radius=0.0068, tip=0.0155)
    right, left = G._Pen(synthetic_hand("R"), **prop), G._Pen(synthetic_hand("L"), **prop)
    rng = np.random.default_rng(2)
    for _ in range(3):
        x = np.clip(np.r_[G.PEN_PRIOR + rng.normal(0, 1, 15) * G.PEN_SIGMA * 0.5, rng.normal(0, 5, 5), np.zeros(3)],
                    G.PEN_LO, G.PEN_HI)
        assert np.allclose(right.residuals(x, "grip"), left.residuals(x, "grip"), atol=1e-6)


def test_pen_frame_moves_with_the_nib_offset(hand):
    prop = {"length": 0.14, "radius": 0.0068, "tip": 0.0155}
    a = G.solve_pen(hand, prop, seeds=1, workers=1, nfev=(5, 5, 5, 0))
    b = G.solve_pen(hand, {**prop, "nib_offset": [0.001, 0.002, 0.07]}, seeds=1, workers=1, nfev=(5, 5, 5, 0))
    Ta, Tb = np.array(a["target_in_wrist"]), np.array(b["target_in_wrist"])
    assert np.allclose(Ta[:3, :3], Tb[:3, :3], atol=1e-12)
    assert np.allclose(Tb[:3, 3], Ta[:3, 3] - Ta[:3, :3] @ [0.001, 0.002, 0.07], atol=1e-12)
    assert Ta[:3, :3].shape == (3, 3) and np.linalg.det(Ta[:3, :3]) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- wheel: power grip on a ring
_WHEEL_SOLVED = {}


def wheel_solved(side, R=0.19, r=0.015, approach=90.0, wrap=1, seeds=2, **tuning):
    key = (side, R, r, approach, wrap, seeds, tuple(sorted(tuning.items())))
    if key not in _WHEEL_SOLVED:
        _WHEEL_SOLVED[key] = G.solve_wheel(HANDS[side], {"radius": R, "tube": r}, approach=approach, wrap=wrap,
                                           seeds=seeds, workers=1, **tuning)
    return _WHEEL_SOLVED[key]


def wheel_measure(hand, res, R, r):
    """The result as a build sees it: bones -> posed skin, target_in_wrist -> the ring frame, gaps to the torus."""
    Q = hand.rotations_from_local(res["bones"])
    D, H = hand.fk(Q)
    P = hand.skin(D, H)
    T = np.array(res["target_in_wrist"])
    Rf = hand.rest_rot[0] @ T[:3, :3]
    pf = hand.h[0] + hand.rest_rot[0] @ T[:3, 3]
    Pg = (P - pf) @ Rf
    gap = np.hypot(np.hypot(Pg[:, 0] + R, Pg[:, 1]) - R, Pg[:, 2]) - r
    m = {"Rf": Rf, "Pg": Pg, "H": H, "gap": gap, "penetration": max(0.0, -float(gap.min())) * 1e3,
         "clash": float(G._Clash(hand).depth(P, H).max()) * 1e3, "palm_c": Rf.T @ (hand.palm_c - pf)}
    for f in G.FOUR:
        m[f] = [hand.surface(hand.F[f][j], hand.ventral, 0.0, 1.0, 0.3) for j in range(3)]
    m["thumb"] = hand.surface(hand.F["thumb"][2], hand.pad_t, 0.0, 1.0, 0.3)
    m["palm"] = np.where((hand.dom == 0) & ((hand.V - hand.palm_c) @ hand.ventral > 0.006))[0]
    return m


def wheel_gap_mm(m, idx):
    return float(m["gap"][idx].min()) * 1e3


def wheel_section(m, R, i):
    """Section angle (deg, [0, 360)) and tangent offset along the ring (m) of grip-frame point i."""
    x, y, z = m["Pg"][i]
    return math.degrees(math.atan2(z, math.hypot(x + R, y) - R)) % 360.0, math.atan2(y, x + R) * R


def wheel_holds(hand, res, R, r, finger_tol=1.5, other_tol=2.0):
    m = wheel_measure(hand, res, R, r)
    for f in G.FOUR:
        for j, name in enumerate(("proximal", "middle", "distal")):
            assert -0.5 <= wheel_gap_mm(m, m[f][j]) <= finger_tol, f"{f} {name}: {wheel_gap_mm(m, m[f][j]):.2f} mm"
    assert -0.5 <= wheel_gap_mm(m, m["palm"]) <= other_tol, f"palm: {wheel_gap_mm(m, m['palm']):.2f} mm"
    assert -0.5 <= wheel_gap_mm(m, m["thumb"]) <= other_tol, f"thumb: {wheel_gap_mm(m, m['thumb']):.2f} mm"
    assert m["penetration"] <= 0.5 and m["clash"] <= 0.5
    return m


# ---------------------------------------------------------------- the design frame (no solving)
@pytest.mark.parametrize("side", ["R", "L"])
@pytest.mark.parametrize("approach", [0.0, 90.0, 180.0, 270.0, 33.0])
@pytest.mark.parametrize("wrap", [1, -1])
def test_wheel_design_faces_the_tube_with_fingers_along_the_wrap(side, approach, wrap):
    """At zero deviation the palm normal points at the tube centreline from the approach side, the fingers run along
    the wrap direction and the knuckle line along the ring (index toward -y for a right hand wrapping +1)."""
    hand = HANDS[side]
    w = G._Wheel(hand, 0.19, 0.015, approach, wrap)
    R, p = w.frame(np.zeros(4))
    a = math.radians(approach)
    u, t = np.array([math.cos(a), 0, math.sin(a)]), np.array([-math.sin(a), 0, math.cos(a)])
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12) and np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(R.T @ hand.ventral, -u, atol=1e-12)
    assert np.allclose(R.T @ hand.along, wrap * t, atol=1e-12)
    index_dir = R.T @ (-w.across)                                    # across: index -> little finger
    assert np.allclose(index_dir, -hand.chir * wrap * np.array([0, 1.0, 0]), atol=1e-12)
    c = R.T @ (hand.palm_c - p)                                      # the palm centre: y = 0, on the approach side
    assert np.allclose(c, (0.015 + 0.012) * u, atol=1e-12)


def test_wheel_approach_is_periodic():
    hand = HANDS["R"]
    base = G._Wheel(hand, 0.19, 0.015, 90.0, 1).frame(np.array([1.0, 2.0, -3.0, 4.0]))
    for a in (450.0, -270.0):
        other = G._Wheel(hand, 0.19, 0.015, a, 1).frame(np.array([1.0, 2.0, -3.0, 4.0]))
        assert np.allclose(base[0], other[0], atol=1e-12) and np.allclose(base[1], other[1], atol=1e-12)


def test_wheel_problem_is_the_same_for_mirrored_hands():
    """Every residual (contacts, thumb place, priors, walls, clashes) agrees for mirrored hands at equal parameters."""
    right, left = (G._Wheel(HANDS[s], 0.19, 0.015, 60.0, -1) for s in "RL")
    rng = np.random.default_rng(5)
    for _ in range(4):
        x = np.clip(np.r_[G.WHEEL_PRIOR + rng.normal(0, 8, 21), rng.normal(0, 6, 4)], right.lo, right.hi)
        assert np.allclose(right.residuals(x), left.residuals(x), atol=1e-6)


def test_wheel_problem_is_the_same_under_reflection_through_the_ring_plane():
    """Turning the torus over (z -> -z) maps a right hand at approach a, wrap w onto a left hand at -a, -w."""
    right, left = G._Wheel(HANDS["R"], 0.19, 0.015, 60.0, -1), G._Wheel(HANDS["L"], 0.19, 0.015, -60.0, 1)
    rng = np.random.default_rng(8)
    for _ in range(4):
        x = np.clip(np.r_[G.WHEEL_PRIOR + rng.normal(0, 8, 21), rng.normal(0, 6, 4)], right.lo, right.hi)
        assert np.allclose(right.residuals(x), left.residuals(x), atol=1e-6)


# ---------------------------------------------------------------- the default grip
@pytest.mark.parametrize("side", ["R", "L"])
def test_wheel_default_grip_touches_with_every_phalanx_the_palm_and_the_thumb(side):
    hand, res = HANDS[side], wheel_solved(side)
    m = wheel_holds(hand, res, 0.19, 0.015)
    rep = res["report"]
    for f in G.FOUR:                                                    # the report is what the skin says
        assert rep["contacts"][f]["prox_mm"] == pytest.approx(wheel_gap_mm(m, m[f][0]), abs=0.05)
        assert rep["contacts"][f]["mid_mm"] == pytest.approx(wheel_gap_mm(m, m[f][1]), abs=0.05)
        assert rep["contacts"][f]["dist_mm"] == pytest.approx(wheel_gap_mm(m, m[f][2]), abs=0.05)
        assert rep["contacts"][f]["gap_mm"] == pytest.approx(min(wheel_gap_mm(m, i) for i in m[f]), abs=0.05)
    assert rep["contacts"]["thumb"]["gap_mm"] == pytest.approx(wheel_gap_mm(m, m["thumb"]), abs=0.05)
    assert rep["contacts"]["palm"]["gap_mm"] == pytest.approx(wheel_gap_mm(m, m["palm"]), abs=0.05)
    assert rep["penetration_mm"] == pytest.approx(m["penetration"], abs=0.01)
    assert rep["finger_clash_mm"] == pytest.approx(m["clash"], abs=0.01)


@pytest.mark.parametrize("side", ["R", "L"])
def test_wheel_joint_angles_stay_in_power_grip_bounds(side):
    ang = wheel_solved(side)["report"]["angles_deg"]
    assert list(ang) == G.DOF
    deg = np.array([ang[k] for k in G.DOF])
    assert np.all(deg >= G.WHEEL_LO - 0.1) and np.all(deg <= G.WHEEL_HI + 0.1)
    for f in G.FOUR:                                                # curled round the tube, no hooks, no hyperextension
        assert 0 <= ang[f + "_mcp"] <= 90 and 20 <= ang[f + "_pip"] <= 100 and 5 <= ang[f + "_dip"] <= 70
        assert ang[f + "_dip"] >= 0.25 * ang[f + "_pip"] and abs(ang[f + "_spr"]) <= 12
    assert ang["t_mcp"] >= -10 and ang["t_ip"] >= -10


@pytest.mark.parametrize("side", ["R", "L"])
def test_wheel_frame_is_the_ring_frame_of_the_palm(side):
    """target_in_wrist is the ring frame: proper rotation, palm centre at y = 0 on the approach side at the reported
    distance from the centreline, palm toward the tube and fingers going the +1 way round."""
    hand, res = HANDS[side], wheel_solved(side)
    m = wheel_measure(hand, res, 0.19, 0.015)
    Rf, c = m["Rf"], m["palm_c"]
    assert np.linalg.det(Rf) == pytest.approx(1.0, abs=1e-9) and np.allclose(Rf.T @ Rf, np.eye(3), atol=1e-9)
    assert abs(c[1]) < 1e-9
    assert math.degrees(math.atan2(c[2], c[0] + 0.0)) == pytest.approx(90.0, abs=1e-6)
    assert math.hypot(c[0], c[2]) * 1e3 == pytest.approx(res["report"]["palm_centre_mm"], abs=0.06)
    assert (Rf.T @ hand.ventral) @ [0, 0, -1.0] > math.cos(math.radians(45))        # palm faces the tube centreline
    assert (Rf.T @ hand.along) @ [-1.0, 0, 0] > 0.5                                   # fingers go toward -x: +1 wrap


@pytest.mark.parametrize("side,approach,wrap", [("R", 90.0, 1), ("L", 90.0, 1), ("R", 0.0, -1), ("L", 0.0, -1)])
def test_wheel_fingers_wrap_one_way_and_the_thumb_comes_from_the_other(side, approach, wrap):
    """Measured on the posed skin: the distal pads lie round the tube on the wrap side of the palm, the thumb pad on
    the opposite side and displaced along the tube toward the index finger."""
    hand = HANDS[side]
    res = wheel_solved(side, approach=approach, wrap=wrap, seeds=1 if approach == 0.0 else 2)
    m = wheel_holds(hand, res, 0.19, 0.015)
    palm = approach
    for f in G.FOUR:
        i = m[f][2][np.argmin(m["gap"][m[f][2]])]
        ahead = ((wheel_section(m, 0.19, i)[0] - palm) * wrap + 180.0) % 360.0 - 180.0
        assert 60.0 <= ahead <= 240.0, f"{f} distal pad {ahead:.0f} deg round the tube"
    i = m["thumb"][np.argmin(m["gap"][m["thumb"]])]
    sec, arc = wheel_section(m, 0.19, i)
    behind = -(((sec - palm) * wrap + 180.0) % 360.0 - 180.0)
    assert 20.0 <= behind <= 140.0, f"thumb pad {behind:.0f} deg behind the palm"
    index_side = -hand.chir * wrap                                   # the sign of y toward the index finger
    assert 0.010 <= index_side * arc <= 0.095, f"thumb {index_side * arc * 1e3:.0f} mm along the tube toward the index"
    thumb = m["Rf"].T @ (m["H"][hand.F["thumb"][3]] - m["H"][hand.F["thumb"][2]])    # posed last thumb bone, IP -> tip
    ring = arc / 0.19                                                # ... runs along the tube there, toward the index
    along_tube = index_side * np.array([-math.sin(ring), math.cos(ring), 0.0])
    assert thumb @ along_tube > 0.5 * np.linalg.norm(thumb)


@pytest.mark.parametrize("side", ["R", "L"])
def test_wheel_mirrored_hands_get_the_mirrored_grip(side):
    other = "L" if side == "R" else "R"
    a, b = wheel_solved(side)["report"], wheel_solved(other)["report"]
    for k in G.DOF:
        assert a["angles_deg"][k] == pytest.approx(b["angles_deg"][k], abs=1.0)
    assert a["palm_centre_mm"] == pytest.approx(b["palm_centre_mm"], abs=0.5)


@pytest.mark.parametrize("side", ["R"])
def test_wheel_does_not_depend_on_where_the_hand_model_sits_in_the_world(side):
    """The grip lives in the hand's own frame: the same hand moved and turned elsewhere gets the same grip."""
    moved_hand = moved(HANDS[side])
    prop = {"radius": 0.19, "tube": 0.015}
    a = G.solve_wheel(HANDS[side], prop, seeds=1, workers=1)
    b = G.solve_wheel(moved_hand, prop, seeds=1, workers=1)
    wheel_holds(moved_hand, b, 0.19, 0.015)
    assert b["solver"]["cost"] == pytest.approx(a["solver"]["cost"], rel=0.05)
    assert np.abs(np.array(a["target_in_wrist"]) - np.array(b["target_in_wrist"])).max() < 0.02
    for k in G.DOF:
        assert a["report"]["angles_deg"][k] == pytest.approx(b["report"]["angles_deg"][k], abs=3.0)


# ---------------------------------------------------------------- sizes
@pytest.mark.parametrize("side,R,r", [("R", 0.19, 0.008), ("L", 0.19, 0.025), ("R", 0.06, 0.015), ("L", 0.5, 0.015)])
def test_wheel_holds_the_smallest_and_largest_tubes_and_rings(side, R, r):
    wheel_holds(HANDS[side], wheel_solved(side, R=R, r=r, seeds=1), R, r)


def test_wheel_a_tube_too_big_to_close_round_still_gives_a_clean_partial_grip():
    """A 60 mm tube radius: the fingers cannot close round it, but nothing is inside it and the report says so."""
    res = wheel_solved("R", r=0.06, seeds=1)
    m = wheel_measure(HANDS["R"], res, 0.19, 0.06)
    rep = res["report"]
    assert m["penetration"] <= 0.5 and m["clash"] <= 0.5
    assert np.isfinite([rep["contacts"][f]["dist_mm"] for f in G.FOUR]).all()
    assert wheel_gap_mm(m, m["palm"]) <= 2.0
    small = wheel_solved("R")["report"]["contacts"]
    wrapped = lambda c: np.mean([c[f]["wrapped_deg"] for f in G.FOUR])
    assert wrapped(rep["contacts"]) < wrapped(small) - 30
    a, b = rep["angles_deg"], wheel_solved("R")["report"]["angles_deg"]
    assert sum(a[f + "_pip"] + a[f + "_dip"] for f in G.FOUR) < sum(b[f + "_pip"] + b[f + "_dip"] for f in G.FOUR)


# ---------------------------------------------------------------- parameters and results
def test_wheel_thumb_section_tuning_moves_the_thumb_round_the_tube():
    res = wheel_solved("R", seeds=1, thumb_section=(95.0, 135.0))
    behind = res["report"]["contacts"]["thumb"]["behind_palm_deg"]
    assert 90.0 <= behind <= 140.0
    assert wheel_solved("R")["report"]["contacts"]["thumb"]["behind_palm_deg"] < 95.0


def test_wheel_result_has_the_uniform_shape_and_serialises():
    res = wheel_solved("L")
    assert res["style"] == "wheel" and res["side"] == "L"
    assert set(res) >= {"style", "side", "bones", "target_in_wrist", "report", "solver"}
    assert res["solver"]["seeds"] == 2 and res["solver"]["cost"] > 0
    assert np.array(res["target_in_wrist"]).shape == (4, 4)
    rep = res["report"]
    for f in G.FOUR:
        assert set(rep["contacts"][f]) >= {"gap_mm", "prox_mm", "mid_mm", "dist_mm", "wrapped_deg"}
    assert set(rep["contacts"]["thumb"]) >= {"gap_mm", "section_deg", "tangent_mm"}
    assert "gap_mm" in rep["contacts"]["palm"]
    assert rep["palm_centre_mm"] > 15.0 and rep["penetration_mm"] >= 0.0 and rep["finger_clash_mm"] >= 0.0
    back = json.loads(json.dumps(res, default=lambda o: np.asarray(o).tolist()))
    assert back["report"]["angles_deg"] == rep["angles_deg"]
    assert all(len(q) == 4 for q in back["bones"].values())


@pytest.mark.parametrize("prop,kw,msg", [
    ({"radius": 0.19}, {}, "tube"),
    ({"tube": 0.015}, {}, "radius"),
    ({"radius": 0.19, "tube": 0.2}, {}, "tube < radius"),
    ({"radius": 0.19, "tube": 0.0}, {}, "tube < radius"),
    ({"radius": 0.19, "tube": -0.01}, {}, "tube < radius"),
    ({"radius": float("nan"), "tube": 0.015}, {}, "tube < radius"),
    ({"radius": "wide", "tube": 0.015}, {}, "numeric"),
    ([0.19, 0.015], {}, "dict"),
    ({"radius": 0.19, "tube": 0.015}, {"wrap": 0}, "wrap"),
    ({"radius": 0.19, "tube": 0.015}, {"wrap": 2}, "wrap"),
    ({"radius": 0.19, "tube": 0.015}, {"approach": float("inf")}, "approach"),
    ({"radius": 0.19, "tube": 0.015}, {"seeds": 0}, "seeds"),
    ({"radius": 0.19, "tube": 0.015}, {"seeds": 1.5}, "seeds"),
    ({"radius": 0.19, "tube": 0.015}, {"bogus": 1}, "unknown wheel parameters: bogus"),
    ({"radius": 0.19, "tube": 0.015}, {"prior": {"index_knee": 3}}, "unknown joint"),
    ({"radius": 0.19, "tube": 0.015}, {"thumb_section": (50.0, 20.0)}, "thumb_section"),
    ({"radius": 0.19, "tube": 0.015}, {"thumb_tangent": (10.0,)}, "thumb_tangent"),
    ({"radius": 0.19, "tube": 0.015}, {"nfev": (0, 5)}, "nfev"),
    ({"radius": 0.19, "tube": 0.015}, {"nfev": 20}, "nfev"),
    ({"radius": 0.19, "tube": 0.015}, {"thumb_section": 40.0}, "thumb_section"),
    ({"radius": 0.19, "tube": 0.015}, {"prior": [1, 2]}, "prior"),
])
def test_wheel_rejects_invalid_input(prop, kw, msg):
    with pytest.raises(ValueError, match=msg):
        G.solve_wheel(HANDS["R"], prop, workers=1, **kw)


def test_wheel_ignores_other_prop_keys():
    kw = dict(approach=90.0, wrap=1, seeds=1, workers=1, nfev=(8, 4))
    a = G.solve_wheel(HANDS["R"], {"radius": 0.19, "tube": 0.015}, **kw)
    b = G.solve_wheel(HANDS["R"], {"radius": 0.19, "tube": 0.015, "name": "wheel", "type": "torus", "colour": 3}, **kw)
    assert np.allclose(a["target_in_wrist"], b["target_in_wrist"], atol=1e-12)


# ---------------------------------------------------------------- pinch: thumb-index pad pinch
PINCH_EDGE = 0.004


def pinch_slab_sdf(L, w, depth=0.15, length=0.16, edge=PINCH_EDGE):
    """Signed distance (m, + outside) of points in the object's frame to the held slab."""
    q = np.abs(L - np.array([depth / 2 - edge, 0.0, 0.0])) - np.array([depth / 2, length / 2, w / 2])
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def pinch_measure(hand, res, w, depth=0.15, length=0.16, edge=PINCH_EDGE):
    """Gaps, penetration, clash and contact points of a result, from its bones and frame alone."""
    Q = hand.rotations_from_local(res["bones"])
    D, H = hand.fk(Q)
    P = hand.skin(D, H)
    T = np.array(res["target_in_wrist"])
    R = hand.rest_rot[0] @ T[:3, :3]
    p = hand.h[0] + hand.rest_rot[0] @ T[:3, 3]
    L = (P - p) @ R
    sdf = pinch_slab_sdf(L, w, depth, length, edge)
    F = hand.F
    out = {"R": R, "p": p, "L": L, "sdf": sdf}
    # the pads: the palmar skin of the distal phalanges (index: ventral side, thumb: the side its pad faces)
    for k, idx, sign in (("index", hand.surface(F["index"][2], hand.ventral, 0.0, 1.0, 0.3), -1.0),
                         ("thumb", hand.surface(F["thumb"][2], hand.pad_t, 0.0, 1.0, 0.3), 1.0)):
        g = sdf[idx]
        out["gap_" + k] = float(g.min()) * 1e3
        plane = sign * L[idx, 2] - w / 2                                   # distance outside this pad's face
        wt = np.exp(-(plane - plane.min()) / 0.001)
        out["pt_" + k] = (wt / wt.sum()) @ L[idx] * 1e3                     # softmin-weighted contact point
        out["patch_" + k] = L[idx[g < g.min() + 0.001]] * 1e3               # the skin within 1 mm of touching
        out["z_" + k] = float(L[idx[np.argmin(g)], 2]) * 1e3                # the closest skin point's height
    a, b = out["patch_index"][:, None, :2], out["patch_thumb"][None, :, :2]
    out["overlap"] = float(np.linalg.norm(a - b, axis=2).min())              # patches' distance in the mid-plane
    out["offset"] = float(np.linalg.norm((out["pt_index"] - out["pt_thumb"])[:2]))
    out["squeeze"] = out["z_thumb"] - out["z_index"] - w * 1e3
    out["pen"] = max(0.0, -float(sdf[hand.dom < hand.nb].min())) * 1e3
    out["clash"] = float(G._Clash(hand).depth(P, H).max()) * 1e3
    outer = [b_ for f in ("middle", "ring", "little") for b_ in F[f][:3]]
    out["outer"] = float(sdf[np.isin(hand.dom, outer)].min()) * 1e3
    return out


_PINCH_SOLVED = {}


def pinch_solved(side, w, seeds=1, workers=1, **kw):
    key = (side, w, seeds, workers, tuple(sorted(kw.items())))
    if key not in _PINCH_SOLVED:
        prop = {"width": w, "name": "card", "type": "pinch"}          # extra keys are ignored
        prop.update({k: kw.pop(k) for k in ("depth", "length") if k in kw})
        _PINCH_SOLVED[key] = G.solve_pinch(HANDS[side], prop, seeds=seeds, workers=workers, **kw)
    return _PINCH_SOLVED[key]


def pinch_holds(hand, res, w, **slab):
    """The acceptance numbers: gaps within [-0.5, 1.0] mm, no penetration, no clash, pads overlapping and centred."""
    m = pinch_measure(hand, res, w, **slab)
    assert -0.5 <= m["gap_index"] <= 1.0, m["gap_index"]
    assert -0.5 <= m["gap_thumb"] <= 1.0, m["gap_thumb"]
    assert m["pen"] <= 0.5, m["pen"]
    assert m["clash"] <= 0.5, m["clash"]
    assert m["offset"] <= 4.0, m["offset"]                               # contact points overlap in the mid-plane
    assert m["overlap"] <= 3.0, m["overlap"]                             # ... and so do the touching patches
    for k in ("pt_index", "pt_thumb"):
        assert abs(m[k][0]) <= 3.0 and abs(m[k][1]) <= 3.0, (k, m[k])    # pads sit `edge` inside the edge, centred
    assert m["pt_index"][2] < 0 < m["pt_thumb"][2]                       # z points from the index pad to the thumb pad
    return m


@pytest.mark.parametrize("side", ["R", "L"])
def test_pinch_holds_a_strap_with_both_hands(side):
    hand, w = HANDS[side], 0.011
    res = pinch_solved(side, w)
    assert res["style"] == "pinch" and res["side"] == side
    m = pinch_holds(hand, res, w)
    # the frame is a proper rotation, z runs index pad -> thumb pad across the object, x leads away from the wrist
    R = m["R"]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0
    assert abs((m["pt_thumb"][2] - m["pt_index"][2]) * 1e-3 - w) < 0.003
    assert (m["p"] - hand.h[0]) @ R[:, 0] > 0.04
    assert res["report"]["heading_deg"] < 40.0
    # bones: a unit quaternion for every finger joint, nothing for the wrist
    assert len(res["bones"]) == 15
    for q in res["bones"].values():
        assert len(q) == 4 and abs(np.linalg.norm(q) - 1.0) < 1e-6


@pytest.mark.parametrize("side,w", [("R", 0.0005), ("L", 0.040)])
def test_pinch_holds_the_thinnest_and_thickest_objects(side, w):
    pinch_holds(HANDS[side], pinch_solved(side, w), w)


@pytest.mark.parametrize("prop", [
    {}, {"width": None}, {"width": 0.0004}, {"width": 0.0}, {"width": -0.01}, {"width": 0.0401}, {"width": 1.0},
    {"width": float("nan")}, {"width": float("inf")}, {"width": "thick"}, {"width": [0.01]},
    {"width": 0.01, "depth": 0.003}, {"width": 0.01, "depth": 0.004}, {"width": 0.01, "length": 0.0},
    {"width": 0.01, "length": -0.1}, {"width": 0.01, "depth": "x"}])
def test_pinch_rejects_unsupported_props(prop):
    with pytest.raises(ValueError):
        G.solve_pinch(HANDS["R"], prop, seeds=1, workers=1)


def test_pinch_rejects_bad_parameters():
    for kw in ({"edge": -0.001}, {"edge": float("nan")}, {"seeds": 0}, {"colour": 3}):
        with pytest.raises(ValueError):
            G.solve_pinch(HANDS["R"], {"width": 0.01}, workers=1, **{"seeds": 1, **kw})


def test_pinch_report_matches_the_measurement():
    hand, w = HANDS["R"], 0.011
    res = pinch_solved("R", w)
    rep, m = res["report"], pinch_measure(hand, res, w)
    assert rep["contacts"]["index"]["gap_mm"] == pytest.approx(m["gap_index"], abs=0.05)
    assert rep["contacts"]["thumb"]["gap_mm"] == pytest.approx(m["gap_thumb"], abs=0.05)
    assert rep["penetration_mm"] == pytest.approx(m["pen"], abs=0.05)
    assert rep["finger_clash_mm"] == pytest.approx(m["clash"], abs=0.05)
    assert rep["outer_fingers_clear_mm"] == pytest.approx(m["outer"], abs=0.1)
    assert rep["pad_offset_mm"] == pytest.approx(m["offset"], abs=0.1)
    assert rep["squeeze_mm"] == pytest.approx(m["squeeze"], abs=0.1)
    assert rep["width_mm"] == pytest.approx(11.0)
    for k in ("index", "thumb"):
        c = rep["contacts"][k]
        assert set(c) >= {"gap_mm", "x_mm", "y_mm"}
        assert c["x_mm"] == pytest.approx(m["pt_" + k][0], abs=0.1)
        assert c["y_mm"] == pytest.approx(m["pt_" + k][1], abs=0.1)
        assert c["facing_deg"] < 40.0
    assert rep["pad_offset_mm"] < 4.0
    assert 0.0 <= rep["squeeze_mm"] < 2.0                          # the pads touch the faces, they do not press through
    assert list(rep["angles_deg"]) == G.DOF


def test_pinch_posture_is_a_natural_pinch():
    for side, w in (("R", 0.002), ("L", 0.025)):
        res = pinch_solved(side, w)
        a = res["report"]["angles_deg"]
        tot = {f: a[f + "_mcp"] + a[f + "_pip"] + a[f + "_dip"] for f in ("index", "middle", "ring", "little")}
        # index flexed, the rest curled more and more; nothing hyperextended
        assert 0.0 <= a["index_mcp"] <= 70.0 and 0.0 <= a["index_pip"] <= 90.0 and 0.0 <= a["index_dip"] <= 45.0
        assert tot["middle"] > tot["index"] and tot["ring"] > tot["middle"] and tot["little"] > tot["ring"]
        assert min(a["t_mcp"], a["t_ip"]) >= -1e-6 and 0.0 <= a["t_palmar"] <= 80.0
        for f in ("index", "middle", "ring", "little"):
            assert min(a[f + "_mcp"], a[f + "_pip"], a[f + "_dip"]) >= -1e-6
        # the outer fingers keep clear of the object (default 10 mm)
        assert res["report"]["outer_fingers_clear_mm"] >= 9.0


@pytest.mark.parametrize("edge,depth,length", [(0.0, 0.15, 0.16), (0.004, 0.03, 0.03)])
def test_pinch_follows_the_edge_and_the_held_extent(edge, depth, length):
    hand, w = HANDS["R"], 0.006
    res = pinch_solved("R", w, edge=edge, depth=depth, length=length)
    # the object ends where the prop says: a small coin-sized slab is held as firmly as a big sheet
    pinch_holds(hand, res, w, depth=depth, length=length, edge=edge)


def test_pinch_wider_objects_open_the_thumb():
    thin, thick = pinch_solved("R", 0.002)["report"]["angles_deg"], pinch_solved("R", 0.040)["report"]["angles_deg"]
    assert thick["t_radial"] > thin["t_radial"]
    assert math.isfinite(thick["t_palmar"])


# ---------------------------------------------------------------- rest: a relaxed hand on a plane
REST_PLANE = {"surface": "plane"}
REST_FACES = ("palm", "back")
REST_REGIONS = ("palm_heel", "index", "middle", "ring", "little", "thumb")


def rest_hand(side, scale=1.0, **kw):
    hand = synthetic_hand(side, **kw)
    if scale == 1.0:
        return hand
    a = G._hand_args(hand)
    for key in ("heads", "tails", "V", "arm_chain"):
        a[key] = hand.h[0] + (np.asarray(a[key]) - hand.h[0]) * scale
    return G.HandModel(**a)


@functools.lru_cache(maxsize=None)
def rest_solved(side, face, scale=1.0, seeds=2, **tuning):
    hand = rest_hand(side, scale)
    kw = dict(tuning)
    if "angles" in kw:
        kw["angles"] = dict(kw["angles"])
    return hand, G.solve_rest(hand, REST_PLANE, face=face, seeds=seeds, workers=1, **kw)


def rest_state(hand, res):
    """Independent of the solver: the posed skin and bone heads from the result's bone rotations, and the grip frame
    (axes R as columns, origin p) in the hand model's frame."""
    D, H = hand.fk(hand.rotations_from_local(res["bones"]))
    T = np.asarray(res["target_in_wrist"])
    return hand.skin(D, H), H, hand.rest_rot[0] @ T[:3, :3], hand.h[0] + hand.rest_rot[0] @ T[:3, 3]


def rest_regions(hand, face):
    return G._Rest(hand, face, **G._rest_tuning(face, {})).regions


def rest_check(hand, res, face):
    P, H, R, p = rest_state(hand, res)
    z = (P - p) @ R[:, 2]                                  # height above the plane = z in the grip frame
    rep = res["report"]
    body = hand.dom < hand.nb
    for name, idx in rest_regions(hand, face).items():
        gap = z[idx].min() * 1e3
        assert -0.5 <= gap <= 1.5, f"{face} {hand.side} {name}: gap {gap:.2f} mm"
        assert rep["contacts"][name]["gap_mm"] == pytest.approx(gap, abs=0.01)
    assert set(rep["contacts"]) == set(REST_REGIONS)
    assert -z[body].min() * 1e3 <= 0.3                     # nothing of the hand below the plane
    assert rep["penetration_mm"] == pytest.approx(max(0.0, -z[body].min()) * 1e3, abs=0.01)
    clash = G._Clash(hand).depth(P, H).max() * 1e3
    assert clash <= 0.5 and rep["finger_clash_mm"] == pytest.approx(clash, abs=0.01)
    return z


@pytest.mark.parametrize("face", REST_FACES)
@pytest.mark.parametrize("side", ["R", "L"])
def test_rest_touches_the_plane_and_nothing_goes_through_it(side, face):
    hand, res = rest_solved(side, face)
    z = rest_check(hand, res, face)
    assert res["style"] == "rest" and res["side"] == side
    assert z[hand.dom < hand.nb].min() * 1e3 < 1.0         # the hand lies on the plane: its lowest skin touches it


@pytest.mark.parametrize("face", REST_FACES)
@pytest.mark.parametrize("side", ["R", "L"])
def test_rest_frame_is_the_plane_below_the_palm_with_x_the_heading(side, face):
    hand, res = rest_solved(side, face)
    _, _, R, p = rest_state(hand, res)
    z, x, y = R[:, 2], R[:, 0], R[:, 1]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(np.cross(z, x), y, atol=1e-9)
    assert np.allclose(x, G._unit(G._perp(hand.along, z)), atol=1e-9)        # the heading projected on the plane
    assert x @ hand.along > 0.9
    # the palm centre sits straight above the origin, at the reported height; the hand lies on its palm or its back
    off = hand.palm_c - p
    assert np.linalg.norm(off - (off @ z) * z) < 1e-9
    assert off @ z == pytest.approx(res["report"]["palm_height_mm"] * 1e-3, abs=6e-5)
    assert (hand.h[0] - p) @ z == pytest.approx(res["report"]["wrist_height_mm"] * 1e-3, abs=6e-5)
    assert z @ (hand.dorsal if face == "palm" else hand.ventral) > 0.9
    assert abs(z @ hand.across) < 0.3                                         # the plane is not rolled on its side
    # the frame is the plane for the solve and the wrist frame the build uses gets the same numbers back
    T = res["target_in_wrist"]
    assert np.allclose(T @ np.array([0, 0, 0, 1.0]), np.r_[hand.rest_rot[0].T @ (p - hand.h[0]), 1.0], atol=1e-12)


@pytest.mark.parametrize("side", ["R", "L"])
def test_rest_palm_down_is_a_relaxed_hand(side):
    hand, res = rest_solved(side, "palm")
    a, rep = res["report"]["angles_deg"], res["report"]
    for f in G.FOUR:
        assert 5 <= a[f"{f}_mcp"] <= 32 and 10 <= a[f"{f}_pip"] <= 42 and 0 <= a[f"{f}_dip"] <= 26, f
        assert -12 <= a[f"{f}_spr"] <= 8, f
    assert a["index_spr"] > a["ring_spr"] > a["little_spr"]                   # a slight fan
    assert a["index_dip"] < a["little_dip"]                       # curling a little more toward the little finger
    assert 3 <= rep["pitch_deg"] <= 22 and abs(rep["roll_deg"]) <= 12         # the wrist a little extended, level
    assert abs(a["t_palmar"]) <= 30 and abs(a["t_radial"]) <= 20 and a["t_mcp"] <= 30 and a["t_ip"] <= 30
    assert rep["wrist_height_mm"] > 2.0                           # the palm arches between heel and fingertips
    assert rep["palm_height_mm"] > rep["contacts"]["palm_heel"]["gap_mm"] + 10.0


@pytest.mark.parametrize("side", ["R", "L"])
def test_rest_on_the_back_the_fingers_curl_up_and_away(side):
    hand, res = rest_solved(side, "back")
    a, rep = res["report"]["angles_deg"], res["report"]
    _, H, R, p = rest_state(hand, res)
    for f in G.FOUR:
        assert -25 <= a[f"{f}_mcp"] <= 10 and 5 <= a[f"{f}_pip"] <= 35 and 3 <= a[f"{f}_dip"] <= 30, f
        tip = (H[hand.F[f][3]] - p) @ R[:, 2]
        mid = (H[hand.F[f][1]] - p) @ R[:, 2]
        assert tip > mid + 0.002, f                               # the fingertip is higher than the middle joint
    assert 0 <= rep["pitch_deg"] <= 20 and abs(rep["roll_deg"]) <= 12
    assert rep["wrist_height_mm"] > 8.0                                       # the wrist rests above the plane


def test_rest_mirrored_hands_give_mirrored_answers():
    for face in REST_FACES:
        right, left = rest_solved("R", face)[1], rest_solved("L", face)[1]
        for k in ("pitch_deg", "roll_deg", "palm_height_mm", "wrist_height_mm"):
            assert right["report"][k] == pytest.approx(left["report"][k], abs=0.15), (face, k)
        for k, v in right["report"]["angles_deg"].items():
            assert v == pytest.approx(left["report"]["angles_deg"][k], abs=0.3), (face, k)


def test_rest_does_not_depend_on_where_the_hand_model_sits_in_the_world():
    base = rest_solved("R", "palm")[1]
    again = G.solve_rest(moved(rest_hand("R")), REST_PLANE, seeds=2, workers=1)
    for k, v in base["report"]["angles_deg"].items():
        assert again["report"]["angles_deg"][k] == pytest.approx(v, abs=0.2), k
    assert np.allclose(again["target_in_wrist"], base["target_in_wrist"], atol=2e-4)


@pytest.mark.parametrize("face,scale", [("back", 0.7), ("palm", 1.35)])
def test_rest_holds_for_small_and_large_hands(face, scale):
    hand, res = rest_solved("R", face, scale)
    rest_check(hand, res, face)
    assert res["report"]["pitch_deg"] == pytest.approx(rest_solved("R", face)[1]["report"]["pitch_deg"], abs=3.0)


def test_rest_pitch_tuning_tilts_the_hand_and_the_contacts_still_hold():
    hand, level = rest_solved("R", "palm", 1.0, 1, pitch=(0.0, 1.0))
    rest_check(hand, level, "palm")
    hand, raised = rest_solved("R", "palm", 1.0, 1, pitch=(22.0, 1.0))
    rest_check(hand, raised, "palm")
    assert level["report"]["pitch_deg"] < 6.0 < 16.0 < raised["report"]["pitch_deg"]
    assert raised["report"]["angles_deg"]["middle_pip"] > level["report"]["angles_deg"]["middle_pip"] + 8.0


def test_rest_angle_tuning_curls_that_joint_and_the_contacts_still_hold():
    hand, curled = rest_solved("R", "palm", 1.0, 1, angles=(("index_pip", 45.0),))
    rest_check(hand, curled, "palm")
    default = rest_solved("R", "palm")[1]["report"]["angles_deg"]["index_pip"]
    assert curled["report"]["angles_deg"]["index_pip"] > default + 3.0


def test_rest_gap_tuning_sets_how_close_the_contacts_sit():
    hand, lifted = rest_solved("R", "palm", 1.0, 1, gap_mm=1.0)
    gaps = [c["gap_mm"] for c in lifted["report"]["contacts"].values()]
    assert all(0.9 <= g <= 1.6 for g in gaps), gaps
    assert min(c["gap_mm"] for c in rest_solved("R", "palm")[1]["report"]["contacts"].values()) < 0.45


def test_rest_rejects_what_it_cannot_solve():
    hand = rest_hand("R")
    for bad_prop in ({}, {"surface": "sphere"}, {"surface": "Plane"}, {"radius": 0.1}, None, "plane"):
        with pytest.raises(ValueError, match="surface"):
            G.solve_rest(hand, bad_prop, seeds=1, workers=1)
    for face in ("front", "", None, "PALM"):
        with pytest.raises(ValueError, match="face"):
            G.solve_rest(hand, REST_PLANE, face=face, seeds=1, workers=1)
    with pytest.raises(ValueError, match="seeds"):
        G.solve_rest(hand, REST_PLANE, seeds=0, workers=1)
    with pytest.raises(ValueError, match="unknown rest tuning .*spoon"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, spoon=1)
    with pytest.raises(ValueError, match="unknown joint"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, angles={"index_knee": 10})
    with pytest.raises(ValueError, match="pitch"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, pitch=(5.0,))
    with pytest.raises(ValueError, match="tolerance"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, roll=(0.0, 0.0))
    with pytest.raises(ValueError, match="gap_mm"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, gap_mm=-1.0)
    with pytest.raises(ValueError, match="angles"):
        G.solve_rest(hand, REST_PLANE, seeds=1, workers=1, angles=[10, 20])


def test_rest_ignores_other_prop_keys():
    hand = rest_hand("R")
    quick = dict(seeds=1, workers=1, nfev=(4, 4, 4))
    a = G.solve_rest(hand, REST_PLANE, **quick)
    b = G.solve_rest(hand, {"surface": "plane", "name": "desk", "height": 0.74}, **quick)
    assert a["solver"]["cost"] == b["solver"]["cost"] and a["bones"] == b["bones"]


# ---------------------------------------------------------------- the CLI's arguments and gates
def _ns(**kw):
    base = dict(style="pen", prop=None, params=None, posture=None, length=None, radius=None, tip=None, tube=None,
                width=None, surface=None, nib_offset=None, approach=None, wrap=None, edge=None, face=None, seeds=None,
                workers=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_flags_override_the_prop_and_params_json(tmp_path):
    f = tmp_path / "prop.json"
    f.write_text(json.dumps({"length": 0.12, "radius": 0.007, "name": "barrel"}))
    assert cli._prop(_ns(prop=f"@{f}", radius=0.0065, nib_offset=[0, 0, 0.01])) == \
        {"length": 0.12, "radius": 0.0065, "name": "barrel", "nib_offset": [0, 0, 0.01]}
    assert cli._params(_ns(style="wheel", params='{"seeds": 3, "orient_starts": 8}', seeds=5, wrap=-1)) == \
        {"seeds": 5, "orient_starts": 8, "wrap": -1}
    with pytest.raises(cli.UsageError):
        cli._prop(_ns(prop="{not json"))
    with pytest.raises(cli.UsageError):
        cli._prop(_ns(prop="[1, 2]"))


def test_cli_rejects_a_flag_that_belongs_to_another_style():
    assert cli._prop(_ns(style="wheel", radius=0.19, tube=0.015)) == {"radius": 0.19, "tube": 0.015}   # shared flag
    with pytest.raises(cli.UsageError, match="--tube is an option of the wheel style, not pen"):
        cli._prop(_ns(tube=0.015))
    with pytest.raises(cli.UsageError, match="--wrap is an option of the wheel style, not rest"):
        cli._params(_ns(style="rest", wrap=1))


def test_cli_gates_name_what_a_grip_misses():
    res = {"report": {"penetration_mm": 0.4, "finger_clash_mm": 2.5,
                      "contacts": {"index": {"gap_mm": 0.3}, "thumb": {"gap_mm": 4.2}, "web": 7}}}
    bad = cli.problems(res, max_gap=3.0, max_penetration=1.0, max_clash=1.0)
    assert len(bad) == 2 and any("finger clash" in b for b in bad) and any("thumb" in b for b in bad)
    assert cli.problems(res, max_gap=5.0, max_penetration=1.0, max_clash=3.0) == []


# ---------------------------------------------------------------- the build's solver interface (python -m)
def _arrays(hand, style, prop, **params):
    """What the pose stage hands the solver: the hand_model arrays and the spec."""
    return {"side": np.array(hand.side), "names": np.array(hand.names), "sem": np.array(hand.sem),
            "parents": hand.parents, "heads": hand.h, "tails": hand.tails, "rest_rot": hand.rest_rot, "V": hand.V,
            "W": hand.W, "arm_world": hand.arm_world, "arm_chain": hand.arm_chain,
            "spec": np.array(json.dumps({"style": style, "prop": prop, "params": params}))}


@pytest.mark.parametrize("side", ["R", "L"])
def test_main_writes_what_the_solver_returns(tmp_path, side):
    hand = HANDS[side]
    params = dict(seeds=1, workers=1, nfev=[10, 10, 10])
    np.savez(tmp_path / "in.npz", **_arrays(hand, "rest", {"surface": "plane", "name": "desk"}, face="back", **params))
    assert G.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")]) == 0
    direct = G.solve_rest(hand, {"surface": "plane"}, face="back", **params)
    with np.load(tmp_path / "out.npz", allow_pickle=False) as z:
        assert set(z.files) == {"bones", "quats", "target_in_wrist", "report"}
        assert list(z["bones"]) == list(direct["bones"]) and z["quats"].shape == (15, 4)
        assert np.allclose(z["quats"], [direct["bones"][b] for b in z["bones"]], atol=1e-12)
        assert np.allclose(np.linalg.norm(z["quats"], axis=1), 1.0, atol=1e-9)
        assert np.allclose(z["target_in_wrist"], direct["target_in_wrist"], atol=1e-12)
        rep = json.loads(str(z["report"]))
    assert rep["style"] == "rest" and rep["side"] == side and rep["solver"]["seconds"] >= 0.0
    assert rep["report"]["contacts"].keys() == direct["report"]["contacts"].keys()
    assert rep["report"]["penetration_mm"] == pytest.approx(direct["report"]["penetration_mm"])


def test_main_keeps_the_pens_writing_orientation(tmp_path):
    posture = {"nib": [0.0, -0.3, 0.74], "shoulder": [0.0, 0.0, 0.93], "pole": [-0.5, 0.2, 0.8], "upper": 0.21,
               "fore": 0.17}
    np.savez(tmp_path / "in.npz", **_arrays(HANDS["R"], "pen", {"length": 0.14, "radius": 0.0068, "tip": 0.0155},
                                            posture=posture, seeds=1, workers=1, orient_starts=1,
                                            nfev=[4, 4, 4, 4]))
    assert G.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")]) == 0
    with np.load(tmp_path / "out.npz", allow_pickle=False) as z:
        rep = json.loads(str(z["report"]))
    assert len(rep["frame_world_quat"]) == 4 and "writing" in rep["report"]


@pytest.mark.parametrize("spec,fragment", [
    ({"style": "spoon", "prop": {}, "params": {}}, "unknown grip style"),
    ({"style": "pinch", "prop": {}, "params": {}}, "width"),
    ({"style": "wheel", "prop": {"radius": 0.19, "tube": 0.015}, "params": {"wrap": 3}}, "wrap"),
    ({"style": "rest", "prop": {"surface": "plane"}, "params": {"nonsense": 1}}, "nonsense"),
    ({"prop": {}}, "style")])
def test_main_exits_2_on_bad_input_and_says_why(tmp_path, capsys, spec, fragment):
    arrays = _arrays(HANDS["R"], "rest", {})
    arrays["spec"] = np.array(json.dumps(spec))
    np.savez(tmp_path / "in.npz", **arrays)
    assert G.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")]) == 2
    err = capsys.readouterr().err
    assert "bad input" in err and fragment in err and not (tmp_path / "out.npz").exists()


def test_main_exits_2_on_missing_files_arrays_and_arguments(tmp_path, capsys):
    assert G.main([]) == 2 and "usage" in capsys.readouterr().err
    assert G.main([str(tmp_path / "nope.npz"), str(tmp_path / "out.npz")]) == 2
    arrays = _arrays(HANDS["R"], "rest", {"surface": "plane"})
    del arrays["V"]
    np.savez(tmp_path / "in.npz", **arrays)
    assert G.main([str(tmp_path / "in.npz"), str(tmp_path / "out.npz")]) == 2 and "'V'" in capsys.readouterr().err


# ---------------------------------------------------------------- grip frames in the world (core.gripframe)
CAR_RING = dict(center=[0.38, -0.30, 0.95], axis=[0.0, 1.0, 0.55], radius=0.19, up=[0.0, 0.0, 1.0])
DRIVER = [0.0, 0.3, 1.2]                                    # a point on the character's side of the ring


def test_ring_frame_reads_the_clock_as_the_character_sees_the_wheel():
    a = GF.unit(CAR_RING["axis"])
    up = GF.unit(np.array(CAR_RING["up"]) - (np.array(CAR_RING["up"]) @ a) * a)      # 12 o'clock
    right = np.cross(-a, up)                                    # 3 o'clock: the character's right (-X facing -Y)
    assert right[0] < 0
    for clock, want in ((12, up), (3, right), (6, -up), (9, -right), (10, 0.5 * up - 0.866025 * right),
                        (2, 0.5 * up + 0.866025 * right)):
        Gf, info = GF.ring_frame(**CAR_RING, clock=clock, toward=DRIVER)
        x, y, z = Gf[:3, 0], Gf[:3, 1], Gf[:3, 2]
        assert np.allclose(x, want, atol=1e-5) and np.allclose(z, a) and not info["flipped"]
        assert np.allclose(np.cross(x, y), z, atol=1e-12) and np.linalg.det(Gf[:3, :3]) == pytest.approx(1.0)
        assert np.allclose(Gf[:3, 3], np.array(CAR_RING["center"]) + 0.19 * x, atol=1e-12)    # on the centreline


def test_ring_frame_points_its_axis_at_the_character():
    behind = [0.0, -2.0, 1.2]                                   # the character on the other side of the ring
    Gf, info = GF.ring_frame(**CAR_RING, clock=10, toward=behind)
    assert info["flipped"] and Gf[:3, 2] @ (np.array(behind) - CAR_RING["center"]) > 0
    Gn, _ = GF.ring_frame(**CAR_RING, clock=10)                 # without a character the card's axis stands
    assert np.allclose(Gn[:3, 2], GF.unit(CAR_RING["axis"]))
    with pytest.raises(ValueError, match="lies flat"):
        GF.ring_frame(center=[0, 0, 1], axis=[0, 0, 1], radius=0.2, up=[0, 0, 1], clock=10)


def test_surface_and_pinch_frames():
    Gs = GF.surface_frame([0.1, 0.2, 0.3], [0.0, 0.0, 2.0], [1.0, 0.5, 4.0])
    assert np.allclose(Gs[:3, 2], [0, 0, 1]) and np.allclose(Gs[:3, 0], GF.unit([1.0, 0.5, 0.0]))
    assert np.allclose(Gs[:3, 3], [0.1, 0.2, 0.3]) and np.linalg.det(Gs[:3, :3]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="heading"):
        GF.surface_frame([0, 0, 0], [0, 0, 1], [0, 0, 3])
    c, n, a = np.array([0.5, 0.0, 0.8]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    for body in ([0.0, 1.0, 1.0], [0.0, -1.0, 1.0]):                # the thumb side faces the character
        Gp = GF.pinch_frame(c, a, n, span=0.024, edge=0.004, toward=body)
        assert np.allclose(Gp[:3, 0], -n) and abs(Gp[:3, 2] @ a) < 1e-12 and abs(Gp[:3, 2] @ n) < 1e-12
        assert Gp[:3, 2] @ (np.array(body) - c) > 0 and np.linalg.det(Gp[:3, :3]) == pytest.approx(1.0)
        assert np.allclose(Gp[:3, 3], c + n * (0.012 - 0.004))     # `edge` inside the outer face
    with pytest.raises(ValueError, match="not parallel"):
        GF.pinch_frame(c, n, n, 0.02, 0.004)


def test_wrist_goal_puts_the_solved_frame_on_the_grip_frame():
    rng = np.random.default_rng(4)
    for _ in range(3):
        T = np.eye(4)
        T[:3, :3] = G._rotm(rng.normal(size=3), rng.uniform(0, 180))
        T[:3, 3] = rng.normal(0, 0.05, 3)
        Gw = np.eye(4)
        Gw[:3, :3] = G._rotm(rng.normal(size=3), rng.uniform(0, 180))
        Gw[:3, 3] = rng.normal(0, 1.0, 3)
        W, goal = GF.wrist_goal(Gw, T, 0.07)
        assert np.allclose(W @ T, Gw, atol=1e-12)                       # frame_world = wrist_world @ target_in_wrist
        assert np.allclose(goal[:3, :3], W[:3, :3]) and np.allclose(goal[:3, 3], W[:3, 3] + 0.07 * W[:3, 1])


def test_card_entries_become_solver_specs():
    ring = {"name": "wheel", "type": "ring", "center": [0, 0, 0], "axis": [0, 1, 0], "radius": 0.19, "tube": 0.016}
    assert GF.card_style(ring, {}) == ("wheel", {"radius": 0.19, "tube": 0.016}, {"approach": 90.0, "wrap": -1})
    assert GF.card_style(ring, {"approach": 0, "wrap": 1, "seeds": 3})[2] == {"seeds": 3, "approach": 0.0, "wrap": 1}
    assert GF.clock_of({}, "L") == 10.0 and GF.clock_of({}, "R") == 2.0 and GF.clock_of({"clock": 9}, "R") == 9.0
    pinch = {"name": "handle", "type": "pinch", "width": 0.011, "span": 0.024, "length": 0.05, "center": [0, 0, 0]}
    assert GF.card_style(pinch, {"edge": 0.003}) == ("pinch", {"width": 0.011, "depth": 0.024, "length": 0.05},
                                                     {"edge": 0.003})
    pen = {"name": "barrel", "type": "pen", "length": 0.14, "radius": [[0, 0.001], [0.02, 0.007]], "tip": 0.0155}
    with pytest.raises(ValueError, match="posture.*do not generalise"):
        GF.card_style(pen, {})
    style, prop, params = GF.card_style(pen, {"posture": {"nib": [0, 0, 0]}})
    assert style == "pen" and prop["length"] == 0.14 and params == {"posture": {"nib": [0, 0, 0]}}
    with pytest.raises(ValueError, match="lacks tube"):
        GF.card_style({"name": "w", "type": "ring", "radius": 0.2}, {})
    with pytest.raises(ValueError, match="no grip style"):
        GF.card_style({"name": "x", "type": "door"}, {})


def _world_gaps(hand, res, Gw, c, a, R, r):
    """The solved grip placed in the world as the pose stage does it: the wrist head frame from G, the hand's posed
    skin carried along (rigidly from its rest wrist frame), gaps (m) of every vertex to the world torus."""
    Wh, _ = GF.wrist_goal(Gw, res["target_in_wrist"], 0.07)
    rest = GF.frame_matrix(hand.rest_rot[0][:, 0], hand.rest_rot[0][:, 1], hand.rest_rot[0][:, 2], hand.h[0])
    M = Wh @ np.linalg.inv(rest)
    D, H = hand.fk(hand.rotations_from_local(res["bones"]))
    P = hand.skin(D, H) @ M[:3, :3].T + M[:3, 3]
    d = P - c
    ax = d @ a
    rho = np.linalg.norm(d - np.outer(ax, a), axis=1)
    return np.hypot(rho - R, ax) - r, P, M


@pytest.mark.parametrize("side,clock", [("L", 10.0), ("R", 2.0)])
def test_wheel_grip_lands_on_the_worlds_rim_at_the_clock_position(side, clock):
    """The whole pose-stage chain without Blender: solve, build G from the ring in the world, place the wrist, skin."""
    hand, r = HANDS[side], 0.016
    res = G.solve_wheel(hand, {"radius": CAR_RING["radius"], "tube": r}, approach=GF.WHEEL_APPROACH,
                        wrap=GF.WHEEL_WRAP, seeds=2, workers=1)
    Gw, info = GF.ring_frame(**CAR_RING, clock=clock, toward=DRIVER)
    c, a = np.array(CAR_RING["center"]), info["axis"]
    gap, P, M = _world_gaps(hand, res, Gw, c, a, CAR_RING["radius"], r)
    body = hand.dom < hand.nb
    assert -gap[body].min() * 1e3 <= 0.5                                     # nothing of the hand inside the rim
    for f in G.FOUR:                                                         # every finger phalanx touches it
        for j in range(3):
            idx = hand.surface(hand.F[f][j], hand.ventral, 0.0, 1.0, 0.3)
            assert -0.5 <= gap[idx].min() * 1e3 <= 1.5, (f, j)
    palm = np.where((hand.dom == 0) & ((hand.V - hand.palm_c) @ hand.ventral > 0.006))[0]
    assert -0.5 <= gap[palm].min() * 1e3 <= 1.5
    pc = M[:3, :3] @ hand.palm_c + M[:3, 3] - c                              # the palm sits at the clock position
    perp = pc - (pc @ a) * a
    assert np.degrees(np.arccos(GF.unit(perp) @ Gw[:3, 0])) < 3.0
    assert pc @ a > 0.0                                                      # ... on the character's side of the rim
    # a steering wheel turned about its axis carries the hand with it: the same grip, rotated rigidly
    turn = G._rotm(a, 35.0)
    gap2, _, _ = _world_gaps(hand, res, np.block([[turn @ Gw[:3, :3], (c + turn @ (Gw[:3, 3] - c))[:, None]],
                                                  [np.zeros((1, 3)), np.ones((1, 1))]]), c, a,
                             CAR_RING["radius"], r)
    assert np.allclose(gap2, gap, atol=1e-9)

