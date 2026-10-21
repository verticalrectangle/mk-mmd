"""A garment of a model mk did not build made into a cloth copy (mkmmd.core.garment, docs/design.md: Cloth): its outside is
the cloth (no lining, no slivers, every face wound outward, two faces to an edge), pinned by height, its sleeves held; an
unbuttoned front slides round the body; a closure that laps over itself is told apart so each side binds to its own
panel. Small synthetic garments round a torso along the z axis; numpy only."""
import numpy as np
import pytest

from mkmmd.core import garment as GM

A = np.array([[0.0, 0.0, 0.8], [0.06, 0.0, 1.4]])              # bone heads: the torso, her left upper arm
B = np.array([[0.0, 0.0, 1.4], [0.4, 0.0, 1.2]])              # bone tails
ARM = np.array([False, True])


def patch(r, th, zs, outward=True):
    """A cylindrical patch round the z axis: vertices at radius r(theta) (a number or a function of theta in degrees, 0
    straight ahead at -y, 90 at her left +x) over the angles `th` and heights `zs`, triangles wound outward (or in)."""
    th, zs = np.asarray(th, float), np.asarray(zs, float)
    rr = np.array([r(t) if callable(r) else r for t in th])
    a = np.radians(th)
    V = np.array([[rr[i] * np.sin(a[i]), -rr[i] * np.cos(a[i]), z] for z in zs for i in range(len(th))])
    T = []
    n = len(th)
    for j in range(len(zs) - 1):
        for i in range(n - 1):
            p, q, s, t = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            T += [(p, q, s), (p, s, t)] if outward else [(p, s, q), (p, t, s)]
    return V, np.array(T, np.int64)


def join(*parts):
    """Patches stacked into one mesh (no vertices shared)."""
    Vs, Ts, off = [], [], 0
    for V, T in parts:
        Vs.append(V)
        Ts.append(T + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Ts)


def weld_mesh(V, T, tol=1e-9):
    idx, W = GM.weld(V, tol)
    return W, idx[T]


def ring_angles(n=36):
    return np.linspace(-180.0, 180.0, n + 1)                   # the last column lies on the first: welded shut


# ------------------------------------------------------------------------------------------------- make
def shell():
    """A jacket body modelled with thickness: the outside (r 0.13, wound out), a lining 2 mm in (wound in) and the hem's
    strip joining them at the bottom, a sliver (a ring of rows 1 mm above the hem) and a fin standing out of the
    outside; weights: the torso, but her left upper arm for the vertices out past x 0.11 above z 1.15 (a sleeve's top)."""
    th, zs = ring_angles(), [0.9, 0.901, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    out, lin = patch(0.13, th, zs), patch(0.128, th, zs, outward=False)
    hem_t = patch(lambda t: 0.13, th, [0.9])[0], patch(0.128, th, [0.9])[0]
    V, T = join(out, lin)
    n, nr = len(th), len(th) * len(zs)
    hem = []
    for i in range(n - 1):                                                            # outside bottom row -> lining bottom row
        a, b, c, d = i, i + 1, nr + i + 1, nr + i
        hem += [(a, d, c), (a, c, b)]
    V = np.concatenate([V, [[0.13 * np.sin(np.radians(5.0)) * 1.0 + 0.0, -0.15, 1.0]]])  # the fin's tip, 2 cm out
    fin_base = (3 * n + 18, 3 * n + 19)                                                 # an edge of the outside at z 1.0
    T = np.concatenate([T, np.array(hem, np.int64), [[fin_base[0], fin_base[1], len(V) - 1]]])
    del hem_t
    V, T = weld_mesh(V, T)
    W = np.zeros((len(V), 2))
    sleeve = (V[:, 0] > 0.11) & (V[:, 2] > 1.15)
    W[:, 0] = np.where(sleeve, 0.2, 1.0)
    W[:, 1] = np.where(sleeve, 0.8, 0.0)
    return V, T, W


def body():
    """The body under it, up to her shoulders: a cylinder of radius 0.11, wound inward (as a model may wind it)."""
    V, T = patch(0.11, ring_angles(), np.linspace(0.85, 1.25, 5), outward=False)
    V, T = weld_mesh(V, T)
    return V, T, np.tile([1.0, 0.0], (len(V), 1))


def make(**kw):
    V, T, W = shell()
    return GM.make(V, T, W, A, B, ARM, body(), **kw)


def test_the_outside_is_the_cloth_no_lining_no_sliver_every_face_outward_two_to_an_edge():
    c = make()
    S, F = c["sim"].astype(float), c["sim_faces"]
    n = GM.face_normals(S, F)
    mid = S[F].mean(1)
    radial = mid * [1.0, 1.0, 0.0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    side = np.abs(n[:, 2]) < 0.5                                                          # the hem's strip lies flat
    r = np.linalg.norm(S[F[side]].reshape(-1, 3)[:, :2], axis=1)
    assert r.min() > 0.1285, r.min()                                                    # no lining face (0.128)
    assert (np.einsum("ij,ij->i", n[side], radial[side]) > 0.9).all()                   # wound outward
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ue, count = np.unique(E, axis=0, return_counts=True)
    assert count.max() <= 2                                                             # the fin's edge carries two faces
    L = np.linalg.norm(S[ue[:, 0]] - S[ue[:, 1]], axis=1)
    assert L.min() >= 0.003                                                             # the 1 mm rows are one
    assert np.allclose(c["sim_weights"].sum(1), 1.0)


def test_the_cloth_is_pinned_by_height_its_sleeves_held_and_the_body_set_in_under_it_wound_outward():
    c = make(pin={"above": 1.25, "fade": 0.1}, sleeves={"hold": 0.8}, inset=0.004)
    S, z = c["sim"].astype(float), c["sim"][:, 2]
    assert (c["pins"][z >= 1.25 - 1e-6] == 1.0).all() and (c["pins"][z <= 1.15 + 1e-6] == 0.0).all()
    sleeve = (S[:, 0] > 0.11) & (z > 1.15)
    assert sleeve.any() and (c["holds"][sleeve] == pytest.approx(0.8)) and (c["collides"][sleeve] == 0).all()
    assert (c["holds"][~sleeve] == 0.0).all() and (c["collides"][~sleeve] == 1).all()
    Vc, Fc = c["collide"].astype(float), c["collide_faces"]
    assert np.linalg.norm(Vc[:, :2], axis=1) == pytest.approx(0.106, abs=2e-4)          # 0.11, 4 mm in
    nc = GM.face_normals(Vc, Fc)
    mid = Vc[Fc].mean(1) * [1.0, 1.0, 0.0]
    assert (np.einsum("ij,ij->i", nc, mid / np.linalg.norm(mid, axis=1, keepdims=True)) > 0.9).all()


def test_collapse_merges_what_a_short_edge_joins_at_its_mean_and_drops_the_faces_that_vanish():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1.002, 0, 0], [2.0, 0.5, 0]], float)
    T = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 3, 2], [4, 5, 3]])
    idx, P, Tn = GM.collapse(V, T, 0.003)
    assert idx[1] == idx[4] and len(P) == 5                                              # 2 mm apart: one vertex
    assert P[idx[1]] == pytest.approx([1.001, 0.0, 0.0])                                  # at their mean
    assert sorted(map(tuple, np.sort(Tn, 1).tolist())) == sorted(map(tuple, np.sort(idx[[[0, 1, 2], [1, 3, 2], [4, 5, 3]]], 1).tolist()))
    idx2, P2, _ = GM.collapse(V, T, 0.001)                                               # shorter than every edge: as it was
    assert len(P2) == len(V) and (idx2 == np.arange(len(V))).all()


# ------------------------------------------------------------------------------------------------- an unbuttoned front
LAP = 12.0                                                                              # degrees the right panel laps over


def jacket():
    """A single-breasted jacket's body: her left panel from 12 degrees right of the front round to the back, her right
    panel from the back round to 12 degrees left of the front, lying 6 mm over the left one in front (its radius eased to
    the left one's by her side), the two sewn together at the back; a flap folded back over the left panel's side (a
    lapel: lying 4 mm on it, joined along its outer edge). Returns (S, F, parts) with parts per vertex: 0 left, 1 right,
    2 the flap."""
    zs = np.linspace(0.9, 1.3, 9)
    lth = np.linspace(-LAP, 180.0, 33)
    rth = np.linspace(-180.0, LAP, 33)
    left = patch(0.13, lth, zs)
    right = patch(lambda t: 0.13 + 0.006 * max(0.0, np.cos(np.radians(t))), rth, zs)
    flap = patch(0.134, np.linspace(40.0, 60.0, 5), zs[2:5], outward=True)
    V, T = join(left, right, flap)
    part = np.r_[np.zeros(len(left[0])), np.ones(len(right[0])), np.full(len(flap[0]), 2.0)].astype(int)
    nl, nr = len(left[0]), len(right[0])
    # the flap's outer column (60 degrees) is the left panel's own column there: one welded edge
    fl = nl + nr + np.arange(len(flap[0]))
    outer = fl[4::5]                                                                        # the flap's 60-degree column
    col = np.argmin(np.abs(lth - 60.0))
    V[outer] = V[[col + j * len(lth) for j in range(2, 5)]]
    idx, S = GM.weld(V, 1e-9)
    F = idx[T]
    p = np.zeros(len(S), int)
    p[idx] = part
    p[idx[[col + j * len(lth) for j in range(2, 5)]]] = 0                                  # the sewn edge is the panel's
    return S, F, p


def angle(P):
    return np.degrees(np.arctan2(P[:, 0], -P[:, 1]))


def test_an_unbuttoned_front_slides_round_the_body_each_panel_to_its_own_side_and_nothing_comes_forward():
    S, F, part = jacket()
    free = np.where(S[:, 2] >= 1.25, 0.0, 1.0)                                           # her shoulders pinned
    O = GM.open_front(S, F, free, 20.0, below=1.1, fade=0.05)
    low = S[:, 2] < 1.05
    front = np.abs(angle(S)) <= LAP + 1e-6
    for p, way in ((0, 1.0), (1, -1.0)):                                                 # her left to +x, her right to -x
        m = low & front & (part == p)
        turn = angle(O[m]) - angle(S[m])
        assert (turn * way > 19.0).all(), (p, turn)                                       # the overlap goes with its panel
    r0 = np.linalg.norm(S[:, :2], axis=1)
    r1 = np.linalg.norm(O[:, :2], axis=1)
    assert r1 == pytest.approx(r0, abs=0.0015)                                           # round the body: nothing comes off it
    assert O[:, 2] == pytest.approx(S[:, 2])
    still = (np.abs(angle(S)) >= 90.0) | (S[:, 2] >= 1.1) | (free == 0.0)              # the back, the top button up
    assert O[still] == pytest.approx(S[still])


def test_a_lapped_closure_is_found_but_a_fold_is_not_and_its_two_sides_take_two_colours():
    S, F, part = jacket()
    pairs = GM.laps(S, F, np.ones(len(S), bool))
    assert len(pairs)
    assert set(map(tuple, np.sort(part[pairs], 1).tolist())) == {(0, 1)}                  # only panel against panel
    assert (np.abs(angle(S[pairs.ravel()])) <= LAP + 9.0).all()                           # where the right laps over (2 cm)
    col = GM.lap_colours(S, F, pairs)
    front = np.abs(angle(S)) < 60.0
    for p in (0, 1):
        assert len(set(col[front & (part == p)].tolist())) == 1                           # each panel one colour
    assert col[front & (part == 0)][0] != col[front & (part == 1)][0]
    assert not GM.lap_colours(S, F, np.zeros((0, 2), int)).any()                          # no laps: one part


def test_a_lining_under_the_overlap_follows_its_own_panel_not_the_one_nearest_it():
    S, F, part = jacket()
    col = GM.lap_colours(S, F, GM.laps(S, F, np.ones(len(S), bool)))
    mine, other = col[(part == 1) & (np.abs(angle(S)) < 1.0)][0], col[(part == 0) & (np.abs(angle(S)) < 1.0)][0]
    # the right panel's lining in front: 2.5 mm over the left panel (nearer it than its own panel, 3.5 mm above),
    # sewn to the right panel along its front edge
    zs = np.linspace(0.9, 1.3, 9)
    th = np.linspace(-LAP, LAP - 3.0, 6)
    Lv = np.array([[0.1325 * np.sin(np.radians(t)), -0.1325 * np.cos(np.radians(t)), z] for z in zs for t in th])
    R = np.concatenate([S, Lv])
    edge = np.flatnonzero((part == 1) & (np.abs(angle(S) - LAP) < 1e-6))                  # the right panel's front edge
    E = []
    n0, k = len(S), len(th)
    for j in range(len(zs)):
        for i in range(k - 1):
            E.append((n0 + j * k + i, n0 + j * k + i + 1))
        if j:
            E += [(n0 + (j - 1) * k + i, n0 + j * k + i) for i in range(k)]
        e = edge[np.argmin(np.abs(S[edge, 2] - zs[j]))]
        E.append((e, n0 + j * k + k - 1))                                                  # sewn at the front edge
    on = np.r_[np.ones(len(S), bool), np.zeros(len(Lv), bool)]
    at = np.r_[col, np.full(len(Lv), other)]                                              # the surface nearest it
    got = GM.bound_colours(R, np.array(E), on, at)
    assert (got[:n0] == col).all() and (got[n0:] == mine).all()
