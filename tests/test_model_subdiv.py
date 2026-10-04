"""Tests for mkmmd.model.subdiv: Catmull-Clark with creases as sparse linear operators."""
import json
import math
import os
import subprocess

import numpy as np
import pytest
import scipy.sparse as sp

from mkmmd.model.subdiv import Subdivided, crease_to_sharpness, subdivide, subdivide_uv

TOL = 1e-12


# ---------------------------------------------------------------------------------------------------- shapes
def cube():
    """Cube [-1, 1]^3, vertex index = bits (x>0) + 2 (y>0) + 4 (z>0), six quads CCW from outside."""
    v = np.array([[x, y, z] for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)])
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return v, f


def box_grid(n=2):
    """Surface of [-1, 1]^3 with n x n quads per side, CCW from outside (n = 2: 26 vertices, 24 quads)."""
    ids, verts, faces = {}, [], []

    def vid(p):
        if p not in ids:
            ids[p] = len(verts)
            verts.append([-1.0 + 2.0 * c / n for c in p])
        return ids[p]

    for a in range(3):
        u, w = (a + 1) % 3, (a + 2) % 3                      # e_u x e_w = e_a
        for sign in (0, n):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = [0, 0, 0]
                        p[a], p[u], p[w] = sign, i + du, j + dw
                        quad.append(vid(tuple(p)))
                    faces.append(quad if sign == n else quad[::-1])
    return np.array(verts), faces


def grid_patch(nx, ny, h=1.0):
    """Flat open patch in z = 0 centred on the origin, nx x ny quads of size h, normal +z."""
    v = np.array([[(i - nx / 2) * h, (j - ny / 2) * h, 0.0] for j in range(ny + 1) for i in range(nx + 1)])
    f = [[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
         for j in range(ny) for i in range(nx)]
    return v, f


def tetra():
    v = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    return v, [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]


def octahedron():
    v = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    f = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    return v, f


def prism(k=5, r=1.0, h=2.0):
    """Closed k-gon prism: two k-gon caps and k quads (an n-gon test input)."""
    ang = 2 * math.pi * np.arange(k) / k
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    v = np.concatenate([np.c_[ring, np.zeros(k)], np.c_[ring, np.full(k, h)]])
    f = [list(range(k))[::-1], list(range(k, 2 * k))]
    f += [[i, (i + 1) % k, k + (i + 1) % k, k + i] for i in range(k)]
    return v, f


def edges_of(faces):
    return sorted({(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])})


def edge_counts(faces):
    cnt = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            e = (min(a, b), max(a, b))
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def signed_volume(v, faces):
    tot = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            tot += float(np.dot(v[f[0]], np.cross(v[f[i]], v[f[i + 1]]))) / 6.0
    return tot


def rows(a, dec=9):
    """Rows of a point set sorted lexicographically (rounded), to compare sets of points."""
    a = np.round(np.asarray(a), dec) + 0.0
    return a[np.lexsort(a.T[::-1])]


def crease_edges(c, edges):
    return {e: c for e in edges}


# ---------------------------------------------------------------------------------------------------- reference
def ref_level(P, faces, sharp, keep_corners):
    """One slow, dictionary based Catmull-Clark level following the documented rules (independent of the module)."""
    nv = len(P)

    def key(a, b):
        return (a, b) if a < b else (b, a)

    edge_faces, inc_edges, inc_faces = {}, [[] for _ in range(nv)], [[] for _ in range(nv)]
    for fi, f in enumerate(faces):
        for j, a in enumerate(f):
            edge_faces.setdefault(key(a, f[(j + 1) % len(f)]), []).append(fi)
            inc_faces[a].append(fi)
    edges = sorted(edge_faces)
    eid = {e: nv + i for i, e in enumerate(edges)}
    for e in edges:
        inc_edges[e[0]].append((e, e[1]))
        inc_edges[e[1]].append((e, e[0]))
    FP = [np.mean([P[v] for v in f], axis=0) for f in faces]

    def sharpness(e):
        return math.inf if len(edge_faces[e]) != 2 else sharp.get(e, 0.0)

    EP = {}
    for e in edges:
        a, b = e
        mid = 0.5 * (P[a] + P[b])
        smooth = 0.25 * (P[a] + P[b] + FP[edge_faces[e][0]] + FP[edge_faces[e][1]]) if len(edge_faces[e]) == 2 else mid
        t = min(1.0, sharpness(e))
        EP[e] = (1 - t) * smooth + t * mid

    VP = []
    for v in range(nv):
        ie = inc_edges[v]
        if not ie or (keep_corners and len(inc_faces[v]) == 1):
            VP.append(P[v].copy())
            continue
        if len(inc_faces[v]) == 1:                                    # corner of an open patch, corners='smooth': the
            a_, b_ = ie[0][1], ie[1][1]                               # crease rule, pulled towards "stay" by the user
            w = min(1.0, sharp.get(ie[0][0], 0.0), sharp.get(ie[1][0], 0.0))   # creases of its two edges (OpenSubdiv)
            VP.append((1 - w) * (0.75 * P[v] + 0.125 * (P[a_] + P[b_])) + w * P[v])
            continue
        n = len(ie)
        f_avg = np.mean([FP[f] for f in inc_faces[v]], axis=0)
        r_avg = np.mean([0.5 * (P[e[0]] + P[e[1]]) for e, _ in ie], axis=0)
        smooth = (f_avg + 2 * r_avg + (n - 3) * P[v]) / n
        # rule now (edges with s > 0) and after this level (edges with s > 1, they survive the decay s -> s - 1)
        now = [(sharpness(e), o) for e, o in ie if sharpness(e) > 0]
        later = [(s, o) for s, o in now if s > 1]

        def rule(k):
            return "smooth" if k < 2 else ("crease" if k == 2 else "corner")

        def point(kind, edges_):
            if kind == "smooth":
                return smooth
            if kind == "corner":
                return P[v].copy()
            return 0.75 * P[v] + 0.125 * (P[edges_[0][1]] + P[edges_[1][1]])

        if rule(len(now)) == rule(len(later)):
            VP.append(point(rule(len(now)), now))
        else:                                                         # edges fade out: blend by their mean sharpness
            fading = [s for s, _ in now if s <= 1]
            w = sum(fading) / len(fading)
            VP.append(w * point(rule(len(now)), now) + (1 - w) * point(rule(len(later)), later))

    new_faces = []
    for fi, f in enumerate(faces):
        k = len(f)
        for j in range(k):
            new_faces.append([f[j], eid[key(f[j], f[(j + 1) % k])], nv + len(edges) + fi,
                              eid[key(f[j - 1], f[j])]])
    new_sharp = {}
    for e, s in sharp.items():
        if s > 0 and e in eid:
            for end in e:
                new_sharp[key(end, eid[e])] = max(0.0, s - 1.0)
    return VP + [EP[e] for e in edges] + FP, new_faces, new_sharp


def ref_subdivide(verts, faces, levels, sharp=None, keep_corners=True):
    P = [np.asarray(p, dtype=float) for p in verts]
    faces = [list(f) for f in faces]
    sharp = {(min(e), max(e)): float(s) for e, s in (sharp or {}).items()}
    for _ in range(levels):
        P, faces, sharp = ref_level(P, faces, sharp, keep_corners)
    return np.array(P), faces


# ---------------------------------------------------------------------------------------------------- basics
def test_crease_to_sharpness():
    assert crease_to_sharpness(0.0) == 0.0
    assert crease_to_sharpness(1.0) == 10.0
    assert crease_to_sharpness(0.5) == pytest.approx(2.5)
    assert crease_to_sharpness(2.0) == 10.0 and crease_to_sharpness(-1.0) == 0.0     # clamped to 0..1
    np.testing.assert_allclose(crease_to_sharpness(np.array([0.0, 0.1, 1.0])), [0.0, 0.1, 10.0])
    assert isinstance(crease_to_sharpness(0.3), float)


def test_cube_level1_matches_analytic_catmull_clark():
    v, f = cube()
    sd = subdivide(v, f, 1)
    assert isinstance(sd, Subdivided) and sd.levels == 1
    assert sd.verts.shape == (26, 3) and len(sd.faces) == 24 and all(len(q) == 4 for q in sd.faces)
    # vertex point: (F_avg + 2 R_avg + 0 P) / 3 = 5/9 of the corner coordinate; edge point (a+b+F1+F2)/4 = (3/4, 3/4, 0)
    # pattern; face point = centre of the face
    np.testing.assert_allclose(sd.verts[:8], v * 5.0 / 9.0, atol=TOL)
    edges = edges_of(f)
    assert len(edges) == 12
    for k, (a, b) in enumerate(edges):
        mid = 0.5 * (v[a] + v[b])                                   # the midpoint has one zero coordinate
        expect = 0.75 * np.where(mid == 0.0, 0.0, np.sign(mid))
        np.testing.assert_allclose(sd.verts[8 + k], expect, atol=TOL)
    for k, face in enumerate(f):
        np.testing.assert_allclose(sd.verts[20 + k], v[face].mean(axis=0), atol=TOL)
    np.testing.assert_array_equal(sd.origin, np.r_[np.arange(8), -np.ones(18, dtype=int)])
    # first quad: [vertex 0, edge point (0, 2), face point 0, edge point (1, 0)]
    assert sd.faces[0] == [0, 8 + edges.index((0, 2)), 20, 8 + edges.index((0, 1))]


def test_tetrahedron_level1_hand_values():
    v, f = tetra()
    sd = subdivide(v, f, 1)
    assert len(sd.faces) == 12 and sd.verts.shape == (14, 3)
    np.testing.assert_allclose(sd.verts[:4], v * 7.0 / 27.0, atol=TOL)           # vertex points (n = 3)
    ev = {tuple(np.round(p, 9)) for p in sd.verts[4:10]}
    assert ev == {(0.666666667, 0, 0), (-0.666666667, 0, 0), (0, 0.666666667, 0), (0, -0.666666667, 0),
                  (0, 0, 0.666666667), (0, 0, -0.666666667)}
    np.testing.assert_allclose(sd.verts[10:], [[1 / 3, 1 / 3, -1 / 3], [1 / 3, -1 / 3, 1 / 3],
                                               [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -1 / 3, -1 / 3]], atol=TOL)
    assert signed_volume(sd.verts, sd.faces) > 0


def test_cube_two_levels_closed_inside_and_round():
    v, f = cube()
    sd = subdivide(v, f, 2)
    assert len(sd.faces) == 4 ** 2 * 6 == 96 and sd.verts.shape == (98, 3)
    assert all(len(q) == 4 for q in sd.faces)
    assert set(edge_counts(sd.faces).values()) == {2}                    # watertight, manifold
    assert 98 - len(edge_counts(sd.faces)) + 96 == 2                       # Euler characteristic of a sphere
    assert np.abs(sd.verts).max() < 1.0                                    # inside the cube
    r = np.linalg.norm(sd.verts, axis=1)
    r1 = np.linalg.norm(subdivide(v, f, 1).verts, axis=1)
    assert r.max() / r.min() < r1.max() / r1.min() < math.sqrt(3.0) and r.max() / r.min() < 1.05   # more spherical
    assert signed_volume(sd.verts, sd.faces) > 0                           # outward winding
    assert np.array_equal(sd.verts, sd.S @ v)


def test_operators_reproduce_positions_and_rows_sum_to_one():
    rng = np.random.default_rng(1)
    v, f = cube()
    v = v + 0.1 * rng.standard_normal(v.shape)
    for levels in (1, 2, 3):
        sd = subdivide(v, f, levels, creases={(0, 1): 0.7, (1, 3): 0.2})
        assert sp.issparse(sd.S) and sp.issparse(sd.L) and sd.S.format == "csr" and sd.L.format == "csr"
        assert sd.S.shape == (sd.verts.shape[0], 8) and sd.L.shape == sd.S.shape
        np.testing.assert_allclose(sd.S @ v, sd.verts, atol=TOL)
        np.testing.assert_allclose(np.asarray(sd.S.sum(axis=1)).ravel(), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(sd.L.sum(axis=1)).ravel(), 1.0, atol=1e-12)
        assert sd.L.min() >= 0.0 and sd.L.max() <= 1.0 + 1e-15
        # the linear operator keeps cage vertices, edge points are the mean of two vertices, face points of the face
        np.testing.assert_allclose(sd.L[:8].toarray(), np.eye(8), atol=0)
        assert sd.apply(v).shape == sd.verts.shape and np.allclose(sd.apply(v), sd.verts)
        assert sd.apply(v[:, 0]).shape == (sd.verts.shape[0],)
        assert sd.n_cage == 8 and "levels=" in repr(sd)


def test_linear_operator_structure_level1():
    v, f = cube()
    sd = subdivide(v, f, 1)
    L = sd.L.toarray()
    edges = edges_of(f)
    for k, (a, b) in enumerate(edges):
        row = L[8 + k]
        assert row[a] == 0.5 and row[b] == 0.5 and np.count_nonzero(row) == 2
    for k, face in enumerate(f):
        row = L[20 + k]
        assert np.count_nonzero(row) == 4 and np.allclose(row[face], 0.25)
    w = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])                              # a single-bone weight
    out = sd.apply(w, attr=True)
    assert out.min() >= 0 and out.max() == 1.0 and out[0] == 1.0
    two = sd.apply(np.stack([w, 1 - w], axis=1), attr=True)
    np.testing.assert_allclose(two.sum(axis=1), 1.0, atol=1e-12)           # weights keep summing to 1


def test_levels_zero_and_input_types():
    v, f = cube()
    sd = subdivide(v, f, 0, uv=np.arange(48.0).reshape(24, 2))
    assert sd.levels == 0 and sd.faces == f and np.allclose(sd.verts, v) and sd.verts is not v
    assert np.array_equal(sd.S.toarray(), np.eye(8)) and np.array_equal(sd.L.toarray(), np.eye(8))
    assert sd.uv.shape == (24, 2) and sd.face_parent.tolist() == list(range(6))
    assert sd.map_edges([(0, 1), (3, 2)]) == [(0, 1), (3, 2)]
    # numpy faces / int vertices are fine
    sd2 = subdivide(v.astype(np.float32), np.array(f), 1)
    assert sd2.verts.dtype == np.float64 and len(sd2.faces) == 24
    assert all(type(i) is int for q in sd2.faces for i in q)
    # tuples of tuples, numpy integer keys and float32 crease values are accepted
    sd3 = subdivide(v, tuple(tuple(q) for q in f), 1, creases={(np.int64(0), np.int64(1)): np.float32(1.0)})
    assert np.array_equal(sd3.verts, subdivide(v, f, 1, creases={(0, 1): 1.0}).verts)
    with pytest.raises(ValueError, match="dict"):
        subdivide(v, f, 1, creases=[(0, 1)])


# ---------------------------------------------------------------------------------------------------- creases
def test_zero_crease_equals_plain():
    v, f = box_grid(2)
    plain = subdivide(v, f, 2)
    zero = subdivide(v, f, 2, creases={e: 0.0 for e in edges_of(f)[:7]})
    assert np.array_equal(plain.verts, zero.verts) and plain.faces == zero.faces
    assert plain.S.shape == zero.S.shape and abs(plain.S - zero.S).max() == 0.0


def test_single_crease_edge_is_midpoint_and_ends_stay_smooth():
    v, f = cube()
    plain = subdivide(v, f, 1).verts
    sd = subdivide(v, f, 1, creases={(7, 3): 1.0})                           # either vertex order
    k = 8 + edges_of(f).index((3, 7))
    diff = np.flatnonzero(np.abs(sd.verts - plain).max(axis=1) > 1e-12)
    assert diff.tolist() == [k]                                              # a crease that just ends is a dart: smooth
    np.testing.assert_allclose(sd.verts[k], [1, 1, 0], atol=TOL)             # midpoint instead of (3/4, 3/4, 0)
    rev = subdivide(v, f, 1, creases={(3, 7): 1.0})
    assert np.array_equal(rev.verts, sd.verts)
    both = subdivide(v, f, 1, creases={(3, 7): 0.0, (7, 3): 1.0})            # duplicates keep the larger value
    assert np.array_equal(both.verts, sd.verts)


def test_creased_face_loop_hand_values():
    v, f = cube()
    loop = [(4, 5), (5, 7), (6, 7), (4, 6)]                                   # boundary of the +z face
    sd = subdivide(v, f, 1, creases=crease_edges(1.0, loop))
    edges = edges_of(f)
    # corners of the loop follow the crease rule: 3/4 P + 1/8 (A + B) -> (3/4, 3/4, 1); others stay at 5/9
    for i in (4, 5, 6, 7):
        np.testing.assert_allclose(sd.verts[i], [0.75 * v[i][0], 0.75 * v[i][1], 1.0], atol=TOL)
    for i in (0, 1, 2, 3):
        np.testing.assert_allclose(sd.verts[i], v[i] * 5 / 9, atol=TOL)
    for e in loop:                                                           # loop edge points are midpoints
        np.testing.assert_allclose(sd.verts[8 + edges.index(e)], 0.5 * (v[e[0]] + v[e[1]]), atol=TOL)
    np.testing.assert_allclose(sd.verts[8 + edges.index((3, 7))], [0.75, 0.75, 0.0], atol=TOL)   # other edges smooth


def test_fractional_crease_blends_smooth_and_sharp():
    v, f = cube()
    loop = [(4, 5), (5, 7), (6, 7), (4, 6)]
    edges = edges_of(f)
    sd = subdivide(v, f, 1, creases=crease_edges(0.5, loop), raw_sharpness=True)              # sharpness 0.5
    c = math.sqrt(0.5 / 10.0)
    assert crease_to_sharpness(c) == pytest.approx(0.5)
    np.testing.assert_allclose(subdivide(v, f, 1, creases=crease_edges(c, loop)).verts, sd.verts, atol=TOL)
    # edge point (5, 7): half way between smooth (3/4, 0, 3/4) and the midpoint (1, 0, 1)
    np.testing.assert_allclose(sd.verts[8 + edges.index((5, 7))], [0.875, 0.0, 0.875], atol=TOL)
    # vertex 7: half way between the smooth point 5/9 and the crease point (3/4, 3/4, 1)
    np.testing.assert_allclose(sd.verts[7], [(5 / 9 + 0.75) / 2, (5 / 9 + 0.75) / 2, (5 / 9 + 1) / 2], atol=TOL)
    # one level is exactly linear in the sharpness while it stays below 1 (every vertex has at most 2 sharp edges)
    p0 = subdivide(v, f, 1).verts
    p1 = subdivide(v, f, 1, creases=crease_edges(1.0, loop)).verts
    for s in (0.1, 0.5, 0.9):
        ps = subdivide(v, f, 1, creases=crease_edges(s, loop), raw_sharpness=True).verts
        np.testing.assert_allclose(ps, (1 - s) * p0 + s * p1, atol=TOL)


def test_vertex_rules_follow_opensubdiv_when_edges_fade():
    """Hand values; each one was cross-checked against Blender's Subdivision Surface modifier (4.2.3 and 5.2.2)."""
    v, f = cube()
    smooth = subdivide(v, f, 1).verts
    # three fractional creases per corner blend smooth -> corner by the (mean) sharpness: (1 - s) 5/9 + s
    for s in (0.4, 0.9):
        sd = subdivide(v, f, 1, creases=crease_edges(s, edges_of(f)), raw_sharpness=True)
        np.testing.assert_allclose(sd.verts[:8], v * ((1 - s) * 5 / 9 + s), atol=TOL)
    # two creases, one lasting (s = 10) and one fading (s = 0.1): only the fading one counts, 10% crease / 90% smooth
    sd = subdivide(v, f, 1, creases={(4, 5): 10.0, (4, 6): 0.1}, raw_sharpness=True)
    crease_pt = np.array([-0.75, -0.75, 1.0])                                  # 3/4 P + 1/8 (v5 + v6)
    np.testing.assert_allclose(sd.verts[4], 0.9 * smooth[4] + 0.1 * crease_pt, atol=TOL)
    np.testing.assert_allclose(sd.verts[[5, 6]], smooth[[5, 6]], atol=TOL)    # the ends of a lone crease stay smooth
    # two fading creases blend by their mean sharpness (0.2 + 0.6) / 2 = 0.4
    sd = subdivide(v, f, 1, creases={(4, 5): 0.2, (4, 6): 0.6}, raw_sharpness=True)
    np.testing.assert_allclose(sd.verts[4], 0.6 * smooth[4] + 0.4 * crease_pt, atol=TOL)
    # two lasting creases (s > 1) are a full crease even if the sharpnesses differ
    sd = subdivide(v, f, 1, creases={(4, 5): 1.5, (4, 6): 7.0}, raw_sharpness=True)
    np.testing.assert_allclose(sd.verts[4], crease_pt, atol=TOL)
    # three creases (10, 3.6, 0.1): corner turning into a crease as the weakest one fades -> 10% corner, 90% crease
    sd = subdivide(v, f, 1, creases={(3, 7): 10.0, (5, 7): 3.6, (6, 7): 0.1}, raw_sharpness=True)
    np.testing.assert_allclose(sd.verts[7], 0.1 * v[7] + 0.9 * (0.75 * v[7] + 0.125 * (v[3] + v[5])), atol=TOL)
    # sharpness exactly 1 is still a full crease for this level
    sd = subdivide(v, f, 1, creases={(4, 5): 1.0, (4, 6): 1.0}, raw_sharpness=True)
    np.testing.assert_allclose(sd.verts[4], crease_pt, atol=TOL)


def top_loop(v, f):
    """The 8 boundary edges of the +z cap of `box_grid(2)`."""
    on_x = lambda e: all(abs(abs(v[i, 0]) - 1.0) < 1e-9 for i in e)
    on_y = lambda e: all(abs(abs(v[i, 1]) - 1.0) < 1e-9 for i in e)
    return [e for e in edges_of(f) if all(abs(v[i, 2] - 1.0) < 1e-9 for i in e) and (on_x(e) or on_y(e))]


def test_crease_value_between_plain_and_sharp():
    v, f = box_grid(2)
    loop = top_loop(v, f)
    assert len(loop) == 8
    vols = []
    for c in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        sd = subdivide(v, f, 2, creases=crease_edges(c, loop))
        vols.append(signed_volume(sd.verts, sd.faces))
    assert vols == sorted(vols) and vols[0] < vols[2] < vols[-1]            # a sharper crease keeps more of the box
    assert vols[4] == pytest.approx(vols[5])                                 # c = 0.5 (s = 2.5) is fully sharp for 2 levels
    assert vols[1] > vols[0]


def test_fully_creased_edge_loop_keeps_straight_crease_line():
    v, f = box_grid(2)                                                        # the cap's boundary: a loop of 8 edges
    loop = top_loop(v, f)
    sd = subdivide(v, f, 2, creases=crease_edges(1.0, loop))
    fine = sd.map_edges(loop)
    ids = sorted({i for e in fine for i in e})
    assert len(fine) == 8 * 4 and len(ids) == 32                              # a closed loop of 32 fine edges
    pts = sd.verts[ids]
    np.testing.assert_allclose(pts[:, 2], 1.0, atol=TOL)                      # the crease stays in the plane of the loop
    # the crease curve is the cubic B-spline of the square loop: exactly straight in the middle of each side
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for t in (-0.25, 0.0, 0.25):
            want = np.array([sx if sx else t, sy if sy else t, 1.0])
            assert np.abs(pts - want).max(axis=1).min() < 1e-12
    # the cap is flat and its edge is sharp: exactly the 9 x 9 cap vertices lie in z = 1, everything else below
    flat = np.flatnonzero(np.abs(sd.verts[:, 2] - 1.0) < 1e-12)
    assert len(flat) == 81
    assert np.delete(sd.verts[:, 2], flat).max() < 1.0 - 1e-3
    plain = subdivide(v, f, 2).verts
    assert np.abs(plain[ids] - pts).max() > 0.05                              # without the crease the edge rounds off


def test_fully_creased_cube_stays_a_cube():
    v, f = cube()
    sd = subdivide(v, f, 3, creases=crease_edges(1.0, edges_of(f)))
    assert sd.verts.shape == (6 * 4 ** 3 + 2, 3)
    np.testing.assert_allclose(np.abs(sd.verts).max(axis=1), 1.0, atol=1e-12)    # every vertex on the cube surface
    np.testing.assert_allclose(sd.verts[:8], v, atol=TOL)                        # corners stay
    assert signed_volume(sd.verts, sd.faces) == pytest.approx(8.0)


def test_semi_sharp_crease_decays_with_levels():
    v, f = cube()
    ring = edges_of(f)
    sd1 = subdivide(v, f, 3, creases=crease_edges(1.0, ring), raw_sharpness=True)      # sharpness 1: sharp for one level only
    sd_big = subdivide(v, f, 3, creases=crease_edges(1.0, ring))                       # sharpness 10: sharp throughout
    sd_plain = subdivide(v, f, 3)
    assert np.abs(sd_big.verts).max(axis=1).min() > 1 - 1e-12                           # sharp cube
    r1 = np.abs(sd1.verts).max(axis=1)
    assert r1.min() < 0.99 and r1.mean() < np.abs(sd_big.verts).max(axis=1).mean()      # softens after the first level
    assert r1.mean() > np.abs(sd_plain.verts).max(axis=1).mean()                        # but sharper than no crease


def test_symmetric_for_fractional_multi_crease_corners():
    """Vertices with three or more fractional creases must not depend on the order of their edges (mirror symmetry)."""
    v, f = cube()
    for s in (0.3, 0.9, 1.7):
        sd = subdivide(v, f, 3, creases=crease_edges(s, edges_of(f)), raw_sharpness=True)
        for perm in ((1, 0, 2), (2, 1, 0), (1, 2, 0)):
            for sign in ((1, 1, 1), (-1, 1, 1), (1, -1, 1)):
                moved = sd.verts[:, perm] * np.array(sign)                 # a cube symmetry maps the point set onto itself
                dist = np.linalg.norm(moved[:, None, :] - sd.verts[None, :, :], axis=2).min(axis=1)
                assert dist.max() < 1e-9, (s, perm, sign)


# ---------------------------------------------------------------------------------------------------- open patches
def test_flat_grid_stays_flat_with_boundary_unchanged():
    v, f = grid_patch(3, 3, 1.0)                                              # 4 x 4 vertices, 3 x 3 quads
    for levels in (1, 2):
        sd = subdivide(v, f, levels)
        h = 2.0 ** -levels
        n = 3 * 2 ** levels + 1
        assert sd.verts.shape == (n * n, 3)
        lattice = np.array([[(i * h - 1.5), (j * h - 1.5), 0.0] for j in range(n) for i in range(n)])
        np.testing.assert_allclose(rows(sd.verts, 9), rows(lattice, 9), atol=1e-9)   # the regular finer grid
        cnt = edge_counts(sd.faces)
        bnd = sorted({i for e, c in cnt.items() if c == 1 for i in e})
        assert len(bnd) == 4 * (n - 1)
        bp = sd.verts[bnd]
        np.testing.assert_allclose(np.maximum(np.abs(bp[:, 0]), np.abs(bp[:, 1])), 1.5, atol=1e-12)   # on the outline
        assert all(np.cross(sd.verts[q[1]] - sd.verts[q[0]], sd.verts[q[3]] - sd.verts[q[0]])[2] > 0 for q in sd.faces)
    # the four patch corners stay put
    np.testing.assert_allclose(subdivide(v, f, 2).verts[[0, 3, 12, 15]], v[[0, 3, 12, 15]], atol=TOL)


def test_open_patch_boundary_stays_on_boundary_for_bumpy_interior():
    rng = np.random.default_rng(5)
    v, f = grid_patch(4, 3, 0.5)
    inner = [i for i, p in enumerate(v) if abs(p[0]) < 1.0 - 1e-9 and abs(p[1]) < 0.75 - 1e-9]
    v[inner, 2] = 0.2 * rng.standard_normal(len(inner))
    sd = subdivide(v, f, 2)
    cnt = edge_counts(sd.faces)
    bnd = sorted({i for e, c in cnt.items() if c == 1 for i in e})
    bp = sd.verts[bnd]
    np.testing.assert_allclose(bp[:, 2], 0.0, atol=TOL)
    on = (np.abs(np.abs(bp[:, 0]) - 1.0) < 1e-12) | (np.abs(np.abs(bp[:, 1]) - 0.75) < 1e-12)
    assert on.all()
    assert np.abs(sd.verts[:, 2]).max() > 0.01                                # the interior is still smoothed, not flat


def test_corner_modes():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    f = [[0, 1, 2, 3]]
    sharp = subdivide(v, f, 1)                                                # corners='sharp' is the default
    np.testing.assert_allclose(sharp.verts[:4], v, atol=TOL)
    smooth = subdivide(v, f, 1, corners="smooth")
    np.testing.assert_allclose(smooth.verts[0], [0.125, 0.125, 0.0], atol=TOL)   # 3/4 P + 1/8 (A + B)
    np.testing.assert_allclose(smooth.verts[2], [1 - 0.125, 1 - 0.125, 0.0], atol=TOL)
    # a corner vertex with two faces is not a patch corner: it gets the crease rule in both modes
    v2, f2 = grid_patch(2, 1, 1.0)
    for mode in ("sharp", "smooth"):
        out = subdivide(v2, f2, 1, corners=mode).verts
        np.testing.assert_allclose(out[1], v2[1], atol=TOL)                   # middle bottom vertex: straight boundary
    # as in OpenSubdiv, user creases on BOTH edges of a patch corner pull it back towards "stay" by min(1, s_a, s_b)
    one = subdivide(v, f, 1, corners="smooth", creases={(0, 1): 0.5}, raw_sharpness=True)
    np.testing.assert_allclose(one.verts[0], [0.125, 0.125, 0.0], atol=TOL)
    both = subdivide(v, f, 1, corners="smooth", creases={(0, 1): 0.5, (0, 3): 0.8}, raw_sharpness=True)
    np.testing.assert_allclose(both.verts[0], [0.0625, 0.0625, 0.0], atol=TOL)
    full = subdivide(v, f, 1, corners="smooth", creases={(0, 1): 2.0, (0, 3): 5.0}, raw_sharpness=True)
    np.testing.assert_allclose(full.verts[0], [0.0, 0.0, 0.0], atol=TOL)
    kept = subdivide(v, f, 1, corners="sharp", creases={(0, 1): 0.5, (0, 3): 0.8}, raw_sharpness=True)
    np.testing.assert_allclose(kept.verts[0], [0.0, 0.0, 0.0], atol=TOL)
    with pytest.raises(ValueError):
        subdivide(v, f, 1, corners="round")


def test_boundary_modes():
    v, f = grid_patch(2, 2)
    with pytest.raises(NotImplementedError):
        subdivide(v, f, 1, boundary="smooth")
    with pytest.raises(ValueError):
        subdivide(v, f, 1, boundary="loop")


# ---------------------------------------------------------------------------------------------------- n-gons and triangles
def test_ngon_gives_n_quads():
    ang = 2 * math.pi * np.arange(5) / 5
    v = np.c_[np.cos(ang), np.sin(ang), np.zeros(5)]
    sd = subdivide(v, [[0, 1, 2, 3, 4]], 1)
    assert len(sd.faces) == 5 and all(len(q) == 4 for q in sd.faces)
    assert sd.verts.shape == (5 + 5 + 1, 3) and sd.face_parent.tolist() == [0] * 5
    np.testing.assert_allclose(sd.verts[10], 0.0, atol=1e-15)                  # face point = centroid
    np.testing.assert_allclose(sd.verts[:5], v, atol=TOL)                      # patch corners stay
    for q in sd.faces:
        a, b, c, d = (sd.verts[i] for i in q)
        assert np.cross(b - a, d - a)[2] > 0                                  # still counter-clockwise
    sd2 = subdivide(v, [[0, 1, 2, 3, 4]], 2)
    assert len(sd2.faces) == 20 and (sd2.face_parent == 0).all()


def test_closed_ngon_and_triangle_meshes():
    v, f = prism(5)
    sd = subdivide(v, f, 1)
    assert len(sd.faces) == 5 + 5 + 5 * 4 and all(len(q) == 4 for q in sd.faces)     # two pentagons -> 10 quads, 5 quads -> 20
    assert set(edge_counts(sd.faces).values()) == {2} and signed_volume(sd.verts, sd.faces) > 0
    vol_level2 = signed_volume(subdivide(v, f, 2).verts, subdivide(v, f, 2).faces)
    assert 0 < vol_level2 < signed_volume(v, f)
    for name, (v, f) in (("tetra", tetra()), ("octa", octahedron())):
        for levels in (1, 2):
            sd = subdivide(v, f, levels)
            assert len(sd.faces) == 3 * len(f) * 4 ** (levels - 1), name
            assert all(len(q) == 4 for q in sd.faces)
            cnt = edge_counts(sd.faces)
            assert set(cnt.values()) == {2} and sd.verts.shape[0] - len(cnt) + len(sd.faces) == 2
            assert signed_volume(sd.verts, sd.faces) > 0
    # mixed triangles, quads and a pentagon in one closed mesh: a cube with one side replaced by a triangle fan
    v, f = cube()
    c = np.r_[v, [[0.0, 0.0, 1.0]]]
    mixed = [q for i, q in enumerate(f) if i != 1] + [[4, 5, 8], [5, 7, 8], [7, 6, 8], [6, 4, 8]]
    sd = subdivide(c, mixed, 2)
    assert set(edge_counts(sd.faces).values()) == {2} and signed_volume(sd.verts, sd.faces) > 0


# ---------------------------------------------------------------------------------------------------- bookkeeping
def test_face_parent():
    v, f = cube()
    sd = subdivide(v, f, 2)
    assert sd.face_parent.shape == (96,) and sorted(set(sd.face_parent.tolist())) == list(range(6))
    assert np.bincount(sd.face_parent).tolist() == [16] * 6
    # each fine face is nearest to the centre of its parent face
    fv = np.array([sd.verts[q].mean(axis=0) for q in sd.faces])
    centres = np.array([v[q].mean(axis=0) for q in f])
    nearest = np.argmin(((fv[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    np.testing.assert_array_equal(nearest, sd.face_parent)
    sd1 = subdivide(v, f, 1)
    for p in range(6):                                                         # the four children share their face point
        kids = [q for q, par in zip(sd1.faces, sd1.face_parent) if par == p]
        assert len(kids) == 4 and all(q[2] == 20 + p for q in kids)
    face_mat = np.array([0, 1, 1, 2, 0, 2])
    assert (face_mat[sd.face_parent] == face_mat[nearest]).all()
    # n-gons and triangles map every child to their face
    v5, f5 = prism(5)
    assert np.bincount(subdivide(v5, f5, 1).face_parent).tolist() == [5, 5, 4, 4, 4, 4, 4]


def test_map_edges():
    v, f = cube()
    edges = edges_of(f)
    for levels in (1, 2, 3):
        sd = subdivide(v, f, levels)
        fine = sd.map_edges([(0, 1), (3, 1)])
        assert len(fine) == 2 * 2 ** levels
        cnt = edge_counts(sd.faces)
        assert all((min(a, b), max(a, b)) in cnt for a, b in fine)             # real edges of the fine mesh
        first, second = fine[:2 ** levels], fine[2 ** levels:]
        for chain, (i, j) in ((first, (0, 1)), (second, (3, 1))):
            assert chain[0][0] == i and chain[-1][1] == j                       # runs from i to j
            assert all(chain[k][1] == chain[k + 1][0] for k in range(len(chain) - 1))
            assert len({a for a, _ in chain} | {chain[-1][1]}) == 2 ** levels + 1
    # a creased cage edge: its level-1 edge point is the exact midpoint and is the joint of the mapped chain
    sd = subdivide(v, f, 1, creases={(0, 1): 1.0})
    chain = sd.map_edges([(1, 0)])
    assert len(chain) == 2 and chain[0][0] == 1 and chain[1][1] == 0 and chain[0][1] == chain[1][0]
    np.testing.assert_allclose(sd.verts[chain[0][1]], [0.0, -1.0, -1.0], atol=TOL)
    assert len(subdivide(v, f, 2, creases={(0, 1): 1.0}).map_edges([(1, 0)])) == 4
    # all fine edges of all cage edges together: every fine edge belongs to exactly one chain or none, no duplicates
    allc = subdivide(v, f, 2).map_edges(edges + [(b, a) for a, b in edges[:3]])
    assert len(allc) == len(edges) * 4 == len({(min(a, b), max(a, b)) for a, b in allc})
    assert subdivide(v, f, 1).map_edges([]) == []
    with pytest.raises(ValueError):
        subdivide(v, f, 1).map_edges([(0, 7)])                                   # a diagonal, not an edge
    with pytest.raises(ValueError):
        subdivide(v, f, 1).map_edges([(0, 99)])


def test_morph_offsets_are_linear():
    rng = np.random.default_rng(7)
    v, f = box_grid(2)
    creases = {e: c for e, c in zip(edges_of(f)[::3], (1.0, 0.3, 0.6, 0.9, 0.1, 0.45, 1.0, 0.2))}
    off = 0.05 * rng.standard_normal(v.shape)
    for levels in (1, 2, 3):
        sd = subdivide(v, f, levels, creases=creases)
        moved = subdivide(v + off, f, levels, creases=creases)
        np.testing.assert_allclose(moved.verts, sd.verts + sd.apply(off), atol=1e-12)
        np.testing.assert_allclose(sd.apply(off), moved.verts - sd.verts, atol=1e-12)
        assert moved.faces == sd.faces
    # and uniform translations stay translations (rows sum to 1)
    sd = subdivide(v, f, 2, creases=creases)
    np.testing.assert_allclose(sd.apply(np.ones((len(v), 3)) * [1, 2, 3]), np.ones((len(sd.verts), 3)) * [1, 2, 3], atol=1e-12)


def test_matches_slow_reference_on_random_creases():
    rng = np.random.default_rng(11)
    cases = []
    v, f = box_grid(2)
    v = v + 0.07 * rng.standard_normal(v.shape)
    cases.append(("box", v, f))
    v, f = grid_patch(4, 3, 0.5)
    v[:, 2] = 0.2 * rng.standard_normal(len(v))
    f = [q for i, q in enumerate(f) if i not in (5, 6)]                          # a hole -> more boundary vertex kinds
    cases.append(("patch with hole", v, f))
    v, f = prism(5)
    cases.append(("prism", v + 0.05 * rng.standard_normal(v.shape), f))
    v, f = octahedron()
    cases.append(("octa", v + 0.05 * rng.standard_normal(v.shape), f))
    v, f = cube()
    c = np.r_[v, [[0.0, 0.0, 1.0]]]
    cases.append(("mixed", c + 0.05 * rng.standard_normal(c.shape),
                  [q for i, q in enumerate(f) if i != 1] + [[4, 5, 8], [5, 7, 8], [7, 6, 8], [6, 4, 8]]))
    cases.append(("non-manifold fin", np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, -1, 1], [1, -1, 1],
                                                [0, -1, -1], [1, -1, -1], [5, 5, 5]]),
                  [[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 7, 6]]))                  # three quads on one edge + a loose vertex
    for name, v, f in cases:
        edges = edges_of(f)
        for trial in range(3):
            pick = rng.random(len(edges)) < (0.5 if trial else 0.9)
            vals = rng.choice([0.05, 0.2, 0.3, 0.45, 0.6, 1.0], size=len(edges))
            creases = {e: float(c) for e, c, p in zip(edges, vals, pick) if p}
            for levels in (1, 2, 3):
                if len(f) > 40 and levels == 3:
                    continue
                for corners in ("sharp", "smooth"):
                    sd = subdivide(v, f, levels, creases=creases, corners=corners)
                    want_v, want_f = ref_subdivide(v, f, levels, {e: crease_to_sharpness(c) for e, c in creases.items()},
                                                   keep_corners=corners == "sharp")
                    assert sd.faces == want_f, (name, trial, levels)
                    np.testing.assert_allclose(sd.verts, want_v, atol=1e-11, err_msg=f"{name} {trial} {levels} {corners}")


def test_isolated_vertex_and_nonmanifold_edge_do_not_break():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, -1, 1], [1, -1, 1], [9, 9, 9]])
    f = [[0, 1, 2, 3], [1, 0, 4, 5]]
    sd = subdivide(v, f, 2)
    assert np.isfinite(sd.verts).all() and np.isfinite(sd.S.toarray()).all()
    np.testing.assert_allclose(sd.verts[6], [9, 9, 9], atol=TOL)                  # the loose vertex stays where it is
    assert len(sd.faces) == 32


# ---------------------------------------------------------------------------------------------------- UV
def uv_expected(faces, uv):
    """Straight loop version of one face-varying linear level (corner order of the documented quads)."""
    out, c0 = [], 0
    for face in faces:
        k = len(face)
        loc = uv[c0:c0 + k]
        mean = loc.mean(axis=0)
        for j in range(k):
            out += [loc[j], 0.5 * (loc[j] + loc[(j + 1) % k]), mean, 0.5 * (loc[j - 1] + loc[j])]
        c0 += k
    return np.array(out)


def test_uv_face_varying_linear():
    rng = np.random.default_rng(3)
    v, f = cube()
    c = np.r_[v, [[0.0, 0.0, 1.0]]]
    f = [q for i, q in enumerate(f) if i != 1] + [[4, 5, 8], [5, 7, 8], [7, 6, 8], [6, 4, 8]]   # quads + triangles
    uv = rng.random((sum(len(q) for q in f), 2))
    sd = subdivide(c, f, 1, uv=uv)
    assert sd.uv.shape == (4 * sum(len(q) for q in f), 2)
    np.testing.assert_allclose(sd.uv, uv_expected(f, uv), atol=TOL)
    q2 = [[int(i) for i in q] for q in sd.faces]
    sd2 = subdivide(c, f, 2, uv=uv)
    np.testing.assert_allclose(sd2.uv, uv_expected(q2, sd.uv), atol=TOL)
    np.testing.assert_allclose(subdivide_uv(f, uv, 2), sd2.uv, atol=TOL)
    np.testing.assert_allclose(subdivide_uv(f, uv, 0), uv)
    assert subdivide_uv(f, uv, 0) is not uv
    # the cage corner UVs survive untouched at corner 0 of every first-level quad
    np.testing.assert_allclose(sd.uv.reshape(-1, 4, 2)[:, 0], uv, atol=0)


def test_uv_seam_does_not_blur():
    """Two quads share an edge but their UV islands are apart: the edge point gets a different UV in each face."""
    v = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]])
    f = [[0, 1, 4, 3], [1, 2, 5, 4]]
    uv = np.array([[0.0, 0], [0.5, 0], [0.5, 1], [0, 1],                          # island A
                   [0.6, 0], [1.0, 0], [1.0, 1], [0.6, 1]])                       # island B (shared vertices 1, 4 differ)
    sd = subdivide(v, f, 1, uv=uv)
    ep = 6 + edges_of(f).index((1, 4))                                            # the edge point on the seam
    corners = [(fi, k) for fi, q in enumerate(sd.faces) for k, i in enumerate(q) if i == ep]
    got = {sd.face_parent[fi]: sd.uv[4 * fi + k] for fi, k in corners}
    np.testing.assert_allclose(got[0], [0.5, 0.5], atol=TOL)
    np.testing.assert_allclose(got[1], [0.6, 0.5], atol=TOL)
    # UVs at the seam vertices are not averaged: vertex 1 keeps (0.5, 0) in island A and (0.6, 0) in island B
    for fi, q in enumerate(sd.faces):
        if q[0] == 1:
            np.testing.assert_allclose(sd.uv[4 * fi], [0.5, 0.0] if sd.face_parent[fi] == 0 else [0.6, 0.0], atol=TOL)
    # a flat uniform grid keeps its positions, so UV = (x, y) stays equal to the fine vertex position
    gv, gf = grid_patch(3, 3, 1.0)
    guv = gv[np.array(gf).ravel(), :2]
    gs = subdivide(gv, gf, 2, uv=guv)
    np.testing.assert_allclose(gs.uv, gs.verts[np.array(gs.faces).ravel(), :2], atol=1e-12)


def test_uv_errors():
    v, f = cube()
    with pytest.raises(ValueError):
        subdivide(v, f, 1, uv=np.zeros((23, 2)))
    with pytest.raises(ValueError):
        subdivide_uv(f, np.zeros((5, 2)), 1)
    assert subdivide(v, f, 1).uv is None


# ---------------------------------------------------------------------------------------------------- validation, size
def test_input_validation():
    v, f = cube()
    with pytest.raises(ValueError, match="not an edge"):
        subdivide(v, f, 1, creases={(0, 7): 1.0})                                  # a space diagonal
    with pytest.raises(ValueError):
        subdivide(v, f, 1, creases={(0, 99): 1.0})
    with pytest.raises(ValueError):
        subdivide(v, f, 1, creases={(0, 1): float("nan")})
    with pytest.raises(ValueError, match="outside"):
        subdivide(v, f + [[0, 1, 9]], 1)
    with pytest.raises(ValueError, match="at least 3"):
        subdivide(v, f + [[0, 1]], 1)
    with pytest.raises(ValueError, match="repeats a vertex"):
        subdivide(v, f + [[0, 1, 2, 1]], 1)
    with pytest.raises(ValueError):
        subdivide(v, f, -1)
    with pytest.raises(ValueError):
        subdivide(v, [], 1)
    with pytest.raises(ValueError):
        subdivide(v[:, 0], f, 1)
    sd = subdivide(v, f, 1)
    with pytest.raises(ValueError):
        sd.apply(np.zeros(7))
    with pytest.raises(ValueError):
        sd.apply(np.zeros((8, 3, 1)))
    assert subdivide(v, [], 0).verts.shape == (8, 3)


def test_reasonable_speed_on_a_larger_cage():
    v, f = grid_patch(40, 40, 0.1)
    v[:, 2] = 0.05 * np.sin(5 * v[:, 0]) * np.cos(4 * v[:, 1])
    sd = subdivide(v, f, 2, creases={(0, 1): 1.0}, uv=v[np.array(f).ravel(), :2])
    assert len(sd.faces) == 40 * 40 * 16 and sd.verts.shape[0] == 161 * 161
    assert sd.S.nnz < 161 * 161 * 30
    assert np.isfinite(sd.verts).all() and sd.verts[:, 2].std() > 0


# ---------------------------------------------------------------------------------------------------- vs Blender
BLENDER_SCRIPT = '''
import bpy, json, sys
import numpy as np
inp, outp = sys.argv[sys.argv.index("--") + 1:][:2]
out = {}
for case in json.load(open(inp)):
    me = bpy.data.meshes.new(case["name"])
    me.from_pydata([tuple(v) for v in case["verts"]], [], [list(f) for f in case["faces"]])
    me.update()
    if case["creases"]:
        index = {tuple(sorted(e.vertices)): e.index for e in me.edges}
        attr = me.attributes.new("crease_edge", "FLOAT", "EDGE")
        for a, b, c in case["creases"]:
            attr.data[index[(min(a, b), max(a, b))]].value = c
    ob = bpy.data.objects.new(case["name"], me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("ss", "SUBSURF")
    mod.levels = mod.render_levels = case["levels"]
    mod.use_limit_surface = False
    mod.use_creases = True
    mod.boundary_smooth = "PRESERVE_CORNERS" if case["corners"] == "sharp" else "ALL"
    em = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    co = np.empty(len(em.vertices) * 3)
    em.vertices.foreach_get("co", co)
    out[case["name"]] = co.reshape(-1, 3).tolist()
json.dump(out, open(outp, "w"))
'''


@pytest.mark.skipif(not os.environ.get("MK_BLENDER_BIN"), reason="set MK_BLENDER_BIN=/path/to/blender to compare with Blender")
def test_matches_blenders_subdivision_surface_modifier(tmp_path):
    """Opt-in: the positions equal Blender's Subdivision Surface modifier (OpenSubdiv) on random creased cages.

    Runs the binary in MK_BLENDER_BIN in the background with all user directories inside tmp_path (and without
    --factory-startup, which would wipe add-on wheels in the real config directory)."""
    rng = np.random.default_rng(2024)
    cases = []

    def add(name, v, f, creases, levels, corners="sharp"):
        cases.append({"name": name, "verts": np.asarray(v).tolist(), "faces": [list(map(int, q)) for q in f],
                      "creases": [[int(a), int(b), float(c)] for (a, b), c in creases.items()], "levels": levels,
                      "corners": corners})

    v, f = cube()
    add("cube plain", v, f, {}, 3)
    add("cube fractional corners", v, f, crease_edges(0.2, edges_of(f)), 3)
    shapes = {"box": box_grid(2), "octa": octahedron(), "prism": prism(5), "patch": grid_patch(4, 3, 0.5),
              "patch hole": (grid_patch(4, 3, 0.5)[0], [q for i, q in enumerate(grid_patch(4, 3, 0.5)[1]) if i != 5])}
    for name, (sv, sf) in shapes.items():
        sv = sv + 0.05 * rng.standard_normal(sv.shape)
        edges = edges_of(sf)
        for trial in range(3):
            vals = np.where(rng.random(len(edges)) < 0.5, rng.random(len(edges)), rng.choice([0.1, 0.3, 0.6, 1.0], len(edges)))
            creases = {e: float(c) for e, c, p in zip(edges, vals, rng.random(len(edges)) < 0.5) if p}
            for corners in (("sharp", "smooth") if name.startswith("patch") else ("sharp",)):
                add(f"{name} {trial} {corners}", sv, sf, creases, 2 + (trial == 0), corners)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "run.py").write_text(BLENDER_SCRIPT)
    env = dict(os.environ, BLENDER_USER_CONFIG=str(tmp_path / "cfg"), BLENDER_USER_SCRIPTS=str(tmp_path / "scripts"),
               BLENDER_USER_EXTENSIONS=str(tmp_path / "ext"))
    run = subprocess.run([os.environ["MK_BLENDER_BIN"], "-b", "--python", str(tmp_path / "run.py"), "--",
                          str(tmp_path / "cases.json"), str(tmp_path / "out.json")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert (tmp_path / "out.json").exists(), run.stdout[-2000:] + run.stderr[-2000:]
    got = json.loads((tmp_path / "out.json").read_text())
    for case in cases:
        creases = {(a, b): c for a, b, c in case["creases"]}
        mine = subdivide(np.array(case["verts"]), case["faces"], case["levels"], creases=creases, corners=case["corners"]).verts
        theirs = np.array(got[case["name"]])
        assert theirs.shape == mine.shape, case["name"]
        dist = np.linalg.norm(mine[:, None, :] - theirs[None, :, :], axis=2)         # vertex order differs: compare as sets
        assert max(dist.min(axis=0).max(), dist.min(axis=1).max()) < 1e-5, case["name"]
