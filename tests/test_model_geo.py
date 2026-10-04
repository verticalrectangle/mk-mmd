"""Tests for mkmmd.model.geo: the shared geometry toolkit of the model part builders (numpy only, no Blender)."""
import inspect
import re
import time

import numpy as np
import pytest

from mkmmd.model import geo as G
from mkmmd.model import part


# ---------------------------------------------------------------------------------------------- helpers
def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def assert_rotation(R, atol=1e-12):
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=atol)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def uv_signed_areas(g):
    """Shoelace area of every face in UV space (positive = counter-clockwise = not mirrored)."""
    out, o = [], 0
    for f in g.faces:
        u = g.uv[o:o + len(f)]
        o += len(f)
        out.append(0.5 * np.sum(u[:, 0] * np.roll(u[:, 1], -1) - np.roll(u[:, 0], -1) * u[:, 1]))
    return np.array(out)


def corner_map(g):
    """{(face index, vertex index): uv} - what a face corner looks like independent of corner order."""
    out, o = {}, 0
    for i, f in enumerate(g.faces):
        for k, v in enumerate(f):
            out[(i, v)] = tuple(np.round(g.uv[o + k], 12))
        o += len(f)
    return out


def solid(g, vol=None, rel=1e-6, convex=True, uv_ccw=True):
    """A closed, manifold, consistently wound, outward mesh with valid UVs."""
    c = G.check(g)
    assert G.is_closed(g.faces)
    assert c["bad_indices"] == 0
    assert c["boundary_edges"] == 0 and c["non_manifold_edges"] == 0 and c["inconsistent_edges"] == 0
    assert c["n_degenerate"] == 0 and c["unused_verts"] == 0
    assert c["uv_ok"] and c["mat_ok"]
    assert c["signed_volume"] > 0
    if vol is not None:
        assert c["signed_volume"] == pytest.approx(vol, rel=rel)
    if convex:
        out = np.einsum("ij,ij->i", G.face_centers(g.verts, g.faces) - g.verts.mean(axis=0),
                        G.face_normals(g.verts, g.faces))
        assert (out > 0).all()
    if g.uv is not None:
        assert g.uv.shape == (g.n_corners, 2)
        if uv_ccw:
            assert (uv_signed_areas(g) > -1e-12).all()


def ring_stack(zs, sides=8, r=0.5):
    t = 2 * np.pi * np.arange(sides) / sides
    return np.array([np.stack([r * np.cos(t), r * np.sin(t), np.full(sides, z)], axis=1) for z in zs])


def polygon_area(sides, r=1.0):
    return 0.5 * sides * r * r * np.sin(2 * np.pi / sides)


def line(n, length=1.0, axis=0):
    p = np.zeros((n, 3))
    p[:, axis] = np.linspace(0.0, length, n)
    return p


# ---------------------------------------------------------------------------------------------- math
def test_unit_is_zero_safe():
    u = G.unit([[3, 0, 4], [0, 0, 0], [1e-15, 0, 0]])
    assert np.allclose(u[0], [0.6, 0, 0.8]) and not u[1].any() and not u[2].any()
    assert np.isfinite(G.unit(np.zeros((4, 3)))).all()
    assert G.unit([0.0, 2.0, 0.0]).tolist() == [0.0, 1.0, 0.0]
    assert G.unit(np.ones((2, 5, 3))).shape == (2, 5, 3)


def test_axis_rotations_and_rodrigues():
    a = 0.7
    assert np.allclose(G.rot_x(a) @ [0, 1, 0], [0, np.cos(a), np.sin(a)])
    assert np.allclose(G.rot_y(a) @ [0, 0, 1], [np.sin(a), 0, np.cos(a)])
    assert np.allclose(G.rot_z(a) @ [1, 0, 0], [np.cos(a), np.sin(a), 0])
    for f, ax in ((G.rot_x, (1, 0, 0)), (G.rot_y, (0, 1, 0)), (G.rot_z, (0, 0, 1))):
        assert np.allclose(f(a), G.rot_axis(ax, a))
    R = G.rot_axis([1, 2, 3], 1.1)
    assert_rotation(R)
    assert np.allclose(R @ G.unit([1, 2, 3]), G.unit([1, 2, 3]))
    assert G.rot_z(np.linspace(0, 1, 5)).shape == (5, 3, 3)
    assert G.rot_axis([[0, 0, 1], [1, 0, 0]], [0.3, 0.4]).shape == (2, 3, 3)
    with pytest.raises(ValueError):
        G.rot_axis([0, 0, 0], 1.0)


def test_rot_between_all_cases():
    rng = np.random.default_rng(1)
    for _ in range(60):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = G.rot_between(a, b)
        assert_rotation(R)
        assert np.allclose(R @ G.unit(a), G.unit(b), atol=1e-12)
        ang = np.arccos(np.clip(G.unit(a) @ G.unit(b), -1, 1))     # shortest: rotation angle == angle between
        assert np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) == pytest.approx(ang, abs=1e-7)
    assert np.allclose(G.rot_between([1, 2, 3], [2, 4, 6]), np.eye(3))
    for a in ([1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 2, 3]):                         # antiparallel
        R = G.rot_between(a, -np.asarray(a, float))
        assert_rotation(R)
        assert np.allclose(R @ G.unit(a), -G.unit(a), atol=1e-12)
    for eps in (1e-6, 1e-9, 1e-12):                                                 # almost antiparallel / parallel
        a = np.array([1.0, 0.0, 0.0])
        for b in (np.array([-1.0, eps, 0.0]), np.array([1.0, eps, eps])):
            R = G.rot_between(a, b)
            assert_rotation(R)
            assert np.allclose(R @ a, G.unit(b), atol=1e-8)
    assert np.allclose(G.rot_between([0, 0, 0], [1, 0, 0]), np.eye(3))


def test_frame_from_single_axis():
    rng = np.random.default_rng(2)
    for name in "xyz":
        for _ in range(20):
            d = rng.normal(size=3)
            F = G.frame_from(**{name: d})
            assert_rotation(F)
            assert np.allclose(F[:, "xyz".index(name)], G.unit(d))
    F = G.frame_from(z=(0, -1, 0))                    # look-at: z forward, y up, x = up x z
    assert np.allclose(F[:, 2], [0, -1, 0]) and np.allclose(F[:, 1], [0, 0, 1]) and np.allclose(F[:, 0], [1, 0, 0])
    F = G.frame_from(x=(1, 0, 0))                     # x given: z is the up-ish axis
    assert np.allclose(F[:, 2], [0, 0, 1])
    F = G.frame_from(x=(1, 0, 1))
    assert F[2, 2] > 0 and F[1, 2] == pytest.approx(0.0)
    F = G.frame_from(y=(0, 1, 0))
    assert np.allclose(F[:, 2], [0, 0, 1])
    F = G.frame_from(z=(0, 0, 1), up=(0, 1, 0))
    assert np.allclose(F[:, 1], [0, 1, 0])
    F = G.frame_from(z=(1, 0, 0), up=(0, 1, 0))       # custom up hint: y is up-ish
    assert F[1, 1] > 0.99


def test_frame_from_parallel_to_up_is_stable():
    assert np.allclose(G.frame_from(z=(0, 0, 1)), np.eye(3))
    for name in "xyz":
        for s in (1.0, -1.0, 1e-9):
            for d in ((0, 0, s), (0, 1e-12, s), (1e-9, 0, s)):
                F = G.frame_from(**{name: d})
                assert np.isfinite(F).all()
                assert_rotation(F)
    F = G.frame_from(z=(0, 0, 1 + 1e-9), up=(0, 0, 5))
    assert np.allclose(F, np.eye(3))
    for up in ((0, 0, 0), (0, 0, 1)):                  # zero or parallel up hint: world fallback
        assert_rotation(G.frame_from(x=(0, 0, 1), up=up))


def test_frame_from_two_axes_and_errors():
    F = G.frame_from(z=(0, 0, 1), x=(1, 1, 0))
    assert np.allclose(F[:, 2], [0, 0, 1]) and np.allclose(F[:, 0], np.array([1, 1, 0]) / np.sqrt(2))
    assert np.allclose(F[:, 1], np.array([-1, 1, 0]) / np.sqrt(2))
    F = G.frame_from(x=(1, 0, 0), y=(0, 1, 1))         # x kept, y orthogonalised, z = x * y
    assert np.allclose(F[:, 0], [1, 0, 0]) and np.allclose(F[:, 1], np.array([0, 1, 1]) / np.sqrt(2))
    assert_rotation(F)
    F = G.frame_from(z=(0, 0, 1), y=(0, 2, 0.5))       # z kept first, then y projected
    assert np.allclose(F, np.eye(3))
    assert np.allclose(G.frame_from(z=(0, 0, 1), x=(0, 0, 3)), np.eye(3))   # parallel second axis ignored
    F = G.frame_from(z=(0, 0, 1), x=(1, 0, 0), y=(0, -1, 0))                # always right-handed
    assert np.allclose(F, np.eye(3))
    with pytest.raises(ValueError):
        G.frame_from()
    with pytest.raises(ValueError):
        G.frame_from(z=(0, 0, 0))


def test_euler_round_trip_random_rotations():
    rng = np.random.default_rng(3)
    Rs = np.array([random_rotation(rng) for _ in range(500)])
    E = G.euler_xyz(Rs)
    assert E.shape == (500, 3)
    assert np.abs(G.matrix_xyz(E) - Rs).max() < 1e-12
    assert np.abs(E[:, 1]).max() <= np.pi / 2 + 1e-12
    for _ in range(100):                                         # euler -> matrix -> euler inside the principal range
        e = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi)])
        assert np.allclose(G.euler_xyz(G.matrix_xyz(e)), e, atol=1e-9)
    assert G.euler_xyz(np.eye(3)).shape == (3,)
    assert np.allclose(G.euler_xyz(np.eye(3)), 0.0)


def test_euler_matrix_convention_is_blender_xyz():
    e = (0.4, -0.7, 1.9)
    assert np.allclose(G.matrix_xyz(e), G.rot_z(1.9) @ G.rot_y(-0.7) @ G.rot_x(0.4))
    assert np.allclose(G.matrix_xyz((0.5, 0, 0)), G.rot_x(0.5))
    assert G.matrix_xyz(np.zeros((4, 3))).shape == (4, 3, 3)


def test_euler_near_gimbal_lock_round_trips():
    rng = np.random.default_rng(4)
    for sign in (1.0, -1.0):
        for d in (0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 5e-3, 2e-2):
            for _ in range(10):
                rx, rz = rng.uniform(-np.pi, np.pi, 2)
                R = G.matrix_xyz((rx, sign * (np.pi / 2 - d), rz))
                assert np.abs(G.matrix_xyz(G.euler_xyz(R)) - R).max() < 1e-12
    R = G.rot_y(np.pi / 2) @ G.rot_x(0.3)                        # exact lock: rz = 0, rx carries the angle
    e = G.euler_xyz(R)
    assert e[2] == 0.0 and e[1] == pytest.approx(np.pi / 2) and e[0] == pytest.approx(0.3)
    R = G.rot_z(0.8) @ G.rot_y(-np.pi / 2) @ G.rot_x(0.1)
    assert np.allclose(G.matrix_xyz(G.euler_xyz(R)), R, atol=1e-12)


def test_euler_for_axis_and_euler_from_axes():
    rng = np.random.default_rng(5)
    for axis in "xyz":
        for _ in range(25):
            d = rng.normal(size=3)
            R = G.matrix_xyz(G.euler_for_axis(d, axis))
            assert np.allclose(R[:, "xyz".index(axis)], G.unit(d), atol=1e-12)
    e = G.euler_for_axis((0, -1, 0))
    assert isinstance(e, tuple) and len(e) == 3 and all(isinstance(v, float) for v in e)
    R = G.matrix_xyz(e)
    assert np.allclose(R @ [0, 0, 1], [0, -1, 0]) and np.allclose(R @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(G.matrix_xyz(G.euler_from_axes(x=(0, 1, 0), z=(1, 0, 0))), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(G.matrix_xyz(G.euler_from_axes(z=(0, 0, 1))), np.eye(3))
    for d in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1e-13, 1)):                # no NaN around the poles
        assert np.isfinite(G.euler_for_axis(d, "z")).all()
    with pytest.raises(ValueError):
        G.euler_for_axis((0, 0, 1), "w")


def test_capsule_between_axis_maps_onto_segment():
    rng = np.random.default_rng(6)
    for _ in range(30):
        a, b = rng.normal(size=3), rng.normal(size=3)
        kw = G.capsule_between(a, b, 0.05)
        assert set(kw) == {"location", "rotation", "size"}
        R = G.matrix_xyz(kw["rotation"])
        assert np.allclose(R @ [0, 0, 1], G.unit(b - a), atol=1e-12)
        assert np.allclose(kw["location"], (a + b) / 2)
        assert kw["size"] == pytest.approx((0.05, np.linalg.norm(b - a), 0.0))
    kw = G.capsule_between((0, 0, 0), (0, 0, 1), 0.1, inclusive=True)       # tips: height = length - 2r
    assert kw["size"] == pytest.approx((0.1, 0.8, 0.0)) and kw["location"] == pytest.approx((0, 0, 0.5))
    assert G.capsule_between((0, 0, 0), (0, 0, 0.1), 0.1, inclusive=True)["size"][1] == 0.0
    kw = G.capsule_between((0, 0, 1), (0, 0, 0), 0.1)                       # pointing down
    assert np.allclose(G.matrix_xyz(kw["rotation"]) @ [0, 0, 1], [0, 0, -1])
    kw = G.capsule_between((1, 2, 3), (1, 2, 3), 0.1)                       # zero length
    assert kw["rotation"] == (0.0, 0.0, 0.0) and kw["size"] == (0.1, 0.0, 0.0)
    assert all(isinstance(v, float) for v in kw["location"] + kw["rotation"] + kw["size"])
    body = part.RigidBody(name="c", bone="b", **G.capsule_between((0, 0, 0), (0, 0.2, 0.1), 0.03))
    assert body.shape == "capsule" and body.size[0] == 0.03


def test_transform_points_order_and_purity():
    P = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    orig = P.copy()
    out = G.transform_points(P, G.rot_z(np.pi / 2), t=(1, 1, 1), scale=(2, 1, 1))     # (p * scale) @ R.T + t
    assert np.allclose(out, [[1, 3, 1], [-1, 1, 1]])
    assert np.array_equal(P, orig)
    assert np.allclose(G.transform_points(P, scale=3.0), P * 3)
    assert G.transform_points(P) is not P and np.array_equal(G.transform_points(P), P)
    assert G.transform_points([[0, 0, 0]], t=(1, 2, 3)).shape == (1, 3)


# ---------------------------------------------------------------------------------------------- Geo container
def test_geo_container_basics():
    g = G.Geo([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.array([[0, 1, 2, 3]]))
    assert g.verts.dtype == np.float64 and g.faces == [[0, 1, 2, 3]] and type(g.faces[0][0]) is int
    assert (g.n_verts, g.n_faces, g.n_corners) == (4, 1, 4) and g.uv is None and g.face_mat is None
    h = g.copy()
    h.verts[0, 0] = 9.0
    h.faces[0][0] = 3
    assert g.verts[0, 0] == 0.0 and g.faces[0][0] == 0
    assert "n_verts=4" in repr(g)
    lo, hi = g.bbox()
    assert np.allclose(lo, 0) and np.allclose(hi, [1, 1, 0])
    e = G.Geo(np.zeros((0, 3)), [])
    assert e.n_verts == 0 and np.allclose(e.bbox()[0], 0)
    with pytest.raises(ValueError):
        G.Geo(np.zeros((3, 2)), [[0, 1, 2]])
    g2 = G.box().with_material(2)
    assert g2.face_mat.tolist() == [2] * 6
    g3 = G.plane().uv_scaled((2, 3), (0.5, 0.25))
    assert np.allclose(g3.uv.min(0), [0.5, 0.25]) and np.allclose(g3.uv.max(0), [2.5, 3.25])
    assert G.Geo(np.zeros((3, 3)), [[0, 1, 2]]).uv_scaled((2, 2)).uv is None


def test_flipped_reverses_faces_and_uv_corners():
    g = G.uv_sphere(1.0, segments=12, rings=6)
    h = g.flipped()
    assert h.faces == [f[::-1] for f in g.faces]
    assert G.signed_volume(h.verts, h.faces) == pytest.approx(-G.signed_volume(g.verts, g.faces))
    assert corner_map(h) == corner_map(g)                       # every corner keeps its UV value
    assert h.flipped().faces == g.faces and np.allclose(h.flipped().uv, g.uv)
    assert np.allclose(h.verts, g.verts) and h.verts is not g.verts


def test_mirrored_x_keeps_outward_and_uv_values():
    g = G.uv_sphere(1.0, segments=12, rings=6, center=(0.3, 0.2, 0.1), scale=(1, 1.2, 1))
    m = g.mirrored_x()
    assert np.allclose(m.verts, g.verts * [-1, 1, 1])
    assert G.signed_volume(m.verts, m.faces) == pytest.approx(G.signed_volume(g.verts, g.faces))
    assert corner_map(m) == corner_map(g)
    assert np.allclose(G.face_normals(m.verts, m.faces) * [-1, 1, 1], G.face_normals(g.verts, g.faces))
    assert (uv_signed_areas(m) < 0).all() and (uv_signed_areas(g) > 0).all()  # mirrored model, shared texture layout


def test_transformed_flips_winding_when_mirroring():
    g = G.uv_sphere(1.0, segments=10, rings=5)
    vol = G.signed_volume(g.verts, g.faces)
    for kwargs in (dict(scale=(1, 1, -1)), dict(scale=(-1, 1, 1)), dict(scale=-1.0),
                   dict(R=np.diag([-1.0, 1.0, 1.0])), dict(R=np.diag([1.0, -1.0, 1.0]), scale=(2, 2, 2)),
                   dict(R=np.diag([-1.0, 1.0, 1.0]), scale=(1, -1, 1))):       # two mirrors = proper again
        h = g.transformed(**kwargs)
        assert G.signed_volume(h.verts, h.faces) > 0
        assert G.check(h)["inconsistent_edges"] == 0
        assert corner_map(h) == corner_map(g)
    h = g.transformed(R=G.rot_x(0.4), t=(1, 2, 3), scale=(1, 2, 3))
    assert h.faces == g.faces and np.allclose(h.uv, g.uv)                      # proper transform: no flip
    assert G.signed_volume(h.verts, h.faces) == pytest.approx(vol * 6 * (1.0), rel=1e-9)
    t = g.translated((1, 2, 3))
    assert np.allclose(t.verts, g.verts + [1, 2, 3]) and t.faces == g.faces
    assert g.verts.min() < -0.9                                                  # original untouched


# ---------------------------------------------------------------------------------------------- curves
def test_arclength_and_resample():
    p = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0.0]])
    assert np.allclose(G.arclength(p), [0, 3, 7])
    assert np.allclose(G.arclength(np.array([[0.0, 0.0], [0.0, 2.0]])), [0, 2])
    q = np.array([[0, 0, 0], [0.1, 0, 0], [1.0, 0, 0.0]])                           # uneven vertices on a line
    r = G.resample(q, 11)
    assert np.allclose(r, line(11)) and r.shape == (11, 3)
    r = G.resample(p, 8)
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)                      # equal along the L, chords at the corner
    assert np.allclose(r[0], p[0]) and np.allclose(r[-1], p[-1]) and r.shape == (8, 3)
    assert np.allclose(d[[0, 1, 2]], 1.0) and np.allclose(G.arclength(r)[-1], 6.0, atol=1)   # 7 / 7 = 1 per step
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    r = G.resample(sq, 8, closed=True)                                              # corners and midpoints, no repeat
    assert r.shape == (8, 3) and np.allclose(r[0], [0, 0, 0]) and np.allclose(r[1], [0.5, 0, 0])
    assert np.allclose(r[3], [1, 0.5, 0]) and np.allclose(r[7], [0, 0.5, 0])
    assert np.allclose(G.resample(np.zeros((3, 3)), 4), 0.0) and G.resample(np.zeros((3, 3)), 4).shape == (4, 3)
    assert G.resample(p, 1).shape == (1, 3)
    with pytest.raises(ValueError):
        G.resample(p, 0)


def test_catmull_rom_interpolates_and_counts():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 1.0], [3, 0, 1]])
    c = G.catmull_rom(p, 8)
    assert c.shape == ((len(p) - 1) * 8 + 1, 3) and np.allclose(c[::8], p) and np.allclose(c[-1], p[-1])
    cc = G.catmull_rom(p, 5, closed=True)
    assert cc.shape == (len(p) * 5, 3) and np.allclose(cc[::5], p)
    assert G.catmull_rom(p[:2], 4).shape == (5, 3)
    two = G.catmull_rom(p[:2], 4)                                                    # two points: a straight segment
    assert np.allclose(two[:, 1:], 0) and (np.diff(two[:, 0]) > 0).all()
    one = G.catmull_rom(p[:1], 4)
    assert one.shape == (1, 3)
    dup = G.catmull_rom(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0.0]]), 6)
    assert np.isfinite(dup).all()
    # centripetal never loops back on uneven spacing (x stays monotonic); the uniform parametrisation does
    q = np.array([[0, 0, 0], [10, 0, 0], [10.2, 0.2, 0], [10.4, 0, 0], [20, 0, 0.0]])
    assert (np.diff(G.catmull_rom(q, 16, alpha=0.5)[:, 0]) >= -1e-12).all()
    assert (np.diff(G.catmull_rom(q, 16, alpha=0.0)[:, 0]) < -1e-3).any()
    c2d = G.catmull_rom(np.array([[0.0, 0], [1, 1], [2, 0]]), 4)
    assert c2d.shape == (9, 2)


def test_bezier():
    b = G.bezier((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), 9)
    assert b.shape == (9, 3) and np.allclose(b[0], [0, 0, 0]) and np.allclose(b[-1], [2, 1, 0])
    assert np.allclose(b[4], (np.array([0, 0, 0]) + 3 * np.array([1, 0, 0]) + 3 * np.array([1, 1, 0]) + [2, 1, 0]) / 8)
    assert G.bezier((0, 0), (1, 0), (1, 1), (2, 1)).shape == (16, 2)
    assert G.bezier((0, 0), (1, 0), (1, 1), (2, 1), 1).shape == (1, 2)


def frames_ok(T, N, B, max_step=0.2):
    for v in (T, N, B):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)
    assert np.abs(np.einsum("ij,ij->i", T, N)).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", T, B)).max() < 1e-9
    assert np.allclose(np.cross(N, B), T, atol=1e-9) and np.allclose(np.cross(T, N), B, atol=1e-9)   # right-handed
    step = np.arccos(np.clip(np.einsum("ij,ij->i", N[1:], N[:-1]), -1, 1))
    assert step.max() < max_step                                                                      # never flips


def test_rmf_frames_helix_and_straight_paths():
    t = np.linspace(0, 6 * np.pi, 200)
    helix = np.stack([np.cos(t), np.sin(t), 0.2 * t], axis=1)
    T, N, B = G.rmf_frames(helix)
    assert T.shape == N.shape == B.shape == (200, 3)
    frames_ok(T, N, B, 0.12)
    assert np.allclose(T[1:-1], G.unit(helix[2:] - helix[:-2]))                       # central differences
    assert np.allclose(N[0], G.unit(np.array([0, 0, 1.0]) - T[0][2] * T[0]))           # first normal from up = Z
    T, N, B = G.rmf_frames(helix, up=(1, 0, 0))
    assert N[0] @ [1, 0, 0] > 0.5
    for axis in range(3):                                                              # straight lines: constant frame
        for s in (1.0, -1.0):
            p = line(10, s, axis)
            T, N, B = G.rmf_frames(p)
            frames_ok(T, N, B, 1e-12)
            assert np.allclose(N, N[0]) and np.allclose(T, np.eye(3)[axis] * s)
    T, N, B = G.rmf_frames(line(10, 1, 2))                                              # tangent parallel to up -> X
    assert np.allclose(N[0], [1, 0, 0]) and np.allclose(B[0], [0, 1, 0])
    T, N, B = G.rmf_frames(line(10, -1, 2))
    assert np.allclose(N[0], [1, 0, 0]) and np.allclose(B[0], [0, -1, 0])
    T, N, B = G.rmf_frames(line(10, 1, 0), up=(1, 0, 0))                               # up parallel to tangent too
    assert np.isfinite(N).all() and abs(N[0] @ T[0]) < 1e-12


def test_rmf_frames_planar_curves_keep_the_plane_normal():
    t = np.linspace(0, np.pi, 80)
    arc_xz = np.stack([np.cos(t), np.zeros_like(t), np.sin(t)], axis=1)               # 180 degree turn in XZ
    T, N, B = G.rmf_frames(arc_xz, up=(0, 1, 0))
    assert np.allclose(N, [0, 1, 0], atol=1e-9)                                       # RMF of a planar curve
    T, N, B = G.rmf_frames(arc_xz)                                                     # up = Z, tangent starts ~ up
    frames_ok(T, N, B, 0.1)
    circle_xy = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    T, N, B = G.rmf_frames(circle_xy)
    assert np.allclose(N, [0, 0, 1], atol=1e-9) and np.allclose(B, np.cross(T, [0, 0, 1]))
    # a half-turn hook that starts along +X and bends up and over: the in-plane normal turns by 180 degrees, no flips
    hook = np.stack([np.sin(t), np.zeros_like(t), 1 - np.cos(t)], axis=1)
    T, N, B = G.rmf_frames(hook)
    frames_ok(T, N, B, 0.1)
    assert np.allclose(N[0], [0, 0, 1], atol=0.05) and np.allclose(N[-1], [0, 0, -1], atol=0.05)
    ang = np.arccos(np.clip(N @ N[0], -1, 1))
    assert (np.diff(ang) > -1e-9).all() and ang[-1] == pytest.approx(np.pi, abs=0.05)   # monotonic half turn


def test_rmf_frames_have_no_twist():
    t = np.linspace(0, 6 * np.pi, 200)
    curves = [np.stack([np.cos(t), np.sin(t), 0.2 * t], axis=1),
              np.stack([np.sin(2 * t / 6) + np.cos(3 * t / 6), np.cos(t / 3) * np.sin(t / 5), t / 6], axis=1)]
    for path in curves:
        T, N, B = G.rmf_frames(path)
        for i in range(len(path) - 1):
            predicted = G.rot_between(T[i], T[i + 1]) @ N[i]              # carry N along with the smallest rotation
            twist = np.arctan2(T[i + 1] @ np.cross(predicted, N[i + 1]), predicted @ N[i + 1])
            assert abs(twist) < 1e-3                                       # second order in the step: ~4e-5 here


def test_rmf_frames_sharp_bends_and_degenerate_points():
    uturn = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0.01, 0], [1, 0.01, 0], [0, 0.01, 0.0]])
    T, N, B = G.rmf_frames(uturn)
    assert np.isfinite(T).all() and np.isfinite(N).all()
    frames_ok(T, N, B, 1e-9)                                                           # planar: constant normal
    right_angles = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1.0]])
    frames_ok(*G.rmf_frames(right_angles), max_step=1.6)
    dup = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0.0]])          # coincident points
    T, N, B = G.rmf_frames(dup)
    assert np.isfinite(T).all() and np.isfinite(N).all()
    frames_ok(T, N, B, 1.0)
    T, N, B = G.rmf_frames(np.zeros((5, 3)))                                            # everything coincident
    assert np.isfinite(N).all() and np.allclose(T, [0, 0, 1])
    with pytest.raises(ValueError):
        G.rmf_frames(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        G.rmf_frames(np.zeros((4, 2)))


def test_rmf_frames_closed_loops_close_up():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    T, N, B = G.rmf_frames(circle, closed=True)
    frames_ok(T, N, B, 0.2)
    assert np.allclose(N, [0, 0, 1], atol=1e-9)
    assert np.allclose(T[0], G.unit(circle[1] - circle[-1]))                           # wrapped central difference
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)                                # a knot: transport has holonomy
    knot = np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)], axis=1)
    T, N, B = G.rmf_frames(knot, closed=True)
    frames_ok(T, N, B, 0.15)
    wrap = np.arccos(np.clip(N[-1] @ N[0], -1, 1))                                     # last -> first is a normal step
    typical = np.arccos(np.clip(np.einsum("ij,ij->i", N[1:], N[:-1]), -1, 1)).mean()
    assert wrap < 2.5 * typical
    with pytest.raises(ValueError):
        G.rmf_frames(knot[:2], closed=True)


# ---------------------------------------------------------------------------------------------- loft
def test_loft_open_tube_closed_with_caps_and_vertex_layout():
    rings = ring_stack([0, 0.5, 1])
    g = G.loft(rings)
    assert (g.n_verts, g.n_faces) == (24, 16)
    assert np.allclose(g.verts, rings.reshape(-1, 3))                    # vertex (ring r, point j) = r*nv+j
    c = G.check(g)
    assert c["boundary_edges"] == 16 and c["non_manifold_edges"] == 0 and not G.is_closed(g.faces)
    radial = G.face_centers(g.verts, g.faces)[:, :2]
    assert (np.einsum("ij,ij->i", G.face_normals(g.verts, g.faces)[:, :2], radial) > 0).all()   # outward
    assert all(len(f) == 4 for f in g.faces)
    area = polygon_area(8, 0.5)
    for cap, nv, nf in (("fan", 26, 32), ("ngon", 24, 18), (True, 24, 18)):
        h = G.loft(rings, cap_start=cap, cap_end=cap)
        assert (h.n_verts, h.n_faces) == (nv, nf)
        solid(h, vol=area * 1.0)
        n = G.face_normals(h.verts, h.faces)
        assert n[16, 2] < -0.99                                                          # the first cap faces the start
        assert n[-1, 2] > 0
    h = G.loft(rings, cap_start="fan", cap_end="fan")
    assert np.allclose(h.verts[24], [0, 0, 0]) and np.allclose(h.verts[25], [0, 0, 1])   # centres: start, end
    assert len(G.loft(rings, cap_end="ngon").faces[-1]) == 8
    h = G.loft(rings, cap_start="ngon")                                                 # one open end only
    assert G.check(h)["boundary_edges"] == 8
    with pytest.raises(ValueError):
        G.loft(rings, cap_start="bogus")


def test_loft_uv_layout_seams_and_options():
    rings = ring_stack([0, 0.5, 1])
    g = G.loft(rings)
    assert g.uv.shape == (g.n_corners, 2) == (64, 2)
    uv = g.uv.reshape(-1, 4, 2)
    assert np.allclose(uv[0], [[0, 0], [1 / 8, 0], [1 / 8, 0.5], [0, 0.5]])
    assert np.allclose(uv[7], [[7 / 8, 0], [1, 0], [1, 0.5], [7 / 8, 0.5]])             # the seam has its own u = 1
    assert np.allclose(uv[8:, :, 1].reshape(-1, 4)[0], [0.5, 0.5, 1, 1])
    assert np.allclose(g.uv.min(0), 0) and np.allclose(g.uv.max(0), 1)
    assert uv_signed_areas(g).min() > 0
    assert G.loft(rings, uv=False).uv is None
    o = G.loft(rings, closed=False)                                                     # open strip: no seam column
    assert o.n_faces == 14 and np.allclose(np.unique(o.uv[:, 0]), np.linspace(0, 1, 8))
    # arc-length u / v
    r2 = np.array([[[0, 0, z], [2, 0, z], [2, 1, z], [0, 1, z]] for z in (0, 0.25, 1)], float)
    a = G.loft(r2, u="arc", v="arc").uv.reshape(-1, 4, 2)
    assert np.allclose(a[0, :2, 0], [0, 2 / 6]) and np.allclose(a[1, :2, 0], [2 / 6, 3 / 6])
    assert np.allclose(a[3, :2, 0], [5 / 6, 1])
    assert np.allclose(a[0, :, 1], [0, 0, 0.25, 0.25]) and np.allclose(a[4, :, 1], [0.25, 0.25, 1, 1])
    # explicit arrays
    e = G.loft(rings, u=np.linspace(0, 2, 9), v=[0, 0.1, 1]).uv.reshape(-1, 4, 2)
    assert np.allclose(e[0], [[0, 0], [0.25, 0], [0.25, 0.1], [0, 0.1]]) and e[:, :, 0].max() == pytest.approx(2.0)
    tab = G.loft(rings, u=np.tile(np.linspace(0, 1, 9), (3, 1)) * np.array([[1], [2], [3]])).uv
    assert tab.shape == (64, 2) and tab[:, 0].max() == pytest.approx(3.0)
    for bad in (dict(u=np.linspace(0, 1, 8)), dict(v=[0, 1]), dict(u="nope"), dict(v="nope"), dict(u=np.zeros((2, 9)))):
        with pytest.raises(ValueError):
            G.loft(rings, **bad)


def test_loft_flip_loop_and_validation():
    rings = ring_stack([0, 0.5, 1])
    g = G.loft(rings, cap_start="fan", cap_end="ngon")
    f = G.loft(rings, cap_start="fan", cap_end="ngon", flip=True)
    assert G.signed_volume(f.verts, f.faces) == pytest.approx(-G.signed_volume(g.verts, g.faces))
    assert corner_map(f) == corner_map(g)
    # loop: last ring joins the first (torus-like), no caps
    a = np.linspace(0, 2 * np.pi, 12, endpoint=False)[:, None]                          # rings run around Z
    b = -2 * np.pi * np.arange(6)[None, :] / 6                          # CCW around the travel direction
    big = np.stack([(1 + 0.2 * np.cos(b)) * np.cos(a), (1 + 0.2 * np.cos(b)) * np.sin(a),
                    0.2 * np.sin(b) * np.ones_like(a)], axis=-1)
    torus = G.loft(big, loop=True)
    assert torus.n_faces == 12 * 6
    solid(torus, convex=False, uv_ccw=True)
    assert torus.uv.reshape(-1, 4, 2)[:, :, 1].max() == pytest.approx(1.0)               # the closing row has v = 1
    assert set(np.round(np.unique(torus.uv[:, 1]), 6)) == set(np.round(np.arange(13) / 12, 6))
    with pytest.raises(ValueError):
        G.loft(big, loop=True, cap_end="ngon")
    for bad in (np.zeros((3, 3)), np.zeros((1, 4, 3)), np.zeros((3, 2, 3)), np.zeros((2, 4, 2))):
        with pytest.raises(ValueError):
            G.loft(bad)
    assert G.loft(np.zeros((2, 2, 3)) + [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]]], closed=False).n_faces == 1


# ---------------------------------------------------------------------------------------------- sweep / tube
def test_tube_straight_volume_counts_uv():
    path = line(6, 1.0, 2)
    for caps, counts in (("ngon", (96, 82)), ("fan", (98, 112)), (None, (96, 80))):
        g = G.tube(path, 0.1, 16, cap_start=caps, cap_end=caps)
        assert (g.n_verts, g.n_faces) == counts
        if caps:
            solid(g, vol=polygon_area(16, 0.1) * 1.0)
    # uv: u around (own seam), v = arc length along the path
    uneven = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.3], [0, 0, 0.6], [0, 0, 1.0]])
    g = G.tube(uneven, 0.05, 6)
    uv = g.uv.reshape(-1, 4, 2)
    assert np.allclose(sorted(set(np.round(uv[:, :, 1].ravel(), 9))), [0, 0.1, 0.3, 0.6, 1.0])
    assert np.allclose(uv[0, :2, 0], [0, 1 / 6]) and np.allclose(uv[5, :2, 0], [5 / 6, 1])
    assert np.allclose(np.unique(uv[:, :, 0]), np.arange(7) / 6)
    assert G.tube(path, 0.1, 6, uv=False).uv is None
    # the profile starts along the frame normal N (up projected perpendicular to the path): path along +X -> N = +Z
    g = G.tube(line(3, 1, 0), 0.1, 8)
    assert np.allclose(g.verts[0], [0, 0, 0.1])
    assert np.allclose(G.tube(line(3, 1, 0), 0.1, 8, up=(0, 1, 0)).verts[0], [0, 0.1, 0])


def test_tube_radius_per_point_elliptical_and_pointed_tip():
    path = line(5, 1.0, 0)
    g = G.tube(path, [0.1, 0.1, 0.08, 0.05, 0.02], 8, cap_start="ngon", cap_end="ngon")
    solid(g)
    r = np.linalg.norm(g.verts[:40].reshape(5, 8, 3)[:, :, 1:], axis=2)
    assert np.allclose(r, np.array([0.1, 0.1, 0.08, 0.05, 0.02])[:, None])
    e = G.tube(path, np.tile([0.1, 0.05], (5, 1)), 16)                                  # (k, 2): elliptical section
    ys, zs = e.verts[:, 1], e.verts[:, 2]                      # sx along N (= Z), sy along B (-Y)
    assert np.abs(zs).max() == pytest.approx(0.1) and np.abs(ys).max() == pytest.approx(0.05)
    with pytest.raises(ValueError):
        G.tube(path, [0.1, 0.1], 8)
    # a radius of exactly 0 leaves a collapsed ring; weld turns it into a pointed tip with triangles
    tip = G.tube(path[:4], [0.1, 0.1, 0.05, 0.0], 8, cap_start="ngon")
    assert tip.n_verts == 4 * 8 and sorted(set(map(len, tip.faces))) == [4, 8]    # tip quads: triangles in disguise
    w, remap = G.weld(tip)
    assert w.n_verts == 3 * 8 + 1 and len(remap) == tip.n_verts
    assert len(set(remap[24:32].tolist())) == 1
    solid(w, convex=False)
    assert sorted(set(map(len, w.faces))) == [3, 4, 8]


def test_sweep_profile_scale_twist_and_open_profile():
    sq = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]]) * 0.05                          # CCW in (N, B)
    path = line(5, 2.0, 0)                                                              # T = +X, N = +Z, B = T x N = -Y
    g = G.sweep(sq, path, cap_start="ngon", cap_end="ngon")
    solid(g, vol=0.01 * 2.0)
    assert np.allclose(g.verts[0], [0, 0.05, 0.05])                                      # 0.05 N - 0.05 B
    # scale (k, 2): taper sx (along N) from 1 to 2 -> volume grows by 1.5, sy untouched
    sc = np.stack([np.linspace(1, 2, 5), np.ones(5)], axis=1)
    solid(G.sweep(sq, path, scale=sc, cap_start="ngon", cap_end="ngon"), vol=0.01 * 2.0 * 1.5)
    solid(G.sweep(sq, path, scale=np.linspace(1, 3, 5), cap_start="fan", cap_end="fan"), vol=0.01 * 2.0 * 13 / 3)
    # a constant twist is a roll offset of the profile (same volume, rotated vertices)
    h = G.sweep(sq, path, twist=np.pi / 2, cap_start="ngon", cap_end="ngon")
    solid(h, vol=0.01 * 2.0)
    assert np.allclose(h.verts[0], [0, -0.05, 0.05])                                    # (0.05, -0.05) -> (0.05, 0.05)
    # a progressive twist stays closed and outward
    solid(G.sweep(sq, path, twist=np.linspace(0, 1.5, 5), cap_start="ngon", cap_end="ngon"), uv_ccw=True)
    rect = np.array([[1, -0.2], [1, 0.2], [-1, 0.2], [-1, -0.2]]) * 0.1
    tw = G.sweep(rect, path, twist=np.linspace(0, np.pi / 2, 5))
    first, last = tw.verts[:4], tw.verts[-4:]
    assert np.abs(first[:, 2]).max() == pytest.approx(0.1) and np.abs(last[:, 1]).max() == pytest.approx(0.1)
    # open profile -> open strip: normals on the outer (CCW) side
    arc = np.stack([np.cos(np.linspace(-1, 1, 7)), np.sin(np.linspace(-1, 1, 7))], axis=1) * 0.1
    o = G.sweep(arc, path, closed_profile=False)
    assert (o.n_verts, o.n_faces) == (35, 24) and G.check(o)["boundary_edges"] == 2 * 6 + 2 * 4
    fn = G.face_normals(o.verts, o.faces)
    assert (fn[:, 2] > 0).all()                                                         # arc bulges towards +N = +Z
    for bad in (dict(scale=[1, 2]), dict(twist=[1, 2, 3]), dict(scale=np.ones((5, 3)))):
        with pytest.raises(ValueError):
            G.sweep(sq, path, **bad)
    with pytest.raises(ValueError):
        G.sweep(np.zeros((4, 3)), path)


def test_sweep_closed_path_makes_a_torus():
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    g = G.tube(ring, 0.2, 12, closed_path=True)
    assert (g.n_verts, g.n_faces) == (48 * 12, 48 * 12)
    solid(g, convex=False)
    exact = polygon_area(12, 0.2) * 2 * np.pi * 1.0                                      # Pappus on the 12-gon section
    assert G.signed_volume(g.verts, g.faces) == pytest.approx(exact, rel=0.01)
    assert g.uv.reshape(-1, 4, 2)[:, :, 1].max() == pytest.approx(1.0)                   # v closes with its own row
    with pytest.raises(ValueError):
        G.tube(ring, 0.2, 12, cap_end="fan", closed_path=True)
    ribbon_loop = G.sweep(G.circle_profile(4, 0.05), ring, closed_path=True)
    solid(ribbon_loop, convex=False)


def test_circle_profile_and_ccw():
    p = G.circle_profile(8, 2.0, start=0.1)
    assert p.shape == (8, 2) and np.allclose(np.linalg.norm(p, axis=1), 2.0)
    assert np.allclose(p[0], 2 * np.array([np.cos(0.1), np.sin(0.1)]))
    cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0.0]])
    fixed = G.ccw(cw)
    assert np.allclose(fixed, cw[::-1]) and np.allclose(G.ccw(fixed), fixed)
    assert G.ccw(cw) is not cw


# ---------------------------------------------------------------------------------------------- ribbon
def test_ribbon_front_normal_faces_up():
    t = np.linspace(0, 2, 25)
    wiggle = np.stack([t, 0.3 * np.sin(3 * t), np.zeros_like(t)], axis=1)                # horizontal, bending path
    g = G.ribbon(wiggle, 0.1)
    assert (g.n_verts, g.n_faces) == (50, 24)
    assert (G.face_normals(g.verts, g.faces)[:, 2] > 0.999).all()                        # default up = +Z
    down = G.ribbon(wiggle, 0.1, up=(0, 0, -1))
    assert (G.face_normals(down.verts, down.faces)[:, 2] < -0.999).all()
    # a vertical strip hanging down, facing -Y (towards the viewer in front of the character)
    hang = np.stack([np.zeros(10), np.zeros(10), -np.linspace(0, 1, 10)], axis=1)
    h = G.ribbon(hang, 0.2, up=(0, -1, 0))
    assert np.allclose(G.face_normals(h.verts, h.faces), [0, -1, 0], atol=1e-9)
    assert np.allclose(np.ptp(h.verts[:, 0]), 0.2)                                       # width lies along X
    # curving in a vertical plane the front keeps facing the same side
    s = np.linspace(0, 1, 30)
    curl = np.stack([0.3 * np.sin(2 * s), np.zeros(30), -s], axis=1)
    c = G.ribbon(curl, 0.1, up=(0, -1, 0))
    assert (G.face_normals(c.verts, c.faces)[:, 1] < -0.999).all()
    # per-point width and the strip layout: column 0 at -B (u = 0), column 1 at +B (u = 1)
    w = np.linspace(0.2, 0.05, 25)
    g = G.ribbon(wiggle, w)
    assert np.allclose(np.linalg.norm(g.verts[1::2] - g.verts[0::2], axis=1), w)
    uv = g.uv.reshape(-1, 4, 2)
    assert np.allclose(uv[:, 0, 0], 0) and np.allclose(uv[:, 1, 0], 1) and (uv_signed_areas(g) > 0).all()
    assert uv[:, :, 1].min() == 0 and uv[:, :, 1].max() == pytest.approx(1.0) and (np.diff(uv[:, 0, 1]) > 0).all()
    with pytest.raises(ValueError):
        G.ribbon(wiggle, np.ones(5))


def test_ribbon_thick_slab_is_closed_and_outward():
    path = line(5, 2.0, 0)
    g = G.ribbon(path, 0.2, thickness=0.02)
    assert (g.n_verts, g.n_faces) == (20, 4 * 4 + 2)
    solid(g, vol=2.0 * 0.2 * 0.02, uv_ccw=False)
    n = G.face_normals(g.verts, g.faces)[:16].reshape(4, 4, 3)                              # (segment, column, xyz)
    assert (n[:, 0, 2] > 0.99).all() and (n[:, 2, 2] < -0.99).all()   # column 0 = front (+N = +Z), column 2 = back
    ar = uv_signed_areas(g)
    assert (ar[:16].reshape(4, 4)[:, [0, 2]] > 0).all()                    # front and back read un-mirrored
    assert (ar >= -1e-12).all() and g.uv.min() >= -1e-12 and g.uv.max() <= 1 + 1e-12
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    loop = G.ribbon(ring, 0.1, thickness=0.01, closed_path=True)                           # a closed band
    solid(loop, convex=False, uv_ccw=False)
    assert G.signed_volume(loop.verts, loop.faces) == pytest.approx(2 * np.pi * 0.1 * 0.01, rel=0.01)
    flat = G.ribbon(ring, 0.1, closed_path=True)
    assert G.check(flat)["boundary_edges"] == 80 and flat.n_faces == 40


# ---------------------------------------------------------------------------------------------- revolve and spheres
def test_revolve_poles_counts_and_orientation():
    prof = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1.0]])
    g = G.revolve(prof, 16)
    assert (g.n_verts, g.n_faces) == (2 + 2 * 16, 3 * 16)                   # pole fans + one band of quads
    assert sorted(set(map(len, g.faces))) == [3, 4]
    solid(g, vol=polygon_area(16, 0.5) * 1.0)
    assert (g.verts[0] == [0, 0, 0]).all() and g.verts[0].tolist() == [0.0, 0.0, 0.0]
    uv = g.uv
    assert uv.shape == (g.n_corners, 2) and np.allclose(uv.min(0), 0) and np.allclose(uv.max(0), 1)
    # a profile that is not a pole stays open at that end
    pipe = G.revolve(np.array([[0.5, 0], [0.5, 1.0]]), 12)
    assert G.check(pipe)["boundary_edges"] == 24 and pipe.n_faces == 12
    half = G.revolve(np.array([[0.5, 0], [0.5, 1.0]]), 16, angle=np.pi)    # partial: segments + 1 samples
    assert half.n_verts == 34 and half.n_faces == 16 and np.allclose(half.verts[:, 1].min(), 0, atol=1e-12)
    assert half.verts[:, 1].min() >= -1e-12 and half.verts[16, 0] == pytest.approx(-0.5)
    assert (G.face_normals(half.verts, half.faces)[:, 1] > 0).all()                       # outward = the +Y side
    neg = G.revolve(prof, 16, angle=-2 * np.pi)                             # clockwise turn: still outward
    solid(neg, vol=polygon_area(16, 0.5))
    inward = G.revolve(prof[::-1], 16)                               # listed top to bottom: inside out
    assert G.signed_volume(inward.verts, inward.faces) < 0
    assert G.signed_volume(*(lambda x: (x.verts, x.faces))(G.revolve(prof[::-1], 16, flip=True))) > 0
    # pole in the middle (lens): both bands fan out of it
    lens = G.revolve(np.array([[0, 0], [0.3, 0.5], [0, 1.0]]), 8)
    solid(lens, vol=None)
    with pytest.raises(ValueError):
        G.revolve(np.array([[0, 0], [0, 1.0]]), 8)
    with pytest.raises(ValueError):
        G.revolve(np.array([[0.5, 0]]), 8)
    with pytest.raises(ValueError):
        G.revolve(prof, 2)


def test_uv_sphere_counts_closed_and_uvs():
    for seg, rings in ((16, 8), (12, 6), (32, 16)):
        g = G.uv_sphere(1.0, segments=seg, rings=rings)
        assert g.n_verts == 2 + (rings - 1) * seg and g.n_faces == seg * rings
        assert (sum(len(f) == 3 for f in g.faces), sum(len(f) == 4 for f in g.faces)) == (2 * seg, seg * (rings - 2))
        solid(g)
        assert np.allclose(np.linalg.norm(g.verts, axis=1), 1.0)
        vol = G.signed_volume(g.verts, g.faces)
        assert 0.85 * 4 / 3 * np.pi < vol < 4 / 3 * np.pi
    g = G.uv_sphere(2.0, center=(1, 2, 3), segments=16, rings=8)
    assert np.allclose(g.verts[0], [1, 2, 1]) and np.allclose(g.verts[-1], [1, 2, 5])  # poles on Z, bottom pole first
    assert np.allclose(g.verts.mean(0), [1, 2, 3], atol=1e-9)
    assert g.uv[:, 0].min() == 0 and g.uv[:, 0].max() == 1 and g.uv[:, 1].min() == 0 and g.uv[:, 1].max() == 1
    uv = g.uv
    # v = latitude fraction from the bottom pole: arccos(-z / R) / pi; u = azimuth from +X / 2 pi (the seam has 0 and 1)
    at_vertex, o = {}, 0
    for f in g.faces:
        for k, vi in enumerate(f):
            at_vertex.setdefault(vi, []).append(uv[o + k])
        o += len(f)
    for vi, rows in at_vertex.items():
        p = g.verts[vi] - [1, 2, 3]
        rows = np.array(rows)
        assert np.allclose(rows[:, 1], np.arccos(np.clip(-p[2] / 2.0, -1, 1)) / np.pi, atol=1e-9)
        if np.hypot(p[0], p[1]) > 1e-9:
            az = (np.arctan2(p[1], p[0]) / (2 * np.pi)) % 1.0
            assert np.allclose((rows[:, 0] - az + 0.5) % 1.0 - 0.5, 0, atol=1e-9)
    assert (uv_signed_areas(g) > 0).all()


def test_uv_sphere_ellipsoid_and_mirrored_scale():
    g = G.uv_sphere((1.0, 2.0, 3.0), center=(1, 1, 1), segments=24, rings=12)
    lo, hi = g.bbox()
    assert np.allclose(hi - lo, [2.0, 4.0, 6.0], atol=1e-9)                 # poles on Z, equator on X and Y
    assert np.allclose(g.verts.mean(0), [1, 1, 1], atol=1e-9)
    solid(g)
    s = G.uv_sphere(1.0, segments=12, rings=6, scale=(1, 1, 2))
    assert s.bbox()[1][2] == pytest.approx(2.0)
    m = G.uv_sphere(1.0, segments=12, rings=6, scale=(-1, 1, 1))
    assert G.signed_volume(m.verts, m.faces) > 0 and G.check(m)["inconsistent_edges"] == 0
    with pytest.raises(ValueError):
        G.uv_sphere(1.0, rings=1)
    with pytest.raises(ValueError):
        G.uv_sphere((1.0, 2.0))


def test_quad_sphere_is_all_welded_quads_and_uniform():
    for n in (1, 2, 3, 4, 8):
        g = G.quad_sphere(1.0, n=n)
        assert g.n_verts == 6 * n * n + 2 and g.n_faces == 6 * n * n and all(len(f) == 4 for f in g.faces)
        solid(g)
        assert np.allclose(np.linalg.norm(g.verts, axis=1), 1.0)
        assert g.uv.shape == (4 * 6 * n * n, 2)
    g = G.quad_sphere(0.5, n=6, center=(1, 0, 0), scale=(1, 2, 1))
    a = G.face_areas(g.verts, g.faces)
    assert np.allclose(g.verts.mean(0), [1, 0, 0], atol=1e-9)
    g = G.quad_sphere(1.0, n=6)
    a = G.face_areas(g.verts, g.faces)
    assert a.max() / a.min() < 1.2 and a.sum() == pytest.approx(4 * np.pi, rel=0.02)    # near-equal-area quads
    val = np.array([len(s) for s in G.adjacency(g.faces, g.n_verts)])
    assert (val == 3).sum() == 8 and (val == 4).sum() == g.n_verts - 8    # only the cube corners have valence 3
    e = G.quad_sphere((1, 2, 3), n=4)
    assert np.allclose(e.bbox()[1], [1, 2, 3], atol=1e-9)
    m = G.quad_sphere(1.0, n=4, scale=(-1, 1, 1))
    assert G.signed_volume(m.verts, m.faces) > 0
    with pytest.raises(ValueError):
        G.quad_sphere(1.0, n=0)
    with pytest.raises(ValueError):
        G.quad_sphere(1.0, uv="bogus")


def test_quad_sphere_uv_variants():
    assert G.quad_sphere(1.0, n=4, uv=None).uv is None
    g = G.quad_sphere(1.0, n=4)                                                         # 'spherical'
    uv = g.uv.reshape(-1, 4, 2)
    assert np.ptp(uv[:, :, 0], axis=1).max() <= 0.25 + 1e-9                             # no face spans the seam
    assert uv.min() >= -1e-9 and uv.max() <= 1 + 1e-9 and (uv_signed_areas(g) > 0).all()
    # u is the azimuth from +X, v the latitude
    corner = g.verts[np.array(g.faces).ravel()]
    az = (np.arctan2(corner[:, 1], corner[:, 0]) / (2 * np.pi)) % 1.0
    far = (np.abs(corner[:, 0]) + np.abs(corner[:, 1])) > 1e-6
    d = np.abs(((az - g.uv[:, 0]) + 0.5) % 1.0 - 0.5)[far]
    assert d.max() < 1e-9
    assert np.allclose(g.uv[:, 1], 0.5 + np.arcsin(np.clip(corner[:, 2], -1, 1)) / np.pi)
    straddles = [bool(g.verts[f][:, 1].min() < -1e-9 and g.verts[f][:, 1].max() > 1e-9 and g.verts[f][:, 0].min() > 0)
                 for f in g.faces]
    assert not any(straddles)                                  # even n: the seam is a cell edge
    odd = G.quad_sphere(1.0, n=5)                                                       # odd n: cells straddle the seam
    ou = odd.uv.reshape(-1, 4, 2)[:, :, 0]
    assert np.isfinite(odd.uv).all() and (ou.min() < 0.0 or ou.max() > 1.0)            # unwrapped per face, beyond 0..1
    c = G.quad_sphere(1.0, n=4, uv="cross")
    cu = c.uv
    assert cu.shape == (384, 2)
    assert cu.min() >= 0 and cu.max() <= 1 and np.allclose(cu[:, 1].min(), 0.125) and np.allclose(cu[:, 1].max(), 0.875)
    assert (uv_signed_areas(c) > 0).all()


# ---------------------------------------------------------------------------------------------- box and friends
def test_box_counts_volume_and_cross_uv():
    for seg in ((1, 1, 1), (2, 3, 4), (1, 5, 2)):
        nx, ny, nz = seg
        g = G.box((0.2, 0.4, 0.6), center=(1, 2, 3), segments=seg)
        assert g.n_verts == (nx + 1) * (ny + 1) * (nz + 1) - (nx - 1) * (ny - 1) * (nz - 1)
        assert g.n_faces == 2 * (nx * ny + ny * nz + nx * nz) and all(len(f) == 4 for f in g.faces)
        solid(g, vol=0.2 * 0.4 * 0.6)
        lo, hi = g.bbox()
        assert np.allclose(lo, [0.9, 1.8, 2.7]) and np.allclose(hi, [1.1, 2.2, 3.3])
    g = G.box((1, 2, 3), segments=(2, 3, 4))
    a = G.face_areas(g.verts, g.faces)
    ua = uv_signed_areas(g)
    assert (ua > 0).all() and np.allclose(ua / a, (ua / a)[0])    # uniform texel density, un-mirrored
    assert g.uv.min() >= 0 and g.uv.max() <= 1
    u = G.box()                                                   # unit cube: the 4 x 3 cross, centred
    assert np.allclose(u.uv.min(0), [0, 0.125]) and np.allclose(u.uv.max(0), [1, 0.875])
    assert G.box(uv=False).uv is None
    assert G.box(segments=2).n_faces == 24
    uvq = u.uv.reshape(-1, 4, 2)
    assert np.allclose(uvq[0], [[0.5, 0.375], [0.75, 0.375], [0.75, 0.625], [0.5, 0.625]])   # +X in column 2
    assert np.allclose(uvq[3].min(0), [0.25, 0.375])                       # -Y: second column of the middle row
    assert np.allclose(uvq[4].min(0), [0.25, 0.625])                       # +Z: above it
    for bad in (dict(size=(1, 0, 1)), dict(size=(-1, 1, 1)), dict(segments=(1, 0, 1))):
        with pytest.raises(ValueError):
            G.box(**bad)
    n = G.face_normals(u.verts, u.faces)                                                  # face order +X -X +Y -Y +Z -Z
    assert np.allclose(n, [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def test_cylinder_cone_capsule():
    g = G.cylinder(0.5, 1.0, 16, 3, cap_top=True, cap_bottom="fan", center=(1, 2, 3))
    assert g.n_verts == 4 * 16 + 1 and g.n_faces == 3 * 16 + 16 + 1
    solid(g, vol=polygon_area(16, 0.5) * 1.0)
    lo, hi = g.bbox()
    assert np.allclose(lo, [0.5, 1.5, 2.5]) and np.allclose(hi, [1.5, 2.5, 3.5])
    assert G.cylinder(cap_top=False, cap_bottom=False).n_faces == 16 and G.cylinder().n_faces == 18
    assert not G.is_closed(G.cylinder(cap_top=False).faces)
    e = G.cylinder((0.5, 0.25), 2.0, 16)
    assert np.ptp(e.verts[:, 0]) == pytest.approx(1.0) and np.ptp(e.verts[:, 1]) == pytest.approx(0.5, rel=0.02)
    solid(e)
    c = G.cone(0.5, 1.0, 16, 2)
    assert (c.n_verts, c.n_faces) == (34, 48)
    solid(c, vol=polygon_area(16, 0.5) / 3)
    nc = G.cone(0.5, 1.0, 16, 1, cap=False)
    assert not G.is_closed(nc.faces) and G.check(nc)["boundary_edges"] == 16
    k = G.capsule(0.5, 1.0, 16, 4)
    assert k.n_verts == 2 + 8 * 16 and k.n_faces == 16 * 2 + 16 * 7
    solid(k)
    lo, hi = k.bbox()
    assert hi[2] == pytest.approx(1.0) and lo[2] == pytest.approx(-1.0) and hi[0] == pytest.approx(0.5)   # h + 2r long
    vol = G.signed_volume(k.verts, k.faces)
    assert 0.9 * (np.pi * 0.25 + 4 / 3 * np.pi * 0.125) < vol < np.pi * 0.25 + 4 / 3 * np.pi * 0.125
    s = G.capsule(0.5, 0.0, 16, 4)                                # zero straight part = a sphere
    assert s.n_verts == 2 + 7 * 16 and G.check(s)["n_degenerate"] == 0
    solid(s)
    moved = G.capsule(0.1, 0.4, 8, 2, center=(0, 0, 5))
    assert np.allclose(moved.bbox()[0][2], 5 - 0.3) and np.allclose(moved.bbox()[1][2], 5 + 0.3)
    with pytest.raises(ValueError):
        G.capsule(rings=0)


def test_disc_plane_and_surface():
    d = G.disc(1.0, 16)
    assert (d.n_verts, d.n_faces, len(d.faces[0])) == (16, 1, 16)
    assert np.allclose(G.face_normals(d.verts, d.faces), [[0, 0, 1]])
    assert np.allclose(G.face_areas(d.verts, d.faces), polygon_area(16))
    assert np.allclose(d.verts[0], [1, 0, 0]) and np.allclose(d.verts.mean(0), 0)
    for normal in ((0, 1, 0), (1, 2, 3), (0, 0, -1)):
        r = G.disc(0.5, 12, center=(1, 2, 3), normal=normal, rings=3)
        assert r.n_verts == 1 + 3 * 12 and r.n_faces == 12 + 2 * 12
        nn = G.face_normals(r.verts, r.faces)
        assert np.allclose(nn, G.unit(normal), atol=1e-9)
        assert G.face_areas(r.verts, r.faces).sum() == pytest.approx(polygon_area(12, 0.5))
        assert np.allclose(r.verts[0], [1, 2, 3]) and (uv_signed_areas(r) > 0).all()
        assert G.check(r)["uv_ok"] and G.check(r)["boundary_edges"] == 12
    assert np.allclose(G.disc(1.0, 8, center=(0, 0, 1)).uv, 0.5 + 0.5 * G.circle_profile(8))
    with pytest.raises(ValueError):
        G.disc(normal=(0, 0, 0))
    with pytest.raises(ValueError):
        G.disc(sides=2)
    p = G.plane((2, 1), (4, 2), center=(0, 0, 1))
    assert (p.n_verts, p.n_faces) == (15, 8) and np.allclose(G.face_normals(p.verts, p.faces), [0, 0, 1])
    assert np.allclose(p.bbox()[0], [-1, -0.5, 1]) and np.allclose(p.bbox()[1], [1, 0.5, 1])
    assert np.allclose(p.uv.min(0), 0) and np.allclose(p.uv.max(0), 1)
    assert np.allclose(p.uv[:4], [[0, 0], [0.25, 0], [0.25, 0.5], [0, 0.5]])
    f = G.plane((2, 1), (4, 2), normal=(0, -1, 0), up=(0, 0, 1))                       # a vertical sheet facing -Y
    assert np.allclose(G.face_normals(f.verts, f.faces), [0, -1, 0])
    assert np.ptp(f.verts[:, 2]) == pytest.approx(1.0) and np.ptp(f.verts[:, 0]) == pytest.approx(2.0)   # y extent -> Z
    assert f.verts[0, 2] == pytest.approx(-0.5) and f.verts[-1, 2] == pytest.approx(0.5)   # rows run along local y = up
    assert (uv_signed_areas(f) > 0).all()                                               # u runs along +X seen from -Y
    assert f.verts[1, 0] > f.verts[0, 0]
    with pytest.raises(ValueError):
        G.plane(normal=(0, 0, 0))


def test_surface_is_called_once_and_oriented():
    calls = []

    def sheet(u, v):
        calls.append((u.shape, v.shape))
        return np.stack([u, v, 0.1 * u * v], axis=-1)

    g = G.surface(sheet, 4, 3)
    assert calls == [((4, 5), (4, 5))] and (g.n_verts, g.n_faces) == (20, 12)
    assert (G.face_normals(g.verts, g.faces)[:, 2] > 0.9).all()                         # dP/du x dP/dv = +Z
    swapped = G.surface(lambda u, v: np.stack([v, u, 0 * u], axis=-1), 4, 3)
    assert (G.face_normals(swapped.verts, swapped.faces)[:, 2] < -0.99).all()
    assert np.allclose(g.uv.min(0), 0) and np.allclose(g.uv.max(0), 1)

    def cyl(u, v):
        a = 2 * np.pi * u
        return np.stack([np.cos(a), np.sin(a), v], axis=-1)

    calls.clear()
    c = G.surface(lambda u, v: (calls.append(u.shape), cyl(u, v))[1], 12, 2, closed_u=True)
    assert calls == [(3, 12)] and c.n_verts == 36 and c.n_faces == 24
    radial = G.face_centers(c.verts, c.faces)[:, :2]
    assert (np.einsum("ij,ij->i", G.face_normals(c.verts, c.faces)[:, :2], radial) > 0).all()
    assert G.check(c)["boundary_edges"] == 24
    assert np.isclose(c.uv.reshape(-1, 4, 2)[:, :, 0].max(), 1.0)                       # seam corner has u = 1

    def torus(u, v):
        a, b = 2 * np.pi * u, 2 * np.pi * v
        r = 1 + 0.3 * np.cos(b)
        return np.stack([r * np.cos(a), r * np.sin(a), 0.3 * np.sin(b)], axis=-1)

    calls.clear()
    t = G.surface(lambda u, v: (calls.append(1), torus(u, v))[1], 24, 12, closed_u=True, closed_v=True)
    assert calls == [1] and (t.n_verts, t.n_faces) == (288, 288)
    solid(t, convex=False)
    assert G.signed_volume(t.verts, t.faces) == pytest.approx(2 * np.pi ** 2 * 1.0 * 0.09, rel=0.08)
    with pytest.raises(ValueError):
        G.surface(lambda u, v: np.zeros(u.shape + (2,)), 4, 4)
    with pytest.raises(ValueError):
        G.surface(sheet, 0, 3)


# ---------------------------------------------------------------------------------------------- triangulate
def test_triangulate_concave_polygon_keeps_uv_corner_order():
    L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)             # concave hexagon in the XZ plane
    g = G.Geo(np.stack([L[:, 0], np.zeros(6), L[:, 1]], axis=1), [[0, 1, 2, 3, 4, 5]], uv=L.copy(), face_mat=[2])
    for method in ("ear", "fan"):
        t = G.triangulate(g, method)
        assert len(t.faces) == 4 and all(len(f) == 3 for f in t.faces)
        assert np.allclose(G.face_areas(t.verts, t.faces).sum(), 3.0)
        corner_v = t.verts[np.array(t.faces).ravel()]
        assert np.allclose(t.uv, corner_v[:, [0, 2]])                                  # every corner carries its own UV
        assert t.face_mat.tolist() == [2, 2, 2, 2] and t.uv.shape == (12, 2)
    ear = G.triangulate(g)
    assert np.allclose(G.face_normals(ear.verts, ear.faces), [0, -1, 0])                # no flipped or degenerate tri
    assert G.check(ear)["n_degenerate"] == 0
    fan = G.triangulate(g, "fan")
    assert fan.faces == [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    assert np.allclose(G.face_normals(fan.verts, fan.faces), [0, -1, 0])    # star-shaped from corner 0: valid
    assert g.n_faces == 1 and len(g.faces[0]) == 6                                       # input untouched


def test_triangulate_random_concave_polygons_in_any_plane():
    rng = np.random.default_rng(7)
    for _ in range(40):
        m = int(rng.integers(5, 30))
        th = 2 * np.pi * (np.arange(m) + rng.uniform(0.05, 0.95, m)) / m
        r = rng.uniform(0.15, 1.0, m)
        p2 = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        R = random_rotation(rng)
        off = rng.normal(size=3)
        p3 = np.column_stack([p2, np.zeros(m)]) @ R.T + off
        area = 0.5 * np.sum(p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1])
        t = G.triangulate(G.Geo(p3, [list(range(m))], uv=p2))
        assert len(t.faces) == m - 2
        assert G.face_areas(t.verts, t.faces).sum() == pytest.approx(area, abs=1e-12)
        assert (G.face_normals(t.verts, t.faces) @ R[:, 2] > 0.999).all()    # all wound like the polygon
        assert np.allclose(t.uv, ((t.verts[np.array(t.faces).ravel()] - off) @ R)[:, :2], atol=1e-12)
    for m in (5, 8, 32):                                                    # convex ngons stay well shaped
        th = 2 * np.pi * np.arange(m) / m
        t = G.triangulate(G.Geo(np.stack([np.cos(th), np.sin(th), np.zeros(m)], axis=1), [list(range(m))]))
        a = G.face_areas(t.verts, t.faces)
        assert len(a) == m - 2 and a.sum() == pytest.approx(polygon_area(m)) and a.min() > 0


def test_triangulate_polygons_with_collinear_vertices_leaves_no_slivers():
    rect = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1.0]])     # extra vertices on the long sides
    grid = [(i, 0) for i in range(4)] + [(4, j) for j in range(3)] + [(i, 3) for i in range(4, 0, -1)] + \
           [(0, j) for j in range(3, 0, -1)]
    comb = np.array([[0, 0], [5, 0], [5, 2], [4, 2], [4, 1], [3, 1], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2.0]])
    R = G.rot_axis([1, 2, 0.3], 0.9)
    for poly in (rect, np.array(grid, float), comb):
        m = len(poly)
        t = G.triangulate(G.Geo(np.column_stack([poly, np.zeros(m)]) @ R.T, [list(range(m))], uv=poly))
        area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        a = G.face_areas(t.verts, t.faces)
        assert len(t.faces) == m - 2 and a.sum() == pytest.approx(area) and a.min() > 1e-9   # no zero-area triangle
        assert (G.face_normals(t.verts, t.faces) @ R[:, 2] > 0.999).all()
    rng = np.random.default_rng(8)                                                           # histogram outlines
    for _ in range(30):
        h = rng.integers(1, 5, int(rng.integers(2, 8))).astype(float)
        pts = [(float(i), 0.0) for i in range(len(h) + 1)]
        for i in range(len(h) - 1, -1, -1):
            pts += [(float(i + 1), h[i]), (float(i), h[i])]
        poly = np.array([p for k, p in enumerate(pts) if k == 0 or p != pts[k - 1]])
        poly = poly[:-1] if (poly[0] == poly[-1]).all() else poly
        m = len(poly)
        t = G.triangulate(G.Geo(np.column_stack([poly, np.zeros(m)]), [list(range(m))]))
        a = G.face_areas(t.verts, t.faces)
        assert len(t.faces) == m - 2 and a.min() > 1e-9 and (G.face_normals(t.verts, t.faces)[:, 2] > 0.999).all()
        assert a.sum() == pytest.approx(h.sum())


def test_triangulate_quads_diagonal_modes_and_materials():
    dart = G.Geo(np.array([[0, 0, 0], [1, 0.2, 0], [2, 0, 0], [1, 3, 0.0]]), [[0, 1, 2, 3]])   # reflex corner at 1
    ear = G.triangulate(dart)
    assert np.allclose(G.face_normals(ear.verts, ear.faces), [0, 0, 1])     # the longer, valid diagonal 1-3
    assert sorted(sorted(f) for f in ear.faces) == [[0, 1, 3], [1, 2, 3]]
    bad = G.triangulate(dart, "fan")
    assert (G.face_normals(bad.verts, bad.faces)[:, 2] < 0).any()
    # convex trapezoid: the shorter diagonal is 0-2
    trap = G.Geo(np.array([[0, 0, 0], [4, 0, 0], [3, 1, 0], [0, 1, 0.0]]), [[0, 1, 2, 3]])
    assert sorted(sorted(f) for f in G.triangulate(trap).faces) == [[0, 1, 2], [0, 2, 3]]
    trap2 = G.Geo(np.array([[0, 0, 0], [4, 0, 0], [4, 1, 0], [1, 1, 0.0]]), [[0, 1, 2, 3]])  # shorter diagonal is 1-3
    assert sorted(sorted(f) for f in G.triangulate(trap2).faces) == [[0, 1, 3], [1, 2, 3]]
    q = G.plane((1, 1), (2, 2))
    q.face_mat = np.array([5, 6, 7, 8])
    t = G.triangulate(q)
    assert t.n_faces == 8 and t.face_mat.tolist() == [5, 5, 6, 6, 7, 7, 8, 8]            # carried, face order kept
    assert np.allclose(t.uv, t.verts[np.array(t.faces).ravel()][:, :2] + 0.5)
    assert q.n_faces == 4
    c = G.cylinder(0.5, 1.0, 12, 2)
    c.face_mat = np.arange(c.n_faces) % 3
    tc = G.triangulate(c)
    assert tc.n_faces == 2 * 24 + 2 * 10 and sorted(set(map(len, tc.faces))) == [3]
    assert G.face_areas(tc.verts, tc.faces).sum() == pytest.approx(G.face_areas(c.verts, c.faces).sum())
    assert G.signed_volume(tc.verts, tc.faces) == pytest.approx(G.signed_volume(c.verts, c.faces))
    assert G.check(tc)["uv_ok"] and G.check(tc)["boundary_edges"] == 0 and len(tc.face_mat) == tc.n_faces
    ng = G.triangulate(c, only_ngons=True)                                                # keeps quads, splits the caps
    assert ng.n_faces == 24 + 2 * 10 and sorted(set(map(len, ng.faces))) == [3, 4]
    assert G.check(ng)["uv_ok"] and (G.face_normals(ng.verts, ng.faces) != 0).any()
    assert G.triangulate(G.cylinder(cap_top="fan", cap_bottom="fan")).n_faces == 16 * 2 + 32
    assert G.triangulate(G.Geo(np.zeros((0, 3)), [])).n_faces == 0
    with pytest.raises(ValueError):
        G.triangulate(c, "bogus")
    mixed = G.merge([G.box(), G.disc(1.0, 6), G.uv_sphere(0.2, segments=6, rings=3)])        # tris + quads + ngon
    tm = G.triangulate(mixed)
    assert sorted(set(map(len, tm.faces))) == [3] and G.check(tm)["uv_ok"] and tm.n_faces == 12 + 4 + (6 * 2 * 2)


# ---------------------------------------------------------------------------------------------- merge
def test_merge_geos_offsets_uv_and_materials():
    a = G.box()
    b = G.uv_sphere(0.3, segments=6, rings=3).translated((2, 0, 0))
    b = G.Geo(b.verts, b.faces, None, None)
    m = G.merge([a, b])
    assert m.n_verts == a.n_verts + b.n_verts and m.n_faces == a.n_faces + b.n_faces
    assert min(min(f) for f in m.faces[a.n_faces:]) == a.n_verts and m.faces[:a.n_faces] == a.faces
    assert m.uv.shape == (m.n_corners, 2) and np.allclose(m.uv[:a.n_corners], a.uv) and not m.uv[a.n_corners:].any()
    assert m.face_mat is None
    a2 = a.with_material(1)
    m2 = G.merge([a2, b])
    assert m2.face_mat.tolist() == [1] * 6 + [0] * b.n_faces                             # None -> material 0
    assert G.merge([b, b]).uv is None
    assert G.merge([]).n_verts == 0 and G.merge([a]).faces == a.faces and G.merge([a]).verts is not a.verts
    assert G.check(m)["uv_ok"] and G.check(m2)["mat_ok"]
    assert G.merge([a, a.translated((3, 0, 0))]).n_verts == 16
    assert G.signed_volume(*(lambda x: (x.verts, x.faces))(G.merge([a, a.translated((3, 0, 0))]))) == pytest.approx(2.0)


def make_mesh(name, geo, mats, **kw):
    return geo.to_mesh(name, mats=mats, **kw)


def test_merge_meshes_remaps_materials_weights_morphs():
    box = G.box()
    sph = G.uv_sphere(0.5, segments=8, rings=4)
    sph.face_mat = np.arange(sph.n_faces) % 2
    m1 = box.to_mesh("a", mats=["skin"], weights={"頭": np.ones(8), "首": np.full(8, 0.25)},
                     morphs={"あ": np.ones((8, 3))}, sharp=[(0, 1)], crease={(2, 3): 1.0})
    m2 = sph.to_mesh("b", mats=["hair", "skin"], weights={"頭": np.full(sph.n_verts, 0.5), "髪": np.ones(sph.n_verts)},
                     morphs={"い": np.full((sph.n_verts, 3), 2.0)}, sharp=[(1, 2)], crease={(0, 1): 0.5},
                     normals=sph.normals(), smooth=False)
    mm = G.merge_meshes([m1, m2], "ab")
    assert mm.name == "ab" and mm.mats == ["skin", "hair"] and mm.smooth is False and mm.subsurf == 0
    assert len(mm.verts) == 8 + sph.n_verts and len(mm.faces) == 6 + sph.n_faces
    assert mm.faces[6:] == [[i + 8 for i in f] for f in sph.faces] and np.allclose(mm.verts[:8], box.verts)
    assert mm.face_mat[:6].tolist() == [0] * 6                  # mats[0] of mesh a -> "skin" = 0
    assert mm.face_mat[6:].tolist() == [1, 0] * (sph.n_faces // 2) + [1, 0][:sph.n_faces % 2]   # hair 1, skin 0
    assert set(mm.weights) == {"首", "髪", "頭"}
    assert np.allclose(mm.weights["頭"], np.r_[np.ones(8), np.full(sph.n_verts, 0.5)])
    assert np.allclose(mm.weights["首"], np.r_[np.full(8, 0.25), np.zeros(sph.n_verts)])   # zero-filled
    assert np.allclose(mm.weights["髪"], np.r_[np.zeros(8), np.ones(sph.n_verts)])
    assert mm.morphs["あ"].shape == mm.morphs["い"].shape == (len(mm.verts), 3)
    assert np.allclose(mm.morphs["あ"][:8], 1) and not mm.morphs["あ"][8:].any() and not mm.morphs["い"][:8].any()
    assert mm.sharp == [(0, 1), (9, 10)] and mm.crease == {(2, 3): 1.0, (8, 9): 0.5}
    assert mm.normals.shape == (len(mm.verts), 3) and np.allclose(mm.normals[8:], sph.normals())
    assert np.allclose(mm.normals[:8], box.normals())                       # computed for the one without
    assert mm.uv.shape == (box.n_corners + sph.n_corners, 2)
    p = part.Part("t", meshes=[mm], materials=[part.Material("skin"), part.Material("hair")])
    part.check(p)
    nouv = G.Geo(sph.verts, sph.faces, None, None).to_mesh("c", mats=["hair"])
    m3 = G.merge_meshes([m1, nouv], "x")
    assert m3.uv.shape[0] == box.n_corners + sph.n_corners and not m3.uv[box.n_corners:].any()
    assert G.merge_meshes([nouv, nouv], "y").uv is None and G.merge_meshes([nouv, nouv], "y").normals is None
    assert m3.face_mat.tolist() == [0] * 6 + [1] * sph.n_faces
    one = G.merge_meshes([m1], "solo")
    assert one.name == "solo" and one.faces == m1.faces and one.verts is not m1.verts and one.face_mat is None
    with pytest.raises(ValueError):
        G.merge_meshes([m1, box.to_mesh("c", mats=["skin"], subsurf=2)], "bad")
    with pytest.raises(ValueError):
        G.merge_meshes([], "none")
    with pytest.raises(ValueError):
        G.merge_meshes([m1, box.to_mesh("nomat")], "nomat")
    s2 = G.merge_meshes([box.to_mesh("p", subsurf=2), box.to_mesh("q", subsurf=2)], "s2")
    assert s2.subsurf == 2 and s2.mats == [] and s2.face_mat is None


# ---------------------------------------------------------------------------------------------- weld / mirror
def test_weld_merges_duplicates_and_returns_remap():
    box = G.box()
    dup = G.merge([box, box.translated((0, 0, 0))])
    w, remap = G.weld(dup)
    assert w.n_verts == 8 and w.n_faces == 12 and remap.shape == (16,)    # faces are kept, only verts merge
    assert remap.tolist() == list(range(8)) * 2
    assert np.allclose(w.verts[remap], dup.verts)
    assert w.uv.shape == (48, 2) and np.allclose(w.uv, dup.uv)
    noisy = G.Geo(dup.verts + np.random.default_rng(0).normal(size=dup.verts.shape) * 1e-9, dup.faces, dup.uv)
    assert G.weld(noisy)[0].n_verts == 8
    far = G.merge([box, box.translated((1e-3, 0, 0))])
    assert G.weld(far)[0].n_verts == 16 and G.weld(far, tol=5e-3)[0].n_verts == 8
    assert G.weld(G.Geo(np.zeros((0, 3)), []))[1].shape == (0,)
    two = G.Geo(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0.0]]), [[0, 1, 2]])
    w2, r2 = G.weld(two)                                                   # vertices merged, face collapsed
    assert w2.n_verts == 2 and w2.n_faces == 0 and r2.tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        G.weld(box, tol=0)


def test_weld_finds_pairs_across_cell_borders():
    tol = 1e-6
    border = 0.5 * tol          # np.rint(x / tol) changes here: the points sit in different cells
    pts = np.array([[border - 1e-13, 0, 0], [border + 1e-13, 0, 0],
                    [border + 1e-13, 3 * tol, 0], [border - 1e-13, 3 * tol, 0.0]])
    w, remap = G.weld(G.Geo(pts, [[0, 1, 2, 3]]), tol)
    assert remap.tolist() == [0, 0, 1, 1] and w.n_verts == 2 and w.n_faces == 0
    corner = np.array([[border - 1e-13] * 3, [border + 1e-13] * 3])         # across a cell corner: 3 axes
    assert G.weld(G.Geo(corner, []), tol)[0].n_verts == 1
    rng = np.random.default_rng(1)                           # many pairs, a few percent of them straddle a cell border
    base = rng.uniform(0, 1, size=(500, 3))
    cloud = np.vstack([base, base + rng.normal(size=base.shape) * 2e-8])
    assert (np.rint(cloud[:500] / tol) != np.rint(cloud[500:] / tol)).any(axis=1).sum() > 5
    w, remap = G.weld(G.Geo(cloud, [[0, 1, 2]]), tol)
    assert w.n_verts == 500 and (remap[:500] == remap[500:]).all()


def test_weld_cleans_faces_and_keeps_uv_per_corner():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0.0]])
    uv = np.arange(16, dtype=float).reshape(8, 2) / 16
    g = G.Geo(v, [[0, 1, 2, 3, 4], [2, 3, 4]], uv=uv, face_mat=[1, 2])
    w, remap = G.weld(g)
    assert remap.tolist() == [0, 1, 2, 2, 3] and w.faces == [[0, 1, 2, 3]] and w.face_mat.tolist() == [1]
    assert w.uv.shape == (4, 2) and np.allclose(w.uv, uv[[0, 1, 2, 4]])    # the dropped corner took its UV along
    assert G.check(w)["uv_ok"]
    # a pinched polygon (touches itself) is split; the degenerate piece vanishes
    v2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0], [1, 0, 0.0]])
    p = G.Geo(v2, [[0, 1, 2, 3, 4, 5]], uv=np.arange(12).reshape(6, 2) / 12.0)
    w, remap = G.weld(p)
    assert w.faces == [[1, 2, 3, 4]] and w.uv.shape == (4, 2)
    # a UV map defined from the vertex position survives the weld on every kept corner
    sph = G.uv_sphere(1.0, segments=12, rings=6)
    cut = G.Geo(sph.verts.copy(), sph.faces, np.array([sph.verts[i][:2] for f in sph.faces for i in f]))
    cut2 = G.merge([cut, cut.translated((0, 0, 0))])
    w, remap = G.weld(cut2)
    assert w.n_verts == cut.n_verts and w.n_faces == 2 * cut.n_faces
    assert np.allclose(w.uv, w.verts[np.array(sum(w.faces, []))][:, :2])
    # weld repairs a tapered tube tip made of a zero-radius ring
    t = G.tube(line(4), [0.1, 0.1, 0.05, 0.0], 6, cap_start="ngon")
    wt, _ = G.weld(t)
    assert G.check(wt)["boundary_edges"] == 0 and G.check(wt)["n_degenerate"] == 0 and wt.n_verts == 3 * 6 + 1


def test_mirror_x_of_half_shape_is_closed_and_symmetric():
    full = G.uv_sphere(1.0, segments=16, rings=8)
    half, _ = G.subset(full, np.array([bool((full.verts[f][:, 0] >= -1e-9).all()) for f in full.faces]))
    assert G.check(half)["boundary_edges"] > 0
    m = G.mirror_x(half)
    c = G.check(m)
    assert c["boundary_edges"] == 0 and c["non_manifold_edges"] == 0 and c["inconsistent_edges"] == 0
    assert c["signed_volume"] > 0 and c["uv_ok"] and c["unused_verts"] == 0
    assert (m.n_verts, m.n_faces) == (full.n_verts, full.n_faces)
    assert c["signed_volume"] == pytest.approx(G.signed_volume(full.verts, full.faces))

    def rows(a):                                            # vertex set, order independent
        a = np.round(a, 9) + 0.0
        return a[np.lexsort(a.T[::-1])]

    assert np.allclose(rows(m.verts), rows(m.verts * [-1, 1, 1]))                         # symmetric vertex set
    assert (m.verts[np.abs(m.verts[:, 0]) < 1e-6][:, 0] == 0.0).all()                     # seam exactly on x = 0
    assert m.faces[:half.n_faces] == half.faces                                            # g's own faces come first
    # a half box that includes its face on the mirror plane: the two copies cancel, leaving a closed box
    hb = G.box((1, 1, 1), center=(0.5, 0, 0))
    mb = G.mirror_x(hb)
    assert G.check(mb)["boundary_edges"] == 0 and G.check(mb)["non_manifold_edges"] == 0
    assert mb.n_faces == 10 and mb.n_verts == 12 and G.signed_volume(mb.verts, mb.faces) == pytest.approx(2.0)
    assert G.check(mb)["uv_ok"] and G.check(mb)["unused_verts"] == 0


# ---------------------------------------------------------------------------------------------- normals / analysis
def test_vertex_normals_angle_vs_area_weighting():
    box = G.box((1, 1, 10))                         # long box: sizes differ a lot
    ang = box.normals()
    assert np.allclose(np.abs(ang), 1 / np.sqrt(3)) and np.allclose(np.linalg.norm(ang, axis=1), 1)
    assert (np.sign(ang) == np.sign(box.verts)).all()
    area = box.normals("area")
    assert not np.allclose(np.abs(area), 1 / np.sqrt(3))                                      # long faces dominate
    assert np.allclose(np.linalg.norm(area, axis=1), 1)
    assert np.allclose(np.abs(G.vertex_normals(box.verts, box.faces, "uniform")), 1 / np.sqrt(3))
    split = G.box((1, 1, 10), segments=(1, 1, 9))   # same corner, finer tessellation
    sc = split.normals()
    corner = np.all(np.isclose(np.abs(split.verts), [0.5, 0.5, 5.0]), axis=1)
    assert np.allclose(np.abs(sc[corner]), 1 / np.sqrt(3))    # angle weights ignore tessellation
    s = G.uv_sphere(1.0, segments=32, rings=16)
    assert np.abs(s.normals() - s.verts).max() < 0.01
    q = G.quad_sphere(1.0, n=8)
    assert np.abs(q.normals() - q.verts).max() < 0.012
    ang6 = np.arange(6) * np.pi / 3
    hexagon = G.Geo(np.stack([np.cos(ang6), np.sin(ang6), np.zeros(6)], axis=1), [list(range(6))])
    assert np.allclose(hexagon.normals(), [0, 0, 1])                                           # ngons
    with pytest.raises(ValueError):
        box.normals("bogus")
    assert G.vertex_normals(np.zeros((3, 3)), []).shape == (3, 3) and not G.vertex_normals(np.zeros((3, 3)), []).any()


def test_degenerate_faces_never_produce_nan():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0.0]])
    g = G.Geo(v, [[0, 1, 2], [0, 1, 3]])                                                       # first face is collinear
    vn = G.vertex_normals(v, g.faces)
    assert np.isfinite(vn).all() and np.allclose(vn[[0, 1, 3]], [0, 0, 1]) and not vn[2].any()
    assert np.allclose(G.face_normals(v, g.faces), [[0, 0, 0], [0, 0, 1]])
    assert np.allclose(G.face_areas(v, g.faces), [0, 0.5])
    z = G.Geo(np.zeros((4, 3)), [[0, 1, 2, 3]])
    assert not G.vertex_normals(z.verts, z.faces).any() and G.face_areas(z.verts, z.faces)[0] == 0
    assert G.check(g)["n_degenerate"] == 1 and G.check(z)["n_degenerate"] == 1


def test_face_areas_centers_and_normals():
    g = G.Geo([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [5, 5, 5]], [[0, 1, 2, 3], [1, 2, 4], [0, 1, 2, 3, 4]])
    areas = G.face_areas(g.verts, g.faces)
    assert areas[0] == pytest.approx(2.0)
    assert areas[1] == pytest.approx(0.5 * np.linalg.norm(np.cross(g.verts[2] - g.verts[1], g.verts[4] - g.verts[1])))
    centers = G.face_centers(g.verts, g.faces)
    assert np.allclose(centers[0], [1, 0.5, 0]) and np.allclose(centers[1], g.verts[[1, 2, 4]].mean(0))
    assert np.allclose(centers[2], g.verts[:5].mean(0))
    assert np.allclose(G.face_normals(g.verts, g.faces)[0], [0, 0, 1])
    assert G.face_areas([[0, 0, 0]], []).shape == (0,) and G.face_centers([[0, 0, 0]], []).shape == (0, 3)
    c = G.box((1, 2, 3))
    assert np.allclose(G.face_areas(c.verts, c.faces), [6, 6, 3, 3, 2, 2])


def test_signed_volume_and_ensure_outward():
    box = G.box((1, 2, 3))
    assert G.signed_volume(box.verts, box.faces) == pytest.approx(6.0)
    assert G.signed_volume(box.verts, box.flipped().faces) == pytest.approx(-6.0)
    far = box.translated((1e3, -2e3, 5e2))
    assert G.signed_volume(far.verts, far.faces) == pytest.approx(6.0, rel=1e-8)
    cap = G.cylinder(0.5, 1.0, 16, 1)                        # ngon caps are fan-triangulated
    assert G.signed_volume(cap.verts, cap.faces) == pytest.approx(polygon_area(16, 0.5))
    assert G.signed_volume(np.zeros((0, 3)), []) == 0.0
    inner = G.uv_sphere(1.0, segments=12, rings=6).flipped()
    outer = G.box((1, 1, 1), center=(3, 0, 0))
    mix = G.merge([inner, outer])
    fixed = G.ensure_outward(mix)
    assert fixed.n_verts == mix.n_verts                                                      # only the bad shell turned
    assert fixed.faces[inner.n_faces:] == [[i + inner.n_verts for i in f] for f in outer.faces]
    assert fixed.faces[:inner.n_faces] == [f[::-1] for f in inner.faces]
    assert G.signed_volume(fixed.verts, fixed.faces) > G.signed_volume(mix.verts, mix.faces)
    assert corner_map(G.Geo(fixed.verts, fixed.faces[:inner.n_faces], fixed.uv[:inner.n_corners])) == corner_map(
        G.Geo(inner.verts, inner.faces, inner.uv))
    assert G.ensure_outward(box).faces == box.faces and G.ensure_outward(box) is not box
    sheet = G.plane().flipped()
    assert G.ensure_outward(sheet).faces == sheet.faces      # open surfaces are left alone
    assert G.ensure_outward(G.Geo(np.zeros((0, 3)), [])).n_faces == 0
    two = G.merge([G.box().flipped(), G.box(center=(4, 0, 0)).flipped()])
    out = G.ensure_outward(two)
    assert G.check(out)["signed_volume"] == pytest.approx(2.0)


def test_edges_boundary_adjacency_components_subset():
    tris = [[0, 1, 2], [2, 1, 3]]
    assert G.edges(tris) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    assert G.boundary_edges(tris) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert G.edges([[0, 1, 2, 3]]) == [(0, 1), (0, 3), (1, 2), (2, 3)]
    assert G.edges([]) == [] and G.boundary_edges([]) == []
    assert all(isinstance(i, int) and isinstance(j, int) and i < j for i, j in G.edges(tris))
    assert [sorted(s) for s in G.adjacency(tris, 5)] == [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2], []]
    assert G.is_closed(G.box().faces) and not G.is_closed(tris) and not G.is_closed([])
    assert G.is_closed(G.uv_sphere().faces) and not G.is_closed(G.cylinder(cap_top=False).faces)
    assert G.boundary_edges(G.box().faces) == []
    lab = G.components(G.merge([G.box(), G.box(center=(3, 0, 0))]).faces, 17)
    assert lab[:8].tolist() == [0] * 8 and lab[8:16].tolist() == [8] * 8 and lab[16] == 16
    chain = [[i, i + 1, i + 2] for i in range(0, 200, 2)]    # long chains converge quickly
    assert set(G.components(chain, 201).tolist()) == {0}
    g = G.merge([G.box().with_material(1), G.box(center=(2, 0, 0)).with_material(2)])
    sub, keep = G.subset(g, np.arange(g.n_faces) >= 6)
    assert sub.n_faces == 6 and sub.n_verts == 8 and keep.tolist() == list(range(8, 16))
    assert sub.face_mat.tolist() == [2] * 6 and sub.uv.shape == (24, 2) and np.allclose(sub.verts, g.verts[keep])
    assert min(min(f) for f in sub.faces) == 0 and G.check(sub)["unused_verts"] == 0
    s2, k2 = G.subset(g, [11, 0])                            # index list keeps its order
    assert s2.n_faces == 2 and sorted(k2.tolist()) == k2.tolist() and s2.face_mat.tolist() == [2, 1]
    assert np.allclose(s2.uv[:4], g.uv[4 * 11:4 * 12]) and np.allclose(s2.uv[4:], g.uv[:4])
    e, ke = G.subset(g, [])
    assert e.n_faces == 0 and len(ke) == 0


def test_smooth_relaxes_noise_and_respects_pins():
    pl = G.plane((1, 1), (8, 8))
    noisy = pl.verts.copy()
    noisy[:, 2] += np.random.default_rng(1).normal(size=len(noisy)) * 0.01
    sm = G.smooth(noisy, pl.faces, 5)
    bv = np.unique(np.array(G.boundary_edges(pl.faces)).ravel())
    assert sm[:, 2].std() < noisy[:, 2].std() * 0.7 and sm.shape == noisy.shape
    assert np.allclose(sm[bv], noisy[bv])                                                    # border stays by default
    assert not np.allclose(G.smooth(noisy, pl.faces, 5, keep_boundary=False)[bv], noisy[bv])
    pin = np.zeros(len(noisy), bool)
    pin[40] = True
    s2 = G.smooth(noisy, pl.faces, 3, pin=pin)
    assert np.allclose(s2[40], noisy[40]) and not np.allclose(s2[41], noisy[41])
    assert np.allclose(G.smooth(noisy, pl.faces, 3, pin=[40, 41])[[40, 41]], noisy[[40, 41]])
    assert np.allclose(G.smooth(noisy, pl.faces, 0), noisy)
    assert not np.shares_memory(G.smooth(noisy, pl.faces, 1), noisy)
    assert np.allclose(G.smooth(noisy, pl.faces, 1, lam=0.0), noisy)
    one = G.smooth(noisy, pl.faces, 1, lam=1.0, keep_boundary=False)    # lam = 1: the neighbour mean
    adj = G.adjacency(pl.faces, len(noisy))
    assert np.allclose(one[40], noisy[sorted(adj[40])].mean(0))
    sph = G.uv_sphere(1.0, segments=16, rings=8)
    shrunk = G.smooth(sph.verts, sph.faces, 3)
    assert np.linalg.norm(shrunk, axis=1).max() < 1.0 + 1e-9 and np.linalg.norm(shrunk, axis=1).mean() < 0.99
    assert np.allclose(G.smooth(np.zeros((3, 3)), [], 4), 0)


def test_check_reports_every_kind_of_defect():
    ok = G.check(G.box())
    assert ok == dict(n_verts=8, n_faces=6, bad_indices=0, n_degenerate=0, boundary_edges=0, non_manifold_edges=0,
                      inconsistent_edges=0, signed_volume=pytest.approx(1.0), unused_verts=0, uv_ok=True, mat_ok=True)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [5, 5, 5.0]])
    c = G.check(G.Geo(v, [[0, 1, 2], [0, 1, 3], [0, 1, 4]]))                # three faces on one edge
    assert c["non_manifold_edges"] == 1 and c["boundary_edges"] == 6 and c["unused_verts"] == 1
    c = G.check(G.Geo(v, [[0, 1, 2], [0, 1, 3]]))                                                # both walk 0 -> 1
    assert c["inconsistent_edges"] == 1
    c = G.check(G.Geo(v, [[0, 1, 2], [1, 0, 3]]))
    assert c["inconsistent_edges"] == 0
    c = G.check(G.Geo(v, [[0, 1, 1], [0, 1, 2, 2]]))
    assert c["n_degenerate"] == 2
    c = G.check(G.Geo(v, [[0, 1, 9]]))
    assert c["bad_indices"] == 1 and c["signed_volume"] is None and c["unused_verts"] is None
    c = G.check(G.Geo(v, [[0, 1, 2]], uv=np.zeros((2, 2))))
    assert c["uv_ok"] is False
    c = G.check(G.Geo(v, [[0, 1, 2]], uv=np.full((3, 2), np.nan)))
    assert c["uv_ok"] is False
    c = G.check(G.Geo(v, [[0, 1, 2]], face_mat=[0, 1]))
    assert c["mat_ok"] is False
    inv = G.check(G.box().flipped())
    assert inv["signed_volume"] == pytest.approx(-1.0) and inv["inconsistent_edges"] == 0
    assert G.check(G.Geo(np.zeros((0, 3)), []))["n_faces"] == 0


# ---------------------------------------------------------------------------------------------- to_mesh
def test_to_mesh_builds_valid_part_meshes():
    g = G.uv_sphere(0.1, segments=8, rings=4)
    g.face_mat = np.arange(g.n_faces) % 2
    w = {"頭": np.ones(g.n_verts)}
    m = {"あ": np.zeros((g.n_verts, 3))}
    mesh = g.to_mesh("head", mats=["skin", "blush"], weights=w, morphs=m, sharp=[(0, 1)], crease={(0, 1): 0.5},
                     subsurf=1, smooth=False, normals=g.normals())
    assert isinstance(mesh, part.Mesh) and mesh.name == "head" and mesh.mats == ["skin", "blush"]
    assert mesh.face_mat.tolist() == g.face_mat.tolist() and mesh.subsurf == 1 and mesh.smooth is False
    assert mesh.sharp == [(0, 1)] and mesh.crease == {(0, 1): 0.5} and mesh.normals.shape == (g.n_verts, 3)
    part.check(part.Part("t", meshes=[mesh], materials=[part.Material("skin"), part.Material("blush")]))
    mesh.verts[0] += 1                                                      # copies: g is untouched
    mesh.faces[0][0] = 99
    mesh.weights["頭"][0] = 7
    assert g.verts[0, 2] != mesh.verts[0, 2] and g.faces[0][0] != 99 and w["頭"][0] == 1
    assert g.to_mesh("x", mats=["skin"]).face_mat is None and g.to_mesh("x").face_mat is None     # <= 1 material: None
    assert G.uv_sphere(0.1).to_mesh("x", mats=["a", "b"]).face_mat is None    # no face_mat on the Geo
    m2 = g.to_mesh("x", mats=[part.Material("a"), part.Material("b")])
    assert m2.mats == ["a", "b"]
    assert g.to_mesh("x", weights=None, morphs=None, crease=None, sharp=None).weights == {}
    for kw in (dict(weights={"b": np.ones(3)}), dict(morphs={"m": np.zeros((g.n_verts, 2))}),
               dict(normals=np.zeros((2, 3))), dict(mats=["a", "b"], morphs={"m": np.zeros(g.n_verts)})):
        with pytest.raises(ValueError):
            g.to_mesh("x", **kw)
    bad = G.Geo(g.verts, g.faces, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        bad.to_mesh("x")
    far = g.copy()
    far.face_mat = np.full(g.n_faces, 5)
    with pytest.raises(ValueError):
        far.to_mesh("x", mats=["a", "b"])
    far.face_mat = far.face_mat[:3]
    with pytest.raises(ValueError):
        far.to_mesh("x", mats=["a", "b"])
    assert far.to_mesh("x", mats=["a"]).face_mat is None


# ---------------------------------------------------------------------------------------------- misc
def test_public_api_is_documented_and_complete():
    names = ["Geo", "unit", "rot_x", "rot_y", "rot_z", "rot_axis", "rot_between", "frame_from", "matrix_xyz",
             "euler_xyz", "euler_for_axis", "euler_from_axes", "capsule_between", "transform_points", "arclength",
             "resample", "catmull_rom", "bezier", "rmf_frames", "loft", "sweep", "tube", "ribbon", "revolve",
             "uv_sphere", "quad_sphere", "box", "cylinder", "capsule", "disc", "plane", "surface", "triangulate",
             "merge", "merge_meshes", "weld", "vertex_normals", "face_normals", "face_areas", "face_centers",
             "signed_volume", "ensure_outward", "edges", "boundary_edges", "is_closed", "adjacency", "smooth",
             "mirror_x", "check"]
    for n in names:
        assert n in G.__all__ and callable(getattr(G, n))
    mine = [n for n, o in vars(G).items() if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
            and o.__module__ == G.__name__]
    assert sorted(mine) == sorted(G.__all__)                                    # nothing public is left out of __all__
    for n in G.__all__:
        assert getattr(G, n).__doc__ and len(getattr(G, n).__doc__) > 30, n
    for m in ("copy", "transformed", "translated", "flipped", "mirrored_x", "normals", "to_mesh", "n_verts", "n_faces",
              "n_corners", "with_material", "uv_scaled", "bbox"):
        assert getattr(G.Geo, m).__doc__, m
    assert "Conventions" in G.__doc__ and "API summary" in G.__doc__


def test_module_is_bpy_free_and_fast_on_big_meshes():
    src = inspect.getsource(G)
    assert not re.search(r"^\s*(import|from)\s+(bpy|mathutils|scipy)\b", src, re.M)
    t0 = time.perf_counter()
    path = np.stack([np.linspace(0, 1, 200), 0.1 * np.sin(np.linspace(0, 6, 200)), np.linspace(0, 1, 200)], axis=1)
    big = G.tube(path, np.linspace(0.02, 0.005, 200), 48, cap_start="fan", cap_end="fan")
    assert big.n_faces == 199 * 48 + 96
    solid(big, convex=False)
    tri = G.triangulate(big)
    assert tri.n_faces == 199 * 48 * 2 + 96
    nrm = big.normals()
    assert np.isfinite(nrm).all() and abs(np.linalg.norm(nrm, axis=1) - 1).max() < 1e-9
    many = G.merge([big.translated((i, 0, 0)) for i in range(5)])
    w, remap = G.weld(many)
    assert w.n_verts == many.n_verts and G.is_closed(many.faces)
    assert time.perf_counter() - t0 < 5.0
