"""Geometry toolkit for part builders: a mesh container (`Geo`), rotations, curves, surfaces and mesh analysis.

Pure numpy (no bpy, no scipy; Python >= 3.11, numpy >= 1.24), so it runs in plain Python and inside Blender. Everything
that touches many faces or vertices is vectorised; Python loops only run over paths, profiles, ngons or union-find
pairs.

Conventions
-----------
* Model space: metres, Z up, the character faces -Y, its left is +X, feet at z = 0, centred on x = 0.
* `verts` is a float64 (n, 3) array; `faces` a list of int lists (tris, quads or ngons) wound counter-clockwise seen
  from OUTSIDE (the right-hand rule gives the outward normal).
* UVs are per face corner: an array (sum(len(f) for f in faces), 2) in face order, exactly like `part.Mesh.uv`; v points
  UP (Blender: (0, 0) is the bottom-left of the image; image files are stored top row first). Since UVs are per corner
  a seam never needs duplicated vertices: closed surfaces store their own u = 1 (or v = 1) corner values.
* `face_mat` is None or an int array (n_faces,) indexing the material list of the mesh it ends up in.
* Weights are dicts bone name -> (n,) float arrays (PMX/Japanese bone names); morphs are name -> (n, 3) offsets.
* Rotations are 3x3 matrices acting on column vectors (p' = R @ p); for (n, 3) point arrays use `P @ R.T`.
  Euler angles are Blender 'XYZ': R = Rz(rz) @ Ry(ry) @ Rx(rx).  A *frame* is a 3x3 matrix whose COLUMNS are the
  orthonormal right-handed axes (x, y, z).
* A path is a (k, 3) polyline; a profile is a (m, 2) polyline in a local plane; both are CCW unless stated otherwise.

Vertex layouts (builders rely on them to attach weights / morphs per ring)
-------------------------------------------------------------------------
* loft / sweep / tube / ribbon / surface / plane / cylinder: vertex (ring r, column j) has index r * nv + j (nv = points
  per ring). Fan-cap centre vertices are appended after the ring vertices: the start cap's centre first, then the end's.
  Side quads come first (ring-major, column-minor), then the start cap, then the end cap.
* revolve / uv_sphere / capsule: one block per profile point, bottom to top; a block holds one vertex per angular sample
  (`segments` for a full turn, `segments + 1` for a partial one); a profile point with r ~ 0 is a single pole vertex.
* box / quad_sphere: welded lattice vertices in raster order (x slowest); faces ordered +X, -X, +Y, -Y, +Z, -Z.

API summary
-----------
Container     Geo(verts, faces, uv=None, face_mat=None): copy, transformed, translated, flipped, mirrored_x,
              normals, to_mesh, with_material, uv_scaled, bbox; n_verts, n_faces, n_corners.
Math          unit, rot_x, rot_y, rot_z, rot_axis, rot_between, frame_from, matrix_xyz, euler_xyz, euler_for_axis,
              euler_from_axes, capsule_between, transform_points.
Curves        arclength, resample, catmull_rom, bezier, rmf_frames, circle_profile, ccw.
Surfaces      loft, sweep, tube, ribbon, revolve, uv_sphere, quad_sphere, box, cylinder, cone, capsule, disc, plane,
              surface.
Topology      triangulate, merge, merge_meshes, weld, subset, components, mirror_x, ensure_outward, edges,
              boundary_edges, is_closed, adjacency, smooth.
Analysis      vertex_normals, face_normals, face_areas, face_centers, signed_volume, check.

UV layouts (all un-mirrored seen from outside; v up)
-------------------------------------------------
* loft / sweep / tube: u around the ring (0..1 with its own seam corner), v along (by ring index for loft, by arc length
  for sweep / tube / ribbon); caps are projected on their plane. ribbon: u across 0..1 (+B side = 1), v along.
* revolve / uv_sphere / capsule / cone: u = angle fraction from +X (counter-clockwise), v = arc-length fraction of the
  profile (bottom = 0). quad_sphere: 'spherical' (equirectangular, seam handled per corner) or 'cross'.
* box: the unfolded cross (middle row -X -Y +X +Y, +Z above and -Z below the -Y face), cells in true proportion.
* plane / disc: planar, 0..1.

Typical use (`G` = this module):

    ring = G.circle_profile(6, 0.01)                                  # CCW profile in the (N, B) plane of the frames
    strand = G.sweep(ring, path, scale=np.linspace(1.0, 0.2, len(path)), cap_end="fan")
    head = G.uv_sphere(0.09, center=(0, 0, 1.5), scale=(1, 1.05, 1.1))
    hair = G.merge([head, strand])
    mesh = hair.to_mesh("hair", mats=["hair"], weights={"頭": np.ones(hair.n_verts)})
"""
import itertools
from dataclasses import dataclass

import numpy as np

from . import part as _part

__all__ = [
    "Geo", "unit", "rot_x", "rot_y", "rot_z", "rot_axis", "rot_between", "frame_from", "matrix_xyz", "euler_xyz",
    "euler_for_axis", "euler_from_axes", "capsule_between", "transform_points",
    "arclength", "resample", "catmull_rom", "bezier", "rmf_frames", "circle_profile", "ccw",
    "loft", "sweep", "tube", "ribbon", "revolve", "uv_sphere", "quad_sphere", "box", "cylinder", "cone", "capsule",
    "disc", "plane", "surface",
    "triangulate", "merge", "merge_meshes", "weld", "subset", "components", "mirror_x", "ensure_outward", "edges",
    "boundary_edges", "is_closed", "adjacency", "smooth",
    "vertex_normals", "face_normals", "face_areas", "face_centers", "signed_volume", "check",
]

_TINY = 1e-12


# ------------------------------------------------------------------------------------------------ flat face arrays
class _Poly:
    """Flattened corner arrays of a face list (internal). Per face: sizes, offs (first corner). Per corner: flat
    (vertex index), fid (face index), pos (position in the face), nxt / prv (corner index of the next / previous
    corner of the same face)."""

    def __init__(self, faces):
        nf = len(faces)
        self.nf = nf
        self.sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=nf)
        self.nc = int(self.sizes.sum())
        self.flat = np.fromiter(itertools.chain.from_iterable(faces), dtype=np.int64, count=self.nc)
        self.offs = np.cumsum(self.sizes) - self.sizes
        self.fid = np.repeat(np.arange(nf, dtype=np.int64), self.sizes)
        base = self.offs[self.fid]
        self.pos = np.arange(self.nc, dtype=np.int64) - base
        size = self.sizes[self.fid]
        self.nxt = base + (self.pos + 1) % size
        self.prv = base + (self.pos - 1) % size


def _unflatten(flat, sizes):
    """Flat vertex array + face sizes -> list of int lists."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if len(sizes) == 0:
        return []
    if (sizes == sizes[0]).all():
        return np.asarray(flat).reshape(-1, int(sizes[0])).tolist()
    fl = np.asarray(flat).tolist()
    offs = (np.cumsum(sizes) - sizes).tolist()
    return [fl[o:o + s] for o, s in zip(offs, sizes.tolist())]


def _gather_corners(offs, sizes, idx):
    """Corner indices (flat) of the faces `idx`, in that order."""
    s = sizes[idx]
    total = int(s.sum())
    start = np.cumsum(s) - s
    return np.repeat(offs[idx] - start, s) + np.arange(total, dtype=np.int64)


def _reverse_index(sizes, offs):
    """Corner index that reverses the corner order inside every face."""
    nf = len(sizes)
    fid = np.repeat(np.arange(nf), sizes)
    pos = np.arange(int(sizes.sum())) - offs[fid]
    return offs[fid] + sizes[fid] - 1 - pos


def _fan(sizes, offs):
    """Fan triangles (0, i, i + 1) of every face: (face index, corner0, corner1, corner2) arrays."""
    cnt = np.maximum(sizes - 2, 0)
    tf = np.repeat(np.arange(len(sizes)), cnt)
    first = np.cumsum(cnt) - cnt
    i = np.arange(len(tf)) - first[tf] + 1
    c0 = offs[tf]
    return tf, c0, c0 + i, c0 + i + 1


def _offset_faces(faces, off):
    if off == 0:
        return [list(f) for f in faces]
    return [[i + off for i in f] for f in faces]


def _face_raw(V, P):
    """Per face: raw Newell vector (length = 2 * area), its length, validity (non-degenerate), centroid; and the corner
    positions."""
    c = V[P.flat]
    cen = np.add.reduceat(c, P.offs, axis=0) / P.sizes[:, None]
    a = c - cen[P.fid]
    raw = np.add.reduceat(np.cross(a, a[P.nxt]), P.offs, axis=0)
    e = c[P.nxt] - c
    l2 = np.maximum.reduceat(np.einsum("ij,ij->i", e, e), P.offs)
    ln = np.linalg.norm(raw, axis=1)
    valid = ln > 1e-9 * np.maximum(l2, 1e-300)
    return raw, ln, valid, cen, c


def _repeated_vertex_faces(P):
    """Boolean (nf,) mask of faces that use the same vertex index twice."""
    out = np.zeros(P.nf, dtype=bool)
    if P.nc == 0:
        return out
    order = np.lexsort((P.flat, P.fid))
    sv, sf = P.flat[order], P.fid[order]
    dup = (sv[1:] == sv[:-1]) & (sf[1:] == sf[:-1])
    out[sf[1:][dup]] = True
    return out


# ------------------------------------------------------------------------------------------------ math
def unit(v):
    """Normalise along the last axis; zero-length vectors (norm <= 1e-12) stay zero instead of becoming NaN.

    Args: v: array-like (..., k). Returns: float64 array, same shape."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = n > _TINY
    return np.where(ok, v / np.where(ok, n, 1.0), 0.0)


def rot_x(a):
    """Rotation matrix about +X (right-handed, column-vector convention: R @ p).

    Args: a: angle in radians, a scalar or an array (...,).
    Returns: (3, 3) float array, or (..., 3, 3) for an array of angles."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = c
    R[..., 1, 2] = -s
    R[..., 2, 1] = s
    R[..., 2, 2] = c
    return R


def rot_y(a):
    """Rotation matrix about +Y (right-handed, column-vector convention: R @ p).

    Args: a: angle in radians, a scalar or an array (...,).
    Returns: (3, 3) float array, or (..., 3, 3) for an array of angles."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 1, 1] = 1.0
    R[..., 0, 0] = c
    R[..., 0, 2] = s
    R[..., 2, 0] = -s
    R[..., 2, 2] = c
    return R


def rot_z(a):
    """Rotation matrix about +Z (right-handed, column-vector convention: R @ p).

    Args: a: angle in radians, a scalar or an array (...,).
    Returns: (3, 3) float array, or (..., 3, 3) for an array of angles."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 2, 2] = 1.0
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


def rot_axis(axis, angle):
    """Rotation matrix about an arbitrary axis (Rodrigues' formula, right-handed).

    Args: axis: (3,) or (..., 3), normalised internally (a zero axis raises ValueError). angle: radians, a scalar or
    (...,); axis and angle broadcast.
    Returns: (3, 3), or (..., 3, 3) when either argument is batched."""
    ax = np.asarray(axis, dtype=float)
    if np.any(np.linalg.norm(ax, axis=-1) <= _TINY):
        raise ValueError("rot_axis: zero-length axis")
    k = unit(ax)
    a = np.asarray(angle, dtype=float)
    c, s = np.cos(a), np.sin(a)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    C = 1.0 - c
    R = np.empty(np.broadcast(x, a).shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def _perp(a):
    """Some unit vector perpendicular to the unit vector `a` (built from the world axis least aligned with it)."""
    e = np.zeros(3)
    e[int(np.argmin(np.abs(a)))] = 1.0
    return unit(np.cross(a, e))


def rot_between(a, b):
    """Shortest rotation taking the direction `a` onto the direction `b`.

    Args: a, b: (3,) vectors (need not be unit; a zero vector gives the identity).
    Returns: (3, 3) rotation matrix R with R @ unit(a) == unit(b). Parallel vectors give the identity, antiparallel
    ones a half turn about an arbitrary axis perpendicular to `a` (accurate for nearly antiparallel input too)."""
    a, b = unit(np.asarray(a, dtype=float)), unit(np.asarray(b, dtype=float))
    if not a.any() or not b.any():
        return np.eye(3)
    c = float(a @ b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if c < 0.0 and s < 1e-3:
        # half turn a -> -a (exact), then the tiny remaining rotation -a -> b
        return rot_between(-a, b) @ rot_axis(_perp(a), np.pi)
    if s == 0.0:
        return np.eye(3)
    return rot_axis(v / s, np.arctan2(s, c))


def _world_perp(a, order):
    """Unit vector perpendicular to `a` from the first world axis (in `order`) that is not too aligned with it."""
    for i in order:
        e = np.zeros(3)
        e[i] = 1.0
        w = e - (e @ a) * a
        if np.linalg.norm(w) > 0.5:
            return unit(w)
    return _perp(a)


def frame_from(z=None, x=None, y=None, up=(0.0, 0.0, 1.0)):
    """Orthonormal right-handed frame whose COLUMNS are the axes (x, y, z).

    Args: z, x, y: up to three axis directions ((3,) array-likes, need not be unit; zero vectors are ignored), at
    least one is required. up: roll hint (default world Z), only used when a single axis is given.
    Returns: (3, 3) rotation matrix [x | y | z] with x × y = z (so R @ (0, 0, 1) is the z axis).

    Priority when several axes are given: z, then x, then y: the first is kept exactly, the second is orthogonalised
    against it (ignored when parallel to it), the third is the cross product (a third given axis is ignored).
    With a single axis the roll is chosen with the `up` hint:
      * z given: x = up × z (horizontal), y = z × x (up-ish)    -> "look-at": z forward, y up
      * x given: y = up × x (horizontal), z = x × y (up-ish)    -> z-up frames, x along a bone
      * y given: x = y × up (horizontal), z = x × y (up-ish)
    When the given axis is parallel to `up` (or `up` is zero) the horizontal axis falls back to a world axis
    (X for z/y given, Y for x given), so frame_from(z=(0,0,1)) is the identity and nothing is ever NaN."""
    given = {}
    for name, v in (("z", z), ("x", x), ("y", y)):
        if v is not None:
            u = unit(np.asarray(v, dtype=float))
            if u.any():
                given[name] = u
    if not given:
        raise ValueError("frame_from needs at least one non-zero axis (x, y or z)")
    names = list(given)
    first = names[0]
    a = given[first]
    second = None
    for name in names[1:]:
        w = given[name] - (given[name] @ a) * a
        if np.linalg.norm(w) > 1e-6:
            second = (name, unit(w))
            break
    if second is not None:
        ax = {first: a, second[0]: second[1]}
        if "x" not in ax:
            ax["x"] = np.cross(ax["y"], ax["z"])
        elif "y" not in ax:
            ax["y"] = np.cross(ax["z"], ax["x"])
        else:
            ax["z"] = np.cross(ax["x"], ax["y"])
    else:
        upv = unit(np.asarray(up, dtype=float))
        h = np.cross(upv, a)
        good = np.linalg.norm(h) >= 1e-6
        if first == "z":
            xa = unit(h) if good else _world_perp(a, (0, 1, 2))
            ax = {"x": xa, "y": np.cross(a, xa), "z": a}
        elif first == "x":
            ya = unit(h) if good else _world_perp(a, (1, 0, 2))
            ax = {"x": a, "y": ya, "z": np.cross(a, ya)}
        else:
            xa = -unit(h) if good else _world_perp(a, (0, 1, 2))
            ax = {"x": xa, "y": a, "z": np.cross(xa, a)}
    return np.column_stack([ax["x"], ax["y"], ax["z"]])


def matrix_xyz(euler):
    """Rotation matrix of a Blender 'XYZ' Euler triple: R = Rz(rz) @ Ry(ry) @ Rx(rx).

    Args: euler: (3,) or (..., 3) radians (rx, ry, rz). Returns: (3, 3) or (..., 3, 3)."""
    e = np.asarray(euler, dtype=float)
    return rot_z(e[..., 2]) @ rot_y(e[..., 1]) @ rot_x(e[..., 0])


def euler_xyz(R):
    """Blender 'XYZ' Euler angles (rx, ry, rz) of rotation matrix `R`, so that matrix_xyz(euler_xyz(R)) == R.

    ry lies in [-pi/2, pi/2]; rx and rz in [-pi, pi]. At gimbal lock (|ry| = pi/2) rz is set to 0 and rx takes the
    whole remaining angle. Close to gimbal lock the two angles are re-solved from the well-conditioned entries, so
    the round trip stays accurate to ~1e-15 instead of degrading like 1e-16 / cos(ry).

    Args: R: (3, 3) or (..., 3, 3). Returns: (3,) or (..., 3) float array."""
    R = np.asarray(R, dtype=float)
    batch = R.shape[:-2]
    M = R.reshape(-1, 3, 3)
    r00, r10, sb = M[:, 0, 0], M[:, 1, 0], -M[:, 2, 0]
    cy = np.hypot(r00, r10)
    ry = np.arctan2(sb, cy)
    rx = np.arctan2(M[:, 2, 1], M[:, 2, 2])
    rz = np.arctan2(r10, r00)
    near = np.nonzero(cy < 1e-2)[0]
    if len(near):
        s = np.where(sb[near] >= 0.0, 1.0, -1.0)
        eta = cy[near] ** 2 / (1.0 + np.abs(sb[near]))                     # 1 - |sin ry|, without cancellation
        a0, c0 = rx[near], rz[near]
        sz, ca, sa = np.sin(c0), np.cos(a0), np.sin(a0)
        phi = np.arctan2(-M[near, 1, 2] - s * eta * sz * ca, M[near, 1, 1] + s * eta * sz * sa)   # = rx - s*rz
        delta = (phi - (a0 - s * c0) + np.pi) % (2 * np.pi) - np.pi
        a1 = a0 + 0.5 * delta
        c1 = c0 - 0.5 * s * delta
        lock = cy[near] < 1e-15
        a1 = np.where(lock, phi, a1)
        c1 = np.where(lock, 0.0, c1)
        rx[near] = (a1 + np.pi) % (2 * np.pi) - np.pi
        rz[near] = (c1 + np.pi) % (2 * np.pi) - np.pi
    return np.stack([rx, ry, rz], axis=-1).reshape(batch + (3,)) + 0.0                  # + 0.0: no negative zeros


def euler_for_axis(direction, axis="z", up=(0.0, 0.0, 1.0)):
    """Euler XYZ angles of a rotation whose local axis points along a given direction.

    Args: direction: (3,) target direction (need not be unit). axis: 'x', 'y' or 'z' = the local axis to aim
    (default 'z', e.g. a capsule's height axis). up: roll hint, see frame_from.
    Returns: tuple (rx, ry, rz) of floats (radians) with matrix_xyz((rx, ry, rz))[:, axis] == unit(direction)."""
    axis = str(axis).lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    R = frame_from(up=up, **{axis: direction})
    return tuple(float(e) for e in euler_xyz(R))


def euler_from_axes(x=None, y=None, z=None, up=(0.0, 0.0, 1.0)):
    """Euler XYZ angles of the frame frame_from(x=x, y=y, z=z, up=up) (see frame_from for the axis priority and the
    roll rule).

    Returns: tuple (rx, ry, rz) of floats (radians)."""
    return tuple(float(e) for e in euler_xyz(frame_from(z=z, x=x, y=y, up=up)))


def capsule_between(a, b, radius, inclusive=False):
    """Rigid-body capsule spanning two points, as keyword arguments for `part.RigidBody(**kw)`.

    Args: a, b: (3,) end points, model space. radius: capsule radius (m). inclusive: False (default): a and b ARE the
    hemisphere centres, height = |b - a|; True: a and b are the extreme tips, height = max(|b - a| - 2 * radius, 0).
    Returns: dict(location=(x, y, z), rotation=(rx, ry, rz), size=(radius, height, 0.0)) of plain floats: location is
    the midpoint of a and b; rotation is the XYZ Euler of the rotation taking the capsule's height axis (local +Z at
    rotation 0) onto b - a (a zero-length segment gets (0, 0, 0)); `height` is the straight section between the
    hemisphere centres."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    length = float(np.linalg.norm(d))
    r = float(radius)
    height = max(length - 2.0 * r, 0.0) if inclusive else length
    rot = euler_for_axis(d, "z") if length > _TINY else (0.0, 0.0, 0.0)
    loc = tuple(float(v) for v in (a + b) * 0.5)
    return dict(location=loc, rotation=rot, size=(r, float(height), 0.0))


def transform_points(P, R=None, t=None, scale=None):
    """p' = (p * scale) @ R.T + t: scale, then rotate, then translate.

    Args: P: points (..., 3). R: (3, 3) rotation (or any linear map; None = identity). t: (3,) translation.
    scale: scalar or (3,), applied first, along the local axes.
    Returns: a new float64 array shaped like P (the input is never modified or returned)."""
    out = np.array(P, dtype=float)
    if scale is not None:
        out = out * np.asarray(scale, dtype=float)
    if R is not None:
        out = out @ np.asarray(R, dtype=float).T
    if t is not None:
        out = out + np.asarray(t, dtype=float)
    return out


# ------------------------------------------------------------------------------------------------ curves
def arclength(path):
    """Cumulative arc length along a polyline.

    Args: path: (k, d) points (any dimension). Returns: (k,) float array, first entry 0, last = total length."""
    p = np.asarray(path, dtype=float)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(path, n, closed=False):
    """Resample a polyline to equally spaced points (by arc length).

    Args: path: (k, d). n: number of points (>= 1). closed: treat path[-1] -> path[0] as a segment (do not repeat the
    first point in `path`).
    Returns: (n, d). Open paths keep both end points; closed ones return n points around the loop starting at path[0]
    (the end is not repeated). A path of zero total length returns its first point n times."""
    p = np.asarray(path, dtype=float)
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if closed:
        p = np.vstack([p, p[:1]])
    s = arclength(p)
    total = s[-1]
    if len(p) < 2 or total <= 0.0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, total, n, endpoint=not closed)
    idx = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(p) - 2)
    seg = s[idx + 1] - s[idx]
    w = np.where(seg > 0.0, (t - s[idx]) / np.where(seg > 0.0, seg, 1.0), 0.0)
    return p[idx] + (p[idx + 1] - p[idx]) * w[:, None]


def catmull_rom(points, samples_per_segment=8, closed=False, alpha=0.5):
    """Catmull-Rom spline through `points`, evaluated with the Barry-Goldman pyramid.

    Args: points: (k, d) control points (d = 3 for paths, 2 for profiles); the curve passes exactly through all of
    them. samples_per_segment: samples per control-point interval (>= 1). closed: join the last point back to the
    first (needs k >= 3, otherwise ignored; do not repeat the first point). alpha: 0.5 = centripetal (default: no
    cusps or self-loops on uneven spacing), 0 uniform, 1 chordal.
    Returns: (m, d) polyline: open -> m = (k - 1) * samples_per_segment + 1 (end points included); closed ->
    m = k * samples_per_segment (the first point is not repeated at the end). The end tangents of an open spline come
    from mirrored phantom points; coincident neighbours are handled (no NaN)."""
    P = np.asarray(points, dtype=float)
    k = len(P)
    sps = max(int(samples_per_segment), 1)
    if k < 2:
        return P.copy()
    if closed and k < 3:
        closed = False
    if closed:
        P0, P1, P2, P3 = np.roll(P, 1, 0), P, np.roll(P, -1, 0), np.roll(P, -2, 0)
    else:
        ext = np.vstack([2.0 * P[0] - P[1], P, 2.0 * P[-1] - P[-2]])
        P0, P1, P2, P3 = ext[:-3], ext[1:-2], ext[2:-1], ext[3:]
    tiny = 1e-12

    def interval(a, b):
        return np.maximum(np.linalg.norm(b - a, axis=1) ** alpha, tiny)[:, None, None]

    d01, d12, d23 = interval(P0, P1), interval(P1, P2), interval(P2, P3)
    u = (np.arange(sps) / sps)[None, :, None]
    t = d01 + u * d12                                   # t0 = 0, t1 = d01, t2 = d01 + d12, t3 = t2 + d23
    t2 = d01 + d12
    p0, p1, p2, p3 = (a[:, None, :] for a in (P0, P1, P2, P3))
    A1 = p0 + (t / d01) * (p1 - p0)
    A2 = p1 + u * (p2 - p1)
    A3 = p2 + ((t - t2) / d23) * (p3 - p2)
    B1 = A1 + (t / t2) * (A2 - A1)
    B2 = A2 + ((t - d01) / (t2 + d23 - d01)) * (A3 - A2)
    C = B1 + u * (B2 - B1)
    C[:, 0, :] = P1                                     # exact interpolation at the knots
    out = C.reshape(-1, P.shape[1])
    if not closed:
        out = np.vstack([out, P[-1:]])
    return out


def bezier(p0, p1, p2, p3, n=16):
    """Cubic Bezier curve.

    Args: p0, p1, p2, p3: control points, each (d,) (the curve starts at p0 and ends at p3). n: number of samples,
    evenly spaced in the parameter on [0, 1].
    Returns: (n, d) array; first sample p0, last p3."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    m = 1.0 - t
    return m ** 3 * p0 + 3.0 * m * m * t * p1 + 3.0 * m * t * t * p2 + t ** 3 * p3


def _fill_zero_rows(T):
    """Replace zero tangent rows by the nearest non-zero one (forward fill, then backward fill)."""
    k = len(T)
    valid = np.any(T != 0.0, axis=1)
    if valid.all():
        return T
    if not valid.any():
        return np.tile(np.array([0.0, 0.0, 1.0]), (k, 1))
    idx = np.maximum.accumulate(np.where(valid, np.arange(k), -1))
    idx = np.where(idx < 0, int(np.argmax(valid)), idx)
    return T[idx]


def _first_normal(t, up):
    """Unit vector perpendicular to the unit tangent `t`, closest to `up` (default Z; then X, Y, Z as fallbacks)."""
    cands = [np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=float)]
    cands += [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for c in cands:
        c = unit(c)
        w = c - (c @ t) * t
        n = float(np.linalg.norm(w))
        if n > 1e-6:
            return w / n
    return _perp(t)


def _transport(x0, x1, t0, t1, r):
    """One double-reflection step (Wang et al. 2008): carry the normal `r` from x0 (tangent t0) to x1 (tangent t1)."""
    v1 = x1 - x0
    c1 = float(v1 @ v1)
    if c1 < 1e-30:
        return r
    rl = r - (2.0 / c1) * float(v1 @ r) * v1
    tl = t0 - (2.0 / c1) * float(v1 @ t0) * v1
    v2 = t1 - tl
    c2 = float(v2 @ v2)
    if c2 < 1e-30:
        return rl
    return rl - (2.0 / c2) * float(v2 @ rl) * v2


def rmf_frames(path, up=None, closed=False):
    """Rotation-minimising frames along a polyline, by the double-reflection method.

    Args: path (k, 3), k >= 2. up: hint for the FIRST normal (default world Z): N[0] is `up` projected perpendicular
    to the tangent; when the tangent is parallel to it world X is used (then Y, Z). closed: the path is a loop
    (path[-1] connects to path[0], do not repeat the first point): tangents wrap and the accumulated twist of the
    transport (holonomy) is spread evenly along the loop so that frame k would coincide with frame 0.
    Returns (T, N, B), each (k, 3): T = unit tangent (central differences, one-sided at open ends; coincident points
    inherit the nearest valid tangent), N the normal, B = T x N, so N x B = T and (N, B, T) is right-handed (it is a
    rotation matrix with columns N, B, T). Frames never flip on straight paths, helices or sharp bends."""
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("rmf_frames: path must be (k >= 2, 3)")
    k = len(P)
    if closed and k < 3:
        raise ValueError("rmf_frames: a closed path needs at least 3 points")
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.empty_like(P)
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    T = _fill_zero_rows(unit(T))
    N = np.empty_like(P)
    N[0] = _first_normal(T[0], up)
    for i in range(k - 1):
        N[i + 1] = _transport(P[i], P[i + 1], T[i], T[i + 1], N[i])
    if closed:
        r_end = _transport(P[-1], P[0], T[-1], T[0], N[-1])
        phi = float(np.arctan2(T[0] @ np.cross(r_end, N[0]), r_end @ N[0]))
        if phi != 0.0:
            s = arclength(np.vstack([P, P[:1]]))
            ang = phi * s[:-1] / s[-1] if s[-1] > 0 else np.zeros(k)
            c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
            N = N * c + np.cross(T, N) * sn
    N = unit(N - np.einsum("ij,ij->i", N, T)[:, None] * T)
    B = np.cross(T, N)
    return T, N, B


def circle_profile(sides=16, radius=1.0, start=0.0):
    """CCW circle profile for sweep().

    Args: sides: number of points. radius: circle radius. start: angle (radians) of the first point, measured from
    the profile's x axis (= the frame normal N).
    Returns: (sides, 2) points radius * (cos a, sin a), a = start + 2 pi j / sides."""
    a = start + 2.0 * np.pi * np.arange(int(sides)) / int(sides)
    return np.stack([np.cos(a), np.sin(a)], axis=1) * radius


def ccw(profile):
    """Make a profile counter-clockwise (the orientation sweep() needs for outward normals).

    Args: profile: (m, 2) closed polyline (not repeating its first point).
    Returns: a new (m, 2) array: the same points, reversed when the signed area was negative."""
    p = np.asarray(profile, dtype=float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p[::-1].copy() if area < 0.0 else p.copy()


# ------------------------------------------------------------------------------------------------ Geo container
@dataclass(eq=False, repr=False)
class Geo:
    """A polygon mesh: `verts` (n, 3) float64, `faces` list of int lists (CCW from outside), `uv` per face corner
    (corners, 2) or None (v up), `face_mat` (n_faces,) int or None. Methods never modify in place; they return new
    objects. Use `check(g)` to validate, `to_mesh(...)` to produce a `part.Mesh`."""

    verts: np.ndarray
    faces: list
    uv: np.ndarray = None
    face_mat: np.ndarray = None

    def __post_init__(self):
        v = np.asarray(self.verts, dtype=float)
        if v.size == 0:
            v = v.reshape(0, 3)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"Geo.verts must be (n, 3), got {v.shape}")
        self.verts = v
        f = self.faces
        if isinstance(f, np.ndarray):
            f = f.tolist()
        elif len(f) and not (type(f[0]) is list and len(f[0]) and type(f[0][0]) is int and type(f[-1]) is list
                             and len(f[-1]) and type(f[-1][0]) is int):
            f = [[int(i) for i in x] for x in f]
        self.faces = f if isinstance(f, list) else list(f)
        if self.uv is not None:
            self.uv = np.asarray(self.uv, dtype=float).reshape(-1, 2)
        if self.face_mat is not None:
            self.face_mat = np.asarray(self.face_mat, dtype=np.int64).reshape(-1)

    def __repr__(self):
        return (f"Geo(n_verts={self.n_verts}, n_faces={self.n_faces}, uv={self.uv is not None}, "
                f"face_mat={self.face_mat is not None})")

    @property
    def n_verts(self):
        """Number of vertices."""
        return len(self.verts)

    @property
    def n_faces(self):
        """Number of faces."""
        return len(self.faces)

    @property
    def n_corners(self):
        """Total number of face corners = number of rows a UV array must have."""
        return sum(map(len, self.faces))

    def copy(self):
        """Deep copy (arrays and face lists are duplicated)."""
        return Geo(self.verts.copy(), [list(f) for f in self.faces],
                   None if self.uv is None else self.uv.copy(),
                   None if self.face_mat is None else self.face_mat.copy())

    def transformed(self, R=None, t=None, scale=None):
        """New Geo with p' = (p * scale) @ R.T + t (scale: scalar or (3,) in the local axes, applied first; R: 3x3;
        t: (3,)). The winding (and the UV corner order with it) is flipped automatically when the transform mirrors
        (negative determinant), so the faces keep pointing outward."""
        g = Geo(transform_points(self.verts, R, t, scale), [list(f) for f in self.faces],
                None if self.uv is None else self.uv.copy(),
                None if self.face_mat is None else self.face_mat.copy())
        L = np.eye(3) if R is None else np.asarray(R, dtype=float)
        if scale is not None:
            L = L * np.broadcast_to(np.asarray(scale, dtype=float), (3,))
        return g.flipped() if np.linalg.det(L) < 0.0 else g

    def translated(self, t):
        """New Geo shifted by the vector `t` (3,)."""
        return self.transformed(t=t)

    def flipped(self):
        """New Geo with every face reversed (and its UV corners reversed with it): outward becomes inward. Each corner
        keeps its UV value (uv rows are re-ordered, not changed)."""
        faces = [f[::-1] for f in self.faces]
        uv = None
        if self.uv is not None:
            sizes = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
            uv = self.uv[_reverse_index(sizes, np.cumsum(sizes) - sizes)]
        return Geo(self.verts.copy(), faces, uv, None if self.face_mat is None else self.face_mat.copy())

    def mirrored_x(self):
        """New Geo mirrored across x = 0 (x -> -x) with the winding fixed so faces still point outward; every corner
        keeps its UV value (the mirror image shares the texture layout, mirrored on the model)."""
        g = Geo(self.verts * np.array([-1.0, 1.0, 1.0]), [list(f) for f in self.faces],
                None if self.uv is None else self.uv.copy(),
                None if self.face_mat is None else self.face_mat.copy())
        return g.flipped()

    def normals(self, weight="angle"):
        """Unit vertex normals (n, 3); weight 'angle' (default), 'area' or 'uniform' (see vertex_normals)."""
        return vertex_normals(self.verts, self.faces, weight)

    def with_material(self, index):
        """New Geo whose faces all use material slot `index` (face_mat = full(n_faces, index))."""
        g = self.copy()
        g.face_mat = np.full(g.n_faces, int(index), dtype=np.int64)
        return g

    def uv_scaled(self, scale=(1.0, 1.0), offset=(0.0, 0.0)):
        """New Geo with uv' = uv * scale + offset (place a primitive's UV island in an atlas). No-op without UVs."""
        g = self.copy()
        if g.uv is not None:
            g.uv = g.uv * np.asarray(scale, dtype=float) + np.asarray(offset, dtype=float)
        return g

    def bbox(self):
        """(min corner, max corner) of the vertices, each (3,); zeros for an empty mesh."""
        if not len(self.verts):
            return np.zeros(3), np.zeros(3)
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def to_mesh(self, name, mats=(), weights=None, morphs=None, normals=None, subsurf=0, crease=None, sharp=None,
                smooth=True):
        """Build a `part.Mesh` (arrays are copied).

        Args: name: object name. mats: material names (or objects with a .name) the mesh uses; face_mat is None
        (= all mats[0]) when mats has <= 1 entry, otherwise this Geo's face_mat. weights: bone name -> (n,).
        morphs: morph name -> (n, 3) offsets. normals: (n, 3) custom vertex normals. subsurf: Catmull-Clark levels
        applied by the assembler. crease: {(i, j): 0..1}. sharp: [(i, j)] sharp edges. smooth: smooth shading.
        Raises ValueError for inconsistent shapes (uv rows, weights, morphs, normals, face_mat range)."""
        names = [getattr(m, "name", m) for m in mats]
        n = self.n_verts
        if self.uv is not None and self.uv.shape != (self.n_corners, 2):
            raise ValueError(f"to_mesh({name}): uv has {len(self.uv)} rows, faces have {self.n_corners} corners")
        fm = None
        if len(names) > 1 and self.face_mat is not None:
            fm = self.face_mat.copy()
            if len(fm) != self.n_faces:
                raise ValueError(f"to_mesh({name}): face_mat length {len(fm)} != {self.n_faces} faces")
            if len(fm) and (fm.min() < 0 or fm.max() >= len(names)):
                raise ValueError(f"to_mesh({name}): face_mat indexes outside the {len(names)} materials")
        w = {}
        for b, a in (weights or {}).items():
            a = np.asarray(a, dtype=float)
            if a.shape != (n,):
                raise ValueError(f"to_mesh({name}): weights[{b!r}] must be ({n},), got {a.shape}")
            w[b] = a.copy()
        m = {}
        for k, a in (morphs or {}).items():
            a = np.asarray(a, dtype=float)
            if a.shape != (n, 3):
                raise ValueError(f"to_mesh({name}): morph {k!r} must be ({n}, 3), got {a.shape}")
            m[k] = a.copy()
        nrm = None
        if normals is not None:
            nrm = np.asarray(normals, dtype=float)
            if nrm.shape != (n, 3):
                raise ValueError(f"to_mesh({name}): normals must be ({n}, 3), got {nrm.shape}")
            nrm = nrm.copy()
        return _part.Mesh(
            name=name, verts=self.verts.copy(), faces=[list(f) for f in self.faces],
            uv=None if self.uv is None else self.uv.copy(), face_mat=fm, mats=names, weights=w, morphs=m,
            normals=nrm, sharp=[(int(i), int(j)) for i, j in (sharp or [])], subsurf=int(subsurf),
            crease={(int(k[0]), int(k[1])): float(v) for k, v in (crease or {}).items()}, smooth=bool(smooth))


# ------------------------------------------------------------------------------------------------ analysis
def vertex_normals(verts, faces, weight="angle"):
    """Smooth per-vertex normals.

    Args: verts: (n, 3). faces: list of vertex-index lists (tris, quads, ngons). weight: 'angle' (default: every face
    contributes its unit normal times the interior angle at the vertex, so the result does not depend on how a surface
    is tessellated), 'area' (area-weighted) or 'uniform'. Quads and ngons use Newell's method; degenerate faces are
    ignored.
    Returns: (n, 3) unit vectors, zero-safe: a vertex without a valid face gets (0, 0, 0)."""
    if weight not in ("angle", "area", "uniform"):
        raise ValueError("weight must be 'angle', 'area' or 'uniform'")
    V = np.asarray(verts, dtype=float)
    n = len(V)
    N = np.zeros((n, 3))
    if len(faces) == 0 or n == 0:
        return N
    P = _Poly(faces)
    raw, ln, valid, _, c = _face_raw(V, P)
    nh = np.where(valid[:, None], raw / np.where(valid, ln, 1.0)[:, None], 0.0)
    if weight == "angle":
        e1, e2 = c[P.nxt] - c, c[P.prv] - c
        w = np.arctan2(np.linalg.norm(np.cross(e1, e2), axis=1), np.einsum("ij,ij->i", e1, e2))
    elif weight == "area":
        w = 0.5 * ln[P.fid]
    else:
        w = np.ones(P.nc)
    w = w * valid[P.fid]
    for k in range(3):
        N[:, k] = np.bincount(P.flat, weights=w * nh[P.fid, k], minlength=n)
    return unit(N)


def face_normals(verts, faces):
    """Unit face normals.

    Args: verts (n, 3); faces list of index lists. Returns: (n_faces, 3) (Newell's method, so quads and ngons work);
    degenerate faces give (0, 0, 0)."""
    if len(faces) == 0:
        return np.zeros((0, 3))
    raw, ln, valid, _, _ = _face_raw(np.asarray(verts, dtype=float), _Poly(faces))
    return np.where(valid[:, None], raw / np.where(valid, ln, 1.0)[:, None], 0.0)


def face_areas(verts, faces):
    """Face areas.

    Args: verts (n, 3); faces list of index lists. Returns: (n_faces,): half the length of the Newell vector (the area
    of the best-fit projection for non-planar polygons)."""
    if len(faces) == 0:
        return np.zeros(0)
    _, ln, _, _, _ = _face_raw(np.asarray(verts, dtype=float), _Poly(faces))
    return 0.5 * ln


def face_centers(verts, faces):
    """Face centres.

    Args: verts (n, 3); faces list of index lists. Returns: (n_faces, 3): the mean of each face's corner positions."""
    if len(faces) == 0:
        return np.zeros((0, 3))
    P = _Poly(faces)
    V = np.asarray(verts, dtype=float)
    return np.add.reduceat(V[P.flat], P.offs, axis=0) / P.sizes[:, None]


def signed_volume(verts, faces):
    """Signed volume enclosed by a face list.

    Args: verts (n, 3); faces list of index lists (ngons are fan-triangulated).
    Returns: float: positive for closed meshes whose faces point outward, negative for inside-out ones. Only
    meaningful for closed meshes (for an open surface it is the flux through it relative to the vertex centroid)."""
    if len(faces) == 0:
        return 0.0
    V = np.asarray(verts, dtype=float)
    P = _Poly(faces)
    V = V - V[P.flat].mean(axis=0)
    _, c0, c1, c2 = _fan(P.sizes, P.offs)
    A, B, C = V[P.flat[c0]], V[P.flat[c1]], V[P.flat[c2]]
    return float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)


def edges(faces):
    """Unique undirected edges of a face list.

    Args: faces: list of vertex-index lists. Returns: sorted list of (i, j) python-int tuples with i < j (edges of
    repeated vertices are skipped)."""
    if len(faces) == 0:
        return []
    P = _Poly(faces)
    a, b = P.flat, P.flat[P.nxt]
    keep = a != b
    lo, hi = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
    if not len(lo):
        return []
    M = int(P.flat.max()) + 1
    key = np.unique(lo * M + hi)
    return list(zip((key // M).tolist(), (key % M).tolist()))


def _edge_counts(P):
    """Unique undirected edges of a _Poly with their usage counts: (keys, M, counts, forward-use counts); the key of
    an edge is lo * M + hi."""
    a, b = P.flat, P.flat[P.nxt]
    keep = a != b
    a, b = a[keep], b[keep]
    M = int(P.flat.max()) + 1
    key = np.minimum(a, b) * M + np.maximum(a, b)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    fwd = np.bincount(inv, weights=(a < b).astype(float), minlength=len(uniq))
    return uniq, M, cnt, fwd


def boundary_edges(faces):
    """Border edges of a face list.

    Args: faces. Returns: sorted list of (i, j), i < j, for the edges used by exactly one face (the open border of a
    surface; empty for a closed mesh)."""
    if len(faces) == 0:
        return []
    uniq, M, cnt, _ = _edge_counts(_Poly(faces))
    k = uniq[cnt == 1]
    return list(zip((k // M).tolist(), (k % M).tolist()))


def is_closed(faces):
    """True when the mesh has no border.

    Args: faces. Returns: bool: the face list is not empty and no edge is used by a single face. Non-manifold edges
    (3+ faces) and winding consistency are reported by `check`, not here."""
    if len(faces) == 0:
        return False
    _, _, cnt, _ = _edge_counts(_Poly(faces))
    return bool(len(cnt)) and not bool((cnt == 1).any())


def adjacency(faces, n):
    """Vertex neighbours through edges.

    Args: faces; n: number of vertices. Returns: list of n sets of neighbouring vertex indices (empty for unused
    vertices)."""
    adj = [set() for _ in range(int(n))]
    for i, j in edges(faces):
        adj[i].add(j)
        adj[j].add(i)
    return adj


def components(faces, n):
    """Connected components of the vertices through shared faces.

    Args: faces; n: number of vertices. Returns: (n,) int array: the component label of every vertex (the smallest
    vertex index of its component); vertices used by no face form their own component."""
    lab = np.arange(int(n), dtype=np.int64)
    if len(faces) == 0:
        return lab
    P = _Poly(faces)
    while True:
        lf = lab[P.flat]
        fmin = np.minimum.reduceat(lf, P.offs)
        new = lab.copy()
        np.minimum.at(new, lf, fmin[P.fid])
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


def smooth(verts, faces, iterations=1, lam=0.5, pin=None, keep_boundary=True):
    """Laplacian relaxation: each free vertex moves `lam` of the way to the mean of its edge neighbours, repeated
    `iterations` times (neighbours are taken from the previous pass). pin: vertex indices or a boolean mask that
    never move; keep_boundary: vertices on border edges (see boundary_edges) never move. Returns the new (n, 3)
    array; the input is untouched. Note: plain Laplacian smoothing shrinks closed shapes."""
    V = np.array(verts, dtype=float)
    n = len(V)
    E = np.array(edges(faces), dtype=np.int64).reshape(-1, 2)
    if not len(E):
        return V
    fixed = np.zeros(n, dtype=bool)
    if pin is not None:
        pin = np.asarray(pin)
        if pin.dtype == bool:
            fixed |= pin
        elif pin.size:
            fixed[pin.astype(np.int64)] = True
    if keep_boundary:
        be = boundary_edges(faces)
        if be:
            fixed[np.array(be, dtype=np.int64).ravel()] = True
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    move = (~fixed) & (deg > 0)
    for _ in range(int(iterations)):
        S = np.empty_like(V)
        for k in range(3):
            S[:, k] = (np.bincount(E[:, 0], weights=V[E[:, 1], k], minlength=n)
                       + np.bincount(E[:, 1], weights=V[E[:, 0], k], minlength=n))
        V[move] += lam * (S[move] / deg[move, None] - V[move])
    return V


def check(g):
    """Diagnose a Geo; returns a dict:
      n_verts, n_faces: sizes.
      bad_indices: face corners whose vertex index is outside 0..n_verts-1 (the geometric entries are None then).
      n_degenerate: faces with a repeated vertex index or (near) zero area.
      boundary_edges: edges used by one face (0 for a closed surface).
      non_manifold_edges: edges used by 3 or more faces.
      inconsistent_edges: edges shared by two faces that traverse it in the same direction (mixed winding).
      signed_volume: float (positive for outward-facing closed meshes) or None.
      unused_verts: vertices that no face uses.
      uv_ok: uv is None or finite with shape (n_corners, 2).
      mat_ok: face_mat is None or has one entry per face."""
    V = g.verts
    n, nf = len(V), len(g.faces)
    out = dict(n_verts=n, n_faces=nf, bad_indices=0, n_degenerate=0, boundary_edges=0, non_manifold_edges=0,
               inconsistent_edges=0, signed_volume=0.0, unused_verts=n, uv_ok=True, mat_ok=True)
    ncorn = sum(map(len, g.faces))
    if g.uv is not None:
        out["uv_ok"] = bool(g.uv.ndim == 2 and g.uv.shape == (ncorn, 2) and np.isfinite(g.uv).all())
    if g.face_mat is not None:
        out["mat_ok"] = bool(g.face_mat.shape == (nf,))
    if nf == 0:
        return out
    P = _Poly(g.faces)
    bad = int(((P.flat < 0) | (P.flat >= n)).sum())
    out["bad_indices"] = bad
    dup = _repeated_vertex_faces(P)
    _, _, cnt, fwd = _edge_counts(P)
    out["boundary_edges"] = int((cnt == 1).sum())
    out["non_manifold_edges"] = int((cnt > 2).sum())
    out["inconsistent_edges"] = int(((cnt == 2) & ((fwd == 0) | (fwd == 2))).sum())
    if bad:
        out["n_degenerate"] = int(dup.sum())
        out["signed_volume"] = None
        out["unused_verts"] = None
        return out
    _, _, valid, _, _ = _face_raw(V, P)
    out["n_degenerate"] = int((dup | ~valid).sum())
    out["unused_verts"] = int(n - len(np.unique(P.flat)))
    out["signed_volume"] = signed_volume(V, g.faces)
    return out


# ------------------------------------------------------------------------------------------------ topology
def _planar_uv(pts, fallback_normal=None):
    """UVs (k, 2) for the points of a (roughly planar) polygon: projected on its best-fit plane, centred at (0.5, 0.5)
    and scaled uniformly so the polygon fits the unit square; seen from outside u runs right and v up."""
    pts = np.asarray(pts, dtype=float)
    a = pts - pts.mean(axis=0)
    n = unit(np.cross(a, np.roll(a, -1, axis=0)).sum(axis=0))
    if not n.any():
        n = unit(np.array([0.0, 0.0, 1.0]) if fallback_normal is None else fallback_normal)
        if not n.any():
            n = np.array([0.0, 0.0, 1.0])
    F = frame_from(z=n)
    xy = np.stack([a @ F[:, 0], a @ F[:, 1]], axis=1)
    ext = float(np.abs(xy).max())
    if ext < 1e-15:
        return np.full((len(pts), 2), 0.5)
    return 0.5 + 0.5 * xy / ext


def _ear_clip(P3):
    """Ear-clip a (possibly concave) polygon given by its 3D points in order. Works in the best-fit plane of the
    polygon (Newell normal); returns a list of (i, j, k) index triples in polygon order, each wound like the polygon."""
    m = len(P3)
    c = P3.mean(axis=0)
    a = P3 - c
    n = np.cross(a, np.roll(a, -1, axis=0)).sum(axis=0)
    if np.linalg.norm(n) < 1e-18:
        return [(0, i, i + 1) for i in range(1, m - 1)]
    F = frame_from(z=n)
    Q = np.stack([a @ F[:, 0], a @ F[:, 1]], axis=1)                   # CCW in this plane (positive area)
    ext = max(float(np.abs(Q).max()), 1e-300)
    eps, ceps = 1e-12 * ext ** 2, (1e-9 * ext) ** 2
    pr0, nx0 = np.roll(Q, 1, axis=0), np.roll(Q, -1, axis=0)
    cr0 = (Q[:, 0] - pr0[:, 0]) * (nx0[:, 1] - Q[:, 1]) - (Q[:, 1] - pr0[:, 1]) * (nx0[:, 0] - Q[:, 0])
    convex = bool((cr0 > eps).all())            # strictly convex (no reflex or collinear corner): no ear can be blocked
    idx = list(range(m))
    tris = []
    while len(idx) > 3:
        pts = Q[idx]
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        e1, e2 = pts - prev, nxt - pts
        crs = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]                # > 0: convex corner
        if convex:
            ok = crs > eps
        else:
            # inside[i, j]: vertex j lies inside or ON the border of the ear triangle (prev[i], pts[i], nxt[i]) (a
            # diagonal through another vertex would leave a degenerate sliver behind), unless j duplicates one of
            # the ear's corners
            d = pts[None, :, :] - prev[:, None, :]
            s1 = e1[:, None, 0] * d[:, :, 1] - e1[:, None, 1] * d[:, :, 0]
            d2 = pts[None, :, :] - pts[:, None, :]
            e3 = nxt - pts
            s2 = e3[:, None, 0] * d2[:, :, 1] - e3[:, None, 1] * d2[:, :, 0]
            d3 = pts[None, :, :] - nxt[:, None, :]
            e4 = prev - nxt
            s3 = e4[:, None, 0] * d3[:, :, 1] - e4[:, None, 1] * d3[:, :, 0]
            dup = ((d * d).sum(-1) <= ceps) | ((d2 * d2).sum(-1) <= ceps) | ((d3 * d3).sum(-1) <= ceps)
            inside = (s1 >= -eps) & (s2 >= -eps) & (s3 >= -eps) & ~dup
            ok = (crs > eps) & ~inside.any(axis=1)
        if ok.any():
            la, lb, lc = (e1 * e1).sum(1), (e2 * e2).sum(1), ((nxt - prev) ** 2).sum(1)
            q = np.where(ok, 4.0 * np.sqrt(3.0) * 0.5 * crs / np.maximum(la + lb + lc, 1e-300), -1.0)
            i = int(np.argmax(q))
        else:
            loose = crs > -eps
            i = int(np.argmax(loose)) if loose.any() else int(np.argmax(crs))
        tris.append((idx[i - 1], idx[i], idx[(i + 1) % len(idx)]))
        del idx[i]
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def triangulate(g, method="ear", only_ngons=False):
    """Triangulate faces; UV corners, face_mat and vertices are carried over.

    Args: g: Geo. method: 'fan' (fan from each face's first corner: right for convex faces) or 'ear' (default): quads
    are split along the diagonal that keeps both triangles valid and is the shorter one, polygons with 5+ corners are
    ear-clipped in their best-fit plane (handles concave outlines and collinear vertices, picks the best-shaped ear
    first). only_ngons: keep tris and quads untouched and only split faces with 5+ corners.
    Returns: a new Geo; the face order is preserved (a face's triangles stay together, in order)."""
    if method not in ("ear", "fan"):
        raise ValueError("method must be 'ear' or 'fan'")
    if g.n_faces == 0:
        return g.copy()
    P = _Poly(g.faces)
    V = g.verts
    sizes, offs = P.sizes, P.offs
    tri_src, tri_cor, keep4_src, keep4_cor = [], [], [], []
    f3 = np.nonzero(sizes == 3)[0]
    if len(f3):
        tri_src.append(f3)
        tri_cor.append(offs[f3][:, None] + np.arange(3))
    f4 = np.nonzero(sizes == 4)[0]
    if len(f4):
        cor = offs[f4][:, None] + np.arange(4)
        if only_ngons:
            keep4_src.append(f4)
            keep4_cor.append(cor)
        else:
            use_bd = np.zeros(len(f4), dtype=bool)
            if method == "ear":
                A, B, C, D = (V[P.flat[cor[:, i]]] for i in range(4))
                n = np.cross(C - A, D - B)
                ok_ac = (np.einsum("ij,ij->i", np.cross(B - A, C - A), n) > 0) & \
                        (np.einsum("ij,ij->i", np.cross(C - A, D - A), n) > 0)
                ok_bd = (np.einsum("ij,ij->i", np.cross(B - A, D - A), n) > 0) & \
                        (np.einsum("ij,ij->i", np.cross(C - B, D - B), n) > 0)
                lac, lbd = np.linalg.norm(C - A, axis=1), np.linalg.norm(D - B, axis=1)
                use_bd = (ok_bd & ~ok_ac) | (ok_ac & ok_bd & (lbd < lac * (1.0 - 1e-9)))
            t1 = np.where(use_bd[:, None], cor[:, [0, 1, 3]], cor[:, [0, 1, 2]])
            t2 = np.where(use_bd[:, None], cor[:, [1, 2, 3]], cor[:, [0, 2, 3]])
            tri_src += [f4, f4]
            tri_cor += [t1, t2]
    fn = np.nonzero(sizes > 4)[0]
    if len(fn):
        srcs, cors = [], []
        for fi in fn.tolist():
            o, s = int(offs[fi]), int(sizes[fi])
            loc = [(0, i, i + 1) for i in range(1, s - 1)] if method == "fan" \
                else _ear_clip(V[P.flat[o:o + s]])
            for t in loc:
                srcs.append(fi)
                cors.append([o + t[0], o + t[1], o + t[2]])
        tri_src.append(np.array(srcs, dtype=np.int64))
        tri_cor.append(np.array(cors, dtype=np.int64))
    src3 = np.concatenate(tri_src) if tri_src else np.zeros(0, dtype=np.int64)
    cor3 = np.concatenate(tri_cor) if tri_cor else np.zeros((0, 3), dtype=np.int64)
    src4 = np.concatenate(keep4_src) if keep4_src else np.zeros(0, dtype=np.int64)
    cor4 = np.concatenate(keep4_cor) if keep4_cor else np.zeros((0, 4), dtype=np.int64)
    # merge the two groups back into the original face order (stable: a face's triangles stay in generation order)
    src_all = np.concatenate([src3, src4])
    order = np.argsort(src_all, kind="stable")
    size_all = np.concatenate([np.full(len(src3), 3), np.full(len(src4), 4)])[order]
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    out_offs = np.cumsum(size_all) - size_all
    out_cor = np.empty(int(size_all.sum()), dtype=np.int64)
    if len(src3):
        out_cor[(out_offs[rank[:len(src3)]][:, None] + np.arange(3)).ravel()] = cor3.ravel()
    if len(src4):
        out_cor[(out_offs[rank[len(src3):]][:, None] + np.arange(4)).ravel()] = cor4.ravel()
    out_src = src_all[order]
    return Geo(V.copy(), _unflatten(P.flat[out_cor], size_all),
               None if g.uv is None else g.uv[out_cor],
               None if g.face_mat is None else g.face_mat[out_src])


def merge(geos):
    """Concatenate Geos into one.

    Args: geos: iterable of Geo (an empty one gives an empty Geo). Vertex indices are offset. UVs: when only some
    inputs have a uv array the others contribute (0, 0) corners (None when none has); face_mat: kept, with None
    counting as material 0 when any other input has face_mat (None when none has).
    Returns: a new Geo (the inputs are not modified)."""
    geos = list(geos)
    if not geos:
        return Geo(np.zeros((0, 3)), [])
    verts = np.concatenate([g.verts for g in geos])
    faces = []
    off = 0
    for g in geos:
        faces.extend(_offset_faces(g.faces, off))
        off += g.n_verts
    uv = None
    if any(g.uv is not None for g in geos):
        uv = np.concatenate([g.uv if g.uv is not None else np.zeros((g.n_corners, 2)) for g in geos])
    fm = None
    if any(g.face_mat is not None for g in geos):
        fm = np.concatenate([g.face_mat if g.face_mat is not None else np.zeros(g.n_faces, dtype=np.int64)
                             for g in geos])
    return Geo(verts, faces, uv, fm)


def merge_meshes(meshes, name):
    """Concatenate `part.Mesh` objects into one Mesh called `name`.

    Args: meshes: list of part.Mesh (at least one). name: name of the result.
    Returns: a new part.Mesh: verts / faces / uv concatenated (indices offset; a mesh without uv contributes zeros
    while another has them, None only if all are None). `mats` = the unique material names in order of first use and
    face_mat is remapped (a mesh with face_mat None uses its mats[0]; a mesh without materials next to meshes that
    have some raises ValueError; face_mat is None when only one material remains). weights and morphs are the union of
    the keys, zero-filled where a mesh lacks them. normals: when any mesh has custom normals the others get computed
    angle-weighted vertex normals. sharp / crease indices are offset; smooth = all(smooth); the meshes must agree on
    `subsurf` (ValueError otherwise)."""
    meshes = list(meshes)
    if not meshes:
        raise ValueError("merge_meshes needs at least one mesh")
    levels = {m.subsurf for m in meshes}
    if len(levels) > 1:
        raise ValueError(f"merge_meshes({name}): meshes disagree on subsurf: {sorted(levels)}")
    ns = [len(m.verts) for m in meshes]
    offs = np.cumsum([0] + ns)
    total = int(offs[-1])
    verts = np.concatenate([np.asarray(m.verts, dtype=float).reshape(-1, 3) for m in meshes])
    faces = []
    for m, o in zip(meshes, offs[:-1]):
        faces.extend(_offset_faces(m.faces, int(o)))
    corners = [sum(map(len, m.faces)) for m in meshes]
    uv = None
    if any(m.uv is not None for m in meshes):
        uv = np.concatenate([np.asarray(m.uv, dtype=float).reshape(-1, 2) if m.uv is not None else np.zeros((c, 2))
                             for m, c in zip(meshes, corners)])
    mats, index = [], {}
    for m in meshes:
        for nm in m.mats:
            if nm not in index:
                index[nm] = len(mats)
                mats.append(nm)
    face_mat = None
    if mats:
        parts = []
        for m in meshes:
            if not m.mats:
                raise ValueError(f"merge_meshes({name}): mesh {m.name!r} has no materials but others do")
            gmap = np.array([index[nm] for nm in m.mats], dtype=np.int64)
            local = np.zeros(len(m.faces), dtype=np.int64) if m.face_mat is None \
                else np.asarray(m.face_mat, dtype=np.int64)
            parts.append(gmap[local])
        if len(mats) > 1:
            face_mat = np.concatenate(parts)
    weights, morphs = {}, {}
    for m, o, n in zip(meshes, offs[:-1], ns):
        for b, w in m.weights.items():
            weights.setdefault(b, np.zeros(total))[o:o + n] = np.asarray(w, dtype=float)
        for k, d in m.morphs.items():
            morphs.setdefault(k, np.zeros((total, 3)))[o:o + n] = np.asarray(d, dtype=float)
    normals = None
    if any(m.normals is not None for m in meshes):
        normals = np.concatenate([np.asarray(m.normals, dtype=float) if m.normals is not None
                                  else vertex_normals(m.verts, m.faces) for m in meshes])
    sharp, crease = [], {}
    for m, o in zip(meshes, offs[:-1]):
        o = int(o)
        sharp.extend((int(i) + o, int(j) + o) for i, j in m.sharp)
        crease.update({(int(k[0]) + o, int(k[1]) + o): float(v) for k, v in m.crease.items()})
    return _part.Mesh(name=name, verts=verts, faces=faces, uv=uv, face_mat=face_mat, mats=mats, weights=weights,
                      morphs=morphs, normals=normals, sharp=sharp, subsurf=int(levels.pop()), crease=crease,
                      smooth=all(m.smooth for m in meshes))


def _hash3(k):
    """Spatial hash of integer cell coordinates (n, 3); wraps around on overflow by design (collisions are verified
    away by the caller)."""
    with np.errstate(over="ignore"):
        return (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)


_HALF_OFFSETS = np.array([d for d in itertools.product((-1, 0, 1), repeat=3)
                          if d > (0, 0, 0)], dtype=np.int64)               # 13 offsets: every cell pair once


def _near_cell_pairs(cells, rep, tol):
    """Pairs of occupied cells (row indices) that are neighbours and whose representative points are within tol."""
    h = _hash3(cells)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    ia_all, jb_all = [], []
    for off in _HALF_OFFSETS:
        nk = cells + off
        hn = _hash3(nk)
        lo = np.searchsorted(hs, hn, side="left")
        cnt = np.searchsorted(hs, hn, side="right") - lo
        sel = np.nonzero(cnt > 0)[0]
        if not len(sel):
            continue
        cnt, lo = cnt[sel], lo[sel]
        ia = np.repeat(sel, cnt)
        pos = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
        jb = order[pos]
        ok = (cells[jb] == nk[ia]).all(axis=1)
        ia, jb = ia[ok], jb[ok]
        close = np.linalg.norm(rep[ia] - rep[jb], axis=1) <= tol
        ia_all.append(ia[close])
        jb_all.append(jb[close])
    if not ia_all:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ia_all), np.concatenate(jb_all)


def _simplify_polygon(vs, cs):
    """Remove repeated vertices from a polygon (vertex list `vs`, corner ids `cs`): consecutive repeats are dropped,
    a polygon that touches itself is split at the repeated vertex; pieces with < 3 vertices vanish. Returns a list of
    (vertices, corners)."""
    out = []
    stack = [(list(vs), list(cs))]
    while stack:
        vs, cs = stack.pop()
        changed = True
        while changed and len(vs) >= 2:
            changed = False
            for i in range(len(vs)):
                j = (i + 1) % len(vs)
                if vs[i] == vs[j]:
                    del vs[j]
                    del cs[j]
                    changed = True
                    break
        if len(vs) < 3:
            continue
        seen = {}
        for i, x in enumerate(vs):
            if x in seen:
                j = seen[x]
                stack.append((vs[:j] + vs[i:], cs[:j] + cs[i:]))
                stack.append((vs[j:i], cs[j:i]))
                break
            seen[x] = i
        else:
            out.append((vs, cs))
    return out


def weld(g, tol=1e-6):
    """Merge coincident vertices -> (new Geo, remap) with remap[old index] = new index.

    Vertices are binned into cubic cells of size `tol`; vertices in the same cell merge, and vertices in adjacent
    cells merge when their cell representatives are within `tol` (so a seam never survives because it straddles a
    cell border). New vertices keep the order of their first member and sit at the members' mean (exactly the shared
    position when members are identical). Faces are re-indexed; consecutive repeated vertices are dropped (a quad can
    become a triangle), self-touching faces are split, faces left with fewer than 3 vertices are removed; UV corners of
    surviving corners and face_mat are carried over. Vertices that no face uses are kept (merged like any other)."""
    if tol <= 0:
        raise ValueError("weld: tol must be > 0")
    V = g.verts
    n = len(V)
    if n == 0:
        return g.copy(), np.zeros(0, dtype=np.int64)
    key = np.rint(V / tol).astype(np.int64)
    cells, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    U = len(cells)
    root = np.arange(U)
    if U > 1:
        ia, jb = _near_cell_pairs(cells, V[first], tol)
        if len(ia):
            parent = list(range(U))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for x, y in zip(ia.tolist(), jb.tolist()):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
            for i in np.unique(np.concatenate([ia, jb])).tolist():
                root[i] = find(i)
    _, cl = np.unique(root, return_inverse=True)
    cl = cl.reshape(-1)
    K = int(cl.max()) + 1
    cfirst = np.full(K, n, dtype=np.int64)
    np.minimum.at(cfirst, cl, first)
    rank = np.empty(K, dtype=np.int64)
    rank[np.argsort(cfirst, kind="stable")] = np.arange(K)
    remap = rank[cl[inv]]
    anchor = np.empty(K, dtype=np.int64)                         # first member (vertex index) of each new vertex
    anchor[rank] = cfirst
    d = V - V[anchor[remap]]
    cnt = np.bincount(remap, minlength=K).astype(float)
    newV = V[anchor] + np.stack([np.bincount(remap, weights=d[:, k], minlength=K) for k in range(3)], axis=1) \
        / cnt[:, None]
    if g.n_faces == 0:
        return Geo(newV, [], g.uv, g.face_mat), remap
    P = _Poly(g.faces)
    new_flat = remap[P.flat]
    bad = np.nonzero(_repeated_vertex_faces(_PolyView(P, new_flat)))[0]
    if not len(bad):
        return Geo(newV, _unflatten(new_flat, P.sizes), None if g.uv is None else g.uv.copy(),
                   None if g.face_mat is None else g.face_mat.copy()), remap
    badset = set(bad.tolist())
    nfl = new_flat.tolist()
    faces, cor, src = [], [], []
    for fi in range(P.nf):
        o, s = int(P.offs[fi]), int(P.sizes[fi])
        if fi in badset:
            for vs, cs in _simplify_polygon(nfl[o:o + s], list(range(o, o + s))):
                faces.append(vs)
                cor.append(cs)
                src.append(fi)
        else:
            faces.append(nfl[o:o + s])
            cor.append(range(o, o + s))
            src.append(fi)
    corner = np.fromiter(itertools.chain.from_iterable(cor), dtype=np.int64)
    src = np.array(src, dtype=np.int64)
    return Geo(newV, faces, None if g.uv is None else g.uv[corner],
               None if g.face_mat is None else g.face_mat[src]), remap


class _PolyView:
    """A _Poly with replaced vertex indices (internal; for duplicate-vertex detection after re-indexing)."""

    def __init__(self, P, flat):
        self.nf, self.nc, self.fid, self.flat = P.nf, P.nc, P.fid, flat


def subset(g, which):
    """Keep only some faces.

    Args: g: Geo. which: boolean mask (n_faces,) or an array of face indices (kept in that order).
    Returns: (new Geo with the vertices compacted, keep): `keep` holds the old index of every kept vertex, in
    ascending order (slice per-vertex data with it: weights[b][keep]). UV corners and face_mat follow the faces."""
    which = np.asarray(which)
    idx = np.nonzero(which)[0] if which.dtype == bool else which.astype(np.int64).reshape(-1)
    if g.n_faces == 0 or not len(idx):
        return Geo(np.zeros((0, 3)), [], None if g.uv is None else np.zeros((0, 2)),
                   None if g.face_mat is None else np.zeros(0, dtype=np.int64)), np.zeros(0, dtype=np.int64)
    P = _Poly(g.faces)
    cor = _gather_corners(P.offs, P.sizes, idx)
    used = P.flat[cor]
    keep = np.unique(used)
    new = np.searchsorted(keep, used)
    return Geo(g.verts[keep], _unflatten(new, P.sizes[idx]), None if g.uv is None else g.uv[cor],
               None if g.face_mat is None else g.face_mat[idx]), keep


def ensure_outward(g):
    """Make closed shells face outward.

    Args: g: Geo, possibly several shells merged together. Returns: a new Geo in which every connected component that
    has no border edges and a negative signed volume has its faces (and UV corners) reversed; open components and
    correctly oriented shells are left as they are."""
    if g.n_faces == 0:
        return g.copy()
    P = _Poly(g.faces)
    lab = components(g.faces, g.n_verts)
    fcomp = lab[P.flat[P.offs]]
    V = g.verts - g.verts[P.flat].mean(axis=0)
    tf, c0, c1, c2 = _fan(P.sizes, P.offs)
    vol_t = np.einsum("ij,ij->i", V[P.flat[c0]], np.cross(V[P.flat[c1]], V[P.flat[c2]])) / 6.0
    vol = np.bincount(fcomp[tf], weights=vol_t, minlength=g.n_verts)
    uniq, M, cnt, _ = _edge_counts(P)
    border = uniq[cnt == 1]
    open_comp = np.zeros(g.n_verts, dtype=bool)
    if len(border):
        open_comp[lab[border // M]] = True
    flip = (vol[fcomp] < 0.0) & ~open_comp[fcomp]
    if not flip.any():
        return g.copy()
    rev = _reverse_index(P.sizes, P.offs)
    perm = np.arange(P.nc)
    sel = np.repeat(flip, P.sizes)
    perm[sel] = rev[sel]
    return Geo(g.verts.copy(), _unflatten(P.flat[perm], P.sizes), None if g.uv is None else g.uv[perm],
               None if g.face_mat is None else g.face_mat.copy())


def _drop_cancelling_faces(g):
    """Remove pairs of faces that cover the same vertices with opposite winding (internal faces)."""
    def canon(f):
        m = f.index(min(f))
        return tuple(f[m:] + f[:m])

    pending, drop = {}, set()
    for i, f in enumerate(g.faces):
        r = canon(f[::-1])
        waiting = pending.get(r)
        if waiting:
            drop.add(i)
            drop.add(waiting.pop())
        else:
            pending.setdefault(canon(f), []).append(i)
    if not drop:
        return g
    keep = np.ones(g.n_faces, dtype=bool)
    keep[list(drop)] = False
    idx = np.nonzero(keep)[0]
    P = _Poly(g.faces)
    cor = _gather_corners(P.offs, P.sizes, idx)
    return Geo(g.verts.copy(), [list(g.faces[i]) for i in idx.tolist()], None if g.uv is None else g.uv[cor],
               None if g.face_mat is None else g.face_mat[idx])


def mirror_x(g, weld_tol=1e-6):
    """`g` plus its mirror image across x = 0, welded along the seam: returns a new Geo (g's vertices come first, then
    the mirrored ones that did not merge). Intended for half shapes that are open at x = 0 (their border vertices on the
    plane merge with their mirror and sit exactly at x = 0); a face lying in the x = 0 plane that cancels with its
    mirror (same vertices, opposite winding) is removed. UV corners are mirrored 'as is' (see Geo.mirrored_x)."""
    both = merge([g, g.mirrored_x()])
    w, _ = weld(both, weld_tol)
    return _drop_cancelling_faces(w)


# ------------------------------------------------------------------------------------------------ surfaces
def _cap_kind(c):
    if c is None or c is False:
        return None
    if c is True:
        return "ngon"
    if c in ("fan", "ngon"):
        return c
    raise ValueError(f"cap must be None, 'fan' or 'ngon', got {c!r}")


def _loft_uv_tables(R3, closed, loop, u, v):
    """(U, V) tables of shape (rows, cols) for a loft: rows = rings (+1 when looped), cols = points (+1 when closed)."""
    nr, nv = R3.shape[:2]
    R = nr + (1 if loop else 0)
    C = nv + (1 if closed else 0)
    if u is None:
        U = np.broadcast_to(np.linspace(0.0, 1.0, C), (R, C))
    elif isinstance(u, str):
        if u != "arc":
            raise ValueError("u must be None, 'arc' or an array")
        pts = R3[np.arange(R) % nr][:, np.arange(C) % nv]
        seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
        cum = np.concatenate([np.zeros((R, 1)), np.cumsum(seg, axis=1)], axis=1)
        tot = cum[:, -1:]
        U = np.where(tot > 0.0, cum / np.where(tot > 0.0, tot, 1.0), np.linspace(0.0, 1.0, C))
    else:
        U = np.asarray(u, dtype=float)
        if U.shape == (C,):
            U = np.broadcast_to(U, (R, C))
        elif U.shape != (R, C):
            raise ValueError(f"u must have shape ({C},) or ({R}, {C}), got {U.shape}")
    if v is None:
        V = np.broadcast_to(np.linspace(0.0, 1.0, R)[:, None], (R, C))
    elif isinstance(v, str):
        if v != "arc":
            raise ValueError("v must be None, 'arc' or an array")
        s = arclength(R3.mean(axis=1)[np.arange(R) % nr])
        V = np.broadcast_to((s / s[-1] if s[-1] > 0.0 else np.linspace(0.0, 1.0, R))[:, None], (R, C))
    else:
        V = np.asarray(v, dtype=float)
        if V.shape == (R,):
            V = np.broadcast_to(V[:, None], (R, C))
        elif V.shape != (R, C):
            raise ValueError(f"v must have shape ({R},) or ({R}, {C}), got {V.shape}")
    return U, V


def loft(rings, closed=True, cap_start=None, cap_end=None, uv=True, u=None, v=None, flip=False, loop=False):
    """Skin a stack of rings with quads.

    Args:
      rings: (nr, nv, 3) ordered along the travel direction; the points of each ring run counter-clockwise AROUND the
        travel direction (right-hand rule), which makes the quad normals point outward.
      closed: ring points wrap (last connects to first); False gives an open strip (nv - 1 quads per row).
      cap_start / cap_end: None, 'fan' (centre vertex + triangles) or 'ngon' (one polygon) (True = 'ngon'); the caps
        face away from the tube (start cap against the travel direction, end cap along it).
      uv: build UVs. u: across the ring, v: along the rings. Default u = linspace(0, 1, nv + 1) for closed rings
        (the seam has its own 0 and 1 corners; nv for open ones) and v = linspace(0, 1, nr) by ring index.
        'arc' gives the arc-length fraction (u: per ring, v: of the ring centroids' path); or pass an array:
        u (cols,) or (rows, cols); v (rows,) or (rows, cols), where cols = nv (+1 if closed), rows = nr (+1 if loop).
        Caps get a planar projection of their ring.
      flip: reverse every face (inside-out).
      loop: also connect the last ring to the first (torus-like; no caps then; v gets its own closing row).
    Returns Geo with vertex (ring r, point j) at index r * nv + j, side quads first (ring-major), then the start cap,
    then the end cap; fan centres are appended after the ring vertices (start first)."""
    R3 = np.asarray(rings, dtype=float)
    if R3.ndim != 3 or R3.shape[2] != 3:
        raise ValueError("rings must have shape (n_rings, n_points, 3)")
    nr, nv = R3.shape[:2]
    cs, ce = _cap_kind(cap_start), _cap_kind(cap_end)
    if loop and (cs or ce):
        raise ValueError("a looped loft has no ends to cap")
    if nr < (3 if loop else 2):
        raise ValueError(f"loft needs at least {3 if loop else 2} rings, got {nr}")
    if nv < (3 if closed else 2):
        raise ValueError(f"loft needs at least {3 if closed else 2} points per ring, got {nv}")
    nrq = nr - 1 + (1 if loop else 0)
    ncq = nv - 1 + (1 if closed else 0)
    rr, cc = np.meshgrid(np.arange(nrq), np.arange(ncq), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    r1, c1 = (rr + 1) % nr, (cc + 1) % nv
    faces = np.stack([rr * nv + cc, rr * nv + c1, r1 * nv + c1, r1 * nv + cc], axis=1).tolist()
    verts = [R3.reshape(-1, 3)]
    uvs = []
    if uv:
        U, V = _loft_uv_tables(R3, closed, loop, u, v)
        T = np.stack([U, V], axis=-1)
        uvs.append(np.stack([T[rr, cc], T[rr, cc + 1], T[rr + 1, cc + 1], T[rr + 1, cc]], axis=1).reshape(-1, 2))
    extra = nr * nv
    for kind, ring, forward in ((cs, 0, False), (ce, nr - 1, True)):
        if kind is None:
            continue
        order = np.arange(nv) if forward else np.arange(nv)[::-1]
        pts = R3[ring][order]
        a = ring * nv + order
        if uv:
            away = R3[ring].mean(axis=0) - R3[ring - 1 if forward else ring + 1].mean(axis=0)
            puv = _planar_uv(pts, away)                    # fallback normal: pointing away from the tube
        if kind == "ngon":
            faces.append(a.tolist())
            if uv:
                uvs.append(puv)
        else:
            verts.append(R3[ring].mean(axis=0)[None, :])
            faces.extend(np.stack([np.full(nv, extra), a, np.roll(a, -1)], axis=1).tolist())
            if uv:
                uvs.append(np.stack([np.full((nv, 2), 0.5), puv, np.roll(puv, -1, axis=0)], axis=1).reshape(-1, 2))
            extra += 1
    g = Geo(np.concatenate(verts), faces, np.concatenate(uvs) if uv else None)
    return g.flipped() if flip else g


def _per_ring(x, k, name, width):
    """Broadcast a per-ring parameter: scalar, (k,) or (k, 2) -> (k, width)."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return np.full((k, width), float(a))
    if a.shape == (k,):
        return np.repeat(a[:, None], width, axis=1)
    if width == 2 and a.shape == (k, 2):
        return a
    raise ValueError(f"{name} must be a scalar, ({k},)" + (f" or ({k}, 2)" if width == 2 else "") +
                     f", got shape {a.shape}")


def _arc_fraction(points, closed):
    """Cumulative arc-length fraction along a polyline (m, d): (m + closed,) from 0 to 1."""
    p = np.asarray(points, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    s = arclength(p)
    return s / s[-1] if s[-1] > 0.0 else np.linspace(0.0, 1.0, len(p))


def sweep(profile, path, scale=1.0, twist=0.0, up=None, closed_profile=True, closed_path=False, cap_start=None,
          cap_end=None, uv=True, u="arc", v="arc"):
    """Sweep a 2D profile along a path with rotation-minimising frames.

    Args:
      profile: (m, 2) points in the (N, B) plane of each frame (x along N, y along B); counter-clockwise in that plane
        == counter-clockwise around the tangent, which gives outward normals (see `ccw`, `circle_profile`).
      path: (k, 3). up: hint for the first normal (see rmf_frames; default world Z).
      scale: scalar, (k,) or (k, 2): size of the profile at each path point (sx along N, sy along B).
      twist: scalar or (k,) radians: rotation of the (scaled) profile about the tangent at each path point; a scalar
        is the same roll everywhere (use np.linspace(0, total, k) for a progressive twist).
      closed_profile: profile is a closed loop (else an open strip). closed_path: the path is a loop (rings wrap, no
        caps). cap_start / cap_end: None | 'fan' | 'ngon' (closed profiles).
      uv: build UVs; u: 'arc' (profile arc-length fraction, default), None (by index) or an array (see loft);
        v: 'arc' (path arc-length fraction, default), None (by ring index) or an array.
    Returns Geo with the loft vertex layout: vertex (path point r, profile point j) = r * m + j."""
    prof = np.asarray(profile, dtype=float)
    if prof.ndim != 2 or prof.shape[1] != 2:
        raise ValueError("profile must be (m, 2)")
    P = np.asarray(path, dtype=float)
    k = len(P)
    _, N, B = rmf_frames(P, up, closed_path)
    sc = _per_ring(scale, k, "scale", 2)
    tw = _per_ring(twist, k, "twist", 1)[:, 0]
    px = prof[None, :, 0] * sc[:, 0:1]
    py = prof[None, :, 1] * sc[:, 1:2]
    if np.any(tw != 0.0):
        c, s = np.cos(tw)[:, None], np.sin(tw)[:, None]
        px, py = c * px - s * py, s * px + c * py
    rings = P[:, None, :] + px[..., None] * N[:, None, :] + py[..., None] * B[:, None, :]
    if isinstance(u, str) and u == "arc":
        u = _arc_fraction(prof, closed_profile)
    if isinstance(v, str) and v == "arc":
        s = arclength(np.vstack([P, P[:1]]) if closed_path else P)
        v = s / s[-1] if s[-1] > 0.0 else None
    return loft(rings, closed=closed_profile, cap_start=cap_start, cap_end=cap_end, uv=uv, u=u, v=v,
                loop=closed_path)


def tube(path, radius, sides=8, cap_start=None, cap_end=None, closed_path=False, up=None, uv=True):
    """Round tube along a path: sweep() of a circle.

    Args: path: (k, 3). radius: scalar, (k,) or (k, 2) (elliptical: sx along N, sy along B). sides: points per ring
    (the first lies along the frame normal N). cap_start / cap_end: None | 'fan' | 'ngon'. closed_path: the path is a
    loop (torus-like, no caps). up: first-normal hint (see rmf_frames). uv: build UVs (u around 0..1 with its own
    seam corner, v = arc-length fraction along the path).
    Returns: Geo with vertex (path point r, side j) = r * sides + j. A radius of exactly 0 (pointed tip) collapses
    that ring to a point: its quads become triangles in disguise (a fan cap there would be degenerate); run weld() to
    turn it into a real tip."""
    return sweep(circle_profile(sides), path, scale=radius, up=up, closed_profile=True, closed_path=closed_path,
                 cap_start=cap_start, cap_end=cap_end, uv=uv)


def ribbon(path, width, up=None, thickness=0.0, uv=True, closed_path=False):
    """Flat strip along a path.

    width: scalar or (k,) (full width, measured along B = T x N). The front face normal is the rotation-minimising
    normal N, which starts as `up` projected perpendicular to the path (default +Z; for a vertical strip pass a
    horizontal up, e.g. (0, -1, 0) to face the viewer). UV: u across 0 -> 1 along +B (so the front reads
    un-mirrored with the path pointing up), v along by arc length.
    thickness = 0: an open strip with 2 vertices per path point (column 0 at -B, column 1 at +B).
    thickness > 0: a closed thin slab centred on the strip with 4 vertices per path point (front -B, front +B, back
    +B, back -B), end faces for open paths; back face UVs run the other way so both sides read un-mirrored, edge
    faces take the border column. closed_path joins the last point to the first (a loop, no end faces)."""
    P = np.asarray(path, dtype=float)
    k = len(P)
    _, N, B = rmf_frames(P, up, closed_path)
    hw = 0.5 * _per_ring(width, k, "width", 1)
    s = arclength(np.vstack([P, P[:1]]) if closed_path else P)
    v = s / s[-1] if s[-1] > 0.0 else None
    if thickness <= 0.0:
        rings = np.stack([P - hw * B, P + hw * B], axis=1)
        return loft(rings, closed=False, uv=uv, u=np.array([0.0, 1.0]), v=v, loop=closed_path)
    t2 = 0.5 * float(thickness)
    rings = np.stack([P + t2 * N - hw * B, P + t2 * N + hw * B, P - t2 * N + hw * B, P - t2 * N - hw * B], axis=1)
    cap = None if closed_path else "ngon"
    g = loft(rings, closed=True, cap_start=cap, cap_end=cap, uv=uv, v=v, loop=closed_path)
    if uv:
        nrq = k - 1 + (1 if closed_path else 0)
        vv = np.linspace(0.0, 1.0, nrq + 1) if v is None else np.asarray(v, dtype=float)
        ua = np.array([0.0, 1.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0, 0.0])
        rr, cc = np.meshgrid(np.arange(nrq), np.arange(4), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        side = np.stack([np.stack([ua[cc], vv[rr]], 1), np.stack([ub[cc], vv[rr]], 1),
                         np.stack([ub[cc], vv[rr + 1]], 1), np.stack([ua[cc], vv[rr + 1]], 1)], axis=1)
        g.uv[:len(rr) * 4] = side.reshape(-1, 2)
    return g


def revolve(profile, segments=16, angle=2.0 * np.pi, uv=True, flip=False):
    """Surface of revolution about the Z axis.

    Args: profile (m, 2) of (r, z) points listed bottom to top, r >= 0; with r > 0 and z increasing the normals point
    outward (use flip=True, or list the profile top to bottom, for the other side). Points with r ~ 0 collapse to one
    pole vertex and the adjacent band becomes a triangle fan. segments: angular subdivisions of the whole `angle`.
    angle: sweep in radians starting on +X and turning counter-clockwise; 2*pi (or more) closes the surface (wraps),
    anything less leaves it open (segments + 1 vertices per ring); a negative angle turns clockwise (faces are
    re-wound so normals still follow the rule above). uv: u = angle fraction (own seam corner for a full turn; for a
    clockwise sweep u = 1 - fraction so the texture is never mirrored seen from outside), v = arc-length fraction of
    the profile; pole triangles take the mean u of their base.
    Returns Geo; vertex layout: one block per profile point (see module docstring)."""
    prof = np.asarray(profile, dtype=float)
    if prof.ndim != 2 or prof.shape[1] != 2 or len(prof) < 2:
        raise ValueError("profile must be (m >= 2, 2) of (r, z)")
    segments = int(segments)
    full = abs(angle) >= 2.0 * np.pi - 1e-9
    if segments < (3 if full else 1):
        raise ValueError("revolve needs segments >= 3 (full turn) or >= 1 (partial)")
    r, z = prof[:, 0], prof[:, 1]
    rmax = float(np.abs(r).max())
    if rmax <= 0.0:
        raise ValueError("revolve: every profile point lies on the axis")
    span = (2.0 * np.pi if angle > 0 else -2.0 * np.pi) if full else float(angle)
    ncol = segments if full else segments + 1
    theta = span * np.arange(ncol) / segments
    ct, st = np.cos(theta), np.sin(theta)
    pole = np.abs(r) <= max(1e-9 * rmax, 1e-12)
    m = len(prof)
    vid = np.zeros(m, dtype=np.int64)
    blocks, count = [], 0
    for i in range(m):
        vid[i] = count
        if pole[i]:
            blocks.append(np.array([[0.0, 0.0, z[i]]]))
            count += 1
        else:
            blocks.append(np.stack([r[i] * ct, r[i] * st, np.full(ncol, z[i])], axis=1))
            count += ncol
    cidx = np.arange(segments)
    c1 = (cidx + 1) % ncol
    ucol = np.arange(segments + 1) / segments
    if span < 0:
        ucol = 1.0 - ucol
    ua, ub = ucol[:-1], ucol[1:]
    um = 0.5 * (ua + ub)
    vv = _arc_fraction(prof, False)
    faces, uvs = [], []
    for i in range(m - 1):
        lo, hi = bool(pole[i]), bool(pole[i + 1])
        v0, v1 = vv[i], vv[i + 1]
        if lo and hi:
            continue
        if not lo and not hi:
            faces.append(np.stack([vid[i] + cidx, vid[i] + c1, vid[i + 1] + c1, vid[i + 1] + cidx], axis=1))
            uvs.append(np.stack([np.stack([ua, np.full(segments, v0)], 1), np.stack([ub, np.full(segments, v0)], 1),
                                 np.stack([ub, np.full(segments, v1)], 1), np.stack([ua, np.full(segments, v1)], 1)],
                                axis=1).reshape(-1, 2))
        elif lo:
            faces.append(np.stack([np.full(segments, vid[i]), vid[i + 1] + c1, vid[i + 1] + cidx], axis=1))
            uvs.append(np.stack([np.stack([um, np.full(segments, v0)], 1), np.stack([ub, np.full(segments, v1)], 1),
                                 np.stack([ua, np.full(segments, v1)], 1)], axis=1).reshape(-1, 2))
        else:
            faces.append(np.stack([vid[i] + cidx, vid[i] + c1, np.full(segments, vid[i + 1])], axis=1))
            uvs.append(np.stack([np.stack([ua, np.full(segments, v0)], 1), np.stack([ub, np.full(segments, v0)], 1),
                                 np.stack([um, np.full(segments, v1)], 1)], axis=1).reshape(-1, 2))
    flist = [f for block in faces for f in block.tolist()]
    g = Geo(np.concatenate(blocks), flist, np.concatenate(uvs) if (uv and uvs) else None)
    return g.flipped() if (flip != (span < 0)) else g


def _vec3(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return np.full(3, float(a))
    if a.shape != (3,):
        raise ValueError(f"{name} must be a scalar or (3,), got shape {a.shape}")
    return a


def uv_sphere(radius=1.0, center=(0.0, 0.0, 0.0), segments=16, rings=8, scale=(1.0, 1.0, 1.0)):
    """Latitude/longitude sphere with poles on Z.

    Args: radius: scalar or (3,) semi-axes (ellipsoid). center: (3,). segments: meridians (>= 3). rings: latitude
    bands (>= 2). scale: extra per-axis factor on the radius (a negative entry mirrors; the faces stay outward).
    Returns: Geo with 2 + (rings - 1) * segments vertices (vertex 0 = bottom pole, last = top pole), 2 * segments
    triangles at the poles and quads elsewhere. UV: equirectangular, u = azimuth from +X (counter-clockwise, own seam
    corner), v = latitude fraction from the bottom pole (v = 0) to the top pole (v = 1)."""
    if rings < 2:
        raise ValueError("uv_sphere needs rings >= 2")
    phi = np.linspace(0.0, np.pi, int(rings) + 1)
    g = revolve(np.stack([np.sin(phi), -np.cos(phi)], axis=1), segments)
    return g.transformed(scale=_vec3(radius, "radius") * _vec3(scale, "scale"), t=center)


def _cube_surface(counts, sizes):
    """Welded surface lattice of a box split into counts = (cx, cy, cz) cells per axis.

    Returns (lat (nv, 3) int lattice coordinates in raster order, quads (nq, 4) vertex indices CCW from outside,
    uvq (nq, 4, 2) cross-layout UVs, fid (nq,) face id 0..5 = +X -X +Y -Y +Z -Z). The cross layout is the unfolded
    box (columns -X -Y +X +Y in the middle row, +Z above and -Z below the -Y face) with cells proportional to
    `sizes`, scaled uniformly into the unit square and centred."""
    cx, cy, cz = counts
    shape = (cx + 1, cy + 1, cz + 1)
    a, b, c = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    on = (a == 0) | (a == cx) | (b == 0) | (b == cy) | (c == 0) | (c == cz)
    idx = np.full(shape, -1, dtype=np.int64)
    idx[on] = np.arange(int(on.sum()))
    lat = np.stack([a[on], b[on], c[on]], axis=1)
    sx, sy, sz = sizes
    col_w, col_l = (sy, sx, sy, sx), (0.0, sy, sy + sx, 2 * sy + sx)
    row_h, row_b = (sy, sz, sy), (0.0, sy, sy + sz)
    W, H = 2.0 * (sx + sy), sz + 2.0 * sy
    sc = 1.0 / max(W, H)
    u0, v0 = 0.5 * (1.0 - W * sc), 0.5 * (1.0 - H * sc)
    #      normal axis, sign, grid axes (u, v) with u x v = outward, net s axis/sign, net t axis/sign, net col, row
    spec = ((0, 1, 1, 2, 1, 1, 2, 1, 2, 1), (0, -1, 2, 1, 1, -1, 2, 1, 0, 1),
            (1, 1, 2, 0, 0, -1, 2, 1, 3, 1), (1, -1, 0, 2, 0, 1, 2, 1, 1, 1),
            (2, 1, 0, 1, 0, 1, 1, 1, 1, 2), (2, -1, 1, 0, 0, 1, 1, -1, 1, 0))
    quads, uvq, fid = [], [], []
    for f, (na, sg, ua, va, sa, ss, ta, ts, col, row) in enumerate(spec):
        i, j = np.meshgrid(np.arange(counts[ua]), np.arange(counts[va]), indexing="ij")
        i, j = i.ravel(), j.ravel()
        co = np.zeros((len(i), 4, 3), dtype=np.int64)
        co[:, :, na] = counts[na] if sg > 0 else 0
        co[:, :, ua] = np.stack([i, i + 1, i + 1, i], axis=1)
        co[:, :, va] = np.stack([j, j, j + 1, j + 1], axis=1)
        quads.append(idx[co[..., 0], co[..., 1], co[..., 2]])
        s = co[:, :, sa] / counts[sa]
        t = co[:, :, ta] / counts[ta]
        s = s if ss > 0 else 1.0 - s
        t = t if ts > 0 else 1.0 - t
        uvq.append(np.stack([u0 + sc * (col_l[col] + s * col_w[col]), v0 + sc * (row_b[row] + t * row_h[row])],
                            axis=-1))
        fid.append(np.full(len(i), f))
    return lat, np.concatenate(quads), np.concatenate(uvq), np.concatenate(fid)


def _spherical_uv(D):
    """Equirectangular per-corner UVs of unit directions D (nq, 4, 3): u = azimuth from +X / 2 pi, v = 0.5 + latitude
    / pi. Each face is unwrapped around its own mean azimuth so faces straddling the seam get continuous (slightly <0
    or >1) u values; corners on a pole take the face's azimuth."""
    th = np.arctan2(D[..., 1], D[..., 0])
    valid = np.hypot(D[..., 0], D[..., 1]) > 1e-9
    ref = np.arctan2((np.sin(th) * valid).sum(axis=1), (np.cos(th) * valid).sum(axis=1))
    d = np.where(valid, (th - ref[:, None] + np.pi) % (2.0 * np.pi) - np.pi, 0.0)
    u = (ref[:, None] / (2.0 * np.pi)) % 1.0 + d / (2.0 * np.pi)
    v = 0.5 + np.arcsin(np.clip(D[..., 2], -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=-1)


def quad_sphere(radius=1.0, center=(0.0, 0.0, 0.0), n=4, scale=(1.0, 1.0, 1.0), uv="spherical"):
    """Cube-sphere: a cube with n x n quads per side pushed onto a sphere (near equal-area mapping).

    Args: radius: scalar or (3,). center: (3,). n: quads per cube edge (>= 1). scale: extra per-axis factor on the
    radius. uv: 'spherical' (equirectangular from the direction: u = azimuth from +X, v = latitude; every face is
    unwrapped around its own azimuth so no face smears across the seam; an even n puts a vertex exactly on each pole
    and makes the seam a cell edge), 'cross' (cube-net layout, see box) or None.
    Returns: Geo with 6 n^2 quads and 6 n^2 + 2 welded vertices (valence 3 only at the 8 cube corners): ideal for
    Catmull-Clark subdivision."""
    n = int(n)
    if n < 1:
        raise ValueError("quad_sphere needs n >= 1")
    if uv not in ("spherical", "cross", None):
        raise ValueError("uv must be 'spherical', 'cross' or None")
    lat, quads, uvq, _ = _cube_surface((n, n, n), (1.0, 1.0, 1.0))
    q = lat / n * 2.0 - 1.0
    x2, y2, z2 = q[:, 0] ** 2, q[:, 1] ** 2, q[:, 2] ** 2
    d = unit(np.stack([q[:, 0] * np.sqrt(1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0),
                       q[:, 1] * np.sqrt(1.0 - z2 / 2.0 - x2 / 2.0 + z2 * x2 / 3.0),
                       q[:, 2] * np.sqrt(1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0)], axis=1))
    verts = d * (_vec3(radius, "radius") * _vec3(scale, "scale")) + np.asarray(center, dtype=float)
    if uv == "spherical":
        coords = _spherical_uv(d[quads]).reshape(-1, 2)
    elif uv == "cross":
        coords = uvq.reshape(-1, 2)
    else:
        coords = None
    g = Geo(verts, quads.tolist(), coords)
    return g.flipped() if np.prod(_vec3(radius, "radius") * _vec3(scale, "scale")) < 0 else g


def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), segments=(1, 1, 1), uv=True):
    """Axis-aligned box.

    Args: size: full extents (x, y, z), all > 0. center: (3,). segments: cells per axis (int or 3 ints). uv: build UVs.
    Returns: Geo: closed, welded vertices (raster order, x slowest), all quads, faces ordered +X, -X, +Y, -Y, +Z, -Z.
    UV: the box unfolded as a cross (middle row -X -Y +X +Y, +Z above and -Z below the -Y face), cell sizes in true
    proportion, scaled uniformly into the unit square and centred (uniform texel density); every face reads
    un-mirrored seen from outside with +Z up (the top and bottom faces have +Y / -Y up)."""
    sz = _vec3(size, "size")
    if (sz <= 0).any():
        raise ValueError("box size must be > 0")
    seg = np.broadcast_to(np.asarray(segments, dtype=np.int64), (3,))
    if (seg < 1).any():
        raise ValueError("box segments must be >= 1")
    lat, quads, uvq, _ = _cube_surface(tuple(int(s) for s in seg), tuple(sz))
    verts = (lat / seg - 0.5) * sz + np.asarray(center, dtype=float)
    return Geo(verts, quads.tolist(), uvq.reshape(-1, 2) if uv else None)


def cylinder(radius=0.5, height=1.0, sides=16, rings=1, cap_top=True, cap_bottom=True, center=(0.0, 0.0, 0.0)):
    """Cylinder along Z from -height/2 to +height/2 about `center`.

    Args: radius: scalar or (rx, ry). height: full length. sides: points per ring. rings: bands along the height
    (rings + 1 vertex rings). cap_top / cap_bottom: True ('ngon'), False, 'fan' or 'ngon'. center: (3,).
    Returns: Geo with vertex (ring r from the bottom, side j) = r * sides + j (fan centres appended: bottom first).
    UV: u around, v along the height, caps planar (see loft)."""
    rad = np.broadcast_to(np.asarray(radius, dtype=float), (2,))
    ang = 2.0 * np.pi * np.arange(int(sides)) / int(sides)
    zs = np.linspace(-0.5, 0.5, int(rings) + 1) * float(height)
    ring = np.stack([rad[0] * np.cos(ang), rad[1] * np.sin(ang)], axis=1)
    R3 = np.empty((len(zs), len(ang), 3))
    R3[:, :, :2] = ring
    R3[:, :, 2] = zs[:, None]
    return loft(R3, True, cap_start=cap_bottom, cap_end=cap_top).translated(center)


def cone(radius=0.5, height=1.0, sides=16, rings=1, cap=True, center=(0.0, 0.0, 0.0)):
    """Cone along Z: base ring at -height/2, apex at +height/2 about `center`.

    Args: radius: base radius. height. sides. rings: bands up the slope. cap: add a base fan (centre vertex).
    center: (3,). Returns: Geo built with revolve (the apex and the cap centre are pole vertices); UV as in revolve."""
    t = np.linspace(0.0, 1.0, int(rings) + 1)
    prof = np.stack([radius * (1.0 - t), (t - 0.5) * height], axis=1)
    if cap:
        prof = np.vstack([[0.0, -0.5 * height], prof])
    return revolve(prof, sides).translated(center)


def capsule(radius=0.5, height=1.0, sides=16, rings=4, center=(0.0, 0.0, 0.0)):
    """Capsule along Z: a cylinder of straight length `height` closed by two hemispheres.

    Args: radius. height: the straight part between the hemisphere centres (at +-height/2 about `center`; 0 gives a
    sphere); the overall length is height + 2 * radius. sides. rings: latitude bands per hemisphere (>= 1).
    center: (3,). Returns: Geo with pole vertices on the axis. UV: u around, v by profile arc length (bottom 0 -> top
    1)."""
    if rings < 1:
        raise ValueError("capsule needs rings >= 1")
    r, h = float(radius), float(height)
    lo = np.linspace(-0.5 * np.pi, 0.0, int(rings) + 1)
    hi = np.linspace(0.0, 0.5 * np.pi, int(rings) + 1)
    low = np.stack([r * np.cos(lo), -0.5 * h + r * np.sin(lo)], axis=1)
    upp = np.stack([r * np.cos(hi), 0.5 * h + r * np.sin(hi)], axis=1)
    prof = np.vstack([low, upp]) if h > _TINY else np.vstack([low, upp[1:]])
    return revolve(prof, sides).translated(center)


def disc(radius=1.0, sides=16, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), rings=0):
    """Flat disc facing `normal`.

    Args: radius. sides: points per ring (>= 3). center: (3,). normal: (3,) facing direction (counter-clockwise seen
    from its tip; the first vertex lies along the x axis of frame_from(z=normal)). rings: 0 = one n-gon with `sides`
    vertices; >= 1 = a centre vertex (index 0) with a triangle fan and, for rings >= 2, quad bands (1 + rings * sides
    vertices).
    Returns: Geo. UV: planar, the disc fills the inscribed circle of the unit square (centre (0.5, 0.5), radius
    0.5)."""
    nrm = unit(np.asarray(normal, dtype=float))
    if not nrm.any():
        raise ValueError("disc: zero normal")
    F = frame_from(z=nrm)
    sides, rings = int(sides), int(rings)
    if sides < 3:
        raise ValueError("disc needs sides >= 3")
    ang = 2.0 * np.pi * np.arange(sides) / sides
    ring_dir = np.cos(ang)[:, None] * F[:, 0] + np.sin(ang)[:, None] * F[:, 1]
    cen = np.asarray(center, dtype=float)
    if rings <= 0:
        uvv = 0.5 + 0.5 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return Geo(cen + radius * ring_dir, [list(range(sides))], uvv)
    rad = radius * np.arange(1, rings + 1) / rings
    verts = np.concatenate([cen[None], (cen + rad[:, None, None] * ring_dir[None]).reshape(-1, 3)])
    rho = np.arange(1, rings + 1) / rings
    uvs = np.concatenate([[[0.5, 0.5]], (0.5 + 0.5 * rho[:, None, None] * np.stack([np.cos(ang), np.sin(ang)], 1)[None])
                          .reshape(-1, 2)])
    j = np.arange(sides)
    j1 = (j + 1) % sides
    faces = np.stack([np.zeros(sides, dtype=np.int64), 1 + j, 1 + j1], axis=1).tolist()
    corners = [np.stack([uvs[np.zeros(sides, dtype=np.int64)], uvs[1 + j], uvs[1 + j1]], axis=1).reshape(-1, 2)]
    for k in range(rings - 1):
        a0, a1 = 1 + k * sides, 1 + (k + 1) * sides
        q = np.stack([a0 + j, a1 + j, a1 + j1, a0 + j1], axis=1)
        faces.extend(q.tolist())
        corners.append(uvs[q].reshape(-1, 2))
    return Geo(verts, faces, np.concatenate(corners))


def plane(size=(1.0, 1.0), segments=(1, 1), center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)):
    """Flat grid facing `normal`.

    Args: size: (extent along local x, along local y). segments: cells per direction (int or 2 ints). center: (3,).
    normal: (3,). up: in-plane direction of local y (projected perpendicular to the normal; local x = y x normal);
    the defaults give the XY plane facing +Z.
    Returns: Geo with vertex (row, col) = row * (nx + 1) + col, rows running along local y. UV: u along local x, v
    along local y, 0..1."""
    nrm = unit(np.asarray(normal, dtype=float))
    if not nrm.any():
        raise ValueError("plane: zero normal")
    sx, sy = np.broadcast_to(np.asarray(size, dtype=float), (2,))
    nx, ny = (int(s) for s in np.broadcast_to(np.asarray(segments), (2,)))
    F = frame_from(z=nrm, y=up)
    X, Y = np.meshgrid((np.linspace(0.0, 1.0, nx + 1) - 0.5) * sx, (np.linspace(0.0, 1.0, ny + 1) - 0.5) * sy)
    P = np.asarray(center, dtype=float) + X[..., None] * F[:, 0] + Y[..., None] * F[:, 1]
    return loft(P, closed=False)


def surface(fn, nu, nv, closed_u=False, closed_v=False, uv=True):
    """Parametric patch P(u, v) tessellated with nu x nv quads.

    Args: fn: callable fn(u, v) -> (..., 3), called ONCE with two meshgrid arrays u, v in [0, 1] (u varies along the
    last axis) of shape (nv + 1, nu + 1), or with one fewer row / column where closed_v / closed_u wrap (the end value
    1 is then not sampled); it must return an array of that shape + (3,). nu, nv: quads along u / v. closed_u,
    closed_v: wrap that direction (cylinder / torus). uv: build UVs.
    Returns: Geo whose quad normals follow dP/du x dP/dv; UV = (u, v) with their own closing corner values on seams;
    vertex (row, col) = row * cols + col."""
    nu, nv = int(nu), int(nv)
    if nu < 1 or nv < 1:
        raise ValueError("surface needs nu >= 1 and nv >= 1")
    us = np.arange(nu if closed_u else nu + 1) / nu
    vs = np.arange(nv if closed_v else nv + 1) / nv
    U, V = np.meshgrid(us, vs)
    P = np.asarray(fn(U, V), dtype=float)
    if P.shape != U.shape + (3,):
        raise ValueError(f"fn must return shape {U.shape + (3,)}, got {P.shape}")
    return loft(P, closed=closed_u, loop=closed_v, uv=uv)
