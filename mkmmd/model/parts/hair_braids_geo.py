"""Queries and small geometry helpers for the braids (bpy-free, numpy only).

`TriMesh` answers exact point-to-surface distances against a triangle soup (the body mesh, the head skin); `body_clearance`
measures capsules and points against the static `col_*` rigid bodies of the body part; the rest are rotation / shell
helpers shared by the plait and bow builders."""
import numpy as np

from .hair_geo import unit


# ---------------------------------------------------------------- triangle soup
def _tri_closest(p, a, b, c):
    """Closest point to p (n,1,3) on triangles a, b, c (1,m,3) -> (n,m,3) (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    q = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]               # inside the face
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)                           # edge BC
    t = (d4 - d3) / np.where(m, (d4 - d3) + (d5 - d6), 1.0)
    q = np.where(m[..., None], b + (c - b) * t[..., None], q)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                                         # edge AC
    t = d2 / np.where(m, d2 - d6, 1.0)
    q = np.where(m[..., None], a + ac * t[..., None], q)
    m = (d6 >= 0) & (d5 <= d6)                                                    # vertex C
    q = np.where(m[..., None], c, q)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                                         # edge AB
    t = d1 / np.where(m, d1 - d3, 1.0)
    q = np.where(m[..., None], a + ab * t[..., None], q)
    m = (d3 >= 0) & (d4 <= d3)                                                    # vertex B
    q = np.where(m[..., None], b, q)
    m = (d1 <= 0) & (d2 <= 0)                                                     # vertex A
    return np.where(m[..., None], a, q)


class TriMesh:
    """Triangle soup with exact nearest-point queries (bounding-box prefiltered, chunked). A closed shell with outward
    (CCW) faces also answers signed distances (negative inside)."""

    def __init__(self, verts, faces):
        V = np.asarray(verts, float)
        tris = np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)], int)
        self.A, self.B, self.C = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
        self.lo = np.minimum(np.minimum(self.A, self.B), self.C)
        self.hi = np.maximum(np.maximum(self.A, self.B), self.C)
        self.n = unit(np.cross(self.B - self.A, self.C - self.A))

    def _nearest(self, P, sel):
        a, b, c = self.A[sel][None], self.B[sel][None], self.C[sel][None]
        out_d = np.full(len(P), np.inf)
        out_q = np.zeros((len(P), 3))
        out_k = np.zeros(len(P), int)
        for i in range(0, len(P), 48):
            p = P[i:i + 48, None, :]
            q = _tri_closest(p, a, b, c)
            d = np.linalg.norm(q - p, axis=-1)
            k = d.argmin(1)
            r = np.arange(len(k))
            out_d[i:i + 48] = d[r, k]
            out_q[i:i + 48] = q[r, k]
            out_k[i:i + 48] = sel[k]
        return out_d, out_q, out_k

    def nearest(self, P, reach=0.06):
        """(distance (n,), nearest surface point (n,3), triangle index (n,)) for points P (n,3). Triangles farther than
        `reach` from the bounding box of P are skipped; points whose nearest hit lies beyond `reach` are re-queried
        against everything."""
        P = np.atleast_2d(np.asarray(P, float))
        sel = np.flatnonzero(((self.hi >= P.min(0) - reach) & (self.lo <= P.max(0) + reach)).all(1))
        if not len(sel):
            sel = np.arange(len(self.A))
        d, q, k = self._nearest(P, sel)
        far = d > reach
        if far.any() and len(sel) < len(self.A):
            d2, q2, k2 = self._nearest(P[far], np.arange(len(self.A)))
            d[far], q[far], k[far] = d2, q2, k2
        return d, q, k

    def signed(self, P, reach=0.06):
        """(signed distance, nearest point, outward normal of the nearest face): negative inside the shell."""
        P = np.atleast_2d(np.asarray(P, float))
        d, q, k = self.nearest(P, reach)
        n = self.n[k]
        s = np.where(((P - q) * n).sum(1) < 0.0, -1.0, 1.0)
        return d * s, q, n


# ---------------------------------------------------------------- static colliders
def euler_matrix(rot):
    """Rotation matrix of an XYZ Euler triple in the Blender convention R = Rz Ry Rx (what RigidBody.rotation holds)."""
    x, y, z = (float(v) for v in rot)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def collider_distance(P, rb):
    """Signed distance (m, > 0 outside) of points P (n,3) to a static rigid body (sphere, capsule or box)."""
    P = np.atleast_2d(np.asarray(P, float))
    c = np.asarray(rb.location, float)
    R = euler_matrix(rb.rotation)
    if rb.shape == "sphere":
        return np.linalg.norm(P - c, axis=1) - rb.size[0]
    if rb.shape == "capsule":
        ax = R @ np.array([0.0, 0.0, 1.0])
        h = 0.5 * float(rb.size[1])
        return _seg_dist(P, c - ax * h, c + ax * h) - rb.size[0]
    if rb.shape == "box":
        q = np.abs((P - c) @ R) - np.asarray(rb.size, float)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)
    raise ValueError(f"unknown rigid body shape {rb.shape!r}")


def colliders(body_part):
    """The static `col_*` bodies of the body part."""
    return [rb for rb in body_part.bodies if rb.name.startswith("col_") and rb.mode == "static"]


def collider_clearance(P, bodies, radius=0.0):
    """Smallest gap (m) between spheres of `radius` centred on P and every collider: min over bodies of the signed
    distance minus `radius` (negative = overlapping), per point (n,)."""
    P = np.atleast_2d(np.asarray(P, float))
    if not bodies:
        return np.full(len(P), np.inf)
    return np.min([collider_distance(P, rb) for rb in bodies], axis=0) - radius


# ---------------------------------------------------------------- rotations and shells
def rot_axis(axis, ang):
    """Rotation matrix about the unit `axis` by `ang` radians (Rodrigues)."""
    k = unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def signed_volume(verts, faces):
    """Signed volume of a closed polygon mesh (positive when the faces are CCW seen from outside)."""
    v = np.asarray(verts, float)
    vol = 0.0
    for f in faces:
        p0 = v[f[0]]
        for k in range(1, len(f) - 1):
            vol += np.dot(p0, np.cross(v[f[k]], v[f[k + 1]]))
    return vol / 6.0


def orient_outward(verts, faces):
    """Faces flipped when needed so that a closed shell has outward (CCW) winding."""
    if signed_volume(verts, faces) < 0:
        return [list(f[::-1]) for f in faces]
    return [list(f) for f in faces]

