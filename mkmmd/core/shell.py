"""Shell toolkit: pure-numpy geometry for props (bpy-free: importable from tests, and fast: every part is built from whole
arrays, never vertex by vertex). The Blender side is `mkmmd.blender.library.shell`; the rules and a worked example are in
docs/modelling.md.

A `Mesh` is vertices V (n, 3), quads Q (m, 4), triangles T (k, 3), one material index per face, optional per-vertex uv and
edge creases (Ce (e, 2) vertex pairs, Cw (e,) weights 0..1: what a Subdivision Surface keeps sharp). Winding is
counter-clockwise seen from outside (normals outward) for every primitive. Smooth shading is meant to use
`Mesh.vertex_normals()` (area weighted, so a big flat panel stays flat next to a small bevel).

Forms (what to model with):
    loft / Station / pt          a body from a few cross-sections along an axis: sections are control polygons of named
                                 points, interpolated smoothly between stations, mirrored, with column / ring creases,
                                 end caps and open-top regions (see `loft`). Meant to be subdivided (a cage).
    sweep / tube / ribbon        a profile along a path (rubber strips, trims, piping, handles)
    lathe / skin                 surfaces of revolution, matching sections skinned together
    rounded_box                  small parts with real fillets (never a hard-edged cuboid)
    extrude_profile              a side profile pulled across x with rounded rims
    arch_cutter                  the solid a wheel arch is cut with (a flared cylinder), for a Boolean in Blender
2D tools:  fillet (rounded polyline corners), arc, rrect (rounded rectangle section), ellipse.
"""
import math
from collections import namedtuple

import numpy as np

EPS = 1e-9

# An animated or separately posed piece: its mesh in its own frame, the frame's origin in the car frame, and the
# base rotation of the frame (Blender XYZ euler, degrees).
Part = namedtuple("Part", "mesh origin rot")

# A cage the builder turns into a smooth object: a Subdivision Surface of `levels` over the creased `mesh`, Boolean
# DIFFERENCE by every mesh of `cutters` (their cut faces take the material role `cutter_role`), then a Bevel
# `bevel = (width m, segments)` on the edges the cuts leave hard.
Shell = namedtuple("Shell", "mesh levels bevel cutters cutter_role")


def smooth(mesh, levels=2, bevel=None, cutters=(), cutter_role=None):
    """A `Shell`: what a part module returns from `static_shells()` for a cage that is to be subdivided."""
    return Shell(mesh, int(levels), bevel, tuple(cutters), cutter_role)


# ===================================================================================================================
# mesh container
# ===================================================================================================================
class Mesh:
    """Vertices, quads, triangles, a material index per face, optional uv per vertex, optional edge creases."""
    __slots__ = ("V", "Q", "T", "Qm", "Tm", "UV", "Ce", "Cw")

    def __init__(self, V=None, Q=None, T=None, Qm=None, Tm=None, UV=None, mat=0, Ce=None, Cw=None):
        self.V = np.zeros((0, 3)) if V is None else np.asarray(V, float).reshape(-1, 3)
        self.Q = np.zeros((0, 4), np.int64) if Q is None else np.asarray(Q, np.int64).reshape(-1, 4)
        self.T = np.zeros((0, 3), np.int64) if T is None else np.asarray(T, np.int64).reshape(-1, 3)
        self.Qm = np.full(len(self.Q), mat, np.int32) if Qm is None else np.asarray(Qm, np.int32).reshape(-1)
        self.Tm = np.full(len(self.T), mat, np.int32) if Tm is None else np.asarray(Tm, np.int32).reshape(-1)
        self.UV = None if UV is None else np.asarray(UV, float).reshape(-1, 2)
        self.Ce = np.zeros((0, 2), np.int64) if Ce is None else np.asarray(Ce, np.int64).reshape(-1, 2)
        self.Cw = np.zeros(0) if Cw is None else np.asarray(Cw, float).reshape(-1)

    # ---- bookkeeping
    def copy(self):
        return Mesh(self.V.copy(), self.Q.copy(), self.T.copy(), self.Qm.copy(), self.Tm.copy(),
                    None if self.UV is None else self.UV.copy(), Ce=self.Ce.copy(), Cw=self.Cw.copy())

    @property
    def nfaces(self):
        return len(self.Q) + len(self.T)

    def empty(self):
        return self.nfaces == 0

    def bbox(self):
        return self.V.min(0), self.V.max(0)

    def size(self):
        lo, hi = self.bbox()
        return hi - lo

    # ---- transforms (all return new meshes)
    def apply(self, R=None, t=(0.0, 0.0, 0.0)):
        """x' = R x + t. A reflecting R (det < 0) flips the winding back to outward."""
        V = self.V if R is None else self.V @ np.asarray(R, float).T
        m = Mesh(V + np.asarray(t, float), self.Q.copy(), self.T.copy(), self.Qm.copy(), self.Tm.copy(),
                 None if self.UV is None else self.UV.copy(), Ce=self.Ce.copy(), Cw=self.Cw.copy())
        if R is not None and np.linalg.det(np.asarray(R, float)) < 0:
            m = m.flip()
        return m

    def triangles(self):
        """All faces as triangles (k, 3): the triangles, then every quad split along its first diagonal."""
        q = self.Q
        return np.concatenate([self.T, q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) if len(q) else self.T.copy()

    def moved(self, t):
        return self.apply(None, t)

    def rot(self, rx=0.0, ry=0.0, rz=0.0, about=(0.0, 0.0, 0.0)):
        """Rotate by Euler X then Y then Z (degrees, intrinsic order of Blender's XYZ euler) about a point."""
        R = rot_matrix(rx, ry, rz)
        c = np.asarray(about, float)
        return self.apply(R, c - R @ c)

    def mirrored_x(self):
        return self.apply(np.diag([-1.0, 1.0, 1.0]))

    def mirrored_y(self):
        return self.apply(np.diag([1.0, -1.0, 1.0]))

    def flip(self):
        return Mesh(self.V.copy(), self.Q[:, ::-1].copy(), self.T[:, ::-1].copy(), self.Qm.copy(), self.Tm.copy(),
                    None if self.UV is None else self.UV.copy(), Ce=self.Ce.copy(), Cw=self.Cw.copy())

    # ---- creases
    def crease(self, pairs, w=1.0):
        """Mark edges (an (e, 2) array / list of vertex index pairs) as creases of weight w (0..1; the largest wins)."""
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            return self
        w = np.broadcast_to(np.asarray(w, float), (len(pairs),))
        e = np.sort(np.concatenate([self.Ce, pairs]), axis=1)
        ww = np.concatenate([self.Cw, w])
        key = e[:, 0] * max(len(self.V), 1) + e[:, 1]
        order = np.lexsort((ww, key))                         # per key, ascending weight: the last one is the largest
        key, e, ww = key[order], e[order], ww[order]
        last = np.concatenate([key[1:] != key[:-1], [True]])
        self.Ce, self.Cw = e[last], np.clip(ww[last], 0.0, 1.0)
        return self

    def crease_dict(self):
        return {(int(a), int(b)): float(w) for (a, b), w in zip(self.Ce, self.Cw)}

    def crease_where(self, fn, w=1.0):
        """Crease every edge (of the quads and triangles) whose two end points satisfy fn(P_a, P_b) -> bool array."""
        parts = [np.stack([f, np.roll(f, -1, axis=1)], -1).reshape(-1, 2) for f in (self.Q, self.T) if len(f)]
        e = np.unique(np.sort(np.concatenate(parts), axis=1), axis=0)
        return self.crease(e[fn(self.V[e[:, 0]], self.V[e[:, 1]])], w)

    # ---- topology clean-up
    def weld(self, tol=1e-7):
        """Merge coincident vertices; a quad that collapses to a triangle becomes one, a collapsed face is dropped."""
        if len(self.V) == 0:
            return self.copy()
        key = np.round(self.V / tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        order = np.argsort(first)                          # keep the first-seen order of vertices
        rank = np.empty(len(first), np.int64)
        rank[order] = np.arange(len(first))
        remap = rank[inv]
        V = self.V[first[order]]
        UV = None if self.UV is None else self.UV[first[order]]
        Q, Qm = remap[self.Q], self.Qm
        T, Tm = remap[self.T], self.Tm
        a, b, c, d = Q.T
        ab, bc, cd, da = a == b, b == c, c == d, d == a
        nd = ab.astype(int) + bc + cd + da
        keep4 = (nd == 0) & (a != c) & (b != d)
        tri = nd == 1
        t_from_q = np.zeros((int(tri.sum()), 3), np.int64)
        qa, qb, qc, qd = a[tri], b[tri], c[tri], d[tri]
        which = np.stack([ab[tri], bc[tri], cd[tri], da[tri]], 1).argmax(1)
        for w, rows in enumerate((np.stack([qb, qc, qd], 1), np.stack([qa, qc, qd], 1), np.stack([qa, qb, qd], 1),
                                  np.stack([qa, qb, qc], 1))):
            t_from_q[which == w] = rows[which == w]
        T2 = np.concatenate([T, t_from_q])
        Tm2 = np.concatenate([Tm, Qm[tri]])
        ok = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
        out = Mesh(V, Q[keep4], T2[ok], Qm[keep4], Tm2[ok], UV)
        if len(self.Ce):
            ce = remap[self.Ce]
            keep = ce[:, 0] != ce[:, 1]
            out.crease(ce[keep], self.Cw[keep])
        return out

    # ---- normals and measures
    def face_vectors(self):
        """Area vectors (|v| = area, direction = outward normal) of the quads and of the triangles."""
        Vq, Vt = self.V[self.Q], self.V[self.T]
        nq = 0.5 * np.cross(Vq[:, 2] - Vq[:, 0], Vq[:, 3] - Vq[:, 1]) if len(self.Q) else np.zeros((0, 3))
        nt = 0.5 * np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0]) if len(self.T) else np.zeros((0, 3))
        return nq, nt

    def vertex_normals(self):
        """Area-weighted vertex normals (unit), for custom split normals: smooth where faces share vertices."""
        nq, nt = self.face_vectors()
        n = len(self.V)
        out = np.zeros((n, 3))
        for idx, vec in ((self.Q, nq), (self.T, nt)):
            if len(idx) == 0:
                continue
            for c in range(3):
                w = np.repeat(vec[:, c], idx.shape[1])
                out[:, c] += np.bincount(idx.ravel(), weights=w, minlength=n)
        ln = np.linalg.norm(out, axis=1)
        out[ln > EPS] /= ln[ln > EPS, None]
        out[ln <= EPS] = (0.0, 0.0, 1.0)
        return out

    def volume(self):
        """Signed volume (positive when the faces point outward): the divergence theorem over a triangulation."""
        v = 0.0
        if len(self.Q):
            P = self.V[self.Q]
            v += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum()
            v += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 2], P[:, 3])).sum()
        if len(self.T):
            P = self.V[self.T]
            v += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum()
        return v / 6.0

    def area(self):
        nq, nt = self.face_vectors()
        return float(np.linalg.norm(nq, axis=1).sum() + np.linalg.norm(nt, axis=1).sum())

    def directed_edges(self):
        """(E, 2) array of every directed edge a->b of every face (consistent winding: each appears once per direction
        on a closed surface)."""
        parts = []
        for idx in (self.Q, self.T):
            if len(idx):
                parts.append(np.stack([idx, np.roll(idx, -1, axis=1)], -1).reshape(-1, 2))
        return np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)

    def is_closed(self):
        """True when every edge is shared by exactly two faces and traversed once in each direction (a closed,
        consistently wound surface)."""
        e = self.directed_edges()
        if len(e) == 0:
            return False
        n = len(self.V)
        fwd = e[:, 0] * n + e[:, 1]
        rev = e[:, 1] * n + e[:, 0]
        if len(np.unique(fwd)) != len(fwd):
            return False
        return bool(np.array_equal(np.sort(fwd), np.sort(rev)))

    def face_centres(self):
        cq = self.V[self.Q].mean(1) if len(self.Q) else np.zeros((0, 3))
        ct = self.V[self.T].mean(1) if len(self.T) else np.zeros((0, 3))
        return cq, ct

    def assign(self, mat, fn):
        """Set the material of the faces for which fn(centres (k, 3), unit normals (k, 3)) is True (in place)."""
        cq, ct = self.face_centres()
        nq, nt = self.face_vectors()
        for c, nv, m in ((cq, nq, self.Qm), (ct, nt, self.Tm)):
            if len(c) == 0:
                continue
            ln = np.linalg.norm(nv, axis=1)
            nrm = nv / np.where(ln > EPS, ln, 1.0)[:, None]
            m[fn(c, nrm)] = mat
        return self


def merge(meshes):
    """One mesh from many (material indices, uv and creases are kept; meshes without uv get zeros when others have it)."""
    meshes = [m for m in meshes if m is not None and not m.empty()]
    if not meshes:
        return Mesh()
    off, Vs, Qs, Ts, Qm, Tm, UVs, Ces, Cws = 0, [], [], [], [], [], [], [], []
    any_uv = any(m.UV is not None for m in meshes)
    for m in meshes:
        Vs.append(m.V)
        Qs.append(m.Q + off)
        Ts.append(m.T + off)
        Qm.append(m.Qm)
        Tm.append(m.Tm)
        Ces.append(m.Ce + off)
        Cws.append(m.Cw)
        if any_uv:
            UVs.append(m.UV if m.UV is not None else np.zeros((len(m.V), 2)))
        off += len(m.V)
    return Mesh(np.concatenate(Vs), np.concatenate(Qs), np.concatenate(Ts), np.concatenate(Qm), np.concatenate(Tm),
                np.concatenate(UVs) if any_uv else None, Ce=np.concatenate(Ces), Cw=np.concatenate(Cws))


def rot_matrix(rx=0.0, ry=0.0, rz=0.0):
    """3x3 rotation for Blender's XYZ euler (degrees): R = Rz Ry Rx."""
    a, b, c = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle(axis, deg):
    """3x3 rotation about a unit axis (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def place(mesh, origin, normal, up=(0.0, 0.0, 1.0)):
    """Put a mesh modelled in its own frame (x right, y up, z out of the surface it sits on) onto a surface: its z along
    `normal`, its y toward `up` (made perpendicular), its origin at `origin`."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    u = np.asarray(up, float)
    u = u - n * (u @ n)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = u - n * (u @ n)
    u = u / np.linalg.norm(u)
    return mesh.apply(np.stack([np.cross(u, n), u, n], axis=1), origin)


def rotation_between(a, b):
    """Shortest 3x3 rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-12:                                   # opposite: any perpendicular axis
        p = np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0))
        return axis_angle(p, 180.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


# ===================================================================================================================
# 2D tools (points are (n, 2) arrays)
# ===================================================================================================================
def arc(c, R, a0, a1, n):
    """n+1 points on a circle (centre c, radius R) from angle a0 to a1 (radians), both ends included."""
    t = np.linspace(a0, a1, n + 1)
    return np.stack([c[0] + R * np.cos(t), c[1] + R * np.sin(t)], 1)


def fillet(pts, radius, n=4, closed=False):
    """Round the corners of a polyline: `radius` is one value or one per point (0 keeps a corner sharp). Every rounded
    corner becomes an arc of n segments (n + 1 points). The tangent length is limited to half of each neighbouring edge,
    so close corners shrink instead of crossing. Returns the new (m, 2) polyline."""
    P = np.asarray(pts, float)
    N = len(P)
    rad = np.broadcast_to(np.asarray(radius, float), (N,)).copy()
    if not closed:
        rad[0] = rad[-1] = 0.0
    out = []
    for i in range(N):
        p = P[i]
        r = rad[i]
        if r <= 0:
            out.append(p[None])
            continue
        p0, p1 = P[(i - 1) % N], P[(i + 1) % N]
        d0, d1 = p - p0, p1 - p
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < EPS or l1 < EPS:
            out.append(p[None])
            continue
        d0, d1 = d0 / l0, d1 / l1
        cross = d0[0] * d1[1] - d0[1] * d1[0]
        phi = math.atan2(abs(cross), float(d0 @ d1))            # deflection angle
        if phi < 1e-6:
            out.append(p[None])
            continue
        # the arc may use at most half of each adjoining edge; a neighbour that is filleted too keeps its half
        t = min(r * math.tan(phi / 2), 0.5 * l0 if (closed or i - 1 > 0) else l0, 0.5 * l1 if (closed or i + 1 < N - 1) else l1)
        r_eff = t / math.tan(phi / 2)
        s = 1.0 if cross > 0 else -1.0                           # left turn: the centre is on the left
        start, end = p - d0 * t, p + d1 * t
        normal = np.array([-d0[1], d0[0]]) * s
        centre = start + normal * r_eff
        a_s = math.atan2(start[1] - centre[1], start[0] - centre[0])
        sweep = phi * s
        ang = a_s + sweep * np.arange(n + 1) / n
        out.append(np.stack([centre[0] + r_eff * np.cos(ang), centre[1] + r_eff * np.sin(ang)], 1))
    return np.concatenate(out)


def rrect(w, h, r, n=3):
    """Rounded rectangle section (CCW), w x h centred, corner radius r with n segments per corner; (m, 2)."""
    r = min(r, w / 2 - 1e-9, h / 2 - 1e-9)
    hw, hh = w / 2, h / 2
    pts = []
    for (cx, cy, a0) in ((hw - r, hh - r, 0.0), (-hw + r, hh - r, math.pi / 2), (-hw + r, -hh + r, math.pi),
                         (hw - r, -hh + r, 1.5 * math.pi)):
        pts.append(arc((cx, cy), r, a0, a0 + math.pi / 2, n))
    return np.concatenate(pts)


def ellipse(a, b, n=24):
    t = 2 * math.pi * np.arange(n) / n
    return np.stack([a * np.cos(t), b * np.sin(t)], 1)


def polygon_area(P):
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mitre_offsets(P):
    """For a closed CCW polygon: per vertex the vector that moves it one unit to the inside along both edges (the
    mitre), (n, 2). Collinear vertices get the edge normal."""
    P = np.asarray(P, float)
    e = np.roll(P, -1, axis=0) - P
    ln = np.linalg.norm(e, axis=1)
    e = e / np.maximum(ln, EPS)[:, None]
    nrm = np.stack([-e[:, 1], e[:, 0]], 1)                       # left of the travel direction = inside for CCW
    n_prev = np.roll(nrm, 1, axis=0)
    den = 1.0 + np.einsum("ij,ij->i", n_prev, nrm)
    return (n_prev + nrm) / np.maximum(den, 1e-6)[:, None]


# ===================================================================================================================
# primitives
# ===================================================================================================================
def _axis_coords(h, r, k, div):
    """Grid lines of one axis of a rounded box with half extent h: the corner strips follow tan so that the projected
    corner arc has equal angles, the flat middle is split every `div` metres (0 = not at all)."""
    u = r * np.tan(np.arange(k + 1) * (math.pi / 4) / k)
    pos = h - r + u
    inner = np.zeros(0)
    if div and h - r > 1e-9:
        cnt = max(1, int(math.ceil(2 * (h - r) / div)))
        inner = np.linspace(-(h - r), h - r, cnt + 1)[1:-1]
    return np.unique(np.round(np.concatenate([-pos, inner, pos]), 12))


def rounded_box(size, center=(0.0, 0.0, 0.0), r=0.01, k=1, div=0.0, mat=0, faces="xXyYzZ"):
    """Box with every edge rounded (radius r, 2k segments per corner arc). `div` splits the flat faces into cells of
    about that size (for later displacement). faces: which sides to build (x = -X, X = +X, y/Y, z/Z), so a box can sit
    against a wall without its back."""
    h = np.asarray(size, float) / 2
    r = float(min(r, h.min() * 0.999))
    coords = [_axis_coords(h[a], r, k, div) for a in range(3)]
    lo, hi = -(h - r), (h - r)
    Vs, Qs, off = [], [], 0
    for a in range(3):
        for sign, ch in ((-1, "xyz"[a]), (1, "XYZ"[a])):
            if ch not in faces:
                continue
            b, c = (a + 1) % 3, (a + 2) % 3
            U, W = np.meshgrid(coords[b], coords[c], indexing="ij")
            q = np.zeros(U.shape + (3,))
            q[..., a] = sign * h[a]
            q[..., b], q[..., c] = U, W
            ctr = np.clip(q, lo, hi)
            d = q - ctr
            ln = np.linalg.norm(d, axis=-1, keepdims=True)
            p = ctr + r * d / np.maximum(ln, EPS)
            nu, nw = U.shape
            idx = np.arange(nu * nw).reshape(nu, nw) + off
            quad = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
            if sign < 0:
                quad = quad[:, ::-1]
            Vs.append(p.reshape(-1, 3))
            Qs.append(quad)
            off += nu * nw
    m = Mesh(np.concatenate(Vs) + np.asarray(center, float), np.concatenate(Qs), mat=mat)
    return m.weld(1e-9)


def grid(origin, u, v, nu=1, nv=1, uv=True, mat=0):
    """Flat patch from `origin` spanned by vectors u and v (normal = u x v), nu x nv cells. uv in 0..1."""
    origin, u, v = (np.asarray(a, float) for a in (origin, u, v))
    s = np.linspace(0, 1, nu + 1)
    t = np.linspace(0, 1, nv + 1)
    S, Tt = np.meshgrid(s, t, indexing="ij")
    V = origin + S[..., None] * u + Tt[..., None] * v
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    Q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    UV = np.stack([S, Tt], -1).reshape(-1, 2) if uv else None
    return Mesh(V.reshape(-1, 3), Q, UV=UV, mat=mat)


def quad_in_plane(center, normal, up, size, mat=0, uv=True):
    """One flat rectangle (w, h) centred at `center` facing `normal`, its top edge toward `up`; uv 0..1 with (0, 0) at
    the bottom left as seen from the front."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up = np.asarray(up, float)
    up = up - n * (up @ n)
    up /= np.linalg.norm(up)
    right = np.cross(up, n)                                      # a viewer in front sees this to the right
    w, h = size
    o = np.asarray(center, float) - right * w / 2 - up * h / 2
    return grid(o, right * w, up * h, 1, 1, uv=uv, mat=mat)


def lathe(profile, seg=32, mat=0, closed_ends=False, phase=0.0):
    """Surface of revolution about +Z. profile: (m, 2) points (r, z), walked so the outside is on the right
    (bottom -> top along an outer wall gives outward normals). A point with r = 0 becomes one apex vertex; two identical
    consecutive points make a crease (the rings are not shared across them). `closed_ends` adds flat discs when the
    first/last point is off the axis."""
    P = np.asarray(profile, float)
    ang = phase + 2 * math.pi * np.arange(seg) / seg
    ca, sa = np.cos(ang), np.sin(ang)
    rings, Vs, off = [], [], 0
    pts = list(map(tuple, P))
    if closed_ends:
        if pts[0][0] > EPS:
            pts.insert(0, (0.0, pts[0][1]))
        if pts[-1][0] > EPS:
            pts.append((0.0, pts[-1][1]))
    for (r, z) in pts:
        if r < EPS:
            rings.append(np.array([off]))
            Vs.append(np.array([[0.0, 0.0, z]]))
            off += 1
        else:
            rings.append(np.arange(seg) + off)
            Vs.append(np.stack([r * ca, r * sa, np.full(seg, z)], 1))
            off += seg
    Qs, Ts = [], []
    j = np.arange(seg)
    j2 = (j + 1) % seg
    for i in range(len(pts) - 1):
        A, B = rings[i], rings[i + 1]
        if pts[i] == pts[i + 1]:
            continue
        if len(A) == 1 and len(B) == 1:
            continue
        if len(A) == 1:
            Ts.append(np.stack([np.full(seg, A[0]), B[j2], B[j]], 1))
        elif len(B) == 1:
            Ts.append(np.stack([A[j], A[j2], np.full(seg, B[0])], 1))
        else:
            Qs.append(np.stack([A[j], A[j2], B[j2], B[j]], 1))
    return Mesh(np.concatenate(Vs), np.concatenate(Qs) if Qs else None, np.concatenate(Ts) if Ts else None, mat=mat)


def _frames(path, up, closed=False):
    """Section frames along a path (n, 3): tangent T, u = B x T and v = B with B the `up` direction made perpendicular
    to T (rotation minimising when `up` is parallel to the path)."""
    n = len(path)
    if closed:
        T = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    else:
        T = np.empty_like(path)
        T[1:-1] = path[2:] - path[:-2]
        T[0] = path[1] - path[0]
        T[-1] = path[-1] - path[-2]
    T = T / np.maximum(np.linalg.norm(T, axis=1), EPS)[:, None]
    up = np.asarray(up, float)
    B = np.empty_like(T)
    last = up
    if np.linalg.norm(up - T[0] * (up @ T[0])) < 1e-6:           # the path starts along `up`: any perpendicular will do
        last = np.array([1.0, 0.0, 0.0]) if abs(T[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    for i in range(n):
        b = up - T[i] * (up @ T[i])
        if np.linalg.norm(b) < 1e-6:                              # parallel to `up` here: carry the previous frame on
            b = last - T[i] * (last @ T[i])
        b = b / max(np.linalg.norm(b), EPS)
        B[i] = last = b
    U = np.cross(B, T)
    return T, U, B


def skin(sections, caps=(True, True), closed=True, mat=0, uv=False):
    """Skin matching sections S (ns, m, 3). Each section must run counter-clockwise seen from the end it faces
    (looking back along the loft), i.e. with the loft direction as its right-hand normal. closed: sections are rings.
    Caps are fans around the section centroid (star-shaped sections only)."""
    S = np.asarray(sections, float)
    ns, m = S.shape[:2]
    V = S.reshape(-1, 3)
    idx = np.arange(ns * m).reshape(ns, m)
    j = np.arange(m if closed else m - 1)
    j2 = (j + 1) % m
    Qs = [np.stack([idx[i][j], idx[i][j2], idx[i + 1][j2], idx[i + 1][j]], 1) for i in range(ns - 1)]
    Q = np.concatenate(Qs)
    Ts, extra = [], []
    nv = len(V)
    if closed:
        for which, flag in ((0, caps[0]), (ns - 1, caps[1])):
            if not flag:
                continue
            c = S[which].mean(0)
            extra.append(c)
            ci = nv + len(extra) - 1
            ring = idx[which]
            if which == 0:                                       # faces the start: reverse
                Ts.append(np.stack([ring[j2], ring[j], np.full(m, ci)], 1))
            else:
                Ts.append(np.stack([ring[j], ring[j2], np.full(m, ci)], 1))
    if extra:
        V = np.concatenate([V, np.array(extra)])
    UV = None
    if uv:
        u = np.broadcast_to(np.linspace(0, 1, ns)[:, None], (ns, m))
        v = np.broadcast_to(np.linspace(0, 1, m)[None, :], (ns, m))
        UV = np.concatenate([np.stack([u, v], -1).reshape(-1, 2), np.zeros((len(extra), 2))])
    return Mesh(V, Q, np.concatenate(Ts) if Ts else None, UV=UV, mat=mat)


def sweep(path, section, scale=None, up=(0.0, 0.0, 1.0), caps=(True, True), closed=False, mat=0, uv=False):
    """A 2D `section` (m, 2) of (u, v) points (counter-clockwise seen along +T) swept along `path` (n, 3):
    point = path + u * U + v * V with V = the up vector made perpendicular to the path and U = V x T. `scale` is one
    value or one per path point (a single factor or (su, sv)). closed: the path is a ring (no caps)."""
    path = np.asarray(path, float)
    sec = np.asarray(section, float)
    T, U, B = _frames(path, up, closed)
    n = len(path)
    sc = np.ones((n, 2)) if scale is None else np.broadcast_to(
        np.asarray(scale, float).reshape(-1, 1) if np.ndim(scale) == 1 else np.asarray(scale, float), (n, 2))
    S = (path[:, None, :] + sec[None, :, 0:1] * sc[:, None, 0:1] * U[:, None, :]
         + sec[None, :, 1:2] * sc[:, None, 1:2] * B[:, None, :])
    if closed:
        # ring path: connect the last section to the first
        S = np.concatenate([S, S[:1]], axis=0)
        return skin(S, caps=(False, False), closed=True, mat=mat, uv=uv)
    return skin(S, caps=caps, closed=True, mat=mat, uv=uv)


def tube(path, radius, sides=12, caps=(True, True), closed=False, mat=0, up=(0.0, 0.0, 1.0), aspect=1.0):
    """Round (or elliptical, `aspect` = v/u) tube along a path; radius one value or per point."""
    t = 2 * math.pi * np.arange(sides) / sides
    sec = np.stack([np.cos(t), np.sin(t) * aspect], 1)
    rad = np.broadcast_to(np.asarray(radius, float), (len(path),))
    return sweep(path, sec, scale=rad, up=up, caps=caps, closed=closed, mat=mat)


def torus_path(R, n=48, plane="xy", center=(0.0, 0.0, 0.0)):
    """Ring path (n points) of radius R around the origin in a coordinate plane."""
    a = 2 * math.pi * np.arange(n) / n
    c = np.asarray(center, float)
    if plane == "xy":
        return np.stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a), np.full(n, c[2])], 1)
    if plane == "yz":
        return np.stack([np.full(n, c[0]), c[1] + R * np.cos(a), c[2] + R * np.sin(a)], 1)
    return np.stack([c[0] + R * np.cos(a), np.full(n, c[1]), c[2] + R * np.sin(a)], 1)


def ribbon(path, width, normal, lift=0.0, thick=0.0, mat=0, closed=False):
    """A flat strip of `width` along a path lying on a surface: the strip's face looks along `normal` (one vector or one
    per point) and floats `lift` above the path; with `thick` > 0 it is a thin solid (closed with side walls)."""
    P = np.asarray(path, float)
    n = len(P)
    N = np.broadcast_to(np.asarray(normal, float), (n, 3))
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.gradient(P, axis=0)
    T = T / np.maximum(np.linalg.norm(T, axis=1), EPS)[:, None]
    S = np.cross(N, T)                                           # across the strip
    S = S / np.maximum(np.linalg.norm(S, axis=1), EPS)[:, None]
    c = P + N * lift
    if thick <= 0:
        L, R = c - S * width / 2, c + S * width / 2
        V = np.concatenate([L, R])
        i = np.arange(n if closed else n - 1)
        i2 = (i + 1) % n
        Q = np.stack([i, n + i, n + i2, i2], 1)
        m = Mesh(V, Q, mat=mat)
        # the winding depends on the path's turn: make the face normals look along N
        nq, _ = m.face_vectors()
        if len(nq) and (nq[0] @ N[0]) < 0:
            m = m.flip()
        return m
    sec = np.array([[-width / 2, 0.0], [width / 2, 0.0], [width / 2, thick], [-width / 2, thick]])
    out = sweep(P + N * lift, sec, up=N[0], closed=closed, mat=mat)
    return out if out.volume() >= 0 else out.flip()


# ===================================================================================================================
# profile extrusion: the body panels
# ===================================================================================================================
def _stations(bottom, top, extra=()):
    """Common y stations of two y-monotone chains (y never decreases; each chain is a function z(y) except that both may
    start and end with a vertical edge), plus the `extra` y values. Returns Y, zb, zt arrays of the same length."""
    bottom = np.asarray(bottom, float)
    top = np.asarray(top, float)
    ys = np.unique(np.round(np.concatenate([bottom[:, 0], top[:, 0], np.asarray(extra, float)]), 9))

    def sample(ch):
        y, z = ch[:, 0], ch[:, 1]
        if np.any(np.diff(y) < -1e-12):
            raise ValueError("a profile chain must not go backwards in y")
        # ties (a vertical step): np.interp takes the later point for an exact hit, so vertical edges are only legal at
        # the two ends, where the first/last y value is the one that is shared
        out = np.interp(ys, y, z)
        return out
    return ys, sample(bottom), sample(top)


def extrude_profile(bottom, top, x0, x1, r0=0.0, r1=0.0, k0=2, k1=2, flush=(True, True), caps=(True, True), mat=0,
                    taper=0.012):
    """Pull a side profile across x with rounded rims.

    bottom, top: (n, 2) chains of (y, z), y increasing, running from the same first y to the same last y; the region
    between them is the section. Look at it from +X (y to the right, z up). x0 < x1 are the two flat sides (the -X
    and +X caps), r0 / r1 the radius of the rounded rim on that side (0 = a sharp edge), k0 / k1 the segments per
    quarter. flush=(front, back): the two vertical end edges stay sharp so a neighbouring panel butts against them; the
    rounding fades in over `taper` metres next to such an end (stations are added there).
    caps: whether the -X / +X side faces are built (a side that is hidden inside another panel can skip its cap).

    The rim is the section of a quarter circle: ring j sits at x1 - r (1 - cos a) with the profile moved inside by
    r (1 - sin a); the flat top/bottom faces then bridge from the +X rim to the -X rim.
    Returns a Mesh (closed when both caps are built)."""
    y_lo, y_hi = min(bottom[0][0], top[0][0]), max(bottom[-1][0], top[-1][0])
    extra = ([y_lo + taper] if flush[0] else []) + ([y_hi - taper] if flush[1] else [])
    Y, zb, zt = _stations(bottom, top, extra)
    ns = len(Y)
    pts = [(Y[i], zb[i]) for i in range(ns)] + [(Y[i], zt[i]) for i in range(ns - 1, -1, -1)]
    idxb = list(range(ns))
    idxt = [2 * ns - 1 - i for i in range(ns)]                    # polygon index of top station i
    drop = []
    if abs(zt[0] - zb[0]) < 1e-7:                                  # the chains meet at the front: one vertex
        drop.append(idxt[0])
        idxt[0] = idxb[0]
    if abs(zt[-1] - zb[-1]) < 1e-7:
        drop.append(idxt[-1])
        idxt[-1] = idxb[-1]
    keep = [i for i in range(len(pts)) if i not in drop]
    newidx = {old: new for new, old in enumerate(keep)}
    P = np.array([pts[i] for i in keep])
    idxb = [newidx[i] for i in idxb]
    idxt = [newidx[i] for i in idxt]
    n = len(P)
    if polygon_area(P) < 0:
        raise ValueError("profile must run bottom chain left to right, then top chain right to left")
    M = mitre_offsets(P)
    # per-vertex radius factor: the vertices of a flush end edge stay sharp
    wgt = np.ones(n)
    if flush[0]:
        wgt[[idxb[0], idxt[0]]] = 0.0
    if flush[1]:
        wgt[[idxb[-1], idxt[-1]]] = 0.0

    def ring(x_side, r, a, sign):
        rr = r * wgt
        s = rr * (1 - math.sin(a))                                  # inset of the profile
        e = rr * (1 - math.cos(a))                                  # distance from the side plane
        xs = x_side - sign * e
        return np.concatenate([xs[:, None], P + M * s[:, None]], 1)       # (n, 3): x, y, z

    rings = []
    ang1 = [(math.pi / 2) * j / k1 for j in range(k1 + 1)] if r1 > 0 else [0.0]
    ang0 = [(math.pi / 2) * j / k0 for j in range(k0 + 1)] if r0 > 0 else [0.0]
    for a in ang1:
        rings.append(ring(np.full(n, x1), r1, a, +1))
    for a in reversed(ang0):
        rings.append(ring(np.full(n, x0), r0, a, -1))
    R = np.array(rings)                                             # (nr, n, 3) as (x, y, z)
    V3 = np.stack([R[..., 0], R[..., 1], R[..., 2]], -1)
    nr = len(rings)
    V = V3.reshape(-1, 3)
    idx = np.arange(nr * n).reshape(nr, n)
    j = np.arange(n)
    j2 = (j + 1) % n
    Qs = [np.stack([idx[t][j], idx[t + 1][j], idx[t + 1][j2], idx[t][j2]], 1) for t in range(nr - 1)]
    Q = np.concatenate(Qs)
    cap_q = []
    for which, flag, face_up in ((0, caps[1], True), (nr - 1, caps[0], False)):
        if not flag:
            continue
        rg = idx[which]
        for i in range(ns - 1):
            q = [rg[idxb[i]], rg[idxb[i + 1]], rg[idxt[i + 1]], rg[idxt[i]]]
            cap_q.append(q if face_up else q[::-1])
    if cap_q:
        Q = np.concatenate([Q, np.array(cap_q)])
    # the profile was set up with (y, z) = the in-plane coordinates; swap to world (x, y, z)
    return Mesh(V, Q, mat=mat).weld(1e-8)


# ===================================================================================================================
# lofted shells: control polygons along an axis
# ===================================================================================================================
Pt = namedtuple("Pt", "name x z y crease")
Station = namedtuple("Station", "y pts hard ring")


def pt(name, x, z, y=0.0, crease=0.0):
    """One control point of a half section: lateral x (>= 0), height z, an offset y along the loft axis (a raked or chevron
    face: the point lies at the station's y + y) and the crease weight (0..1) of the edges that run along the loft through
    it (a character line stays sharp where the weight is 1)."""
    return Pt(str(name), float(x), float(z), float(y), float(crease))


def arc_points(name, cx, cz, r, a0, a1, n, y=0.0, crease=0.0):
    """n control points `name0`, `name1`, ... on the circle of radius r about (cx, cz) from angle a0 to a1 (degrees,
    0 = +x, 90 = +z): a rounded corner of a section."""
    a = np.radians(np.linspace(a0, a1, n))
    return [pt(f"{name}{k}", cx + r * math.cos(t), cz + r * math.sin(t), y, crease) for k, t in enumerate(a)]


def station(y, pts, hard=False, ring=0.0):
    """A section at position y along the loft axis. `pts` is the half section (see `loft`). hard=True breaks the smooth
    interpolation there (the surface is only continuous, a sharp turn along the axis: put a ring crease on it). `ring` is
    the crease weight of the edges that run around the section at this station: a float for the whole ring, or
    (weight, first point name, last point name) for the part of it between two named points (both sides)."""
    return Station(float(y), tuple(pts), bool(hard), ring)


def _pchip_end(h0, h1, d0, d1):
    d = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    d = np.where(np.sign(d) != np.sign(d0), 0.0, d)
    return np.where((np.sign(d0) != np.sign(d1)) & (np.abs(d) > 3 * np.abs(d0)), 3 * d0, d)


def pchip(x, y, xq):
    """Monotone cubic Hermite interpolation (Fritsch-Carlson: no overshoot between stations) of the columns of y (n, K)
    given at x (n,), evaluated at xq (m,) -> (m, K)."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    n = len(x)
    if n == 1:
        return np.repeat(y[:1], len(xq), 0)
    hh = np.diff(x)
    h = hh[:, None]
    delta = np.diff(y, axis=0) / h
    if n == 2:
        d = np.vstack([delta[0], delta[0]])
    else:
        d = np.zeros_like(y)
        w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
        dk, dk1 = delta[:-1], delta[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            d[1:-1] = np.where(dk * dk1 > 0, (w1 + w2) / (w1 / dk + w2 / dk1), 0.0)
        d[0] = _pchip_end(h[0], h[1], delta[0], delta[1])
        d[-1] = _pchip_end(h[-1], h[-2], delta[-1], delta[-2])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    t = ((xq - x[i]) / hh[i])[:, None]
    hi = hh[i][:, None]
    return ((1 + 2 * t) * (1 - t) ** 2 * y[i] + t * (1 - t) ** 2 * hi * d[i]
            + t * t * (3 - 2 * t) * y[i + 1] + t * t * (t - 1) * hi * d[i + 1])


def sample_stations(ys, spacing):
    """The sample positions of a loft: every key station plus extra ones so no gap is wider than `spacing` (a number, or a
    function of y)."""
    out = [ys[0]]
    for a, b in zip(ys[:-1], ys[1:]):
        sp = spacing(0.5 * (a + b)) if callable(spacing) else spacing
        n = max(1, int(math.ceil((b - a) / sp - 1e-9)))
        out.extend(a + (b - a) * np.arange(1, n + 1) / n)
    return np.array(out)


class Loft:
    """A lofted cage (`mesh`) and the maps to address it. Rows are the sample positions `ys`, columns go around the section
    (`C` of them: the half section's `K` points, then its mirror image without the two centre points). `grid[r, c]` is the
    vertex index; body quad (r, c) has index r * C + c and spans rows r, r + 1 and columns c, c + 1 (mod C). `order[c]` is
    the index of the named point a column belongs to, `side[c]` is +1 on the +x half, -1 on the mirrored half and 0 for the
    centre-line points (shared by both halves)."""

    def __init__(self, mesh, ys, grid, names, order, side, mirror):
        self.mesh, self.ys, self.grid, self.names, self.order, self.side, self.mirror = mesh, ys, grid, list(names), order, side, mirror
        self.R, self.C = grid.shape
        self.K = len(self.names)

    # ---- addressing
    def row(self, y):
        """The row nearest to position y."""
        return int(np.argmin(np.abs(self.ys - y)))

    def rows(self, y=None):
        """Row indices between positions y = (y0, y1) inclusive (None: all rows)."""
        if y is None:
            return np.arange(self.R)
        return np.nonzero((self.ys >= y[0] - 1e-9) & (self.ys <= y[1] + 1e-9))[0]

    def cols(self, cols=None, side=None):
        """Column indices of the points from the named point cols[0] to cols[1] (in section order, inclusive; None: all) on
        the given side (+1, -1; None: both halves). The centre-line points belong to both sides."""
        k0, k1 = (0, self.K - 1) if cols is None else sorted((self.names.index(cols[0]), self.names.index(cols[1])))
        ok = (self.order >= k0) & (self.order <= k1)
        if side is not None and self.mirror:
            ok &= (self.side == side) | (self.side == 0)
        return np.nonzero(ok)[0]

    def col(self, name, side=+1):
        """The column of a named point on the +x (side=+1) or -x side."""
        c = self.cols((name, name), side)
        return int(c[0])

    def vid(self, y, name, side=+1):
        """Vertex index of a named point at the row nearest y."""
        return int(self.grid[self.row(y), self.col(name, side)])

    def ring(self, y):
        """The control ring at position y (C, 3): linear between the two nearest sample rows."""
        V = self.mesh.V[self.grid.ravel()].reshape(self.R, self.C, 3)
        i = int(np.clip(np.searchsorted(self.ys, y, side="right") - 1, 0, self.R - 2))
        t = (y - self.ys[i]) / (self.ys[i + 1] - self.ys[i])
        return V[i] * (1 - t) + V[i + 1] * t

    def point(self, name, y, side=+1):
        """A named control point at position y (x, y, z)."""
        return self.ring(y)[self.col(name, side)]

    def x_at(self, y, z, side=+1, cols=None):
        """Where the control ring at position y first crosses the height z going from the first named point of `cols` to the
        last (None: bottom centre to top centre) along the +x half (side +1) or the -x half (side -1): (x, y) of the crossing,
        x signed, or None when it does not cross."""
        cs = self.cols(cols, side)
        cs = cs[np.argsort(self.order[cs], kind="stable")]
        P = self.ring(y)[cs]
        for p, q in zip(P[:-1], P[1:]):
            if (p[2] - z) * (q[2] - z) <= 0 and p[2] != q[2]:
                u = p + (q - p) * ((z - p[2]) / (q[2] - p[2]))
                return float(u[0]), float(u[1])
        return None

    # ---- editing the cage after the loft
    def assign(self, mat, y=None, cols=None, side=None):
        """Material `mat` for the body quads between positions y = (y0, y1) (None: all) whose two columns both lie between
        the named points `cols` (None: all) on the given side (None: both). Subdivision keeps it: the fine faces inherit."""
        rows = self.rows(y)
        rows = rows[rows < self.R - 1]
        cs = set(self.cols(cols, side).tolist())
        fc = np.array([c for c in range(self.C) if c in cs and (c + 1) % self.C in cs], np.int64)
        if len(rows) and len(fc):
            self.mesh.Qm[(rows[:, None] * self.C + fc[None, :]).ravel()] = mat
        return self

    def crease_ring(self, y, w=1.0, cols=None, side=None):
        """Crease the edges that run around the section at position y, between the named points `cols` (None: all)."""
        r = self.row(y)
        cs = set(self.cols(cols, side).tolist())
        self.mesh.crease([(self.grid[r, c], self.grid[r, (c + 1) % self.C]) for c in range(self.C)
                          if c in cs and (c + 1) % self.C in cs], w)
        return self

    def crease_col(self, name, w=1.0, y=None, side=None):
        """Crease the edges that run along the loft through a named point between positions y = (y0, y1)."""
        rows = self.rows(y)
        pairs = [(self.grid[a, c], self.grid[a + 1, c]) for c in self.cols((name, name), side) for a in rows[:-1]]
        self.mesh.crease(pairs, w)
        return self


def _cap(V, ring_ids, axis, rings, bulge):
    """Faces closing a ring of vertices that looks along the unit vector `axis`: `rings` shrunken copies of the ring toward
    its centre (a flat disc of quads: no single pinched pole on a big flat face), then a fan to the centre, which is pushed
    out along `axis` by `bulge` (a dome; the rings follow a parabola). Returns (new vertices (k, 3), triangles, quads); the
    indices count from len(V) and the faces look along `axis`."""
    ring = V[ring_ids]
    c = ring.mean(0)
    n0, C = len(V), len(ring_ids)
    axis = np.asarray(axis, float)
    new, Q = [], []
    prev = np.asarray(ring_ids)
    for k in range(1, rings + 1):
        t = 1.0 - k / (rings + 1.0)
        ids = n0 + len(new) + np.arange(C)
        new.extend(c + (ring - c) * t + axis * (bulge * (1 - t * t)))
        Q.append(np.stack([prev, np.roll(prev, -1), np.roll(ids, -1), ids], 1))
        prev = ids
    ai = n0 + len(new)
    new.append(c + axis * bulge)
    T = np.stack([prev, np.roll(prev, -1), np.full(C, ai)], 1)
    Qa = np.concatenate(Q) if Q else np.zeros((0, 4), np.int64)
    Vall = np.concatenate([V, np.array(new)])
    look = 0.0
    for F in (T, Qa):
        if len(F):
            P = Vall[F]
            look += np.cross(P[:, 1] - P[:, 0], P[:, -1] - P[:, 0]).sum(0) @ axis
    if look < 0:
        T, Qa = T[:, ::-1], Qa[:, ::-1]
    return np.array(new), T, Qa


Gap = namedtuple("Gap", "y width depth cols mat")


def gap(y, width=0.006, depth=0.003, cols=None, mat=0):
    """A panel gap across a loft at position y: a groove `width` wide and `depth` deep (a V: two creased rows either side
    of one row that is pushed in along the section's normal), over the points `cols` = (first name, last name) of the
    half section (None: the whole ring). The faces between the creased rows get material `mat` (the dark of the gap)."""
    return Gap(float(y), float(width), float(depth), cols, int(mat))


def loft(stations, spacing=0.06, mirror=True, caps=("fan", "fan"), cap_bulge=(0.0, 0.0), cap_rings=2, gaps=(), uv=False,
         mat=0):
    """Loft a body from sections along y (the loft axis; rotate the result for another axis).

    stations: a list of `station(y, pts)` with strictly increasing y. A station's `pts` is a HALF cross-section, the control
    polygon of the points `pt(name, x, z)` from the underside on the centre line (x = 0) around the +x side to the top on
    the centre line (x = 0); every station has the same names in the same order. mirror=True builds the -x half as a mirror
    image (the centre points are shared); mirror=False takes `pts` as the whole cyclic ring (no centre-line rule).
    The control polygons are interpolated along y with monotone cubics (no overshoot), so a few stations describe a long
    smooth shape; at a `hard` station the shape only continues (a sharp turn), and its ring is usually creased. Positions
    between stations are sampled every `spacing` metres at most (a number, or a function of y); every station is a row.
    caps: what closes the first / last ring: "fan" (`cap_rings` shrunken rings of quads and a fan to a centre point pushed out
    by cap_bulge) or None (open). gaps: `gap(...)` grooves across the loft. uv=True gives every vertex u = the share of the
    ring's length (from the first point) and v = the share of the loft's length.
    The result is a cage: subdivide it (a Subdivision Surface) and the polygon rounds off into a smooth surface; creases keep
    edges sharp (point creases along the body, ring creases around it).
    Open-top regions (a cockpit): let the top points of the stations in that range run down a lip, the inner wall and across
    a floor instead of over the top: the section is then a U, and the hard stations at its ends turn the top surface into
    the opening's front and back walls.
    Returns a `Loft`."""
    S = list(stations)
    if len(S) < 2:
        raise ValueError("loft needs at least two stations")
    names = [p.name for p in S[0].pts]
    for st in S:
        if [p.name for p in st.pts] != names:
            raise ValueError(f"station y={st.y}: the point names differ from the first station's")
    Y = np.array([st.y for st in S])
    if np.any(np.diff(Y) <= 0):
        raise ValueError("stations must be given in strictly increasing y")
    K = len(names)
    X, Z, DY, CR = (np.array([[getattr(p, f) for p in st.pts] for st in S]) for f in ("x", "z", "y", "crease"))
    if mirror:
        if np.abs(X[:, [0, -1]]).max() > 1e-9:
            raise ValueError("a half section starts and ends on the centre line (x = 0)")
        if K < 3:
            raise ValueError("a half section needs at least three points")
    keys = np.unique(np.round(np.concatenate([Y] + [[g.y - 0.5 * g.width, g.y, g.y + 0.5 * g.width] for g in gaps]), 9))
    ys = sample_stations(keys, spacing)
    R = len(ys)
    hard = [0] + [i for i in range(1, len(S) - 1) if S[i].hard] + [len(S) - 1]

    def interp(Q):
        out = np.empty((R, K))
        for a, b in zip(hard[:-1], hard[1:]):
            sel = (ys >= Y[a] - 1e-12) & (ys <= Y[b] + 1e-12)
            out[sel] = pchip(Y[a:b + 1], Q[a:b + 1], ys[sel])
        return out

    PX, PZ, PDY, PCR = (interp(Q) for Q in (X, Z, DY, CR))
    span = lambda cols: (0, K - 1) if cols is None else tuple(sorted((names.index(cols[0]), names.index(cols[1]))))
    gap_rows = []
    for g in gaps:                                           # the groove: the middle row is pushed in along the normal
        r = int(np.argmin(np.abs(ys - g.y)))
        k0, k1 = span(g.cols)
        P = np.stack([PX[r], PZ[r]], 1)
        tv = np.empty_like(P)
        tv[1:-1], tv[0], tv[-1] = P[2:] - P[:-2], P[1] - P[0], P[-1] - P[-2]
        nv = np.stack([tv[:, 1], -tv[:, 0]], 1)
        nv /= np.maximum(np.linalg.norm(nv, axis=1), EPS)[:, None]
        if mirror:
            nv[0], nv[-1] = (0.0, -1.0), (0.0, 1.0)
        PX[r, k0:k1 + 1] -= g.depth * nv[k0:k1 + 1, 0]
        PZ[r, k0:k1 + 1] -= g.depth * nv[k0:k1 + 1, 1]
        gap_rows.append((r, k0, k1, g))
    if mirror:
        order = np.concatenate([np.arange(K), np.arange(K - 2, 0, -1)])
        sign = np.concatenate([np.ones(K), -np.ones(K - 2)])
        side = np.concatenate([[0], np.ones(K - 2), [0], -np.ones(K - 2)]).astype(int)
    else:
        order, sign, side = np.arange(K), np.ones(K), np.ones(K, int)
    C = len(order)
    V = np.stack([PX[:, order] * sign, ys[:, None] + PDY[:, order], PZ[:, order]], -1).reshape(-1, 3)
    grid = np.arange(R * C).reshape(R, C)
    r, c = np.arange(R - 1)[:, None], np.arange(C)[None, :]
    c2 = (c + 1) % C
    Q = np.stack([r * C + c, r * C + c2, (r + 1) * C + c2, (r + 1) * C + c], -1).reshape(-1, 4)
    # orientation: the body quads face away from the axis (the section's centre) on the whole
    P = V[Q]
    nrm = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    ctr = V.reshape(R, C, 3).mean(1)[np.repeat(np.arange(R - 1), C)]
    if np.einsum("ij,ij->i", nrm, P.mean(1) - ctr).sum() < 0:
        Q = Q[:, ::-1]
    UV = None
    if uv:
        P3 = V.reshape(R, C, 3)
        seg = np.linalg.norm(np.roll(P3, -1, axis=1) - P3, axis=2)
        u = (np.cumsum(seg, axis=1) - seg) / np.maximum(seg.sum(1, keepdims=True), EPS)
        UV = np.stack([u, np.broadcast_to(((ys - ys[0]) / (ys[-1] - ys[0]))[:, None], (R, C))], -1).reshape(-1, 2)
    # end caps (outward along y)
    T, extraQ, newV = [], [], []
    for end, kind in enumerate(caps):
        if kind is None:
            continue
        if kind != "fan":
            raise ValueError(f"cap {kind!r}: use 'fan' or None")
        axis = (0.0, -1.0 if end == 0 else 1.0, 0.0)
        nv, t, q = _cap(np.concatenate([V] + newV) if newV else V, grid[0 if end == 0 else R - 1], axis, cap_rings, cap_bulge[end])
        newV.append(nv)
        T.append(t)
        extraQ.append(q)
        if uv:
            UV = np.concatenate([UV, np.tile([0.5, float(end)], (len(nv), 1))])
    if newV:
        V = np.concatenate([V] + newV)
    if extraQ:
        Q = np.concatenate([Q] + extraQ)
    mesh = Mesh(V, Q, np.concatenate(T) if T else None, UV=UV, mat=mat)
    # creases: along the body (per point, averaged over the two rows of an edge), around it (per station and gap rim)
    pairs, ws = [], []
    for k in range(K):
        w = 0.5 * (PCR[:-1, k] + PCR[1:, k])
        m = w > 1e-4
        for col in np.nonzero(order == k)[0]:
            pairs.append(np.stack([grid[:-1, col], grid[1:, col]], 1)[m])
            ws.append(w[m])

    def ring_edges(row, k0, k1):
        return [(grid[row, col], grid[row, (col + 1) % C]) for col in range(C)
                if k0 <= order[col] <= k1 and k0 <= order[(col + 1) % C] <= k1 and (mirror or col + 1 < C or (k0 == 0 and k1 == K - 1))]

    for st in S:
        spec = st.ring
        if not spec:
            continue
        w, k0, k1 = (float(spec), 0, K - 1) if np.isscalar(spec) else (float(spec[0]), names.index(spec[1]), names.index(spec[2]))
        ids = ring_edges(int(np.argmin(np.abs(ys - st.y))), min(k0, k1), max(k0, k1))
        pairs.append(np.array(ids, np.int64).reshape(-1, 2))
        ws.append(np.full(len(ids), w))
    for r, k0, k1, g in gap_rows:
        for rr in (r - 1, r + 1):
            ids = ring_edges(rr, k0, k1)
            pairs.append(np.array(ids, np.int64).reshape(-1, 2))
            ws.append(np.ones(len(ids)))
        for qr in (r - 1, r):
            for col in range(C):
                if k0 <= order[col] <= k1 and k0 <= order[(col + 1) % C] <= k1:
                    mesh.Qm[qr * C + col] = g.mat
    if pairs:
        mesh.crease(np.concatenate(pairs), np.concatenate(ws))
    return Loft(mesh, ys, grid, names, order, side, mirror)


# ===================================================================================================================
# cutters
# ===================================================================================================================
def arch_cutter(center, radius, x0, x1, flare=0.025, lip=0.03, seg=64, lip_n=4, mat=0):
    """The solid a round wheel arch is cut with: a cylinder about the X axis through (y, z) = `center`, of `radius`, from
    x0 (inside the body) to x1 (outside it), whose outer end flares out by `flare` along a quarter-round over the last
    `lip` metres. Subtract it from a closed body (a Boolean) and the arch gets a rounded rim. x1 < x0 makes the cutter on
    the -x side. Closed, outward normals."""
    (yc, zc), R = center, radius
    sgn = -1.0 if x1 < x0 else 1.0                       # built on the +x side, mirrored for the -x side
    a, b = sgn * x0, sgn * x1
    if flare > 0:
        t = np.linspace(0.0, 0.5 * math.pi, lip_n + 1)
        rim = [(R + flare * (1 - math.cos(u)), b - lip * (1 - math.sin(u))) for u in t]
    else:
        rim = [(R, b)]
    m = lathe([(0.0, a), (R, a)] + rim + [(0.0, b)], seg=seg, mat=mat).rot(ry=90.0).moved((0.0, yc, zc))
    return m.mirrored_x() if sgn < 0 else m


# ===================================================================================================================
# cages from rings, outlines and panels
# ===================================================================================================================
def cage(rings, closed=True, loop=False, caps=(None, None), cap_axes=(None, None), cap_bulge=(0.0, 0.0), cap_rings=2,
         crease_cols=None, crease_rings=None, flip=None, uv=False, mat=0):
    """A quad cage from a grid of points `rings` (ns, m, 3): ring i runs through m points (closed: round a section, the last
    point joins the first; open: across a strip) and the rings follow one another (loop: the last ring joins the first, a
    torus). Unlike `loft` the rings may lie anywhere (a dash along x, a surround along y, a pleated seat).
    crease_cols (m,) weights 0..1 of the edges that run from ring to ring through point j; crease_rings (ns,) weights of
    the edges that run round ring i. caps ("fan" or None for the first / last ring of a closed tube): shrunken rings of
    quads and a fan to a centre pushed out by cap_bulge along `cap_axes` (default: away from the neighbouring ring).
    Faces face outward: automatically for a closed tube or a torus, as the winding of the points gives for an open strip
    (quad normal = along the ring x from ring to ring; flip=True reverses). uv=True: u along the ring, v ring to ring."""
    G = np.asarray(rings, float)
    ns, m = G.shape[:2]
    idx = np.arange(ns * m).reshape(ns, m)
    j = np.arange(m if closed else m - 1)
    j2 = (j + 1) % m
    i = np.arange(ns if loop else ns - 1)
    i2 = (i + 1) % ns
    Q = np.stack([idx[i][:, j], idx[i][:, j2], idx[i2][:, j2], idx[i2][:, j]], -1).reshape(-1, 4)
    V = G.reshape(-1, 3)
    if flip is None:
        if loop:
            flip = Mesh(V, Q).volume() < 0
        elif closed:
            P = V[Q]
            ctr = G.mean(1)[np.repeat(np.arange(len(i)), len(j))]
            flip = np.einsum("ij,ij->i", np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), P.mean(1) - ctr).sum() < 0
        else:
            flip = False
    if flip:
        Q = Q[:, ::-1]
    UV = None
    if uv:
        u = np.broadcast_to(np.linspace(0.0, 1.0, m)[None, :], (ns, m))
        v = np.broadcast_to(np.linspace(0.0, 1.0, ns)[:, None], (ns, m))
        UV = np.stack([u, v], -1).reshape(-1, 2)
    T, extraQ, newV = [], [], []
    if closed and not loop:
        for end, kind in enumerate(caps):
            if kind is None:
                continue
            if kind != "fan":
                raise ValueError(f"cap {kind!r}: use 'fan' or None")
            ring_i = 0 if end == 0 else ns - 1
            axis = cap_axes[end]
            if axis is None:
                d = G[0].mean(0) - G[1].mean(0) if end == 0 else G[-1].mean(0) - G[-2].mean(0)
                axis = d / max(np.linalg.norm(d), EPS)
            nv, t, q = _cap(np.concatenate([V] + newV) if newV else V, idx[ring_i], axis, cap_rings, cap_bulge[end])
            newV.append(nv)
            T.append(t)
            extraQ.append(q)
            if uv:
                UV = np.concatenate([UV, np.tile([0.5, float(end)], (len(nv), 1))])
    if newV:
        V = np.concatenate([V] + newV)
    if extraQ:
        Q = np.concatenate([Q] + extraQ)
    mesh = Mesh(V, Q, np.concatenate(T) if T else None, UV=UV, mat=mat)
    pairs, ws = [], []
    if crease_cols is not None:
        cc = np.broadcast_to(np.asarray(crease_cols, float), (m,))
        for col in np.nonzero(cc > 1e-4)[0]:
            pairs.append(np.stack([idx[i][:, col], idx[i2][:, col]], 1))
            ws.append(np.full(len(i), cc[col]))
    if crease_rings is not None:
        cr = np.broadcast_to(np.asarray(crease_rings, float), (ns,))
        for ring_i in np.nonzero(cr > 1e-4)[0]:
            pairs.append(np.stack([idx[ring_i][j], idx[ring_i][j2]], 1))
            ws.append(np.full(len(j), cr[ring_i]))
    if pairs:
        mesh.crease(np.concatenate(pairs), np.concatenate(ws))
    return mesh


def superellipse(w, h, p=3.0, n=48):
    """Closed CCW outline (n, 2) of |x / a|^p + |y / b|^p = 1 with a = w / 2, b = h / 2: p = 2 is an ellipse, 3-4 the
    rounded square of a switch cap or a horn pad, large p a rectangle."""
    t = 2.0 * math.pi * np.arange(n) / n
    c, s = np.cos(t), np.sin(t)
    return np.stack([0.5 * w * np.sign(c) * np.abs(c) ** (2.0 / p), 0.5 * h * np.sign(s) * np.abs(s) ** (2.0 / p)], 1)


def stadium(w, h, n=8):
    """Closed CCW outline of a w x h rectangle whose short ends are half circles (n segments per quarter)."""
    return rrect(w, h, 0.5 * min(w, h), n)


def resample_loop(P, n):
    """n points evenly spaced by arc length on the closed polygon P (m, 2 or 3)."""
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(Q.shape[1])], 1)


def offset_rings(outline, profile):
    """Rings (len(profile), m, 3) for a closed CCW outline (m, 2): one ring per (d, h) of `profile`, the outline moved d
    inward (negative: outward; mitred, so walls stay parallel to the outline) at height h."""
    P = np.asarray(outline, float)
    if polygon_area(P) < 0:
        P = P[::-1]
    M = mitre_offsets(P)
    return np.stack([np.column_stack([P + d * M, np.full(len(P), h)]) for d, h in profile])


def _roll(r, n):
    """Quarter-round of radius r in n steps from vertical to horizontal as (inward d, height drop) pairs, excluding the
    start: a convex rim rolled over."""
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)[1:]
    return [(r * (1.0 - math.cos(t)), r * (1.0 - math.sin(t))) for t in a]


def rounded_panel(outline, thickness, r_edge=0.003, n=3, dome=0.0, rings=2, uv=False, mat=0):
    """A plate with a rolled rim, its back at z = 0 and its face at z = thickness: the closed 2D `outline` (m, 2), the top
    edge rounded with radius r_edge (n segments; keep it below the outline's own corner radii) and the face domed up by
    `dome` at the centre. Closed, outward normals. uv=True maps the outline's bounding box onto 0..1 (a grain along x:
    make the outline longer along x)."""
    r = min(r_edge, thickness)
    prof = [(0.0, 0.0), (0.0, thickness - r)] + [(d, thickness - dh) for d, dh in _roll(r, n)]
    R = offset_rings(outline, prof)
    m = cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_bulge=(0.0, dome), cap_rings=rings,
             mat=mat)
    if uv:
        lo, hi = np.asarray(outline).min(0), np.asarray(outline).max(0)
        m.UV = (m.V[:, :2] - lo) / np.maximum(hi - lo, EPS)
    return m


def rounded_frame(outer, inner, height, r_out=0.003, r_in=0.002, n=3, floor=None, mat=0):
    """A frame (bezel, grille surround, vent): the closed 2D outlines `outer` and `inner` (inside it) pulled up to `height`
    from z = 0 with both top edges rolled over (r_out, r_in). floor=None: a through opening; floor=d: the opening is a
    pocket whose floor lies d below the top face (a closed lid on the back). Closed, outward normals. The outlines are
    resampled to the same point count when they differ."""
    A, B = np.asarray(outer, float), np.asarray(inner, float)
    if polygon_area(A) < 0:
        A = A[::-1]
    if polygon_area(B) < 0:
        B = B[::-1]
    if len(A) != len(B):
        m = max(len(A), len(B))
        A, B = resample_loop(A, m), resample_loop(B, m)
    ro, ri = min(r_out, height), min(r_in, height)
    ring = lambda P, d, h: offset_rings(P, [(d, h)])[0]
    R = [ring(A, 0.0, 0.0), ring(A, 0.0, height - ro)]
    R += [ring(A, d, height - dh) for d, dh in _roll(ro, n)]
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)
    R += [ring(B, -ri + ri * math.sin(t), height - ri + ri * math.cos(t)) for t in a]
    bottom = 0.0 if floor is None else max(height - floor, 0.0)
    R.append(ring(B, 0.0, bottom))
    R = np.stack(R)
    if floor is None:
        return cage(R, closed=True, loop=True, mat=mat)
    return cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_rings=2, mat=mat)


# ===================================================================================================================
# probing a surface
# ===================================================================================================================
class Probe:
    """Ray casts against a triangle mesh: put parts exactly on a smooth (subdivided) surface. Build it from the evaluated
    mesh Blender hands over (`mkmmd.blender.library.shell.evaluated_mesh`) or from any `Mesh`:
        p = Probe.from_mesh(mesh);  hit = p.ray(origin, direction)   ->  (t, point, unit normal) or None."""

    def __init__(self, V, T):
        V, T = np.asarray(V, float), np.asarray(T, np.int64)
        self.a = V[T[:, 0]]
        self.e1 = V[T[:, 1]] - self.a
        self.e2 = V[T[:, 2]] - self.a
        n = np.cross(self.e1, self.e2)
        self.n = n / np.maximum(np.linalg.norm(n, axis=1), EPS)[:, None]

    @classmethod
    def from_mesh(cls, mesh):
        return cls(mesh.V, mesh.triangles())

    def ray(self, origin, direction):
        """Nearest hit of the ray origin + t * direction (t > 0, both faces count): (t, point, normal facing the origin)
        or None."""
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        p = np.cross(d, self.e2)
        det = np.einsum("ij,ij->i", self.e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - self.a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, self.e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", self.e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        if not hit.any():
            return None
        k = int(np.argmin(np.where(hit, t, np.inf)))
        n = self.n[k] if self.n[k] @ d < 0 else -self.n[k]
        return float(t[k]), o + d * t[k], n

    def height(self, x, y, z0=10.0):
        """z of the top-most surface at (x, y) (a ray dropped from z0), or None."""
        h = self.ray((x, y, z0), (0.0, 0.0, -1.0))
        return None if h is None else float(h[1][2])
