"""Decals: thin strips (lashes, brows, creases, lip lines, nose line, blush patches) that lie on the skin.

A decal vertex is BOUND to the skin mesh: a skin triangle, barycentric weights and a height above the surface. Its
position and its morph offsets come from the skin's, so lashes follow the lids and lip lines follow the lips with no
gaps by construction; the skin's morph shapes are evaluated once and the decals ride on them."""
import numpy as np


def triangles(F):
    """Fan-triangulate a polygon list; returns (m, 3) int array."""
    out = []
    for f in F:
        for k in range(1, len(f) - 1):
            out.append((f[0], f[k], f[k + 1]))
    return np.asarray(out, int)


def vertex_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


class SkinSurface:
    """The front-facing skin triangles of the face mesh and a point lookup in the front (x, z) plane."""

    def __init__(self, V, F, group, skin_group=0):
        self.T = triangles([f for f, g in zip(F, group) if g == skin_group])
        self._place(V)

    def _place(self, V):
        self.V = np.asarray(V, float)
        self.N = vertex_normals(self.V, self.T)
        P = self.V[self.T]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.front = n[:, 1] < -1e-12                     # facing -y
        self.ids = np.where(self.front)[0]
        self.vert_ids = np.unique(self.T[self.ids].ravel())                # vertices of the front-facing skin
        Q = P[self.ids][:, :, [0, 2]]
        self.lo = Q.min(1)
        self.hi = Q.max(1)

    def moved(self, V):
        """The same skin triangles at other vertex positions (the skin under a morph)."""
        out = object.__new__(SkinSurface)
        out.T = self.T
        out._place(V)
        return out

    def bind(self, x, z, chunk=256, strict=True):
        """Triangle index (into self.T) and barycentric weights of the front-most skin triangle at each (x, z); points
        outside the front skin raise ValueError."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        n = len(x)
        tri = np.full(n, -1, int)
        bary = np.zeros((n, 3))
        depth = np.full(n, np.inf)
        Q = self.V[self.T[self.ids]][:, :, [0, 2]]
        Y = self.V[self.T[self.ids]][:, :, 1]
        for s in range(0, n, chunk):
            xs, zs = x[s:s + chunk], z[s:s + chunk]
            m = (xs[:, None] >= self.lo[None, :, 0]) & (xs[:, None] <= self.hi[None, :, 0]) & \
                (zs[:, None] >= self.lo[None, :, 1]) & (zs[:, None] <= self.hi[None, :, 1])
            qi, ti = np.nonzero(m)
            if len(qi) == 0:
                continue
            a, b, c = Q[ti, 0], Q[ti, 1], Q[ti, 2]
            p = np.stack([xs[qi], zs[qi]], -1)
            d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
            ok = np.abs(d) > 1e-18
            w0 = np.where(ok, ((b[:, 1] - c[:, 1]) * (p[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[:, 1] - c[:, 1])) / np.where(ok, d, 1), -1)
            w1 = np.where(ok, ((c[:, 1] - a[:, 1]) * (p[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[:, 1] - c[:, 1])) / np.where(ok, d, 1), -1)
            w2 = 1.0 - w0 - w1
            inside = ok & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            for q, t, a0, a1, a2 in zip(qi[inside], ti[inside], w0[inside], w1[inside], w2[inside]):
                yy = a0 * Y[t, 0] + a1 * Y[t, 1] + a2 * Y[t, 2]
                if yy < depth[s + q]:                        # front-most
                    depth[s + q] = yy
                    tri[s + q] = self.ids[t]
                    bary[s + q] = (a0, a1, a2)
        if strict and (tri < 0).any():
            raise ValueError(f"{int((tri < 0).sum())} decal points are not over the front skin")
        return tri, bary

    def point(self, tri, bary, V=None, N=None):
        V = self.V if V is None else V
        T = self.T[tri]
        return np.einsum("nk,nkd->nd", bary, V[T])

    def normal(self, tri, bary, N=None):
        N = self.N if N is None else N
        T = self.T[tri]
        n = np.einsum("nk,nkd->nd", bary, N[T])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


class Decal:
    """A set of bound vertices: skin triangle, barycentric weights, offset height along the skin normal."""

    def __init__(self, surf, tri, bary, height):
        self.surf, self.tri, self.bary = surf, np.asarray(tri), np.asarray(bary)
        self.h = np.broadcast_to(np.asarray(height, float), (len(self.tri),)).copy()

    def positions(self, V=None, N=None):
        """Vertex positions on the (possibly morphed) skin: `V` the skin's vertices, `N` its normals (None = rest)."""
        s = self.surf
        if V is not None and N is None:
            N = vertex_normals(V, s.T)
        p = s.point(self.tri, self.bary, V)
        return p + self.h[:, None] * s.normal(self.tri, self.bary, N)


def strip_faces(n, closed_ends=(False, False)):
    """Quads of a strip of n cross-sections of 2 vertices each (vertex 2k = lower/inner edge, 2k+1 = upper/outer edge),
    CCW seen from the front when the strip runs left to right with the upper edge on top. An end whose two vertices
    coincide (a pointed end) becomes a triangle: pass closed_ends=(start_pointed, end_pointed)."""
    F = []
    for k in range(n - 1):
        a0, a1, b0, b1 = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        F.append((a0, b0, b1, a1))
    return F


_QUAD_UV = np.array([(u, v) for u in (0.2, 0.5, 0.8) for v in (0.15, 0.38, 0.62, 0.85)])
_TRI_BC = np.array([(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5),
                    (0.5, 0.0, 0.5)])


def _deficits(s, P, faces, clearance, along_normal=True):
    """Per vertex of the faces (strip quads / triangles over the positions P), how far it must rise for them to lie
    `clearance` in front of the skin: measured at sample points of each face (along the skin's normal, or along -y) and at
    the skin's vertices under it (the piecewise-linear skin peaks there)."""
    pts, owners = [], []
    for f in faces:
        if len(f) == 4:
            A, B, C, D = (P[i] for i in f)           # strip order: lo_k, lo_k+1, hi_k+1, hi_k
            for u, v in _QUAD_UV:
                pts.append((1 - v) * ((1 - u) * A + u * B) + v * ((1 - u) * D + u * C))
                owners.append(f)
        else:
            for bc in _TRI_BC:
                pts.append(sum(w * P[i] for w, i in zip(bc, f[:3])))
                owners.append(f)
    pts = np.asarray(pts)
    tri, bary = s.bind(pts[:, 0], pts[:, 2], strict=False)
    ok = tri >= 0
    add = np.zeros(len(P))
    if ok.any():
        sk = s.point(tri[ok], bary[ok])
        if along_normal:
            height = ((pts[ok] - sk) * s.normal(tri[ok], bary[ok])).sum(1)
        else:
            height = sk[:, 1] - pts[ok, 1]
        deficit = np.maximum(clearance - height, 0.0)
        for d, i in zip(deficit, np.nonzero(ok)[0]):
            if d > 0:
                for v in owners[i]:
                    add[v] = max(add[v], d)
    sv = s.V[s.vert_ids]
    lo, hi = P[:, [0, 2]].min(0), P[:, [0, 2]].max(0)
    sv = sv[(sv[:, 0] >= lo[0]) & (sv[:, 0] <= hi[0]) & (sv[:, 2] >= lo[1]) & (sv[:, 2] <= hi[1])]   # the faces' extent
    for f in faces:
        tris = [(f[0], f[1], f[2]), (f[0], f[2], f[3])] if len(f) == 4 else [tuple(f[:3])]
        for t in tris:
            A, B, C = P[t[0]], P[t[1]], P[t[2]]
            d = (B[2] - C[2]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[2] - C[2])
            if abs(d) < 1e-14:
                continue
            w0 = ((B[2] - C[2]) * (sv[:, 0] - C[0]) + (C[0] - B[0]) * (sv[:, 2] - C[2])) / d
            w1 = ((C[2] - A[2]) * (sv[:, 0] - C[0]) + (A[0] - C[0]) * (sv[:, 2] - C[2])) / d
            w2 = 1.0 - w0 - w1
            m = (w0 > 1e-6) & (w1 > 1e-6) & (w2 > 1e-6)
            if not m.any():
                continue
            y_face = w0[m] * A[1] + w1[m] * B[1] + w2[m] * C[1]
            over = y_face - (sv[m, 1] - clearance)             # > 0: the face is behind the skin vertex (seen from -y)
            over = np.maximum(over, 0.0)
            if over.max() > 0:
                for v in t:
                    add[v] = max(add[v], float(over.max()) / 0.6)
    return add


def clear_skin(dec, faces, clearance=1.6e-4, iters=12, max_lift=0.0015):
    """Raise the decal's vertices until the faces lie at least `clearance` above the skin: a strip is flat between its vertices,
    so over a concave or ridged stretch of skin (the socket around an eye) its quads would otherwise cut below the surface
    and the skin would show through. Each face is sampled on a small grid (quads) or at 7 barycentric points (triangles).
    No vertex rises more than `max_lift` (a lift moves it along the skin's normal, sideways too where the skin curves into a
    hole, which can find it new skin to clear without end)."""
    faces = [list(f) for f in faces]
    h_top = dec.h + max_lift
    for _ in range(iters):
        add = _deficits(dec.surf, dec.positions(), faces, clearance)
        if add.max() < 1e-5:
            return
        dec.h = np.minimum(dec.h + 1.1 * add, h_top)


def clear_points(s, P, faces, clearance=1.6e-4, iters=12, max_lift=0.0015):
    """clear_skin for free positions (a morph's target, (n, 3)) over the skin surface `s`: the vertices rise toward the viewer
    (-y), at most `max_lift`, until the faces lie `clearance` in front of the skin where they are over it. Returns the
    raised positions."""
    faces = [list(f) for f in faces]
    P = np.array(P, float)
    y_top = P[:, 1] - max_lift
    for _ in range(iters):
        add = _deficits(s, P, faces, clearance, along_normal=False)
        if add.max() < 1e-5:
            break
        P[:, 1] = np.maximum(P[:, 1] - 1.1 * add, y_top)
    return P
