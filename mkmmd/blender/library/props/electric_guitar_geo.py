"""2D and slab geometry for the electric guitar (pure numpy, no bpy: importable from tests).

The body, pickguard and headstock are plates with a concave, many-cornered outline (horns, bays, a waist) that `core.shell`'s
`rounded_panel` cannot close (its cap shrinks the outline toward its centre, which folds a non-convex shape). Here the outline
is a closed spline, the rim is a stack of offset rings (a rolled edge), and the flat faces are filled by a Delaunay
triangulation of the rim ring plus a lattice of points inside it, so the faces follow any smooth height function (the forearm
and belly contours of the body) with triangles of one size.

    spline(points, corners, spacing)       closed Catmull-Rom curve through control points (some of them sharp), resampled
    slab(outline, top, bottom, r, ...)     a closed solid plate: rolled rim, front face y = top(x, z), back face y = bottom(x, z)

Plates lie in the x-z plane: the front faces -y, the thickness runs along +y (the guitar frame, see electric_guitar_layout).
"""
import math

import numpy as np

from ....core import shell as S

EPS = 1e-12


# ===================================================================================================================
# curves
# ===================================================================================================================
def poly_area(P):
    return S.polygon_area(P)


def ccw(P):
    """The closed polygon P (n, 2) counter-clockwise."""
    P = np.asarray(P, float)
    return P if poly_area(P) > 0 else P[::-1].copy()


def _cr_segment(P0, P1, P2, P3, t, alpha=0.5):
    """Centripetal Catmull-Rom between P1 and P2 at parameters t in 0..1 (m,) -> (m, 2) (Barry-Goldman)."""
    t0 = 0.0
    t1 = t0 + max(np.linalg.norm(P1 - P0), EPS) ** alpha
    t2 = t1 + max(np.linalg.norm(P2 - P1), EPS) ** alpha
    t3 = t2 + max(np.linalg.norm(P3 - P2), EPS) ** alpha
    u = (t1 + (t2 - t1) * np.asarray(t, float))[:, None]
    A1 = (t1 - u) / (t1 - t0) * P0 + (u - t0) / (t1 - t0) * P1
    A2 = (t2 - u) / (t2 - t1) * P1 + (u - t1) / (t2 - t1) * P2
    A3 = (t3 - u) / (t3 - t2) * P2 + (u - t2) / (t3 - t2) * P3
    B1 = (t2 - u) / (t2 - t0) * A1 + (u - t0) / (t2 - t0) * A2
    B2 = (t3 - u) / (t3 - t1) * A2 + (u - t1) / (t3 - t1) * A3
    return (t2 - u) / (t2 - t1) * B1 + (u - t1) / (t2 - t1) * B2


def _curve(Q, closed, per=24):
    """Dense points (k, 2) of the Catmull-Rom curve through Q (n, 2); closed: periodic (the last point is not repeated),
    else open with reflected end points (the end points are included)."""
    Q = np.asarray(Q, float)
    n = len(Q)
    if closed:
        ext = np.vstack([Q[-1:], Q, Q[:2]])
        segs = range(n)
    else:
        ext = np.vstack([2 * Q[0] - Q[1], Q, 2 * Q[-1] - Q[-2]])
        segs = range(n - 1)
    t = np.linspace(0.0, 1.0, per, endpoint=False)
    out = [_cr_segment(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], t) for i in segs]
    if not closed:
        out.append(Q[-1:])
    return np.concatenate(out)


def _resample(Pd, step, closed):
    """Points spaced evenly by arc length along the dense polyline Pd (the first point is kept; open pieces keep the last)."""
    Q = np.vstack([Pd, Pd[:1]]) if closed else Pd
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    n = max(int(round(s[-1] / step)), 2)
    q = np.linspace(0.0, s[-1], n + 1)[:-1] if closed else np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(q, s, Q[:, 0]), np.interp(q, s, Q[:, 1])], 1)


def spline(points, corners=(), spacing=0.004):
    """A closed smooth curve through the control `points` (n, 2), sampled every `spacing` metres along its length (a closed
    polygon without the first point repeated). `corners` are indices of points where the curve turns sharply (the curve is cut
    there into pieces that each end at the corner): the corners themselves are samples."""
    P = np.asarray(points, float)
    n = len(P)
    corners = sorted(set(int(c) % n for c in corners))
    if not corners:
        return _resample(_curve(P, True), spacing, True)
    pieces = []
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        idx = [i % n for i in range(a, b + 1)]
        pieces.append(_resample(_curve(P[idx], False), spacing, False)[:-1])
    return np.concatenate(pieces)


# ===================================================================================================================
# polygon queries
# ===================================================================================================================
def inside(Q, poly):
    """Even-odd point-in-polygon: Q (q, 2), poly (n, 2) -> bool (q,)."""
    Q = np.asarray(Q, float)
    P = np.asarray(poly, float)
    A, B = P, np.roll(P, -1, axis=0)
    out = np.zeros(len(Q), bool)
    for lo in range(0, len(Q), 512):
        q = Q[lo:lo + 512]
        x, y = q[:, 0:1], q[:, 1:2]
        cond = (A[None, :, 1] > y) != (B[None, :, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = A[None, :, 0] + (y - A[None, :, 1]) * (B[None, :, 0] - A[None, :, 0]) / (B[None, :, 1] - A[None, :, 1])
        out[lo:lo + 512] = (cond & (x < xi)).sum(1) % 2 == 1
    return out


def distance(Q, poly):
    """Distance from points Q (q, 2) to the closed polygon's edges."""
    Q = np.asarray(Q, float)
    P = np.asarray(poly, float)
    A, B = P, np.roll(P, -1, axis=0)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), EPS)
    out = np.empty(len(Q))
    for lo in range(0, len(Q), 512):
        q = Q[lo:lo + 512]
        t = np.clip(((q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        d = q[:, None, :] - (A[None] + t[..., None] * AB[None])
        out[lo:lo + 512] = np.sqrt((d * d).sum(2).min(1))
    return out


def edge_normals(P):
    """Outward unit normals of the edges of a CCW closed polygon (n, 2): edge i runs from P[i] to P[i + 1]."""
    e = np.roll(P, -1, axis=0) - P
    n = np.stack([e[:, 1], -e[:, 0]], 1)
    return n / np.maximum(np.linalg.norm(n, axis=1), EPS)[:, None]


def inset(P, d):
    """The CCW closed polygon P moved `d` inward (negative: outward) along the mitre of every corner."""
    P = ccw(P)
    return P + d * S.mitre_offsets(P)


# ===================================================================================================================
# Delaunay triangulation (Bowyer-Watson, numpy)
# ===================================================================================================================
def _circles(pts, T):
    a, b, c = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    d = np.where(np.abs(d) < EPS, EPS, d)
    a2, b2, c2 = (a * a).sum(1), (b * b).sum(1), (c * c).sum(1)
    ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
    uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    U = np.stack([ux, uy], 1)
    return U, ((a - U) ** 2).sum(1)


def delaunay(P):
    """Delaunay triangles (k, 3) of the points P (n, 2), counter-clockwise (Bowyer-Watson: each point cuts out the triangles
    whose circumcircle holds it and fans the hole to it). Points are inserted in the given order."""
    P = np.asarray(P, float)
    n = len(P)
    c = 0.5 * (P.min(0) + P.max(0))
    r = max(float(np.ptp(P, axis=0).max()), 1e-9) * 1000.0
    pts = np.vstack([P, [c[0] - r, c[1] - 0.7 * r], [c[0] + r, c[1] - 0.7 * r], [c[0], c[1] + r]])
    T = np.array([[n, n + 1, n + 2]], np.int64)
    CC, R2 = _circles(pts, T)
    for i in range(n):
        p = pts[i]
        d = CC - p
        bad = (d * d).sum(1) < R2 * (1.0 - 1e-10)
        tb = T[bad]
        E = np.concatenate([tb[:, [0, 1]], tb[:, [1, 2]], tb[:, [2, 0]]])
        key = np.minimum(E[:, 0], E[:, 1]) * (n + 3) + np.maximum(E[:, 0], E[:, 1])
        _, first, count = np.unique(key, return_index=True, return_counts=True)
        edge = E[first[count == 1]]
        new = np.concatenate([edge, np.full((len(edge), 1), i, np.int64)], 1)
        nc, nr = _circles(pts, new)
        keep = ~bad
        T = np.concatenate([T[keep], new])
        CC = np.concatenate([CC[keep], nc])
        R2 = np.concatenate([R2[keep], nr])
    return T[(T < n).all(1)]


def lattice(poly, spacing, margin):
    """A triangular lattice of points inside the polygon, at least `margin` from its edges (a deterministic hair of jitter
    keeps four points from being cocircular)."""
    P = np.asarray(poly, float)
    lo, hi = P.min(0), P.max(0)
    h = spacing * math.sqrt(3.0) / 2.0
    rows = np.arange(lo[1], hi[1] + h, h)
    pts = []
    for k, z in enumerate(rows):
        xs = np.arange(lo[0] + (0.5 * spacing if k % 2 else 0.0), hi[0] + spacing, spacing)
        pts.append(np.stack([xs, np.full(len(xs), z)], 1))
    Q = np.concatenate(pts)
    Q = Q + (np.random.default_rng(7).random(Q.shape) - 0.5) * spacing * 1e-3
    Q = Q[inside(Q, P)]
    return Q[distance(Q, P) >= margin]


def fill(ring, spacing=0.009, margin=0.7):
    """Triangles filling the closed CCW polygon `ring` (m, 2): (V (n, 2), T (k, 3)). The first m points are the ring itself
    (every ring edge is a triangle edge), the others a lattice of `spacing` inside it, `margin * spacing` clear of the ring.
    The triangles are counter-clockwise."""
    ring = np.asarray(ring, float)
    Q = lattice(ring, spacing, margin * spacing)
    V = np.vstack([ring, Q])
    T = delaunay(V)
    return V, T[inside(V[T].mean(1), ring)]


# ===================================================================================================================
# plates
# ===================================================================================================================
def _orient(V, T, direction):
    """The triangles T turned so that their normals point along `direction` (a unit vector)."""
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) @ np.asarray(direction, float)
    out = T.copy()
    out[n < 0] = out[n < 0][:, ::-1]
    return out


def fillet_profile(r, n):
    """(d inward, drop) of a quarter round of radius r in n steps from the flat face (d = r, drop 0) to the wall (d = 0,
    drop r): d = r (1 - cos phi), drop = r (1 - sin phi) for phi from 90 to 0 degrees."""
    phi = np.linspace(0.5 * math.pi, 0.0, n + 1)
    return r * (1.0 - np.cos(phi)), r * (1.0 - np.sin(phi))


def slab(outline, top, bottom, r=0.004, n=4, wall=1, spacing=0.009, mat=0, mat_wall=None, mat_back=None):
    """A closed plate in the x-z plane. `outline` the CCW polygon (m, 2) of (x, z) at the wall, `top(x, z)` / `bottom(x, z)` the
    y of the front (-y facing) and back (+y facing) flat faces (numpy functions of arrays: contours are just functions), `r`
    the radius of the rolled front and back edges (`n` segments each; n = 1 is a chamfer), `wall` the quad rows on the straight
    wall between them. The flat faces are triangulated (`fill`), the rim is quads. Materials: front face and its fillet `mat`,
    the wall `mat_wall` (one value, or one per wall row from the front: plies), the back face and its fillet `mat_back`
    (default: all `mat`). Closed, outward normals."""
    mat_wall = mat if mat_wall is None else mat_wall
    wall_mats = list(mat_wall) if hasattr(mat_wall, "__len__") else [mat_wall] * wall
    mat_back = mat if mat_back is None else mat_back
    W = ccw(outline)
    m = len(W)
    M = S.mitre_offsets(W)
    rings = []
    d_t, h_t = fillet_profile(r, n)
    for d, h in zip(d_t, h_t):                                    # front: the flat face edge ... the wall
        p = W + d * M
        rings.append(np.column_stack([p[:, 0], top(p[:, 0], p[:, 1]) + h, p[:, 1]]))
    for k in range(1, wall):                                      # the wall between the two fillets
        t = k / wall
        rings.append(np.column_stack([W[:, 0], (1 - t) * (top(W[:, 0], W[:, 1]) + r) + t * (bottom(W[:, 0], W[:, 1]) - r), W[:, 1]]))
    for d, h in zip(d_t[::-1], h_t[::-1]):                        # back: the wall ... the flat face edge
        p = W + d * M
        rings.append(np.column_stack([p[:, 0], bottom(p[:, 0], p[:, 1]) - h, p[:, 1]]))
    R = np.stack(rings)                                           # (K, m, 3), front ring first
    K = len(R)
    idx = np.arange(K * m).reshape(K, m)
    j = np.arange(m)
    j2 = (j + 1) % m
    Q = np.concatenate([np.stack([idx[k][j], idx[k + 1][j], idx[k + 1][j2], idx[k][j2]], 1) for k in range(K - 1)])
    qm = np.concatenate([np.full(m, mat if k < n else (wall_mats[k - n] if k < n + wall else mat_back), np.int32)
                         for k in range(K - 1)])
    V2, T2 = fill(W + r * M, spacing)                             # one triangulation of the inner ring, used front and back
    ni = len(V2) - m
    inner = V2[m:]
    Vtop = np.column_stack([inner[:, 0], top(inner[:, 0], inner[:, 1]), inner[:, 1]])
    Vbot = np.column_stack([inner[:, 0], bottom(inner[:, 0], inner[:, 1]), inner[:, 1]])
    base = K * m
    V = np.concatenate([R.reshape(-1, 3), Vtop, Vbot])
    front_ids = np.concatenate([idx[0], base + np.arange(ni)])
    back_ids = np.concatenate([idx[K - 1], base + ni + np.arange(ni)])
    Tf = _orient(V, front_ids[T2], (0.0, -1.0, 0.0))
    Tb = _orient(V, back_ids[T2], (0.0, 1.0, 0.0))
    T = np.concatenate([Tf, Tb])
    tm = np.concatenate([np.full(len(Tf), mat, np.int32), np.full(len(Tb), mat_back, np.int32)])
    mesh = S.Mesh(V, Q, T, Qm=qm, Tm=tm)
    return mesh if mesh.volume() > 0 else mesh.flip()
