"""Geometry the van's parts share (pure numpy on core.shell, no bpy): rounded slabs with grid lines where materials
change, sections pulled straight (`prism`) or round a plan outline (`plan_sweep`: rounded corners, mitred where a
section reaches deeper than the corner's radius), ear-clipped end caps, rivet heads and surface lamps.

Every mesh comes back closed with its faces outward. Sections are closed polygons of (u, v) points; `rounded` fillets
their corners, so an extrusion has the rolled edges a real one has (docs/modelling.md: no visible cuboid, no hard edge)."""
import math

import numpy as np

from ....core import shell as S


def pos(m):
    """The mesh with its faces outward (a sweep's winding follows its section's)."""
    return m if m.volume() > 0 else m.flip()


def rounded(points, radii, n=3):
    """A closed polygon (m, 2) with the corner at every point rounded by its radius (0 keeps it sharp), n segments each."""
    return S.fillet(np.asarray(points, float), radii, n=n, closed=True)


def arc_pts(c, r, a0, a1, n):
    """n + 1 points on the circle about c (2D) from angle a0 to a1 (degrees), both ends included."""
    return S.arc(c, r, math.radians(a0), math.radians(a1), n)


# ===================================================================================================================
# caps and skins
# ===================================================================================================================
def _clean(P, tol=1e-9):
    """The closed polygon without repeated consecutive points."""
    keep = np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1) > tol
    return P[keep], np.nonzero(keep)[0]


def ear_clip(P):
    """Triangles (k, 3) covering the simple polygon P (m, 2) of either winding, as indices into P, wound
    counter-clockwise. Collinear points are clipped as slivers when no proper ear is left."""
    P = np.asarray(P, float)
    idx = list(range(len(P)))
    if S.polygon_area(P) < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    tris = []
    while len(idx) > 3:
        n = len(idx)
        best = None
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = P[j]
                if cross(a, b, p) >= -1e-14 and cross(b, c, p) >= -1e-14 and cross(c, a, p) >= -1e-14:
                    inside = True
                    break
            if not inside:
                best = k
                break
        if best is None:                                  # only collinear or reflex corners left: clip the flattest
            best = int(np.argmin([abs(cross(P[idx[k - 1]], P[idx[k]], P[idx[(k + 1) % n]])) for k in range(n)]))
        tris.append((idx[best - 1], idx[best], idx[(best + 1) % n]))
        idx.pop(best)
    tris.append(tuple(idx))
    return np.array(tris, np.int64)


def ring_skin(rings, loop=False, caps=True, mat=0):
    """A closed mesh through rings (n, m, 3) of matching points (each ring a closed planar polygon): quads from ring to
    ring, and unless `loop` (the last ring joins the first) the two end rings closed by ear-clipped caps."""
    G = np.asarray(rings, float)
    n, m = G.shape[:2]
    V = G.reshape(-1, 3)
    idx = np.arange(n * m).reshape(n, m)
    j, j2 = np.arange(m), (np.arange(m) + 1) % m
    i = np.arange(n if loop else n - 1)
    i2 = (i + 1) % n
    Q = np.stack([idx[i][:, j], idx[i][:, j2], idx[i2][:, j2], idx[i2][:, j]], -1).reshape(-1, 4)
    T = []
    if caps and not loop:
        for end in (0, n - 1):
            R = G[end]
            c = R.mean(0)
            _, _, vt = np.linalg.svd(R - c)
            e1, e2 = vt[0], vt[1]
            P2 = np.stack([(R - c) @ e1, (R - c) @ e2], 1)
            t = ear_clip(P2)
            T.append(idx[end][t])
    m_out = S.Mesh(V, Q, np.concatenate(T) if T else None, mat=mat)
    # make every face agree with the side quads: flip the caps whose winding disagrees, then the whole mesh outward
    if T:
        m_out = _orient_caps(m_out, len(Q))
    return pos(m_out.weld(1e-9))


def _orient_caps(mesh, nq):
    """Flip cap triangles so each shares its edges with the side quads in opposite directions (a consistent winding)."""
    Q, T = mesh.Q, mesh.T.copy()
    fwd = set()
    for q in Q[:nq]:
        for a, b in zip(q, np.roll(q, -1)):
            fwd.add((int(a), int(b)))
    # group the triangles by cap (the two caps share no vertices): test each triangle against the side edges it touches
    for k, t in enumerate(T):
        edges = [(int(a), int(b)) for a, b in zip(t, np.roll(t, -1))]
        if any(e in fwd for e in edges):
            T[k] = t[::-1]
    # interior cap edges: propagate the winding from the triangles that touch the rim
    for _ in range(len(T)):
        changed = False
        directed = {}
        for k, t in enumerate(T):
            for a, b in zip(t, np.roll(t, -1)):
                directed.setdefault((int(a), int(b)), []).append(k)
        for (a, b), ks in directed.items():
            if len(ks) > 1:                                 # an edge used twice in one direction: flip a triangle
                k = ks[-1]
                T[k] = T[k][::-1]
                changed = True
                break
        if not changed:
            break
    return S.Mesh(mesh.V, mesh.Q, T, mesh.Qm, mesh.Tm, mesh.UV)


def prism(outline, axis, a0, a1, mat=0):
    """The closed polygon `outline` (m, 2) pulled straight along `axis` ("x", "y" or "z") from a0 to a1. The outline's
    coordinates are the two other axes in order: (y, z) for "x", (x, z) for "y", (x, y) for "z"."""
    P, _ = _clean(np.asarray(outline, float))
    k = "xyz".index(axis)
    others = [a for a in range(3) if a != k]
    rings = np.zeros((2, len(P), 3))
    for r, a in enumerate((a0, a1)):
        rings[r][:, k] = a
        rings[r][:, others[0]] = P[:, 0]
        rings[r][:, others[1]] = P[:, 1]
    return ring_skin(rings, mat=mat)


def sweep_rings(path, section, up=(0.0, 0.0, 1.0), scale=None):
    """Rings (n, m, 3) of a section (m, 2) of (u, v) swept along `path` (n, 3) with core.shell's frames (v toward `up`),
    optionally scaled per point ((n,) or (n, 2))."""
    path = np.asarray(path, float)
    sec = np.asarray(section, float)
    T, U, B = S._frames(path, up, False)
    n = len(path)
    sc = np.ones((n, 2)) if scale is None else np.broadcast_to(
        np.asarray(scale, float).reshape(-1, 1) if np.ndim(scale) == 1 else np.asarray(scale, float), (n, 2))
    return (path[:, None, :] + sec[None, :, 0:1] * sc[:, None, 0:1] * U[:, None, :]
            + sec[None, :, 1:2] * sc[:, None, 1:2] * B[:, None, :])


def bar(path, section, up=(0.0, 0.0, 1.0), ends=0.0, mat=0):
    """A section swept along a path with ear-clipped ends; `ends` > 0 rolls both ends over that length (the section
    shrinks along a quarter round: a rounded end instead of a sawn one)."""
    path = np.asarray(path, float)
    if ends > 0:
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total = seg.sum()
        t = np.linspace(0.0, 0.5 * math.pi, 4)[1:]
        d0 = path[1] - path[0]
        d1 = path[-1] - path[-2]
        d0, d1 = d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)
        sec = np.asarray(section, float)
        size = 0.5 * (sec.max(0) - sec.min(0))
        k = np.clip(ends / np.maximum(size, 1e-9), 0.0, 1.0)
        head = [path[0] + d0 * ends * (1.0 - math.sin(a)) for a in t[::-1]]
        tail = [path[-1] - d1 * ends * (1.0 - math.sin(a)) for a in t]
        inner = list(path)
        inner[0] = path[0] + d0 * ends
        inner[-1] = path[-1] - d1 * ends
        P = np.array(head + inner + tail)
        sc = np.ones((len(P), 2))
        for r, a in enumerate(t[::-1]):                    # the head rings, outermost first
            sc[r] = 1.0 - k * (1.0 - math.cos(a))
        for r, a in enumerate(t):
            sc[len(P) - 3 + r] = 1.0 - k * (1.0 - math.cos(a))
        del total
        return ring_skin(sweep_rings(P, section, up, sc), mat=mat)
    return ring_skin(sweep_rings(path, section, up), mat=mat)


def tube(path, radius, sides=10, up=(0.0, 0.0, 1.0), ends=0.0, aspect=1.0, mat=0):
    """A round (or elliptical) bar along a path."""
    t = 2.0 * math.pi * (np.arange(sides) + 0.5) / sides
    return bar(path, np.stack([radius * np.cos(t), radius * aspect * np.sin(t)], 1), up, ends, mat)


def superellipse(w, h, p, n=24):
    """A rounded-square section (counter-clockwise), the toolkit's superellipse started half a step off the axes."""
    t = 2.0 * math.pi * (np.arange(n) + 0.5) / n
    c, s = np.cos(t), np.sin(t)
    return np.stack([0.5 * w * np.sign(c) * np.abs(c) ** (2.0 / p), 0.5 * h * np.sign(s) * np.abs(s) ** (2.0 / p)], 1)


# ===================================================================================================================
# slabs
# ===================================================================================================================
def slab(lo, hi, r, lines=((), (), ()), k=1, mat=0):
    """A box from `lo` to `hi` with every edge rounded by r (2k segments per quarter round: a plate's rolled rims), its
    flat faces cut along the planes `lines` = (x values, y values, z values) where a material is to change. Closed,
    outward; the flat faces are exactly flat and exactly at lo / hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    r = float(min(r, h.min() * 0.999))
    u = r * np.tan(np.arange(k + 1) * (math.pi / 4) / k)
    coords = []
    for a in range(3):
        edge = h[a] - r + u
        inner = [float(v) - c[a] for v in lines[a] if abs(float(v) - c[a]) < h[a] - r - 1e-6]
        coords.append(np.unique(np.round(np.concatenate([-edge, inner, edge]), 12)))
    flat = h - r
    Vs, Qs, off = [], [], 0
    for a in range(3):
        for sign in (-1, 1):
            b, cc = (a + 1) % 3, (a + 2) % 3
            U, W = np.meshgrid(coords[b], coords[cc], indexing="ij")
            q = np.zeros(U.shape + (3,))
            q[..., a] = sign * h[a]
            q[..., b], q[..., cc] = U, W
            ctr = np.clip(q, -flat, flat)
            d = q - ctr
            p = ctr + r * d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
            nu, nw = U.shape
            idx = np.arange(nu * nw).reshape(nu, nw) + off
            quad = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
            if sign < 0:
                quad = quad[:, ::-1]
            Vs.append(p.reshape(-1, 3))
            Qs.append(quad)
            off += nu * nw
    return S.Mesh(np.concatenate(Vs) + c, np.concatenate(Qs), mat=mat).weld(1e-9)


def assign_face(mesh, mat, axis, sign, at, box, tol=1e-6):
    """Material `mat` for the flat faces of `mesh` that look along `sign` * `axis` (0, 1, 2), lie in the plane
    axis = `at` and whose centres fall inside `box` = ((u0, u1), (v0, v1)) over the two other axes in order."""
    a, b = [k for k in range(3) if k != axis]

    def fn(c, n):
        return ((n[:, axis] * sign > 0.999) & (np.abs(c[:, axis] - at) < tol) & (c[:, a] > box[0][0]) & (c[:, a] < box[0][1])
                & (c[:, b] > box[1][0]) & (c[:, b] < box[1][1]))
    return mesh.assign(mat, fn)


# ===================================================================================================================
# a section round a plan outline
# ===================================================================================================================
def plan_path(corners, radii, closed, n=6):
    """Samples of a plan outline (the outer face line of a frame member) whose inside lies on its right: per sample the
    corner it belongs to (or -1 on an open end), its outward direction and its corner centre. Used by `plan_sweep`."""
    P = np.asarray(corners, float)
    N = len(P)
    out = []
    for i in range(N):
        if not closed and i in (0, N - 1):
            d = P[1] - P[0] if i == 0 else P[-1] - P[-2]
            d = d / np.linalg.norm(d)
            nin = np.array([d[1], -d[0]])
            out.append(("end", P[i], nin, None, 0.0))
            continue
        d0 = P[i] - P[i - 1]
        d1 = P[(i + 1) % N] - P[i]
        d0, d1 = d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)
        n0, n1 = np.array([d0[1], -d0[0]]), np.array([d1[1], -d1[0]])
        mit = (n0 + n1) / (1.0 + n0 @ n1)
        r = float(radii[i])
        C = P[i] + r * mit
        a0, a1 = math.atan2(-n0[1], -n0[0]), math.atan2(-n1[1], -n1[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            a = a0 + da * k / n
            out.append(("arc", C, np.array([math.cos(a), math.sin(a)]), mit, r))
    return out


def plan_sweep(corners, radii, section, closed=False, n=6, mat=0):
    """A frame member: `section` (m, 2) of (s, z) points, s the depth inward from the plan outline `corners` (the
    member's outer face line, its inside on the right, radii per corner), z the height, swept round the outline. On a
    rounded corner a point of depth s follows the radius r - s about the corner's centre; one deeper than the radius
    meets its neighbour in a mitre (the offset of the outline), so nothing folds. Open outlines get end caps."""
    sec = np.asarray(section, float)
    rings = []
    for kind, p, d, mit, r in plan_path(corners, radii, closed, n):
        if kind == "end":
            xy = p[None, :] + sec[:, :1] * d[None, :]
        else:
            s = sec[:, :1]
            xy = np.where(s < r, p[None, :] + (r - s) * d[None, :], p[None, :] + (s - r) * mit[None, :])
        rings.append(np.column_stack([xy, sec[:, 1]]))
    return ring_skin(np.array(rings), loop=closed, caps=not closed, mat=mat)


# ===================================================================================================================
# small hardware
# ===================================================================================================================
def _frame(normal, up):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    u = np.asarray(up, float)
    u = u - n * (u @ n)
    u = u / np.linalg.norm(u)
    return np.stack([np.cross(u, n), u, n], axis=1)


def rivets(points, normal, up=(0.0, 0.0, 1.0), r=0.0055, h=0.0035, sink=0.0008, sides=6, mat=0):
    """Rivet heads: a low cone (`sides` facets, flat underside; smooth normals shade it as a dome) at every point (k, 3),
    standing `h` out of the surface whose outward normal is `normal`, its foot sunk `sink` into it. One mesh, built as
    whole arrays: 2 * sides - 2 triangles a rivet."""
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) == 0:
        return S.Mesh()
    t = 2.0 * math.pi * np.arange(sides) / sides
    ring = np.stack([r * np.cos(t), r * np.sin(t), np.full(sides, -sink)], 1)
    tpl = np.concatenate([ring, [[0.0, 0.0, h]]]) @ _frame(normal, up).T
    j, j2 = np.arange(sides), (np.arange(sides) + 1) % sides
    T = np.concatenate([np.stack([j, j2, np.full(sides, sides)], 1),
                        np.array([[0, k + 2, k + 1] for k in range(sides - 2)])])
    k = len(tpl)
    V = (P[:, None, :] + tpl[None]).reshape(-1, 3)
    offs = (np.arange(len(P)) * k)[:, None, None]
    return pos(S.Mesh(V, None, (T[None] + offs).reshape(-1, 3), mat=mat))


def lamp(w, h, depth, lens_mat, base_mat, origin, normal, up=(0.0, 0.0, 1.0), round_=False, sink=0.003):
    """A marker or tail lamp on a flat surface: a dark rubber grommet ring with a domed lens in it (`round_` a round
    lamp of diameter w), its back sunk `sink` into the surface at `origin`, looking along `normal`."""
    if round_:
        outer, inner = S.ellipse(0.5 * w + 0.012, 0.5 * w + 0.012, 28), S.ellipse(0.5 * w, 0.5 * w, 28)
        lens_out = S.ellipse(0.5 * w - 0.001, 0.5 * w - 0.001, 28)
    else:
        rr = 0.45 * min(w, h)
        outer = S.rrect(w + 0.016, h + 0.016, rr + 0.008, 4)
        inner = S.rrect(w, h, rr, 4)
        lens_out = S.rrect(w - 0.002, h - 0.002, rr - 0.001, 4)
    base = S.rounded_frame(outer, inner, depth, r_out=min(0.004, 0.4 * depth), r_in=0.002, n=2, floor=0.6 * depth,
                           mat=base_mat)
    lens = S.rounded_panel(lens_out, 0.55 * depth, r_edge=min(0.003, 0.25 * depth), n=2,
                           dome=0.18 * min(w, h if not round_ else w), mat=lens_mat).moved((0.0, 0.0, 0.42 * depth))
    m = S.merge([base, lens]).moved((0.0, 0.0, -sink))
    return S.place(m, origin, normal, up)
