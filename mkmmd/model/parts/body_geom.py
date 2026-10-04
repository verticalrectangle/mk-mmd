"""Geometry kit of the body part: monotone splines, limb paths, ring sweeps with caps, quad/UV bookkeeping, mirroring.
numpy only (it also runs under Blender's Python 3.11 / numpy 1.24).

Conventions (model space: metres, Z up, the character faces -Y, her left is +X)
- A *tube* is a sweep of closed rings along a `Path`; ring point k of M sits at the angle 2*pi*k/M measured in the
  section plane from the section's "front" axis `ef` towards `eb = t x ef` (t = the path tangent, so (ef, eb, t) is a
  right-handed frame). The quad (i,k) (i,k+1) (i+1,k+1) (i+1,k) then faces OUTWARD for rings listed along +t.
- Every shell reports `s`, the arclength of the ring it sits on (caps extend beyond [0, L]), which the weight code uses.
- UVs are per face corner, u around the ring (0..1), v along the tube (0..1)."""
import numpy as np

TAU = 2.0 * np.pi


# ---------------------------------------------------------------- small maths
def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def rot_axis(axis, angle):
    """3x3 rotation about `axis` by `angle` (Rodrigues)."""
    a = unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def pchip(xs, ys):
    """Monotone cubic Hermite interpolant (Fritsch-Carlson, as scipy's PchipInterpolator): returns f(x) clamped to the
    table's range; `ys` is (n,) or (n, k)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    if n < 2:
        raise ValueError("pchip needs two knots")
    shp = (-1,) + (1,) * (ys.ndim - 1)
    h = np.diff(xs).reshape(shp)
    d = np.diff(ys, axis=0) / h
    m = np.zeros_like(ys)
    if n == 2:
        m[:] = d[0]
    else:
        for k in range(1, n - 1):
            same = d[k - 1] * d[k] > 0
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            den = np.where(same, w1 / np.where(same, d[k - 1], 1.0) + w2 / np.where(same, d[k], 1.0), 1.0)
            m[k] = np.where(same, (w1 + w2) / den, 0.0)

        def end(h0, h1, d0, d1):
            e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            e = np.where(np.sign(e) != np.sign(d0), 0.0, e)
            return np.where((np.sign(d0) != np.sign(d1)) & (np.abs(e) > 3 * np.abs(d0)), 3 * d0, e)

        m[0] = end(h[0], h[1], d[0], d[1])
        m[-1] = end(h[-1], h[-2], d[-1], d[-2])

    def f(x):
        x = np.clip(np.asarray(x, float), xs[0], xs[-1])
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
        hh = (xs[i + 1] - xs[i])
        t = (x - xs[i]) / hh
        t2, t3 = t * t, t * t * t
        h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
        sh = (-1,) + (1,) * (ys.ndim - 1)
        hh = hh.reshape(sh) if ys.ndim > 1 else hh
        h00, h10, h01, h11 = [a.reshape(sh) if ys.ndim > 1 else a for a in (h00, h10, h01, h11)]
        return h00 * ys[i] + h10 * hh * m[i] + h01 * ys[i + 1] + h11 * hh * m[i + 1]

    return f


class Table:
    """Named columns over a common abscissa, interpolated with pchip: `Table(xs, a=[..], b=[..])(x) -> dict`."""

    def __init__(self, xs, **cols):
        self.names = list(cols)
        self.f = pchip(xs, np.stack([np.asarray(cols[k], float) for k in self.names], -1))

    def __call__(self, x):
        v = self.f(x)
        return {k: v[..., i] for i, k in enumerate(self.names)}


def superellipse(theta, n):
    """(x, y) on the unit superellipse |x|^n + |y|^n = 1 at the polar parameter theta (n=2: a circle)."""
    c, s = np.cos(theta), np.sin(theta)
    e = 2.0 / np.asarray(n, float)
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


# ---------------------------------------------------------------- paths
class Path:
    """A polyline through joint points with a smoothed tangent: the tangent turns from one segment's direction to the
    next within +-`blend` metres of each joint (rings near a bend are mitred). Beyond the ends it continues straight."""

    def __init__(self, pts, blend=0.02):
        self.pts = np.asarray(pts, float)
        d = np.diff(self.pts, axis=0)
        self.seg = np.linalg.norm(d, axis=1)
        self.dirs = d / self.seg[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.length = float(self.cum[-1])
        self.blend = float(blend)

    def point(self, s):
        s = np.asarray(s, float)
        sc = np.clip(s, 0.0, self.length)
        i = np.clip(np.searchsorted(self.cum, sc, side="right") - 1, 0, len(self.seg) - 1)
        p = self.pts[i] + self.dirs[i] * (sc - self.cum[i])[..., None]
        p = np.where((s < 0)[..., None], self.pts[0] + self.dirs[0] * s[..., None], p)
        p = np.where((s > self.length)[..., None], self.pts[-1] + self.dirs[-1] * (s - self.length)[..., None], p)
        return p

    def tangent(self, s):
        s = np.asarray(s, float)
        sc = np.clip(s, 0.0, self.length)
        i = np.clip(np.searchsorted(self.cum, sc, side="right") - 1, 0, len(self.seg) - 1)
        t = self.dirs[i].copy()
        for j in range(1, len(self.pts) - 1):                   # joint j between segment j-1 and j
            w = min(self.blend, 0.5 * self.seg[j - 1], 0.5 * self.seg[j])
            u = smoothstep((sc - (self.cum[j] - w)) / max(2 * w, 1e-9))
            near = np.abs(sc - self.cum[j]) < w
            tj = unit(self.dirs[j - 1][None] * (1 - u)[..., None] + self.dirs[j][None] * u[..., None])
            t = np.where(near[..., None], tj, t)
        return unit(t)

    def project(self, p):
        """Arclength of the closest point of the polyline to each point of p (n, 3) (clamped to [0, length])."""
        p = np.asarray(p, float)
        best = np.full(len(p), np.inf)
        out = np.zeros(len(p))
        for j in range(len(self.seg)):
            a, d = self.pts[j], self.dirs[j]
            t = np.clip((p - a) @ d, 0.0, self.seg[j])
            dist = np.linalg.norm(p - (a + t[:, None] * d), axis=1)
            better = dist < best
            best = np.where(better, dist, best)
            out = np.where(better, self.cum[j] + t, out)
        return out


def fillet_polyline(pts, radii, step=0.004):
    """Dense polyline through `pts` with the interior corner j replaced by a circular arc of radius radii[j] (0 or
    None keeps the corner sharp); points are about `step` metres apart on the arcs."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for j in range(1, len(pts) - 1):
        r = radii[j] if radii[j] else 0.0
        a, b, c = pts[j - 1], pts[j], pts[j + 1]
        u, v = unit(a - b), unit(c - b)
        ang = np.arccos(np.clip(u @ v, -1.0, 1.0))                  # interior angle at the corner
        if r <= 0 or ang > np.pi - 1e-3:
            out.append(b)
            continue
        d = r / np.tan(ang / 2.0)
        d = min(d, 0.5 * np.linalg.norm(a - b), 0.5 * np.linalg.norm(c - b))
        r = d * np.tan(ang / 2.0)
        p0, p1 = b + u * d, b + v * d
        ctr = b + unit(u + v) * (r / np.sin(ang / 2.0))
        e0, e1 = unit(p0 - ctr), unit(p1 - ctr)
        sweep = np.arccos(np.clip(e0 @ e1, -1.0, 1.0))
        k = max(2, int(np.ceil(sweep * r / step)))
        axis = unit(np.cross(e0, e1))
        for i in range(k + 1):
            out.append(ctr + rot_axis(axis, sweep * i / k) @ (e0 * r))
    out.append(pts[-1])
    return np.array(out)


def rmf_frames(t, ef0):
    """Rotation-minimising `ef` vectors along unit tangents t (R, 3), starting from ef0 projected perpendicular to t[0]."""
    t = np.asarray(t, float)
    ef = np.zeros_like(t)
    e = np.asarray(ef0, float)
    e = unit(e - (e @ t[0]) * t[0])
    ef[0] = e
    for i in range(1, len(t)):
        ax = np.cross(t[i - 1], t[i])
        sn = np.linalg.norm(ax)
        if sn > 1e-9:
            ang = np.arctan2(sn, t[i - 1] @ t[i])
            e = rot_axis(ax / sn, ang) @ e
        e = unit(e - (e @ t[i]) * t[i])
        ef[i] = e
    return ef


# ---------------------------------------------------------------- shells
class Shell:
    """A mesh patch: vertices, faces (index lists), per-corner UVs, and per-vertex shape data used by the weights."""

    def __init__(self, name):
        self.name = name
        self.verts = np.zeros((0, 3))
        self.faces = []
        self.uv = []              # one (len(face), 2) array per face
        self.s = np.zeros(0)      # arclength parameter per vertex
        self.theta = np.zeros(0)  # ring angle parameter per vertex (nan on caps' apexes)
        self.ring = np.zeros(0, int)
        self.rings = None         # (R, M, 3) ring points for tubes (None otherwise)
        self.ring_index = None    # (R, M) vertex indices of those rings
        self.mat = 0              # material index of the faces
        self.s_range = (0.0, 1.0)  # arclengths mapped to the tile's v range
        self.info = {}
        self.ext = {}             # local vertex -> (shell name, vertex index in that shell): welded, not stored here
        self.face_mat = []

    def __len__(self):
        return len(self.verts)

    def flip(self):
        self.faces = [f[::-1] for f in self.faces]
        self.uv = [u[::-1] for u in self.uv]

    def signed_volume(self):
        """Volume enclosed (positive when the faces point outward), triangle-fan estimate."""
        vol = 0.0
        v = self.verts
        for f in self.faces:
            for k in range(1, len(f) - 1):
                vol += np.dot(v[f[0]], np.cross(v[f[k]], v[f[k + 1]]))
        return vol / 6.0

    def orient_outward(self):
        if self.signed_volume() < 0:
            self.flip()
        return self


def mirror_x(shell, name=None):
    """The shell mirrored in x = 0 (winding reversed, UVs kept so mirrored limbs share their atlas tile)."""
    m = Shell(name or shell.name)
    m.verts = shell.verts * np.array([-1.0, 1.0, 1.0])
    m.faces = [f[::-1] for f in shell.faces]
    m.uv = [u[::-1].copy() for u in shell.uv]
    m.s, m.theta, m.ring = shell.s.copy(), shell.theta.copy(), shell.ring.copy()
    if shell.rings is not None:
        m.rings = shell.rings * np.array([-1.0, 1.0, 1.0])
        m.ring_index = shell.ring_index.copy()
    m.mat = shell.mat
    m.face_mat = list(shell.face_mat)
    m.s_range = shell.s_range
    m.info = shell.info
    m.ext = {k: (n[:-2] + "_R" if n.endswith("_L") else n, i) for k, (n, i) in shell.ext.items()}
    return m


def _grid_faces(R, M, base, closed=True):
    """Quads between consecutive rings (vertex index = base + i*M + k)."""
    faces = []
    for i in range(R - 1):
        for k in range(M if closed else M - 1):
            k1 = (k + 1) % M
            faces.append([base + i * M + k, base + i * M + k1, base + (i + 1) * M + k1, base + (i + 1) * M + k])
    return faces


def tube(name, path, ss, section, M, ref, cap0="dome", cap1="dome", cap_rings=3, theta0=0.0, uv_rect=(0, 0, 1, 1),
         s_range=None, cap_len=(1.0, 1.0)):
    """Sweep rings along `path` at the arclengths `ss` (ascending) with `section(s) -> dict(rx, ry, ox, oy, n)`.

    Ring point k: c(s) + (ox + rx*C) ef + (oy + ry*S) eb with (C, S) the superellipse point at angle theta0 + 2*pi*k/M.
    `ref` is the world direction the section's `ef` axis follows (projected perpendicular to the tangent). A cap is
    'dome' (a quarter-ellipsoid of `cap_rings` rings closing in one apex vertex), 'flat' (one centre vertex) or 'open'.
    `cap_len` scales the dome depth at the two ends (a blunt palm end, a pointed fingertip). UV rectangle (u0, v0, u1, v1)
    places the tube in the atlas: u around the ring, v along (s_range = (s0, s1) maps to v0..v1; caps stay inside)."""
    ss = np.asarray(ss, float)
    R = len(ss)
    c = path.point(ss)
    t = path.tangent(ss)
    if isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[0], str):          # ("rmf", ef0): transported frame
        ef = rmf_frames(t, ref[1])
    else:
        ref = np.asarray(ref, float)
        ef = unit(ref[None] - (t @ ref)[:, None] * t)
    eb = np.cross(t, ef)
    sec = section(ss)
    th = theta0 + TAU * np.arange(M) / M
    n_ = np.broadcast_to(np.asarray(sec["n"], float).reshape(-1, 1), (R, M))
    C, S = superellipse(th[None, :], n_)
    a = np.asarray(sec["ox"], float)[:, None] + np.asarray(sec["rx"], float)[:, None] * C
    b = np.asarray(sec["oy"], float)[:, None] + np.asarray(sec["ry"], float)[:, None] * S
    P = c[:, None, :] + a[..., None] * ef[:, None, :] + b[..., None] * eb[:, None, :]       # (R, M, 3)

    rings, s_rings, ring_id = [], [], []
    apex0 = apex1 = None
    if cap0 in ("dome", "flat"):
        rad0 = float(np.mean([sec["rx"][0], sec["ry"][0]])) * cap_len[0]
        if cap0 == "dome":
            for j in range(cap_rings - 1, 0, -1):
                ang = 0.5 * np.pi * j / cap_rings
                shift = -np.sin(ang) * rad0
                rings.append(c[0] + (P[0] - c[0]) * np.cos(ang) + t[0] * shift)
                s_rings.append(ss[0] + shift)
                ring_id.append(-1)
            apex0 = (c[0] - t[0] * rad0, ss[0] - rad0)
        else:
            apex0 = (c[0].copy(), ss[0])
    n_pre = len(rings)
    for i in range(R):
        rings.append(P[i])
        s_rings.append(ss[i])
        ring_id.append(i)
    if cap1 in ("dome", "flat"):
        rad1 = float(np.mean([sec["rx"][-1], sec["ry"][-1]])) * cap_len[1]
        if cap1 == "dome":
            for j in range(1, cap_rings):
                ang = 0.5 * np.pi * j / cap_rings
                shift = np.sin(ang) * rad1
                rings.append(c[-1] + (P[-1] - c[-1]) * np.cos(ang) + t[-1] * shift)
                s_rings.append(ss[-1] + shift)
                ring_id.append(-1)
            apex1 = (c[-1] + t[-1] * rad1, ss[-1] + rad1)
        else:
            apex1 = (c[-1].copy(), ss[-1])
    RR = len(rings)
    verts = [np.asarray(r) for r in rings]
    base = 0
    sh = Shell(name)
    vlist, s_par, th_par, rg_par = [], [], [], []
    if apex0 is not None:
        vlist.append(apex0[0][None])
        s_par.append(np.array([apex0[1]]))
        th_par.append(np.array([np.nan]))
        rg_par.append(np.array([-1]))
        base = 1
    for r, sv, rid in zip(verts, s_rings, ring_id):
        vlist.append(r)
        s_par.append(np.full(M, sv))
        th_par.append(th.copy())
        rg_par.append(np.full(M, rid))
    if apex1 is not None:
        vlist.append(apex1[0][None])
        s_par.append(np.array([apex1[1]]))
        th_par.append(np.array([np.nan]))
        rg_par.append(np.array([-1]))
    sh.verts = np.concatenate(vlist, 0)
    sh.s = np.concatenate(s_par)
    sh.theta = np.concatenate(th_par)
    sh.ring = np.concatenate(rg_par)
    faces = _grid_faces(RR, M, base)
    u0, v0, u1, v1 = uv_rect
    sa, sb = (ss[0], ss[-1]) if s_range is None else s_range

    def vv(s):
        return v0 + (v1 - v0) * float(np.clip((s - sa) / max(sb - sa, 1e-9), 0.0, 1.0))

    def uu(k):
        return u0 + (u1 - u0) * k / M

    uvs = []
    for i in range(RR - 1):
        va, vb = vv(s_rings[i]), vv(s_rings[i + 1])
        for k in range(M):
            uvs.append(np.array([[uu(k), va], [uu(k + 1), va], [uu(k + 1), vb], [uu(k), vb]]))
    if apex0 is not None:
        va, vb = vv(apex0[1]), vv(s_rings[0])
        for k in range(M):
            k1 = (k + 1) % M
            faces.append([0, base + k1, base + k])
            uvs.append(np.array([[0.5 * (uu(k) + uu(k + 1)), va], [uu(k + 1), vb], [uu(k), vb]]))
    if apex1 is not None:
        ai = len(sh.verts) - 1
        last = base + (RR - 1) * M
        va, vb = vv(s_rings[-1]), vv(apex1[1])
        for k in range(M):
            k1 = (k + 1) % M
            faces.append([last + k, last + k1, ai])
            uvs.append(np.array([[uu(k), va], [uu(k + 1), va], [0.5 * (uu(k) + uu(k + 1)), vb]]))
    sh.faces = faces
    sh.uv = uvs
    sh.s_range = (float(sa), float(sb))
    sh.rings = P
    sh.ring_index = base + (n_pre + np.arange(R))[:, None] * M + np.arange(M)[None, :]
    sh.face_mat = [0] * len(faces)
    if cap0 != "open" and cap1 != "open":
        sh.orient_outward()
    return sh


def surface_point(path, s, section, ref, theta):
    """Points of a tube's surface at arclength s (array) and ring angle theta (array, same shape), as `tube` places its
    ring points (`ref` as for `tube`; fixed-reference frames only), plus the outward unit normal."""
    s = np.atleast_1d(np.asarray(s, float))
    th = np.broadcast_to(np.asarray(theta, float), s.shape)
    c = path.point(s)
    t = path.tangent(s)
    ref = np.asarray(ref, float)
    ef = unit(ref[None] - (t @ ref)[:, None] * t)
    eb = np.cross(t, ef)
    sec = section(s)
    C, S = superellipse(th, np.asarray(sec["n"], float))
    a = np.asarray(sec["ox"], float) + np.asarray(sec["rx"], float) * C
    b = np.asarray(sec["oy"], float) + np.asarray(sec["ry"], float) * S
    P = c + a[:, None] * ef + b[:, None] * eb
    # normal from the derivatives along the ring and along the path
    d = 1e-4
    C2, S2 = superellipse(th + d, np.asarray(sec["n"], float))
    P_th = (np.asarray(sec["ox"], float) + np.asarray(sec["rx"], float) * C2 - a)[:, None] * ef + \
           (np.asarray(sec["oy"], float) + np.asarray(sec["ry"], float) * S2 - b)[:, None] * eb
    s2 = s + 1e-4
    c2, t2 = path.point(s2), path.tangent(s2)
    ef2 = unit(ref[None] - (t2 @ ref)[:, None] * t2)
    eb2 = np.cross(t2, ef2)
    sec2 = section(s2)
    C3, S3 = superellipse(th, np.asarray(sec2["n"], float))
    a3 = np.asarray(sec2["ox"], float) + np.asarray(sec2["rx"], float) * C3
    b3 = np.asarray(sec2["oy"], float) + np.asarray(sec2["ry"], float) * S3
    P_s = c2 + a3[:, None] * ef2 + b3[:, None] * eb2 - P
    nrm = unit(np.cross(P_th, P_s))
    out = P - c
    flip = np.where((nrm * out).sum(1, keepdims=True) < 0, -1.0, 1.0)
    return P, nrm * flip


def join(shells):
    """Concatenate shells with global indices: (verts, faces, uv, face_mat, ranges {name: (v0, v1)}, s, theta, ring,
    gidx {name: (n_local,) global index of every local vertex}). Vertices a shell marks as `ext` are welded to a vertex
    of another (earlier) shell and are not stored again; the ranges cover each shell's own vertices only."""
    verts, faces, uvs, fm, s, th, ring = [], [], [], [], [], [], []
    ranges, gidx = {}, {}
    off = 0
    for sh in shells:
        n = len(sh)
        g = np.zeros(n, int)
        own = np.ones(n, bool)
        for i, (nm, j) in sh.ext.items():
            own[i] = False
        g[own] = off + np.arange(int(own.sum()))
        for i, (nm, j) in sh.ext.items():
            g[i] = gidx[nm][j]
        ranges[sh.name] = (off, off + int(own.sum()))
        gidx[sh.name] = g
        verts.append(sh.verts[own])
        faces += [[int(g[i]) for i in f] for f in sh.faces]
        uvs += sh.uv
        fm += list(sh.face_mat)
        s.append(sh.s[own])
        th.append(sh.theta[own])
        ring.append(sh.ring[own])
        off += int(own.sum())
    return (np.concatenate(verts, 0), faces, uvs, np.array(fm, int), ranges, np.concatenate(s), np.concatenate(th),
            np.concatenate(ring), gidx)


def vertex_normals(verts, faces):
    """Area-weighted smooth vertex normals of a polygon soup (triangle fans)."""
    n = np.zeros_like(verts)
    for f in faces:
        for k in range(1, len(f) - 1):
            a, b, c = verts[f[0]], verts[f[k]], verts[f[k + 1]]
            fn = np.cross(b - a, c - a)
            n[f[0]] += fn
            n[f[k]] += fn
            n[f[k + 1]] += fn
    return unit(n)


def triangulated(faces):
    """All triangles of a polygon list as an (m, 3) index array (fan triangulation)."""
    tris = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.array(tris, int)


def weld_check(verts, tol=1e-4):
    """Number of distinct vertex pairs closer than tol (diagnostic for unintended doubles)."""
    from collections import defaultdict
    cell = defaultdict(list)
    q = np.floor(verts / tol).astype(np.int64)
    for i, key in enumerate(map(tuple, q)):
        cell[key].append(i)
    return sum(len(v) - 1 for v in cell.values() if len(v) > 1)
