"""Garment geometry kit of the outfit part: rings and lofts, pleated frills, ribbon strips, bows, ellipsoids, and a
`Soup` that merges patches into one `Part` mesh. numpy only (it also runs under Blender's Python 3.11 / numpy 1.24).

Conventions (model space: metres, Z up, the character faces -Y, her left is +X)
- theta, the angle around the vertical axis, runs from the front (-Y) towards her left (+X):
  direction(theta) = (sin t, -cos t, 0); increasing theta is counter-clockwise seen from above.
- A ring is an (M, 3) closed polyline; the rings of a loft are (K, M, 3). Counter-clockwise rings listed from top to
  bottom give OUTWARD faces (`loft(flip=True)` for the other order). Every primitive documents its orientation.
- UVs are per face corner, u to the right and v up (v = 1 is the top row of an image).
- Weights are per vertex dicts `bone -> (n,)` (capped at four bones and normalised by `cap_weights`)."""
from dataclasses import dataclass

import numpy as np

from ..part import Mesh

TAU = 2.0 * np.pi


# ---------------------------------------------------------------- small maths
def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ring_theta(m, start=0.0):
    return start + np.arange(m) * (TAU / m)


def hdir(theta):
    """(.., 3) horizontal outward directions for angles theta (0 = front -Y, +pi/2 = her left +X)."""
    t = np.asarray(theta, float)
    return np.stack([np.sin(t), -np.cos(t), np.zeros_like(t)], -1)


def tangent_ccw(theta):
    """Counter-clockwise (seen from above) horizontal unit tangent at theta."""
    t = np.asarray(theta, float)
    return np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], -1)


def frame_from_axis(axis, ref=(0.0, -1.0, 0.0)):
    """Orthonormal (u, v) with u x v = axis; u is the projection of `ref` perpendicular to the axis."""
    a = unit(axis)
    ref = np.asarray(ref, float)
    u = ref - a * np.dot(ref, a)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) - a * a[0]
    u = unit(u)
    return u, np.cross(a, u)


def rot_axis(axis, angle):
    """3x3 rotation about `axis` by `angle` (Rodrigues)."""
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def periodic_noise(theta, seed, harmonics=5, decay=1.0):
    """Smooth periodic noise in about [-1, 1] over theta (sum of a few harmonics with random phases)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(np.asarray(theta, float))
    tot = 0.0
    for k in range(1, harmonics + 1):
        a = 1.0 / k ** decay
        out = out + a * np.sin(k * theta + rng.uniform(0, TAU))
        tot += a
    return out / tot * 1.6


def catmull(points, n):
    """Resample a polyline of control points (P, d) to n points along a Catmull-Rom spline (ends clamped)."""
    P = np.asarray(points, float)
    if len(P) < 3 or n <= len(P):
        t = np.linspace(0.0, len(P) - 1.0, n)
        i = np.minimum(t.astype(int), len(P) - 2)
        return P[i] + (P[i + 1] - P[i]) * (t - i)[:, None]
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0.0, len(P) - 1.0, n)
    i = np.minimum(t.astype(int), len(P) - 2)
    s = (t - i)[:, None]
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3)


def resample_ring(ring, m):
    """A closed ring (n, 3) resampled to m points by periodic Catmull-Rom through its points (parametrised by index);
    returns (points (m, 3), source index of each new point (m,) = the nearest original vertex)."""
    P = np.asarray(ring, float)
    n = len(P)
    t = np.arange(m) * (n / m)
    i = np.floor(t).astype(int)
    f = (t - i)[:, None]
    p0, p1, p2, p3 = P[(i - 1) % n], P[i % n], P[(i + 1) % n], P[(i + 2) % n]
    pts = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    return pts, np.rint(t).astype(int) % n


def smooth_circular(r, passes=2, width=2):
    """Box-blur a periodic profile (n,) `passes` times with half-width `width` samples."""
    r = np.asarray(r, float)
    for _ in range(passes):
        acc = np.zeros_like(r)
        for s in range(-width, width + 1):
            acc += np.roll(r, s)
        r = acc / (2 * width + 1)
    return r


def dilate_circular(r, width):
    """Moving maximum of a periodic profile over +-width samples."""
    r = np.asarray(r, float)
    out = r.copy()
    for s in range(1, width + 1):
        out = np.maximum(out, np.maximum(np.roll(r, s), np.roll(r, -s)))
    return out


# ---------------------------------------------------------------- patches
@dataclass
class Patch:
    v: np.ndarray                 # (n, 3)
    f: list                       # tuples of vertex indices
    uv: np.ndarray = None         # (sum of face sizes, 2)
    w: dict = None                # bone -> (n,) weights
    mat: int = 0                  # index into the mats of the Soup's mesh
    tag: str = ""

    def copy(self):
        return Patch(self.v.copy(), list(self.f), None if self.uv is None else self.uv.copy(),
                     None if self.w is None else {k: np.array(x) for k, x in self.w.items()}, self.mat, self.tag)


def face_normals(v, faces):
    """Area-weighted (Newell) face normals (nf, 3) of arbitrary polygons."""
    out = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        p = v[list(f)]
        q = np.roll(p, -1, axis=0)
        out[i] = [np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])), np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
                  np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))]
    return out


def tri_arrays(faces):
    """Fan-triangulate polygons: (nt, 3) vertex indices and (nt,) source face index."""
    tris, src = [], []
    for i, f in enumerate(faces):
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
            src.append(i)
    return np.array(tris, int).reshape(-1, 3), np.array(src, int)


def vertex_normals(v, faces):
    """Area-weighted vertex normals (n, 3) (unit)."""
    fn = face_normals(v, faces)
    out = np.zeros((len(v), 3))
    for i, f in enumerate(faces):
        out[list(f)] += fn[i]
    return unit(out)


def flip_patch(p):
    """Reverse the winding of every face (and its UV corners)."""
    p.f = [tuple(reversed(f)) for f in p.f]
    if p.uv is not None:
        out, k = np.empty_like(p.uv), 0
        for f in p.f:
            n = len(f)
            out[k:k + n] = p.uv[k:k + n][::-1]
            k += n
        p.uv = out
    return p


def orient_outward(p, inside):
    """Flip the patch when most face normals point towards `inside` (a point (3,) or per-vertex points (n, 3))."""
    inside = np.asarray(inside, float)
    fn = face_normals(p.v, p.f)
    cen = np.array([p.v[list(f)].mean(0) for f in p.f])
    ref = inside if inside.ndim == 1 else np.array([inside[list(f)].mean(0) for f in p.f])
    if np.sum(np.einsum("ij,ij->i", fn, cen - ref)) < 0:
        flip_patch(p)
    return p


def join(patches, mat=None):
    """Concatenate patches (vertex indices shifted). Weights are zero-filled where a patch lacks a bone."""
    patches = [p for p in patches if p is not None and len(p.f)]
    v, f, uv, off = [], [], [], 0
    wl = []
    for p in patches:
        v.append(p.v)
        f += [tuple(i + off for i in face) for face in p.f]
        uv.append(p.uv if p.uv is not None else np.zeros((sum(len(x) for x in p.f), 2)))
        wl.append((off, len(p.v), p.w))
        off += len(p.v)
    V = np.vstack(v) if v else np.zeros((0, 3))
    W = None
    if any(w is not None for _, _, w in wl):
        W = {}
        for o, n, w in wl:
            for b, arr in (w or {}).items():
                if b not in W:
                    W[b] = np.zeros(off)
                W[b][o:o + n] = arr
    return Patch(V, f, np.vstack(uv) if uv else np.zeros((0, 2)), W, patches[0].mat if mat is None and patches else (mat or 0))


def transform_patch(p, M=None, t=None):
    q = p.copy()
    if M is not None:
        q.v = q.v @ np.asarray(M, float).T
    if t is not None:
        q.v = q.v + np.asarray(t, float)
    return q


def cap_weights(w, n, k=4, floor=1e-4):
    """Normalise per-vertex weights (dict bone -> (n,)) and keep the k largest per vertex. Returns a new dict."""
    names = list(w)
    A = np.stack([np.asarray(w[b], float) for b in names], 1) if names else np.zeros((n, 0))
    if A.shape[1] > k:
        idx = np.argsort(-A, axis=1, kind="stable")[:, :k]
        keep = np.zeros(A.shape, bool)
        np.put_along_axis(keep, idx, True, axis=1)
        A = np.where(keep, A, 0.0)
    A = np.where(A < floor, 0.0, A)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    return {b: A[:, i].copy() for i, b in enumerate(names) if A[:, i].max() > 0.0}


# ---------------------------------------------------------------- lofts
def grid_quads(K, M, closed=True):
    """Quad vertex indices (nf, 4) of a (K, M) vertex grid (index = k * M + j), stepping k then j; ring k to k+1."""
    ncol = M if closed else M - 1
    k = np.arange(K - 1)[:, None]
    j = np.arange(ncol)[None, :]
    j2 = (j + 1) % M
    return np.stack([k * M + j, (k + 1) * M + j, (k + 1) * M + j2, k * M + j2], -1).reshape(-1, 4)


def loft(rings, closed=True, u=None, v=None, flip=False, mat=0, tag="", w=None):
    """Quad surface through rings (K, M, 3). Closed rings wrap j; the UV seam is explicit. `u`: (ncol+1,) or (K, ncol+1),
    `v`: (K,) or (K, ncol+1) (defaults 0..1). Faces step (k,j) -> (k+1,j) -> (k+1,j+1) -> (k,j+1): outward for
    counter-clockwise rings listed top to bottom. `w`: dict of (K, M) weight arrays."""
    R = np.asarray(rings, float)
    K, M = R.shape[:2]
    ncol = M if closed else M - 1
    q = grid_quads(K, M, closed)
    if u is None:
        u = np.linspace(0.0, 1.0, ncol + 1)
    if v is None:
        v = np.linspace(1.0, 0.0, K)
    u = np.broadcast_to(np.asarray(u, float), (K, ncol + 1)) if np.ndim(u) == 1 else np.asarray(u, float)
    v = np.broadcast_to(np.asarray(v, float)[:, None], (K, ncol + 1)) if np.ndim(v) == 1 else np.asarray(v, float)
    k = np.arange(K - 1)[:, None]
    j = np.arange(ncol)[None, :]
    corners = np.stack([np.stack([u[k, j], v[k, j]], -1), np.stack([u[k + 1, j], v[k + 1, j]], -1),
                        np.stack([u[k + 1, j + 1], v[k + 1, j + 1]], -1), np.stack([u[k, j + 1], v[k, j + 1]], -1)], 2)
    corners = corners.reshape(-1, 4, 2)
    if flip:
        q = q[:, ::-1]
        corners = corners[:, ::-1]
    ww = None if w is None else {b: np.asarray(a, float).reshape(-1) for b, a in w.items()}
    return Patch(R.reshape(-1, 3), [tuple(int(i) for i in r) for r in q], corners.reshape(-1, 2), ww, mat, tag)


def fan(apex, ring, flip=False, uv_apex=(0.5, 0.5), mat=0, tag=""):
    """Triangle fan closing a ring (M, 3) at `apex`. Faces (apex, j, j+1): outward (away from the apex side) for
    counter-clockwise rings seen from the apex side; flip for the other side."""
    ring = np.asarray(ring, float)
    M = len(ring)
    v = np.vstack([np.asarray(apex, float)[None], ring])
    f, uv = [], []
    for j in range(M):
        j2 = (j + 1) % M
        tri = (0, 1 + j, 1 + j2)
        f.append(tri[::-1] if flip else tri)
        uvs = [uv_apex, (0.5 + 0.5 * np.cos(TAU * j / M), 0.5 + 0.5 * np.sin(TAU * j / M)),
               (0.5 + 0.5 * np.cos(TAU * (j + 1) / M), 0.5 + 0.5 * np.sin(TAU * (j + 1) / M))]
        uv += uvs[::-1] if flip else uvs
    return Patch(v, f, np.array(uv), None, mat, tag)


def ring_uv(m, closed=True):
    return np.linspace(0.0, 1.0, m + 1 if closed else m)


# ---------------------------------------------------------------- frills
def frill_rings(base, out, along, profile, pleats=40, amp=0.012, amp_pow=1.2, theta=None, phase=0.0,
                wobble=0.0, hem_wobble=0.0, ridge=0.18, seed=0):
    """Rings (R, M, 3) of a pleated frill hung from the closed curve `base` (M, 3).

    out (M, 3): unit direction pointing away from the cloth the frill hangs on; along (M, 3): unit direction the frill
    hangs in (down for a skirt, towards the hand for a cuff, up for a collar). profile (R, 2): (outward, along) offset
    in metres of each ring from the base (row 0 = the attachment); the radial pleat wave grows like (row/(R-1))^amp_pow
    up to `amp`; `ridge` mixes a third harmonic in (box pleats instead of a sine); `wobble` randomises pleat phase and
    height smoothly; `hem_wobble` lifts/lowers the hem (scalloped edge) along `along`."""
    base, out, along = (np.asarray(a, float) for a in (base, out, along))
    prof = np.asarray(profile, float)
    R, M = len(prof), len(base)
    th = ring_theta(M) if theta is None else np.asarray(theta, float)
    n1 = periodic_noise(th, seed + 1, 6, 0.8)
    n2 = periodic_noise(th, seed + 2, 6, 0.8)
    x = pleats * th + phase + wobble * 1.4 * n1
    wave = np.sin(x) + ridge * np.sin(3 * x + 0.5)
    wave = wave / (1.0 + ridge)
    height_var = 1.0 + wobble * 0.35 * n2
    t = np.linspace(0.0, 1.0, R)
    rings = np.empty((R, M, 3))
    for i in range(R):
        ti = t[i]
        off_out = prof[i, 0] + amp * ti ** amp_pow * wave * (1.0 + wobble * 0.5 * n2)
        off_along = prof[i, 1] * height_var - hem_wobble * ti ** 2 * np.cos(2 * x)
        rings[i] = base + out * off_out[:, None] + along * np.broadcast_to(off_along, (M,))[:, None]
    return rings


# ---------------------------------------------------------------- strips, tubes, ellipsoids, bows
def strip(path, half_width, side, ncols=2, flip=False, mat=0, tag="", u=None, closed=False, edge_lift=0.0):
    """Ribbon band along `path` (N, 3): across the band `side` (N, 3 or (3,)) unit vectors, half_width scalar or (N,).
    ncols vertices across (>= 2). Faces point along tangent x side (flip reverses). closed wraps the path (a loop).
    u: (N + closed,) texture coordinate along the path (default 0..1), v runs across the band 0..1; edge_lift cups the
    band (edge vertices move along the face normal by edge_lift at the borders, quadratically)."""
    P = np.asarray(path, float)
    N = len(P)
    S = np.broadcast_to(unit(side), P.shape) if np.ndim(side) == 1 else unit(side)
    hw = np.broadcast_to(np.asarray(half_width, float), (N,))
    fr = np.linspace(-1.0, 1.0, ncols)
    g = P[:, None, :] + S[:, None, :] * (hw[:, None] * fr[None, :])[:, :, None]       # (N, ncols, 3)
    if edge_lift:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0) if closed else np.gradient(P, axis=0)
        n = unit(np.cross(T, S))
        g = g + n[:, None, :] * (edge_lift * (fr ** 2))[None, :, None]
    nseg = N if closed else N - 1
    k = np.arange(nseg)[:, None]
    j = np.arange(ncols - 1)[None, :]
    k2 = (k + 1) % N
    q = np.stack([k * ncols + j, k2 * ncols + j, k2 * ncols + j + 1, k * ncols + j + 1], -1).reshape(-1, 4)
    uu = np.linspace(0.0, 1.0, N + (1 if closed else 0)) if u is None else np.asarray(u, float)
    vv = np.linspace(0.0, 1.0, ncols)
    kk = np.broadcast_to(k, (nseg, ncols - 1))
    jj = np.broadcast_to(j, (nseg, ncols - 1))
    corners = np.stack([np.stack([uu[kk], vv[jj]], -1), np.stack([uu[kk + 1], vv[jj]], -1),
                        np.stack([uu[kk + 1], vv[jj + 1]], -1), np.stack([uu[kk], vv[jj + 1]], -1)], 2).reshape(-1, 4, 2)
    if flip:
        q = q[:, ::-1]
        corners = corners[:, ::-1]
    return Patch(g.reshape(-1, 3), [tuple(int(i) for i in r) for r in q], corners.reshape(-1, 2), None, mat, tag)


def ellipsoid(center, radii, axes=None, n_lat=8, n_lon=12, mat=0, tag="", u_scale=1.0):
    """UV-ellipsoid, outward faces. axes (3, 3) rows = unit axes of the radii (default world)."""
    A = np.eye(3) if axes is None else np.asarray(axes, float)
    r = np.asarray(radii, float)
    lat = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    lon = np.arange(n_lon) * (TAU / n_lon)
    verts = [np.array([0.0, 0.0, 1.0])]
    for la in lat:
        for lo in lon:
            verts.append([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)])
    verts.append(np.array([0.0, 0.0, -1.0]))
    V = np.array(verts)
    P = np.asarray(center, float) + (V * r) @ A
    faces, uv = [], []
    nl = len(lat)
    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        faces.append((0, 1 + j2, 1 + j))
        uv += [((j + 0.5) / n_lon * u_scale, 1.0), ((j + 1) / n_lon * u_scale, 1 - 1.0 / (nl + 1)), (j / n_lon * u_scale, 1 - 1.0 / (nl + 1))]
    for k in range(nl - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            a, b = 1 + k * n_lon + j, 1 + k * n_lon + j2
            c, d = 1 + (k + 1) * n_lon + j2, 1 + (k + 1) * n_lon + j
            faces.append((a, b, c, d))
            v0, v1 = 1 - (k + 1) / (nl + 1), 1 - (k + 2) / (nl + 1)
            uv += [(j / n_lon * u_scale, v0), ((j + 1) / n_lon * u_scale, v0), ((j + 1) / n_lon * u_scale, v1), (j / n_lon * u_scale, v1)]
    last = len(V) - 1
    base = 1 + (nl - 1) * n_lon
    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        faces.append((last, base + j, base + j2))
        uv += [((j + 0.5) / n_lon * u_scale, 0.0), (j / n_lon * u_scale, 1.0 / (nl + 1)), ((j + 1) / n_lon * u_scale, 1.0 / (nl + 1))]
    p = Patch(P, faces, np.array(uv), None, mat, tag)
    # make sure faces point outwards whatever the handedness of `axes`
    return orient_outward(p, np.asarray(center, float))


def sweep_frames(path):
    """Rotation-minimising frames along a polyline (N, 3): tangents T, normals Nn, binormals B (each (N, 3))."""
    P = np.asarray(path, float)
    T = np.gradient(P, axis=0)
    T = unit(T)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = unit(ref - T[0] * np.dot(ref, T[0]))
    N = np.empty_like(P)
    N[0] = n
    for i in range(1, len(P)):
        v1 = P[i] - P[i - 1]
        c1 = np.dot(v1, v1)
        if c1 < 1e-18:
            N[i] = N[i - 1]
            continue
        rL = N[i - 1] - (2.0 / c1) * np.dot(v1, N[i - 1]) * v1
        tL = T[i - 1] - (2.0 / c1) * np.dot(v1, T[i - 1]) * v1
        v2 = T[i] - tL
        c2 = np.dot(v2, v2)
        N[i] = rL - (2.0 / max(c2, 1e-18)) * np.dot(v2, rL) * v2 if c2 > 1e-18 else rL
        N[i] = unit(N[i] - T[i] * np.dot(N[i], T[i]))
    return T, N, np.cross(T, N)


def tube(path, radius, n=10, caps=(True, True), mat=0, tag="", u_scale=1.0, dome=0.6):
    """Tube around `path` (N, 3) with radius scalar or (N,) (a (N, 2) array gives elliptical radii along the frame
    normal and binormal). Outward faces; round caps are single-apex fans."""
    P = np.asarray(path, float)
    T, Nn, B = sweep_frames(P)
    r = np.broadcast_to(np.asarray(radius, float), (len(P),) if np.ndim(radius) < 2 else (len(P), 2))
    ra, rb = (r, r) if r.ndim == 1 else (r[:, 0], r[:, 1])
    ang = ring_theta(n)
    rings = P[:, None, :] + (np.cos(ang)[None, :, None] * Nn[:, None, :] * ra[:, None, None]
                             + np.sin(ang)[None, :, None] * B[:, None, :] * rb[:, None, None])
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    p = loft(rings, True, u=np.linspace(0, u_scale, n + 1), v=1.0 - L / max(L[-1], 1e-9), mat=mat, tag=tag)
    p = orient_outward(p, np.repeat(P, n, axis=0))
    parts = [p]
    if caps[0]:
        c = fan(P[0] - T[0] * ra[0] * dome, rings[0], mat=mat, tag=tag)
        parts.append(orient_outward(c, P[0] + T[0]))
    if caps[1]:
        c = fan(P[-1] + T[-1] * ra[-1] * dome, rings[-1], mat=mat, tag=tag)
        parts.append(orient_outward(c, P[-1] - T[-1]))
    return parts


def bow_ex(origin, right, up, forward, size=0.04, loop_len=1.0, loop_w=0.55, ribbon_w=0.3, tail_len=1.2, tail_splay=0.35,
           droop=0.12, knot=0.22, mat=0, tag="bow", n_loop=18, tail_rows=8, notch=0.22, tilt=0.0, asym=0.0, seed=0, cup=0.0,
           cols=3):
    """A ribbon bow: knot, two loops and two tails. `size` is the half span of one loop at loop_len = 1.

    Frame: lobes spread along +-right, the ribbon width runs along `up`, `forward` is the bow's facing direction (away
    from the body). Loop curves lie in the (right, forward) plane; tails hang along -up with a slight outward splay.
    `cup`: how far the ribbon edges curl, as a fraction of the widest half width (0 = flat sheets, which throw one flat
    sheen; a cupped ribbon catches a highlight band); `cols`: vertices across the ribbon (>= 3, odd: the centre line is
    every cols-th vertex from cols // 2).
    Returns a dict: parts (Patches: loops, knot, tails; outward faces; single sheets, use a double-sided material), tails
    (list of (centre line (tail_rows, 3), patch), the +right tail first) and knot (the knot's centre)."""
    O = np.asarray(origin, float)
    r_, u_, f_ = unit(right), unit(up), unit(forward)
    rng = np.random.default_rng(seed)
    parts = []
    phi = np.linspace(0.0, TAU, n_loop, endpoint=False)
    for sgn, scale in ((1.0, 1.0 + asym), (-1.0, 1.0 - asym)):
        L = size * loop_len * scale
        x = L * (1.0 - np.cos(phi)) / 2.0                       # 0 .. L .. 0 along the lobe
        xn = x / max(L, 1e-9)
        y = 0.5 * size * loop_w * np.sin(phi) * (0.30 + 0.70 * xn ** 0.8)
        path = (O + (sgn * x)[:, None] * r_ + y[:, None] * f_ - (droop * size * xn ** 2)[:, None] * u_
                + (tilt * size * xn)[:, None] * u_ * sgn)
        hw = 0.5 * size * ribbon_w * (0.7 + 0.5 * xn)
        p = strip(path, hw, u_, ncols=cols, closed=True, mat=mat, tag=tag, edge_lift=cup * float(hw.max()))
        parts.append(orient_outward(p, O + sgn * 0.5 * L * r_))
    parts.append(ellipsoid(O + f_ * (0.02 * size), (knot * size * 0.9, knot * size * 0.8, size * ribbon_w * 0.62),
                           np.array([r_, f_, u_]), 6, 10, mat=mat, tag=tag))
    tails = []
    for sgn in (1.0, -1.0):
        ang = sgn * tail_splay * (1.0 + 0.15 * rng.uniform(-1, 1))
        t = np.linspace(0.0, 1.0, tail_rows)
        L = size * tail_len
        path = (O + (-u_)[None, :] * (L * t * np.cos(ang))[:, None] + r_[None, :] * (L * t * np.sin(ang))[:, None]
                + f_[None, :] * (0.06 * size * np.sin(np.pi * t) + 0.02 * size)[:, None])
        side = unit(np.cross(f_, np.gradient(path, axis=0)))
        hw = 0.5 * size * ribbon_w * (0.65 + 0.35 * t)
        p = strip(path, hw, side, ncols=cols, mat=mat, tag=tag, edge_lift=cup * float(hw.max()))
        # swallow-tail notch: the middle of the last row climbs back along the ribbon
        mid = (tail_rows - 1) * cols + cols // 2
        p.v[mid] += u_ * (notch * size * ribbon_w * 1.2)
        p = orient_outward(p, O - f_ * (0.5 * size))
        parts.append(p)
        tails.append((path, p))
    return {"parts": parts, "tails": tails, "knot": O + f_ * (0.02 * size)}


def bow(*args, **kw):
    """`bow_ex(...)["parts"]`: the list of Patches of a ribbon bow (see bow_ex)."""
    return bow_ex(*args, **kw)["parts"]


# ---------------------------------------------------------------- accumulation into a Part mesh
class Soup:
    """Collects patches (each with a material index into `mats`) and builds one `Mesh`."""

    def __init__(self, name, mats):
        self.name = name
        self.mats = list(mats)
        self.patches = []

    def add(self, patch, mat=None):
        if patch is None or not len(patch.f):
            return patch
        if mat is not None:
            patch = Patch(patch.v, patch.f, patch.uv, patch.w, mat, patch.tag)
        self.patches.append(patch)
        return patch

    def extend(self, patches, mat=None):
        for p in patches:
            self.add(p, mat)

    def mesh(self, weld=1e-6, require_weights=True, cap=4):
        P = join(self.patches)
        fmat = np.concatenate([np.full(len(p.f), p.mat, int) for p in self.patches]) if self.patches else np.zeros(0, int)
        verts, faces, uv, W = P.v, P.f, P.uv, P.w
        if weld:
            verts, faces, uv, fmat, W = _weld(verts, faces, uv, fmat, W, weld)
        faces, uv, fmat = _drop_slivers(verts, faces, uv, fmat)
        weights = {}
        if W is not None:
            weights = cap_weights(W, len(verts), cap)
            tot = sum(weights.values()) if weights else np.zeros(len(verts))
            if require_weights and (np.asarray(tot) < 0.5).any():
                bad = int((np.asarray(tot) < 0.5).sum())
                raise ValueError(f"outfit mesh {self.name}: {bad} vertices have no weights")
        used = sorted(set(int(m) for m in fmat))
        remap = {m: i for i, m in enumerate(used)}
        return Mesh(self.name, verts, [tuple(f) for f in faces], np.asarray(uv, float),
                    np.array([remap[int(m)] for m in fmat], int), [self.mats[m] for m in used], weights)


def _drop_slivers(verts, faces, uv, fmat, eps=1e-10):
    """Remove zero-area triangles and clean the quads that hide one: a triangle with (almost) no area is dropped; a quad
    with three collinear corners (a triangle with an extra vertex on one edge, from which the assembler's ear clipping
    would cut a zero-area triangle) becomes the triangle of its other three corners; a quad without area is dropped.
    Returns (faces, uv, fmat)."""
    if not len(faces):
        return faces, uv, fmat
    lens = np.array([len(f) for f in faces])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    q = verts[np.array([list(f) + [f[-1]] * (4 - len(f)) if len(f) < 4 else list(f)[:4] for f in faces])]
    T = {3: (0, 1, 2), 2: (0, 1, 3), 1: (0, 2, 3), 0: (1, 2, 3)}       # corner k omitted -> the triangle left (original order)
    A = {k: 0.5 * np.linalg.norm(np.cross(q[:, t[1]] - q[:, t[0]], q[:, t[2]] - q[:, t[0]]), axis=1) for k, t in T.items()}
    bad = (A[3] <= eps) | ((lens == 4) & ((A[2] <= eps) | (A[1] <= eps) | (A[0] <= eps)))
    if not bad.any():
        return faces, uv, fmat
    nf, nuv, nm = [], [], []
    for i, f in enumerate(faces):
        s = starts[i]
        if not bad[i]:
            cut = [tuple(range(len(f)))]
        elif len(f) == 4:
            k = max(range(4), key=lambda j: A[j][i])                   # the omission that leaves the biggest triangle
            cut = [T[k]] if A[k][i] > eps else []
        else:
            cut = []
        for t in cut:
            nf.append(tuple(f[c] for c in t))
            nuv.append(uv[[s + c for c in t]])
            nm.append(fmat[i])
    return nf, (np.vstack(nuv) if nuv else np.zeros((0, 2))), np.array(nm, int)


def _weld(verts, faces, uv, fmat, W, tol):
    """Merge vertices closer than tol (quantised; weights of merged vertices are averaged) and collapse faces that
    repeat a vertex (a quad with two merged corners becomes a triangle; fewer than 3 distinct corners: dropped)."""
    key = np.round(verts / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    nv_count = int(inv.max()) + 1 if len(inv) else 0
    if nv_count == len(verts):
        return verts, faces, uv, fmat, W
    cnt = np.bincount(inv, minlength=nv_count).astype(float)
    nv = np.zeros((nv_count, 3))
    np.add.at(nv, inv, verts)
    nv /= cnt[:, None]
    nW = None
    if W is not None:
        nW = {}
        for b, a in W.items():
            acc = np.zeros(nv_count)
            np.add.at(acc, inv, a)
            nW[b] = acc / cnt
    nf, nuv, nm = [], [], []
    k = 0
    for fi, f in enumerate(faces):
        n = len(f)
        idx = [int(inv[i]) for i in f]
        cuv = uv[k:k + n]
        k += n
        keep = [c for c in range(n) if idx[c] != idx[(c + 1) % n]]       # drop a corner equal to its successor
        out = [idx[c] for c in keep]
        if len(out) < 3 or len(set(out)) != len(out):
            continue
        nf.append(tuple(out))
        nuv.append(cuv[keep])
        nm.append(fmat[fi])
    return nv, nf, (np.vstack(nuv) if nuv else np.zeros((0, 2))), np.array(nm, int), nW
