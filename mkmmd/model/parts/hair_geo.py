"""Hair geometry (bpy-free, numpy only): polyline helpers, swept clumps ("strips") and a mesh accumulator.

A strip is one clump of hair, a thin lens-shaped shell swept along a centreline: rings of 2 * k - 2 vertices (k on the
outer arc, k - 2 on the inner arc, edge vertices shared), a pointed / rounded / blunt tip, optionally a closed root.
Frames follow the centreline with an outward hint N (the width direction is B = T x N), so a clump hugs the head when
N is the radial direction. All lengths are metres, model space."""
from dataclasses import dataclass, field

import numpy as np

from ..part import Mesh


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- polylines
def arclen(P):
    """Cumulative arc length of a polyline (m,3) -> (m,), starting at 0."""
    P = np.asarray(P, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def resample(P, n, s_query=None):
    """n points evenly spaced in arc length along the polyline P (or at the arc positions s_query)."""
    P = np.asarray(P, float)
    s = arclen(P)
    q = np.linspace(0.0, s[-1], n) if s_query is None else np.asarray(s_query, float)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], -1)


def smooth_polyline(P, iters=2, keep=(True, True)):
    """Laplacian smoothing of the interior points (ends kept unless keep says otherwise)."""
    P = np.array(P, float)
    for _ in range(iters):
        Q = P.copy()
        Q[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
        if not keep[0]:
            Q[0] = 0.75 * P[0] + 0.25 * P[1]
        if not keep[1]:
            Q[-1] = 0.75 * P[-1] + 0.25 * P[-2]
        P = Q
    return P


def spline(C, n=60, alpha=0.5):
    """Centripetal Catmull-Rom spline through the control points C (k,3) -> n points evenly spaced in arc length."""
    C = np.asarray(C, float)
    if len(C) < 3:
        return resample(C, n)
    P = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    out = []
    dense = 24
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        t = np.linspace(t1, t2, dense, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(C[-1][None, :])
    return resample(np.concatenate(out, 0), n)


def tangents(P):
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    return unit(T)


def frames(P, N):
    """Frames along the centreline P (m,3) for the outward hint N ((3,) or (m,3)): T, B = T x N, N' = B x T."""
    P = np.asarray(P, float)
    T = tangents(P)
    N = np.broadcast_to(np.asarray(N, float), P.shape)
    N = N - (N * T).sum(1, keepdims=True) * T
    B = np.cross(T, N)
    bad = np.linalg.norm(B, axis=1) < 0.2 * np.maximum(np.linalg.norm(N, axis=1), 1e-9)
    if bad.any():
        good = np.flatnonzero(~bad)
        if not len(good):
            raise ValueError("frames: the outward hint is parallel to the centreline everywhere")
        for i in np.flatnonzero(bad):
            j = good[np.argmin(np.abs(good - i))]
            B[i] = B[j] - (B[j] @ T[i]) * T[i]
    # keep the sign continuous
    for i in range(1, len(B)):
        if B[i] @ B[i - 1] < 0:
            B[i] = -B[i]
    B = unit(B)
    Nn = unit(np.cross(B, T))
    return T, B, Nn


# ---------------------------------------------------------------- widths
def taper_factor(sp, kind="point", start=0.55, power=1.5):
    """Width multiplier along s' in [0, 1]: 1 until `start`, then to 0 at the tip.
    point: (1 - t)^power (power > 1 concave needle tips, 1 straight); round: a half ellipse; blunt: constant."""
    sp = np.asarray(sp, float)
    t = np.clip((sp - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    if kind == "point":
        return (1.0 - t) ** power
    if kind == "round":
        return np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))
    if kind == "blunt":
        return np.ones_like(sp)
    raise ValueError(f"unknown tip kind {kind!r}")


@dataclass
class Piece:
    """What a sub-builder (braids, ears, tails ...) hands to hair.py: meshes and materials (bones, bodies and joints go
    straight into the shared `Rig`), plus `info` published in the part's info under its own key."""
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)          # display frame name -> bone names


@dataclass
class Strip:
    """A swept clump. verts (n,3); faces (list of vertex index lists); per vertex: q across (-1 left edge .. +1 right
    edge, B direction), side (+1 outer arc, -1 inner arc, 0 edge or tip), arc (m from the root along the centreline),
    ctr (the ring centre), ring index (-1 for the tip vertex)."""
    verts: np.ndarray
    faces: list
    q: np.ndarray
    side: np.ndarray
    arc: np.ndarray
    ctr: np.ndarray
    ring: np.ndarray
    length: float
    R: int = 0                                  # vertices per ring
    rings: int = 0
    extra: dict = field(default_factory=dict)


def sweep(P, N, width, thick, *, tip="point", tip_start=0.55, tip_power=1.5, root_w=1.0, root_len=0.15, curv=0.0,
          rings=None, ring_step=0.018, k_outer=5, outer_frac=0.6, bulge=0.7, tilt=0.0, close_root=True,
          width_fn=None, thick_fn=None, end_taper=0.0):
    """Sweep a clump along the centreline P (m,3) with outward hint N.

    width, thick   full width / thickness (m) of the clump; width_fn(s') / thick_fn(s') -> multipliers (s' in 0..1)
    tip            "point" (concave needle), "round" or "blunt" (flat cut, `tilt` slants it: fraction of the width)
    tip_start      where the taper starts (fraction of the length); tip_power the needle sharpness
    root_w         width multiplier at the very root, easing to 1 at root_len (fraction of the length)
    curv           lateral cupping (1/m): the edges bend towards the head; 0 = flat across
    rings          number of rings (default: one per ring_step metres, at least 7)
    outer_frac     share of the thickness on the outer side of the centreline; bulge shapes the lens (< 1 = fuller)
    end_taper      blunt tips only: the width falls by this fraction over the last 15 % (rounded corners)"""
    P = np.asarray(P, float)
    L = float(arclen(P)[-1])
    if rings is None:
        rings = max(7, int(round(L / ring_step)))
    pointed = tip in ("point", "round")
    # ring parameters: pointed tips end with a tip vertex at s' = 1, blunt ones with a ring at s' = 1
    n_r = rings
    sp = np.arange(n_r) / n_r if pointed else np.linspace(0.0, 1.0, n_r)
    s_abs = sp * L
    sa = arclen(P)
    Pc = np.stack([np.interp(s_abs, sa, P[:, k]) for k in range(3)], -1)
    # frames from the dense polyline, then sampled at the rings (keeps tangents accurate)
    T, B, _ = frames(P, np.broadcast_to(np.asarray(N, float), P.shape))
    Tc = unit(np.stack([np.interp(s_abs, sa, T[:, k]) for k in range(3)], -1))
    Bc = unit(np.stack([np.interp(s_abs, sa, B[:, k]) for k in range(3)], -1))
    Nn = unit(np.cross(Bc, Tc))
    w = width * taper_factor(sp, tip, tip_start, tip_power)
    if width_fn is not None:
        w = w * width_fn(sp)
    if end_taper and tip == "blunt":
        w = w * (1.0 - end_taper * smoothstep((sp - 0.85) / 0.15))
    if root_w != 1.0:
        w = w * (root_w + (1.0 - root_w) * smoothstep(sp / max(root_len, 1e-6)))
    th = thick * taper_factor(sp, tip, tip_start, 1.0) ** 0.6
    if thick_fn is not None:
        th = th * thick_fn(sp)
    # cross-section
    m = k_outer
    qo = np.linspace(-1.0, 1.0, m)
    qi = qo[1:-1][::-1]                                         # inner arc, right to left, interior only
    qs = np.concatenate([qo, qi])
    sd = np.concatenate([np.ones(m), -np.ones(len(qi))])
    sd[0] = sd[m - 1] = 0.0
    Rn = len(qs)
    lens = np.clip(1.0 - qs * qs, 0.0, 1.0) ** bulge
    verts = np.empty((n_r * Rn, 3))
    for i in range(n_r):
        x = qs * w[i] * 0.5
        y = np.where(sd >= 0, th[i] * outer_frac, -th[i] * (1.0 - outer_frac)) * lens - 0.5 * curv * x * x
        verts[i * Rn:(i + 1) * Rn] = Pc[i] + np.outer(x, Bc[i]) + np.outer(y, Nn[i])
    if tip == "blunt" and tilt:
        i = n_r - 1
        verts[i * Rn:(i + 1) * Rn] -= np.outer(qs * w[i] * 0.5 * tilt, Tc[i])
    q_all = np.tile(qs, n_r)
    side_all = np.tile(sd, n_r)
    arc_all = np.repeat(s_abs, Rn)
    ctr_all = np.repeat(Pc, Rn, axis=0)
    ring_all = np.repeat(np.arange(n_r), Rn)
    faces = []
    for i in range(n_r - 1):
        for a in range(Rn):
            b = (a + 1) % Rn
            faces.append([i * Rn + a, i * Rn + b, (i + 1) * Rn + b, (i + 1) * Rn + a])
    if pointed:
        tip_pt = P[-1]
        verts = np.vstack([verts, tip_pt])
        q_all = np.append(q_all, 0.0)
        side_all = np.append(side_all, 0.0)
        arc_all = np.append(arc_all, L)
        ctr_all = np.vstack([ctr_all, tip_pt])
        ring_all = np.append(ring_all, -1)
        t_idx = len(verts) - 1
        base = (n_r - 1) * Rn
        for a in range(Rn):
            faces.append([base + a, base + (a + 1) % Rn, t_idx])
    else:
        base = (n_r - 1) * Rn
        faces.append(list(range(base, base + Rn)))
    if close_root:
        faces.append(list(range(Rn - 1, -1, -1)))
    return Strip(verts, faces, q_all, side_all, arc_all, ctr_all, ring_all, L, Rn, n_r)


def face_normals(verts, faces):
    """Unit normal of each face (Newell), (len(faces), 3)."""
    out = np.zeros((len(faces), 3))
    for k, f in enumerate(faces):
        p = verts[list(f)]
        out[k] = np.cross(p[1] - p[0], p[2] - p[0]) if len(f) == 3 else np.sum(
            np.cross(p, np.roll(p, -1, axis=0)), axis=0)
    return unit(out)


def vertex_normals(verts, faces):
    """Area-weighted smooth vertex normals."""
    verts = np.asarray(verts, float)
    acc = np.zeros_like(verts)
    for f in faces:
        p = verts[list(f)]
        n = np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0) if len(f) > 3 else np.cross(p[1] - p[0], p[2] - p[0])
        for i in f:
            acc[i] += n
    return unit(acc)


def strip_normals(st, hint, mix=0.6, edge_mix=None):
    """Custom smooth normals for a `Strip`: the outer arc blends the geometric normal towards `hint` ((n,3) unit vectors,
    e.g. the normals of a smooth head-shaped proxy; mix = share of the hint) so a clump shades like part of one big form with
    a gradient along its length; the inner arc mirrors it; edge vertices keep the geometric normal unless `edge_mix` is given
    (then they blend towards the hint by that share, so a clump has no darker rim of its own); the tip vertex takes the mean
    normal of the last ring (a needle's own normal is ill-defined)."""
    geo = vertex_normals(st.verts, st.faces)
    hint = unit(hint)
    n = geo.copy()
    outer = st.side > 0
    inner = st.side < 0
    n[outer] = unit(mix * hint[outer] + (1 - mix) * geo[outer])
    n[inner] = unit(-mix * hint[inner] + (1 - mix) * geo[inner])
    if edge_mix is not None:
        edge = (st.side == 0) & (st.ring >= 0)
        n[edge] = unit(edge_mix * hint[edge] + (1 - edge_mix) * geo[edge])
    tip = st.ring < 0
    if tip.any():
        last = (st.ring == st.rings - 1) & outer
        n[tip] = unit(n[last].mean(0))[None, :]
    return n


# ---------------------------------------------------------------- accumulate into Mesh objects
class MeshAccum:
    """Collects blocks of vertices, faces, UVs, materials and bone weights into one `Mesh`."""

    def __init__(self, name, mats):
        self.name = name
        self.mats = list(mats)
        self._v, self._f, self._uv, self._fm, self._w, self._n = [], [], [], [], [], []
        self.count = 0

    def add(self, verts, faces, uv, mat=0, weights=None, normals=None):
        """Add a block. uv: per-vertex (n,2); mat: index into self.mats (or an int per face); weights: bone -> (n,)."""
        verts = np.asarray(verts, float)
        n = len(verts)
        off = self.count
        uv = np.asarray(uv, float)
        if uv.shape != (n, 2):
            raise ValueError(f"{self.name}: uv must be (n, 2) per vertex, got {uv.shape} for {n} vertices")
        self._v.append(verts)
        self._f.extend([[i + off for i in f] for f in faces])
        self._uv.append(uv)
        self._fm.extend([mat] * len(faces) if np.isscalar(mat) else list(mat))
        self._w.append((off, n, {b: np.asarray(w, float) for b, w in (weights or {}).items()}))
        self._n.append(None if normals is None else np.asarray(normals, float))
        self.count += n
        return off

    def to_mesh(self, smooth=True):
        verts = np.concatenate(self._v, 0) if self._v else np.zeros((0, 3))
        uv_v = np.concatenate(self._uv, 0) if self._uv else np.zeros((0, 2))
        corners = np.array([i for f in self._f for i in f], int)
        uv = uv_v[corners] if len(corners) else np.zeros((0, 2))
        weights = {}
        for off, n, ws in self._w:
            for b, w in ws.items():
                if b not in weights:
                    weights[b] = np.zeros(self.count)
                weights[b][off:off + n] = w
        normals = None
        if self._n and all(x is not None for x in self._n):
            normals = np.concatenate(self._n, 0)
        mats = [m for m in self.mats]
        return Mesh(name=self.name, verts=verts, faces=self._f, uv=uv, face_mat=np.asarray(self._fm, int),
                    mats=mats, weights=weights, normals=normals, smooth=smooth)
