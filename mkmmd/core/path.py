"""Paths (roads, rails, camera tracks): a smooth centerline through control points, sampled densely and
parameterised by arc length. numpy only (runs in Blender and the CLI).

Frames: tangent T (direction of travel), left L (horizontal, to the traveller's left), up U = T x ... kept close to +Z
(banking is ignored: roads stay level across). Lateral offsets are + to the LEFT of the direction of travel."""
import math

import numpy as np


def catmull_rom(points, samples_per_seg=24, closed=False):
    """Centripetal Catmull-Rom through points (n, 3) -> dense polyline (m, 3)."""
    P = np.asarray(points, float)
    if len(P) < 2:
        raise ValueError("a path needs at least two points")
    if len(P) == 2:
        t = np.linspace(0, 1, samples_per_seg + 1)[:, None]
        return P[0] * (1 - t) + P[1] * t
    ext = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    nseg = len(P) if closed else len(P) - 1
    for i in range(nseg):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-9) ** 0.5
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_seg, endpoint=False)[:, None]
        a1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        a2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        a3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        b1 = (t2 - ts) / (t2 - t0) * a1 + (ts - t0) / (t2 - t0) * a2
        b2 = (t3 - ts) / (t3 - t1) * a2 + (ts - t1) / (t3 - t1) * a3
        out.append((t2 - ts) / (t2 - t1) * b1 + (ts - t1) / (t2 - t1) * b2)
    out.append(P[:1] if closed else P[-1:])
    return np.vstack(out)


class Path:
    """Dense centerline with arc-length lookup. s in metres from the start."""

    def __init__(self, points, samples_per_seg=24, closed=False):
        self.control = np.asarray(points, float)
        self.P = catmull_rom(self.control, samples_per_seg, closed)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def _interp(self, s, arr):
        s = np.clip(np.asarray(s, float), 0.0, self.length)
        return np.stack([np.interp(s, self.s, arr[:, i]) for i in range(arr.shape[1])], -1)

    def point(self, s):
        return self._interp(s, self.P)

    def tangent(self, s, ds=0.5):
        s = np.asarray(s, float)
        d = self.point(np.minimum(s + ds, self.length)) - self.point(np.maximum(s - ds, 0.0))
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)

    def frame(self, s):
        """(T, L, U): unit tangent, left (horizontal), up, each (..., 3)."""
        T = self.tangent(s)
        up = np.zeros_like(T)
        up[..., 2] = 1.0
        L = np.cross(up, T)
        L /= np.maximum(np.linalg.norm(L, axis=-1, keepdims=True), 1e-12)
        U = np.cross(T, L)
        return T, L, U

    def offset(self, s, lateral=0.0, height=0.0):
        """Point at arc length s shifted `lateral` m to the left and `height` m up."""
        T, L, U = self.frame(s)
        return self.point(s) + L * np.asarray(lateral, float)[..., None] + U * np.asarray(height, float)[..., None]

    def heading(self, s):
        """Yaw (rad) of the direction of travel, 0 = +X, counter-clockwise."""
        T = self.tangent(s)
        return np.arctan2(T[..., 1], T[..., 0])

    def curvature(self, s, ds=2.0):
        """Signed curvature (1/m), + turning left."""
        h0, h1 = self.heading(np.maximum(np.asarray(s, float) - ds, 0)), self.heading(
            np.minimum(np.asarray(s, float) + ds, self.length))
        dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        return dh / (2 * ds)

    def project(self, X):
        """Arc length of the nearest centerline sample to points X (n, 3)."""
        X = np.atleast_2d(np.asarray(X, float))
        d = np.linalg.norm(self.P[None, :, :2] - X[:, None, :2], axis=2)
        return self.s[np.argmin(d, axis=1)]


def gentle_road(length=2000.0, seed=0, max_turn_deg=12.0, leg=250.0, z=0.0, start=(0.0, 0.0), heading_deg=-90.0):
    """Control points of a gently curving road: legs of `leg` m, each turning at most max_turn_deg (seeded)."""
    rng = np.random.default_rng(seed)
    pts = [np.array([start[0], start[1], z])]
    h = math.radians(heading_deg)
    n = max(2, int(math.ceil(length / leg)))
    for _ in range(n):
        h += math.radians(rng.uniform(-max_turn_deg, max_turn_deg))
        pts.append(pts[-1] + leg * np.array([math.cos(h), math.sin(h), 0.0]))
    return np.array(pts)


def speed_profile(keys, fps, frames):
    """Distance travelled (m) per frame from speed keys [[t, m/s], ...] (clip seconds, linear between keys).
    frames are clip-relative frame indices (0 = clip time 0); negative frames run before clip start."""
    keys = sorted(keys)
    kt = np.array([k[0] for k in keys], float)
    kv = np.array([k[1] for k in keys], float)
    t = np.asarray(frames, float) / fps
    v = np.interp(t, kt, kv)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    i0 = int(np.argmin(np.abs(t)))
    return s - s[i0], v
