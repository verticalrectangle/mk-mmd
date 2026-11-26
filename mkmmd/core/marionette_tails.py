"""Tails, bpy-free (docs/design.md: Marionette, tails): light threads tied to her at one end and free at the other (the six
a member wears with no control at all, or the pieces a `snip` leaves on her), streaming opposite to her motion through
the air and with a wind, floating where gravity is eased out, held where something holds them (her fist, the hand that
hauls them in), kept out of her body; settled on the first frame they are seen. mkmmd/blender/build/marionette.py
samples where they are tied, what holds them and her body's collision shapes, and draws what this returns.

The model: a position-based rope per thread, `n` particles a fixed distance apart (a cotton thread does not stretch).
Each substep the particles take gravity (scaled: 0 floats them) and are drawn toward the air's velocity, implicitly (a
light thread takes the air's speed in a fraction of a second: `drag`); then rounds of: the tied end on her and each held
point on its holder, every segment back to its length (alternate segments, the held side not moving), a little bending
stiffness (a thread curves, it does not kink), no particle farther from the tied end than its length of thread; then
her body's shapes push the particles out (mkmmd.solvers.geom), except the first few centimetres at the tie, which lie
on her. The air is a mean wind (constant, or on every frame: the member's own `[sim.<cast>] wind` as
mkmmd.solvers.strands.Wind gives it, keys and gusts) and a few travelling ripples (seeded) that the threads fly
through: they flutter. `settle` runs the first frame for that long first, in the frame of each tie moving on at its
first velocity, so threads seen from their first frame stream as they will (nothing flings)."""
import math

import numpy as np

from ..solvers import geom

GRAVITY = np.array([0.0, 0.0, -9.81])
SUBSTEPS = 6
ITERS = 5
DRAG = 14.0             # 1/s: a light thread's air drag (its fall in still air tops out near 0.7 m/s)
BEND = 0.06             # share of an inner particle's offset from its neighbours' midpoint taken out per round
RADIUS = 0.004          # m: a thread's collision radius (its thickness and a hair more)
ROOT_FREE = 0.07        # m along a thread from its tie that lies on her: no collisions there
RIPPLE = (0.45, 0.7)    # m/s and m: the travelling ripples' speed amplitude and wavelength (the flutter)


def _unit(v, fallback=(0.0, 0.0, -1.0)):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    fb = np.broadcast_to(np.asarray(fallback, float), v.shape)
    return np.where(n > 1e-9, v / np.where(n > 1e-9, n, 1.0), fb)


def resample(P, n):
    """n points evenly along the polyline P (m, 3) by its length."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(P[:1], n, 0)
    u = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(u, cum, P[:, j]) for j in range(3)], 1)


def stretched(P, length, n, way):
    """n points of a thread `length` m long laid along the polyline P from its first point, going on straight along
    `way` past P's end (or cut short where its length runs out)."""
    P = np.asarray(P, float)
    have = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if have < length:
        P = np.vstack([P, P[-1] + _unit(way) * (length - have)])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    u = np.linspace(0.0, length, n)
    return np.stack([np.interp(u, cum, P[:, j]) for j in range(3)], 1)


def arc(P):
    """m along the polyline P."""
    return float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())


class Air:
    """The air's velocity at points and times: a mean wind (`mean` (3,) m/s, or (F, 3) at the clip seconds `ts`, linear
    between them) and RIPPLE's travelling waves, divergence-free, carried along with it (seeded)."""

    def __init__(self, mean=(0.0, 0.0, 0.0), ts=None, ripple=RIPPLE[0], seed=0):
        m = np.asarray(mean, float)
        self.ts = None if m.ndim == 1 else np.asarray(ts, float)
        self.mean = m
        self.drift = m if m.ndim == 1 else m.mean(axis=0)
        rng = np.random.default_rng(int(seed))
        k = rng.normal(size=(4, 3))
        k = _unit(k) * (2.0 * math.pi / RIPPLE[1]) * rng.uniform(0.7, 1.4, (4, 1))
        d = np.cross(k, rng.normal(size=(4, 3)))
        self.k, self.d = k, _unit(d) * float(ripple) / 2.0
        self.w = rng.uniform(1.5, 3.5, 4)
        self.ph = rng.uniform(0.0, 2.0 * math.pi, 4)

    def mean_at(self, t):
        if self.ts is None:
            return self.mean
        return np.array([np.interp(t, self.ts, self.mean[:, j]) for j in range(3)])

    def __call__(self, X, t):
        X = np.asarray(X, float)
        arg = np.einsum("...j,mj->...m", X - self.drift * t, self.k) - self.w * t + self.ph
        return self.mean_at(t) + np.einsum("...m,mj->...j", np.sin(arg), self.d)


def simulate(ts, roots, lengths, init=None, gravity=1.0, air=None, pins=(), shapes=None, n=40, drag=DRAG,
             settle=0.0, substeps=SUBSTEPS, free=ROOT_FREE, v_init=None):
    """(F, K, n, 3): K threads over the frames at clip seconds `ts` (F,), each tied at `roots` (F, K, 3) and `lengths`
    (K,) m long. `init` (K) polylines from each root: their shape on the first frame (resampled; a polyline shorter
    than its thread goes on straight past its end); None: each streams from its root along the air going by it.
    `gravity` a scale or (F,) scales of the world's. `air` an Air (still air without). `pins` [(k, s, P (F, 3), on (F,)
    bool)]: the point `s` m along thread k is held at P on the frames where `on`. `shapes(i)` the world collision shapes
    on frame i (mkmmd.solvers.geom.world), None for none. `settle` s: the first frame run first in the frame of each tie
    moving on at its first velocity. `v_init` (K, 3): the threads' first velocity (default their ties')."""
    ts = np.asarray(ts, float)
    roots = np.asarray(roots, float)
    F, K = roots.shape[:2]
    lengths = np.maximum(np.asarray(lengths, float).reshape(K), 1e-3)
    air = air or Air()
    gs = np.asarray(gravity, float) * np.ones(F) if np.ndim(gravity) == 0 else np.asarray(gravity, float)
    seg = lengths / (n - 1)
    along = np.arange(n)[None, :] * seg[:, None]                  # (K, n) thread from the tie
    held = np.zeros((F, K, n), bool)
    hold_at = np.zeros((F, K, n, 3))
    for k, s, P, on in pins:
        j = int(np.clip(round(float(s) / seg[k]), 1, n - 1))
        held[:, k, j] = np.asarray(on, bool)
        hold_at[:, k, j] = np.asarray(P, float)
    collide = along > free
    v0 = (roots[1] - roots[0]) / max(ts[1] - ts[0], 1e-6) if F > 1 else np.zeros((K, 3))
    x = np.empty((K, n, 3))
    for k in range(K):
        if init is not None and init[k] is not None:
            Pk = np.asarray(init[k], float)
            way = Pk[-1] - Pk[-2] if len(Pk) > 1 else air(roots[0, k], ts[0]) - v0[k]
            x[k] = stretched(Pk, lengths[k], n, way)
        else:
            way = air(roots[0, k], ts[0]) - v0[k] + gs[0] * GRAVITY * 0.05
            x[k] = roots[0, k] + np.outer(np.linspace(0.0, lengths[k], n), _unit(way))
    x[held[0]] = hold_at[0][held[0]]
    v = np.repeat((v0 if v_init is None else np.asarray(v_init, float))[:, None, :], n, 1)
    W0 = shapes(0) if shapes is not None else None
    if settle > 0.0:                                              # ride each tie at its first velocity for `settle` s
        h = 1.0 / (60.0 * substeps)
        base = roots[0][:, None, :]
        rel = x - base
        vr = np.zeros_like(v)
        for step in range(int(round(settle / h))):
            t = ts[0] - settle + step * h
            vr = (vr + h * (gs[0] * GRAVITY + drag * (air(rel + base, t) - v0[:, None, :]))) / (1.0 + h * drag)
            p = _constrain(rel + h * vr, np.zeros((K, 3)), seg, along, held[0], hold_at[0] - base)
            if W0 is not None:
                p = _push(p + base, W0, collide) - base
                p[:, 0] = 0.0
            vr = (p - rel) / h
            rel = p
        x = rel + base
        v = vr + v0[:, None, :]
    out = np.empty((F, K, n, 3))
    out[0] = x
    for i in range(1, F):
        dt = ts[i] - ts[i - 1]
        h = dt / substeps
        W = shapes(i) if shapes is not None else None
        for s in range(1, substeps + 1):
            a = s / substeps
            t = ts[i - 1] + a * dt
            r = roots[i - 1] * (1.0 - a) + roots[i] * a
            g = gs[i - 1] * (1.0 - a) + gs[i] * a
            ha = hold_at[i - 1] * (1.0 - a) + hold_at[i] * a
            v = (v + h * (g * GRAVITY + drag * air(x, t))) / (1.0 + h * drag)
            p = _constrain(x + h * v, r, seg, along, held[i], ha)
            if W is not None:
                p = _push(p, W, collide)
                p[:, 0] = r
            v = (p - x) / h
            x = p
        out[i] = x
    return out


def _constrain(p, root, seg, along, held, hold_at):
    """Rounds of: the tie and the held points where they are held, every segment back to its length (a held side does
    not move), a little bending stiffness; then no particle farther from the tie than its thread."""
    K, n = p.shape[:2]
    w = np.ones((K, n))
    w[:, 0] = 0.0
    w[held] = 0.0
    p[:, 0] = root
    p[held] = hold_at[held]
    ev, od = np.arange(0, n - 1, 2), np.arange(1, n - 1, 2)
    for _ in range(ITERS):
        for j in (ev, od):
            d = p[:, j + 1] - p[:, j]
            L = np.linalg.norm(d, axis=2)
            c = ((L - seg[:, None]) / np.maximum(L, 1e-9))[..., None] * d
            wa, wb = w[:, j], w[:, j + 1]
            sw = np.maximum(wa + wb, 1e-9)
            p[:, j] += (wa / sw)[..., None] * c
            p[:, j + 1] -= (wb / sw)[..., None] * c
        mid = 0.5 * (p[:, :-2] + p[:, 2:])
        p[:, 1:-1] += BEND * w[:, 1:-1, None] * (mid - p[:, 1:-1])
    d = p - p[:, :1]
    L = np.linalg.norm(d, axis=2)
    over = (L > along + 1e-6) & (w > 0)
    p[over] = np.repeat(p[:, :1], n, 1)[over] + d[over] * (along[over] / L[over])[:, None]
    p[held] = hold_at[held]
    return p


def _push(p, W, collide):
    """Particles pushed out of the collision shapes W (geom.world) by their deepest penetration, each kind in turn."""
    K, n = p.shape[:2]
    X = p.reshape(-1, 3).copy()
    m = collide.reshape(-1)
    if not m.any():
        return p
    sel = np.where(m)[0]
    for kind, (pen, nrm) in geom.penetration(X[sel], np.full(len(sel), RADIUS), W, None).items():
        k = np.argmax(pen, axis=1)
        rows = np.arange(len(k))
        d = pen[rows, k]
        hit = d > 0.0
        if hit.any():
            X[sel[hit]] += nrm[rows, k][hit] * d[hit][:, None]
    return X.reshape(K, n, 3)


def report(out, lengths):
    """{stretch_pct_max, points}: the most any thread is longer than its length (held at both ends by more than it
    reaches, it stretches straight)."""
    seg = np.linalg.norm(np.diff(out, axis=2), axis=3).sum(axis=2)    # (F, K)
    L = np.asarray(lengths, float)[None, :]
    return {"stretch_pct_max": round(float(np.max(seg / np.maximum(L, 1e-9) - 1.0)) * 100.0, 2),
            "points": int(out.shape[2])}
