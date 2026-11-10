"""A cable hanging from a point to the floor (a guitar lead from its jack): the control points of a soft curve. numpy only.

The cord leaves its socket along `top_dir`, bends down under its own weight, falls to the floor in a lazy S (it swings out
toward `tail` a little as it falls) and lies on the floor trailing away along `tail`. Heights are over `floor_z`; the
cord's centre line never goes below `floor_z + radius`. The Blender side (mkmmd.blender.build.wear) makes a bevelled
NURBS curve of these points and hooks its top to the jack so that it follows the guitar; the sim stage then swings the
whole cord from this shape (mkmmd.solvers.cable)."""
import math

import numpy as np


def _unit(v, fallback):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.asarray(fallback, float)


def lie(F, tail, side, tail_len, slack):
    """Points of a cord lying on the floor from `F` to `F + tail * tail_len` in loose curves, `slack` (a share) longer
    than the straight run: a wave about every 0.9 m, flat where it leaves `F` and where it ends, wide enough for the
    length asked for (a cord thrown on a floor is never straight, and the slack is what a wearer steps and turns in)."""
    waves = max(1, int(round(float(tail_len) / 0.9)))
    s = np.linspace(0.0, 1.0, 8 * waves + 1)[1:]
    shape = np.sin(2.0 * np.pi * waves * s) * np.sin(np.pi * s) ** 0.5

    def pts(a):
        return F + np.outer(s * tail_len, tail) + np.outer(a * shape, side)

    def over(a):
        return float(np.linalg.norm(np.diff(np.vstack([F, pts(a)]), axis=0), axis=1).sum()) / float(tail_len) - 1.0

    lo, hi = 0.0, float(tail_len)
    if slack <= 0.0:
        return pts(0.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if over(mid) < slack else (lo, mid)
    return pts(0.5 * (lo + hi))


def hang(top, top_dir, floor_z, tail=(0.0, 1.0), radius=0.0032, out=0.05, sway=0.035, reach=0.12, tail_len=0.9,
         slack=0.15):
    """(m, 3) control points from `top` to the far end on the floor.

    top        where the cord leaves the plug (metres, world)
    top_dir    the way it leaves (unit vector; the horizontal part is how it first swings out)
    tail       horizontal direction (x, y) the cord trails away along on the floor
    out        metres the cord goes straight on before it bends down
    sway       sideways swing of the fall (the S), metres
    reach      how far from under the jack the cord meets the floor, along `tail`
    tail_len   how far along `tail` the cord lies on the floor
    slack      the share more cord than that run lying there, in loose curves (`lie`)"""
    T = np.asarray(top, float)
    d = _unit(top_dir, (0.0, 0.0, -1.0))
    tail = np.array([float(tail[0]), float(tail[1]), 0.0])
    tail = _unit(tail, (0.0, 1.0, 0.0))
    side = np.array([-tail[1], tail[0], 0.0])                   # horizontal, across the tail
    dh = np.array([d[0], d[1], 0.0])
    dh = dh / np.linalg.norm(dh) if np.linalg.norm(dh) > 1e-6 else tail
    z_f = float(floor_z) + float(radius)
    H = T[2] - z_f
    if H <= 0.05:
        raise ValueError(f"the jack is {H * 1000:.0f} mm above the floor: no room for a hanging cable")
    P1 = T + d * float(out)
    P2 = P1 + dh * 0.04 + np.array([0.0, 0.0, -0.10])
    F = np.array([T[0], T[1], z_f]) + dh * 0.03 + tail * float(reach)
    mid1 = (P2 + F) / 2.0 + side * float(sway)
    mid2 = np.array([*(0.5 * (P2[:2] + F[:2]) - side[:2] * float(sway) * 0.4), z_f + 0.12 * H])
    pts = [T, P1, P2, mid1, mid2, F, *lie(F, tail, side, float(tail_len), float(slack))]
    P = np.array(pts, float)
    P[1:, 2] = np.maximum(P[1:, 2], z_f)
    return P


def length(points):
    """Polyline length of control points (a lower bound of the curve's length)."""
    P = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
