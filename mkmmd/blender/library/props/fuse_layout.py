"""Layout of the cartoon fuse (pure Python + numpy, no bpy: importable from tests): its options, the default curl, the
spark's spiky ball and the card.

Frame (metres): the root is where the fuse leaves its host, z up; the cord runs from the root through `points` to the tip
(the first point is the root). Three root properties drive it, keyed with `[[key]]` (docs/design.md: Prop card, the
library table): `grow` (how much of the cord has sprouted from the root, 0..1, along its length), `burn` (how much has
burned down from the tip, 0..1) and `lit` (the spark's size, 0..1). The cord shows from the root up to min(grow, 1 - burn)
of its length and the spark sits at that end, so a lit fuse burns down toward its host.

Colour roles (`slots` of the prop: a palette slot or a hex): cord (text: the rope), spark (gold: the spiky star at the
burning end), cap (base: the socket at the root)."""
import math

import numpy as np

CURL = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.10), (0.025, -0.02, 0.19), (0.085, -0.035, 0.245), (0.15, -0.03, 0.225),
        (0.17, -0.015, 0.16))                               # a cartoon fuse: up, over toward the front, curling down
RADIUS = 0.014                                              # the cord's radius
CAP = (0.032, 0.026)                                        # the socket at the root: radius, height
SPARK = 0.05                                                # the spark's spike length (centre to tip) at lit = 1
ROLES = {"cord": "text", "spark": "gold", "cap": "base"}
PARAMS = {"grow": (1.0, "how much of the cord has sprouted from the root, 0..1 of its length"),
          "burn": (0.0, "how much of the cord has burned down from the tip, 0..1 of its length"),
          "lit": (0.0, "the spark's size, 0..1 (0: no spark)")}
MIN_SCALE = 0.0005                                          # never exactly 0: a singular matrix upsets what inverts it


def options(slots):
    """(points (n, 3), radius, {param: starting value}) from the `slots` of the [[prop]], checked: a bad value is a
    ValueError that says what to change."""
    slots = slots or {}
    pts = slots.get("points", CURL)
    try:
        P = np.asarray(pts, float)
    except (TypeError, ValueError):
        raise ValueError(f"fuse: points must be [[x, y, z], ...] in metres, got {pts!r}") from None
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError(f"fuse: points must be at least two [x, y, z], got {pts!r}")
    if np.linalg.norm(P[0]) > 1e-9:
        raise ValueError(f"fuse: the first point is the root, [0, 0, 0] (got {P[0].tolist()})")
    if (np.linalg.norm(np.diff(P, axis=0), axis=1) < 1e-4).any():
        raise ValueError("fuse: two consecutive points coincide")
    r = slots.get("radius", RADIUS)
    if not isinstance(r, (int, float)) or not 0.001 <= r <= 0.1:
        raise ValueError(f"fuse: radius is metres, 0.001 .. 0.1 (got {r!r})")
    start = {}
    for k, (default, _doc) in PARAMS.items():
        v = slots.get(k, default)
        if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            raise ValueError(f"fuse: {k} is 0..1 (got {v!r})")
        start[k] = float(v)
    return P, float(r), start


def spark_mesh(length=SPARK):
    """The spark: an icosphere whose twelve corner vertices are drawn out into spikes, so it reads as a star from every
    side. (vertices (42, 3), triangles (80, 3)), centred on the origin, spikes `length` long, outward winding."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7),
         (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    mid = {}

    def half(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            v = V[a] + V[b]
            V.append(v / np.linalg.norm(v))
            mid[k] = len(V) - 1
        return mid[k]

    T = []
    for a, b, c in F:
        ab, bc, ca = half(a, b), half(b, c), half(c, a)
        T += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    V = np.array(V)
    scale = np.full(len(V), 0.38 * length)                     # the body between the spikes
    scale[:12] = length                                         # the twelve corners are the spikes' tips
    return V * scale[:, None], np.array(T, np.int64)


def length(points):
    """The length of the polyline through `points` (the cord, a smooth curve through them, is close to it)."""
    P = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def card(name, slots, points, radius):
    """The prop card (docs/design.md: Prop card): look points `base` (the root) and `tip` (the end of the cord, where the
    spark is lit), bounds round the cord, the cap and a full spark."""
    P = np.asarray(points, float)
    pad = max(radius, SPARK)
    lo = np.minimum(P.min(0) - pad, [-CAP[0], -CAP[0], -CAP[1]])
    hi = np.maximum(P.max(0) + pad, [CAP[0], CAP[0], 0.0])
    looks = [{"name": "base", "point": [0.0, 0.0, 0.0]}, {"name": "tip", "point": [round(float(v), 5) for v in P[-1]]}]
    return {"size": [round(float(h - l), 4) for l, h in zip(lo, hi)], "origin": "base", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]}, "layers": [],
            "slots": dict(slots or {}), "use": {"look": looks},
            "params": {k: v[1] for k, v in PARAMS.items()}, "fuse": {"length": round(length(P), 4), "cord": f"{name}_cord",
                                                                    "spark": f"{name}_spark", "cap": f"{name}_cap"}}
