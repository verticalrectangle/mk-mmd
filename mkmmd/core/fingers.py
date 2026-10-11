"""Finger curls: how far each finger's first three joints bend toward the palm, as a preset name or a table; and how far
the fingers fan apart.

`curls("relaxed")` and `curls({"index": [8, 10], "thumb": [0, 8]})` both give {finger: (a, b, c)} in degrees for every
finger of mkmmd.core.bonemap.FINGERS; `spreads({"spread": 8, ...})` gives {finger: degrees} each finger's base joint turns
away from the middle finger in the plane of the palm (the thumb away from the index). The pose stage turns them into bone
rotations about each finger's own flexion axis and the palm normal (docs/design.md: Grips). bpy-free, so it is tested
without Blender.

A thumb can be `"tuck"` instead of angles (the `point` preset's is): folded over the curled middle finger, as a fist
holds it. Curled about its own flexion axis alone, a thumb lying beside the palm only sticks straight out of it, so
`tuck_thumb` turns it across the palm (a negative spread, toward the index) and curls it, searched per hand from the
rest pose: its tip rests on the middle finger's curled middle and end segments, past the index (else it stands up
beside it), and no point of it comes nearer to the middle and index fingers than they are thick. Every model holds its
thumb at rest its own way; one table of angles tucks one model's thumb and runs another's through its fingers."""
import numpy as np

from . import bonemap

SPREAD = {"index": 1.0, "middle": 0.0, "ring": 0.6, "little": 1.2, "thumb": 0.8}    # share of `spread` each finger fans out
PRESETS = {                    # curl per joint (deg) for (fingers 1-3); the thumb separately ("tuck": `tuck_thumb`)
    "flat": ((0, 0, 0), (0, 0, 0)),
    "relaxed": ((14, 22, 14), (8, 10, 8)),
    "curled": ((35, 50, 35), (14, 18, 14)),
    "fist": ((80, 95, 65), (25, 35, 40)),
    "point": ((80, 95, 65), "tuck"),
}
TUCK_GAP = 0.22                # the tucked thumb's tip rests this share of the middle finger's length from its bones
TUCK_CLEAR = 0.18             # ... the rest of it stays this share away from the middle and index fingers' bones
TUCK_SLACK = 0.04             # ... all to within this share; the least turned of those thumbs wins
TUCK_ACROSS = 0.25            # ... its tip at most this share of the index-middle knuckle spacing short of their midline


def tucked(spec):
    """Whether the shape folds the thumb over the curled middle finger (`tuck_thumb`) instead of curling it by angles."""
    if isinstance(spec, str):
        return spec in PRESETS and PRESETS[spec][1] == "tuck"
    return isinstance(spec, dict) and spec.get("thumb") == "tuck"


def curls(spec):
    """{finger: (a, b, c)} degrees of the first three joints of every finger. `spec` is a preset name (PRESETS; "point"
    leaves the index straight) or a table {finger: [a, b, c], spread} where a finger left out, or a joint left off the end
    of its list, stays straight (`spread` is read by `spreads`). A tucked thumb (`tucked`) is (0, 0, 0) here: the pose
    stage solves it per hand. Raises ValueError naming what is wrong."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"finger preset {spec!r} (have {sorted(PRESETS)}, or a table {{finger: [a, b, c]}})")
        fing, thumb = PRESETS[spec]
        thumb = (0.0, 0.0, 0.0) if thumb == "tuck" else thumb
        return {f: (thumb if f == "thumb" else (0.0, 0.0, 0.0) if spec == "point" and f == "index" else fing)
                for f in bonemap.FINGERS}
    if isinstance(spec, dict):
        bad = sorted(set(spec) - set(bonemap.FINGERS) - {"spread"})
        if bad:
            raise ValueError(f"fingers {bad}: unknown (have {list(bonemap.FINGERS)} and spread)")
        out = {}
        for f in bonemap.FINGERS:
            v = spec.get(f, ())
            if f == "thumb" and v == "tuck":
                out[f] = (0.0, 0.0, 0.0)
                continue
            v = [v] if isinstance(v, (int, float)) else list(v)
            if len(v) > 3 or not all(isinstance(a, (int, float)) for a in v):
                what = "up to three angles in degrees" + (' or "tuck"' if f == "thumb" else "")
                raise ValueError(f"fingers.{f}: give {what}, got {spec[f]!r}")
            out[f] = tuple(float(a) for a in v) + (0.0,) * (3 - len(v))
        return out
    raise ValueError(f"fingers: a preset name or a table {{finger: [a, b, c]}}, got {spec!r}")


def spreads(spec):
    """{finger: degrees} each finger's base joint turns away from the middle finger in the plane of the palm (the thumb
    away from the index): a table's `spread` (deg, the index's; the others fan by SPREAD's shares, the middle finger stays)
    or {finger: deg}; nothing for a preset name or a table without it. Negative closes the fingers together."""
    s = spec.get("spread") if isinstance(spec, dict) else None
    if s is None:
        return {f: 0.0 for f in bonemap.FINGERS}
    if isinstance(s, (int, float)):
        return {f: float(s) * SPREAD[f] for f in bonemap.FINGERS}
    if isinstance(s, dict):
        bad = sorted(set(s) - set(bonemap.FINGERS))
        if bad or not all(isinstance(v, (int, float)) for v in s.values()):
            raise ValueError(f"fingers.spread: degrees, or {{finger: degrees}} for {list(bonemap.FINGERS)}, got {s!r}")
        return {f: float(s.get(f, 0.0)) for f in bonemap.FINGERS}
    raise ValueError(f"fingers.spread: degrees, or {{finger: degrees}}, got {s!r}")


# ---------------------------------------------------------------------------------------------------------- the tuck
def _turn(v, axis, deg):
    """Vectors v (..., 3) turned about the unit `axis` by `deg` (...,) degrees (Rodrigues; positive: right-handed)."""
    t = np.radians(np.asarray(deg, float))[..., None]
    a = np.broadcast_to(axis, v.shape)
    return v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (np.sum(a * v, -1, keepdims=True)) * (1.0 - np.cos(t))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def posed(points, palm, angles, spread_axis=None, spread=0.0):
    """The rest points (n <= 4, 3) of a finger chain (its joints' heads, then its tip) with its joints curled by `angles`
    (deg, one per joint) about the finger's flexion axis (its first segment x the palm normal) and its base first turned
    by `spread` (deg) about `spread_axis`, as the pose stage turns the bones. Angles and spreads may be arrays (k,): k
    poses at once, (k, n, 3)."""
    P = np.asarray(points, float)
    axis = _unit(np.cross(_unit(P[1] - P[0]), palm))
    angles = [np.asarray(a, float) for a in angles]
    k = np.broadcast(*angles, np.asarray(spread, float)).shape
    out = [np.broadcast_to(P[0], k + (3,))]
    total = np.zeros(k)
    for i in range(len(P) - 1):
        total = total + angles[i]
        v = _turn(np.broadcast_to(P[i + 1] - P[i], k + (3,)).copy(), axis, total)    # one axis: the curls add up
        if spread_axis is not None:
            v = _turn(v, spread_axis, np.broadcast_to(spread, k))
        out.append(out[-1] + v)
    return np.stack(out, -2)


def _seg_dist(X, a, b):
    """Distances (...,) of points X (..., 3) to the segment a-b."""
    ab = b - a
    t = np.clip(np.sum((X - a) * ab, -1) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(X - (a + t[..., None] * ab), axis=-1)


def tuck_thumb(thumb, index, middle, palm, index_curls, middle_curls):
    """(spread deg, (a, b, c) deg) that fold the thumb over the curled middle finger, from the rest points (heads, then
    tip) of the thumb, index and middle fingers and the rest palm normal (the fingers curl toward it); the index and middle
    fingers posed by their curls. A grid search: the thumb's tip TUCK_GAP of the middle finger's length from its middle
    and end segments and across it (at most TUCK_ACROSS of the knuckles' spacing short of the line halfway between the
    index and middle knuckles; else it stands up beside the index), every other point of it at least TUCK_CLEAR away
    from both fingers' bones; then the least turning, the base joint costing most."""
    th, ix, md, palm = (np.asarray(x, float) for x in (thumb, index, middle, palm))
    hand = float(np.linalg.norm(np.diff(md, axis=0), axis=1).sum())                  # the middle finger's length
    m = posed(md, palm, middle_curls)
    x = posed(ix, palm, index_curls)
    d = _unit(th[1] - th[0])
    away = th[0] - ix[0]
    away = away - (away @ palm) * palm
    spin = _unit(np.cross(d, _unit(away)))
    S, A, B, C = np.meshgrid(np.arange(-80.0, 21.0, 5.0), np.arange(-20.0, 41.0, 10.0), np.arange(0.0, 91.0, 10.0),
                             np.arange(0.0, 91.0, 10.0), indexing="ij")
    S, A, B, C = S.ravel(), A.ravel(), B.ravel(), C.ravel()
    t = posed(th, palm, (A, B, C), spin, S)                                          # (k, n, 3)
    tip = t[:, -1]
    rest_of = np.concatenate([t[:, 1:-1], 0.5 * (t[:, :-1] + t[:, 1:])], 1)          # its joints and segment middles
    gap = np.minimum(_seg_dist(tip, m[1], m[2]), _seg_dist(tip, m[2], m[3]))
    near = np.min([_seg_dist(rest_of, a, b).min(1) for a, b in zip(np.r_[m[:-1], x[:-1]], np.r_[m[1:], x[1:]])], 0)
    miss = np.abs(gap - TUCK_GAP * hand) + np.maximum(0.0, TUCK_CLEAR * hand - near)    # m from resting, clear
    side = md[0] - ix[0]
    side = side - (side @ palm) * palm                                                # index -> middle, in the palm
    across = (tip - 0.5 * (ix[0] + md[0])) @ _unit(side) + TUCK_ACROSS * np.linalg.norm(side)
    effort = 2.0 * (np.abs(S) + np.abs(A)) + 0.3 * (B + C)     # the base joint and the swing across cost the most
    ok = (miss <= TUCK_SLACK * hand) & (across >= 0.0)
    k = int(np.argmin(np.where(ok, effort, np.inf))) if ok.any() else int(np.argmin(miss - np.minimum(across, 0.0)))
    return float(S[k]), (float(A[k]), float(B[k]), float(C[k]))
