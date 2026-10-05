"""Finger curls: how far each finger's first three joints bend toward the palm, as a preset name or a table; and how far
the fingers fan apart.

`curls("relaxed")` and `curls({"index": [8, 10], "thumb": [0, 8]})` both give {finger: (a, b, c)} in degrees for every
finger of mkmmd.core.bonemap.FINGERS; `spreads({"spread": 8, ...})` gives {finger: degrees} each finger's base joint turns
away from the middle finger in the plane of the palm (the thumb away from the index). The pose stage turns them into bone
rotations about each finger's own flexion axis and the palm normal (docs/design.md: Grips). numpy-free and bpy-free, so it
is tested without Blender."""
from . import bonemap

SPREAD = {"index": 1.0, "middle": 0.0, "ring": 0.6, "little": 1.2, "thumb": 0.8}    # share of `spread` each finger fans out
PRESETS = {                    # curl per joint (deg) for (fingers 1-3); the thumb separately
    "flat": ((0, 0, 0), (0, 0, 0)),
    "relaxed": ((14, 22, 14), (8, 10, 8)),
    "curled": ((35, 50, 35), (14, 18, 14)),
    "fist": ((80, 95, 65), (25, 35, 40)),
    "point": ((80, 95, 65), (25, 35, 40)),
}


def curls(spec):
    """{finger: (a, b, c)} degrees of the first three joints of every finger. `spec` is a preset name (PRESETS; "point"
    leaves the index straight) or a table {finger: [a, b, c], spread} where a finger left out, or a joint left off the end
    of its list, stays straight (`spread` is read by `spreads`). Raises ValueError naming what is wrong."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"finger preset {spec!r} (have {sorted(PRESETS)}, or a table {{finger: [a, b, c]}})")
        fing, thumb = PRESETS[spec]
        return {f: (thumb if f == "thumb" else (0.0, 0.0, 0.0) if spec == "point" and f == "index" else fing)
                for f in bonemap.FINGERS}
    if isinstance(spec, dict):
        bad = sorted(set(spec) - set(bonemap.FINGERS) - {"spread"})
        if bad:
            raise ValueError(f"fingers {bad}: unknown (have {list(bonemap.FINGERS)} and spread)")
        out = {}
        for f in bonemap.FINGERS:
            v = spec.get(f, ())
            v = [v] if isinstance(v, (int, float)) else list(v)
            if len(v) > 3 or not all(isinstance(a, (int, float)) for a in v):
                raise ValueError(f"fingers.{f}: give up to three angles in degrees, got {spec[f]!r}")
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
