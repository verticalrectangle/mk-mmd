"""Finger curls: how far each finger's first three joints bend toward the palm, as a preset name or a table.

`curls("relaxed")` and `curls({"index": [8, 10], "thumb": [0, 8]})` both give {finger: (a, b, c)} in degrees for every
finger of mkmmd.core.bonemap.FINGERS; the pose stage turns them into bone rotations about each finger's own flexion
axis (docs/design.md: Grips). numpy-free and bpy-free, so it is tested without Blender."""
from . import bonemap

PRESETS = {                    # curl per joint (deg) for (fingers 1-3); the thumb separately
    "flat": ((0, 0, 0), (0, 0, 0)),
    "relaxed": ((14, 22, 14), (8, 10, 8)),
    "curled": ((35, 50, 35), (14, 18, 14)),
    "fist": ((80, 95, 65), (25, 35, 40)),
    "point": ((80, 95, 65), (25, 35, 40)),
}


def curls(spec):
    """{finger: (a, b, c)} degrees of the first three joints of every finger. `spec` is a preset name (PRESETS; "point"
    leaves the index straight) or a table {finger: [a, b, c]} where a finger left out, or a joint left off the end of
    its list, stays straight. Raises ValueError naming what is wrong."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"finger preset {spec!r} (have {sorted(PRESETS)}, or a table {{finger: [a, b, c]}})")
        fing, thumb = PRESETS[spec]
        return {f: (thumb if f == "thumb" else (0.0, 0.0, 0.0) if spec == "point" and f == "index" else fing)
                for f in bonemap.FINGERS}
    if isinstance(spec, dict):
        bad = sorted(set(spec) - set(bonemap.FINGERS))
        if bad:
            raise ValueError(f"fingers {bad}: unknown (have {list(bonemap.FINGERS)})")
        out = {}
        for f in bonemap.FINGERS:
            v = spec.get(f, ())
            v = [v] if isinstance(v, (int, float)) else list(v)
            if len(v) > 3 or not all(isinstance(a, (int, float)) for a in v):
                raise ValueError(f"fingers.{f}: give up to three angles in degrees, got {spec[f]!r}")
            out[f] = tuple(float(a) for a in v) + (0.0,) * (3 - len(v))
        return out
    raise ValueError(f"fingers: a preset name or a table {{finger: [a, b, c]}}, got {spec!r}")
