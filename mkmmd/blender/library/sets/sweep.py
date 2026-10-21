"""Studio sets: `sweep`, a photo studio's seamless paper sweep."""
import math

import bmesh
import numpy as np

from ....core.palette import resolve
from ..mesh import material, mesh_obj
from . import register


def profile(depth, height, radius, segments):
    """(n, 2) points (y, z) of the sweep's cross-section, front to top: the floor from y = -depth to 0, a quarter circle
    of `radius` up to (radius, radius), then the wall straight up to `height`."""
    a = np.linspace(0.0, math.pi / 2, segments + 1)
    cove = np.stack([radius * np.sin(a), radius * (1.0 - np.cos(a))], 1)
    return np.concatenate([[[-depth, 0.0]], cove, [[radius, height]]])


@register("sweep")
def sweep(name, coll, root, spec, palette):
    """spec (all optional): width (m along x, default 8), depth (the floor in front of the curve, 6), height (the wall, 4),
    radius (the curve from the floor up into the wall, 1.2), color (palette slot or #hex, default "base"), rough (0.8),
    segments (of the curve, 24). The floor lies on z = 0 from y = -depth to 0, curves up and stands as a wall at
    y = radius; the subject stands in front of the curve (toward -y), the camera further toward -y."""
    known = {"name", "kind", "at", "yaw", "width", "depth", "height", "radius", "color", "rough", "segments"}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise ValueError(f"sweep {name!r}: unknown keys {unknown} (known: {sorted(known - {'name', 'kind', 'at', 'yaw'})})")
    w = float(spec.get("width", 8.0))
    depth, height = float(spec.get("depth", 6.0)), float(spec.get("height", 4.0))
    r = float(spec.get("radius", 1.2))
    if min(w, depth, height, r) <= 0.0 or height <= r:
        raise ValueError(f"set {name!r} (sweep): width, depth and radius must be positive and height above radius")
    P = profile(depth, height, r, int(spec.get("segments", 24)))
    bm = bmesh.new()
    left = [bm.verts.new((-w / 2, y, z)) for y, z in P]
    right = [bm.verts.new((w / 2, y, z)) for y, z in P]
    for i in range(len(P) - 1):                       # facing the subject: up on the floor, toward -y on the wall
        f = bm.faces.new((left[i], right[i], right[i + 1], left[i + 1]))
        f.smooth = True
    mat = material(f"{name}_paper", resolve(str(spec.get("color", "base")), palette), float(spec.get("rough", 0.8)))
    mesh_obj(f"{name}_sweep", bm, coll, root, mat)
    return {"kind": "sweep", "paths": {},
            "use": {"look": [{"name": "centre", "point": [0.0, 0.0, 1.0]}],
                    "surface": [{"name": "wall", "center": [0.0, r, (r + height) / 2], "normal": [0.0, -1.0, 0.0],
                                 "up": [0.0, 0.0, 1.0], "size": [w, height - r]}]},
            "colliders": [], "lights": []}
