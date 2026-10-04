"""Basic sets: `test_road`, a flat two-lane S-curve on a ground plane (vehicle and camera tests)."""
import bmesh
import numpy as np

from ....core.path import Path
from ..mesh import material, mesh_obj
from . import register


def ribbon(name, path, width, coll, parent, mat, step=2.0, z=0.0):
    """A flat strip `width` m wide along `path` (core.path.Path), sampled every `step` m."""
    s = np.arange(0.0, path.length + 1e-6, step)
    left = path.offset(s, width / 2, z)
    right = path.offset(s, -width / 2, z)
    bm = bmesh.new()
    vl = [bm.verts.new(tuple(p)) for p in left]
    vr = [bm.verts.new(tuple(p)) for p in right]
    for i in range(len(s) - 1):
        bm.faces.new((vr[i], vr[i + 1], vl[i + 1], vl[i]))
    return mesh_obj(name, bm, coll, parent, mat)


@register("test_road")
def test_road(name, coll, root, spec, palette):
    """spec: length (m, default 600), lane_width (3.6), curve (lateral swing of the S, m, default 40)."""
    L = float(spec.get("length", 600.0))
    sw = float(spec.get("curve", 40.0))
    lw = float(spec.get("lane_width", 3.6))
    pts = np.array([[0.0, 0.0, 0.0], [0.0, -L * 0.25, 0.0], [sw, -L * 0.5, 0.0], [0.0, -L * 0.75, 0.0],
                    [0.0, -L, 0.0]])
    path = Path(pts)
    road = material(f"{name}_asphalt", palette["overlay"], 0.5)
    ground = material(f"{name}_ground", palette["hl_low"], 0.9)
    ribbon(f"{name}_road", path, 2 * lw + 1.0, coll, root, road, z=0.0)
    ribbon(f"{name}_ground", path, 200.0, coll, root, ground, step=10.0, z=-0.02)
    return {"kind": "test_road",
            "paths": {"road": {"points": pts.tolist(), "width": 2 * lw + 1.0,
                               "lanes": [{"name": "R1", "offset": -lw / 2, "dir": 1},
                                         {"name": "L1", "offset": lw / 2, "dir": -1}]}},
            "use": {"look": [{"name": "end", "point": pts[-1].tolist()}]}, "colliders": [], "lights": []}
