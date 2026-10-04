"""keys: declarative keys on set, prop and object properties (storm flashes, fog, steam, light levels, visibility).

[[key]] keys
  target = "cafe"           a set or prop name (its root), or any object name
  prop = "fog"              a custom property of the target (set and prop cards document theirs), or an RNA data
                            path such as "location" / "hide_render" / "data.energy"
  index = 0                 array index for vector paths
  keys = [[t, v], ...]      clip seconds and values
  interp = "BEZIER" | "LINEAR" | "CONSTANT"
"""
import bpy

from .. import keys as K
from . import BuildError


def _target(ctx, name):
    if name in getattr(ctx, "sets", {}):
        return ctx.sets[name].root
    if name in ctx.props:
        return ctx.props[name].root
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise BuildError(f"key target {name!r}: no set, prop or object of that name")
    return ob


def run(ctx):
    out = {}
    for spec in ctx.data.get("key", []):
        ob = _target(ctx, spec["target"])
        prop = spec["prop"]
        id_data, path = ob, prop
        if prop in ob.keys():
            path = f'["{prop}"]'
        elif prop.startswith("data.") and ob.data is not None:
            id_data, path = ob.data, prop[5:]
        pts = sorted(spec["keys"])
        if not pts:
            raise BuildError(f"[[key]] {spec['target']}.{prop}: no keys")
        try:
            K.key_prop(id_data, path, [ctx.frame(float(t)) for t, _ in pts], [float(v) for _, v in pts],
                       index=int(spec.get("index", 0)), interp=spec.get("interp", "BEZIER"))
        except (TypeError, RuntimeError) as e:
            raise BuildError(f"[[key]] {spec['target']}.{prop}: {e}")
        out[f"{spec['target']}.{prop}"] = len(pts)
    if out:
        ctx.log("keys", len(out))
    return out
