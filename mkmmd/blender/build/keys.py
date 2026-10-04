"""keys: declarative keys on set, prop and object properties (storm flashes, fog, steam, light levels, visibility).

[[key]] keys
  target = "cafe"           a set or prop name (its root), or any object name
  prop = "fog"              a custom property of the target (set and prop cards document theirs), or an RNA data
                            path such as "location" / "hide_render" / "data.energy"
  index = 0                 array index for vector paths
  keys = [[t, v], ...]      clip seconds and values
  interp = "BEZIER" | "LINEAR" | "CONSTANT"
  relative = false          true: the values are offsets added to what the property already does at each key (a
                            value another stage solved or animated, e.g. a hand's grip target), and the existing
                            animation outside the keys' time span is kept
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


def _base_values(id_data, path, index, frames):
    """The property's current value at each frame: its fcurve when animated, else its static value."""
    ad = getattr(id_data, "animation_data", None)
    fc = ad.action.fcurves.find(path, index=index) if ad and ad.action else None
    if fc is not None and len(fc.keyframe_points):
        return [fc.evaluate(f) for f in frames]
    val = id_data.path_resolve(path)
    try:
        val = val[index]
    except TypeError:
        pass
    return [float(val)] * len(frames)


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
        index = int(spec.get("index", 0))
        frames = [ctx.frame(float(t)) for t, _ in pts]
        values = [float(v) for _, v in pts]
        relative = bool(spec.get("relative", False))
        try:
            if relative:
                values = [b + v for b, v in zip(_base_values(id_data, path, index, frames), values)]
            K.key_prop(id_data, path, frames, values, index=index, interp=spec.get("interp", "BEZIER"),
                       replace=not relative)
        except (TypeError, RuntimeError, ValueError) as e:
            raise BuildError(f"[[key]] {spec['target']}.{prop}: {e}")
        out[f"{spec['target']}.{prop}"] = len(pts)
    if out:
        ctx.log("keys", len(out))
    return out
