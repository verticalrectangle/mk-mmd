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

The objects a [[glitch]] names (docs/design.md: Shots: Glitches) are hidden from its `to` on (a CONSTANT `hide_render` key
added to what they already have), so they stay gone once it has broken them up.
"""
import bpy

from ...core import shotstyle as SS
from ...core import transition as TR
from .. import keys as K
from ..styles import object_records
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
    hidden = _glitched(ctx)
    if hidden:
        out["_glitch_hidden"] = hidden
    return out


def _glitched(ctx):
    """{glitch index: objects hidden}: every object a [[glitch]] names hidden from the frame its window ends on."""
    specs = ctx.data.get("glitch") or []
    specs = [specs] if isinstance(specs, dict) else specs
    recs = object_records(bpy.context.scene) if specs else []
    out = {}
    for i, g in enumerate(specs):
        pats = [g["objects"]] if isinstance(g["objects"], str) else list(g["objects"])
        names = SS.select(recs, pats)
        if not names:
            ctx.log("WARNING", f"glitch {i}: no object matches {pats}, so nothing is hidden after it")
            continue
        f1 = TR.frame_of(g["to"], ctx.fps, ctx.frame0)
        for n in names:
            ob = bpy.data.objects[n]
            before = _base_values(ob, "hide_render", 0, [f1 - 1])[0]          # as it was on the window's last frame
            K.key_prop(ob, "hide_render", [f1 - 1, f1], [before, 1.0], interp="CONSTANT", replace=False)
        out[i] = len(names)
    return out
