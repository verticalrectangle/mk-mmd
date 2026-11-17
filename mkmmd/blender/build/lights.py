"""lights: [[light]] entries and the [look] colour pipeline.

[[light]] keys
  name, kind = "point" | "spot" | "area" | "sun"
  color = "gold"            palette slot or "#hex"
  power = 40                watts (point / spot / area) or strength (sun)
  mount = "car"             rides on a prop, set or object (default: the world)
  at = [x, y, z]            in the mount's frame
  look = target             aim (spot / area / sun); default straight down
  size = 0.1                radius (point / spot) or edge (area, m); spot = {angle = 45, blend = 0.3}
  shadow = true, specular = 1.0, diffuse = 1.0, volume = 0.0
  keys = [[t, multiplier], ...]   intensity over clip time (multiplies power)
[look] keys: palette, slots, view = "AgX" | "Standard" | "Filmic", contrast = "Medium High Contrast" (AgX look),
exposure (stops), gamma; [[look.span]] = {from, to, exposure, contrast, gamma}: a part of the clip seen through another
look (mkmmd.core.lookspan), the scene's view settings keyed CONSTANT so that each frame has its own."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import lookspan as LS
from ...core.palette import linear, resolve
from .. import keys as K
from . import BuildError, collection, targets
from .shots import _mount

KINDS = {"point": "POINT", "spot": "SPOT", "area": "AREA", "sun": "SUN"}
VIEW = ("look", "exposure", "gamma")        # the view settings a [[look.span]] keys


def _contrast(vs, name):
    """Blender's look for a contrast name: with the AgX transform the `AgX - ` prefix is optional."""
    return name if name.startswith("AgX") or vs.view_transform != "AgX" else f"AgX - {name}"


def colour_management(ctx):
    look = ctx.data.get("look", {})
    vs = bpy.context.scene.view_settings
    vs.view_transform = look.get("view", "AgX")
    if look.get("contrast"):
        vs.look = _contrast(vs, look["contrast"])
    vs.exposure = float(look.get("exposure", 0.0))
    vs.gamma = float(look.get("gamma", 1.0))
    out = {"view": vs.view_transform, "look": vs.look, "exposure": vs.exposure}
    spans = _spans(ctx, look.get("span"))
    if spans:
        out["spans"] = spans
    return out


def _spans(ctx, specs):
    """[[look.span]] (mkmmd.core.lookspan) on the scene's view settings, after colour_management wrote [look]'s: each
    span's look is [look]'s with what the span sets, keyed CONSTANT on its first frame, and [look]'s is keyed on the
    frame after it and on the scene's first frame, so every frame outside the spans (the pre-roll too) keeps [look]'s.
    Only what some span changes is keyed; `look`, an enum, is keyed as its value. The spans as applied, [{frames:
    [first, last], look, exposure, gamma}], are kept with [look]'s own (`base`) in scene["mk_look_spans"]: the bake
    captures [look]'s from it, and a freeze shows the real frame's look (mkmmd.blender.freeze)."""
    last = ctx.frame0 + int(round(ctx.duration * ctx.fps)) - 1
    try:
        spans = LS.normalize(specs, ctx.fps, ctx.frame0, last)
    except LS.LookSpanError as e:
        raise BuildError(str(e)) from None
    if not spans:
        return []
    sc = bpy.context.scene
    vs = sc.view_settings
    base = {k: getattr(vs, k) for k in VIEW}
    looks = []
    for s in spans:
        name = _contrast(vs, s["contrast"]) if "contrast" in s else base["look"]
        try:
            vs.look = name
        except TypeError as e:
            why = str(e).split(": ")[-1]
            raise BuildError(f"look span {s['index']}: contrast = {s['contrast']!r}: {why}") from None
        vs.exposure = s.get("exposure", base["exposure"])
        vs.gamma = s.get("gamma", base["gamma"])
        looks.append({k: getattr(vs, k) for k in VIEW})
    steps = {min(ctx.start, spans[0]["from"]): base}   # frame: the look from there on
    for s, lk in zip(spans, looks):
        steps.setdefault(s["to"], base)           # [look]'s after it, unless the next span starts there
        steps[s["from"]] = lk
    frames = sorted(steps)
    for k in VIEW:
        if all(steps[f][k] == base[k] for f in frames):
            continue                              # no span changes it
        was = None
        for f in frames:
            if steps[f][k] != was:
                setattr(vs, k, steps[f][k])
                vs.keyframe_insert(k, frame=f)
                was = steps[f][k]
        for p in sc.animation_data.action.fcurves.find(f"view_settings.{k}").keyframe_points:
            p.interpolation = "CONSTANT"
    for k in VIEW:
        setattr(vs, k, base[k])
    applied = [{"frames": [s["from"], s["to"] - 1], **lk} for s, lk in zip(spans, looks)]
    sc["mk_look_spans"] = json.dumps({"base": base, "spans": applied})
    return applied


def run(ctx):
    out = {"look": colour_management(ctx)}
    coll = collection("Lights")
    sc = bpy.context.scene
    sc.frame_set(ctx.frame0)
    for spec in ctx.data.get("light", []):
        name = spec["name"]
        kind = spec.get("kind", "point")
        if kind not in KINDS:
            raise BuildError(f"light {name!r}: kind must be one of {sorted(KINDS)}")
        ld = bpy.data.lights.new(name, KINDS[kind])
        ld.color = linear(resolve(spec.get("color", "text"), ctx.palette))
        ld.energy = float(spec.get("power", 40.0))
        ld.use_shadow = bool(spec.get("shadow", True))
        ld.specular_factor = float(spec.get("specular", 1.0))
        ld.diffuse_factor = float(spec.get("diffuse", 1.0))
        ld.volume_factor = float(spec.get("volume", 0.0))
        size = spec.get("size", 0.1)
        if kind in ("point", "spot"):
            ld.shadow_soft_size = float(size)
        if kind == "spot":
            sp = spec.get("spot", {})
            ld.spot_size = math.radians(float(sp.get("angle", 45.0)))
            ld.spot_blend = float(sp.get("blend", 0.3))
        if kind == "area":
            ld.shape = "RECTANGLE" if isinstance(size, list) else "SQUARE"
            if isinstance(size, list):
                ld.size, ld.size_y = float(size[0]), float(size[1])
            else:
                ld.size = float(size)
        if kind == "sun":
            ld.angle = math.radians(float(spec.get("angle", 1.0)))
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        mount = _mount(ctx, spec.get("mount"))
        at = Vector(spec.get("at", (0.0, 0.0, 3.0)))
        world_at = (mount.matrix_world @ at) if mount is not None else at
        if spec.get("look") is not None:
            aim = targets.point(ctx, spec["look"])
            q = (aim - world_at).to_track_quat("-Z", "Y")
        else:
            q = Vector((0, 0, -1)).to_track_quat("-Z", "Y")
        mw = Matrix.Translation(world_at) @ q.to_matrix().to_4x4()   # explicit: matrix_world lags .location until a
        if mount is not None:                                         # depsgraph update, so never copy it back
            ob.parent = mount
        ob.matrix_world = mw
        if spec.get("keys"):
            pts = sorted(spec["keys"])
            K.set_fcurve(ld, "energy", 0, [ctx.frame(t) for t, _ in pts], [ld.energy * m for _, m in pts])
        out[name] = {"kind": kind, "color": spec.get("color", "text"), "power": ld.energy,
                     "mount": spec.get("mount")}
        ctx.log("light", name, kind)
    return out
