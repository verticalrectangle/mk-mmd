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
exposure (stops), gamma."""
import math

import bpy
import numpy as np
from mathutils import Vector

from ...core.palette import linear, resolve
from .. import keys as K
from . import BuildError, collection, targets
from .shots import _mount

KINDS = {"point": "POINT", "spot": "SPOT", "area": "AREA", "sun": "SUN"}


def colour_management(ctx):
    look = ctx.data.get("look", {})
    vs = bpy.context.scene.view_settings
    vs.view_transform = look.get("view", "AgX")
    if look.get("contrast"):
        name = look["contrast"]
        vs.look = name if name.startswith("AgX") or vs.view_transform != "AgX" else f"AgX - {name}"
    vs.exposure = float(look.get("exposure", 0.0))
    vs.gamma = float(look.get("gamma", 1.0))
    return {"view": vs.view_transform, "look": vs.look, "exposure": vs.exposure}


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
        ob.matrix_world = q.to_matrix().to_4x4()
        ob.location = world_at
        if mount is not None:
            mw = ob.matrix_world.copy()
            ob.parent = mount
            ob.matrix_world = mw
        if spec.get("keys"):
            pts = sorted(spec["keys"])
            K.set_fcurve(ld, "energy", 0, [ctx.frame(t) for t, _ in pts], [ld.energy * m for _, m in pts])
        out[name] = {"kind": kind, "color": spec.get("color", "text"), "power": ld.energy,
                     "mount": spec.get("mount")}
        ctx.log("light", name, kind)
    return out
