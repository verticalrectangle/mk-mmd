"""Target references used across the build spec:
  [x, y, z]                       a world point
  {prop = "car", point = [..]}    a point in a prop's local frame (follows the prop)
  {cast = "rin", point = [..]}    a point in a cast member's frame: its model root, at `[[cast]] at`, turned by `yaw` (x its
                                  left, y behind it, z up: the model faces -Y), so a pose, a gaze or a camera written with
                                  it keeps working when the member is moved to another place in the world. Read where the
                                  root stands when the stage asks (a `sit` pose moves its member's root before it reads
                                  its own targets)
  {path = "road:road", s = 640, offset = -6.0 | "fwd1", z = 1.2}   a point beside a set's path: arc length s (m),
                                  offset (m left of the centreline, or a lane name), height above the road
  "car:road"                      a prop use point by name (look / rest / grip / sit / feet / anchor / surface / pose)
  "cast:rin" | "cast:rin.head"    another character's eyes (default) or one of their bones (semantic name)
  "camera"                        the active camera"""
import bpy
import numpy as np
from mathutils import Vector

from .. import scene as S
from . import BuildError


def path_point(ctx, ref):
    """{path = "set:path", s, offset (m or lane name), z}: a point beside a set's path, in world space."""
    set_name, _, path_name = ref["path"].partition(":")
    st = getattr(ctx, "sets", {}).get(set_name)
    if st is None or path_name not in st.card.get("paths", {}):
        raise BuildError(f"target {ref!r}: no path {path_name!r} on set {set_name!r}")
    off = ref.get("offset", 0.0)
    if isinstance(off, str):
        lanes = {ln["name"]: float(ln["offset"]) for ln in st.card["paths"][path_name].get("lanes", [])}
        if off not in lanes:
            raise BuildError(f"target {ref!r}: lane {off!r} not in {sorted(lanes)}")
        off = lanes[off]
    path = st.path(path_name)
    s = float(ref.get("s", 0.0))
    if not 0.0 <= s <= path.length:
        raise BuildError(f"target {ref!r}: s outside the path (0..{path.length:.0f} m)")
    return Vector(tuple(path.offset(np.array([s]), float(off), float(ref.get("z", 0.0)))[0]))


def eyes(member):
    arm = member.arm
    smap = S.semantic_map(arm)
    pts = [arm.matrix_world @ arm.pose.bones[smap[k]].head for k in ("eye.L", "eye.R") if k in smap]
    if not pts:
        pts = [arm.matrix_world @ arm.pose.bones[smap["head"]].tail]
    return sum(pts, Vector()) / len(pts)


def point(ctx, ref, frame=None):
    """World position of a reference at the current scene frame (callers frame_set first when it moves)."""
    if isinstance(ref, (list, tuple)) and len(ref) == 3:
        return Vector(ref)
    if isinstance(ref, dict) and "prop" in ref:
        return ctx.props[ref["prop"]].world(ref.get("point", (0, 0, 0)))
    if isinstance(ref, dict) and "cast" in ref:
        m = ctx.cast.get(ref["cast"])
        if m is None:
            raise BuildError(f"target {ref!r}: no cast member {ref['cast']!r}")
        return m.root.matrix_world @ Vector(ref.get("point", (0, 0, 0)))
    if isinstance(ref, dict) and "path" in ref:
        return path_point(ctx, ref)
    if isinstance(ref, str):
        if ref == "camera":
            cam = bpy.context.scene.camera
            if cam is None:
                raise BuildError("target \"camera\": there is no camera yet (the shots stage makes them), so `camera` is only valid "
                                 "in a [[light]] `look` (the lights stage runs after shots) or in a stage after shots, and "
                                 "the project needs a [[shot]]")
            return cam.matrix_world.translation.copy()
        if ref.startswith("cast:"):
            name, _, bone = ref[5:].partition(".")
            m = ctx.cast.get(name)
            if m is None:
                raise BuildError(f"{ref}: no cast member {name!r}")
            if not bone:
                return eyes(m)
            b = S.resolve_bone(m.arm, bone)
            return m.arm.matrix_world @ m.arm.pose.bones[b].head
        prop, _, use = ref.partition(":")
        if prop in ctx.props:
            p = ctx.props[prop]
            for kind in ("look", "rest", "grip", "sit", "anchor", "surface", "pose"):
                for u in p.card.get("use", {}).get(kind, []):
                    if u["name"] == use:
                        if u.get("object") and u["object"] in bpy.data.objects:     # use points that ride an
                            return bpy.data.objects[u["object"]].matrix_world.translation.copy()   # object
                        if "point" in u:
                            return p.world(u["point"])
                        if "center" in u:
                            return p.world(u["center"])
                        if "hip" in u:
                            return p.world(u["hip"])
                        if "a" in u:
                            return (p.world(u["a"]) + p.world(u["b"])) / 2
            raise BuildError(f"{ref}: prop {prop!r} has no use point {use!r}")
    raise BuildError(f"cannot resolve target {ref!r}")
