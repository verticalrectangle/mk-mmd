"""Target references used across the build spec:
  [x, y, z]                       a world point
  {prop = "car", point = [..]}    a point in a prop's local frame (follows the prop)
  "car:road"                      a prop use point by name (look / rest / grip / sit / feet)
  "cast:rin" | "cast:rin.head"    another character's eyes (default) or one of their bones (semantic name)
  "camera"                        the active camera"""
import bpy
from mathutils import Vector

from .. import scene as S
from . import BuildError


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
    if isinstance(ref, str):
        if ref == "camera":
            return bpy.context.scene.camera.matrix_world.translation.copy()
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
            for kind in ("look", "rest", "grip", "sit"):
                for u in p.card.get("use", {}).get(kind, []):
                    if u["name"] == use:
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
