"""Core ops: ping (environment), list (names in the scene), q (evaluate an expression over frames)."""
import math
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from . import scene as S
from .runtime import CTX, op


@op("ping")
def ping(args):
    sc = bpy.context.scene
    return {"blender": bpy.app.version_string, "python": sys.version.split()[0], "numpy": np.__version__,
            "mmd_tools": CTX["mmd_tools"], "file": bpy.data.filepath or None,
            "frame_range": [sc.frame_start, sc.frame_end], "fps": sc.render.fps / sc.render.fps_base,
            "resolution": [sc.render.resolution_x, sc.render.resolution_y]}


@op("list")
def list_names(args):
    kind = args.get("kind", "objects")
    sc = bpy.context.scene
    if kind == "armatures":
        return [{"name": a.name, "root": a.parent.name if a.parent else None, "bones": len(a.pose.bones),
                 "mmd": S.is_mmd(a)} for a in S.armatures()]
    if kind == "bones":
        arm = S.find_armature(args.get("armature"))
        sem = {v: k for k, v in S.semantic_map(arm).items()}
        return [{"name": pb.name, "jp": S.name_j(pb), "semantic": sem.get(pb.name),
                 "parent": pb.parent.name if pb.parent else None} for pb in arm.pose.bones]
    if kind == "semantic":
        arm = S.find_armature(args.get("armature"))
        return S.semantic_map(arm)
    if kind == "morphs":
        arm = S.find_armature(args.get("armature"))
        out = []
        for o in S.model_meshes(arm):
            sk = o.data.shape_keys
            if sk:
                out += [{"mesh": o.name, "name": kb.name} for kb in sk.key_blocks[1:]]
        return out
    if kind == "cameras":
        return [{"name": o.name, "lens": o.data.lens, "loc": o.matrix_world.translation} for o in sc.objects
                if o.type == "CAMERA"]
    if kind == "markers":
        return [{"name": m.name, "frame": m.frame, "camera": m.camera.name if m.camera else None}
                for m in sorted(sc.timeline_markers, key=lambda m: m.frame)]
    if kind == "collections":
        return [{"name": c.name, "objects": len(c.objects)} for c in bpy.data.collections]
    if kind == "actions":
        return [{"name": a.name, "fcurves": len(a.fcurves), "range": list(a.frame_range)} for a in bpy.data.actions]
    if kind == "objects":
        return [{"name": o.name, "type": o.type, "parent": o.parent.name if o.parent else None,
                 "hide_render": o.hide_render} for o in sc.objects]
    raise ValueError(f"unknown kind {kind!r}: armatures bones semantic morphs cameras markers collections actions "
                     f"objects")


def _namespace(arm_name):
    sc = bpy.context.scene
    proj = CTX["project"] or {}
    arms = {}

    def arm(name=None):
        key = name or arm_name
        if key not in arms:
            arms[key] = S.find_armature(key)
        return arms[key]

    def bone(name, armature=None):
        return S.BoneView(arm(armature), name)

    def obj(name):
        return S.ObjView(name)

    def morph(name, armature=None):
        return S.morph_value(arm(armature), name)

    def screen(p, camera=None):
        cam = bpy.data.objects[camera] if camera else sc.camera
        return world_to_camera_view(sc, cam, Vector(p))

    def dist(a, b):
        return (Vector(a) - Vector(b)).length

    def angle(a, b):
        return math.degrees(Vector(a).angle(Vector(b)))

    return {"bpy": bpy, "np": np, "math": math, "Vector": Vector, "scene": sc, "arm": arm, "bone": bone, "obj": obj,
            "morph": morph, "screen": screen, "dist": dist, "angle": angle, "cam": lambda: S.ObjView(sc.camera),
            "fps": proj.get("fps"), "frame0": proj.get("frame0")}


@op("q")
def query(args):
    expr = args["expr"]
    frames = args["frames"]
    code = compile(expr, "<mk q>", "eval")
    ns = _namespace(args.get("armature"))
    sc = bpy.context.scene
    proj = CTX["project"] or {}
    values = []
    for f in frames:
        fi = int(math.floor(f))
        sc.frame_set(fi, subframe=float(f) - fi)
        ns["frame"] = f
        ns["t"] = (f - proj["frame0"]) / proj["fps"] if proj.get("fps") else None
        values.append(eval(code, ns))  # noqa: S307 - the caller's own expression
    return {"frames": frames, "values": values}
