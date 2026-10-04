"""Blender ops of `mk model`.

model_finish  args: pmx, expected (prefix of Assembled.save_expected, or null), blend (path or null), scale (0.08),
              studio (true), physics_in_blend (true), verify (true), describe (true), height (m, studio framing)
              Imports the PMX the way `mk cast` does, verifies it against the assembled data (positions, weights, morphs,
              UVs, normals, winding, bones, flags, axes, grants, materials, morph panels, display frames, rigid bodies,
              joints), describes it (rig.json content, same as `mk inspect`) and saves a review .blend with a neutral
              studio (rigid body world off, so `mk look` never simulates)."""
import os
import time

import bpy

from . import studio, verify
from .. import ops_inspect as OI
from ..runtime import op


def import_pmx(path, scale=0.08, physics=True, clean=True):
    types = {"MESH", "ARMATURE", "DISPLAY", "MORPHS"}
    if physics:
        types.add("PHYSICS")
    before = set(bpy.data.objects)
    bpy.ops.mmd_tools.import_model(filepath=path, types=types, scale=scale, clean_model=clean, log_level="ERROR")
    new = [o for o in bpy.data.objects if o not in before]
    root = next(o for o in new if getattr(o, "mmd_type", "") == "ROOT")
    arm = next(o for o in root.children_recursive if o.type == "ARMATURE")
    return root, arm


def _reset():
    OI._clear_scene()
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.cameras,
                 bpy.data.lights, bpy.data.shape_keys):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)


@op("model_finish")
def model_finish(args):
    path = os.path.abspath(os.path.expanduser(args["pmx"]))
    scale = float(args.get("scale", 0.08))
    out, t0 = {}, time.time()
    if args.get("expected") and args.get("verify", True):
        exp, meta = verify.load_expected(args["expected"])
        _reset()
        physics = bool(meta["bodies"] or meta["joints"])
        root, arm = import_pmx(path, scale, physics=physics, clean=False)
        res = verify.verify(root, arm, exp, meta)
        md = verify.verify_metadata(root, arm, meta)
        res["numbers"].update(md["numbers"])
        res["problems"] += md["problems"]
        if physics:
            ph = verify.verify_physics(root, arm, meta)
            res["numbers"].update(ph["numbers"])
            res["problems"] += ph["problems"]
        out["verify"] = res
        out["seconds_verify"] = round(time.time() - t0, 2)
    t1 = time.time()
    _reset()
    root, arm = import_pmx(path, scale, physics=bool(args.get("physics_in_blend", True)), clean=True)
    if args.get("describe", True):
        out["rig"] = OI.describe(root, arm, path, scale)
    out["seconds_import"] = round(time.time() - t1, 2)
    sc = bpy.context.scene
    if sc.rigidbody_world is not None:
        sc.rigidbody_world.enabled = False
    sc.frame_start, sc.frame_end, sc.frame_current = 1, 1, 1
    if args.get("studio", True):
        out["studio"] = studio.setup(float(args.get("height", 1.6)))
    blend = args.get("blend")
    if blend:
        blend = os.path.abspath(os.path.expanduser(blend))
        os.makedirs(os.path.dirname(blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
        out["blend"] = blend
    return out


@op("model_studio")
def model_studio(args):
    """args: out (.blend to write), lights (false: the scene brings its own), floors ([[x, y], ...]), height. Adds the
    neutral review studio (grey world, floor discs, optionally key/fill/rim) to the scene `mk model studio` opened and
    saves it elsewhere, so a project's own .blend is left alone."""
    made = studio.setup(float(args.get("height", 1.6)), lights=bool(args.get("lights", False)),
                        floors=[tuple(f) for f in args.get("floors", [(0.0, 0.0)])])
    out = os.path.abspath(os.path.expanduser(args["out"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    return {"out": out, "made": made}
