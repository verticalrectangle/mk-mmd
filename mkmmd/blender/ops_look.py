"""look: render quick views of a scene (never saved): through the scene camera (the cut, timeline markers
respected), through a named camera, or from preset directions around a target point (a bone, an object, any
expression), relative to a model's facing.

args: frames; views [{name, kind: shot|camera|orbit, camera, target (expr), facing (armature or null), yaw, elev,
dist, lens}]; sizes [{name, w, h}]; engine eevee|workbench|cycles; samples; out (folder); hide [object names];
only [object names: hide every other mesh]; styles (default true: shots with a render-time look, `style = "silhouette"`,
`"vector"` or `reflection`, are rendered in it, see mkmmd.blender.styles); layers {dir, demands} (the layers transitions
and inserts need for the cut's frames, drawn per size into <dir>/<size name>/ before the views: mkmmd.blender.transition;
the CLI composites them over the cut's images). Each image of a shot view reports `screen`: the RGBA layer of its screen
type (<out>/<size name>/screen/<frame>.png, mkmmd.core.screentype) or None; the CLI lays it over the image last, as
`mk post` does."""
import math
import os

import bpy
from mathutils import Vector

from ..core import screentype as SRT
from . import freeze as FRZ
from . import scene as S
from . import styles as ST
from . import transition as TRN
from .ops_core import _namespace
from .runtime import op

ENGINES = {"eevee": "BLENDER_EEVEE_NEXT", "workbench": "BLENDER_WORKBENCH", "cycles": "CYCLES"}


def _facing(arm_name):
    """Unit XY vector the model faces in world (MMD models face -Y in armature space)."""
    if not arm_name:
        return Vector((0.0, -1.0, 0.0))
    arm = S.find_armature(arm_name)
    v = arm.matrix_world.to_3x3() @ Vector((0.0, -1.0, 0.0))
    v.z = 0.0
    return v.normalized() if v.length > 1e-6 else Vector((0.0, -1.0, 0.0))


def _orbit(cam, target, facing, yaw, elev, dist):
    a, e = math.radians(yaw), math.radians(elev)
    d = Vector((facing.x * math.cos(a) - facing.y * math.sin(a), facing.x * math.sin(a) + facing.y * math.cos(a), 0.0))
    loc = target + (d * math.cos(e) + Vector((0, 0, math.sin(e)))) * dist
    cam.location = loc
    cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()


@op("look")
def look(args):
    sc = bpy.context.scene
    r = sc.render
    engine = args.get("engine", "eevee")
    r.engine = ENGINES[engine]
    if engine == "eevee":
        sc.eevee.taa_render_samples = int(args.get("samples", 16))
    elif engine == "cycles":
        sc.cycles.samples = int(args.get("samples", 32))
    elif engine == "workbench":
        sc.display.shading.light = "STUDIO"
        sc.display.shading.color_type = "MATERIAL"
    r.use_motion_blur = False
    r.resolution_percentage = 100
    r.image_settings.file_format = "JPEG"
    r.image_settings.quality = 90
    for name in args.get("hide", []):
        bpy.data.objects[name].hide_render = True
    if args.get("only"):
        keep = set(args["only"])
        for o in sc.objects:
            if o.type == "MESH" and o.name not in keep and not o.hide_render:
                o.hide_render = True
    os.makedirs(args["out"], exist_ok=True)
    ns = _namespace(args.get("armature"))
    views = args["views"]
    looks = ST.Looks(sc) if args.get("styles", True) else None
    markers = [(m.frame, m.camera) for m in sc.timeline_markers]
    freezes = FRZ.windows(sc)
    if any(v["kind"] != "shot" for v in views):
        cd = bpy.data.cameras.new("mk_look")
        cd.clip_start = 0.005
        tmp = bpy.data.objects.new("mk_look", cd)
        sc.collection.objects.link(tmp)
    if args.get("layers") and looks is not None:
        for size in args["sizes"]:
            r.resolution_x, r.resolution_y = int(size["w"]), int(size["h"])
            S.bind_aspect(size["name"], sc)
            layers = TRN.Layers(sc, looks, size["name"], os.path.join(args["layers"]["dir"], size["name"]),
                                args["layers"]["demands"])
            for f in sorted(layers.demands):
                FRZ.frame_set(sc, f, freezes)
                layers.run(f)
        looks.leave()
    out = []
    for v in views:
        if v["kind"] == "shot":                          # restore the cut
            if not sc.timeline_markers and markers:
                for f, cam in markers:
                    sc.timeline_markers.new(f"S{f}", frame=f).camera = cam
        else:                                            # markers would switch the camera back on every frame
            for m in list(sc.timeline_markers):
                sc.timeline_markers.remove(m)
            if v["kind"] == "camera":
                sc.camera = bpy.data.objects[v["camera"]]
            else:
                sc.camera = tmp
                cd.lens = float(v.get("lens", 50.0))
                facing = _facing(v.get("facing"))
                code = compile(v["target"], "<mk look target>", "eval")
        for f in args["frames"]:
            if v["kind"] == "shot":
                FRZ.frame_set(sc, int(f), freezes)
            else:
                sc.frame_set(int(f))
            if v["kind"] == "orbit":
                ns["frame"] = f
                target = Vector(eval(code, ns))  # noqa: S307 - the caller's own expression
                _orbit(tmp, target, facing, float(v.get("yaw", 0)), float(v.get("elev", 10)), float(v.get("dist", 1.0)))
                sc.frame_set(int(f))
            for size in args["sizes"]:
                r.resolution_x, r.resolution_y = int(size["w"]), int(size["h"])
                if v["kind"] == "shot" and S.bind_aspect(size["name"], sc):
                    FRZ.frame_set(sc, int(f), freezes)   # the markers switch the camera on frame change
                elif v["kind"] != "shot":
                    S.show_aspect(size["name"], sc)      # per-output type: only this output's
                path = os.path.join(args["out"], f"{v['name']}_{size['name']}_{int(f):05d}.jpg")
                r.filepath = path
                kind = looks.prepare(int(f), size["name"]) if looks is not None and v["kind"] == "shot" else None
                if looks is not None and kind is None:
                    looks.leave()
                if kind:
                    layer = os.path.join(args["out"], size["name"], SRT.layer_rel(int(f)))
                    if os.path.exists(layer):
                        os.unlink(layer)                  # nothing stale: a frame with no screen type has no layer
                    looks.render(path, layer=layer)
                else:
                    layer = None
                    bpy.ops.render.render(write_still=True)
                out.append({"view": v["name"], "size": size["name"], "frame": int(f), "path": path,
                            "camera": sc.camera.name if sc.camera else None,
                            "screen": layer if layer and os.path.exists(layer) else None})
    FRZ.release()
    if looks is not None:
        looks.close()
    return out
