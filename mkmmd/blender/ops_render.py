"""render_frames: render frames of one output aspect into a folder, claiming each frame with an empty file first so
several Blender processes can share a range (an empty file left by a crashed process is cleared by the CLI before
it starts again). Frames are named <frame>.png (Blender frame numbers). Shots with a render-time look (`style =
"silhouette"` or `"vector"`, `reflection = {...}`, see mkmmd.blender.styles) are rendered in that look;
args["styles"] = false renders every shot as it is lit. args["demands"] ({frame: [item]},
mkmmd.core.transition.demands) are the layers the transitions and inserts of the project need next to those frames
(plates, mattes, projected points: see mkmmd.blender.transition), drawn on the same terms. Screen type (docs/design.md:
Text, Screen type) is kept off the frame: `screen/<frame>.png` (RGBA, written before the frame) holds it, for `mk post`
to lay over the cut's frame and its effects. args["keep"] ({armature, materials}: `[post] keep`, mkmmd.core.keep) adds
`keep/<frame>.png`, the mask of where those materials are seen (written before the frame; a frame on disk without one
gets it). args["mk"] lists frames mk's own renderer draws (mkmmd.render): this job leaves them alone and draws only the
layers of those in args["mk_layers"] (its share), the screen type first."""
import os
import time

import bpy

from ..core import keep as KP
from ..core import screentype as SRT
from . import freeze as FRZ
from . import scene as S
from . import styles as ST
from . import transition as TRN
from .build.cast import material_name
from .runtime import op

ENGINES = {"eevee": "BLENDER_EEVEE_NEXT", "cycles": "CYCLES", "workbench": "BLENDER_WORKBENCH"}


def kept_materials(keep):
    """The bpy materials `[post] keep` names (args["keep"]: armature, materials, PMX names) on its member's meshes; a name
    the model does not have is an error that lists the ones it has."""
    arm = S.find_armature(keep["armature"])
    have = {}
    for ob in S.model_meshes(arm):
        for s in ob.material_slots:
            if s.material:
                have.setdefault(material_name(s.material), set()).add(s.material)
    lacking = [n for n in keep["materials"] if n not in have]
    if lacking:
        raise ValueError(f"[post] keep: {arm.name} has no material {', '.join(lacking)} (it has {', '.join(sorted(have))})")
    return set().union(*(have[n] for n in keep["materials"]))


@op("render_frames")
def render_frames(args):
    sc = bpy.context.scene
    r = sc.render
    out_dir = args["out"]
    os.makedirs(out_dir, exist_ok=True)
    w, h = args["size"]
    r.resolution_x, r.resolution_y = int(w), int(h)
    r.resolution_percentage = int(args.get("percent", 100))
    if args.get("engine"):
        r.engine = ENGINES[args["engine"]]
    if r.engine == "BLENDER_EEVEE_NEXT" and args.get("samples"):
        sc.eevee.taa_render_samples = int(args["samples"])
    if r.engine == "CYCLES" and args.get("samples"):
        sc.cycles.samples = int(args["samples"])
    r.use_motion_blur = bool(args.get("motion_blur", False))
    if r.use_motion_blur:
        r.motion_blur_shutter = float(args.get("shutter", 0.35))
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = str(args.get("depth", 8))
    r.image_settings.compression = 15
    bound = S.bind_aspect(args["aspect"], sc) if args.get("aspect") else False
    looks = ST.Looks(sc) if args.get("styles", True) else None
    layers = TRN.Layers(sc, looks, args.get("aspect"), out_dir, args["demands"]) if looks and args.get("demands") else None
    freezes = FRZ.windows(sc)
    others, mine = {int(f) for f in args.get("mk") or ()}, {int(f) for f in args.get("mk_layers") or ()}
    t0, done, skipped, drawn, screens, masks = time.time(), 0, 0, 0, 0, 0
    kinds = {}
    keep = args.get("keep")
    kept = kept_materials(keep) if keep else None
    kl = (looks or ST.Looks(sc)) if keep else None                  # the mask is a pass of the scene as it is lit

    def keep_layer(kpath):
        if looks:
            looks.leave()                                            # a flat look of the frame before is left
        kl.keep_layer(kpath, kept)
        return 1
    try:
        for f in args["frames"]:
            f = int(f)
            path = os.path.join(out_dir, f"{f:05d}.png")
            kpath = os.path.join(out_dir, KP.rel(f)) if keep else None
            want_keep = kpath is not None and not os.path.exists(kpath)
            if f in others:                               # mk's renderer draws the frame: its screen type and layers here
                if f not in mine:
                    continue
                layer = os.path.join(out_dir, SRT.layer_rel(f))
                screen = looks is not None and any(looks.screen_live(f, args.get("aspect"))) and not os.path.exists(layer)
                if screen or want_keep or (layers and layers.pending(f)):
                    FRZ.frame_set(sc, f, freezes)
                    if want_keep:
                        masks += keep_layer(kpath)
                    if screen:
                        looks.prepare(f, args.get("aspect"))
                        screens += looks.screen_layer(layer)
                    if layers:
                        drawn += layers.run(f)
                continue
            have = os.path.exists(path)
            if have and not want_keep and not (layers and layers.pending(f)):
                skipped += 1
                continue
            FRZ.frame_set(sc, f, freezes)
            if not have:
                try:
                    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))      # claim, atomically
                except FileExistsError:
                    have = True                           # another process took it between the check and now
            if want_keep:                                 # before the frame: a frame on disk always has its mask
                masks += keep_layer(kpath)
            if have:
                skipped += 1
            else:
                r.filepath = path
                kind = looks.prepare(f, args.get("aspect")) if looks else None
                if kind:
                    kinds[kind] = kinds.get(kind, 0) + 1
                    looks.render(path, layer=os.path.join(out_dir, SRT.layer_rel(f)))
                else:
                    bpy.ops.render.render(write_still=True)
                done += 1
                if done % 10 == 0:
                    print(f"RENDER {args.get('aspect')} {done} frames, {(time.time() - t0) / done:.2f} s/frame", flush=True)
            if layers:
                drawn += layers.run(f)
    finally:
        FRZ.release()
        if looks:
            looks.close()
    return {"rendered": done, "skipped": skipped, "seconds": round(time.time() - t0, 1),
            "s_per_frame": round((time.time() - t0) / max(done, 1), 2), "aspect_bound": bound,
            "engine": r.engine, "samples": sc.eevee.taa_render_samples if r.engine == "BLENDER_EEVEE_NEXT" else None,
            "looks": kinds, "layers": drawn, "screen_layers": screens, "keep_layers": masks}
