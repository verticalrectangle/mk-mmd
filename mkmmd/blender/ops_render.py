"""render_frames: render frames of one output aspect into a folder, claiming each frame with an empty file first so
several Blender processes can share a range (an empty file left by a crashed process is cleared by the CLI before
it starts again). Frames are named <frame>.png (Blender frame numbers). Shots with a render-time look (`style =
"silhouette"`, `reflection = {...}`, see mkmmd.blender.styles) are rendered in that look; args["styles"] = false
renders every shot as it is lit. args["demands"] ({frame: [item]}, mkmmd.core.transition.demands) are the layers the
transitions and inserts of the project need next to those frames (plates, mattes, projected points: see
mkmmd.blender.transition), drawn on the same terms."""
import os
import time

import bpy

from . import scene as S
from . import styles as ST
from . import transition as TRN
from .runtime import op

ENGINES = {"eevee": "BLENDER_EEVEE_NEXT", "cycles": "CYCLES", "workbench": "BLENDER_WORKBENCH"}


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
    t0, done, skipped, drawn = time.time(), 0, 0, 0
    kinds = {}
    try:
        for f in args["frames"]:
            f = int(f)
            path = os.path.join(out_dir, f"{f:05d}.png")
            have = os.path.exists(path)
            if have and not (layers and layers.pending(f)):
                skipped += 1
                continue
            sc.frame_set(f)
            if not have:
                try:
                    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))      # claim, atomically
                except FileExistsError:
                    have = True                           # another process took it between the check and now
            if have:
                skipped += 1
            else:
                r.filepath = path
                kind = looks.prepare(f, args.get("aspect")) if looks else None
                if kind:
                    kinds[kind] = kinds.get(kind, 0) + 1
                    looks.render(path)
                else:
                    bpy.ops.render.render(write_still=True)
                done += 1
                if done % 10 == 0:
                    print(f"RENDER {args.get('aspect')} {done} frames, {(time.time() - t0) / done:.2f} s/frame", flush=True)
            if layers:
                drawn += layers.run(f)
    finally:
        if looks:
            looks.close()
    return {"rendered": done, "skipped": skipped, "seconds": round(time.time() - t0, 1),
            "s_per_frame": round((time.time() - t0) / max(done, 1), 2), "aspect_bound": bound,
            "engine": r.engine, "samples": sc.eevee.taa_render_samples if r.engine == "BLENDER_EEVEE_NEXT" else None,
            "looks": kinds, "layers": drawn}
