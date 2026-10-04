"""render_frames: render frames of one output aspect into a folder, claiming each frame with an empty file first so
several Blender processes can share a range (an empty file left by a crashed process is cleared by the CLI before
it starts again). Frames are named <frame>.png (Blender frame numbers). Shots with a render-time look (`style =
"silhouette"`, `reflection = {...}`, see mkmmd.blender.styles) are rendered in that look; args["styles"] = false
renders every shot as it is lit."""
import os
import time

import bpy

from . import scene as S
from . import styles as ST
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
    t0, done, skipped = time.time(), 0, 0
    kinds = {}
    try:
        for f in args["frames"]:
            path = os.path.join(out_dir, f"{int(f):05d}.png")
            if os.path.exists(path):
                skipped += 1
                continue
            open(path, "wb").close()                          # claim
            sc.frame_set(int(f))
            r.filepath = path
            kind = looks.prepare(int(f), args.get("aspect")) if looks else None
            if kind:
                kinds[kind] = kinds.get(kind, 0) + 1
                looks.render(path)
            else:
                bpy.ops.render.render(write_still=True)
            done += 1
            if done % 10 == 0:
                print(f"RENDER {args.get('aspect')} {done} frames, {(time.time() - t0) / done:.2f} s/frame", flush=True)
    finally:
        if looks:
            looks.close()
    return {"rendered": done, "skipped": skipped, "seconds": round(time.time() - t0, 1),
            "s_per_frame": round((time.time() - t0) / max(done, 1), 2), "aspect_bound": bound,
            "engine": r.engine, "samples": sc.eevee.taa_render_samples if r.engine == "BLENDER_EEVEE_NEXT" else None,
            "looks": kinds}
