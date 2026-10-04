"""scene: an empty scene with the project's fps, frame range (pre-roll included) and render size."""
import bpy

from . import clear_scene


def run(ctx):
    clear_scene()
    sc = bpy.context.scene
    sc.render.fps, sc.render.fps_base = int(round(ctx.fps)), 1.0
    sc.frame_start, sc.frame_end = ctx.start, ctx.end
    outs = ctx.project.get("outputs") or []
    if outs:
        sc.render.resolution_x, sc.render.resolution_y = outs[0]["size"]
    sc.render.resolution_percentage = 100
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.unit_settings.system = "METRIC"
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    return {"frames": [ctx.start, ctx.end], "frame0": ctx.frame0, "fps": ctx.fps}
