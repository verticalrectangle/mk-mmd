"""Fisheye cameras in Blender (docs/design.md: Shots: Fisheye shots; the maths is mkmmd.core.fisheye): a camera's lens as
mk projects it, world points through any camera (what world_to_camera_view answers for a perspective one, right for a
fisheye too), and the engine a fisheye needs: Cycles alone draws a panoramic camera, EEVEE and Workbench draw it as a
perspective one. `mk render` refuses a fisheye frame under another engine (`refuse`); `mk look` draws such frames with
Cycles (`cycles`)."""
import contextlib

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from ..core import fisheye as FE
from . import scene as S
from .runtime import CTX

CYCLES = "CYCLES"
FIELDS = ("type", "panorama_type", "fisheye_lens", "fisheye_fov", "sensor_width", "sensor_height", "sensor_fit",
          "shift_x", "shift_y") + tuple(f"fisheye_polynomial_k{i}" for i in range(5))


def fields(cd):
    """The camera data's fields mkmmd.core.fisheye.from_blender reads ({name: value})."""
    return {k: getattr(cd, k) for k in FIELDS}


def is_fisheye(cam):
    """Whether camera object `cam` is panoramic (only Cycles draws it)."""
    return cam is not None and cam.type == "CAMERA" and cam.data.type == "PANO"


def lens(cam, size=None):
    """The fisheye lens (mkmmd.core.fisheye.Lens) of camera object `cam` in a picture `size` (w, h; default the scene's
    resolution), None for a perspective or orthographic camera; a FisheyeError for a panoramic camera mk does not
    project."""
    if not is_fisheye(cam):
        return None
    if size is None:
        r = bpy.context.scene.render
        size = (r.resolution_x, r.resolution_y)
    try:
        return FE.from_blender(fields(cam.data), size)
    except FE.FisheyeError as e:
        raise FE.FisheyeError(f"camera {cam.name!r}: {e}") from None


def to_camera(cam, points):
    """World points (n, 3) in the axes of camera object `cam` (x right, y up, looking down -z)."""
    M = np.array(cam.matrix_world.normalized(), float)
    return (np.asarray(points, float).reshape(-1, 3) - M[:3, 3]) @ M[:3, :3]


def _size(sc):
    return sc.render.resolution_x, sc.render.resolution_y


def output_size(sc, cam):
    """The picture (w, h) a camera is made for: its output's size when it is a shot camera of one of the project's
    outputs (the shot table, mkmmd.blender.build.shots), else the scene's resolution."""
    sizes = {o["name"]: o["size"] for o in (CTX["project"] or {}).get("outputs") or []}
    for e in S.shot_table(sc):
        for out, name in e["cameras"].items():
            if cam is not None and name == cam.name and out in sizes:
                return tuple(sizes[out])
    return _size(sc)


def view(sc, cam, co, size=None):
    """A world point through camera object `cam` as world_to_camera_view gives it: Vector((x, y, depth)), x and y in 0..1
    from the bottom left of a picture `size` (default the scene's resolution: what a render draws). Through a fisheye:
    its place in the fisheye's picture and, as its depth, its distance from the camera, negative where the lens images
    nothing of it (past its field)."""
    size = tuple(size or _size(sc))
    fl = lens(cam, size)
    if fl is None:
        if size == tuple(_size(sc)):
            return world_to_camera_view(sc, cam, Vector(co))
        r, old = sc.render, _size(sc)
        r.resolution_x, r.resolution_y = int(size[0]), int(size[1])
        try:
            return world_to_camera_view(sc, cam, Vector(co))
        finally:
            r.resolution_x, r.resolution_y = old
    uv, d = fl.project(to_camera(cam, [tuple(co)]))
    return Vector((float(uv[0, 0]), float(uv[0, 1]), float(d[0])))


def scale(sc, cam, co):
    """Frame heights a metre spans at world point `co` seen through camera object `cam`: in the camera's plane there for a
    perspective camera (a metre up it, projected), facing the camera for a fisheye (mkmmd.core.fisheye.Lens.scale)."""
    fl = lens(cam, _size(sc))
    if fl is not None:
        return float(fl.scale(to_camera(cam, [tuple(co)]))[0])
    co = Vector(co)
    p = world_to_camera_view(sc, cam, co)
    q = world_to_camera_view(sc, cam, co + cam.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0)))
    w, h = _size(sc)
    return float(np.hypot((q.x - p.x) * w / h, q.y - p.y))


def _cut_cameras(sc, frames):
    """{frame: camera object} the timeline markers bind on each of `frames`, as Blender does (the marker at or before the
    frame; before the first marker, the first marker's camera; no markers, the scene camera)."""
    marks = sorted(((m.frame, m.camera) for m in sc.timeline_markers if m.camera is not None and not m.camera.hide_render),
                   key=lambda fc: fc[0])
    out = {}
    for f in frames:
        cam = sc.camera
        if marks:
            cam = marks[0][1]
            for mf, mc in marks:
                if mf <= f:
                    cam = mc
        out[int(f)] = cam
    return out


def fisheye_frames(sc, frames, aspect=None, demands=None):
    """{camera name: [frames]} of the fisheye cameras a render of `frames` draws with the scene's engine: the cut's camera
    on each (the markers as they are bound) and the camera of a shot a demanded plate (or a glitch's bare frame) of one
    of `demands`' frames is drawn through (mkmmd.core.transition.demands)."""
    out = {}
    for f, cam in _cut_cameras(sc, frames).items():
        if is_fisheye(cam):
            out.setdefault(cam.name, []).append(f)
    if demands and aspect:
        cams = {e["name"]: e["cameras"].get(aspect) for e in S.shot_table(sc)}
        for f, items in demands.items():
            for it in items:
                cam = bpy.data.objects.get(cams.get(it.get("shot")) or "")
                if it.get("kind") in ("plate", "bare") and is_fisheye(cam):
                    out.setdefault(cam.name, []).append(int(f))
    return {k: sorted(set(v)) for k, v in out.items()}


def refuse(sc, frames, aspect=None, demands=None):
    """A RuntimeError, before anything is drawn, when the scene's engine is not Cycles and a render of `frames` (and of the
    layers `demands` asks for them) would look through a fisheye camera: EEVEE and Workbench would draw it as a
    perspective one. The message names the frames, the camera and the fix."""
    if sc.render.engine == CYCLES:
        return
    seen = fisheye_frames(sc, frames, aspect, demands)
    if not seen:
        return
    cam, fs = min(seen.items(), key=lambda kv: kv[1][0])
    where = f"output {aspect!r} " if aspect else ""
    raise RuntimeError(f"{where}frame {fs[0]} ({sum(len(v) for v in seen.values())} in all) is seen through {cam!r}, a "
                       f"fisheye camera, and only Cycles renders a fisheye ({sc.render.engine} would draw it as a "
                       f"perspective camera): set [render] engine = \"cycles\" in mk.toml, or run mk render --engine cycles")


@contextlib.contextmanager
def cycles(sc):
    """Render through the scene camera with Cycles while it is a fisheye and the engine is another (mk look: its preview
    of a fisheye frame), the engine put back afterwards; yields the engine that renders."""
    r = sc.render
    if r.engine == CYCLES or not is_fisheye(sc.camera):
        yield r.engine
        return
    old = r.engine
    r.engine = CYCLES
    try:
        yield CYCLES
    finally:
        r.engine = old
