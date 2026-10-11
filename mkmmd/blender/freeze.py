"""Freezes at render time (mkmmd.core.freeze, docs/design.md: Shots: Freezes): `frame_set(scene, f)` sets the world to the
frame a freeze holds on `f` and every camera to where its own keys put it on `f`, with the camera the timeline markers bind
on `f` as the scene camera; outside a freeze it is `scene.frame_set(f)`. A render evaluates the animation again and binds
the camera of the markers on the world's frame, so a held camera's action and the markers are set aside (the camera's
values written by hand) until the next `frame_set` or `release()`, which a render loop calls when it ends (and
mkmmd.blender.scene.bind_aspect before it binds the markers). The scene is updated after the cameras move, so what reads
them next (a camera's matrix_world: the bake, a ring's centre seen through the camera) sees them where the render will."""
import json

import bpy

from ..core import freeze as FZ


def windows(sc):
    """The scene's freezes as [[f0, f1], ...] Blender frames (the shots stage keeps them in scene["mk_freeze"])."""
    try:
        return json.loads(sc.get("mk_freeze", "[]"))
    except ValueError:
        return []


def camera_at(sc, f):
    """The camera the timeline markers bind on frame `f` (the scene camera when no marker does)."""
    best = None
    for m in sc.timeline_markers:
        if m.camera is not None and m.frame <= f and (best is None or m.frame > best.frame):
            best = m
    return best.camera if best is not None else sc.camera


_HELD = []                    # (owner, action): camera actions set aside while a freeze holds the world
_MARKS = []                   # (scene, [(name, frame, camera)]): the timeline markers set aside meanwhile


def release():
    """Give the held cameras their actions back, and the scene its markers."""
    while _HELD:
        owner, action = _HELD.pop()
        owner.animation_data.action = action
    while _MARKS:
        sc, marks = _MARKS.pop()
        for name, frame, cam in marks:
            sc.timeline_markers.new(name, frame=frame).camera = cam


def _hold_at(owner, f):
    """Write the values `owner`'s action has on frame `f` onto it (location, rotation, lens, shift, focus ...) and set the
    action aside, so the render does not evaluate it again on the world's frame."""
    ad = getattr(owner, "animation_data", None)
    if ad is None or ad.action is None:
        return
    values = [(fc.data_path, fc.array_index, fc.evaluate(f)) for fc in ad.action.fcurves]
    _HELD.append((owner, ad.action))
    ad.action = None
    for path, index, v in values:
        head, _, attr = path.rpartition(".")
        try:
            target = owner.path_resolve(head) if head else owner
            cur = getattr(target, attr)
        except (AttributeError, ValueError):
            continue
        if hasattr(cur, "__len__") and not isinstance(cur, str):
            cur[index] = v
        else:
            setattr(target, attr, v)


def frame_set(sc, f, wins=None):
    """Go to frame `f` (see the module docstring). Returns the frame the world is at."""
    release()
    src = FZ.source(windows(sc) if wins is None else wins, f)
    sc.frame_set(src)
    if src != f:
        for cam in [o for o in sc.objects if o.type == "CAMERA"]:
            _hold_at(cam, f)
            _hold_at(cam.data, f)
        cam = camera_at(sc, f)
        marks = [(m.name, m.frame, m.camera) for m in sc.timeline_markers]
        if marks:                                         # else they bind the camera of the world's frame
            for m in list(sc.timeline_markers):
                sc.timeline_markers.remove(m)
            _MARKS.append((sc, marks))
        if cam is not None:
            sc.camera = cam
        bpy.context.view_layer.update()
    return src
