"""Freezes at render time (mkmmd.core.freeze, docs/design.md: Shots: Freezes): `frame_set(scene, f)` sets the world to the
frame a freeze holds on `f` and every camera to where its own keys put it on `f`, with the camera the timeline markers bind
on `f` as the scene camera; outside a freeze it is `scene.frame_set(f)`. Screen type is graphics over the picture, not a
thing in the world, so the words on screen go on with the clock like the cameras (a word that lands inside a freeze
lands, one that is up leaves on its own time). A render evaluates the animation again and binds the camera of the markers
on the world's frame, so what is kept on the real frame has its action set aside (its values written by hand), and the
markers with them, until the next `frame_set` or `release()`, which a render loop calls when it ends (and
mkmmd.blender.scene.bind_aspect before it binds the markers). The scene is updated after they move, so what reads them
next (a camera's matrix_world: the bake, a ring's centre seen through the camera, a word's evaluated mesh) sees them
where the render will."""
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
    """Write the values `owner`'s action has on frame `f` onto it (location, rotation, lens, shift, focus, a node's
    value, a modifier's input ...) and set the action aside, so the render does not evaluate it again on the world's
    frame."""
    ad = getattr(owner, "animation_data", None)
    if ad is None or ad.action is None:
        return
    values = [(fc.data_path, fc.array_index, fc.evaluate(f)) for fc in ad.action.fcurves]
    _HELD.append((owner, ad.action))
    ad.action = None
    for path, index, v in values:
        item = path.endswith("]")                         # ...["name"]: a custom property, a modifier's input
        head, _, attr = path.rpartition("[" if item else ".")
        try:
            target = owner.path_resolve(head) if head else owner
            key = bpy.utils.unescape_identifier(attr[1:-2]) if item else attr
            cur = target[key] if item else getattr(target, attr)
        except (AttributeError, ValueError, KeyError):
            continue
        if hasattr(cur, "__len__") and not isinstance(cur, str):
            cur[index] = v
            continue
        v = bool(round(v)) if isinstance(cur, bool) else int(round(v)) if isinstance(cur, int) else v
        if item:
            target[key] = v
        else:
            setattr(target, attr, v)


def _screen_ids(sc, f):
    """The datablocks that animate the screen type on screen on frame `f` (the text stage's objects with `mk_screen`,
    inside their `mk_show` frames): each object and the empties it hangs from, its data, the node groups of its
    modifiers (and the groups inside them), its materials (slots and Set Material nodes) and their node trees."""
    out, seen = [], set()

    def add(idb):
        if idb is None or idb.as_pointer() in seen:
            return False
        seen.add(idb.as_pointer())
        out.append(idb)
        return True

    for ob in sc.objects:
        if "mk_screen" not in ob.keys():
            continue
        show = ob.get("mk_show")
        if show is not None and not int(show[0]) - 1 <= f <= int(show[1]):
            continue
        add(ob)
        add(ob.data)
        p = ob.parent
        while p is not None and p.type == "EMPTY":
            add(p)
            p = p.parent
        mats = [s.material for s in ob.material_slots]
        todo = [m.node_group for m in ob.modifiers if m.type == "NODES"]
        while todo:
            ng = todo.pop()
            if not add(ng):
                continue
            for n in ng.nodes:
                todo.append(getattr(n, "node_tree", None))
                if n.bl_idname == "GeometryNodeSetMaterial":
                    mats.append(n.inputs["Material"].default_value)
        for m in mats:
            if add(m):
                add(m.node_tree)
    return out


def frame_set(sc, f, wins=None):
    """Go to frame `f` (see the module docstring). Returns the frame the world is at."""
    release()
    src = FZ.source(windows(sc) if wins is None else wins, f)
    sc.frame_set(src)
    if src != f:
        for cam in [o for o in sc.objects if o.type == "CAMERA"]:
            _hold_at(cam, f)
            _hold_at(cam.data, f)
        for idb in _screen_ids(sc, f):                    # type over the picture: not of the world, it goes on
            _hold_at(idb, f)
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
