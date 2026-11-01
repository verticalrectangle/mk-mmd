"""Evaluate only what a pass reads. A pass that samples the scene on every frame (a bone's matrix, an object's place) pays
for everything the scene animates on each frame change: on a set dressed with a thousand animated props, over two
thirds of a frame change went to props the pass never looked at (13 ms a frame against 4). Inside `only(objects)`,
every object but those and what they depend on is left out of the view layer's evaluation (hide_viewport), and put back
as it was after; the objects read are evaluated exactly as before (their parents, constraint and driver targets and the
objects their modifiers name come along, followed through)."""
import contextlib

import bpy


def _pointers(struct):
    """The objects a constraint's or modifier's pointer properties name."""
    out = []
    for p in struct.bl_rna.properties:
        if p.type == "POINTER" and getattr(p, "fixed_type", None) is not None and p.fixed_type.identifier == "Object":
            ob = getattr(struct, p.identifier, None)
            if isinstance(ob, bpy.types.Object):
                out.append(ob)
        elif p.type == "COLLECTION" and p.identifier == "targets":     # the Armature constraint's targets
            for t in getattr(struct, "targets", ()):
                ob = getattr(t, "target", None)
                if isinstance(ob, bpy.types.Object):
                    out.append(ob)
    return out


def _driven(idb):
    """The objects the drivers on `idb` read."""
    out = []
    ad = getattr(idb, "animation_data", None)
    if ad is None:
        return out
    for d in ad.drivers:
        for v in d.driver.variables:
            for t in v.targets:
                if isinstance(t.id, bpy.types.Object):
                    out.append(t.id)
    return out


def depends(objects):
    """`objects` and everything their evaluation depends on: parents, constraint targets (on the object and its pose
    bones), the objects their modifiers name, and the objects the drivers on them, their data and shape keys read."""
    seen, todo = set(), [o for o in objects if o is not None]
    while todo:
        o = todo.pop()
        if o.name in seen:
            continue
        seen.add(o.name)
        more = []
        if o.parent is not None:
            more.append(o.parent)
        for c in o.constraints:
            more += _pointers(c)
        if o.pose is not None:
            for pb in o.pose.bones:
                for c in pb.constraints:
                    more += _pointers(c)
        for m in o.modifiers:
            more += _pointers(m)
        more += _driven(o)
        if o.data is not None:
            more += _driven(o.data)
            keys = getattr(o.data, "shape_keys", None)
            if keys is not None:
                more += _driven(keys)
        todo += [x for x in more if x.name not in seen]
    return seen


@contextlib.contextmanager
def only(objects):
    """Inside, the view layer evaluates `objects` and what they depend on (`depends`) and nothing else; on leaving, every
    object it left out is shown again as it was."""
    keep = depends(objects)
    hidden = [o for o in bpy.context.view_layer.objects if o.name not in keep and not o.hide_viewport]
    for o in hidden:
        o.hide_viewport = True
    bpy.context.view_layer.update()
    try:
        yield
    finally:
        for o in hidden:
            o.hide_viewport = False
        bpy.context.view_layer.update()
