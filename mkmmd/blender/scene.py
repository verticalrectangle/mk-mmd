"""Scene helpers used by every Blender-side op: find armatures, resolve semantic bone names, world-space views."""
import math

import bpy
from mathutils import Vector

from ..core import bonemap


def depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def armatures():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def name_j(pb):
    mb = getattr(pb, "mmd_bone", None)
    return (getattr(mb, "name_j", "") or "") if mb is not None else ""


def is_mmd(arm):
    return any(name_j(pb) for pb in arm.pose.bones[:20])


def find_armature(name=None):
    """By exact name, then by case-insensitive substring of the armature or its parent (the MMD root);
    with no name, the only (MMD) armature in the scene."""
    arms = armatures()
    if name:
        for a in arms:
            if a.name == name:
                return a
        low = name.lower()
        hits = [a for a in arms if low in a.name.lower() or (a.parent and low in a.parent.name.lower())]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"armature {name!r}: {len(hits)} matches among {[a.name for a in arms]}")
    mmd = [a for a in arms if is_mmd(a)] or arms
    if len(mmd) == 1:
        return mmd[0]
    raise KeyError(f"{len(mmd)} armatures in the scene, name one: {[a.name for a in mmd]}")


_MAPS = {}


def semantic_map(arm):
    key = (arm.name, len(arm.pose.bones))
    if key not in _MAPS:
        _MAPS[key] = bonemap.build_map({pb.name: name_j(pb) for pb in arm.pose.bones})
    return _MAPS[key]


def resolve_bone(arm, name):
    """Blender bone name for a Blender name, a semantic name (`wrist.R`) or a PMX name (右手首)."""
    if name in arm.pose.bones:
        return name
    m = semantic_map(arm)
    if name in m:
        return m[name]
    nn = bonemap.norm(name)
    for pb in arm.pose.bones:
        if bonemap.norm(name_j(pb)) == nn:
            return pb.name
    raise KeyError(f"bone {name!r} not in {arm.name}; list them with `mk q --list bones`")


class BoneView:
    """World-space view of a posed bone at the current frame."""

    def __init__(self, arm, name):
        self.arm = arm
        self.name = resolve_bone(arm, name)

    def _ev(self):
        ae = self.arm.evaluated_get(depsgraph())
        return ae, ae.pose.bones[self.name]

    @property
    def head(self):
        ae, pb = self._ev()
        return ae.matrix_world @ pb.head

    @property
    def tail(self):
        ae, pb = self._ev()
        return ae.matrix_world @ pb.tail

    @property
    def center(self):
        return (self.head + self.tail) / 2

    @property
    def dir(self):
        return (self.tail - self.head).normalized()

    @property
    def length(self):
        return (self.tail - self.head).length

    @property
    def matrix(self):
        ae, pb = self._ev()
        return ae.matrix_world @ pb.matrix

    @property
    def quat(self):
        return self.matrix.to_quaternion()

    @property
    def local_quat(self):
        return self._ev()[1].rotation_quaternion.copy()

    def __repr__(self):
        return f"<bone {self.name}>"


class ObjView:
    def __init__(self, name):
        self.ob = bpy.data.objects[name] if isinstance(name, str) else name
        self.name = self.ob.name

    def _ev(self):
        return self.ob.evaluated_get(depsgraph())

    @property
    def matrix(self):
        return self._ev().matrix_world.copy()

    @property
    def loc(self):
        return self.matrix.translation.copy()

    @property
    def quat(self):
        return self.matrix.to_quaternion()

    @property
    def euler(self):
        return Vector([math.degrees(a) for a in self.matrix.to_euler()])

    @property
    def dims(self):
        return self.ob.dimensions.copy()

    @property
    def bbox(self):
        m = self.matrix
        pts = [m @ Vector(c) for c in self.ob.bound_box]
        return [Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])]

    @property
    def visible(self):
        return not self.ob.hide_render

    def __repr__(self):
        return f"<object {self.name}>"


def model_meshes(arm):
    """Mesh objects deformed by `arm` (children of its MMD root, or of the armature itself)."""
    root = arm.parent or arm
    out = []
    for o in root.children_recursive:
        if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers):
            out.append(o)
    return out


def morph_value(arm, name):
    for o in model_meshes(arm):
        sk = o.data.shape_keys
        if sk and name in sk.key_blocks:
            return sk.key_blocks[name].value
    raise KeyError(f"morph {name!r} not found on {arm.name}")


def shot_table(sc=None):
    """The shot table `mk build` stores on the scene: [{name, from, to, cameras: {aspect: camera}}]."""
    import json
    sc = sc or bpy.context.scene
    return json.loads(sc.get("mk_shots", "[]"))


def show_aspect(aspect, sc=None):
    """Render only the objects made for this output: an object with the custom property `mk_aspect` (per-output type, see
    the text stage) is hidden from the render of every other output. Screen type (`mk_screen`) is not touched: it is hidden
    from every scene render and drawn, for its own output, by the screen layer (mkmmd.blender.styles). Returns how many
    objects it hid."""
    sc = sc or bpy.context.scene
    hidden = 0
    for ob in sc.objects:
        if "mk_aspect" in ob.keys() and "mk_screen" not in ob.keys():
            ob.hide_render = ob["mk_aspect"] != aspect
            hidden += int(ob.hide_render)
    return hidden


def bind_aspect(aspect, sc=None):
    """Point every shot marker at the given output aspect's camera and show the objects made for that output only
    (`show_aspect`). Returns False when the scene has no shot table or no cameras for that aspect (its markers are left
    alone)."""
    sc = sc or bpy.context.scene
    show_aspect(aspect, sc)
    by = {s["name"]: s["cameras"].get(aspect) for s in shot_table(sc)}
    if not any(by.values()):
        return False
    for m in sc.timeline_markers:
        cam = by.get(m.name)
        if cam and cam in bpy.data.objects:
            m.camera = bpy.data.objects[cam]
    return True
