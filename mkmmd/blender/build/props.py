"""props: place [[prop]] entries. A prop is a library builder (card = "library:<name>") or a card file (JSON) whose
`builder` names a library builder or whose `source` is a .blend to append from.

[[prop]] keys: name, card, at [x, y, z], yaw (deg), parent (object), slots {slot: "#hex"} (colours). Use points stay
in the prop's local frame; `world(prop, point)` maps them with the prop root's current matrix, and IK targets are
parented to the root so they follow a moving prop."""
import json
import math

import bpy
from mathutils import Euler, Vector

from ..library import props as LIB
from . import BuildError, collection


class Prop:
    def __init__(self, name, root, card):
        self.name, self.root, self.card = name, root, card

    def use(self, kind, name=None):
        items = self.card.get("use", {}).get(kind, [])
        if name is None:
            if len(items) == 1:
                return items[0]
            raise BuildError(f"prop {self.name!r} has {len(items)} {kind} points: name one "
                             f"({[i['name'] for i in items]})")
        for it in items:
            if it["name"] == name:
                return it
        raise BuildError(f"prop {self.name!r} has no {kind} point {name!r} ({[i['name'] for i in items]})")

    def world(self, p):
        return self.root.matrix_world @ Vector(p)

    def world_dir(self, d):
        return (self.root.matrix_world.to_3x3() @ Vector(d)).normalized()

    @property
    def colliders(self):
        return list(self.card.get("colliders", []))


def _append_blend(path, coll, root):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    for o in dst.objects:
        if o is None:
            continue
        coll.objects.link(o)
        if o.parent is None:
            o.parent = root


def run(ctx):
    LIB.load_all()
    coll = collection("Props")
    out = {}
    for spec in ctx.data.get("prop", []):
        name = spec["name"]
        root = bpy.data.objects.new(name, None)
        root.empty_display_size = 0.2
        coll.objects.link(root)
        ref = spec["card"]
        slots = {**ctx.palette, **(spec.get("slots") or {})}      # builders colour by palette slot
        if ref.startswith("library:"):
            key = ref.split(":", 1)[1]
            if key not in LIB.BUILDERS:
                raise BuildError(f"prop {name!r}: no library prop {key!r} (have {sorted(LIB.BUILDERS)})")
            card = LIB.BUILDERS[key](name, coll, root, slots)
        else:
            with open(ctx.path(ref), encoding="utf-8") as fh:
                card = json.load(fh)
            if card.get("builder", "").startswith("library:"):
                card = dict(card, **LIB.BUILDERS[card["builder"].split(":", 1)[1]](name, coll, root, slots))
            elif card.get("source"):
                _append_blend(ctx.path(card["source"]), coll, root)
            else:
                raise BuildError(f"prop {name!r}: card {ref} has neither a library builder nor a source")
        if spec.get("parent"):
            par = bpy.data.objects.get(spec["parent"])
            if par is None:
                raise BuildError(f"prop {name!r}: parent {spec['parent']!r} not found (props are built in order)")
            root.parent = par
        root.location = Vector(spec.get("at", (0.0, 0.0, 0.0)))
        rx, ry, rz = spec.get("rot", (0.0, 0.0, spec.get("yaw", 0.0)))
        root.rotation_euler = Euler((math.radians(float(rx)), math.radians(float(ry)), math.radians(float(rz))))
        ctx.props[name] = Prop(name, root, card)
        out[name] = {"card": ref, "uses": {k: [u["name"] for u in v] for k, v in card.get("use", {}).items()},
                     "colliders": len(card.get("colliders", []))}
        ctx.log("prop", name, ref)
    bpy.context.view_layer.update()
    return out
