"""sets: place [[set]] entries (library set builders, mkmmd.blender.library.sets).

[[set]] keys: name, kind (builder name), at [x, y, z], yaw (deg), plus the builder's own keys. The project's palette
comes from [look] palette (default rose-pine-moon) and [look.slots] overrides (ctx.palette)."""
import math

import bpy
from mathutils import Euler, Vector

from ..library import sets as LIB
from . import BuildError, collection


class Set:
    def __init__(self, name, root, card):
        self.name, self.root, self.card = name, root, card

    def world(self, p):
        return self.root.matrix_world @ Vector(p)

    def path_points(self, name):
        if name not in self.card.get("paths", {}):
            raise BuildError(f"set {self.name!r} has no path {name!r} ({sorted(self.card.get('paths', {}))})")
        return [tuple(self.world(p)) for p in self.card["paths"][name]["points"]]


def run(ctx):
    builders = LIB.load_all()
    coll = collection("Sets")
    out = {}
    for spec in ctx.data.get("set", []):
        name, kind = spec["name"], spec["kind"]
        if kind not in builders:
            raise BuildError(f"set {name!r}: no set builder {kind!r} (have {sorted(builders)})")
        root = bpy.data.objects.new(name, None)
        root.empty_display_size = 1.0
        coll.objects.link(root)
        root.location = Vector(spec.get("at", (0.0, 0.0, 0.0)))
        root.rotation_euler = Euler((0.0, 0.0, math.radians(float(spec.get("yaw", 0.0)))))
        sub = bpy.data.collections.new(name)
        coll.children.link(sub)
        card = builders[kind](name, sub, root, spec, ctx.palette)
        ctx.sets[name] = Set(name, root, card)
        out[name] = {"kind": kind, "paths": sorted(card.get("paths", {})),
                     "surfaces": [s["name"] for s in card.get("use", {}).get("surface", [])],
                     "lights": len(card.get("lights", [])), "objects": len(sub.all_objects)}
        ctx.log("set", name, kind)
    bpy.context.view_layer.update()
    return out
