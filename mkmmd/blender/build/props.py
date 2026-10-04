"""props: place [[prop]] entries and [[scatter]] clutter (docs/design.md: Prop card, PMX props, Placement).

A prop is a library builder (card = "library:<name>"), a card file (JSON) whose `builder` names a library builder or
whose `source` is a .blend to append from, or an MMD accessory model: card = "pmx:PATH" (a .pmx / .pmd file or a folder
with one) or the slug of an asset registered as kind prop (its path is a model, a card file or a folder of either).

[[prop]] keys: name, card, at [x, y, z], yaw (deg), rot [rx, ry, rz] (deg), parent (object), slots {slot: "#hex"}
(colours; a PMX prop keeps its own materials), card_extra {...} (merged over the card: tables key by key, lists of
entries with a `name` by name, anything else replaces), place {on, at, facing, align, clear, avoid, distance, bearing,
seed} (instead of `at`: where it lands on a surface, see mkmmd.core.place), and for PMX props scale and origin. Use points
stay in the prop's local frame; `world(prop, point)` maps them with the prop root's current matrix, and IK targets are
parented to the root so they follow a moving prop.

[[scatter]] keys: name, props [card refs], on, count, region [[u0, v0], [u1, v1]], min_dist, yaw [lo, hi], seed, clear,
avoid, slots, card_extra: `count` props picked from `props` (seeded) land on a surface, named `<name>_<i>`. Scatters are
placed after every [[prop]].

Every prop's card gets `bounds` {min, max} (its visible geometry in its own frame) unless it has them; the stage output
reports size, colliders and, for placed props, where they landed (position, yaw, clearance in mm)."""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Euler, Vector

from ...core import place as PL
from ...core import propcard as PC
from ..library import props as LIB
from . import BuildError, collection, targets
from . import place as PLACE
from . import pmxprop as PMX
from . import form_warn as FORM


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


def _registry(ctx):
    path = os.path.join(os.path.expanduser(ctx.assets or "~/mk-assets"), "registry.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {e["slug"]: e for e in json.load(fh)}


def resolve(ctx, name, ref):
    """(kind, value) of a card reference: ("library", key), ("pmx", path) or ("card", path to a JSON card file)."""
    if ref.startswith("library:"):
        return "library", ref.split(":", 1)[1]
    if ref.startswith("pmx:"):
        return "pmx", PMX.resolve_path(ctx, ref[4:])
    path = ctx.path(ref)
    if os.path.isfile(path):
        return ("pmx", path) if path.lower().endswith(PMX.EXTENSIONS) else ("card", path)
    reg = _registry(ctx)
    entry = reg.get(ref)
    if entry is None:
        have = sorted(s for s, e in reg.items() if e.get("kind") == "prop")
        raise BuildError(f"prop {name!r}: card {ref!r} is not \"library:<name>\", \"pmx:PATH\", a card file or an asset "
                         f"slug of kind prop (props in the registry: {have})")
    if entry.get("kind") != "prop":
        raise BuildError(f"prop {name!r}: asset {ref!r} is a {entry.get('kind')}, not a prop")
    target = entry["path"]
    if os.path.isdir(target) and os.path.isfile(os.path.join(target, "card.json")):
        target = os.path.join(target, "card.json")
    if os.path.isdir(target) or target.lower().endswith(PMX.EXTENSIONS):
        return "pmx", PMX.resolve_path(ctx, target)
    return "card", target


def _build(ctx, coll, spec, name, ref, extra_slots=None):
    """Make the prop `name` from card reference `ref`: (root, card). Nothing is placed yet."""
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.2
    coll.objects.link(root)
    kind, value = resolve(ctx, name, ref)
    slots = {**ctx.palette, **(spec.get("slots") or {}), **(extra_slots or {})}      # builders colour by palette slot
    extras = set(spec) & PMX.KEYS
    if kind != "pmx" and extras:
        raise BuildError(f"prop {name!r}: {sorted(extras)} only apply to pmx props")
    if kind == "library":
        if value not in LIB.BUILDERS:
            raise BuildError(f"prop {name!r}: no library prop {value!r} (have {sorted(LIB.BUILDERS)})")
        card = LIB.BUILDERS[value](name, coll, root, slots)
    elif kind == "pmx":
        if spec.get("slots"):
            ctx.log("WARNING", f"prop {name!r}: slots are ignored, a pmx prop keeps its own materials")
        card = PMX.load(ctx, name, coll, root, value, spec)
    else:
        with open(value, encoding="utf-8") as fh:
            card = json.load(fh)
        if card.get("builder", "").startswith("library:"):
            card = dict(card, **LIB.BUILDERS[card["builder"].split(":", 1)[1]](name, coll, root, slots))
        elif card.get("source"):
            _append_blend(ctx.path(card["source"]), coll, root)
        else:
            raise BuildError(f"prop {name!r}: card {ref} has neither a library builder nor a source")
    card = PC.merge_extra(card, spec.get("card_extra"))
    return root, card


def _measure(prop):
    """Put `bounds`, `size` and `layers` on the card: the visible geometry in the prop's own frame (what the placement
    rules use); a prop without geometry takes them from the card's `size` and `origin`."""
    card = prop.card
    lo, hi = (None, None)
    if "bounds" not in card:
        lo, hi = PLACE.local_bounds(prop.root)
        if not any(h - l for l, h in zip(lo, hi)) and card.get("size"):
            lo, hi = PL.bounds_from_size(card["size"], card.get("origin", "floor_center"))
        card["bounds"] = {"min": [round(x, 4) for x in lo], "max": [round(x, 4) for x in hi]}
        card.setdefault("size", [round(h - l, 4) for l, h in zip(lo, hi)])
    if "layers" not in card:
        card["layers"] = [list(layer) for layer in PLACE.local_layers(prop.root)]


def _report(ref, card):
    return {"card": ref, "size": card.get("size"),
            "uses": {k: [u["name"] for u in v] for k, v in card.get("use", {}).items()},
            "colliders": len(card.get("colliders", [])), "collider_specs": card.get("colliders", [])}


def _parent(root, spec, name):
    if spec.get("parent"):
        par = bpy.data.objects.get(spec["parent"])
        if par is None:
            raise BuildError(f"prop {name!r}: parent {spec['parent']!r} not found (props are built in order)")
        root.parent = par


def _explicit(ctx, root, spec):
    """`at` = [x, y, z] (world) and `yaw` / `rot` (deg); or `at` = {path = "set:path", s, offset, z} (a point beside a set's
    path, targets.path_point), and then `yaw` turns the prop from facing the traffic that comes along the path (yaw 0: its
    front, -Y, looks back along the path at the cars coming)."""
    at = spec.get("at", (0.0, 0.0, 0.0))
    rx, ry, rz = spec.get("rot", (0.0, 0.0, spec.get("yaw", 0.0)))
    if isinstance(at, dict):
        if "path" not in at:
            raise BuildError(f"prop {spec['name']!r}: `at` is [x, y, z] or {{path = \"set:path\", s, offset, z}}")
        root.location = targets.path_point(ctx, at)
        set_name, _, path_name = at["path"].partition(":")
        head = float(ctx.sets[set_name].path(path_name).heading(np.array([float(at.get("s", 0.0))]))[0])
        rz = float(rz) + math.degrees(head) - 90.0
    else:
        root.location = Vector(at)
    root.rotation_euler = Euler((math.radians(float(rx)), math.radians(float(ry)), math.radians(float(rz))))


def _prop(ctx, coll, placer, spec):
    name, ref = spec["name"], spec["card"]
    who = f"prop {name!r}"
    if name in ctx.props:
        raise BuildError(f"{who}: the name is used twice")
    if spec.get("place") is not None and (spec.get("at") is not None or spec.get("rot") is not None):
        raise BuildError(f"{who}: give `place` or `at`/`rot`, not both (yaw is the rule's default facing)")
    root, card = _build(ctx, coll, spec, name, ref)
    _parent(root, spec, name)
    prop = Prop(name, root, card)
    _measure(prop)
    res = None
    if spec.get("place") is None:
        _explicit(ctx, root, spec)
    else:
        res = placer.solve(prop, dict(spec["place"]), who, default_yaw=float(spec.get("yaw", 0.0)))
        placer.apply(root, res)
    ctx.props[name] = prop
    rep = _report(ref, card)
    if res is not None:
        rep["placement"] = res.report()
        placer.log(res)
    ctx.log("prop", name, ref)
    FORM.guard(ctx, prop, ref)
    return rep


def _scatter(ctx, coll, placer, spec):
    name = spec.get("name")
    who = f"scatter {name!r}"
    if not name:
        raise BuildError("scatter: every [[scatter]] needs a name")
    refs = spec.get("props")
    if not refs or not isinstance(refs, list):
        raise BuildError(f"{who}: props must be a list of card references")
    count = int(spec.get("count", 0))
    if count < 1:
        raise BuildError(f"{who}: count must be at least 1")
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 1000003)        # which card, apart from the placing stream
    picks = [refs[int(rng.integers(len(refs)))] for _ in range(count)]
    built = []
    for i, ref in enumerate(picks):
        item = f"{name}_{i}"
        if item in ctx.props:
            raise BuildError(f"{who}: the prop name {item!r} is used twice")
        root, card = _build(ctx, coll, spec, item, ref)
        prop = Prop(item, root, card)
        _measure(prop)
        built.append((item, ref, prop))
    results = placer.solve_scatter([b[2] for b in built], spec, who)
    out = {}
    for (item, ref, prop), res in zip(built, results):
        placer.apply(prop.root, res)
        ctx.props[item] = prop
        rep = _report(ref, prop.card)
        rep["placement"] = res.report()
        rep["scatter"] = name
        out[item] = rep
        placer.log(res)
        ctx.log("prop", item, ref)
        FORM.guard(ctx, prop, ref)
    return out


def run(ctx):
    LIB.load_all()
    coll = collection("Props")
    placer = PLACE.Placer(ctx)
    out = {}
    for spec in ctx.data.get("prop", []):
        out[spec["name"]] = _prop(ctx, coll, placer, spec)
    for spec in ctx.data.get("scatter", []):
        out.update(_scatter(ctx, coll, placer, spec))
    bpy.context.view_layer.update()
    return out
