"""burst: pieces (puffy hearts, stars and smoke puffs) flung out from the root at a clip second, easing out to their reach
while they spin and pop in, then shrinking away; drivers on the scene frame (the maths is mkmmd/core/burst.py), so a
freeze holds them mid-air.

    [[prop]]
    name = "blast"
    card = "library:burst"
    at = [0.0, 1.0, 0.6]                  # where it goes off; the pieces fly mostly across the picture (it faces -Y)
    slots = { start = 9.39, count = 30, reach = 1.6, mix = { heart = 0.5, star = 0.3, puff = 0.2 } }

Objects `<name>_<kind><i>` (one mesh per kind: a heart, a five-point star, a cloud puff, each a pillow facing -Y) with
the materials `<name>_heart`, `<name>_star`, `<name>_puff`; the hearts are tagged `mk_heart`, so a silhouette shot can take
them as accents. Options (the `slots`): count (24), seed (1), mix ({kind: share}), the colours `heart` (love), `star`
(gold), `puff` (text) as palette slots or hex, glow (0.3: emission strength; 0 = lit by the scene), and the live root
properties `start`, `reach`, `life`, `spin`, `size`, `amount` (core.burst.PARAMS), which `[[key]]` keys (`amount` shrinks
every piece). Card keys: size, bounds (the whole burst), front "-Y", params, `burst` {objects, accent, count, fps}."""
import re

import bpy

from ....core import burst as BU
from .. import shell as SH
from . import register
from .cafe_kit import clear_drivers
from .hearts import _colour, _material

LIVE = tuple(BU.PARAMS) + ("f0", "fps")                  # the root custom properties the drivers read
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _driver(owner, path, expr, root, index=-1):
    """A scripted driver `expr` on `owner[path]` (a piece): the root's properties and the piece's helper `u`, by name."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    names = sorted(set(NAME.findall(expr)) - BU.FUNCS - {"frame"})
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for nm in names:
            if nm not in LIVE and nm not in BU.HELPERS:
                raise ValueError(f"burst: expression for {path} uses {nm!r}, which is neither a root property nor a helper")
            v = d.variables.new()
            v.name, v.type = nm, "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = root if nm in LIVE else owner
            v.targets[0].data_path = f'["{nm}"]'
        if not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"burst: the driver on {path} is not a Blender simple expression ({expr})")


@register("burst")
def burst(name, coll, root, slots=None):
    slots = dict(slots or {})
    count, seed, mix, P, colours, glow = BU.options(slots)
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    for old in [o for o in bpy.data.objects if any(o.name.startswith(f"{name}_{k}") and o.name[len(name) + len(k) + 1:]
                                                     .isdigit() for k in BU.KINDS)]:
        SH.purge(old.name)
    clear_drivers(root)
    values = dict(P, f0=float(sc.get("mk_frame0", sc.frame_start)), fps=float(fps))
    for k, v in values.items():
        root[k] = float(v)
        root.id_properties_ui(k).update(description=BU.PARAMS[k][2] if k in BU.PARAMS else
                                        {"f0": "the frame of clip second 0", "fps": "frames per second"}[k])
    root["count"], root["seed"] = count, seed
    meshes = {}
    for kind in BU.KINDS:
        if mix.get(kind, 0.0) > 0:
            mat = _material(f"{name}_{kind}", _colour(colours[kind], slots), glow)
            meshes[kind], _ = SH.mesh_data(f"{name}_{kind}", BU.mesh(kind), lambda _role, m=mat: m)
    objs, at = [], {k: 0 for k in BU.KINDS}
    for piece in BU.layout(count, seed, mix):
        kind = piece["kind"]
        o = bpy.data.objects.new(f"{name}_{kind}{at[kind]}", meshes[kind])
        at[kind] += 1
        coll.objects.link(o)
        o.parent = root
        o.rotation_mode = "XYZ"
        if kind == "heart":
            o["mk_heart"] = True
        o["u"] = 0.0
        ex = BU.expressions(piece)
        _driver(o, '["u"]', ex["u"], root)
        for i, ch in enumerate("xyz"):
            _driver(o, "location", ex[ch], root, i)
        _driver(o, "rotation_euler", ex["rx"], root, 0)
        _driver(o, "rotation_euler", ex["ry"], root, 1)
        for i in range(3):
            _driver(o, "scale", ex["s"], root, i)
        objs.append(o)
    half = P["reach"] + P["size"]
    lo, hi = [-half, -BU.FLAT * half, -half], [half, BU.FLAT * half, half]
    return {"size": [round(h - l, 4) for l, h in zip(lo, hi)], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}, "layers": [],
            "slots": colours, "params": P,
            "burst": {"objects": [o.name for o in objs], "accent": f"{name}_heart*", "count": count, "fps": fps}}
