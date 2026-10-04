"""hearts: a field of puffy hearts that rise and sway around a point, animated by drivers on the scene frame (no handler, no
Python at render time; the maths is mkmmd/core/hearts.py, which also lists every root property).

    [[prop]]
    name = "hearts"
    card = "library:hearts"
    at = [0.0, 0.0, 1.4]            # the middle of the column the hearts climb; the field faces -Y like a character
    slots = { count = 16, size = [0.06, 0.16], clear = 0.3, spread = 0.65 }

Every heart is an object `<name>_heart<i>` (all share one mesh: a classic two-lobed heart, rounded, inflated to a pillow whose
face looks along -Y) tagged with the custom property `mk_heart`, so a silhouette shot picks them up as accents:
`accent = ["hearts_heart*"]` (or `["prop:mk_heart"]`), with `keep = [..., "hearts_heart*"]`. Each is born at the bottom of the
column with a pop, climbs, sways and turns a little, shrinks away at the top and is reborn: hearts at every height, never two
births together, never a jump that is not at scale 0. Hearts stay outside the column |x| < `clear` (the prop's own X, the
camera's right when the camera looks along +Y) up to the height `clear_top` above the root, through every sway and turn, so a
figure standing in it is never crossed by a heart in a front view; above `clear_top` the column closes over 0.3 m and the
hearts arch over a head.

Options (the `slots` of the [[prop]]; the colours are palette slots or hex):
  heart   colour of the hearts (love)             glow    emission strength of the lit hearts (0.3; 0 = lit by the scene)
  count   number of hearts (16)                   seed    the layout (1): phases, sides, sizes, drifts
  size    [smallest, largest] width, m (the root properties `size_min` / `size_max`)
  puff    thickness of the pillow as a share of the width (0.42)
  rise, height, spread, fan, clear, clear_top, depth, sway, spin, tilt, pop, fade, amount    the live parameters below
The live parameters are custom properties of the root, read by the drivers every frame; keep them with `[[key]]`
(`target = "hearts", prop = "amount", keys = [[t, v], ...]` swells the field in or out; `spread`, `sway`, `spin` are safe to
key as well; `rise` and `height` are a clock rate: changing them mid-clip moves the hearts, they do not accelerate). The room a
heart keeps beside the clear column is sized at build for the `spin` and `tilt` it is built with (card `hearts.hw`): keying
them higher later can let a turned heart touch the column.

Frame: the root is the middle of the column (z up), the hearts face -Y, x is right seen from the front. Card keys: size,
bounds (the whole field), front "-Y", params, `hearts` {objects, accent, hw (the room factor), fps}.
"""
import re

import bpy

from ....core import hearts as HR
from ....core import palette as PAL
from .. import shell as SH
from . import register
from .cafe_kit import clear_drivers

LIVE = tuple(HR.PARAMS)                                  # the root custom properties the drivers read
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _colour(spec, slots):
    palette = {k: v for k, v in slots.items() if isinstance(v, str) and (v.startswith("#") or k in PAL.PALETTES["rose-pine"])}
    return PAL.linear(PAL.resolve(spec, palette))


def _material(name, rgb, glow):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.32
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.15
    b.inputs["Emission Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Strength"].default_value = glow
    m.diffuse_color = (*rgb, 1.0)
    return m


def _driver(owner, path, expr, root, index=-1):
    """A scripted driver `expr` on `owner[path]` (a heart): the root's properties and the heart's own helper properties, both
    by name, are its variables."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    fcs = res if isinstance(res, list) else [res]
    names = sorted(set(NAME.findall(expr)) - HR.FUNCS - {"frame"})
    for fc in fcs:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for nm in names:
            if nm not in LIVE and nm not in HR.HELPERS:
                raise ValueError(f"hearts: expression for {path} uses {nm!r}, which is neither a root property nor a helper")
            v = d.variables.new()
            v.name = nm
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = root if nm in LIVE else owner
            v.targets[0].data_path = f'["{nm}"]'
        if not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"hearts: the driver on {path} is not a Blender simple expression ({expr})")
    return fcs


def _root_props(root, P, count, seed, puff):
    for k, (default, unit, doc) in HR.PARAMS.items():
        root[k] = float(P[k])
        ui = {} if k == "clear_top" else {"min": 0.0, "soft_min": 0.0}          # (the clear column's top may be below the root)
        root.id_properties_ui(k).update(description=f"{doc}" + (f" ({unit})" if unit else ""), default=float(default), **ui)
    root["count"], root["seed"], root["puff"] = count, seed, puff
    for k, doc in (("count", "number of hearts (fixed at build)"), ("seed", "layout seed (fixed at build)"),
                   ("puff", "pillow thickness / width (fixed at build)")):
        root.id_properties_ui(k).update(description=doc)


@register("hearts")
def hearts(name, coll, root, slots=None):
    slots = dict(slots or {})
    count, seed, P, puff, colour, glow = HR.options(slots)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    rgb = _colour(colour, slots)
    mat = _material(f"{name}_heart", rgb, glow)
    for old in [o for o in bpy.data.objects if o.name.startswith(f"{name}_heart") and o.name[len(name) + 6:].isdigit()]:
        SH.purge(old.name)
    mesh = HR.heart_mesh(puff)
    me, _ = SH.mesh_data(f"{name}_heart", mesh, lambda _role: mat)
    hw = HR.halfwidth(P["spin"], P["tilt"], puff, mesh=mesh)
    if P["spread"] < P["clear"] + hw * P["size_max"] + P["sway"]:
        raise ValueError(f"hearts: spread {P['spread']} leaves no room beyond clear {P['clear']} for hearts of "
                         f"{P['size_max']} m (and a sway of {P['sway']}): spread must exceed "
                         f"{P['clear'] + hw * P['size_max'] + P['sway']:.3f}")
    _root_props(root, P, count, seed, puff)
    clear_drivers(root)
    L = HR.Layout(count, seed)
    objs = []
    for i in range(count):
        o = bpy.data.objects.new(f"{name}_heart{i}", me)
        coll.objects.link(o)
        o.parent = root
        o.rotation_mode = "XYZ"
        o["mk_heart"] = True
        for h in HR.HELPERS:
            o[h] = 0.0
        ex = HR.expressions(L, i, fps, hw)
        for h in HR.HELPERS:
            _driver(o, f'["{h}"]', ex[h], root)
        for ch, (path, idx) in {"x": ("location", 0), "y": ("location", 1), "z": ("location", 2),
                                "rx": ("rotation_euler", 0), "ry": ("rotation_euler", 1), "rz": ("rotation_euler", 2)}.items():
            _driver(o, path, ex[ch], root, idx)
        for idx in range(3):
            _driver(o, "scale", ex["s"], root, idx)
        objs.append(o)
    pad = hw * P["size_max"]
    half = P["spread"] + pad + P["sway"]
    lo = [-half, -(P["depth"] + pad), -0.5 * P["height"] - pad]
    hi = [half, P["depth"] + pad, 0.5 * P["height"] + pad]
    return {"size": [round(h - l, 4) for l, h in zip(lo, hi)], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}, "layers": [],
            "slots": {"heart": colour}, "params": {k: P[k] for k in LIVE},
            "hearts": {"count": count, "seed": seed, "objects": [o.name for o in objs], "accent": f"{name}_heart*",
                       "tag": "mk_heart", "hw": round(hw, 4), "fps": fps}}
