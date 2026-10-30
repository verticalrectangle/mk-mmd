"""butterfly: one butterfly about 9 cm across, flapping (the maths is mkmmd/core/butterfly.py, the wing patterns
mkmmd/core/butterfly_pattern.py, the materials and meshes butterfly_parts.py). Drivers on the root's properties and the
scene frame: nothing runs at render time, and a freeze holds it mid-beat.

    [[prop]]
    name = "bfly"
    card = "library:butterfly"
    at = [0.25, -0.3, 1.45]
    slots = { style = "morpho", flap = 2.5 }

    [[key]]                          # one wing beat on every kick (and none in between)
    target = "bfly"
    prop = "hit"
    lane = "kick"

Options (the `slots`): style ("swallowtail", "monarch", "morpho", "blue"; "monarch"), size (the span, 0.09 m), way (a
colourway of the style, 0..2), sheen (the iridescence: 1 a morpho's, 0.6 a blue's, 0 none), tile (the pattern's pixels per
wing, 1024), the colour roles (ground, under, ink, spot, accent, sheen, body, eye: palette slots, hexes or blends
"slot:2,slot:1") and the starting values of the live root properties. Live root properties for [[key]]: flap (wing beats a
second; the keys stage integrates it into `beats`), flap_amp (the full stroke, degrees), flap_mid (its middle above flat),
phase (where in its beat the wing is at beat 0: with flap 0 it is the pose the wings rest in), hit (the clip second of the
last hit: one more beat starts then, over hit_time seconds; a [[key]] on a lane keys it to every hit's time).

Objects `<name>_body` (the root's child, scaled to the span; it bobs with the beat), `<name>_wing_L` and `<name>_wing_R`
(the body's children, turning about their hinges); materials `<name>_wing`, `<name>_body`, `<name>_eye`. Front -Y, origin
the middle of the thorax. Card: `use.look` head, body, tip_L, tip_R; `use.anchor` perch (under the feet: `attach` it to a
finger with that offset); `rates` flap -> beats."""
import re

import bpy

from ....core import butterfly as BF
from ...keys import set_fcurve
from .. import shell as SH
from . import butterfly_parts as BP
from . import register
from .cafe_kit import clear_drivers

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _driver(owner, path, index, expr, root):
    """A scripted driver `expr` on owner.path[index] reading the root's properties by name; refused unless it is a Blender
    simple expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for nm in sorted(set(NAME.findall(expr)) - BF.FUNCS - {"frame"}):
        if nm not in BF.NAMES:
            raise ValueError(f"butterfly: the driver on {path} reads {nm!r}, which is not a root property")
        v = d.variables.new()
        v.name, v.type = nm, "SINGLE_PROP"
        v.targets[0].id_type, v.targets[0].id = "OBJECT", root
        v.targets[0].data_path = f'["{nm}"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"butterfly: the driver on {owner.name}.{path}[{index}] is not a Blender simple expression ({expr})")
    return fc


def _beats(root, flap):
    """`beats` at a steady `flap`: 0 at clip second 0, linear both ways (the keys stage rewrites it when `flap` is keyed)."""
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    fc = set_fcurve(root, '["beats"]', 0, [f0, f0 + fps], [0.0, flap], interp="LINEAR")
    fc.extrapolation = "LINEAR"


def _object(name, me, coll, parent):
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.rotation_mode = "XYZ"
    return o


@register("butterfly")
def butterfly(name, coll, root, slots=None):
    slots = dict(slots or {})
    O, P = BF.options(slots)
    style, size, way = O["style"], O["size"], O["way"]
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    clear_drivers(root)
    for k, (_d, unit, doc) in BF.FLAP_PARAMS.items():
        root[k] = P[k]
        root.id_properties_ui(k).update(description=f"{doc} ({unit})", default=P[k])
    root["beats"] = 0.0
    root.id_properties_ui("beats").update(description="wing beats since clip second 0 (from flap)")
    _beats(root, P["flap"])
    root["style"], root["way"] = style, way
    for part in ("body", "wing_L", "wing_R"):                  # a rebuild in the same session: the old parts go first
        SH.purge(f"{name}_{part}")
    meshes = BP.part_meshes(name, style, slots, way, O["sheen"], O["tile"])
    body = _object(f"{name}_body", meshes["body"], coll, root)
    body.scale = (size, size, size)
    wings = {}
    for side in ("L", "R"):
        w = _object(f"{name}_wing_{side}", meshes[f"wing_{side}"], coll, body)
        w.location = BF.hinge(side)
        wings[side] = w
    ex = BF.expressions(f0, fps, size)
    for o in (body, *wings.values()):
        clear_drivers(o)
    _driver(body, "location", 2, ex["bob"], root)
    for side, w in wings.items():
        _driver(w, "rotation_euler", 1, ex[f"wing_{side}"], root)
    bounds = BF.bounds(style, size, P)
    return {"size": [round(h - lo, 5) for lo, h in zip(bounds["min"], bounds["max"])], "origin": "center", "front": "-Y",
            "blocks": False, "bounds": bounds, "layers": [], "use": BF.use_points(style, size), "colliders": [],
            "slots": BF.roles(style, slots, way), "params": P,
            "rates": [{"rate": "flap", "into": "beats", "per": 1.0}],
            "butterfly": {"style": style, "way": way, "size": size, "sheen": O["sheen"], "glow": O["glow"],
                          "objects": [body.name, wings["L"].name, wings["R"].name]}}
