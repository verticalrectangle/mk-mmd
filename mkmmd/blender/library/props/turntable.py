"""turntable: a direct-drive record deck with a 7-inch single on it, built procedurally (no downloaded model, no textures):
an aluminium top plate with a rolled edge and the arm well over a dark base on four knurled feet, a 332 mm platter with
four rows of strobe dots on its bevel, a felt slipmat, the single (its grooves and their sheen, a paper label, the 45
adapter in the big centre hole), an S-shaped tonearm on its gimbal (headshell with finger lift, cartridge and stylus,
counterweight), the arm rest, the cue and its lever, an anti-skate knob, the power knob with the strobe lamp, start / stop
and 33 / 45 buttons, the pop-up target light and the pitch slider. Unbranded: no logo, no lettering.

    [[prop]]
    name = "deck"
    card = "library:turntable"
    slots = { label = "pine", rpm = 45 }          # recolour by role (below); starting values of the live properties

    [[key]]                                        # spin up, swing the arm over the lead-in and drop the needle
    target = "deck"
    prop = "rpm"
    keys = [[0.0, 0.0], [0.7, 45.0]]

Live root properties for [[key]] (docs/design.md: Keys): `rpm` (the platter's speed; the keys stage integrates it into
`turns`, the revolutions since clip second 0, which the platter's driver reads, so a speed change never jumps the
record), `arm` (0 on its rest, 1 over the lead-in groove, 1 .. 2 the side played in to the last groove) and `cue` (0 down,
1 lifted): turntable_layout has the kinematics. Drivers only: nothing runs at render time.

Objects `<name>_<part>`: plinth, feet, controls, arm_base (static); platter (turns) carrying mat, record, label (a graphic
layer) and adapter; yoke (swings) and arm (swings and lifts) about the pivot, with the empty `<name>_stylus` at the
stylus tip; cue_bar and cue_lever; the point light `<name>_strobe` (the strobe lamp's red on the dots). The label's
`use.surface` panels ride the platter, so `[[text]] on = "<name>:label"` is printed on the label and turns with it.

Colour roles (`slots` of the prop; a palette slot, a hex, or a blend "gold:2,rose:1"; `ink` and `paper` are the
palette's darkest and lightest): plate (the aluminium), body (the dark base, feet, knobs), metal (arm, spindle, adapter),
mat (the felt), vinyl (the record), label, print (the label's printed rings), cartridge, lamp (the lamps' glow). Option
`strobe` (watts of the strobe light, 0.15)."""
import math

import bpy

from ...keys import set_fcurve
from .. import shell as SH
from . import register
from . import turntable_layout as LAY
from . import turntable_mats as TM
from . import turntable_parts as PT
from .cafe_kit import clear_drivers


def _drive(owner, path, index, expr, prop, root):
    """A scripted driver `expr` on owner.path[index] reading the root's property `prop`; refused unless it is a Blender
    simple expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    v = d.variables.new()
    v.name, v.type = prop, "SINGLE_PROP"
    v.targets[0].id_type, v.targets[0].id = "OBJECT", root
    v.targets[0].data_path = f'["{prop}"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"turntable: the driver on {owner.name}.{path}[{index}] is not a Blender simple expression ({expr})")
    return fc


def _turns(root, rpm):
    """`turns` at a steady `rpm`: 0 at clip second 0, linear both ways (the keys stage rewrites it when `rpm` is keyed)."""
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    fc = set_fcurve(root, '["turns"]', 0, [f0, f0 + fps], [0.0, rpm / 60.0], interp="LINEAR")
    fc.extrapolation = "LINEAR"


def _empty(name, coll, parent, loc, size=0.006):
    SH.purge(name)
    e = bpy.data.objects.new(name, None)
    e.empty_display_type, e.empty_display_size = "ARROWS", size
    coll.objects.link(e)
    e.parent = parent
    e.location = tuple(float(v) for v in loc)
    return e


def _strobe(name, coll, root, colour, watts):
    """The strobe lamp's light: a small red point at the lens by the power knob, shining on the dots."""
    SH.purge(f"{name}_strobe")
    ld = bpy.data.lights.new(f"{name}_strobe", "POINT")
    ld.energy, ld.color, ld.shadow_soft_size = float(watts), tuple(colour), 0.003
    ob = bpy.data.objects.new(f"{name}_strobe", ld)
    coll.objects.link(ob)
    ob.parent = root
    to = (LAY.SPINDLE[0] - LAY.POWER[0], LAY.SPINDLE[1] - LAY.POWER[1])
    n = math.hypot(*to)
    ob.location = (LAY.POWER[0] + to[0] / n * (LAY.POWER_R + 0.0062), LAY.POWER[1] + to[1] / n * (LAY.POWER_R + 0.0062),
                   LAY.Z_PLATE + 0.0045)
    return ob


@register("turntable")
def turntable(name, coll, root, slots=None):
    slots = dict(slots or {})
    start = LAY.options(slots)
    watts = float(slots.get("strobe", LAY.STROBE_W))
    if watts < 0:
        raise ValueError(f"turntable: strobe is watts (>= 0), got {watts:g}")
    pal = LAY.Pal(slots)
    clear_drivers(root)
    for k, (_d, (lo, hi), doc) in LAY.PARAMS.items():
        root[k] = start[k]
        root.id_properties_ui(k).update(description=doc, default=start[k], soft_min=lo, soft_max=hi)
    root["turns"] = 0.0
    root.id_properties_ui("turns").update(description="the platter's revolutions since clip second 0 (from rpm)")
    _turns(root, start["rpm"])
    mats = TM.Materials(name, pal, root)
    objs = {}
    for key, part in PT.parts().items():
        on_platter = key in PT.SPIN
        parent = objs["platter"] if on_platter else root
        loc = (0.0, 0.0, 0.0) if on_platter else part.origin
        objs[key] = SH.mesh_object(f"{name}_{key}", part.mesh, coll, parent, mats.of, loc=loc, rot=part.rot)
    SH.exempt(objs["label"])
    _drive(objs["platter"], "rotation_euler", 2, LAY.EXPR["spin"], "turns", root)
    for key in ("arm", "yoke"):
        clear_drivers(objs[key])
        _drive(objs[key], "rotation_euler", 2, LAY.EXPR["yaw"], "arm", root)
    _drive(objs["arm"], "rotation_euler", 1, LAY.EXPR["lift"], "cue", root)
    bar = objs["cue_bar"]
    _drive(bar, "location", 2, f"{PT.bar_rest_z():.6f} + {LAY.EXPR['bar']}", "cue", root)
    _drive(objs["cue_lever"], "rotation_euler", 1, LAY.EXPR["lever"], "cue", root)
    _empty(f"{name}_stylus", coll, objs["arm"], LAY.STYLUS_LOCAL)
    _strobe(name, coll, root, pal.lin("lamp"), watts)
    return LAY.card(name, pal.roles(), start)
