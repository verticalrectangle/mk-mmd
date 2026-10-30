"""record_treadmill: a giant turntable to run in place on, built procedurally (no downloaded model, no textures): a low
rounded-square plinth on four rubber feet, a 1.68 m platter with four rows of strobe dots on its bevel and the spindle,
and a 1.60 m record on it (grooves and their sheen, a raised bead and label area, an original printed label with no
lettering). The platter turns clockwise seen from above, so a character standing on the run spot facing the front has her
feet carried backward. Numbers, meshes, the print and the card: record_treadmill_layout.py; materials:
record_treadmill_mats.py.

    [[prop]]
    name = "record"
    card = "library:record_treadmill"
    slots = { label = "iris", rpm = 45 }          # recolour by role; the starting speed

    [[cast]]
    name = "rin"
    asset = "my_model"
    at = [-0.45, 0.0, 0.16]                        # on the run spot, standing on the record's top

    [pose.rin]
    feet = "record:run"

    [[key]]                                        # spin up over half a second
    target = "record"
    prop = "rpm"
    keys = [[0.0, 0.0], [0.5, 33.3]]

Live root property for [[key]] (docs/design.md: Keys): `rpm` (33.3; the keys stage integrates it into `turns`, the
platter's revolutions since clip second 0, which the platter's driver reads, so a speed change never jumps the record).
A driver only: nothing runs at render time.

Objects `<name>_<part>`: plinth and feet (static); platter (turns, the spindle on it) carrying record and label (a graphic
layer). The label's `use.surface` rides the platter, so `[[text]] on = "<name>:label"` turns with the record (give it
`lift = 0.0003`)."""
import bpy

from ...keys import set_fcurve
from .. import shell as SH
from . import record_treadmill_layout as LAY
from . import record_treadmill_mats as RM
from . import register
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
        raise ValueError(f"{LAY.WHO}: the driver on {owner.name}.{path}[{index}] is not a Blender simple expression "
                         f"({expr})")
    return fc


def _turns(root, rpm):
    """`turns` at a steady `rpm`: 0 at clip second 0, linear both ways (the keys stage rewrites it when `rpm` is keyed)."""
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    fc = set_fcurve(root, '["turns"]', 0, [f0, f0 + fps], [0.0, rpm / 60.0], interp="LINEAR")
    fc.extrapolation = "LINEAR"


@register("record_treadmill")
def record_treadmill(name, coll, root, slots=None):
    slots = dict(slots or {})
    start = LAY.options(slots)
    pal = LAY.Pal(slots)
    pal.checked()
    clear_drivers(root)
    for k, (_d, (lo, hi), doc) in LAY.PARAMS.items():
        root[k] = start[k]
        root.id_properties_ui(k).update(description=doc, default=start[k], soft_min=lo, soft_max=hi)
    root["turns"] = 0.0
    root.id_properties_ui("turns").update(description="the platter's revolutions since clip second 0 (from rpm)")
    _turns(root, start["rpm"])
    mats = RM.Materials(name, pal)
    objs = {}
    for key, part in LAY.parts().items():
        parent = objs["platter"] if key in LAY.SPIN else root
        objs[key] = SH.mesh_object(f"{name}_{key}", part.mesh, coll, parent, mats.of, loc=part.origin, rot=part.rot)
    for key in LAY.GRAPHIC:
        SH.exempt(objs[key])
    clear_drivers(objs["platter"])
    _drive(objs["platter"], "rotation_euler", 2, LAY.EXPR["spin"], "turns", root)
    return LAY.card(name, pal.roles(), start)
