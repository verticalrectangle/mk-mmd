"""paper_scissors: a child's craft scissors of card on a brass split pin, 0.13 m long (the drawing, the turn and the
card are mkmmd/core/paper_scissors.py, the meshes paper_scissors_parts.py beside it). Drivers on the root's `open` turn
the two pieces: nothing runs at render time.

    [[prop]]
    name = "scissors"
    card = "library:paper_scissors"
    attach = "<cast>:wrist.R"                    # in a hand: it steps with her [[stepped]] window
    slots = { open = 0.0, loop_b = "iris" }

    [[key]]
    target = "scissors"
    prop = "open"
    keys = [[0.0, 0.0], [0.4, 1.0]]

Frame: the origin on the split pin, the blades along +Z (tips at 0.055), the loops along -Z (down to -0.075), the card
in the X-Z plane (faces toward -Y and +Y). Options (the `slots`): open (its unkeyed value, 0 shut .. 1 open) and the
colour roles of mkmmd.core.paper_scissors.ROLES. Root property for [[key]]: open (0 shut: the blades over each other,
the tips together; 1: a V of 60 degrees; clamped to 0 .. 1 by the drivers).

Objects `<name>_blade_a` (the front piece: its blade, its handle and thumb loop, turning +30 x open about +Y through the
pin) and `<name>_blade_b` (the back piece with the finger loop, -30 x open), `<name>_pin` (the brass split pin, the back
piece's child), and empties under the root: `<name>_pivot` (on the pin), `<name>_tip` (midway between the blades' tips,
sliding down the V's middle line as they open), `<name>_cut` (on the V's middle line 18 mm out, where a thread is cut:
the blades close over it at open 0.27) and `<name>_hold` (midway between the loops' middles). Materials
`<name>_<material>` of mkmmd.core.paper_scissors.MATS: a flat look draws each blade's pencil line crisply in its blade's
piece when `<name>_blade_a` and `<name>_pencil_a` share a `[[vector.materials]]` group (and b's). The two pieces are
flat card (exempt from the form guard); the pin is measured."""
import bpy

from ....core import paper_scissors as PS
from .. import shell as SH
from ..sets import nightkit as NK
from . import parent_cutout_mats as MT
from . import paper_scissors_parts as PSP
from . import register
from . import showkit as SK
from .cafe_kit import clear_drivers

FLAT = "flat card: a child's paper scissors (its two pieces of card and their pencil lines)"


def _materials(name, col):
    """{material: material} (mkmmd.core.paper_scissors.MATS): card for the blades and handles (its cut edge a little
    lighter), a pencil line for each blade, brass."""
    mats = {}
    for key in PS.MATS:
        role = PS.ROLE_OF[key]
        c, nm = col[role], f"{name}_{key}"
        if role == "brad":
            m = MT.brass(nm, c)
        elif role == "pencil":
            m, nb = NK.new_material(nm)
            nb.output(nb.principled((*c, 1.0), rough=0.45, spec=0.45))
            m.diffuse_color = (*c, 1.0)
        else:
            m = MT.paper(nm, c, tuple(min(1.0, 1.12 * v + 0.02) for v in c))
        mats[key] = m
    return mats


def _driver(owner, path, index, expr, root):
    """A scripted driver `expr` on owner.path[index] reading the root's `open`; refused unless Blender keeps it a simple
    expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    v = d.variables.new()
    v.name, v.type = "open", "SINGLE_PROP"
    v.targets[0].id_type, v.targets[0].id = "OBJECT", root
    v.targets[0].data_path = '["open"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"paper_scissors: the driver on {owner.name}.{path}[{index}] is not a Blender simple "
                         "expression")
    return fc


def _empty(name, coll, parent, loc):
    SH.purge(name)
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = 0.01
    o.empty_display_type = "PLAIN_AXES"
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    return o


@register("paper_scissors")
def paper_scissors(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = PS.options(slots)
    roles = SK.roles(slots, PS.ROLES, "paper_scissors")
    col = {k: SK.rgb(v, slots, f"paper_scissors: {k}") for k, v in roles.items()}
    mats = _materials(name, col)

    def mat_of(i):
        return mats[PS.MATS[i]]

    clear_drivers(root)
    root["open"] = O["open"]
    root.id_properties_ui("open").update(description="0 shut .. 1 open: the blades a V of 60 degrees", min=0.0,
                                         max=1.0, soft_min=0.0, soft_max=1.0)
    ob = PS.objects(name)
    ex = PS.expressions()
    M = PSP.parts()
    pieces = {}
    for side in ("a", "b"):
        o = SH.mesh_object(ob[f"blade_{side}"], M[f"blade_{side}"], coll, root, mat_of)
        o.rotation_mode = "XYZ"
        SH.exempt(o, FLAT)
        _driver(o, "rotation_euler", 1, ex[f"blade_{side}"], root)
        pieces[side] = o
    SH.mesh_object(ob["pin"], M["pin"], coll, pieces["b"], mat_of)
    _empty(ob["pivot"], coll, root, (0.0, 0.0, 0.0))
    _driver(_empty(ob["tip"], coll, root, tuple(PS.tip(O["open"])[0])), "location", 2, ex["tip"], root)
    _empty(ob["cut"], coll, root, (0.0, 0.0, PS.CUT))
    hold = _empty(ob["hold"], coll, root, tuple(PS.hold(O["open"])[0]))
    _driver(hold, "location", 0, ex["hold_x"], root)
    _driver(hold, "location", 2, ex["hold_z"], root)
    return PS.card(name, O, roles)
