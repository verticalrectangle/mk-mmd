"""paper_cafe_chair and paper_cafe_table: the paper theatre café's furniture, of card and paper as the theatre is (the
pieces and the cards are mkmmd/core/paper_cafe.py; docs/design.md: Props), sized round a figure 1.5 m tall.

    [[prop]]
    name = "chair"
    card = "library:paper_cafe_chair"
    at = [0.0, 0.0, 0.0]                         # [pose.<cast>] sit = "chair:seat"

    [[prop]]
    name = "table"
    card = "library:paper_cafe_table"
    at = [-0.52, -0.12, 0.0]                     # beside her right hand
    slots = { rattle = [30.15], step = 12, step_from = 0.0 }

The chair: objects `<name>_body` (the bent strips: legs, ring, the seat's band, the back and its loop), `<name>_seat`
(the cane seat and its pencilled weave), `<name>_brads` (brass split pins at the legs), hidden colliders
`<name>_col_seat` (a cylinder) and `<name>_col_back` (a box); card `use.sit` / `use.feet` `seat`, `use.rest`
`seat_front`, `use.look` `back_top`. Colour roles wood, cane, cane_line, brad.

The table: `<name>_table` (the top, its gilt band, the pedestal's two slotted silhouettes, the bearer), `<name>_saucer`
and `<name>_cup` (with its coffee and the paper lily pad floating on it: they jump and chatter on each `rattle`, held on
the `step` grid as the theatre's motion is), a hidden collider `<name>_col_top`; card `use.anchor` `scissors` (where the
scissors lie), `use.look` `cup` (riding the cup), `use.rest` `top`. Colour roles top, rim, base, cup, cup_band, saucer,
coffee, pad. Card is flat on purpose: the flat pieces are scissor-cut plates (`mk_form_exempt`)."""
import math

import bpy
import numpy as np

from ....core import paper_cafe as PC
from ....core import paper_theatre as PT
from ....core import shell as CS
from ...keys import set_fcurve
from .. import shell as SH
from ..sets import paper_theatre_parts as PTP
from . import parent_cutout_layout as PCL
from . import parent_cutout_mats as MT
from . import register
from . import showkit as SK

CARD = "card and paper: a toy theatre's furniture"


def _mats(name, roles, slots, kind):
    mats = {}
    for role, spec in roles.items():
        c = SK.rgb(spec, slots, f"{kind}: {role}")
        nm = f"{name}_{role}"
        mats[role] = MT.brass(nm, c) if role == "brad" else MT.paper(nm, c, tuple(min(1.0, 1.1 * v + 0.02) for v in c))
    return mats


def _mesh(pieces, mats_list):
    """One Mesh of the pieces: Meshes as they are, Plates (mkmmd.core.paper_theatre) cut and rolled by the theatre's
    parts; a Plate's role is looked up in `mats_list` (the prop's material order)."""
    out = []
    for k, p in enumerate(pieces):
        if isinstance(p, PT.Plate):
            m = PTP.plate_mesh(p._replace(role="cardboard", edge=None), PTP.seed_of("paper_cafe", k))   # cut, rolled
            idx = mats_list.index(p.role)
            m = CS.Mesh(m.V, m.Q, m.T, np.full(len(m.Q), idx), np.full(len(m.T), idx))
            out.append(m)
        else:
            out.append(p)
    return CS.merge(out)


def _object(name, mesh, coll, parent, mats, order, loc=(0.0, 0.0, 0.0)):
    o = SH.mesh_object(name, mesh, coll, parent, lambda i: mats[order[i]], loc=loc)
    o.rotation_mode = "XYZ"
    return o


def _collider(name, mesh, coll, root, mat, loc=(0.0, 0.0, 0.0)):
    o = SH.mesh_object(name, mesh, coll, root, lambda i: mat, loc=loc)
    SH.as_collider(o)
    SH.exempt(o, "a collider: never rendered")
    return o


@register("paper_cafe_chair")
def paper_cafe_chair(name, coll, root, slots=None):
    slots = dict(slots or {})
    roles = SK.roles(slots, PC.CHAIR_ROLES, "paper_cafe_chair")
    mats = _mats(name, roles, slots, "paper_cafe_chair")
    order = PC.CHAIR_MATS
    P = PC.chair_pieces()
    _object(f"{name}_body", _mesh(P["body"], order), coll, root, mats, order)
    SH.exempt(_object(f"{name}_seat", _mesh(P["seat"], order), coll, root, mats, order), CARD)
    pins = []
    for dx, dy in P["brads"]:                               # a split pin on the seat's band over each leg, facing out
        m = PCL.brad((0.0, 0.0), -0.0015, 0.004)
        phi = math.atan2(dx, -dy)
        R = np.array([[math.cos(phi), -math.sin(phi), 0.0], [math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
        r = PC.SEAT_R + 0.003 + 0.5 * PC.BAND[1]
        pins.append(m.apply(R, (dx * r, dy * r, PC.SEAT_Z - 0.022)))
    b = CS.merge(pins)
    b = CS.Mesh(b.V, b.Q, b.T, np.full(len(b.Q), order.index("brad")), np.full(len(b.T), order.index("brad")))
    _object(f"{name}_brads", b, coll, root, mats, order)
    hid = mats["wood"]
    seat = _collider(f"{name}_col_seat", CS.rounded_box((2 * PC.SEAT_R, 2 * PC.SEAT_R, 0.04), r=0.012), coll, root,
                     hid, loc=(0.0, 0.0, PC.SEAT_Z - 0.02))
    lean = math.radians(PC.BACK[3])
    back = _collider(f"{name}_col_back", CS.rounded_box((0.3, 0.03, 0.42), r=0.01), coll, root, hid,
                     loc=(0.0, PC.BACK[1] + 0.22 * math.tan(lean) + 0.01, PC.SEAT_Z + 0.22))
    back.rotation_euler = (-lean, 0.0, 0.0)
    colliders = [{"type": "box", "object": back.name, "rnd": 0.012, "tag": name},
                 {"type": "cylinder", "object": seat.name, "half_h": 0.02, "rnd": 0.012, "tag": name}]
    return PC.chair_card(name, roles, colliders)


@register("paper_cafe_table")
def paper_cafe_table(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = PC.table_options(slots)
    roles = SK.roles(slots, PC.TABLE_ROLES, "paper_cafe_table")
    mats = _mats(name, roles, slots, "paper_cafe_table")
    order = PC.TABLE_MATS
    P = PC.table_pieces()
    SH.exempt(_object(f"{name}_table", _mesh(P["table"], order), coll, root, mats, order), CARD)
    cx, cy = PC.CUP_AT
    saucer = _object(f"{name}_saucer", _mesh(P["saucer"], order), coll, root, mats, order,
                     loc=(cx, cy, PC.TOP_Z))
    cup = _object(f"{name}_cup", _mesh(P["cup"], order), coll, saucer, mats, order, loc=(0.0, 0.0, PC.SAUCER[1] - 0.006))
    top = _collider(f"{name}_col_top", CS.rounded_box((2 * PC.TOP_R, 2 * PC.TOP_R, PC.TOP_T), r=0.005), coll, root,
                    mats["top"], loc=(0.0, 0.0, PC.TOP_Z - 0.5 * PC.TOP_T))
    if O["rattle"]:
        sc = bpy.context.scene
        f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
        keys, times, interp = PT.key_times(sc.frame_start, sc.frame_end, f0, fps, O["step"], O["step_from"])
        cz, tx, ty, sz = PC.rattle(O, times)
        set_fcurve(saucer, "location", 2, keys, PC.TOP_Z + sz, interp=interp)
        set_fcurve(cup, "location", 2, keys, PC.SAUCER[1] - 0.006 + cz, interp=interp)
        set_fcurve(cup, "rotation_euler", 0, keys, tx, interp=interp)
        set_fcurve(cup, "rotation_euler", 1, keys, ty, interp=interp)
    colliders = [{"type": "cylinder", "object": top.name, "half_h": 0.5 * PC.TOP_T, "rnd": 0.005, "tag": name}]
    return PC.table_card(name, O, roles, cup.name, colliders)
