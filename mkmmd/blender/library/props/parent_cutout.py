"""parent_cutout: a giant picture-book parent cut out of construction paper (`card = "library:parent_cutout"`).

    [[prop]]
    name = "dad"
    card = "library:parent_cutout"
    at = [-1.4, 0.8, 0.0]                          # stands on its foot line, faces -Y
    slots = { figure = "father", paper = "pine" }  # or figure = "mother"; colour roles below

    [[key]]                                        # the scolding finger wags: an automatic tick-tock for two seconds
    target = "dad"
    prop = "wagging"
    keys = [[6.5, 0.0], [6.7, 18.0], [8.3, 18.0], [8.5, 0.0]]

`figure` "father" (default: a man in a narrow-lapel suit and slim tie, side-parted hair, his left forearm raised with the
index finger up, scolding) or "mother" (an A-line shift with a Peter Pan collar and a hem band, a bouffant with a flip and
a bow, hands on her hips): flat silhouettes 3.29 / 3.16 m tall to the top of their white die-cut border, a soft drop
shadow behind (parent_cutout_layout has the geometry, parent_cutout_parts the drawings).

Objects: `<name>_figure` (the white card, the paper and its details; the mother's arms on it), `<name>_shadow` (the drop
shadows: graphic layers, exempt from the form check), `<name>_brad` (the brass paper fasteners); the father's
`<name>_hand` (his forearm and hand, its origin on the elbow's brad) and `<name>_hand_shadow` turn in the card's plane by
the drivers on their rotation_euler[1]: angle = wag + wagging * sin(2 pi wag_hz t), t the clip second ((frame - f0) / fps,
`f0` and `fps` root properties written at build). Root properties (father): `wag` (degrees counter-clockwise seen from the
front, key it), `wagging` (degrees of the tick-tock, 0), `wag_hz` (2.5); starting values in `slots`.

Colour roles (`slots`: a palette slot, a hex, a role or a 'name:weight,...' blend; `ink` and `chalk` are the darker and the
lighter of the palette's base and text, `black` darkens a blend): paper (father pine, mother rose), shade (the lapels:
paper:3,black:1), hair (ink: hair and shoes), detail (love: his tie, her dress), trim (chalk: shirt, collars, cuffs,
buttons), accent (gold: her bow and hem band), rim (chalk: the border's card), shadow (ink), brad (gold)."""
import re

import bpy

from ....core import shell as S
from .. import shell as SH
from . import parent_cutout_layout as LAY
from . import parent_cutout_mats as MT
from . import register
from .cafe_kit import clear_drivers

LIVE = tuple(LAY.PARAMS) + ("f0", "fps")          # the root properties the wag drivers read
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FUNCS = {"sin"}


def _drive(owner, path, index, expr, root):
    """A scripted driver `expr` on owner.path[index] reading the root's properties by name; refused unless it is a
    Blender simple expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for nm in sorted(set(NAME.findall(expr)) - FUNCS - {"frame"}):
        if nm not in LIVE:
            raise ValueError(f"parent_cutout: the driver on {owner.name}.{path} reads {nm!r}, not a root property")
        v = d.variables.new()
        v.name, v.type = nm, "SINGLE_PROP"
        v.targets[0].id_type, v.targets[0].id = "OBJECT", root
        v.targets[0].data_path = f'["{nm}"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"parent_cutout: the driver on {owner.name}.{path}[{index}] is not a Blender simple expression "
                         f"({expr})")
    return fc


def _shadows(name, coll, parent, sheets, pal, loc=(0.0, 0.0, 0.0)):
    """One object of shadow sheets (each its own material and alpha image): a graphic layer, no shadow of its own."""
    meshes, mats = [], {}
    for k, (sh, mesh) in enumerate(sheets):
        img = MT.alpha_image(f"{name}_{k}" if len(sheets) > 1 else name, sh.alpha)
        mats[k] = MT.shadow(f"{name}_{k}" if len(sheets) > 1 else name, img, pal.lin("shadow"), LAY.SHADOW_ALPHA)
        m = mesh.copy()
        m.Qm[:] = k
        meshes.append(m)
    o = SH.mesh_object(name, S.merge(meshes), coll, parent, lambda k: mats[k], loc=loc)
    o.visible_shadow = False
    return SH.exempt(o, "a drop shadow printed behind the paper")


@register("parent_cutout")
def parent_cutout(name, coll, root, slots=None):
    slots = dict(slots or {})
    figure, start = LAY.options(slots)
    pal = LAY.Pal(slots, figure)
    pal.check()
    L = LAY.build(figure)
    for nm in LAY.objects(name, "father"):            # a rebuild in the same session, whichever figure it was
        SH.purge(nm)
    clear_drivers(root)
    mats = {}

    def mat_of(i):
        """The material of a role, made the first time a mesh asks for it."""
        role = LAY.MATS[i]
        if role not in mats:
            c = pal.lin(role)
            if role == "brad":
                mats[role] = MT.brass(f"{name}_{role}", c)
            elif role == "rim":
                mats[role] = MT.paper(f"{name}_{role}", c, tuple(v * w for v, w in zip(c, (0.93, 0.88, 0.78))),
                                      back=tuple(0.9 * v for v in c), rough=0.8, mottle=0.02, fleck=0.03, grain=0.06)
            else:
                mats[role] = MT.paper(f"{name}_{role}", c, tuple(min(1.0, 1.12 * v + 0.02) for v in c))
        return mats[role]

    SH.mesh_object(f"{name}_figure", L.body, coll, root, mat_of)
    SH.mesh_object(f"{name}_brad", L.brads, coll, root, mat_of)
    _shadows(f"{name}_shadow", coll, root, L.sheets, pal)
    if L.hand is not None:
        sc = bpy.context.scene
        values = dict(start, f0=float(sc.get("mk_frame0", sc.frame_start)), fps=float(sc.render.fps / sc.render.fps_base))
        docs = {k: doc for k, (_d, _r, doc) in LAY.PARAMS.items()}
        docs.update(f0="the frame of clip second 0", fps="frames per second")
        for k, v in values.items():
            root[k] = v
            ui = {"description": docs[k], "default": v}
            if k in LAY.PARAMS:
                lo, hi = LAY.PARAMS[k][1]
                ui.update(soft_min=lo, soft_max=hi)
            root.id_properties_ui(k).update(**ui)
        hand = SH.mesh_object(f"{name}_hand", L.hand, coll, root, mat_of, loc=L.pivot)
        ghost = _shadows(f"{name}_hand_shadow", coll, root, [L.hand_sheet], pal, loc=L.hand_at)
        for o in (hand, ghost):
            o.rotation_mode = "XYZ"
            _drive(o, "rotation_euler", 1, LAY.EXPR, root)
    return LAY.card(name, figure, pal.roles(), start)
