"""paper_plane: a big folded paper dart of construction paper, 1.5 m long, a figure 1.5 m tall can ride it side-saddle
(the maths is mkmmd/core/paper_plane.py: the folded sheet, the seat, the motion and the card).

    [[prop]]
    name = "plane"
    card = "library:paper_plane"
    at = [0.0, 0.0, 1.2]
    yaw = -90                           # its nose (-Y) toward -X
    slots = { land = 19.05, bank = [[19.3, 0.0], [19.75, 22.0], [20.15, -17.0], [20.5, 0.0]], step = 12, step_from = 17.66 }

    [pose.rin]
    sit = "plane:seat"                  # side-saddle on the spine, legs over its left wing, her right side to the nose

Options (the `slots`): size (the sheet's length, 1.5 m: the plane is as long), sheet (its width over its length, 0.773),
keel (the keel's depth at the tail over the length, 0.11), seat (where she sits, a share of the length from the nose;
default where the leading edge is 0.24 m ahead of her hips), side ("L": her legs over the left wing; "R"), bank (degrees,
+ rolls its left wing down) and pitch (degrees, + its nose up), each a number or keys [[t, degrees], ...] in clip seconds,
flutter (degrees the wings flex about their root creases, 2; or keys [[t, degrees], ...]; 0 still), land (a clip second
or a list: something lands on it then: it dips `sink` 0.05 m x 0.6 under the weight and comes back, the wings kick),
step / step_from (as the paper theatre's: poses a second and the grid's first clip second; 0 every frame), seed (the
doodles' hand), doodles (true) and the colour roles of mkmmd.core.paper_plane.ROLES.

The bank, the pitch and a landing's dip are keyed on the ROOT (rotation mode YXZ: the bank about its own long axis
first, then the pitch, then the heading of `[[prop]] yaw`; the dip on its delta_location z), so a figure seated on it
rides every bank and dip; `at` and `[[key]]`s on `location` and `rotation_euler` index 2 (the heading, radians) fly it.
A `[[key]]` on `rotation_euler` index 0 (minus the pitch) or 1 (the bank), radians, replaces those slots' keys. The pose
stage seats her on the scene's first frame: keep the plane level there.

Objects `<name>_body` (the keel: both walls, the flaps' ends folded inside them, a star on each side), `<name>_wing_L`
and `<name>_wing_R` (each wing with its flap, its leading edge folded two plies thick, its doodles; turning about its root
crease, local Y, for the flutter), `<name>_tip_L` and `<name>_tip_R` (empties at the wing tips, children of the wings)
and the rider's colliders, hidden boxes whose tops lie on the paper's top (the card's `colliders`, which a seated
figure's sim takes by default: her hair and skirt lie on the paper): `<name>_col_wing_L` and `<name>_col_wing_R` (a band
along each leading edge, riding its wing) and `<name>_col_spine` (along the spine under her seat, riding the root).
Materials `<name>_<role>`. Flat paper is flat on purpose: every part is tagged `mk_form_exempt`; measured all the same
(a `form` check does when nothing else is left), its rolled edges score 0.03."""
import math

import bpy
import numpy as np
from mathutils import Matrix

from ....core import paper_plane as PL
from ....core import paper_theatre as PT
from ....core import shell as CS
from ...keys import set_fcurve
from .. import shell as SH
from . import register
from . import showkit as SK
from .dragonfly import paper

FLAT = "flat paper: a folded paper plane"


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mesh(name, mesh, coll, parent, mat_of, loc=(0.0, 0.0, 0.0), yaw=0.0):
    """A flat-shaded part (paper has hard folds) under `parent`, tagged as flat paper."""
    SH.purge(name)
    me, _ = SH.mesh_data(name, mesh, mat_of, smooth=False, normals=False)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    o.rotation_euler = (0.0, 0.0, yaw)
    return SH.exempt(o, FLAT)


def _empty(name, coll, parent, loc, size=0.03):
    SH.purge(name)
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = size
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    return o


@register("paper_plane")
def paper_plane(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = PL.options(slots)
    roles = SK.roles(slots, PL.ROLES, "paper_plane")
    col = {k: SK.rgb(v, slots, f"paper_plane: {k}") for k, v in roles.items()}
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    if O["step"] > fps:
        raise ValueError(f"paper_plane: step = {O['step']:g} poses a second is more than the scene's {fps:g} frames")
    mats = {role: paper(f"{name}_{role}", col[role], 0.05 if role in ("star", "stripe", "ink") else 0.07)
            for role in PL.MATS}

    def mat_of(i):
        return mats[PL.MATS[i]]

    F = PL.Fold(O["size"], O["sheet"], O["keel"])
    P = PL.parts(O["size"], O["sheet"], O["keel"], O["doodles"], O["seed"])
    ob = PL.objects(name)
    _mesh(ob["body"], P["body"], coll, root, mat_of)
    wings = {}
    for side in "LR":
        o, yaw = PL.hinge(F, side)
        R = _rz(-yaw)                                                   # the prop's frame -> the wing's (its hinge)
        wings[side] = _mesh(ob[f"wing_{side}"], P[f"wing_{side}"].apply(R, -R @ o), coll, root, mat_of, tuple(o), yaw)
        _empty(ob[f"tip_{side}"], coll, wings[side], tuple(R @ (PL.tip(F, side) - o)))
    for key, (c, R, size) in PL.colliders(F).items():                   # the rider's colliders, hidden: a wing's rides
        M = np.eye(4)                                                   # its wing (the flutter too), the spine's the root
        M[:3, :3], M[:3, 3] = R, c
        parent = root
        if key.startswith("wing"):
            o, yaw = PL.hinge(F, key[-1])
            W = np.eye(4)
            W[:3, :3], W[:3, 3] = _rz(yaw), o
            M, parent = np.linalg.inv(W) @ M, wings[key[-1]]
        box = SH.mesh_object(ob[f"col_{key}"], CS.rounded_box(tuple(size), r=PL.COL["rnd"]), coll, parent,
                             lambda i: mats["inner"])
        box.matrix_basis = Matrix(M.tolist())
        SH.as_collider(box)
        SH.exempt(box, "a collider: the paper under her, never rendered")
    root.rotation_mode = "YXZ"                                          # the bank first, then the pitch, then the heading
    moving = PL.moves(O)
    if any(moving.values()):
        keys, times, interp = PT.key_times(sc.frame_start, sc.frame_end, f0, fps, O["step"], O["step_from"])
        m = PL.motion(O, times)
        if moving["turn"]:
            set_fcurve(root, "rotation_euler", 0, keys, -m["pitch"], interp=interp)
            set_fcurve(root, "rotation_euler", 1, keys, m["bank"], interp=interp)
        if moving["dip"]:
            set_fcurve(root, "delta_location", 2, keys, m["dip"], interp=interp)
        if moving["flex"]:
            set_fcurve(wings["L"], "rotation_euler", 1, keys, -m["flex_L"], interp=interp)
            set_fcurve(wings["R"], "rotation_euler", 1, keys, m["flex_R"], interp=interp)
    return PL.card(name, O, roles)
