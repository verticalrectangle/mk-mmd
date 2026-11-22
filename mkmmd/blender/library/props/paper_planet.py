"""paper_planet: a paper planet on a wooden skewer with a toothpick flag, 0.8 m across (the maths is
mkmmd/core/paper_planet.py): a ball of paper gores, its top flattened under a calotte (the landing spot), cut-paper
craters, a striped cocktail flag. Its motion is keyed at build from its options, every frame or held on a step grid.

    [[prop]]
    name = "mars"
    card = "library:paper_planet"
    at = [0.0, 0.0, 0.6]
    slots = { up = 17.0, land = 17.26, step = 12, step_from = 15.12 }

Options (the `slots`): size (the ball's diameter, 0.8 m), up (the clip second it is in place: it comes up from `depth`
1.6 m below over `rise` 0.7 s, arriving ON it; absent: in place all along), turn (turns a second about the skewer,
0.03, counted from `up`), land (a clip second or a list: something lands on its top then; it sinks `sink` 0.06 m on its
skewer, bobs, tips and settles `settle` 0.018 m lower), whip (how hard the flag whips, 1), step / step_from (as the
paper theatre's: poses a second, 0 every frame), seed and craters (their places and how many, 3 and 7), stripes (the
flag's colours from the top: foam, paper, iris) and the colour roles of mkmmd.core.paper_planet.ROLES.

Objects `<name>_lift` (an empty: the rise, bob and tip), under it `<name>_ball` (the gores and calotte, turning; scaled
to the size) with `<name>_craters`, `<name>_skewer`, `<name>_pick` and its `<name>_flag` (shape key `whip`) and
`<name>_flag_tip`, and `<name>_top` (an empty on the landing spot that rides the bob, not the turn). Materials
`<name>_<role>` and `<name>_flag<i>` (the stripes). The form guard measures every part (the paper layers too)."""
import math

import bpy
import numpy as np
from mathutils import Matrix

from ....core import paper_planet as PP
from ....core import paper_theatre as PT
from ...keys import set_fcurve
from .. import shell as SH
from ..sets import nightkit as NK
from . import register
from . import showkit as SK
from .dragonfly import paper


def _wood(name, c):
    """A bamboo stick's wood: its colour streaked along its length."""
    m, nb = NK.new_material(name)
    P = nb.vmath("MULTIPLY", nb.texcoord("Object"), (90.0, 90.0, 3.0))
    k = nb.math("ADD", 0.9, nb.math("MULTIPLY", nb.noise(P, scale=1.0, detail=3.0, rough=0.6), 0.2))
    b = nb.principled(nb.vmath("SCALE", (*c, 1.0), scale=k), rough=0.55, spec=0.35)
    nb.set(b.inputs["Coat Weight"], 0.15)
    nb.output(b)
    m.diffuse_color = (*c, 1.0)
    return m


def _empty(name, coll, parent, loc=(0.0, 0.0, 0.0), size=0.05):
    SH.purge(name)
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = size
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    return o


def _mesh(name, mesh, coll, parent, mat_of, loc=(0.0, 0.0, 0.0)):
    SH.purge(name)
    me, _ = SH.mesh_data(name, mesh, mat_of)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    return o


@register("paper_planet")
def paper_planet(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = PP.options(slots)
    roles = SK.roles(slots, PP.ROLES, "paper_planet")
    col = {k: SK.rgb(v, slots, f"paper_planet: {k}") for k, v in roles.items()}
    stripes = [SK.rgb(c, slots, "paper_planet: stripes") for c in O["stripes"]]
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    if O["step"] > fps:
        raise ValueError(f"paper_planet: step = {O['step']:g} poses a second is more than the scene's {fps:g} frames")
    mats = {}
    for role in PP.MATS:
        nm = f"{name}_{role}"
        mats[role] = _wood(nm, col[role]) if role in ("skewer", "pick") else paper(nm, col[role], 0.09)
    flags = [paper(f"{name}_flag{i}", c, 0.05) for i, c in enumerate(stripes)]
    P = PP.parts(O["craters"], O["seed"])
    ob = PP.objects(name)
    size = O["size"]

    def mat_of(r):
        return mats[PP.MATS[r]]

    lift = _empty(ob["lift"], coll, root, size=0.2)
    ball = _mesh(ob["ball"], P["ball"], coll, lift, mat_of)
    ball.scale = (size,) * 3
    _mesh(ob["craters"], P["craters"], coll, ball, mat_of)
    _mesh(ob["skewer"], P["skewer"], coll, ball, mat_of)
    o, a, x = PP.pick_frame()
    pick = _mesh(ob["pick"], P["pick"], coll, ball, mat_of, loc=tuple(o))
    pick.matrix_basis = Matrix.Translation(o) @ Matrix(np.column_stack([x, np.cross(a, x), a]).tolist()).to_4x4()
    pick.rotation_mode = "XYZ"
    flag = _mesh(ob["flag"], P["flag"], coll, pick, lambda r: flags[min(r, len(flags) - 1)],
                 loc=(0.0, 0.0, PP.flag_place()))
    flag.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    flag.shape_key_add(name="Basis", from_mix=False)
    key = flag.shape_key_add(name="whip", from_mix=False)
    key.slider_min, key.slider_max = -1.5, 1.5
    co = np.empty(len(flag.data.vertices) * 3, np.float32)
    flag.data.vertices.foreach_get("co", co)
    key.data.foreach_set("co", (co.reshape(-1, 3) + P["whip"].astype(np.float32)).ravel())
    _empty(ob["tip"], coll, flag, loc=tuple(PP.flag_tip()), size=0.03)
    _empty(ob["top"], coll, lift, loc=(0.0, 0.0, PP.Z_TOP * size), size=0.1)
    keys, times, interp = PT.key_times(sc.frame_start, sc.frame_end, f0, fps, O["step"], O["step_from"])
    m = PP.motion(O, times)
    set_fcurve(lift, "location", 2, keys, m["z"], interp=interp)
    set_fcurve(lift, "rotation_euler", 0, keys, m["tip_x"], interp=interp)
    set_fcurve(lift, "rotation_euler", 1, keys, m["tip_y"], interp=interp)
    set_fcurve(ball, "rotation_euler", 2, keys, m["spin"], interp=interp)
    set_fcurve(flag, "rotation_euler", 2, keys, m["yaw"], interp=interp)
    set_fcurve(flag.data.shape_keys, 'key_blocks["whip"].value', 0, keys, m["whip"], interp=interp)
    return PP.card(name, O, roles)
