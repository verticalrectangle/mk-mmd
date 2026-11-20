"""marionette: the control and the threads of a string puppet (`card = "library:marionette"`): a wooden aeroplane control
(a main bar along the puppet's facing with the head bar across it, a hand bar hung under its front end) and a leg bar
held apart in front of her, screw eyes where the threads are tied, and the threads themselves. It takes a cast member
through `[perform.<cast>] marionette = {prop = "<name>", ...}`: the perform stage puts the cross over her head, keys it
and her body, ties the threads' ends to her bones, and the build's `marionette` stage draws the threads
(mkmmd.core.marionette; docs/design.md: Marionettes).

    [[prop]]
    name = "strings"
    card = "library:marionette"
    at = [60.0, 0.0, 0.0]                       # anywhere: the perform stage keys the bars over her
    slots = { wood = "gold:base:0.45", thread = "text", radius = 0.0015 }

Objects: `<name>_cross` (the main bar, the head bar, their eyes), `<name>_handbar` (a child of the cross),
`<name>_legbar`, the thread anchors `<name>_at_<part>` (empties on the eyes), the ends on her `<name>_end_<part>`
(empties the perform stage bone-parents to her), the threads `<name>_thread_<part>` (and `<name>_cut_<part>`, the piece
left on her after a cut) as bevelled poly curves, and `<name>_snip_<part>` (empties on each thread at its cut point, for
effects to follow). Parts: head, back, wrist_L, wrist_R, knee_L, knee_R. Colour roles (`slots`): wood, eye (the screw
eyes), thread; `radius` the threads' (m, 0.0015: 3 mm thick, readable at 1080 in a medium shot)."""
import bpy

from ....core import marionette as MR
from .. import shell as SH
from . import register
from . import showkit as SK
from .cafe_kit import Kit


def _empty(name, coll, parent, loc=(0.0, 0.0, 0.0), size=0.02):
    SH.purge(name)
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = size
    o.empty_display_type = "PLAIN_AXES"
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    return o


def _thread(name, coll, root, mat, radius, n, start):
    """A bevelled poly curve of n points hanging 1 m straight down from `start` (root frame) until the build keys it."""
    SH.purge(name)
    cu = bpy.data.curves.get(name)
    if cu is not None:
        bpy.data.curves.remove(cu)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(n - 1)
    for i, p in enumerate(sp.points):
        p.co = (start[0], start[1], start[2] - i / (n - 1), 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    coll.objects.link(o)
    o.parent = root
    o.hide_render = True
    return o


@register("marionette")
def marionette(name, coll, root, slots=None):
    slots = dict(slots or {})
    roles = SK.roles(slots, MR.ROLES, "marionette")
    radius = SK.number(slots, "radius", MR.THREAD["radius"], "marionette", 0.0003, 0.01)
    K = Kit(name, coll, root, slots)
    looks = {"wood": dict(rough=0.75, spec=0.3), "eye": dict(rough=0.35, metal=0.8), "thread": dict(rough=0.9, spec=0.2)}
    mats = {}
    for role in MR.ROLES:
        c = SK.rgba(roles[role], slots, f"marionette: {role}")
        mats[role] = K.simple_mat(role, c, **looks[role])
        mats[role].diffuse_color = c
    mat_of = lambda i: mats[MR.MATS[i]]                                                         # noqa: E731
    M = MR.cross_meshes()
    cross = SH.mesh_object(f"{name}_cross", M["cross"], coll, root, mat_of)
    hand = SH.mesh_object(f"{name}_handbar", M["hand"], coll, cross, mat_of, loc=MR.HAND_BAR["at"])
    leg = SH.mesh_object(f"{name}_legbar", M["leg"], coll, root, mat_of, loc=MR.LEG_BAR["home"])
    for o in (cross, hand, leg):
        o.rotation_mode = "QUATERNION"
    bars = {"cross": cross, "hand": hand, "leg": leg}
    home = {"cross": (0.0, 0.0, 0.0), "hand": MR.HAND_BAR["at"], "leg": MR.LEG_BAR["home"]}
    for part, (bar, pt) in MR.ANCHORS.items():
        _empty(f"{name}_at_{part}", coll, bars[bar], pt, 0.01)
        at = tuple(h + p for h, p in zip(home[bar], pt))
        _empty(f"{name}_end_{part}", coll, root, (at[0], at[1], at[2] - 1.0), 0.015)
        _empty(f"{name}_snip_{part}", coll, root, (at[0], at[1], at[2] - 0.5), 0.015)
        for k in ("thread", "cut"):
            _thread(f"{name}_{k}_{part}", coll, root, mats["thread"], radius, MR.THREAD["points"], at)
    return MR.card(name, roles)
