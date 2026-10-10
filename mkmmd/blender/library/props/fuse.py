"""The cartoon fuse (`card = "library:fuse"`): a rope that sprouts from its root, a spark that is lit at its tip and burns
it down. Three objects: `<name>_cap` (the socket the rope leaves) and `<name>_cord` (a bevelled Bezier curve through the
layout's points: it shows from the root up to min(grow, 1 - burn) of its length) under the root, and `<name>_spark` (a
spiky star sized by `lit`, flickering and turning), which rides that end on a Follow Path constraint and takes the root's
world scale (it has no parent: the constraint already places it in the world, a parent's move would count twice).
Drivers on the scene frame and three root properties do it all (no Python at render time); `[[key]]` keys them:

    [[prop]]
    name = "fuse"
    card = "library:fuse"
    at = [0.0, 1.0, 1.2]
    [[key]]
    target = "fuse"
    prop = "grow"
    keys = [[6.30, 0.0], [6.50, 0.4], [6.85, 1.0]]

Leave `grow` at its default (1) in the slots and key it from 0: the props stage measures the prop's form as it is built,
and a rope that has not grown is two end caps. Numbers, the colour roles and the card: fuse_layout.py."""
import bpy
import numpy as np

from ....core import shell as S
from .. import shell as SH
from . import fuse_layout as LAY
from . import register
from .cafe_kit import Kit

END = f"max({LAY.MIN_SCALE}, min(grow, 1.0 - burn))"                      # the share of the cord that shows
FLICKER = f"max({LAY.MIN_SCALE}, lit * (1.0 + 0.22 * sin(frame * 2.3) + 0.12 * sin(frame * 5.7 + 1.0)))"
CAP = f"max({LAY.MIN_SCALE}, min(1.0, grow * 12.0))"                       # the socket pops in as the rope starts


def _driver(owner, path, expr, root, index=-1):
    """A scripted driver `expr` on `owner[path]`; its variables are the root's properties (LAY.PARAMS) it names."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for v in list(d.variables):
            d.variables.remove(v)
        for name in LAY.PARAMS:
            if name in expr:
                var = d.variables.new()
                var.name, var.type = name, "SINGLE_PROP"
                var.targets[0].id_type = "OBJECT"
                var.targets[0].id = root
                var.targets[0].data_path = f'["{name}"]'
        if not d.is_simple_expression:
            raise ValueError(f"fuse: the driver on {path} is not a Blender simple expression ({expr})")


def _cord(K, points, radius, mat):
    """`<name>_cord`: a Bezier curve with auto handles through `points`, a round bevel of `radius`, capped."""
    name = f"{K.name}_cord"
    SH.purge(name)
    old = bpy.data.curves.get(name)
    if old is not None and old.users == 0:
        bpy.data.curves.remove(old)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution, cu.use_fill_caps = radius, 3, True
    cu.bevel_factor_mapping_end = "SPLINE"
    cu.use_path = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(points) - 1)
    for bp, p in zip(sp.bezier_points, points):
        bp.co = tuple(float(c) for c in p)
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    K.coll.objects.link(o)
    o.parent = K.root
    return o


@register("fuse")
def fuse(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    points, radius, start = LAY.options(K.slots)
    for k, (_, doc) in LAY.PARAMS.items():
        root[k] = start[k]
        root.id_properties_ui(k).update(description=doc, min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
    mats = {"cord": K.simple_mat("cord", K.role("cord", LAY.ROLES), rough=0.8),
            "spark": K.simple_mat("spark", K.role("spark", LAY.ROLES), rough=0.4, emit=K.role("spark", LAY.ROLES),
                                  emit_strength=2.0),
            "cap": K.simple_mat("cap", K.role("cap", LAY.ROLES), rough=0.5, metal=0.3)}
    roles = list(mats)
    mat_of = lambda i: mats[roles[i]]                                                          # noqa: E731
    r, h = LAY.CAP
    cap = SH.mesh_object(f"{name}_cap", S.lathe(S.fillet(np.array([(0.0, -h), (r, -h), (r, 0.0), (0.0, 0.0)]), 0.003, 3),
                                                  seg=32, mat=roles.index("cap")), coll, root, mat_of)
    for i in range(3):
        _driver(cap, "scale", CAP, root, i)
    cord = _cord(K, points, radius, mats["cord"])
    _driver(cord.data, "bevel_factor_end", END, root)
    V, T = LAY.spark_mesh()
    spark = SH.mesh_object(f"{name}_spark", S.Mesh(V=V, T=T, mat=roles.index("spark")), coll, None, mat_of)
    grow_with = spark.constraints.new("COPY_SCALE")
    grow_with.target, grow_with.use_offset = root, True                    # the prop's scale (and its parent's)
    con = spark.constraints.new("FOLLOW_PATH")
    con.target, con.use_fixed_location, con.use_curve_follow = cord, True, False
    _driver(con, "offset_factor", END, root)
    for i in range(3):
        _driver(spark, "scale", FLICKER, root, i)
    _driver(spark, "rotation_euler", "frame * 0.35", root, 2)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in roles}, points, radius)
