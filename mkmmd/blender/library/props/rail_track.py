"""The railroad track that lays itself (`card = "library:rail_track"`): an 1860s standard-gauge track on a ballast bed,
straight or along a gentle curve, laid as the root's live property `laid` (metres along it) moves.

Objects under the root (every one moved by drivers on `laid` alone: no Python at render time):
  <name>_bed<k>       bed pieces, one tie spacing each, growing from their start          (meshes <name>_bed_first, _bed)
  <name>_tie<i>       ties dropping in with a small bounce                                  (meshes <name>_tie_v<m>: six)
  <name>_spikes<i>    each tie's four spikes, popping up once its rail has landed          (mesh <name>_spikes)
  <name>_rail<j>      rail spans (both rails, the joint bars at their start), sliding in   (meshes <name>_rail_<length>[_j])
  <name>_front        an empty on the centre line where `laid` is, at the rails' top: aim a camera at the front
The schedule, the plan and the drivers' maths: mkmmd/core/rail_track.py; the meshes, roles and card: rail_track_layout.py.

    [[prop]]
    name = "track"
    card = "library:rail_track"
    slots = { length = 60, radius = 400, laid = 0 }     # `laid` keyed from 0 lays it; unkeyed it shows its starting value
"""
import math
import re

import bpy

from ....core import rail_track as RT
from .. import shell as SH
from . import rail_track_layout as LAY
from . import register
from .cafe_kit import Kit, L, N, clear_drivers, principled, ramp

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KEYWORDS = frozenset({"if", "else", "and", "or", "not"})
PARTS = ("bed", "tie", "spikes", "rail", "front")


def _drive(owner, path, expr, root, index=-1):
    """A scripted driver `expr` on owner.path[index] whose one variable `l` is the root's `laid`; refused unless it is a
    Blender simple expression."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for nm in sorted(set(NAME.findall(expr)) - RT.FUNCS - KEYWORDS):
            if nm != "l":
                raise ValueError(f"rail_track: the driver on {owner.name}.{path} reads {nm!r} ({expr})")
            v = d.variables.new()
            v.name, v.type = "l", "SINGLE_PROP"
            t = v.targets[0]
            t.id_type, t.id, t.data_path = "OBJECT", root, '["laid"]'
        if len(expr) > RT.MAX_EXPR or not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"rail_track: the driver on {owner.name}.{path} is not a Blender simple expression ({expr})")


def _purge(name):
    """Remove what an earlier build of this track left (objects and the meshes they shared)."""
    pat = re.compile(rf"^{re.escape(name)}_({'|'.join(PARTS)})")
    for o in [o for o in bpy.data.objects if pat.match(o.name)]:
        bpy.data.objects.remove(o, do_unlink=True)
    for me in [me for me in bpy.data.meshes if pat.match(me.name) and me.users == 0]:
        bpy.data.meshes.remove(me)


def _object(name, data, coll, root, loc=(0.0, 0.0, 0.0), rz=0.0):
    o = bpy.data.objects.new(name, data)
    coll.objects.link(o)
    o.parent = root
    o.location = loc
    o.rotation_mode = "XYZ"
    o.rotation_euler = (0.0, 0.0, rz)
    return o


# ======================================================================================================== materials
def _materials(K):
    """{material: Blender material} for LAY.MATS: gravel, weathered wood, iron with rust, the polished railhead."""
    col = {role: K.role(role, LAY.ROLES) for role in LAY.ROLES}
    out = {}

    m, nt, o = K.new_mat("ballast")                     # stones: a Voronoi of cells in world space, domed by the bump
    geo = N(nt, "ShaderNodeNewGeometry", (-1400, 0))
    vor = N(nt, "ShaderNodeTexVoronoi", (-1150, 100), inputs={"Scale": 22.0, "Randomness": 0.9})
    L(nt, geo.outputs["Position"], vor.inputs["Vector"])
    dark = tuple(0.62 * c for c in col["ballast"][:3]) + (1.0,)
    rp = ramp(nt, (-700, 200), [(0.0, col["ballast_dark"]), (0.35, dark), (0.7, col["ballast"]), (1.0, col["ballast"])])
    L(nt, vor.outputs["Color"], rp.inputs["Fac"])                 # a cell's random colour, averaged: its stone's tone
    nz = N(nt, "ShaderNodeTexNoise", (-1150, -250), inputs={"Scale": 140.0, "Detail": 4.0})
    L(nt, geo.outputs["Position"], nz.inputs["Vector"])
    hgt = N(nt, "ShaderNodeMath", (-900, -100), operation="MULTIPLY_ADD", inputs={1: -1.0, 2: 1.0})
    L(nt, vor.outputs["Distance"], hgt.inputs[0])
    mix = N(nt, "ShaderNodeMath", (-700, -150), operation="MULTIPLY_ADD", inputs={1: 0.15})
    L(nt, nz.outputs["Fac"], mix.inputs[0])
    L(nt, hgt.outputs[0], mix.inputs[2])
    bump = N(nt, "ShaderNodeBump", (-450, -150), inputs={"Strength": 0.55, "Distance": 0.02})
    L(nt, mix.outputs[0], bump.inputs["Height"])
    b = principled(nt, o, (0, 0), Roughness=0.88, **{"Specular IOR Level": 0.3})
    L(nt, rp.outputs["Color"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    m.diffuse_color = col["ballast"]
    out["ballast"] = m

    m, nt, o = K.new_mat("wood")                        # weathered hewn wood: streaks and checks along the tie's length
    tc = N(nt, "ShaderNodeTexCoord", (-1700, 0))
    info = N(nt, "ShaderNodeObjectInfo", (-1700, -250))
    off = N(nt, "ShaderNodeVectorMath", (-1450, -200), operation="SCALE", inputs={"Scale": 37.0})
    L(nt, info.outputs["Random"], off.inputs[0])
    add = N(nt, "ShaderNodeVectorMath", (-1250, 0), operation="ADD")
    L(nt, tc.outputs["Object"], add.inputs[0])
    L(nt, off.outputs["Vector"], add.inputs[1])
    grain = N(nt, "ShaderNodeMapping", (-1050, 100), inputs={"Scale": (55.0, 1.6, 55.0)})
    L(nt, add.outputs["Vector"], grain.inputs["Vector"])
    streak = N(nt, "ShaderNodeTexNoise", (-850, 100), inputs={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.6})
    L(nt, grain.outputs["Vector"], streak.inputs["Vector"])
    checks_map = N(nt, "ShaderNodeMapping", (-1050, -200), inputs={"Scale": (9.0, 0.45, 9.0)})
    L(nt, add.outputs["Vector"], checks_map.inputs["Vector"])
    checks = N(nt, "ShaderNodeTexNoise", (-850, -200), inputs={"Scale": 1.0, "Detail": 2.0})
    L(nt, checks_map.outputs["Vector"], checks.inputs["Vector"])
    crack = ramp(nt, (-600, -200), [(0.47, (1.0, 1.0, 1.0, 1.0)), (0.5, (0.0, 0.0, 0.0, 1.0)),
                                    (0.53, (1.0, 1.0, 1.0, 1.0))])
    L(nt, checks.outputs["Fac"], crack.inputs["Fac"])
    rp = ramp(nt, (-600, 150), [(0.25, col["grain"]), (0.55, col["tie"]), (0.8, col["tie"])])
    L(nt, streak.outputs["Fac"], rp.inputs["Fac"])
    dk = N(nt, "ShaderNodeMix", (-300, 100), data_type="RGBA", blend_type="MULTIPLY", inputs={"Factor": 0.85})
    L(nt, rp.outputs["Color"], dk.inputs[6])
    L(nt, crack.outputs["Color"], dk.inputs[7])
    hmix = N(nt, "ShaderNodeMath", (-300, -250), operation="MULTIPLY")
    L(nt, streak.outputs["Fac"], hmix.inputs[0])
    L(nt, crack.outputs["Color"], hmix.inputs[1])
    bump = N(nt, "ShaderNodeBump", (-100, -250), inputs={"Strength": 0.35, "Distance": 0.004})
    L(nt, hmix.outputs[0], bump.inputs["Height"])
    b = principled(nt, o, (200, 0), Roughness=0.86, **{"Specular IOR Level": 0.35})
    L(nt, dk.outputs[2], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    m.diffuse_color = col["tie"]
    out["tie"] = m

    def iron(base, role, rough, metal, rust):
        m, nt, o = K.new_mat(base)
        tc = N(nt, "ShaderNodeTexCoord", (-1100, 0))
        nz = N(nt, "ShaderNodeTexNoise", (-850, 0), inputs={"Scale": 6.0, "Detail": 5.0})
        L(nt, tc.outputs["Object"], nz.inputs["Vector"])
        rp = ramp(nt, (-600, 0), [(0.35, col[role]), (0.75, tuple(c * (1 - rust) + r * rust for c, r in
                                                                    zip(col[role], col["rust"])))])
        L(nt, nz.outputs["Fac"], rp.inputs["Fac"])
        b = principled(nt, o, (0, 0), Roughness=rough, Metallic=metal)
        L(nt, rp.outputs["Color"], b.inputs["Base Color"])
        m.diffuse_color = col[role]
        return m

    out["rail"] = iron("iron", "rail", 0.62, 0.7, 0.35)
    out["railhead"] = K.simple_mat("railhead", col["railhead"], rough=0.28, metal=0.9)
    out["railhead"].diffuse_color = col["railhead"]
    out["spike"] = iron("spike", "spike", 0.55, 0.65, 0.25)
    out["plate"] = iron("plate", "plate", 0.6, 0.65, 0.4)
    return out


# ======================================================================================================== builder
@register("rail_track")
def rail_track(name, coll, root, slots=None):
    try:
        opts, plan = LAY.options(slots)
    except LAY.LayoutError as e:
        raise ValueError(f"prop {name!r}: {e}") from None
    K = Kit(name, coll, root, slots)
    _purge(name)
    clear_drivers(root)
    K.prop("laid", opts["laid"], doc=f"metres of track laid (0 .. {plan.done:.2f}: all of it in place); key it")
    mats = _materials(K)
    mat_of = lambda i: mats[LAY.MATS[i]]

    beds = [SH.mesh_data(f"{name}_bed_first", LAY.bed_mesh(plan.k, first=True), mat_of)[0],
            SH.mesh_data(f"{name}_bed", LAY.bed_mesh(plan.k), mat_of)[0]]
    for k in range(plan.n):
        x, y, h = (float(v) for v in plan.pose(k * RT.SPACING))
        o = _object(f"{name}_bed{k}", beds[k > 0], coll, root, (x, y, 0.0), h)
        ex = RT.bed_expr(plan, k)
        _drive(o, "scale", ex["sx"], root, 0)
        _drive(o, "scale", ex["syz"], root, 1)
        _drive(o, "scale", ex["syz"], root, 2)

    ties = [SH.mesh_data(f"{name}_tie_v{v}", LAY.tie_mesh(v, opts["seed"], opts["plates"]), mat_of)[0]
            for v in range(LAY.VARIANTS)]
    spikes = SH.mesh_data(f"{name}_spikes", LAY.spikes_mesh(opts["plates"]), mat_of)[0]
    for i, (v, dx, dy, yaw, tip) in enumerate(RT.tie_variety(plan, opts["seed"], LAY.VARIANTS, opts["plates"])):
        s = float(plan.ties[i])
        p = plan.point([s + dx], y=dy)[0]
        o = _object(f"{name}_tie{i}", ties[v], coll, root, (p[0], p[1], LAY.TIE_TOP),
                    float(plan.pose(s + dx)[2]) + math.radians(yaw))
        ex = RT.tie_expr(plan, i, LAY.TIE_TOP, RT.TILT * tip)
        _drive(o, "location", ex["z"], root, 2)
        _drive(o, "rotation_euler", ex["rx"], root, 0)
        for c in range(3):
            _drive(o, "scale", ex["s"], root, c)
        x, y, h = (float(a) for a in plan.pose(s))
        o = _object(f"{name}_spikes{i}", spikes, coll, root, (x, y, LAY.TIE_TOP - LAY.SPIKE_DROP), h)
        _drive(o, "scale", RT.spike_expr(plan, i)["sz"], root, 2)

    rails = {}
    seat = LAY.seat_z(opts["plates"])
    for j, (r, e) in enumerate(plan.spans):
        key = (round(e - r, 6), j > 0)
        if key not in rails:
            rails[key] = SH.mesh_data(f"{name}_rail_{e - r:.3f}{'_joint' if j else ''}",
                                      LAY.rail_mesh(plan, e - r, joint=j > 0), mat_of)[0]
        loc = (r, 0.0, seat) if plan.k == 0.0 else (plan.centre[0], plan.centre[1], seat)
        o = _object(f"{name}_rail{j}", rails[key], coll, root, loc)
        ex = RT.rail_expr(plan, j, seat)
        if plan.k == 0.0:
            _drive(o, "location", ex["slide"], root, 0)
        else:
            _drive(o, "rotation_euler", ex["slide"], root, 2)
        _drive(o, "location", ex["z"], root, 2)
        for c in range(3):
            _drive(o, "scale", ex["s"], root, c)

    rail_top = seat + RT.RAIL_H
    front = _object(f"{name}_front", None, coll, root, (0.0, 0.0, rail_top))
    front.empty_display_type, front.empty_display_size = "SINGLE_ARROW", 0.5
    for ch, expr in RT.front_expr(plan).items():
        if ch == "rz":
            _drive(front, "rotation_euler", expr, root, 2)
        else:
            _drive(front, "location", expr, root, "xy".index(ch))
    roles = {role: K.colors.hex(K.slots.get(role, LAY.ROLES[role])) for role in LAY.ROLES}
    return LAY.card(name, plan, opts, roles)
