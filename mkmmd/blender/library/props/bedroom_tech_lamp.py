"""The spring-arm desk lamp of the bedroom tech props (helper module, registers no builder): geometry, materials, the light
and the card of `desk_lamp`; its docstring in bedroom_tech.py is the reference.

Pose and frames: bedroom_tech_maths.lamp_joints gives the joints; every part is built in its own frame so the arms can be
turned by hand: `arm_lower` (origin = base pivot, +Z along the arm), `arm_upper` (origin = elbow, +Z along the arm) and
`shade` (origin = bulb centre, +Z toward the neck, the light leaves along -Z) form a parent chain below `base`."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import bedroom_tech_geo as G
from . import bedroom_tech_maths as M
from .bedroom_tech_tape import accent
from .cafe_kit import L, N, bm_box, bm_lathe, bm_tube, drive, principled

# ------------------------------------------------------------------ dimensions (m)
BASE_R, BASE_H = 0.080, 0.0272          # weighted base: radius and height
PAD_R = 0.0715                          # felt / rubber pad under it
CHEEK_X = 0.0165                        # centre of the two cheeks that hold the lower arm's pivot
BAR_OFF = 0.0125                        # the two bars of an arm sit at +-BAR_OFF (in the plane of the arm)
BAR_R = 0.0040                          # bar radius
SPRING_X = 0.0255                       # springs run outside the arm at x = +-SPRING_X
SPRING_R, WIRE_R, PITCH = 0.0058, 0.00085, 0.0105
SHADE_R = 0.0675                        # rim radius
Z_APEX, Z_EQ, Z_RIM = 0.058, -0.012, -0.042       # shade axis: apex, widest ring, rim plane
NECK_R, NECK_Z0, NECK_Z1 = 0.0158, 0.050, 0.094   # socket housing behind the dome
TH = 0.0013                             # sheet thickness of the shade
BULB_TOP = 0.048                        # top of the bulb's metal base (z in the shade frame)
SPOT_DEG, SPOT_BLEND, SPOT_RADIUS = 75.0, 0.5, 0.03
DEFAULT_POWER = 60.0                    # W of the spot at `power`, `on` = 1
BULB_STRENGTH = 7.0                     # emission strength of the glass at `on` = 1
FORK_START, FORK_END = 0.01575, 0.0185  # x of the plates of a fork at the start / at the end of an arm

_ROT_X = Matrix.Rotation(math.pi / 2, 3, "Y")          # +Z -> +X
_ROT_NX = Matrix.Rotation(-math.pi / 2, 3, "Y")         # +Z -> -X


# ================================================================= materials
def make_materials(K, paint_spec):
    """dict of the lamp's materials: paint, inner (shade interior, lit), steel, rubber, knob, bulb."""
    pa, _ = accent(K, paint_spec)
    mats = {}
    # painted metal: semi-gloss enamel with a fine orange-peel bump and a clear coat that gives the bright rim
    m, nt, out = K.new_mat("paint")
    tc = N(nt, "ShaderNodeTexCoord", (-1100, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-900, 0), inputs={"Scale": 420.0, "Detail": 4.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (-600, -200), inputs={"Strength": 0.03, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    rr = N(nt, "ShaderNodeMapRange", (-600, 100), inputs={"From Min": 0.3, "From Max": 0.7, "To Min": 0.30, "To Max": 0.40})
    L(nt, nz.outputs["Fac"], rr.inputs["Value"])
    b = principled(nt, out, **{"Base Color": pa, "Specular IOR Level": 0.55, "Coat Weight": 0.5, "Coat Roughness": 0.08})
    L(nt, rr.outputs["Result"], b.inputs["Roughness"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    mats["paint"] = m
    # shade interior: pale warm enamel that glows from the bulb (more near the neck, less at the rim)
    m, nt, out = K.new_mat("inner")
    on = K.param(nt, "on", loc=(-1100, 300))
    tc = N(nt, "ShaderNodeTexCoord", (-1100, -100))
    sp = N(nt, "ShaderNodeSeparateXYZ", (-900, -100))
    L(nt, tc.outputs["Object"], sp.inputs["Vector"])
    fall = N(nt, "ShaderNodeMapRange", (-700, -100),
             inputs={"From Min": Z_RIM, "From Max": Z_APEX, "To Min": 0.30, "To Max": 1.0})
    fall.clamp = True
    L(nt, sp.outputs["Z"], fall.inputs["Value"])
    g = N(nt, "ShaderNodeMath", (-450, 100), operation="MULTIPLY", inputs={1: 1.8})
    L(nt, fall.outputs["Result"], g.inputs[0])
    g2 = N(nt, "ShaderNodeMath", (-250, 100), operation="MULTIPLY")
    L(nt, g.outputs[0], g2.inputs[0])
    L(nt, on, g2.inputs[1])
    b = principled(nt, out, **{"Base Color": K.blend(text=.55, gold=.45), "Roughness": 0.42, "Specular IOR Level": 0.4,
                               "Emission Color": K.light(gold=.75, rose=.15, text=.10), "Emission Strength": 0.0})
    L(nt, g2.outputs[0], b.inputs["Emission Strength"])
    mats["inner"] = m
    mats["steel"] = K.simple_mat("steel", K.blend(subtle=.5, text=.3, hl_high=.2), rough=0.30, metal=0.85, spec=0.6)
    mats["rubber"] = K.simple_mat("rubber", K.blend(surface=.6, base=.4), rough=0.85, spec=0.2)
    mats["knob"] = K.simple_mat("knob", K.blend(overlay=.6, hl_med=.4), rough=0.45, spec=0.4, coat=0.2)
    # bulb: frosted warm glass, brighter in the middle than at the edge, follows `on`
    m, nt, out = K.new_mat("bulb")
    on = K.param(nt, "on", loc=(-1100, 300))
    geo = N(nt, "ShaderNodeNewGeometry", (-1100, -100))
    dt = N(nt, "ShaderNodeVectorMath", (-900, -100), operation="DOT_PRODUCT")
    L(nt, geo.outputs["Normal"], dt.inputs[0])
    L(nt, geo.outputs["Incoming"], dt.inputs[1])
    mx = N(nt, "ShaderNodeMath", (-700, -100), operation="MAXIMUM", inputs={1: 0.0})
    L(nt, dt.outputs["Value"], mx.inputs[0])
    fc = N(nt, "ShaderNodeMath", (-520, -100), operation="MULTIPLY_ADD", inputs={1: 0.45, 2: 0.55})
    L(nt, mx.outputs[0], fc.inputs[0])
    s1 = N(nt, "ShaderNodeMath", (-340, 100), operation="MULTIPLY", inputs={1: BULB_STRENGTH})
    L(nt, fc.outputs[0], s1.inputs[0])
    s2 = N(nt, "ShaderNodeMath", (-160, 100), operation="MULTIPLY")
    L(nt, s1.outputs[0], s2.inputs[0])
    L(nt, on, s2.inputs[1])
    bulb_c = K.blend(gold=.65, text=.35)
    b = principled(nt, out, **{"Base Color": bulb_c, "Roughness": 0.3, "Emission Color": bulb_c, "Specular IOR Level": 0.5})
    L(nt, s2.outputs[0], b.inputs["Emission Strength"])
    mats["bulb"] = m
    return mats


# ================================================================= small mesh helpers
def cyl_x(bm, x0, x1, r, loc=(0.0, 0.0), mat=0, segs=14):
    """Cylinder along X from x0 to x1 at (y, z) = loc, radius r."""
    b = bmesh.new()
    bm_lathe(b, [(0.0, 0.0), (r, 0.0), (r, x1 - x0), (0.0, x1 - x0)], segs=segs, mat=mat, uv=False)
    G.put(bm, b, loc=(x0, loc[0], loc[1]), rot=_ROT_X)


def dome_x(bm, x, r, h, sign, loc=(0.0, 0.0), mat=0, segs=14):
    """Rounded screw head of radius r and height h standing on a face at x, bulging toward `sign` (+1 / -1) along X."""
    b = bmesh.new()
    bm_lathe(b, [(0.0, 0.0), (r, 0.0), (r, h * 0.35), (r * 0.80, h * 0.80), (r * 0.45, h), (0.0, h)], segs=segs, mat=mat,
             uv=False)
    G.put(bm, b, loc=(x, loc[0], loc[1]), rot=_ROT_X if sign > 0 else _ROT_NX)


def rod(bm, a, b, r, mat=0, sides=10):
    pts = [Vector(a), (Vector(a) + Vector(b)) * 0.5, Vector(b)]
    bm_tube(bm, pts, r, sides=sides, mat=mat, cap="flat", uv=False)


# ================================================================= base
def build_base(P):
    """The weighted base: pad (material 3 rubber), the disc with a rolled edge and domed top (0 paint), a pivot boss and the
    two cheeks that hold the lower arm (0 paint), their screw heads (1 steel). `P` = lamp_joints."""
    bm = bmesh.new()
    prof = [(0.0, 0.0), (PAD_R - 0.0010, 0.0), (PAD_R, 0.0012), (PAD_R, 0.0012),
            (BASE_R - 0.0012, 0.0012), (BASE_R, 0.0032), (BASE_R + 0.0002, 0.0100), (BASE_R - 0.0004, 0.0172),
            (BASE_R - 0.0040, 0.0228), (BASE_R - 0.0100, 0.0250), (0.048, 0.0262), (0.026, 0.0270), (0.0, 0.0274)]
    bm_lathe(bm, prof, segs=72, mat=0, mat_ranges=[(0, 3), (3, 0)])
    py, pz = P["pivot"][1], P["pivot"][2]
    b = bmesh.new()                                    # pivot boss
    bm_lathe(b, [(0.0, 0.0), (0.0300, 0.0), (0.0300, 0.0045), (0.0270, 0.0100), (0.0, 0.0105)], segs=40, mat=0, uv=False)
    G.put(bm, b, loc=(0.0, py, BASE_H - 0.0006))
    for s in (-1, 1):                                  # cheeks: a plate with a rounded top about the pivot
        lo, hi = (CHEEK_X - 0.0020), (CHEEK_X + 0.0020)
        if s < 0:
            lo, hi = -hi, -lo
        h = pz - BASE_H
        bm_box(bm, (hi - lo, 0.0372, h), loc=((lo + hi) / 2, py, BASE_H + h / 2), mat=0)
        cyl_x(bm, lo, hi, 0.0186, loc=(py, pz), mat=0, segs=28)
        dome_x(bm, hi if s > 0 else lo, 0.0062, 0.0030, s, loc=(py, pz), mat=1)
    G.soften(bm, 0.0012)
    return bm


# ================================================================= arms
def _block(bm, z):
    """Narrow end barrel of an arm (it sits between the base cheeks / inside the elbow fork)."""
    cyl_x(bm, -0.0105, 0.0105, 0.0185, loc=(0.0, z), mat=0, segs=28)


def _fork(bm, z, inward, half):
    """Fork at an arm end: two round plates at x = +-half joined by a cross bar on the arm side (`inward` = +1 when the arm
    continues toward +Z, -1 when it continues toward -Z)."""
    for s in (-1, 1):
        cyl_x(bm, s * half - 0.00175, s * half + 0.00175, 0.0185, loc=(0.0, z), mat=0, segs=28)
    cb = bmesh.new()                                   # the cross piece: a pillow with 3.5 mm rounded edges, not a box
    bm_box(cb, (2 * half + 0.0035, 0.0370, 0.0100), loc=(0.0, 0.0, 0.0), mat=0)
    G.soften(cb, 0.0035, segments=3)
    G.put(bm, cb, loc=(0.0, 0.0, z + inward * 0.0170))


def build_arm(length, start, end, elbow_knob=False):
    """One arm in its own frame: +Z from the start pivot (origin) to the end pivot (0, 0, length). Two painted bars at
    y = +-BAR_OFF, end pieces ('block' or 'fork'), steel pins, and two tension springs outside the bars. Materials:
    0 paint, 1 steel, 2 knob."""
    bm = bmesh.new()
    z0 = 0.0085 if start == "block" else 0.0270
    z1 = length - (0.0085 if end == "block" else 0.0270)
    for s in (-1, 1):
        rod(bm, (0.0, s * BAR_OFF, z0 - 0.003), (0.0, s * BAR_OFF, z1 + 0.003), BAR_R, mat=0, sides=10)
    if start == "block":
        _block(bm, 0.0)
    else:
        _fork(bm, 0.0, +1, FORK_START)
    if end == "block":
        _block(bm, length)
    else:
        _fork(bm, length, -1, FORK_END)
    # pins: through the base cheeks (block start: their screw heads are on the base), through an elbow fork or the neck
    if start == "fork":
        h = FORK_START + 0.00175
        cyl_x(bm, -h, h, 0.0042, loc=(0.0, 0.0), mat=1, segs=12)
        dome_x(bm, -h, 0.0062, 0.0030, -1, loc=(0.0, 0.0), mat=1)
        dome_x(bm, h, 0.0062, 0.0030, +1, loc=(0.0, 0.0), mat=1)
    if end == "fork":
        h = FORK_END + 0.00175
        cyl_x(bm, -h, h, 0.0042, loc=(0.0, length), mat=1, segs=12)
        dome_x(bm, -h, 0.0062, 0.0030, -1, loc=(0.0, length), mat=1)
        dome_x(bm, h, 0.0062, 0.0030, +1, loc=(0.0, length), mat=1)
    if start == "block":
        cyl_x(bm, -(CHEEK_X + 0.0020), CHEEK_X + 0.0020, 0.0042, loc=(0.0, 0.0), mat=1, segs=12)
    if elbow_knob:       # a big ridged knob on the +X end of the elbow pin
        kb = bmesh.new()
        bm_lathe(kb, [(0.0, 0.0), (0.0140, 0.0), (0.0148, 0.0012), (0.0148, 0.0075), (0.0140, 0.0088), (0.0075, 0.0094),
                      (0.0, 0.0094)], segs=24, mat=2, uv=False)
        G.put(bm, kb, loc=(FORK_START + 0.0045, 0.0, 0.0), rot=_ROT_X)
    # spring seats: rivets through the bars carry the spring hooks
    a_z, b_z = 0.034, length - 0.034
    seats = []
    for s in (-1, 1):
        xa = s * SPRING_X
        pa, pb = (xa, -BAR_OFF, a_z), (xa, +BAR_OFF, b_z)
        seats.append((pa, pb))
        for yy, zz in ((pa[1], pa[2]), (pb[1], pb[2])):
            lo, hi = sorted((0.0, xa + s * 0.0030))
            cyl_x(bm, lo, hi, 0.0026, loc=(yy, zz), mat=1, segs=8)
    G.soften(bm, 0.0010)                       # the solids: every hard edge rounded; then the springs' wire is added
    for pa, pb in seats:
        pts = [Vector(p) for p in M.coil_path(pa, pb, SPRING_R, PITCH, per_turn=12)]
        bm_tube(bm, pts, WIRE_R, sides=6, mat=1, cap="none", uv=False)
    return bm


# ================================================================= shade and bulb
def shade_profile():
    """(r, z) profile of the shade's dome, in its frame: from the inner apex down the inside, across the rim edge and up the
    outside to the outer apex. -> (points, index of the first outside segment)."""
    n = 16
    outer = []
    for k in range(n + 1):
        t = 0.5 * math.pi * k / n
        outer.append((SHADE_R * math.sin(t), Z_EQ + (Z_APEX - Z_EQ) * math.cos(t)))          # apex -> equator
    outer.append((SHADE_R * 1.010, Z_RIM + 0.0060))                                              # slight flare to the rim
    outer.append((SHADE_R * 1.014, Z_RIM))
    inner = []
    for i, (r, z) in enumerate(outer):
        a, b = outer[max(i - 1, 0)], outer[min(i + 1, len(outer) - 1)]
        dr, dz = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dr, dz) or 1.0
        nr, nz = -dz / ln, dr / ln                          # outward normal of the apex -> rim polyline
        inner.append((max(r - nr * TH, 0.0), z - nz * TH))
    inner[0] = (0.0, Z_APEX - TH)
    # the sheet's edge at the rim: a half round of the sheet's thickness (no hard edge; the rolled bead hides it)
    (ri, zi), (ro, zo) = inner[-1], outer[-1]
    cr, cz = 0.5 * (ri + ro), 0.5 * (zi + zo)
    rho, a0 = 0.5 * math.hypot(ro - ri, zo - zi), math.atan2(zi - cz, ri - cr)
    edge = [(cr + rho * math.cos(a0 + math.pi * k / 6), cz + rho * math.sin(a0 + math.pi * k / 6)) for k in range(1, 6)]
    prof = inner + edge + list(reversed(outer))
    return prof, len(inner) - 1


def build_shade():
    """The shade in its frame (origin = bulb centre, axis Z, opening toward -Z): dome (0 paint outside incl. the rim bead, 1
    inner glow inside), socket housing at the back (0 paint) with a steel ring (2) and the lamp holder inside (2)."""
    prof, i_out = shade_profile()
    bm = bmesh.new()
    bm_lathe(bm, prof, segs=64, mat=0, mat_ranges=[(0, 1), (i_out, 0)])
    rb = 0.0022                                                              # the rolled bead on the rim
    rim_r = SHADE_R * 1.014
    ring = [(rim_r - 0.0004 + rb * math.cos(2 * math.pi * k / 12), Z_RIM + rb * 0.55 + rb * math.sin(2 * math.pi * k / 12))
            for k in range(13)]
    b = bmesh.new()
    bm_lathe(b, ring, segs=64, mat=0, uv=False)
    G.put(bm, b)
    b = bmesh.new()                                                          # socket housing (hollow below, closed at the back)
    bm_lathe(b, [(0.0, NECK_Z0), (NECK_R, NECK_Z0), (NECK_R, NECK_Z1 - 0.004), (NECK_R - 0.0030, NECK_Z1),
                 (0.0, NECK_Z1)], segs=32, mat=0, uv=False)
    G.put(bm, b)
    b = bmesh.new()
    bm_lathe(b, [(NECK_R, 0.0600), (NECK_R + 0.0014, 0.0600), (NECK_R + 0.0014, 0.0625), (NECK_R, 0.0625)], segs=32, mat=2,
             uv=False)
    G.put(bm, b)
    b = bmesh.new()                                                          # lamp holder inside the dome
    bm_lathe(b, [(0.0124, 0.0300), (0.0124, NECK_Z0 + 0.002), (0.0, NECK_Z0 + 0.002)], segs=24, mat=2, uv=False)
    G.put(bm, b)
    G.soften(bm, 0.0007)
    return bm


def build_bulb():
    """Frosted glass bulb (material 0 bulb) with a screw base (1 steel); origin = bulb centre, neck toward +Z."""
    bm = bmesh.new()
    glass = [(0.0, -0.0315), (0.0100, -0.0300), (0.0185, -0.0250), (0.0245, -0.0165), (0.0272, -0.0070), (0.0278, 0.0030),
             (0.0262, 0.0130), (0.0225, 0.0215), (0.0170, 0.0285), (0.0125, 0.0330), (0.0108, 0.0360)]
    bm_lathe(bm, glass, segs=32, mat=0)
    b = bmesh.new()
    bm_lathe(b, [(0.0108, 0.0360), (0.0105, 0.0372), (0.0108, 0.0384), (0.0105, 0.0396), (0.0108, 0.0408),
                 (0.0105, 0.0420), (0.0100, BULB_TOP), (0.0, BULB_TOP)], segs=24, mat=1, uv=False)
    G.soften(b, 0.0005)
    G.put(bm, b)
    return bm


# ================================================================= colliders
def collider_capsule(K, base, parent, a, b, r):
    """A hidden capsule (a, b in the frame of `parent`, radius r) drawn as a wire mesh; returns (object, spec)."""
    bm = bmesh.new()
    pa, pb = Vector(a), Vector(b)
    ax = (pb - pa)
    ln = ax.length
    rot = Vector((0, 0, 1)).rotation_difference(ax.normalized()).to_matrix()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=12, radius1=r, radius2=r, depth=ln)
    bmesh.ops.translate(bm, vec=Vector((0, 0, ln / 2)), verts=bm.verts)
    for z0, flip in ((0.0, -1.0), (ln, 1.0)):
        g = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=r)
        bmesh.ops.translate(bm, vec=Vector((0, 0, z0)), verts=g["verts"])
    bmesh.ops.transform(bm, matrix=Matrix.Translation(pa) @ rot.to_4x4(), verts=bm.verts)
    o = K.collider(K.to_obj(base, bm, [], parent=parent))
    return o, {"type": "capsule", "object": o.name, "a": [round(v, 4) for v in a], "b": [round(v, 4) for v in b],
               "R": round(r, 4), "rnd": 0.004, "tag": K.name}


# ================================================================= assembly
def _local(parent_world, world):
    """Transform of the world matrix `world` relative to `parent_world` -> (location, euler angles) for K.to_obj."""
    rel = parent_world.inverted() @ world
    loc, quat, _ = rel.decompose()
    return tuple(loc), tuple(quat.to_euler())


def assemble(K, paint_spec, power=DEFAULT_POWER):
    """Build the lamp's objects, light and custom properties. -> dict with the joints `P`, the collider specs, and the objects
    `light`, `shade`, `bulb`, `base`."""
    P = M.lamp_joints()
    K.prop("power", float(power), 0.0, None, "power of the lamp's spot light in watts when `on` is 1")
    K.prop("on", 1.0, 0.0, 1.0, "0 = off, 1 = on: scales the light, the bulb's glow and the shade's glow")
    mats = make_materials(K, paint_spec)
    m_paint, m_inner, m_steel, m_rubber = mats["paint"], mats["inner"], mats["steel"], mats["rubber"]

    # world matrices of the parts in the prop frame
    pivot, elbow, bulb = (Vector(P[k]) for k in ("pivot", "elbow", "bulb"))
    th1 = M.rot_x_for(P["lower_dir"])
    th2 = M.rot_x_for(P["upper_dir"])
    ths = M.rot_x_for(tuple(-c for c in P["out"]))
    W1 = Matrix.Translation(pivot) @ Matrix.Rotation(th1, 4, "X")
    W2 = Matrix.Translation(elbow) @ Matrix.Rotation(th2, 4, "X")
    WS = Matrix.Translation(bulb) @ Matrix.Rotation(ths, 4, "X")

    base = K.to_obj("base", build_base(P), [m_paint, m_steel, mats["knob"], m_rubber])
    loc, rot = _local(Matrix.Identity(4), W1)
    arm1 = K.to_obj("arm_lower", build_arm(M.LAMP["lower_len"], "block", "block"), [m_paint, m_steel, mats["knob"]],
                    loc=loc, rot=rot, parent=base)
    loc, rot = _local(W1, W2)
    arm2 = K.to_obj("arm_upper", build_arm(M.LAMP["upper_len"], "fork", "fork", elbow_knob=True),
                    [m_paint, m_steel, mats["knob"]], loc=loc, rot=rot, parent=arm1)
    loc, rot = _local(W2, WS)
    shade = K.to_obj("shade", build_shade(), [m_paint, m_inner, m_steel], loc=loc, rot=rot, parent=arm2)
    bulb_o = K.to_obj("bulb", build_bulb(), [mats["bulb"], m_steel], parent=shade)
    bulb_o.visible_shadow = False

    # the light: a warm spot at the bulb, looking out of the shade (-Z of the shade frame)
    lname = K.oname("light")
    K.purge(lname)
    old = bpy.data.lights.get(lname)
    if old is not None and old.users == 0:
        bpy.data.lights.remove(old)
    ld = bpy.data.lights.new(lname, "SPOT")
    ld.color = K.light(gold=.75, rose=.15, text=.10)[:3]
    ld.energy = float(power)
    ld.use_shadow = True
    ld.spot_size = math.radians(SPOT_DEG)
    ld.spot_blend = SPOT_BLEND
    ld.shadow_soft_size = SPOT_RADIUS
    lo = K.obj("light", ld, loc=(0.0, 0.0, -0.002), parent=shade)
    fc = drive(ld, "energy", "power * on", var=("power", K.root, '["power"]'))
    v = fc.driver.variables.new()
    v.name = "on"
    v.type = "SINGLE_PROP"
    v.targets[0].id_type = "OBJECT"
    v.targets[0].id = K.root
    v.targets[0].data_path = '["on"]'

    # colliders: base cylinder, one capsule per arm segment, the shade as a sphere
    cols = []
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=BASE_R, radius2=BASE_R, depth=BASE_H)
    col_base = K.collider(K.to_obj("col_base", bm, [], loc=(0.0, 0.0, BASE_H / 2), parent=base))
    cols.append({"type": "cylinder", "object": col_base.name, "R": BASE_R, "half_h": round(BASE_H / 2, 4), "rnd": 0.004,
                 "tag": K.name})
    _, s1 = collider_capsule(K, "col_arm_lower", arm1, (0, 0, 0), (0, 0, M.LAMP["lower_len"]), 0.030)
    _, s2 = collider_capsule(K, "col_arm_upper", arm2, (0, 0, 0), (0, 0, M.LAMP["upper_len"]), 0.030)
    cols += [s1, s2]
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=SHADE_R)
    col_shade = K.collider(K.to_obj("col_shade", bm, [], loc=(0.0, 0.0, 0.010), parent=shade))
    cols.append({"type": "sphere", "object": col_shade.name, "c": [0.0, 0.0, 0.0], "R": SHADE_R, "tag": K.name})
    return {"P": P, "colliders": cols, "light": lo, "shade": shade, "bulb": bulb_o, "base": base}
