"""Cafe mug set: a cream saucer, and a hand-built speckled mug with a D handle, amber tea, a paper tea-bag tag on a cotton
string and a wisp of steam. Two builders, each in its own frame (+Z up, metres):

    cafe_saucer   origin = centre of the underside, on the table top (the contact point). Hand-thrown wobble, cream satin
                  glaze with sparse speckles, bare foot ring. The well floor is MUG_SEAT_Z (5.8 mm) above the origin.
    cafe_mug      origin = centre of the foot ring base, where the mug stands in the saucer well: a project places it at
                  `saucer at + (0, 0, 0.0058)`; yawing the root spins it in place. Handle azimuth -25 deg (0 = +X, -90 = -Y),
                  tag and string at 215 deg (opposite the handle). Stoneware (rose outside, cream inside, bare-clay foot,
                  throwing rings, 95 mm tall), the tea surface 13.5 mm below the lip, the paper tag, the string over the lip
                  and the steam are separate objects, all children of the root with identity transforms (their meshes live
                  in the mug frame).

Custom properties (cafe_mug root):
    steam   0..1, default 0.6   steam strength: plume alpha and wisp density. Key it on the root.

The steam noise scrolls with the scene time (frame / fps) and the plume sways through two shape keys driven by sin(time),
so both follow the scene's frame rate. The materials are procedural in object space (metres); the shaders read the vertex
attributes `glaze_in` (mug: 0 = outside glaze, 1 = inside glaze, blended over the lip) and `seed` (steam: one per piece).
"""
import math
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Vector

from . import register
from .cafe_kit import Kit, L, N, bm_lathe, bm_tube, catmull, drive, mix, principled, ramp

MUG_HANDLE_DEG = -25.0                    # handle azimuth: 0 = +X, -90 = -Y
MUG_TAG_DEG = 215.0                       # tea tag / string azimuth, opposite the handle
MUG_H = 0.095                             # mug height (lip top above the foot)
MUG_SAUCER_R = 0.078                      # saucer outer radius
MUG_SAUCER_H = 0.016                      # saucer height
MUG_SEAT_Z = 0.0058                       # height of the saucer well floor above the saucer origin = mug origin height
MUG_TEA_Z = MUG_H - 0.0135                # tea surface height in the mug frame (13.5 mm below the lip)

_MG_LIPR = 0.0025                         # lip radius (wall at the lip = 5 mm)
_MG_ZW0 = 0.0066                          # the straight body wall starts here (above the foot-ring shoulder)
_MG_ZW1 = MUG_H - _MG_LIPR                # ... and ends where the lip arc begins
_MG_R0, _MG_R1 = 0.0361, 0.0425           # outer radius at ZW0 / ZW1 (slightly tapered, wider at the lip)
_MG_FLOOR = 0.0075                        # inner floor height
_MG_SEGS = 120
_MG_TAG_W, _MG_TAG_H, _MG_TAG_T = 0.014, 0.022, 0.0003     # paper tag size and thickness
_MG_TAG_HY = _MG_TAG_H / 2 - 0.0035                         # punched hole centre (tag-local y, origin = sheet centre)
_MG_TAG_ZH = MUG_H - 0.027                                  # hole height on the mug wall
_MG_TAG_TILT = math.radians(-5.0)                           # the hanging tag sits a little crooked
_MG_STRING_R = 0.0004                                       # cotton string radius

_MG_HANDLE_F = 0.0035                       # fillet radius where the handle grows out of the wall
_MG_HANDLE_D = 0.010                        # distance over which the wall curvature is blended out of the handle rings
_MG_HANDLE_ZT, _MG_HANDLE_ZB = 0.0800, 0.0255                                        # attachment heights (top, bottom)
_MG_HANDLE_RA = [(0.0, 0.0072), (0.10, 0.0078), (0.30, 0.0070), (0.55, 0.0064), (0.80, 0.0060), (1.0, 0.0063)]
_MG_HANDLE_RB = [(0.0, 0.0050), (0.10, 0.0042), (0.30, 0.0046), (0.55, 0.0048), (0.80, 0.0043), (1.0, 0.0047)]


# ---------------------------------------------------------------- tiny maths helpers
def _mg_ss(a, b, x):
    t = (x - a) / (b - a)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def _mg_key(keys, t):
    """Piecewise smooth interpolation of [(t, value), ...] (sorted by t)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            u = u * u * (3.0 - 2.0 * u)
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def _mg_prof_smooth(pts, n=4):
    """Catmull-Rom through (r, z) key points -> dense (r, z) list."""
    sm = catmull([(r, z, 0.0) for r, z in pts], n)
    return [(p.x, p.y) for p in sm]


# ---------------------------------------------------------------- mug wall (pure maths, mug frame, z up)
_MG_RING_K = 2 * math.pi / 0.0099                  # throwing-ring pitch ~ 10 mm


def _mg_ring(z):
    """Throwing-ring relief (metres) at height z: irregular pitch, amplitude 0.05 .. 0.08 mm."""
    ph = _MG_RING_K * (z - 0.012) + 1.1 * math.sin(83.0 * z) + 0.5 * math.sin(37.0 * z)
    return 0.00008 * (0.65 + 0.35 * math.sin(53.0 * z + 0.7)) * math.sin(ph)


def _mg_r_out(z):
    """Outer wall radius at height z (z >= ZW0) before the hand-made wobble: gentle taper + throwing rings."""
    s = (z - _MG_ZW0) / (_MG_ZW1 - _MG_ZW0)
    r = _MG_R0 + (_MG_R1 - _MG_R0) * (0.80 * s + 0.20 * s * s)
    env = _mg_ss(0.0, 0.10, s) * (1.0 - _mg_ss(0.90, 1.0, s))
    r += _mg_ring(z) * env                                                                         # throwing rings
    r += 0.00028 * math.sin(2 * math.pi * z / 0.062 + 1.3)                                         # hand-built wave
    return r


def _mg_wall(z):
    """Wall thickness: 5 mm at the lip, thicker towards the base."""
    s = max(0.0, min(1.0, (z - _MG_ZW0) / (_MG_ZW1 - _MG_ZW0)))
    return 0.0050 + 0.0016 * (1.0 - s) ** 2


def _mg_r_in(z):
    return _mg_r_out(z) - _mg_wall(z)


def _mg_wob(th, z):
    """Multiplicative radial wobble (hand-made, out-of-round, growing towards the lip)."""
    s = max(0.0, min(1.0, z / MUG_H))
    return (1.0 + 0.0030 * math.cos(2 * th - 0.7) * (0.25 + 0.75 * s)
            + 0.0022 * math.cos(th + 2.1) * s
            + 0.0012 * math.cos(3 * th + 0.4) * s * s)


def _mg_dz(th, z):
    """Lip height wobble (the rim is not perfectly level)."""
    s = max(0.0, min(1.0, z / MUG_H)) ** 3
    return s * (0.00030 * math.sin(th - 0.9) + 0.00014 * math.sin(2 * th + 0.3))


def _mg_wall_pt(z, th, off=0.0):
    """Point on the outer wall at profile height z / azimuth th (offset `off` along the wall normal)."""
    r = _mg_r_out(z) * _mg_wob(th, z)
    p = Vector((r * math.cos(th), r * math.sin(th), z + _mg_dz(th, z)))
    if off:
        p += _mg_wall_n(z, th) * off
    return p


def _mg_wall_n(z, th):
    e = 2e-4
    dz = (_mg_wall_pt(z + e, th) - _mg_wall_pt(z - e, th))
    dt = (_mg_wall_pt(z, th + 1e-3) - _mg_wall_pt(z, th - 1e-3))
    n = dt.cross(dz).normalized()
    if n.x * math.cos(th) + n.y * math.sin(th) < 0:
        n = -n
    return n


def _mg_in_pt(z, th, off=0.0):
    """Point on the INNER wall (offset `off` towards the axis)."""
    r = _mg_r_in(z) * _mg_wob(th, z)
    p = Vector((r * math.cos(th), r * math.sin(th), z + _mg_dz(th, z)))
    if off:
        p -= Vector((math.cos(th), math.sin(th), 0.0)) * off
    return p


# ---------------------------------------------------------------- shader node helpers (return output sockets)
def _mg_loc(nt):
    k = len(nt.nodes)
    return (-2300 + 205 * (k % 16), 800 - 270 * (k // 16))


def _mg_set(nt, n, i, v):
    if v is None:
        return
    if isinstance(v, bpy.types.NodeSocket):
        nt.links.new(v, n.inputs[i])
    else:
        n.inputs[i].default_value = v


def _mg_math(nt, op, a, b=None, c=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.location = _mg_loc(nt)
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b, c)):
        _mg_set(nt, n, i, v)
    return n.outputs[0]


def _mg_sstep(nt, lo, hi, x):
    """smoothstep(lo, hi, x) via Map Range; lo / hi / x may be sockets or floats."""
    n = nt.nodes.new("ShaderNodeMapRange")
    n.location = _mg_loc(nt)
    n.interpolation_type = "SMOOTHSTEP"
    n.clamp = True
    _mg_set(nt, n, 0, x)
    _mg_set(nt, n, 1, lo)
    _mg_set(nt, n, 2, hi)
    n.inputs[3].default_value = 0.0
    n.inputs[4].default_value = 1.0
    return n.outputs[0]


def _mg_mixf(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.location = _mg_loc(nt)
    n.data_type = "FLOAT"
    _mg_set(nt, n, 0, fac)
    _mg_set(nt, n, 2, a)
    _mg_set(nt, n, 3, b)
    return n.outputs[0]


def _mg_mixc(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.location = _mg_loc(nt)
    n.data_type = "RGBA"
    _mg_set(nt, n, 0, fac)
    _mg_set(nt, n, 6, a)
    _mg_set(nt, n, 7, b)
    return n.outputs[2]


def _mg_xyz(nt, x, y, z):
    n = nt.nodes.new("ShaderNodeCombineXYZ")
    n.location = _mg_loc(nt)
    for i, v in enumerate((x, y, z)):
        _mg_set(nt, n, i, v)
    return n.outputs[0]


def _mg_sepxyz(nt, vec):
    n = nt.nodes.new("ShaderNodeSeparateXYZ")
    n.location = _mg_loc(nt)
    nt.links.new(vec, n.inputs[0])
    return n.outputs[0], n.outputs[1], n.outputs[2]


def _mg_noise(nt, vec, scale, detail=2.0, rough=0.5, w=None):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.location = _mg_loc(nt)
    if w is not None:
        n.noise_dimensions = "4D"
        _mg_set(nt, n, 1, w)
    nt.links.new(vec, n.inputs[0])
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    return n.outputs["Fac"]


def _mg_speck(nt, vec, scale, gate, rmin, rmax):
    """Round speckles: one random-radius dot per Voronoi cell (cell = 1/scale metres), kept when a per-cell random exceeds
    `gate` (0 = every cell, 1 = none; may be a socket).  Dot radius rmin..rmax in cell units.  Returns a 0..1 mask."""
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.location = _mg_loc(nt)
    vo.voronoi_dimensions = "3D"
    vo.feature = "F1"
    vo.inputs["Scale"].default_value = scale
    vo.inputs["Randomness"].default_value = 1.0
    nt.links.new(vec, vo.inputs["Vector"])
    sp = nt.nodes.new("ShaderNodeSeparateColor")
    sp.location = _mg_loc(nt)
    nt.links.new(vo.outputs["Color"], sp.inputs[0])
    rad = _mg_math(nt, "MULTIPLY_ADD", sp.outputs[1], rmax - rmin, rmin)
    lo = _mg_math(nt, "MULTIPLY", rad, 0.55)
    out_of_dot = _mg_sstep(nt, lo, rad, vo.outputs["Distance"])
    dot = _mg_math(nt, "SUBTRACT", 1.0, out_of_dot)
    keep = _mg_math(nt, "GREATER_THAN", sp.outputs[0], gate)
    return _mg_math(nt, "MULTIPLY", dot, keep)


# ---------------------------------------------------------------- colours
def _mg_colors(K):
    """The set's colours as linear RGBA: palette blends fitted to the original hand-picked values (Rose Pine Dawn)."""
    b = K.blend
    return {
        "rose": K.slot("rose"),                                           # mug glaze outside
        "love": K.slot("love"),                                           # deeper pooled glaze, tag text bars
        "oat": b(overlay=.7, gold=.2, foam=.1),                           # pale clay mixed into the thin glaze
        "speck": b(text=.75, gold=.25),                                   # muted plum-brown speckle
        "cream_lo": b(surface=.75, gold=.25),                             # inside glaze, saucer glaze (dark end)
        "cream_hi": b(surface=.95, gold=.05),                             # inside glaze (light end), string highlight
        "clay_lo": b(surface=.65, gold=.35, k=.75),                       # bare oatmeal clay (dark end)
        "clay_hi": b(overlay=.8, gold=.2),                                # bare oatmeal clay (light end)
        "tea_lo": b(gold=1.0, k=.83),                                     # tea in the middle of the cup
        "tea_mid": b(gold=1.0, k=.71),
        "tea_hi": b(gold=.9, love=.1, k=.5),                              # tea at the wall (deeper)
        "string_lo": b(overlay=.75, gold=.25),                            # cotton ply shadow
        "paper_lo": b(gold=.6, overlay=.4, hue=20, chroma=1.5),           # pastel-gold paper
        "paper_hi": b(overlay=.6, gold=.4, hue=20, chroma=2),
        "eyelet": b(gold=.9, overlay=.1, hue=10, k=.9),                   # darker ring round the punched hole
        "leaf": b(pine=.85, foam=.15),                                    # the printed leaf
        "steam_lo": b(hl_med=.65, iris=.35),                              # thin steam veil, cool lilac grey
        "steam_hi": K.slot("surface"),                                    # dense steam core, warm white
    }


# ---------------------------------------------------------------- materials
def _mg_mat_glaze(K, C, base, saucer=False):
    """Hand-glazed ceramic in object space (metres).  Mug: rose outside / cream inside (vertex attribute `glaze_in` blends
    over the lip with a wobbly dipped line), pooled deeper rose near the base, thin pale band near the lip, throwing-ring
    tint, two sizes of plum-brown speckles, bare oatmeal clay foot below a wavy glaze line with a rolled edge.
    Saucer: cream glaze + sparse speckles + bare foot ring."""
    m, nt, out = K.new_mat(base)
    tc = N(nt, "ShaderNodeTexCoord", (-2500, 0))
    P = tc.outputs["Object"]
    X, Y, Z = _mg_sepxyz(nt, P)
    H = MUG_H

    rose, love, oat = C["rose"], C["love"], C["oat"]
    rose_deep = mix(rose, love, 0.45)
    rose_thin = mix(rose, oat, 0.38)
    spk_col = C["speck"]

    # --- glaze line (bare clay below) and its rolled edge
    ang = _mg_xyz(nt, _mg_math(nt, "MULTIPLY", X, 55.0), _mg_math(nt, "MULTIPLY", Y, 55.0), 0.37)
    nl = _mg_noise(nt, ang, 1.0, 1.0, 0.5)
    if saucer:
        zline = _mg_math(nt, "MULTIPLY_ADD", _mg_math(nt, "SUBTRACT", nl, 0.5), 0.0010, 0.00085)
    else:
        zline = _mg_math(nt, "MULTIPLY_ADD", _mg_math(nt, "SUBTRACT", nl, 0.5), 0.0036, 0.0049)
    clay = _mg_math(nt, "SUBTRACT", 1.0, _mg_sstep(nt, _mg_math(nt, "SUBTRACT", zline, 0.00015),
                                                   _mg_math(nt, "ADD", zline, 0.00015), Z))
    if not saucer:
        up1 = _mg_sstep(nt, zline, _mg_math(nt, "ADD", zline, 0.0004), Z)
        up2 = _mg_sstep(nt, _mg_math(nt, "ADD", zline, 0.0004), _mg_math(nt, "ADD", zline, 0.0016), Z)
        roll = _mg_math(nt, "MULTIPLY", up1, _mg_math(nt, "SUBTRACT", 1.0, up2))

    # --- inside / outside (mug only)
    if saucer:
        inside = 1.0
    else:
        gi = N(nt, "ShaderNodeAttribute", (-2500, -400), attribute_type="GEOMETRY", attribute_name="glaze_in")
        nlip = _mg_noise(nt, _mg_xyz(nt, _mg_math(nt, "MULTIPLY", X, 75.0), _mg_math(nt, "MULTIPLY", Y, 75.0), 0.1),
                         1.0, 1.0, 0.5)
        v = _mg_math(nt, "MULTIPLY_ADD", _mg_math(nt, "SUBTRACT", nlip, 0.5), 0.55, gi.outputs["Fac"])
        inside = _mg_sstep(nt, 0.40, 0.60, v)

    # --- rose glaze thickness g (0 thin .. 1 pooled): pooled near the base, thin near the lip, ring tint, broad noise
    if not saucer:
        t = _mg_math(nt, "MULTIPLY", Z, 1.0 / H)
        deep = _mg_math(nt, "SUBTRACT", 1.0, _mg_sstep(nt, 0.02, 0.42, t))
        thin = _mg_sstep(nt, 0.89, 0.995, t)
        ph = _mg_math(nt, "MULTIPLY_ADD", Z, _MG_RING_K, -0.012 * _MG_RING_K)
        ph = _mg_math(nt, "ADD", ph, _mg_math(nt, "MULTIPLY", _mg_math(nt, "SINE", _mg_math(nt, "MULTIPLY", Z, 83.0)), 1.1))
        ph = _mg_math(nt, "ADD", ph, _mg_math(nt, "MULTIPLY", _mg_math(nt, "SINE", _mg_math(nt, "MULTIPLY", Z, 37.0)), 0.5))
        ring = _mg_math(nt, "MULTIPLY_ADD", _mg_math(nt, "SINE", ph), 0.5, 0.5)      # 1 on a ridge, 0 in a groove
        nb = _mg_noise(nt, P, 22.0, 3.0, 0.55)
        g = _mg_math(nt, "MULTIPLY_ADD", deep, 0.40, 0.37)
        g = _mg_math(nt, "MULTIPLY_ADD", thin, -0.40, g)
        g = _mg_math(nt, "MULTIPLY_ADD", ring, -0.10, g)                               # thin on ridges, pooled in grooves
        g = _mg_math(nt, "MULTIPLY_ADD", nb, 0.40, g, clamp=True)
        rose_ramp = ramp(nt, (-1200, 400), [(0.0, rose_thin), (0.5, rose), (1.0, rose_deep)])
        L(nt, g, rose_ramp.inputs[0])

    # --- cream
    ncr = _mg_noise(nt, P, 14.0, 2.0, 0.5)
    cream_ramp = ramp(nt, (-1200, 100), [(0.30, C["cream_lo"]), (0.70, C["cream_hi"])])
    L(nt, ncr, cream_ramp.inputs[0])

    # --- speckles (two sizes; fewer on the rose outside)
    if saucer:
        g1, g2 = 0.55, 0.86
    else:
        g1 = _mg_mixf(nt, inside, 0.66, 0.40)
        g2 = _mg_mixf(nt, inside, 0.86, 0.66)
    s1 = _mg_speck(nt, P, 1250.0, g1, 0.10, 0.22)
    s2 = _mg_speck(nt, P, 380.0, g2, 0.09, 0.16)
    spk = _mg_math(nt, "MAXIMUM", s1, s2)

    # --- colour
    if saucer:
        glaze = cream_ramp.outputs[0]
    else:
        glaze = _mg_mixc(nt, inside, rose_ramp.outputs[0], cream_ramp.outputs[0])
    glaze = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", spk, 0.9), glaze, spk_col)
    if not saucer:
        glaze = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", roll, 0.55), glaze, rose_deep)
    # bare clay: oatmeal, grainy, denser iron specks
    ncl = _mg_noise(nt, P, 260.0, 4.0, 0.6)
    clay_ramp = ramp(nt, (-1200, -200), [(0.30, C["clay_lo"]), (0.70, C["clay_hi"])])
    L(nt, ncl, clay_ramp.inputs[0])
    s3 = _mg_speck(nt, P, 900.0, 0.45, 0.10, 0.24)
    clay_c = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", s3, 0.8), clay_ramp.outputs[0], spk_col)
    base_c = _mg_mixc(nt, clay, glaze, clay_c)

    # --- roughness / coat / specular
    if saucer:
        r_glaze = 0.30
        coat = _mg_math(nt, "MULTIPLY", 0.18, _mg_math(nt, "SUBTRACT", 1.0, clay))
    else:
        r_rose = _mg_math(nt, "MULTIPLY_ADD", g, -0.14, 0.30)
        r_glaze = _mg_mixf(nt, inside, r_rose, 0.30)
        coat = _mg_math(nt, "MULTIPLY", _mg_mixf(nt, inside, 0.32, 0.18), _mg_math(nt, "SUBTRACT", 1.0, clay))
    rough = _mg_mixf(nt, clay, _mg_math(nt, "MULTIPLY_ADD", spk, 0.12, r_glaze), 0.88)
    spec = _mg_mixf(nt, clay, 0.5, 0.25)

    # --- bump: speckles + fine grain (base layer only, the coat stays smooth like a real glaze)
    nf = _mg_noise(nt, P, 1100.0, 3.0, 0.6)
    hgt = _mg_math(nt, "MULTIPLY_ADD", spk, 0.60, _mg_math(nt, "MULTIPLY", nf, 0.30))
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = _mg_loc(nt)
    bump.inputs["Distance"].default_value = 0.0004
    _mg_set(nt, bump, 0, _mg_mixf(nt, clay, 0.14, 0.60))
    _mg_set(nt, bump, 2, hgt)

    b = principled(nt, out, (700, 0), **{"Metallic": 0.0, "Coat Roughness": 0.06, "Coat IOR": 1.5})
    L(nt, base_c, b.inputs["Base Color"])
    L(nt, rough, b.inputs["Roughness"])
    L(nt, coat, b.inputs["Coat Weight"])
    L(nt, spec, b.inputs["Specular IOR Level"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _mg_mat_tea(K, C):
    """Amber tea: glossy surface, colour deeper towards the wall / lighter in the middle (depth cue), faint subsurface."""
    m, nt, out = K.new_mat("tea")
    tc = N(nt, "ShaderNodeTexCoord", (-1600, 0))
    X, Y, Z = _mg_sepxyz(nt, tc.outputs["Object"])
    rr = _mg_math(nt, "POWER", _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", X, X), _mg_math(nt, "MULTIPLY", Y, Y)), 0.5)
    f = _mg_sstep(nt, 0.0, 0.0375, rr)
    ra = ramp(nt, (-1000, 100), [(0.0, C["tea_lo"]), (0.55, C["tea_mid"]), (1.0, C["tea_hi"])])
    L(nt, f, ra.inputs[0])
    nw = _mg_noise(nt, _mg_xyz(nt, _mg_math(nt, "MULTIPLY", X, 90.0), _mg_math(nt, "MULTIPLY", Y, 90.0), 0.2), 1.0, 2.0, 0.5)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-600, -300)
    bump.inputs["Strength"].default_value = 0.05
    bump.inputs["Distance"].default_value = 0.0002
    L(nt, nw, bump.inputs["Height"])
    b = principled(nt, out, (700, 0), **{"Roughness": 0.04, "IOR": 1.33, "Specular IOR Level": 0.5,
                                         "Subsurface Weight": 0.30, "Subsurface Radius": (1.0, 0.45, 0.12),
                                         "Subsurface Scale": 0.01})
    L(nt, ra.outputs[0], b.inputs["Base Color"])
    L(nt, ra.outputs[0], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 0.06
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _mg_mat_string(K, C):
    """Two-ply cotton string: diagonal twist stripes along the tube UV (u = metres along, v = metres around)."""
    m, nt, out = K.new_mat("string")
    tc = N(nt, "ShaderNodeTexCoord", (-1500, 0))
    u, v, _z = _mg_sepxyz(nt, tc.outputs["UV"])
    ph = _mg_math(nt, "MULTIPLY", _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", u, 770.0), _mg_math(nt, "MULTIPLY", v, 400.0)),
                  2.0 * math.pi)
    ply = _mg_math(nt, "MULTIPLY_ADD", _mg_math(nt, "SINE", ph), 0.5, 0.5)
    col = _mg_mixc(nt, ply, C["string_lo"], C["cream_hi"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-500, -300)
    bump.inputs["Strength"].default_value = 0.7
    bump.inputs["Distance"].default_value = 0.00015
    L(nt, ply, bump.inputs["Height"])
    b = principled(nt, out, (700, 0), **{"Roughness": 0.92, "Specular IOR Level": 0.25, "Sheen Weight": 0.5,
                                         "Sheen Roughness": 0.6})
    L(nt, col, b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _mg_mat_tag(K, C):
    """Pastel-gold paper tag (UV 0..1 over 14 x 22 mm): fibre noise, a pine leaf with a pale midrib, two rose text bars,
    a punched hole (alpha cut-out) with a darker eyelet ring."""
    m, nt, out = K.new_mat("tag")
    tc = N(nt, "ShaderNodeTexCoord", (-2600, 0))
    u, v, _z = _mg_sepxyz(nt, tc.outputs["UV"])
    x = _mg_math(nt, "MULTIPLY", _mg_math(nt, "SUBTRACT", u, 0.5), _MG_TAG_W)
    y = _mg_math(nt, "MULTIPLY", _mg_math(nt, "SUBTRACT", v, 0.5), _MG_TAG_H)

    def soft_lt(val, edge, w=0.00006):          # 1 inside (val < edge), soft edge
        return _mg_math(nt, "SUBTRACT", 1.0, _mg_sstep(nt, edge - w, edge + w, val))

    # leaf: intersection of two discs, rotated 35 deg, centred slightly below the middle
    ca, sa = math.cos(math.radians(35.0)), math.sin(math.radians(35.0))
    yl = _mg_math(nt, "ADD", y, 0.0016)
    xr = _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", x, ca), _mg_math(nt, "MULTIPLY", yl, sa))
    yr = _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", x, -sa), _mg_math(nt, "MULTIPLY", yl, ca))
    hl, hw = 0.0042, 0.0020
    Rr = (hl * hl + hw * hw) / (2.0 * hw)
    cc = Rr - hw
    xx = _mg_math(nt, "MULTIPLY", xr, xr)
    d1 = _mg_math(nt, "POWER", _mg_math(nt, "ADD", xx, _mg_math(nt, "POWER", _mg_math(nt, "SUBTRACT", yr, cc), 2.0)), 0.5)
    d2 = _mg_math(nt, "POWER", _mg_math(nt, "ADD", xx, _mg_math(nt, "POWER", _mg_math(nt, "ADD", yr, cc), 2.0)), 0.5)
    leaf = _mg_math(nt, "MULTIPLY", soft_lt(d1, Rr), soft_lt(d2, Rr))
    rib = _mg_math(nt, "MULTIPLY", soft_lt(_mg_math(nt, "ABSOLUTE", yr), 0.00011, 0.00003),
                   soft_lt(_mg_math(nt, "ABSOLUTE", _mg_math(nt, "ADD", xr, 0.0004)), 0.0034, 0.0003))
    leaf = _mg_math(nt, "MULTIPLY", leaf, _mg_math(nt, "SUBTRACT", 1.0, _mg_math(nt, "MULTIPLY", rib, 0.9)))

    # text bars
    def bar(cx_half, cy, hh):
        return _mg_math(nt, "MULTIPLY", soft_lt(_mg_math(nt, "ABSOLUTE", x), cx_half, 0.0002),
                        soft_lt(_mg_math(nt, "ABSOLUTE", _mg_math(nt, "SUBTRACT", y, cy)), hh, 0.00008))
    bars = _mg_math(nt, "MAXIMUM", bar(0.0042, -0.0074, 0.00038), bar(0.0027, -0.0088, 0.00038))
    # paper
    nfib = _mg_noise(nt, tc.outputs["Object"], 900.0, 4.0, 0.6)
    paper_ramp = ramp(nt, (-1500, 300), [(0.30, C["paper_lo"]), (0.70, C["paper_hi"])])
    L(nt, nfib, paper_ramp.inputs[0])
    # punched hole + eyelet ring
    hy = _MG_TAG_HY
    dh = _mg_math(nt, "POWER", _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", x, x),
                                        _mg_math(nt, "POWER", _mg_math(nt, "SUBTRACT", y, hy), 2.0)), 0.5)
    alpha = _mg_sstep(nt, 0.00078, 0.00090, dh)
    ringm = _mg_math(nt, "SUBTRACT", 1.0, _mg_sstep(nt, 0.0, 0.00030, _mg_math(nt, "ABSOLUTE", _mg_math(nt, "SUBTRACT", dh, 0.00115))))
    col = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", ringm, 0.55), paper_ramp.outputs[0], C["eyelet"])
    col = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", bars, 0.85), col, C["love"])
    col = _mg_mixc(nt, _mg_math(nt, "MULTIPLY", leaf, 0.92), col, C["leaf"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-500, -300)
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0002
    L(nt, nfib, bump.inputs["Height"])
    b = principled(nt, out, (700, 0), **{"Roughness": 0.74, "Specular IOR Level": 0.3, "Sheen Weight": 0.25,
                                         "Sheen Roughness": 0.5})
    L(nt, col, b.inputs["Base Color"])
    L(nt, alpha, b.inputs["Alpha"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    m.surface_render_method = "DITHERED"
    return m


def _mg_mat_steam(K, C):
    """Wispy steam: alpha = steam * across-bell * height fade * scrolling stretched noise (time = frame / fps, seconds).
    The strength is the root's `steam` property."""
    m, nt, out = K.new_mat("steam")
    steam = _mg_math(nt, "MAXIMUM", _mg_math(nt, "MINIMUM", K.param(nt, "steam", loc=(-2600, 600)), 1.0), 0.0)
    tc = N(nt, "ShaderNodeTexCoord", (-2600, 0))
    u, v, _z = _mg_sepxyz(nt, tc.outputs["UV"])
    seed = N(nt, "ShaderNodeAttribute", (-2600, -400), attribute_type="GEOMETRY", attribute_name="seed").outputs["Fac"]
    # across bell (0 at the ribbon edges) and height fade (in over the first 10 %, out from 30 %)
    edge = _mg_math(nt, "ABSOLUTE", _mg_math(nt, "MULTIPLY_ADD", u, 2.0, -1.0))
    bell = _mg_math(nt, "POWER", _mg_sstep(nt, 0.0, 1.0, _mg_math(nt, "SUBTRACT", 1.0, edge)), 1.3)
    fade = _mg_math(nt, "MULTIPLY", _mg_sstep(nt, 0.0, 0.10, v),
                    _mg_math(nt, "SUBTRACT", 1.0, _mg_sstep(nt, 0.30, 1.0, v)))
    # time in seconds (driven by the frame counter)
    tv = K.clock(nt, loc=(-2600, -700))

    # fibrous layer (stretched along z, rising 7 cm/s) and a slower, larger billow layer
    def scrolled(kxy, kz, speed, so):
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.location = _mg_loc(nt)
        mp.inputs["Scale"].default_value = (kxy, kxy, kz)
        loc = _mg_xyz(nt, _mg_math(nt, "MULTIPLY", seed, 31.7 * so), _mg_math(nt, "MULTIPLY", seed, 17.3 * so),
                      _mg_math(nt, "MULTIPLY", tv, -speed))
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(loc, mp.inputs["Location"])
        return mp.outputs[0]
    n1 = _mg_noise(nt, scrolled(85.0, 22.0, 1.55, 1.0), 1.0, 4.0, 0.55)
    n2 = _mg_noise(nt, scrolled(24.0, 13.0, 0.95, 0.7), 1.0, 3.0, 0.5)
    mx = _mg_math(nt, "ADD", _mg_math(nt, "MULTIPLY", n1, 0.62), _mg_math(nt, "MULTIPLY", n2, 0.38))
    thr = _mg_math(nt, "MULTIPLY_ADD", steam, -0.12, 0.50)
    mask = _mg_sstep(nt, thr, _mg_math(nt, "ADD", thr, 0.26), mx)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = _mg_loc(nt)
    dt = nt.nodes.new("ShaderNodeVectorMath")
    dt.location = _mg_loc(nt)
    dt.operation = "DOT_PRODUCT"
    nt.links.new(geo.outputs["Normal"], dt.inputs[0])
    nt.links.new(geo.outputs["Incoming"], dt.inputs[1])
    facing = _mg_sstep(nt, 0.10, 0.55, _mg_math(nt, "ABSOLUTE", dt.outputs["Value"]))     # edge-on sheets vanish
    alpha = _mg_math(nt, "MULTIPLY", _mg_math(nt, "MULTIPLY", steam, 0.85),
                     _mg_math(nt, "MULTIPLY", _mg_math(nt, "MULTIPLY", bell, fade), _mg_math(nt, "MULTIPLY", mask, facing)))
    cs = _mg_math(nt, "MULTIPLY", bell, mask)
    # dense cores warm white, thinner veils a cool lilac grey so the plume still reads on a pale wall
    col = _mg_mixc(nt, _mg_sstep(nt, 0.25, 1.0, cs), C["steam_lo"], C["steam_hi"])
    b = principled(nt, out, (700, 0), **{"Roughness": 1.0, "Specular IOR Level": 0.0})
    L(nt, col, b.inputs["Base Color"])
    L(nt, alpha, b.inputs["Alpha"])
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    m.show_transparent_back = True
    return m


# ---------------------------------------------------------------- saucer
def _mg_saucer(K, mat_glaze):
    prof = [(0.0, 0.0030), (0.0300, 0.0030), (0.0368, 0.0029), (0.0382, 0.0018), (0.0390, 0.0006), (0.0402, 0.0),
            (0.0436, 0.0), (0.0450, 0.0007), (0.0458, 0.0022), (0.0468, 0.0036)]
    # underside rising to the rim
    under = _mg_prof_smooth([(0.0468, 0.0036), (0.0520, 0.0056), (0.0620, 0.0088), (0.0710, 0.0120), (0.0760, 0.0136),
                             (0.0779, 0.0146)], 4)
    prof += under[1:]
    # rounded lip (radius 1.4 mm), then down the inside
    cr, cz, lr = 0.0766, 0.0146, 0.0014
    for k in range(1, 10):
        a = math.pi * k / 9.0
        prof.append((cr + lr * math.cos(a), cz + lr * math.sin(a)))
    inner = _mg_prof_smooth([(cr - lr, cz), (0.0725, 0.0132), (0.0660, 0.0112), (0.0585, 0.0090), (0.0515, 0.0073),
                             (0.0462, 0.0064), (0.0435, 0.0060), (0.0400, 0.0058)], 4)
    prof += inner[1:]
    prof += [(0.0200, 0.0058), (0.0, 0.0058)]
    bm = bmesh.new()
    rings = bm_lathe(bm, prof, segs=_MG_SEGS, mat=0)
    for ring, (r, z) in zip(rings, prof):          # hand-thrown: slightly out of round, rim a little uneven (well stays flat)
        w = (r / MUG_SAUCER_R) ** 3
        for j, v in enumerate(ring):
            if len(ring) > 1:
                th = 2 * math.pi * j / _MG_SEGS
                rr = r * (1.0 + w * (0.0022 * math.cos(2 * th + 0.4) + 0.0012 * math.cos(3 * th + 1.7)))
                v.co = Vector((rr * math.cos(th), rr * math.sin(th),
                               z + w * (0.00025 * math.sin(th + 1.1) + 0.00012 * math.sin(2 * th + 2.3))))
    return K.to_obj("body", bm, [mat_glaze])


# ---------------------------------------------------------------- mug body (lathe + handle)
def _mg_mug_profile():
    """[(r, z, glaze_in)] bottom centre -> foot ring -> outside -> lip -> inside -> floor centre.
    glaze_in: 0 = outside glaze, 1 = inside glaze (blended over the lip)."""
    P = []
    foot = [(0.0, 0.0034), (0.0200, 0.0034), (0.0262, 0.0032), (0.0278, 0.0022), (0.0287, 0.0009), (0.0297, 0.0001),
            (0.0313, 0.0), (0.0330, 0.0002), (0.0339, 0.0011), (0.0343, 0.0026)]
    P += [(r, z, 0.0) for r, z in foot]
    r6 = _mg_r_out(_MG_ZW0)
    sh = _mg_prof_smooth([(0.0343, 0.0026), (0.0351, 0.0043), (0.0357, 0.0056), (r6, _MG_ZW0),
                          (_mg_r_out(_MG_ZW0 + 0.003), _MG_ZW0 + 0.003)], 4)
    P += [(r, z, 0.0) for r, z in sh[1:]]
    n = 76
    for i in range(1, n + 1):
        z = _MG_ZW0 + 0.003 + (_MG_ZW1 - _MG_ZW0 - 0.003) * i / n
        P.append((_mg_r_out(z), z, 0.0))
    # lip
    cr, cz = _mg_r_out(_MG_ZW1) - _MG_LIPR, _MG_ZW1
    for k in range(1, 11):
        a = math.pi * k / 10.0
        P.append((cr + _MG_LIPR * math.cos(a), cz + _MG_LIPR * math.sin(a), k / 10.0))
    # inside wall down to the floor fillet
    zf = _MG_FLOOR + 0.006
    n = 64
    for i in range(1, n + 1):
        z = _MG_ZW1 - (_MG_ZW1 - zf) * i / n
        P.append((_mg_r_in(z), z, 1.0))
    rf = _mg_r_in(zf)
    for k in range(1, 9):
        a = 0.5 * math.pi * k / 8.0
        P.append((rf - 0.006 + 0.006 * math.cos(a), _MG_FLOOR + 0.006 - 0.006 * math.sin(a), 1.0))
    P.append((0.0100, _MG_FLOOR, 1.0))
    P.append((0.0, _MG_FLOOR, 1.0))
    return P


def _mg_wall_at(y, z, az):
    """Outer wall point at handle-plane coordinates: tangential offset y from azimuth az, height z."""
    th = az
    for _ in range(4):
        r = _mg_r_out(z) * _mg_wob(th, z)
        th = az + math.asin(max(-0.95, min(0.95, y / r)))
    r = _mg_r_out(z) * _mg_wob(th, z)
    return Vector((r * math.cos(th), r * math.sin(th), z))


def _mg_handle_path(az):
    """Centre line of the D handle in its plane at azimuth `az`: er / et (radial and tangential unit vectors), z_t / z_b
    and rho_t / rho_b (height and wall radius of the two attachments), path (Vectors (rho, z, 0)), s (arc length at each
    path point) and L_ (total arc length)."""
    er = Vector((math.cos(az), math.sin(az), 0.0))
    et = Vector((-math.sin(az), math.cos(az), 0.0))
    f = _MG_HANDLE_F
    z_t, z_b = _MG_HANDLE_ZT, _MG_HANDLE_ZB
    rho_t = _mg_wall_at(0.0, z_t, az).dot(er)
    rho_b = _mg_wall_at(0.0, z_b, az).dot(er)
    keys = [(rho_t + f, z_t), (rho_t + f + 0.0065, z_t + 0.0003), (0.0600, 0.0812), (0.0670, 0.0787), (0.0724, 0.0716),
            (0.0745, 0.0620), (0.0733, 0.0485), (0.0678, 0.0360), (0.0594, 0.0282), (rho_b + f + 0.0095, z_b - 0.0003),
            (rho_b + f, z_b)]
    path = catmull([(r, z, 0.0) for r, z in keys], 12)
    s = [0.0]
    for i in range(1, len(path)):
        s.append(s[-1] + (path[i] - path[i - 1]).length)
    return SimpleNamespace(er=er, et=et, z_t=z_t, z_b=z_b, rho_t=rho_t, rho_b=rho_b, path=path, s=s, L_=s[-1])


def _mg_handle(bm, gl, az):
    """D-shaped hand-pulled handle: a flat-oval strap swept along a planar path (rho, z) at azimuth `az`, thumb-rest flat
    at the top, a little asymmetric.  Both ends grow out of the wall through a concave fillet (quarter circle, radius f) that
    is exactly tangent to the glaze, so the join reads as pressed-on clay, not as a pipe stuck into a cylinder.
    Returns the vertex rings."""
    H = _mg_handle_path(az)
    er, et, rho_t, rho_b, path, s, L_ = H.er, H.et, H.rho_t, H.rho_b, H.path, H.s, H.L_
    f, D = _MG_HANDLE_F, _MG_HANDLE_D      # fillet radius / distance over which the wall curvature is blended out
    z_t, z_b = H.z_t, H.z_b
    n = len(path)
    ras, rbs = _MG_HANDLE_RA, _MG_HANDLE_RB
    sides = 20
    tube = []
    for i in range(n):
        t = s[i] / L_
        j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
        tg = path[j1] - path[j0]
        T = (er * tg.x + Vector((0.0, 0.0, 1.0)) * tg.y).normalized()
        B = T.cross(et)
        ra, rb = _mg_key(ras, t), _mg_key(rbs, t)
        c = er * path[i].x + Vector((0.0, 0.0, 1.0)) * path[i].y
        pts = [c + et * (math.cos(2 * math.pi * k / sides) * ra) + B * (math.sin(2 * math.pi * k / sides) * rb)
               for k in range(sides)]
        # near the wall the ring follows the (curved) wall: shift along er by the sag, fading out over D
        top = s[i] < L_ - s[i]
        dec = 1.0 - _mg_ss(0.0, D, s[i] if top else L_ - s[i])
        if dec > 0.0:
            ref = rho_t if top else rho_b
            pts = [p - er * (dec * (ref - _mg_wall_at(p.dot(et), p.z, az).dot(er))) for p in pts]
        tube.append(pts)

    def fillet_rows(base, ra, rb, zc, phis):
        rows = []
        for phi in phis:
            off, h = f * (1.0 - math.sin(phi)), f * (1.0 - math.cos(phi)) - 0.00005
            row = []
            for p in base:
                yk, zk = p.dot(et), p.z
                ny, nz = yk / (ra * ra), (zk - zc) / (rb * rb)
                nl = math.hypot(ny, nz)
                row.append(_mg_wall_at(yk + off * ny / nl, zk + off * nz / nl, az) + er * h)
            rows.append(row)
        return rows

    phis = [math.radians(a) for a in (0.0, 2.0, 5.0, 10.0, 17.0, 26.0, 37.0, 50.0, 65.0, 78.0)]
    rows = fillet_rows(tube[0], ras[0][1], rbs[0][1], z_t, phis) + tube \
        + fillet_rows(tube[-1], ras[-1][1], rbs[-1][1], z_b, phis[::-1])
    rings = []
    for row in rows:
        ring = []
        for p in row:
            v = bm.verts.new(p)
            v[gl] = 0.0
            ring.append(v)
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            try:
                fc = bm.faces.new((rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]))
            except ValueError:
                continue
            fc.smooth = True
    return rings


def _mg_mug(K, mat):
    """The mug object and the handle's vertex positions (for the card)."""
    bm = bmesh.new()
    gl = bm.verts.layers.float.new("glaze_in")
    P = _mg_mug_profile()
    rings = bm_lathe(bm, [(r, z) for r, z, g in P], segs=_MG_SEGS, mat=0)
    for ring, (r, z, g) in zip(rings, P):
        for j, v in enumerate(ring):
            v[gl] = g
            if len(ring) > 1:
                th = 2 * math.pi * j / _MG_SEGS
                w = _mg_wob(th, z)
                v.co = Vector((r * w * math.cos(th), r * w * math.sin(th), z + _mg_dz(th, z)))
    hrings = _mg_handle(bm, gl, math.radians(MUG_HANDLE_DEG))
    handle_pts = [v.co.copy() for ring in hrings for v in ring]
    return K.to_obj("body", bm, [mat]), handle_pts


# ---------------------------------------------------------------- tea
def _mg_tea(K, mat):
    zt = MUG_TEA_Z
    segs = _MG_SEGS
    wm, hm = 0.0017, 0.0010                         # meniscus width / rise
    fr = [0.0, 0.16, 0.32, 0.48, 0.63, 0.76, 0.87, 0.95, 1.0]
    men = [0.30, 0.58, 0.82, 1.0]
    bm = bmesh.new()
    rows = [[bm.verts.new((0.0, 0.0, zt))]]
    for f in fr[1:]:
        ring = []
        for j in range(segs):
            th = 2 * math.pi * j / segs
            rf = _mg_r_in(zt) * _mg_wob(th, zt) - wm
            ring.append(bm.verts.new((rf * f * math.cos(th), rf * f * math.sin(th), zt)))
        rows.append(ring)
    for e in men:
        ring = []
        for j in range(segs):
            th = 2 * math.pi * j / segs
            z = zt + hm * e ** 2.2
            rf = _mg_r_in(zt) * _mg_wob(th, zt) - wm
            r = rf + wm * e + (0.00018 if e >= 1.0 else 0.0)
            ring.append(bm.verts.new((r * math.cos(th), r * math.sin(th), z)))
        rows.append(ring)
    uvl = bm.loops.layers.uv.verify()
    for i in range(len(rows) - 1):
        A, B = rows[i], rows[i + 1]
        for j in range(segs):
            j2 = (j + 1) % segs
            vv = (A[0], B[j], B[j2]) if len(A) == 1 else (A[j], B[j], B[j2], A[j2])
            f = bm.faces.new(vv)
            f.smooth = True
            for lp in f.loops:
                lp[uvl].uv = (lp.vert.co.x * 10.0 + 0.5, lp.vert.co.y * 10.0 + 0.5)
    return K.to_obj("tea", bm, [mat])


# ---------------------------------------------------------------- tea tag (paper sheet resting on the wall) + string
def _mg_tag_pt(x, y):
    """Tag-local (x right, y up; origin = sheet centre, seen from outside) -> (mid-surface point in the mug frame,
    unit normal pointing away from the wall).  The sheet is wrapped on the wall with a stand-off that lifts the top
    (the string runs behind it) and curls the bottom edge, so it never clips the glaze."""
    az = math.radians(MUG_TAG_DEG)
    dy = y - _MG_TAG_HY
    c, s = math.cos(_MG_TAG_TILT), math.sin(_MG_TAG_TILT)
    z = _MG_TAG_ZH + s * x + c * dy
    th = az + (c * x - s * dy) / _mg_r_out(_MG_TAG_ZH)
    w = (y + _MG_TAG_H / 2) / _MG_TAG_H
    u = x / (_MG_TAG_W / 2)
    off = 0.00035 + 0.0009 * _mg_ss(0.3, 1.0, w) + 0.0014 * (1.0 - w) ** 3 * (1.0 + 0.5 * u) + 0.0005 * u * u
    n = _mg_wall_n(z, th)
    return _mg_wall_pt(z, th) + n * off, n


def _mg_rrect(a, b, hw, hh, rc):
    """Unit-square grid coordinates (a, b in [-1, 1]) -> point of a hw x hh half-size rounded rectangle (corner radius rc)
    by radial remapping, so a regular grid keeps its topology."""
    px, py = a * hw, b * hh
    m = max(abs(a), abs(b))
    if m < 1e-9:
        return 0.0, 0.0
    qx, qy = px / m, py / m
    if abs(qx) > hw - rc and abs(qy) > hh - rc:
        cx = (hw - rc) * (1.0 if qx > 0 else -1.0)
        cy = (hh - rc) * (1.0 if qy > 0 else -1.0)
        ql = math.hypot(qx, qy)
        ux, uy = qx / ql, qy / ql
        dc = ux * cx + uy * cy
        t = dc + math.sqrt(max(dc * dc - (cx * cx + cy * cy - rc * rc), 0.0))
        k = t / ql
        return px * k, py * k
    return px, py


def _mg_tag(K, mat):
    nx, ny = 16, 24
    hw, hh = _MG_TAG_W / 2, _MG_TAG_H / 2
    S = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = _mg_rrect(-1.0 + 2.0 * i / nx, -1.0 + 2.0 * j / ny, hw, hh, 0.0022)
            S[i, j] = (_mg_tag_pt(x, y)[0], x, y)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    F, B = {}, {}
    for (i, j), (p, x, y) in S.items():
        ip, im, jp, jm = min(i + 1, nx), max(i - 1, 0), min(j + 1, ny), max(j - 1, 0)
        n = (S[ip, j][0] - S[im, j][0]).cross(S[i, jp][0] - S[i, jm][0]).normalized()
        F[i, j] = bm.verts.new(p + n * (_MG_TAG_T / 2))
        B[i, j] = bm.verts.new(p - n * (_MG_TAG_T / 2))

    def uv_of(v, i, j):
        return ((S[i, j][1] + hw) / _MG_TAG_W, (S[i, j][2] + hh) / _MG_TAG_H)

    def quad(vs, ij, smooth):
        f = bm.faces.new(vs)
        f.smooth = smooth
        for lp, (i, j) in zip(f.loops, ij):
            lp[uvl].uv = uv_of(lp.vert, i, j)

    for j in range(ny):
        for i in range(nx):
            c4 = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            quad([F[c] for c in c4], c4, True)
            quad([B[c] for c in c4[::-1]], c4[::-1], True)
    # edge strip around the outline, walking it counter-clockwise (seen from outside)
    loop = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)] + [(i, ny) for i in range(nx, 0, -1)] \
        + [(0, j) for j in range(ny, 0, -1)]
    for k, a in enumerate(loop):
        b = loop[(k + 1) % len(loop)]
        quad([B[a], B[b], F[b], F[a]], [a, b, b, a], False)
    return K.to_obj("tag", bm, [mat])


def _mg_string(K, mat):
    """Cotton string: out of the tea, up the inside wall, over the lip, down the outside, behind the top of the tag, out
    through its punched hole and a short tail across the front of the tag."""
    th = math.radians(MUG_TAG_DEG)
    rs = _MG_STRING_R
    g = rs + 0.00006
    pts = []
    z = MUG_TEA_Z - 0.006
    while z < _MG_ZW1 - 0.002:
        pts.append(_mg_in_pt(z, th, g))
        z += 0.004
    rc = (_mg_r_out(_MG_ZW1) - _MG_LIPR) * _mg_wob(th, _MG_ZW1)
    R = _MG_LIPR + g
    for k in range(13):
        phi = math.pi * (1.0 - k / 12.0)
        r_, z_ = rc + R * math.cos(phi), _MG_ZW1 + R * math.sin(phi)
        pts.append(Vector((r_ * math.cos(th), r_ * math.sin(th), z_ + _mg_dz(th, z_))))
    z = _MG_ZW1 - 0.003
    while z > _MG_TAG_ZH + 0.0032:
        pts.append(_mg_wall_pt(z, th, g))
        z -= 0.004
    pts.append(_mg_wall_pt(_MG_TAG_ZH + 0.0032, th, g))
    hy, half = _MG_TAG_HY, _MG_TAG_T / 2 + rs + 0.00004
    for x_, y_, side in ((0.0, hy + 0.0016, -1.0), (0.0, hy + 0.0007, -0.6), (0.0, hy, 0.0), (0.0, hy - 0.0007, 0.7),
                         (0.0002, hy - 0.0022, 1.0), (0.0006, hy - 0.0040, 1.0), (0.0011, hy - 0.0057, 1.0)):
        p, n = _mg_tag_pt(x_, y_)
        pts.append(p + n * (half * side))
    path = catmull(pts, 3)
    bm = bmesh.new()
    bm_tube(bm, path, rs, sides=6, cap="round")
    return K.to_obj("string", bm, [mat])


# ---------------------------------------------------------------- steam
def _mg_steam(K, mat):
    """3-4 S-curved twisting ribbons rising from the tea (widening upward) + two horizontal veils so the plume also reads
    from above.  UV: u across 0..1, v along 0..1 (veils: u = 0.5 + 0.5 * radius); float vertex attribute `seed` per piece."""
    bm = bmesh.new()
    sl = bm.verts.layers.float.new("seed")
    uvl = bm.loops.layers.uv.verify()
    z0 = MUG_TEA_Z + 0.002
    ribbons = [  # x0, y0, azimuth0 (deg), twist (deg), height, amplitude, phase, drift (dx, dy), seed
        (-0.006, 0.004, 0.0, 85.0, 0.185, 0.014, 0.00, (0.010, 0.004), 0.13),
        (0.007, -0.003, 60.0, -70.0, 0.165, 0.017, 1.70, (-0.012, 0.006), 0.47),
        (0.000, 0.008, 120.0, 60.0, 0.150, 0.012, 3.10, (0.004, -0.010), 0.81),
        (-0.004, -0.008, 30.0, -50.0, 0.130, 0.010, 4.60, (0.008, 0.010), 0.29),
        (0.010, 0.002, 95.0, 40.0, 0.175, 0.015, 2.40, (-0.006, -0.008), 0.66),
        (-0.010, -0.002, 150.0, -80.0, 0.140, 0.013, 5.30, (0.009, 0.001), 0.92),
    ]
    n = 30
    for x0, y0, az0, tw, hs, amp, ph, (dx, dy), seed in ribbons:
        pts = []
        for i in range(n):
            t = i / (n - 1.0)
            pts.append(Vector((x0 + amp * math.sin(2 * math.pi * 1.15 * t + ph) * t ** 0.7 + dx * t * t,
                               y0 + amp * math.cos(2 * math.pi * 0.9 * t + ph * 0.7) * t ** 0.7 + dy * t * t,
                               z0 + hs * t)))
        prev = None
        for i, p in enumerate(pts):
            t = i / (n - 1.0)
            T = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
            a = math.radians(az0 + tw * t)
            A = Vector((math.cos(a), math.sin(a), 0.0))
            A = (A - T * A.dot(T)).normalized()
            w = 0.010 + 0.055 * t ** 0.75
            vl, vr = bm.verts.new(p - A * (w / 2)), bm.verts.new(p + A * (w / 2))
            vl[sl] = vr[sl] = seed
            if prev is not None:
                f = bm.faces.new((prev[0], prev[1], vr, vl))
                t0 = (i - 1) / (n - 1.0)
                for lp, uv in zip(f.loops, ((0.0, t0), (1.0, t0), (1.0, t), (0.0, t))):
                    lp[uvl].uv = uv
            prev = (vl, vr)
    for zc, rad, seed, tilt in ((z0 + 0.050, 0.030, 0.62, 0.10), (z0 + 0.100, 0.036, 0.93, -0.12)):
        segs, rings = 20, (0.0, 0.35, 0.65, 0.88, 1.0)
        rows = []
        for k, rho in enumerate(rings):
            row = []
            for j in range(1 if rho == 0 else segs):
                a = 2 * math.pi * j / segs
                x, y = rad * rho * math.cos(a), rad * rho * math.sin(a)
                v_ = bm.verts.new((x, y, zc + tilt * x + 0.004 * (1.0 - rho * rho)))
                v_[sl] = seed
                row.append(v_)
            rows.append(row)
        for k in range(len(rings) - 1):
            A, B = rows[k], rows[k + 1]
            for j in range(segs):
                j2 = (j + 1) % segs
                vv = (A[0], B[j], B[j2]) if len(A) == 1 else (A[j], B[j], B[j2], A[j2])
                f = bm.faces.new(vv)
                for lp in f.loops:
                    rr = math.hypot(lp.vert.co.x, lp.vert.co.y) / rad
                    lp[uvl].uv = (0.5 + 0.5 * rr, 0.5)
    o = K.to_obj("steam", bm, [mat])
    o.visible_shadow = False
    # gentle sway: two blend shapes (lateral drift growing with height, a different direction per ribbon), their
    # weights driven by sin(time) at slightly different rates so the plume wanders instead of standing like a pipe
    me = o.data
    o.shape_key_add(name="Basis")
    seeds = me.attributes["seed"].data
    for sname, a0 in (("swayA", 0.6), ("swayB", 2.7)):
        k = o.shape_key_add(name=sname)
        for i, v in enumerate(me.vertices):
            h = max(v.co.z - z0, 0.0) / 0.18
            a = a0 + seeds[i].value * 6.0
            k.data[i].co = Vector((v.co.x + 0.024 * h ** 1.4 * math.cos(a), v.co.y + 0.024 * h ** 1.4 * math.sin(a), v.co.z))
    # rates in rad per second (2.07 and 3.21 rad/s = 0.069 and 0.107 rad per frame at 30 fps)
    drive(me.shape_keys, 'key_blocks["swayA"].value', f"0.5 + 0.5 * sin(frame / {K.fps:g} * 2.07)")
    drive(me.shape_keys, 'key_blocks["swayB"].value', f"0.5 + 0.5 * sin(frame / {K.fps:g} * 3.21 + 1.9)")
    return o


# ---------------------------------------------------------------- use points (the same maths as the geometry)
def _mg_cylinder_bm(radius, depth, segments=32):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth)
    return bm


def _mg_handle_use(handle_pts):
    """Pinch grip on the handle's vertical run and the handle's outermost point. The strap is a flat oval: wide along the
    tangent (et), thinner in the handle plane. The pads close across its width (squeeze axis = et), fingertips pointing
    into the opening towards the wall (x = -er), the strap running up the grip's y axis. -> (grip entry, outermost point)"""
    az = math.radians(MUG_HANDLE_DEG)
    H = _mg_handle_path(az)
    er, et = H.er, H.et
    i = max(range(len(H.path)), key=lambda k: H.path[k].x)          # outermost centre-line point: the strap is vertical there
    rho, z = H.path[i].x, H.path[i].y
    t = H.s[i] / H.L_
    ra, rb = _mg_key(_MG_HANDLE_RA, t), _mg_key(_MG_HANDLE_RB, t)
    wall = _mg_wall_at(0.0, z, az).dot(er)
    straight = 0.0                                                    # length of the run within 25 deg of vertical
    for lo, hi in zip(H.path, H.path[1:]):
        d = hi - lo
        if abs(d.y) >= math.cos(math.radians(25.0)) * d.length:
            straight += d.length
    center = er * rho + Vector((0.0, 0.0, z))
    grip = {"name": "handle", "type": "pinch", "center": [round(v, 5) for v in center],
            "axis": [0.0, 0.0, 1.0], "normal": [round(v, 5) for v in et], "x": [round(-v, 5) for v in er],
            "width": round(2 * ra, 5), "length": round(straight, 5), "depth": round(2 * rb, 5),
            "span": round(rho - rb - wall, 5)}
    outer = max(handle_pts, key=lambda p: p.dot(er))
    return grip, outer


@register("cafe_saucer")
def cafe_saucer(name, coll, root, slots=None):
    """The cream saucer, 156 mm across and 16 mm tall: out of round by a fraction of a millimetre, rim a little uneven,
    well flat. Origin = centre of the underside, on the table top; +Z up. The well floor is MUG_SEAT_Z = 5.8 mm above the
    origin: that is where a `cafe_mug` root goes. No custom properties."""
    K = Kit(name, coll, root, slots)
    C = _mg_colors(K)
    _mg_saucer(K, _mg_mat_glaze(K, C, "glaze", saucer=True))
    col = K.collider(K.to_obj("col", _mg_cylinder_bm(MUG_SAUCER_R, MUG_SAUCER_H), loc=(0.0, 0.0, MUG_SAUCER_H / 2)))
    use = {
        "rest": [{"name": "well_floor", "type": "plane", "center": [0.0, 0.0, MUG_SEAT_Z], "normal": [0.0, 0.0, 1.0],
                  "radius": 0.040}],
        "look": [{"name": "well", "point": [0.0, 0.0, MUG_SEAT_Z]}],
        "anchor": [{"name": "mug", "point": [0.0, 0.0, MUG_SEAT_Z]}],
    }
    return K.card(use=use, origin="table_top", front="-Y",
                  colliders=[{"type": "cylinder", "object": col.name, "R": MUG_SAUCER_R, "half_h": MUG_SAUCER_H / 2,
                              "rnd": 0.002, "tag": name}])


@register("cafe_mug")
def cafe_mug(name, coll, root, slots=None):
    """The mug with its tea, paper tag, cotton string and steam. Origin = centre of the foot ring base (stands in the well
    of a `cafe_saucer` at `saucer at + (0, 0, 0.0058)`), +Z up; the handle is at azimuth -25 deg (0 = +X, -90 = -Y), the tag
    and string at 215 deg. Objects (all children of the root, identity transforms): <name>_body, _tea, _tag, _string,
    _steam, and the hidden collider _col.

    Custom property on the root: `steam` (0..1, default 0.6) - strength of the steam plume. The plume scrolls with the
    scene time and sways through two shape keys driven by sin(time); the rates follow the scene's fps.

    Card: grip `handle` (pinch on the strap's vertical run: `normal` = the squeeze direction along the mug's tangent,
    `x` = fingertips towards the mug, `axis` = up the strap, `width` = strap width between the pads, `depth` = strap
    thickness along x, `length` = straight run, `span` = clear opening between strap and wall) and `body` (cylinder wrap on
    the wall); look points rim / tea / tag / handle_out; rest `thumb_rest` (the flat top of the handle); surface `tag`."""
    K = Kit(name, coll, root, slots)
    K.prop("steam", 0.6, 0.0, 1.0, "steam strength: plume alpha and wisp density")
    C = _mg_colors(K)
    body, handle_pts = _mg_mug(K, _mg_mat_glaze(K, C, "glaze"))
    _mg_tea(K, _mg_mat_tea(K, C))
    _mg_tag(K, _mg_mat_tag(K, C))
    _mg_string(K, _mg_mat_string(K, C))
    _mg_steam(K, _mg_mat_steam(K, C))
    col = K.collider(K.to_obj("col", _mg_cylinder_bm(0.045, MUG_H), loc=(0.0, 0.0, MUG_H / 2)))

    grip, outer = _mg_handle_use(handle_pts)
    zm = (_MG_ZW0 + _MG_ZW1) / 2
    tag_c, tag_n = _mg_tag_pt(0.0, 0.0)
    e = 1e-4
    tag_up = (_mg_tag_pt(0.0, e)[0] - _mg_tag_pt(0.0, -e)[0]).normalized()
    r = lambda v: [round(c, 5) for c in v]
    use = {
        "grip": [grip,
                 {"name": "body", "type": "cylinder", "center": [0.0, 0.0, round(zm, 5)], "axis": [0.0, 0.0, 1.0],
                  "radius": round(_mg_r_out(zm), 5), "height": round(_MG_ZW1 - _MG_ZW0, 5)}],
        "look": [{"name": "rim", "point": [0.0, 0.0, MUG_H]},
                 {"name": "tea", "point": [0.0, 0.0, MUG_TEA_Z]},
                 {"name": "tag", "point": r(tag_c)},
                 {"name": "handle_out", "point": r(outer)}],
        "surface": [{"name": "tag", "center": r(tag_c), "normal": r(tag_n), "up": r(tag_up),
                     "size": [_MG_TAG_W, _MG_TAG_H]}],
    }
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for c in body.bound_box:
        lo, hi = Vector(map(min, lo, c)), Vector(map(max, hi, c))
    return K.card(use=use, origin="foot_center", front="-Y", size=[round(v, 4) for v in (hi - lo)],
                  colliders=[{"type": "cylinder", "object": col.name, "R": 0.045, "half_h": MUG_H / 2, "rnd": 0.01,
                              "tag": name}])
