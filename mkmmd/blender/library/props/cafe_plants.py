"""Plants of the cafe: a trailing pothos on the window sill (`cafe_pothos`), a haworthia in a small white pot
(`cafe_haworthia`) and a floor monstera (`cafe_monstera`). Each is its own prop with its own root; they share a private
leaf toolkit (spine frames, polylines, shader-node shorthands) and the per-vertex data their shaders read.

Every plant is a pot (lathe), a soil surface (same lathe, second material slot) and ONE plant mesh (stems and blades)
whose faces carry the data the leaf shaders need: two per-vertex floats `pl_seed` (random per leaf) and `pl_age`
(0 = young, 1 = old) and the UV layers `UVMap` (u across / v along a blade, or around / along a tube), `UVGore` (gore
width, vein length) and `UVLeaf` (planar metres). The geometry is deterministic (fixed seeds), so a leaf always lands
where it did when the plant was first assembled.

Custom properties: none (the plants are static).

Colours are palette-slot blends (`COLORS`); the greens are rotations of the foam slot (cool) and the gold slot (warm), so
a project palette recolours them consistently."""
import functools
import math
import random

import bmesh
from mathutils import Vector

from . import register
from .cafe_kit import Kit, N, bm_lathe, bm_tube, catmull, principled, ramp
from .cafe_kit import L as link

# ================================================================= colours
# Kit.blend recipes (fitted on Rose Pine Dawn, OKLab error x100 <= 1.7, most < 0.5).
COLORS = {
    # potting soil (shared by the pothos and the monstera)
    "soil_dark": dict(hl_high=.65, gold=.35, k=.4),
    "soil_mid": dict(hl_high=.65, gold=.35, k=.5),
    "soil_light": dict(hl_high=.7, gold=.3, k=.7),
    "soil_fleck": dict(hl_high=.8, gold=.2),
    # pothos pot: blush glaze dipped over raw terracotta
    "glaze_light": dict(surface=.5, rose=.5),
    "glaze_deep": dict(rose=.65, base=.35),
    "glaze_fleck_light": dict(overlay=1),
    "glaze_fleck_dark": dict(rose=.65, overlay=.35, k=.6),
    "terracotta_a": dict(gold=.5, muted=.5, k=1.2),
    "terracotta_b": dict(gold=.5, hl_high=.5),
    # monstera pot: speckled cream stoneware dipped in dusty sage
    "cream_a": dict(overlay=1),
    "cream_b": dict(overlay=.75, gold=.25),
    "cream_fleck_cool": dict(muted=1),
    "cream_fleck_warm": dict(overlay=.6, gold=.4, k=.55),
    "sage_a": dict(foam=.65, hl_low=.35, k=1.1, hue=-70, chroma=1.5),
    "sage_b": dict(foam=.65, overlay=.35, k=.95, hue=-65, chroma=2.0),
    "sage_fleck": dict(surface=.8, foam=.2),
    # pothos blades: deep green -> green -> lime -> yellow -> cream marbling, pale veins, paler underside
    "poth_deep": dict(foam=1, k=.65, hue=-45, chroma=1.25),
    "poth_mid": dict(foam=1, k=.95, hue=-50, chroma=1.25),
    "poth_lime": dict(gold=.65, overlay=.35, k=.7, hue=65, chroma=1.5),
    "poth_yellow": dict(gold=.65, base=.35, k=.95, hue=30, chroma=1.25),
    "poth_cream": dict(gold=.5, surface=.5, k=1.15, hue=35),
    "poth_young": dict(gold=.65, surface=.35, k=1.05, hue=60, chroma=1.25),
    "poth_vein": dict(gold=.2, overlay=.8, k=1.1, hue=60, chroma=1.75),
    "poth_under": dict(foam=.65, overlay=.35, k=1.3, hue=-65, chroma=2.0),
    "poth_stem_dark": dict(foam=.8, overlay=.2, k=.75, hue=-60, chroma=2.25),
    "poth_stem_light": dict(foam=.8, overlay=.2, k=1.15, hue=-65, chroma=2.0),
    # monstera blades and stem: deep pine/foam greens, lighter veins and margins, paler underside
    "mons_dark": dict(foam=1, k=.65, hue=-40, chroma=1.25),
    "mons_light": dict(foam=.8, surface=.2, k=.65, hue=-40, chroma=2.5),
    "mons_margin": dict(foam=.8, hl_low=.2, k=1.05, hue=-45, chroma=1.75),
    "mons_cross": dict(foam=.8, surface=.2, k=1.3, hue=-45, chroma=1.5),
    "mons_young": dict(foam=.8, base=.2, k=1.35, hue=-55, chroma=2.25),
    "mons_vein": dict(foam=.8, surface=.2, k=1.2, hue=-45, chroma=1.75),
    "mons_rib": dict(foam=.65, overlay=.35, k=1.2, hue=-40, chroma=2.5),
    "mons_under": dict(foam=.8, surface=.2, k=1.2, hue=-45, chroma=1.5),
    "mons_stem_dark": dict(foam=1, k=1.15, hue=-55),
    "mons_stem_light": dict(foam=.8, base=.2, k=1.25, hue=-60, chroma=1.75),
    "aerial_root": dict(hl_med=.55, gold=.45, k=.7),
    # haworthia: matte teal leaf with pale bands, white glazed pot, sand and pale stones
    "haw_leaf": dict(foam=.65, hl_high=.35, k=.53, hue=-50, chroma=2.0),
    "haw_leaf_light": dict(foam=.65, hl_high=.35, k=.62, hue=-50, chroma=2.0),
    "haw_band": dict(gold=.2, surface=.8, k=.8, hue=35, chroma=1.25),
    "haw_tip": dict(surface=.7, foam=.3),
    "haw_pot": dict(overlay=.8, surface=.2, k=.8),
    "haw_pot_cool": dict(overlay=.8, muted=.2, k=.9, chroma=.5),
    "haw_pot_dark": dict(overlay=.8, muted=.2, k=.55, chroma=.5),
    "haw_sand_dark": dict(hl_high=.65, gold=.35, k=.36),
    "haw_sand_light": dict(hl_high=.65, gold=.35, k=.45),
    "haw_stone_a": dict(overlay=.65, surface=.35, k=.8, chroma=1.75),
    "haw_stone_b": dict(base=.5, surface=.5, k=.65, chroma=2.25),
    "haw_stone_c": dict(base=.8, rose=.2, k=.8, chroma=1.25),
}


def _c(K, name):
    return K.blend(**COLORS[name])


_UP = Vector((0.0, 0.0, 1.0))
_GRAVITY = Vector((0.0, 0.0, -1.0))


# ================================================================= small maths
def _smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def _rodrigues(v, axis, c, s):
    return v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))


def _perp(t):
    a = Vector((1.0, 0.0, 0.0)) if abs(t.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    return (a - t * a.dot(t)).normalized()


def _basis(m0, n_pref, roll=0.0):
    """Orthonormal (T, N): T along m0, N = n_pref made perpendicular to T, optionally rolled about T."""
    T = Vector(m0).normalized()
    npf = Vector(n_pref)
    N_ = npf - T * npf.dot(T)
    if N_.length < 1e-4:
        N_ = _perp(T)
    N_.normalize()
    if roll:
        N_ = N_ * math.cos(roll) + T.cross(N_) * math.sin(roll)
    return T, N_


class _Frame:
    """Leaf spine: rigid frame (X across, T along, N normal) transported along a circular arc that bends toward D
    (default: gravity) with curvature k [rad/m], optional twist [rad/m] about T.   at(y): y = arc length from base."""

    def __init__(self, O, T0, N0, kappa=0.0, D=None, twist=0.0):
        T0 = Vector(T0).normalized()
        N0 = Vector(N0)
        N0 = N0 - T0 * N0.dot(T0)
        N0 = N0.normalized() if N0.length > 1e-6 else _perp(T0)
        self.O, self.T0, self.N0, self.X0 = Vector(O), T0, N0, T0.cross(N0)
        if D is None:
            D = _GRAVITY - T0 * _GRAVITY.dot(T0)
            D = D.normalized() if D.length > 1e-3 else -N0
        else:
            D = Vector(D)
            D = (D - T0 * D.dot(T0)).normalized()
        self.D, self.B = D, T0.cross(D)
        self.k, self.tw = kappa, twist

    def at(self, y):
        th = self.k * y
        c, s = math.cos(th), math.sin(th)
        if abs(self.k) > 1e-9:
            S = self.O + self.T0 * (s / self.k) + self.D * ((1.0 - c) / self.k)
        else:
            S = self.O + self.T0 * y
        X = _rodrigues(self.X0, self.B, c, s)
        T = _rodrigues(self.T0, self.B, c, s)
        N_ = _rodrigues(self.N0, self.B, c, s)
        if self.tw:
            a = self.tw * y
            ca, sa = math.cos(a), math.sin(a)
            X, N_ = X * ca + N_ * sa, N_ * ca - X * sa
        return S, X, T, N_

    def pt(self, x, y, z=0.0):
        S, X, _, N_ = self.at(y)
        return S + X * x + N_ * z


def _layers(bm):
    """Per-vertex float layers (leaf seed / age) and the UV layers read by the leaf shaders."""
    return {"seed": bm.verts.layers.float.new("pl_seed"), "age": bm.verts.layers.float.new("pl_age"),
            "uv0": bm.loops.layers.uv.new("UVMap"), "uv1": bm.loops.layers.uv.new("UVGore"),
            "uv2": bm.loops.layers.uv.new("UVLeaf")}


def _emit(bm, lay, m2, place, mat, seed, age):
    """2D leaf mesh {'verts': [(x,y)], 'faces': [idx...], 'uv': [[(u,v,gw,al)...]]} -> bmesh via place(x, y)."""
    vs = []
    for (x, y) in m2["verts"]:
        v = bm.verts.new(place(x, y))
        v[lay["seed"]] = seed
        v[lay["age"]] = age
        vs.append(v)
    for idx, fuv in zip(m2["faces"], m2["uv"]):
        vl = [vs[i] for i in idx]
        if len(set(vl)) < len(vl):
            continue
        try:
            f = bm.faces.new(vl)
        except ValueError:
            continue
        f.smooth = True
        f.material_index = mat
        for lp, i, (u, v, gw, al) in zip(f.loops, idx, fuv):
            lp[lay["uv0"]].uv = (u, v)
            lp[lay["uv1"]].uv = (gw, al)
            lp[lay["uv2"]].uv = m2["verts"][i]
    return vs


def _path_cum(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    return cum


def _path_at(pts, cum, s):
    """Point and unit tangent at arc length s along a dense polyline."""
    s = min(max(s, 0.0), cum[-1] - 1e-9)
    i = 0
    while i + 2 < len(cum) and cum[i + 1] < s:
        i += 1
    t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
    p = pts[i] * (1 - t) + pts[i + 1] * t
    d = pts[min(i + 2, len(pts) - 1)] - pts[max(i - 1, 0)]
    return p, d.normalized()


def _smooth_poly(pts, iters=2):
    pts = [Vector(p) for p in pts]
    for _ in range(iters):
        q = [pts[0]]
        for i in range(1, len(pts) - 1):
            q.append(pts[i] * 0.5 + (pts[i - 1] + pts[i + 1]) * 0.25)
        q.append(pts[-1])
        pts = q
    return pts


def _bezier(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t * t
        d = t ** 3
        out.append(p0 * a + p1 * b + p2 * c + p3 * d)
    return out


# ================================================================= shader-node shorthands
def _math(nt, op, a, b=None, c=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _mul(nt, *xs):
    """Product of sockets and numbers."""
    out = xs[0]
    for x in xs[1:]:
        out = _math(nt, "MULTIPLY", out, x)
    return out


def _range(nt, val, fmin, fmax, tmin=0.0, tmax=1.0, smooth=False, clamp=True):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
    n.clamp = clamp
    nt.links.new(val, n.inputs["Value"])
    for k, v in (("From Min", fmin), ("From Max", fmax), ("To Min", tmin), ("To Max", tmax)):
        if isinstance(v, (int, float)):
            n.inputs[k].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[k])
    return n.outputs["Result"]


def _mix(nt, fac, a, b, blend="MIX", clamp=True):
    """Colour mix (ShaderNodeMix, RGBA); a/b are RGBA tuples or colour sockets, fac a float or socket."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    n.clamp_result = clamp
    if isinstance(fac, (int, float)):
        n.inputs[0].default_value = float(fac)
    else:
        nt.links.new(fac, n.inputs[0])
    for i, v in ((6, a), (7, b)):
        if isinstance(v, tuple):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[2]


def _attr(nt, name):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_type = "GEOMETRY"
    n.attribute_name = name
    return n.outputs["Fac"]


def _noise(nt, vec, scale, detail=4.0, rough=0.55, distort=0.0, dims="3D"):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = dims
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = distort
    nt.links.new(vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def _speckles(nt, vec, scale, density, radius=0.5):
    """Round random dots: Voronoi cells (random per cell) thresholded by density; returns 0..1 mask socket."""
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.voronoi_dimensions = "3D"
    v.feature = "F1"
    v.inputs["Scale"].default_value = scale
    v.inputs["Randomness"].default_value = 1.0
    nt.links.new(vec, v.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(v.outputs["Color"], sep.inputs["Color"])
    on = _range(nt, sep.outputs["Red"], 1.0 - density - 0.02, 1.0 - density + 0.02, 0.0, 1.0)
    dot = _range(nt, v.outputs["Distance"], radius * 0.55, radius, 1.0, 0.0, smooth=True)
    return _math(nt, "MULTIPLY", on, dot)


def _ramp(nt, fac, stops, interp="LINEAR"):
    r = ramp(nt, (0, 0), stops, interp)
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def _bump(nt, height, strength, distance):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    link(nt, height, b.inputs["Height"])
    return b.outputs["Normal"]


# ================================================================= materials
def _mat_soil(K):
    """Soft muted potting soil: fine grain noise, pale perlite flecks (never dark)."""
    m, nt, out = K.new_mat("soil")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    n1 = _noise(nt, ob, 160.0, 6.0, 0.62)
    n2 = _noise(nt, ob, 22.0, 3.0, 0.5)
    mixn = _math(nt, "ADD", _math(nt, "MULTIPLY", n1, 0.7), _math(nt, "MULTIPLY", n2, 0.3))
    col = _ramp(nt, _range(nt, mixn, 0.30, 0.66), [(0.0, _c(K, "soil_dark")), (0.5, _c(K, "soil_mid")),
                                                   (1.0, _c(K, "soil_light"))])
    fl = _speckles(nt, ob, 420.0, 0.10, 0.5)
    col = _mix(nt, _math(nt, "MULTIPLY", fl, 0.8), col, _c(K, "soil_fleck"))
    nrm = _bump(nt, _math(nt, "ADD", n1, _math(nt, "MULTIPLY", fl, 0.5)), 0.4, 0.0025)
    b = principled(nt, out, **{"Roughness": 0.93, "Specular IOR Level": 0.25})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_blush(K):
    """Pothos pot: blush glaze dipped over raw terracotta (wobbly dip line ~3 cm above the base), light + warm speckles."""
    m, nt, out = K.new_mat("blush_glaze")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    sep = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, ob, sep.inputs["Vector"])
    wob = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", _noise(nt, ob, 26.0, 2.0, 0.5, 0.4), 0.5), 0.02)
    dip = _range(nt, _math(nt, "ADD", sep.outputs["Z"], wob), 0.027, 0.0315, 0.0, 1.0, smooth=True)
    var = _range(nt, _noise(nt, ob, 9.0, 3.0, 0.5, 0.3), 0.35, 0.65)
    glaze = _mix(nt, var, _c(K, "glaze_light"), _c(K, "glaze_deep"))
    sp1 = _speckles(nt, ob, 330.0, 0.16, 0.45)
    sp2 = _speckles(nt, ob, 210.0, 0.07, 0.5)
    glaze = _mix(nt, _math(nt, "MULTIPLY", sp1, 0.85), glaze, _c(K, "glaze_fleck_light"))
    glaze = _mix(nt, _math(nt, "MULTIPLY", sp2, 0.7), glaze, _c(K, "glaze_fleck_dark"))
    raw = _mix(nt, _range(nt, _noise(nt, ob, 120.0, 4.0, 0.6), 0.3, 0.7), _c(K, "terracotta_a"),
               _c(K, "terracotta_b"))
    col = _mix(nt, dip, raw, glaze)
    rough = _math(nt, "ADD", _math(nt, "MULTIPLY", dip, -0.55), 0.85)
    coat = _math(nt, "MULTIPLY", dip, 0.4)
    nrm = _bump(nt, _noise(nt, ob, 260.0, 3.0, 0.6), 0.25, 0.001)
    b = principled(nt, out, **{"Specular IOR Level": 0.5, "Coat Roughness": 0.12})
    link(nt, col, b.inputs["Base Color"])
    link(nt, rough, b.inputs["Roughness"])
    link(nt, coat, b.inputs["Coat Weight"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_cream_sage(K):
    """Monstera planter: speckled cream stoneware, upper half dipped in glossy dusty-sage glaze (wavy dip line)."""
    m, nt, out = K.new_mat("cream_sage")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    sep = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, ob, sep.inputs["Vector"])
    ang = _math(nt, "ARCTAN2", sep.outputs["Y"], sep.outputs["X"])
    wob = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "SINE", _math(nt, "MULTIPLY", ang, 3.0)),
                                      _math(nt, "MULTIPLY", _math(nt, "SINE", _math(nt, "ADD", _math(
                                          nt, "MULTIPLY", ang, 7.0), 1.3)), 0.6)), 0.012)
    wob = _math(nt, "ADD", wob, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", _noise(nt, ob, 14.0, 2.0, 0.5, 0.3), 0.5),
                                      0.03))
    dip = _range(nt, _math(nt, "ADD", sep.outputs["Z"], wob), 0.150, 0.158, 0.0, 1.0, smooth=True)
    var = _range(nt, _noise(nt, ob, 6.0, 3.0, 0.5, 0.3), 0.3, 0.7)
    cream = _mix(nt, var, _c(K, "cream_a"), _c(K, "cream_b"))
    sage = _mix(nt, var, _c(K, "sage_a"), _c(K, "sage_b"))
    sp = _speckles(nt, ob, 300.0, 0.20, 0.42)
    sp_b = _speckles(nt, ob, 170.0, 0.06, 0.5)
    cream = _mix(nt, _math(nt, "MULTIPLY", sp, 0.85), cream, _c(K, "cream_fleck_cool"))
    cream = _mix(nt, _math(nt, "MULTIPLY", sp_b, 0.7), cream, _c(K, "cream_fleck_warm"))
    sage = _mix(nt, _math(nt, "MULTIPLY", sp, 0.5), sage, _c(K, "sage_fleck"))
    col = _mix(nt, dip, cream, sage)
    rough = _math(nt, "ADD", _math(nt, "MULTIPLY", dip, -0.38), 0.72)
    coat = _math(nt, "MULTIPLY", dip, 0.35)
    nrm = _bump(nt, _noise(nt, ob, 220.0, 3.0, 0.6), 0.3, 0.0015)
    b = principled(nt, out, **{"Specular IOR Level": 0.5, "Coat Roughness": 0.15})
    link(nt, col, b.inputs["Base Color"])
    link(nt, rough, b.inputs["Roughness"])
    link(nt, coat, b.inputs["Coat Weight"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_pothos_leaf(K):
    """Golden-green / cream marbled pothos leaf, glossy; planar UVMap (u across 0..1, v along 0..1), pl_seed/pl_age
    attributes. Underside (Backfacing) paler and rougher; lateral veins + midrib from the UV, bumped."""
    m, nt, out = K.new_mat("leaf")
    m.use_backface_culling = False
    tc = N(nt, "ShaderNodeTexCoord")
    uvn = N(nt, "ShaderNodeUVMap", uv_map="UVMap")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, uvn.outputs["UV"], sep.inputs["Vector"])
    xa = _math(nt, "MULTIPLY", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", sep.outputs["X"], 0.5)), 0.8)
    yv = sep.outputs["Y"]
    t = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", yv, _math(nt, "MULTIPLY", xa, 1.5)), 0.15)
    d = _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", _math(nt, "FRACT", _math(nt, "ADD", t, 0.5)), 0.5))
    vein = _range(nt, d, 0.012, 0.05, 1.0, 0.0, smooth=True)
    vein = _math(nt, "MULTIPLY", vein, _range(nt, xa, 0.02, 0.36, 0.6, 1.0))
    mid = _range(nt, xa, 0.007, 0.03, 1.0, 0.0, smooth=True)
    mid = _math(nt, "MULTIPLY", mid, _range(nt, yv, 0.1, 0.95, 1.0, 0.25))
    seed = _attr(nt, "pl_seed")
    age = _attr(nt, "pl_age")
    off = N(nt, "ShaderNodeCombineXYZ")
    for ax in ("X", "Y", "Z"):
        link(nt, _math(nt, "MULTIPLY", seed, {"X": 41.0, "Y": 23.0, "Z": 11.0}[ax]), off.inputs[ax])
    vadd = N(nt, "ShaderNodeVectorMath", operation="ADD")
    link(nt, tc.outputs["Object"], vadd.inputs[0])
    link(nt, off.outputs["Vector"], vadd.inputs[1])
    n1 = _noise(nt, vadd.outputs["Vector"], 40.0, 4.0, 0.55, 0.4)
    n2 = _noise(nt, vadd.outputs["Vector"], 130.0, 2.0, 0.5)
    shift = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", _math(nt, "FRACT", _math(nt, "MULTIPLY", seed, 7.31)), 0.5),
                  0.16)
    shift = _math(nt, "ADD", shift, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, age), 0.07))
    f = _math(nt, "ADD", _math(nt, "ADD", n1, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", n2, 0.5), 0.18)), shift)
    f = _range(nt, f, 0.42, 0.74)
    col = _ramp(nt, f, [(0.0, _c(K, "poth_deep")), (0.30, _c(K, "poth_mid")), (0.50, _c(K, "poth_lime")),
                        (0.70, _c(K, "poth_yellow")), (1.0, _c(K, "poth_cream"))])
    young = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, age), 0.4)
    col = _mix(nt, young, col, _c(K, "poth_young"))
    col = _mix(nt, _math(nt, "MULTIPLY", vein, 0.14), col, _c(K, "poth_vein"))
    col = _mix(nt, _math(nt, "MULTIPLY", mid, 0.40), col, _c(K, "poth_vein"))
    back = N(nt, "ShaderNodeNewGeometry").outputs["Backfacing"]
    under = _mix(nt, 0.55, col, _c(K, "poth_under"))
    col = _mix(nt, back, col, under)
    rough = _math(nt, "ADD", 0.30, _math(nt, "MULTIPLY", back, 0.2))
    nrm = _bump(nt, _math(nt, "ADD", _math(nt, "MULTIPLY", vein, 0.6), _math(nt, "MULTIPLY", mid, 0.5)), 0.35, 0.0015)
    b = principled(nt, out, **{"Specular IOR Level": 0.42, "Coat Weight": 0.08, "Coat Roughness": 0.12})
    link(nt, col, b.inputs["Base Color"])
    link(nt, rough, b.inputs["Roughness"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_stem(K, base, c_dark, c_light, rough=0.45, streak=60.0):
    """Green stem / petiole: lengthwise streaks from the tube UVs (u along, v around)."""
    m, nt, out = K.new_mat(base)
    uvn = N(nt, "ShaderNodeUVMap", uv_map="UVMap")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, uvn.outputs["UV"], sep.inputs["Vector"])
    comb = N(nt, "ShaderNodeCombineXYZ")
    link(nt, _math(nt, "MULTIPLY", sep.outputs["Y"], streak), comb.inputs["X"])
    link(nt, _math(nt, "MULTIPLY", sep.outputs["X"], 2.5), comb.inputs["Y"])
    n = _noise(nt, comb.outputs["Vector"], 1.0, 3.0, 0.55)
    col = _ramp(nt, _range(nt, n, 0.25, 0.75), [(0.0, _c(K, c_dark)), (1.0, _c(K, c_light))])
    nrm = _bump(nt, n, 0.25, 0.0008)
    b = principled(nt, out, **{"Roughness": rough, "Specular IOR Level": 0.5, "Coat Weight": 0.08})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_monstera_leaf(K):
    """Deep soft pine/foam green, glossy-satin, two-sided (paler underside). Lobe-local UVs (u across the gore, v along
    the vein), UVGore = (gore width, vein length), UVLeaf = planar metres: lateral veins (lighter, thinner toward the
    margin), midrib, faint cross veins, quilted bump, per-leaf age tint."""
    m, nt, out = K.new_mat("leaf")
    m.use_backface_culling = False
    tc = N(nt, "ShaderNodeTexCoord")
    sa = N(nt, "ShaderNodeSeparateXYZ")
    sb = N(nt, "ShaderNodeSeparateXYZ")
    sc_ = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, N(nt, "ShaderNodeUVMap", uv_map="UVMap").outputs["UV"], sa.inputs["Vector"])
    link(nt, N(nt, "ShaderNodeUVMap", uv_map="UVGore").outputs["UV"], sb.inputs["Vector"])
    link(nt, N(nt, "ShaderNodeUVMap", uv_map="UVLeaf").outputs["UV"], sc_.inputs["Vector"])
    u, f = sa.outputs["X"], sa.outputs["Y"]
    gw, al = sb.outputs["X"], sb.outputs["Y"]
    xl = sc_.outputs["X"]
    # distance (metres) from the nearest lateral vein
    dn = _math(nt, "MULTIPLY", _math(nt, "MINIMUM", u, _math(nt, "SUBTRACT", 1.0, u)), gw)
    vw = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, f), 0.0028), 0.0009)
    vein = _range(nt, dn, vw, _math(nt, "MULTIPLY", vw, 2.4), 1.0, 0.0, smooth=True)
    # midrib band from the planar x
    xa = _math(nt, "ABSOLUTE", xl)
    mid = _range(nt, xa, 0.0022, 0.0075, 1.0, 0.0, smooth=True)
    # faint cross veins (angled lines between laterals), only in the lamina
    sd = _math(nt, "MULTIPLY", f, al)
    cr = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", sd, _math(nt, "MULTIPLY", dn, 1.1)), 0.0105)
    cd = _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", _math(nt, "FRACT", cr), 0.5))
    cross = _range(nt, cd, 0.30, 0.5, 0.0, 1.0)
    cross = _math(nt, "MULTIPLY", cross, _range(nt, dn, 0.003, 0.012, 0.0, 1.0))
    seed = _attr(nt, "pl_seed")
    age = _attr(nt, "pl_age")
    off = N(nt, "ShaderNodeCombineXYZ")
    for ax, k in (("X", 17.0), ("Y", 9.0), ("Z", 5.0)):
        link(nt, _math(nt, "MULTIPLY", seed, k), off.inputs[ax])
    vadd = N(nt, "ShaderNodeVectorMath", operation="ADD")
    link(nt, tc.outputs["Object"], vadd.inputs[0])
    link(nt, off.outputs["Vector"], vadd.inputs[1])
    nlo = _noise(nt, vadd.outputs["Vector"], 3.0, 3.0, 0.5, 0.5)
    nhi = _noise(nt, vadd.outputs["Vector"], 38.0, 3.0, 0.5)
    tone = _range(nt, _math(nt, "ADD", _math(nt, "MULTIPLY", nlo, 0.8), _math(nt, "MULTIPLY", nhi, 0.2)), 0.30, 0.70)
    col = _mix(nt, tone, _c(K, "mons_dark"), _c(K, "mons_light"))
    # lighter toward the margin of each lobe and along the midrib region
    col = _mix(nt, _math(nt, "MULTIPLY", _range(nt, dn, 0.0, 0.05, 0.0, 1.0), 0.06), col, _c(K, "mons_margin"))
    col = _mix(nt, _math(nt, "MULTIPLY", cross, 0.02), col, _c(K, "mons_cross"))
    young = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, age), 0.85)
    col = _mix(nt, young, col, _c(K, "mons_young"))
    col = _mix(nt, _math(nt, "MULTIPLY", vein, 0.34), col, _c(K, "mons_vein"))
    col = _mix(nt, _math(nt, "MULTIPLY", mid, 0.38), col, _c(K, "mons_rib"))
    back = N(nt, "ShaderNodeNewGeometry").outputs["Backfacing"]
    under = _mix(nt, 0.5, col, _c(K, "mons_under"))
    col = _mix(nt, back, col, under)
    rough = _math(nt, "ADD", 0.36, _math(nt, "MULTIPLY", back, 0.14))
    hv = _math(nt, "MULTIPLY", vein, -0.9)
    nrm = _bump(nt, _math(nt, "ADD", hv, _math(nt, "MULTIPLY", mid, 0.6)), 0.35, 0.003)
    b = principled(nt, out, **{"Specular IOR Level": 0.28, "Coat Weight": 0.03, "Coat Roughness": 0.2})
    link(nt, col, b.inputs["Base Color"])
    link(nt, rough, b.inputs["Roughness"])
    link(nt, nrm, b.inputs["Normal"])
    return m


# ================================================================= pots (lathe profiles: (radius, height) from the base)
def _pot_collider(K, radius, height, rnd=0.004):
    """Hidden cylinder over a pot, base at the prop origin (the card's collider spec)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=radius, radius2=radius, depth=height)
    o = K.collider(K.to_obj("col_pot", bm, loc=(0.0, 0.0, height / 2)))
    return {"type": "cylinder", "object": o.name, "R": radius, "half_h": height / 2, "rnd": rnd, "tag": K.name}


_POTHOS_PROF = [(0.0, 0.0), (0.036, 0.0), (0.0405, 0.0015), (0.0425, 0.005), (0.0445, 0.02), (0.0475, 0.045),
                (0.0525, 0.075), (0.0590, 0.103), (0.0640, 0.1155), (0.0660, 0.1180), (0.0658, 0.1205),
                (0.0640, 0.1222), (0.0615, 0.1210), (0.0600, 0.1182), (0.0580, 0.1120), (0.0566, 0.0985),
                # soil surface (material 1): hugging the wall, gentle dome
                (0.0555, 0.0990), (0.0450, 0.1008), (0.0300, 0.1028), (0.0150, 0.1040), (0.0, 0.1044)]
_POTHOS_SOIL_FIRST = 15      # profile index where the soil material starts


def _pothos_pot_r(z):
    """Outer radius of the pothos pot at height z (piecewise linear from the profile's outer wall)."""
    pr = _POTHOS_PROF[:15]
    if z <= pr[1][1]:
        return pr[1][0]
    for i in range(1, len(pr) - 1):
        if pr[i][1] <= z <= pr[i + 1][1] and pr[i + 1][1] > pr[i][1]:
            t = (z - pr[i][1]) / (pr[i + 1][1] - pr[i][1])
            return pr[i][0] * (1 - t) + pr[i + 1][0] * t
    return 0.0658


_MONS_POT_PROF = [(0.0, 0.0), (0.100, 0.0), (0.108, 0.004), (0.1115, 0.012), (0.1160, 0.05), (0.1260, 0.12),
                  (0.1420, 0.20), (0.1590, 0.26), (0.1685, 0.292), (0.1715, 0.296), (0.1720, 0.300), (0.1700, 0.3035),
                  (0.1670, 0.3025), (0.1650, 0.299), (0.1620, 0.285), (0.1590, 0.2625),
                  # soil (material 1)
                  (0.1570, 0.2630), (0.1400, 0.2655), (0.0900, 0.2690), (0.0450, 0.2715), (0.0, 0.2725)]
_MONS_SOIL_FIRST = 16
_MONS_SOIL_Z = 0.262


# ================================================================= pothos
# The sill the pothos stands on is part of the room, not of the prop: the vines are shaped to hang over its rounded
# front edge and to keep out of the window frame, the mullion bars and the wall behind it. Built in the prop's frame
# (origin = pot base centre = the sill top, +X = toward the room, +Y along the sill):
SILL_EDGE_X = 0.10        # front edge of the sill board
SILL_THICK = 0.042        # board thickness below the sill top
FRAME_FACE_X = -0.063     # front face of the window frame's bottom member (+2 mm), for z < FRAME_TOP_Z
FRAME_TOP_Z = 0.09        # top of that member above the sill
BAR_FACE_X = -0.1225      # 3 cm in front of the mullion bars, above the frame member
WALL_X = -0.075           # the wall face under the sill (+5 mm)


@functools.lru_cache(maxsize=None)
def _pothos_blade2d():
    """Heart-shaped pothos blade (acuminate tip, rounded basal lobes, shallow sinus) as a fan of 3 concentric rings:
    (verts, faces, uv) in unit coordinates, origin at the leaf centre line; UVMap = (u across, v along)."""
    right = [(0.000, 1.000), (0.020, 0.950), (0.058, 0.885), (0.115, 0.800), (0.185, 0.700), (0.255, 0.585),
             (0.318, 0.460), (0.360, 0.335), (0.378, 0.220), (0.360, 0.115), (0.310, 0.040), (0.240, 0.000),
             (0.170, 0.005), (0.105, 0.040), (0.050, 0.085), (0.000, 0.105)]
    dense = catmull([(x, y, 0.0) for x, y in right], 2)
    ring = [(p.x, p.y) for p in dense]
    ring = ring + [(-p.x, p.y) for p in reversed(dense[1:-1])]
    R = len(ring)
    cx, cy = 0.0, 0.40
    base = [(cx, cy)]
    for s in (0.34, 0.67, 1.0):
        for (x, y) in ring:
            base.append((cx + s * (x - cx), cy + s * (y - cy)))
    faces = [(0, 1 + (i + 1) % R, 1 + i) for i in range(R)]
    for k in range(2):
        a0, a1 = 1 + k * R, 1 + (k + 1) * R
        for i in range(R):
            j = (i + 1) % R
            faces.append((a0 + i, a0 + j, a1 + j, a1 + i))
    uv = [[(0.5 + base[i][0] / 0.8, base[i][1], 0.0, 0.0) for i in f] for f in faces]
    return base, faces, uv


def _pothos_leaf2d(L, aspect):
    """The blade in metres, origin at the sinus (petiole attachment), +y toward the tip."""
    base, faces, uv = _pothos_blade2d()
    y0 = 0.105
    return {"verts": [(x * aspect * L, (y - y0) * L) for (x, y) in base], "faces": faces, "uv": uv}


def _pothos_leaf(bm, lay, rng, O, m0, n_pref, L, seed, age, mat=0):
    """One blade with V-fold, margin sag + wave, drooping/twisting spine."""
    T, Nn = _basis(m0, n_pref, rng.uniform(-0.55, 0.55))
    fr = _Frame(O, T, Nn, kappa=rng.uniform(3.5, 9.0), twist=rng.uniform(-2.2, 2.2))
    aspect = rng.uniform(0.88, 1.06)
    kf = rng.uniform(0.22, 0.52)
    ks = rng.uniform(0.6, 2.0)
    wph = rng.uniform(0, 6.28)
    wamp = rng.uniform(0.012, 0.03)

    def place(x, y):
        xa = abs(x)
        z = kf * xa - ks * x * x / L + wamp * L * math.sin(10.0 * y / L + wph) * min(1.0, xa / (0.3 * L))
        return fr.pt(x, y, z)

    _emit(bm, lay, _pothos_leaf2d(L, aspect), place, mat, seed, age)
    return fr


def _pothos_node_leaf(bm, lay, rng, node, out_h, hang, L, seed, age, stem_mat=1):
    """Leaf at a stem node: petiole from `node` to the blade base; `hang` in [-0.6, 1]: -ve rising leaf, 1 hanging."""
    out_h = Vector(out_h).normalized()
    if out_h.x < 0.0:                                    # keep blades from leaning into the window recess
        out_h.x *= 0.2
        out_h.normalize()
    pd = (out_h * 0.8 + _UP * (0.25 - 0.55 * hang)).normalized()
    m0 = (out_h * (0.55 - 0.45 * hang) + _UP * (0.15 - 1.0 * hang) + pd * 0.2).normalized()
    n_pref = (_UP * (0.9 - 0.6 * hang) + out_h * (0.2 + 0.7 * hang)).normalized()
    lp = L * rng.uniform(0.35, 0.60)
    O = node + pd * lp
    mid = node + pd * (lp * 0.5) + _UP * (lp * 0.12)
    _pothos_leaf(bm, lay, rng, O, m0, n_pref, L, seed, age)
    pts = catmull([node, mid, O], 4)
    bm_tube(bm, pts, lambda t: 0.00115 - 0.0004 * t, sides=4, mat=stem_mat, cap=("none", "flat"))


def _pothos_vine(bm, lay, rng, y_h, y_s, depth, side0, kind="drop"):
    """One trailing vine: soil -> over the pot lip -> ('drop': straight to the sill edge | 'crawl': lies on the sill
    first) -> over the rounded sill edge -> hangs at x ~ SILL_EDGE_X + 0.015 for `depth` metres, swaying gently in y."""
    xe = SILL_EDGE_X
    ctrl = [(0.012, y_s * 0.35, 0.097), (0.038, y_s * 0.65, 0.131), (0.0665, y_s * 0.90 + (y_h - y_s) * 0.15, 0.1318)]
    if kind == "drop":
        ctrl += [(xe - 0.012, y_s + (y_h - y_s) * 0.7, 0.062), (xe + 0.0015, y_h, 0.006)]
    else:
        ctrl += [(0.0745, y_s + (y_h - y_s) * 0.35, 0.050), (0.0765, y_s + (y_h - y_s) * 0.7, 0.0045),
                 (xe - 0.011, y_h, 0.0021), (xe + 0.0015, y_h, 0.0030)]
    ctrl.append((xe + 0.010, y_h, -0.017))
    nh = max(3, int(depth / 0.07))
    ph = rng.uniform(0, 6.28)
    for k in range(1, nh + 1):
        z = -0.017 - depth * k / nh
        ctrl.append((xe + 0.015 + 0.004 * math.sin(ph + k * 1.3) + 0.002 * k / nh,
                     y_h + 0.016 * math.sin(ph * 0.7 + k * 0.95), z))
    pts = catmull(ctrl, 8)
    cum = _path_cum(pts)
    total = cum[-1]
    bm_tube(bm, pts, lambda t: 0.0019 - 0.0007 * t, sides=5, mat=1, cap=("flat", "round"))
    s = rng.uniform(0.035, 0.05)
    side = side0
    while s < total - 0.02:
        p, t = _path_at(pts, cum, s)
        u = s / total
        hang = max(-0.1, min(1.0, -t.z * 1.15))
        az = side * rng.uniform(0.05, 1.45) + rng.uniform(-0.25, 0.25)
        out_h = Vector((math.cos(az), math.sin(az), 0.0))
        age = max(0.1, min(1.0, 1.0 - 0.9 * u))
        L = (0.050 + 0.034 * age) * rng.uniform(0.88, 1.15)
        _pothos_node_leaf(bm, lay, rng, p, out_h, hang, L, rng.random(), age)
        side = -side
        s += rng.uniform(0.050, 0.070) * (0.85 + 0.3 * (1 - age))
    return pts


def _pothos_crown(bm, lay, rng, n_shoots=16):
    """Bushy crown: curved upright shoots from the soil, each with a tip leaf + 2-3 side leaves fanning out/drooping."""
    for i in range(n_shoots):
        az = 2.0 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n_shoots
        r0 = rng.uniform(0.0, 0.03)
        p0 = Vector((r0 * math.cos(az + 1.0), r0 * math.sin(az + 1.0), 0.097))
        el = math.radians(rng.uniform(52, 80) if math.cos(az) > -0.7 else rng.uniform(72, 84))
        ln = rng.uniform(0.04, 0.10) * (0.6 if math.cos(az) < -0.3 else 1.0)
        d0 = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        d1 = Vector((math.cos(az) * math.cos(el - 0.75), math.sin(az) * math.cos(el - 0.75), math.sin(el - 0.75)))
        pts = [p0 + d0 * (ln * t / 6.0) * (1 - t / 6.0 * 0.5) + d1 * (ln * t / 6.0) * (t / 6.0 * 0.5) for t in range(7)]
        pts = catmull(pts, 4)
        bm_tube(bm, pts, lambda t: 0.0022 - 0.0008 * t, sides=5, mat=1, cap=("flat", "round"))
        cum = _path_cum(pts)
        age = rng.uniform(0.3, 0.95)
        a1 = az + rng.uniform(-0.6, 0.6)
        L = (0.056 + 0.034 * age) * rng.uniform(0.95, 1.15)
        upright = ln < 0.062 and rng.random() < 0.6
        _pothos_node_leaf(bm, lay, rng, pts[-1], Vector((math.cos(a1), math.sin(a1), 0.0)),
                          -0.55 if upright else rng.uniform(-0.1, 0.45), L, rng.random(), age)
        for s_rel, sd in ((0.35, 1), (0.62, -1), (0.85, 1)):
            if rng.random() < 0.8:
                p, t = _path_at(pts, cum, cum[-1] * s_rel)
                azl = az + sd * rng.uniform(0.7, 1.7)
                age2 = rng.uniform(0.5, 1.0)
                L2 = (0.054 + 0.034 * age2) * rng.uniform(0.95, 1.15)
                _pothos_node_leaf(bm, lay, rng, p, Vector((math.cos(azl), math.sin(azl), 0.0)),
                                  rng.uniform(0.0, 0.6), L2, rng.random(), age2)


def _pothos_fix(bm):
    """Keep the plant out of the window frame / mullion bars / wall, the sill board and the pot wall (flatten or push
    instead of intersecting)."""
    for v in bm.verts:
        p = v.co
        lim = FRAME_FACE_X if p.z < FRAME_TOP_Z else BAR_FACE_X
        if -SILL_THICK < p.z < 0.0:
            lim = max(lim, SILL_EDGE_X + 0.002)
        elif p.z <= -SILL_THICK:
            lim = WALL_X
        if p.x < lim:
            p.x = lim
        if p.z < 0.0016 and p.x < SILL_EDGE_X:
            p.z = 0.0016
        r = math.hypot(p.x, p.y)
        if p.z < 0.1225:
            R = _pothos_pot_r(max(p.z, 0.002)) + 0.0022
            if 0.0575 < r < R and p.z > 0.0:
                k = R / max(r, 1e-6)
                p.x *= k
                p.y *= k


@register("cafe_pothos")
def cafe_pothos(name, coll, root, slots=None):
    """Trailing pothos in a blush-glazed terracotta pot, standing on a window sill: 5 vines hang over the sill's front
    edge down to ~0.4 m, a bushy crown of upright shoots; 101 heart-shaped blades (V-fold, droop, wave) with marbled
    green/cream colouring.

    Frame: origin = centre of the pot base = the sill top, +Z up, the pot is 0.132 m wide and 0.122 m tall. The sill and
    the window are ASSUMED to be there, in the prop's frame: the sill board's front edge at x = +0.10 (+X = toward the
    room; the board runs from the wall to that edge, 0.042 m thick below z = 0), the window frame's bottom member
    (front face x = -0.063, up to z = 0.09) and the mullion bars behind it (faces x = -0.1525): vertices that would
    cross them are pushed out, vines are routed over the rounded edge. Yaw the prop with the window. Objects: `pot`
    (slots: 0 glaze, 1 soil), `plant` (slots: 0 leaf, 1 stem), `col_pot` (hidden collider).

    Card: look `crown`; colliders: a cylinder over the pot."""
    K = Kit(name, coll, root, slots)
    rng = random.Random(20260930)
    m_leaf = _mat_pothos_leaf(K)
    m_stem = _mat_stem(K, "stem", "poth_stem_dark", "poth_stem_light", 0.42, streak=30.0)
    m_glaze, m_soil = _mat_blush(K), _mat_soil(K)
    bm = bmesh.new()
    bm_lathe(bm, _POTHOS_PROF, segs=72, mat_ranges=[(0, 0), (_POTHOS_SOIL_FIRST, 1)])
    K.to_obj("pot", bm, [m_glaze, m_soil])
    bm = bmesh.new()
    lay = _layers(bm)
    vines = [(-0.085, -0.040, 0.385, 1, "drop"), (-0.040, -0.020, 0.440, -1, "drop"), (0.002, 0.000, 0.285, 1, "crawl"),
             (0.047, 0.026, 0.405, -1, "drop"), (0.092, 0.045, 0.335, 1, "crawl")]
    for (yh, ys, depth, sd, kind) in vines:
        _pothos_vine(bm, lay, rng, yh + rng.uniform(-0.006, 0.006), ys, depth + rng.uniform(-0.02, 0.02), sd, kind)
    _pothos_crown(bm, lay, rng)
    _pothos_fix(bm)
    K.to_obj("plant", bm, [m_leaf, m_stem])
    col = _pot_collider(K, 0.066, 0.1222)
    return K.card(use={"look": [{"name": "crown", "point": [0.02, 0.0, 0.13]}]}, colliders=[col], origin="sill_top",
                  front="-Y")


# ================================================================= monstera
_MS_B = [0.00, 0.05, 0.12, 0.20, 0.29, 0.39, 0.50, 0.61, 0.72, 0.83]        # midrib start of lateral veins 0..9 (x L)
_MS_A0 = [133, 112, 97, 84, 73, 64, 56, 49, 43, 38]                         # initial heading, deg from +y (tip)
_MS_TAU = [0.9, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]             # heading turn toward the tip, rad per L
_MS_FS = [0.62, 0.55, 0.48, 0.44, 0.42, 0.45, 0.50, 0.58, None, None]        # slit start (fraction of vein length)


@functools.lru_cache(maxsize=None)
def _ms_outline_table():
    """Polar table r_out(theta) (theta measured from +y toward +x about c=(0,0.4)) of the heart-shaped blade margin
    (L=1)."""
    pts = [(0.0, 1.0), (0.05, 0.92), (0.13, 0.83), (0.26, 0.745), (0.385, 0.655), (0.47, 0.54), (0.505, 0.40),
           (0.49, 0.26), (0.435, 0.13), (0.35, 0.02), (0.25, -0.09), (0.14, -0.17), (0.03, -0.20)]
    dense = catmull([(x, y, 0.0) for x, y in pts], 8)
    cx, cy = 0.0, 0.40
    ang = [(math.atan2(p.x - cx, p.y - cy), math.hypot(p.x - cx, p.y - cy)) for p in dense]
    ang.append((math.pi, math.hypot(0.0, -0.215 - cy)))
    N_ = 721
    tab = []
    j = 0
    for k in range(N_):
        th = math.pi * k / (N_ - 1)
        while j + 2 < len(ang) and ang[j + 1][0] < th:
            j += 1
        a0, r0 = ang[j]
        a1, r1 = ang[j + 1]
        t = 0.0 if a1 <= a0 else min(1.0, max(0.0, (th - a0) / (a1 - a0)))
        tab.append(r0 * (1 - t) + r1 * t)
    return tuple(tab)


def _ms_inside(x, y):
    tab = _ms_outline_table()
    dx, dy = abs(x), y - 0.40
    th = math.atan2(dx, dy)
    return math.hypot(dx, dy) < tab[min(720, max(0, int(th / math.pi * 720 + 0.5)))]


def _ms_veins(rng, jit=1.0):
    """Lateral veins as dicts {b, a0, tau, A}: closed-form curved rays from the midrib; A = length to the margin.
    The last entry is the pseudo-vein at the tip (A=0)."""
    out = []
    for k in range(len(_MS_B)):
        v = {"b": _MS_B[k], "a0": math.radians(_MS_A0[k] + rng.uniform(-3, 3) * jit),
             "tau": _MS_TAU[k] * (1 + rng.uniform(-0.12, 0.12) * jit)}
        s_lo, s_hi = 0.0, 1.3
        for _ in range(60):
            sm = 0.5 * (s_lo + s_hi)
            a = v["a0"] - v["tau"] * sm
            x = (math.cos(a) - math.cos(v["a0"])) / v["tau"]
            y = v["b"] - (math.sin(a) - math.sin(v["a0"])) / v["tau"]
            if _ms_inside(x, y):
                s_lo = sm
            else:
                s_hi = sm
        v["A"] = s_lo * 0.985
        out.append(v)
    out.append({"b": 1.0, "a0": 0.0, "tau": 0.0, "A": 0.0})
    return out


def _ms_vp(v, f):
    if v["A"] <= 0.0:
        return 0.0, v["b"]
    s = f * v["A"]
    a = v["a0"] - v["tau"] * s
    return ((math.cos(a) - math.cos(v["a0"])) / v["tau"], v["b"] - (math.sin(a) - math.sin(v["a0"])) / v["tau"])


def _ms_vt(v, f):
    if v["A"] <= 0.0:
        return 0.0, 1.0
    a = v["a0"] - v["tau"] * f * v["A"]
    return math.sin(a), math.cos(a)


def _ms_specs(rng, mature=True):
    specs = []
    for g in range(len(_MS_B)):
        sp = {}
        fs = _MS_FS[g]
        if mature and fs is not None and (g < 2 or rng.random() > 0.08):
            sp["fs"] = fs + rng.uniform(-0.05, 0.05)
            sp["wmax"] = rng.uniform(0.30, 0.44)
            sp["round"] = 0.62
        if mature and 1 <= g <= 5 and rng.random() < 0.85:
            fc = rng.uniform(0.17, 0.27)
            sp["hole"] = (fc, rng.uniform(0.065, 0.095), rng.uniform(0.30, 0.45))
        specs.append(sp)
    return specs


def _ms_gap(sp, f):
    w = 0.0
    fs = sp.get("fs")
    if fs is not None and f > fs:
        t = (f - fs) / (1.0 - fs)
        w = sp["wmax"] * (t * t * (3.0 - 2.0 * t)) ** 0.6
    h = sp.get("hole")
    if h:
        d = (f - h[0]) / h[1]
        if abs(d) < 1.0:
            w = max(w, h[2] * math.sqrt(1.0 - d * d))
    return w


def _ms_half(m2, shared, veins, specs, L, side, mh, rows):
    """Right (side=+1) or mirrored left (-1) half of a blade: one structured grid per gore between lateral veins; the
    gap (slit / fenestra) at a gore's centre is carved out of the grid. Appends to m2 (metres, origin = petiole
    junction)."""
    verts, faces, uvs = m2["verts"], m2["faces"], m2["uv"]
    local = {}
    nG = len(veins) - 1
    cols = 2 * mh + 2

    def get(key, x, y):
        x *= side
        if abs(x) < 1e-9:
            key, d = (round(x * 1e7), round(y * 1e7)), shared
        else:
            d = local
        i = d.get(key)
        if i is None:
            i = len(verts)
            verts.append((x * L, y * L))
            d[key] = i
        return i

    for g in range(nG):
        sp = specs[g]
        v0, v1 = veins[g], veins[g + 1]
        rec = []
        for i, f in enumerate(rows):
            w = _ms_gap(sp, f)
            c0, c1 = _ms_vp(v0, f), _ms_vp(v1, f)
            t0, t1 = _ms_vt(v0, f), _ms_vt(v1, f)
            gw = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
            dx, dy = t0[0] + t1[0], t0[1] + t1[1]
            dl = math.hypot(dx, dy) or 1.0
            dx, dy = dx / dl, dy / dl
            fs = sp.get("fs")
            hw = 0.5 - 0.5 * w
            tipf = 0.0
            if fs is not None and f > fs:
                tipf = sp["round"] * ((f - fs) / (1.0 - fs)) ** 2.0
            row = []
            for c in range(cols):
                u = (c / mh) * hw if c <= mh else 1.0 - ((cols - 1 - c) / mh) * hw
                x = c0[0] * (1 - u) + c1[0] * u
                y = c0[1] * (1 - u) + c1[1] * u
                if tipf:
                    qq = min(1.0, min(u, 1.0 - u) / max(hw, 1e-6))          # 0 at the vein .. 1 at the slit border
                    sh = tipf * hw * gw * (1.0 - math.sqrt(max(0.0, 1.0 - qq * qq)))
                    x -= dx * sh
                    y -= dy * sh
                elif fs is None and f > 0.5:
                    q = min(u, 1.0 - u) * 2.0
                    sh = 0.10 * gw * (1.0 - q * q) * f ** 6
                    x += dx * sh
                    y += dy * sh
                if c == 0:
                    key = ("v", g, i)
                elif c == cols - 1:
                    key = ("v", g + 1, i)
                elif w == 0.0 and c in (mh, mh + 1):
                    key = ("m", g, i)
                else:
                    key = ("c", g, c, i)
                row.append((get(key, x, y), u))
            rec.append((row, gw * L, 0.5 * (v0["A"] + v1["A"]) * L, f))
        for i in range(len(rows) - 1):
            ra, fa, rb, fb = rec[i][0], rec[i], rec[i + 1][0], rec[i + 1]
            for c in list(range(0, mh)) + list(range(mh + 1, 2 * mh + 1)):
                quad = [(ra[c], fa), (ra[c + 1], fa), (rb[c + 1], fb), (rb[c], fb)]
                quad = [quad[0], quad[3], quad[2], quad[1]] if side > 0 else quad
                ids = [q[0][0] for q in quad]
                keep = [k for k in range(4) if ids[k] != ids[(k + 1) % 4]]
                if len(keep) < 3:
                    continue
                faces.append(tuple(ids[k] for k in keep))
                uvs.append([(quad[k][0][1], quad[k][1][3], quad[k][1][1], quad[k][1][2]) for k in keep])


def _ms_leaf2d(rng, L, mature=True, mh=3, nrows=26):
    veins_r = _ms_veins(rng)
    veins_l = _ms_veins(rng)
    bj = [b + (rng.uniform(-0.008, 0.008) if 0 < k < len(_MS_B) else 0.0) for k, b in enumerate(_MS_B)]
    for vs in (veins_r, veins_l):
        for k, v in enumerate(vs[:-1]):
            v["b"] = bj[k]
    rows = [i / nrows for i in range(nrows + 1)]
    m2 = {"verts": [], "faces": [], "uv": []}
    shared = {}
    _ms_half(m2, shared, veins_r, _ms_specs(rng, mature), L, 1, mh, rows)
    _ms_half(m2, shared, veins_l, _ms_specs(rng, mature), L, -1, mh, rows)
    return m2


def _monstera_leaf(bm, lay, rng, O, m0, n_pref, L, seed, age, mature=True, unroll=1.0, kappa=1.4, roll=0.0,
                   pet=None, mats=(0, 1)):
    """Blade (+ underside midrib tube continuing the petiole). pet = (points, r_start): petiole centre-line ending
    at O."""
    T, Nn = _basis(m0, n_pref, roll)
    fr = _Frame(O, T, Nn, kappa=kappa, twist=rng.uniform(-0.6, 0.6))
    kf = rng.uniform(0.05, 0.10)
    kd = rng.uniform(0.07, 0.17)
    wamp = rng.uniform(0.006, 0.014)
    wph = rng.uniform(0, 6.28)
    yo = unroll * 1.12
    m2 = _ms_leaf2d(rng, L, mature)

    def place(x, y):
        xa, sg = abs(x), (1.0 if x >= 0 else -1.0)
        xn = xa / (0.5 * L)
        z = L * (kf * xn - kd * xn * xn) + wamp * L * math.sin(9.0 * y / L + 2.5 * xn + wph) * min(1.0, xn * 2.0)
        z -= 0.05 * L * max(0.0, y / L - 0.55) ** 2 * 2.0
        xr = xa
        if unroll < 1.0:
            S = _smooth(yo - 0.30, yo + 0.12, y / L)
            if S > 1e-4:
                p0 = 0.25 + 0.95 * S
                k = S * 3.0 * math.pi / (0.5 * L)
                ph1 = p0 + k * xa
                xr = (math.sin(ph1) - math.sin(p0)) / k
                z = z * (1 - S) + (math.cos(p0) - math.cos(ph1)) / k
        return fr.pt(sg * xr, y, z)

    _emit(bm, lay, m2, place, mats[0], seed, age)
    # petiole -> midrib tube (centre-line pushed below the crease so only a rib shows on top)
    spine = []
    rs = []
    nspine = 16
    r_blade = 0.0042 * L / 0.4
    for k in range(1, nspine + 1):
        yy = L * 0.985 * k / nspine
        rr = r_blade * (1.0 - 0.85 * k / nspine)
        spine.append(fr.pt(0.0, yy, -0.5 * rr))
        rs.append(rr)
    if pet is not None:
        pts, r_start = pet
        n = len(pts)
        path = list(pts) + spine
        rad = [r_start + (r_blade * 1.0 - r_start) * (i / (n - 1)) ** 0.8 for i in range(n)] + rs
    else:
        path = [fr.pt(0.0, 0.0, -0.5 * r_blade)] + spine
        rad = [r_blade] + rs
    path = _smooth_poly(path, 2)
    bm_tube(bm, path, rad, sides=8, mat=mats[1], cap=("flat", "round"))
    return fr


# name, blade azimuth (deg from +x), blade-base radius, blade-base z, L, blade pitch (deg), droop k, age, unroll, roll
_MS_LEAVES = [
    ("A", -62, 0.20, 0.86, 0.43, 8, 1.5, 0.95, 1.0, 0.0),
    ("B", -18, 0.24, 0.96, 0.41, 16, 1.3, 0.9, 1.0, 0.1),
    ("C", 24, 0.17, 0.90, 0.40, 10, 1.6, 0.85, 1.0, -0.1),
    ("D", -98, 0.13, 0.84, 0.36, 4, 2.0, 1.0, 1.0, 0.0),
    ("E", -42, 0.30, 0.62, 0.38, -8, 1.6, 1.0, 1.0, 0.1),
    ("F", 62, 0.14, 1.00, 0.40, 12, 1.4, 0.8, 1.0, 0.0),
    ("G", 98, 0.10, 0.92, 0.35, 12, 1.4, 0.9, 1.0, 0.0),
    ("H", 2, 0.07, 1.02, 0.33, 34, 1.1, 0.35, 1.0, 0.0),
    ("I", -82, 0.07, 0.96, 0.30, 26, 1.2, 0.3, 1.0, 0.0),
    ("Y1", -30, 0.04, 0.86, 0.28, 70, 0.4, 0.05, 0.22, 0.0),
    ("Y2", 38, 0.07, 0.76, 0.26, 52, 0.7, 0.1, 0.60, 0.0),
    ("S", -112, 0.10, 0.52, 0.24, -4, 1.8, 0.6, 1.0, 0.0),
    ("J", 52, 0.15, 0.68, 0.34, 4, 1.7, 0.9, 1.0, 0.0),
    ("K", 8, 0.25, 0.54, 0.32, -12, 1.5, 0.85, 1.0, 0.0),
    ("M", -75, 0.22, 0.56, 0.30, -6, 1.7, 0.8, 1.0, 0.0),
]


@register("cafe_monstera")
def cafe_monstera(name, coll, root, slots=None):
    """Floor monstera in a cream stoneware planter dipped in sage glaze: a short thick main stem, 15 blades on arching
    petioles (13 mature ones with slits and fenestrae carved into the geometry, 2 rolled young leaves) and 3 aerial roots.

    Frame: origin = centre of the pot base on the floor, +Z up (pot 0.344 m wide, 0.30 m tall; the plant is ~1.19 m high
    and spreads ~0.83 x 1.0 m, its stem leans slightly toward +X and the blades are arranged to be seen from -Y). Objects:
    `pot` (slots: 0 cream/sage glaze, 1 soil), `plant` (slots: 0 leaf, 1 stem, 2 aerial root), `col_pot` (hidden
    collider).

    Card: look `crown`; colliders: a cylinder over the pot."""
    K = Kit(name, coll, root, slots)
    rng = random.Random(777)
    m_leaf = _mat_monstera_leaf(K)
    m_stem = _mat_stem(K, "stem", "mons_stem_dark", "mons_stem_light", 0.38, streak=40.0)
    m_root = K.simple_mat("aerial_root", _c(K, "aerial_root"), 0.72)
    m_pot, m_soil = _mat_cream_sage(K), _mat_soil(K)
    bm = bmesh.new()
    bm_lathe(bm, _MONS_POT_PROF, segs=96, mat_ranges=[(0, 0), (_MONS_SOIL_FIRST, 1)])
    K.to_obj("pot", bm, [m_pot, m_soil])
    bm = bmesh.new()
    lay = _layers(bm)
    view = Vector((0.45, -0.89, 0.0))
    # main stem: short, thick, slightly leaning; petioles leave it at heights that grow with the leaf level
    stem_pts = catmull([(0.0, 0.0, 0.23), (0.010, -0.006, 0.36), (0.016, -0.010, 0.50), (0.014, -0.012, 0.61)], 8)
    bm_tube(bm, stem_pts, lambda t: 0.0185 - 0.0075 * t, sides=10, mat=1, cap=("flat", "round"))

    def stem_at(h):
        for i in range(len(stem_pts) - 1):
            a, b = stem_pts[i], stem_pts[i + 1]
            if a.z <= h <= b.z:
                return a.lerp(b, (h - a.z) / max(b.z - a.z, 1e-9))
        return stem_pts[-1].copy()

    for (_name, az, rb, zb, L, pitch, kap, age, unroll, rollv) in _MS_LEAVES:
        phi = math.radians(az)
        eps = math.radians(pitch)
        yaw = math.radians(rng.uniform(-10, 10))
        mdir = Vector((math.cos(phi + yaw) * math.cos(eps), math.sin(phi + yaw) * math.cos(eps), math.sin(eps)))
        O = Vector((rb * math.cos(phi), rb * math.sin(phi), zb))
        n_pref = _UP * 0.75 + view * 0.45
        h0 = min(0.58, max(0.30, 0.30 + 0.34 * (zb - 0.50) / 0.5))
        p0 = stem_at(h0) + Vector((math.cos(phi), math.sin(phi), 0.0)) * 0.010
        d_end = (Vector((math.cos(phi), math.sin(phi), 0.0)) * 0.55 + _UP * 0.85).normalized()
        chord = (O - p0).length
        p1 = p0 + _UP * (0.35 * (O.z - p0.z)) + Vector((math.cos(phi), math.sin(phi), 0.0)) * (0.05 * chord)
        p2 = O - d_end * (0.30 * chord)
        pet = _bezier(p0, p1, p2, O, 22)
        _monstera_leaf(bm, lay, rng, O, mdir, n_pref, L, rng.random(), age, mature=(unroll >= 1.0 and age > 0.2),
                       unroll=unroll, kappa=kap, roll=rollv, pet=(pet, 0.0085 * (0.8 + 0.4 * (L / 0.4))))
    # aerial roots: tan cords growing out of the stem, wiggling down along it into the soil
    sx, sy = 0.007, -0.004
    for k in range(3):
        a = 0.7 + 2.3 * k + rng.uniform(-0.3, 0.3)
        z0 = rng.uniform(0.38, 0.50)
        ex = rng.uniform(0.0, 0.012)
        ca, sa = math.cos(a), math.sin(a)
        tx, ty = -sa, ca
        pts = []
        for j in range(7):
            t = j / 6.0
            rr = 0.016 + (0.026 + ex) * (t ** 0.8)
            wig = 0.0045 * math.sin(2.6 * math.pi * t + k * 1.7) * (0.3 + t)
            z = z0 + (_MONS_SOIL_Z - 0.014 - z0) * t
            pts.append(Vector((sx + rr * ca + wig * tx, sy + rr * sa + wig * ty, z)))
        pts = catmull(pts, 5)
        bm_tube(bm, pts, lambda t: 0.0040 - 0.0020 * t, sides=6, mat=2, cap=("flat", "round"))
    K.to_obj("plant", bm, [m_leaf, m_stem, m_root])
    col = _pot_collider(K, 0.172, 0.3035, rnd=0.006)
    return K.card(use={"look": [{"name": "crown", "point": [0.15, -0.04, 0.85]}]}, colliders=[col],
                  origin="floor_center", front="-Y")


# ================================================================= haworthia
# The rosette: 39 leaves, one row each (mm in the prop's frame, degrees): spine base, spine middle and spine tip (a
# quadratic curve through them), then the roll of the broad face about the spine (0 = facing up, or toward the rosette
# axis for upright leaves). Longest leaves first; the outer ones arch outward and down, the inner ones stand up around
# the heart of the plant, which sits 1.5 cm off the pot axis.
_HAW_LEAVES = [
    (10.0, -13.2, 116.3, -17.5, -2.4, 154.8, -50.9, 20.8, 180.2, -18),
    (3.4, -9.4, 108.9, -14.1, 17.8, 140.1, -24.6, 59.1, 151.5, 8),
    (15.9, -16.0, 133.0, 36.7, -27.5, 167.4, 49.5, -39.0, 205.4, -13),
    (14.8, -9.9, 132.1, 40.1, -4.8, 164.4, 53.8, 9.8, 200.5, 69),
    (11.4, -13.1, 132.1, 22.8, -39.3, 161.7, 21.2, -68.1, 190.8, -53),
    (14.7, -20.9, 144.4, 25.9, -32.5, 181.2, 32.0, -42.6, 219.6, -83),
    (16.7, -14.6, 138.2, 22.4, -3.6, 176.4, 13.7, 16.1, 210.0, -81),
    (3.6, -12.9, 129.8, -9.8, -31.5, 160.9, -30.0, -43.2, 191.6, -37),
    (13.5, -11.9, 137.0, -4.7, 0.3, 169.1, -21.4, 24.3, 194.4, -8),
    (7.5, -9.1, 127.3, 32.6, 15.2, 145.0, 48.2, 49.8, 153.3, 37),
    (1.4, -12.2, 120.4, 4.6, 19.2, 144.0, 4.2, 57.5, 149.5, 29),
    (15.1, -19.6, 135.7, 31.2, -27.9, 169.0, 39.9, -34.5, 205.3, -45),
    (2.6, -14.8, 126.5, -11.1, -49.8, 130.4, -23.9, -82.5, 143.7, -23),
    (13.4, -14.7, 130.1, 43.0, 4.0, 143.3, 66.7, 30.2, 155.2, 23),
    (4.4, -14.2, 134.6, -12.4, -9.7, 168.7, -41.1, 5.0, 188.5, -3),
    (5.4, -9.8, 129.2, -8.4, 10.8, 159.6, -27.3, 43.6, 165.6, 19),
    (26.8, -16.2, 150.4, 44.8, -13.1, 181.7, 59.8, -8.8, 214.3, 6),
    (7.2, -9.4, 130.2, 8.4, 18.1, 156.4, 2.1, 54.7, 160.0, 42),
    (6.8, -9.3, 133.7, 25.1, 16.6, 153.8, 31.7, 52.7, 150.8, 50),
    (7.6, -20.1, 132.8, 19.7, -53.1, 131.6, 33.3, -82.3, 145.4, -14),
    (14.5, -12.4, 141.1, 7.7, 2.0, 173.3, -12.9, 25.5, 190.6, 46),
    (10.7, -18.9, 123.4, 41.9, -32.1, 132.7, 68.2, -34.1, 155.6, -42),
    (11.6, -15.0, 129.2, 38.8, -24.8, 147.0, 61.2, -34.9, 170.3, -16),
    (23.0, -13.4, 149.9, 25.8, -20.6, 182.3, 29.7, -23.9, 215.2, 55),
    (4.1, -16.9, 131.8, 8.3, -39.9, 158.8, 25.4, -37.0, 190.0, -31),
    (10.4, -12.3, 122.5, 38.2, -22.6, 138.7, 52.3, -21.4, 168.9, -28),
    (9.5, -17.7, 147.1, 12.5, -25.3, 178.2, -2.9, -33.8, 204.9, 42),
    (3.3, -3.7, 121.6, -21.3, -17.5, 128.5, -46.7, -27.5, 137.8, -37),
    (6.2, -13.0, 127.6, 17.2, 10.9, 138.0, 23.7, 36.2, 149.0, 39),
    (4.8, -12.3, 131.1, -7.8, -34.4, 139.6, -19.2, -53.4, 154.5, -30),
    (7.2, -18.4, 138.4, 2.2, -32.3, 162.3, -18.8, -43.6, 176.8, -84),
    (26.7, -16.6, 155.0, 33.8, -15.1, 176.3, 40.2, -13.2, 197.9, 59),
    (11.2, -17.6, 132.4, 23.3, -31.8, 136.9, 32.9, -47.2, 142.9, -29),
    (29.4, -18.7, 163.6, 32.7, -20.7, 177.5, 39.9, -19.6, 189.9, 87),
    (7.2, -23.6, 128.2, 10.6, -34.2, 136.6, 9.9, -41.7, 148.7, -56),
    (16.1, -8.8, 136.7, 21.4, -2.3, 147.8, 31.6, 1.3, 156.9, 5),
    (3.6, -18.7, 148.4, 14.7, -14.4, 146.9, 26.4, -12.6, 149.2, 12),
    (30.7, -20.1, 162.1, 35.6, -20.8, 172.6, 40.5, -21.4, 183.0, -22),
    (-2.0, 15.5, 116.0, -5.5, 18.0, 126.8, -9.4, 23.7, 136.0, -21),
]
HAW_LEAF_WIDTH = (0.0016, 0.19)      # blade width = a + b * length (m), tapering as 1 - s ** 1.4 (s = fraction along)
HAW_LEAF_FLARE = 0.005               # extra width of the clasping base, dying out within the first 10 % of the length
HAW_LEAF_THICK = 0.045               # thickness at the base = this * length; thins toward the tip as (1 - s) ** 0.7

# (radius, height) from the base: foot ring, flared wall, a shoulder where the lip band starts (a faint seam), the lip,
# its rolled rim, then down the inside to the soil level
_HAW_POT_PROF = [(0.0, 0.0014), (0.0345, 0.0014), (0.0385, 0.0004), (0.0415, 0.0004), (0.0450, 0.0019),
                 (0.0470, 0.0036), (0.0487, 0.0077), (0.0517, 0.0235), (0.0569, 0.0568), (0.0618, 0.0898),
                 (0.0640, 0.1045), (0.0647, 0.1072), (0.0653, 0.1077), (0.0660, 0.1081), (0.0665, 0.1090),
                 (0.0670, 0.1117), (0.0678, 0.1169), (0.0685, 0.1221), (0.0688, 0.1248), (0.0685, 0.1259),
                 (0.0674, 0.1269), (0.0658, 0.1275), (0.0643, 0.1272), (0.0631, 0.1265), (0.0626, 0.1256),
                 (0.0624, 0.1234), (0.0624, 0.1192), (0.0622, 0.1149), (0.0621, 0.1126), (0.0617, 0.1117),
                 (0.0608, 0.1110), (0.0599, 0.1103), (0.0585, 0.1097), (0.0521, 0.1092), (0.0, 0.1090)]
_HAW_SAND_PROF = [(0.0, 0.1170), (0.020, 0.1168), (0.040, 0.1163), (0.055, 0.1158), (0.0625, 0.1150)]


# leaf cross-section (across in units of the half width, normal in units of the thickness), starting at the left margin:
# back flank, keel, back flank, right margin, then the flat face. The face side (+) takes 40 % of the thickness, the back
# 60 %. The margins are sharp edges; the perimeter fraction j / 8 is the UV v (keel 0.25, face middle 0.75).
_HAW_SECTION = [(-1.0, 0.0), (-0.6, -0.85), (0.0, -1.0), (0.6, -0.85), (1.0, 0.0), (0.6, 0.9), (0.0, 0.9), (-0.6, 0.9)]


def _haw_leaf(bm, lay, row, seed, K=18):
    """One thick lanceolate leaf swept along its spine (a quadratic curve through the row's base, middle and tip points):
    a flat-faced, thin-edged section (adaxial face up, the back keeled) tapering to a point, K rings. UVMap = (distance
    along the spine in metres, perimeter fraction), UVAlong = (fraction along, length)."""
    J = len(_HAW_SECTION)
    P0, M, P2 = (Vector(row[i:i + 3]) * 0.001 for i in (0, 3, 6))
    C = M * 2.0 - (P0 + P2) * 0.5
    S, T = [], []
    for k in range(K + 1):
        u = k / K
        S.append(P0 * (1 - u) ** 2 + C * (2 * u * (1 - u)) + P2 * (u * u))
        T.append(((C - P0) * (2 * (1 - u)) + (P2 - C) * (2 * u)).normalized())
    cum = _path_cum(S)
    ln = cum[-1]
    # frame: the face normal looks up (or toward the axis for upright leaves), rolled about the spine, then transported
    el, az = math.asin(max(-1.0, min(1.0, T[0].z))), math.atan2(T[0].y, T[0].x)
    Nr = Vector((-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)))
    roll = math.radians(row[9])
    Ns = [Nr * math.cos(roll) + T[0].cross(Nr) * math.sin(roll)]
    for k in range(1, K + 1):
        Ns.append((Ns[-1] - T[k] * Ns[-1].dot(T[k])).normalized())
    wm = HAW_LEAF_WIDTH[0] + HAW_LEAF_WIDTH[1] * ln
    uvl = lay["uv0"], lay["uv1"]
    rings = []
    for k in range(K):
        s = cum[k] / ln
        w = 0.5 * (wm * (1.0 - s ** 1.4) + HAW_LEAF_FLARE * math.exp(-s / 0.05))
        th = HAW_LEAF_THICK * ln * (1.0 - s) ** 0.7
        X = T[k].cross(Ns[k])
        ring = []
        for sx, sz in _HAW_SECTION:
            v = bm.verts.new(S[k] + X * (sx * w) + Ns[k] * (sz * th * (0.4 if sz > 0.0 else 0.6)))
            v[lay["seed"]] = seed
            ring.append(v)
        rings.append(ring)
    apex = bm.verts.new(S[K])
    apex[lay["seed"]] = seed

    def face(vs, us, ps):
        f = bm.faces.new(vs)
        f.smooth = True
        for lp, u, p in zip(f.loops, us, ps):
            lp[uvl[0]].uv = (u, p)
            lp[uvl[1]].uv = (u / ln, ln)

    for k in range(K - 1):
        for j in range(J):
            j2 = (j + 1) % J
            face((rings[k][j], rings[k + 1][j], rings[k + 1][j2], rings[k][j2]), (cum[k], cum[k + 1], cum[k + 1], cum[k]),
                 (j / J, j / J, (j + 1) / J, (j + 1) / J))
        for j in (0, J // 2):
            bm.edges.get((rings[k][j], rings[k + 1][j])).smooth = False
    for j in range(J):
        j2 = (j + 1) % J
        face((rings[K - 1][j], apex, rings[K - 1][j2]), (cum[K - 1], ln, cum[K - 1]), (j / J, (j + 0.5) / J, (j + 1) / J))
    for j in (0, J // 2):
        bm.edges.get((rings[K - 1][j], apex)).smooth = False
    base = bm.verts.new(S[0])
    base[lay["seed"]] = seed
    for j in range(J):
        j2 = (j + 1) % J
        face((base, rings[0][j2], rings[0][j]), (0.0, 0.0, 0.0), ((j + 0.5) / J, (j + 1) / J, j / J))


def _haw_sand_z(r):
    """Height of the sand surface at radius r."""
    for (r0, z0), (r1, z1) in zip(_HAW_SAND_PROF, _HAW_SAND_PROF[1:]):
        if r <= r1:
            return z0 + (z1 - z0) * (r - r0) / (r1 - r0)
    return _HAW_SAND_PROF[-1][1]


def _haw_stone(bm, seed_layer, rng, cx, cy, zb, a, seed):
    """A chunky marble chip resting with its bottom at height zb: an 80-face icosphere with every vertex pushed in or out
    at random, squashed to a semi-axis a with a roughly flat underside, flat-shaded (angular facets)."""
    tmp = bmesh.new()
    bmesh.ops.create_icosphere(tmp, subdivisions=1, radius=1.0)
    b = a * rng.uniform(0.72, 1.0)
    h = a * rng.uniform(0.40, 0.65)
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    tilt = rng.uniform(-0.15, 0.15)
    ct, st = math.cos(tilt), math.sin(tilt)
    vmap = {}
    for v in tmp.verts:
        d = v.co.copy()
        r = rng.uniform(0.80, 1.12)
        x, y, z = d.x * r * a, d.y * r * b, max(d.z * r * h, -0.62 * h)
        x, z = x * ct + z * st, -x * st + z * ct
        nv = bm.verts.new((cx + x * cy_ - y * sy_, cy + x * sy_ + y * cy_, zb + 0.62 * h + z))
        nv[seed_layer] = seed
        vmap[v] = nv
    for f in tmp.faces:
        try:
            nf = bm.faces.new([vmap[v] for v in f.verts])
        except ValueError:
            continue
        nf.material_index = 1
    tmp.free()


def _haw_stones(rng):
    """Stones as (x, y, semi-axis, base height): a thick crescent of stones round the rim toward -X/+Y, thinner elsewhere
    and a bare patch of sand toward +X/-Y; a second pass piles stones where a first-layer stone can carry them."""
    first, piled = [], []
    for layer, n, touch in ((0, 150, 0.72), (1, 40, 0.35)):
        placed = 0
        for _ in range(5000):
            if placed >= n:
                break
            th = rng.uniform(0.0, 2.0 * math.pi)
            if rng.random() > 0.15 + 0.85 * (0.5 + 0.5 * math.cos(th - 2.45)) ** 0.8:
                continue
            a = rng.uniform(0.0036, 0.0072)
            r = 0.0595 * math.sqrt(rng.uniform(0.06, 1.0))
            x, y = r * math.cos(th), r * math.sin(th)
            if r + a > 0.0608:
                continue
            if layer == 0:
                if all(math.hypot(x - px, y - py) > touch * (a + pa) for px, py, pa, pz in first):
                    first.append((x, y, a, _haw_sand_z(r) - 0.0012))
                    placed += 1
                continue
            under = [(pa, pz) for px, py, pa, pz in first if math.hypot(x - px, y - py) < 0.8 * (a + pa)]
            if under and all(math.hypot(x - px, y - py) > touch * (a + pa) for px, py, pa, pz in piled):
                pa, pz = max(under)
                piled.append((x, y, a, pz + 1.05 * pa - 0.0012))
                placed += 1
    return first + piled


def _mat_haw_leaf(K):
    """Matte teal-green leaf: pale chevron bands of short dashes (full on the back, only near the margins on the flat
    face, a plain strip along the keel), pale tip and margin dots, a fine grain. Reads UVMap (u metres along, perimeter
    fraction) and UVAlong (fraction along, length) and the per-leaf `pl_seed`."""
    m, nt, out = K.new_mat("leaf")
    sepu = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, N(nt, "ShaderNodeUVMap", uv_map="UVMap").outputs["UV"], sepu.inputs["Vector"])
    sepl = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, N(nt, "ShaderNodeUVMap", uv_map="UVAlong").outputs["UV"], sepl.inputs["Vector"])
    u, p = sepu.outputs["X"], sepu.outputs["Y"]
    s = sepl.outputs["X"]
    seed = _attr(nt, "pl_seed")
    back = _math(nt, "LESS_THAN", p, 0.5)
    fr = _math(nt, "FRACT", _math(nt, "MULTIPLY", p, 2.0))
    a = _math(nt, "MULTIPLY", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", fr, 0.5)), 2.0)   # 0 on the keel .. 1 at a margin
    # chevron rows of dashes, V pointing toward the base, wobbled by noise
    ph = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", u, _math(nt, "MULTIPLY", a, 0.0018)), 0.0037)
    row = _math(nt, "FLOOR", ph)
    wav = N(nt, "ShaderNodeCombineXYZ")
    link(nt, _math(nt, "MULTIPLY", a, 3.0), wav.inputs["X"])
    link(nt, _math(nt, "MULTIPLY", u, 110.0), wav.inputs["Y"])
    link(nt, _math(nt, "MULTIPLY", seed, 7.0), wav.inputs["Z"])
    wobble = _math(nt, "SUBTRACT", _noise(nt, wav.outputs["Vector"], 1.0, 1.0, 0.5), 0.5)
    ph = _math(nt, "ADD", ph, _math(nt, "MULTIPLY", wobble, 0.6))
    line = _range(nt, _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", _math(nt, "FRACT", ph), 0.5)), 0.06, 0.17, 1.0, 0.0,
                  smooth=True)
    cut = N(nt, "ShaderNodeCombineXYZ")
    link(nt, _math(nt, "MULTIPLY", a, 2.4), cut.inputs["X"])
    link(nt, _math(nt, "MULTIPLY", row, 3.17), cut.inputs["Y"])
    link(nt, _math(nt, "MULTIPLY", seed, 13.0), cut.inputs["Z"])
    dashes = _range(nt, _noise(nt, cut.outputs["Vector"], 1.0, 1.0, 0.5), 0.40, 0.54)
    across = _math(nt, "MULTIPLY", _range(nt, a, 0.10, 0.22), _range(nt, a, 0.88, 1.0, 1.0, 0.0))
    along = _math(nt, "MULTIPLY", _range(nt, s, 0.12, 0.30), _range(nt, s, 0.88, 0.97, 1.0, 0.0))
    edge_only = _range(nt, a, 0.50, 0.85)                     # the adaxial face only carries dashes near the margins
    face_k = _math(nt, "ADD", back, _mul(nt, _math(nt, "SUBTRACT", 1.0, back), edge_only, 0.6))
    bands = _mul(nt, line, dashes, across, along, face_k)
    # pale dots along the margin, pale tip
    tf = _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", _math(nt, "FRACT", _math(nt, "DIVIDE", u, 0.0026)), 0.5))
    teeth = _math(nt, "MULTIPLY", _range(nt, a, 0.86, 0.99), _range(nt, tf, 0.14, 0.32, 1.0, 0.0))
    tip = _range(nt, s, 0.93, 0.995)
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    tone = _range(nt, _math(nt, "ADD", _math(nt, "MULTIPLY", _noise(nt, ob, 22.0, 3.0, 0.5, 0.3), 0.6),
                            _math(nt, "MULTIPLY", _noise(nt, ob, 90.0, 3.0, 0.5), 0.4)), 0.30, 0.70)
    col = _mix(nt, tone, _c(K, "haw_leaf"), _c(K, "haw_leaf_light"))
    col = _mix(nt, _math(nt, "MULTIPLY", bands, 0.75), col, _c(K, "haw_band"))
    col = _mix(nt, _math(nt, "MULTIPLY", teeth, 0.55), col, _c(K, "haw_band"))
    col = _mix(nt, _math(nt, "MULTIPLY", tip, 0.75), col, _c(K, "haw_tip"))
    grain = _noise(nt, ob, 5000.0, 2.0, 0.7)
    nrm = _bump(nt, _math(nt, "ADD", grain, _math(nt, "MULTIPLY", bands, 0.6)), 0.5, 0.0002)
    b = principled(nt, out, **{"Roughness": 0.85, "Specular IOR Level": 0.4})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_haw_pot(K):
    """White glazed pot: satin, faintly mottled, with fine throwing streaks and a faint hairline seam across the lip."""
    m, nt, out = K.new_mat("pot")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    st = N(nt, "ShaderNodeVectorMath", operation="MULTIPLY", inputs={1: (6.0, 6.0, 140.0)})
    link(nt, ob, st.inputs[0])
    streak = _noise(nt, st.outputs["Vector"], 1.0, 2.0, 0.5)
    mott = _range(nt, _noise(nt, ob, 14.0, 3.0, 0.5, 0.2), 0.3, 0.7)
    col = _mix(nt, mott, _c(K, "haw_pot"), _c(K, "haw_pot_cool"))
    col = _mix(nt, _math(nt, "MULTIPLY", streak, 0.12), col, _c(K, "haw_pot_cool"))
    sep = N(nt, "ShaderNodeSeparateXYZ")
    link(nt, ob, sep.inputs["Vector"])
    ang = _math(nt, "ARCTAN2", sep.outputs["Y"], sep.outputs["X"])
    centre = _math(nt, "ADD", -1.808, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 0.1275, sep.outputs["Z"]), 4.67))
    dist = _math(nt, "MULTIPLY", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", ang, centre)), 0.067)   # metres from the line
    seam = _math(nt, "MULTIPLY", _range(nt, dist, 0.0002, 0.0007, 1.0, 0.0, smooth=True),
                 _range(nt, sep.outputs["Z"], 0.1040, 0.1080))
    col = _mix(nt, _math(nt, "MULTIPLY", seam, 0.4), col, _c(K, "haw_pot_dark"))
    nrm = _bump(nt, _math(nt, "ADD", streak, _noise(nt, ob, 320.0, 3.0, 0.6)), 0.12, 0.0008)
    b = principled(nt, out, **{"Roughness": 0.43, "Specular IOR Level": 0.4})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_haw_sand(K):
    """Fine pinkish-taupe sand between the stones."""
    m, nt, out = K.new_mat("sand")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    n1 = _noise(nt, ob, 700.0, 4.0, 0.6)
    n2 = _noise(nt, ob, 40.0, 3.0, 0.5)
    mixn = _math(nt, "ADD", _math(nt, "MULTIPLY", n1, 0.65), _math(nt, "MULTIPLY", n2, 0.35))
    col = _ramp(nt, _range(nt, mixn, 0.30, 0.66), [(0.0, _c(K, "haw_sand_dark")), (1.0, _c(K, "haw_sand_light"))])
    nrm = _bump(nt, n1, 0.5, 0.002)
    b = principled(nt, out, **{"Roughness": 0.88, "Specular IOR Level": 0.3})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


def _mat_haw_stone(K):
    """Pale marble-chip stones, each its own tint (from `pl_seed`): white, warm grey, faint pink."""
    m, nt, out = K.new_mat("stone")
    ob = N(nt, "ShaderNodeTexCoord").outputs["Object"]
    seed = _attr(nt, "pl_seed")
    col = _ramp(nt, seed, [(0.0, _c(K, "haw_stone_a")), (0.5, _c(K, "haw_stone_b")), (1.0, _c(K, "haw_stone_c"))])
    nz = _range(nt, _noise(nt, ob, 160.0, 4.0, 0.6), 0.3, 0.7)
    col = _mix(nt, _math(nt, "MULTIPLY", nz, 0.18), col, _c(K, "haw_stone_b"))
    nrm = _bump(nt, _noise(nt, ob, 400.0, 3.0, 0.6), 0.3, 0.0008)
    b = principled(nt, out, **{"Roughness": 0.8, "Specular IOR Level": 0.3})
    link(nt, col, b.inputs["Base Color"])
    link(nt, nrm, b.inputs["Normal"])
    return m


@register("cafe_haworthia")
def cafe_haworthia(name, coll, root, slots=None):
    """Haworthia (zebra plant) in a small white glazed pot with a pebble-topped soil: a rosette of 39 thick, pointed,
    grey-green leaves with pale chevron bands, built procedurally (the leaf layout is a table of spine curves; the pot is
    a lathe, the stones are seeded chips).

    Frame: origin = centre of the pot base, +Z up (pot 0.138 m wide and 0.1275 m tall, the rosette reaches 0.22 m, its
    heart is 1.5 cm off the pot axis toward -Y; for a sill or a table). Objects: `pot`, `soil` (slots: 0 sand, 1 stones),
    `plant` (leaves), `col_pot` (hidden collider).

    Card: look `crown`; colliders: a cylinder over the pot."""
    K = Kit(name, coll, root, slots)
    rng = random.Random(4041)
    m_leaf, m_pot, m_sand, m_stone = _mat_haw_leaf(K), _mat_haw_pot(K), _mat_haw_sand(K), _mat_haw_stone(K)
    bm = bmesh.new()
    bm_lathe(bm, _HAW_POT_PROF, segs=72)
    K.to_obj("pot", bm, [m_pot])
    bm = bmesh.new()
    seed_layer = bm.verts.layers.float.new("pl_seed")
    bm_lathe(bm, _HAW_SAND_PROF, segs=72, mat=0)
    for x, y, a, zb in _haw_stones(rng):
        _haw_stone(bm, seed_layer, rng, x, y, zb, a, rng.random())
    K.to_obj("soil", bm, [m_sand, m_stone])
    bm = bmesh.new()
    lay = {"seed": bm.verts.layers.float.new("pl_seed"), "uv0": bm.loops.layers.uv.new("UVMap"),
           "uv1": bm.loops.layers.uv.new("UVAlong")}
    for row in _HAW_LEAVES:
        _haw_leaf(bm, lay, row, rng.random())
    K.to_obj("plant", bm, [m_leaf])
    col = _pot_collider(K, 0.0688, 0.1275)
    return K.card(use={"look": [{"name": "crown", "point": [0.012, -0.015, 0.165]}]}, colliders=[col],
                  origin="sill_top", front="-Y")
