"""Wall decor of the cafe: `cafe_fairy_lights` (two runs of string lights) and `cafe_poster` (a paper print taped to the
back wall). Both are built to the layout of the original scene: the wall planes and the anchors below are in the
world-aligned cafe frame, the frame the room set uses as well."""
import math
import os
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import register
from .cafe_kit import Kit, L, N, bm_append, bm_lathe, bm_tube, bm_uv_sphere, catmull, principled, ramp

# ================================================================= layout
# The string lights hang on the room's own geometry, so `cafe_fairy_lights` is built in the world-aligned cafe frame of
# the original layout: root at the origin with yaw 0, floor at z = 0, +Z up, the camera side of the room at -Y.
WINDOW_X = -0.80                    # window wall plane; the room is at x > WINDOW_X
BACK_WALL_Y = 1.05                  # back wall plane; the room is at y < BACK_WALL_Y

WIN_RUN_X = WINDOW_X + 0.06         # window run: the wire hangs 6 cm into the room from the window plane
WIN_RUN_Z = 2.90                    # height of its brass hooks: on the top casing of the window opening (z 2.85-2.925)
WIN_RUN_Y0 = -1.75                  # y of its first (front) hook
WIN_RUN_SPANS = (0.50, 0.46, 0.49, 0.45, 0.50)      # hook spacings along +Y (m): five swags, the last hook at y = +0.65

BACK_PINS_X = (-0.74, 0.14, 1.00)   # back swag: the three pins of its double scallop
BACK_PINS_Z = (2.900, 2.866, 2.890)
BACK_BULB_Y = BACK_WALL_Y - 0.010   # its bulbs hang 1 to 1.6 cm off the wall

WIRE_R = 0.00075                    # wire radius (m)
PIGTAIL_R = 0.0005                  # radius of the little cable a bulb hangs from
WIRE_STEP = 0.008                   # distance between the wire's rings (m)

# The poster: a 0.30 x 0.40 m sheet. Its frame has the origin at the sheet's centre on the wall plane, the wall is the
# plane y = 0 and the sheet's mean plane stands STANDOFF in front of it (-Y is the room), +Z up, +X to the right as seen
# from the room.
POSTER_W, POSTER_H = 0.30, 0.40     # sheet size (m)
STANDOFF = 0.005                    # distance of the sheet's mean plane from the wall (m)

# ================================================================= colours
# Recipes (kwargs of Kit.blend) fitted on Rose Pine Dawn to the colours of the original scene (OKLab error x100 in the
# comments; below 2 is a close match).
WIRE = dict(foam=0.5, base=0.3, gold=0.2, k=1.3)            # pale grey-green cable (1.4)
BULB_WARM = dict(gold=0.85, surface=0.15, k=1.25)           # glass tint of a bulb, warm end (0.1) ...
BULB_PALE = dict(base=0.5, gold=0.5, k=1.3)                 # ... and pale end (0.2)
BULB_RIM = dict(gold=0.75, love=0.25, k=1.3)                # amber at the rim of the glass (1.3)
BULB_BASE = dict(gold=0.7, surface=0.3, k=1.3)              # base colour of the glass (0.7)
CREAM = dict(base=0.95, gold=0.05)                          # warm off-white: paper (0.4) and tape pattern (0.3)
ROSE_TAPE = dict(rose=0.95, overlay=0.05, k=1.05)           # rose ground of the stripe and check tapes (0.1)
GOLD_TAPE = dict(gold=0.9, hl_med=0.1, hue=10, k=1.2)       # amber of the dot and check tapes (0.5)
# Neutral shading factors, not palette colours: a faint warm cast of the whole sheet (multiplied over the print) and
# the lightness of the paper grain.
PAPER_TINT = (1.0, 0.975, 0.935, 1.0)
GRAIN_DARK, GRAIN_LIGHT = (0.955, 0.95, 0.94, 1.0), (1.0, 1.0, 1.0, 1.0)
TAPE_GRAIN_DARK = (0.94, 0.94, 0.94, 1.0)


def _pt(p):
    """A point for the card: [x, y, z] rounded to 0.1 mm."""
    return [round(float(v), 4) for v in p]


# ================================================================= fairy lights: wire paths
def _smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _catenary_k(ratio):
    """Catenary shape factor k with (cosh k - 1) / (2 k) = sag / span."""
    lo, hi = 1e-4, 14.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if (math.cosh(mid) - 1.0) / (2.0 * mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _swag(a, b, sag, off, step=WIRE_STEP):
    """Hanging catenary from a to b (Vectors) with its lowest point `sag` below the chord; `off` (Vector) is a
    sideways bulge applied at mid-span (zero at both clips). Returns the dense polyline incl. both ends."""
    span = (b - a).length
    n = max(10, int(math.ceil(span / step)))
    k = _catenary_k(sag / span)
    ck = math.cosh(k)
    out = []
    for i in range(n + 1):
        t = i / n
        drop = (ck - math.cosh(k * (2.0 * t - 1.0))) / (ck - 1.0)
        out.append(a.lerp(b, t) + Vector((0.0, 0.0, -sag * drop)) + off * math.sin(math.pi * t))
    return out


def _resample(pts, step):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    n = max(2, int(round(cum[-1] / step)))
    out, j = [], 0
    for k in range(n + 1):
        s = cum[-1] * k / n
        while j < len(pts) - 2 and cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        out.append(pts[j].lerp(pts[j + 1], 0.0 if seg < 1e-12 else min(1.0, (s - cum[j]) / seg)))
    return out


def _tail(anchor, rel, step=WIRE_STEP):
    """Loose cable hanging from `anchor` through the relative offsets `rel` (first one (0,0,0)); smooth, even
    spacing."""
    return _resample(catmull([Vector(anchor) + Vector(r) for r in rel], 6), step)


def _chain(parts):
    """Join polylines end to start. marks[0] = 0 and marks[i] = index of the last vertex of parts[i-1]."""
    pts, marks = [], [0]
    for p in parts:
        if pts and (pts[-1] - p[0]).length < 1e-7:
            p = p[1:]
        pts.extend(p)
        marks.append(len(pts) - 1)
    return pts, marks


def _cum(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    return cum


def _point_at(pts, cum, s):
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= s:
            lo = mid
        else:
            hi = mid
    seg = cum[hi] - cum[lo]
    t = 0.0 if seg < 1e-12 else min(1.0, max(0.0, (s - cum[lo]) / seg))
    return pts[lo].lerp(pts[hi], t)


# ================================================================= fairy lights: parts
def _proto_lathe(prof_mm, segs, mat):
    bm = bmesh.new()
    bm_lathe(bm, [(r * 1e-3, z * 1e-3) for r, z in prof_mm], segs=segs, mat=mat)
    return bm


def _protos():
    """Tiny parts built once per run: brass socket cap (material 1) and teardrop bulb (material 2) hanging below the
    origin (z = 0 is the top of the cap, the bulb tip sits at z = -20 mm), brass wall rosette and brass pin head
    (material 1, axis +Z)."""
    cap = _proto_lathe([(3.0, -6.5), (3.25, -6.2), (3.25, -1.5), (3.6, -1.15), (3.6, -0.35), (2.7, 0.0), (0.0, 0.0)],
                       14, 1)
    bulb_prof = [(0.0, 0.0), (1.5, 0.25), (3.0, 0.9), (4.5, 2.0), (5.7, 3.5), (6.4, 5.3), (6.6, 7.2), (6.2, 9.0),
                 (5.2, 10.8), (4.0, 12.3), (3.2, 13.5), (2.9, 14.5)]
    bulb = _proto_lathe([(r, z - 20.0) for r, z in bulb_prof], 16, 2)
    rose = _proto_lathe([(0.0, 0.0), (6.6, 0.0), (7.0, 0.5), (6.8, 1.4), (5.5, 2.3), (3.0, 2.9), (0.0, 3.0)], 16, 1)
    pin = _proto_lathe([(2.7, 0.0), (2.7, 0.3), (2.45, 0.9), (1.9, 1.5), (1.0, 1.85), (0.0, 2.0)], 12, 1)
    return {"cap": cap, "bulb": bulb, "rose": rose, "pin": pin}


def _put(bm, proto, mat4, layer=None, val=None):
    """Append a prototype and (optionally) stamp the per-bulb variation colour on its loops."""
    vs = bm_append(bm, proto, mat4)
    if layer is not None:
        for v in vs:
            for lp in v.link_loops:
                lp[layer] = val
    return vs


def _down(d):
    """Rotation (4x4) taking the prototypes' hanging axis (-Z) onto direction d."""
    return Vector((0.0, 0.0, -1.0)).rotation_difference(d.normalized()).to_matrix().to_4x4()


def _unit(bm, lay, pr, rng, anchor, lean, length, final=False, tangent=None):
    """One hanging bulb: knot on the wire, pigtail down to anchor + lean + (0, 0, -length), brass cap, glass bulb.
    final=True: the wire itself ends in the cap (no pigtail); `tangent` = the wire's direction at its end."""
    var = (rng.random(), rng.random(), 0.0, 1.0)
    if final:
        p3, dr = Vector(anchor), tangent.normalized()
    else:
        a = Vector(anchor)
        p1 = a + Vector((lean.x * 0.15, lean.y * 0.15, -0.30 * length))
        p2 = a + Vector((lean.x * 0.65, lean.y * 0.65, -0.70 * length))
        p3 = a + Vector((lean.x, lean.y, -length))
        dr = (p3 - p2).normalized()
        dr = (dr + Vector((rng.uniform(-0.09, 0.09), rng.uniform(-0.09, 0.09), 0.0))).normalized()
        bm_uv_sphere(bm, 0.0011, loc=a, seg=6, rings=4, mat=0)
        bm_tube(bm, catmull([a, p1, p2, p3, p3 + dr * 0.0016], 4), PIGTAIL_R, sides=5, mat=0, cap=("none", "none"))
    sc = rng.uniform(0.93, 1.07)
    m = Matrix.Translation(p3) @ _down(dr) @ Matrix.Scale(sc, 4)
    _put(bm, pr["cap"], m)
    _put(bm, pr["bulb"], m, lay, var)


def _bulb_arcs(s0, s1, avoid, rng, gap=(0.088, 0.112), margin=0.030):
    """Arc positions of the bulbs along the wire, one every 9-11 cm; a bulb that would land within `margin` of a clip is
    nudged clear of it (back if that keeps >= 8 cm to its neighbour, else forward)."""
    out, s = [], s0 + rng.uniform(0.0, 0.04)
    while s < s1:
        for a in avoid:
            if abs(s - a) < margin:
                back = a - margin
                s = back if (not out or back - out[-1] >= 0.08) else a + margin
        if s < s1:
            out.append(s)
        s += rng.uniform(*gap)
    return out


def _plug(bm, top, yaw):
    """Pebble plug dangling from the cable end `top` (prongs down)."""
    body = _proto_lathe([(0.0, -30.0), (9.0, -30.0), (10.8, -28.3), (11.8, -24.5), (12.0, -18.0), (11.9, -10.0),
                         (11.0, -4.5), (8.5, -1.2), (5.0, 0.0), (3.0, 1.2), (2.5, 4.5), (2.4, 9.0), (0.0, 9.4)],
                        24, 0)
    for x in (-0.0045, 0.0045):
        bm_tube(body, [Vector((x, 0.0, -0.0295)), Vector((x, 0.0, -0.0425))], 0.0011, sides=8, mat=1, cap="flat")
    bm_append(bm, body, Matrix.Translation(Vector(top) - Vector((0.0, 0.0, 0.0075))) @ Matrix.Rotation(yaw, 4, "Z"))
    body.free()


# ================================================================= fairy lights: the two runs
def _run_window(rng, pr, bm, lay):
    """Along the top of the window frame (room side): five catenary swags resting on ball-tipped brass peg hooks
    (rosette on the wall plane, 6.6 cm rod), a loose tail at the front end, a cable drop to a pebble plug at the back
    end. Returns the wire polyline and the indices of the hooks on it."""
    ys = [WIN_RUN_Y0]
    for d in WIN_RUN_SPANS:
        ys.append(ys[-1] + d)
    clips = [Vector((WIN_RUN_X, y, WIN_RUN_Z + rng.uniform(-0.004, 0.004))) for y in ys]
    sags = (0.070, 0.086, 0.058, 0.082, 0.066)
    spans = [_swag(clips[i], clips[i + 1], sags[i], Vector((rng.uniform(-0.004, 0.004), 0.0, 0.0)))
             for i in range(5)]
    front = _tail(clips[0], [(0, 0, 0), (0.001, -0.030, -0.004), (0.003, -0.055, -0.020), (0.006, -0.072, -0.052),
                             (0.009, -0.082, -0.092), (0.011, -0.086, -0.138), (0.012, -0.088, -0.172)])[::-1]
    drop = _tail(clips[5], [(0, 0, 0), (0.0, 0.022, -0.003), (0.002, 0.036, -0.026), (0.004, 0.040, -0.075),
                            (0.004, 0.0405, -0.150), (0.004, 0.0405, -0.2425)])
    pts, marks = _chain([front] + spans + [drop])
    cum = _cum(pts)
    clip_i = marks[1:7]
    bm_tube(bm, pts, WIRE_R, sides=6, mat=0, cap=("none", "none"))
    tip = WIN_RUN_X + 0.0065
    for i in clip_i:                                   # peg hook: rosette on the wall, rod under the wire, ball tip
        p = pts[i]
        zr = p.z - 0.00175
        _put(bm, pr["rose"], Matrix.Translation((WINDOW_X - 0.0002, p.y, zr)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
        bm_tube(bm, [Vector((WINDOW_X + 0.002, p.y, zr)), Vector((0.5 * (WINDOW_X + tip), p.y, zr)),
                     Vector((tip, p.y, zr))], 0.0009, sides=6, mat=1, cap=("none", "none"))
        bm_uv_sphere(bm, 0.0017, loc=(tip, p.y, zr), seg=8, rings=5, mat=1)
    avoid = [cum[i] for i in clip_i]
    for s in _bulb_arcs(0.015, cum[clip_i[-1]] - 0.02, avoid, rng):
        a = _point_at(pts, cum, s)
        lean = Vector((rng.uniform(0.003, 0.009), rng.uniform(-0.004, 0.004), 0.0))
        _unit(bm, lay, pr, rng, a, lean, rng.uniform(0.010, 0.020))
    _unit(bm, lay, pr, rng, pts[0], None, 0.0, final=True, tangent=pts[0] - pts[1])
    _plug(bm, pts[-1], math.radians(24.0))
    return pts, clip_i


def _run_back(rng, pr, bm, lay):
    """Double scallop on the back wall: brass pins, the wire ~ 1 cm off the wall at the belly, a cable drop to a plug
    in the corner at the start, a loose tail with a last bulb at the far end. Returns the wire polyline and the
    indices of the pins on it."""
    wy = BACK_WALL_Y - 0.0012
    pins = [Vector((x, wy, z)) for x, z in zip(BACK_PINS_X, BACK_PINS_Z)]
    sags = (0.230, 0.190)
    spans = [_swag(pins[i], pins[i + 1], sags[i], Vector((0.0, -rng.uniform(0.006, 0.009), 0.0))) for i in range(2)]
    start = _tail(pins[0], [(0, 0, 0), (0.0, -0.0025, -0.030), (0.0015, -0.012, -0.090), (0.0030, -0.024, -0.170),
                            (0.0030, -0.031, -0.2625)])[::-1]
    end = _tail(pins[2], [(0, 0, 0), (0.040, -0.002, -0.006), (0.074, -0.004, -0.032), (0.094, -0.006, -0.078),
                          (0.104, -0.007, -0.125), (0.107, -0.008, -0.160)])
    pts, marks = _chain([start] + spans + [end])
    cum = _cum(pts)
    pin_i = marks[1:4]
    bm_tube(bm, pts, WIRE_R, sides=6, mat=0, cap=("none", "none"))
    for i in pin_i:
        p = pts[i]
        _put(bm, pr["pin"], Matrix.Translation((p.x, BACK_WALL_Y - 0.0003, p.z)) @ Matrix.Rotation(math.pi / 2, 4, "X"))
    avoid = [cum[i] for i in pin_i]
    for s in _bulb_arcs(cum[pin_i[0]] + 0.04, cum[-1] - 0.02, avoid, rng):
        a = _point_at(pts, cum, s)
        lean = Vector((rng.uniform(-0.004, 0.004), (BACK_BULB_Y - rng.uniform(0.0, 0.006)) - a.y, 0.0))
        _unit(bm, lay, pr, rng, a, lean, rng.uniform(0.010, 0.020))
    _unit(bm, lay, pr, rng, pts[-1], None, 0.0, final=True, tangent=pts[-1] - pts[-2])
    _plug(bm, pts[0], math.radians(-31.0))
    return pts, pin_i


# ================================================================= fairy lights: materials
def _wire_mat(K):
    return K.simple_mat("wire", K.blend(**WIRE), rough=0.5, spec=0.5)


def _brass_mat(K):
    return K.simple_mat("brass", K.slot("gold"), rough=0.35, metal=0.6)


def _bulb_mat(K):
    """Warm emissive glass: white-hot core, amber rim (facing falloff). Per-bulb variation comes from the corner colour
    attribute 'bulb_var' (R = warm..pale tint, G = brightness, core strength 6 .. 9.5). Every bulb is scaled by the
    root's `bulb_gain`."""
    m, nt, out = K.new_mat("bulb")
    vc = N(nt, "ShaderNodeVertexColor", (-1300, 0), layer_name="bulb_var")
    sp = N(nt, "ShaderNodeSeparateColor", (-1080, 0))
    L(nt, vc.outputs["Color"], sp.inputs["Color"])
    tint = N(nt, "ShaderNodeMixRGB", (-860, 260),
             inputs={"Color1": K.blend(**BULB_WARM), "Color2": K.blend(**BULB_PALE)})
    L(nt, sp.outputs["Red"], tint.inputs["Fac"])
    geo = N(nt, "ShaderNodeNewGeometry", (-1300, -420))
    dt = N(nt, "ShaderNodeVectorMath", (-1080, -420), operation="DOT_PRODUCT")
    L(nt, geo.outputs["Normal"], dt.inputs[0])
    L(nt, geo.outputs["Incoming"], dt.inputs[1])
    mx = N(nt, "ShaderNodeMath", (-880, -420), operation="MAXIMUM", inputs={1: 0.0})
    L(nt, dt.outputs["Value"], mx.inputs[0])
    fc = N(nt, "ShaderNodeMath", (-680, -420), operation="POWER", inputs={1: 0.8})              # 1 core .. 0 rim
    L(nt, mx.outputs[0], fc.inputs[0])
    rim = N(nt, "ShaderNodeMixRGB", (-480, 260), inputs={"Color1": K.blend(**BULB_RIM)})
    L(nt, fc.outputs[0], rim.inputs["Fac"])
    L(nt, tint.outputs["Color"], rim.inputs["Color2"])
    fall = N(nt, "ShaderNodeMath", (-480, -300), operation="MULTIPLY_ADD", inputs={1: 0.75, 2: 0.25})
    L(nt, fc.outputs[0], fall.inputs[0])
    st = N(nt, "ShaderNodeMath", (-860, -100), operation="MULTIPLY_ADD", inputs={1: 3.5, 2: 6.0})
    L(nt, sp.outputs["Green"], st.inputs[0])
    gain = K.param(nt, "bulb_gain", loc=(-860, -260))
    gain.default_value = K.root["bulb_gain"]
    m1 = N(nt, "ShaderNodeMath", (-260, -120), operation="MULTIPLY")
    L(nt, st.outputs[0], m1.inputs[0])
    L(nt, gain, m1.inputs[1])
    m2 = N(nt, "ShaderNodeMath", (-60, -200), operation="MULTIPLY")
    L(nt, m1.outputs[0], m2.inputs[0])
    L(nt, fall.outputs[0], m2.inputs[1])
    b = principled(nt, out, **{"Base Color": K.blend(**BULB_BASE), "Roughness": 0.25, "Emission Strength": 8.0,
                               "Emission Color": K.blend(**BULB_BASE), "Specular IOR Level": 0.5})
    L(nt, rim.outputs["Color"], b.inputs["Emission Color"])
    L(nt, m2.outputs[0], b.inputs["Emission Strength"])
    return m


@register("cafe_fairy_lights")
def cafe_fairy_lights(name, coll, root, slots=None):
    """The two runs of string lights of the cafe, as objects `<name>_StringLightsWindow` and `<name>_StringLightsBack`
    (wire, brass and warm emissive bulbs: three material slots in that order).

    Frame: the world-aligned cafe frame of the original layout (floor z = 0, +Z up), so the root belongs at the origin
    with yaw 0; the runs hang on the room's geometry (WINDOW_X, BACK_WALL_Y and the anchors at the top of the module).
      window run  x = WINDOW_X + 6 cm, y -1.75 .. +0.65, z ~ 2.9 on the room side of the window frame: five catenary
                  swags on brass peg hooks, a loose tail with a last bulb at the front end, a cable drop to a plug at
                  the back end.
      back swag   a double scallop between pins at x = -0.74 / 0.14 / 1.00 on the back wall, z ~ 2.87-2.90: a cable drop
                  to a plug at the start, a loose tail with a last bulb at the far end.
    Bulbs are teardrop glass on a short pigtail every 9-11 cm; each has its own tint and brightness.

    Root custom properties:
      bulb_gain  (1.0, >= 0) multiplies the emission strength of every bulb. Key it per shot as a twinkle; tiny
                 out-of-focus bulbs need about 15-40 to read as bokeh discs.

    Card: look points window_start / window_end (first and last hook of the window run) and swag (centre of the back
    swag); no colliders."""
    K = Kit(name, coll, root, slots)
    K.prop("bulb_gain", 1.0, 0.0, None, "multiplier of the emission strength of every bulb")
    mats = [_wire_mat(K), _brass_mat(K), _bulb_mat(K)]
    runs = {}
    for base, fn, seed in (("StringLightsWindow", _run_window, 101), ("StringLightsBack", _run_back, 202)):
        rng = random.Random(seed)
        pr = _protos()
        bm = bmesh.new()
        lay = bm.loops.layers.float_color.new("bulb_var")
        runs[base] = fn(rng, pr, bm, lay)
        for p in pr.values():
            p.free()
        K.to_obj(base, bm, mats)
    win_pts, hooks = runs["StringLightsWindow"]
    back_pts, pins = runs["StringLightsBack"]
    scallops = back_pts[pins[0]:pins[-1] + 1]
    swag = (Vector(map(min, *scallops)) + Vector(map(max, *scallops))) / 2
    look = [{"name": "window_start", "point": _pt(win_pts[hooks[0]])},
            {"name": "window_end", "point": _pt(win_pts[hooks[-1]])}, {"name": "swag", "point": _pt(swag)}]
    return K.card(use={"look": look}, origin="cafe_frame", front="-Y")


# ================================================================= poster: paper, bow, print
def _poster_w(un, vn):
    """Paper bow along the sheet normal (+ = toward the viewer, metres) at normalised sheet coordinates un, vn in
    [-1, 1]. Every term carries a factor that vanishes at un = +-1 and vn = +-1 together, so the four corners stay
    put."""
    un = max(-1.0, min(1.0, un))
    vn = max(-1.0, min(1.0, vn))
    return (0.0026 * (un * un - vn * vn)                         # saddle: long sides toward the viewer, ends away
            + 0.0013 * (1.0 - un * un) * (1.0 - vn * vn)         # gentle belly
            + 0.0011 * un * (1.0 - un * un) * (0.6 + 0.4 * vn)   # S-curl across the width, stronger up top
            + 0.0008 * vn * (1.0 - vn * vn) * (0.5 - 0.5 * un))  # slow wave down the height


def _poster_y(x, z):
    """y (prop frame) of the bowed sheet under (x, z) (bow held at its edge value outside the sheet)."""
    return -STANDOFF - _poster_w(x / (POSTER_W * 0.5), z / (POSTER_H * 0.5))


def _corners():
    """Corners of the sheet in the prop frame, as seen from the room: TL, TR, BR, BL. The bow is exactly zero there."""
    hw, hh = POSTER_W * 0.5, POSTER_H * 0.5
    return {"TL": (-hw, -STANDOFF, hh), "TR": (hw, -STANDOFF, hh), "BR": (hw, -STANDOFF, -hh),
            "BL": (-hw, -STANDOFF, -hh)}


def _poster_mesh():
    """24 x 32 quads in the local XY plane (+Z = toward the viewer once rotated X = +90 deg); u along +X, v along +Y."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    nx, ny = 24, 32
    hw, hh = POSTER_W * 0.5, POSTER_H * 0.5
    vs = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            row.append(bm.verts.new((-hw + POSTER_W * i / nx, -hh + POSTER_H * j / ny,
                                     _poster_w(2.0 * i / nx - 1.0, 2.0 * j / ny - 1.0))))
        vs.append(row)
    for j in range(ny):
        for i in range(nx):
            f = bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
            f.smooth = True
            for lp, uv in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (uv[0] / nx, uv[1] / ny)
    return bm


def _paper_mat(K, image):
    """Paper: the print (an RGBA image composited over a warm paper white) with a faint warm tint, paper-fibre grain and
    bump. `image` is a file path or None for plain paper."""
    m, nt, out = K.new_mat("paper")
    paper = K.blend(**CREAM)
    uv = N(nt, "ShaderNodeTexCoord", (-1800, 100))
    tint = N(nt, "ShaderNodeMixRGB", (-450, 320), blend_type="MULTIPLY", inputs={"Fac": 1.0, "Color2": PAPER_TINT})
    if image:
        pr = N(nt, "ShaderNodeMixRGB", (-700, 320), inputs={"Fac": 1.0, "Color1": paper})
        img = bpy.data.images.load(image, check_existing=True)
        if img.packed_file is None:
            img.reload()
        img.colorspace_settings.name = "sRGB"
        tx = N(nt, "ShaderNodeTexImage", (-1150, 560), image=img, interpolation="Linear", extension="CLIP")
        L(nt, uv.outputs["UV"], tx.inputs["Vector"])
        L(nt, tx.outputs["Color"], pr.inputs["Color2"])
        L(nt, tx.outputs["Alpha"], pr.inputs["Fac"])
        L(nt, pr.outputs["Color"], tint.inputs["Color1"])
    else:
        tint.inputs["Color1"].default_value = paper
    mp = N(nt, "ShaderNodeMapping", (-1550, -280), inputs={"Scale": (300.0, 400.0, 1.0)})      # 1 unit = 1 mm
    L(nt, uv.outputs["UV"], mp.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", (-1300, -280), inputs={"Scale": 1.3, "Detail": 7.0, "Roughness": 0.65})
    L(nt, mp.outputs["Vector"], nz.inputs["Vector"])
    gr = ramp(nt, (-1050, -120), [(0.30, GRAIN_DARK), (0.70, GRAIN_LIGHT)])
    L(nt, nz.outputs["Fac"], gr.inputs["Fac"])
    fib = N(nt, "ShaderNodeMixRGB", (-200, 300), blend_type="MULTIPLY", inputs={"Fac": 1.0})
    L(nt, tint.outputs["Color"], fib.inputs["Color1"])
    L(nt, gr.outputs["Color"], fib.inputs["Color2"])
    bp = N(nt, "ShaderNodeBump", (-700, -420), inputs={"Strength": 0.15, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bp.inputs["Height"])
    b = principled(nt, out, **{"Roughness": 0.75, "Specular IOR Level": 0.3, "Sheen Weight": 0.1,
                               "Sheen Roughness": 0.6})
    L(nt, fib.outputs["Color"], b.inputs["Base Color"])
    L(nt, bp.outputs["Normal"], b.inputs["Normal"])
    return m


# ================================================================= poster: washi tape
def _tape_mat(K, base, ground, pattern, kind, a_base, a_alt):
    """Translucent patterned paper tape. The pattern mask (1 = pattern colour) comes from the strip UV, which is in
    metres: u across the 15 mm width, v along the 45 mm length. kinds: stripe (diagonal), dots (diamond lattice),
    check."""
    m, nt, out = K.new_mat(base)
    m.surface_render_method = "BLENDED"
    m.show_transparent_back = False
    uv = N(nt, "ShaderNodeUVMap", (-1700, 0))
    sep = N(nt, "ShaderNodeSeparateXYZ", (-1500, 0))
    L(nt, uv.outputs["UV"], sep.inputs["Vector"])
    if kind == "stripe":
        sm = N(nt, "ShaderNodeMath", (-1300, 100), operation="ADD")
        L(nt, sep.outputs["X"], sm.inputs[0])
        L(nt, sep.outputs["Y"], sm.inputs[1])
        sc = N(nt, "ShaderNodeMath", (-1130, 100), operation="MULTIPLY", inputs={1: 170.0})
        L(nt, sm.outputs[0], sc.inputs[0])
        fr = N(nt, "ShaderNodeMath", (-960, 100), operation="FRACT")
        L(nt, sc.outputs[0], fr.inputs[0])
        rp = ramp(nt, (-780, 100), [(0.0, (0, 0, 0, 1)), (0.60, (0, 0, 0, 1)), (0.66, (1, 1, 1, 1)),
                                    (0.97, (1, 1, 1, 1)), (1.0, (0, 0, 0, 1))])
        L(nt, fr.outputs[0], rp.inputs["Fac"])
        mask = rp.outputs["Color"]
    elif kind == "dots":
        mp = N(nt, "ShaderNodeMapping", (-1300, 100), inputs={"Rotation": (0.0, 0.0, math.radians(45.0)),
                                                              "Scale": (1.0, 1.0, 1.0)})
        L(nt, uv.outputs["UV"], mp.inputs["Vector"])
        vo = N(nt, "ShaderNodeTexVoronoi", (-1100, 100), voronoi_dimensions="2D", feature="F1",
               inputs={"Scale": 250.0, "Randomness": 0.0})
        L(nt, mp.outputs["Vector"], vo.inputs["Vector"])
        rp = ramp(nt, (-880, 100), [(0.0, (1, 1, 1, 1)), (0.25, (1, 1, 1, 1)), (0.31, (0, 0, 0, 1))])
        L(nt, vo.outputs["Distance"], rp.inputs["Fac"])
        mask = rp.outputs["Color"]
    else:                                                 # check
        ck = N(nt, "ShaderNodeTexChecker", (-1100, 100), inputs={"Scale": 330.0})
        L(nt, uv.outputs["UV"], ck.inputs["Vector"])
        mask = ck.outputs["Fac"]
    col = N(nt, "ShaderNodeMixRGB", (-500, 200), inputs={"Color1": K.blend(**ground), "Color2": K.blend(**pattern)})
    L(nt, mask, col.inputs["Fac"])
    al = N(nt, "ShaderNodeMath", (-500, -120), operation="MULTIPLY_ADD", inputs={1: a_alt - a_base, 2: a_base})
    L(nt, mask, al.inputs[0])
    mp2 = N(nt, "ShaderNodeMapping", (-1300, -350), inputs={"Scale": (1000.0, 1000.0, 1.0)})
    L(nt, uv.outputs["UV"], mp2.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", (-1100, -350), inputs={"Scale": 1.0, "Detail": 6.0, "Roughness": 0.7})
    L(nt, mp2.outputs["Vector"], nz.inputs["Vector"])
    gr = ramp(nt, (-880, -300), [(0.3, TAPE_GRAIN_DARK), (0.7, GRAIN_LIGHT)])
    L(nt, nz.outputs["Fac"], gr.inputs["Fac"])
    fib = N(nt, "ShaderNodeMixRGB", (-250, 200), blend_type="MULTIPLY", inputs={"Fac": 1.0})
    L(nt, col.outputs["Color"], fib.inputs["Color1"])
    L(nt, gr.outputs["Color"], fib.inputs["Color2"])
    bp = N(nt, "ShaderNodeBump", (-500, -420), inputs={"Strength": 0.12, "Distance": 0.0002})
    L(nt, nz.outputs["Fac"], bp.inputs["Height"])
    b = principled(nt, out, **{"Roughness": 0.65, "Specular IOR Level": 0.35})
    L(nt, fib.outputs["Color"], b.inputs["Base Color"])
    L(nt, al.outputs[0], b.inputs["Alpha"])
    L(nt, bp.outputs["Normal"], b.inputs["Normal"])
    return m


def _rect_sdf(x, z):
    """Signed distance (m) from (x, z) to the sheet's rectangle (negative inside)."""
    dx = abs(x) - POSTER_W * 0.5
    dz = abs(z) - POSTER_H * 0.5
    return math.hypot(max(dx, 0.0), max(dz, 0.0)) + min(max(dx, dz), 0.0)


def _tape_mesh(edge, ax, ang_deg, zig, rng, W=0.015, Lp=0.012, Lw=0.033):
    """One strip straddling the sheet's top / bottom edge: Lp over the paper, Lw out over the wall, W wide, rotated
    ang_deg (anticlockwise as seen from the front) about its crossing point (ax, z_edge). The strip follows the bowed
    paper (0.5 mm in front of it) and ramps smoothly to the wall (0.5 mm in front of the wall) over 7 mm beyond the
    edge. One short end (zig = 'wall' | 'poster') is cut in a zigzag and peels up 0.7 mm. Returns the bmesh and its
    origin (the crossing point, 0.5 mm in front of the sheet's mean plane)."""
    sig = 1.0 if edge == "top" else -1.0
    z_edge = sig * POSTER_H * 0.5
    th = math.radians(ang_deg)
    e = (math.cos(th), math.sin(th))
    d = (-sig * math.sin(th), sig * math.cos(th))
    nc, nr = 10, 30
    trim = [0.0 if i % 2 == 0 else 0.0013 * rng.uniform(0.85, 1.15) for i in range(nc + 1)]
    org = Vector((ax, -STANDOFF - 0.0005, z_edge))
    y_wall = -0.0005
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    grid = []
    for i in range(nc + 1):
        a = -W * 0.5 + W * i / nc
        s_a = -Lp + (trim[i] if zig == "poster" else 0.0)
        s_b = Lw - (trim[i] if zig == "wall" else 0.0)
        col_v = []
        for j in range(nr + 1):
            f = j / nr
            s = s_a + (s_b - s_a) * f
            x = ax + a * e[0] + s * d[0]
            z = z_edge + a * e[1] + s * d[1]
            yp = _poster_y(x, z) - 0.0005
            y = yp + (y_wall - yp) * _smooth(_rect_sdf(x, z) / 0.007)
            peel = (s_b - s) if zig == "wall" else (s - s_a)
            y -= 0.0007 * _smooth(1.0 - peel / 0.006)
            col_v.append((bm.verts.new((x - org.x, y - org.y, z - org.z)), (a + W * 0.5, s + Lp)))
        grid.append(col_v)
    for i in range(nc):
        for j in range(nr):
            q = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
            if sig < 0:
                q = q[::-1]
            f = bm.faces.new([c[0] for c in q])
            f.smooth = True
            for lp, c in zip(f.loops, q):
                lp[uvl].uv = c[1]
    return bm, org


# name, edge, crossing point on the edge: (anchor on the sheet's width, offset m), angle deg, zigzag end, design
TAPES = (
    ("tape1", "top", ("L", 0.010), 11.0, "wall", "stripe"),
    ("tape2", "top", ("R", -0.010), -7.0, "poster", "dots"),
    ("tape3", "bottom", ("C", 0.075), 4.0, "wall", "check"),
)
# design: (material, ground recipe, pattern recipe, ground alpha, pattern alpha)
TAPE_DESIGNS = {
    "stripe": ("tape_stripe", ROSE_TAPE, CREAM, 0.80, 0.90),
    "dots": ("tape_dots", GOLD_TAPE, CREAM, 0.78, 0.92),
    "check": ("tape_check", ROSE_TAPE, GOLD_TAPE, 0.82, 0.82),
}


@register("cafe_poster")
def cafe_poster(name, coll, root, slots=None):
    """A paper print taped to the back wall: a 0.30 x 0.40 m sheet, slightly bowed and curled, printed on warm paper
    (`<name>_print`), three translucent patterned washi tape strips over its top and bottom edges (`<name>_tape1..3`:
    stripes, dots, check; they follow the paper and peel at one end, no shadow) and four corner empties
    (`<name>_TL / TR / BR / BL`).

    Frame: origin = the sheet's centre on the wall plane. The wall is the plane y = 0, the sheet faces local -Y (the
    room) and its mean plane stands 5 mm in front of the wall (y = -0.005), +Z up, +X to the right as seen from the
    room. On the original back wall: at = [-0.25, 1.05, 1.42], yaw 0.

    slots: palette-slot colour overrides as for every prop, and `image`: the absolute path of the printed image (an
    entry that is not a colour; a PNG with alpha is composited over the paper, its UV runs over the whole sheet, 3:4).
    Without it, or when the file does not exist, the sheet is plain paper.

    Card: anchors TL TR BR BL (the sheet's corners), surface `print` (centre, normal -Y, up +Z, size 0.30 x 0.40), look
    `print` (the sheet's centre); no colliders."""
    K = Kit(name, coll, root, slots)
    image = K.slots.get("image")
    if image:
        image = os.path.expanduser(str(image))
        if not os.path.isfile(image):
            print(f"cafe_poster {name}: image {image!r} not found, plain paper")
            image = None
    paper = _paper_mat(K, image)
    K.to_obj("print", _poster_mesh(), [paper], loc=(0.0, -STANDOFF, 0.0), rot=(math.pi / 2, 0.0, 0.0))
    corners = _corners()
    for tag, p in corners.items():
        K.empty(tag, loc=p, kind="PLAIN_AXES", size=0.012).hide_render = True
    mats = {}
    for design in dict.fromkeys(t[5] for t in TAPES):
        base, ground, pattern, a0, a1 = TAPE_DESIGNS[design]
        mats[design] = _tape_mat(K, base, ground, pattern, design, a0, a1)
    rng = random.Random(303)
    hw = POSTER_W * 0.5
    for tname, edge, (anchor, off), ang, zig, design in TAPES:
        ax = {"L": -hw, "R": hw, "C": 0.0}[anchor] + off
        bm, org = _tape_mesh(edge, ax, ang, zig, rng)
        K.to_obj(tname, bm, [mats[design]], loc=tuple(org)).visible_shadow = False
    centre = (0.0, -STANDOFF, 0.0)
    use = {"anchor": [{"name": tag, "point": list(p)} for tag, p in corners.items()],
           "surface": [{"name": "print", "center": list(centre), "normal": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0],
                        "size": [POSTER_W, POSTER_H]}],
           "look": [{"name": "print", "point": list(centre)}]}
    return K.card(use=use, origin="wall_center", front="-Y")
