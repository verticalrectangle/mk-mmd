"""The cafe table set: bentwood cane chair, terrazzo bistro table, a letter page and a fountain pen. The rest of the
cafe (mug, saucer, iPod, vase, earbuds, plants, fairy lights, poster) lives in cafe_mug.py, cafe_ipod.py,
cafe_plants.py and cafe_deco.py; helpers in cafe_kit.py. All register themselves with @register when imported.

Layout of the original scene (room frame of sets/cafe.py: window plane x = -0.80, back wall y = 1.05; the seated
character faces -Y; the table top is at z = 0.74):

    [[prop]]  name = "chair"   card = "library:cafe_chair"   at = [0, 0, 0]
    [[prop]]  name = "table"   card = "library:cafe_table"   at = [0, -0.48, 0]
    [[prop]]  name = "page"    card = "library:cafe_page"    at = [-0.06, -0.31, 0.7408]   yaw = 188
    [[prop]]  name = "pen"     card = "library:cafe_pen"     parent = "page_pen_rest"      # lying beside the page

Colours come from the project's palette (see cafe_kit.py); `[[prop]] slots = {pine = "#123456"}` overrides single slots.

cafe_chair   origin = floor under the seat centre, seat top z = 0.45, the seated character faces -Y (the backrest is
             at +Y). Card: use.sit [seat], use.feet [floor], colliders: backrest box + seat cylinder (hidden objects).
cafe_table   origin = floor under the table centre, top (diameter 0.64) at z = 0.74. Card: use.rest [top plane,
             four rim edges], use.look [top], collider: the top as a cylinder.
cafe_page    a 160 x 220 mm letter page lying on the table: origin = page centre on the table, local +X = page right as
             read, +Y = toward the top edge, +Z up, so it sits at `at` z = table top + 0.0008. Page millimetres
             (x to the right, y down, from the top-left corner as read) map to local metres with page_to_local() and to
             UVs with page_uv(): u = x_mm / 160, v = 1 - y_mm / 220. The paper material has a mix node named
             "PaperTone" (feeds Base Color) where ink hooks in. Card: use.surface [page], use.rest [page plane, hold
             (where the other hand holds the page down)], use.look [first_line], and the empty "<name>_pen_rest"
             (page-local pose of a pen lying beside the top edge: parent a pen prop to it).
cafe_pen     a fountain pen: origin = the nib tip, local +Z from the nib toward the cap, local +X = nib top / clip
             side. Length 0.14 m. Card: use.grip [barrel: type "pen", length, radius profile, tip, nib_offset], capsule
             collider on the pen's own mesh, use.look [nib, cap]."""
import math

import bmesh
from mathutils import Matrix, Vector

from . import register
from .cafe_kit import Kit, N, L, bm_box, bm_lathe, bm_tube, bm_uv_sphere, catmull, circle_pts, principled, ramp

SEAT_Z = 0.45                      # top of the chair seat
TABLE_Z = 0.74                     # table top height
TABLE_R = 0.32                     # 0.64 m top
TABLE_T = 0.022
PAGE_W_MM, PAGE_H_MM = 160.0, 220.0
PEN_LEN = 0.140

_RING_R = 0.205
_RING_Z = SEAT_Z - 0.016           # ring centre line; ring top = seat top
_BACK_TILT = math.atan2(0.244 - 0.170, 0.853 - 0.450)    # rake of the backrest (rad)


def _back_y(z):
    """Backrest plane: y as a function of height."""
    return 0.170 + (z - 0.45) * math.tan(_BACK_TILT)


def page_to_local(x_mm, y_mm):
    """Page millimetres from the top-left corner as read (x right, y down) -> page-local metres (x right, y up, z 0)."""
    return ((x_mm - PAGE_W_MM / 2.0) * 1e-3, (PAGE_H_MM / 2.0 - y_mm) * 1e-3, 0.0)


def page_uv(x_mm, y_mm):
    return (x_mm / PAGE_W_MM, 1.0 - y_mm / PAGE_H_MM)


# ================================================================= materials
def _wood_mat(K, base, light, dark, tone, lacquer=0.28):
    """Honey-oak bentwood: grain streaks along the tube UV (u = along, v = around, metres), lacquered satin."""
    m, nt, out = K.new_mat(base)
    tc = N(nt, "ShaderNodeTexCoord", (-1500, 0))
    sp = N(nt, "ShaderNodeSeparateXYZ", (-1300, 0))
    L(nt, tc.outputs["UV"], sp.inputs["Vector"])
    mu = N(nt, "ShaderNodeMath", (-1100, 120), operation="MULTIPLY", inputs={1: 3.0})
    mv = N(nt, "ShaderNodeMath", (-1100, -60), operation="MULTIPLY", inputs={1: 520.0})
    L(nt, sp.outputs["X"], mu.inputs[0])
    L(nt, sp.outputs["Y"], mv.inputs[0])
    cb = N(nt, "ShaderNodeCombineXYZ", (-900, 0))
    L(nt, mu.outputs[0], cb.inputs["X"])
    L(nt, mv.outputs[0], cb.inputs["Y"])
    n1 = N(nt, "ShaderNodeTexNoise", (-700, 200), inputs={"Scale": 1.0, "Detail": 5.0, "Roughness": 0.62, "Distortion": 0.35})
    L(nt, cb.outputs[0], n1.inputs["Vector"])
    # slow, long-wavelength tone drift along the object
    mu2 = N(nt, "ShaderNodeMath", (-1100, 300), operation="MULTIPLY", inputs={1: 0.8})
    L(nt, sp.outputs["X"], mu2.inputs[0])
    cb2 = N(nt, "ShaderNodeCombineXYZ", (-900, 320))
    L(nt, mu2.outputs[0], cb2.inputs["X"])
    cb2.inputs["Y"].default_value = 3.1
    n2 = N(nt, "ShaderNodeTexNoise", (-700, 420), inputs={"Scale": 1.0, "Detail": 2.0, "Roughness": 0.5})
    L(nt, cb2.outputs[0], n2.inputs["Vector"])
    # grain contrast: pass noise through a ramp
    r1 = ramp(nt, (-500, 80), [(0.34, light), (0.66, dark)])
    L(nt, n1.outputs["Fac"], r1.inputs["Fac"])
    tint = N(nt, "ShaderNodeMixRGB", (-250, 120), blend_type="MIX", inputs={"Fac": 0.0})
    L(nt, n2.outputs["Fac"], tint.inputs["Fac"])
    L(nt, r1.outputs["Color"], tint.inputs["Color1"])
    tint.inputs["Color2"].default_value = tone
    bump = N(nt, "ShaderNodeBump", (-250, -150), inputs={"Strength": 0.05, "Distance": 0.002})
    L(nt, n1.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, loc=(500, 0), **{"Roughness": 0.42, "Coat Weight": lacquer, "Coat Roughness": 0.25,
                                             "Specular IOR Level": 0.5})
    L(nt, tint.outputs["Color"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _cane_mat(K):
    """Woven cane seat (octagon weave) from four wave-band directions, object coordinates (metres)."""
    m, nt, out = K.new_mat("cane")
    tc = N(nt, "ShaderNodeTexCoord", (-1500, 0))
    strands = []
    for i, (rot, direction) in enumerate(((0.0, "X"), (0.0, "Y"), (math.pi / 4, "X"), (math.pi / 4, "Y"))):
        mp = N(nt, "ShaderNodeMapping", (-1300, 300 - i * 230), inputs={"Scale": (1.0, 1.0, 1.0)})
        mp.inputs["Rotation"].default_value = (0.0, 0.0, rot)
        L(nt, tc.outputs["Object"], mp.inputs["Vector"])
        wv = N(nt, "ShaderNodeTexWave", (-1050, 300 - i * 230), wave_type="BANDS", bands_direction=direction,
               wave_profile="SIN", inputs={"Scale": 46.0, "Distortion": 0.8, "Detail": 1.0, "Detail Scale": 6.0})
        L(nt, mp.outputs[0], wv.inputs["Vector"])
        sm = N(nt, "ShaderNodeMapRange", (-800, 300 - i * 230), inputs={"From Min": 0.74, "From Max": 0.9}, clamp=True)
        L(nt, wv.outputs["Fac"], sm.inputs["Value"])
        strands.append(sm)
    mx = strands[0].outputs[0]
    for i, s in enumerate(strands[1:]):
        mxn = N(nt, "ShaderNodeMath", (-550, 300 - i * 150), operation="MAXIMUM")
        L(nt, mx, mxn.inputs[0])
        L(nt, s.outputs[0], mxn.inputs[1])
        mx = mxn.outputs[0]
    nz = N(nt, "ShaderNodeTexNoise", (-800, -700), inputs={"Scale": 260.0, "Detail": 3.0})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    straw = ramp(nt, (-500, -600), [(0.3, K.blend(gold=.6, surface=.4)), (0.7, K.blend(gold=.65, foam=.35, k=1.3))])
    L(nt, nz.outputs["Fac"], straw.inputs["Fac"])
    mixc = N(nt, "ShaderNodeMixRGB", (-150, -200), blend_type="MIX")
    mixc.inputs["Color1"].default_value = K.blend(gold=.7, foam=.3, k=.7)        # gaps between strands (deeper honey, not dark)
    L(nt, mx, mixc.inputs["Fac"])
    L(nt, straw.outputs["Color"], mixc.inputs["Color2"])
    bump = N(nt, "ShaderNodeBump", (-150, -450), inputs={"Strength": 0.5, "Distance": 0.002})
    L(nt, mx, bump.inputs["Height"])
    b = principled(nt, out, loc=(500, 0), **{"Roughness": 0.55, "Sheen Weight": 0.3})
    L(nt, mixc.outputs["Color"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _terrazzo_mat(K):
    """Cream terrazzo: three Voronoi chip layers (large / medium / specks), pastel rose-gold-iris-pine-foam chips."""
    m, nt, out = K.new_mat("terrazzo")
    tc = N(nt, "ShaderNodeTexCoord", (-2000, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-1800, 500), inputs={"Scale": 22.0, "Detail": 4.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    base = ramp(nt, (-1550, 500), [(0.30, K.blend(overlay=1)), (0.72, K.blend(overlay=.75, gold=.25))])
    L(nt, nz.outputs["Fac"], base.inputs["Fac"])
    cur = base.outputs["Color"]
    cream = K.blend(base=1)
    mixr = lambda c, t: tuple(a * (1 - t) + b * t for a, b in zip(c, cream))
    chip_cols = [mixr(c, 0.04) for c in (K.blend(rose=.9, overlay=.1), K.blend(gold=.9, foam=.1, k=1.15),
                                         K.blend(iris=.9, hl_low=.1), K.blend(foam=.55, pine=.45),
                                         K.blend(muted=.55, overlay=.45), K.blend(foam=.9, base=.1, k=1.25),
                                         K.blend(rose=.85, surface=.15))]
    # (voronoi scale per metre, chance threshold, min margin, margin variation)
    layers = [(300.0, 0.46, 0.10, 0.18), (120.0, 0.54, 0.10, 0.20), (52.0, 0.64, 0.12, 0.22)]
    x = -1300
    for li, (S, thr, m0, m1) in enumerate(layers):
        y = 200 - li * 500
        off_ = N(nt, "ShaderNodeVectorMath", (x - 220, y), operation="ADD",
                 inputs={1: (13.7 * li + 3.1, 5.3 * li + 7.7, 9.1 * li + 1.3)})
        L(nt, tc.outputs["Object"], off_.inputs[0])
        v1 = N(nt, "ShaderNodeTexVoronoi", (x, y), feature="F1", voronoi_dimensions="3D", inputs={"Scale": S, "Randomness": 1.0})
        L(nt, off_.outputs[0], v1.inputs["Vector"])
        v2 = N(nt, "ShaderNodeTexVoronoi", (x, y - 220), feature="DISTANCE_TO_EDGE", voronoi_dimensions="3D",
                inputs={"Scale": S, "Randomness": 1.0})
        L(nt, off_.outputs[0], v2.inputs["Vector"])
        sc_ = N(nt, "ShaderNodeSeparateColor", (x + 220, y))
        L(nt, v1.outputs["Color"], sc_.inputs["Color"])
        keep = N(nt, "ShaderNodeMath", (x + 440, y + 80), operation="GREATER_THAN", inputs={1: thr})
        L(nt, sc_.outputs["Red"], keep.inputs[0])
        marg = N(nt, "ShaderNodeMath", (x + 440, y - 60), operation="MULTIPLY_ADD", inputs={1: m1, 2: m0})
        L(nt, sc_.outputs["Blue"], marg.inputs[0])
        dif = N(nt, "ShaderNodeMath", (x + 640, y - 120), operation="SUBTRACT")
        L(nt, v2.outputs["Distance"], dif.inputs[0])
        L(nt, marg.outputs[0], dif.inputs[1])
        sh = N(nt, "ShaderNodeMath", (x + 820, y - 120), operation="MULTIPLY", use_clamp=True, inputs={1: 18.0})
        L(nt, dif.outputs[0], sh.inputs[0])
        msk = N(nt, "ShaderNodeMath", (x + 1000, y), operation="MULTIPLY")
        L(nt, keep.outputs[0], msk.inputs[0])
        L(nt, sh.outputs[0], msk.inputs[1])
        off = li * 2
        cols = [chip_cols[(k + off) % len(chip_cols)] for k in range(len(chip_cols))]
        pal = ramp(nt, (x + 440, y - 330), [(i / len(cols), c) for i, c in enumerate(cols)], interp="CONSTANT")
        L(nt, sc_.outputs["Green"], pal.inputs["Fac"])
        mx = N(nt, "ShaderNodeMixRGB", (x + 1200, y), blend_type="MIX")
        L(nt, msk.outputs[0], mx.inputs["Fac"])
        L(nt, cur, mx.inputs["Color1"])
        L(nt, pal.outputs["Color"], mx.inputs["Color2"])
        cur = mx.outputs["Color"]
    nb = N(nt, "ShaderNodeTexNoise", (-300, -700), inputs={"Scale": 420.0, "Detail": 2.0})
    L(nt, tc.outputs["Object"], nb.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (200, -400), inputs={"Strength": 0.05, "Distance": 0.0006})
    L(nt, nb.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, loc=(600, 0), **{"Roughness": 0.40, "Coat Weight": 0.06, "Coat Roughness": 0.3,
                                             "Specular IOR Level": 0.4})
    L(nt, cur, b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _paint_mat(K, base_name, color, rough=0.42, coat=0.2):
    """Satin enamel paint with a hint of hand-applied variation."""
    m, nt, out = K.new_mat(base_name)
    tc = N(nt, "ShaderNodeTexCoord", (-900, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-700, 0), inputs={"Scale": 18.0, "Detail": 3.0})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    r = ramp(nt, (-450, 0), [(0.35, tuple(c * 0.96 for c in color[:3]) + (1,)),
                             (0.65, tuple(min(c * 1.05, 1) for c in color[:3]) + (1,))])
    L(nt, nz.outputs["Fac"], r.inputs["Fac"])
    b = principled(nt, out, loc=(300, 0), **{"Roughness": rough, "Coat Weight": coat, "Coat Roughness": 0.2})
    L(nt, r.outputs["Color"], b.inputs["Base Color"])
    return m


def _paper_mat(K):
    m, nt, out = K.new_mat("paper")
    tc = N(nt, "ShaderNodeTexCoord", (-1200, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-1000, 200), inputs={"Scale": 260.0, "Detail": 6.0, "Roughness": 0.7})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    nz2 = N(nt, "ShaderNodeTexNoise", (-1000, -100), inputs={"Scale": 14.0, "Detail": 2.0})
    L(nt, tc.outputs["Object"], nz2.inputs["Vector"])
    base = ramp(nt, (-700, 150), [(0.3, K.blend(surface=.8, gold=.2)), (0.7, K.blend(surface=.7, gold=.3))])
    L(nt, nz2.outputs["Fac"], base.inputs["Fac"])
    # faint toning toward the edges (UV distance from the centre)
    sp = N(nt, "ShaderNodeSeparateXYZ", (-1000, -400))
    L(nt, tc.outputs["UV"], sp.inputs["Vector"])
    ex = N(nt, "ShaderNodeMath", (-800, -330), operation="SUBTRACT", inputs={1: 0.5})
    L(nt, sp.outputs["X"], ex.inputs[0])
    ey = N(nt, "ShaderNodeMath", (-800, -480), operation="SUBTRACT", inputs={1: 0.5})
    L(nt, sp.outputs["Y"], ey.inputs[0])
    ax = N(nt, "ShaderNodeMath", (-640, -330), operation="ABSOLUTE")
    L(nt, ex.outputs[0], ax.inputs[0])
    ay = N(nt, "ShaderNodeMath", (-640, -480), operation="ABSOLUTE")
    L(nt, ey.outputs[0], ay.inputs[0])
    mxe = N(nt, "ShaderNodeMath", (-480, -400), operation="MAXIMUM")
    L(nt, ax.outputs[0], mxe.inputs[0])
    L(nt, ay.outputs[0], mxe.inputs[1])
    edge = N(nt, "ShaderNodeMapRange", (-300, -400), inputs={"From Min": 0.40, "From Max": 0.5, "To Min": 0.0, "To Max": 0.35}, clamp=True)
    L(nt, mxe.outputs[0], edge.inputs["Value"])
    tone = N(nt, "ShaderNodeMixRGB", (-80, 100), blend_type="MIX", name="PaperTone", label="paper colour (feeds Base Color; ink hooks in here)")
    L(nt, edge.outputs[0], tone.inputs["Fac"])
    L(nt, base.outputs["Color"], tone.inputs["Color1"])
    tone.inputs["Color2"].default_value = K.blend(gold=.4, base=.3, foam=.3, k=1.3)
    bump = N(nt, "ShaderNodeBump", (-80, -250), inputs={"Strength": 0.08, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, loc=(300, 0), **{"Roughness": 0.82, "Sheen Weight": 0.15, "Sheen Roughness": 0.5,
                                             "Subsurface Weight": 0.06, "Specular IOR Level": 0.35})
    b.inputs["Subsurface Radius"].default_value = (0.004, 0.003, 0.002)
    b.inputs["Subsurface Scale"].default_value = 0.2
    b.name = "PaperBSDF"
    L(nt, tone.outputs["Color"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _gold_mat(K, base, rough=0.28):
    return K.simple_mat(base, K.blend(gold=.85, foam=.15, k=1.25), rough=rough, metal=1.0, spec=0.6)


# ================================================================= chair
@register("cafe_chair")
def cafe_chair(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    wood = _wood_mat(K, "wood", K.blend(gold=.8, pine=.2), K.blend(gold=.75, love=.25, k=.5),
                     K.blend(gold=.75, text=.25, k=.8))
    cane = _cane_mat(K)
    bm = bmesh.new()
    zc = _RING_Z
    # seat ring (elliptical section: 16 mm tall half-height, 12 mm radial half-width)
    bm_tube(bm, circle_pts(_RING_R, 72, zc), 0.014, sides=14, closed=True, profile=(16 / 14, 12 / 14))
    # woven seat panel, top slightly below the ring top
    bm_lathe(bm, [(0.0, zc - 0.008), (0.19, zc - 0.008), (0.1945, zc - 0.0065), (0.1945, zc + 0.0075),
                  (0.19, zc + 0.0095), (0.0, zc + 0.0095)], segs=72, mat=0, mat_ranges=[(0, 0), (4, 1)])
    # front legs (turned, tapering, slightly splayed)
    for s in (-1, 1):
        pts = catmull([(s * 0.122, -0.150, zc + 0.004), (s * 0.140, -0.180, 0.24), (s * 0.158, -0.210, 0.0)], 10)
        n = len(pts)
        bm_tube(bm, pts, [0.0175 - 0.0065 * (i / (n - 1)) + 0.0018 * math.sin(i / (n - 1) * math.pi * 3) for i in range(n)],
                sides=14, cap=("none", "flat"))

    # rear legs + posts + crest as ONE bent piece (foot -> left post -> crest -> right post -> foot)
    def side(sg):
        return [(sg * 0.184, 0.290, 0.0), (sg * 0.178, 0.258, 0.15), (sg * 0.166, 0.222, 0.30),
                (sg * 0.148, 0.184, 0.42), (sg * 0.145, _back_y(0.46), 0.46), (sg * 0.151, _back_y(0.56), 0.56),
                (sg * 0.160, _back_y(0.66), 0.66), (sg * 0.168, _back_y(0.75), 0.75)]
    left = side(-1)
    right = side(1)
    top = [(-0.158, _back_y(0.815) + 0.004, 0.815), (-0.118, 0.241, 0.842), (-0.062, 0.2435, 0.8515),
           (0.0, 0.244, 0.853), (0.062, 0.2435, 0.8515), (0.118, 0.241, 0.842), (0.158, _back_y(0.815) + 0.004, 0.815)]
    path = catmull(left + top + right[::-1], 10)
    zs = [p.z for p in path]
    rad = [0.0112 + 0.0038 * max(0.0, 1 - abs(z - 0.42) / 0.3) for z in zs]
    bm_tube(bm, path, rad, sides=14, cap=("flat", "flat"))
    # inner ring of the backrest (oval, in the raked plane), tied to the crest and to the seat ring
    zr, rx, rz = 0.69, 0.086, 0.066
    ring = []
    for i in range(48):
        a = 2 * math.pi * i / 48
        z = zr + rz * math.sin(a)
        ring.append(Vector((rx * math.cos(a), _back_y(z) + 0.006, z)))
    bm_tube(bm, ring, 0.0092, sides=12, closed=True, up=(0, 1, 0))
    bm_tube(bm, catmull([(0, _back_y(zr + rz) + 0.006, zr + rz), (0, 0.2395, 0.80), (0, 0.2435, 0.8515)], 6), 0.0088, sides=10,
            cap=("none", "none"))
    bm_tube(bm, catmull([(0, _back_y(zr - rz) + 0.006, zr - rz), (0, 0.196, 0.53), (0, 0.2, zc)], 6), 0.0088, sides=10,
            cap=("none", "none"))
    # floor ring joining the four legs
    zf = 0.20
    fr = [(0.0, -0.204, zf), (0.1465, -0.187, zf), (0.196, 0.040, zf), (0.1744, 0.2438, zf), (0.0, 0.272, zf),
          (-0.1744, 0.2438, zf), (-0.196, 0.040, zf), (-0.1465, -0.187, zf)]
    bm_tube(bm, catmull(fr, 10, closed=True), 0.0085, sides=10, closed=True)
    K.to_obj("body", bm, [wood, cane])
    # hidden collision shapes (plain boxes, not rendered): the backrest and the seat
    bmb = bmesh.new()
    bm_box(bmb, (0.38, 0.030, 0.46), (0, 0, 0))
    col_back = K.collider(K.to_obj("col_back", bmb, [], loc=(0.0, _back_y(0.66), 0.66), rot=(-_BACK_TILT, 0, 0)))
    bms = bmesh.new()
    bm_box(bms, (0.42, 0.42, 0.04), (0, 0, 0))
    col_seat = K.collider(K.to_obj("col_seat", bms, [], loc=(0.0, 0.0, SEAT_Z - 0.02)))
    use = {
        "sit": [{"name": "seat", "hip": [0.0, 0.035, SEAT_Z + 0.075], "facing": [0, -1, 0], "seat_z": SEAT_Z,
                 "floor_z": 0.0, "pelvis_deg": 6.0, "back_deg": 0.0, "back_tilt_deg": round(math.degrees(_BACK_TILT), 1)}],
        "feet": [{"name": "floor", "L": [0.115, -0.33, None], "R": [-0.105, -0.36, None], "floor_z": 0.0}],
        "rest": [{"name": "seat_front", "type": "edge", "a": [-0.15, -0.14, SEAT_Z], "b": [0.15, -0.14, SEAT_Z],
                  "normal": [0, 0, 1]}],
    }
    colliders = [{"type": "box", "object": col_back.name, "rnd": 0.012, "tag": name},
                 {"type": "cylinder", "object": col_seat.name, "half_h": 0.02, "rnd": 0.012, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y")


# ================================================================= table
@register("cafe_table")
def cafe_table(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    terr = _terrazzo_mat(K)
    paint = _paint_mat(K, "paint", K.slot("pine"))
    gold = _gold_mat(K, "brass", 0.34)
    z0, z1 = TABLE_Z - TABLE_T, TABLE_Z
    bm = bmesh.new()
    # top slab: soft chamfer all around; underside painted (mat 1), rim + top terrazzo (mat 0)
    R = TABLE_R
    prof = [(0.0, z0), (R - 0.006, z0), (R - 0.0025, z0 + 0.0008), (R, z0 + 0.004), (R, z1 - 0.0045), (R - 0.0015, z1 - 0.0015),
            (R - 0.0045, z1), (0.0, z1)]
    bm_lathe(bm, prof, segs=96, mat=0, mat_ranges=[(0, 1), (2, 0)])
    # painted base: hub, four arms, four slim tapered splayed legs, brass ferrules.  The legs sit at +-25 / +-155 deg from the
    # table centre: wide of the sitter's knees (the near pair) and leaving the view between the far pair open for her shins.
    zh = z0 - 0.006
    bm_lathe(bm, [(0.0, zh - 0.008), (0.062, zh - 0.008), (0.066, zh - 0.006), (0.066, zh + 0.004), (0.062, zh + 0.006), (0.0, zh + 0.006)],
             segs=48, mat=1)
    for k, ang in enumerate((25.0, 155.0, 205.0, 335.0)):
        a = math.radians(ang)
        d = Vector((math.cos(a), math.sin(a), 0.0))
        top = d * 0.215 + Vector((0, 0, zh - 0.004))
        foot = d * 0.268 + Vector((0, 0, 0.012))
        arm = catmull([d * 0.04 + Vector((0, 0, zh)), d * 0.12 + Vector((0, 0, zh - 0.006)), top + Vector((0, 0, 0.004))], 6)
        bm_tube(bm, arm, 0.0085, sides=10, mat=1, cap=("none", "round"))
        leg = catmull([top, (top + foot) * 0.5 + d * 0.004, foot], 10)
        bm_tube(bm, leg, lambda t: 0.0135 - 0.0038 * t, sides=12, mat=1, cap=("none", "flat"))
        # brass ferrule
        bm_lathe(bm, [(0.0, 0.0), (0.0085, 0.0), (0.0105, 0.0025), (0.0112, 0.018), (0.0095, 0.019), (0.0, 0.019)], segs=24, mat=2,
                 center=(foot.x, foot.y, 0.0))
    K.to_obj("body", bm, [terr, paint, gold])
    # hidden collision shape: the slab as a cylinder about its own axis
    bmc = bmesh.new()
    bmesh.ops.create_cone(bmc, cap_ends=True, segments=48, radius1=R, radius2=R, depth=TABLE_T)
    col_top = K.collider(K.to_obj("col_top", bmc, [], loc=(0.0, 0.0, TABLE_Z - 0.011)))
    rim = R - 0.04
    use = {
        "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, TABLE_Z], "normal": [0, 0, 1], "radius": R},
                 {"name": "rim_near", "type": "edge", "a": [-0.14, rim, TABLE_Z], "b": [0.14, rim, TABLE_Z], "normal": [0, 0, 1]},
                 {"name": "rim_far", "type": "edge", "a": [-0.14, -rim, TABLE_Z], "b": [0.14, -rim, TABLE_Z], "normal": [0, 0, 1]},
                 {"name": "rim_left", "type": "edge", "a": [rim, -0.14, TABLE_Z], "b": [rim, 0.14, TABLE_Z], "normal": [0, 0, 1]},
                 {"name": "rim_right", "type": "edge", "a": [-rim, -0.14, TABLE_Z], "b": [-rim, 0.14, TABLE_Z], "normal": [0, 0, 1]}],
        "look": [{"name": "top", "point": [0.0, 0.0, TABLE_Z]}],
    }
    colliders = [{"type": "cylinder", "object": col_top.name, "R": R, "half_h": 0.011, "rnd": 0.008, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y")


# ================================================================= page
@register("cafe_page")
def cafe_page(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    mat = _paper_mat(K)
    bm = bmesh.new()
    hw, hh = PAGE_W_MM * 0.5e-3, PAGE_H_MM * 0.5e-3
    # local frame: +X page-right, +Y page-up; vertices BL, BR, TR, TL (counter-clockwise seen from +Z)
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    f = bm.faces.new([bm.verts.new((x, y, 0.0)) for x, y in corners])
    uvl = bm.loops.layers.uv.verify()
    for lp, (x, y) in zip(f.loops, corners):
        x_mm = x * 1e3 + PAGE_W_MM / 2
        y_mm = PAGE_H_MM / 2 - y * 1e3
        lp[uvl].uv = page_uv(x_mm, y_mm)
    f.smooth = False
    K.to_obj("sheet", bm, [mat])
    # a pen lying beside the top edge, parallel to it, nib toward the reader's left (the pen's own frame: origin = nib,
    # +Z nib -> cap along the page's +X, +X = clip side up, 3.4 mm above the page)
    nx, ny, _ = page_to_local(14.0, -38.0)
    pen_rot = Matrix(((0, 0, 1), (0, -1, 0), (1, 0, 0))).transposed().to_euler()
    pen_rest = K.empty("pen_rest", loc=(nx, ny, 0.0034), rot=pen_rot, kind="ARROWS", size=0.03)
    hx, hy, _ = page_to_local(8.0, 168.0)
    use = {
        "surface": [{"name": "page", "center": [0.0, 0.0, 0.0], "normal": [0, 0, 1], "up": [0, 1, 0],
                     "size": [PAGE_W_MM * 1e-3, PAGE_H_MM * 1e-3]}],
        "rest": [{"name": "page", "type": "plane", "center": [0.0, 0.0, 0.0], "normal": [0, 0, 1],
                  "size": [PAGE_W_MM * 1e-3, PAGE_H_MM * 1e-3]},
                 {"name": "hold", "type": "plane", "center": [hx + 0.03, hy + 0.05, 0.0], "normal": [0, 0, 1],
                  "radius": 0.05}],
        "look": [{"name": "first_line", "point": list(page_to_local(20.0, 25.0))}],
        "pose": [{"name": "pen_rest", "object": pen_rest.name, "point": [nx, ny, 0.0034]}],
    }
    return K.card(use=use, colliders=[], origin="center", front="-Y",
                  page={"size_mm": [PAGE_W_MM, PAGE_H_MM], "reading": "from the top-left corner as read: x right, y down"})


# ================================================================= pen
def _nib(bm, mat):
    """Gold nib: a scoop cut from a cone whose apex IS the origin (the tip).  Two skins (0.3 mm) + edge faces."""
    Ln, r_n, th_max = 0.0185, 0.0036, math.radians(78)
    nz, nth = 14, 14
    thick = 0.00032
    uvl = bm.loops.layers.uv.verify()
    skins = []
    for k, off in enumerate((0.0, -thick)):
        grid = []
        for i in range(nz + 1):
            s = i / nz
            z = Ln * s
            r = r_n * (1 - (1 - s) ** 2.1) + off          # outer skin radius; inner skin is thinner
            r = max(r, 0.0)
            th_m = th_max * (0.18 + 0.82 * min(1.0, s * 2.2))
            row = []
            for j in range(nth + 1):
                th = -th_m + 2 * th_m * j / nth
                row.append(bm.verts.new((r * math.cos(th), r * math.sin(th), z)))
            grid.append(row)
        skins.append(grid)
    for k, grid in enumerate(skins):
        for i in range(nz):
            for j in range(nth):
                vv = (grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j])
                if k == 1:
                    vv = vv[::-1]
                try:
                    f = bm.faces.new(vv)
                except ValueError:
                    continue
                f.smooth = True
                f.material_index = mat
                for lp in f.loops:
                    lp[uvl].uv = (i / nz, j / nth)
    # side edges + back edge
    for i in range(nz):
        for j in (0, nth):
            vv = (skins[0][i][j], skins[0][i + 1][j], skins[1][i + 1][j], skins[1][i][j])
            if j == nth:
                vv = vv[::-1]
            try:
                f = bm.faces.new(vv)
                f.material_index = mat
            except ValueError:
                pass
    for j in range(nth):
        vv = (skins[0][nz][j], skins[1][nz][j], skins[1][nz][j + 1], skins[0][nz][j + 1])
        try:
            f = bm.faces.new(vv)
            f.material_index = mat
        except ValueError:
            pass


_PEN_BARREL = [          # (r, z) of the grip section, trim rings and barrel (revolved about +Z, the nib at z = 0)
    (0.0, 0.0155),
    (0.0042, 0.0155), (0.0046, 0.0165), (0.0047, 0.0180),            # collar behind the nib
    (0.0050, 0.0181), (0.0050, 0.0195), (0.0047, 0.0196),            # gold front ring
    (0.0044, 0.0215), (0.0042, 0.030), (0.0046, 0.040), (0.0053, 0.0485),   # grip section flaring to the thread
    (0.0057, 0.0492), (0.0057, 0.0502),
    (0.0058, 0.0505), (0.0058, 0.0515), (0.0057, 0.0516),            # gold thread ring
    (0.0057, 0.0525), (0.00585, 0.060), (0.0059, 0.085), (0.0057, 0.100), (0.0052, 0.1075), (0.0, 0.1075),
]
_PEN_CAP = [(0.0, 0.0955), (0.0064, 0.0955), (0.0067, 0.0965), (0.0067, 0.1015), (0.0069, 0.1025), (0.0069, 0.1032),
            (0.0067, 0.1033), (0.0066, 0.1040), (0.0066, 0.130), (0.0063, 0.1375), (0.0053, 0.1400), (0.0, 0.1400)]


def pen_radius_profile():
    """[[distance from the nib, outer radius], ...] of the pen's outline (the grip solver's `radius`)."""
    def nib(z):
        return 0.0036 * (1 - (1 - z / 0.0185) ** 2.1) if z < 0.0185 else 0.0

    def outline(prof, z):
        r = 0.0
        for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
            if z0 <= z <= z1 and z1 > z0:
                r = max(r, r0 + (r1 - r0) * (z - z0) / (z1 - z0))
        return r
    zs = sorted({0.0, 0.0185, 0.0215, 0.030, 0.040, 0.0485, 0.0525, 0.060, 0.085, 0.0955, 0.100, 0.1075, 0.130, 0.1375,
                 PEN_LEN} | {0.004, 0.008, 0.012, 0.0155})
    out = []
    for z in zs:
        r = max(nib(z), outline(sorted(_PEN_BARREL[1:-1], key=lambda p: p[1]), z), outline(_PEN_CAP[1:-1], z))
        out.append([round(z, 4), round(r, 5)])
    return out


@register("cafe_pen")
def cafe_pen(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    pine = K.simple_mat("lacquer", K.slot("pine"), rough=0.22, coat=0.9, spec=0.6)
    pine.node_tree.nodes["Principled BSDF"].inputs["Coat Roughness"].default_value = 0.08
    gold = _gold_mat(K, "gold", 0.24)
    bm = bmesh.new()
    _nib(bm, 1)
    # grip section + trim ring + barrel (mat 0 = lacquer, mat 1 = gold)
    bm_lathe(bm, _PEN_BARREL, segs=32, mat=0, mat_ranges=[(0, 1), (3, 1), (6, 0), (11, 1), (14, 0)])
    # posted cap
    bm_lathe(bm, _PEN_CAP, segs=32, mat=0, mat_ranges=[(0, 0), (4, 1), (6, 0), (10, 1)])
    # clip on the cap (+X side): flat strip standing 1 mm off the cap, bead at the lower end
    clip_pts = [Vector((0.0074, 0.0, 0.1000)), Vector((0.0076, 0.0, 0.1100)), Vector((0.0078, 0.0, 0.1230)), Vector((0.0074, 0.0, 0.1350))]
    cp = catmull(clip_pts, 6)
    bm_tube(bm, cp, 0.0006, sides=8, mat=1, profile=(0.75, 3.3), up=(1, 0, 0), cap=("round", "flat"))
    bm_uv_sphere(bm, 0.0011, loc=(0.0076, 0.0, 0.1000), seg=12, rings=6, mat=1)
    body = K.to_obj("body", bm, [pine, gold])
    use = {
        "grip": [{"name": "barrel", "type": "pen", "length": PEN_LEN, "radius": pen_radius_profile(), "tip": 0.0155,
                  "nib_offset": [0.0, 0.0, 0.0], "object": body.name}],
        "look": [{"name": "nib", "point": [0.0, 0.0, 0.0]}, {"name": "cap", "point": [0.0, 0.0, PEN_LEN]}],
    }
    colliders = [{"type": "capsule", "object": body.name, "a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, PEN_LEN], "R": 0.0075,
                  "tag": name}]
    return K.card(use=use, colliders=colliders, origin="nib_tip", front="+X")
