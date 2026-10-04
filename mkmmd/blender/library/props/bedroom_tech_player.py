"""The radio-cassette player of the bedroom tech props (helper module, registers no builder): geometry, materials and card
pieces of `cassette_player`; its docstring in bedroom_tech.py is the reference.

Everything on the front face is laid out in the front frame (u = +X to the right, v = +Z up, depth outward = -Y) from
the constants below. The front is built as separate plates standing 1.2 mm off the body (so the seams between them are
real grooves), the cassette bay is a pocket cut into the body behind a clear door, the cassette inside is the mesh of
bedroom_tech_tape."""
import math
from types import SimpleNamespace

import bmesh
from mathutils import Matrix, Vector

from . import bedroom_tech_cabinet as CAB
from . import bedroom_tech_geo as G
from . import bedroom_tech_maths as M
from . import bedroom_tech_tape as TAPE
from .cafe_kit import L, N, bm_box, bm_lathe, principled

# ------------------------------------------------------------------ body (m)
BODY_W, BODY_D = 0.340, 0.115
Z0, Z1 = 0.005, 0.160                   # underside of the body (the feet are below) and its top
YF = -BODY_D / 2                        # front wall plane
# the moulded cabinet (bedroom_tech_cabinet): flat front / top / upper back, rounded rims and domed ends, a dished underside
# and a leaning lower back
CAB_L, CAB_DOME = 0.168, 0.004          # flat sides at x = +-0.168, the domed ends reach +-0.172
R_FT, R_BT, R_FB, R_BB, R_END = 0.008, 0.012, 0.012, 0.012, 0.005     # fillets: front-top, back-top, front/back-bottom, ends
DISH, LEAN, KNEE = 0.006, 0.016, 0.080  # underside lift in the middle, how far the lower back leans in, height of its knee

# ------------------------------------------------------------------ front layout
PLATE_W = 0.134                         # width of the control column (x +-0.067)
PLATE_D = 0.0012                        # plates stand this far off the wall
TOP_PLATE = (0.1302, 0.1528)            # z range of the dial row plate
BAY_FRAME = (0.0492, 0.1290)            # z range of the bay door frame
KEY_PLATE = (0.0150, 0.0480)            # z range of the key row plate
BAY_WIN = (-0.0585, 0.0585, 0.0537, 0.1245)      # clear window of the door: x0, x1, z0, z1
BAY_CAV = (-0.0605, 0.0605, 0.0517, 0.1265)      # the pocket behind it
BAY_DEPTH = 0.022
BAY_ZC = 0.5 * (BAY_WIN[2] + BAY_WIN[3])         # 0.0891: centre of the window
CASSETTE_D = 0.0045                              # the cassette's label face is this far behind the front plane
SPK_X, SPK_Z, SPK_R = 0.1145, 0.0845, 0.0455     # speaker centre (+-x), height, bezel outer radius
GRILLE_R = 0.0405                                # radius of the cloth inside the bezel

DIAL_CX, DIAL_Z = -0.0215, 0.5 * (TOP_PLATE[0] + TOP_PLATE[1])      # dial window centre
DIAL_OUT = (0.086, 0.0170)              # bezel outer size
DIAL_IN = (0.078, 0.0108)               # opening = the glowing scale
LED_X = 0.0555
METER_X = (0.0325, 0.0405)              # the two columns of the level meter
KEY_X0, KEY_PITCH, KEY_W, KEY_H = -0.0630, 0.0176, 0.0158, 0.0240      # first key's left edge, pitch, size
KEY_Z = 0.5 * (KEY_PLATE[0] + KEY_PLATE[1])
KEY_D = 0.0072                          # keys stand this far off the wall
KNOB_X = (0.0345, 0.0455, 0.0565)       # three small knobs right of the keys
KNOB_S = 0.78                           # their size relative to the profile in build_body

HANDLE = (0.232, 0.0190)                # handle bar: length (x) and width (y); folded flat in a recess of the top face
RECESS = (0.240, 0.0225, 0.0045)        # x length, y width, depth of that recess
GLOW_STRENGTH = 0.9                     # emission strength at glow = 1 and vertex-colour alpha 1


def dial_text_surface():
    """(centre, size) of the usable part of the dial strip (above the tick marks), in the prop frame (m)."""
    return ([DIAL_CX, round(YF - 0.0022, 4), round(DIAL_Z + 0.0015, 4)], [0.072, 0.0070])


# ================================================================= colours
def colours(K):
    b, s = K.blend, K.slot
    return SimpleNamespace(
        body=b(subtle=.50, text=.40, hl_high=.10), bezel=b(subtle=.40, text=.60), panel=b(overlay=.65, surface=.35),
        pocket=b(surface=.60, base=.40), trough=b(surface=.7, base=.3),
        stripe1=s("pine"), stripe2=s("love"), cloth=b(overlay=.55, muted=.45), rib=b(subtle=.5, text=.5),
        key_rec=s("love"), key_play=s("pine"), key_rew=s("iris"), key_ff=s("iris"), key_stop=s("gold"),
        ink_dark=b(base=.6, surface=.4), ink_light=b(text=.9, rose=.1),
        knob=b(overlay=.6, hl_med=.4), cap=b(subtle=.55, text=.45), point=s("love"),
        handle=b(subtle=.40, text=.60), recess=b(k=.40, subtle=.5, text=.4, hl_high=.1),
        dial=s("gold"), dial_tick=b(base=.55, overlay=.45), pointer=s("love"),
        led=s("love"), seg_lo=s("foam"), seg_mid=s("gold"), seg_hi=s("love"), seg_off=b(overlay=.7, muted=.3),
        glass=b(overlay=.5, hl_med=.5), rubber=b(surface=.6, base=.4), metal=b(subtle=.5, text=.5),
        vent=b(base=.6, surface=.4))


# ================================================================= materials
def make_materials(K, C):
    """dict: plastic (corner colours `vc`), cloth, metal, glass, glow (emission from `vc`, gain = the `glow` property),
    rubber."""
    mats = {}
    # satin ABS plastic: colour per corner, fine grain bump, clear coat
    m, nt, out = K.new_mat("plastic")
    at = N(nt, "ShaderNodeAttribute", (-900, 200), attribute_type="GEOMETRY", attribute_name=G.VC)
    tc = N(nt, "ShaderNodeTexCoord", (-1300, -150))
    nz = N(nt, "ShaderNodeTexNoise", (-1100, -150), inputs={"Scale": 520.0, "Detail": 3.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    nz2 = N(nt, "ShaderNodeTexNoise", (-1100, -400), inputs={"Scale": 7.0, "Detail": 2.0})
    L(nt, tc.outputs["Object"], nz2.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (-700, -150), inputs={"Strength": 0.05, "Distance": 0.0003})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    rr = N(nt, "ShaderNodeMapRange", (-700, -400), inputs={"From Min": 0.3, "From Max": 0.7, "To Min": 0.34, "To Max": 0.46})
    L(nt, nz2.outputs["Fac"], rr.inputs["Value"])
    b = principled(nt, out, **{"Specular IOR Level": 0.5, "Coat Weight": 0.12, "Coat Roughness": 0.25})
    L(nt, at.outputs["Color"], b.inputs["Base Color"])
    L(nt, rr.outputs["Result"], b.inputs["Roughness"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    mats["plastic"] = m
    # speaker cloth: a weave of two crossing bands (bump + brightness), matt
    m, nt, out = K.new_mat("cloth")
    tc = N(nt, "ShaderNodeTexCoord", (-1300, 0))
    ck = N(nt, "ShaderNodeTexChecker", (-1000, 0), inputs={"Scale": 1100.0})
    L(nt, tc.outputs["Object"], ck.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (-700, -150), inputs={"Strength": 0.5, "Distance": 0.0004})
    L(nt, ck.outputs["Fac"], bump.inputs["Height"])
    mx = N(nt, "ShaderNodeMix", (-700, 150), data_type="RGBA", blend_type="MULTIPLY")
    mx.inputs["Factor"].default_value = 0.45
    mx.inputs["A"].default_value = C.cloth
    mx.inputs["B"].default_value = (0.55, 0.55, 0.60, 1.0)
    L(nt, ck.outputs["Fac"], mx.inputs["Factor"])
    b = principled(nt, out, **{"Roughness": 0.9, "Specular IOR Level": 0.15, "Sheen Weight": 0.4, "Sheen Roughness": 0.5})
    L(nt, mx.outputs["Result"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    mats["cloth"] = m
    mats["metal"] = K.simple_mat("metal", C.metal, rough=0.26, metal=0.85, spec=0.6)
    mats["rubber"] = K.simple_mat("rubber", C.rubber, rough=0.85, spec=0.2)
    # clear smoky window
    m, nt, out = K.new_mat("glass")
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = True
    m.show_transparent_back = False
    principled(nt, out, **{"Base Color": C.glass, "Roughness": 0.05, "Alpha": 0.16, "Specular IOR Level": 0.7})
    mats["glass"] = m
    # backlit parts: colour and gain per corner (alpha), scaled by the `glow` property
    m, nt, out = K.new_mat("glow")
    glow = K.param(nt, "glow", loc=(-900, 350))
    at = N(nt, "ShaderNodeAttribute", (-900, 100), attribute_type="GEOMETRY", attribute_name=G.VC)
    g1 = N(nt, "ShaderNodeMath", (-600, 250), operation="MULTIPLY", inputs={1: GLOW_STRENGTH})
    L(nt, at.outputs["Alpha"], g1.inputs[0])
    g2 = N(nt, "ShaderNodeMath", (-400, 250), operation="MULTIPLY")
    L(nt, g1.outputs[0], g2.inputs[0])
    L(nt, glow, g2.inputs[1])
    dim = N(nt, "ShaderNodeMix", (-600, -100), data_type="RGBA", blend_type="MULTIPLY")
    dim.inputs["Factor"].default_value = 1.0
    dim.inputs["B"].default_value = (0.18, 0.18, 0.18, 1.0)
    L(nt, at.outputs["Color"], dim.inputs["A"])
    b = principled(nt, out, **{"Roughness": 0.35, "Specular IOR Level": 0.4})
    L(nt, dim.outputs["Result"], b.inputs["Base Color"])
    L(nt, at.outputs["Color"], b.inputs["Emission Color"])
    L(nt, g2.outputs[0], b.inputs["Emission Strength"])
    mats["glow"] = m
    return mats


# ================================================================= small helpers
_TO_FRONT = G.rot_z_to((0.0, -1.0, 0.0))         # +Z of a lathe -> out of the front face
_TO_X = Matrix.Rotation(math.pi / 2, 3, "Y")      # +Z -> +X


def front_lathe(bm, prof, x, z, col, segs=32, mat=0, y=YF, smooth=True, scale=1.0):
    """Revolve `prof` [(r, h)] (h outward) about an axis out of the front wall through (x, z); `scale` shrinks the part about
    its base point."""
    b = bmesh.new()
    bm_lathe(b, prof, segs=segs, mat=mat, uv=False, smooth=smooth)
    G.paint_all(b, col)
    G.put(bm, b, loc=(x, y, z), rot=_TO_FRONT, scale=scale)


def cyl_x(bm, x0, x1, r, y, z, col=None, mat=0, segs=12):
    b = bmesh.new()
    bm_lathe(b, [(0.0, 0.0), (r, 0.0), (r, x1 - x0), (0.0, x1 - x0)], segs=segs, mat=mat, uv=False)
    if col is not None:
        G.paint_all(b, col)
    G.put(bm, b, loc=(x0, y, z), rot=_TO_X)


def icon(bm, ff, kind, cx, cz, d, col):
    """Transport symbol on a key face (flat shapes): play, rew, ff, stop, rec."""
    if kind == "play":
        G.flat_poly(bm, ff, [(cx - 0.0030, cz - 0.0042), (cx + 0.0046, cz), (cx - 0.0030, cz + 0.0042)], d, 0, col)
    elif kind == "stop":
        G.flat_quad(bm, ff, cx, cz, 0.0074, 0.0074, d, 0, col, r=0.0006)
    elif kind == "rec":
        G.flat_disc(bm, ff, cx, cz, 0.0040, d, 20, 0, col)
    elif kind == "ff":
        for dx in (-0.0034, 0.0030):
            G.flat_poly(bm, ff, [(cx + dx - 0.0027, cz - 0.0038), (cx + dx + 0.0027, cz), (cx + dx - 0.0027, cz + 0.0038)],
                        d, 0, col)
    elif kind == "rew":
        for dx in (-0.0030, 0.0034):
            G.flat_poly(bm, ff, [(cx + dx + 0.0027, cz - 0.0038), (cx + dx - 0.0027, cz), (cx + dx + 0.0027, cz + 0.0038)],
                        d, 0, col)


# ================================================================= body and its plates
def cabinet_mesh():
    """(section polygon (y, z), V, Q, T) of the moulded cabinet (bedroom_tech_cabinet) in the prop frame."""
    P = CAB.section(YF, -YF, Z0, Z1, R_FT, R_BT, R_FB, R_BB, dish=DISH, lean=LEAN, knee=KNEE)
    return (P,) + CAB.cabinet(CAB_L, P, R_END, dome=CAB_DOME)


def build_body(C):
    """Body shell with the bay pocket and the handle recess, front plates, bezels, keys, knobs, stripes, speaker bezels,
    back details and the handle: one mesh, material 0 (plastic, colour per corner)."""
    bm = bmesh.new()
    G.layer(bm)
    ft = G.top_frame(0.0)
    P, V, Q, T = cabinet_mesh()
    G.from_arrays(bm, V, Q, T, col=C.body, mat=0, smooth=True)
    ff = G.front_frame(YF)
    G.pocket(bm, ff, BAY_CAV, BAY_DEPTH, mat=0, col=C.pocket, draft_deg=12.0)
    G.pocket(bm, G.top_frame(Z1), (-RECESS[0] / 2, RECESS[0] / 2, -RECESS[1] / 2, RECESS[1] / 2), RECESS[2], mat=0,
             col=C.recess, draft_deg=15.0)
    lift = CAB.cap_lift(P, R_END, CAB_DOME)

    # accent stripes along the foot of the front
    G.flat_quad(bm, ff, 0.0, 0.0195, 0.3180, 0.0020, 0.0002, 0, C.stripe1)
    G.flat_quad(bm, ff, 0.0, 0.0221, 0.3180, 0.0020, 0.0002, 0, C.stripe2)

    # the three plates of the control column and the frame of the bay door
    def plate(z0, z1, col):
        G.slab(bm, ff, 0.0, 0.5 * (z0 + z1), PLATE_W, z1 - z0, 0.0030, 0.0, PLATE_D, lip=0.0006, ln=2, mat=0, col=col)
    plate(*TOP_PLATE, C.panel)
    plate(*KEY_PLATE, C.panel)
    bx0, bx1, bz0, bz1 = BAY_WIN
    G.frame_ring(bm, ff, (0.0, 0.5 * (BAY_FRAME[0] + BAY_FRAME[1]), PLATE_W, BAY_FRAME[1] - BAY_FRAME[0], 0.0035),
                 (0.0, BAY_ZC, bx1 - bx0 + 0.004, bz1 - bz0 + 0.004, 0.0022), -0.0003, 0.0018, 0.0, lip=0.0008, n=5, ln=3,
                 mat=0, col=C.panel, in_shrink=0.002, closed=False)      # a countersunk opening: 117 x 71 mm at the bottom
    # a lug and a latch on the door frame
    G.flat_quad(bm, ff, 0.0, BAY_FRAME[1] - 0.0022, 0.016, 0.0016, 0.0018 + 0.0001, 0, C.bezel, r=0.0006)
    G.flat_quad(bm, ff, 0.0, BAY_FRAME[0] + 0.0022, 0.040, 0.0010, 0.0018 + 0.0001, 0, C.stripe2, r=0.0004)

    # dial bezel, the meter's dark window and the LED's ring
    dz = DIAL_Z
    G.frame_ring(bm, ff, (DIAL_CX, dz, DIAL_OUT[0], DIAL_OUT[1], 0.0030),
                 (DIAL_CX, dz, DIAL_IN[0] + 0.0024, DIAL_IN[1] + 0.0024, 0.0020), PLATE_D, PLATE_D + 0.0012, PLATE_D,
                 lip=0.0005, n=4, ln=2, mat=0, col=C.bezel, in_shrink=0.0012, closed=False)
    G.flat_quad(bm, ff, 0.5 * (METER_X[0] + METER_X[1]), dz, 0.0170, 0.0180, PLATE_D + 0.0002, 0, C.trough, r=0.0012)
    r_led, d_led = 0.0031, PLATE_D + 0.0004           # a small bead ring round the LED: a half circle of 0.5 mm radius
    arch = [(r_led + 0.0005 * math.cos(math.radians(a)), d_led + 0.0005 * math.sin(math.radians(a)))
            for a in (30, 60, 90, 120, 150)]
    front_lathe(bm, [(0.0036, PLATE_D), (0.0036, d_led)] + arch + [(0.0026, d_led), (0.0026, PLATE_D)], LED_X, dz, C.bezel,
                segs=20)
    # pointer of the dial (a thin red bar standing on the scale)
    px = DIAL_CX - DIAL_IN[0] / 2 + 0.30 * DIAL_IN[0]
    G.slab(bm, ff, px, dz, 0.0013, DIAL_IN[1] - 0.0008, 0.0004, PLATE_D + 0.0002, PLATE_D + 0.0014, lip=0.0003, ln=1, mat=0,
           col=C.pointer)

    # key trough, keys and their symbols
    kx_mid = KEY_X0 + 2 * KEY_PITCH + KEY_W / 2
    G.flat_quad(bm, ff, kx_mid, KEY_Z, 5 * KEY_PITCH + 0.0030, KEY_H + 0.0030, PLATE_D + 0.0002, 0, C.trough, r=0.0016)
    keys = [("rec", C.key_rec, C.ink_light), ("play", C.key_play, C.ink_light), ("rew", C.key_rew, C.ink_dark),
            ("ff", C.key_ff, C.ink_dark), ("stop", C.key_stop, C.ink_dark)]
    for i, (kind, kc, ic) in enumerate(keys):
        cx = KEY_X0 + i * KEY_PITCH + KEY_W / 2
        G.slab(bm, ff, cx, KEY_Z, KEY_W, KEY_H, 0.0022, PLATE_D, KEY_D, lip=0.0018, ln=3, mat=0, col=kc)
        icon(bm, ff, kind, cx, KEY_Z, KEY_D + 0.0001, ic)
    # knobs: skirt, cap, pointer (profiles from the front wall's plate face; scaled to KNOB_S)
    for kx in KNOB_X:
        k0 = (YF - PLATE_D)
        front_lathe(bm, [(0.0, 0.0), (0.0066, 0.0), (0.0066, 0.0026), (0.0058, 0.0032), (0.0058, 0.0056), (0.0, 0.0056)],
                    kx, KEY_Z, C.knob, segs=24, y=k0, scale=KNOB_S)
        front_lathe(bm, [(0.0, 0.0056), (0.0052, 0.0056), (0.0050, 0.0066), (0.0, 0.0068)], kx, KEY_Z, C.cap, segs=24, y=k0,
                    scale=KNOB_S)
        G.flat_quad(bm, ff, kx, KEY_Z + 0.0027 * KNOB_S * 1.1, 0.0009, 0.0034, PLATE_D + 0.0067 * KNOB_S, 0, C.point)
    # scale dots on an arc above each knob
    for kx in KNOB_X:
        for k in range(5):
            ang = math.radians(-60 + 30 * k)
            G.flat_disc(bm, ff, kx + 0.0074 * math.sin(ang), KEY_Z + 0.0074 * math.cos(ang), 0.00036, PLATE_D + 0.0002, 8, 0,
                        C.ink_light)

    # speaker bezels
    for sgn in (-1, 1):
        front_lathe(bm, [(SPK_R, 0.0), (SPK_R, 0.0012), (SPK_R - 0.0009, 0.0020), (GRILLE_R + 0.0006, 0.0020),
                         (GRILLE_R, 0.0017), (GRILLE_R, 0.0)], sgn * SPK_X, SPK_Z, C.bezel, segs=72)
    # back: vents on the vertical upper part; the battery door, rating label and DC socket on the leaning lower part
    fb = G.back_frame(-YF)
    for i in range(14):
        G.flat_quad(bm, fb, -0.0780 + i * 0.0120, 0.1175, 0.0035, 0.0140, 0.0006, 0, C.vent, r=0.0012)
    z_lo, z_kn = Z0 + R_BB, Z0 + KNEE
    up = Vector((0.0, LEAN, z_kn - z_lo)).normalized()
    fs = G.Frame((0.0, -YF - 0.5 * LEAN, 0.5 * (z_lo + z_kn)), (-1, 0, 0), up, Vector((0.0, up.z, -up.y)))
    G.flat_quad(bm, fs, 0.0, 0.0, 0.1700, 0.0500, 0.0006, 0, C.panel, r=0.0040)              # battery door
    G.flat_quad(bm, fs, 0.0, 0.0, 0.1580, 0.0380, 0.0010, 0, C.body, r=0.0030)
    G.flat_quad(bm, fs, 0.1250, 0.0, 0.0400, 0.0240, 0.0006, 0, C.bezel, r=0.0020)           # rating label
    G.flat_quad(bm, fs, -0.1300, 0.0, 0.0160, 0.0160, 0.0006, 0, C.vent, r=0.0050)           # DC socket

    # sides, built on the end planes and carried onto the domed ends: a seam; on the right a headphone socket and three slide
    # switches, on the left vent slots
    for sign in (1, -1):
        sb = bmesh.new()
        G.layer(sb)
        fr = G.Frame((sign * CAB_L, 0.0, 0.0), (0, sign, 0), (0, 0, 1), (sign, 0, 0))
        G.ribbon(sb, fr, [(-0.043 + 0.086 * i / 12, 0.0950) for i in range(13)], 0.0009, 0.0002, 0, C.pocket)
        if sign > 0:
            jk = bmesh.new()
            bm_lathe(jk, [(0.0058, 0.0), (0.0058, 0.0008), (0.0050, 0.0014), (0.0034, 0.0014), (0.0034, 0.0)], segs=24, mat=0,
                     uv=False)
            G.paint_all(jk, C.bezel)
            G.put(sb, jk, loc=(CAB_L, -0.0150, 0.0450), rot=G.rot_z_to((1.0, 0.0, 0.0)))
            G.flat_disc(sb, fr, -0.0150, 0.0450, 0.0030, 0.0004, 16, 0, C.vent)
            for z, cap in ((0.0680, C.key_play), (0.0820, C.key_rec), (0.0960, C.key_stop)):
                G.flat_quad(sb, fr, 0.0090, z, 0.0200, 0.0052, 0.0003, 0, C.trough, r=0.0024)
                G.slab(sb, fr, 0.0090 + (0.0045 if cap is C.key_rec else -0.0045), z, 0.0090, 0.0042, 0.0014, 0.0003, 0.0015,
                       lip=0.0005, ln=2, mat=0, col=cap)
        else:
            for k in range(7):
                G.ribbon(sb, fr, [(-0.023 + 0.046 * i / 6, 0.0420 + k * 0.0072) for i in range(7)], 0.0026, 0.0003, 0, C.vent)
        G.conform(sb.verts, lambda co, s=sign: Vector((s * lift(co.y, co.z), 0.0, 0.0)))
        G.put(bm, sb)

    # folded handle in its recess
    hb = bmesh.new()
    G.layer(hb)
    G.slab(hb, ft, 0.0, 0.0, HANDLE[0], HANDLE[1], 0.0045, Z1 - RECESS[2] + 0.0006, Z1 - 0.0003, lip=0.0012, ln=3, mat=0,
           col=C.handle, base=True)
    G.put(bm, hb)
    ab = bmesh.new()                                      # hinge block of the folded antenna: a rounded bar along X at the back
    bm_lathe(ab, [(0.0, 0.0), (0.0026, 0.0003), (0.0038, 0.0020), (0.0040, 0.0045), (0.0040, 0.0195), (0.0038, 0.0220),
                  (0.0026, 0.0237), (0.0, 0.0240)], segs=16, mat=0, uv=False)
    G.paint_all(ab, C.knob)
    G.put(bm, ab, loc=(0.1310, -YF + 0.0005, 0.1385), rot=_TO_X)
    G.soften(bm, 0.0015)
    return bm


# ================================================================= glass, glow, grilles, feet, antenna
def build_glass(C):
    """The clear window of the bay door and the glass of the dial (material 0)."""
    bm = bmesh.new()
    ff = G.front_frame(YF)
    bx0, bx1, bz0, bz1 = BAY_WIN
    G.flat_quad(bm, ff, 0.0, BAY_ZC, bx1 - bx0, bz1 - bz0, -0.0008, 0, None, r=0.0020)
    G.flat_quad(bm, ff, DIAL_CX, DIAL_Z, DIAL_IN[0], DIAL_IN[1], PLATE_D + 0.0016, 0, None, r=0.0014)
    return bm


def _glow_plate(bm, ff, cx, cz, w, h, d, col, nx=24, ny=3):
    """The backlit scale: a grid of quads whose corner alpha (the emission gain) falls off toward the ends and the edges, as a
    light box behind a dial does."""
    uvl = G.layer(bm)
    rows = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            u, v = 2.0 * i / nx - 1.0, 2.0 * j / ny - 1.0
            gain = (1.0 - 0.42 * u ** 2) * (1.0 - 0.18 * v ** 2)
            row.append((bm.verts.new(ff.pt(cx + u * w / 2, cz + v * h / 2, d)), (*col[:3], gain)))
        rows.append(row)
    for j in range(ny):
        for i in range(nx):
            f = G.make_face(bm, (rows[j][i][0], rows[j][i + 1][0], rows[j + 1][i + 1][0], rows[j + 1][i][0]), ff.d, 0, False)
            for lp, (_, c) in zip(f.loops, (rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i])):
                lp[uvl] = c


def build_glow(C):
    """Backlit dial scale and its tick marks, the LED and the level meter (material 0 glow: colour per corner, alpha =
    relative emission)."""
    bm = bmesh.new()
    G.layer(bm)
    ff = G.front_frame(YF)
    dz = DIAL_Z
    _glow_plate(bm, ff, DIAL_CX, dz, DIAL_IN[0], DIAL_IN[1], PLATE_D + 0.0004, C.dial)
    span = DIAL_IN[0] - 0.0080
    for x, kind in M.ticks(41, span):
        h = (0.0020, 0.0028, 0.0036)[kind]
        G.flat_quad(bm, ff, DIAL_CX + x, dz - DIAL_IN[1] / 2 + 0.0004 + h / 2, 0.00040, h, PLATE_D + 0.0006, 0,
                    (*C.dial_tick[:3], 0.0))
    # a few station marks: dashes in two groups under the strip's centre line (no text)
    # LED
    b = bmesh.new()
    bm_lathe(b, [(0.0022, 0.0), (0.0022, 0.0006), (0.0016, 0.0013), (0.0007, 0.0017), (0.0, 0.0018)], segs=16, mat=0, uv=False)
    G.paint_all(b, (*C.led[:3], 1.5))
    G.put(bm, b, loc=(LED_X, YF - PLATE_D, dz), rot=_TO_FRONT)
    # level meter: two columns of six segments, the lower ones lit
    lit = (5, 4)
    cols = [C.seg_lo, C.seg_lo, C.seg_lo, C.seg_mid, C.seg_mid, C.seg_hi]
    for ci, mx in enumerate(METER_X):
        for k in range(6):
            on = k < lit[ci]
            col = cols[k] if on else C.seg_off
            G.flat_quad(bm, ff, mx, dz - 0.0057 + k * 0.0023, 0.0050, 0.0016, PLATE_D + 0.0004, 0,
                        (*col[:3], 1.0 if on else 0.05), r=0.0004)
    return bm


def build_grilles(C):
    """Both speaker grilles: cloth (material 0), concentric ribs and the centre dome (1 metal)."""
    bm = bmesh.new()
    for sgn in (-1, 1):
        cl = bmesh.new()
        bm_lathe(cl, [(GRILLE_R, 0.0004), (GRILLE_R - 0.0004, 0.0006), (0.0, 0.0010)], segs=72, mat=0, uv=False)
        G.put(bm, cl, loc=(sgn * SPK_X, YF, SPK_Z), rot=_TO_FRONT)
        mt = bmesh.new()
        for r in M.rib_radii(0.0118, 0.0372, 6):
            w = 0.0011
            bm_lathe(mt, [(r + w, 0.0006), (r + w * 0.7, 0.0014), (r, 0.0017), (r - w * 0.7, 0.0014), (r - w, 0.0006)],
                     segs=72, mat=1, uv=False)
        bm_lathe(mt, [(0.0090, 0.0008), (0.0086, 0.0016), (0.0064, 0.0027), (0.0030, 0.0033), (0.0, 0.0034)], segs=32, mat=1,
                 uv=False)
        G.put(bm, mt, loc=(sgn * SPK_X, YF, SPK_Z), rot=_TO_FRONT)
    return bm


def build_hinges():
    """Two steel hinge knuckles on the lower edge of the cassette door (material 0 metal)."""
    bm = bmesh.new()
    for sx in (-0.047, 0.047):
        cyl_x(bm, sx - 0.0070, sx + 0.0070, 0.0017, YF - 0.0012, BAY_FRAME[0] + 0.0010, segs=12)
    G.soften(bm, 0.0004)
    return bm


def build_feet():
    bm = bmesh.new()
    for sx in (-0.145, 0.145):
        for sy in (-0.035, 0.018):
            b = bmesh.new()
            bm_lathe(b, [(0.0, 0.0), (0.0078, 0.0), (0.0085, 0.0012), (0.0085, 0.0072), (0.0078, 0.0086), (0.0060, 0.0092),
                         (0.0, 0.0094)], segs=24, mat=0, uv=False)
            G.put(bm, b, loc=(sx, sy, 0.0))
    return bm


def build_antenna(C):
    """Telescopic antenna folded along the back, near the top: base block at the right, three sections, a tip ball."""
    bm = bmesh.new()
    y, z = -YF + 0.0004, 0.1385
    secs = [(0.1330, 0.0560, 0.0028), (0.0560, -0.0200, 0.0022), (-0.0200, -0.0850, 0.0017)]
    for x1, x0, r in secs:
        cyl_x(bm, x0, x1, r, y, z, segs=14)
    b = bmesh.new()
    bm_lathe(b, [(0.0, -0.0030), (0.0020, -0.0020), (0.0030, 0.0), (0.0020, 0.0020), (0.0, 0.0030)], segs=14, mat=0, uv=False)
    G.put(bm, b, loc=(-0.0880, y, z))
    G.soften(bm, 0.0003)
    return bm


# ================================================================= the bay's cassette
def build_cassette(K, label_spec, rng):
    """The standing cassette in the bay: (shell bmesh, body bmesh, shell materials, body materials, loc, rot) for K.to_obj
    (its label faces front)."""
    C = TAPE.colours(K, label_spec)
    shell, body = TAPE.build_cassette(K, C, rng)
    sm, bmats = TAPE.cassette_materials(K, C, prefix="cass_")
    loc = (0.0, YF + CASSETTE_D + TAPE.T, BAY_ZC)
    return shell, body, sm, bmats, loc, (math.pi / 2, 0.0, 0.0)


# ================================================================= assembly
def assemble(K, label_spec, rng):
    """Build the player's objects and custom property; -> (use dict, collider specs) for the card."""
    K.prop("glow", 1.0, 0.0, 1.0, "emission of the dial, LED and level meter (0 = dark, 1 = normal)")
    C = colours(K)
    mats = make_materials(K, C)
    K.to_obj("body", build_body(C), [mats["plastic"]])
    K.to_obj("grilles", build_grilles(C), [mats["cloth"], mats["metal"]])
    K.to_obj("glass", build_glass(C), [mats["glass"]]).visible_shadow = False
    K.to_obj("glow", build_glow(C), [mats["glow"]])
    K.to_obj("hinges", build_hinges(), [mats["metal"]])
    K.to_obj("feet", build_feet(), [mats["rubber"]])
    K.to_obj("antenna", build_antenna(C), [mats["metal"]])
    shell, body, sm, bmats, loc, rot = build_cassette(K, label_spec, rng)
    K.to_obj("cassette", body, bmats, loc=loc, rot=rot)
    K.to_obj("cassette_shell", shell, sm, loc=loc, rot=rot).visible_shadow = False
    bm = bmesh.new()
    bm_box(bm, (BODY_W, BODY_D, Z1), loc=(0.0, 0.0, Z1 / 2))
    col = K.collider(K.to_obj("col_body", bm, []))
    centre, size = dial_text_surface()
    use = {
        "look": [{"name": "dial", "point": [DIAL_CX, round(YF - 0.0025, 4), round(DIAL_Z, 4)]},
                 {"name": "bay", "point": [0.0, round(YF - 0.0020, 4), round(BAY_ZC, 4)]}],
        "surface": [{"name": "dial", "center": centre, "normal": [0, -1, 0], "up": [0, 0, 1], "size": size}],
        "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, Z1], "normal": [0, 0, 1], "size": [0.30, 0.09]}],
    }
    colliders = [{"type": "box", "object": col.name, "rnd": 0.006, "tag": K.name}]
    return use, colliders
