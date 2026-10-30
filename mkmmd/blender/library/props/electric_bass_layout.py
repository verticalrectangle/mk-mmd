"""Layout of the short-scale electric bass (pure Python + numpy, no bpy: importable from tests): every number the parts agree
on, the colour roles, the `WEAR` entry and the prop card (`card`).

Frame (metres; the contract with the hand, strum and wear code, the guitar's): the bass stands UPRIGHT on its tail. Z runs up
along the strings toward the nut and headstock, the FACE (the strings' side) looks toward -Y (card `front` = "-Y"), the back
is +Y. X is across the bass seen from the front: the LOW E string (4th) is at -X, the G at +X, the long (upper) horn at -X. A
right-handed player holding it (neck to her left, low E on top) is this bass turned +90 degrees about +Y. ORIGIN: on the centre
line, on the body's flat front face plane (y = 0), at the nominal SADDLE LINE (z = 0): the body slab occupies y in [0,
BODY_T], everything in front of it (pickguard, pickup, bridge, strings, the neck's board) has y < 0, the nut is at z = SCALE
and fret n at SCALE * 2 ** (-n / 12).

Modelled on a Fender Mustang Bass (1966-81: 30 inch scale, 19 frets, an offset body with rounded horns, one split single coil,
a string-through bridge with four saddles, a chrome control plate with two knobs and the jack): measured on a front photograph
rectified to metres by its ten visible fret wires (frets 10 to 19 fitted with 12-TET on the known scale and a scale that grows
toward the neck, residual 0.7 mm; the 12th fret's double dot and the 15th, 17th and 19th dots confirm the numbering), the body
then 0.310 across, within 1.5 % of the 12 3/8 inch (0.314 m) a dealer measured, so every x is scaled by 0.314 / 0.310: the
tail at z -0.072, the lower bout's widest at z 0.02-0.04, the bass waist 0.096 from the centre at z 0.21 and the treble one
0.105 at z 0.15, the long horn's tip at z 0.376 (the 12th fret) and the short one's at 0.310 (the 15.6th), the bays down to
z 0.326 (bass) and 0.279 (treble). The headstock is a Fender bass headstock read off a front photograph of one with a 1.75 inch
nut and scaled to this neck's 41.3 mm (0.93): 0.229 from the nut to the tip, its posts on a line 12.8 degrees off the neck.
The numbers below are landmarks read off those pictures and rounded, never a traced contour.

Colour roles (`[[prop]] slots = {body = "foam"}`; a palette slot, a hex, or a blend 'slot:w,slot:w'): see `ROLES`. A `stripe`
in the slots adds the racing stripes of the 70s Competition finish (three bands across the lower bout) in that colour.
"""

import numpy as np

from . import convertible_palette as CP

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = {
    "body": "love",                           # the gloss coat of the body
    "pickguard": "text",                      # the guard (a pale pearloid by default)
    "neck": "gold:5,rose:3,overlay:2",        # warm maple (neck, headstock)
    "fretboard": "rose:2,gold:1,base:2",      # darker warm wood
    "hardware": "text",                       # bright metal: bridge, control plate, tuners, frets, screws, strap buttons, plug
    "strings": "text:2,muted:1",
    "knobs": "base:3,overlay:1",              # the black plastic of the knobs and the pickup covers (dark, never black)
    "strap": "base:2,iris:1",                 # the strap's band: dark, never black
    "cable": "text",                          # the cord
    "pick": "gold",                           # the plectrum
    "inlay": "text:3,gold:1",                 # nut and fret markers (bone)
    "dark": "base",                           # gaps and holes: the darkest colour of the palette, never black
    "stripe": "text",                         # the racing stripes, drawn only when the prop's slots name `stripe`
}

# material index of a role in the meshes (`Mesh.Qm / Tm`); the builder turns them into Blender materials
MATS = {name: i for i, name in enumerate(("body", "pickguard", "neck", "fretboard", "hardware", "strings", "knobs", "strap",
                                          "cable", "pick", "inlay", "dark", "guard_core", "frets", "stripe"))}
ROLE_OF_MAT = {"guard_core": "dark", "frets": "hardware"}      # materials that take another role's colour


class Pal(CP.Pal):
    """The bass's colour roles (`ROLES`) over the palette; a role may name another role."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}
        self.stripes = "stripe" in slots

    def lin(self, spec):
        seen = 0
        while spec in self.specs and seen < 8:
            spec, seen = self.specs[spec], seen + 1
        return super().lin(spec)


# ===================================================================================================================
# the neck, frets and strings
# ===================================================================================================================
SCALE = 0.762                       # saddle line to nut, m (30 inches)
N_FRETS = 19
Z_NUT = SCALE
Y_BOARD = -0.0115                   # the board's crown plane: the neck stands 11.5 mm proud of the body's face
BODY_T = 0.038                      # body thickness (1.5 inch): y in [0, BODY_T]
FRET_H = 0.0011                     # a (vintage, narrow) wire stands this far out of the board
FRET_W = 0.0022                     # crown width
BOARD_R = 0.18415                   # 7.25 inch radius
BOARD_T = 0.0060                    # rosewood slab thickness at the crown (the neck wood starts under it)
W_NUT, W_LAST = 0.0413, 0.0570      # board / neck width at the nut (1 5/8 inch) and at the last fret
D_NUT, D_LAST = 0.0210, 0.0240      # depth from the board's crown to the back at the centre line (nut, last fret)
P_SECTION = 2.6                     # superellipse exponent of the back of the neck (a C)
NUT_T, NUT_H = 0.0045, 0.0040       # nut thickness along z (z in [SCALE, SCALE + NUT_T]) and its top above the board's crown
TUNING = (28, 33, 38, 43)           # E1 A1 D2 G2 (MIDI), the lowest string first


def fret_z(n):
    """z of fret wire n (0 = the nut) on the nominal scale."""
    return SCALE * 2.0 ** (-np.asarray(n, float) / 12.0) if not isinstance(n, int) else SCALE * 2.0 ** (-n / 12.0)


Z_LAST = fret_z(N_FRETS)
Z_NUT_BACK = Z_NUT + NUT_T
Z_HEEL = Z_LAST - 0.0110            # the board and the neck end here, on the body (the board runs 11 mm past the last wire)
DOTS = (3, 5, 7, 9, 12, 15, 17, 19)  # fret markers (between wire n - 1 and n), a pair at 12


def section_wd(z):
    """(width, depth) of the neck solid at z: linear between the nut and the last fret, held beyond (the card's `section`)."""
    t = np.clip((Z_NUT - np.asarray(z, float)) / (Z_NUT - Z_LAST), 0.0, 1.0)
    return W_NUT + t * (W_LAST - W_NUT), D_NUT + t * (D_LAST - D_NUT)


def board_sag(x):
    """How far the board's surface falls below the crown at x (its 7.25 inch radius)."""
    x = np.asarray(x, float)
    return BOARD_R - np.sqrt(np.maximum(BOARD_R ** 2 - x ** 2, 0.0))


def back_half_width(v, w, d, p=P_SECTION):
    """Half width of the neck's back at depth v below the crown plane: |2x/w|^p + (v/d)^p = 1 (v in 0..d)."""
    r = np.clip(1.0 - (np.asarray(v, float) / d) ** p, 0.0, None)
    return 0.5 * w * r ** (1.0 / p)


STRING_NAMES = ("E", "A", "D", "G")                                    # the lowest first: x increasing
GAUGE_IN = (0.105, 0.085, 0.065, 0.045)                                # a 45-105 set
STRING_R = tuple(0.5 * 0.0254 * g for g in GAUGE_IN)
NUT_PITCH, BRIDGE_PITCH = 0.0106, 0.0177                               # string spacing at the nut and the saddles
NUT_X = tuple((i - 1.5) * NUT_PITCH for i in range(4))
SADDLE_X = tuple((i - 1.5) * BRIDGE_PITCH for i in range(4))
SETBACK = (0.0090, 0.0072, 0.0058, 0.0042)                             # saddle z = -setback (intonation: the thick strings longer)
ACTION12 = (0.0030, 0.0028, 0.0026, 0.0024)                            # clearance over the 12th wire's crown, graded
NUT_CLEAR = 0.0006                                                     # a string's underside clears the first wire's crown by this at the nut


def h_nut(i):
    """Height of string i's axis over the board's crown plane at the nut (its slot is cut to the same clearance over the first
    wire: the thick strings sit higher)."""
    return FRET_H + NUT_CLEAR + STRING_R[i]


def string_end_heights():
    """(h_nut, h_saddle) of every string axis above the board's crown plane: a straight line from the nut slot to the saddle
    that clears the 12th wire's crown by ACTION12 (the neck has no tilt, so the saddles stand higher than the board)."""
    out = []
    for i in range(4):
        zs = -SETBACK[i]
        t = (fret_z(12) - zs) / (Z_NUT - zs)
        h12 = FRET_H + ACTION12[i] + STRING_R[i]
        out.append((h_nut(i), (h12 - h_nut(i) * t) / (1.0 - t)))
    return out


def string_ends(i):
    """(nut point, saddle point) of string i on its axis, prop frame."""
    hn, hs = string_end_heights()[i]
    return (NUT_X[i], Y_BOARD - hn, Z_NUT), (SADDLE_X[i], Y_BOARD - hs, -SETBACK[i])


def string_at(i, z):
    """Point of string i's axis at z."""
    a, b = (np.array(p) for p in string_ends(i))
    return a + (b - a) * ((z - a[2]) / (b[2] - a[2]))


# ===================================================================================================================
# the body
# ===================================================================================================================
X_SCALE = 0.314 / 0.310             # the photograph's x, rescaled to the measured lower bout

# control points of the body outline (x, z) as read off the rectified photograph (before X_SCALE), counter-clockwise from
# the tail's centre: the treble side up to the short horn, its bay, round the neck pocket under the neck (hidden by it), the
# bass bay, the long horn, down the bass side back to the tail. The spline runs through them.
_OUTLINE = (
    (0.000, -0.0718), (0.025, -0.0718), (0.050, -0.0716), (0.070, -0.0690), (0.090, -0.0650), (0.105, -0.0590),
    (0.120, -0.0500), (0.132, -0.0400), (0.142, -0.0290), (0.150, -0.0170), (0.1555, -0.0030), (0.1585, 0.0100),
    (0.1590, 0.0230), (0.1580, 0.0360), (0.1550, 0.0490), (0.1505, 0.0620), (0.1440, 0.0760), (0.1365, 0.0900),
    (0.1280, 0.1030), (0.1195, 0.1160), (0.1115, 0.1290), (0.1068, 0.1410), (0.1055, 0.1520), (0.1065, 0.1630),
    (0.1090, 0.1760), (0.1130, 0.1920), (0.1180, 0.2100), (0.1215, 0.2250), (0.1240, 0.2420), (0.1255, 0.2600),
    (0.1255, 0.2750), (0.1240, 0.2880), (0.1205, 0.2990), (0.1140, 0.3070), (0.1060, 0.3105), (0.0980, 0.3095),
    (0.0915, 0.3040),                                                           # the short horn's round tip
    (0.0870, 0.2900), (0.0800, 0.2835), (0.0700, 0.2795), (0.0600, 0.2785), (0.0500, 0.2805), (0.0420, 0.2855),
    (0.0360, 0.2930), (0.0330, 0.3030), (0.0310, 0.3140),                       # its bay, up to the neck
    (0.0300, 0.3240), (0.0200, 0.3300), (0.0000, 0.3310), (-0.0200, 0.3300),    # the pocket's top edge, under the neck
    (-0.0300, 0.3300), (-0.0400, 0.3260), (-0.0500, 0.3255), (-0.0600, 0.3280), (-0.0700, 0.3330), (-0.0770, 0.3380),
    (-0.0820, 0.3470), (-0.0860, 0.3590), (-0.0900, 0.3690),                    # the bass bay
    (-0.0965, 0.3760), (-0.1040, 0.3740), (-0.1100, 0.3660),                    # the long horn's tip
    (-0.1145, 0.3560), (-0.1168, 0.3420), (-0.1172, 0.3270), (-0.1162, 0.3120), (-0.1148, 0.2970), (-0.1120, 0.2830),
    (-0.1080, 0.2700), (-0.1030, 0.2530), (-0.0985, 0.2380), (-0.0960, 0.2250), (-0.0955, 0.2120), (-0.0970, 0.2000),
    (-0.0998, 0.1870), (-0.1038, 0.1720), (-0.1098, 0.1570), (-0.1168, 0.1420), (-0.1240, 0.1280), (-0.1300, 0.1130),
    (-0.1355, 0.0990), (-0.1395, 0.0850), (-0.1430, 0.0720), (-0.1465, 0.0580), (-0.1495, 0.0440), (-0.1510, 0.0300),
    (-0.1505, 0.0160), (-0.1480, 0.0030), (-0.1430, -0.0090), (-0.1370, -0.0200), (-0.1290, -0.0300), (-0.1190, -0.0390),
    (-0.1070, -0.0470), (-0.0920, -0.0540), (-0.0750, -0.0600), (-0.0550, -0.0655), (-0.0300, -0.0695), (-0.0120, -0.0712),
)


def body_control_points():
    """The outline's control points (n, 2), counter-clockwise, in the prop's frame (x rescaled)."""
    P = np.array(_OUTLINE, float)
    P[:, 0] *= X_SCALE
    return P


# strap buttons: where the button leaves the body, and the way it points
STRAP_TOP = {"point": (-0.0990, BODY_T / 2, 0.3755), "dir": (-0.25, 0.0, 0.9682)}   # the long horn's tip
STRAP_BOTTOM = {"point": (-0.0050, BODY_T / 2, -0.0717), "dir": (0.0, 0.0, -1.0)}   # the tail edge
BUTTON_H = 0.0155                   # a strap button stands this far out of the edge

# the contours of the 70s body (see electric_bass_body): the forearm bevel on the bass edge of the front, the belly cut on the back
FOREARM = {"depth": 0.0060, "width": 0.040, "z": (0.020, 0.065, 0.150, 0.200)}      # ramps in at z 0.02..0.065, out at 0.15..0.2
BELLY = {"center": (-0.075, 0.170), "radii": (0.070, 0.110), "depth": 0.0090}       # elliptical scoop on the back, bass side
BODY_EDGE_R = 0.0050                # the body's rolled edge (front and back)

# the pickguard (read off the photograph, before X_SCALE): counter-clockwise from the bass side of the neck's notch
GUARD_T = 0.0024
_GUARD = (
    (-0.0262, 0.3080), (-0.0330, 0.3078), (-0.0370, 0.3045), (-0.0398, 0.2950), (-0.0443, 0.2850), (-0.0490, 0.2710),
    (-0.0515, 0.2580), (-0.0532, 0.2450), (-0.0560, 0.2300), (-0.0600, 0.2170), (-0.0620, 0.2050), (-0.0630, 0.1950),
    (-0.0665, 0.1840), (-0.0700, 0.1740), (-0.0712, 0.1650), (-0.0705, 0.1530), (-0.0688, 0.1420), (-0.0660, 0.1310),
    (-0.0625, 0.1200), (-0.0570, 0.1110), (-0.0490, 0.1015), (-0.0400, 0.0965), (-0.0300, 0.0948), (0.0000, 0.0944),
    (0.0300, 0.0944), (0.0440, 0.0950),                                     # the bottom edge, down to the control plate
    (0.0490, 0.0965), (0.0545, 0.1050), (0.0620, 0.1145), (0.0715, 0.1215), (0.0820, 0.1255), (0.0935, 0.1268),
    (0.0985, 0.1290),                                                       # along the plate's upper edge (1 mm clear of it)
    (0.0955, 0.1380), (0.0945, 0.1500), (0.0955, 0.1650), (0.0978, 0.1820), (0.1010, 0.2000), (0.1040, 0.2150),
    (0.1065, 0.2320), (0.1100, 0.2480), (0.1130, 0.2630), (0.1150, 0.2780), (0.1140, 0.2895),
    (0.1100, 0.2950),                                                       # the guard's tip in the short horn
    (0.1025, 0.2925), (0.0950, 0.2825), (0.0875, 0.2710), (0.0775, 0.2645), (0.0650, 0.2620), (0.0525, 0.2630),
    (0.0425, 0.2670), (0.0340, 0.2745), (0.0312, 0.2790),                   # along the treble bay
    (0.0283, 0.2800),                                                       # to the neck (a blunt tip: a sliver would fold)
    (0.0283, 0.2480), (0.0230, 0.2405), (0.0000, 0.2385), (-0.0230, 0.2405), (-0.0283, 0.2480),   # round the neck's end
    (-0.0283, 0.3000),
)
GUARD_CORNERS = (0, len(_GUARD) - 7, len(_GUARD) - 6, len(_GUARD) - 2, len(_GUARD) - 1)


def guard_control_points():
    """The pickguard's control points (n, 2), counter-clockwise, and the indices where it turns sharply (the notch for the neck)."""
    P = np.array(_GUARD, float)
    P[:, 0] *= X_SCALE
    return P, GUARD_CORNERS


GUARD_SCREWS = ((-0.0350, 0.2980), (-0.0545, 0.2350), (-0.0640, 0.1720), (-0.0560, 0.1180), (-0.0300, 0.0995), (0.0150, 0.0990),
                (0.0480, 0.1010), (0.0960, 0.1560), (0.1015, 0.2120), (0.1065, 0.2850), (0.0700, 0.2690), (0.0420, 0.2700))

# the split single coil: two covers (stadiums) under the E-A and D-G pairs, the bass half nearer the neck
PICKUP_L, PICKUP_W = 0.0395, 0.0180          # cover length (across) and width (along the strings)
PICKUP_H = 0.0100                            # cover height above the guard
PICKUPS = {"bass": {"center": (-0.0145, 0.1545)}, "treble": {"center": (0.0195, 0.1340)}}    # midway between their screws
PICKUP_SCREW = 0.0245                        # the height screws stand this far from a cover's centre, beyond its ends

# the string-through bridge: a chrome plate (a rounded trapezoid), four saddles, a rolled tail where the strings go through
BRIDGE = {"top": 0.0370, "bottom": -0.0455, "top_x": (-0.0505, 0.0515), "bottom_x": (-0.0405, 0.0440), "t": 0.0020,
          "r_top": 0.0120, "r_bottom": 0.0050}
TAIL = {"z": -0.0525, "x": (-0.0395, 0.0430), "r": 0.0068}                 # the rolled tail: a half-round across the plate's end
STRING_HOLE_Z = -0.0395                      # the strings go down through the plate and the body here
SADDLE = {"w": 0.0100, "l": 0.0210, "top": 0.0050}
BRIDGE_SCREWS = ((-0.0360, 0.0250), (-0.0385, 0.0100), (0.0410, 0.0250), (0.0435, 0.0100))
FLANGE_Z = 0.0300                            # the intonation screws' heads stand on the plate here, ahead of the saddles

# the chrome control plate along the lower treble edge, its two knobs and the jack (read off the photograph, before X_SCALE)
_PLATE = ((0.0940, 0.1225), (0.1030, 0.1215), (0.1100, 0.1180), (0.1170, 0.1115), (0.1240, 0.1035), (0.1300, 0.0950),
          (0.1360, 0.0860), (0.1410, 0.0760), (0.1460, 0.0660), (0.1500, 0.0560), (0.1530, 0.0460), (0.1550, 0.0375),
          (0.1555, 0.0295), (0.1545, 0.0215), (0.1515, 0.0140), (0.1460, 0.0090), (0.1390, 0.0080), (0.1325, 0.0110),
          (0.1260, 0.0165), (0.1180, 0.0265), (0.1080, 0.0405), (0.0980, 0.0560), (0.0880, 0.0700), (0.0760, 0.0820),
          (0.0640, 0.0895), (0.0560, 0.0915), (0.0520, 0.0925), (0.0525, 0.0975), (0.0580, 0.1050), (0.0650, 0.1115),
          (0.0730, 0.1170), (0.0830, 0.1210))
PLATE_T = 0.0012


def plate_control_points():
    P = np.array(_PLATE, float)
    P[:, 0] *= X_SCALE
    return P


KNOBS = ((0.0850, 0.0976), (0.1135, 0.0640))           # volume, tone (x already in the prop's frame)
KNOB_R, KNOB_H = 0.0098, 0.0170
PLATE_SCREWS = ((0.0600, 0.0950), (0.0960, 0.1180), (0.1415, 0.0150))
JACK = {"center": (0.1295, 0.0320), "plug_len": 0.052, "mouth": 0.0}    # front-mounted on the plate, straight out of the face

# the headstock: (u, v) = (distance from the nut's front face along z, x) in metres, bass (min x) and treble (max x) edge,
# read off a Fender bass headstock (1.75 inch nut) and scaled by HEAD_K
HEAD_K = 0.93
_HEAD_BASS = ((0.0, -0.0221), (6.0, -0.0228), (13.4, -0.0281), (20.9, -0.0387), (29.4, -0.0472), (40.0, -0.0515),
              (48.5, -0.0526), (57.0, -0.0511), (67.7, -0.0483), (88.9, -0.0419), (110.2, -0.0355), (131.5, -0.0291),
              (152.8, -0.0228), (174.0, -0.0185), (195.3, -0.0153), (216.6, -0.0132), (229.4, -0.0117), (240.0, -0.0079),
              (245.3, -0.0004), (246.4, 0.0102))
_HEAD_TREBLE = ((0.0, 0.0223), (6.0, 0.0230), (14.5, 0.0251), (25.1, 0.0294), (35.7, 0.0368), (46.4, 0.0464), (56.0, 0.0513),
                (61.3, 0.0523), (78.3, 0.0496), (99.6, 0.0460), (120.9, 0.0417), (142.1, 0.0374), (159.1, 0.0357),
                (169.8, 0.0362), (178.3, 0.0389), (185.7, 0.0443), (192.1, 0.0506), (199.6, 0.0545), (210.2, 0.0560),
                (223.0, 0.0538), (233.6, 0.0464), (241.1, 0.0357), (245.3, 0.0230), (246.4, 0.0102))


def head_outline():
    """(bass, treble): (u, v) tables of the headstock's two edges in metres (u from the nut's front face), scaled to this
    neck. Both start at the neck's width at the nut and end at the tip (u = HEAD_L, v = the tip's x)."""
    out = []
    for tab in (_HEAD_BASS, _HEAD_TREBLE):
        P = np.array(tab, float)
        out.append(np.stack([P[:, 0] * 1e-3 * HEAD_K, P[:, 1] * HEAD_K], 1))
    return out


HEAD_L = 0.2464 * HEAD_K                     # nut front face to the tip
HEAD_P = 6.0                                 # the headstock's section exponent (a rounded rectangle)
POSTS = tuple((SCALE + 1e-3 * HEAD_K * u, HEAD_K * v) for u, v in ((45.3, -0.0217), (94.3, -0.0106), (143.6, 0.0006),
                                                                   (193.6, 0.0119)))   # tuner posts (z, x), string 4 (E) first
POST_R = 0.0042                              # a bass tuner post's radius (the string's winding shaft)
KEY_DIR = (-1.0, 0.0, 0.0)                   # from a post's gear toward its key: the keys stick out of the bass edge
KEY_OFFSET = 0.026                           # the key's shaft runs this far toward the tip from its post (the gear's worm)
KEY_LEN = 0.048                              # from the post's axis to the key button's centre
TREE = (SCALE + 1e-3 * HEAD_K * 79.4, HEAD_K * 0.0134)    # the string retainer for the D and G strings (z, x)

# the plectrum (351 shape) in the pinch frame
PICK = {"thickness": 0.0010, "length": 0.031, "width": 0.026, "tip": 0.008}
PICK_REST = {"at": (-0.0450, 0.0560)}        # x, z where it lies on the body's face (tip toward +z), between the bridge and guard
STRUM_ZONE = (0.0800, 0.1200)                # between the bridge and the pickup's covers: a pick player's hand rests its heel
                                             # by the bridge (PALM) and picks a few centimetres up the strings from it
PALM = {"z": (-0.0150, 0.0400), "across": 0.0700, "lift": 0.0120}   # the patch over the strings by the saddles the heel of
                                             # a picking hand rests over (along z, across, at most `lift` over the strings)
TECHNIQUE = "fingers"                        # the default way the strum hand plays it: rest strokes with the index and middle
                                             # fingers, as most bassists do (`[perform.<cast>] strum` technique = "pick" picks)
FINGER_ZONE = (0.1000, 0.1900)               # along the strings: where the fingers pluck (over and around the pickup)
THUMB_CLEAR = 0.0020                         # the thumb's rest on the bass cover keeps this far inside its end and off the E

# cable (a bevelled curve) and its end
CABLE_R = 0.0032
FLOOR_Z = -0.14                              # the upright bass's floor: the cord ends here by default

WEAR = {"name": "stand", "bone": "upper_body2",
        "pivot": [0.0, BODY_T, 0.0],
        "ref": {"top": 1.70, "shoulder_width": 0.18},
        "at": [-0.23, -0.07, -0.13], "scale": ["shoulder_width", "top", "top"],    # worn at the hip, the neck 25 degrees up:
        "neck_deg": 25.0, "yaw_deg": 8.0, "roll_deg": 0.0,                         # a line from the 7th position and the pick zone reach
        "strap": {"top": "strap_top", "bottom": "strap_bottom", "over": "shoulder.L", "width": 0.05, "thickness": 0.004,
                  "material": "<name>_strap"},
        "cable": {"object": "<name>_cable", "anchor": "jack", "radius": CABLE_R}}

COLLIDERS = {                                # hidden objects: key -> (centre, size)  (box specs use the object's bounds)
    "body": ((0.0040, 0.0190, 0.0500), (0.3140, 0.0380, 0.2440)),
    "horn": ((-0.1000, 0.0190, 0.2950), (0.0400, 0.0380, 0.1600)),
    "neck": ((0.0000, 0.0010, 0.5050), (0.0570, 0.0240, 0.5200)),
    "head": ((0.0000, -0.0010, SCALE + 0.1150), (0.1000, 0.0170, 0.2300)),
}
NECK_CAPSULE = {"a": (0.0, 0.0010, 0.2450), "b": (0.0, 0.0010, 0.7650), "R": 0.0300}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def jack_geometry():
    """(mouth, dir, end): the socket's mouth on the control plate, the unit vector out of the socket (straight out of the face)
    and the plug's end where the cord leaves."""
    cx, cz = JACK["center"]
    mouth = np.array([cx, -PLATE_T - 0.0030, cz])
    d = np.array([0.0, -1.0, 0.0])
    return mouth, d, mouth + d * JACK["plug_len"]


def strum_center():
    """The pick tip's rest point: centred across the four strings, on their plane, between the bridge and the pickups."""
    z = 0.5 * sum(STRUM_ZONE)
    y = float(np.mean([string_at(i, z)[1] for i in range(4)]))
    return (0.0, y, z)


def palm_entry():
    """The strum entry's `palm`: the patch over the strings by the saddles that a picking hand's heel rests over (the pinch
    solve keeps the heel at most `lift` above it)."""
    z = 0.5 * sum(PALM["z"])
    y = min(float(string_at(i, z)[1]) - STRING_R[i] for i in range(4))                # the strings' tops (the face looks to -y)
    return {"center": _r((0.0, y, z)), "normal": [0, -1, 0], "size": _r((PALM["across"], PALM["z"][1] - PALM["z"][0])),
            "lift": PALM["lift"]}


def fingers_entry():
    """The strum entry's `fingers` table (mkmmd.core.strum.pluck_prop): the zone the fingers pluck in, the thumb's rest while
    the low E rings (the bass cover's top between its end and the E string), the guard's face under the strings and the two
    pickup covers standing on it."""
    y_base = -GUARD_T + 0.0001                                           # the covers stand on the guard (hw: pickups_mesh)
    y_top = y_base - PICKUP_H
    cx, cz = PICKUPS["bass"]["center"]
    end = cx - 0.5 * PICKUP_L + THUMB_CLEAR
    e_x = float(string_at(0, cz)[0]) - STRING_R[0] - THUMB_CLEAR
    blocks = [{"center": _r((x, 0.5 * (y_base + y_top), z)), "size": _r((PICKUP_L, PICKUP_H, PICKUP_W))}
              for x, z in (p["center"] for p in PICKUPS.values())]
    return {"zone": list(FINGER_ZONE),
            "anchor": {"center": _r((0.5 * (end + e_x), y_top, cz)), "normal": [0, -1, 0],
                       "size": _r((e_x - end, 0.5 * PICKUP_W))},
            "face": {"point": _r((0.0, -GUARD_T, 0.0)), "normal": [0, -1, 0]}, "blocks": blocks}


def wear_entry(name):
    e = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in WEAR.items()}
    e["strap"]["material"] = e["strap"]["material"].replace("<name>", name)
    e["cable"]["object"] = e["cable"]["object"].replace("<name>", name)
    return e


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, roles=None, size=None):
    """The prop card (docs/design.md: Prop card; the `use` entries are the contract with the grip, strum and wear code)."""
    zs = [float(fret_z(n)) for n in range(N_FRETS + 1)]
    strings = []
    for i in range(4):
        a, b = string_ends(i)
        strings.append({"name": STRING_NAMES[i], "nut": _r(a), "bridge": _r(b), "radius": round(STRING_R[i], 6),
                        "midi": TUNING[i], "object": f"{name}_string_{4 - i}"})
    center = strum_center()
    mouth, jdir, jend = jack_geometry()
    top, bot = STRAP_TOP, STRAP_BOTTOM
    head = (0.0, -0.0050, SCALE + 0.55 * HEAD_L)
    neck = {"name": "neck", "type": "neck",
            "frame": {"along": [0, 0, 1], "across": [1, 0, 0], "normal": [0, -1, 0]},
            "thumb": [0, 1, 0], "scale": SCALE,
            "frets": [[0.0, Y_BOARD, round(z, 6)] for z in zs],
            "strings": strings, "tuning": list(TUNING),
            "section": {"width": [W_NUT, W_LAST], "depth": [D_NUT, D_LAST], "p": P_SECTION,
                        "board_width": [W_NUT, W_LAST], "board_radius": BOARD_R},
            "fret_height": FRET_H, "object": f"{name}_neck"}
    strum = {"name": "strum", "type": "strum", "center": _r(center), "along": [0, 0, 1], "across": [1, 0, 0],
             "normal": [0, -1, 0], "zone": list(STRUM_ZONE), "avoid": [f"{name}_body"],
             "pick": {"object": f"{name}_pick", "thickness": PICK["thickness"], "length": PICK["length"],
                      "width": PICK["width"], "tip": PICK["tip"]},
             "technique": TECHNIQUE, "fingers": fingers_entry(), "palm": palm_entry()}
    anchors = [{"name": "strap_top", "point": _r(top["point"]), "dir": _r(_unit(top["dir"])), "object": f"{name}_strap_top"},
               {"name": "strap_bottom", "point": _r(bot["point"]), "dir": _r(_unit(bot["dir"])),
                "object": f"{name}_strap_bottom"},
               {"name": "jack", "point": _r(jend), "dir": _r(jdir), "object": f"{name}_jack"},
               {"name": "head", "point": _r(head)}]
    looks = [{"name": "neck", "point": _r((0.0, Y_BOARD, float(fret_z(5))))},
             {"name": "strum", "point": _r(center)},
             {"name": "head", "point": _r(head)}]
    cols = [{"type": "box", "object": f"{name}_col_{k}", "rnd": 0.008, "tag": name} for k in ("body", "horn", "head")]
    cols.append({"type": "capsule", "object": f"{name}_col_neck", "a": _r(NECK_CAPSULE["a"]), "b": _r(NECK_CAPSULE["b"]),
                 "R": NECK_CAPSULE["R"], "tag": name})
    out = {"size": _r(size if size is not None else (0.33, 0.09, 1.07), 4), "origin": "bridge", "front": "-Y",
           "slots": dict(roles) if roles is not None else dict(ROLES),
           "use": {"grip": [neck, strum], "anchor": anchors, "look": looks, "wear": [wear_entry(name)]},
           "colliders": cols}
    return out
