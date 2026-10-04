"""Layout of the 80s electric guitar (pure Python + numpy, no bpy: importable from tests): every number the parts agree on,
the colour roles, the `WEAR` entry and the prop card (`card`).

Frame (metres; the contract with the hand, strum and wear code): the guitar stands UPRIGHT on its tail. Z runs up along the
strings toward the nut and headstock, the FACE (the strings' side) looks toward -Y (card `front` = "-Y"), the back is +Y.
X is across the guitar seen from the front (the viewer at -Y looking along +Y has +X on her right): the LOW E string (6th) is
at -X, the high e at +X, the long (upper) horn at -X. A right-handed player holding it (neck to her left, low E on top) is this
guitar turned +90 degrees about +Y. ORIGIN: on the centre line, on the body's flat front face plane (y = 0), at the SADDLE
LINE (z = 0): the body slab occupies y in [0, BODY_T], everything in front of it (pickguard, pickups, bridge, strings, the
neck's board) has y < 0, the nut is at z = SCALE and fret n at SCALE * 2 ** (-n / 12).

Modelled on a Fender Stratocaster-type body and neck, measured on a perspective-corrected (rectified) photograph whose scale
came from its 20 visible fret wires: body 0.315 wide at the lower bout, 0.2252 at the waist (z 0.125), tail edge at z -0.1155,
the long horn's tip at z 0.314 (the 12.6th fret), the short horn's at 0.2565 (the 16th), the bays reaching the neck at z 0.2545
(long side) and 0.205 (short side, a deep cutaway); headstock 0.1826 long and 0.088 wide. The numbers below are landmarks
read off that picture and rounded, never a traced contour.

Colour roles (`[[prop]] slots = {body = "iris"}`; a palette slot, a hex, or a blend 'slot:w,slot:w'): see `ROLES`.
"""
import math

import numpy as np

from . import convertible_palette as CP

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = {
    "body": "love",                           # the gloss coat of the body
    "pickguard": "text",                      # the guard's plies, white
    "neck": "gold:5,rose:3,overlay:2",        # warm maple (neck, headstock)
    "fretboard": "rose:2,gold:1,base:2",      # darker warm wood
    "hardware": "text",                       # bright metal: bridge, tuners, frets, pole pieces, plates, strap buttons, plug
    "strings": "text:2,muted:1",
    "knobs": "pickguard",                     # the knobs, the switch tip, the pickup covers (default: the guard's colour)
    "strap": "base:2,iris:1",                 # the strap's band: dark, never black
    "cable": "text",                          # the cord
    "pick": "gold",                           # the plectrum
    "inlay": "text:3,gold:1",                 # nut and fret markers (bone)
    "dark": "base",                           # gaps and holes: the darkest colour of the palette, never black
}

# material index of a role in the meshes (`Mesh.Qm / Tm`); the builder turns them into Blender materials
MATS = {name: i for i, name in enumerate(("body", "pickguard", "neck", "fretboard", "hardware", "strings", "knobs", "strap",
                                          "cable", "pick", "inlay", "dark", "guard_core", "frets"))}
ROLE_OF_MAT = {"guard_core": "dark", "frets": "hardware"}      # materials that take another role's colour


class Pal(CP.Pal):
    """The guitar's colour roles (`ROLES`) over the palette; a role may default to another role (knobs = pickguard)."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}

    def lin(self, spec):
        seen = 0
        while spec in self.specs and seen < 8:
            spec, seen = self.specs[spec], seen + 1
        return super().lin(spec)


# ===================================================================================================================
# the neck, frets and strings
# ===================================================================================================================
SCALE = 0.648                       # saddle line to nut, m
N_FRETS = 22
Z_NUT = SCALE
Y_BOARD = -0.011                    # the board's crown plane: the neck sits 11 mm proud of the body's face
BODY_T = 0.045                      # body thickness: y in [0, BODY_T]
FRET_H = 0.0012                     # a wire stands this far out of the board
FRET_W = 0.0026                     # crown width
BOARD_R = 0.2413                    # 9.5 inch radius
BOARD_T = 0.0060                    # rosewood slab thickness at the crown (the neck wood starts under it)
W_NUT, W_LAST = 0.0420, 0.0560      # board / neck width at the nut and at the last fret
D_NUT, D_LAST = 0.0205, 0.0250      # depth from the board's crown to the back at the centre line (nut, last fret)
P_SECTION = 2.6                     # superellipse exponent of the back of the neck (2 = ellipse, 2.6 a C)
NUT_T, NUT_H = 0.0034, 0.0036       # nut thickness along z (z in [SCALE, SCALE + NUT_T]) and its top above the board's crown


def fret_z(n):
    """z of fret wire n (0 = the nut) on the nominal scale."""
    return SCALE * 2.0 ** (-np.asarray(n, float) / 12.0) if not isinstance(n, int) else SCALE * 2.0 ** (-n / 12.0)


Z_LAST = fret_z(N_FRETS)
Z_NUT_BACK = Z_NUT + NUT_T
Z_HEEL = Z_LAST - 0.0040            # the board and the neck end here, inside the body's neck pocket


def section_wd(z):
    """(width, depth) of the neck solid at z: linear between the nut and the last fret, held beyond (the card's `section`)."""
    t = np.clip((Z_NUT - np.asarray(z, float)) / (Z_NUT - Z_LAST), 0.0, 1.0)
    return W_NUT + t * (W_LAST - W_NUT), D_NUT + t * (D_LAST - D_NUT)


def board_sag(x):
    """How far the board's surface falls below the crown at x (its 9.5 inch radius)."""
    x = np.asarray(x, float)
    return BOARD_R - np.sqrt(np.maximum(BOARD_R ** 2 - x ** 2, 0.0))


def back_half_width(v, w, d, p=P_SECTION):
    """Half width of the neck's back at depth v below the crown plane: |2x/w|^p + (v/d)^p = 1 (v in 0..d)."""
    r = np.clip(1.0 - (np.asarray(v, float) / d) ** p, 0.0, None)
    return 0.5 * w * r ** (1.0 / p)


STRING_NAMES = ("E", "A", "D", "G", "B", "e")                          # low E first: x increasing
GAUGE_IN = (0.046, 0.036, 0.026, 0.017, 0.013, 0.010)                  # a light-medium set
STRING_R = tuple(0.5 * 0.0254 * g for g in GAUGE_IN)
NUT_PITCH, BRIDGE_PITCH = 0.0070, 0.0105                               # string spacing at the nut and the saddles
NUT_X = tuple((i - 2.5) * NUT_PITCH for i in range(6))
SADDLE_X = tuple((i - 2.5) * BRIDGE_PITCH for i in range(6))
SETBACK = (0.0055, 0.0035, 0.0045, 0.0015, 0.0035, 0.0005)             # saddle z = -setback (intonation stagger)
ACTION12 = (0.0024, 0.0022, 0.0020, 0.0019, 0.0017, 0.0016)            # clearance over the 12th wire's crown, graded
NUT_CLEAR = 0.0005                                                     # a string's underside clears the first wire's crown by this at the nut


def h_nut(i):
    """Height of string i's axis over the board's crown plane at the nut (its slot is cut to the same clearance over the first
    wire: the thick strings sit higher)."""
    return FRET_H + NUT_CLEAR + STRING_R[i]


def string_end_heights():
    """(h_nut, h_saddle) of every string axis above the board's crown plane: a straight line from the nut slot to the saddle
    that clears the 12th wire's crown by ACTION12 (the line rises toward the bridge, so the saddle stands higher than the
    11 mm the neck sits proud: the neck has no tilt)."""
    out = []
    for i in range(6):
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
# control points of the body outline (x, z), counter-clockwise from the tail centre: the treble side up to the short horn and
# its deep bay, the neck pocket's lobe, the bass bay, the long horn, the bass side down. The spline runs through them.
HALF_LOWER = ((0.0, -0.1155), (0.030, -0.1152), (0.058, -0.1130), (0.082, -0.1085), (0.103, -0.1010), (0.121, -0.0905),
              (0.135, -0.0780), (0.1455, -0.0640), (0.1520, -0.0495), (0.1558, -0.0350), (0.1572, -0.0200), (0.1566, -0.0050),
              (0.1540, 0.0100), (0.1495, 0.0250), (0.1440, 0.0400), (0.1375, 0.0550), (0.1305, 0.0700), (0.1235, 0.0850),
              (0.1178, 0.0980), (0.1140, 0.1100), (0.1124, 0.1200))      # tail centre to the waist, symmetric about x = 0
TREBLE_UPPER = ((0.1124, 0.1290), (0.1136, 0.1360), (0.1160, 0.1450), (0.1198, 0.1550), (0.1245, 0.1650), (0.1284, 0.1750),
                (0.1320, 0.1850), (0.1346, 0.1950), (0.1360, 0.2050), (0.1364, 0.2150), (0.1355, 0.2250), (0.1332, 0.2350),
                (0.1292, 0.2440), (0.1240, 0.2510), (0.1170, 0.2550), (0.1105, 0.2565), (0.1050, 0.2540), (0.1005, 0.2480),
                (0.0980, 0.2400),                                          # the short horn's tip, round
                (0.0968, 0.2330), (0.0948, 0.2250), (0.0910, 0.2150), (0.0830, 0.2060), (0.0710, 0.1995), (0.0560, 0.1965),
                (0.0420, 0.1975), (0.0320, 0.2010),                        # its bay
                (0.0262, 0.2090))                                          # the neck pocket's right wall (a sharp corner)
LOBE = ((0.0262, 0.2260), (0.0262, 0.2440), (0.0225, 0.2517), (0.0140, 0.2545), (0.0, 0.2548), (-0.0140, 0.2545),
        (-0.0225, 0.2517), (-0.0272, 0.2470))                             # the pocket end: under the neck, corners rounded
BASS_UPPER = ((-0.0330, 0.2430), (-0.0450, 0.2405), (-0.0555, 0.2398), (-0.0650, 0.2410), (-0.0730, 0.2460), (-0.0810, 0.2530),
              (-0.0870, 0.2610), (-0.0915, 0.2700), (-0.0938, 0.2790), (-0.0946, 0.2880), (-0.0950, 0.2980),
              (-0.0962, 0.3070), (-0.0990, 0.3120), (-0.1040, 0.3140), (-0.1100, 0.3128), (-0.1160, 0.3090),   # long horn tip
              (-0.1220, 0.3040), (-0.1280, 0.2990), (-0.1322, 0.2950), (-0.1355, 0.2850), (-0.1379, 0.2750), (-0.1390, 0.2650),
              (-0.1397, 0.2550), (-0.1394, 0.2450), (-0.1382, 0.2350), (-0.1363, 0.2250), (-0.1337, 0.2150),
              (-0.1309, 0.2050), (-0.1284, 0.1950), (-0.1256, 0.1850), (-0.1221, 0.1750), (-0.1186, 0.1650),
              (-0.1158, 0.1550), (-0.1140, 0.1450), (-0.1130, 0.1350), (-0.1126, 0.1290))
OUTLINE_CORNERS = (len(HALF_LOWER) + len(TREBLE_UPPER) - 1,)             # index of the pocket wall corner in the list below


def body_control_points():
    """The outline's control points (n, 2), counter-clockwise, and the indices where it turns sharply."""
    right = list(HALF_LOWER) + list(TREBLE_UPPER)
    left_lower = [(-x, z) for x, z in reversed(HALF_LOWER[1:])]
    pts = right + list(LOBE) + list(BASS_UPPER) + left_lower
    return np.array(pts, float), OUTLINE_CORNERS


# strap buttons: where the button leaves the body, and the way it points
STRAP_TOP = {"point": (-0.1040, BODY_T / 2, 0.3140), "dir": (-0.15, 0.0, 0.9887)}   # the long horn's tip edge
STRAP_BOTTOM = {"point": (0.0, BODY_T / 2, -0.1155), "dir": (0.0, 0.0, -1.0)}       # the tail edge, on the centre line
BUTTON_H = 0.0155                   # a strap button stands this far out of the edge

# the contours (see electric_guitar_body): the forearm bevel on the bass edge of the front, the belly cut on the back
FOREARM = {"depth": 0.0080, "width": 0.045, "z": (0.020, 0.070, 0.165, 0.205)}      # ramps in at z 0.02..0.07, out at 0.165..0.205
BELLY = {"center": (-0.095, 0.050), "radii": (0.075, 0.125), "depth": 0.0115}       # elliptical scoop on the back, bass side
BODY_EDGE_R = 0.0045                # the body's rolled edge (front and back)

# the pickguard (11 holes): control points (x, z), counter-clockwise from the bridge side
GUARD_T = 0.0030
GUARD = ((-0.0482, 0.0100), (-0.0100, 0.0100), (0.0357, 0.0098), (0.0600, 0.0057), (0.0778, -0.0032), (0.0957, -0.0126),
         (0.1118, -0.0215), (0.1250, -0.0262), (0.1370, -0.0235), (0.1450, -0.0140), (0.1475, -0.0020), (0.1465, 0.0130),
         (0.1425, 0.0260),
         (0.1385, 0.0360), (0.1325, 0.0500), (0.1255, 0.0640), (0.1185, 0.0780), (0.1122, 0.0900), (0.1068, 0.1030),
         (0.1052, 0.1170), (0.1058, 0.1290), (0.1078, 0.1400), (0.1106, 0.1500), (0.1145, 0.1600), (0.1188, 0.1700),
         (0.1226, 0.1800), (0.1258, 0.1900), (0.1278, 0.2000), (0.1290, 0.2100), (0.1288, 0.2200), (0.1270, 0.2300),
         (0.1238, 0.2390), (0.1190, 0.2455), (0.1136, 0.2490), (0.1085, 0.2475), (0.1050, 0.2420),
         (0.1030, 0.2340), (0.1006, 0.2250), (0.0966, 0.2150), (0.0890, 0.2025), (0.0770, 0.1945), (0.0630, 0.1912),
         (0.0480, 0.1905), (0.0380, 0.1915),
         (0.0292, 0.1900), (0.0292, 0.1830),                                # down the neck's right edge to the heel end
         (-0.0292, 0.1830), (-0.0292, 0.1920), (-0.0292, 0.2080), (-0.0330, 0.2120), (-0.0420, 0.2125),
         (-0.0500, 0.2105), (-0.0538, 0.2050), (-0.0545, 0.1900), (-0.0540, 0.1600), (-0.0540, 0.1300), (-0.0562, 0.1100),
         (-0.0620, 0.0950), (-0.0700, 0.0790), (-0.0739, 0.0620), (-0.0725, 0.0450), (-0.0665, 0.0290), (-0.0600, 0.0150))
GUARD_CORNERS = tuple(GUARD.index(c) for c in ((0.0292, 0.1900), (0.0292, 0.1830), (-0.0292, 0.1830), (-0.0292, 0.2080)))

GUARD_SCREWS = ((-0.0475, 0.2060), (0.0350, 0.1805), (0.1086, 0.1714), (0.1164, 0.2300), (-0.0525, 0.1121), (-0.0679, 0.0629),
                (-0.0507, 0.0139), (0.0429, 0.0154), (0.1236, 0.0343), (0.1350, -0.0175))        # and the switch's two
# pickups: centre (x = 0, z), slant (degrees, + turns the bass end toward the neck)
PICKUP_L, PICKUP_W = 0.0700, 0.0185          # cover length (across) and width (along the strings)
PICKUP_H = 0.0078                            # cover height above the guard
POLE_R, POLE_PITCH, POLE_H = 0.0024, 0.0105, 0.0014
PICKUPS = {"neck": {"z": 0.1590, "slant": 0.0}, "middle": {"z": 0.1000, "slant": 0.0}, "bridge": {"z": 0.0450, "slant": 9.3}}
PICKUP_SCREW_X = 0.0386                      # the height screws stand outside the covers' ends

# bridge: a vintage-style tremolo plate with six saddles
BRIDGE = {"center": (0.0, -0.0010), "size": (0.0750, 0.0425), "t": 0.0030, "corner": 0.0045}
SADDLE = {"w": 0.0090, "l": 0.0150, "top": 0.0055}
NECK_PLATE_Z, COVER_Z = 0.2200, -0.0300      # centres of the neck plate and of the tremolo cavity's cover on the back
BRIDGE_HOLE_Z = -0.0185                      # the strings go down through the plate here, behind the saddles

# controls: knobs (x, z), the five-way switch, the jack
KNOBS = ((0.0539, 0.0416), (0.0950, 0.0204), (0.1256, -0.0056))          # volume, tone 1, tone 2
KNOB_R, KNOB_H = 0.0095, 0.0165
SWITCH = {"center": (0.0945, 0.0628), "angle": -47.0, "slot": (0.0355, 0.0030)}   # slot length, width; angle of the slot from +x
JACK = {"center": (0.0905, -0.0479), "angle": -35.6, "plate": (0.0715, 0.0330), "t": 0.0035, "elev": 35.0,
        "mouth": 0.012, "plug_len": 0.052}

# the headstock: (u, v) = (distance from the nut's front face along z, x) in metres, bass (min x) and treble (max x) edge
HEAD_BASS = ((0.0034, -0.0210), (0.0084, -0.0220), (0.0148, -0.0265), (0.0213, -0.0314), (0.0277, -0.0338), (0.0342, -0.0343),
             (0.0681, -0.0257), (0.1003, -0.0149), (0.1326, -0.0049), (0.1632, 0.0054), (0.1729, 0.0106), (0.1794, 0.0187),
             (0.1826, 0.0267))
HEAD_TREBLE = ((0.0034, 0.0210), (0.0084, 0.0213), (0.0116, 0.0226), (0.0277, 0.0331), (0.0423, 0.0428), (0.0495, 0.0447),
               (0.0681, 0.0415), (0.1003, 0.0367), (0.1165, 0.0339), (0.1294, 0.0320), (0.1318, 0.0356), (0.1366, 0.0412),
               (0.1423, 0.0477), (0.1487, 0.0525), (0.1568, 0.0538), (0.1648, 0.0525), (0.1729, 0.0476), (0.1794, 0.0412),
               (0.1826, 0.0331))
HEAD_L = 0.1826                              # nut front face to the tip
HEAD_P = 6.0                                 # the headstock's section exponent (a rounded rectangle)
POSTS = tuple((SCALE + u, v) for u, v in ((0.0423, -0.0214), (0.0648, -0.0140), (0.0877, -0.0069), (0.1111, 0.0001),
                                          (0.1342, 0.0070), (0.1568, 0.0138)))     # tuner posts (z, x), string 6 first
POST_R = 0.0030                              # a tuner post's radius
KEY_DIR = (-0.94, 0.0, 0.34)                 # from a post toward its key button (x, y, z): the keys stick out of the bass edge
KEY_LEN = 0.022
TREES = ((SCALE + 0.0584, -0.0015), (SCALE + 0.0713, 0.0130))              # two round string retainers (z, x)
TRUSS = {"center": (0.0, SCALE + 0.0226), "size": (0.0105, 0.0215)}       # the truss rod's cover on the headstock face

# the plectrum (351 shape) in the pinch frame
PICK = {"thickness": 0.0008, "length": 0.031, "width": 0.026, "tip": 0.008}
PICK_REST = {"at": (0.0440, 0.0750)}         # x, z where it lies on the pickguard (tip toward +z)
STRUM_ZONE = (0.0625, 0.0880)                # between the bridge and middle pickups' covers

# cable (a bevelled curve) and its end
CABLE_R = 0.0032
FLOOR_Z = -0.14                              # the upright guitar's floor: the cord ends here by default

WEAR = {"name": "stand", "bone": "upper_body2",
        "pivot": [0.0, 0.045, 0.0],
        "ref": {"top": 1.70, "shoulder_width": 0.18},
        "at": [-0.14, -0.10, 0.0], "scale": ["shoulder_width", "top", "top"],
        "neck_deg": 35.0, "yaw_deg": 0.0, "roll_deg": 0.0,
        "strap": {"top": "strap_top", "bottom": "strap_bottom", "over": "shoulder.L", "width": 0.05, "thickness": 0.004,
                  "material": "<name>_strap"},
        "cable": {"object": "<name>_cable", "anchor": "jack", "radius": 0.0032}}

COLLIDERS = {                                # hidden objects: key -> (centre, size)  (box specs use the object's bounds)
    "body": ((0.0, 0.0225, 0.040), (0.314, 0.045, 0.310)),
    "horn": ((-0.1170, 0.0225, 0.2650), (0.0450, 0.045, 0.1000)),
    "neck": ((0.0, 0.0025, 0.4100), (0.056, 0.026, 0.4700)),
    "head": ((0.0050, -0.0010, 0.7400), (0.0850, 0.0160, 0.1800)),
}
NECK_CAPSULE = {"a": (0.0, 0.0025, 0.1800), "b": (0.0, 0.0025, 0.6500), "R": 0.0300}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def jack_geometry():
    """(mouth, dir, end): the socket's mouth on the face, the unit vector out of the socket (the plug's axis: along the oval
    plate's long axis, raised `elev` degrees out of the face) and the plug's end where the cord leaves."""
    a, e = math.radians(JACK["angle"]), math.radians(JACK["elev"])
    t = np.array([math.cos(a), 0.0, math.sin(a)])
    c = np.array([JACK["center"][0], 0.0, JACK["center"][1]])
    mouth = c - t * JACK["mouth"] + np.array([0.0, -0.0036, 0.0])
    d = _unit(math.cos(e) * t + math.sin(e) * np.array([0.0, -1.0, 0.0]))
    return mouth, d, mouth + d * JACK["plug_len"]


def strum_center():
    """The pick tip's rest point: centred across the six strings, on their plane, between the pickups."""
    z = 0.5 * sum(STRUM_ZONE)
    y = float(np.mean([string_at(i, z)[1] for i in range(6)]))
    return (0.0, y, z)


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
    for i in range(6):
        a, b = string_ends(i)
        strings.append({"name": STRING_NAMES[i], "nut": _r(a), "bridge": _r(b), "radius": round(STRING_R[i], 6)})
    center = strum_center()
    mouth, jdir, jend = jack_geometry()
    top, bot = STRAP_TOP, STRAP_BOTTOM
    neck = {"name": "neck", "type": "neck",
            "frame": {"along": [0, 0, 1], "across": [1, 0, 0], "normal": [0, -1, 0]},
            "thumb": [0, 1, 0], "scale": SCALE,
            "frets": [[0.0, Y_BOARD, round(z, 6)] for z in zs],
            "strings": strings,
            "section": {"width": [W_NUT, W_LAST], "depth": [D_NUT, D_LAST], "p": P_SECTION,
                        "board_width": [W_NUT, W_LAST], "board_radius": BOARD_R},
            "fret_height": FRET_H, "object": f"{name}_neck"}
    strum = {"name": "strum", "type": "strum", "center": _r(center), "along": [0, 0, 1], "across": [1, 0, 0],
             "normal": [0, -1, 0], "zone": list(STRUM_ZONE),
             "pick": {"object": f"{name}_pick", "thickness": PICK["thickness"], "length": PICK["length"],
                      "width": PICK["width"], "tip": PICK["tip"]}}
    anchors = [{"name": "strap_top", "point": _r(top["point"]), "dir": _r(_unit(top["dir"])), "object": f"{name}_strap_top"},
               {"name": "strap_bottom", "point": _r(bot["point"]), "dir": _r(_unit(bot["dir"])),
                "object": f"{name}_strap_bottom"},
               {"name": "jack", "point": _r(jend), "dir": _r(jdir), "object": f"{name}_jack"},
               {"name": "head", "point": _r((0.0, -0.0050, SCALE + 0.55 * HEAD_L))}]
    looks = [{"name": "neck", "point": _r((0.0, Y_BOARD, float(fret_z(5))))},
             {"name": "strum", "point": _r(center)},
             {"name": "head", "point": _r((0.0, -0.0050, SCALE + 0.55 * HEAD_L))}]
    cols = [{"type": "box", "object": f"{name}_col_{k}", "rnd": 0.008, "tag": name} for k in ("body", "horn", "head")]
    cols.append({"type": "capsule", "object": f"{name}_col_neck", "a": _r(NECK_CAPSULE["a"]), "b": _r(NECK_CAPSULE["b"]),
                 "R": NECK_CAPSULE["R"], "tag": name})
    out = {"size": _r(size if size is not None else (0.315, 0.09, 0.96), 4), "origin": "bridge", "front": "-Y",
           "slots": dict(roles) if roles is not None else dict(ROLES),
           "use": {"grip": [neck, strum], "anchor": anchors, "look": looks, "wear": [wear_entry(name)]},
           "colliders": cols}
    return out
