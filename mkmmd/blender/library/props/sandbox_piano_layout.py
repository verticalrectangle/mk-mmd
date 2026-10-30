"""Layout of the sandbox piano (pure Python + numpy, no bpy: importable from tests): every number the parts agree on, the
colour roles, the sand's surface, the piano's plan, the lid's hinge and the prop card.

A children's wooden sandbox with a toy grand piano in it, the cartoon homage to a piano kept in a sandbox: chunky,
rounded, candy-coloured, at a size a character sits at and plays.

Frame (metres; everything in the card is in it): ORIGIN on the floor at the sandbox's middle, Z up, the front toward -Y
(card `front` = "-Y"), X to the right seen from the front. The pianist sits on the bench facing -Y, toward the keyboard,
the piano's body beyond it to -Y: her left (+X) is the bass end and the straight spine, her right (-X) the treble end and
the curved bentside, the side the lid opens to.

The sandbox: BOX_W x BOX_D outside, each side two stacked boards (BOARD_H tall, BOARD_T thick, every edge rounded) up to
BOARD_TOP, a triangular seat on each corner, the sand at Z_SAND with gentle bumps, banked against the boards, heaped a
little round every leg that goes into it, and flat where her feet go (`sand_height`).

The piano (in proportion on toy grands: a 2-octave keyboard of chunky keys, C to C, the white keys' top KEYS_Z = 0.70 m
over the sand, so a seated character's knees go under the key bed): a rim on the grand's plan (`plan`: the straight spine,
the S of the bentside, the rounded tail) with a rolled top and a coved soundboard inside, a gold plate and strings, the lid
hinged along the spine and propped open LID_DEG on its stick, the music desk, the keyboard between two cheek blocks over
the key bed and the key slip, three turned legs and the pedal lyre. The bench (seat SEAT_Z = 0.45 m over the sand): a
domed cushion on a rounded rail and four turned legs.

Colour roles (`slots` of the prop: a palette slot, a hex, a blend "gold:2,rose:1" or another role; `ink` and `paper` are
the palette's darkest and lightest; no role is named after a palette slot): wood (the sandbox's boards), seats (its
corner seats), sand, piano (the case, legs' collars, cheeks, key bed), lid, inside (the soundboard), plate (the plate,
strings, hinges, pedals), trim (legs, lyre, stick, desk), keys_white, keys_black, bench (the cushion), bench_wood (its rail
and legs)."""
import itertools
import math

import numpy as np

from ....core import shell as S
from .record_treadmill_layout import Pal as _Pal

WHO = "sandbox_piano"

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = {
    "wood": "gold:3,rose:2,text:1",     # the sandbox's boards: a warm wood
    "seats": "rose",                    # its corner seats
    "sand": "gold:3,paper:2,muted:1",   # warm sand
    "piano": "love",                    # the case: candy lacquer
    "lid": "piano",
    "inside": "gold:2,paper:1",         # the soundboard
    "plate": "gold",                    # the plate, strings, hinges, pedals
    "trim": "gold:3,paper:1",           # legs, lyre, the lid's stick, the music desk
    "keys_white": "paper",
    "keys_black": "ink:2,text:1",
    "bench": "iris",                    # the bench's cushion
    "bench_wood": "piano",              # its rail and legs
}
MATS = {name: i for i, name in enumerate(ROLES)}


def Pal(slots=None):
    """The sandbox piano's colour roles over the palette (see `record_treadmill_layout.Pal`)."""
    return _Pal(slots, ROLES, WHO)


# ===================================================================================================================
# the sandbox
# ===================================================================================================================
BOX_W, BOX_D = 2.20, 2.40           # outside, x and y
BOARD_T = 0.05                      # a board's thickness
BOARD_H = 0.15                      # its height; two stacked make a side
BOARD_TOP = 2 * BOARD_H
BOARD_R = 0.020                     # the rounding of its long edges (both faces)
BOARD_CORNER = 0.030                # the corners of its outline
SEAT_LEG = 0.46                     # a corner seat: a right triangle this long along each side
SEAT_T = 0.035
SEAT_OVER = 0.015                   # it overhangs the boards outward by this much
INNER_W, INNER_D = BOX_W - 2 * BOARD_T, BOX_D - 2 * BOARD_T     # the sand's area
Z_SAND = 0.22                       # the sand's level
SAND_TUCK = 0.012                   # the sand's edge runs this far into the boards (no gap shows)
SINK = 0.035                        # legs go this far into the sand

# ===================================================================================================================
# the piano
# ===================================================================================================================
KEYS_Z = Z_SAND + 0.70              # the white keys' top
KEY_Y0 = 0.30                       # the white keys' front edge
WHITE = 15                          # C .. C over two octaves; the lowest C at the +X end (her left)
KEY_PITCH = 0.057                   # white keys' pitch across, and their size
KEY_W, KEY_L, KEY_H = 0.052, 0.150, 0.028
KEY_R = 0.014                       # the corners of a key's outline
BLACK_AFTER = (0, 1, 3, 4, 5)       # white keys of an octave (C = 0) a black key follows
BLACK_W, BLACK_L, BLACK_H = 0.030, 0.092, 0.024
BLACK_RISE = 0.020                  # a black key's top over the white keys' top
KEYS_W = WHITE * KEY_PITCH          # the keyboard's width
BED_T = 0.045                       # the key bed under the keys
Z_BED0 = KEYS_Z - KEY_H - BED_T     # its underside
CASE_FRONT = KEY_Y0 - KEY_L - 0.020     # the rim's front face (y): the keys end 20 mm short of it
CASE_W = KEYS_W + 2 * 0.080         # the case's width: the keyboard and two cheek blocks
CASE_L = 1.10                       # the case's length from its front to the tail
RIM_Z0 = Z_BED0 - 0.060             # the case's underside: a deep toy case
RIM_Z1 = KEYS_Z + 0.170             # the rim's top
RIM_T = 0.045                       # the rim's wall
RIM_R = 0.020                       # its rolled top edge (and the bottom's)
WELL = 0.140                        # the soundboard this far under the rim's top
COVE = 0.020                        # the cove where the soundboard meets the wall
CHEEK_W = 0.070                     # a cheek block's thickness (x)

# the plan: (x, back) points, `back` = metres behind the case's front face; the straight spine on +X, the bentside's S on
# -X from the treble shoulder round the rounded tail (convex at the shoulder, concave in the middle, convex at the tail)
PLAN = ((0.5, 0.0), (0.5, 0.86), (0.49, 0.97), (0.42, 1.06), (0.30, 1.10), (0.16, 1.08), (0.03, 1.00), (-0.07, 0.88),
        (-0.15, 0.72), (-0.23, 0.56), (-0.32, 0.42), (-0.43, 0.30), (-0.5, 0.16), (-0.5, 0.0))
PLAN_ROUND = (0.13, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.15, 0.13)   # fillet per point

LID_DEG = 38.0                      # the lid open, turned about its hinge line along the spine
LID_T = 0.025
LID_OVER = 0.012                    # the lid overhangs the rim by this much
HINGE = (CASE_W / 2 + LID_OVER, RIM_Z1)     # the hinge line (x, z), along Y
STICK_BACK = 0.52                   # the stick stands on the bentside's rim this far behind the front
STICK_R = 0.013

LEG_R = 0.075                       # a leg's widest turning
LEGS = ((0.40, 0.12), (-0.40, 0.12), (0.30, 0.92))   # (x share of the width, back) of the two front legs and the tail's
LYRE_Y = CASE_FRONT - 0.170         # the lyre hangs from the case's underside, in the middle, behind her knees
PEDAL_Z = Z_SAND + 0.075            # the pedals' tops
DESK = {"w": 0.54, "h": 0.15, "t": 0.020, "back": 0.040, "tilt": 16.0}   # the music desk on the rim's front

# the bench and the pianist: a seated character's shoulder-to-wrist reach is short (0.37 m on a 1.55 m model), so the
# hips sit HIP_TO_KEYS behind the keys' front edge, the knees under the key bed, the ankles a little in front of the
# knees with the toes over the pedals
SEAT_Z = Z_SAND + 0.45              # the bench's seat (the cushion's top)
HIP_TO_KEYS = 0.245
BENCH_Y = KEY_Y0 + HIP_TO_KEYS + 0.020     # its middle
BENCH_W, BENCH_D = 0.62, 0.34
CUSHION_T = 0.065
BENCH_LEGS = 0.24, 0.12             # its legs at (+-x, middle +- y)
HIP = (0.0, KEY_Y0 + HIP_TO_KEYS, SEAT_Z + 0.075)    # where the hip joints go: on the front half of the bench
FEET = {"L": (0.11, KEY_Y0 - 0.120), "R": (-0.11, KEY_Y0 - 0.120)}   # ankles, her left at +X, under the keys


def _plan_closed():
    P = np.array([(x * CASE_W, CASE_FRONT - b) for x, b in PLAN])
    return P, np.array(PLAN_ROUND)


def plan(n=240):
    """The case's plan: the closed counter-clockwise outline (n, 2) in the prop's frame, evenly spaced."""
    P, r = _plan_closed()
    if S.polygon_area(P) < 0:
        P, r = P[::-1], r[::-1]
    return S.resample_loop(S.fillet(P, r, 8, closed=True), n)


def plan_back(x, P=None):
    """The y of the plan's back edge (the bentside or the tail) at x: the smallest y of the outline there."""
    P = plan() if P is None else P
    Q = np.vstack([P, P[:1]])
    ys = []
    for (x0, y0), (x1, y1) in itertools.pairwise(Q):
        if (x0 - x) * (x1 - x) <= 0 and x0 != x1:
            ys.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return min(ys)


def white_x(i):
    """The centre x of white key i (0 the lowest C, at the +X end)."""
    return KEYS_W / 2 - (i + 0.5) * KEY_PITCH


def black_keys():
    """[(white index it follows, centre x)] of the black keys: on the line between their two white keys."""
    out = []
    for i in range(WHITE - 1):
        if i % 7 in BLACK_AFTER:
            out.append((i, KEYS_W / 2 - (i + 1) * KEY_PITCH))
    return out


def lid_matrix(deg=LID_DEG):
    """3x3 rotation of the lid about its hinge line (+Y): positive lifts the treble side (-X)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def lid_under_z(x, deg=LID_DEG):
    """The z of the open lid's underside over x (the hinge line at HINGE)."""
    return HINGE[1] + (HINGE[0] - x) * math.tan(math.radians(deg))


# ===================================================================================================================
# the sand
# ===================================================================================================================
MOUNDS = ((0.62, -0.78, 0.30, 0.050), (-0.78, 0.52, 0.26, 0.042), (0.84, 0.70, 0.20, 0.034), (-0.70, -0.88, 0.24, -0.022),
          (0.05, -1.00, 0.30, 0.024), (-0.86, -0.20, 0.22, 0.032), (0.80, -0.10, 0.18, -0.016), (-0.35, 0.95, 0.20, 0.022),
          (-0.62, -0.55, 0.11, 0.045), (0.70, 0.30, 0.12, 0.030))
BANK = (0.016, 0.22)                # sand banked up against the boards: height and reach
HEAP = (0.018, 0.075)               # heaped round a leg: height and spread (from its surface)
FLAT = 0.20                         # the sand is level within this of her ankles


def leg_spots():
    """[(x, y, radius)] where something goes into the sand: the piano's legs and the bench's."""
    out = [(x * CASE_W, CASE_FRONT - b, LEG_R * 0.75) for x, b in LEGS]
    bx, by = BENCH_LEGS
    out += [(sx * bx, BENCH_Y + sy * by, 0.030) for sx in (-1, 1) for sy in (-1, 1)]
    return out


def sand_height(x, y):
    """The sand's surface z at (x, y) (arrays or numbers): Z_SAND, gentle mounds and ripples, a bank against the boards,
    heaps round the legs, level round her feet."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = 0.008 * np.sin(5.1 * x + 1.3) * np.sin(4.3 * y + 0.7) + 0.005 * np.sin(9.7 * x - 7.1 * y + 0.4)
    for mx, my, r, a in MOUNDS:
        h = h + a * np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (r * r))
    wall = np.minimum(INNER_W / 2 - np.abs(x), INNER_D / 2 - np.abs(y))
    h = h + BANK[0] * np.clip(1.0 - wall / BANK[1], 0.0, 1.0) ** 2
    for lx, ly, lr in leg_spots():
        d = np.maximum(np.hypot(x - lx, y - ly) - lr, 0.0)
        h = h + HEAP[0] * np.exp(-(d / HEAP[1]) ** 2)
    for fx, fy in FEET.values():
        t = np.clip(np.hypot(x - fx, y - fy) / FLAT - 1.0, 0.0, 1.0)      # 0 within FLAT, 1 from 2 FLAT
        h = h * (t * t * (3.0 - 2.0 * t))
    return Z_SAND + h


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, roles=None):
    """The prop card (docs/design.md: Prop card): the bench's seat and feet (as the cafe chair's), the keyboard's front edge
    and the sand to rest on, the keys to look at."""
    x0 = KEYS_W / 2 - 0.02
    use = {"sit": [{"name": "bench", "hip": _r(HIP), "seat_z": round(SEAT_Z, 5), "facing": [0.0, -1.0, 0.0],
                    "floor_z": round(Z_SAND, 5), "pelvis_deg": 6.0, "back_deg": 0.0}],
           "feet": [{"name": "bench", "L": [*_r(FEET["L"]), None], "R": [*_r(FEET["R"]), None],
                     "floor_z": round(Z_SAND, 5)}],
           "rest": [{"name": "keys", "type": "edge", "a": _r((-x0, KEY_Y0 - 0.030, KEYS_Z)),
                     "b": _r((x0, KEY_Y0 - 0.030, KEYS_Z)), "normal": [0.0, 0.0, 1.0]},
                    {"name": "sand", "type": "plane", "center": _r((0.0, 0.0, Z_SAND)), "normal": [0.0, 0.0, 1.0],
                     "size": _r((INNER_W, INNER_D))}],
           "look": [{"name": "keys", "point": _r((0.0, KEY_Y0 - 0.060, KEYS_Z))}]}
    a = math.radians(LID_DEG)                                  # the open lid's treble edge is the top
    top = HINGE[1] + (CASE_W + 2 * LID_OVER) * math.sin(a) + LID_T * math.cos(a)
    return {"size": _r((BOX_W, BOX_D, top), 4), "origin": "floor_center", "front": "-Y", "slots": dict(roles or {}),
            "use": use, "colliders": [], "params": {},
            "objects": [f"{name}_{k}" for k in ("boards", "seats", "sand", "case", "lid", "inside", "keys", "trim",
                                                 "bench")]}
