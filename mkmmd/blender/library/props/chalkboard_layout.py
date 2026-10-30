"""Layout of the chalkboard (pure Python + numpy, no bpy: importable from tests): every number the parts agree on, the
colour roles, the meshes, the chalk drawing and the prop card.

A school chalkboard on an A-frame easel, chunky and rounded: two front legs splayed like an A and joined at the top by
the hinge's barrel (a turned knob on it), the back leg hinged there and running back to the floor, a crossbar low down,
and across the front legs the ledge: a shelf with a lip, the chalk tray. The board (a green-black slate SLATE_W x
SLATE_H in a rounded wooden frame) stands on the ledge and leans back LEAN degrees against the front legs; in the tray a
white and a pink chalk stick, a stub and a felt eraser. The slate's faint chalk dust is drawn by its material.

Frame (metres; everything in the card is in it): ORIGIN on the floor at the middle of the easel's footprint, Z up, the
front (the slate's face) toward -Y (card `front` = "-Y"), X to the right seen from the front. The board's own frame: x
across (+X), `UP` along the slate (up and back), `NORMAL` out of the slate (toward the viewer and up), origin `CENTRE`
the slate's middle on its face.

Option `drawing`: "none" (default) or "noughts", a game of noughts and crosses in chalk on the slate (`drawing_strokes`):
the grid, a cross in the middle, a nought and a cross along the top, a nought at the bottom left. Chalk strokes are a
graphic layer (exempt from `form`).

Colour roles (`slots` of the prop: a palette slot, a hex, a blend "gold:2,rose:1" or another role; `ink` and `paper` are
the palette's darkest and lightest; no role is named after a palette slot): slate, wood (the easel and the frame), chalk
(the white stick and the drawing), chalk_pink, felt (the eraser's pad), eraser (its wooden back)."""
import math

import numpy as np

from ....core import shell as S
from .record_treadmill_layout import Pal as _Pal

WHO = "chalkboard"

ROLES = {
    "slate": "#26332d",                 # a painted chalkboard's green-black (the palette has no green)
    "wood": "gold:3,rose:2,text:1",     # the easel and the frame: a warm wood
    "chalk": "paper",
    "chalk_pink": "love:1,paper:2",
    "felt": "muted:2,ink:1",
    "eraser": "pine",
}
MATS = {name: i for i, name in enumerate(ROLES)}
DRAWINGS = ("none", "noughts")


def Pal(slots=None):
    """The chalkboard's colour roles over the palette (see `record_treadmill_layout.Pal`)."""
    return _Pal(slots, ROLES, WHO)


def options(slots):
    """The prop's non-colour options from its `slots` (`drawing`), checked."""
    d = dict(slots or {}).get("drawing", "none")
    if d not in DRAWINGS:
        raise ValueError(f"{WHO}: drawing is 'none' (a clean slate) or 'noughts' (noughts and crosses), got {d!r}")
    return {"drawing": d}


# ===================================================================================================================
# numbers
# ===================================================================================================================
LEAN = 12.0                         # the board and the front legs lean back this much from vertical
SLATE_W, SLATE_H = 0.90, 0.65       # the slate inside the frame
SLATE_T = 0.010
FRAME_W, FRAME_D = 0.055, 0.034     # the frame's moulding: its width (in the board's plane) and depth
FRAME_R = 0.0155                    # the moulding's rounded corners
FRAME_BACK = 0.014                  # the moulding's back lies this far behind the slate's face
APEX_Z = 1.585                      # the hinge's axis at the top of the easel
LEG_FOOT_X = 0.40                   # the front legs' feet (x), splayed from the top
LEG_TOP_X = 0.045                   # and their tops
LEG_W, LEG_D = 0.052, 0.038         # a leg's section (across, deep)
LEG_FOOT_Y = -0.318                 # the front legs' centre line at the floor
BACK_FOOT_Y = 0.312                 # the back leg's foot
BARREL_R = 0.030                    # the hinge's barrel across the front legs' tops
CROSS_Z = 0.300                     # the low crossbar between the front legs
TRAY_Z = 0.780                      # the ledge's top: the board stands on it
LIP = (0.018, 0.024)                # the tray's front lip: thickness and height over the ledge
TRAY_L, TRAY_D, TRAY_T = 1.06, 0.120, 0.020    # the ledge's length, depth (from the legs' front faces) and thickness
CHALK_R, CHALK_L = 0.0058, 0.082

_t = math.radians(LEAN)
UP = np.array([0.0, math.sin(_t), math.cos(_t)])       # along the slate, up and back
NORMAL = np.array([0.0, -math.cos(_t), math.sin(_t)])  # out of the slate's face


def leg_line_y(z):
    """The y of the front legs' centre line at height z (they lean back LEAN)."""
    return LEG_FOOT_Y + z * math.tan(_t)


def legs_front_y(z):
    """The y of the plane of the front legs' front faces at height z."""
    return leg_line_y(z) - 0.5 * LEG_D / math.cos(_t)


def leg_x(z):
    """|x| of a front leg's centre line at height z (splayed: LEG_FOOT_X at the floor, LEG_TOP_X at the apex)."""
    return LEG_FOOT_X + (LEG_TOP_X - LEG_FOOT_X) * z / APEX_Z


# the board leans on the legs' front faces with its frame's back, the lowest point of its rounded bottom edge on the ledge
_rest = FRAME_R * (math.cos(_t) + math.sin(_t) - 1.0)   # how far that point is over the edge's sharp corner
_back_bottom = np.array([0.0, legs_front_y(TRAY_Z) - 0.001, TRAY_Z - _rest])
CENTRE = _back_bottom + (SLATE_H / 2 + FRAME_W) * UP + FRAME_BACK * NORMAL
TRAY_BACK_Y = legs_front_y(TRAY_Z)                     # the ledge's back, against the legs


def board_point(a, b, c=0.0):
    """A point of the board's frame: a across, b up along the slate, c out of its face (metres from CENTRE)."""
    return CENTRE + np.array([a, 0.0, 0.0]) + b * UP + c * NORMAL


# ===================================================================================================================
# meshes
# ===================================================================================================================
def _lathe(profile, mat, seg=32, fillets=None, n=4):
    P = np.asarray(profile, float)
    if fillets is not None:
        P = S.fillet(P, fillets, n)
    return S.lathe(P, seg=seg, mat=mat)


def _rod(a, b, w, d, mat, up, taper=(1.0, 1.0), n=12):
    """A straight wooden rod from a to b, its section a rounded w x d superellipse (d along `up` made perpendicular),
    scaled from taper[0] at a to taper[1] at b; flat ends (buried in what they join)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    path = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
    sc = np.linspace(taper[0], taper[1], n)
    return S.sweep(path, S.superellipse(w, d, 2.8, 20), scale=sc, up=up, mat=mat)


def board_mesh():
    """The slate (a thin panel, edges rolled, its face at CENTRE) and the frame round it (a rounded moulding swept round
    the slate's edge), in the prop's frame."""
    slate = S.rounded_panel(S.rrect(SLATE_W + 0.04, SLATE_H + 0.04, 0.02, 4), SLATE_T, r_edge=0.0045, n=3,
                            r_back=0.0045, mat=MATS["slate"])
    slate = S.place(slate.moved((0.0, 0.0, -SLATE_T)), CENTRE, NORMAL, UP)
    loop = S.resample_loop(S.rrect(SLATE_W + FRAME_W, SLATE_H + FRAME_W, 0.035, 8), 160)
    path = np.array([board_point(a, b, FRAME_D / 2 - FRAME_BACK) for a, b in loop])
    frame = S.sweep(path, S.rrect(FRAME_W, FRAME_D, FRAME_R, 4), closed=True, up=tuple(NORMAL), mat=MATS["wood"])
    return S.merge([slate, frame.weld(1e-9)])


def easel_mesh():
    """The two front legs, the back leg, the hinge's barrel and its knob, the crossbar and the ledge with its lip."""
    out = []
    top = APEX_Z - 0.010
    for s in (-1, 1):
        foot = (s * LEG_FOOT_X, leg_line_y(0.0), 0.0)
        head = (s * LEG_TOP_X, leg_line_y(top), top)
        out.append(_rod(foot, head, LEG_W, LEG_D, MATS["wood"], up=tuple(NORMAL), taper=(1.0, 0.82)))
    hinge = np.array([0.0, leg_line_y(APEX_Z) + LEG_D * 0.5 + BARREL_R * 0.6, APEX_Z])
    out.append(_rod((0.0, BACK_FOOT_Y, 0.0), hinge - [0.0, 0.0, 0.01], LEG_W, LEG_D, MATS["wood"], up=(1, 0, 0),
                    taper=(1.0, 0.85)))
    barrel = _lathe([(0.0, -0.11), (BARREL_R, -0.11), (BARREL_R, 0.11), (0.0, 0.11)], MATS["wood"],
                    fillets=[0, 0.012, 0.012, 0]).rot(ry=90.0)
    out.append(barrel.moved((0.0, leg_line_y(APEX_Z) + 0.008, APEX_Z)))
    knob = _lathe([(0.0, 0.0), (0.020, 0.0), (0.030, 0.022), (0.016, 0.040), (0.022, 0.052), (0.0, 0.064)],
                  MATS["wood"], fillets=[0, 0.004, 0.012, 0.006, 0.008, 0])
    out.append(knob.moved((0.0, leg_line_y(APEX_Z) + 0.008, APEX_Z + BARREL_R - 0.010)))
    xc = leg_x(CROSS_Z) + 0.015
    out.append(_rod((-xc, leg_line_y(CROSS_Z), CROSS_Z), (xc, leg_line_y(CROSS_Z), CROSS_Z), 0.032, 0.032,
                    MATS["wood"], up=(0, 0, 1)))
    shelf = S.rounded_panel(S.rrect(TRAY_L, TRAY_D, 0.025, 4), TRAY_T, r_edge=0.008, n=3, r_back=0.008,
                            mat=MATS["wood"])
    out.append(shelf.moved((0.0, TRAY_BACK_Y - TRAY_D / 2 + 0.008, TRAY_Z - TRAY_T)))
    ly = TRAY_BACK_Y - TRAY_D + 0.008 + LIP[0] / 2
    lip = np.stack([np.linspace(-TRAY_L / 2 + 0.006, TRAY_L / 2 - 0.006, 16), np.full(16, ly),
                    np.full(16, TRAY_Z + LIP[1] / 2 - 0.004)], 1)
    out.append(S.sweep(lip, S.rrect(LIP[0], LIP[1] + 0.008, 0.0075, 4), up=(0, 0, 1), mat=MATS["wood"]))
    m = S.merge(out)
    m.V[:, 2] = np.maximum(m.V[:, 2], 0.0)              # the legs' ends stand flat on the floor
    return m


def tray_spot(u):
    """(x, y) of a point in the tray's trough at share u (0 .. 1) of its length."""
    y = 0.5 * (TRAY_BACK_Y - 0.040 + TRAY_BACK_Y - TRAY_D + 0.008 + LIP[0])
    return (-TRAY_L / 2 + 0.06 + u * (TRAY_L - 0.12), y)


def chalk_mesh():
    """In the tray: a white chalk stick, a pink one, a white stub (lathes lying along X) and the felt eraser (a wooden
    back on a felt pad)."""
    out = []
    for u, length, mat in ((0.16, CHALK_L, "chalk"), (0.33, CHALK_L, "chalk_pink"), (0.47, 0.034, "chalk")):
        h = length / 2
        stick = _lathe([(0.0, -h), (CHALK_R, -h), (CHALK_R, h), (0.0, h)], MATS[mat], seg=20,
                       fillets=[0, 0.0022, 0.0022, 0]).rot(ry=90.0)
        x, y = tray_spot(u)
        out.append(stick.moved((x, y, TRAY_Z + CHALK_R)))
    x, y = tray_spot(0.82)
    felt = S.rounded_panel(S.rrect(0.128, 0.041, 0.009, 4), 0.012, r_edge=0.0045, n=3, r_back=0.0045, mat=MATS["felt"])
    back = S.rounded_panel(S.rrect(0.132, 0.044, 0.013, 4), 0.020, r_edge=0.009, n=3, r_back=0.007, mat=MATS["eraser"])
    out += [felt.moved((x, y, TRAY_Z)), back.moved((x, y, TRAY_Z + 0.011))]
    return S.merge(out)


# ===================================================================================================================
# the drawing: noughts and crosses in chalk (strokes in the board's frame: a across, b up, metres from CENTRE)
# ===================================================================================================================
CELL = 0.165                        # a square of the grid
GRID_AT = (-0.03, 0.0)              # the grid's middle on the slate
STROKE = 0.013                      # chalk line width
GAME = (("X", 1, 1), ("O", 0, 0), ("X", 2, 0), ("O", 0, 2))     # (mark, column from the left, row from the top)


def _wobble(P, seed, amp=0.0025):
    """A hand-drawn line: the polyline P (k, 2) moved across itself by a smooth, seeded wobble."""
    P = np.asarray(P, float)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    w = amp * (np.sin(s * 23.0 + seed * 1.7) + 0.5 * np.sin(s * 57.0 + seed * 3.1))
    return P + n * w[:, None]


def drawing_strokes():
    """The noughts-and-crosses strokes as polylines (k, 2) in the board's frame: two lines each way for the grid, two
    for a cross, an overlapping loop for a nought."""
    gx, gy = GRID_AT
    span = 1.5 * CELL
    out = []
    for k, off in enumerate((-0.5 * CELL, 0.5 * CELL)):
        t = np.linspace(-span, span, 40)
        out.append(_wobble(np.stack([np.full(40, gx + off), gy + t], 1), 2 * k))
        out.append(_wobble(np.stack([gx + t, np.full(40, gy + off)], 1), 2 * k + 1))
    for i, (mark, col, row) in enumerate(GAME):
        cx, cy = gx + (col - 1) * CELL, gy - (row - 1) * CELL
        r = 0.5 * CELL - 0.030
        if mark == "X":
            t = np.linspace(-r, r, 16)
            out.append(_wobble(np.stack([cx + t, cy + t], 1), 10 + i, 0.0015))
            out.append(_wobble(np.stack([cx + t, cy - t], 1), 20 + i, 0.0015))
        else:
            a = np.linspace(0.6, 0.6 + 2.0 * math.pi + 0.5, 48)
            rr = r * (1.0 + 0.06 * np.sin(a * 1.0 + i))
            out.append(np.stack([cx + rr * np.cos(a), cy + 1.08 * rr * np.sin(a)], 1))
    return out


def drawing_mesh(lift=0.0007):
    """The chalk strokes as flat ribbons STROKE wide just over the slate's face (a graphic layer), in the prop's frame."""
    out = []
    for P in drawing_strokes():
        path = np.array([board_point(a, b) for a, b in P])
        out.append(S.ribbon(path, STROKE, tuple(NORMAL), lift=lift, mat=MATS["chalk"]))
    return S.merge(out)


GRAPHIC = ("drawing",)


def parts(drawing="none"):
    """{key: Part(mesh, origin, rot)}, everything in the prop's frame: board, easel, chalk, and the drawing when asked."""
    zero = (0.0, 0.0, 0.0)
    out = {"board": S.Part(board_mesh(), zero, zero), "easel": S.Part(easel_mesh(), zero, zero),
           "chalk": S.Part(chalk_mesh(), zero, zero)}
    if drawing == "noughts":
        out["drawing"] = S.Part(drawing_mesh(), zero, zero)
    return out


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, roles=None, opts=None):
    """The prop card (docs/design.md: Prop card): the slate to write on (`[[text]] on = "<name>:board"`), the tray's edge
    and the board to look at."""
    opts = dict(opts or {"drawing": "none"})
    lip_y = TRAY_BACK_Y - TRAY_D + 0.008 + LIP[0] / 2
    lip_z = TRAY_Z + LIP[1] - 0.004 + 0.004
    x = TRAY_L / 2 - 0.03
    use = {"surface": [{"name": "board", "center": _r(CENTRE), "normal": _r(NORMAL), "up": _r(UP),
                        "size": [SLATE_W, SLATE_H]}],
           "rest": [{"name": "tray", "type": "edge", "a": _r((-x, lip_y, lip_z)), "b": _r((x, lip_y, lip_z)),
                     "normal": [0.0, 0.0, 1.0]}],
           "look": [{"name": "board", "point": _r(CENTRE)}]}
    objs = ["board", "easel", "chalk"] + (["drawing"] if opts["drawing"] == "noughts" else [])
    top = APEX_Z + BARREL_R - 0.010 + 0.064
    width = max(TRAY_L, SLATE_W + 2 * FRAME_W, 2 * LEG_FOOT_X + LEG_W)
    return {"size": _r((width, BACK_FOOT_Y - LEG_FOOT_Y + 0.5 * (LEG_W + LEG_D), top), 4), "origin": "floor_center",
            "front": "-Y", "slots": dict(roles or {}), "use": use, "colliders": [], "params": {}, "options": opts,
            "objects": [f"{name}_{k}" for k in objs]}
