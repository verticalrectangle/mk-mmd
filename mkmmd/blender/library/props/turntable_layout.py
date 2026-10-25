"""Layout of the turntable (pure Python + numpy, no bpy: importable from tests): every number the parts agree on, the colour
roles, the tonearm's kinematics (the `arm` and `cue` values to its swing and lift, the stylus point), the drivers'
expressions and the prop card.

Frame (metres; everything in the card is in it): ORIGIN on the floor under the middle of the plinth, Z up, the front edge
(the start / stop and speed buttons) toward -Y (card `front` = "-Y"), X to the right seen from the front. The platter turns
about the spindle at SPINDLE, clockwise seen from above (as a record does); the tonearm's pivot is at the back right
(PIVOT) and its arm swings in over the record from its rest at the front right.

Modelled in proportion on a Technics SL-1200 MK2-type direct-drive deck, unbranded (no logo, no lettering): a 453 x 360 mm
plinth (an aluminium top plate over a dark base on four big feet), a 332 mm platter with four rows of strobe dots on its
bevel, an S-shaped tonearm of 230 mm effective length with 15 mm overhang (the pivot 215 mm from the spindle) and a
headshell offset 22 degrees. Measured on a front three-quarter photograph rectified by the plate's four corners: the
spindle 179 mm from the left edge, centred front to back, the pivot 24 degrees round from the line through the spindle to
the right, the arm well 56 mm in radius round it, the power knob, the start / stop button and the 33 / 45 buttons at the
front left, the pop-up target light in front of the platter's right, the pitch slider along the right edge, the 45-adapter
well in the back left corner. The record is a 7-inch single to the RIAA / IEC 60098 dimensions: 174.6 mm across, a 38.2 mm
centre hole that a 45 adapter fills, the recorded area from 168.3 mm (the lead-in groove just outside it) to 108 mm, the
finishing groove at 98.4 mm, an 89 mm label.

The arm: `arm` 0 is the arm on its rest, 1 the stylus over the lead-in groove (LEADIN_R), 1 .. 2 the side played, the
stylus moving in at a steady pitch to the last groove (GROOVE_IN) at 2 (a 7-inch side is three to four minutes; key 1 at
the drop and 1 + seconds / 210 later). The swing from the rest to the lead-in is even in angle. `cue` 0 lowers the arm
(the stylus on the record's surface, Z_REC, wherever it is: off the record it floats at that height), 1 lifts it LIFT_DEG
about the pivot, the stylus LIFT_MM over the record. Real decks need about 1.5 s to lower the arm: key `cue` 1 -> 0 over
a second or two for a needle drop."""
import math

import numpy as np

from . import convertible_palette as CP
from ....core import palette as PAL

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = {
    "plate": "paper:2,muted:1",     # the aluminium top plate, the platter's lips, the strobe dots, the pitch slider
    "body": "ink",                  # the base, feet, arm well, knob caps, counterweight, the strobe band: dark rubber
    "metal": "paper:3,subtle:1",    # bright metal: the tonearm, spindle, 45 adapter, collars, the pop-up light
    "mat": "love",                  # the felt slipmat
    "vinyl": "ink",                 # the record (a hex or a slot for coloured vinyl)
    "label": "gold",                # its paper label
    "print": "label:3,ink:1",       # the printed rings on the label (default: a darker shade of the label)
    "cartridge": "love",            # the cartridge's body
    "lamp": "love",                 # the strobe lamp, the speed lamps and the target light (they glow)
}
# material index of a role in the meshes (`Mesh.Qm / Tm`); the builder turns them into Blender materials
MATS = {name: i for i, name in enumerate(("plate", "body", "metal", "mat", "vinyl", "label", "print", "cartridge", "lamp",
                                          "rim", "rubber", "led33", "led45", "dark"))}
ROLE_OF_MAT = {"rim": "body", "rubber": "body", "led33": "lamp", "led45": "lamp", "dark": "body"}
INK_K = 0.35                        # `ink`: the darker of the palette's base and text at this share of its light


class Pal(CP.Pal):
    """The turntable's colour roles (`ROLES`) over the palette. Two slots of its own: `ink` (the darker of the palette's
    base and text at INK_K of its brightness: black rubber and vinyl in a light palette as in a dark one) and `paper`
    (the lighter of the two: bright metal). A role may default to another role (print = a shade of label)."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}

    def _ends(self):
        a, b = (PAL.linear(self.palette[k]) for k in ("base", "text"))
        lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]      # noqa: E731
        return (a, b) if lum(a) < lum(b) else (b, a)

    def slot_lin(self, name):
        if name == "ink":
            return tuple(INK_K * c for c in self._ends()[0])
        if name == "paper":
            return self._ends()[1]
        return super().slot_lin(name)

    def lin(self, spec, depth=0):
        """Linear RGB of a role, a slot (`ink` and `paper` too), a hex or a 'slot:weight,...' blend whose parts may be
        roles themselves."""
        if depth > 8:
            raise ValueError(f"turntable: the colour roles refer to each other in a loop ({spec!r})")
        if spec in self.specs:
            return self.lin(self.specs[spec], depth + 1)
        out, tot = [0.0, 0.0, 0.0], 0.0
        for part in spec.split(","):
            nm, w = CP.parse(part)
            c = self.lin(nm, depth + 1) if nm in self.specs else self.slot_lin(nm)
            out = [o + w * x for o, x in zip(out, c)]
            tot += w
        return tuple(o / tot for o in out)


# ===================================================================================================================
# the plinth (z up from the floor)
# ===================================================================================================================
W, D = 0.453, 0.360                 # plinth width (x) and depth (y)
CORNER_R = 0.008                    # plan radius of the plinth's corners
FOOT_H = 0.028                      # the feet: the base stands this high
FOOT_R = 0.034
FEET = ((-0.1865, -0.140), (0.1865, -0.140), (-0.1865, 0.140), (0.1865, 0.140))
BASE_TOP = 0.060                    # the dark base from FOOT_H to here, inset BASE_INSET under the plate
BASE_INSET = 0.0015
Z_PLATE = 0.073                     # the top plate's upper face (13 mm thick)
PLATE_EDGE_R = 0.004                # its rolled top edge

# ===================================================================================================================
# platter, mat, record
# ===================================================================================================================
SPINDLE = (-0.0475, 0.0)            # the spindle's axis (x, y)
PLATTER_R = 0.166                   # 332 mm
PLATTER_TOP = Z_PLATE + 0.019       # the platter's upper face, where the mat lies
TOP_R = 0.1505                      # the upper face's edge, inside the strobe bevel
DOT_ROWS = ((0.0080, 183), (0.0105, 180), (0.0130, 176), (0.0155, 172))   # (height over the plate, dots) of the rows
DOT_R = 0.0011                      # a strobe dot's radius
MAT_R, MAT_T = 0.1500, 0.0030       # the felt slipmat
Z_MAT = PLATTER_TOP + MAT_T         # the record lies here
SPINDLE_R = 0.0036                  # 7.24 mm

REC_R = 0.0873                      # 174.6 mm
HOLE_R = 0.0191                     # 38.2 mm
REC_T = 0.0011                      # the recorded area's thickness
BEAD_T = 0.0013                     # the rim's bead
LABEL_AREA_R = 0.0490               # the raised label area (BEAD over the grooves)
LABEL_AREA_T = 0.0015
LEADIN_R = 0.0855                   # the lead-in groove: where the stylus is set down (arm = 1)
GROOVE_OUT = 0.08415                # the outermost groove at recording pitch (168.3 mm)
GROOVE_IN = 0.0540                  # the last music groove (108 mm): arm = 2
STOP_R = 0.0492                     # the finishing groove (98.4 mm)
LABEL_R = 0.0445                    # 89 mm
LABEL_T = 0.0001                    # the paper
Z_REC = Z_MAT + REC_T               # the playing surface: the stylus tip on the record
Z_LABEL = Z_MAT + LABEL_AREA_T + LABEL_T

ADAPTER_R = 0.0189                  # the 45 adapter fills the centre hole
ADAPTER_TOP = Z_MAT + 0.0110        # and stands 9.5 mm over the label
SPINDLE_TOP = ADAPTER_TOP + 0.0012  # the spindle's tip shows through it

# the label's two text panels (the text frame: x along +X, y along +Y, read from the front), above and below the hole;
# the largest rectangles between the adapter (with 2.4 mm to spare) and a 2 mm margin inside the label's edge
LABEL_PANEL = {"w": 0.052, "y0": 0.0215, "y1": 0.0335}

# ===================================================================================================================
# the tonearm
# ===================================================================================================================
ARM_L = 0.230                       # effective length: the pivot to the stylus
OVERHANG = 0.015
PIVOT_D = ARM_L - OVERHANG          # 215 mm from the spindle
PIVOT_DEG = 24.0                    # the pivot seen from the spindle, degrees counter-clockwise from +X
PIVOT = (SPINDLE[0] + PIVOT_D * math.cos(math.radians(PIVOT_DEG)),
         SPINDLE[1] + PIVOT_D * math.sin(math.radians(PIVOT_DEG)))
OFFSET_DEG = 22.0                   # the headshell's offset angle (toward the spindle)
H_TUBE = 0.021                      # the tube's axis over the stylus tip
Z_ARM = Z_REC + H_TUBE              # the gimbal's horizontal axle: the arm's frame origin
REST_DEG = 266.0                    # the arm line on its rest, degrees like PIVOT_DEG (270 points at the front)
LIFT_DEG = 2.6                      # cue = 1: the arm turned up this much about its axle
LIFT_MM = round(1000.0 * ARM_L * math.sin(math.radians(LIFT_DEG)), 1)
R_TUBE = 0.0045
WELL_R = 0.056                      # the arm well round the pivot, WELL_DEPTH deep
WELL_DEPTH = 0.008
CUE_S = 0.046                       # the cue's lift bar: this far from the pivot, under the tube
REST_S = 0.072                      # the arm rest: this far along the tube from the pivot, at the well's front edge
CUE_LEVER = {"deg": -38.0, "s": 0.043, "len": 0.030, "swing": 34.0}       # its hinge round the pivot; the lever's swing
ANTISKATE = {"deg": 12.0, "s": 0.043, "r": 0.0105}

# controls on the plate: (x, y) centres
POWER = (-0.2025, -0.1115)
POWER_R, POWER_H = 0.0135, 0.021
START = (-0.1950, -0.1505)
START_SIZE = (0.048, 0.030)
SPEED = {"33": (-0.1547, -0.1636), "45": (-0.1293, -0.1636)}
SPEED_SIZE = (0.022, 0.009)
TARGET = (0.0349, -0.1605)          # the pop-up target light
TARGET_R, TARGET_H = 0.0062, 0.026
PITCH = {"x": 0.1920, "y": (-0.1265, -0.0205), "knob": -0.0735}         # the slot and the knob's centre (0 %)
ADAPTER_WELL = (-0.1917, 0.1483)    # the empty well where the adapter is kept
HINGES = (-0.122, 0.081)            # x of the dust cover's hinge sockets on the back edge (the cover is not modelled)
HINGE_SIZE = (0.046, 0.017, 0.013)  # width, depth, height
STROBE_W = 0.15                     # watts of the strobe lamp's red light on the dots

RPM_45 = 39.0                       # rpm above which the 45 lamp is lit (else the 33's: the deck wakes at 33)


# ===================================================================================================================
# kinematics
# ===================================================================================================================
THETA_SP = math.radians(PIVOT_DEG + 180.0)           # the arm line pointing at the spindle
A_COS, B_COS = PIVOT_D ** 2 + ARM_L ** 2, 2.0 * PIVOT_D * ARM_L


def phi(r):
    """The angle at the pivot between the spindle and a stylus `r` from the spindle (radians)."""
    return math.acos(max(-1.0, min(1.0, (A_COS - r * r) / B_COS)))


THETA_LEAD = THETA_SP + phi(LEADIN_R)
THETA_REST = math.radians(REST_DEG)
THETA_IN = THETA_SP + phi(GROOVE_IN)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def stylus_radius(arm):
    """Where the stylus is from the spindle at `arm` 1 .. 2 (held outside)."""
    return LEADIN_R + (GROOVE_IN - LEADIN_R) * _clamp(arm - 1.0, 0.0, 1.0)


def yaw(arm):
    """The arm line's direction (radians, counter-clockwise from +X) at `arm`: from the rest to the lead-in evenly, then
    the stylus moving in at a steady pitch."""
    a = float(arm)
    return (THETA_REST + (THETA_LEAD - THETA_REST) * _clamp(a, 0.0, 1.0) + THETA_SP - THETA_LEAD
            + phi(stylus_radius(a)))


def lift(cue):
    """The arm's turn up about its axle (radians, positive lifts the stylus) at `cue`."""
    return math.radians(LIFT_DEG) * _clamp(float(cue), 0.0, 1.0)


def arm_matrix(arm, cue):
    """3x3 rotation of the arm's frame (x along the arm line toward the stylus, z up) in the prop's frame: the lift about
    its own y (the axle), then the swing about z (Blender's XYZ euler (0, -lift, yaw))."""
    t, b = yaw(arm), -lift(cue)
    Rz = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0], [-math.sin(b), 0.0, math.cos(b)]])
    return Rz @ Ry


STYLUS_LOCAL = (ARM_L, 0.0, -H_TUBE)                 # the stylus tip in the arm's frame


def stylus(arm, cue):
    """The stylus tip in the prop's frame at (`arm`, `cue`)."""
    return np.array([PIVOT[0], PIVOT[1], Z_ARM]) + arm_matrix(arm, cue) @ np.array(STYLUS_LOCAL)


def around(p, deg, s):
    """The point `s` from `p` (x, y) in the direction `deg` (degrees counter-clockwise from +X)."""
    a = math.radians(deg)
    return (p[0] + s * math.cos(a), p[1] + s * math.sin(a))


# ===================================================================================================================
# drivers (Blender simple expressions over the root's properties; tests evaluate them against the functions above)
# ===================================================================================================================
def _f(v):
    return f"{v:.7f}"


BAR_GAP = 0.0015                    # the lift bar waits this far under the tube while the arm is down
BAR_RISE = BAR_GAP + CUE_S * math.sin(math.radians(LIFT_DEG))     # and rises this much to hold it up at cue = 1
EXPR = {
    "yaw": (f"{_f(THETA_REST)} + {_f(THETA_LEAD - THETA_REST)}*clamp(arm, 0, 1) + {_f(THETA_SP - THETA_LEAD)} + acos(("
            f"{_f(A_COS)} - pow({_f(LEADIN_R)} + {_f(GROOVE_IN - LEADIN_R)}*clamp(arm - 1, 0, 1), 2))/{_f(B_COS)})"),
    "lift": f"-{_f(math.radians(LIFT_DEG))}*clamp(cue, 0, 1)",
    "spin": "-6.2831853*turns",
    "lever": f"-{_f(math.radians(CUE_LEVER['swing']))}*clamp(cue, 0, 1)",
    "bar": f"{_f(BAR_RISE)}*clamp(cue, 0, 1)",
    "led45": f"1.0 if abs(rpm) > {RPM_45:g} else 0.0",
    "led33": f"0.0 if abs(rpm) > {RPM_45:g} else 1.0",
}
SIMPLE = {"clamp": lambda v, lo, hi: _clamp(v, lo, hi), "acos": math.acos, "pow": math.pow, "abs": abs}

PARAMS = {                          # root properties for [[key]]: (default, soft range, description)
    "rpm": (0.0, (-80.0, 80.0), "platter speed, revolutions a minute (0, 33.3, 45; negative turns it back)"),
    "arm": (0.0, (0.0, 2.0), "tonearm: 0 on its rest, 1 over the lead-in groove, 1..2 playing in to the last groove"),
    "cue": (0.0, (0.0, 1.0), "cue lever: 0 down (the stylus on the record), 1 lifted"),
}


def options(slots):
    """The starting values of the root properties from the prop's `slots` (`rpm`, `arm`, `cue`), checked."""
    slots = dict(slots or {})
    out = {}
    for k, (d, _rng, _doc) in PARAMS.items():
        v = slots.get(k, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"turntable: {k} must be a number, got {v!r}")
        out[k] = float(v)
    if not 0.0 <= out["arm"] <= 2.0:
        raise ValueError(f"turntable: arm is 0 (on its rest) .. 1 (the lead-in) .. 2 (the last groove), got {out['arm']:g}")
    if not 0.0 <= out["cue"] <= 1.0:
        raise ValueError(f"turntable: cue is 0 (down) .. 1 (lifted), got {out['cue']:g}")
    return out


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def label_panels(name):
    """The label's two `use.surface` panels, `label` (above the hole) and `label_lower` (below it): they ride the platter."""
    p = LABEL_PANEL
    cy = 0.5 * (p["y0"] + p["y1"])
    out = []
    for nm, s in (("label", 1.0), ("label_lower", -1.0)):
        out.append({"name": nm, "center": _r((SPINDLE[0], SPINDLE[1] + s * cy, Z_LABEL)), "normal": [0.0, 0.0, 1.0],
                    "up": [0.0, 1.0, 0.0], "size": _r((p["w"], p["y1"] - p["y0"])), "object": f"{name}_platter"})
    return out


def card(name, roles=None, start=None):
    """The prop card (docs/design.md: Prop card): the label surfaces, look points, anchors, the live root properties and
    the `rates` entry that makes the keys stage turn `rpm` into `turns` (the platter's revolutions)."""
    looks = [{"name": "record", "point": _r((SPINDLE[0], SPINDLE[1], Z_LABEL))},
             {"name": "stylus", "point": _r(stylus(0.0, 0.0)), "object": f"{name}_stylus"}]
    anchors = [{"name": "spindle", "point": _r((SPINDLE[0], SPINDLE[1], SPINDLE_TOP)), "dir": [0.0, 0.0, 1.0]},
               {"name": "stylus", "point": _r(stylus(0.0, 0.0)), "dir": [0.0, 0.0, -1.0], "object": f"{name}_stylus"}]
    start = dict(start or {k: v[0] for k, v in PARAMS.items()})
    return {"origin": "floor_center", "front": "-Y", "slots": dict(roles or {}),
            "use": {"surface": label_panels(name), "look": looks, "anchor": anchors},
            "params": start, "rates": [{"rate": "rpm", "into": "turns", "per": 60.0}]}
