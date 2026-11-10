"""Layout of the offset electric guitar (pure Python + numpy, no bpy: importable from tests): every number its parts agree on,
the colour roles, the `WEAR` entry and the prop card (`card`).

An offset-waist guitar of the late 1950s kind: a long body whose waist is skewed (the bass side's low, the treble side's
high, so it balances on a seated player's knee), a large pickguard, two wide flat single-coil pickups, a rocking bridge and,
behind it, a floating tremolo: a plate the strings are anchored to and a long arm. Shoegaze players strum with that arm in
the hand, the chords bending as they play. Unbranded: no logo, no lettering.

The NECK is the electric guitar's (`electric_guitar_layout`: the 25.5 inch scale, 22 frets, six strings, the board, the
headstock; `electric_guitar_neck` builds it), so the frame and the contract with the hand, strum and wear code are that
guitar's: the guitar stands UPRIGHT on its tail, Z up along the strings toward the nut, the FACE looks toward -Y, X across it
with the low E and the long horn at -X. ORIGIN: on the centre line, on the body's flat front face (y = 0), at the SADDLE LINE
(z = 0); the body slab occupies y in [0, BODY_T].

The body, guard and hardware are landmarks read off a front-on photograph of such a guitar whose scale came from its nut and
12th fret (1508 px for the 648 mm): the body's edges at stations 15 mm apart, rounded to the millimetre, and the middles
and sizes of the parts on it. Never a traced contour. The body is 0.348 wide at the lower bout (z -0.115 on the treble side,
-0.04 on the bass side), 0.234 at the bass side's waist (z 0.125), its tail at z -0.195, the long horn's tip at 0.293.

Colour roles (`[[prop]] slots = {body = "#efe6cf"}`; a palette slot, a hex, or a blend 'slot:w,slot:w'): see `ROLES`.
`slots = {guard = "plain"}` makes the pickguard one colour (`pickguard`) instead of tortoiseshell; `slots = {arm = false}`
leaves the tremolo arm off (its collet stays), `arm = {swing = 8, lift = 0.016, length = 0.165}` sets how it lies.
"""
import math

import numpy as np

from . import electric_guitar_layout as EL

# ===================================================================================================================
# what it shares with the electric guitar: the neck and the strings
# ===================================================================================================================
SCALE, N_FRETS, Y_BOARD, BODY_T = EL.SCALE, EL.N_FRETS, EL.Y_BOARD, EL.BODY_T
STRING_R, SADDLE_X, SETBACK = EL.STRING_R, EL.SADDLE_X, EL.SETBACK
MATS, ROLE_OF_MAT = EL.MATS, EL.ROLE_OF_MAT
GUARD_T, CABLE_R, FLOOR_Z = EL.GUARD_T, EL.CABLE_R, -0.22
# the plectrum: a small one (a jazz pick, 24 x 20 mm), held as a plectrum is: the thumb across its wide end (the card's
# strum `hold`), the pads 14 mm from the tip, so the thumb covers all but the point, and no skin nearer the strings than
# PICK_CLEAR (the card's strum `clear`)
PICK = dict(EL.PICK, length=0.024, width=0.020, tip=0.014)
PICK_HOLD, PICK_CLEAR = "across", 0.007
fret_z, string_ends, string_at = EL.fret_z, EL.string_ends, EL.string_at

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = dict(EL.ROLES)
ROLES.update({
    "body": "text:6,gold:1",                  # an old white gone cream
    "pickguard": "love:2,gold:3,base:2",      # tortoiseshell: the amber flames ...
    "guard_dark": "base:5,love:2",            # ... in the dark ground
    "knobs": "text:8,gold:1",                 # white plastic: the knobs, the pickup covers, the switch's and the arm's tips
})
GUARDS = ("tortoise", "plain")
ARM = {"swing": 8.0, "lift": 0.016, "length": 0.165}     # degrees from the strings toward the treble side; m the tip stands
#                                                        # further off the face than the bend; m from the bend to the tip's end


class Pal(EL.Pal):
    """The offset guitar's colour roles (`ROLES`) over the palette, the guard's kind (`guard`) and the arm (`arm`: a table
    of `ARM`'s keys, or None for no arm)."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}
        self.guard = str(slots.get("guard", GUARDS[0]))
        if self.guard not in GUARDS:
            raise ValueError(f"offset_guitar: guard = {self.guard!r}: one of {', '.join(GUARDS)}")
        arm = slots.get("arm", True)
        if arm is False:
            self.arm = None
        else:
            given = {} if arm is True else dict(arm)
            if set(given) - set(ARM):
                raise ValueError(f"offset_guitar: arm = {arm!r}: false, or a table of {', '.join(ARM)}")
            self.arm = {k: float(given.get(k, v)) for k, v in ARM.items()}


# ===================================================================================================================
# the body
# ===================================================================================================================
BODY_EDGE_R = 0.0060                # the body's rolled edge (front and back)
# control points of the body outline (x, z), counter-clockwise from the tail's lowest point: the treble side up to its
# shoulder and the shallow bay beside the neck, the neck pocket's lobe (the electric guitar's), the bass bay, the long horn,
# the bass side down and round the tail. The spline runs through them.
TREBLE = ((0.050, -0.1950), (0.080, -0.1945), (0.105, -0.1925), (0.1259, -0.1900), (0.1375, -0.1850), (0.1525, -0.1750),
          (0.1650, -0.1600), (0.1706, -0.1450), (0.1736, -0.1300), (0.1745, -0.1150), (0.1732, -0.1000), (0.1702, -0.0850),
          (0.1650, -0.0700), (0.1580, -0.0550), (0.1495, -0.0400), (0.1405, -0.0250), (0.1328, -0.0100), (0.1263, 0.0050),
          (0.1215, 0.0200), (0.1182, 0.0350), (0.1195, 0.0500), (0.1242, 0.0650), (0.1310, 0.0800),      # the treble waist
          (0.1358, 0.0950), (0.1395, 0.1100), (0.1418, 0.1250), (0.1427, 0.1400), (0.1414, 0.1550), (0.1364, 0.1700),
          (0.1306, 0.1800), (0.1212, 0.1900), (0.1100, 0.1945), (0.0980, 0.1950), (0.0870, 0.1925),      # the shoulder
          (0.0760, 0.1880), (0.0640, 0.1845), (0.0520, 0.1845), (0.0420, 0.1870), (0.0330, 0.1915),      # its bay
          (0.0262, 0.1985))                                                # the neck pocket's right wall (a sharp corner)
LOBE = EL.LOBE                                                             # the pocket end: under the neck
BASS = ((-0.0330, 0.2445), (-0.0450, 0.2435), (-0.0570, 0.2465), (-0.0649, 0.2500), (-0.0752, 0.2600),    # the bass bay
        (-0.0834, 0.2700), (-0.0928, 0.2800), (-0.1000, 0.2865), (-0.1085, 0.2915), (-0.1180, 0.2930),    # the long horn
        (-0.1262, 0.2905), (-0.1318, 0.2855), (-0.1354, 0.2800), (-0.1392, 0.2700), (-0.1405, 0.2600), (-0.1401, 0.2500),
        (-0.1392, 0.2400), (-0.1379, 0.2300), (-0.1358, 0.2200), (-0.1332, 0.2100), (-0.1302, 0.2000), (-0.1272, 0.1900),
        (-0.1246, 0.1800), (-0.1220, 0.1700), (-0.1182, 0.1550), (-0.1160, 0.1400), (-0.1152, 0.1250),    # the bass waist
        (-0.1160, 0.1100), (-0.1199, 0.0950), (-0.1259, 0.0800), (-0.1332, 0.0650), (-0.1414, 0.0500), (-0.1491, 0.0350),
        (-0.1556, 0.0200), (-0.1620, 0.0050), (-0.1676, -0.0100), (-0.1715, -0.0250), (-0.1732, -0.0400),
        (-0.1727, -0.0550), (-0.1702, -0.0700), (-0.1663, -0.0850), (-0.1599, -0.1000), (-0.1508, -0.1150),
        (-0.1384, -0.1300), (-0.1212, -0.1450), (-0.0980, -0.1600), (-0.0602, -0.1750), (-0.0350, -0.1835),
        (-0.0120, -0.1895), (0.0200, -0.1935))
OUTLINE_CORNERS = (len(TREBLE) - 1,)                                       # the pocket wall corner


def body_control_points():
    """The outline's control points (n, 2), counter-clockwise, and the indices where it turns sharply."""
    return np.array(list(TREBLE) + list(LOBE) + list(BASS), float), OUTLINE_CORNERS


# strap buttons: where the button leaves the body, and the way it points
STRAP_TOP = {"point": (-0.1180, BODY_T / 2, 0.2930), "dir": (-0.20, 0.0, 0.98)}     # the long horn's tip
STRAP_BOTTOM = {"point": (0.0040, BODY_T / 2, -0.1915), "dir": (-0.13, 0.0, -0.99)}  # the tail, on the centre line

# the contours: the forearm bevel on the bass edge of the lower bout's front, the belly cut on the back at the bass waist
FOREARM = {"depth": 0.0080, "width": 0.050, "z": (-0.165, -0.110, 0.010, 0.060)}
BELLY = {"center": (-0.105, 0.085), "radii": (0.075, 0.120), "depth": 0.0115}

# the pickguard: control points (x, z), counter-clockwise from its corner under the bridge's bass end
GUARD = ((-0.0540, -0.0100), (-0.0395, -0.0200), (-0.0050, -0.0225), (0.0380, -0.0212), (0.0500, -0.0235),
         (0.0590, -0.0280), (0.0700, -0.0380), (0.0780, -0.0480), (0.0880, -0.0620), (0.1000, -0.0770),    # the lower edge
         (0.1150, -0.0880), (0.1300, -0.0935), (0.1430, -0.0940),                                          # of its lobe
         (0.1510, -0.0900), (0.1538, -0.0820), (0.1504, -0.0700), (0.1444, -0.0600), (0.1375, -0.0500),    # its tip
         (0.1311, -0.0400), (0.1246, -0.0300), (0.1186, -0.0200), (0.1134, -0.0100), (0.1104, 0.0000),
         (0.1057, 0.0100), (0.1036, 0.0200), (0.1023, 0.0350), (0.1040, 0.0500), (0.1066, 0.0600),         # along the waist
         (0.1100, 0.0700), (0.1139, 0.0800), (0.1173, 0.0900), (0.1205, 0.1020), (0.1225, 0.1150), (0.1242, 0.1300),
         (0.1242, 0.1450), (0.1216, 0.1550), (0.1173, 0.1650), (0.1130, 0.1730), (0.1040, 0.1785), (0.0900, 0.1790),
         (0.0680, 0.1765), (0.0470, 0.1775), (0.0350, 0.1795),                                             # the top edge
         (0.0292, 0.1830),                                                 # which meets the neck at its heel end
         (-0.0292, 0.1830), (-0.0292, 0.1920), (-0.0292, 0.2080),          # and goes up its left edge
         (-0.0357, 0.2150), (-0.0540, 0.2240), (-0.0722, 0.2330), (-0.0851, 0.2375), (-0.0950, 0.2380),    # the bass tip
         (-0.1010, 0.2350), (-0.1027, 0.2250), (-0.1018, 0.2150), (-0.1006, 0.2050), (-0.0980, 0.1950),
         (-0.0958, 0.1850), (-0.0932, 0.1750), (-0.0902, 0.1650), (-0.0877, 0.1550), (-0.0851, 0.1450),
         (-0.0816, 0.1350), (-0.0782, 0.1250), (-0.0752, 0.1150), (-0.0718, 0.1050), (-0.0683, 0.0900),
         (-0.0645, 0.0800), (-0.0610, 0.0700), (-0.0584, 0.0600), (-0.0576, 0.0500), (-0.0572, 0.0350),
         (-0.0574, 0.0200), (-0.0570, 0.0080), (-0.0563, 0.0000))
GUARD_CORNERS = tuple(GUARD.index(c) for c in ((0.0292, 0.1830), (-0.0292, 0.1830), (-0.0292, 0.2080)))
GUARD_SCREWS = ((-0.0950, 0.2300), (-0.0370, 0.2084), (-0.0560, 0.1835), (-0.0821, 0.1569), (-0.0560, 0.1150),
                (-0.0580, 0.0840), (-0.0480, 0.0060), (0.0580, -0.0150), (0.0971, 0.0077), (0.1180, -0.0400),
                (0.1000, -0.0650), (0.1430, -0.0860), (0.1122, 0.0941), (0.1070, 0.1697), (0.0524, 0.1600),
                (-0.0920, 0.2110), (-0.0650, 0.2110))                      # the last two hold the slide switch

# ===================================================================================================================
# hardware
# ===================================================================================================================
# pickups: wide flat single coils under plastic covers, their middles on the centre line
PICKUP_L, PICKUP_W = 0.0935, 0.0440          # cover length (across) and width (along the strings)
PICKUP_H = 0.0088                            # cover height above the guard: 2 mm under the lowest string
PICKUP_R = 0.0058                            # the cover's corner radius
POLE_R, POLE_PITCH, POLE_H = 0.0024, 0.0105, 0.0006
PICKUPS = {"neck": {"z": 0.1430}, "bridge": {"z": 0.0385}}
PICKUP_SCREWS = ((-0.0215, -0.0190), (0.0215, -0.0190), (-0.0215, 0.0190), (0.0215, 0.0190))   # from a cover's middle

# the bridge: a bar rocking on two posts in cups, six threaded barrel saddles on it
BRIDGE = {"z": -0.0052, "size": (0.0860, 0.0165), "t": 0.0028, "posts": 0.0370, "cup_r": 0.0052}
SADDLE = {"r": 0.0040, "l": 0.0098}          # a barrel's radius and its length across the strings

# the tremolo: a plate on the face (round toward the bridge), the bar the strings are anchored to, the arm's collet
TREM = {"z": (-0.1450, -0.0546), "w": 0.0970, "t": 0.0025, "corner": 0.0065, "arch": 0.0400}   # tail edge, bridge end
ANCHOR = {"z": -0.1335, "size": (0.0660, 0.0085), "h": 0.0046}      # the bar the strings end in, on the plate
COLLET = {"at": (0.0335, -0.0718), "r": 0.0046, "h": 0.0070}      # (x, z): the arm's socket, clear of the high e string
LOCK = {"at": (-0.0130, -0.0900), "r": 0.0042, "h": 0.0034}                       # the trem-lock button
TREM_SCREWS = ((-0.0420, -0.1380), (0.0420, -0.1380), (-0.0420, -0.1000), (0.0420, -0.1000), (-0.0210, -0.0660),
               (0.0210, -0.0660))
ARM_R = 0.0024                               # the arm's bar
ARM_STUB = 0.0125                            # the bar stands this far out of the collet before it bends
ARM_TIP = {"l": 0.0300, "r": 0.0046}         # the plastic tip

# controls of the lead circuit on the treble side: volume and tone knobs, the three-way toggle, the jack on the guard
KNOBS = ((0.0834, 0.0262), (0.0941, -0.0266))
TOGGLE = {"at": (0.1096, 0.1418), "lean": (-0.42, 0.0, 0.34)}                      # the lever's lean (x, y, z), then -y
JACK = {"at": (0.1285, -0.0745), "elev": 62.0, "toward": (0.35, 0.0, -1.0), "mouth": 0.0020, "plug_len": 0.052}
# and of the rhythm circuit on the bass horn: a slide switch and two thumb wheels in slots of the guard
SLIDE = {"at": (-0.0778, 0.2110), "slot": (0.0125, 0.0042)}
ROLLERS = ((-0.0683, 0.1697), (-0.0683, 0.1268))
ROLLER = {"slot": (0.0052, 0.0170), "r": 0.0100, "out": 0.0017, "w": 0.0040}
NECK_PLATE_Z = EL.NECK_PLATE_Z

PICK_REST = {"at": (0.0800, 0.0900)}         # x, z where the pick lies on the pickguard (tip toward +z)
STRUM_ZONE = (0.0700, 0.1120)                # between the two pickups' covers
TECHNIQUE = "pick"

WEAR = {"name": "stand", "bone": "upper_body2",
        "pivot": [0.0, 0.045, 0.0],
        "ref": {"top": 1.70, "shoulder_width": 0.18},
        "at": [-0.14, -0.10, 0.0], "scale": ["shoulder_width", "top", "top"],
        "neck_deg": 35.0, "yaw_deg": 0.0, "roll_deg": 0.0,
        "strap": {"top": "strap_top", "bottom": "strap_bottom", "over": "shoulder.L", "width": 0.05, "thickness": 0.004,
                  "material": "<name>_strap"},
        "cable": {"object": "<name>_cable", "anchor": "jack", "radius": 0.0032}}

COLLIDERS = {                                # hidden objects: key -> (centre, size)  (box specs use the object's bounds)
    "body": ((0.0, 0.0225, -0.0090), (0.348, 0.045, 0.374)),
    "horn": ((-0.1030, 0.0225, 0.2350), (0.0760, 0.045, 0.1180)),
    "neck": EL.COLLIDERS["neck"],
    "head": EL.COLLIDERS["head"],
}
NECK_CAPSULE = EL.NECK_CAPSULE


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def jack_geometry():
    """(mouth, dir, end): the socket's mouth on the guard, the unit vector out of the socket (the plug's axis: `elev`
    degrees out of the face, leaning `toward`) and the plug's end, where the cord leaves."""
    e = math.radians(JACK["elev"])
    t = _unit(JACK["toward"])
    mouth = np.array([JACK["at"][0], -GUARD_T - JACK["mouth"], JACK["at"][1]])
    d = _unit(math.cos(e) * t + math.sin(e) * np.array([0.0, -1.0, 0.0]))
    return mouth, d, mouth + d * JACK["plug_len"]


def strum_center():
    """The pick tip's rest point: centred across the six strings, on their plane, between the pickups."""
    z = 0.5 * sum(STRUM_ZONE)
    y = float(np.mean([string_at(i, z)[1] for i in range(6)]))
    return (0.0, y, z)


def arm_geometry(arm=None):
    """(bend, tip, R): the tremolo arm's bend (where the bar turns over, above the collet: the arm object's origin), its
    tip's end at rest, and the 3x3 rotation of the arm object at rest: its +Z runs from the bend to the tip, its +Y is as
    far out of the face as that allows. `arm`: `ARM`'s keys (default `ARM`)."""
    a = dict(ARM, **(arm or {}))
    (cx, cz), s = COLLET["at"], math.radians(a["swing"])
    bend = np.array([cx, -TREM["t"] - COLLET["h"] - ARM_STUB, cz])
    lift = min(a["lift"], 0.9 * a["length"])
    run = math.sqrt(a["length"] ** 2 - lift ** 2)
    tip = bend + np.array([run * math.sin(s), -lift, run * math.cos(s)])
    z = _unit(tip - bend)
    out = np.array([0.0, -1.0, 0.0])
    y = _unit(out - (out @ z) * z)
    return bend, tip, np.column_stack([np.cross(y, z), y, z])


def wear_entry(name):
    """The card's `use.wear` entry with the object and material names of the prop called `name`."""
    e = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in WEAR.items()}
    e["strap"]["material"] = e["strap"]["material"].replace("<name>", name)
    e["cable"]["object"] = e["cable"]["object"].replace("<name>", name)
    return e


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, roles=None, size=None, arm=None, has_arm=True):
    """The prop card (docs/design.md: Prop card): the electric guitar's (the same neck: its `neck` grip is that guitar's)
    with this body's strum point, anchors, colliders and wear. With an arm, the anchor `trem` is the arm's tip at rest:
    it names the pick hand's bone and the empty the arm aims at, so `[[prop]] anchor_to = "<cast>"` puts the arm in that
    hand (docs/design.md: Props, `offset_guitar`)."""
    c = EL.card(name, roles if roles is not None else ROLES, size if size is not None else (0.348, 0.10, 1.03))
    strum = next(e for e in c["use"]["grip"] if e["name"] == "strum")
    center = strum_center()
    strum.update({"center": _r(center), "zone": list(STRUM_ZONE), "technique": TECHNIQUE})
    strum["pick"].update({k: PICK[k] for k in ("thickness", "length", "width", "tip")})
    strum["clear"], strum["hold"] = PICK_CLEAR, PICK_HOLD
    _mouth, jdir, jend = jack_geometry()
    place = {"strap_top": (STRAP_TOP["point"], _unit(STRAP_TOP["dir"])),
             "strap_bottom": (STRAP_BOTTOM["point"], _unit(STRAP_BOTTOM["dir"])), "jack": (jend, jdir)}
    for a in c["use"]["anchor"]:
        if a["name"] in place:
            a["point"], a["dir"] = _r(place[a["name"]][0]), _r(place[a["name"]][1])
    if has_arm:
        _bend, tip, _R = arm_geometry(arm)
        c["use"]["anchor"].append({"name": "trem", "point": _r(tip), "bone": "wrist.R", "object": f"{name}_trem_aim"})
    for look in c["use"]["look"]:
        if look["name"] == "strum":
            look["point"] = _r(center)
    c["use"]["wear"] = [wear_entry(name)]
    return c
