"""Layout of the giant carrot (`card = "library:giant_carrot"`; pure Python + numpy, no bpy: importable from tests): the
modelled profile, the bites' faces, the chunks' flight and their driver expressions, the colour roles, the options and the
card. The meshes are giant_carrot_parts.py.

Frame (metres): the origin at the TIP, the axis +Z from the tip to the shoulder (L = 0.60 up: the widest point, R_S = 0.05
in radius), a domed crown above the shoulder with the stems coming out of it and leafy greens to about GREENS_H above the
shoulder; front -Y. The root is a long cone with a slightly full side (`radius(z)`: a monotone cubic through PROFILE) that
tapers to a rounded point.

Bites. Root properties `bite1` .. `bite4` (BITES_MAX) are the clip seconds the bites happen (NEVER = 1e4 by default: a gag
keys them CONSTANT, k counting on across chomps, in order). Bite k eats the end of the carrot up to its face, centred
BITE_STEP above bite k - 1's (the first: above the tip): the face's rim is scalloped by round tooth notches between sharp
cusps and tilted a little (`rim`), the notches fade toward the middle as tooth tracks (`face`). Its shape key holds
stage(k) - stage(k - 1) (non-additive: all four on is stage 4) and is driven to clamp((t - bite_k) / BITE_TIME, 0, 1)
(`key_expression`), t = (frame - f0) / fps the clip second (`f0`, `fps`: root properties set at build). Each bite throws
its own chunks (`chunks`): small lumpy pieces that pop in at its bite point, fly a ballistic arc, tumble and shrink away
within LIFE seconds, at scale ~0 before and after (`chunk_pose`, `chunk_expressions`). Gravity is the world's: the drivers
read the root's world rotation (rx, ry) and turn the world's up into the root's frame, so the pieces fall down whatever the
hold (`up_in_root`).

Colour roles (`slots`: a palette slot or a hex; true produce colours by default): skin, flesh (the cut flesh between the
skin and the core), core (the paler middle a bite shows), crown (the green-tinged shoulder), stem, leaf."""
import math
import re

import numpy as np

from ....core import palette as PAL
from ....core import shell as S

L = 0.60                    # tip -> shoulder
R_S = 0.05                  # the shoulder's radius
DOME_H = 0.03               # the crown's dome above the shoulder
DOME_P = 2.2                # its superellipse exponent (2: an ellipse; more: flatter on top)
GREENS_H = 0.33             # the greens' top above the shoulder (the tallest frond)
# the modelled profile: z / L -> r / R_S, a monotone cubic through these (rounded point, full side, flat shoulder)
PROFILE_X = (0.0, 0.0008, 0.003, 0.01, 0.03, 0.08, 0.16, 0.28, 0.42, 0.56, 0.70, 0.82, 0.91, 0.965, 1.0)
PROFILE_R = (0.0, 0.026, 0.05, 0.095, 0.16, 0.25, 0.355, 0.48, 0.61, 0.725, 0.83, 0.91, 0.965, 0.993, 1.0)

HOLD_FROM, HOLD_TO = 0.08, 0.58     # the grippable shaft: from 0.08 above the tip to just under the shoulder
PROFILE_SAMPLES = 8                 # (s, r) samples of the shaft in the grip's `profile`

BITES_MAX = 4
BITE_STEP = 0.06            # each bite's face is centred this much above the last one's (the first: above the tip)
BITE_TIME = 0.05            # seconds a bite takes to carve in
NEVER = 1e4                 # a bite's default clip second: never
# per bite: (tooth notches round the rim, their phase (deg), the face's tilt (share of its radius), the tilt's azimuth)
BITE_SHAPES = ((5, 12.0, 0.10, 200.0), (6, 40.0, 0.12, 30.0), (6, 5.0, 0.10, 290.0), (7, 25.0, 0.12, 120.0))
NOTCH = 0.32                # a notch's depth into the carrot, share of the face's radius
NOTCH_P = 0.75              # |sin|^p round the rim: round notches, sharp cusps between them
RELIEF = 4.0                # the tooth tracks fade toward the face's middle as (rho / R) ** RELIEF

CHUNKS = (3, 2, 4, 3)       # pieces bite k throws
CHUNK_SEED = 11
LIFE = 0.6                  # seconds a piece flies
POP = 0.04                  # seconds it takes to pop in
FADE = 0.32                 # seconds after which it shrinks away (gone at LIFE)
GRAVITY = 9.81
MIN_SCALE = 0.0005          # never exactly 0: a singular matrix upsets what inverts one
NDIGITS = 6                 # decimals of every constant of an expression (the layout is rounded the same)
# the body material reads rel = (distance from the axis) / R(z): the core inside CORE, the skin from SKIN_IN out (so the
# rings and dimples may cut in less than 1 - SKIN_IN of the radius, and a bite's face shows flesh inside SKIN_IN)
CORE, SKIN_IN = 0.46, 0.93

ROLES = {"skin": "#e0650f", "flesh": "#ee6f0a", "core": "#f39b22", "crown": "#869a2c", "stem": "#7aa83f",
         "leaf": "#3f8a2e"}
MATS = ("carrot", "cut", "stem", "leaf")    # material index -> material: the body (skin, flesh and core by the radius),
                                            # a chunk's broken flesh, the stems, the leaflets
SLOTS = tuple(PAL.PALETTES["rose-pine-dawn"])
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
FUNCS = frozenset({"min", "max", "clamp", "smoothstep", "sin", "cos"})
ROOT_VARS = ("f0", "fps") + tuple(f"bite{k}" for k in range(1, BITES_MAX + 1))
TURN_VARS = ("rx", "ry")    # the root's world rotation (Euler XYZ, radians): driver variables of type TRANSFORMS
HELPERS = ("a",)            # a chunk's own custom property, driven: its age, clamped to 0 .. LIFE


def _n(x):
    """A number as plain decimal text (no exponent: Blender's expression parser reads plain decimals)."""
    s = f"{float(x):.{NDIGITS}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _r(x):
    return round(float(x), NDIGITS)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ================================================================================================= the profile
def radius(z):
    """The modelled profile R(z) of the root (m), 0 <= z <= L (clamped): the envelope of the skin, which the rings and
    dimples only cut into."""
    z = np.asarray(z, float)
    x = np.clip(z.ravel() / L, 0.0, 1.0)
    r = S.pchip(np.array(PROFILE_X), np.array(PROFILE_R)[:, None], x)[:, 0] * R_S
    return r.reshape(z.shape) if z.ndim else float(r[0])


def dome(phi):
    """(r, z) of the crown's dome at phi (0 at the shoulder .. pi / 2 at the top): a superellipse quadrant."""
    phi = np.asarray(phi, float)
    c, s = np.clip(np.cos(phi), 0.0, 1.0), np.clip(np.sin(phi), 0.0, 1.0)
    return R_S * c ** (2.0 / DOME_P), L + DOME_H * s ** (2.0 / DOME_P)


def hold_profile():
    """[[s, r], ...]: PROFILE_SAMPLES exact samples of the modelled profile along the grippable shaft, s metres from its tip
    end toward the shoulder."""
    s = np.linspace(0.0, HOLD_TO - HOLD_FROM, PROFILE_SAMPLES)
    return [[_r(v), _r(radius(HOLD_FROM + v))] for v in s]


# z / L where the body material samples the profile (a colour ramp of at most 32 stops, straight lines between them)
RAMP_X = (0.0, 0.0008, 0.003, 0.006, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.19, 0.22, 0.25, 0.28,
          0.32, 0.36, 0.42, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.94, 1.0)


def profile_ramp():
    """[(z / L, R (m)), ...]: the profile as the body material reads it (to tell skin, flesh and core apart)."""
    return [(x, float(radius(x * L))) for x in RAMP_X]


# ================================================================================================= the bites
def bite_height(k):
    """The height of the middle of bite k's face (k = 1 .. BITES_MAX): its bite point is (0, 0, this)."""
    return BITE_STEP * k


def _shape(k):
    n, ph, tilt, az = BITE_SHAPES[k - 1]
    rb = radius(bite_height(k))
    return n, math.radians(ph), NOTCH * rb, tilt * rb, math.radians(az)


def _scallop(n, ph, theta):
    return np.abs(np.sin(0.5 * n * (np.asarray(theta, float) - ph))) ** NOTCH_P


_SCALLOP_MEAN = float(np.mean(np.abs(np.sin(np.linspace(0.0, math.pi, 4001)[:-1])) ** NOTCH_P))


def face(k, u, theta):
    """The height of bite k's face at the share u = rho / R of its radius (0 the middle, 1 the rim) and angle theta: the
    tooth notches' relief fading toward the middle, a slight tilt."""
    n, ph, depth, tilt, az = _shape(k)
    u, theta = np.asarray(u, float), np.asarray(theta, float)
    return (bite_height(k) + u ** RELIEF * depth * (_scallop(n, ph, theta) - _SCALLOP_MEAN)
            + u * tilt * np.cos(theta - az))


def rim(k, theta):
    """The height of bite k's rim at angle theta: what is below it is eaten."""
    return face(k, 1.0, theta)


def rim_range(k):
    """(lowest, highest) point of bite k's rim."""
    z = rim(k, np.linspace(0.0, 2.0 * math.pi, 3601))
    return float(z.min()), float(z.max())


def key_expression(k):
    """The driver expression of shape key `bite<k>`: 0 before bite k, 1 from BITE_TIME after it."""
    return f"clamp(((frame - f0) / fps - bite{k}) / {_n(BITE_TIME)}, 0, 1)"


def key_value(k, t, bites):
    """What `key_expression(k)` gives at clip second t with the bite times `bites` (a list of BITES_MAX)."""
    return float(np.clip((t - bites[k - 1]) / BITE_TIME, 0.0, 1.0))


# ================================================================================================= the chunks
def chunks():
    """Every bitten-off piece: [{bite, j, size, seed, p0 (3,), v (3,), up (its speed along the world's up), rot0 (3,),
    spin (3,)}], the numbers rounded as the expressions write them."""
    out = []
    for k in range(1, BITES_MAX + 1):
        rng = np.random.RandomState(CHUNK_SEED + 7 * k)
        h, rb = bite_height(k), radius(bite_height(k))
        n = CHUNKS[k - 1]
        for j in range(n):
            az = 2.0 * math.pi * (j + rng.uniform(0.15, 0.85)) / n
            out_speed = rng.uniform(0.5, 0.95)                       # out from the axis
            out.append({
                "bite": k, "j": j, "size": _r((0.32 + 0.2 * rng.rand()) * rb + 0.004),
                "seed": int(CHUNK_SEED * 100 + 10 * k + j),
                "p0": [_r(0.45 * rb * math.cos(az)), _r(0.45 * rb * math.sin(az)), _r(h - 0.3 * BITE_STEP)],
                "v": [_r(out_speed * math.cos(az)), _r(out_speed * math.sin(az)), _r(rng.uniform(0.02, 0.25))],
                "up": _r(rng.uniform(0.9, 1.5)),
                "rot0": [_r(v) for v in rng.uniform(-math.pi, math.pi, 3)],
                "spin": [_r(v) for v in rng.choice([-1.0, 1.0], 3) * rng.uniform(8.0, 20.0, 3)]})
    return out


def up_in_root(rx, ry):
    """The world's up (+Z) in the frame of a root whose world rotation is the Euler XYZ (rx, ry, rz) (rz does not matter):
    the last row of R = Rz Ry Rx."""
    return np.array([-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)])


def chunk_pose(c, t, bites, rx=0.0, ry=0.0):
    """Where piece `c` is at clip second t with the bite times `bites`, the root turned (rx, ry) in the world: (position (3,)
    in the root's frame, Euler XYZ rotation (3,), scale), as its drivers have it."""
    a = float(np.clip(t - bites[c["bite"] - 1], 0.0, LIFE))
    lift = a * c["up"] - 0.5 * GRAVITY * a * a
    pos = np.array(c["p0"]) + a * np.array(c["v"]) + lift * up_in_root(rx, ry)
    rot = np.array(c["rot0"]) + a * np.array(c["spin"])
    s = max(MIN_SCALE, min(a / POP, 1.0) * (1.0 - float(smoothstep(FADE, LIFE, a))))
    return pos, rot, s


def chunk_expressions(c):
    """The driver expressions of piece `c`: {channel: text} for its helper `a` (its age, first), x, y, z, rx, ry, rz and
    s (the scale of all three axes). They read the root's `f0`, `fps`, `bite<k>`, its world rotation `rx`, `ry` (TRANSFORMS
    variables), `frame` and the piece's `a`."""
    k = c["bite"]
    g = _n(0.5 * GRAVITY)
    lift = f"(a * {_n(c['up'])} - {g} * a * a)"
    up = ("-sin(ry)", "cos(ry) * sin(rx)", "cos(ry) * cos(rx)")
    out = {"a": f"clamp((frame - f0) / fps - bite{k}, 0, {_n(LIFE)})"}
    for i, ch in enumerate("xyz"):
        out[ch] = f"{_n(c['p0'][i])} + a * {_n(c['v'][i])} + {lift} * {up[i]}"
    for i, ch in enumerate(("rx", "ry", "rz")):
        out[ch] = f"{_n(c['rot0'][i])} + a * {_n(c['spin'][i])}"
    out["s"] = f"max({_n(MIN_SCALE)}, min(a / {_n(POP)}, 1) * (1 - smoothstep({_n(FADE)}, {_n(LIFE)}, a)))"
    for ch, text in out.items():
        if len(text) > 255:
            raise ValueError(f"giant_carrot: the {ch} expression is {len(text)} characters (Blender keeps 255)")
    return out


def chunk_name(name, c):
    return f"{name}_chunk{c['bite']}_{c['j']}"


# ================================================================================================= options and card
def options(slots):
    """{"bites": [starting clip second of each bite], "roles": {role: slot or hex}} from the `slots` of the [[prop]] (the
    palette's slots and keys this prop does not know are ignored), checked: a bad value is a ValueError saying what to
    change."""
    slots = dict(slots or {})
    bites = []
    for k in range(1, BITES_MAX + 1):
        v = slots.get(f"bite{k}", NEVER)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"giant_carrot: bite{k} is the clip second of bite {k} (a number; {NEVER:g} = never), "
                             f"got {v!r}")
        bites.append(float(v))
    roles = {}
    for role, default in ROLES.items():
        v = slots.get(role, default)
        if not isinstance(v, str) or not (v in SLOTS or HEX.match(v)):
            raise ValueError(f"giant_carrot: {role} is a palette slot ({', '.join(SLOTS)}) or a hex \"#rrggbb\", "
                             f"got {v!r}")
        roles[role] = v
    return {"bites": bites, "roles": roles}


def objects(name):
    return [f"{name}_body", f"{name}_greens"] + [chunk_name(name, c) for c in chunks()]


def hold_grip():
    """The `use.grip` entry `hold` (type "hold"): the shaft a fist closes round, its axis toward the tip."""
    length = HOLD_TO - HOLD_FROM
    return {"name": "hold", "type": "hold", "center": [0.0, 0.0, _r(HOLD_FROM + 0.5 * length)], "axis": [0.0, 0.0, -1.0],
            "length": _r(length), "radius": _r(radius(HOLD_TO)), "radius_tip": _r(radius(HOLD_FROM)),
            "profile": hold_profile()}


def card(name, roles, bites, size):
    """The prop card (docs/design.md: Prop card): `use.grip` hold, `use.look` grip and tip, `use.bite` bite1 .. bite4; the
    bite times its live root properties. `roles` {role: hex}, `size` the meshes' extent [x, y, z]."""
    hold = hold_grip()
    use = {"grip": [hold],
           "look": [{"name": "grip", "point": list(hold["center"])}, {"name": "tip", "point": [0.0, 0.0, 0.0]}],
           "bite": [{"name": f"bite{k}", "point": [0.0, 0.0, _r(bite_height(k))], "dir": [0.0, 0.0, -1.0],
                     "prop": f"bite{k}"} for k in range(1, BITES_MAX + 1)]}
    return {"size": [round(float(v), 4) for v in size], "origin": "tip", "front": "-Y", "slots": dict(roles),
            "use": use, "colliders": [], "params": {f"bite{k}": float(t) for k, t in enumerate(bites, 1)},
            "objects": objects(name)}
