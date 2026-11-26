"""A child's craft scissors of card on a brass split pin (the library prop `paper_scissors`, docs/design.md: the library
table), bpy-free: the options, the two pieces as drawn (before the scissors: mkmmd/blender/library/props/
paper_scissors_parts.py cuts them and makes the card plates), how the blades turn with `open`, the drivers' expressions
and the card.

The prop's frame (metres): the origin on the split pin (the pivot), the blades along +Z (their tips at TIP), the loops
along -Z (their lowest point at TIP - LENGTH), the card sheet in the X-Z plane (its faces toward -Y and +Y). Piece `a`
lies in front (-Y): shut, its blade on the +X side of the middle line and its loop (the thumb's, round) on -X; piece `b`
behind it is its mirror with the longer finger loop. Each piece is a blade of grey card with a pencil line along its
cutting edge and a handle of coloured card glued on its outer face (round the pin, down the shank, round the loop).
`open` 0 .. 1 turns piece a by +TURN x open about +Y through the pin and piece b by -TURN x open: shut, the blades lie
over each other with their tips together; open, a V of 2 TURN degrees.

Modelled on children's safety scissors (blunt round tips; a round thumb loop and a longer loop for two fingers; the
plastic handles moulded over the blades' tangs up past the pivot), 13 cm long at the scale of a figure 1.5 m tall."""
import functools
import math

import numpy as np

from . import paper_theatre as PT

LENGTH = 0.13                   # tip to the bottom of the finger loop
TIP = 0.055                     # the blades' tips from the pin
TURN = 30.0                     # degrees each piece turns at open 1 (the blades a V of 60)
EDGE = 0.0025                   # a blade's cutting edge runs this far across the pin; its tip's round has this radius
BACK = 0.0072                   # the blade's back at its root, out from the pin's line (shut, the blades 2 BACK across)
ROOT_Z = 0.006                  # where the back leaves the disc round the pin
BACK_P = 1.6                    # how the back tapers to the tip (1 straight; more keeps it wide longer)
DISC = 0.0080                   # the blade card's disc round the pin
BOSS = ((0.0, 0.0015), (0.0090, 0.0120))     # the handle's round end over the pin: middle, half width and height
SHANK = 0.0052                  # the handle's half width between the boss and the loop
BLEND = 0.002                   # the handle's parts flow into each other over this
LOOPS = {"a": ((-0.018, -0.0574), (0.0095, 0.0085), 0.0052),     # middle, the hole's half width and height, its rim
         "b": ((0.021, -0.0605), (0.0130, 0.0090), 0.0055)}
CARD = {"blade": 0.0010, "handle": 0.0008, "gap": 0.0003}      # card thicknesses; the play between the two blades
CUT = 0.018                     # the cut point: on the V's middle line this far from the pin
PENCIL = (0.0018, 0.0155, 0.046, 0.001)      # a blade's pencil line: in from its cutting edge, from z, to z, its width
PIN = {"head": 0.0058, "dome": 0.0020, "shank": 0.0010, "leg": (0.0070, 0.0024, 0.00035)}   # leg: length, width, thick
PX = 0.00015                    # the drawing grid
STEP = 0.0008                   # the drawn outlines' point spacing

MATS = ("blade_a", "blade_b", "loop_a", "loop_b", "pencil_a", "pencil_b", "brad")    # material index -> material
MI = {k: i for i, k in enumerate(MATS)}
# material -> colour role: a material for each blade's pencil line, so a flat look can draw it in its blade's group
ROLE_OF = {k: "pencil" if k.startswith("pencil") else k for k in MATS}
ROLES = {"blade_a": "subtle:surface:0.45", "blade_b": "subtle:surface:0.25", "loop_a": "love", "loop_b": "gold",
         "pencil": "text:surface:0.2", "brad": "gold"}
ROLE_DOCS = {"blade_a": "the front piece's blade (card)", "blade_b": "the back piece's blade (card)",
             "loop_a": "the front piece's handle and thumb loop", "loop_b": "the back piece's handle and finger loop",
             "pencil": "the pencil lines along the cutting edges", "brad": "the brass split pin"}
DEFAULTS = {"open": 0.0}


# ===================================================================================================================
# options
# ===================================================================================================================
def options(slots):
    """The prop's options from its `slots`, checked: {open} (its unkeyed value, 0 shut .. 1 open)."""
    v = (slots or {}).get("open", DEFAULTS["open"])
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0.0 <= v <= 1.0:
        raise ValueError(f"paper_scissors: open is a number from 0 (shut) to 1 (open) (got {v!r})")
    return {"open": float(v)}


# ===================================================================================================================
# the pieces as drawn (x across, z along, in the piece's own frame: the prop's frame when shut)
# ===================================================================================================================
def _smax(f, g, k):
    """The union of two fields with a round fillet about k wide where they meet."""
    def h(X, Z):
        a, b = f(X, Z), g(X, Z)
        return np.maximum(a, b) + np.maximum(k - np.abs(a - b), 0.0) ** 2 / (4.0 * k)
    return h


def _tube(path, r):
    """A field r - (the distance to the open polyline `path`)."""
    A = np.asarray(path, float)[:-1]
    AB = np.asarray(path, float)[1:] - A
    L2 = np.maximum((AB * AB).sum(1), 1e-18)

    def f(X, Z):
        Q = np.stack([X.ravel(), Z.ravel()], 1)
        d = np.full(len(Q), np.inf)
        for a, ab, l2 in zip(A, AB, L2):
            t = np.clip(((Q - a) @ ab) / l2, 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(Q - (a + t[:, None] * ab), axis=1))
        return (r - d).reshape(X.shape)
    return f


def blade_polygon():
    """Piece a's blade (x, z), counter-clockwise: its straight cutting edge at x = -EDGE, the round tip on the pin's
    line with its top at TIP, the back tapering from BACK at ROOT_Z into the tip; on down into the disc round the pin."""
    u = np.linspace(0.0, 1.0, 25)
    back = np.stack([EDGE + (BACK - EDGE) * (1.0 - u ** BACK_P), ROOT_Z + u * (TIP - EDGE - ROOT_Z)], 1)
    a = np.linspace(0.0, math.pi, 13)
    tip = np.stack([EDGE * np.cos(a), TIP - EDGE + EDGE * np.sin(a)], 1)
    low = -0.5 * DISC
    return np.concatenate([[[BACK, low]], back, tip[1:], [[-EDGE, low]]])


def _mirror(P):
    Q = np.asarray(P, float) * [-1.0, 1.0]
    return Q[::-1].copy()


@functools.lru_cache(maxsize=1)
def _outlines():
    blade = PT.drawn(_smax(PT.shape(blade_polygon()), PT.disc(0.0, 0.0, DISC), BLEND),
                     (-DISC - 0.001, -DISC - 0.001), (DISC + 0.001, TIP + 0.001), PX, STEP)
    out = {"blade_a": blade, "blade_b": _mirror(blade)}
    (bx, bz), (ba, bb) = BOSS
    for side, ((cx, cz), (hx, hz), rim) in LOOPS.items():
        t = np.linspace(0.0, 1.0, 16)[:, None]
        p0, p1, p2 = np.array([0.0, -0.002]), np.array([0.35 * cx, 0.5 * cz]), np.array([cx, cz])
        path = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t ** 2 * p2           # the shank: a gentle curve
        field = _smax(_smax(PT.oval(bx, bz, ba, bb), _tube(path, SHANK), BLEND), PT.oval(cx, cz, hx + rim, hz + rim),
                      BLEND)
        lo = (min(cx - hx - rim, -ba) - 0.002, cz - hz - rim - 0.002)
        hi = (max(cx + hx + rim, ba) + 0.002, bz + bb + 0.002)
        out[f"handle_{side}"] = PT.drawn(field, lo, hi, PX, STEP)
        out[f"hole_{side}"] = PT.drawn(PT.oval(cx, cz, hx, hz), (cx - hx - 0.001, cz - hz - 0.001),
                                       (cx + hx + 0.001, cz + hz + 0.001), PX, STEP)
    for v in out.values():
        v.setflags(write=False)
    return out


def outlines():
    """The pieces as drawn, each (m, 2) of (x, z) counter-clockwise in its piece's frame: `blade_a`, `blade_b` (the grey
    cards: blade and the disc round the pin), `handle_a`, `handle_b` (the coloured cards: the boss over the pin, the
    shank, the loop) and `hole_a`, `hole_b` (the loops' holes)."""
    return {k: v.copy() for k, v in _outlines().items()}


def layers():
    """y of each card's front and back face, front to back: {"handle_a", "blade_a", "blade_b", "handle_b": (front,
    back)}; the blades' cards face each other across the gap at y = 0."""
    g, b, h = 0.5 * CARD["gap"], CARD["blade"], CARD["handle"]
    return {"handle_a": (-g - b - h, -g - b), "blade_a": (-g - b, -g), "blade_b": (g, g + b),
            "handle_b": (g + b, g + b + h)}


# ===================================================================================================================
# the turn
# ===================================================================================================================
def angles(open_):
    """(piece a, piece b): their turns about +Y in radians at `open` (clamped to 0 .. 1, as the drivers clamp it)."""
    t = np.clip(np.asarray(open_, float), 0.0, 1.0) * math.radians(TURN)
    return t, -t


def turned(P, angle):
    """Points (n, 2) of (x, z) in a piece's frame, the piece turned by `angle` (radians) about +Y: the prop's frame."""
    P = np.asarray(P, float)
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.stack([c * P[:, 0] + s * P[:, 1], -s * P[:, 0] + c * P[:, 1]], 1)


def tip(open_):
    """The point midway between the two blades' tips at `open` (n, 3) (what `<name>_tip` rides): on the V's middle line,
    TIP x cos(the turn) from the pin."""
    a, _b = angles(np.atleast_1d(open_))
    return np.stack([np.zeros_like(a), np.zeros_like(a), TIP * np.cos(a)], 1)


def hold(open_=0.0):
    """The point midway between the two loops' middles at `open` (n, 3) (what `<name>_hold` rides): where the thumb and
    the fingers go. The loops swing apart and up toward the pin as the scissors open."""
    a, _b = angles(np.atleast_1d(open_))
    (ax, az), (bx, bz) = LOOPS["a"][0], LOOPS["b"][0]
    x = 0.5 * ((ax + bx) * np.cos(a) + (az - bz) * np.sin(a))
    z = 0.5 * ((bx - ax) * np.sin(a) + (az + bz) * np.cos(a))
    return np.stack([x, np.zeros_like(a), z], 1)


def cut_shut():
    """The `open` below which the blades cover the cut point: their cutting edges meet over it as they close."""
    return math.degrees(math.asin(EDGE / CUT)) / TURN


def expressions():
    """The drivers' expressions (Blender simple expressions of the root's `open`): `blade_a` and `blade_b` their
    rotation_euler[1], `tip` the tip empty's location[2], `hold_x` and `hold_z` the hold empty's location[0] and [2]."""
    t = f"clamp(open, 0.0, 1.0) * {math.radians(TURN):.10f}"
    (ax, az), (bx, bz) = LOOPS["a"][0], LOOPS["b"][0]
    return {"blade_a": t, "blade_b": f"-{t}", "tip": f"{TIP:g} * cos({t})",
            "hold_x": f"{0.5 * (ax + bx):.6f} * cos({t}) + {0.5 * (az - bz):.6f} * sin({t})",
            "hold_z": f"{0.5 * (bx - ax):.6f} * sin({t}) + {0.5 * (az + bz):.6f} * cos({t})"}


# ===================================================================================================================
# the card
# ===================================================================================================================
def objects(name):
    return {"blade_a": f"{name}_blade_a", "blade_b": f"{name}_blade_b", "pin": f"{name}_pin",
            "pivot": f"{name}_pivot", "tip": f"{name}_tip", "cut": f"{name}_cut", "hold": f"{name}_hold"}


def bounds():
    """{min, max}: everything the scissors cover at any opening in the prop's frame (the cut's wobble and the pin in)."""
    O = _outlines()
    pts = []
    for o in np.linspace(0.0, 1.0, 13):
        ta, tb = angles(o)
        for side, t in (("a", ta), ("b", tb)):
            pts += [turned(O[f"blade_{side}"], t), turned(O[f"handle_{side}"], t)]
    P = np.concatenate(pts)
    lay = layers()
    m = 0.0006
    lo = [P[:, 0].min() - m, lay["handle_a"][0] - PIN["dome"], P[:, 1].min() - m]
    hi = [P[:, 0].max() + m, lay["handle_b"][1] + PIN["leg"][2], P[:, 1].max() + m]
    return {"min": [round(float(v), 5) for v in lo], "max": [round(float(v), 5) for v in hi]}


def card(name, O, roles):
    """The card: `length`, `bounds` (every opening), use.grip `hold` (riding `<name>_hold`; with the fields of a pinch
    of both handles across the gap between the loops, its near edge the loops' bottom), use.look `pivot`, `tip` and
    `cut` (each riding its empty) and `scissors` (the numbers and the objects)."""
    ob = objects(name)
    b = bounds()
    hx, _hy, hz = hold(0.0)[0]
    lay = layers()
    (ax, _az), (ha, _), ra = LOOPS["a"]
    (bx, _bz), (hb, _), rb = LOOPS["b"]
    r5 = lambda v: round(float(v), 5)                                          # noqa: E731
    grip = {"name": "hold", "point": [r5(hx), 0.0, r5(hz)], "object": ob["hold"], "dir": [0.0, 0.0, 1.0],
            "palm": [0.0, -1.0, 0.0], "type": "pinch", "center": [r5(hx), 0.0, r5(hz)], "axis": [1.0, 0.0, 0.0],
            "normal": [0.0, 0.0, -1.0], "width": r5(lay["handle_b"][1] - lay["handle_a"][0]),
            "span": r5(2.0 * (hz - (TIP - LENGTH))), "length": r5(bx + hb + rb - (ax - ha - ra))}
    return {"size": [r5(h - lo) for lo, h in zip(b["min"], b["max"])], "origin": "keep", "front": "-Y",
            "blocks": False, "bounds": b, "layers": [], "colliders": [], "slots": dict(roles), "length": LENGTH,
            "use": {"grip": [grip],
                    "look": [{"name": "pivot", "point": [0.0, 0.0, 0.0], "object": ob["pivot"]},
                             {"name": "tip", "point": [0.0, 0.0, TIP], "object": ob["tip"]},
                             {"name": "cut", "point": [0.0, 0.0, CUT], "object": ob["cut"]}]},
            "scissors": {"length": LENGTH, "tip": TIP, "turn": TURN, "cut": CUT, "cut_shut": r5(cut_shut()),
                         "open": O["open"], "objects": ob}}
