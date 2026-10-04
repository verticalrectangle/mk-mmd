"""Grip frames in the world: where a prop's grip frame lies and where the wrist goes for a solved grip (docs/design.md:
Grips). numpy only, so the pose stage (Blender's Python) and the tests (no Blender) share one implementation.

A grip solver (mkmmd.solvers.grip) returns `target_in_wrist`: the style's grip frame expressed in the wrist bone's rest
frame at the wrist head. The build knows the grip frame G in the world (from the prop's card and its current
transform), so the wrist head frame is `W = G @ inv(target_in_wrist)`; the arm IK goal acts on the bone's tail, the
head frame moved along the bone by its length. Frames are 4x4 matrices whose columns are x, y, z and the origin."""
import math

import numpy as np

CLOCK = {"L": 10.0, "R": 2.0}        # hours on a clock face as the character sees a wheel: 12 = top, 3 = its right
WHEEL_APPROACH = 90.0                 # palm on the side of the rim that faces the character
WHEEL_WRAP = -1                       # fingers round the outside of the rim, then the front
RING_TYPES = {"ring", "wheel"}


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("a direction of zero length")
    return v / n


def frame_matrix(x, y, z, origin):
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


def ring_frame(center, axis, radius, up, clock, toward=None):
    """The grip frame on a ring (steering wheel, hoop) in the world: origin on the tube's centreline at `clock`
    o'clock, x radially outward there, z the ring axis, y = z cross x (counter-clockwise seen from +z).

    The character looks at the ring along -z (z points at the character: pass `toward`, a point on the character's
    side, and the axis is flipped if it points away). `up` is the prop's up direction; 12 o'clock is its component
    perpendicular to the axis and 3 o'clock lies to the character's right. Returns (4x4 frame, info) with the axis
    actually used and whether it was flipped."""
    c, a, u = np.asarray(center, float), unit(axis), np.asarray(up, float)
    flipped = False
    if toward is not None and float((np.asarray(toward, float) - c) @ a) < 0.0:
        a, flipped = -a, True
    u = u - (u @ a) * a
    if np.linalg.norm(u) < 1e-6:
        raise ValueError("the ring lies flat (its axis is vertical): 12 o'clock is undefined")
    u = unit(u)
    right = unit(np.cross(-a, u))                      # forward x up, the viewer looking along -axis
    th = math.radians(30.0 * float(clock))
    x = math.cos(th) * u + math.sin(th) * right
    return frame_matrix(x, np.cross(a, x), a, c + float(radius) * x), {"axis": a, "flipped": flipped}


def surface_frame(point, normal, heading):
    """The grip frame on a surface: origin at `point`, z the normal (out of the surface), x the heading projected on
    the surface, y = z cross x."""
    z = unit(normal)
    h = np.asarray(heading, float)
    h = h - (h @ z) * z
    if np.linalg.norm(h) < 1e-6:
        raise ValueError("the hand's heading is along the surface normal")
    x = unit(h)
    return frame_matrix(x, np.cross(z, x), z, np.asarray(point, float))


def pinch_frame(center, axis, normal, span, edge, toward=None):
    """The grip frame on a strap or handle: z the squeeze axis (axis x normal; the thumb side is the one toward the
    character when `toward` is given), x = -normal (from the hand toward the held part, the strap's depth `span` runs
    along it), y = z cross x (along the strap). The origin is `edge` inside the strap's outer face."""
    c, n, a = np.asarray(center, float), unit(normal), unit(axis)
    z = np.cross(a, n)
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("a pinch grip needs an axis (along the strap) and a normal (outward) that are not parallel")
    z = unit(z)
    if toward is not None and float((np.asarray(toward, float) - c) @ z) < 0.0:
        z = -z
    x = -n
    return frame_matrix(x, np.cross(z, x), z, c + n * (float(span) / 2.0 - float(edge)))


def wrist_goal(G, target_in_wrist, bone_length):
    """(wrist head frame, IK goal): the wrist whose `target_in_wrist` frame lies on G, and that frame moved along the
    bone by its length (the arm IK acts on the bone's tail)."""
    W = np.asarray(G, float) @ np.linalg.inv(np.asarray(target_in_wrist, float))
    T = np.eye(4)
    T[1, 3] = float(bone_length)
    return W, W @ T


def strum_frame(center, normal, along, tip, thumb="neck"):
    """The pinch frame of a pick whose tip rests at `center` on a guitar's strings: x = the pick's direction toward its tip
    = into the face (minus the face `normal`), z = the squeeze axis from the index pad to the thumb pad (the string
    direction `along`, toward the nut when the thumb is on the `neck` side, away from it for `bridge`), y = z cross x; the
    origin (midway between the pads) is `tip` above `center`, since the pick's tip protrudes `tip` beyond the pads."""
    n, a = unit(normal), unit(along)
    a = a - (a @ n) * n
    if np.linalg.norm(a) < 1e-6:
        raise ValueError("a strum zone's string direction lies along its normal")
    if thumb not in ("neck", "bridge"):
        raise ValueError(f"thumb must be \"neck\" or \"bridge\", got {thumb!r}")
    z = unit(a) * (1.0 if thumb == "neck" else -1.0)
    x = -n
    return frame_matrix(x, np.cross(z, x), z, np.asarray(center, float) + n * float(tip))


def clock_of(hand, side):
    return float(hand.get("clock", CLOCK[side]))


def card_style(entry, hand):
    """(style, prop dict, solver params) for a use.grip card entry and the pose.hands table that names it.

    ring / wheel   style wheel, prop {radius, tube}; hand keys approach (default 90) and wrap (default -1)
    pen            style pen, prop {length, radius, tip, nib_offset}; needs hand['posture'] (pen grips are tuned for
                   one hand shape and do not generalise yet)
    pinch          style pinch, prop {width, depth = span, length}; hand key edge
    neck           style neck, prop = the neck problem (mkmmd.core.fretting.solver_prop) for the hand's `fret` (the position:
                   the index finger's fret) and `chord` (a name or a table); hand key press (the pad's place behind the wire)
    strum          style pinch of the card's `pick` {thickness, length, width, tip}: width = thickness, depth = length, length =
                   width, edge = length - tip (hand key `tip` overrides the card's: how far the pick sticks out of the pads)
    Raises ValueError naming what is missing."""
    kind, name = entry.get("type"), entry.get("name", "?")

    def need(*keys):
        missing = [k for k in keys if entry.get(k) is None]
        if missing:
            raise ValueError(f"grip {name!r} ({kind}): the card lacks {', '.join(missing)}")
        return {k: entry[k] for k in keys}

    params = {}
    if hand.get("seeds") is not None:
        params["seeds"] = int(hand["seeds"])
    if kind in RING_TYPES:
        prop = need("radius", "tube")
        params.update(approach=float(hand.get("approach", WHEEL_APPROACH)), wrap=int(hand.get("wrap", WHEEL_WRAP)))
        return "wheel", prop, params
    if kind == "pen":
        if not isinstance(hand.get("posture"), dict):
            raise ValueError(f"grip {name!r} (pen): a pen grip needs `posture` (a table with the nib, shoulder, pole "
                             f"and table of the writing arm, see mkmmd.solvers.grip); pen grips are tuned for one hand "
                             f"shape and do not generalise yet")
        prop = need("length", "radius")
        prop.update({k: entry[k] for k in ("tip", "nib_offset") if entry.get(k) is not None})
        params["posture"] = hand["posture"]
        return "pen", prop, params
    if kind == "pinch":
        prop = need("width")
        for key, src in (("depth", "span"), ("length", "length")):
            if entry.get(src) is not None:
                prop[key] = entry[src]
        if hand.get("edge") is not None:
            params["edge"] = float(hand["edge"])
        return "pinch", prop, params
    if kind == "neck":
        from . import fretting as FR
        if hand.get("fret") is None or hand.get("chord") is None:
            raise ValueError(f"grip {name!r} (neck): give `fret` (the position, under the index finger) and `chord` "
                             f"({sorted(FR.CHORDS)} or a table)")
        return "neck", FR.solver_prop(entry, hand["chord"], int(hand["fret"]), float(hand.get("press", FR.PRESS)))[0], params
    if kind == "strum":
        pick = entry.get("pick")
        if not pick or any(pick.get(k) is None for k in ("thickness", "length", "width")):
            raise ValueError(f"grip {name!r} (strum): the card needs `pick` {{thickness, length, width, tip}}")
        length = float(pick["length"])
        tip = float(hand.get("tip", pick.get("tip", 0.008)))
        if not 0.0 < tip < length:
            raise ValueError(f"grip {name!r} (strum): the pick's tip ({tip * 1e3:g} mm) must be inside its length ({length * 1e3:g} mm)")
        params["edge"] = length - tip
        return "pinch", {"width": float(pick["thickness"]), "depth": length, "length": float(pick["width"])}, params
    raise ValueError(f"grip {name!r}: type {kind!r} has no grip style (ring, pen, pinch, neck, strum)")
