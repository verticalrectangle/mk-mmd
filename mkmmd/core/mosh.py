"""Mosh-pit moves, bpy-free (docs/design.md: Moves): the routines of a mosh pit in the moves library. Like the gag moves
(mkmmd.core.gags) they are played by mkmmd.core.moves.compile as it plays the others (their hands travel, rest, stay
natural and keep off the body the same way), their places are merged into its tables, and `plan` gives one entry's
keys:
- `pound`: a mallet (`prop`: `use.grip` `hold`, its handle, and `use.anchor` `face`, the middle of its striking face and
  the way it faces) in both fists, the move's hand toward the head and the other at the handle's end, raised overhead and
  swung down on each hit, the face meeting `target` (a stake at the floor, a giant mochi bowl) on the hit (`strike`
  solves where along the handle the fists hold it and how far it points down to bring the hands to a natural place); the
  wind-up is its first place, the body leans into the blow (`lean`) and the knees give after it (`crouch`) while the face
  stays on the target. The mallet rides the move's hand (`held`), the other fist closes round the handle where it is;
- `headbang`: deep nods of the upper body (`lean`) and the head (`head_nod`) down on each hit and up between, `depth` a
  share of a full one; the hands free (they hang from the shoulders and swing with the body) or on the hips (`hands =
  "hips"`). The hair whips by itself in the sim;
- `pogo`: a straight-up jump taking off on each hit, as high as a body thrown up flies in the time to the next one (a
  parabola, `rise`, at most POGO["height"]), the legs tucking in the air (`air`), the knees giving on each landing and
  pushing off into the next (`crouch`), both fists up (the bounce's `pump_up`), dipping as she lands;
- `wotagei`: a penlight in each fist (`props = {L, R}`: each `use.grip` `hold`, its axis toward the light) waved in a
  routine on the beats (`routine`, ROUTINES): `arcs` (both lights swept overhead from side to side, a side a beat),
  `eights` (each light drawing a figure eight, mirrored, one every two beats), `punch` (the fists punching up and
  forward in turns on every hit, or on the beats and halfway between: the eighths) or `circle` (both arms circling in
  front, a turn every two beats, half a turn apart: a windmill);
- `whistle`: a train conductor pulling the whistle's cord: the move's hand up by the head pulls it down on each hit
  and lets it up between, the other hand on the hip;
- `bass_face`: a face on each hit (`style`: a `scowl`, the brows down and the mouth turned down, or a `pout`), the eyes
  narrowed, with a short nod of the head (`head_nod`).

A held prop rides the fist of its hand as the gag moves' do (mkmmd.core.gags: the grip's middle in the fist's hole, its
axis out of the thumb side): the mallet in the move's hand with the grip its blow sets (`strike`), a penlight in each fist
(`_stick`). Everything is in the character's frame (x its left, y behind it, z up, metres from its root), the places
written for the left hand and mirrored for the right, like mkmmd.core.moves' own; a move's body keys (`lean`,
`head_nod`) are curves of its own, keyed in perform as any move's are. The hands ride the chest, which leans about the
waist (the landmark `waist`, the upper body's joint) and drops with the crouch, so a place that must meet the world (the
mallet's face on the target) is written where the chest carries it there (`carried`)."""
import math

import numpy as np

from . import gags as GG

GRAVITY = 9.81


def _mv():
    """mkmmd.core.moves, which imports this module for its tables (so it is imported when it is needed, not before)."""
    from . import moves
    return moves


def _v(*a):
    return np.array(a, float)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------------------------------------- the tables
OUT = {  # place: (direction from the arm joint for the left hand, x mirrored for the right; share of the reach)
    "cord": ((0.42, -0.30, 1.0), 0.93),           # the whistle's cord gripped up beside the head ...
    "cord_pull": ((0.55, -0.42, 0.72), 0.80),     # ... and pulled down
    "wota_side": ((0.55, -0.60, 0.60), 0.92),     # arcs: up and out to the hand's own side, in front ...
    "wota_top": ((0.0, -0.55, 0.85), 0.93),       # ... up in front of the head ...
    "wota_over": ((-0.50, -0.75, 0.45), 0.92),    # ... and over to the other side
    "wota_eight": ((0.25, -0.75, 0.55), 0.85),    # eights: the middle of the figure, in front of the shoulder
    "wota_punch": ((-0.05, -0.85, 0.50), 0.96),   # punch: thrown up and forward ...
    "wota_pull": ((0.35, -0.85, 0.05), 0.62),     # ... and pulled back in by the shoulder
    "wota_ring_top": ((0.15, -0.60, 0.78), 0.93),  # circle: its top, its outer side, its bottom and its inner side
    "wota_ring_out": ((0.80, -0.50, 0.25), 0.90),
    "wota_ring_low": ((0.35, -0.70, -0.45), 0.86),
    "wota_ring_in": ((-0.30, -0.90, 0.15), 0.78),
}
AIM = {  # a fist round a handle: where it points out of the thumb side, for the left hand (as mkmmd.core.moves.AIM),
    # chosen so the hand continues its forearm within about 40 degrees and the forearm turns no further than it
    # comfortably does, on any body, the palm toward it: the cord up and back from the fist; a penlight up, each leaning
    # in over the head
    "cord": (-0.27, 0.49, 0.83),
    "cord_pull": (-0.27, 0.49, 0.83),
    "wota_side": (-0.28, 0.11, 0.95),
    "wota_top": (-0.56, 0.44, 0.70),
    "wota_over": (-0.17, 0.32, 0.93),
    "wota_eight": (-0.33, 0.19, 0.92),
    "wota_punch": (-0.20, 0.14, 0.97),
    "wota_pull": (0.02, -0.05, 1.0),
    "wota_ring_top": (-0.40, 0.30, 0.86),
    "wota_ring_out": (-0.20, 0.10, 0.97),
    "wota_ring_low": (0.10, -0.50, 0.86),
    "wota_ring_in": (-0.20, 0.20, 0.96),
}
POLES = {}
SHAPES = {}
BOX = {}
STRAIGHT = {}
WOTA = ("wota_side", "wota_top", "wota_over", "wota_eight", "wota_punch", "wota_pull", "wota_ring_top", "wota_ring_out",
        "wota_ring_low", "wota_ring_in")
PLACES = ("cord", "cord_pull") + WOTA
ROUTINES = {  # routine: the places each hand plays (rows: the hand checks hold its hands there) and whether the two
    # hands mirror each other while they do (the hand_mirror check)
    "arcs": (("wota_side", "wota_top", "wota_over"), False),
    "eights": (("wota_eight",), True),
    "punch": (("wota_punch", "wota_pull"), False),
    "circle": (("wota_ring_top", "wota_ring_out", "wota_ring_low", "wota_ring_in"), False),
}
EIGHT = (0.09, 0.06)                               # m: the figure eight's half width and half height
STYLES = {  # bass_face: the morphs of each face and their values on the hit
    "scowl": (("angry", 0.9), ("jito", 0.8), ("mouth_down", 0.7)),
    "pout": (("u", 0.55), ("jito", 0.7), ("troubled", 0.4)),
}
ACCENT = {"attack": 0.05, "hold": 0.1, "release": 0.22, "low": 0.55,   # s: the face's attack, hold and release; the
          "nod": 9.0, "nod_at": 0.06, "nod_back": 0.24}                # share it keeps between close hits; the nod (deg),
#                                                                        when it is deepest after the hit and back (s)
BANG = {"lean": 20.0, "nod": 34.0, "lean_up": -3.0, "nod_up": -8.0,    # degrees down on the hit, and up between, at
        "up": 0.55, "most": 2.0}                                       # depth 1; the share of the gap it is up at; the
#                                                                        deepest depth
POGO = {"flight": 0.42, "ground": 0.13, "least": 0.16, "height": 0.2,  # s in the air at most, on the ground at least, a
        "knees": 0.06, "tuck": 0.1, "dip": 0.05, "steps": 8}           # hop's least flight, m at most; m the knees give,
#                                                                        the legs tuck, the fists dip; keys per parabola
POUND = {
    "spread": 0.11,              # m between the two fists on the handle (side by side; the move's hand toward the head)
    "end": 0.03,                 # m: the lower fist this far in from the end of the grip span
    "fist": (-42.0, 0.90),       # the lower fist at the blow, in the frame the hands ride: degrees below ahead from the
                                 # middle of the arm joints and distance (share of `reach`): low before the belly
    "tilt": (-40.0, -12.0, -26.0),   # degrees the handle points down from level at the blow in that frame (-: up, the
                                     # body's lean bringing it down level): least, most, preferred; a fist round a handle
                                     # pointing further along the forearm would bend the wrist past what it can
    "top": (15.0, 0.80, 125.0),  # the lower fist at the top of the wind-up (degrees above ahead, share of reach), the
                                 # handle's angle from ahead there (past upright: the head up and behind hers)
    "lean": (5.0, 45.0, 30.0),   # degrees the body leans into the blow: least, most, preferred ...
    "brace": (0.0, 0.12, 0.0),   # ... and the hips drop (m), lower targets taking more of both
    "clear": 0.11,               # m: the lower fist at least this far ahead of the waist at the blow (off the thighs)
    "near": 0.015,               # m: a blow lands the face this close to its target by the lean, brace, tilt and grip
    "lean_top": -6.0,            # degrees the body leans back at the top of the wind-up
    "give": 0.045,               # m the knees give after the blow, deepest as the face is held on the target ...
    "give_back": 0.30,           # ... and back up this long after the blow (s) when no other comes before
    "hold": 0.07,                # s the face stays on the target after the blow
    "swing": 0.24,               # s the down-swing takes from the top (less for a lower wind-up)
    "full": 0.85,                # s from one blow's hold to the next blow that raise it all the way ...
    "least": 0.3,                # ... and the least share of the way it goes up
    "raise": 0.6,                # s the raise takes at most (it then waits at the top)
    "step": 30.0,                # degrees the handle turns between two keys of the swing, at most
}
GRIP_REACH = GG.GRIP_REACH
WAIST = 0.22                     # the waist below the chest, a share of `reach`, for a member without `waist`

# hand: "mosh" (keys made by `plan`); both: both hands play it; prop: "need" (a held prop, `prop`); props: a prop in
# each hand (`props = {L, R}`); hits: it plays on its own hits; keys: the entry keys of its own; the rest as
# mkmmd.core.moves.MOVES (its face held over it, its body keys)
MOVES = {
    "pound": {"hand": "mosh", "prop": "need", "hits": True, "keys": ("prop", "target"), "face": [("serious", 0.5)]},
    "headbang": {"hand": "mosh", "hits": True, "keys": ("depth", "hands")},
    "pogo": {"hand": "mosh", "both": True, "hits": True, "keys": (),
             "face": [("cheerful", 0.5), ("mouth_smile", 0.4)]},
    "wotagei": {"hand": "mosh", "both": True, "hits": True, "keys": ("props", "routine"),
                "face": [("cheerful", 0.5), ("smile_eyes", 0.3)]},
    "whistle": {"hand": "mosh", "hits": True, "mirror": True, "lean": -4.0, "head_tilt": 5.0, "keys": (),
                "face": [("cheerful", 0.5), ("smile_eyes", 0.4)]},
    "bass_face": {"hand": "mosh", "hits": True, "keys": ("style",)},
}
KEYS = ("target", "depth", "hands", "props", "routine", "style")   # the entry keys only mosh moves read


def place(name, side, marks):
    """(point, dir, palm, shape) of the mosh place `name` (PLACES) for the hand `side` as written, in the character's
    frame (as mkmmd.core.moves.place): the wrist, the way the fingers point, the way the palm faces, the finger shape.
    The places with a fist round a handle (AIM) are aimed by mkmmd.core.moves.resolve; their `dir` and `palm` only say
    the hand's way before it is."""
    s = 1.0 if side == "L" else -1.0
    arm = np.asarray(marks[f"arm.{side}"], float)
    R = float(marks["reach"])
    d = _unit(OUT[name][0])
    at = arm + OUT[name][1] * R * _v(s * d[0], d[1], d[2])
    shape = "curled" if name.startswith("cord") else "fist"
    return at, _v(0.0, -0.3, 1.0), _v(-s, 0.0, 0.0), shape


# ------------------------------------------------------------------------------------------------- the frame of a lean
def waist(marks):
    """The point the upper body leans about (perform's `lean`: the upper body's joint), in the character's frame: the
    landmark `waist`, else WAIST of `reach` below the chest."""
    w = marks.get("waist")
    if w is not None:
        return np.asarray(w, float)
    return np.asarray(marks["chest"], float) - _v(0.0, 0.0, WAIST * float(marks["reach"]))


def _rx(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def carried(point, marks, lean=0.0, crouch=0.0):
    """Where a hand riding the chest must be written (the character's frame at rest) for the chest to carry it to `point`
    when the upper body leans `lean` degrees forward about the waist (`waist`) and the hips drop `crouch` metres."""
    W = waist(marks)
    return W + _rx(-lean) @ (np.asarray(point, float) + _v(0.0, 0.0, crouch) - W)


def lifted(point, marks, lean=0.0, crouch=0.0):
    """Where the chest carries a point written at `point` (the inverse of `carried`)."""
    W = waist(marks)
    return W + _rx(lean) @ (np.asarray(point, float) - W) - _v(0.0, 0.0, crouch)


# ---------------------------------------------------------------------------------------------------------- the clock
def _curve(f, a, b, n):
    """[[t, f(u)], ...] at n + 1 times from a to b (u from 0 to 1): a shape perform's eased keys follow closely."""
    return [[round(a + (b - a) * k / n, 4), round(float(f(k / n)), 5)] for k in range(n + 1)]


def _gaps(hs, t1):
    """[(hit, the time to the next one, or to t1 after the last)]."""
    return [(h, nb - h) for h, nb in zip(hs, list(hs[1:]) + [t1])]


def _eighths(e, t0, t1, beats):
    """The entry's own hits, or the beats inside it and halfway between them (and to the end): its eighths."""
    hs = GG._times(e, t0, t1, beats)
    if "hits" in e:
        return hs
    out = []
    for b, nb in zip(hs, hs[1:] + [t1]):
        out += [b, 0.5 * (b + nb)] if nb - b > 0.2 else [b]
    return out


# ---------------------------------------------------------------------------------------------------------- holding
def _face(prop, geom):
    """(point, dir) of a mallet's striking face (`use.anchor` `face`), scaled."""
    a, k = GG._anchor(prop, geom, "face")
    if "point" not in a:
        raise GG.GagError(f"prop {prop!r}: its use.anchor `face` needs a point (and dir, the way the face faces)")
    return np.asarray(a["point"], float) * k, _unit(a.get("dir", (1.0, 0.0, 0.0)))


def clasp(side, marks, X, m, hole, rounds=8, pole=None):
    """(wrist, dir, palm, forearm, pole) of the hand `side` with its fist (`hole`: mkmmd.core.gags.fist's) round a handle
    along `m` (toward its business end, out of the thumb side) at the point X: the hand continues its forearm as far as
    the fist allows (mkmmd.core.moves.aimed), the elbow turned to keep the arm's twists in range (from `pole`, a point,
    when it does)."""
    MV = _mv()
    X, m = np.asarray(X, float), _unit(m)
    W = X.copy()
    d = p = f = None
    for _ in range(rounds):
        d, p, f, pole = MV.arm_for(side, marks, W, lambda ff: MV.aimed(side, ff, m), pole, MV.HELD, None)
        miss = X - (W + MV._off(hole, d, p, side))
        W = W + miss
        if float(np.linalg.norm(miss)) < 1e-5:
            break
    d, p, f, pole = MV.arm_for(side, marks, W, lambda ff: MV.aimed(side, ff, m), pole, MV.HELD, None)
    return W, d, p, f, pole


def _swing_plane(marks, target):
    """(C, h, n): the middle of the arm joints, the level way from it to `target` (straight ahead without one) and the
    way square to it on the level (its left): the plane a two-hand swing stays in."""
    C = 0.5 * (np.asarray(marks["arm.L"], float) + np.asarray(marks["arm.R"], float))
    h = _v(0.0, -1.0, 0.0)
    if target is not None:
        g = (np.asarray(target, float) - C) * _v(1.0, 1.0, 0.0)
        if float(np.linalg.norm(g)) > 0.05:
            h = _unit(g)
    return C, h, _unit(np.cross(_v(0.0, 0.0, 1.0), h))


def _polar(C, h, n, X):
    """(degrees above ahead, distance in the plane, distance to the left of it) of X about C in the swing plane."""
    r = np.asarray(X, float) - C
    a, z = float(r @ h), float(r[2])
    return math.degrees(math.atan2(z, a)), math.hypot(a, z), float(r @ n)


def _at(C, h, n, deg, rho, side=0.0):
    t = math.radians(deg)
    return C + rho * (math.cos(t) * h + math.sin(t) * _v(0.0, 0.0, 1.0)) + side * n


def _handle(h, deg):
    """The way along the swing plane `deg` degrees above ahead (`h`): a handle's way from the fists to the head."""
    t = math.radians(deg)
    return _unit(math.cos(t) * h + math.sin(t) * _v(0.0, 0.0, 1.0))


def strike(prop, geom, side, marks, target=None, low=None):
    """The blow of a mallet held in both fists, the hand `side` toward the head: {"lean", "brace": the degrees the body
    leans into it and the metres the hips drop, "tilt": degrees the handle points down from level in the frame the hands
    ride (negative: up), "low", "up": metres from the business end of the grip span where the other fist (nearer the
    handle's end) and the hand `side` hold it, "target": where the face meets in that frame (`carried`), "face": the way
    the face faces there, "R", "origin": the mallet's pose there, "m": the handle's way, "plane": (C, h, n), "point",
    "dir": the face's point and way on the mallet, "hold": its grip span}. The fists are side by side (POUND["spread"]),
    the lower one at POUND["fist"] in the frame the hands ride, and the face meets `target` (the character's frame at
    rest; None: wherever that blow brings it, with the lean, brace, tilt and grip preferred) by the lean and the brace of
    the body, the tilt of the handle and where along it the fists hold it, the nearest the preferred of each that brings
    the face within POUND["near"] of it with the lower fist off the thighs (POUND["clear"] ahead of the waist), the
    grip at `low` when a first blow set it. The face's own `dir` is turned down onto it, square to the handle in the
    swing plane (through the middle of the arm joints toward the target)."""
    hold = GG.hold_of(prop, geom)
    fp, fd = _face(prop, geom)
    span, spread = hold["length"], POUND["spread"]
    if span < spread + POUND["end"] + 0.02:
        raise GG.GagError(f"prop {prop!r}: its grip span ({span:.2f} m) is too short for two fists")
    C, h, n = _swing_plane(marks, target)
    X = _at(C, h, n, POUND["fist"][0], POUND["fist"][1] * float(marks["reach"]))
    up = _v(0.0, 0.0, 1.0)
    lows = np.array([float(low)] if low is not None else np.arange(span - POUND["end"], spread - 1e-9, -0.01))

    def pose(tilt):
        m = _handle(h, -tilt)
        face_to = -_unit(math.sin(math.radians(tilt)) * h + math.cos(math.radians(tilt)) * up)
        return m, face_to, GG.turned(hold["axis"], fd, m, face_to)

    def faces(tilt):
        """(len(lows), 3) where the face is with the lower fist at X, the handle tilted so, for each grip."""
        _, _, Rm = pose(tilt)
        G = np.array([GG.grip_point(hold, s) for s in lows])
        return X + (fp - G) @ Rm.T

    (l0, l1, lp), (b0, b1, bp), (a0, a1, ap) = POUND["lean"], POUND["brace"], POUND["tilt"]
    W0 = waist(marks)
    if target is None:
        lean, brace, tilt, k = lp, bp, ap, 0
    else:
        T = np.asarray(target, float)
        tilts = np.arange(a0, a1 + 1e-9, 1.0)
        F = np.stack([faces(t) for t in tilts])                     # (tilts, lows, 3)
        best = None
        for lean in np.arange(l0, l1 + 1e-9, 1.0):
            for brace in np.arange(b0, b1 + 1e-9, 0.01):
                if float((lifted(X, marks, lean, brace) - W0) @ h) < POUND["clear"]:
                    continue                                     # the fists would come down on the thighs
                miss = np.linalg.norm(F - carried(T, marks, lean, brace), axis=2)
                cost = (((lean - lp) / 10.0) ** 2 + (brace / 0.04) ** 2 + ((tilts[:, None] - ap) / 6.0) ** 2
                        + ((lows[0] - lows[None, :]) / 0.15) ** 2 + (miss / POUND["near"]) ** 2)
                cost = np.where(miss <= POUND["near"], cost, np.inf)
                i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
                if np.isfinite(cost[i, j]) and (best is None or cost[i, j] < best[0]):
                    best = (float(cost[i, j]), float(lean), float(brace), float(tilts[i]), j)
        if best is None:
            m, face_to, Rm = pose(ap)
            near = lifted(faces(ap)[0], marks, lp, bp)
            raise GG.GagError(f"its target {[round(float(v), 2) for v in target]} is out of reach of the mallet in her "
                              f"hands (her own frame: x her left, y behind her, z up): put it about "
                              f"{[round(float(v), 2) for v in near]}")
        _, lean, brace, tilt, k = best
    m, face_to, Rm = pose(tilt)
    lo = float(lows[k])
    origin = X - Rm @ GG.grip_point(hold, lo)
    goal = carried(target, marks, lean, brace) if target is not None else origin + Rm @ fp
    return {"lean": lean, "brace": brace, "tilt": tilt, "low": lo, "up": lo - spread, "target": goal,
            "face": face_to, "R": Rm, "origin": origin, "m": m, "plane": (C, h, n), "point": fp, "dir": fd,
            "hold": hold}


def rigid(name):
    """Whether the place `name` is one of a pound's (`pound_places`): both fists on one handle, so a trip between two
    of them is keyed by the move itself (mkmmd.core.moves._via puts no key halfway, which would take one fist off it)."""
    return name.startswith("pound") and "@" in name


def pound_places(prop, geom, side, marks, target, held, tag):
    """The places of a pound of `prop` with the move's hand `side` toward the head: ([u, ...], {"L": {name: resolved},
    "R": {...}}, the blow) for each share u of the way up of a key of the swing (0 the blow, 1 the top of the wind-up),
    named f"pound{k}@{prop}{tag}" (k the step of `us`), and f"pound_give@{prop}{tag}": the blow held on the target
    while the knees give (the mallet POUND["give"] higher in the frame the hands ride, so the face stays down; a place of
    its own, its hands turned for the arms that holds them). The first blow (`strike`) sets the grip (`held`: the mallet
    is held one way, so a later pound of it holds it where the first did); every other key is the swing's: the lower fist
    on the arc about the middle of the arm joints from the blow's to the top's (POUND["top"]), the handle turning in the
    plane from its tilt at the blow to its angle at the top at most POUND["step"] degrees a key, the mallet in the hand
    `side` (`fit`) and the other fist closed round it at its grip (`clasp`), each elbow from the one the key before
    has, so the arms swing round with the mallet."""
    other = "R" if side == "L" else "L"
    st = strike(prop, geom, side, marks, target, held.get("low"))
    hold, (C, h, n) = st["hold"], st["plane"]
    grip_up = held.get("grip") or GG._grip(side, marks, hold, st["up"])
    shape_low, hole_low = GG.fist(marks, GG.radius_at(hold, st["low"]))
    low_pt = GG.grip_point(hold, st["low"])
    rel = held.get("rel")
    W, d, p, Rm, origin, f, pole = GG.fit(side, marks, st["target"], st["point"], st["m"], grip_up, rel,
                                          (st["dir"], st["face"]))
    if rel is None:                                     # the first blow sets how it sits in the fists
        H = GG._hand_frame(d, p)
        rel = (H.T @ Rm, H.T @ (origin - W))
        held.update(grip=grip_up, rel=rel, low=st["low"], ref=(W, d, p, Rm, origin))
    a0, r0, s0 = _polar(C, h, n, origin + Rm @ low_pt)
    m0 = Rm @ hold["axis"]
    p0 = math.degrees(math.atan2(float(m0[2]), float(m0 @ h)))
    a1, r1, p1 = POUND["top"][0], POUND["top"][1] * float(marks["reach"]), POUND["top"][2]
    steps = max(math.ceil(abs(p1 - p0) / POUND["step"]), 1)
    us = [k / steps for k in range(steps + 1)]
    out = {"L": {}, "R": {}}
    poles = {}

    def both(name, Wk, dk, pk, Rk, ok, fk, polek):
        out[side][name] = GG._resolved(side, marks, Wk, dk, pk, fk, polek, grip_up[0])
        Wo, do, po, fo, poles[other] = clasp(other, marks, ok + Rk @ low_pt, Rk @ hold["axis"], hole_low,
                                             pole=poles.get(other))
        out[other][name] = GG._resolved(other, marks, Wo, do, po, fo, poles[other], shape_low)

    give = GG.fit(side, marks, st["target"] + _v(0.0, 0.0, POUND["give"]), st["point"], st["m"], grip_up, rel,
                  pole=pole)
    both(f"pound_give@{prop}{tag}", *give)
    for k, u in enumerate(us):
        if k:
            X = _at(C, h, n, a0 + (a1 - a0) * u, r0 + (r1 - r0) * u, s0 * (1.0 - u))
            W, d, p, Rm, origin, f, pole = GG.fit(side, marks, X, low_pt, _handle(h, p0 + (p1 - p0) * u), grip_up,
                                                  rel, pole=pole)
        both(f"pound{k}@{prop}{tag}", W, d, p, Rm, origin, f, pole)
    return us, out, st


def _stick(side, marks, hold, front):
    """(grip, rel) of a stick held in the fist of the hand `side` by the middle of its grip span: the fist closed round
    it (`fist`), its axis out of the thumb side (as every aimed place has it: mkmmd.core.moves.aimed) and its `front`
    toward the knuckles, (rotation, offset) in the hand's frame (columns its direction, its palm and their cross)."""
    shape, hole = GG.fist(marks, GG.radius_at(hold, 0.5 * hold["length"]))
    G = GG.grip_point(hold, 0.5 * hold["length"])
    s = 1.0 if side == "L" else -1.0
    thumb = _v(0.0, 0.0, -s)                           # the thumb's side in the hand's frame (mkmmd.core.moves.aimed)
    Rr = GG.turned(hold["axis"], front, thumb, _v(1.0, 0.0, 0.0))
    hole_h = _v(hole[0], hole[1], s * hole[2])         # the fist's hole in the hand's frame (mkmmd.core.moves._off)
    return (shape, hole, G, hold["axis"]), (Rr, hole_h - Rr @ G)


# ---------------------------------------------------------------------------------------------------------- the moves
def _check_keys(e, name):
    for k in KEYS:
        if k in e and k not in MOVES[name]["keys"]:
            users = [mv for mv, v in MOVES.items() if k in v["keys"]]
            raise GG.GagError(f"`{k}` is for {', '.join(users)}")


def plan(e, m, t0, t1, beats, downbeats, hand, mic, marks, props, clear, bites, held):
    """The keys of the mosh entry `e` (its move `m`: MOVES) from clip second t0 to t1 played with the hand `hand` (the
    member's hands at the mic in `mic`, {side: bool}), as mkmmd.core.gags.plan gives a gag's: {"plays": {side: [(t,
    place, offset)]}, "places": {side: {name: resolved}}, "perform": {"rise", "crouch": [[t, m]], "air": [span]},
    "curves": {morph: [[t, v]]}, "mouth": [(t0, t1)], "values": {}} and "body": {lean | head_nod: [[t, deg], ...]} (its
    own curves, from 0 back to 0). `props` {name: {"use", "bounds", "scale"}}: the cards of the props it may hold;
    `held` {prop: {...}}: each held prop's grip and attach, set by the first move that holds it."""
    name = e["name"]
    other = "R" if hand == "L" else "L"
    out = {"plays": {}, "places": {"L": {}, "R": {}}, "perform": {}, "curves": {}, "mouth": [], "values": {},
           "body": {}}
    _check_keys(e, name)
    plays = out["plays"]

    def need(*sides):
        for sd in sides:
            if mic[sd]:
                raise GG.GagError(f"needs the {sd} hand, and it holds the mic")

    def prop_of(p, what="prop"):
        if not isinstance(p, str) or p not in props:
            raise GG.GagError(f"{what} = {p!r}: no such prop ([[prop]] names: {', '.join(sorted(props)) or 'none'})")
        return p

    def wield(prop, side):
        h = held.setdefault(prop, {})
        if h.get("side", side) != side or h.get("bone") == "chest":
            raise GG.GagError(f"prop {prop!r} is held {'in the ' + h['side'] + ' hand' if 'side' in h else 'to the chest'}"
                              f" by another move")
        return h

    if name == "pound":
        need(hand, other)
        if "prop" not in e:
            raise GG.GagError("needs `prop`, the [[prop]] mallet it swings (use.grip `hold`, use.anchor `face`)")
        prop = prop_of(e["prop"])
        target = e.get("target")
        if target is not None and not (isinstance(target, (list, tuple)) and len(target) == 3
                                       and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in target)):
            raise GG.GagError("target is [x, y, z], where the mallet's face meets (the member's own frame)")
        hs = GG._times(e, t0, t1, beats)
        h = wield(prop, hand)
        made = h.setdefault("pounds", [])                # a mallet pounding other targets: places of their own
        key = None if target is None else tuple(round(float(v), 4) for v in target)
        if key not in made:
            made.append(key)
        tag = "" if made.index(key) == 0 else f"#{made.index(key)}"
        us, places, st = pound_places(prop, props[prop], hand, marks, None if target is None else list(target), h, tag)
        h.update(side=hand, bone="wrist")
        for sd in ("L", "R"):
            out["places"][sd].update(places[sd])
        out["perform"], out["body"]["lean"], seq = _pound_clock(t0, t1, hs, us, prop, tag, st["lean"], st["brace"])
        plays[hand] = plays[other] = seq
    elif name == "headbang":
        depth = e.get("depth", 1.0)
        if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not 0.0 < float(depth) <= BANG["most"]:
            raise GG.GagError(f"depth is a share of a full headbang, more than 0 and at most {BANG['most']:g}")
        hands = e.get("hands", "free")
        if hands not in ("free", "hips"):
            raise GG.GagError(f"hands = {hands!r}: \"free\" (they hang and swing with her) or \"hips\"")
        hs = GG._times(e, t0, t1, beats)
        if hs[0] - t0 < 0.1:
            raise GG.GagError(f"its first nod comes {hs[0] - t0:.2f} s after it starts: start it before the hit (the "
                              f"head comes up first)")
        k = float(depth)
        lean, nod = [[round(t0, 4), 0.0]], [[round(t0, 4), 0.0]]
        if hs[0] - t0 > 0.2:                              # the head comes up before the first nod: its wind-up
            u = max(t0 + 0.08, hs[0] - 0.5 * min(hs[0] - t0, (hs[1] - hs[0]) if len(hs) > 1 else 0.6))
            lean.append([round(u, 4), round(BANG["lean_up"] * k, 3)])
            nod.append([round(u, 4), round(BANG["nod_up"] * k, 3)])
        for hit, gap in _gaps(hs, t1):
            lean.append([round(hit, 4), round(BANG["lean"] * k, 3)])
            nod.append([round(hit, 4), round(BANG["nod"] * k, 3)])
            u = hit + BANG["up"] * gap
            if u < t1 - 0.06:
                lean.append([round(u, 4), round(BANG["lean_up"] * k, 3)])
                nod.append([round(u, 4), round(BANG["nod_up"] * k, 3)])
        lean.append([round(t1, 4), 0.0])
        nod.append([round(t1, 4), 0.0])
        out["body"] = {"lean": lean, "head_nod": nod}
        if hands == "hips":
            for sd in ("L", "R"):
                if not mic[sd]:
                    plays[sd] = [(t0, "hip", (0, 0, 0))]
    elif name == "pogo":
        hs = GG._times(e, t0, t1, beats)
        rise, crouch, air, lands = [], [], [], []
        ground = t0                                       # when the feet were last put down (or she stood still)
        for hit, gap in _gaps(hs, t1):
            T = min(POGO["flight"], gap - POGO["ground"])
            if T < POGO["least"] or hit < ground - 1e-6:
                continue                                  # no time to fly before the next one
            H = min(POGO["height"], GRAVITY * T * T / 8.0)
            push = min(hit - ground, 0.25)
            if push > 0.06:                               # the knees bend and push off into it
                depth = POGO["knees"] * min(1.0, H / 0.1)
                crouch += _curve(lambda u, d=depth: d * math.sin(math.pi * u), hit - push, hit, 6)
            rise += _curve(lambda u, H=H: 4.0 * H * u * (1.0 - u), hit, hit + T, POGO["steps"])
            air.append({"from": round(hit + 0.01, 4), "to": round(hit + T - 0.01, 4), "tuck": POGO["tuck"],
                        "back": 0.03, "rise": round(min(0.09, 0.35 * T), 3), "extend": round(min(0.1, 0.35 * T), 3),
                        "point": 20.0})
            lands.append((hit, hit + T))
            ground = hit + T
        if not lands:
            raise GG.GagError(f"its hits come too close for a jump (at least {POGO['least'] + POGO['ground']:g} s apart)")
        end = lands[-1][1]
        if t1 - end > 0.08:                               # the last landing: the knees give and come back
            back = min(0.25, t1 - end)
            crouch += _curve(lambda u: POGO["knees"] * math.sin(math.pi * u), end, end + back, 6)
        out["perform"] = {"rise": _dedupe(rise), "crouch": _dedupe(crouch), "air": air}
        for sd in ("L", "R"):
            if mic[sd]:
                continue
            seq = [(t0, "pump_up", (0, 0, 0))] if lands[0][0] - t0 > 0.05 else []
            for a, b in lands:                            # up as she takes off, dipping as she lands
                seq += [(a, "pump_up", (0, 0, 0)), (b, "pump_up", (0.0, 0.0, -POGO["dip"]))]
            plays[sd] = GG._increasing(seq)
    elif name == "wotagei":
        need("L", "R")
        ps = e.get("props")
        if not isinstance(ps, dict) or set(ps) != {"L", "R"}:
            raise GG.GagError("needs `props = {L = \"<prop>\", R = \"<prop>\"}`: the penlight in each fist")
        if ps["L"] == ps["R"]:
            raise GG.GagError(f"props: one prop ({ps['L']!r}) cannot be in both fists")
        routine = e.get("routine", "arcs")
        if routine not in ROUTINES:
            raise GG.GagError(f"routine = {routine!r}: one of {', '.join(ROUTINES)}")
        hs = _eighths(e, t0, t1, beats) if routine == "punch" else GG._times(e, t0, t1, beats)
        MV = _mv()
        for sd in ("L", "R"):
            prop = prop_of(ps[sd], f"props.{sd}")
            hold = GG.hold_of(prop, props[prop])
            h = wield(prop, sd)
            if h.get("rel") is None:                      # the light up out of the thumb's side, its front (-Y) out
                h["grip"], h["rel"] = _stick(sd, marks, hold, _v(0.0, -1.0, 0.0))
            h.update(side=sd, bone="wrist")
            for pl in ROUTINES[routine][0]:
                r = MV.resolve(pl, sd, marks, clear)
                out["places"][sd][f"{pl}@{prop}"] = r[:3] + (h["grip"][0],) + r[4:]
            plays[sd] = _wota_seq(routine, sd, hs, t0, t1, prop)
    elif name == "whistle":
        need(hand)
        hs = GG._times(e, t0, t1, beats)
        seq = [(t0, "cord", (0, 0, 0))] if hs[0] - t0 > 0.05 else []      # its wind-up, unless it starts on a hit
        for hit, gap in _gaps(hs, t1):
            seq += [(hit, "cord_pull", (0, 0, 0)), (hit + 0.5 * gap, "cord", (0, 0, 0))]
        plays[hand] = GG._increasing(seq)
        if not mic[other]:
            plays[other] = [(t0, "hip", (0, 0, 0))]
    elif name == "bass_face":
        style = e.get("style", "scowl")
        if style not in STYLES:
            raise GG.GagError(f"style = {style!r}: one of {', '.join(STYLES)}")
        hs = GG._times(e, t0, t1, beats)
        for morph, v in STYLES[style]:
            out["curves"][morph] = _accents(hs, t1, v)
        nod = []
        for hit, gap in _gaps(hs, t1):
            back = min(ACCENT["nod_back"], gap)
            nod += [[round(hit - 0.04, 4), 0.0], [round(hit + min(ACCENT["nod_at"], 0.5 * back), 4), ACCENT["nod"]],
                    [round(hit + back, 4), 0.0]]
        out["body"]["head_nod"] = _dedupe([[round(t0, 4), 0.0]] + nod + [[round(t1, 4), 0.0]])
        if any(mo in GG.MOUTH_MORPHS for mo, _ in STYLES[style]):
            out["mouth"].append((t0, t1))
    else:
        raise GG.GagError(f"no mosh move {name!r}")
    return out


def _dedupe(keys, gap=1e-3):
    """[[t, v], ...] in time order, keys closer than `gap` to the one before merged (the later one kept)."""
    out = []
    for t, v in sorted(keys, key=lambda k: k[0]):
        if out and t - out[-1][0] < gap:
            out[-1] = [t, v]
        else:
            out.append([t, v])
    return out


def _accents(hs, t1, v):
    """[[t, value], ...] of a face on each hit: up to `v` over ACCENT["attack"] s before it, held, and back down over
    ACCENT["release"] s, or only down to ACCENT["low"] of it when the next hit comes before that."""
    a, hold, rel = ACCENT["attack"], ACCENT["hold"], ACCENT["release"]
    keys = []
    for i, (hit, gap) in enumerate(_gaps(hs, t1)):
        if not keys or keys[-1][1] == 0.0:
            keys.append([round(hit - a, 4), 0.0])
        keys.append([round(hit + 0.02, 4), round(v, 3)])
        last = i == len(hs) - 1
        if gap >= hold + rel + a or last:
            keys += [[round(hit + min(hold, 0.5 * gap), 4), round(v, 3)],
                     [round(hit + min(hold + rel, gap if not last else hold + rel), 4), 0.0]]
        else:
            keys.append([round(hit + 0.5 * gap, 4), round(ACCENT["low"] * v, 3)])
    return _dedupe(keys)


def _pound_clock(t0, t1, hs, us, prop, tag, lean_hit, brace):
    """(perform keys {"crouch"}, lean keys, the hands' plays) of a pound: from the top of its wind-up on t0 down on each
    hit (an accelerating swing), the body leaning `lean_hit` degrees into the blow and the hips dropping `brace` metres
    (`strike`), held on the target while the knees give, then up again as far as the time to the next blow allows
    (POUND["full"] all the way, else less, at least POUND["least"]), waiting at the top until its down-swing; the lean
    and the brace follow the share of the way up. The places are `pound_places`' (us: their shares of the way up)."""
    P = POUND
    n = len(us) - 1

    def at(u):
        """The step of `us` nearest the share u of the way up (rounded up) and its place."""
        k = min(math.ceil(u * n - 1e-6), n)
        return k, f"pound{k}@{prop}{tag}"

    def way(gap):
        return max(P["least"], min(1.0, gap / P["full"]))

    def bent(u):
        """(lean, hips' drop) at the share u of the way up."""
        return round(lean_hit + (P["lean_top"] - lean_hit) * u, 3), round(brace * (1.0 - u), 4)

    first = hs[0] - t0
    if first < 0.15:
        raise GG.GagError(f"its first blow comes {first:.2f} s after it starts: start it before the hit (its wind-up)")
    k_top, top = at(1.0)                                # it starts at the top: its wind-up
    seq = [(t0, top, (0, 0, 0))]
    lean, crouch = [[round(t0, 4), 0.0]], [[round(t0, 4), 0.0]]
    lean.append([round(min(t0 + 0.12, 0.5 * (t0 + hs[0])), 4), bent(1.0)[0]])
    prev_end = t0                                       # the hands are at the top from its start
    for i, hit in enumerate(hs):
        u = us[k_top]
        swing = P["swing"] * math.sqrt(u)
        start = max(hit - swing, prev_end + 0.02)
        if start > prev_end + 0.05:
            seq.append((start, top, (0, 0, 0)))         # waits at the top
        lean.append([round(start, 4), bent(u)[0]])
        crouch.append([round(start, 4), bent(u)[1]])
        last = start
        for k in range(k_top - 1, 0, -1):               # down, faster and faster (keys too close in a short swing left
            tk = start + (hit - start) * math.sqrt(1.0 - us[k] / u)   # out, the blow's always kept)
            if tk - last >= 0.02 and hit - tk >= 0.02:
                seq.append((tk, f"pound{k}@{prop}{tag}", (0, 0, 0)))
                last = tk
        seq.append((hit, f"pound0@{prop}{tag}", (0, 0, 0)))
        lean.append([round(hit, 4), round(lean_hit, 3)])
        crouch.append([round(hit, 4), round(brace, 4)])
        nxt = hs[i + 1] if i + 1 < len(hs) else None
        hold_at = hit + P["hold"]
        k_top, top = at(way(nxt - hold_at)) if nxt is not None else (0, None)
        swing = P["swing"] * math.sqrt(us[k_top])
        rise = min(P["raise"], nxt - swing - hold_at) if nxt is not None else None
        if nxt is not None and rise < 0.08:
            raise GG.GagError(f"its hits at {hit:g} and {nxt:g} s come too close for a blow each (at least "
                              f"{P['hold'] + 0.08 + P['swing'] * math.sqrt(us[1]):.2f} s apart)")
        if nxt is None and t1 - hit < P["hold"] + 0.03:
            break                                       # it ends on the blow: the face rests on the target to its end
        if nxt is None or rise > P["hold"] + 0.05:      # the knees give: the hands come up as much, the face stays down
            crouch.append([round(hold_at, 4), round(brace + P["give"], 4)])
            seq.append((hold_at, f"pound_give@{prop}{tag}", (0, 0, 0)))
        else:
            seq.append((hold_at, f"pound0@{prop}{tag}", (0, 0, 0)))
        lean.append([round(hold_at, 4), round(lean_hit, 3)])
        if nxt is None:
            if t1 - hit > P["give_back"] + 0.05:
                crouch.append([round(hit + P["give_back"], 4), round(brace, 4)])
            break
        for k in range(1, k_top + 1):                  # back up
            seq.append((hold_at + rise * us[k] / us[k_top], f"pound{k}@{prop}{tag}", (0, 0, 0)))
        prev_end = hold_at + rise
        lean.append([round(prev_end, 4), bent(us[k_top])[0]])
        crouch.append([round(prev_end, 4), bent(us[k_top])[1]])
    lean.append([round(t1, 4), 0.0])
    crouch.append([round(t1, 4), 0.0])
    return {"crouch": _dedupe(crouch)}, _dedupe(lean), GG._increasing(seq)


def _wota_seq(routine, side, hs, t0, t1, prop):
    """The plays [(t, place, offset)] of the hand `side` in a wotagei routine on the hits `hs` (module docstring)."""
    s = 1.0 if side == "L" else -1.0
    q = lambda pl: f"{pl}@{prop}"                       # noqa: E731
    seq = []
    if routine == "arcs":                               # a side a beat, the top halfway: both lights to the left first
        own = side == "L"
        seq.append((t0, q("wota_side" if own else "wota_over"), (0, 0, 0)))
        for i, (hit, gap) in enumerate(_gaps(hs, t1)):
            left = i % 2 == 0
            seq.append((hit, q("wota_side" if left == own else "wota_over"), (0, 0, 0)))
            seq.append((hit + 0.5 * gap, q("wota_top"), (0, 0, 0)))
    elif routine == "eights":                           # half an eight a beat, through its middle on the beat
        seq.append((t0, q("wota_eight"), (0, 0, 0)))
        for i, (hit, gap) in enumerate(_gaps(hs, t1)):
            for j in range(4):
                ph = math.pi * (i + 0.25 * j)
                off = (round(s * EIGHT[0] * math.sin(ph), 4), 0.0, round(EIGHT[1] * math.sin(2.0 * ph), 4))
                seq.append((hit + 0.25 * j * gap, q("wota_eight"), off))
    elif routine == "punch":                            # in turns, the left first
        first = side == "L"
        seq.append((t0, q("wota_pull"), (0, 0, 0)))
        for i, hit in enumerate(hs):
            seq.append((hit, q("wota_punch" if (i % 2 == 0) == first else "wota_pull"), (0, 0, 0)))
    else:                                               # a turn every two beats, the right half a turn behind
        ring = ("wota_ring_top", "wota_ring_out", "wota_ring_low", "wota_ring_in")
        k0 = 0 if side == "L" else 2
        seq.append((t0, q(ring[k0]), (0, 0, 0)))
        for i, (hit, gap) in enumerate(_gaps(hs, t1)):
            for j in range(2):
                seq.append((hit + 0.5 * j * gap, q(ring[(k0 + 2 * i + j) % 4]), (0, 0, 0)))
    return GG._increasing(seq)


def rows(e, mic):
    """[(side, places, mirrored)] of the hands a mosh entry plays (the hand_room / hand_mirror / wrist_bend checks hold
    its hands at these places; `mic` {side: bool} the hands that stay at the mic): the places as the moves name them."""
    name = e.get("name")
    hand = e.get("hand", "R")
    other = "R" if hand == "L" else "L"
    free = [sd for sd in ("L", "R") if not mic.get(sd)]
    if name == "pound":
        return [(hand, ["pound"], False), (other, ["pound"], False)]
    if name == "headbang":           # each hand on its own hip, kept off the body as it is there: not a mirror image
        return [(sd, ["hip"], False) for sd in free] if e.get("hands") == "hips" else []
    if name == "pogo":
        return [(sd, ["pump_up"], True) for sd in free]
    if name == "wotagei":
        places, mirrored = ROUTINES.get(e.get("routine", "arcs"), ROUTINES["arcs"])
        return [(sd, list(places), mirrored) for sd in ("L", "R")]
    if name == "whistle":
        return [(hand, ["cord", "cord_pull"], False)] + ([(other, ["hip"], False)] if not mic.get(other) else [])
    return []
