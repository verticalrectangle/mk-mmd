"""Moves, bpy-free (docs/design.md: Moves): `[[move.<cast>]]` entries place named moves of a small library on the clock,
and `compile` turns them into what the pose and perform stages already read: hand keys (`pose.<cast>.hands.L/R.keys`:
places in the character's frame, directions, finger shapes), upper-body lean and tilt keys, head twitches and keyed
expressions (`perform.<cast>`). A move is keys like any other, so a project can still key anything by hand beside them.

Hand places are built from the member's landmarks in its own frame (x its left, y behind it, z up, metres from its root;
the build reads them from the rest pose: `arm.L` / `arm.R` the arm joints, `chest`, `mouth`, `eye`, `upper` and `fore`
the arm's two lengths, `reach` from the arm joint to the wrist, `hand` from the wrist to the middle fingertip), so one
library fits any body: a place near the body is an offset from a landmark, one with the arm out (OUT) a direction from
the arm joint and a share of `reach`, so it is never out of reach. Places by the face and the chest are on the hand's own
side, clear of a mic held at the mouth and of its cord.

Every place is made natural before it is keyed (`resolve`):
- the wrist: the elbow an arm IK gives (the pose stage's default pole, mkmmd.core.armreach) fixes the forearm, and a hand
  that would bend more than WRIST degrees from it is turned toward it; a mic place is aimed instead (AIM): the hand
  continues the forearm as far as a grip on a mic pointing that way allows; two hands that meet in one shape (SHAPED, the
  two-hand heart) keep the orientation written for them, sized to the arm so it suits any body;
- the skin: given the build's distance to the member's own body, clothes and hair (`clear`), the hand comes in from
  outside and stops where it first comes within its margin: an arm held out (OUT) swings in round the shoulder, so it
  stays in reach (a hanging hand rises off a wide skirt), a hand by the body comes in straight (TOUCH, else straight out
  from the body's axis); a touching place (a pat on the chest, hands over the ears) ends on the surface, any other one
  where it was asked unless that is too close to, or inside, the body.
A hand that has played its move goes back to its rest place (`rest`: hanging by the hip; `mic`: holding a mic at the
mouth) unless its next move starts within HOLD seconds. A hand that rests at the mic keeps its grip in every move: a
two-hand move leaves it at the mic (except one that raises it, `hands_up`), and one that needs both hands free
(`heart_push`) is refused."""
import math

import numpy as np

from . import armreach as AR


class MoveError(ValueError):
    """A bad `[[move.<cast>]]`."""


HOLD = 0.35                      # seconds: a hand whose next move starts sooner goes straight on to it
EASE = 0.12                      # seconds a move takes to get into its first place, and back out of its last
WRIST = 60.0                     # degrees: the most a hand bends away from its forearm (as predicted: the IK's lands within ~10)
FAR = 0.3                        # m: how far out a hand by the body starts its approach to it
STEP = 0.01                      # m: the approach's step (the contact is then found to 0.003 mm by halving)
SWING = math.radians(70.0)       # how far up and away an arm held out starts its swing in to its place
SWING_STEP = math.radians(2.0)   # ... and its step
MARGIN = 0.012                   # m: the room a place that does not touch keeps from the body, clothes and hair
TOUCH_MARGIN = 0.003             # m: ... and a touching one
SHAPES = {                       # finger shapes (mkmmd.core.fingers tables) the moves use
    "relaxed": "relaxed", "flat": "flat", "fist": "fist", "point": "point", "curled": "curled",
    "peace": {"index": [0, 0, 0], "middle": [0, 0, 0], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": [30, 40, 40], "spread": 14},
    "paw": {"index": [55, 70, 45], "middle": [55, 70, 45], "ring": [55, 70, 45], "little": [55, 70, 45],
            "thumb": [20, 30, 20]},
    "heart": {"index": [55, 45, 15], "middle": [80, 95, 65], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": [5, 10, 20]},
    "sparkle": {"thumb": [0, 5, 5], "spread": 24},
    "heart2": {"index": [15, 75, 45], "middle": [20, 85, 50], "ring": [30, 90, 55], "little": [40, 95, 60],
               "thumb": [45, 20, 10], "spread": 5},             # half of a two-hand heart: the tips curl down to meet
}
BOX = {  # shape: (how far the hand reaches along its direction, a share of `hand`; half its thickness, m)
    "flat": (1.0, 0.012), "relaxed": (0.95, 0.014), "sparkle": (1.0, 0.012), "peace": (1.0, 0.016),
    "point": (0.95, 0.018), "curled": (0.75, 0.022), "paw": (0.7, 0.024), "heart": (0.65, 0.024), "fist": (0.6, 0.025),
    "heart2": (0.7, 0.024),
}
WIDTH = 0.21                     # half the hand's width, a share of `hand`
MARKS = ("arm.L", "arm.R", "chest", "mouth", "eye", "reach", "upper", "fore", "hand")
OUT = {  # place: (direction from the arm joint for the left hand, x mirrored for the right; share of the reach)
    "rest": ((0.13, 0.03, -1.0), 0.97),
    "mic_lens": ((-0.05, -1.0, 0.05), 0.9),
    "mic_up": ((0.14, -0.14, 1.0), 0.9),
    "bounce": ((0.26, -0.75, -0.61), 0.85),
    "point": ((-0.14, -1.0, 0.05), 0.95),
    "up": ((0.17, -0.09, 1.0), 0.93),
    "mic_across": ((-0.6, -0.8, 0.0), 0.85),
}
AIM = {  # a mic place: where the mic points (its grille), for the left hand, x mirrored for the right. A fist holds a mic
    # across its palm, so a straight wrist keeps the mic square to the forearm: the aims lean from upright, no more
    "mic_lens": (0.0, -0.4, 0.92),                     # held out, leaning to the lens
    "mic_up": (0.0, -0.9, 0.44),                       # raised, the grille forward over the head
    "mic_across": (-0.25, -0.3, 0.92),                 # held out to the other side, upright for a partner there
}
TOUCH = {  # a place on the body: the way the hand comes in to it, for the left hand, x mirrored for the right
    "chest": (0.0, -1.0, 0.0),
    "ears": (1.0, 0.0, 0.0),
}
FIXED = ("mic",)                 # places keyed as written: the mic's grip at the mouth (the mic is fitted to it)
SHAPED = ("heart_push",)         # two hands that meet in one shape: their orientation is kept as written


def _v(*a):
    return np.array(a, float)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def place(name, side, marks):
    """(point, dir, palm, shape) of a named hand place for the hand `side` ("L" / "R") as written, in the character's
    frame: `point` where the wrist goes, `dir` the way the fingers point, `palm` the way the palm faces. `resolve` makes
    it natural."""
    s = 1.0 if side == "L" else -1.0
    arm, chest, mouth, eye = (np.asarray(marks[k], float) for k in (f"arm.{side}", "chest", "mouth", "eye"))

    def out(name):
        d = _unit(OUT[name][0])
        return arm + OUT[name][1] * marks["reach"] * np.array([s * d[0], d[1], d[2]])

    P = {
        "rest": (out("rest"), _v(0, 0, -1), _v(-s, 0, 0), "relaxed"),
        "mic": (mouth + _v(s * 0.082, -0.082, -0.078), _v(-s, -0.3, 0), _v(-s * 0.3, 1, 0), "curled"),
        "mic_lens": (out("mic_lens"), _v(0, -1, 0), _v(-s, 0, 0), "curled"),
        "mic_up": (out("mic_up"), _v(0, 0, 1), _v(-s, 0, 0), "curled"),
        "mic_across": (out("mic_across"), _v(-s, -1, 0), _v(0, 0, 1), "curled"),
        "chest": (chest + _v(s * 0.08, 0.02, 0.0), _v(-s * 0.4, 0, 0.9), _v(0, 1, 0), "flat"),
        "bounce": (out("bounce"), _v(0, -1, 0), _v(0, 0, -1), "relaxed"),
        "point": (out("point"), _v(-s * 0.15, -1, 0.05), _v(-s, 0, 0), "point"),
        "heart": (mouth + _v(s * 0.11, -0.13, -0.09), _v(-s * 0.15, -0.25, 1), _v(-s, 0, 0), "heart"),
        "peace": (eye + _v(s * 0.10, -0.12, -0.12), _v(-s * 0.2, 0, 1), _v(0, -1, 0), "peace"),
        "paw": (mouth + _v(s * 0.09, -0.12, -0.06), _v(0, -0.4, -0.9), _v(0, 1, 0), "paw"),
        "sparkle": (eye + _v(s * 0.14, -0.12, -0.12), _v(s * 0.2, -0.2, 1), _v(0, -1, 0), "sparkle"),
        "up": (out("up"), _v(s * 0.1, 0, 1), _v(0, -1, 0), "flat"),
        "drip_in": (chest + _v(s * 0.12, -0.26, 0.02), _v(-s * 0.6, -0.2, 0.7), _v(0, 1, 0), "relaxed"),
        "drip_out": (chest + _v(s * 0.12, -0.28, 0.04), _v(-s * 0.6, -0.2, 0.7), _v(0, -1, 0), "relaxed"),
        "ears": (eye + _v(s * 0.06, 0.03, -0.07), _v(-s * 0.15, 0, 1), _v(-s, 0, 0), "flat"),
        "heart_push": (chest + marks["reach"] * _v(s * 0.21, -0.69, 0.21), _v(-s * 0.8, 0, 0.6), _v(-s * 0.52, -0.5, -0.69),
                       "heart2"),
        "bunny": (eye + _v(s * 0.09, 0.01, 0.17), _v(s * 0.1, -0.2, 1), _v(0, -1, 0), "paw"),
    }
    if name not in P:
        raise MoveError(f"no hand place {name!r} (have {', '.join(sorted(P))})")
    return P[name]


def forearm(side, marks, wrist):
    """The forearm's direction (elbow -> wrist, unit) an arm IK with the pose stage's default pole gives the hand `side`
    for its wrist at `wrist`."""
    s = 1.0 if side == "L" else -1.0
    S = np.asarray(marks[f"arm.{side}"], float)
    P = S + _v(s * AR.POLE[0], AR.POLE[1], -AR.POLE[2])
    W = np.asarray(wrist, float)
    return _unit(W - AR.pole_elbow(S, W, marks["upper"], marks["fore"], P))


def _turn(v, axis, ang):
    """v turned by `ang` radians about the unit `axis` (right hand)."""
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * float(axis @ v) * (1.0 - math.cos(ang))


def bend(d, f):
    """Degrees between a hand's direction and its forearm's."""
    return float(np.degrees(np.arccos(np.clip(_unit(d) @ _unit(f), -1.0, 1.0))))


def natural(d, p, f, limit=WRIST):
    """(dir, palm): the hand (pointing along `d`, the palm toward `p`) turned toward its forearm `f` as a whole until it
    bends at most `limit` degrees from it."""
    d = _unit(d)
    p = _unit(np.asarray(p, float) - (np.asarray(p, float) @ d) * d)
    th = bend(d, f)
    if th <= limit:
        return d, p
    axis = np.cross(d, _unit(f))
    if np.linalg.norm(axis) < 1e-9:                     # straight back along the forearm: turn it about the palm's side
        axis = np.cross(d, p)
    axis = _unit(axis)
    ang = math.radians(th - limit)
    return _unit(_turn(d, axis, ang)), _unit(_turn(p, axis, ang))


def aimed(side, f, aim):
    """(dir, palm) of a fist round a mic pointing along `aim` (the grille's way) whose hand continues the forearm `f` as
    far as the grip allows: the mic crosses the palm on the thumb side (as at the mic place)."""
    m = _unit(aim)
    d = np.asarray(f, float) - (np.asarray(f, float) @ m) * m
    if np.linalg.norm(d) < 0.15:                         # the mic along the forearm: the hand points the way it can
        d = np.cross(m, _v(1, 0, 0)) if abs(m[0]) < 0.9 else np.cross(m, _v(0, 1, 0))
    d = _unit(d)
    return d, (1.0 if side == "R" else -1.0) * _unit(np.cross(m, d))


def hand_box(d, p, length, shape):
    """(18, 3) points of the hand relative to its wrist: a box along `d` as far as the shape reaches, across the palm and
    through it (along `p`)."""
    reach, half = BOX.get(shape if isinstance(shape, str) else "", (0.75, 0.022))
    d, p = _unit(d), _unit(p)
    r = _unit(np.cross(d, p))
    return np.array([t * reach * length * d + a * WIDTH * length * r + b * half * p
                     for t in (0.25, 0.6, 1.0) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 1.0)])


def sweep(path, hand, clear, margin, far, step):
    """The approach's parameter where the hand stops: the hand (`hand(wrist)`: its points (n, 3) with its wrist there, as
    it turns on the way) comes in along `path(s)` (the wrist at s; s = 0 is the place asked for) from s = `far` and
    stops where `clear(points)` (the smallest distance from points to the body) first falls to `margin`: 0 when it never
    does on the way, `far` when it touches even there. Coming from outside, it never ends inside the body."""
    def touches(s):
        return clear(hand(path(s))) <= margin

    if touches(far):
        return far
    s = far
    while s > 0.0:
        nxt = max(s - step, 0.0)
        if touches(nxt):
            lo, hi = nxt, s
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if touches(mid) else (lo, mid)
            return hi
        s = nxt
    return 0.0


def line(point, approach):
    """A straight approach to `point` from along `approach` (s in metres)."""
    a, point = _unit(approach), np.asarray(point, float)
    return lambda s: point + a * s


def swing(point, shoulder, out):
    """An approach to `point` swinging the whole arm about `shoulder` from the side `out` (s in radians): the hand stays
    as far from the shoulder as at the place, so it stays in reach."""
    S = np.asarray(shoulder, float)
    v = np.asarray(point, float) - S
    axis = np.cross(v, _unit(out))
    if np.linalg.norm(axis) < 1e-9:
        axis = np.cross(v, _v(0, -1, 0))
    axis = _unit(axis)
    return lambda s: S + _turn(v, axis, s)


def resolve(name, side, marks, clear=None):
    """The place `name` for the hand `side` made natural (module docstring): (point, dir, palm, shape, moved, bend):
    `moved` the metres the approach kept it out by, `bend` its wrist's degrees from the forearm."""
    pt, d, p, shape = place(name, side, marks)
    if name in FIXED:
        return pt, _unit(d), _unit(p), shape, 0.0, bend(d, forearm(side, marks, pt))
    s = 1.0 if side == "L" else -1.0

    def orient(at):
        f = forearm(side, marks, at)
        if name in AIM:
            a = AIM[name]
            return aimed(side, f, (s * a[0], a[1], a[2])) + (f,)
        if name in SHAPED:
            dd = _unit(d)
            return dd, _unit(np.asarray(p, float) - (np.asarray(p, float) @ dd) * dd), f
        return natural(d, p, f) + (f,)

    dd, pp, f = orient(pt)
    moved = 0.0
    if clear is not None:
        out = _v(pt[0], pt[1] - float(marks["chest"][1]), 0.0)
        out = out if np.linalg.norm(out) > 0.03 else _v(0, -1, 0)
        if name in TOUCH:
            t = TOUCH[name]
            path, margin, far, step = line(pt, (s * t[0], t[1], t[2])), TOUCH_MARGIN, FAR, STEP
        elif name in OUT:                                # the arm out: it swings up and away round the shoulder
            path, margin, far, step = swing(pt, marks[f"arm.{side}"], out), MARGIN, SWING, SWING_STEP
        else:
            path, margin, far, step = line(pt, out), MARGIN, FAR, STEP
        k = sweep(path, lambda w: w + hand_box(*orient(w)[:2], marks["hand"], shape), clear, margin, far, step)
        if k:
            new = path(k)
            moved = float(np.linalg.norm(new - pt))
            pt = new
            dd, pp, f = orient(pt)
    return pt, dd, pp, shape, moved, bend(dd, f)


# hand: "places" ((place, share of dur), ...) or "beats" (one place: `on` the beat, `off` halfway to the next, offsets in
# the character's frame); both: both hands play it (mic_ok: the mic hand too; two_hands: it needs both free); lean / tilt:
# degrees held over the move; twitch: a head flick on its start; face: (semantic morph, value) held over the move
MOVES = {
    "bounce_hand": {"hand": "beats", "place": "bounce", "on": (0, 0, -0.05), "off": (0, 0, 0)},
    "chest_pat": {"hand": "beats", "place": "chest", "on": (0, 0, 0), "off": (0, -0.035, 0.01), "face": [("blush", 0.6)]},
    "paws": {"hand": "beats", "place": "paw", "on": (0, 0, -0.035), "off": (0, 0, 0), "both": True, "face": [("omega", 0.8)]},
    "point": {"hand": "places", "places": (("point", 0.0),)},
    "heart_wink": {"hand": "places", "places": (("heart", 0.0),), "face": [("wink_r", 1.0), ("mouth_smile", 0.6)]},
    "peace_eye": {"hand": "places", "places": (("peace", 0.0),), "face": [("wink_l", 1.0)]},
    "sparkle": {"hand": "places", "places": (("sparkle", 0.0),), "both": True, "face": [("smile_eyes", 0.7)]},
    "drip_check": {"hand": "places", "places": (("drip_in", 0.0), ("drip_out", 0.5)), "face": [("jito", 0.5)]},
    "hands_up": {"hand": "places", "places": (("up", 0.0),), "both": True, "mic_ok": True},
    "mic_lens": {"hand": "places", "places": (("mic_lens", 0.0),)},
    "mic_up": {"hand": "places", "places": (("mic_up", 0.0),)},
    "mic_across": {"hand": "places", "places": (("mic_across", 0.0),)},
    "ears": {"hand": "places", "places": (("ears", 0.0),), "both": True, "face": [("hau", 1.0)]},
    "heart_push": {"hand": "places", "places": (("heart_push", 0.0),), "both": True, "two_hands": True,
                   "face": [("cheerful", 0.8)]},
    "bunny_paws": {"hand": "places", "places": (("bunny", 0.0),), "both": True, "face": [("wink_r", 1.0)]},
    "into_lens": {"lean": 14.0},
    "lean_back": {"lean": -11.0, "tilt": 7.0},
    "stank": {"twitch": {"bones": ["head"], "deg": -12.0, "axis": [1, 0, 0], "dur": 0.3}, "face": [("hau", 1.0)]},
    "face": {},
}
KEYS = {"name", "t", "dur", "hand", "morph", "value"}
REST_PLACES = ("rest", "mic")


def compile(entries, marks, beats=(), yaw=0.0, clear=None):
    """The keys of one cast member's moves: {"hands": {side: [key]}, "lean": [[t, deg]], "tilt": [[t, deg]], "twitch":
    [...], "expressions": [{morph, keys}], "places": {side: {place: {moved_mm, wrist_deg}}}}. `entries` are its
    `[[move]]` tables {name, t, dur = 1, hand = "R", morph, value}, and `{name = "rest", hand, place}` where that hand
    waits (REST_PLACES; default "rest"); `marks` its landmarks (MARKS); `beats` clip seconds (a beat move pats on each
    inside it); `yaw` the member's (degrees: hand directions are turned into the world); `clear(points)` the smallest
    distance from points (n, 3) in the character's frame to its body, clothes and hair (None: hands are not kept off
    them)."""
    rest = {"L": "rest", "R": "rest"}
    for e in entries:
        if isinstance(e, dict) and e.get("name") == "rest":
            if set(e) - {"name", "hand", "place"} or e.get("hand") not in ("L", "R") or e.get("place") not in REST_PLACES:
                raise MoveError(f"rest = {e!r}: expected {{name = \"rest\", hand = \"L\" | \"R\", place = "
                                f"{' | '.join(REST_PLACES)}}}")
            rest[e["hand"]] = e["place"]
    entries = [e for e in entries if not (isinstance(e, dict) and e.get("name") == "rest")]
    missing = [k for k in MARKS if k not in marks]
    if missing:
        raise MoveError(f"landmarks missing: {missing}")
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    world = lambda d: [round(c * d[0] - s * d[1], 4), round(s * d[0] + c * d[1], 4), round(float(d[2]), 4)]   # noqa: E731
    plays = {"L": [], "R": []}                                       # (t0, t1, [(t, place, offset)])
    body, faces = [], {}
    out = {"hands": {"L": [], "R": []}, "lean": [], "tilt": [], "twitch": [], "expressions": [], "places": {"L": {}, "R": {}}}
    for i, e in enumerate(entries):
        what = f"move {i} ({e.get('name')!r})" if isinstance(e, dict) else f"move {i}"
        if not isinstance(e, dict) or set(e) - KEYS or "name" not in e or "t" not in e:
            raise MoveError(f"{what}: expected {{name, t, dur, hand, morph, value}}")
        if e["name"] not in MOVES:
            raise MoveError(f"{what}: no such move (have {', '.join(sorted(MOVES))})")
        m, t0, dur = MOVES[e["name"]], float(e["t"]), float(e.get("dur", 1.0))
        hand = e.get("hand", "R")
        if dur <= 0 or hand not in ("L", "R"):
            raise MoveError(f"{what}: dur must be positive and hand \"L\" or \"R\"")
        t1 = t0 + dur
        sides = ("L", "R") if m.get("both") else (hand,)
        if m.get("both"):
            held = [sd for sd in sides if rest[sd] == "mic"]
            if held and m.get("two_hands"):
                raise MoveError(f"{what}: needs both hands free, and the {held[0]} hand holds the mic")
            if not m.get("mic_ok"):
                sides = tuple(sd for sd in sides if rest[sd] != "mic")       # the mic hand keeps the mic at the mouth
        if m.get("hand") == "places":
            for side in sides:
                plays[side].append((t0, t1, [(t0 + share * dur, p, (0, 0, 0)) for p, share in m["places"]]))
        elif m.get("hand") == "beats":
            bs = [b for b in beats if t0 - 1e-6 <= b < t1 - 1e-6]
            if not bs:
                raise MoveError(f"{what}: no beat between {t0} and {t1} s (the member's timeline gives the beats)")
            for side in sides:
                seq = []
                for b, nb in zip(bs, bs[1:] + [t1]):
                    seq += [(b, m["place"], m["on"]), (0.5 * (b + nb), m["place"], m["off"])]
                plays[side].append((t0, t1, seq))
        if "lean" in m or "tilt" in m:
            body.append((t0, t1, m.get("lean", 0.0), m.get("tilt", 0.0), what))
        if "twitch" in m:
            out["twitch"].append(dict(m["twitch"], t=t0))
        if e["name"] == "face" and "morph" not in e:
            raise MoveError(f"{what}: a face needs `morph` (a morph's semantic name)")
        for morph, value in m.get("face", []) + ([(str(e["morph"]), float(e.get("value", 1.0)))] if "morph" in e else []):
            faces.setdefault(morph, []).append([t0, t1, value])
    for side in ("L", "R"):
        made = {}

        def at(name, side=side, made=made):
            if name not in made:
                made[name] = resolve(name, side, marks, clear)
            return made[name]

        out["hands"][side] = _hand(sorted(plays[side], key=lambda p: p[0]), side, rest[side], world, at)
        out["places"][side] = {n: {"moved_mm": round(r[4] * 1000.0, 1), "wrist_deg": round(r[5], 1)}
                               for n, r in sorted(made.items())}
    body.sort()
    for (a0, a1, *_, wa), (b0, *_, wb) in zip(body, body[1:]):
        if b0 < a1 - 1e-6:
            raise MoveError(f"{wa} and {wb} both lean the body at {b0} s")
    for t0, t1, lean, tilt, _ in body:
        for key, v in (("lean", lean), ("tilt", tilt)):
            if v:
                out[key] += [[t0, 0.0], [min(t0 + EASE, 0.5 * (t0 + t1)), v], [max(t1 - EASE, 0.5 * (t0 + t1)), v], [t1, 0.0]]
    for morph, spans in faces.items():
        merged = []
        for t0, t1, v in sorted(spans):
            if merged and t0 < merged[-1][1] + 0.12:                 # close enough to hold through
                merged[-1][1], merged[-1][2] = max(merged[-1][1], t1), max(merged[-1][2], v)
            else:
                merged.append([t0, t1, v])
        keys = []
        for t0, t1, v in merged:
            keys += [[t0 - 0.06, 0.0], [t0 + 0.04, v], [max(t1 - 0.06, t0 + 0.05), v], [t1 + 0.06, 0.0]]
        out["expressions"].append({"morph": morph, "keys": [[round(t, 4), round(v, 3)] for t, v in keys]})
    for key in ("lean", "tilt"):
        out[key] = [[round(t, 4), round(v, 3)] for t, v in out[key]]
    return out


def _hand(spans, side, rest, world, at):
    """The keys of one hand: its moves' places in turn (`at(place)` resolves one), home to `rest` between them."""
    grip = SHAPES["curled"] if rest == "mic" else None             # a mic hand never lets go of the mic
    keys = []

    def key(t, name, off=(0, 0, 0)):
        point, d, palm, shape, *_ = at(name)
        keys.append({"t": round(t, 4), "at": [round(float(p + o), 4) for p, o in zip(point, off)], "dir": world(d),
                     "palm": world(palm), "fingers": grip if grip is not None else SHAPES[shape]})

    if not spans and rest != "rest":
        key(0.0, rest)                                               # it waits there from the start
        return keys
    for k, (t0, t1, seq) in enumerate(spans):
        if k and t0 < spans[k - 1][1] - 1e-6:
            raise MoveError(f"the {side} hand plays two moves at once (at {spans[k - 1][0]} and {t0} s)")
        if not keys or t0 - spans[k - 1][1] > HOLD:
            key(max(seq[0][0] - EASE, 0.0), rest)
        for t, p, off in seq:
            key(t, p, off)
        if t1 - seq[-1][0] > 1e-6:                                  # the last place holds to the end of the move
            key(t1, seq[-1][1], seq[-1][2])
        if k + 1 == len(spans) or spans[k + 1][0] - t1 > HOLD:
            key(t1 + EASE, rest)
    return keys
