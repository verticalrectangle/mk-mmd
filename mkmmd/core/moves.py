"""Moves, bpy-free (docs/design.md: Moves): `[[move.<cast>]]` entries place named moves of a small library on the clock,
and `compile` turns them into what the pose and perform stages already read: hand keys (`pose.<cast>.hands.L/R.keys`:
places in the character's frame, directions, finger shapes), upper-body lean and tilt keys, head twitches and keyed
expressions (`perform.<cast>`). A move is keys like any other, so a project can still key anything by hand beside them.

Hand places are built from the member's landmarks in its own frame (x its left, y behind it, z up, metres from its root;
the build reads them from the rest pose: `arm.L` / `arm.R` the arm joints, `chest`, `mouth`, `eye`, and `reach`,
metres from the arm joint to the wrist), so one library fits any body: a place near the body is an offset from a
landmark, one with the arm out (OUT) a direction from the arm joint and a share of `reach`, so it is never out of reach.
A hand that has played its move goes back to its rest place (`rest`: hanging by the hip; `mic`: holding a mic at the
mouth) unless its next move starts within HOLD seconds."""
import math


class MoveError(ValueError):
    """A bad `[[move.<cast>]]`."""


HOLD = 0.35                      # seconds: a hand whose next move starts sooner goes straight on to it
EASE = 0.12                      # seconds a move takes to get into its first place, and back out of its last
SHAPES = {                       # finger shapes (mkmmd.core.fingers tables) the moves use
    "relaxed": "relaxed", "flat": "flat", "fist": "fist", "point": "point", "curled": "curled",
    "peace": {"index": [0, 0, 0], "middle": [0, 0, 0], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": [30, 40, 40], "spread": 14},
    "paw": {"index": [55, 70, 45], "middle": [55, 70, 45], "ring": [55, 70, 45], "little": [55, 70, 45],
            "thumb": [20, 30, 20]},
    "heart": {"index": [55, 45, 15], "middle": [80, 95, 65], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": [5, 10, 20]},
    "sparkle": {"thumb": [0, 5, 5], "spread": 24},
}
MARKS = ("arm.L", "arm.R", "chest", "mouth", "eye", "reach")
OUT = {  # place: (direction from the arm joint for the left hand, x mirrored for the right; share of the reach)
    "rest": ((0.13, 0.03, -1.0), 0.97),
    "mic_lens": ((-0.05, -1.0, 0.05), 0.9),
    "mic_up": ((0.14, -0.14, 1.0), 0.9),
    "bounce": ((0.26, -0.75, -0.61), 0.85),
    "point": ((-0.14, -1.0, 0.05), 0.95),
    "up": ((0.17, -0.09, 1.0), 0.93),
    "heart_push": ((-0.12, -1.0, -0.2), 0.85),
    "mic_across": ((-0.6, -0.8, 0.0), 0.85),
}


def _v(*a):
    return [float(x) for x in a]


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def place(name, side, marks):
    """(point, dir, palm, shape) of a named hand place for the hand `side` ("L" / "R") in the character's frame: `point`
    where the wrist goes, `dir` the way the fingers point, `palm` the way the palm faces."""
    s = 1.0 if side == "L" else -1.0
    arm, chest, mouth, eye = (marks[k] for k in (f"arm.{side}", "chest", "mouth", "eye"))

    def out(name):
        d, share = OUT[name]
        k = share * marks["reach"] / math.sqrt(sum(x * x for x in d))
        return _add(arm, [s * d[0] * k, d[1] * k, d[2] * k])

    P = {
        "rest": (out("rest"), _v(0, 0, -1), _v(-s, 0, 0), "relaxed"),
        "mic": (_add(mouth, _v(s * 0.082, -0.082, -0.078)), _v(-s, -0.3, 0), _v(-s * 0.3, 1, 0), "curled"),
        "mic_lens": (out("mic_lens"), _v(-s, 0, 0), _v(0, 0, 1), "curled"),
        "mic_up": (out("mic_up"), _v(0, -0.2, 1), _v(-s, 0, 0), "curled"),
        "chest": (_add(chest, _v(s * 0.05, -0.11, 0.0)), _v(-s * 0.6, 0, 0.8), _v(0, 1, 0), "flat"),
        "bounce": (out("bounce"), _v(0, -1, 0), _v(0, 0, -1), "relaxed"),
        "point": (out("point"), _v(-s * 0.15, -1, 0.05), _v(-s, 0, 0), "point"),
        "heart": (_add(mouth, _v(s * 0.07, -0.14, -0.11)), _v(-s * 0.3, -0.3, 0.9), _v(0, -1, 0), "heart"),
        "peace": (_add(eye, _v(s * 0.10, -0.12, -0.12)), _v(-s * 0.2, 0, 1), _v(0, -1, 0), "peace"),
        "paw": (_add(mouth, _v(s * 0.09, -0.12, -0.06)), _v(0, -0.4, -0.9), _v(0, 1, 0), "paw"),
        "sparkle": (_add(eye, _v(s * 0.14, -0.12, -0.12)), _v(s * 0.2, -0.2, 1), _v(0, -1, 0), "sparkle"),
        "up": (out("up"), _v(s * 0.1, 0, 1), _v(0, -1, 0), "flat"),
        "drip_in": (_add(chest, _v(s * 0.12, -0.26, 0.02)), _v(-s * 0.6, -0.2, 0.7), _v(0, 1, 0), "relaxed"),
        "drip_out": (_add(chest, _v(s * 0.12, -0.28, 0.04)), _v(-s * 0.6, -0.2, 0.7), _v(0, -1, 0), "relaxed"),
        "ears": (_add(eye, _v(s * 0.10, 0.03, -0.07)), _v(-s * 0.15, 0, 1), _v(-s, 0, 0), "flat"),
        "heart_push": (out("heart_push"), _v(-s * 0.35, -0.2, 0.9), _v(0, -1, 0), "curled"),
        "bunny": (_add(eye, _v(s * 0.09, 0.01, 0.17)), _v(s * 0.1, -0.2, 1), _v(0, -1, 0), "paw"),
        "mic_across": (out("mic_across"), _v(-s * 0.6, -0.8, 0), _v(0, 0, 1), "curled"),
    }
    if name not in P:
        raise MoveError(f"no hand place {name!r} (have {', '.join(sorted(P))})")
    return P[name]


# hand: "places" ((place, share of dur), ...) or "beats" (one place: `on` the beat, `off` halfway to the next, offsets in
# the character's frame); both: both hands play it; lean / tilt: degrees held over the move; twitch: a head flick on its
# start; face: (semantic morph, value) held over the move
MOVES = {
    "bounce_hand": {"hand": "beats", "place": "bounce", "on": (0, 0, -0.05), "off": (0, 0, 0)},
    "chest_pat": {"hand": "beats", "place": "chest", "on": (0, 0, 0), "off": (0, -0.035, 0.01), "face": [("blush", 0.6)]},
    "paws": {"hand": "beats", "place": "paw", "on": (0, 0, -0.035), "off": (0, 0, 0), "both": True, "face": [("omega", 0.8)]},
    "point": {"hand": "places", "places": (("point", 0.0),)},
    "heart_wink": {"hand": "places", "places": (("heart", 0.0),), "face": [("wink_r", 1.0), ("mouth_smile", 0.6)]},
    "peace_eye": {"hand": "places", "places": (("peace", 0.0),), "face": [("wink_l", 1.0)]},
    "sparkle": {"hand": "places", "places": (("sparkle", 0.0),), "both": True, "face": [("smile_eyes", 0.7)]},
    "drip_check": {"hand": "places", "places": (("drip_in", 0.0), ("drip_out", 0.5)), "face": [("jito", 0.5)]},
    "hands_up": {"hand": "places", "places": (("up", 0.0),), "both": True},
    "mic_lens": {"hand": "places", "places": (("mic_lens", 0.0),)},
    "mic_up": {"hand": "places", "places": (("mic_up", 0.0),)},
    "mic_across": {"hand": "places", "places": (("mic_across", 0.0),)},
    "ears": {"hand": "places", "places": (("ears", 0.0),), "both": True, "face": [("hau", 1.0)]},
    "heart_push": {"hand": "places", "places": (("heart_push", 0.0),), "both": True, "face": [("cheerful", 0.8)]},
    "bunny_paws": {"hand": "places", "places": (("bunny", 0.0),), "both": True, "face": [("wink_r", 1.0)]},
    "into_lens": {"lean": 14.0},
    "lean_back": {"lean": -11.0, "tilt": 7.0},
    "stank": {"twitch": {"bones": ["head"], "deg": -12.0, "axis": [1, 0, 0], "dur": 0.3}, "face": [("hau", 1.0)]},
    "face": {},
}
KEYS = {"name", "t", "dur", "hand", "morph", "value"}
REST_PLACES = ("rest", "mic")


def compile(entries, marks, beats=(), yaw=0.0):
    """The keys of one cast member's moves: {"hands": {side: [key]}, "lean": [[t, deg]], "tilt": [[t, deg]], "twitch":
    [...], "expressions": [{morph, keys}]}. `entries` are its `[[move]]` tables {name, t, dur = 1, hand = "R", morph,
    value}, and `{name = "rest", hand, place}` where that hand waits (REST_PLACES; default "rest"); `marks` its landmarks
    (MARKS); `beats` clip seconds (a beat move pats on each inside it); `yaw` the member's (degrees: hand directions are
    turned into the world)."""
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
    out = {"hands": {"L": [], "R": []}, "lean": [], "tilt": [], "twitch": [], "expressions": []}
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
        out["hands"][side] = _hand(sorted(plays[side], key=lambda p: p[0]), side, marks, rest[side], world)
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


def _hand(spans, side, marks, rest, world):
    keys, home = [], place(rest, side, marks)
    if not spans and rest != "rest":
        return [_key(0.0, home, world)]                          # it waits there from the start
    for k, (t0, t1, seq) in enumerate(spans):
        if k and t0 < spans[k - 1][1] - 1e-6:
            raise MoveError(f"the {side} hand plays two moves at once (at {spans[k - 1][0]} and {t0} s)")
        if not keys or t0 - spans[k - 1][1] > HOLD:
            keys.append(_key(max(seq[0][0] - EASE, 0.0), home, world))
        for t, p, off in seq:
            keys.append(_key(t, place(p, side, marks), world, off))
        if t1 - seq[-1][0] > 1e-6:                                  # the last place holds to the end of the move
            keys.append(_key(t1, place(seq[-1][1], side, marks), world, seq[-1][2]))
        if k + 1 == len(spans) or spans[k + 1][0] - t1 > HOLD:
            keys.append(_key(t1 + EASE, home, world))
    return keys


def _key(t, pl, world, off=(0, 0, 0)):
    point, d, palm, shape = pl
    return {"t": round(t, 4), "at": [round(p + o, 4) for p, o in zip(point, off)], "dir": world(d), "palm": world(palm),
            "fingers": SHAPES[shape]}
