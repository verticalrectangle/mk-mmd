"""Fretting: chord shapes as finger tables, single notes along a bass line, and the fingertip targets they give on a prop
card's neck (docs/design.md: Grips, `neck`; Playing a worn guitar). numpy only, so the pose stage (Blender's Python), the
solver, the strum check and the tests share it.

A SHAPE is a table of fingers: `finger -> [string, frets]` (string 1 = the highest ... n = the lowest of the card's n
strings, as players number them: 6 = low E on a guitar, 4 = low E on a bass; `frets` = how many frets above the POSITION
fret the finger presses, 0 = on the position fret). The position fret (`fret = n` in `[pose.<cast>.hands.L]`) is the fret
under the index finger's reach in the shape's own numbering: a power chord at `fret = 5` has its index on fret 5 and its ring
and little fingers on fret 7; the open chords are written for position 1 (`fret = 1`: the nut's neighbour), so open E at
`fret = 1` presses frets 1, 2 and 2, A presses fret 2 three times. Strings no finger presses are left alone (open or muted:
the hand does not touch them). Fingers a shape does not use hover over the board, relaxed. A finger given as
`[string, frets, "hover"]` hovers just over its string at that fret without pressing it (the finger that let go of a note
while an open string rings).

A BASS LINE (`notes`, see `line`) is a list of notes (clip seconds, MIDI pitch) that one fretting hand plays one at a time:
each note gets a string and fret (`place`: the least hand travel over the whole line, one finger per fret, open strings
allowed), and each change of note becomes a shape keyed at the note's time (`hand_keys`).

The card's `use.grip` entry of `type = "neck"` is in the prop's frame (metres): `frame` (along: toward the nut, across: lowest
string -> highest, normal: out of the board), `thumb` (from the board toward the neck's back), `frets` (the board-surface
centre under each fret wire, 0 = the nut), `strings` (the lowest first: `nut` and `bridge` points on the string's axis),
`section` (the neck solid under the board), `fret_height` (optional, 1.2 mm), `tuning` (optional: the MIDI pitch of each open
string, the lowest first). The NECK FRAME N of a position fret is a right-handed frame on the board: origin on the centre line
under that fret's wire, x along the strings toward the nut, z out of the board, y = z cross x (toward the lowest string's
side when z is the card's normal and x is toward the nut)."""
import math

import numpy as np

FINGERS = ("index", "middle", "ring", "little")
STRING_NAMES = {6: "E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "e"}
FRET_HEIGHT = 0.0012          # a fret wire stands this far out of the board (m) when the card does not say
PRESS = 0.30                  # a fingertip presses this share of the gap between two wires behind the wire it plays
PRESS_LO, PRESS_HI = 0.12, 0.70   # ... and may sit anywhere between these shares of the gap (from the playing wire)
HOVER = 0.003                 # a hovering finger's pad stays this far (m) over the top of its string
HOVER_MARK = "hover"          # the third element of a finger that hovers: [string, frets, "hover"]

# shape -> {finger: (string, frets above the position fret)}
CHORDS = {
    "power": {"index": (6, 0), "ring": (5, 2), "little": (4, 2)},                      # root on the low E, fifth, octave
    "power5": {"index": (5, 0), "ring": (4, 2), "little": (3, 2)},                     # root on the A string
    "E": {"index": (3, 0), "middle": (5, 1), "ring": (4, 1)},                          # open E major (position 1)
    "A": {"index": (4, 1), "middle": (3, 1), "ring": (2, 1)},                          # open A major: three fingers on fret 2
    "D": {"index": (3, 1), "middle": (1, 1), "ring": (2, 2)},                          # open D major
    "G": {"middle": (6, 2), "index": (5, 1), "ring": (1, 2)},                          # open G major
    "C": {"ring": (5, 2), "middle": (4, 1), "index": (2, 0)},                          # open C major
    "Em": {"middle": (5, 1), "ring": (4, 1)},                                          # open E minor
    "Am": {"middle": (4, 1), "ring": (3, 1), "index": (2, 0)},                         # open A minor
}


class FrettingError(ValueError):
    pass


def shape(chord, strings=6):
    """{finger: (string, frets above the position fret)} of a chord name (CHORDS) or of a table `{finger: [string, frets]}`
    (or `[string, frets, "hover"]`: see `hovering`) on a neck of `strings` strings. Raises FrettingError for an unknown
    name, finger or string, or two pressing fingers on one string."""
    if isinstance(chord, str):
        if chord not in CHORDS:
            raise FrettingError(f"unknown chord {chord!r} (have {sorted(CHORDS)}); or give a table {{finger = [string, frets]}}")
        out = dict(CHORDS[chord])
        if any(s > strings for s, _ in out.values()):
            raise FrettingError(f"chord {chord!r} is for six strings, the neck has {strings}: give a table {{finger = [string, frets]}}")
        return out
    if not hasattr(chord, "items") or not chord:
        raise FrettingError(f"a chord is a name or a table {{finger = [string, frets]}}, got {chord!r}")
    out = {}
    for finger, spec in chord.items():
        if finger not in FINGERS:
            raise FrettingError(f"chord finger {finger!r} must be one of {FINGERS}")
        try:
            s, k = int(spec[0]), int(spec[1])
            mark = spec[2] if len(spec) > 2 else None
        except (TypeError, ValueError, IndexError, KeyError):
            raise FrettingError(f"chord {finger} = {spec!r}: expected [string 1-{strings}, frets above the position fret]") from None
        if not (1 <= s <= strings) or k < 0 or len(spec) > 3 or mark not in (None, HOVER_MARK):
            raise FrettingError(f"chord {finger} = {spec!r}: string must be 1..{strings}, the frets 0 or more, and a third "
                                f"element only {HOVER_MARK!r}")
        out[finger] = (s, k)
    pressing = [s for f, (s, _) in out.items() if f not in hovering(chord)]
    if len(set(pressing)) != len(pressing):
        raise FrettingError(f"two fingers on one string in {chord!r}")
    return out


def hovering(chord):
    """The fingers of a chord table marked `[string, frets, "hover"]`: they hover just over their string at that fret
    instead of pressing it (a chord name has none)."""
    if not hasattr(chord, "items"):
        return set()
    return {f for f, spec in chord.items() if isinstance(spec, (list, tuple)) and len(spec) > 2 and spec[2] == HOVER_MARK}


def _arr(v):
    return np.asarray(v, float)


def _unit(v):
    v = _arr(v)
    return v / np.linalg.norm(v)


def _entry_frame(entry):
    """(along, across, normal) of a neck entry as unit vectors."""
    f = entry["frame"]
    return _unit(f["along"]), _unit(f["across"]), _unit(f["normal"])


def fret_z(entry):
    """(n + 1,) the position of every fret wire (index 0 = the nut) along the card's `along` axis (metres, prop frame)."""
    along = _entry_frame(entry)[0]
    return np.array([float(_arr(p) @ along) for p in entry["frets"]])


def string_entry(entry, string):
    """The card's string `string` (1 = the highest ... n = the lowest of its n strings, which the card lists lowest first)."""
    n = len(entry["strings"])
    if not 1 <= int(string) <= n:
        raise FrettingError(f"string {string} is not on the neck (1..{n})")
    return entry["strings"][n - int(string)]


def string_point(entry, string, s):
    """The point on string `string` (1 = the highest .. n = the lowest) at the along-coordinate `s` (metres, prop frame), on
    the line from the nut slot to the saddle."""
    st = string_entry(entry, string)
    along = _entry_frame(entry)[0]
    a, b = _arr(st["nut"]), _arr(st["bridge"])
    za, zb = float(a @ along), float(b @ along)
    t = (s - za) / (zb - za)
    return a + t * (b - a)


def neck_frame(entry, fret):
    """(origin (3,), R (3, 3)) of the neck frame N at the position fret `fret` in the prop frame (columns x, y, z: along
    toward the nut, z cross x, out of the board). The origin is the board-surface point under that wire."""
    n = len(entry["frets"]) - 1
    if not 0 <= int(fret) <= n:
        raise FrettingError(f"fret {fret} is not on the neck (0..{n})")
    along, _across, normal = _entry_frame(entry)
    z = normal - (normal @ along) * along
    z = z / np.linalg.norm(z)
    return _arr(entry["frets"][int(fret)]), np.stack([along, np.cross(z, along), z], 1)


def finger_targets(entry, chord, fret, press=PRESS):
    """The fingertip targets of `chord` with the index finger's position at `fret`: a list of dicts (one per finger the shape
    uses)

        finger, string (1..n), fret (the wire it plays), point (prop frame: on the string, `press` of the gap behind the
        wire, at the height of the string pressed onto the wire; a hovering finger's is HOVER over the top of the string where
        it lies, unpressed), lo / hi (the along-coordinates, prop frame, the pad may sit between: PRESS_LO..PRESS_HI of the
        gap), radius (the string's radius), hover (bool).

    Raises FrettingError when a finger would leave the neck or sit on the nut (fret 0 is the nut: positions start at 1)."""
    n = len(entry["frets"]) - 1
    if int(fret) < 1:
        raise FrettingError("the position fret starts at 1 (fret 0 is the nut)")
    zs = fret_z(entry)
    normal = _entry_frame(entry)[2]
    hover = hovering(chord)
    out = []
    for finger, (string, off) in shape(chord, len(entry["strings"])).items():
        k = int(fret) + int(off)
        if k > n:
            raise FrettingError(f"{finger} would play fret {k}, the neck has {n}")
        z_k, z_up = zs[k], zs[k - 1]                      # `up` = the wire toward the nut
        gap = z_up - z_k
        p = string_point(entry, string, z_k + float(press) * gap)
        r = float(string_entry(entry, string).get("radius", 0.0006))
        if finger in hover:                               # over the string as it lies (it rings open under the pad)
            p = p + normal * (r + HOVER)
        else:                                             # the pressed string lies on the fret wire's crown: wire height +
            p = p - (p - _arr(entry["frets"][k])) @ normal * normal + normal * (float(entry.get("fret_height", FRET_HEIGHT)) + r)
        out.append({"finger": finger, "string": string, "fret": k, "point": p.tolist(), "radius": r,
                    "lo": float(z_k + PRESS_LO * gap), "hi": float(z_k + PRESS_HI * gap), "hover": finger in hover})
    return out


def section_at(entry, s):
    """(width, depth) of the neck solid at the along-coordinate s (linear between the nut and the last fret, held beyond):
    the board's width and the centre-line depth from the board surface to the back of the neck."""
    sec = entry["section"]
    zs = fret_z(entry)
    t = float(np.clip((s - zs[0]) / (zs[-1] - zs[0]), 0.0, 1.0))
    w = sec["width"][0] + t * (sec["width"][1] - sec["width"][0])
    d = sec["depth"][0] + t * (sec["depth"][1] - sec["depth"][0])
    return float(w), float(d)


def solver_prop(entry, chord, fret, press=PRESS):
    """The neck problem the grip solver takes (`style = "neck"`), everything in the neck frame N of the position fret
    (metres; x toward the nut, y toward the lowest string, z out of the board, origin on the board under the position wire):

        targets   [{finger, string, fret, point [x, y, z], x [lo, hi], radius}]: the pads' contact points on the pressed
                  strings (a hovering finger's point is over its string) and the range along the neck a pad may sit in
        strings   [[nut xyz, bridge xyz, radius]] per string in N (the lowest first)
        section   {x [x_a, x_b], width [..], depth [..], p}: the neck solid below z = 0 over the span the hand can reach
                  (the superellipse back of the card's `section`; width and depth linear in x)
        frets     x of every fret wire in N (index 0 = the nut); fret_height; position (the fret)

    Returns (prop dict, (origin, R) of N in the prop frame)."""
    origin, R = neck_frame(entry, fret)
    zs = fret_z(entry)
    o_along = float(origin @ R[:, 0])

    def to_n(p):
        return (R.T @ (_arr(p) - origin)).tolist()

    targets = [{"finger": t["finger"], "string": t["string"], "fret": t["fret"], "point": to_n(t["point"]),
                "x": [t["lo"] - o_along, t["hi"] - o_along], "radius": t["radius"]}
               for t in finger_targets(entry, chord, fret, press)]
    xs = zs - o_along
    x_lo = float(xs[min(len(xs) - 1, int(fret) + 5)]) - 0.02              # the hand reaches 5 frets toward the body
    x_hi = float(xs[max(0, int(fret) - 3)]) + 0.03                         # and 3 toward the nut
    (wa, da), (wb, db) = section_at(entry, x_lo + o_along), section_at(entry, x_hi + o_along)
    prop = {"targets": targets,
            "strings": [[to_n(st["nut"]), to_n(st["bridge"]), float(st.get("radius", 0.0006))] for st in entry["strings"]],
            "section": {"x": [x_lo, x_hi], "width": [wa, wb], "depth": [da, db], "p": float(entry["section"].get("p", 2.6))},
            "frets": xs.tolist(), "fret_height": float(entry.get("fret_height", FRET_HEIGHT)), "position": int(fret)}
    return prop, (origin, R)


# =============================================================================================== bass lines: one note at a time
NOTES_KEYS = ("lane", "times", "tuning", "from", "to", "octave", "min_db", "position")
MOVE = 0.12                   # s: a hand moves to its next shape over this long before the note (the hand's `move` overrides)
MOVE_SHARE = 0.75             # ... and never over more than this share of the time since the note before
SHIFT = 0.005                 # m of travel a shift of position costs on top of its length (stay put when it is all the same)
OPEN = 0.004                  # m of travel an open string costs (a fretted note shows the line; an open one saves a shift)
CROSS = 0.0005                # m per string crossed (a tie-breaker)
LOW = 1e-5                    # m per fret (prefer the lower of two equal places)


def notes_of(spec, hits=None):
    """[(t, midi)] of a notes table, sorted by time: `times` = [[t, midi], ...] (clip seconds, MIDI note numbers), or `lane` =
    a note lane of the hits file (`hits`, {lane: [{t, db, midi}]}: `mk timeline hits --notes`), the hits at or above `min_db`;
    both inside `from`..`to` (clip seconds, inclusive), every pitch rounded to a semitone after adding `octave` octaves."""
    if not hasattr(spec, "items"):
        raise FrettingError(f"notes = {spec!r}: a table {{lane = \"bass_notes\" | times = [[t, midi], ...], tuning, from, to, octave}}")
    unknown = sorted(set(spec) - set(NOTES_KEYS))
    if unknown:
        raise FrettingError(f"notes: unknown key {unknown[0]!r} (have {', '.join(NOTES_KEYS)})")
    if (spec.get("lane") is None) == (spec.get("times") is None):
        raise FrettingError("notes: give `lane` (a note lane of the [audio] hits file) or `times` ([[t, midi], ...]), one of them")
    t0, t1 = float(spec.get("from", -math.inf)), float(spec.get("to", math.inf))
    octave = spec.get("octave", 0)
    if isinstance(octave, bool) or not isinstance(octave, int):
        raise FrettingError(f"notes: octave = {octave!r}: a whole number of octaves (-1 plays the line an octave lower)")
    raw = []
    if spec.get("times") is not None:
        for i, item in enumerate(spec["times"]):
            try:
                raw.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError, IndexError, KeyError):
                raise FrettingError(f"notes: times[{i}] = {item!r}: expected [t, midi]") from None
    else:
        lane = spec["lane"]
        if hits is None:
            raise FrettingError(f"notes: lane {lane!r} reads the [audio] hits file (write it with `mk timeline hits --notes`)")
        if lane not in hits or not isinstance(hits[lane], list):
            raise FrettingError(f"notes: no lane {lane!r} in the hits file (have {', '.join(sorted(hits))})")
        min_db = spec.get("min_db")
        for i, h in enumerate(hits[lane]):
            if not isinstance(h, dict) or not isinstance(h.get("t"), (int, float)) or not isinstance(h.get("midi"), (int, float)):
                raise FrettingError(f"notes: lane {lane!r} hit {i} has no pitch ({{t, midi}}): give a note lane "
                                    f"(`mk timeline hits --notes NAME=STEM:LO-HI`)")
            if min_db is not None and (h.get("db") is None or float(h["db"]) < float(min_db)):
                continue
            raw.append((float(h["t"]), float(h["midi"])))
    out = sorted((t, int(round(m + 12 * octave))) for t, m in raw if t0 - 1e-9 <= t <= t1 + 1e-9)
    if not out:
        raise FrettingError(f"notes: no note inside {spec.get('from', '-inf')}..{spec.get('to', 'inf')}")
    return out


def tuning_of(spec, entry):
    """The MIDI pitch of every open string, the lowest first: the notes table's `tuning`, else the card's neck `tuning`."""
    tun = spec.get("tuning") if hasattr(spec, "get") else None
    if tun is None:
        tun = entry.get("tuning")
    n = len(entry["strings"])
    if tun is None:
        raise FrettingError(f"notes: give `tuning` (the MIDI pitch of each of the {n} open strings, the lowest first, e.g. "
                            f"[28, 33, 38, 43] for a bass): the card has none")
    try:
        tun = [int(v) for v in tun]
    except (TypeError, ValueError):
        raise FrettingError(f"notes: tuning = {tun!r}: MIDI note numbers, the lowest string first") from None
    if len(tun) != n:
        raise FrettingError(f"notes: tuning has {len(tun)} strings, the neck {n}")
    return tun


def place(notes, tuning, frets, position=1):
    """Where a fretting hand plays each note of a line: [{t, midi, string (1 = the highest), fret (0 = open), position (the
    fret under the index finger), finger (None on an open string)}]. `tuning` is the open strings' MIDI pitches (the lowest
    first), `frets` the along-coordinate of every wire (`fret_z`, index 0 = the nut). One finger per fret: a note on fret f at
    position p is played by finger f - p (index .. little), so p is f - 3 .. f; an open string leaves the position as it is.
    `position` is the lowest the hand may go (the index finger never below that fret: a line kept up the neck, near the
    body), so a fretted note there is on that fret or higher. The places minimise, over the whole line (dynamic programming),
    the hand's travel along the neck (metres between the position wires) plus SHIFT for every shift, OPEN for every open
    string, CROSS per string crossed and LOW per fret. Raises FrettingError naming the notes that are not on the neck, or
    not playable from that position up."""
    fz = np.asarray(frets, float)
    n_frets, ns = len(fz) - 1, len(tuning)
    if n_frets < 4:
        raise FrettingError("a neck with fewer than four frets cannot be played one finger per fret")
    lo, hi = min(tuning), max(tuning) + n_frets
    bad = [(t, m) for t, m in notes if not lo <= m <= hi]
    if bad:
        listed = ", ".join(f"t {t:.3f} (MIDI {m})" for t, m in bad[:6]) + (" ..." if len(bad) > 6 else "")
        raise FrettingError(f"notes not on the neck (MIDI {lo}..{hi} with tuning {list(tuning)}): {listed}; shift the line with "
                            f"`octave` or leave them out with `from` / `to`")
    pos = int(position)
    if not 1 <= pos <= n_frets:
        raise FrettingError(f"position {position!r}: a fret of the neck, 1..{n_frets}")
    cands = []                                            # per note: [(string, fret, position)]
    for _t, m in notes:
        c = []
        for i, open_pitch in enumerate(tuning):
            f, s = m - int(open_pitch), ns - i
            if f == 0:
                c += [(s, 0, p) for p in range(pos, n_frets + 1)]
            elif pos <= f <= n_frets:
                c += [(s, f, p) for p in range(max(pos, f - 3), f + 1)]
        cands.append(c)
    low = [(t, m) for (t, m), c in zip(notes, cands) if not c]
    if low:
        listed = ", ".join(f"t {t:.3f} (MIDI {m})" for t, m in low[:6]) + (" ..." if len(low) > 6 else "")
        raise FrettingError(f"notes with no string and fret from position {pos} up: {listed}; lower `position`, or shift the "
                            f"line with `octave`")

    def own(c):
        return (OPEN if c[1] == 0 else 0.0) + LOW * c[1]

    def step(a, b):
        return abs(fz[a[2]] - fz[b[2]]) + (SHIFT if a[2] != b[2] else 0.0) + CROSS * abs(a[0] - b[0])

    cost = [np.array([own(c) for c in cands[0]])]
    back = []
    for k in range(1, len(notes)):
        prev, cur = cands[k - 1], cands[k]
        T = np.array([[step(a, b) for b in cur] for a in prev]) + cost[-1][:, None]
        back.append(np.argmin(T, axis=0))
        cost.append(T.min(axis=0) + np.array([own(c) for c in cur]))
    j = int(np.argmin(cost[-1]))
    path = [j]
    for b in reversed(back):
        j = int(b[j])
        path.append(j)
    path.reverse()
    out = []
    for (t, m), c, j in zip(notes, cands, path):
        s, f, p = c[j]
        out.append({"t": float(t), "midi": int(m), "string": s, "fret": f, "position": p,
                    "finger": FINGERS[f - p] if f else None})
    return out


def hand_keys(placed, move=MOVE):
    """The fretting hand's shapes along a placed line (`place`): [{t, fret (the position), chord, move}], the first the shape
    the hand starts in. A fretted note is `{finger: [string, fret - position]}`; on an open string the finger of the note
    before lets go and hovers where it was (`[string, frets, "hover"]`; the index over the open string when that finger is
    not under the hand any more). A note that needs no change makes no key; `move` (the seconds the hand takes to the next
    shape) is cut to MOVE_SHARE of the time since the note before, so the hand leaves a note after it sounded."""
    keys, last, t_prev = [], None, None
    for n in placed:
        p = n["position"]
        if n["fret"] > 0:
            chord = {n["finger"]: [n["string"], n["fret"] - p]}
            last = (n["string"], n["fret"])
        elif last is not None and 0 <= last[1] - p <= 3:
            chord = {FINGERS[last[1] - p]: [last[0], last[1] - p, HOVER_MARK]}
        else:
            chord = {"index": [n["string"], 0, HOVER_MARK]}
        if not keys or (keys[-1]["fret"], keys[-1]["chord"]) != (p, chord):
            mv = float(move) if t_prev is None else min(float(move), MOVE_SHARE * (n["t"] - t_prev))
            keys.append({"t": n["t"], "fret": p, "chord": chord, "move": round(mv, 4)})
        t_prev = n["t"]
    return keys


def line(spec, entry, hits=None, move=MOVE):
    """A bass line on a card's neck entry: (placed notes, hand keys) of a notes table (`notes_of`, `tuning_of`, `place`,
    `hand_keys`); `hits` is the [audio] hits file's lanes when the table names a `lane`."""
    notes = notes_of(spec, hits)
    position = spec.get("position", 1)
    if isinstance(position, bool) or not isinstance(position, int):
        raise FrettingError(f"notes: position = {position!r}: a whole fret number (the lowest the index finger goes)")
    placed = place(notes, tuning_of(spec, entry), fret_z(entry), position)
    return placed, hand_keys(placed, move)
