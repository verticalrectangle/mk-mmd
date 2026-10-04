"""Fretting: chord shapes as finger tables, and the fingertip targets they give on a prop card's neck (docs/design.md:
Grips, `neck`). numpy only, so the pose stage (Blender's Python), the solver and the tests share it.

A SHAPE is a table of fingers: `finger -> [string, frets]` (string 6 = low E ... 1 = high e, as guitarists number them;
`frets` = how many frets above the POSITION fret the finger presses, 0 = on the position fret). The position fret (`fret =
n` in `[pose.<cast>.hands.L]`) is the fret under the index finger's reach in the shape's own numbering: a power chord at
`fret = 5` has its index on fret 5 and its ring and little fingers on fret 7; the open chords are written for position 1
(`fret = 1`: the nut's neighbour), so open E at `fret = 1` presses frets 1, 2 and 2, A presses fret 2 three times.
Strings no finger presses are left alone (open or muted: the hand does not touch them). Fingers a shape does not use hover
over the board, relaxed.

The card's `use.grip` entry of `type = "neck"` is in the prop's frame (metres): `frame` (along: toward the nut, across: low E
-> high e, normal: out of the board), `thumb` (from the board toward the neck's back), `frets` (the board-surface centre
under each fret wire, 0 = the nut), `strings` (low E first: `nut` and `bridge` points on the string's axis), `section` (the
neck solid under the board), `fret_height` (optional, 1.2 mm). The NECK FRAME N of a position fret is a right-handed frame
on the board: origin on the centre line under that fret's wire, x along the strings toward the nut, z out of the board, y =
z cross x (toward the low E side when z is the card's normal and x is toward the nut)."""
import math

import numpy as np

FINGERS = ("index", "middle", "ring", "little")
STRING_NAMES = {6: "E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "e"}
FRET_HEIGHT = 0.0012          # a fret wire stands this far out of the board (m) when the card does not say
PRESS = 0.30                  # a fingertip presses this share of the gap between two wires behind the wire it plays
PRESS_LO, PRESS_HI = 0.12, 0.70   # ... and may sit anywhere between these shares of the gap (from the playing wire)

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


def shape(chord):
    """{finger: (string, frets above the position fret)} of a chord name (CHORDS) or of a table `{finger: [string,
    frets]}`. Raises FrettingError for an unknown name, finger or string, or two fingers on one string."""
    if isinstance(chord, str):
        if chord not in CHORDS:
            raise FrettingError(f"unknown chord {chord!r} (have {sorted(CHORDS)}); or give a table {{finger = [string, frets]}}")
        return dict(CHORDS[chord])
    if not hasattr(chord, "items") or not chord:
        raise FrettingError(f"a chord is a name or a table {{finger = [string, frets]}}, got {chord!r}")
    out = {}
    for finger, spec in chord.items():
        if finger not in FINGERS:
            raise FrettingError(f"chord finger {finger!r} must be one of {FINGERS}")
        try:
            s, k = int(spec[0]), int(spec[1])
        except (TypeError, ValueError, IndexError):
            raise FrettingError(f"chord {finger} = {spec!r}: expected [string 1-6, frets above the position fret]") from None
        if not (1 <= s <= 6) or k < 0:
            raise FrettingError(f"chord {finger} = {spec!r}: string must be 1..6 and the frets 0 or more")
        out[finger] = (s, k)
    if len({s for s, _ in out.values()}) != len(out):
        raise FrettingError(f"two fingers on one string in {chord!r}")
    return out


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


def string_point(entry, string, s):
    """The point on string `string` (6..1) at the along-coordinate `s` (metres, prop frame), on the line from the nut slot
    to the saddle."""
    st = entry["strings"][6 - int(string)]
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
    """The fingertip targets of `chord` with the index finger's position at `fret`: a list of dicts (one per pressing finger)

        finger, string (6..1), fret (the wire it plays), point (prop frame: on the string, `press` of the gap behind the
        wire, at the height of the string pressed onto the wire), lo / hi (the along-coordinates, prop frame, the pad may sit
        between: PRESS_LO..PRESS_HI of the gap), radius (the string's radius).

    Raises FrettingError when a finger would leave the neck or sit on the nut (fret 0 is the nut: positions start at 1)."""
    n = len(entry["frets"]) - 1
    if int(fret) < 1:
        raise FrettingError("the position fret starts at 1 (fret 0 is the nut)")
    zs = fret_z(entry)
    normal = _entry_frame(entry)[2]
    out = []
    for finger, (string, off) in shape(chord).items():
        k = int(fret) + int(off)
        if k > n:
            raise FrettingError(f"{finger} would play fret {k}, the neck has {n}")
        z_k, z_up = zs[k], zs[k - 1]                      # `up` = the wire toward the nut
        gap = z_up - z_k
        p = string_point(entry, string, z_k + float(press) * gap)
        st = entry["strings"][6 - string]
        r = float(st.get("radius", 0.0006))
        # the pressed string lies on the fret wire's crown: wire height + string radius above the board surface
        p = p - (p - _arr(entry["frets"][k])) @ normal * normal + normal * (float(entry.get("fret_height", FRET_HEIGHT)) + r)
        out.append({"finger": finger, "string": string, "fret": k, "point": p.tolist(), "radius": r,
                    "lo": float(z_k + PRESS_LO * gap), "hi": float(z_k + PRESS_HI * gap)})
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
    (metres; x toward the nut, y toward the low E, z out of the board, origin on the board under the position wire):

        targets   [{finger, string, fret, point [x, y, z], x [lo, hi], radius}]: the pads' contact points on the pressed
                  strings and the range along the neck a pad may sit in
        strings   [[nut xyz, bridge xyz, radius]] x 6 in N (low E first)
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
