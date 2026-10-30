"""Strumming and picking: which strokes a pick hand makes and where its pick tip is at every moment (docs/design.md: Perform,
`strum`; Playing a worn guitar). numpy only (runs in Blender's Python, in the tests and in the checks).

A STROKE is one pass of the pick across the strings: it has a strike time `t` (clip seconds: the pick meets the first string),
a direction (`+1` down: across the strings from the lowest to the highest, `-1` up) and an amplitude. The pick tip's place is
`u` along the strum axis (metres from the strum centre, positive = the down-stroke direction) and `h` above the string plane
(negative = pressed into the strings). Both are measured from the REST pose of the grip (the pick tip on the string plane at
the strum centre), so a hand that is not strumming has u = h = 0.

Timeline of one stroke (times from its strike time t; sigma = half the width of the strings, half = half the stroke):

    t - lead         the pick leaves the far side of the strings (u = -d half) and accelerates ...
    t                ... meets the first string (u = -d sigma) at the speed of the sweep ...
    t .. t + attack  ... and sweeps across the strings at that speed (the audible spread of a strum, 60-90 ms)
    t + attack + follow   leaves the last string (u = +d sigma) and eases to the far side (u = +d half)

Strokes that alternate (down, up, down ...) share their turn-round: the follow-through of one is the lead of the next, so a
run of eighth notes is one continuous oscillation. A stroke in the same direction as the one before first has to go back over
the strings, and that return is lifted off them. After a pause longer than `approach + retreat` the hand goes back to rest
and comes in again. `plan` chooses the strokes from the rhythm, `path` evaluates the tip's place over time.

PICKING (`rhythm = "notes"`) plays one string per stroke: the strokes are the notes of a bass line (mkmmd.core.fretting.line,
the fretting hand's `notes`), each centred on its note's string (`c`, the string's place on the strum axis) with a narrow
contact (`w`, the string's radius and a little: the stroke meets the string at u = c - d w and leaves it at c + d w), small
(`PICK` overrides the strum DEFAULTS) and strictly alternating, down first and again down after a pause the hand rests in.

FINGERSTYLE (`technique = "fingers"`, `technique` of the spec or of the card's strum entry) plays a bass line with REST STROKES:
the index and middle fingers alternate (the index first after a pause), each fingertip swinging from hovering over its note's
string on the side toward the higher strings, through the string toward the lower strings (every stroke has dir -1) to rest
on the next lower string (toward the thumb when it is the lowest); `Stroke.t` is when the fingertip crosses the string's axis
(lets it go). The hand's poses come from the grip style `pluck` (mkmmd.solvers.grip_pluck), one STATION per string:
`pluck_keys` says which pose each finger is in and at which station the hand is at every moment (`FINGERS` overrides the
DEFAULTS), `pluck_prop` builds a station's problem from the card."""
import math

import numpy as np

from . import fretting as FR

DEFAULTS = {
    "span": 0.09,            # m: the full sweep of a normal stroke, centred on the strum centre
    "depth": 0.003,          # m: how far the pick tip is pressed into the strings at the strike
    "lift": 0.014,           # m: how high above the strings the pick is at the far ends of the sweep
    "attack": 0.075,         # s: from the first string to the last (the audible spread of a strum)
    "lead": 0.09, "follow": 0.09,        # s: the run-up before the first string and the follow-through after the last
    "approach": 0.30, "retreat": 0.30,   # s: rest -> the first stroke's start, the last stroke's end -> rest
    "return_lift": 0.022,    # m: how high the pick goes over the strings when it comes back for a stroke in the same direction
    "accent_gain": 1.25,     # amplitude and depth of an accented stroke (the first beat of a bar)
    "up_scale": 0.85,        # amplitude and depth of an up stroke relative to a down stroke
    "accent_window": 0.06,   # s: a strike this close to an accent time is accented
    "min_gap": 0.09,         # s: onsets closer than this are one stroke
    "beat_tol": 0.22,        # beats: how close to a beat or an offbeat a strike must be to take its direction from it
    "share": 0.6,            # share of the sideways travel the hand makes by turning about the wrist (see the build)
}
PICK = {                     # picking single strings (rhythm "notes"): what differs from strumming
    "span": 0.026,           # m: a stroke sweeps 13 mm either side of its string
    "depth": 0.002,          # m: the tip goes this far past the string's axis plane to catch it
    "lift": 0.006,           # m: and clears the strings by this at the ends of the sweep (over the neighbours)
    "attack": 0.012,         # s: the pick pushing through the string
    "lead": 0.05, "follow": 0.05,
    "approach": 0.25, "retreat": 0.25,
    "return_lift": 0.008,
    "accent_gain": 1.15,
    "up_scale": 1.0,         # alternate picking: up strokes as strong as down strokes
    "min_gap": 0.05,
    "share": 0.85,           # the stroke is a turn of the hand about the wrist (`wrist_turn`): the arm hardly moves
}
FINGERS = {                  # fingerstyle (technique "fingers", rhythm "notes"): rest strokes, what differs from picking
    "attack": 0.11,          # s: a rest stroke's swing from hovering through its string to its rest on the next lower one,
                             #    at a steady speed (the string is let go at Stroke.t, where the fingertip crosses it)
    "hold": 0.25,            # s: the finger rests there at most this long, then lifts ...
    "recover": 0.12,         # s: ... back to hovering over this long
    "shift": 0.12,           # s: the hand moves to another station (string, thumb rest) over this long, done when the stroke
                             #    that needs it starts
    "pause": 0.8,            # s: a gap this long between two notes starts the alternation again with the index
    "accent_gain": 1.0,      # a rest stroke is as wide on an accent (the string's own amplitude is the strings' business)
    "up_scale": 1.0,
    "min_gap": 0.05,         # s: notes closer than this are one stroke
}
TECHNIQUES = ("pick", "fingers")
POSES = ("hover", "cross", "rest")    # a plucking finger's poses, in the order a rest stroke goes through them
TRAVEL = {"hover": 0.008, "cross": 0.0, "rest": -0.015}   # m: the fingertip's place across from its string in each pose
                             # (+ toward the higher strings) when the station's solve gives none
MIN_SHIFT = 0.04             # s: the shortest move between stations
CONTACT = 0.0005             # m: a picked string's contact half-width is its radius plus this (the pick's half thickness)
TURN_MIN = 0.03              # m per radian: a turn about the wrist that moves the tip less sideways than this is left out
TURN_SIDEWAYS = (0.5, 0.85)  # the share of the turn's tip motion that goes sideways: no turn below the first, all above


class StrumError(ValueError):
    pass


class Stroke:
    __slots__ = ("t", "dir", "amp", "accent", "depth", "attack", "lead", "follow", "c", "w", "string", "finger")

    def __init__(self, t, dir, amp=1.0, accent=False, depth=None, c=None, w=None, string=None, finger=None):
        self.t, self.dir, self.amp, self.accent = float(t), int(dir), float(amp), bool(accent)
        self.depth = None if depth is None else float(depth)    # m: this stroke's own depth (a spec's), else the path's
        self.attack = self.lead = self.follow = None            # filled in by `path`
        self.c = None if c is None else float(c)                # m: a picked string's place on the strum axis (None: a strum,
        self.w = None if w is None else float(w)                # centred on 0) and the half-width of its contact (None: sigma)
        self.string = string                                    # the picked string's number (1 = the highest), for reports
        self.finger = finger                                    # fingerstyle: "index" or "middle" plucks it (None: a pick)

    def __repr__(self):
        what = "" if self.string is None else f", string {self.string}"
        what += "" if self.finger is None else f", {self.finger}"
        return f"Stroke(t={self.t:.3f}, {'down' if self.dir > 0 else 'up'}, amp={self.amp:.2f}{', accent' if self.accent else ''}{what})"


def picking(spec):
    """True for a spec that picks single strings (`rhythm = "notes"`)."""
    return spec.get("rhythm") == "notes"


def technique(spec, entry=None):
    """"fingers" or "pick": the strum spec's own `technique`, else the card strum entry's (the library bass's is "fingers", the
    guitar's "pick"), else "pick". StrumError for anything else."""
    t = spec.get("technique")
    if t is None:
        t = (entry or {}).get("technique") or "pick"
    if t not in TECHNIQUES:
        raise StrumError(f"technique {t!r}: \"fingers\" (rest strokes with the index and middle fingers, a bass line) or "
                         f"\"pick\"")
    return t


def resolved(spec, entry=None):
    """The spec with its `technique` decided (`technique`): what `params` and `plan` read."""
    return dict(spec, technique=technique(spec, entry))


def fingerstyle(spec):
    """True for a (resolved) spec played with the fingers."""
    return spec.get("technique") == "fingers"


def params(spec):
    """The path numbers of a (resolved) spec: DEFAULTS, PICK's on top for a picking spec and FINGERS' for one played with the
    fingers, then the spec's own."""
    own = {k: spec[k] for k in (*DEFAULTS, *FINGERS) if k in spec}
    return {**DEFAULTS, **(PICK if picking(spec) else {}), **(FINGERS if fingerstyle(spec) else {}), **own}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def hermite(p0, m0, p1, m1, tau):
    """Cubic Hermite from p0 (start slope m0, per unit tau) to p1 (end slope m1) at tau in 0..1."""
    t = np.asarray(tau, float)
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


# =============================================================================================== which strokes
def nearest_beat(t, beats):
    """(index of the beat nearest to t, signed distance to it in beats; the spacing around that beat is the unit)."""
    b = np.asarray(beats, float)
    if len(b) < 2:
        raise StrumError("the timeline needs at least two beats")
    k = int(np.clip(np.searchsorted(b, t), 0, len(b) - 1))
    if k > 0 and abs(b[k - 1] - t) <= abs(b[k] - t):
        k -= 1
    lo, hi = max(k - 1, 0), min(k + 1, len(b) - 1)
    return k, float((t - b[k]) / ((b[hi] - b[lo]) / (hi - lo)))


def directions(times, beats, beat_tol=DEFAULTS["beat_tol"]):
    """Stroke directions for strike times: a strike on a beat goes down, one on the offbeat (half a beat later) goes up, any
    other (a sixteenth, a pickup) alternates with the stroke before it (the first goes down)."""
    out = []
    for t in times:
        ph = abs(nearest_beat(t, beats)[1])
        if ph <= beat_tol:
            out.append(1)
        elif abs(ph - 0.5) <= beat_tol:
            out.append(-1)
        else:
            out.append(-out[-1] if out else 1)
    return out


def beat_grid(beats, per_beat, t0=-math.inf, t1=math.inf):
    """(times, sub): strike times on a grid of `per_beat` strokes per beat (2 = eighths) built from the beat list and the
    index within the beat (0 = on the beat), for the grid points inside [t0, t1]."""
    b = np.asarray(beats, float)
    if len(b) < 2 or per_beat < 1:
        raise StrumError("a beat grid needs at least two beats and one stroke per beat")
    t = (b[:-1, None] + np.arange(per_beat)[None, :] * np.diff(b)[:, None] / per_beat).ravel()
    sub = np.tile(np.arange(per_beat), len(b) - 1)
    t, sub = np.r_[t, b[-1]], np.r_[sub, 0]
    keep = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    return t[keep], sub[keep]


def parse_rhythm(rhythm):
    """('onsets', stem) | ('beats', strokes per bar) | ('times', [t, ...]) | ('notes', None) from `rhythm = "onsets:other"`,
    `"beats:8"` (eight strokes per bar of four beats: eighths), a list of strike times or `"notes"` (the bass line the
    fretting hand plays, one string per stroke)."""
    if isinstance(rhythm, (list, tuple)):
        return "times", [float(t) for t in rhythm]
    if rhythm == "notes":
        return "notes", None
    kind, _, arg = str(rhythm).partition(":")
    if kind == "onsets" and arg:
        return "onsets", arg
    if kind == "beats" and arg.isdigit() and int(arg) >= 4 and int(arg) % 4 == 0:
        return "beats", int(arg)
    raise StrumError(f"rhythm {rhythm!r}: expected \"onsets:<stem>\", \"beats:<strokes per bar: 4, 8, 16>\" (beats:8 = eighths), "
                     f"\"notes\" (the fretting hand's bass line) or a list of times")


def plan(spec, timeline, duration=None, notes=None):
    """The strokes of one (resolved: `resolved`) strum spec over a timeline (clip seconds):

        rhythm   "onsets:other" (timeline["onsets"]["other"]), "beats:8" (strokes per bar of four beats from the beat list),
                 "notes" (picking: the `notes` given, [(t, c, w, string)] from `pick_notes`) or a list of strike times
        from, to the window (clip seconds, default the whole clip): only strikes inside it are played
        accent   "downbeats" | "beats" | a list of times: strikes within `accent_window` of those are accented
        technique "fingers" plays the notes with rest strokes (needs rhythm "notes"), else a pick
        + the DEFAULTS keys accent_gain, up_scale, accent_window, min_gap, beat_tol (PICK's defaults when picking)

    Directions come from the beat grid (`directions`; the grid's own parity for `beats:N`); picked notes alternate strictly,
    down first and down again after a pause long enough for the hand to rest (lead, attack, follow, approach and retreat
    and 0.1 s). Notes played with the fingers all go toward the lower strings (dir -1) and alternate the index and middle
    fingers strictly, the index first and again after a gap longer than `pause` (FINGERS). Strikes closer than
    min_gap are one stroke (the earlier). Returns a sorted list of Stroke (a picked one with its string's `c`, `w` and
    `string`, one played with the fingers also with its `finger`)."""
    p = params(spec)
    fingers = fingerstyle(spec)
    beats = list(timeline.get("beats") or (timeline.get("tempo") or {}).get("beats") or [])
    downbeats = list(timeline.get("downbeats") or (timeline.get("tempo") or {}).get("downbeats") or [])
    t0 = float(spec.get("from", -math.inf))
    t1 = float(spec.get("to", math.inf if duration is None else duration))
    kind, arg = parse_rhythm(spec.get("rhythm", "beats:8"))
    if fingers and kind != "notes":
        raise StrumError(f"technique \"fingers\" plucks a bass line one note at a time: give rhythm = \"notes\" (the fretting "
                         f"hand's `notes`), or technique = \"pick\" to play rhythm = {spec.get('rhythm', 'beats:8')!r}")
    sub = geo = None
    if kind == "onsets":
        on = (timeline.get("onsets") or {}).get(arg)
        if on is None:
            raise StrumError(f"the timeline has no onsets for {arg!r} (have {sorted(timeline.get('onsets') or {})}): run `mk timeline onsets`")
        times = np.array(sorted(float(t) for t in on))
    elif kind == "beats":
        times, sub = beat_grid(beats, arg // 4, t0, t1)
    elif kind == "notes":
        if notes is None:
            raise StrumError("rhythm \"notes\" picks the bass line the fretting hand plays (its `notes`): none was given")
        geo = sorted((tuple(n) for n in notes), key=lambda n: float(n[0]))
        times = np.array([float(n[0]) for n in geo])
    else:
        times = np.array(sorted(arg))
    keep = (times >= t0 - 1e-9) & (times <= t1 + 1e-9)
    times = times[keep]
    sub = None if sub is None else sub[keep]
    geo = None if geo is None else [g for g, k in zip(geo, keep) if k]
    kept = []
    for i, t in enumerate(times):                              # doubled onsets (a detector's ghost) are one stroke
        if not kept or t - times[kept[-1]] >= float(p["min_gap"]):
            kept.append(i)
    if not kept:
        return []
    times = times[kept]
    who = [None] * len(times)
    if geo is not None:
        geo = [geo[i] for i in kept]
        rest = (float(p["pause"]) if fingers else
                sum(float(p[k]) for k in ("lead", "attack", "follow", "approach", "retreat")) + 0.1)
        dirs = []
        for i, t in enumerate(times):                          # alternate picking, down after every rest
            dirs.append(1 if not dirs or t - times[i - 1] > rest else -dirs[-1])
        if fingers:                                            # index and middle alternate, the index after every rest
            who = ["index" if d > 0 else "middle" for d in dirs]
            dirs = [-1] * len(times)                           # every rest stroke pulls toward the lower strings
    elif sub is not None:
        dirs = [1 if int(s) % 2 == 0 else -1 for s in sub[kept]]       # beats down, the subdivisions between them alternate
    elif len(beats) >= 2:
        dirs = directions(times, beats, float(p["beat_tol"]))
    else:
        dirs = [1 if i % 2 == 0 else -1 for i in range(len(times))]
    marks = spec.get("accent")
    if marks == "downbeats":
        marks = downbeats
    elif marks == "beats":
        marks = beats
    elif marks in (None, False):
        marks = []
    elif isinstance(marks, str):
        raise StrumError(f"accent {marks!r}: expected \"downbeats\", \"beats\" or a list of times")
    marks = np.asarray(marks, float)
    out = []
    for i, (t, d) in enumerate(zip(times, dirs)):
        acc = bool(len(marks) and np.min(np.abs(marks - t)) <= float(p["accent_window"]))
        amp = (float(p["accent_gain"]) if acc else 1.0) * (float(p["up_scale"]) if d < 0 else 1.0)
        if geo is None:
            out.append(Stroke(t, d, amp, acc, spec.get("depth")))
        else:
            _t, c, w, string = geo[i]
            out.append(Stroke(t, d, amp, acc, spec.get("depth"), c=c, w=w, string=string, finger=who[i]))
    return out


def pick_notes(placed, neck, entry):
    """[(t, c, w, string)] of a placed bass line (mkmmd.core.fretting.place / line): per note the place `c` of its string on
    the strum axis of the card's strum entry `entry` (metres from its centre along `across`, at the centre's along-coordinate
    on the neck entry's strings) and the half-width `w` of the pick's contact with it (the string's radius + CONTACT)."""
    a, c = (np.asarray(entry[k], float) for k in ("along", "across"))
    a, c = a / np.linalg.norm(a), c / np.linalg.norm(c)
    centre = np.asarray(entry["center"], float)
    s_c = float(centre @ a)
    out = []
    for n in placed:
        st = FR.string_entry(neck, n["string"])
        u = float((FR.string_point(neck, n["string"], s_c) - centre) @ c)
        out.append((float(n["t"]), u, float(st.get("radius", 0.0006)) + CONTACT, int(n["string"])))
    return out


# =============================================================================================== where the pick is
def geometry(stroke, sigma):
    """(c, w) of a stroke: where on the strum axis it strikes and the half-width of what it crosses: a picked string's own
    place and contact, else the strum's 0 and `sigma` (half the width of the strings)."""
    return (0.0 if stroke.c is None else stroke.c), (float(sigma) if stroke.w is None else stroke.w)


def half_travel(stroke, span, sigma):
    """Half the travel of a stroke that crosses a width 2 sigma about its centre: it always clears it by a few millimetres."""
    return max(0.5 * span * stroke.amp, sigma + 0.008)


def height(u, half, sigma, depth, lift, c=0.0):
    """The pick tip's height h over the strings at sideways place u of a stroke centred on c: pressed in (-depth) across the
    width 2 sigma it strikes, rising to `lift` over the stretch from there to the far end of the sweep (`half` from c)."""
    q = np.clip((np.abs(np.asarray(u, float) - c) - sigma) / max(half - sigma, 1e-6), 0.0, 1.0)
    return -depth + (lift + depth) * smoothstep(q)


def depth_of(stroke, p):
    d = float(p["depth"]) if stroke.depth is None else stroke.depth
    return d * (float(p["accent_gain"]) if stroke.accent else 1.0) * (float(p["up_scale"]) if stroke.dir < 0 else 1.0)


def turn_share(share, pivot, centre, along, across):
    """(share, kappa): the share of the sideways travel the pick hand makes by turning about its wrist (about an axis along
    the strings through `pivot`, the wrist head) and the tip's sideways travel per radian of that turn (m). Turning moves the
    tip along `along` x r, r the lever from the wrist to the strum `centre` across the strings' axis: sideways when the wrist
    is out of the face over the pick (a forearm rotation), into the face when the wrist hangs above the strings on the bass
    side. The turn is faded out as its sideways part falls from TURN_SIDEWAYS[1] to [0] of its motion, so a hand does not
    swing out of the face to make up for a turn that pushes the tip into it (the arm carries the stroke instead)."""
    a, c = (np.asarray(v, float) for v in (along, across))
    a, c = a / np.linalg.norm(a), c / np.linalg.norm(c)
    r = np.asarray(centre, float) - np.asarray(pivot, float)
    r = r - (r @ a) * a
    kappa = float(np.cross(a, r) @ c)
    lever = float(np.linalg.norm(r))
    if abs(kappa) <= TURN_MIN or lever < 1e-9:
        return 0.0, kappa
    lo, hi = TURN_SIDEWAYS
    return float(share) * float(smoothstep((abs(kappa) / lever - lo) / (hi - lo))), kappa


def wrist_turn(pivot, centre, across):
    """(axis, lever) of the turn a picking hand makes about its wrist: the axis square to the sweep across the strings and to
    the lever from the wrist (`pivot`) to the pick tip at `centre` (its part across the strings removed), so that turning by
    a small angle moves the tip across the strings by `lever` metres per radian (the wrist deviating and the forearm
    rolling, as the hand lies). (None, 0.0) for a lever shorter than TURN_MIN."""
    c = np.asarray(across, float)
    c = c / np.linalg.norm(c)
    r = np.asarray(centre, float) - np.asarray(pivot, float)
    r = r - (r @ c) * c
    lever = float(np.linalg.norm(r))
    if lever < TURN_MIN:
        return None, 0.0
    k = np.cross(r, c)
    return k / np.linalg.norm(k), lever


def _timing(strokes, p):
    """Each stroke's attack, lead and follow-through: neighbours share the time between their strikes."""
    for s in strokes:
        s.attack, s.lead, s.follow = float(p["attack"]), float(p["lead"]), float(p["follow"])
    for a, b in zip(strokes, strokes[1:]):
        a.attack = min(a.attack, max(b.t - a.t - 0.03, 0.02))
        free = (b.t - a.t) - a.attack
        if a.dir != b.dir:                                     # turn-round: one continuous oscillation
            a.follow = b.lead = min(float(p["follow"]), 0.5 * free)
        else:                                                  # same direction: the return in between is a move of its own
            a.follow = b.lead = min(0.06, 0.2 * free)


def extremes(strokes, span, sigma):
    """[(u at the start of the run-up, u at the end of the follow-through)] per stroke: an alternating pair turns round at
    the farther of the two strokes' far ends (the one that clears both)."""
    geo = [geometry(s, sigma) for s in strokes]
    half = [half_travel(s, span, w) for s, (_c, w) in zip(strokes, geo)]
    ends = [[c - s.dir * h, c + s.dir * h] for s, (c, _w), h in zip(strokes, geo, half)]
    for i in range(len(strokes) - 1):
        if strokes[i].dir != strokes[i + 1].dir:
            d = strokes[i].dir
            e = d * max(d * ends[i][1], d * ends[i + 1][0])
            ends[i][1] = ends[i + 1][0] = e
    return ends


def path(strokes, ts, sigma, **kw):
    """The pick tip's place at the times `ts` (clip seconds) for `strokes` (call `plan`): {u, h (arrays, metres from the rest
    pose), contact (bool: between the first and last string of a stroke), stroke (the stroke a time belongs to, -1 between
    strokes and at rest)}. `sigma` = half the width of the strings (m; a picked stroke crosses only its own string); kw
    overrides the DEFAULTS (pass `params(spec)` for a picking spec). A picked stroke comes in from rest and goes back to it
    over the strings, lifted by `return_lift`."""
    p = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    ts = np.asarray(ts, float)
    u, h = np.zeros(len(ts)), np.zeros(len(ts))
    contact, owner = np.zeros(len(ts), bool), np.full(len(ts), -1, int)
    if not strokes:
        return {"u": u, "h": h, "contact": contact, "stroke": owner}
    span, lift, rlift, sg = float(p["span"]), float(p["lift"]), float(p["return_lift"]), float(sigma)
    _timing(strokes, p)
    ext = extremes(strokes, span, sg)
    info = []                                    # per stroke: t_start, t_end, u_start, u_end, half, depth, c, w, picked
    for i, s in enumerate(strokes):
        c, w = geometry(s, sg)
        half, dep = half_travel(s, span, w), depth_of(s, p)
        u_s, u_e = ext[i]
        u_1, u_2 = c - s.dir * w, c + s.dir * w
        t_s, t_1, t_2, t_e = s.t - s.lead, s.t, s.t + s.attack, s.t + s.attack + s.follow
        v = (u_2 - u_1) / s.attack
        for lo, hi, kind in ((t_s, t_1, "A"), (t_1, t_2, "B"), (t_2, t_e, "C")):
            m = (ts >= lo) & ((ts <= hi) if kind == "C" else (ts < hi))
            if not m.any():
                continue
            if kind == "A":
                dist = u_1 - u_s
                m1 = math.copysign(min(abs(v) * s.lead, 3.0 * abs(dist)), dist)       # monotone: slope <= 3 x the travel
                uu = hermite(u_s, 0.0, u_1, m1, (ts[m] - lo) / max(hi - lo, 1e-9))
            elif kind == "B":
                uu = u_1 + v * (ts[m] - lo)
                contact[m] = True
            else:
                dist = u_e - u_2
                m0 = math.copysign(min(abs(v) * s.follow, 3.0 * abs(dist)), dist)
                uu = hermite(u_2, m0, u_e, 0.0, (ts[m] - lo) / max(hi - lo, 1e-9))
            u[m] = uu
            h[m] = height(uu, half, w, dep, lift, c)
            owner[m] = i
        info.append((t_s, t_e, u_s, u_e, half, dep, c, w, s.c is not None))
    n = len(strokes)
    for i in range(n + 1):                                     # before the first stroke, between strokes, after the last
        if i == 0:
            a = None
            b = info[0]
        elif i == n:
            a = info[-1]
            b = None
        else:
            a, b = info[i - 1], info[i]
        ua = 0.0 if a is None else a[3]
        ha = 0.0 if a is None else float(height(a[3], a[4], a[7], a[5], lift, a[6]))
        ub = 0.0 if b is None else b[2]
        hb = 0.0 if b is None else float(height(b[2], b[4], b[7], b[5], lift, b[6]))
        t_a = None if a is None else a[1]
        t_b = None if b is None else b[0]
        bump_out = rlift if a is not None and a[8] else 0.0  # a picked stroke's hand rises over the strings to and from rest
        bump_in = rlift if b is not None and b[8] else 0.0
        if a is None:                                          # in from rest
            legs = [(t_b - float(p["approach"]), t_b, 0.0, 0.0, ub, hb, bump_in)]
        elif b is None:                                        # back to rest
            legs = [(t_a, t_a + float(p["retreat"]), ua, ha, 0.0, 0.0, bump_out)]
        elif t_b - t_a > float(p["approach"]) + float(p["retreat"]) + 0.1:     # a pause: rest in between
            legs = [(t_a, t_a + float(p["retreat"]), ua, ha, 0.0, 0.0, bump_out),
                    (t_b - float(p["approach"]), t_b, 0.0, 0.0, ub, hb, bump_in)]
        else:                                                  # straight over; across the strings the return is lifted
            legs = [(t_a, t_b, ua, ha, ub, hb, rlift if abs(ub - ua) > 0.5 * span else 0.0)]
        for a0, a1, u0, h0, u1, h1, bump in legs:
            if a1 <= a0 + 1e-9:
                continue
            m = (ts > a0) & (ts < a1) & (owner < 0)
            if not m.any():
                continue
            r = (ts[m] - a0) / (a1 - a0)
            e = smoothstep(r)
            u[m] = u0 + (u1 - u0) * e
            h[m] = h0 + (h1 - h0) * e + bump * np.sin(np.pi * r)
    return {"u": u, "h": h, "contact": contact, "stroke": owner}


def window(strokes, **kw):
    """(first, last) clip time the plan moves the pick (the first approach and the last retreat), or None without strokes."""
    if not strokes:
        return None
    p = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    _timing(strokes, p)
    return (strokes[0].t - strokes[0].lead - float(p["approach"]), strokes[-1].t + strokes[-1].attack + strokes[-1].follow + float(p["retreat"]))


def strike_times(strokes):
    """([down strike times], [up strike times]): the moments the pick meets the first string of each stroke."""
    return [s.t for s in strokes if s.dir > 0], [s.t for s in strokes if s.dir < 0]


# =============================================================================================== the plucking hand (fingerstyle)
def stroke_times(stroke, attack, travel=None):
    """{pose: clip time} of one rest stroke: the fingertip swings at a steady speed from "hover" through "cross" to "rest", its
    place across the string in each pose `travel` {pose: m} (TRAVEL's for the ones missing), timed so that it crosses the
    string's axis (place 0) at the stroke's `t` and the whole swing (hover -> rest) takes `attack` seconds."""
    u = {**TRAVEL, **(travel or {})}
    if not u["hover"] > u["cross"] > u["rest"]:
        u = dict(TRAVEL)
    v = (u["hover"] - u["rest"]) / float(attack)
    return {p: stroke.t - u[p] / v for p in POSES}


def pluck_keys(strokes, travel=None, **kw):
    """Where the plucking hand is for strokes played with the fingers (`plan` with technique "fingers"): {"index": [(t, pose)],
    "middle": [(t, pose)], "station": [(t, string)]}, each in time order (poses: POSES). A stroke of finger f on string s
    swings from "hover" through "cross" (at the stroke's t) to "rest" on the next lower string (`stroke_times`, with the
    fingertip's places `travel` {string: {finger: {pose: m}}} the station's solve measured; TRAVEL without), its `attack` cut to
    0.45 of the time to the finger's stroke before and after; the finger stays at rest `hold` (less when its next stroke needs
    the time) and lifts back to "hover" over `recover`, where it waits. The hand is at the station of the string it plays
    (`station` keys: its moves between stations are eased, the others hold) and moves to the next note's station after a
    crossing, done when the stroke that needs it starts (MIN_SHIFT .. `shift` s). kw: the FINGERS numbers (`params(spec)`)."""
    p = {**DEFAULTS, **FINGERS, **{k: v for k, v in kw.items() if v is not None}}
    attack, hold, recover, shift = (float(p[k]) for k in ("attack", "hold", "recover", "shift"))
    travel = travel or {}
    out = {"index": [], "middle": [], "station": []}
    strokes = sorted(strokes, key=lambda s: s.t)
    if not strokes:
        return out
    seq = {f: [i for i, s in enumerate(strokes) if s.finger == f] for f in ("index", "middle")}
    if sum(len(v) for v in seq.values()) != len(strokes):
        bad = next(s for s in strokes if s.finger not in seq)
        raise StrumError(f"the stroke at {bad.t:.3f} s has no finger: plan it with technique = \"fingers\"")
    when = [None] * len(strokes)
    for f, idx in seq.items():
        for j, i in enumerate(idx):
            s = strokes[i]
            gaps = [s.t - strokes[idx[j - 1]].t] if j else []
            gaps += [strokes[idx[j + 1]].t - s.t] if j + 1 < len(idx) else []
            when[i] = stroke_times(s, min([attack] + [0.45 * g for g in gaps]), (travel.get(s.string) or {}).get(f))
    for f, idx in seq.items():
        keys = []
        for j, i in enumerate(idx):
            w = when[i]
            keys += [(w["hover"], "hover"), (w["cross"], "cross"), (w["rest"], "rest")]
            if j + 1 < len(idx):                               # lift off and wait for the next
                free = max(when[idx[j + 1]]["hover"] - w["rest"], 0.0)
                rc = min(recover, 0.6 * free)
                hd = min(hold, max(free - rc, 0.0))
                keys += [(w["rest"] + hd, "rest"), (w["rest"] + hd + rc, "hover")]
            else:
                keys += [(w["rest"] + hold, "rest"), (w["rest"] + hold + recover, "hover")]
        out[f] = keys
    st = [(when[0]["hover"] - shift, strokes[0].string)]
    for i in range(1, len(strokes)):
        if strokes[i].string == st[-1][1]:
            continue
        b = max(when[i]["hover"], strokes[i - 1].t + MIN_SHIFT)
        a = max(strokes[i - 1].t, min(b - MIN_SHIFT, max(b - shift, when[i - 1]["rest"])))
        st += [(a, st[-1][1]), (b, strokes[i].string)]
    out["station"] = st
    return out


def _track(keys, ts, linear=()):
    """(k0, k1, w) per time: the two keyed values a track blends between and the weight of the second, held before the first
    key and after the last; eased between keys, linear between the value pairs in `linear`."""
    ts = np.asarray(ts, float)
    tk = np.array([k[0] for k in keys], float)
    vals = [k[1] for k in keys]
    at = np.searchsorted(tk, ts, side="right") - 1             # the last key at or before each time (-1: before the first)
    k0, k1, w = [], [], np.zeros(len(ts))
    for i, j in enumerate(at):
        j = min(max(int(j), 0), len(keys) - 1)
        va = vals[j]
        if j == len(keys) - 1 or ts[i] < tk[0] or vals[j + 1] == va or tk[j + 1] <= tk[j] + 1e-9:
            k0.append(va)
            k1.append(va)
            continue
        x = (ts[i] - tk[j]) / (tk[j + 1] - tk[j])
        k0.append(va)
        k1.append(vals[j + 1])
        w[i] = x if (va, vals[j + 1]) in linear else float(smoothstep(x))
    return k0, k1, w


def pose_track(keys, ts):
    """(pose0, pose1, w) per time of one finger's keys (`pluck_keys`): the stroke itself (hover -> cross -> rest) is linear in
    time, so the fingertip crosses the string at a steady speed; the lift back to hovering is eased."""
    if not keys:
        n = len(np.asarray(ts))
        return ["hover"] * n, ["hover"] * n, np.zeros(n)
    return _track(keys, ts, linear=(("hover", "cross"), ("cross", "rest")))


def retime(w, us):
    """The blend weight that puts the fingertip where a steady swing has it a share `w` of the way through a stroke segment:
    `us` the fingertip's place across the string at evenly spaced blend weights 0..1 of the segment (the station's solve
    measures them, mkmmd.solvers.grip_pluck `curves`). Without a monotone measurement the weight is `w` itself."""
    us = np.asarray(us, float)
    if len(us) < 2 or not (np.all(np.diff(us) < 0) or np.all(np.diff(us) > 0)):
        return float(w)
    grid = np.linspace(0.0, 1.0, len(us))
    target = us[0] + (us[-1] - us[0]) * float(w)
    return float(np.interp(target, us, grid) if us[-1] > us[0] else np.interp(-target, -us, grid))


def station_track(keys, ts):
    """(string0, string1, w) per time of the hand's station keys (`pluck_keys`)."""
    return _track(keys, ts)


def pluck_window(keys):
    """(first, last) clip time the plucking hand moves (`pluck_keys`), or None without keys."""
    t = [k[0] for v in keys.values() for k in v]
    return (min(t), max(t)) if t else None


def pluck_frame(entry):
    """(3, 3) the station frame's axes (columns) in the prop frame for a strum entry: x across the strings toward the higher
    ones, y along them toward the nut, z out of the face."""
    z = np.asarray(entry.get("normal", (0.0, -1.0, 0.0)), float)
    z = z / np.linalg.norm(z)
    y = np.asarray(entry.get("along", (0.0, 0.0, 1.0)), float)
    y = y - (y @ z) * z
    if np.linalg.norm(y) < 1e-6:
        raise StrumError("the strum entry's `along` lies along its `normal`")
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    if x @ np.asarray(entry.get("across", (1.0, 0.0, 0.0)), float) <= 0.0:
        raise StrumError("the strum entry's `across` (lowest string -> highest) must be along x `along` cross `normal`")
    return np.stack([x, y, z], 1)


def pluck_prop(neck, entry, string):
    """(prop, (origin, R)): the station problem of the grip style `pluck` (mkmmd.solvers.grip_pluck) for string `string`
    (1 = the highest) of the card's neck entry, from its strum entry's `fingers` table {zone: [along0, along1] (where the
    fingers pluck), anchor: {center, normal, size: [across, along]} (where the thumb rests while the lowest string rings: the
    pickup cover's top beyond it), face: {point, normal} (the surface under the strings), blocks: [{center, size}] (boxes
    standing on it: the pickup covers)}, all in the prop frame; and the station frame in the prop frame (`pluck_frame`), its
    origin on the string's axis in the middle of the zone. The thumb rests on the anchor when the string is the lowest, on the
    lowest string otherwise."""
    fg = entry.get("fingers")
    if not fg or fg.get("zone") is None or fg.get("face") is None:
        raise StrumError(f"the card's strum entry {entry.get('name', 'strum')!r} has no `fingers` table (zone, anchor, face): "
                         f"it is played with a pick (technique = \"pick\")")
    n = len(neck["strings"])
    FR.string_entry(neck, string)
    if int(string) == n and fg.get("anchor") is None:
        raise StrumError("the strum entry's `fingers` has no `anchor` (where the thumb rests while the lowest string rings)")
    R = pluck_frame(entry)
    zone = sorted(float(v) for v in fg["zone"])
    o = FR.string_point(neck, string, 0.5 * (zone[0] + zone[1]))
    put = lambda v: (R.T @ (np.asarray(v, float) - o)).tolist()                     # noqa: E731
    turn = lambda v: (R.T @ np.asarray(v, float)).tolist()                           # noqa: E731
    prop = {"strings": [{"a": put(s["nut"]), "b": put(s["bridge"]), "radius": float(s.get("radius", 0.0006))}
                        for s in neck["strings"]],
            "string": n - int(string), "zone": [zone[0] - 0.5 * sum(zone), zone[1] - 0.5 * sum(zone)],
            "face": {"point": put(fg["face"]["point"]), "normal": turn(fg["face"]["normal"])},
            "blocks": [{"center": put(b["center"]), "axes": R.T.tolist(), "size": [float(v) for v in b["size"]]}
                       for b in fg.get("blocks") or ()]}
    if int(string) == n:
        a = fg["anchor"]
        prop["anchor"] = {"type": "plane", "center": put(a["center"]), "normal": turn(a["normal"]),
                          "size": [float(v) for v in a["size"]]}
    else:
        prop["anchor"] = {"type": "string", "string": 0}
    return prop, (o, R)
