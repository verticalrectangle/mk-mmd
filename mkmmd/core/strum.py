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
(`PICK` overrides the strum DEFAULTS) and strictly alternating, down first and again down after a pause the hand rests in."""
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
}
CONTACT = 0.0005             # m: a picked string's contact half-width is its radius plus this (the pick's half thickness)


class StrumError(ValueError):
    pass


class Stroke:
    __slots__ = ("t", "dir", "amp", "accent", "depth", "attack", "lead", "follow", "c", "w", "string")

    def __init__(self, t, dir, amp=1.0, accent=False, depth=None, c=None, w=None, string=None):
        self.t, self.dir, self.amp, self.accent = float(t), int(dir), float(amp), bool(accent)
        self.depth = None if depth is None else float(depth)    # m: this stroke's own depth (a spec's), else the path's
        self.attack = self.lead = self.follow = None            # filled in by `path`
        self.c = None if c is None else float(c)                # m: a picked string's place on the strum axis (None: a strum,
        self.w = None if w is None else float(w)                # centred on 0) and the half-width of its contact (None: sigma)
        self.string = string                                    # the picked string's number (1 = the highest), for reports

    def __repr__(self):
        what = "" if self.string is None else f", string {self.string}"
        return f"Stroke(t={self.t:.3f}, {'down' if self.dir > 0 else 'up'}, amp={self.amp:.2f}{', accent' if self.accent else ''}{what})"


def picking(spec):
    """True for a spec that picks single strings (`rhythm = "notes"`)."""
    return spec.get("rhythm") == "notes"


def params(spec):
    """The path numbers of a spec: DEFAULTS, PICK's on top for a picking spec, then the spec's own."""
    return {**DEFAULTS, **(PICK if picking(spec) else {}), **{k: spec[k] for k in DEFAULTS if k in spec}}


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
    """The strokes of one strum spec over a timeline (clip seconds):

        rhythm   "onsets:other" (timeline["onsets"]["other"]), "beats:8" (strokes per bar of four beats from the beat list),
                 "notes" (picking: the `notes` given, [(t, c, w, string)] from `pick_notes`) or a list of strike times
        from, to the window (clip seconds, default the whole clip): only strikes inside it are played
        accent   "downbeats" | "beats" | a list of times: strikes within `accent_window` of those are accented
        + the DEFAULTS keys accent_gain, up_scale, accent_window, min_gap, beat_tol (PICK's defaults when picking)

    Directions come from the beat grid (`directions`; the grid's own parity for `beats:N`); picked notes alternate strictly,
    down first and down again after a pause long enough for the hand to rest (lead, attack, follow, approach and retreat
    and 0.1 s). Strikes closer than min_gap are one stroke (the earlier). Returns a sorted list of Stroke (a picked one with
    its string's `c`, `w` and `string`)."""
    p = params(spec)
    beats = list(timeline.get("beats") or (timeline.get("tempo") or {}).get("beats") or [])
    downbeats = list(timeline.get("downbeats") or (timeline.get("tempo") or {}).get("downbeats") or [])
    t0 = float(spec.get("from", -math.inf))
    t1 = float(spec.get("to", math.inf if duration is None else duration))
    kind, arg = parse_rhythm(spec.get("rhythm", "beats:8"))
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
    if geo is not None:
        geo = [geo[i] for i in kept]
        rest = sum(float(p[k]) for k in ("lead", "attack", "follow", "approach", "retreat")) + 0.1
        dirs = []
        for i, t in enumerate(times):                          # alternate picking, down after every rest
            dirs.append(1 if not dirs or t - times[i - 1] > rest else -dirs[-1])
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
            out.append(Stroke(t, d, amp, acc, spec.get("depth"), c=c, w=w, string=string))
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
