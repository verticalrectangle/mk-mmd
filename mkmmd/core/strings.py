"""Strings that ring: which strings of a worn guitar-like prop vibrate when and over what length, and how a fretted string
bends down under the finger (docs/design.md: Playing a worn guitar, Strings that ring). numpy only: the perform stage keys
it, the `string_play` check replays it and the tests run it without Blender.

Strings are numbered as players number them (1 = the highest ... n = the lowest of the card's n strings, which the card's
neck entry lists lowest first). Along a string, u runs from 0 at the nut to 1 at the saddle (the card's `nut` and `bridge`
points on its axis).

    PRESSES  {string, fret, on, off, in, out}: a fretting finger holds the string down just behind wire `fret` from `on` to
             `off` (clip seconds), coming down over `in` seconds before `on` and letting go over `out` after `off`
             (`presses`: from the fretting hand's shapes, the pose stage's neck states).
    HITS     (t, string, gain): the pick or a finger lets go of the string at t (`hits`: from the strokes of a strum plan,
             each string at the moment the pick or the fingertip crosses it).
    RINGS    (t0, t1, string, fret, gain): the string vibrates from t0 between wire `fret` (0: the nut) and the saddle, its
             amplitude decaying with time constant DECAY, until t1 (the next hit on it, the finger that frets it letting
             go, or the next note of a line played one note at a time) when it is damped over DAMP (`rings`).

What each frame shows (`frames`): per string the pressed share of each fret held on it (0..1), the ring's amplitude `amp`
(metres: the half-width the blur reaches midway along its vibrating length), its start `u0` (the vibrating length runs
from u0 to 1), its `fret` and a `phase` for the shimmer.

The shape (`pressed_axis`, `ring_offsets`): a pressed string runs straight from the nut down to the finger (`press` of the
gap behind its wire, at the height of the wire's crown plus its radius, as mkmmd.core.fretting.finger_targets puts the pad),
along the crown to the wire and straight up to the saddle; a ringing one is a spindle around that axis whose section is an
ellipse of half-axes r + a p(u) across the board and r + ECC a p(u) out of it, p(u) = sin(pi (u - u0) / (1 - u0)): the
time-averaged shape of a string swinging (mostly sideways, a little up and down) in its first mode, as a camera at 30
frames a second sees it."""
import math

import numpy as np

from . import fretting as FR
from . import strum as SM

AMP = 0.0031                 # m of blur half-width at the antinode per metre of the scale: 2.4 mm on a 30 inch bass
DECAY = 0.9                  # s: the amplitude falls to 1/e this long after the string is let go
DAMP = 0.06                  # s: a string a finger lets go of (or the next note mutes) stops over this long
SHIMMER = 0.06               # the amplitude flickers this share frame to frame (a camera catching the swing at random)
ECC = 0.45                   # out of the board a string swings this share of its sideways swing
PRESS_IN, PRESS_OUT = 0.05, 0.04     # s: a finger comes down over this long before its shape lands, lets go over this long
PRESS_SHARE = 0.25           # ... and never over more than this share of the move (the hand lifts its fingers in the first
                             # quarter of a change and lands them in the last: mkmmd.blender.build.pose)
FULL = 0.5                   # a string held down at least this share sounds from that fret
MIN_AMP = 1e-5               # m: below this a string is still
EDGE = 0.002                 # m: a strum crosses the strings within its sweep and this far past its edges (the outer ones)
RING_KEYS = ("amp", "decay", "damp", "shimmer")

# the open strings each named chord of mkmmd.core.fretting.CHORDS lets ring (its standard voicing; the others are muted)
OPEN = {"power": (), "power5": (), "E": (6, 2, 1), "A": (5, 1), "D": (4,), "G": (4, 3, 2), "C": (3, 1), "Em": (6, 3, 2, 1),
        "Am": (5, 1)}


class StringsError(ValueError):
    pass


def smoothstep(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ring_params(spec, scale):
    """The numbers of a strum spec's `ring` (true / absent: the defaults; false: no string rings, None returned; a table
    {amp, decay, damp, shimmer}: those win): amp in metres (default AMP times the neck's `scale`), decay, damp (s),
    shimmer (share)."""
    ring = spec.get("ring", True) if hasattr(spec, "get") else True
    if ring is False:
        return None
    if ring is True or ring is None:
        ring = {}
    if not hasattr(ring, "items"):
        raise StringsError(f"ring = {ring!r}: true, false or a table {{{', '.join(RING_KEYS)}}}")
    bad = sorted(set(ring) - set(RING_KEYS))
    if bad:
        raise StringsError(f"ring: unknown key {bad[0]!r} (have {', '.join(RING_KEYS)})")
    out = {"amp": AMP * float(scale), "decay": DECAY, "damp": DAMP, "shimmer": SHIMMER}
    for k, v in ring.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise StringsError(f"ring {k} = {v!r}: a number >= 0")
        out[k] = float(v)
    if out["decay"] <= 0.0:
        raise StringsError("ring decay must be > 0 seconds")
    return out


# =============================================================================================== presses, hits, rings
def presses(states, strings):
    """The presses [{string, fret, on, off, in, out}] of a fretting hand's shapes on a neck of `strings` strings. `states` =
    [(t, fret, chord, move)] as the pose stage solves them (mkmmd.blender.build.pose `_neck_grips`): shape `chord` at
    position `fret` in place at clip time t (None: from the start), reached over `move` seconds before it. A pressing finger
    holds its string from its shape's landing until the next shape starts moving (t_next - move_next); a finger that stays
    on the same string and fret through a change keeps holding it; "hover" fingers hold nothing."""
    out, held = [], {}
    for i, (t, fret, chord, move) in enumerate(states):
        land = -math.inf if t is None else float(t)
        shape = FR.shape(chord, strings)
        hover = FR.hovering(chord)
        now = {(s, int(fret) + int(k)) for f, (s, k) in shape.items() if f not in hover}
        leave = math.inf
        if i + 1 < len(states):
            t_n, _f, _c, mv_n = states[i + 1]
            leave = float(t_n) - float(mv_n)
        for key in list(held):
            if key not in now:
                del held[key]
        for key in now:
            if key in held:
                held[key]["off"] = leave
            else:
                p = {"string": key[0], "fret": key[1], "on": land, "off": leave,
                     "in": min(PRESS_IN, PRESS_SHARE * float(move)) if t is not None else 0.0, "out": 0.0}
                held[key] = p
                out.append(p)
        for p in held.values():
            if i + 1 < len(states):
                p["out"] = min(PRESS_OUT, PRESS_SHARE * float(states[i + 1][3]))
    return sorted(out, key=lambda p: (p["on"], p["string"]))


def press_value(p, ts):
    """How far press p holds its string down at the times ts (0..1): eased up over p["in"] before p["on"] and down over
    p["out"] after p["off"]."""
    ts = np.asarray(ts, float)
    up = np.where(ts >= p["on"], 1.0, smoothstep((ts - (p["on"] - p["in"])) / p["in"]) if p["in"] > 0 else 0.0)
    down = np.where(ts <= p["off"], 1.0, 1.0 - smoothstep((ts - p["off"]) / p["out"]) if p["out"] > 0 else 0.0)
    return up * down


def fretted(press_list, string, t):
    """The fret string `string` sounds from at time t: the highest one a finger holds it down at (a share FULL or more),
    0 (open) when none."""
    held = [p["fret"] for p in press_list if p["string"] == string and float(press_value(p, [t])[0]) >= FULL]
    return max(held) if held else 0


def sounding(states, strings):
    """f(string, t) -> whether a strum lets that string ring at time t: the strings the fretting hand's shape at t holds
    down and the open strings its named chord lets ring (`OPEN`; a chord table rings only the strings it presses); every
    string when there is no fretting hand (`states` empty)."""
    if not states:
        return lambda string, t: True
    marks = [(-math.inf if t is None else float(t) - float(mv)) for t, _f, _c, mv in states]

    def chord_at(t):
        i = max(k for k, m in enumerate(marks) if m <= t) if t >= marks[0] else 0
        return states[i]

    def ring(string, t):
        _t, fret, chord, _mv = chord_at(t)
        shape = FR.shape(chord, strings)
        hover = FR.hovering(chord)
        if any(s == string for f, (s, _k) in shape.items() if f not in hover):
            return True
        return isinstance(chord, str) and string in OPEN.get(chord, ())
    return ring


def hits(strokes, centres, sigma, p, ring=None):
    """[(t, string, gain)] sorted by time: when each stroke lets go of each string it plays. `strokes` are a strum plan's
    (mkmmd.core.strum.plan; their timing is filled here as `path` fills it), `centres` {string: its place on the strum axis
    (m from the strum centre along `across`)}, `sigma` half the width of the strings, `p` the path numbers
    (mkmmd.core.strum.params), `ring` f(string, t) -> whether a strum sounds that string (`sounding`; default all). A picked
    note sounds when the pick tip crosses its string's axis (the stroke's `t` + half its `attack`), a plucked one (a
    fingerstyle stroke, `finger` set) at its `t`; a strum crosses every string within its sweep one after another, at the
    contact speed of `path`, from its `t` (the first string) to `t + attack` (the last). Gain: the stroke's amplitude."""
    if not strokes:
        return []
    SM._timing(strokes, {**SM.DEFAULTS, **p})
    out = []
    for s in strokes:
        if s.string is not None:
            t = s.t if getattr(s, "finger", None) else s.t + 0.5 * float(s.attack)
            out.append((t, int(s.string), float(s.amp)))
            continue
        c, w = SM.geometry(s, sigma)
        start = c - s.dir * w
        for string, cn in centres.items():
            share = (float(cn) - start) * s.dir / (2.0 * w)
            if -EDGE / (2.0 * w) <= share <= 1.0 + EDGE / (2.0 * w):                  # the outer strings lie on the edges
                t = s.t + min(max(share, 0.0), 1.0) * float(s.attack)
                if ring is None or ring(string, t):
                    out.append((t, int(string), float(s.amp)))
    return sorted(out)


def rings(press_list, hit_list, mono=False):
    """[(t0, t1, string, fret, gain)] of the hits (`hits`) on strings held down as `press_list` says (`presses`): each
    rings from its hit at the fret it is held at then (`fretted`) until the next hit on that string, the moment the finger
    that holds it there lets go (when it was fretted) or, `mono` (a line played one note at a time), the next note on any
    string; t1 is inf for a string left to ring out."""
    hs = sorted(hit_list)
    out = []
    for i, (t, string, gain) in enumerate(hs):
        fret = fretted(press_list, string, t)
        t1 = math.inf
        for t2, s2, _g in hs[i + 1:]:
            if s2 == string or (mono and t2 > t + 1e-6):
                t1 = t2
                break
        if fret:
            offs = [p["off"] for p in press_list
                    if p["string"] == string and p["fret"] == fret and p["on"] - p["in"] <= t <= p["off"] + p["out"]]
            if offs:
                t1 = min(t1, max(min(offs), t))
        out.append((float(t), float(t1), int(string), int(fret), float(gain)))
    return out


def _noise(n, seed):
    """n values in [-1, 1], the same for the same seed (a string's shimmer frame to frame)."""
    return np.random.default_rng(int(seed)).uniform(-1.0, 1.0, int(n))


def frames(ring_list, press_list, ts, params, u_of_fret, strings):
    """What the frames at clip times ts show, per string 1..strings: {string: {"amp" (F,) m, "u0" (F,), "fret" (F,) int,
    "phase" (F,), "press" {fret: (F,) 0..1}}}. `params` are `ring_params`' (amp, decay, damp, shimmer), `u_of_fret(string,
    fret)` the u of a wire on a string (`fret_u`). A string shows the last ring that started at or before the frame, its
    amplitude amp * gain * exp(-(t - t0) / decay), eased to nothing over `damp` after its t1, flickering by `shimmer`."""
    ts = np.asarray(ts, float)
    nf = len(ts)
    out = {}
    for string in range(1, int(strings) + 1):
        amp, u0, fret = np.zeros(nf), np.zeros(nf), np.zeros(nf, int)
        mine = sorted(r for r in ring_list if r[2] == string)
        for k, (t0, t1, _s, fr, gain) in enumerate(mine):
            nxt = mine[k + 1][0] if k + 1 < len(mine) else math.inf
            sel = (ts >= t0) & (ts < nxt)
            if not sel.any():
                continue
            dt = ts[sel] - t0
            a = params["amp"] * gain * np.exp(-dt / params["decay"])
            if math.isfinite(t1):
                a = a * (1.0 - (smoothstep((ts[sel] - t1) / params["damp"]) if params["damp"] > 0 else (ts[sel] >= t1)))
            amp[sel], u0[sel], fret[sel] = a, u_of_fret(string, fr), fr
        if params["shimmer"]:
            amp = amp * (1.0 + params["shimmer"] * _noise(nf, 7919 * string + 17))
        amp[amp < MIN_AMP] = 0.0
        phase = np.round(np.abs(_noise(nf, 104729 * string + 3)) * 100.0, 3)
        held = {}
        for p in press_list:
            if p["string"] == string:
                v = press_value(p, ts)
                held[p["fret"]] = np.maximum(held.get(p["fret"], np.zeros(nf)), v)
        out[string] = {"amp": amp, "u0": u0, "fret": fret, "phase": phase, "press": held}
    return out


# =============================================================================================== the shape of a string
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def axis(entry, string):
    """(nut, bridge): the string's axis points (3,) at the nut and the saddle, the prop's frame."""
    st = FR.string_entry(entry, string)
    return np.asarray(st["nut"], float), np.asarray(st["bridge"], float)


def u_of(entry, string, points):
    """(k,) the place of points along the string, by projection on its axis: 0 at the nut, 1 at the saddle (beyond them
    outside 0..1: the headstock, the tail)."""
    a, b = axis(entry, string)
    d = b - a
    return (np.asarray(points, float).reshape(-1, 3) - a) @ d / float(d @ d)


def u_at(entry, string, s):
    """u of the along-coordinate s (the neck's `along`, prop frame) on the string: where across it a point of the neck
    at that coordinate lies along the string."""
    along = FR._entry_frame(entry)[0]
    a, b = axis(entry, string)
    return (float(a @ along) - float(s)) / (float(a @ along) - float(b @ along))


def fret_u(entry, string, fret):
    """u of wire `fret` (0: the nut) on the string: where its vibrating length starts when it is held there."""
    if int(fret) == 0:
        return 0.0
    return u_at(entry, string, float(np.asarray(entry["frets"][int(fret)], float) @ FR._entry_frame(entry)[0]))


def _held_point(entry, string, s, fret):
    """The string's axis held down at the along-coordinate s behind wire `fret`: at the wire crown's height plus its
    radius (mkmmd.core.fretting.finger_targets)."""
    normal = FR._entry_frame(entry)[2]
    r = float(FR.string_entry(entry, string).get("radius", 0.0006))
    p = FR.string_point(entry, string, s)
    base = np.asarray(entry["frets"][int(fret)], float)
    return p - ((p - base) @ normal) * normal + normal * (float(entry.get("fret_height", FR.FRET_HEIGHT)) + r)


def pressed_axis(entry, string, fret, u, press=FR.PRESS):
    """(k, 3): where the string's axis lies at the places u (`u_of`) while a finger holds it down behind wire `fret`
    (`press` of the gap behind it): straight from the nut to the finger, along the crown to the wire, straight to the
    saddle; outside 0..1 where it lies anyway."""
    zs = FR.fret_z(entry)
    k = int(fret)
    if not 1 <= k < len(zs):
        raise StringsError(f"fret {fret} is not a wire of the neck (1..{len(zs) - 1})")
    a, b = axis(entry, string)
    s_f = zs[k] + float(press) * (zs[k - 1] - zs[k])
    knots_u = np.array([0.0, u_at(entry, string, s_f), fret_u(entry, string, k), 1.0])
    knots = np.array([a, _held_point(entry, string, s_f, k), _held_point(entry, string, zs[k], k), b])
    u = np.asarray(u, float)
    uc = np.clip(u, 0.0, 1.0)
    P = np.stack([np.interp(uc, knots_u, knots[:, j]) for j in range(3)], 1)
    straight = a + u[:, None] * (b - a)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside[:, None], P, straight)


def press_offsets(entry, string, points, fret, press=FR.PRESS):
    """(k, 3): how far each point of the string's surface (prop frame) moves when a finger holds it down behind wire
    `fret`: its ring moves with the axis (`pressed_axis`)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    u = u_of(entry, string, pts)
    a, b = axis(entry, string)
    return pressed_axis(entry, string, fret, u, press) - (a + u[:, None] * (b - a))


def profile(u, u0):
    """sin(pi (u - u0) / (1 - u0)) on u0..1, 0 elsewhere: the first mode of a string vibrating from u0 to the saddle."""
    u = np.asarray(u, float)
    s = (u - float(u0)) / max(1.0 - float(u0), 1e-9)
    return np.where((s >= 0.0) & (s <= 1.0), np.sin(np.pi * np.clip(s, 0.0, 1.0)), 0.0)


def ring_offsets(entry, string, points, u0, unit):
    """(k, 3): how far each point of the string's surface (prop frame) moves when it rings `unit` metres wide at its
    antinode with its vibrating length from u0 to the saddle: its offset from the axis grows by unit p(u) across the board
    (the neck's `across`, square to the string) and by ECC unit p(u) out of it, so the ring of points becomes the
    ellipse the swinging string sweeps."""
    pts = np.asarray(points, float).reshape(-1, 3)
    a, b = axis(entry, string)
    d = _unit(b - a)
    _along, across, normal = FR._entry_frame(entry)
    ex = _unit(across - (across @ d) * d)
    ey = _unit(np.cross(d, ex))
    if float(ey @ normal) < 0.0:
        ey = -ey
    u = u_of(entry, string, pts)
    r = pts - (a + u[:, None] * (b - a))
    n = np.linalg.norm(r, axis=1)
    rh = r / np.maximum(n, 1e-12)[:, None]
    w = float(unit) * profile(u, u0)
    return w[:, None] * ((rh @ ex)[:, None] * ex + ECC * (rh @ ey)[:, None] * ey)
