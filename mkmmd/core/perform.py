"""Performance curves (pure numpy; run in Blender and in the CLI): easing, followers, gaze event envelopes, blink
schedules, breathing / sway / nod, beat bob. Times are clip seconds unless a name says frames.

Reference numbers (people at rest, tracked in reference clips): blink ~0.157 s; blinks every ~3-6 s when talking or
looking around, 5-20 s apart when concentrating; breathing ~16 / min; head nod ~0.5 deg; torso sway ~0.4 deg."""
import math
import random

import numpy as np

BLINK_S = 0.157


def smooth(x):
    """Smoothstep on [0, 1], clamped (scalar or array)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def lowpass(arr, tau_frames):
    """Critically damped follow (two cascaded exponentials), forward in time, along axis 0."""
    arr = np.asarray(arr, float)
    if tau_frames <= 0:
        return arr.copy()
    a = 1.0 - math.exp(-1.0 / tau_frames)
    out = arr.copy()
    for _ in range(2):
        y = out[0].copy()
        for i in range(len(out)):
            y = y + a * (out[i] - y)
            out[i] = y
    return out


def envelope(t, t_on, t_hold, t_back, rise, fall=None):
    """0 -> 1 -> 0: rises over `rise` s from t_on, holds until t_hold, returns by t_back (array or scalar t)."""
    fall = fall if fall is not None else max(t_back - t_hold, 1e-3)
    return smooth((np.asarray(t, float) - t_on) / max(rise, 1e-3)) * (1.0 - smooth((np.asarray(t, float) - t_hold) / fall))


def eased_keys(ts, keys):
    """Values over time from keys [[t, v], ...]: smoothstep-eased between consecutive keys, held before the first and
    after the last; zeros without keys."""
    ts = np.asarray(ts, float)
    if not keys:
        return np.zeros(len(ts))
    ks = sorted((float(t), float(v)) for t, v in keys)
    out = np.full(len(ts), ks[0][1])
    for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
        m = ts >= t0
        out[m] = v0 + (v1 - v0) * smooth((ts[m] - t0) / max(t1 - t0, 1e-6))
    return out


class GazeEvents:
    """Look-at events [(t_on, t_hold_end, t_back, target_index, rise_s)] over an idle target. weights(t) -> (E, F)."""

    def __init__(self, events):
        self.events = sorted(events, key=lambda e: e[0])

    def weights(self, ts):
        ts = np.asarray(ts, float)
        W = np.zeros((len(self.events), len(ts)))
        for i, (t_on, t_hold, t_back, _tgt, rise) in enumerate(self.events):
            W[i] = envelope(ts, t_on, t_hold, t_back, rise)
        total = W.sum(0)
        over = total > 1.0
        W[:, over] /= total[over]
        return W


def glance_weights(ts, glances):
    """(G, F) weights of eye-only glances [(t_on, dur, rise, fall)]: a smooth rise over `rise` s from t_on, held for `dur`
    s, a smooth fall over `fall` s. Where glances overlap their weights are scaled to add up to one."""
    ts = np.asarray(ts, float)
    W = np.zeros((len(glances), len(ts)))
    for i, (t_on, dur, rise, fall) in enumerate(glances):
        W[i] = envelope(ts, t_on, t_on + dur, t_on + dur + fall, rise, fall)
    total = W.sum(0)
    over = total > 1.0
    W[:, over] /= total[over]
    return W


def blink_schedule(t0, t1, per_min=15.0, seed=0, extra=(), avoid=(), min_gap=1.2, jitter=0.45):
    """[(t, duration)] between t0 and t1: `extra` blinks (t, dur) at chosen moments (gaze shifts, startles), then
    natural blinks with mean interval 60/per_min s (+- jitter share), at least min_gap s from any other, never inside
    an `avoid` interval (t_a, t_b)."""
    rnd = random.Random(seed)
    out = [(float(t), float(d)) for t, d in extra]
    mean = 60.0 / max(per_min, 1e-3)
    t = t0 - rnd.uniform(0, mean)
    while t < t1:
        t += mean * rnd.uniform(1.0 - jitter, 1.0 + jitter)
        if t < t0 or t > t1:
            continue
        if any(abs(t - b) < min_gap for b, _ in out) or any(a <= t <= b for a, b in avoid):
            continue
        out.append((t, BLINK_S))
    return sorted(out)


def blink_curve(ts, blinks):
    """Lid closure 0..1 per time for a blink list (fast close, short hold, slower open)."""
    ts = np.asarray(ts, float)
    out = np.zeros(len(ts))
    for bt, d in blinks:
        u = (ts - bt) / d
        m = (u >= -0.01) & (u <= 1.0)
        if not m.any():
            continue
        uu = u[m]
        v = np.where(uu < 0.4, smooth(uu / 0.4), np.where(uu < 0.55, 1.0, 1.0 - smooth((uu - 0.55) / 0.45)))
        out[m] = np.maximum(out[m], v)
    return out


def breathing(ts, per_min=16.5, deg=0.6, phase=0.0):
    """Chest pitch (rad) for breathing."""
    return math.radians(deg) * np.sin(2 * math.pi * np.asarray(ts, float) * per_min / 60.0 + phase)


def sway(ts, period=2.5, deg=0.37, phase=0.8):
    return math.radians(deg) * np.sin(2 * math.pi * np.asarray(ts, float) / period + phase)


def rock(ts, period=4.0, deg=3.0, phase=0.0):
    """Slow sideways rock (rad, + toward the model's left): `deg` * sin(2 pi t / `period` + `phase`) of clip time, so it is
    the same sway at whatever time a clip is cut (keyed tilts only cover their own span)."""
    return math.radians(deg) * np.sin(2 * math.pi * np.asarray(ts, float) / period + phase)


def beat_pulse(ts, beats, attack=0.06, decay=0.22, accent=None):
    """A pulse on each beat (0..1 times the beat's `accent` weight): a quick rise ending on the beat, an exponential fall
    after it. Overlapping pulses take the larger."""
    ts = np.asarray(ts, float)
    out = np.zeros(len(ts))
    for i, b in enumerate(beats):
        a = 1.0 if accent is None else float(accent[i])
        u = ts - b
        m = (u > -attack) & (u < decay * 4)
        if not m.any():
            continue
        uu = u[m]
        shape = smooth((uu + attack) / attack) * np.exp(-np.maximum(uu, 0.0) / decay)
        out[m] = np.maximum(out[m], a * shape)
    return out


def beat_bob(ts, beats, deg=1.5, attack=0.06, decay=0.22, accent=None):
    """Head nod (rad) on the beats: a quick dip after each beat, easing back; accent (per beat weight) scales it."""
    return math.radians(deg) * beat_pulse(ts, beats, attack, decay, accent)


def tap(ts, beats, lift=0.16, accent=None):
    """A finger tap landing on each beat (0..1 times the beat's `accent` weight): the finger rises over the first 55 % of
    the `lift` seconds before the beat and falls, faster and faster, onto it (the strike is the fastest moment, on the
    beat); it rests from the beat to the next rise. Overlapping taps take the larger."""
    ts = np.asarray(ts, float)
    out = np.zeros(len(ts))
    lift = float(lift)
    up = 0.55 * lift
    down = lift - up
    for i, b in enumerate(beats):
        a = 1.0 if accent is None else float(accent[i])
        u = ts - (float(b) - lift)
        m = (u > 0.0) & (u < lift)
        if not m.any():
            continue
        uu = u[m]
        shape = np.where(uu < up, smooth(uu / up), 1.0 - ((uu - up) / down) ** 2)
        out[m] = np.maximum(out[m], a * shape)
    return out


def startles(ts, times, deg=4.5, tau=0.35):
    """Backward jolt (rad) at given times: fast rise, exponential settle."""
    ts = np.asarray(ts, float)
    out = np.zeros(len(ts))
    for s in times:
        m = ts >= s - 0.06
        out[m] += np.exp(-np.maximum(ts[m] - s, 0.0) / tau) * smooth((ts[m] - s) / 0.06 + 1.0)
    return math.radians(deg) * out


def drift(n, seed=11, tau_frames=40, scale=(0.02, 0.01, 0.015)):
    """Slow random wander (n, 3) (m) for gaze targets: attention is never perfectly still."""
    rng = np.random.default_rng(seed)
    return lowpass(rng.normal(0, 1, (n, 3)), tau_frames) * np.asarray(scale)


def keys_from_points(points, fps, frame0):
    """[(t, v)] -> [(frame, v)]."""
    return [(frame0 + t * fps, v) for t, v in points]


HEAD_LIMITS = {"yaw": 75.0, "up": 35.0, "down": 45.0}      # deg: head + neck together, comfortable human range


def yaw_elevation(f, u, d):
    """Angles (rad) that take the head's forward f (unit, perpendicular to its up u) to look along d: yaw about u
    (positive toward u x f, the character's left for a -Y facing, Z up model) and elevation above the plane normal to
    u (positive up)."""
    f, u, d = (np.asarray(v, float) for v in (f, u, d))
    d = d / (np.linalg.norm(d) + 1e-12)
    dh = d - d.dot(u) * u
    yaw = math.atan2(float(np.cross(f, dh).dot(u)), float(f.dot(dh))) if np.linalg.norm(dh) > 1e-9 else 0.0
    elev = math.asin(float(np.clip(d.dot(u), -1.0, 1.0)))
    return yaw, elev


def clamp_look(f, u, d, share, limits=None):
    """Where the head should aim for a look along d: None when `share` of the turn from its forward f (unit,
    perpendicular to its up u) stays inside the head's range {yaw, up, down} (deg; defaults HEAD_LIMITS), so the
    look is used as it is; otherwise the unit direction with yaw and elevation clipped so that it does. The head
    still turns by the minimal rotation toward it; the eyes take what is left, up to their own maximum."""
    if share <= 0:
        return None
    lim = dict(HEAD_LIMITS, **(limits or {}))
    yaw, elev = yaw_elevation(f, u, d)
    y_max = math.radians(lim["yaw"]) / share
    e_lo, e_hi = -math.radians(lim["down"]) / share, math.radians(lim["up"]) / share
    if abs(yaw) <= y_max and e_lo <= elev <= e_hi:
        return None
    y, e = float(np.clip(yaw, -y_max, y_max)), float(np.clip(elev, e_lo, e_hi))
    f, u = np.asarray(f, float), np.asarray(u, float)
    out = math.cos(e) * (math.cos(y) * f + math.sin(y) * np.cross(u, f)) + math.sin(e) * u
    return tuple(out / np.linalg.norm(out))


# ------------------------------------------------------------------------------ the whole body: turning round, airborne
TURN_LEAD = 0.05                 # s: the chest turns this far ahead of the hips, and unwinds as they arrive ...
CHEST_MAX = math.radians(30.0)   # ... by at most about this much (the spine's twist)
TURN_LIFT = 0.045                # m: a stepping foot's lift halfway through its step
TURN_DIP = 0.02                  # m: the hips sink this much while the feet step (the knees give)
TURN_STEPS = ((0.0, 0.58), (0.42, 1.0))     # shares of a turn's span: the first foot's step, then the other's ...
ROLL = 0.15                      # ... each rolling off and setting down, unmoved, over this share of its step: one foot
#                                  is always down, and a foot down never slides
TWIST_MAX = math.radians(35.0)   # a foot turns at most this far from the hips: farther, it swivels on its ankle
TURN_KEYS = ("t", "deg", "dur", "first", "lift", "dip", "move")
AIR = {"tuck": 0.24, "back": 0.10, "first": "L", "share": 0.7, "lag": 0.06, "rise": 0.14, "extend": 0.22, "point": 30.0}
AIR_KEYS = ("from", "to") + tuple(AIR)


def turn_dur(deg):
    """Seconds a brisk turn on the spot takes: 0.2 s and 0.11 s per 100 degrees (145 degrees: 0.36 s)."""
    return 0.2 + 0.0011 * abs(float(deg))


def _side(v, where):
    if v not in ("L", "R"):
        raise ValueError(f"{where}: \"L\" or \"R\", not {v!r}")
    return v


def turn_plan(ts, turns, feet):
    """Turning round on the spot as a person does, not as a statue on a turntable. `turns` [{t, deg, dur, first, lift,
    dip, move}]: from `t` the body turns `deg` degrees toward its left (negative: its right) over `dur` seconds (default
    `turn_dur`), and travels `move` [x, y] metres meanwhile (in the character's frame before its first turn: x its left, y
    behind). The chest leads the hips by TURN_LEAD (at most about CHEST_MAX ahead) and unwinds as they arrive; the foot
    `first` steps over the first TURN_STEPS share of the span and the other foot over the second (by default the order
    that keeps the feet further apart: turning about a point in front of the ankles or behind them, one order sets a foot
    down where the other still stands), each rolling off and setting down over ROLL of its step and lifted `lift` metres
    (TURN_LIFT) between, set down where the turned (and moved) stance puts it; a foot never turns more than TWIST_MAX from
    the hips (planted, it swivels on its ankle); the hips sink `dip` metres (TURN_DIP) meanwhile. `feet` {side: (x, y)}
    are the ankles' floor positions relative to the point the body turns about, in that frame. Returns {"yaw": the hips'
    turn (F,) rad, "shift": the point's travel (F, 2) m, "chest": the chest's lead on top of the turn (F,) rad, "dip": (F,)
    m, "feet": {side: (F, 2) floor positions}, "lift": {side: (F,) m}, "foot_yaw": {side: (F,) rad}, "gap": the least
    distance between the ankles during the turns, m (inf with one foot)}, positions and foot yaws in that frame. Raises
    ValueError for a bad or overlapping turn."""
    ts = np.asarray(ts, float)
    n = len(ts)
    out = {"yaw": np.zeros(n), "shift": np.zeros((n, 2)), "chest": np.zeros(n), "dip": np.zeros(n), "feet": {}, "lift": {},
           "foot_yaw": {}, "gap": math.inf}
    state = {}                                                   # side: [offset in the body's frame, yaw, neutral yaw]
    for s, (x, y) in feet.items():
        _side(s, "turn feet")
        state[s] = [np.array([float(x), float(y)]), 0.0, 0.0]
        out["feet"][s] = np.tile([float(x), float(y)], (n, 1))
        out["lift"][s], out["foot_yaw"][s] = np.zeros(n), np.zeros(n)
    body, shift, end = 0.0, np.zeros(2), -math.inf
    for i, tr in enumerate(sorted(turns, key=lambda tr: float(tr["t"]) if isinstance(tr, dict) and "t" in tr else 0.0)):
        where = f"turn {i}"
        if not isinstance(tr, dict) or "t" not in tr or "deg" not in tr or set(tr) - set(TURN_KEYS):
            raise ValueError(f"{where}: expected {{{', '.join(TURN_KEYS)}}} with t and deg")
        t0, th = float(tr["t"]), math.radians(float(tr["deg"]))
        dur = float(tr.get("dur", turn_dur(tr["deg"])))
        lift, dip = float(tr.get("lift", TURN_LIFT)), float(tr.get("dip", TURN_DIP))
        move = np.asarray(tr.get("move", (0.0, 0.0)), float)
        if dur <= 0 or lift < 0 or dip < 0 or move.shape != (2,):
            raise ValueError(f"{where}: dur must be positive, lift and dip not negative, move [x, y] metres")
        if t0 < end - 1e-6:
            raise ValueError(f"{where} at {t0:g} s starts before the turn before it ends ({end:g} s)")
        end = t0 + dur
        lead = min(TURN_LEAD, dur / 3.0)
        m = ts >= t0
        tm = ts[m]
        go = smooth((tm - t0 - lead) / (dur - lead))
        hips = body + th * go
        out["yaw"][m] = hips
        out["shift"][m] = shift + go[:, None] * move
        ahead = th * (smooth((tm - t0) / (dur - lead)) - go)
        out["chest"][m] = CHEST_MAX * np.tanh(ahead / CHEST_MAX)
        out["dip"] += dip * np.sin(np.pi * np.clip((ts - t0) / dur, 0.0, 1.0))

        def steps(first):
            """{side: (positions, lift, yaw)} over tm with the foot `first` stepping first."""
            got = {}
            for s, (local, yaw0, neutral) in state.items():
                k0, k1 = TURN_STEPS[0] if s == first else TURN_STEPS[1]
                s0, s1 = t0 + k0 * dur, t0 + k1 * dur
                u = np.clip((tm - s0) / (s1 - s0), 0.0, 1.0)
                w = smooth(np.clip((u - ROLL) / (1.0 - 2.0 * ROLL), 0.0, 1.0))      # its way, off the ground
                ang = body + th * w
                c, sn = np.cos(ang), np.sin(ang)
                pos = shift + w[:, None] * move + np.stack([c * local[0] - sn * local[1], sn * local[0] + c * local[1]], 1)
                want = np.where(tm < s0, yaw0, yaw0 + (body + th + neutral - yaw0) * w)
                got[s] = (pos, lift * np.where((w > 0.0) & (w < 1.0), np.sin(np.pi * w), 0.0),
                          np.clip(want, hips + neutral - TWIST_MAX, hips + neutral + TWIST_MAX))
            return got

        def gap(got):
            return float(np.linalg.norm(got["L"][0] - got["R"][0], axis=1).min()) if len(got) == 2 and len(tm) else math.inf

        if "first" in tr:
            got = steps(_side(tr["first"], f"{where} first"))
        else:                                                    # the order that keeps the feet furthest apart
            got = max((steps(s) for s in ("L", "R")), key=gap)
        out["gap"] = min(out["gap"], gap(got))
        for s, (pos, up, yaw) in got.items():
            out["feet"][s][m], out["lift"][s][m], out["foot_yaw"][s][m] = pos, up, yaw
            state[s][1] = body + th + state[s][2]
        body, shift = body + th, shift + move
    return out


def air_plan(ts, spans):
    """The legs of a body in the air (a jump, a blast): each span {from, to, tuck, back, first, share, lag, rise, extend,
    point} runs from take-off `from` to landing `to` (clip seconds). Over `rise` s the feet come up toward the hips by
    `tuck` metres and back by `back` (the knees bend and come forward), the foot `first` ahead and the other `lag` s later,
    by `share` of it; the toes point down by `point` degrees; over the last `extend` s the legs reach down again, so the
    feet are back under the hips at the landing. Defaults AIR. Returns ({side: (F, 3) offsets (x its left, y behind, z
    up) m}, {side: (F,) the toes' pitch down, rad}); overlapping spans take the larger. Raises ValueError for a bad span."""
    ts = np.asarray(ts, float)
    off = {s: np.zeros((len(ts), 3)) for s in ("L", "R")}
    pitch = {s: np.zeros(len(ts)) for s in ("L", "R")}
    for i, sp in enumerate(spans):
        where = f"air {i}"
        if not isinstance(sp, dict) or "from" not in sp or "to" not in sp or set(sp) - set(AIR_KEYS):
            raise ValueError(f"{where}: expected {{{', '.join(AIR_KEYS)}}} with from and to")
        c = dict(AIR, **{k: v for k, v in sp.items() if k in AIR})
        t0, t1 = float(sp["from"]), float(sp["to"])
        first = _side(c["first"], f"{where} first")
        rise, extend, share, lag = (float(c[k]) for k in ("rise", "extend", "share", "lag"))
        if t1 <= t0 or rise <= 0 or extend <= 0 or lag < 0 or not 0.0 <= share <= 1.0:
            raise ValueError(f"{where}: from before to, rise and extend positive, lag not negative, share within 0..1")
        down = 1.0 - smooth((ts - (t1 - extend)) / extend)
        for s in ("L", "R"):
            k, late = (1.0, 0.0) if s == first else (share, lag)
            e = smooth((ts - t0 - late) / rise) * down * k
            v = np.stack([np.zeros(len(ts)), float(c["back"]) * e, float(c["tuck"]) * e], 1)
            off[s] = np.where(np.abs(v) > np.abs(off[s]), v, off[s])
            pitch[s] = np.maximum(pitch[s], math.radians(float(c["point"])) * e)
    return off, pitch


# ------------------------------------------------------------------------------ walking, hopping, tumbling
WALK = {"stride": 0.55, "first": "L", "lift": 0.07, "dip": 0.015, "slide": 0.0, "roll": 0.15, "hop": 0.0, "arms": 0.0,
        "cross": 0.0, "sway": 0.0}
WALK_KEYS = ("steps", "hops") + tuple(WALK)
HOP_KEYS = ("from", "to", "move", "height")
TUMBLE_KEYS = ("from", "to", "deg", "axis")
TUMBLE_AXES = ("flip", "cartwheel")       # about the character's left axis (heels over head) or its forward axis
GRAVITY = 9.81
FIRST_SWING = 0.35                        # s: the swing of a step from a stand (the first, or one after a pause)
WALK_PAUSE = 1.2                          # s: a step longer than this after the last (or a landing) is from a stand
ARMS_SETTLE = 0.25                        # s: the arms' swing dies away over this once the feet have stopped


def _hops(spec):
    out = []
    for i, h in enumerate(spec.get("hops") or []):
        if not isinstance(h, dict) or "from" not in h or "to" not in h or set(h) - set(HOP_KEYS):
            raise ValueError(f"hops[{i}]: expected {{{', '.join(HOP_KEYS)}}} with from and to")
        t0, t1 = float(h["from"]), float(h["to"])
        move = h.get("move")
        if t1 <= t0 or (move is not None and np.asarray(move, float).shape != (2,)):
            raise ValueError(f"hops[{i}]: from before to, move [x, y] metres (its left, behind)")
        out.append({"from": t0, "to": t1, "move": None if move is None else np.asarray(move, float),
                    "height": h.get("height")})
    return sorted(out, key=lambda h: h["from"])


def walk_plan(ts, spec, feet):
    """Walking, a step on each of `steps` (clip seconds: when each foot comes down, alternating from `first`), and
    hopping, both feet off the ground over each of `hops` [{from, to, move, height}]. In the character's frame (x its
    left, y behind, so forward is -y): a step sets its foot down `stride` m ahead of the other foot, on its own side of
    the body (where `feet` {side: (x, y)}, the ankles' floor positions at rest relative to the root, put it) or, with
    `cross`, that share of the way in to the body's centre line (1: on it, a catwalk; up to 1.5: across it); the foot
    rolls off the ground `roll` of the step's time after the other foot came down, swings through `lift` m high and comes
    down on the step; the body moves straight, at a steady pace, so that it is midway between the feet each time one
    comes down, and the hips dip `dip` m as it does. A step more than WALK_PAUSE s after the one before starts from a
    stand, as the first does: the body waits where it stood and the foot swings over the last FIRST_SWING s, so a pose
    between two walks holds still. `sway` (m) carries the hips over the foot that is down, out and back
    while the other one swings (the strut's hips). A hop lifts the body off its feet from `from` to `to` on a
    parabola `height` m high (default: what the flight time gives under gravity, g T^2 / 8; `hop` sets one height for
    every hop) and carries it `move` [x, y] (default: a stride forward); the feet land a quarter stride either side of
    the body, the one that was ahead ahead, and the next step goes on from there. `slide` (0..1) puts the walk on a
    treadmill: the body goes 1 - slide of the way, the feet down slide back under it (1: the body stays, a running man).
    `arms` (m) swings each hand forward as the other side's foot goes forward of where it stood, back as it goes
    behind, while the feet move; it dies away over ARMS_SETTLE s once they stop, so a hand posed between two walks is
    where its pose put it ("swing": {side: (F,)} the hand's forward offset; 0: none).
    Returns {"shift": the root's travel (F, 2), "height": (F,), "feet": {side: (F, 2) the foot's floor position relative
    to the root}, "lift": {side: (F,)}, "dip": (F,), "sway": (F,) the hips' offset toward the left, "air": [(from, to)]
    the hops' flights, "swing", "steps", "hops"}. Raises ValueError for steps out of order or inside a hop."""
    ts = np.asarray(ts, float)
    o = dict(WALK, **{k: v for k, v in spec.items() if k in WALK})
    bad = sorted(set(spec) - set(WALK_KEYS))
    if bad:
        raise ValueError(f"unknown key(s) {bad} (known: {', '.join(WALK_KEYS)})")
    first = _side(o["first"], "walk first")
    stride, lift, dip, slide, roll = (float(o[k]) for k in ("stride", "lift", "dip", "slide", "roll"))
    cross, sway = float(o["cross"]), float(o["sway"])
    if stride < 0 or lift < 0 or dip < 0 or not 0.0 <= slide <= 1.0 or not 0.0 <= roll < 1.0:
        raise ValueError("stride, lift and dip not negative, slide within 0..1, roll within 0..1")
    if not 0.0 <= cross <= 1.5 or sway < 0:
        raise ValueError("cross within 0..1.5 (1: the feet on the centre line), sway not negative")
    steps = sorted(float(t) for t in (spec.get("steps") or []))
    if any(b - a < 1e-6 for a, b in zip(steps, steps[1:])):
        raise ValueError("two steps at one time: each step is one foot coming down")
    hops = _hops(spec)
    for h in hops:
        if any(h["from"] < t < h["to"] for t in steps):
            raise ValueError(f"a step inside the hop {h['from']:g}..{h['to']:g} s: the feet are off the ground then")
    for a, b in zip(hops, hops[1:]):
        if b["from"] < a["to"]:
            raise ValueError(f"the hop at {b['from']:g} s starts before the hop before it lands ({a['to']:g} s)")
    fwd, lat = np.array([0.0, -1.0]), {s: np.array([float(feet[s][0]), 0.0]) for s in feet}
    pos = {s: np.array([float(feet[s][0]), float(feet[s][1])]) for s in feet}
    body = np.zeros(2)
    keys = []                                        # (t, body position) of the body's path
    swings = {s: [] for s in feet}                   # (t0, t1, from, to, arc)
    dips, air, stance = [], [], []                   # stance: (t0, t1, side) a foot carries the body while the other swings
    events = [("step", t, None) for t in steps] + [("hop", h["from"], h) for h in hops]
    events.sort(key=lambda e: e[1])
    side, last = first, None
    for kind, t, h in events:
        if kind == "step":
            if side not in pos:                      # one leg only: it steps every time
                side = next(iter(pos))
            other = next((s for s in pos if s != side), side)
            stood = last is None or t - last > WALK_PAUSE    # from a stand: the first step, or one after a pause
            t0 = t - FIRST_SWING if stood else last + roll * (t - last)
            if stood:
                keys.append((t0, body.copy()))       # the body waits where it stood until the foot lifts
            ahead = max(float(pos[s] @ fwd) for s in pos)
            new = np.array([body[0], 0.0]) + lat[side] * (1.0 - cross) + fwd * (ahead + stride)   # a stride past the foot ahead
            stance.append((t0 if stood else last, t, other))
            swings[side].append((t0, t, pos[side].copy(), new, True))
            pos[side] = new
            body = np.array([body[0], 0.5 * (pos[side][1] + pos[other][1])])         # midway between the feet
            keys.append((t, body.copy()))
            dips.append(t)
            last, side = t, other
        else:
            t0, t1 = h["from"], h["to"]
            keys.append((t0, body.copy()))
            move = h["move"] if h["move"] is not None else fwd * stride
            body = body + move
            front = max(pos, key=lambda s: float(pos[s] @ fwd))
            for s in pos:
                sign = 1.0 if s == front or len(pos) == 1 else -1.0
                new = body + lat[s] + fwd * (0.25 * stride * sign)
                swings[s].append((t0, t1, pos[s].copy(), new, False))
                pos[s] = new
            keys.append((t1, body.copy()))
            air.append((t0, t1, h["height"]))
            last = t1
            side = next((s for s in pos if s != front), front)       # the foot behind steps next
    n = len(ts)
    if keys:
        kt = np.array([k[0] for k in keys])
        kp = np.array([k[1] for k in keys])
        path = np.stack([np.interp(ts, kt, kp[:, 0]), np.interp(ts, kt, kp[:, 1])], 1)
    else:
        path = np.zeros((n, 2))
    height = np.zeros(n)
    for t0, t1, hh in air:
        T = t1 - t0
        H = float(hh) if hh is not None else (float(o["hop"]) or GRAVITY * T * T / 8.0)
        u = np.clip((ts - t0) / T, 0.0, 1.0)
        height = np.maximum(height, 4.0 * H * u * (1.0 - u))
    out_feet, out_lift = {}, {}
    for s in pos:
        p = np.tile(np.array([float(feet[s][0]), float(feet[s][1])]), (n, 1))
        up = np.zeros(n)
        for t0, t1, a, b, arc in swings[s]:
            u = np.clip((ts - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
            w = smooth(u)
            moving = ts >= t0
            p[moving] = a + (b - a) * w[moving, None]
            if arc:
                up = np.maximum(up, lift * np.sin(np.pi * u) * ((u > 0) & (u < 1)))
        out_feet[s] = p - path                                    # relative to the body, whatever the treadmill does
        out_lift[s] = up
    sag = np.zeros(n)
    for t in dips:
        sag = np.maximum(sag, dip * np.exp(-((ts - t) / 0.08) ** 2))
    hips = np.zeros(n)                                            # over the foot that is down, out and back
    if sway > 0:
        for t0, t1, s in stance:
            u = (ts - t0) / max(t1 - t0, 1e-6)
            inside = (u > 0.0) & (u < 1.0)
            hips[inside] = sway * math.copysign(1.0, lat[s][0] or 1.0) * np.sin(np.pi * u[inside])
    runs = []                                                     # the walks: swings less than two settles apart run on
    for t0, t1 in sorted((a, b) for s in swings for a, b, *_ in swings[s]):
        if runs and t0 <= runs[-1][1] + 2.0 * ARMS_SETTLE:
            runs[-1][1] = max(runs[-1][1], t1)
        else:
            runs.append([t0, t1])
    moving = np.zeros(n)                                          # the arms swing over a walk, coming and going smoothly
    for t0, t1 in runs:
        moving = np.maximum(moving, smooth((ts - t0) / ARMS_SETTLE) * (1.0 - smooth((ts - t1) / ARMS_SETTLE)))
    swing = {}                                                    # each hand goes forward as the other side's foot does
    arms = float(o["arms"])
    for s in out_feet:
        other = next((x for x in out_feet if x != s), s)
        ahead = -(out_feet[other][:, 1] - float(feet[other][1]))  # that foot's forward offset from where it stood
        swing[s] = (arms * moving * np.clip(ahead / max(0.5 * stride, 1e-6), -1.0, 1.0) if arms > 0
                    else np.zeros(n))
    return {"shift": (1.0 - slide) * path, "height": height, "feet": out_feet, "lift": out_lift, "dip": sag,
            "sway": hips, "air": [(a, b) for a, b, _ in air], "swing": swing, "steps": len(steps), "hops": len(hops)}


def tumble_plan(ts, tumbles):
    """Turns of the whole body in the air about its hips: each {from, to, deg, axis} turns it `deg` degrees (360: heels
    over head and round) over from..to, eased in and out, about its left axis (`flip`; positive: head forward) or its
    forward axis (`cartwheel`; positive: toward its left), the turns adding up and holding after. Returns {axis: (F,) rad}.
    Raises ValueError for a bad or overlapping tumble."""
    ts = np.asarray(ts, float)
    out = {a: np.zeros(len(ts)) for a in TUMBLE_AXES}
    end = {a: -math.inf for a in TUMBLE_AXES}
    for i, tb in enumerate(sorted(tumbles, key=lambda x: float(x["from"]) if isinstance(x, dict) and "from" in x else 0)):
        if not isinstance(tb, dict) or not {"from", "to", "deg"} <= set(tb) or set(tb) - set(TUMBLE_KEYS):
            raise ValueError(f"tumble {i}: expected {{{', '.join(TUMBLE_KEYS)}}} with from, to and deg")
        axis = tb.get("axis", "flip")
        if axis not in TUMBLE_AXES:
            raise ValueError(f"tumble {i}: axis is \"flip\" or \"cartwheel\", not {axis!r}")
        t0, t1 = float(tb["from"]), float(tb["to"])
        if t1 <= t0:
            raise ValueError(f"tumble {i}: from before to")
        if t0 < end[axis] - 1e-6:
            raise ValueError(f"tumble {i} at {t0:g} s starts before the {axis} before it ends ({end[axis]:g} s)")
        end[axis] = t1
        out[axis] += math.radians(float(tb["deg"])) * smooth((ts - t0) / (t1 - t0))
    return out
