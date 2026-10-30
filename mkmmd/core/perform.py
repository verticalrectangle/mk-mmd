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


SQUASH = {"depth": 0.1, "stretch": 0.05, "attack": 0.04, "rebound": 0.14}
SQUASH_KEYS = ("lane", "min_db", "every", "from", "to", "hits") + tuple(SQUASH)


def squash(ts, hits, depth=0.1, stretch=0.05, attack=0.04, rebound=0.14):
    """A cartoon squash and stretch of a whole body on each of `hits` (clip seconds): (F,) the scale of its height over
    `ts`, 1 at rest (the body keeps its volume: its width and depth scale by 1 / sqrt of it). Over `attack` s it squashes
    to 1 - `depth` on the hit, then springs back up past 1 to 1 + `stretch` `rebound` s after the hit and settles: a
    damped spring let go at rest at the squash (`depth` e^-u/tau (cos wu + sin wu / (w tau)), w = pi / rebound, tau =
    rebound / ln(depth / stretch)). The hits' springs add up where they overlap."""
    ts = np.asarray(ts, float)
    depth, stretch, attack, rebound = float(depth), float(stretch), float(attack), float(rebound)
    if not 0.0 < depth < 0.6 or not 0.0 < stretch < depth or attack <= 0.0 or rebound <= 0.0:
        raise ValueError("depth within 0..0.6, stretch above 0 and under depth, attack and rebound above 0")
    w = math.pi / rebound
    tau = rebound / math.log(depth / stretch)
    out = np.zeros(len(ts))
    for h in hits:
        u = ts - float(h)
        before = (u > -attack) & (u < 0.0)
        out[before] -= depth * smooth((u[before] + attack) / attack)
        after = (u >= 0.0) & (u < 8.0 * tau)
        ua = u[after]
        out[after] -= depth * np.exp(-ua / tau) * (np.cos(w * ua) + np.sin(w * ua) / (w * tau))
    return 1.0 + out


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
AIR_KEYS = ("from", "to") + tuple(AIR) + ("kicks",)
KICKS = {"knee": 10.0, "hip": 2.0, "foot": 15.0, "period": 3.0, "every": 0, "wander": 0.3, "seed": 0, "ease": 1.0}
KICKS_KEYS = tuple(KICKS) + ("beats", "timeline")
KICK_HIP_LEAD = 0.15             # cycles: a kicking thigh swings ahead of its knee (a wave down the leg, not a piston) ...
KICK_FOOT_LAG = 0.1              # ... and the foot points after the knee has straightened, trailing it
KICK_DRIFT = 0.2                 # cycles per unit of `wander`: how far the legs drift from taking turns exactly
WANDER_MAX = 0.6                 # more, and a leg drifting from its turn could kick backward for a moment
KICK_KNEE_MIN = math.radians(5.0)   # the straightest a kicking knee gets: from a straight tuck the kicks bend it first
WOBBLE = ((1.618, 0.5), (2.718, 0.3), (4.236, 0.2))   # (period in leg cycles, weight): the wander's slow sines
SPIN = {"spot": True, "windup": 20.0, "overshoot": 8.0, "twist": 35.0, "neck": 70.0, "whip": 0.3, "lift": 0.1}
SPIN_KEYS = ("from", "to", "deg", "foot") + tuple(SPIN)
SPIN_TWIST_MAX = 50.0            # deg: the most `twist` may be (a spine twists 60 at most: joint_limits' spine_twist) ...
SPIN_NECK_MAX = 80.0             # ... and `neck` (85: its neck_twist; the gaze's head_limits yaw, 75, clips it first)
SPIN_LAG = 0.12                  # s: the hips trail the chest by this much (at most 8 % of the spin)
SPIN_EASE = (0.45, 0.2)          # s, share of the spin: the wind-up and the settle back each take the shorter of these
SPIN_WHIP = 0.25                 # the head's whip takes at most this share of the turn
SPIN_SOFT = math.radians(20.0)   # the head eases into the end of the neck's range, and onto a spot, over this much
SPIN_OWN = 0.25                  # s: a spin takes the head from the gaze over this much before it and gives it back after
SPIN_LIFT = 0.15                 # s: on the ground the other foot lifts off, and the heel, over this much as the hips go
SPIN_LAND = 0.2                  # s: and they come down over the last this much, once the body has settled
RELEVE = math.radians(20.0)      # on the ground the foot turned on rises onto its toes: the heel this far up
SPIN_TUCK = 0.5                  # the other foot comes this share of the way in toward the one turned on
EYE_LEAD = 0.07                  # s: the eyes lead the head, as they lead a gaze
ALOFT = 0.02                     # m: a body `rise` lifts more than this is off the ground


def aloft(ts, air=None, rise=None):
    """Where the body is off the ground, (F,) bools over `ts`: inside an `air` span (a table {from, to} or a list of them)
    or lifted more than ALOFT by the `rise` keys."""
    ts = np.asarray(ts, float)
    up = np.zeros(len(ts), bool)
    for sp in air if isinstance(air, list) else [air] if air else []:
        if isinstance(sp, dict) and "from" in sp and "to" in sp:
            up |= (ts >= float(sp["from"])) & (ts <= float(sp["to"]))
    if rise:
        up |= eased_keys(ts, rise) > ALOFT
    return up


def eased_rate(ts, keys):
    """The rate of `eased_keys` (per second): zero before the first key, after the last and on every key."""
    ts = np.asarray(ts, float)
    out = np.zeros(len(ts))
    ks = sorted((float(t), float(v)) for t, v in keys)
    for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
        m = (ts >= t0) & (ts < t1)
        x = (ts[m] - t0) / max(t1 - t0, 1e-6)
        out[m] = (v1 - v0) * 6.0 * x * (1.0 - x) / max(t1 - t0, 1e-6)
    return out


def _smax(a, b, k):
    """The larger of a and b, eased over a band k wide where they meet (exact outside it, never below either)."""
    return np.maximum(a, b) + np.maximum(k - np.abs(a - b), 0.0) ** 2 / (4.0 * k)


def _smin(a, b, k):
    return -_smax(-a, -b, k)


def spin_curves(ts, sp, ground=False, where="spin"):
    """One spin {from, to, deg, spot, windup, overshoot, twist, neck, whip} (defaults SPIN) of a dancer, not of a figure
    on a turntable: the turns of her hips, chest, head and eyes about the vertical, relative to where she faced before it,
    rad, + toward her left. From `from` she winds up `windup` degrees the other way, turns `deg` and `overshoot` degrees
    past it, and settles back by `to` (by SPIN_LAND earlier on the `ground`, so the feet come down on a body at rest),
    each eased (SPIN_EASE); the chest leads and the hips trail it by SPIN_LAG, so the spine twists with the speed, by
    `twist` degrees at most. With `spot` the head holds where she faced (a dancer's spot: the camera in front of her) while
    the neck lets it, `neck` degrees at most, then whips round in `whip` s (SPIN_WHIP of the turn at most) ahead of the
    chest, as far ahead as the neck lets it, onto the next spot (every full turn, then where she ends facing) and holds
    it as the chest comes round under it; without, it leads the chest by the speed, `neck` at most. The eyes lead the
    head by EYE_LEAD. Returns {"hips", "chest", "head", "eye": (F,) rad, "phases": {"windup", "turn", "settle": the
    hips' (t, t), "whips": [t, ...], "land": t}}. Raises ValueError for a bad spin."""
    ts = np.asarray(ts, float)
    t0, t1 = _spin_span(sp, where)
    c = dict(SPIN, **{k: v for k, v in sp.items() if k in SPIN})
    deg = float(sp["deg"])
    windup, over, twist, neck, whip = (float(c[k]) for k in ("windup", "overshoot", "twist", "neck", "whip"))
    if not isinstance(c["spot"], bool):
        raise ValueError(f"{where}: spot is true or false, not {c['spot']!r}")
    if deg == 0 or windup < 0 or over < 0 or whip <= 0 or not 0 < twist <= SPIN_TWIST_MAX or not 0 < neck <= SPIN_NECK_MAX:
        raise ValueError(f"{where}: deg not 0, windup and overshoot not negative, whip positive, twist within "
                         f"0..{SPIN_TWIST_MAX:g} and neck within 0..{SPIN_NECK_MAX:g} degrees")
    lag = min(SPIN_LAG, 0.08 * (t1 - t0))
    end = t1 - lag - (SPIN_LAND if ground else 0.0)              # the chest has settled; the hips a lag later
    ease = min(SPIN_EASE[0], SPIN_EASE[1] * (end - t0))
    up, back = (ease if windup else 0.0), (ease if over else 0.0)
    if end - t0 - up - back < 0.2:
        raise ValueError(f"{where}: {t1 - t0:g} s is too short a spin (from before to, the turn itself 0.2 s at least)")
    whip = min(whip, SPIN_WHIP * (end - t0 - up - back))
    sign, D = math.copysign(1.0, deg), math.radians(abs(deg))
    keys = [(t0, 0.0)] + ([(t0 + up, -math.radians(windup))] if up else []) + [(end - back, D + math.radians(over))]
    keys += [(end, D)] if back else []
    T, N = math.radians(twist), math.radians(neck)

    def chest(t):
        return eased_keys(t - lag, keys) + T * np.tanh((eased_keys(t, keys) - eased_keys(t - lag, keys)) / T)

    spots = [2.0 * math.pi * k for k in range(int(math.floor(D / (2.0 * math.pi) + 1e-9)) + 1)]
    spots += [D] if D - spots[-1] > 1e-6 else []
    whips, after = [], t0
    if c["spot"]:
        tf = np.arange(t0, t1, 0.002)
        cf = chest(tf)
        for k in range(len(spots) - 1):                          # the head whips as the neck runs out of turn
            hold = min(N - 0.5 * SPIN_SOFT, 0.5 * (spots[k + 1] - spots[k]))
            past = np.nonzero((cf - spots[k] >= hold) & (tf >= after))[0]
            whips.append(float(tf[past[0]]) if len(past) else max(after, end - back - whip))
            after = whips[-1] + whip

    def head(t):
        ch = chest(t)
        if not c["spot"]:
            return ch + N * np.tanh(lag * eased_rate(t, keys) / N)

        def on(s):                                               # the spot, as far as the neck lets the head hold it
            return _smin(_smax(np.full(len(t), s), ch - N, SPIN_SOFT), ch + N, SPIN_SOFT)
        out = on(spots[0])
        for k, tw in enumerate(whips):
            out = out + smooth((t - tw) / whip) * (on(spots[k + 1]) - on(spots[k]))
        return out

    return {"hips": sign * eased_keys(ts - lag, keys), "chest": sign * chest(ts), "head": sign * head(ts),
            "eye": sign * head(ts + EYE_LEAD),
            "phases": {"windup": (t0 + lag, t0 + lag + up), "turn": (t0 + lag + up, end + lag - back),
                       "settle": (end + lag - back, end + lag), "whips": whips, "land": t1 - SPIN_LAND}}


def turn_dur(deg):
    """Seconds a brisk turn on the spot takes: 0.2 s and 0.11 s per 100 degrees (145 degrees: 0.36 s)."""
    return 0.2 + 0.0011 * abs(float(deg))


def _side(v, where):
    if v not in ("L", "R"):
        raise ValueError(f"{where}: \"L\" or \"R\", not {v!r}")
    return v


def _rot(ang, v):
    """Floor vectors v (2,) or (k, 2) turned by `ang` (rad or (k,) rad, + toward the left: x left, y behind)."""
    c, s, v = np.cos(ang), np.sin(ang), np.asarray(v, float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def turn_plan(ts, turns, feet, spins=(), aloft=None, toes=None, legs=None):
    """Turning round as a person does, not as a statue on a turntable: stepping round on the spot (`turns`) and spinning
    (`spins`), one after the other in time order. `turns` [{t, deg, dur, first, lift, dip, move}]: from `t` the body
    turns `deg` degrees toward its left (negative: its right) over `dur` seconds (default `turn_dur`), and travels `move`
    [x, y] metres meanwhile (in the character's frame before its first turn: x its left, y behind). The chest leads the
    hips by TURN_LEAD (at most about CHEST_MAX ahead) and unwinds as they arrive; the foot `first` steps over the first
    TURN_STEPS share of the span and the other foot over the second (by default the order that keeps the feet further
    apart: turning about a point in front of the ankles or behind them, one order sets a foot down where the other still
    stands), each rolling off and setting down over ROLL of its step and lifted `lift` metres (TURN_LIFT) between, set
    down where the turned (and moved) stance puts it; a foot never turns more than TWIST_MAX from the hips (planted, it
    swivels on its ankle); the hips sink `dip` metres (TURN_DIP) meanwhile. `feet` {side: (x, y)} are the ankles' floor
    positions relative to the point the body turns about, in that frame.
    `spins` [{from, to, deg, spot, windup, overshoot, twist, neck, whip, foot, lift}] turn the hips, chest, head and eyes
    as `spin_curves` does. One all in the air (`aloft`, (F,) bools: where the body is off the ground) turns about the
    point turned about, the legs going round with the hips. One on the ground turns about the toe point of the foot `foot`
    (default the side it turns toward), `toes` {side: (x, y, up)} its floor offset from the ankle in that frame and the
    ankle's height over it (without, about the ankle): the weight goes over that foot as she winds up, the other foot
    stays down until the hips set off and then lifts `lift` m (SPIN["lift"]), drawn SPIN_TUCK of the way in and under the
    hips, while the foot turned on rises onto its toes, the heel RELEVE up about them (the body rising with it and the
    hips coming over them, sinking as far as `legs` {side: {"hip": the hip joint from the ankle as she stands}} need for
    that leg to stay no straighter than it stands), and turns on them with the hips; both come down where the turned
    stance puts them over the last SPIN_LAND s, the body at rest.
    Returns {"yaw": the hips' turn (F,) rad, "shift": the point's travel (F, 2) m, "chest": the chest's lead on top of the
    turn (F,) rad, "dip": (F,) m, "feet": {side: (F, 2) floor positions}, "lift": {side: (F,) m}, "foot_yaw": {side:
    (F,) rad}, "pitch": {side: (F,) the heel's lift about the toe point, rad}, "hips": the hips' shift (F, 2) m (x its
    left, y behind), "look", "eye": the head's and the eyes' turn from the hips (F,) rad, "own": how much a spin has the
    head (F,) 0..1, "gap": the least distance between the ankles during the step turns, m (inf with one foot), "spins": [a
    report per spin]}, positions and foot yaws in that frame. Raises ValueError for a bad or overlapping turn."""
    ts = np.asarray(ts, float)
    n = len(ts)
    out = {"yaw": np.zeros(n), "shift": np.zeros((n, 2)), "chest": np.zeros(n), "dip": np.zeros(n), "feet": {}, "lift": {},
           "foot_yaw": {}, "pitch": {}, "hips": np.zeros((n, 2)), "look": np.zeros(n), "eye": np.zeros(n), "own": np.zeros(n),
           "gap": math.inf, "spins": []}
    state = {}                                                   # side: [offset in the body's frame, yaw, neutral yaw]
    for s, (x, y) in feet.items():
        _side(s, "turn feet")
        state[s] = [np.array([float(x), float(y)]), 0.0, 0.0]
        out["feet"][s] = np.tile([float(x), float(y)], (n, 1))
        out["lift"][s], out["foot_yaw"][s], out["pitch"][s] = np.zeros(n), np.zeros(n), np.zeros(n)
    body, shift, end = 0.0, np.zeros(2), -math.inf
    events = [(tr["t"] if isinstance(tr, dict) and "t" in tr else 0.0, i, tr, None) for i, tr in enumerate(turns)]
    events += [(sp["from"] if isinstance(sp, dict) and "from" in sp else 0.0, i, None, sp) for i, sp in enumerate(spins)]
    for _t, i, tr, sp in sorted(events, key=lambda e: (float(e[0]), e[2] is None, e[1])):
        if sp is not None:
            where = f"spin {i}"
            t0, t1 = _spin_span(sp, where)
            if t0 < end - 1e-6:
                raise ValueError(f"{where} at {t0:g} s starts before the turn before it ends ({end:g} s)")
            end = t1
            body, shift = _spin(ts, sp, where, state, body, shift, out, aloft, toes, legs)
            continue
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


def _spin_span(sp, where):
    """A spin's (from, to), clip seconds, after checking its keys."""
    if not isinstance(sp, dict) or not {"from", "to", "deg"} <= set(sp) or set(sp) - set(SPIN_KEYS):
        raise ValueError(f"{where}: expected {{{', '.join(SPIN_KEYS)}}} with from, to and deg")
    return float(sp["from"]), float(sp["to"])


def _spin(ts, sp, where, state, body, shift, out, aloft, toes, legs):
    """turn_plan's spin `sp`, from the body's turn `body` and the point turned about at `shift` (its feet in `state`):
    keys `out` from its start on and returns the turn and the point after it."""
    t0, t1 = _spin_span(sp, where)
    inside = (ts >= t0) & (ts <= t1)
    up = aloft is not None and bool(inside.any()) and bool(np.asarray(aloft, bool)[inside].all())
    if aloft is not None and not up and bool(np.asarray(aloft, bool)[inside].any()):
        raise ValueError(f"{where} ({t0:g}-{t1:g} s) is partly in the air: a spin turns all in the air or all on the "
                         f"ground")
    cur = spin_curves(ts, sp, ground=not up, where=where)
    deg = math.radians(float(sp["deg"]))
    m = ts >= t0
    tm, hips = ts[m], cur["hips"][m]
    out["yaw"][m] = body + hips
    out["chest"][m] = (cur["chest"] - cur["hips"])[m]
    out["look"][m] = (cur["head"] - cur["hips"])[m]
    out["eye"][m] = (cur["eye"] - cur["hips"])[m]
    out["own"] = np.maximum(out["own"], smooth((ts - t0) / SPIN_OWN + 1.0) * (1.0 - smooth((ts - t1) / SPIN_OWN)))
    info = {"from": t0, "to": t1, "deg": float(sp["deg"]), "air": up, "foot": None,
            "whips": [round(t, 3) for t in cur["phases"]["whips"]],
            "twist_deg": round(float(np.degrees(np.abs(cur["chest"] - cur["hips"]).max())), 1),
            "neck_deg": round(float(np.degrees(np.abs(cur["head"] - cur["chest"]).max())), 1)}
    out["spins"].append(info)
    if up:                                                       # the legs go round with the hips
        if "foot" in sp:
            raise ValueError(f"{where}: foot: she is in the air all through it, with no foot on the ground to turn on")
        for s, (local, _yaw, neutral) in state.items():
            out["feet"][s][m] = shift + _rot(body + hips, local)
            out["foot_yaw"][s][m] = body + hips + neutral
            out["lift"][s][m] = out["pitch"][s][m] = 0.0
            state[s][1] = body + deg + neutral
        return body + deg, shift
    sup = _side(sp.get("foot", "L" if deg > 0 else "R"), f"{where} foot")
    lift = float(sp.get("lift", SPIN["lift"]))
    if sup not in state or lift < 0:
        raise ValueError(f"{where}: on the ground she turns on a foot, here {sup} (its leg IK), and lifts the other "
                         f"`lift` m, not negative")
    info["foot"] = sup
    ph = cur["phases"]
    t_off, t_land = ph["turn"][0], ph["land"]
    local, _yaw, neutral = state[sup]
    tx, ty, rise = (float(v) for v in (toes or {}).get(sup, (0.0, 0.0, 0.0)))
    toe = local + _rot(neutral, (tx, ty))                        # the point she turns on, in the body's frame ...
    pivot = shift + _rot(body, toe)                              # ... and on the floor: it never moves
    root = pivot - _rot(body + hips, toe)
    out["shift"][m] = root
    length = math.hypot(tx, ty)                                  # the toe point to the ankle
    heel = np.array([-tx, -ty]) / length if length > 1e-9 else np.zeros(2)
    rising, coming = smooth((tm - t_off) / SPIN_LIFT), smooth((tm - t_land) / SPIN_LAND)
    pitch = (RELEVE if length > 1e-9 else 0.0) * rising * (1.0 - coming)   # up on its toes
    back = length * np.cos(pitch) - rise * np.sin(pitch)        # the ankle behind the toe point, and over it
    over = length * np.sin(pitch) + rise * np.cos(pitch)
    swivel = hips * rising                                       # flat on the floor until the heel rises, then round
    out["feet"][sup][m] = pivot + _rot(body + neutral + swivel, back[:, None] * heel)
    out["foot_yaw"][sup][m] = body + neutral + swivel
    out["lift"][sup][m], out["pitch"][sup][m] = over - rise, pitch
    out["dip"][m] -= over - rise                                 # the body rises with the heel
    weight = smooth((tm - t0) / max(t_off - t0, 1e-6)) * (1.0 - coming)
    out["hips"][m] += np.stack([toe[0] * weight, toe[1] * rising * (1.0 - coming)], 1)   # over the foot, then its toes
    leg = (legs or {}).get(sup)
    if leg is not None:                                          # that leg no straighter than it stands: the hips sink
        hip = np.asarray(leg["hip"], float)
        ankle = toe + _rot(neutral + swivel - hips, back[:, None] * heel)      # both in the body's frame
        flat = np.linalg.norm(local + hip[:2] + out["hips"][m] - ankle, axis=1)
        out["dip"][m] += np.maximum(hip[2] - np.sqrt(np.maximum(hip @ hip - flat ** 2, 0.0)), 0.0)
    for s, (other, _y, neu) in state.items():
        if s == sup:
            continue
        tuck = other + np.array([SPIN_TUCK * (local[0] - other[0]), toe[1]])     # in under the hips
        u_off, u_on = (tm - t_off) / SPIN_LIFT, (tm - t_land) / SPIN_LAND
        off = smooth((u_off - ROLL) / (1.0 - 2.0 * ROLL))        # its way, off the ground
        on = smooth((u_on - ROLL) / (1.0 - 2.0 * ROLL))
        planted = shift + _rot(body, other)
        carried = root + _rot(body + hips, tuck + on[:, None] * (other - tuck))
        out["feet"][s][m] = planted + off[:, None] * (carried - planted)
        out["foot_yaw"][s][m] = body + neu + off * hips
        out["lift"][s][m], out["pitch"][s][m] = lift * smooth(u_off) * (1.0 - smooth(u_on)), 0.0
    for s in state:
        state[s][1] = body + deg + state[s][2]
    return body + deg, pivot - _rot(body + deg, toe)


def air_plan(ts, spans, legs=None):
    """The legs of a body in the air (a jump, a blast, a float): each span {from, to, tuck, back, first, share, lag, rise,
    extend, point, kicks} runs from take-off `from` to landing `to` (clip seconds). Over `rise` s the feet come up toward
    the hips by `tuck` metres and back by `back` (the knees bend and come forward), the foot `first` ahead and the other
    `lag` s later, by `share` of it; the toes point down by `point` degrees; over the last `extend` s the legs reach down
    again, so the feet are back under the hips at the landing. Defaults AIR. `kicks` (true or a table: `air_kicks`) kicks
    the tucked legs lazily while they hold, from when both are tucked until they set off to land; it needs `legs` (see
    `air_kicks`). Returns ({side: (F, 3) offsets (x its left, y behind, z up) m}, {side: (F,) the toes' pitch down, rad},
    [a report per span that kicks]); overlapping spans take the larger tuck and add their kicks. Raises ValueError for a
    bad span."""
    ts = np.asarray(ts, float)
    off = {s: np.zeros((len(ts), 3)) for s in ("L", "R")}
    pitch = {s: np.zeros(len(ts)) for s in ("L", "R")}
    kick_off = {s: np.zeros((len(ts), 3)) for s in ("L", "R")}
    kick_pitch = {s: np.zeros(len(ts)) for s in ("L", "R")}
    reports = []
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
        tucked = {}
        for s in ("L", "R"):
            k, late = (1.0, 0.0) if s == first else (share, lag)
            e = smooth((ts - t0 - late) / rise) * down * k
            v = np.stack([np.zeros(len(ts)), float(c["back"]) * e, float(c["tuck"]) * e], 1)
            off[s] = np.where(np.abs(v) > np.abs(off[s]), v, off[s])
            pitch[s] = np.maximum(pitch[s], math.radians(float(c["point"])) * e)
            tucked[s] = (0.0, float(c["back"]) * k, float(c["tuck"]) * k)
        kicks = kicks_spec(sp.get("kicks"), f"{where} kicks")
        if kicks is not None:
            if legs is None:
                raise ValueError(f"{where} kicks: the legs (hip joints, thigh and shin lengths) are needed to kick")
            o, p, rep = air_kicks(ts, t0 + rise + lag, t1 - extend, kicks, legs, tucked, first, f"{where} kicks")
            for s in ("L", "R"):
                kick_off[s] += o[s]
                kick_pitch[s] += p[s]
            reports.append(rep)
    return ({s: off[s] + kick_off[s] for s in ("L", "R")}, {s: pitch[s] + kick_pitch[s] for s in ("L", "R")}, reports)


def kicks_spec(v, where="kicks"):
    """An air span's `kicks`: None when it is missing or false, the defaults for true, else the table, checked."""
    if v is None or v is False:
        return None
    if v is True:
        return {}
    if not isinstance(v, dict) or set(v) - set(KICKS_KEYS):
        raise ValueError(f"{where}: true or {{{', '.join(KICKS_KEYS)}}}, not {v!r}")
    return dict(v)


def kick_beats(beats, downbeats=(), every=1):
    """Every `every`-th of `beats` (clip seconds), counted from the beat nearest the first downbeat (from the first beat
    without downbeats): where a float's kicks land, on the strong beats of the bar."""
    bs = np.sort(np.asarray(list(beats), float))
    every = int(every)
    if every < 1:
        raise ValueError(f"kicks every {every} beats: a count of beats, 1 or more")
    if not len(bs):
        return []
    i0 = int(np.argmin(np.abs(bs - float(downbeats[0])))) if len(downbeats) else 0
    return [float(b) for i, b in enumerate(bs) if (i - i0) % every == 0]


def leg_angles(p, thigh, shin):
    """(the thigh's angle forward of straight down, the knee's bend), radians, of a leg whose ankle is at p = (forward,
    up) metres from its hip joint, the knee bending backward; a point out of reach straightens it."""
    f, u = float(p[0]), float(p[1])
    d = min(math.hypot(f, u), thigh + shin)
    knee = math.acos(float(np.clip((d * d - thigh * thigh - shin * shin) / (2.0 * thigh * shin), -1.0, 1.0)))
    return math.atan2(f, -u) + math.atan2(shin * math.sin(knee), thigh + shin * math.cos(knee)), knee


def leg_ankle(phi, knee, thigh, shin):
    """The ankle (forward, up) from the hip joint of a leg whose thigh is `phi` forward of straight down and whose knee is
    bent `knee` (radians, scalars or arrays): the inverse of leg_angles."""
    return (thigh * np.sin(phi) + shin * np.sin(phi - knee), -thigh * np.cos(phi) - shin * np.cos(phi - knee))


def _wobble(ts, cycle, phases):
    """A slow irregular signal within -1..1: the WOBBLE sines, periods in units of `cycle` seconds, at `phases`."""
    ts = np.asarray(ts, float)
    return sum(c * np.sin(2.0 * math.pi * ts / (cycle * r) + p) for (r, c), p in zip(WOBBLE, phases))


def _wobble_area(ts, cycle, phases):
    """An antiderivative of _wobble over time (seconds)."""
    ts = np.asarray(ts, float)
    return sum(-c * cycle * r / (2.0 * math.pi) * np.cos(2.0 * math.pi * ts / (cycle * r) + p)
               for (r, c), p in zip(WOBBLE, phases))


def air_kicks(ts, t0, t1, spec, legs, tucked, first="L", where="kicks"):
    """Lazy kicks of legs floating in the air (treading water, dangling weightless) from t0 (both legs tucked) to t1
    (they set off to reach down for the landing). The legs take turns: each knee straightens and bends again by `knee`
    degrees either side of its tucked bend, the thigh swinging `hip` degrees forward and back a little ahead of it
    (KICK_HIP_LEAD) and the foot pointing and flexing against the shin by `foot` degrees a little after it
    (KICK_FOOT_LAG), so each kick runs down the leg. A leg kicks every `period` s, the foot `first` first; with `beats`
    (clip seconds) its knee is straightest on every `every`-th of them at or after t0 instead, the legs taking turns
    (`kick_beats` picks a timeline's). `wander` (0..WANDER_MAX) makes them irregular, smoothly and seeded by `seed`: each
    kick's size and, off the beats, its pace vary by up to that share, the legs drifting a little from taking turns
    exactly (KICK_DRIFT). The kicks ease in over `ease` s from t0 and out by t1. A knee never straightens past
    KICK_KNEE_MIN: from a tuck too straight for its kick it bends further on average. Defaults KICKS.
    `legs` {side: {"hip": (x, y, z) the hip joint from the ankle's place at the base pose, "thigh", "shin": lengths}} and
    `tucked` {side: the ankle's air offset while the legs hold} are metres in the character's axes (x its left, y behind,
    z up); the bends are turned into ankle positions in each leg's side-on plane, which the leg IK bends back into them.
    Returns ({side: (F, 3) ankle offsets on top of `tucked`}, {side: (F,) the foot's pitch down on top of the air's,
    rad}, a report). Raises ValueError for a bad table or too few beats."""
    c = dict(KICKS, **spec)
    knee, hip, foot = (math.radians(float(c[k])) for k in ("knee", "hip", "foot"))
    period, wander, ease, every = float(c["period"]), float(c["wander"]), float(c["ease"]), int(c["every"])
    if min(knee, hip, foot) < 0 or period <= 0 or ease <= 0 or every < 0 or not 0.0 <= wander <= WANDER_MAX:
        raise ValueError(f"{where}: knee, hip and foot not negative, period and ease positive, every a count of beats, "
                         f"wander within 0..{WANDER_MAX:g}")
    ts = np.asarray(ts, float)
    n = len(ts)
    off = {s: np.zeros((n, 3)) for s in ("L", "R")}
    pitch = {s: np.zeros(n) for s in ("L", "R")}
    report = {"from": round(float(t0), 3), "to": round(float(t1), 3)}
    if t1 <= t0:
        report["kicks"] = {"L": 0, "R": 0}
        return off, pitch, report
    ease = min(ease, 0.5 * (t1 - t0))
    w = smooth((ts - t0) / ease) * (1.0 - smooth((ts - (t1 - ease)) / ease))
    rnd = random.Random(int(c["seed"]))
    ph = {k: [rnd.uniform(0.0, 2.0 * math.pi) for _ in WOBBLE] for k in ("pace", "drift", "L", "R")}
    other = "R" if first == "L" else "L"
    beats = c.get("beats")
    if beats is not None:
        if not isinstance(beats, (list, tuple, np.ndarray)):
            raise ValueError(f"{where}: beats = [t, ...], clip seconds, not {beats!r}")
        lands = np.unique(np.asarray(beats, float))
        lands = lands[lands >= t0 - 1e-9][::max(every, 1)]
        if len(lands) < 2 or lands[1] > t1:
            raise ValueError(f"{where}: fewer than two beats to kick on between {t0:g} and {t1:g} s")
        k = np.interp(ts, lands, np.arange(len(lands), dtype=float))
        k = np.where(ts < lands[0], (ts - lands[0]) / (lands[1] - lands[0]), k)
        k = np.where(ts > lands[-1], len(lands) - 1 + (ts - lands[-1]) / (lands[-1] - lands[-2]), k)
        cycle, drift = 2.0 * float(np.mean(np.diff(lands))), 0.0
        report["on"] = "beats"
    elif every:
        raise ValueError(f"{where}: every {every} beats needs the beats (a timeline's, see kick_beats)")
    else:
        cycle = period
        pace = _wobble_area(ts, period, ph["pace"]) - _wobble_area(t0, period, ph["pace"])
        k = -0.5 + 2.0 / period * (ts - t0 + wander * pace)
        drift = 2.0 * math.pi * KICK_DRIFT * wander * _wobble(ts, period, ph["drift"])
        report["on"] = "period"
    report["cycle_s"] = round(cycle, 3)
    theta = {first: math.pi * k, other: math.pi * (k - 1.0) + drift}
    report.update(kicks={}, landings={}, tuck_knee_deg={}, knee_deg={}, foot_mm={})
    for s in ("L", "R"):
        lg = legs[s]
        a, b = float(lg["thigh"]), float(lg["shin"])
        hy, hz = float(lg["hip"][1]), float(lg["hip"][2])
        ty, tz = float(tucked[s][1]), float(tucked[s][2])
        phi, bend = leg_angles((hy - ty, tz - hz), a, b)          # the tucked ankle from the hip: (forward, up)
        amp = w * (1.0 + wander * _wobble(ts, cycle, ph[s]))
        th = theta[s]
        dk = -knee * amp * np.cos(th) + w * max(0.0, knee * (1.0 + wander) + KICK_KNEE_MIN - bend)
        dh = hip * amp * np.cos(th + 2.0 * math.pi * KICK_HIP_LEAD)
        dr = foot * amp * np.cos(th - 2.0 * math.pi * KICK_FOOT_LAG)
        f0, u0 = leg_ankle(phi, bend, a, b)
        f1, u1 = leg_ankle(phi + dh, bend + dk, a, b)
        off[s] = np.stack([np.zeros(n), f0 - f1, u1 - u0], 1)
        pitch[s] = dk - dh + dr                                   # the shin's turn back, and the foot's own on it
        turn = np.floor(th / (2.0 * math.pi))
        at = [float(ts[j]) for j in np.nonzero(np.diff(turn) > 0)[0] + 1 if w[j] >= 0.5]
        report["kicks"][s] = len(at)
        report["landings"][s] = [round(x, 2) for x in at]
        report["tuck_knee_deg"][s] = round(math.degrees(bend), 1)
        live = w > 0
        report["knee_deg"][s] = [round(math.degrees(bend + float(x)), 1) for x in (dk[live].min(), dk[live].max())] \
            if live.any() else [round(math.degrees(bend), 1)] * 2
        report["foot_mm"][s] = round(float(np.linalg.norm(off[s], axis=1).max()) * 1000, 1)
    return off, pitch, report


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
