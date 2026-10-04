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


def beat_bob(ts, beats, deg=1.5, attack=0.06, decay=0.22, accent=None):
    """Head nod (rad) on the beats: a quick dip after each beat, easing back; accent (per beat weight) scales it."""
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
    return math.radians(deg) * out


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
