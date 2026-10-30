"""Wobbling letters, the numbers (docs/design.md: Text, Lyrics, Wobble): the bpy-free half of `lyrics.wobble`.

While a word is sung its letters bob and sway on a wave that travels along the word, from its first letter to its last,
each tilting with the slope of the wave under it, like jelly letters on a ripple. Letter i (0 the first) of a word is at
phase theta_i = 2 pi (phase(t) - i / wave): it rises `amount` em x amp(t) x sin(theta_i), sways sideways `sway` times that
on cos(theta_i) and turns -`tilt` degrees x amp(t) x cos(theta_i) (counter-clockwise positive). `phase` runs at `rate`
cycles a second from a start of its own for every word (so neighbours never bob in step); `amp` is the voice: 0 before the
note, rising over `attack` s as it starts, following the smoothed level of the timeline's `vocal_db` while the note is
held (the level that `weight` uses, 0 at -34 dB and 1 at -5 dB re the vocal's peak, so a note that swells and fades
wobbles more and less with it) and dying away over `release` s after it. `follow` (0..1) is how much of the amplitude is
the level: 1 bobs as loud as she sings, 0 bobs fully whenever the voice sounds at all; either way a silent voice keeps the
letters still. The rate is chosen, not measured: the timeline's per-frame level is too coarse to read a vibrato from.

The text stage builds exactly `letter` in the node tree from two keyed numbers per word (`WobAmp`, `WobPhase`) and the
constants; everything is a function of the timeline and the word's numbers, so every build is the same."""
import math

WOBBLE_DEFAULTS = {"amount": 0.08, "sway": 0.35, "tilt": 7.0, "rate": 2.2, "wave": 4.5, "follow": 1.0,
                   "attack": 0.12, "release": 0.35, "min_hold": 0.0}
SOUNDING = 0.15                 # level at which a voice counts as fully sounding when the amplitude does not follow it


class WobbleError(ValueError):
    """A wobble table that makes no sense; messages carry numbers only."""


def options(tab):
    """The checked `wobble` table with its defaults (True or {} is all defaults)."""
    tab = {} if tab is True else tab
    if not isinstance(tab, dict):
        raise WobbleError("wobble is true or a table {amount, sway, tilt, rate, wave, follow, attack, release, min_hold}")
    unknown = sorted(set(tab) - set(WOBBLE_DEFAULTS))
    if unknown:
        raise WobbleError(f"wobble keys {unknown} are unknown (known: {sorted(WOBBLE_DEFAULTS)})")
    o = {**WOBBLE_DEFAULTS, **tab}
    for key in WOBBLE_DEFAULTS:
        v = o[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise WobbleError(f"wobble {key} must be a number")
        o[key] = float(v)
    if min(o["amount"], o["sway"], o["tilt"], o["min_hold"]) < 0:
        raise WobbleError("wobble amount, sway, tilt and min_hold are not negative")
    if not (o["rate"] > 0 and o["wave"] > 0 and o["attack"] > 0 and o["release"] > 0):
        raise WobbleError("wobble rate, wave, attack and release are positive")
    if not 0.0 <= o["follow"] <= 1.0:
        raise WobbleError("wobble follow is a share, 0..1")
    return o


def held(t0, t1, o):
    """Whether a note sung from t0 to t1 wobbles at all (it is held `min_hold` s or more)."""
    return t1 - t0 >= o["min_hold"]


def _sstep(a, b, x):
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


def gate(t, t0, t1, attack, release):
    """0..1: the note's own envelope, rising over `attack` s from t0 and dying over `release` s after t1."""
    return _sstep(0.0, attack, t - t0) * (1.0 - _sstep(0.0, release, t - t1))


def amplitude(t, t0, t1, level, o):
    """0..1 share of the full wobble at clip time t of a note sung t0..t1, `level` the voice's 0..1 level at t."""
    g = gate(t, t0, t1, o["attack"], o["release"])
    if g <= 0.0 or level <= 0.0:
        return 0.0
    f = o["follow"]
    return g * (f * min(level, 1.0) + (1.0 - f) * _sstep(0.0, SOUNDING, level))


def letter(i, phase, amp, o):
    """Letter i's wobble at wave phase `phase` (cycles) and amplitude `amp` (0..1): {dx, dy} em, {rot} degrees
    counter-clockwise. The node tree builds these expressions."""
    th = 2.0 * math.pi * (phase - i / o["wave"])
    s, c = math.sin(th), math.cos(th)
    return {"dx": amp * o["sway"] * o["amount"] * c, "dy": amp * o["amount"] * s, "rot": -amp * o["tilt"] * c}


def keys(t0, t1, land, last, fps, level_at, o, phase0=0.0):
    """The kinetic `wobble` table of a word sung t0..t1 on screen over clip frames land..last: {amp: [[t, 0..1], ...] one key
    a frame from the landing until the release has died (or the word is gone), phase: [[t, cycles], ...] (linear: two keys),
    and the constants amount, sway, tilt, wave}, or None when the voice never moves it (silent, or gone first).
    `level_at(t)` is the voice's 0..1 level at clip time t; `phase0` the word's own start of the wave (cycles)."""
    c1 = min(int(last), int(math.ceil((t1 + o["release"]) * fps)))
    if c1 < land:
        return None
    amp = [(c / fps, amplitude(c / fps, t0, t1, level_at(c / fps), o)) for c in range(int(land), c1 + 1)]
    if max(v for _, v in amp) <= 1e-6:
        return None
    ta, tb = land / fps, max(c1, land + 1) / fps
    return {"amp": [[t, v] for t, v in amp], "phase": [[ta, phase0], [tb, phase0 + o["rate"] * (tb - ta)]],
            "amount": o["amount"], "sway": o["sway"], "tilt": o["tilt"], "wave": o["wave"]}
