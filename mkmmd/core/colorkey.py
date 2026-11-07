"""Colour keys pushed too far, bpy-free (docs/design.md: Frame effects, Keys pushed too far): `[[fx]] kind = "key"`
checked and timed. The picture work is mkmmd/colorkey.py.

A colour of the picture (`color`, by default the palette's `base`) is keyed out and what lies behind shows through: a
video (`clip`, fitted to cover the frame, mkmmd.footage), a flat colour (`back`) or the cut's own frame `delay` frames
before. The key is meant to be pushed past clean: a pixel is keyed fully within `tolerance` of the colour, not at all
past `tolerance` + `soft` (the distance measured in RGB, in the chroma plane alone or in brightness alone: `space`),
and each of the red, green and blue planes keys at its own tolerance (`split`), so the edge of the key comes apart into
colour fringes; `bands` crushes the matte into that many hard steps, `gain` pushes the planes of what shows through,
`spill` stains what is left near the key in a colour, and `burn` keeps the key of the frames before, fading, so moving
edges smear into one another. The tolerance moves with its keys (`[[t, v], ...]`, as a `strength`), with a stem's
loudness (`follow`, the song's voice pouring the world in while it sings) and on the hits (`pump` on a `lane`, dying
away over `decay`), and with a `spread` it is lower the further a pixel lies from a `center` (a scene point too), so
the key opens out from there. A `mask` keeps it off the figure (`ground`) or on it alone (`figure`)."""
import math

import numpy as np

KINDS = ("key",)
SPACES = ("rgb", "chroma", "luma")               # how far a pixel is from the key colour: all of it, its hue and
#                                                  saturation alone (BT.709 Cb Cr), its brightness alone
SOURCES = ("clip", "back", "delay")              # what shows through: a video, a flat colour, the cut's frame before
FOLLOW = {"stem": "vocals", "amount": 0.2, "smooth": 0.12, "range": 24.0, "timeline": "audio/timeline.json"}
SPILL = {"amount": 0.5, "reach": 0.15}
ATTACK = 0.02                                    # seconds a pump rises over on its hit (then exp(-age / decay))
KEYS = {"key": {"color", "space", "tolerance", "soft", "split", "bands", "burn", "fade", "clip", "in", "rate", "loop",
                "focus", "mirror", "back", "delay", "gain", "spill", "follow", "pump", "decay", "times", "lane",
                "min_db", "every", "mask", "objects", "center", "spread"}}
DEFAULTS = {"key": {"color": "base", "space": "rgb", "tolerance": 0.15, "soft": 0.1, "split": 0.0, "bands": 0,
                    "burn": 0, "fade": 0.7, "in": 0.0, "rate": 1.0, "loop": False, "focus": [0.5, 0.5],
                    "mirror": False, "back": None, "delay": 0, "gain": [1.0, 1.0, 1.0], "spill": None,
                    "follow": None, "pump": 0.0, "decay": 0.25, "min_db": None, "every": 1, "mask": None,
                    "center": None, "spread": 0.0}}


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A key's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    out["color"] = FX._colour(out["color"], palette, f"{what} color")
    if out["space"] not in SPACES:
        raise FX.FxError(f"{what}: space {out['space']!r} is none of {', '.join(SPACES)}")
    out["tolerance"] = FX._keys(out["tolerance"], f"{what} tolerance")
    out["soft"] = FX._num(out["soft"], f"{what} soft", 0.0, 2.0)
    out["spread"] = FX._num(out["spread"], f"{what} spread", 0.0, 20.0)
    if out["spread"] > 0.0:                          # the key opens from a point: tolerance lost with the distance
        out["center"] = FX._center([0.5, 0.5] if out["center"] is None else out["center"], f"{what} center")
    elif "center" in spec:
        raise FX.FxError(f"{what}: center is where a key with a `spread` opens from: give the spread")
    s = out["split"]
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        s = FX._num(s, f"{what} split", -1.0, 1.0)
        out["split"] = [s, 0.0, -s]                   # red keys wider, blue narrower: the fringes run red, white, blue
    elif isinstance(s, (list, tuple)) and len(s) == 3:
        out["split"] = [FX._num(x, f"{what} split", -1.0, 1.0) for x in s]
    else:
        raise FX.FxError(f"{what}: split = {s!r}: expected a number or [red, green, blue] tolerance offsets")
    out["bands"] = FX._num(out["bands"], f"{what} bands", 0, 64, integer=True)
    out["burn"] = FX._num(out["burn"], f"{what} burn", 0, 60, integer=True)
    out["fade"] = FX._num(out["fade"], f"{what} fade", 0.0, 1.0)
    given = [k for k in SOURCES if spec.get(k) not in (None, 0)]
    if len(given) != 1:
        raise FX.FxError(f"{what}: give one of `clip` (a video), `back` (a colour) or `delay` (frames before) for what "
                         f"shows through the key{'' if not given else ', not ' + ' and '.join(given)}")
    out["source"] = given[0]
    if out["source"] == "clip":
        FX._clip_keys(out, what)
    else:
        loose = sorted(k for k in ("in", "rate", "loop", "focus", "mirror") if k in spec)
        if loose:
            raise FX.FxError(f"{what}: {', '.join(loose)} are a clip's: give the `clip` or none of them")
    if out["source"] == "back":
        out["back"] = FX._colour(out["back"], palette, f"{what} back")
    if out["source"] == "delay":
        out["delay"] = FX._num(out["delay"], f"{what} delay", 1, 600, integer=True)
    g = out["gain"]
    if not (isinstance(g, (list, tuple)) and len(g) == 3):
        raise FX.FxError(f"{what}: gain = {g!r}: expected [red, green, blue] multipliers")
    out["gain"] = [FX._num(x, f"{what} gain", 0.0, 8.0) for x in g]
    out["spill"] = _spill(FX, out["spill"], palette, what)
    out["follow"] = _follow(FX, out["follow"], what)
    out["pump"] = FX._num(out["pump"], f"{what} pump", -2.0, 2.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", 0.01, 10.0)
    FX._hits(spec, out, what, required=False)
    hits = bool(out.get("times")) or out.get("lane") is not None
    if out["pump"] != 0.0 and not hits:
        raise FX.FxError(f"{what}: a pump widens the key on the hits: give `times` or a `lane`")
    if out["pump"] == 0.0 and hits:
        raise FX.FxError(f"{what}: hits (`times`, `lane`) pump the key: give the `pump` (tolerance added on a hit)")
    FX._mask(spec, out, what)


def _spill(FX, v, palette, what):
    """`spill`: a colour, or {color, amount, reach}: what is left of the picture near the key stained toward the
    colour, by `amount` (0.5) where it all but keys, fading out `reach` (0.15) further from the key colour."""
    if v is None:
        return None
    if not isinstance(v, dict):
        v = {"color": v}
    extra = sorted(set(v) - {"color", "amount", "reach"})
    if extra or "color" not in v:
        raise FX.FxError(f"{what}: spill = {v!r}: expected a colour or {{color, amount, reach}}")
    f = {**SPILL, **v}
    return {"color": FX._colour(f["color"], palette, f"{what} spill"),
            "amount": FX._num(f["amount"], f"{what} spill amount", 0.0, 1.0),
            "reach": FX._num(f["reach"], f"{what} spill reach", 0.001, 2.0)}


def _follow(FX, v, what):
    """`follow`: {stem, amount, smooth, range, timeline}: the tolerance grows by `amount` times the stem's loudness,
    0 at `range` dB under its loud passages, 1 at them, smoothed over `smooth` s (mkmmd.core.framefx.STEMS: the stems
    cached when `timeline` was analysed)."""
    if v is None:
        return None
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: follow = {v!r}: expected true or {{stem, amount, smooth, range, timeline}}")
    extra = sorted(set(v) - set(FOLLOW))
    if extra:
        raise FX.FxError(f"{what}: follow: unknown keys {extra} (known: {sorted(FOLLOW)})")
    f = {**FOLLOW, **v}
    if f["stem"] not in FX.STEMS:
        raise FX.FxError(f"{what}: follow stem {f['stem']!r} is none of {', '.join(FX.STEMS)}")
    if not isinstance(f["timeline"], str) or not f["timeline"].strip():
        raise FX.FxError(f"{what}: follow timeline is the timeline file whose stems it reads")
    return {"stem": f["stem"], "timeline": f["timeline"],
            "amount": FX._num(f["amount"], f"{what} follow amount", -2.0, 2.0),
            "smooth": FX._num(f["smooth"], f"{what} follow smooth", 0.0, 5.0),
            "range": FX._num(f["range"], f"{what} follow range", 1.0, 80.0)}


def sources(fx, frame):
    """The cut's frames a key reads at `frame`: those its `burn` keys again (the trail), and the frame before that shows
    through with a `delay`."""
    out = [frame - j for j in range(1, fx["burn"] + 1)]
    if fx["source"] == "delay" and frame - fx["delay"] not in out:
        out.append(frame - fx["delay"])
    return out


def levels(samples, rate, pre, t0, t1, fps, smooth, range_db):
    """0..1 loudness of a stem on each frame from clip second `t0` up to `t1` (int(round((t1 - t0) * fps)) frames; frame
    i at t0 + i / fps): the RMS over the frame's own 1 / fps s in dB, smoothed by a Gaussian of `smooth` s, 1 at the
    span's loud passages (its 95th percentile) and 0 `range_db` under them. `samples` (mono) start `pre` s before clip
    second 0, at `rate` per second."""
    x = np.asarray(samples, np.float64)
    n = max(int(round((t1 - t0) * fps)), 1)
    half = 0.5 / fps
    t = t0 + np.arange(n) / fps
    a = np.clip(np.round((pre + t - half) * rate).astype(np.int64), 0, len(x))
    b = np.clip(np.round((pre + t + half) * rate).astype(np.int64), 0, len(x))
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    ms = (c[b] - c[a]) / np.maximum(b - a, 1)
    db = 10.0 * np.log10(ms + 1e-12)
    sigma = smooth * fps
    if sigma > 0.25:
        r = int(math.ceil(3.0 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        db = np.convolve(np.pad(db, r, mode="edge"), k / k.sum(), mode="valid")
    hi = float(np.percentile(db, 95.0))
    return np.clip((db - (hi - range_db)) / range_db, 0.0, 1.0)


def tolerance(fx, t, level=0.0, times=()):
    """A key's tolerance at clip second `t`: its keys there, plus `follow.amount` x the stem's `level` (0..1, levels)
    and `pump` x the hits' punch (each rising over ATTACK s and dying as exp(-age / decay), the strongest counting)."""
    FX = _fx()
    keys = fx["tolerance"]
    v = FX.strength({"strength": keys}, t)
    if fx["follow"] is not None:
        v += fx["follow"]["amount"] * float(level)
    if fx["pump"] and times:
        v += fx["pump"] * float(FX.punch(t, times, ATTACK, fx["decay"]))
    return float(v)
