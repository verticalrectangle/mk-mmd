"""Picture effects, bpy-free (docs/design.md: Shots, Frame effects): the `[[fx]]` kinds that treat the finished picture
as a medium or draw on it, checked, timed and laid out here; mkmmd/picturefx.py does the picture work.

    pixelsort  runs of pixels whose luma, hue or saturation lies in a band, sorted along the rows (or the columns):
               steady, or firing on hits and dying away
    film       a motion-picture print: the picture held `fps` times a second, a film curve (black and white with
               `mono`), grain, exposure flicker, gate weave, dust, scratches that persist and drift, a hair in the gate
    camcorder  a home video tape: a tracking band rolling up the frame, colour bleed, scanlines, line jitter, soft luma
               and noise, and the date and time stamp drawn crisp over it
    doodle     purikura-pen marks (a star, a heart, an X over an eye ...) popping in with a spring, boiling
    booth      a photo-booth strip: on each of the first hits the cut's frame is captured into a cell, with a flash,
               the captures dropping into their cells, doodles round them

Times are clip seconds. Doodle shapes are strokes in a box -0.5..0.5 about their centre, x right and y down, as on
screen. The randomness is seeded per effect and frame (or film frame, or boil step): the same seed draws the same frames
the same way, in any order.
"""
import math
import re

import numpy as np

KINDS = ("pixelsort", "film", "camcorder", "doodle", "booth")
HOLDS = ("film", "booth")                       # the kinds that show a frame of the cut from before (a held frame, a capture)
SORT_BY = ("luma", "hue", "sat")
SORT_AXES = ("x", "y")
SHAPES = ("star", "heart", "sparkle", "x", "swirl", "scribble", "arrow", "crown", "cat_ears")
STICKERS = ("star", "heart", "sparkle", "swirl", "scribble", "arrow", "crown")   # what a booth draws round its photos
ACCENTS = ("love", "gold", "rose", "pine", "foam", "iris")                       # the palette slots it draws them in
CORNERS = ("bottom-right", "bottom-left", "top-left", "top-right")
INDICATORS = ("PLAY", "REC")
GRAIN_SIZE = 1.5                 # px at 1080: a film's grain when `grain` is a number
DUST_SIZE = 3.0                  # px at 1080: a speck of dust when `dust` is a number
TRACK_SPEED = 0.25               # frame heights a second: a camcorder's tracking band when `tracking` is a number
TRACK_BAND = 0.07                # a tracking band's half height, frame heights
POP = 0.3                        # seconds: a doodle's spring as it pops in
WOBBLE_EVERY = 0.12              # a boiling stroke bends this often along itself (shape units)
BOOTH_PEN = {"fps": 8.0, "width": 7.0, "outline": 4.5, "wobble": 1.2}   # a booth's doodles: boils a second, px at 1080
CLOCK = re.compile(r"^\s*(?:(AM|PM)\s*)?(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(AM|PM))?\s*$", re.IGNORECASE)


def normalize(spec, out, palette, what):
    """A picture effect's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    {"pixelsort": _pixelsort, "film": _film, "camcorder": _camcorder, "doodle": _doodle, "booth": _booth}[out["kind"]](
        FX, spec, out, palette, what)


def _bool(FX, out, k, what):
    if not isinstance(out[k], bool):
        raise FX.FxError(f"{what}: {k} is true or false")


def _text(FX, v, what, longest=40):
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip() or len(v) > longest:
        raise FX.FxError(f"{what} = {v!r}: expected a short text (1..{longest} characters)")
    return v


def _font(FX, v, what):
    if v is not None and (not isinstance(v, str) or not v.strip()):
        raise FX.FxError(f"{what} = {v!r}: expected a font slug of the asset registry or a font file")
    return v


def _amount(FX, v, what, first, second, default):
    """A number (the `first` value) or a table {first, second}: -> {first, second} (a film's grain {amount, size} and
    dust {count, size}, a camcorder's tracking {amount, speed}); `second` defaults to `default`."""
    if not isinstance(v, dict):
        v = {first: v}
    if first not in v or set(v) - {first, second}:
        raise FX.FxError(f"{what} = {v!r}: expected a number or {{{first}, {second}}}")
    return {first: FX._num(v[first], f"{what} {first}", lo=0.0),
            second: FX._num(v.get(second, default), f"{what} {second}", lo=1e-3)}


def _pixelsort(FX, spec, out, palette, what):
    if out["by"] not in SORT_BY:
        raise FX.FxError(f"{what}: by {out['by']!r} is none of {', '.join(SORT_BY)}")
    out["lo"], out["hi"] = FX._num(out["lo"], f"{what} lo", 0.0, 1.0), FX._num(out["hi"], f"{what} hi", 0.0, 1.0)
    if out["hi"] <= out["lo"]:
        raise FX.FxError(f"{what}: lo ({out['lo']:g}) .. hi ({out['hi']:g}) holds nothing: hi must be above lo")
    if out["axis"] not in SORT_AXES:
        raise FX.FxError(f"{what}: axis {out['axis']!r} is none of {', '.join(SORT_AXES)} (x sorts along the rows)")
    _bool(FX, out, "reverse", what)
    FX._mask(spec, out, what)
    out["decay"] = FX._num(out["decay"], f"{what} decay", lo=1e-3)
    FX._hits(spec, out, what, required=False)


def _film(FX, spec, out, palette, what):
    _bool(FX, out, "mono", what)
    out["fps"] = None if out["fps"] is None else FX._num(out["fps"], f"{what} fps", 1.0, 240.0)
    out["grain"] = _amount(FX, out["grain"], f"{what} grain", "amount", "size", GRAIN_SIZE)
    out["dust"] = _amount(FX, out["dust"], f"{what} dust", "count", "size", DUST_SIZE)
    out["flicker"] = FX._num(out["flicker"], f"{what} flicker", 0.0, 1.0)
    out["weave"] = FX._num(out["weave"], f"{what} weave", lo=0.0)
    out["scratches"] = FX._num(out["scratches"], f"{what} scratches", 0, 32, integer=True)
    out["hair"] = FX._num(out["hair"], f"{what} hair", 0.0, 1.0)
    out["vignette"] = FX._num(out["vignette"], f"{what} vignette", 0.0, 1.0)
    out["contrast"] = FX._num(out["contrast"], f"{what} contrast", 0.0, 4.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


def _camcorder(FX, spec, out, palette, what):
    out["date"] = _text(FX, out["date"], f"{what} date", 24)
    out["time"] = _text(FX, out["time"], f"{what} time", 24)
    _bool(FX, out, "tick", what)
    out["clock"] = None
    if out["tick"]:
        if out["time"] is None:
            raise FX.FxError(f"{what}: tick counts the seconds of the `time`: give the time (\"8:41 PM\")")
        out["clock"] = clock(out["time"], f"{what} time", FX.FxError)
    out["font"] = _font(FX, out["font"], f"{what} font")
    if out["corner"] not in CORNERS:
        raise FX.FxError(f"{what}: corner {out['corner']!r} is none of {', '.join(CORNERS)}")
    ind = out["indicator"]
    if isinstance(ind, str) and ind.upper() in INDICATORS:
        out["indicator"] = ind.upper()
    elif ind is None or ind is False or (isinstance(ind, str) and ind.lower() == "none"):
        out["indicator"] = None
    else:
        raise FX.FxError(f"{what}: indicator {ind!r} is \"PLAY\", \"REC\" or \"none\"")
    out["color"] = FX._colour(out["color"], palette, f"{what} color")
    out["size"] = FX._num(out["size"], f"{what} size", 0.005, 0.3)
    out["tracking"] = _amount(FX, out["tracking"], f"{what} tracking", "amount", "speed", TRACK_SPEED)
    for k in ("chroma", "jitter", "soft", "noise"):
        out[k] = FX._num(out[k], f"{what} {k}", lo=0.0)
    out["scanlines"] = FX._num(out["scanlines"], f"{what} scanlines", 0.0, 1.0)
    out["saturation"] = FX._num(out["saturation"], f"{what} saturation", 0.0, 2.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


_MARK = {"shape", "at", "size", "size_m", "color", "angle", "t"}


def _doodle(FX, spec, out, palette, what):
    marks = out.get("marks")
    if not isinstance(marks, (list, tuple)) or not marks or not all(isinstance(m, dict) for m in marks):
        raise FX.FxError(f"{what}: `marks` is required: a list of {{shape, at, size, color, angle, t}}")
    out["marks"] = []
    for j, m in enumerate(marks):
        w = f"{what} mark {j}"
        extra = sorted(set(m) - _MARK)
        if extra:
            raise FX.FxError(f"{w}: unknown keys {extra} (known: {sorted(_MARK)})")
        if m.get("shape") not in SHAPES:
            raise FX.FxError(f"{w}: shape {m.get('shape')!r} is none of {', '.join(SHAPES)}")
        if "at" not in m:
            raise FX.FxError(f"{w}: `at` is required: [x, y] of the frame or {{point = \"<expression>\"}}")
        at = FX._center(m["at"], f"{w} at")
        n = {"shape": m["shape"], "at": at, "color": FX._colour(m.get("color", "love"), palette, f"{w} color"),
             "angle": FX._num(m.get("angle", 0.0), f"{w} angle"), "t": FX._num(m.get("t", out["from"]), f"{w} t")}
        if "size_m" in m:
            if "size" in m:
                raise FX.FxError(f"{w}: give `size` (frame heights) or `size_m` (metres), not both")
            if at["mode"] != "point":
                raise FX.FxError(f"{w}: `size_m` measures at a scene point's depth: give at = {{point = ...}}")
            n["size_m"] = FX._num(m["size_m"], f"{w} size_m", lo=1e-6)
        else:
            n["size"] = FX._num(m.get("size", 0.08), f"{w} size", lo=1e-4)
        if not out["from"] <= n["t"] < out["to"]:
            raise FX.FxError(f"{w}: t = {n['t']:g} is outside the span ({out['from']:g} .. {out['to']:g})")
        out["marks"].append(n)
    out["fps"] = FX._num(out["fps"], f"{what} fps", 0.5, 60.0)
    for k in ("width", "outline", "wobble"):
        out[k] = FX._num(out[k], f"{what} {k}", lo=0.0)
    if out["width"] <= 0:
        raise FX.FxError(f"{what}: width = 0: the pen draws nothing")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


def _booth(FX, spec, out, palette, what):
    out["count"] = FX._num(out["count"], f"{what} count", 1, 12, integer=True)
    FX._hits(spec, out, what)
    for k in ("flash", "border"):
        out[k] = FX._num(out[k], f"{what} {k}", lo=0.0)
    out["drop"] = FX._num(out["drop"], f"{what} drop", lo=1e-3)
    out["tilt"] = FX._num(out["tilt"], f"{what} tilt", -45.0, 45.0)
    for k in ("paper", "back", "color"):
        out[k] = FX._colour(out[k], palette, f"{what} {k}")
    lay = out["layout"]
    if lay is not None:
        lay = FX._pair(lay, f"{what} layout")
        if any(int(v) != v or v < 1 for v in lay) or lay[0] * lay[1] < out["count"]:
            raise FX.FxError(f"{what}: layout = {out['layout']!r}: [columns, rows], whole numbers with room for "
                             f"{out['count']} photos")
        lay = [int(lay[0]), int(lay[1])]
    out["layout"] = lay
    out["ratio"] = FX._num(out["ratio"], f"{what} ratio", 0.2, 5.0)
    out["focus"] = FX._pair(out["focus"], f"{what} focus")
    out["size"] = FX._num(out["size"], f"{what} size", 0.1, 1.0)
    out["doodles"] = FX._num(out["doodles"], f"{what} doodles", 0, 8, integer=True)
    out["caption"] = _text(FX, out["caption"], f"{what} caption", 40)
    out["font"] = _font(FX, out["font"], f"{what} font")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["stickers"] = [FX._colour(c, palette, f"{what} doodle colour") for c in ACCENTS]


# ================================================================================================= timing
def frame_of(t, fps, frame0):
    return int(round(frame0 + float(t) * fps))


def film_hold(fx):
    """Clip frames per film frame: the clip's frame rate over the film's `fps` (1 without one, or when it is faster)."""
    return 1.0 if fx["fps"] is None else max(1.0, fx["frame_rate"] / fx["fps"])


def film_step(fx, frame):
    """(k, g): the film frame shown on clip frame `frame` (0 on the span's first frame) and the clip frame it was taken
    from, its first, so that the picture changes only `fps` times a second."""
    r = film_hold(fx)
    k = max(int(math.floor((frame - fx["f0"]) / r + 1e-9)), 0)
    return k, fx["f0"] + int(math.ceil(k * r - 1e-9))


def captures(fx, times, fps, frame0):
    """A booth's captures, [(clip second, clip frame)]: its first `count` hits (`times`, clip seconds) inside the span,
    two hits on one frame being one capture."""
    out = []
    for t in sorted(times):
        g = frame_of(t, fps, frame0)
        if fx["from"] <= t < fx["to"] and fx["f0"] <= g < fx["f1"] and all(g != h for _, h in out):
            out.append((float(t), g))
    return out[:fx["count"]]


def pop(age, dur=POP):
    """A doodle's scale `age` seconds after it pops in: 0 before, then a small spring from 0 that overshoots a little
    (12 %) at dur / 2 and settles on 1."""
    if age <= 0:
        return 0.0
    return float(1.0 - math.exp(-age / (dur / 4.3)) * math.cos(2.0 * math.pi * age / dur))


def flash(age, dur):
    """The white of a booth's flash `age` seconds after a capture: 1 on it, gone after `dur`."""
    if dur <= 0 or age < 0 or age >= dur:
        return 0.0
    return float((1.0 - age / dur) ** 2)


def drop(age, dur):
    """(scale, height) of a capture dropping into its cell `age` seconds after it was taken: it falls from 1.3 times its
    size, faster and faster, lands at 0.7 x `dur` and bounces a little; height 1 .. 0 drives its shadow."""
    if age >= dur:
        return 1.0, 0.0
    u = max(age, 0.0) / dur
    if u < 0.7:
        h = 1.0 - (u / 0.7) ** 2
        return 1.0 + 0.3 * h, h
    b = 0.025 * math.sin(math.pi * (u - 0.7) / 0.3)
    return 1.0 + b, b / 0.3


def tracking_centre(fx, t):
    """Where a camcorder's tracking band is at clip second `t`, frame fractions down the frame: it rolls up from below
    the frame to above it at its `speed`, and round again."""
    band = TRACK_BAND
    span = 1.0 + 4.0 * band
    u = (fx["tracking"]["speed"] * (t - fx["from"]) + 0.37 + 0.13 * fx["seed"]) % span / span
    return 1.0 + 2.0 * band - u * span


# ================================================================================================= the camcorder's stamp
def clock(text, what="time", err=ValueError):
    """A stamp's time ("8:41 PM", "PM 8:41", "20:41:07") -> {seconds (of the day), ampm ("pre", "post" or None), pad
    (hours with two digits: written "08", or a 24-hour clock's), upper (AM / PM in capitals)}."""
    m = CLOCK.match(text)
    if not m or (m.group(1) and m.group(5)):
        raise err(f"{what} = {text!r}: expected a time such as \"8:41 PM\", \"PM 8:41\" or \"20:41:07\"")
    pre, hh, mm, ss, post = m.groups()
    h, mi, s = int(hh), int(mm), int(ss or 0)
    ap = (pre or post or "").upper()
    if mi > 59 or s > 59 or (ap and not 1 <= h <= 12) or (not ap and h > 23):
        raise err(f"{what} = {text!r}: not a time of day")
    if ap:
        h = h % 12 + (12 if ap == "PM" else 0)
    return {"seconds": h * 3600 + mi * 60 + s, "ampm": "pre" if pre else "post" if post else None,
            "pad": hh.startswith("0") or (not ap and len(hh) == 2), "upper": text.upper() == text}


def clock_text(c, seconds):
    """The time of clock `c` (clock()) `seconds` later, with its seconds, in the style it was written in."""
    s = int(c["seconds"] + math.floor(seconds + 1e-9)) % 86400
    h, mi, ss = s // 3600, s // 60 % 60, s % 60
    if c["ampm"] is None:
        return f"{h:02d}:{mi:02d}:{ss:02d}" if c["pad"] else f"{h}:{mi:02d}:{ss:02d}"
    ap = "PM" if h >= 12 else "AM"
    ap = ap if c["upper"] else ap.lower()
    h12 = h % 12 or 12
    hm = f"{h12:02d}" if c["pad"] else f"{h12}"
    return f"{ap} {hm}:{mi:02d}:{ss:02d}" if c["ampm"] == "pre" else f"{hm}:{mi:02d}:{ss:02d} {ap}"


def stamp(fx, t):
    """What a camcorder shows at clip second `t`: {lines (the time over the date, those given), indicator ("PLAY",
    "REC" or None), dot (a REC's dot is lit: on for half of each second)}."""
    tm = fx["time"] if fx["clock"] is None else clock_text(fx["clock"], t - fx["from"])
    lines = [x for x in (tm, fx["date"]) if x]
    return {"lines": lines, "indicator": fx["indicator"], "dot": ((t - fx["from"]) % 1.0) < 0.5}


# ================================================================================================= a film's damage
def _rng(*key):
    return np.random.default_rng([int(k) for k in key])


def film_jolt(fx, k):
    """(gain, dx, dy) of film frame k: its exposure (1 +- `flicker`) and where the gate weave has the picture (px at
    1080): a slow wander made of two slow swings each way, plus a little shake from frame to frame."""
    r = _rng(fx["seed"], fx["index"], 3, k)
    ph = _rng(fx["seed"], fx["index"], 4).uniform(0.0, 2.0 * np.pi, 4)
    t = k * film_hold(fx) / fx["frame_rate"]
    w = fx["weave"]
    dx = w * (0.65 * math.sin(2.0 * math.pi * 0.83 * t + ph[0]) + 0.35 * math.sin(2.0 * math.pi * 2.1 * t + ph[1]))
    dy = 0.5 * w * (0.7 * math.sin(2.0 * math.pi * 0.61 * t + ph[2]) + 0.3 * math.sin(2.0 * math.pi * 1.7 * t + ph[3]))
    gain = 1.0 + fx["flicker"] * float(r.uniform(-1.0, 1.0))
    return gain, dx + 0.2 * w * float(r.normal()), dy + 0.1 * w * float(r.normal())


def film_dust(fx, k):
    """The specks on film frame k: [{x, y (frame fractions), r (px at 1080), light (a white speck, else dark), alpha,
    fibre (a short bent fibre: its length px at 1080 and its angle) or None}], about `dust` of them a frame."""
    d = fx["dust"]
    r = _rng(fx["seed"], fx["index"], 5, k)
    out = []
    for _ in range(int(r.poisson(d["count"])) if d["count"] > 0 else 0):
        fib = r.random() < 0.2
        out.append({"x": float(r.random()), "y": float(r.random()), "r": d["size"] * float(r.uniform(0.3, 1.0)),
                    "light": bool(r.random() < 0.5), "alpha": float(r.uniform(0.55, 1.0)),
                    "fibre": (d["size"] * float(r.uniform(3.0, 8.0)), float(r.uniform(0.0, np.pi)),
                              float(r.uniform(-1.0, 1.0))) if fib else None})
    return out


def film_scratches(fx, k):
    """The scratches on film frame k: [{x (frame fractions), y (top, bottom fractions), width (px at 1080), alpha,
    wiggle (a phase), id ((slot, generation): the same scratch while it lasts)}]. Each of `scratches` slots holds a
    scratch for a few film frames (4..12), drifting a little sideways frame to frame, then another somewhere else (now
    and then none for a while)."""
    out = []
    for i in range(fx["scratches"]):
        s = _rng(fx["seed"], fx["index"], 7, i)
        life = int(s.integers(4, 13))
        gen, age = divmod(k + int(s.integers(0, life)), life)
        g = _rng(fx["seed"], fx["index"], 7, i, gen)
        if g.random() < 0.25:
            continue
        x0, v = float(g.uniform(0.04, 0.96)), float(g.normal(0.0, 0.0025))
        y = (0.0, 1.0) if g.random() < 0.6 else tuple(sorted(float(u) for u in g.uniform(0.0, 1.0, 2)))
        j = _rng(fx["seed"], fx["index"], 8, i, gen, age)
        out.append({"x": x0 + v * age + float(j.normal(0.0, 0.0008)), "y": y, "width": float(g.uniform(0.9, 2.2)),
                    "alpha": float(g.uniform(0.45, 0.95)) * float(j.uniform(0.75, 1.0)),
                    "wiggle": float(g.uniform(0.0, 2.0 * np.pi)), "id": (i, gen)})
    return out


def film_hairs(fx, k):
    """The hairs caught in the gate on film frame k: [{side (0 top, 1 right, 2 bottom, 3 left), along (where on that
    side, a fraction of it), points ((n, 2) frame heights from there, x right and y down)}]. In each second of the span
    a hair turns up with the chance `hair`, in from a side of the frame, for 0.4 .. 1.2 s, trembling a little."""
    if fx["hair"] <= 0:
        return []
    t = k * film_hold(fx) / fx["frame_rate"]
    out = []
    for n in range(max(int(math.floor(t)) - 2, 0), int(math.floor(t)) + 1):
        r = _rng(fx["seed"], fx["index"], 9, n)
        if r.random() >= fx["hair"]:
            continue
        t0 = n + float(r.random())
        if not t0 <= t < t0 + float(r.uniform(0.4, 1.2)):
            continue
        side, along, length = int(r.integers(0, 4)), float(r.uniform(0.15, 0.85)), float(r.uniform(0.06, 0.2))
        u = np.linspace(0.0, 1.0, 24)
        bend = np.cumsum(np.interp(u, np.linspace(0.0, 1.0, 4), r.normal(0.0, 0.35, 4))) * (3.0 / len(u))
        tremble = _rng(fx["seed"], fx["index"], 10, n, k).normal(0.0, 0.004, 2)
        step = np.stack([np.cos(bend), np.sin(bend)], 1) * (length / (len(u) - 1))   # (in, across) along the hair
        pts = np.cumsum(step, 0) - step[0] - [0.01, 0.0]                              # from just outside the frame
        d = np.array({0: (0.0, 1.0), 1: (-1.0, 0.0), 2: (0.0, -1.0), 3: (1.0, 0.0)}[side])   # into the frame
        out.append({"side": side, "along": along,
                    "points": pts[:, :1] * d + pts[:, 1:] * np.array([-d[1], d[0]]) + tremble})
    return out


# ================================================================================================= doodles
def _quad(p0, c, p1, n=16):
    u = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - u) ** 2 * np.asarray(p0, float) + 2 * (1 - u) * u * np.asarray(c, float) + u ** 2 * np.asarray(p1, float)


def _ring(cx, cy, r, n=14):
    a = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _close(p, over=0.25):
    """A closed outline as a pen draws it: back to the start and a little on past it."""
    return np.concatenate([p, p[:1], p[:1] + (p[1:2] - p[:1]) * over])


def _sparkle(cx, cy, s):
    """A four-pointed sparkle about (cx, cy), `s` its height: its sides curve in nearly to the middle."""
    tips = np.array([(0.0, -0.5), (0.36, 0.0), (0.0, 0.5), (-0.36, 0.0)])
    path = [_quad(tips[i], 0.06 * np.sign(tips[i] + tips[(i + 1) % 4]), tips[(i + 1) % 4], 10)[:-1] for i in range(4)]
    return _close(np.concatenate(path) * s + (cx, cy), 0.15)


def shape(name):
    """The strokes of a doodle `name` (SHAPES): [(n, 2) points], in a box -0.5..0.5 about its centre, x right, y down."""
    if name == "star":
        a = -np.pi / 2 + np.arange(10) * np.pi / 5
        r = np.where(np.arange(10) % 2 == 0, 0.5, 0.21)
        return [_close(np.stack([r * np.cos(a), r * np.sin(a) + 0.05], 1))]
    if name == "heart":
        t = np.linspace(0.0, 2.0 * np.pi, 72)[:-1]
        x, y = 16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return [_close(np.stack([x / 33.0, (y - 2.5) / 33.0], 1), 0.1)]
    if name == "sparkle":
        return [_sparkle(-0.06, 0.06, 0.88), _sparkle(0.33, -0.33, 0.3)]
    if name == "x":
        return [_quad((-0.42, -0.42), (0.04, -0.06), (0.43, 0.42)), _quad((0.42, -0.44), (0.03, 0.05), (-0.43, 0.41))]
    if name == "swirl":
        th = np.linspace(0.0, 2.6 * 2.0 * np.pi, 120)
        r = 0.48 * th / th[-1]
        return [np.stack([r * np.cos(th), r * np.sin(th)], 1)]
    if name == "scribble":
        u = np.linspace(0.0, 8.0 * np.pi, 160)
        return [np.stack([-0.36 + 0.72 * u / u[-1] - 0.11 * np.sin(u), 0.17 * np.cos(u) - 0.06 * u / u[-1]], 1)]
    if name == "arrow":
        tip, ctrl = np.array([0.42, -0.24]), np.array([-0.06, -0.3])
        d = (tip - ctrl) / np.linalg.norm(tip - ctrl)
        head = [tip - 0.22 * (np.array([[c, -s], [s, c]]) @ d) for c, s in
                ((math.cos(0.5), math.sin(0.5)), (math.cos(-0.5), math.sin(-0.5)))]
        return [_quad((-0.46, 0.34), ctrl, tip, 20), np.array([head[0], tip, head[1]])]
    if name == "crown":
        body = np.array([(-0.42, 0.34), (0.42, 0.34), (0.46, -0.22), (0.2, 0.04), (0.0, -0.36), (-0.2, 0.04),
                         (-0.46, -0.22)])
        return [_close(body, 0.3), np.array([(-0.4, 0.2), (0.4, 0.2)]), _ring(-0.46, -0.31, 0.05),
                _ring(0.0, -0.45, 0.05), _ring(0.46, -0.31, 0.05)]
    if name == "cat_ears":                                       # two rounded ears, tipped outward, a fold in each
        ear = np.concatenate([_quad((-0.5, 0.24), (-0.5, -0.08), (-0.37, -0.32), 12),
                              _quad((-0.37, -0.32), (-0.18, -0.2), (-0.05, 0.12), 12)[1:]])
        inner = np.concatenate([_quad((-0.41, 0.13), (-0.42, -0.06), (-0.35, -0.17), 8),
                                _quad((-0.35, -0.17), (-0.25, -0.1), (-0.17, 0.05), 8)[1:]])
        flip = np.array([-1.0, 1.0])
        return [ear, inner, ear[::-1] * flip, inner[::-1] * flip]
    raise ValueError(f"no doodle shape {name!r} (shapes: {', '.join(SHAPES)})")


def resample(p, step):
    """Points every `step` along the polyline `p` (its ends kept)."""
    p = np.asarray(p, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p[:1]
    u = np.linspace(0.0, s[-1], max(int(math.ceil(s[-1] / step)), 1) + 1)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)


def boil(strokes, rng, wobble, turn=3.0, grow=0.03):
    """The strokes as a hand draws them again: every point moved by a smooth random wobble of about `wobble` (in the
    strokes' own units, bending every WOBBLE_EVERY of them), the whole turned by up to `turn` degrees and scaled by up
    to `grow`. `rng` a numpy Generator: one boil step's draw."""
    a = math.radians(float(rng.uniform(-turn, turn)))
    k = 1.0 + float(rng.uniform(-grow, grow))
    rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) * k     # counter-clockwise on screen
    out = []
    for st in strokes:
        p = resample(st, 0.02)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) if len(p) > 1 else np.zeros(1)
        m = max(int(math.ceil(s[-1] / WOBBLE_EVERY)) + 1, 2)
        knots = rng.normal(0.0, wobble, (m, 2))
        u = np.linspace(0.0, max(s[-1], 1e-9), m)
        p = p + np.stack([np.interp(s, u, knots[:, 0]), np.interp(s, u, knots[:, 1])], 1)
        out.append(p @ rot.T)
    return out


def turned(strokes, angle):
    """The strokes turned `angle` degrees counter-clockwise on screen (y down)."""
    a = math.radians(angle)
    rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    return [np.asarray(p, float) @ rot.T for p in strokes]


def boil_step(fx, t):
    """The boil step (the doodles are drawn again `fps` times a second) at clip second `t`."""
    return max(int(math.floor((t - fx["from"]) * fx["fps"] + 1e-9)), 0)


# ================================================================================================= the booth's strip
def booth_layout(fx, size):
    """(columns, rows) of a booth's photos in a frame of `size` (width, height): its `layout`, else a column in a tall
    frame, a row in a wide one and as square a block as fits `count` in between."""
    if fx["layout"] is not None:
        return tuple(fx["layout"])
    w, h = size
    n = fx["count"]
    if h > 1.25 * w:
        return 1, n
    if w > 1.25 * h:
        return n, 1
    c = int(math.ceil(math.sqrt(n)))
    return c, int(math.ceil(n / c))


def booth_strip(fx, size, px):
    """The strip of a booth in a frame of `size` (width, height; px the frame's px per px at 1080): {cols, rows, photo
    (w, h) px, border px, foot px (under the photos: room for the caption), strip (w, h) px, cells [(x, y) of each photo's
    top left in the strip, row by row]}. The strip, tilted `tilt` degrees, is as large as fits `size` x the frame."""
    cols, rows = booth_layout(fx, size)
    b = fx["border"] * px
    ratio = fx["ratio"]
    foot = 0.34 if fx["caption"] else 0.1                     # the strip's foot, in photo heights
    # strip w = cols pw + (cols + 1) b, h = (rows + foot) pw / ratio + (rows + 1) b; its tilted box inside the room
    a = math.radians(abs(fx["tilt"]))
    ca, sa = math.cos(a), math.sin(a)
    sw1, sw0 = cols, (cols + 1) * b
    sh1, sh0 = (rows + foot) / ratio, (rows + 1) * b
    room_w, room_h = fx["size"] * size[0], fx["size"] * size[1]
    pw = min((room_w - sw0 * ca - sh0 * sa) / (sw1 * ca + sh1 * sa),
             (room_h - sw0 * sa - sh0 * ca) / (sw1 * sa + sh1 * ca))
    pw = max(pw, 4.0)
    ph = pw / ratio
    sw, sh = cols * pw + (cols + 1) * b, rows * ph + (rows + 1) * b + foot * ph
    cells = [(b + i * (pw + b), b + j * (ph + b)) for j in range(rows) for i in range(cols)][:fx["count"]]
    return {"cols": cols, "rows": rows, "photo": (pw, ph), "border": b, "foot": foot * ph, "strip": (sw, sh),
            "cells": cells}


def booth_doodles(fx, cell, photo, t_capture):
    """The doodles round photo `cell` (its number) of a booth: [{shape, at (x, y) px in the photo, size px, colour,
    angle, t (clip seconds: each pops in a moment after the photo lands)}], `doodles` of them, near its edges and
    corners (the middle is the face), seeded."""
    r = _rng(fx["seed"], fx["index"], 13, cell)
    pw, ph = photo
    out = []
    for j in range(fx["doodles"]):
        a = r.uniform(0.0, 2.0 * np.pi)
        rad = r.uniform(0.33, 0.44)
        out.append({"shape": STICKERS[int(r.integers(0, len(STICKERS)))],
                    "at": (pw * (0.5 + rad * math.cos(a)), ph * (0.5 + rad * math.sin(a))),
                    "size": ph * float(r.uniform(0.17, 0.27)), "colour": fx["stickers"][int(r.integers(0, len(ACCENTS)))],
                    "angle": float(r.uniform(-25.0, 25.0)), "t": t_capture + fx["drop"] + 0.09 * (j + 1)})
    return out
