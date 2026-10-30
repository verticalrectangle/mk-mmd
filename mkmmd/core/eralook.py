"""Era looks, bpy-free (docs/design.md: Frame effects, Era looks): the `[[fx]]` kinds that put the picture in another
era of pop or on the air, checked, timed and laid out here; mkmmd/eralook.py draws them.

    digicam   a 2005 point-and-shoot: barrel distortion, purple fringes round the highlights, the camera's own low
              resolution, its white balance off and hunting, a hard digital tone curve, CCD noise, oversharpened halos,
              JPEG's 8 x 8 blocks and an orange date stamp in seven-segment digits
    chrome    now, hyperpop: an iridescent chrome sheen (on the figure alone with `mask = "figure"`), glossy bloom,
              twinkling sparkle stars and floating Y2K bubbles
    gradmap   the picture through a gradient map (its luma onto a list of colours), the next map of `maps` on each cut
              of the cut list (or on each hit): the same shot comes back regraded
    invert    the picture inverted (its hues turned back toward the palette with `hue`), eased in and out, for a window
              or on hits, never more than three big flashes a second
    testcard  an homage to the layout of 1967's Test Card F: the scene in a circle centred on a point, colour bars,
              castellations, a grey scale and gratings round it on a ruled grey ground, small labels
    cards     TV colour-bar cards (mkmmd.core.colorbars) flying in on hits, tumbling in 3D

`ERAS` are presets of the existing picture effects (mkmmd.core.picturefx): `era = "1967"` makes a `film` a Technicolor
print, `era = "1985"` makes a `camcorder` a VHS tape; any key given beside `era` wins over the preset. Times are clip
seconds; the randomness is seeded per effect, frame or flight: the same seed draws the same frames the same way, in any
order."""
import math

import numpy as np

from . import colorbars as CB

KINDS = ("digicam", "chrome", "gradmap", "invert", "testcard", "cards")
HITS = {"times", "lane", "min_db", "every"}
CORNERS = ("bottom-right", "bottom-left", "top-left", "top-right")
PATHS = ("in", "across")                 # a card's flight: out of the depth at the camera and past it, or across
FLASH_RATE = 3                           # big flashes a second at most (WCAG 2.3.1, the general flash threshold)
FLASH_LEVEL = 0.5                        # an invert past this is a big flash (its picture's light and dark swapped)
SPARKLE_LIFE = 0.7                       # seconds: a sparkle star's twinkle, from nothing to nothing
BURST = 0.45                             # seconds: the sparkles a hit sets off, fading
SWING = (0.15, 0.4)                      # chrome: a hit swings the sheen this far (in bands) and back over this long (s)
LENS = 1.3                               # the cards' camera: focal length in frame heights (about 42 degrees down)
CARD_RATIO = 4.0 / 3.0                   # a card's width over its height: a TV's
CARD_THICK = 0.03                        # a card's thickness, in its heights
CARD_GAP = 0.06                          # seconds between the cards one hit sets off
LABEL_MAX = 28                           # characters in a test card's label

MAPS = {"teal_lime": ["#0f2a33", "pine", "foam", "#a8cf5a", "#e6f2c8", "#fbfdf2"],     # dark to light, the lights
        "pink_purple": ["#2b1a3f", "#6c4a8a", "love", "rose", "#f5d5d8", "#fdf4f2"],  # going to the palette's cream
        "sepia_orange": ["#2a1a12", "#7b4a2d", "#d8732f", "gold", "#f5dfb4", "#fdf6ea"],
        "blue_gold": ["#13203f", "pine", "#8aa6c8", "gold", "#f6e2b0", "#fff9ec"]}
NONE_MAP = "none"                        # a gradmap's entry for the picture as it is

ERAS = {"film": {"1967": {"fps": 24.0, "contrast": 1.3, "saturation": 1.35, "dye": 1.0, "halation": 0.35,
                          "grain": {"amount": 0.035, "size": 1.3}, "flicker": 0.03, "weave": 1.2,
                          "dust": {"count": 2, "size": 2.5}, "scratches": 1, "hair": 0.08, "vignette": 0.22}},
        "camcorder": {"1985": {"date": "JUL. 4 1985", "time": "PM 3:12", "tick": True, "indicator": "PLAY",
                               "tracking": {"amount": 0.3, "speed": 0.18}, "chroma": 9.0, "scanlines": 0.2,
                               "jitter": 1.5, "soft": 2.0, "saturation": 0.7, "noise": 0.06, "switching": 0.7}}}

KEYS = {"digicam": {"noise", "sharpen", "halo", "jpeg", "pixel", "fringe", "barrel", "wb", "hunt", "contrast", "date",
                    "corner", "color", "size", "seed"},
        "chrome": {"mask", "objects", "sheen", "iris", "bands", "colors", "bloom", "sparkles", "bubbles", "speed",
                   "seed"} | HITS,
        "gradmap": {"maps", "offset", "mask", "objects"} | HITS,
        "invert": {"hue", "in", "hold", "out", "rate", "mask", "objects"} | HITS,
        "testcard": {"center", "at", "radius", "zoom", "grid", "bars", "labels", "font", "ground", "line", "open"},
        "cards": {"count", "size", "life", "spin", "path", "style", "shadow", "seed"} | HITS}
DEFAULTS = {"digicam": {"noise": 0.05, "sharpen": 1.2, "halo": 2.0, "jpeg": 45, "pixel": 2.0, "fringe": 0.6,
                        "barrel": 0.04, "wb": -0.3, "hunt": 0.25, "contrast": 1.2, "date": "'05 10 28",
                        "corner": "bottom-right", "color": "#ff7a1a", "size": 0.045, "seed": 1},
            "chrome": {"mask": None, "sheen": 1.0, "iris": 0.6, "bands": 2.5, "colors": ["rose", "foam", "iris", "gold"],
                       "bloom": 0.5, "sparkles": 8, "bubbles": 5, "speed": 0.1, "seed": 1, "min_db": None,
                       "every": 1},
            "gradmap": {"maps": ["teal_lime", "pink_purple", "sepia_orange", "blue_gold"], "offset": 0, "mask": None,
                        "min_db": None, "every": 1},
            "invert": {"hue": 0.0, "in": 0.05, "hold": 0.12, "out": 0.18, "rate": 3.0, "mask": None, "min_db": None,
                       "every": 1},
            "testcard": {"center": [0.5, 0.5], "at": [0.5, 0.5], "radius": 0.36, "zoom": 1.0, "grid": 9,
                         "bars": "dawn", "labels": ["PLEASE STAND BY"], "font": None, "ground": "muted",
                         "line": "base", "open": 0.0},
            "cards": {"count": 1, "size": 0.32, "life": 1.0, "spin": 1.25, "path": "in", "style": "both",
                      "shadow": 0.3, "seed": 1, "min_db": None, "every": 1}}

# ================================================================================================= seven segments
SEG_W, SEG_T, SEG_GAP, SEG_SPACE = 0.58, 0.13, 0.025, 0.18   # a digit's width, a segment's thickness, the gaps: heights
SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
            "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g"}
SEG_CHARS = "0123456789-'/.: "


def _hseg(xa, xb, y, t):
    return np.array([(xa, y), (xa + t / 2, y - t / 2), (xb - t / 2, y - t / 2), (xb, y), (xb - t / 2, y + t / 2),
                     (xa + t / 2, y + t / 2)], float)


def _vseg(x, ya, yb, t):
    return np.array([(x, ya), (x + t / 2, ya + t / 2), (x + t / 2, yb - t / 2), (x, yb), (x - t / 2, yb - t / 2),
                     (x - t / 2, ya + t / 2)], float)


def _digit(segs):
    t, g = SEG_T, SEG_GAP
    left, right, top, mid, bottom = t / 2, SEG_W - t / 2, t / 2, 0.5, 1.0 - t / 2
    shapes = {"a": _hseg(left + g, right - g, top, t), "b": _vseg(right, top + g, mid - g, t),
              "c": _vseg(right, mid + g, bottom - g, t), "d": _hseg(left + g, right - g, bottom, t),
              "e": _vseg(left, mid + g, bottom - g, t), "f": _vseg(left, top + g, mid - g, t),
              "g": _hseg(left + g, right - g, mid, t)}
    return [shapes[s] for s in segs]


def glyphs(text):
    """The seven-segment digits of a date stamp: ([polygon (n, 2)], width), in a box one unit tall (x right, y down from
    its top), upright (the drawing slants them). Digits, `-`, `'`, `/`, `.`, `:` and spaces."""
    polys, x = [], 0.0
    t = SEG_T
    for ch in text:
        if ch in SEGMENTS:
            polys += [p + (x, 0.0) for p in _digit(SEGMENTS[ch])]
            w = SEG_W
        elif ch == "'":
            polys.append(np.array([(0.0, 0.0), (t, 0.0), (t * 0.7, 0.3), (0.0, 0.3)]) + (x, 0.0))
            w = t
        elif ch == "/":
            polys.append(np.array([(0.0, 1.0), (t * 1.2, 1.0), (SEG_W, 0.0), (SEG_W - t * 1.2, 0.0)]) + (x, 0.0))
            w = SEG_W
        elif ch == ".":
            polys.append(np.array([(0.0, 1.0 - t), (t, 1.0 - t), (t, 1.0), (0.0, 1.0)]) + (x, 0.0))
            w = t
        elif ch == ":":
            for y in (0.28, 0.72):
                polys.append(np.array([(0.0, y - t / 2), (t, y - t / 2), (t, y + t / 2), (0.0, y + t / 2)]) + (x, 0.0))
            w = t
        elif ch == " ":
            w = SEG_W * 0.5
        else:
            raise ValueError(f"no seven-segment glyph for {ch!r} (have {SEG_CHARS!r})")
        x += w + SEG_SPACE
    return polys, max(x - SEG_SPACE, 0.0)


# ================================================================================================= checks
def normalize(spec, out, palette, what):
    """An era look's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    {"digicam": _digicam, "chrome": _chrome, "gradmap": _gradmap, "invert": _invert, "testcard": _testcard,
     "cards": _cards}[out["kind"]](FX, spec, out, palette, what)


def era(kind, spec, what):
    """The preset an entry's `era` picks for its kind (ERAS), with `era` itself: defaults the entry's own keys override;
    {} for a kind without eras or an entry without one."""
    if kind not in ERAS or spec.get("era") is None:
        return {}
    from . import framefx as FX
    name = spec["era"]
    name = str(int(name)) if isinstance(name, (int, float)) and not isinstance(name, bool) and name == int(name) \
        else name
    if name not in ERAS[kind]:
        raise FX.FxError(f"{what}: era {spec['era']!r} is none of {', '.join(ERAS[kind])}")
    return {**ERAS[kind][name], "era": name}


def _corner(FX, out, what):
    if out["corner"] not in CORNERS:
        raise FX.FxError(f"{what}: corner {out['corner']!r} is none of {', '.join(CORNERS)}")


def _digicam(FX, spec, out, palette, what):
    out["noise"] = FX._num(out["noise"], f"{what} noise", 0.0, 1.0)
    out["sharpen"] = FX._num(out["sharpen"], f"{what} sharpen", 0.0, 5.0)
    out["halo"] = FX._num(out["halo"], f"{what} halo", lo=0.1)
    out["jpeg"] = FX._num(out["jpeg"], f"{what} jpeg", 1, 100, integer=True)
    out["pixel"] = FX._num(out["pixel"], f"{what} pixel", 1.0, 16.0)
    out["fringe"] = FX._num(out["fringe"], f"{what} fringe", 0.0, 1.0)
    out["barrel"] = FX._num(out["barrel"], f"{what} barrel", -0.3, 0.3)
    out["wb"] = FX._num(out["wb"], f"{what} wb", -1.0, 1.0)
    out["hunt"] = FX._num(out["hunt"], f"{what} hunt", 0.0, 1.0)
    out["contrast"] = FX._num(out["contrast"], f"{what} contrast", 0.0, 3.0)
    d = out["date"]
    if d is False or d is None:
        out["date"] = None
    elif not isinstance(d, str) or not d.strip() or len(d) > 16 or any(ch not in SEG_CHARS for ch in d):
        raise FX.FxError(f"{what}: date = {d!r}: expected a short stamp of digits and {SEG_CHARS[10:]!r} (\"'05 10 28\", "
                         f"\"2005/10/28\") or false")
    _corner(FX, out, what)
    out["color"] = FX._colour(out["color"], palette, f"{what} color")
    out["size"] = FX._num(out["size"], f"{what} size", 0.005, 0.3)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


def _ramp(FX, v, palette, what, lo=2, hi=8):
    if isinstance(v, str) or not isinstance(v, (list, tuple)) or not lo <= len(v) <= hi:
        raise FX.FxError(f"{what} = {v!r}: expected a list of {lo} to {hi} colours (palette slots, #hex)")
    return [FX._colour(c, palette, what) for c in v]


def _chrome(FX, spec, out, palette, what):
    FX._mask(spec, out, what)
    out["sheen"] = FX._num(out["sheen"], f"{what} sheen", 0.0, 1.0)
    out["iris"] = FX._num(out["iris"], f"{what} iris", 0.0, 1.0)
    out["bands"] = FX._num(out["bands"], f"{what} bands", 0.0, 12.0)
    out["colors"] = _ramp(FX, out["colors"], palette, f"{what} colors", 1, 8)
    out["bloom"] = FX._num(out["bloom"], f"{what} bloom", 0.0, 2.0)
    out["sparkles"] = FX._num(out["sparkles"], f"{what} sparkles", 0, 64, integer=True)
    out["bubbles"] = FX._num(out["bubbles"], f"{what} bubbles", 0, 32, integer=True)
    out["speed"] = FX._num(out["speed"], f"{what} speed")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._hits(spec, out, what, required=False)


def _gradmap(FX, spec, out, palette, what):
    maps = out["maps"]
    if isinstance(maps, str) or not isinstance(maps, (list, tuple)) or not maps:
        raise FX.FxError(f"{what}: maps = {maps!r}: expected a list of maps (each a name: {', '.join(MAPS)}, "
                         f"{NONE_MAP!r}, or a list of 2 to 8 colours, dark to light)")
    res = []
    for j, m in enumerate(maps):
        if m == NONE_MAP:
            res.append(None)
        elif isinstance(m, str):
            if m not in MAPS:
                raise FX.FxError(f"{what}: maps[{j}] = {m!r} is none of {', '.join(MAPS)}, {NONE_MAP!r} (or give a "
                                 f"list of colours)")
            res.append(_ramp(FX, MAPS[m], palette, f"{what} maps[{j}]"))
        else:
            res.append(_ramp(FX, m, palette, f"{what} maps[{j}]"))
    out["maps"] = res
    out["offset"] = FX._num(out["offset"], f"{what} offset", lo=0, integer=True)
    FX._mask(spec, out, what)
    FX._hits(spec, out, what, required=False)
    out["cuts"] = []                     # set by timed(): the frames shots of the cut start on inside the span


def _invert(FX, spec, out, palette, what):
    out["hue"] = FX._num(out["hue"], f"{what} hue", -360.0, 360.0)
    for k in ("in", "hold", "out"):
        out[k] = FX._num(out[k], f"{what} {k}", 0.0, 10.0)
    out["rate"] = FX._num(out["rate"], f"{what} rate", 0.1, float(FLASH_RATE))
    FX._mask(spec, out, what)
    FX._hits(spec, out, what, required=False)
    if not out.get("times") and out.get("lane") is None and out["in"] + out["out"] > out["to"] - out["from"] + 1e-9:
        raise FX.FxError(f"{what}: in + out ({out['in'] + out['out']:g} s) is longer than the window "
                         f"({out['to'] - out['from']:g} s)")


def _labels(FX, v, what):
    v = [v] if isinstance(v, str) else v
    if v is None or v is False:
        return []
    if not isinstance(v, (list, tuple)) or len(v) > 4 or \
            not all(isinstance(s, str) and s.strip() and len(s) <= LABEL_MAX for s in v):
        raise FX.FxError(f"{what}: labels = {v!r}: expected up to 4 short texts (1..{LABEL_MAX} characters each)")
    return [s.strip() for s in v]


def _testcard(FX, spec, out, palette, what):
    out["center"] = FX._center(out["center"], f"{what} center")
    out["at"] = FX._pair(out["at"], f"{what} at")
    if not all(0.0 <= x <= 1.0 for x in out["at"]):
        raise FX.FxError(f"{what}: at = {out['at']}: [x, y] in 0..1 of the frame")
    out["radius"] = FX._num(out["radius"], f"{what} radius", 0.05, 0.5)
    out["zoom"] = FX._num(out["zoom"], f"{what} zoom", 0.1, 10.0)
    out["grid"] = FX._num(out["grid"], f"{what} grid", 4, 24, integer=True)
    if out["bars"] not in CB.STYLES:
        raise FX.FxError(f"{what}: bars {out['bars']!r} is none of {', '.join(CB.STYLES)}")
    out["labels"] = _labels(FX, out["labels"], what)
    if out["font"] is not None and (not isinstance(out["font"], str) or not out["font"].strip()):
        raise FX.FxError(f"{what}: font = {out['font']!r}: expected a font slug of the asset registry or a font file")
    out["ground"] = FX._colour(out["ground"], palette, f"{what} ground")
    out["line"] = FX._colour(out["line"], palette, f"{what} line")
    out["open"] = FX._num(out["open"], f"{what} open", 0.0, out["to"] - out["from"])
    out["inks"] = _card_inks(palette, out["bars"])


def _card_inks(palette, bars):
    """The colours a test card is printed in, resolved from the palette: its rows of bars, its white and black, its
    accents (the castellations)."""
    white, black = CB.ends(palette)
    dawn = CB.colours(palette, "dawn")
    rows = {"dawn": [dawn], "true": [CB.colours(palette, "true")], "both": [dawn, CB.colours(palette, "true")]}[bars]
    rnd = lambda c: [round(float(x), 6) for x in c]              # noqa: E731
    return {"rows": [[rnd(c) for c in r] for r in rows], "white": rnd(white), "black": rnd(black),
            "accents": [rnd(c) for c in dawn[1:]]}


def _cards(FX, spec, out, palette, what):
    FX._hits(spec, out, what)
    out["count"] = FX._num(out["count"], f"{what} count", 1, 12, integer=True)
    out["size"] = FX._num(out["size"], f"{what} size", 0.02, 2.0)
    out["life"] = FX._num(out["life"], f"{what} life", 0.1, 10.0)
    out["spin"] = FX._num(out["spin"], f"{what} spin", -8.0, 8.0)
    if out["path"] not in PATHS:
        raise FX.FxError(f"{what}: path {out['path']!r} is none of {', '.join(PATHS)}")
    if out["style"] not in CB.STYLES:
        raise FX.FxError(f"{what}: style {out['style']!r} is none of {', '.join(CB.STYLES)}")
    out["shadow"] = FX._num(out["shadow"], f"{what} shadow", 0.0, 1.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["palette"] = {k: v for k, v in palette.items() if isinstance(v, str)}


# ================================================================================================= timing
def frame_of(t, fps, frame0):
    return int(round(frame0 + float(t) * fps))


def smooth(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def timed(fx, data, fps, frame0):
    """What an era look needs of the project beyond its own keys, at plan time: a gradmap's `cuts` (the frames the
    shots of the cut start on inside its span); an invert over a window has its flashes checked (FLASH_RATE a second
    at most)."""
    from . import framefx as FX
    if fx["kind"] == "gradmap":
        starts = sorted({frame_of(s["from"], fps, frame0) for s in (data.get("shot") or [])
                         if isinstance(s, dict) and not s.get("plate") and "from" in s})
        fx["cuts"] = [g for g in starts if fx["f0"] < g < fx["f1"]]
    elif fx["kind"] == "invert" and not fx.get("times") and fx.get("lane") is None:
        ts = [(g - frame0) / fps for g in range(fx["f0"], fx["f1"])]
        rises = flashes([level(fx, t) * min(max(FX.strength(fx, t), 0.0), 1.0) for t in ts], ts)
        worst = busiest(rises)
        if worst[0] > FLASH_RATE:
            raise FX.FxError(f"fx {fx['index']} (invert): it flashes {worst[0]} times in the second from "
                             f"{worst[1]:.2f} s: at most {FLASH_RATE} big flashes a second (key `strength` slower)")


def map_step(fx, frame, times, fps, frame0):
    """The map a gradmap shows on `frame` (its index into `maps`, round and round): `offset` plus the cuts that came after
    the span's first frame up to `frame`, or with hits (`times`, clip seconds) the hits that did."""
    marks = [frame_of(h, fps, frame0) for h in times] if (fx.get("times") or fx.get("lane")) else fx["cuts"]
    return (fx["offset"] + sum(1 for g in marks if fx["f0"] < g <= frame)) % len(fx["maps"])


def thin(times, rate=FLASH_RATE):
    """The hits an invert keeps: in time order, a hit is skipped while `rate` kept hits began in the second before it,
    so that no more than `rate` flashes start in any second."""
    kept = []
    for h in sorted(times):
        if sum(1 for k in kept if h - k < 1.0) < rate:
            kept.append(float(h))
    return kept


def pulse(fx, age):
    """An invert's level `age` seconds after a hit: eased in over `in`, held `hold`, eased out over `out`."""
    if age < 0.0:
        return 0.0
    rise = smooth(age / fx["in"]) if fx["in"] > 0 else 1.0
    a = age - fx["in"] - fx["hold"]
    fall = (1.0 - smooth(a / fx["out"]) if fx["out"] > 0 else float(a < 0)) if a > 0 else 1.0
    return rise * fall


def level(fx, t, kept=None):
    """How inverted the picture is at clip second `t`, 0..1, before the strength: with hits (`kept`, thin()'s) the
    strongest of their pulses, else the window eased in over `in` from `from` and out over `out` to `to`."""
    if kept is not None:
        return max((pulse(fx, t - h) for h in kept if t >= h), default=0.0)
    a = smooth((t - fx["from"]) / fx["in"]) if fx["in"] > 0 else float(t >= fx["from"])
    b = smooth((fx["to"] - t) / fx["out"]) if fx["out"] > 0 else float(t < fx["to"])
    return a * b


def flashes(values, times, at=FLASH_LEVEL):
    """The times a level rises through `at` (from at or below it to above): each a big flash, for it comes back."""
    out, low = [], True
    for v, t in zip(values, times):
        if low and v > at:
            out.append(float(t))
        low = v <= at
    return out


def busiest(onsets, span=1.0):
    """(n, t0): the most `onsets` (seconds) in any `span` seconds, and where that run starts."""
    best = (0, onsets[0] if onsets else 0.0)
    for i, t in enumerate(onsets):
        n = sum(1 for u in onsets[i:] if u - t < span)
        if n > best[0]:
            best = (n, t)
    return best


def wb_gains(fx, t):
    """A digicam's white balance at clip second `t`: (r, g, b) gains, its shift `wb` (below 0 cool and a little green,
    above warm) hunting about it by up to `hunt` as auto white balance does, slowly."""
    r = np.random.default_rng([fx["seed"], fx.get("index", 0), 51])
    p1, p2 = r.uniform(0.0, 2.0 * math.pi, 2)
    u = t - fx["from"]
    s = fx["wb"] + fx["hunt"] * 0.6 * (math.sin(2.0 * math.pi * 0.17 * u + p1) + 0.5 * math.sin(2.0 * math.pi * 0.43 * u + p2))
    s = min(max(s, -1.0), 1.0)
    if s >= 0.0:
        return (1.0 + 0.16 * s, 1.0 + 0.03 * s, 1.0 - 0.24 * s)
    a = -s
    return (1.0 - 0.18 * a, 1.0 + 0.04 * a, 1.0 + 0.2 * a)


def sheen_phase(fx, t, hits):
    """How far a chrome's sheen has moved at clip second `t`, in bands: on by `speed` bands a second, and on each hit
    since the span's start swung SWING[0] bands forward and back over SWING[1] seconds (a shimmer that settles)."""
    a, d = SWING
    return fx["speed"] * (t - fx["from"]) + a * sum(math.sin(math.pi * (t - h) / d) for h in hits
                                                    if fx["from"] <= h <= t < h + d)


def sparkles(fx, t, hits):
    """The sparkle stars of a chrome at clip second `t`: [{x, y (0..1 across and down the figure's box, or the
    frame), size (0..1 of their full size), angle (degrees)}]. Each twinkles from nothing to nothing over about
    SPARKLE_LIFE seconds and comes back elsewhere; a hit sets them all off at once, fading over BURST seconds."""
    out = []
    n, seed, i = fx["sparkles"], fx["seed"], fx.get("index", 0)
    for j in range(n):
        r = np.random.default_rng([seed, i, 61, j])
        life = SPARKLE_LIFE * r.uniform(0.75, 1.3)
        u = (t - fx["from"]) / life + r.uniform(0.0, 1.0)
        k = int(math.floor(u))
        q = np.random.default_rng([seed, i, 62, j, k])
        x, y, a = q.uniform(0.05, 0.95), q.uniform(0.05, 0.95), q.uniform(0.0, 45.0)
        out.append({"x": float(x), "y": float(y), "size": float(math.sin(math.pi * (u - k)) ** 2), "angle": float(a)})
    last = [h for h in hits if h <= t]
    if last and t - last[-1] < BURST:
        h, b = last[-1], 1.0 - (t - last[-1]) / BURST
        for j in range(n):
            q = np.random.default_rng([seed, i, 63, j, int(round(h * 1000))])
            out.append({"x": float(q.uniform(0.05, 0.95)), "y": float(q.uniform(0.05, 0.95)),
                        "size": float(b * q.uniform(0.7, 1.2)), "angle": float(q.uniform(0.0, 45.0))})
    return out


def bubbles(fx, t):
    """A chrome's Y2K bubbles at clip second `t`: [{x, y (frame fractions), r (short sides), turn (0..1, where its
    sheen has come round to)}], each drifting up at its own speed and wobbling, coming in again at the foot."""
    out = []
    for j in range(fx["bubbles"]):
        r = np.random.default_rng([fx["seed"], fx.get("index", 0), 71, j])
        x0, speed, size, y0, ph = r.uniform(0.06, 0.94), r.uniform(0.05, 0.12), r.uniform(0.025, 0.06), \
            r.uniform(0.0, 1.0), r.uniform(0.0, 1.0)
        u = t - fx["from"]
        span = 1.0 + 4.0 * size
        y = (y0 - speed * u) % span - 2.0 * size
        x = x0 + 0.02 * math.sin(2.0 * math.pi * (0.45 * u + ph))
        out.append({"x": float(x), "y": float(y), "r": float(size), "turn": float((0.15 * u + ph) % 1.0)})
    return out


def open_share(fx, t):
    """How far a test card has blown open at clip second `t`, 0..1: over its last `open` seconds the circle grows
    (slowly, then fast) until it covers the frame."""
    if fx["open"] <= 0:
        return 0.0
    u = (t - (fx["to"] - fx["open"])) / fx["open"]
    u = min(max(u, 0.0), 1.0)
    return u ** 3


def card_layout(size, fx):
    """Where a test card's parts lie in a frame (or a rect's part of one) of `size` (w, h), px (pixel i spanning
    i..i+1): {s (the short side), period (the grid's squares), edge (the castellations' depth), circle (cx, cy, r),
    bars [(x0, y0, x1, y1) per row], wedge [box per step, black to white], gratings [box per grating, coarse to fine],
    corners [box], label (cx, cy, cap height): the first label's box, on the circle's foot}. The grey scale and the
    gratings flank the circle left and right in a frame as wide as it is tall or wider, above and below it (the gratings
    under the label) in a tall one, as wide as the room there allows; with no room for them they are left out."""
    w, h = size
    s = float(min(w, h))
    p = s / fx["grid"]
    e = 0.42 * p
    cx, cy, r = fx["at"][0] * w, fx["at"][1] * h, fx["radius"] * s
    rows = len(fx["inks"]["rows"])
    bar_h = 0.8 * p / rows
    y = e + 0.12 * p
    bars = []
    for _ in range(rows):
        bars.append((e + 0.12 * p, y, w - e - 0.12 * p, y + bar_h))
        y += bar_h
    inner = (e + 0.12 * p, bars[-1][3] + 0.15 * p, w - e - 0.12 * p, h - e - 0.12 * p)
    cap = 0.24 * p
    foot = cy + r + 1.2 * cap + 1.0                          # the first label's box ends here
    n, step = 6, 0.5 * p
    wedge, grat = [], []
    if w >= h:
        room = min(cx - r - inner[0], inner[2] - cx - r)
        gap = min(0.35 * p, 0.25 * room)
        bw = min(0.85 * p, room - 2.0 * gap)
        if bw >= 0.3 * p:
            wx, gx, top = cx - r - gap - bw, cx + r + gap, cy - n * step / 2.0
            wedge = [(wx, top + k * step, wx + bw, top + (k + 1) * step) for k in range(n)]
            grat = [(gx, top + k * step, gx + bw, top + (k + 1) * step) for k in range(n)]
    else:
        room = min(cy - r - inner[1], inner[3] - foot)
        gap = min(0.35 * p, 0.25 * room)
        bh = min(0.85 * p, room - 2.0 * gap)
        if bh >= 0.3 * p:
            wy, gy, left = cy - r - gap - bh, foot + gap, cx - n * step / 2.0
            wedge = [(left + k * step, wy, left + (k + 1) * step, wy + bh) for k in range(n)]
            grat = [(left + k * step, gy, left + (k + 1) * step, gy + bh) for k in range(n)]
    c = 0.8 * p
    corners = [(inner[0], inner[1], inner[0] + c, inner[1] + c), (inner[2] - c, inner[1], inner[2], inner[1] + c),
               (inner[0], inner[3] - c, inner[0] + c, inner[3]), (inner[2] - c, inner[3] - c, inner[2], inner[3])]
    return {"s": s, "period": p, "edge": e, "circle": (cx, cy, r), "bars": bars, "wedge": wedge, "gratings": grat,
            "corners": corners, "label": (cx, cy + r, cap)}


# ================================================================================================= flying cards
def _rotation(axis, theta):
    """The 3 x 3 rotation by `theta` radians about the unit `axis` (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def card_starts(fx, times):
    """[(clip second, card number)]: the flights the hits (`times`) set off, `count` cards a hit CARD_GAP s apart."""
    return [(float(h) + c * CARD_GAP, i * fx["count"] + c) for i, h in enumerate(sorted(times))
            for c in range(fx["count"])]


def card_pose(fx, j, age, aspect):
    """(centre (x, y, z), rotation 3 x 3) of card `j` `age` seconds into its flight, in the cards' camera's space: x right,
    y up, z away from the lens, a point showing at (LENS x / z, LENS y / z) frame heights from the frame's middle (a frame
    `aspect` wide for one high). A card of `size` shows `size` frame heights tall at the depth LENS. `in`: from far behind
    the middle of the frame at the lens, growing, veering off past an edge; `across`: from beyond one side to beyond the
    other at a middle distance, on a gentle arc. It turns `spin` turns a second about its own axis."""
    r = np.random.default_rng([fx["seed"], fx.get("index", 0), 91, j])
    e = min(max(age / fx["life"], 0.0), 1.0)
    if fx["path"] == "in":
        u0, v0 = r.uniform(-0.25, 0.25) * min(aspect, 1.0), r.uniform(-0.15, 0.15)
        ang = r.uniform(0.0, 2.0 * math.pi)
        reach = 0.5 * math.hypot(aspect, 1.0) + 0.6
        u1, v1 = reach * math.cos(ang), 0.8 * reach * math.sin(ang)
        z0, z1 = 7.0, 0.45
        z = z0 * (z1 / z0) ** e
        k = e ** 1.5
        u, v = u0 + (u1 - u0) * k, v0 + (v1 - v0) * k
    else:
        side = 1.0 if r.integers(0, 2) else -1.0
        z = r.uniform(1.0, 2.2)
        v0, arc = r.uniform(-0.3, 0.3), r.uniform(-0.12, 0.12)
        edge = 0.5 * aspect + 0.6 * fx["size"] * LENS / z * CARD_RATIO
        u, v = side * (-edge + 2.0 * edge * e), v0 + arc * 4.0 * e * (1.0 - e)
    axis = r.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    theta = r.uniform(0.0, 2.0 * math.pi) + 2.0 * math.pi * fx["spin"] * r.uniform(0.8, 1.2) * age
    return np.array([u * z / LENS, v * z / LENS, z]), _rotation(axis, theta)


def cards_at(fx, t, times, aspect):
    """The cards in flight at clip second `t`: [(j, age, centre, rotation)], the farthest first (drawn first)."""
    out = []
    for t0, j in card_starts(fx, times):
        age = t - t0
        if 0.0 <= age < fx["life"]:
            c, R = card_pose(fx, j, age, aspect)
            out.append((j, age, c, R))
    out.sort(key=lambda it: -it[2][2])
    return out


def card_corners(fx, centre, R):
    """The 8 corners of a card's slab (centre and rotation from card_pose), camera space: the front face's four
    (top left, top right, bottom right, bottom left as its bars read) then the back's behind them."""
    hh = 0.5 * fx["size"]
    hw, ht = hh * CARD_RATIO, 0.5 * CARD_THICK * fx["size"]
    local = np.array([(-hw, hh, -ht), (hw, hh, -ht), (hw, -hh, -ht), (-hw, -hh, -ht),
                      (-hw, hh, ht), (hw, hh, ht), (hw, -hh, ht), (-hw, -hh, ht)])
    return local @ np.asarray(R).T + np.asarray(centre)
