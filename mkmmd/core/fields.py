"""Colour fields, bpy-free (docs/design.md: Frame effects, Colour fields): the `[[fx]]` kinds that paint the picture with
a field of colour, checked and timed here; mkmmd/fields.py paints them.

    liquid  a 1960s oil-and-water light show: translucent inks between glass flowing, merging and parting, their edges lit
            by the light they bend, oil cells and bubbles in them; it swells on hits
    spiral  op-art: a spiral, rings, a moire, a sunburst or a warped checkerboard in 2 to 4 colours, turning or flowing,
            stepping on hits
    flood   a colour blooming out from a point to fill the frame, ink dropped in water, then receding or fading

A collage's `back` may be a pattern too, `{pattern = "spiral", colors = [...], ...}` with the spiral kind's own keys
(`backs`), painted where its flat colour would be and stepping on the collage's own hits.

A pattern is a stripe coordinate q over the picture: the colour of a pixel is colors[floor(q) % len(colors)], so `speed`
(stripes a second) turns a spiral or a sunburst and makes rings, a moire or a checkerboard flow, and `step` (stripes) is
how far a hit moves it. Times are clip seconds.
"""
import math

from . import tween as TW

KINDS = ("liquid", "spiral", "flood")
PATTERNS = ("spiral", "rings", "moire", "sunburst", "checker")
MOIRES = ("rings", "lines")                      # a moire's two sets: rings about two centres, or lines at an angle
BLENDS = ("over", "screen", "multiply", "soft-light")
HITS = {"times", "lane", "min_db", "every"}
STEP = 0.12                                      # seconds: a pattern's step on a hit, eased out (a snap)
ATTACK = 0.05                                    # seconds: a liquid's swell on a hit rises this fast, then decays
FILL_EASE = 1.6                                  # a flood's front: share = 1 - (1 - u) ** FILL_EASE, fast then slowing
PATTERN = {"pattern": "spiral", "colors": ["rose", "base"], "speed": 0.5, "step": 1.0, "center": [0.5, 0.5]}
PATTERN_OWN = {"spiral": {"arms": 5, "twist": 1.0},                  # each pattern's own keys and their defaults
               "rings": {"size": 0.05},
               "moire": {"size": 0.018, "moire": "rings", "offset": 4.0, "angle": 6.0},
               "sunburst": {"arms": 12},
               "checker": {"size": 0.1, "warp": 0.6, "angle": 0.0}}
_OWN = sorted({k for own in PATTERN_OWN.values() for k in own})      # every pattern's own keys
_LAY = {"blend", "opacity", "mask", "objects"}   # how a field is laid on the picture: blended, and kept to a part of it
KEYS = {"liquid": _LAY | HITS | {"inks", "light", "scale", "speed", "pulse", "decay", "cells", "bubbles", "center",
                                 "seed"},
        "spiral": _LAY | HITS | set(PATTERN) | set(_OWN),
        "flood": {"mask", "objects", "color", "center", "grow", "hold", "recede", "fade", "soft", "wobble", "seed"} | HITS}
DEFAULTS = {"liquid": {"inks": ["love", "gold", "foam", "iris"], "light": "base", "blend": "over", "opacity": 1.0,
                       "scale": 1.0, "speed": 1.0, "pulse": 0.15, "decay": 0.45, "cells": 0.35, "bubbles": 6,
                       "center": [0.5, 0.5], "seed": 1, "mask": None, "min_db": None, "every": 1},
            "spiral": {**PATTERN, **{k: None for k in _OWN}, "blend": "over", "opacity": 1.0, "mask": None,
                       "min_db": None, "every": 1},
            "flood": {"color": "love", "center": [0.5, 0.5], "grow": 0.8, "hold": 0.4, "recede": 0.6, "fade": 0.0,
                      "soft": 24.0, "wobble": 0.5, "seed": 1, "mask": None, "min_db": None, "every": 1}}


def normalize(spec, out, palette, what):
    """A colour field's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    {"liquid": _liquid, "spiral": _spiral, "flood": _flood}[out["kind"]](FX, spec, out, palette, what)


def _lay(FX, spec, out, what):
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    out["opacity"] = FX._num(out["opacity"], f"{what} opacity", 0.0, 1.0)
    FX._mask(spec, out, what)


def _liquid(FX, spec, out, palette, what):
    inks = out["inks"]
    if isinstance(inks, str) or not isinstance(inks, (list, tuple)) or not 1 <= len(inks) <= 6:
        raise FX.FxError(f"{what}: inks = {inks!r}: expected a list of 1 to 6 colours (palette slots, #hex)")
    out["inks"] = [FX._colour(c, palette, f"{what} inks") for c in inks]
    out["light"] = FX._colour(out["light"], palette, f"{what} light")
    _lay(FX, spec, out, what)
    out["scale"] = FX._num(out["scale"], f"{what} scale", 0.2, 5.0)
    out["speed"] = FX._num(out["speed"], f"{what} speed", 0.0, 20.0)
    out["pulse"] = FX._num(out["pulse"], f"{what} pulse", 0.0, 1.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", lo=1e-3)
    out["cells"] = FX._num(out["cells"], f"{what} cells", 0.0, 1.0)
    out["bubbles"] = FX._num(out["bubbles"], f"{what} bubbles", 0, 40, integer=True)
    out["center"] = FX._center(out["center"], f"{what} center")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._hits(spec, out, what, required=False)


def _spiral(FX, spec, out, palette, what):
    given = {k: spec[k] for k in set(PATTERN) | set(_OWN) if k in spec and k != "center"}
    out.update(_pattern(FX, {"pattern": out["pattern"], **given}, palette, what))
    for k in _OWN:                       # the keys of other patterns are not the entry's
        if k not in PATTERN_OWN[out["pattern"]]:
            out.pop(k, None)
    out["center"] = FX._center(out["center"], f"{what} center")
    _lay(FX, spec, out, what)
    FX._hits(spec, out, what, required=False)


def _pattern(FX, v, palette, what):
    """A pattern {pattern, colors, speed, step, its own keys} -> with PATTERN's and its own defaults, colours resolved.
    A key of another pattern is an error that says whose it is. `center` is left to the caller."""
    p = v.get("pattern")
    if p not in PATTERNS:
        raise FX.FxError(f"{what}: pattern {p!r} is none of {', '.join(PATTERNS)}")
    own = PATTERN_OWN[p]
    alien = sorted(k for k in set(v) & set(_OWN) if k not in own)
    if alien:
        whose = ["{} (a key of {})".format(k, " or ".join(q for q in PATTERNS if k in PATTERN_OWN[q])) for k in alien]
        raise FX.FxError(f"{what}: {', '.join(whose)} does not shape a {p}: it takes {', '.join(sorted(own))}")
    f = {**PATTERN, **own, **{k: x for k, x in v.items() if x is not None}}
    cs = f["colors"]
    if isinstance(cs, str) or not isinstance(cs, (list, tuple)) or not 2 <= len(cs) <= 4:
        raise FX.FxError(f"{what}: colors = {cs!r}: expected a list of 2 to 4 colours")
    out = {"pattern": p, "colors": [FX._colour(c, palette, f"{what} colors") for c in cs],
           "speed": FX._num(f["speed"], f"{what} speed", -60.0, 60.0),
           "step": FX._num(f["step"], f"{what} step", -16.0, 16.0)}
    if "arms" in own:
        out["arms"] = FX._num(f["arms"], f"{what} arms", 1, 64, integer=True)
    if "twist" in own:
        out["twist"] = FX._num(f["twist"], f"{what} twist", -20.0, 20.0)
    if "size" in own:
        out["size"] = FX._num(f["size"], f"{what} size", 0.002, 2.0)
    if p == "moire":
        if f["moire"] not in MOIRES:
            raise FX.FxError(f"{what}: moire {f['moire']!r} is none of {', '.join(MOIRES)} (two ring sets or two of "
                             f"lines)")
        out["moire"] = f["moire"]
        out["offset"] = FX._num(f["offset"], f"{what} offset", 0.0, 100.0)
    if "angle" in own:
        out["angle"] = FX._num(f["angle"], f"{what} angle", -180.0, 180.0)
    if "warp" in own:
        out["warp"] = FX._num(f["warp"], f"{what} warp", 0.0, 1.0)
    return out


def backs(v, palette, what):
    """A collage's `back`: a colour, a pattern ({pattern, colors, ...}: the spiral kind's own keys, its `center` [x, y]
    fractions of the box it fills), "picture" (the frame as it reached the effect), or a list of them -> [rgb, pattern
    or None (the picture), ...]."""
    from . import framefx as FX
    one = isinstance(v, (str, dict)) or (isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))
    items = [v] if one else v
    if not isinstance(items, (list, tuple)) or not items:
        raise FX.FxError(f"{what} = {v!r}: expected a colour, a pattern ({{pattern = \"spiral\", colors = [...]}}), "
                         f"\"picture\" or a list of them")
    out = []
    for j, b in enumerate(items):
        if b == FX.PICTURE:
            out.append(None)
            continue
        if not isinstance(b, dict):
            out.append(FX._colour(b, palette, what))
            continue
        w = f"{what} {j}" if len(items) > 1 else what
        extra = sorted(set(b) - set(PATTERN) - set(_OWN))
        if extra:
            raise FX.FxError(f"{w}: unknown keys {extra} in a pattern back (known: {sorted(set(PATTERN) | set(_OWN))})")
        if "pattern" not in b:
            raise FX.FxError(f"{w}: a pattern back names its `pattern` ({', '.join(PATTERNS)})")
        p = _pattern(FX, b, palette, w)
        p["center"] = FX._pair(b.get("center", PATTERN["center"]), f"{w} center")
        out.append(p)
    return out


def _flood(FX, spec, out, palette, what):
    out["color"] = FX._colours(out["color"], palette, f"{what} color")
    out["center"] = FX._center(out["center"], f"{what} center")
    FX._hits(spec, out, what)
    for k in ("grow", "hold", "recede", "fade"):
        out[k] = FX._num(out[k], f"{what} {k}", lo=0.0)
    if out["grow"] <= 0:
        raise FX.FxError(f"{what}: grow = 0: the seconds the colour takes to fill the frame, more than 0")
    out["soft"] = FX._num(out["soft"], f"{what} soft", 0.0, 400.0)
    out["wobble"] = FX._num(out["wobble"], f"{what} wobble", 0.0, 1.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._mask(spec, out, what)


# ================================================================================================= timing
def phase(spec, t, t0, steps):
    """The stripe offset of a pattern at clip second `t` of a span from `t0`: `speed` stripes a second, and `step`
    stripes on each of `steps` (clip seconds) gone by, each eased out over STEP seconds."""
    q = spec["speed"] * (t - t0)
    for h in steps:
        if t > h:
            q += spec["step"] * TW.ease((t - h) / STEP, "out")
    return q


def flood_drops(fx, t, times):
    """The drops of a flood on at clip second `t`: [(k, share, alpha)] in the order they fell (`times`, clip seconds),
    k the drop's number (its colour is `color` entry k, round and round), `share` how far its front is on its way to
    fill the frame (0 at the point, 1 the whole frame) and `alpha` its opacity. A front runs out over `grow` seconds,
    fast and slowing (FILL_EASE), the frame stays full for `hold`, then it draws back to the point over `recede`
    (eased in and out) and / or fades over `fade`; with neither it stays."""
    out = []
    for k, h in enumerate(sorted(times)):
        a = t - h
        if a < 0:
            continue
        share, alpha = 1.0 - (1.0 - min(a / fx["grow"], 1.0)) ** FILL_EASE, 1.0
        b = a - fx["grow"] - fx["hold"]
        if b > 0:
            if fx["recede"] > 0:
                share *= 1.0 - TW.ease(b / fx["recede"], "inout")
            if fx["fade"] > 0:
                alpha = 1.0 - TW.clamp01(b / fx["fade"])
        if share > 0 and alpha > 0:
            out.append((k, share, alpha))
    return out


def swell(t, times, decay):
    """0..1 of a liquid's swell at clip second `t`: each hit (`times`) rises over ATTACK seconds and dies away as
    exp(-age / decay); the strongest counts."""
    out = 0.0
    for h in times:
        a = t - h
        if a < 0:
            continue
        rise = 1.0 - (1.0 - min(a / ATTACK, 1.0)) ** 2
        out = max(out, rise * math.exp(-max(a - ATTACK, 0.0) / decay))
    return out
