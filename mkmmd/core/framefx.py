"""Frame effects, bpy-free (docs/design.md: Shots, Frame effects): the `[[fx]]` entries of a project checked and timed.

Each entry works on the finished pictures of a span of the cut, `from`..`to` (clip seconds), after the transitions, inserts
and glitches and before the screen type and the grade (`mk post`, previewed by `mk look`); unlike those they may overlap, and
they are applied in the order of the table. The picture work is mkmmd/framefx.py.

    lens      a point mass bends the picture round a point: an Einstein ring, the far side folded in
    echo      copies of the frames before, fading: a figure leaves Muybridge echoes
    slitscan  each column (or row) is a frame of its own age, up to `depth` frames back: time smeared across the frame
    push      the picture zooms into a disc about a point (a pupil) that shows the clip's frame at `inside`, until it fills
              the frame at `to`: the seam of a loop
    scope     a stem of the song drawn as an oscilloscope's XY trace: the signal against itself a moment later
    zoom      a punch: on each hit (`times`, or a `lane` of the [audio] hits file) the picture jumps in by `amount` about a
              point and eases back
    shock     on each hit a ring runs out from a point, bending the picture as a pressure wave would, its front lit
    slab      a slanted bar of colour slams in from one side, rests, and slides on out the far side: Persona's panels
    tint      the picture mapped through a gradient by its brightness (dark, colour, light): the Dark Hour's green
    halftone  comic print: the shadows screened with Ben-Day dots, the colour plates printed out of register
    split     a broken digital signal: bands of the frame torn sideways, the red and blue planes pulled apart
    mosh      a datamosh: the frame before the span carried along the cut's own motion in blocks, smeared over the new
              shot as a codec does when the key frame of a cut is lost

The collages cut a figure (`objects`, drawn as its coverage beside the frame by `mk render`) out of the picture and lay it
on flat colour, its picture or a flat silhouette (`fill`); on each hit (`times`, or a `lane`, as a zoom's) they step on:
a list of colours, fills or sizes is read at the number of hits so far, plus the number of the cell, copy, panel or
region it paints, so a list of two colours is a chessboard that flips on every hit.

    grid      the frame shrunk into `cells` x `cells`, the figure in each on a chessboard of `back`; a `piece` (a flat
              figure in one cell) hops a knight's move on each hit
    orbit     the figure moved to `center`, `scale` times its size, and `count` copies of it on a tilted ring round it,
              turning
    mirror    the frame folded about a vertical (`axes` "x"), horizontal ("y") or both ("xy") line through `center`: the
              half with the figure in it (moved to `place` first) and its reflections, each region its own colours
    panels    the frame cut into slanted panels across its long side, the figure in each, `delay` frames apart
    stack     the figure in front of a stack of flat copies of it (`bands`), each `step` further and `scale` as large: a
              banded echo, or with a `delay` the poses before
"""
import hashlib

import numpy as np

from . import lanes as LN
from . import shotstyle as SS

FIGURES = ("grid", "orbit", "mirror", "panels", "stack")         # the collages: a figure cut out onto flat colour
KINDS = ("lens", "echo", "slitscan", "push", "scope", "zoom", "shock", "slab", "tint", "halftone", "split",
         "mosh") + FIGURES
PICTURE = "picture"                                     # a collage's fill: the figure as the frame shows it
BLENDS = ("lighten", "darken")
EDGES = ("left", "right", "top", "bottom")
EASES = ("log", "smooth", "linear")
STEMS = ("drums", "bass", "other", "vocals", "mix")
_COMMON = {"kind", "from", "to", "strength"}
_HITS = {"times", "lane", "min_db", "every"}            # zoom and shock fire on these, a collage steps on them
_FIGURE = {"objects", "back", "fill", "edge"} | _HITS   # a collage's figure, its colours and the hits it steps on
KEYS = {"lens": _COMMON | {"center", "radius", "radius_m", "shadow"},
        "echo": _COMMON | {"count", "every", "fade", "blend", "tint"},
        "slitscan": _COMMON | {"depth", "now"},
        "push": (_COMMON - {"strength"}) | {"center", "radius", "radius_m", "inside", "play", "ease"},
        "scope": _COMMON | {"stem", "at", "size", "color", "width", "glow", "window", "lag", "gain", "timeline"},
        "zoom": _COMMON | _HITS | {"center", "amount", "attack", "decay"},
        "shock": _COMMON | _HITS | {"center", "speed", "width", "amount", "life", "color", "glow"},
        "slab": _COMMON | {"at", "size", "slant", "angle", "color", "alpha", "dir", "in", "out", "shutter"},
        "tint": _COMMON | {"color", "dark", "light"},
        "halftone": _COMMON | {"cell", "angle", "level", "ink", "lift", "offset"},
        "split": _COMMON | {"bands", "shift", "share", "rgb", "hold", "seed"},
        "mosh": _COMMON | {"block"},
        "grid": _COMMON | _FIGURE | {"cells", "piece", "seed"},
        "orbit": _COMMON | _FIGURE | {"center", "count", "radius", "tilt", "scale", "size", "turn", "spin"},
        "mirror": _COMMON | _FIGURE | {"center", "axes", "place"},
        "panels": _COMMON | _FIGURE | {"count", "slant", "size", "delay"},
        "stack": _COMMON | _FIGURE | {"count", "step", "scale", "center", "bands", "delay"}}
_FIGURE_DEFAULTS = {"back": "base", "fill": PICTURE, "edge": None, "min_db": None, "every": 1}
DEFAULTS = {"lens": {"radius": 0.1, "shadow": 0.0},
            "echo": {"count": 4, "every": 3, "fade": 0.55, "blend": "lighten", "tint": None},
            "slitscan": {"depth": 24, "now": "right"},
            "push": {"radius": 0.02, "inside": 0.0, "play": False, "ease": "log"},
            "scope": {"stem": "bass", "at": [0.5, 0.5], "size": 0.3, "color": "text", "width": 2.0, "glow": 0.6,
                      "window": 0.04, "lag": 0.0025, "gain": 1.0, "timeline": "audio/timeline.json"},
            "zoom": {"center": [0.5, 0.5], "amount": 0.06, "attack": 0.02, "decay": 0.16, "min_db": None, "every": 1},
            "shock": {"center": [0.5, 0.5], "speed": 1.6, "width": 0.035, "amount": 0.012, "life": 0.55,
                      "color": "text", "glow": 0.6, "min_db": None, "every": 1},
            "slab": {"at": [0.5, 0.5], "size": [1.2, 0.18], "slant": 18.0, "angle": -8.0, "color": "love", "alpha": 1.0,
                     "dir": "right", "in": 0.1, "out": 0.12, "shutter": 1.0},
            "tint": {"color": "#46E07A", "dark": "#000000", "light": "#FFFFFF"},
            "halftone": {"cell": 9.0, "angle": 45.0, "level": 0.5, "ink": "#000000", "lift": 0.35, "offset": [3.0, 2.0]},
            "split": {"bands": 14, "shift": 0.08, "share": 0.35, "rgb": 10.0, "hold": 0.05, "seed": 1},
            "mosh": {"block": 16.0},
            "grid": {**_FIGURE_DEFAULTS, "cells": 3, "piece": None, "seed": 1},
            "orbit": {**_FIGURE_DEFAULTS, "center": [0.5, 0.5], "count": 6, "radius": 0.36, "tilt": 60.0, "scale": 1.0,
                      "size": 0.6, "turn": 30.0, "spin": 0.0},
            "mirror": {**_FIGURE_DEFAULTS, "center": [0.5, 0.5], "axes": "x", "place": None},
            "panels": {**_FIGURE_DEFAULTS, "count": 3, "slant": 12.0, "size": 1.0, "delay": 0},
            "stack": {**_FIGURE_DEFAULTS, "count": 5, "step": [-0.02, -0.006], "scale": 1.0, "center": [0.5, 0.5],
                      "bands": ["text", "base"], "delay": 0}}
SLAB_DIRS = ("right", "left", "up", "down")            # the way a slab travels: in from the other side, out this way
AXES = ("x", "y", "xy")                                 # a mirror's fold: about a vertical line, a horizontal one, both


class FxError(ValueError):
    """A bad `[[fx]]` entry: an unknown kind or key, a value out of range, a span that does not fit the clip."""


def frame_of(t, fps, frame0):
    """The frame of clip second `t` (the build's rounding)."""
    return int(round(frame0 + float(t) * fps))


def _num(v, what, lo=None, hi=None, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and int(v) != v):
        raise FxError(f"{what} = {v!r}: expected {'a whole number' if integer else 'a number'}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        bounds = f"{lo:g}" if hi is None else f"{lo:g}..{hi:g}" if lo is not None else f"at most {hi:g}"
        raise FxError(f"{what} = {v!r}: expected {'at least ' if hi is None else ''}{bounds}")
    return int(v) if integer else float(v)


def _pair(v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise FxError(f"{what} = {v!r}: expected [x, y]")
    return [_num(x, what) for x in v]


def _keys(v, what):
    """`strength`: a number (constant) or [[t, v], ...] clip seconds, linear between keys, held outside them."""
    if not isinstance(v, (list, tuple)):
        return [[0.0, _num(v, what)]]
    if not v or not all(isinstance(k, (list, tuple)) and len(k) == 2 for k in v):
        raise FxError(f"{what}: expected a number or [[t, value], ...]")
    keys = sorted([_num(t, what), _num(x, what)] for t, x in v)
    if any(a[0] == b[0] for a, b in zip(keys, keys[1:])):
        raise FxError(f"{what}: two keys at the same second")
    return keys


def _center(v, what):
    """`center`: [x, y] fractions of the frame (0, 0 the top left) or {point = expr}, a point of the scene (`mk q`
    expression) seen through the camera of the shot in the cut on every frame."""
    if isinstance(v, dict):
        if set(v) != {"point"} or not isinstance(v["point"], str) or not v["point"].strip():
            raise FxError(f"{what} = {v!r}: expected {{point = \"<expression>\"}} or [x, y]")
        return {"mode": "point", "expr": v["point"]}
    return {"mode": "frame", "at": _pair(v, what)}


def _colour(v, palette, what):
    try:
        return [round(float(c), 6) for c in SS.resolve_colour(v, palette)]
    except SS.StyleError as e:
        raise FxError(f"{what}: {e}") from None


def normalize(spec, palette, what="fx"):
    """One `[[fx]]` entry with its kind's defaults and its colours resolved (display-space floats)."""
    if not isinstance(spec, dict):
        raise FxError(f"{what} {spec!r}: expected a table")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise FxError(f"{what}: kind {kind!r} is none of {', '.join(KINDS)}")
    what = f"{what} ({kind})"
    extra = sorted(set(spec) - KEYS[kind])
    if extra:
        raise FxError(f"{what}: unknown keys {extra} (known: {sorted(KEYS[kind])})")
    for k in ("from", "to"):
        if k not in spec:
            raise FxError(f"{what}: `{k}` (clip seconds) is required")
    out = {**DEFAULTS[kind], **spec}
    out["from"], out["to"] = _num(spec["from"], f"{what} from"), _num(spec["to"], f"{what} to")
    if out["to"] <= out["from"]:
        raise FxError(f"{what}: `to` ({out['to']:g}) must come after `from` ({out['from']:g})")
    out["strength"] = _keys(spec.get("strength", 1.0), f"{what} strength")
    if kind in ("lens", "push"):
        if "center" not in spec:
            raise FxError(f"{what}: `center` is required ([x, y] of the frame or {{point = \"<expression>\"}})")
        out["center"] = _center(spec["center"], f"{what} center")
        if "radius_m" in spec:
            if "radius" in spec:
                raise FxError(f"{what}: give `radius` (a share of the frame height) or `radius_m` (metres), not both")
            if out["center"]["mode"] != "point":
                raise FxError(f"{what}: `radius_m` measures at a scene point's depth: give center = {{point = ...}}")
            out["radius_m"] = _num(spec["radius_m"], f"{what} radius_m", lo=1e-6)
            out.pop("radius")
        else:
            out["radius"] = _num(out["radius"], f"{what} radius", lo=1e-6)
    if kind == "lens":
        out["shadow"] = _num(out["shadow"], f"{what} shadow", 0.0, 1.0)
    elif kind == "echo":
        out["count"] = _num(out["count"], f"{what} count", lo=1, integer=True)
        out["every"] = _num(out["every"], f"{what} every", lo=1, integer=True)
        out["fade"] = _num(out["fade"], f"{what} fade", 0.0, 1.0)
        if out["blend"] not in BLENDS:
            raise FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
        out["tint"] = None if out["tint"] is None else _colour(out["tint"], palette, f"{what} tint")
    elif kind == "slitscan":
        out["depth"] = _num(out["depth"], f"{what} depth", lo=1, integer=True)
        if out["now"] not in EDGES:
            raise FxError(f"{what}: now {out['now']!r} is none of {', '.join(EDGES)} (the edge that shows the present)")
    elif kind == "push":
        out["inside"] = _num(out["inside"], f"{what} inside", lo=0.0)
        if not isinstance(out["play"], bool):
            raise FxError(f"{what}: play is true or false")
        if out["ease"] not in EASES:
            raise FxError(f"{what}: ease {out['ease']!r} is none of {', '.join(EASES)}")
    elif kind == "scope":
        if out["stem"] not in STEMS:
            raise FxError(f"{what}: stem {out['stem']!r} is none of {', '.join(STEMS)}")
        out["at"] = _pair(out["at"], f"{what} at")
        for k, lo in (("size", 1e-6), ("width", 1e-6), ("window", 1e-4), ("lag", 0.0), ("gain", 1e-6)):
            out[k] = _num(out[k], f"{what} {k}", lo=lo)
        out["glow"] = _num(out["glow"], f"{what} glow", 0.0, 1.0)
        out["color"] = _colour(out["color"], palette, f"{what} color")
    elif kind in ("zoom", "shock"):
        _hits(spec, out, what)
        out["center"] = _center(out["center"], f"{what} center")
        if kind == "zoom":
            out["amount"] = _num(out["amount"], f"{what} amount", -0.5, 1.0)
            out["attack"] = _num(out["attack"], f"{what} attack", lo=0.0)
            out["decay"] = _num(out["decay"], f"{what} decay", lo=1e-4)
        else:
            for k, lo in (("speed", 1e-6), ("width", 1e-6), ("amount", 0.0), ("life", 1e-4), ("glow", 0.0)):
                out[k] = _num(out[k], f"{what} {k}", lo=lo)
            out["color"] = _colour(out["color"], palette, f"{what} color")
    elif kind == "slab":
        out["at"] = _pair(out["at"], f"{what} at")
        out["size"] = _pair(out["size"], f"{what} size")
        if min(out["size"]) <= 0:
            raise FxError(f"{what}: size = {out['size']}: [width, height] in frame heights, both positive")
        out["slant"] = _num(out["slant"], f"{what} slant", -60.0, 60.0)
        out["angle"] = _num(out["angle"], f"{what} angle")
        out["alpha"] = _num(out["alpha"], f"{what} alpha", 0.0, 1.0)
        out["shutter"] = _num(out["shutter"], f"{what} shutter", 0.0, 4.0)
        out["color"] = _colour(out["color"], palette, f"{what} color")
        if out["dir"] not in SLAB_DIRS:
            raise FxError(f"{what}: dir {out['dir']!r} is none of {', '.join(SLAB_DIRS)} (the way it travels)")
        for k in ("in", "out"):
            out[k] = _num(out[k], f"{what} {k}", lo=0.0)
        if out["in"] + out["out"] > out["to"] - out["from"] + 1e-9:
            raise FxError(f"{what}: in + out ({out['in'] + out['out']:g} s) is longer than the slab is on "
                          f"({out['to'] - out['from']:g} s)")
    elif kind == "tint":
        for k in ("color", "dark", "light"):
            out[k] = _colour(out[k], palette, f"{what} {k}")
    elif kind == "halftone":
        out["cell"] = _num(out["cell"], f"{what} cell", lo=2.0)
        out["angle"] = _num(out["angle"], f"{what} angle")
        out["level"] = _num(out["level"], f"{what} level", 0.01, 1.0)
        out["lift"] = _num(out["lift"], f"{what} lift", 0.0, 1.0)
        out["offset"] = _pair(out["offset"], f"{what} offset")
        out["ink"] = _colour(out["ink"], palette, f"{what} ink")
    elif kind == "split":
        out["bands"] = _num(out["bands"], f"{what} bands", 1, 400, integer=True)
        out["shift"] = _num(out["shift"], f"{what} shift", 0.0, 1.0)
        out["share"] = _num(out["share"], f"{what} share", 0.0, 1.0)
        out["rgb"] = _num(out["rgb"], f"{what} rgb", lo=0.0)
        out["hold"] = _num(out["hold"], f"{what} hold", lo=1e-3)
        out["seed"] = _num(out["seed"], f"{what} seed", lo=0, integer=True)
    elif kind == "mosh":
        out["block"] = _num(out["block"], f"{what} block", lo=1.0)
    elif kind in FIGURES:
        _figure(spec, out, palette, what)
    return out


def _hits(spec, out, what, required=True):
    """A zoom's or a shock's hits (a collage's steps): `times` (clip seconds) or a `lane` of the [audio] hits file, picked
    at `every` / `min_db` when the frames are made (FrameFx, which has the file). A collage may give neither: `times` is
    then empty and it never steps."""
    if not required and "times" not in spec and "lane" not in spec:
        if "min_db" in spec or "every" in spec:
            raise FxError(f"{what}: min_db and every pick from a lane: give the `lane`")
        out["times"] = []
        return
    if ("times" in spec) == ("lane" in spec):
        raise FxError(f"{what}: give `times` (clip seconds) or `lane` (one of the [audio] hits file), one of them")
    if "times" in spec:
        t = spec["times"]
        if not isinstance(t, (list, tuple)) or not t:
            raise FxError(f"{what}: times is a list of clip seconds")
        if "min_db" in spec or "every" in spec:
            raise FxError(f"{what}: min_db and every pick from a lane; give the times you want")
        out["times"] = sorted(_num(x, f"{what} times") for x in t)
        return
    try:
        c = LN.choice({k: spec.get(k, out.get(k)) for k in ("lane", "min_db", "every")}, what)
    except LN.LaneError as e:
        raise FxError(str(e)) from None
    out.update(lane=c["lane"], min_db=c["min_db"], every=c["every"])


def _colours(v, palette, what, picture=False):
    """A colour or a list of them -> [rgb, ...]; with `picture`, "picture" (the figure as the frame shows it) may stand
    for a colour, as None. Three numbers are one colour."""
    one = isinstance(v, str) or (isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))
    items = [v] if one else v
    if not isinstance(items, (list, tuple)) or not items:
        kinds = 'a colour or "picture"' if picture else "a colour"
        raise FxError(f"{what} = {v!r}: expected {kinds}, or a list of them")
    return [None if picture and c == PICTURE else _colour(c, palette, what) for c in items]


def _figure(spec, out, palette, what):
    """A collage's keys: the figure (`objects`), its colours and fills, an edge, the hits it steps on, its layout."""
    if "objects" not in spec:
        raise FxError(f"{what}: `objects` is required: the figure, object patterns (`name*`, `@collection`, `prop:key`)")
    objs = [spec["objects"]] if isinstance(spec["objects"], str) else spec["objects"]
    if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
        raise FxError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                      f"`prop:key`), a list or one")
    out["objects"] = list(objs)
    out["back"] = _colours(out["back"], palette, f"{what} back")
    out["fill"] = _colours(out["fill"], palette, f"{what} fill", picture=True)
    e = out["edge"]
    if e is not None:
        if not isinstance(e, dict) or "color" not in e or set(e) - {"color", "width"}:
            raise FxError(f"{what}: edge = {e!r}: expected {{color, width}} (width in px at 1080 on the short side)")
        out["edge"] = {"color": _colour(e["color"], palette, f"{what} edge"),
                       "width": _num(e.get("width", 3.0), f"{what} edge width", lo=0.0)}
    _hits(spec, out, what, required=False)
    kind = out["kind"]
    if kind == "grid":
        cells = out["cells"] if isinstance(out["cells"], (list, tuple)) else [out["cells"]]
        if not cells:
            raise FxError(f"{what}: cells is a whole number (per side) or a list of them")
        out["cells"] = [_num(c, f"{what} cells", 1, 16, integer=True) for c in cells]
        out["piece"] = None if out["piece"] is None else _colour(out["piece"], palette, f"{what} piece")
        out["seed"] = _num(out["seed"], f"{what} seed", lo=0, integer=True)
    elif kind == "orbit":
        out["center"] = _center(out["center"], f"{what} center")
        out["count"] = _num(out["count"], f"{what} count", 1, 32, integer=True)
        out["radius"] = _num(out["radius"], f"{what} radius", lo=0.0)
        out["tilt"] = _num(out["tilt"], f"{what} tilt", 0.0, 89.0)
        out["scale"] = _num(out["scale"], f"{what} scale", 0.25, 4.0)
        out["size"] = _num(out["size"], f"{what} size", lo=1e-3)
        out["turn"] = _num(out["turn"], f"{what} turn")
        out["spin"] = _num(out["spin"], f"{what} spin")
    elif kind == "mirror":
        out["center"] = _center(out["center"], f"{what} center")
        axes = [out["axes"]] if isinstance(out["axes"], str) else out["axes"]
        if not isinstance(axes, (list, tuple)) or not axes or any(a not in AXES for a in axes):
            raise FxError(f"{what}: axes = {out['axes']!r}: one of {', '.join(AXES)}, or a list of them")
        out["axes"] = list(axes)
        out["place"] = None if out["place"] is None else _pair(out["place"], f"{what} place")
    elif kind == "panels":
        out["count"] = _num(out["count"], f"{what} count", 2, 12, integer=True)
        out["slant"] = _num(out["slant"], f"{what} slant", -45.0, 45.0)
        out["size"] = _num(out["size"], f"{what} size", lo=1e-3)
        out["delay"] = _num(out["delay"], f"{what} delay", lo=0, integer=True)
    else:
        out["count"] = _num(out["count"], f"{what} count", 1, 32, integer=True)
        out["step"] = _pair(out["step"], f"{what} step")
        out["scale"] = _num(out["scale"], f"{what} scale", 0.25, 2.0)
        out["center"] = _center(out["center"], f"{what} center")
        out["bands"] = _colours(out["bands"], palette, f"{what} bands", picture=True)
        out["delay"] = _num(out["delay"], f"{what} delay", lo=0, integer=True)


def hit_times(fx, hits):
    """The clip seconds a zoom or a shock fires on (a collage steps on): its `times`, or its lane's hits from `hits`
    ({lane: [...]}, the [audio] hits file) inside its span. Raises FxError when it names a lane and there is no file or no
    such lane."""
    if fx.get("times") is not None:
        return list(fx["times"])
    if hits is None:
        raise FxError(f"fx {fx.get('index')} ({fx['kind']}): lane {fx['lane']!r} needs the [audio] hits file "
                      f"(`mk timeline hits` writes it)")
    try:
        return LN.pick(hits, fx["lane"], fx["min_db"], fx["every"], fx["from"], fx["to"])
    except LN.LaneError as e:
        raise FxError(f"fx {fx.get('index')} ({fx['kind']}): {e}") from None


def punch(t, times, attack, decay):
    """0..1 of a zoom punch at clip second(s) `t`: each hit rises over `attack` (ease-out) and dies away exp(-age /
    decay); the strongest of them counts (hits do not stack)."""
    t = np.asarray(t, float)
    out = np.zeros(t.shape)
    for h in times:
        a = t - h
        rise = np.clip(a / attack, 0.0, 1.0) if attack > 0 else (a >= 0).astype(float)
        rise = 1.0 - (1.0 - rise) ** 2
        out = np.maximum(out, np.where(a >= 0, rise * np.exp(-np.maximum(a - attack, 0.0) / decay), 0.0))
    return out


def slab_offset(fx, t):
    """How far along its way (in shares of its run in, 0 at rest, -1 fully off the side it came from; and out, 0 to 1
    off the far side) a slab is at clip second `t`: in with an ease-out (it slams), out with an ease-in (it is flung)."""
    a, b = fx["from"], fx["to"]
    if fx["in"] > 0 and t < a + fx["in"]:
        u = min(max((t - a) / fx["in"], 0.0), 1.0)
        return -((1.0 - u) ** 3)
    if fx["out"] > 0 and t > b - fx["out"]:
        u = min(max((t - (b - fx["out"])) / fx["out"], 0.0), 1.0)
        return u ** 3
    return 0.0


def plan(data, fps, frame0, palette, last_frame=None):
    """The project's `[[fx]]` entries (mk.toml `data`) normalised and timed: each with `index`, `f0`..`f1` (Blender
    frames; the effect is on f0 .. f1 - 1), and for a `push` the frame `inside_frame` it shows. The span must lie in the
    clip (frame0 .. last_frame); a push's inside frame too."""
    specs = data.get("fx") or []
    if isinstance(specs, dict):
        specs = [specs]
    out = []
    for i, raw in enumerate(specs):
        fx = normalize(raw, palette, f"fx {i}")
        f0, f1 = frame_of(fx["from"], fps, frame0), frame_of(fx["to"], fps, frame0)
        if f1 - f0 < 1:
            raise FxError(f"fx {i} ({fx['kind']}): {fx['from']:g}..{fx['to']:g} s is under a frame")
        if f0 < frame0 or (last_frame is not None and f1 - 1 > last_frame):
            raise FxError(f"fx {i} ({fx['kind']}): frames {f0}..{f1 - 1} reach outside the clip ({frame0}..{last_frame})")
        fx.update(index=i, f0=f0, f1=f1)
        if fx["kind"] == "push":
            fx["inside_frame"] = frame_of(fx["inside"], fps, frame0)
            last_inside = fx["inside_frame"] + (f1 - 1 - f0 if fx["play"] else 0)
            if last_frame is not None and last_inside > last_frame:
                raise FxError(f"fx {i} (push): the frames inside ({fx['inside_frame']}..{last_inside}) run past the clip")
        out.append(fx)
    return out


def point_key(fx):
    return f"x{fx['index']}"


def at(fxs, frame):
    """The effects on `frame`, in table order."""
    return [fx for fx in fxs if fx["f0"] <= frame < fx["f1"]]


def strength(fx, t):
    """The effect's strength at clip second `t`: linear between its keys, held outside them."""
    keys = fx["strength"]
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def sources(fx, frame, frame0):
    """The other frames of the cut that the effect reads at `frame` (Blender frames, the clip's own only): an echo's
    copies, a slitscan's ages, a push's inside frame, a mosh's frame before its span and every frame since (the motion
    it carries the held picture along, worked out again from its start when the frames come out of order), the poses
    before of the panels and the stack of a collage with a `delay` (their figures too)."""
    k = fx["kind"]
    if k == "echo":
        out = [frame - j * fx["every"] for j in range(1, fx["count"] + 1)]
    elif k == "slitscan":
        out = [frame - j for j in range(1, fx["depth"] + 1)]
    elif k == "push":
        out = [fx["inside_frame"] + (frame - fx["f0"] if fx["play"] else 0)]
    elif k == "mosh":
        out = list(range(fx["f0"] - 1, frame + 1))
    elif k in ("panels", "stack") and fx["delay"] > 0:
        out = [frame - j * fx["delay"] for j in range(1, fx["count"] + (k == "stack"))]
    else:
        out = []
    return [f for f in out if f >= frame0]


def progress(fx, frame):
    """0 .. 1 through the effect's span: 0 on its first frame, 1 on its last."""
    n = fx["f1"] - 1 - fx["f0"]
    return 1.0 if n <= 0 else (frame - fx["f0"]) / n


def figure_key(fx):
    """The folder of a collage's figure layer (figure/<key>/<frame>.png), named from its object patterns: collages of the
    same figure share it."""
    return "f" + hashlib.sha1("\n".join(sorted(fx["objects"])).encode("utf-8")).hexdigest()[:10]


def pick(values, i):
    """Entry `i` of a collage's list, round and round: a list of colours, fills or sizes read at a step."""
    return values[int(i) % len(values)]


def step(fx, frame, times, fps, frame0):
    """How many of a collage's hits (`times`, clip seconds) have come by `frame`: those on a frame after its first, up to
    `frame` (a hit on its first frame is where it starts, the first entry of its lists)."""
    return sum(1 for t in times if fx["f0"] < frame_of(t, fps, frame0) <= frame)


KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


def piece_cell(fx, k):
    """The (column, row) a grid's piece stands on after `k` hits: from the middle of the top row, a knight's move on each
    hit (seeded; not straight back to the cell it came from while another is open), kept inside the grid of each step.
    It stays put on a grid too small for a knight to move in."""
    rng = np.random.default_rng([fx["seed"], fx["index"]])
    n = pick(fx["cells"], 0)
    pos, prev = ((n - 1) // 2, 0), None
    for s in range(1, k + 1):
        n = pick(fx["cells"], s)
        pos = (min(pos[0], n - 1), min(pos[1], n - 1))
        moves = [(pos[0] + dx, pos[1] + dy) for dx, dy in KNIGHT if 0 <= pos[0] + dx < n and 0 <= pos[1] + dy < n]
        fresh = [p for p in moves if p != prev] or moves
        if fresh:
            prev, pos = pos, fresh[int(rng.integers(len(fresh)))]
    return pos


def summary(fxs):
    """Numbers for reports."""
    return [{"index": fx["index"], "kind": fx["kind"], "frames": [fx["f0"], fx["f1"] - 1]} for fx in fxs]
