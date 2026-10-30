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
              shot as a codec does when the key frame of a cut is lost; or, with a `flow`, the span's first frame carried
              along a made-up motion (a drift, a zoom, a swirl, noise) as a P-frame played again and again: a still moshes
              too. A `mask` keeps it to the figure or the ground

The collages cut a figure (`objects`, drawn as its coverage beside the frame by `mk render`) out of the picture and lay it
on flat colour, its picture, a flat silhouette or a silkscreen print of it (`fill`); on each hit (`times`, or a `lane`, as
a zoom's) they step on: a list of colours, fills or sizes is read at the number of hits so far, plus the number of the
cell, copy, panel or region it paints, so a list of two colours is a chessboard that flips on every hit. Their figure may
come from another shot's own render (`shot`, a plate: its camera, look and lights) instead of the cut's frame, be set
into the frame (`at` and `height`, about its `pivot`), move (`keys`: turned, moved and scaled about the pivot), wear an
`edge` and cast a drop `shadow`.

    grid      the frame shrunk into cells (`cells`: n x n, or [columns, rows]), the figure in each on a chessboard of
              `back`; with `divide` each step's new cells grow out of the old ones (cell division, and back); a `piece`
              (a flat figure in one cell) hops a knight's move on each hit
    orbit     the figure moved to `center`, `scale` times its size, and `count` copies of it on a tilted ring round it,
              turning
    mirror    the frame folded about a vertical (`axes` "x"), horizontal ("y") or both ("xy") line through `center`: the
              half with the figure in it (moved to `place` first) and its reflections, each region its own colours
    panels    the frame cut into slanted panels across its long side, the figure in each, `delay` frames apart
    stack     the figure in front of a stack of flat copies of it (`bands`), each `step` further and `scale` as large: a
              banded echo, or with a `delay` the poses before; with `mirror` the stack again, reflected left for right
              about the vertical line through `center`, its echo stepping the mirrored way. Its bands may be wallpapers
              of sprites, its copies drawn in eras' `looks`, a tunnel that `recede`s forever, busy clones that `peel`
              off to places of their own, all of it squashing on the hits (`squash`): mkmmd.core.stack
    footage   the figure laid over the frames of a video (`clip`, mkmmd.footage), cut out like a sticker

A collage's `back` may be a pattern instead of a flat colour (mkmmd.core.fields.backs), or "picture": the frame as it
reached the effect, her world behind the figure. The colour fields `liquid`, `spiral` and `flood` (mkmmd.core.fields)
paint the picture with a light show, op-art or a colour blooming from a point. A `mandala` (rings of copies of the
figure, sprites or stickers) and `trails` (rainbow after-images of the figure) are mkmmd.core.mandala's. The remaps
`kaleido` (a kaleidoscope), `tunnel` (a Droste tunnel falling inward) and `melt` (the picture breathing, rippling,
swirling, dripping, wobbling) move the picture's pixels about (mkmmd.core.remap). The era looks `digicam`, `chrome`,
`gradmap`, `invert`, `testcard` and `cards` are mkmmd.core.eralook's, and so are the `era` presets of a film and a
camcorder. The toy paint app `paintapp` (its chrome, a cursor, stamps, bucket fills, an eraser across a cut, undo) is
mkmmd.core.paintapp's. The paper world's `activity` (a workbook page and its exercises done on the hits) and `label`
(tiny editorial type with dotted leaders) are mkmmd.core.activity's and mkmmd.core.labels'.

Any entry may give `aspect.<output>` tables: its own keys for one output (the span, the figure and the frames it reads
stay the entry's).
"""
import hashlib
import math

import numpy as np

from . import activity as AC
from . import eralook as ER
from . import fields as FL
from . import labels as LB
from . import lanes as LN
from . import mandala as MD
from . import paintapp as PA
from . import picturefx as PF
from . import remap as RM
from . import shotstyle as SS
from . import stack as SK
from . import tween as TW

FIGURES = ("grid", "orbit", "mirror", "panels", "stack", "footage")    # the collages: a figure cut out onto colour
KINDS = ("lens", "echo", "slitscan", "push", "scope", "zoom", "shock", "slab", "tint", "halftone", "split",
         "mosh") + FIGURES + PF.KINDS + FL.KINDS + MD.KINDS + RM.KINDS + AC.KINDS + LB.KINDS + ER.KINDS + PA.KINDS
PICTURE = "picture"                                     # a collage's fill: the figure as the frame shows it
SILK = "silkscreen"                                     # a collage's fill: the figure printed in flat inks, Warhol's
BLENDS = ("lighten", "darken")
EDGES = ("left", "right", "top", "bottom")
EASES = ("log", "smooth", "linear")
STEMS = ("drums", "bass", "other", "vocals", "mix")
_COMMON = {"kind", "from", "to", "strength", "aspect", "rect", "feather"}
_HITS = {"times", "lane", "min_db", "every"}            # zoom and shock fire on these, a collage steps on them
_FIGURE = {"objects", "back", "fill", "edge", "shadow", "keys", "pivot", "inks", "offset", "shot", "at",
           "height"} | _HITS      # a collage's figure, the shot it comes from and where it stands in the frame, its
#                                  colours, prints and moves, the hits it steps on
_FOOTAGE = {"clip", "in", "rate", "loop", "fit", "focus", "mirror"}
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
        "mosh": _COMMON | {"block", "flow", "mask", "objects"},
        "grid": _COMMON | _FIGURE | {"cells", "piece", "seed", "divide"},
        "orbit": _COMMON | _FIGURE | {"center", "count", "radius", "tilt", "scale", "size", "turn", "spin"},
        "mirror": _COMMON | _FIGURE | {"center", "axes", "place"},
        "panels": _COMMON | _FIGURE | {"count", "slant", "size", "delay"},
        "stack": _COMMON | _FIGURE | set(SK.KEYS),
        "footage": _COMMON | _FIGURE | _FOOTAGE}
KEYS.update({"pixelsort": _COMMON | _HITS | {"by", "lo", "hi", "axis", "reverse", "mask", "objects", "decay"},
             "film": _COMMON | {"mono", "fps", "grain", "flicker", "weave", "dust", "scratches", "hair", "vignette",
                                "contrast", "saturation", "dye", "halation", "era", "seed"},
             "camcorder": _COMMON | {"date", "time", "tick", "font", "corner", "indicator", "color", "size", "tracking",
                                     "chroma", "scanlines", "jitter", "soft", "saturation", "noise", "switching", "era",
                                     "seed"},
             "doodle": _COMMON | {"marks", "fps", "width", "outline", "wobble", "seed"},
             "booth": _COMMON | _HITS | {"count", "flash", "drop", "border", "paper", "back", "tilt", "layout", "ratio",
                                         "focus", "size", "doodles", "caption", "font", "color", "seed"}})
_FIGURE_DEFAULTS = {"back": "base", "fill": PICTURE, "edge": None, "shadow": None, "keys": None, "pivot": "base",
                    "inks": None, "offset": [4.0, -3.0], "min_db": None, "every": 1, "shot": None, "at": None,
                    "height": None}
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
            "mosh": {"block": 16.0, "flow": None, "mask": None},
            "grid": {**_FIGURE_DEFAULTS, "cells": 3, "piece": None, "seed": 1, "divide": 0.0},
            "orbit": {**_FIGURE_DEFAULTS, "center": [0.5, 0.5], "count": 6, "radius": 0.36, "tilt": 60.0, "scale": 1.0,
                      "size": 0.6, "turn": 30.0, "spin": 0.0},
            "mirror": {**_FIGURE_DEFAULTS, "center": [0.5, 0.5], "axes": "x", "place": None},
            "panels": {**_FIGURE_DEFAULTS, "count": 3, "slant": 12.0, "size": 1.0, "delay": 0},
            "stack": {**_FIGURE_DEFAULTS, **SK.DEFAULTS},
            "footage": {**_FIGURE_DEFAULTS, "in": 0.0, "rate": 1.0, "loop": False, "fit": "cover", "focus": [0.5, 0.5],
                        "mirror": False}}
DEFAULTS.update({"pixelsort": {"by": "luma", "lo": 0.25, "hi": 0.8, "axis": "x", "reverse": False, "mask": None,
                               "decay": 0.12, "min_db": None, "every": 1},
                 "film": {"mono": False, "fps": None, "grain": 0.06, "flicker": 0.05, "weave": 1.5, "dust": 3,
                          "scratches": 1, "hair": 0.0, "vignette": 0.3, "contrast": 1.0, "saturation": 1.0, "dye": 0.0,
                          "halation": 0.0, "era": None, "seed": 1},
                 "camcorder": {"date": None, "time": None, "tick": False, "font": None, "corner": "bottom-right",
                               "indicator": "PLAY", "color": "#FFFFFF", "size": 0.05, "tracking": 0.3, "chroma": 6.0,
                               "scanlines": 0.15, "jitter": 1.0, "soft": 1.2, "saturation": 0.8, "noise": 0.04,
                               "switching": 0.0, "era": None, "seed": 1},
                 "doodle": {"fps": 8.0, "width": 9.0, "outline": 5.0, "wobble": 1.5, "seed": 1},
                 "booth": {"count": 4, "flash": 0.15, "drop": 0.3, "border": 16.0, "paper": "#FFFFFF", "back": "rose",
                           "tilt": -4.0, "layout": None, "ratio": 0.8, "focus": [0.5, 0.42], "size": 0.86,
                           "doodles": 2, "caption": None, "font": None, "color": "text", "seed": 1, "min_db": None,
                           "every": 1}})
KEYS.update({k: _COMMON | own for k, own in FL.KEYS.items()})
DEFAULTS.update(FL.DEFAULTS)
KEYS.update({k: _COMMON | own for k, own in MD.KEYS.items()})
DEFAULTS.update(MD.DEFAULTS)
KEYS.update({k: _COMMON | _HITS | own for k, own in RM.KEYS.items()})
DEFAULTS.update(RM.DEFAULTS)
KEYS.update({k: _COMMON | own for k, own in {**AC.KEYS, **LB.KEYS}.items()})
DEFAULTS.update({**AC.DEFAULTS, **LB.DEFAULTS})
KEYS.update({k: _COMMON | own for k, own in ER.KEYS.items()})
DEFAULTS.update(ER.DEFAULTS)
KEYS.update({k: _COMMON | own for k, own in PA.KEYS.items()})
DEFAULTS.update(PA.DEFAULTS)
SLAB_DIRS = ("right", "left", "up", "down")            # the way a slab travels: in from the other side, out this way
AXES = ("x", "y", "xy")                                 # a mirror's fold: about a vertical line, a horizontal one, both
FITS = ("cover", "contain")                             # how footage meets the frame: filled and cropped, or whole
FIXED = ("kind", "from", "to", "objects", "shot", "aspect")   # what an aspect.<output> table may not set
MOVE = {"turn": 0.0, "dx": 0.0, "dy": 0.0, "scale": 1.0}             # a figure's move at rest: degrees, heights, times
MOVE_EASES = ("inout", "in", "out", "linear", "back")
BACK = 0.1                                              # how far a `back` ease overshoots its key: 10 % of the way
PIVOTS = ("base", "center")
SHADOW = {"color": "#000000", "offset": [0.008, 0.012], "blur": 10.0, "alpha": 0.4}
INKS = (("love", "gold"), ("pine", "rose"), ("iris", "foam"), ("gold", "love"))   # the default inks after the key
MASKS = ("figure", "ground")                            # a pixel sort's or a mosh's mask: inside the figure, outside it
FLOWS = ("drift", "zoom", "swirl", "noise")             # a mosh's made-up motions
FLOW = {"amount": 6.0, "steps": 12, "jitter": 0.25, "stuck": 0.3, "seed": 1}   # every flow's keys and defaults
FLOW_OWN = {"drift": {"angle": 0.0}, "zoom": {"center": [0.5, 0.5]}, "swirl": {"center": [0.5, 0.5]},
            "noise": {"size": 0.3}}                     # and each kind's own


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


def _rect(spec, what):
    """`rect` [x0, y0, x1, y1] (frame fractions, 0, 0 the top left; false or none: the whole frame), where the effect
    applies, and its `feather` (px at 1080 on the short side: a soft edge across it; 0, the default, a hard one)."""
    r = spec.get("rect")
    if r is None or r is False:
        if spec.get("feather"):
            raise FxError(f"{what}: feather softens the edge of a `rect`: give the rect")
        return None, 0.0
    if not (isinstance(r, (list, tuple)) and len(r) == 4):
        raise FxError(f"{what}: rect = {r!r}: expected [x0, y0, x1, y1] (frame fractions, 0, 0 the top left)")
    r = [_num(v, f"{what} rect", 0.0, 1.0) for v in r]
    if r[2] <= r[0] or r[3] <= r[1]:
        raise FxError(f"{what}: rect = {r}: x1 must be right of x0 and y1 below y0")
    return r, _num(spec.get("feather", 0.0), f"{what} feather", lo=0.0)


def _colour(v, palette, what):
    try:
        return [round(float(c), 6) for c in SS.resolve_colour(v, palette)]
    except SS.StyleError as e:
        raise FxError(f"{what}: {e}") from None


def normalize(spec, palette, what="fx"):
    """One `[[fx]]` entry with its kind's defaults and its colours resolved (display-space floats); its `aspect.<output>`
    tables become entries of their own in `aspect` (`for_output`)."""
    if not isinstance(spec, dict):
        raise FxError(f"{what} {spec!r}: expected a table")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise FxError(f"{what}: kind {kind!r} is none of {', '.join(KINDS)}")
    label, what = what, f"{what} ({kind})"
    extra = sorted(set(spec) - KEYS[kind])
    if extra:
        raise FxError(f"{what}: unknown keys {extra} (known: {sorted(KEYS[kind])})")
    for k in ("from", "to"):
        if k not in spec:
            raise FxError(f"{what}: `{k}` (clip seconds) is required")
    out = {**DEFAULTS[kind], **ER.era(kind, spec, what), **spec}
    out["from"], out["to"] = _num(spec["from"], f"{what} from"), _num(spec["to"], f"{what} to")
    if out["to"] <= out["from"]:
        raise FxError(f"{what}: `to` ({out['to']:g}) must come after `from` ({out['from']:g})")
    out["strength"] = _keys(spec.get("strength", 1.0), f"{what} strength")
    out["rect"], out["feather"] = _rect(spec, what)
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
        out["flow"] = _flow(out["flow"], what)
        _mask(spec, out, what)
    elif kind in FIGURES:
        _figure(spec, out, palette, what)
    elif kind in PF.KINDS:
        PF.normalize(spec, out, palette, what)
    elif kind in FL.KINDS:
        FL.normalize(spec, out, palette, what)
    elif kind in MD.KINDS:
        MD.normalize(spec, out, palette, what)
    elif kind in RM.KINDS:
        RM.normalize(spec, out, palette, what)
    elif kind in AC.KINDS:
        AC.normalize(spec, out, palette, what)
    elif kind in LB.KINDS:
        LB.normalize(spec, out, palette, what)
    elif kind in ER.KINDS:
        ER.normalize(spec, out, palette, what)
    elif kind in PA.KINDS:
        PA.normalize(spec, out, palette, what)
    out["aspect"] = _aspects(spec, palette, label, what)
    return out


def _aspects(spec, palette, label, what):
    """An entry's `aspect.<output>` tables, {output: entry}: each the entry with that table's keys over its own,
    normalised as it is. They may not set FIXED: every output draws one span of one figure."""
    v = spec.get("aspect")
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise FxError(f"{what}: aspect = {v!r}: expected tables of the entry's keys by output ([fx.aspect.<output>])")
    base = {k: x for k, x in spec.items() if k != "aspect"}
    out = {}
    for name, over in v.items():
        if not isinstance(over, dict):
            raise FxError(f"{what}: aspect.{name} = {over!r}: expected a table of the entry's keys")
        fixed = sorted(set(over) & set(FIXED))
        if fixed:
            raise FxError(f"{what}: aspect.{name} may not set {fixed}: every output has the same span and figure")
        out[str(name)] = normalize({**base, **over}, palette, f"{label} aspect.{name}")
    return out


def for_output(fx, output):
    """The entry as the output named `output` draws it: its `aspect.<output>` entry when it has one, else itself."""
    return (fx.get("aspect") or {}).get(output, fx) if output is not None else fx


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


def _mask(spec, out, what):
    """A pixel sort's or a mosh's `mask` and the `objects` of its figure (the figure layer mk render draws): "figure"
    works inside the figure's coverage, "ground" outside it; both or neither."""
    if out["mask"] is not None and out["mask"] not in MASKS:
        raise FxError(f"{what}: mask {out['mask']!r} is none of {', '.join(MASKS)}")
    if (out["mask"] is None) != ("objects" not in spec):
        raise FxError(f"{what}: a `mask` (figure: inside the figure, ground: outside it) needs the figure's `objects`, "
                      f"and `objects` is only the figure of a mask: give both or neither")
    if out["mask"] is None:
        return
    objs = [spec["objects"]] if isinstance(spec["objects"], str) else spec["objects"]
    if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
        raise FxError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                      f"`prop:key`), a list or one")
    out["objects"] = list(objs)


def _flow(v, what):
    """A mosh's made-up motion: {kind, amount, steps, jitter, stuck, seed} with FLOW's defaults and its kind's own key
    (a drift's `angle`, a zoom's or a swirl's `center`, a noise's `size`); None (the default): the cut's own motion."""
    if v is None:
        return None
    if not isinstance(v, dict) or v.get("kind") not in FLOWS:
        raise FxError(f"{what}: flow = {v!r}: expected a table whose `kind` is one of {', '.join(FLOWS)}")
    kind = v["kind"]
    known = {"kind"} | set(FLOW) | set(FLOW_OWN[kind])
    extra = sorted(set(v) - known)
    if extra:
        raise FxError(f"{what}: flow ({kind}): unknown keys {extra} (known: {sorted(known)})")
    f = {**FLOW, **FLOW_OWN[kind], **v}
    out = {"kind": kind,
           "amount": _num(f["amount"], f"{what} flow amount", -1080.0, 1080.0),
           "steps": _num(f["steps"], f"{what} flow steps", 1, 500, integer=True),
           "jitter": _num(f["jitter"], f"{what} flow jitter", 0.0, 4.0),
           "stuck": _num(f["stuck"], f"{what} flow stuck", 0.0, 1.0),
           "seed": _num(f["seed"], f"{what} flow seed", lo=0, integer=True)}
    if kind == "drift":
        out["angle"] = _num(f["angle"], f"{what} flow angle")
    elif kind == "noise":
        out["size"] = _num(f["size"], f"{what} flow size", 0.02, 4.0)
    else:
        out["center"] = _pair(f["center"], f"{what} flow center")
    return out


def _colours(v, palette, what, picture=False):
    """A colour or a list of them -> [rgb, ...]; with `picture`, "picture" (the figure as the frame shows it) may stand
    for a colour, as None, and "silkscreen" (the figure printed in flat inks) as itself. Three numbers are one colour."""
    one = isinstance(v, str) or (isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))
    items = [v] if one else v
    if not isinstance(items, (list, tuple)) or not items:
        kinds = 'a colour, "picture" or "silkscreen"' if picture else "a colour"
        raise FxError(f"{what} = {v!r}: expected {kinds}, or a list of them")
    return [(None if c == PICTURE else c if c == SILK else _colour(c, palette, what)) if picture
            else _colour(c, palette, what) for c in items]


def _backs(v, palette, what):
    """A collage's `back`: a colour, a pattern (`{pattern = "spiral", colors = [...], ...}`, mkmmd.core.fields), "picture"
    (the frame as it reached the effect) or a list of them -> [rgb, pattern or None (the picture), ...]."""
    return FL.backs(v, palette, what)


def _shadow(v, palette, what):
    """A collage's drop `shadow`: {color, offset, blur, alpha} with SHADOW's defaults (offset [dx, dy] in frame heights,
    blur px at 1080 on the short side); None: none."""
    if v is None:
        return None
    if not isinstance(v, dict) or set(v) - set(SHADOW):
        raise FxError(f"{what}: shadow = {v!r}: expected {{color, offset, blur, alpha}} (offset [dx, dy] in frame "
                      f"heights, blur px at 1080)")
    s = {**SHADOW, **v}
    return {"color": _colour(s["color"], palette, f"{what} shadow color"),
            "offset": _pair(s["offset"], f"{what} shadow offset"),
            "blur": _num(s["blur"], f"{what} shadow blur", lo=0.0),
            "alpha": _num(s["alpha"], f"{what} shadow alpha", 0.0, 1.0)}


def _moves(v, what):
    """A collage's `keys`: [{t, turn, dx, dy, scale, ease}, ...] -> sorted by `t`, each with every channel of MOVE: what
    a key leaves out stays as the key before left it (at rest before the first)."""
    if v is None:
        return []
    if not isinstance(v, (list, tuple)) or not v or not all(isinstance(k, dict) for k in v):
        raise FxError(f"{what}: keys = {v!r}: expected [{{t, turn, dx, dy, scale, ease}}, ...]")
    known = {"t", "ease"} | set(MOVE)
    raw = []
    for k in v:
        extra = sorted(set(k) - known)
        if extra:
            raise FxError(f"{what} keys: unknown keys {extra} (known: {sorted(known)})")
        if "t" not in k:
            raise FxError(f"{what} keys: {k!r} has no `t` (clip seconds)")
        if k.get("ease", "inout") not in MOVE_EASES:
            raise FxError(f"{what} keys: ease {k['ease']!r} is none of {', '.join(MOVE_EASES)}")
        raw.append((_num(k["t"], f"{what} keys t"), k))
    raw.sort(key=lambda p: p[0])
    if any(a[0] == b[0] for a, b in zip(raw, raw[1:])):
        raise FxError(f"{what} keys: two keys at the same second")
    out, last = [], dict(MOVE)
    for t, k in raw:
        cur = {c: _num(k.get(c, last[c]), f"{what} keys {c}") for c in MOVE}
        if cur["scale"] <= 0:
            raise FxError(f"{what} keys: scale = {cur['scale']:g}: expected a positive number (times)")
        out.append({"t": t, **cur, "ease": k.get("ease", "inout")})
        last = cur
    return out


def _inks(v, palette, what):
    """A silkscreen's `inks`: ink sets of 2 to 4 colours each, the darkest band's first (one set may be given alone);
    default: the palette's darker of `base` and `text` as the key, then two of its accents, four sets (INKS)."""
    if v is None:
        key = min(("base", "text"), key=lambda s: sum(c * w for c, w in zip(
            SS.resolve_colour(s, palette), (0.2126, 0.7152, 0.0722))))
        v = [[key, a, b] for a, b in INKS]
    if isinstance(v, (list, tuple)) and v and all(isinstance(c, str) for c in v):
        v = [v]
    if not isinstance(v, (list, tuple)) or not v or not all(isinstance(s, (list, tuple)) and 2 <= len(s) <= 4 for s in v):
        raise FxError(f"{what}: inks = {v!r}: expected ink sets of 2 to 4 colours, the darkest band's first")
    return [[_colour(c, palette, f"{what} inks") for c in s] for s in v]


def _cells(c, what):
    """A grid's layout: n (n x n) or [columns, rows] -> [columns, rows], each 1..16."""
    if isinstance(c, (list, tuple)):
        if len(c) != 2:
            raise FxError(f"{what} = {c!r}: a layout is n (n x n) or [columns, rows]")
        return [_num(x, what, 1, 16, integer=True) for x in c]
    n = _num(c, what, 1, 16, integer=True)
    return [n, n]


def _figure(spec, out, palette, what):
    """A collage's keys: the figure (`objects`), its colours and fills, an edge, the hits it steps on, its layout."""
    if "objects" not in spec:
        raise FxError(f"{what}: `objects` is required: the figure, object patterns (`name*`, `@collection`, `prop:key`)")
    objs = [spec["objects"]] if isinstance(spec["objects"], str) else spec["objects"]
    if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
        raise FxError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                      f"`prop:key`), a list or one")
    out["objects"] = list(objs)
    out["back"] = _backs(out["back"], palette, f"{what} back")
    out["fill"] = _colours(out["fill"], palette, f"{what} fill", picture=True)
    e = out["edge"]
    if e is not None:
        if not isinstance(e, dict) or "color" not in e or set(e) - {"color", "width"}:
            raise FxError(f"{what}: edge = {e!r}: expected {{color, width}} (width in px at 1080 on the short side)")
        out["edge"] = {"color": _colour(e["color"], palette, f"{what} edge"),
                       "width": _num(e.get("width", 3.0), f"{what} edge width", lo=0.0)}
    out["shadow"] = _shadow(out["shadow"], palette, what)
    out["keys"] = _moves(out["keys"], what)
    pv = out["pivot"]
    if isinstance(pv, str) and pv not in PIVOTS:
        raise FxError(f"{what}: pivot {pv!r}: \"base\", \"center\" or [x, y] (frame fractions)")
    out["pivot"] = pv if isinstance(pv, str) else _pair(pv, f"{what} pivot")
    sh = out["shot"]
    if sh is not None and (not isinstance(sh, str) or not sh.strip()):
        raise FxError(f"{what}: shot = {sh!r}: expected the name of a [[shot]] (typically one with plate = true) whose "
                      f"own render the figure is taken from")
    out["at"] = None if out["at"] is None else _pair(out["at"], f"{what} at")
    out["height"] = None if out["height"] is None else _num(out["height"], f"{what} height", 0.01, 10.0)
    out["inks"] = _inks(out["inks"], palette, what)
    out["offset"] = _pair(out["offset"], f"{what} offset")
    _hits(spec, out, what, required=False)
    kind = out["kind"]
    if kind == "grid":
        cells = out["cells"] if isinstance(out["cells"], (list, tuple)) else [out["cells"]]
        if not cells:
            raise FxError(f"{what}: cells is a whole number (per side), [columns, rows], or a list of them")
        out["cells"] = [_cells(c, f"{what} cells") for c in cells]
        out["divide"] = _num(out["divide"], f"{what} divide", lo=0.0)
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
    elif kind == "stack":
        SK.normalize(spec, out, palette, what)
    else:
        if not isinstance(out.get("clip"), str) or not out["clip"].strip():
            raise FxError(f"{what}: `clip` is required: a video file (project-relative or absolute) or an asset slug")
        out["in"] = _num(out["in"], f"{what} in", lo=0.0)
        out["rate"] = _num(out["rate"], f"{what} rate", 0.0, 16.0)
        for k in ("loop", "mirror"):
            if not isinstance(out[k], bool):
                raise FxError(f"{what}: {k} is true or false")
        if out["fit"] not in FITS:
            raise FxError(f"{what}: fit {out['fit']!r} is none of {', '.join(FITS)}")
        out["focus"] = _pair(out["focus"], f"{what} focus")
        if not all(0.0 <= x <= 1.0 for x in out["focus"]):
            raise FxError(f"{what}: focus = {out['focus']}: [x, y] in 0..1 of the video")


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
        for e in [fx, *fx["aspect"].values()]:
            e.update(index=i, f0=f0, f1=f1)
            if e["kind"] == "push":
                e["inside_frame"] = frame_of(e["inside"], fps, frame0)
            if e["kind"] in PF.HOLDS or e["kind"] == "stack":
                e["frame_rate"] = float(fps)
            if e["kind"] in ER.KINDS:
                ER.timed(e, data, fps, frame0)
            if e["kind"] in PA.KINDS:
                PA.timed(e, fps, frame0)
        if fx["kind"] == "push":
            last_inside = fx["inside_frame"] + (f1 - 1 - f0 if fx["play"] else 0)
            if last_frame is not None and last_inside > last_frame:
                raise FxError(f"fx {i} (push): the frames inside ({fx['inside_frame']}..{last_inside}) run past the clip")
        for name, e in fx["aspect"].items():
            _same_reads(fx, e, name, frame0)
        out.append(fx)
    return out


def _same_reads(fx, over, name, frame0):
    """An `aspect.<output>` entry reads no frame its entry does not on any frame of the span: the layers mk render
    draws are planned from the entry."""
    for f in range(fx["f0"], fx["f1"]):
        extra = sorted(set(sources(over, f, frame0)) - set(sources(fx, f, frame0)))
        if extra:
            raise FxError(f"fx {fx['index']} ({fx['kind']}): aspect.{name} reads frames the entry does not ({extra[:4]} "
                          f"on frame {f}): give the keys that choose them to the entry itself")
    if set(points(over)) - set(points(fx)) or (uses_figure(over) and not uses_figure(fx)):
        raise FxError(f"fx {fx['index']} ({fx['kind']}): aspect.{name} reads scene points or a figure the entry does not: "
                      f"give the `center`, the marks' {{point}} and the `mask` to the entry itself")


def point_key(fx):
    return f"x{fx['index']}"


def points(fx):
    """[(key, expression)]: the scene points an effect needs projected through the camera of the shot in the cut on
    the frames of its span (point/<key>/): its `center` (key x<i>), a doodle's marks at a {point} (x<i>m<j>), the
    points a label's leaders run to (x<i>l<j>) and a paint app's bucket fills' {point} seeds (x<i>b<j>)."""
    out = []
    c = fx.get("center")
    if isinstance(c, dict) and c["mode"] == "point":
        out.append((point_key(fx), c["expr"]))
    if fx["kind"] == "doodle":
        out += [(f"{point_key(fx)}m{j}", m["at"]["expr"]) for j, m in enumerate(fx["marks"])
                if m["at"]["mode"] == "point"]
    if fx["kind"] in LB.KINDS:
        out += LB.points(fx)
    if fx["kind"] in PA.KINDS:
        out += PA.points(fx, point_key(fx))
    return out


def uses_figure(fx):
    """Whether an effect reads the layer of its figure (figure/<key>/): a collage, or an effect with a `mask` (a pixel
    sort, a mosh, a colour field) kept inside or outside the figure, a mandala of it, its trails, an activity page
    tracing her silhouette."""
    return fx["kind"] in FIGURES or fx.get("mask") is not None or (fx["kind"] in MD.KINDS and MD.uses_figure(fx)) or \
        (fx["kind"] in AC.KINDS and AC.uses_figure(fx))


def holds(fx):
    """Whether an effect shows frames of the cut from before as they reached it (a film's held frame, a booth's
    captures, the first frame a mosh with a `flow` holds, the frame a paint app's eraser rubs out): the effects before it
    in the table are applied to such a frame again when it was not kept."""
    return fx["kind"] in PF.HOLDS or (fx["kind"] == "mosh" and fx["flow"] is not None) or (
        fx["kind"] in PA.KINDS and PA.holds(fx))


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


def sources(fx, frame, frame0, hits=None):
    """The other frames of the cut that the effect reads at `frame` (Blender frames, the clip's own only): an echo's
    copies, a slitscan's ages, a push's inside frame, a mosh's frame before its span and every frame since (the motion
    it carries the held picture along, worked out again from its start when the frames come out of order) or, with a
    `flow`, its first frame alone (its figure too, with a `mask`), the poses before of the panels with a `delay` and of
    a stack's copies (their figures too: mkmmd.core.stack.sources, a tunnel's from its lane in `hits`) and of a trail,
    the first frame of a collage or a mandala whose `keys` turn its figure, or whose `at` or `height` set it into the
    frame, or a stack whose copies `peel` off, about a pivot found on it (`base`, `center`), a film's held frame (the
    first of the film frame it shows), a booth's captures so far (those of a lane only with `hits`, the [audio] hits
    file) and the frame before a paint app's eraser, while it rubs that out. A collage with its own `shot` reads those
    frames of that shot (its plates)."""
    k = fx["kind"]
    if k == "echo":
        out = [frame - j * fx["every"] for j in range(1, fx["count"] + 1)]
    elif k == "slitscan":
        out = [frame - j for j in range(1, fx["depth"] + 1)]
    elif k == "push":
        out = [fx["inside_frame"] + (frame - fx["f0"] if fx["play"] else 0)]
    elif k == "mosh":
        out = [fx["f0"]] if fx["flow"] is not None else list(range(fx["f0"] - 1, frame + 1))
    elif k == "panels" and fx["delay"] > 0:
        out = [frame - j * fx["delay"] for j in range(1, fx["count"])]
    elif k == "stack":
        out = SK.sources(fx, frame, frame0, hits)
    elif k in MD.KINDS:
        out = MD.sources(fx, frame)
    elif k == "film":
        out = [PF.film_step(fx, frame)[1]]
    elif k == "booth":
        times = fx["times"] if fx.get("times") is not None else [] if hits is None else hit_times(fx, hits)
        out = [g for _, g in PF.captures(fx, times, fx["frame_rate"], frame0) if g <= frame]
    elif k in PA.KINDS:
        out = PA.sources(fx, frame)
    else:
        out = []
    if k in FIGURES and _found_pivot(fx) and frame != fx["f0"] and fx["f0"] not in out:
        out.append(fx["f0"])
    return [f for f in out if f >= frame0]


def progress(fx, frame):
    """0 .. 1 through the effect's span: 0 on its first frame, 1 on its last."""
    n = fx["f1"] - 1 - fx["f0"]
    return 1.0 if n <= 0 else (frame - fx["f0"]) / n


def figure_key(fx):
    """The folder of a collage's figure layer (figure/<key>/<frame>.png), named from its object patterns and the shot it
    is seen through when it has its own `shot`: collages of the same figure share it."""
    name = "\n".join(sorted(fx["objects"])) + (f"\n@{fx['shot']}" if fx.get("shot") else "")
    return "f" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def placed(fx):
    """Whether a collage sets its figure into the frame (`at`, `height`) instead of leaving it where its frame shows it."""
    return fx.get("at") is not None or fx.get("height") is not None


def _found_pivot(fx):
    """Whether a collage moves (`keys`) or sets (`at`, `height`) its figure, or peels a stack's copies off to places of
    their own (`peel`), about a pivot found on it on the span's first frame (`base`, `center`), in any output: an `aspect`
    table's own placement reads that frame too."""
    return any((e["keys"] or placed(e) or e.get("peel")) and isinstance(e["pivot"], str)
               for e in [fx, *(fx.get("aspect") or {}).values()])


def placement(anchor, to, s):
    """The 2 x 3 affine map (pixels to pixels) that scales a figure `s` times about its pixel `anchor` and puts that at
    pixel `to`."""
    return [[float(s), 0.0, float(to[0]) - s * float(anchor[0])], [0.0, float(s), float(to[1]) - s * float(anchor[1])]]


def compose(a, b):
    """The 2 x 3 affine map `a` after `b`."""
    return [[a[r][0] * b[0][0] + a[r][1] * b[1][0], a[r][0] * b[0][1] + a[r][1] * b[1][1],
             a[r][0] * b[0][2] + a[r][1] * b[1][2] + a[r][2]] for r in (0, 1)]


def pick(values, i):
    """Entry `i` of a collage's list, round and round: a list of colours, fills or sizes read at a step."""
    return values[int(i) % len(values)]


def step(fx, frame, times, fps, frame0):
    """How many of a collage's hits (`times`, clip seconds) have come by `frame`: those on a frame after its first, up to
    `frame` (a hit on its first frame is where it starts, the first entry of its lists)."""
    return sum(1 for t in times if fx["f0"] < frame_of(t, fps, frame0) <= frame)


KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


def dims(cells):
    """A grid's layout (n, or [columns, rows]) -> (columns, rows)."""
    if isinstance(cells, (list, tuple)):
        return int(cells[0]), int(cells[1])
    return int(cells), int(cells)


def piece_cell(fx, k):
    """The (column, row) a grid's piece stands on after `k` hits: from the middle of the top row, a knight's move on each
    hit (seeded; not straight back to the cell it came from while another is open), kept inside the grid of each step.
    It stays put on a grid too small for a knight to move in."""
    rng = np.random.default_rng([fx["seed"], fx["index"]])
    c, r = dims(pick(fx["cells"], 0))
    pos, prev = ((c - 1) // 2, 0), None
    for s in range(1, k + 1):
        c, r = dims(pick(fx["cells"], s))
        pos = (min(pos[0], c - 1), min(pos[1], r - 1))
        moves = [(pos[0] + dx, pos[1] + dy) for dx, dy in KNIGHT if 0 <= pos[0] + dx < c and 0 <= pos[1] + dy < r]
        fresh = [p for p in moves if p != prev] or moves
        if fresh:
            prev, pos = pos, fresh[int(rng.integers(len(fresh)))]
    return pos


def box(d, i, j):
    """Cell (i, j) of a grid layout `d` (columns, rows): (x0, y0, x1, y1) in frame fractions."""
    return (i / d[0], j / d[1], (i + 1) / d[0], (j + 1) / d[1])


def _owner(i, a, b):
    """Along one side: the cell of a layout of `a` cells that holds the middle of cell `i` of a layout of `b`."""
    return min(int((i + 0.5) * a / b), a - 1)


def grid_cells(fx, frame, times, fps, frame0):
    """The cells a grid draws on `frame`, in drawing order: [((x0, y0, x1, y1) frame fractions, (i, j))], (i, j) the cell
    of the step's layout whose colours, fill and piece it shows. After k hits the layout is `cells` entry k. With a
    `divide`, on a step to a layout of more cells each new cell grows out of its parent (the cell of the layout before
    that holds its middle), its box sliding and scaling from the parent's to its own over `divide` seconds, eased out;
    to one of fewer cells the old cells slide into the new cell that holds their middle, in its colours (division
    played back). A step's move ends early when the next hit comes first."""
    k = step(fx, frame, times, fps, frame0)
    b = dims(pick(fx["cells"], k))
    rest = [(box(b, i, j), (i, j)) for j in range(b[1]) for i in range(b[0])]
    if k == 0 or fx["divide"] <= 0:
        return rest
    a = dims(pick(fx["cells"], k - 1))
    hits = sorted(g for g in (frame_of(t, fps, frame0) for t in times) if g > fx["f0"])
    h = hits[k - 1]
    later = [g for g in hits if g > h]
    run = fx["divide"] * fps if not later else min(fx["divide"] * fps, later[0] - h)
    u = (frame - h) / run if run > 0 else 1.0
    if a == b or u >= 1.0:
        return rest
    e = TW.ease(u, "out")

    def lerp(p, q):
        return tuple(x + (y - x) * e for x, y in zip(p, q))
    if b[0] * b[1] >= a[0] * a[1]:
        return [(lerp(box(a, _owner(i, a[0], b[0]), _owner(j, a[1], b[1])), box(b, i, j)), (i, j))
                for j in range(b[1]) for i in range(b[0])]
    out = []
    for j in range(a[1]):
        for i in range(a[0]):
            p = (_owner(i, b[0], a[0]), _owner(j, b[1], a[1]))
            out.append((lerp(box(a, i, j), box(b, *p)), p))
    return out


def _ease(u, kind):
    if kind == "linear":
        return TW.clamp01(u)
    if kind == "back":
        return TW.pop(u, BACK)
    return TW.ease(u, kind)


def move_at(keys, t):
    """A collage figure's move at clip second `t`, {turn, dx, dy, scale}: from key to key by the ease of the key it
    arrives at (`back` overshoots it a little and settles), held before the first key and after the last; at rest
    without keys."""
    if not keys:
        return dict(MOVE)
    if t <= keys[0]["t"]:
        return {c: keys[0][c] for c in MOVE}
    for a, b in zip(keys, keys[1:]):
        if t < b["t"]:
            e = _ease((t - a["t"]) / (b["t"] - a["t"]), b["ease"])
            return {c: a[c] + (b[c] - a[c]) * e for c in MOVE}
    return {c: keys[-1][c] for c in MOVE}


def move_matrix(move, pivot, height):
    """The 2 x 3 affine map (pixels to pixels, y down) of a move: turned `turn` degrees counter-clockwise on screen and
    scaled `scale` times about the pixel `pivot`, then shifted `dx`, `dy` frame heights (`height` px; dy down)."""
    a = math.radians(move["turn"])
    c, s = move["scale"] * math.cos(a), move["scale"] * math.sin(a)
    px, py = float(pivot[0]), float(pivot[1])
    return [[c, s, px + move["dx"] * height - (c * px + s * py)],
            [-s, c, py + move["dy"] * height - (-s * px + c * py)]]


def summary(fxs):
    """Numbers for reports."""
    return [{"index": fx["index"], "kind": fx["kind"], "frames": [fx["f0"], fx["f1"] - 1]} for fx in fxs]
