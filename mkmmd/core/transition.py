"""Cut effects between and over shots, bpy-free (docs/design.md: Shots: Transitions and inserts, Glitches, Frame effects):
the `[[transition]]`, `[[insert]]`, `[[glitch]]` and `[[fx]]` specs, the plan that checks them against the cut, the frames
each one needs rendered and the files those frames are stored in. `mk build` (keys the cameras the effects need), `mk render`
(renders the missing layers), `mk post` and `mk look` (composite them, mkmmd.cutfx and mkmmd.framefx) all read the one plan,
`plan()`, made from mk.toml alone.

    [[transition]]  at, kind = "expand" | "collapse" | "slash" | "whip" | "slice" | "spin" | "page", dur, ease
        expand / collapse: scale = [1, "fill"], turn = [0, 90], center = "subject" | [x, y] | [x, y, z] | "<expression>",
                           edge = {color, width}
        slash:             angle = -20, width = 0.3, color = "love", second = {color, width, offset}, dir = "right"
        whip:              dir = "left", blur = 1.0: a whip pan, the outgoing shot flung off the frame and the incoming
                           one after it, smeared along their way
        slice:             center = [0.5, 0.5] | "figure", angle = -20, edge = {color, width}: the outgoing frame cut in
                           two along a line, the halves sliding apart off the frame, the incoming shot in the gap
        spin:              after = dur, turns = 1.5, blur = 1.0, scale = 0.9, center = [0.5, 0.5], back = "mirror" | colour
                           (no `ease`): the outgoing frame spun up to the cut, the incoming one down to rest after it
        page:              dir = "left", angle = 20, radius = 0.12, back, through = 0.18, shadow = 0.45: a picture-book
                           page turn onto the incoming shot; at the clip's end the loop's wrap onto its first frame, not
                           a cut effect: `wrap_of` (mkmmd.core.pageturn, mkmmd.pageturn.wrap)
    [[insert]]      from, to, shot, shape = "thought", anchor, radius, size, offset, ratio, outline = {color, width},
                    pop = {dur, overshoot}, out = "pop" | "expand", expand = {dur, turn, ease}, aspect.<output>
                    shape = "split": shot | clip = "<video>" | {src, in, rate, loop, fit, mirror, back}, side, share,
                    focus = {host, inner}, divider = {color, width}, in, out, swap = [t, ...], aspect.<output> =
                    {side, share, focus}: the frame split in two, the inner half sliding in, trading sides, sliding out
                    shape = "iris": shot | clip, anchor, radius | size = 0.3, open = {dur, ease}, out = "close" |
                    "expand", expand = {dur, ease}, rim = {width, dark, color, glint, fringe} | false, edge = {color,
                    width}, ring = "vinyl" | "plate" | "lens" | {kind, ...}, aspect.<output> = {radius | size}: a round
                    window opening from the anchor, holding, closing back onto it or growing until its picture is the
                    frame (mkmmd.core.iris), a record, a plate or chrome rings round it (mkmmd.core.portal)
    [[glitch]]      from, to, objects, seed = 1, shift = 0.06, split = 6, colors = ["love", "foam"]
    [[fx]]          from, to, kind = "lens" | "echo" | "slitscan" | "push" | "scope" | "zoom" | "shock" | "slab" | "tint"
                    | ... | "grid" | "orbit" | "mirror" | "panels" | "stack" (mkmmd.core.framefx); unlike the others
                    they may overlap one another and any window, and stack in table order
A window is a run of frames over which an effect changes the picture: the `dur` seconds that END at the cut (`at`) for a
transition (a spin's runs on for its `after` seconds past it), the whole `from`..`to` for an insert or a glitch. Frames of
the other shot inside a window are *plates*,
rendered next to the cut's frames (`plate/<shot>/<frame>.png`); the silhouette whose figure is the matte of an expand /
collapse is rendered as `matte/` (its coverage) and `back/` (its frame without the figure); a glitch's shot with its
objects hidden is `bare/<key>/<frame>.png`; a collage's figure, its objects' coverage through the camera of the shot in
the cut (or of its own `shot`, whose plates are its picture then), is `figure/<key>/<frame>.png`; anchors and centres that
need Blender (a bone seen through a camera) are `point/<key>/<frame>.json`: {"p": [x, y] frame fractions, "depth", "m":
frame heights per metre there}."""
import json
import math
from pathlib import Path

from . import framefx as FX
from . import iris as IR
from . import pageturn as PG
from . import shotstyle as SS
from . import tween as TW

KINDS = ("expand", "collapse", "slash", "whip", "slice", "spin", "page")
SHAPES = ("thought", "split", "iris")
SLASH_DIRS = ("right", "left")
WHIP_DIRS = ("left", "right", "up", "down")   # the way the pictures travel: a pan to the right sends them left
SIDES = ("left", "right", "top", "bottom")    # where a split's inner half sits
FITS = ("cover", "contain")                    # how a split's video fills its half
MIN_WINDOW = 2                          # frames
LAYER_DIRS = ("plate", "matte", "back", "bare", "point", "figure")    # folders of an output's frame folder that hold layers
_GEOMETRY = {"size", "offset", "ratio"}  # the thought insert keys an `aspect.<output>` table may override
_SPLIT_GEOMETRY = {"side", "share", "focus"}  # and a split's

_COMMON = {"at", "kind", "dur", "ease"}
TRANSITION_KEYS = {"expand": _COMMON | {"scale", "turn", "center", "edge"},
                   "collapse": _COMMON | {"scale", "turn", "center", "edge"},
                   "slash": _COMMON | {"angle", "width", "color", "second", "dir"},
                   "whip": _COMMON | {"dir", "blur"},
                   "slice": _COMMON | {"center", "angle", "edge"},
                   "spin": {"at", "kind", "dur", "after", "turns", "blur", "scale", "center", "back"},
                   "page": PG.KEYS}
DEFAULT_DUR = {"slash": 0.18, "whip": 0.15, "slice": 0.3, "spin": 0.3, "page": PG.DUR}   # s; expand / collapse: 0.4
DEFAULT_EASE = {"slash": "inout", "whip": "inout", "page": PG.EASE}     # the others ease `in`; a spin has no ease
INSERT_KEYS = {"from", "to", "shot", "shape", "anchor", "radius", "size", "offset", "ratio", "outline", "pop", "out", "expand",
               "aspect"}
SPLIT_KEYS = {"from", "to", "shot", "clip", "shape", "side", "share", "focus", "divider", "in", "out", "swap", "aspect"}
CLIP_KEYS = {"src", "in", "rate", "loop", "fit", "mirror", "back"}
GLITCH_KEYS = {"from", "to", "objects", "seed", "shift", "split", "colors"}


class TransitionError(ValueError):
    """A bad [[transition]] / [[insert]] / [[glitch]] key, or one that does not fit the cut."""


def frame_of(t, fps, frame0):
    """The frame of clip second `t` (the build's rounding)."""
    return int(round(frame0 + float(t) * fps))


# ================================================================================================= normalising specs
def _pair(v, what):
    try:
        a, b = (float(x) for x in v)
    except (TypeError, ValueError):
        raise TransitionError(f"{what} = {v!r}: expected [a, b]") from None
    return [a, b]


def _unknown(spec, allowed, what):
    extra = sorted(set(spec) - allowed)
    if extra:
        raise TransitionError(f"{what}: unknown keys {extra} (known: {sorted(allowed)})")


def _colour(spec, palette, what):
    try:
        return [round(float(v), 6) for v in SS.resolve_colour(spec, palette)]
    except SS.StyleError as e:
        raise TransitionError(f"{what}: {e}") from None


def _ring(spec, palette, what, default_width=4.0):
    """{color, width} -> {"color": rgb, "width": px at 1080 on the short side}."""
    if not isinstance(spec, dict):
        raise TransitionError(f"{what} = {spec!r}: expected {{color, width}}")
    _unknown(spec, {"color", "width"}, what)
    width = float(spec.get("width", default_width))
    if width <= 0:
        raise TransitionError(f"{what}.width must be positive")
    return {"color": _colour(spec.get("color", "text"), palette, f"{what}.color"), "width": width}


def _center(spec):
    """`center`: the figure's centroid, a point of the frame, or a point of the scene seen through the matte shot's camera."""
    if spec is None or spec == "subject":
        return {"mode": "subject"}
    if isinstance(spec, str):
        return {"mode": "point", "expr": spec}
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return {"mode": "frame", "at": _pair(spec, "center")}
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        x, y, z = (float(v) for v in spec)
        return {"mode": "point", "expr": f"({x!r}, {y!r}, {z!r})"}
    raise TransitionError(f"center = {spec!r}: expected 'subject', [x, y] (frame fractions), [x, y, z] (a world point) "
                          f"or an expression such as 'bone(\"spine\").head'")


def _fractions(v, what):
    """[x, y]: a point of the frame as fractions from its top left, each in 0..1."""
    x, y = _pair(v, what)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise TransitionError(f"{what} = {v!r}: expected frame fractions in 0..1")
    return [x, y]


def _slice_center(spec, what):
    """A slice's `center`: a point of the frame (default its middle), or "figure", the centroid of the outgoing flat shot's
    figure on the window's first frame."""
    if spec is None:
        return {"mode": "frame", "at": [0.5, 0.5]}
    if spec == "figure":
        return {"mode": "figure"}
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return {"mode": "frame", "at": _fractions(spec, f"{what}: center")}
    raise TransitionError(f"{what}: center = {spec!r}: expected [x, y] (frame fractions) or \"figure\" (the centroid of "
                          f"the outgoing flat shot's figure)")


def normalize_transition(spec, palette):
    """One `[[transition]]` with its defaults, colours resolved (display-space floats)."""
    if not isinstance(spec, dict):
        raise TransitionError(f"transition {spec!r}: expected a table")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise TransitionError(f"transition kind = {kind!r}: expected one of {KINDS}")
    what = f"transition {kind} at {spec.get('at')}"
    _unknown(spec, TRANSITION_KEYS[kind], what)
    if "at" not in spec:
        raise TransitionError(f"{what}: needs `at` (clip seconds of the cut)")
    out = {"kind": kind, "at": float(spec["at"]), "dur": float(spec.get("dur", DEFAULT_DUR.get(kind, 0.4)))}
    if out["dur"] <= 0:
        raise TransitionError(f"{what}: dur must be positive")
    if "ease" in TRANSITION_KEYS[kind]:
        out["ease"] = spec.get("ease", DEFAULT_EASE.get(kind, "in"))
        if out["ease"] not in TW.EASES:
            raise TransitionError(f"{what}: ease = {out['ease']!r}, expected one of {TW.EASES}")
    if kind == "whip":
        out["dir"] = spec.get("dir", "left")
        if out["dir"] not in WHIP_DIRS:
            raise TransitionError(f"{what}: dir = {out['dir']!r}, expected one of {WHIP_DIRS} (the way the pictures go)")
        out["blur"] = float(spec.get("blur", 1.0))
        if not 0.0 <= out["blur"] <= 4.0:
            raise TransitionError(f"{what}: blur = {out['blur']}: the shutter in frames, 0 .. 4")
        return out
    if kind == "slash":
        out["angle"] = float(spec.get("angle", -20.0))
        out["width"] = float(spec.get("width", 0.3))
        if not 0.0 < out["width"] <= 1.5:
            raise TransitionError(f"{what}: width = {out['width']}: expected a fraction of the frame diagonal in (0, 1.5]")
        out["color"] = _colour(spec.get("color", "love"), palette, f"{what}: color")
        out["dir"] = spec.get("dir", "right")
        if out["dir"] not in SLASH_DIRS:
            raise TransitionError(f"{what}: dir = {out['dir']!r}, expected one of {SLASH_DIRS}")
        out["second"] = None
        if spec.get("second") is not None:
            sec = spec["second"]
            if not isinstance(sec, dict):
                raise TransitionError(f"{what}: second = {sec!r}: expected {{color, width, offset}}")
            _unknown(sec, {"color", "width", "offset"}, f"{what}: second")
            out["second"] = {"color": _colour(sec.get("color", "text"), palette, f"{what}: second.color"),
                             "width": float(sec.get("width", 0.04)), "offset": float(sec.get("offset", -0.03))}
            if out["second"]["width"] <= 0:
                raise TransitionError(f"{what}: second.width must be positive")
        return out
    if kind == "slice":
        out["angle"] = float(spec.get("angle", -20.0))
        out["center"] = _slice_center(spec.get("center"), what)
        out["edge"] = _ring(spec["edge"], palette, f"{what}: edge") if spec.get("edge") is not None else None
        return out
    if kind == "page":
        return PG.normalize(spec, out, palette, what)
    if kind == "spin":
        out["after"] = float(spec.get("after", out["dur"]))
        if out["after"] <= 0:
            raise TransitionError(f"{what}: after = {out['after']}: the seconds it spins on after the cut, positive")
        out["turns"] = float(spec.get("turns", 1.5))
        if out["turns"] == 0:
            raise TransitionError(f"{what}: turns = 0: a spin turns (clockwise; negative turns go counter-clockwise)")
        out["blur"] = float(spec.get("blur", 1.0))
        if not 0.0 <= out["blur"] <= 4.0:
            raise TransitionError(f"{what}: blur = {out['blur']}: the shutter in frames, 0 .. 4")
        out["scale"] = float(spec.get("scale", 0.9))
        if not 0.0 < out["scale"] <= 2.0:
            raise TransitionError(f"{what}: scale = {out['scale']}: the picture's scale at the cut, in (0, 2]")
        out["center"] = {"mode": "frame", "at": _fractions(spec.get("center", [0.5, 0.5]), f"{what}: center")}
        back = spec.get("back", "mirror")
        out["back"] = "mirror" if back == "mirror" else _colour(back, palette, f"{what}: back")
        return out
    sc = spec.get("scale", [1.0, "fill"])
    try:
        lo, hi = sc
        lo, hi = float(lo), None if hi == "fill" else float(hi)
    except (TypeError, ValueError):
        raise TransitionError(f"{what}: scale = {sc!r}: expected [lo, hi] or [lo, \"fill\"]") from None
    if lo <= 0.0 or (hi is not None and hi <= lo):
        raise TransitionError(f"{what}: scale = {sc!r}: expected 0 < lo < hi")
    out["scale"] = [lo, hi]                                      # hi None: whatever fills the frame
    out["turn"] = _pair(spec.get("turn", [0.0, 90.0]), "turn")
    out["center"] = _center(spec.get("center"))
    out["edge"] = _ring(spec["edge"], palette, f"{what}: edge") if spec.get("edge") is not None else None
    return out


def normalize_insert(spec, palette):
    """One `[[insert]]` with its defaults, colours resolved: a thought bubble, a split screen (normalize_split) or an iris
    (mkmmd.core.iris.normalize)."""
    if not isinstance(spec, dict):
        raise TransitionError(f"insert {spec!r}: expected a table")
    what = f"insert {spec.get('from')}-{spec.get('to')}"
    shape = spec.get("shape", "thought")
    if shape not in SHAPES:
        raise TransitionError(f"{what}: shape = {shape!r}, expected one of {SHAPES}")
    if shape == "split":
        return normalize_split(spec, palette, what)
    if shape == "iris":
        return IR.normalize(spec, palette, what)
    _unknown(spec, INSERT_KEYS, what)
    for k in ("from", "to", "shot", "anchor"):
        if k not in spec:
            raise TransitionError(f"{what}: needs `{k}`")
    out = {"from": float(spec["from"]), "to": float(spec["to"]), "shot": str(spec["shot"]),
           "shape": spec.get("shape", "thought"), "anchor": str(spec["anchor"]), "radius": float(spec.get("radius", 0.11)),
           "size": float(spec.get("size", 0.34)),
           "offset": _pair(spec.get("offset", [0.12, -0.30]), "offset"), "ratio": float(spec.get("ratio", 1.35)),
           "outline": _ring(spec.get("outline", {}), palette, f"{what}: outline", 5.0), "aspect": {}}
    if out["to"] <= out["from"]:
        raise TransitionError(f"{what}: `to` must come after `from`")
    if not 0.0 < out["size"] <= 1.0:
        raise TransitionError(f"{what}: size = {out['size']}: expected a fraction of the frame height in (0, 1]")
    if out["radius"] < 0:
        raise TransitionError(f"{what}: radius = {out['radius']}: expected metres, not negative")
    if out["ratio"] <= 0:
        raise TransitionError(f"{what}: ratio must be positive")
    pop = spec.get("pop", {})
    if not isinstance(pop, dict):
        raise TransitionError(f"{what}: pop = {pop!r}: expected {{dur, overshoot}}")
    _unknown(pop, {"dur", "overshoot"}, f"{what}: pop")
    out["pop"] = {"dur": float(pop.get("dur", 0.3)), "overshoot": float(pop.get("overshoot", 0.12))}
    if out["pop"]["dur"] <= 0 or out["pop"]["overshoot"] < 0:
        raise TransitionError(f"{what}: pop.dur must be positive and pop.overshoot not negative")
    out["out"] = spec.get("out", "pop")
    if out["out"] not in ("pop", "expand"):
        raise TransitionError(f"{what}: out = {out['out']!r}, expected 'pop' or 'expand'")
    ex = spec.get("expand", {})
    if not isinstance(ex, dict):
        raise TransitionError(f"{what}: expand = {ex!r}: expected {{dur, turn, ease}}")
    _unknown(ex, {"dur", "turn", "ease"}, f"{what}: expand")
    if ex and out["out"] != "expand":
        raise TransitionError(f"{what}: `expand` only applies to out = 'expand'")
    out["expand"] = {"dur": float(ex.get("dur", 0.4)), "turn": float(ex.get("turn", 0.0)), "ease": ex.get("ease", "in")}
    if out["expand"]["dur"] <= 0 or out["expand"]["ease"] not in TW.EASES:
        raise TransitionError(f"{what}: expand.dur must be positive and expand.ease one of {TW.EASES}")
    for name, over in (spec.get("aspect") or {}).items():
        if not isinstance(over, dict):
            raise TransitionError(f"{what}: aspect.{name} = {over!r}: expected a table")
        _unknown(over, _GEOMETRY, f"{what}: aspect.{name}")
        out["aspect"][name] = {k: (_pair(v, f"aspect.{name}.{k}") if k == "offset" else float(v)) for k, v in over.items()}
    return out


def insert_for(ins, output):
    """The insert's geometry (size, offset, ratio) for one output aspect: its `aspect.<output>` overrides merged in."""
    g = {k: ins[k] for k in _GEOMETRY}
    g.update(ins["aspect"].get(output, {}))
    return g


def _side(v, what):
    if v not in SIDES:
        raise TransitionError(f"{what}: side = {v!r}, expected one of {SIDES} (where the inner half sits)")
    return v


def _share(v, what):
    try:
        s = float(v)
    except (TypeError, ValueError):
        s = -1.0
    if not 0.0 < s < 1.0:
        raise TransitionError(f"{what}: share = {v!r}: expected the inner half's share of the frame, in (0, 1)")
    return s


def _focus(v, what):
    """A split's `focus`: {host, inner}, [x, y] frame fractions each; either may be left out."""
    if not isinstance(v, dict):
        raise TransitionError(f"{what}: focus = {v!r}: expected {{host = [x, y], inner = [x, y]}} (frame fractions)")
    _unknown(v, {"host", "inner"}, f"{what}: focus")
    return {k: _fractions(p, f"{what}: focus.{k}") for k, p in v.items()}


def _split_clip(v, palette, what):
    """A split's video: a file or an asset slug, or {src, in, rate, loop, fit, mirror, back} (mkmmd.footage plays it)."""
    if isinstance(v, str):
        v = {"src": v}
    if not isinstance(v, dict) or not isinstance(v.get("src"), str) or not v["src"]:
        raise TransitionError(f"{what}: clip = {v!r}: expected a video (a file or an asset slug) or "
                              f"{{src, in, rate, loop, fit, mirror, back}}")
    _unknown(v, CLIP_KEYS, f"{what}: clip")
    out = {"src": v["src"], "in": float(v.get("in", 0.0)), "rate": float(v.get("rate", 1.0)), "loop": v.get("loop", False),
           "fit": v.get("fit", "cover"), "mirror": v.get("mirror", False),
           "back": _colour(v.get("back", "base"), palette, f"{what}: clip.back")}
    if out["in"] < 0:
        raise TransitionError(f"{what}: clip.in = {out['in']}: seconds into the video, not negative")
    if not 0.0 <= out["rate"] <= 16.0:
        raise TransitionError(f"{what}: clip.rate = {out['rate']}: the playback speed, 0 (a still) .. 16")
    if not isinstance(out["loop"], bool) or not isinstance(out["mirror"], bool):
        raise TransitionError(f"{what}: clip.loop and clip.mirror are true or false")
    if out["fit"] not in FITS:
        raise TransitionError(f"{what}: clip.fit = {out['fit']!r}, expected one of {FITS}")
    return out


def normalize_split(spec, palette, what):
    """A `shape = "split"` insert with its defaults: the inner half's picture (`shot`, or `clip`: a video and how it plays),
    `side` (None: split_for decides per output), `share`, `focus`, `divider`, the seconds of the slides in and out, the
    clip seconds of the swaps (sorted) and the per-output overrides of side, share and focus."""
    _unknown(spec, SPLIT_KEYS, what)
    for k in ("from", "to"):
        if k not in spec:
            raise TransitionError(f"{what}: needs `{k}`")
    if ("shot" in spec) == ("clip" in spec):
        raise TransitionError(f"{what}: a split needs exactly one of `shot` (the [[shot]] in its inner half) and `clip` "
                              f"(a video there)")
    out = {"from": float(spec["from"]), "to": float(spec["to"]), "shape": "split",
           "shot": str(spec["shot"]) if "shot" in spec else None,
           "clip": _split_clip(spec["clip"], palette, what) if "clip" in spec else None,
           "side": None if spec.get("side") is None else _side(spec["side"], what),
           "share": _share(spec.get("share", 0.5), what),
           "focus": {"host": [0.5, 0.5], "inner": [0.5, 0.5], **_focus(spec.get("focus", {}), what)},
           "divider": _ring(spec["divider"], palette, f"{what}: divider", 6.0) if spec.get("divider") is not None else None,
           "in": float(spec.get("in", 0.3)), "out": float(spec.get("out", 0.3)), "aspect": {}}
    if out["to"] <= out["from"]:
        raise TransitionError(f"{what}: `to` must come after `from`")
    if out["in"] < 0 or out["out"] < 0:
        raise TransitionError(f"{what}: in = {out['in']}, out = {out['out']}: the seconds the inner half slides in and "
                              f"out, 0 (a cut) or more")
    swap = spec.get("swap", [])
    swap = [swap] if isinstance(swap, (int, float)) and not isinstance(swap, bool) else swap
    try:
        out["swap"] = sorted(float(t) for t in swap)
    except (TypeError, ValueError):
        raise TransitionError(f"{what}: swap = {spec.get('swap')!r}: expected clip seconds, a list or one") from None
    for t in out["swap"]:
        if not out["from"] < t < out["to"]:
            raise TransitionError(f"{what}: swap at {t} s: expected clip seconds inside the window ({out['from']} .. "
                                  f"{out['to']})")
    for name, over in (spec.get("aspect") or {}).items():
        if not isinstance(over, dict):
            raise TransitionError(f"{what}: aspect.{name} = {over!r}: expected a table")
        here = f"{what}: aspect.{name}"
        _unknown(over, _SPLIT_GEOMETRY, here)
        out["aspect"][name] = {k: (_side(v, here) if k == "side" else _share(v, here) if k == "share" else _focus(v, here))
                               for k, v in over.items()}
    return out


def split_for(ins, output, size):
    """A split's layout in one output: {"side", "share", "focus": {"host", "inner"}}, its `aspect.<output>` overrides merged
    in (a focus entry by entry). No `side` anywhere is "right", or "bottom" in a frame taller than wide (`size` = (w, h))."""
    over = ins["aspect"].get(output, {})
    side = over.get("side", ins["side"])
    if side is None:
        side = "bottom" if size[1] > size[0] else "right"
    return {"side": side, "share": over.get("share", ins["share"]), "focus": {**ins["focus"], **over.get("focus", {})}}


def normalize_glitch(spec, palette):
    """One `[[glitch]]` with its defaults, the ghosts' colours resolved."""
    if not isinstance(spec, dict):
        raise TransitionError(f"glitch {spec!r}: expected a table")
    what = f"glitch {spec.get('from')}-{spec.get('to')}"
    _unknown(spec, GLITCH_KEYS, what)
    for k in ("from", "to", "objects"):
        if k not in spec:
            raise TransitionError(f"{what}: needs `{k}`")
    objects = [spec["objects"]] if isinstance(spec["objects"], str) else spec["objects"]
    if not isinstance(objects, list) or not objects or not all(isinstance(p, str) and p for p in objects):
        raise TransitionError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                              f"`prop:key`), a list or one")
    out = {"from": float(spec["from"]), "to": float(spec["to"]), "objects": list(objects), "seed": int(spec.get("seed", 1)),
           "shift": float(spec.get("shift", 0.06)), "split": float(spec.get("split", 6.0))}
    if out["to"] <= out["from"]:
        raise TransitionError(f"{what}: `to` must come after `from`")
    if not 0.0 <= out["shift"] <= 0.5:
        raise TransitionError(f"{what}: shift = {out['shift']}: expected a share of the frame width in 0..0.5")
    if out["split"] < 0:
        raise TransitionError(f"{what}: split = {out['split']}: expected pixels at 1080 on the short side, not negative")
    colors = spec.get("colors", ["love", "foam"])
    if not isinstance(colors, list) or len(colors) != 2:
        raise TransitionError(f"{what}: colors = {colors!r}: expected the two ghosts' colours, [a, b]")
    out["colors"] = [_colour(c, palette, f"{what}: colors") for c in colors]
    return out


# ================================================================================================= the plan
def has_figure(shot, output):
    """Whether the `[[shot]]` table (raw mk.toml) has a flat look in the output aspect (`style = "silhouette"` or
    `"vector"`): a figure that can be a transition's matte."""
    merged = dict(shot)
    merged.update((shot.get("aspect") or {}).get(output) or {})
    return merged.get("style") in ("silhouette", "vector")


def cut_shot(cuts, frame):
    """The name of the shot in the cut at `frame` (SS.shot_at over the plan's cut table), None for an empty cut."""
    e = SS.shot_at(cuts, frame)
    return e["name"] if e else None


def _window_overlap(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def plan(data, fps, frame0, palette):
    """The transitions, inserts, glitches and frame effects of a project (mk.toml `data`) checked against its shots:

        {"fps", "frame0", "cuts": [{name, from, to}] (the cut, in frames), "plates": [shots that are not in the cut],
         "transitions": [...], "inserts": [...], "glitches": [...], "fx": [...]}

    Every transition carries its `cut` frame, window (`n` frames, `first`..`last`: the frames before the cut, and for a spin
    those after it too), the outgoing and incoming shots (`out`, `in`) and which of them gives the figure (`matte`: an
    expand's, a collapse's, a slice's cut through its figure) and which plays inside it (`plate`); every insert its
    `f0`..`f1` (the window is f0 .. f1 - 1), `host` shot and the frame counts of its pop and exit (a split's: its slides in and
    out, `n_in` and `n_exit`, and `swaps`, the frames its swaps land on; an iris's: its opening and its close or expand);
    every glitch its `f0`..`f1` and `host` shot; every frame effect what mkmmd.core.framefx.plan gives it. [] entries when
    the project has none."""
    shots = data.get("shot") or []
    specs = data.get("transition") or []
    inserts = data.get("insert") or []
    glitches = data.get("glitch") or []
    if isinstance(specs, dict):
        specs = [specs]
    if isinstance(inserts, dict):
        inserts = [inserts]
    if isinstance(glitches, dict):
        glitches = [glitches]
    outputs = [o["name"] for o in data.get("output") or []] or ["main"]
    cuts, plates, by_name = [], [], {}
    for s in shots:
        name = s.get("name")
        by_name[name] = s
        if s.get("plate"):
            plates.append(name)
            continue
        if "from" not in s or "to" not in s:
            raise TransitionError(f"shot {name!r}: needs from and to (or plate = true)")
        cuts.append({"name": name, "from": frame_of(s["from"], fps, frame0), "to": frame_of(s["to"], fps, frame0)})
    cuts.sort(key=lambda c: c["from"])
    out = {"fps": float(fps), "frame0": int(frame0), "cuts": cuts, "plates": plates, "transitions": [], "inserts": [],
           "glitches": []}
    windows = []                                                  # (first, last, label) of every effect

    for i, raw in enumerate(specs):
        tr = normalize_transition(raw, palette)
        what = f"transition {i} ({tr['kind']} at {tr['at']} s)"
        cut = frame_of(tr["at"], fps, frame0)
        incoming = [c for c in cuts if c["from"] == cut]
        still = None
        if tr["kind"] == "page" and not incoming:                # the clip's end: the loop's wrap onto its first frame
            onto = PG.wrap(data, cut, cuts, fps, frame0)
            if onto is not None:
                incoming, still = [onto], int(frame0)
        if len(incoming) != 1:
            starts = ", ".join(f"{(c['from'] - frame0) / fps:.2f}" for c in cuts)
            wraps = (" or at the clip's end ([project] duration), the loop's wrap onto its first frame"
                     if tr["kind"] == "page" else "")
            raise TransitionError(f"{what}: {'no shot' if not incoming else 'several shots'} start at that frame "
                                  f"(cuts at {starts} s{wraps})")
        before = SS.shot_at(cuts, cut - 1)
        if before is None or (before["name"] == incoming[0]["name"] and still is None):
            raise TransitionError(f"{what}: no shot before the cut")
        n = max(MIN_WINDOW, int(round(tr["dur"] * fps)))
        first = cut - n
        if first < before["from"]:
            raise TransitionError(f"{what}: the {n}-frame window starts before shot {before['name']!r} does "
                                  f"(it is {cut - before['from']} frames long); shorten `dur`")
        last = cut - 1
        if tr["kind"] == "spin":                                  # it spins on into the incoming shot, down to rest
            after = max(MIN_WINDOW, int(round(tr["after"] * fps)))
            end = min([incoming[0]["to"]] + [c["from"] for c in cuts if c["from"] > cut])
            if cut + after > end:
                raise TransitionError(f"{what}: the {after}-frame spin after the cut runs past the end of shot "
                                      f"{incoming[0]['name']!r} (it is {end - cut} frames long); shorten `after`")
            last = cut + after - 1
        tr.update(index=i, cut=cut, n=last + 1 - first, first=first, last=last, out=before["name"],
                  **{"in": incoming[0]["name"]})
        if tr["kind"] == "page":
            tr["still"] = still                                   # the wrap's: the first frame, held under the page
        if tr["kind"] == "expand":
            tr["matte"], tr["plate"] = tr["out"], tr["in"]
        elif tr["kind"] == "collapse":
            tr["matte"], tr["plate"] = tr["in"], tr["out"]
        elif tr["kind"] == "spin":                                # the cut's own frames, turned: no other shot's
            tr["matte"], tr["plate"] = None, None
        else:                                                     # a slice through the figure takes it from the outgoing shot
            tr["matte"] = tr["out"] if tr["kind"] == "slice" and tr["center"]["mode"] == "figure" else None
            tr["plate"] = tr["in"]
        if tr["matte"]:
            role = ("its figure's centroid holds the cut (center = \"figure\")" if tr["kind"] == "slice"
                    else f"its figure is the shape that {tr['kind']}s")
            for o in outputs:
                if not has_figure(by_name[tr["matte"]], o):
                    raise TransitionError(f"{what}: shot {tr['matte']!r} must have a flat look (style = \"silhouette\" "
                                          f"or \"vector\") in output {o!r}: {role}")
        windows.append((first, last, what))
        out["transitions"].append(tr)

    for j, raw in enumerate(inserts):
        ins = normalize_insert(raw, palette)
        what = f"insert {j} ({ins['from']}-{ins['to']} s)"
        f0, f1 = frame_of(ins["from"], fps, frame0), frame_of(ins["to"], fps, frame0)
        if ins["shot"] is not None and ins["shot"] not in by_name:
            raise TransitionError(f"{what}: no shot named {ins['shot']!r}")
        host = SS.shot_at(cuts, f0)
        if host is None or any(f0 < c["from"] < f1 for c in cuts) or f0 < host["from"]:
            raise TransitionError(f"{what}: it must lie inside one shot of the cut (no cut between {f0} and {f1 - 1})")
        if host["name"] == ins["shot"]:
            raise TransitionError(f"{what}: the shot inside the shape is the host shot itself")
        if ins["shape"] == "split":
            ins.update(_split_frames(ins, what, f0, f1, fps, frame0))
        elif ins["shape"] == "iris":
            ins.update(IR.frames(ins, what, f0, f1, fps, cuts))
        else:
            n_in = max(MIN_WINDOW, int(round(ins["pop"]["dur"] * fps)))
            if ins["out"] == "expand":
                starts = [c for c in cuts if c["from"] == f1]
                if not starts or starts[0]["name"] != ins["shot"]:
                    got = repr(starts[0]["name"]) if starts else "no shot starts there"
                    raise TransitionError(f"{what}: out = 'expand' grows into the shot that starts at `to` ({got}), so "
                                          f"`shot` must be that shot, not {ins['shot']!r}")
                n_exit = max(MIN_WINDOW, int(round(ins["expand"]["dur"] * fps)))
            else:
                n_exit = n_in
            if n_in + n_exit > f1 - f0:
                raise TransitionError(f"{what}: {f1 - f0} frames are too few for a {n_in}-frame pop and a {n_exit}-frame "
                                      f"exit")
            ins.update(n_in=n_in, n_exit=n_exit)
        ins.update(index=j, f0=f0, f1=f1, host=host["name"])
        windows.append((f0, f1 - 1, what))
        out["inserts"].append(ins)

    for g_i, raw in enumerate(glitches):
        g = normalize_glitch(raw, palette)
        what = f"glitch {g_i} ({g['from']}-{g['to']} s)"
        f0, f1 = frame_of(g["from"], fps, frame0), frame_of(g["to"], fps, frame0)
        host = SS.shot_at(cuts, f0)
        if host is None or any(f0 < c["from"] < f1 for c in cuts) or f0 < host["from"]:
            raise TransitionError(f"{what}: it must lie inside one shot of the cut (no cut between {f0} and {f1 - 1})")
        if f1 - f0 < MIN_WINDOW:
            raise TransitionError(f"{what}: {f1 - f0} frames are too few (at least {MIN_WINDOW})")
        g.update(index=g_i, f0=f0, f1=f1, host=host["name"])
        windows.append((f0, f1 - 1, what))
        out["glitches"].append(g)

    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            if _window_overlap(windows[a], windows[b]):
                raise TransitionError(f"{windows[a][2]} and {windows[b][2]} overlap in frames "
                                      f"{max(windows[a][0], windows[b][0])}..{min(windows[a][1], windows[b][1])}")
    duration = (data.get("project") or {}).get("duration")
    last = None if duration is None else int(frame0) + int(round(float(duration) * fps)) - 1
    try:
        out["fx"] = FX.plan(data, fps, frame0, palette, last)
    except FX.FxError as e:
        raise TransitionError(str(e)) from None
    for fx in out["fx"]:
        sh = fx.get("shot")
        if sh is None:
            continue
        what = f"fx {fx['index']} ({fx['kind']})"
        if sh not in by_name:
            raise TransitionError(f"{what}: shot = {sh!r}: there is no [[shot]] of that name")
        inside = [f for f in range(fx["f0"], fx["f1"]) if cut_shot(cuts, f) == sh]
        if inside:
            raise TransitionError(f"{what}: shot {sh!r} is the shot in the cut on frames {inside[0]}..{inside[-1]} of its "
                                  f"span: `shot` takes the figure from another view (a plate); without it the collage "
                                  f"cuts the figure out of the cut's own frame")
    return out


def _slide(seconds, fps):
    """The frames a split's slide of `seconds` takes: none for 0 (a cut), else at least one."""
    return 0 if seconds <= 0 else max(1, int(round(seconds * fps)))


def _split_frames(ins, what, f0, f1, fps, frame0):
    """{n_in, n_exit, swaps}: a split's slides in and out in frames and the frames its swaps land on, checked against its
    window. A swap slides as long as the slide in (`in`), from the frame the inner half is in at the earliest, and lands on
    its frame by the time the inner half starts to slide out."""
    n_in, n_exit = _slide(ins["in"], fps), _slide(ins["out"], fps)
    if n_in + n_exit > f1 - f0:
        raise TransitionError(f"{what}: {f1 - f0} frames are too few for a {n_in}-frame slide in and a {n_exit}-frame "
                              f"slide out")
    latest = min(f1 - n_exit, f1 - 1)
    swaps, free, prev = [], f0 + n_in, None                      # free: the first frame a swap may start sliding on
    for t in ins["swap"]:
        fs = frame_of(t, fps, frame0)
        if fs - n_in < free or fs <= f0 or (prev is not None and fs <= prev[1]):
            after = f"the swap at {prev[0]} s" if prev is not None else f"the slide in (the inner half is in on frame {free})"
            raise TransitionError(f"{what}: the swap at {t} s (landing on frame {fs} after a {n_in}-frame slide) comes too "
                                  f"soon after {after}")
        if fs > latest:
            raise TransitionError(f"{what}: the swap at {t} s lands on frame {fs}, too late: a swap must land by frame "
                                  f"{latest}, before the inner half slides out")
        swaps.append(fs)
        free, prev = fs, (t, fs)
    return {"n_in": n_in, "n_exit": n_exit, "swaps": swaps}


# ================================================================================================= what a frame needs
def has_effects(pl):
    """Whether the plan has any effect to render layers for and composite."""
    return bool(pl and (pl["transitions"] or pl["inserts"] or pl["glitches"] or pl.get("fx")))


def wrap_of(pl):
    """The loop's wrap of a plan: its page turn at the clip's end, onto frame 0 (`still`), or None. Unlike the other
    effects it is not composited over the cut's frames: mk post and mk look turn each frame of its window as finished,
    its frame effects and screen type on, over frame 0 as finished, held still (mkmmd.pageturn.wrap)."""
    return next((tr for tr in (pl or {}).get("transitions") or [] if tr.get("still") is not None), None)


def in_force(pl, shot, frame):
    """Whether `shot` is the shot of the cut at `frame` (then the cut's own frame is its plate)."""
    return cut_shot(pl["cuts"], frame) == shot


def demands(pl, frames=None):
    """The layers to render, {frame: [item, ...]} (frames restricted to `frames` when given). Items:

        {"kind": "plate", "shot": S}               S seen through its own camera and look, when S is not in the cut there
        {"kind": "matte", "shot": S}               S's figure alone (matte/) and S's frame without it (back/)
        {"kind": "bare", "shot": S, "key", "hide"} S with the objects `hide` matches hidden (bare/<key>/): a glitch's
                                                    shot, its figure gone
        {"kind": "point", "key", "expr", "shot"}   the world point `expr` (the `mk q` expression language) seen through
                                                    S's camera, as fractions of the frame (point/<key>/)
        {"kind": "figure", "key", "objects", "shot"}  the coverage of the objects `objects` matches, alone, through S's
                                                    camera (figure/<key>/): a collage's figure, on its frames and the
                                                    frames it reads"""
    want = None if frames is None else set(int(f) for f in frames)
    out = {}

    def add(f, item, always=False):
        if not always and want is not None and f not in want:
            return
        bucket = out.setdefault(f, [])
        if item not in bucket:
            bucket.append(item)

    def plate(f, shot):
        if not in_force(pl, shot, f):
            add(f, {"kind": "plate", "shot": shot})

    for tr in pl["transitions"]:
        if tr["kind"] == "spin" or tr.get("still") is not None:     # the cut's own frames turned, or the loop's wrap (the
            continue                                                # finished frames over frame 0): nothing to draw
        window = range(tr["first"], tr["last"] + 1)
        matte = tr["kind"] in ("expand", "collapse")
        for f in window:
            if matte:
                add(f, {"kind": "matte", "shot": tr["matte"]})
            plate(f, tr["plate"])                            # a slash, a whip, a slice, a page: the incoming shot's plates
            if matte and tr["center"]["mode"] == "point":
                add(f, {"kind": "point", "key": f"t{tr['index']}", "expr": tr["center"]["expr"], "shot": tr["matte"]})
        if tr["kind"] == "slice" and tr["matte"] and (want is None or any(f in want for f in window)):
            add(tr["first"], {"kind": "matte", "shot": tr["matte"]}, always=True)   # its figure there holds the cut
    for ins in pl["inserts"]:
        for f in range(ins["f0"], ins["f1"]):
            if ins["shot"] is not None:                      # a split's or an iris's video comes from its file
                plate(f, ins["shot"])
            if ins["shape"] in ("thought", "iris"):          # the anchor through the host's camera
                add(f, {"kind": "point", "key": f"i{ins['index']}", "expr": ins["anchor"], "shot": ins["host"]})
    for g in pl["glitches"]:
        for f in range(g["f0"], g["f1"]):
            add(f, {"kind": "bare", "shot": g["host"], "key": f"g{g['index']}", "hide": g["objects"]})
    for fx in pl.get("fx") or []:
        for key, expr in FX.points(fx):
            for f in range(fx["f0"], fx["f1"]):
                shot = cut_shot(pl["cuts"], f)
                if shot is not None:
                    add(f, {"kind": "point", "key": key, "expr": expr, "shot": shot})
        if FX.uses_figure(fx):
            for f in range(fx["f0"], fx["f1"]):
                for g in [f] + FX.sources(fx, f, pl["frame0"]):
                    item = figure_item(pl, fx, g)
                    if item is not None:
                        add(g, item)
                    if fx.get("shot"):                       # a figure from another shot: its picture is that shot's
                        plate(g, fx["shot"])
    return dict(sorted(out.items()))


def figure_item(pl, fx, frame):
    """The layer item of a collage's figure on `frame`: through its own `shot` when it has one, else through the shot in
    the cut there (None outside the cut)."""
    shot = fx.get("shot") or cut_shot(pl["cuts"], frame)
    if shot is None:
        return None
    return {"kind": "figure", "key": FX.figure_key(fx), "objects": fx["objects"], "shot": shot}


def _figure_reads(pl, fx, frame):
    """The layers an effect's figure is read from on `frame`: its coverage (a collage, a masked effect), and for a
    collage's figure from its own `shot` that shot's picture, a plate even where the shot is in the cut (`mk look` has not
    drawn the cut's frame there)."""
    out = []
    if FX.uses_figure(fx):
        item = figure_item(pl, fx, frame)
        if item is not None:
            out.append(item)
    if fx.get("shot"):
        out.append({"kind": "plate", "shot": fx["shot"]})
    return out


def fx_frames(pl, frames, hits=None, before=None):
    """{frame: [items]}: the cut's own frames that the frame effects on `frames` read (an echo's copies, a slitscan's
    ages, a push's inside, a collage's poses before, a film's held frame, a booth's captures: those on a lane only with
    `hits`, the [audio] hits file), as plates of the shot in the cut there, with a collage's figure on them, and on a held
    frame or a capture the points and figures the effects before it in the table read there (they are applied to it
    again) and the holder's own points (a paint app paints that frame's fills on the frame its eraser holds). A collage
    whose figure comes from its own `shot` reads that shot's plates and its figure through it instead.
    `mk post` has them all already; `mk look`, which draws only the frames asked for, renders these beside them. A held
    frame is finished again by the effects before its holder, so what those read from other frames (a stack's poses
    before, an echo's copies) is read too (`before`: only the effects ahead of that table index)."""
    out = {}
    fxs = pl.get("fx") or []
    for f in sorted(set(int(x) for x in frames)):
        for fx in FX.at(fxs, f):
            if before is not None and fx["index"] >= before:
                continue
            for g in FX.sources(fx, f, pl["frame0"], hits):
                shot = cut_shot(pl["cuts"], g)
                if g == f or (shot is None and not fx.get("shot")):
                    continue
                bucket = out.setdefault(g, [])
                items = ([] if fx.get("shot") else [{"kind": "plate", "shot": shot}]) + _figure_reads(pl, fx, g)
                for e in FX.at(fxs, g) if FX.holds(fx) else []:
                    if e["index"] >= fx["index"]:
                        break
                    items += [{"kind": "point", "key": k, "expr": x, "shot": shot} for k, x in FX.points(e)]
                    items += _figure_reads(pl, e, g)
                if FX.holds(fx) or FX.points_before(fx):
                    items += [{"kind": "point", "key": k, "expr": x, "shot": shot} for k, x in FX.points(fx)]
                bucket += [it for it in items if it not in bucket]
                if FX.holds(fx):                         # the held frame is drawn again: what its effects read too
                    for g2, more in fx_frames(pl, [g], hits, before=fx["index"]).items():
                        b2 = out.setdefault(g2, [])
                        b2 += [it for it in more if it not in b2]
    return out


def needs(pl):
    """{shot: [first frame, last frame]} a shot's camera must be keyed over for the plates, mattes and bare frames taken
    from it."""
    out = {}
    for f, items in demands(pl).items():
        for it in items:
            if it["kind"] in ("plate", "matte", "bare"):
                lo, hi = out.get(it["shot"], (f, f))
                out[it["shot"]] = [min(lo, f), max(hi, f)]
    return out


# ================================================================================================= the files
def rel_paths(item, frame):
    """The files (relative to an output's frame folder) an item is stored in. The last one is written last: it is the
    mark of a finished item."""
    f = f"{int(frame):05d}"
    if item["kind"] == "plate":
        return [f"plate/{item['shot']}/{f}.png"]
    if item["kind"] == "matte":
        return [f"back/{item['shot']}/{f}.png", f"matte/{item['shot']}/{f}.png"]
    if item["kind"] == "point":
        return [f"point/{item['key']}/{f}.json"]
    if item["kind"] == "bare":
        return [f"bare/{item['key']}/{f}.png"]
    if item["kind"] == "figure":
        return [f"figure/{item['key']}/{f}.png"]
    raise ValueError(f"unknown item {item!r}")


def plate_rel(shot, frame):
    return rel_paths({"kind": "plate", "shot": shot}, frame)[0]


def matte_rel(shot, frame):
    return rel_paths({"kind": "matte", "shot": shot}, frame)[1]


def back_rel(shot, frame):
    return rel_paths({"kind": "matte", "shot": shot}, frame)[0]


def bare_rel(key, frame):
    return rel_paths({"kind": "bare", "key": key}, frame)[0]


def point_rel(key, frame):
    return rel_paths({"kind": "point", "key": key}, frame)[0]


def figure_rel(key, frame):
    return rel_paths({"kind": "figure", "key": key}, frame)[0]


def pending_items(frames_dir, dem):
    """[(frame, item)] of the demands `dem` whose finishing file is not in `frames_dir` (missing or empty)."""
    root = Path(frames_dir)
    out = []
    for f, items in dem.items():
        for item in items:
            p = root / rel_paths(item, f)[-1]
            if not p.exists() or p.stat().st_size == 0:
                out.append((f, item))
    return out


def stale_point(path):
    """Whether a point file is empty, unreadable or from before `m` was stored (it cannot size anything in metres)."""
    try:
        return "m" not in json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True


def clear_claims(frames_dir):
    """Delete the layer files that cannot be used: the empty files a stopped render left (claims nobody finished) and point
    files from before their scale `m` was stored. `mk render` draws what is missing afterwards."""
    root = Path(frames_dir)
    for sub in LAYER_DIRS:
        for p in (root / sub).rglob("*"):
            if p.is_file() and (p.stat().st_size == 0 or (sub == "point" and stale_point(p))):
                p.unlink()


# ================================================================================================= timing
def pose(tr, k, hi=None):
    """(scale, turn in degrees) of the matte of an expand / collapse at its window frame `k` (0 .. n - 1). Expand runs
    from the figure itself (scale lo, turn t0) at k = 0 to the frame-filling size (hi, t1) at k = n - 1, collapse is the
    same path walked backwards. `hi` replaces the spec's top scale: mkmmd.cutfx passes the scale that fills the frame
    when the spec says "fill" (None) or its number is too small; scales are interpolated by ratio, turns linearly, both
    by the eased progress."""
    lo, top = tr["scale"]
    top = top if hi is None else hi
    if top is None:
        raise ValueError("scale = [lo, 'fill'] needs the scale that fills the frame: pass hi")
    e = TW.ease(k / (tr["n"] - 1), tr["ease"])
    p = e if tr["kind"] == "expand" else 1.0 - e
    t0, t1 = tr["turn"]
    return TW.geometric(lo, top, p), TW.lerp(t0, t1, p)


def covering_frame(tr):
    """The window frame at which the matte must fill the frame: the last of an expand, the first of a collapse."""
    return tr["last"] if tr["kind"] == "expand" else tr["first"]


def slash_progress(tr, k):
    """0..1 position of a slash's band along its sweep at window frame `k`: never exactly 0 or 1, so the band is on screen in
    every frame of the window (the cut itself is the first frame without it)."""
    return TW.ease((k + 1) / (tr["n"] + 1), tr["ease"])


def whip_progress(tr, k):
    """0..1 of the way a whip has carried the pictures at window frame `k` (fractional frames for its speed): eased, so
    it starts and ends slowly; never exactly 0 or 1 inside the window, as a slash's band."""
    u = min(max((k + 1) / (tr["n"] + 1), 0.0), 1.0)
    return TW.ease(u, tr["ease"])


def slice_normal(angle):
    """(nx, ny): the unit normal of a slice's cut at `angle` (degrees off vertical, clockwise on screen; x right, y down),
    pointing the way the half on its side travels."""
    a = math.radians(angle)
    return math.cos(a), math.sin(a)


def slice_reach(size, centre, normal):
    """(px, px): how far each half of a frame `size` = (w, h) cut through `centre` (index coordinates) across `normal` must
    travel to leave it, the half on the normal's side first: from the cut to the farthest corner, a pixel to spare."""
    w, h = size
    s = [(x - centre[0]) * normal[0] + (y - centre[1]) * normal[1] for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)]
    return max(s) + 1.0, 1.0 - min(s)


def slice_progress(tr, k):
    """0..1 of the way a slice's halves have travelled at window frame `k`: eased, never exactly 0 or 1 inside the window,
    so the cut shows on its first frame and the halves are off the frame at the cut."""
    return TW.ease((k + 1) / (tr["n"] + 1), tr["ease"])


def _spin_times(tr):
    """(t0, tc, t1) of a spin, frame times: at rest upright on the frame before its window, at its top speed between the
    outgoing shot's last frame and the incoming one's first, at rest again on the frame after its window."""
    return tr["first"] - 1.0, tr["cut"] - 0.5, tr["last"] + 1.0


def spin_angle(tr, t):
    """Degrees a spin has turned at frame time `t` (fractions allowed), clockwise on screen for positive `turns`: faster and
    faster (a steady angular acceleration) from rest to its top speed at the cut, then slowing at a steady rate to rest,
    `turns` x 360 in all. Each side of the cut turns its share in proportion to its time, so the speed is the same on both."""
    t0, tc, t1 = _spin_times(tr)
    total = 360.0 * tr["turns"]
    a = total * (tc - t0) / (t1 - t0)                               # turned by the cut
    if t <= t0:
        return 0.0
    if t >= t1:
        return total
    if t <= tc:
        return a * ((t - t0) / (tc - t0)) ** 2
    v = (t - tc) / (t1 - tc)
    return a + (total - a) * (1.0 - (1.0 - v) ** 2)


def spin_state(tr, frame):
    """What a spin shows at `frame`: {"picture": "out" before the cut, "in" from it (the cut's own frame either way), "turn":
    degrees that picture is turned (the incoming one comes to rest upright), "smear": [lo, hi] degrees about `turn` the
    shutter sweeps (`blur` frames about the frame), "scale": easing from 1 down to `scale` at the cut and back}."""
    t0, tc, t1 = _spin_times(tr)
    th = spin_angle(tr, frame)
    half = tr["blur"] / 2.0
    smear = sorted([spin_angle(tr, frame - half) - th, spin_angle(tr, frame + half) - th])
    dip = TW.ease((frame - t0) / (tc - t0), "inout") if frame <= tc else TW.ease((t1 - frame) / (t1 - tc), "inout")
    incoming = frame >= tr["cut"]
    return {"picture": "in" if incoming else "out", "turn": th - 360.0 * tr["turns"] if incoming else th, "smear": smear,
            "scale": 1.0 - (1.0 - tr["scale"]) * dip}


def insert_state(ins, frame):
    """What an insert shows at `frame` of its window: {"phase": "in" | "hold" | "out" | "expand", "bubble", "dots": [3],
    "p"}. `bubble` and `dots` are the pop scale (1 at rest, over 1 in the overshoot) of the shape and of its three trailing
    circles (0 = the one nearest the anchor), `p` the growth of an expanding shape (0 = as at rest .. 1 = filling the frame)."""
    i = frame - ins["f0"]                                        # frames since the insert began
    j = ins["f1"] - 1 - frame                                    # frames until its last
    over = ins["pop"]["overshoot"]
    if not 0 <= i < ins["f1"] - ins["f0"]:
        return {"phase": "none", "bubble": 0.0, "dots": [0.0] * 3, "p": 0.0}
    if i < ins["n_in"]:
        dots, bubble = pop_scales((i + 1) / ins["n_in"], over)
        return {"phase": "in", "bubble": bubble, "dots": dots, "p": 0.0}
    if ins["out"] == "pop" and j < ins["n_exit"]:
        dots, bubble = pop_scales(j / ins["n_exit"], over)
        return {"phase": "out", "bubble": bubble, "dots": dots, "p": 0.0}
    if ins["out"] == "expand" and j < ins["n_exit"]:
        e = TW.ease((ins["n_exit"] - 1 - j) / (ins["n_exit"] - 1), ins["expand"]["ease"])
        dots, _ = pop_scales(TW.clamp01(1.0 - e / DOTS_GONE), over)
        return {"phase": "expand", "bubble": 1.0, "dots": dots, "p": e}
    return {"phase": "hold", "bubble": 1.0, "dots": [1.0] * 3, "p": 0.0}


DOT_STARTS = (0.0, 0.14, 0.28)           # when each trailing circle starts to pop, as shares of the pop (0 = nearest the anchor)
DOT_SPAN = 0.30                          # how much of the pop each circle takes
BUBBLE_START = 0.45                      # when the shape itself starts to pop
DOTS_GONE = 0.35                         # an expanding shape has lost its circles by this share of the growth


def pop_scales(u, overshoot):
    """([scale of each trailing circle], scale of the shape) at pop progress u (0..1): the circles pop one after another,
    the nearest the anchor first, then the shape; every one is a springy 0 -> 1 that overshoots by `overshoot`."""
    dots = [TW.pop(TW.stagger(u, s, DOT_SPAN), overshoot) for s in DOT_STARTS]
    return dots, TW.pop(TW.stagger(u, BUBBLE_START, 1.0 - BUBBLE_START), overshoot)


def split_state(ins, frame):
    """What a split shows at `frame`: {"phase": "in" | "hold" | "out" | "none", "u": how far its inner half is in (0 .. 1,
    1 at rest), "swaps": how many swaps have landed, "q": the progress of a swap under way (0 when none)}. The half slides
    in eased out (settling into place), out eased in, and a swap (as long as the slide in) eased in and out; every frame of
    a slide shows it moving, as a slash's band is on screen in every frame of its window."""
    i, j = frame - ins["f0"], ins["f1"] - 1 - frame
    if not 0 <= i < ins["f1"] - ins["f0"]:
        return {"phase": "none", "u": 0.0, "swaps": 0, "q": 0.0}
    n_in, n_out = ins["n_in"], ins["n_exit"]
    if i < n_in:
        phase, u = "in", TW.ease((i + 1) / (n_in + 1), "out")
    elif j < n_out:
        phase, u = "out", 1.0 - TW.ease((n_out - j) / (n_out + 1), "in")
    else:
        phase, u = "hold", 1.0
    q = 0.0
    for fs in ins["swaps"]:
        if fs - n_in <= frame < fs:
            q = TW.ease((frame - fs + n_in + 1) / (n_in + 1), "inout")
    return {"phase": phase, "u": u, "swaps": sum(1 for fs in ins["swaps"] if fs <= frame), "q": q}


def split_panels(state, share, length, side):
    """The panels of a split along its axis at `state` (split_state): [(role, start, size)], role "host" or "inner", in
    pixels along the frame's width (a left / right split) or height (top / bottom) from its left / top edge, in order. A
    panel's picture is placed within its own `size`: a sliding panel's is its size at rest, the host's while the inner
    half slides in or out the room left to it. The panels tile the frame and may run past it. The inner half enters from
    its `side` and moves on the same way when the halves trade places: the row of panels runs round like a carousel."""
    L = float(length)
    a = share * L
    b = L - a
    u, q, odd = state["u"], state["q"], state["swaps"] % 2
    if q > 0.0 and not odd:                       # [host | inner] -> [inner | host], the host coming round behind
        panels = [("host", -q * b, b), ("inner", b - q * b, a), ("host", L - q * b, b)]
    elif q > 0.0:                                 # [inner | host] -> [host | inner]
        panels = [("inner", -q * a, a), ("host", a - q * a, b), ("inner", L - q * a, a)]
    elif not odd:                                 # the inner half at the far end: sliding in, at rest, sliding out
        panels = [("host", 0.0, L - a * u), ("inner", L - a * u, a)]
    else:                                         # at the near end, after an odd number of swaps
        panels = [("inner", -a * (1.0 - u), a), ("host", a * u, L - a * u)]
    if side in ("left", "top"):                   # laid out with the inner half entering at the far end: mirror it
        panels = [(r, L - s - n, n) for r, s, n in reversed(panels)]
    return panels


def panel_offset(start, size, focus, length):
    """Where a picture `length` long (the frame's own) starts along the axis in a panel at `start` of `size`: its point
    `focus` (a share of it) mid-panel, as near as it can be while the picture covers the panel (cropped, never squeezed)."""
    return start + min(max(size / 2.0 - focus * length, size - length), 0.0)


def split_dividers(panels, length):
    """Where the divider lines run along the axis: the seams between a host panel and an inner one inside the frame."""
    return [s1 for (r0, _, _), (r1, s1, _) in zip(panels, panels[1:]) if r0 != r1 and 0.0 < s1 < length]


def summary(pl):
    """A short JSON-able description of the plan for reports."""
    return {"transitions": [{"index": t["index"], "kind": t["kind"], "at": t["at"], "frames": [t["first"], t["last"]],
                             "out": t["out"], "in": t["in"]} for t in pl["transitions"]],
            "inserts": [{"index": s["index"], "shape": s["shape"], "frames": [s["f0"], s["f1"] - 1], "host": s["host"],
                         **({"shot": s["shot"]} if s["shot"] is not None else {"clip": s["clip"]["src"]}), "out": s["out"]}
                        for s in pl["inserts"]],
            "glitches": [{"index": g["index"], "frames": [g["f0"], g["f1"] - 1], "host": g["host"], "objects": g["objects"]}
                         for g in pl["glitches"]],
            "fx": FX.summary(pl.get("fx") or [])}
