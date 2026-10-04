"""Screen type, the numbers (docs/design.md: Text, Screen type): the bpy-free half of the text stage's `screen` key and of the
screen layer that `mkmmd/blender/styles.py` draws.

A `[[text]]` with `screen = ...` is not a thing in the scene: it is graphics laid over the finished picture of the cut, so it
stays where it is put whichever camera is cutting, whatever stands in front of it and however the shot is lit or graded. Its
panel is the picture, measured in FRAME HEIGHTS: 1 is the height of the picture, x is right and y up from its centre, so a cap
height, a margin or a band means the same in every shot and the 16:9 and 9:16 outputs only differ in how wide the frame is
(w / h frame heights). The text stage builds such a text as an ordinary text object on the plane z = 0 (one unit = one frame
height, hidden from every render); a frame that has any on screen gets a second pass, the type alone through an orthographic
camera one frame height tall, which `mk render` keeps as a file of its own (`screen/<frame>.png`, straight RGBA) so that `mk post`
lays it over whatever the cut effects make of the frame (`composite`). This module is the maths: the per-output overlay of an
entry (`aspect.<output>`), `extends`, the band that `screen = {anchor, margin, height}` stands for, the layer's file name and the
compositing. Errors are `TextError`s without the entry's name (the stage adds it)."""
import numpy as np

from .typeset import TextError

MARGIN = 0.04               # frame heights kept clear at the edges of a band
BAND = 0.25                 # frame heights: the default height of a band at the top or bottom
SCREEN_KEYS = {"anchor", "side", "margin", "height", "width"}
ANCHORS = ("top", "bottom", "center")
SIDES = ("left", "center", "right")


# ------------------------------------------------------------------------------------------------------ the entry
def deep_merge(base, over):
    """A copy of `base` with `over` laid over it: tables merge key by key (recursively), everything else (numbers, strings,
    lists) is replaced. Neither argument changes."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_entries(entries):
    """The `[[text]]` entries with `extends` resolved and the `abstract` ones left out. `extends = "<name>"` makes an entry
    the named entry (itself resolved first, so chains work) with its own keys laid over it by `deep_merge`: a look that eight
    lyric lines share is written once, in an entry with `abstract = true` (a base: never built), and each line says only what
    differs. An extending entry needs a name of its own; `abstract` is not inherited. Returns new dicts."""
    by = {}
    for e in entries:
        if e.get("name") is not None:
            by.setdefault(e["name"], e)

    def resolve(e, chain=()):
        base_name = e.get("extends")
        if base_name is None:
            return {k: v for k, v in e.items() if k != "abstract"}
        name = e.get("name")
        if name is None:
            raise TextError(f"a [[text]] that extends {base_name!r} needs a name of its own")
        if base_name == name or base_name in chain:
            raise TextError(f"text {name!r}: extends loops ({' -> '.join(chain + (name, base_name))})")
        if base_name not in by:
            raise TextError(f"text {name!r}: extends {base_name!r}, which no [[text]] is named")
        return deep_merge(resolve(by[base_name], chain + (name,)),
                          {k: v for k, v in e.items() if k not in ("extends", "abstract")})

    return [resolve(e) for e in entries if not e.get("abstract")]


def per_output(spec, outputs):
    """[(output name or None, spec)]: the entry as each output sees it. An entry with neither `aspect` nor `screen` is one
    object for every output (None). Any other is built once per output, `aspect.<output>` laid over it by `deep_merge`
    (the `aspect` key itself is dropped); an `aspect` table naming an output the project does not have is an error."""
    asp = spec.get("aspect")
    if asp is not None and not isinstance(asp, dict):
        raise TextError("aspect is a table of per-output overrides: [text.aspect.<output>]")
    for name, over in (asp or {}).items():
        if name not in outputs:
            raise TextError(f"aspect.{name}: the project has no output {name!r} (outputs: {list(outputs)})")
        if not isinstance(over, dict):
            raise TextError(f"aspect.{name} is a table of overrides")
    if asp is None and spec.get("screen") in (None, False):
        return [(None, spec)]
    base = {k: v for k, v in spec.items() if k != "aspect"}
    return [(name, deep_merge(base, (asp or {}).get(name))) for name in outputs]


# ------------------------------------------------------------------------------------------------------ the frame
def screen_spec(spec):
    """The validated `screen` of an entry as {anchor, side, margin, height, width} (`anchor` None: no band, the whole frame
    inside the margin), or None when the entry is not screen type. `screen = true` is all defaults."""
    s = spec.get("screen")
    if s is None or s is False:
        return None
    s = {} if s is True else s
    if not isinstance(s, dict):
        raise TextError("screen is true or a table (anchor, side, margin, height, width)")
    unknown = sorted(set(s) - SCREEN_KEYS)
    if unknown:
        raise TextError(f"unknown screen keys {unknown} (known: {sorted(SCREEN_KEYS)})")
    anchor = s.get("anchor")
    if anchor is not None and anchor not in ANCHORS:
        raise TextError(f"screen anchor is one of {list(ANCHORS)}")
    side = s.get("side", "center")
    if side not in SIDES:
        raise TextError(f"screen side is one of {list(SIDES)}")
    out = {"anchor": anchor, "side": side, "margin": float(s.get("margin", MARGIN)),
           "height": None if s.get("height") is None else float(s["height"]), "width": float(s.get("width", 1.0))}
    if not 0.0 <= out["margin"] < 0.5 or not 0.0 < out["width"] <= 1.0:
        raise TextError("screen margin is in [0, 0.5) and width in (0, 1]")
    if out["height"] is not None and not 0.0 < out["height"] <= 1.0:
        raise TextError("screen height is in (0, 1] frame heights")
    return out


def frame_width(size):
    """The picture's width in frame heights."""
    return float(size[0]) / float(size[1])


def ortho_scale(size):
    """Blender's `ortho_scale` of the camera that frames one picture of `size` pixels exactly: the larger side, in frame
    heights (AUTO sensor fit)."""
    return max(frame_width(size), 1.0)


def panel(screen, size, at=None, box=None):
    """((u, v), (w, h)) in frame heights: where a screen text's panel is and how big, for an output of `size` pixels. The band
    `screen` asks for (`anchor` top / bottom / center, `height` default 0.25, `margin` from the edges, `width` the share of what
    is left, `side` left / center / right: against that edge inside the margin, or centred) unless the entry gives `at` /
    `box` itself, which win over it."""
    W, m = frame_width(size), screen["margin"]
    if screen["anchor"] is None:
        h, v = 1.0 - 2.0 * m, 0.0
        if screen["height"] is not None:
            h = screen["height"]
    else:
        h = screen["height"] if screen["height"] is not None else BAND
        v = {"top": 0.5 - m - 0.5 * h, "bottom": -(0.5 - m - 0.5 * h), "center": 0.0}[screen["anchor"]]
    bw = max(W - 2.0 * m, 1e-6) * screen["width"]
    u = {"center": 0.0, "left": -(0.5 * W - m - 0.5 * bw), "right": 0.5 * W - m - 0.5 * bw}[screen["side"]]
    return ((u, v) if at is None else (float(at[0]), float(at[1]))), ((bw, h) if box is None else (float(box[0]), float(box[1])))


# ------------------------------------------------------------------------------------------------------ the layer files
LAYER_DIR = "screen"


def layer_rel(frame):
    """Path of a frame's screen layer inside a frame folder: `screen/<frame>.png` (straight RGBA, the frame's size; absent when
    nothing is on screen). `mk render` writes it next to the frame, `mk post` and `mk look` lay it over the cut's frame."""
    return f"{LAYER_DIR}/{int(frame):05d}.png"


# ------------------------------------------------------------------------------------------------------ compositing
def composite(base, layer):
    """`layer` (h, w, 4) straight RGBA laid over `base` (h, w, 3) RGB, both float in display space, top row first: the
    picture with the screen type on it. Returns float32 (h, w, 3); neither argument changes."""
    base = np.asarray(base, np.float32)
    layer = np.asarray(layer, np.float32)
    a = np.clip(layer[..., 3:4], 0.0, 1.0)
    return base[..., :3] * (1.0 - a) + layer[..., :3] * a
