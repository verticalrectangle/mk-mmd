"""Video stickers, bpy-free (docs/design.md: Frame effects, Video stickers): `[[fx]] kind = "clip"` checked and laid out.
The picture work is mkmmd/videosticker.py.

A video clip cut out in a shape and laid over the picture as a die-cut sticker: the picture stays and the clip is the
sticker, where a `footage` collage lays a figure over a clip that fills the frame. The clip is read and fitted by
mkmmd.footage with a footage collage's own keys (`clip`, `in`, `rate`, `loop`, `focus`, `mirror`), always to cover the
sticker's box. Its `shape`: a rounded `rect` (its corners `radius` of the box's short side), a `circle` (an ellipse by
the box), a `heart` or a `star`. The box stands with its middle at `at` (frame fractions), `size` frame heights tall and
`ratio` times as wide (by default the shape's own proportions), and `keys` move it about its middle as they move a
collage's figure (mkmmd.core.framefx `move_at`, `move_matrix`). A die-cut `edge` is grown round the shape and the sticker
casts a drop `shadow`, both as a collage's and both as much larger as a key scales the sticker. It is laid `over` the
picture (an opaque sticker), or its face is light, `screen`ed or `add`ed onto it so that the clip's black vanishes
(night light trails): its edge then stays paper, laid over, and only that rim casts the shadow. A `negative` sticker
shows no clip: its face is the picture under it inverted (type that answers whatever is behind it), laid as an opaque
sticker is.

Outlines are polygons in the box's own pixels about its middle, x right and y down, touching its four sides."""
import math

import numpy as np

from . import mandala as MD

KINDS = ("clip",)
OUTLINED = ("rect", "circle", "heart", "star")             # shapes with an outline (a polygon) and proportions of their own
SHAPES = OUTLINED + ("text",)                              # and keyed type: the letters of a text are the shape
BLENDS = ("over", "screen", "add", "negative")   # an opaque sticker, a face of light (the clip's black vanishes), or the
                                                 # picture under it inverted, no clip
VIDEO = ("clip", "in", "rate", "loop", "focus", "mirror")  # the clip's own keys, which a negative sticker has none of
STAR = (0.5, 0.11)                      # a star's inner radius over its outer, its corners' rounding over its outer
STEP = 1.5                              # px: an outline's points lie about this far apart along its curves
KEYS = {"clip": {"clip", "in", "rate", "loop", "focus", "mirror", "shape", "radius", "at", "size", "ratio", "keys",
                 "edge", "shadow", "blend", "text", "font"}}
DEFAULTS = {"clip": {"in": 0.0, "rate": 1.0, "loop": False, "focus": [0.5, 0.5], "mirror": False, "shape": "rect",
                     "radius": 0.12, "at": [0.5, 0.5], "size": 0.4, "ratio": None, "keys": None, "edge": None,
                     "shadow": None, "blend": "over", "text": None, "font": None}}


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A video sticker's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    if out["blend"] == "negative":                       # its face is the picture under it: there is no video to read
        given = [k for k in VIDEO if k in spec]
        if given:
            raise FX.FxError(f"{what}: a negative sticker shows the picture under it inverted: give it no "
                             f"{', '.join(given)}")
    else:
        FX._clip_keys(out, what)
    if out["shape"] not in SHAPES:
        raise FX.FxError(f"{what}: shape {out['shape']!r} is none of {', '.join(SHAPES)}")
    if out["shape"] != "rect" and "radius" in spec:
        raise FX.FxError(f"{what}: radius rounds a rect's corners; a {out['shape']} has none")
    if out["shape"] == "text":                           # the clip shows through the letters of `text`
        if not isinstance(out["text"], str) or not out["text"].strip():
            raise FX.FxError(f"{what}: a text sticker needs `text`, the words its letters show the clip through")
        if "ratio" in spec:
            raise FX.FxError(f"{what}: a text sticker takes its proportions from its letters: give it no ratio")
        if out["font"] is not None and not isinstance(out["font"], str):
            raise FX.FxError(f"{what}: font is a font file or a font slug of the asset registry")
    elif "text" in spec or "font" in spec:
        raise FX.FxError(f"{what}: text and font are a text sticker's (shape = \"text\")")
    out["radius"] = FX._num(out["radius"], f"{what} radius", 0.0, 0.5)
    out["at"] = FX._pair(out["at"], f"{what} at")
    out["size"] = FX._num(out["size"], f"{what} size", 0.01, 10.0)
    if out["shape"] != "text":
        out["ratio"] = own_ratio(out["shape"]) if out["ratio"] is None else FX._num(out["ratio"], f"{what} ratio", 0.05,
                                                                                     20.0)
    out["keys"] = FX._moves(out["keys"], what)
    out["edge"] = MD._edge(FX, out["edge"], palette, what)
    out["shadow"] = FX._shadow(out["shadow"], palette, what)
    if out["blend"] in ("screen", "add") and out["shadow"] is not None and (out["edge"] is None or
                                                                            out["edge"]["width"] <= 0):
        raise FX.FxError(f"{what}: a {out['blend']} sticker's face is light and casts no shadow: give it an `edge` (its "
                         f"rim casts the shadow) or no `shadow`")


def _unit(shape, n):
    """A heart (mkmmd.core.mandala's, its point down: `n` points) or a star (five points, the first up; its ten corners
    rounded, `n` points each) about (0, 0), x right and y down."""
    if shape == "heart":
        return MD._heart(n)
    a = np.radians(np.arange(10) * 36.0)
    rad = np.where(np.arange(10) % 2 == 0, 0.5, 0.5 * STAR[0])
    return MD._rounded(np.stack([rad * np.sin(a), -rad * np.cos(a)], 1), 0.5 * STAR[1], n)


def own_ratio(shape):
    """A shape's own width over its height, a sticker's `ratio` by default: 1 for a rect and a circle (a square, a round
    circle), a heart's and a star's proportions."""
    if shape in ("rect", "circle"):
        return 1.0
    p = _unit(shape, 720 if shape == "heart" else 24)
    span = p.max(0) - p.min(0)
    return round(float(span[0] / span[1]), 4)


def outline(shape, w, h, radius=0.0, step=STEP):
    """The outline of a sticker `shape` filling a box `w` x `h` px: (n, 2) points about the box's middle, x right and y
    down, touching its four sides, about `step` px apart along its curves; a rect's corners rounded by `radius` of the
    box's short side (0 square .. 0.5, its short sides half circles)."""
    w, h, step = float(w), float(h), max(float(step), 1e-3)
    if shape == "rect":
        r = min(max(float(radius), 0.0), 0.5) * min(w, h)
        if r < 0.25:
            return np.array([[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]])
        u = np.linspace(0.0, 0.5 * math.pi, max(int(math.ceil(0.5 * math.pi * r / step)), 2))
        corners = ((w / 2.0 - r, h / 2.0 - r), (r - w / 2.0, h / 2.0 - r), (r - w / 2.0, r - h / 2.0),
                   (w / 2.0 - r, r - h / 2.0))                       # bottom right, bottom left, top left, top right
        return np.concatenate([np.stack([cx + r * np.cos(u + 0.5 * math.pi * k), cy + r * np.sin(u + 0.5 * math.pi * k)],
                                        1) for k, (cx, cy) in enumerate(corners)])
    if shape == "circle":
        a, b = w / 2.0, h / 2.0
        around = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))      # Ramanujan's perimeter
        n = 4 * max(int(math.ceil(around / step / 4.0)), 4)            # a point on each end of both axes
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return np.stack([a * np.cos(t), b * np.sin(t)], 1)
    if shape not in ("heart", "star"):
        raise ValueError(f"no outline for a sticker shape {shape!r} (outlined: {', '.join(OUTLINED)})")
    big = max(w, h)
    n = max(int(math.ceil(4.3 * big / step)), 64) if shape == "heart" else \
        max(int(math.ceil(math.pi * 0.5 * STAR[1] * big / step)), 3)    # the heart's fastest stretch; a corner's arc
    p = _unit(shape, n)
    lo, hi = p.min(0), p.max(0)
    return (p - (lo + hi) / 2.0) / (hi - lo) * (w, h)


def place(fx, t, size, ratio=None):
    """Where a video sticker stands at clip second `t` in a frame of `size` (w, h): (M, box, scale). M is the 2 x 3 map
    from its box's own px (about its middle, at rest) to the frame's px (pixel i centred on i): its middle at `at`, moved
    by its `keys` there (turned and scaled about its middle, then shifted); box (w, h) its size at rest, px; scale the
    keys' scale there. `ratio` (width over height) is a text sticker's, from its letters; other shapes have their own."""
    FX = _fx()
    W, H = float(size[0]), float(size[1])
    bh = fx["size"] * H
    c = (fx["at"][0] * W - 0.5, fx["at"][1] * H - 0.5)
    mv = FX.move_at(fx["keys"], t)
    M = FX.compose(FX.move_matrix(mv, c, H), [[1.0, 0.0, c[0]], [0.0, 1.0, c[1]]])
    return M, ((fx["ratio"] if ratio is None else ratio) * bh, bh), float(mv["scale"])
