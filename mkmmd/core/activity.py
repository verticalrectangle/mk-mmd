"""The activity book, bpy-free (docs/design.md: Frame effects, Activity book): `[[fx]] kind = "activity"` checked and
timed. The picture work is mkmmd/activity.py.

The frame becomes a page of a children's workbook: cream paper with its fibres, faint ruled lines and a margin, the
picture drawn on it (multiplied: a crayon drawing of her, mkmmd.core.shotstyle's crayon sketch, sits on the page as
drawn on it), and exercises printed on it that a crayon does on the hits (`times`, or a `lane` of the [audio] hits file):

    trace   a dotted line (a shape's outline, a sprite's, her own silhouette grown by `offset`, or `points`) that a
            crayon traces, a share of it on each hit, its crayon riding the tip
    dots    numbered dots round an outline, joined one by one (`per` a hit); when the last meets the first the picture
            they outline appears in it (a sprite, googly eyes and all, or the shape coloured in)
    paint   colour by number: a sprite cut into regions by its colours (or a shape into wedges), a small number in
            each, a region coloured in on each hit in its colour

An item's outline is a drawn shape (`shape`: star, heart, circle, clover, flower, butterfly; mkmmd.core.mandala's), a
registry sprite (`sprites`: a slug, a file or {tag}, as a mandala's, `pick` choosing among a tag's; `eyes` and `mouth`
as any sprite source), her own silhouette (`shape = "figure"`, a trace's only: the entry's `objects`) or a list of
`points`. Each steps on the entry's hits from its own `t` on, finishing in `steps` hits (default all of those left in the
span)."""
import math

from . import mandala as MD

KINDS = ("activity",)
ITEMS = ("trace", "dots", "paint")
SHAPES = ("star", "heart", "circle", "clover", "flower", "butterfly")
FIGURE = "figure"                      # a trace's shape: her silhouette, grown
PICTURES = ("multiply", "none")        # how the frame sits on the page: drawn on it, or left off (a blank page)
CRAYONS = ("love", "gold", "pine", "iris", "rose", "foam")   # a colour-by-number shape's colours, in turn
KEYS = {"activity": {"times", "lane", "min_db", "every", "items", "paper", "grain", "rule", "margin", "picture", "key",
                     "ink", "crayon", "font", "objects", "snap", "fps", "boil", "seed"}}
DEFAULTS = {"activity": {"paper": "#F6EEDC", "grain": 0.6, "rule": True, "margin": True, "picture": "multiply",
                         "key": None, "ink": "text", "crayon": "love", "font": None, "objects": None, "snap": 0.22,
                         "fps": 12.0, "boil": 1.2, "seed": 1, "min_db": None, "every": 1}}
RULE = {"color": "#9DB6D6", "pitch": 0.075, "alpha": 0.35, "width": 1.6}      # pitch frame heights, width px at 1080
MARGIN = {"color": "#E68A8A", "at": 0.09, "alpha": 0.5, "width": 2.0}         # at: frame fraction from the left
COMMON = {"kind", "at", "size", "angle", "t", "steps", "shape", "sprites", "pick", "points", "closed", "color", "eyes",
          "mouth"}
ITEM_KEYS = {"trace": COMMON | {"dash", "width", "offset", "tip"},
             "dots": COMMON | {"count", "per", "dot", "reveal", "numbers", "width"},
             "paint": COMMON | {"colors", "per", "legend", "count"}}
ITEM_DEFAULTS = {"trace": {"dash": 14.0, "width": 10.0, "offset": 26.0, "tip": True},
                 "dots": {"count": 16, "per": None, "dot": 5.0, "reveal": True, "numbers": True, "width": 8.0},
                 "paint": {"colors": None, "per": None, "legend": True, "count": 6}}
MAX_DOTS, MAX_REGIONS = 60, 24


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """An activity page's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    FX._hits(spec, out, what, required=False)
    for k, lo, hi in (("grain", 0.0, 1.0), ("snap", 0.0, 2.0), ("fps", 0.1, 120.0), ("boil", 0.0, 20.0)):
        out[k] = FX._num(out[k], f"{what} {k}", lo, hi)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    for k in ("paper", "ink", "crayon"):
        out[k] = FX._colour(out[k], palette, f"{what} {k}")
    out["key"] = out["paper"] if out["key"] is None else FX._colour(out["key"], palette, f"{what} key")
    if out["picture"] not in PICTURES:
        raise FX.FxError(f"{what}: picture {out['picture']!r} is none of {', '.join(PICTURES)}")
    out["rule"] = _line(FX, out["rule"], RULE, palette, f"{what} rule", ("pitch",))
    out["margin"] = _line(FX, out["margin"], MARGIN, palette, f"{what} margin", ("at",))
    if out["font"] is not None and not isinstance(out["font"], str):
        raise FX.FxError(f"{what}: font = {out['font']!r}: a font slug of the asset registry or a font file")
    items = spec.get("items")
    if not isinstance(items, (list, tuple)) or not items:
        raise FX.FxError(f"{what}: `items` is required: a list of tables {{kind = \"trace\" | \"dots\" | \"paint\", "
                         f"shape or sprites or points, at, size, ...}}")
    out["items"] = [_item(FX, i, v, out, palette, f"{what} items[{i}]") for i, v in enumerate(items)]
    figure = any(it["shape"] == FIGURE for it in out["items"])
    if figure and spec.get("objects") is None:
        raise FX.FxError(f"{what}: an item traces her silhouette (shape = \"figure\"): give the entry its `objects`")
    if spec.get("objects") is not None:
        if not figure:
            raise FX.FxError(f"{what}: `objects` is the figure a trace follows (shape = \"figure\"); no item does")
        out["objects"] = MD._objects(FX, spec, what)


def _line(FX, v, base, palette, what, own):
    """A page's `rule` or `margin`: true (the defaults), false (none) or {color, alpha, width, <own>}."""
    if v is False or v is None:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict) or set(v) - set(base):
        raise FX.FxError(f"{what} = {v!r}: true, false or {{{', '.join(base)}}}")
    o = {**base, **v}
    out = {"color": FX._colour(o["color"], palette, f"{what} color"),
           "alpha": FX._num(o["alpha"], f"{what} alpha", 0.0, 1.0),
           "width": FX._num(o["width"], f"{what} width", 0.1, 40.0)}
    for k in own:
        out[k] = FX._num(o[k], f"{what} {k}", 0.005, 1.0)
    return out


def _item(FX, i, v, entry, palette, what):
    if not isinstance(v, dict) or v.get("kind") not in ITEMS:
        raise FX.FxError(f"{what} = {v!r}: a table with kind = {' | '.join(ITEMS)}")
    kind = v["kind"]
    unknown = sorted(set(v) - ITEM_KEYS[kind])
    if unknown:
        raise FX.FxError(f"{what} ({kind}): unknown keys {unknown} (known: {sorted(ITEM_KEYS[kind])})")
    out = {"kind": kind, "index": i, **ITEM_DEFAULTS[kind]}
    out.update({k: v[k] for k in v if k in ITEM_DEFAULTS[kind]})
    out["at"] = FX._pair(v.get("at", [0.5, 0.5]), f"{what} at")
    out["size"] = FX._num(v.get("size", 0.3), f"{what} size", 0.01, 4.0)
    out["angle"] = FX._num(v.get("angle", 0.0), f"{what} angle")
    out["t"] = FX._num(v.get("t", entry["from"]), f"{what} t")
    if not entry["from"] <= out["t"] < entry["to"]:
        raise FX.FxError(f"{what}: t = {out['t']:g} is outside the page's span ({entry['from']:g} .. {entry['to']:g})")
    steps = v.get("steps")
    out["steps"] = None if steps is None else FX._num(steps, f"{what} steps", 1, 10000, integer=True)
    out["color"] = FX._colour(v.get("color", entry["crayon"]), palette, f"{what} color")
    sources = [k for k in ("shape", "sprites", "points") if k in v]
    if len(sources) != 1:
        raise FX.FxError(f"{what}: give one of `shape` ({', '.join(SHAPES)}{', figure' if kind == 'trace' else ''}), "
                         f"`sprites` (a slug, a file or {{tag}}) or `points` ([[x, y], ...] of the frame)")
    out["shape"] = out["sprites"] = out["points"] = None
    out["face"] = None
    if "shape" in v:
        shapes = SHAPES + ((FIGURE,) if kind == "trace" else ())
        if v["shape"] not in shapes:
            raise FX.FxError(f"{what}: shape {v['shape']!r} is none of {', '.join(shapes)}")
        out["shape"] = v["shape"]
    elif "sprites" in v:
        out["sprites"] = MD._refs(FX, v["sprites"], f"{what} sprites")
        from . import eyes as EY
        out["face"] = EY.face(v, f"{what} sprites")
    else:
        if kind == "paint":
            raise FX.FxError(f"{what}: colour by number needs regions: a `shape` or `sprites`, not points")
        pts = v["points"]
        if not isinstance(pts, (list, tuple)) or len(pts) < 2:
            raise FX.FxError(f"{what}: points = {pts!r}: at least two [x, y] of the frame (0, 0 the top left)")
        out["points"] = [FX._pair(p, f"{what} points") for p in pts]
    if out["sprites"] is None and any(k in v for k in ("eyes", "mouth")):
        raise FX.FxError(f"{what}: eyes and mouth are a sprite's (`sprites`)")
    out["pick"] = FX._num(v.get("pick", 0), f"{what} pick", 0, 10000, integer=True)
    closed = v.get("closed", out["points"] is None)
    if not isinstance(closed, bool):
        raise FX.FxError(f"{what}: closed is true or false")
    out["closed"] = closed
    if kind == "trace":
        for k, lo in (("dash", 2.0), ("width", 1.0), ("offset", 0.0)):
            out[k] = FX._num(out[k], f"{what} {k}", lo, 400.0)
        if not isinstance(out["tip"], bool):
            raise FX.FxError(f"{what}: tip is true or false (the crayon drawn at the tracing's tip)")
    elif kind == "dots":
        out["count"] = FX._num(out["count"], f"{what} count", 3, MAX_DOTS, integer=True)
        out["per"] = None if out["per"] is None else FX._num(out["per"], f"{what} per", 1, MAX_DOTS, integer=True)
        out["dot"] = FX._num(out["dot"], f"{what} dot", 0.5, 40.0)
        out["width"] = FX._num(out["width"], f"{what} width", 1.0, 400.0)
        for k in ("reveal", "numbers"):
            if not isinstance(out[k], bool):
                raise FX.FxError(f"{what}: {k} is true or false")
    else:
        out["per"] = None if out["per"] is None else FX._num(out["per"], f"{what} per", 1, MAX_REGIONS, integer=True)
        out["count"] = FX._num(out["count"], f"{what} count", 2, MAX_REGIONS, integer=True)
        if not isinstance(out["legend"], bool):
            raise FX.FxError(f"{what}: legend is true or false (the key of colours under the picture)")
        c = out["colors"]
        if out["sprites"] is not None:
            out["colors"] = 5 if c is None else FX._num(c, f"{what} colors", 2, 12, integer=True)
        else:
            c = list(CRAYONS) if c is None else c
            if not isinstance(c, (list, tuple)) or not c:
                raise FX.FxError(f"{what}: colors = {c!r}: a shape's crayons, a list of colours (a sprite's: how many)")
            out["colors"] = [FX._colour(x, palette, f"{what} colors") for x in c]
    return out


def uses_figure(fx):
    """Whether a page reads its figure's layer: an item traces her silhouette."""
    return fx.get("objects") is not None


# ================================================================================================= timing
def item_hits(item, times):
    """The hits an item steps on: the page's from its `t` on (a hit on its `t` too)."""
    return sorted(h for h in times if h >= item["t"] - 1e-9)


def steps(item, times):
    """How many hits an item takes to finish: its `steps`, else every hit it has (at least one)."""
    return item["steps"] if item["steps"] is not None else max(len(item_hits(item, times)), 1)


def progress(item, t, times, snap):
    """How many steps an item has made by clip second `t` (a real number): one on each of its hits, eased in over
    `snap` seconds from the hit on (smoothstep: the crayon starts and stops gently), never past `steps`."""
    done = 0.0
    for h in item_hits(item, times):
        if t < h:
            break
        u = 1.0 if snap <= 0 else min(max((t - h) / snap, 0.0), 1.0)
        done += u * u * (3.0 - 2.0 * u)
    return min(done, float(steps(item, times)))


def per_step(item, count, times):
    """How many dots (regions) an item joins (colours in) a step: its `per`, else as many as finish `count` of them
    in its steps."""
    return item["per"] if item["per"] is not None else max(int(math.ceil(count / float(steps(item, times)))), 1)


def boil_step(fx, t):
    """Which drawing of the crayon a frame shows: a new one `fps` times a second."""
    return int(math.floor(float(t) * fx["fps"] + 1e-6))


# ================================================================================================= outlines
def resample(path, n, closed=True):
    """`n` points evenly along a polyline `path` ((m, 2)) by its length, the first on its first point (a closed one's
    last before it)."""
    pts = [tuple(map(float, p)) for p in path]
    if closed:
        pts.append(pts[0])
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    if total <= 0:
        return [pts[0]] * n
    out, k, acc = [], 0, 0.0
    for j in range(n):
        s = total * j / (n if closed else max(n - 1, 1))
        while k < len(seg) - 1 and acc + seg[k] < s:
            acc += seg[k]
            k += 1
        u = 0.0 if seg[k] <= 0 else min(max((s - acc) / seg[k], 0.0), 1.0)
        a, b = pts[k], pts[k + 1]
        out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    return out


def head(path, share, closed=True):
    """The first `share` (0..1) of a polyline `path` by its length: its points up to there and the point there."""
    pts = [tuple(map(float, p)) for p in path]
    if closed:
        pts.append(pts[0])
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    s = total * min(max(share, 0.0), 1.0)
    out, acc = [pts[0]], 0.0
    for (a, b), d in zip(zip(pts, pts[1:]), seg):
        if acc + d >= s:
            u = 0.0 if d <= 0 else (s - acc) / d
            out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
            return out
        out.append(b)
        acc += d
    return out


def start_top(path):
    """A closed outline started at its topmost point (the smallest y; the leftmost of those) and running clockwise
    on screen (y down), as a child is taught to draw it."""
    pts = [tuple(map(float, p)) for p in path]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    if area < 0:                                         # counter-clockwise on screen (y down): turn it round
        pts = pts[::-1]
    i = min(range(len(pts)), key=lambda j: (round(pts[j][1], 6), pts[j][0]))
    return pts[i:] + pts[:i]
