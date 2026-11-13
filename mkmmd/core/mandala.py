"""Mandalas and trails, bpy-free (docs/design.md: Frame effects, Mandalas and trails): the `[[fx]]` kinds `mandala` and
`trails` checked and timed, the rings laid out and the stickers' shapes drawn as polygons; mkmmd/mandala.py does the
picture work.

    mandala  concentric rings of copies about a `center` on a `back`: copies of the figure (cut out as a collage cuts
             it), of sprites (pictures with alpha: registry slugs, tags, files, globs) or of stickers drawn here (flat
             ones: a clover, a flower, a heart, a star, a butterfly; and two kits drawn with a pen's ink line: a moon, a
             rabbit, a carrot and dango; a daisy, a cassette, a pedal and cherries). Each ring turns its own way, steps
             on the hits and blooms outward on them; with `grow` its copies pop in one by one; every copy may wear a
             die-cut `edge` and cast a `shadow`, and the figure itself may stay in the middle (`center_figure`)
    trails   rainbow after-images of the figure: its coverage `delay` frames apart back to `count` copies, each in a
             colour of a ramp that cycles (`cycle` steps a second, and one on each hit), fading with age, behind it

Angles are degrees counter-clockwise on screen. A ring's copy at angle `a` stands at (-sin a, -cos a) times the radius
from the centre (0 is straight up), radii in short sides of the frame; a copy turned `a` has its top pointing that way.
Shapes are polygons and lines about their middle (0, 0), x right and y down, their box's longer side 1.
"""
import math

import numpy as np

from . import eyes as EY
from . import picturefx as PC
from . import tween as TW

KINDS = ("mandala", "trails")
SHAPES = ("clover", "flower", "heart", "star", "butterfly", "moon", "rabbit", "carrot", "dango", "daisy", "cassette",
          "pedal", "cherry", "book", "teacup", "potion")
ROLES = ("main", "dark", "light", "accent", "accent_dark", "ink", "paper")   # what a shape's part is painted in
ORIENTS = ("radial", "upright")               # a copy's top: away from the centre, or up the frame
ORDERS = ("rings", "inward", "around", "random")   # the order a grow pops the copies in
TRAIL_BLENDS = ("over", "screen", "add")
RAINBOW = ("love", "rose", "gold", "foam", "pine", "iris")    # a trail's ramp: the palette's accents round the wheel
SHAPE_COLORS = {"clover": ("pine", "foam"), "flower": ("love", "rose", "iris"), "heart": ("love", "rose"),
                "star": ("gold", "rose"), "butterfly": ("iris", "love", "foam", "gold"), "moon": ("gold", "rose"),
                "rabbit": ("rose", "love", "iris"), "carrot": ("gold", "rose"), "dango": ("rose", "love", "iris"),
                "daisy": ("gold", "rose", "love"), "cassette": ("iris", "foam", "love", "gold"),
                "pedal": ("love", "pine", "iris", "gold"), "cherry": ("love", "rose"),
                "book": ("iris", "love", "pine"), "teacup": ("rose", "iris", "foam"),
                "potion": ("iris", "foam", "love")}
SHAPE_ACCENT = {"clover": "gold", "flower": "gold", "heart": "rose", "star": "love", "butterfly": "gold",
                "moon": "rose", "rabbit": "love", "carrot": "pine", "dango": "foam", "daisy": "gold", "cassette": "gold",
                "pedal": "gold", "cherry": "pine", "book": "gold", "teacup": "gold", "potion": "gold"}
PEN = 0.022                                  # the ink line round a drawn sticker's parts, of its box before scaling
PENNED = ("moon", "rabbit", "carrot", "dango", "daisy", "cassette", "pedal", "cherry", "book", "teacup",
          "potion")                          # drawn with it: the three kits
SCALE = {"figure": 0.25, "sticker": 0.12}    # a ring's `scale` by default: a copy's height (the figure's) or longer side
RING = {"count": None, "radius": None, "scale": None, "turn": 12.0, "step": 0.0, "phase": 0.0, "mirror": False,
        "spin": 0.0, "orient": "radial", "angle": 0.0, "source": "figure", "fill": None}
BLOOM = {"amount": 0.15, "grow": 0.1, "attack": 0.05, "decay": 0.3, "lag": 0.04}
GROW = {"dur": 0.35, "gap": 0.06, "order": "rings", "per": None}
MAX_RINGS, MAX_COPIES = 12, 64                # rings in a mandala, copies on a ring
KEYS = {"mandala": {"objects", "back", "fill", "edge", "shadow", "keys", "pivot", "inks", "offset", "times", "lane",
                    "min_db", "every", "center", "rings", "bloom", "grow", "center_figure", "alternate", "seed", "snap"},
        "trails": {"objects", "count", "delay", "colors", "cycle", "spread", "fade", "blend", "step", "tint", "snap",
                   "times", "lane", "min_db", "every"}}
DEFAULTS = {"mandala": {"back": "base", "fill": "picture", "edge": None, "shadow": None, "keys": None, "pivot": "base",
                        "inks": None, "offset": [4.0, -3.0], "min_db": None, "every": 1, "center": [0.5, 0.5],
                        "bloom": None, "grow": None, "center_figure": False, "alternate": True, "seed": 1, "snap": 0.2},
            "trails": {"count": 6, "delay": 2, "colors": list(RAINBOW), "cycle": 1.0, "spread": 1.0, "fade": 0.85,
                       "blend": "over", "step": [0.0, 0.0], "tint": 1.0, "snap": 0.12, "min_db": None, "every": 1}}


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """A mandala's or a trail's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    (_mandala if out["kind"] == "mandala" else _trails)(FX, spec, out, palette, what)


def _bool(FX, out, k, what):
    if not isinstance(out[k], bool):
        raise FX.FxError(f"{what}: {k} is true or false")


def _objects(FX, spec, what):
    objs = [spec["objects"]] if isinstance(spec["objects"], str) else spec["objects"]
    if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
        raise FX.FxError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                         f"`prop:key`), a list or one")
    return list(objs)


def _edge(FX, e, palette, what):
    if e is None:
        return None
    if not isinstance(e, dict) or "color" not in e or set(e) - {"color", "width"}:
        raise FX.FxError(f"{what}: edge = {e!r}: expected {{color, width}} (width in px at 1080 on the short side)")
    return {"color": FX._colour(e["color"], palette, f"{what} edge"),
            "width": FX._num(e.get("width", 3.0), f"{what} edge width", lo=0.0)}


def _refs(FX, v, what, key="sprites"):
    """A sprite source's pictures (a print's photographs: `key` names them in messages): a registry slug, a file or a
    glob (project-relative, absolute or ~/...), or {tag} (one tag or a list: the registry's pictures that have them
    all); one or a list -> [str or {"tag": [tags]}]."""
    items = [v] if isinstance(v, (str, dict)) else v
    if not isinstance(items, (list, tuple)) or not items:
        raise FX.FxError(f"{what}: {key} = {v!r}: expected slugs, files, globs or {{tag = ...}}, one or a list")
    out = []
    for r in items:
        if isinstance(r, str) and r.strip():
            out.append(r.strip())
            continue
        tags = r.get("tag") if isinstance(r, dict) and set(r) == {"tag"} else None
        tags = [tags] if isinstance(tags, str) else tags
        if not isinstance(tags, (list, tuple)) or not tags or not all(isinstance(x, str) and x.strip() for x in tags):
            raise FX.FxError(f"{what}: {key[:-1]} {r!r}: a slug, a file, a glob or {{tag = \"...\"}} (or a list of "
                             f"tags)")
        out.append({"tag": [x.strip() for x in tags]})
    return out


def _source(FX, v, palette, what):
    """A ring's `source`: "figure", {sprites = [...], eyes, mouth} or {shape = name or [names], colors, accent, ink,
    pen}."""
    if v == "figure":
        return {"kind": "figure"}
    if isinstance(v, dict) and len({"sprites", "shape"} & set(v)) == 1:
        if "sprites" in v:
            extra = sorted(set(v) - {"sprites"} - set(EY.KEYS))
            if extra:
                raise FX.FxError(f"{what}: source: unknown keys {extra} of a sprite source (known: "
                                 f"{sorted({'sprites'} | set(EY.KEYS))})")
            return {"kind": "sprites", "refs": _refs(FX, v["sprites"], what), "face": EY.face(v, f"{what} source")}
        extra = sorted(set(v) - {"shape", "colors", "accent", "ink", "pen"})
        if extra:
            raise FX.FxError(f"{what}: source: unknown keys {extra} of a shape (known: accent, colors, ink, pen, shape)")
        if not isinstance(v.get("pen", False), bool):
            raise FX.FxError(f"{what}: source: pen is true or false, not {v['pen']!r}")
        names = [v["shape"]] if isinstance(v["shape"], str) else v["shape"]
        if not isinstance(names, (list, tuple)) or not names or any(n not in SHAPES for n in names):
            raise FX.FxError(f"{what}: shape = {v['shape']!r}: one of {', '.join(SHAPES)}, or a list of them")
        own = {n: {"colors": [FX._colour(c, palette, f"{what} {n}") for c in SHAPE_COLORS[n]],
                   "accent": FX._colour(SHAPE_ACCENT[n], palette, f"{what} {n}")} for n in names}
        return {"kind": "shape", "shapes": list(names), "pen": v.get("pen", False),
                "colors": None if v.get("colors") is None else FX._colours(v["colors"], palette, f"{what} colors"),
                "accent": None if v.get("accent") is None else FX._colour(v["accent"], palette, f"{what} accent"),
                "ink": FX._colour(v.get("ink", "text"), palette, f"{what} ink"), "own": own}
    raise FX.FxError(f"{what}: source = {v!r}: \"figure\", {{sprites = [...]}} (slugs, files, globs or {{tag = ...}}) or "
                     f"{{shape = ...}} (one of {', '.join(SHAPES)})")


def _ring(FX, i, v, palette, what):
    what = f"{what} rings[{i}]"
    if not isinstance(v, dict):
        raise FX.FxError(f"{what} = {v!r}: expected a table {{count, radius, scale, turn, step, phase, mirror, spin, "
                         f"orient, angle, source, fill}}")
    extra = sorted(set(v) - set(RING))
    if extra:
        raise FX.FxError(f"{what}: unknown keys {extra} (known: {sorted(RING)})")
    r = {**RING, **v}
    src = _source(FX, r["source"], palette, what)
    sticker = src["kind"] != "figure"
    out = {"source": src,
           "count": FX._num(6 + 4 * i if r["count"] is None else r["count"], f"{what} count", 1, MAX_COPIES,
                            integer=True),
           "radius": FX._num(0.15 * (i + 1) if r["radius"] is None else r["radius"], f"{what} radius", lo=0.0),
           "scale": FX._num(SCALE["sticker" if sticker else "figure"] if r["scale"] is None else r["scale"],
                            f"{what} scale", 1e-3, 8.0)}
    for k in ("turn", "step", "phase", "spin", "angle"):
        out[k] = FX._num(r[k], f"{what} {k}")
    _bool(FX, r, "mirror", what)
    out["mirror"] = r["mirror"]
    if r["orient"] not in ORIENTS:
        raise FX.FxError(f"{what}: orient {r['orient']!r} is none of {', '.join(ORIENTS)}")
    out["orient"] = r["orient"]
    if r["fill"] is not None and sticker:
        raise FX.FxError(f"{what}: `fill` draws the figure: a ring of {src['kind']} takes its colours from its source")
    out["fill"] = None if r["fill"] is None else FX._colours(r["fill"], palette, f"{what} fill", picture=True)
    return out


def _bloom(FX, v, what):
    """`bloom`: a number (`amount`) or {amount, grow, attack, decay, lag}: on each hit the rings jump out by `amount` of
    their radius and their copies grow by `grow`, the inner ring first and each next `lag` s later, and settle."""
    if v is None:
        return None
    if not isinstance(v, dict):
        v = {"amount": v}
    extra = sorted(set(v) - set(BLOOM))
    if extra:
        raise FX.FxError(f"{what}: bloom: unknown keys {extra} (known: {sorted(BLOOM)})")
    b = {**BLOOM, **v}
    return {"amount": FX._num(b["amount"], f"{what} bloom amount", -0.9, 4.0),
            "grow": FX._num(b["grow"], f"{what} bloom grow", -0.9, 4.0),
            "attack": FX._num(b["attack"], f"{what} bloom attack", lo=0.0),
            "decay": FX._num(b["decay"], f"{what} bloom decay", lo=1e-3),
            "lag": FX._num(b["lag"], f"{what} bloom lag", lo=0.0)}


def _grow(FX, v, what):
    """`grow`: {times or lane (min_db, every), dur, gap, order, per}: the copies pop in one by one."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: grow = {v!r}: expected {{times or lane, dur, gap, order, per}}")
    known = set(GROW) | {"times", "lane", "min_db", "every"}
    extra = sorted(set(v) - known)
    if extra:
        raise FX.FxError(f"{what}: grow: unknown keys {extra} (known: {sorted(known)})")
    g = {**GROW, **v, "min_db": v.get("min_db"), "every": v.get("every", 1)}
    FX._hits(v, g, f"{what} grow")
    g["dur"] = FX._num(g["dur"], f"{what} grow dur", lo=1e-3)
    g["gap"] = FX._num(g["gap"], f"{what} grow gap", lo=0.0)
    if g["order"] not in ORDERS:
        raise FX.FxError(f"{what}: grow order {g['order']!r} is none of {', '.join(ORDERS)}")
    g["per"] = None if g["per"] is None else FX._num(g["per"], f"{what} grow per", lo=1, integer=True)
    return g


def _mandala(FX, spec, out, palette, what):
    rings = spec.get("rings")
    if not isinstance(rings, (list, tuple)) or not rings:
        raise FX.FxError(f"{what}: `rings` is required: a list of tables {{count, radius, scale, turn, step, phase, "
                         f"mirror, spin, orient, angle, source, fill}}")
    if len(rings) > MAX_RINGS:
        raise FX.FxError(f"{what}: {len(rings)} rings: at most {MAX_RINGS}")
    out["rings"] = [_ring(FX, i, r, palette, what) for i, r in enumerate(rings)]
    _bool(FX, out, "center_figure", what)
    _bool(FX, out, "alternate", what)
    figure = out["center_figure"] or any(r["source"]["kind"] == "figure" for r in out["rings"])
    if figure != ("objects" in spec):
        raise FX.FxError(f"{what}: a ring of the figure (source = \"figure\", the default) or `center_figure` needs the "
                         f"figure's `objects`, and `objects` is only the figure of those: give both or neither")
    out["objects"] = _objects(FX, spec, what) if figure else None
    out["back"] = FX._backs(out["back"], palette, f"{what} back")
    out["fill"] = FX._colours(out["fill"], palette, f"{what} fill", picture=True)
    out["edge"] = _edge(FX, out["edge"], palette, what)
    out["shadow"] = FX._shadow(out["shadow"], palette, what)
    out["keys"] = FX._moves(out["keys"], what)
    pv = out["pivot"]
    if isinstance(pv, str) and pv not in FX.PIVOTS:
        raise FX.FxError(f"{what}: pivot {pv!r}: \"base\", \"center\" or [x, y] (frame fractions)")
    out["pivot"] = pv if isinstance(pv, str) else FX._pair(pv, f"{what} pivot")
    out["inks"] = FX._inks(out["inks"], palette, what)
    out["offset"] = FX._pair(out["offset"], f"{what} offset")
    FX._hits(spec, out, what, required=False)
    out["center"] = FX._center(out["center"], f"{what} center")
    out["bloom"] = _bloom(FX, out["bloom"], what)
    if out["bloom"] is not None and "times" in out and not out["times"]:
        raise FX.FxError(f"{what}: bloom fires on the mandala's hits: give `times` or `lane`")
    out["grow"] = _grow(FX, out["grow"], what)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["snap"] = FX._num(out["snap"], f"{what} snap", lo=0.0)


def _trails(FX, spec, out, palette, what):
    if "objects" not in spec:
        raise FX.FxError(f"{what}: `objects` is required: the figure, object patterns (`name*`, `@collection`, "
                         f"`prop:key`)")
    out["objects"] = _objects(FX, spec, what)
    out["count"] = FX._num(out["count"], f"{what} count", 1, 32, integer=True)
    out["delay"] = FX._num(out["delay"], f"{what} delay", 1, 60, integer=True)
    out["colors"] = FX._colours(out["colors"], palette, f"{what} colors")
    for k, lo, hi in (("cycle", None, None), ("spread", None, None), ("fade", 0.0, 1.0), ("tint", 0.0, 1.0),
                      ("snap", 0.0, None)):
        out[k] = FX._num(out[k], f"{what} {k}", lo, hi)
    if out["blend"] not in TRAIL_BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(TRAIL_BLENDS)}")
    out["step"] = FX._pair(out["step"], f"{what} step")
    FX._hits(spec, out, what, required=False)


# ================================================================================================= what they read
def uses_figure(fx):
    """Whether a mandala or a trail reads its figure's layer: a trail always, a mandala with a ring of the figure or the
    figure in its middle."""
    return fx["kind"] == "trails" or fx.get("objects") is not None


def sources(fx, frame):
    """The other frames of the cut a mandala or a trail reads at `frame`: a trail's copies, `delay` frames apart back to
    `count` of them; the first frame of a mandala with a ring of the figure (its copies are sized by the figure's height
    there) or whose `keys` turn its figure about a pivot found on it."""
    if fx["kind"] == "trails":
        return [frame - j * fx["delay"] for j in range(1, fx["count"] + 1)]
    if fx["objects"] is None or frame == fx["f0"]:
        return []
    if any(r["source"]["kind"] == "figure" for r in fx["rings"]) or (fx["keys"] and isinstance(fx["pivot"], str)):
        return [fx["f0"]]
    return []


# ================================================================================================= timing
def _frame_of(t, fps, frame0):
    return int(round(frame0 + float(t) * fps))


_BACK = TW.back_coeff(0.1)              # a step's ease-out-back: it overshoots its end 10 % and settles


def _back_ease(u):
    """0 -> 1 eased out, overshooting its end 10 % and settling (a ring's step, a ramp's)."""
    u = TW.clamp01(u)
    return 1.0 + (_BACK + 1.0) * (u - 1.0) ** 3 + _BACK * (u - 1.0) ** 2


def stepped(fx, t, times, fps, frame0):
    """How far an entry has stepped on its hits (`times`, clip seconds) by clip second `t`: each hit on a frame after its
    first counts one, eased in over `snap` s from its frame on (a little overshoot that settles; 0: at once)."""
    total = 0.0
    for h in times:
        g = _frame_of(h, fps, frame0)
        if g <= fx["f0"]:
            continue
        a = t - (g - frame0) / fps
        if a < -1e-9:
            continue
        total += 1.0 if fx["snap"] <= 0 else _back_ease(a / fx["snap"])
    return total


def punch(t, times, attack, decay):
    """0..1 of a bloom at clip second `t`: each hit rises over `attack` (eased out) and dies away exp(-age / decay); the
    strongest counts."""
    out = 0.0
    for h in times:
        a = t - h
        if a < 0:
            continue
        rise = min(a / attack, 1.0) if attack > 0 else 1.0
        out = max(out, (1.0 - (1.0 - rise) ** 2) * math.exp(-max(a - attack, 0.0) / decay))
    return out


def turn(fx, i, t, times, fps, frame0):
    """Ring i's turn at clip second `t`, degrees counter-clockwise: `turn` a second since the span's start and `step` on
    each hit, every other ring the other way when `alternate`."""
    r = fx["rings"][i]
    sign = -1.0 if fx["alternate"] and i % 2 else 1.0
    return sign * (r["turn"] * (t - fx["from"]) + r["step"] * stepped(fx, t, times, fps, frame0))


def bloom(fx, i, t, times):
    """(radius, scale) factors of ring i at clip second `t` (1, 1 without a bloom): out and larger on each hit, ring i
    `lag` x i s after the first, then settling."""
    b = fx["bloom"]
    if b is None:
        return 1.0, 1.0
    p = punch(t - b["lag"] * i, times, b["attack"], b["decay"])
    return 1.0 + b["amount"] * p, 1.0 + b["grow"] * p


def _rest_angle(fx, i, j):
    r = fx["rings"][i]
    return r["phase"] + 360.0 * j / r["count"]


def grow_order(fx):
    """The copies of a mandala, [(ring, j)], in the order its `grow` pops them: `rings` the inner ring first (by radius),
    round each from its first copy; `inward` the outer ring first; `around` round the clock from the top across the
    rings at rest, the inner first at one angle; `random` shuffled (seeded by `seed` and the entry)."""
    rings = fx["rings"]
    every = [(i, j) for i in range(len(rings)) for j in range(rings[i]["count"])]
    order = fx["grow"]["order"]
    if order in ("rings", "inward"):
        sign = 1.0 if order == "rings" else -1.0
        return sorted(every, key=lambda c: (sign * rings[c[0]]["radius"], c[0], c[1]))
    if order == "around":
        return sorted(every, key=lambda c: (round(_rest_angle(fx, *c) % 360.0, 6), rings[c[0]]["radius"], c[0], c[1]))
    perm = np.random.default_rng([fx["seed"], fx.get("index", 0), 7]).permutation(len(every))
    return [every[int(p)] for p in perm]


def grow_schedule(fx, times):
    """{(ring, j): the clip second copy j of ring i pops in} for a mandala's `grow` on `times` (clip seconds): the
    copies in its order dealt into groups, one group a time (`per` copies each, default all of them shared evenly over
    the times; the copies left over join the last), a group's copies `gap` s apart from its time on. Empty without times."""
    times = sorted(times)
    if not times:
        return {}
    order = grow_order(fx)
    per = fx["grow"]["per"] or int(math.ceil(len(order) / len(times)))
    out = {}
    for n, c in enumerate(order):
        g = min(n // per, len(times) - 1)
        out[c] = times[g] + (n - g * per) * fx["grow"]["gap"]
    return out


def copies(fx, t, times, fps, frame0, schedule=None):
    """The copies of a mandala at clip second `t`, in drawing order (the outermost ring first, round each ring), as
    dicts: `ring`, `j`, `x`, `y` (from its centre, short sides of the frame, y down), `angle` (degrees the copy is turned
    counter-clockwise: radial, its top away from the centre, or upright, plus its ring's `angle` and `spin`), `scale`
    (its ring's, bloomed), `pop` (with a grow `schedule`, mandala.grow_schedule: 0 before its time, then the doodles'
    spring to 1 over the grow's `dur`, overshooting 12 %; else 1) and `mirror` (every other copy of a mirrored ring
    flipped left to right)."""
    out = []
    rings = fx["rings"]
    dur = fx["grow"]["dur"] if fx.get("grow") else 1.0
    for i in sorted(range(len(rings)), key=lambda i: (-rings[i]["radius"], i)):
        r = rings[i]
        rot = turn(fx, i, t, times, fps, frame0)
        kr, ks = bloom(fx, i, t, times)
        radius = r["radius"] * kr
        for j in range(r["count"]):
            a = _rest_angle(fx, i, j) + rot
            ar = math.radians(a)
            p = 1.0 if schedule is None else PC.pop(t - schedule.get((i, j), math.inf), dur)
            out.append({"ring": i, "j": j, "x": -radius * math.sin(ar), "y": -radius * math.cos(ar),
                        "angle": (a if r["orient"] == "radial" else 0.0) + r["angle"] + r["spin"] * (t - fx["from"]),
                        "scale": r["scale"] * ks, "pop": p, "mirror": bool(r["mirror"] and j % 2)})
    return out


def sprite_pick(fx, ring, j, n):
    """Which of `n` sprites copy j of a ring shows: the sprites shuffled per ring (seeded by `seed`, the entry and the
    ring) and dealt round it, every one before any comes again."""
    perm = np.random.default_rng([fx["seed"], fx.get("index", 0), ring]).permutation(n)
    return int(perm[j % n])


def shape_of(src, j):
    """The shape copy j of a shape ring draws: its `shape` list round the ring."""
    return src["shapes"][j % len(src["shapes"])]


def shape_colours(src, j, k):
    """(main, accent, ink) of copy j of a shape ring after k hits: its `colors` read at k + j round and round (the
    shape's own colours without them), its `accent` (the shape's own) and `ink`."""
    name = shape_of(src, j)
    cs = src["colors"] or src["own"][name]["colors"]
    return cs[(k + j) % len(cs)], src["accent"] or src["own"][name]["accent"], src["ink"]


def shape_pen(src, j):
    """The width of the ink line to draw round the whole of copy j of a shape ring, of its box's longer side: the
    pen's round a flat shape of a ring with `pen` (its stars and hearts drawn as the kits are), 0 otherwise: a kit's
    shape has the line already."""
    return PEN if src["pen"] and shape_of(src, j) not in PENNED else 0.0


# ================================================================================================= trails
def trail_copies(fx, frame, t, times, fps, frame0):
    """A trail's copies on `frame` at clip second `t`, the oldest first: [(j, frame g it shows, opacity, place on the
    colour ramp)]: copy j (1 the newest) is the figure j x `delay` frames before at opacity `fade` ** j, its colour
    `spread` x (j - 1) entries along the ramp less `cycle` a second and one a hit (eased over `snap`): the colours run
    out along the trail."""
    shift = fx["cycle"] * (t - fx["from"]) + stepped(fx, t, times, fps, frame0)
    return [(j, frame - j * fx["delay"], fx["fade"] ** j, fx["spread"] * (j - 1) - shift)
            for j in range(fx["count"], 0, -1)]


def ramp(colors, u):
    """The colour at place `u` (entries, any real number) of a cyclic ramp: entry floor(u) blended into the next."""
    n = len(colors)
    if n == 1:
        return list(colors[0])
    f = math.floor(u)
    a, b = colors[int(f) % n], colors[(int(f) + 1) % n]
    w = u - f
    return [x + (y - x) * w for x, y in zip(a, b)]


# ================================================================================================= sticker shapes
def _ellipse(cx, cy, rx, ry, angle=0.0, n=48):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    p = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return p @ np.array([[c, s], [-s, c]]) + (cx, cy)


def _turned(p, angle):
    """Points turned `angle` degrees counter-clockwise on screen (y down) about (0, 0)."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return np.asarray(p, float) @ np.array([[c, -s], [s, c]])


def _smooth(p, n=12):
    """A closed Catmull-Rom curve through the points `p`, `n` points a span."""
    p = np.asarray(p, float)
    k = len(p)
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    out = []
    for i in range(k):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % k], p[(i + 2) % k]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.concatenate(out)


def _heart(n=96):
    """A heart, its point down, its box 1 wide, about its box's middle."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    p = np.stack([16.0 * np.sin(t) ** 3, -(13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t))], 1)
    p -= (p.min(0) + p.max(0)) / 2.0
    return p / (p[:, 0].max() - p[:, 0].min())


def _inset(p, k):
    """A polygon shrunk `k` times about its points' mean."""
    p = np.asarray(p, float)
    c = p.mean(0)
    return c + (p - c) * k


def _mirrored(parts):
    """The parts and their mirror images across x = 0."""
    return parts + [(role, kind, np.asarray(p, float) * (-1.0, 1.0), w) for role, kind, p, w in parts]


def _rounded(p, r, n=10):
    """A closed polygon with each corner rounded by an arc of radius `r` tangent to both its sides (a reflex corner by
    one from outside)."""
    p = np.asarray(p, float)
    out = []
    for i in range(len(p)):
        P, A, B = p[i], p[i - 1], p[(i + 1) % len(p)]
        u, v = (A - P) / np.linalg.norm(A - P), (B - P) / np.linalg.norm(B - P)
        half = math.acos(float(np.clip(u @ v, -1.0, 1.0))) / 2.0
        c = P + (u + v) / np.linalg.norm(u + v) * (r / math.sin(half))
        a0, a1 = P + u * (r / math.tan(half)) - c, P + v * (r / math.tan(half)) - c
        t0 = math.atan2(a0[1], a0[0])
        dt = (math.atan2(a1[1], a1[0]) - t0 + math.pi) % (2.0 * math.pi) - math.pi
        ts = t0 + dt * np.linspace(0.0, 1.0, n)
        out.append(c + r * np.stack([np.cos(ts), np.sin(ts)], 1))
    return np.concatenate(out)


def _curve(p0, c, p1, n=16):
    """A quadratic Bezier from p0 to p1 pulled toward c."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - u) ** 2 * np.asarray(p0, float) + 2 * (1 - u) * u * np.asarray(c, float) + u ** 2 * np.asarray(p1, float)


def _clover():
    half = [[0.05, -0.06], [0.12, -0.15], [0.17, -0.25], [0.175, -0.32], [0.14, -0.385], [0.08, -0.405],
            [0.03, -0.375]]
    leaf = _smooth([[0.0, 0.0]] + half + [[0.0, -0.33]] + [[-x, y] for x, y in half[::-1]], n=8)  # a heart, its point in
    parts, marks = [], []
    for a in (45.0, 135.0, 225.0, 315.0):
        parts.append(("main", "fill", _turned(leaf - (0.0, 0.035), a), 0.0))
        marks.append(("dark", "line", _turned([[0.0, -0.08], [0.0, -0.31]], a), 0.014))
        marks.append(("light", "line", _turned([[-0.085, -0.28], [0.0, -0.21], [0.085, -0.28]], a), 0.03))
    stem = _curve([0.0, 0.0], [0.02, 0.42], [0.17, 0.62])                 # out through the gap under the leaves
    return [("dark", "line", stem, 0.055)] + parts + marks + [("dark", "fill", _ellipse(0.0, 0.0, 0.04, 0.04), 0.0)]


def _flower():
    parts = []
    for j in range(6):
        a = 60.0 * j
        parts.append(("main", "fill", _turned(_ellipse(0.0, -0.27, 0.165, 0.25), a), 0.0))
    for j in range(6):
        a = 60.0 * j
        parts.append(("light", "fill", _turned(_ellipse(0.0, -0.33, 0.06, 0.13), a), 0.0))
    parts.append(("accent", "fill", _ellipse(0.0, 0.0, 0.17, 0.17), 0.0))
    parts += [("accent_dark", "fill", _ellipse(0.1 * math.sin(math.radians(72 * j + 20)),
                                               -0.1 * math.cos(math.radians(72 * j + 20)), 0.026, 0.026, n=16), 0.0)
              for j in range(5)]
    return parts


def _star():
    a = np.radians(np.arange(10) * 36.0)
    rad = np.where(np.arange(10) % 2 == 0, 0.5, 0.25)
    pts = _rounded(np.stack([rad * np.sin(a), -rad * np.cos(a)], 1), 0.055)    # a puffy star, its points rounded
    shine = _ellipse(-0.08, -0.13, 0.05, 0.1, angle=-30.0, n=24)
    return [("main", "fill", pts, 0.0), ("light", "fill", shine, 0.0)]


def _heart_parts():
    h = _heart()
    shine = _ellipse(-0.24, -0.14, 0.06, 0.11, angle=-35.0, n=24)
    return [("main", "fill", h, 0.0), ("light", "fill", shine, 0.0)]


def _butterfly():
    fore = _smooth([[0.03, -0.06], [0.1, -0.22], [0.25, -0.37], [0.43, -0.42], [0.48, -0.31], [0.42, -0.13],
                    [0.28, -0.03], [0.12, 0.0]])
    hind = _smooth([[0.03, 0.01], [0.18, 0.03], [0.33, 0.09], [0.38, 0.21], [0.32, 0.34], [0.2, 0.39], [0.09, 0.33],
                    [0.03, 0.18]])
    wings = [("dark", "fill", fore, 0.0), ("dark", "fill", hind, 0.0),
             ("main", "fill", _inset(fore, 0.8), 0.0), ("main", "fill", _inset(hind, 0.78), 0.0),
             ("accent", "fill", _ellipse(0.26, -0.2, 0.05, 0.04, angle=-30.0, n=24), 0.0),
             ("accent", "fill", _ellipse(0.24, 0.22, 0.065, 0.065, n=32), 0.0),
             ("ink", "fill", _ellipse(0.24, 0.22, 0.028, 0.028, n=20), 0.0)]
    wings += [("light", "fill", _ellipse(x, y, 0.022, 0.022, n=16), 0.0)
              for x, y in ((0.43, -0.33), (0.4, -0.24), (0.36, -0.15))]
    antenna = [("ink", "line", np.array([[0.008, -0.2], [0.06, -0.33], [0.13, -0.42]]), 0.014),
               ("ink", "fill", _ellipse(0.13, -0.42, 0.022, 0.022, n=16), 0.0)]
    body = [("ink", "fill", _ellipse(0.0, 0.03, 0.036, 0.2, n=32), 0.0), ("ink", "fill", _ellipse(0.0, -0.19, 0.042, 0.04,
                                                                                                    n=24), 0.0)]
    return _mirrored(wings + antenna) + body


def _closed(p):
    """A polygon's points and the first again: its outline as a line."""
    p = np.asarray(p, float)
    return np.concatenate([p, p[:1]])


def _drawn(role, p, width=PEN):
    """A polygon filled in `role` with the pen's ink line round it."""
    return [(role, "fill", np.asarray(p, float), 0.0), ("ink", "line", _closed(p), width)]


def _box(x0, y0, x1, y1, r):
    """A box, its corners rounded by `r`."""
    return _rounded([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], r)


def _tilted(parts, angle):
    """The parts turned `angle` degrees counter-clockwise: a sticker drawn upright and laid askew."""
    return [(role, kind, _turned(p, angle), w) for role, kind, p, w in parts]


def _crescent():
    """A crescent moon, its disc 1 wide about (0, 0), its horns to the right."""
    d, r1 = 0.2, 0.41                                        # the bite out of the disc: a circle r1 wide, d to the right
    x = (d * d + 0.25 - r1 * r1) / (2.0 * d)
    y = math.sqrt(0.25 - x * x)                              # the horns: where the two circles cross
    a, b = math.atan2(y, x), math.atan2(y, x - d)
    ta, tb = np.linspace(a, 2.0 * math.pi - a, 56), np.linspace(2.0 * math.pi - b, b, 44)
    return np.concatenate([0.5 * np.stack([np.cos(ta), np.sin(ta)], 1),
                           ((d, 0.0) + r1 * np.stack([np.cos(tb), np.sin(tb)], 1))[1:-1]])


def _moon():
    ts = np.radians(np.linspace(195.0, 250.0, 16))
    shine = 0.44 * np.stack([np.cos(ts), np.sin(ts)], 1)
    asleep = _curve([-0.43, 0.0], [-0.375, 0.07], [-0.32, 0.0])             # its eye, shut
    eye = [("light", "line", shine, 0.03), ("ink", "line", asleep, 0.02),
           ("accent", "fill", _ellipse(-0.37, 0.13, 0.05, 0.032, n=24), 0.0)]
    return _tilted(_drawn("main", _crescent()) + eye, 24.0)


def _book():
    ribbon = np.array([[0.1, 0.4], [0.21, 0.4], [0.21, 0.64], [0.155, 0.585], [0.1, 0.64]])   # its bookmark, hanging out
    parts = _drawn("accent", ribbon, 0.016)
    parts += _drawn("paper", _box(-0.27, -0.38, 0.41, 0.5, 0.04), 0.016)                       # the pages, under the cover
    parts += [("ink", "line", np.array([[0.35, y], [0.4, y]]), 0.008) for y in (-0.2, 0.0, 0.2)]
    parts += _drawn("main", _box(-0.37, -0.46, 0.33, 0.43, 0.05))
    parts += [("dark", "line", np.array([[-0.26, -0.44], [-0.26, 0.41]]), 0.026)]              # the spine's hinge
    parts += _drawn("accent", _turned(0.3 * _crescent(), 30.0) + (0.05, -0.12), 0.014)          # a moon on its cover
    parts += [("light", "line", np.array([[-0.1, 0.17], [0.2, 0.17]]), 0.024),                 # and its title
              ("light", "line", np.array([[-0.04, 0.25], [0.14, 0.25]]), 0.024)]
    return _tilted(parts, 10.0)


def _teacup():
    ts = np.radians(np.linspace(-100.0, 130.0, 32))                                            # its ends behind the cup
    handle = (0.27, 0.1) + np.stack([0.1 * np.cos(ts), 0.105 * np.sin(ts)], 1)
    parts = _drawn("paper", _ellipse(0.0, 0.31, 0.43, 0.075, n=56))                             # the saucer
    parts += [("ink", "line", handle, 0.082), ("paper", "line", handle, 0.038)]               # the handle, behind the cup
    side = _curve([0.28, -0.06], [0.275, 0.27], [0.07, 0.29])
    cup = np.concatenate([side, side[::-1] * (-1.0, 1.0)])
    parts += _drawn("paper", cup)
    parts += _drawn("main", _ellipse(0.0, -0.06, 0.28, 0.055, n=48), 0.016)                     # the tea
    parts += [("accent", "fill", _ellipse(x, 0.11 - 0.9 * x * x, 0.022, 0.022, n=14), 0.0) for x in (-0.13, 0.0, 0.13)]
    ys = np.linspace(-0.2, -0.47, 24)
    for x0, s in ((-0.08, 1.0), (0.09, -1.0)):                                                 # two wisps of steam
        parts.append(("light", "line", np.stack([x0 + 0.035 * s * np.sin((ys + 0.2) * 23.0), ys], 1), 0.03))
    return parts


def _potion():
    level, r, cy, lean = 0.14, 0.3, 0.18, 14.0               # a round flask: its liquid's surface, its bulb, its lean
    parts = _drawn("accent", _box(-0.07, -0.49, 0.07, -0.36, 0.025), 0.016)                     # the cork
    parts += _drawn("paper", _box(-0.09, -0.32, 0.09, 0.0, 0.01))                               # the neck
    parts += _drawn("paper", _ellipse(0.0, cy, r, r, n=64))
    parts += [("paper", "fill", _box(-0.078, -0.16, 0.078, -0.06, 0.005), 0.0)]             # the neck opens into the bulb
    parts += _drawn("paper", _box(-0.13, -0.38, 0.13, -0.3, 0.03))                              # its lip
    a = math.asin((level - cy) / (r - PEN))
    ts = np.linspace(a, math.pi - a, 40)
    liquid = (0.0, cy) + _turned((r - PEN) * np.stack([np.cos(ts), np.sin(ts)], 1), lean)     # level as the flask leans
    parts += [("main", "fill", liquid, 0.0), ("dark", "line", liquid[[0, -1]], 0.016)]
    parts += [("paper", "fill", _ellipse(x, y, k, k, n=14), 0.0) for x, y, k in ((-0.1, 0.3, 0.03), (0.07, 0.24, 0.02),
                                                                                 (0.02, 0.38, 0.016))]
    shine = np.radians(np.linspace(200.0, 245.0, 12))
    parts += [("accent", "fill", _ellipse(0.04, 0.02, 0.022, 0.022, n=14), 0.0),               # a bubble on its way up
              ("light", "line", (0.0, cy) + 0.23 * np.stack([np.cos(shine), np.sin(shine)], 1), 0.03)]
    return _tilted(parts, -lean)


def _rabbit():
    parts = []
    for s in (-1.0, 1.0):                                    # the ears behind the head, their insides in the colour
        for role, kind, p, w in _drawn("paper", _ellipse(0.0, 0.0, 0.1, 0.3)) + [
                ("main", "fill", _ellipse(0.0, 0.03, 0.045, 0.2, n=32), 0.0)]:
            parts.append((role, kind, _turned(p, -9.0 * s) + (0.15 * s, -0.24), w))
    parts += _drawn("paper", _ellipse(0.0, 0.2, 0.31, 0.25, n=64))
    for s in (-1.0, 1.0):                                    # eyes with a glint, cheeks
        parts += [("accent", "fill", _ellipse(0.125 * s, 0.16, 0.034, 0.044, n=24), 0.0),
                  ("paper", "fill", _ellipse(0.125 * s - 0.012, 0.145, 0.012, 0.012, n=12), 0.0),
                  ("light", "fill", _ellipse(0.215 * s, 0.26, 0.05, 0.03, n=24), 0.0)]
    mouth = np.array([[-0.05, 0.3], [-0.025, 0.315], [0.0, 0.285], [0.025, 0.315], [0.05, 0.3]])
    return parts + [("main", "fill", _ellipse(0.0, 0.235, 0.028, 0.02, n=16), 0.0), ("ink", "line", mouth, 0.014)]


def _carrot():
    root = _smooth([[-0.13, -0.2], [0.0, -0.25], [0.13, -0.2], [0.115, 0.0], [0.055, 0.3], [0.0, 0.47], [-0.055, 0.3],
                    [-0.115, 0.0]], n=8)
    parts = []
    for a in (-32.0, 32.0, 0.0):                             # three leaves fanned from its top, the middle one over
        parts += _drawn("accent", _turned(_ellipse(0.0, -0.17, 0.05, 0.15, n=32), -a) + (0.0, -0.2))
    parts += _drawn("main", root)
    parts += [("dark", "line", np.array(m), 0.016) for m in ([[-0.11, -0.07], [-0.03, -0.055]],
                                                             [[0.1, 0.07], [0.035, 0.085]],
                                                             [[-0.06, 0.22], [-0.005, 0.235]])]
    return _tilted(parts + [("light", "fill", _ellipse(-0.055, -0.12, 0.018, 0.06, angle=4.0, n=20), 0.0)], -28.0)


def _dango():
    parts = [("ink", "line", np.array([[0.0, -0.29], [0.0, 0.68]]), 0.03)]     # the stick, its tip inside the top one
    for y, role in ((-0.29, "main"), (0.0, "paper"), (0.29, "accent")):
        parts += _drawn(role, _ellipse(0.0, y, 0.16, 0.15, n=40))
        parts.append(("paper", "fill", _ellipse(-0.065, y - 0.06, 0.03, 0.045, angle=35.0, n=16), 0.0))
    return _tilted(parts, -24.0)


def _daisy():
    parts = []
    for j in range(10):
        parts += _drawn("paper", _turned(_ellipse(0.0, -0.3, 0.083, 0.2, n=32), 36.0 * j), 0.018)
    parts += _drawn("main", _ellipse(0.0, 0.0, 0.15, 0.15, n=40), 0.018)
    return parts + [("dark", "fill", _ellipse(0.07 * math.sin(math.radians(72 * j + 15)),
                                              -0.07 * math.cos(math.radians(72 * j + 15)), 0.018, 0.018, n=12), 0.0)
                    for j in range(5)]


def _cassette():
    parts = _drawn("main", _box(-0.5, -0.32, 0.5, 0.32, 0.055)) + _drawn("paper", _box(-0.42, -0.25, 0.42, 0.1, 0.03), 0.016)
    parts += [("accent", "line", np.array([[-0.35, -0.185], [0.35, -0.185]]), 0.036),          # the label's stripe
              ("ink", "fill", _box(-0.27, -0.115, 0.27, 0.045, 0.045), 0.0)]                   # the window
    for s in (-1.0, 1.0):                                    # the reels
        parts += [("paper", "fill", _ellipse(0.15 * s, -0.035, 0.056, 0.056, n=28), 0.0),
                  ("ink", "fill", _ellipse(0.15 * s, -0.035, 0.022, 0.022, n=14), 0.0)]
    parts += _drawn("dark", np.array([[-0.3, 0.32], [-0.24, 0.17], [0.24, 0.17], [0.3, 0.32]]), 0.016)
    parts += [("ink", "fill", _ellipse(0.12 * s, 0.25, 0.022, 0.022, n=14), 0.0) for s in (-1.0, 1.0)]
    return parts + [("dark", "fill", _ellipse(0.445 * sx, 0.265 * sy, 0.017, 0.017, n=12), 0.0)
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def _pedal():
    parts = []
    for s in (-1.0, 1.0):                                    # the jacks, out of its sides
        parts += _drawn("dark", _box(min(0.27 * s, 0.38 * s), -0.02, max(0.27 * s, 0.38 * s), 0.1, 0.02), 0.016)
    body = _rounded([[-0.27, -0.5], [0.27, -0.5], [0.32, 0.5], [-0.32, 0.5]], 0.07)
    parts += _drawn("main", body)
    for x, a in ((-0.16, -40.0), (0.0, 10.0), (0.16, 55.0)):                # three knobs, each turned its own way
        tip = (x + 0.05 * math.sin(math.radians(a)), -0.36 - 0.05 * math.cos(math.radians(a)))
        parts += [("ink", "fill", _ellipse(x, -0.36, 0.062, 0.062, n=28), 0.0),
                  ("paper", "line", np.array([[x, -0.36], tip]), 0.018)]
    band = np.array([[-0.283, -0.22], [0.283, -0.22], [0.294, -0.02], [-0.294, -0.02]])
    xs = np.linspace(-0.2, 0.2, 41)
    wave = np.stack([xs, -0.12 + 0.05 * np.sin((xs + 0.2) / 0.4 * 4.0 * math.pi)], 1)
    parts += [("paper", "fill", band, 0.0), ("ink", "line", band[:2], 0.014), ("ink", "line", band[2:], 0.014),
              ("accent", "line", wave, 0.024)]                                # its label: a band with a wave on it
    parts += _drawn("accent", _ellipse(0.0, 0.07, 0.026, 0.026, n=16), 0.012)                   # the lamp
    parts += _drawn("light", _ellipse(0.0, 0.29, 0.11, 0.11, n=40), 0.016)                      # the switch
    parts += _drawn("paper", _ellipse(0.0, 0.29, 0.062, 0.062, n=28), 0.014)
    return parts + [("ink", "line", _closed(body), PEN)]


def _cherry():
    top = np.array([0.04, -0.44])
    parts = [("accent_dark", "line", _curve(top, [-0.2, -0.2], [-0.21, 0.06]), 0.03),
             ("accent_dark", "line", _curve(top, [0.2, -0.2], [0.23, 0.13]), 0.03)]
    leaf = _smooth([[0.04, -0.44], [0.16, -0.54], [0.33, -0.55], [0.46, -0.47], [0.33, -0.39], [0.17, -0.39]], n=8)
    parts += _drawn("accent", leaf, 0.018) + [("accent_dark", "line", np.array([[0.06, -0.445], [0.4, -0.47]]), 0.014)]
    for cx, cy, r in ((-0.21, 0.24, 0.21), (0.24, 0.3, 0.2)):
        parts += _drawn("main", _ellipse(cx, cy, r, r * 0.96))
        parts.append(("light", "fill", _ellipse(cx - 0.08, cy - 0.08, 0.035, 0.06, angle=40.0, n=20), 0.0))
    return parts


_SHAPE_PARTS = {"clover": _clover, "flower": _flower, "heart": _heart_parts, "star": _star, "butterfly": _butterfly,
                "moon": _moon, "rabbit": _rabbit, "carrot": _carrot, "dango": _dango, "daisy": _daisy,
                "cassette": _cassette, "pedal": _pedal, "cherry": _cherry, "book": _book, "teacup": _teacup,
                "potion": _potion}


def shape_parts(name):
    """The parts of sticker shape `name` in the order they are painted: [(role, "fill" or "line", (n, 2) points, line
    width)], about the shape's middle (0, 0), x right and y down, scaled so its box (lines' widths in) is 1 on its
    longer side. Roles: `main` (the copy's colour), `dark` and `light` (it darkened toward the ink and lightened toward
    white), `accent` (and `accent_dark`), `ink`, `paper` (white: a rabbit's fur, a daisy's petals, a label)."""
    if name not in _SHAPE_PARTS:
        raise ValueError(f"no shape {name!r} (shapes: {', '.join(SHAPES)})")
    parts = _SHAPE_PARTS[name]()
    lo, hi = shape_box(parts)
    k = 1.0 / float(max(hi - lo))
    return [(role, kind, np.asarray(p, float) * k, w * k) for role, kind, p, w in parts]


def shape_box(parts):
    """(min (x, y), max (x, y)) of shape parts, their lines' widths in."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for _, _, p, w in parts:
        p = np.asarray(p, float)
        lo, hi = np.minimum(lo, p.min(0) - w / 2.0), np.maximum(hi, p.max(0) + w / 2.0)
    return lo, hi
