"""Echoes of a sheet's live sticker, bpy-free (docs/design.md: Frame effects, Mandalas and trails): the keys a `trails`
entry takes when it follows a sheet's sticker (`sheet`, `shrink`, `train`, `spiral`) checked, where a train drops its
copies and where a spiral lays them; mkmmd/echoes.py draws them.

    trails with `sheet`   the copies are that sheet's live sticker as the sheet laid it on frames before (its plate's
                          figure, its die-cut rim, its placement, keys, slap and drift: wherever its 2D keys moved it),
                          each her picture washed toward its ink; put the trail before the sheet in the table and the
                          sticker lies over it. `delay` frames apart, or:
    `train`               a copy dropped each time the sticker has travelled `spacing` of its own height on the screen,
                          living `life` seconds, fading and shrinking as it ages, the newest nearest her: fast, a long
                          train of separate figures; slow, two or three; still, none
    `spiral`              the copies wound out round her middle instead, a nautilus: the first `out` of her height from
                          it, each next `turn` degrees further round, `widen` times further out and `scale` times the
                          size, `near` to `far` opaque, each `delay` frames further back in her own motion (the arm
                          lags her); the arm turns `spin` degrees a second and breathes with the beat; the copies arrive
                          one by one from the trail's `from` and are drawn back into her along the spiral over its last
                          `gather` seconds

Places are frame px (y down), angles degrees counter-clockwise on screen, times clip seconds."""
import math

from . import tween as TW

TRAIN = {"spacing": 0.17, "life": 0.5}
SPIRAL = {"out": 0.6, "turn": 50.0, "widen": 1.25, "scale": 0.85, "near": 0.8, "far": 0.2, "spin": 6.0,
          "phase": 0.0, "squash": 1.0, "breathe": 0.06, "beat": 0.6935, "at": 0.325, "arrive": 0.3, "gather": 0.5}
POP = 0.25             # seconds a spiral's copy takes to pop out to its size as it arrives (a 10 % overshoot)


def normalize(FX, spec, out, palette, what):
    """A trail's `sheet`, `shrink`, `train` and `spiral` checked and filled in, in place (its other keys are
    mandala's)."""
    if not isinstance(out["sheet"], str) or not out["sheet"].strip():
        raise FX.FxError(f"{what}: sheet = {out['sheet']!r}: the `name` of a sheet whose live sticker leaves the trail")
    if out.get("mask") is not None:
        raise FX.FxError(f"{what}: a sheet's trail lies under its sticker (put it before the sheet in the table): no "
                         f"`mask`")
    out["shrink"] = FX._num(out["shrink"], f"{what} shrink", 0.0, 0.9)
    out["train"] = _table(FX, out["train"], TRAIN, what, "train")
    out["spiral"] = _table(FX, out["spiral"], SPIRAL, what, "spiral")
    if out["train"] is not None and out["spiral"] is not None:
        raise FX.FxError(f"{what}: a `train` (copies dropped as she travels) or a `spiral` (copies wound round her), "
                         f"not both")
    if out["train"] is not None:
        t = out["train"]
        out["train"] = {"spacing": FX._num(t["spacing"], f"{what} train spacing", 0.01, 10.0),
                        "life": FX._num(t["life"], f"{what} train life", 0.02, 10.0)}
    if out["spiral"] is not None:
        s = out["spiral"]
        lim = {"out": (0.0, 20.0), "widen": (0.5, 3.0), "scale": (0.05, 2.0), "near": (0.0, 1.0), "far": (0.0, 1.0),
               "breathe": (0.0, 1.0), "beat": (0.05, 10.0), "arrive": (0.0, 5.0), "gather": (0.0, 5.0),
               "squash": (0.1, 10.0)}
        out["spiral"] = {k: FX._num(s[k], f"{what} spiral {k}", *lim.get(k, (None, None))) for k in SPIRAL}


def _table(FX, v, defaults, what, name):
    if v is None or v is False:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict) or set(v) - set(defaults):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected {{{', '.join(defaults)}}}")
    return {**defaults, **v}


def link(fxs):
    """Each trail that follows a sheet (and its aspect entries) gets `sheet_index`, the table index of the sheet of its
    `sheet` name that lays a live sticker of its own; an FxError when there is none."""
    from . import framefx as FX
    for fx in fxs:
        if fx["kind"] != "trails" or fx.get("sheet") is None:
            continue
        got = [s for s in fxs if s["kind"] == "sheet" and s.get("name") == fx["sheet"]]
        if not got or got[0].get("objects") is None:
            raise FX.FxError(f"fx {fx['index']} (trails): sheet = {fx['sheet']!r}: no sheet of that `name` lays a live "
                             f"sticker (`objects`)")
        for e in [fx, *fx["aspect"].values()]:
            if e.get("sheet") != fx["sheet"]:
                raise FX.FxError(f"fx {fx['index']} (trails): an aspect.<output> table may not change the sheet")
            e["sheet_index"] = got[0]["index"]


def reach(fx, fps):
    """How many frames back a sheet's trail reads its sticker: a train's copies' life, a trail's or a spiral's
    `count` x `delay`."""
    if fx.get("train") is not None:
        return int(math.ceil(fx["train"]["life"] * fps)) + 1
    return fx["count"] * fx["delay"] + 1


def safe(fx, frame):
    """Whether `frame` may start a run of mk post's frames: not inside a train after its first frame (where it drops
    its copies hangs on the path the sticker took since)."""
    return fx.get("train") is None or frame <= fx["f0"]


# ================================================================================================= a train
def travel(s, last, c, h, spacing):
    """One frame further along a train's path: (the travel so far, counted in `spacing` x the sticker's own height `h`
    px; its middle `c` px (the last one it had, None where it was not there); the index of the copy dropped on this
    frame, or None). A copy is dropped where the travel passes a whole number."""
    if c is None or h is None or h <= 0:
        return s, None, None
    if last is None:
        return s, c, None
    after = s + math.hypot(c[0] - last[0], c[1] - last[1]) / (spacing * h)
    return after, c, (int(math.floor(after)) if math.floor(after) > math.floor(s) else None)


def dropped(path, spacing):
    """The copies a train drops along a sticker's path [(frame, (x, y) its middle px or None, its height px)], in order
    from the trail's first frame: [(index along the path, the frame it is dropped on)] (travel()'s rule)."""
    out, s, last = [], 0.0, None
    for g, c, h in path:
        s, last, k = travel(s, last, c, h, spacing)
        if k is not None:
            out.append((k, g))
    return out


AGE = 0.8              # how a train's copy fades as it ages: (1 - age / life) ** AGE, held up most of its life


def train_copies(fx, frame, drops, fps):
    """The copies of a train shown on `frame` from its `drops` (dropped()'s), the oldest first: [(index along the path,
    frame it shows, opacity, scale)]: those dropped less than its `life` before (never on `frame` itself: she is there),
    the newest `count` of them, each at `fade` x (1 - age / life) ** AGE opacity and (1 - `shrink` x age / life) of its
    size."""
    life = fx["train"]["life"] * fps
    live = [(k, g) for k, g in drops if 0 < frame - g < life]
    live = live[-fx["count"]:]
    out = []
    for k, g in live:
        u = (frame - g) / life
        out.append((k, g, fx["fade"] * (1.0 - u) ** AGE, 1.0 - fx["shrink"] * u))
    return out


# ================================================================================================= a spiral
def spiral_at(spiral, j, count, t, t0, t1):
    """Copy j (1 the nearest) of `count` of a spiral trail over clip seconds t0..t1 at second `t`: (its distance from
    her middle, her heights; its angle round, degrees counter-clockwise from the right; its scale, her size; its
    opacity). It arrives `arrive` x (j - 1) s after t0, popping out over POP; its distance breathes `breathe` of itself
    with the beat (`beat` s, out on `at`), the farther copies later; over the last `gather` s before t1 it slides back
    along the spiral into her middle, shrinking and fading (eased in)."""
    s = float(j - 1)
    shown = spiral["near"] + (spiral["far"] - spiral["near"]) * (s / max(count - 1, 1))
    born = t - (t0 + spiral["arrive"] * s)
    if born < 0.0:
        return 0.0, 0.0, 0.0, 0.0
    grow = TW.pop(min(born / POP, 1.0), 0.1)
    shown *= min(born / (0.5 * POP), 1.0)
    g = 0.0
    if spiral["gather"] > 0.0 and t > t1 - spiral["gather"]:
        g = TW.ease(min((t - (t1 - spiral["gather"])) / spiral["gather"], 1.0), "in")
        s *= 1.0 - g                                             # back along the arm toward the first place
    r = spiral["out"] * spiral["widen"] ** s * (1.0 - g)
    r *= 1.0 + spiral["breathe"] * math.sin(2.0 * math.pi * ((t - spiral["at"]) / spiral["beat"] + 0.25 - 0.1 * s))
    a = spiral["phase"] + spiral["spin"] * (t - t0) + s * spiral["turn"]
    return r, a, spiral["scale"] ** (s + 1.0) * grow * (1.0 - 0.5 * g), shown * (1.0 - g)


def spiral_map(spiral, j, count, t, t0, t1, own, middle, height):
    """(the 2 x 3 map, frame px to frame px, that lays copy j of a spiral trail at clip second `t`, its opacity): the
    copy, whose own middle is `own`, scaled about it and moved to its place round `middle` (spiral_at), `height` px
    her height there."""
    r, a, k, shown = spiral_at(spiral, j, count, t, t0, t1)
    rad = math.radians(a)
    to = (middle[0] + r * height * spiral["squash"] * math.cos(rad), middle[1] - r * height * math.sin(rad))
    return [[k, 0.0, to[0] - k * own[0]], [0.0, k, to[1] - k * own[1]]], shown
