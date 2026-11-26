"""Skies, bpy-free (docs/design.md: Frame effects, Skies): `[[fx]] kind = "sky"` checked and timed, and the `smear` a
table (mkmmd.core.table) takes from it; mkmmd/sky.py draws them.

A sky is real footage seen through cameras of its own, clip after clip: `clips`, segments in time order, each from its
`t` until the next one's (the last to the span's end). A segment is a clip (`clip`, read by mkmmd.footage: its second
`in` on the segment's `t`, playing at `rate`, `loop`) seen through its `view` (a table's camera: mkmmd.core.table) and
graded by the sky's `grade` (a table's grade by warmth: the stretched luma through a cool and a warm ramp, mixed by each
pixel's warmth) with its own `lean` (one city warm, the other cool); or the clear sky, `paper` (Dawn's blue paper, pale
at the top); or `layers` of those, each over the ones before it in a box of the frame (`area`, `feather`: a city hanging
upside down over another); or a table's window (`table`, the name of a sheet with a `clip` that draws it: the sky draws
nothing there and smears out of it).

On a segment's `t` the picture before it is held and smeared into it (`smear`): the segment before's last frame, or
`from`'s (a table's ground: its clip through its view and grade, never what lies on it; or another sky's picture),
carried from frame to frame along the new segment's own motion (its picture's optical flow, `gain` times, averaged over
`block` squares so it moves in chunks, as a codec's motion vectors do) or along a made-up `flow` (a mosh's: a drift, a
zoom, a swirl, noise), while the new picture breaks through block by block (`patch`: in patches; each block whole on the
frame it breaks), all of it through by `dur`, eased; the held picture's colours drift toward the new one's and a new
block keeps a little of the old one's for a while (`bleed`). The first frame of the new motion is the segment's own:
the change lands on its frame. A table takes the same `smear`, from a named sky or table, onto its own ground.

`melt` streaks the picture along its columns over a box of the frame on times: with `run = "light"` what is darker than
`level` runs into the light beside it, `dir` up (a city's towers melting up into the sky: the column under each edge
stretched up and fading) or down; with `run = "all"` every pixel is smeared along its column, a motion streak. Each
column takes its own share of `length` (frame heights) in seeded bands, so they run as streaks, times `strength`.

Everything is a function of the clip second: any frame comes out the same in any order (a smear's held picture is worked
out again from its start when frames come out of order)."""
from . import table as TB
from . import tween as TW

KINDS = ("sky",)
KEYS = {"sky": {"grade", "clips", "melt", "seed"}}
DEFAULTS = {"sky": {"grade": None, "melt": None, "seed": 1}}
LAYER = {"clip": None, "in": 0.0, "rate": 1.0, "loop": False, "view": None, "lean": None, "at": None, "stretch": None,
         "melt": None, "keep": None, "paper": None, "area": None, "feather": 0.0}
OVER = ("area", "feather", "keep")       # a layer's own: how it lies over the layers before it
SEGMENT = {"t": None, "layers": None, "table": None, "smear": None, **{k: v for k, v in LAYER.items() if k not in OVER}}
KEEPS = ("dark",)                        # a layer that keeps only what is darker than its own sky round it (a flock)
PAPER = {"top": "pine:foam:0.4", "bottom": "foam:surface:0.45", "mottle": 0.5, "tooth": 1.0, "seed": 1}
SMEAR = {"dur": 0.5, "block": 16.0, "gain": 2.0, "bleed": 0.5, "flow": None, "jump": None, "from": None, "ease": "in",
         "patch": 0.6, "seed": 1}
SMEAR_EASES = ("in", "out", "inout", "linear")
MELT = {"dir": "up", "run": "light", "length": 0.3, "level": 0.78, "area": None, "feather": 0.0, "strength": 1.0,
        "seed": 1}
MELT_DIRS = ("up", "down")
MELT_RUNS = ("light", "all")


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def _table(FX, v, defaults, what, name):
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table ({', '.join(sorted(defaults))})")
    extra = sorted(set(v) - set(defaults))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {sorted(defaults)})")
    return {**defaults, **v}


def _box(FX, v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        raise FX.FxError(f"{what} = {v!r}: expected [x0, y0, x1, y1] (frame fractions, 0, 0 the top left)")
    b = [FX._num(x, what) for x in v]
    if b[2] <= b[0] or b[3] <= b[1]:
        raise FX.FxError(f"{what} = {b}: x1 must be right of x0 and y1 below y0")
    return b


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """A sky's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    g = spec.get("grade")
    if g is not None and not isinstance(g, dict):
        raise FX.FxError(f"{what}: grade = {g!r}: a table of a grade by warmth's keys (cool, warm, lean, warmth, "
                         f"stretch, at, soften)")
    out["grade"] = TB._grade(FX, {"lean": 0.0, **(g or {})}, palette, what)
    segs = spec.get("clips")
    if not isinstance(segs, (list, tuple)) or not segs or not all(isinstance(s, dict) for s in segs):
        raise FX.FxError(f"{what}: `clips` is required: the segments, [{{t, clip, in, rate, view, lean, smear}}, ...] "
                         f"(or {{t, paper}}, {{t, layers}}, {{t, table}})")
    rows = sorted((_segment(FX, s, palette, f"{what} clips[{i}]") for i, s in enumerate(segs)), key=lambda s: s["t"])
    if any(a["t"] == b["t"] for a, b in zip(rows, rows[1:])):
        raise FX.FxError(f"{what}: clips: two segments start on the same second")
    if rows[0]["t"] > out["from"] + 1e-9:
        raise FX.FxError(f"{what}: clips: the first segment starts at {rows[0]['t']:g}, after the sky's `from` "
                         f"({out['from']:g}): give one from its start")
    if rows[-1]["t"] >= out["to"]:
        raise FX.FxError(f"{what}: clips: a segment starts at {rows[-1]['t']:g}, at or after the sky's `to`")
    for a, b in zip(rows, rows[1:] + [None]):
        a["t1"] = out["to"] if b is None else b["t"]
    for i, s in enumerate(rows):
        sm = s["smear"]
        if sm is None:
            continue
        if i == 0 and sm["from"] is None:
            raise FX.FxError(f"{what}: clips: the first segment's smear has nothing before it in the sky: give its "
                             f"`from` (a table's or a sky's name)")
        if s["t"] + sm["dur"] > s["t1"] + 1e-9:
            raise FX.FxError(f"{what}: clips: the smear at {s['t']:g} lasts {sm['dur']:g} s, past the segment's end "
                             f"({s['t1']:g})")
        if s["table"] is not None:
            raise FX.FxError(f"{what}: clips: the segment at {s['t']:g} is the table {s['table']!r}'s: give the smear "
                             f"into it to that table (its own `smear`)")
    out["clips"] = rows
    out["melt"] = [_melt(FX, m, f"{what} melt[{i}]") for i, m in enumerate(_list(spec.get("melt")))]
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


def _list(v):
    return [] if v is None else [v] if isinstance(v, dict) else list(v)


def _segment(FX, v, palette, what):
    s = _table(FX, v, SEGMENT, what, "segment")
    if s["t"] is None:
        raise FX.FxError(f"{what}: `t` (the clip second it starts) is required")
    s["t"] = FX._num(s["t"], f"{what} t")
    given = [k for k in ("clip", "paper", "layers", "table") if s[k] is not None]
    if len(given) != 1:
        raise FX.FxError(f"{what}: a segment is one of `clip`, `paper`, `layers` or `table` (given: "
                         f"{', '.join(given) or 'none'})")
    out = {"t": s["t"], "smear": None if s["smear"] is None else smear(FX, s["smear"], f"{what} smear"),
           "table": None, "layers": None}
    if s["table"] is not None:
        if not isinstance(s["table"], str) or not s["table"].strip():
            raise FX.FxError(f"{what}: table = {s['table']!r}: the `name` of a sheet with a `clip`")
        lone = sorted(k for k in LAYER if k not in OVER and v.get(k) is not None)
        if lone:
            raise FX.FxError(f"{what}: {', '.join(lone)}: a table's segment is drawn by the table")
        out["table"] = s["table"]
        return out
    if s["layers"] is not None:
        ls = s["layers"]
        if not isinstance(ls, (list, tuple)) or not ls or not all(isinstance(x, dict) for x in ls):
            raise FX.FxError(f"{what}: layers = {ls!r}: a list of layers ({{clip, ...}} or {{paper}}, with `area`)")
        lone = sorted(k for k in LAYER if k not in OVER and v.get(k) is not None)
        if lone:
            raise FX.FxError(f"{what}: {', '.join(lone)} belong to its layers")
        out["layers"] = [_layer(FX, x, palette, f"{what} layers[{i}]") for i, x in enumerate(ls)]
        first = sorted(k for k in ("area", "keep") if out["layers"][0][k] is not None)
        if first:
            raise FX.FxError(f"{what}: layers[0]: the first layer is the ground under the others: it covers the frame "
                             f"(leave out its {', '.join(first)})")
    else:
        out["layers"] = [_layer(FX, {k: x for k, x in v.items() if k in LAYER}, palette, what)]
    return out


def _layer(FX, v, palette, what):
    s = _table(FX, v, LAYER, what, "layer")
    if (s["clip"] is None) == (s["paper"] is None):
        raise FX.FxError(f"{what}: a layer is a `clip` or the `paper`, one of the two")
    out = {"clip": None, "paper": None, "keep": None, "melt": [],
           "area": None if s["area"] is None else _box(FX, s["area"], f"{what} area"),
           "feather": FX._num(s["feather"], f"{what} feather", 0.0, 1.0)}
    if s["paper"] is not None:
        lone = sorted(k for k in ("in", "rate", "loop", "view", "lean", "at", "stretch", "keep", "melt")
                      if v.get(k) is not None)
        if lone:
            raise FX.FxError(f"{what}: {', '.join(lone)} belong to a clip, not the paper")
        p = _table(FX, {} if s["paper"] is True else s["paper"], PAPER, what, "paper")
        out["paper"] = {"top": FX._colour(p["top"], palette, f"{what} paper top"),
                        "bottom": FX._colour(p["bottom"], palette, f"{what} paper bottom"),
                        "mottle": FX._num(p["mottle"], f"{what} paper mottle", 0.0, 4.0),
                        "tooth": FX._num(p["tooth"], f"{what} paper tooth", 0.0, 4.0),
                        "seed": FX._num(p["seed"], f"{what} paper seed", lo=0, integer=True)}
        return out
    if not isinstance(s["clip"], str) or not s["clip"].strip():
        raise FX.FxError(f"{what}: clip = {s['clip']!r}: a video file (project-relative or absolute) or an asset slug")
    if not isinstance(s["loop"], bool):
        raise FX.FxError(f"{what}: loop is true or false")
    if s["keep"] is not None and s["keep"] not in KEEPS:
        raise FX.FxError(f"{what}: keep {s['keep']!r} is none of {', '.join(KEEPS)}")
    stretch = None
    if s["stretch"] is not None:
        stretch = FX._pair(s["stretch"], f"{what} stretch")
        if not 0.0 <= stretch[0] < stretch[1] <= 100.0:
            raise FX.FxError(f"{what}: stretch = {s['stretch']!r}: [low, high] percentiles of the luma, 0 <= low < "
                             f"high <= 100")
    out.update(clip=s["clip"], loop=s["loop"],
               **{"in": FX._num(s["in"], f"{what} in", lo=0.0), "rate": FX._num(s["rate"], f"{what} rate", 0.0, 16.0)},
               view=TB._view(FX, s["view"], what),
               lean=None if s["lean"] is None else FX._num(s["lean"], f"{what} lean", -1.0, 1.0),
               at=None if s["at"] is None else FX._num(s["at"], f"{what} at", lo=0.0), stretch=stretch,
               keep=s["keep"], melt=[_melt(FX, m, f"{what} melt[{i}]") for i, m in enumerate(_list(s["melt"]))])
    return out


def smear(FX, v, what, table=False):
    """A `smear`: {dur, block, gain, bleed, flow, jump, from, ease, patch, seed} over SMEAR (a table's needs `from`).
    `jump` ([dx, dy] frame heights, x right, y down: where she went at the zap) drags the held picture that far that way
    over `dur` (a made-up drift); `flow` is any made-up motion; neither: the new clip's own."""
    s = _table(FX, v, SMEAR, what, "smear")
    if s["from"] is not None and (not isinstance(s["from"], str) or not s["from"].strip()):
        raise FX.FxError(f"{what}: from = {s['from']!r}: the `name` of a sky or a table (a sheet with a `clip`)")
    if table and s["from"] is None:
        raise FX.FxError(f"{what}: a table smears in from a sky's or another table's picture: give its `from` (their "
                         f"`name`)")
    if s["ease"] not in SMEAR_EASES:
        raise FX.FxError(f"{what}: ease {s['ease']!r} is none of {', '.join(SMEAR_EASES)}")
    if s["flow"] is not None and s["jump"] is not None:
        raise FX.FxError(f"{what}: `jump` makes the smear's motion (a drift her way): give it or a `flow`, not both")
    return {"dur": FX._num(s["dur"], f"{what} dur", 1e-3, 30.0), "block": FX._num(s["block"], f"{what} block", 2.0, 256.0),
            "gain": FX._num(s["gain"], f"{what} gain", 0.0, 64.0), "bleed": FX._num(s["bleed"], f"{what} bleed", 0.0, 1.0),
            "flow": FX._flow(s["flow"], what), "from": s["from"], "ease": s["ease"],
            "jump": None if s["jump"] is None else FX._pair(s["jump"], f"{what} jump"),
            "patch": FX._num(s["patch"], f"{what} patch", 0.0, 1.0),
            "seed": FX._num(s["seed"], f"{what} seed", lo=0, integer=True)}


def _melt(FX, v, what):
    m = _table(FX, v, {**MELT, "from": None, "to": None}, what, "melt")
    for k in ("from", "to"):
        if m[k] is None:
            raise FX.FxError(f"{what}: `{k}` (clip seconds) is required")
    a, b = FX._num(m["from"], f"{what} from"), FX._num(m["to"], f"{what} to")
    if b <= a:
        raise FX.FxError(f"{what}: `to` ({b:g}) must come after `from` ({a:g})")
    if m["dir"] not in MELT_DIRS:
        raise FX.FxError(f"{what}: dir {m['dir']!r} is none of {', '.join(MELT_DIRS)}")
    if m["run"] not in MELT_RUNS:
        raise FX.FxError(f"{what}: run {m['run']!r} is none of {', '.join(MELT_RUNS)}")
    return {"from": a, "to": b, "dir": m["dir"], "run": m["run"],
            "length": FX._num(m["length"], f"{what} length", 0.0, 4.0),
            "level": FX._num(m["level"], f"{what} level", 0.0, 1.0),
            "area": None if m["area"] is None else _box(FX, m["area"], f"{what} area"),
            "feather": FX._num(m["feather"], f"{what} feather", 0.0, 1.0),
            "strength": FX._keys(m["strength"], f"{what} strength"),
            "seed": FX._num(m["seed"], f"{what} seed", lo=0, integer=True)}


# ================================================================================================= timing
def segment_at(fx, t):
    """The index of the segment on screen at clip second `t` (the first before its own start)."""
    i = 0
    for j, s in enumerate(fx["clips"]):
        if s["t"] <= t + 1e-9:
            i = j
    return i


def smear_share(sm, t, t0):
    """How much of the new picture has broken through a smear that starts at clip second `t0`, at clip second `t`: 0 on
    its first frame, eased (`ease`) to 1 at `dur`; None outside its window."""
    u = (t - t0) / sm["dur"]
    if u < -1e-9 or u >= 1.0:
        return None
    u = TW.clamp01(u)
    return u if sm["ease"] == "linear" else float(TW.ease(u, sm["ease"]))


def clip_second(layer, t0, t):
    """The second of a layer's clip shown at clip second `t` of a segment that starts at `t0` (before mkmmd.footage's
    loop or hold at its end)."""
    return layer["in"] + layer["rate"] * (t - t0)


def keyed(keys, t):
    """`[[t, v], ...]` keys at clip second `t`: linear between them, held outside."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def melts_at(fx, t):
    """The melts going on at clip second `t`, each with its index and its strength there: [(index, melt, strength)]
    (strength above 0)."""
    out = []
    for k, m in enumerate(fx.get("melt") or ()):
        if m["from"] <= t < m["to"]:
            s = keyed(m["strength"], t)
            if s > 0.0:
                out.append((k, m, s))
    return out


def named(fxs, name):
    """The entry of `fxs` whose `name` is `name`, a sky or a table (a sheet with a `clip`), or None."""
    for e in fxs:
        if e.get("name") == name and (e["kind"] in KINDS or (e["kind"] == "sheet" and e.get("clip") is not None)):
            return e
    return None


def clips(spec):
    """The clips a raw `[[fx]]` sky `spec` shows (its segments' and their layers', its `aspect.<output>` tables' too),
    each once: what the credits name."""
    out = []
    for e in [spec, *(v for v in (spec.get("aspect") or {}).values() if isinstance(v, dict))]:
        for s in e.get("clips") or ():
            if not isinstance(s, dict):
                continue
            for x in [s, *(y for y in (s.get("layers") or ()) if isinstance(y, dict))]:
                c = x.get("clip")
                if isinstance(c, str) and c.strip() and c not in out:
                    out.append(c)
    return out
