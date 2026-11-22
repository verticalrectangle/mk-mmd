"""Sticker sheets, bpy-free (docs/design.md: Frame effects, Sticker sheets): the `[[fx]]` kinds `sheet` and `fling`
checked, timed and laid out; mkmmd/sheet.py draws them.

    sheet   a sheet of paper seen from above, a camera hung over the desk drifting slowly over it (`paper`, `drift`). On
            it a live figure sticker (the figure of the cut's frame, or of a plate `shot`, set in by `at` and `height` and
            moved by `keys`, as a collage's) in a white die-cut rim with a soft shadow, slapped down (`slap`: it lands big,
            squashes, bounces and settles) and peeled off (`peel`: its corner lifts, it rolls back over itself and leaves
            the frame, its hollow outline left where it was); `hollows`, the outlines of stickers peeled off before (the
            figure's rim frozen at a clip second, nothing but the paper inside, a sheen where the glue was, a dashed cut
            line), or of one that is still there and is peeled off in the span (a hollow's own `peel`: a blank die cut
            of that shape in a flat colour lies in it, fades, and is rolled off as the figure's sticker is); and
            `doodles`, stickers of the pen-drawn kits scattered clear of the figure and of given boxes, jumping
            out from under the slap, slapped down one by one on their times, hopping on the hits and peeled off one by
            one. With a `clip` the sheet is a table: the clip under everything instead of paper, seen through a `view`
            of its own, a cup of coffee with pads, rings and drops on it, punched through into the next shot
            (mkmmd.core.table)
    fling   stickers thrown from a point (a scene point: a guitar's body) at the lens on the hits, growing and turning as
            they come, their white backs showing as they flip, and leaving by the frame's edges; or (`away`) thrown
            away from the lens: coming in large past the frame's edges and falling away into the point, shrinking
    thread  a thread tied to a sheet's live sticker, streaming out past the frame's edge (mkmmd.core.thread)

Times are clip seconds, places frame fractions (0, 0 the top left), sizes frame heights. Every motion is a function of the
clip second and the seed (the slap's spring, a jump's arc, a hop, a peel's pull: mkmmd.core.sheet3d.hand, as the peel
transition's), so any frame comes out the same in any order."""
import math

import numpy as np

from . import peel as PE
from . import sheet3d as S3
from . import table as TB
from . import thread as TH

KINDS = ("sheet", "fling") + TH.KINDS          # a thread is tied to a sheet's live sticker (mkmmd.core.thread)
DOODLE_SHAPES = ("star", "heart", "pick", "note", "sparkle", "cloud", "strawberry", "moon", "rabbit", "carrot")
POPS = ("love", "gold", "rose", "pine", "foam", "iris")          # the palette's pops, a doodle's colour in turn
EDGE = {"color": "#ffffff", "width": 16.0}                      # a sticker's die-cut rim, px at 1080 on the short side
SHADOW = {"color": "text", "offset": [0.004, 0.007], "blur": 6.0, "alpha": 0.3}
PAPER = {"color": "base", "shade": "overlay", "light": "surface", "mottle": 1.0, "fibres": 1.0, "tooth": 1.0,
         "grid": None, "spacing": 0.04, "ink": "text", "scale": 1.0, "lamp": 0.025}
GRIDS = ("dots", "lines", "squares")
DRIFT = {"amount": 0.006, "speed": 0.05, "turn": 0.3}           # frame heights, cycles a second, degrees
SLAP = {"size": 1.3, "drop": 0.08, "squash": 0.15, "settle": 0.25, "dust": 80}
STRETCH = 0.35         # the stretch after a squash, a share of the spring's swing: she never goes thin
PEEL = {"dur": 0.8, "corner": "top-right", "radius": 0.05, "back": "surface"}
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
HOLLOW = {"at": None, "height": None, "turn": 0.0, "mirror": False, "sheen": 0.35, "dash": True, "peel": None}
BLANK = {**PEEL, "color": "muted", "fade": None}   # a hollow's own peel: the blank sticker that lies in it until then
BLANK_FADE = 0.8       # how far a blank sticker's colour has gone toward the paper's light by the end of its `fade`
DOODLES = {"count": 10, "shapes": list(DOODLE_SHAPES), "colors": None, "size": 0.07, "vary": 0.25, "tilt": 25.0,
           "area": [0.03, 0.03, 0.97, 0.97], "clear": [], "gap": 0.015, "hop": None, "jump": None, "peel": None,
           "slap": None, "seed": 1}
DSLAP = {"size": 1.3, "drop": 0.08, "squash": 0.15, "settle": 0.25}     # a doodle slapped down: the figure's slap's
HOP = {"height": 0.22, "dur": 0.2, "ripple": 0.03, "min_db": None, "every": 1}
JUMP = {"from": "figure", "dur": 0.5, "stagger": 0.06, "height": 0.8, "spin": 120.0}
DPEEL = {"dur": 0.35, "radius": 0.1, "order": "shuffle"}
DPEEL_ORDERS = ("shuffle", "given", "near", "far")
TAPES = {"count": 6, "colors": None, "patterns": ["stripes", "dots", "gingham"], "width": 0.035, "length": 0.17,
         "alpha": 0.82, "seed": 1}
PATTERNS = ("stripes", "dots", "gingham")
SCRIBBLES = {"count": 8, "shapes": ["heart", "star", "spiral", "underline", "arrow", "rabbit"], "inks": ["text", "love"],
             "size": 0.05, "width": 3.5, "fps": 8.0, "wobble": 1.2, "seed": 1}
SCRIBBLE_SHAPES = ("heart", "star", "spiral", "underline", "arrow", "rabbit")
FLING = {"count": 6, "shapes": list(DOODLE_SHAPES), "colors": None, "size": 0.035, "grow": 12.0, "life": 0.55,
         "spread": 0.12, "spin": 160.0, "flip": 0.5, "aim": None, "aimed": 2, "away": False,
         "edge": {"color": "#ffffff", "width": 12.0}, "shadow": {"alpha": 0.25}, "seed": 1, "min_db": None,
         "every": 1}
KEYS = {"sheet": {"objects", "shot", "at", "height", "pivot", "keys", "edge", "shadow", "paper", "drift", "slap", "peel",
                  "hollows", "doodles", "tapes", "scribbles", "seed"} | set(TB.KEYS),
        "fling": {"center", "times", "lane", "min_db", "every", "count", "shapes", "colors", "size", "grow", "life",
                  "spread", "spin", "flip", "aim", "aimed", "away", "edge", "shadow", "seed"}, **TH.KEYS}
DEFAULTS = {"sheet": {"objects": None, "shot": None, "at": None, "height": None, "pivot": "base", "keys": None,
                      "edge": {}, "shadow": {}, "paper": {}, "drift": {}, "slap": None, "peel": None, "hollows": None,
                      "doodles": None, "tapes": None, "scribbles": None, "seed": 1, **{k: None for k in TB.KEYS}},
            "fling": {k: v for k, v in FLING.items()}, **TH.DEFAULTS}
REACH = 0.62           # a doodle's reach about its middle, a share of its longer side (a shape turned a little)
TRIES = 96             # candidates a doodle's place is picked from (the one farthest from the rest)
JUMP_LAND = 0.78       # the share of a jump that its arc takes; the rest is the little hop it lands with
JUMP_BOUNCE = 0.16     # that hop's height, a share of the arc's
LAND = {"squash": 0.16, "settle": 0.16}     # the squash a doodle lands with off a jump or a hop
FLING_UP = 0.55        # how far a thrown sticker's heading leans up the frame (a toss up past her, not a spray)
FLING_POP = 0.05       # seconds a thrown sticker takes to pop out to its size
FLING_STAGGER = 0.06   # seconds over which one hit's stickers leave
FLING_FLIP = 0.07      # seconds a thrown sticker that flips takes to turn over once (its back shows half of that)
FLING_AIM = 0.8        # the share of its life at which an aimed sticker passes the `aim` point
HOP_WOBBLE = 0.35      # seconds a doodle's wobble takes to settle after a hop lands


def beat_times(beats, downbeats, every):
    """The beats a hop with `beats = true` lands on: every beat (`every` 1), or every `every`-th counted from the first
    downbeat (4: the downbeats), from the timeline's `beats` and `downbeats` (clip seconds)."""
    beats = sorted(float(b) for b in beats)
    if every <= 1 or not downbeats:
        return beats[::max(int(every), 1)]
    d0 = min(float(d) for d in downbeats)
    i0 = min(range(len(beats)), key=lambda i: abs(beats[i] - d0)) if beats else 0
    return beats[i0 % every::every]


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """A sheet's, a fling's or a thread's own keys checked and filled in, in place (`out` has its kind's defaults under
    `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    if out["kind"] in TH.KINDS:
        return TH.normalize(spec, out, palette, what)
    (_sheet if out["kind"] == "sheet" else _fling)(FX, spec, out, palette, what)


def _table(FX, v, defaults, what, name):
    """A sub-table (`{...}`) over its defaults; `false` (or None where the default is None) for none."""
    if v is False or v is None:
        return None
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table ({{{', '.join(sorted(defaults))}}}) or false")
    extra = sorted(set(v) - set(defaults))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {sorted(defaults)})")
    return {**defaults, **v}


def _list(v):
    return [] if v is None else [v] if isinstance(v, dict) else list(v)


def _edge(FX, v, palette, what, default=EDGE):
    e = _table(FX, v, default, what, "edge")
    if e is None:
        return None
    return {"color": FX._colour(e["color"], palette, f"{what} edge"),
            "width": FX._num(e["width"], f"{what} edge width", lo=0.0)}


def _shadow(FX, v, palette, what):
    s = _table(FX, v, SHADOW, what, "shadow")
    return None if s is None else FX._shadow(s, palette, what)


def _box(FX, v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        raise FX.FxError(f"{what} = {v!r}: expected [x0, y0, x1, y1] (frame fractions, 0, 0 the top left)")
    b = [FX._num(x, what) for x in v]
    if b[2] <= b[0] or b[3] <= b[1]:
        raise FX.FxError(f"{what} = {b}: x1 must be right of x0 and y1 below y0")
    return b


def _shapes(FX, v, what):
    from . import mandala as MD
    names = [v] if isinstance(v, str) else v
    if not isinstance(names, (list, tuple)) or not names or any(n not in MD.SHAPES for n in names):
        raise FX.FxError(f"{what}: shapes = {v!r}: one of {', '.join(MD.SHAPES)}, or a list of them")
    return list(names)


def _own(FX, names, palette, what):
    """{shape: (accent, its second colour, its own colours)}: a drawn sticker's own accent, what stands in for it on a
    sticker of that very colour, and the colours it comes in when the entry gives none (mkmmd.core.mandala)."""
    from . import mandala as MD
    return {n: (FX._colour(MD.SHAPE_ACCENT[n], palette, what), FX._colour(MD.SHAPE_COLORS[n][1], palette, what),
                [FX._colour(c, palette, what) for c in MD.SHAPE_COLORS[n]]) for n in names}


def colours(entry, shape):
    """The colours a doodle or a thrown sticker of `shape` is dealt from: the entry's `colors`, else the shape's own."""
    return entry["colors"] if entry["colors"] is not None else entry["own"][shape][2]


def _times(FX, v, what):
    t = [v] if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    if not isinstance(t, (list, tuple)) or not t:
        raise FX.FxError(f"{what}: expected clip seconds, one or a list")
    return sorted(FX._num(x, what) for x in t)


def _sheet(FX, spec, out, palette, what):
    TB.normalize(FX, spec, out, palette, what)          # first: a table (a `clip`) has no paper, hollows, peel, tapes
    table = TB.table_of(spec)
    objs = out["objects"]
    if objs is not None:
        objs = [objs] if isinstance(objs, str) else objs
        if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
            raise FX.FxError(f"{what}: objects = {spec['objects']!r}: expected object patterns (`name*`, `@collection`, "
                             f"`prop:key`), a list or one")
        out["objects"] = list(objs)
    else:
        lone = sorted(k for k in ("shot", "at", "height", "keys", "slap", "peel", "hollows", "track", "lift")
                      if spec.get(k) is not None)
        if lone:
            raise FX.FxError(f"{what}: {', '.join(lone)} belong to the figure sticker: give its `objects`")
    sh = out["shot"]
    if sh is not None and (not isinstance(sh, str) or not sh.strip()):
        raise FX.FxError(f"{what}: shot = {sh!r}: expected the name of a [[shot]] (a plate) the figure is taken from")
    out["at"] = None if out["at"] is None else FX._pair(out["at"], f"{what} at")
    out["height"] = None if out["height"] is None else FX._num(out["height"], f"{what} height", 0.01, 10.0)
    pv = out["pivot"]
    if isinstance(pv, str) and pv not in FX.PIVOTS:
        raise FX.FxError(f"{what}: pivot {pv!r}: \"base\", \"center\" or [x, y] (frame fractions)")
    out["pivot"] = pv if isinstance(pv, str) else FX._pair(pv, f"{what} pivot")
    out["keys"] = FX._moves(out["keys"], what)
    out["edge"] = _edge(FX, out["edge"], palette, what)
    out["shadow"] = _shadow(FX, out["shadow"], palette, what)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["paper"] = None if table else _paper(FX, out["paper"], palette, what)
    d = out["drift"] if not table or spec.get("drift") is not None else None    # a table's camera holds still unless
                                                                                  # it is given a drift
    if isinstance(d, (int, float)) and not isinstance(d, bool):
        d = {"amount": d}
    d = _table(FX, d, DRIFT, what, "drift")
    out["drift"] = None if d is None else {k: FX._num(d[k], f"{what} drift {k}", lo=0.0) for k in DRIFT}
    out["slap"] = [_slap(FX, s, what) for s in _list(out["slap"])]
    if not table:
        p = _table(FX, out["peel"], {**PEEL, "t": None}, what, "peel")
        if p is not None:
            if p["t"] is None:
                raise FX.FxError(f"{what}: peel needs `t`, the clip second the corner starts to lift")
            out["peel"] = _peel(FX, p, palette, f"{what} peel")
        out["hollows"] = [_hollow(FX, h, palette, what) for h in _list(out["hollows"])]
        out["tapes"] = _tapes(FX, out["tapes"], palette, what)
        out["scribbles"] = _scribbles(FX, out["scribbles"], palette, what)
    out["doodles"] = _doodles(FX, out["doodles"], palette, spec, what)
    out["drops"] = TB.drops(FX, spec.get("drops"), palette, spec, what, _doodles)


def _tapes(FX, v, palette, what):
    t = _table(FX, v, TAPES, what, "tapes")
    if t is None:
        return None
    w = f"{what} tapes"
    pats = [t["patterns"]] if isinstance(t["patterns"], str) else t["patterns"]
    if not isinstance(pats, (list, tuple)) or not pats or any(p not in PATTERNS for p in pats):
        raise FX.FxError(f"{w}: patterns = {t['patterns']!r}: {', '.join(PATTERNS)}, one or a list")
    return {"count": FX._num(t["count"], f"{w} count", 0, 24, integer=True),
            "colors": FX._colours(list(POPS) if t["colors"] is None else t["colors"], palette, f"{w} colors"),
            "patterns": list(pats), "width": FX._num(t["width"], f"{w} width", 0.005, 0.3),
            "length": FX._num(t["length"], f"{w} length", 0.02, 2.0),
            "alpha": FX._num(t["alpha"], f"{w} alpha", 0.0, 1.0), "seed": FX._num(t["seed"], f"{w} seed", lo=0,
                                                                                    integer=True)}


def _scribbles(FX, v, palette, what):
    s = _table(FX, v, SCRIBBLES, what, "scribbles")
    if s is None:
        return None
    w = f"{what} scribbles"
    shapes = [s["shapes"]] if isinstance(s["shapes"], str) else s["shapes"]
    if not isinstance(shapes, (list, tuple)) or not shapes or any(x not in SCRIBBLE_SHAPES for x in shapes):
        raise FX.FxError(f"{w}: shapes = {s['shapes']!r}: {', '.join(SCRIBBLE_SHAPES)}, one or a list")
    return {"count": FX._num(s["count"], f"{w} count", 0, 64, integer=True), "shapes": list(shapes),
            "inks": FX._colours(s["inks"], palette, f"{w} inks"), "size": FX._num(s["size"], f"{w} size", 0.005, 1.0),
            "width": FX._num(s["width"], f"{w} width", 0.5, 40.0), "fps": FX._num(s["fps"], f"{w} fps", 0.0, 60.0),
            "wobble": FX._num(s["wobble"], f"{w} wobble", 0.0, 20.0),
            "seed": FX._num(s["seed"], f"{w} seed", lo=0, integer=True)}


def _paper(FX, v, palette, what):
    p = _table(FX, v, PAPER, what, "paper")
    if p is None:
        return None
    out = {k: FX._colour(p[k], palette, f"{what} paper {k}") for k in ("color", "shade", "light", "ink")}
    for k in ("mottle", "fibres", "tooth"):
        out[k] = FX._num(p[k], f"{what} paper {k}", 0.0, 4.0)
    out["lamp"] = FX._num(p["lamp"], f"{what} paper lamp", 0.0, 0.3)
    out["scale"] = FX._num(p["scale"], f"{what} paper scale", 0.1, 10.0)
    out["spacing"] = FX._num(p["spacing"], f"{what} paper spacing", 0.005, 0.5)
    if p["grid"] not in (None, False) and p["grid"] not in GRIDS:
        raise FX.FxError(f"{what}: paper grid {p['grid']!r} is none of {', '.join(GRIDS)} (or false)")
    out["grid"] = p["grid"] or None
    return out


def _slap(FX, v, what):
    s = _table(FX, v, {**SLAP, "t": None}, what, "slap")
    if s is None or s["t"] is None:
        raise FX.FxError(f"{what}: a slap is {{t, size, drop, squash, settle, dust}}, `t` the clip second it lands")
    return {"t": FX._num(s["t"], f"{what} slap t"), "size": FX._num(s["size"], f"{what} slap size", 1.0, 4.0),
            "drop": FX._num(s["drop"], f"{what} slap drop", 0.0, 2.0),
            "squash": FX._num(s["squash"], f"{what} slap squash", 0.0, 0.6),
            "settle": FX._num(s["settle"], f"{what} slap settle", 0.05, 2.0),
            "dust": FX._num(s["dust"], f"{what} slap dust", 0, 400, integer=True)}


def _peel(FX, p, palette, what):
    if p["corner"] not in CORNERS:
        raise FX.FxError(f"{what}: corner {p['corner']!r} is none of {', '.join(CORNERS)}")
    out = {"t": FX._num(p["t"], f"{what} t"), "dur": FX._num(p["dur"], f"{what} dur", 0.05, 10.0),
           "corner": p["corner"], "radius": FX._num(p["radius"], f"{what} radius", 0.005, 0.5)}
    out["back"] = FX._colour(p["back"], palette, f"{what} back")
    return out


def _hollow(FX, v, palette, what):
    h = _table(FX, v, {**HOLLOW, "t": None}, what, "hollows")
    if h is None or h["t"] is None:
        raise FX.FxError(f"{what}: a hollow is {{t, at, height, turn, mirror, sheen, dash, peel}}, `t` the clip second "
                         f"whose figure it is the outline of")
    if not isinstance(h["mirror"], bool) or not isinstance(h["dash"], bool):
        raise FX.FxError(f"{what}: a hollow's mirror and dash are true or false")
    return {"t": FX._num(h["t"], f"{what} hollow t", lo=0.0),
            "at": None if h["at"] is None else FX._pair(h["at"], f"{what} hollow at"),
            "height": None if h["height"] is None else FX._num(h["height"], f"{what} hollow height", 0.01, 10.0),
            "turn": FX._num(h["turn"], f"{what} hollow turn"), "mirror": h["mirror"],
            "sheen": FX._num(h["sheen"], f"{what} hollow sheen", 0.0, 1.0), "dash": h["dash"],
            "peel": _blank(FX, h["peel"], palette, f"{what} hollow peel")}


def _blank(FX, v, palette, what):
    """A hollow's own `peel`: None (the sticker was peeled off before), or the blank sticker that still lies in it:
    {t, dur, corner, radius, back} as the figure's peel, its flat `color` and `fade` [from, to], the clip seconds over
    which that colour goes BLANK_FADE of the way to the paper's."""
    p = _table(FX, v, {**BLANK, "t": None}, what, "peel")
    if p is None:
        return None
    if p["t"] is None:
        raise FX.FxError(f"{what}: needs `t`, the clip second the corner starts to lift")
    out = _peel(FX, p, palette, what)
    out["color"] = FX._colour(p["color"], palette, f"{what} color")
    out["fade"] = None
    if p["fade"] is not None:
        a, b = FX._pair(p["fade"], f"{what} fade")
        if b <= a:
            raise FX.FxError(f"{what}: fade = {p['fade']!r}: [from, to] clip seconds, `to` after `from`")
        out["fade"] = [a, b]
    return out


def _doodles(FX, v, palette, spec, what, least=0.005):
    d = _table(FX, v, DOODLES, what, "doodles")
    if d is None:
        return None
    w = f"{what} doodles"
    out = {"count": FX._num(d["count"], f"{w} count", 0, 64, integer=True),
           "shapes": _shapes(FX, d["shapes"], w),
           "colors": None if d["colors"] is None else FX._colours(d["colors"], palette, f"{w} colors"),
           "size": FX._num(d["size"], f"{w} size", least, 1.0), "vary": FX._num(d["vary"], f"{w} vary", 0.0, 0.9),
           "tilt": FX._num(d["tilt"], f"{w} tilt", 0.0, 180.0), "area": _box(FX, d["area"], f"{w} area"),
           "clear": [_box(FX, b, f"{w} clear") for b in (d["clear"] or [])],
           "gap": FX._num(d["gap"], f"{w} gap", lo=0.0), "seed": FX._num(d["seed"], f"{w} seed", lo=0, integer=True)}
    out["own"] = _own(FX, out["shapes"], palette, w)
    out["ink"] = FX._colour("text", palette, w)
    hop = _table(FX, d["hop"], {**HOP, "times": None, "lane": None, "beats": None}, w, "hop")
    if hop is not None:
        h = {k: FX._num(hop[k], f"{w} hop {k}", lo=0.0) for k in ("height", "dur", "ripple")}
        if h["dur"] <= 0:
            raise FX.FxError(f"{w} hop: dur must be positive")
        if hop["beats"] is not None:
            if hop["beats"] is not True or hop["times"] is not None or hop["lane"] is not None:
                raise FX.FxError(f"{w} hop: beats = true (the timeline's beats) or `times` or a `lane`, one of them")
            h.update(beats=True, every=FX._num(hop["every"], f"{w} hop every", 1, 64, integer=True), times=None)
        else:
            given = {k: x for k, x in hop.items() if k in ("times", "lane", "min_db", "every") and x is not None
                     and not (k == "every" and x == 1)}
            h.update(min_db=None, every=1)                  # a lane's picks when the hop leaves them out
            FX._hits(given, h, f"{w} hop")
            h["beats"] = False
        out["hop"] = h
    else:
        out["hop"] = None
    j = _table(FX, d["jump"], {**JUMP, "t": None}, w, "jump")
    if j is not None:
        if j["t"] is None:
            raise FX.FxError(f"{w}: jump needs `t`, the clip second they leave from under the figure")
        src = j["from"]
        if src != "figure":
            src = FX._pair(src, f"{w} jump from")
        elif spec.get("objects") is None:
            raise FX.FxError(f"{w}: jump from \"figure\" needs the sheet's `objects`: give `from = [x, y]`")
        out["jump"] = {"t": FX._num(j["t"], f"{w} jump t"), "from": src,
                       "dur": FX._num(j["dur"], f"{w} jump dur", 0.05, 5.0),
                       "stagger": FX._num(j["stagger"], f"{w} jump stagger", 0.0, 5.0),
                       "height": FX._num(j["height"], f"{w} jump height", 0.0, 4.0),
                       "spin": FX._num(j["spin"], f"{w} jump spin")}
    else:
        out["jump"] = None
    p = _table(FX, d["peel"], {**DPEEL, "times": None, "corner": None}, w, "peel")
    if p is not None:
        if p["times"] is None:
            raise FX.FxError(f"{w}: peel needs `times`, the clip seconds a doodle starts to come off, one each")
        if p["order"] not in DPEEL_ORDERS:
            raise FX.FxError(f"{w} peel: order {p['order']!r} is none of {', '.join(DPEEL_ORDERS)}")
        if p["corner"] is not None and p["corner"] not in CORNERS:
            raise FX.FxError(f"{w} peel: corner {p['corner']!r} is none of {', '.join(CORNERS)} (default: seeded)")
        out["peel"] = {"times": _times(FX, p["times"], f"{w} peel times"),
                       "dur": FX._num(p["dur"], f"{w} peel dur", 0.05, 10.0),
                       "radius": FX._num(p["radius"], f"{w} peel radius", 0.01, 1.0), "order": p["order"],
                       "corner": p["corner"]}
        if len(out["peel"]["times"]) > out["count"]:
            raise FX.FxError(f"{w} peel: {len(out['peel']['times'])} times for {out['count']} doodles")
    else:
        out["peel"] = None
    s = _table(FX, d["slap"], {**DSLAP, "times": None}, w, "slap")
    out["slap"] = None
    if s is not None:
        if s["times"] is None:
            raise FX.FxError(f"{w}: slap needs `times`, the clip seconds a doodle lands on, one each")
        out["slap"] = {"times": _times(FX, s["times"], f"{w} slap times"),
                       "size": FX._num(s["size"], f"{w} slap size", 1.0, 4.0),
                       "drop": FX._num(s["drop"], f"{w} slap drop", 0.0, 2.0),
                       "squash": FX._num(s["squash"], f"{w} slap squash", 0.0, 0.6),
                       "settle": FX._num(s["settle"], f"{w} slap settle", 0.05, 2.0)}
        if len(out["slap"]["times"]) > out["count"]:
            raise FX.FxError(f"{w} slap: {len(out['slap']['times'])} times for {out['count']} doodles")
    return out


def _fling(FX, spec, out, palette, what):
    if "center" not in spec:
        raise FX.FxError(f"{what}: `center` is required: where they are thrown from, [x, y] or {{point = ...}}")
    out["center"] = FX._center(out["center"], f"{what} center")
    FX._hits(spec, out, what)
    out["count"] = FX._num(out["count"], f"{what} count", 1, 32, integer=True)
    out["shapes"] = _shapes(FX, out["shapes"], what)
    out["colors"] = None if out["colors"] is None else FX._colours(out["colors"], palette, f"{what} colors")
    out["own"] = _own(FX, out["shapes"], palette, what)
    out["ink"] = FX._colour("text", palette, what)
    out["size"] = FX._num(out["size"], f"{what} size", 0.002, 1.0)
    out["grow"] = FX._num(out["grow"], f"{what} grow", 1.0, 100.0)
    out["life"] = FX._num(out["life"], f"{what} life", 0.05, 10.0)
    out["spread"] = FX._num(out["spread"], f"{what} spread", 0.0, 4.0)
    out["spin"] = FX._num(out["spin"], f"{what} spin")
    out["flip"] = FX._num(out["flip"], f"{what} flip", 0.0, 1.0)
    out["aim"] = None if out["aim"] is None else FX._pair(out["aim"], f"{what} aim")
    out["aimed"] = FX._num(out["aimed"], f"{what} aimed", 0, 32, integer=True)
    if not isinstance(out["away"], bool):
        raise FX.FxError(f"{what}: away = {out['away']!r}: true (thrown away from the lens, into the point) or false")
    out["edge"] = _edge(FX, out["edge"], palette, what, FLING["edge"])
    out["shadow"] = _shadow(FX, out["shadow"], palette, what)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


# ================================================================================================= what it reads
def uses_figure(fx):
    """Whether a sheet reads its figure's layer (it has a live sticker)."""
    return fx["kind"] == "sheet" and fx.get("objects") is not None


def frozen(fx, fps, frame0):
    """The Blender frames whose figure a sheet freezes: each hollow's, and the peel's (the outline it leaves)."""
    out = [int(round(frame0 + h["t"] * fps)) for h in fx["hollows"]]
    if fx["peel"] is not None:
        out.append(int(round(frame0 + fx["peel"]["t"] * fps)))
    return out


def sources(fx, frame, frame0):
    """The other frames a sheet reads at `frame`: the span's first (its figure's box there places it, its doodles keep
    clear of it) and the frames its hollows and its peel freeze."""
    if not uses_figure(fx):
        return []
    out = [] if frame == fx["f0"] else [fx["f0"]]
    return out + [g for g in frozen(fx, fx["frame_rate"], frame0) if g != frame and g not in out]


# ================================================================================================= the sheet's motion
def drift_at(drift, t, seed=1):
    """The camera's drift over the desk at clip second `t`: (dx, dy) frame heights and a turn in degrees; three slow
    sines of unrelated periods, so it never repeats within a video and never stops. (0, 0, 0) without one."""
    if drift is None or drift["amount"] <= 0 and drift["turn"] <= 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng([int(seed), 5])
    p = rng.uniform(0.0, 2.0 * math.pi, 3)
    w = 2.0 * math.pi * drift["speed"]
    return (drift["amount"] * math.sin(w * t + p[0]), 0.8 * drift["amount"] * math.sin(1.37 * w * t + p[1]),
            drift["turn"] * math.sin(0.83 * w * t + p[2]))


def spring(u, squash, settle):
    """A landing's squash at `u` seconds after touching down: 0 at touch, `squash` at its deepest a few hundredths of a
    second later, a stretch past rest (negative) and back, nothing left by `settle` seconds."""
    if u <= 0.0 or u >= settle or squash <= 0.0:
        return 0.0
    w = 2.0 * math.pi / (0.52 * settle)
    tau = settle / 4.5
    peak = math.atan(w * tau) / w
    norm = math.exp(-peak / tau) * math.sin(w * peak)
    fade = 1.0 - _smooth(u / settle, 0.8)
    return squash * math.exp(-u / tau) * math.sin(w * u) / norm * fade


def _smooth(v, a):
    s = min(max((v - a) / (1.0 - a), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def slap_at(slaps, t):
    """The figure sticker's slap at clip second `t`: {scale, sx, sy, lift}. Before a slap's `drop` it is held `size`
    times as large, lifted (lift 1); it falls over `drop` seconds (an ease-in) and lands on `t`, then squashes by
    `squash` of its height (sy, and sx = 1 / sy: its area kept), stretches back past rest and settles by `settle`.
    The latest slap that has begun counts; at rest without one."""
    rest = {"scale": 1.0, "sx": 1.0, "sy": 1.0, "lift": 0.0}
    if not slaps:
        return rest
    cur = slaps[0]
    for s in slaps:
        if t >= s["t"] - s["drop"]:
            cur = s
    u = t - cur["t"]
    if u < 0.0:
        e = 0.0 if cur["drop"] <= 0 or u < -cur["drop"] else ((u + cur["drop"]) / cur["drop"]) ** 2
        return {"scale": cur["size"] + (1.0 - cur["size"]) * e, "sx": 1.0, "sy": 1.0, "lift": 1.0 - e}
    q = spring(u, cur["squash"], cur["settle"])
    q = q if q >= 0.0 else STRETCH * q
    sy = 1.0 - q
    return {"scale": 1.0 + 0.15 * max(-q, 0.0), "sx": 1.0 / sy, "sy": sy, "lift": 0.0}


def peel_progress(peel, t):
    """How far through its time a peel is at clip second `t`: None before it starts, then 0..1 (the fold travels along
    a hand's pull over it, mkmmd.core.sheet3d.hand: a slow pinch, then a pull that speeds up into a flick), 1 after."""
    if peel is None or t < peel["t"]:
        return None
    return min((t - peel["t"]) / peel["dur"], 1.0)


def peeled(peel, t):
    """Whether the peel is over at `t`: the sticker has left."""
    return peel is not None and t >= peel["t"] + peel["dur"]


def peel_roll(n, c0, lo, R, u):
    """(fold line px along `n`, tail angle radians) of a peel `u` of its time through, as the peel transition rolls its
    sheet (mkmmd.core.peel): the fold runs along a hand's pull (mkmmd.core.sheet3d.hand) from `c0` (peel_start: a pixel
    past the sticker's grip, nothing lifted) to where the whole sticker, lifted off at the last angle, lies beyond `lo`
    (px along n: the frame's far side for the figure, the sticker's own for a doodle); the tail swings up as it lifts,
    lies back over the sticker and is flicked up toward the lens."""
    last = math.radians(PE.PHI[2])
    c1 = lo - (R * math.sin(last) - last * R * math.cos(last) + 2.0) / (1.0 - math.cos(last))
    a, b, c = PE.PHI
    if u < PE.FLICK:
        s = min(u / PE.PHI_SWING, 1.0)
        f = a + (b - a) * s * s * (3.0 - 2.0 * s)
    else:
        s = (u - PE.FLICK) / (1.0 - PE.FLICK)
        f = b + (c - b) * s * s * (3.0 - 2.0 * s)
    return c0 + (c1 - c0) * S3.hand(u), math.radians(f)


def grip_of(cov, corner):
    """(x, y) px: the point of a sticker's coverage `cov` (its rim in) farthest toward its pinched `corner` along the
    fold's normal. A figure's box has empty paper in its corners: the peel starts here, where the sticker is, so the
    sticker is seen to lift from the peel's `t`, and a hand takes it here. None for an empty coverage."""
    n = PE.normal({"from": corner, "angle": 0.0})
    ys, xs = np.nonzero(np.asarray(cov) >= 0.5)
    if len(xs) == 0:
        return None
    k = int(np.argmax(xs * n[0] + ys * n[1]))
    return float(xs[k]), float(ys[k])


def peel_start(grip, n):
    """The fold line's first place along its normal `n` (px): a pixel past the sticker's `grip` (grip_of)."""
    return grip[0] * n[0] + grip[1] * n[1] + 1.0


def peel_grip(peel, t, grip, size):
    """Where a sticker's `grip` (grip_of, px as the sticker lay when the peel began) is seen at clip second `t` of its
    `peel`, for a hand keyed to it: {"p": [x, y] frame fractions (outside 0..1 once it has left the frame), "z": its
    height toward the lens in short sides of the frame, "scale": how much larger it is seen there (1 on the sheet)}, in
    a frame `size` = (w, h). Before the peel the grip lies on the sheet, after it where the last moment left it. The
    same roll the sheet draws for the figure's sticker and a hollow's blank one (peel_roll from peel_start)."""
    w, h = size
    n = PE.normal({"from": peel["corner"], "angle": 0.0})
    c0 = peel_start(grip, n)
    lo = min(x * n[0] + y * n[1] for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5))
    u = 0.0 if t <= peel["t"] else min((t - peel["t"]) / peel["dur"], 1.0)
    R = peel["radius"] * min(size)
    c, f = peel_roll(n, c0, lo, R, u)
    xp, yp = grip[0] * n[0] + grip[1] * n[1], grip[1] * n[0] - grip[0] * n[1]
    xq, z = PE.lift(xp - c, c, R, f, end=c0 - c)
    x, y = xq * n[0] - yp * n[1], xq * n[1] + yp * n[0]
    px, py, m = S3.project(x, y, z, size)
    return {"p": [(px + 0.5) / w, (py + 0.5) / h], "z": z / min(size), "scale": m}


# ================================================================================================= doodles
def scatter(radii, aspect, area, clear, gap, seed, away=None):
    """Places for doodles of reach `radii` (frame heights, about their middles) in a frame `aspect` = width / height
    wide: [(x, y) frame fractions, or None for one that finds no room], inside `area` [x0, y0, x1, y1] and `gap` frame
    heights clear of every box of `clear`, of each other and, with `away` (callable((n, 2) points in frame heights from
    the top left) -> (n,) their distances in frame heights from a shape: the figure sticker as it lies), of that shape.
    The largest first, each the best of TRIES seeded candidates (the one farthest from what is there, up to a little
    more than the gap): spread out as a hand lays stickers, the same for the same seed."""
    rng = np.random.default_rng([int(seed), len(radii), 7])
    boxes = [(b[0] * aspect, b[1], b[2] * aspect, b[3]) for b in clear]
    placed, out = [], [None] * len(radii)
    for j in sorted(range(len(radii)), key=lambda i: (-radii[i], i)):
        r = radii[j]
        lo = np.array([area[0] * aspect + r, area[1] + r])
        hi = np.array([area[2] * aspect - r, area[3] - r])
        cand = lo + rng.random((TRIES, 2)) * np.maximum(hi - lo, 0.0)
        if (hi < lo).any():
            continue
        room = np.full(TRIES, np.inf) if away is None else np.asarray(away(cand), float) - r
        for X, Y, R in placed:
            room = np.minimum(room, np.hypot(cand[:, 0] - X, cand[:, 1] - Y) - R - r)
        for x0, y0, x1, y1 in boxes:
            dx = np.maximum(np.maximum(x0 - cand[:, 0], cand[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - cand[:, 1], cand[:, 1] - y1), 0.0)
            room = np.minimum(room, np.hypot(dx, dy) - r)
        ok = room >= gap
        if not ok.any():
            continue
        k = int(np.argmax(np.where(ok, np.minimum(room, 4.0 * gap + r), -np.inf) + 1e-9 * rng.random(TRIES)))
        placed.append((cand[k, 0], cand[k, 1], r))
        out[j] = (float(cand[k, 0] / aspect), float(cand[k, 1]))
    return out


def deal(doodles, seed):
    """Each doodle's own draw: [{shape, color (index into `colors`), size (a share of `size`), angle}], seeded. The
    shapes are dealt shuffled, every one before any comes again."""
    n = doodles["count"]
    rng = np.random.default_rng([int(seed), int(doodles["seed"]), 3])
    shapes = []
    while len(shapes) < n:
        shapes += [doodles["shapes"][i] for i in rng.permutation(len(doodles["shapes"]))]
    cols = rng.permutation(12 * n + 1)
    return [{"shape": shapes[j], "color": int(cols[j]),
             "size": 1.0 + doodles["vary"] * float(rng.uniform(-1.0, 1.0)),
             "angle": doodles["tilt"] * float(rng.uniform(-1.0, 1.0)),
             "spin": float(rng.uniform(-1.0, 1.0)), "twist": float(rng.choice([-1.0, 1.0])),
             "corner": CORNERS[int(rng.integers(4))]} for j in range(n)]


def peel_order(doodles, places, centre, seed):
    """{doodle: the clip second its peel starts}: the peel's times dealt to the doodles that found room, in `order`
    (shuffled by the seed, as given, nearest the figure's `centre` first, or farthest first)."""
    p = doodles["peel"]
    if p is None:
        return {}
    have = [j for j, at in enumerate(places) if at is not None]
    if p["order"] == "shuffle":
        rng = np.random.default_rng([int(seed), int(doodles["seed"]), 9])
        have = [have[i] for i in rng.permutation(len(have))]
    elif p["order"] in ("near", "far"):
        c = centre or (0.5, 0.5)
        have.sort(key=lambda j: math.hypot(places[j][0] - c[0], places[j][1] - c[1]), reverse=p["order"] == "far")
    return {j: t for j, t in zip(have, p["times"])}


def slap_order(doodles, places):
    """{doodle: its slap ({t, size, drop, squash, settle})}: the slap's times dealt to the doodles that found room, in
    the order they were dealt; a doodle past the last time is on the sheet from the start."""
    s = doodles.get("slap")
    if s is None:
        return {}
    have = [j for j, at in enumerate(places) if at is not None]
    return {j: {"t": t, **{k: s[k] for k in DSLAP}} for j, t in zip(have, s["times"])}


def ripple(places, centre):
    """Each doodle's share (0..1) of a hop's ripple: by its distance from `centre`, the nearest 0 and the farthest 1."""
    d = [None if at is None else math.hypot(at[0] - centre[0], at[1] - centre[1]) for at in places]
    have = [x for x in d if x is not None]
    if not have:
        return [0.0] * len(places)
    lo, hi = min(have), max(have)
    return [0.0 if x is None or hi <= lo else (x - lo) / (hi - lo) for x in d]


def tape_places(tapes, aspect, seed, away=None):
    """Where a sheet's washi tapes lie: [{at (x, y) frame fractions of the strip's middle, angle (degrees counter-
    clockwise), length and width (frame heights), pattern, color (index)}], `count` of them at the sheet's corners
    (across them, as a sheet is taped down) and edges (straddling them), each a little turned, seeded; one whose middle
    lies within its length of `away` (callable, as scatter's) or of another strip is left out."""
    rng = np.random.default_rng([int(seed), int(tapes["seed"]), 13])
    L, Wd = tapes["length"], tapes["width"]
    cands = []
    for cx, cy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):            # across a corner, square to its diagonal
        dx, dy = (1.0 if cx == 0 else -1.0), (1.0 if cy == 0 else -1.0)
        r = 0.18 * L
        at = (cx + dx * r / math.sqrt(2.0) / aspect, cy + dy * r / math.sqrt(2.0))
        cands.append((at, (45.0 if dx * dy > 0 else -45.0) + rng.normal(0.0, 6.0)))
    for _ in range(12):                                                          # straddling an edge, along it or across
        side = int(rng.integers(4))
        s = float(rng.uniform(0.15, 0.85))
        across = rng.random() < 0.6
        inset = (0.12 * L if across else 0.25 * Wd)
        at = {0: (s, inset), 1: (1.0 - inset / aspect, s), 2: (s, 1.0 - inset), 3: (inset / aspect, s)}[side]
        angle = (90.0 if side in (0, 2) else 0.0) if across else (0.0 if side in (0, 2) else 90.0)
        cands.append((at, angle + rng.normal(0.0, 9.0)))
    order = rng.permutation(len(cands))
    out, kept = [], []
    for i in order:
        if len(out) >= tapes["count"]:
            break
        at, angle = cands[i]
        p = np.array([[at[0] * aspect, at[1]]])
        if away is not None and float(away(p)[0]) < 0.6 * L:
            continue
        if any(math.hypot((at[0] - q[0]) * aspect, at[1] - q[1]) < 0.9 * L for q in kept):
            continue
        kept.append(at)
        out.append({"at": at, "angle": float(angle), "length": L * float(rng.uniform(0.8, 1.2)),
                    "width": Wd * float(rng.uniform(0.85, 1.15)),
                    "pattern": tapes["patterns"][int(rng.integers(len(tapes["patterns"])))],
                    "color": int(rng.integers(len(tapes["colors"])))})
    return out


def scribble_lines(shape):
    """A pen scribble `shape` as polylines [(n, 2) points] about (0, 0), x right and y down, its box about 1 across:
    drawn on the paper by hand, not stuck on it."""
    a = np.linspace(0.0, 2.0 * math.pi, 64)
    if shape == "heart":
        p = np.stack([16 * np.sin(a) ** 3, -(13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a))], 1)
        return [p / 34.0]
    if shape == "star":
        k = np.arange(11)
        r = np.where(k % 2 == 0, 0.5, 0.22)
        return [np.stack([r * np.sin(k * math.pi / 5), -r * np.cos(k * math.pi / 5)], 1)]
    if shape == "spiral":
        s = np.linspace(0.0, 5.5 * math.pi, 90)
        r = 0.03 + 0.47 * s / s[-1]
        return [np.stack([r * np.cos(s), r * np.sin(s)], 1)]
    if shape == "underline":
        x = np.linspace(-0.5, 0.5, 40)
        return [np.stack([x, 0.06 * np.sin(x * 9.0)], 1), np.stack([x[4:-6] * 0.9, 0.16 + 0.04 * np.sin(x[4:-6] * 7)], 1)]
    if shape == "arrow":
        s = np.linspace(0.0, 1.0, 30)
        shaft = np.stack([-0.5 + s, 0.25 - 0.55 * s + 0.3 * s * (1 - s)], 1)
        tip, d = shaft[-1], shaft[-1] - shaft[-4]
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        return [shaft, np.array([tip - 0.18 * d + 0.12 * n, tip, tip - 0.18 * d - 0.12 * n])]
    head = np.stack([0.3 * np.cos(a), 0.12 + 0.26 * np.sin(a)], 1)                  # a rabbit's face
    ear = np.stack([0.07 * np.cos(a), -0.28 + 0.2 * np.sin(a)], 1)
    return [head, ear + (-0.12, 0.0), ear + (0.12, 0.0), np.array([[-0.1, 0.08], [-0.1, 0.1]]),
            np.array([[0.1, 0.08], [0.1, 0.1]]), np.array([[-0.05, 0.2], [0.0, 0.23], [0.05, 0.2]])]


def doodle_at(t, d, at, wave, jump, hops, hop, peel_t, peel_dur, origin, slap=None):
    """Doodle `d` (deal()) placed `at` (frame fractions) at clip second `t`: {shown, at, angle, scale, sx, sy, lift,
    peel}. `wave` is its share of a ripple out from the figure (ripple(): 0 the nearest, 1 the farthest). A jump: hidden
    until `jump.t` + wave x stagger, then thrown from `origin` along an arc `height` high (lift 1 at the top: the sticker
    seen larger, its shadow farther), turning `spin` x its own share down to its angle, landing with a little hop and a
    squash. A hop on each of `hops` (clip seconds) lands on it, wave x ripple later (it leaves the paper `dur` before):
    up `height` and down, a twist, a squash and a few degrees of wobble as it settles. `peel`: None, or 0..1 of its
    peel's time from `peel_t`; gone after it. `slap` ({t, size, drop, squash, settle}: its own): hidden until `drop`
    before its `t`, it falls from `size` times as large, lifted, and lands on `t` with the figure's slap (slap_at)."""
    out = {"shown": True, "at": tuple(at), "angle": d["angle"], "scale": 1.0, "sx": 1.0, "sy": 1.0, "lift": 0.0,
           "peel": None}
    if peel_t is not None and t >= peel_t:
        if t >= peel_t + peel_dur:
            out["shown"] = False
            return out
        out["peel"] = (t - peel_t) / peel_dur
        return out
    if slap is not None:
        if t < slap["t"] - slap["drop"]:
            out["shown"] = False
            return out
        if t < slap["t"] + slap["settle"]:
            s = slap_at([slap], t)
            out.update(scale=s["scale"], sx=s["sx"], sy=s["sy"], lift=s["lift"])
            return out
    land = None
    if jump is not None:
        t0 = jump["t"] + wave * jump["stagger"]
        if t < t0:
            out["shown"] = False
            return out
        u = (t - t0) / jump["dur"]
        if u < 1.0:
            a = min(u / JUMP_LAND, 1.0)
            e = 1.0 - (1.0 - a) ** 2.2
            out["at"] = (origin[0] + (at[0] - origin[0]) * e, origin[1] + (at[1] - origin[1]) * e)
            if u < JUMP_LAND:
                z = math.sin(math.pi * a)
            else:
                z = JUMP_BOUNCE * math.sin(math.pi * (u - JUMP_LAND) / (1.0 - JUMP_LAND))
            out["lift"] = jump["height"] * z
            out["angle"] = d["angle"] + jump["spin"] * d["spin"] * (1.0 - e)
            out["scale"] = 1.0 + 0.6 * out["lift"]
            if u >= JUMP_LAND:
                land = (u - JUMP_LAND) * jump["dur"]
            if land is not None and land > 0:
                q = spring(land, LAND["squash"], LAND["settle"])
                out["sx"], out["sy"] = 1.0 / (1.0 - q), 1.0 - q
            return out
        land = t - (t0 + jump["dur"])
    wobble = 0.0
    if hop is not None:
        k = None
        for i, h in enumerate(hops):
            if t >= h - hop["dur"] + wave * hop["ripple"]:
                k = i
        if k is not None:
            h0 = hops[k] - hop["dur"] + wave * hop["ripple"]
            u = (t - h0) / hop["dur"]
            sign = 1.0 if k % 2 == 0 else -1.0
            if u < 1.0:
                z = math.sin(math.pi * u)
                out["lift"] = hop["height"] * z
                out["scale"] = 1.0 + 0.6 * out["lift"]
                out["angle"] = d["angle"] + 7.0 * d["twist"] * sign * z
                return out
            after = t - (h0 + hop["dur"])
            if land is None or after < land:
                land = after
                wobble = 4.0 * d["twist"] * sign * spring(after, 1.0, HOP_WOBBLE)
    if land is not None and 0.0 < land < LAND["settle"]:
        q = spring(land, LAND["squash"], LAND["settle"])
        out["sx"], out["sy"] = 1.0 / (1.0 - q), 1.0 - q
    out["angle"] += wobble
    return out


# ================================================================================================= fling
def fling_items(fx, times, aim=None):
    """The stickers a fling throws: on each hit of `times` its `count`, each {t0, v (heading, frame heights a unit of
    depth), shape, color (index), size (share), angle, spin (degrees a second), flip (the age it turns over at, or
    None)}; seeded by the hit's place among them, so a lane's hits throw the same stickers whatever span asks. They fan
    round the clock, leaning up; with `aim` ((dx, dy) frame heights from the point they leave from: her face) the first
    `aimed` of each hit are thrown to pass over it at FLING_AIM of their life (thrown `away`, at 1 - FLING_AIM)."""
    out = []
    for i, h in enumerate(times):
        rng = np.random.default_rng([int(fx["seed"]), i, 11])
        a0 = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(fx["count"]):
            a = a0 + 2.0 * math.pi * (k + rng.uniform(-0.3, 0.3)) / fx["count"]    # around the clock, not in a clump
            v = np.array([math.cos(a), math.sin(a) - FLING_UP])
            v = v / np.linalg.norm(v) * fx["spread"] * rng.uniform(0.8, 1.3)
            if aim is not None and k < fx["aimed"]:
                z = 1.0 - (1.0 - 1.0 / fx["grow"]) * FLING_AIM
                v = np.asarray(aim, float) * z / FLING_AIM + rng.normal(0.0, 0.004, 2)
            flips = rng.random() < fx["flip"]
            out.append({"t0": h + rng.uniform(0.0, FLING_STAGGER), "v": (float(v[0]), float(v[1])),
                        "shape": fx["shapes"][int(rng.integers(len(fx["shapes"])))],
                        "color": int(rng.integers(1 << 16)), "size": float(rng.uniform(0.8, 1.2)),
                        "angle": float(rng.uniform(-30.0, 30.0)), "spin": fx["spin"] * float(rng.uniform(-1.0, 1.0)),
                        "flip": float(rng.uniform(0.15, 0.6)) * fx["life"] if flips else None})
    return out


def fling_at(fx, item, t):
    """Where a thrown sticker is at clip second `t`: None before it leaves and after its `life`, else {off (dx, dy) frame
    heights from the point it is thrown from, scale (frame heights: its longer side), angle (degrees), face (the cosine
    of its flip, 1 face up: under 0 its back shows, for a frame or two of a quick turn over), depth (1 at the point,
    1 / grow at the end: the nearer drawn last)}. It flies straight at the lens at a steady speed: seen at depth z it is
    1 / z as large and its heading 1 / z as far out, so it grows ever faster and leaves by the frame's edges. Thrown
    `away` its flight is played backwards (its spin and flip still run forward): it comes in from its heading `grow`
    times as far out and `grow` times its size, falls away from the lens slowing and shrinking into the point, and is
    gone there in a pop (FLING_POP)."""
    age = t - item["t0"]
    if age < 0.0 or age > fx["life"]:
        return None
    u = age / fx["life"]
    pop = age
    if fx["away"]:
        u, pop = 1.0 - u, fx["life"] - age
    z = 1.0 - (1.0 - 1.0 / fx["grow"]) * u
    pop = min(pop / FLING_POP, 1.0)
    pop = 1.0 - (1.0 - pop) ** 3
    face = 1.0
    if item["flip"] is not None:
        face = math.cos(2.0 * math.pi * min(max((age - item["flip"]) / FLING_FLIP, 0.0), 1.0))
    return {"off": (item["v"][0] * u / z, item["v"][1] * u / z), "scale": fx["size"] * item["size"] * pop / z,
            "angle": item["angle"] + item["spin"] * age, "face": face, "depth": z}
