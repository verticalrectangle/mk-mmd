"""The table, bpy-free (docs/design.md: Frame effects, Sticker sheets: the table): a sticker sheet laid over a video
instead of paper and seen through a camera of its own. mkmmd.core.sheet checks a sheet's table keys with `normalize`;
the timing is here (numpy, tested: a function of the clip second, so any frame comes out the same in any order) and
mkmmd/table.py draws it.

A sheet with a `clip` is a table: the clip (a video read by mkmmd.footage, `in`, `rate`, `loop`) lies under everything,
its luma (softened by `soften` first) through a gradient map of palette colours (`grade`: spread over a percentile
stretch, or each colour placed at a luma of its own, `lumas`), and everything on it is placed
in the table's own coordinates: fractions of the clip's frame (0, 0 its top left), sizes in its heights. Its `view` is
the camera: keys of the part of the table the frame shows (the table point `at` in the frame's middle, `height` the table
heights the frame is tall or `width` those it is wide at least, a `turn`), so one table is seen close and wide, every
part of it drawn at the output's own size; past the clip's edges the table goes on, the clip reflected there. On it lie,
back to front:

    cup      the coffee's surface: an ellipse of the table, through a map of its own if it has one; what floats lies on
             it, cut by its rim, a thin line of light drawn along that rim at the output's size
    rings    drift rings running out over the coffee from each touch: still op-art that seems to turn by itself (the
             `spiral` kind's `pattern = "drift"`, mkmmd.fields.drift), fading over their life
    pads     cut-paper lily pads floating there, each bobbing as the rings pass under it; the pad under a touch dips
    drops    droplet stickers thrown out from the rim of the pad a touch lands on: a doodle group of their own
             (mkmmd.core.sheet's doodles) that lands on the coffee, clear of the pads and her
    figure   her live sticker (`objects`, from a plate `shot`), lying on the first pad and dipping with it; with `track`
             the plate's camera may zoom and move: a world point seen through it keeps her where she lies on the table
    doodles  the sheet's doodle stickers (slapped down on their times, hopping on the hits)

A `touch` is something landing on the coffee at a second and a place: the pad there dips and bobs, rings run out, drops
jump. A `punch` zooms the view into the cup until the coffee's ellipse covers the frame, landing on its `t`, while the
picture under the table (the cut's own frame: the next shot, cut in under the table at the punch's start) shows through
the coffee, growing with it until on `t` it is the whole frame; `out = true` plays it backwards, out of the cup.

The view moves like a zoom lens with a shift: between two keys its zoom (frame px per table px) and turn go by the key
arrived at's ease, and the middle of the closer of the two (the larger zoom) slides straight across the frame between
where each shows it, so a push in on her face carries her face to the frame's middle as the lens closes on it; a plate
camera keyed at the same seconds with lenses in proportion to the view's zoom covers it on every frame between them."""
import math

import numpy as np

from . import pin as PIN
from . import tween as TW

KEYS = ("clip", "in", "rate", "loop", "grade", "view", "cup", "inside", "touch", "pads", "rings", "drops", "track",
        "punch", "lift", "smear")
KEYS += PIN.KEYS      # a figure pinned to a seat its clip's camera moves (mkmmd.core.pin)
NOT_ON_TABLE = ("paper", "hollows", "peel", "tapes", "scribbles", "slap")   # a paper sheet's own: a table has none
GRADE = {"map": ["text", "iris", "rose", "overlay", "surface"], "stretch": [1.0, 99.0], "at": None, "lumas": None,
         "soften": 0.0, "cool": None, "warm": None, "lean": None, "warmth": None}
BY_WARMTH = ("cool", "warm", "lean", "warmth")      # a grade's keys by warmth: any of them makes it a grade of two ramps
RAMPS = {"cool": [[0.0, "text"], [0.28, "pine"], [0.55, "foam"], [0.8, "foam:surface:0.6"], [1.0, "surface"]],
         "warm": [[0.0, "text"], [0.25, "love"], [0.5, "rose"], [0.75, "gold:surface:0.45"], [1.0, "surface"]]}
WARMTH = 2.2           # how hard a pixel's warmth (its red less its blue, 0..1) picks between a grade's two ramps
SOFTEN_MAX = 0.05      # table heights: the most a table's luma is softened before its map
VIEW_EASES = ("inout", "in", "out", "linear", "step")
VIEW = {"at": [0.5, 0.5], "height": 1.0, "width": None, "turn": 0.0, "ease": "inout"}
CUP_LINE = {"color": "surface", "width": 2.0, "alpha": 0.55}
CUP = {"line": CUP_LINE, "map": None, "lumas": None}
INNER = {"color": "text", "alpha": 0.35, "width": 0.012, "offset": [0.008, 0.012]}   # table heights
INSIDE = {"shadow": INNER}
TOUCH = {"at": None, "dip": 0.05, "settle": 1.3}
PAD = {"size": 0.1, "turn": 0.0, "notch": 36.0, "color": "pine", "light": "foam", "veins": True, "edge": 2.0,
       "seed": 1}
RINGS = {"colors": ["text", "iris", "rose", "surface"], "size": 0.012, "speed": 0.09, "count": 3, "gap": 0.24,
         "life": 2.0, "band": 3.0, "opacity": 0.45, "motif": "tie", "seed": 1}
DROPS = {"count": 8, "shapes": ["drop"], "size": 0.014, "vary": 0.3, "gap": 0.004}
DROP_LEAST = 5e-4      # table heights: the smallest drop (a doodle on paper is at least 0.005 frame heights)
REACH = 0.07           # drops land within this many table heights of the touch they jump from, by default
PUNCH = {"dur": 0.3, "out": False, "fade": 0.35}
RING_IN = 0.12         # seconds a ring front takes to come up after it leaves the touch
RING_FALL = 0.18       # each later front of a touch is this much fainter than the one before
BOB = (0.012, 1.8, 1.1)  # a pad's bob as a ring front passes under it: its size's share, Hz, seconds to die away


def _fx():
    from . import framefx as FX          # framefx imports sheet, which imports this module
    return FX


# ================================================================================================= the entries
def table_of(spec):
    """Whether a sheet `spec` is a table: it has a `clip`."""
    return spec.get("clip") is not None


def normalize(FX, spec, out, palette, what):
    """A sheet's table keys checked and filled in, in place: without a `clip` none may be given (each is the table's);
    with one, paper's own furniture is refused (the clip is the ground)."""
    if not table_of(spec):
        given = sorted(k for k in KEYS if k != "clip" and spec.get(k) is not None)
        if given:
            raise FX.FxError(f"{what}: {', '.join(given)} belong to a table: give its `clip` (the video under it)")
        out.update({k: None for k in KEYS})
        return
    paper = sorted(k for k in NOT_ON_TABLE if spec.get(k) not in (None, False))
    if paper:
        raise FX.FxError(f"{what}: {', '.join(paper)}: not on a table (its clip is the ground, and her sticker lands on "
                         f"it by a `touch`): leave them out")
    out.update(paper=None, hollows=[], peel=None, tapes=None, scribbles=None, slap=[])
    if not isinstance(out["clip"], str) or not out["clip"].strip():
        raise FX.FxError(f"{what}: clip = {out['clip']!r}: a video file (project-relative or absolute) or an asset slug")
    out["in"] = FX._num(spec.get("in", 0.0), f"{what} in", lo=0.0)
    out["rate"] = FX._num(spec.get("rate", 1.0), f"{what} rate", 0.0, 16.0)
    out["loop"] = spec.get("loop", False)
    if not isinstance(out["loop"], bool):
        raise FX.FxError(f"{what}: loop is true or false")
    out["grade"] = _grade(FX, spec.get("grade"), palette, what)
    out["view"] = _view(FX, spec.get("view"), what)
    out["cup"] = _cup(FX, spec.get("cup"), palette, what)
    g, c = spec.get("grade") or {}, out["cup"]
    if out["grade"]["lumas"] is not None and ("stretch" in g or "at" in g) and (
            c is None or c["map"] is None or c["lumas"] is not None):
        raise FX.FxError(f"{what}: grade: `lumas` place the map's colours on the clip's own luma, so no map is spread "
                         f"over a measured `stretch` (or one at `at`): leave them out")
    out["inside"] = _inside(FX, spec.get("inside"), palette, what)
    if out["inside"] is not None and out["cup"] is None:
        raise FX.FxError(f"{what}: inside: the cut's own frame lies in the cup: give the `cup`")
    out["pads"] = [_pad(FX, p, palette, f"{what} pads[{i}]") for i, p in enumerate(_list(spec.get("pads")))]
    out["touch"] = sorted((_touch(FX, t, f"{what} touch") for t in _list(spec.get("touch"))), key=lambda k: k["t"])
    if any(t["at"] is None for t in out["touch"]) and not out["pads"]:
        raise FX.FxError(f"{what}: a touch without `at` lands on the first pad: give `pads`, or the touch's `at`")
    out["rings"] = _rings(FX, spec.get("rings"), palette, what)
    if out["rings"] is not None and not out["touch"]:
        raise FX.FxError(f"{what}: rings run out from a touch: give `touch`")
    if (out["rings"] is not None or out["pads"]) and out["cup"] is None:
        raise FX.FxError(f"{what}: rings and pads float on the coffee: give the `cup`")
    out["track"] = _track(FX, spec.get("track"), spec, what)
    out["pin"] = PIN.normalize(FX, spec.get("pin"), spec, palette, what)
    out["punch"] = sorted((_punch(FX, p, what) for p in _list(spec.get("punch"))), key=lambda p: p["t"])
    if out["punch"] and out["cup"] is None:
        raise FX.FxError(f"{what}: a punch goes through the cup: give the `cup`")
    for a, b in zip(out["punch"], out["punch"][1:]):
        if window(a)[1] > window(b)[0] + 1e-9:
            raise FX.FxError(f"{what}: the punches at {a['t']:g} and {b['t']:g} s overlap")
    out["lift"] = None if spec.get("lift") is None else FX._keys(spec["lift"], f"{what} lift")
    from . import sky as SY                # mkmmd.core.sky imports this module
    out["smear"] = None if spec.get("smear") is None else SY.smear(FX, spec["smear"], f"{what} smear", table=True)
    if out["smear"] is not None and out["from"] + out["smear"]["dur"] > out["to"] + 1e-9:
        raise FX.FxError(f"{what}: smear: {out['smear']['dur']:g} s is longer than the table is on")


def drops(FX, v, palette, spec, what, doodles):
    """A table's `drops`: a doodle group (`doodles`, mkmmd.core.sheet's checker, over DROPS' defaults; sizes in table
    heights, as small as a drop seen from afar) that lands on the coffee within `reach` table heights of the first touch
    (or in its own `area`), thrown out on that touch's second (or its own `jump`'s `t`) from the rim of the pad the
    touch lands on (drop_origins), unless its jump gives a `from`. None without them."""
    if v is None or v is False:
        return None
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: drops = {v!r}: expected a table (the doodles' keys and `reach`) or false")
    if not table_of(spec):
        raise FX.FxError(f"{what}: drops belong to a table: give its `clip`")
    if not _list(spec.get("touch")) and (v.get("jump") is None or v.get("area") is None):
        raise FX.FxError(f"{what}: drops jump from a touch and land round it: give `touch` (or their own `jump` and "
                         f"`area`)")
    reach = FX._num(v.get("reach", REACH), f"{what} drops reach", 1e-3, 2.0)
    d = doodles(FX, {**DROPS, **{k: x for k, x in v.items() if k != "reach"}}, palette, spec, f"{what} drops",
                least=DROP_LEAST)
    d["reach"] = reach
    d["own_area"] = v.get("area") is not None
    if d["jump"] is None:
        from . import sheet as ST
        d["jump"] = {**{k: x for k, x in ST.JUMP.items()}, "t": None, "from": None}    # the first touch's, when drawn
    return d


def _list(v):
    return [] if v is None else [v] if isinstance(v, dict) else list(v)


def _table(FX, v, defaults, what, name, required=()):
    known = sorted(set(defaults) | set(required))
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table ({{{', '.join(known)}}})")
    extra = sorted(set(v) - set(known))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {known})")
    lack = [k for k in required if v.get(k) is None]
    if lack:
        raise FX.FxError(f"{what}: {name} needs {', '.join(f'`{k}`' for k in lack)}")
    return {**defaults, **v}


def _ramp(FX, m, lumas, palette, what, name):
    """A gradient map's colours (2 to 8, dark to light) and the luma (0..1) each lies at, rising, or None."""
    if isinstance(m, str) or not isinstance(m, (list, tuple)) or not 2 <= len(m) <= 8:
        raise FX.FxError(f"{what}: {name} map = {m!r}: 2 to 8 colours, dark to light (palette slots)")
    cols = [FX._colour(c, palette, f"{what} {name} map") for c in m]
    if lumas is None:
        return cols, None
    if isinstance(lumas, str) or not isinstance(lumas, (list, tuple)) or len(lumas) != len(m):
        raise FX.FxError(f"{what}: {name} lumas = {lumas!r}: one luma (0..1) for each of the map's {len(m)} colours")
    at = [FX._num(x, f"{what} {name} lumas", 0.0, 1.0) for x in lumas]
    if any(b <= a for a, b in zip(at, at[1:])):
        raise FX.FxError(f"{what}: {name} lumas = {lumas!r}: rising, as the map's colours go dark to light")
    return cols, at


def _ramp_by(FX, m, palette, what, name):
    """A ramp of a grade by warmth: {colors, at}, 2 to 8 colours dark to light each at its place (0..1 of the stretched
    luma, rising): given as `[colour, ...]` (spread evenly) or `[[u, colour], ...]` (each at its u)."""
    if isinstance(m, str) or not isinstance(m, (list, tuple)) or not 2 <= len(m) <= 8:
        raise FX.FxError(f"{what}: grade {name} = {m!r}: 2 to 8 colours, dark to light, or [[u, colour], ...]")
    pairs = [isinstance(c, (list, tuple)) for c in m]
    if all(pairs):
        if any(len(c) != 2 for c in m):
            raise FX.FxError(f"{what}: grade {name} = {m!r}: each [u, colour] (u 0..1 of the stretched luma)")
        at = [FX._num(c[0], f"{what} grade {name} u", 0.0, 1.0) for c in m]
        if any(b <= a for a, b in zip(at, at[1:])):
            raise FX.FxError(f"{what}: grade {name} = {m!r}: its u rising, as its colours go dark to light")
        cols = [FX._colour(c[1], palette, f"{what} grade {name}") for c in m]
    elif not any(pairs):
        at = np.linspace(0.0, 1.0, len(m)).tolist()
        cols = [FX._colour(c, palette, f"{what} grade {name}") for c in m]
    else:
        raise FX.FxError(f"{what}: grade {name} = {m!r}: all colours (spread evenly) or all [u, colour]")
    return {"colors": cols, "at": at}


def _grade(FX, v, palette, what):
    """A table's `grade`: one gradient `map` over the luma (its `lumas`, or spread over a measured `stretch`), or, given
    any of BY_WARMTH, a grade by warmth: the stretched luma through a `cool` and a `warm` ramp (RAMPS' by default), the
    two mixed by each pixel's warmth (its red less its blue) times `warmth`, plus `lean` (-1 .. 1: the whole clip toward
    the warm ramp, or below 0 the cool one)."""
    given = {} if v is None else v
    g = _table(FX, given, GRADE, what, "grade")
    lo, hi = FX._pair(g["stretch"], f"{what} grade stretch")
    if not 0.0 <= lo < hi <= 100.0:
        raise FX.FxError(f"{what}: grade stretch = {g['stretch']!r}: [low, high] percentiles of the luma, 0 <= low < high "
                         f"<= 100")
    out = {"stretch": [lo, hi], "at": None if g["at"] is None else FX._num(g["at"], f"{what} grade at", lo=0.0),
           "soften": FX._num(g["soften"], f"{what} grade soften", 0.0, SOFTEN_MAX)}
    if not any(given.get(k) is not None for k in BY_WARMTH):
        cols, lumas = _ramp(FX, g["map"], g["lumas"], palette, what, "grade")
        return {"map": cols, "lumas": lumas, **out, "cool": None, "warm": None, "lean": None, "warmth": None}
    one = sorted(k for k in ("map", "lumas") if given.get(k) is not None)
    if one:
        raise FX.FxError(f"{what}: grade: {', '.join(one)} make a grade of one map, `cool`, `warm`, `lean` and `warmth` "
                         f"a grade by warmth: give one or the other")
    return {"map": None, "lumas": None, **out,
            "cool": _ramp_by(FX, RAMPS["cool"] if g["cool"] is None else g["cool"], palette, what, "cool"),
            "warm": _ramp_by(FX, RAMPS["warm"] if g["warm"] is None else g["warm"], palette, what, "warm"),
            "lean": FX._num(0.0 if g["lean"] is None else g["lean"], f"{what} grade lean", -1.0, 1.0),
            "warmth": FX._num(WARMTH if g["warmth"] is None else g["warmth"], f"{what} grade warmth", 0.0, 20.0)}


def _view(FX, v, what):
    """`view`: one key or a list of {t, at, height, width, turn, ease} (sorted by t); a key's missing parts are the key
    before's (the first's: VIEW's). None: the whole table's height in the frame."""
    if v is None:
        return None
    keys = [v] if isinstance(v, dict) else v
    if not isinstance(keys, (list, tuple)) or not keys or not all(isinstance(k, dict) for k in keys):
        raise FX.FxError(f"{what}: view = {v!r}: a key or a list of {{t, at, height, width, turn, ease}}")
    raw = []
    for k in keys:
        extra = sorted(set(k) - {"t"} - set(VIEW))
        if extra:
            raise FX.FxError(f"{what}: view: unknown keys {extra} (known: t, {', '.join(sorted(VIEW))})")
        if k.get("t") is None and len(keys) > 1:
            raise FX.FxError(f"{what}: view: {k!r} has no `t` (clip seconds)")
        raw.append((FX._num(k.get("t", 0.0), f"{what} view t"), k))
    raw.sort(key=lambda p: p[0])
    if any(a[0] == b[0] for a, b in zip(raw, raw[1:])):
        raise FX.FxError(f"{what}: view: two keys at the same second")
    out, last = [], dict(VIEW)
    for t, k in raw:
        cur = {**last, **{c: x for c, x in k.items() if c != "t"}}
        if "height" in k and "width" not in k:
            cur["width"] = None
        if "width" in k and "height" not in k:
            cur["height"] = None
        if cur["ease"] not in VIEW_EASES:
            raise FX.FxError(f"{what}: view ease {cur['ease']!r} is none of {', '.join(VIEW_EASES)}")
        h = None if cur["height"] is None else FX._num(cur["height"], f"{what} view height", 1e-3, 100.0)
        w = None if cur["width"] is None else FX._num(cur["width"], f"{what} view width", 1e-3, 100.0)
        if h is None and w is None:
            raise FX.FxError(f"{what}: view: a key needs its `height` or `width` (table heights)")
        out.append({"t": t, "at": FX._pair(cur["at"], f"{what} view at"), "height": h, "width": w,
                    "turn": FX._num(cur["turn"], f"{what} view turn"), "ease": cur["ease"]})
        last = {**cur, "height": h, "width": w}
    return out


def _cup(FX, v, palette, what):
    if v is None:
        return None
    c = _table(FX, v, CUP, what, "cup", required=("center", "radius"))
    r = c["radius"]
    rx, ry = FX._pair([r, r] if isinstance(r, (int, float)) and not isinstance(r, bool) else r, f"{what} cup radius")
    if min(rx, ry) <= 0:
        raise FX.FxError(f"{what}: cup radius = {c['radius']!r}: table heights, positive (or [across, down])")
    line = c["line"]
    if line in (None, False):
        line = None
    else:
        line = _table(FX, line, CUP_LINE, what, "cup line")
        line = {"color": FX._colour(line["color"], palette, f"{what} cup line"),
                "width": FX._num(line["width"], f"{what} cup line width", 0.0, 40.0),
                "alpha": FX._num(line["alpha"], f"{what} cup line alpha", 0.0, 1.0)}
    if c["map"] is None and c["lumas"] is not None:
        raise FX.FxError(f"{what}: cup lumas place the coffee's own `map`: give it")
    cols, lumas = (None, None) if c["map"] is None else _ramp(FX, c["map"], c["lumas"], palette, what, "cup")
    return {"center": FX._pair(c["center"], f"{what} cup center"), "radius": [rx, ry], "line": line, "map": cols,
            "lumas": lumas}


def _inside(FX, v, palette, what):
    """`inside`: the cut's own frame in the cup, `{point, at, metre, shadow}`: its world `point` (seen through the camera
    of the shot in the cut) on the table's `at`, a metre at its depth `metre` table heights; `shadow` the soft inner
    shadow the cup's wall casts on it (`{color, alpha, width, offset}`, table heights; false: none)."""
    if v is None:
        return None
    s = _table(FX, v, INSIDE, what, "inside", required=("point", "at", "metre"))
    if not isinstance(s["point"], str) or not s["point"].strip():
        raise FX.FxError(f"{what}: inside point = {s['point']!r}: a world point (an `mk q` expression) that stays still")
    sh = s["shadow"]
    if sh not in (None, False):
        sh = _table(FX, sh, INNER, what, "inside shadow")
        sh = {"color": FX._colour(sh["color"], palette, f"{what} inside shadow"),
              "alpha": FX._num(sh["alpha"], f"{what} inside shadow alpha", 0.0, 1.0),
              "width": FX._num(sh["width"], f"{what} inside shadow width", 1e-4, 1.0),
              "offset": list(FX._pair(sh["offset"], f"{what} inside shadow offset"))}
    return {"point": s["point"], "at": FX._pair(s["at"], f"{what} inside at"),
            "metre": FX._num(s["metre"], f"{what} inside metre", 1e-5, 100.0), "shadow": sh or None}


def _touch(FX, v, what):
    t = _table(FX, v, TOUCH, what, "touch", required=("t",))
    return {"t": FX._num(t["t"], f"{what} t"), "at": None if t["at"] is None else FX._pair(t["at"], f"{what} at"),
            "dip": FX._num(t["dip"], f"{what} dip", 0.0, 0.5), "settle": FX._num(t["settle"], f"{what} settle", 0.1, 10.0)}


def _pad(FX, v, palette, what):
    p = _table(FX, v, PAD, what, "pad", required=("at",))
    if not isinstance(p["veins"], bool):
        raise FX.FxError(f"{what}: veins is true or false")
    return {"at": FX._pair(p["at"], f"{what} at"), "size": FX._num(p["size"], f"{what} size", 1e-3, 4.0),
            "turn": FX._num(p["turn"], f"{what} turn"), "notch": FX._num(p["notch"], f"{what} notch", 0.0, 120.0),
            "color": FX._colour(p["color"], palette, f"{what} color"),
            "light": FX._colour(p["light"], palette, f"{what} light"), "veins": p["veins"],
            "edge": FX._num(p["edge"], f"{what} edge", 0.0, 40.0),
            "seed": FX._num(p["seed"], f"{what} seed", lo=0, integer=True)}


def _rings(FX, v, palette, what):
    if v is None or v is False:
        return None
    r = _table(FX, {} if v is True else v, RINGS, what, "rings")
    cs = r["colors"]
    if isinstance(cs, str) or not isinstance(cs, (list, tuple)) or not 2 <= len(cs) <= 4:
        raise FX.FxError(f"{what}: rings colors = {cs!r}: 2 to 4 colours (the drift's tones, by their luma)")
    from . import fields as CF
    if r["motif"] not in CF.DRIFT_MOTIFS:
        raise FX.FxError(f"{what}: rings motif {r['motif']!r} is none of {', '.join(CF.DRIFT_MOTIFS)}")
    w = f"{what} rings"
    return {"colors": [FX._colour(c, palette, f"{w} colors") for c in cs], "motif": r["motif"],
            "size": FX._num(r["size"], f"{w} size", 1e-4, 1.0), "speed": FX._num(r["speed"], f"{w} speed", 1e-4, 10.0),
            "count": FX._num(r["count"], f"{w} count", 1, 12, integer=True),
            "gap": FX._num(r["gap"], f"{w} gap", 0.0, 10.0), "life": FX._num(r["life"], f"{w} life", 0.05, 20.0),
            "band": FX._num(r["band"], f"{w} band", 0.5, 40.0),
            "opacity": FX._num(r["opacity"], f"{w} opacity", 0.0, 1.0),
            "seed": FX._num(r["seed"], f"{w} seed", lo=0, integer=True)}


def _track(FX, v, spec, what):
    if v is None:
        return None
    t = _table(FX, v, {}, what, "track", required=("point", "at", "metre"))
    if spec.get("shot") is None:
        raise FX.FxError(f"{what}: track follows the camera of the figure's plate: give its `shot`")
    placed = sorted(k for k in ("at", "height") if spec.get(k) is not None)
    if placed:
        raise FX.FxError(f"{what}: {', '.join(placed)}: a tracked figure lies where `track` puts it (its `at`, `metre`)")
    if not isinstance(t["point"], str) or not t["point"].strip():
        raise FX.FxError(f"{what}: track point = {t['point']!r}: a world point (an `mk q` expression) that stays still")
    return {"point": t["point"], "at": FX._pair(t["at"], f"{what} track at"),
            "metre": FX._num(t["metre"], f"{what} track metre", 1e-5, 100.0)}


def _punch(FX, v, what):
    p = _table(FX, v, PUNCH, what, "punch", required=("t",))
    if not isinstance(p["out"], bool):
        raise FX.FxError(f"{what}: punch out is true or false")
    return {"t": FX._num(p["t"], f"{what} punch t"), "dur": FX._num(p["dur"], f"{what} punch dur", 0.02, 10.0),
            "out": p["out"], "fade": FX._num(p["fade"], f"{what} punch fade", 0.0, 1.0)}


def window(p):
    """(start, end) clip seconds of a punch: an in punch the `dur` before its `t`, an out one the `dur` after it."""
    return (p["t"], p["t"] + p["dur"]) if p["out"] else (p["t"] - p["dur"], p["t"])


# ================================================================================================= what it reads
def track_key(fx):
    return f"x{fx['index']}t"


def inside_key(fx):
    return f"x{fx['index']}i"


def own_points(fx):
    """[(key, expression, shot)]: the world points a table reads through its figure's own plate `shot`, on the frames it
    reads that figure (point/<key>/): its `track`'s point."""
    if fx.get("track") is None:
        return []
    return [(track_key(fx), fx["track"]["point"], fx["shot"])]


def points(fx):
    """[(key, expression)]: the world points a table reads through the camera of the shot in the cut on the frames of
    its span (point/<key>/): its `inside`'s point."""
    return [] if fx.get("inside") is None else [(inside_key(fx), fx["inside"]["point"])]


# ================================================================================================= the view
def frame_height(key, aspect):
    """The table heights tall the frame of a view key is in an output `aspect` (width / height) wide: its `height`, or as
    tall as its `width` needs, whichever is more."""
    return max(key["height"] or 0.0, (key["width"] or 0.0) / aspect)


def key_map(key, size, table):
    """(zoom, turn radians, middle px) of view `key` in an output of `size` (w, h) over a table of `table` (w, h) px:
    frame px per table px, its turn, and the table px in the frame's middle."""
    W, H = size
    return (H / (frame_height(key, W / H) * table[1]), math.radians(key["turn"]),
            (key["at"][0] * table[0] - 0.5, key["at"][1] * table[1] - 0.5))


def similarity(p, z, a, q):
    """The 2 x 3 map (table px -> frame px) that puts table px `q` at frame px `p`, zoomed `z` and turned `a` radians
    counter-clockwise on screen about it."""
    c, s = math.cos(a), math.sin(a)
    A = z * np.array([[c, s], [-s, c]])
    return np.hstack([A, (np.asarray(p, float) - A @ np.asarray(q, float))[:, None]])


def middle(size):
    """The frame px in the middle of a picture of `size` (w, h)."""
    return ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)


def _ease(u, kind):
    if kind == "step":
        return 1.0 if u >= 1.0 else 0.0
    return TW.clamp01(u) if kind == "linear" else TW.ease(u, kind)


def between(a, b, e, size):
    """The view `e` of the way (eased) from key map `a` to key map `b` (key_map's), a 2 x 3 map: the zoom and turn go
    straight from the one to the other, and the middle of the closer view (the larger zoom) slides straight across the
    frame from where the one shows it to where the other does."""
    C = np.asarray(middle(size), float)
    (za, ta, qa), (zb, tb, qb) = a, b
    P = np.asarray(qb if zb >= za else qa, float)
    pa = similarity(C, za, ta, qa) @ np.r_[P, 1.0]
    pb = similarity(C, zb, tb, qb) @ np.r_[P, 1.0]
    return similarity(pa + (pb - pa) * e, za + (zb - za) * e, ta + (tb - ta) * e, P)


def view_map(view, t, size, table, half=0.0):
    """The 2 x 3 map (table px -> frame px) of a table's `view` at clip second `t` in an output of `size` (w, h) over a
    table of `table` (w, h) px: held before its first key and after its last; between two keys by the ease of the key
    arrived at (`step`: held until its second, then there on the frame of it: `half`, half a frame's seconds, the
    rounding of a second to its frame). Without a view the table's height fills the frame."""
    view = view or [dict(VIEW, t=0.0)]
    if t <= view[0]["t"] or len(view) == 1:
        return similarity(middle(size), *key_map(view[0], size, table))
    for a, b in zip(view, view[1:]):
        if b["ease"] == "step" and b["t"] - half <= t < b["t"]:      # the frame of a step's second shows it
            return view_map(view, b["t"], size, table, half)
        if t < b["t"]:
            u = (t - a["t"]) / (b["t"] - a["t"])
            return between(key_map(a, size, table), key_map(b, size, table), _ease(u, b["ease"]), size)
    return similarity(middle(size), *key_map(view[-1], size, table))


def zoom_of(M):
    """The zoom of a 2 x 3 similarity (its scale)."""
    M = np.asarray(M, float)
    return float(math.hypot(M[0, 0], M[1, 0]))


def turn_of(M):
    """The turn of a 2 x 3 similarity, degrees counter-clockwise on screen."""
    M = np.asarray(M, float)
    return math.degrees(math.atan2(M[0, 1], M[0, 0]))


def centre_of(M, size):
    """The table px a 2 x 3 similarity shows in the middle of a frame of `size`."""
    M = np.asarray(M, float)
    return tuple(np.linalg.solve(M[:, :2], np.asarray(middle(size), float) - M[:, 2]))


# ================================================================================================= the punch
def punch_at(punch, t):
    """The punch going on at clip second `t`: (punch, u 0..1 through its window), or None outside every window."""
    for p in punch or ():
        a, b = window(p)
        if a <= t < b:
            return p, (t - a) / (b - a)
    return None


def gone(punch, t):
    """Whether the table is not there at clip second `t`: after an in punch's `t` (the cut's own frame is the picture)
    until an out punch's `t`; before the first punch when that is an out punch."""
    if not punch:
        return False
    state = punch[0]["out"]                  # gone at the start when the first punch brings the table in
    for p in punch:
        if p["t"] <= t:
            state = not p["out"]
    return state


def disc_zoom(cup, size, table):
    """The zoom (frame px per table px) at which the cup's ellipse, centred in the frame, covers it: its shorter radius
    reaches the frame's corners."""
    return math.hypot(size[0], size[1]) / 2.0 / (min(cup["radius"]) * table[1])


def punch_map(p, u, rest, size, table, cup):
    """(the 2 x 3 view map, the scale of the picture shown in the cup) `u` of the way through punch `p`'s window: from
    `rest` (the view's own map at the window's far end: its start for an in punch, its end for an out one) into the
    cup, whose ellipse is centred in the frame and covers it at the punch's `t` (in: eased in, falling into the cup; out:
    eased out). The picture in the cup is shown at the zoom's share of that last one: its own size on `t`."""
    zd = disc_zoom(cup, size, table)
    r = (zoom_of(rest), math.radians(turn_of(rest)), centre_of(rest, size))
    d = (zd, r[1], (cup["center"][0] * table[0] - 0.5, cup["center"][1] * table[1] - 0.5))
    M = between(r, d, TW.ease(u, "in"), size) if not p["out"] else between(d, r, TW.ease(u, "out"), size)
    return M, zoom_of(M) / zd


def inside_alpha(p, u):
    """How much of the cup shows the picture under the table `u` of the way through punch `p`: it dissolves in over the
    first `fade` of an in punch's window and out over the last `fade` of an out one's."""
    if p["fade"] <= 0.0:
        return 1.0
    k = u / p["fade"] if not p["out"] else (1.0 - u) / p["fade"]
    return float(TW.ease(TW.clamp01(k), "inout"))


# ================================================================================================= what floats
def touch_place(touch, pads):
    """Where a touch lands (table fractions): its `at`, or the middle of the first pad."""
    return tuple(touch["at"]) if touch["at"] is not None else tuple(pads[0]["at"])


def touch_pad(touch, pads, aspect):
    """The pad a touch lands on (an index), the one whose middle is nearest its place if it lies within that pad's
    reach, else None; the first pad for a touch without `at`."""
    if touch["at"] is None:
        return 0
    best, near = None, math.inf
    for i, p in enumerate(pads or ()):
        d = math.hypot((p["at"][0] - touch["at"][0]) * aspect, p["at"][1] - touch["at"][1])
        if d <= 0.6 * p["size"] and d < near:
            best, near = i, d
    return best


def dip_at(touches, pads, i, t, aspect):
    """How far pad `i` is pushed into the coffee at clip second `t` (a share of its size; under 0 it rides up): each
    touch landing on it from its second, a landing's spring (mkmmd.core.sheet.spring): down at once, deepest a tenth of
    a second later, a bob back up past rest and down, nothing left by its `settle`."""
    from .sheet import spring
    return sum(spring(t - k["t"], k["dip"], k["settle"]) for k in touches or () if touch_pad(k, pads, aspect) == i)


def ring_starts(touches, pads, aspect):
    """The radius (table heights) each touch's rings start from: the rim of the pad it lands on (it pushes the water out
    from under the pad), its own place when it lands on none."""
    out = []
    for k in touches or ():
        i = touch_pad(k, pads, aspect)
        out.append(0.0 if i is None else 0.5 * pads[i]["size"])
    return out


def drop_origins(places, centre, start, aspect):
    """Where each drop leaves from (table fractions; None for one that found no room): `start` table heights from the
    touch's `centre` toward where it lands (`places`), so drops thrown out from under a pad leave from its rim, where the
    water is pushed out (ring_starts' radius), never across what lies on it."""
    out = []
    for p in places:
        if p is None:
            out.append(None)
            continue
        dx, dy = (p[0] - centre[0]) * aspect, p[1] - centre[1]
        k = min(start / max(math.hypot(dx, dy), 1e-12), 1.0)
        out.append((centre[0] + dx * k / aspect, centre[1] + dy * k))
    return out


def ring_fronts(touches, rings, t, places, starts=None):
    """The ring fronts on the coffee at clip second `t`: [(centre (x, y) table fractions, radius table heights,
    alpha)], `count` from each touch (`places`: where each landed; `starts`: the radius each starts from, ring_starts)
    `gap` seconds apart, running out at `speed`, coming up over RING_IN and fading as (1 - age / life)^2, each later
    one RING_FALL fainter."""
    out = []
    for i, (k, c) in enumerate(zip(touches or (), places)):
        r0 = starts[i] if starts else 0.0
        for j in range(rings["count"] if rings else 0):
            age = t - k["t"] - j * rings["gap"]
            if 0.0 < age < rings["life"]:
                a = min(age / RING_IN, 1.0) * (1.0 - age / rings["life"]) ** 2 * max(1.0 - RING_FALL * j, 0.1)
                out.append((c, r0 + rings["speed"] * age, a))
    return out


def pad_bob(pad, fronts, rings, aspect):
    """A pad's bob from the ring fronts that have passed under its middle (a share of its size): each sets it rising
    and falling at BOB's rate, dying away."""
    out = 0.0
    for c, r, a in fronts:
        d = math.hypot((pad["at"][0] - c[0]) * aspect, pad["at"][1] - c[1])
        since = (r - d) / rings["speed"] if rings else -1.0
        if 0.0 < since < BOB[2]:
            out += BOB[0] * a * math.sin(2.0 * math.pi * BOB[1] * since) * (1.0 - since / BOB[2])
    return out


def lift_at(fx, t):
    """The figure's lift at clip second `t` (0 lying on the table, its shadow close under it; 1 a sticker's height up)."""
    keys = fx.get("lift")
    if not keys:
        return 0.0
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def track_map(track, p, m, height, table, turn=0.0):
    """The 2 x 3 map (a picture's px -> table px) of a tracked picture (a figure's plate, or the cut's own frame in the
    cup) on a frame where its world point is seen at px `p`, `m` picture heights a metre at its depth, in a picture
    `height` px tall: the point goes to the table's `at`, a metre there is `metre` table heights, the picture turned
    `turn` radians counter-clockwise on the table."""
    return similarity((track["at"][0] * table[0] - 0.5, track["at"][1] * table[1] - 0.5),
                      track["metre"] * table[1] / (m * height), turn, p)
