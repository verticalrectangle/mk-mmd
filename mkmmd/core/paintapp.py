"""The toy paint app, bpy-free (docs/design.md: Frame effects, Toy paint app): the `[[fx]]` kind `paintapp` checked,
timed and laid out here; mkmmd/paintapp.py draws it.

A children's paint program "editing" the video live over the finished pictures: its chrome (a canvas border, a chunky
toolbar of six tools and a palette strip, laid out per output), a cursor (a hand or a pencil) gliding between keyed
points and the places it acts on and clicking there with a squash, and its tools:

    stamps    on each hit a sprite (registry slugs, files, globs or tags; googly `eyes` too) is stamped where the cursor
              clicks, popping in with an overshoot and a burst of sparkles; it stays, fades after its `life`, scrolls
    bucket    on each hit the picture's colour region round a point is flooded with a colour that spreads out from the
              point; the region is found again on every frame, so the paint keeps to it as the picture moves
    eraser    a transition across a cut: the outgoing picture, held with its paint, is rubbed out stroke by stroke
              (`wipe`) or blown into confetti (`confetti`), revealing the incoming shot under it
    undo      the last stamp or fill still on the canvas is taken back, its pop or its spread played in reverse

On its way the cursor picks a tool on the toolbar before it uses it, and a colour on the strip before a fill, when there
is time. Lengths are shares of the frame's short side and points [x, y] fractions of the frame (0, 0 the top left)
unless said otherwise; times are clip seconds. Everything is a function of the clip second, the output's size and the
seed, so a frame comes out the same in any order, in mk look and mk post. Flash-safe: a fill spreads over at least
MIN_GROW s, an eraser takes at least MIN_ERASE s, and no more than FLASHES fills landing or leaving and erasers starting
fall in any second.
"""
import math

import numpy as np

from . import eyes as EY
from . import lanes as LN
from . import mandala as MD
from . import tween as TW

KINDS = ("paintapp",)
TOOLS = ("pencil", "crayon", "stamp", "bucket", "eraser", "undo")    # the toolbar, in order
SIDES = ("left", "right", "top", "bottom")      # where the toolbar stands
CURSORS = ("hand", "pencil")
MODES = ("confetti", "wipe")                    # how an eraser takes the picture away
SWATCHES = ("love", "rose", "gold", "pine", "foam", "iris", "text", "base")   # the palette strip by default
FILLS = ("gold", "love", "foam", "iris", "rose", "pine")                       # a bucket's colours, fill by fill
ACCENTS = ("love", "gold", "rose", "pine", "foam", "iris")                     # sparkles, the backs of the confetti
ART = ("base", "surface", "text", "subtle", "hl_high") + ACCENTS                # the slots the app's art is drawn in
TOOL_OF = {"stamp": "stamp", "fill": "bucket", "erase": "eraser"}               # the tool an action needs in hand
HITS = ("times", "lane", "min_db", "every")
MAX_SWATCHES = 12
CURSOR = {"kind": "hand", "size": 0.1, "keys": None, "travel": 0.3}
STAMPS = {"sprites": None, "size": 0.16, "at": None, "angle": 12.0, "life": None, "fade": 0.25, "scroll": [0.0, 0.0],
          "pop": 0.3, "sparkle": True, "edge": {"color": "#ffffff", "width": 8.0},
          "shadow": {"color": "text", "offset": [0.004, 0.008], "blur": 6.0, "alpha": 0.3}}
BUCKET = {"at": None, "color": list(FILLS), "tolerance": 0.1, "grow": 0.3, "life": None, "fade": 0.3, "opacity": 1.0}
ERASER = {"at": None, "dur": 0.5, "mode": "confetti", "center": [0.5, 0.5], "pieces": 14, "width": 0.3}
UNDO = {"back": 0.3}
KEYS = {"paintapp": {"toolbar", "scale", "border", "panel", "ink", "swatches", "ui", "cursor", "stamps", "bucket",
                     "eraser", "undo", "seed"}}
DEFAULTS = {"paintapp": {"toolbar": "auto", "scale": 1.0, "border": "gold:base:0.45", "panel": "surface", "ink": "text",
                         "swatches": list(SWATCHES), "ui": True, "cursor": {}, "stamps": None, "bucket": None,
                         "eraser": None, "undo": None, "seed": 1}}
# the chrome, px at 1080 on the short side (times `scale`): the canvas's margin where no panel stands, the gap round a
# panel, a button's side, the room between buttons and round them in their panel, a swatch's width and the room between
# and round them, the radius of the canvas's corners and its outline's width
UI = {"margin": 22.0, "gap": 16.0, "button": 104.0, "spacing": 12.0, "pad": 14.0, "swatch": 58.0, "swatch_gap": 14.0,
      "strip_pad": 12.0, "corner": 30.0, "outline": 7.0}
TALL = 1.2                       # a frame this much taller than wide stands its toolbar at the bottom by default
HOME = (0.8, 0.75)               # where the cursor rests at the span's start, canvas fractions
ARC = 0.12                       # how far the cursor's way bows out of the straight line, shares of its length
PICK = 0.7                       # seconds the cursor needs between two clicks to pick a tool or a colour on its way
PRESS, RELEASE, BOUNCE = 0.05, 0.22, 0.3   # a click: pressed down over PRESS s, springing back over RELEASE s, past rest
RIPPLE = 0.35                    # seconds a click's ring spreads and fades
POP_OVER = 0.15                  # a stamp's pop overshoots its size by this share
SETTLE = 18.0                    # degrees a stamp turns as it lands, settling to its angle
SPARKLE, SPARKLES = 0.45, 6      # seconds a stamp's sparkles fly, and how many
MIN_GROW, MIN_ERASE = 0.1, 0.2   # seconds: the fastest a fill may spread and an eraser take the picture (flash-safe)
FLASHES = 3                      # the most fills landing or leaving and erasers starting in any second (flash-safe)
JITTER = 0.28                    # how far a confetti cut's inner corner strays, shares of a cell
SPLIT = 0.35                     # the share of the confetti's cells cut in two triangles
WAVE, LAG = 0.42, 0.06           # the confetti's pieces leave over the first WAVE of the eraser, nearest the centre first
LIFT = 0.18                      # a piece grows this much as it lifts off
UP, GRAVITY = 0.55, 2.6          # a piece's kick up (shares of its speed) and the fall (short sides per dur squared)
SHRINK = 0.38                    # the pieces shrink away over the last SHRINK of the eraser
OVERLAP = 0.7                    # a wipe's strokes lie this share of its width apart
SCRUB = (0.6, 0.13)              # and scrub up and down as they go: a wiggle this many widths long, this many high
_POP_C = TW.back_coeff(POP_OVER)
_R2 = (0.7548776662466927, 0.5698402909980532)   # the plastic constant's sequence: stamps spread without bunching


def _pop(u):
    """0 -> 1 eased out, overshooting POP_OVER before it settles (TW.pop with its coefficient worked out once)."""
    u = TW.clamp01(u)
    return 1.0 + (_POP_C + 1.0) * (u - 1.0) ** 3 + _POP_C * (u - 1.0) ** 2


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """The paint app's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    side = out["toolbar"]
    if side != "auto" and side not in SIDES:
        raise FX.FxError(f"{what}: toolbar {side!r} is none of auto, {', '.join(SIDES)} (the side it stands on)")
    out["scale"] = FX._num(out["scale"], f"{what} scale", 0.25, 4.0)
    for k in ("border", "panel", "ink"):
        out[k] = FX._colour(out[k], palette, f"{what} {k}")
    out["swatches"] = FX._colours(out["swatches"], palette, f"{what} swatches")
    if len(out["swatches"]) > MAX_SWATCHES:
        raise FX.FxError(f"{what}: swatches: at most {MAX_SWATCHES} colours on the strip")
    if not isinstance(out["ui"], bool):
        raise FX.FxError(f"{what}: ui is true or false (false: the cursor and the paint without the app's chrome)")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["art"] = {s: FX._colour(s, palette, f"{what} art") for s in ART}
    out["cursor"] = _cursor(FX, out["cursor"], what)
    out["stamps"] = None if out["stamps"] is None else _stamps(FX, out["stamps"], palette, what)
    out["bucket"] = None if out["bucket"] is None else _bucket(FX, out["bucket"], palette, what)
    out["eraser"] = _erasers(FX, out["eraser"], out, what)
    out["undo"] = None if out["undo"] is None else _undo(FX, out["undo"], what)
    if out["undo"] is not None and out["stamps"] is None and out["bucket"] is None:
        raise FX.FxError(f"{what}: undo takes back stamps and fills: give `stamps` or `bucket`")


def _table(FX, v, known, what):
    if not isinstance(v, dict):
        raise FX.FxError(f"{what} = {v!r}: expected a table of {sorted(known)}")
    extra = sorted(set(v) - set(known))
    if extra:
        raise FX.FxError(f"{what}: unknown keys {extra} (known: {sorted(known)})")


def _bool(FX, v, what):
    if not isinstance(v, bool):
        raise FX.FxError(f"{what} is true or false")
    return v


def _life(FX, v, what):
    return None if v is None else FX._num(v, what, lo=1e-3)


def _when(FX, v, what):
    """A tool's hits: `times` (clip seconds) or a `lane` of the [audio] hits file (`min_db`, `every`), as a zoom's."""
    out = {"times": None, "lane": None, "min_db": None, "every": 1}
    FX._hits(v, out, what)
    return out


def _spots(FX, v, what, points=False):
    """One point or a list of them: [x, y] frame fractions, or with `points` {point = "<expression>"} too -> a list of
    [x, y] (or of FX._center's {mode, ...} with `points`)."""
    one = isinstance(v, dict) or (isinstance(v, (list, tuple)) and len(v) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))
    items = [v] if one else v
    if not isinstance(items, (list, tuple)) or not items:
        kinds = '[x, y] or {point = "<expression>"}' if points else "[x, y]"
        raise FX.FxError(f"{what} = {v!r}: expected {kinds} (frame fractions), or a list of them")
    if points:
        return [FX._center(p, what) for p in items]
    return [FX._pair(p, what) for p in items]


def _cursor(FX, v, what):
    """`cursor`: false (none), true or {kind, size, keys, travel} -> CURSOR's keys filled in, its `keys` [{t, at, click}]
    sorted by `t`; None for none."""
    if v is False or v is None:
        return None
    w = f"{what} cursor"
    v = {} if v is True else v
    _table(FX, v, CURSOR, w)
    c = {**CURSOR, **v}
    if c["kind"] not in CURSORS:
        raise FX.FxError(f"{w}: kind {c['kind']!r} is none of {', '.join(CURSORS)}")
    out = {"kind": c["kind"], "size": FX._num(c["size"], f"{w} size", 0.02, 0.5),
           "travel": FX._num(c["travel"], f"{w} travel", 0.05, 2.0), "keys": []}
    keys = c["keys"] or []
    if not isinstance(keys, (list, tuple)) or not all(isinstance(k, dict) for k in keys):
        raise FX.FxError(f"{w}: keys = {keys!r}: expected [{{t, at, click}}, ...]")
    for k in keys:
        _table(FX, k, ("t", "at", "click"), f"{w} keys")
        if "t" not in k or "at" not in k:
            raise FX.FxError(f"{w} keys: {k!r}: `t` (clip seconds) and `at` ([x, y] frame fractions) are required")
        out["keys"].append({"t": FX._num(k["t"], f"{w} keys t"), "at": FX._pair(k["at"], f"{w} keys at"),
                            "click": _bool(FX, k.get("click", False), f"{w} keys click")})
    out["keys"].sort(key=lambda k: k["t"])
    if any(a["t"] == b["t"] for a, b in zip(out["keys"], out["keys"][1:])):
        raise FX.FxError(f"{w} keys: two keys at the same second")
    return out


def _stamps(FX, v, palette, what):
    """`stamps`: the sprites stamped on each hit, where, how large, how they land and how long they stay."""
    w = f"{what} stamps"
    _table(FX, v, set(STAMPS) | set(HITS) | set(EY.KEYS), w)
    s = {**STAMPS, **v}
    if s["sprites"] is None:
        raise FX.FxError(f"{w}: `sprites` is required: registry slugs, files, globs or {{tag = ...}}, one or a list")
    out = {"refs": MD._refs(FX, s["sprites"], w),
           "size": FX._num(s["size"], f"{w} size", 0.01, 1.0),
           "at": None if s["at"] is None else _spots(FX, s["at"], f"{w} at"),
           "angle": FX._num(s["angle"], f"{w} angle", 0.0, 180.0),
           "life": _life(FX, s["life"], f"{w} life"),
           "fade": FX._num(s["fade"], f"{w} fade", lo=0.0),
           "scroll": FX._pair(s["scroll"], f"{w} scroll"),
           "pop": FX._num(s["pop"], f"{w} pop", 0.0, 2.0),
           "sparkle": _bool(FX, s["sparkle"], f"{w} sparkle"),
           "edge": MD._edge(FX, None if s["edge"] is False else s["edge"], palette, w),
           "shadow": None if s["shadow"] is None or s["shadow"] is False else FX._shadow(s["shadow"], palette, w),
           "face": EY.face(v, w)}
    out.update(_when(FX, v, w))
    return out


def _bucket(FX, v, palette, what):
    """`bucket`: on each hit the colour region round a point flooded, spreading out from it."""
    w = f"{what} bucket"
    _table(FX, v, set(BUCKET) | set(HITS), w)
    b = {**BUCKET, **v}
    if b["at"] is None:
        raise FX.FxError(f"{w}: `at` is required: the point a fill floods round, [x, y] of the frame or "
                         f"{{point = \"<expression>\"}}, or a list of them used in turn")
    out = {"at": _spots(FX, b["at"], f"{w} at", points=True),
           "color": FX._colours(b["color"], palette, f"{w} color"),
           "tolerance": FX._num(b["tolerance"], f"{w} tolerance", 0.005, 1.0),
           "grow": FX._num(b["grow"], f"{w} grow", MIN_GROW, 10.0),
           "life": _life(FX, b["life"], f"{w} life"),
           "fade": FX._num(b["fade"], f"{w} fade", lo=0.0),
           "opacity": FX._num(b["opacity"], f"{w} opacity", 0.0, 1.0)}
    out.update(_when(FX, v, w))
    return out


def _erasers(FX, v, fx, what):
    """`eraser`: a table or a list of them, each {at, dur, mode, center, pieces, width} -> sorted by `at`, none
    overlapping, each inside the span."""
    if v is None:
        return []
    items = [v] if isinstance(v, dict) else v
    if not isinstance(items, (list, tuple)) or not items or not all(isinstance(e, dict) for e in items):
        raise FX.FxError(f"{what}: eraser = {v!r}: expected {{at, dur, mode, ...}} or a list of them")
    out = []
    for e in items:
        w = f"{what} eraser"
        _table(FX, e, ERASER, w)
        e = {**ERASER, **e}
        if e["at"] is None:
            raise FX.FxError(f"{w}: `at` is required: the clip second of the cut it reveals")
        if e["mode"] not in MODES:
            raise FX.FxError(f"{w}: mode {e['mode']!r} is none of {', '.join(MODES)}")
        n = {"at": FX._num(e["at"], f"{w} at"), "dur": FX._num(e["dur"], f"{w} dur", MIN_ERASE, 10.0),
             "mode": e["mode"], "center": FX._pair(e["center"], f"{w} center"),
             "pieces": FX._num(e["pieces"], f"{w} pieces", 3, 40, integer=True),
             "width": FX._num(e["width"], f"{w} width", 0.05, 1.0)}
        if not (fx["from"] <= n["at"] and n["at"] + n["dur"] <= fx["to"] + 1e-9):
            raise FX.FxError(f"{w}: {n['at']:g} .. {n['at'] + n['dur']:g} s runs outside the span "
                             f"({fx['from']:g} .. {fx['to']:g})")
        out.append(n)
    out.sort(key=lambda e: e["at"])
    for a, b in zip(out, out[1:]):
        if a["at"] + a["dur"] > b["at"] + 1e-9:
            raise FX.FxError(f"{what}: the erasers at {a['at']:g} and {b['at']:g} s overlap")
    return out


def _undo(FX, v, what):
    w = f"{what} undo"
    _table(FX, v, set(UNDO) | set(HITS), w)
    out = {"back": FX._num(v.get("back", UNDO["back"]), f"{w} back", 0.05, 5.0)}
    out.update(_when(FX, v, w))
    return out


# ================================================================================================= what it reads
def timed(fx, fps, frame0):
    """Each eraser's frames worked out at `fps` from `frame0`, in place: its first `frame` and how many, `frames` (at
    least 2). They lie inside the span, and the frame before the first, the picture it holds, inside the clip."""
    from . import framefx as FX
    for e in fx["eraser"]:
        e["frame"] = FX.frame_of(e["at"], fps, frame0)
        e["frames"] = max(int(round(e["dur"] * fps)), 2)
        if e["frame"] - 1 < frame0:
            raise FX.FxError(f"fx {fx['index']} (paintapp): the eraser at {e['at']:g} s has no frame before it to hold")
        last = e["frame"] + e["frames"] - 1
        if e["frame"] < fx["f0"] or last >= fx["f1"]:
            raise FX.FxError(f"fx {fx['index']} (paintapp): the eraser's frames {e['frame']}..{last} run outside the "
                             f"span ({fx['f0']}..{fx['f1'] - 1})")


def eraser_at(fx, frame):
    """The eraser at work on `frame` (an entry of fx["eraser"]) or None."""
    for e in fx["eraser"]:
        if e["frame"] <= frame < e["frame"] + e["frames"]:
            return e
    return None


def sources(fx, frame):
    """The frames of the cut a paint app reads at `frame`: the frame before an eraser at work, the picture it holds."""
    e = eraser_at(fx, frame)
    return [] if e is None else [e["frame"] - 1]


def holds(fx):
    """Whether it holds a frame of the cut from before as it reached it (an eraser's)."""
    return bool(fx["eraser"])


def points(fx, key):
    """[(key, expression)]: the scene points it reads, its bucket's {point} seeds (`<key>b<j>`)."""
    if fx["bucket"] is None:
        return []
    return [(f"{key}b{j}", p["expr"]) for j, p in enumerate(fx["bucket"]["at"]) if p["mode"] == "point"]


def tool_times(fx, hits):
    """{"stamps", "bucket", "undo": [clip seconds]}: each tool's `times`, or its lane's hits inside the span from `hits`
    (the [audio] hits file, {lane: [...]}; None: a lane is an FxError that says to write it)."""
    from . import framefx as FX
    out = {}
    for name in ("stamps", "bucket", "undo"):
        tool = fx[name]
        if tool is None:
            out[name] = []
        elif tool["times"] is not None:
            out[name] = [t for t in tool["times"] if fx["from"] <= t < fx["to"]]
        else:
            if hits is None:
                raise FX.FxError(f"fx {fx.get('index')} (paintapp) {name}: lane {tool['lane']!r} needs the [audio] "
                                 f"hits file (`mk timeline hits` writes it)")
            try:
                out[name] = LN.pick(hits, tool["lane"], tool["min_db"], tool["every"], fx["from"], fx["to"])
            except LN.LaneError as e:
                raise FX.FxError(f"fx {fx.get('index')} (paintapp) {name}: {e}") from None
    return out


# ================================================================================================= what happens
def history(fx, times):
    """What the app does, from its tools' clip seconds (`times`, tool_times'; the erasers' are their own): {items,
    actions}. `items`, the stamps and fills in the order they land, each {kind ("stamp" or "fill"), j (its number), t,
    end, how}: on the canvas from `t`; `end` the clip second it leaves (None: it stays), `how` "undo" (taken back in
    reverse from `end`), "erase" (rubbed out with the picture by the eraser that starts at `end`) or "fade" (its `life`
    over: gone at `end`, fading out before). An undo takes back the last stamp or fill still on the canvas (none: it
    clicks for nothing). `actions`, every click a tool makes, {t, kind ("stamp", "fill", "erase", "undo"), j}, in time
    order (at one second an eraser first, then what lands, then an undo)."""
    ev = [(e["at"], 0, "erase", k) for k, e in enumerate(fx["eraser"])]
    for name, kind in (("stamps", "stamp"), ("bucket", "fill"), ("undo", "undo")):
        ev += [(t, 2 if kind == "undo" else 1, kind, j) for j, t in enumerate(times.get(name) or [])]
    ev.sort()
    items, on = [], []
    for t, _, kind, j in ev:
        for it in [x for x in on if x["life"] is not None and x["life"] <= t]:
            it.update(end=it["life"], how="fade")
            on.remove(it)
        if kind == "erase":
            for it in on:
                it.update(end=t, how="erase")
            on = []
        elif kind == "undo":
            if on:
                on.pop().update(end=t, how="undo")
        else:
            life = fx["stamps" if kind == "stamp" else "bucket"]["life"]
            it = {"kind": kind, "j": j, "t": t, "end": None, "how": None, "life": None if life is None else t + life}
            items.append(it)
            on.append(it)
    for it in on:
        if it["life"] is not None:
            it.update(end=it["life"], how="fade")
    for it in items:
        del it["life"]
    return {"items": items, "actions": [{"t": t, "kind": kind, "j": j} for t, _, kind, j in ev]}


def swatch_of(fx, colour):
    """The swatch of the strip that is `colour` (rgb), or None."""
    for i, c in enumerate(fx["swatches"]):
        if all(abs(a - b) < 1e-4 for a, b in zip(c, colour)):
            return i
    return None


def fill_colour(fx, j):
    """Fill j's colour: the bucket's `color` list read fill by fill, round and round."""
    cs = fx["bucket"]["color"]
    return cs[j % len(cs)]


def clicks(fx, hist):
    """The cursor's clicks and what the app shows picked: (clicks, tools, picked). `clicks` [{t, on}] in time order, `on`
    ("stamp", j), ("fill", j), ("erase", k), ("undo", j), ("tool", name), ("swatch", i) or ("key", k) (a cursor key with
    `click`; the others are stops it passes without one). Before an action whose tool is not in hand (and with the chrome
    shown) the cursor clicks the tool's button, and before a fill whose colour is a swatch not picked, the swatch, each
    when PICK s since the click before allow it, evenly between them; else the tool or the colour simply changes as the
    action comes. `tools` [(t, tool)] and `picked` [(t, swatch)]: the tool in hand (the pencil at first) and the swatch
    picked (the first) from each time on."""
    tool, swatch = "pencil", 0
    tools, picked, out = [(-math.inf, tool)], [(-math.inf, swatch)], []
    prev = fx["from"]
    for a in hist["actions"]:
        need = TOOL_OF.get(a["kind"])
        want = swatch_of(fx, fill_colour(fx, a["j"])) if a["kind"] == "fill" else None
        picks = []
        if fx["ui"] and need is not None and need != tool:
            picks.append(("tool", need))
        if fx["ui"] and want is not None and want != swatch:
            picks.append(("swatch", want))
        picks = picks[:max(int((a["t"] - prev) // PICK), 0)]
        for i, p in enumerate(picks):
            tp = prev + (a["t"] - prev) * (i + 1) / (len(picks) + 1)
            out.append({"t": tp, "on": p})
            if p[0] == "tool":
                tool = p[1]
                tools.append((tp, tool))
            else:
                swatch = p[1]
                picked.append((tp, swatch))
        if need is not None and need != tool:
            tool = need
            tools.append((a["t"], tool))
        if want is not None and want != swatch:
            swatch = want
            picked.append((a["t"], swatch))
        out.append({"t": a["t"], "on": (a["kind"], a["j"])})
        prev = a["t"] + (fx["eraser"][a["j"]]["dur"] if a["kind"] == "erase" else 0.0)
    if fx["cursor"] is not None:
        out += [{"t": k["t"], "on": ("key", i)} for i, k in enumerate(fx["cursor"]["keys"])]
    out.sort(key=lambda c: c["t"])
    return out, tools, picked


def in_force(timeline, t):
    """The value of a [(t, value), ...] timeline (in time order) at clip second `t`: the last one from at or before it."""
    v = timeline[0][1]
    for tt, x in timeline:
        if tt > t:
            break
        v = x
    return v


# ================================================================================================= the chrome
def side_of(fx, size):
    """The side the toolbar stands on in a frame of `size` (w, h): `toolbar`, or by default the bottom of a frame TALL
    times taller than wide and the left of any other."""
    if fx["toolbar"] != "auto":
        return fx["toolbar"]
    return "bottom" if size[1] > TALL * size[0] else "left"


def _run(lo, hi, length, count, item, gap, pad):
    """Items along a panel centred in lo..hi: (panel start, panel end, [item starts], item size, scale), the panel
    shrunk to fit."""
    s = min(1.0, (hi - lo) / length) if length > 0 else 1.0
    a = 0.5 * (lo + hi) - 0.5 * length * s
    return a, a + length * s, [a + (pad + i * (item + gap)) * s for i in range(count)], item * s, s


def layout(fx, size):
    """The app's chrome in a frame of `size` (w, h): {side, unit (px per px at 1080 on the short side, times `scale`),
    short, size, canvas (x0, y0, x1, y1), corner and outline (the canvas's corners' radius and its line's width), toolbar
    (its panel's box), buttons [(tool, box)], strip (the palette panel's box), swatches [(cx, cy, r)], home (where the
    cursor rests at first)}. Boxes are px from the frame's top left corner (its edges at 0 and w, 0 and h); points (the
    swatches' middles, home) are px with pixel middles at whole numbers, as every point of the effects is. The toolbar
    stands on its side, the palette strip along the bottom (beside the toolbar when it stands at the top or the bottom),
    a panel too long for its side shrunk to fit; the canvas is what they leave inside a border all round."""
    w, h = float(size[0]), float(size[1])
    k = min(w, h) / 1080.0 * fx["scale"]
    B, S, P, G, M = (UI[n] * k for n in ("button", "spacing", "pad", "gap", "margin"))
    D, DG, DP = (UI[n] * k for n in ("swatch", "swatch_gap", "strip_pad"))
    T, st = B + 2.0 * P, D + 2.0 * DP
    side = side_of(fx, size)
    if side in ("left", "right"):
        tb = (G, G + T) if side == "left" else (w - G - T, w - G)
        sp = (h - G - st, h - G)
        canvas = (G + T + G if side == "left" else M, M, w - M if side == "left" else w - G - T - G, sp[0] - G)
        tb_along, sp_along = (canvas[1], canvas[3]), (canvas[0], canvas[2])
    else:
        if side == "bottom":
            tb = (h - G - T, h - G)
            sp = (tb[0] - G - st, tb[0] - G)
            canvas = (M, M, w - M, sp[0] - G)
        else:
            tb = (G, G + T)
            sp = (tb[1] + G, tb[1] + G + st)
            canvas = (M, sp[1] + G, w - M, h - M)
        tb_along = sp_along = (canvas[0], canvas[2])
    n = len(fx["swatches"])
    a0, a1, starts, b, s = _run(tb_along[0], tb_along[1], 6 * B + 5 * S + 2 * P, 6, B, S, P)
    mid = 0.5 * (tb[0] + tb[1])
    across = (mid - 0.5 * T * s, mid + 0.5 * T * s)
    c0, c1 = mid - 0.5 * b, mid + 0.5 * b
    vertical = side in ("left", "right")
    toolbar = (across[0], a0, across[1], a1) if vertical else (a0, across[0], a1, across[1])
    buttons = [(tool, (c0, y, c1, y + b) if vertical else (y, c0, y + b, c1)) for tool, y in zip(TOOLS, starts)]
    q0, q1, sw, d, ss = _run(sp_along[0], sp_along[1], n * D + (n - 1) * DG + 2.0 * DP, n, D, DG, DP)
    smid = 0.5 * (sp[0] + sp[1])
    strip = (q0, smid - 0.5 * st * ss, q1, smid + 0.5 * st * ss)
    swatches = [(x + 0.5 * d - 0.5, smid - 0.5, 0.5 * d) for x in sw]
    x0, y0, x1, y1 = canvas
    return {"side": side, "unit": k, "short": min(w, h), "size": (w, h), "canvas": canvas,
            "corner": UI["corner"] * k, "outline": UI["outline"] * k, "toolbar": toolbar, "buttons": buttons,
            "strip": strip, "swatches": swatches,
            "home": (x0 + HOME[0] * (x1 - x0) - 0.5, y0 + HOME[1] * (y1 - y0) - 0.5)}


def button_of(lay, tool):
    """The middle (x, y) px of `tool`'s button (pixel middles at whole numbers)."""
    for t, (x0, y0, x1, y1) in lay["buttons"]:
        if t == tool:
            return 0.5 * (x0 + x1) - 0.5, 0.5 * (y0 + y1) - 0.5
    raise KeyError(tool)


# ================================================================================================= the cursor
def _arc(a, b, u, bend):
    """The point `u` of the way from `a` to `b` along a quadratic curve bowed `bend` of its length to one side."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    cx, cy = 0.5 * (ax + bx) - bend * dy, 0.5 * (ay + by) + bend * dx
    v = 1.0 - u
    return v * v * ax + 2.0 * v * u * cx + u * u * bx, v * v * ay + 2.0 * v * u * cy + u * u * by


def glide(stops, t, travel):
    """The cursor's place at clip second `t` going from stop to stop ([(t, (x, y)), ...] in time order): it waits at a
    stop until `travel` s (at most 0.8 of the time between them) before the next, then glides there on a gentle arc,
    eased in and out, arriving on time, each leg bowing the other way; before the first stop it is at it, after the last
    it stays. None without stops."""
    if not stops:
        return None
    if t <= stops[0][0]:
        return stops[0][1]
    for i in range(len(stops) - 1):
        (ta, pa), (tb, pb) = stops[i], stops[i + 1]
        if t < tb:
            d = min(travel, 0.8 * (tb - ta))
            if d <= 0.0 or t <= tb - d:
                return pa
            return _arc(pa, pb, TW.ease((t - (tb - d)) / d, "inout"), ARC if i % 2 == 0 else -ARC)
    return stops[-1][1]


def press(times, t):
    """How far the cursor is pressed down at clip second `t` by its clicks at `times`: 0 at rest, 1 fully, negative
    springing back past rest. It presses down over PRESS s before a click and springs back over RELEASE s after it."""
    p = 0.0
    for c in times:
        a = t - c
        if -PRESS <= a < 0.0:
            q = TW.ease((a + PRESS) / PRESS, "out")
        elif 0.0 <= a < RELEASE:
            q = 1.0 - TW.pop(a / RELEASE, BOUNCE)
        else:
            continue
        if abs(q) > abs(p):
            p = q
    return p


def ripples(times, t):
    """The rings of the clicks at `times` spreading at clip second `t`: [0..1 of the way through each]."""
    return [(t - c) / RIPPLE for c in times if 0.0 <= t - c < RIPPLE]


# ================================================================================================= stamps and fills
def stamp_spot(fx, lay, j):
    """Where stamp j lands, px: entry j of `at` round and round, or spread over the canvas (the plastic constant's
    low-discrepancy sequence from the seed: the stamps never bunch), at least half a stamp inside it."""
    st, (w, h) = fx["stamps"], lay["size"]
    if st["at"] is not None:
        x, y = st["at"][j % len(st["at"])]
        return x * w - 0.5, y * h - 0.5
    x0, y0, x1, y1 = lay["canvas"]
    r = 0.5 * st["size"] * lay["short"]
    xa, xb = (x0 + r, x1 - r) if x1 - x0 > 2.0 * r else (0.5 * (x0 + x1),) * 2
    ya, yb = (y0 + r, y1 - r) if y1 - y0 > 2.0 * r else (0.5 * (y0 + y1),) * 2
    n = j + 1 + 29 * fx["seed"]
    u, v = (0.5 + _R2[0] * n) % 1.0, (0.5 + _R2[1] * n) % 1.0
    return xa + u * (xb - xa) - 0.5, ya + v * (yb - ya) - 0.5


def stamp_angle(fx, j):
    """Stamp j's angle, degrees counter-clockwise: within `angle` either way (seeded)."""
    rng = np.random.default_rng([fx["seed"], fx["index"], 3, j])
    return float(rng.uniform(-1.0, 1.0)) * fx["stamps"]["angle"]


def stamp_pick(fx, j, n):
    """Which of `n` sprites stamp j shows: dealt shuffled (by the seed), every one before any comes again."""
    rng = np.random.default_rng([fx["seed"], fx["index"], 5, j // n])
    return int(rng.permutation(n)[j % n])


def stamp_state(fx, it, t):
    """(scale, turn, alpha, age) of a stamp at clip second `t` on the live canvas: its pop (from nothing, overshooting
    POP_OVER), the turn it settles from (degrees), its opacity as it fades, the seconds since it landed; None when it is
    not on it (before it lands, after it is erased or has faded, once it is taken back: its pop played in reverse)."""
    st = fx["stamps"]
    age = t - it["t"]
    if age < 0.0:
        return None
    k = 1.0
    if it["end"] is not None and t >= it["end"]:
        if it["how"] != "undo":
            return None
        b = (t - it["end"]) / fx["undo"]["back"]
        if b >= 1.0:
            return None
        k = _pop(1.0 - b)
    u = age / st["pop"] if st["pop"] > 0.0 else 1.0
    alpha = 1.0
    if it["how"] == "fade" and st["fade"] > 0.0:
        alpha = TW.clamp01((it["end"] - t) / st["fade"])
    return _pop(u) * k, SETTLE * (1.0 - TW.ease(u, "out")), alpha, age


def fill_state(fx, it, t):
    """(spread, alpha) of a fill at clip second `t`: how far out it has spread (0 its point .. 1 its whole region),
    eased out over `grow`, and its opacity; None when it is not on the canvas (a fill taken back draws back into its
    point over the undo's `back`)."""
    bu = fx["bucket"]
    age = t - it["t"]
    if age < 0.0:
        return None
    spread = TW.ease(age / bu["grow"], "out")
    alpha = bu["opacity"]
    if it["end"] is not None and t >= it["end"]:
        if it["how"] != "undo":
            return None
        b = (t - it["end"]) / fx["undo"]["back"]
        if b >= 1.0:
            return None
        spread *= 1.0 - TW.ease(b, "in")
    elif it["how"] == "fade" and bu["fade"] > 0.0:
        alpha *= TW.clamp01((it["end"] - t) / bu["fade"])
    return spread, alpha


def sparkles(fx, j, age):
    """The sparkles round stamp j `age` s after it lands: [(dx, dy, size, angle, accent, alpha)], the offset from its
    middle and the size in shares of the stamp's size, the angle in degrees, the accent an index into ACCENTS; none
    after SPARKLE."""
    if age < 0.0 or age >= SPARKLE or not fx["stamps"]["sparkle"]:
        return []
    u = age / SPARKLE
    rng = np.random.default_rng([fx["seed"], fx["index"], 11, j])
    a0 = rng.uniform(0.0, 2.0 * math.pi)
    jit, far, big, k0 = (rng.uniform(-0.3, 0.3, SPARKLES), rng.uniform(0.8, 1.2, SPARKLES),
                         rng.uniform(0.75, 1.3, SPARKLES), int(rng.integers(len(ACCENTS))))
    out = []
    for i in range(SPARKLES):
        a = a0 + 2.0 * math.pi * i / SPARKLES + jit[i]
        r = (0.45 + 0.45 * TW.ease(u, "out")) * far[i]
        size = 0.14 * big[i] * math.sin(math.pi * min(1.0, 0.15 + u))
        out.append((r * math.cos(a), -r * math.sin(a), size, 90.0 * u, (k0 + i) % len(ACCENTS),
                    1.0 - TW.ease(u, "in")))
    return out


# ================================================================================================= the eraser
def wipe_path(size, width):
    """The way an eraser `width` px wide rubs out a frame of `size` (w, h): strokes along its long side, back and
    forth from edge to edge, each scrubbing up and down as it goes (SCRUB), OVERLAP of the width apart and joined by a
    half turn past the edge, so that its round end covers the frame's edges and it leaves nothing: (n, 2) px points
    (pixel middles at whole numbers)."""
    w, h = float(size[0]), float(size[1])
    across = w >= h
    L, S = (w, h) if across else (h, w)
    n = max(int(math.ceil(S / (OVERLAP * width))), 1)
    gap, wave, amp = S / n, SCRUB[0] * width, SCRUB[1] * width
    xs = np.linspace(0.0, L, max(int(math.ceil(8.0 * L / wave)), 2))
    pts = []
    for i in range(n):
        line = np.stack([xs, (i + 0.5) * gap + amp * np.sin(2.0 * math.pi * xs / wave + i)], 1)
        line = line if i % 2 == 0 else line[::-1]
        if pts:                                                         # the half turn from the last stroke's end
            a, b = pts[-1][-1], line[0]
            c, r = 0.5 * (a + b), 0.5 * float(np.hypot(*(b - a)))
            out = 1.0 if a[0] > 0.5 * L else -1.0
            th = np.linspace(0.0, math.pi, 12)[1:-1]
            pts.append(np.stack([c[0] + out * r * np.sin(th), c[1] - r * np.cos(th)], 1))
        pts.append(line)
    p = np.concatenate(pts) - 0.5
    return p if across else p[:, ::-1].copy()


def _lengths(path):
    seg = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def along(path, u):
    """The point `u` (0..1) of the way along the polyline `path`, by length."""
    cum = _lengths(path)
    s = TW.clamp01(u) * cum[-1]
    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(path) - 2)
    i = max(i, 0)
    seg = cum[i + 1] - cum[i]
    f = 0.0 if seg <= 0.0 else (s - cum[i]) / seg
    return tuple(path[i] + (path[i + 1] - path[i]) * f)


def upto(path, u):
    """The polyline `path` from its start to `u` (0..1) of the way along it: (m, 2) points, ending on along(path, u)."""
    cum = _lengths(path)
    s = TW.clamp01(u) * cum[-1]
    keep = path[cum < s]
    return np.vstack([keep, np.asarray(along(path, u))[None]]) if len(keep) else np.asarray([path[0]])


def confetti(size, pieces, seed, key=()):
    """The pieces an eraser blows a frame of `size` (w, h) into: a grid of about `pieces` cells along its short side,
    every inner corner strayed JITTER of a cell and every cell a quad or, SPLIT of them, two triangles cut along a
    diagonal (seeded by `seed` and `key`): [(k, 2) polygons], in px from the frame's top left corner (its edges at 0, w
    and 0, h), that tile the frame exactly."""
    w, h = float(size[0]), float(size[1])
    c = min(w, h) / pieces
    nx, ny = max(int(round(w / c)), 1), max(int(round(h / c)), 1)
    V = np.stack(np.meshgrid(np.linspace(0.0, w, nx + 1), np.linspace(0.0, h, ny + 1)), -1)
    rng = np.random.default_rng([seed, 17, *key])
    J = rng.uniform(-JITTER, JITTER, V.shape) * np.array([w / nx, h / ny])
    J[0, :, 1] = J[-1, :, 1] = 0.0
    J[:, 0, 0] = J[:, -1, 0] = 0.0
    V = V + J
    cut, diag = rng.random((ny, nx)) < SPLIT, rng.random((ny, nx)) < 0.5
    out = []
    for j in range(ny):
        for i in range(nx):
            a, b, c2, d = V[j, i], V[j, i + 1], V[j + 1, i + 1], V[j + 1, i]
            if not cut[j, i]:
                out.append(np.array([a, b, c2, d]))
            elif diag[j, i]:
                out += [np.array([a, b, c2]), np.array([a, c2, d])]
            else:
                out += [np.array([a, b, d]), np.array([b, c2, d])]
    return out


def flights(polys, centre, size, dur, seed, key=()):
    """How each confetti piece flies when the eraser blows the frame from `centre` (px) over `dur` s: [{c (its middle,
    px), t0 (s after the eraser starts that it leaves: the nearer the centre the sooner, all within WAVE of the way),
    v (px/s: out from the centre and kicked up), spin (degrees/s), flip (radians/s: it flutters, showing its back),
    back (an index into ACCENTS: its back's colour)}]."""
    short = float(min(size))
    C = np.array([p.mean(0) for p in polys])
    off = C - np.asarray(centre, float)
    d = np.hypot(off[:, 0], off[:, 1])
    far = max(float(d.max()), 1.0)
    rng = np.random.default_rng([seed, 23, *key])
    n = len(polys)
    turn = rng.uniform(-0.4, 0.4, n) + np.where(d < 1e-6, rng.uniform(0.0, 2.0 * math.pi, n), 0.0)
    base = np.arctan2(off[:, 1], np.where(d < 1e-6, 1.0, off[:, 0]))
    ang = base + turn
    speed = rng.uniform(0.9, 1.8, n) * short / dur
    t0 = (WAVE * d / far + rng.uniform(0.0, LAG, n)) * dur
    spin = rng.uniform(180.0, 540.0, n) * rng.choice([-1.0, 1.0], n)
    flip = rng.uniform(5.0, 10.0, n)
    back = rng.integers(len(ACCENTS), size=n)
    return [{"c": (float(C[i, 0]), float(C[i, 1])), "t0": float(t0[i]),
             "v": (float(speed[i] * math.cos(ang[i])), float(speed[i] * math.sin(ang[i]) - UP * speed[i])),
             "spin": float(spin[i]), "flip": float(flip[i]), "back": int(back[i])} for i in range(n)]


def flight_pose(fl, s, u, short, dur):
    """Where a confetti piece is `s` s after it left, `u` (0..1) of the way through the eraser: (middle (x, y) px,
    angle (degrees counter-clockwise), x scale, y scale, back (whether its back shows)), its scales growing LIFT as it
    lifts off, its width fluttering with its flip and both shrinking to nothing over the eraser's last SHRINK; None
    before it leaves or once it is gone."""
    if s < 0.0:
        return None
    g = 1.0 - TW.ease(TW.clamp01((u - (1.0 - SHRINK)) / SHRINK), "in")
    if g <= 0.0:
        return None
    lift = 1.0 + LIFT * TW.ease(min(s / 0.08, 1.0), "out")
    x = fl["c"][0] + fl["v"][0] * s
    y = fl["c"][1] + fl["v"][1] * s + 0.5 * GRAVITY * short / (dur * dur) * s * s
    f = math.cos(fl["flip"] * s)
    return (x, y), fl["spin"] * s, lift * g * f, lift * g, f < 0.0


# ================================================================================================= flash-safe
def flash_check(fx, hist):
    """An FxError when more than FLASHES big changes of the picture fall in any second: fills landing, fills leaving
    (taken back or faded out) and erasers starting (WCAG's general flash threshold, three a second)."""
    from . import framefx as FX
    ts = [e["at"] for e in fx["eraser"]]
    for it in hist["items"]:
        if it["kind"] == "fill":
            ts.append(it["t"])
            if it["how"] in ("undo", "fade"):
                ts.append(it["end"])
    ts.sort()
    for i, t in enumerate(ts):
        n = sum(1 for x in ts[i:] if x < t + 1.0)
        if n > FLASHES:
            raise FX.FxError(f"fx {fx.get('index')} (paintapp): {n} fills and erasers within a second from {t:g} s: "
                             f"at most {FLASHES} big changes of the picture a second (flash-safe)")
