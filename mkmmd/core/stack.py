"""Stacks, bpy-free (docs/design.md: Frame effects, Rainbow stacks): the keys of a `kind = "stack"` collage checked and
its copies laid out in time; mkmmd/stack.py does the picture work.

A plain stack is the figure in front of `count` copies of it, copy j `scale` ** j as large about `center` and j x `step`
further, drawn in `bands` entry step + j - 1 and j x `delay` frames behind (the poses before). Beyond that:

    bands   an entry may be a table {color, sprites, eyes, mouth, size, scroll, tint, line, seed}: that copy is filled
            with a wallpaper of those sprites (googly-eyed with `eyes`), tiled on a ground of `color`, tinted toward it,
            slowly scrolling, a line of it round its rim: eat the rainbow
    looks   the copies drawn in eras' media, read as `bands` is: "halftone" (60s print dots), "flat" (Peter Max's flat
            poster colours in a thick line), "vhs" (scanlines, the colour bled sideways), "pixel" (a chunky mosaic),
            "chrome" (a metal sheen), "none" (as its band draws it)
    recede  {count, scale, fade, speed, snap}: `count` copies shrinking toward `center` into a tunnel that falls inward
            forever, one place deeper on each hit (eased over `snap` s) and `speed` places a second: a new copy comes out
            from behind the figure, the farthest fade out, and nothing jumps
    delay   frames each copy is behind the one before it, or a list: each copy's own (busy clones)
    peel    {at, t, dur, gap, scale, ease}: the copies slide off the stack one after another to their own places
    squash  {times | lane, min_db, every, amount, decay, attack}: on each hit the figure and its copies are pressed down
            and spring back up past their height, about their feet, their area kept; a copy behind in time does it
            when its moment comes, so a squash runs back through the echo

A copy stands `depth` places back: copy j of a plain stack j, a tunnel's copy anywhere between 0 (the figure's place)
and `count` (gone).
"""
import math

from . import eyes as EY
from . import mandala as MD
from . import tween as TW

LOOKS = ("none", "halftone", "flat", "vhs", "pixel", "chrome")
KEYS = ("count", "step", "scale", "center", "bands", "delay", "mirror", "looks", "recede", "peel", "squash")
DEFAULTS = {"count": 5, "step": [-0.02, -0.006], "scale": 1.0, "center": [0.5, 0.5], "bands": ["text", "base"],
            "delay": 0, "mirror": False, "looks": None, "recede": None, "peel": None, "squash": None}
BAND = {"size": 0.1, "scroll": [0.02, -0.04], "tint": 0.3, "line": 6.0, "seed": 1}    # with `color` and `sprites`
RECEDE = {"count": 16, "scale": 0.8, "fade": 0.3, "speed": 0.0, "snap": 0.15}
PEEL = {"t": None, "dur": 0.45, "gap": 0.12, "scale": 1.0, "ease": "back"}          # with `at`
SQUASH = {"amount": 0.2, "decay": 0.15, "attack": 0.03, "min_db": None, "every": 1}   # with `times` or `lane`
REBOUND = 0.9            # a squash's first stretch comes this many `decay`s after its squash, e^-REBOUND as big
PRESS = (-0.5, 0.8)      # how far down (and up, negative) the squashes of hits close together add up to, at most
SETTLED = 8.0            # a squash this many `decay`s old is over


# ================================================================================================= the keys
def normalize(spec, out, palette, what):
    """A stack's own keys checked and filled in, in place (`out` has its defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    out["count"] = FX._num(out["count"], f"{what} count", 1, 32, integer=True)
    out["step"] = FX._pair(out["step"], f"{what} step")
    out["scale"] = FX._num(out["scale"], f"{what} scale", 0.25, 2.0)
    out["center"] = FX._center(out["center"], f"{what} center")
    out["bands"] = _bands(FX, out["bands"], palette, what)
    out["delay"] = _delay(FX, out["delay"], what)
    if not isinstance(out["mirror"], bool):
        raise FX.FxError(f"{what}: mirror is true or false (true: the stack again, reflected about `center`)")
    out["looks"] = _looks(FX, out["looks"], what)
    out["recede"] = _recede(FX, out["recede"], what)
    out["peel"] = _peel(FX, out["peel"], what)
    if out["recede"] is not None and out["peel"] is not None:
        raise FX.FxError(f"{what}: recede and peel: a tunnel's copies stay in it; give one of them")
    out["squash"] = _squash(FX, out["squash"], what)


def _bands(FX, v, palette, what):
    """`bands`: colours, "picture", "silkscreen" or sprite tables, one or a list -> [rgb, None, "silkscreen" or a band
    table, ...]."""
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, (list, tuple)) or not any(isinstance(b, dict) for b in v):
        return FX._colours(v, palette, f"{what} bands", picture=True)
    out = []
    for i, b in enumerate(v):
        if isinstance(b, dict):
            out.append(_band(FX, b, palette, f"{what} bands[{i}]"))
            continue
        got = FX._colours(b, palette, f"{what} bands[{i}]", picture=True)
        if len(got) != 1:
            raise FX.FxError(f"{what} bands[{i}] = {b!r}: a band is a colour, \"picture\", \"silkscreen\" or a table "
                             f"{{color, sprites, ...}}")
        out.append(got[0])
    return out


def _band(FX, b, palette, what):
    """A band of sprites: {color, sprites, eyes, mouth, size, scroll, tint, line, seed} checked, with its `face`."""
    known = {"color", "sprites"} | set(BAND) | set(EY.KEYS)
    extra = sorted(set(b) - known)
    if extra:
        raise FX.FxError(f"{what}: unknown keys {extra} (known: {sorted(known)})")
    for k in ("color", "sprites"):
        if k not in b:
            raise FX.FxError(f"{what}: `{k}` is required (a band of sprites is {{color, sprites, ...}})")
    p = {**BAND, **b}
    return {"color": FX._colour(p["color"], palette, f"{what} color"),
            "sprites": MD._refs(FX, p["sprites"], what),
            "size": FX._num(p["size"], f"{what} size", 0.01, 1.0),
            "scroll": FX._pair(p["scroll"], f"{what} scroll"),
            "tint": FX._num(p["tint"], f"{what} tint", 0.0, 1.0),
            "line": FX._num(p["line"], f"{what} line", lo=0.0),
            "seed": FX._num(p["seed"], f"{what} seed", lo=0, integer=True),
            "face": EY.face(b, what)}


def is_sprites(band):
    """Whether a stack's band (as normalised) is a wallpaper of sprites."""
    return isinstance(band, dict) and "sprites" in band


def _delay(FX, v, what):
    if isinstance(v, (list, tuple)):
        if not v:
            raise FX.FxError(f"{what}: delay = []: frames between copies, or a list of each copy's own")
        return [FX._num(d, f"{what} delay", lo=0, integer=True) for d in v]
    return FX._num(v, f"{what} delay", lo=0, integer=True)


def _looks(FX, v, what):
    if v is None:
        return None
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, (list, tuple)) or not items or any(x not in LOOKS for x in items):
        raise FX.FxError(f"{what}: looks = {v!r}: one of {', '.join(LOOKS)}, or a list of them (read round the copies)")
    return list(items)


def _table(FX, v, name, keys, what):
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table {{{', '.join(keys)}}}")
    extra = sorted(set(v) - set(keys))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {sorted(keys)})")
    return v


def _recede(FX, v, what):
    """`recede`: true or {count, scale, fade, speed, snap}; None (or false): a plain stack."""
    if v is None or v is False:
        return None
    r = {**RECEDE, **_table(FX, v, "recede", RECEDE, what)}
    return {"count": FX._num(r["count"], f"{what} recede count", 2, 32, integer=True),
            "scale": FX._num(r["scale"], f"{what} recede scale", 0.3, 0.98),
            "fade": FX._num(r["fade"], f"{what} recede fade", 0.0, 1.0),
            "speed": FX._num(r["speed"], f"{what} recede speed"),
            "snap": FX._num(r["snap"], f"{what} recede snap", lo=0.0)}


def _peel(FX, v, what):
    """`peel`: {at, t, dur, gap, scale, ease}, `at` one [x, y] or a list of them; None: the copies stay in the stack."""
    if v is None:
        return None
    v = _table(FX, v, "peel", ("at",) + tuple(PEEL), what)
    if "at" not in v:
        raise FX.FxError(f"{what}: peel: `at` is required: where the copies go, [[x, y], ...] (frame fractions, copy j "
                         f"the j-th, round and round)")
    at = v["at"]
    one = isinstance(at, (list, tuple)) and len(at) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in at)
    pts = [at] if one else at
    if not isinstance(pts, (list, tuple)) or not pts:
        raise FX.FxError(f"{what}: peel at = {at!r}: expected [x, y] or [[x, y], ...] (frame fractions)")
    p = {**PEEL, **v}
    scales = p["scale"] if isinstance(p["scale"], (list, tuple)) and p["scale"] else [p["scale"]]
    if p["ease"] not in FX.MOVE_EASES:
        raise FX.FxError(f"{what}: peel ease {p['ease']!r} is none of {', '.join(FX.MOVE_EASES)}")
    return {"at": [FX._pair(q, f"{what} peel at") for q in pts],
            "t": None if p["t"] is None else FX._num(p["t"], f"{what} peel t"),
            "dur": FX._num(p["dur"], f"{what} peel dur", lo=1e-3),
            "gap": FX._num(p["gap"], f"{what} peel gap", lo=0.0),
            "scale": [FX._num(s, f"{what} peel scale", 0.05, 4.0) for s in scales],
            "ease": p["ease"]}


def _squash(FX, v, what):
    """`squash`: {times or lane (min_db, every), amount, decay, attack}; None: none."""
    if v is None:
        return None
    v = _table(FX, v, "squash", ("times", "lane") + tuple(SQUASH), what)
    out = {**SQUASH, **v}
    FX._hits(v, out, f"{what} squash")
    out["amount"] = FX._num(out["amount"], f"{what} squash amount", 0.0, 0.8)
    out["decay"] = FX._num(out["decay"], f"{what} squash decay", lo=1e-3)
    out["attack"] = FX._num(out["attack"], f"{what} squash attack", lo=0.0)
    return out


# ================================================================================================= in time
def delay_of(fx, depth):
    """The frames a copy `depth` places back shows her behind: j x `delay` for copy j (or entry j of a list of them,
    round and round), between two places the share of the way (a tunnel's copies age as they fall), whole frames."""
    d = fx["delay"]
    if not isinstance(d, list):
        return int(round(depth * d))
    lo = int(math.floor(depth))
    at = lambda j: 0 if j <= 0 else d[(j - 1) % len(d)]         # noqa: E731
    return int(round(at(lo) + (at(lo + 1) - at(lo)) * (depth - lo)))


def phase(fx, t, times, fps, frame0):
    """How many places a tunnel has fallen by clip second `t`: one on each of its hits (`times`) after its first frame,
    eased over `snap` s, and `speed` a second since its start."""
    r = fx["recede"]
    return MD.stepped({"f0": fx["f0"], "snap": r["snap"]}, t, times, fps, frame0) + r["speed"] * (t - fx["from"])


def fade(r, depth):
    """A tunnel copy's opacity `depth` places back: whole, then over the last `fade` of the tunnel down to nothing."""
    if depth >= r["count"]:
        return 0.0
    if r["fade"] <= 0.0:
        return 1.0
    return min(max((r["count"] - depth) / (r["fade"] * r["count"]), 0.0), 1.0)


def peeled(fx, j, t):
    """How far copy j has slid off the stack to its place by clip second `t`: 0 in the stack, 1 there (a `back` ease
    overshoots and settles)."""
    p = fx["peel"]
    t0 = (fx["from"] if p["t"] is None else p["t"]) + (j - 1) * p["gap"]
    u = (t - t0) / p["dur"]
    if u <= 0.0:
        return 0.0
    return 1.0 if u >= 1.0 else _ease(u, p["ease"])


def _ease(u, kind):
    from . import framefx as FX
    return FX._ease(u, kind)


def copies(fx, t, k, times, fps, frame0):
    """A stack's copies at clip second `t` (`k` the hits so far, `times` them all), the farthest first: [{j, band,
    depth, delay, alpha, peel}]: j its place (1 the nearest) or a tunnel copy's own number, band its entry of `bands`
    and `looks`, depth the places back it stands, delay the frames it shows her behind, alpha its opacity, peel how far it
    has slid to its own place (None: a stack without a peel). A tunnel's copies keep their number, band and look as they
    fall; one at the figure's own place is behind her and left out."""
    r = fx["recede"]
    if r is None:
        return [{"j": j, "band": k + j - 1, "depth": float(j), "delay": delay_of(fx, j), "alpha": 1.0,
                 "peel": None if fx["peel"] is None else peeled(fx, j, t)} for j in range(fx["count"], 0, -1)]
    p = phase(fx, t, times, fps, frame0)
    q = math.floor(p)
    f = p - q
    out = []
    for n in range(r["count"], -1, -1):
        u = n + f
        a = fade(r, u)
        if u <= 1e-9 or a <= 0.0:
            continue
        out.append({"j": n - q, "band": n - q - 1, "depth": u, "delay": delay_of(fx, u), "alpha": a, "peel": None})
    return out


def squash_at(sq, t, times):
    """(sx, sy) of a squash at clip second `t` on its hits (`times`, sorted): each presses the figure down by `amount` of
    its height over `attack` s, then it swings back up past its height and settles, a damped swing whose first stretch
    comes REBOUND x `decay` s after the squash; hits close together add up (to PRESS at most). sx = 1 / sy: the area
    is kept."""
    v = 0.0
    for h in times:
        a = t - h
        if a < 0.0:
            break
        if a < sq["attack"]:
            v += sq["amount"] * TW.ease(a / sq["attack"], "out")
            continue
        b = a - sq["attack"]
        if b < SETTLED * sq["decay"]:
            v += sq["amount"] * math.exp(-b / sq["decay"]) * math.cos(math.pi * b / (REBOUND * sq["decay"]))
    sy = 1.0 - min(max(v, PRESS[0]), PRESS[1])
    return 1.0 / sy, sy


def sources(fx, frame, frame0, hits=None):
    """The other frames of the cut a stack's copies show at `frame` (Blender frames), the nearest copy's first: their
    delays. A tunnel's copies age as they fall, so its hits say which: from `hits` (the [audio] hits file) for a lane,
    and without it every frame its farthest copy could show."""
    r = fx["recede"]
    if r is None:
        ds = [delay_of(fx, j) for j in range(1, fx["count"] + 1)]
    else:
        reach = max(delay_of(fx, float(j)) for j in range(1, r["count"] + 1))
        times = fx.get("times")
        if reach == 0:
            ds = []
        elif times is None and hits is None:
            ds = list(range(1, reach + 1))
        else:
            if times is None:
                from . import framefx as FX
                times = FX.hit_times(fx, hits)
            fps = fx["frame_rate"]
            ds = [c["delay"] for c in reversed(copies(fx, (frame - frame0) / fps, 0, times, fps, frame0))]
    out = []
    for d in ds:
        if d > 0 and frame - d not in out:
            out.append(frame - d)
    return out


def squash_reach(fx, fps):
    """Seconds before its span the hits of a stack's squash still move it: its oldest copy's delay (`fps` frames a
    second) and a squash's length."""
    n = fx["count"] if fx["recede"] is None else fx["recede"]["count"]
    return max(delay_of(fx, float(j)) for j in range(1, n + 1)) / fps + SETTLED * fx["squash"]["decay"] + \
        fx["squash"]["attack"]
