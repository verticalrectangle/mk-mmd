"""Festival lasers, bpy-free (docs/design.md: Frame effects, Lasers): `[[fx]] kind = "lasers"` checked and timed. The
picture work is mkmmd/lasers.py.

Fans of thin bright beams laid over the picture, a festival's lasers cutting through haze. Each fan leaves an `origin`:
[x, y] of the frame (it may lie outside it: a bank of lasers under the stage) or {point = "<expression>"}, a point of the
scene projected through the camera of the shot in the cut on every frame (a headlamp: the fan rides it); a list of them
is a fan from each. A fan is `count` beams spread over `spread` degrees about its `aim` (degrees counter-clockwise on
screen from the right: 90 is straight up). The fans sweep `sweep` degrees either side of the aim and breathe, closing
by `fan` of their spread: steadily, `rate` swings a second, or on hits (`times`, or a `lane` of the [audio] hits file):
each hit swings them over to the other side in `snap` seconds, throws them wide open and lights the beams (`pulse`),
dying away as exp(-age / `decay`), and steps their colours on. With `mirror` every other fan is the one before it
reflected left for right (its aim about the vertical, its sweep the other way), so two headlamps' fans cross and part.

Flash-safe: a hit sooner than PACE seconds after the last one a laser followed is skipped (`paced`), so its pulses,
colour steps and swings come three times a second at most.
"""
import math

from . import lanes as LN
from . import tween as TW

KINDS = ("lasers",)
BY = ("fan", "beam")                # what the colours step over: a colour per fan, or per beam of a fan
BLENDS = ("add", "screen")          # how the light is laid on the picture
PACE = 1.0 / 3.0                    # seconds: the shortest gap between two hits a laser follows (three a second at most)
ATTACK = 0.03                       # seconds: a hit's pulse and its opening fan rise this fast
MAX_FANS = 16
KEYS = {"lasers": {"origin", "count", "spread", "aim", "sweep", "fan", "rate", "colors", "by", "width", "glow", "haze",
                   "length", "fade", "pulse", "snap", "decay", "mirror", "blend", "mask", "objects", "seed", "times",
                   "lane", "min_db", "every"}}
DEFAULTS = {"lasers": {"origin": [0.5, 1.0], "count": 7, "spread": 50.0, "aim": 90.0, "sweep": 30.0, "fan": 0.5,
                       "rate": 0.25, "colors": ["love", "foam"], "by": "fan", "width": 2.5, "glow": 0.6, "haze": 0.35,
                       "length": 4.0, "fade": 0.35, "pulse": 0.4, "snap": 0.18, "decay": 0.35, "mirror": True,
                       "blend": "add", "mask": None, "seed": 1, "min_db": None, "every": 1}}


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def _origins(FX, v, what):
    """`origin`: one ([x, y] or {point}) or a list of them -> [center, ...] (framefx's `_center` form)."""
    many = isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(o, (dict, list, tuple)) for o in v)
    items = list(v) if many else [v]
    if not items or len(items) > MAX_FANS:
        raise FX.FxError(f"{what}: origin = {v!r}: expected [x, y] of the frame, {{point = \"<expression>\"}}, or a "
                         f"list of 1 to {MAX_FANS} of them")
    return [FX._center(o, f"{what} origin") for o in items]


def normalize(spec, out, palette, what):
    """A lasers entry's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    out["origin"] = _origins(FX, out["origin"], what)
    out["count"] = FX._num(out["count"], f"{what} count", 1, 64, integer=True)
    out["spread"] = FX._num(out["spread"], f"{what} spread", 0.0, 360.0)
    out["aim"] = FX._num(out["aim"], f"{what} aim", -360.0, 360.0)
    out["sweep"] = FX._num(out["sweep"], f"{what} sweep", 0.0, 180.0)
    out["fan"] = FX._num(out["fan"], f"{what} fan", 0.0, 1.0)
    out["rate"] = FX._num(out["rate"], f"{what} rate", 0.0, 20.0)
    out["colors"] = FX._colours(out["colors"], palette, f"{what} colors")
    if out["by"] not in BY:
        raise FX.FxError(f"{what}: by {out['by']!r} is none of {', '.join(BY)} (a colour per fan or per beam)")
    out["width"] = FX._num(out["width"], f"{what} width", 0.2, 40.0)
    out["glow"] = FX._num(out["glow"], f"{what} glow", 0.0, 4.0)
    out["haze"] = FX._num(out["haze"], f"{what} haze", 0.0, 2.0)
    out["length"] = FX._num(out["length"], f"{what} length", 0.01, 20.0)
    out["fade"] = FX._num(out["fade"], f"{what} fade", 0.0, 1.0)
    out["pulse"] = FX._num(out["pulse"], f"{what} pulse", 0.0, 1.0)
    out["snap"] = FX._num(out["snap"], f"{what} snap", 0.0, 2.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", lo=1e-3)
    if not isinstance(out["mirror"], bool):
        raise FX.FxError(f"{what}: mirror is true or false")
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._mask(spec, out, what)
    FX._hits(spec, out, what, required=False)


def point_key(fx, j):
    return f"x{fx['index']}o{j}"


def points(fx):
    """[(key, expression)]: the scene points a lasers entry's fans leave from (x<i>o<j>)."""
    return [(point_key(fx, j), o["expr"]) for j, o in enumerate(fx["origin"]) if o["mode"] == "point"]


# ================================================================================================= timing
def paced(times, gap=PACE):
    """The hits a laser follows: `times` (clip seconds) in order, less every one sooner than `gap` seconds after the
    last one kept (flash-safe: three a second at most by default)."""
    return LN.pace(times, gap)


def side_of(k):
    """Where the sweep rests after `k` hits: +1 (turned counter-clockwise by `sweep`) before the first, then the other
    side on every hit."""
    return 1.0 if k % 2 == 0 else -1.0


def state(fx, t, hits):
    """The fans at clip second `t`: (k, side, open, light). `hits` are the clip seconds it follows (paced); k is how many
    have gone by (its colours' step), `side` -1 .. 1 where the sweep has the fans (aim + side x sweep), `open` 0 .. 1 how
    far they are opened (1 their `spread`, 0 closed by `fan`) and `light` 0 .. 1 the beams' brightness. Without hits the
    fans swing steadily, sin(2 pi rate (t - from)), widest through the middle of each swing and closing toward its ends,
    at full light. On hits each swings them to the other side over `snap` seconds (eased in and out), opens them and
    lights the beams, rising over ATTACK and dying as exp(-age / decay): light = 1 - pulse + pulse x that."""
    if not hits:
        ph = 2.0 * math.pi * fx["rate"] * (t - fx["from"])
        return 0, math.sin(ph), 0.5 + 0.5 * math.cos(2.0 * ph), 1.0
    past = [h for h in hits if h <= t]
    k = len(past)
    if k == 0:
        return 0, side_of(0), 0.0, 1.0 - fx["pulse"]
    age = t - past[-1]
    u = 1.0 if fx["snap"] <= 0 else TW.ease(age / fx["snap"], "inout")
    side = side_of(k - 1) + (side_of(k) - side_of(k - 1)) * u
    rise = 1.0 if age >= ATTACK else 1.0 - (1.0 - age / ATTACK) ** 2
    flash = rise * math.exp(-max(age - ATTACK, 0.0) / fx["decay"])
    return k, side, flash, 1.0 - fx["pulse"] + fx["pulse"] * flash


def fan(fx, i, st):
    """The beams of fan `i` in state `st` (`state`): [(angle, colour index)], angles in degrees counter-clockwise on
    screen from the right, the colour index into `colors` (round and round)."""
    k, side, opened, _ = st
    flip = fx["mirror"] and i % 2 == 1
    aim = (180.0 - fx["aim"]) if flip else fx["aim"]
    aim += (-1.0 if flip else 1.0) * fx["sweep"] * side
    spread = fx["spread"] * (1.0 - fx["fan"] * (1.0 - opened))
    n = fx["count"]
    out = []
    for j in range(n):
        a = aim if n == 1 else aim - 0.5 * spread + spread * j / (n - 1)
        out.append((a, k + (j if fx["by"] == "beam" else i)))
    return out
