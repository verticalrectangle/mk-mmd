"""Ring portals, bpy-free (docs/design.md: Transitions and inserts, ring portals): the `ring` of an `[[insert]] shape =
"iris"` (mkmmd.core.iris), a thing round its round window that opens, holds and closes or grows with it, every part of it
sized in window radii (the window's radius is 1). This module checks a ring and lays it out; mkmmd.portal draws it.

    ring = "vinyl" | "plate" | "lens", or a table {kind, size, ...} with the kind's own keys:
      vinyl   a record whose centre hole is the window: round it the label (an original design of its own: a ground, a
              band arched over its top, a line of small print under it, a ring at its rim), the dead wax, the grooves
              with their track gaps and the sheen of the light across them (two bright wedges, shimmering and slowly
              swinging), the lead-in edge; all of it turning clockwise at `rpm`, the sheen standing still in the light
              size = 2.8          the record's radius
              rpm = 33.333        clockwise; negative turns it the other way
              color               the vinyl (default the palette's darkest, deepened to near black)
              label = [ground, band, print]   (default ["gold", "love", "text"])
              sheen = 0.8         the light across the grooves, 0..1
      plate   a dinner plate whose well is the window: the slope up from the well, the broad glazed rim with two thin
              coloured lines, the rounded lip, lit from the upper left (a crisp glaze highlight on the lip and a soft
              one across the rim), a soft shadow under it; optionally sprites (veggies) standing round the rim
              size = 1.46         the plate's radius
              color               the glaze (default the palette's lightest)
              band = "pine"       the two lines' colour; false for none
              sprites             a mandala's sprite forms: registry slugs, files, globs, {tag = "..."}, one or a list
              count = 8           sprites round the rim, the first at the top
              scale = 0.3         a sprite's longer side, in window radii
              orient = "upright"  "upright" or "radial" (each one's top away from the middle)
              turn = 0            degrees a second the sprites travel round the rim, counter-clockwise
              seed = 1            how the sprites are dealt round it
              eyes, mouth         googly eyes and a smile on the sprites (mkmmd.core.eyes.face)
      lens    concentric chrome rings, glass tubes that each show the world inside the window refracted (flipped and
              swirled across the tube, its colours pulled apart toward the tube's edges) under a chrome sheen with a
              horizon line and a highlight; some rings broken into arcs; each ring turning its own way; a soft glow
              round them
              size = 2.1          the outer ring's outer radius
              count = 4           rings, 1 .. 8, the first round the window's own edge
              chroma = 5          px at 1080 on the short side: the colour split at a tube's edges
              turn = 30           degrees a second the rings turn, every other ring the other way, the outer faster
              tint = "iris"       the glow and the chrome's cast
              seed = 1            the rings' widths, gaps, breaks and what each one shows
A ring sets the iris's `rim` defaults (the vignette inside the window, `RIMS`); the insert's own `rim` keys are laid over
them, and `rim = false` still turns it off."""
import math
import random

from . import palette as PAL

KINDS = ("vinyl", "plate", "lens")
ORIENTS = ("upright", "radial")
COMMON = {"kind", "size"}
KEYS = {"vinyl": COMMON | {"rpm", "color", "label", "sheen"},
        "plate": COMMON | {"color", "band", "sprites", "count", "scale", "orient", "turn", "seed", "eyes", "mouth"},
        "lens": COMMON | {"count", "chroma", "turn", "tint", "seed"}}
SIZE = {"vinyl": 2.8, "plate": 1.46, "lens": 2.1}
RIMS = {"vinyl": {"width": 0.12, "dark": 0.6, "glint": 0.0, "fringe": 0.0},     # the hole's wall, no lens glint
        "plate": {"width": 0.16, "dark": 0.3, "glint": 0.0, "fringe": 0.0},     # the rim's soft shade in the well
        "lens": {"width": 0.25, "dark": 0.6, "glint": 0.85, "fringe": 3.0}}     # a lens's rim, as a plain iris's
VINYL = {"rpm": 100.0 / 3.0, "label": ("gold", "love", "text"), "sheen": 0.8}
VINYL_DEEP = 0.22               # the default vinyl: the palette's darkest at this share of its sRGB value
# the record across its band (1 .. size): the label, the dead wax and the grooves end at these shares of it
LABEL, WAX, GROOVES = 0.25, 0.33, 0.975
TRACKS = (0.17, 0.35, 0.52, 0.68, 0.84)    # the gaps between tracks, shares of the grooves' band
PLATE = {"band": "pine", "count": 8, "scale": 0.3, "orient": "upright", "turn": 0.0, "seed": 1}
SLOPE, FLANGE = 0.22, 0.88      # the plate across its band (1 .. size): the slope up from the well and the rim end here
BANDS = ((0.2, 0.045), (0.78, 0.03))       # the rim's two lines: (middle, width) as shares of the rim
LENS = {"count": 4, "chroma": 5.0, "turn": 30.0, "tint": "iris", "seed": 1}
MAX_RINGS = 8


def _error(msg):
    from . import transition as TR      # transition imports iris, which imports this module: all there when it runs
    return TR.TransitionError(msg)


def _num(v, what, lo=None, hi=None, positive=False):
    if isinstance(v, bool):
        raise _error(f"{what} = {v!r}: expected a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise _error(f"{what} = {v!r}: expected a number") from None
    if (positive and x <= 0.0) or (lo is not None and x < lo) or (hi is not None and x > hi):
        rng = "positive" if positive else f"in {lo} .. {hi}" if hi is not None else f"at least {lo}"
        raise _error(f"{what} = {x}: expected a number {rng}")
    return x


def _int(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _error(f"{what} = {v!r}: expected a whole number in {lo} .. {hi}")
    return int(v)


def normalize(v, palette, what):
    """An iris's `ring` ("vinyl", "plate" or "lens", or {kind, ...}) with its kind's defaults, colours resolved
    (display-space floats); None when there is none."""
    from . import transition as TR
    from .iris import _lightest          # iris imports this module: it is all there by the time this runs
    if v is None or v is False:
        return None
    if isinstance(v, str):
        v = {"kind": v}
    if not isinstance(v, dict):
        raise _error(f"{what}: ring = {v!r}: expected one of {KINDS} or {{kind = ..., ...}}")
    kind = v.get("kind")
    if kind not in KINDS:
        raise _error(f"{what}: ring.kind = {kind!r}: expected one of {KINDS}")
    here = f"{what}: ring ({kind})"
    TR._unknown(v, KEYS[kind], here)
    out = {"kind": kind, "size": _num(v.get("size", SIZE[kind]), f"{here}: size", lo=1.05, hi=8.0)}
    if kind == "vinyl":
        out["rpm"] = _num(v.get("rpm", VINYL["rpm"]), f"{here}: rpm", lo=-300.0, hi=300.0)
        deep = "#" + "".join(f"{round(c * VINYL_DEEP * 255):02x}" for c in PAL.srgb(PAL.darkest(palette)))
        out["color"] = TR._colour(v.get("color", deep), palette, f"{here}: color")
        label = v.get("label", list(VINYL["label"]))
        label = [label] if isinstance(label, str) else label
        if not isinstance(label, (list, tuple)) or not 1 <= len(label) <= 3:
            raise _error(f"{here}: label = {label!r}: expected [ground, band, print]: one to three colours")
        label = list(label) + list(VINYL["label"][len(label):])
        out["label"] = [TR._colour(c, palette, f"{here}: label[{i}]") for i, c in enumerate(label)]
        out["sheen"] = _num(v.get("sheen", VINYL["sheen"]), f"{here}: sheen", lo=0.0, hi=1.0)
        out["light"] = TR._colour(_lightest(palette), palette, here)
        out["shade"] = TR._colour(PAL.darkest(palette), palette, here)
        return out
    if kind == "plate":
        out["color"] = TR._colour(v.get("color", _lightest(palette)), palette, f"{here}: color")
        band = v.get("band", PLATE["band"])
        out["band"] = None if band is False else TR._colour(band, palette, f"{here}: band")
        out["count"] = _int(v.get("count", PLATE["count"]), f"{here}: count", 1, 64)
        out["scale"] = _num(v.get("scale", PLATE["scale"]), f"{here}: scale", positive=True, hi=2.0)
        out["orient"] = v.get("orient", PLATE["orient"])
        if out["orient"] not in ORIENTS:
            raise _error(f"{here}: orient = {out['orient']!r}: expected one of {ORIENTS}")
        out["turn"] = _num(v.get("turn", PLATE["turn"]), f"{here}: turn")
        out["seed"] = _int(v.get("seed", PLATE["seed"]), f"{here}: seed", 0, 2 ** 31 - 1)
        out["sprites"] = _sprites(v["sprites"], here) if "sprites" in v else None
        out["face"] = _face(v, here) if out["sprites"] is not None else None
        if out["sprites"] is None and any(k in v for k in ("eyes", "mouth", "count", "scale", "orient", "turn", "seed")):
            extra = sorted(k for k in ("eyes", "mouth", "count", "scale", "orient", "turn", "seed") if k in v)
            raise _error(f"{here}: {extra} are about the sprites round the rim: give `sprites` too")
        out["light"] = TR._colour(_lightest(palette), palette, here)
        out["shade"] = TR._colour(PAL.darkest(palette), palette, here)
        return out
    out["count"] = _int(v.get("count", LENS["count"]), f"{here}: count", 1, MAX_RINGS)
    out["chroma"] = _num(v.get("chroma", LENS["chroma"]), f"{here}: chroma", lo=0.0, hi=64.0)
    out["turn"] = _num(v.get("turn", LENS["turn"]), f"{here}: turn")
    out["tint"] = TR._colour(v.get("tint", LENS["tint"]), palette, f"{here}: tint")
    out["seed"] = _int(v.get("seed", LENS["seed"]), f"{here}: seed", 0, 2 ** 31 - 1)
    out["light"] = TR._colour(_lightest(palette), palette, here)
    out["shade"] = TR._colour(PAL.darkest(palette), palette, here)
    out["bands"] = lens_bands(out)
    return out


def _sprites(v, what):
    """The plate's sprites in a mandala ring's forms (mkmmd.core.mandala._refs)."""
    from . import framefx as FX
    from . import mandala as MD
    try:
        return MD._refs(FX, v, what)
    except FX.FxError as e:
        raise _error(str(e)) from None


def _face(v, what):
    """The sprites' googly eyes and smile (mkmmd.core.eyes.face): None without them."""
    from . import eyes as EY
    from . import framefx as FX
    try:
        return EY.face({k: v[k] for k in EY.KEYS if k in v}, what)
    except FX.FxError as e:
        raise _error(str(e)) from None


def rim_defaults(ring):
    """The defaults of the iris's `rim` under a ring (None: the plain iris's own)."""
    return None if ring is None else RIMS[ring["kind"]]


# ================================================================================================= layouts
def vinyl_layout(ring):
    """The record's parts as (inner, outer) radii in window radii: label, wax (the smooth run-out round the label),
    grooves, edge (the lead-in's rounded rim), and `tracks`, the radii of the gaps between its tracks."""
    s = ring["size"] - 1.0
    label, wax, grooves = 1.0 + LABEL * s, 1.0 + WAX * s, 1.0 + GROOVES * s
    return {"label": (1.0, label), "wax": (label, wax), "grooves": (wax, grooves), "edge": (grooves, ring["size"]),
            "tracks": [wax + f * (grooves - wax) for f in TRACKS]}


def plate_layout(ring):
    """The plate's parts as (inner, outer) radii in window radii: slope (up from the well), flange (the broad rim), bead
    (the rounded lip), `bands` (the rim's two lines) and `seat`, the radius the sprites stand on (the rim's middle)."""
    s = ring["size"] - 1.0
    slope, flange = 1.0 + SLOPE * s, 1.0 + FLANGE * s
    w = flange - slope
    return {"slope": (1.0, slope), "flange": (slope, flange), "bead": (flange, ring["size"]),
            "bands": [(slope + (m - b / 2.0) * w, slope + (m + b / 2.0) * w) for m, b in BANDS],
            "seat": (slope + flange) / 2.0}


def lens_bands(ring):
    """The lens rings, the innermost first, each {a, b (inner and outer radius, window radii), turn (degrees a second,
    counter-clockwise), phase (degrees), breaks [[start, span] degrees in the ring's own turn], zoom (where across the
    window the ring's picture comes from, a share of its own radius), flip (its picture's scale across the tube; negative:
    turned over)}: seeded widths and gaps between them, filling 1 .. size."""
    rng = random.Random(ring["seed"] * 7919 + ring["count"])
    n = ring["count"]
    widths = [rng.uniform(0.35, 1.6) for _ in range(n)]
    gaps = [rng.uniform(0.2, 0.45) for _ in range(n - 1)]
    total = sum(widths) + sum(gaps) * (sum(widths) / n)
    unit = (ring["size"] - 1.0) / total
    out, at = [], 1.0
    for i in range(n):
        a, b = at, at + widths[i] * unit
        k = rng.choices((0, 1, 2, 3), (0.3, 0.3, 0.25, 0.15))[0]
        breaks = sorted([rng.uniform(0.0, 360.0), rng.uniform(12.0, 50.0)] for _ in range(k))
        out.append({"a": a, "b": b, "turn": ring["turn"] * (1.0 + 0.35 * i) * (1.0 if i % 2 == 0 else -1.0),
                    "phase": rng.uniform(0.0, 360.0), "breaks": breaks, "zoom": rng.uniform(0.45, 0.85),
                    "flip": rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.4)})
        at = b + (gaps[i] * sum(widths) / n * unit if i < n - 1 else 0.0)
    out[-1]["b"] = ring["size"]                      # rounding: the last ring ends exactly at the size
    return out


# ================================================================================================= motion
def spin(ring, t):
    """Degrees a record has turned, clockwise on screen, `t` seconds into the window."""
    return ring["rpm"] / 60.0 * 360.0 * t


def seats(ring, t):
    """[(angle, j)] the plate's sprites at `t` seconds into the window: copy j's place round the rim in degrees
    counter-clockwise from straight up, the first at the top, evenly round, travelling `turn` degrees a second."""
    return [((360.0 * j / ring["count"] + ring["turn"] * t) % 360.0, j) for j in range(ring["count"])]


def sprite_pick(ring, j, n):
    """Which of `n` sprites copy j shows: shuffled by the seed and dealt round the rim, every one before any comes again."""
    perm = list(range(n))
    random.Random(ring["seed"] * 104729 + n).shuffle(perm)
    return perm[j % n]
