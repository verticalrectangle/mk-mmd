"""3D paint ribbons in the scene, bpy-free (docs/design.md: Ribbons): `[[ribbon]]` entries checked (`check`), their
centre lines (an orbit round a bone, a spiral up a body, a swoop through points, a fall out of a prop's nozzle, each
optionally led out of that nozzle and ending in a pile), resampled evenly with the direction their flat side faces, a
crayon's hand-drawn wobble, the draw-on clock (`clock`, `clock_keys`), the beat pulses (`pulse_keys`) and the push that
keeps a ribbon clear of the cast's bodies (`push_clear`). The build's `ribbons` stage (mkmmd/blender/build/ribbons.py)
resolves the targets, makes the geometry and keys the clock; the `ribbon_clear` check (mkmmd/checks/ribbons.py) measures
the scene it made with `visible` and the points it stored.

A ribbon's state at a clip second is (grow, tail): it shows from `tail * L` to `grow * L` of its arc length L. It draws on
over `draw` (grow 0 -> 1, eased), then optionally `retract`s (grow 1 -> 0: the stroke rewinds into its start) or is
`chase`d (tail 0 -> 1: it leaves along its own path); both ends are rounded caps one radius long."""
import math

import numpy as np

from . import lanes as LN

STYLES = ("crayon", "toothpaste", "neon")
PATHS = ("orbit", "spiral", "swoop", "fall")
EASES = ("linear", "smooth", "in", "out")
STRIPES = ("love", "gold", "foam", "iris", "rose")          # a toothpaste's stripes, round its profile
STYLE = {                                                   # per style: width (m), thickness as a share of it, colour
    "crayon": {"width": 0.05, "flat": 0.45, "color": "love", "shadow": True},
    "toothpaste": {"width": 0.056, "flat": 1.0, "shadow": True},
    "neon": {"width": 0.03, "flat": 0.15, "color": "iris", "shadow": False},
}
WOBBLE = 0.3                 # a crayon's wobble, as a share of its width
HALO = {"size": 4.0, "strength": 0.35}    # a neon's glow shell: radius in half widths, light as a share of the ribbon's
GLOW = 6.0                   # a neon's emission
HALO_SEEN = 0.55             # how far out of a halo's radius its glow still shows (the clearance counts that much)
MARGIN = 0.03                # clearance kept from the bodies (m)
PULSE = {"gain": 0.35, "attack": 0.04, "decay": 0.3}
RAW_STEP = 0.004             # spacing of the raw centre lines before the even resample (m)

TOP = ("name", "style", "orbit", "spiral", "swoop", "fall", "from", "pile", "draw", "ease", "retract", "chase", "width",
       "thickness", "color", "stripes", "glow", "halo", "wobble", "seed", "pulse", "ride", "clear", "at", "face",
       "squeeze", "spacing", "shadow")
TABLES = {"orbit": ("around", "radius", "height", "rise", "turns", "start", "tilt"),
          "spiral": ("around", "radius", "bottom", "top", "turns", "start"),
          "swoop": ("points", "cast", "closed"),
          "fall": ("reach", "floor"),
          "pile": ("turns", "radius", "shrink", "step"),
          "pulse": ("lane", "hits", "gain", "attack", "decay", "min_db", "every", "from", "to"),
          "clear": ("margin", "cast"),
          "halo": ("size", "strength")}


class RibbonError(ValueError):
    """A [[ribbon]] entry the build cannot use: an unknown key, a bad number, a path that cannot be made."""


# ------------------------------------------------------------------------------------------------------ checking
def _num(v, what, lo=None, hi=None, strict=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise RibbonError(f"{what} = {v!r}: a number")
    v = float(v)
    if lo is not None and (v <= lo if strict else v < lo):
        raise RibbonError(f"{what} = {v:g}: {'above' if strict else 'at least'} {lo:g}")
    if hi is not None and v > hi:
        raise RibbonError(f"{what} = {v:g}: at most {hi:g}")
    return v


def _span(v, what, after=None):
    """[a, b] clip seconds, b > a (and a >= `after`)."""
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise RibbonError(f"{what} = {v!r}: [from, to] clip seconds")
    a, b = _num(v[0], f"{what}[0]"), _num(v[1], f"{what}[1]")
    if b <= a:
        raise RibbonError(f"{what} = {v!r}: `to` after `from`")
    if after is not None and a < after - 1e-9:
        raise RibbonError(f"{what} = {v!r}: starts before the draw ends ({after:g} s)")
    return a, b


def _pair(v, what, lo=None, strict=True):
    """A number or [a, b] -> (a, b)."""
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise RibbonError(f"{what} = {v!r}: a number or [start, end]")
        return _num(v[0], f"{what}[0]", lo, strict=strict), _num(v[1], f"{what}[1]", lo, strict=strict)
    x = _num(v, what, lo, strict=strict)
    return x, x


def _table(spec, key, where, required=()):
    t = spec.get(key)
    if t is True:
        t = {}
    if not isinstance(t, dict):
        raise RibbonError(f"{where}: {key} = {t!r}: a table of {', '.join(TABLES[key])}")
    unknown = sorted(set(t) - set(TABLES[key]))
    if unknown:
        raise RibbonError(f"{where}: {key} has unknown keys {unknown} (known: {', '.join(TABLES[key])})")
    for k in required:
        if k not in t:
            raise RibbonError(f"{where}: {key} needs `{k}`")
    return t


def _target(v, what):
    if isinstance(v, str) and v:
        return v
    if isinstance(v, dict) and ("cast" in v or "prop" in v or "path" in v):
        return dict(v)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return [_num(x, what) for x in v]
    raise RibbonError(f"{what} = {v!r}: a target (\"cast:<name>.<bone>\", [x, y, z], {{cast, point}}, ...)")


def _source(v, where):
    """`from = "prop:<name>.<use point>"` -> {prop, use}."""
    if not (isinstance(v, str) and v.startswith("prop:") and "." in v[5:]):
        raise RibbonError(f"{where}: from = {v!r}: \"prop:<prop>.<use point>\" (a nozzle: \"prop:tube.nozzle\")")
    prop, _, use = v[5:].partition(".")
    if not prop or not use:
        raise RibbonError(f"{where}: from = {v!r}: \"prop:<prop>.<use point>\"")
    return {"prop": prop, "use": use}


def _ride(v, where):
    if v is None or v is False:
        return None
    if v is True:
        return True
    if isinstance(v, str) and (v.startswith("cast:") or v.startswith("prop:")) and len(v) > 5:
        return v
    raise RibbonError(f"{where}: ride = {v!r}: false, true (what the path goes round), \"cast:<name>[.<bone>]\" or "
                      f"\"prop:<name>\"")


def _path(spec, kind, where):
    t = _table(spec, kind, where, required={"orbit": ("around",), "spiral": ("around",), "swoop": ("points",)}.get(kind, ()))
    w = f"{where}: {kind}"
    if kind == "orbit":
        tilt = t.get("tilt", 0.0)
        tilt = _pair(tilt, f"{w}.tilt", strict=False) if isinstance(tilt, (list, tuple)) else (_num(tilt, f"{w}.tilt"), 0.0)
        turns = _num(t.get("turns", 1.0), f"{w}.turns")
        if abs(turns) < 1e-3:
            raise RibbonError(f"{w}.turns = {turns:g}: not zero (negative turns the other way)")
        return {"kind": kind, "around": _target(t["around"], f"{w}.around"),
                "radius": _pair(t.get("radius", 0.45), f"{w}.radius", 0.0),
                "height": _num(t.get("height", 0.0), f"{w}.height"), "rise": _num(t.get("rise", 0.0), f"{w}.rise"),
                "turns": turns, "start": None if t.get("start") is None else _num(t["start"], f"{w}.start"),
                "tilt": tilt}
    if kind == "spiral":
        turns = _num(t.get("turns", 2.5), f"{w}.turns")
        if abs(turns) < 1e-3:
            raise RibbonError(f"{w}.turns = {turns:g}: not zero")
        out = {"kind": kind, "around": _target(t["around"], f"{w}.around"),
               "radius": _pair(t.get("radius", 0.4), f"{w}.radius", 0.0),
               "bottom": _num(t.get("bottom", 0.1), f"{w}.bottom"),
               "top": None if t.get("top") is None else _num(t["top"], f"{w}.top"), "turns": turns,
               "start": _num(t.get("start", 0.0), f"{w}.start")}
        if out["top"] is not None and out["top"] <= out["bottom"]:
            raise RibbonError(f"{w}: top ({out['top']:g}) above bottom ({out['bottom']:g})")
        return out
    if kind == "swoop":
        pts = t["points"]
        if not isinstance(pts, list) or len(pts) < 2:
            raise RibbonError(f"{w}.points: at least two targets")
        cast = t.get("cast")
        if cast is not None and not (isinstance(cast, str) and cast):
            raise RibbonError(f"{w}.cast = {cast!r}: a cast member's name (plain [x, y, z] points are then in her frame)")
        closed = t.get("closed", False)
        if not isinstance(closed, bool):
            raise RibbonError(f"{w}.closed = {closed!r}: true or false")
        if closed and len(pts) < 3:
            raise RibbonError(f"{w}: a closed swoop needs three points")
        return {"kind": kind, "points": [_target(p, f"{w}.points[{i}]") for i, p in enumerate(pts)], "cast": cast,
                "closed": closed}
    return {"kind": kind, "reach": _num(t.get("reach", 0.25), f"{w}.reach", 0.0),
            "floor": _num(t.get("floor", 0.0), f"{w}.floor")}


def _pulse(spec, where):
    t = _table(spec, "pulse", where)
    w = f"{where}: pulse"
    if ("lane" in t) == ("hits" in t):
        raise RibbonError(f"{w}: give `lane` (a lane of the [audio] hits file) or `hits` (clip seconds), one of them")
    out = {k: _num(t.get(k, PULSE[k]), f"{w}.{k}", 0.0, strict=k != "gain") for k in PULSE}
    if "hits" in t:
        hits = t["hits"]
        if not isinstance(hits, list) or not hits:
            raise RibbonError(f"{w}.hits = {hits!r}: a list of clip seconds")
        out["hits"] = sorted(_num(h, f"{w}.hits") for h in hits)
        out["lane"] = None
    else:
        try:
            out["lane"] = LN.choice(t, w)
        except LN.LaneError as e:
            raise RibbonError(str(e)) from None
        out["hits"] = None
    return out


def check(spec, where=None):
    """One [[ribbon]] entry -> the normalised table the stage works from (every default filled in that does not need the
    scene: `width` stays None when the entry leaves it to a `from` use point's radius or the style)."""
    if not isinstance(spec, dict):
        raise RibbonError(f"[[ribbon]]: expected a table, got {spec!r}")
    name = spec.get("name")
    where = where or f"ribbon {name!r}"
    if not isinstance(name, str) or not name:
        raise RibbonError(f"{where}: `name` (the root object's name and its objects' prefix)")
    unknown = sorted(set(spec) - set(TOP))
    if unknown:
        raise RibbonError(f"{where}: unknown keys {unknown} (known: {', '.join(TOP)})")
    style = spec.get("style", "crayon")
    if style not in STYLES:
        raise RibbonError(f"{where}: style = {style!r} (one of {', '.join(STYLES)})")
    kinds = [k for k in PATHS if k in spec]
    src = _source(spec["from"], where) if "from" in spec else None
    if len(kinds) > 1:
        raise RibbonError(f"{where}: one path: {' or '.join(kinds)}")
    if not kinds and src is None:
        raise RibbonError(f"{where}: give a path: orbit, spiral or swoop = {{...}} (or `from` a nozzle: it falls)")
    kind = kinds[0] if kinds else "fall"
    if kind == "fall" and src is None:
        raise RibbonError(f"{where}: fall needs `from` (what the paste falls out of)")
    path = _path(dict(spec, fall=spec.get("fall", {})), kind, where)
    draw = _span(spec.get("draw"), f"{where}: draw")
    if "retract" in spec and "chase" in spec:
        raise RibbonError(f"{where}: retract (rewinds into its start) or chase (leaves along its path), not both")
    retract = _span(spec["retract"], f"{where}: retract", draw[1]) if "retract" in spec else None
    chase = _span(spec["chase"], f"{where}: chase", draw[1]) if "chase" in spec else None
    ease = spec.get("ease", "linear")
    if ease not in EASES:
        raise RibbonError(f"{where}: ease = {ease!r} (one of {', '.join(EASES)})")
    st = STYLE[style]
    width = None if spec.get("width") is None else _num(spec["width"], f"{where}: width", 0.0, strict=True)
    thick = None if spec.get("thickness") is None else _num(spec["thickness"], f"{where}: thickness", 0.0, strict=True)
    if thick is not None and style == "toothpaste":
        raise RibbonError(f"{where}: thickness is for the flat styles (crayon, neon); a toothpaste is round: give width")
    color = spec.get("color", st.get("color"))
    if style == "toothpaste":
        if "color" in spec:
            raise RibbonError(f"{where}: a toothpaste has stripes = [...] colours, not color")
        stripes = spec.get("stripes", list(STRIPES))
        if not (isinstance(stripes, list) and 2 <= len(stripes) <= 8 and all(isinstance(c, str) and c for c in stripes)):
            raise RibbonError(f"{where}: stripes = {stripes!r}: 2 to 8 colours (palette slots or #hex)")
    else:
        if "stripes" in spec:
            raise RibbonError(f"{where}: stripes are a toothpaste's; a {style} has one color")
        if not isinstance(color, str) or not color:
            raise RibbonError(f"{where}: color = {color!r}: a palette slot or #hex")
        stripes = None
    for k, ok in (("glow", "neon"), ("halo", "neon"), ("wobble", "crayon")):
        if k in spec and style != ok:
            raise RibbonError(f"{where}: {k} is a {ok}'s")
    halo = None
    if style == "neon":
        h = spec.get("halo", True)
        if h is not False:
            t = {} if h is True else _table(spec, "halo", where)
            halo = {"size": _num(t.get("size", HALO["size"]), f"{where}: halo.size", 1.0, strict=True),
                    "strength": _num(t.get("strength", HALO["strength"]), f"{where}: halo.strength", 0.0)}
    pile = None
    if spec.get("pile") not in (None, False):
        t = _table(spec, "pile", where)
        pile = {"turns": _num(t.get("turns", 2.5), f"{where}: pile.turns", 0.0, strict=True),
                "radius": None if t.get("radius") is None else _num(t["radius"], f"{where}: pile.radius", 0.0, strict=True),
                "shrink": _num(t.get("shrink", 0.6), f"{where}: pile.shrink", 0.0, 0.95),
                "step": None if t.get("step") is None else _num(t["step"], f"{where}: pile.step", 0.0)}
    elif kind == "fall" and "pile" not in spec:
        pile = {"turns": 2.5, "radius": None, "shrink": 0.6, "step": None}
    clear = spec.get("clear", True)
    if clear is False:
        clear = None
    elif clear is True:
        clear = {"margin": MARGIN, "cast": None}
    elif isinstance(clear, (int, float)):
        clear = {"margin": _num(clear, f"{where}: clear", 0.0), "cast": None}
    else:
        t = _table(spec, "clear", where)
        cast = t.get("cast")
        cast = [cast] if isinstance(cast, str) else cast
        if cast is not None and not (isinstance(cast, list) and cast and all(isinstance(c, str) for c in cast)):
            raise RibbonError(f"{where}: clear.cast = {t.get('cast')!r}: a cast member's name or a list of them")
        clear = {"margin": _num(t.get("margin", MARGIN), f"{where}: clear.margin", 0.0), "cast": cast}
    face = spec.get("face", "out" if kind in ("orbit", "spiral") else "up")
    if isinstance(face, (list, tuple)):
        face = [_num(x, f"{where}: face") for x in face]
        if len(face) != 3 or math.hypot(*face) < 1e-9:
            raise RibbonError(f"{where}: face = {spec['face']!r}: \"out\", \"up\" or a direction [x, y, z]")
    elif face not in ("out", "up"):
        raise RibbonError(f"{where}: face = {face!r}: \"out\", \"up\" or a direction [x, y, z]")
    squeeze = spec.get("squeeze")
    if squeeze is not None and not isinstance(squeeze, bool):
        squeeze = _pair(squeeze, f"{where}: squeeze", 0.0, strict=False)
    if squeeze and src is None:
        raise RibbonError(f"{where}: squeeze keys the prop of `from`; there is none")
    seed = spec.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RibbonError(f"{where}: seed = {seed!r}: a whole number")
    shadow = spec.get("shadow", st["shadow"])
    if not isinstance(shadow, bool):
        raise RibbonError(f"{where}: shadow = {shadow!r}: true or false")
    return {"name": name, "style": style, "path": path, "from": src, "pile": pile, "draw": draw, "ease": ease,
            "retract": retract, "chase": chase, "width": width, "thickness": thick, "color": color, "stripes": stripes,
            "glow": _num(spec.get("glow", GLOW), f"{where}: glow", 0.0) if style == "neon" else None, "halo": halo,
            "wobble": (None if spec.get("wobble") is None else _num(spec["wobble"], f"{where}: wobble", 0.0))
            if style == "crayon" else None,
            "seed": seed, "pulse": _pulse(spec, where) if spec.get("pulse") is not None else None,
            "ride": _ride(spec.get("ride"), where), "clear": clear,
            "at": _num(spec["at"], f"{where}: at") if "at" in spec else draw[0], "face": face, "squeeze": squeeze,
            "spacing": None if spec.get("spacing") is None else _num(spec["spacing"], f"{where}: spacing", 0.0, strict=True),
            "shadow": shadow}


def sizes(r, width):
    """(width, thickness, spacing) of a checked ribbon whose width is now known (the entry's, a nozzle's or the style's)."""
    st = STYLE[r["style"]]
    w = float(width)
    t = w if r["style"] == "toothpaste" else (r["thickness"] if r["thickness"] is not None else st["flat"] * w)
    return w, float(t), float(r["spacing"] if r["spacing"] is not None else min(0.01, w / 4.0))


# ------------------------------------------------------------------------------------------------------ the clock
def ease(kind, u):
    """The share of a move done at u (0..1) of its time: linear, smooth (smoothstep), in (quadratic), out."""
    u = min(max(float(u), 0.0), 1.0)
    if kind == "smooth":
        return u * u * (3.0 - 2.0 * u)
    if kind == "in":
        return u * u
    if kind == "out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    return u


def clock(r, t):
    """(grow, tail) of a checked ribbon at clip second t: it shows from tail to grow (shares of its length)."""
    t0, t1 = r["draw"]
    grow = ease(r["ease"], (t - t0) / (t1 - t0))
    tail = 0.0
    if r["retract"] is not None and t >= r["retract"][0]:
        a, b = r["retract"]
        grow = 1.0 - ease(r["ease"], (t - a) / (b - a))
    if r["chase"] is not None and t >= r["chase"][0]:
        a, b = r["chase"]
        tail = ease(r["ease"], (t - a) / (b - a))
    return grow, tail


def clock_keys(r, fps, frame0):
    """[(frame, grow, tail)] on every whole frame of the draw window and of a retract or chase (frames held CONSTANT
    outside them by the fcurve's own extrapolation, LINEAR between): the values `clock` gives there."""
    frames = set()
    for span in (r["draw"], r["retract"], r["chase"]):
        if span is None:
            continue
        f0, f1 = math.floor(frame0 + span[0] * fps), math.ceil(frame0 + span[1] * fps)
        frames.update(range(f0, f1 + 1))
    return [(f, *clock(r, (f - frame0) / fps)) for f in sorted(frames)]


def pulse_keys(hits, gain, attack, decay):
    """[(t, value)] of the thickness pulse, linear between: 1, rising to 1 + gain over `attack` s up to each hit and back
    to 1 over `decay` s after it; overlapping pulses take the larger (their crossing is a key, so the keys are exact)."""
    hits = sorted(float(h) for h in hits)
    if not hits or gain <= 0.0:
        return []

    def tri(h, t):
        if attack > 0.0 and h - attack <= t < h:
            return (t - (h - attack)) / attack
        if h <= t <= h + decay:
            return 1.0 - (t - h) / decay if decay > 0.0 else float(t == h)
        return 0.0

    ts = set()
    for h in hits:
        ts.update((h - attack, h, h + decay))
    for a, b in zip(hits, hits[1:]):
        if b - attack < a + decay and attack + decay > 0.0:
            ts.add((a * attack + b * decay) / (attack + decay))       # the fall of a meets the rise of b
    return [(t, 1.0 + gain * max(tri(h, t) for h in hits)) for t in sorted(ts)]


def value_at(keys, t, default=1.0):
    """A [(t, value)] curve (linear, held outside) at clip second t."""
    if not keys:
        return default
    ts, vs = zip(*keys)
    return float(np.interp(t, ts, vs))


def squeeze_keys(r, lo=0.0, hi=1.0):
    """[(t, value)] of the source prop's `squeeze` (lo -> hi over the draw, eased like it; back over a retract)."""
    pts = [(r["draw"][0], lo), (r["draw"][1], hi)]
    if r["retract"] is not None:
        pts += [(r["retract"][0], hi), (r["retract"][1], lo)]
    return pts


def visible(s, length, grow, tail):
    """Mask of the points at arc lengths `s` (n,) a ribbon of `length` shows at (grow, tail)."""
    s = np.asarray(s, float)
    eps = 1e-6
    return (s >= tail * length - eps) & (s <= grow * length + eps) & (grow - tail > 1e-6)


# ------------------------------------------------------------------------------------------------------ geometry
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def arclen(P):
    P = np.asarray(P, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def tangents(P):
    """Unit tangents (n, 3) of a polyline (central differences, one-sided at the ends)."""
    P = np.asarray(P, float)
    return _unit(np.gradient(P, axis=0))


def _rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _samples(length):
    return max(64, int(math.ceil(length / RAW_STEP)) + 1)


def orbit(center, axes, o):
    """(P (n, 3), out (n, 3)) of an orbit: round `center` in the frame `axes` (3x3 columns: x her left, y her back, z up;
    the identity for the world), angle 0 in front (-y) and 90 at her left, `turns` from `start` (degrees; negative turns
    go clockwise seen from above), the radius running from its first to its second value, `height` above the centre,
    climbing `rise` over the turns, the plane leaning by `tilt` (front up, left up; degrees). `out` points away from the
    orbit's axis in its plane."""
    r0, r1 = o["radius"]
    turns = o["turns"]
    n = _samples(2.0 * math.pi * max(r0, r1) * abs(turns) + abs(o["rise"]))
    u = np.linspace(0.0, 1.0, n)
    th = math.radians(o["start"] or 0.0) + 2.0 * math.pi * turns * u
    r = r0 + (r1 - r0) * u
    local = np.stack([r * np.sin(th), -r * np.cos(th), o["height"] + o["rise"] * u], 1)
    out = np.stack([np.sin(th), -np.cos(th), np.zeros(n)], 1)
    R = np.asarray(axes, float) @ _rx(-o["tilt"][0]) @ _ry(-o["tilt"][1])
    return np.asarray(center, float) + local @ R.T, out @ R.T


def spiral(base, axes, o, top):
    """(P, out) of a spiral up a body: round the vertical axis through `base` (the floor under it) from `bottom` to `top`
    metres above it (`top` from the entry or the caller's default), `turns` from `start` (degrees, 0 in front)."""
    r0, r1 = o["radius"]
    z0, z1 = o["bottom"], o["top"] if o["top"] is not None else top
    if z1 <= z0:
        raise RibbonError(f"spiral: its top ({z1:g} m) is not above its bottom ({z0:g} m)")
    turns = o["turns"]
    n = _samples(math.hypot(2.0 * math.pi * max(r0, r1) * abs(turns), z1 - z0))
    u = np.linspace(0.0, 1.0, n)
    th = math.radians(o["start"]) + 2.0 * math.pi * turns * u
    r = r0 + (r1 - r0) * u
    local = np.stack([r * np.sin(th), -r * np.cos(th), z0 + (z1 - z0) * u], 1)
    out = np.stack([np.sin(th), -np.cos(th), np.zeros(n)], 1)
    A = np.asarray(axes, float)
    return np.asarray(base, float) + local @ A.T, out @ A.T


def catmull(points, closed=False):
    """A centripetal Catmull-Rom curve through `points` (k, 3), sampled about RAW_STEP apart (n, 3); closed: a loop
    (the first point repeated at the end)."""
    Q = np.asarray(points, float)
    if len(Q) < 2:
        raise RibbonError("a swoop needs at least two points")
    if np.any(np.linalg.norm(np.diff(Q, axis=0), axis=1) < 1e-6):
        raise RibbonError("a swoop has two points in the same place")
    if closed:
        ext = np.vstack([Q[-1], Q, Q[0], Q[1]])
    else:
        ext = np.vstack([2.0 * Q[0] - Q[1], Q, 2.0 * Q[-1] - Q[-2]])
    out = []
    for i in range(len(ext) - 3):
        p0, p1, p2, p3 = ext[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** 0.5
        n = max(8, int(math.ceil(np.linalg.norm(p2 - p1) * 1.6 / RAW_STEP)))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(Q[:1] if closed else Q[-1:])
    return np.vstack(out)


def hermite(p0, d0, p1, d1, n=None):
    """A cubic from p0 heading d0 to p1 heading d1 (unit directions; tangent lengths the chord's)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    chord = float(np.linalg.norm(p1 - p0))
    m0, m1 = _unit(d0) * chord, _unit(d1) * chord
    n = n or max(16, int(math.ceil(chord * 1.5 / RAW_STEP)))
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def lead_in(p0, d0, P, inset):
    """The path `P` led out of a nozzle at `p0` heading `d0`: a straight `inset` inside the nozzle up to its mouth, then a
    cubic into P's first point along its first tangent. Returns (points, how many of them come before P)."""
    d0 = _unit(d0)
    P = np.asarray(P, float)
    inside = np.linspace(0.0, 1.0, max(3, int(math.ceil(inset / RAW_STEP)) + 1))[:-1, None]
    start = (np.asarray(p0, float) - d0 * inset) + inside * d0 * inset
    if np.linalg.norm(P[0] - p0) < 1e-4:
        lead = start
    else:
        lead = np.vstack([start, hermite(p0, d0, P[0], tangents(P[:3])[0])[:-1]])
    return np.vstack([lead, P]), len(lead)


def fall(p0, d0, f, width):
    """A paste rope out of a nozzle at `p0` heading `d0`: out along it, bending down under its weight to land `reach`
    metres further on (across) on the floor at height `floor`, resting on it (its centre half a width up)."""
    d0 = _unit(d0)
    h = np.array([d0[0], d0[1], 0.0])
    h = _unit(h) if np.linalg.norm(h) > 1e-6 else np.array([1.0, 0.0, 0.0])
    land = np.asarray(p0, float) + h * f["reach"]
    land[2] = f["floor"] + 0.5 * width
    if land[2] >= p0[2] - 1e-3:
        raise RibbonError(f"fall: the nozzle ({p0[2]:.3f} m up) is not above the floor it falls onto ({f['floor']:g} m)")
    return hermite(p0, d0, land, _unit(h * 0.6 - np.array([0.0, 0.0, 1.0])))


def pile(P, width, p):
    """A pile at the end of the path `P`: a coil that starts at its end turning on from its last heading (made level),
    counter-clockwise seen from above, and climbs `step` a turn (0.8 widths: each turn rests on the one under it), its
    radius (1.6 widths) shrinking by `shrink` over `turns`. Returns the coil's points after P's end (m, 3) and the way
    out from its axis at each (m, 3)."""
    P = np.asarray(P, float)
    end = P[-1]
    h = tangents(P[-3:])[-1]
    h = np.array([h[0], h[1], 0.0])
    h = _unit(h) if np.linalg.norm(h) > 1e-6 else np.array([1.0, 0.0, 0.0])
    r0 = p["radius"] if p["radius"] is not None else 1.6 * width
    step = p["step"] if p["step"] is not None else 0.8 * width
    c = end + np.array([-h[1], h[0], 0.0]) * r0              # the coil's axis, to the left of the heading
    a0 = math.atan2(end[1] - c[1], end[0] - c[0])
    n = _samples(2.0 * math.pi * r0 * p["turns"])
    u = np.linspace(0.0, 1.0, n)[1:]
    th = a0 + 2.0 * math.pi * p["turns"] * u
    r = r0 * (1.0 - p["shrink"] * u)
    out = np.stack([np.cos(th), np.sin(th), np.zeros(len(u))], 1)
    coil = np.stack([c[0] + r * out[:, 0], c[1] + r * out[:, 1], end[2] + step * p["turns"] * u], 1)
    return coil, out


def faces(P, face, out=None):
    """The direction a flat ribbon's face looks at each point (n, 3), square to the path: "out" the path's own outward
    direction (`out`: away from an orbit's or spiral's axis) or, for a swoop, away from its bends; "up" the world's up;
    [x, y, z] that direction; each made square to the tangent, carried on smoothly where the tangent runs along it."""
    P = np.asarray(P, float)
    T = tangents(P)
    if isinstance(face, str) and face == "out" and out is not None:
        want = np.asarray(out, float)
    elif isinstance(face, str) and face == "out":
        k = np.gradient(T, axis=0)                          # toward the centre of the bend; smoothed, then outward
        k = _smooth(k, max(3, len(P) // 40))
        want = -k
    else:
        v = np.array([0.0, 0.0, 1.0]) if isinstance(face, str) else _unit(np.asarray(face, float))
        want = np.tile(v, (len(P), 1))
    F = np.zeros_like(P)
    prev = None
    for i in range(len(P)):
        f = want[i] - (want[i] @ T[i]) * T[i]
        nf = float(np.linalg.norm(f))
        if prev is not None:
            carried = prev - (prev @ T[i]) * T[i]
            carried /= max(float(np.linalg.norm(carried)), 1e-12)
            w = min(1.0, nf / 0.25)                         # where the wanted direction runs along the path, carry on
            f = w * (f / max(nf, 1e-12)) + (1.0 - w) * carried if nf > 1e-9 else carried
        elif nf < 1e-9:
            f = np.cross(T[i], [1.0, 0.0, 0.0]) if abs(T[i][0]) < 0.9 else np.cross(T[i], [0.0, 1.0, 0.0])
        F[i] = f / max(float(np.linalg.norm(f)), 1e-12)
        prev = F[i]
    return F


def _smooth(X, k):
    """X (n, ...) smoothed along its first axis with a Gaussian of sigma k samples (the ends held by reflection)."""
    X = np.asarray(X, float)
    if k <= 0 or len(X) < 3:
        return X.copy()
    half = int(3 * k)
    w = np.exp(-0.5 * (np.arange(-half, half + 1) / k) ** 2)
    w /= w.sum()
    pad = np.concatenate([X[half:0:-1], X, X[-2:-half - 2:-1]]) if len(X) > half + 1 else np.pad(
        X, [(half, half)] + [(0, 0)] * (X.ndim - 1), mode="edge")
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, j], w, mode="valid") for j in range(flat.shape[1])], 1)
    return out.reshape((len(X),) + X.shape[1:])


def resample(P, F, ds):
    """P (and the face directions F) evenly spaced `ds` apart by arc length, both ends kept: (P, F, s)."""
    P = np.asarray(P, float)
    s = arclen(P)
    keep = np.r_[True, np.diff(s) > 1e-9]
    P, s = P[keep], s[keep]
    L = float(s[-1])
    if L <= 0.0:
        raise RibbonError("the path has no length")
    n = max(2, int(math.ceil(L / ds)) + 1)
    s2 = np.linspace(0.0, L, n)
    Q = np.stack([np.interp(s2, s, P[:, j]) for j in range(3)], 1)
    if F is None:
        return Q, None, s2
    F = np.asarray(F, float)[keep]
    G = np.stack([np.interp(s2, s, F[:, j]) for j in range(3)], 1)
    T = tangents(Q)
    G = G - np.einsum("ij,ij->i", G, T)[:, None] * T
    return Q, _unit(G), s2


def wobble(P, F, s, width, amp, seed):
    """A crayon's hand: the path pushed sideways (in the ribbon's plane, a little across it) by smooth noise of `amp`
    metres, its width varying by up to 12 % and its face rocking by up to 10 degrees, all slow along the stroke.
    Returns (P, F, width scale per point)."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s, float)
    scale = max(width / 0.05, 0.2)

    def noise():
        lam = np.array([0.95, 0.41, 0.17]) * scale * (0.85 + 0.3 * rng.random(3))
        ph = rng.random(3) * 2.0 * math.pi
        a = np.array([1.0, 0.55, 0.25])
        v = sum(a[j] * np.sin(2.0 * math.pi * s / lam[j] + ph[j]) for j in range(3))
        return v / a.sum()

    T = tangents(P)
    B = np.cross(T, F)
    n1, n2, n3, n4 = noise(), noise(), noise(), noise()
    P2 = np.asarray(P, float) + np.reshape(np.asarray(amp, float), (-1, 1)) * (n1[:, None] * B + 0.35 * n2[:, None] * F)
    ang = np.radians(10.0) * n4
    F2 = F * np.cos(ang)[:, None] + np.cross(T, F) * np.sin(ang)[:, None]
    return P2, _unit(F2), 1.0 + 0.12 * n3


def orbit_start(center, axes, o, p0):
    """The start angle (degrees) of an orbit nearest a point `p0` (a nozzle): where the lead-in to it is shortest."""
    R = np.asarray(axes, float) @ _rx(-o["tilt"][0]) @ _ry(-o["tilt"][1])
    q = R.T @ (np.asarray(p0, float) - np.asarray(center, float))
    return math.degrees(math.atan2(q[0], -q[1]))


def centre_line(r, geo, width, ds):
    """The centre line of a checked ribbon from what the scene resolved (`geo`: an orbit's {center, axes}, a spiral's
    {base, axes, top}, a swoop's {points}; with `from` the nozzle {from: (point, direction)}): the path led out of the
    nozzle, ending in its pile (the joint smoothed), resampled evenly `ds` apart, a crayon's wobble on it. Returns {P,
    F (the face directions), s, wscale (width scale per point), pinned (the points from inside the nozzle to a width past
    its mouth, which stay put), lead_m (the lead-in's length)}."""
    p = r["path"]
    kind = p["kind"]
    src = geo.get("from")
    out = None
    if kind == "orbit":
        o = dict(p)
        if o["start"] is None:
            o["start"] = orbit_start(geo["center"], geo["axes"], o, src[0]) if src is not None else 0.0
        P, out = orbit(geo["center"], geo["axes"], o)
    elif kind == "spiral":
        P, out = spiral(geo["base"], geo["axes"], p, geo["top"])
    elif kind == "swoop":
        P = catmull(geo["points"], p["closed"])
    else:
        P = fall(src[0], src[1], p, width)
    want = out if r["face"] == "out" else None
    lead = 0
    if src is not None:
        d0 = src[1] if src[1] is not None else _unit(P[0] - np.asarray(src[0], float))   # no direction: toward the path
        P, lead = lead_in(src[0], d0, P, inset=1.5 * width)
        if want is not None:
            want = np.vstack([np.tile(want[0], (lead, 1)), want])
    if r["pile"] is not None:
        join = len(P)
        coil, cout = pile(P, width, r["pile"])
        P = np.vstack([P, coil])
        if want is not None:
            want = np.vstack([want, cout])
        k = max(2, int(round(1.5 * width / RAW_STEP)))               # round the turn into the coil off
        bump = np.exp(-0.5 * ((np.arange(len(P)) - join) / (1.5 * k)) ** 2)
        bump[:lead] = 0.0
        P = P + (_smooth(P, k) - P) * bump[:, None]
    lead_m = float(arclen(P[:lead + 1])[-1]) if lead else 0.0
    F = faces(P, r["face"], want)
    P, F, s = resample(P, F, ds)
    pinned = int(np.searchsorted(s, lead_m + width, side="right")) if src is not None else 0
    wscale = np.ones(len(P))
    if r["style"] == "crayon":
        amp = (r["wobble"] if r["wobble"] is not None else WOBBLE * width) * np.clip(
            (np.arange(len(P)) - pinned) / 12.0, 0.0, 1.0)
        P, F, wscale = wobble(P, F, s, width, amp, r["seed"])
        F, s = settle(P, F)
    return {"P": P, "F": F, "s": s, "wscale": wscale, "pinned": pinned, "lead_m": lead_m}


def settle(P, F):
    """(F, s) for a centre line `P` moved after its faces were set (pushed clear): each face direction made square to the
    new tangent, carried on smoothly; s its arc lengths."""
    return faces(P, "out", F), arclen(P)


def respace(P, F, wscale, ds):
    """A moved centre line spaced `ds` apart again, its face directions carried and squared and its width scales
    following by arc length: (P, F, wscale, s)."""
    s0 = arclen(P)
    keep = np.r_[True, np.diff(s0) > 1e-9]
    Q, G, _ = resample(P, F, ds)
    G, s = settle(Q, G)
    w = np.interp(s / s[-1], s0[keep] / s0[-1], np.asarray(wscale, float)[keep])
    return Q, G, w, s


def envelope(r, width, wscale=None, thick_max=1.0):
    """The radius (n,) round each centre-line point that holds the ribbon's surface at its fattest (its half width, a
    crayon's width scale, a neon's halo as far as its glow shows, the largest pulse): what keeps clear of the bodies."""
    half = 0.5 * width * (np.ones(1) if wscale is None else np.asarray(wscale, float))
    if r["halo"] is not None:
        half = np.maximum(half, HALO_SEEN * r["halo"]["size"] * 0.5 * width)
    return half * thick_max


def _dilate(x, w):
    """The largest value of x (n,) within w samples either side of each sample."""
    if w <= 0 or len(x) < 2:
        return np.asarray(x, float).copy()
    pad = np.pad(np.asarray(x, float), w, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)


def push_clear(P, measure, pinned=0, iters=24, sigma=4.0, tol=5e-4):
    """Move the points of P (n, 3) out of what `measure` reports until it reports nothing: `measure(P)` -> (need (n,):
    how far a point must go along `dir` to be clear, <= 0 when it is, -inf where nothing is near; dir (n, 3) unit, away
    from what is nearest). Each round every point within two `sigma` of one that needs to move goes that far along its
    own direction, smoothed along the path (a Gaussian of `sigma` points), so the ribbon bends round a body instead of
    kinking; the first `pinned` points stay where they are (a nozzle's lead-in) and the moves fade in after them.
    Returns (P, {"rounds", "moved_m", "points", "need_m"}): a last `need_m` above tol is what it could not clear (bodies
    on both sides of the path closer together than the ribbon is fat)."""
    P = np.array(P, float)
    P0 = P.copy()
    n = len(P)
    fade = np.ones(n)
    if pinned:
        fade[:pinned] = 0.0
        ramp = np.arange(1, min(n - pinned, int(6 * sigma)) + 1)
        fade[pinned:pinned + len(ramp)] = ramp / (len(ramp) + 1.0)
    touched = np.zeros(n, bool)
    worst, rounds = 0.0, 0
    for rounds in range(1, iters + 2):
        need, d = measure(P)
        worst = float(need.max()) if n else 0.0                       # what is left, pinned points included
        free = np.where(fade > 0.0, need, -np.inf)
        if not n or free.max() <= tol or rounds > iters:
            break
        go = np.where(free > 0.0, free + 0.5 * tol, 0.0)
        touched |= go > 0.0
        P += _smooth(_dilate(go, int(2 * sigma))[:, None] * d, sigma) * fade[:, None]
    return P, {"rounds": rounds - 1, "moved_m": round(float(np.linalg.norm(P - P0, axis=1).max()) if n else 0.0, 4),
               "points": int(touched.sum()), "need_m": round(max(worst, 0.0), 4)}
