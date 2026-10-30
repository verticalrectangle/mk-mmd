"""The iris insert, bpy-free and numpy-free (docs/design.md: Transitions and inserts): `[[insert]] shape = "iris"`, a round
window that opens from a point of the scene (its `anchor`, projected through the host shot's camera on every window frame)
onto another shot or a video, holds, and closes back onto its point or grows until its picture is the whole frame. Unlike
a thought bubble it has no trail. This module checks and times it and does its circle maths; mkmmd.iris draws it.

    from, to                    clip seconds: the window (frames f0 .. f1 - 1), inside one shot of the cut, the host
    shot | clip                 what it shows: a [[shot]] (in the cut or `plate = true`, not the host), or a video, as a
                                split's (a file or an asset slug, or {src, in, rate, loop, fit, mirror, back})
    anchor                      a `mk q` expression: the circle's centre (the `point` layer `i<j>`)
    radius | size               the rest radius in metres at the anchor's depth (it stays the size of a disc in the scene),
                                or the rest diameter as a share of the frame's short side (default size 0.3)
    open = {dur, ease}          0.25 s, "out": from the anchor to the rest radius (dur 0: open on the first frame)
    out = "close" | "expand"    back onto its point over open.dur, gone on `to`; or grown until its picture is the frame
    expand = {dur, ease}        0.6 s, "in" (only with out = "expand")
    rim = {width, dark, color, glint, fringe} | false
                                the inner rim, a lens at the end of a tube: a vignette over the outer `width` of the radius
                                darkening toward `color` (the palette's darkest) by `dark` at the edge, a glint (a crescent
                                in the palette's lightest colour at the upper left, a fainter one opposite) and a chromatic
                                `fringe` (px at 1080 on the short side)
    edge = {color, width}       a ring round the outside (none by default)
    ring = "vinyl" | "plate" | "lens" | {kind, ...}
                                a portal's ring round the window, growing with it (mkmmd.core.portal): a record whose
                                centre hole it is, a dinner plate whose well it is, or concentric chrome rings; it sets
                                the rim's defaults
    aspect.<output> = {radius} | {size}   that output's own rest size
"""
import math

from . import palette as PAL
from . import portal as PT
from . import tween as TW

KEYS = {"from", "to", "shot", "clip", "shape", "anchor", "radius", "size", "open", "out", "expand", "rim", "edge", "ring",
        "aspect"}
RIM_KEYS = {"width", "dark", "color", "glint", "fringe"}
OUTS = ("close", "expand")
SIZE = 0.3                      # the rest diameter, a share of the frame's short side, when neither radius nor size is given
OPEN = {"dur": 0.25, "ease": "out"}
EXPAND = {"dur": 0.6, "ease": "in"}
RIM = {"width": 0.3, "dark": 0.8, "glint": 0.85, "fringe": 2.0}    # its colour: the palette's darkest
COVER = 1.04                    # the picture at rest covers the circle, this much to spare
FILL_MARGIN = 1.02              # the radius that fills the frame, this much to spare
MIN_RADIUS = 0.5                # px: a smaller circle draws nothing


def _error(msg):
    from . import transition as TR      # transition imports this module; its helpers are all there by the time this runs
    return TR.TransitionError(msg)


def _num(v, what):
    if isinstance(v, bool):
        raise _error(f"{what} = {v!r}: expected a number")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise _error(f"{what} = {v!r}: expected a number") from None


def _table(v, keys, what):
    from . import transition as TR
    if not isinstance(v, dict):
        raise _error(f"{what} = {v!r}: expected {{{', '.join(sorted(keys))}}}")
    TR._unknown(v, keys, what)
    return v


def _extent(v, what):
    """{"radius", "size"}: the rest radius in metres or the rest diameter as a share of the short side; one of them."""
    if "radius" in v and "size" in v:
        raise _error(f"{what}: radius and size together: give one (radius: metres at the anchor's depth; size: the "
                     f"diameter as a share of the frame's short side)")
    if "radius" in v:
        r = _num(v["radius"], f"{what}: radius")
        if r <= 0.0:
            raise _error(f"{what}: radius = {r}: expected metres, positive")
        return {"radius": r, "size": None}
    s = _num(v.get("size", SIZE), f"{what}: size")
    if not 0.0 < s <= 1.0:
        raise _error(f"{what}: size = {s}: expected the diameter as a share of the frame's short side, in (0, 1]")
    return {"radius": None, "size": s}


def _timing(v, default, what, zero):
    t = _table(v, {"dur", "ease"}, what)
    out = {"dur": _num(t.get("dur", default["dur"]), f"{what}.dur"), "ease": t.get("ease", default["ease"])}
    if out["dur"] < 0.0 or (out["dur"] == 0.0 and not zero):
        raise _error(f"{what}.dur = {out['dur']}: expected seconds, {'0 or more' if zero else 'positive'}")
    if out["ease"] not in TW.EASES:
        raise _error(f"{what}.ease = {out['ease']!r}, expected one of {TW.EASES}")
    return out


def _lightest(palette):
    """The palette's lightest colour by luminance (hex): Dawn's `surface`, a warm white."""
    def luminance(hexstr):
        r, g, b = PAL.linear(hexstr)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return max(palette.values(), key=luminance)


def _rim(v, palette, what, defaults=None):
    """`rim`: {width, dark, color, glint, fringe} with its defaults (`defaults`: a ring's, laid under the keys given; else
    RIM) and `light` (the glint's colour), or None for `false`."""
    from . import transition as TR
    if v is False:
        return None
    if not isinstance(v, dict):
        raise _error(f"{what}: rim = {v!r}: expected {{width, dark, color, glint, fringe}} or false")
    TR._unknown(v, RIM_KEYS, f"{what}: rim")
    out = {k: _num(v.get(k, d), f"{what}: rim.{k}") for k, d in (defaults or RIM).items()}
    if not 0.0 < out["width"] < 1.0:
        raise _error(f"{what}: rim.width = {out['width']}: expected the vignette's share of the radius, in (0, 1)")
    for k in ("dark", "glint"):
        if not 0.0 <= out[k] <= 1.0:
            raise _error(f"{what}: rim.{k} = {out[k]}: expected 0..1")
    if not 0.0 <= out["fringe"] <= 32.0:
        raise _error(f"{what}: rim.fringe = {out['fringe']}: expected px at 1080 on the short side, 0..32")
    out["color"] = TR._colour(v.get("color", PAL.darkest(palette)), palette, f"{what}: rim.color")
    out["light"] = TR._colour(_lightest(palette), palette, f"{what}: rim")
    return out


def normalize(spec, palette, what):
    """A `shape = "iris"` insert with its defaults, colours resolved: the picture inside (`shot`, or `clip`: a video and
    how it plays), the anchor, the rest size (`radius` metres or `size`, the other None), the opening and the way out,
    the ring (mkmmd.core.portal; None without one), the rim (None for `rim = false`; a ring's defaults under it), the
    edge (None without one) and the per-output rest sizes."""
    from . import transition as TR
    TR._unknown(spec, KEYS, what)
    for k in ("from", "to", "anchor"):
        if k not in spec:
            raise _error(f"{what}: needs `{k}`")
    if ("shot" in spec) == ("clip" in spec):
        raise _error(f"{what}: an iris needs exactly one of `shot` (the [[shot]] inside it) and `clip` (a video there)")
    if not isinstance(spec["anchor"], str) or not spec["anchor"].strip():
        raise _error(f"{what}: anchor = {spec['anchor']!r}: expected a `mk q` expression such as 'obj(\"toy_eyehole\").loc'")
    out = {"from": _num(spec["from"], f"{what}: from"), "to": _num(spec["to"], f"{what}: to"), "shape": "iris",
           "shot": str(spec["shot"]) if "shot" in spec else None,
           "clip": TR._split_clip(spec["clip"], palette, what) if "clip" in spec else None,
           "anchor": spec["anchor"], **_extent(spec, what),
           "open": _timing(spec.get("open", {}), OPEN, f"{what}: open", True), "out": spec.get("out", "close")}
    if out["to"] <= out["from"]:
        raise _error(f"{what}: `to` must come after `from`")
    if out["out"] not in OUTS:
        raise _error(f"{what}: out = {out['out']!r}, expected one of {OUTS}")
    if spec.get("expand") and out["out"] != "expand":
        raise _error(f"{what}: `expand` only applies to out = 'expand'")
    out["expand"] = _timing(spec.get("expand", {}), EXPAND, f"{what}: expand", False)
    out["ring"] = PT.normalize(spec.get("ring"), palette, what)
    out["rim"] = _rim(spec.get("rim", {}), palette, what, PT.rim_defaults(out["ring"]))
    out["edge"] = TR._ring(spec["edge"], palette, f"{what}: edge") if spec.get("edge") is not None else None
    aspect = spec.get("aspect") or {}
    if not isinstance(aspect, dict):
        raise _error(f"{what}: aspect = {aspect!r}: expected a table of outputs")
    out["aspect"] = {}
    for name, over in aspect.items():
        here = f"{what}: aspect.{name}"
        _table(over, {"radius", "size"}, here)
        if not over:
            raise _error(f"{here}: expected {{radius}} or {{size}}")
        out["aspect"][name] = _extent(over, here)
    return out


# ================================================================================================= the plan
def frames(ins, what, f0, f1, fps, cuts):
    """{n_in, n_exit}: the frames of the opening (0 for open.dur = 0) and of the close (as many) or the expand, checked
    against the window `f0` .. `f1` - 1 and the cut (an expand into a `shot` must end where that shot starts)."""
    from . import transition as TR
    n_in = 0 if ins["open"]["dur"] <= 0.0 else max(TR.MIN_WINDOW, int(round(ins["open"]["dur"] * fps)))
    if ins["out"] == "expand":
        if ins["shot"] is not None:
            starts = [c for c in cuts if c["from"] == f1]
            if not starts or starts[0]["name"] != ins["shot"]:
                got = repr(starts[0]["name"]) if starts else "no shot starts there"
                raise _error(f"{what}: out = 'expand' grows into the shot that starts at `to` ({got}), so `shot` must be "
                             f"that shot, not {ins['shot']!r}")
        n_exit = max(TR.MIN_WINDOW, int(round(ins["expand"]["dur"] * fps)))
    else:
        n_exit = n_in
    need = max(TR.MIN_WINDOW, n_in + n_exit)
    if f1 - f0 < need:
        raise _error(f"{what}: {f1 - f0} frames are too few: its opening takes {n_in} and its {ins['out']} {n_exit}, at "
                     f"least {need} in all")
    return {"n_in": n_in, "n_exit": n_exit}


# ================================================================================================= timing
def state(ins, frame):
    """What the iris shows at `frame`: {"phase": "in" | "hold" | "out" | "expand" | "none", "open": its radius as a share
    of the rest radius, "p": the growth of an expand (0 at rest .. 1, its picture the whole frame)}. Every frame of the
    opening shows the circle bigger, the first small (not 0), the last at rest; the close is the opening played backwards,
    ending on the window's last frame with a small circle (gone on `to`); an expand moves on every frame and reaches
    p = 1 exactly on the window's last frame."""
    i, j = frame - ins["f0"], ins["f1"] - 1 - frame
    if not 0 <= i < ins["f1"] - ins["f0"]:
        return {"phase": "none", "open": 0.0, "p": 0.0}
    n_in, n_exit = ins["n_in"], ins["n_exit"]
    if i < n_in:
        return {"phase": "in", "open": TW.ease((i + 1) / n_in, ins["open"]["ease"]), "p": 0.0}
    if j < n_exit and ins["out"] == "close":
        return {"phase": "out", "open": TW.ease((j + 1) / n_exit, ins["open"]["ease"]), "p": 0.0}
    if j < n_exit:
        return {"phase": "expand", "open": 1.0, "p": TW.ease((n_exit - j) / n_exit, ins["expand"]["ease"])}
    return {"phase": "hold", "open": 1.0, "p": 0.0}


# ================================================================================================= the circle
def extent(ins, output):
    """{"radius", "size"} of the rest circle in one output: its `aspect.<output>` when it has one (one replaces the other)."""
    return ins["aspect"].get(output) or {"radius": ins["radius"], "size": ins["size"]}


def rest_radius(ins, output, size, per_metre):
    """The rest radius in px in a frame `size` = (w, h): `radius` metres seen at the anchor's depth (`per_metre`, the
    point layer's `m`: frame heights per metre there), or `size` times half the short side."""
    g = extent(ins, output)
    if g["radius"] is not None:
        return g["radius"] * float(per_metre) * size[1]
    return g["size"] * min(size) / 2.0


def fill_radius(size, rim):
    """The radius about the frame's middle at which the circle covers the frame with its vignette (the rim's outer
    `width` of the radius) outside it too: the half diagonal / (1 - width), 2 % to spare."""
    half = math.hypot(size[0], size[1]) / 2.0
    return half / (1.0 - rim["width"]) * FILL_MARGIN if rim else half * FILL_MARGIN


def _span(c, n, o, m, r):
    """(lo, hi): the pixel centres along one axis (a frame `n` pixels long) that a circle of radius `r` about (c, o)
    reaches inside the frame, `o` its place along the other axis (`m` pixels long); None when none of it is inside."""
    d = 0.0 if 0.0 <= o <= m - 1.0 else min(abs(o), abs(o - (m - 1.0)))
    if d >= r:
        return None
    half = math.sqrt(r * r - d * d)
    lo, hi = max(0.0, c - half), min(n - 1.0, c + half)
    return None if hi < lo else (lo, hi)


def cover(size, centre, radius, k):
    """(scale, centre) of a frame-sized picture shown inside a circle (`radius` px about `centre`) in a frame `size` =
    (w, h): at least the scale `k` and at least what covers the part of the circle inside the frame, its centre on the
    circle's as far as it still covers that part (else moved just enough), so it never shows its edge in the circle."""
    w, h = size
    r = radius + 0.5                                         # the pixels its antialiased edge touches
    sx, sy = _span(centre[0], w, centre[1], h, r), _span(centre[1], h, centre[0], w, r)
    if sx is None or sy is None:                             # nothing of it inside the frame
        return k, tuple(centre)
    (x0, x1), (y0, y1) = sx, sy
    k = max(k, (x1 - x0) / (w - 1.0), (y1 - y0) / (h - 1.0))
    hx, hy = k * (w - 1.0) / 2.0, k * (h - 1.0) / 2.0          # the picture's half extent, pixel centre to pixel centre
    return k, (min(max(centre[0], x1 - hx), x0 + hx), min(max(centre[1], y1 - hy), y0 + hy))


def circle(ins, st, output, size, anchor, per_metre):
    """The iris at state `st` (state()) in a frame `size` = (w, h), the anchor projected to `anchor` (px, index
    coordinates) with `per_metre` frame heights per metre at its depth: {"centre", "radius" (px), "k" (the inner picture's
    scale about its own centre), "at" (where that centre lands), "glint" (its strength), "p"}. At rest the circle is the
    rest radius about the anchor and the picture just covers it, centred on it; the opening and the close scale the
    radius only, so the iris uncovers the picture as it opens. An expand moves the centre to the frame's middle, grows
    the radius geometrically to fill_radius and the picture geometrically to the frame (never less than what covers the
    circle's part inside the frame, see cover()), the glint fading as it goes; at p = 1 the frame is the picture."""
    w, h = size
    mid = ((w - 1) / 2.0, (h - 1) / 2.0)
    rim = ins["rim"]
    glint = rim["glint"] if rim else 0.0
    if st["p"] >= 1.0:
        return {"centre": mid, "radius": fill_radius(size, rim), "k": 1.0, "at": mid, "glint": 0.0, "p": 1.0}
    rest = max(rest_radius(ins, output, size, per_metre), 0.0)
    k0 = 2.0 * max(rest, MIN_RADIUS) / min(w, h) * COVER
    if st["phase"] != "expand":                                  # the picture as at rest: the iris uncovers it
        k, at = cover(size, anchor, max(rest, MIN_RADIUS), k0)
        return {"centre": tuple(anchor), "radius": rest * st["open"], "k": k, "at": at, "glint": glint, "p": 0.0}
    p = st["p"]
    centre = (TW.lerp(anchor[0], mid[0], p), TW.lerp(anchor[1], mid[1], p))
    r = TW.geometric(max(rest, MIN_RADIUS), fill_radius(size, rim), p)
    k, at = cover(size, centre, r, TW.geometric(k0, 1.0, p))
    return {"centre": centre, "radius": r, "k": k, "at": at, "glint": glint * (1.0 - p), "p": p}
