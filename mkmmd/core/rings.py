"""Rings, bpy-free (docs/design.md: Looks: vector, rings): `[[ring]]` entries are discs and bands that sweep out from a
point of the frame or of the scene, flipping a vector shot's picture to its look's `opposite` palette inside them. A
pixel inside an odd number of rings is flipped and inside an even number it is not, so two rings that overlap cut back
into each other; a thin edge runs on each wavefront. A `hold` ring is a disc that keeps the frame flipped once it has
swept past it, until the next ring starts.

The Blender half (mkmmd.blender.styles) projects each live ring's centre through the shot's camera on every frame and
hands `shapes` the centres as fractions of the frame; the shots stage keeps the normalised rings in scene["mk_rings"]."""
import math

from . import shotstyle as SS


class RingError(ValueError):
    """A bad `[[ring]]`."""


KEYS = {"at", "center", "dur", "width", "hold", "edge", "ease"}
EASES = ("out", "linear", "inout")
EDGE = {"color": "#FFFFFF", "width": 3.0}                        # the wavefront: px wide at 1080 on the short side


def _center(spec, what):
    if isinstance(spec, str):
        return {"mode": "point", "expr": spec}
    if isinstance(spec, (list, tuple)) and len(spec) in (2, 3):
        try:
            v = [float(x) for x in spec]
        except (TypeError, ValueError):
            v = None
        if v is not None:
            return {"mode": "frame", "at": v} if len(v) == 2 else {"mode": "point", "expr": f"({v[0]!r}, {v[1]!r}, {v[2]!r})"}
    raise RingError(f"{what}: center = {spec!r}: expected [x, y] (frame fractions from the top left), [x, y, z] (a world "
                    f"point) or an expression such as 'obj(\"boombox\").matrix_world.translation'")


def _edge(spec, palette, what):
    if spec is False:
        return None
    spec = {} if spec is None or spec is True else spec
    if not isinstance(spec, dict) or set(spec) - {"color", "width"}:
        raise RingError(f"{what}: edge = {spec!r}: expected {{color, width}} or false")
    width = float(spec.get("width", EDGE["width"]))
    if width <= 0:
        raise RingError(f"{what}: edge width must be positive")
    try:
        color = [round(float(c), 6) for c in SS.resolve_colour(spec.get("color", EDGE["color"]), palette)]
    except SS.StyleError as e:
        raise RingError(f"{what}: edge {e}") from None
    return {"color": color, "width": width}


def normalize(specs, palette):
    """The `[[ring]]` entries, sorted by `at`: {index, at, center, dur, width, hold, ease, edge, until}, `until` the clip
    second the ring is gone (a band: when it has swept past the frame; a held disc: when the next ring starts)."""
    out = []
    for i, spec in enumerate(specs or []):
        what = f"ring {i}"
        if not isinstance(spec, dict):
            raise RingError(f"{what} = {spec!r}: expected a table")
        unknown = sorted(set(spec) - KEYS)
        if unknown:
            raise RingError(f"{what}: unknown keys {unknown} (known: {sorted(KEYS)})")
        for k in ("at", "center"):
            if k not in spec:
                raise RingError(f"{what}: needs `{k}`")
        dur, width = float(spec.get("dur", 0.6)), float(spec.get("width", 0.25))
        if dur <= 0 or width <= 0:
            raise RingError(f"{what}: dur and width must be positive")
        ease = spec.get("ease", "out")
        if ease not in EASES:
            raise RingError(f"{what}: ease = {ease!r}: one of {', '.join(EASES)}")
        out.append({"index": i, "at": float(spec["at"]), "center": _center(spec["center"], what), "dur": dur,
                    "width": width, "hold": bool(spec.get("hold", False)), "ease": ease,
                    "edge": _edge(spec.get("edge"), palette, what)})
    out.sort(key=lambda r: r["at"])
    for k, r in enumerate(out):
        nxt = out[k + 1]["at"] if k + 1 < len(out) else math.inf
        r["until"] = nxt if r["hold"] else r["at"] + r["dur"]
    return out


def live(rings, t):
    """The rings on screen at clip second `t`."""
    return [r for r in rings if r["at"] <= t < r["until"]]


def _ease(u, kind):
    u = min(max(u, 0.0), 1.0)
    if kind == "linear":
        return u
    if kind == "inout":
        return u * u * (3.0 - 2.0 * u)
    return 1.0 - (1.0 - u) ** 2


def shapes(rings, t, centres, size):
    """The discs and bands of the rings live at clip second `t`, in pixels of a frame `size` = (w, h): [{c, outer, inner,
    edge}], `c` the centre, `outer` and `inner` radii (a disc has inner 0), `edge` the wavefront's {color, width} or None.
    `centres` maps a ring's index to its centre as frame fractions from the top left (None: not on this frame, a point
    behind the camera). A band sweeps out until its inner rim has passed the frame's farthest corner from the centre, a
    held disc until it covers the frame, `ease`d over `dur`; `width` is in frame heights."""
    w, h = size
    out = []
    for r in live(rings, t):
        at = centres.get(r["index"])
        if at is None:
            continue
        c = (at[0] * w, at[1] * h)
        far = max(math.hypot(c[0] - x, c[1] - y) for x in (0.0, w) for y in (0.0, h))
        band = r["width"] * h
        u = (t - r["at"]) / r["dur"]
        if r["hold"]:
            outer = _ease(u, r["ease"]) * far
            out.append({"c": c, "outer": outer, "inner": 0.0, "edge": r["edge"] if u < 1.0 else None})
        else:
            outer = _ease(u, r["ease"]) * (far + band)
            out.append({"c": c, "outer": outer, "inner": max(outer - band, 0.0), "edge": r["edge"]})
    return out


def masks(shapes_, size, scale):
    """(flip, edges) at `scale` times the frame `size` = (w, h), pixel centres sampled: `flip` (H, W) bool where an odd
    number of the shapes lie, `edges` [(mask, rgb)] their wavefronts (`edge.width` px at 1080 on the short side)."""
    import numpy as np
    w, h = size
    W, H = int(round(w * scale)), int(round(h * scale))
    ys = ((np.arange(H) + 0.5) / scale)[:, None]
    xs = ((np.arange(W) + 0.5) / scale)[None, :]
    flip = np.zeros((H, W), bool)
    edges = []
    px = min(w, h) / 1080.0
    for s in shapes_:
        d = np.hypot(xs - s["c"][0], ys - s["c"][1])
        flip ^= (d < s["outer"]) & (d >= s["inner"])
        if s["edge"] is not None and s["outer"] > 0:
            edges.append((np.abs(d - s["outer"]) <= s["edge"]["width"] * px / 2.0, s["edge"]["color"]))
    return flip, edges
