"""Editorial labels, bpy-free (docs/design.md: Frame effects, Labels): `[[fx]] kind = "label"` checked and timed. The
picture work is mkmmd/labels.py.

Tiny type set beside the picture, the way a magazine spread or a museum plate annotates it: a few characters ("1967",
"SIDE A", a track number) in a small grotesk or typewriter face from the asset registry, each with a dotted leader line
running to the point of the scene it names (`at`: [x, y] of the frame or {point = "<expression>"}, projected through the
camera of the shot in the cut on every frame, as a doodle's marks are) and a dot there. A label types itself out from
its `t` and its leader draws out to the point after it. The text is the entry's own: never words of the song."""
import math

KINDS = ("label",)
KEYS = {"label": {"labels", "font", "size", "color", "leader", "tracking", "box", "type", "draw"}}
DEFAULTS = {"label": {"font": None, "size": 0.02, "color": "text", "leader": True, "tracking": 1.0, "box": False,
                      "type": 0.25, "draw": 0.2}}
LEADER = {"color": None, "dot": 2.2, "gap": 6.0, "end": 3.5}        # px at 1080: dot size, gap between, the end dot
BOX = {"color": "base", "pad": 5.0, "alpha": 0.85}                  # px at 1080 round the text
LABEL_KEYS = {"text", "at", "place", "offset", "t", "align", "color", "size"}
ALIGNS = ("left", "right")
MAX_TEXT = 40
OFFSET = [0.07, -0.06]          # frame heights from its point to a label's text by default: up and to the right
EDGE = 0.025                    # frame heights a label's text keeps from the frame's edges (it slides in, in any output)


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A label entry's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    out["size"] = FX._num(out["size"], f"{what} size", 0.004, 0.5)
    out["tracking"] = FX._num(out["tracking"], f"{what} tracking", -10.0, 60.0)
    out["type"] = FX._num(out["type"], f"{what} type", 0.0, 10.0)
    out["draw"] = FX._num(out["draw"], f"{what} draw", 0.0, 10.0)
    out["color"] = FX._colour(out["color"], palette, f"{what} color")
    if out["font"] is not None and not isinstance(out["font"], str):
        raise FX.FxError(f"{what}: font = {out['font']!r}: a font slug of the asset registry or a font file")
    out["leader"] = _table(FX, out["leader"], LEADER, palette, f"{what} leader", out["color"])
    out["box"] = _table(FX, out["box"], BOX, palette, f"{what} box", out["color"])
    labels = spec.get("labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise FX.FxError(f"{what}: `labels` is required: a list of {{text, at, place or offset, t, align}}")
    out["labels"] = [_label(FX, j, v, out, palette, f"{what} labels[{j}]") for j, v in enumerate(labels)]


def _table(FX, v, base, palette, what, colour):
    """A `leader` or a `box`: true (its defaults), false (none) or a table of its keys."""
    if v is False or v is None:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict) or set(v) - set(base):
        raise FX.FxError(f"{what} = {v!r}: true, false or {{{', '.join(base)}}}")
    o = {**base, **v}
    out = {"color": colour if o["color"] is None else FX._colour(o["color"], palette, f"{what} color")}
    for k in base:
        if k != "color":
            out[k] = FX._num(o[k], f"{what} {k}", 0.0, 400.0)
    return out


def _label(FX, j, v, entry, palette, what):
    if not isinstance(v, dict):
        raise FX.FxError(f"{what} = {v!r}: a table {{text, at, place or offset, t, align, color, size}}")
    unknown = sorted(set(v) - LABEL_KEYS)
    if unknown:
        raise FX.FxError(f"{what}: unknown keys {unknown} (known: {sorted(LABEL_KEYS)})")
    text = v.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT or "\n" in text:
        raise FX.FxError(f"{what}: text is a short line of the entry's own (at most {MAX_TEXT} characters)")
    if "at" not in v:
        raise FX.FxError(f"{what}: `at` is required: the point the leader runs to, [x, y] of the frame or "
                         f"{{point = \"<expression>\"}}")
    out = {"text": text, "at": FX._center(v["at"], f"{what} at")}
    if "place" in v and "offset" in v:
        raise FX.FxError(f"{what}: give `place` ([x, y] of the frame) or `offset` (frame heights from its point)")
    out["place"] = FX._pair(v["place"], f"{what} place") if "place" in v else None
    out["offset"] = FX._pair(v.get("offset", OFFSET), f"{what} offset")
    out["t"] = FX._num(v.get("t", entry["from"]), f"{what} t")
    if not entry["from"] <= out["t"] < entry["to"]:
        raise FX.FxError(f"{what}: t = {out['t']:g} is outside the entry's span ({entry['from']:g} .. {entry['to']:g})")
    align = v.get("align")
    if align is not None and align not in ALIGNS:
        raise FX.FxError(f"{what}: align {align!r} is none of {', '.join(ALIGNS)} (default: away from its point)")
    out["align"] = align
    out["color"] = FX._colour(v["color"], palette, f"{what} color") if "color" in v else entry["color"]
    out["size"] = FX._num(v["size"], f"{what} size", 0.004, 0.5) if "size" in v else entry["size"]
    return out


def point_key(fx, j):
    return f"x{fx['index']}l{j}"


def points(fx):
    """[(key, expression)]: the scene points a label entry's leaders run to (x<i>l<j>)."""
    return [(point_key(fx, j), lb["at"]["expr"]) for j, lb in enumerate(fx["labels"]) if lb["at"]["mode"] == "point"]


def shown(label, t, typing):
    """How many characters of a label are up at clip second `t`: none before its `t`, then typed out over `typing`
    seconds."""
    if t < label["t"]:
        return 0
    n = len(label["text"])
    if typing <= 0:
        return n
    return min(n, int(math.floor((t - label["t"]) / typing * n + 1e-9)) + 1)


def drawn(label, t, typing, draw):
    """0..1 of a label's leader drawn out at clip second `t`: from the label to its point over `draw` seconds, once
    the label is typed."""
    u = t - label["t"] - typing
    if u < 0:
        return 0.0
    return 1.0 if draw <= 0 else min(u / draw, 1.0)


def layout(label, point, size, h):
    """Where a label's text sits, from its point (x, y) px in a frame `h` px tall: (anchor (x, y) px of its baseline's
    start or end, its side "left" (the text runs right from the anchor) or "right")."""
    if label["place"] is not None:
        q = (label["place"][0] * size[0] - 0.5, label["place"][1] * size[1] - 0.5)
    else:
        q = (point[0] + label["offset"][0] * h, point[1] + label["offset"][1] * h)
    side = label["align"] or ("left" if q[0] >= point[0] else "right")
    return q, side
