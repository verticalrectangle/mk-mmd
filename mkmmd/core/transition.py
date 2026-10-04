"""Cut effects between and over shots, bpy-free (docs/design.md: Shots: Transitions and inserts): the `[[transition]]` and
`[[insert]]` specs, the plan that checks them against the cut, the frames each one needs rendered and the files those
frames are stored in. `mk build` (keys the cameras the effects need), `mk render` (renders the missing layers), `mk post`
and `mk look` (composite them, mkmmd.cutfx) all read the one plan, `plan()`, made from mk.toml alone.

    [[transition]]  at, kind = "expand" | "collapse" | "slash", dur, ease
        expand / collapse: scale = [1, "fill"], turn = [0, 90], center = "subject" | [x, y] | [x, y, z] | "<expression>",
                           edge = {color, width}
        slash:             angle = -20, width = 0.3, color = "love", second = {color, width, offset}, dir = "right"
    [[insert]]      from, to, shot, shape = "thought", anchor, size, offset, ratio, outline = {color, width},
                    pop = {dur, overshoot}, out = "pop" | "expand", expand = {dur, turn, ease}, aspect.<output>

A window is a run of frames over which an effect changes the picture: the `dur` seconds that END at the cut (`at`) for a
transition, the whole `from`..`to` for an insert. Frames of the other shot inside a window are *plates*, rendered next to
the cut's frames (`plate/<shot>/<frame>.png`); the silhouette whose figure is the matte of an expand / collapse is rendered
as `matte/` (its coverage) and `back/` (its frame without the figure); anchors and centres that need Blender (a bone seen
through a camera) are `point/<key>/<frame>.json`."""
import math
from pathlib import Path

from . import shotstyle as SS
from . import tween as TW

KINDS = ("expand", "collapse", "slash")
SHAPES = ("thought",)
SLASH_DIRS = ("right", "left")
MIN_WINDOW = 2                          # frames
LAYER_DIRS = ("plate", "matte", "back", "point")      # folders of an output's frame folder that hold layers
_GEOMETRY = {"size", "offset", "ratio"}  # the insert keys an `aspect.<output>` table may override

_COMMON = {"at", "kind", "dur", "ease"}
TRANSITION_KEYS = {"expand": _COMMON | {"scale", "turn", "center", "edge"},
                   "collapse": _COMMON | {"scale", "turn", "center", "edge"},
                   "slash": _COMMON | {"angle", "width", "color", "second", "dir"}}
INSERT_KEYS = {"from", "to", "shot", "shape", "anchor", "size", "offset", "ratio", "outline", "pop", "out", "expand",
               "aspect"}


class TransitionError(ValueError):
    """A bad [[transition]] / [[insert]] key, or one that does not fit the cut."""


def frame_of(t, fps, frame0):
    """The frame of clip second `t` (the build's rounding)."""
    return int(round(frame0 + float(t) * fps))


# ================================================================================================= normalising specs
def _pair(v, what):
    try:
        a, b = (float(x) for x in v)
    except (TypeError, ValueError):
        raise TransitionError(f"{what} = {v!r}: expected [a, b]") from None
    return [a, b]


def _unknown(spec, allowed, what):
    extra = sorted(set(spec) - allowed)
    if extra:
        raise TransitionError(f"{what}: unknown keys {extra} (known: {sorted(allowed)})")


def _colour(spec, palette, what):
    try:
        return [round(float(v), 6) for v in SS.resolve_colour(spec, palette)]
    except SS.StyleError as e:
        raise TransitionError(f"{what}: {e}") from None


def _ring(spec, palette, what, default_width=4.0):
    """{color, width} -> {"color": rgb, "width": px at 1080 on the short side}."""
    if not isinstance(spec, dict):
        raise TransitionError(f"{what} = {spec!r}: expected {{color, width}}")
    _unknown(spec, {"color", "width"}, what)
    width = float(spec.get("width", default_width))
    if width <= 0:
        raise TransitionError(f"{what}.width must be positive")
    return {"color": _colour(spec.get("color", "text"), palette, f"{what}.color"), "width": width}


def _center(spec):
    """`center`: the figure's centroid, a point of the frame, or a point of the scene seen through the matte shot's camera."""
    if spec is None or spec == "subject":
        return {"mode": "subject"}
    if isinstance(spec, str):
        return {"mode": "point", "expr": spec}
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return {"mode": "frame", "at": _pair(spec, "center")}
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        x, y, z = (float(v) for v in spec)
        return {"mode": "point", "expr": f"({x!r}, {y!r}, {z!r})"}
    raise TransitionError(f"center = {spec!r}: expected 'subject', [x, y] (frame fractions), [x, y, z] (a world point) "
                          f"or an expression such as 'bone(\"spine\").head'")


def normalize_transition(spec, palette):
    """One `[[transition]]` with its defaults, colours resolved (display-space floats)."""
    if not isinstance(spec, dict):
        raise TransitionError(f"transition {spec!r}: expected a table")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise TransitionError(f"transition kind = {kind!r}: expected one of {KINDS}")
    what = f"transition {kind} at {spec.get('at')}"
    _unknown(spec, TRANSITION_KEYS[kind], what)
    if "at" not in spec:
        raise TransitionError(f"{what}: needs `at` (clip seconds of the cut)")
    out = {"kind": kind, "at": float(spec["at"]), "dur": float(spec.get("dur", 0.18 if kind == "slash" else 0.4)),
           "ease": spec.get("ease", "inout" if kind == "slash" else "in")}
    if out["dur"] <= 0:
        raise TransitionError(f"{what}: dur must be positive")
    if out["ease"] not in TW.EASES:
        raise TransitionError(f"{what}: ease = {out['ease']!r}, expected one of {TW.EASES}")
    if kind == "slash":
        out["angle"] = float(spec.get("angle", -20.0))
        out["width"] = float(spec.get("width", 0.3))
        if not 0.0 < out["width"] <= 1.5:
            raise TransitionError(f"{what}: width = {out['width']}: expected a fraction of the frame diagonal in (0, 1.5]")
        out["color"] = _colour(spec.get("color", "love"), palette, f"{what}: color")
        out["dir"] = spec.get("dir", "right")
        if out["dir"] not in SLASH_DIRS:
            raise TransitionError(f"{what}: dir = {out['dir']!r}, expected one of {SLASH_DIRS}")
        out["second"] = None
        if spec.get("second") is not None:
            sec = spec["second"]
            if not isinstance(sec, dict):
                raise TransitionError(f"{what}: second = {sec!r}: expected {{color, width, offset}}")
            _unknown(sec, {"color", "width", "offset"}, f"{what}: second")
            out["second"] = {"color": _colour(sec.get("color", "text"), palette, f"{what}: second.color"),
                             "width": float(sec.get("width", 0.04)), "offset": float(sec.get("offset", -0.03))}
            if out["second"]["width"] <= 0:
                raise TransitionError(f"{what}: second.width must be positive")
        return out
    sc = spec.get("scale", [1.0, "fill"])
    try:
        lo, hi = sc
        lo, hi = float(lo), None if hi == "fill" else float(hi)
    except (TypeError, ValueError):
        raise TransitionError(f"{what}: scale = {sc!r}: expected [lo, hi] or [lo, \"fill\"]") from None
    if lo <= 0.0 or (hi is not None and hi <= lo):
        raise TransitionError(f"{what}: scale = {sc!r}: expected 0 < lo < hi")
    out["scale"] = [lo, hi]                                      # hi None: whatever fills the frame
    out["turn"] = _pair(spec.get("turn", [0.0, 90.0]), "turn")
    out["center"] = _center(spec.get("center"))
    out["edge"] = _ring(spec["edge"], palette, f"{what}: edge") if spec.get("edge") is not None else None
    return out


def normalize_insert(spec, palette):
    """One `[[insert]]` with its defaults, colours resolved."""
    if not isinstance(spec, dict):
        raise TransitionError(f"insert {spec!r}: expected a table")
    what = f"insert {spec.get('from')}-{spec.get('to')}"
    _unknown(spec, INSERT_KEYS, what)
    for k in ("from", "to", "shot", "anchor"):
        if k not in spec:
            raise TransitionError(f"{what}: needs `{k}`")
    out = {"from": float(spec["from"]), "to": float(spec["to"]), "shot": str(spec["shot"]),
           "shape": spec.get("shape", "thought"), "anchor": str(spec["anchor"]), "size": float(spec.get("size", 0.34)),
           "offset": _pair(spec.get("offset", [0.12, -0.30]), "offset"), "ratio": float(spec.get("ratio", 1.35)),
           "outline": _ring(spec.get("outline", {}), palette, f"{what}: outline", 5.0), "aspect": {}}
    if out["to"] <= out["from"]:
        raise TransitionError(f"{what}: `to` must come after `from`")
    if out["shape"] not in SHAPES:
        raise TransitionError(f"{what}: shape = {out['shape']!r}, expected one of {SHAPES}")
    if not 0.0 < out["size"] <= 1.0:
        raise TransitionError(f"{what}: size = {out['size']}: expected a fraction of the frame height in (0, 1]")
    if out["ratio"] <= 0:
        raise TransitionError(f"{what}: ratio must be positive")
    pop = spec.get("pop", {})
    if not isinstance(pop, dict):
        raise TransitionError(f"{what}: pop = {pop!r}: expected {{dur, overshoot}}")
    _unknown(pop, {"dur", "overshoot"}, f"{what}: pop")
    out["pop"] = {"dur": float(pop.get("dur", 0.3)), "overshoot": float(pop.get("overshoot", 0.12))}
    if out["pop"]["dur"] <= 0 or out["pop"]["overshoot"] < 0:
        raise TransitionError(f"{what}: pop.dur must be positive and pop.overshoot not negative")
    out["out"] = spec.get("out", "pop")
    if out["out"] not in ("pop", "expand"):
        raise TransitionError(f"{what}: out = {out['out']!r}, expected 'pop' or 'expand'")
    ex = spec.get("expand", {})
    if not isinstance(ex, dict):
        raise TransitionError(f"{what}: expand = {ex!r}: expected {{dur, turn, ease}}")
    _unknown(ex, {"dur", "turn", "ease"}, f"{what}: expand")
    if ex and out["out"] != "expand":
        raise TransitionError(f"{what}: `expand` only applies to out = 'expand'")
    out["expand"] = {"dur": float(ex.get("dur", 0.4)), "turn": float(ex.get("turn", 0.0)), "ease": ex.get("ease", "in")}
    if out["expand"]["dur"] <= 0 or out["expand"]["ease"] not in TW.EASES:
        raise TransitionError(f"{what}: expand.dur must be positive and expand.ease one of {TW.EASES}")
    for name, over in (spec.get("aspect") or {}).items():
        if not isinstance(over, dict):
            raise TransitionError(f"{what}: aspect.{name} = {over!r}: expected a table")
        _unknown(over, _GEOMETRY, f"{what}: aspect.{name}")
        out["aspect"][name] = {k: (_pair(v, f"aspect.{name}.{k}") if k == "offset" else float(v)) for k, v in over.items()}
    return out


def insert_for(ins, output):
    """The insert's geometry (size, offset, ratio) for one output aspect: its `aspect.<output>` overrides merged in."""
    g = {k: ins[k] for k in _GEOMETRY}
    g.update(ins["aspect"].get(output, {}))
    return g


# ================================================================================================= the plan
def is_silhouette(shot, output):
    """Whether the `[[shot]]` table (raw mk.toml) is a silhouette in the output aspect."""
    merged = dict(shot)
    merged.update((shot.get("aspect") or {}).get(output) or {})
    return merged.get("style") == "silhouette"


def cut_shot(cuts, frame):
    """The name of the shot in the cut at `frame` (SS.shot_at over the plan's cut table), None for an empty cut."""
    e = SS.shot_at(cuts, frame)
    return e["name"] if e else None


def _window_overlap(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def plan(data, fps, frame0, palette):
    """The transitions and inserts of a project (mk.toml `data`) checked against its shots:

        {"fps", "frame0", "cuts": [{name, from, to}] (the cut, in frames), "plates": [shots that are not in the cut],
         "transitions": [...], "inserts": [...]}

    Every transition carries its `cut` frame, window (`n` frames, `first`..`last`), the outgoing and incoming shots
    (`out`, `in`) and which of them gives the figure (`matte`) and which plays inside it (`plate`); every insert its
    `f0`..`f1` (the window is f0 .. f1 - 1), `host` shot and the frame counts of its pop and exit. [] entries when the
    project has none."""
    shots = data.get("shot") or []
    specs = data.get("transition") or []
    inserts = data.get("insert") or []
    if isinstance(specs, dict):
        specs = [specs]
    if isinstance(inserts, dict):
        inserts = [inserts]
    outputs = [o["name"] for o in data.get("output") or []] or ["main"]
    cuts, plates, by_name = [], [], {}
    for s in shots:
        name = s.get("name")
        by_name[name] = s
        if s.get("plate"):
            plates.append(name)
            continue
        if "from" not in s or "to" not in s:
            raise TransitionError(f"shot {name!r}: needs from and to (or plate = true)")
        cuts.append({"name": name, "from": frame_of(s["from"], fps, frame0), "to": frame_of(s["to"], fps, frame0)})
    cuts.sort(key=lambda c: c["from"])
    out = {"fps": float(fps), "frame0": int(frame0), "cuts": cuts, "plates": plates, "transitions": [], "inserts": []}
    windows = []                                                  # (first, last, label) of every effect

    for i, raw in enumerate(specs):
        tr = normalize_transition(raw, palette)
        what = f"transition {i} ({tr['kind']} at {tr['at']} s)"
        cut = frame_of(tr["at"], fps, frame0)
        incoming = [c for c in cuts if c["from"] == cut]
        if len(incoming) != 1:
            starts = ", ".join(f"{(c['from'] - frame0) / fps:.2f}" for c in cuts)
            raise TransitionError(f"{what}: {'no shot' if not incoming else 'several shots'} start at that frame "
                                  f"(cuts at {starts} s)")
        before = SS.shot_at(cuts, cut - 1)
        if before is None or before["name"] == incoming[0]["name"]:
            raise TransitionError(f"{what}: no shot before the cut")
        n = max(MIN_WINDOW, int(round(tr["dur"] * fps)))
        first = cut - n
        if first < before["from"]:
            raise TransitionError(f"{what}: the {n}-frame window starts before shot {before['name']!r} does "
                                  f"(it is {cut - before['from']} frames long); shorten `dur`")
        tr.update(index=i, cut=cut, n=n, first=first, last=cut - 1, out=before["name"], **{"in": incoming[0]["name"]})
        if tr["kind"] == "expand":
            tr["matte"], tr["plate"] = tr["out"], tr["in"]
        elif tr["kind"] == "collapse":
            tr["matte"], tr["plate"] = tr["in"], tr["out"]
        else:
            tr["matte"], tr["plate"] = None, tr["in"]
        if tr["matte"]:
            for o in outputs:
                if not is_silhouette(by_name[tr["matte"]], o):
                    raise TransitionError(f"{what}: shot {tr['matte']!r} must be a silhouette (style = \"silhouette\") "
                                          f"in output {o!r}: its figure is the shape that {tr['kind']}s")
        windows.append((first, cut - 1, what))
        out["transitions"].append(tr)

    for j, raw in enumerate(inserts):
        ins = normalize_insert(raw, palette)
        what = f"insert {j} ({ins['from']}-{ins['to']} s)"
        f0, f1 = frame_of(ins["from"], fps, frame0), frame_of(ins["to"], fps, frame0)
        if ins["shot"] not in by_name:
            raise TransitionError(f"{what}: no shot named {ins['shot']!r}")
        host = SS.shot_at(cuts, f0)
        if host is None or any(f0 < c["from"] < f1 for c in cuts) or f0 < host["from"]:
            raise TransitionError(f"{what}: it must lie inside one shot of the cut (no cut between {f0} and {f1 - 1})")
        if host["name"] == ins["shot"]:
            raise TransitionError(f"{what}: the shot inside the shape is the host shot itself")
        n_in = max(MIN_WINDOW, int(round(ins["pop"]["dur"] * fps)))
        if ins["out"] == "expand":
            starts = [c for c in cuts if c["from"] == f1]
            if not starts or starts[0]["name"] != ins["shot"]:
                got = repr(starts[0]["name"]) if starts else "no shot starts there"
                raise TransitionError(f"{what}: out = 'expand' grows into the shot that starts at `to` ({got}), so `shot` "
                                      f"must be that shot, not {ins['shot']!r}")
            n_exit = max(MIN_WINDOW, int(round(ins["expand"]["dur"] * fps)))
        else:
            n_exit = n_in
        if n_in + n_exit > f1 - f0:
            raise TransitionError(f"{what}: {f1 - f0} frames are too few for a {n_in}-frame pop and a {n_exit}-frame exit")
        ins.update(index=j, f0=f0, f1=f1, host=host["name"], n_in=n_in, n_exit=n_exit)
        windows.append((f0, f1 - 1, what))
        out["inserts"].append(ins)

    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            if _window_overlap(windows[a], windows[b]):
                raise TransitionError(f"{windows[a][2]} and {windows[b][2]} overlap in frames "
                                      f"{max(windows[a][0], windows[b][0])}..{min(windows[a][1], windows[b][1])}")
    return out


# ================================================================================================= what a frame needs
def in_force(pl, shot, frame):
    """Whether `shot` is the shot of the cut at `frame` (then the cut's own frame is its plate)."""
    return cut_shot(pl["cuts"], frame) == shot


def demands(pl, frames=None):
    """The layers to render, {frame: [item, ...]} (frames restricted to `frames` when given). Items:

        {"kind": "plate", "shot": S}               S seen through its own camera and look, when S is not in the cut there
        {"kind": "matte", "shot": S}               S's figure alone (matte/) and S's frame without it (back/)
        {"kind": "point", "key", "expr", "shot"}   the world point `expr` (the `mk q` expression language) seen through
                                                    S's camera, as fractions of the frame (point/<key>/)"""
    want = None if frames is None else set(int(f) for f in frames)
    out = {}

    def add(f, item):
        if want is not None and f not in want:
            return
        bucket = out.setdefault(f, [])
        if item not in bucket:
            bucket.append(item)

    def plate(f, shot):
        if not in_force(pl, shot, f):
            add(f, {"kind": "plate", "shot": shot})

    for tr in pl["transitions"]:
        for f in range(tr["first"], tr["last"] + 1):
            if tr["kind"] == "slash":
                plate(f, tr["plate"])
                continue
            add(f, {"kind": "matte", "shot": tr["matte"]})
            plate(f, tr["plate"])
            if tr["center"]["mode"] == "point":
                add(f, {"kind": "point", "key": f"t{tr['index']}", "expr": tr["center"]["expr"], "shot": tr["matte"]})
    for ins in pl["inserts"]:
        for f in range(ins["f0"], ins["f1"]):
            plate(f, ins["shot"])
            add(f, {"kind": "point", "key": f"i{ins['index']}", "expr": ins["anchor"], "shot": ins["host"]})
    return dict(sorted(out.items()))


def needs(pl):
    """{shot: [first frame, last frame]} a shot's camera must be keyed over for the plates and mattes taken from it."""
    out = {}
    for f, items in demands(pl).items():
        for it in items:
            if it["kind"] in ("plate", "matte"):
                lo, hi = out.get(it["shot"], (f, f))
                out[it["shot"]] = [min(lo, f), max(hi, f)]
    return out


# ================================================================================================= the files
def rel_paths(item, frame):
    """The files (relative to an output's frame folder) an item is stored in. The last one is written last: it is the
    mark of a finished item."""
    f = f"{int(frame):05d}"
    if item["kind"] == "plate":
        return [f"plate/{item['shot']}/{f}.png"]
    if item["kind"] == "matte":
        return [f"back/{item['shot']}/{f}.png", f"matte/{item['shot']}/{f}.png"]
    if item["kind"] == "point":
        return [f"point/{item['key']}/{f}.json"]
    raise ValueError(f"unknown item {item!r}")


def plate_rel(shot, frame):
    return rel_paths({"kind": "plate", "shot": shot}, frame)[0]


def matte_rel(shot, frame):
    return rel_paths({"kind": "matte", "shot": shot}, frame)[1]


def back_rel(shot, frame):
    return rel_paths({"kind": "matte", "shot": shot}, frame)[0]


def point_rel(key, frame):
    return rel_paths({"kind": "point", "key": key}, frame)[0]


def pending_items(frames_dir, dem):
    """[(frame, item)] of the demands `dem` whose finishing file is not in `frames_dir` (missing or empty)."""
    root = Path(frames_dir)
    out = []
    for f, items in dem.items():
        for item in items:
            p = root / rel_paths(item, f)[-1]
            if not p.exists() or p.stat().st_size == 0:
                out.append((f, item))
    return out


def clear_claims(frames_dir):
    """Delete the empty files a stopped render left in the layer folders: claims nobody finished."""
    root = Path(frames_dir)
    for sub in LAYER_DIRS:
        for p in (root / sub).rglob("*"):
            if p.is_file() and p.stat().st_size == 0:
                p.unlink()


# ================================================================================================= timing
def pose(tr, k, hi=None):
    """(scale, turn in degrees) of the matte of an expand / collapse at its window frame `k` (0 .. n - 1). Expand runs
    from the figure itself (scale lo, turn t0) at k = 0 to the frame-filling size (hi, t1) at k = n - 1, collapse is the
    same path walked backwards. `hi` replaces the spec's top scale: mkmmd.cutfx passes the scale that fills the frame
    when the spec says "fill" (None) or its number is too small; scales are interpolated by ratio, turns linearly, both
    by the eased progress."""
    lo, top = tr["scale"]
    top = top if hi is None else hi
    if top is None:
        raise ValueError("scale = [lo, 'fill'] needs the scale that fills the frame: pass hi")
    e = TW.ease(k / (tr["n"] - 1), tr["ease"])
    p = e if tr["kind"] == "expand" else 1.0 - e
    t0, t1 = tr["turn"]
    return TW.geometric(lo, top, p), TW.lerp(t0, t1, p)


def covering_frame(tr):
    """The window frame at which the matte must fill the frame: the last of an expand, the first of a collapse."""
    return tr["last"] if tr["kind"] == "expand" else tr["first"]


def slash_progress(tr, k):
    """0..1 position of a slash's band along its sweep at window frame `k`: never exactly 0 or 1, so the band is on screen in
    every frame of the window (the cut itself is the first frame without it)."""
    return TW.ease((k + 1) / (tr["n"] + 1), tr["ease"])


def insert_state(ins, frame):
    """What an insert shows at `frame` of its window: {"phase": "in" | "hold" | "out" | "expand", "bubble", "dots": [3],
    "p"}. `bubble` and `dots` are the pop scale (1 at rest, over 1 in the overshoot) of the shape and of its three trailing
    circles (0 = the one nearest the anchor), `p` the growth of an expanding shape (0 = as at rest .. 1 = filling the frame)."""
    i = frame - ins["f0"]                                        # frames since the insert began
    j = ins["f1"] - 1 - frame                                    # frames until its last
    over = ins["pop"]["overshoot"]
    if not 0 <= i < ins["f1"] - ins["f0"]:
        return {"phase": "none", "bubble": 0.0, "dots": [0.0] * 3, "p": 0.0}
    if i < ins["n_in"]:
        dots, bubble = pop_scales((i + 1) / ins["n_in"], over)
        return {"phase": "in", "bubble": bubble, "dots": dots, "p": 0.0}
    if ins["out"] == "pop" and j < ins["n_exit"]:
        dots, bubble = pop_scales(j / ins["n_exit"], over)
        return {"phase": "out", "bubble": bubble, "dots": dots, "p": 0.0}
    if ins["out"] == "expand" and j < ins["n_exit"]:
        e = TW.ease((ins["n_exit"] - 1 - j) / (ins["n_exit"] - 1), ins["expand"]["ease"])
        dots, _ = pop_scales(TW.clamp01(1.0 - e / DOTS_GONE), over)
        return {"phase": "expand", "bubble": 1.0, "dots": dots, "p": e}
    return {"phase": "hold", "bubble": 1.0, "dots": [1.0] * 3, "p": 0.0}


DOT_STARTS = (0.0, 0.14, 0.28)           # when each trailing circle starts to pop, as shares of the pop (0 = nearest the anchor)
DOT_SPAN = 0.30                          # how much of the pop each circle takes
BUBBLE_START = 0.45                      # when the shape itself starts to pop
DOTS_GONE = 0.35                         # an expanding shape has lost its circles by this share of the growth


def pop_scales(u, overshoot):
    """([scale of each trailing circle], scale of the shape) at pop progress u (0..1): the circles pop one after another,
    the nearest the anchor first, then the shape; every one is a springy 0 -> 1 that overshoots by `overshoot`."""
    dots = [TW.pop(TW.stagger(u, s, DOT_SPAN), overshoot) for s in DOT_STARTS]
    return dots, TW.pop(TW.stagger(u, BUBBLE_START, 1.0 - BUBBLE_START), overshoot)


def summary(pl):
    """A short JSON-able description of the plan for reports."""
    return {"transitions": [{"index": t["index"], "kind": t["kind"], "at": t["at"], "frames": [t["first"], t["last"]],
                             "out": t["out"], "in": t["in"]} for t in pl["transitions"]],
            "inserts": [{"index": s["index"], "frames": [s["f0"], s["f1"] - 1], "host": s["host"], "shot": s["shot"],
                         "out": s["out"]} for s in pl["inserts"]]}
