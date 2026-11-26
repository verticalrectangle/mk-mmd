"""Video stickers, bpy-free (docs/design.md: Frame effects, Video stickers): `[[fx]] kind = "clip"` and `kind = "cutout"`
checked and laid out. The picture work is mkmmd/videosticker.py.

A video clip cut out in a shape and laid over the picture as a die-cut sticker: the picture stays and the clip is the
sticker, where a `footage` collage lays a figure over a clip that fills the frame. The clip is read and fitted by
mkmmd.footage with a footage collage's own keys (`clip`, `in`, `rate`, `loop`, `focus`, `mirror`), always to cover the
sticker's box; with `crop` a part of its frame alone is the sticker's video (a notification lifted off a screen
recording). Its `shape`: a rounded `rect` (its corners `radius` of the box's short side), a `circle` (an ellipse by
the box), a `heart` or a `star`. The box stands with its middle at `at` (frame fractions), `size` frame heights tall and
`ratio` times as wide (by default the shape's own proportions), and `keys` move it about its middle as they move a
collage's figure (mkmmd.core.framefx `move_at`, `move_matrix`). A die-cut `edge` is grown round the shape and the sticker
casts a drop `shadow`, both as a collage's and both as much larger as a key scales the sticker. It is laid `over` the
picture (an opaque sticker), or its face is light, `screen`ed or `add`ed onto it so that the clip's black vanishes
(night light trails): its edge then stays paper, laid over, and only that rim casts the shadow. A `negative` sticker
shows no clip: its face is the picture under it inverted (type that answers whatever is behind it), laid as an opaque
sticker is.

A `cutout` is a clip cut out by its own matte (mkmmd.footage: `<stem>.matte<suffix>` beside it), frame by frame: a real
hand photographed and cut out like a zine's, its shape changing as it moves. Its whole frame (or its `crop`) is the box,
`size` frame heights tall in its own proportions, and its point `anchor` (a fingertip, the scissors' crossing) stands at
`at`, a frame point or a scene point (`{point = ...}`, the camera of the shot in the cut on each frame: a hand riding
something that moves), turned `turn` degrees and moved by `keys` about that point. A hand that moves in its own frame (it
rocks, it reaches) takes its fingertip with it: `anchor` is then that point's track over the video's seconds
(`anchor_at`), and the fingertip stays on `at` while the arm moves. It plays stepped: `fps` poses a second,
each held (`clock`), and `sync` = [video second, clip second] puts a moment of the video (a contact, a snip) on a beat,
a step starting exactly there. It is a sticker like a sheet's: it wears the sheet's white `edge` and casts the sheet's
`shadow` unless its own keys say otherwise (`false`: none). A light `grade` sits its colours in the palette: lifted
toward a paper colour, warmed, a little less saturated; a clip named by a registry slug brings the grade stored with it
(mkmmd.project), so one clip shows one skin in every project.

Outlines are polygons in the box's own pixels about its middle, x right and y down, touching its four sides."""
import math

import numpy as np

from . import mandala as MD

KINDS = ("clip", "cutout")
OUTLINED = ("rect", "circle", "heart", "star")             # shapes with an outline (a polygon) and proportions of their own
SHAPES = OUTLINED + ("text",)                              # and keyed type: the letters of a text are the shape
BLENDS = ("over", "screen", "add", "negative")   # an opaque sticker, a face of light (the clip's black vanishes), or the
                                                 # picture under it inverted, no clip
VIDEO = ("clip", "in", "rate", "loop", "focus", "mirror", "crop")  # the clip's own keys: a negative sticker has none
STAR = (0.5, 0.11)                      # a star's inner radius over its outer, its corners' rounding over its outer
STEP = 1.5                              # px: an outline's points lie about this far apart along its curves
KEYS = {"clip": {"clip", "in", "rate", "loop", "focus", "mirror", "crop", "shape", "radius", "at", "size", "ratio",
                 "keys", "edge", "shadow", "blend", "text", "font"},
        "cutout": {"clip", "in", "rate", "loop", "mirror", "crop", "fps", "sync", "anchor", "at", "size", "turn", "keys",
                   "edge", "shadow", "grade", "extend"}}
DEFAULTS = {"clip": {"in": 0.0, "rate": 1.0, "loop": False, "focus": [0.5, 0.5], "mirror": False, "crop": None,
                     "shape": "rect", "radius": 0.12, "at": [0.5, 0.5], "size": 0.4, "ratio": None, "keys": None,
                     "edge": None, "shadow": None, "blend": "over", "text": None, "font": None},
            "cutout": {"in": 0.0, "rate": 1.0, "loop": False, "focus": [0.5, 0.5], "mirror": False, "crop": None,
                       "fps": 12.0, "sync": None, "anchor": [0.5, 0.5], "at": [0.5, 0.5], "size": 1.0, "turn": 0.0,
                       "keys": None, "edge": None, "shadow": None, "grade": None, "extend": True}}
GRADE = {"gain": 1.0, "lift": 0.08, "warm": 1.0, "saturation": 0.9, "toward": "base"}   # a cutout's grade, `grade =
#                                   {}`: lifted 8 % toward the paper, warmed (WARM at 1), a touch less saturated
WARM = (1.06, 1.015, 0.92)          # a cutout's warm grade at 1: its red, green and blue gains (0: none)
PAPER = (0.78, 0.10)                # near-white paper a grade leaves as shot: luma from this up, chroma under this
RIDES = ("peel", "tear", "sticker", "table", "rewind")  # what a cutout's `at` may ride: a peel transition's lifted
#                                     corner, a tear's grips, the grip of a sticker peeled off a sheet, a point of a table
#                                     (a sheet with a clip, mkmmd.core.table) carried by its view, a rewind's playhead
FPS_MAX = 120.0


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A video sticker's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    if out["kind"] == "cutout":
        _cutout(spec, out, palette, what)
        return
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    if out["blend"] == "negative":                       # its face is the picture under it: there is no video to read
        given = [k for k in VIDEO if k in spec]
        if given:
            raise FX.FxError(f"{what}: a negative sticker shows the picture under it inverted: give it no "
                             f"{', '.join(given)}")
    else:
        FX._clip_keys(out, what)
        if out["crop"] is not None:
            out["crop"] = FX._part(out["crop"], f"{what}: crop")
    if out["shape"] not in SHAPES:
        raise FX.FxError(f"{what}: shape {out['shape']!r} is none of {', '.join(SHAPES)}")
    if out["shape"] != "rect" and "radius" in spec:
        raise FX.FxError(f"{what}: radius rounds a rect's corners; a {out['shape']} has none")
    if out["shape"] == "text":                           # the clip shows through the letters of `text`
        if not isinstance(out["text"], str) or not out["text"].strip():
            raise FX.FxError(f"{what}: a text sticker needs `text`, the words its letters show the clip through")
        if "ratio" in spec:
            raise FX.FxError(f"{what}: a text sticker takes its proportions from its letters: give it no ratio")
        if out["font"] is not None and not isinstance(out["font"], str):
            raise FX.FxError(f"{what}: font is a font file or a font slug of the asset registry")
    elif "text" in spec or "font" in spec:
        raise FX.FxError(f"{what}: text and font are a text sticker's (shape = \"text\")")
    out["radius"] = FX._num(out["radius"], f"{what} radius", 0.0, 0.5)
    out["at"] = FX._pair(out["at"], f"{what} at")
    out["size"] = FX._num(out["size"], f"{what} size", 0.01, 10.0)
    if out["shape"] != "text":
        out["ratio"] = own_ratio(out["shape"]) if out["ratio"] is None else FX._num(out["ratio"], f"{what} ratio", 0.05,
                                                                                     20.0)
    out["keys"] = FX._moves(out["keys"], what)
    out["edge"] = MD._edge(FX, out["edge"], palette, what)
    out["shadow"] = FX._shadow(out["shadow"], palette, what)
    if out["blend"] in ("screen", "add") and out["shadow"] is not None and (out["edge"] is None or
                                                                            out["edge"]["width"] <= 0):
        raise FX.FxError(f"{what}: a {out['blend']} sticker's face is light and casts no shadow: give it an `edge` (its "
                         f"rim casts the shadow) or no `shadow`")


def _cutout(spec, out, palette, what):
    """A cutout's own keys checked and filled in, in place: its video (`in` or a `sync`), its stepping, where its anchor
    stands, its turn and moves, its edge, shadow and grade."""
    FX = _fx()
    rate = FX._num(out["rate"], f"{what} rate", -16.0, 16.0)        # below 0 the video plays backwards (a lift: a press)
    out["rate"] = abs(rate)
    FX._clip_keys(out, what)
    out["rate"] = rate
    if out["crop"] is not None:
        out["crop"] = FX._part(out["crop"], f"{what}: crop")
    out["fps"] = FX._num(out["fps"], f"{what} fps", 0.0, FPS_MAX)
    if out["sync"] is not None:
        if "in" in spec:
            raise FX.FxError(f"{what}: give `in` (the video's second on the span's first frame) or `sync` ([video second, "
                             f"clip second]: that moment of the video on that frame), not both")
        v, t = FX._pair(out["sync"], f"{what} sync")
        if v < 0:
            raise FX.FxError(f"{what}: sync = {out['sync']}: [video second (0 or more), clip second]")
        out["sync"] = [v, t]
        out["in"] = v + out["rate"] * (out["from"] - t)          # the video's second on the span's first frame
    out["anchor"] = _anchor(out["anchor"], f"{what} anchor")
    out["at"] = _at(out["at"], f"{what} at")
    if not isinstance(out["extend"], bool):
        raise FX.FxError(f"{what}: extend is true or false")
    out["size"] = FX._num(out["size"], f"{what} size", 0.01, 10.0)
    out["turn"] = FX._num(out["turn"], f"{what} turn")
    out["keys"] = FX._moves(out["keys"], what)
    out["edge"] = MD._edge(FX, _sheets(out["edge"], {k: FX.ST.EDGE[k] for k in ("color", "width")}), palette, what)
    out["shadow"] = FX._shadow(_sheets(out["shadow"], FX.ST.SHADOW), palette, what)
    out["grade"] = _grade(out["grade"], palette, what)


def _sheets(v, own):
    """A cutout's `edge` or `shadow`: a cut-out is a sticker, so the sticker sheet's `own` (mkmmd.core.sheet EDGE,
    SHADOW) with the entry's keys over it, all of it when the entry gives none; false: none. Anything else is left for
    the key's own check to refuse."""
    if v is False:
        return None
    if v is None:
        return dict(own)
    return {**own, **v} if isinstance(v, dict) else v


def _anchor(v, what):
    """A cutout's `anchor`: [x, y], or the track of a point that moves in the clip's own frame (a fingertip while the hand
    rocks): [[video second, x, y], ...], two rows or more, their seconds rising."""
    FX = _fx()
    if not (isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple))):
        return FX._pair(v, what)
    if len(v) < 2 or not all(isinstance(r, (list, tuple)) and len(r) == 3 for r in v):
        raise FX.FxError(f"{what} = {v!r}: a track is [[video second, x, y], ...], two rows or more")
    rows = [[FX._num(c, what) for c in r] for r in v]
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise FX.FxError(f"{what}: a track's video seconds rise from row to row ({[r[0] for r in rows]})")
    return rows


def anchor_at(fx, video):
    """A cutout's anchor (fractions of its clip's frame as the file has it) at the video's second `video`: its point, or
    its track read there, straight between rows and the end rows' beyond them."""
    a = fx["anchor"]
    if not isinstance(a[0], list):
        return a[0], a[1]
    v = [r[0] for r in a]
    return float(np.interp(video, v, [r[1] for r in a])), float(np.interp(video, v, [r[2] for r in a]))


def _at(v, what):
    """A cutout's `at`: [x, y] or {point = expr} (mkmmd.core.framefx._center), or a grab point it rides: a transition's,
    {peel = <its `at` second>} (the lifted corner) or {tear = <its `at` second>, grip = 0 or 1} (a grip on a half),
    {sticker = <second>}, the grip of the sticker a sheet peels off from that second (mkmmd.core.sheet.grip_of), or
    {table = [x, y]}, a point of the table (fractions of its clip's frame) under the cutout's span: the table's view
    carries it, and zooms and turns the cutout with the table (its `size` and its keys' moves are then table heights),
    or {rewind = <its `from` second>}, the playhead of that rewind (mkmmd.core.rewind.grip): a finger dragging it."""
    FX = _fx()
    ride = [k for k in RIDES if isinstance(v, dict) and k in v]
    if not ride:
        return FX._center(v, what)
    kind = ride[0]
    extra = sorted(set(v) - {kind} - ({"grip"} if kind == "tear" else set()))
    if len(ride) > 1 or extra:
        raise FX.FxError(f"{what} = {v!r}: expected {{peel = <second>}}, {{tear = <second>, grip = 0 or 1}} or "
                         f"{{sticker = <second>}} (or {{table = [x, y]}}: a point of a table, or {{rewind = <second>}}: "
                         f"a rewind's playhead)")
    if kind == "table":
        return {"mode": kind, "at": FX._pair(v[kind], f"{what} table"), "tr": None}
    grip = v.get("grip", 0)
    if grip not in (0, 1) or isinstance(grip, bool):
        raise FX.FxError(f"{what}: grip = {grip!r}: 0 or 1, the tear's two hands")
    return {"mode": kind, "t": FX._num(v[kind], f"{what} {kind}"), "grip": int(grip), "tr": None}


def ride(fxs, transitions, fps):
    """Each cutout (and its aspect.<output> entries) whose `at` rides something gets it in `at["tr"]`: the transition of
    its kind at that second (within half a frame), for a `sticker` the sheet whose own peel or whose hollow's peel
    starts then ({"sheet": its index, "hollow": the hollow's place or None}), for a `table` the last table in the table
    whose span holds the cutout's ({"sheet": its index}), for a `rewind` the rewind whose span starts then, earlier in
    the table than the cutout so that the finger lies over its bar ({"rewind": its index}); a TransitionError-worthy
    FxError else."""
    FX = _fx()
    peels = [(s["index"], i, p["t"]) for s in fxs if s["kind"] == "sheet"
             for i, p in [(None, s["peel"])] + [(i, h["peel"]) for i, h in enumerate(s["hollows"])] if p is not None]
    for fx in fxs:
        for e in [fx, *(fx.get("aspect") or {}).values()]:
            a = e.get("at") if e["kind"] == "cutout" else None
            if not isinstance(a, dict) or a["mode"] not in RIDES:
                continue
            if a["mode"] == "table":
                got = [s["index"] for s in fxs if s["kind"] == "sheet" and s.get("clip") is not None
                       and s["f0"] <= e["f0"] and e["f1"] <= s["f1"]]
                if not got:
                    raise FX.FxError(f"fx {e['index']} (cutout): at rides a table, and no table (a sheet with a `clip`) "
                                     f"spans its {e['from']:g}..{e['to']:g} s")
                a["tr"] = {"sheet": got[-1]}
                continue
            if a["mode"] == "rewind":
                got = [r["index"] for r in fxs if r["kind"] == "rewind" and abs(r["from"] - a["t"]) <= 0.5 / fps]
                if not got:
                    raise FX.FxError(f"fx {e['index']} (cutout): at rides the rewind from {a['t']:g} s; the rewinds "
                                     f"start at {[r['from'] for r in fxs if r['kind'] == 'rewind'] or 'none'}")
                if got[0] > e["index"]:
                    raise FX.FxError(f"fx {e['index']} (cutout): it rides the playhead of fx {got[0]} (rewind), which "
                                     f"comes after it in the table: put the cutout after the rewind, over its bar")
                a["tr"] = {"rewind": got[0]}
                continue
            if a["mode"] == "sticker":
                got = [{"sheet": s, "hollow": i} for s, i, t in peels if abs(t - a["t"]) <= 0.5 / fps]
                if not got:
                    raise FX.FxError(f"fx {e['index']} (cutout): at rides the sticker peeled from {a['t']:g} s; the "
                                     f"sheets' peels start at {sorted(t for _, _, t in peels) or 'none'}")
                a["tr"] = got[0]
                continue
            got = [tr for tr in transitions if tr["kind"] == a["mode"] and abs(tr["at"] - a["t"]) <= 0.5 / fps]
            if not got:
                have = [tr["at"] for tr in transitions if tr["kind"] == a["mode"]]
                raise FX.FxError(f"fx {e['index']} (cutout): at rides the {a['mode']} at {a['t']:g} s; the "
                                 f"{a['mode']} transitions are at {have or 'none'}")
            a["tr"] = got[0]


def _grade(v, palette, what):
    """A cutout's `grade`: None or false (as shot) or {gain, lift, warm, saturation, toward} with GRADE's defaults (`{}`
    all of them)."""
    FX = _fx()
    if v is None or v is False:
        return None
    if not isinstance(v, dict) or set(v) - set(GRADE):
        raise FX.FxError(f"{what}: grade = {v!r}: expected {{gain, lift, warm, saturation, toward}} ({{}}: the defaults)")
    g = {**GRADE, **v}
    return {"gain": FX._num(g["gain"], f"{what} grade gain", 0.2, 4.0),
            "lift": FX._num(g["lift"], f"{what} grade lift", 0.0, 1.0),
            "warm": FX._num(g["warm"], f"{what} grade warm", -2.0, 2.0),
            "saturation": FX._num(g["saturation"], f"{what} grade saturation", 0.0, 2.0),
            "toward": FX._colour(g["toward"], palette, f"{what} grade toward")}


def clock(fx, t, frame_rate=None):
    """A cutout at clip second `t`: (the clip second its pose is held from, the video's second shown). Its steps are
    1 / `fps` long, counted from its `sync` clip second (else from its span's start), that second taken on the output's
    frame of it (`frame_rate`, the build's rounding: mkmmd.core.framefx.frame_of) so a step starts on the very frame the
    beat falls on; `fps` 0 holds nothing. The video runs at `rate` from `in` (or from the sync's video second there)."""
    sync = fx["sync"]
    t0 = sync[1] if sync is not None else fx["from"]
    if frame_rate:
        t0 = round(t0 * frame_rate) / frame_rate
    held = t if fx["fps"] <= 0 else t0 + math.floor((t - t0) * fx["fps"] + 1e-6) / fx["fps"]
    return held, (sync[0] if sync is not None else fx["in"]) + fx["rate"] * (held - t0)


def place_cutout(fx, t, size, ratio, at, video, view=None):
    """Where a cutout stands at clip second `t` (its held pose's, `clock`) in a frame of `size` (w, h), its anchor at the
    frame pixel `at`: (M, box, scale) as `place` gives them. Its box (its clip's frame or crop, `ratio` its width over its
    height) is `size` frame heights tall; M carries the box's own px (about its middle) so that its `anchor` (fractions of
    the clip's frame as the file has it, mirrored with it; a track's point at the pose's video second `video`, `clock`'s)
    lands on `at`, turned `turn` and moved by its `keys` about it. `view` (frame heights per table height, turn degrees):
    a cutout riding a table, its size and its keys' moves in table heights, turned with the table."""
    FX = _fx()
    H = float(size[1]) * (1.0 if view is None else view[0])
    bh = fx["size"] * H
    bw = float(ratio) * bh
    ax, ay = anchor_at(fx, video)
    if fx["mirror"]:
        ax = 1.0 - ax
    a = ((ax - 0.5) * bw, (ay - 0.5) * bh)
    mv = dict(FX.move_at(fx["keys"], t))
    mv["turn"] += fx["turn"] + (0.0 if view is None else view[1])
    M = FX.compose(FX.move_matrix(mv, at, H), [[1.0, 0.0, at[0] - a[0]], [0.0, 1.0, at[1] - a[1]]])
    return M, (bw, bh), float(mv["scale"])


def points(fx):
    """[(key, expression)]: a cutout's `at` when it is a scene point (key x<i>)."""
    if fx["kind"] == "cutout" and fx["at"]["mode"] == "point":
        return [(_fx().point_key(fx), fx["at"]["expr"])]
    return []


def _unit(shape, n):
    """A heart (mkmmd.core.mandala's, its point down: `n` points) or a star (five points, the first up; its ten corners
    rounded, `n` points each) about (0, 0), x right and y down."""
    if shape == "heart":
        return MD._heart(n)
    a = np.radians(np.arange(10) * 36.0)
    rad = np.where(np.arange(10) % 2 == 0, 0.5, 0.5 * STAR[0])
    return MD._rounded(np.stack([rad * np.sin(a), -rad * np.cos(a)], 1), 0.5 * STAR[1], n)


def own_ratio(shape):
    """A shape's own width over its height, a sticker's `ratio` by default: 1 for a rect and a circle (a square, a round
    circle), a heart's and a star's proportions."""
    if shape in ("rect", "circle"):
        return 1.0
    p = _unit(shape, 720 if shape == "heart" else 24)
    span = p.max(0) - p.min(0)
    return round(float(span[0] / span[1]), 4)


def outline(shape, w, h, radius=0.0, step=STEP):
    """The outline of a sticker `shape` filling a box `w` x `h` px: (n, 2) points about the box's middle, x right and y
    down, touching its four sides, about `step` px apart along its curves; a rect's corners rounded by `radius` of the
    box's short side (0 square .. 0.5, its short sides half circles)."""
    w, h, step = float(w), float(h), max(float(step), 1e-3)
    if shape == "rect":
        r = min(max(float(radius), 0.0), 0.5) * min(w, h)
        if r < 0.25:
            return np.array([[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]])
        u = np.linspace(0.0, 0.5 * math.pi, max(int(math.ceil(0.5 * math.pi * r / step)), 2))
        corners = ((w / 2.0 - r, h / 2.0 - r), (r - w / 2.0, h / 2.0 - r), (r - w / 2.0, r - h / 2.0),
                   (w / 2.0 - r, r - h / 2.0))                       # bottom right, bottom left, top left, top right
        return np.concatenate([np.stack([cx + r * np.cos(u + 0.5 * math.pi * k), cy + r * np.sin(u + 0.5 * math.pi * k)],
                                        1) for k, (cx, cy) in enumerate(corners)])
    if shape == "circle":
        a, b = w / 2.0, h / 2.0
        around = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))      # Ramanujan's perimeter
        n = 4 * max(int(math.ceil(around / step / 4.0)), 4)            # a point on each end of both axes
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return np.stack([a * np.cos(t), b * np.sin(t)], 1)
    if shape not in ("heart", "star"):
        raise ValueError(f"no outline for a sticker shape {shape!r} (outlined: {', '.join(OUTLINED)})")
    big = max(w, h)
    n = max(int(math.ceil(4.3 * big / step)), 64) if shape == "heart" else \
        max(int(math.ceil(math.pi * 0.5 * STAR[1] * big / step)), 3)    # the heart's fastest stretch; a corner's arc
    p = _unit(shape, n)
    lo, hi = p.min(0), p.max(0)
    return (p - (lo + hi) / 2.0) / (hi - lo) * (w, h)


def place(fx, t, size, ratio=None):
    """Where a video sticker stands at clip second `t` in a frame of `size` (w, h): (M, box, scale). M is the 2 x 3 map
    from its box's own px (about its middle, at rest) to the frame's px (pixel i centred on i): its middle at `at`, moved
    by its `keys` there (turned and scaled about its middle, then shifted); box (w, h) its size at rest, px; scale the
    keys' scale there. `ratio` (width over height) is a text sticker's, from its letters; other shapes have their own."""
    FX = _fx()
    W, H = float(size[0]), float(size[1])
    bh = fx["size"] * H
    c = (fx["at"][0] * W - 0.5, fx["at"][1] * H - 0.5)
    mv = FX.move_at(fx["keys"], t)
    M = FX.compose(FX.move_matrix(mv, c, H), [[1.0, 0.0, c[0]], [0.0, 1.0, c[1]]])
    return M, ((fx["ratio"] if ratio is None else ratio) * bh, bh), float(mv["scale"])
