"""Rewinds, bpy-free (docs/design.md: Frame effects, Rewinds): `[[fx]] kind = "rewind"` checked and timed, its scrub bar
laid out. The picture work is mkmmd/rewind.py.

A finger scrubs the video back. A paper scrub bar is slapped down onto the picture on `land` (craft type's slap): a strip
of masking tape that is the clip from end to end (`bar`), pen ticks a second apart (every fifth longer), the part before
the playhead painted; a round die-cut sticker on it is the playhead (`head`), a piece of label tape beside it carries a
raised double arrow, the rewind mark (`mark`), and a paper tag the clip second shown, digits only (`counter`).

The picture is the cut's own until the first segment of the time `map`. Each segment {t, from, to} shows the clip's own
finished picture at source seconds `from` down to `to`, from clip second `t` until the next segment starts (the last until
the span's end), in poses held `fps` a second counted from `t` (its first pose shows `from`, its last `to`), with a hard
jump between segments, as a scrub jumps. Those pictures are the cut's frames with the effects before the rewind in the
table on them as they are at those seconds (a push's frame inside is the precedent), so they carry no type: mk post lays
type on after the frame effects, on the frame it finishes. The playhead stands on the bar where the second shown is (the
clip's own second before the first segment) and squashes under the finger from `press`; a cutout riding it, `at =
{rewind = <the rewind's from second>}`, keeps its fingertip on it (`grip`). Each segment's start tears the picture as a
tape shuttling does (split's torn bands, a ghost of the picture before), the tear dying away over `decay` s and a little
of it (`base`) staying while the scrub runs; those jumps come three a second at most (mkmmd.core.eralook.FLASH_RATE).

Every motion is a function of the frame: any frame comes out the same in any order."""
import math

from . import crafttype as CT
from . import eralook as ER

KINDS = ("rewind",)
KEYS = {"rewind": {"map", "fps", "snap", "land", "press", "bar", "head", "mark", "counter", "tear", "shadow", "seed"}}
DEFAULTS = {"rewind": {"fps": 12.0, "snap": 0.25, "land": None, "press": None, "bar": None, "head": None, "mark": None,
                       "counter": None, "tear": None, "shadow": None, "seed": 1}}
# The parts' own keys and defaults: `at` frame fractions, sizes frame heights (a bar's length frame widths), px at 1080.
BAR = {"at": [0.07, 0.775], "length": 0.71, "height": 0.03, "tilt": -1.2, "color": "surface:gold:0.38",
       "played": "love", "ink": "text", "ticks": 1.0}
HEAD = {"size": 0.056, "color": "love", "rim": 14.0}
MARK = {"at": None, "size": 0.042, "color": "iris", "ink": "base", "tilt": -4.0}
COUNTER = {"at": None, "size": 0.026, "color": "surface", "ink": "text", "font": None, "tilt": 2.5}
TEAR = {"base": 0.15, "decay": 0.12, "ghost": 0.25, "linger": 0.033, "bands": 14, "shift": 0.06, "share": 0.35,
        "rgb": 8.0, "hold": 0.05}
MARK_FROM_BAR = (0.075, -0.062)          # frame heights from the bar's left end: the mark's middle, above it
COUNTER_FROM_MARK = 0.135                # frame heights: the counter's middle right of the mark's
PRESS = {"squash": 0.12, "td": 0.06, "tp": 0.16, "held": 0.9}   # the playhead under the finger: a squash that rings
#                                          out (craft type's spring) to `held` of its size, pressed flat on the paper
TICK_EVERY = 5                           # every fifth tick is a long one


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def _part(v, defaults, what, palette, colours=(), allow_false=False):
    """A part's table (`bar`, `head`, `mark`, `counter`, `tear`) with its defaults under it, its colours resolved; None
    (left out) gives the defaults, false (where `allow_false`) none of the part."""
    FX = _fx()
    if v is False and allow_false:
        return None
    if v is None:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what} = {v!r}: expected a table of {sorted(defaults)}" +
                         (" or false (none)" if allow_false else ""))
    extra = sorted(set(v) - set(defaults))
    if extra:
        raise FX.FxError(f"{what}: unknown keys {extra} (known: {sorted(defaults)})")
    out = {**defaults, **v}
    for k in colours:
        if out[k] is not False:
            out[k] = FX._colour(out[k], palette, f"{what} {k}")
    return out


def normalize(spec, out, palette, what):
    """A rewind's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`). Its timing
    against the clip (the map's frames, its sources inside the clip, the land and the press) is `timed`'s."""
    FX = _fx()
    segs = out.get("map")
    if not isinstance(segs, (list, tuple)) or not segs or not all(isinstance(s, dict) for s in segs):
        raise FX.FxError(f"{what}: `map` is required: the time map, [{{t, from, to}}, ...] (clip seconds: from `t` on, "
                         f"source seconds `from` down to `to`)")
    rows = []
    for s in segs:
        if set(s) != {"t", "from", "to"}:
            raise FX.FxError(f"{what}: map entry {s!r}: expected {{t, from, to}} (clip seconds)")
        rows.append({k: FX._num(s[k], f"{what} map {k}", lo=0.0) for k in ("t", "from", "to")})
    if any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
        raise FX.FxError(f"{what}: map: the segments' `t` rise ({[r['t'] for r in rows]})")
    for r in rows:
        if max(r["from"], r["to"]) > r["t"] + 1e-9:
            raise FX.FxError(f"{what}: map entry at {r['t']:g} s shows {r['from']:g}..{r['to']:g} s: a rewind shows "
                             f"seconds that came before, at most its own `t`")
    if not out["from"] <= rows[0]["t"] < out["to"] or rows[-1]["t"] >= out["to"]:
        raise FX.FxError(f"{what}: map: the segments start inside the span ({out['from']:g}..{out['to']:g} s), "
                         f"at {[r['t'] for r in rows]}")
    jumps = [r["t"] for r in rows]
    n, t0 = ER.busiest(jumps)
    if n > ER.FLASH_RATE:
        raise FX.FxError(f"{what}: map: {n} jumps in the second from {t0:g} s, each a tear: at most {ER.FLASH_RATE} "
                         f"a second (nothing strobes)")
    out["map"] = rows
    out["fps"] = FX._num(out["fps"], f"{what} fps", 0.0, 120.0)
    out["snap"] = FX._num(out["snap"], f"{what} snap", 0.0, 1.0)
    first = rows[0]["t"]
    land = CT.ARRIVE["slap"]["fall"] + out["from"] if out["land"] is None else FX._num(out["land"], f"{what} land")
    press = first if out["press"] is None else FX._num(out["press"], f"{what} press")
    if not out["from"] <= land <= press <= first:
        raise FX.FxError(f"{what}: from ({out['from']:g}) <= land ({land:g}: the bar lands) <= press ({press:g}: the "
                         f"finger presses the playhead) <= the first segment's t ({first:g})")
    out["land"], out["press"] = land, press
    bar = _part(out["bar"], BAR, f"{what} bar", palette, ("color", "played", "ink"))
    bar["at"] = FX._pair(bar["at"], f"{what} bar at")
    for k, lo, hi in (("length", 0.05, 1.0), ("height", 0.004, 0.2), ("tilt", -30.0, 30.0), ("ticks", 0.0, 60.0)):
        bar[k] = FX._num(bar[k], f"{what} bar {k}", lo, hi)
    out["bar"] = bar
    head = _part(out["head"], HEAD, f"{what} head", palette, ("color",))
    head["size"] = FX._num(head["size"], f"{what} head size", 0.005, 0.5)
    head["rim"] = FX._num(head["rim"], f"{what} head rim", 0.0, 100.0)
    out["head"] = head
    mark = _part(out["mark"], MARK, f"{what} mark", palette, ("color", "ink"), allow_false=True)
    if mark is not None:
        mark["at"] = None if mark["at"] is None else FX._pair(mark["at"], f"{what} mark at")
        mark["size"] = FX._num(mark["size"], f"{what} mark size", 0.005, 0.5)
        mark["tilt"] = FX._num(mark["tilt"], f"{what} mark tilt", -45.0, 45.0)
    out["mark"] = mark
    counter = _part(out["counter"], COUNTER, f"{what} counter", palette, ("color", "ink"), allow_false=True)
    if counter is not None:
        counter["at"] = None if counter["at"] is None else FX._pair(counter["at"], f"{what} counter at")
        counter["size"] = FX._num(counter["size"], f"{what} counter size", 0.005, 0.5)
        counter["tilt"] = FX._num(counter["tilt"], f"{what} counter tilt", -45.0, 45.0)
        if counter["font"] is not None and (not isinstance(counter["font"], str) or not counter["font"].strip()):
            raise FX.FxError(f"{what}: counter font = {counter['font']!r}: a font slug or file (none: the registry's "
                             f"first `mono` font)")
    out["counter"] = counter
    tear = _part(out["tear"], TEAR, f"{what} tear", palette, allow_false=True)
    if tear is not None:
        for k, lo, hi in (("base", 0.0, 1.0), ("decay", 1e-3, 10.0), ("ghost", 0.0, 1.0), ("linger", 0.0, 10.0),
                          ("shift", 0.0, 1.0), ("share", 0.0, 1.0), ("rgb", 0.0, 200.0), ("hold", 1e-3, 10.0)):
            tear[k] = FX._num(tear[k], f"{what} tear {k}", lo, hi)
        tear["bands"] = FX._num(tear["bands"], f"{what} tear bands", 1, 400, integer=True)
    out["tear"] = tear
    sh = out["shadow"]
    out["shadow"] = None if sh is False else FX._shadow({**CT.SHADOW, **sh} if isinstance(sh, dict) else
                                                        CT.SHADOW if sh is None else sh, palette, what)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


def timed(fx, fps, frame0, last_frame):
    """A rewind against the clip, at plan time: its length (the bar is the clip from end to end: `[project] duration`
    is needed), each segment's frames (`f` its first, `end` past its last, `g0` and `g1` the source frames of its
    `from` and `to`, inside the clip) and the frames the bar lands and the finger presses on."""
    FX = _fx()
    what = f"fx {fx['index']} (rewind)"
    if last_frame is None:
        raise FX.FxError(f"{what}: its bar is the clip from end to end: give [project] duration")
    fx["duration"] = (last_frame - frame0 + 1) / float(fps)
    fx["frame_rate"], fx["frame0"] = float(fps), int(frame0)
    starts = [FX.frame_of(s["t"], fps, frame0) for s in fx["map"]]
    ends = starts[1:] + [fx["f1"]]
    for s, a, b in zip(fx["map"], starts, ends):
        if b <= a:
            raise FX.FxError(f"{what}: the segment at {s['t']:g} s is under a frame long")
        s.update(f=a, end=b, g0=FX.frame_of(s["from"], fps, frame0), g1=FX.frame_of(s["to"], fps, frame0))
        if not frame0 <= min(s["g0"], s["g1"]) or (max(s["g0"], s["g1"]) > last_frame):
            raise FX.FxError(f"{what}: the segment at {s['t']:g} s shows {s['from']:g}..{s['to']:g} s: outside the clip")
    fx["land_frame"] = FX.frame_of(fx["land"], fps, frame0)
    fx["press_frame"] = FX.frame_of(fx["press"], fps, frame0)


def check_windows(fxs, windows):
    """No rewind shows a frame inside a cut effect's window (`windows` [(first frame, last frame, what)]): it shows the
    rendered frames with the frame effects before it, never a transition's, an insert's or a glitch's composite."""
    FX = _fx()
    for fx in fxs:
        if fx["kind"] not in KINDS:
            continue
        for s in fx["map"]:
            lo, hi = sorted((s["g0"], s["g1"]))
            for a, b, what in windows:
                if lo <= b and a <= hi:
                    raise FX.FxError(f"fx {fx['index']} (rewind): the segment at {s['t']:g} s shows {s['from']:g}.."
                                     f"{s['to']:g} s, into {what}: a rewind shows the rendered frames with the effects "
                                     f"before it, not the cut effects: choose seconds outside the window")


# ================================================================================================= the time map
def segment_at(fx, frame):
    """The segment shown on `frame` (its index into `map`), or -1 before the first."""
    k = -1
    for i, s in enumerate(fx["map"]):
        if frame >= s["f"]:
            k = i
    return k


def _pose(fx, s, frame):
    """(pose index, poses in segment `s`) on `frame` inside it: poses `fps` a second, counted from its first frame."""
    if fx["fps"] <= 0:
        return frame - s["f"], s["end"] - s["f"]
    per = fx["frame_rate"] / fx["fps"]                       # frames a pose is held
    k = int(math.floor((frame - s["f"]) / per + 1e-6))
    return k, int(math.floor((s["end"] - 1 - s["f"]) / per + 1e-6)) + 1


def shown(fx, frame):
    """(the clip second the picture on `frame` is of, its frame, the segment's index or -1): before the first segment
    the frame's own; inside a segment its pose's, its first pose `from` and its last `to`, evenly between."""
    i = segment_at(fx, frame)
    if i < 0:
        return (frame - fx["frame0"]) / fx["frame_rate"], frame, -1
    s = fx["map"][i]
    k, n = _pose(fx, s, frame)
    u = 0.0 if n <= 1 else min(k / (n - 1.0), 1.0)
    sec = s["from"] + (s["to"] - s["from"]) * u
    g = s["g0"] + int(round((s["g1"] - s["g0"]) * u)) if n > 1 else s["g0"]
    return sec, g, i


def pose_start(fx, frame):
    """The first frame of the pose shown on `frame` (the frame itself before the first segment)."""
    i = segment_at(fx, frame)
    if i < 0:
        return frame
    s = fx["map"][i]
    k, _ = _pose(fx, s, frame)
    if fx["fps"] <= 0:
        return s["f"] + k
    per = fx["frame_rate"] / fx["fps"]
    return max(s["f"] + int(math.ceil(k * per - 1e-6)), s["f"])


def tear_at(fx, frame):
    """How hard the picture is torn on `frame`, 0..1: each segment's start a full tear dying away as exp(-age / decay),
    `base` while the scrub runs, nothing before it (or without a `tear`)."""
    if fx["tear"] is None:
        return 0.0
    i = segment_at(fx, frame)
    if i < 0:
        return 0.0
    age = (frame - fx["map"][i]["f"]) / fx["frame_rate"]
    return max(fx["tear"]["base"], math.exp(-age / fx["tear"]["decay"]))


def ghost_at(fx, frame):
    """(the frame whose picture shows as a ghost on `frame`, its opacity), or (None, 0): only just after a jump, the
    picture before it (the previous segment's last pose, the cut's own frame before the first segment) at `ghost`
    fading to nothing over `linger` s, so the new pose is clean two frames after its jump."""
    tr = fx["tear"]
    i = segment_at(fx, frame)
    if tr is None or tr["ghost"] <= 0 or i < 0:
        return None, 0.0
    f = fx["map"][i]["f"]
    a = tr["ghost"] * (1.0 - (frame - f) / fx["frame_rate"] / tr["linger"]) if tr["linger"] > 0 else 0.0
    return (shown(fx, f - 1)[1], a) if a > 1e-6 else (None, 0.0)


def sources(fx, frame, frame0):
    """The frames of the cut a rewind reads on `frame`: the pose's (and the ghost's just after a jump), none before
    its first segment."""
    if segment_at(fx, frame) < 0:
        return []
    out = [shown(fx, frame)[1]]
    g = ghost_at(fx, frame)[0]
    if g is not None and g not in out:
        out.append(g)
    return [f for f in out if f >= frame0]


def placed(fx, frame):
    """The clip second the playhead stands at on `frame`. Before the first segment the second shown (the clip's own);
    from it, as a finger drags: steadily from the second shown just before the scrub to the last segment's `to`, pose
    by pose with the picture, and `snap` of the way from that steady line to the second shown (1: the second shown,
    every jump taken whole; 0: a steady drag alone), so it snaps a little on each jump. The counter shows the second
    shown."""
    sec, _, i = shown(fx, frame)
    if i < 0 or fx["snap"] >= 1.0:
        return sec
    a = fx["map"][0]["f"] - 1
    s0, s1 = shown(fx, a)[0], shown(fx, fx["f1"] - 1)[0]
    u = (pose_start(fx, frame) - a) / max(pose_start(fx, fx["f1"] - 1) - a, 1)
    steady = s0 + (s1 - s0) * min(max(u, 0.0), 1.0)
    return steady + fx["snap"] * (sec - steady)


# ================================================================================================= the scrub bar
def slap(fx, frame, sign=1.0):
    """A piece slapped down on the bar's `land` frame as craft type's `slap` (mkmmd.core.crafttype ARRIVE): (sx, sy,
    turn degrees, dy frame heights (down), lift 0..1), or None before its fall begins."""
    a = CT.ARRIVE["slap"]
    u = (frame - fx["land_frame"]) / fx["frame_rate"]
    if u < -a["fall"] - 1e-9:
        return None
    if u < 0:                                                    # coming down: faster and faster
        q = max(1.0 + u / a["fall"], 0.0) ** 2
        s = a["big"] + (1.0 - a["big"]) * q
        return s, s, a["spin"] * (1.0 - q) * sign, -a["drop"] * (1.0 - q), \
            min(1.0, (1.0 - q) * (a["big"] - 1.0) / 0.5 + 0.15)
    sq = CT.spring(u, a["squash"], a["td"], a["tp"])               # the contact: a squash that rings out
    return 1.0 + sq, 1.0 - sq, CT.spring(u, 0.35 * a["spin"], 2.0 * a["td"], 1.6 * a["tp"]) * sign, 0.0, 0.0


def _affine(cx, cy, sx, sy, deg):
    """The 2 x 3 map from a piece's own px (about its middle, y down) to the frame's: scaled on its own axes, turned
    `deg` counter-clockwise on screen, its middle at (cx, cy)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c * sx, s * sy, cx], [-s * sx, c * sy, cy]]


def bar_map(fx, frame, size):
    """The bar on `frame` in a frame of `size` (w, h): (M, lift), M the 2 x 3 map from its own px (its middle, x along
    it, y down) to the frame's px, or None before it comes down. Its length is `length` frame widths (the clip from end
    to end), its middle line at `at`'s height, its left end at `at`'s x."""
    st = slap(fx, frame, 1.0)
    if st is None:
        return None
    w, h = float(size[0]), float(size[1])
    b = fx["bar"]
    cx = (b["at"][0] + 0.5 * b["length"]) * w - 0.5
    cy = b["at"][1] * h - 0.5 + st[3] * h
    return _affine(cx, cy, st[0], st[1], b["tilt"] + st[2]), st[4]


def playhead(fx, frame, size):
    """The playhead on `frame`: (centre px (x, y) in a frame of `size`, the clip second it stands at, `placed`), or None
    before the bar comes down: 0 at the bar's left end, the clip's length at the right."""
    got = bar_map(fx, frame, size)
    if got is None:
        return None
    M, _ = got
    sec = placed(fx, frame)
    x = (min(max(sec / fx["duration"], 0.0), 1.0) - 0.5) * fx["bar"]["length"] * float(size[0])
    return (M[0][0] * x + M[0][2], M[1][0] * x + M[1][2]), sec


def grip(fx, frame, size):
    """Where a cutout riding the playhead puts its anchor on `frame` (px): the playhead's centre, the frame taken into
    the rewind's span (a hand there before the bar comes down meets it where it lands)."""
    f = min(max(frame, fx["f0"], fx["land_frame"]), fx["f1"] - 1)
    return playhead(fx, f, size)[0]


def pressed(fx, frame):
    """How the playhead stands under the finger on `frame`: (scale, 0..1 how flat on the paper it is) from `press`: it
    squashes on the touch and rings out to PRESS["held"] of its size; before it, at rest (1, 0)."""
    u = (frame - fx["press_frame"]) / fx["frame_rate"]
    if u < 0:
        return 1.0, 0.0
    return PRESS["held"] - CT.spring(u, PRESS["squash"], PRESS["td"], PRESS["tp"]), 1.0


def piece_at(fx, part, size):
    """The middle (px) of the `mark` or the `counter` in a frame of `size`: its `at`, or above the bar's left end (the
    mark), the counter right of the mark."""
    w, h = float(size[0]), float(size[1])
    p = fx[part]
    if p["at"] is not None:
        return p["at"][0] * w - 0.5, p["at"][1] * h - 0.5
    b = fx["bar"]
    mx, my = b["at"][0] * w - 0.5 + MARK_FROM_BAR[0] * h, b["at"][1] * h - 0.5 + MARK_FROM_BAR[1] * h
    if part == "mark":
        return mx, my
    if fx["mark"] is not None and fx["mark"]["at"] is not None:
        mx, my = fx["mark"]["at"][0] * w - 0.5, fx["mark"]["at"][1] * h - 0.5
    return mx + COUNTER_FROM_MARK * h, my


def piece_map(fx, part, frame, size, sign):
    """(M, lift) of the `mark` or the `counter` on `frame`, slapped down with the bar, or None before."""
    st = slap(fx, frame, sign)
    if st is None:
        return None
    cx, cy = piece_at(fx, part, size)
    return _affine(cx, cy + st[3] * float(size[1]), st[0], st[1], fx[part]["tilt"] + st[2]), st[4]


def counter_text(sec):
    """The counter's digits for clip second `sec`: minutes, seconds and tenths (`0:18.4`), never below 0."""
    d = max(int(math.floor(max(sec, 0.0) * 10.0 + 1e-6)), 0)
    return f"{d // 600}:{(d // 10) % 60:02d}.{d % 10}"


def ticks(duration, every):
    """The bar's ticks: [(clip second, long)], one each `every` seconds from 0 to the clip's length, every
    TICK_EVERY-th long; none for `every` 0."""
    if every <= 0:
        return []
    n = int(math.floor(duration / every + 1e-6))
    return [(k * every, k % TICK_EVERY == 0) for k in range(n + 1)]
