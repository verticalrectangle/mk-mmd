"""Craft type, the numbers (docs/design.md: Text, Craft type): bpy-free. A `[[text]]` with `screen`, `lyrics` and `craft`
is a lyric line made by hand and laid on the picture by `mk post` (mkmmd/crafttype.py draws it): embossed label tape,
handwriting on a strip of masking tape, a die-cut letter sticker, cut-paper labels, brush letters. Nothing of it is built in
Blender.

Its words arrive in UNITS, never one at a time: the whole line (`units = "line"`, the default), each word (`"words"`), or
chunks, `[{words = [i, j], t, at, tilt, scale, to, leave}, ...]`, ranges of the line's words that arrive together on the
first word's onset or on `t` (clip seconds). A unit is one rigid piece: drawn once at rest and carried by one map per
frame, so its letters keep their places relative to each other while it arrives, sits and leaves.

This module is the maths: the entry's keys checked, the units chosen and timed (a unit lands on the frame a word would,
`floor(t * fps + 0.4)`), where they stand (stacked rows fitted in the entry's screen panel, or each at its own `at`), and
how each moves on every frame (`state`): the arrival of its craft (slap, write, paint), the ink's reveal, its settle a few
degrees off level, and its leaving (peel, drop, cut). Lengths are in frame heights (the screen type's unit), times in clip
seconds, frames are clip frames (0 at the clip's start). Errors are TextErrors that name keys and (line, word) numbers,
never the words."""
import math

from . import screentype as SR
from . import wordtype as LY
from .shotstyle import StyleError, resolve_colour
from .typeset import TextError

KINDS = ("emboss", "masking", "sticker", "cutpaper", "brush")
STYLES = ("slap", "write", "paint", "none")
LEAVES = ("cut", "peel", "drop", "wipe")
ENTRY_KEYS = {"name", "screen", "at", "box", "size", "fit", "align", "offset", "font", "craft", "under", "lyrics",
              "aspect", "extends", "abstract", "lead"}
LYRICS_KEYS = {"timeline", "line", "words", "case", "clean", "units", "to", "leave", "gap", "stagger"}
UNIT_KEYS = {"words", "t", "at", "tilt", "scale", "to", "leave", "shift", "rows", "late"}
LATE = 0.25                     # s: a unit hand-timed further than this after its first word is sung is refused
CRAFT_KEYS = {"kind", "color", "ink", "colors", "tape", "rim", "shadow", "arrive", "tilt", "tracking", "seed",
              "alpha", "halo", "jitter", "dry", "write", "rows"}

# What each craft is by default: its colours (palette slots), how it arrives, how far off level it settles (degrees, a
# seeded draw in +-tilt), its die-cut rim (px at 1080; 0: none) and the letter case it sets.
DEFAULTS = {
    "emboss": {"color": "love", "ink": "base", "style": "slap", "tilt": 3.0, "rim": 0.0, "case": "upper",
               "tracking": 0.12, "halo": 0.55},
    "masking": {"color": "surface:gold:0.22", "ink": "text", "style": "write", "tilt": 2.5, "rim": 0.0, "case": "keep",
                "tracking": 0.0, "alpha": 0.88, "write": 0.4},
    "sticker": {"colors": ["love", "pine", "gold", "iris", "foam", "rose"], "rim": 16.0, "style": "slap", "tilt": 3.0,
                "case": "keep", "tracking": 0.02},
    "cutpaper": {"color": ["surface", "overlay", "base"], "ink": "text",
                 "colors": ["love", "gold", "pine", "rose", "foam", "iris"],
                 "tape": "surface:gold:0.22", "style": "slap", "tilt": 2.0, "rim": 0.0, "case": "upper",
                 "tracking": 0.06, "jitter": 4.0},
    "brush": {"color": "love", "style": "paint", "tilt": 2.0, "rim": 0.0, "case": "keep", "tracking": 0.0,
              "dry": 0.45, "write": 0.3},
}
SHADOW = {"color": "text", "offset": [0.004, 0.007], "blur": 6.0, "alpha": 0.3}   # the sheet's: frame heights, px at 1080
RIM_COLOR = "#ffffff"

# The arrivals. A unit's moment is its time less the entry's `lead`: the frame `floor(moment * fps + 0.4)` is its contact
# frame, the first at its resting place. slap: before it, over `fall` s, the unit comes down from `big` times its size,
# faster and faster; on it, it squashes (a damped spring of its two axes, `squash` deep, `td` / `tp` its decay and
# period) and settles to its tilt from `spin` degrees further round (the same spring). write / paint: the pen or the
# brush starts `lead` s before the moment, so it runs ahead of the voice (the strip of a `write` slaps on lightly just
# before it: its contact is the pen's start); each word is laid at the speed of one go (the whole unit in the craft's
# `write` s) and never later than its own sung moment (its onset less the entry's lead): faster where it must be.
ARRIVE = {
    "slap": {"big": 1.55, "fall": 0.08, "squash": 0.14, "td": 0.07, "tp": 0.17, "spin": 9.0, "drop": 0.04},
    "write": {"big": 1.12, "fall": 0.05, "squash": 0.06, "td": 0.05, "tp": 0.14, "spin": 4.0, "drop": 0.01,
              "lead": 0.15},
    "paint": {"big": 1.0, "fall": 0.0, "squash": 0.0, "td": 0.05, "tp": 0.14, "spin": 0.0, "drop": 0.0, "lead": 0.15},
    "none": {"big": 1.0, "fall": 0.0, "squash": 0.0, "td": 0.05, "tp": 0.14, "spin": 0.0, "drop": 0.0},
}
LEAD = 0.08               # s: every unit is on screen this much before its sung moment (an eye needs a moment to find it)
# The leavings, from `to`. peel, a hand's timing: a corner lifts slowly over `lift` s and rolls back over a cylinder,
# the roll is pulled fast across the unit over `pull` s, and the unit is flicked off toward that corner over `flick`
# s. drop: it falls `g` frame heights/s^2, turning `spin` degrees a second, gone after `life` s, fading over its
# second half. wipe (paint): a thumb smears it off sideways over `life` s. cut: gone on the frame of `to`.
LEAVE = {
    "peel": {"lift": 0.3, "pull": 0.12, "flick": 0.16, "corner": "auto"},
    "drop": {"g": 8.0, "spin": 60.0, "life": 0.5},
    "wipe": {"life": 0.3},
    "cut": {},
}
CORNERS = ("auto", "top_left", "top_right", "bottom_left", "bottom_right")
ROW_GAP = 0.35            # caps between stacked rows
FIT = 0.9                 # share of the panel the stacked units may fill


# ------------------------------------------------------------------------------------------------------ small helpers
def _num(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise TextError(f"{what} = {v!r}: expected a number")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise TextError(f"{what} = {v:g}: expected a number in [{lo if lo is not None else '-inf'}, "
                        f"{hi if hi is not None else 'inf'}]")
    return v


def _pair(v, what):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise TextError(f"{what} = {v!r}: expected [x, y]")
    return [_num(v[0], what), _num(v[1], what)]


def _colour(v, palette, what):
    try:
        return tuple(round(float(c), 6) for c in resolve_colour(v, palette))
    except StyleError as e:
        raise TextError(f"{what}: {e}") from None


def _colours(v, palette, what):
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, (list, tuple)) or not items:
        raise TextError(f"{what} = {v!r}: expected a colour or a list of them")
    return [_colour(c, palette, what) for c in items]


def eo3(x):
    return LY.eo3(x)


def ease_io(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def spring(t, a, td, tp):
    """A damped spring released from `a` at t = 0: a e^(-t/td) cos(2 pi t / tp)."""
    return a * math.exp(-t / td) * math.cos(2.0 * math.pi * t / tp) if t >= 0 else a


# ------------------------------------------------------------------------------------------------------ the entry
def is_craft(spec):
    """Whether a `[[text]]` entry is craft type (drawn by mk post, never built in Blender)."""
    return isinstance(spec, dict) and spec.get("craft") is not None


def entries(texts, outputs):
    """[(name, {output: spec})]: the project's craft entries (`extends` resolved, `abstract` left out), each as every
    output sees it (`aspect.<output>` laid over it)."""
    out = []
    for e in SR.resolve_entries([t for t in texts or [] if isinstance(t, dict)]):
        if not is_craft(e):
            continue
        name = e.get("name")
        if not isinstance(name, str) or not name:
            raise TextError("a craft [[text]] needs a `name`")
        if e.get("screen") in (None, False):
            raise TextError(f"text {name!r}: craft type is laid on the picture: give it `screen` (true or a band)")
        try:
            out.append((name, dict(SR.per_output(e, list(outputs)))))
        except TextError as err:
            raise TextError(f"text {name!r}: {err}") from None
    return out


def craft_spec(c, palette, what):
    """The `craft` table checked, with its kind's defaults and its colours resolved (display-space floats)."""
    if not isinstance(c, dict):
        raise TextError(f"{what}: craft = {{kind = ...}} is a table")
    extra = sorted(set(c) - CRAFT_KEYS)
    if extra:
        raise TextError(f"{what}: unknown craft keys {extra} (known: {sorted(CRAFT_KEYS)})")
    kind = c.get("kind")
    if kind not in KINDS:
        raise TextError(f"{what}: craft kind {kind!r} is none of {', '.join(KINDS)}")
    d = DEFAULTS[kind]
    o = {"kind": kind}
    for k in ("color", "ink", "tape"):                           # `color` may be a list: cycled over the units
        v = c.get(k, d.get(k))
        if k == "color" and isinstance(v, (list, tuple)):
            o[k] = _colours(v, palette, f"{what} craft color")
        else:
            o[k] = None if v is None else _colour(v, palette, f"{what} craft {k}")
    first = c.get("color", d.get("color", "text"))
    o["colors"] = _colours(c.get("colors", d.get("colors", first if isinstance(first, list) else [first])), palette,
                           f"{what} craft colors")
    o["tilt"] = _num(c.get("tilt", d["tilt"]), f"{what} craft tilt", 0.0, 45.0)
    o["tracking"] = _num(c.get("tracking", d["tracking"]), f"{what} craft tracking", -0.3, 1.0)
    o["seed"] = int(_num(c.get("seed", 1), f"{what} craft seed"))
    o["alpha"] = _num(c.get("alpha", d.get("alpha", 1.0)), f"{what} craft alpha", 0.05, 1.0)
    o["halo"] = _num(c.get("halo", d.get("halo", 0.0)), f"{what} craft halo", 0.0, 1.0)
    o["jitter"] = _num(c.get("jitter", d.get("jitter", 0.0)), f"{what} craft jitter", 0.0, 20.0)
    o["dry"] = _num(c.get("dry", d.get("dry", 0.0)), f"{what} craft dry", 0.0, 1.0)
    o["write"] = _num(c.get("write", d.get("write", 0.4)), f"{what} craft write", 0.02, 5.0)
    o["rows"] = int(_num(c.get("rows", 1), f"{what} craft rows", 1, 6))
    rim = c.get("rim", d["rim"])
    if rim is False:
        rim = 0.0
    if isinstance(rim, dict):
        if set(rim) - {"color", "width"}:
            raise TextError(f"{what}: craft rim = {rim!r}: false, a width (px at 1080) or {{color, width}}")
        o["rim"] = {"color": _colour(rim.get("color", RIM_COLOR), palette, f"{what} craft rim color"),
                    "width": _num(rim.get("width", 16.0), f"{what} craft rim width", 0.0, 80.0)}
    else:
        w = _num(rim, f"{what} craft rim", 0.0, 80.0)
        o["rim"] = {"color": _colour(RIM_COLOR, palette, f"{what} craft rim"), "width": w} if w > 0 else None
    sh = c.get("shadow", {})
    if sh is False:
        o["shadow"] = None
    else:
        if not isinstance(sh, dict) or set(sh) - set(SHADOW):
            raise TextError(f"{what}: craft shadow = {sh!r}: false or {{color, offset, blur, alpha}}")
        s = {**SHADOW, **sh}
        o["shadow"] = {"color": _colour(s["color"], palette, f"{what} craft shadow color"),
                       "offset": _pair(s["offset"], f"{what} craft shadow offset"),
                       "blur": _num(s["blur"], f"{what} craft shadow blur", 0.0, 200.0),
                       "alpha": _num(s["alpha"], f"{what} craft shadow alpha", 0.0, 1.0)}
    arr = c.get("arrive", {})
    if not isinstance(arr, dict):
        raise TextError(f"{what}: craft arrive = {arr!r}: a table {{style, ...}}")
    style = arr.get("style", d["style"])
    if style not in STYLES:
        raise TextError(f"{what}: craft arrive style {style!r} is none of {', '.join(STYLES)}")
    extra = sorted(set(arr) - {"style"} - set(ARRIVE[style]))
    if extra:
        raise TextError(f"{what}: craft arrive keys {extra} are not {style!r}'s (it has {sorted(ARRIVE[style])})")
    o["arrive"] = {"style": style, **{k: _num(arr.get(k, v), f"{what} craft arrive {k}", 0.0)
                                      for k, v in ARRIVE[style].items()}}
    o["arrive"]["big"] = max(o["arrive"]["big"], 0.05)
    o["arrive"]["td"] = max(o["arrive"]["td"], 1e-3)
    o["arrive"]["tp"] = max(o["arrive"]["tp"], 1e-3)
    return o


def leave_spec(v, what):
    """`leave`: "cut" | "peel" | "drop" or {style, ...} with that style's numbers."""
    if v is None:
        v = "cut"
    if isinstance(v, str):
        v = {"style": v}
    if not isinstance(v, dict) or v.get("style") not in LEAVES:
        raise TextError(f"{what} = {v!r}: one of {', '.join(LEAVES)} or {{style, ...}}")
    st = v["style"]
    extra = sorted(set(v) - {"style"} - set(LEAVE[st]))
    if extra:
        raise TextError(f"{what}: keys {extra} are not {st!r}'s (it has {sorted(LEAVE[st])})")
    o = {"style": st}
    for k, d in LEAVE[st].items():
        if k == "corner":
            c = v.get(k, d)
            if c not in CORNERS:
                raise TextError(f"{what} corner {c!r} is none of {', '.join(CORNERS)}")
            o[k] = c
        else:
            o[k] = _num(v.get(k, d), f"{what} {k}", 0.0)
    return o


def plan(spec, tl, palette, fps, size, what=None):
    """One craft entry as one output (`size` its pixels) shows it: {name, craft, under, font, units, panel, size, fit,
    align, offset, gap, stagger, case}, `units` [{k, words: [Word], land (clip frame), t, to, leave, gone (the clip frame
    it is off screen from, None: never), at, tilt, scale, shift, seed}] in order. Placement is finished by `layout`, which
    needs each unit's measured size. `tl` is the timeline (a dict)."""
    name = spec.get("name", "?")
    what = what or f"text {name!r}"
    extra = sorted(set(spec) - ENTRY_KEYS)
    if extra:
        raise TextError(f"{what}: unknown keys {extra} for craft type (known: {sorted(ENTRY_KEYS)})")
    craft = craft_spec(spec.get("craft"), palette, what)
    under = spec.get("under", False)
    if under not in (True, False) and not (isinstance(under, str) and under.strip()):
        raise TextError(f"{what}: under = {under!r}: true (laid on before every [[fx]], which act on it), false (over "
                        "them) or the `name` of the [[fx]] it is laid just before")
    try:
        scr = SR.screen_spec(spec)
    except TextError as e:
        raise TextError(f"{what}: {e}") from None
    at = None if spec.get("at") is None else _pair(spec["at"], f"{what} at")
    box = None if spec.get("box") is None else _pair(spec["box"], f"{what} box")
    pnl = SR.panel(scr, size, at, box)
    lyr = spec.get("lyrics")
    if not isinstance(lyr, dict):
        raise TextError(f"{what}: craft type sets sung words: give `lyrics = {{timeline, line, units, ...}}`")
    extra = sorted(set(lyr) - LYRICS_KEYS)
    if extra:
        raise TextError(f"{what}: unknown lyrics keys {extra} for craft type (known: {sorted(LYRICS_KEYS)})")
    if lyr.get("line") is None:
        raise TextError(f"{what}: lyrics.line (one line of the timeline) is required")
    case = lyr.get("case", DEFAULTS[craft["kind"]]["case"])
    try:
        words = LY.select(tl, line=lyr["line"], words=lyr.get("words"), fps=fps, case=case,
                          clean=bool(lyr.get("clean", False)))
    except LY.LyricsError as e:
        raise TextError(f"{what}: {e}") from None
    to = _num(lyr["to"], f"{what} lyrics.to", 0.0) if lyr.get("to") is not None else max(w.end for w in words)
    leave = leave_spec(lyr.get("leave"), f"{what} lyrics.leave")
    units = _units(lyr.get("units", "line"), words, to, leave, fps, what, craft["rows"])
    lead = _num(spec.get("lead", LEAD), f"{what} lead", 0.0, 2.0)
    arr = craft["arrive"]
    for u in units:
        u["seed"] = craft["seed"] * 7919 + words[0].line * 131 + u["k"]
        if u["tilt"] is None:                                     # a few degrees off level, the same every build
            u["tilt"] = (2.0 * LY.h01(u["seed"], 3) - 1.0) * craft["tilt"]
        st = u["leave"]["style"]
        if (st == "peel") == (craft["kind"] == "brush") and st in ("peel", "wipe"):
            raise TextError(f"{what}: unit {u['k'] + 1}: leave = {st!r} is for "
                            f"{'stickers and tape, not paint (use drop or wipe)' if st == 'peel' else 'paint (brush)'}")
        _time(u, craft, arr, lead, fps)
        if u["leave_frame"] <= u["contact"]:
            raise TextError(f"{what}: unit {u['k'] + 1} leaves (to = {u['to']:g} s) before it lands ({u['t']:g} s)")
    size_ = None if spec.get("size") is None else _num(spec["size"], f"{what} size", 1e-4, 1.0)
    fit = _num(spec.get("fit", FIT), f"{what} fit", 0.05, 1.0)
    align = spec.get("align", "center")
    if align not in ("left", "center", "right"):
        raise TextError(f"{what}: align is left, center or right")
    return {"name": name, "craft": craft, "under": under, "font": spec.get("font"), "units": units, "panel": pnl,
            "size": size_, "fit": fit, "align": align,
            "offset": [0.0, 0.0] if spec.get("offset") is None else _pair(spec["offset"], f"{what} offset"),
            "gap": _num(lyr.get("gap", ROW_GAP), f"{what} lyrics.gap", 0.0, 5.0),
            "stagger": _num(lyr.get("stagger", 0.0), f"{what} lyrics.stagger", -5.0, 5.0)}


def _units(v, words, to, leave, fps, what, rows=1):
    """The units of a selection: "line" (one), "words" (one each) or a list of chunks {words = [i, j] (or one number:
    1-based within the selected line), t, at, tilt, scale, shift, to, leave}. Chunks may not overlap; words no chunk
    takes are not shown."""
    if v == "line":
        raw = [{"words": [words[0].index, words[-1].index]}]
    elif v == "words":
        raw = [{"words": [w.index, w.index]} for w in words]
    elif isinstance(v, (list, tuple)) and v and all(isinstance(u, dict) for u in v):
        raw = list(v)
    else:
        raise TextError(f"{what}: lyrics.units = {v!r}: \"line\", \"words\" or [{{words = [i, j], t, ...}}, ...]")
    by = {w.index: w for w in words}
    taken, out = set(), []
    for k, u in enumerate(raw):
        uw = f"{what} lyrics.units[{k}]"
        extra = sorted(set(u) - UNIT_KEYS)
        if extra:
            raise TextError(f"{uw}: unknown keys {extra} (known: {sorted(UNIT_KEYS)})")
        r = u.get("words")
        if isinstance(r, int) and not isinstance(r, bool):
            r = [r, r]
        if not isinstance(r, (list, tuple)) or len(r) != 2 or not all(isinstance(x, int) for x in r) or r[0] > r[1]:
            raise TextError(f"{uw}: words = {r!r}: [first, last] (1-based within the line)")
        idx = list(range(r[0], r[1] + 1))
        missing = [i for i in idx if i not in by]
        if missing:
            raise TextError(f"{uw}: words {missing} are not in the selection (it has {min(by)}..{max(by)})")
        twice = sorted(taken & set(idx))
        if twice:
            raise TextError(f"{uw}: words {twice} are in another unit already")
        taken |= set(idx)
        ws = [by[i] for i in idx]
        t = _num(u["t"], f"{uw} t", 0.0) if u.get("t") is not None else ws[0].start
        # a line is there when it is sung: labels hand-timed to their nouns once came up after their first words
        if t - ws[0].start > LATE and not u.get("late"):
            raise TextError(f"{uw}: t = {t:g} lands {t - ws[0].start:.2f} s after its first word (word {idx[0]}, sung at "
                            f"{ws[0].start:g} s): a line is there when it is sung. Leave `t` out (the unit goes by that "
                            "word), give a time near it, or say `late = true` when it is meant to come after it")
        uto = _num(u["to"], f"{uw} to", 0.0) if u.get("to") is not None else to
        lv = leave if u.get("leave") is None else leave_spec(u["leave"], f"{uw} leave")
        leave_f = LY.cut_frame(uto, fps)
        out.append({"k": k, "words": ws, "t": t, "to": uto, "leave_frame": leave_f, "leave": lv,
                    "gone": gone_frame(leave_f, lv, fps),
                    "at": None if u.get("at") is None else _pair(u["at"], f"{uw} at"),
                    "tilt": None if u.get("tilt") is None else _num(u["tilt"], f"{uw} tilt", -90.0, 90.0),
                    "scale": _num(u.get("scale", 1.0), f"{uw} scale", 0.1, 10.0),
                    "shift": _num(u.get("shift", 0.0), f"{uw} shift", -10.0, 10.0),
                    "rows": int(_num(u.get("rows", rows), f"{uw} rows", 1, 6))})
    return out


def _time(u, craft, arr, lead, fps):
    """A unit's frames, in place: `moment` (s, its time less the entry's `lead`), `contact` (the clip frame it is first
    at rest: `floor(moment * fps + 0.4)`; for a pen or a brush, the frame it starts, `arrive.lead` before), `land` (the
    first frame it is on screen: its approach before the contact; below 0 when it lands in the clip's first frames,
    which then show it where its motion has got to) and, for the crafts that write, `pen`: [(start, finish)] clip
    seconds per word, each word at the one-go speed and done by its own sung moment (onset less `lead`, on its frame)."""
    u["moment"] = u["t"] - lead
    ink = craft["kind"] in ("masking", "brush")
    head = arr.get("lead", 0.0) if ink else 0.0
    u["contact"] = LY.landing_frame(u["moment"] - head, fps)
    u["land"] = u["contact"] - int(math.ceil(arr["fall"] * fps - 1e-9))
    u["pen"] = None
    if ink:
        sizes = [len(w.text) + 1 for w in u["words"]]
        per = craft["write"] / max(sum(sizes), 1)
        start, pen = u["contact"] / fps, []
        for w, n in zip(u["words"], sizes):
            due = LY.landing_frame(max(w.start, u["t"]) - lead, fps) / fps
            end = max(min(start + per * n, due), start + 1.0 / fps)
            pen.append((start, end))
            start = end
        u["pen"] = pen


def pen_progress(u, frame, fps):
    """How far the pen has got on clip frame `frame`: i + f with words 0..i-1 laid and f of word i (len(words): all)."""
    x = frame / fps
    for i, (a, b) in enumerate(u["pen"]):
        if x < b:
            return i + (ease_io((x - a) / (b - a)) if x > a else 0.0)
    return float(len(u["pen"]))


def split_rows(widths, n, space=0.0):
    """The words (their `widths`) set in at most `n` rows of about equal width, in reading order: [[word index, ...],
    ...], the widest row as narrow as it can be (`space` is the gap a row adds between two words)."""
    k = len(widths)
    n = max(1, min(int(n), k))
    best = {}

    def row_w(a, b):
        return sum(widths[a:b]) + space * (b - a - 1)

    def go(i, r):                                  # the narrowest widest row for words i.. in r rows
        if (i, r) in best:
            return best[(i, r)]
        if r == 1:
            best[(i, r)] = (row_w(i, k), [k])
            return best[(i, r)]
        out = None
        for j in range(i + 1, k - r + 2):
            w, cuts = go(j, r - 1)
            got = (max(row_w(i, j), w), [j] + cuts)
            if out is None or got[0] < out[0] - 1e-9:
                out = got
        best[(i, r)] = out
        return out
    cuts = [0] + go(0, n)[1]
    return [list(range(a, b)) for a, b in zip(cuts, cuts[1:])]


def gone_frame(leave_f, leave, fps):
    """The clip frame a unit is off screen from: its leaving frame for `cut`, after the peel or the fall otherwise."""
    st = leave["style"]
    if st == "cut":
        return leave_f
    dur = {"peel": leave.get("lift", 0) + leave.get("pull", 0) + leave.get("flick", 0)}.get(st, leave.get("life", 0))
    return leave_f + max(int(math.ceil(dur * fps)), 1)


# ------------------------------------------------------------------------------------------------------ placement
def layout(p, sizes):
    """Where each unit stands: `sizes` [(w, h)] each unit's measured size at rest at a cap height of 1 (its strip, rim
    and all, unturned). Sets p["cap"] (frame heights: `size`, or the largest at which the stacked rows fit `fit` of the
    panel, at most `size`) and each unit's "center" [x, y] (frame heights from the picture's centre, y up) and "dims"
    [w, h] at that cap. A unit with its own `at` stands there; the others are stacked in rows, top to bottom in their
    order, `gap` caps apart, each set against the panel's `align` margin and shifted `stagger` caps (alternating sides)
    plus its own `shift`."""
    (pu, pv), (pw, ph) = p["panel"]
    units = p["units"]
    rows = [k for k, u in enumerate(units) if u["at"] is None]
    if rows:
        need_w = max(sizes[k][0] * units[k]["scale"] + abs(p["stagger"]) + abs(units[k]["shift"]) for k in rows)
        need_h = sum(sizes[k][1] * units[k]["scale"] for k in rows) + p["gap"] * (len(rows) - 1)
        fit_cap = min(p["fit"] * pw / max(need_w, 1e-9), p["fit"] * ph / max(need_h, 1e-9))
        cap = fit_cap if p["size"] is None else min(p["size"], fit_cap)
    else:
        cap = p["size"] if p["size"] is not None else 0.06
    p["cap"] = cap
    ox, oy = p["offset"]
    total = sum(sizes[k][1] * units[k]["scale"] for k in rows) * cap + p["gap"] * cap * (len(rows) - 1)
    y = pv + 0.5 * total
    for j, k in enumerate(rows):
        u = units[k]
        w, h = sizes[k][0] * u["scale"] * cap, sizes[k][1] * u["scale"] * cap
        shift = (p["stagger"] * (1 if j % 2 else -1) * (1 if len(rows) > 1 else 0) + u["shift"]) * cap
        x = {"left": pu - 0.5 * pw * p["fit"] + 0.5 * w, "center": pu,
             "right": pu + 0.5 * pw * p["fit"] - 0.5 * w}[p["align"]] + shift
        u["center"] = [x + ox, y - 0.5 * h + oy]
        u["dims"] = [w, h]
        y -= h + p["gap"] * cap
    for k, u in enumerate(units):
        if u["at"] is not None:
            u["center"] = [u["at"][0] + ox, u["at"][1] + oy]
            u["dims"] = [sizes[k][0] * u["scale"] * cap, sizes[k][1] * u["scale"] * cap]
    return p


# ------------------------------------------------------------------------------------------------------ motion
def state(p, u, frame, fps):
    """How unit `u` of plan `p` stands on clip frame `frame`, or None when it is not on screen: {center [x, y] (frame
    heights, y up), rot (degrees counter-clockwise), sx, sy (its own axes' scale), alpha, lift (0..1: how far above the
    paper it is, which carries its shadow out), reveal (0..1 of its ink, 1 for crafts that do not write), peel (None, or
    {fold: 0..1 of the way the curl has come over the unit, corner})}. The map of the whole unit is one: its pieces keep
    their places relative to each other."""
    if frame < u["land"] or (u["gone"] is not None and frame >= u["gone"]):
        return None
    arr = p["craft"]["arrive"]
    t = (frame - u["contact"]) / fps                              # < 0: the approach
    cx, cy = u["center"]
    rot, sx, sy, alpha, lift = u["tilt"], 1.0, 1.0, 1.0, 0.0
    if arr["big"] != 1.0 or arr["squash"] > 0 or arr["fall"] > 0:
        fall = arr["fall"]
        if t < 0:                                                # coming down: faster and faster (ease-in)
            q = max(1.0 + t / fall, 0.0) ** 2 if fall > 0 else 1.0
            s = arr["big"] + (1.0 - arr["big"]) * q
            sx = sy = s
            rot += arr["spin"] * (1.0 - q) * (1 if u["seed"] % 2 else -1)
            cy += arr["drop"] * (1.0 - q)
            lift = min(1.0, (1.0 - q) * (arr["big"] - 1.0) / 0.5 + 0.15)    # opaque from the first frame: a slap
        else:                                                    # the contact: a squash that rings out, the spin settles
            sq = spring(t, arr["squash"], arr["td"], arr["tp"])
            sx, sy = 1.0 + sq, 1.0 - sq
            rot += spring(t, 0.35 * arr["spin"], 2.0 * arr["td"], 1.6 * arr["tp"]) * (1 if u["seed"] % 2 else -1)
    reveal = 1.0 if u["pen"] is None else pen_progress(u, frame, fps)
    peel, wipe = None, None
    lv = u["leave"]
    if frame >= u["leave_frame"] and lv["style"] != "cut":
        tl = (frame - u["leave_frame"]) / fps
        if lv["style"] == "peel":                                # a hand: a slow lift, a fast pull, a flick
            corner = lv["corner"] if lv["corner"] != "auto" else ("top_right" if u["seed"] % 2 else "top_left")
            a, b = lv["lift"], lv["lift"] + lv["pull"]
            if tl < a:                                           # the corner lifts and rolls back, slowly
                fold = 0.2 * (tl / a) ** 1.6
                lift = max(lift, 0.4 * tl / a)
            elif tl < b:                                         # the roll is pulled across, fast
                fold = 0.2 + 0.55 * ease_io((tl - a) / max(lv["pull"], 1e-6))
                lift = 0.6
            else:                                                # the flick: off toward the corner, turning, gone
                q = min((tl - b) / max(lv["flick"], 1e-6), 1.0)
                fold = 0.75 + 0.2 * q
                side = 1.0 if corner.endswith("right") else -1.0
                up = 1.0 if corner.startswith("top") else -1.0
                e = q ** 1.4
                cx += side * 0.8 * e
                cy += up * 0.6 * e
                rot -= side * up * 35.0 * e
                sx *= 1.0 + 0.15 * e
                sy *= 1.0 + 0.15 * e
                lift = 1.0
                alpha *= 1.0 - LY.sstep(0.6, 1.0, q)
            peel = {"fold": fold, "corner": corner}
        elif lv["style"] == "wipe":                              # a thumb drags the paint off sideways
            wipe = min(tl / max(lv["life"], 1e-6), 1.0)
        else:                                                    # drop: gravity and a turn, fading at the end
            cy -= 0.5 * lv["g"] * tl * tl
            rot += lv["spin"] * tl * (1 if u["seed"] % 2 else -1)
            lift = max(lift, min(1.0, tl * 6.0))
            alpha *= 1.0 - LY.sstep(0.55 * lv["life"], lv["life"], tl)
    return {"center": [cx, cy], "rot": rot, "sx": sx, "sy": sy, "alpha": max(0.0, min(1.0, alpha)), "lift": lift,
            "reveal": reveal, "peel": peel, "wipe": wipe}


def matrix(st, size):
    """The 2 x 3 affine map (as nested lists) that carries a unit's sprite coordinates (pixels at the output's size,
    origin at the unit's centre, y down) to the frame's pixels (origin top left), for `st` = state(...) and the output's
    `size` (w, h) pixels: scale (sx, sy) on the unit's own axes, then the turn (counter-clockwise on screen), then the
    move to its centre."""
    w, h = size
    a = math.radians(st["rot"])
    c, s = math.cos(a), math.sin(a)
    sx, sy = st["sx"], st["sy"]
    X = 0.5 * w + st["center"][0] * h
    Y = 0.5 * h - st["center"][1] * h
    return [[c * sx, s * sy, X], [-s * sx, c * sy, Y]]


def on_screen(p, frame):
    """The units of plan `p` on screen on clip frame `frame`."""
    return [u for u in p["units"] if u["land"] <= frame and (u["gone"] is None or frame < u["gone"])]
