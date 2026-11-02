"""Calls, the numbers (docs/design.md: Text, Calls): the bpy-free half of `mkmmd/blender/build/calltype.py`.

A `[[text]]` with `calls = {...}` sets words of its own, written in mk.toml, one on each hit: the shouts idol fans yell at a
show, a hype word, an onomatopoeia (never the song's words: those are lyric type). The hits are `times` (clip seconds) or a
lane of the project's [audio] hits file (`lane`, at or above `min_db`, every `every`-th), inside `from` .. `to`, and the
words take them round-robin: hit k shows `words[k mod len]`. Each call becomes one ordinary text spec with a `kinetic`
table, made by the lyric machinery (mkmmd/core/wordtype.py) from a `Call` per hit: it lands on its hit's frame,
`floor(t * fps + 0.4)`, arrives in a lyric style (`slam`: big, down fast, ringing, a shake), stands where the layout puts
it (`same`, `slots`, `flow`, `stack`, `ring`, `bleed`), takes its colour, tilt, size, nudge and strip from lists cycled
over the hits (colours by call, or by hit), and leaves when the next call takes its slot (`recycle`, with its fade and
slide), when its `hold` is over (the same fade and slide), or at `to`.

A call has no note: it starts and ends on its hit, so nothing spreads, wobbles, breathes or is typed over it (the lyric
keys that follow a sung note are not calls keys). Its object is `<name>_c<hit>` (the hit's number among those picked,
from 1). Calls are not lyrics: their text may be logged and reported (the text stage reports each call's string)."""
import math

from . import lanes as LN
from . import wordtype as LY

CALL_KEYS = {"words", "times", "lane", "min_db", "every", "from", "to", "hold", "style", "arrive", "colors", "color_by",
             "tilt", "sizes", "offsets", "scales", "backing", "recycle", "layout"}
LOOK = ("style", "arrive", "colors", "tilt", "sizes", "offsets", "scales", "backing", "recycle", "layout")  # the lyric
#                                                                        machinery's own keys, handed on as they are
COLOR_BY = {"call": "line", "hit": "word"}      # what a `colors` list cycles with, and the lyric machinery's name for it
NO_NOTE = ("type",)                             # styles that play over a sung note, which a call does not have
MID_DB = LY.DB_FLOOR + 0.5 * LY.DB_RANGE        # the level of a word without a voice (nothing reads it for a call)


class CallError(ValueError):
    """A calls entry that cannot be expanded."""


class Call(LY.Word):
    """A call on its hit, in the lyric machinery's terms: `line` is the call's number in `words`, `index` the hit's (both
    from 1), `k` the hit's place (from 0); it starts and ends on the hit."""

    def __repr__(self):
        return f"Call({self.line},{self.index})"

    __str__ = __repr__

    @property
    def tag(self):
        return f"c{self.index}"


def _number(v, what, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CallError(f"calls: {what} = {v!r}: a number")
    if positive and not float(v) > 0:
        raise CallError(f"calls: {what} = {v!r}: more than 0")
    return float(v)


def options(calls):
    """`calls` checked: {words, color_by, t0, t1, hold} (the hits are hit_times'; the look keys are the lyric machinery's
    and are checked there)."""
    if not isinstance(calls, dict):
        raise CallError("calls is a table {words, times | lane, style, ...}")
    unknown = sorted(set(calls) - CALL_KEYS)
    if unknown:
        lyric = sorted(set(unknown) & LY.LYRICS_KEYS)
        raise CallError(f"calls keys {unknown} are unknown (known: {sorted(CALL_KEYS)})"
                        + (f"; {lyric} belong to lyrics: calls are set as written and have no note" if lyric else ""))
    words = calls.get("words")
    words = [words] if isinstance(words, str) else words
    if not isinstance(words, (list, tuple)) or not words or not all(isinstance(w, str) and w.strip() for w in words):
        raise CallError("calls: words is the list of calls (strings, taken round-robin by the hits)")
    style = calls.get("style", "pop")
    if style in NO_NOTE:
        raise CallError(f"calls: style {style!r} plays over a sung note, and a call has none")
    color_by = calls.get("color_by", "call")
    if color_by not in COLOR_BY:
        raise CallError(f"calls: color_by is {' or '.join(repr(c) for c in COLOR_BY)} (a list of colors cycles with the "
                        "call, or with the hit)")
    t0 = None if calls.get("from") is None else _number(calls["from"], "from")
    t1 = None if calls.get("to") is None else _number(calls["to"], "to")
    if t0 is not None and t1 is not None and not t1 > t0:
        raise CallError(f"calls: to = {t1:g} comes before from = {t0:g}")
    hold = None if calls.get("hold") is None else _number(calls["hold"], "hold", positive=True)
    return {"words": [str(w) for w in words], "color_by": color_by, "t0": t0, "t1": t1, "hold": hold}


def hit_times(calls, hits=None):
    """The clip seconds the calls land on, sorted: `times`, or the hits of the `lane` in `hits` ({lane: [...]}: the [audio]
    hits file) at or above `min_db`, every `every`-th; both from `from` up to (not including) `to`."""
    o = options(calls)
    lo = -math.inf if o["t0"] is None else o["t0"]
    hi = math.inf if o["t1"] is None else o["t1"]
    if ("times" in calls) == ("lane" in calls):
        raise CallError("calls: give `times` (clip seconds) or `lane` (one of the [audio] hits file), one of them")
    if "times" in calls:
        if "min_db" in calls or "every" in calls:
            raise CallError("calls: min_db and every pick from a lane; give the times you want")
        ts = calls["times"]
        ts = [ts] if isinstance(ts, (int, float)) and not isinstance(ts, bool) else ts
        if not isinstance(ts, (list, tuple)) or not ts:
            raise CallError("calls: times is a list of clip seconds")
        return sorted(t for t in (_number(x, "times[]") for x in ts) if lo <= t < hi)
    if hits is None:
        raise CallError(f"calls: lane {calls['lane']!r} reads the [audio] hits file, and the project names none (`mk "
                        "timeline hits` writes it)")
    try:
        c = LN.choice(calls, "calls")
        return LN.pick(hits, c["lane"], c["min_db"], c["every"], c["t0"], c["t1"])
    except LN.LaneError as e:
        msg = str(e)
        raise CallError(msg if msg.startswith("calls") else f"calls: {msg}") from None


def expand(entry, *, fps, frame0=0, hits=None, end=None, panel=None, measure=None):
    """Turn a `[[text]]` entry with `calls = {...}` into (specs, summary): one ordinary text spec per hit whose call is ever on
    screen, named `<name>_c<hit>`, with the call as its `text` and a `kinetic` table, plus a summary {calls, hits, words,
    style, layout, skipped (hit numbers), first_frame, last_frame, cut_frame (clip frames)} (with a ring layout, its
    `rings` and `ring_track` as mkmmd.core.wordtype.expand gives them).

    entry    the [[text]] table: `name`, the placement and look keys every call inherits (as a lyrics entry's), `calls`
    fps, frame0   the project's; `to` rounds to a frame as the shots stage cuts
    hits     the [audio] hits file ({lane: [...]}), for `calls.lane`
    end      clip seconds where the calls are gone when `to` does not say (the end of the build)
    panel, measure   as mkmmd.core.wordtype.expand's: the (w, h) of the entry's surface, `box` or screen band, and the
             ink box of a string (only the layouts that set calls side by side ask)"""
    name = entry.get("name")
    if not name:
        raise CallError("[[text]] with calls needs a name")
    try:
        for bad in ("text", "value", "kinetic", "reveal", "lyric", "lyrics"):
            if bad in entry:
                raise CallError(f"a calls entry takes its words from `calls.words`: drop `{bad}`")
        calls = entry.get("calls")
        o = options(calls)
        times = hit_times(calls, hits)
        if not times:
            raise CallError("calls: no hit to land on" + (f" (lane {calls['lane']!r}: mind from, to and min_db)"
                                                          if "lane" in calls else " between from and to"))
        t_to = o["t1"] if o["t1"] is not None else end
        if t_to is None:
            raise CallError("calls: give `to`, where the calls are gone")
        ws = o["words"]
        seq = [Call(k % len(ws) + 1, k + 1, k, t, t, t, MID_DB, ws[k % len(ws)]) for k, t in enumerate(times)]
        lyr = {key: calls[key] for key in LOOK if key in calls}
        lyr.update(color_by=COLOR_BY[o["color_by"]], spread=False, to=float(t_to))
        base = {k: v for k, v in entry.items() if k != "calls"}
        specs, summ = LY._expand(base, name, lyr, {}, float(fps), int(frame0), panel, measure, words=seq, hold=o["hold"],
                                 headroom=False)
    except LY.LyricsError as e:                       # the lyric machinery speaks of lyrics: these are calls
        msg = str(e)
        msg = "calls: " + msg[len("lyrics: "):] if msg.startswith("lyrics: ") else msg
        raise CallError(msg if msg.startswith("text ") else f"text {name!r}: {msg}") from None
    except CallError as e:
        msg = str(e)
        raise CallError(msg if msg.startswith("text ") else f"text {name!r}: {msg}") from None
    for s in specs:
        s.pop("lyric", None)                          # a call is not a sung word
    summary = {"calls": len(specs), "hits": len(times), "words": len(ws), "style": summ["style"], "layout": summ["layout"],
               "skipped": [hit for _, hit in summ["skipped"]], "first_frame": summ["first_frame"],
               "last_frame": summ["last_frame"], "cut_frame": summ["cut_frame"]}
    for key in ("rings", "ring_track"):
        if key in summ:
            summary[key] = summ[key]
    return specs, summary
