"""wordtype: sung words as type (docs/design.md: Text, `lyrics`). Called by the text stage for a `[[text]]` entry with
`lyrics = {...}`; returns one ordinary text spec per word, built by `mkmmd.core.wordtype` from the timeline.

    [[text]]
    name = "mug"
    mount = "mug"; at = [0, 0, 0.14]; facing = [0.5, -0.8, 0.2]; box = [0.16, 0.2]      # placement and look: as any text
    font = "fraunces_italic"; fit = 0.8; color = "love"; glow = 0.8
    lyrics = {timeline = "audio/timeline.json", line = 1, words = [1, 4], style = "rise", from = 1.3, to = 2.6,
              layout = {kind = "stack", dir = "up", rows = 4}, colors = ["love", "gold", "pine"]}

The timeline is read here and nowhere else; the words never reach a log line, an error, a report or an object property
(objects are `<name>_l<line>w<word>`, `..._<shot>` with `lyrics.zones`, and carry `mk_lyric = [line, word]`). With `zones` the
cut (the project's `[[shot]]` names and times) is handed to the expansion, and every zone's panel is asked for on its own."""
import json

from ...core import wordtype as LY
from . import BuildError


def _timeline(ctx, spec):
    name = spec.get("name", "?")
    ref = (spec.get("lyrics") or {}).get("timeline")
    if not ref:
        raise BuildError(f"text {name!r}: lyrics needs `timeline` = the path of the timeline JSON")
    path = ctx.path(ref)
    cache = ctx.__dict__.setdefault("_lyric_timelines", {})
    if path not in cache:
        try:
            with open(path, encoding="utf-8") as fh:
                cache[path] = json.load(fh)
        except OSError as e:
            raise BuildError(f"text {name!r}: cannot read the timeline {path}: {e.strerror}") from None
        except ValueError as e:                     # JSONDecodeError: its message holds a position, not the content
            raise BuildError(f"text {name!r}: the timeline {path} is not valid JSON ({type(e).__name__})") from None
    return cache[path]


def _panel(ctx, spec):
    """(w, h) metres of the entry's surface or box, checking every surface of an `on` list on the way."""
    from . import text as TX
    probe = {k: v for k, v in spec.items() if k != "lyrics"}
    on = spec.get("on")
    refs = on if isinstance(on, (list, tuple)) else [on]
    panels = []
    for ref in refs:
        if ref is None:
            probe.pop("on", None)
        else:
            probe["on"] = ref
        panels.append(TX._place(ctx, probe)[3])
    return panels[0]


def _shots(ctx):
    """The project's cut as [{name, from, to}] clip seconds, for `lyrics.zones` (a shot that only lends frames has none)."""
    return [{"name": s["name"], "from": s["from"], "to": s["to"]} for s in ctx.data.get("shot", [])
            if s.get("name") and "from" in s and "to" in s]


def expand(ctx, spec, measure=None, report=None):
    """The word texts of a lyrics entry. `measure` (string[, font] -> ink box, see mkmmd.core.wordtype.expand) serves the
    layouts that put words next to each other; `report` (a dict) receives the summary: counts and (line, word) numbers."""
    name = spec.get("name", "?")
    tl = _timeline(ctx, spec)
    zoned = (spec.get("lyrics") or {}).get("zones") is not None       # each zone's panel is its own: asked per block
    try:
        specs, summary = LY.expand(spec, tl, fps=ctx.fps, frame0=ctx.frame0, panel=None if zoned else _panel(ctx, spec),
                                   measure=measure, shots=_shots(ctx), panel_of=lambda e: _panel(ctx, e))
    except LY.LyricsError as e:
        raise BuildError(str(e)) from None
    for key in ("first_frame", "last_frame", "cut_frame"):      # clip frames -> Blender frames (what `mk look` takes)
        if summary.get(key) is not None:
            summary[key] += ctx.frame0
    for b in summary.get("blocks", []):
        for key in ("first_frame", "last_frame"):
            if b.get(key) is not None:
                b[key] += ctx.frame0
    if report is not None:
        report.update(summary)
    ctx.log("lyrics", name, f"{summary['words']} word(s) of line(s) {summary['lines']}, Blender frames "
            f"{summary['first_frame']}..{summary['last_frame']}, cut at {summary['cut_frame']}"
            + (f", {len(summary['blocks'])} shot(s)" if summary.get("blocks") else ""))
    return specs
