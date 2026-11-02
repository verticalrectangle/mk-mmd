"""calltype: words of the entry's own slammed onto hits (docs/design.md: Text, Calls). Called by the text stage for a
`[[text]]` with `calls = {...}`; returns one ordinary text spec per hit, built by `mkmmd.core.calltype` with the lyric
machinery (an arrival style, a layout, recycle, cycled strips, tilts and colours).

    [[text]]
    name = "calls"
    screen = true; font = "my_heavy_kana_font"; size = 0.08
    outline = { color = "base", width = 0.12, alpha = 1.0 }
    calls = { words = ["ハイ！", "オイ！"], lane = "kick", min_db = -3, style = "slam", tilt = { random = 10 },
              backing = { color = "love", torn = 0, slant = 14 }, recycle = { fade = 3, slide = 6 }, hold = 1.2 }

The hits are `calls.times` or a lane of the project's `[audio] hits` file (read only when `calls.lane` names one). Calls are
not lyrics: the log may name them. Objects are `<name>_c<hit>` (`@<output>` for one output of several); a ring laid round a
point of the scene gets its `<name>_ring` empty as lyric type's does."""
from ...core import calltype as CT
from ...core import lanes as LN
from . import BuildError
from . import wordtype as LYB


def _hits(ctx, name, calls):
    """The [audio] hits file when `calls.lane` names a lane, else None."""
    if not isinstance(calls, dict) or calls.get("lane") is None:
        return None
    rel = (ctx.data.get("audio") or {}).get("hits")
    if not rel:
        raise BuildError(f"text {name!r}: calls.lane reads the [audio] hits file and the project names none (`mk timeline "
                         f"hits` writes it)")
    try:
        return LN.load(ctx.path(rel))
    except LN.LaneError as e:
        raise BuildError(f"text {name!r}: calls: {e}") from None


def expand(ctx, spec, measure=None, report=None):
    """The call texts of a calls entry. `measure` (string[, font] -> ink box, see mkmmd.core.wordtype.expand) serves the
    layouts that set calls side by side; `report` (a dict) receives the summary (Blender frames)."""
    name = spec.get("name", "?")
    calls = spec.get("calls")
    try:
        specs, summary = CT.expand(spec, fps=ctx.fps, frame0=ctx.frame0, hits=_hits(ctx, name, calls),
                                   end=ctx.time(ctx.end + 1), panel=LYB._panel(ctx, spec), measure=measure)
    except CT.CallError as e:
        raise BuildError(str(e)) from None
    for key in ("first_frame", "last_frame", "cut_frame"):      # clip frames -> Blender frames (what `mk look` takes)
        if summary.get(key) is not None:
            summary[key] += ctx.frame0
    for r in summary.get("rings", []):
        if r.get("first_frame") is not None:
            r["first_frame"] += ctx.frame0
    if summary.get("ring_track") is not None:
        LYB._ring_track(ctx, spec, specs, summary)
    if report is not None:
        report.update(summary)
    ctx.log("calls", name, f"{summary['calls']} call(s) of {summary['words']} on {summary['hits']} hit(s), Blender frames "
            f"{summary['first_frame']}..{summary['last_frame']}, gone by {summary['cut_frame']}"
            + (f", {len(summary['skipped'])} never on screen" if summary["skipped"] else ""))
    return specs
