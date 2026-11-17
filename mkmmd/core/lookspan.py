"""`[[look.span]]`, bpy-free (docs/design.md: Lights and the look): a part of the clip seen through another look, the
project's `[look]` with an exposure, a contrast look or a gamma of its own (two scenes of one clip: a room at night with
a darker exposure and a harder contrast, then a figure on a sweep in day light).

    [[look.span]]
    from = 0.0                          # clip seconds (required, from < to)
    to = 7.656
    exposure = -0.5                     # any of exposure, contrast, gamma (one at least); what a span leaves out is
    contrast = "Medium High Contrast"   # [look]'s

A span holds the frames a [[shot]] with the same `from` and `to` holds: from the frame of `from` up to the one before
the frame of `to`, each the build's rounding of its clip second (mkmmd.core.transition.frame_of, as the cut's), so a
span switches on the frame a cut written with the same number does. No `view`: a project has one view transform. The
lights stage keys the scene's view settings on the spans (mkmmd/blender/build/lights.py) and the bake captures every
look the frames are seen through (mkmmd/blender/build/bake_lit.py `view`)."""
from . import transition as TR

LOOK = ("exposure", "contrast", "gamma")
KEYS = ("from", "to") + LOOK


class LookSpanError(ValueError):
    """A [[look.span]] the lights stage cannot use: an unknown key, a bad value, outside the clip, two at once."""


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def normalize(spans, fps, frame0, last_frame):
    """The [[look.span]] entries as [{index, from, to, exposure?, contrast?, gamma?}] in time order (`index` their place
    in the file): `from` and `to` Blender frames, the span holding from <= frame < to; what the span sets as given
    (exposure and gamma floats, contrast the look's name as written). `last_frame` is the clip's last Blender frame.
    [] without spans; LookSpanError names the span and what is wrong."""
    if spans is None:
        return []
    if isinstance(spans, dict):
        spans = [spans]
    if not isinstance(spans, list):
        raise LookSpanError("[[look.span]]: expected an array of tables")
    out = []
    for i, e in enumerate(spans):
        where = f"look span {i}"
        if not isinstance(e, dict):
            raise LookSpanError(f"{where}: expected a table {{from, to, exposure, contrast, gamma}}")
        if "view" in e:
            raise LookSpanError(f"{where}: view: a project has one view transform ([look] view); a span sets "
                                f"{', '.join(LOOK)}")
        unknown = sorted(set(e) - set(KEYS))
        if unknown:
            raise LookSpanError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
        for k in ("from", "to"):
            if not _number(e.get(k)):
                raise LookSpanError(f"{where}: {k} = {e.get(k)!r}: clip seconds")
        where = f"look span {i} ({e['from']} .. {e['to']} s)"
        if not e["from"] < e["to"]:
            raise LookSpanError(f"{where}: from is not before to")
        if not any(k in e for k in LOOK):
            raise LookSpanError(f"{where}: sets none of {', '.join(LOOK)}: a span is another look")
        span = {"index": i, "from": TR.frame_of(e["from"], fps, frame0), "to": TR.frame_of(e["to"], fps, frame0)}
        for k in ("exposure", "gamma"):
            if k in e:
                if not _number(e[k]):
                    raise LookSpanError(f"{where}: {k} = {e[k]!r}: a number")
                span[k] = float(e[k])
        if "contrast" in e:
            if not isinstance(e["contrast"], str) or not e["contrast"]:
                raise LookSpanError(f"{where}: contrast = {e['contrast']!r}: the name of a Blender look")
            span["contrast"] = e["contrast"]
        if span["to"] <= span["from"]:
            raise LookSpanError(f"{where}: under a frame (frame {span['from']} both ends)")
        if span["from"] < frame0 or span["to"] - 1 > last_frame:
            raise LookSpanError(f"{where}: frames {span['from']}..{span['to'] - 1} reach outside the clip "
                                f"({frame0}..{last_frame})")
        out.append(span)
    out.sort(key=lambda s: s["from"])
    for a, b in zip(out, out[1:]):
        if b["from"] < a["to"]:
            raise LookSpanError(f"look span {a['index']} and look span {b['index']} overlap in frames "
                                f"{b['from']}..{min(a['to'], b['to']) - 1}: one look a frame")
    return out
