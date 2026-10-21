"""Medium shifts, bpy-free (docs/design.md: Shots: Medium shifts): `[[medium]]` entries draw the whole picture in another
look for a window of clip seconds, whatever shot is in the cut, the way a picture flickers into another universe for a
beat (a cel, cartoon or sketch flash on a hit). An entry is a shot's look for a span of time: `from`, `to` and the look
keys of a [[shot]] (`style`, `colors`, `tones`, `materials`, `lines`, `sketch`, `hide`, `keep`, a silhouette's
`accent`...), with `aspect.<output>` tables laid over it as over a shot (`style = "none"` leaves an output alone).

The shots stage normalises the windows per output (mkmmd.core.shotstyle.normalize) and keeps them in scene["mk_medium"]
as Blender frames; mkmmd.blender.styles draws a window's look instead of the shot's, on the cut's frames and on the plates
of transitions alike."""
from . import shotspec as SP
from . import shotstyle as SS

LOOK = tuple(k for k in SP.LOOK if k != "reflection")      # a window is a flat look: the window reflection is a shot's
KEYS = ("from", "to", "aspect") + LOOK


class MediumError(ValueError):
    """A [[medium]] entry the shots stage cannot use: an unknown key, a bad span, two windows at once, a bad look."""


def normalize(entries, outputs, palette, vector, fps, frame0, last_frame=None):
    """The [[medium]] entries as [{index, from, to, styles: {output: {kind: spec} or {}}}] in time order: `from` and `to`
    Blender frames (the window holds from <= frame < to), `styles` the normalised look per output name in `outputs`
    (`vector` is the project's [vector] table). `last_frame` is the clip's last Blender frame (None: not checked)."""
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise MediumError("[[medium]]: expected an array of tables")
    out = []
    for i, e in enumerate(entries):
        where = f"medium {i}"
        if not isinstance(e, dict):
            raise MediumError(f"{where}: expected a table {{from, to, style, ...}}")
        unknown = sorted(set(e) - set(KEYS))
        if unknown:
            raise MediumError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
        for k in ("from", "to"):
            if isinstance(e.get(k), bool) or not isinstance(e.get(k), (int, float)):
                raise MediumError(f"{where}: {k} = {e.get(k)!r}: clip seconds")
        f0, f1 = int(round(frame0 + e["from"] * fps)), int(round(frame0 + e["to"] * fps))
        if f1 <= f0:
            raise MediumError(f"{where}: {e['from']} .. {e['to']} s is under a frame")
        if f0 < frame0 or (last_frame is not None and f1 - 1 > last_frame):
            raise MediumError(f"{where}: frames {f0}..{f1 - 1} reach outside the clip ({frame0}..{last_frame})")
        if e.get("style") not in SS.STYLES:
            raise MediumError(f"{where}: style = {e.get('style')!r}: a window is drawn in a look "
                              f"({', '.join(SS.STYLES)})")
        aspects = e.get("aspect") or {}
        if not isinstance(aspects, dict):
            raise MediumError(f"{where}: `aspect` holds one table per output, [medium.aspect.<output>]")
        for name, over in aspects.items():
            if name not in outputs:
                raise MediumError(f"{where}: aspect.{name} is for an output the project does not have "
                                  f"(outputs: {', '.join(outputs)})")
            if not isinstance(over, dict):
                raise MediumError(f"{where}: aspect.{name} is a table of the look keys it changes")
            bad = sorted(set(over) - set(LOOK))
            if bad:
                raise MediumError(f"{where} aspect.{name}: unknown keys {bad} (known: {', '.join(LOOK)})")
        own = {k: v for k, v in e.items() if k in LOOK}
        styles = {}
        for name in outputs:
            try:
                styles[name] = SS.normalize(SP.merged(own, aspects.get(name)), palette, vector)
            except SS.StyleError as err:
                raise MediumError(f"{where} ({name}): {str(err).replace('shot None: ', '')}") from None
        out.append({"index": i, "from": f0, "to": f1, "styles": styles})
    out.sort(key=lambda w: w["from"])
    for a, b in zip(out, out[1:]):
        if b["from"] < a["to"]:
            raise MediumError(f"medium {a['index']} and medium {b['index']} overlap in frames "
                              f"{b['from']}..{min(a['to'], b['to']) - 1}: one look at a time")
    return out


def at(windows, frame):
    """The window that holds Blender `frame` (from <= frame < to), or None."""
    for w in windows or ():
        if w["from"] <= frame < w["to"]:
            return w
    return None


def summary(windows):
    """Numbers for reports: [{index, frames: [first, last], looks: {output: kind or None}}]."""
    return [{"index": w["index"], "frames": [w["from"], w["to"] - 1],
             "looks": {o: next(iter(s), None) for o, s in w["styles"].items()}} for w in windows]
