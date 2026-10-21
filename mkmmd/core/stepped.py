"""Stepped motion, bpy-free (docs/design.md: Stepped motion): `[[stepped]]` entries put cast members on twos for a
window of clip seconds, the way hand-drawn animation is shot: a new pose `fps` times a second (12: on twos at 24), each
held until the next, while the cameras keep every frame. The build's `stepped` stage keys the holds
(mkmmd/blender/build/stepped.py)."""

KEYS = ("from", "to", "cast", "fps")
FPS = 12.0                   # poses a second: on twos


class StepError(ValueError):
    """A [[stepped]] entry the build cannot use: an unknown key or member, a bad span or rate, two windows at once."""


def steps(f0, f1, clip_fps, fps):
    """The frames in [f0, f1) a new pose is drawn on: f0 + round(k x clip_fps / fps) for k = 0, 1, ..."""
    out, k = [], 0
    while True:
        g = f0 + int(round(k * clip_fps / fps))
        if g >= f1:
            return out
        if not out or g > out[-1]:
            out.append(g)
        k += 1


def held(window, frame):
    """The frame whose pose `frame` shows (itself outside the window): the last step at or before it."""
    if not window["from"] <= frame < window["to"]:
        return frame
    return max(s for s in window["steps"] if s <= frame)


def normalize(entries, cast, fps, frame0, last_frame=None):
    """[{index, cast: [names], from, to (Blender frames; the window holds from <= frame < to), fps, steps}] in time
    order: `cast` holds the project's member names (an entry without one takes them all), `fps` is the clip's,
    `last_frame` the clip's last Blender frame (None: not checked). A member is in one window at a time."""
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise StepError("[[stepped]]: expected an array of tables")
    out = []
    for i, e in enumerate(entries):
        where = f"stepped {i}"
        if not isinstance(e, dict):
            raise StepError(f"{where}: expected a table {{from, to, cast, fps}}")
        unknown = sorted(set(e) - set(KEYS))
        if unknown:
            raise StepError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
        for k in ("from", "to"):
            if isinstance(e.get(k), bool) or not isinstance(e.get(k), (int, float)):
                raise StepError(f"{where}: {k} = {e.get(k)!r}: clip seconds")
        f0, f1 = int(round(frame0 + e["from"] * fps)), int(round(frame0 + e["to"] * fps))
        if f1 <= f0:
            raise StepError(f"{where}: {e['from']} .. {e['to']} s is under a frame")
        if f0 < frame0 or (last_frame is not None and f1 - 1 > last_frame):
            raise StepError(f"{where}: frames {f0}..{f1 - 1} reach outside the clip ({frame0}..{last_frame})")
        rate = e.get("fps", FPS)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= fps:
            raise StepError(f"{where}: fps = {rate!r}: poses a second, above 0 and at most the clip's {fps:g}")
        names = e.get("cast", list(cast))
        names = [names] if isinstance(names, str) else names
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise StepError(f"{where}: cast = {e.get('cast')!r}: a member's name or a list of them")
        missing = sorted(set(names) - set(cast))
        if missing:
            raise StepError(f"{where}: no cast member {', '.join(repr(n) for n in missing)} "
                            f"(cast: {', '.join(cast) or 'none'})")
        out.append({"index": i, "cast": names, "from": f0, "to": f1, "fps": float(rate),
                    "steps": steps(f0, f1, fps, float(rate))})
    out.sort(key=lambda w: w["from"])
    for a in out:
        for b in out:
            if a["index"] < b["index"] and a["from"] < b["to"] and b["from"] < a["to"] and set(a["cast"]) & set(b["cast"]):
                raise StepError(f"stepped {a['index']} and stepped {b['index']} overlap for "
                                f"{', '.join(sorted(set(a['cast']) & set(b['cast'])))}: one rate at a time")
    return out
