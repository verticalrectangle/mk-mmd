"""Timeline files (docs/design.md: Timeline): beats, downbeats, sections and word timings of a clip, in clip seconds.
Older timelines keep beats under "tempo"; both layouts are read."""


def beats(tl):
    """(beats, downbeats) lists in clip seconds."""
    b = tl.get("beats") or (tl.get("tempo") or {}).get("beats") or []
    d = tl.get("downbeats") or (tl.get("tempo") or {}).get("downbeats") or []
    return list(b), list(d)


def bpm(tl):
    return tl.get("bpm") or (tl.get("tempo") or {}).get("bpm")
