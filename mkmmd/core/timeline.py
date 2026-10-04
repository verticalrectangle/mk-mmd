"""Timeline files (docs/design.md: Timeline): beats, downbeats, sections and word timings of a clip, in clip seconds.
Older timelines keep beats under "tempo"; both layouts are read."""


def beats(tl):
    """(beats, downbeats) lists in clip seconds."""
    b = tl.get("beats") or (tl.get("tempo") or {}).get("beats") or []
    d = tl.get("downbeats") or (tl.get("tempo") or {}).get("downbeats") or []
    return list(b), list(d)


def ticks(tl):
    """Tempo ticks of a timeline in clip seconds, sorted: its `ticks` (or `tempo.ticks`) when it has them, else its beats
    (the only tempo marks `mk timeline analyze` writes)."""
    t = tl.get("ticks") or (tl.get("tempo") or {}).get("ticks") or beats(tl)[0]
    return sorted(float(x) for x in t)


def bpm(tl):
    return tl.get("bpm") or (tl.get("tempo") or {}).get("bpm")
