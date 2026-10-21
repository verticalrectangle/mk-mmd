"""Hit lanes, bpy-free (docs/design.md: Timeline, hit lanes): the moments of a project's `[audio] hits` file (`mk timeline
hits` writes it) picked for something that fires on them: a `[[key]]` on a prop's `start`, a zoom punch, a shock ring. A
lane is a list of hits, each a clip second or a table `{t, db, ...}` (`db`: the hit's level against the lane's loudest,
0 the loudest); `pick` keeps the hits of one lane inside a span, at or above a level, every n-th one."""
import json
import math

KEYS = ("lane", "min_db", "every", "from", "to")      # what a table that fires on a lane may say


class LaneError(ValueError):
    """A lane that cannot be read or picked from: a missing file or lane, a bad hit or a bad choice."""


def load(path):
    """The hits file at `path` as {lane: [hit, ...]}."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise LaneError(f"hits file {path}: {e}") from None
    if not isinstance(doc, dict):
        raise LaneError(f"hits file {path}: not a table of lanes")
    return doc


def _hit(h, lane, i):
    if isinstance(h, (int, float)) and not isinstance(h, bool):
        return float(h), None
    if isinstance(h, dict) and isinstance(h.get("t"), (int, float)):
        db = h.get("db")
        return float(h["t"]), None if db is None else float(db)
    raise LaneError(f"lane {lane!r}: hit {i} is neither clip seconds nor {{t, db}}")


def pick(hits, lane, min_db=None, every=1, t0=None, t1=None):
    """Clip seconds of the hits of `lane` in `hits` ({lane: [...]}), sorted: those from `t0` up to (not including) `t1`,
    at or above `min_db` (a hit without a level is kept only when no level is asked for), then every `every`-th one,
    starting with the first."""
    if lane not in hits:
        raise LaneError(f"no lane {lane!r} in the hits file (have {', '.join(sorted(hits))})")
    if not isinstance(hits[lane], list):
        raise LaneError(f"lane {lane!r}: not a list of hits")
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise LaneError(f"every = {every!r}: a whole number, at least 1")
    lo = -math.inf if t0 is None else float(t0)
    hi = math.inf if t1 is None else float(t1)
    out = []
    for i, h in enumerate(hits[lane]):
        t, db = _hit(h, lane, i)
        if not lo <= t < hi:
            continue
        if min_db is not None and (db is None or db < float(min_db)):
            continue
        out.append(t)
    return sorted(out)[::every]


def choice(spec, what):
    """The lane choice of a table that fires on hits (KEYS): {lane, min_db, every, t0, t1}, checked."""
    lane = spec.get("lane")
    if not isinstance(lane, str) or not lane:
        raise LaneError(f"{what}: `lane` names a lane of the [audio] hits file")
    out = {"lane": lane, "min_db": spec.get("min_db"), "every": spec.get("every", 1), "t0": spec.get("from"),
           "t1": spec.get("to")}
    for k in ("min_db", "t0", "t1"):
        v = out[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise LaneError(f"{what}: {'from' if k == 't0' else 'to' if k == 't1' else k} = {v!r}: a number")
    if isinstance(out["every"], bool) or not isinstance(out["every"], int) or out["every"] < 1:
        raise LaneError(f"{what}: every = {out['every']!r}: a whole number, at least 1")
    return out
