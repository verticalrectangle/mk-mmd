"""The `[[shot]]` table, bpy-free (docs/design.md: Shots): the keys a shot takes, the per-output tables laid over it, and the
checks the shots stage makes before it builds a camera. The stage reads the keys; this is the list of the ones it reads (and
`mkmmd.core.shotstyle` the look's), so a key added to the stage is added here too."""
from .screentype import deep_merge

LOOK = ("style", "colors", "tones", "hide", "keep", "accent", "tint", "knockout", "grow", "samples", "reflection")
OWN = ("name", "from", "to", "plate", "aspect")                     # a shot's own: no per-output table changes them
SHOT = OWN + ("mount", "at", "look", "lens", "roll", "lag", "shake", "keys", "frame", "shift", "dof", "follow") + LOOK
ASPECT = tuple(k for k in SHOT if k not in OWN)
FRAME = ("subject", "fill", "solve")
SOLVES = ("lens", "distance", "dolly")
DOF = ("focus", "fstop", "offset")
FOLLOW = ("target", "axes", "lag", "from")
KEYS = ("t", "at", "look", "lens", "roll", "shift")
TABLES = ("frame", "dof", "follow", "colors", "tones", "knockout", "reflection")   # laid over the shot's own key by key


class ShotError(ValueError):
    """A shot the stage cannot build: an unknown key or output, or `from` without `to`."""


def _keys(table, known, where):
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ShotError(f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(repr(k) for k in unknown)} "
                        f"(known: {', '.join(known)})")


def _body(spec, known, where):
    """The keys of a shot, or of one output's table over it, and of the tables inside them."""
    _keys(spec, known, where)
    for key, inner in (("frame", FRAME), ("dof", DOF), ("follow", FOLLOW)):
        if isinstance(spec.get(key), dict):
            _keys(spec[key], inner, f"{where} {key}")
    if isinstance(spec.get("keys"), list):
        for i, k in enumerate(spec["keys"]):
            if isinstance(k, dict):
                _keys(k, KEYS, f"{where} keys[{i}]")


def follow_spec(sp, where):
    """The `follow` of a shot as one output sees it (`merged`), checked: None when it has none (or `follow = false`),
    else {"target", "axes": (indices of x, y, z), "lag": seconds, "from": clip seconds or None (the shot's `from`)}."""
    fol = sp.get("follow")
    if fol is None or fol is False:
        return None
    if not isinstance(fol, dict):
        raise ShotError(f"{where}: follow is a table {{target, axes, lag, from}} (or false in an aspect table)")
    if fol.get("target") is None:
        raise ShotError(f"{where} follow: give `target`, what the camera moves with (a target, e.g. \"cast:rin\")")
    axes = fol.get("axes", "x")
    if not isinstance(axes, str) or not axes or set(axes) - set("xyz") or len(set(axes)) != len(axes):
        raise ShotError(f"{where} follow: axes = {axes!r} is not a string of distinct letters from \"xyz\" (\"x\", \"xz\")")
    lag = fol.get("lag", 0.0)
    if isinstance(lag, bool) or not isinstance(lag, (int, float)) or lag < 0:
        raise ShotError(f"{where} follow: lag = {lag!r} is not seconds (0 or more)")
    t0 = fol.get("from")
    if t0 is not None and (isinstance(t0, bool) or not isinstance(t0, (int, float))):
        raise ShotError(f"{where} follow: from = {t0!r} is not clip seconds")
    if sp.get("mount"):
        raise ShotError(f"{where}: `follow` and `mount` together: a camera rides a mount or follows a target, not both")
    return {"target": fol["target"], "axes": tuple("xyz".index(c) for c in axes), "lag": float(lag),
            "from": None if t0 is None else float(t0)}


def _settings(sp, where):
    """What a shot as one output sees must make sense: its `follow` and its `frame` solve."""
    follow_spec(sp, where)
    fr = sp.get("frame")
    if isinstance(fr, dict) and fr.get("solve", "lens") not in SOLVES:
        raise ShotError(f"{where} frame: solve = {fr['solve']!r} is none of {', '.join(SOLVES)}")


def check(spec, outputs):
    """ShotError for a shot with an unknown key (in the shot, `frame`, `dof`, `follow`, `keys[]` or an `aspect` table), a
    plate with `from` but no `to` (or the reverse), an `aspect` table for an output the project does not have (`outputs`:
    their names), or, as any output sees the shot, a `follow` that is wrong or rides with a `mount`, or an unknown solve."""
    where = f"shot {spec.get('name')!r}"
    _body(spec, SHOT, where)
    if ("from" in spec) != ("to" in spec) and spec.get("plate"):
        raise ShotError(f"{where}: a plate takes `from` and `to` together (or neither): it has only "
                        f"`{'from' if 'from' in spec else 'to'}`")
    aspects = spec.get("aspect") or {}
    if not isinstance(aspects, dict):
        raise ShotError(f"{where}: `aspect` holds one table per output, [shot.aspect.<output>]")
    for out, over in aspects.items():
        if out not in outputs:
            raise ShotError(f"{where}: [shot.aspect.{out}] is for an output the project does not have "
                            f"(outputs: {', '.join(outputs)})")
        if not isinstance(over, dict):
            raise ShotError(f"{where}: [shot.aspect.{out}] is a table of the keys it changes")
        _body(over, ASPECT, f"{where} aspect.{out}")
    _settings(spec, where)                                         # the shot as it is, then as each output's table makes it
    for out, over in aspects.items():
        _settings(merged(spec, over), f"{where} aspect.{out}")


def merged(spec, over):
    """The shot as one output sees it: its own keys with the `aspect.<output>` table `over` laid on. The tables of TABLES
    (`frame`, `dof`, the look's) merge key by key, so an output can change `frame = {fill = 0.6}` and keep the `subject`;
    every other key (`at`, `look`, `keys`, targets all) is replaced whole."""
    out = dict(spec)
    for k, v in (over or {}).items():
        out[k] = deep_merge(spec[k], v) if k in TABLES and isinstance(v, dict) and isinstance(spec.get(k), dict) else v
    return out
