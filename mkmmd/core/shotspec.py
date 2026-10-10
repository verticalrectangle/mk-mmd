"""The `[[shot]]` table, bpy-free (docs/design.md: Shots): the keys a shot takes, the per-output tables laid over it, and the
checks the shots stage makes before it builds a camera. The stage reads the keys; this is the list of the ones it reads (and
`mkmmd.core.shotstyle` the look's), so a key added to the stage is added here too."""
from .screentype import deep_merge

LOOK = ("style", "colors", "tones", "hide", "keep", "accent", "tint", "knockout", "grow", "samples", "reflection")
OWN = ("name", "from", "to", "plate", "aspect")                     # a shot's own: no per-output table changes them
SHOT = OWN + ("mount", "at", "look", "lens", "roll", "lag", "shake", "keys", "frame", "shift", "dof") + LOOK
ASPECT = tuple(k for k in SHOT if k not in OWN)
FRAME = ("subject", "fill", "solve")
DOF = ("focus", "fstop", "offset")
KEYS = ("t", "at", "look", "lens", "shift")
TABLES = ("frame", "dof", "colors", "tones", "knockout", "reflection")   # laid over the shot's own key by key; the rest is replaced


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
    for key, inner in (("frame", FRAME), ("dof", DOF)):
        if isinstance(spec.get(key), dict):
            _keys(spec[key], inner, f"{where} {key}")
    if isinstance(spec.get("keys"), list):
        for i, k in enumerate(spec["keys"]):
            if isinstance(k, dict):
                _keys(k, KEYS, f"{where} keys[{i}]")


def check(spec, outputs):
    """ShotError for a shot with an unknown key (in the shot, `frame`, `dof`, `keys[]` or an `aspect` table), a plate with `from`
    but no `to` (or the reverse), or an `aspect` table for an output the project does not have (`outputs`: their names)."""
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


def merged(spec, over):
    """The shot as one output sees it: its own keys with the `aspect.<output>` table `over` laid on. The tables of TABLES
    (`frame`, `dof`, the look's) merge key by key, so an output can change `frame = {fill = 0.6}` and keep the `subject`;
    every other key (`at`, `look`, `keys`, targets all) is replaced whole."""
    out = dict(spec)
    for k, v in (over or {}).items():
        out[k] = deep_merge(spec[k], v) if k in TABLES and isinstance(v, dict) and isinstance(spec.get(k), dict) else v
    return out
