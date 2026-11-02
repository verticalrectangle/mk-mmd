"""Shared bits of the show props (library: speaker_stack, penlight, idol_stage, confetti), bpy-free so their layouts and
tests import it: colours resolved from the `slots` the props stage hands a builder (the project's palette with the
prop's own overrides) and the checks of their options.

A colour is a palette slot (`love`), `ink` or `paper` (the palette's darkest and lightest), a `#hex`, or a mix of two
such (`rose:text:0.4`, the second weighing 0.4); a colour role of a prop is its `slots` entry when it has one, else the
role's default. Everything is resolved on every call from the slots given (no colour is kept between builds), so a
palette that breathes over the clip re-colours the props."""
import math

from ....core import palette as PAL
from ....core import shotstyle as SS

SLOTS = frozenset(PAL.PALETTES["rose-pine"])     # the palette's slot names


def _luma(hexstr):
    r, g, b = PAL.linear(hexstr)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def palette(slots, extremes=True):
    """{slot: '#hex'}: the palette slots among a prop's `slots` (the props stage hands every builder the project's whole
    palette), over the default palette for a slot they lack (a builder called on its own, in a test), plus (`extremes`)
    `ink` and `paper`: the palette's darkest and lightest colours (base and text in the dark Rosé Pines, the other way
    round in Dawn), for what is dark or light by nature whatever the palette (a speaker cone, a hole)."""
    p = {**PAL.get(), **{k: v for k, v in (slots or {}).items() if k in SLOTS and isinstance(v, str)}}
    if extremes:
        lum = {k: _luma(v) for k, v in p.items()}
        p["ink"], p["paper"] = p[min(lum, key=lum.get)], p[max(lum, key=lum.get)]
    return p


def rgb(spec, slots, what="colour"):
    """Linear RGB (what Blender's colour inputs take) of a colour `spec` (a slot, `ink`, `paper`, '#hex' or 'a:b:t'),
    resolved from `slots`. A bad spec is a ValueError that names `what`."""
    extremes = isinstance(spec, str) and any(part in ("ink", "paper") for part in spec.split(":"))
    try:
        c = SS.resolve_colour(spec, palette(slots, extremes))
    except SS.StyleError as e:
        raise ValueError(f"{what}: {e}") from None
    return tuple(SS.srgb_to_linear(v) for v in c)


def rgba(spec, slots, what="colour"):
    return (*rgb(spec, slots, what), 1.0)


def role(slots, name, roles, prop):
    """The colour spec of the colour role `name`: the prop's `slots` entry for it, else its default in `roles`."""
    spec = (slots or {}).get(name, roles[name])
    rgb(spec, slots, f"{prop}: {name}")                    # checked here, so a bad role fails before anything is built
    return spec


def roles(slots, defaults, prop):
    """{role: spec} of every role in `defaults`, checked (the card's `slots`)."""
    return {k: role(slots, k, defaults, prop) for k in defaults}


def number(slots, key, default, prop, lo=-math.inf, hi=math.inf, whole=False):
    """The option `key` of a prop's `slots` (or `default`), a finite number in lo..hi (a whole number with `whole`);
    anything else is a ValueError that says what to change."""
    v = (slots or {}).get(key, default)
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi
    if ok and whole:
        ok = float(v) == int(v)
    if not ok:
        rng = f" {lo:g} .. {hi:g}" if math.isfinite(lo) and math.isfinite(hi) else \
            f" >= {lo:g}" if math.isfinite(lo) else f" <= {hi:g}" if math.isfinite(hi) else ""
        raise ValueError(f"{prop}: {key} is {'a whole number' if whole else 'a number'}{rng} (got {v!r})")
    return int(v) if whole else float(v)
