"""A palette that breathes (docs/design.md: Lights and the look; Post): `[look] breathe` mixes the project's palette (mix 0)
with another one (mix 1) over clip seconds, so the world turns from one to the other and back.

    [look.breathe]
    palette = "rose-pine-dawn"          # the other palette (its slot names are the project's)
    slots = { gold = "#e8a33a" }        # overrides of the other palette's slots (the project's [look.slots] are its own)
    keys = [[0, 0], [12, 1], [24, 1], [29.6, 0]]   # [t, mix] or [t, mix, ease]: clip seconds, 0..1
    ease = "smooth"                     # between keys: "smooth" (default), "linear" or "hold"
    lights = { "*" = 0.6, sun = 3.0 }   # light strengths at mix 1, times their own (a number: every light)

The mix is held before the first key and after the last; a key's own ease (its third item) shapes the way from it to the
next. Slots the project's [look.slots] adds under names no palette has carry over to the other palette unchanged.

The build drives every value it took from the palette between its two builds (mkmmd/blender/build/breathe.py: colours
mixed in a display-like space, `GAMMA`, numbers linearly); post resolves every colour with both palettes and mixes the two
by the frame's mix (`Pair`: colours as display-space floats mix in sRGB, as `"a:b:t"` mixes do). numpy-free: runs in
Blender and in the CLI."""
import bisect
import fnmatch
import re

from . import palette as PAL

KEYS = ("palette", "slots", "keys", "ease", "lights")
EASES = ("smooth", "linear", "hold")
GAMMA = 2.2                     # the build mixes colours as c ** (1 / GAMMA), close to sRGB's curve
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class BreatheError(ValueError):
    pass


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BreatheError(f"{what} = {v!r}: expected a number")
    return float(v)


def spec(look, what="[look] breathe"):
    """`[look] breathe` checked and filled in: {"palette", "slots", "keys": [[t, mix, ease], ...], "ease", "lights":
    [[pattern, factor], ...]}, or None when the project does not breathe. A BreatheError says what is wrong."""
    v = (look or {}).get("breathe")
    if v is None or v is False:
        return None
    if not isinstance(v, dict):
        raise BreatheError(f"{what} is a table {{palette, keys, ease, slots, lights}}")
    bad = sorted(set(v) - set(KEYS))
    if bad:
        raise BreatheError(f"{what}: unknown key{'s' if len(bad) > 1 else ''} {', '.join(bad)} (known: {', '.join(KEYS)})")
    name = v.get("palette")
    if name not in PAL.PALETTES:
        raise BreatheError(f"{what}: palette = {name!r}: one of {', '.join(sorted(PAL.PALETTES))}")
    slots = v.get("slots") or {}
    if not isinstance(slots, dict):
        raise BreatheError(f"{what}: slots is a table {{slot = \"#rrggbb\"}}")
    for k, c in slots.items():
        if not isinstance(c, str) or not HEX.match(c):
            raise BreatheError(f"{what}: slots.{k} = {c!r}: expected \"#rrggbb\"")
    ease = v.get("ease", "smooth")
    if ease not in EASES:
        raise BreatheError(f"{what}: ease = {ease!r}: one of {', '.join(EASES)}")
    keys = v.get("keys")
    if not isinstance(keys, list) or not keys:
        raise BreatheError(f"{what}: keys = [[t, mix], ...] (clip seconds, mix 0..1) is required")
    out, last = [], None
    for i, k in enumerate(keys):
        kw = f"{what}: keys[{i}]"
        if not isinstance(k, (list, tuple)) or len(k) not in (2, 3):
            raise BreatheError(f"{kw} = {k!r}: expected [t, mix] or [t, mix, ease]")
        t, m = _num(k[0], f"{kw} t"), _num(k[1], f"{kw} mix")
        if not 0.0 <= m <= 1.0:
            raise BreatheError(f"{kw}: mix = {m}: expected 0..1 (0 the project's palette, 1 {name})")
        e = k[2] if len(k) == 3 else ease
        if e not in EASES:
            raise BreatheError(f"{kw}: ease = {e!r}: one of {', '.join(EASES)}")
        if last is not None and t < last:
            raise BreatheError(f"{kw}: t = {t} comes before the key before it ({last}): keys are in time order")
        out.append([t, m, e])
        last = t
    lights = v.get("lights", 1.0)
    if isinstance(lights, dict):
        pairs = [[str(p), _num(f, f"{what}: lights.{p}")] for p, f in lights.items()]
    else:
        pairs = [["*", _num(lights, f"{what}: lights")]]
    for p, f in pairs:
        if f < 0:
            raise BreatheError(f"{what}: lights.{p} = {f}: a factor is 0 or more")
    return {"palette": name, "slots": dict(slots), "keys": out, "ease": ease, "lights": pairs}


def palettes(look, what="[look] breathe"):
    """(the project's palette, the other one or None): {slot: "#hex"} each. The project's [look.slots] that name a slot
    of the other palette are the project's palette's own; those under names no palette has carry over."""
    look = look or {}
    a = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
    sp = spec(look, what)
    if sp is None:
        return a, None
    base = PAL.get(sp["palette"])
    own = {k: c for k, c in (look.get("slots") or {}).items() if k not in base}
    missing = sorted(set(sp["slots"]) - set(base) - set(own))
    if missing:
        raise BreatheError(f"{what}: slots {', '.join(missing)}: not slots of {sp['palette']} ({', '.join(sorted(base))})")
    return a, {**base, **own, **sp["slots"]}


def mix_at(sp, t):
    """The mix (0: the project's palette, 1 the other) at clip second `t`: held outside the keys, each key's ease from it
    to the next ("smooth": smoothstep, "linear", "hold": its own value until the next key). Two keys at one time are a
    jump: from that time on the later one counts."""
    keys = sp["keys"]
    times = [k[0] for k in keys]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        return keys[0][1]
    if i >= len(keys) - 1:
        return keys[-1][1]
    (t0, m0, ease), (t1, m1, _) = keys[i], keys[i + 1]
    u = (t - t0) / (t1 - t0)
    w = 0.0 if ease == "hold" else u if ease == "linear" else u * u * (3.0 - 2.0 * u)
    return m0 + (m1 - m0) * w


def frame_mix(sp, fps, frame0, frame):
    """The mix on Blender frame `frame` (clip second (frame - frame0) / fps)."""
    return mix_at(sp, (frame - frame0) / float(fps))


def mixed(a, b, m):
    """The palette `m` of the way from `a` to `b`, {slot: "#hex"}, each slot mixed in sRGB as `"a:b:t"` mixes are."""
    return {k: PAL.mix(a[k], b[k], m) if k in b else a[k] for k in a}


def light_factor(sp, name):
    """A light's strength factor at mix 1 (`lights`): the first pattern that matches its object's name, else 1."""
    for p, f in sp["lights"]:
        if fnmatch.fnmatchcase(name, p):
            return f
    return 1.0


# ------------------------------------------------------------------------------------------- the build's driver maths
def _g(v):
    return f"{v:.9g}"


def colour_expr(a, b, var="m"):
    """A driver expression (Blender's simple expressions: no Python) for one linear colour channel that is `a` at mix 0
    and `b` at mix 1, mixed as c ** (1 / GAMMA): close to how `"a:b:t"` mixes in sRGB."""
    pa, pb = max(float(a), 0.0) ** (1.0 / GAMMA), max(float(b), 0.0) ** (1.0 / GAMMA)
    return f"pow(lerp({_g(pa)}, {_g(pb)}, {var}), {_g(GAMMA)})"


def number_expr(a, b, var="m"):
    """A driver expression for a number that is `a` at mix 0 and `b` at mix 1, mixed linearly."""
    return f"lerp({_g(float(a))}, {_g(float(b))}, {var})"


def colour_at(a, b, m):
    """What `colour_expr(a, b)` evaluates to at mix `m` (the tests' and the bake check's reference)."""
    pa, pb = max(float(a), 0.0) ** (1.0 / GAMMA), max(float(b), 0.0) ** (1.0 / GAMMA)
    return (pa + (pb - pa) * m) ** GAMMA


# ---------------------------------------------------------------------------------------------- post: two resolutions
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _array(v):
    return hasattr(v, "dtype") and hasattr(v, "shape")


class Pair:
    """The same thing resolved with each palette (`a` with the project's, `b` with the other: a normalised [[fx]] entry,
    a plan, a list of colours): `at(m)` is `a` with every number that differs mixed `m` of the way toward `b`'s, and
    every "#rrggbb" that differs mixed in sRGB; containers along the way are copied, the rest is shared with `a`.
    Colours as display-space floats so mix in sRGB, as `"a:b:t"` mixes do. Two that differ otherwise (a string, a flag,
    a key, a length, an integer) are a BreatheError naming the place: the thing is not the same thing in both."""

    def __init__(self, a, b, what="it"):
        self.a, self.b = a, b
        self.diffs = []                     # (path, a value, b value)
        self._walk(a, b, (), what)
        self.static = not self.diffs

    def _walk(self, a, b, path, what):
        def where():
            return what + "".join(f"[{p!r}]" for p in path)
        if a is b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if list(a) != list(b):
                if set(a) != set(b):
                    raise BreatheError(f"{where()}: keys {sorted(map(str, set(a) ^ set(b)))} only in one palette's")
            for k in a:
                self._walk(a[k], b[k], path + (k,), what)
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and type(a) is type(b):
            if len(a) != len(b):
                raise BreatheError(f"{where()}: {len(a)} items with one palette, {len(b)} with the other")
            for i, (x, y) in enumerate(zip(a, b)):
                self._walk(x, y, path + (i,), what)
            return
        if _array(a) and _array(b):
            if a.shape != b.shape:
                raise BreatheError(f"{where()}: shapes {a.shape} and {b.shape} differ between the palettes")
            if (a == b).all():
                return
            if a.dtype.kind != "f" or b.dtype.kind != "f":
                raise BreatheError(f"{where()}: whole numbers that differ between the palettes cannot be mixed")
            self.diffs.append((path, a, b))
            return
        if _is_num(a) and _is_num(b):
            if a == b:
                return
            if isinstance(a, int) and isinstance(b, int):
                raise BreatheError(f"{where()}: {a} with one palette, {b} with the other: a whole number cannot be mixed")
            self.diffs.append((path, float(a), float(b)))
            return
        if isinstance(a, str) and isinstance(b, str):
            if a == b:
                return
            if HEX.match(a) and HEX.match(b):
                self.diffs.append((path, a, b))
                return
        elif type(a) is type(b) and a == b:
            return
        raise BreatheError(f"{where()}: {a!r} with one palette, {b!r} with the other: only numbers and colours can breathe")

    def _get(self, path):
        v = self.a
        for p in path:
            v = v[p]
        return v

    def at(self, m):
        """`a` at mix `m` (0: `a` itself, 1: `b` itself)."""
        if self.static or m <= 0.0:
            return self.a
        if m >= 1.0:
            return self.b
        copies = {}                         # path -> a mutable copy of `a`'s container there, held by its parent's copy

        def own(path):
            if path not in copies:
                src = self._get(path)
                copies[path] = dict(src) if isinstance(src, dict) else list(src) if isinstance(src, (list, tuple)) \
                    else src.copy()
                if path:
                    own(path[:-1])[path[-1]] = copies[path]
            return copies[path]

        for path, va, vb in self.diffs:
            v = PAL.mix(va, vb, m) if isinstance(va, str) else va + (vb - va) * m
            if not path:
                return v
            own(path[:-1])[path[-1]] = v
        for path in sorted(copies, key=len, reverse=True):            # tuples back into tuples, deepest first
            if isinstance(self._get(path), tuple):
                copies[path] = tuple(copies[path])
                if path:
                    copies[path[:-1]][path[-1]] = copies[path]
        return copies[()]
