"""A burst, bpy-free (the library prop `burst`, docs/design.md: the library table): pieces (puffy hearts, stars and smoke
puffs) flung out from one point at a clip second, each easing out to its own reach while it spins and pops in, then
shrinking away. Driver expressions on the frame do it (mkmmd/blender/library/props/burst.py lays them on the pieces), so
a freeze (mkmmd.core.freeze), which holds the scene on a frame, holds them mid-air.

The pieces fly mostly across the picture (the directions are flattened toward the frame's plane, the prop's x-z: it
faces -Y like a character), each with its own reach, size, a little delay and a spin; all from the `seed`, the same on
every machine. Every piece is a pillow (mkmmd.core.hearts.pillow) of its outline: the heart's, a five-point star's
with rounded tips, a cloud's."""
import hashlib
import math

import numpy as np

from . import hearts as HR

KINDS = ("heart", "star", "puff")
PARAMS = {  # root property: (default, unit, doc); keyed with [[key]]
    "start": (0.0, "s", "the clip second it goes off"),
    "reach": (1.2, "m", "how far the farthest piece flies"),
    "life": (1.4, "s", "how long a piece lives, out and away"),
    "spin": (360.0, "deg", "how far a piece turns over its life"),
    "size": (0.16, "m", "the largest piece's width"),
    "amount": (1.0, "", "overall scale of every piece (0 hides them)"),
}
MIX = {"heart": 0.45, "star": 0.35, "puff": 0.2}  # default share of each kind
COLOURS = {"heart": "love", "star": "gold", "puff": "text"}
POP = 0.08                       # share of its life a piece takes to pop in
FADE = 0.7                       # share of its life after which it shrinks away
FLAT = 0.45                      # how much of a direction toward or away from the camera is kept
TUMBLE = 20.0                    # degrees a piece tumbles out of the picture's plane at most: it keeps showing its face
MIN_SCALE = HR.MIN_SCALE
HELPERS = ("u",)                 # custom property of each piece, driven: its age as a share of its life
FUNCS = HR.FUNCS


def _unit(seed, *key):
    """A deterministic number in [0, 1) from a seed and a key (the same on every platform)."""
    h = hashlib.blake2b(("/".join(str(k) for k in (seed,) + key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


def options(slots):
    """(count, seed, mix {kind: share}, params {root property: value}, colours {kind: spec}, glow) from the `slots` of the
    [[prop]], checked: a bad value is a ValueError that says what to change."""
    slots = slots or {}
    count, seed = slots.get("count", 24), slots.get("seed", 1)
    if not isinstance(count, int) or not 1 <= count <= 200:
        raise ValueError(f"burst: count is a whole number of pieces, 1 .. 200 (got {count!r})")
    if not isinstance(seed, int):
        raise ValueError(f"burst: seed is a whole number (got {seed!r})")
    mix = slots.get("mix", MIX)
    if not isinstance(mix, dict) or set(mix) - set(KINDS) or not mix or \
            not all(isinstance(v, (int, float)) and v >= 0 for v in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError(f"burst: mix is {{kind: share}} of {', '.join(KINDS)}, shares >= 0 (got {mix!r})")
    P = {}
    for k, (default, unit, _) in PARAMS.items():
        v = slots.get(k, default)
        if not isinstance(v, (int, float)) or (k != "start" and v < 0) or (k in ("reach", "life", "size") and v <= 0):
            raise ValueError(f"burst: {k} must be a {'positive ' if k in ('reach', 'life', 'size') else ''}number"
                             f"{' (' + unit + ')' if unit else ''} (got {v!r})")
        P[k] = float(v)
    colours = {k: slots.get(k, COLOURS[k]) for k in KINDS}
    glow = slots.get("glow", 0.3)
    if not isinstance(glow, (int, float)) or glow < 0:
        raise ValueError(f"burst: glow is a number >= 0 (got {glow!r})")
    return count, seed, {k: float(v) for k, v in mix.items()}, P, colours, float(glow)


def layout(count, seed, mix):
    """[{kind, dir (3,), reach (share), size (share), delay (share of the life), spin (share, +-), tumble (+-1)}] of the
    pieces: the kinds in proportion to `mix` (largest remainder), spread round the whole burst, every number from the seed
    and rounded as the expressions print it (so `pose` is what the drivers do)."""
    total = sum(mix.values())
    want = {k: count * v / total for k, v in mix.items()}
    n = {k: int(math.floor(w)) for k, w in want.items()}
    for k in sorted(want, key=lambda k: (n[k] - want[k], KINDS.index(k)))[:count - sum(n.values())]:
        n[k] += 1
    kinds = [k for k in KINDS for _ in range(n.get(k, 0))]
    order = sorted(range(count), key=lambda i: _unit(seed, "order", i))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out, R = [], lambda x: round(float(x), HR.NDIGITS)                 # noqa: E731
    for slot, i in enumerate(order):
        z = 1.0 - 2.0 * (slot + 0.5) / count                     # a Fibonacci sphere: no clumps, no gaps
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = golden * slot + 2.0 * math.pi * _unit(seed, "turn")
        d = np.array([r * math.cos(a), FLAT * r * math.sin(a), z])  # toward or away from the camera, a piece flies less far
        out.append({"kind": kinds[i], "dir": np.round(d, HR.NDIGITS),
                    "reach": R(0.55 + 0.45 * _unit(seed, "reach", i)), "size": R(0.5 + 0.5 * _unit(seed, "size", i)),
                    "delay": R(0.06 * _unit(seed, "delay", i)),
                    "spin": R((1 if _unit(seed, "sy", i) < 0.5 else -1) * (0.4 + 0.6 * _unit(seed, "ay", i))),
                    "tumble": 1 if _unit(seed, "tx", i) < 0.5 else -1})
    return out


def pose(t, P, piece):
    """Where piece `piece` is at clip second(s) t: (position (..., 3) in the prop's frame, rx (the tumble), ry (the spin
    in the picture's plane) radians, scale), as the drivers have it."""
    t = np.asarray(t, float)
    u = np.clip((t - P["start"] - piece["delay"] * P["life"]) / P["life"], 0.0, 1.0)
    out = 1.0 - (1.0 - u) ** 3
    pos = np.multiply.outer(out * P["reach"] * piece["reach"], piece["dir"])
    fade = HR._smoothstep(FADE, 1.0, u)
    s = np.maximum(MIN_SCALE, P["size"] * piece["size"] * P["amount"] * np.minimum(u / POP, 1.0) * (1.0 - fade))
    ry = math.radians(P["spin"]) * u * piece["spin"]
    return pos, math.radians(TUMBLE) * np.sin(u * 9.0) * piece["tumble"], ry, s


def expressions(piece):
    """The driver expressions of a piece: {channel: text} for u (its helper, first), x, y, z, rx (a tumble of +-TUMBLE
    degrees, so it always shows its face), ry (its spin in the picture's plane), s. They read the root's properties by
    their names in PARAMS, `f0` (the frame of clip second 0), `fps`, `frame` and the piece's `u`."""
    n = HR._n
    d = piece["dir"]
    reach = f"reach * {n(piece['reach'])} * (1 - (1 - u) * (1 - u) * (1 - u))"
    out = {"u": f"clamp(((frame - f0) / fps - start - {n(piece['delay'])} * life) / life, 0, 1)"}
    for ch, c in zip("xyz", d):
        out[ch] = f"{n(c)} * {reach}"
    out["rx"] = f"radians({n(TUMBLE)}) * sin(u * 9) * {n(piece['tumble'])}"
    out["ry"] = f"radians(spin) * u * {n(piece['spin'])}"
    out["s"] = (f"max({n(MIN_SCALE)}, size * {n(piece['size'])} * amount * min(u / {n(POP)}, 1) * "
                f"(1 - smoothstep({n(FADE)}, 1, u)))")
    for ch, text in out.items():
        if len(text) > HR.MAX_EXPR:
            raise ValueError(f"burst: the {ch} expression is {len(text)} characters (Blender keeps {HR.MAX_EXPR})")
    return out


def star_outline(n=80, points=5, inner=0.48, rounding=6.0):
    """A star's outline with rounded tips: (n, 2) points (x, z), unit width, round the origin counter-clockwise seen
    from the front, a tip straight up."""
    k = np.arange(2 * points)
    ang = math.pi / 2 + math.pi * k / points
    rad = np.where(k % 2 == 0, 1.0, inner)
    P = HR._resample_closed(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), 1024)
    F = np.fft.fft(P[:, 0] + 1j * P[:, 1])
    f = np.fft.fftfreq(len(F), 1.0 / len(F))
    F = np.fft.ifft(F * np.exp(-(f / float(rounding * points)) ** 2))
    Q = HR._resample_closed(np.stack([F.real, F.imag], 1), n)
    Q -= HR.area_centroid(Q)
    return Q / (Q[:, 0].max() - Q[:, 0].min())


def puff_outline(n=72, lobes=6, bump=0.13):
    """A cloud's outline: a circle with `lobes` round bumps, (n, 2) points (x, z), unit width, counter-clockwise."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    r = 1.0 + bump * np.cos(lobes * t)
    Q = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return Q / (Q[:, 0].max() - Q[:, 0].min())


def mesh(kind):
    """The unit piece of a kind (`core.shell.Mesh`, unit width, facing -Y): a heart, a star, a cloud puff."""
    if kind == "heart":
        return HR.heart_mesh()
    if kind == "star":
        return HR.pillow(star_outline(), 0.38, 16)
    if kind == "puff":
        return HR.pillow(puff_outline(), 0.7, 16)
    raise ValueError(f"burst: no piece kind {kind!r} (have {', '.join(KINDS)})")
