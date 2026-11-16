"""A burst, bpy-free (the library prop `burst`, docs/design.md: the library table): pieces (puffy hearts, stars, smoke
puffs, slivers of glass, broken bits of pottery, drops of a liquid) flung out from one point at a clip second, each
easing out to its own reach while it spins and pops in, then shrinking away. Driver expressions on the frame do it
(mkmmd/blender/library/props/burst.py lays them on the pieces), so a freeze (mkmmd.core.freeze), which holds the scene
on a frame, holds them mid-air.

The pieces fly mostly across the picture (the directions are flattened toward the frame's plane, the prop's x-z: it
faces -Y like a character), each with its own reach, size, a little delay and a spin; all from the `seed`, the same on
every machine. Or they fly out in a cone about an `aim` (a cup smashing on the floor: up and out all round),
tumbling end over end (`tumble`) as they go, falling as they fly (the live `gravity`) down to a `ground` they stay on,
and staying where they come to rest (`fade` 1). A heart, a star or a puff is a pillow (mkmmd.core.hearts.pillow) of its
outline (the heart's, a five-point star's with rounded tips, a cloud's); a shard is a thin flat sliver with sharp edges,
mirror-bright; a sherd a thick curved bit of a pot's wall, glazed; a drop a rounded drop of a liquid."""
import hashlib
import math

import numpy as np

from . import hearts as HR
from . import shell as CS

KINDS = ("heart", "star", "puff", "shard", "sherd", "drop")
PARAMS = {  # root property: (default, unit, doc); keyed with [[key]]
    "start": (0.0, "s", "the clip second it goes off"),
    "reach": (1.2, "m", "how far the farthest piece flies"),
    "life": (1.4, "s", "how long a piece lives, out and away"),
    "spin": (360.0, "deg", "how far a piece turns over its life"),
    "size": (0.16, "m", "the largest piece's width"),
    "amount": (1.0, "", "overall scale of every piece (0 hides them)"),
    "gravity": (0.0, "m/s2", "how fast a piece falls (down the prop's -Z) as it flies"),
    "ground": (-100.0, "m", "the height in the prop's frame no piece goes under: one coming down to it stays on it"),
}
MIX = {"heart": 0.45, "star": 0.35, "puff": 0.2}  # default share of each kind
COLOURS = {"heart": "love", "star": "gold", "puff": "text", "shard": "text", "sherd": "rose", "drop": "gold"}
POP = 0.08                       # share of its life a piece takes to pop in
FADE = 0.7                       # share of its life after which it shrinks away
FLAT = 0.45                      # how much of a direction toward or away from the camera is kept
TUMBLE = 20.0                    # degrees a piece tumbles out of the picture's plane at most: it keeps showing its face
CONE = 60.0                      # degrees: the half-angle of the cone the pieces fly in about an `aim`
MIN_SCALE = HR.MIN_SCALE
HELPERS = ("u",)                 # custom property of each piece, driven: its age as a share of its life
FUNCS = HR.FUNCS


def _unit(seed, *key):
    """A deterministic number in [0, 1) from a seed and a key (the same on every platform)."""
    h = hashlib.blake2b(("/".join(str(k) for k in (seed,) + key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


def options(slots):
    """(count, seed, mix {kind: share}, params {root property: value}, colours {kind: spec}, glow, form) from the
    `slots` of the [[prop]], checked: a bad value is a ValueError that says what to change. `form` says how the pieces
    fly: {"aim": the axis (3,) of the cone they fly in, unit, in the prop's frame, or None (all round, across the
    picture), "cone": its half-angle (degrees), "tumble": the degrees a piece turns end over end over its life, or None
    (a wobble that keeps its face to the lens), "fade": the share of its life after which a piece shrinks away (1: it
    stays)}."""
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
        positive = k in ("reach", "life", "size")
        if not isinstance(v, (int, float)) or (k not in ("start", "ground") and v < 0) or (positive and v <= 0):
            raise ValueError(f"burst: {k} must be a {'positive ' if positive else ''}number"
                             f"{' (' + unit + ')' if unit else ''} (got {v!r})")
        P[k] = float(v)
    colours = {k: slots.get(k, COLOURS[k]) for k in KINDS}
    glow = slots.get("glow", 0.3)
    if not isinstance(glow, (int, float)) or glow < 0:
        raise ValueError(f"burst: glow is a number >= 0 (got {glow!r})")
    aim = slots.get("aim")
    if aim is not None:
        if not (isinstance(aim, (list, tuple)) and len(aim) == 3 and all(isinstance(c, (int, float)) for c in aim)
                and math.hypot(*aim) > 1e-9):
            raise ValueError(f"burst: aim is the way the pieces fly, [x, y, z] in the prop's frame (got {aim!r})")
        aim = [float(c) / math.hypot(*aim) for c in aim]
    cone = slots.get("cone", CONE)
    if not isinstance(cone, (int, float)) or not 0 < cone <= 180:
        raise ValueError(f"burst: cone is the half-angle of the cone about `aim`, degrees in (0, 180] (got {cone!r})")
    tumble = slots.get("tumble")
    if tumble is not None and (not isinstance(tumble, (int, float)) or tumble < 0):
        raise ValueError(f"burst: tumble is the degrees a piece turns end over end over its life, >= 0 "
                         f"(got {tumble!r})")
    fade = slots.get("fade", FADE)
    if not isinstance(fade, (int, float)) or not 0 < fade <= 1:
        raise ValueError(f"burst: fade is the share of its life after which a piece shrinks away, in (0, 1]; 1 keeps "
                         f"it (got {fade!r})")
    form = {"aim": aim, "cone": float(cone), "tumble": None if tumble is None else float(tumble), "fade": float(fade)}
    return count, seed, {k: float(v) for k, v in mix.items()}, P, colours, float(glow), form


def _onto(v, axis):
    """`v` turned by the rotation that takes +Z onto the unit `axis` (about any perpendicular: the cap is round)."""
    a = np.asarray(axis, float)
    e1 = np.cross([0.0, 1.0, 0.0] if abs(a[1]) < 0.9 else [1.0, 0.0, 0.0], a)
    e1 /= np.linalg.norm(e1)
    return v[0] * e1 + v[1] * np.cross(a, e1) + v[2] * a


def layout(count, seed, mix, aim=None, cone=CONE):
    """[{kind, dir (3,), reach (share), size (share), delay (share of the life), spin (share, +-), tumble (+-1)}] of the
    pieces: the kinds in proportion to `mix` (largest remainder), spread round the whole burst (flattened toward the
    picture) or, with an `aim`, evenly over the cone of half-angle `cone` about it, every number from the seed and
    rounded as the expressions print it (so `pose` is what the drivers do)."""
    total = sum(mix.values())
    want = {k: count * v / total for k, v in mix.items()}
    n = {k: int(math.floor(w)) for k, w in want.items()}
    for k in sorted(want, key=lambda k: (n[k] - want[k], KINDS.index(k)))[:count - sum(n.values())]:
        n[k] += 1
    kinds = [k for k in KINDS for _ in range(n.get(k, 0))]
    order = sorted(range(count), key=lambda i: _unit(seed, "order", i))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    low = 1.0 - math.cos(math.radians(cone))
    out, R = [], lambda x: round(float(x), HR.NDIGITS)                 # noqa: E731
    for slot, i in enumerate(order):
        a = golden * slot + 2.0 * math.pi * _unit(seed, "turn")
        if aim is None:
            z = 1.0 - 2.0 * (slot + 0.5) / count                     # a Fibonacci sphere: no clumps, no gaps
            r = math.sqrt(max(0.0, 1.0 - z * z))
            d = np.array([r * math.cos(a), FLAT * r * math.sin(a), z])  # toward or away from the camera, less far
        else:
            z = 1.0 - low * (slot + 0.5) / count                     # the same over a cap of the sphere about the aim
            r = math.sqrt(max(0.0, 1.0 - z * z))
            d = _onto(np.array([r * math.cos(a), r * math.sin(a), z]), aim)
        out.append({"kind": kinds[i], "dir": np.round(d, HR.NDIGITS),
                    "reach": R(0.55 + 0.45 * _unit(seed, "reach", i)), "size": R(0.5 + 0.5 * _unit(seed, "size", i)),
                    "delay": R(0.06 * _unit(seed, "delay", i)),
                    "spin": R((1 if _unit(seed, "sy", i) < 0.5 else -1) * (0.4 + 0.6 * _unit(seed, "ay", i))),
                    "tumble": 1 if _unit(seed, "tx", i) < 0.5 else -1})
    return out


def pose(t, P, piece, tumble=None, fade=FADE):
    """Where piece `piece` is at clip second(s) t: (position (..., 3) in the prop's frame, rx (the tumble), ry (the spin
    in the picture's plane) radians, scale), as the drivers have it. It eases out to its reach, falls by gravity times
    the square of its age over two as it goes, never under `ground`; `tumble` (degrees, `options`' form) turns it end
    over end and slows its spin with its flight, and it shrinks away from `fade` of its life (1: never)."""
    t = np.asarray(t, float)
    u = np.clip((t - P["start"] - piece["delay"] * P["life"]) / P["life"], 0.0, 1.0)
    out = 1.0 - (1.0 - u) ** 3
    pos = np.multiply.outer(out * P["reach"] * piece["reach"], piece["dir"])
    age = u * P["life"]
    pos[..., 2] = np.maximum(P["ground"], pos[..., 2] - 0.5 * P["gravity"] * age * age)
    shrink = HR._smoothstep(fade, 1.0, u) if fade < 1.0 else 0.0
    s = np.maximum(MIN_SCALE, P["size"] * piece["size"] * P["amount"] * np.minimum(u / POP, 1.0) * (1.0 - shrink))
    spin = math.radians(P["spin"]) * piece["spin"]
    if tumble is None:
        return pos, math.radians(TUMBLE) * np.sin(u * 9.0) * piece["tumble"], spin * u, s
    return pos, math.radians(tumble) * out * piece["tumble"], spin * out, s


def expressions(piece, tumble=None, fade=FADE):
    """The driver expressions of a piece: {channel: text} for u (its helper, first), x, y, z, rx (a tumble of +-TUMBLE
    degrees, so it always shows its face, or end over end by `tumble` degrees), ry (its spin in the picture's plane), s.
    They read the root's properties by their names in PARAMS, `f0` (the frame of clip second 0), `fps`, `frame` and the
    piece's `u`."""
    n = HR._n
    d = piece["dir"]
    ease = "(1 - (1 - u) * (1 - u) * (1 - u))"
    reach = f"reach * {n(piece['reach'])} * {ease}"
    out = {"u": f"clamp(((frame - f0) / fps - start - {n(piece['delay'])} * life) / life, 0, 1)",
           "x": f"{n(d[0])} * {reach}", "y": f"{n(d[1])} * {reach}",
           "z": f"max(ground, {n(d[2])} * {reach} - 0.5 * gravity * u * life * u * life)"}
    if tumble is None:
        out["rx"] = f"radians({n(TUMBLE)}) * sin(u * 9) * {n(piece['tumble'])}"
        out["ry"] = f"radians(spin) * u * {n(piece['spin'])}"
    else:
        out["rx"] = f"radians({n(tumble)}) * {ease} * {n(piece['tumble'])}"
        out["ry"] = f"radians(spin) * {ease} * {n(piece['spin'])}"
    shrink = f" * (1 - smoothstep({n(fade)}, 1, u))" if fade < 1.0 else ""
    out["s"] = f"max({n(MIN_SCALE)}, size * {n(piece['size'])} * amount * min(u / {n(POP)}, 1){shrink})"
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


def shard_outline():
    """A sliver of broken glass: a long, thin, uneven triangle, (3, 2) points (x, z), unit width, counter-clockwise."""
    return np.array([[-0.5, -0.11], [0.5, 0.02], [-0.36, 0.17]])


def sherd_outline():
    """A bit broken off a pot's wall: an uneven five-sided piece, (5, 2) points (x, z), unit width, anticlockwise."""
    return np.array([[-0.5, -0.17], [0.1, -0.33], [0.5, -0.07], [0.26, 0.29], [-0.31, 0.23]])


SHERD = (0.12, 0.9)              # a sherd's thickness and the radius of the wall it was part of, shares of its width
DROP = [(0.0, -0.62), (0.22, -0.57), (0.38, -0.45), (0.47, -0.27), (0.5, -0.05), (0.46, 0.17), (0.36, 0.36),
        (0.2, 0.52), (0.0, 0.6)]  # a drop's profile (r, z): a blob a little longer than it is wide


def mesh(kind):
    """The unit piece of a kind (`core.shell.Mesh`, unit width, facing -Y): a heart, a star, a cloud puff, a shard, a
    sherd (its glazed outside to -Y, curved round the pot it broke off), a drop."""
    if kind == "heart":
        return HR.heart_mesh()
    if kind == "star":
        return HR.pillow(star_outline(), 0.38, 16)
    if kind == "puff":
        return HR.pillow(puff_outline(), 0.7, 16)
    if kind == "shard":
        return CS.rounded_panel(shard_outline(), 0.02, r_edge=0.003, n=1).rot(90.0, 0.0, 0.0)
    if kind == "sherd":
        thick, wall = SHERD
        m = CS.rounded_panel(sherd_outline(), thick, r_edge=0.25 * thick, n=2).rot(90.0, 0.0, 0.0)
        x, y = m.V[:, 0].copy(), m.V[:, 1].copy()                 # its back at y = 0, its face at y = -thick
        rho, ang = wall - y, x / wall                              # round the pot's axis, `wall` behind the back
        m.V[:, 0], m.V[:, 1] = rho * np.sin(ang), wall - rho * np.cos(ang)
        return m.apply(np.eye(3) / float(np.ptp(m.V[:, 0])))       # unit width again
    if kind == "drop":
        return CS.lathe(DROP, 24)
    raise ValueError(f"burst: no piece kind {kind!r} (have {', '.join(KINDS)})")
