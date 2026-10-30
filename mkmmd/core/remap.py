"""Remapping frame effects, bpy-free (docs/design.md: Frame effects, Remaps): the `[[fx]]` kinds that move the finished
picture's pixels about rather than paint over it, checked and timed here with the maths of where each pixel of the frame
comes from; mkmmd/remap.py does the picture work.

    kaleido  the frame folded into `segments` wedges about `center`, alternately mirrored as a two-mirror kaleidoscope
             (or plain rotational copies), each showing one source wedge of the picture: it turns (`turn`), steps on
             hits (`step`), spins as a whole (`spin`), zooms (`zoom` keys, a `pulse` on hits), and with a source
             `radius` folds back on itself in rings
    tunnel   an infinite Droste tunnel: the frame holds itself shrunk `scale` times about `center`, each level turned
             `twist` degrees, `depth` levels deep, a `frame` line round each; it falls inward `speed` levels a second
             and `step` levels on each hit
    melt     the picture breathes and melts (`mode`): a scale pulse (breathe), rings from the centre (ripple), a twist
             that grows to the centre (swirl), the picture running down like wax (drip), heat haze to a liquid wobble
             (noise); `amount` swells on hits and dies away over `decay`, or follows the strength keys

Lengths are shares of the frame's short side (one look in every output), angles degrees counter-clockwise on screen,
times clip seconds; a pixel's position is (x right, y down) px with pixel centres on whole numbers. With a `mask` an
effect remaps the ground with the figure kept on top as rendered, or the figure alone over the ground; either way the
figure's place in what it remaps is filled from the ground round it, so no copy of her is folded, nested or melted
into the ground. The maths here is numpy only; the coordinates it gives are where each pixel samples the picture.
"""
import math
from functools import lru_cache

import numpy as np

KINDS = ("kaleido", "tunnel", "melt")
MODES = ("breathe", "ripple", "swirl", "drip", "noise")
ATTACK = 0.04                   # seconds: a hit's swell (a melt's amount, a kaleidoscope's zoom pulse) rises this fast
_MASKING = {"mask", "objects", "edge", "shadow", "back"}
KEYS = {"kaleido": _MASKING | {"segments", "mirror", "center", "source", "turn", "spin", "step", "glide", "zoom",
                               "pulse", "decay"},
        "tunnel": _MASKING | {"scale", "twist", "depth", "center", "speed", "step", "glide", "frame"},
        "melt": _MASKING | {"mode", "amount", "scale", "speed", "center", "decay", "seed"}}
_MASK_DEFAULTS = {"mask": None, "edge": None, "shadow": None, "back": None, "min_db": None, "every": 1}
DEFAULTS = {"kaleido": {**_MASK_DEFAULTS, "segments": 6, "mirror": True, "center": [0.5, 0.5], "source": None,
                        "turn": 8.0, "spin": 0.0, "step": 30.0, "glide": 0.3, "zoom": 1.0, "pulse": 0.0, "decay": 0.25},
            "tunnel": {**_MASK_DEFAULTS, "scale": 0.5, "twist": 0.0, "depth": 10, "center": [0.5, 0.5], "speed": 0.5,
                       "step": 1.0, "glide": 0.35, "frame": None},
            "melt": {**_MASK_DEFAULTS, "mode": None, "amount": None, "scale": None, "speed": None,
                     "center": [0.5, 0.5], "decay": 0.4, "seed": 1}}
SOURCE = {"center": None, "angle": 90.0, "radius": None}  # a kaleidoscope's source wedge: its apex (default the
#                                                     centre), the way its middle points (90: up), where it folds back
FRAME_WIDTH = 6.0               # px at 1080: a tunnel's frame line when its `frame` gives no width
SPREAD = 0.3                    # the spread of a smooth noise field (B-splines of values in -1 .. 1): divided out
FRONT = 1.2                     # short sides a second: how fast a ripple's swell on a hit runs out from the centre
# a melt's own defaults by mode: `amount` (breathe: how much larger the middle grows, a share; ripple, drip, noise:
# short sides; swirl: degrees at the centre), `scale` (short sides: the bulge's radius, the rings' wavelength, the
# twist's radius, a drip's width, a wobble's size) and `speed` (breaths, rings, twists a second; drip: how fast its
# drips run down; noise: how many times a second the wobble changes)
MELT = {"breathe": {"amount": 0.08, "scale": 0.7, "speed": 1.0 / 3.0},
        "ripple": {"amount": 0.015, "scale": 0.12, "speed": 0.8},
        "swirl": {"amount": 90.0, "scale": 0.45, "speed": 0.0},
        "drip": {"amount": 0.25, "scale": 0.1, "speed": 0.5},
        "noise": {"amount": 0.012, "scale": 0.1, "speed": 1.0}}
MELT_RANGE = {"breathe": (-0.5, 2.0), "ripple": (-0.25, 0.25), "swirl": (-1080.0, 1080.0), "drip": (0.0, 1.0),
              "noise": (0.0, 0.5)}


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """A remapping effect's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    _masking(FX, spec, out, palette, what)
    FX._hits(spec, out, what, required=False)
    out["center"] = FX._center(out["center"], f"{what} center")
    {"kaleido": _kaleido, "tunnel": _tunnel, "melt": _melt}[out["kind"]](FX, spec, out, palette, what)


def _masking(FX, spec, out, palette, what):
    """`mask` and `objects` (FX._mask), and what dresses the figure under a mask: an `edge` and a `shadow` as a
    collage's sticker has them, and a figure mask's `back` (a flat colour under the remapped figure)."""
    FX._mask(spec, out, what)
    given = [k for k in ("edge", "shadow", "back") if out[k] is not None]
    if given and out["mask"] is None:
        raise FX.FxError(f"{what}: {', '.join(given)} dress the figure a `mask` keeps or remaps: give the mask and its "
                         f"objects")
    if out["back"] is not None and out["mask"] != "figure":
        raise FX.FxError(f"{what}: back is the colour under the figure a figure mask remaps (mask = \"figure\")")
    e = out["edge"]
    if e is not None:
        if not isinstance(e, dict) or "color" not in e or set(e) - {"color", "width"}:
            raise FX.FxError(f"{what}: edge = {e!r}: expected {{color, width}} (width in px at 1080 on the short side)")
        out["edge"] = {"color": FX._colour(e["color"], palette, f"{what} edge"),
                       "width": FX._num(e.get("width", 3.0), f"{what} edge width", lo=0.0)}
    out["shadow"] = FX._shadow(out["shadow"], palette, what)
    out["back"] = None if out["back"] is None else FX._colour(out["back"], palette, f"{what} back")


def _positive_keys(FX, v, what):
    keys = FX._keys(v, what)
    if any(x <= 0 for _, x in keys):
        raise FX.FxError(f"{what}: {v!r}: expected positive numbers (times as large)")
    return keys


def _kaleido(FX, spec, out, palette, what):
    out["segments"] = FX._num(out["segments"], f"{what} segments", 2, 64, integer=True)
    if not isinstance(out["mirror"], bool):
        raise FX.FxError(f"{what}: mirror is true or false")
    if out["mirror"] and out["segments"] % 2:
        raise FX.FxError(f"{what}: segments = {out['segments']}: a mirrored kaleidoscope's wedges alternate, so it has "
                         f"an even number of them (or mirror = false: plain rotational copies)")
    src = out["source"] if out["source"] is not None else {}
    if not isinstance(src, dict) or set(src) - set(SOURCE):
        raise FX.FxError(f"{what}: source = {out['source']!r}: expected {{center, angle, radius}} (center [x, y] frame "
                         f"fractions, angle degrees, radius short sides)")
    src = {**SOURCE, **src}
    out["source"] = {"center": None if src["center"] is None else FX._pair(src["center"], f"{what} source center"),
                     "angle": FX._num(src["angle"], f"{what} source angle"),
                     "radius": None if src["radius"] is None else FX._num(src["radius"], f"{what} source radius",
                                                                          lo=1e-3)}
    for k in ("turn", "spin", "step"):
        out[k] = FX._num(out[k], f"{what} {k}")
    out["glide"] = FX._num(out["glide"], f"{what} glide", lo=0.0)
    out["zoom"] = _positive_keys(FX, out["zoom"], f"{what} zoom")
    out["pulse"] = FX._num(out["pulse"], f"{what} pulse", -0.9, 4.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", lo=1e-3)


def _tunnel(FX, spec, out, palette, what):
    out["scale"] = FX._num(out["scale"], f"{what} scale", 0.05, 0.95)
    out["twist"] = FX._num(out["twist"], f"{what} twist", -180.0, 180.0)
    out["depth"] = FX._num(out["depth"], f"{what} depth", 1, 40, integer=True)
    out["speed"] = FX._num(out["speed"], f"{what} speed", -30.0, 30.0)
    out["step"] = FX._num(out["step"], f"{what} step", -8.0, 8.0)
    out["glide"] = FX._num(out["glide"], f"{what} glide", lo=0.0)
    f = out["frame"]
    if f is not None:
        if not isinstance(f, dict) or "color" not in f or set(f) - {"color", "width"}:
            raise FX.FxError(f"{what}: frame = {f!r}: expected {{color, width}} (a colour or a list of them, level by "
                             f"level; width px at 1080)")
        out["frame"] = {"color": FX._colours(f["color"], palette, f"{what} frame color"),
                        "width": FX._num(f.get("width", FRAME_WIDTH), f"{what} frame width", lo=0.0)}


def _melt(FX, spec, out, palette, what):
    mode = out["mode"]
    if mode not in MODES:
        raise FX.FxError(f"{what}: mode {mode!r} is none of {', '.join(MODES)} (`mode` is required)")
    for k, v in MELT[mode].items():
        if out[k] is None:
            out[k] = v
    lo, hi = MELT_RANGE[mode]
    out["amount"] = FX._num(out["amount"], f"{what} amount", lo, hi)
    out["scale"] = FX._num(out["scale"], f"{what} scale", 1e-3, 8.0)
    out["speed"] = FX._num(out["speed"], f"{what} speed", -30.0, 30.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", lo=1e-3)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)


# ================================================================================================= timing
def keyed(keys, t):
    """[[t, v], ...] at clip second `t`: linear between keys, held outside them."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def steps(times, t, glide):
    """How many hits (`times`, clip seconds) have come by clip second `t`, each counted in over `glide` seconds with an
    ease-out (it starts at once and settles): a whole number between hits, gliding from one to the next."""
    n = 0.0
    for h in times:
        a = t - h
        if a < 0:
            continue
        u = 1.0 if glide <= 0 else min(a / glide, 1.0)
        n += 1.0 - (1.0 - u) ** 2
    return n


def swell(t, times, attack, decay):
    """0..1 on hits at clip second `t`: each rises over `attack` (ease-out) and dies away as exp(-age / decay); the
    strongest counts (mkmmd.core.framefx.punch)."""
    from . import framefx as FX
    return float(FX.punch(t, times, attack, decay))


def kaleido_turn(fx, t, times):
    """Degrees the source wedge has turned by clip second `t`: `turn` a second since the span's start and `step` on
    each hit (`times`), eased over `glide`."""
    return fx["turn"] * (t - fx["from"]) + fx["step"] * steps(times, t, fx["glide"])


def kaleido_spin(fx, t):
    """Degrees the whole kaleidoscope has turned on screen by clip second `t`."""
    return fx["spin"] * (t - fx["from"])


def kaleido_zoom(fx, t, times):
    """The kaleidoscope's zoom at clip second `t`: its `zoom` keys, `pulse` larger on each hit and dying away."""
    z = keyed(fx["zoom"], t)
    if fx["pulse"] and times:
        z *= 1.0 + fx["pulse"] * swell(t, times, ATTACK, fx["decay"])
    return z


def tunnel_phase(fx, t, times):
    """How many levels the tunnel has fallen through by clip second `t`: `speed` a second since the span's start and
    `step` on each hit (`times`), eased over `glide`. Its whole part changes nothing (the tunnel is the same a level
    down) but the frame lines' colours, which travel with their levels."""
    return fx["speed"] * (t - fx["from"]) + fx["step"] * steps(times, t, fx["glide"])


def melt_level(fx, t, times, strength):
    """A melt's amount at clip second `t` (its mode's units): `amount` times `strength` (callable(t)) and, with hits
    (`times`), their swell: on each it jumps up and dies away over `decay`. Hits picked from a lane that fall outside
    the span leave it still."""
    a = fx["amount"] * strength(t)
    if fx.get("times") == [] or a == 0:
        return a
    return a * swell(t, times, ATTACK, fx["decay"])


def fires(fx):
    """Whether an effect fires on hits (`times` or a `lane`) rather than holding steady."""
    return fx.get("times") != []


# ================================================================================================= the kaleidoscope
@lru_cache(maxsize=8)
def grid(size):
    """(xs, ys) float32 (h, w): each pixel's position (pixel centres on whole numbers); kept for a few sizes (each
    output's, and the half sizes a kaleidoscope's map is worked out at)."""
    w, h = size
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def fold(rel, n, mirror):
    """Angles `rel` (radians, from the first mirror line) folded into the first wedge, 0 .. 2 pi / n: with `mirror`
    alternate wedges are its reflections (a two-mirror kaleidoscope: the fold is continuous), else its turned copies."""
    a = 2.0 * np.pi / n
    if not mirror:
        return np.mod(rel, a)
    u = np.mod(rel, 2.0 * a)
    return np.where(u > a, 2.0 * a - u, u)


def kaleido_coords(xs, ys, centre, n, mirror, axis, turn, spin, source, zoom, radius=None):
    """Where each pixel (xs, ys) of a kaleidoscope samples the picture: (sx, sy) float32. The frame is cut into `n`
    wedges about `centre` (px), the first one's middle pointing `axis` + `spin` degrees (counter-clockwise from the
    right); every wedge shows (mirrored in turn with `mirror`) the picture's wedge about `source` (px) whose middle
    points `axis` + `turn` degrees, `zoom` times as large. With `radius` (px) the source wedge ends there and folds
    back, ring after ring. At turn = spin = 0, zoom 1 and source = centre the first wedge is the picture as it is."""
    a = 2.0 * np.pi / n
    dx = xs - np.float32(centre[0])
    dy = np.float32(centre[1]) - ys                                                      # y up
    r = np.sqrt(dx * dx + dy * dy)
    rel = np.arctan2(dy, dx) - np.float32(math.radians(axis + spin) - a / 2.0)
    psi = fold(rel, n, mirror) + np.float32(math.radians(axis + turn) - a / 2.0)
    rho = r * np.float32(1.0 / zoom)
    if radius is not None:
        R = np.float32(radius)
        rho = R - np.abs(np.mod(rho, 2.0 * R) - R)
    return ((np.float32(source[0]) + rho * np.cos(psi)).astype(np.float32),
            (np.float32(source[1]) - rho * np.sin(psi)).astype(np.float32))


def kaleido_seams(xs, ys, centre, n, axis, spin):
    """Plain rotational copies (no mirror) meet at a jump: (d, side) per pixel, d its distance in px from the nearest
    wedge edge and side +1 when that edge is the wedge's start (the copy before lies a wedge on), -1 at its end."""
    a = 2.0 * np.pi / n
    dx = xs - np.float32(centre[0])
    dy = np.float32(centre[1]) - ys
    r = np.sqrt(dx * dx + dy * dy)
    u = np.mod(np.arctan2(dy, dx) - np.float32(math.radians(axis + spin) - a / 2.0), a)
    near_start = u < a / 2.0
    d = r * np.sin(np.minimum(np.where(near_start, u, a - u), np.float32(np.pi / 2.0)))
    return d.astype(np.float32), np.where(near_start, 1.0, -1.0).astype(np.float32)


# ================================================================================================= the tunnel
def level_box(size, centre, m, theta):
    """The screen box (x0, y0, x1, y1) whole pixels, clipped to the frame of `size`, of the frame drawn `m` times as
    large and turned `theta` radians counter-clockwise about `centre` (px); None when it misses the frame."""
    w, h = size
    cx, cy = centre
    corners = [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]
    c, s = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for qx, qy in corners:
        dx, dy = (qx - cx) * m, (qy - cy) * m
        xs.append(cx + c * dx + s * dy)                     # counter-clockwise on screen with y down
        ys.append(cy - s * dx + c * dy)
    x0, y0 = max(int(math.floor(min(xs))), 0), max(int(math.floor(min(ys))), 0)
    x1, y1 = min(int(math.ceil(max(xs))) + 1, w), min(int(math.ceil(max(ys))) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def level_covers(size, centre, m, theta):
    """Whether the frame drawn `m` times as large and turned `theta` about `centre` covers every pixel of the frame."""
    w, h = size
    cx, cy = centre
    c, s = math.cos(theta), math.sin(theta)
    for px, py in ((0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)):
        dx, dy = px - cx, py - cy
        qx, qy = cx + (c * dx - s * dy) / m, cy + (s * dx + c * dy) / m       # the screen corner in the level's frame
        if not (-0.5 <= qx <= w - 0.5 and -0.5 <= qy <= h - 0.5):
            return False
    return True


def tunnel_levels(size, centre, scale, twist, phase, depth):
    """The levels a Droste tunnel draws on a frame of `size` (outermost first): [(k, m, theta, alpha)], level k the
    frame `m` = scale ** (k - f) times as large and turned theta = (k - f) x `twist` (radians) about `centre`, laid at
    opacity alpha; f is the fractional part of `phase` (the levels fallen through). The first is the outermost that
    covers the whole frame (level 0, or one round it when the twist turns level 0's corners into the frame); then the
    levels in to `depth`, and level depth + 1 faded in by f, so that as one level falls past the frame's edge the
    next appears at the far end without a jump; none less than a pixel across."""
    f = phase - math.floor(phase)
    tw = math.radians(twist)
    k0 = 0
    while not level_covers(size, centre, scale ** (k0 - f), (k0 - f) * tw) and k0 > -8:
        k0 -= 1
    out = []
    for k in range(k0, depth + 2):
        m = scale ** (k - f)
        alpha = 1.0 if k <= depth else f
        if m * min(size) < 1.0 or alpha <= 0.0:
            break
        out.append((k, m, (k - f) * tw, alpha))
    return out


def tunnel_queue(size, centre, scale, twist, phase, depth):
    """The copies of a figure's tunnel (a figure mask), far to near: [(k, m, theta, alpha)] as tunnel_levels gives
    them but for level 0 and in, each a whole copy of the cut-out laid over the farther ones (not clipped to its
    level's frame): level 0 is the figure as rendered when the phase is whole and fades out as it grows past the
    viewer (1 - f), the deepest fades in (f), so the queue of copies runs out of the vanishing point forever."""
    f = phase - math.floor(phase)
    out = [(k, m, theta, alpha * (1.0 - f) if k == 0 else alpha)
           for k, m, theta, alpha in tunnel_levels(size, centre, scale, twist, phase, depth) if k >= 0]
    return [lv for lv in reversed(out) if lv[3] > 0.0]


def level_map(centre, m, theta):
    """The 2 x 3 affine map (screen px -> the frame's px) of a level drawn `m` times as large and turned `theta`
    radians counter-clockwise on screen about `centre`."""
    cx, cy = centre
    c, s = math.cos(theta) / m, math.sin(theta) / m
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]])


def tunnel_source(xs, ys, size, centre, scale, twist, phase, depth):
    """Where each pixel (xs, ys) of a Droste tunnel samples the frame and on which level: (sx, sy, k), the deepest of
    the levels tunnel_levels draws whose frame holds the pixel (the painter's order: an inner level over the outer; a
    level fading in counts once it is more than half there). This is the tunnel's picture without antialiasing, for
    checking the one mkmmd.remap draws."""
    w, h = size
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    sx, sy, lev = np.full(xs.shape, np.nan), np.full(xs.shape, np.nan), np.full(xs.shape, -99, int)
    for k, m, theta, alpha in tunnel_levels(size, centre, scale, twist, phase, depth):
        if alpha <= 0.5:
            continue
        A = level_map(centre, m, theta)
        qx, qy = A[0, 0] * xs + A[0, 1] * ys + A[0, 2], A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
        inside = (qx >= -0.5) & (qx <= w - 0.5) & (qy >= -0.5) & (qy <= h - 0.5)
        sx, sy, lev = np.where(inside, qx, sx), np.where(inside, qy, sy), np.where(inside, k, lev)
    return sx, sy, lev


def frame_colour(colours, k, phase):
    """The frame line colour of level k when the tunnel has fallen `phase` levels: by the level's own number since the
    span's start, so a line keeps its colour as it falls past the frame's edge."""
    return colours[int(k + math.floor(phase)) % len(colours)]


# ================================================================================================= melting
def _bspline(f):
    """Cubic B-spline weights of the four lattice points round a coordinate whose fraction past the second is `f`:
    positive and summing to 1, so a field built from them never leaves the lattice's range."""
    f = np.asarray(f, np.float64)
    g = 1.0 - f
    return np.stack([g ** 3 / 6.0, (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) / 6.0,
                     (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) / 6.0, f ** 3 / 6.0], -1)


def spread(n, cell, shift=0.0):
    """(n, m) weights carrying a lattice of points `cell` px apart (moved `shift` cells along) onto `n` pixels by cubic
    B-splines: a smooth field (twice differentiable) out of m lattice values."""
    u = (np.arange(n) + 0.5) / cell + shift
    i = np.floor(u).astype(np.int64)
    lo = int(i.min()) - 1
    m = int(i.max()) + 3 - lo
    W = np.zeros((n, m), np.float32)
    wts = _bspline(u - i)
    for j in range(4):
        W[np.arange(n), i - lo - 1 + j] = wts[:, j]
    return W


def lattice(seed, slot, shape):
    """The random lattice of time slot `slot` (values in -1 .. 1, seeded): the same slot gives the same values."""
    return np.random.default_rng([int(s) for s in seed] + [int(slot) + (1 << 20)]).uniform(-1.0, 1.0, shape)


def noise(seed, shape, tau):
    """A lattice of `shape` moving smoothly through time `tau` (slots): the cubic B-spline of the slots round it, so it
    changes without a jolt; its values stay in -1 .. 1."""
    i = int(math.floor(tau))
    wts = _bspline(tau - i)
    return sum(wts[j] * lattice(seed, i - 1 + j, shape) for j in range(4))


def saturate(x):
    """A field of about unit spread squeezed smoothly into -1 .. 1 (x / sqrt(1 + x^2)): bounded, with no corner."""
    return x / np.sqrt(1.0 + x * x)


def _radial(xs, ys, centre):
    dx, dy = xs - np.float32(centre[0]), ys - np.float32(centre[1])
    return dx, dy, np.sqrt(dx * dx + dy * dy)


def melt_field(mode, size, centre, level, scale, speed, age, unit, seed=(1, 0), front=None):
    """(dx, dy) float32 (h, w): how far (px) each pixel of a frame of `size` reaches for its picture in melt `mode`, at
    `age` seconds into the span: the pixel at p shows the picture at p + (dx, dy). `level(age)` is the amount then (the
    mode's own units: breathe a share, swirl degrees, the others short sides), `scale` short sides, `speed` the mode's
    own rate, `unit` px a short side, `centre` px. A ripple with a `front` (px a second) reads the level at the age its
    stretch of rings left the centre, so a swell on a hit runs out across the frame (none before the span). Zero
    everywhere at amount 0; never further than the amount (breathe: a share of a pixel's distance from the centre,
    under half the bulge's radius; swirl: the turn itself)."""
    w, h = size
    xs, ys = grid((w, h))
    zero = np.zeros((h, w), np.float32)
    if mode == "ripple":
        lam = scale * unit
        dx, dy, r = _radial(xs, ys, centre)
        if front:
            rs = np.linspace(0.0, float(r.max()) + 1.0, 256)
            amp = np.array([level(age - x / front) if age - x / front >= 0 else 0.0 for x in rs], np.float32)
            if not amp.any():
                return zero, zero.copy()
            a = np.interp(r.ravel(), rs, amp * np.float32(unit)).reshape(r.shape).astype(np.float32)
        else:
            a = np.float32(level(age) * unit)
            if a == 0:
                return zero, zero.copy()
        ring = np.sin(np.float32(2.0 * np.pi) * (r / np.float32(lam) - np.float32(speed * age)))
        k = a * ring * (1.0 - np.exp(-(r / np.float32(lam)) ** 2)) / np.maximum(r, np.float32(1e-3))
        return (-k * dx).astype(np.float32), (-k * dy).astype(np.float32)   # a crest pushes the picture out
    a = float(level(age))
    if a == 0.0:
        return zero, zero.copy()
    wave = math.sin(2.0 * math.pi * speed * age) if speed and mode in ("breathe", "swirl") else 1.0
    if mode == "breathe":
        dx, dy, r = _radial(xs, ys, centre)
        g = max(a * wave, -0.9)
        k = (g / (1.0 + g)) * np.exp(-(r / np.float32(scale * unit)) ** 2)
        return (-k * dx).astype(np.float32), (-k * dy).astype(np.float32)
    if mode == "swirl":
        dx, dy, r = _radial(xs, ys, centre)
        th = np.float32(math.radians(a * wave)) * np.exp(-(r / np.float32(scale * unit)) ** 2)
        c, s = np.cos(th), np.sin(th)
        # the picture turned th counter-clockwise on screen about the centre: the pixel shows what lies th back
        return (c * dx - s * dy - dx).astype(np.float32), (s * dx + c * dy - dy).astype(np.float32)
    if mode == "drip":
        return drip(size, a * unit, scale * unit, speed, age, seed)
    if mode == "noise":
        cell = max(scale * unit, 2.0)
        Wy, Wx = spread(h, cell), spread(w, cell)
        L = noise(seed, (Wy.shape[1], Wx.shape[1], 2), speed * age).astype(np.float32)
        fx, fy = (Wy @ L[..., 0] @ Wx.T) / np.float32(SPREAD), (Wy @ L[..., 1] @ Wx.T) / np.float32(SPREAD)
        k = np.float32(a * unit) / np.sqrt(1.0 + fx * fx + fy * fy)     # both ways together squeezed under 1
        return (fx * k).astype(np.float32), (fy * k).astype(np.float32)
    raise ValueError(f"melt mode {mode!r} is none of {', '.join(MODES)}")


def drip(size, reach, width, speed, age, seed=(1, 0)):
    """(dx, dy) float32 (h, w) of the picture running down like melting wax: it hangs from the top of the frame and
    sags more the lower it lies (as the depth to the power 1.5), most in drips about `width` px wide with rounded
    tongues (about a third of the frame's width drips, the rest sags a fifth as far), a gentle swell running down them
    `speed` times a second and the drips slowly growing and shrinking. A pixel shows the picture straight above it:
    dx = 0, 0 <= -dy <= reach; and while reach stays under 0.45 of the frame's height a row never shows the picture
    below what the row under it shows (the picture stretches, it does not fold)."""
    w, h = size
    ys = grid((w, h))[1]
    cell = max(width, 2.0)
    tau = 0.15 * speed * age                                       # the drips' lengths change slowly
    Wx = spread(w, cell)
    drips = saturate((Wx @ noise(list(seed) + [1], (Wx.shape[1],), tau)) / SPREAD).astype(np.float32)   # -1 .. 1
    u = np.clip((drips - np.float32(0.15)) / np.float32(0.55), 0.0, 1.0)
    col = 0.2 + 0.8 * u * u * (3.0 - 2.0 * u)                     # 0.2 .. 1 across the frame: where it drips
    depth = ys / np.float32(h)                                     # 0 at the top, 1 at the bottom
    run = 0.9 + 0.1 * np.sin(np.float32(2.0 * np.pi) * (depth - np.float32(speed * age)) + np.float32(3.0) * drips)
    sag = np.float32(reach) * depth * np.sqrt(depth) * col[None, :] * run
    return np.zeros_like(sag), (-sag).astype(np.float32)
