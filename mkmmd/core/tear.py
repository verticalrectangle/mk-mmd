"""The tear, bpy-free and numpy-free (docs/design.md: Transitions and inserts): `[[transition]] kind = "tear"`. The
outgoing picture is torn in two by two hands along a jagged line: the rip runs along the line from one end over `run`
seconds, the halves bending apart behind it, then they swing apart about their far corners and leave the frame by the
cut, their torn edges fringed with white paper fibre. This module checks it and works out the line, the halves and where
they are on each window frame; mkmmd.tear draws them.

    center = [0.5, 0.5]     a point the line runs through, [x, y] frame fractions from the top left
    angle = 6               degrees the line is turned off vertical, clockwise positive, -45 .. 45
    from = "top"            the end the rip starts at: "top" or "bottom"
    jag = 0.02              how far the line wanders, a share of the frame's short side, 0 .. 0.1
    run = 0.13              seconds the rip takes to run the line's length (from the window's first frame); the rest
                            of the window the halves swing off. At least a frame each
    seed = 1                another tear
    back                    the paper's core, the white of the torn edges (default the palette's lightest)
    shadow = 0.4            how dark the shadows the halves cast, 0..1

Coordinates along the line: d is its direction from the start end, e = (-d_y, d_x) points to half "a" (the left half of a
tear run downward); a point p is at l = (p - centre) . d along it and at q = (p - centre) . e across it. The torn line is
q = jag(l) (`rip`); the print of half "a" is where q > jag(l), of half "b" where q < jag(l): together they cover the
frame once. Under the print the paper tore along another line: one half carries a strip of bare paper past the print
(`fringe`: its width and which half carries it wander along the line), hidden under the other half's print until they
part. Behind the rip's front the halves bend apart (`open`, growing toward the start end); then each turns about its
pivot, just outside the frame's corner at the far end on its side, by up to the angle that takes it wholly out of the
frame, and tilts up toward the lens about it, the hands' end first (seen in perspective: it grows and darkens a little as
it turns from the lamp). `grips` says where two hands hold the halves on each window frame."""
import math
import random

from . import sheet3d as S3

KEYS = {"at", "kind", "dur", "center", "angle", "from", "jag", "run", "seed", "back", "shadow"}
DEFAULTS = {"center": [0.5, 0.5], "angle": 6.0, "from": "top", "jag": 0.02, "run": 0.13, "seed": 1, "shadow": 0.4}
DUR = 0.7
ENDS = ("top", "bottom")
STEP = 0.004                    # the rip's points are this far apart (short sides)
OCTAVES = ((0.07, 1.0), (0.022, 0.45), (0.008, 0.22))   # (spacing, weight): the line's wander, coarse to fine
FRINGE = 0.02                   # the widest strip of bare paper (short sides)
OPEN = 0.16                     # how far apart the halves have bent at the start end when the rip is through (short sides)
TILT = math.radians(35.0)       # how far the halves are tilted toward the lens (the hands' end up) when they have gone
THROW = 0.6                     # how far the halves' pivots have been thrown out of the frame then (short sides)
SWING = (20, 170)               # degrees: the least and the most a half turns by the window's end
BOW = (0.03, 0.3)               # the bow: how far the torn edge is drawn back as it lifts, and over how much of the half
CRUMBS = 26                     # paper crumbs thrown from the rip
GRAVITY = 2.5                   # the crumbs' fall, short sides a second squared
GRIP = (0.09, 0.11)             # where a hand holds a half: along the line from its start end, and in from it (short sides)
FIBRE_REACH = 0.02              # how far the paper's fibres may stand out of the torn edge (short sides)
PIVOT_OUT = 0.1                 # how far outside the frame's corner a half's pivot lies (short sides)


def normalize(spec, out, palette, what):
    """A tear's own keys with their defaults laid into `out` (the transition's common keys), colours resolved."""
    from . import transition as TR
    from .iris import _lightest
    out["center"] = TR._fractions(spec.get("center", DEFAULTS["center"]), f"{what}: center")
    end = spec.get("from", DEFAULTS["from"])
    if end not in ENDS:
        raise TR.TransitionError(f"{what}: from = {end!r}, expected one of {ENDS} (the end the rip starts at)")
    out["from"] = end
    for k, lo, hi in (("angle", -45.0, 45.0), ("jag", 0.0, 0.1), ("run", 0.0, 60.0), ("shadow", 0.0, 1.0)):
        v = spec.get(k, DEFAULTS[k])
        try:
            if isinstance(v, bool):
                raise TypeError
            out[k] = float(v)
        except (TypeError, ValueError):
            raise TR.TransitionError(f"{what}: {k} = {v!r}: expected a number") from None
        if not lo <= out[k] <= hi:
            raise TR.TransitionError(f"{what}: {k} = {out[k]}: expected {lo} .. {hi}")
    if out["run"] <= 0.0 or out["run"] >= out["dur"]:
        raise TR.TransitionError(f"{what}: run = {out['run']}: the rip takes part of the window, in (0, dur = "
                                 f"{out['dur']})")
    v = spec.get("seed", DEFAULTS["seed"])
    if isinstance(v, bool) or not isinstance(v, int):
        raise TR.TransitionError(f"{what}: seed = {v!r}: expected a whole number")
    out["seed"] = v
    out["back"] = TR._colour(spec.get("back", _lightest(palette)), palette, f"{what}: back")
    out["ink"] = TR._colour(S3.darkest(palette), palette, f"{what}: shadow")
    return out


def run_frames(tr, fps):
    """The window frames the rip takes (at least one, leaving at least one to swing off)."""
    return max(1, min(int(round(tr["run"] * fps)), tr["n"] - 1))


# ------------------------------------------------------------------------------------------------ the line
def axes(tr, size):
    """(origin px, d, e): the line's point and its direction from the start end and toward half "a"."""
    a = math.radians(tr["angle"])
    d = (-math.sin(a), math.cos(a))                   # down, turned clockwise
    if tr["from"] == "bottom":
        d = (-d[0], -d[1])
    w, h = size
    o = (tr["center"][0] * w - 0.5, tr["center"][1] * h - 0.5)
    return o, d, (-d[1], d[0])


def to_line(o, d, e, x, y):
    """(l, q): a point's place along the line and across it."""
    return (x - o[0]) * d[0] + (y - o[1]) * d[1], (x - o[0]) * e[0] + (y - o[1]) * e[1]


def from_line(o, d, e, l, q):
    return o[0] + l * d[0] + q * e[0], o[1] + l * d[1] + q * e[1]


def span(tr, size):
    """(l0, l1): the frame's extent along the line (px): the start end and the far end."""
    o, d, e = axes(tr, size)
    w, h = size
    ls = [to_line(o, d, e, x, y)[0] for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)]
    return min(ls), max(ls)


def _noise(rng, n, spacing, step):
    """n samples `step` apart of a seeded piecewise-linear wander whose knots lie `spacing` apart, in -1..1."""
    knots = [rng.uniform(-1.0, 1.0) for _ in range(int(n * step / spacing) + 3)]
    out = []
    for i in range(n):
        t = i * step / spacing
        k = int(t)
        out.append(knots[k] + (knots[k + 1] - knots[k]) * (t - k))
    return out


def rip(tr, size):
    """{"l": [...] px along the line, "q": the torn line's offset across it there (px), "fringe": the bare paper's width
    there (px, positive on half "a", negative on half "b")}, from a little before the start end to a little past the far
    end, STEP short sides apart."""
    key = (tr["angle"], tr["from"], tr["jag"], tr["seed"], tuple(tr["center"]), tuple(size))
    hit = _RIP.get(key)
    if hit is not None:
        return hit
    S = float(min(size))
    l0, l1 = span(tr, size)
    step = STEP * S
    n = int(math.ceil((l1 - l0 + 4.0 * step) / step)) + 1
    rng = random.Random(f"tear:{tr['seed']}")
    q = [0.0] * n
    total = sum(wt for _, wt in OCTAVES)
    for spacing, wt in OCTAVES:
        for i, v in enumerate(_noise(rng, n, spacing * S, step)):
            q[i] += v * wt / total * tr["jag"] * S
    side = _noise(rng, n, 0.05 * S, step)
    fine = _noise(rng, n, 0.006 * S, step)
    fringe = [FRINGE * S * (0.75 * s + 0.25 * f) for s, f in zip(side, fine)]
    _RIP[key] = {"l": [l0 - 2.0 * step + i * step for i in range(n)], "q": q, "fringe": fringe}
    return _RIP[key]


_RIP = {}


def sample(line, key, l):
    """The rip's `key` ("q" or "fringe") at l px along the line (linear between its points, held past its ends)."""
    ls = line["l"]
    t = (l - ls[0]) / (ls[1] - ls[0])
    if t <= 0.0:
        return line[key][0]
    if t >= len(ls) - 1:
        return line[key][-1]
    k = int(t)
    return line[key][k] + (line[key][k + 1] - line[key][k]) * (t - k)


def half_of(line, o, d, e, x, y):
    """"a" or "b": the half whose print holds the point (x, y) px."""
    l, q = to_line(o, d, e, x, y)
    return "a" if q > sample(line, "q", l) else "b"


# ------------------------------------------------------------------------------------------------ the motion
def pivots(tr, size):
    """{"a": (x, y), "b": (x, y)}: the point each half turns about: the frame's corner at the far end on its side, moved
    PIVOT_OUT short sides out of the frame (a half turning about a point of its own would never leave the frame)."""
    o, d, e = axes(tr, size)
    w, h = size
    out = {}
    for half, sign in (("a", 1.0), ("b", -1.0)):
        best = None
        for x in (-0.5, w - 0.5):
            for y in (-0.5, h - 0.5):
                l, q = to_line(o, d, e, x, y)
                key = (l + 0.25 * sign * q)
                if sign * q > 0 and (best is None or key > best[0]):
                    best = (key, (x, y))
        (x, y), out_by = best[1], PIVOT_OUT * min(size)
        out[half] = (x + (out_by if x > 0 else -out_by), y + (out_by if y > 0 else -out_by))
    return out


def opening(tr, size, l, front):
    """How far (px) each half has bent away from the line at l px along it with the rip's front at `front`."""
    l0, l1 = span(tr, size)
    if l >= front:
        return 0.0
    return 0.5 * OPEN * min(size) * ((front - l) / (l1 - l0)) ** 0.8


def bow(tr, k, size, l, front):
    """How far (px) a half's torn edge has been drawn back across the line at l px along it, at window frame `k`: the
    paper bows away from the tear, its torn edge lifting toward the lens, from where the rip has passed, more as the
    window goes on (mkmmd.tear bends each half by it over BOW[1] short sides from its edge)."""
    l0, l1 = span(tr, size)
    behind = min(max((front - l) / (0.2 * (l1 - l0)), 0.0), 1.0)
    return BOW[0] * min(size) * behind * (0.35 + 0.65 * ((k + 1) / tr["n"]) ** 0.8)


def crumbs(tr, fps, size):
    """The paper crumbs thrown from the rip: [{"l" (where along the line), "k" (the window frame the rip's front passes
    it), "v" (px a second: x, y, toward the lens), "r" (half its size, px), "spin" (radians a second), "a0"}], seeded."""
    S = float(min(size))
    rng = random.Random(f"tear-crumbs:{tr['seed']}")
    l0, l1 = span(tr, size)
    o, d, e = axes(tr, size)
    fronts = [state(tr, k, fps, size)["front"] for k in range(run_frames(tr, fps))]
    out = []
    for _ in range(CRUMBS):
        l = rng.uniform(l0 + 0.03 * (l1 - l0), l1 - 0.1 * (l1 - l0))
        k = next((i for i, f in enumerate(fronts) if f >= l), len(fronts) - 1)
        side = rng.choice((-1.0, 1.0))
        sp = S * rng.uniform(0.5, 1.6)
        out.append({"l": l, "k": k, "v": (side * e[0] * sp + rng.uniform(-0.3, 0.3) * S,
                                          side * e[1] * sp - rng.uniform(0.2, 0.9) * S, rng.uniform(0.3, 1.2) * S),
                    "r": S * rng.uniform(0.002, 0.006), "spin": rng.uniform(-25.0, 25.0), "a0": rng.uniform(0.0, 6.3)})
    return out


def crumb_at(tr, c, k, fps, size):
    """(x, y, z, angle) of crumb `c` at window frame `k` (px, radians), None before it is thrown."""
    if k < c["k"]:
        return None
    o, d, e = axes(tr, size)
    x0, y0 = from_line(o, d, e, c["l"], sample(rip(tr, size), "q", c["l"]))
    t = (k - c["k"]) / fps
    z = min(c["v"][2] * t, 0.5 * S3.distance(size))
    return (x0 + c["v"][0] * t, y0 + c["v"][1] * t + 0.5 * GRAVITY * min(size) * t * t, z, c["a0"] + c["spin"] * t)


def state(tr, k, fps, size):
    """Where the halves are at window frame `k`: {"front" (the rip's front, px along the line), "turn": {"a", "b"}
    (radians, clockwise positive on screen), "tilt" (radians each half is lifted toward the lens about the axis through
    its pivot square to its hinge: the hands' end comes up), "throw" (px each half's pivot has moved out of the frame,
    away from its middle: the hands throw the halves off), "lift" (px: how high the halves are for their shadows while
    the rip runs)}."""
    l0, l1 = span(tr, size)
    nr = run_frames(tr, fps)
    S = float(min(size))
    if k < nr:
        u = (k + 1) / nr
        front = l0 + (l1 - l0 + 2.0) * u ** 1.2
        return {"front": front, "turn": {"a": 0.0, "b": 0.0}, "tilt": 0.0, "throw": {"a": 0.0, "b": 0.0},
                "lift": 0.02 * S * u, "k": k}
    v = (k - nr + 1) / (tr["n"] - nr)
    ends = swing(tr, size)
    e = v ** 1.4
    return {"front": l1 + 2.0, "turn": {"a": -ends["a"] * e, "b": ends["b"] * e}, "tilt": TILT * e,
            "throw": {h: ends["throw"][h] * e ** 1.6 for h in "ab"}, "lift": 0.02 * S, "k": k}


def bow_shift(tr, size, st, l, r):
    """How far (px) the point r px into a half from the torn line is drawn back by the bow in state `st`: all of the
    bow at the edge and past it (the fringe), easing to none BOW[1] short sides in."""
    if "k" not in st:
        return 0.0
    W = BOW[1] * min(size)
    return bow(tr, st["k"], size, l, st["front"]) * (1.0 if r <= 0.0 else max(1.0 - r / W, 0.0) ** 2)


def hinge(tr, size, half):
    """(pivot, u, out): the point a half turns about, the unit vector from it toward where the hand holds the half and
    the unit vector the pivot is thrown along (from the frame's middle through it)."""
    S = float(min(size))
    o, d, e = axes(tr, size)
    line = rip(tr, size)
    l0, _ = span(tr, size)
    sign = 1.0 if half == "a" else -1.0
    l = l0 + GRIP[0] * S
    gx, gy = from_line(o, d, e, l, sample(line, "q", l) + sign * GRIP[1] * S)
    px, py = pivots(tr, size)[half]
    n = math.hypot(gx - px, gy - py)
    cx, cy = S3.centre(size)
    m = math.hypot(px - cx, py - cy)
    return (px, py), ((gx - px) / n, (gy - py) / n), ((px - cx) / m, (py - cy) / m)


def place(tr, size, half, x, y, st):
    """Where the point (x, y) px of the picture on `half` is seen in state `st` (px): bent away from the line behind
    the rip's front, turned about the pivot, tilted toward the lens about it, thrown with it and seen in perspective
    (mkmmd.core.sheet3d)."""
    o, d, e = axes(tr, size)
    l, q = to_line(o, d, e, x, y)
    sign = 1.0 if half == "a" else -1.0
    r = sign * (q - sample(rip(tr, size), "q", l))                  # how far into the half from the torn line
    q += sign * (opening(tr, size, l, st["front"]) + bow_shift(tr, size, st, l, r))
    x, y = from_line(o, d, e, l, q)
    (px, py), u0, out = hinge(tr, size, half)
    a = st["turn"][half]
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = x - px, y - py
    dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
    u = (u0[0] * ca - u0[1] * sa, u0[0] * sa + u0[1] * ca)
    v = (-u[1], u[0])
    along, across = dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]
    b = st["tilt"]
    X = px + out[0] * st["throw"][half] + along * math.cos(b) * u[0] + across * v[0]
    Y = py + out[1] * st["throw"][half] + along * math.cos(b) * u[1] + across * v[1]
    sx, sy, _ = S3.project(X, Y, along * math.sin(b), size)
    return sx, sy


def outline(tr, size, half):
    """The half's print as a polygon (px): the rip's points and the frame's corners on its side (a little past the
    frame's edges: what must leave the frame)."""
    o, d, e = axes(tr, size)
    line = rip(tr, size)
    w, h = size
    sign = 1.0 if half == "a" else -1.0
    pts = [from_line(o, d, e, l, q + sign * abs(f)) for l, q, f in zip(line["l"], line["q"], line["fringe"])]
    corners = [(x, y) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)
               if sign * to_line(o, d, e, x, y)[1] > sign * sample(line, "q", to_line(o, d, e, x, y)[0])]
    return pts + corners


def swing(tr, size):
    """{"a", "b", "throw": {"a", "b"}}: how far each half has turned (radians) and been thrown (px) by the window's end:
    SWING[1] degrees and THROW short sides scaled together by the least share (in steps of 2 %) that takes all of it out
    of the frame, tilted toward the lens as it is then; so it is still leaving on the window's last frame."""
    key = (tr["angle"], tr["from"], tr["jag"], tr["seed"], tuple(tr["center"]), tuple(size))
    hit = _SWING.get(key)
    if hit is not None:
        return hit
    w, h = size
    m = (FIBRE_REACH + BOW[0]) * min(size)          # the fibres stand out past the outline, and the bow draws it back
    out = {"throw": {}}
    for half, sign in (("a", -1.0), ("b", 1.0)):
        pts = outline(tr, size, half)
        out[half], out["throw"][half] = math.radians(SWING[1]), THROW * min(size)
        for step in range(1, 51):
            s = step / 50.0
            turn, throw = s * math.radians(SWING[1]), s * THROW * min(size)
            st = {"front": span(tr, size)[1] + 2.0, "turn": {half: sign * turn}, "tilt": TILT,
                  "throw": {half: throw}}
            seen = [place(tr, size, half, x, y, st) for x, y in pts]
            xs, ys = [p[0] for p in seen], [p[1] for p in seen]
            if max(xs) < -1.0 - m or min(xs) > w + m or max(ys) < -1.0 - m or min(ys) > h + m:
                out[half], out["throw"][half] = turn, throw
                break
    _SWING[key] = out
    return out


_SWING = {}


def grips(tr, frame, fps, size):
    """Where two hands hold the halves on Blender `frame` of the tear's window: {"a": [x, y], "b": [x, y]} frame fractions
    from the top left (outside 0..1 once a half has left the frame), each a little in from the torn line near its start
    end. Before the window they are where the hands take hold, after it the last frame's places."""
    w, h = size
    S = float(min(size))
    o, d, e = axes(tr, size)
    line = rip(tr, size)
    l0, _ = span(tr, size)
    k = int(frame) - tr["first"]
    st = None if k < 0 else state(tr, min(k, tr["n"] - 1), fps, size)
    out = {}
    for half, sign in (("a", 1.0), ("b", -1.0)):
        l = l0 + GRIP[0] * S
        x, y = from_line(o, d, e, l, sample(line, "q", l) + sign * GRIP[1] * S)
        if st is not None:
            x, y = place(tr, size, half, x, y, st)
        out[half] = [(x + 0.5) / w, (y + 0.5) / h]
    return out
