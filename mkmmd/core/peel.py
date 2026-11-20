"""The peel, bpy-free and numpy-free (docs/design.md: Transitions and inserts): `[[transition]] kind = "peel"`. The
outgoing picture is a sticker pulled off the incoming one by a hand: it lifts at a corner, rolls back over itself into a
tube, its paper back outward, and the tube rolls off the frame by the cut. This module checks it and works out where the
sheet is on each window frame; mkmmd.peel draws it.

    from = "bottom-right"   the corner the hand pinches: "top-left", "top-right", "bottom-left" or "bottom-right"
    angle = 0               degrees the fold line is turned clockwise off 45 degrees across that corner, -40 .. 40
    radius = 0.11           the roll's radius, a share of the frame's short side, 0.01 .. 0.3
    back                    the paper of its back (default the palette's lightest)
    shadow = 0.45           how dark the shadow it casts, 0..1 (toward the palette's darkest colour)
    edge                    {color, width, round}: a die-cut rim round the sticker, `width` px at 1080 on the short side
                            (default 16), corners rounded by `round` of the short side (default 0.03), `color` the
                            palette's lightest by default. No rim, square corners, without the key

The geometry, in px along the fold's normal n (from the stuck part toward the pinched corner; x' of a point is its place
along n): with the fold line at c and `end` px of the sheet lifted past it (its pinched corner the farthest), the point
u px past the line is rolled up a spiral, the same along the whole line: it leaves the frame at c heading along n and
turns up round a circle of radius r0 = min(R, end / CURL) (a corner just lifted is curled tight, CURL radians round,
so its back shows at once; from end = CURL R on, the roll's full radius R), over the top (its paper back up) and back
down toward the stuck part; from `phi` radians round (half a turn) it winds in on itself, its radius of curvature
shrinking PITCH r0 a turn down to CORE r0, so the rest of it lies rolled up inside as a tube. The radius only shrinks:
every turn lies inside the one before it, and the whole of it above the frame. Every lifted point is seen in the
camera's perspective (mkmmd.core.sheet3d).

The timing is a hand's (mkmmd.core.sheet3d.hand): a slow pinch that lifts the corner, then a pull that speeds up into a
flick. The fold travels from the pinched corner (nothing lifted) to where the whole sheet, rolled up, has left the frame
past its far side; `phi` goes from 150 degrees as the corner lifts (its curl tighter) to 180 by the pinch's end and
170 in the flick. `corner` says where the pinched corner is on each window frame (inside the tube once it is rolled
up), for a hand that rides it."""
import math

from . import sheet3d as S3

KEYS = {"at", "kind", "dur", "from", "angle", "radius", "back", "shadow", "edge", "pinch"}
CORNERS = {"top-left": (-1.0, -1.0), "top-right": (1.0, -1.0), "bottom-left": (-1.0, 1.0), "bottom-right": (1.0, 1.0)}
DEFAULTS = {"from": "bottom-right", "angle": 0.0, "radius": 0.11, "shadow": 0.45, "pinch": 0.2}
EDGE = {"width": 16.0, "round": 0.03}
DUR = 1.0
MAX_ANGLE = 40.0
PHI = (150.0, 180.0, 170.0)     # degrees round the roll it goes at its full radius before it winds in: as the corner
                                # lifts, by the pinch's end, in the flick
PHI_SWING = 0.2                 # the share of the window by which it reaches the second: the default pinch's end
FLICK = 0.75                    # from this share of the window on it goes to the third, the flick
CURL = 1.5 * math.pi            # radians round a curl that has not reached the roll's radius: a corner just lifted
PITCH = 0.12                    # how much the radius shrinks a turn once it winds in, a share of the roll's radius
CORE = 0.3                      # and the tube's hollow: the share of the radius it winds on round once it gets there


def normalize(spec, out, palette, what):
    """A peel's own keys with their defaults laid into `out` (the transition's common keys), colours resolved."""
    from . import transition as TR
    from .iris import _lightest
    corner = spec.get("from", DEFAULTS["from"])
    if corner not in CORNERS:
        raise TR.TransitionError(f"{what}: from = {corner!r}, expected one of {tuple(CORNERS)} (the corner it is "
                                 f"pinched at)")
    out["from"] = corner
    for k, lo, hi in (("angle", -MAX_ANGLE, MAX_ANGLE), ("radius", 0.01, 0.3), ("shadow", 0.0, 1.0),
                      ("pinch", 0.05, 0.5)):
        out[k] = _number(spec.get(k, DEFAULTS[k]), k, lo, hi, what)
    out["back"] = TR._colour(spec.get("back", _lightest(palette)), palette, f"{what}: back")
    out["ink"] = TR._colour(S3.darkest(palette), palette, f"{what}: shadow")
    edge = spec.get("edge")
    if edge is None:
        out["edge"] = None
    else:
        if not isinstance(edge, dict):
            raise TR.TransitionError(f"{what}: edge = {edge!r}: expected {{color, width, round}}")
        TR._unknown(edge, {"color", "width", "round"}, f"{what}: edge")
        out["edge"] = {"color": TR._colour(edge.get("color", _lightest(palette)), palette, f"{what}: edge.color"),
                       "width": _number(edge.get("width", EDGE["width"]), "edge.width", 0.0, 200.0, what),
                       "round": _number(edge.get("round", EDGE["round"]), "edge.round", 0.0, 0.25, what)}
    return out


def _number(v, k, lo, hi, what):
    from . import transition as TR
    try:
        if isinstance(v, bool):
            raise TypeError
        x = float(v)
    except (TypeError, ValueError):
        raise TR.TransitionError(f"{what}: {k} = {v!r}: expected a number") from None
    if not lo <= x <= hi:
        raise TR.TransitionError(f"{what}: {k} = {x}: expected {lo} .. {hi}")
    return x


def normal(tr):
    """(nx, ny): the unit normal of the fold line, from the stuck part toward the pinched corner (x right, y down): 45
    degrees into that corner, turned `angle` degrees clockwise."""
    sx, sy = CORNERS[tr["from"]]
    a = math.atan2(sy, sx) + math.radians(tr["angle"])
    return (math.cos(a), math.sin(a))


def roll(tr, size):
    """The roll's radius in px in a frame `size` = (w, h)."""
    return tr["radius"] * min(size)


def corner_px(tr, size):
    """The pinched corner of the frame (index coordinates, at the pixels' outer edge)."""
    sx, sy = CORNERS[tr["from"]]
    w, h = size
    return (-0.5 if sx < 0 else w - 0.5, -0.5 if sy < 0 else h - 0.5)


def _along(n, x, y):
    return x * n[0] + y * n[1]


def reach(tr, size):
    """(lo, hi): the frame's extent along the normal (px): `hi` at the pinched corner (or the frame's farthest point that
    way), `lo` at the opposite side."""
    n = normal(tr)
    w, h = size
    xs = [_along(n, x, y) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)]
    return min(xs), max(xs)


def phi(tr, k):
    """How far round the roll the sheet goes at its full radius before it winds in (radians) at window frame `k`."""
    u = (k + 1) / tr["n"]
    a, b, c = PHI
    if u < FLICK:
        s = min(u / PHI_SWING, 1.0)
        return math.radians(a + (b - a) * s * s * (3.0 - 2.0 * s))
    s = (u - FLICK) / (1.0 - FLICK)
    return math.radians(b + (c - b) * s * s * (3.0 - 2.0 * s))


def travel(tr, size):
    """(c0, c1): where the fold line runs along the normal (px) from nothing lifted (the pinched corner) to the whole
    sheet rolled up past the frame's opposite side: no point of the roll lies more than its radius past the line, and
    perspective only pushes it farther out."""
    lo, hi = reach(tr, size)
    return hi, lo - roll(tr, size) - 2.0


def progress(tr, k):
    """0..1 of the way the fold has travelled at window frame `k`: a hand's pull, lifted on the first window frame and 1
    on the last, where the flick has just taken the sheet out of the frame (it leaves at its fastest)."""
    return S3.hand((k + 1) / tr["n"], tr["pinch"])


def line(tr, k, size):
    """Where the fold line is at window frame `k` (px along the normal)."""
    c0, c1 = travel(tr, size)
    return c0 + (c1 - c0) * progress(tr, k)


def spiral(R, f, end):
    """The roll's shape for `end` px of a sheet lifted past its fold (the module's doc), the roll's radius `R`, `f`
    radians round at its full radius: (r0, f, b, (u, x, z) where it starts to wind in, its angle and (u, x, z) where it
    reaches its hollow, the hollow's radius), x and z from the fold line; None when nothing is lifted."""
    r0 = min(R, end / CURL)
    if r0 <= 0.0:
        return None
    b = PITCH * r0 / (2.0 * math.pi)                                   # the radius lost a radian
    rc = CORE * r0
    tm = (r0 - rc) / b
    p_f = (r0 * f, r0 * math.sin(f), r0 * (1.0 - math.cos(f)))
    S = (r0, f, b, p_f, f + tm, None, rc)
    S = S[:5] + (_point(f + tm, S), rc)
    return S


def _point(th, S):
    """(u, x, z) of the roll's point where the sheet heads `th` radians up from along n (the module's spiral `S`)."""
    r0, f, b, (uf, xf, zf), thm, pm, rc = S
    if th <= f:
        return r0 * th, r0 * math.sin(th), r0 * (1.0 - math.cos(th))
    if pm is None or th <= thm:                                         # winding in: the radius r0 - b (th - f)
        t = th - f
        r = r0 - b * t
        return (uf + r0 * t - 0.5 * b * t * t,
                xf + r * math.sin(th) - b * math.cos(th) - r0 * math.sin(f) + b * math.cos(f),
                zf - r * math.cos(th) - b * math.sin(th) + r0 * math.cos(f) + b * math.sin(f))
    um, xm, zm = pm                                                     # round the hollow
    return um + rc * (th - thm), xm + rc * (math.sin(th) - math.sin(thm)), zm - rc * (math.cos(th) - math.cos(thm))


def _angle(u, S):
    """Where the sheet heads (radians up from along n) u px past the fold (the spiral `S`)."""
    r0, f, b, (uf, _, _), thm, (um, _, _), rc = S
    if u <= uf:
        return u / r0
    if u <= um:
        return f + (r0 - math.sqrt(max(r0 * r0 - 2.0 * b * (u - uf), 0.0))) / b
    return thm + (u - um) / rc


def lift(u, c, R, f, end=None):
    """Where the sheet's point u px past the fold line at c lies: (x' along the normal, height). `end` px of the sheet
    are lifted (default u: the point asked for is the pinched corner), `R` the roll's radius, `f` radians round it at
    that radius before it winds in."""
    if u <= 0.0:
        return c + u, 0.0
    S = spiral(R, f, u if end is None else max(end, u))
    _, x, z = _point(_angle(u, S), S)
    return c + x, z


def profile(R, f, end, step=0.5, hollow=1.0):
    """The roll sampled every `step` px of the sheet from the fold to `end` px past it: lists (u, x, z, th) (x from the
    fold line, th where the sheet heads), stopped `hollow` turns round the tube's hollow (more of it lies on that one
    circle, unseen)."""
    S = spiral(R, f, end)
    if S is None:
        return [], [], [], []
    top = min(end, S[5][0] + hollow * 2.0 * math.pi * S[6])
    m = max(int(math.ceil(top / step)), 1)
    us, xs, zs, ts = [], [], [], []
    for i in range(m + 1):
        th = _angle(top * i / m, S)
        u, x, z = _point(th, S)
        us.append(u)
        xs.append(x)
        zs.append(z)
        ts.append(th)
    return us, xs, zs, ts


def corner(tr, frame, size):
    """Where the pinched corner of the sheet is seen on Blender `frame` of the peel's window, for a hand that rides it:
    {"p": [x, y] frame fractions from the top left (outside 0..1 once it has left the frame), "z": its height off the
    frame toward the lens in the frame's short sides, "scale": how much bigger it is seen there (1 on the frame)}. Before
    the window it is the frame's corner, after it the last frame's place."""
    w, h = size
    n = normal(tr)
    cx, cy = corner_px(tr, size)
    if int(frame) < tr["first"]:
        return {"p": [(cx + 0.5) / w, (cy + 0.5) / h], "z": 0.0, "scale": 1.0}
    k = min(int(frame) - tr["first"], tr["n"] - 1)
    xp, yp = _along(n, cx, cy), cy * n[0] - cx * n[1]
    c = line(tr, k, size)
    xq, z = lift(xp - c, c, roll(tr, size), phi(tr, k))
    x, y = xq * n[0] - yp * n[1], xq * n[1] + yp * n[0]
    sx, sy, m = S3.project(x, y, z, size)
    return {"p": [(sx + 0.5) / w, (sy + 0.5) / h], "z": z / min(size), "scale": m}
