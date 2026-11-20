"""Layout of the paper bunting (pure Python + numpy, no bpy: importable from tests): the cord, the flags, the pins or
tape at its ends, the options and the card.

A craft-store paper bunting hung on a wall: a twine cord pinned (or taped) at two points hangs in a catenary `sag` below
the line between them, paper pennants folded over it at even spacing. Measured on craft bunting (`mk ref photos 'bunting
flags'`, `'paper bunting'`) and makers' templates: pennants 120-150 mm wide and 150-200 mm long on 1.5-2 mm twine,
40-60 mm apart, the top folded over the cord in a 10-15 mm flap glued behind; on a wall the end pins hold the cord a
centimetre off it, its loose ends hanging past them.

Frame (metres, "wall_center", as `cafe_poster`'s): the wall is the plane y = 0, the room at -Y, +Z up, +X to the right
as seen from the room; the root lies on the wall midway between the pins (or where `ends` puts them). The cord runs
STANDOFF in front of the wall (TAPED at its taped ends, easing out to STANDOFF over the first EASE metres).

A flag is ONE open sheet of paper: the flap behind, over the cord in a half round, down the front to its point (`shape`
"triangle"), its two tails ("swallowtail") or its round end ("round"). It hangs in the cord's frame (its top along the
cord, so near the pins it leans with the cord), swung a few degrees off the wall at its foot, its lower part twisted a
little and its sides curled (seeded; nothing comes nearer the wall than CLEAR). Colours cycle through `colors` with a
little tone between flags of one colour.

Options (`slots`): ends ([[x, z], [x, z]] the pins in the prop's frame; default `span` (1.6 m) between pins at
(-span / 2, 0) and (span / 2, 0)), sag (0.14 m), size ([0.14, 0.18]: a flag's width and length), shape ("triangle",
"swallowtail", "round"), spacing (m between flag middles along the cord; default the width + 0.035) or count (that many
flags spread evenly), seed (1), colors (["love", "gold", "rose", "pine", "foam", "iris"]), tone (0.07), mount ("pins"
or "tape"). Colour roles: cord ("gold:subtle:0.35": jute twine), pin ("love"), tape ("overlay:gold:0.25").

Card: origin "wall_center", front "-Y", blocks false; `use.anchor` `end_L` / `end_R` (where the cord meets each pin or
tape, `dir` -Y), `use.look` `middle` (the cord's lowest point) and `flags` (the middle of the flags' band)."""
import math
from collections import namedtuple

import numpy as np

from ....core import shell as S
from . import papercraft as PC
from . import showkit as SK

SHAPES = ("triangle", "swallowtail", "round")
MOUNTS = ("pins", "tape")
COLORS = ("love", "gold", "rose", "pine", "foam", "iris")
ROLES = {"cord": "gold:subtle:0.35", "pin": "love", "tape": "overlay:gold:0.25"}
MATS = ("paper", "cord", "pin", "tape")
STANDOFF = 0.009                 # the cord's axis in front of the wall (pins)
TAPED = 0.0016                   # ... where it is taped flat to the wall
EASE = 0.05                      # m along the cord over which a taped cord eases out to STANDOFF
CORD_R = 0.0008                  # twine 1.6 mm thick
WRAP = 0.0024                    # the fold's radius round the cord's axis
FLAP = 0.012                     # the flap glued behind the flag
MARGIN = 0.05                    # from a pin to the nearest flag's edge
GAP = 0.035                      # the default gap between flags
CLEAR = 0.0015                   # no paper nearer the wall than this
TAIL = 0.07                      # a loose end's length past its pin
STEP = 0.004                     # the cord's sampling along its length
Options = namedtuple("Options", "a b sag size shape spacing count seed colors tone mount")


# ----------------------------------------------------------------------------------------------------------- options
def _pair(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(
        isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in v)


def options(slots):
    """The checked options (a bad value is a ValueError that says what to change)."""
    slots = slots or {}
    P = "paper_bunting"
    if "ends" in slots:
        e = slots["ends"]
        if not (isinstance(e, (list, tuple)) and len(e) == 2 and all(_pair(p) for p in e)):
            raise ValueError(f"{P}: ends is [[x, z], [x, z]], the pins in the prop's frame (got {e!r})")
        a, b = (float(e[0][0]), float(e[0][1])), (float(e[1][0]), float(e[1][1]))
        if a[0] > b[0]:
            a, b = b, a
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.3:
            raise ValueError(f"{P}: the ends are {math.hypot(b[0] - a[0], b[1] - a[1]):.2f} m apart; give at least 0.3")
    else:
        span = SK.number(slots, "span", 1.6, P, 0.3, 20.0)
        a, b = (-0.5 * span, 0.0), (0.5 * span, 0.0)
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    sag = SK.number(slots, "sag", min(0.14, 0.2 * chord), P, 0.005, 0.6 * chord)
    size = slots.get("size", [0.14, 0.18])
    if not (_pair(size) and 0.04 <= size[0] <= 0.5 and 0.04 <= size[1] <= 0.6):
        raise ValueError(f"{P}: size is [width, length] of a flag in metres, 0.04 .. 0.5 by 0.04 .. 0.6 (got {size!r})")
    shape = slots.get("shape", "triangle")
    if shape not in SHAPES:
        raise ValueError(f"{P}: shape is {' or '.join(repr(s) for s in SHAPES)} (got {shape!r})")
    mount = slots.get("mount", "pins")
    if mount not in MOUNTS:
        raise ValueError(f"{P}: mount is {' or '.join(repr(s) for s in MOUNTS)} (got {mount!r})")
    colors = slots.get("colors", list(COLORS))
    if isinstance(colors, str):
        colors = [colors]
    if not (isinstance(colors, (list, tuple)) and colors and all(isinstance(c, str) for c in colors)):
        raise ValueError(f"{P}: colors is a list of palette slots, #hex or 'a:b:t' mixes (got {colors!r})")
    for k, c in enumerate(colors):
        SK.rgb(c, slots, f"{P}: colors[{k}]")
    w = float(size[0])
    spacing = SK.number(slots, "spacing", w + GAP, P, 0.6 * w, 5.0) if "spacing" in slots else None
    count = SK.number(slots, "count", 2, P, 1, 200, whole=True) if "count" in slots else None
    if spacing is not None and count is not None:
        raise ValueError(f"{P}: give spacing or count, not both")
    return Options(a, b, sag, (w, float(size[1])), shape, spacing, count,
                   SK.number(slots, "seed", 1, P, 0, 1e9, whole=True), list(colors),
                   SK.number(slots, "tone", 0.07, P, 0.0, 0.5), mount)


# -------------------------------------------------------------------------------------------------------------- cord
def catenary_k(ratio):
    """The catenary's shape factor k with (cosh k - 1) / (2 k) = sag / chord."""
    lo, hi = 1e-6, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (math.cosh(mid) - 1.0) / (2.0 * mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def resample(P, step):
    """Points evenly spaced by arc length along a polyline (both ends kept), and their arc lengths."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])], 1), t


def cord_y(s, length, mount):
    """The cord axis' y at arc length s: STANDOFF, or with tape easing out from TAPED at both ends."""
    if mount == "pins":
        return np.full_like(np.asarray(s, float), -STANDOFF)
    k = PC.smoothstep(0.0, EASE, np.minimum(s, length - s))
    return -(TAPED + (STANDOFF - TAPED) * k)


def cord(O, n=400):
    """(points (m, 3) every STEP along the hanging cord from pin to pin, their arc lengths): a catenary through both
    ends with its lowest point `sag` under the middle of the line between them."""
    a, b = np.array(O.a), np.array(O.b)
    chord = float(np.linalg.norm(b - a))
    k = catenary_k(O.sag / chord)
    t = np.linspace(0.0, 1.0, n + 1)
    drop = (math.cosh(k) - np.cosh(k * (2.0 * t - 1.0))) / (math.cosh(k) - 1.0)
    xz = a + (b - a) * t[:, None]
    xz[:, 1] -= O.sag * drop
    P, s = resample(np.column_stack([xz[:, 0], np.zeros(len(t)), xz[:, 1]]), STEP)
    P[:, 1] = cord_y(s, s[-1], O.mount)
    return P, s


def frame_at(P, s, at):
    """(point, along, up) of the cord at arc length `at`: the tangent and the in-plane normal turned up."""
    i = int(np.clip(np.searchsorted(s, at), 1, len(s) - 1))
    f = (at - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
    p = P[i - 1] + (P[i] - P[i - 1]) * f
    j0, j1 = max(i - 3, 0), min(i + 2, len(P) - 1)
    t = P[j1] - P[j0]
    t[1] = 0.0
    t /= np.linalg.norm(t)
    return p, t, np.array([-t[2], 0.0, t[0]])


def flag_places(O, length):
    """The arc lengths of the flags' middles along a cord `length` long."""
    w = O.size[0]
    room = length - 2.0 * MARGIN - w
    if room < 0:
        raise ValueError(f"paper_bunting: the cord is {length:.2f} m long, too short for a flag {w:g} m wide")
    if O.count is not None:
        n = O.count
        step = room / (n - 1) if n > 1 else 0.0
        if n > 1 and step < 0.85 * w:
            raise ValueError(f"paper_bunting: {n} flags {w:g} m wide overlap on a {length:.2f} m cord; give at most "
                             f"{int(room / (0.85 * w)) + 1}")
    else:
        step = O.spacing if O.spacing is not None else w + GAP
        n = int(room // step) + 1
    mid = 0.5 * length
    return [mid + (i - 0.5 * (n - 1)) * step for i in range(n)]


# ------------------------------------------------------------------------------------------------------------- flags
def outline(shape, w, L, f, u):
    """(x, z) on a flag's front at depth fraction f (0 at the cord, 1 at its foot) and across fraction u (-1 .. 1), in
    its own frame (x along the cord, z up, the cord at z = 0)."""
    hw = 0.5 * w
    if shape == "triangle":
        return u * (hw * (1.0 - f) + 0.0006 * f), np.full_like(u, -f * L)
    if shape == "swallowtail":
        return u * hw * (1.0 - 0.1 * f), -f * (L - 0.3 * L * (1.0 - np.abs(u)))
    return u * hw, -f * ((L - hw) + hw * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)))


def flag_local(shape, w, L, nu=10, nf=14, nwrap=7, nflap=3):
    """One flag in its own frame (x along the cord, y toward the wall, z up; the cord's axis is the x axis): a grid of
    rows (rows, nu + 1, 3) from the flap's foot behind, up over the cord in a half round, down the front to the foot,
    and each row's depth fraction (the flap's and the wrap's are 0)."""
    u = np.linspace(-1.0, 1.0, nu + 1)
    rows, depth = [], []
    for k in range(nflap, 0, -1):                                      # the flap, glued up behind the front
        g = k / nflap
        x, z = outline(shape, w, L, g * FLAP / L, u)
        y = WRAP - (2.0 * WRAP - 0.0006) * PC.smoothstep(0.0, 1.0, g)
        rows.append(np.stack([x, np.full_like(u, y), z], 1))
        depth.append(0.0)
    for ph in np.linspace(math.pi, 0.0, nwrap + 1)[:-1]:              # over the cord
        x, _ = outline(shape, w, L, 0.0, u)
        rows.append(np.stack([x, np.full_like(u, -WRAP * math.cos(ph)), np.full_like(u, WRAP * math.sin(ph))], 1))
        depth.append(0.0)
    for f in np.linspace(0.0, 1.0, nf + 1):                            # down the front
        x, z = outline(shape, w, L, f, u)
        rows.append(np.stack([x, np.full_like(u, -WRAP), z], 1))
        depth.append(f)
    return np.stack(rows), np.array(depth)


def hang(G, depth, w, L, swing, twist, curl, tip):
    """Hang a flag: swing it `swing` degrees off the wall about the cord (its foot toward the room), twist its lower part
    about its own upright axis (up to `twist` degrees at the foot), curl its sides (`curl` m at the foot's corners,
    + toward the wall) and its foot (`tip` m toward the room)."""
    G = G.copy()
    x, y, z = G[..., 0], G[..., 1], G[..., 2]
    f = depth[:, None]
    th = np.radians(twist) * f
    y = y + x * np.sin(th) + curl * (x / (0.5 * w)) ** 2 * f ** 0.7 - tip * f ** 3
    x = x * np.cos(th)
    a = math.radians(swing)
    y, z = y * math.cos(a) + z * math.sin(a), -y * math.sin(a) + z * math.cos(a)
    return np.stack([x, y, z], -1)


def flags(O):
    """[{s, point, along, up, colour, rows}] of every flag: its place on the cord and its sheet's rows in the prop's
    frame."""
    P, s = cord(O)
    rng = np.random.default_rng(int(O.seed))
    w, L = O.size
    G0, depth = flag_local(O.shape, w, L)
    out = []
    for i, at in enumerate(flag_places(O, float(s[-1]))):
        p, t, up = frame_at(P, s, at)
        G = hang(G0, depth, w, L, swing=rng.uniform(0.5, 4.5), twist=rng.uniform(-4.0, 4.0),
                 curl=rng.uniform(-0.003, 0.003), tip=rng.uniform(0.0, 0.004))
        n = np.array([0.0, 1.0, 0.0])
        Wd = p + G[..., 0:1] * t + G[..., 1:2] * n + G[..., 2:3] * up
        if Wd[..., 1].max() > -CLEAR:                                   # too near the wall: hang it plainly, off it
            G = hang(G0, depth, w, L, swing=6.0, twist=0.0, curl=-0.002, tip=0.003)
            Wd = p + G[..., 0:1] * t + G[..., 1:2] * n + G[..., 2:3] * up
        out.append({"s": float(at), "point": p, "along": t, "up": up, "colour": i % len(O.colors), "rows": Wd})
    return out


def flags_mesh(O):
    """(Mesh of every flag's sheet, material paper; its per-vertex flag index)."""
    F = flags(O)
    meshes = [PC.sheet(f["rows"], mat=MATS.index("paper")) for f in F]
    idx = np.concatenate([np.full(len(m.V), i) for i, m in enumerate(meshes)]) if meshes else np.zeros(0, int)
    return S.merge(meshes), idx


# -------------------------------------------------------------------------------------------------- the cord's mesh
def tail_path(end, side, mount):
    """The loose end past a pin or tape: from the end out a little and down, curling, TAIL long."""
    x0, y0, z0 = end
    t = np.linspace(0.0, 1.0, 24)
    x = x0 + side * (0.012 * np.sin(1.4 * t) + 0.006 * t)
    z = z0 - TAIL * t ** 1.15
    y = np.full_like(t, -TAPED) if mount == "tape" else y0 + 0.002 * t
    return np.stack([x, y, z], 1)


def twine(path, radius, sides=8):
    """A round tube along `path` (radius one value or per point), material cord, its UV u the arc length in metres and
    v the turn round it (0 .. 1): what the twine's twist is drawn on."""
    t = 2.0 * math.pi * np.arange(sides) / sides
    r = np.broadcast_to(np.asarray(radius, float), (len(path),))
    m = S.sweep(path, np.stack([np.cos(t), np.sin(t)], 1), scale=r, up=(0.0, -1.0, 0.0), uv=True,
                mat=MATS.index("cord"))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    k = np.arange(len(path) * sides)
    UV = np.zeros((len(m.V), 2))
    UV[k, 0], UV[k, 1] = s[k // sides], (k % sides) / sides
    m.UV = UV
    return m if m.volume() > 0 else m.flip()


def cord_mesh(O):
    """The twine: the hanging cord and its two loose ends, one tube each (material cord)."""
    P, _ = cord(O)
    tubes = [twine(P, CORD_R)]
    for end, side in ((P[0], -1.0), (P[-1], 1.0)):
        tubes.append(twine(tail_path(end, side, O.mount), CORD_R * np.linspace(1.0, 0.8, 24)))
    return S.merge(tubes)


def pin_mesh():
    """A map pin about +Z (out of the wall): the needle from inside the wall past the cord, a collar and a round head."""
    pts = np.array([(0.0, -0.004), (0.00045, -0.0036), (0.00045, STANDOFF + 0.0012), (0.0016, STANDOFF + 0.0016),
                    (0.0018, STANDOFF + 0.0030)])
    r, c = 0.0036, STANDOFF + 0.0058
    a = np.linspace(math.asin((STANDOFF + 0.0030 - c) / r), 0.5 * math.pi, 14)
    prof = np.vstack([pts, np.stack([r * np.cos(a), c + r * np.sin(a)], 1)])
    prof[-1, 0] = 0.0
    return S.lathe(prof, seg=20, mat=MATS.index("pin"))


def ends(O):
    """The cord's two ends (where it meets its pins or tape) in the prop's frame."""
    P, _ = cord(O)
    return P[0], P[-1]


def mounts_mesh(O):
    """The pins (mount "pins"), or (mount "tape") a strip of masking tape over each end, across the cord."""
    e0, e1 = ends(O)
    if O.mount == "pins":
        return S.merge([S.place(pin_mesh(), (p[0], 0.0, p[2]), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)) for p in (e0, e1)])
    rng = np.random.default_rng(int(O.seed) + 7)
    P, _ = cord(O)
    near = np.vstack([P[:40], P[-40:], tail_path(e0, -1.0, O.mount), tail_path(e1, 1.0, O.mount)])

    def ground(x, z):                                       # the wall, and a tent where a strip crosses the cord
        d = np.min(np.hypot(x[:, None] - near[None, :, 0], z[:, None] - near[None, :, 2]), axis=1)
        return -(TAPED + CORD_R) * PC.smoothstep(0.0032, 0.0, d)

    strips = []
    for p, ang in ((e0, 72.0), (e1, 108.0)):
        strips.append(PC.tape_strip((p[0], p[2]), ang + rng.uniform(-14.0, 14.0), 0.05, ground, rng,
                                    mat=MATS.index("tape"), nr=56))
    return S.merge(strips)


# --------------------------------------------------------------------------------------------------------------- card
def card(name, O, roles=None):
    """The prop card (docs/design.md: Prop card)."""
    P, s = cord(O)
    e0, e1 = P[0], P[-1]
    low = P[int(np.argmin(P[:, 2]))]
    F = flags(O)
    band = np.concatenate([f["rows"].reshape(-1, 3) for f in F]) if F else P
    mid = 0.5 * (band.min(0) + band.max(0))
    r = lambda v: [round(float(c), 5) for c in v]                                              # noqa: E731
    return {"origin": "wall_center", "front": "-Y", "blocks": False, "slots": dict(roles or {}),
            "use": {"anchor": [{"name": "end_L", "point": r(e0), "dir": [0.0, -1.0, 0.0]},
                               {"name": "end_R", "point": r(e1), "dir": [0.0, -1.0, 0.0]}],
                    "look": [{"name": "middle", "point": r(low)}, {"name": "flags", "point": r(mid)}]},
            "colliders": [], "bunting": {"flags": len(F), "cord_m": round(float(s[-1]), 4), "shape": O.shape,
                                         "colors": list(O.colors), "mount": O.mount},
            "objects": [f"{name}_{p}" for p in ("flags", "cord", "mounts")]}
