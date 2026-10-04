"""Skyline set: a distant night city (dark facades, shader-lit windows, glow, beacons) around the set root.

The numpy half (layout, parts, facades, ribbons) is pure and unit-tested; the Blender half (`skyline`) imports
`nightkit` lazily so this module also loads without bpy."""
import math

import numpy as np

from . import nightgeo as G
from . import register

SLAB, STEPPED, PODIUM, SLANT, TAPER, WIDE, TWIN, CROWN = range(8)     # building types
SUNK = 3.0          # ground-level parts start this far below the root plane (the land strip hides the foot)
LAND_Z = -2.0       # height of the land strip: clear of any ground at z = 0 for the depth buffer
PIXEL = 6.0e-4      # assumed angular size of one render pixel (rad): window detail fades to its mean around there
GLOW_LAYERS = (("gold", 0.16, 0.9), ("rose", 0.55, 0.9), ("iris", 1.0, 0.25))   # backdrop glow: slot, reach, level
VERTS = 20          # vertices per part: four wall quads and the roof
_CORN = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])     # counter-clockwise from above
BUILD_KEYS = {"name", "kind", "at", "yaw"}                      # the build stage's own keys
KEYS = BUILD_KEYS | {"shape", "az", "arc", "width", "distance", "depth", "count", "density", "height", "footprint",
                     "core", "seed", "lit", "window", "aviation", "haze", "backdrop", "ground"}


# ---------------------------------------------------------------------------------------------------- layout


def frontage(shape, az, arc, width, r, s):
    """Points (n, 2) at depth `r` (m from the root along the heading) and frontage coordinate `s` (-0.5 .. 0.5 across
    the city): an arc of `arc` degrees round the root, or a straight band `width` m wide across the heading."""
    r, s = np.broadcast_arrays(np.asarray(r, float), np.asarray(s, float))
    a0 = math.radians(az)
    if shape == "band":
        d, side = np.array([math.cos(a0), math.sin(a0)]), np.array([-math.sin(a0), math.cos(a0)])
        return d[None] * r.reshape(-1, 1) + side[None] * (s * width).reshape(-1, 1)
    a = a0 + s * math.radians(arc)
    return np.stack([r * np.cos(a), r * np.sin(a)], -1).reshape(-1, 2)


def _fold(s):
    """Reflect values past +-0.5 back inside, so clusters near the ends do not pile up on the edge."""
    s = np.where(s > 0.5, 1.0 - s, s)
    s = np.where(s < -0.5, -1.0 - s, s)
    return np.clip(s, -0.5, 0.5)


def rects_overlap(c, half, yaw, C, HALF, YAW, gap=0.0):
    """True where the rectangle (centre c (2,), half sizes (2,), yaw rad) is closer than `gap` m to any of the
    rectangles C (m, 2), HALF (m, 2), YAW (m,): separating-axis test."""
    if not len(C):
        return False
    d = C - c
    near = np.einsum("ij,ij->i", d, d) < (np.hypot(*half) + np.hypot(HALF[:, 0], HALF[:, 1]) + gap) ** 2
    if not near.any():                                          # most pairs are far apart: skip the exact test
        return False
    d, HALF, YAW = d[near], HALF[near], YAW[near]
    ca, sa = math.cos(yaw), math.sin(yaw)
    ua, va = np.array([ca, sa]), np.array([-sa, ca])
    cb, sb = np.cos(YAW), np.sin(YAW)
    ub, vb = np.stack([cb, sb], 1), np.stack([-sb, cb], 1)
    apart = np.zeros(len(d), bool)
    for ax in (ua, va):
        ra = half[0] * abs(ua @ ax) + half[1] * abs(va @ ax)
        rb = HALF[:, 0] * np.abs(ub @ ax) + HALF[:, 1] * np.abs(vb @ ax)
        apart |= np.abs(d @ ax) > ra + rb + gap
    for ax, k in ((ub, 0), (vb, 1)):
        ra = half[0] * np.abs(ax @ ua) + half[1] * np.abs(ax @ va)
        apart |= np.abs(np.einsum("ij,ij->i", d, ax)) > ra + HALF[:, k] + gap
    return bool((~apart).any())


def layout(n, seed, shape="arc", az=-90.0, arc=110.0, width=1600.0, distance=800.0, depth=260.0,
           height=(20.0, 170.0), footprint=(16.0, 48.0), core=0.6, gap=3.0):
    """Place `n` buildings. Returns arrays (all length n): x, y (centre), yaw (rad, local -Y faces the root), fw, fd
    (footprint incl. podiums), H (top of the main roof above the ground), q (rank 0..1), kind (building type), s
    (frontage coordinate).

    Seeded. Positions follow a frontage coordinate across the city; with `core` > 0 they cluster into a downtown
    (and two smaller centres) where the tall buildings are. Every building has a pool of candidate spots: the tallest
    are placed first and a building that would come closer than `gap` m to another tries its next spot; if none is
    free the last one is used (it overlaps)."""
    n = int(n)
    rng = np.random.default_rng(seed)
    m = max(12 * n, 96)
    u = rng.random((m, 16))
    g = rng.standard_normal(m)
    core = float(np.clip(core, 0.0, 1.0))
    side = rng.choice([-1.0, 1.0])
    cen = np.array([rng.uniform(-0.08, 0.08), side * rng.uniform(0.22, 0.34), -side * rng.uniform(0.18, 0.30)])
    sig, amp = np.array([0.11, 0.07, 0.08]), np.array([1.0, 0.6, 0.5])
    phase = rng.random(2)
    k = np.searchsorted(np.cumsum(amp / amp.sum()), u[:, 1] * 0.999999)
    retry = np.arange(m) // n                                           # which try of its building a candidate is
    p_uni = np.clip(1.0 - 0.65 * core + 0.13 * retry, 0.0, 1.0)         # later tries spread out of the clusters
    s = _fold(np.where(u[:, 0] < p_uni, u[:, 2] - 0.5, cen[k] + sig[k] * g))
    mid = np.max([a * np.exp(-0.5 * ((s - c) / sg) ** 2) for a, c, sg in zip(amp, cen, sig)], axis=0)
    rr = u[:, 3] - 0.5
    # height rank: flat low-rise with a few towers anywhere (core 0) .. towers only downtown (core 1)
    q = (1 - core) * u[:, 4] ** 2.2 + core * np.clip(u[:, 4] ** 1.3 * (0.12 + 0.88 * mid) + 0.1 * mid, 0.0, 1.0)
    H = height[0] + (height[1] - height[0]) * q
    # type by rank
    wts = np.stack([np.full(m, 0.30), 0.10 + 0.32 * q, np.full(m, 0.10), np.full(m, 0.08), 0.03 + 0.12 * q,
                    0.20 * np.clip(1 - 3.5 * q, 0, 1), np.full(m, 0.06), 0.04 + 0.12 * q], 1)
    cum = np.cumsum(wts, 1) / wts.sum(1, keepdims=True)
    kind = np.minimum((u[:, 5][:, None] > cum).sum(1), CROWN)
    # footprint: bigger for taller, never thinner than a 1:7.5 tower, scaled up for podium / wide / twin types
    kq = 0.8 + 0.5 * q
    fw = (footprint[0] + (footprint[1] - footprint[0]) * u[:, 6]) * kq
    fd = (footprint[0] + (footprint[1] - footprint[0]) * u[:, 7]) * kq
    fw, fd = np.maximum(fw, H / 7.5), np.maximum(fd, H / 7.5)
    fx, fy = np.ones(m), np.ones(m)
    for t, x0, xv, y0, yv in ((PODIUM, 1.55, 0.45, 1.35, 0.30), (WIDE, 2.2, 1.3, 1.1, 0.5), (TWIN, 1.5, 0.4, 1.0, 0.0)):
        sel = kind == t
        fx[sel], fy[sel] = x0 + xv * u[sel, 8], y0 + yv * u[sel, 9]
    fw, fd = fw * fx, fd * fy
    H = np.where(kind == WIDE, np.maximum(8.0, height[0] * (0.5 + 0.9 * u[:, 10])), H)
    q = np.where(kind == WIDE, np.minimum(q, 0.15), q)
    # position and orientation (neighbours share a street grid; local -Y faces the root)
    xy = frontage(shape, az, arc, width, distance + rr * depth, s)
    a0 = math.radians(az)
    face = (np.arctan2(xy[:, 1], xy[:, 0]) if shape != "band" else np.full(m, a0)) - math.pi / 2
    grid = math.radians(16.0) * (0.6 * np.sin(2 * math.pi * (1.7 * s + phase[0])) +
                                 0.4 * np.sin(2 * math.pi * (4.1 * s + phase[1])))
    yaw = face + grid + math.radians(5.0) * (2 * u[:, 11] - 1)
    half = np.stack([fw, fd], 1) / 2
    slots = np.argsort(-H[:n], kind="stable")                  # candidate j belongs to building j % n
    got, placed = np.zeros(n, int), []
    for i in slots:
        pool = np.arange(i, m, n)
        pick = pool[-1]
        for shrink in (1.0, 0.8, 0.64, 0.5):                   # no free spot: try again with a smaller building
            free = next((j for j in pool if not placed or not rects_overlap(
                xy[j], half[j] * shrink, yaw[j], xy[placed], half[placed], yaw[placed], gap)), None)
            if free is not None:
                pick = free
                break
        fw[pick], fd[pick], half[pick] = fw[pick] * shrink, fd[pick] * shrink, half[pick] * shrink
        placed.append(pick)
        got[i] = pick
    return {"x": xy[got, 0], "y": xy[got, 1], "yaw": yaw[got], "fw": fw[got], "fd": fd[got], "H": H[got], "q": q[got],
            "kind": kind[got], "s": s[got]}


# ---------------------------------------------------------------------------------------------------- style


def _lin_mix(a, b, t):
    return a + (b - a) * np.asarray(t, float)[..., None]


def style(L, rng, lit, win_rgb, win_w, wall_dark, wall_light, tints, strength):
    """Per-building window and wall parameters (arrays over buildings).

    ch, cw: floor height and window module width (m); ww, wh: window size as a share of the module; pf, pc: share of lit
    floors and of lit windows on a lit floor (their product is the building's lit share); st: emission strength; cth
    (n, 3): cumulative colour thresholds over the window colours; avg (n, 3): the mean window colour; wall (n, 3):
    unlit wall colour (linear rgb). `win_rgb` (k, 3), `win_w` (k,), up to 4 colours."""
    n = len(L["x"])
    r = rng.random((n, 12))
    ch = 3.0 + 1.2 * r[:, 0]
    cw = 2.0 + 1.8 * r[:, 1]
    ribbon = r[:, 2] < 0.18
    ww = np.where(ribbon, 0.88 + 0.1 * r[:, 3], 0.4 + 0.35 * r[:, 3])
    wh = 0.38 + 0.32 * r[:, 4]
    lit_b = np.clip(lit * np.exp(0.95 * rng.standard_normal(n) - 0.45), 0.0, 0.98)
    lit_b = np.where(r[:, 10] < 0.08, lit_b * 0.1, lit_b)                      # a few buildings are nearly dark
    lit_b = np.where(L["kind"] == WIDE, lit_b * 0.45, lit_b)
    a = 0.15 + 0.75 * r[:, 5]
    pf, pc = lit_b ** a, lit_b ** (1 - a)
    st = strength * (0.65 + 0.7 * r[:, 6])
    k = len(win_w)
    w = np.asarray(win_w, float)[None] * np.exp(0.8 * rng.standard_normal((n, k)))
    w /= w.sum(1, keepdims=True)
    cum = np.cumsum(w, 1)
    cth = np.ones((n, 3))
    cth[:, :min(3, k - 1)] = cum[:, :min(3, k - 1)]
    avg = w @ np.asarray(win_rgb, float)
    t = r[:, 7] ** 0.9
    wall = _lin_mix(np.asarray(wall_dark, float)[None], np.asarray(wall_light, float)[None], t)
    if len(tints):
        pick = np.minimum((r[:, 8] * 2.2 * len(tints)).astype(int), len(tints))      # most buildings stay untinted
        amount = 0.14 * r[:, 9]
        for i, c in enumerate(tints):
            sel = pick == i
            wall[sel] = _lin_mix(wall[sel], np.asarray(c, float)[None], amount[sel])
    return {"ch": ch, "cw": cw, "ww": ww, "wh": wh, "pf": pf, "pc": pc, "st": st, "cth": cth, "avg": avg, "wall": wall}


# ---------------------------------------------------------------------------------------------------- parts


PART_FIELDS = ("b", "cx", "cy", "yaw", "z0", "w", "d", "h", "tx", "ty", "sl", "wm")


class _Frame:
    """Appends boxes given in one building's frame (x right, y back, origin at the footprint centre) to shared rows."""

    def __init__(self, rows, i, x, y, yaw, ch):
        self.rows, self.i, self.x, self.y, self.yaw, self.ch = rows, i, x, y, yaw, ch
        self.c, self.s = math.cos(yaw), math.sin(yaw)

    def floors(self, h):
        """`h` rounded to whole floors (at least one)."""
        return max(1, round(h / self.ch)) * self.ch

    def add(self, ox, oy, z0, w, d, h, tx=1.0, ty=1.0, sl=0.0, wm=1.0):
        """A box of w x d x h from z0 at offset (ox, oy); the top is scaled by tx, ty; the +x top corners rise by sl and
        the -x ones fall by sl; wm = 0: no windows."""
        vals = (self.i, self.x + self.c * ox - self.s * oy, self.y + self.s * ox + self.c * oy, self.yaw, z0, w, d, h,
                tx, ty, sl, wm)
        for key, val in zip(PART_FIELDS, vals):
            self.rows[key].append(val)


# The building types. Each takes the frame, footprint w x d, height H, rank q and random numbers r, adds its boxes
# and returns its roof (centre offset x, y, height at the centre, width, depth) and the roof's slope dz/dx.


def _slab(f, w, d, H, q, r):
    h = f.floors(H)
    f.add(0.0, 0.0, -SUNK, w, d, h + SUNK)
    return (0.0, 0.0, h, w, d), 0.0


def _stepped(f, w, d, H, q, r):
    """Two to four tiers, each narrower than the one below and shifted on it."""
    nt = 2 + int(q > 0.3) + int(q > 0.65)
    wt = np.array([2.0, 1.4, 1.0, 0.7][:nt]) * (0.8 + 0.4 * np.array(r[:nt]))
    hs = [f.floors(H * v / wt.sum()) for v in wt]
    sizes, offs = [(w, d)], [(0.0, 0.0)]
    for j in range(1, nt):
        sx, sy = sizes[-1][0] * (0.62 + 0.24 * r[4 + j]), sizes[-1][1] * (0.62 + 0.24 * r[8 + j])
        offs.append((offs[-1][0] + (sizes[-1][0] - sx) / 2 * (r[12 + j] * 1.6 - 0.8),
                     offs[-1][1] + (sizes[-1][1] - sy) / 2 * (r[16 + j] * 1.6 - 0.8)))
        sizes.append((sx, sy))
    z = 0.0
    for j in range(nt):
        f.add(offs[j][0], offs[j][1], -SUNK if j == 0 else z, sizes[j][0], sizes[j][1],
              hs[j] + (SUNK if j == 0 else 0.0))
        z += hs[j]
    return (offs[-1][0], offs[-1][1], z, sizes[-1][0], sizes[-1][1]), 0.0


def _podium(f, w, d, H, q, r):
    """A wide low base with a tower standing on it, sometimes a second smaller one."""
    hp = f.floors(min(max(H * (0.08 + 0.14 * r[0]), 8.0), 36.0))
    f.add(0.0, 0.0, -SUNK, w, d, hp + SUNK)
    tw, td = w * (0.38 + 0.17 * r[1]), d * (0.45 + 0.2 * r[2])
    ox, oy = (w - tw) / 2 * (r[3] * 1.7 - 0.85), (d - td) / 2 * (r[4] * 1.7 - 0.85)
    ht = f.floors(max(H - hp, f.ch * 2))
    f.add(ox, oy, hp, tw, td, ht)
    if r[5] < 0.35:
        sw, sd = tw * 0.55, td * 0.55
        f.add(-ox + (w - sw) / 2 * (r[6] - 0.5) * 0.4, oy, hp, sw, sd, f.floors(ht * (0.4 + 0.4 * r[7])))
    return (ox, oy, hp + ht, tw, td), 0.0


def _slanted(f, w, d, H, q, r):
    """A slab with a sloping roof."""
    sl = H * (0.05 + 0.08 * r[0]) * (1.0 if r[1] < 0.5 else -1.0)
    h = f.floors(max(H - abs(sl), f.ch * 2))
    f.add(0.0, 0.0, -SUNK, w, d, h + SUNK, sl=sl)
    return (0.0, 0.0, h, w, d), sl / (w / 2)


def _tapered(f, w, d, H, q, r):
    """A tower narrowing to the top."""
    t = 0.45 + 0.35 * r[0]
    h = f.floors(H)
    f.add(0.0, 0.0, -SUNK, w, d, h + SUNK, tx=t, ty=t)
    return (0.0, 0.0, h, w * t, d * t), 0.0


def _twin(f, w, d, H, q, r):
    """Two towers side by side, one lower."""
    w1, w2 = w * 0.47, w * 0.40
    d1, d2 = d * (0.75 + 0.25 * r[0]), d * (0.65 + 0.3 * r[1])
    h1, h2 = f.floors(H), f.floors(H * (0.55 + 0.35 * r[2]))
    f.add(-(w - w1) / 2, 0.0, -SUNK, w1, d1, h1 + SUNK)
    f.add((w - w2) / 2, 0.0, -SUNK, w2, d2, h2 + SUNK)
    return (-(w - w1) / 2, 0.0, h1, w1, d1), 0.0


def _crown(f, w, d, H, q, r):
    """A slab with a chamfered crown: a short frustum set in a little from the walls."""
    h1 = f.floors(H * 0.9)
    tc = 0.5 + 0.15 * r[0]
    hc = f.floors(max(H - h1, f.ch))
    f.add(0.0, 0.0, -SUNK, w, d, h1 + SUNK)
    f.add(0.0, 0.0, h1, w * 0.92, d * 0.92, hc, tx=tc, ty=tc)
    return (0.0, 0.0, h1 + hc, w * 0.92 * tc, d * 0.92 * tc), 0.0


# SLAB and WIDE use `_slab`
TYPES = {STEPPED: _stepped, PODIUM: _podium, SLANT: _slanted, TAPER: _tapered, TWIN: _twin, CROWN: _crown}


def _roof_gear(f, rng, r, kind, H, q, roof, slope):
    """An antenna mast (taller buildings more often) and a few windowless boxes of rooftop clutter on a roof (centre
    offset x, y, height, width, depth) that slopes by `slope` dz/dx. Returns the building's highest z and where its
    warning light goes (x, y, z in the building frame)."""
    rx, ry, rz, rw, rd = roof
    top = rz + abs(slope) * rw / 2
    light = (rx + (r[20] - 0.5) * rw * 0.7, ry + (r[21] - 0.5) * rd * 0.7, rz)
    if kind != WIDE and r[22] < 0.08 + 0.55 * q:
        mw = (0.5 + 0.7 * r[23]) * (1.0 + 0.6 * q)
        mh = 8.0 + (0.05 * H + 35.0 * q) * (0.4 + 0.6 * r[24])
        dx, dy = (r[25] - 0.5) * rw * 0.5, (r[26] - 0.5) * rd * 0.5
        z0 = rz + slope * dx - abs(slope) * mw / 2 - 0.8
        f.add(rx + dx, ry + dy, z0, mw, mw, mh + 0.8 + abs(slope) * mw, wm=0.0)
        ztop = z0 + mh + 0.8 + abs(slope) * mw
        if r[27] < 0.4:                                                          # a thinner second stage
            f.add(rx + dx, ry + dy, ztop - 0.5, mw * 0.5, mw * 0.5, mh * 0.6, wm=0.0)
            ztop += mh * 0.6 - 0.5
        light, top = (rx + dx, ry + dy, ztop), max(top, ztop)
    else:
        light = (light[0], light[1], rz + slope * (light[0] - rx) + 1.0)
    for _ in range(int(r[28] * 3.2)):
        ew, ed = rw * (0.08 + 0.14 * rng.random()) + 1.5, rd * (0.08 + 0.14 * rng.random()) + 1.5
        hc = 2.5 + 5.0 * rng.random()
        dx, dy = (rng.random() - 0.5) * (rw - ew) * 0.9, (rng.random() - 0.5) * (rd - ed) * 0.9
        z0 = rz + slope * dx - abs(slope) * ew / 2 - 0.8
        f.add(rx + dx, ry + dy, z0, ew, ed, hc + 0.8 + abs(slope) * ew, wm=0.0)
        top = max(top, z0 + hc + 0.8 + abs(slope) * ew)
    return top, light


def parts(L, S, rng):
    """Boxes of every building: arrays b (building index), cx, cy, yaw, z0, w, d, h, tx, ty (top scale), sl (roof tilt:
    the +x corners rise by sl, the -x corners fall by sl), wm (1: windows), plus `top` (n,) the highest point of each
    building and `tip` (n, 3) a point for its warning light (world frame, root frame of the set). Tiers end on whole
    floors."""
    n = len(L["x"])
    rows = {k: [] for k in PART_FIELDS}
    top, tip = np.zeros(n), np.zeros((n, 3))
    fields = [L[k].tolist() for k in ("x", "y", "yaw", "fw", "fd", "H", "q", "kind")] + [S["ch"].tolist()]
    for i, (x, y, yaw, w, d, H, q, kind, ch) in enumerate(zip(*fields)):
        f = _Frame(rows, i, x, y, yaw, ch)
        r = rng.random(32).tolist()
        roof, slope = TYPES.get(kind, _slab)(f, w, d, H, q, r)
        top[i], (lx, ly, lz) = _roof_gear(f, rng, r, kind, H, q, roof, slope)
        tip[i] = (x + f.c * lx - f.s * ly, y + f.s * lx + f.c * ly, lz)
    out = {k: np.array(v, float) for k, v in rows.items()}
    out["b"] = out["b"].astype(np.int64)
    out["top"], out["tip"] = top, tip
    return out


def facades(P, S, rng):
    """Wall and roof quads of every part: V (n*20, 3), Q (n*5, 4), UV (n*20, 2) and the per-vertex attributes `bid`
    (building index), `wm` (1: windows), `win` (module width, floor height, window width and height as shares of the
    module), `lite` (lit floors, lit windows, strength), `cth` (colour thresholds), `avg` (mean window colour), `wall`.

    The uv of a wall counts window modules: u = column (a whole number of modules per wall, plus a random whole offset
    per wall so walls of one building differ), v = floor (height / floor height); roofs have uv 0."""
    n = len(P["b"])
    b = P["b"]
    size = np.stack([P["w"], P["d"]], 1)
    bot = _CORN[None] * size[:, None, :]
    top = bot * np.stack([P["tx"], P["ty"]], 1)[:, None, :]
    zb = np.broadcast_to(P["z0"][:, None], (n, 4))
    zt = (P["z0"] + P["h"])[:, None] + P["sl"][:, None] * np.sign(_CORN[None, :, 0])
    c, s = np.cos(P["yaw"])[:, None], np.sin(P["yaw"])[:, None]

    def world(xy):
        return np.stack([xy[..., 0] * c - xy[..., 1] * s + P["cx"][:, None],
                         xy[..., 0] * s + xy[..., 1] * c + P["cy"][:, None]], -1)
    B, T = world(bot), world(top)
    k0, k1 = np.arange(4), (np.arange(4) + 1) % 4
    V = np.empty((n, 4, 4, 3))
    V[:, :, 0, :2], V[:, :, 0, 2] = B[:, k0], zb[:, k0]
    V[:, :, 1, :2], V[:, :, 1, 2] = B[:, k1], zb[:, k1]
    V[:, :, 2, :2], V[:, :, 2, 2] = T[:, k1], zt[:, k1]
    V[:, :, 3, :2], V[:, :, 3, 2] = T[:, k0], zt[:, k0]
    roof = np.concatenate([T, zt[..., None]], -1)
    face_w = np.linalg.norm(B[:, k1] - B[:, k0], axis=-1)
    cols = np.maximum(1.0, np.round(face_w / S["cw"][b][:, None]))
    off = rng.integers(0, 256, (n, 4)).astype(float)
    u = np.stack([off, off + cols, off + cols, off], -1)
    v = V[..., 2] / S["ch"][b][:, None, None]
    uv = np.concatenate([np.stack([u, v], -1).reshape(n, 16, 2), np.zeros((n, 4, 2))], 1).reshape(-1, 2)
    Vall = np.concatenate([V.reshape(n, 16, 3), roof], 1).reshape(-1, 3)
    Q = ((np.arange(n) * VERTS)[:, None, None] + np.arange(VERTS).reshape(5, 4)[None]).reshape(-1, 4)

    def per(a):
        return np.repeat(np.asarray(a, float)[b], VERTS, axis=0)
    attrs = {"bid": np.repeat(b.astype(float), VERTS), "wm": np.repeat(P["wm"], VERTS),
             "win": per(np.stack([S["cw"], S["ch"], S["ww"], S["wh"]], 1)),
             "lite": per(np.stack([S["pf"], S["pc"], S["st"]], 1)),
             "cth": per(S["cth"]), "avg": per(S["avg"]), "wall": per(S["wall"])}
    return Vall, Q, uv, attrs


# ---------------------------------------------------------------------------------------------------- ribbons


def city_profile(L, s, sigma):
    """Glow strength (0..1) and glow height (m) at frontage coordinates `s`, from the buildings' volume and height."""
    vol = L["fw"] * L["fd"] * L["H"]
    k = np.exp(-0.5 * ((s[:, None] - L["s"][None]) / sigma) ** 2)
    mass = (k * vol[None]).sum(1)
    reach = (np.exp(-0.5 * ((s[:, None] - L["s"][None]) / (1.7 * sigma)) ** 2) * L["H"][None]).max(1)
    return (mass / max(mass.max(), 1e-9)) ** 0.6, reach


def tree_line(rng, length, spacing=2.0, height=(5.0, 16.0), crown=(3.0, 12.0)):
    """Heights (m) of a row of trees along `length` m, sampled every `spacing` m: round crowns of random radius side by
    side under a slow swell of tallness, now and then a narrow conifer or a clearing. Returns (positions (k,), heights
    (k,), tone (k,)): a random tone 0..1 per tree (conifers dark)."""
    x = np.arange(0.0, length + spacing, spacing)
    f, ph = np.array([1 / 420.0, 1 / 190.0, 1 / 90.0, 1 / 45.0]), rng.random(4) * 2 * math.pi
    a = np.array([1.0, 0.7, 0.5, 0.3])

    def swell(p):
        return 0.5 + 0.5 * (a * np.sin(2 * math.pi * np.asarray(p, float)[..., None] * f + ph)).sum(-1) / a.sum()
    cx, rad, hgt, con = [], [], [], []
    p = -crown[1]
    while p < length + crown[1]:
        conifer = rng.random() < 0.22
        r = rng.uniform(crown[0], crown[1]) * (0.45 if conifer else 1.0)
        if rng.random() < 0.08:                                           # a clearing
            p += rng.uniform(8.0, 30.0)
        p += r
        h = height[0] + (height[1] - height[0]) * swell(p) * (0.45 + 0.55 * rng.random())
        cx.append(p)
        rad.append(r)
        hgt.append(h * (1.35 if conifer else 1.0))
        con.append(conifer)
        p += r * rng.uniform(0.35, 0.9)                                   # neighbouring crowns overlap
    cx, rad, hgt, con = np.array(cx), np.array(rad), np.array(hgt), np.array(con)
    t = np.abs(x[:, None] - cx[None]) / rad[None]
    prof = np.where(con[None], np.clip(1 - t, 0.0, 1.0) ** 1.3, np.sqrt(np.clip(1 - t * t, 0.0, 1.0))) * hgt[None]
    best = prof.argmax(1)
    tone = rng.random(len(cx)) * np.where(con, 0.15, 1.0)
    return x, np.maximum(prof.max(1), height[0] * 0.35), tone[best]


# ---------------------------------------------------------------------------------------------------- builder


def _opt(v, **defaults):
    """A table-or-bool option: True/False switch it, a table overrides the defaults (and `on = false` switches off)."""
    out = dict(defaults)
    if isinstance(v, dict):
        out.update({k: x for k, x in v.items() if k != "on"})
        out["on"] = bool(v.get("on", True))
    else:
        out["on"] = bool(v)
    return out


def _config(name, spec, K, palette):
    """The spec with every default filled in and checked."""
    bad = sorted(set(spec) - KEYS)
    if bad:
        raise ValueError(f"skyline {name!r}: unknown keys {bad} (have {sorted(KEYS - BUILD_KEYS)})")
    c = {"shape": spec.get("shape", "arc"), "az": float(spec.get("az", -90.0)), "arc": float(spec.get("arc", 110.0)),
         "width": float(spec.get("width", 1600.0)), "distance": float(spec.get("distance", 800.0)),
         "depth": float(spec.get("depth", 260.0)), "height": tuple(float(v) for v in spec.get("height", (20.0, 170.0))),
         "footprint": tuple(float(v) for v in spec.get("footprint", (16.0, 48.0))),
         "core": float(spec.get("core", 0.6)),
         "seed": int(spec.get("seed", 1)), "lit": float(spec.get("lit", 0.35)),
         "aviation": max(0, int(spec.get("aviation", 12)))}
    if c["shape"] not in ("arc", "band"):
        raise ValueError(f"skyline {name!r}: shape must be 'arc' or 'band', not {c['shape']!r}")
    c["length"] = c["width"] if c["shape"] == "band" else c["distance"] * math.radians(c["arc"])
    count = spec.get("count")
    if count is None:
        count = round(float(spec["density"]) * c["length"] * c["depth"] / 1e6) if "density" in spec else 160
    c["count"] = max(1, int(count))
    win = dict(spec.get("window", {}))
    c["strength"] = float(win.pop("strength", 1.5))
    win = {k: float(v) for k, v in (win or {"gold": 6.0, "rose": 1.0, "foam": 1.2, "text": 0.8}).items()
           if float(v) > 0}
    if not win:
        raise ValueError(f"skyline {name!r}: window needs at least one colour with a positive weight")
    c["window"] = dict(sorted(win.items(), key=lambda kv: -kv[1])[:4])
    c["haze"] = _opt(spec.get("haze", True), colour=K.haze_hex(palette), distance=c["distance"], cap=0.55)
    c["haze_rgb"] = K.rgb(palette, c["haze"]["colour"])
    c["backdrop"] = _opt(spec.get("backdrop", True), strength=1.0)
    c["ground"] = _opt(spec.get("ground", True))
    return c


def _backdrop(K, name, coll, root, palette, c, L):
    """The city glow: a sheet behind the city on a dome round the root (every point as far from it as the foot of the
    city's back edge, so a camera there never clips it). Strength across it follows the buildings' mass, height their
    tops."""
    fr = (c["shape"], c["az"], c["arc"], c["width"])
    ext = 0.12
    sigma = 0.04 + 0.5 * float(np.mean(np.hypot(L["fw"], L["fd"]))) / max(c["length"], 1.0)
    ns = np.linspace(-0.5 - ext, 0.5 + ext, int(min(160, max(48, c["length"] / 12))))
    gl, reach = city_profile(L, ns, sigma)
    gl = gl * np.clip((0.5 + ext - np.abs(ns)) / ext, 0.0, 1.0) ** 1.5
    gh = np.maximum(reach * 1.5, 0.35 * c["height"][1])
    zt = 2.4 * float(gh.max())
    pts = frontage(*fr, np.full(len(ns), c["distance"] + c["depth"] / 2 + 45.0), ns)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    zs = np.linspace(-SUNK, zt, 14)
    reach_xy = np.sqrt(np.maximum(rad[:, None] ** 2 - zs[None, :] ** 2, 0.0))          # horizontal reach at each height
    V = np.concatenate([pts[:, None, :] / rad[:, None, None] * reach_xy[..., None],
                        np.broadcast_to(zs, reach_xy.shape)[..., None]], -1)
    uv = np.stack([np.broadcast_to(ns[:, None], reach_xy.shape), np.broadcast_to(zs, reach_xy.shape)], -1)
    mesh = G.Mesh()
    mesh.add_grid(V, 0, uv, flip=True, smooth=False, gl=np.repeat(gl[:, None], len(zs), 1),
                  gh=np.repeat(gh[:, None], len(zs), 1))
    K.clean(f"{name}_glow")
    mat = _glow_material(K, f"{name}_glow", palette, c["backdrop"]["strength"], zt)
    return K.to_object(mesh, f"{name}_glow", coll, root, [mat], shadow=False, glossy=False)


def _land(K, name, coll, root, palette, c):
    """The land strip (emerging from the horizon colour at its near edge, so it has no hard line where nothing else
    is on the ground) and two tree lines in front of the city. Returns (object, vertices)."""
    fr = (c["shape"], c["az"], c["arc"], c["width"])
    d, depth, length, ext = c["distance"], c["depth"], c["length"], 0.14
    rng = np.random.default_rng([c["seed"], 11])
    mesh, verts = G.Mesh(), []
    ns = np.linspace(-0.5 - ext, 0.5 + ext, int(max(24, min(96, length / 30))))
    r_in, r_out = d - depth / 2 - 220.0, d + depth / 2 + 90.0
    radii, fade = (r_in, r_in + 40.0, r_in + 120.0, r_out), np.array([0.0, 0.3, 1.0, 1.0])
    V = np.stack([np.concatenate([frontage(*fr, np.full(len(ns), r), ns), np.full((len(ns), 1), LAND_Z)], 1)
                  for r in radii], 1)
    mesh.add_grid(V, 0, None, flip=True, smooth=False, tone=np.full(V.shape[:2], 0.3),
                  fade=np.broadcast_to(fade, V.shape[:2]))
    verts.append(V.reshape(-1, 3))
    span = (1.0 + 2 * ext) * length
    for dr, hs in ((-depth / 2 - 30.0, 1.0), (-depth / 2 + 25.0, 0.6)):          # a near and a farther row of trees
        xs, hg, tone = tree_line(rng, span, 2.5, (5.0 * hs, 16.0 * hs), (3.0, 12.0 * (0.8 + 0.2 * hs)))
        ns = xs / span * (1 + 2 * ext) - 0.5 - ext
        base = frontage(*fr, np.full(len(ns), d + dr), ns)
        V = np.stack([np.concatenate([base, np.full((len(ns), 1), LAND_Z - 1.0)], 1),
                      np.concatenate([base, (LAND_Z + hg)[:, None]], 1)], 1)
        mesh.add_grid(V, 0, None, flip=True, smooth=False, tone=np.repeat(tone[:, None], 2, 1) * 0.9,
                      fade=np.ones(V.shape[:2]))
        verts.append(V.reshape(-1, 3))
    K.clean(f"{name}_land")
    hz = c["haze"]
    mat = _land_material(K, f"{name}_land", palette, c["haze_rgb"], hz["distance"] if hz["on"] else None, hz["cap"])
    return K.to_object(mesh, f"{name}_land", coll, root, [mat], shadow=False), np.concatenate(verts)


def _beacons(K, name, coll, root, palette, P, n):
    """Red warning lights at the tops of the `n` highest buildings: a small hot core and a soft halo each."""
    at = P["tip"][np.argsort(-P["top"])[:n]]
    dist = np.hypot(at[:, 0], at[:, 1])
    cores, halos = G.Mesh(), G.Mesh()
    cores.add(G.instance(G.sphere(1.0, 8, 5), None, at, np.clip(0.0009 * dist, 0.7, 2.5)), 0, True)
    halos.add(G.instance(G.sphere(1.0, 10, 6), None, at, np.clip(0.0035 * dist, 2.0, 10.0)), 0, True)
    K.clean(f"{name}_beacons")
    K.clean(f"{name}_halos")
    core, nb = K.new_material(f"{name}_beacons")
    nb.output(nb.emission(K.rgb(palette, "love"), 6.0))
    halo = K.glow_material(f"{name}_halos", K.rgb(palette, "love"), 2.0, 3.0)
    return [K.to_object(cores, f"{name}_beacons", coll, root, [core], shadow=False),
            K.to_object(halos, f"{name}_halos", coll, root, [halo], shadow=False, glossy=False)]


@register("skyline")
def skyline(name, coll, root, spec, palette):
    """spec (every key optional; metres, degrees, colours = palette slot names or '#hex'):

      shape = "arc"        "arc": buildings spread over an arc round the root; "band": along a straight line across
      az = -90             heading of the city centre from the root (0 = +X, counter-clockwise; -90 = toward -Y)
      arc = 110            angular extent of an arc (deg)
      width = 1600         length of a band (m)
      distance = 800       to the middle of the city (keep <= ~900 to stay inside a 1000 m clip; a camera with a
                           bigger clip_end can see farther cities)
      depth = 260          thickness of the city in distance, so buildings layer
      count = 160          number of buildings (or `density` = buildings per km2 of the city's area)
      height = [20, 170]   shortest and tallest building (m)
      footprint = [16, 48] smallest and largest base width (m); wide blocks and podiums come out bigger
      core = 0.6           0..1: how much taller and denser the middle (downtown) is
      seed = 1
      lit = 0.35           share of windows that are lit (buildings vary: some floors lit, some dark, some dim)
      window = {gold = 6, rose = 1, foam = 1.2, text = 0.8, strength = 1.5}
                           relative weights of the lit window colours (palette slots or '#hex', up to four, a table
                           of your own replaces these; each building leans its own way) and their emission strength
      aviation = 12        red warning lights (love) on the tallest roofs and masts, with soft halos
      haze = {colour = <night horizon haze>, distance = <distance>, cap = 0.55}
                           aerial perspective on the walls (windows stay bright); false switches it off
      backdrop = true      soft city glow behind the buildings: gold at the foot, rose above, fading out; or
                           {strength = 1.0}
      ground = true        a dark strip of land with a tree line in front of the city, so it does not float

    Objects: <name>_city (every wall, one mesh), <name>_glow, <name>_land, <name>_beacons, <name>_halos.
    Walls are emission-only (silhouette tone, haze, ground fog) with a shader window grid: uv counts window modules,
    attribute `bid` seeds the per-floor and per-window randomness. Detail fades to its average brightness when a window
    gets smaller than ~1-2 px (assumes ~0.6 mrad per pixel).
    card: use.look = downtown (middle of the tallest cluster), skyline (middle of the frontage); bounds = {min, max}
    over the walls and the land (root frame)."""
    from . import nightkit as K
    c = _config(name, spec, K, palette)
    rng = np.random.default_rng([c["seed"], 7])
    L = layout(c["count"], c["seed"], c["shape"], c["az"], c["arc"], c["width"], c["distance"], c["depth"], c["height"],
               c["footprint"], c["core"])
    rgb = lambda s: np.array(K.rgb(palette, s))                                              # noqa: E731
    S = style(L, rng, c["lit"], [rgb(s) for s in c["window"]], list(c["window"].values()), rgb("base"), rgb("overlay"),
              [rgb("iris"), rgb("pine"), rgb("rose")], c["strength"])
    P = parts(L, S, rng)
    V, Q, UV, attrs = facades(P, S, rng)
    K.clean(f"{name}_city")
    mesh = G.Mesh()
    mesh.add(G.Geom(V, Q, None, UV), 0, False, **attrs)
    mat = _facade_material(K, f"{name}_city", palette, list(c["window"]), c["haze_rgb"], c["haze"])
    objs, verts = [K.to_object(mesh, f"{name}_city", coll, root, [mat], shadow=False)], [V]
    if c["backdrop"]["on"]:
        objs.append(_backdrop(K, name, coll, root, palette, c, L))
    if c["ground"]["on"]:
        obj, v = _land(K, name, coll, root, palette, c)
        objs.append(obj)
        verts.append(v)
    if c["aviation"]:
        objs += _beacons(K, name, coll, root, palette, P, min(c["aviation"], len(P["top"])))
    box = np.concatenate(verts)
    tall = np.argsort(-L["H"])[:max(3, len(L["H"]) // 10)]
    downtown = [float(L["x"][tall].mean()), float(L["y"][tall].mean()), 0.55 * float(L["H"][tall].mean())]
    mid = frontage(c["shape"], c["az"], c["arc"], c["width"], c["distance"], 0.0)[0]
    return {"kind": "skyline", "paths": {},
            "use": {"look": [{"name": "downtown", "point": downtown},
                             {"name": "skyline", "point": [float(mid[0]), float(mid[1]), 0.4 * float(P["top"].max())]}],
                    "surface": []},
            "colliders": [], "lights": [], "bounds": {"min": box.min(0).tolist(), "max": box.max(0).tolist()}}


# ---------------------------------------------------------------------------------------------------- materials


def _facade_material(K, name, palette, win_slots, haze_rgb, hz):
    """Walls: flat silhouette colour, ground fog and aerial perspective, plus emissive windows added after the haze."""
    m, nb = K.new_material(name)
    mul = lambda a, b: nb.math("MULTIPLY", a, b)                                            # noqa: E731
    add = lambda a, b: nb.math("ADD", a, b)                                                 # noqa: E731
    sub = lambda a, b: nb.math("SUBTRACT", a, b)                                            # noqa: E731
    div = lambda a, b: nb.math("DIVIDE", a, b)                                              # noqa: E731
    lt = lambda a, b: nb.math("LESS_THAN", a, b)                                            # noqa: E731
    gt = lambda a, b: nb.math("GREATER_THAN", a, b)                                         # noqa: E731
    uv = nb.texcoord("UV")
    u, v, _ = nb.sep(uv)
    bid = nb.math("ROUND", nb.attr("bid", "Fac"))
    wm = nb.attr("wm", "Fac")
    win, wa = nb.attr("win", "Color"), nb.attr("win", "Alpha")
    cw, ch, ww = nb.sep(win)
    wh = wa
    lite = nb.attr("lite", "Color")
    pf, pc, st = nb.sep(lite)
    c0, c1, c2 = nb.sep(nb.attr("cth", "Color"))
    avg, wall = nb.attr("avg", "Color"), nb.attr("wall", "Color")
    nrm, inc = nb.geometry("Normal"), nb.geometry("Incoming")
    nz = nb.math("ABSOLUTE", nb.sep(nrm)[2])
    ndv = nb.math("ABSOLUTE", nb.vdot(nrm, inc))
    dist = nb.cam_distance()
    # --- window grid: what one pixel covers, in metres, decides how much detail survives
    fp = mul(dist, PIXEL)
    wmin = nb.math("MINIMUM", mul(cw, ww), mul(ch, wh))
    a_win = sub(1.0, nb.smooth(0.45, 1.1, div(fp, wmin)))
    xc, xf = div(fp, cw), div(fp, ch)
    a_col = sub(1.0, nb.smooth(0.7, 1.6, xc))
    a_flr = sub(1.0, nb.smooth(0.7, 1.6, xf))
    fu, fv = nb.math("FRACT", u), nb.math("FRACT", v)
    iu, iv = sub(u, fu), sub(v, fv)
    hf = nb.node("ShaderNodeTexWhiteNoise", {0: nb.comb(bid, iv, 7.0)}, noise_dimensions="3D")
    hc = nb.node("ShaderNodeTexWhiteNoise", {0: nb.comb(iu, iv, bid)}, noise_dimensions="3D")
    rf, rc = hf.outputs["Value"], hc.outputs["Value"]
    f_tone = nb.sep(hf.outputs["Color"])[0]
    r_col, r_str, _ = nb.sep(hc.outputs["Color"])
    lit_floor, lit_cell = lt(rf, pf), lt(rc, pc)
    x_cell = mul(lit_floor, lit_cell)

    def edge(f, half, soft):
        return nb.math("MULTIPLY_ADD", sub(half, nb.math("ABSOLUTE", sub(f, 0.5))),
                       div(1.0, nb.math("MAXIMUM", soft, 0.04)), 0.5, clamp=True)
    rect = mul(edge(fu, mul(ww, 0.5), mul(xc, 0.7)), edge(fv, mul(wh, 0.5), mul(xf, 0.7)))
    area = mul(ww, wh)
    v_win = nb.mixf(a_win, mul(x_cell, area), mul(x_cell, rect))
    v_col = nb.mixf(a_col, mul(mul(lit_floor, pc), area), v_win)
    v_pat = nb.mixf(a_flr, mul(mul(pf, pc), area), v_col)
    # --- window colour and strength
    colours = [K.rgb(palette, s) for s in win_slots] + [K.rgb(palette, win_slots[-1])] * (4 - len(win_slots))
    cell_col = nb.mixc(gt(r_col, c0), colours[0], colours[1])
    cell_col = nb.mixc(gt(r_col, c1), cell_col, colours[2])
    cell_col = nb.mixc(gt(r_col, c2), cell_col, colours[3])
    colour = nb.mixc(a_col, avg, cell_col)
    gain = mul(nb.mixf(a_col, 1.0, add(0.6, mul(r_str, 0.6))), nb.mixf(a_flr, 1.0, add(0.75, mul(f_tone, 0.4))))
    side = nb.mixf(nb.math("POWER", ndv, 0.5), 0.45, 1.0)
    upright = sub(1.0, nb.smooth(0.22, 0.4, nz))
    above = gt(v, 0.0)
    ext = nb.math("EXPONENT", mul(div(dist, 3.0 * hz["distance"] if hz["on"] else 1e9), -1.0))
    power = mul(mul(mul(v_pat, mul(st, gain)), mul(side, mul(upright, mul(wm, above)))), ext)
    lights = nb.emission(colour, power)
    # --- walls: silhouette tone, lighter roofs and edges, ground fog, aerial perspective
    z = mul(v, ch)
    roof = mul(nb.smooth(0.7, 0.95, nz), 0.35)
    rim = mul(nb.math("POWER", sub(1.0, ndv), 2.0), 0.2)
    tint = K.rgb(palette, "hl_med")
    glass = mul(mul(rect, a_win), mul(upright, mul(wm, above)))                              # unlit windows: dark glass
    col = nb.mixc(mul(glass, 0.4), wall, nb.vmath("SCALE", wall, None, None, 0.7))
    col = nb.mixc(roof, col, tint)
    col = nb.mixc(rim, col, tint)
    shader = nb.emission(col, 1.0)
    if hz["on"]:
        shader = K.haze(nb, shader, haze_rgb, hz["distance"], hz["cap"])
    fog = mul(0.28, nb.math("EXPONENT", mul(nb.math("MAXIMUM", z, 0.0), -1.0 / 45.0)))
    fog_col = tuple(0.8 * a + 0.2 * b for a, b in zip(haze_rgb, K.rgb(palette, "gold")))
    shader = nb.mix_shader(fog, shader, nb.emission(fog_col, 1.0))
    nb.output(nb.add_shader(shader, lights))
    return m


def _glow_material(K, name, palette, strength, top):
    """The city's glow: additive layers of colour that fall off exponentially with height (`GLOW_LAYERS`: slot, reach as
    a share of the local glow height, level), scaled by the vertex attributes (`gl`: how much city there is across the
    frontage, `gh`: how high it reaches there) and faded to nothing at `top` m, where the sheet ends."""
    m, nb = K.new_material(name, blended=True)
    m.use_backface_culling = False
    _, z, _ = nb.sep(nb.texcoord("UV"))
    gl, gh = nb.attr("gl", "Fac"), nb.attr("gh", "Fac")
    z = nb.math("MAXIMUM", z, 0.0)
    end = nb.math("SUBTRACT", 1.0, nb.smooth(0.45 * top, top, z))
    shader = nb.transparent()
    for colour, reach, level in GLOW_LAYERS:
        fall = nb.math("EXPONENT", nb.math("DIVIDE", z, nb.math("MULTIPLY", gh, -reach)))
        gain = nb.math("MULTIPLY", nb.math("MULTIPLY", fall, gl), nb.math("MULTIPLY", end, level * strength))
        shader = nb.add_shader(shader, nb.emission(K.rgb(palette, colour), gain))
    nb.output(shader)
    return m


def _land_material(K, name, palette, haze_rgb, haze_dist, cap):
    """Dark land and trees: base to hl_low by the vertex tone, fading into the haze with distance, and (vertex `fade`
    0 .. 1) emerging from the horizon colour."""
    m, nb = K.new_material(name)
    col = nb.mixc(nb.attr("tone", "Fac"), K.rgb(palette, "base"), K.rgb(palette, "hl_low"))
    shader = nb.emission(col, 1.0)
    if haze_dist is not None:
        shader = K.haze(nb, shader, haze_rgb, haze_dist, cap)
    nb.output(nb.mix_shader(nb.smooth(0.0, 1.0, nb.attr("fade", "Fac")), nb.emission(haze_rgb, 1.0), shader))
    return m
