"""Shapes, patterns and layout helpers for the printed posters (pure Python, no bpy; builds `bedroom_decor_field`
expressions).

Design space: the printed area is `aspect` wide and 1 high, centred on the origin, y up; x in [-aspect/2, aspect/2],
y in [-0.5, 0.5]. A *point* is a pair (px, py) of scalar expressions (default `P0` = (X, Y)); `at(p, x, y, deg)` is the
point in a frame placed at (x, y) and turned `deg` degrees anticlockwise, so `disc(0.1, at(P0, 0.2, 0.3))` is a disc
of radius 0.1 centred at (0.2, 0.3). Shapes return a *coverage mask* (a scalar expression: 1 inside, 0 outside, a soft
edge `SOFT` wide); `cov` makes one from a signed distance (negative inside). Masks combine with `*` (and), `union`
(or), `1 - m` (not), `merge` (tessellated pieces: clamped sum, no seams)."""
import math
import random

from .bedroom_decor_field import (X, Y, add, atan2, dot2, fabs, floor, fmax, fmin, fract, len2, mul, pingpong, ramp,
                                  remap, sin, sqrt, sub, xf)

SOFT = 0.0008                       # width of an anti-aliased edge (printed heights)
P0 = (X, Y)


def cov(d, soft=SOFT):
    """Coverage of the signed distance d (negative inside) with a soft edge `soft` wide."""
    return remap(d, soft * 0.5, -soft * 0.5, 0.0, 1.0)


def at(p, x=0.0, y=0.0, deg=0.0, scale=1.0):
    """The point p in the frame placed at (x, y), turned `deg` degrees anticlockwise and scaled."""
    return xf(p[0], p[1], (x, y), math.radians(deg), scale)


def merge(*ms):
    """Clamped sum of masks that tile a shape (no hairline between neighbouring pieces)."""
    tot = ms[0]
    for m in ms[1:]:
        tot = add(tot, m)
    return fmin(tot, 1.0)


def inv(m):
    return sub(1.0, m)


# ------------------------------------------------------------------ signed distances
def box_d(hw, hh, p=P0, rad=0.0):
    """Signed distance to a box of half sizes (hw, hh) about the origin of p, corners rounded by `rad`."""
    qx, qy = sub(fabs(p[0]), hw - rad), sub(fabs(p[1]), hh - rad)
    if rad <= 0.0:
        return fmax(qx, qy)
    return sub(add(len2(fmax(qx, 0.0), fmax(qy, 0.0)), fmin(fmax(qx, qy), 0.0)), rad)


def poly_d(pts, p=P0):
    """Signed distance (max over edge half-planes: exact inside, a lower bound outside) of a convex polygon given
    counter-clockwise."""
    ds = []
    n = len(pts)
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        ln = math.hypot(ex, ey)
        nx, ny = ey / ln, -ex / ln
        ds.append(sub(dot2(p[0], p[1], nx, ny), nx * x0 + ny * y0))
    return fmax(*ds)


def seg_d(a, b, p=P0):
    """Distance from p to the segment a-b (>= 0)."""
    (ax, ay), (bx, by) = a, b
    ex, ey = bx - ax, by - ay
    ll = ex * ex + ey * ey
    t = remap(mul(dot2(sub(p[0], ax), sub(p[1], ay), ex, ey), 1.0 / ll), 0.0, 1.0, 0.0, 1.0)
    return len2(sub(sub(p[0], ax), mul(t, ex)), sub(sub(p[1], ay), mul(t, ey)))



# Shapes as signed-distance functions of a point (so a design can fill, outline and shadow the same shape):
def D_disc(r):
    return lambda p: sub(len2(*p), r)


def D_box(hw, hh, rad=0.0):
    return lambda p: box_d(hw, hh, p, rad)


def D_poly(pts):
    return lambda p: poly_d(pts, p)


def D_ring(r, w):
    return lambda p: sub(fabs(sub(len2(*p), r)), w * 0.5)


def D_tri(side, deg=0.0):
    """Equilateral triangle of the given side, centred on its centroid, apex up, turned deg degrees."""
    ci = side / (2.0 * math.sqrt(3.0))                       # inradius
    r = 2.0 * ci                                               # circumradius
    angs = [math.radians(90.0 + deg + 120.0 * k) for k in range(3)]
    pts = [(r * math.cos(t), r * math.sin(t)) for t in angs]
    return D_poly(pts)


def repeat_y(p, pitch):
    """The point folded into the strip of height `pitch` (horizontal copies of whatever is drawn about y = 0)."""
    return (p[0], mul(sub(fract(add(mul(p[1], 1.0 / pitch), 0.5)), 0.5), pitch))

# ------------------------------------------------------------------ shapes
def disc(r, p=P0):
    return cov(sub(len2(*p), r))


def ring(r, w, p=P0):
    return cov(sub(fabs(sub(len2(*p), r)), w * 0.5))


def rect(hw, hh, p=P0, rad=0.0):
    return cov(box_d(hw, hh, p, rad))


def poly(pts, p=P0):
    return cov(poly_d(pts, p))


def band(f, lo, hi):
    """lo < f < hi for any scalar expression f."""
    return cov(fmax(sub(lo, f), sub(f, hi)))


def line(a, b, w, p=P0):
    """Segment a-b, w thick, round caps."""
    return cov(sub(seg_d(a, b, p), w * 0.5))


def sparkle(s, p=P0):
    """A four-point star (astroid) about the origin of p reaching out to distance s along the axes."""
    f = sub(add(sqrt(fabs(p[0])), sqrt(fabs(p[1]))), math.sqrt(s))
    return cov(mul(f, math.sqrt(s) / math.sqrt(2.0)))


def stripes(f, period, duty=0.5, phase=0.0):
    """Periodic bands of f: the fraction `duty` of every `period`, the first centred on f = phase * period."""
    t = sub(fract(add(mul(f, 1.0 / period), 0.5 - phase)), 0.5)
    return cov(mul(sub(fabs(t), duty * 0.5), period))


def tri_wave(f, period):
    """Triangle wave of f in [0, 1] (0 at f = 0, period `period`)."""
    return mul(pingpong(f, period * 0.5), 2.0 / period)


def zigzag(p, period, amp, w):
    """A zigzag stroke along x about the origin of p: peak-to-peak `2 amp`, w thick."""
    y = sub(p[1], mul(sub(tri_wave(p[0], period), 0.5), 2.0 * amp))
    return cov(sub(fabs(y), w * 0.5))


def wave(p, period, amp, w):
    """A sine stroke along x about the origin of p."""
    y = sub(p[1], mul(sin(mul(p[0], 2.0 * math.pi / period)), amp))
    return cov(sub(fabs(y), w * 0.5))


def dots(p, period, r, deg=0.0, stagger=False):
    """Lattice of discs of radius r (square, or hexagonal when stagger) turned `deg` degrees."""
    q = at(p, 0.0, 0.0, deg)
    if stagger:
        row = floor(add(mul(q[1], 1.0 / period), 0.5))
        shift = mul(sub(row, mul(floor(mul(row, 0.5)), 2.0)), 0.5)          # 0, .5, 0, .5 ...
        lx = sub(fract(add(add(mul(q[0], 1.0 / period), 0.5), shift)), 0.5)
    else:
        lx = sub(fract(add(mul(q[0], 1.0 / period), 0.5)), 0.5)
    ly = sub(fract(add(mul(q[1], 1.0 / period), 0.5)), 0.5)
    return disc(r, (mul(lx, period), mul(ly, period)))


def checker(p, period):
    """Checkerboard of squares `period` wide (1 on the squares where sin * sin > 0)."""
    s = mul(sin(mul(p[0], math.pi / period)), sin(mul(p[1], math.pi / period)))
    k = SOFT * math.pi / period * 0.5
    return remap(s, -k, k, 0.0, 1.0)


def rays(p, n, duty=0.5, phase=0.0):
    """n rays fanning out of the origin of p (a pizza of equal wedges, `duty` of each wedge covered)."""
    ang = mul(atan2(p[1], p[0]), n / (2.0 * math.pi))
    t = sub(fract(add(ang, 0.5 - phase)), 0.5)
    arc = mul(len2(*p), 2.0 * math.pi / n)                                   # wedge width at this radius
    return cov(mul(sub(fabs(t), duty * 0.5), arc))


def sprinkle(p, cell, r, keep=0.5, seed=0, shape="disc", jitter=0.30, grow=0.45):
    """Scattered shapes, one candidate per `cell` of a square lattice: deterministic pseudo-random offset, size
    (r * (1 - grow .. 1)) and presence (`keep` = the fraction of cells that have one) from sines of the cell index.
    shape: disc | plus | diamond | ring. The shape stays inside its cell while r <= (0.5 - jitter) * cell."""
    gx, gy = mul(p[0], 1.0 / cell), mul(p[1], 1.0 / cell)
    ix, iy = floor(gx), floor(gy)

    def rnd(a, b, ph):                                                     # smooth hash of the cell index, -1 .. 1
        return sin(add(add(mul(ix, a), mul(iy, b)), ph + seed * 1.7))

    jx, jy = mul(rnd(2.399963, 5.531327, 0.3), jitter), mul(rnd(4.168453, 1.870923, 1.1), jitter)
    lx = mul(sub(sub(sub(gx, ix), 0.5), jx), cell)
    ly = mul(sub(sub(sub(gy, iy), 0.5), jy), cell)
    size = mul(add(rnd(3.713101, 6.047739, 2.3), 1.0), 0.5 * grow)          # 0 .. grow
    size = mul(sub(1.0, size), r)
    thr = math.cos(math.pi * keep)                                          # sin > thr for a fraction `keep`
    present = remap(rnd(5.281643, 3.162774, 0.7), thr, thr + 0.12, 0.0, 1.0)
    q = (lx, ly)
    if shape == "disc":
        m = cov(sub(len2(*q), size))
    elif shape == "ring":
        m = cov(sub(fabs(sub(len2(*q), mul(size, 0.8))), mul(size, 0.2)))
    elif shape == "diamond":
        m = cov(sub(add(fabs(lx), fabs(ly)), size))
    elif shape == "plus":
        arm = mul(size, 0.28)
        m = fmax(cov(fmax(sub(fabs(lx), size), sub(fabs(ly), arm))), cov(fmax(sub(fabs(lx), arm), sub(fabs(ly), size))))
    else:
        raise ValueError(shape)
    return mul(m, present)


# ------------------------------------------------------------------ gradients
def grad_y(y0, y1):
    """0 at y0, 1 at y1 (clamped)."""
    return remap(Y, y0, y1, 0.0, 1.0)


def grad_dir(p, x0, y0, x1, y1):
    """Linear gradient from (x0, y0) (0) to (x1, y1) (1), clamped."""
    ex, ey = x1 - x0, y1 - y0
    ll = ex * ex + ey * ey
    return remap(sub(dot2(p[0], p[1], ex, ey), ex * x0 + ey * y0), 0.0, ll, 0.0, 1.0)


def radial(p, r0, r1):
    """0 at radius r0, 1 at r1 (clamped; r0 > r1 allowed)."""
    return remap(len2(*p), r0, r1, 0.0, 1.0)


def profile(f, lo, hi, pts, interp="LINEAR"):
    """Height lookup: `pts` [(t, value)] with t in 0..1 across f in [lo, hi] -> scalar expression."""
    return ramp(remap(f, lo, hi, 0.0, 1.0), pts, interp)


def bands(f, lo, hi, intervals):
    """1 where f lies in one of `intervals` [(a, b)] (hard edges, resolution (hi - lo) / 256): one constant Colour Ramp
    over f in [lo, hi], whatever the number of intervals."""
    span = hi - lo
    stops, last = [(0.0, 0.0)], 0.0
    for a_, b_ in sorted(intervals):
        ta, tb = min(max((a_ - lo) / span, 0.0), 1.0), min(max((b_ - lo) / span, 0.0), 1.0)
        if tb - ta < 1.0 / 512.0 or ta < last + 1e-6:
            continue
        stops += [(ta, 1.0), (tb, 0.0)]
        last = tb
    return ramp(remap(f, lo, hi, 0.0, 1.0), stops, "CONSTANT")


class BlockRow:
    """A row of abstract slanted blocks (a logotype that spells nothing): every block has its own width, its top and
    bottom edge cut back by a random amount, and some have a window. The whole row is a handful of constant Colour
    Ramps looked up along the sheared x, whatever the number of blocks. `mask(p, grow)` is the coverage at the point
    p with every block grown by `grow` (an outline), `windows=False` fills the windows; `fill_t(p)` is the height
    across the row, 0 at its top and 1 at its bottom (for a vertical gradient)."""

    def __init__(self, cx, cy, h, widths, gap=0.012, slant=12.0, seed=0, cut=0.5, windows=0.4):
        rng = random.Random(seed)
        self.cx, self.cy, self.h = cx, cy, h
        self.shear = math.tan(math.radians(slant))
        total = sum(widths) + gap * (len(widths) - 1)
        self.lo, self.hi = cx - total / 2 - gap, cx + total / 2 + gap
        x = cx - total / 2
        self.blocks = []                                # (x0, x1, top, bottom, window (x0, x1) or None)
        for i, w in enumerate(widths):
            top = h / 2 - h * cut * rng.choice((0.0, 0.25, 0.5)) if i % 3 != 1 else h / 2
            bot = -h / 2 + h * cut * rng.choice((0.0, 0.3)) if i % 2 == 0 else -h / 2
            win = (x + w * 0.34, x + w * 0.66) if (rng.random() < windows and w > 4 * gap) else None
            self.blocks.append((x, x + w, top, bot, win))
            x += w + gap

    def _t(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def sheared(self, p):
        return sub(p[0], mul(sub(p[1], self.cy), self.shear))

    def mask(self, p=P0, grow=0.0, windows=True):
        xs = remap(self.sheared(p), self.lo, self.hi, 0.0, 1.0)
        pres, top, bot, win = [(0.0, 0.0)], [], [], [(0.0, 0.0)]
        for x0, x1, tp, bt, w in self.blocks:
            t0, t1 = self._t(x0 - grow), self._t(x1 + grow)
            pres += [(t0, 1.0), (t1, 0.0)]
            top.append((t0, self.cy + tp + grow))
            bot.append((t0, self.cy + bt - grow))
            if w is not None:
                win += [(self._t(w[0]), 1.0), (self._t(w[1]), 0.0)]
        top, bot = [(0.0, top[0][1])] + top, [(0.0, bot[0][1])] + bot
        d = fmax(sub(ramp(xs, bot, "CONSTANT"), p[1]), sub(p[1], ramp(xs, top, "CONSTANT")))
        out = mul(ramp(xs, pres, "CONSTANT"), cov(d))
        if windows and len(win) > 1:
            hole = mul(ramp(xs, win, "CONSTANT"), cov(sub(fabs(sub(p[1], self.cy)), self.h * 0.1)))
            out = mul(out, inv(hole))
        return out

    def fill_t(self, p=P0):
        return remap(p[1], self.cy + self.h / 2, self.cy - self.h / 2, 0.0, 1.0)


# ------------------------------------------------------------------ layout maths (plain Python)
def ridge(seed, n, lo, hi, valley=None, width=0.2, sink=0.15, rough=0.6):
    """Mountain ridge profile: n+1 vertices [(t, height)] with t in 0..1, heights in [lo, hi] (alternating peaks and
    saddles with `rough` random variation). `valley` = t of a gap the ridge sinks into (to `lo + sink * (hi - lo)`)
    over a half width `width` (so a sun can stand there)."""
    rng = random.Random(seed)
    pts = []
    for i in range(n + 1):
        t = i / n
        peak = i % 2 == 1
        base = 0.78 if peak else 0.30
        h = lo + (hi - lo) * min(1.0, max(0.0, base + rough * (rng.random() - 0.5) * (0.7 if peak else 0.5)))
        if valley is not None:
            k = min(1.0, abs(t - valley) / width)
            k = k * k * (3.0 - 2.0 * k)
            h = lo + (hi - lo) * sink + (h - lo - (hi - lo) * sink) * k
        pts.append((t, h))
    return pts


def poisson(seed, n, w, h, min_dist, avoid=(), margin=0.0, tries=2000):
    """Up to n points in [-w/2, w/2] x [-h/2, h/2] (inset by `margin`) at least `min_dist` apart and outside every
    (cx, cy, r) circle of `avoid`; deterministic from `seed`. Returns [(x, y)]."""
    rng = random.Random(seed)
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        x = (rng.random() - 0.5) * (w - 2 * margin)
        y = (rng.random() - 0.5) * (h - 2 * margin)
        if any(math.hypot(x - q[0], y - q[1]) < min_dist for q in pts):
            continue
        if any(math.hypot(x - ax, y - ay) < ar for ax, ay, ar in avoid):
            continue
        pts.append((x, y))
    return pts

