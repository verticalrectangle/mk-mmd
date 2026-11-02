"""Colour fields on the finished pictures (docs/design.md: Frame effects, Colour fields): the picture work of the `liquid`,
`spiral` and `flood` kinds and of the collages' pattern backs; mkmmd.core.fields checks and times them. Pictures are
float32 RGB in display space (0..1), (h, w, 3); a figure is its coverage, (h, w) in 0..1 over the same picture.

Lengths are in units of the picture's short side about a centre, so every output shows the same field there, cropped to
its own frame. The noise behind the liquid and the flood is value noise on an integer lattice whose values come from a
hash of the lattice point, its time slice and the seed: the same point is the same value in any frame, picture or order.
"""
import math
from collections import OrderedDict

import cv2
import numpy as np

from . import footage as FT
from .core import fields as CF
from .core import framefx as FX

NOISE_UP = 32                    # a noise lattice is first resized this many times by bicubic, then remapped
WORK = 270                       # px per short side the liquid's fields are worked out at (then resized up)
RENDER = 720                     # px: a liquid is drawn at most this on its short side (a whole fraction of the picture)
BOX = 0.95                       # units: a liquid's balls and bubbles are seeded over a square twice this across
BALLS = 6.0                      # balls of oil to a unit square, over all the dishes of a liquid (at scale 1)
LEVEL = 0.42                     # a liquid's edge: where its balls' sum crosses this (a ball alone: at 0.93 x its radius)
ROUGH = 0.3                      # how much noise roughens a liquid's balls
SWIRL = 0.12                     # the water's swirl: its curl warp's reach, in its own size
CLOUD = 0.7                      # the deepest an ink's clouds of dye colour the water (an oil blob of it: 1)
DISPLACE = 0.7                   # how much of the dyed water an ink's oil pushes aside (its own dye shows the more)
MARBLE = 2.2                     # how far the clouds of dye are swirled into each other (in their own size)
LIGHT_DIR = (0.6, 0.8)           # a liquid's drops are lit from the top left: their inward slope along this is lit
SWELL = 0.12                     # how far a hit lowers a liquid's level (its blobs grow)
CELL_CUT = (0.78, 0.36)          # oil cells show where a patch noise is over 0.78 - 0.36 x `cells`
BULGE = (0.55, 0.42)             # a checkerboard's bulge at warp 1: how much it swells its middle, and how far (units)
GRIDS = 24                       # coordinate grids kept (sizes and centres), at most
GRID_BYTES = 200 * 2 ** 20       # and no more than this much of them (a full HD polar grid is 33 MB)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
DRIFT_HUB = 0.5                  # a drift's rings start this many ring widths out from its centre (a still hub)
DRIFT = {"tie": {"pitch": 0.5, "long": 0.86, "wide": 0.56, "round": 0.5, "rim": 0.27},
         "wheel": {"pitch": 1.0, "radius": 0.43, "rim": 0.22, "hub": 0.3}}
# a drift's motif in shares of its cell (a ring or row width across, `pitch` of it along): a tie `long` across and
# `wide` along it, rounded by `round` of its half width, its dark and light sides `rim` of its width deep; a wheel of
# `radius`, its sides `rim` of its diameter deep, a hub of `hub` of its radius in the light tone
DRIFT_AREA = (0.1, 0.3, 0.5, 0.1)  # the shares of a drift's darkest, dark, light and lightest tones, roughly: its mean
_M32 = np.uint64(0xFFFFFFFF)


# ================================================================================================= noise
def _hash(ix, iy, k, seed):
    """Uniform 0..1 float32 of the integer lattice points (ix, iy) of time slice `k`: the same point, slice and seed the
    same value wherever and whenever it is asked."""
    salt = np.uint64((int(k) * 0xCB1AB31F + int(seed) * 0x9E3779B1 + 0x632BE5AB) & 0xFFFFFFFF)
    h = ((np.asarray(ix).astype(np.int64).astype(np.uint64) * np.uint64(0x8DA6B343)) ^
         (np.asarray(iy).astype(np.int64).astype(np.uint64) * np.uint64(0xD8163841)) ^ salt) & _M32
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x7FEB352D)) & _M32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x846CA68B)) & _M32
    h ^= h >> np.uint64(16)
    return (h >> np.uint64(8)).astype(np.float32) / np.float32(16777216.0)


def _catmull(f):
    """The weights of four time slices k - 1 .. k + 2 at fraction `f` past slice k (Catmull-Rom: C1 in time)."""
    f2, f3 = f * f, f * f * f
    return ((-f3 + 2.0 * f2 - f) / 2.0, (3.0 * f3 - 5.0 * f2 + 2.0) / 2.0, (-3.0 * f3 + 4.0 * f2 + f) / 2.0,
            (f3 - f2) / 2.0)


def noise(X, Y, tau, seed):
    """Smooth value noise at lattice coordinates X, Y (float arrays of one shape; one unit is a lattice cell) at time
    `tau` (in lattice slices): float32 about 0..1, bicubic in space and Catmull-Rom through four slices in time. The
    lattice is resized NOISE_UP times by bicubic once, then read linearly between those fine samples (OpenCV's own
    bicubic read ripples a field's slope with the samples)."""
    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
    x0, y0 = int(math.floor(float(X.min()))) - 2, int(math.floor(float(Y.min()))) - 2
    x1, y1 = int(math.floor(float(X.max()))) + 3, int(math.floor(float(Y.max()))) + 3
    gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    k = math.floor(tau)
    lat = np.zeros(gx.shape, np.float32)
    for s, wt in enumerate(_catmull(tau - k)):
        lat += np.float32(wt) * _hash(gx, gy, k - 1 + s, seed)
    fine = cv2.resize(lat, None, fx=NOISE_UP, fy=NOISE_UP, interpolation=cv2.INTER_CUBIC)
    return cv2.remap(fine, (X - np.float32(x0 - 0.5)) * np.float32(NOISE_UP) - np.float32(0.5),
                     (Y - np.float32(y0 - 0.5)) * np.float32(NOISE_UP) - np.float32(0.5), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def bump(x):
    """A smooth bump: 1 at 0, falling to 0 at +-1 (and 0 beyond), (1 - x^2)^2."""
    b = np.clip(np.float32(1.0) - x * x, 0.0, 1.0)
    return b * b


class _Memo(OrderedDict):
    """A few computed arrays kept by key, the oldest dropped past GRIDS entries or GRID_BYTES (the newest stays)."""

    def get_or(self, key, make):
        got = self.get(key)
        if got is None:
            got = self[key] = make()
            while len(self) > GRIDS or (len(self) > 1 and sum(map(_nbytes, self.values())) > GRID_BYTES):
                self.popitem(last=False)
        else:
            self.move_to_end(key)
        return got


def _nbytes(v):
    """Bytes of the arrays in a memo's value (an array or tuples of them)."""
    if isinstance(v, np.ndarray):
        return v.nbytes
    return sum(_nbytes(x) for x in v) if isinstance(v, (tuple, list)) else 0


def polar(size, centre, unit, memo=None):
    """(dx, dy, r, theta) float32 (h, w) of a picture of `size` (w, h) about pixel `centre`: dx right and dy down in
    units of `unit` px, r the distance and theta the angle counter-clockwise on screen from the right (-pi .. pi)."""
    def make():
        w, h = size
        dx = (np.arange(w, dtype=np.float32) - np.float32(centre[0])) / np.float32(unit)
        dy = (np.arange(h, dtype=np.float32) - np.float32(centre[1])) / np.float32(unit)
        DX, DY = np.broadcast_to(dx[None, :], (h, w)), np.broadcast_to(dy[:, None], (h, w))
        return (np.ascontiguousarray(DX), np.ascontiguousarray(DY), np.hypot(DX, DY).astype(np.float32),
                np.arctan2(-DY, DX).astype(np.float32))
    if memo is None:
        return make()
    return memo.get_or(("polar", int(size[0]), int(size[1]), float(centre[0]), float(centre[1]), float(unit)), make)


# ================================================================================================= op-art patterns
def _axis(q, g):
    """Along one stripe coordinate `q` (stripes per px `g`, an array or a number): (k, lo, hi), the stripe k (int32) a
    pixel is in and the shares of its footprint (a box a px wide) in stripes k - 1 and k + 1: antialiased edges."""
    k = np.floor(q)
    f = q - k
    g = np.maximum(g, np.float32(1e-6))
    lo = np.clip(np.float32(0.5) - f / g, 0.0, 1.0)
    hi = np.clip(np.float32(0.5) - (np.float32(1.0) - f) / g, 0.0, 1.0)
    return k.astype(np.int32), lo, hi


def _fine(g):
    """How much of a pattern is too fine to draw where its stripes are `g` per px: 0 for stripes two px wide or more,
    1 at a px or less (it shows its colours' mean there)."""
    return np.clip(np.float32(2.0) * g - np.float32(1.0), 0.0, 1.0)


def _gather(idx, C):
    """(C[idx % m] float32 (h, w, 3), idx % m as uint8) for int32 indices `idx`: a table lookup."""
    m = len(C)
    k8 = ((idx & (m - 1)) if m & (m - 1) == 0 else idx % m).astype(np.uint8)
    lut = np.zeros((256, 1, 3), np.float32)
    lut[:m, 0] = C
    return cv2.LUT(cv2.merge([k8, k8, k8]), lut), k8


def _edges(out, k8, C, terms):
    """`out` (C[k] at each px) with Σ wt x (C[k + s] - C[k]) added over terms [(s, wt)], where wt is not 0 (the edges:
    a few px of each stripe)."""
    m = len(C)
    for s, wt in terms:
        e = np.nonzero(wt > 0)
        if e[0].size:
            j = k8[e].astype(np.int16)
            out[e] += wt[e][:, None] * (C[(j + s) % m] - C[j])


def _mean(out, C, fine):
    """`out` mixed toward the colours' mean by `fine` (a pattern too fine to draw there)."""
    e = np.nonzero(fine > 0)
    if e[0].size:
        out[e] += (C.mean(0) - out[e]) * fine[e][:, None]


def _parity(q, g):
    """The share of a px in the odd stripes of a stripe coordinate `q` (stripes per px `g`), antialiased, half where the
    stripes are too fine to draw."""
    k, lo, hi = _axis(q, g)
    odd = (k & 1).astype(np.float32)
    mid = np.float32(1.0) - lo - hi
    v = mid * odd + (lo + hi) * (np.float32(1.0) - odd)
    return v + (np.float32(0.5) - v) * _fine(g)


def drift_tones(colors):
    """The four tones of a drift, (darkest, dark, light, lightest) float32 rgb, from its 2 to 4 colours by their luma
    (sorted on each call: a palette whose light and dark trade places keeps the illusion turning the same way). Two
    colours are the darkest and the lightest, the others mixed between them (0.3, 0.7); of three the middle is the dark
    tone when it lies nearer the darkest (the light tone mixed halfway to the lightest), else the light one."""
    C = np.asarray(colors, np.float32)
    S = C[np.argsort(C @ LUMA, kind="stable")]
    if len(S) == 4:
        return S[0], S[1], S[2], S[3]
    if len(S) == 3:
        b, m, w = S
        if float((m - b) @ LUMA) <= float((w - m) @ LUMA):
            return b, m, m + (w - m) * np.float32(0.5), w
        return b, b + (m - b) * np.float32(0.5), m, w
    b, w = S
    return b, b + (w - b) * np.float32(0.3), b + (w - b) * np.float32(0.7), w


def _drift_grid(spec, size, centre, unit, memo):
    """What of a drift's cells is the same on every frame: (a, v, lu, out). `a` is how far along its ring or row each px
    lies, in cells, counted the way that ring or row drifts; v across it and lu its cells' length along, in units (short
    sides); `out` the px of no cell (a ring layout's hub) or None. Rings about `centre` each `size` wide, the motifs of
    each a whole number round it, every other ring drifting the other way; rows `size` apart through `centre`, turned
    `angle` degrees counter-clockwise, every other one drifting the other way and offset half a cell."""
    def make():
        s = np.float32(spec["size"])
        pitch = np.float32(DRIFT[spec["motif"]]["pitch"])
        dx, dy, r, th = polar(size, centre, unit, memo)
        if spec["layout"] == "rings":
            rr = r / s - np.float32(DRIFT_HUB)
            k = np.floor(rr)
            v = (rr - k - np.float32(0.5)) * s
            rk = (k + np.float32(0.5 + DRIFT_HUB)) * s
            n = np.maximum(np.round(np.float32(2.0 * math.pi) * rk / (s * pitch)), np.float32(3.0))
            sign = np.float32(1.0) - np.float32(2.0) * np.mod(k, np.float32(2.0))
            return th * (n / np.float32(2.0 * math.pi)) * sign, v, np.float32(2.0 * math.pi) * rk / n, rr < 0
        ang = math.radians(spec["angle"])
        ca, sa = np.float32(math.cos(ang)), np.float32(math.sin(ang))
        x, y = dx * ca - dy * sa, dx * sa + dy * ca
        k = np.floor(y / s + np.float32(0.5))
        lu = s * pitch
        odd = np.mod(k, np.float32(2.0))
        a = (x / lu + np.float32(0.5) * odd) * (np.float32(1.0) - np.float32(2.0) * odd)
        return a, y - k * s, np.full(y.shape, lu, np.float32), None
    if memo is None:
        return make()
    key = ("drift", int(size[0]), int(size[1]), float(centre[0]), float(centre[1]), float(unit), spec["layout"],
           spec["motif"], float(spec["size"]), float(spec["angle"]))
    return memo.get_or(key, make)


def _drift_cells(spec, size, q0, centre, unit, memo):
    """Where each px of a drift lies in its motif's cell: (u, v, lu, out), u along the way the motif drifts (forward is
    + u), 0 at the motif's middle, in units; the motifs moved on `q0` cells, each ring or row its own way (_drift_grid
    has the rest)."""
    a, v, lu, out = _drift_grid(spec, size, centre, unit, memo)
    a = a - np.float32(q0)
    return (a - np.floor(a) - np.float32(0.5)) * lu, v, lu, out


def _box(u, v, hu, hv, rc):
    """The signed distance (negative inside) of px (u, v) to a box of half sides hu, hv rounded by rc, about 0."""
    qx = np.abs(u) - hu + rc
    qy = np.abs(v) - hv + rc
    return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - rc


def drift(spec, size, q0, centre, unit, memo=None):
    """A drift (`spec`: colors, layout, motif, size, angle) over a picture of `size` (w, h) about pixel `centre`, its
    lengths in units of `unit` px, its motifs moved on `q0` cells: float32 RGB (h, w, 3). Each motif is the dark tone
    with its back side (- u) the darkest and its front side (+ u) the lightest, on the light tone: darkest, dark,
    lightest, light, the steps that make it seem to creep forward. Edges antialiased; where the motifs' sides run
    thinner than a px the picture is its tones' mean."""
    B, DG, LG, W = drift_tones(spec["colors"])
    u, v, lu, out = _drift_cells(spec, size, q0, centre, unit, memo)
    geo = DRIFT[spec["motif"]]
    s, px = np.float32(spec["size"]), np.float32(unit)
    if spec["motif"] == "tie":
        hu, hv = np.float32(0.5 * geo["wide"]) * lu, np.float32(0.5 * geo["long"]) * s
        rc = np.float32(geo["round"]) * np.minimum(hu, hv)
        rim = np.float32(2.0 * geo["rim"]) * hu
        dist = lambda du: _box(u - du, v, hu, hv, rc)                                      # noqa: E731
        hub = None
    else:
        rad = np.float32(geo["radius"]) * np.minimum(lu, s)
        rim = np.float32(2.0 * geo["rim"]) * rad
        dist = lambda du: np.hypot(u - du, v) - rad                                      # noqa: E731
        hub = np.clip(np.float32(0.5) - (np.hypot(u, v) - np.float32(geo["hub"]) * rad) * px, 0.0, 1.0)
    cov = lambda d: np.clip(np.float32(0.5) - d * px, 0.0, 1.0)                          # noqa: E731
    inside = cov(dist(0.0))
    if out is not None:
        inside[out] = 0.0
    back = inside * (np.float32(1.0) - cov(dist(rim)))          # what the motif moved forward leaves: its back side
    front = inside * (np.float32(1.0) - cov(dist(-rim)))        # and moved back: its front side
    body = np.maximum(inside - back - front, 0.0)
    if hub is not None:
        hub = np.minimum(hub, body)
        body = body - hub
    D = np.stack([DG - LG, B - LG, W - LG]).astype(np.float32)
    pic = np.stack([body, back, front], -1) @ D + LG
    fine = _fine(np.float32(1.0) / np.maximum(rim * px, np.float32(1e-6)))
    if out is not None:
        fine[out] = 0.0
    mean = sum(np.float32(a) * c for a, c in zip(DRIFT_AREA, (B, DG, LG, W)))
    _mean(pic, mean[None], fine)
    return pic.astype(np.float32)


def pattern(spec, size, q0, centre, unit, memo=None):
    """A pattern (`spec`: pattern, colors and its own keys, mkmmd.core.fields) painted over a picture of `size` (w, h)
    about pixel `centre`, its lengths in units of `unit` px (the short side), moved on by `q0` stripes (fields.phase):
    float32 RGB (h, w, 3), every edge antialiased and a pattern too fine for its pixels its colours' mean."""
    if spec["pattern"] == "drift":
        return drift(spec, size, q0, centre, unit, memo)
    dx, dy, r, th = polar(size, centre, unit, memo)
    C = np.asarray(spec["colors"], np.float32)
    p, m, u, one = spec["pattern"], len(C), np.float32(unit), np.float32(1.0)
    q0 = np.float32(q0)
    if p in ("spiral", "rings", "sunburst"):
        if p == "rings":
            q, g = r / np.float32(spec["size"]) - q0, np.float32(1.0 / (spec["size"] * unit))
        else:
            n = np.float32(m * spec["arms"])
            q = n * th / np.float32(2.0 * math.pi) - q0
            g = n / (np.float32(2.0 * math.pi) * np.maximum(r, np.float32(1e-6)) * u)
            if p == "spiral":
                tw = np.float32(spec["twist"] / 0.5)                         # turns of an arm per half short side
                q = q + n * tw * r
                g = np.hypot(g, n * tw / u)
        k, lo, hi = _axis(q, g)
        out, k8 = _gather(k, C)
        _edges(out, k8, C, ((-1, lo), (1, hi)))
        if isinstance(g, np.ndarray):
            _mean(out, C, _fine(g))
        elif g > 0.5:
            out += (C.mean(0) - out) * _fine(g)
        return out
    if p == "moire":
        s = np.float32(spec["size"])
        g = np.float32(1.0 / (spec["size"] * unit))
        if spec["moire"] == "lines":
            a = math.radians(spec["angle"])
            q1 = dx / s
            q2 = (dx * np.float32(math.cos(a)) - dy * np.float32(math.sin(a))) / s - q0
        else:
            o = np.float32(spec["offset"] * spec["size"] / 2.0)
            q1 = np.hypot(dx + o, dy) / s
            q2 = np.hypot(dx - o, dy) / s - q0
        v1, v2 = _parity(q1, g), _parity(q2, g)
        if m == 2:                                                           # either set, not both: the moire
            return C[0] + (v1 + v2 - np.float32(2.0) * v1 * v2)[..., None] * (C[1] - C[0])
        n1, n2 = one - v1, one - v2
        if m == 3:                                                           # neither, one, both
            ws = (n1 * n2, v1 * n2 + n1 * v2, v1 * v2)
        else:
            ws = (n1 * n2, v1 * n2, n1 * v2, v1 * v2)
        out = np.zeros(r.shape + (3,), np.float32)
        for wt, c in zip(ws, C):
            out += wt[..., None] * c
        return out
    a = math.radians(spec["angle"])                                          # checker: the board turned, bulged out
    ca, sa = np.float32(math.cos(a)), np.float32(math.sin(a))
    x, y = dx * ca - dy * sa, dx * sa + dy * ca
    E = np.float32(BULGE[0] * spec["warp"]) * np.exp(-(r / np.float32(BULGE[1])) ** 2)
    kf = one - E
    s = np.float32(spec["size"])
    q1, q2 = x * kf / s - q0, y * kf / s - q0
    c2 = np.float32(2.0 / BULGE[1] ** 2) * E                                 # the bulge's slope: d(x kf)/dx and so on
    j12 = c2 * x * y
    g1 = np.hypot(kf + c2 * x * x, j12) / (s * u)
    g2 = np.hypot(j12, kf + c2 * y * y) / (s * u)
    k1, lo1, hi1 = _axis(q1, g1)
    k2, lo2, hi2 = _axis(q2, g2)
    out, k8 = _gather(k1 + k2, C)                                            # cell (i, j) is colour (i + j) % m
    e = np.nonzero((lo1 > 0) | (hi1 > 0) | (lo2 > 0) | (hi2 > 0))
    if e[0].size:
        l1, h1, l2, h2 = lo1[e], hi1[e], lo2[e], hi2[e]
        m1, m2 = one - l1 - h1, one - l2 - h2
        j = k8[e].astype(np.int16)
        add = np.zeros((j.size, 3), np.float32)
        for st, wt in ((-2, l1 * l2), (-1, l1 * m2 + m1 * l2), (1, h1 * m2 + m1 * h2), (2, h1 * h2)):
            add += wt[:, None] * (C[(j + st) % m] - C[j])
        out[e] += add
    _mean(out, C, _fine(np.maximum(g1, g2)))
    return out


class Painter:
    """A pattern back on one frame: callable(size (w, h)) -> its picture at that size, about its `center` (fractions of
    the size) with the size's short side as its unit; kept by size, so a back painted in many cells of one size or in
    many panels is painted once."""

    def __init__(self, spec, q, memo):
        self.spec, self.q, self._memo, self._got = spec, float(q), memo, {}

    def __call__(self, size):
        w, h = int(size[0]), int(size[1])
        got = self._got.get((w, h))
        if got is None:
            c = self.spec["center"]
            got = self._got[w, h] = pattern(self.spec, (w, h), self.q, (c[0] * w - 0.5, c[1] * h - 0.5), min(w, h),
                                            self._memo)
        return got


def plate(clip, t, size, fx, back):
    """A footage collage's clip `t` seconds into its span fitted to `size`, a contained one centred on its `back`: a
    colour, a pattern's Painter or a picture of `size` (nothing of a back shows round a clip that covers the frame)."""
    if not callable(back) and not isinstance(back, np.ndarray):
        return clip.picture(t, size, fx["in"], fx["rate"], fx["loop"], fx["fit"], fx["focus"], fx["mirror"], back)
    if fx["fit"] == "cover":
        return clip.picture(t, size, fx["in"], fx["rate"], fx["loop"], "cover", fx["focus"], fx["mirror"])
    return FT.fit(clip.frame(clip.index(t, fx["in"], fx["rate"], fx["loop"])), size, "contain", fx["focus"],
                  fx["mirror"], back(size) if callable(back) else back)


# ================================================================================================= laying a field on
def blend(img, pic, alpha, mode):
    """`img` with `pic` laid on it by `mode` (over, screen, multiply or soft-light) and mixed in by `alpha`, a number or
    (h, w) 0..1."""
    if mode == "over":
        top = pic
    elif mode == "screen":
        top = 1.0 - (1.0 - img) * (1.0 - pic)
    elif mode == "multiply":
        top = img * pic
    else:                                                                    # soft light (Pegtop's): no hard seam at grey
        top = (1.0 - 2.0 * pic) * img * img + 2.0 * pic * img
    a = alpha[..., None] if isinstance(alpha, np.ndarray) else np.float32(alpha)
    return (img + (top - img) * a).astype(np.float32)


# ================================================================================================= the liquid light show
def _dishes(fx):
    """The liquid's dishes from the seed: one per ink and one of clear oil (the last), [(colour or None, turn rad/s, turn
    at 0, pivot (x, y) units, [ball, ...])], turning alternate ways about pivots of their own. A ball is (x, y) units
    where it starts, its radius in units, its wander (x, y amplitudes in units, frequencies in Hz, phases) and its
    breathing (frequency, phase); the balls are seeded over a square BOX x 2 units across about the centre, BALLS of
    them to a unit square over all the dishes (the clear dish half an ink's), fewer as `scale` makes them larger."""
    inks, sc = fx["inks"], fx["scale"]
    per = BALLS * (2.0 * BOX) ** 2 / ((len(inks) + 0.5) * sc * sc)
    out = []
    for i, colour in enumerate(list(inks) + [None]):
        r = np.random.default_rng([fx["seed"], 1013, i])
        u = r.random(4)
        balls = []
        for _ in range(max(2, int(round(per * (0.5 if colour is None else 1.0))))):
            b = r.random(10)
            balls.append((BOX * (2.0 * b[0] - 1.0), BOX * (2.0 * b[1] - 1.0), sc * (0.07 + 0.16 * b[2] ** 2),
                          (0.06 + 0.1 * b[3], 0.06 + 0.1 * b[4]), (0.015 + 0.04 * b[5], 0.015 + 0.04 * b[6]),
                          (6.2832 * b[7], 6.2832 * b[8]), (0.03 + 0.05 * b[9], 6.2832 * ((7.0 * b[9]) % 1.0))))
        out.append((colour, (-1.0) ** i * (0.035 + 0.05 * u[0]), 6.2832 * u[1],
                    (0.4 * (u[2] - 0.5), 0.4 * (u[3] - 0.5)), balls))
    return out


def _ball_at(ball, turn, pivot, t, sp):
    """(x, y, radius) units of a ball at clip second `t`: its start turned `turn` radians about `pivot`, wandering and
    breathing at `sp` times its own pace."""
    bx, by, rad, amp, fr, ph, (fb, pb) = ball
    c, s = math.cos(turn), math.sin(turn)
    x = pivot[0] + (bx - pivot[0]) * c - (by - pivot[1]) * s + amp[0] * math.sin(6.2832 * fr[0] * sp * t + ph[0])
    y = pivot[1] + (bx - pivot[0]) * s + (by - pivot[1]) * c + amp[1] * math.sin(6.2832 * fr[1] * sp * t + ph[1])
    return x, y, rad * (1.0 + 0.16 * math.sin(6.2832 * fb * sp * t + pb))


def _turned(X, Y, turn, pivot):
    """Points X, Y (units) in the frame of a dish turned `turn` radians about `pivot`: what of the dish is there."""
    ca, sa = np.float32(math.cos(turn)), np.float32(math.sin(turn))
    x, y = X - np.float32(pivot[0]), Y - np.float32(pivot[1])
    return x * ca + y * sa, y * ca - x * sa


def _work(size, centre, unit, memo):
    """The liquid's work grid of a picture of `size`: (f, (X, Y)), f the px of the picture per work px (a whole number),
    X, Y float32 the work px centres in units about `centre`."""
    f = max(1, int(round(unit / WORK)))

    def make():
        w, h = size
        ww, hh = -(-w // f), -(-h // f)
        xs = ((np.arange(ww, dtype=np.float32) + 0.5) * f - 0.5 - np.float32(centre[0])) / np.float32(unit)
        ys = ((np.arange(hh, dtype=np.float32) + 0.5) * f - 0.5 - np.float32(centre[1])) / np.float32(unit)
        return (np.ascontiguousarray(np.broadcast_to(xs[None, :], (hh, ww))),
                np.ascontiguousarray(np.broadcast_to(ys[:, None], (hh, ww))))
    return f, memo.get_or(("work", int(size[0]), int(size[1]), float(centre[0]), float(centre[1]), float(unit)), make)


def _up(a, f, size):
    """An array resized up `f` times (bicubic, px centres aligned) and cut to `size` (w, h)."""
    if f == 1:
        return np.ascontiguousarray(a[:size[1], :size[0]])
    return np.ascontiguousarray(cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)[:size[1], :size[0]])


def _cells(X, Y, size_u, seed):
    """Oil cells: Worley noise over X, Y (units), cells about `size_u` units across: (the distance to the nearest cell
    wall, the distance to the cell's own middle), in units."""
    cx, cy = X / np.float32(size_u), Y / np.float32(size_u)
    ix, iy = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
    d1 = np.full(X.shape, 9.0, np.float32)
    d2 = np.full(X.shape, 9.0, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            jx, jy = ix + ox, iy + oy
            d = np.hypot(cx - (jx + 0.1 + 0.8 * _hash(jx, jy, 0, seed)),
                         cy - (jy + 0.1 + 0.8 * _hash(jx, jy, 1, seed))).astype(np.float32)
            d2 = np.where(d < d1, d1, np.minimum(d2, d))
            d1 = np.minimum(d1, d)
    return (d2 - d1) * np.float32(0.5 * size_u), d1 * np.float32(size_u)


def liquid(fx, size, t, centre, swell, memo):
    """The light show of a liquid entry (`fx`: inks, light, scale, speed, pulse, cells, bubbles, seed) at clip second
    `t` over a picture of `size` about pixel `centre`, swollen by `swell` (0..1, the hits): float32 RGB (h, w, 3).

    The lamp's `light` through a dish of water with each ink in it twice: as clouds of dye in the water (soft, no edge)
    and as oil, balls of it (metaballs: a sum of Gaussians, its edge where the sum crosses a level) that wander, breathe
    and turn with their ink's dish, the dishes turning alternate ways; balls of clear oil push the dyed water aside and
    let the light through. A little noise roughens the oil and the water's swirl (a curl-noise warp) bends it all: blobs
    that flow, merge and part. The inks are dyes (the light through them, deeper in a blob's middle); along a blob's
    edge runs a dark meniscus line with the light it bends lit just inside it. Oil cells show in patches of the oil,
    bubbles ride the first dish. A hit swells everything out from the centre by `pulse` and lowers the oil's level, the
    edges flaring. It is drawn at most RENDER px on its short side and resized up (a projection is soft); the bubbles
    at the picture's own size."""
    w, h = size
    unit = float(min(w, h))
    div = max(1, int(math.ceil(unit / RENDER - 1e-9)))
    rs = (-(-w // div), -(-h // div))
    runit, rc = unit / div, ((centre[0] + 0.5) / div - 0.5, (centre[1] + 0.5) / div - 0.5)
    rpx = runit / 1080.0
    f, (X, Y) = _work(rs, rc, runit, memo)
    sp, sc = fx["speed"], fx["scale"]
    z = np.float32(1.0 + fx["pulse"] * swell)
    Xs, Ys = X / z, Y / z
    lw = 0.6 * sc                                                            # the water's swirl
    psi = cv2.GaussianBlur(noise(Xs / lw, Ys / lw, 0.08 * sp * t, fx["seed"] * 7 + 3), (0, 0), 2.0)
    gy, gx = np.gradient(psi, np.float32(f / runit))
    amp = np.float32(SWIRL * sc * lw)
    Xw, Yw = Xs + amp * gy, Ys - amp * gx
    one = np.float32(1.0)
    dishes = _dishes(fx)
    light = np.asarray(fx["light"], np.float32)
    P = np.empty((3, rs[1], rs[0]), np.float32)
    P[:] = light[:, None, None]
    lc, seed = 0.5 * sc, fx["seed"] * 211
    pres = []
    for i, (ink, om, th0, pivot, _) in enumerate(dishes[:-1]):            # the clouds of dye, marbled
        Xi, Yi = _turned(Xw, Yw, om * sp * t + th0, pivot)
        Xi, Yi, tau = Xi / np.float32(lc), Yi / np.float32(lc), 0.06 * sp * t + 17.0 * i
        two = np.float32(2.0)
        qx = noise(Xi, Yi, tau, seed + 5 * i) + np.float32(0.5) * noise(two * Xi, two * Yi, tau, seed + 5 * i + 1)
        qy = noise(Xi, Yi, tau, seed + 5 * i + 2) + np.float32(0.5) * noise(two * Xi, two * Yi, tau, seed + 5 * i + 3)
        m = np.float32(MARBLE)
        cl = noise(Xi + m * (qx - np.float32(0.75)), Yi + m * (qy - np.float32(0.75)), tau, seed + 5 * i + 4)
        pres.append(np.clip((cl - np.float32(0.38)) / np.float32(0.3), 0.0, 1.0))
    pres = np.stack(pres)
    k = pres ** 6
    share = k / np.maximum(k.sum(0), np.float32(1e-6))                       # each place mostly one ink's cloud
    dens = _up(np.float32(CLOUD) * pres.max(0) ** 1.5, f, rs)
    tint = np.einsum("nhw,nc->chw", share, np.asarray(fx["inks"], np.float32))
    for ch in range(3):
        P[ch] *= one - dens * (one - _up(np.ascontiguousarray(tint[ch]), f, rs))
    level = np.float32(LEVEL - SWELL * swell)
    flare = np.float32(0.55 * (1.0 + 0.7 * swell))
    aa = np.float32(max(1.1 * rpx, 0.7))
    oil = np.zeros((rs[1], rs[0]), np.float32)
    for i, (ink, om, th0, pivot, balls) in enumerate([dishes[-1]] + dishes[:-1]):   # clear oil first, then the inks'
        turn = om * sp * t + th0
        G = np.zeros(X.shape, np.float32)
        for ball in balls:
            bx, by, rad = _ball_at(ball, turn, pivot, t, sp)
            G += np.exp(-((Xw - np.float32(bx)) ** 2 + (Yw - np.float32(by)) ** 2) / np.float32(rad * rad))
        F = G + np.float32(ROUGH) * (noise(Xw / (0.25 * sc), Yw / (0.25 * sc), 0.15 * sp * t + 31.0 * i,
                                           fx["seed"] * 101 + i) - np.float32(0.5))
        gyF, gxF = np.gradient(F)
        gn = np.maximum(np.hypot(gxF, gyF), np.float32(1e-6))
        Fu = _up(F, f, rs)
        gu = _up(gn / np.float32(f), f, rs)
        d = (Fu - level) / np.maximum(gu, np.float32(1e-5))                  # render px inside the edge
        c = np.clip(d / aa + np.float32(0.5), 0.0, 1.0)
        oil = oil + c * (one - oil)
        dark = one - np.float32(0.3) * bump(d / np.float32(1.7 * rpx))
        lit = flare * bump((d - np.float32(4.0 * rpx)) / np.float32(2.8 * rpx))
        lens = _up((np.float32(LIGHT_DIR[0]) * gxF + np.float32(LIGHT_DIR[1]) * gyF) / gn, f, rs) * c * \
            np.clip(one - d / np.float32(0.06 * runit), 0.0, 1.0) ** 2   # a drop lit from the top left
        dark = dark * (one - np.float32(0.22) * np.maximum(-lens, 0.0))
        lit = lit + np.float32(0.3) * np.maximum(lens, 0.0)
        if ink is None:                                                      # clear oil: the light through
            a = np.float32(0.85) * c
            for ch in range(3):
                P[ch] += (light[ch] - P[ch]) * a
        else:                                                                # an ink's oil pushes the dyed water
            a = c * (np.float32(0.55) + np.float32(0.45) * np.clip((Fu - level) / np.float32(0.6), 0.0, 1.0))
            push = np.float32(DISPLACE) * c                                  # aside: its own dye, a little of what
            for ch in range(3):                                              # lies under it showing through
                P[ch] += (light[ch] - P[ch]) * push
                P[ch] *= one - a * np.float32(1.0 - ink[ch])
        for ch in range(3):
            P[ch] *= dark
            P[ch] += (one - P[ch]) * lit
    if fx["cells"] > 0:
        _cell_walls(P, fx, oil, (Xw, Yw), dishes[0], f, rs, runit, t)
    out = np.ascontiguousarray(P.transpose(1, 2, 0))
    if div > 1:
        out = np.ascontiguousarray(cv2.resize(out, None, fx=div, fy=div, interpolation=cv2.INTER_CUBIC)[:h, :w])
    if fx["bubbles"] > 0:
        _bubbles(out, fx, t, centre, unit, dishes[0])
    return out


def _cell_walls(P, fx, oil, frame, first, f, size, unit, t):
    """Oil cells in patches (`cells` of the frame) of the oil (`oil`, its coverage), riding the first dish: droplets
    packed into a foam, each a lens, the light gathered in its middle and darker toward its thin dark wall."""
    _, om, th0, pivot, _ = first
    Xi, Yi = _turned(frame[0], frame[1], om * fx["speed"] * t + th0, pivot)
    px = unit / 1080.0
    size_u = 0.06 * fx["scale"]
    wall, mid = _cells(Xi, Yi, size_u, fx["seed"] * 13 + 5)
    wall = _up(wall * np.float32(unit), f, size)
    lens = _up(np.clip(np.float32(1.0) - mid / np.float32(0.5 * size_u), 0.0, 1.0) ** 2, f, size)
    patch = noise(Xi / (0.45 * fx["scale"]), Yi / (0.45 * fx["scale"]), 0.05 * fx["speed"] * t, fx["seed"] * 13 + 6)
    pm = _up(np.clip((patch - np.float32(CELL_CUT[0] - CELL_CUT[1] * fx["cells"])) / np.float32(0.05), 0.0, 1.0), f,
             size) * np.clip(oil * np.float32(1.4), 0.0, 1.0)
    one = np.float32(1.0)
    dark = one - pm * (np.float32(0.18) * bump(wall / np.float32(1.5 * px)) +
                       np.float32(0.2) * np.clip(one - wall / np.float32(size_u * unit * 0.22), 0.0, 1.0) ** 2)
    lit = pm * (np.float32(0.28) * lens + np.float32(0.3) * bump((wall - np.float32(2.4 * px)) / np.float32(1.4 * px)))
    for ch in range(3):
        P[ch] *= dark
        P[ch] += (one - P[ch]) * lit


def bubbles(fx, t, centre, unit, first, size):
    """The liquid's bubbles at clip second `t` over a picture of `size` (w, h) about pixel `centre` (`unit` px its short
    side): `bubbles` to a unit square of the frame, seeded over the square the balls are (BOX x 2 units across) and
    riding the first dish (`first`) with a wobble, each growing in and shrinking away over a life of its own, another
    then taking its slot. Yields (cx, cy, rad) px and the box (x0, y0, x1, y1) a bubble is drawn in, in order."""
    w, h = size
    px = unit / 1080.0
    _, om, th0, (ox, oy), _ = first
    for j in range(int(round(fx["bubbles"] * (2.0 * BOX) ** 2))):
        r0 = np.random.default_rng([fx["seed"], 2027, j]).random(2)
        life = 3.0 + 4.0 * r0[0]
        age = fx["speed"] * t / life + r0[1]
        u, gen = age % 1.0, int(math.floor(age))
        r = np.random.default_rng([fx["seed"], 2029, j, gen]).random(5)
        rad = (0.006 + 0.02 * r[0] ** 2) * fx["scale"] * min(u / 0.12, 1.0, (1.0 - u) / 0.12) * unit
        if rad < 1.5 * px:
            continue
        a = om * fx["speed"] * t + th0
        bx, by = BOX * (2.0 * r[1] - 1.0), BOX * (2.0 * r[2] - 1.0)
        x = ox + (bx - ox) * math.cos(a) - (by - oy) * math.sin(a) + 0.02 * math.sin(6.2832 * (1.3 * u + r[3]))
        y = oy + (bx - ox) * math.sin(a) + (by - oy) * math.cos(a) + 0.02 * math.cos(6.2832 * (1.1 * u + r[4]))
        cx, cy = centre[0] + x * unit, centre[1] + y * unit
        reach = rad + 4.0 * px + 2.0
        x0, x1 = max(int(cx - reach), 0), min(int(cx + reach) + 2, w)
        y0, y1 = max(int(cy - reach), 0), min(int(cy + reach) + 2, h)
        if x1 > x0 and y1 > y0:
            yield (cx, cy, rad), (x0, y0, x1, y1)


def _bubbles(out, fx, t, centre, unit, first):
    """`out` (in place) with the liquid's bubbles (`bubbles`): a rim lit, a darker ring outside it, a glint."""
    px = unit / 1080.0
    one = np.float32(1.0)
    for (cx, cy, rad), (x0, y0, x1, y1) in bubbles(fx, t, centre, unit, first, (out.shape[1], out.shape[0])):
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - np.float32(cx), yy - np.float32(cy)) - np.float32(rad)
        inside = np.clip(np.float32(0.5) - d, 0.0, 1.0)
        lit = np.float32(0.75) * bump((d + np.float32(1.4 * px)) / np.float32(1.8 * px))
        dark = np.float32(0.3) * bump((d - np.float32(1.5 * px)) / np.float32(1.6 * px))
        gl = np.float32(0.9) * bump(np.hypot(xx - np.float32(cx - 0.38 * rad), yy - np.float32(cy - 0.38 * rad)) /
                                    np.float32(max(0.2 * rad, 1.2 * px)))
        sub = out[y0:y1, x0:x1]
        sub *= (one - dark)[..., None]
        sub += (one - sub) * (np.float32(0.18) * inside + lit + gl)[..., None]
        np.clip(sub, 0.0, 1.0, out=sub)


# ================================================================================================= a flood
def flood_front(fx, k, size, centre, unit, memo):
    """(rho px, mottle 0..1) of drop `k` of a flood over a picture of `size` about pixel `centre`: rho is how far the
    ink's front has to run to reach each px, the distance from the centre stretched and squeezed by two noises (lobes
    a third of a unit across, a ragged edge a sixteenth across; `wobble` how much), so the front runs out blotchy and
    fingered; the mottle is the ink's own unevenness."""
    def make():
        dx, dy, r, _ = polar(size, centre, unit, memo)
        seed = fx["seed"] * 1009 + 31 * k + 17
        n1 = noise(dx / np.float32(0.33), dy / np.float32(0.33), 0.0, seed)
        n2 = noise(dx / np.float32(0.065), dy / np.float32(0.065), 0.0, seed + 1)
        wob = np.float32(fx["wobble"])
        stretch = np.float32(1.0) + wob * (np.float32(1.3) * (n1 - np.float32(0.5)) +
                                           np.float32(0.35) * (n2 - np.float32(0.5)))
        rho = r * np.maximum(stretch, np.float32(0.2)) * np.float32(unit)
        mottle = noise(dx / np.float32(0.12), dy / np.float32(0.12), 0.0, seed + 2)
        return rho.astype(np.float32), mottle
    return memo.get_or(("flood", fx["index"], k, int(size[0]), int(size[1]), float(centre[0]), float(centre[1]),
                        float(unit)), make)


def flood(fx, size, centre, drops, memo):
    """The colour of a flood's `drops` ([(k, share, alpha)], fields.flood_drops) over a picture of `size` about pixel
    `centre`, the later over the earlier: (float32 RGB (h, w, 3), coverage (h, w)). Each front is feathered `soft` px at
    1080 with a paler wash a little ahead of it and a darker line of ink along it (as a drop's edge dries darker); the
    colour is a little uneven inside. At share 1 the front is past every px: the frame is filled."""
    w, h = size
    unit = float(min(w, h))
    px = unit / 1080.0
    soft = max(fx["soft"] * px, 1.0)
    pre = np.zeros((h, w, 3), np.float32)                                    # premultiplied by the coverage
    cov = np.zeros((h, w), np.float32)
    one = np.float32(1.0)
    for k, share, alpha in drops:
        rho, mottle = flood_front(fx, k, size, centre, unit, memo)
        R = np.float32(share * (float(rho.max()) + 4.0 * soft))
        ahead = R - rho
        a = np.clip(ahead / np.float32(soft) + np.float32(0.5), 0.0, 1.0)
        wash = np.float32(0.22) * np.clip(ahead / np.float32(3.0 * soft) + one, 0.0, 1.0)
        a = np.maximum(a, wash) * np.float32(alpha)
        shade = one - np.float32(0.07) * (mottle - np.float32(0.5)) * np.float32(2.0 * fx["wobble"]) - \
            np.float32(0.16) * bump((ahead - np.float32(0.8 * soft)) / np.float32(1.4 * soft))
        colour = np.asarray(FX.pick(fx["color"], k), np.float32)
        a3 = a[..., None]
        pre = pre * (one - a3) + (colour * shade[..., None]) * a3
        cov = cov + a * (one - cov)
    return pre / np.maximum(cov, np.float32(1e-6))[..., None], cov


# ================================================================================================= over a cut
class FieldFx:
    """The colour fields' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, frame0, _t, _at,
    _hit_times and _masked), and the backs of its collages (`_back`). `_field_init` is called once."""

    def _field_init(self):
        self._memo = _Memo()             # coordinate grids and flood fronts, by size and centre
        self._painters = {}              # (a collage's index, its back entry) -> (frame, Painter)

    def _on(self, fx, first=True):
        """An entry's hits as clip seconds of their frames (a hit acts from its frame): with `first`, those on the
        span's first frame too, else only those after it (a step on the first frame is where it starts)."""
        out = []
        for h in self._hit_times(fx):
            g = FX.frame_of(h, self.fps, self.frame0)
            if g > fx["f0"] or (first and g == fx["f0"]):
                out.append((g - self.frame0) / self.fps)
        return out

    def _lay(self, fx, frame, img, pic, cover=None):
        """`pic` laid on `img` by the entry's blend (`over` with a coverage `cover`), mixed in by its opacity, its
        strength and, with a `mask`, the figure's coverage or what it leaves."""
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0)) * fx.get("opacity", 1.0)
        if s <= 0.0:
            return img
        region = self._masked(fx, frame)
        a = s if cover is None else cover * np.float32(s)
        if region is not None:
            a = region * a
        return blend(img, pic, a, fx.get("blend", "over"))

    def _liquid(self, fx, frame, img):
        t = self._t(frame)
        swell = CF.swell(t, self._on(fx), fx["decay"])
        return self._lay(fx, frame, img, liquid(fx, self.size, t, self._at(fx, frame)[0], swell, self._memo))

    def _spiral(self, fx, frame, img):
        q = CF.phase(fx, self._t(frame), fx["from"], self._on(fx, first=False))
        return self._lay(fx, frame, img, pattern(fx, self.size, q, self._at(fx, frame)[0], min(self.size),
                                                 self._memo))

    def _flood(self, fx, frame, img):
        # every drop counts, one that fell before the span too: it is already on its way on the span's first frame
        # (a clip that opens full of the colour it closed on), where a liquid's swell or a pattern's step would not be
        times = [(FX.frame_of(h, self.fps, self.frame0) - self.frame0) / self.fps for h in self._hit_times(fx)]
        drops = CF.flood_drops(fx, self._t(frame), times)
        if not drops:
            return img
        pic, cov = flood(fx, self.size, self._at(fx, frame)[0], drops, self._memo)
        return self._lay(fx, frame, img, pic, cov)

    def _back(self, fx, frame, i, img):
        """Entry `i` of a collage's `back` on `frame` (round and round): a colour, a pattern's Painter at that frame,
        moved on by its speed since the span's start and by a step on each of the collage's own hits, or `img` (the
        frame as it reached the effect) for "picture"."""
        b = FX.pick(fx["back"], i)
        if b is None:
            return img
        if not isinstance(b, dict):
            return b
        key = (fx["index"], int(i) % len(fx["back"]))
        got = self._painters.get(key)
        if got is None or got[0] != frame:
            q = CF.phase(b, self._t(frame), fx["from"], self._on(fx, first=False))
            got = self._painters[key] = (frame, Painter(b, q, self._memo))
        return got[1]
