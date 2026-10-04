"""Matte maths of the cut effects (docs/design.md: Shots: Transitions and inserts), the post layer: numpy and OpenCV.

A matte is kept as a signed distance field, in pixels, negative inside the shape. Turning and scaling a field moves its zero
contour exactly (the distance scales with the shape), so whatever the zoom the edge of a matte is one pixel wide and
follows the real shape; a coverage image scaled the same way would blur 18-fold. From the field:

    coverage(d)      the shape, antialiased over one pixel across its edge
    inside_band(d, w)  the shape grown by w pixels (a rim under it, an outline round it)

Fields: `AlphaField` (any silhouette's antialiased coverage, `signed_distance`) and `CloudField` (the thought bubble);
`placed` evaluates one, turned and scaled about a pivot, over a frame. Images are float32 RGB in display space, 0..1."""
from functools import lru_cache

import cv2
import numpy as np

EPS = 0.5 / 255.0                        # a coverage within this of 0 or 1 is not partial
FAR = 1.0e4                              # distance given to pixels no shape reaches


@lru_cache(maxsize=6)
def grid(h, w):
    """Pixel index coordinates (x, y), float32 (h, w) each: pixel (i, j) is centred on x = j, y = i."""
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.float32), ys.astype(np.float32)


def frac_to_px(f, size):
    """A point in frame fractions (from the top-left corner) -> index coordinates (pixel centres on integers)."""
    return f[0] * size[0] - 0.5, f[1] * size[1] - 0.5


# ================================================================================================= distance from coverage
def _edge_offset(a, nx, ny):
    """Signed distance (px, positive outside) from a pixel's centre to a straight edge that covers the share `a` of the
    pixel, the edge's outward normal being (nx, ny): the exact inverse of the area of a unit square on one side of a
    line. With c, s the larger and smaller of |nx|, |ny| the area is linear in the offset, 0.5 - c * offset, between a =
    s / 2c and 1 - s / 2c and quadratic in the two corners beyond."""
    c, s = np.maximum(np.abs(nx), np.abs(ny)), np.minimum(np.abs(nx), np.abs(ny))
    x1 = (c + s) / 2.0
    lo = s / (2.0 * np.maximum(c, 1e-6))
    t = np.where(a <= lo, -x1 + np.sqrt(np.maximum(2.0 * s * c * a, 0.0)),
                 np.where(a >= 1.0 - lo, x1 - np.sqrt(np.maximum(2.0 * s * c * (1.0 - a), 0.0)), c * (a - 0.5)))
    return -t


def signed_distance(alpha):
    """Signed distance (px, negative inside) to the 0.5 contour of an antialiased coverage image (h, w) in 0..1.

    The contour passes through the partly covered pixels, where the coverage and the edge's direction (the gradient of the
    coverage) give each pixel's distance to it. Pixels within two of those take it from the best straight edge among them
    (the distance to the line through a neighbour's contour point), which is exact for straight edges wherever the
    neighbour lies along the edge; the others the distance to the nearest of them plus its offset (Gustavson and Strand's
    antialiased distance transform). Straight edges come out within a few hundredths of a pixel, so the contour stays
    straight when zoomed far past the picture's resolution."""
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    h, w = a.shape
    inside = a >= 0.5
    partial = (a > EPS) & (a < 1.0 - EPS)
    exact = ~partial                                             # pixels that say only inside or outside
    step = np.zeros(a.shape, bool)                               # steps between two such pixels (no antialiasing there)
    flip = (inside[:, 1:] != inside[:, :-1]) & exact[:, 1:] & exact[:, :-1]
    step[:, 1:] |= flip
    step[:, :-1] |= flip
    flip = (inside[1:] != inside[:-1]) & exact[1:] & exact[:-1]
    step[1:] |= flip
    step[:-1] |= flip
    edge = partial | step
    if not edge.any():
        return np.full(a.shape, -FAR if inside.any() else FAR, np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3, scale=0.125, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3, scale=0.125, borderType=cv2.BORDER_REPLICATE)
    g = np.hypot(gx, gy)
    ok = g > 1e-6
    nx = np.where(ok, -gx / np.maximum(g, 1e-6), 1.0).astype(np.float32)     # outward: coverage falls toward the outside
    ny = np.where(ok, -gy / np.maximum(g, 1e-6), 0.0).astype(np.float32)
    offset = _edge_offset(a, nx, ny).astype(np.float32)          # signed: positive when the pixel centre is outside
    dist, labels = cv2.distanceTransformWithLabels((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    table = np.zeros(int(labels.max()) + 1, np.float32)
    table[labels[edge]] = offset[edge]
    side = np.where(inside, -1.0, 1.0).astype(np.float32)
    sdf = table[labels] + side * dist
    near = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~edge
    by, bx = np.nonzero(near)
    best = np.full(by.size, np.inf, np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            ey, ex = np.clip(by + dy, 0, h - 1), np.clip(bx + dx, 0, w - 1)
            line = offset[ey, ex] - (nx[ey, ex] * dx + ny[ey, ex] * dy)      # offset + n . (p - e)
            line = np.where(edge[ey, ex] & (by + dy >= 0) & (by + dy < h) & (bx + dx >= 0) & (bx + dx < w), line, np.inf)
            best = np.where(np.abs(line) < np.abs(best), line, best)
    found = np.isfinite(best)
    sdf[by[found], bx[found]] = side[by, bx][found] * np.maximum(np.abs(best[found]), 0.5)
    return sdf.astype(np.float32)


def centroid(alpha):
    """Centre of mass (x, y) of a coverage image in index coordinates; the middle of the frame when it is empty."""
    a = np.asarray(alpha, np.float32)
    total = float(a.sum())
    h, w = a.shape
    if total <= 1e-6:
        return (w - 1) / 2.0, (h - 1) / 2.0
    xs, ys = grid(h, w)
    return float((a * xs).sum() / total), float((a * ys).sum() / total)


def inside_point(sdf, point, depth=0.5):
    """`point` (x, y) when it lies at least `depth` px inside the shape of the field `sdf`, else the nearest pixel that does
    (the deepest pixel when the shape is shallower than that everywhere); None when there is no shape."""
    h, w = sdf.shape
    x = min(max(int(round(point[0])), 0), w - 1)
    y = min(max(int(round(point[1])), 0), h - 1)
    if sdf[y, x] <= -depth:
        return float(point[0]), float(point[1])
    ys, xs = np.nonzero(sdf <= -depth)
    if ys.size == 0:
        deepest = float(sdf.min())
        if deepest >= 0.0:
            return None
        ys, xs = np.nonzero(sdf <= deepest + 1e-3)
    k = int(np.argmin((xs - point[0]) ** 2 + (ys - point[1]) ** 2))
    return float(xs[k]), float(ys[k])


# ================================================================================================= fields
def _catmull_rom(sdf, qx, qy):
    """Catmull-Rom bicubic samples of `sdf` at the 1-D coordinate arrays qx, qy (border repeated). Unlike OpenCV's cubic (its
    kernel is -0.75, which ripples on a ramp) this reproduces linear and quadratic fields exactly, so a straight edge stays
    straight at any zoom."""
    h, w = sdf.shape
    x0, y0 = np.floor(qx).astype(np.int64), np.floor(qy).astype(np.int64)
    fx, fy = (qx - x0).astype(np.float32), (qy - y0).astype(np.float32)

    def weights(f):
        f2, f3 = f * f, f * f * f
        return (0.5 * (-f3 + 2.0 * f2 - f), 0.5 * (3.0 * f3 - 5.0 * f2 + 2.0), 0.5 * (-3.0 * f3 + 4.0 * f2 + f),
                0.5 * (f3 - f2))

    wx, wy = weights(fx), weights(fy)
    out = np.zeros(qx.shape, np.float32)
    for j in range(4):
        yy = np.clip(y0 + (j - 1), 0, h - 1)
        row = np.zeros(qx.shape, np.float32)
        for i in range(4):
            row += wx[i] * sdf[yy, np.clip(x0 + (i - 1), 0, w - 1)]
        out += wy[j] * row
    return out


class AlphaField:
    """A signed distance map (`signed_distance`) sampled at frame index coordinates, the border repeated beyond the picture:
    bilinear everywhere, Catmull-Rom bicubic within `BAND` map pixels of the contour, where a zoomed edge shows every wobble.
    `ratio` is how many map pixels make one frame pixel along a side (the matte may be drawn finer than the frame): the
    field answers in frame pixels."""

    BAND = 3.0

    def __init__(self, sdf, ratio=1.0):
        self.sdf = np.ascontiguousarray(sdf, np.float32)
        self.ratio = float(ratio)

    def sample(self, qx, qy):
        k = self.ratio
        mx = np.ascontiguousarray((qx + 0.5) * k - 0.5, np.float32)          # frame pixel centres -> map pixel centres
        my = np.ascontiguousarray((qy + 0.5) * k - 0.5, np.float32)
        d = cv2.remap(self.sdf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        near = np.abs(d) < self.BAND
        if near.any():
            d[near] = _catmull_rom(self.sdf, mx[near], my[near])
        return d / k


class CloudField:
    """A thought bubble's cloud about its own centre (0, 0): a filled ellipse of half-width `a` and half-height `b` whose rim
    is scalloped by lumps (discs) sitting on it. The distance is exact outside the shape (a union takes the nearest part)."""

    LUMP = 0.34                                                  # lump radius, as a share of the half-height

    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)
        r = self.LUMP * self.b
        ea, eb = max(self.a - r, 1e-3), max(self.b - r, 1e-3)    # the ellipse the lump centres sit on
        perimeter = 2.0 * np.pi * np.sqrt((ea * ea + eb * eb) / 2.0)
        n = max(8, int(round(perimeter / (1.35 * r))))
        th = 2.0 * np.pi * (np.arange(n) + 0.5) / n
        radii = r * (1.0 + 0.10 * np.sin(2.3 * np.arange(n) + 0.4))
        self.core = (ea, eb)
        self.lumps = np.stack([ea * np.cos(th), eb * np.sin(th), radii], axis=1).astype(np.float32)

    def sample(self, qx, qy):
        ea, eb = self.core
        k0 = np.hypot(qx / ea, qy / eb)
        k1 = np.hypot(qx / (ea * ea), qy / (eb * eb))
        d = np.where(k1 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -min(ea, eb))     # first-order distance to the ellipse
        for cx, cy, r in self.lumps:
            d = np.minimum(d, np.hypot(qx - cx, qy - cy) - r)
        return d.astype(np.float32)

    def radius(self):
        """A circle about the centre that holds the whole shape."""
        return float(np.hypot(self.a, self.b))


def disc(size, centre, radius, crop=True):
    """Signed distance (h, w) of a disc (index coordinates), FAR outside its neighbourhood when `crop`."""
    h, w = size[1], size[0]
    xs, ys = grid(h, w)
    if crop:
        x0, x1 = max(0, int(centre[0] - radius - 4)), min(w, int(centre[0] + radius + 6))
        y0, y1 = max(0, int(centre[1] - radius - 4)), min(h, int(centre[1] + radius + 6))
        d = np.full((h, w), FAR, np.float32)
        if x1 > x0 and y1 > y0:
            d[y0:y1, x0:x1] = np.hypot(xs[y0:y1, x0:x1] - centre[0], ys[y0:y1, x0:x1] - centre[1]) - radius
        return d
    return (np.hypot(xs - centre[0], ys - centre[1]) - radius).astype(np.float32)


def _maps(xs, ys, pivot, dest, scale, turn_deg):
    """Where each output pixel (xs, ys) reads the field: turn_deg clockwise and `scale` about `pivot`, the pivot standing
    at `dest` in the output."""
    t = np.radians(turn_deg)
    c, s = float(np.cos(t)), float(np.sin(t))
    dx, dy = xs - dest[0], ys - dest[1]
    return (pivot[0] + (c * dx + s * dy) / scale).astype(np.float32), (pivot[1] + (-s * dx + c * dy) / scale).astype(np.float32)


def placed(field, size, pivot, dest, scale, turn_deg=0.0, crop=None):
    """The distance (px, h x w, positive outside, FAR where nothing was evaluated) of `field` over a `size` = (w, h) frame
    when it is turned `turn_deg` degrees clockwise and scaled by `scale` about its `pivot` (in the field's coordinates) with
    the pivot at `dest` (frame index coordinates). `crop` = (x0, y0, x1, y1) limits the pixels evaluated. Distances scale
    with the shape, so the edge stays one pixel wide at any zoom."""
    w, h = size
    xs, ys = grid(h, w)
    d = np.full((h, w), FAR, np.float32)
    if crop is None:
        d[:] = field.sample(*_maps(xs, ys, pivot, dest, scale, turn_deg)) * scale
        return d
    x0, y0, x1, y1 = (int(v) for v in crop)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        d[y0:y1, x0:x1] = field.sample(*_maps(xs[y0:y1, x0:x1], ys[y0:y1, x0:x1], pivot, dest, scale, turn_deg)) * scale
    return d


def cover_scale(field, size, pivot, dest, turn_deg=0.0, margin=1.5, limit=1.0e4, points=240):
    """The smallest scale at which `field`, turned and placed as in `placed`, covers the whole frame with its edge at least
    `margin` px outside it (so every pixel is solid): found by bisection on a grid of `points` samples across the frame,
    corners included. `inf` when no scale up to `limit` does (the pivot is not inside the shape)."""
    w, h = size
    gx = np.linspace(0.0, w - 1.0, min(points, w)).astype(np.float32)
    gy = np.linspace(0.0, h - 1.0, min(points, h)).astype(np.float32)
    xs, ys = np.meshgrid(gx, gy)

    def covered(scale):
        return float(field.sample(*_maps(xs, ys, pivot, dest, scale, turn_deg)).max()) * scale <= -margin

    if not covered(limit):
        return float("inf")
    lo, hi = 1.0e-3, limit
    for _ in range(40):
        mid = float(np.sqrt(lo * hi))
        lo, hi = (lo, mid) if covered(mid) else (mid, hi)
    return hi


# ================================================================================================= painting
def coverage(d):
    """Antialiased coverage of the shape whose distance field is `d`: 1 inside, 0 outside, a one-pixel ramp on the edge."""
    return np.clip(0.5 - d, 0.0, 1.0)


def inside_band(d, width):
    """Coverage of the shape grown by `width` px."""
    return np.clip(width + 0.5 - d, 0.0, 1.0)


def paint(base, over, cov):
    """`over` (an image or an RGB colour) laid on `base` with the per-pixel coverage `cov` (h, w)."""
    over = np.asarray(over, np.float32)
    return base + (over - base) * cov[..., None].astype(np.float32)


def warp_scaled(img, dest, k):
    """The picture `img` (h, w, 3) scaled by `k` about its own centre, which lands at `dest` (index coordinates), on a frame
    of the same size (the border repeats). Pictures shrunk below half size go through a pyramid first, so the reduction is
    smooth."""
    h, w = img.shape[:2]
    src, kk = img, float(k)
    while kk < 0.5 and min(src.shape[:2]) > 2:
        src = cv2.pyrDown(src)
        kk *= 2.0
    sh, sw = src.shape[:2]
    sx, sy = (sw - 1) / 2.0, (sh - 1) / 2.0
    # the pyramid's pixels are 2^n apart: its centre is the picture's centre, its scale kk lands it at k overall
    m = np.array([[kk, 0.0, dest[0] - kk * sx], [0.0, kk, dest[1] - kk * sy]], np.float64)
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
