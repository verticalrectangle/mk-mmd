"""Memphis-style print of the bedroom's quilt and rug: pure numpy, no bpy, importable from tests.

The print is a lattice of cells; every cell holds one motif (or nothing). A motif is a signed-distance function in the
cell's own rotated frame, scaled by `R = MOTIF_R * min(cell size) * scale`:

  1 triangle   equilateral, pointing +y, circumradius R
  2 dots       three discs of different sizes
  3 ring       a circle outline
  4 squiggle   a sine wave stroke with round ends
  5 zigzag     a triangle wave stroke with round ends
  6 stripes    three bold bars cut to a square patch
  7 arch       a thick half ring (a rainbow)
  8 cross      a plus sign (an X once rotated)

`layout()` draws the cells with a seeded generator (no neighbours with the same motif or colour). `lut_stops()` packs
the cells into colour-ramp stops (constant interpolation = a lookup table of up to 32 entries, which is how the shader
reads them); the shader (`bedroom_soft_nodes.memphis`) evaluates the same functions on the GPU, `render()` evaluates them
here for tests and for judging a layout without Blender. Colours are only *indices* into a palette list; the builders
decide which palette slots they are."""
import math
import random

import numpy as np

TYPES = ("none", "triangle", "dots", "ring", "squiggle", "zigzag", "stripes", "arch", "cross")
NONE, TRIANGLE, DOTS, RING, SQUIGGLE, ZIGZAG, STRIPES, ARCH, CROSS = range(9)
MOTIF_R = 0.44                      # motif radius R as a fraction of the smaller cell side (scale 1)
SCALE_LO, SCALE_HI = 0.5, 1.5       # the scale range the lookup table can hold
LUT_SIZE = 32                       # entries of one colour-ramp table
MAX_COLORS = 8
TAU = 2.0 * math.pi
S3 = 0.8660254037844386

# ================================================================================================== motif distances
# Motif dimensions, in units of the motif radius R (every motif fits in a circle of radius R about its centre).
TRI_IN = 0.5                                       # triangle: inradius (circumradius = R)
DOT_SET = ((-0.42, -0.26, 0.33), (0.45, -0.12, 0.24), (0.02, 0.52, 0.17))        # dots: x, y, radius
RING_R, RING_W = 0.60, 0.18                        # ring: radius of the centre line, half width
SQ_A, SQ_P, SQ_W, SQ_L = 0.22, 0.76, 0.15, 0.90    # squiggle: amplitude, wavelength, stroke half width, half length
ZZ_A, ZZ_P, ZZ_W, ZZ_L = 0.19, 0.50, 0.14, 0.90    # zigzag: the same
ST_PITCH, ST_W, ST_BOX = 0.50, 0.15, 0.66          # stripes: bar pitch, bar half width, half side of the square patch
ARCH_R, ARCH_W, ARCH_SHIFT = 0.50, 0.20, 0.30      # arch: radius of the centre line, half width, centre shift
CROSS_L, CROSS_W = 0.72, 0.19                      # cross: arm half length, arm half width


def _hyp(a, b):
    return np.sqrt(a * a + b * b)


def _stroke(x, y, yc, g, Lh, w):
    """Distance to a stroke of half width w along a curve y = yc(x) (distance scaled by g across the slope), half length
    Lh, round ends."""
    q1, q2 = np.abs(x) - (Lh - w), np.abs(y - yc) * g
    return _hyp(np.maximum(q1, 0.0), np.maximum(q2, 0.0)) + np.minimum(np.maximum(q1, q2), 0.0) - w


def sdf(kind, x, y, R):
    """Signed distance (metres, negative inside) of motif `kind` at (x, y) in its own frame, for motif radius R."""
    if kind == TRIANGLE:
        return np.maximum(np.maximum(-y, S3 * x + 0.5 * y), -S3 * x + 0.5 * y) - TRI_IN * R
    if kind == DOTS:
        return np.minimum(np.minimum(*[_hyp(x - cx * R, y - cy * R) - r * R for cx, cy, r in DOT_SET[:2]]),
                          _hyp(x - DOT_SET[2][0] * R, y - DOT_SET[2][1] * R) - DOT_SET[2][2] * R)
    if kind == RING:
        return np.abs(_hyp(x, y) - RING_R * R) - RING_W * R
    if kind == SQUIGGLE:
        k, A = TAU / (SQ_P * R), SQ_A * R
        g = 1.0 / np.sqrt(1.0 + (A * k * np.cos(k * x)) ** 2)
        return _stroke(x, y, A * np.sin(k * x), g, SQ_L * R, SQ_W * R)
    if kind == ZIGZAG:
        P, A = ZZ_P * R, ZZ_A * R
        tri = 4.0 * np.abs((x / P - np.floor(x / P)) - 0.5) - 1.0
        g = 1.0 / np.sqrt(1.0 + (4.0 * ZZ_A / ZZ_P) ** 2)
        return _stroke(x, y, A * tri, g, ZZ_L * R, ZZ_W * R)
    if kind == STRIPES:
        pitch = ST_PITCH * R
        bars = np.abs((y / pitch + 0.5) - np.floor(y / pitch + 0.5) - 0.5) * pitch - ST_W * R
        return np.maximum(bars, np.maximum(np.abs(x), np.abs(y)) - ST_BOX * R)
    if kind == ARCH:
        yy = y + ARCH_SHIFT * R
        return np.maximum(np.abs(_hyp(x, yy) - ARCH_R * R) - ARCH_W * R, -yy)
    if kind == CROSS:
        return np.minimum(np.maximum(np.abs(x) - CROSS_L * R, np.abs(y) - CROSS_W * R),
                          np.maximum(np.abs(x) - CROSS_W * R, np.abs(y) - CROSS_L * R))
    return np.full(np.broadcast(x, y).shape, 1e3)


# ================================================================================================== layout
class Cell:
    """One lattice cell: motif type, rotation (rad), scale, jitter (fraction of the cell, from its centre) and the colour
    indices of the motif (c1) and of its second colour (c2)."""

    def __init__(self, kind=NONE, rot=0.0, scale=1.0, jx=0.0, jy=0.0, c1=0, c2=0):
        self.kind, self.rot, self.scale, self.jx, self.jy, self.c1, self.c2 = kind, rot, scale, jx, jy, c1, c2

    def __repr__(self):
        return f"Cell({TYPES[self.kind]}, c1={self.c1}, rot={math.degrees(self.rot):.0f})"


def _pick(rng, items, banned=()):
    """Weighted choice (item, weight) skipping the banned items (falls back to the full list when all are banned)."""
    pool = [(i, w) for i, w in items if i not in banned] or list(items)
    tot = sum(w for _, w in pool)
    r = rng.random() * tot
    for i, w in pool:
        r -= w
        if r <= 0:
            return i
    return pool[-1][0]


def layout(nx, ny, seed, weights, pools, empty=0.08, jitter=0.05, scale=(0.80, 1.0), n_colors=6):
    """Cells (row-major: index = cx + nx * cy) with a seeded generator. `weights`: [(type, weight)]; `pools`: {type:
    [(colour index, weight)]}. No cell shares its motif type or its colour with its left or lower neighbour."""
    rng = random.Random(seed)
    cells = [None] * (nx * ny)
    for cy in range(ny):
        for cx in range(nx):
            left = cells[cx - 1 + nx * cy] if cx else None
            below = cells[cx + nx * (cy - 1)] if cy else None
            nb = [c for c in (left, below) if c is not None]
            c = Cell()
            if rng.random() < empty and not any(n.kind == NONE for n in nb):
                cells[cx + nx * cy] = c
                rng.random(), rng.random(), rng.random(), rng.random(), rng.random()
                continue
            c.kind = _pick(rng, [(t, w) for t, w in weights if t != NONE], {n.kind for n in nb})
            c.rot = rng.random() * TAU
            c.scale = scale[0] + (scale[1] - scale[0]) * rng.random()
            c.jx, c.jy = (rng.random() - 0.5) * 2 * jitter, (rng.random() - 0.5) * 2 * jitter
            c.c1 = _pick(rng, pools[c.kind], {n.c1 for n in nb if n.kind != NONE})
            c.c2 = _pick(rng, pools[c.kind], {c.c1})
            cells[cx + nx * cy] = c
    return cells


def lut_stops(cells, n=LUT_SIZE):
    """Colour-ramp stops of a table for up to `n` cells: (A, B) with stops [(position, (r, g, b, a))]. Stop i sits at
    i / n (constant interpolation: read it at (i + 0.5) / n).
      A = (type / 8, rotation / 2pi, scale - 0.5, 0)       B = (jx + 0.5, jy + 0.5, c1 / 7, c2 / 7)"""
    A, B = [], []
    for i in range(n):
        c = cells[i] if i < len(cells) else Cell()
        A.append((i / n, (c.kind / 8.0, (c.rot % TAU) / TAU, min(max(c.scale - SCALE_LO, 0.0), 1.0), 0.0)))
        B.append((i / n, (min(max(c.jx + 0.5, 0.0), 1.0), min(max(c.jy + 0.5, 0.0), 1.0), c.c1 / 7.0, c.c2 / 7.0)))
    return A, B


def decode(a, b):
    """Inverse of lut_stops for one entry -> Cell (what the shader does with a table read)."""
    return Cell(int(round(a[0] * 8.0)), a[1] * TAU, a[2] + SCALE_LO, b[0] - 0.5, b[1] - 0.5, int(round(b[2] * 7.0)),
                int(round(b[3] * 7.0)))


# ================================================================================================== evaluation
def cell_at(u, v, nx, ny, sx, sy):
    """Cell index and the position relative to the cell centre for flat coordinates (u, v) (metres from the lattice
    corner); points outside the lattice read the nearest cell."""
    cx = np.clip(np.floor(u / sx), 0, nx - 1)
    cy = np.clip(np.floor(v / sy), 0, ny - 1)
    return (cx + nx * cy).astype(int), u - (cx + 0.5) * sx, v - (cy + 0.5) * sy


def motif_distance(cells, nx, ny, sx, sy, u, v):
    """Signed distance of each point to its cell's motif plus the cell index (the reference of the shader)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    idx, px, py = cell_at(u, v, nx, ny, sx, sy)
    kind = np.array([c.kind for c in cells])[idx]
    rot = np.array([c.rot for c in cells])[idx]
    scale = np.array([c.scale for c in cells])[idx]
    px = px - np.array([c.jx for c in cells])[idx] * sx
    py = py - np.array([c.jy for c in cells])[idx] * sy
    cs, sn = np.cos(rot), np.sin(rot)
    x, y = cs * px + sn * py, -sn * px + cs * py
    R = MOTIF_R * min(sx, sy) * scale
    d = np.full(u.shape, 1e3)
    for k in range(1, 9):
        m = kind == k
        if m.any():
            d[m] = sdf(k, x[m], y[m], R[m])
    return d, idx


def render(cells, nx, ny, sx, sy, colors, ground, res=600, aa=0.0025):
    """RGB float image (v up) of the lattice over a ground colour, for tests and previews. `colors`: list of RGB."""
    W, H = nx * sx, ny * sy
    w = res
    h = int(round(res * H / W))
    uu, vv = np.meshgrid((np.arange(w) + 0.5) / w * W, (np.arange(h)[::-1] + 0.5) / h * H)
    d, idx = motif_distance(cells, nx, ny, sx, sy, uu, vv)
    m = np.clip(0.5 - d / (2 * aa), 0.0, 1.0)[..., None]
    col = np.array([colors[c.c1] for c in cells])[idx]
    return np.asarray(ground, float) * (1 - m) + col * m
