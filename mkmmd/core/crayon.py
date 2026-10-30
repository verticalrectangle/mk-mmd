"""Crayon and paper, numpy only (docs/design.md: Looks, the sketch's crayon; Frame effects, Activity book): the textures
a waxy crayon line and a sheet of paper are drawn with, shared by the vector look's crayon sketch and cut-out paper
(mkmmd.core.shotstyle, mkmmd.core.cutout: inside Blender) and the activity book's tracing and colouring (mkmmd.activity).

A crayon leaves its wax on the paper's tooth: on the tops of the grain and not in its pits, more of them the harder it
is pressed. So a crayon line is grainy, pressed hardest along its middle and lightest at its broken edges, and the paper
shows through it in specks. `tooth` is the paper's grain (fixed by a seed: the sheet does not move under the drawing),
`wax` the share of a pixel a crayon covers at a pressure, `stroke` a whole crayon line from where it runs (a mask) and
`scribble` a region coloured in with back-and-forth strokes, as a child colours. `fibres` is a sheet's look (the mottle
and fibres of construction paper or a workbook's page), `noise` the smooth noise they are all made of and `blur` a box
blur; nothing here needs OpenCV or SciPy."""
from functools import lru_cache

import numpy as np

TOOTH_CELLS = ((1.3, 1.3, 0.55), (1.6, 11.0, 0.3), (6.0, 6.0, 0.15))   # (cell y, cell x px at 1080, weight) of the tooth
TOOTH_SPREAD = 0.2         # the tooth's standard deviation about its middle (0.5)
WAX_SOFT = 0.12            # how gradually the wax takes the tooth as the pressure grows
EDGE_CELL = 2.5            # px at 1080: the grain of a crayon line's broken edge
PRESS_CELL = 60.0          # px at 1080: how far along a line its pressure drifts
SCRIBBLE = 12.0           # px at 1080 between the back-and-forth strokes of a colouring-in


def smoothstep(a, b, x):
    """0 below `a`, 1 above `b`, a smooth S between (`a` and `b` numbers or arrays the shape of `x`)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    u = np.clip((x - a) / np.maximum(b - a, np.float32(1e-6)), 0.0, 1.0)
    return u * u * (np.float32(3.0) - np.float32(2.0) * u)


def noise(rng, shape, cell):
    """(H, W) float32 noise in -1..1 that bends once every `cell` pixels (a number, or (cell y, cell x) for a grain
    stretched one way): random values on a coarse grid, blended with smoothstep weights one axis at a time."""
    H, W = int(shape[0]), int(shape[1])
    cy, cx = (cell, cell) if np.isscalar(cell) else cell
    cy, cx = max(float(cy), 1.0), max(float(cx), 1.0)
    grid = rng.uniform(-1.0, 1.0, (int(H / cy) + 2, int(W / cx) + 2)).astype(np.float32)

    def axis(n, c):
        u = np.arange(n, dtype=np.float64) / c
        i = np.floor(u).astype(np.intp)
        f = (u - i).astype(np.float32)
        return i, f * f * (np.float32(3.0) - np.float32(2.0) * f)
    iy, fy = axis(H, cy)
    ix, fx = axis(W, cx)
    rows = grid[iy] * (1.0 - fy)[:, None] + grid[iy + 1] * fy[:, None]
    return (rows[:, ix] * (1.0 - fx)[None, :] + rows[:, ix + 1] * fx[None, :]).astype(np.float32)


def blur(a, r):
    """`a` (H, W) boxed `r` px each way (2r + 1 wide, twice over: a soft tent), its edges held."""
    r = int(round(r))
    if r < 1:
        return np.asarray(a, np.float32)
    out = np.asarray(a, np.float32)
    for _ in range(2):
        for ax in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=ax, dtype=np.float32)
            n = out.shape[ax]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
            lo = np.take(c, np.arange(0, n), axis=ax)
            out = (hi - lo) / np.float32(2 * r + 1)
    return out


@lru_cache(maxsize=6)
def _tooth(shape, px, seed):
    rng = np.random.default_rng([int(seed), 7007])
    t = sum(w * noise(rng, shape, (cy * px, cx * px)) for cy, cx, w in TOOTH_CELLS)
    t = (t - t.mean()) / max(float(t.std()), 1e-6)
    out = np.clip(np.float32(0.5) + np.float32(TOOTH_SPREAD) * t, 0.0, 1.0).astype(np.float32)
    out.setflags(write=False)
    return out


def tooth(shape, px, seed=1):
    """The paper's tooth, (H, W) float32 in 0..1 about 0.5: a fine grain, a few fibres along the sheet and soft blotches,
    `px` pixels per px at 1080 on the short side; the same sheet for the same `seed` (read-only, kept)."""
    return _tooth((int(shape[0]), int(shape[1])), round(float(px), 3), int(seed))


def wax(press, grain, soft=WAX_SOFT):
    """The share of each pixel a crayon covers pressed `press` (0..1) on paper whose tooth is `grain` (0..1): the wax
    catches the tops of the tooth first, more of them the harder it is pressed, and leaves the pits."""
    return smoothstep(1.0 - press - soft, 1.0 - press + soft, grain)


def stroke(mask, width, px, rng, grain, amount=0.35, pressure=1.0):
    """A crayon line's coverage, (H, W) float32 0..1, where `mask` (H, W bool, the line `width` px wide) runs: pressed
    hardest along its middle (the mask boxed into a ridge), its edge broken by a fine noise, its pressure drifting
    along it (`rng`, a numpy Generator: a new one each time the line is drawn again), and the wax left on the paper's
    tooth `grain` (`tooth`) by `amount` (0: solid wax)."""
    shape = mask.shape
    ridge = blur(mask.astype(np.float32), max(0.3 * width, 1.0))
    edge = ridge + np.float32(0.22) * noise(rng, shape, EDGE_CELL * px)
    inside = smoothstep(0.3, 0.42, edge)
    press = np.float32(pressure) * (np.float32(0.8) + np.float32(0.14) * noise(rng, shape, PRESS_CELL * px))
    press = press * (np.float32(0.45) + np.float32(0.4) * np.minimum(ridge * np.float32(1.4), 1.0))
    covered = wax(press, grain)
    return (inside * (np.float32(1.0 - amount) + np.float32(amount) * covered)).astype(np.float32)


def scribble(region, px, rng, grain, angle, density=0.8, pitch=SCRIBBLE):
    """A region coloured in with back-and-forth crayon strokes, (H, W) float32 0..1 of `region` (H, W bool): strokes
    `pitch` px apart (at 1080) leaning `angle` degrees, wandering a little, pressed `density` (0..1) hard, the paper's
    tooth `grain` showing through."""
    H, W = region.shape
    a = np.radians(float(angle))
    ys, xs = np.arange(H, dtype=np.float32)[:, None], np.arange(W, dtype=np.float32)[None, :]
    per = np.float32(max(pitch * px, 1.0))
    u = (xs * np.float32(np.cos(a)) + ys * np.float32(np.sin(a))) / per + np.float32(0.5) * noise(rng, (H, W), 60.0 * px)
    lines = np.float32(1.0) - np.abs(np.float32(2.0) * (u - np.floor(u)) - np.float32(1.0))     # 1 on a stroke, 0 between
    press = np.float32(density) * (np.float32(0.3) + np.float32(0.7) * smoothstep(0.25, 0.85, lines))
    press = press * (np.float32(0.8) + np.float32(0.2) * noise(rng, (H, W), 2.0 * PRESS_CELL * px))
    return (region * wax(press, grain)).astype(np.float32)


def fibres(shape, px, seed=1, amount=1.0):
    """A sheet of paper's look, (H, W) float32 about 1 to multiply a colour by: a soft mottle, short fibres lying along
    the sheet and a fine grain, `amount` (0..1) of them; the same sheet for the same `seed`."""
    rng = np.random.default_rng([int(seed), 1515])
    mottle = noise(rng, shape, 90.0 * px)
    fib = noise(rng, shape, (1.4 * px, 10.0 * px)) * noise(rng, shape, (30.0 * px, 30.0 * px))
    fine = noise(rng, shape, 1.2 * px)
    return (np.float32(1.0) + np.float32(amount) * (np.float32(0.035) * mottle + np.float32(0.06) * fib
                                                    + np.float32(0.03) * fine)).astype(np.float32)
