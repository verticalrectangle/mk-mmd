"""shapes on the GPU (mkr-post fx/shapes.rs): the layout's cells worked out as mkmmd.shapes.picture works them out (the
lattice, the deal, the turn, the pop, which shape and ink each cell wears; and the layout before, which shows where a
new cell's shape has not begun to pop in), each pixel's cell and place in it as the CPU path finds them in closed form,
all written once for the frame as float32 tables; mkr-post averages the picture under each cell (the box blur sampled
at its middle), picks its colour, skips it near a flat back, and draws the shapes' signed distances and the mix."""
import math
import os

import numpy as np

from ..core import framefx as FX
from ..core import shapes as SH
from .numbers import f32, f32s

KINDS = {"circle": 0, "square": 1, "diamond": 2, "triangle": 3}      # the codes fx_shapes.wgsl draws by
NEAREST, OWN = -1.0, -2.0                                            # a cell's ink: the nearest of the inks, or none


def tables(size, cell, layout, shapes, size_k, turn, spin, inks, pick, pop, stagger, seed, step, age, t):
    """One layout's tables (float32, one flat array) and what mkr-post needs to read them: (table, cells, columns, blur
    side), as mkmmd.shapes._layer works it out (`age` None: every shape at its size, begun). Per cell (row by row,
    columns left to right): 8 numbers (the pixel its colour is sampled at, its shape's code, its ink, the cosine and
    sine of its turn, its radius, 1 when its shape has begun to pop in); per picture row: (the row's first cell, the
    pixel's offset down from the cell's middle, 1 in an offset row, 0); per picture column, for rows without and with
    the offset: (the cell's column, the pixel's offset along from its middle)."""
    W, H = size
    dx, dy = SH.lattice((W, H), cell, layout)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r_lo, r_hi = int(math.floor(-cy / dy)) - 1, int(math.ceil((H - 1 - cy) / dy)) + 1
    q_lo, q_hi = int(math.floor(-cx / dx)) - 2, int(math.ceil((W - 1 - cx) / dx)) + 2
    rows, cols = np.arange(r_lo, r_hi + 1), np.arange(q_lo, q_hi + 1)
    hexed = layout == "hex"
    off = np.where(hexed & ((rows & 1) == 1), dx / 2.0, 0.0)
    xc = cx + cols[None, :] * dx + off[:, None]
    yc = np.broadcast_to(cy + rows[:, None] * dy, xc.shape)
    k = max(int(round(0.8 * min(dx, dy))), 1)
    d = SH.deal(cols[None, :], rows[:, None], step, seed, 4)
    which = np.minimum((d[..., 0] * len(shapes)).astype(np.intp), len(shapes) - 1)
    turned = np.radians((d[..., 1] * 2.0 - 1.0) * turn + spin * t)
    if age is None or pop <= 0:
        grow, begun = np.ones(which.shape), np.ones(which.shape, bool)
    else:
        since = age - d[..., 2] * stagger
        grow, begun = SH.pop(since / pop), since > 0.0
    rad = np.minimum(0.5 * min(dx, dy) * size_k * grow, 0.5 * min(dx, dy))
    a = turned.astype(np.float32)
    ink = np.full(which.shape, OWN)
    if inks is not None:
        if pick == "nearest":
            ink[:] = NEAREST
        elif pick == "shape":
            ink = (which % len(inks)).astype(np.float64)
        else:
            ink = np.minimum((d[..., 3] * len(inks)).astype(np.intp), len(inks) - 1).astype(np.float64)
    ncells = which.size
    cells = np.zeros((ncells, 8), np.float32)
    cells[:, 0] = np.clip(np.rint(xc), 0, W - 1).ravel()
    cells[:, 1] = np.clip(np.rint(yc), 0, H - 1).ravel()
    cells[:, 2] = np.array([KINDS[s] for s in shapes], np.float32)[which].ravel()
    cells[:, 3] = ink.ravel()
    cells[:, 4] = np.cos(a).ravel()
    cells[:, 5] = np.sin(a).ravel()
    cells[:, 6] = rad.astype(np.float32).ravel()
    cells[:, 7] = begun.ravel()
    ys, xs = np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64)
    rr = np.rint((ys - cy) / dy).astype(np.intp)
    shift = off[rr - r_lo]
    rowtab = np.zeros((H, 4), np.float32)
    rowtab[:, 0] = (rr - r_lo) * len(cols)
    rowtab[:, 1] = (ys - cy - rr * dy).astype(np.float32)
    rowtab[:, 2] = shift != 0.0
    coltab = np.zeros((2, W, 2), np.float32)
    for i, sh in enumerate((0.0, dx / 2.0 if hexed else 0.0)):
        qq = np.rint((xs - cx - sh) / dx).astype(np.intp)
        coltab[i, :, 0] = qq - q_lo
        coltab[i, :, 1] = (xs - cx - sh - qq * dx).astype(np.float32)
    return np.concatenate([cells.ravel(), rowtab.ravel(), coltab.ravel()]), ncells, len(cols), k


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None or paths.scratch is None:
        return None
    t = ffx._t(frame)
    if FX.strength(fx, t) <= 0.0:
        return []
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    mask = None
    if fx["mask"] is not None:
        fig = paths.figure(FX.figure_key(fx), frame)
        if not paths.fits(fig):
            return None
        mask = {"figure": str(fig), "inside": fx["mask"] == "figure"}
    cell, step, age, clip_t, before = ffx.shapes_at(fx, frame)
    layers, parts = [], []
    for c, st, a in [(cell, step, age)] + ([] if before is None else [(before[0], before[1], None)]):
        table, ncells, ncols, k = tables(ffx.size, c, fx["layout"], fx["shape"], fx["size"], fx["turn"], fx["spin"],
                                         fx["inks"], fx["pick"], fx["pop"], fx["stagger"], fx["seed"], st, a, clip_t)
        parts.append(table)
        layers.append({"cells": ncells, "columns": ncols, "blur": k})
    if fx["inks"] is not None:
        parts.append(np.pad(np.asarray(fx["inks"], np.float32), ((0, 0), (0, 1))).ravel())
    table = np.concatenate(parts)
    path = os.path.join(paths.scratch, f"shapes-{os.getpid()}-{fx['index']}-{frame}.f32")
    table.tofile(path)
    return [{"op": "shapes", "table": path, "count": int(table.size), "layers": layers,
             "inks": 0 if fx["inks"] is None else len(fx["inks"]),
             "back": None if fx["back"] is None else f32s(fx["back"]),
             "skip": None if fx["back"] is None else f32(fx["skip"]),
             "mask": mask, "s": [f32(s), f32(1.0 - s)]}]
