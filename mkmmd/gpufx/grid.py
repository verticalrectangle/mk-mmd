"""grid on the GPU (mkr-post fx/grid.rs): the boxes on the frame (mkmmd.core.framefx.grid_cells, as FrameFx._grid
steps them), their backs and fills (a silkscreen's print: its number of inks, softness, lines and key offset; its inks
by box), the frame's shrunk size and the tile's padding worked out here as mkmmd.framefx.cells works them out; the
figure's tile, its silkscreen plates (fx/silk.rs) and every box drawn by mkr-post. A pattern back stays on the CPU."""
import math

import numpy as np

from ..core import framefx as FX
from . import figure as GF


def ops(ffx, fx, frame):
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    times = ffx._hit_times(fx)
    k = FX.step(fx, frame, times, ffx.fps, ffx.frame0)
    boxes = FX.grid_cells(fx, frame, times, ffx.fps, ffx.frame0)
    piece = None if fx["piece"] is None else (FX.piece_cell(fx, k), fx["piece"])
    backs, fills, silks = [], [], []                     # silks: each box's own fill when it is a silkscreen (cells)
    for _, (i, j) in boxes:
        b = FX.pick(fx["back"], k + i + j)
        own = ffx._fill(fx, k + i + j)
        if isinstance(b, dict):
            return None
        backs.append(b)
        fills.append(piece[1] if piece is not None and tuple(piece[0]) == (i, j) else own)
        if isinstance(own, dict):
            silks.append(own)
    prints = {}                                          # (inks, key offset) -> its index in the op's silks
    for f in silks:
        prints.setdefault((len(f["inks"]), (float(f["offset"][0]), float(f["offset"][1]))), len(prints))
    figure = GF.cutout(ffx, fx, frame, picture=bool(silks) or any(f is None for f in fills))
    if figure is None:
        return None
    w, h = ffx.size
    edge, shadow = ffx._edge(fx), ffx._shadow(fx)
    cells, sw, sh, pad, sc = [], w, h, 2, 1.0
    if boxes:
        x0, y0, x1, y1 = boxes[0][0]
        sc = min(x1 - x0, y1 - y0)
        sw, sh = max(int(round(w * sc)), 1), max(int(round(h * sc)), 1)
        ew = 0.0 if edge is None else edge[1]
        reach = 0.0 if shadow is None else math.hypot(*shadow[1]) * sc + 3.0 * shadow[2] * sc
        pad = int(math.ceil(ew + reach + max([math.hypot(*f["offset"]) for f in silks] or [0.0]))) + 2
    for ((bx0, by0, bx1, by1), _), b, f in zip(boxes, backs, fills):
        X0, Y0, X1, Y1 = int(round(bx0 * w)), int(round(by0 * h)), int(round(bx1 * w)), int(round(by1 * h))
        if X1 <= X0 or Y1 <= Y0:
            continue
        at = (int(round((bx0 + bx1) * w / 2.0 - sw / 2.0)) - X0, int(round((by0 + by1) * h / 2.0 - sh / 2.0)) - Y0)
        if b is None:                                    # the frame shrunk, reflected past its edges (_framed)
            bw, bh = X1 - X0, Y1 - Y0
            px, py = max(-at[0], 0), max(-at[1], 0)
            pw, ph = min(sw, bw - at[0]) - px, min(sh, bh - at[1]) - py
            if pw <= 0 or ph <= 0:
                return None
            back = {"picture": {"part": [px, py, pw, ph], "at": [X0 + at[0] + px, Y0 + at[1] + py]}}
        else:
            back = {"color": GF.colour(b)}
        cell = {"box": [X0, Y0, X1, Y1], "back": back, "at": [X0 + at[0], Y0 + at[1]], "fill": None}
        if isinstance(f, dict):
            cell["silk"] = {"band": prints[len(f["inks"]), (float(f["offset"][0]), float(f["offset"][1]))],
                            "inks": [GF.colour(c) for c in f["inks"]]}
        else:
            cell["fill"] = GF.colour(f)
        cells.append(cell)
    sh_ = None
    if shadow is not None:
        sh_ = {"color": GF.colour(shadow[0]), "offset": [shadow[1][0] * sc, shadow[1][1] * sc], "blur": shadow[2] * sc,
               "alpha": float(shadow[3])}
    bands = [None] * len(prints)
    for f in silks:
        n, off = len(f["inks"]), (float(f["offset"][0]), float(f["offset"][1]))
        bands[prints[n, off]] = {"n": n, "soft": float(f["soft"]), "line": float(f["line"]), "offset": list(off)}
    return [{"op": "grid", "figure": figure, "small": [sw, sh], "pad": pad, "cells": cells, "silks": bands,
             "edge": GF.edge(ffx, fx), "shadow": sh_, "strength": s}]
