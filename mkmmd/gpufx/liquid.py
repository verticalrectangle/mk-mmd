"""liquid on the GPU (mkr-post fx/liquid.rs): mkmmd.fields.liquid's sizes, the dishes' turns and balls at the frame's
second, the bubbles (mkmmd.fields.bubbles), the noise calls and every constant worked out here as liquid() works them
out (float32 where it uses float32); the fields, the oil, the cells, the bubbles and the lay drawn by mkr-post."""
import math

import numpy as np

from .. import fields as FL
from ..core import fields as CF
from ..core import framefx as FX
from .fields import BLENDS, call, f32, region


def ops(ffx, fx, frame):
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0)) * fx.get("opacity", 1.0)
    if s <= 0.0:
        return []
    reg = region(ffx, fx, frame)
    if reg is False:
        return None
    swell = CF.swell(t, ffx._on(fx), fx["decay"])
    centre = ffx._at(fx, frame)[0]
    w, h = ffx.size
    unit = float(min(w, h))
    div = max(1, int(math.ceil(unit / FL.RENDER - 1e-9)))
    rs = (-(-w // div), -(-h // div))
    runit, rc = unit / div, ((centre[0] + 0.5) / div - 0.5, (centre[1] + 0.5) / div - 0.5)
    rpx = runit / 1080.0
    f = max(1, int(round(runit / FL.WORK)))
    sp, sc = fx["speed"], fx["scale"]
    lw = 0.6 * sc
    dishes = FL._dishes(fx)
    calls = [call(0.08 * sp * t, fx["seed"] * 7 + 3)]

    def turn(om, th0, pivot):
        a = om * sp * t + th0
        return [f32(math.cos(a)), f32(math.sin(a)), f32(pivot[0]), f32(pivot[1])]
    dyes, seed = [], fx["seed"] * 211
    for i, (ink, om, th0, pivot, _) in enumerate(dishes[:-1]):
        dyes.append({"turn": turn(om, th0, pivot), "ink": np.asarray(ink, np.float32).tolist()})
        calls += [call(0.06 * sp * t + 17.0 * i, seed + 5 * i + q) for q in range(5)]
    oils = []
    for i, (ink, om, th0, pivot, balls) in enumerate([dishes[-1]] + dishes[:-1]):
        a = om * sp * t + th0
        at = [FL._ball_at(b, a, pivot, t, sp) for b in balls]
        oils.append({"turn": turn(om, th0, pivot),
                     "unink": None if ink is None else [f32(1.0 - c) for c in ink],
                     "balls": [[f32(x), f32(y), f32(r * r)] for x, y, r in at]})
        calls.append(call(0.15 * sp * t + 31.0 * i, fx["seed"] * 101 + i))
    cells = None
    if fx["cells"] > 0:
        size_u = 0.06 * sc
        calls.append(call(0.05 * sp * t, fx["seed"] * 13 + 6))
        cells = {"seed": int(fx["seed"] * 13 + 5),
                 "k": [f32(size_u), f32(0.5 * size_u), f32(0.45 * sc), f32(FL.CELL_CUT[0] - FL.CELL_CUT[1] * fx["cells"]),
                       f32(1.5 * rpx), f32(size_u * runit * 0.22), f32(2.4 * rpx), f32(1.4 * rpx)]}
    bubbles = []
    if fx["bubbles"] > 0:
        for (bx, by, rad), box in FL.bubbles(fx, t, centre, unit, dishes[0], (w, h)):
            bubbles.append({"c": [f32(bx), f32(by), f32(rad), f32(max(0.2 * rad, 1.2 * unit / 1080.0))],
                            "g": [f32(bx - 0.38 * rad), f32(by - 0.38 * rad)], "box": [int(v) for v in box]})
    px = unit / 1080.0
    return [{"op": "liquid", "work": [-(-rs[0] // f), -(-rs[1] // f)], "render": [int(rs[0]), int(rs[1])], "f": f,
             "div": div, "rc": [f32(rc[0]), f32(rc[1])], "runit": f32(runit), "z": f32(1.0 + fx["pulse"] * swell),
             "lw": f32(lw), "lc": f32(0.5 * sc), "amp": f32(FL.SWIRL * sc * lw), "gdx": f32(f / runit),
             "level": f32(FL.LEVEL - FL.SWELL * swell), "flare": f32(0.55 * (1.0 + 0.7 * swell)),
             "aa": f32(max(1.1 * rpx, 0.7)), "rpx": [f32(1.7 * rpx), f32(4.0 * rpx), f32(2.8 * rpx), f32(0.06 * runit)],
             "rough": f32(0.25 * sc), "marble": f32(FL.MARBLE), "light": np.asarray(fx["light"], np.float32).tolist(),
             "dyes": dyes, "dishes": oils, "cells": cells, "bubbles": bubbles,
             "bub": [f32(1.4 * px), f32(1.8 * px), f32(1.5 * px), f32(1.6 * px)], "calls": calls,
             "blend": BLENDS[fx.get("blend", "over")], "strength": f32(s), "mask": reg}]
