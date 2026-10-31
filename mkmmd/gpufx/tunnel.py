"""tunnel on the GPU (mkr-post fx/tunnel.rs): the levels from RemapFx.tunnel_at and mkmmd.core.remap.tunnel_levels (a
figure mask: tunnel_queue), each with its box, its warps from the pyramid's levels and level_cover's map, as
mkmmd.remap.tunnel draws them; its mask (gpufx.masked) and strength."""
import numpy as np

from .. import remap as RMF
from ..core import framefx as FX
from ..core import remap as RM
from .masked import masking
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    if ffx.paths is None:
        return None
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    if s <= 0.0:
        return []
    c, args, colour_of, width = ffx.tunnel_at(fx, frame)
    mask = masking(ffx, fx, frame)
    if mask is False:
        return None
    nest = fx["mask"] != "figure"                 # a figure mask remaps the figure alone: a queue of whole copies
    lined = colour_of is not None and width > 0
    w, h = ffx.size
    levels = []
    for k, m, theta, alpha in (RM.tunnel_levels if nest else RM.tunnel_queue)(*args):
        A = RM.level_map(c, m, theta)
        box = (0, 0, w, h) if nest and not levels else RM.level_box((w, h), c, m, theta)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        Ab = np.hstack([A[:, :2], (A[:, :2] @ np.array([x0, y0], float) + A[:, 2])[:, None]])
        lv = {"bx": [x0, y0, x1, y1], "a": A.astype(np.float32).ravel().tolist(), "m": f32(m), "alpha": f32(alpha),
              "lods": [[j, f32(wt), (Ab * 0.5 ** j).astype(np.float32).ravel().tolist()]
                       for j, wt in RMF.lods(1.0 / m)]}
        if lined:
            lv["colour"] = f32s(list(colour_of(k)) + [1.0])
        levels.append(lv)
    op = {"op": "tunnel", "nest": nest, "levels": levels, "width": f32(width) if lined else 0.0,
          "wr": f32(w - 0.5), "hr": f32(h - 0.5)}
    if s < 1.0:
        op["mix"] = s
    if mask is not None:
        op["mask"] = mask
    return [op]
