"""kaleido on the GPU (mkr-post fx/kaleido.rs): the wedges, turns, zoom and pyramid levels from RemapFx.kaleido_at and
mkmmd.core.remap.kaleido_coords' constants, the half map's size (mkmmd.remap.half_map), the seams' across copies
(mkmmd.remap.seam_blend), its mask (gpufx.masked) and strength, as the CPU path uses them."""
import math

import numpy as np

from .. import remap as RMF
from ..core import framefx as FX
from .masked import masking
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    if ffx.paths is None:
        return None
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    if s <= 0.0:
        return []
    c, sc, zoom, turn, spin, radius = ffx.kaleido_at(fx, frame)
    w, h = ffx.size
    n = fx["segments"]
    a = 2.0 * np.pi / n
    axis = fx["source"]["angle"]
    op = {"op": "kaleido", "mirror": fx["mirror"], "c": f32s(c), "s": f32s(sc),
          "off1": f32(math.radians(axis + spin) - a / 2.0), "off2": f32(math.radians(axis + turn) - a / 2.0),
          "a": f32(a), "a2": f32(2.0 * a), "inv_zoom": f32(1.0 / zoom),
          "radius": None if radius is None else f32(radius),
          "lods": [[j, f32(wt)] for j, wt in RMF.lods(1.0 / zoom)]}
    if fx["mirror"]:
        hw, hh = (w + 1) // 2, (h + 1) // 2
        op.update(half=[hw, hh], kx=f32(w / hw), ky=f32(h / hh))
    else:
        wedge = 360.0 / n
        op.update(half_a=f32(a / 2.0), pi2=f32(np.pi / 2.0),
                  off2p=f32(math.radians(axis + (turn + 1.0 * wedge)) - a / 2.0),
                  off2m=f32(math.radians(axis + (turn + -1.0 * wedge)) - a / 2.0))
    if s < 1.0:
        op["mix"] = s
    mask = masking(ffx, fx, frame)
    if mask is False:
        return None
    if mask is not None:
        op["mask"] = mask
    return [op]
