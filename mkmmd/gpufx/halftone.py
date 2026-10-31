"""halftone on the GPU (mkr-post fx/halftone.rs): FrameFx._halftone's strength and mkmmd.framefx.halftone's constants;
the dots, the plates off register and the mix drawn by mkr-post."""
import numpy as np

from .. import framefx as FF
from ..core import framefx as FX
from .fields import f32


def ops(ffx, fx, frame):
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    if s <= 0.0:
        return []
    cell, px = fx["cell"] * ffx.px, ffx.px
    a = np.radians(fx["angle"])
    offset = (fx["offset"][0] * px, fx["offset"][1] * px)
    return [{"op": "halftone", "k": [f32(fx["level"]), f32(np.cos(a)), f32(np.sin(a)), f32(cell), f32(FF.DOT * cell),
                                     f32(fx["lift"])],
             "ink": np.asarray(fx["ink"], np.float32).tolist(),
             "offset": [offset[0], offset[1]] if offset[0] or offset[1] else None, "s": [f32(s), f32(1.0 - s)]}]
