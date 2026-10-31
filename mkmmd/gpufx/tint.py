"""tint on the GPU (mkr-post fx/tint.rs): FrameFx._tint's strength and colours; the gradient map and its mix drawn by
mkr-post."""
import numpy as np

from ..core import framefx as FX
from .fields import f32


def ops(ffx, fx, frame):
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    if s <= 0.0:
        return []
    c = lambda v: np.asarray(v, np.float32).tolist()      # noqa: E731
    return [{"op": "tint", "dark": c(fx["dark"]), "color": c(fx["color"]), "light": c(fx["light"]),
             "s": [f32(s), f32(1.0 - s)]}]
