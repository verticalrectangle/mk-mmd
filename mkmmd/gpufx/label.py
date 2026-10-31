"""label on the GPU (mkr-post fx/inks.rs): what LabelFx._label_inks lays on the frame (each label's box, its text typed
so far, its dotted leader; set and drawn here, as the CPU sets them), written for mkr-post; the inks and the strength's
mix laid by mkr-post."""
import numpy as np

from ..core import framefx as FX
from .figure import ink_list


def ops(ffx, fx, frame):
    folder = None if ffx.paths is None else ffx.paths.scratch
    if folder is None:
        return None
    inks = ink_list(ffx._label_inks(fx, frame), folder, f"label-{fx['index']}-{frame}")
    return [{"op": "inks", "inks": inks, "strength": float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))}]
