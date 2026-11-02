"""spiral on the GPU (mkr-post fx/spiral.rs): the pattern's phase on the frame (mkmmd.core.fields.phase, as
FieldFx._spiral steps it) and its constants worked out here; the pattern and its lay drawn by mkr-post. A drift stays on
the CPU."""
import numpy as np

from ..core import fields as CF
from ..core import framefx as FX
from .fields import BLENDS, drawable, f32, pattern, region


def ops(ffx, fx, frame):
    if not drawable(fx):
        return None
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0)) * fx.get("opacity", 1.0)
    if s <= 0.0:
        return []
    reg = region(ffx, fx, frame)
    if reg is False:
        return None
    q = CF.phase(fx, t, fx["from"], ffx._on(fx, first=False))
    return [{"op": "spiral", "pattern": pattern(fx, q, ffx._at(fx, frame)[0], min(ffx.size)), "blend": BLENDS[fx["blend"]],
             "strength": f32(s), "mask": reg}]
