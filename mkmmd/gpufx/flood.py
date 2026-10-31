"""flood on the GPU (mkr-post fx/flood.rs): the drops on the frame (mkmmd.core.fields.flood_drops, as FieldFx._flood
times them), their noise calls and colours, the centre and the constants of mkmmd.fields.flood worked out here; the
fronts, the colour and the lay drawn by mkr-post."""
import numpy as np

from ..core import fields as CF
from ..core import framefx as FX
from .fields import call, f32, region


def ops(ffx, fx, frame):
    t = ffx._t(frame)
    times = [(FX.frame_of(h, ffx.fps, ffx.frame0) - ffx.frame0) / ffx.fps for h in ffx._hit_times(fx)]
    drops = CF.flood_drops(fx, t, times)
    if not drops:
        return []
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0)) * fx.get("opacity", 1.0)
    if s <= 0.0:
        return []
    reg = region(ffx, fx, frame)
    if reg is False:
        return None
    unit = float(min(ffx.size))
    soft = max(fx["soft"] * unit / 1080.0, 1.0)
    centre = ffx._at(fx, frame)[0]
    out = []
    for k, share, alpha in drops:
        seed = fx["seed"] * 1009 + 31 * k + 17
        out.append({"share": f32(share), "alpha": f32(alpha),
                    "color": np.asarray(FX.pick(fx["color"], k), np.float32).tolist(),
                    "calls": [call(0.0, seed), call(0.0, seed + 1), call(0.0, seed + 2)]})
    return [{"op": "flood", "center": [f32(centre[0]), f32(centre[1])], "unit": f32(unit), "wobble": f32(fx["wobble"]),
             "wob2": f32(2.0 * fx["wobble"]), "soft": [f32(soft), f32(3.0 * soft), f32(0.8 * soft), f32(1.4 * soft)],
             "drops": out, "strength": f32(s), "mask": reg}]
