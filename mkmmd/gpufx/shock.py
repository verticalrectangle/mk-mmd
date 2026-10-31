"""shock on the GPU (mkr-post fx/shock.rs): its centre and waves from FrameFx.shock_at, as mkmmd.framefx.shock uses
them."""
from .. import framefx as FF
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    c, waves, width, amount = ffx.shock_at(fx, frame)
    if not waves:
        return []
    return [{"op": "shock", "c": [f32(c[0]), f32(c[1])], "width": f32(width), "rim": f32(FF.SHOCK_RIM),
             "waves": [[f32(R), f32(amount * s * 2.33), f32(s)] for R, s in waves],
             "glow": f32(fx["glow"]) if fx["glow"] > 0 else 0.0, "colour": f32s(fx["color"])}]
