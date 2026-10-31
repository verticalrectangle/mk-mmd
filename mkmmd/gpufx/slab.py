"""slab on the GPU (mkr-post fx/slab.rs): where it stands over its shutter from FrameFx.slab_at, each place's box from
mkmmd.framefx.slab_frame, as slab_cover sums them."""
from .. import framefx as FF
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    centres, sw, sh, alpha = ffx.slab_at(fx, frame)
    (ca, sa, cs, t), places = FF.slab_frame(ffx.size, centres, sw, sh, fx["slant"], fx["angle"])
    if alpha == 0 or all(x1 <= x0 or y1 <= y0 for _, _, (x0, x1, y0, y1) in places):
        return []                                        # nothing covered: the picture as it is
    return [{"op": "slab", "places": [[[f32(cx), f32(cy)], list(box)] for cx, cy, box in places],
             "ca": float(ca), "sa": float(sa), "cs": float(cs), "t": f32(t), "hh": f32(sh / 2.0), "hw": f32(sw / 2.0),
             "n": max(len(centres), 1), "alpha": f32(alpha), "colour": f32s(fx["color"])}]
