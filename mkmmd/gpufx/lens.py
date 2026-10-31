"""lens on the GPU (mkr-post fx/lens.rs): its centre and Einstein radius from FrameFx.lens_at, as mkmmd.framefx.lens
uses them."""
from .numbers import below, f32


def ops(ffx, fx, frame):
    c, theta = ffx.lens_at(fx, frame)
    if theta <= 0:
        return []
    op = {"op": "lens", "c": [f32(c[0]), f32(c[1])], "theta2": f32(theta * theta)}
    if fx["shadow"] > 0:
        op["shadow"] = below(float((fx["shadow"] * theta) ** 2))
    return [op]
