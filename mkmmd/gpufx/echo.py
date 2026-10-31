"""echo on the GPU (mkr-post fx/echo.rs): the copies' frames and opacities from FrameFx.echo_at, the frames the cut has
(the CPU path leaves out the others too)."""
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None:
        return None
    copies = []
    for f, a in ffx.echo_at(fx, frame):
        p = paths.frame(f) if f >= ffx.frame0 else None
        if p is None:
            continue
        if not paths.fits(p):
            return None
        if a > 0:
            copies.append([str(p), f32(a)])
    if not copies:
        return []
    return [{"op": "echo", "copies": copies, "lighten": fx["blend"] == "lighten",
             "tint": None if fx["tint"] is None else f32s(fx["tint"])}]
