"""push on the GPU (mkr-post fx/push.rs): its zoom from FrameFx.push_at and mkmmd.framefx.push_terms, the frame inside
from the cut (a FileNotFoundError when it is not there, as the CPU path raises). The frame inside is the cut's as it
reached the push at its own time: where an effect before the push in the table acts on it, it is no file of the cut
and the CPU draws the push."""
from .. import framefx as FF
from .numbers import f32


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None:
        return None
    c, r0, g, u = ffx.push_at(fx, frame)
    if any(e["index"] < fx["index"] for e in ffx.entries(g)):
        return None
    inner = paths.frame(g)
    if inner is None:
        raise FileNotFoundError(f"fx {fx['index']} (push): the frame inside, {fx['inside_frame']}, is not there")
    if not paths.fits(inner):
        return None
    z, p, m, r, s = FF.push_terms(ffx.size, c, r0, u, fx["ease"])
    return [{"op": "push", "inner": str(inner), "c": [f32(c[0]), f32(c[1])], "p": [f32(p[0]), f32(p[1])],
             "m": [f32(m[0]), f32(m[1])], "z": f32(z), "s": f32(s), "rr": f32(r + 0.5)}]
