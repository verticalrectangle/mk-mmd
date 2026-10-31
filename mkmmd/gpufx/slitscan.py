"""slitscan on the GPU (mkr-post fx/slitscan.rs): each column's (or row's) age from mkmmd.framefx.slit_ages, handed
over as bands of columns that show one frame: the frame of that age, or the youngest older one the cut has before it,
as mkmmd.framefx.slitscan stands it in (columns with none keep the picture)."""
import numpy as np

from .. import framefx as FF


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None:
        return None
    depth = ffx.slit_depth(fx, frame)
    if depth < 1:
        return []
    w, h = ffx.size
    across = fx["now"] in ("left", "right")
    ages = FF.slit_ages(w if across else h, depth, reverse=fx["now"] in ("left", "top"))
    bands, last = [], None
    for k in range(1, int(ages.max()) + 1 if len(ages) else 1):
        g = frame - k
        p = paths.frame(g) if g >= ffx.frame0 else None
        if p is not None:
            if not paths.fits(p):
                return None
            last = p
        sel = np.flatnonzero(ages == k)
        if sel.size and last is not None:
            if sel[-1] - sel[0] + 1 != sel.size:            # the ages run one way: each age's columns are a band
                return None
            bands.append([int(sel[0]), int(sel[-1]) + 1, str(last)])
    return [{"op": "slitscan", "across": across, "bands": bands}] if bands else []
