"""scope on the GPU (mkr-post fx/scope.rs): the trace from FrameFx.scope_at drawn here as the CPU path draws it
(mkmmd.framefx.trace_mask: OpenCV's antialiased polyline), its coverage over the box it touches handed over in the
scratch folder; the glow and the sum are mkr-post's."""
import os
from pathlib import Path

import numpy as np

from .. import framefx as FF
from .numbers import f32, f32s


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None or paths.scratch is None:
        return None
    pts, centre, radius, width, opacity = ffx.scope_at(fx, frame)
    if len(pts) < 2 or opacity <= 0:
        return []
    mask = FF.trace_mask(ffx.size, pts, centre, radius, width)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    x0, y0, x1, y1 = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1) if rows.size else (0, 0, 0, 0)
    path = Path(paths.scratch) / f"{os.getpid()}-scope{fx['index']}-{frame}.f32"
    if x1 > x0:
        (mask[y0:y1, x0:x1].astype(np.float32) / 255.0).tofile(path)
    return [{"op": "scope", "a": str(path), "bx": [x0, y0, x1 - x0, y1 - y0],
             "glow": f32(fx["glow"]) if fx["glow"] > 0 else 0.0, "sigma": FF.GLOW_SIGMA * ffx.px,
             "opacity": f32(opacity), "colour": f32s(fx["color"])}]
