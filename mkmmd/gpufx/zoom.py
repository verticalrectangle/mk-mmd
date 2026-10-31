"""zoom on the GPU: the punch on the hits as one affine warp (mkr-post fx/warp.rs), its map from mkmmd.framefx.scale_map
and FrameFx.zoom_at as the CPU path makes it."""
from .. import framefx as FF

BORDER_REFLECT = 2          # cv2.BORDER_REFLECT


def ops(ffx, fx, frame):
    M = FF.scale_map(*ffx.zoom_at(fx, frame))
    return [] if M is None else [{"op": "warp", "m": [float(v) for v in M.ravel()], "border": BORDER_REFLECT}]
