"""A remap's `mask` for mkr-post (fx/masked.rs Masking): its figure layer, the reach of the fill and the sticker's edge,
shadow and back, as mkmmd.remap.RemapFx._remapped works them out."""
import math

from .. import remap as RMF
from ..core import framefx as FX
from .numbers import f32s


def masking(ffx, fx, frame):
    """The `mask` of remap `fx` on `frame` (None without one); False when mkr-post would read its layer otherwise than
    the CPU path; a FileNotFoundError when the layer is missing, as the CPU path raises."""
    if fx["mask"] is None:
        return None
    paths = ffx.paths
    fig = paths.figure(FX.figure_key(fx), frame)
    if not paths.fits(fig):
        return False
    edge, shadow = ffx._edge(fx), ffx._shadow(fx)
    reach = RMF.HOLE_MARGIN * ffx.px + (edge[1] if edge else 0.0)
    if shadow is not None:
        reach += math.hypot(*shadow[1]) + 2.5 * shadow[2]
    return {"figure": str(fig), "mask": fx["mask"], "reach": reach,
            "edge": None if edge is None else {"color": f32s(edge[0]), "width": edge[1]},
            "shadow": None if shadow is None else {"color": f32s(shadow[0]), "offset": list(shadow[1]),
                                                   "blur": shadow[2], "alpha": shadow[3]},
            "back": None if fx["back"] is None else f32s(fx["back"])}
