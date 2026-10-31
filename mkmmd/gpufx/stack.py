"""stack on the GPU (mkr-post fx/stack.rs): the copies on the frame (mkmmd.core.stack.copies, as StackFx._stacked lays
them out), the frames and fills they are drawn from, their maps (StackFx._sk_map) and opacities, the mirror's
reflections worked out here as mkmmd.stack.draw and lay work them out; each layer's box and every pixel drawn by
mkr-post. Wallpaper bands, era looks, a squash, silkscreen fills and copies shrunk under SHRINK_BY_AREA stay on the
CPU."""
import math

import numpy as np

from .. import framefx as FF
from .. import stack as SKF
from ..core import framefx as FX
from ..core import stack as SK
from . import figure as GF
from .mandala import back as back_of


def ops(ffx, fx, frame):
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    if fx["squash"] is not None:
        return None
    paths = ffx.paths
    times = ffx._hit_times(fx)
    k = FX.step(fx, frame, times, ffx.fps, ffx.frame0)
    centre = ffx._at(fx, frame)[0]
    w, h = ffx.size
    edge = ffx._edge(fx)
    layers = []                                          # (cutout, fill, map, alpha, thin)
    for cp in SK.copies(fx, t, k, times, ffx.fps, ffx.frame0):
        band = FX.pick(fx["bands"], cp["band"])
        look = None if fx["looks"] is None else FX.pick(fx["looks"], cp["band"])
        if SK.is_sprites(band) or band == FX.SILK or look not in (None, "none"):
            return None
        g = frame - cp["delay"]
        if g < ffx.frame0:
            continue
        fill = ffx._fill(fx, cp["band"], fx["bands"])
        cut = GF.cutout(ffx, fx, g, picture=fill is None)
        if cut is None:
            return None
        if fill is None and cp["delay"] and not fx.get("shot"):     # a copy's picture: the cut's frame then
            p = paths.frame(g)
            if p is None:
                continue
            if not paths.fits(p):
                return None
            cut["plate"] = str(p)
        layers.append((cut, fill, ffx._sk_map(fx, cp, centre), cp["alpha"], fx["recede"] is not None))
    fill = ffx._fill(fx, k)
    if isinstance(fill, dict):
        return None
    main = GF.cutout(ffx, fx, frame, picture=fill is None)
    if main is None:
        return None
    layers.append((main, fill, SKF.IDENTITY, 1.0, False))
    X = int(round(2.0 * centre[0])) if fx["mirror"] else None
    out = []
    for flip in ([None] if X is None else [None, X]):
        for cut, fl, M, alpha, thin in layers:
            A = np.asarray(M, np.float64).reshape(2, 3)
            if flip is not None:
                A = np.array([[A[0, 0], -A[0, 1], flip - A[0, 0] * flip - A[0, 2]],
                              [-A[1, 0], A[1, 1], A[1, 2] + A[1, 0] * flip]])
            sc = math.sqrt(abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))
            if sc < FF.SHRINK_BY_AREA:                   # shrunk by area on the frame's grid first: the CPU's
                return None
            ew = 0.0 if edge is None or edge[1] <= 0 else float(edge[1]) * (sc if thin else 1.0)
            out.append({"figure": cut, "fill": GF.colour(fl), "map": [float(v) for v in A.ravel()], "flip": flip,
                        "alpha": float(alpha), "s": sc, "edge": ew})
    return [{"op": "stack", "back": back_of(ffx, fx, frame, k), "layers": out,
             "edge": None if edge is None else GF.colour(edge[0]), "shadow": GF.shadow(ffx, fx), "strength": s}]
