"""trails on the GPU (mkr-post fx/trails.rs): the copies from MandalaFx.trails_at, each with its figure layer and, when
its colour is mixed with its own picture (`tint` under 1), the cut's frame of it, as MandalaFx._trails lays them."""
from ..core import framefx as FX
from .numbers import f32, f32s

BLENDS = {"over": 0, "screen": 1, "add": 2}


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None:
        return None
    s, copies = ffx.trails_at(fx, frame)
    if s <= 0.0:
        return []
    key = FX.figure_key(fx)
    out = []
    for g, opacity, colour, (dx, dy) in copies:
        if g < ffx.frame0:                              # before the clip: no figure there
            continue
        fig = paths.figure(key, g)
        if not paths.fits(fig):
            return None
        c = {"figure": str(fig), "frame": None, "alpha": f32(opacity), "colour": f32s(colour)}
        if fx["tint"] < 1.0:
            pic = paths.frame(g)
            if pic is not None:
                if not paths.fits(pic):
                    return None
                c["frame"] = str(pic)
        if dx or dy:
            c["shift"] = [dx, dy]
        out.append(c)
    fig = paths.figure(key, frame) if frame >= ffx.frame0 else None
    if fig is not None and not paths.fits(fig):
        return None
    return [{"op": "trails", "copies": out, "blend": BLENDS[fx["blend"]], "tint": f32(fx["tint"]),
             "figure": None if fig is None else str(fig)}]
