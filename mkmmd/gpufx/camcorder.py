"""camcorder on the GPU (mkr-post fx/camcorder.rs): the tape's random parts as mkmmd.picturefx.tape_draws draws them
(the noise before it is stretched over the picture, each line's tear, the tracking band's snow), each line's wash and
scanline, and the stamp's coverages (picturefx.stamp_inks: type set by Pillow here), written for mkr-post; the tape
played, the stamp laid and the strength's mix drawn by mkr-post."""
import os

import numpy as np

from .. import picturefx as PFX
from ..core import framefx as FX
from .fields import f32
from .figure import ink_list


def _floats(a, path):
    np.ascontiguousarray(a, np.float32).tofile(path)
    return str(path)


def ops(ffx, fx, frame):
    folder = None if ffx.paths is None else ffx.paths.scratch
    if folder is None:
        return None
    t, px, (w, h) = ffx._t(frame), ffx.px, ffx.size
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    tag = f"camcorder-{os.getpid()}-{fx['index']}-{frame}"
    d = PFX.tape_draws(fx, frame, t, px, w, h)
    rows = np.zeros((h, 4), np.float32)
    rows[:, 0] = d["off"]
    rows[:, 3] = 1.0
    op = {"op": "camcorder", "soft": None, "chroma": None, "saturation": f32(fx["saturation"]), "noise": None,
          "lines": h, "tear": bool(np.abs(d["off"]).max() > 0.01), "snow": None, "strength": s}
    if fx["soft"] > 0:
        sg = fx["soft"] * px
        op["soft"] = [sg, max(0.35 * sg, 0.3)]
    if fx["chroma"] > 0:
        r = fx["chroma"] * px
        op["chroma"] = {"n": max(int(round(2.0 * r)), 1), "sigma": [max(0.5 * r, 0.3), max(0.6 * px, 0.3)],
                        "shift": PFX.CHROMA_SHIFT * r}
    if d["ny"] is not None:
        ny, nc = d["ny"], d["nc"]
        op["noise"] = {"ny": _floats(ny, folder / f"{tag}-ny.f32"), "ny_size": [ny.shape[1], ny.shape[0]],
                       "nc": _floats(nc, folder / f"{tag}-nc.f32"), "nc_size": [nc.shape[1], nc.shape[0]],
                       "amount": [f32(fx["noise"]), f32(0.5 * fx["noise"])]}
    if d["snow"] is not None:
        band, snow = d["band"], d["snow"]
        rows[:, 1] = 0.45 * band
        rows[:, 2] = np.minimum(band * 1.6, 1.0)
        op["snow"] = {"path": _floats(snow, folder / f"{tag}-snow.f32"), "size": [snow.shape[1], snow.shape[0]]}
    if fx["scanlines"] > 0:
        rows[:, 3] = PFX._scanlines(h, float(fx["scanlines"]), float(PFX.SCANLINE * px))
    op["rows"] = _floats(rows, folder / f"{tag}-rows.f32")
    op["inks"] = ink_list(PFX.stamp_inks((w, h), fx, t, px, ffx._fonts[fx["index"]]), folder,
                          f"camcorder-{fx['index']}-{frame}")
    return [op]
