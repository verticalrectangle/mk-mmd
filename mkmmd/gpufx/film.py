"""film on the GPU (mkr-post fx/film.rs): the film frame a clip frame shows (mkmmd.core.picturefx.film_step), and on a
step's first frame how its print is made: the jolt (gain, weave), the dust, scratches and hairs drawn here into masks
(mkmmd.picturefx.damage_marks, sent as the parts they cover), the grain's normals from numpy's generator as grain_field
draws them, and the constants of mkmmd.picturefx.film. mkr-post keeps the print for the step's other frames: this
process plans a step's frames in order (mkmmd.postframes chunks start on a step's first frame), so it knows the print
is kept when it sent it, and leaves a frame to the CPU when it did not. A film limited to a `rect` (RECT: mkmmd.postgpu
lets it through) prints the rect's box as a picture of its own (PictureFx._region), kept inside the rect by its ramps
(picturefx.rect_ramps, as FrameFx._apply keeps it)."""
import math
import os

import cv2

import numpy as np

from .. import picturefx as PFX
from ..core import framefx as FX
from ..core import picturefx as PC
from .fields import f32


def _kept(ffx):
    """{entry index: (film frame, clip frame)} of the prints this process sent mkr-post to keep."""
    got = getattr(ffx, "_gpu_films", None)
    if got is None:
        got = ffx._gpu_films = {}
    return got


def dropped(ffx, frame):
    """The ops worked out for `frame` were not sent (mkmmd.postframes.Output.plan): a print made there is not kept."""
    kept = _kept(ffx)
    for i in [i for i, (_, g) in kept.items() if g == frame]:
        del kept[i]


def _mask(m, folder, name):
    """A damage mask (uint8, frame-sized) as mkr-post's Mask: the part its marks cover, v / 255 as numpy makes it;
    None when nothing is on it."""
    x0, y0, bw, bh = cv2.boundingRect(m)
    if bw == 0 or bh == 0:
        return None
    x1, y1 = x0 + bw, y0 + bh
    path = folder / name
    (m[y0:y1, x0:x1].astype(np.float32) / 255.0).astype(np.float32).tofile(path)
    return {"box": [x0, y0, x1, y1], "part": str(path)}


RECT = True


def _print(ffx, fx, k, folder, tag, size):
    """How film frame k's print of a picture of `size` (w, h) is made (mkr-post's Print), as mkmmd.picturefx.film makes
    it."""
    w, h = size
    px = ffx.px
    gain, dx, dy = PC.film_jolt(fx, k)
    c = 4.0 * fx["contrast"]
    out = {"mono": bool(fx["mono"]), "dye_on": bool(fx["dye"] > 0 or fx["saturation"] != 1.0),
           "sat": f32(fx["saturation"]), "dye": [f32(0.45 * fx["dye"]), f32(0.3 * fx["dye"]), f32(fx["dye"])],
           "curve": None if c < 1e-3 else [f32(c / 2.0), f32(math.tanh(c / 4.0))],
           "weave": [f32(dx * px), f32(dy * px)] if abs(dx) + abs(dy) > 1e-3 else None,
           "soft": 0.5 * px, "gain": f32(gain), "light": None, "dark": None, "halation": None, "grain": None,
           "vignette": f32(fx["vignette"]) if fx["vignette"] > 0 else None}
    light, dark = PFX.damage_marks(fx, k, (w, h), px)
    if light is not None:
        out["light"] = _mask(light, folder, f"{tag}-light.f32")
    if dark is not None:
        out["dark"] = _mask(dark, folder, f"{tag}-dark.f32")
    if fx["halation"] > 0:
        out["halation"] = {"amount": f32(fx["halation"]), "near": max(5.0 * px, 0.5), "far": max(16.0 * px, 0.5),
                           "color": np.asarray((0.8, 0.8, 0.8) if fx["mono"] else PFX.HALATION, np.float32).tolist()}
    g = fx["grain"]
    if g["amount"] > 0:
        rng = np.random.default_rng([fx["seed"], fx["index"], 6, k])
        size = g["size"] * px
        gh, gw = max(int(round(h / max(size, 1.0))), 2), max(int(round(w / max(size, 1.0))), 2)
        path = folder / f"{tag}-grain.f32"
        rng.standard_normal((gh, gw), dtype=np.float32).tofile(path)
        out["grain"] = {"normals": str(path), "size": [gw, gh], "amount": f32(g["amount"])}
    return out


def ops(ffx, fx, frame):
    folder = None if ffx.paths is None else ffx.paths.scratch
    if folder is None:
        return None
    k, g = PC.film_step(fx, frame)
    kept = _kept(ffx)
    if frame != g and kept.get(fx["index"], (None,))[0] != k:
        return None
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    tag = f"film-{os.getpid()}-{fx['index']}-{frame}"
    size, region = ffx.size, None
    if fx.get("rect") is not None:
        feather = fx["feather"] * ffx.px
        x0, y0, x1, y1 = PFX.rect_box(ffx.size, fx["rect"], feather)
        rows, cols = PFX.rect_ramps(ffx.size, tuple(fx["rect"]), feather)
        path = folder / f"{tag}-ramps.f32"
        np.concatenate([rows[y0:y1], cols[x0:x1]]).astype(np.float32).tofile(path)
        size, region = (x1 - x0, y1 - y0), {"box": [x0, y0, x1, y1], "ramps": str(path)}
    if frame == g:
        kept[fx["index"]] = (k, frame)
        return [{"op": "film", "entry": fx["index"], "step": k, "print": _print(ffx, fx, k, folder, tag, size),
                 "strength": s, "region": region}]
    return [{"op": "film", "entry": fx["index"], "step": k, "print": None, "strength": s, "region": region}]
