"""The collages' figure on the GPU (mkr-post fx/figure.rs): what mkmmd.framefx.FrameFx cuts out of a frame for a collage
(`_collage`, `_posed`), its sticker edge and drop shadow, as the files and maps mkr-post reads; and sticker tiles
(mkmmd.mandala.Tile) written once for mkr-post to read. Shared by the collage kinds' modules (footage, grid, stack,
mandala ...)."""
import hashlib
import itertools
import os
from pathlib import Path

import numpy as np

from .. import framefx as FF
from ..core import framefx as FX

_ONCE = itertools.count()                     # this process's tiles written for one frame alone


def tile_files(ffx, t, key, once=None):
    """Sticker tile `t` (mkmmd.mandala.Tile) on disk for mkr-post, in the scratch folder of `ffx.paths`, written once by
    whichever process asks first (named by `key`, which says what it is): its picture and coverage as h x w vec4
    float32 (rgb premultiplied, a) and its ring (in the first of four); mkr-post's Tile::File. With `once` (a frame:
    the tile is in colours few frames show, a breathing palette's on its way: FrameFx.passing) it is written for that
    frame alone, under a name of its own, and mkr-post removes it once read, so a breath leaves no trail of tiles behind
    (forget_once removes it when the frame's ops are not sent). None without a scratch folder."""
    folder = getattr(ffx.paths, "scratch", None)
    if folder is None:
        return None
    name = hashlib.sha1(repr(key).encode()).hexdigest()[:24]
    if once is not None:
        name = f"{name}-{os.getpid()}-{next(_ONCE)}"
    h, w = t.a.shape
    out = {"kind": "file", "rgba": str(Path(folder) / f"tile-{name}.rgba"), "ring": None, "size": [int(w), int(h)],
           "once": once is not None}
    planes = [("rgba", np.dstack([t.rgb, t.a]))]
    if t.ring is not None:
        out["ring"] = str(Path(folder) / f"tile-{name}.ring")
        ring = np.zeros((h, w, 4), np.float32)
        ring[..., 0] = t.ring
        planes.append(("ring", ring))
    for which, data in planes:
        p = Path(out[which])
        if not p.exists():
            tmp = p.with_name(f"{p.name}.{os.getpid()}")
            np.ascontiguousarray(data, np.float32).tofile(tmp)
            os.replace(tmp, p)
    if once is not None:
        got = getattr(ffx, "_gpu_once", None)
        if got is None or got[0] != once:
            got = ffx._gpu_once = (once, [])
        got[1].extend(p for p in (out["rgba"], out["ring"]) if p is not None)
    return out


def forget_once(ffx, frame):
    """The tiles written for `frame` alone (tile_files `once`) removed: its ops were not sent."""
    got = getattr(ffx, "_gpu_once", None)
    if got is not None and got[0] == frame:
        for p in got[1]:
            Path(p).unlink(missing_ok=True)
        ffx._gpu_once = None


def ink_list(inks, folder, tag):
    """Coverages laid by flat colours ([(coverage (h, w) 0..1, colour, x0, y0, alpha)], mkmmd.labels._ink's arguments) as
    mkr-post's Ink list, each coverage written for this frame into `folder` (a solid box: none written)."""
    out = []
    for i, (cov, colour, x0, y0, alpha) in enumerate(inks):
        cov = np.ascontiguousarray(cov, np.float32)
        h, w = cov.shape
        ink = {"at": [int(x0), int(y0)], "size": [int(w), int(h)], "mask": None,
               "color": np.asarray(colour, np.float32).tolist(), "alpha": float(np.float32(alpha))}
        if w == 0 or h == 0:
            continue
        if cov.min() < 1.0:
            path = Path(folder) / f"{tag}-{os.getpid()}-{i}.f32"
            cov.tofile(path)
            ink["mask"] = str(path)
        out.append(ink)
    return out


def warp(M, shape):
    """The JSON of mkr-post's WarpSpec for carrying a picture of `shape` ((h, w, ...)) by the affine map `M`, as
    mkmmd.framefx._warp does it (framefx.warp_spec): {m: the float32 map, shrink: [w, h] or None}."""
    Mf, shrink = FF.warp_spec(M, shape)
    return {"m": [float(v) for v in Mf.ravel()], "shrink": None if shrink is None else [int(shrink[0]), int(shrink[1])]}


def cutout(ffx, fx, g, picture=True):
    """The figure of collage `fx` on frame `g` as mkr-post reads it (FrameFx._collage / _posed): {layer: its figure
    layer, plate: the picture it is cut from (None: the picture as it reaches the op; else its own `shot`'s, a PNG),
    warp: the map that sets it into the frame and moves it by its keys (None: as it is)}; `picture` False when only its
    coverage is drawn (a flat fill). None when mkr-post cannot read it as the CPU does (a layer or a plate of another
    size; no paths) or before the clip; a FileNotFoundError when a layer or a plate is missing, as the CPU path raises."""
    paths = ffx.paths
    if paths is None or g < ffx.frame0:
        return None
    layer = paths.figure(FX.figure_key(fx), g)
    plate = paths.plate(fx["shot"], g) if picture and fx.get("shot") else None
    if not paths.fits(layer) or (plate is not None and not paths.fits(plate)):
        return None
    M = ffx._pose_map(fx, g)
    h, w = ffx.size[1], ffx.size[0]
    return {"layer": str(layer), "plate": None if plate is None else str(plate),
            "warp": None if M is None else warp(M, (h, w))}


def edge(ffx, fx):
    """A collage's die-cut edge as mkr-post's Edge ({color, width px}; FrameFx._edge), or None."""
    e = ffx._edge(fx)
    return None if e is None else {"color": [float(c) for c in e[0]], "width": float(e[1])}


def shadow(ffx, fx):
    """A collage's drop shadow as mkr-post's Shadow ({color, offset px, blur px, alpha}; FrameFx._shadow), or None."""
    s = ffx._shadow(fx)
    return None if s is None else {"color": [float(c) for c in s[0]], "offset": [float(s[1][0]), float(s[1][1])],
                                   "blur": float(s[2]), "alpha": float(s[3])}


def colour(c):
    """A colour as JSON (three floats), None as None."""
    return None if c is None else [float(v) for v in c]
