"""mk's own renderer, the finish (docs/design.md: Rendering): a frame's passes as the page sent them (passes.js: id and
shade, depth, the texture pass; bottom row first), composed with mkmmd.core.shotstyle.compose_vector and the rings'
masks exactly as mkmmd.blender.styles composes Blender's passes, and written as the frame's 8-bit PNG (whole or not at
all). Runs in worker processes."""
import os

import numpy as np

from ..core import rings as RG
from ..core import shotstyle as SS


def passes(buf, size, tex):
    """The passes compose_vector reads, top row first, from the page's bytes for passes of `size` [W, H]."""
    W, H = size
    n = W * H * 4
    a = np.frombuffer(buf, np.uint8)
    if a.size != n * (3 if tex else 2):
        raise ValueError(f"{a.size} bytes of passes, expected {n * (3 if tex else 2)} for {W} x {H}")
    idsh = a[:n].reshape(H, W, 4)[::-1]
    p = {"id": idsh[..., 0].astype(np.int32) | (idsh[..., 1].astype(np.int32) << 8),
         "shade": idsh[..., 2].astype(np.float32) / 255.0,
         "depth": a[n:2 * n].view("<f4").reshape(H, W)[::-1]}
    if tex:
        p["tex"] = a[2 * n:].reshape(H, W, 4)[::-1, :, :3].astype(np.float32) / 255.0
    return p


def finish(path, buf, size, tex, spec, table, rings, t, centres):
    """Compose one frame and write it to `path`: `rings` the ring table's rings, `t` the clip second, `centres` the live
    rings' centres as frame fractions."""
    p = passes(buf, size, tex)
    flip, edges = None, ()
    live = RG.live(rings, t)
    if live:
        fs = (size[0] / SS.VECTOR_SCALE, size[1] / SS.VECTOR_SCALE)
        flip, edges = RG.masks(RG.shapes(rings, t, centres, fs), fs, SS.VECTOR_SCALE)
    img = SS.compose_vector(p, spec, (table[0], table[1]), flip=flip, edges_on=edges)
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        fh.write(SS.png_bytes(img8))
    os.replace(tmp, path)
