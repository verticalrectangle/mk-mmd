"""doodle on the GPU (mkr-post fx/doodle.rs): each mark's strokes on the frame (mkmmd.picturefx.doodle_strokes, as
PictureFx._doodle boils them) and their masks, drawn here with OpenCV's antialiased lines in the box they cover as
mkmmd.picturefx.draw_strokes draws them (a few hundred px), written for mkr-post; the rim, the ink and the mix laid by
mkr-post."""
import math
import os

import cv2
import numpy as np

from .. import picturefx as PFX
from ..core import framefx as FX
from ..core import picturefx as PC


def ops(ffx, fx, frame):
    folder = None if ffx.paths is None else ffx.paths.scratch
    if folder is None:
        return None
    t, (w, h) = ffx._t(frame), ffx.size
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    b = PC.boil_step(fx, t)
    marks = []
    for j, m in enumerate(fx["marks"]):
        if t <= m["t"]:
            continue
        if m["at"]["mode"] == "frame":
            c, per_m = (m["at"]["at"][0] * w - 0.5, m["at"]["at"][1] * h - 0.5), None
        else:
            c, per_m = ffx._point_of(f"{FX.point_key(fx)}m{j}", frame)
        size = m["size"] * h if "size" in m else m["size_m"] * per_m * h
        rng = np.random.default_rng([fx["seed"], fx["index"], j, b])
        strokes, k = PFX.doodle_strokes(m["shape"], c, size, m["angle"], t - m["t"], rng, fx["wobble"] * ffx.px)
        width, outline = max(fx["width"] * ffx.px * min(k, 1.0), 1.0), fx["outline"] * ffx.px * min(k, 1.0)
        strokes = [p for p in strokes if len(p) >= 2]           # as draw_strokes draws them, in their box
        if not strokes:
            continue
        pts = np.concatenate(strokes)
        pad = width / 2.0 + outline + 3.0
        x0, y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
        x1, y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, w), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        q = [PFX._q(p - (x0, y0)) for p in strokes]
        rim, ink = np.zeros((y1 - y0, x1 - x0), np.uint8), np.zeros((y1 - y0, x1 - x0), np.uint8)
        if outline > 0:
            cv2.polylines(rim, q, False, 255, max(int(round(width + 2.0 * outline)), 1), cv2.LINE_AA, 4)
        cv2.polylines(ink, q, False, 255, max(int(round(width)), 1), cv2.LINE_AA, 4)
        path = folder / f"doodle-{os.getpid()}-{frame}-{fx['index']}-{j}.f32"
        np.stack([rim.astype(np.float32) / 255.0, ink.astype(np.float32) / 255.0], -1).astype(np.float32).tofile(path)
        marks.append({"box": [x0, y0, x1, y1], "masks": str(path), "color": np.asarray(m["color"], np.float32).tolist()})
    return [{"op": "doodle", "marks": marks, "strength": s}]
