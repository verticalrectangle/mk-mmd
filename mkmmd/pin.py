"""Pinned stickers on the finished pictures (docs/design.md: Pinned stickers): the picture work of a table's `pin`, which
mkmmd.core.pin checks. mkmmd.table calls `carry` where it places the figure on the table (her sticker carried with her
seat from the clip frame shown on the span's first frame to the one shown now), `under` before it lays the figure (the
contact shadows on the seat) and `over` after it (the pieces of the clip in front of her, the table's own picture laid
again inside their outlines). Each takes the table's drawing object (mkmmd.table.TableFx: its size, px, fps and frame
clock) and the table's state (its clip and size). Images are float RGB in display space (0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from .core import framefx as FX
from .core import pin as PN
from .core import table as TB


def second(fx, clip, t):
    """The video second of the clip frame a table shows at clip second `t`: that frame's own (its index over the clip's
    frame rate), so a pin steps with the clip's frames."""
    return clip.index(t - fx["from"], fx["in"], fx["rate"], fx["loop"]) / clip.fps


def _state(tfx, fx, st):
    """What a pin keeps on its table's state: the video second shown on the span's first frame, once its track's rows
    are found to reach over every clip frame the span shows (a FxError otherwise: past them she would slide)."""
    ps = st.get("pin")
    if ps is None:
        secs = [second(fx, st["clip"], tfx._t(g)) for g in range(fx["f0"], fx["f1"])]
        rows = fx["pin"]["track"]
        if not PN.covers(rows, min(secs), max(secs)):
            raise FX.FxError(f"fx {fx['index']} (sheet): pin track: its rows run over video seconds {rows[0][0]:g} .. "
                             f"{rows[-1][0]:g}, and the span shows the clip's frames from {min(secs):.4f} to "
                             f"{max(secs):.4f} s: track every frame it shows")
        ps = st["pin"] = {"s0": secs[0]}
    return ps


def carry(tfx, fx, st, g, M):
    """`M` (the figure's plate px -> table px on frame `g`, placed as on the span's first frame) carried with the seat
    to the clip frame shown on `g`; `M` itself without a pin."""
    if fx.get("pin") is None:
        return M
    return FX.compose(_seat(tfx, fx, st, g).tolist(), M)


def _seat(tfx, fx, st, g):
    """The 2 x 3 map (table px as on the span's first frame -> table px on frame `g`) of a pinned table's seat."""
    s0 = _state(tfx, fx, st)["s0"]
    return PN.carry(fx["pin"]["track"], s0, second(fx, st["clip"], tfx._t(g)), st["table"])


def _box(pts, pad, size):
    W, H = size
    x0, y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, W), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, H)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _fill(pts, box):
    """The antialiased coverage (0..1, float32) of a polygon `pts` (frame px) over the frame's `box`."""
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.rint((pts - (x0, y0)) * 16.0).astype(np.int32)
    cv2.fillPoly(m, [q], 255, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255.0


def under(tfx, fx, st, frame, D, out):
    """`out` (the table as drawn under the figure; left as it is) with the pin's contact shadows on it: each ellipse
    carried with the seat and through the view `D`, darkening toward its colour by its alpha at this second, fading
    out from its middle, cut to its `on` outline."""
    pin = fx.get("pin")
    if pin is None or not pin["shadow"]:
        return out
    t = tfx._t(frame)
    A = np.asarray(FX.compose(np.asarray(D).tolist(), _seat(tfx, fx, st, frame).tolist()), float)
    z, turn = TB.zoom_of(A), math.radians(TB.turn_of(A))
    Wt, Ht = st["table"]
    res = None
    for s in pin["shadow"]:
        a = PN.alpha_at(s["alpha"], t)
        if a <= 0.0:
            continue
        c = PN.through(A, PN.outline_px([s["at"]], st["table"]))[0]
        ra, rb = 0.5 * s["size"][0] * Ht * z, 0.5 * s["size"][1] * Ht * z
        box = _box(np.array([c]), max(ra, rb) + 2.0, tfx.size)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        th = math.radians(s["turn"]) + turn
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(c[0])
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(c[1])
        u = dx * np.float32(math.cos(th)) - dy * np.float32(math.sin(th))
        v = dx * np.float32(math.sin(th)) + dy * np.float32(math.cos(th))
        m = PN.falloff(np.sqrt((u / np.float32(ra)) ** 2 + (v / np.float32(rb)) ** 2), s["soft"]) * np.float32(a)
        if s["on"] is not None:
            m = m * _fill(PN.through(A, PN.outline_px(s["on"], st["table"])), box)
        if not m.any():
            continue
        if res is None:
            res = out.copy()
        region = res[y0:y1, x0:x1]
        region += (np.asarray(s["color"], np.float32) - region) * m[..., None].astype(np.float32)
    return out if res is None else res


def over(tfx, fx, st, frame, D, ground, out):
    """`out` (the table with her sticker laid) with the pin's pieces of the clip in front of her: `ground`, the table as
    it was drawn before her, laid again inside each outline (carried with the seat and through the view `D`, its edge
    feathered), only where its luma lies in the piece's `key` when it has one."""
    pin = fx.get("pin")
    if pin is None or not pin["front"]:
        return out
    A = np.asarray(FX.compose(np.asarray(D).tolist(), _seat(tfx, fx, st, frame).tolist()), float)
    for f in pin["front"]:
        pts = PN.through(A, PN.outline_px(f["outline"], st["table"]))
        sigma = 0.5 * f["soft"] * tfx.px
        box = _box(pts, 3.0 * sigma + 2.0, tfx.size)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        m = _fill(pts, box)
        if sigma > 0.3:
            m = cv2.GaussianBlur(m, (0, 0), sigma)
        g = ground[y0:y1, x0:x1]
        if f["key"] is not None:
            lo, hi = f["key"]["luma"]
            k = np.float32(max(f["key"]["soft"], 1e-4))
            L = g[..., 0] * np.float32(0.299) + g[..., 1] * np.float32(0.587) + g[..., 2] * np.float32(0.114)
            m = m * np.clip((L - np.float32(lo)) / k + 1.0, 0.0, 1.0) * np.clip((np.float32(hi) - L) / k + 1.0, 0.0, 1.0)
        if not m.any():
            continue
        region = out[y0:y1, x0:x1]
        region += (g - region) * m[..., None]
    return out
