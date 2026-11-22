"""Threads on the finished pictures (docs/design.md: Frame effects, Sticker sheets): the picture work of the `[[fx]]` kind
`thread` that mkmmd.core.thread checks and shapes. Its tie is a point of a sheet's live sticker (a point of the figure's
own frame, or a world point seen through the camera the figure is read through, point/x<i>a/) where that sheet lays it
(mkmmd.sheet.SheetFx: the figure's map, its keys and slap, and the desk's drift) at each second the shape asks for; the
thread is drawn antialiased at SUPER times the size and shrunk by area, so a thread under a pixel wide still shows, over
its soft shadow. Images are float RGB in display space (0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from .core import framefx as FX
from .core import thread as TH

SUPER = 2            # the line is drawn this many times larger and shrunk by area
SHIFT = 4            # fixed-point bits of its points (sub-pixel)


def sheet_of(ffx, fx):
    """This output's entry of the sheet whose live sticker thread `fx` is tied to, by its `name`."""
    for s in ffx.fxs:
        if s["kind"] == "sheet" and s.get("name") == fx["sheet"]:
            if s.get("objects") is None or s.get("clip") is not None:
                raise FX.FxError(f"fx {fx['index']} (thread): sheet {fx['sheet']!r} lays no live sticker of its own: tie "
                                 f"a thread to a sheet with `objects` and no `clip`")
            return s
    raise FX.FxError(f"fx {fx['index']} (thread): no sheet is named {fx['sheet']!r} (give that sheet `name = ...`)")


def own_point(ffx, fx):
    """callable(Blender frame, any fraction of one) -> the px of the figure's own frame thread `fx` is tied at: its `at`
    fractions, or its world point seen through the camera the figure is read through (its point layer), straight
    between the frames about it and, before the first frame it is read on, held there."""
    W, H = ffx.size
    a = fx["at"]
    if a["mode"] == "frame":
        p = (a["at"][0] * W - 0.5, a["at"][1] * H - 0.5)
        return lambda g: p
    seen = {}

    def read(g):
        if g not in seen:
            q, m = ffx._point_of(TH.tie_key(fx), g)
            if m is None or m <= 0:
                raise FX.FxError(f"fx {fx['index']} (thread): at: its point has no size on frame {g}: it lies behind the "
                                 f"camera its sheet's figure is seen through, or at it")
            seen[g] = (float(q[0]), float(q[1]))
        return seen[g]

    def point(g):
        g = max(g, a["first"])
        g0 = int(math.floor(g + 1e-6))
        u = g - g0
        p = read(g0)
        if u <= 1e-6:
            return p
        q = read(g0 + 1)
        return p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u
    return point


def tie_of(ffx, fx, sheet):
    """callable(clip second) -> the frame px where thread `fx` is tied at that second: its point of `sheet`'s live
    sticker (own_point) where the sheet lays the figure then, by its map of the figure (placement, keys, slap) and the
    desk's drift."""
    st = ffx._sheet_state(sheet)
    own = own_point(ffx, fx)

    def tie(tau):
        g = ffx.frame0 + tau * ffx.fps
        M, _ = ffx._figure_map(sheet, g, st)
        A = FX.compose(ffx.desk(sheet, tau), M)
        x, y = own(g)
        return A[0][0] * x + A[0][1] * y + A[0][2], A[1][0] * x + A[1][1] * y + A[1][2]
    return tie


def coverage(pts, size, width, reach=0.0):
    """((x0, y0, x1, y1), coverage (h, w) float32 0..1): a line through `pts` (frame px) `width` px wide, antialiased,
    in the box round its part on a frame of `size` (`reach` px more on every side); None when none of it is on it."""
    W, H = size
    r = width / 2.0 + 2.0 + reach
    x0, y0 = max(int(math.floor(pts[:, 0].min() - r)), 0), max(int(math.floor(pts[:, 1].min() - r)), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max() + r)) + 1, W), min(int(math.ceil(pts[:, 1].max() + r)) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    big = np.zeros(((y1 - y0) * SUPER, (x1 - x0) * SUPER), np.uint8)
    q = ((pts - np.array([x0, y0], float)) * SUPER + (SUPER - 1) / 2.0) * (1 << SHIFT)
    cv2.polylines(big, [np.rint(q).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                  max(int(round(width * SUPER)), 1), cv2.LINE_AA, SHIFT)
    cov = cv2.resize(big.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return (x0, y0, x1, y1), np.clip(cov, 0.0, 1.0)


def _lay(out, got, color, alpha=1.0, blur=0.0):
    """`out` (in place) under `color` where coverage `got` (coverage()'s) lies, blurred `blur` px, at `alpha`."""
    if got is None:
        return
    (x0, y0, x1, y1), a = got
    if blur > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    k = (a * np.float32(alpha))[..., None]
    region = out[y0:y1, x0:x1]
    region += (np.asarray(color, np.float32) - region) * k


def draw(ffx, fx, frame, img):
    """Thread `fx` on `frame` over `img`: its shadow, then the thread, from the sticker's tie out to its `end`."""
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return img
    W, H = ffx.size
    tie = tie_of(ffx, fx, sheet_of(ffx, fx))
    pts = TH.shape(tie, (fx["end"][0] * W - 0.5, fx["end"][1] * H - 0.5), t, fx["wave"], fx["lag"], H)
    out = np.array(img, np.float32)
    w = fx["width"] * ffx.px
    sh = fx["shadow"]
    if sh is not None and sh["alpha"] > 0:
        b = sh["blur"] * ffx.px
        _lay(out, coverage(pts + np.array(sh["offset"], float) * H, ffx.size, w, 3.0 * b), sh["color"], sh["alpha"], b)
    _lay(out, coverage(pts, ffx.size, w), fx["color"])
    return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)
