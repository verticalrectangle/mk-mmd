"""The picture as flat shapes on the finished pictures (docs/design.md: Frame effects, Shapes): the picture work of the
`shapes` kind that mkmmd.core.shapes checks and times. Images are float RGB in display space (0..1), (h, w, 3); pixel
(i, j) is centred on (i, j). Every pixel finds its cell in closed form (its row, then its column in that row's
offset), and each cell's shape is drawn as a signed distance, antialiased over a pixel, inside its own cell: a shape
never reaches past the circle its cell's shorter side holds, whatever its turn."""
import math

import cv2
import numpy as np

from .core import framefx as FX
from .core import shapes as SH

SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


def _sdf(name, up, vp, rad):
    """Signed distance (px, negative inside) of a shape of circumradius `rad` about its cell's middle, at the pixel's
    offsets `up`, `vp` from it in the shape's own turned frame (x right, y down)."""
    if name == "circle":
        return np.sqrt(up * up + vp * vp) - rad
    if name == "square":                                   # half its side rad / sqrt 2: its corners on the circle
        return np.maximum(np.abs(up), np.abs(vp)) - rad * np.float32(1.0 / SQ2)
    if name == "diamond":                                  # its corners on the circle, on the axes
        return (np.abs(up) + np.abs(vp) - rad) * np.float32(1.0 / SQ2)
    rh = rad * np.float32(SQ3 / 2.0)                       # an equilateral triangle, a point up, about its centroid
    px = np.abs(up) - rh
    py = -vp + rh * np.float32(1.0 / SQ3)
    fold = px + np.float32(SQ3) * py > 0.0
    px, py = (np.where(fold, (px - np.float32(SQ3) * py) * np.float32(0.5), px),
              np.where(fold, (-np.float32(SQ3) * px - py) * np.float32(0.5), py))
    px = px - np.clip(px, -2.0 * rh, 0.0)
    return -np.sqrt(px * px + py * py) * np.sign(py)


def _layer(img, cell, layout, shapes, size, turn, spin, inks, pick, back, skip, seed, step, t, age=None, pop=0.0,
           stagger=0.0):
    """One layout of shapes over `img`: (paint (h, w, 3), alpha (h, w), started (h, w) bool), each pixel's cell's shape
    colour, its coverage, and whether that cell's shape has begun to pop in (`age` s after the layout was laid, its
    cell's delay up to `stagger` s; without an `age` every shape is there at its size)."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    dx, dy = SH.lattice((W, H), cell, layout)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r_lo, r_hi = int(math.floor(-cy / dy)) - 1, int(math.ceil((H - 1 - cy) / dy)) + 1
    q_lo, q_hi = int(math.floor(-cx / dx)) - 2, int(math.ceil((W - 1 - cx) / dx)) + 2
    rows, cols = np.arange(r_lo, r_hi + 1), np.arange(q_lo, q_hi + 1)
    off = np.where((layout == "hex") & ((rows & 1) == 1), dx / 2.0, 0.0)          # each row's shift along
    # ---- each cell: its colour, shape, turn, size
    xc = cx + cols[None, :] * dx + off[:, None]
    yc = np.broadcast_to(cy + rows[:, None] * dy, xc.shape)
    k = max(int(round(0.8 * min(dx, dy))), 1)
    soft = cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT) if k > 1 else img
    col = soft[np.clip(np.rint(yc), 0, H - 1).astype(np.intp), np.clip(np.rint(xc), 0, W - 1).astype(np.intp)]
    d = SH.deal(cols[None, :], rows[:, None], step, seed, 4)
    which = np.minimum((d[..., 0] * len(shapes)).astype(np.intp), len(shapes) - 1)
    turned = np.radians((d[..., 1] * 2.0 - 1.0) * turn + spin * t)
    if age is None or pop <= 0:
        grow, begun = np.ones(which.shape), np.ones(which.shape, bool)
    else:
        since = age - d[..., 2] * stagger
        grow, begun = SH.pop(since / pop), since > 0.0
    rad = np.minimum(0.5 * min(dx, dy) * size * grow, 0.5 * min(dx, dy))   # a pop's overshoot stops at the cell
    on = np.ones(which.shape, bool)
    if back is not None:
        on = np.sqrt(np.sum((col - np.asarray(back, np.float32)) ** 2, axis=-1) / 3.0) > skip
    if inks is not None:
        ink = np.asarray(inks, np.float32)
        if pick == "nearest":
            j = np.argmin(np.sum((col[..., None, :] - ink) ** 2, axis=-1), axis=-1)
        elif pick == "shape":
            j = which % len(ink)
        else:
            j = np.minimum((d[..., 3] * len(ink)).astype(np.intp), len(ink) - 1)
        col = ink[j]
    # ---- each pixel: its cell, its place in it, the cell's shape there
    ys, xs = np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64)
    rr = np.rint((ys - cy) / dy).astype(np.intp)
    shift = off[rr - r_lo]
    qq = np.rint((xs[None, :] - cx - shift[:, None]) / dx).astype(np.intp)
    u = (xs[None, :] - cx - shift[:, None] - qq * dx).astype(np.float32)
    v = np.broadcast_to((ys - cy - rr * dy).astype(np.float32)[:, None], u.shape)
    cid = (rr - r_lo)[:, None] * len(cols) + (qq - q_lo)
    a = turned.ravel()[cid].astype(np.float32)
    c, s = np.cos(a), np.sin(a)
    up, vp = c * u + s * v, c * v - s * u
    r = rad.ravel()[cid].astype(np.float32)
    sid = which.ravel()[cid]
    sdf = np.full((H, W), np.float32(1e6), np.float32)
    for i, name in enumerate(shapes):
        if name not in shapes[:i]:                          # each kind once, over every cell dealt it
            sdf = np.where(np.isin(sid, [j for j, n in enumerate(shapes) if n == name]), _sdf(name, up, vp, r), sdf)
    alpha = np.clip(0.5 - sdf, 0.0, 1.0) * on.ravel()[cid]
    return col.reshape(-1, 3).astype(np.float32)[cid], alpha.astype(np.float32), begun.ravel()[cid]


def picture(img, cell, layout, shapes, size, turn, spin, inks, pick, back, skip, pop, stagger, seed, step, age, t,
            before=None):
    """`img` as flat shapes, a new picture: cells `cell` short sides apart (mkmmd.core.shapes.lattice) centred on the
    frame's middle, each its shape dealt from `shapes` (and turned within ±`turn` degrees, spinning `spin` a second at
    clip second `t`) in the picture's colour averaged over about a cell at its middle, or one of `inks` (`pick`: the
    nearest, the shape's own, dealt), on the picture or on the colour `back` (a cell within `skip` of it draws nothing);
    `age` s after the layout was laid each shape pops in over `pop` s, `stagger` s apart at most. `step` and `seed`
    choose the deal. `before`, (cell, step) of the layout before: until a cell's new shape begins to pop, the old
    layout's shape stays there, so a change of layout never leaves the picture empty (the cells divide)."""
    img = np.asarray(img, np.float32)
    paint, alpha, begun = _layer(img, cell, layout, shapes, size, turn, spin, inks, pick, back, skip, seed, step, t,
                                 age, pop, stagger)
    if before is not None and not begun.all():
        p0, a0, _ = _layer(img, before[0], layout, shapes, size, turn, spin, inks, pick, back, skip, seed, before[1], t)
        paint, alpha = np.where(begun[..., None], paint, p0), np.where(begun, alpha, a0)
    base = img if back is None else np.broadcast_to(np.asarray(back, np.float32), img.shape)
    return (base + (paint - base) * alpha[..., None]).astype(np.float32)


# ================================================================================================= over a cut
class ShapesFx:
    """The shapes' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's fps, frame0, _t, _hit_times, _masked and
    _mix)."""

    def shapes_at(self, fx, frame):
        """(cell short sides, layout's step, seconds since it was laid, clip second, (cell, step) of the layout before
        or None) of a shapes entry on `frame`: its hits taken on the frames they land on."""
        t = self._t(frame)
        times = [(FX.frame_of(h, self.fps, self.frame0) - self.frame0) / self.fps for h in self._hit_times(fx)]
        cell, since, step = SH.cell_at(fx, t, times)
        before = None
        if step > 0 and fx["pop"] > 0 and t - since < fx["pop"] + fx["stagger"]:
            before = SH.cell_at(fx, since - 0.5 / self.fps, times)[0], step - 1
        return cell, step, t - since, t, before

    def _shapes(self, fx, frame, img):
        if FX.strength(fx, self._t(frame)) <= 0.0:
            return img
        cell, step, age, t, before = self.shapes_at(fx, frame)
        out = picture(img, cell, fx["layout"], fx["shape"], fx["size"], fx["turn"], fx["spin"], fx["inks"], fx["pick"],
                      fx["back"], fx["skip"], fx["pop"], fx["stagger"], fx["seed"], step, age, t, before)
        work = self._masked(fx, frame)
        if work is not None:
            out = (img + (out - img) * np.asarray(work, np.float32)[..., None]).astype(np.float32)
        return self._mix(fx, frame, img, out)
