"""Mitosis on the finished pictures (docs/design.md: Frame effects, Cell division): the picture work of the `[[fx]]` kind
`mitosis` that mkmmd.core.mitosis checks, times and lays out. MitosisFx is the part of mkmmd.framefx.FrameFx that draws
it: the figure cut out of the picture, its place there closed from the ground round it, a copy of it in every cell, and
the cells' membrane: the outlines of all the copies blurred and summed, drawn where the sum passes a level, so two cells
that are still close share one skin and the skin pinches as they part. Images are float RGB in display space (0..1),
(h, w, 3)."""
import math

import cv2
import numpy as np

from .core import framefx as FX
from .core import mitosis as MT

FIELD = 4                       # the membrane's field is worked out at this share of the frame's size (one in four)
ISO = 0.2                       # the membrane lies where the blurred outlines sum to this
SOFT = 0.05                     # ... the cytoplasm's edge this wide in the field's own units
LINE = 0.03                     # ... and the membrane's line this wide
SHINE = 0.55                    # the light on the membrane's upper left, at its brightest
LIGHT = (-0.6, -0.8)            # where that light comes from (x right, y down)
MARGIN = 2                      # px round the figure's box that are cut out with it


def ground(img, m):
    """`img` with the figure (coverage `m`, (h, w) 0..1) closed from what is round it: averaged down level by level
    counting the ground alone, each level's holes then filled from the level above (push-pull). A plain ground closes
    without a seam, a busy one as a blur."""
    a = (1.0 - m).astype(np.float32)
    c = img * a[..., None]
    levels = []
    while min(a.shape) > 2:
        levels.append((c, a))
        size = ((a.shape[1] + 1) // 2, (a.shape[0] + 1) // 2)
        c, a = cv2.resize(c, size, interpolation=cv2.INTER_AREA), cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    fill = c / np.maximum(a, 1e-4)[..., None]
    for c, a in reversed(levels):
        up = cv2.resize(fill, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        fill = c + up * (1.0 - a)[..., None]
    return (img + (fill - img) * m[..., None]).astype(np.float32)


def _maps(centre, at, scale, angle, stretch):
    """The 2 x 2 part and the shift of each cell's map from the figure's own pixels: scaled about `centre`, drawn long
    along its angle, moved to `at` (px)."""
    out = []
    for i in range(len(scale)):
        ca, sa = math.cos(angle[i]), -math.sin(angle[i])
        turn = np.array([[ca, -sa], [sa, ca]])
        A = turn @ np.diag([scale[i] * (1.0 + stretch[i]), scale[i] * (1.0 - 0.5 * stretch[i])]) @ turn.T
        out.append((A, np.asarray(at[i], float) - A @ np.asarray(centre, float)))
    return out


def _warp(src, origin, A, shift, size):
    """(piece, (x0, y0)): `src` (a crop whose corner is at `origin` in the frame) through the map x -> A x + shift, the
    part of it that lands in a frame of `size` (w, h); None when none does."""
    h, w = src.shape[:2]
    corners = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]]) + np.asarray(origin, float)
    to = corners @ A.T + shift
    x0, y0 = max(int(math.floor(to[:, 0].min())) - 1, 0), max(int(math.floor(to[:, 1].min())) - 1, 0)
    x1, y1 = min(int(math.ceil(to[:, 0].max())) + 1, size[0]), min(int(math.ceil(to[:, 1].max())) + 1, size[1])
    if x0 >= x1 or y0 >= y1:
        return None
    M = np.c_[A, A @ np.asarray(origin, float) + shift - np.array([x0, y0], float)].astype(np.float32)
    return cv2.warpAffine(src, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR), (x0, y0)


def divide(img, m, box, at, scale, angle, stretch, fx, back):
    """The picture of a mitosis: `back` (h, w, 3), the cells' cytoplasm and membrane, and a copy of the figure (`img`
    under its coverage `m`, cut out by its `box` (x0, y0, x1, y1) px) in every cell: `at` (n, 2) px the cells' middles,
    `scale`, `angle` and `stretch` (n,) as mkmmd.core.mitosis.cells gives them, the lowest cell drawn last."""
    h, w = m.shape
    short = min(w, h)
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(x0 - MARGIN, 0), max(y0 - MARGIN, 0), min(x1 + MARGIN, w), min(y1 + MARGIN, h)
    pic, cov = np.ascontiguousarray(img[y0:y1, x0:x1]), np.ascontiguousarray(m[y0:y1, x0:x1])
    centre = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    hs, ws = max(h // FIELD, 2), max(w // FIELD, 2)
    small = cv2.resize(cov, (max((x1 - x0) // FIELD, 1), max((y1 - y0) // FIELD, 1)), interpolation=cv2.INTER_AREA)
    k = np.array([ws / w, hs / h])                               # frame px -> field px
    field = np.zeros((hs, ws), np.float32)
    out = back.astype(np.float32).copy()
    maps = _maps(centre, at, scale, angle, stretch)
    pieces = []
    for i in np.argsort(at[:, 1], kind="stable"):
        A, shift = maps[i]
        got = _warp(small, (x0 * k[0], y0 * k[1]), np.diag(k) @ A @ np.diag(1.0 / k), shift * k, (ws, hs))
        if got is not None:
            piece, (px, py) = got
            field[py:py + piece.shape[0], px:px + piece.shape[1]] += piece
        pieces.append((_warp(pic, (x0, y0), A, shift, (w, h)), _warp(cov, (x0, y0), A, shift, (w, h))))
    sigma = max(fx["body"] * short * float(np.mean(scale)) / FIELD, 0.5)
    field = cv2.GaussianBlur(field, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    gx, gy = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    lit = np.maximum(-(gx * LIGHT[0] + gy * LIGHT[1]) / np.maximum(np.hypot(gx, gy), 1e-6), 0.0)   # light on its way in
    F = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    lit = cv2.resize(lit, (w, h), interpolation=cv2.INTER_LINEAR)
    inside = np.clip((F - ISO) / SOFT, 0.0, 1.0)
    inside = inside * inside * (3.0 - 2.0 * inside)
    line = np.exp(-((F - ISO) / LINE) ** 2)
    out += (np.asarray(fx["cytoplasm"], np.float32) - out) * (np.float32(fx["alpha"]) * inside)[..., None]
    covered = np.zeros((h, w), np.float32)
    for got_pic, got_cov in pieces:
        if got_pic is None:
            continue
        piece, (px, py) = got_pic
        a = got_cov[0]
        roi = out[py:py + a.shape[0], px:px + a.shape[1]]
        roi += (piece - roi) * a[..., None]
        covered[py:py + a.shape[0], px:px + a.shape[1]] = np.maximum(covered[py:py + a.shape[0], px:px + a.shape[1]], a)
    bare = 1.0 - covered
    out += (np.asarray(fx["membrane"], np.float32) - out) * (0.9 * line * bare)[..., None]
    out += (1.0 - out) * (np.float32(SHINE) * line * lit * lit * bare)[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class MitosisFx:
    """The mitosis's part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, frame0, _t, _hit_times,
    _figure, and for a figure from a `shot` of its own _shot_frame and _posed, as a collage does)."""

    def _mitosis_init(self):
        self._mito = {}                  # a mitosis's index -> its figure's box on the span's first frame (px)

    def mitosis_box(self, fx):
        """(x0, y0, x1, y1) px: the figure's box on the span's first frame it is in the frame on, which sizes the cells
        and their steps for the whole span (her arm moving does not shake the colony); None when it never is."""
        if fx["index"] not in self._mito:
            got = None
            for g in range(fx["f0"], fx["f1"]):
                m = self._mitosis_figure(fx, g)[1]
                if m is not None and (m >= 0.5).any():
                    ys, xs = np.nonzero(m >= 0.5)
                    got = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                    break
            self._mito[fx["index"]] = got
        return self._mito[fx["index"]]

    def _mitosis_figure(self, fx, g, img=None):
        """(picture, coverage) of a mitosis's figure on frame `g` as it stands in the frame: cut from `img` (the cut's
        frame there; None: its coverage alone) or, with a `shot`, from what that shot shows, and set in by `at` and
        `height` as a collage's figure is."""
        m = self._figure(fx, g)
        if m is None or not (fx.get("shot") or FX.placed(fx)):
            return img, m
        return self._posed(fx, g, None if img is None else self._shot_frame(fx, g) if fx.get("shot") else img, m)

    def mitosis_at(self, fx, frame):
        """(at (n, 2) px, scale, angle, stretch) of a mitosis's cells on `frame`, fitted into the frame; None when the
        figure is never in the frame."""
        box = self.mitosis_box(fx)
        if box is None:
            return None
        w, h = self.size
        short = float(min(w, h))
        hits = [(FX.frame_of(x, self.fps, self.frame0) - self.frame0) / self.fps for x in self._hit_times(fx)]
        gen, phase, since = MT.state(fx, self._t(frame), hits)
        size = ((box[2] - box[0]) / short, (box[3] - box[1]) / short)
        centre = np.array([0.5 * (box[0] + box[2]) / short, 0.5 * (box[1] + box[3]) / short])
        at, scale, angle, stretch = MT.cells(fx, gen, phase, since, size)
        k, shift = MT.fit(at, scale, stretch, size, centre, (w / short, h / short), fx["fit"])
        return (centre + k * at + shift) * short, scale * k, angle, stretch

    def _mitosis(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        pic, m = self._mitosis_figure(fx, frame, img) if s > 0.0 else (img, None)
        cells = self.mitosis_at(fx, frame) if m is not None else None
        if cells is None or not (m >= 0.5).any():
            return img
        if fx["back"] != MT.PICTURE:
            back = np.broadcast_to(np.asarray(fx["back"], np.float32), img.shape)
        else:                                # her place in the picture closed: a figure from a plate was never in it
            back = img if fx.get("shot") else ground(img, self._figure(fx, frame))
        ys, xs = np.nonzero(m >= 0.5)
        now = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        box = self.mitosis_box(fx)
        move = np.array([0.5 * (now[0] + now[2] - box[0] - box[2]), 0.5 * (now[1] + now[3] - box[1] - box[3])])
        out = divide(pic, m, now, cells[0] + move * cells[1][:, None], cells[1], cells[2], cells[3], fx, back)
        return out if s >= 1.0 else (img + (out - img) * np.float32(s)).astype(np.float32)
