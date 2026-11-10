"""Video stickers on the finished pictures (docs/design.md: Frame effects, Video stickers): the picture work of the `clip`
kind that mkmmd.core.videosticker checks and lays out. The clip's frame is fitted (cover) into the sticker's box at the
size the box shows at, carried into the frame with the box's outline and cut out by it (antialiased), and laid on the
picture under its die-cut edge and drop shadow (mkmmd.framefx `_grow` and `_cast`, as a collage's figure wears them),
all of it worked on the sticker's part of the frame alone. Images are float RGB in display space (0..1), (h, w, 3); pixel
(i, j) is centred on (i, j), as OpenCV's."""
import math

import cv2
import numpy as np

from . import mandala as MDF
from .core import framefx as FX
from .core import videosticker as VS

REACH = 4.0                      # a shadow's blur reaches this many of its sigmas
MARGIN = 3.0                     # px round the sticker, its edge and its shadow that its part of the frame takes in
SUPER = 4                        # an outline is filled at this many times the frame's resolution, then averaged down


def _ff():
    from . import framefx as FF          # framefx imports this module for StickerFx: its helpers are there by now
    return FF


def tile_map(M, box, shape):
    """The 2 x 3 map from the pixels of a tile of `shape` ((h, w, ...): the clip fitted into the box) to the frame's:
    the tile spread over the box (`box` (w, h) px at rest, about its middle), then carried by the box's map `M`."""
    bw, bh = float(box[0]), float(box[1])
    th, tw = shape[:2]
    return FX.compose(M, [[bw / tw, 0.0, 0.5 * bw / tw - bw / 2.0], [0.0, bh / th, 0.5 * bh / th - bh / 2.0]])


def inset(points, d):
    """The closed polygon `points` (n, 2) moved `d` px in along its normals, each corner by the miter that keeps both
    its sides `d` in (at most about 3 d at a spike); repeated points dropped."""
    p = np.asarray(points, np.float64)
    p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9]]
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) <= 1e-9:
        p = p[:-1]
    if len(p) < 3:
        return p
    e = np.roll(p, -1, axis=0) - p                                         # edge i runs from point i to point i + 1
    e /= np.hypot(e[:, 0], e[:, 1])[:, None]
    turn = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])   # > 0: clockwise on screen
    n = np.stack([-e[:, 1], e[:, 0]], 1) * (1.0 if turn > 0 else -1.0)     # each edge's normal, pointing in
    n0 = np.roll(n, 1, axis=0)                                             # the edge's before each point
    return p + d * (n0 + n) / np.maximum(1.0 + np.sum(n0 * n, 1), 0.25)[:, None]


def cover(points, size, origin=(0, 0)):
    """(h, w) float32 0..1 over a part of the frame of `size` (w, h) whose top left pixel is the frame's `origin`
    (x, y): the polygon `points` (frame px) filled, antialiased by its share of each pixel: filled SUPER x SUPER times
    finer and averaged down (OpenCV's own antialiased fill draws a line round the polygon too, half a pixel too big),
    moved half a finer pixel in first, as OpenCV fills every finer pixel whose centre lies within half a pixel of it."""
    w, h = int(size[0]), int(size[1])
    m = np.zeros((h * SUPER, w * SUPER), np.uint8)
    p = inset(((np.asarray(points, np.float64) - np.asarray(origin, np.float64)) + 0.5) * SUPER - 0.5, 0.5)
    cv2.fillPoly(m, [np.rint(p * 16.0).astype(np.int32)], 255, cv2.LINE_8, 4)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255.0)


def carried(src, A, size):
    """`src` carried by the 2 x 3 map `A` (its pixels to the result's) into a picture of `size` (w, h), shrunk by area
    first where A shrinks it (mkmmd.framefx.warp_spec), its edge repeated past it: an outline's antialiased rim along
    the box's sides takes the clip's own edge, not black."""
    src = np.ascontiguousarray(src, np.float32)
    Mf, shrink = _ff().warp_spec(A, src.shape)
    if shrink is not None:
        src = cv2.resize(src, shrink, interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(src, Mf, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def text_mask(text, font, px=160):
    """The letters of `text` (lines split at "\n", centred) in the font file `font`, `px` px a line, as float32 coverage
    0..1 cropped to the ink: the shape a text sticker shows its clip through."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(str(font), px)
    pad = px
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = probe.multiline_textbbox((0, 0), text, font=f, align="center")
    l, t, r, b = math.floor(l), math.floor(t), math.ceil(r), math.ceil(b)
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).multiline_text((pad - l, pad - t), text, font=f, fill=255, align="center")
    m = np.asarray(im, np.float32) / np.float32(255.0)
    ys, xs = np.nonzero(m > 0)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else m


def mask_cover(mask, M, box, size, origin):
    """The text mask spread over the box (`box` (w, h) px at rest), carried into the frame by its map `M` and cut to the
    part of the frame of `size` (w, h) whose top left pixel is `origin`: (h, w) float32 0..1, nothing outside the box."""
    A = np.asarray(tile_map(M, box, mask.shape), np.float64)
    A[:, 2] -= origin
    k = math.sqrt(abs(np.linalg.det(A[:, :2])))                  # px on screen per px of the mask
    if k < 1.0:                                                    # shrink by area first, as a tile is
        small = cv2.resize(mask, (max(int(round(mask.shape[1] * k)), 1), max(int(round(mask.shape[0] * k)), 1)),
                           interpolation=cv2.INTER_AREA)
        A = A @ np.array([[mask.shape[1] / small.shape[1], 0.0, 0.0], [0.0, mask.shape[0] / small.shape[0], 0.0],
                          [0.0, 0.0, 1.0]])
        mask = small
    return cv2.warpAffine(mask, A, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def lay(img, tile, M, box, shape, radius=0.0, blend="over", edge=None, shadow=None, scale=1.0, strength=1.0, mask=None):
    """`img` with a video sticker on it, a new picture: `tile` (the clip fitted into the sticker's box) spread over the
    box (`box` (w, h) px at rest) and carried into the frame by its map `M` (mkmmd.core.videosticker.place) with the
    box's outline (`shape`, a rect's corner `radius`; a text sticker's `mask`, its letters, text_mask), cut out by it;
    an `edge` (colour, width px) grown round it and a `shadow` (colour, (dx, dy) px, blur px, alpha) cast, both `scale`
    times as large; laid `over` the picture (the shape and its edge casting the shadow), or its face `screen`ed or
    `add`ed onto it, light whose black vanishes (the edge then stays paper, laid over, and the rim alone casts the
    shadow); a `negative` face (no `tile`) is the picture under it inverted, laid as `over` lays a clip; mixed in by
    `strength`."""
    FF = _ff()
    M = np.asarray(M, np.float64)
    if mask is None:
        pts = VS.outline(shape, box[0], box[1], radius, VS.STEP / max(scale, 1e-3))  # its points STEP px apart on screen
    else:
        pts = np.array([[-box[0], -box[1]], [box[0], -box[1]], [box[0], box[1]], [-box[0], box[1]]]) / 2.0
    P = pts @ M[:, :2].T + M[:, 2]
    ew = 0.0 if edge is None or edge[1] <= 0 else float(edge[1]) * scale
    opaque = blend in ("over", "negative")
    cast = shadow is not None and shadow[3] > 0 and (opaque or ew > 0)
    reach = (math.hypot(*shadow[1]) + REACH * shadow[2]) * scale if cast else 0.0
    pad = ew + reach + MARGIN
    H, W = img.shape[:2]
    X0, Y0 = max(int(math.floor(P[:, 0].min() - pad)), 0), max(int(math.floor(P[:, 1].min() - pad)), 0)
    X1, Y1 = min(int(math.ceil(P[:, 0].max() + pad)) + 1, W), min(int(math.ceil(P[:, 1].max() + pad)) + 1, H)
    if X1 <= X0 or Y1 <= Y0:
        return img
    size = (X1 - X0, Y1 - Y0)
    cov = cover(P, size, (X0, Y0)) if mask is None else mask_cover(mask, M, box, size, (X0, Y0))
    if blend == "negative":
        rgb = np.clip(1.0 - img[Y0:Y1, X0:X1], 0.0, 1.0)                 # the picture under it, before its edge and shadow
    else:
        A = np.asarray(tile_map(M, box, tile.shape), np.float64)
        A[:, 2] -= (X0, Y0)
        rgb = carried(tile, A, size)
    base = img[Y0:Y1, X0:X1]
    region = base
    ring = FF._grow(cov, ew) if ew > 0 else None
    paper = ring if ring is None or opaque else np.clip(ring - cov, 0.0, 1.0)    # over a face of light: the rim
    if cast:
        region = FF._cast(region, cov if paper is None else paper, shadow, scale)
    if paper is not None:
        g = paper[..., None]
        region = region * (1.0 - g) + np.asarray(edge[0], np.float32) * g
    region = MDF.blend(region, rgb, cov[..., None], "over" if blend == "negative" else blend)
    out = np.array(img, np.float32, copy=True)
    out[Y0:Y1, X0:X1] = region if strength >= 1.0 else base + (region - base) * np.float32(strength)
    return out


# ================================================================================================= over a cut
class StickerFx:
    """The video stickers' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, _t, _edge, _shadow and
    _footage_of)."""

    def sticker_at(self, fx, frame):
        """(M, box (w, h) px, scale, strength) of a video sticker on `frame`: where it stands (mkmmd.core.videosticker
        .place) and how much of it is mixed in."""
        t = self._t(frame)
        ratio = None
        if fx["shape"] == "text":
            m = self._letters(fx)
            ratio = m.shape[1] / m.shape[0]
        M, box, scale = VS.place(fx, t, self.size, ratio)
        return M, box, scale, float(np.clip(FX.strength(fx, t), 0.0, 1.0))

    def _letters(self, fx):
        """A text sticker's letter mask (text_mask), made once."""
        memo = self.__dict__.setdefault("_letter_masks", {})
        if fx["index"] not in memo:
            memo[fx["index"]] = text_mask(fx["text"], self._font_of(fx["font"]))
        return memo[fx["index"]]

    def _clip(self, fx, frame, img):
        """The clip's picture at this frame (its `crop` of it) fitted (cover) into the sticker's box at the size it shows
        at (none for a negative sticker, whose face is the picture under it), laid on the picture in its shape, edge and
        shadow (lay)."""
        M, box, scale, s = self.sticker_at(fx, frame)
        if s <= 0.0:
            return img
        tile = None
        if fx["blend"] != "negative":
            if self._footage_of is None:
                raise FileNotFoundError(f"fx {fx['index']} (clip): no footage to read")
            clip = _ff().clip_of(fx, self._footage_of)
            tile = clip.picture((frame - fx["f0"]) / self.fps, (max(int(round(box[0] * scale)), 1),
                                                                 max(int(round(box[1] * scale)), 1)),
                                fx["in"], fx["rate"], fx["loop"], "cover", fx["focus"], fx["mirror"], crop=fx["crop"])
        return lay(img, tile, M, box, fx["shape"], fx["radius"], fx["blend"], self._edge(fx), self._shadow(fx), scale, s,
                   self._letters(fx) if fx["shape"] == "text" else None)
