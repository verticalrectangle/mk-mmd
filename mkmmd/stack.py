"""Stacks on the finished pictures (docs/design.md: Frame effects, Rainbow stacks): the picture work of the `[[fx]]` kind
`stack`, which mkmmd.core.stack checks and lays out in time. `stack` is the plain banded echo as a pure function; StackFx
is the part of mkmmd.framefx.FrameFx that draws every stack.

The figure and each copy is a Layer: its coverage in the figure's own frame (frame-sized, as mk render draws it), the
picture its fill draws from (hers, a wallpaper of sprites, or none for a flat colour), its fill, the affine map that lays
it into the picture (the stack's step and scale, a tunnel's depth, a peel's slide, a squash about its feet), its opacity
and its era's look. `lay` draws one within the box it covers (as mkmmd.framefx._paste draws a figure over the whole
frame, the same pixels): its fill, its look, its die-cut edge, its drop shadow. A look is drawn where the copy lands, its
dots, lines and blocks sized with the copy (never under a few pixels) and fixed to it as it moves. Images are float RGB in
display space (0..1), (h, w, 3); a coverage is (h, w) in 0..1."""
import math
from collections import OrderedDict, namedtuple
from functools import lru_cache

import cv2
import numpy as np

from . import eralook as ERA
from . import eyes as EYD
from . import mandala as MDF
from . import picturefx as PFX
from .core import framefx as FX
from .core import stack as SK

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
WALLS = 32                       # wallpaper tiles kept from frame to frame
SPACING = 1.3                    # a wallpaper's sprites stand this many of their sizes apart, in a staggered lattice
COLS, ROWS = 4, 4                # sprites across and down one tile of a wallpaper (ROWS even: the rows stagger)
TURN = 16.0                      # degrees a wallpaper's sprite is turned at most, either way
JITTER = 0.12                    # how much larger or smaller a wallpaper's sprite is drawn at most
SPRITE_EDGE = 3.0                # px at 1080 on the short side: the white die-cut round a wallpaper's sprite
WHITE = (1.0, 1.0, 1.0)
SHARE = 0.3                      # a look is sized with its copy down to this share of its size, and no smaller
PAPER = (0.97, 0.94, 0.86)       # halftone: the newsprint
DOTS = (10.0, 15.0)              # halftone: px at 1080 between the dots, the screen's angle (degrees)
INK_SAT = (1.6, 0.2)             # halftone: an ink's saturation, its colour's times this and plus that
INK_DARK = 0.88                  # halftone: an ink's brightness, its colour's times this
DOT_AREA = (0.1, 0.72)           # halftone: the least and most of its cell a dot covers (they never quite merge)
FLAT = {"smooth": 9.0, "line": 7.0, "inner": 3.0, "hues": 12, "values": (0.3, 0.62, 0.95), "sat": 0.85}
INK = (0.13, 0.07, 0.2)          # flat: the line (Peter Max's black, a little purple)
TAPE = {"tracking": 0.0, "chroma": 7.0, "scanlines": 0.35, "jitter": 1.2, "soft": 1.4, "saturation": 0.85,
        "noise": 0.05, "seed": 7}      # vhs: a camcorder's tape (px at 1080 at the copy's full size), no tracking band
GHOST = 0.55                     # vhs: the colour bled past the copy's right edge, this opaque
PIXEL = (12.0, 4)                # pixel: px at 1080 a block is across, levels per colour channel
CHROME_DRIFT = 0.15              # chrome: bands a second its reflection drifts
CLEAN = 4                        # a squashed figure's own place is filled in at this share (1 / CLEAN) of the size
REACH = 20.0                     # px at 1080 on the short side, times the copy's size: how far a look reaches out of it

# one thing a stack draws: `pic` the picture its fill draws from ((h, w, 3) of the figure's frame, a painter
# callable(box) -> its part over box, or None), `m` its coverage there, `fill` (None: the picture; a colour; a silkscreen
# {inks, offset px, soft px, line px}), `M` the affine map (2 x 3) from the figure's frame to the picture's, `alpha` its
# opacity, `look` an era's medium (mkmmd.core.stack.LOOKS; None: none), `thin` whether its edge is sized with it (a
# tunnel's copies, far away) or the width given
Layer = namedtuple("Layer", "pic m fill M alpha look thin", defaults=(1.0, None, False))


def _ff():
    from . import framefx as FF          # framefx imports this module for StackFx: its helpers are there by now
    return FF


# ================================================================================================= laying
def scaled(s, a, p):
    """The affine map that scales `s` times about pixel `a` and puts `a` at pixel `p`."""
    return ((s, 0.0, p[0] - s * a[0]), (0.0, s, p[1] - s * a[1]))


def compose(a, b):
    """The affine map `a` after `b` (2 x 3 each)."""
    return tuple(tuple(v) for v in FX.compose(a, b))


def mixed(a, b, e):
    """The affine map `e` of the way from `a` to `b`, entry by entry (for maps that only scale and move: the scale and
    where any point lands, both that far)."""
    return tuple(tuple(x + (y - x) * e for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def stack(img, m, back, copies, step, scale, centre, fill, edge=None, shadow=None, mirror=False):
    """The figure (coverage `m` of `img`, drawn `fill`: None its picture) on `back` (a colour, a pattern's painter or a
    picture), in front of a stack of copies, the farthest drawn first: copies[j - 1] = (picture, coverage, fill) is copy
    j, `scale` ** j times as large about pixel `centre` and j x `step` (dx, dy) px further; a copy without coverage is
    left out. Each with its `edge` and `shadow`. With `mirror` the stack is drawn again over all that, reflected left for
    right about the vertical line through `centre` (put on the nearest half pixel: its figure is the same pixels
    flipped): its copies step the mirrored way, so both echoes trail outward, and its edges, shadows (one falling down
    and right falls down and left) and silkscreen key plates are flipped with it. The back is not reflected."""
    h, w = m.shape
    c = (float(centre[0]), float(centre[1]))
    layers = [Layer(pic, cov, f, scaled(scale ** j, c, (c[0] + j * step[0], c[1] + j * step[1])))
              for j, (pic, cov, f) in reversed(list(enumerate(copies, 1))) if cov is not None]
    return draw(_ff()._flat((w, h), back), layers + [Layer(img, m, fill, IDENTITY)], edge, shadow,
                c[0] if mirror else None)


def draw(out, layers, edge=None, shadow=None, mirror=None, px=1.0, t=0.0):
    """`out` (in place, and returned) with `layers` laid on it in order, each over those before it (the farthest first),
    each with its `edge` (colour, width px) and `shadow` (colour, (dx, dy) px, blur px, alpha); with `mirror` (the x px
    of a vertical line) all of them again over that, reflected left for right about it (put on the nearest half pixel,
    so a figure's reflection is its own pixels flipped), its shadows falling the mirrored way. `px` (the short side over
    1080) and `t` (clip seconds) size and time the looks."""
    seen = {}
    for L in layers:
        lay(out, L, edge, shadow, px=px, t=t, seen=seen)
    if mirror is None:
        return out
    X = int(round(2.0 * mirror))                                      # x -> X - x: whole pixels, nothing resampled
    sh = None if shadow is None else (shadow[0], (-shadow[1][0], shadow[1][1]), shadow[2], shadow[3])
    for L in layers:
        lay(out, L, edge, sh, X=X, px=px, t=t, seen=seen)
    return out


def support(m, seen=None):
    """(x0, y0, x1, y1) px of all that coverage `m` covers at all (x1, y1 past it), None for nothing; kept in `seen` (by
    the coverage, for the frame) when given."""
    got = None if seen is None else seen.get(id(m))
    if got is not None and got[0] is m:
        return got[1]
    x, y, w, h = cv2.boundingRect((np.asarray(m) > 0).astype(np.uint8))
    box = None if w == 0 or h == 0 else (x, y, x + w, y + h)
    if seen is not None:
        seen[id(m)] = (m, box)
    return box


def feet(m, seen=None):
    """Where a coverage stands (x, y px): the middle of its box (where it covers half a pixel or more) along the box's
    bottom edge; None for nothing."""
    got = None if seen is None else seen.get(("feet", id(m)))
    if got is not None and got[0] is m:
        return got[1]
    x, y, w, h = cv2.boundingRect((np.asarray(m) >= 0.5).astype(np.uint8))
    at = None if w == 0 or h == 0 else (x + (w - 1) / 2.0, y + h - 0.5)
    if seen is not None:
        seen[("feet", id(m))] = (m, at)
    return at


def warped(src, at, frame, M, origin, size):
    """What mkmmd.framefx._warp draws of a frame-sized picture (`frame` (w, h)) that is `src` with its top left at whole
    pixel `at` and nothing round it, carried by the affine map `M` (2 x 3, the frame's px to the picture's) into the box
    of `size` (w, h) at `origin` (x, y px) of the picture: shrunk by area first on the whole frame's grid when M shrinks
    it under SHRINK_BY_AREA, its map cast to float32 as _warp casts it. The same pixels, worked on the box alone."""
    FF = _ff()
    M = np.asarray(M, np.float64)
    s = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    src, x0, y0 = np.ascontiguousarray(src, np.float32), at[0], at[1]
    if s < FF.SHRINK_BY_AREA:
        w, h = frame
        k = np.array([max(int(round(w * s)), 1) / w, max(int(round(h * s)), 1) / h])
        i0, j0 = int(math.floor(x0 * k[0])), int(math.floor(y0 * k[1]))
        i1 = min(int(math.ceil((x0 + src.shape[1]) * k[0])), int(round(w * k[0])))
        j1 = min(int(math.ceil((y0 + src.shape[0]) * k[1])), int(round(h * k[1])))
        src = _area(src, (x0, y0), k, (i0, j0, i1, j1))
        M = np.hstack([M[:, :2] / k, (M[:, 2] + M[:, :2] @ (0.5 / k - 0.5))[:, None]])
        x0, y0 = i0, j0
    M = M.astype(np.float32).astype(np.float64)
    M[:, 2] += M[:, :2] @ np.array([x0, y0], np.float64) - np.array(origin, np.float64)
    return cv2.warpAffine(src, M, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _area(src, at, k, cells):
    """Columns i0 .. i1 - 1 and rows j0 .. j1 - 1 (`cells`) of a frame shrunk by area to `k` (kx, ky) times its size, as
    cv2.INTER_AREA shrinks a whole frame (each pixel the mean over the frame's pixels it covers, parts of them at its
    edges), from `src`: the frame at whole pixel `at` with nothing round it."""
    def along(a, axis, n0, n1, kk, off):
        n = a.shape[axis]
        a = np.moveaxis(np.asarray(a, np.float64), axis, 0)
        e = np.clip(np.arange(n0, n1 + 1, dtype=np.float64) / kk - off, 0.0, n)     # the cells' edges in `a`
        F = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
        lo = np.minimum(np.floor(e).astype(np.int64), n - 1)
        Fe = F[lo] + (e - lo).reshape((-1,) + (1,) * (a.ndim - 1)) * a[lo]
        return np.moveaxis((Fe[1:] - Fe[:-1]) * kk, 0, axis)
    i0, j0, i1, j1 = cells
    return along(along(src, 1, i0, i1, k[0], at[0]), 0, j0, j1, k[1], at[1]).astype(np.float32)


def lay(out, layer, edge=None, shadow=None, X=None, px=1.0, t=0.0, seen=None):
    """`out` (in place, and returned) with one Layer on it: its coverage carried by its map into the picture, drawn as
    its fill says (its picture, a flat colour or a silkscreen print) in its look, over its `edge` (colour, width px)
    grown round it, the two casting its `shadow` (colour, (dx, dy) px, blur px, alpha; offset and blur scaled with it),
    all at its opacity. With `X` (whole px) it is first reflected left for right about x = X / 2 (pixel x shows pixel X -
    x of it) and its map with it, a silkscreen's key plate printed off the other way. Worked on the box it covers only:
    without a look the same pixels as mkmmd.framefx._paste."""
    FF = _ff()
    pic, m, fill, M, alpha, look, thin = layer
    box = None if alpha <= 0.0 else support(m, seen)
    if box is None:
        return out
    silk = isinstance(fill, dict)
    x0, y0, x1, y1 = box
    if silk:                                                    # a silkscreen's blurs reach this far round the figure
        g = int(math.ceil(3.0 * (fill["soft"] + fill["line"]))) + 1
        x0, y0, x1, y1 = max(x0 - g, 0), max(y0 - g, 0), min(x1 + g, m.shape[1]), min(y1 + g, m.shape[0])
    mc = m[y0:y1, x0:x1]
    pc = None
    if fill is None or silk:
        pc = pic((x0, y0, x1, y1)) if callable(pic) else pic[y0:y1, x0:x1]
    A = np.asarray(M, np.float64).reshape(2, 3)
    frame = (m.shape[1], m.shape[0])
    if X is not None:                                           # the reflection: its pixels flipped, its map with them
        mc, pc = mc[:, ::-1], None if pc is None else pc[:, ::-1]
        x0, x1 = X - (x1 - 1), X - x0 + 1
        lo, hi = max(-x0, 0), max(x1 - frame[0], 0)             # what lands past the frame's edges is not there
        mc, pc = mc[:, lo:mc.shape[1] - hi], None if pc is None else pc[:, lo:pc.shape[1] - hi]
        x0, x1 = x0 + lo, x1 - hi
        if x1 <= x0:
            return out
        A = np.array([[A[0, 0], -A[0, 1], X - A[0, 0] * X - A[0, 2]], [-A[1, 0], A[1, 1], A[1, 2] + A[1, 0] * X]])
        if silk:
            fill = {**fill, "offset": (-fill["offset"][0], fill["offset"][1])}
    s = math.sqrt(abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))
    ew = 0.0 if edge is None or edge[1] <= 0 else float(edge[1]) * (s if thin else 1.0)
    cast = shadow is not None and shadow[3] > 0
    reach = (math.hypot(*shadow[1]) + 4.0 * shadow[2]) * s if cast else 0.0
    pad = 2.0 * ew + reach + (REACH * px * max(s, SHARE) if look not in (None, "none") else 0.0) + 3.0
    pts = np.array([[x0 - 0.5, y0 - 0.5], [x1 - 0.5, y0 - 0.5], [x0 - 0.5, y1 - 0.5], [x1 - 0.5, y1 - 0.5]]) @ A[:, :2].T \
        + A[:, 2]
    H, W = out.shape[:2]
    X0, Y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
    X1, Y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, W), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, H)
    if X1 <= X0 or Y1 <= Y0:
        return out
    size = (X1 - X0, Y1 - Y0)

    def warp(a):
        return warped(a, (x0, y0), frame, A, (X0, Y0), size)
    cov = warp(mc)
    if silk:
        key, steps = FF.silk_bands(pc, mc, len(fill["inks"]), fill["soft"], fill["line"])
        k = warp(key)
        rgb = FF.silk_print(np.zeros((size[1], size[0], 3), np.float32), cov, k, [warp(x) for x in steps],
                            fill["inks"], fill["offset"])
        off = fill["offset"]
        k = FF.plate_shift(k, off[0], off[1]) if off[0] or off[1] else k
        a = cov + k - cov * k
    elif fill is None:
        rgb, a = warp(pc * mc[..., None]), cov
    else:
        rgb, a = np.asarray(fill, np.float32) * cov[..., None], cov
    base = cov
    if look not in (None, "none"):
        rgb, a = LOOKS[look](rgb, a, max(s, SHARE) * px, (A[0, 2] - X0, A[1, 2] - Y0), t)
        base = a
    region = out[Y0:Y1, X0:X1]
    ring = FF._grow(base, ew) if ew > 0 else None
    if cast:
        region = FF._cast(region, base if ring is None else ring, (shadow[0], shadow[1], shadow[2], shadow[3] * alpha), s)
    if ring is not None:
        g = (ring if alpha >= 1.0 else ring * np.float32(alpha))[..., None]
        region = region * (1.0 - g) + np.asarray(edge[0], np.float32) * g
    if alpha < 1.0:
        rgb, a = rgb * np.float32(alpha), a * np.float32(alpha)
    out[Y0:Y1, X0:X1] = rgb + region * (1.0 - a[..., None])
    return out


# ================================================================================================= the looks
def _colour(rgb, a):
    """The colour under premultiplied `rgb` (coverage `a`), filled out past the edge from round it (its colours carried
    out a few px) so nothing dark bleeds in when it is blurred or moved."""
    den = cv2.GaussianBlur(a, (0, 0), 3.0) + np.float32(1e-6)
    out = cv2.GaussianBlur(rgb, (0, 0), 3.0) / den[..., None]
    inside = a > 1e-3
    out[inside] = rgb[inside] / a[inside][:, None]
    return np.clip(out, 0.0, 1.0)


def _grid(shape, origin, cell, angle):
    """(u, v) of each pixel on a screen of `cell` px turned `angle` degrees about `origin` (x, y px), fractions of a cell
    from the nearest of its points."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx -= np.float32(origin[0])
    yy -= np.float32(origin[1])
    ca, sa = np.float32(math.cos(math.radians(angle))), np.float32(math.sin(math.radians(angle)))
    u = (xx * ca + yy * sa) / np.float32(cell)
    v = (yy * ca - xx * sa) / np.float32(cell)
    return u - np.rint(u), v - np.rint(v)


def halftone(rgb, a, k, origin, t):
    """60s print: the copy's colours as Ben-Day dots of ink on newsprint, each dot its colour's own hue made an ink
    (INK_SAT more saturated, INK_DARK as bright), as much of its cell as the colour is dark against the paper (within
    DOT_AREA), on a screen DOTS[0] x `k` px apart (k: px at 1080 times the copy's size) turned DOTS[1] degrees, fixed to
    the copy at `origin`."""
    c = _colour(rgb, a)
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.float32(INK_SAT[0]) + np.float32(INK_SAT[1]), 0.0, 1.0)
    hsv[..., 2] *= np.float32(INK_DARK)
    ink = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    paper = np.asarray(PAPER, np.float32)
    lp = np.float32(paper @ LUMA)
    d = np.clip((lp - c @ LUMA) / np.maximum(lp - ink @ LUMA, np.float32(1e-3)), DOT_AREA[0], DOT_AREA[1])
    cell = max(DOTS[0] * k, 3.0)
    u, v = _grid(a.shape, origin, cell, DOTS[1])
    r = np.sqrt(d / np.float32(math.pi)) * np.float32(cell)
    dot = np.clip(r - np.sqrt(u * u + v * v) * np.float32(cell) + np.float32(0.5), 0.0, 1.0)[..., None]
    out = paper * (1.0 - dot) + ink * dot
    return (out * a[..., None]).astype(np.float32), a


def flat(rgb, a, k, origin, t):
    """Peter Max's poster: the copy's colours smoothed into flat shapes, each pushed to one of FLAT's vivid hues at one
    of its values, every shape and the copy's own rim drawn round in a thick INK line (sized with the copy)."""
    c = _colour(rgb, a)
    n = max(int(round(FLAT["smooth"] * k)) | 1, 3)
    c8 = cv2.medianBlur((c * 255.0 + 0.5).astype(np.uint8), n)
    hsv = cv2.cvtColor(c8.astype(np.float32) / np.float32(255.0), cv2.COLOR_RGB2HSV)
    step = 360.0 / FLAT["hues"]
    hsv[..., 0] = (np.rint(hsv[..., 0] / step) * step) % 360.0
    grey = hsv[..., 1] < 0.12
    hsv[..., 1] = np.where(grey, 0.0, FLAT["sat"])
    vals = np.asarray(FLAT["values"], np.float32)
    hsv[..., 2] = vals[np.clip(np.searchsorted((vals[1:] + vals[:-1]) / 2.0, hsv[..., 2]), 0, len(vals) - 1)]
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    q = (out * 255.0 + 0.5).astype(np.int32)
    seam = np.zeros(a.shape, np.uint8)
    seam[:, 1:] |= (q[:, 1:] != q[:, :-1]).any(-1)
    seam[1:, :] |= (q[1:, :] != q[:-1, :]).any(-1)
    inner = cv2.distanceTransform(1 - seam, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rim = cv2.distanceTransform((a >= 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    line = np.maximum(np.clip(np.float32(max(FLAT["line"] * k, 1.5) + 0.5) - rim, 0.0, 1.0),
                      np.clip(np.float32(max(FLAT["inner"] * k, 1.0) * 0.5 + 0.5) - inner, 0.0, 1.0))[..., None]
    out = out * (1.0 - line) + np.asarray(INK, np.float32) * line
    return (out * a[..., None]).astype(np.float32), a


@lru_cache(maxsize=1)
def _tape():
    """The tape a "vhs" copy plays on: a camcorder entry of TAPE's keys (mkmmd.core.framefx), as mkmmd.picturefx.tape
    takes it."""
    return {**FX.normalize({"kind": "camcorder", "from": 0.0, "to": 1.0, **TAPE}, {}), "index": 0}


def vhs(rgb, a, k, origin, t):
    """80s video tape (mkmmd.picturefx.tape, sized with the copy): soft luma, the colour bled sideways and to the right
    and past the copy's edge, a little washed out, noise in streaks, lines jittered sideways, scanlines."""
    c = _colour(rgb, a)
    out = PFX.tape(c, _tape(), int(round(t * 1000.0)), t, k)
    r = TAPE["chroma"] * k
    ghost = cv2.blur(a, (max(int(round(2.0 * r)), 1), 1))
    ghost = cv2.warpAffine(ghost, np.float32([[1, 0, PFX.CHROMA_SHIFT * r], [0, 1, 0]]), (a.shape[1], a.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a2 = np.maximum(a, np.float32(GHOST) * ghost)
    return (out * a2[..., None]).astype(np.float32), a2


def pixel(rgb, a, k, origin, t):
    """00s pixels: the copy in blocks of PIXEL[0] x `k` px (whole px, the grid fixed to the copy at `origin`), each the
    mean of what it holds in PIXEL[1] levels a colour channel, there or not as it is covered at least half."""
    b = max(int(round(PIXEL[0] * k)), 2)
    h, w = a.shape
    ox, oy = (-int(round(origin[0]))) % b, (-int(round(origin[1]))) % b
    H, W = -(-(h + oy) // b) * b, -(-(w + ox) // b) * b
    big = np.zeros((H, W, 4), np.float32)
    big[oy:oy + h, ox:ox + w, :3], big[oy:oy + h, ox:ox + w, 3] = rgb, a
    blocks = big.reshape(H // b, b, W // b, b, 4).mean(axis=(1, 3))
    cov = (blocks[..., 3] >= 0.5).astype(np.float32)
    lv = PIXEL[1] - 1
    col = np.rint(np.clip(blocks[..., :3] / np.maximum(blocks[..., 3:], 1e-6), 0.0, 1.0) * lv) / lv
    up = lambda x: np.repeat(np.repeat(x, b, 0), b, 1)[oy:oy + h, ox:ox + w]       # noqa: E731
    a2 = up(cov)
    return (up(col) * a2[..., None]).astype(np.float32), a2


def chrome(rgb, a, k, origin, t):
    """Now: the copy as iridescent liquid chrome (mkmmd.eralook.chrome_sheen, a puffy dome sized with the copy), its bands
    of sky and horizon drifting CHROME_DRIFT bands a second."""
    return (ERA.chrome_sheen(_colour(rgb, a), cov=a, phase=CHROME_DRIFT * t, px=k) * a[..., None]).astype(np.float32), a


LOOKS = {"halftone": halftone, "flat": flat, "vhs": vhs, "pixel": pixel, "chrome": chrome}


# ================================================================================================= wallpapers
def wall_tile(files, band, size, px, seed, face=None):
    """A wallpaper's tile (rgb (h, w, 3), its sprites [(tile, map, number)]): COLS x ROWS of `files` (picture files) on
    a ground of the band's colour in a staggered lattice SPACING x `size` px apart, each `size` px long (JITTER either
    way), turned up to TURN degrees, pulled toward the band's colour by its `tint`, in a white die-cut SPRITE_EDGE x
    `px` px wide, with the band's `face`; the sprites over the tile's edges drawn again across it, so tiles side by side
    meet without a seam. Seeded by `seed` (whole numbers)."""
    g = SPACING * size
    W, H = max(int(round(COLS * g)), 1), max(int(round(ROWS * g * math.sqrt(3.0) / 2.0)), 1)
    out = np.empty((H, W, 3), np.float32)
    out[:] = np.asarray(band["color"], np.float32)
    rng = np.random.default_rng([int(x) for x in seed])
    order = rng.permutation(COLS * ROWS)                    # each file as often as the others, in no order
    tiles, placed = {}, []
    for n in range(COLS * ROWS):
        i, j = n % COLS, n // COLS
        f = files[int(order[n]) % len(files)]
        if f not in tiles:
            tiles[f] = MDF.sprite_tile(f, size, SPRITE_EDGE * px, face=face, tint=(band["color"], band["tint"]))
        t = tiles[f]
        th, tw = t.a.shape
        at = ((i + 0.5 + 0.5 * (j % 2)) * W / COLS, (j + 0.5) * H / ROWS)
        angle, s = rng.uniform(-TURN, TURN), 1.0 + rng.uniform(-JITTER, JITTER)
        for dx in (-W, 0, W):
            for dy in (-H, 0, H):
                M = MDF.placed(t, (at[0] + dx, at[1] + dy), angle, s)
                pts = np.array([[0, 0], [tw, 0], [0, th], [tw, th]], np.float64) @ M[:, :2].T + M[:, 2]
                if pts[:, 0].max() < -1 or pts[:, 0].min() > W or pts[:, 1].max() < -1 or pts[:, 1].min() > H:
                    continue                                    # this one falls off the tile
                MDF.lay(out, t, M, WHITE)
                placed.append((t, M, n))
    return out, placed


def wallpaper(tile, scroll, box):
    """The wallpaper over `box` (x0, y0, x1, y1 px of the figure's frame): `tile` side by side from the frame's top left,
    moved `scroll` (dx, dy) px (sub-pixel)."""
    x0, y0, x1, y1 = box
    h, w = tile.shape[:2]
    M = np.float32([[1, 0, scroll[0] % w - x0], [0, 1, scroll[1] % h - y0]])
    return cv2.warpAffine(tile, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def rimmed(pic, m, box, colour, width):
    """`pic` (the part over `box` (x0, y0, x1, y1) of a copy's picture, `m` its coverage in the whole frame) with a line
    `width` px wide of `colour` inside the copy's rim: none where it meets the frame's edge."""
    if width <= 0:
        return pic
    x0, y0, x1, y1 = box
    h, w = m.shape
    a0, b0 = max(x0 - 1, 0), max(y0 - 1, 0)
    inside = (m[b0:min(y1 + 1, h), a0:min(x1 + 1, w)] >= 0.5).astype(np.uint8)
    inside = cv2.copyMakeBorder(inside, 1 - (y0 - b0), 1 - (min(y1 + 1, h) - y1), 1 - (x0 - a0),
                                1 - (min(x1 + 1, w) - x1), cv2.BORDER_REPLICATE)    # the frame's edge goes on
    d = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    line = np.clip(np.float32(width + 0.5) - d, 0.0, 1.0)[..., None]
    return (pic * (1.0 - line) + np.asarray(colour, np.float32) * line).astype(np.float32)


# ================================================================================================= a figure's own place
def clean(img, m):
    """`img` with what coverage `m` covers (grown a few px) painted over from round it, at 1 / CLEAN of the size
    (cv2.inpaint) within the box it covers: the picture without the figure, where a squashed figure no longer hides
    where it stood."""
    box = support(m)
    if box is None:
        return img
    h, w = m.shape
    grow = max(int(round(0.04 * min(h, w))), 4 * CLEAN)        # room round it to fill in from, at the small size too
    x0, y0, x1, y1 = max(box[0] - grow, 0), max(box[1] - grow, 0), min(box[2] + grow, w), min(box[3] + grow, h)
    part = img[y0:y1, x0:x1]
    sw, sh = max((x1 - x0) // CLEAN, 2), max((y1 - y0) // CLEAN, 2)
    small = cv2.resize(np.ascontiguousarray(part, np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    hole = cv2.resize(np.ascontiguousarray(m[y0:y1, x0:x1], np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.01
    hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8))
    filled = cv2.inpaint((np.clip(small, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), hole, 3, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled.astype(np.float32) / np.float32(255.0), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    k = cv2.dilate(np.ascontiguousarray(m[y0:y1, x0:x1], np.float32), np.ones((5, 5), np.uint8))
    k = cv2.GaussianBlur(k, (0, 0), 1.5)[..., None]
    out = np.array(img, np.float32, copy=True)
    out[y0:y1, x0:x1] = part * (1.0 - k) + filled * k
    return out


# ================================================================================================= over a cut
class StackFx:
    """The stacks' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t, _get,
    _figure, _shot_frame, _posed, _placing, _pivot, _collage, _back, _fill, _edge, _shadow, _at, _hit_times, _hits and
    MandalaFx's _sprite_of). `_stack_init` is called once its effects and the sprite finder are known."""

    def _stack_init(self):
        """The bands' sprites found now and read (FrameFx `preload`): a missing or unreadable one is an FX.FxError
        before any frame is drawn."""
        self._sk_files, self._sk_walls, self._sk_squashes = {}, OrderedDict(), {}
        for fx in self.fxs:
            if fx["kind"] != "stack":
                continue
            for i, b in enumerate(fx["bands"]):
                if SK.is_sprites(b):
                    files = [str(f) for f in self._sprite_of(b["sprites"])]
                    for f in files if self._preload else ():
                        MDF._sprite(f)
                    self._sk_files[fx["index"], i] = files

    def _stack(self, fx, frame, img):
        return self._collage(fx, frame, img, lambda k, pic, m: self._stacked(fx, frame, img, k, pic, m))

    def _stacked(self, fx, frame, img, k, pic, m):
        """The stack on `frame` (k its hits so far; the figure's picture and coverage there): its copies laid farthest
        first, each its band's fill in its look, behind the figure, on its back; mirrored with `mirror`."""
        FF = _ff()
        t = self._t(frame)
        back = self._back(fx, frame, k, img)
        out = FF._flat(self.size, back)
        centre = self._at(fx, frame)[0]
        figs, walls, seen, layers = {0: (pic, m)}, {}, {}, []
        for cp in SK.copies(fx, t, k, self._hit_times(fx), self.fps, self.frame0):
            band = FX.pick(fx["bands"], cp["band"])
            pg, mg = self._sk_copy(fx, frame, cp["delay"], band, figs)
            if mg is None:
                continue
            M = self._sk_squashed(fx, self._sk_map(fx, cp, centre), mg, t - cp["delay"] / self.fps, seen)
            if SK.is_sprites(band):
                pg, fill = self._sk_painter(fx, cp["band"] % len(fx["bands"]), band, mg, t, walls), None
            else:
                fill = self._fill(fx, cp["band"], fx["bands"])
            look = None if fx["looks"] is None else FX.pick(fx["looks"], cp["band"])
            layers.append(Layer(pg, mg, fill, M, cp["alpha"], look, fx["recede"] is not None))
        if m is not None:
            M = self._sk_squashed(fx, IDENTITY, m, t, seen)
            if M is not IDENTITY and FF._picture(back) and not fx.get("shot"):
                out = clean(out, m)                         # she no longer covers where she stands in the picture
            layers.append(Layer(pic, m, self._fill(fx, k), M))
        return draw(out, layers, self._edge(fx), self._shadow(fx), centre[0] if fx["mirror"] else None, self.px, t)

    def _sk_copy(self, fx, frame, d, band, figs):
        """(picture or None, coverage) of a stack's copy `d` frames behind `frame`, moved by its keys there, its picture
        only when its band draws from it ("picture", "silkscreen"); (None, None) before the clip. Kept in `figs` (by d)
        for the frame."""
        need = band is None or band == FX.SILK
        got = figs.get(d)
        if got is not None and (got[0] is not None or not need):
            return got
        g = frame - d
        mg = self._figure(fx, g)
        if mg is None:
            got = (None, None)
        elif not need:
            got = self._posed(fx, g, None, mg)
        else:
            p = self._shot_frame(fx, g) if fx.get("shot") else self._get(g)
            got = self._posed(fx, g, p, mg) if p is not None else (None, None)
        figs[d] = got
        return got

    def _sk_map(self, fx, cp, centre):
        """The map of a copy: `scale` ** j about `center` and j x `step` further (a tunnel's: its recede scale ** depth
        about `center`), slid that far toward its own `at` (a peel: its pivot there, its peel scale)."""
        c = (float(centre[0]), float(centre[1]))
        r = fx["recede"]
        if r is not None:
            return scaled(r["scale"] ** cp["depth"], c, c)
        unit = min(self.size)
        j, step = cp["j"], (fx["step"][0] * unit, fx["step"][1] * unit)
        M = scaled(fx["scale"] ** j, c, (c[0] + j * step[0], c[1] + j * step[1]))
        if not cp["peel"]:
            return M
        pl, (w, h) = fx["peel"], self.size
        at = FX.pick(pl["at"], j - 1)
        placing = self._placing(fx)[1]
        pivot = self._pivot(fx) if placing is None else placing
        return mixed(M, scaled(FX.pick(pl["scale"], j - 1), pivot, (at[0] * w - 0.5, at[1] * h - 0.5)), cp["peel"])

    def _sk_squashed(self, fx, M, m, t, seen):
        """Map `M` after the stack's squash at clip second `t`, about where coverage `m` stands once laid by it."""
        sq = fx["squash"]
        if sq is None:
            return M
        sx, sy = SK.squash_at(sq, t, self._sk_squash_times(fx))
        at = None if abs(sy - 1.0) < 1e-4 else feet(m, seen)
        if at is None:
            return M
        A = np.asarray(M, np.float64)
        p = A[:, :2] @ np.asarray(at, np.float64) + A[:, 2]
        return compose(((sx, 0.0, p[0] - sx * p[0]), (0.0, sy, p[1] - sy * p[1])), M)

    def _sk_squash_times(self, fx):
        """The clip seconds a stack's squash fires on (its `times`, or its lane's hits from a little before the span: a
        copy behind in time may still be squashing on one)."""
        got = self._sk_squashes.get(fx["index"])
        if got is None:
            sq = fx["squash"]
            got = sq["times"] if sq.get("times") is not None else FX.hit_times(
                {**sq, "kind": "stack squash", "index": fx["index"], "from": fx["from"] - SK.squash_reach(fx, self.fps),
                 "to": fx["to"]}, self._hits)
            self._sk_squashes[fx["index"]] = got = sorted(got)
        return got

    def _sk_painter(self, fx, i, band, m, t, walls):
        """callable(box) -> the wallpaper of band `i` over a box of the figure's frame at clip second `t` (scrolled, its
        sprites' pupils swinging), a line of the band's colour inside the rim of coverage `m`."""
        tile = walls.get(i)
        if tile is None:
            tile = walls[i] = self._sk_wall(fx, i, band, t)
        unit = float(min(self.size))
        scroll = (band["scroll"][0] * unit * t, band["scroll"][1] * unit * t)
        width = band["line"] * self.px

        def paint(box):
            return rimmed(wallpaper(tile, scroll, box), m, box, band["color"], width)
        return paint

    def _sk_wall(self, fx, i, band, t):
        """Band `i`'s wallpaper tile on this frame: kept from frame to frame, its sprites' pupils drawn in at `t`."""
        unit = float(min(self.size))
        size = max(band["size"] * unit, 4.0)
        key = (fx["index"], i, round(size, 2))
        got = self._sk_walls.get(key)
        if got is None:
            got = self._sk_walls[key] = wall_tile(self._sk_files[fx["index"], i], band, size, self.px,
                                                  (band["seed"], fx["index"], i), band["face"])
            while len(self._sk_walls) > WALLS:
                self._sk_walls.popitem(last=False)
        else:
            self._sk_walls.move_to_end(key)
        tile, placed = got
        face = band["face"]
        if face is None or not face["eyes"]:
            return tile
        out = np.array(tile, np.float32, copy=True)
        hits = EYD.hit_times(face, self._hit_times(fx), self._hits)
        offs = {}
        for tl, M, n in placed:
            if n not in offs:
                offs[n] = EYD.pupils(face, tl, t, hits, key=(fx["index"], i, n), look=EYD.look_at(face, tl.a.shape[::-1]))
            EYD.draw_pupils(out, tl.face, M, offs[n])
        return out
