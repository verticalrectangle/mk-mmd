"""Frame effects on the finished pictures (docs/design.md: Shots, Frame effects): the picture work of the `[[fx]]` entries
that mkmmd.core.framefx checks and times. `mk post` hands each frame of the cut to FrameFx.frame after the cut's own
effects (mkmmd.cutfx) and before the screen type and the grade; `mk look` does the same for the frames it draws. Images are
float RGB in display space (0..1), (h, w, 3); a figure is its coverage, (h, w) in 0..1 over the same picture."""
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import activity as ACF
from . import colorkey as CKF
from . import eralook as ERF
from . import fields as FL
from . import footage as FT
from . import haze as HZF
from . import labels as LBF
from . import lasers as LZF
from . import mandala as MDF
from . import mitosis as MTF
from . import paintapp as PAF
from . import picturefx as PF
from . import prints as PRF
from . import remap as RMF
from . import shapes as SHF
from . import stack as SKF
from . import videosticker as VSF
from .core import breathe as BR
from .core import framefx as FX
from .core import lanes as LN
from .core import picturefx as PC

GLOW_SIGMA = 6.0                 # px at 1080 on the short side: the scope trace's glow
LINE_SHIFT = 4                   # fixed-point bits of the trace's points (sub-pixel lines)
CACHE = 64                       # decoded frames kept for echoes and slitscans
SLAB_SAMPLES = 8                 # positions a moving slab is drawn at over its shutter (motion blur)
SHOCK_RIM = 0.6                  # the lit front of a shock ring, as a share of its width
MOSH_SCALE = 0.25                # a mosh's optical flow is worked out on pictures this share of the frame's size
MOSH_PATCH = 6                   # blocks across a patch of a made-up flow's stuck blocks
MOSH_SOLID = 0.98                # a masked mosh's blocks take nothing from pixels its mask gives less than this
DOT = 0.72                       # a halftone dot's radius at full darkness, in cells: the dots merge into solid ink
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SHRINK_BY_AREA = 0.75            # a figure laid smaller than this is shrunk by area first (no shimmer in fine hair)
DEPTH_SCALE = 0.3                # how much larger an orbit's nearest copy is, and smaller its farthest, at a tilt of 90
SILK_KEY = 0.2                   # the share of a figure's luma a silkscreen's key plate prints (its darkest band)
SILK_SOFT = 1.5                  # px at 1080 on the short side: the luma smoothed before a silkscreen cuts it in bands
SILK_LINE = 4.0                  # px at 1080: the reach of a silkscreen's lines (the key where luma dips under its mean)
SILK_DEPTH = 0.06                # how far under its mean over SILK_LINE the luma of a line lies


def scale_map(c, z):
    """The 2 x 3 float32 map (each pixel of the result -> the picture's) of a zoom `z` times about pixel `c`, or None for
    no zoom (`z` within 1e-6 of 1)."""
    if abs(z - 1.0) < 1e-6:
        return None
    return np.array([[1.0 / z, 0.0, c[0] * (1.0 - 1.0 / z)], [0.0, 1.0 / z, c[1] * (1.0 - 1.0 / z)]], np.float32)


def scale_about(img, c, z):
    """The picture zoomed `z` times about pixel `c` (x, y; z > 1 is in), the frame's edge reflected where it runs out."""
    M = scale_map(c, z)
    if M is None:
        return img
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def shock(img, c, waves, width, amount, colour, glow):
    """`img` bent by rings running out from pixel `c`: each wave (radius px, weight 0..1) pushes the picture out just
    ahead of its front and pulls it in just behind (a derivative of a Gaussian across `width` px, `amount` px at its
    peak), and `glow` lights its front in `colour`."""
    if not waves:
        return img
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = xs - np.float32(c[0]), ys - np.float32(c[1])
    r = np.sqrt(dx * dx + dy * dy) + np.float32(1e-3)
    disp, rim = np.zeros_like(r), np.zeros_like(r)
    for R, s in waves:
        u = (r - np.float32(R)) / np.float32(width)
        disp += np.float32(amount * s * 2.33) * u * np.exp(-u * u)       # 2.33: the curve's peak is 1 (at u = 0.707)
        rim += np.float32(s) * np.exp(-(u / SHOCK_RIM) ** 2)
    out = cv2.remap(img, xs - disp * dx / r, ys - disp * dy / r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if glow > 0:
        out = np.clip(out + np.float32(glow) * rim[..., None] * np.asarray(colour, np.float32), 0.0, 1.0)
    return out


def slab_frame(size, centres, w, h, slant, angle):
    """What slab_cover sums with: ((ca, sa, cs) float32, the turn's cosine and sine and the slant's cosine, and t the
    slant's tangent) and [(cx, cy, (x0, x1, y0, y1))], each place with the box of pixels its coverage is summed over
    (clipped to the frame of `size`; empty off it)."""
    W, H = int(size[0]), int(size[1])
    a, t = np.radians(angle), np.tan(np.radians(slant))
    ca, sa, cs = np.float32(np.cos(a)), np.float32(np.sin(a)), np.float32(np.cos(np.radians(slant)))
    reach = 0.5 * (w + abs(h * t)) + 0.5 * h + 2.0                       # the slab's half-extent, any way, plus the edge
    places = []
    for cx, cy in centres:
        x0, x1 = max(int(np.floor(cx - reach)), 0), min(int(np.ceil(cx + reach)) + 1, W)
        y0, y1 = max(int(np.floor(cy - reach)), 0), min(int(np.ceil(cy + reach)) + 1, H)
        places.append((cx, cy, (x0, x1, y0, y1)))
    return (ca, sa, cs, t), places


def slab_cover(size, centres, w, h, slant, angle):
    """Coverage (rows, cols) 0..1 of a parallelogram `w` x `h` px whose sides lean `slant` degrees (to the right going up,
    as italics do) and which is turned `angle` degrees counter-clockwise on screen, averaged over the pixel `centres`
    [(x, y), ...] it stands at in turn (one: still; several along its way: blurred by its motion)."""
    W, H = int(size[0]), int(size[1])
    out = np.zeros((H, W), np.float32)
    (ca, sa, cs, t), places = slab_frame(size, centres, w, h, slant, angle)
    for cx, cy, (x0, x1, y0, y1) in places:
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - np.float32(cx),
                           np.float32(cy) - np.arange(y0, y1, dtype=np.float32))   # y up
        u, v = X * ca + Y * sa, -X * sa + Y * ca
        d = np.maximum(np.abs(v) - np.float32(h / 2.0), (np.abs(u - v * np.float32(t)) - np.float32(w / 2.0)) * cs)
        out[y0:y1, x0:x1] += np.clip(0.5 - d, 0.0, 1.0)
    return out / max(len(centres), 1)


def gradient_map(img, dark, colour, light):
    """Each pixel by its brightness through dark (black) .. `colour` (mid grey) .. light (white)."""
    L = np.clip(img @ LUMA, 0.0, 1.0)[..., None]
    lo = np.asarray(dark, np.float32) + (np.asarray(colour, np.float32) - np.asarray(dark, np.float32)) * (2.0 * L)
    hi = np.asarray(colour, np.float32) + (np.asarray(light, np.float32) - np.asarray(colour, np.float32)) * (2.0 * L - 1.0)
    return np.where(L < 0.5, lo, hi).astype(np.float32)


# ================================================================================================= the effects, pure
def lens(img, c, theta, shadow=0.0):
    """A point mass at pixel `c` (x, y) with Einstein radius `theta` (px): a pixel at distance r from c shows the picture
    at r - theta^2 / r along the same line (inside the ring that is the far side, folded in; on the ring, c itself),
    nothing from outside the frame; `shadow` x theta about c is black."""
    h, w = img.shape[:2]
    if theta <= 0:
        return img
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - np.float32(c[0]), yy - np.float32(c[1])
    r2 = np.maximum(dx * dx + dy * dy, np.float32(1e-6))
    k = 1.0 - np.float32(theta * theta) / r2
    out = cv2.remap(np.ascontiguousarray(img, np.float32), np.float32(c[0]) + dx * k, np.float32(c[1]) + dy * k,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
    if shadow > 0:
        out[r2 < (shadow * theta) ** 2] = 0.0
    return out


def echo(img, copies, blend="lighten", tint=None):
    """`img` with `copies` [(picture, alpha), ...] laid on it: `lighten` keeps the brighter of the two (a light figure
    on a dark ground leaves light echoes), `darken` the darker (a dark figure on a light ground). A `tint` (rgb) colours
    the echoes where they differ from the ground: a copy of the ground itself changes nothing."""
    out = np.array(img, np.float32, copy=True)
    t = None if tint is None else np.asarray(tint, np.float32)
    for pic, a in copies:
        if a <= 0:
            continue
        if blend == "lighten":
            np.maximum(out, a * pic * (1.0 if t is None else t), out=out)
        else:
            np.minimum(out, 1.0 - a * (1.0 - pic) * (1.0 if t is None else 1.0 - t), out=out)
    return out


def slit_ages(n, depth, reverse):
    """The age (frames back) of each of `n` columns or rows: 0 at the present edge, `depth` at the far one."""
    u = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    a = np.rint(depth * (u if reverse else 1.0 - u)).astype(int)
    return a


def slitscan(img, older, now="right", depth=24):
    """Each column (`now` left / right) or row (top / bottom) of the picture from a frame of its own age: `older(k)` is
    the frame k back (k >= 1; None when the clip has none so old: the oldest there is stands in)."""
    h, w = img.shape[:2]
    across = now in ("left", "right")
    ages = slit_ages(w if across else h, depth, reverse=now in ("left", "top"))
    out = np.array(img, np.float32, copy=True)
    last = img
    for k in range(1, int(ages.max()) + 1 if len(ages) else 1):
        pic = older(k)
        last = last if pic is None else pic
        sel = ages == k
        if sel.any():
            if across:
                out[:, sel] = last[:, sel]
            else:
                out[sel] = last[sel]
    return out


def zoom(log_zmax, u, ease):
    """(zoom factor, share of the way in log zoom) at progress u (0..1)."""
    if ease == "linear":
        z = 1.0 + (np.exp(log_zmax) - 1.0) * u
        return z, (np.log(z) / log_zmax if log_zmax > 0 else 1.0)
    g = u * u * (3.0 - 2.0 * u) if ease == "smooth" else u
    return float(np.exp(log_zmax * g)), g


def push_terms(size, c, r0, u, ease="log"):
    """(z, p, m, r, s) of a push into a disc of radius r0 (px) about `c` at progress u on a frame of `size` (w, h): the
    zoom, the disc's centre on screen (gliding to the frame's middle `m`), its radius and the inner frame's scale."""
    w, h = size
    m = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    R = float(np.hypot(w, h)) / 2.0 + 1.0
    z, g = zoom(np.log(max(R / max(r0, 1e-6), 1.0)), float(u), ease)
    p = np.asarray(c, float) + (m - np.asarray(c, float)) * g                       # the disc's centre on screen
    r = r0 * z
    return z, p, m, r, r / R                                                         # s: the inner frame's scale


def push(img, inner, c, r0, u, ease="log"):
    """The picture zooming into a disc of radius r0 (px) about `c` that shows `inner`: at u = 0 the picture as it is, at
    u = 1 the disc, its centre glided to the frame's middle, covers the frame and `inner` is shown at its own size."""
    h, w = img.shape[:2]
    z, p, m, r, s = push_terms((w, h), c, r0, u, ease)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - np.float32(p[0]), yy - np.float32(p[1])
    outer = cv2.remap(np.ascontiguousarray(img, np.float32), np.float32(c[0]) + dx / np.float32(z),
                      np.float32(c[1]) + dy / np.float32(z), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inside = cv2.remap(np.ascontiguousarray(inner, np.float32), np.float32(m[0]) + dx / np.float32(s),
                       np.float32(m[1]) + dy / np.float32(s), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cov = np.clip(r + 0.5 - np.sqrt(dx * dx + dy * dy), 0.0, 1.0)[..., None]        # a one-pixel soft edge
    return outer * (1.0 - cov) + inside * cov


def scope_points(x, i, n, lag, gain):
    """(n, 2) points in -1..1 of an XY trace of the mono signal `x` over samples i - n .. i: the signal against itself
    `lag` samples later, scaled by `gain`."""
    a = max(0, i - n - lag)
    seg = np.asarray(x[a:i], np.float32)
    if len(seg) <= lag + 1:
        return np.zeros((0, 2), np.float32)
    return np.clip(np.stack([seg[:len(seg) - lag], seg[lag:]], 1) * gain, -1.0, 1.0) if lag else \
        np.clip(np.stack([np.linspace(-1, 1, len(seg)), seg], 1) * [1.0, gain], -1.0, 1.0)


def trace_mask(size, pts, centre, radius, width):
    """The trace `pts` (n, 2, -1..1; y up) drawn as an antialiased line of `width` px about `centre` (px) at `radius` px:
    a uint8 mask of a frame of `size` (w, h)."""
    w, h = size
    q = np.stack([centre[0] + pts[:, 0] * radius, centre[1] - pts[:, 1] * radius], 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.rint(q * (1 << LINE_SHIFT)).astype(np.int32)], False, 255, max(1, int(round(width))),
                  cv2.LINE_AA, shift=LINE_SHIFT)
    return mask


def draw_trace(img, pts, centre, radius, colour, width, glow, opacity=1.0, px=1.0):
    """`img` with the trace `pts` (n, 2, -1..1; y up) drawn as a line of `width` px about `centre` (px) at `radius` px,
    added in `colour` with a glow of share `glow`."""
    h, w = img.shape[:2]
    if len(pts) < 2 or opacity <= 0:
        return img
    a = trace_mask((w, h), pts, centre, radius, width).astype(np.float32) / 255.0
    if glow > 0:
        a = a + glow * cv2.GaussianBlur(a, (0, 0), GLOW_SIGMA * px) * 3.0
    return np.clip(img + opacity * a[..., None] * np.asarray(colour, np.float32), 0.0, 1.0)


def plate_shift(a, dx, dy):
    """One colour plate `a` (h, w) printed `dx`, `dy` px off (sub-pixel), its edge repeated."""
    h, w = a.shape
    M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(np.ascontiguousarray(a, np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def halftone(img, cell, angle, level, ink, lift, offset):
    """Comic print (`cell` and `offset` in px): where the luma is under `level` the picture is lifted toward white by
    `lift` x its darkness d (0 at `level`, 1 at black) and screened with round `ink` dots on a grid `cell` px apart turned
    `angle` degrees, a dot's radius DOT x cell x sqrt(d) (they touch at a quarter and merge into solid ink at black, as
    dots gain on cheap paper); the lights stay clean. Then the red plate is printed `offset` (dx, dy) px off and the blue
    as far the other way, a press out of register."""
    h, w = img.shape[:2]
    L = np.clip(img @ LUMA, 0.0, 1.0)
    d = np.clip((np.float32(level) - L) / np.float32(level), 0.0, 1.0)
    a = np.radians(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ca, sa = np.float32(np.cos(a)), np.float32(np.sin(a))
    u = (xx * ca + yy * sa) / np.float32(cell)
    v = (yy * ca - xx * sa) / np.float32(cell)
    u -= np.rint(u)
    v -= np.rint(v)
    dist = np.sqrt(u * u + v * v) * np.float32(cell)
    cov = (np.clip(np.float32(DOT * cell) * np.sqrt(d) - dist + np.float32(0.5), 0.0, 1.0) * (d > 0))[..., None]
    base = img + (1.0 - img) * (np.float32(lift) * d)[..., None]
    out = (base * (1.0 - cov) + np.asarray(ink, np.float32) * cov).astype(np.float32)
    if offset[0] or offset[1]:
        out[..., 0] = plate_shift(out[..., 0], offset[0], offset[1])
        out[..., 2] = plate_shift(out[..., 2], -offset[0], -offset[1])
    return out


def split_shifts(h, rng, bands, shift, share, rgb):
    """How a broken digital signal tears a frame of `h` rows (split's randomness, drawn in its order): [(first row, past
    the last, (red, green, blue) px that band's planes roll sideways)], the bands covering the rows in order."""
    n = min(int(bands), h)
    cuts = np.sort(rng.choice(np.arange(1, h), size=n - 1, replace=False)) if n > 1 else np.zeros(0, int)
    edges = np.concatenate([[0], cuts, [h]]).astype(int)
    base = int(round(rgb / 3.0))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if rng.random() >= share:
            out.append((int(a), int(b), (base, 0, -base)))
            continue
        dx, s = int(round(rng.uniform(-1.0, 1.0) * shift)), int(round(rgb))
        out.append((int(a), int(b), (dx + s, dx, dx - s)))
    return out


def split_rolled(img, bands):
    """`img` with each band of split_shifts' rolled sideways, its planes each by their own px."""
    out = np.empty_like(img)
    for a, b, shifts in bands:
        for ch, s in enumerate(shifts):
            out[a:b, :, ch] = np.roll(img[a:b, :, ch], s, 1)
    return out


def split(img, rng, bands, shift, share, rgb):
    """A broken digital signal (`shift` and `rgb` in px, `rng` a numpy Generator): the frame cut into `bands` bands of
    random heights, `share` of them torn sideways by up to `shift` px (what leaves one side comes in at the other), the
    red and blue planes pulled `rgb` px apart in a torn band and a third of that everywhere else."""
    return split_rolled(img, split_shifts(img.shape[0], rng, bands, shift, share, rgb))


def _grey8(img, size):
    small = cv2.resize(np.ascontiguousarray(img, np.float32), size, interpolation=cv2.INTER_AREA)
    return (np.clip(small @ LUMA, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def mosh_blocks(prev, cur, block):
    """The cut's own motion from `prev` to `cur` as mosh_step carries a picture along it: Farneback optical flow on grey
    pictures at MOSH_SCALE of the frame, averaged over `block` px squares: (blocks (gh, gw, 2) float32, their side b
    px at that scale, the flow's size (sw, sh))."""
    h, w = cur.shape[:2]
    sw, sh = max(8, int(round(w * MOSH_SCALE))), max(8, int(round(h * MOSH_SCALE)))
    flow = cv2.calcOpticalFlowFarneback(_grey8(prev, (sw, sh)), _grey8(cur, (sw, sh)), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    b = max(1, int(round(block * MOSH_SCALE)))
    gh, gw = -(-sh // b), -(-sw // b)
    pad = np.pad(flow, ((0, gh * b - sh), (0, gw * b - sw), (0, 0)), mode="edge")
    return np.ascontiguousarray(pad.reshape(gh, b, gw, b, 2).mean((1, 3)), np.float32), b, (sw, sh)


def mosh_field(blocks, b, small, size):
    """(h, w, 2) float32: each pixel's move in a frame of `size` (w, h) from mosh_blocks' (blocks, b, small): its block's,
    nearest, as a codec's motion vectors move whole blocks."""
    (sw, sh), (w, h), (gh, gw) = small, size, blocks.shape[:2]
    full = cv2.resize(blocks, (gw * b, gh * b), interpolation=cv2.INTER_NEAREST)[:sh, :sw]
    return cv2.resize(full, (int(w), int(h)), interpolation=cv2.INTER_NEAREST) / np.float32(MOSH_SCALE)


def mosh_step(held, prev, cur, block):
    """`held` carried one frame along the motion from `prev` to `cur` (the cut's own pictures, mosh_blocks), each pixel
    taking what stood its block's move behind it."""
    h, w = cur.shape[:2]
    full = mosh_field(*mosh_blocks(prev, cur, block), (w, h))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(held, np.float32), xs - full[..., 0], ys - full[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _block_of(n, b):
    """The block (0, 1, ...) each of `n` pixels along a side lies in, blocks of `b` px (1 or more, a fraction too) from
    its first pixel."""
    return (np.arange(n) / b).astype(np.int64)


def mosh_flow(flow, size, block, px, index):
    """A mosh's made-up motion over pictures of `size` (w, h), as a codec's motion vectors: (moves, b), each block's
    move per step in whole px ((rows, cols, 2) int32, x right, y down) and b, the blocks' side in px (`block`, a
    fraction too: a frame of another size of the same shape is cut into the same blocks and moshed the same way, its
    moves rounded at its own size). The flow's kind at each block's middle, its `amount` (px at 1080, `px` px each) the
    farthest move: `drift` the same move everywhere, `angle` degrees counter-clockwise from the right; `zoom` the
    picture pushed out from `center` ([x, y] frame fractions) as a camera pushing in does, the frame's farthest corner
    from it moving most (drawn in toward it with a negative amount); `swirl` round it, counter-clockwise on screen
    (clockwise with a negative amount); `noise` a smooth random field whose swirls are `size` frame heights across.
    Each block's move is then off by up to `jitter` x amount at random (a motion search's errors), and a share `stuck`
    of the blocks, in patches, does not move at all (skip blocks). Seeded by the flow's `seed` and the entry's `index`,
    each part on its own: another `jitter` or `stuck` leaves the rest as it was, and a larger `stuck` holds more of the
    same patches."""
    w, h = int(size[0]), int(size[1])
    b = max(float(block), 1.0)
    gw, gh = int(_block_of(w, b)[-1]) + 1, int(_block_of(h, b)[-1]) + 1
    amount, kind, seed = flow["amount"] * px, flow["kind"], [flow["seed"], index]
    xs, ys = np.meshgrid((np.arange(gw) + 0.5) * b - 0.5, (np.arange(gh) + 0.5) * b - 0.5)
    if kind == "drift":
        a = math.radians(flow["angle"])
        v = np.stack([np.full_like(xs, amount * math.cos(a)), np.full_like(xs, -amount * math.sin(a))], -1)
    elif kind == "noise":
        cell = flow["size"] * h
        lattice = np.random.default_rng(seed + [0]).standard_normal(
            (int(math.ceil(h / cell)) + 2, int(math.ceil(w / cell)) + 2, 2))
        v = cv2.resize(lattice.astype(np.float32), (gw, gh), interpolation=cv2.INTER_CUBIC).astype(np.float64)
        v *= amount / max(float(np.hypot(v[..., 0], v[..., 1]).max()), 1e-9)
    else:
        cx, cy = flow["center"][0] * w - 0.5, flow["center"][1] * h - 0.5
        reach = max(math.hypot(x - cx, y - cy) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5))
        dx, dy = (xs - cx) * (amount / reach), (ys - cy) * (amount / reach)
        v = np.stack([dx, dy] if kind == "zoom" else [dy, -dx], -1)
    rng = np.random.default_rng(seed + [1])
    r = np.sqrt(rng.random((gh, gw))) * (flow["jitter"] * abs(amount))
    t = rng.random((gh, gw)) * (2.0 * math.pi)
    v = v + np.stack([r * np.cos(t), r * np.sin(t)], -1)
    n = int(round(flow["stuck"] * gw * gh))
    if n:
        patches = np.random.default_rng(seed + [2]).standard_normal(
            (-(-gh // MOSH_PATCH) + 1, -(-gw // MOSH_PATCH) + 1))
        s = cv2.resize(patches.astype(np.float32), (gw, gh), interpolation=cv2.INTER_CUBIC)
        v[np.unravel_index(np.argsort(s, axis=None, kind="stable")[:n], (gh, gw))] = 0.0
    return np.rint(v).astype(np.int32), b


def mosh_stay(moves, b, outside):
    """Block `moves` (mosh_flow's, blocks of b px) with every block whose source (the block its move behind it, the
    frame's edge repeated past the frame) holds a pixel of `outside` ((h, w) bool) kept still: a masked mosh moves the
    picture of its region alone, and what lies outside it never smears in."""
    h, w = outside.shape
    gh, gw = moves.shape[:2]
    s = np.zeros((h + 1, w + 1), np.int64)
    s[1:, 1:] = np.cumsum(np.cumsum(outside, 0, dtype=np.int64), 1)
    bx, by = _block_of(w, b), _block_of(h, b)
    cols, rows = np.arange(gw), np.arange(gh)
    x0 = np.searchsorted(bx, cols)[None, :] - moves[..., 0]
    x1 = np.searchsorted(bx, cols, side="right")[None, :] - moves[..., 0]
    y0 = np.searchsorted(by, rows)[:, None] - moves[..., 1]
    y1 = np.searchsorted(by, rows, side="right")[:, None] - moves[..., 1]
    ax, cx = np.clip(x0, 0, w - 1), np.clip(x1 - 1, 0, w - 1) + 1
    ay, cy = np.clip(y0, 0, h - 1), np.clip(y1 - 1, 0, h - 1) + 1
    out = moves.copy()
    out[(s[cy, cx] - s[ay, cx] - s[cy, ax] + s[ay, ax]) > 0] = 0
    return out


def mosh_carry(held, moves, b, steps):
    """`held` carried `steps` steps along block `moves` (mosh_flow's, blocks of b px): each pixel takes what stood its
    block's move behind it, a whole pixel, so the picture moves in sharp chunks as a P-frame's blocks do, a block that
    pulls from one that stays smearing that one's edge out in a trail; past the frame's edge its edge repeats."""
    out = np.ascontiguousarray(held, np.float32)
    if steps < 1 or not moves.any():
        return out
    h, w = out.shape[:2]
    full = moves[_block_of(h, b)[:, None], _block_of(w, b)[None, :]].astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xs - full[..., 0], ys - full[..., 1]
    for _ in range(int(steps)):
        out = cv2.remap(out, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return out


# ================================================================================================= collages
def centroid(m):
    """The centre (x, y) in pixels of a coverage (h, w), its coverage-weighted mean; None when nothing is covered."""
    mo = cv2.moments(np.ascontiguousarray(m, np.float32))
    if mo["m00"] < 1.0:
        return None
    return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]


def _picture(back):
    """Whether a collage's back is a picture (the frame as it reached the effect), not a colour or a pattern's painter."""
    return isinstance(back, np.ndarray)


def _flat(size, colour):
    """A picture of `size` (w, h) of a collage's back: a flat colour, what a pattern back (a painter: callable(size) ->
    picture, mkmmd.fields) paints there, or a copy of a picture back (the frame as it reached the effect, that size)."""
    if callable(colour):
        return np.array(colour(size), np.float32)
    if _picture(colour):
        return np.array(colour, np.float32)
    w, h = size
    return np.full((h, w, 3), colour, np.float32)


def _grow(a, width):
    """Coverage `a` grown by `width` px all round, antialiased, its outline rounded where it bends in as a die cut's is:
    laid under a figure it shows as an outline, or a sticker's border."""
    r = max(float(width), 0.5)
    inside = (a >= 0.5).astype(np.uint8)
    if not inside.any():
        return a
    d = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)            # px from the figure
    g = np.clip(np.float32(r + 0.5) - d, 0.0, 1.0)
    if r >= 3.0:
        sig = 0.25 * r
        g = np.clip((cv2.GaussianBlur(g, (0, 0), sig) - 0.5) * np.float32(sig * 2.5066) + 0.5, 0.0, 1.0)
    return np.maximum(g, a)


def warp_spec(M, shape):
    """How `_warp` carries a picture of `shape` ((h, w, ...)) by the affine map `M` (2 x 3, its pixels to the result's):
    (the float32 map it warps by, from the picture as it warps it; the size (w, h) the picture is first shrunk to by
    area when M shrinks it under SHRINK_BY_AREA, else None)."""
    M = np.asarray(M, np.float64)
    s = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    if s >= SHRINK_BY_AREA:
        return M.astype(np.float32), None
    sw, sh = max(int(round(shape[1] * s)), 1), max(int(round(shape[0] * s)), 1)
    k = np.array([sw / shape[1], sh / shape[0]])
    return np.hstack([M[:, :2] / k, (M[:, 2] + M[:, :2] @ (0.5 / k - 0.5))[:, None]]).astype(np.float32), (sw, sh)


def _warp(src, M, size):
    """`src` ((h, w) or (h, w, 3)) carried by the affine map `M` (2 x 3, its pixels to the picture's) into a picture of
    `size` (nothing outside it), shrunk by area first when M shrinks it under SHRINK_BY_AREA (warp_spec)."""
    src = np.ascontiguousarray(src, np.float32)
    Mf, shrink = warp_spec(M, src.shape)
    if shrink is not None:
        src = cv2.resize(src, shrink, interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(src, Mf, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _laid(src, s, a, p, size):
    """`src` ((h, w) or (h, w, 3)) scaled `s` times about pixel `a`, with `a` put at pixel `p`, in a picture of `size`
    (nothing outside it)."""
    w, h = size
    if abs(s - 1.0) < 1e-9 and abs(p[0] - a[0]) < 1e-6 and abs(p[1] - a[1]) < 1e-6 and src.shape[:2] == (h, w):
        return src
    return _warp(src, [[s, 0.0, p[0] - s * a[0]], [0.0, s, p[1] - s * a[1]]], size)


def _cast(dst, cov, shadow, s=1.0):
    """`dst` under the drop shadow of a coverage `cov`: shadow = (colour, (dx, dy) px, blur px (a Gaussian's sigma),
    alpha), its offset and blur `s` times as large (a smaller copy casts a smaller shadow)."""
    colour, (dx, dy), blur, alpha = shadow
    sh = plate_shift(cov, dx * s, dy * s)
    if blur * s > 0.3:
        sh = cv2.GaussianBlur(sh, (0, 0), blur * s)
    a = (np.float32(alpha) * sh)[..., None]
    return (dst * (1.0 - a) + np.asarray(colour, np.float32) * a).astype(np.float32)


def silk_levels(L, m, n):
    """The luma thresholds (n - 1, rising) of a silkscreen print in `n` inks: quantiles of the figure's own luma `L` where
    its coverage `m` is over half, the darkest band SILK_KEY of it and the others in equal shares, each half a code value
    up (a flat area the quantile falls on prints in the band below, not half way between two inks); None for no figure."""
    vals = L[m > 0.5]
    if vals.size == 0:
        return None
    qs = [SILK_KEY] + [SILK_KEY + (1.0 - SILK_KEY) * j / (n - 1) for j in range(1, n - 1)]
    return np.quantile(vals[::max(1, vals.size // 200000)], qs) + 0.5 / 255.0


def _within(x, m, sigma):
    """`x` (h, w) blurred `sigma` px inside the coverage `m` (a normalised convolution: what lies outside the figure does
    not bleed in at its edge)."""
    den = cv2.GaussianBlur(np.ascontiguousarray(m, np.float32), (0, 0), sigma)
    return cv2.GaussianBlur(np.ascontiguousarray(x * m, np.float32), (0, 0), sigma) / np.maximum(den, np.float32(1e-4))


def silk_bands(pic, m, n, soft, line=0.0):
    """A silkscreen print of a figure (picture `pic`, coverage `m`) in `n` inks as coverages over the picture: (key,
    steps), key the key plate inside the figure: its darkest band (SILK_KEY of its luma) and, with `line` (px), the
    lines, eyes and creases, where the luma lies SILK_DEPTH under its mean over `line` px round it; steps[j] where the
    luma is past threshold j + 1 (band j + 2's ink over band j + 1's). Each a hard edge antialiased over a pixel: no
    gradient inside a band. The luma is smoothed `soft` px first (inside the figure), so the bands print as shapes."""
    L = np.ascontiguousarray(np.asarray(pic, np.float32) @ LUMA)
    if soft > 0:
        L = _within(L, m, soft)
    lv = silk_levels(L, m, n)
    if lv is None:
        return np.zeros_like(m), [np.zeros_like(m) for _ in range(n - 2)]
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=1)
    g = np.float32(0.5) * np.sqrt(gx * gx + gy * gy) + np.float32(1e-4)    # luma per px across a band's edge

    def past(t):
        return np.clip((L - t) / g + 0.5, 0.0, 1.0)
    key = 1.0 - past(np.float32(lv[0]))
    if line > 0:
        key = np.maximum(key, 1.0 - past(_within(L, m, line) - np.float32(SILK_DEPTH)))
    return key * m, [past(np.float32(t)) for t in lv[1:]]


def silk_print(dst, cov, key, steps, inks, offset):
    """`dst` with a figure printed in flat inks (coverage `cov`; `key` and `steps` from silk_bands, laid as it is): the
    colour plates first, inks[1] and each next ink where its step is, then the key plate in inks[0] printed `offset`
    (dx, dy) px off register, so the band under it shows where it misses and it prints on the ground past the figure."""
    inks = [np.asarray(c, np.float32) for c in inks]
    colour = np.empty(dst.shape, np.float32)
    colour[:] = inks[1]
    for j, st in enumerate(steps):
        colour += (inks[j + 2] - inks[j + 1]) * st[..., None]
    c = cov[..., None]
    out = dst * (1.0 - c) + colour * c
    k = (plate_shift(key, offset[0], offset[1]) if offset[0] or offset[1] else key)[..., None]
    return (out * (1.0 - k) + inks[0] * k).astype(np.float32)


def _paste(dst, pic, m, fill, s=1.0, a=(0.0, 0.0), p=(0.0, 0.0), edge=None, shadow=None):
    """`dst` with a figure laid on it: coverage `m` scaled `s` about pixel `a` and put with `a` at pixel `p`, drawn as its
    picture `pic` (`fill` None), in the flat colour `fill`, or printed (`fill` a silkscreen: {inks, offset px, soft px}),
    over an `edge` (colour, width px) grown round it, the two casting a `shadow` (colour, (dx, dy) px, blur px, alpha)."""
    size = (dst.shape[1], dst.shape[0])
    cov = _laid(m, s, a, p, size)
    ring = None if edge is None or edge[1] <= 0 else _grow(cov, edge[1])
    if shadow is not None and shadow[3] > 0:
        dst = _cast(dst, cov if ring is None else ring, shadow, s)
    if ring is not None:
        g = ring[..., None]
        dst = dst * (1.0 - g) + np.asarray(edge[0], np.float32) * g
    if isinstance(fill, dict):
        key, steps = silk_bands(pic, m, len(fill["inks"]), fill["soft"], fill["line"])
        return silk_print(dst, cov, _laid(key, s, a, p, size), [_laid(x, s, a, p, size) for x in steps], fill["inks"],
                          fill["offset"])
    if fill is None:
        rgb = _laid(pic * m[..., None], s, a, p, size)
    else:
        rgb = np.asarray(fill, np.float32) * cov[..., None]
    return (rgb + dst * (1.0 - cov[..., None])).astype(np.float32)


def _part(W, H, x, y, tw, th):
    """Where a tw x th tile put with its top left at (x, y) meets a W x H picture: (picture slices, tile slices), None
    when it falls outside."""
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + tw, W), min(y + th, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))


def _framed(pic, size, at):
    """A picture of `size` (w, h) with `pic` in it, its top left at the whole pixel `at` (x, y), and `pic` reflected
    across its edges past them: a grid cell's picture back (the frame shrunk and laid as the cell's figure is)."""
    W, H = int(size[0]), int(size[1])
    ph, pw = pic.shape[:2]
    x0, y0 = max(-at[0], 0), max(-at[1], 0)                                 # what of `pic` falls inside
    part = np.ascontiguousarray(pic[y0:min(ph, H - at[1]), x0:min(pw, W - at[0])])
    left, top = at[0] + x0, at[1] + y0
    return cv2.copyMakeBorder(part, top, H - top - part.shape[0], left, W - left - part.shape[1], cv2.BORDER_REFLECT)


def cells(img, m, boxes, colour_at, fill_at, piece=None, edge=None, shadow=None):
    """Boxes of a frame each showing the whole frame contain-fitted and centred in it, the figure (coverage `m` of `img`)
    on its back: boxes [((x0, y0, x1, y1) frame fractions, (i, j)), ...] drawn in order, each on colour_at(i, j) (a
    colour, a pattern's painter, or a picture: the frame, shrunk and laid as the figure is and reflected past its edges
    to fill the box) with the figure drawn fill_at(i, j) (None: its picture; a colour; a silkscreen {inks, offset px,
    soft px}) over an `edge` (colour, width px) and its `shadow` (colour, (dx, dy) px, blur px, alpha; scaled with the
    figure); `piece` ((i, j), colour) draws that cell's figure flat in its colour. The boxes of one call are of one size
    (a layout, or a step's cells on their way), so the figure is shrunk once."""
    h, w = m.shape
    out = np.zeros((h, w, 3), np.float32)
    if not boxes:
        return out
    x0, y0, x1, y1 = boxes[0][0]
    s = min(x1 - x0, y1 - y0)                                           # the frame in a box, contain-fitted
    sw, sh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    ew = 0.0 if edge is None else edge[1]
    reach = 0.0 if shadow is None else math.hypot(*shadow[1]) * s + 3.0 * shadow[2] * s
    silks = [f for f in (fill_at(*b[1]) for b in boxes) if isinstance(f, dict)]
    pad = int(math.ceil(ew + reach + max([math.hypot(*f["offset"]) for f in silks] or [0.0]))) + 2
    tw, th = sw + 2 * pad, sh + 2 * pad

    def tile(x):
        t = np.zeros((th, tw) + x.shape[2:], np.float32)
        t[pad:pad + sh, pad:pad + sw] = cv2.resize(np.ascontiguousarray(x, np.float32), (sw, sh),
                                                   interpolation=cv2.INTER_AREA)
        return t
    cov, pic = tile(m), tile(img * m[..., None])
    ring = None if ew <= 0 else _grow(cov, ew)
    under = cov if ring is None else ring
    dark = None
    if shadow is not None and shadow[3] > 0:
        dark = plate_shift(under, shadow[1][0] * s, shadow[1][1] * s)
        if shadow[2] * s > 0.3:
            dark = cv2.GaussianBlur(dark, (0, 0), shadow[2] * s)
        dark = np.float32(shadow[3]) * dark
    bands = {}                                       # (inks, key offset) -> the key plate shifted, the colour steps
    for f in silks:
        n, off = len(f["inks"]), (float(f["offset"][0]), float(f["offset"][1]))
        if (n, off) not in bands:
            key, steps = silk_bands(img, m, n, f["soft"], f["line"])
            key = tile(key)
            bands[n, off] = (plate_shift(key, *off) if off[0] or off[1] else key, [tile(st) for st in steps])
    spans, small = [], {}                            # small: a picture back shrunk as the figure is, by the picture
    for (bx0, by0, bx1, by1), (i, j) in boxes:                      # the grounds first, in order
        X0, Y0, X1, Y1 = int(round(bx0 * w)), int(round(by0 * h)), int(round(bx1 * w)), int(round(by1 * h))
        if X1 > X0 and Y1 > Y0:
            at = (int(round((bx0 + bx1) * w / 2.0 - sw / 2.0)) - X0,      # the shrunk frame's corner in the box
                  int(round((by0 + by1) * h / 2.0 - sh / 2.0)) - Y0)
            back = colour_at(i, j)
            if _picture(back):
                if id(back) not in small:
                    small[id(back)] = cv2.resize(np.ascontiguousarray(back, np.float32), (sw, sh),
                                                 interpolation=cv2.INTER_AREA)
                out[Y0:Y1, X0:X1] = _framed(small[id(back)], (X1 - X0, Y1 - Y0), at)
            else:
                out[Y0:Y1, X0:X1] = _flat((X1 - X0, Y1 - Y0), back)
            spans.append(((X0, Y0, X1, Y1), at, (i, j)))
    for (X0, Y0, X1, Y1), at, (i, j) in spans:                     # then each figure inside its box: copies on their
        part = _part(X1 - X0, Y1 - Y0, at[0] - pad, at[1] - pad, tw, th)      # way overlap, none is cut off
        if part is None:
            continue
        rs, ts = part
        region = out[Y0:Y1, X0:X1]
        sub = region[rs]
        if dark is not None:
            a = dark[ts][..., None]
            sub = sub * (1.0 - a) + np.asarray(shadow[0], np.float32) * a
        if ring is not None:
            g = ring[ts][..., None]
            sub = sub * (1.0 - g) + np.asarray(edge[0], np.float32) * g
        fill = piece[1] if piece is not None and tuple(piece[0]) == (i, j) else fill_at(i, j)
        c = cov[ts]
        if isinstance(fill, dict):
            key, steps = bands[len(fill["inks"]), (float(fill["offset"][0]), float(fill["offset"][1]))]
            sub = silk_print(sub, c, key[ts], [st[ts] for st in steps], fill["inks"], (0.0, 0.0))
        elif fill is None:
            sub = pic[ts] + sub * (1.0 - c[..., None])
        else:
            sub = np.asarray(fill, np.float32) * c[..., None] + sub * (1.0 - c[..., None])
        region[rs] = sub
    return out


def grid(img, m, n, colour_at, fill_at, piece=None, edge=None, shadow=None):
    """The frame shrunk into cells, `n` x `n` (or n = (columns, rows)), each the whole frame contain-fitted and centred
    in its cell, the figure (coverage `m` of `img`) on its back: colour_at(i, j) (cells()) and fill_at(i, j) (None: the
    figure's picture; a colour; a silkscreen) of the cell in column i, row j; `piece` ((i, j), colour) draws that cell's
    figure flat in its colour; an `edge` (colour, width px) round each figure and its `shadow`."""
    d = FX.dims(n)
    return cells(img, m, [(FX.box(d, i, j), (i, j)) for j in range(d[1]) for i in range(d[0])], colour_at, fill_at,
                 piece, edge, shadow)



def orbit(img, m, back, centre, radius, tilt, theta0, count, scale, size, fill_at, edge=None, shadow=None):
    """The figure (coverage `m` of `img`) moved to pixel `centre` on `back` (a colour, a pattern's painter or a picture),
    `scale` times its size, and `count` copies of it, `size` times as large again, on a ring of `radius` px round it
    tilted `tilt` degrees from face-on: an ellipse whose near half (the lower on screen) is drawn over the figure and
    larger, its far half under it and smaller. Copy j (1 .. count) stands at angle theta0 + 2 pi (j - 1) / count, radians
    clockwise on screen from the right; fill_at(0) is the figure's fill and fill_at(j) copy j's (None: its picture); each
    with its `edge` and `shadow`."""
    h, w = m.shape
    out = _flat((w, h), back)
    a = centroid(m)
    if a is None:
        return out
    ct, st = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    order = [(0.0, 1, 0, (float(centre[0]), float(centre[1])), scale)]         # the figure, over the copies level with it
    for j in range(1, count + 1):
        th = theta0 + 2.0 * math.pi * (j - 1) / count
        depth = round(math.sin(th), 9)                                           # sin(pi) is level, not a hair in front
        order.append((depth, 0, j, (centre[0] + radius * math.cos(th), centre[1] + radius * ct * math.sin(th)),
                      scale * size * (1.0 + DEPTH_SCALE * st * depth)))
    for _, _, j, p, s in sorted(order, key=lambda o: (o[0], o[1])):
        out = _paste(out, img, m, fill_at(j), s, a, p, edge, shadow)
    return out


def mirror(img, m, centre, axes, colour_at, fill_at, place=None, edge=None, shadow=None):
    """The figure (coverage `m` of `img`; its centroid moved to pixel `place` when one is given) on its back, folded
    about the vertical line through pixel `centre` (`axes` "x"), the horizontal one ("y") or both ("xy"): the side or
    quarter the figure's centroid is on is shown as it is, the others are its reflections. Region j (0 that side, 1
    across the vertical line, 2 across the horizontal, 3 across both) is colour_at(j) (a colour, a pattern's painter or a
    picture, folded with the figure: past the frame's edge it is reflected again) with the figure drawn fill_at(j) (None:
    its picture), with its `edge` and `shadow`."""
    h, w = m.shape
    a = centroid(m)
    p = (centre[0] - 1.0, centre[1] - 1.0) if a is None else a if place is None else (float(place[0]), float(place[1]))
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    rx = ((xs >= centre[0]) if p[0] < centre[0] else (xs < centre[0])) & ("x" in axes)    # across the vertical line
    ry = ((ys >= centre[1]) if p[1] < centre[1] else (ys < centre[1])) & ("y" in axes)    # across the horizontal one
    mx = np.ascontiguousarray(np.broadcast_to(np.where(rx, 2.0 * centre[0] - xs, xs)[None, :], (h, w)), np.float32)
    my = np.ascontiguousarray(np.broadcast_to(np.where(ry, 2.0 * centre[1] - ys, ys)[:, None], (h, w)), np.float32)
    region = rx[None, :].astype(np.int8) + 2 * ry[:, None].astype(np.int8)
    out = np.empty((h, w, 3), np.float32)
    for j in np.unique(region):
        j = int(j)
        bg = colour_at(j)
        comp = _flat((w, h), bg) if a is None else _paste(_flat((w, h), bg), img, m, fill_at(j), 1.0, a, p, edge, shadow)
        if j:
            border = dict(borderMode=cv2.BORDER_REFLECT) if callable(bg) or _picture(bg) else dict(
                borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(c) for c in bg))
            comp = cv2.remap(comp, mx, my, cv2.INTER_LINEAR, **border)
        sel = region == j
        out[sel] = comp[sel]
    return out


def panels(size, n, slant, figures, colour_at, fill_at, scale=1.0, edge=None, shadow=None):
    """The frame cut into `n` panels across its long side (columns of a wide or square frame, rows of a tall one) by lines
    leaning `slant` degrees (a column's edges to the right going up, as italics do; a row's rising to the right): panel j
    is colour_at(j) (a colour, a pattern's painter or a picture) with figures[j] = (picture, coverage) on it, moved so its
    centroid sits on the panel's middle line and `scale` times its size (and n times smaller in the rows of a tall frame,
    so a figure framed for the whole frame fits one), drawn fill_at(j) (None: its picture) with its `edge` and `shadow`;
    a figure None leaves its panel plain."""
    w, h = size
    tall = h > w
    span, tn = (h if tall else w), math.tan(math.radians(slant))
    xs, ys = np.arange(w, dtype=np.float32)[None, :] + 0.5, np.arange(h, dtype=np.float32)[:, None] + 0.5
    u = (ys + (xs - w / 2.0) * tn) if tall else (xs + (ys - h / 2.0) * tn)       # across the panels (from pixel edges)
    s = scale / n if tall else scale
    out = np.zeros((h, w, 3), np.float32)
    for j in range(n):
        lo, hi = j * span / n, (j + 1) * span / n
        cov = np.ones((h, w), np.float32)
        if j > 0:
            cov = cov * np.clip(u - lo + 0.5, 0.0, 1.0)
        if j < n - 1:
            cov = cov * np.clip(hi - u + 0.5, 0.0, 1.0)
        tile = _flat(size, colour_at(j))
        fig = figures[j]
        a = None if fig is None else centroid(fig[1])
        if a is not None:
            mid = (lo + hi) / 2.0 - 0.5                                  # the middle line, in pixel centres
            p = (a[0], mid - (a[0] + 0.5 - w / 2.0) * tn) if tall else (mid - (a[1] + 0.5 - h / 2.0) * tn, a[1])
            tile = _paste(tile, fig[0], fig[1], fill_at(j), s, a, p, edge, shadow)
        out += tile * cov[..., None]
    return out


_STEMS = {}                 # (root, stem cache, timeline, its mtime) -> its stems as stem_source gives them


def stem_source(root, cache_dir):
    """callable(timeline, stem) -> (mono float32 samples, sample rate, pre) for a project at `root`: the stems of the
    window `timeline` (project-relative) was analysed on, from the stem cache `cache_dir` only (mkmmd.timeline.hits
    .cached_stems), each loaded once a process while its timeline stays as it is (an output's FrameFx and the next
    one's share them); the timeline's audio file, like every path of a project, is project-relative. A
    FileNotFoundError says what to run when they are not there."""
    import json
    import subprocess
    from pathlib import Path

    from .timeline import hits as H
    from .timeline import signal as S

    def get(timeline, stem):
        p = Path(root) / Path(timeline).expanduser()
        try:
            key = (str(root), str(cache_dir), timeline, p.stat().st_mtime_ns)
        except OSError as e:
            raise FileNotFoundError(f"scope: {p}: {e}") from None
        if key not in _STEMS:
            try:
                tl = json.loads(p.read_text(encoding="utf-8"))
                au = dict(tl.get("audio") or {})
                if au.get("file"):
                    f = Path(au["file"]).expanduser()
                    au["file"] = str(f if f.is_absolute() else Path(root) / f)
                st, pre = H.cached_stems({**tl, "audio": au}, str(cache_dir))
            except (OSError, ValueError, subprocess.CalledProcessError) as e:     # HitsError is a ValueError
                raise FileNotFoundError(f"scope: {p}: {e}") from None
            _STEMS[key] = ({k: np.asarray(v, np.float32).mean(0) for k, v in st.items()}, pre)
        st, pre = _STEMS[key]
        return st[stem], S.SR, pre
    return get


def project_hits(root, data):
    """The [audio] hits file of a project (mk.toml `data`, at `root`) as {lane: [...]}, or None when it names none; a
    zoom or a shock on a lane is then an error that says to write it."""
    rel = (data.get("audio") or {}).get("hits")
    if not rel:
        return None
    p = Path(rel).expanduser()
    try:
        return LN.load(p if p.is_absolute() else Path(root) / p)
    except LN.LaneError as e:
        raise FileNotFoundError(str(e)) from None


def clip_of(fx, footage_of):
    """The clip of an entry that shows one (a footage collage, a video sticker: FX.CLIPS; `footage_of(clip)`,
    mkmmd.footage.source) with its span checked against its length: a FootageError naming the entry when the clip is
    missing or unreadable, or its span starts past its end without `loop`."""
    try:
        clip = footage_of(fx["clip"])
        clip.index(0.0, fx["in"], fx["rate"], fx["loop"])
    except FT.FootageError as e:
        raise FT.FootageError(f"fx {fx['index']} ({fx['kind']}): {e}") from None
    return clip


def check_footage(fxs, footage_of):
    """Every clip the entries show (footage collages and video stickers, their `aspect.<output>` entries' too; a negative
    sticker shows none) opened and its span checked, before anything is drawn (`mk post` calls it before it encodes): a
    FootageError says what is wrong."""
    for fx in fxs:
        for e in [fx, *(fx.get("aspect") or {}).values()]:
            if e["kind"] in FX.CLIPS and e.get("clip") is not None:
                clip_of(e, footage_of)


# ================================================================================================= over a cut
class FrameFx(PF.PictureFx, MDF.MandalaFx, RMF.RemapFx, FL.FieldFx, ACF.ActivityFx, LBF.LabelFx, ERF.EraFx,
              SKF.StackFx, PAF.PaintAppFx, LZF.LaserFx, VSF.StickerFx, PRF.PrintsFx, CKF.KeyFx, SHF.ShapesFx,
              HZF.HazeFx, MTF.MitosisFx):
    """The frame effects of a plan's `fx` over one output's frames.

    size        (width, height) of the pictures
    frame_of    callable(Blender frame) -> picture or None: the cut's frame there, for echoes, slitscans, pushes, moshes
                and the poses before of a collage
    point_of    callable(key, frame) -> ((x, y) px, frame heights per metre at its depth): a projected scene point
    figure_of   callable(key, frame) -> coverage (h, w): a collage's figure, the figure a pixel sort or a mosh is masked by
                (mkmmd.cutfx.read_figure; None: those are an error)
    stem_of     callable(timeline, stem name) -> (mono float32 samples, sample rate, song seconds before clip time 0 in
                them), for scopes (None: a scope is an error); `stem_source` makes one for a project
    hits        the project's [audio] hits file ({lane: [...]}, mkmmd.core.lanes), for zooms, shocks and collages on a
                lane
    output      the output's name: entries draw as their `aspect.<output>` tables say (FX.for_output)
    footage_of  callable(clip) -> mkmmd.footage.Clip, for footage and video stickers (mkmmd.footage.source(root); None:
                those are an error)
    font_of     callable(font or None) -> a font file, for camcorders, booth captions, labels, an activity page's
                numbers and a test card's labels (mkmmd.picturefx.font_source; None: the asset registry's, and files as
                they are given)
    sprite_of   callable(refs) -> picture files, a mandala ring's sprites, a stack's bands', an activity page's and a
                paint app's stamps, a prints entry's photographs (mkmmd.mandala.sprite_source(root); None: the asset
                registry's, and files as they are given)
    plate_of    callable(shot, frame) -> picture or None: what `shot` shows at `frame` through its own camera and look
                (its plate, or the cut's frame where it is the shot in the cut), for a collage whose figure comes from
                its own `shot` (None: such a collage is an error)
    preload     read every sprite the effects use now, so a missing or unreadable one is an error before any frame is
                drawn (False: found now, read when drawn: mk post's workers, after the parent checked them)
    paths       where the pictures `frame_of` and `figure_of` read are on disk, for mkr-post (mkmmd.gpufx.Paths; None:
                the effects are drawn here only)
    breath      a breathing palette (mkmmd.post.Breath; None: the palette holds still): on each frame an entry is drawn
                as `entry_at` gives it, its colours mixed by that frame's mix. What a kind keeps from frame to frame is
                kept by the colours it was drawn in, never by the entry alone."""

    def __init__(self, fxs, size, *, fps, frame0, frame_of, point_of, figure_of=None, stem_of=None, hits=None,
                 output=None, footage_of=None, font_of=None, sprite_of=None, plate_of=None, preload=True, paths=None,
                 breath=None):
        self._preload = preload
        self.fxs = [FX.for_output(fx, output) for fx in fxs]
        self._breath, self._pairs = breath, {}           # an entry's index -> its Pair for this output, where it breathes
        for fx in fxs if breath is not None else ():
            p = breath.pairs.get(("fx", fx["index"]))
            if p is not None and not p.static:
                p = BR.Pair(FX.for_output(p.a, output), FX.for_output(p.b, output), f"fx {fx['index']}")
                if not p.static:
                    self._pairs[fx["index"]] = p
        self.size, self.fps, self.frame0, self.output = (int(size[0]), int(size[1])), float(fps), int(frame0), output
        self._frame_of, self._point_of, self._figure_of, self._plate_of = frame_of, point_of, figure_of, plate_of
        self._stem_of, self._hits, self._footage_of = stem_of, hits, footage_of
        self.paths = paths
        self.px = min(self.size) / 1080.0
        self._cache = OrderedDict()
        self._gain, self._times = {}, {}
        self._held = {}                  # a mosh's index -> (frame, the picture it holds there)
        self._flows = {}                 # a mosh's index -> (its made-up flow's block moves, block px)
        self._pivots = {}                # a collage's index -> the pixel its figure's moves turn about
        self._places = {}                # a collage's index -> (the map that sets its figure in the frame, its pivot)
        self._picture_init(font_of)
        self._mandala_init(sprite_of)
        self._stack_init()
        self._field_init()
        self._activity_init()
        self._label_init()
        self._era_init()
        self._paintapp_init()
        self._prints_init()
        self._key_init()
        self._mitosis_init()

    def active(self, frame):
        return bool(FX.at(self.fxs, frame))

    def safe(self, frame, rendered):
        """Whether this object draws `frame` exactly as a pass through every frame before it does, wherever it starts
        (fresh, or after other frames): no effect on `frame` holds a picture that only the frames before would have
        kept (a film's frame over its step, a booth's earlier captures, the frame an eraser holds, the picture a mosh
        with a flow carries; a mosh without one works its picture out again from the cut's frames, which `rendered`,
        the frames on disk, must all hold). mk post starts a worker's chunk of frames on such a frame."""
        for fx in FX.at(self.fxs, frame):
            kind = fx["kind"]
            if kind == "film" and PC.film_step(fx, frame)[1] != frame:
                return False
            if kind == "booth" and any(g < frame for _, g in PC.captures(fx, self._hit_times(fx), self.fps,
                                                                          self.frame0)):
                return False
            if kind == "paintapp" and any(e["frame"] <= frame < e["frame"] + e["frames"] for e in fx["eraser"]):
                return False
            if kind == "mosh" and (frame != fx["f0"] if fx["flow"] is not None else
                                   any(g not in rendered for g in range(max(fx["f0"] - 1, self.frame0), frame + 1))):
                return False
        return True

    def entry_at(self, fx, frame):
        """Entry `fx` (one of `fxs`) as `frame` draws it: with a breathing palette its colours mixed by the frame's mix
        (a copy only where they differ), else the entry itself."""
        p = self._pairs.get(fx["index"])
        return fx if p is None else p.at(self._breath.mix(frame))

    def entries(self, frame):
        """The effects on `frame`, in table order, each as `frame` draws it (entry_at)."""
        on = FX.at(self.fxs, frame)
        return [self.entry_at(fx, frame) for fx in on] if self._pairs else on

    def passing(self, fx, frame):
        """Whether entry `fx` is drawn on `frame` in colours strictly between its two palettes' (a breath on its way):
        what is drawn in them is seen on few frames, so mkr-post gets it for that frame alone (gpufx.figure.tile_files)."""
        return fx["index"] in self._pairs and 0.0 < self._breath.mix(frame) < 1.0

    def frame(self, frame, img):
        """`img` (the cut's frame `frame`, its cut effects composited) with every effect on that frame, in table order."""
        for fx in self.entries(frame):
            img = self._apply(fx, frame, img)
        return img

    def _apply(self, fx, frame, img):
        """`img` with the effect `fx` on `frame`, kept inside its `rect` (the frame outside it as it was, and a pixel the
        effect left as it was stays exactly so across a feathered edge: a figure a masked effect keeps)."""
        out = getattr(self, "_" + fx["kind"])(fx, frame, img)
        if fx.get("rect") is None:
            return out
        m = PF.rect_mask(self.size, tuple(fx["rect"]), fx["feather"] * self.px)[..., None]
        return (img + (out - img) * m).astype(np.float32)

    # ---------------------------------------------------------------- inputs
    def _get(self, f):
        if f in self._cache:
            self._cache.move_to_end(f)
            return self._cache[f]
        pic = self._frame_of(f) if f >= self.frame0 else None
        self._cache[f] = pic
        while len(self._cache) > CACHE:
            self._cache.popitem(last=False)
        return pic

    def _at(self, fx, frame):
        """(centre px, frame heights per metre at its depth or None) of an effect's `center` on `frame`."""
        w, h = self.size
        if fx["center"]["mode"] == "frame":
            x, y = fx["center"]["at"]
            return (x * w - 0.5, y * h - 0.5), None
        return self._point_of(FX.point_key(fx), frame)

    def _centre(self, fx, frame):
        """(centre px, radius px) of a lens or push on `frame`."""
        c, m = self._at(fx, frame)
        h = self.size[1]
        if "radius_m" in fx:
            return c, fx["radius_m"] * m * h
        return c, fx["radius"] * h

    def _hit_times(self, fx):
        if fx["index"] not in self._times:
            self._times[fx["index"]] = FX.hit_times(fx, self._hits)
        return self._times[fx["index"]]

    def _t(self, frame):
        return (frame - self.frame0) / self.fps

    # ---------------------------------------------------------------- kinds
    def lens_at(self, fx, frame):
        """(centre px, Einstein radius px) of a lens on `frame`: its mass goes as the strength."""
        c, theta = self._centre(fx, frame)
        return c, theta * np.sqrt(max(FX.strength(fx, self._t(frame)), 0.0))

    def _lens(self, fx, frame, img):
        return lens(img, *self.lens_at(fx, frame), fx["shadow"])

    def echo_at(self, fx, frame):
        """[(frame, opacity)] of an echo's copies on `frame`, nearest first, whether the cut has those frames or not."""
        a0 = fx["fade"] * FX.strength(fx, self._t(frame))
        return [(f, a0 * fx["fade"] ** j) for j, f in enumerate(FX.sources(fx, frame, self.frame0))]

    def _echo(self, fx, frame, img):
        copies = []
        for f, a in self.echo_at(fx, frame):
            pic = self._get(f)
            if pic is not None:
                copies.append((pic, a))
        return echo(img, copies, fx["blend"], fx["tint"])

    def slit_depth(self, fx, frame):
        """How many frames back a slitscan reaches on `frame` (under 1: none)."""
        return int(round(fx["depth"] * max(FX.strength(fx, self._t(frame)), 0.0)))

    def _slitscan(self, fx, frame, img):
        depth = self.slit_depth(fx, frame)
        if depth < 1:
            return img
        return slitscan(img, lambda k: self._get(frame - k), fx["now"], depth)

    def push_at(self, fx, frame):
        """(centre px, disc radius px, the frame inside, progress 0..1) of a push on `frame`."""
        c, r0 = self._centre(fx, frame)
        return c, r0, FX.sources(fx, frame, self.frame0)[0], FX.progress(fx, frame)

    def _push(self, fx, frame, img):
        """The frame inside is the cut's as it reached the push at its own time (the effects before the push in the
        table on it, PictureFx._reached): at the span's last frame the picture is that finished frame, the seam of a
        loop. A still is finished once and kept; one that plays is finished again on every frame."""
        c, r0, g, u = self.push_at(fx, frame)
        if fx["play"]:
            self._seen.pop(fx["index"], None)
        inner = self._reached(fx, g, frame, img)
        if inner is None:
            raise FileNotFoundError(f"fx {fx['index']} (push): the frame inside, {fx['inside_frame']}, is not there")
        return push(img, inner, c, r0, u, fx["ease"])

    def scope_at(self, fx, frame):
        """(trace points, centre px, radius px, line width px, opacity) of a scope on `frame` (draw_trace's)."""
        if self._stem_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (scope): no audio to draw")
        x, sr, pre = self._stem_of(fx["timeline"], fx["stem"])
        if fx["index"] not in self._gain:                       # one gain over the effect's span: the trace keeps its size
            a, b = int((pre + fx["from"]) * sr), int((pre + fx["to"]) * sr)
            peak = float(np.percentile(np.abs(x[max(a, 0):max(b, a + 1)]), 99.5)) if b > a else 0.0
            self._gain[fx["index"]] = fx["gain"] / max(peak, 1e-6)
        i = int(round((pre + self._t(frame)) * sr))
        pts = scope_points(x, i, int(fx["window"] * sr), int(round(fx["lag"] * sr)), self._gain[fx["index"]])
        w, h = self.size
        return (pts, (fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5), fx["size"] * h, fx["width"] * self.px,
                FX.strength(fx, self._t(frame)))

    def _scope(self, fx, frame, img):
        pts, centre, radius, width, opacity = self.scope_at(fx, frame)
        return draw_trace(img, pts, centre, radius, fx["color"], width, fx["glow"], opacity, self.px)

    def zoom_at(self, fx, frame):
        """(centre px, zoom factor) of a zoom on `frame`: punched in on each hit, eased back."""
        t = self._t(frame)
        k = float(FX.punch(t, self._hit_times(fx), fx["attack"], fx["decay"])) * FX.strength(fx, t)
        return self._at(fx, frame)[0], 1.0 + fx["amount"] * k

    def _zoom(self, fx, frame, img):
        return scale_about(img, *self.zoom_at(fx, frame))

    def shock_at(self, fx, frame):
        """(centre px, waves [(radius px, weight)], front width px, amount px) of a shock on `frame`."""
        t, h = self._t(frame), self.size[1]
        s = FX.strength(fx, t)
        waves = [(fx["speed"] * (t - th) * h, s * (1.0 - (t - th) / fx["life"]) ** 2)
                 for th in self._hit_times(fx) if 0.0 <= t - th < fx["life"]]
        return self._at(fx, frame)[0], waves, fx["width"] * h, fx["amount"] * h

    def _shock(self, fx, frame, img):
        return shock(img, *self.shock_at(fx, frame), fx["color"], fx["glow"])

    def slab_at(self, fx, frame):
        """(centres px, width px, height px, opacity) of a slab on `frame`: where it stands at its shutter's samples."""
        w, h = self.size
        sw, sh = fx["size"][0] * h, fx["size"][1] * h
        C = np.array([fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5])
        v = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}[fx["dir"]]
        rb = 0.5 * float(np.hypot(sw + abs(sh * np.tan(np.radians(fx["slant"]))), sh))   # its reach any way
        ahead = (w - C[0]) if v[0] > 0 else C[0] if v[0] < 0 else (h - C[1]) if v[1] > 0 else C[1]
        behind = (w if v[0] else h) - ahead
        t = self._t(frame)
        n = SLAB_SAMPLES if fx["shutter"] > 0 else 1
        times = [t + fx["shutter"] / self.fps * ((j + 0.5) / n - 0.5) for j in range(n)] if n > 1 else [t]
        centres = []
        for tj in times:
            u = FX.slab_offset(fx, tj)
            d = u * ((behind + rb) if u < 0 else (ahead + rb))
            centres.append((C[0] + v[0] * d, C[1] + v[1] * d))
        return centres, sw, sh, fx["alpha"] * FX.strength(fx, t)

    def _slab(self, fx, frame, img):
        centres, sw, sh, alpha = self.slab_at(fx, frame)
        cov = slab_cover(self.size, centres, sw, sh, fx["slant"], fx["angle"])
        a = (alpha * cov)[..., None]
        return (img * (1.0 - a) + np.asarray(fx["color"], np.float32) * a).astype(np.float32)

    def _tint(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        if s <= 0.0:
            return img
        return (img * (1.0 - s) + gradient_map(img, fx["dark"], fx["color"], fx["light"]) * s).astype(np.float32)

    def _halftone(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        if s <= 0.0:
            return img
        out = halftone(img, fx["cell"] * self.px, fx["angle"], fx["level"], fx["ink"], fx["lift"],
                       (fx["offset"][0] * self.px, fx["offset"][1] * self.px))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def split_at(self, fx, frame):
        """The bands a split tears `frame` into (split_shifts), None when its strength is 0 there. The tear holds `hold`
        seconds, then the signal breaks another way (seeded: the same frames again break the same); the strength
        scales how far it tears."""
        t = self._t(frame)
        s = max(FX.strength(fx, t), 0.0)
        if s <= 0.0:
            return None
        step = max(int(np.floor((t - fx["from"]) / fx["hold"] + 1e-6)), 0)
        rng = np.random.default_rng([fx["seed"], fx["index"], step])
        return split_shifts(self.size[1], rng, fx["bands"], fx["shift"] * self.size[0] * s, fx["share"],
                            fx["rgb"] * self.px * s)

    def _split(self, fx, frame, img):
        bands = self.split_at(fx, frame)
        return img if bands is None else split_rolled(img, bands)

    def _mosh(self, fx, frame, img):
        """The held picture mixed over the frame by the strength, and with a `mask` by the figure's coverage too: the
        figure's outline as antialiased as it is, a figure under a ground mask left exactly as it was."""
        held = self._moshed(fx, frame, img)
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        if s <= 0.0:
            return img
        region = self._masked(fx, frame)
        if region is None:
            return (img * (1.0 - s) + held * s).astype(np.float32)
        a = (np.float32(s) * region)[..., None]
        return (img * (1.0 - a) + held * a).astype(np.float32)

    def _moshed(self, fx, frame, img):
        """The picture a mosh holds at `frame`. With a `flow`: the span's first frame as it reached the mosh, carried
        `steps` steps along the flow's block moves on that frame and as many more on each frame after (a P-frame played
        again and again: a still moshes too). Without: the cut's frame before its span (its first when the clip starts
        inside it) as it reached the mosh at its own time (the effects before the mosh in the table on it: the
        outgoing picture keeps its grade), carried along the cut's own motion from each frame to the next since its
        first frame (the motion across the cut itself is the key frame a mosh loses). One step on from the frame before
        when that was the last asked, else worked out again from the start."""
        i, f0 = fx["index"], fx["f0"]
        state = self._held.get(i)
        if fx["flow"] is not None:
            moves, b = self._flow(fx)
            if state is not None and state[0] == frame - 1:
                held, n = state[1], 1
            else:
                held, n = self._reached(fx, f0, frame, img), frame - f0 + 1
                held = img if held is None else held
            held = mosh_carry(held, moves, b, n * fx["flow"]["steps"])
            self._held[i] = (frame, held)
            return held
        pic = lambda g: (img if g == frame else None) if self._get(g) is None else self._get(g)     # noqa: E731
        if state is not None and state[0] == frame - 1:
            held, first = state[1], frame
        else:
            held, first = self._reached(fx, f0 - 1, frame, img), f0 + 1
            if held is None:
                held = self._reached(fx, f0, frame, img)
        for g in range(max(first, f0 + 1), frame + 1):
            a, b = pic(g - 1), pic(g)
            if a is not None and b is not None and held is not None:
                held = mosh_step(held, a, b, fx["block"] * self.px)
        held = img if held is None else held
        self._held[i] = (frame, held)
        return held

    def _flow(self, fx):
        """(block moves, block px) of a mosh's made-up flow on this output's pictures (mosh_flow); with a `mask`, the
        blocks whose source holds anything of the rest on the span's first frame (the figure's outline included) are
        kept still (mosh_stay): the figure's colours never smear into the ground, nor the ground's into her."""
        got = self._flows.get(fx["index"])
        if got is None:
            moves, b = mosh_flow(fx["flow"], self.size, fx["block"] * self.px, self.px, fx["index"])
            region = self._masked(fx, fx["f0"])
            if region is not None:
                moves = mosh_stay(moves, b, region < MOSH_SOLID)
            got = self._flows[fx["index"]] = (moves, b)
        return got

    # ---------------------------------------------------------------- collages
    def _figure(self, fx, frame):
        """The coverage of a collage's figure on `frame` (a FileNotFoundError when its layer is missing), None before the
        clip: through the shot in the cut, or through its own `shot`."""
        if frame < self.frame0:
            return None
        if self._figure_of is None:
            raise FileNotFoundError(f"fx {fx['index']} ({fx['kind']}): no figure layers to read")
        return self._figure_of(FX.figure_key(fx), frame)

    def _shot_frame(self, fx, g):
        """What a collage's own `shot` shows at frame `g`, the picture its figure is cut from (a FileNotFoundError when it
        is not there)."""
        if self._plate_of is None:
            raise FileNotFoundError(f"fx {fx['index']} ({fx['kind']}): no plates to read (its figure comes from shot "
                                    f"{fx['shot']!r})")
        pic = self._plate_of(fx["shot"], g)
        if pic is None:
            raise FileNotFoundError(f"fx {fx['index']} ({fx['kind']}): shot {fx['shot']!r} at frame {g}: its plate is "
                                    f"missing (mk render draws it)")
        return pic

    def _collage(self, fx, frame, img, draw):
        """`img` with the collage `draw(k, pic, m)` makes (k the hits so far; the figure's picture and coverage on
        `frame`, cut from the frame or from what its own `shot` shows, set into the frame and moved by its `keys`) mixed
        in by the strength."""
        pic, m = self._posed(fx, frame, self._shot_frame(fx, frame) if fx.get("shot") else img,
                             self._figure(fx, frame))
        out = draw(FX.step(fx, frame, self._hit_times(fx), self.fps, self.frame0), pic, m)
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _posed(self, fx, g, pic, m):
        """A collage's figure (picture or None, coverage) on frame `g` set into the frame and moved by its keys there
        (_pose_map)."""
        if m is None:
            return pic, m
        M = self._pose_map(fx, g)
        if M is None:
            return pic, m
        return (None if pic is None else _warp(pic, M, self.size)), _warp(m, M, self.size)

    def _pose_map(self, fx, g):
        """The 2 x 3 map that sets a collage's figure into the frame on frame `g`, by `at` and `height` (the same map on
        every frame: _placing), and moves it by its `keys` at that frame: turned, scaled and shifted about its pivot (as
        it is at rest); None for a figure left where its frame shows it."""
        M, pivot = self._placing(fx)
        if fx["keys"]:
            mv = FX.move_at(fx["keys"], self._t(g))
            if mv != FX.MOVE:
                K = FX.move_matrix(mv, self._pivot(fx) if pivot is None else pivot, self.size[1])
                M = K if M is None else FX.compose(K, M)
        return M

    def _placing(self, fx):
        """(map, pivot) of a collage whose `at` or `height` sets its figure into the frame: the 2 x 3 map from the
        figure's own frame that scales it about its pivot (found on the span's first frame) so that its box there is
        `height` of the frame's height tall, and moves the pivot to `at` (either left as it is when not given), and where
        the pivot lands. One map for the whole span: the figure moves in the frame as it moves in its own, nothing jumps
        when its box changes (an arm strumming). (None, None) for a figure left where its frame shows it."""
        got = self._places.get(fx["index"])
        if got is None:
            got = (None, None)
            if FX.placed(fx):
                a = self._pivot(fx)
                w, h = self.size
                s = 1.0 if fx["height"] is None else fx["height"] * h / self._tall(fx)
                to = a if fx["at"] is None else (fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5)
                got = (FX.placement(a, to, s), to)
            self._places[fx["index"]] = got
        return got

    def _tall(self, fx):
        """The height in px of a collage's figure, its box where it covers half a pixel or more, on the span's first
        frame."""
        m = self._figure(fx, fx["f0"])
        rows = np.nonzero((m >= 0.5).any(1))[0] if m is not None else np.zeros(0, int)
        if not rows.size:
            raise FX.FxError(f"fx {fx['index']} ({fx['kind']}): height: the figure is not in the frame on the span's "
                             f"first frame ({fx['f0']}), which sizes it")
        return float(rows[-1] - rows[0] + 1)

    def _pivot(self, fx):
        """The pixel a collage's figure turns and scales about: `pivot` [x, y] (fractions of the figure's own frame), or
        found on the figure on the span's first frame, the middle of its box's bottom edge (`base`) or its centroid
        (`center`)."""
        got = self._pivots.get(fx["index"])
        if got is None:
            w, h = self.size
            pv = fx["pivot"]
            if not isinstance(pv, str):
                got = (pv[0] * w - 0.5, pv[1] * h - 0.5)
            else:
                m = self._figure(fx, fx["f0"])
                if m is None or not (m >= 0.5).any():
                    raise FX.FxError(f"fx {fx['index']} ({fx['kind']}): pivot {pv!r}: the figure is not in the frame on "
                                     f"the span's first frame ({fx['f0']}): give pivot = [x, y]")
                if pv == "center":
                    got = centroid(m)
                else:
                    ys, xs = np.nonzero(m >= 0.5)
                    got = ((xs.min() + xs.max()) / 2.0, ys.max() + 0.5)
            self._pivots[fx["index"]] = got
        return got

    def _edge(self, fx):
        e = fx["edge"]
        return None if e is None else (e["color"], e["width"] * self.px)

    def _shadow(self, fx):
        s, h = fx["shadow"], self.size[1]
        return None if s is None else (s["color"], (s["offset"][0] * h, s["offset"][1] * h), s["blur"] * self.px,
                                       s["alpha"])

    def _fill(self, fx, i, values=None):
        """Entry `i` of a collage's fills (`values`, default its `fill`) as the picture code takes it: None (its picture),
        a colour, or a silkscreen {inks: entry i of `inks`, offset, soft and line px}."""
        f = FX.pick(fx["fill"] if values is None else values, i)
        if f != FX.SILK:
            return f
        return {"inks": FX.pick(fx["inks"], i), "offset": (fx["offset"][0] * self.px, fx["offset"][1] * self.px),
                "soft": SILK_SOFT * self.px, "line": SILK_LINE * self.px}

    def _before(self, fx, frame, j, fill):
        """(picture, coverage) of a collage's copy or panel j, `delay` frames apart: the figure on frame `frame` - j x
        delay, moved by its keys there, its picture only when it is drawn from it (`fill` None or a silkscreen); (None,
        None) before the clip."""
        g = frame - j * fx["delay"]
        m = self._figure(fx, g)
        if m is None:
            return None, None
        if fill is not None and fill != FX.SILK:
            return self._posed(fx, g, None, m)
        pic = self._shot_frame(fx, g) if fx.get("shot") else self._get(g)
        return self._posed(fx, g, pic, m) if pic is not None else (None, None)

    def _grid(self, fx, frame, img):
        def draw(k, pic, m):
            piece = None if fx["piece"] is None else (FX.piece_cell(fx, k), fx["piece"])
            boxes = FX.grid_cells(fx, frame, self._hit_times(fx), self.fps, self.frame0)
            return cells(pic, m, boxes, lambda i, j: self._back(fx, frame, k + i + j, img),
                         lambda i, j: self._fill(fx, k + i + j), piece, self._edge(fx), self._shadow(fx))
        return self._collage(fx, frame, img, draw)

    def _orbit(self, fx, frame, img):
        def draw(k, pic, m):
            theta0 = math.radians(fx["spin"] * (self._t(frame) - fx["from"]) + fx["turn"] * k)
            return orbit(pic, m, self._back(fx, frame, k, img), self._at(fx, frame)[0], fx["radius"] * min(self.size),
                         fx["tilt"], theta0, fx["count"], fx["scale"], fx["size"], lambda j: self._fill(fx, k + j),
                         self._edge(fx), self._shadow(fx))
        return self._collage(fx, frame, img, draw)

    def _mirror(self, fx, frame, img):
        def draw(k, pic, m):
            w, h = self.size
            place = None if fx["place"] is None else (fx["place"][0] * w - 0.5, fx["place"][1] * h - 0.5)
            return mirror(pic, m, self._at(fx, frame)[0], FX.pick(fx["axes"], k),
                          lambda j: self._back(fx, frame, k + j, img),
                          lambda j: self._fill(fx, k + j), place, self._edge(fx), self._shadow(fx))
        return self._collage(fx, frame, img, draw)

    def _panels(self, fx, frame, img):
        def draw(k, pic, m):
            figures = [(pic, m)]
            for j in range(1, fx["count"]):
                p, cov = (pic, m) if fx["delay"] == 0 else self._before(fx, frame, j, FX.pick(fx["fill"], k + j))
                figures.append(None if cov is None else (p, cov))
            return panels(self.size, fx["count"], fx["slant"], figures, lambda j: self._back(fx, frame, k + j, img),
                          lambda j: self._fill(fx, k + j), fx["size"], self._edge(fx), self._shadow(fx))
        return self._collage(fx, frame, img, draw)

    def _footage(self, fx, frame, img):
        """The figure laid over the clip's picture at this frame (fitted to the output, `back` round a contained one: a
        colour, a pattern or the picture as it reached the effect)."""
        if self._footage_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (footage): no footage to read")
        clip = clip_of(fx, self._footage_of)

        def draw(k, pic, m):
            plate = FL.plate(clip, (frame - fx["f0"]) / self.fps, self.size, fx, self._back(fx, frame, k, img))
            return _paste(plate, pic, m, self._fill(fx, k), edge=self._edge(fx), shadow=self._shadow(fx))
        return self._collage(fx, frame, img, draw)
