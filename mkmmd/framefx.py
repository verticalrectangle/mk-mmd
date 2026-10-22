"""Frame effects on the finished pictures (docs/design.md: Shots, Frame effects): the picture work of the `[[fx]]` entries
that mkmmd.core.framefx checks and times. `mk post` hands each frame of the cut to FrameFx.frame after the cut's own
effects (mkmmd.cutfx) and before the screen type and the grade; `mk look` does the same for the frames it draws. Images are
float RGB in display space (0..1), (h, w, 3); a figure is its coverage, (h, w) in 0..1 over the same picture."""
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .core import framefx as FX
from .core import lanes as LN

GLOW_SIGMA = 6.0                 # px at 1080 on the short side: the scope trace's glow
LINE_SHIFT = 4                   # fixed-point bits of the trace's points (sub-pixel lines)
CACHE = 64                       # decoded frames kept for echoes and slitscans
SLAB_SAMPLES = 8                 # positions a moving slab is drawn at over its shutter (motion blur)
SHOCK_RIM = 0.6                  # the lit front of a shock ring, as a share of its width
MOSH_SCALE = 0.25                # a mosh's optical flow is worked out on pictures this share of the frame's size
DOT = 0.72                       # a halftone dot's radius at full darkness, in cells: the dots merge into solid ink
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SHRINK_BY_AREA = 0.75            # a figure laid smaller than this is shrunk by area first (no shimmer in fine hair)
DEPTH_SCALE = 0.3                # how much larger an orbit's nearest copy is, and smaller its farthest, at a tilt of 90


def scale_about(img, c, z):
    """The picture zoomed `z` times about pixel `c` (x, y; z > 1 is in), the frame's edge reflected where it runs out."""
    if abs(z - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = np.array([[1.0 / z, 0.0, c[0] * (1.0 - 1.0 / z)], [0.0, 1.0 / z, c[1] * (1.0 - 1.0 / z)]], np.float32)
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


def slab_cover(size, centres, w, h, slant, angle):
    """Coverage (rows, cols) 0..1 of a parallelogram `w` x `h` px whose sides lean `slant` degrees (to the right going up,
    as italics do) and which is turned `angle` degrees counter-clockwise on screen, averaged over the pixel `centres`
    [(x, y), ...] it stands at in turn (one: still; several along its way: blurred by its motion)."""
    W, H = int(size[0]), int(size[1])
    out = np.zeros((H, W), np.float32)
    a, t = np.radians(angle), np.tan(np.radians(slant))
    ca, sa, cs = np.float32(np.cos(a)), np.float32(np.sin(a)), np.float32(np.cos(np.radians(slant)))
    reach = 0.5 * (w + abs(h * t)) + 0.5 * h + 2.0                       # the slab's half-extent, any way, plus the edge
    for cx, cy in centres:
        x0, x1 = max(int(np.floor(cx - reach)), 0), min(int(np.ceil(cx + reach)) + 1, W)
        y0, y1 = max(int(np.floor(cy - reach)), 0), min(int(np.ceil(cy + reach)) + 1, H)
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


def push(img, inner, c, r0, u, ease="log"):
    """The picture zooming into a disc of radius r0 (px) about `c` that shows `inner`: at u = 0 the picture as it is, at
    u = 1 the disc, its centre glided to the frame's middle, covers the frame and `inner` is shown at its own size."""
    h, w = img.shape[:2]
    m = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    R = float(np.hypot(w, h)) / 2.0 + 1.0
    z, g = zoom(np.log(max(R / max(r0, 1e-6), 1.0)), float(u), ease)
    p = np.asarray(c, float) + (m - np.asarray(c, float)) * g                       # the disc's centre on screen
    r = r0 * z
    s = r / R                                                                        # the inner frame's scale
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


def draw_trace(img, pts, centre, radius, colour, width, glow, opacity=1.0, px=1.0):
    """`img` with the trace `pts` (n, 2, -1..1; y up) drawn as a line of `width` px about `centre` (px) at `radius` px,
    added in `colour` with a glow of share `glow`."""
    h, w = img.shape[:2]
    if len(pts) < 2 or opacity <= 0:
        return img
    q = np.stack([centre[0] + pts[:, 0] * radius, centre[1] - pts[:, 1] * radius], 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.rint(q * (1 << LINE_SHIFT)).astype(np.int32)], False, 255, max(1, int(round(width))),
                  cv2.LINE_AA, shift=LINE_SHIFT)
    a = mask.astype(np.float32) / 255.0
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


def split(img, rng, bands, shift, share, rgb):
    """A broken digital signal (`shift` and `rgb` in px, `rng` a numpy Generator): the frame cut into `bands` bands of
    random heights, `share` of them torn sideways by up to `shift` px (what leaves one side comes in at the other), the
    red and blue planes pulled `rgb` px apart in a torn band and a third of that everywhere else."""
    h = img.shape[0]
    n = min(int(bands), h)
    cuts = np.sort(rng.choice(np.arange(1, h), size=n - 1, replace=False)) if n > 1 else np.zeros(0, int)
    edges = np.concatenate([[0], cuts, [h]]).astype(int)
    base = int(round(rgb / 3.0))
    out = np.empty_like(img)
    out[..., 0], out[..., 1], out[..., 2] = np.roll(img[..., 0], base, 1), img[..., 1], np.roll(img[..., 2], -base, 1)
    for a, b in zip(edges[:-1], edges[1:]):
        if rng.random() >= share:
            continue
        dx, s = int(round(rng.uniform(-1.0, 1.0) * shift)), int(round(rgb))
        band = img[a:b]
        out[a:b, :, 0] = np.roll(band[..., 0], dx + s, 1)
        out[a:b, :, 1] = np.roll(band[..., 1], dx, 1)
        out[a:b, :, 2] = np.roll(band[..., 2], dx - s, 1)
    return out


def _grey8(img, size):
    small = cv2.resize(np.ascontiguousarray(img, np.float32), size, interpolation=cv2.INTER_AREA)
    return (np.clip(small @ LUMA, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def mosh_step(held, prev, cur, block):
    """`held` carried one frame along the motion from `prev` to `cur` (the cut's own pictures): Farneback optical flow on
    grey pictures at MOSH_SCALE of the frame, averaged over `block` px squares so the picture moves in chunks as a
    codec's motion vectors move it, each pixel taking what stood `flow` behind it."""
    h, w = cur.shape[:2]
    sw, sh = max(8, int(round(w * MOSH_SCALE))), max(8, int(round(h * MOSH_SCALE)))
    flow = cv2.calcOpticalFlowFarneback(_grey8(prev, (sw, sh)), _grey8(cur, (sw, sh)), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    b = max(1, int(round(block * MOSH_SCALE)))
    gh, gw = -(-sh // b), -(-sw // b)
    pad = np.pad(flow, ((0, gh * b - sh), (0, gw * b - sw), (0, 0)), mode="edge")
    blocks = np.ascontiguousarray(pad.reshape(gh, b, gw, b, 2).mean((1, 3)), np.float32)
    full = cv2.resize(blocks, (gw * b, gh * b), interpolation=cv2.INTER_NEAREST)[:sh, :sw]
    full = cv2.resize(full, (w, h), interpolation=cv2.INTER_NEAREST) / np.float32(MOSH_SCALE)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(held, np.float32), xs - full[..., 0], ys - full[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ================================================================================================= collages
def centroid(m):
    """The centre (x, y) in pixels of a coverage (h, w), its coverage-weighted mean; None when nothing is covered."""
    mo = cv2.moments(np.ascontiguousarray(m, np.float32))
    if mo["m00"] < 1.0:
        return None
    return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]


def _flat(size, colour):
    w, h = size
    return np.full((h, w, 3), colour, np.float32)


def _grow(a, width):
    """Coverage `a` grown by `width` px all round: laid under a figure it shows as an outline."""
    r = max(int(round(width)), 1)
    return cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def _laid(src, s, a, p, size):
    """`src` ((h, w) or (h, w, 3)) scaled `s` times about pixel `a`, with `a` put at pixel `p`, in a picture of `size`
    (nothing outside it)."""
    w, h = size
    if abs(s - 1.0) < 1e-9 and abs(p[0] - a[0]) < 1e-6 and abs(p[1] - a[1]) < 1e-6 and src.shape[:2] == (h, w):
        return src
    if s < SHRINK_BY_AREA:
        sw, sh = max(int(round(src.shape[1] * s)), 1), max(int(round(src.shape[0] * s)), 1)
        kx, ky = sw / src.shape[1], sh / src.shape[0]
        src = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        M = np.float32([[1.0, 0.0, p[0] - ((a[0] + 0.5) * kx - 0.5)], [0.0, 1.0, p[1] - ((a[1] + 0.5) * ky - 0.5)]])
    else:
        M = np.float32([[s, 0.0, p[0] - s * a[0]], [0.0, s, p[1] - s * a[1]]])
    return cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _paste(dst, pic, m, fill, s=1.0, a=(0.0, 0.0), p=(0.0, 0.0), edge=None):
    """`dst` with a figure laid on it: coverage `m` scaled `s` about pixel `a` and put with `a` at pixel `p`, drawn as its
    picture `pic` (`fill` None) or in the flat colour `fill`, over an `edge` (colour, width px) grown round it."""
    size = (dst.shape[1], dst.shape[0])
    cov = _laid(m, s, a, p, size)
    if edge is not None and edge[1] > 0:
        g = _grow(cov, edge[1])[..., None]
        dst = dst * (1.0 - g) + np.asarray(edge[0], np.float32) * g
    if fill is None:
        rgb = _laid(pic * m[..., None], s, a, p, size)
    else:
        rgb = np.asarray(fill, np.float32) * cov[..., None]
    return (rgb + dst * (1.0 - cov[..., None])).astype(np.float32)


def grid(img, m, n, colour_at, fill_at, piece=None, edge=None):
    """The frame shrunk into `n` x `n` cells, each the figure (coverage `m` of `img`) on a flat colour: colour_at(i, j)
    and fill_at(i, j) (None: the figure's picture, else a flat colour) of the cell in column i, row j; `piece` ((i, j),
    colour) draws that cell's figure flat in its colour; an `edge` (colour, width px) round each figure."""
    h, w = m.shape
    cw, ch = -(-w // n), -(-h // n)
    sm = cv2.resize(np.ascontiguousarray(m, np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    sp = cv2.resize(img * m[..., None], (cw, ch), interpolation=cv2.INTER_AREA)
    ring = None if edge is None or edge[1] <= 0 else _grow(sm, edge[1])[..., None]
    out = np.empty((ch * n, cw * n, 3), np.float32)
    for j in range(n):
        for i in range(n):
            tile = _flat((cw, ch), colour_at(i, j))
            if ring is not None:
                tile = tile * (1.0 - ring) + np.asarray(edge[0], np.float32) * ring
            fill = piece[1] if piece is not None and tuple(piece[0]) == (i, j) else fill_at(i, j)
            rgb = sp if fill is None else np.asarray(fill, np.float32) * sm[..., None]
            out[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw] = rgb + tile * (1.0 - sm[..., None])
    return out if out.shape[:2] == (h, w) else cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)


def orbit(img, m, back, centre, radius, tilt, theta0, count, scale, size, fill_at, edge=None):
    """The figure (coverage `m` of `img`) moved to pixel `centre` on a flat `back`, `scale` times its size, and `count`
    copies of it, `size` times as large again, on a ring of `radius` px round it tilted `tilt` degrees from face-on: an
    ellipse whose near half (the lower on screen) is drawn over the figure and larger, its far half under it and smaller.
    Copy j (1 .. count) stands at angle theta0 + 2 pi (j - 1) / count, radians clockwise on screen from the right;
    fill_at(0) is the figure's fill and fill_at(j) copy j's (None: its picture)."""
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
        out = _paste(out, img, m, fill_at(j), s, a, p, edge)
    return out


def mirror(img, m, centre, axes, colour_at, fill_at, place=None, edge=None):
    """The figure (coverage `m` of `img`; its centroid moved to pixel `place` when one is given) on flat colour, folded
    about the vertical line through pixel `centre` (`axes` "x"), the horizontal one ("y") or both ("xy"): the side or
    quarter the figure's centroid is on is shown as it is, the others are its reflections. Region j (0 that side, 1
    across the vertical line, 2 across the horizontal, 3 across both) is colour_at(j) with the figure drawn fill_at(j)
    (None: its picture)."""
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
        comp = _flat((w, h), bg) if a is None else _paste(_flat((w, h), bg), img, m, fill_at(j), 1.0, a, p, edge)
        if j:
            comp = cv2.remap(comp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=tuple(float(c) for c in bg))
        sel = region == j
        out[sel] = comp[sel]
    return out


def panels(size, n, slant, figures, colour_at, fill_at, scale=1.0, edge=None):
    """The frame cut into `n` panels across its long side (columns of a wide or square frame, rows of a tall one) by lines
    leaning `slant` degrees (a column's edges to the right going up, as italics do; a row's rising to the right): panel j
    is a flat colour_at(j) with figures[j] = (picture, coverage) on it, moved so its centroid sits on the panel's middle
    line and `scale` times its size (and n times smaller in the rows of a tall frame, so a figure framed for the whole
    frame fits one), drawn fill_at(j) (None: its picture); a figure None leaves its panel plain."""
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
            tile = _paste(tile, fig[0], fig[1], fill_at(j), s, a, p, edge)
        out += tile * cov[..., None]
    return out


def stack(img, m, back, copies, step, scale, centre, fill, edge=None):
    """The figure (coverage `m` of `img`, drawn `fill`: None its picture) on a flat `back`, in front of a stack of copies,
    the farthest drawn first: copies[j - 1] = (picture, coverage, fill) is copy j, `scale` ** j times as large about pixel
    `centre` and j x `step` (dx, dy) px further; a copy without coverage is left out."""
    h, w = m.shape
    out = _flat((w, h), back)
    c = (float(centre[0]), float(centre[1]))
    for j in range(len(copies), 0, -1):
        pic, cov, f = copies[j - 1]
        if cov is not None:
            out = _paste(out, pic, cov, f, scale ** j, c, (c[0] + j * step[0], c[1] + j * step[1]), edge)
    return _paste(out, img, m, fill, edge=edge)


def stem_source(root, cache_dir):
    """callable(timeline, stem) -> (mono float32 samples, sample rate, pre) for a project at `root`: the stems of the
    window `timeline` (project-relative) was analysed on, from the stem cache `cache_dir` only (mkmmd.timeline.hits
    .cached_stems), each loaded once; the timeline's audio file, like every path of a project, is project-relative.
    A FileNotFoundError says what to run when they are not there."""
    import json
    import subprocess
    from pathlib import Path

    from .timeline import hits as H
    from .timeline import signal as S
    loaded = {}

    def get(timeline, stem):
        if timeline not in loaded:
            p = Path(root) / Path(timeline).expanduser()
            try:
                tl = json.loads(p.read_text(encoding="utf-8"))
                au = dict(tl.get("audio") or {})
                if au.get("file"):
                    f = Path(au["file"]).expanduser()
                    au["file"] = str(f if f.is_absolute() else Path(root) / f)
                st, pre = H.cached_stems({**tl, "audio": au}, str(cache_dir))
            except (OSError, ValueError, subprocess.CalledProcessError) as e:     # HitsError is a ValueError
                raise FileNotFoundError(f"scope: {p}: {e}") from None
            loaded[timeline] = ({k: np.asarray(v, np.float32).mean(0) for k, v in st.items()}, pre)
        st, pre = loaded[timeline]
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


# ================================================================================================= over a cut
class FrameFx:
    """The frame effects of a plan's `fx` over one output's frames.

    size       (width, height) of the pictures
    frame_of   callable(Blender frame) -> picture or None: the cut's frame there, for echoes, slitscans, pushes, moshes
               and the poses before of a collage
    point_of   callable(key, frame) -> ((x, y) px, frame heights per metre at its depth): a projected scene point
    figure_of  callable(key, frame) -> coverage (h, w): a collage's figure (mkmmd.cutfx.read_figure; None: a collage is
               an error)
    stem_of    callable(timeline, stem name) -> (mono float32 samples, sample rate, song seconds before clip time 0 in
               them), for scopes (None: a scope is an error); `stem_source` makes one for a project
    hits       the project's [audio] hits file ({lane: [...]}, mkmmd.core.lanes), for zooms, shocks and collages on a
               lane"""

    def __init__(self, fxs, size, *, fps, frame0, frame_of, point_of, figure_of=None, stem_of=None, hits=None):
        self.fxs, self.size, self.fps, self.frame0 = fxs, (int(size[0]), int(size[1])), float(fps), int(frame0)
        self._frame_of, self._point_of, self._figure_of = frame_of, point_of, figure_of
        self._stem_of, self._hits = stem_of, hits
        self.px = min(self.size) / 1080.0
        self._cache = OrderedDict()
        self._gain, self._times = {}, {}
        self._held = {}                  # a mosh's index -> (frame, the picture it holds there)

    def active(self, frame):
        return bool(FX.at(self.fxs, frame))

    def frame(self, frame, img):
        """`img` (the cut's frame `frame`, its cut effects composited) with every effect on that frame, in table order."""
        for fx in FX.at(self.fxs, frame):
            img = getattr(self, "_" + fx["kind"])(fx, frame, img)
        return img

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
    def _lens(self, fx, frame, img):
        c, theta = self._centre(fx, frame)
        return lens(img, c, theta * np.sqrt(max(FX.strength(fx, self._t(frame)), 0.0)), fx["shadow"])

    def _echo(self, fx, frame, img):
        a0 = fx["fade"] * FX.strength(fx, self._t(frame))
        copies = []
        for j, f in enumerate(FX.sources(fx, frame, self.frame0)):
            pic = self._get(f)
            if pic is not None:
                copies.append((pic, a0 * fx["fade"] ** j))
        return echo(img, copies, fx["blend"], fx["tint"])

    def _slitscan(self, fx, frame, img):
        depth = int(round(fx["depth"] * max(FX.strength(fx, self._t(frame)), 0.0)))
        if depth < 1:
            return img
        return slitscan(img, lambda k: self._get(frame - k), fx["now"], depth)

    def _push(self, fx, frame, img):
        c, r0 = self._centre(fx, frame)
        inner = self._get(FX.sources(fx, frame, self.frame0)[0])
        if inner is None:
            raise FileNotFoundError(f"fx {fx['index']} (push): the frame inside, {fx['inside_frame']}, is not there")
        return push(img, inner, c, r0, FX.progress(fx, frame), fx["ease"])

    def _scope(self, fx, frame, img):
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
        return draw_trace(img, pts, (fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5), fx["size"] * h, fx["color"],
                          fx["width"] * self.px, fx["glow"], FX.strength(fx, self._t(frame)), self.px)

    def _zoom(self, fx, frame, img):
        t = self._t(frame)
        k = float(FX.punch(t, self._hit_times(fx), fx["attack"], fx["decay"])) * FX.strength(fx, t)
        return scale_about(img, self._at(fx, frame)[0], 1.0 + fx["amount"] * k)

    def _shock(self, fx, frame, img):
        t, h = self._t(frame), self.size[1]
        s = FX.strength(fx, t)
        waves = [(fx["speed"] * (t - th) * h, s * (1.0 - (t - th) / fx["life"]) ** 2)
                 for th in self._hit_times(fx) if 0.0 <= t - th < fx["life"]]
        return shock(img, self._at(fx, frame)[0], waves, fx["width"] * h, fx["amount"] * h, fx["color"], fx["glow"])

    def _slab(self, fx, frame, img):
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
        cov = slab_cover(self.size, centres, sw, sh, fx["slant"], fx["angle"])
        a = (fx["alpha"] * FX.strength(fx, t) * cov)[..., None]
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

    def _split(self, fx, frame, img):
        """The tear holds `hold` seconds, then the signal breaks another way (seeded: the same frames again break the
        same); the strength scales how far it tears."""
        t = self._t(frame)
        s = max(FX.strength(fx, t), 0.0)
        if s <= 0.0:
            return img
        step = max(int(np.floor((t - fx["from"]) / fx["hold"] + 1e-6)), 0)
        rng = np.random.default_rng([fx["seed"], fx["index"], step])
        return split(img, rng, fx["bands"], fx["shift"] * self.size[0] * s, fx["share"], fx["rgb"] * self.px * s)

    def _mosh(self, fx, frame, img):
        held = self._moshed(fx, frame, img)
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        return img if s <= 0.0 else (img * (1.0 - s) + held * s).astype(np.float32)

    def _moshed(self, fx, frame, img):
        """The picture a mosh holds at `frame`: the cut's frame before its span (its first when the clip starts inside
        it), carried along the cut's own motion from each frame to the next since its first frame (the motion across the
        cut itself is the key frame a mosh loses); one step on from the frame before when that was the last asked, else
        worked out again from the start."""
        i, f0 = fx["index"], fx["f0"]
        pic = lambda g: (img if g == frame else None) if self._get(g) is None else self._get(g)     # noqa: E731
        state = self._held.get(i)
        if state is not None and state[0] == frame - 1:
            held, first = state[1], frame
        else:
            held, first = self._get(f0 - 1), f0 + 1
            if held is None:
                held = pic(f0)
        for g in range(max(first, f0 + 1), frame + 1):
            a, b = pic(g - 1), pic(g)
            if a is not None and b is not None and held is not None:
                held = mosh_step(held, a, b, fx["block"] * self.px)
        held = img if held is None else held
        self._held[i] = (frame, held)
        return held

    # ---------------------------------------------------------------- collages
    def _figure(self, fx, frame):
        """The coverage of a collage's figure on `frame` (a FileNotFoundError when its layer is missing), None before the
        clip."""
        if frame < self.frame0:
            return None
        if self._figure_of is None:
            raise FileNotFoundError(f"fx {fx['index']} ({fx['kind']}): no figure layers to read")
        return self._figure_of(FX.figure_key(fx), frame)

    def _collage(self, fx, frame, img, draw):
        """`img` with the collage `draw(k, m)` makes (k the hits so far, m the figure) mixed in by the strength."""
        out = draw(FX.step(fx, frame, self._hit_times(fx), self.fps, self.frame0), self._figure(fx, frame))
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _edge(self, fx):
        e = fx["edge"]
        return None if e is None else (e["color"], e["width"] * self.px)

    def _before(self, fx, frame, j, fill):
        """(picture, coverage) of a collage's copy or panel j, `delay` frames apart: frame `frame` - j x delay, its
        picture only when it is drawn as such (`fill` None); (None, None) before the clip."""
        g = frame - j * fx["delay"]
        m = self._figure(fx, g)
        if m is None or fill is not None:
            return None, m
        pic = self._get(g)
        return (pic, m) if pic is not None else (None, None)

    def _grid(self, fx, frame, img):
        def draw(k, m):
            piece = None if fx["piece"] is None else (FX.piece_cell(fx, k), fx["piece"])
            return grid(img, m, FX.pick(fx["cells"], k), lambda i, j: FX.pick(fx["back"], k + i + j),
                        lambda i, j: FX.pick(fx["fill"], k + i + j), piece, self._edge(fx))
        return self._collage(fx, frame, img, draw)

    def _orbit(self, fx, frame, img):
        def draw(k, m):
            theta0 = math.radians(fx["spin"] * (self._t(frame) - fx["from"]) + fx["turn"] * k)
            return orbit(img, m, FX.pick(fx["back"], k), self._at(fx, frame)[0], fx["radius"] * min(self.size),
                         fx["tilt"], theta0, fx["count"], fx["scale"], fx["size"], lambda j: FX.pick(fx["fill"], k + j),
                         self._edge(fx))
        return self._collage(fx, frame, img, draw)

    def _mirror(self, fx, frame, img):
        def draw(k, m):
            w, h = self.size
            place = None if fx["place"] is None else (fx["place"][0] * w - 0.5, fx["place"][1] * h - 0.5)
            return mirror(img, m, self._at(fx, frame)[0], FX.pick(fx["axes"], k), lambda j: FX.pick(fx["back"], k + j),
                          lambda j: FX.pick(fx["fill"], k + j), place, self._edge(fx))
        return self._collage(fx, frame, img, draw)

    def _panels(self, fx, frame, img):
        def draw(k, m):
            figures = [(img, m)]
            for j in range(1, fx["count"]):
                pic, cov = (img, m) if fx["delay"] == 0 else self._before(fx, frame, j, FX.pick(fx["fill"], k + j))
                figures.append(None if cov is None else (pic, cov))
            return panels(self.size, fx["count"], fx["slant"], figures, lambda j: FX.pick(fx["back"], k + j),
                          lambda j: FX.pick(fx["fill"], k + j), fx["size"], self._edge(fx))
        return self._collage(fx, frame, img, draw)

    def _stack(self, fx, frame, img):
        def draw(k, m):
            copies = []
            for j in range(1, fx["count"] + 1):
                f = FX.pick(fx["bands"], k + j - 1)
                pic, cov = (img, m) if fx["delay"] == 0 else self._before(fx, frame, j, f)
                copies.append((pic, cov, f))
            unit = min(self.size)
            return stack(img, m, FX.pick(fx["back"], k), copies, (fx["step"][0] * unit, fx["step"][1] * unit),
                         fx["scale"], self._at(fx, frame)[0], FX.pick(fx["fill"], k), self._edge(fx))
        return self._collage(fx, frame, img, draw)
