"""Picture effects on the finished pictures (docs/design.md: Frame effects): the picture work of the `[[fx]]` kinds
pixelsort, film, camcorder, doodle and booth that mkmmd.core.picturefx checks, times and lays out, and the region any
effect may be limited to (`rect`). PictureFx is the part of mkmmd.framefx.FrameFx that runs them. Images are float RGB in
display space (0..1), (h, w, 3); `px` is a frame's pixels per pixel at 1080 on its short side."""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import framefx as FX
from .core import picturefx as PC
from .core import purikura as PK

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
FONT_EXT = (".ttf", ".otf", ".ttc")
MONO_TAGS = ("mono", "monospace")        # a registry font tagged so is a monospace font: a camcorder's default
REC_RED = (0.95, 0.12, 0.1)              # a camcorder's REC dot
STAMP_MARGIN = 0.065                     # a camcorder's stamp from the frame's edges, in short sides
SCANLINE = 4.0                           # px at 1080: the period of a camcorder's scanlines
CHROMA_SHIFT = 0.6                       # a camcorder's colour runs this share of its `chroma` to the right
TEAR = 22.0                              # px at 1080: how far a tracking band tears a line sideways at full amount
SWITCH_BAND = 0.022                      # a camcorder's head-switching tear: its share of the frame's height at the foot
SWITCH_TEAR = 30.0                       # px at 1080: how far it tears the last line sideways at `switching` 1
HALATION = (1.0, 0.42, 0.28)             # the red-orange of a print's halation round its lights


# ================================================================================================= the region
@lru_cache(maxsize=16)
def rect_ramps(size, rect, feather):
    """(rows (h,), columns (w,)) float32 0..1, the ramps rect_mask multiplies: each row's and each column's share inside
    `rect` ((x0, y0, x1, y1) frame fractions) of a frame of `size` (w, h)."""
    w, h = size

    def ramp(n, a, b):
        x = np.arange(n, dtype=np.float32)
        if feather <= 0:
            return np.clip(np.minimum(x + 1.0, b * n) - np.maximum(x, a * n), 0.0, 1.0)
        lo = x + 0.5 - a * n if a > 0 else np.full(n, np.inf, np.float32)
        hi = b * n - x - 0.5 if b < 1 else np.full(n, np.inf, np.float32)
        u = np.clip(np.minimum(lo, hi) / feather + 0.5, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)
    return ramp(h, rect[1], rect[3]), ramp(w, rect[0], rect[2])


@lru_cache(maxsize=16)
def rect_mask(size, rect, feather):
    """(h, w) float32 0..1: where an effect limited to `rect` ((x0, y0, x1, y1) frame fractions) applies in a frame of
    `size` (w, h): each pixel's share inside it (a hard edge, antialiased), or with `feather` px a smooth ramp that wide
    across the edge (not along the frame's own edges)."""
    rows, cols = rect_ramps(size, rect, feather)
    return (rows[:, None] * cols[None, :]).astype(np.float32)


def rect_box(size, rect, feather):
    """(x0, y0, x1, y1) whole pixels holding all a `rect` touches, its feather too; the whole frame without one."""
    w, h = size
    if rect is None:
        return 0, 0, w, h
    f = feather / 2.0 + 1.0
    return (max(int(math.floor(rect[0] * w - f)), 0), max(int(math.floor(rect[1] * h - f)), 0),
            min(int(math.ceil(rect[2] * w + f)), w), min(int(math.ceil(rect[3] * h + f)), h))


# ================================================================================================= pixel sort
def sort_key(img, by):
    """(h, w) float32 0..1, what a pixel sort sorts by: luma, hue (round the wheel from red) or saturation (HSV)."""
    if by == "luma":
        return np.clip(img @ LUMA, 0.0, 1.0)
    hsv = cv2.cvtColor(np.ascontiguousarray(np.clip(img, 0.0, 1.0), np.float32), cv2.COLOR_RGB2HSV)
    return hsv[..., 0] / np.float32(360.0) if by == "hue" else hsv[..., 1]


def sort_runs(img, key, eligible, axis="x", reverse=False):
    """`img` with every run of `eligible` pixels (unbroken along a row, axis "x", or a column, "y") sorted by `key`,
    rising to the right (down), falling with `reverse`. All runs at once: one sort of int64s that pack the run, the key
    (16 bits) and the pixel."""
    if axis == "y":
        out = sort_runs(img.transpose(1, 0, 2), key.T, eligible.T, "x", reverse)
        return np.ascontiguousarray(out.transpose(1, 0, 2))
    h, w = key.shape
    n = h * w
    e = np.ascontiguousarray(eligible, bool)
    idx = np.flatnonzero(e)
    if idx.size < 2:
        return img
    starts = e.copy()
    starts[:, 1:] &= ~e[:, :-1]
    run = np.cumsum(starts.ravel(), dtype=np.int64)[idx]
    ib = max(int(n).bit_length(), 1)
    kb = min(16, 62 - 2 * ib)
    top = (1 << kb) - 1
    q = np.rint(np.ascontiguousarray(key, np.float32).ravel()[idx] * top).astype(np.int64)
    if reverse:
        q = top - q
    packed = (run << (kb + ib)) | (q << ib) | idx.astype(np.int64)
    packed.sort()
    flat = np.ascontiguousarray(img, np.float32).reshape(n, -1)
    out = flat.copy()
    out[idx] = flat[packed & ((1 << ib) - 1)]
    return out.reshape(img.shape)


# ================================================================================================= film
def film_curve(x, contrast):
    """A print's S curve: a soft toe and shoulder about mid grey, steeper with `contrast` (1 the usual, 0 none)."""
    c = 4.0 * contrast
    if c < 1e-3:
        return x
    return (0.5 + 0.5 * np.tanh(np.float32(c / 2.0) * (x - np.float32(0.5))) / np.float32(math.tanh(c / 4.0))).astype(
        np.float32)


@lru_cache(maxsize=8)
def vignette(w, h, amount):
    """(h, w) float32: 1 in the middle, falling to 1 - `amount` in the corners, an ellipse of the frame's shape."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx + 0.5) / (w / 2.0) - 1.0, (yy + 0.5) / (h / 2.0) - 1.0) / math.sqrt(2.0)
    u = np.clip((r - 0.3) / 0.7, 0.0, 1.0)
    return (1.0 - amount * u * u * (3.0 - 2.0 * u)).astype(np.float32)


def grain_field(rng, shape, size):
    """(h, w) float32 noise of unit spread whose clumps are `size` px across: film grain."""
    h, w = shape
    gh, gw = max(int(round(h / max(size, 1.0))), 2), max(int(round(w / max(size, 1.0))), 2)
    g = rng.standard_normal((gh, gw), dtype=np.float32)
    if (gh, gw) != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return g / np.float32(max(float(g.std()), 1e-6))


def _q(p):
    """Points (n, 2) px -> OpenCV's fixed point (4 bits)."""
    return np.rint(np.asarray(p, np.float64) * 16.0).astype(np.int32)


def damage_marks(fx, k, size, px):
    """(light, dark): (h, w) uint8 masks of what film frame k carries as drawn (before `damage` blurs them), None for
    one with nothing on it: its dust (white specks and dark ones, some of them fibres), its scratches (white) and a hair
    in the gate (dark)."""
    w, h = size
    light, dark = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    drawn = [False, False]
    for s in PC.film_dust(fx, k):
        tgt, a = (light, 0) if s["light"] else (dark, 1)
        c = (s["x"] * w, s["y"] * h)
        v = int(round(255 * s["alpha"]))
        if s["fibre"] is None:
            cv2.circle(tgt, tuple(int(x) for x in _q(c)), int(round(max(s["r"] * px, 0.7) * 16)), v, -1, cv2.LINE_AA, 4)
        else:
            length, ang, bend = s["fibre"]
            d = np.array([math.cos(ang), math.sin(ang)]) * length * px / 2.0
            mid = np.array(c) + np.array([-d[1], d[0]]) * bend * 0.5
            pts = PC._quad(np.array(c) - d, mid, np.array(c) + d, 12)
            cv2.polylines(tgt, [_q(pts)], False, v, max(int(round(0.45 * s["r"] * px)), 1), cv2.LINE_AA, 4)
        drawn[a] = True
    for sc in PC.film_scratches(fx, k):
        ys = np.linspace(sc["y"][0] * h, sc["y"][1] * h, 32)
        xs = sc["x"] * w + 0.8 * px * np.sin(ys / h * 9.0 + sc["wiggle"])
        cv2.polylines(light, [_q(np.stack([xs, ys], 1))], False, int(round(255 * sc["alpha"])),
                      max(int(round(sc["width"] * px)), 1), cv2.LINE_AA, 4)
        drawn[0] = True
    for hr in PC.film_hairs(fx, k):
        anchor = {0: (hr["along"] * w, 0.0), 1: (w, hr["along"] * h), 2: (hr["along"] * w, h),
                  3: (0.0, hr["along"] * h)}[hr["side"]]
        cv2.polylines(dark, [_q(np.asarray(anchor) + hr["points"] * h)], False, 235, max(int(round(1.7 * px)), 1),
                      cv2.LINE_AA, 4)
        drawn[1] = True
    return light if drawn[0] else None, dark if drawn[1] else None


def damage(fx, k, size, px):
    """(light, dark): (h, w) float32 0..1 masks of what film frame k carries (damage_marks), blurred a little, or None
    for none."""
    soft = 0.5 * px

    def done(m):
        if m is None:
            return None
        m = m.astype(np.float32) / 255.0
        return cv2.GaussianBlur(m, (0, 0), soft) if soft >= 0.3 else m
    return tuple(done(m) for m in damage_marks(fx, k, size, px))


def dye_print(x, dye, saturation):
    """`x` as a three-strip dye print shows it: its saturation times `saturation`, and with `dye` (0..2; 1 a
    Technicolor print's) each primary purer than the others let it be, reds and oranges richest (lipstick and skin),
    the shadows a little cool and the lights warm."""
    L = (x @ LUMA)[..., None]
    out = L + (x - L) * np.float32(saturation)
    if dye > 0:
        mean = out.mean(axis=2, keepdims=True)
        out = out + np.float32(0.45 * dye) * (out - mean)
        red = np.clip(2.0 * (out[..., :1] - np.maximum(out[..., 1:2], out[..., 2:3])), 0.0, 1.0)
        L = (out @ LUMA)[..., None]
        out = out + np.float32(0.3 * dye) * red * (out - L)
        out = out + np.float32(dye) * (np.float32([-0.02, 0.0, 0.035]) * (1.0 - L) ** 2 +
                                       np.float32([0.03, 0.01, -0.03]) * L ** 2)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def halation(x, amount, px, colour=HALATION):
    """`x` with a glow round its lights as a print's halation (light scattered back through the film): their light
    blurred near and far in `colour`, screened over it by `amount`."""
    hi = np.clip((x @ LUMA - 0.72) / 0.28, 0.0, 1.0)
    g = cv2.GaussianBlur(hi, (0, 0), max(5.0 * px, 0.5)) + 0.5 * cv2.GaussianBlur(hi, (0, 0), max(16.0 * px, 0.5))
    add = np.clip(np.float32(amount) * g, 0.0, 1.0)[..., None] * np.asarray(colour, np.float32)
    return (1.0 - (1.0 - x) * (1.0 - add)).astype(np.float32)


def film(img, fx, k, px):
    """The film print of `img` (a frame, or a rect's part of one) as film frame k shows it: through the print's curve
    (in black and white with `mono`, else its colours through a dye print: `saturation`, `dye`), weaving in the gate,
    its dust, scratches and hairs, its exposure flickering, a halation round its lights, its grain (strongest in the mid
    tones) and a vignette."""
    h, w = img.shape[:2]
    gain, dx, dy = PC.film_jolt(fx, k)
    x = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    if fx["mono"]:
        out = np.repeat(film_curve(np.ascontiguousarray(x @ LUMA), fx["contrast"])[..., None], 3, axis=2)
    else:
        if fx["dye"] > 0 or fx["saturation"] != 1.0:
            x = dye_print(x, fx["dye"], fx["saturation"])
        out = film_curve(x, fx["contrast"])
    if abs(dx) + abs(dy) > 1e-3:
        out = cv2.warpAffine(out, np.float32([[1, 0, dx * px], [0, 1, dy * px]]), (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    light, dark = damage(fx, k, (w, h), px)
    if dark is not None:
        out = out * (1.0 - 0.92 * dark)[..., None]
    if light is not None:
        out = out + (1.0 - out) * light[..., None]
    out = out * np.float32(gain)
    if fx["halation"] > 0:
        out = halation(np.clip(out, 0.0, 1.0), fx["halation"], px, (0.8, 0.8, 0.8) if fx["mono"] else HALATION)
    g = fx["grain"]
    if g["amount"] > 0:
        n = grain_field(np.random.default_rng([fx["seed"], fx["index"], 6, k]), (h, w), g["size"] * px)
        lum = np.clip(out @ LUMA, 0.0, 1.0)
        out = out + (np.float32(g["amount"]) * n * (0.35 + 2.6 * lum * (1.0 - lum)))[..., None]
    if fx["vignette"] > 0:
        out = out * vignette(w, h, fx["vignette"])[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ================================================================================================= camcorder
@lru_cache(maxsize=8)
def _scanlines(h, amount, period):
    y = np.arange(h, dtype=np.float32)
    return (1.0 - amount * (0.5 - 0.5 * np.cos(2.0 * np.pi * y / period))).astype(np.float32)


@lru_cache(maxsize=8)
def _grid(w, h):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def tape_draws(fx, frame, t, px, w, h):
    """The random parts of a tape's picture `w` x `h` on clip frame `frame` (second `t`), drawn in order from the
    frame's generator: {ny, nc: the luma and colour noise before they are stretched over the picture (None without
    `noise`), off: how far each line is torn sideways (px), band: the tracking band's strength by line (None without
    one), snow: which streaks across the band are snow, before they are stretched (None: no band)}."""
    rng = np.random.default_rng([fx["seed"], fx["index"], 21, frame])
    out = {"ny": None, "nc": None, "band": None, "snow": None}
    if fx["noise"] > 0:
        streak = max(int(round(3.0 * px)), 1)
        out["ny"] = rng.standard_normal((h, max(w // streak, 2)), dtype=np.float32)
        out["nc"] = rng.standard_normal((max(h // 2, 2), max(w // (4 * streak), 2), 2), dtype=np.float32)
    off = np.zeros(h, np.float32)
    if fx["jitter"] > 0:
        off += np.float32(fx["jitter"] * px) * np.convolve(rng.standard_normal(h), np.ones(3) / 3.0, "same").astype(
            np.float32)
    band = None
    if fx["tracking"]["amount"] > 0:
        yc, hb = PC.tracking_centre(fx, t) * h, PC.TRACK_BAND * h
        band = (fx["tracking"]["amount"] * np.exp(-((np.arange(h) - yc) / hb) ** 2)).astype(np.float32)
        tear = np.convolve(rng.standard_normal(h), np.ones(2) / 2.0, "same") + 0.6
        off += band * np.float32(TEAR * px) * tear.astype(np.float32)
    if fx["switching"] > 0:                                     # the heads switch a few lines before the field ends
        foot = max(int(round(SWITCH_BAND * h)), 2)
        u = np.linspace(0.0, 1.0, foot).astype(np.float32)
        off[h - foot:] += np.float32(fx["switching"] * SWITCH_TEAR * px) * u ** 1.5 * \
            (1.0 + 0.35 * rng.standard_normal(foot)).astype(np.float32)
    out["off"] = off
    if band is not None and band.max() > 0.01:
        streak = max(int(round(9.0 * px)), 1)
        out["band"] = band
        out["snow"] = (rng.random((h, max(w // streak, 2))) < 0.45 * band[:, None]).astype(np.float32)
    return out


def tape(img, fx, frame, t, px):
    """`img` (a frame, or a rect's part of one) as a home video tape plays it on clip frame `frame` (second `t`): soft
    luma, colour bled sideways and to the right, `saturation`, noise in streaks, lines jittered sideways, a tracking
    band rolling up the picture (lines torn sideways, snow, washed out), the head-switching tear along its foot
    (`switching`), scanlines."""
    h, w = img.shape[:2]
    d = tape_draws(fx, frame, t, px, w, h)
    ycc = cv2.cvtColor(np.ascontiguousarray(np.clip(img, 0.0, 1.0), np.float32), cv2.COLOR_RGB2YCrCb)
    y, c = ycc[..., 0], ycc[..., 1:] - np.float32(0.5)
    if fx["soft"] > 0:
        s = fx["soft"] * px
        y = cv2.GaussianBlur(y, (0, 0), sigmaX=s, sigmaY=max(0.35 * s, 0.3))
    if fx["chroma"] > 0:
        r = fx["chroma"] * px
        c = cv2.blur(c, (max(int(round(2.0 * r)), 1), 1))
        c = cv2.GaussianBlur(c, (0, 0), sigmaX=max(0.5 * r, 0.3), sigmaY=max(0.6 * px, 0.3))
        c = cv2.warpAffine(c, np.float32([[1, 0, CHROMA_SHIFT * r], [0, 1, 0]]), (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    c = c * np.float32(fx["saturation"])
    if d["ny"] is not None:
        y = y + np.float32(fx["noise"]) * cv2.resize(d["ny"], (w, h), interpolation=cv2.INTER_LINEAR)
        c = c + np.float32(0.5 * fx["noise"]) * cv2.resize(d["nc"], (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.cvtColor(np.dstack([y, c + np.float32(0.5)]).astype(np.float32), cv2.COLOR_YCrCb2RGB)
    off = d["off"]
    if np.abs(off).max() > 0.01:
        xs, ys = _grid(w, h)
        out = cv2.remap(out, xs + off[:, None], ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if d["snow"] is not None:
        band = d["band"]
        snow = cv2.resize(d["snow"], (w, h), interpolation=cv2.INTER_NEAREST) * np.minimum(band * 1.6, 1.0)[:, None]
        grey = out.mean(axis=2, keepdims=True)
        wash = (0.45 * band)[:, None, None]
        out = out * (1.0 - wash) + (grey * 0.8 + 0.2) * wash
        out = out + (1.0 - out) * (0.85 * snow)[..., None]
    if fx["scanlines"] > 0:
        out = out * _scanlines(h, float(fx["scanlines"]), float(SCANLINE * px))[:, None, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=32)
def _em(path, cap):
    """The font size (em, px) of the font at `path` whose capitals stand `cap` px tall."""
    f = ImageFont.truetype(path, 200)
    l, t, r, b = f.getbbox("H")
    return max(int(round(200.0 * cap / max(b - t, 1))), 4)


@lru_cache(maxsize=128)
def _line(text, path, em, edge):
    """(fill, rim, baseline row, ink start x, advance) of one line of `text`: masks (float32 0..1) of its letters and
    of its letters grown by `edge` px."""
    font = ImageFont.truetype(path, em)
    asc, desc = font.getmetrics()
    adv = float(font.getlength(text))
    W, H = int(math.ceil(adv)) + 2 * edge + 6, asc + desc + 2 * edge + 6
    x0, base = edge + 3, edge + 3 + asc
    fill, rim = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    ImageDraw.Draw(fill).text((x0, base), text, font=font, fill=255, anchor="ls")
    ImageDraw.Draw(rim).text((x0, base), text, font=font, fill=255, anchor="ls", stroke_width=edge, stroke_fill=255)
    return np.asarray(fill, np.float32) / 255.0, np.asarray(rim, np.float32) / 255.0, base, x0, adv


@lru_cache(maxsize=16)
def _glyph(kind, cap, edge):
    """(fill, rim, baseline row, ink start x, advance) of a PLAY triangle ("play") or a REC dot ("dot"), `cap` px
    tall, as _line gives them."""
    s = 4                                                         # drawn 4 times larger, boxed down: smooth edges
    wid = 0.8 * cap if kind == "play" else 0.78 * cap
    W, H = int(math.ceil(wid)) + 2 * edge + 6, int(math.ceil(cap)) + 2 * edge + 6
    x0, base = edge + 3, edge + 3 + cap
    out = []
    for grow in (0, edge):
        im = Image.new("L", (W * s, H * s), 0)
        d = ImageDraw.Draw(im)
        if kind == "play":
            pts = [(x0 * s, (base - cap) * s), ((x0 + wid) * s, (base - cap / 2.0) * s), (x0 * s, base * s)]
            d.polygon(pts, fill=255, outline=255, width=max(2 * grow * s, 1) if grow else 1)
        else:
            r, cx, cy = wid / 2.0 + grow, x0 + wid / 2.0, base - cap / 2.0
            d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=255)
        out.append(np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0)
    return out[0], out[1], int(round(base)), x0, wid


def _lay(img, mask, x0, y0, colour, alpha=1.0):
    """`img` (in place) with `colour` laid where `mask` is, its top left at pixel (x0, y0), cut at the frame's edge."""
    h, w = img.shape[:2]
    mh, mw = mask.shape
    ax, ay = max(x0, 0), max(y0, 0)
    bx, by = min(x0 + mw, w), min(y0 + mh, h)
    if bx <= ax or by <= ay:
        return
    m = (alpha * mask[ay - y0:by - y0, ax - x0:bx - x0])[..., None]
    img[ay:by, ax:bx] = img[ay:by, ax:bx] * (1.0 - m) + np.asarray(colour, np.float32) * m


def _stamp_item(inks, item, x, base, right, colour, shadow):
    """One piece of a stamp (a _line or a _glyph) with its rim and shadow, its ink starting at x (ending there when
    `right`) on the baseline row `base`, added to `inks` as stamp_inks lists them; its advance."""
    fill, rim, b, ix, adv = item
    x0 = int(round((x - adv if right else x) - ix))
    y0 = int(round(base - b))
    inks += [(rim, (0.0, 0.0, 0.0), x0 + shadow, y0 + shadow, 0.45), (rim, (0.02, 0.02, 0.03), x0, y0, 0.8),
             (fill, colour, x0, y0, 1.0)]
    return adv


def stamp_inks(size, fx, t, px, font):
    """A camcorder's stamp at clip second `t` over a frame of `size` (w, h), crisp: the time over the date in its
    `corner`, `size` short sides tall at 1080, in `color` with a dark rim and shadow; its indicator (PLAY and a
    triangle, or a blinking red dot and REC) at the top, on the left unless the stamp is there. As the coverages laid
    in order ([(mask, colour, x0, y0, alpha)], _lay's arguments)."""
    w, h = size
    st = PC.stamp(fx, t)
    cap = fx["size"] * 1080.0 * px
    em, edge = _em(font, round(cap, 1)), max(int(round(0.08 * cap)), 1)
    shadow = max(int(round(0.07 * cap)), 1)
    margin = STAMP_MARGIN * 1080.0 * px
    right, bottom = fx["corner"].endswith("right"), fx["corner"].startswith("bottom")
    lead = 1.45 * cap
    n = len(st["lines"])
    inks = []
    for i, text in enumerate(st["lines"]):
        base = h - margin - (n - 1 - i) * lead if bottom else margin + cap + i * lead
        _stamp_item(inks, _line(text, font, em, edge), w - margin if right else margin, base, right, fx["color"], shadow)
    if st["indicator"] is None:
        return inks
    word = _line(st["indicator"], font, em, edge)
    mark = _glyph("play" if st["indicator"] == "PLAY" else "dot", int(round(cap)), edge)
    gap = 0.4 * cap
    items = [(word, fx["color"]), (mark, fx["color"])] if st["indicator"] == "PLAY" else \
        [(mark, REC_RED if st["dot"] else None), (word, fx["color"])]
    at_right = fx["corner"] == "top-left"
    total = sum(it[0][4] for it in items) + gap
    x = (w - margin - total) if at_right else margin
    for it, col in items:
        if col is not None:
            _stamp_item(inks, it, x, margin + cap, False, col, shadow)
        x += it[4] + gap
    return inks


def draw_stamp(img, fx, t, px, font):
    """`img` (in place) with a camcorder's stamp at clip second `t` (stamp_inks)."""
    for mask, colour, x0, y0, alpha in stamp_inks((img.shape[1], img.shape[0]), fx, t, px, font):
        _lay(img, mask, x0, y0, colour, alpha)
    return img


# ================================================================================================= doodles
def doodle_strokes(shape, centre, size, angle, age, rng, wobble):
    """([(n, 2) px], scale): the strokes of a doodle `shape` about pixel `centre`, `size` px across, turned `angle`
    degrees counter-clockwise, `age` seconds after it popped in (its spring: the scale), drawn again by `rng`'s hand
    (one boil step) with a wobble of `wobble` px; ([], 0) before it pops in."""
    s = PC.pop(age)
    if s <= 0 or size <= 0:
        return [], 0.0
    unit = PC.turned(PC.boil(PC.shape(shape), rng, wobble / size), angle)
    return [np.asarray(centre, float) + p * (size * s) for p in unit], s


def draw_strokes(img, strokes, colour, width, outline):
    """`img` (in place) with `strokes` ([(n, 2) px]) drawn as a pen line `width` px wide in `colour` over a white rim
    `outline` px wide all round it, as a photo-booth pen draws."""
    h, w = img.shape[:2]
    strokes = [p for p in strokes if len(p) >= 2]
    if not strokes:
        return img
    pts = np.concatenate(strokes)
    pad = width / 2.0 + outline + 3.0
    x0, y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, w), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return img
    q = [_q(p - (x0, y0)) for p in strokes]
    rim, ink = np.zeros((y1 - y0, x1 - x0), np.uint8), np.zeros((y1 - y0, x1 - x0), np.uint8)
    if outline > 0:
        cv2.polylines(rim, q, False, 255, max(int(round(width + 2.0 * outline)), 1), cv2.LINE_AA, 4)
    cv2.polylines(ink, q, False, 255, max(int(round(width)), 1), cv2.LINE_AA, 4)
    reg = img[y0:y1, x0:x1]
    a, b = (rim.astype(np.float32) / 255.0)[..., None], (ink.astype(np.float32) / 255.0)[..., None]
    reg = reg * (1.0 - a) + a
    img[y0:y1, x0:x1] = reg * (1.0 - b) + np.asarray(colour, np.float32) * b
    return img


# ================================================================================================= the booth
def cover(pic, size, focus):
    """`pic` cut to the shape of `size` (w, h) about `focus` ([x, y] fractions of it, kept as near the middle as the cut
    allows) and resized to it."""
    H, W = pic.shape[:2]
    tw, th = size
    a = tw / th
    cw, ch = (H * a, float(H)) if W / H > a else (float(W), W / a)
    x0 = min(max(focus[0] * W - cw / 2.0, 0.0), W - cw)
    y0 = min(max(focus[1] * H - ch / 2.0, 0.0), H - ch)
    crop = pic[int(round(y0)):int(round(y0 + ch)), int(round(x0)):int(round(x0 + cw))]
    return cv2.resize(np.ascontiguousarray(crop, np.float32), (tw, th), interpolation=cv2.INTER_AREA)


def _card(strip, pic, cell, b, s, lift, paper, px):
    """`strip` (in place) with photo `pic` on its paper card (`b` px of paper round it) dropping into `cell` ((x, y) of
    the photo's top left): `s` times its size about the cell's middle, its shadow `lift` (0 .. 1) below it."""
    ph, pw = pic.shape[:2]
    rim = max(int(round(b)), 0)
    card = np.empty((ph + 2 * rim, pw + 2 * rim, 3), np.float32)
    card[:] = paper
    card[rim:rim + ph, rim:rim + pw] = pic
    if abs(s - 1.0) > 1e-3:
        card = cv2.resize(card, (max(int(round(card.shape[1] * s)), 1), max(int(round(card.shape[0] * s)), 1)),
                          interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
    ch, cw = card.shape[:2]
    x0 = int(round(cell[0] + pw / 2.0 - cw / 2.0))
    y0 = int(round(cell[1] + ph / 2.0 - ch / 2.0))
    if lift > 0.01:
        off, blur = int(round(lift * 0.08 * ph)), max(6.0 * px, 0.5)
        pad = int(math.ceil(3 * blur))
        sh = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
        sh[pad:pad + ch, pad:pad + cw] = 1.0
        sh = cv2.GaussianBlur(sh, (0, 0), blur)
        _lay(strip, sh, x0 - pad + off, y0 - pad + off, (0.0, 0.0, 0.0), 0.35 * min(lift * 2.0, 1.0))
    H, W = strip.shape[:2]
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + cw, W), min(y0 + ch, H)
    if bx > ax and by > ay:
        strip[ay:by, ax:bx] = card[ay - y0:by - y0, ax - x0:bx - x0]


def booth(fx, size, px, t, photos, font=None):
    """A photo-booth strip at clip second `t` filling a frame (or a rect's part of one) of `size` (w, h): the strip of
    PC.booth_strip on `back`, turned `tilt` degrees, its shadow under it; `photos` [(capture second, picture at the
    photo's size)] the captures so far, each dropping into its cell, doodles popping round it once it has landed; the
    `caption` in the strip's foot (`font` its file); the flash of a capture over all of it. The purikura look
    (mkmmd.purikura) dots the paper, frames each photo through its beauty filter, stamps and writes on it."""
    w, h = size
    geo = PC.booth_strip(fx, size, px)
    pw, ph = geo["photo"]
    PW, PH = max(int(round(pw)), 1), max(int(round(ph)), 1)
    SW, SH = int(math.ceil(geo["strip"][0])), int(math.ceil(geo["strip"][1]))
    paper = np.asarray(fx["paper"], np.float32)
    strip = np.empty((SH, SW, 3), np.float32)
    strip[:] = paper
    look = None
    if fx.get("purikura") is not None:
        from . import purikura as look                           # it draws with this module's pens
        look.paper(strip, fx, px)
    for cx, cy in geo["cells"]:
        x0, y0 = int(round(cx)), int(round(cy))
        if look is None:
            strip[y0:y0 + PH, x0:x0 + PW] = paper * np.float32(0.86)
        else:
            look.slot(strip, (x0, y0), (PW, PH), paper)
    pen = PC.BOOTH_PEN
    for j, (tc, pic) in enumerate(photos):
        if j >= len(geo["cells"]) or t < tc:
            continue
        cell = geo["cells"][j]
        if look is not None:
            look.photo(strip, fx, j, pic, cell, t, tc, px)
            continue
        s, lift = PC.drop(t - tc, fx["drop"])
        _card(strip, pic, cell, geo["border"], s, lift, paper, px)
        b = max(int(math.floor((t - fx["from"]) * pen["fps"] + 1e-9)), 0)
        for i, d in enumerate(PC.booth_doodles(fx, j, (PW, PH), tc)):
            rng = np.random.default_rng([fx["seed"], fx["index"], 14, j, i, b])
            strokes, k = doodle_strokes(d["shape"], (cell[0] + d["at"][0], cell[1] + d["at"][1]), d["size"],
                                        d["angle"], t - d["t"], rng, pen["wobble"] * px)
            draw_strokes(strip, strokes, d["colour"], max(pen["width"] * px * min(k, 1.0), 1.0),
                         pen["outline"] * px * min(k, 1.0))
    if fx["caption"] and font is not None:
        band_top = geo["rows"] * (ph + geo["border"])            # under the last photo: its border and the foot
        band = SH - band_top
        ch = 0.42 * band
        em = _em(font, round(ch, 1))
        fill, _, base, ix, adv = _line(fx["caption"], font, em, 1)
        if adv > 0.9 * SW:                                        # too long for the strip: smaller
            k = 0.9 * SW / adv
            em, ch = max(int(em * k), 4), ch * k
            fill, _, base, ix, adv = _line(fx["caption"], font, em, 1)
        _lay(strip, fill, int(round(SW / 2.0 - adv / 2.0 - ix)), int(round(band_top + band / 2.0 + ch / 2.0 - base)),
             fx["color"])
    out = np.empty((h, w, 3), np.float32)
    out[:] = np.asarray(fx["back"], np.float32)
    M = cv2.getRotationMatrix2D((SW / 2.0, SH / 2.0), fx["tilt"], 1.0)
    M[:, 2] += (w / 2.0 - SW / 2.0, h / 2.0 - SH / 2.0)
    cover_ = cv2.warpAffine(np.ones((SH, SW), np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    colour = cv2.warpAffine(strip, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    shade = cv2.GaussianBlur(cover_, (0, 0), 14.0 * px)
    shade = cv2.warpAffine(shade, np.float32([[1, 0, 5.0 * px], [0, 1, 9.0 * px]]), (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = out * (1.0 - 0.3 * shade)[..., None]
    out = out * (1.0 - cover_)[..., None] + colour
    fl = max([PC.flash(t - tc, fx["flash"]) for tc in PK.flashing(fx, [tc for tc, _ in photos])] + [0.0])
    if fl > 0:
        out = out + (1.0 - out) * np.float32(fl)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ================================================================================================= fonts
def font_source(root=None, registry=None):
    """callable(ref) -> the path of a font file, for a camcorder's stamp and a booth's caption: `ref` a font file
    (project-relative to `root`, absolute or ~/...) or a font slug of the asset registry (`registry`, an
    mkmmd.assets.Registry; the configured one by default, read when first needed); None picks the registry's first font
    tagged `mono`. FX.FxError says what is missing."""
    held = [registry]

    def reg():
        if held[0] is None:
            from .assets import Registry
            held[0] = Registry()
        return held[0]

    def path(r, e, what):
        p = Path(e["path"]).expanduser()
        p = p if p.is_absolute() else r.root / p
        if not p.is_file():
            raise FX.FxError(f"{what}: {p} does not exist")
        return str(p)

    def get(ref):
        if ref is None:
            r = reg()
            for e in r.entries:
                if e.get("kind") == "font" and set(MONO_TAGS) & set(e.get("tags") or []):
                    return path(r, e, f"font {e['slug']!r} (the registry's first font tagged mono)")
            raise FX.FxError(f"no font tagged `mono` in the asset registry ({r.path}): register a monospace font "
                             f"(mk assets add FILE --kind font --tags mono) or give the effect a `font`")
        s = str(ref)
        if any(ch in s for ch in "/\\~") or Path(s).suffix.lower() in FONT_EXT:
            p = Path(s).expanduser()
            p = p if p.is_absolute() or root is None else Path(root) / p
            if not p.is_file():
                raise FX.FxError(f"font {ref!r}: {p} is not a file")
            return str(p)
        r = reg()
        e = next((x for x in r.entries if x.get("slug") == s), None)
        if e is None or e.get("kind") != "font":
            fonts = sorted(x["slug"] for x in r.entries if x.get("kind") == "font")
            raise FX.FxError(f"font {ref!r}: not a font file and not a font in the asset registry (have {fonts})")
        return path(r, e, f"font {ref!r}")
    return get


# ================================================================================================= over a cut
class PictureFx:
    """The picture effects' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs,
    entries, _t, _get, _apply, _hit_times, _figure and _point_of). `_picture_init` is called once its effects are known."""

    def _picture_init(self, font_of):
        """The fonts of the camcorders and the booths' captions, found now: a missing one is an FX.FxError before any
        frame is drawn."""
        self._font_of = font_of or font_source()
        self._seen, self._cells, self._films, self._fonts = {}, {}, {}, {}   # _seen: pictures as they reached an fx
        for fx in self.fxs:
            if fx["kind"] == "camcorder" or (fx["kind"] == "booth" and fx["caption"]) or (fx["kind"] == "clip" and
                                                                                          fx["shape"] == "text"):
                self._fonts[fx["index"]] = self._font_of(fx["font"])

    def _reached(self, fx, g, frame, img):
        """The picture of clip frame `g` as it reached `fx` (a film's held frame, a booth's capture, the first frame a
        mosh with a flow holds): `img` on `g` itself, kept for the frames after (mk post goes through them in order);
        else what was kept then; else the cut's frame `g` with the effects before `fx` in the table applied again, as `g`
        draws them (frames asked out of order, as mk look asks them). None when the cut has no frame `g`."""
        memo = self._seen.setdefault(fx["index"], {})
        if g == frame:
            memo[g] = img
            return img
        if g in memo:
            return memo[g]
        pic = self._get(g)
        if pic is not None:
            for e in self.entries(g):
                if e["index"] >= fx["index"]:
                    break
                pic = self._apply(e, g, pic)
            memo[g] = pic
        return pic

    def _region(self, fx, img, draw):
        """`draw(picture, box)` run on the part of `img` an effect's `rect` covers (its box (x0, y0, x1, y1) px), as a
        frame of its own, and laid back into a copy of `img` (FrameFx._apply then keeps it inside the rect)."""
        box = rect_box(self.size, fx["rect"], fx["feather"] * self.px)
        if box == (0, 0, self.size[0], self.size[1]):
            return draw(img, box)
        x0, y0, x1, y1 = box
        out = np.array(img, np.float32, copy=True)
        out[y0:y1, x0:x1] = draw(np.ascontiguousarray(img[y0:y1, x0:x1]), box)
        return out

    def _mix(self, fx, frame, img, out):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _masked(self, fx, frame):
        """(h, w) float32 0..1, how much of each pixel an effect with a `mask` (a pixel sort, a mosh) works on: its
        figure's coverage on `frame` ("figure") or what that leaves ("ground"); None without a mask. A pixel sort
        snaps it at a half (its runs stop at the figure's outline), a mosh mixes by it (the outline antialiased)."""
        if fx["mask"] is None:
            return None
        m = self._figure(fx, frame)
        return m if fx["mask"] == "figure" else 1.0 - m

    def pixelsort_at(self, fx, frame):
        """A pixel sort's strength on `frame`: steady at `strength`, or on hits full on the frame of each and dying as
        exp(-age / decay), the strongest counting."""
        t = self._t(frame)
        times = [(FX.frame_of(h, self.fps, self.frame0) - self.frame0) / self.fps for h in self._hit_times(fx)]
        on = 1.0 if fx.get("lane") is None and not times else float(FX.punch(t, times, 0.0, fx["decay"]))
        return float(np.clip(FX.strength(fx, t) * on, 0.0, 1.0))

    def _pixelsort(self, fx, frame, img):
        s = self.pixelsort_at(fx, frame)
        if s <= 1e-3:
            return img
        region = self._masked(fx, frame)
        inside = None if fx["rect"] is None else rect_mask(self.size, tuple(fx["rect"]), fx["feather"] * self.px) > 0.5

        def draw(pic, box):
            x0, y0, x1, y1 = box
            key = sort_key(pic, fx["by"])
            ok = (key >= fx["lo"]) & (key <= fx["hi"])
            if inside is not None:
                ok &= inside[y0:y1, x0:x1]
            if region is not None:
                ok &= region[y0:y1, x0:x1] > 0.5
            return sort_runs(pic, key, ok, fx["axis"], fx["reverse"])
        out = self._region(fx, img, draw)
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _film(self, fx, frame, img):
        k, g = PC.film_step(fx, frame)
        last = self._films.get(fx["index"])
        if last is None or last[0] != k:
            held = self._reached(fx, g, frame, img)
            held = img if held is None else held
            self._seen[fx["index"]] = {g: held}
            last = self._films[fx["index"]] = (k, self._region(fx, held, lambda pic, box: film(pic, fx, k, self.px)))
        return self._mix(fx, frame, img, last[1])

    def _camcorder(self, fx, frame, img):
        t, font = self._t(frame), self._fonts[fx["index"]]
        out = self._region(fx, img, lambda pic, box: draw_stamp(tape(pic, fx, frame, t, self.px), fx, t, self.px, font))
        return self._mix(fx, frame, img, out)

    def _doodle(self, fx, frame, img):
        t, (w, h) = self._t(frame), self.size
        out = np.array(img, np.float32, copy=True)
        b = PC.boil_step(fx, t)
        for j, m in enumerate(fx["marks"]):
            if t <= m["t"]:
                continue
            if m["at"]["mode"] == "frame":
                c, per_m = (m["at"]["at"][0] * w - 0.5, m["at"]["at"][1] * h - 0.5), None
            else:
                c, per_m = self._point_of(f"{FX.point_key(fx)}m{j}", frame)
            size = m["size"] * h if "size" in m else m["size_m"] * per_m * h
            rng = np.random.default_rng([fx["seed"], fx["index"], j, b])
            strokes, k = doodle_strokes(m["shape"], c, size, m["angle"], t - m["t"], rng, fx["wobble"] * self.px)
            draw_strokes(out, strokes, m["color"], max(fx["width"] * self.px * min(k, 1.0), 1.0),
                         fx["outline"] * self.px * min(k, 1.0))
        return self._mix(fx, frame, img, out)

    def _booth(self, fx, frame, img):
        t = self._t(frame)
        caps = [((g - self.frame0) / self.fps, g) for _, g in PC.captures(fx, self._hit_times(fx), self.fps, self.frame0)
                if g <= frame]                                   # captured on its frame: the flash starts there

        def draw(pic, box):
            x0, y0, x1, y1 = box
            geo = PC.booth_strip(fx, (x1 - x0, y1 - y0), self.px)
            size = (max(int(round(geo["photo"][0])), 1), max(int(round(geo["photo"][1])), 1))
            photos = []
            for tc, g in caps:
                key = (fx["index"], g, box, size)
                if key not in self._cells:
                    held = self._reached(fx, g, frame, img)
                    held = img if held is None else held
                    self._cells[key] = cover(held[y0:y1, x0:x1], size, fx["focus"])
                    self._seen.get(fx["index"], {}).pop(g, None)
                photos.append((tc, self._cells[key]))
            return booth(fx, (x1 - x0, y1 - y0), self.px, t, photos, self._fonts.get(fx["index"]))
        return self._mix(fx, frame, img, self._region(fx, img, draw))
