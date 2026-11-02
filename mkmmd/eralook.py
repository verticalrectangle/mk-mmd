"""Era looks on the finished pictures (docs/design.md: Frame effects, Era looks): the picture work of the `[[fx]]` kinds
digicam, chrome, gradmap, invert, testcard and cards that mkmmd.core.eralook checks, times and lays out. EraFx is the
part of mkmmd.framefx.FrameFx that runs them. Images are float RGB in display space (0..1), (h, w, 3); a coverage is
(h, w) 0..1 over the same picture; `px` is a frame's pixels per pixel at 1080 on its short side. Points in px are
OpenCV's: pixel (i, j) is centred on (i, j)."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import colorbars as CB
from .core import eralook as EC
from .core import framefx as FX

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
FRINGE = (0.55, 0.16, 0.86)              # a digicam's purple fringe
CA = 0.0022                              # a digicam's red and blue planes this share apart at the corners at fringe 1
STAMP_GLOW = 0.22                        # a date stamp's glow, in its digits' heights
STAMP_SLANT = 8.0                        # degrees a date stamp's digits lean
STAMP_MARGIN = 0.055                     # a date stamp from the frame's edges, in short sides
DOME = 60.0                              # px at 1080: a chrome figure's puffy dome is full this far in from its outline
RELIEF = 28.0                            # px at 1080: how steeply a chrome's relief tilts its surface
HOLO = 0.9                               # a chrome's colours come round this many times across its picture's long side
ENV = np.array([(0.0, 1.0), (0.22, 0.92), (0.38, 0.7), (0.46, 0.25), (0.5, 0.1), (0.56, 0.35), (0.75, 0.72),
                (1.0, 1.0)], np.float32)   # a chrome's reflection round one band: sky, the dark horizon, ground, sky
STAR = 0.045                             # a sparkle star's radius at full size, in short sides
CARD_TEX = (512, 384)                    # a flying card's faces, px
CARD_ROUND = 0.05                        # a flying card's corners' radius, in its heights
LIGHT = np.array([-0.35, 0.55, -0.75]) / np.linalg.norm([-0.35, 0.55, -0.75])   # toward the cards' light, camera space
GRATINGS = (18.0, 12.0, 8.0, 6.0, 4.5, 3.5)   # px at 1080: a test card's gratings, coarse to fine
FACES = (("front", (0, 1, 2, 3), (0.0, 0.0, -1.0)), ("back", (5, 4, 7, 6), (0.0, 0.0, 1.0)),
         ("edge", (4, 5, 1, 0), (0.0, 1.0, 0.0)), ("edge", (1, 5, 6, 2), (1.0, 0.0, 0.0)),
         ("edge", (3, 2, 6, 7), (0.0, -1.0, 0.0)), ("edge", (4, 0, 3, 7), (-1.0, 0.0, 0.0)))
# a card slab's faces: which of card_corners' points (as the face reads from outside) and the outward normal, card space


def luma(img):
    return np.clip(np.asarray(img, np.float32) @ LUMA, 0.0, 1.0)


def _over(dst, colour, a):
    """`dst` with `colour` laid over it by coverage `a` (h, w)."""
    a = np.asarray(a, np.float32)[..., None]
    return (dst * (1.0 - a) + np.asarray(colour, np.float32) * a).astype(np.float32)


def _screen(dst, add):
    return (1.0 - (1.0 - dst) * (1.0 - np.clip(add, 0.0, 1.0))).astype(np.float32)


def _q(p):
    """Points (n, 2) px -> OpenCV's fixed point (4 bits)."""
    return np.rint(np.asarray(p, np.float64) * 16.0).astype(np.int32)


def _fill(shape, polys):
    """(h, w) float32 0..1: the polygons (px) filled, antialiased."""
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [_q(p)], 255, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255.0


def _roi(points, size, grow):
    """(x0, y0, x1, y1) whole px holding `points` (n, 2) grown by `grow` px, cut to a frame of `size`; None outside it."""
    w, h = size
    x0, y0 = max(int(math.floor(points[:, 0].min() - grow)), 0), max(int(math.floor(points[:, 1].min() - grow)), 0)
    x1, y1 = min(int(math.ceil(points[:, 0].max() + grow)) + 1, w), min(int(math.ceil(points[:, 1].max() + grow)) + 1, h)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


# ================================================================================================= digicam
@lru_cache(maxsize=4)
def _barrel_maps(w, h, k):
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    R = math.hypot(cx, cy)
    dx, dy = (xs - cx) / R, (ys - cy) / R
    f = (1.0 + k * (dx * dx + dy * dy)) / (1.0 + k)
    return (cx + dx * f * R).astype(np.float32), (cy + dy * f * R).astype(np.float32)


def barrel(img, k):
    """`img` through a cheap zoom lens's barrel distortion `k` (0.04 slight; below 0 a pincushion), its corners kept
    filled: a pixel at radius r (of the half diagonal) shows the picture at r (1 + k r^2) / (1 + k)."""
    if abs(k) < 1e-4:
        return img
    h, w = img.shape[:2]
    mx, my = _barrel_maps(w, h, round(float(k), 5))
    return cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def fringe(img, amount, px):
    """Purple fringes where a highlight meets darker picture, as a cheap lens's: a purple halo just outside the clipped
    lights, and the red and blue planes a little apart toward the corners (lateral chromatic aberration)."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    out = np.array(img, np.float32, copy=True)
    s = CA * amount
    for ch, z in ((0, 1.0 + s), (2, 1.0 - s)):
        M = np.float32([[z, 0.0, (1.0 - z) * (w - 1) / 2.0], [0.0, z, (1.0 - z) * (h - 1) / 2.0]])
        out[..., ch] = cv2.warpAffine(np.ascontiguousarray(img[..., ch]), M, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
    L = luma(out)
    hi = np.clip((L - 0.8) / 0.15, 0.0, 1.0)
    near = cv2.GaussianBlur(hi, (0, 0), max(2.5 * px, 0.6))
    ring = np.clip(1.8 * near - hi, 0.0, 1.0) * np.clip(1.0 - L, 0.0, 1.0) ** 0.5
    tone = (0.3 + 0.7 * near)[..., None] * np.asarray(FRINGE, np.float32)
    a = (0.85 * amount * ring)[..., None]
    return (out * (1.0 - a) + tone * a).astype(np.float32)


def ccd_noise(img, amount, rng):
    """`img` with a small sensor's noise: grainy luma noise, strongest in the shadows, and blotchy colour noise."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    ycc = cv2.cvtColor(np.ascontiguousarray(np.clip(img, 0.0, 1.0), np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0].copy()
    ycc[..., 0] = y + np.float32(amount) * (0.45 + 1.4 * (1.0 - y) ** 2) * rng.standard_normal((h, w), np.float32)
    small = rng.standard_normal((max(h // 3, 2), max(w // 3, 2), 2), np.float32)
    ycc[..., 1:] += np.float32(0.9 * amount) * cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) * \
        (0.4 + (1.0 - y) ** 2)[..., None]
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def oversharpen(img, amount, radius):
    """`img` through a camera's heavy unsharp mask on its luma: bright and dark halos either side of every edge."""
    if amount <= 0:
        return img
    y = np.ascontiguousarray(img @ LUMA)
    d = (y - cv2.GaussianBlur(y, (0, 0), max(radius, 0.3))) * np.float32(amount)
    return (img + d[..., None]).astype(np.float32)


def jpeg(img, quality):
    """`img` saved and read back as a JPEG of `quality` (1..100): 8 x 8 blocks, ringing, its colour in 16 x 16."""
    u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("OpenCV could not encode a JPEG")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def date_stamp(img, text, corner, colour, size, px):
    """`img` (in place) with an orange date imprint: `text` in leaning seven-segment digits `size` short sides tall at
    1080 in its `corner`, glowing a little as a camera's LEDs burned it in."""
    h, w = img.shape[:2]
    polys, width = EC.glyphs(text)
    if not polys:
        return img
    H = size * 1080.0 * px
    lean = math.tan(math.radians(STAMP_SLANT))
    margin = STAMP_MARGIN * min(w, h)
    x0 = w - margin - (width + lean) * H if corner.endswith("right") else margin
    y0 = h - margin - H if corner.startswith("bottom") else margin
    pts = [np.stack([x0 + (p[:, 0] + (1.0 - p[:, 1]) * lean) * H, y0 + p[:, 1] * H], 1) for p in polys]
    box = _roi(np.concatenate(pts), (w, h), 3.0 * STAMP_GLOW * H + 2.0)
    if box is None:
        return img
    a, b, c, d = box
    m = _fill((d - b, c - a), [p - (a, b) for p in pts])
    glow = cv2.GaussianBlur(m, (0, 0), max(STAMP_GLOW * H * 0.5, 0.6))
    part = _screen(img[b:d, a:c], (0.55 * glow)[..., None] * np.asarray(colour, np.float32))
    img[b:d, a:c] = _over(part, colour, 0.94 * m)
    return img


def digicam(img, fx, t, px, rng):
    """`img` (a frame, or a rect's part of one) as a 2005 point-and-shoot takes it: through its lens (barrel distortion,
    purple fringes), at its sensor's resolution (a pixel `pixel` px at 1080 across: the frame taken down to it, or up
    when the frame is smaller, so a preview shows the blocks a full frame does) with its noise, its white balance off and
    hunting, its hard tone curve and heavy sharpening, saved as a JPEG of quality `jpeg` and shown at the frame's size,
    the date stamped on."""
    h, w = img.shape[:2]
    x = barrel(np.clip(np.asarray(img, np.float32), 0.0, 1.0), fx["barrel"])
    x = fringe(x, fx["fringe"], px)
    k = fx["pixel"] * px                                         # frame px per camera pixel
    sw, sh = max(int(round(w / k)), 16), max(int(round(h / k)), 16)
    if (sw, sh) != (w, h):
        x = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA if sw < w else cv2.INTER_LINEAR)
    x = ccd_noise(x, fx["noise"], rng)
    x = x * np.asarray(EC.wb_gains(fx, t), np.float32)
    x = np.clip(0.5 + (x - 0.5) * np.float32(fx["contrast"]) + 0.03, 0.0, 1.0)
    x = oversharpen(x, fx["sharpen"], fx["halo"] / fx["pixel"])
    x = jpeg(x, fx["jpeg"])
    if (sw, sh) != (w, h):
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR if sw < w else cv2.INTER_AREA)
    if fx["date"]:
        date_stamp(x, fx["date"], fx["corner"], fx["color"], fx["size"], px)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ================================================================================================= chrome
def _ramp(colors, u):
    """The colours round a loop at `u` (any shape, wrapping at 1): linear between neighbours."""
    c = np.asarray(colors, np.float32)
    n = len(c)
    u = np.asarray(u, np.float32)
    if n == 1:
        return np.broadcast_to(c[0], u.shape + (3,)).astype(np.float32)
    v = (u % 1.0) * n
    i = np.floor(v).astype(int) % n
    f = (v - np.floor(v))[..., None]
    return (c[i] * (1.0 - f) + c[(i + 1) % n] * f).astype(np.float32)


@lru_cache(maxsize=16)
def _foil(colors, n=1024):
    """(n, 3) float32: the colours (a tuple of RGB tuples) round a loop at n steps, each at its full brightness."""
    t = _ramp(colors, np.arange(n, dtype=np.float32) / n)
    return (t / np.maximum(t.max(axis=1, keepdims=True), np.float32(1e-3))).astype(np.float32)


def chrome_sheen(pic, cov=None, phase=0.0, colors=((1.0, 0.8, 0.95), (0.75, 0.95, 1.0)), sheen=1.0, iris=0.65,
                 bands=2.5, px=1.0):
    """`pic` (h, w, 3) as iridescent chrome, over the whole array (lay it by `cov` yourself): a liquid-metal reflection of
    a sky over a dark horizon and the ground, in bands that follow its shading and wrap round its outline, moving on
    with `phase` (in bands), tinted round the loop of `colors` (display RGB: thin-film colours, each taken at its full
    brightness) by `iris` (0..1) and mixed over its own colours by `sheen`. With a coverage `cov` (h, w) the surface
    bulges in from the outline (a puffy dome), without one its luma is its relief. `px`: its pixels per px at 1080."""
    pic = np.clip(np.asarray(pic, np.float32), 0.0, 1.0)
    L = cv2.GaussianBlur(luma(pic), (0, 0), max(1.2 * px, 0.3))
    if cov is not None:
        inside = (np.asarray(cov) > 0.5).astype(np.uint8)
        d = cv2.distanceTransform(inside, cv2.DIST_L2, 5) if inside.any() else np.zeros(L.shape, np.float32)
        relief = 0.65 * np.sqrt(np.clip(d / np.float32(DOME * px), 0.0, 1.0)) + 0.35 * L
    else:
        relief = cv2.GaussianBlur(L, (0, 0), max(6.0 * px, 0.5))
    k = np.float32(RELIEF * px / 8.0)
    gx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3) * k
    gy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3) * k
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    up, right = gy / norm, -gx / norm                         # where the surface faces: up the screen, to the right
    q = np.float32(0.05 + phase) - 0.5 * up + np.float32(bands * 0.3) * (1.0 - L)   # the lights stay bright, the
    env = np.interp((q % 1.0).ravel(), ENV[:, 0], ENV[:, 1]).reshape(q.shape).astype(np.float32)   # shades band
    turn = cv2.phase(right, up) / np.float32(2.0 * math.pi)
    h, w = L.shape
    side = np.float32(HOLO / max(w, h))
    holo = np.arange(w, dtype=np.float32)[None, :] * side + np.arange(h, dtype=np.float32)[:, None] * side
    foil = _foil(tuple(tuple(float(v) for v in c) for c in colors))
    u = 0.8 * q + 0.35 * turn + holo + np.float32(0.5 * phase)     # the colours sweep across it as a foil's do
    tint = foil[(np.floor((u % 1.0) * len(foil)).astype(np.int32)) % len(foil)]
    hot = np.clip((env - 0.85) / 0.15, 0.0, 1.0)[..., None]
    metal = env[..., None] * ((1.0 - iris) + iris * tint)
    metal = metal * (1.0 - hot) + env[..., None] * hot
    return np.clip(pic * (1.0 - sheen) + metal * sheen, 0.0, 1.0).astype(np.float32)


def glow(img, px):
    """The light of `img`'s brightest parts blurred near and far (10 and 32 px at 1080): a bloom to screen over it. The
    blurs run on the picture taken down by up to 4 (a bloom has no fine detail to lose) and come back up."""
    hi = np.clip((luma(img) - 0.7) / 0.3, 0.0, 1.0) ** 2
    lit = img * hi[..., None]
    h, w = lit.shape[:2]
    k = int(max(1, min(4, round(10.0 * px / 2.5))))
    small = lit if k == 1 else cv2.resize(lit, (max(w // k, 1), max(h // k, 1)), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(small, (0, 0), max(10.0 * px / k, 0.5)) + \
        0.6 * cv2.GaussianBlur(small, (0, 0), max(32.0 * px / k, 1.0))
    return (out if k == 1 else cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)).astype(np.float32)


def _star(shape, cx, cy, R, angle):
    """(h, w) coverage of a sparkle star about (cx, cy): four long thin rays, four short ones between them."""
    polys = []
    c = np.array([cx, cy])
    for k in range(8):
        a = math.radians(angle) + k * math.pi / 4.0
        r = R if k % 2 == 0 else 0.42 * R
        d, n = np.array([math.cos(a), -math.sin(a)]), np.array([math.sin(a), math.cos(a)])
        polys.append(np.stack([c + n * 0.11 * R, c + d * r, c - n * 0.11 * R]))
    core = np.stack([(cx + 0.16 * R * math.cos(a), cy + 0.16 * R * math.sin(a))
                     for a in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)])
    return _fill(shape, polys + [core])


def sparkle(img, stars, colour):
    """`img` (in place) with sparkle stars [(x px, y px, radius px, angle)], white with a soft glow in `colour`."""
    h, w = img.shape[:2]
    for x, y, R, ang in stars:
        if R < 0.8:
            continue
        box = _roi(np.array([(x, y)]), (w, h), 1.6 * R + 2.0)
        if box is None:
            continue
        a, b, c, d = box
        m = _star((d - b, c - a), x - a, y - b, R, ang)
        g = cv2.GaussianBlur(m, (0, 0), max(0.18 * R, 0.5))
        part = _screen(img[b:d, a:c], (0.7 * g)[..., None] * np.asarray(colour, np.float32))
        img[b:d, a:c] = _over(part, (1.0, 1.0, 1.0), m)
    return img


def _disc(xx, yy, cx, cy, r):
    return np.clip(r - np.hypot(xx - cx, yy - cy) + 0.5, 0.0, 1.0)


def bubble(img, cx, cy, r, turn, colors):
    """`img` (in place) with a glossy Y2K bubble of radius `r` px about (cx, cy): a clear body faintly tinted, an
    iridescent rim whose colours run round it (from `turn`), a white oval of a highlight at its upper left and a small
    one at its lower right."""
    h, w = img.shape[:2]
    box = _roi(np.array([(cx, cy)]), (w, h), r + 3.0)
    if box is None or r < 2.0:
        return img
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    body = _disc(xx, yy, cx, cy, r)
    d = np.hypot(xx - cx, yy - cy)
    rim = np.clip(0.07 * r - np.abs(d - 0.93 * r) + 0.5, 0.0, 1.0) * body
    edge = np.clip((d / max(r, 1.0) - 0.55) / 0.45, 0.0, 1.0) ** 2 * body
    tint = _ramp(colors, np.arctan2(yy - cy, xx - cx) / np.float32(2.0 * math.pi) + turn)
    part = img[y0:y1, x0:x1]
    part = part * (1.0 - 0.12 * body[..., None]) + tint * (0.12 * body)[..., None]
    part = _screen(part, tint * (0.5 * edge + 0.6 * rim)[..., None])
    u, v = (xx - (cx - 0.38 * r)) / (0.42 * r), (yy - (cy - 0.42 * r)) / (0.24 * r)
    c, s = math.cos(math.radians(35.0)), math.sin(math.radians(35.0))
    oval = np.clip((1.0 - np.hypot(u * c + v * s, -u * s + v * c)) * 0.126 * r + 0.5, 0.0, 1.0)
    part = _over(part, (1.0, 1.0, 1.0), 0.85 * oval * body)
    img[y0:y1, x0:x1] = _over(part, (1.0, 1.0, 1.0), 0.6 * _disc(xx, yy, cx + 0.45 * r, cy + 0.45 * r, 0.1 * r))
    return img


def _box(m):
    """(x0, y0, x1, y1) px of where coverage `m` is half or more; None when nowhere."""
    rows, cols = np.nonzero((m >= 0.5).any(1))[0], np.nonzero((m >= 0.5).any(0))[0]
    if not rows.size:
        return None
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


# ================================================================================================= gradmap, invert
def lut(colors, n=1024):
    """(n, 3) float32: the gradient through `colors` (dark to light), evenly spaced over luma 0..1."""
    c = np.asarray(colors, np.float32)
    x = np.linspace(0.0, 1.0, len(c))
    u = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(u, x, c[:, k]) for k in range(3)], 1).astype(np.float32)


def gradient_map(img, table):
    """Each pixel of `img` by its luma through `table` (lut())."""
    return table[np.rint(luma(img) * (len(table) - 1)).astype(np.int32)]


def invert(img, hue=0.0):
    """`img` inverted, its hues then turned `hue` degrees (180: back to their own, light and dark swapped)."""
    out = 1.0 - np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    if abs(hue) % 360.0 > 1e-6:
        hsv = cv2.cvtColor(np.ascontiguousarray(out, np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + np.float32(hue)) % 360.0
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ================================================================================================= test card
def _lines(n, period, half, centre):
    """Coverage 0..1 along a side of `n` px of lines every `period` px through `centre` (px from the side's start, pixel
    i spanning i..i+1), `half` px either side of each."""
    x = np.arange(n, dtype=np.float32) + 0.5 - centre
    d = np.abs((x + period / 2.0) % period - period / 2.0)
    return np.clip(half - d + 0.5, 0.0, 1.0)


def _rect(img, box, colour, alpha=1.0):
    """`img` (in place) with the box (x0, y0, x1, y1) px (pixel i spanning i..i+1) filled in `colour`, antialiased."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    a, b = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    c, d = min(int(math.ceil(x1)), w), min(int(math.ceil(y1)), h)
    if c <= a or d <= b:
        return img
    xs, ys = np.arange(a, c, dtype=np.float32), np.arange(b, d, dtype=np.float32)
    cx = np.clip(np.minimum(xs + 1.0, x1) - np.maximum(xs, x0), 0.0, 1.0)
    cy = np.clip(np.minimum(ys + 1.0, y1) - np.maximum(ys, y0), 0.0, 1.0)
    img[b:d, a:c] = _over(img[b:d, a:c], colour, alpha * cy[:, None] * cx[None, :])
    return img


def _stripes(img, box, period, angle, dark, light):
    """`img` (in place) with the box filled with stripes of `dark` and `light`, `period` px apart, turned `angle`
    degrees from vertical, their edges softened to a pixel."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    a = math.radians(angle)
    u = (xx * math.cos(a) + yy * math.sin(a)) * np.float32(2.0 * math.pi / period)
    v = np.clip(0.5 + 0.5 * np.sin(u) * max(period / 3.0, 1.0), 0.0, 1.0)[..., None]
    img[y0:y1, x0:x1] = np.asarray(dark, np.float32) * (1.0 - v) + np.asarray(light, np.float32) * v
    return img


def _castellations(img, lay, inks):
    """`img` (in place) with blocks round its edge, half the grid's squares long: the palette's accents between white
    along the top, black and white down the sides (the right's half a block off the left's) and along the foot."""
    h, w = img.shape[:2]
    e, half = lay["edge"], lay["period"] / 2.0
    white, black, acc = inks["white"], inks["black"], inks["accents"]
    for side in ("top", "bottom", "left", "right"):
        n = w if side in ("top", "bottom") else h
        k0 = int(math.ceil((n / 2.0) / half)) + 1
        shift = half / 2.0 if side == "right" else 0.0
        for k in range(-k0, k0 + 1):
            a, b = n / 2.0 + k * half + shift, n / 2.0 + (k + 1) * half + shift
            if side == "top":
                col, box = (acc[(k // 2) % len(acc)] if k % 2 == 0 else white), (a, 0.0, b, e)
            elif side == "bottom":
                col, box = (black if k % 2 == 0 else white), (a, h - e, b, h)
            elif side == "left":
                col, box = (white if k % 2 == 0 else black), (0.0, a, e, b)
            else:
                col, box = (black if k % 2 == 0 else white), (w - e, a, w, b)
            _rect(img, box, col)
    for box in ((0, 0, e, e), (w - e, 0, w, e), (0, h - e, e, h), (w - e, h - e, w, h)):
        _rect(img, box, black)


def _arrows(img, lay, colour):
    """`img` (in place) with a small arrowhead in the middle of each edge, pointing out at it: a test card's marks of
    the picture's size."""
    h, w = img.shape[:2]
    e, a = lay["edge"], 0.32 * lay["period"]
    polys = []
    for (x, y), (dx, dy) in (((w / 2.0, e), (0, -1)), ((w / 2.0, h - e), (0, 1)), ((e, h / 2.0), (-1, 0)),
                             ((w - e, h / 2.0), (1, 0))):
        base = np.array([x - dx * a, y - dy * a])
        n = np.array([-dy, dx]) * a * 0.6
        polys.append(np.stack([(x - 0.5, y - 0.5), base + n - 0.5, base - n - 0.5]))
    img[:] = _over(img, colour, _fill((h, w), polys))


def card_ground(size, fx):
    """The still part of a test card over a frame of `size` (w, h): its grey ruled in squares, the castellations round
    its edge, the rows of colour bars across its top, the grey scale and the gratings flanking where the circle goes,
    stripes in its corners, arrowheads at its edges."""
    w, h = size
    lay = EC.card_layout(size, fx)
    inks = fx["inks"]
    img = np.empty((h, w, 3), np.float32)
    img[:] = fx["ground"]
    half = max(lay["s"] / 540.0, 0.6)
    gx, gy = _lines(w, lay["period"], half, w / 2.0), _lines(h, lay["period"], half, h / 2.0)
    img = _over(img, fx["line"], np.maximum(gx[None, :], gy[:, None]))
    _castellations(img, lay, inks)
    for (x0, y0, x1, y1), row in zip(lay["bars"], inks["rows"]):
        n = len(row)
        for k, c in enumerate(row):
            _rect(img, (x0 + (x1 - x0) * k / n, y0, x0 + (x1 - x0) * (k + 1) / n, y1), c)
    rim = max(lay["s"] / 400.0, 1.0)
    steps = len(lay["wedge"])
    for k, (x0, y0, x1, y1) in enumerate(lay["wedge"]):
        _rect(img, (x0 - rim, y0 - rim, x1 + rim, y1 + rim), inks["black"])
        u = k / (steps - 1)
        _rect(img, (x0, y0, x1, y1), [b + (a - b) * u for a, b in zip(inks["white"], inks["black"])])
    s = lay["s"] / 1080.0
    for (x0, y0, x1, y1), period in zip(lay["gratings"], GRATINGS):
        _rect(img, (x0 - rim, y0 - rim, x1 + rim, y1 + rim), inks["black"])
        _stripes(img, (x0, y0, x1, y1), max(period * s, 2.0), 0.0, inks["black"], inks["white"])
    for box in lay["corners"]:
        _stripes(img, box, max(9.0 * s, 2.0), 45.0, inks["black"], inks["white"])
    _arrows(img, lay, inks["black"])
    return img


@lru_cache(maxsize=64)
def _font(path, em):
    return ImageFont.truetype(path, em)


@lru_cache(maxsize=64)
def _cap_em(path, cap):
    """The font size (em, px) of the font at `path` whose capitals stand `cap` px tall."""
    f = ImageFont.truetype(path, 200)
    l, t, r, b = f.getbbox("H")
    return max(int(round(200.0 * cap / max(b - t, 1))), 4)


def label(img, text, cx, cy, cap, ink, paper, font, alpha=1.0):
    """`img` (in place) with `text` in capitals `cap` px tall in `ink` on a rounded box of `paper`, centred at (cx, cy)."""
    h, w = img.shape[:2]
    f = _font(font, _cap_em(font, max(int(round(cap)), 4)))
    pad = 0.6 * cap
    W, H = int(math.ceil(f.getlength(text) + 2.0 * pad)) + 2, int(math.ceil(2.4 * cap)) + 2
    x0, y0 = int(round(cx - W / 2.0)), int(round(cy - H / 2.0))
    box = Image.new("L", (W * 2, H * 2), 0)
    ImageDraw.Draw(box).rounded_rectangle([0, 0, W * 2 - 2, H * 2 - 2], radius=int(cap), fill=255)
    box = np.asarray(box.resize((W, H), Image.BOX), np.float32) / 255.0
    ink_m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ink_m).text((W / 2.0, H / 2.0), text, font=f, fill=255, anchor="mm")
    ink_m = np.asarray(ink_m, np.float32) / 255.0
    a, b = max(x0, 0), max(y0, 0)
    c, d = min(x0 + W, w), min(y0 + H, h)
    if c <= a or d <= b:
        return img
    sl = (slice(b - y0, d - y0), slice(a - x0, c - x0))
    part = _over(img[b:d, a:c], paper, alpha * box[sl])
    img[b:d, a:c] = _over(part, ink, alpha * ink_m[sl])
    return img


def testcard(img, fx, ground, centre, share, font):
    """`img` (a frame, or a rect's part of one) on a test card: its still `ground` (card_ground) with the picture inside
    the circle, moved so its point `centre` (px) sits in the circle's middle and scaled `zoom` times about it, the rim
    round the circle, the first label in a box on the rim's foot and the others small in the corners; blown open by
    `share` (0..1: the circle grown to cover the frame, the labels fading)."""
    h, w = img.shape[:2]
    lay = EC.card_layout((w, h), fx)
    cx, cy, r0 = lay["circle"]                                    # pixel i spans i..i+1 here
    cover = max(math.hypot(x - cx, y - cy) for x in (0.0, w) for y in (0.0, h)) + 4.0
    r = r0 + (cover - r0) * share
    z = fx["zoom"]
    tx, ty = cx - 0.5 - z * centre[0], cy - 0.5 - z * centre[1]   # OpenCV's pixels are centred on whole numbers
    if abs(z - 1.0) < 1e-6 and abs(tx) < 1e-3 and abs(ty) < 1e-3:
        scene = img
    else:
        scene = cv2.warpAffine(np.ascontiguousarray(img, np.float32), np.float32([[z, 0.0, tx], [0.0, z, ty]]), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    d = np.hypot(xx - cx, yy - cy)
    inside = np.clip(r - d + 0.5, 0.0, 1.0)[..., None]
    out = (ground * (1.0 - inside) + scene * inside).astype(np.float32)
    rim = max(lay["s"] / 110.0, 1.5)
    inks = fx["inks"]
    out = _over(out, inks["black"], np.clip(0.5 * rim + 1.0 - np.abs(d - (r + 0.5 * rim + 0.6)) + 0.5, 0.0, 1.0))
    out = _over(out, fx["line"], np.clip(0.5 * rim - np.abs(d - (r + 0.5 * rim)) + 0.5, 0.0, 1.0))
    if font is not None and fx["labels"] and share < 1.0:
        fade = 1.0 - share
        label(out, fx["labels"][0], *lay["label"], inks["white"], inks["black"], font, fade)
        p, e = lay["period"], lay["edge"]
        cap = 0.16 * p
        f = _font(font, _cap_em(font, max(int(round(cap)), 4)))
        for text, (x, y, side) in zip(fx["labels"][1:], ((e + 0.25 * p, h - e - 0.6 * p, -1),
                                                         (w - e - 0.25 * p, h - e - 0.6 * p, 1),
                                                         (w / 2.0, lay["bars"][-1][3] + 0.45 * p, 0))):
            half = (f.getlength(text) + 1.2 * cap) / 2.0
            label(out, text, x - side * half, y, cap, inks["white"], inks["black"], font, fade)
    return out


# ================================================================================================= flying cards
def _rounded_alpha(w, h, radius):
    """(h, w) float32: a rounded rectangle filling the picture, its corners `radius` px round, antialiased."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    qx = np.maximum(np.abs(xx - w / 2.0) - (w / 2.0 - radius), 0.0)
    qy = np.maximum(np.abs(yy - h / 2.0) - (h / 2.0 - radius), 0.0)
    straight = np.minimum(np.minimum(xx, w - xx), np.minimum(yy, h - yy))
    d = np.where((qx > 0) & (qy > 0), radius - np.hypot(qx, qy), straight)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def card_faces(palette, style):
    """(front, back): a flying card's two faces as premultiplied RGBA pictures CARD_TEX in size, corners rounded."""
    w, h = CARD_TEX
    a = _rounded_alpha(w, h, CARD_ROUND * h)[..., None]
    return (np.concatenate([CB.texture(w, h, palette, style) * a, a], 2).astype(np.float32),
            np.concatenate([CB.back(w, h, palette) * a, a], 2).astype(np.float32))


def project(p, size):
    """Camera-space points (n, 3) -> px (n, 2) of a frame of `size` (w, h) through the cards' lens."""
    w, h = size
    p = np.asarray(p, float)
    return np.stack([(w - 1) / 2.0 + EC.LENS * p[:, 0] / p[:, 2] * h, (h - 1) / 2.0 - EC.LENS * p[:, 1] / p[:, 2] * h], 1)


def _shade(n, centre):
    """(diffuse gain, glint) of a card face with outward normal `n` whose middle is at `centre`, camera space."""
    diffuse = max(float(np.dot(n, LIGHT)), 0.0)
    view = -np.asarray(centre) / max(float(np.linalg.norm(centre)), 1e-9)
    half = LIGHT + view
    half = half / max(float(np.linalg.norm(half)), 1e-9)
    return 0.62 + 0.48 * diffuse, 0.3 * max(float(np.dot(n, half)), 0.0) ** 40


def cards(img, fx, flights, faces, paper, opacity):
    """`img` with the cards in flight (cards_at's) drawn over it through the cards' lens, the farthest first: the faces
    of each slab that look at the lens (the front its colour bars, the back its back: `faces`, card_faces'; the edges
    `paper`), lit from the upper left, over a soft shadow; all at `opacity`."""
    h, w = img.shape[:2]
    out = np.array(img, np.float32, copy=True)
    for j, age, c, R in flights:
        P = EC.card_corners(fx, c, R)
        if P[:, 2].min() < 0.05:
            continue
        Q = project(P, (w, h))
        tall = EC.LENS * fx["size"] / c[2] * h
        box = _roi(Q, (w, h), 0.06 * tall + 3.0 * max(0.05 * tall, 0.6) + 2.0)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        o = (x0, y0)
        bw, bh = x1 - x0, y1 - y0
        part = out[y0:y1, x0:x1]
        if fx["shadow"] > 0:
            hull = cv2.convexHull(np.float32(Q)).reshape(-1, 2)
            m = cv2.GaussianBlur(_fill((bh, bw), [hull - o + (0.03 * tall, 0.05 * tall)]), (0, 0),
                                 max(0.05 * tall, 0.6))
            part = part * (1.0 - np.float32(0.6 * fx["shadow"] * opacity) * m[..., None])
        layer, cover = np.zeros((bh, bw, 3), np.float32), np.zeros((bh, bw), np.float32)
        for kind, idx, n0 in FACES:
            n = np.asarray(R) @ np.asarray(n0)
            mid = P[list(idx)].mean(0)
            if float(np.dot(n, mid)) >= 0.0:
                continue                                          # it looks away from the lens
            gain, glint = _shade(n, mid)
            quad = Q[list(idx)] - o
            if kind == "edge":
                a = _fill((bh, bw), [quad])
                colour = np.clip(np.asarray(paper, np.float32) * gain + glint, 0.0, 1.0)
                layer = layer * (1.0 - a[..., None]) + colour * a[..., None]
            else:
                tex = faces[0 if kind == "front" else 1]
                th, tw = tex.shape[:2]
                H = cv2.getPerspectiveTransform(np.float32([(0, 0), (tw, 0), (tw, th), (0, th)]) - 0.5,
                                                np.float32(quad))
                got = cv2.warpPerspective(tex, H, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
                a = got[..., 3]
                lit = np.clip(got[..., :3] * np.float32(gain) + np.float32(glint) * a[..., None], 0.0, 1.0)
                layer = layer * (1.0 - a[..., None]) + lit
            cover = cover + a * (1.0 - cover)
        k = np.float32(opacity)
        out[y0:y1, x0:x1] = part * (1.0 - k * cover[..., None]) + k * layer
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ================================================================================================= over a cut
class EraFx:
    """The era looks' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t, _at and
    _hit_times, and PictureFx's _region, _mix, _masked and _font_of). `_era_init` is called once, after `_picture_init`:
    a test card's font is found then, so a missing one is an FX.FxError before any frame is drawn."""

    def _era_init(self):
        self._era_fonts = {fx["index"]: self._font_of(fx["font"]) for fx in self.fxs
                           if fx["kind"] == "testcard" and fx["labels"]}
        self._era_luts, self._era_kept, self._era_grounds, self._era_faces = {}, {}, {}, {}

    def _on_frames(self, fx):
        """An entry's hits as the clip seconds of their frames (a hit acts from its frame)."""
        return [(FX.frame_of(h, self.fps, self.frame0) - self.frame0) / self.fps for h in self._hit_times(fx)]

    def _digicam(self, fx, frame, img):
        t = self._t(frame)
        rng = np.random.default_rng([fx["seed"], fx["index"], 41, frame])
        return self._mix(fx, frame, img, self._region(fx, img, lambda pic, box: digicam(pic, fx, t, self.px, rng)))

    def _chrome(self, fx, frame, img):
        """The sheen mixed in by the strength (and the figure's coverage with a `mask`; worked out over the figure's box
        alone), its bloom spilling past the outline, the sparkles on it (on the figure, with a mask), the bubbles over
        everything."""
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0:
            return img
        hits = self._on_frames(fx)
        region = self._masked(fx, frame)
        w, h = self.size
        box = (0.0, 0.0, float(w), float(h)) if region is None else _box(region)
        out = np.array(img, np.float32, copy=True)
        if box is not None:
            a0, b0 = max(int(box[0]) - 2, 0), max(int(box[1]) - 2, 0)
            a1, b1 = min(int(math.ceil(box[2])) + 2, w), min(int(math.ceil(box[3])) + 2, h)
            pic = out[b0:b1, a0:a1]
            cov = None if region is None else region[b0:b1, a0:a1]
            metal = chrome_sheen(pic, cov, EC.sheen_phase(fx, t, hits), fx["colors"], fx["sheen"], fx["iris"],
                                 fx["bands"], self.px)
            a = np.float32(s) if cov is None else (cov * np.float32(s))[..., None]
            out[b0:b1, a0:a1] = pic + (metal - pic) * a
            if fx["bloom"] > 0:
                g = int(math.ceil(96.0 * self.px)) + 2                # the far glow's reach: three of its sigmas
                c0, d0, c1, d1 = max(a0 - g, 0), max(b0 - g, 0), min(a1 + g, w), min(b1 + g, h)
                lit = np.zeros((d1 - d0, c1 - c0, 3), np.float32)
                lit[b0 - d0:b1 - d0, a0 - c0:a1 - c0] = metal if cov is None else metal * cov[..., None]
                out[d0:d1, c0:c1] = _screen(out[d0:d1, c0:c1], np.float32(fx["bloom"] * s) * glow(lit, self.px))
        if box is not None and fx["sparkles"]:
            x0, y0, x1, y1 = box
            stars = []
            for sp in EC.sparkles(fx, t, hits):
                x, y = x0 + sp["x"] * (x1 - x0), y0 + sp["y"] * (y1 - y0)
                if region is not None and region[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)] < 0.5:
                    continue                                      # off the figure
                stars.append((x, y, sp["size"] * STAR * min(w, h) * s, sp["angle"]))
            sparkle(out, stars, fx["colors"][0])
        for b in EC.bubbles(fx, t):
            bubble(out, b["x"] * w, b["y"] * h, b["r"] * min(w, h), b["turn"], fx["colors"])
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _gradmap(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        k = EC.map_step(fx, frame, self._hit_times(fx), self.fps, self.frame0)
        colours = fx["maps"][k]
        if s <= 0.0 or colours is None:
            return img
        got = self._era_luts.get((fx["index"], k))
        if got is None or got[0] != colours:                  # drawn again when its colours move (a breathing palette)
            got = self._era_luts[fx["index"], k] = (colours, lut(colours))
        region = self._masked(fx, frame)
        a = np.float32(s) if region is None else (region * np.float32(s))[..., None]
        return (img + (gradient_map(img, got[1]) - img) * a).astype(np.float32)

    def _invert(self, fx, frame, img):
        v = self._invert_level(fx, frame)
        if v <= 1e-4:
            return img
        region = self._masked(fx, frame)
        a = np.float32(v) if region is None else (region * np.float32(v))[..., None]
        return (img + (invert(img, fx["hue"]) - img) * a).astype(np.float32)

    def _invert_level(self, fx, frame):
        """How far an invert is on at `frame`: its level (eased in and out, on its thinned hits) times its strength."""
        t = self._t(frame)
        kept = None
        if fx.get("times") or fx.get("lane"):
            if fx["index"] not in self._era_kept:
                self._era_kept[fx["index"]] = EC.thin(self._on_frames(fx), fx["rate"])
            kept = self._era_kept[fx["index"]]
        return EC.level(fx, t, kept) * float(np.clip(FX.strength(fx, t), 0.0, 1.0))

    def _testcard(self, fx, frame, img):
        c, _ = self._at(fx, frame)
        share = EC.open_share(fx, self._t(frame))
        font = self._era_fonts.get(fx["index"])

        def draw(pic, box):
            x0, y0, x1, y1 = box
            key, inks = (fx["index"], box), (fx["ground"], fx["line"], fx["inks"])
            got = self._era_grounds.get(key)
            if got is None or got[0] != inks:                 # drawn again when its colours move (a breathing palette)
                got = self._era_grounds[key] = (inks, card_ground((x1 - x0, y1 - y0), fx))
            return testcard(pic, fx, got[1], (c[0] - x0, c[1] - y0), share, font)
        return self._mix(fx, frame, img, self._region(fx, img, draw))

    def _cards(self, fx, frame, img):
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        w, h = self.size
        flights = EC.cards_at(fx, t, self._on_frames(fx), w / h)
        if s <= 0.0 or not flights:
            return img
        got = self._era_faces.get(fx["index"])
        if got is None or got[0] != fx["palette"]:            # drawn again when its colours move (a breathing palette)
            got = self._era_faces[fx["index"]] = (fx["palette"], card_faces(fx["palette"], fx["style"]))
        return cards(img, fx, flights, got[1], CB.ends(fx["palette"])[0], s)
