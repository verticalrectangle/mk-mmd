"""Texture helpers for part builders: colours, gradients, antialiased drawing in UV space, noise, MMD toon ramps and
sphere maps, atlas packing. bpy-free; numpy + Pillow only (numpy-1.24 compatible, Python 3.11 syntax).

CONVENTIONS
    Images       float32 (h, w, 4) RGBA, 0..1, sRGB-ENCODED, straight (NON-premultiplied) alpha, row 0 = TOP row.
                 Functions that return images always return fresh arrays; inputs are never modified.
    UV space     (u, v) over the full image: u to the right, v UP, (0, 0) = bottom-left corner, v = 1 = the top row.
                 Image files are stored top row first; the PMX exporter flips v, so build everything with v up.
    Lengths      widths, radii and corner radii are UV units measured against the image WIDTH (0.01 = 1 % of the
                 width) in BOTH directions, so circles stay round on non-square images. Positions are plain UV.
    Colours      anything `color()` accepts: '#rgb' '#rrggbb' '#rrggbbaa', a 3/4-tuple of floats 0..1, a float array,
                 a uint8 array (0..255). Alpha defaults to 1. Colours are sRGB values: painters' arithmetic
                 (gradients, toon ramps, blends) happens on the sRGB numbers unless a function says `linear`.
    Blend modes  'normal' 'multiply' 'screen' 'add' 'overlay' (+ 'erase' for Canvas/composite), evaluated on the
                 sRGB numbers with the W3C compositing equations (alpha-correct on straight-alpha images).
    Toon / sphere  see toon_ramp / sphere_map: the TOP row of a toon texture is the lit side; a sphere map is sampled
                 with the VIEW-space normal: u = 0.5 + 0.5 nx, v (up) = 0.5 + 0.5 ny.
    Sampling     Gradients, noise, sphere maps and masks are evaluated at PIXEL CENTRES, (i + 0.5) / w, so the first
                 row of a top-to-bottom gradient is half a pixel step away from stop 0. Toon textures are lookup
                 tables: row r is t = r / (h - 1), the first and last rows are exactly the first and last stops.
    Transparent texels: bilinear filtering blends in the RGB of fully transparent texels; `bleed()` an image that has
                 transparent areas (lashes, blush, frills) before saving it so no dark fringe appears.

API (details in each docstring)
    colour      color  mix  srgb_to_linear  linear_to_srgb  shade  shift_hue  lighten  darken  hsv  to_hex  smoothstep
    images      new  to_uint8  to_float  save_png  load_png  composite  paste  resize  blur  bleed  colorize  uv_grid
    noise       value_noise  streaks
    gradients   linear_gradient  radial_gradient
    drawing     Canvas  (fill_all polygon polyline ellipse circle rect curve bezier fill_gradient fill_radial stamp
                shape_mask image save)  catmull_rom  bezier_points
    MMD         toon_ramp  toon_bands  sphere_map  sphere_highlight  sphere_rim  sphere_flat
    packing     Atlas  (alloc rects blit map_uv image)  uv_rect_pixels  remap_uv

    from mkmmd.model import tex
    cv = tex.Canvas(512, 512, bg="#f1e7d6")
    cv.circle((0.5, 0.5), 0.2, fill="#dd5555", stroke="#552222", width=0.01)
    cv.curve([(0.2, 0.8), (0.5, 0.9), (0.8, 0.8)], 0.012, "#222", widths=[0.002, 0.012, 0.002])
    grain = tex.colorize(tex.value_noise(512, 512, 24, seed=3) * 0.15, "#000")      # noise as a faint overlay
    tex.save_png("skin.png", tex.composite(cv.image(), grain))
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "MODES", "color", "mix", "srgb_to_linear", "linear_to_srgb", "shade", "shift_hue", "lighten", "darken", "hsv",
    "to_hex", "smoothstep", "new", "to_uint8", "to_float", "save_png", "load_png", "composite", "paste", "resize",
    "blur", "bleed", "colorize", "uv_grid", "value_noise", "streaks", "linear_gradient", "radial_gradient", "Canvas",
    "catmull_rom", "bezier_points", "toon_ramp", "toon_bands", "sphere_map", "sphere_highlight", "sphere_rim",
    "sphere_flat", "Atlas", "uv_rect_pixels", "remap_uv",
]

_EPS = 1e-6
MODES = ("normal", "multiply", "screen", "add", "overlay", "erase")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


# ===================================================================================================== colours
def _parse_hex(s):
    h = s.strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) in (3, 4):
        h = "".join(ch * 2 for ch in h)
    if len(h) not in (6, 8):
        raise ValueError("bad colour string %r (use '#rgb', '#rrggbb' or '#rrggbbaa')" % (s,))
    try:
        vals = [int(h[i:i + 2], 16) for i in range(0, len(h), 2)]
    except ValueError:
        raise ValueError("bad colour string %r (not hexadecimal)" % (s,)) from None
    if len(vals) == 3:
        vals.append(255)
    return np.array(vals, dtype=np.float32) / np.float32(255.0)


def color(c):
    """Any colour spec -> float32 array (4,) RGBA, straight alpha, sRGB, clipped to 0..1.

    Accepts '#rgb', '#rrggbb', '#rrggbbaa' (also '#rgba', the '#' is optional), a 3- or 4-sequence of floats 0..1,
    a float array of that shape, a uint8 array (0..255) or a single float (grey). Alpha defaults to 1. A tuple or
    list outside 0..1 (e.g. 0..255 ints) raises ValueError instead of silently clipping to white; float arrays
    (which come out of computations) are clipped silently. Returns a new array every call."""
    if isinstance(c, str):
        return _parse_hex(c)
    if isinstance(c, np.ndarray):
        a = c.astype(np.float32) / np.float32(255.0) if c.dtype == np.uint8 else c.astype(np.float32)
        strict = False
    else:
        try:
            a = np.asarray(c, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("cannot read a colour from %r" % (c,)) from None
        strict = True
    if a.ndim == 0:
        a = np.full(3, float(a))
    if a.ndim != 1 or a.shape[0] not in (3, 4):
        raise ValueError("a colour needs 3 or 4 components, got shape %s" % (a.shape,))
    if not np.all(np.isfinite(a)):
        raise ValueError("colour components must be finite numbers (got %r)" % (c,))
    if strict and (a.min() < -1e-3 or a.max() > 1.0 + 1e-3):
        raise ValueError("colour components must be in 0..1 (got %r); use '#rrggbb' for 0..255 values" % (c,))
    out = np.ones(4, np.float32)
    out[:a.shape[0]] = a
    return np.clip(out, 0.0, 1.0, out=out)


def _rgba(x):
    """Colour spec OR float/uint8 array whose last axis is 3 or 4 -> float32 (..., 4) (new array)."""
    if isinstance(x, str):
        return color(x)
    a = np.asarray(x)
    if a.ndim <= 1:
        return color(x)
    a = a.astype(np.float32) / np.float32(255.0) if a.dtype == np.uint8 else a.astype(np.float32)
    n = a.shape[-1]
    if n == 4:
        return a
    if n == 3:
        return np.concatenate([a, np.ones(a.shape[:-1] + (1,), np.float32)], axis=-1)
    raise ValueError("expected an array with 3 or 4 channels, got shape %s" % (a.shape,))


def _f(x):
    a = np.asarray(x)
    return a if a.dtype in (np.float32, np.float64) else a.astype(np.float32)


def srgb_to_linear(x):
    """sRGB-encoded values -> linear light (elementwise, the standard piecewise curve). Input clipped to 0..1.
    Works on any shape and does not treat alpha specially: pass RGB only. float64 in -> float64 out, else float32."""
    x = np.clip(_f(x), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(x.dtype, copy=False)


def linear_to_srgb(x):
    """Linear light -> sRGB-encoded values (inverse of srgb_to_linear; elementwise, input clipped to 0..1)."""
    x = np.clip(_f(x), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(x.dtype, copy=False)


def mix(a, b, t, linear=True):
    """Blend colour `a` towards `b` by `t` (0 -> a, 1 -> b) in linear light (`linear=True`, physically right) or on
    the plain sRGB numbers (`linear=False`, what painters expect from a gradient).

    a, b: colour specs OR float arrays (..., 3|4) such as images. t: float or an array holding one factor per pixel
    (it must broadcast against the pixel dimensions of a and b, i.e. shape (h, w) for (h, w, 4) images).
    Alpha is interpolated linearly and the colours are weighted by alpha (premultiplied interpolation), so fading
    to a transparent colour fades out instead of passing through its (meaningless) RGB. For opaque colours this is
    the plain interpolation. Returns float32 (..., 4): shape (4,) for a scalar t and two colour specs."""
    A, B = _rgba(a), _rgba(b)
    T = np.asarray(t, dtype=np.float32)[..., None]
    ca, cb = A[..., :3], B[..., :3]
    if linear:
        ca, cb = srgb_to_linear(ca), srgb_to_linear(cb)
    aa, ab = A[..., 3:], B[..., 3:]
    alpha = aa * (1 - T) + ab * T
    ok = alpha > _EPS
    rgb = np.where(ok, (ca * aa * (1 - T) + cb * ab * T) / np.where(ok, alpha, 1.0), ca * (1 - T) + cb * T)
    if linear:
        rgb = linear_to_srgb(rgb)
    return np.concatenate([rgb, alpha], axis=-1).astype(np.float32)


def shade(c, f):
    """Multiply a colour by `f` in LINEAR light (f < 1 darker, f > 1 lighter, alpha kept), clipped to the gamut.

    c: colour spec or image-like array (..., 3|4); f: one factor per colour / pixel (a scalar, or an array that
    broadcasts against the pixel dimensions; NOT per channel). 0.5 is about 'one stop' darker. Returns (..., 4)."""
    A = _rgba(c)
    F = np.asarray(f, dtype=np.float32)[..., None]
    rgb = linear_to_srgb(np.clip(srgb_to_linear(A[..., :3]) * F, 0.0, 1.0))
    return np.concatenate([rgb, np.broadcast_to(A[..., 3:], rgb.shape[:-1] + (1,))], axis=-1).astype(np.float32)


def _rgb_to_hsv(rgb):
    """(..., 3) 0..1 -> hue (degrees 0..360), saturation, value arrays."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    d = mx - rgb.min(axis=-1)
    nz = d > 1e-12
    dd = np.where(nz, d, 1.0)
    h = np.where(mx == r, ((g - b) / dd) % 6.0, np.where(mx == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0))
    h = np.where(nz, h, 0.0) * 60.0
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    h = (np.asarray(h) % 360.0) / 60.0
    i = np.floor(h)
    f = h - i
    i = i.astype(np.int64) % 6
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def shift_hue(c, degrees):
    """Rotate the hue of a colour by `degrees` (HSV on the sRGB numbers; saturation and value kept, alpha kept).
    +120 turns red into green. c: colour spec or image-like array; degrees: scalar or per-pixel array."""
    A = _rgba(c)
    h, s, v = _rgb_to_hsv(A[..., :3])
    rgb = _hsv_to_rgb(h + np.asarray(degrees, dtype=np.float64), s, v)
    return np.concatenate([rgb, np.broadcast_to(A[..., 3:], rgb.shape[:-1] + (1,))], axis=-1).astype(np.float32)


def hsv(h, s=1.0, v=1.0, a=1.0):
    """Colour from hue (degrees), saturation and value (0..1), alpha `a` -> float32 (4,) sRGB."""
    rgb = _hsv_to_rgb(np.float64(h), np.float64(s), np.float64(v))
    return np.clip(np.array([rgb[0], rgb[1], rgb[2], a], dtype=np.float32), 0.0, 1.0)


def lighten(c, t, linear=True):
    """Mix a colour towards white by `t` (0 = unchanged, 1 = white); alpha kept. Same arguments as `mix` (linear
    light by default, so 0.5 is a clearly paler colour). c: colour spec / image-like array; t: scalar or per pixel."""
    A = _rgba(c)
    target = np.ones_like(A)
    target[..., 3] = A[..., 3]
    return mix(A, target, t, linear)


def darken(c, t, linear=True):
    """Mix a colour towards black by `t` (0 = unchanged, 1 = black); alpha kept. Same arguments as `mix`; for a
    multiplicative 'f stops darker' use `shade` instead."""
    A = _rgba(c)
    target = np.zeros_like(A)
    target[..., 3] = A[..., 3]
    return mix(A, target, t, linear)


def to_hex(c):
    """'#rrggbb' for a colour spec ('#rrggbbaa' when alpha < 1), handy for logs and TOML."""
    u = np.floor(color(c) * 255.0 + 0.5).astype(np.int64)
    s = "#%02x%02x%02x" % (u[0], u[1], u[2])
    return s if u[3] == 255 else s + "%02x" % u[3]


def smoothstep(e0, e1, x):
    """Hermite smoothstep of `x` between edges e0 and e1 (0 below e0, 1 above e1; a hard step when e0 == e1).
    Scalar or array x; returns float64."""
    x = np.asarray(x, dtype=np.float64)
    if e1 == e0:
        return (x >= e0).astype(np.float64)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ======================================================================================================= images
def _to4(a, one):
    """(h, w) | (h, w, 1|3|4) -> (h, w, 4); `one` is the opaque alpha value of a.dtype."""
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError("expected an image (h, w[, 1|3|4]), got shape %s" % (a.shape,))
    c = a.shape[2]
    if c == 4:
        return a.copy()
    h, w = a.shape[:2]
    if c == 1:
        a = np.repeat(a, 3, axis=2)
    return np.concatenate([a, np.full((h, w, 1), one, a.dtype)], axis=2)


def to_float(img):
    """Any uint8 / float / bool image with 1, 3 or 4 channels (or (h, w) grey) -> a NEW float32 (h, w, 4) in 0..1.
    uint8 is divided by 255, floats are clipped to 0..1, missing alpha becomes 1, grey is replicated."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        f = a.astype(np.float32) / np.float32(255.0)
    elif a.dtype == np.bool_ or a.dtype.kind == "f":
        f = a.astype(np.float32)
    else:
        raise TypeError("images must be uint8 or float, got %s" % a.dtype)
    f = _to4(f, 1.0)
    return np.clip(f, 0.0, 1.0, out=f)


def to_uint8(img, dither=False):
    """Any uint8 / float / bool image with 1, 3 or 4 channels -> NEW uint8 (h, w, 4), rounding to nearest
    (floats are clipped to 0..1 first). `dither=True` rounds the RGB stochastically instead (deterministic noise,
    unbiased on average; alpha is never dithered) to hide banding in very smooth gradients."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return _to4(a, 255)
    f = to_float(a)
    scaled = f * np.float32(255.0)
    if dither:
        noise = np.random.default_rng(0).random(scaled.shape[:2] + (3,), dtype=np.float32)
        out = np.floor(scaled + 0.5)
        out[..., :3] = np.floor(scaled[..., :3] + noise)
    else:
        out = np.floor(scaled + 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def new(w, h, c=(0, 0, 0, 0)):
    """A new float32 (h, w, 4) image filled with the colour `c` (default fully transparent black)."""
    img = np.empty((int(h), int(w), 4), np.float32)
    img[...] = color(c)
    return img


def save_png(path, img, dither=False):
    """Write `img` (any input `to_uint8` accepts) as an 8-bit RGBA PNG (compress_level 6), creating parent folders.
    Straight alpha is stored as is (no premultiplication). `~` is expanded. Returns the file's `pathlib.Path`."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img, dither=dither)).save(p, format="PNG", compress_level=6)
    return p


def load_png(path):
    """Read any PNG (palette, grey, RGB, RGBA, 16-bit) as float32 (h, w, 4) straight alpha, 0..1."""
    with Image.open(Path(path).expanduser()) as im:
        if im.mode in ("I;16", "I;16L", "I;16B", "I"):
            return to_float(np.asarray(im, dtype=np.float32) / 65535.0)
        return to_float(np.asarray(im.convert("RGBA")))


def _img(x):
    """Float32 (h, w, 4) input without copying when it already is one (read-only use)."""
    if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 3 and x.shape[2] == 4:
        return x
    return to_float(x)


def _premul(img):
    p = img.copy()
    p[..., :3] *= p[..., 3:4]
    return p


def _unpremul(p, fallback=None):
    """Premultiplied (..., 4) -> straight alpha; where alpha ~ 0 the RGB comes from `fallback` (or is 0)."""
    a = p[..., 3:4]
    ok = a > _EPS
    rgb = np.where(ok, p[..., :3] / np.where(ok, a, 1.0), 0.0 if fallback is None else fallback)
    return np.clip(np.concatenate([rgb, a], axis=-1), 0.0, 1.0).astype(np.float32)


def _blend_planes(P, rgb, a, mode):
    """Core blend on channel planes (the fast form: planes are 1-D gathers, 2-D views or scalars).

    P: four premultiplied backdrop channel arrays (r, g, b, alpha) of equal shape; rgb: three straight source
    colour channels (arrays or scalars); a: source alpha (array or scalar). Returns four NEW premultiplied channel
    arrays (W3C compositing equations in premultiplied form: no division except for 'overlay')."""
    if mode not in MODES:
        raise ValueError("unknown blend mode %r (use one of %s)" % (mode, ", ".join(MODES)))
    pr, pg, pb, ab = P
    if mode == "erase":
        k = 1 - a
        return [pr * k, pg * k, pb * k, ab * k]
    ar = a + ab - a * ab
    out = []
    for p, c in zip((pr, pg, pb), rgb):
        if mode == "normal":
            r = c * a + p * (1 - a)
        elif mode == "multiply":
            r = (1 - ab) * a * c + (1 - a) * p + a * p * c
        elif mode == "screen":
            r = p + a * c * (1 - p)
        elif mode == "add":
            r = np.minimum(p + a * c, ar)
        else:  # overlay: the backdrop colour decides between multiply and screen
            cb = p / np.maximum(ab, _EPS)
            r = (1 - ab) * a * c + (1 - a) * p + a * ab * np.where(cb <= 0.5, 2 * cb * c, 1 - 2 * (1 - cb) * (1 - c))
        out.append(r)
    out.append(ar)
    return out


def _blend_pm(P, rgb, a, mode):
    """`_blend_planes` for arrays. P: premultiplied backdrop (..., 4); rgb: straight source colour (..., 3) or (3,);
    a: source alpha (...) or scalar. Returns the NEW premultiplied float32 result (..., 4)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    out = _blend_planes([P[..., k] for k in range(4)], [rgb[..., k] for k in range(3)],
                        np.asarray(a, dtype=np.float32), mode)
    return np.stack(out, axis=-1).astype(np.float32, copy=False)


def composite(dst, src, mode="normal", opacity=1.0):
    """Composite `src` over `dst` (same size) with a blend mode; returns a NEW float32 (h, w, 4) image.

    Straight-alpha images, W3C compositing: the blend function acts on the sRGB numbers, weighted by the backdrop's
    alpha. mode: 'normal', 'multiply', 'screen', 'add' (clipped), 'overlay' (backdrop decides: dark -> multiply,
    light -> screen) or 'erase' (src alpha cuts dst alpha away). opacity: scalar or an (h, w) array that scales
    src's alpha (a per-pixel mask). Where the result is fully transparent dst's RGB is kept."""
    d, s = _img(dst), _img(src)
    if d.shape != s.shape:
        raise ValueError("composite needs images of equal size, got %s and %s" % (d.shape, s.shape))
    a = s[..., 3] * np.asarray(opacity, dtype=np.float32)
    res = _blend_pm(_premul(d), s[..., :3], a, mode)
    return _unpremul(res, fallback=d[..., :3])


def _planes_pm(img):
    """Straight float32 RGBA -> four PIL 'F' planes (premultiplied r, g, b, alpha) for alpha-correct resampling."""
    p = _premul(img)
    return [Image.fromarray(np.ascontiguousarray(p[..., k])) for k in range(4)]


def _resample_pm(planes, out_w, out_h, box=None):
    """Lanczos-resample premultiplied planes (optionally only the source `box`) -> premultiplied (h, w, 4)."""
    out = np.stack([np.asarray(p.resize((out_w, out_h), _LANCZOS, box=box), dtype=np.float32) for p in planes], axis=-1)
    a = np.clip(out[..., 3], 0.0, 1.0)
    out[..., 3] = a
    out[..., :3] = np.clip(out[..., :3], 0.0, a[..., None])
    return out


def resize(img, w, h):
    """Lanczos resize to (w, h) pixels, alpha-correct (colours are weighted by alpha, so transparent texels do not
    darken edges). Returns a NEW float32 (h, w, 4) straight-alpha image."""
    s = _img(img)
    w, h = int(w), int(h)
    if s.shape[:2] == (h, w):
        return s.copy()
    return _unpremul(_resample_pm(_planes_pm(s), w, h))


def uv_rect_pixels(rect, w, h):
    """UV rect (u0, v0, u1, v1) (v up) of a w x h image -> pixel box (x0, y0, x1, y1) with y counted from the TOP,
    rounded to integers, ready for `img[y0:y1, x0:x1]`. The corners may be given in any order."""
    u0, v0, u1, v1 = (float(x) for x in rect)
    x0, x1 = sorted((int(round(u0 * w)), int(round(u1 * w))))
    y0, y1 = sorted((int(round((1.0 - v1) * h)), int(round((1.0 - v0) * h))))
    return x0, y0, x1, y1


def remap_uv(uv, rect):
    """Map local UVs (0..1 over an island) into a UV rect (u0, v0, u1, v1) of an atlas: u' = u0 + u (u1 - u0),
    v' = v0 + v (v1 - v0). uv: array (..., 2); returns a new float64 array of the same shape."""
    u0, v0, u1, v1 = (float(x) for x in rect)
    a = np.array(uv, dtype=np.float64)
    a[..., 0] = u0 + a[..., 0] * (u1 - u0)
    a[..., 1] = v0 + a[..., 1] * (v1 - v0)
    return a


def paste(dst, src, uv_rect, mode="normal"):
    """Resample `src` (Lanczos, alpha-correct) into the UV rectangle (u0, v0, u1, v1) of `dst` and composite it there
    with `mode` ('normal' by default, any `composite` mode). The rectangle is rounded to whole pixels; the part
    outside `dst` is cropped. Returns a NEW float32 image of dst's size; dst is untouched."""
    d, s = _img(dst), _img(src)
    H, W = d.shape[:2]
    x0, y0, x1, y1 = uv_rect_pixels(uv_rect, W, H)
    out = d.copy()
    rw, rh = x1 - x0, y1 - y0
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if rw <= 0 or rh <= 0 or cx0 >= cx1 or cy0 >= cy1:
        return out
    sh, sw = s.shape[:2]
    if (rh, rw) == (sh, sw):
        part = s[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    else:
        box = ((cx0 - x0) * sw / rw, (cy0 - y0) * sh / rh, (cx1 - x0) * sw / rw, (cy1 - y0) * sh / rh)
        part = _unpremul(_resample_pm(_planes_pm(s), cx1 - cx0, cy1 - cy0, box=box))
    out[cy0:cy1, cx0:cx1] = composite(out[cy0:cy1, cx0:cx1], part, mode)
    return out


def colorize(mask, c):
    """A float (h, w) mask (coverage / noise, 0..1) -> an (h, w, 4) image of colour `c` whose alpha is the mask times
    c's alpha. The bridge between per-pixel numpy work (`Canvas.shape_mask`, `value_noise`) and compositing: stamp
    the result with `composite(img, colorize(mask, c), mode)`, or skip the intermediate image with
    `composite(img, new(w, h, c), mode, opacity=mask)` (opacity may be an (h, w) array)."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    col = color(c)
    out = np.empty(m.shape + (4,), np.float32)
    out[..., :3] = col[:3]
    out[..., 3] = m * col[3]
    return out


def uv_grid(w, h):
    """Pixel-centre UV coordinates of a w x h image: (u, v) float32 arrays of shape (h, w); v is UP (row 0 has the
    largest v). For custom per-pixel work: `u, v = uv_grid(w, h); mask = (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.04`."""
    u = (np.arange(w, dtype=np.float32) + 0.5) / np.float32(w)
    v = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / np.float32(h)
    return np.broadcast_to(u[None, :], (h, w)).copy(), np.broadcast_to(v[:, None], (h, w)).copy()


# ------------------------------------------------------------------------------------------- blur and bleed
def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3.5 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def _fft_len(n):
    """Smallest 2^a 3^b 5^c >= n (a fast FFT length)."""
    best = 1 << max(0, int(n) - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = -(-int(n) // p35)
            best = min(best, p35 << max(0, (m - 1).bit_length()))
            p35 *= 3
        p5 *= 5
    return best


def _conv_axis(a, kern, r, axis, wrap):
    """Convolve float32 array `a` along `axis` with the symmetric kernel (length 2r+1); clamped or wrapped edges."""
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="wrap" if wrap else "edge")
    sl = [slice(None)] * a.ndim
    if kern.size <= 33:
        out = np.zeros(a.shape, np.float32)
        for i, kv in enumerate(kern):
            sl[axis] = slice(i, i + n)
            out += np.float32(kv) * p[tuple(sl)]
        return out
    length = _fft_len(n + 4 * r)
    spec = np.fft.rfft(p, n=length, axis=axis)
    kspec = np.fft.rfft(kern, n=length)
    shape = [1] * a.ndim
    shape[axis] = kspec.size
    y = np.fft.irfft(spec * kspec.reshape(shape), n=length, axis=axis)
    sl[axis] = slice(2 * r, 2 * r + n)
    return y[tuple(sl)].astype(np.float32)


def blur(img, sigma, wrap=False):
    """Gaussian blur, `sigma` in PIXELS (separable; small kernels directly, large ones by FFT). Returns a new array.

    img: an RGBA image (h, w, 4) is blurred ALPHA-WEIGHTED (colours are averaged with their alpha as weight, so
    transparent texels never darken the colour; fully transparent areas keep their RGB); any other array ((h, w)
    masks, (h, w, 3), ...) is blurred per channel as is. Edges are clamped, or wrap around with `wrap=True`
    (tileable textures). sigma <= 0 returns a copy."""
    a = np.asarray(img)
    a = a.astype(np.float32) / np.float32(255.0) if a.dtype == np.uint8 else a.astype(np.float32)
    if sigma <= 0:
        return a
    kern, r = _gauss_kernel(float(sigma))

    def run(arr):
        return _conv_axis(_conv_axis(arr, kern, r, 0, wrap), kern, r, 1, wrap)

    if a.ndim == 3 and a.shape[2] == 4:
        al = a[..., 3:4]
        b = run(np.concatenate([a[..., :3] * al, al], axis=-1))
        ba = b[..., 3:4]
        ok = ba > _EPS
        rgb = np.where(ok, b[..., :3] / np.where(ok, ba, 1.0), a[..., :3])
        return np.clip(np.concatenate([rgb, ba], axis=-1), 0.0, 1.0).astype(np.float32)
    return run(a)


def bleed(img, iterations=8):
    """Spread the RGB of non-transparent pixels into fully transparent neighbours, alpha untouched.

    Each iteration grows the coloured area by one pixel (8-neighbourhood, a filled pixel takes the mean RGB of the
    already-coloured neighbours); `iterations` is therefore the reach in pixels. Bilinear filtering and mipmaps
    blend the RGB of transparent texels into the edge; with the colour bled out there is no dark fringe. Pixels
    with alpha < 1/510 (which round to 0 in an 8-bit file) count as transparent. Returns a NEW float32 image."""
    out = to_float(img)
    H, W = out.shape[:2]
    valid = out[..., 3] >= 0.5 / 255.0
    if iterations < 1 or valid.all() or not valid.any():
        return out
    inv = ~valid
    near = np.zeros_like(valid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                near[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] |= \
                    inv[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    frontier = np.flatnonzero(valid & near)
    rgb = out[..., :3].reshape(-1, 3)
    vflat = valid.reshape(-1).copy()
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for _ in range(int(iterations)):
        if frontier.size == 0:
            break
        y, x = np.divmod(frontier, W)
        ids, srcs = [], []
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nid = ny[ok] * W + nx[ok]
            keep = ~vflat[nid]
            ids.append(nid[keep])
            srcs.append(frontier[ok][keep])
        cid = np.concatenate(ids)
        if cid.size == 0:
            break
        src = np.concatenate(srcs)
        uniq, inv_idx = np.unique(cid, return_inverse=True)
        inv_idx = inv_idx.reshape(-1)
        cnt = np.bincount(inv_idx).astype(np.float64)
        for k in range(3):
            rgb[uniq, k] = (np.bincount(inv_idx, weights=rgb[src, k].astype(np.float64)) / cnt).astype(np.float32)
        vflat[uniq] = True
        frontier = uniq
    return out


# ======================================================================================================= noise
def _scale2(scale):
    sx, sy = (scale, scale) if np.ndim(scale) == 0 else scale
    sx, sy = float(sx), float(sy)
    if not (sx > 0.0 and sy > 0.0 and math.isfinite(sx) and math.isfinite(sy)):
        raise ValueError("noise scale must be positive and finite, got %r" % (scale,))
    return sx, sy


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _noise_layer_add(acc, amp, sx, sy, seed, octave, tileable):
    """Add `amp` x one octave of lattice value noise (0..1, sx x sy lattice cells across the image) to the float32
    accumulator `acc` (h, w) in place. Only (h, w) float32 temporaries are made (no float64 images)."""
    h, w = acc.shape
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(octave)])
    if tileable:
        nx, ny = max(2, int(round(sx))), max(2, int(round(sy)))
        lat = rng.random((ny, nx))
        gx = (np.arange(w) + 0.5) * (nx / w)
        gy = (np.arange(h) + 0.5) * (ny / h)
        ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix, gy - iy
        ix0, ix1, iy0, iy1 = ix % nx, (ix + 1) % nx, iy % ny, (iy + 1) % ny
    else:
        lat = rng.random((int(math.ceil(sy)) + 2, int(math.ceil(sx)) + 2))
        gx = (np.arange(w) + 0.5) * (sx / w)
        gy = (np.arange(h) + 0.5) * (sy / h)
        ix0, iy0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix0, gy - iy0
        ix1, iy1 = ix0 + 1, iy0 + 1
    lat = lat.astype(np.float32)
    tx = _fade(fx).astype(np.float32)
    wy = _fade(fy)
    w0, w1 = ((1.0 - wy) * amp).astype(np.float32), (wy * amp).astype(np.float32)     # amp folded into the v weights
    rows = lat[:, ix0] * (1.0 - tx) + lat[:, ix1] * tx                                # (lattice rows, w)
    tmp = rows[iy0]
    tmp *= w0[:, None]
    acc += tmp
    tmp = rows[iy1]
    tmp *= w1[:, None]
    acc += tmp


def value_noise(w, h, scale=8.0, seed=0, octaves=1, tileable=False, gain=0.5, normalize=False):
    """Smooth lattice value noise (fBm over `octaves`) -> float32 (h, w) in 0..1.

    scale: lattice cells ACROSS THE IMAGE, in u and in v: a float (same in both) or (su, sv). On a non-square image
    pass a tuple to get square cells (e.g. (8, 4) on a 2:1 image). Octave k has scale * 2^k cells and amplitude
    gain^k; the sum is divided by the total amplitude, so values stay in 0..1 (several octaves cluster around 0.5;
    `normalize=True` stretches the result to span exactly 0..1). The lattice depends only on seed, octave and scale,
    never on w/h: the same call at 128 px and at 1024 px gives the same picture. tileable=True makes the image wrap
    seamlessly in u and v (every octave's scale is rounded to whole cells, at least 2). Deterministic for a given
    seed. Use it as a mask / modulation: `colorize(value_noise(w, h, 12) * 0.2, '#000')` is a 20 % dirt overlay,
    `shade(img, 0.9 + 0.2 * value_noise(...))` a brightness mottle."""
    w, h = int(w), int(h)
    sx, sy = _scale2(scale)
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    for o in range(max(1, int(octaves))):
        _noise_layer_add(acc, amp, sx * 2 ** o, sy * 2 ** o, seed, o, tileable)
        total += amp
        amp *= gain
    acc *= np.float32(1.0 / total)
    if normalize:
        lo, hi = acc.min(), acc.max()
        if hi > lo:
            acc -= lo
            acc *= np.float32(1.0 / (hi - lo))
    return np.clip(acc, 0.0, 1.0, out=acc)


def streaks(w, h, scale=(2.0, 64.0), seed=0, octaves=2, tileable=False, normalize=False):
    """Anisotropic noise for hair / fabric grain -> float32 (h, w) in 0..1 (see `value_noise`).

    scale = (su, sv): lattice cells across the image in u and v. The default (2, 64) varies slowly along u and
    fast along v: long HORIZONTAL streaks, fine across them. For strands that run along v (vertical fibres)
    pass (64, 2). A second octave (default) adds finer breakup."""
    return value_noise(w, h, scale=scale, seed=seed, octaves=octaves, tileable=tileable, normalize=normalize)


# ==================================================================================================== gradients
def _stops(stops):
    items = sorted(((float(t), color(c)) for t, c in stops), key=lambda s: s[0])
    if not items:
        raise ValueError("a gradient needs at least one stop")
    ts = np.array([s[0] for s in items], dtype=np.float64)
    cols = np.stack([s[1] for s in items]).astype(np.float64)
    return ts, cols


def _ramp(t, ts, cols, linear):
    """Evaluate gradient stops at parameters `t` (any shape) -> float32 (..., 4) straight sRGB RGBA. Colours are
    weighted by alpha (premultiplied interpolation); `linear` interpolates the RGB in linear light."""
    t = np.asarray(t, dtype=np.float64)
    flat = t.reshape(-1)
    rgb = cols[:, :3]
    al = cols[:, 3]
    if linear:
        rgb = srgb_to_linear(rgb)
    if np.all(al >= 1.0):
        out = np.stack([np.interp(flat, ts, rgb[:, k]) for k in range(3)], axis=-1)
        a = np.ones_like(flat)
    else:
        pm = rgb * al[:, None]
        a = np.interp(flat, ts, al)
        out = np.stack([np.interp(flat, ts, pm[:, k]) for k in range(3)], axis=-1)
        ok = a > _EPS
        res = out / np.where(ok, a, 1.0)[:, None]
        if not ok.all():
            straight = np.stack([np.interp(flat, ts, rgb[:, k]) for k in range(3)], axis=-1)
            res = np.where(ok[:, None], res, straight)
        out = res
    if linear:
        out = linear_to_srgb(out)
    return np.concatenate([out, a[:, None]], axis=-1).astype(np.float32).reshape(t.shape + (4,))


def _tparam_linear(u, v, p0, p1):
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    l2 = dx * dx + dy * dy
    if l2 <= 0.0:
        raise ValueError("gradient points p0 and p1 coincide")
    return ((u - float(p0[0])) * dx + (v - float(p0[1])) * dy) / l2


def _tparam_radial(u, v, center, radius, aspect, hw):
    """u, v: UV arrays; hw = image height / width (UV-v units -> UV-width units)."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    ex = (u - float(center[0])) / radius
    ey = (v - float(center[1])) * (hw * float(aspect) / radius)
    return np.sqrt(ex * ex + ey * ey)


def linear_gradient(w, h, stops, p0=(0.5, 1.0), p1=(0.5, 0.0), linear=False):
    """Linear gradient image (h, w, 4).

    stops: list of (t, colour); t = 0 at UV point p0 and t = 1 at p1 (clamped outside, stops may be unsorted and
    may lie outside 0..1). Default p0 -> p1 runs from the TOP edge to the BOTTOM edge. The geometry is in UV
    space (on non-square images a diagonal p0 -> p1 is skewed accordingly). `linear=True` interpolates in linear
    light, the default interpolates the sRGB numbers (painter's gradient). Colours are alpha-weighted."""
    w, h = int(w), int(h)
    ts, cols = _stops(stops)
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    if float(p1[0]) == float(p0[0]):                      # vertical: one ramp per row
        col = _ramp(_tparam_linear(0.0, v, p0, p1), ts, cols, linear)
        return np.repeat(col[:, None, :], w, axis=1)
    if float(p1[1]) == float(p0[1]):                      # horizontal: one ramp per column
        col = _ramp(_tparam_linear(u, 0.0, p0, p1), ts, cols, linear)
        return np.repeat(col[None, :, :], h, axis=0)
    return _ramp(_tparam_linear(u[None, :], v[:, None], p0, p1), ts, cols, linear)


def radial_gradient(w, h, stops, center=(0.5, 0.5), radius=0.5, aspect=1.0, linear=False):
    """Radial gradient image (h, w, 4): t = 0 at `center` (UV), t = 1 at distance `radius` (UV-width units, so the
    gradient is circular on non-square images). `aspect` = horizontal radius / vertical radius (> 1 = wider than
    tall, the vertical radius is radius / aspect). Stops and interpolation as in `linear_gradient`."""
    w, h = int(w), int(h)
    ts, cols = _stops(stops)
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    t = _tparam_radial(u[None, :], v[:, None], center, float(radius), aspect, h / w)
    return _ramp(t, ts, cols, linear)


# ============================================================================================ canvas: geometry
def _circle_n(radius, tol=0.02):
    """Vertices of a polygon whose sagitta error is below `tol` samples for a circle of `radius` samples."""
    r = max(float(radius), 1e-3)
    if r <= tol:
        return 8
    n = int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - tol / r))))
    return int(min(max(n, 12), 4096))


def _ring_edges(rings):
    """List of (n, 2) closed rings -> edge arrays (P0, P1)."""
    p0 = np.concatenate(rings, axis=0)
    p1 = np.concatenate([np.roll(r, -1, axis=0) for r in rings], axis=0)
    return p0, p1


def _scan(P0, P1, x0, y0, x1, y1, evenodd=False):
    """Coverage of the closed polygon(s) given as edges P0 -> P1 (sample coordinates, y down) sampled at the sample
    CENTRES of the window [x0, x1) x [y0, y1): boolean (y1 - y0, x1 - x0). Scanline crossings accumulated in a
    difference array (vectorised); nonzero winding (union of consistently oriented pieces) or even-odd."""
    H, W = int(y1 - y0), int(x1 - x0)
    mask = np.zeros((max(H, 0), max(W, 0)), bool)
    if H <= 0 or W <= 0 or len(P0) == 0:
        return mask
    ay, by = P0[:, 1], P1[:, 1]
    keep = ay != by
    if not keep.any():
        return mask
    ax, ay, bx, by = P0[keep, 0], P0[keep, 1], P1[keep, 0], P1[keep, 1]
    swap = by < ay
    sign = np.where(swap, -1.0, 1.0)
    xa, ya = np.where(swap, bx, ax), np.where(swap, by, ay)
    xb, yb = np.where(swap, ax, bx), np.where(swap, ay, by)
    ja = np.maximum(np.ceil(ya - 0.5), y0).astype(np.int64)
    jb = np.minimum(np.ceil(yb - 0.5), y1).astype(np.int64)
    cnt = jb - ja
    ok = (cnt > 0) & (np.minimum(xa, xb) < x1 + 1)
    if not ok.any():
        return mask
    xa, ya, xb, yb, sign, ja, cnt = xa[ok], ya[ok], xb[ok], yb[ok], sign[ok], ja[ok], cnt[ok]
    slope = (xb - xa) / (yb - ya)
    total = int(cnt.sum())
    eidx = np.repeat(np.arange(cnt.size), cnt)
    first = np.cumsum(cnt) - cnt
    j = ja[eidx] + (np.arange(total) - first[eidx])
    x = xa[eidx] + (j + 0.5 - ya[eidx]) * slope[eidx]
    c = np.clip(np.ceil(x - 0.5).astype(np.int64) - x0, 0, W)
    flat = (j - y0) * (W + 1) + c
    diff = np.bincount(flat, weights=sign[eidx], minlength=H * (W + 1)).reshape(H, W + 1)
    cum = np.cumsum(diff[:, :W], axis=1)
    if evenodd:
        return (np.rint(cum).astype(np.int64) & 1).astype(bool)
    return np.abs(cum) > 0.5


def _orient(Q):
    """(k, m, 2) polygons -> the same with a positive shoelace area (so nonzero fill == union)."""
    x, y = Q[..., 0], Q[..., 1]
    area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    neg = area < 0
    if neg.any():
        Q = Q.copy()
        Q[neg] = Q[neg][:, ::-1]
    return Q


def _discs(centers, radii, n):
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    cs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    return centers[:, None, :] + radii[:, None, None] * cs[None, :, :]


def _stroke_pieces(P, r, closed, cap, join, miter_limit=4.0, tol=0.04):
    """Convex pieces ((k, m, 2) arrays, all positively oriented) whose union is the stroke of the polyline P
    (sample coordinates) with per-vertex half widths r (samples). Segments are trapezoids, joins are round wedges /
    bevel triangles / miter quads on the outer side, caps are discs / squares / nothing."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    r = np.maximum(np.asarray(r, dtype=np.float64).reshape(-1), 0.0)
    if len(P) > 1:
        keep = np.ones(len(P), bool)
        keep[1:] = np.hypot(*np.diff(P, axis=0).T) > 1e-9
        P, r = P[keep], r[keep]
        if closed and len(P) > 1 and np.hypot(*(P[0] - P[-1])) <= 1e-9:
            P, r = P[:-1], r[:-1]
    n = len(P)
    if n == 0:
        return []
    if n < 3:
        closed = False
    pieces = []
    nmax = _circle_n(r.max() if n else 0.0)
    if n == 1:
        if cap == "round":
            pieces.append(_discs(P, r, nmax))
        elif cap == "square":
            x, y = P[0]
            rr = r[0]
            pieces.append(np.array([[[x - rr, y - rr], [x + rr, y - rr], [x + rr, y + rr], [x - rr, y + rr]]]))
        return pieces
    if closed:
        A, B, rA, rB = P, np.roll(P, -1, axis=0), r, np.roll(r, -1)
    else:
        A, B, rA, rB = P[:-1], P[1:], r[:-1], r[1:]
    d = B - A
    seg_len = np.hypot(d[:, 0], d[:, 1])
    u = d / seg_len[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], axis=1)
    quad = np.stack([A + nrm * rA[:, None], B + nrm * rB[:, None],
                     B - nrm * rB[:, None], A - nrm * rA[:, None]], axis=1)
    pieces.append(_orient(quad))
    # joins on the outer side of every turn
    if closed:
        V, rV, n1, n2, u1, u2 = P, r, np.roll(nrm, 1, axis=0), nrm, np.roll(u, 1, axis=0), u
    else:
        V, rV, n1, n2, u1, u2 = P[1:-1], r[1:-1], nrm[:-1], nrm[1:], u[:-1], u[1:]
    if len(V):
        cross = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
        dot = np.sum(u1 * u2, axis=1)
        need = ((np.abs(cross) > 1e-4) | (dot < 0)) & (rV > 1e-6)
        if need.any():
            V, rV, n1, n2, cross, dot = V[need], rV[need], n1[need], n2[need], cross[need], dot[need]
            s = np.where(cross >= 0, -1.0, 1.0)
            if join == "round":
                turn = np.arctan2(cross, dot)
                phi = np.arctan2(s * n1[:, 1], s * n1[:, 0])
                step = 2.0 * np.arccos(np.clip(1.0 - tol / np.maximum(rV, tol), -1.0, 1.0))
                m = np.maximum(1, np.ceil(np.abs(turn) / np.maximum(step, 1e-3)).astype(np.int64))
                for mm in np.unique(m):
                    sel = m == mm
                    ang = phi[sel, None] + turn[sel, None] * np.linspace(0.0, 1.0, int(mm) + 1)[None, :]
                    arc = V[sel, None, :] + rV[sel, None, None] * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
                    pieces.append(_orient(np.concatenate([V[sel, None, :], arc], axis=1)))
            else:
                a_out = V + (s * rV)[:, None] * n1
                b_out = V + (s * rV)[:, None] * n2
                tip = b_out
                if join == "miter":
                    den = 1.0 + dot
                    use = (den > 1e-6) & (2.0 / np.maximum(den, 1e-6) <= miter_limit ** 2)
                    miter = V + (s * rV)[:, None] * (n1 + n2) / np.maximum(den, 1e-6)[:, None]
                    tip = np.where(use[:, None], miter, b_out)
                pieces.append(_orient(np.stack([V, a_out, tip, b_out], axis=1)))
    # caps
    if not closed:
        if cap == "round":
            pieces.append(_discs(P[[0, -1]], r[[0, -1]], nmax))
        elif cap == "square":
            for pt, rr, uu, nn in ((P[0], r[0], -u[0], nrm[0]), (P[-1], r[-1], u[-1], nrm[-1])):
                q = np.array([[pt + nn * rr, pt + uu * rr + nn * rr, pt + uu * rr - nn * rr, pt - nn * rr]])
                pieces.append(_orient(q))
    return pieces


def _pieces_edges(pieces):
    p0 = np.concatenate([q.reshape(-1, 2) for q in pieces], axis=0)
    p1 = np.concatenate([np.roll(q, -1, axis=1).reshape(-1, 2) for q in pieces], axis=0)
    return p0, p1


def _rings(points):
    """A ring (n, 2) or a list of rings -> list of float64 (n, 2) arrays."""
    pts = points if isinstance(points, np.ndarray) else list(points)
    if len(pts) and np.ndim(pts[0]) == 2:
        rings = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in pts]
    else:
        rings = [np.asarray(pts, dtype=np.float64).reshape(-1, 2)]
    for r in rings:
        if not np.all(np.isfinite(r)):
            raise ValueError("points must be finite")
    return rings


def _rounded_rect(x0, y0, x1, y1, r, n_arc):
    if r <= 0:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    parts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90.0), (x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0)):
        ang = np.radians(a0 + np.linspace(0.0, 90.0, n_arc + 1))
        parts.append(np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1))
    return np.concatenate(parts, axis=0)


def catmull_rom(points, samples=12, closed=False, alpha=0.5, return_param=False):
    """Centripetal Catmull-Rom spline through `points` (n, 2) -> (m, 2) polyline points (float64).

    `samples` points per span; open curves end exactly on the last control point (end tangents by reflection),
    closed curves wrap (the first point is not repeated at the end). alpha = 0.5 is centripetal (no cusps or
    loops), 0 uniform, 1 chordal. `return_param=True` also returns the control-point index (fractional) of every
    output point, for interpolating per-point widths with np.interp."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(P)
    samples = max(1, int(samples))
    if n < 2:
        return (P.copy(), np.zeros(n)) if return_param else P.copy()
    if closed and n >= 3:
        ext = np.concatenate([P[-1:], P, P[:2]], axis=0)
        nseg = n
    else:
        closed = False
        ext = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]], axis=0)
        nseg = n - 1
    P0, P1, P2, P3 = (ext[i:i + nseg][:, None, :] for i in range(4))

    def knot(a, b):
        return np.maximum(np.hypot(*(b - a)[:, 0, :].T), 1e-9) ** alpha

    t1 = knot(P0, P1)
    t2 = t1 + knot(P1, P2)
    t3 = t2 + knot(P2, P3)
    t1, t2, t3 = t1[:, None, None], t2[:, None, None], t3[:, None, None]
    t0 = np.zeros_like(t1)
    s = np.linspace(0.0, 1.0, samples, endpoint=False)
    tt = (t1 + (t2 - t1) * s[None, :, None])

    def lerp(Pa, Pb, ta, tb):
        return ((tb - tt) * Pa + (tt - ta) * Pb) / (tb - ta)

    A1, A2, A3 = lerp(P0, P1, t0, t1), lerp(P1, P2, t1, t2), lerp(P2, P3, t2, t3)
    B1, B2 = lerp(A1, A2, t0, t2), lerp(A2, A3, t1, t3)
    out = lerp(B1, B2, t1, t2).reshape(-1, 2)
    param = (np.arange(nseg)[:, None] + s[None, :]).reshape(-1)
    if not closed:
        out = np.concatenate([out, P[-1:]], axis=0)
        param = np.concatenate([param, [n - 1.0]])
    return (out, param) if return_param else out


def bezier_points(p0, p1, p2, p3, n=24):
    """Cubic Bezier through p0 .. p3 (2-D points) -> (n + 1, 2) points from p0 to p3 (float64)."""
    P = np.array([p0, p1, p2, p3], dtype=np.float64)
    t = np.linspace(0.0, 1.0, max(1, int(n)) + 1)[:, None]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]


# ============================================================================================= canvas: paints
class _Solid:
    """Constant straight-alpha colour: `chan` = its r, g, b as float32 scalars, `a` = its alpha (float32)."""

    def __init__(self, c):
        c = color(c)
        self.chan = [c[0], c[1], c[2]]
        self.a = c[3]


class _GradPaint:
    """Linear / radial gradient evaluated at sample centres (xs, ys: int arrays of absolute sample indices)."""

    def __init__(self, stops, linear, kind, geom, ws, hs):
        self.ts, self.cols = _stops(stops)
        self.linear, self.kind, self.geom, self.ws, self.hs = linear, kind, geom, ws, hs

    def sample(self, xs, ys):
        u = (xs + 0.5) / self.ws
        v = 1.0 - (ys + 0.5) / self.hs
        if self.kind == "linear":
            t = _tparam_linear(u, v, *self.geom)
        else:
            center, radius, aspect = self.geom
            t = _tparam_radial(u, v, center, radius, aspect, self.hs / self.ws)
        rgba = _ramp(t, self.ts, self.cols, self.linear)
        return [np.ascontiguousarray(rgba[:, k]) for k in range(3)], np.ascontiguousarray(rgba[:, 3])


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError("unknown blend mode %r (use one of %s)" % (mode, ", ".join(MODES)))
    return mode


def _paint_mask(buf, ox, oy, win, mask, paint, mode):
    """Blend `paint` into the premultiplied sample buffer where the boolean `mask` (shaped like `win`) is set.

    Works per channel on strided views: boolean gathers of 1-D channel vectors are several times faster than
    (n, 4) row gathers; opaque 'normal' paint is a plain assignment; a fully covered window skips the mask."""
    x0, y0, x1, y1 = win
    n = int(np.count_nonzero(mask))
    if n == 0:
        return
    planes = [buf[y0 - oy:y1 - oy, x0 - ox:x1 - ox, k] for k in range(4)]
    full = n == mask.size
    if isinstance(paint, _Solid):
        rgb, a = paint.chan, paint.a
        sel = Ellipsis if full else mask
        if mode == "normal" and a >= 1.0:
            for p, v in zip(planes, rgb + [a]):
                p[sel] = v
            return
    else:
        ys, xs = np.nonzero(mask)
        rgb, a = paint.sample(xs + x0, ys + y0)
        sel = mask
    new = _blend_planes([p[sel] for p in planes], rgb, a, mode)
    for p, v in zip(planes, new):
        p[sel] = v


def _paint_full(region, paint, mode):
    """Blend a solid paint over a whole premultiplied (h, w, 4) region (a view into the sample buffer)."""
    planes = [region[..., k] for k in range(4)]
    if mode == "normal" and paint.a >= 1.0:
        for p, v in zip(planes, paint.chan + [paint.a]):
            p[...] = v
        return
    new = _blend_planes(planes, paint.chan, paint.a, mode)
    for p, v in zip(planes, new):
        p[...] = v


# ======================================================================================================= canvas
class Canvas:
    """Antialiased vector-style drawing in UV space onto an RGBA image.

    Everything is drawn at `ss` times the resolution (ss x ss samples per pixel; colour is accumulated
    premultiplied, so transparent edges never fringe) and box-downsampled in `image()`. Drawing calls are
    recorded and rendered tile by tile in `image()`, so memory stays bounded for big canvases (a 4096 px canvas at
    ss = 4 needs no more RAM than a small one); every call returns `self` for chaining.

    Space: (u, v) in 0..1, v UP (a rect near v = 1 lands in the TOP rows); widths, radii and corner radii in UV
    units measured against the image WIDTH, in both directions. Strokes are centred on the outline (half inside,
    half outside). `fill=None` / `stroke=None` skips that part (stroke is drawn over fill). Colours are anything
    `color()` accepts. `mode` ('normal', 'multiply', 'screen', 'add', 'overlay' or 'erase') applies to every
    call. ss = 1 turns antialiasing off; the coverage resolution is 1/ss pixel, so raise `ss` for razor-sharp
    near-horizontal edges.

        cv = Canvas(512, 512, bg="#00000000")
        cv.ellipse((0.5, 0.5), (0.2, 0.12), fill="#ff8800", stroke="#552200", width=0.01, angle=20)
        cv.curve([(0.1, 0.2), (0.4, 0.3), (0.9, 0.2)], 0.01, "#222", widths=[0.002, 0.01, 0.002])
        img = cv.image()            # float32 (512, 512, 4), straight alpha
    """

    TILE = 1024                      # tile edge in SAMPLES (an output tile is TILE // ss pixels)

    def __init__(self, w, h, bg=(0, 0, 0, 0), ss=4):
        """w, h: output size in pixels; bg: initial background colour (its RGB also fills fully transparent output
        pixels, so a transparent bg of your texture's main colour avoids dark fringes); ss: supersampling factor
        per axis (memory per tile is fixed, time grows with ss^2)."""
        self.w, self.h, self.ss = int(w), int(h), int(ss)
        if self.w < 1 or self.h < 1 or self.ss < 1:
            raise ValueError("Canvas needs w, h, ss >= 1")
        self.bg = color(bg)
        self._bg_pm = np.concatenate([self.bg[:3] * self.bg[3], self.bg[3:]]).astype(np.float32)
        self._ws, self._hs = float(self.w * self.ss), float(self.h * self.ss)
        self._ops = []

    # ------------------------------------------------------------------------------------------- plumbing
    def _xy(self, points):
        """UV points (n, 2) -> sample coordinates (x right, y DOWN)."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(p)):
            raise ValueError("points must be finite")
        return np.stack([p[:, 0] * self._ws, (1.0 - p[:, 1]) * self._hs], axis=1)

    def _len(self, x):
        """UV-width length(s) -> samples."""
        return np.asarray(x, dtype=np.float64) * self._ws

    def _push(self, lo, hi, fn):
        x0, y0 = max(int(math.floor(lo[0])), 0), max(int(math.floor(lo[1])), 0)
        x1, y1 = min(int(math.ceil(hi[0])), int(self._ws)), min(int(math.ceil(hi[1])), int(self._hs))
        if x1 > x0 and y1 > y0:
            self._ops.append(((x0, y0, x1, y1), fn))

    def _push_edges(self, p0, p1, paint, mode, evenodd=False):
        if len(p0) == 0:
            return
        lo = np.minimum(p0.min(axis=0), p1.min(axis=0))
        hi = np.maximum(p0.max(axis=0), p1.max(axis=0))

        def draw(buf, ox, oy, win):
            mask = _scan(p0, p1, win[0], win[1], win[2], win[3], evenodd)
            _paint_mask(buf, ox, oy, win, mask, paint, mode)

        self._push(lo, hi, draw)

    def _fill_rings(self, rings, paint, mode, rule):
        if any(len(r) < 3 for r in rings):
            raise ValueError("a filled polygon needs at least 3 points")
        evenodd = (len(rings) > 1) if rule is None else (rule == "evenodd")
        p0, p1 = _ring_edges([self._xy(r) for r in rings])
        self._push_edges(p0, p1, paint, mode, evenodd)

    def _stroke_xy(self, pts, widths, c, closed, cap, join, mode):
        """Stroke the polyline `pts` (sample coordinates) with full widths `widths` (samples)."""
        if cap not in ("round", "butt", "square"):
            raise ValueError("cap must be 'round', 'butt' or 'square'")
        if join not in ("round", "miter", "bevel"):
            raise ValueError("join must be 'round', 'miter' or 'bevel'")
        pieces = _stroke_pieces(pts, widths / 2.0, closed, cap, join)
        if pieces:
            p0, p1 = _pieces_edges(pieces)
            self._push_edges(p0, p1, _Solid(c), mode)

    def _widths(self, n, width, widths):
        if widths is not None:
            wv = self._len(widths).reshape(-1)
            if len(wv) != n:
                raise ValueError("widths needs one value per point (%d), got %d" % (n, len(wv)))
        else:
            wv = np.full(n, float(self._len(width)))
        if not np.all(np.isfinite(wv)):
            raise ValueError("stroke widths must be finite")
        return wv

    # -------------------------------------------------------------------------------------------- drawing
    def fill_all(self, c, mode="normal"):
        """Paint the whole canvas with colour `c` (mode 'erase' clears it)."""
        paint = _Solid(c)
        _check_mode(mode)

        def draw(buf, ox, oy, win):
            _paint_full(buf[win[1] - oy:win[3] - oy, win[0] - ox:win[2] - ox], paint, mode)

        self._push((0, 0), (self._ws, self._hs), draw)
        return self

    def polygon(self, points, fill=None, stroke=None, width=0.004, mode="normal", join="miter", rule=None):
        """Polygon through `points` (n, 2) UV, closed automatically. `fill` colour / `stroke` colour (outline
        centred on the edge, `width` in UV-width units, `join` 'miter' (default, sharp corners; falls back to bevel
        on very acute angles), 'round' or 'bevel'). `points` may also be a list of rings (holes): then the default
        fill rule is 'evenodd'; `rule` forces 'nonzero' or 'evenodd'."""
        _check_mode(mode)
        rings = _rings(points)
        if fill is not None:
            self._fill_rings(rings, _Solid(fill), mode, rule)
        if stroke is not None:
            for r in rings:
                xy = self._xy(r)
                self._stroke_xy(xy, self._widths(len(xy), width, None), stroke, True, "butt", join, mode)
        return self

    def polyline(self, points, width, c, cap="round", join="round", closed=False, widths=None, mode="normal"):
        """Thick line through `points` (n, 2) UV with full `width` (UV-width units) and colour `c`.
        cap: 'round' | 'butt' | 'square'; join: 'round' | 'miter' (falls back to a bevel when the spike would be
        longer than 4 half-widths) | 'bevel'; closed=True joins the last point to the first (no caps). `widths` (one
        value per point) gives a variable width, e.g. a tapered stroke; the width changes linearly along each segment.
        A single point draws a dot (round cap) or nothing (butt). The whole stroke is painted ONCE as a union, so a
        translucent colour or a blend mode does not double up where segments, joins or the path itself overlap."""
        _check_mode(mode)
        xy = self._xy(points)
        self._stroke_xy(xy, self._widths(len(xy), width, widths), c, closed, cap, join, mode)
        return self

    def _ellipse_xy(self, center, radii, angle):
        cx, cy = float(center[0]) * self._ws, (1.0 - float(center[1])) * self._hs
        rx, ry = float(radii[0]) * self._ws, float(radii[1]) * self._ws
        n = _circle_n(max(abs(rx), abs(ry)))
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        ex, ey = rx * np.cos(t), ry * np.sin(t)
        return np.stack([cx + ex * ca - ey * sa, cy - (ex * sa + ey * ca)], axis=1)

    def ellipse(self, center, radii, fill=None, stroke=None, width=0.004, angle=0.0, mode="normal"):
        """Ellipse at `center` (UV) with radii (rx, ry) in UV-width units, rotated `angle` degrees counter-clockwise
        (as seen in the image). Stroke centred on the outline (uniform width, also on thin ellipses)."""
        _check_mode(mode)
        xy = self._ellipse_xy(center, radii, angle)
        if fill is not None:
            p0, p1 = _ring_edges([xy])
            self._push_edges(p0, p1, _Solid(fill), mode)
        if stroke is not None:
            self._stroke_xy(xy, np.full(len(xy), float(self._len(width))), stroke, True, "butt", "round", mode)
        return self

    def circle(self, center, r, fill=None, stroke=None, width=0.004, mode="normal"):
        """Circle at `center` (UV), radius `r` in UV-width units; same as `ellipse((r, r))`."""
        return self.ellipse(center, (r, r), fill=fill, stroke=stroke, width=width, mode=mode)

    def rect(self, u0, v0, u1, v1, fill=None, stroke=None, width=0.004, radius=0.0, mode="normal", join="miter"):
        """Rectangle between the UV corners (u0, v0) and (u1, v1) (any order; v up: v1 > v0 is the upper edge),
        with rounded corners of `radius` (UV-width units, clipped to half the shorter side). The stroke is centred on
        the outline; `join` ('miter' default) matters only for sharp corners (radius = 0)."""
        _check_mode(mode)
        xa, xb = sorted((float(u0) * self._ws, float(u1) * self._ws))
        ya, yb = sorted(((1.0 - float(v0)) * self._hs, (1.0 - float(v1)) * self._hs))
        r = min(max(float(self._len(radius)), 0.0), (xb - xa) / 2.0, (yb - ya) / 2.0)
        xy = _rounded_rect(xa, ya, xb, yb, r, max(2, _circle_n(r) // 4) if r > 0 else 1)
        if fill is not None:
            p0, p1 = _ring_edges([xy])
            self._push_edges(p0, p1, _Solid(fill), mode)
        if stroke is not None:
            self._stroke_xy(xy, np.full(len(xy), float(self._len(width))), stroke, True, "butt",
                            "round" if r > 0 else join, mode)
        return self

    def curve(self, points, width, c, closed=False, samples=12, widths=None, mode="normal", cap="round"):
        """Smooth stroke through `points` (n, 2) UV: a centripetal Catmull-Rom spline drawn as a polyline
        (`samples` points per span). `widths` (one per control point) tapers the stroke smoothly along the curve;
        closed=True makes a closed loop. Two points give a straight line."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if widths is not None and len(widths) != len(pts):
            raise ValueError("widths needs one value per control point (%d), got %d" % (len(pts), len(widths)))
        sm, param = catmull_rom(pts, samples=samples, closed=closed, return_param=True)
        ws = None
        if widths is not None:
            wv = np.asarray(widths, dtype=np.float64)
            if closed and len(pts) >= 3:
                ws = np.interp(param, np.arange(len(pts) + 1), np.concatenate([wv, wv[:1]]))
            else:
                ws = np.interp(param, np.arange(len(pts)), wv)
        return self.polyline(sm, width, c, cap=cap, join="round", closed=closed and len(pts) >= 3, widths=ws, mode=mode)

    def bezier(self, p0, p1, p2, p3, width, c, mode="normal", cap="round"):
        """Cubic Bezier stroke from p0 to p3 with control points p1, p2 (UV). `width` is a number or a pair
        (start, end) for a linear taper (UV-width units)."""
        pts = np.array([p0, p1, p2, p3], dtype=np.float64)
        length_px = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) * self.w
        n = int(min(max(length_px / 3.0, 12), 512))
        sm = bezier_points(p0, p1, p2, p3, n)
        if np.ndim(width) == 0:
            ws = None
            wv = float(width)
        else:
            ws = np.linspace(float(width[0]), float(width[1]), len(sm))
            wv = 0.0
        return self.polyline(sm, wv, c, cap=cap, join="round", widths=ws, mode=mode)

    def fill_gradient(self, polygon_points, stops, p0, p1, mode="normal", linear=False):
        """Fill a polygon (UV points; None = the whole canvas) with a linear gradient: `stops` [(t, colour), ...],
        t = 0 at UV point p0 and 1 at p1, exactly as `linear_gradient`. `linear=True` interpolates in linear light."""
        _check_mode(mode)
        paint = _GradPaint(stops, linear, "linear", (p0, p1), self._ws, self._hs)
        _tparam_linear(0.0, 0.0, p0, p1)
        return self._fill_paint(polygon_points, paint, mode)

    def fill_radial(self, polygon_points, stops, center, radius, aspect=1.0, mode="normal", linear=False):
        """Fill a polygon (UV points; None = the whole canvas) with a radial gradient around `center` (UV) reaching
        t = 1 at `radius` (UV-width units); `aspect` = horizontal / vertical radius, as `radial_gradient`."""
        _check_mode(mode)
        if radius <= 0:
            raise ValueError("radius must be positive")
        paint = _GradPaint(stops, linear, "radial", (center, float(radius), float(aspect)), self._ws, self._hs)
        return self._fill_paint(polygon_points, paint, mode)

    def _fill_paint(self, polygon_points, paint, mode):
        if polygon_points is None:
            polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self._fill_rings(_rings(polygon_points), paint, mode, None)
        return self

    def stamp(self, img, uv_rect, mode="normal"):
        """Composite an RGBA image (any input `to_float` accepts) into the UV rect (u0, v0, u1, v1), resampled with
        Lanczos at the supersampled resolution (alpha-correct). The rect is rounded to 1/ss pixel; `mode` as usual.
        The image is copied, later changes to it do not matter."""
        _check_mode(mode)
        src = to_float(img)
        u0, v0, u1, v1 = (float(x) for x in uv_rect)
        X0, X1 = sorted((int(round(u0 * self._ws)), int(round(u1 * self._ws))))
        Y0, Y1 = sorted((int(round((1.0 - v1) * self._hs)), int(round((1.0 - v0) * self._hs))))
        if X1 <= X0 or Y1 <= Y0:
            return self
        sh, sw = src.shape[:2]
        planes = _planes_pm(src)

        def draw(buf, ox, oy, win):
            x0, y0, x1, y1 = win
            box = ((x0 - X0) * sw / (X1 - X0), (y0 - Y0) * sh / (Y1 - Y0),
                   (x1 - X0) * sw / (X1 - X0), (y1 - Y0) * sh / (Y1 - Y0))
            res = _unpremul(_resample_pm(planes, x1 - x0, y1 - y0, box=box))
            dst = [buf[y0 - oy:y1 - oy, x0 - ox:x1 - ox, k] for k in range(4)]
            new = _blend_planes(dst, [res[..., k] for k in range(3)], res[..., 3], mode)
            for p, v in zip(dst, new):
                p[...] = v

        self._push((X0, Y0), (X1, Y1), draw)
        return self

    def shape_mask(self, points, rule=None):
        """Antialiased coverage of the polygon `points` (UV; or a list of rings) at the OUTPUT resolution:
        float32 (h, w) in 0..1 (sum = area in pixels). For custom per-pixel work, e.g.
        `colorize(cv.shape_mask(poly) * value_noise(w, h), '#fff')` then `stamp` / `composite` it."""
        rings = _rings(points)
        evenodd = (len(rings) > 1) if rule is None else (rule == "evenodd")
        p0, p1 = _ring_edges([self._xy(r) for r in rings])
        ss = self.ss
        lo, hi = np.minimum(p0.min(axis=0), p1.min(axis=0)), np.maximum(p0.max(axis=0), p1.max(axis=0))
        px0, py0 = max(int(math.floor(lo[0] / ss)), 0), max(int(math.floor(lo[1] / ss)), 0)
        px1, py1 = min(int(math.ceil(hi[0] / ss)), self.w), min(int(math.ceil(hi[1] / ss)), self.h)
        out = np.zeros((self.h, self.w), np.float32)
        if px1 > px0 and py1 > py0:
            m = _scan(p0, p1, px0 * ss, py0 * ss, px1 * ss, py1 * ss, evenodd)
            out[py0:py1, px0:px1] = m.reshape(py1 - py0, ss, px1 - px0, ss).mean(axis=(1, 3))
        return out

    # ------------------------------------------------------------------------------------------- output
    def _render_tile(self, tx, ty, tw, th, boxes):
        """Render the output tile at pixel (tx, ty) of size (tw, th): premultiplied samples -> straight-alpha pixels.
        `boxes`: int (n_ops, 4) sample bounding boxes of self._ops, for culling."""
        ss = self.ss
        wx0, wy0, wx1, wy1 = tx * ss, ty * ss, (tx + tw) * ss, (ty + th) * ss
        hit = np.nonzero((boxes[:, 0] < wx1) & (boxes[:, 2] > wx0) & (boxes[:, 1] < wy1) & (boxes[:, 3] > wy0))[0]
        if hit.size == 0:
            tile = np.empty((th, tw, 4), np.float32)
            tile[...] = self.bg
            return tile
        buf = np.empty((th * ss, tw * ss, 4), np.float32)
        buf[...] = self._bg_pm
        for i in hit:
            (bx0, by0, bx1, by1), fn = self._ops[i]
            fn(buf, wx0, wy0, (max(bx0, wx0), max(by0, wy0), min(bx1, wx1), min(by1, wy1)))
        pm = buf.reshape(th, ss, tw * ss * 4).sum(axis=1).reshape(th, tw, ss, 4).sum(axis=2)
        pm *= np.float32(1.0 / (ss * ss))
        return _unpremul(pm, fallback=self.bg[:3])

    def image(self):
        """Render the canvas: a NEW float32 (h, w, 4) straight-alpha image (rows top to bottom). Fully transparent
        pixels get the RGB of `bg`. Nothing is cached: every call renders again (draw more and call again freely)."""
        out = np.empty((self.h, self.w, 4), np.float32)
        boxes = np.array([b for b, _ in self._ops], dtype=np.int64).reshape(-1, 4)
        t = max(1, int(self.TILE) // self.ss)
        for ty in range(0, self.h, t):
            for tx in range(0, self.w, t):
                tw, th = min(t, self.w - tx), min(t, self.h - ty)
                out[ty:ty + th, tx:tx + tw] = self._render_tile(tx, ty, tw, th, boxes)
        return out

    def save(self, path, dither=False):
        """Render and write an RGBA PNG (see `save_png`); returns the Path."""
        return save_png(path, self.image(), dither=dither)


# ==================================================================================================== MMD helpers
def _size2(size):
    if np.ndim(size) == 0:
        return int(size), int(size)
    w, h = size
    return int(w), int(h)


def _row_t(h):
    """Row r of a toon texture -> t = r / (h - 1): the first row is exactly t = 0 (lit), the last exactly t = 1."""
    return np.arange(h, dtype=np.float64) / max(h - 1, 1)


def toon_bands(stops, size=(32, 128)):
    """Toon texture from arbitrary stops -> float32 (h, w, 4), constant along the width.

    stops: [(t, colour), ...] with t = 0 at the TOP row (fully lit side, n.l = 1) down to t = 1 at the BOTTOM row
    (fully shadowed, n.l = -1); colours are interpolated linearly on the sRGB numbers, alpha-weighted. Give two
    stops with the same t for a hard edge. size = (w, h) in pixels (MMD toon textures are sampled by lighting
    only, so the width is irrelevant; 32 x 128 is plenty). Row r is t = r / (h - 1): the top row is exactly stop
    t = 0 and the bottom row exactly t = 1."""
    w, h = _size2(size)
    ts, cols = _stops(stops)
    col = _ramp(_row_t(h), ts, cols, False)
    return np.repeat(col[:, None, :], w, axis=1)


def toon_ramp(shadow, light=(1, 1, 1), threshold=0.5, softness=0.05, deep=None, size=(32, 128), deep_start=None):
    """Cel-shading toon texture -> float32 (h, w, 4), constant along the width.

    An MMD toon texture is sampled by lighting: the TOP row is the fully lit side (n.l = 1) and the BOTTOM row
    the fully shadowed side (n.l = -1). The ramp is `light` above the threshold and `shadow` below it, with a smooth
    (smoothstep) edge. Row r is t = r / (h - 1) (t = 0 top, t = 1 bottom), so as long as the soft edge stays inside
    the texture the top row is exactly `light` and the bottom row exactly `shadow` (or `deep`).

    shadow, light: colours (MMD multiplies the toon colour into the lit colour, so `light` is normally white and
        `shadow` a pale tinted grey / pink / lilac). threshold: where the edge is centred, as the same t as in
        `toon_bands`, a fraction of the height measured from the TOP: t = (1 - n.l) / 2, so 0.5 is the terminator
        (n.l = 0), 0.25 is n.l = 0.5 and a SMALLER value means MORE of the surface in shadow. softness: width of
        the edge as a fraction of the height (0 = a hard one-row step). deep: optional colour that fades in
        (smoothstep) over the lower part of the shadow side, the 'reflected light' / core-shadow feel; it starts at
        `deep_start` (t, default the middle of the shadow area) and is reached exactly at the bottom row.
        size = (w, h) pixels. Colours are mixed on the sRGB numbers."""
    w, h = _size2(size)
    t = _row_t(h)
    edge = smoothstep(threshold - softness / 2.0, threshold + softness / 2.0, t) if softness > 0 \
        else (t >= threshold).astype(np.float64)
    col = mix(light, shadow, edge, linear=False)
    if deep is not None:
        d0 = (threshold + softness / 2.0 + 1.0) / 2.0 if deep_start is None else float(deep_start)
        d1 = max(1.0, d0 + 1e-6)
        col = mix(col, deep, smoothstep(d0, d1, t), linear=False)
    return np.repeat(col[:, None, :], w, axis=1)


def sphere_map(size, fn):
    """Sphere map (MMD environment / highlight texture) from a function of the VIEW-space normal -> float32
    (h, w, 4).

    A surface whose view-space normal is (nx, ny, nz) samples the image at u = 0.5 + 0.5 nx and v (up) = 0.5 + 0.5 ny:
    the top of the image is where normals point UP on screen, the centre is the part facing the camera (nz = 1),
    the rim of the disc is nz = 0. Pixel (col, row) is evaluated at its centre, nx = 2 (col + 0.5) / w - 1 and
    ny = 1 - 2 (row + 0.5) / h. `fn(nx, ny, nz)` receives three float64 arrays (h, w), evaluated on the unit disc
    only (nz = sqrt(1 - nx^2 - ny^2)); outside the disc the value at the nearest rim point is repeated. It returns
    an RGBA / RGB / grey array of shape (h, w, 4|3|1), (h, w) or a single colour (3,) / (4,), values 0..1.
    size: int (square) or (w, h)."""
    w, h = _size2(size)
    cx = 2.0 * ((np.arange(w) + 0.5) / w) - 1.0
    cy = 1.0 - 2.0 * ((np.arange(h) + 0.5) / h)
    nx, ny = np.meshgrid(cx, cy)
    r2 = nx * nx + ny * ny
    scale = np.where(r2 > 1.0, 1.0 / np.sqrt(np.maximum(r2, 1e-12)), 1.0)
    nx, ny = nx * scale, ny * scale
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))
    out = np.asarray(fn(nx, ny, nz))
    if out.shape == (h, w):
        out = out[..., None]
    if out.ndim == 1:
        out = np.broadcast_to(out, (h, w, out.shape[0]))
    if out.ndim != 3 or out.shape[:2] != (h, w):
        raise ValueError("sphere_map: fn must return an array shaped (h, w[, 1|3|4]) or a colour, got %s"
                         % (out.shape,))
    return to_float(out)


def _sphere_layer(size, color_, strength, mode, profile):
    """Sphere map from a 0..1 `profile(nx, ny, nz)`: mode 'add' -> black + colour * profile * strength; mode 'mul' ->
    white fading to the colour by profile * strength (white = no change)."""
    if mode not in ("add", "mul"):
        raise ValueError("mode must be 'add' or 'mul'")
    col = color(color_)
    k = float(strength) * float(col[3])

    def fn(nx, ny, nz):
        amount = np.clip(profile(nx, ny, nz), 0.0, 1.0)[..., None] * k
        base = 0.0 if mode == "add" else 1.0
        rgb = base + (col[:3] - base) * amount
        return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)

    return sphere_map(size, fn)


def sphere_highlight(size=256, band=(0.35, 0.12), color=(1, 1, 1, 1), strength=1.0, mode="add", side_fade=0.0):
    """Soft horizontal 'angel ring' band sphere map -> float32 (size, size, 4), alpha 1.

    band = (centre, half_width) in view-space ny (+1 = top of the sphere): the band peaks at ny = centre, falls to
    zero at +-half_width with a cosine profile, and is uniform along nx (`side_fade` > 0 dims it towards the left /
    right limb: profile *= 1 - side_fade * nx^2). The peak row of the image is
    row = (1 - centre) / 2 * size from the top. `strength` scales the peak (and the colour's alpha does too).
    Blend modes (the sphere mode of the PMX material):
        mode='add': MMD ADDS the map to the lit colour (spa). The background is BLACK (adds nothing) and the band
                    is `color`: a highlight. This is the one for angel rings and specular glints.
        mode='mul': MMD MULTIPLIES the map into the lit colour (sph). The background is WHITE (no change) and the
                    band blends to `color` by strength: it can only tint or darken, never lighten, so pass a
                    darker colour (a white `color` gives a map that changes nothing)."""
    c0, hw = float(band[0]), max(float(band[1]), 1e-6)

    def profile(nx, ny, nz):
        d = np.clip(np.abs(ny - c0) / hw, 0.0, 1.0)
        p = 0.5 * (1.0 + np.cos(math.pi * d))
        return p * (1.0 - float(side_fade) * nx * nx) if side_fade else p

    return _sphere_layer(size, color, strength, mode, profile)


def sphere_rim(size=256, color=(1, 1, 1, 1), power=3.0, strength=0.6, mode="add"):
    """Rim-light sphere map -> float32 (size, size, 4), alpha 1: the effect grows where the normal turns away from
    the camera, profile = (1 - nz)^power (0 in the centre, 1 on the silhouette ring). `power` sharpens the rim.
    mode='add': black background plus `color` * profile * strength (a glow on the edges, the usual use);
    mode='mul': white background blending to `color` towards the rim (an edge tint / darkening, `color` should be
    darker than white). Both as in `sphere_highlight`."""
    p = float(power)
    return _sphere_layer(size, color, strength, mode, lambda nx, ny, nz: (1.0 - nz) ** p)


def sphere_flat(size=64, color=(1, 1, 1, 1)):
    """Uniform sphere map of a single colour (float32 (size, size, 4), alpha as given): a neutral map (white for
    'mul', black for 'add' changes nothing) or a constant tint / lift for a material."""
    w, h = _size2(size)
    return new(w, h, color)


# ===================================================================================================== atlas
class Atlas:
    """Shelf-packed texture atlas: allocate UV rectangles for islands, blit each island's texture into its slot.

        atlas = Atlas(2048, 2048, padding=2)
        r_face = atlas.alloc(1024, 1024, "face")        # (u0, v0, u1, v1), v up
        atlas.blit("face", face_img)
        uv_atlas = remap_uv(uv_local, r_face)           # or atlas.map_uv("face", uv_local)
        png = atlas.image()

    Slots are placed left to right on shelves from the TOP of the image downwards (online first-fit: allocate
    the biggest slots first for the tightest packing). Every slot owns `padding` pixels of gutter on every side
    (so two slots are 2 * padding pixels apart, and a slot never touches the atlas edge); `blit` fills the gutter
    by replicating the slot's edge pixels, which keeps bilinear filtering and mip-maps from bleeding between
    islands. The returned rect is the slot's content area, without the gutter."""

    def __init__(self, w, h, padding=2):
        """w, h: atlas size in pixels; padding: gutter pixels around every slot (0 packs slots edge to edge).
        `rects` maps slot names to their UV rects; `img` is the live float32 (h, w, 4) atlas (transparent black until
        blitted); `image()` returns a copy."""
        self.w, self.h, self.padding = int(w), int(h), int(padding)
        if self.w < 1 or self.h < 1 or self.padding < 0:
            raise ValueError("Atlas needs w, h >= 1 and padding >= 0")
        self.rects = {}
        self.img = np.zeros((self.h, self.w, 4), np.float32)    # the live atlas image (transparent black)
        self._shelves = []                                      # [y, height, x_cursor] in pixels incl. gutters

    def alloc(self, w_px, h_px, name=None):
        """Reserve a w_px x h_px slot (plus the gutter) and return its UV rect (u0, v0, u1, v1) with v up. A `name`
        registers it in `rects`. Raises ValueError when the slot cannot fit (atlas full, or bigger than the
        atlas) or the name is taken."""
        w_px, h_px, p = int(w_px), int(h_px), self.padding
        if w_px < 1 or h_px < 1:
            raise ValueError("slot sizes must be >= 1 pixel")
        if name is not None and name in self.rects:
            raise ValueError("atlas slot %r already exists" % (name,))
        fw, fh = w_px + 2 * p, h_px + 2 * p
        if fw > self.w or fh > self.h:
            raise ValueError("slot %dx%d (+%d px gutter) is bigger than the %dx%d atlas"
                             % (w_px, h_px, p, self.w, self.h))
        for sh in self._shelves:
            if fh <= sh[1] and sh[2] + fw <= self.w:
                x, y = sh[2], sh[0]
                sh[2] += fw
                break
        else:
            last = self._shelves[-1] if self._shelves else None
            if last is not None and fh > last[1] and last[0] + fh <= self.h and last[2] + fw <= self.w:
                last[1] = fh                                    # the open shelf grows to the taller slot
                x, y = last[2], last[0]
                last[2] += fw
            else:
                y = last[0] + last[1] if last is not None else 0
                if y + fh > self.h:
                    raise ValueError("atlas %dx%d is full (no room for %dx%d)" % (self.w, self.h, w_px, h_px))
                self._shelves.append([y, fh, fw])
                x = 0
        x0, y0 = x + p, y + p
        rect = (x0 / self.w, 1.0 - (y0 + h_px) / self.h, (x0 + w_px) / self.w, 1.0 - y0 / self.h)
        if name is not None:
            self.rects[name] = rect
        return rect

    def _rect(self, name_or_rect):
        if isinstance(name_or_rect, str):
            return self.rects[name_or_rect]
        return tuple(float(x) for x in name_or_rect)

    def blit(self, name_or_rect, img, mode="replace"):
        """Write `img` into a slot (a name from `rects`, or any UV rect). An image of another size is resampled
        (Lanczos, alpha-correct) to the slot's pixel size. mode 'replace' overwrites the slot; any `composite` mode
        composites over what is there. The gutter around the slot is refilled from the slot's edge pixels.
        Returns the rect."""
        rect = self._rect(name_or_rect)
        x0, y0, x1, y1 = uv_rect_pixels(rect, self.w, self.h)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x1 <= x0 or y1 <= y0:
            return rect
        src = to_float(img)
        if src.shape[:2] != (y1 - y0, x1 - x0):
            src = resize(src, x1 - x0, y1 - y0)
        region = self.img[y0:y1, x0:x1]
        region[...] = src if mode == "replace" else composite(region, src, mode)
        p = self.padding
        if p > 0:
            gx0, gy0, gx1, gy1 = max(0, x0 - p), max(0, y0 - p), min(self.w, x1 + p), min(self.h, y1 + p)
            self.img[gy0:gy1, gx0:gx1] = np.pad(
                self.img[y0:y1, x0:x1], ((y0 - gy0, gy1 - y1), (x0 - gx0, gx1 - x1), (0, 0)), mode="edge")
        return rect

    def map_uv(self, name_or_rect, uv):
        """Map local UVs (0..1 over an island, array (..., 2)) into a slot's UV rect (see `remap_uv`)."""
        return remap_uv(uv, self._rect(name_or_rect))

    def image(self):
        """A NEW float32 (h, w, 4) copy of the atlas (use `.img` for the live array)."""
        return self.img.copy()
