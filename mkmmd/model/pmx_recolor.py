"""Colour edits for the textures of an imported head (numpy only): hue, saturation and value remaps over a rectangle of a
texture, so that an iris, lashes or a tinted patch can take the character's colours while the painting (gradients,
highlights, shading) stays as it is.

A rule is a dict (from the spec, `[[head.pmx.recolor]]`):
  texture   source path of the texture (used by head_pmx to pick the rules of a file)
  rect      [u0, v0, u1, v1] fractions of the image, v from the top row (the PMX convention); default the whole image
  hue       [[src_deg, dst_deg], ...] piecewise-linear hue map on the colour circle (sorted by src; wraps around)
  sat       scalar multiplier, or [[src, dst], ...] curve over saturation 0..1
  val       scalar multiplier, or [[src, dst], ...] curve over value 0..1
  protect   [s0, s1]: pixels with saturation below s0 stay as they are, fully edited from s1 up (keeps white highlights)
  feather   pixels: soft edge of the rectangle (default 0)
Rules apply in order; the colour ops never touch alpha, pixels with alpha 0 are skipped. Two more ops reshape a painted shape
(all centres and radii are fractions of the image, v from the top):
  warp      {centre = [u, v], radius = [ru, rv], sx = 0.8, sy = 1.0}   scale the picture about `centre` by (sx, sy) inside the
            ellipse of radius (ru, rv), fading to nothing at its rim (the neighbours flow in): narrows a highlight or a pupil
  almond    {centre = [u, v], half = [hu, hv], soft = 0.3, tip = 1.7}   multiply alpha by a soft vertical almond (a lens that
            ends in points: half width hu (1 - |dy / hv|^tip) at the height dy from the centre); what lies under it shows
            through outside. Applied after `warp`; both run after the colour ops of the same rule."""
import numpy as np


def rgb_to_hsv(rgb):
    """(..., 3) float 0..1 -> h degrees [0, 360), s, v (each (...,))."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    safe = np.where(nz, d, 1.0)
    h = np.where(nz & (mx == r), ((g - b) / safe) % 6.0, h)
    h = np.where(nz & (mx == g) & (mx != r), (b - r) / safe + 2.0, h)
    h = np.where(nz & (mx == b) & (mx != r) & (mx != g), (r - g) / safe + 4.0, h)
    s = np.where(mx > 1e-9, d / np.where(mx > 1e-9, mx, 1.0), 0.0)
    return (h * 60.0) % 360.0, s, mx


def hsv_to_rgb(h, s, v):
    h = (np.asarray(h, float) % 360.0) / 60.0
    c = v * s
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))
    z = np.zeros_like(c)
    k = np.floor(h).astype(int) % 6
    r = np.choose(k, [c, x, z, z, x, c])
    g = np.choose(k, [x, c, c, x, z, z])
    b = np.choose(k, [z, z, x, c, c, x])
    m = v - c
    return np.stack([r + m, g + m, b + m], -1)


def hue_map(h, pairs):
    """Piecewise-linear map of hue degrees on the circle through the (src, dst) pairs (src ascending in [0, 360))."""
    src = np.array([p[0] for p in pairs], float)
    dst = np.array([p[1] for p in pairs], float)
    o = np.argsort(src)
    src, dst = src[o], dst[o]
    dst = np.unwrap(np.radians(dst)) * 180.0 / np.pi           # the target may cross 360: keep it continuous
    src3 = np.concatenate([src - 360.0, src, src + 360.0])
    dst3 = np.concatenate([dst - 360.0, dst, dst + 360.0])
    return np.interp(np.asarray(h, float), src3, dst3) % 360.0


def curve(x, spec):
    """A scalar multiplier or a piecewise-linear curve [[src, dst], ...] applied to x."""
    if np.isscalar(spec):
        return np.asarray(x) * float(spec)
    pts = sorted((float(a), float(b)) for a, b in spec)
    return np.interp(x, [p[0] for p in pts], [p[1] for p in pts])


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rect(rule, shape):
    h, w = shape[:2]
    u0, v0, u1, v1 = rule.get("rect", [0.0, 0.0, 1.0, 1.0])
    return int(round(v0 * h)), int(round(v1 * h)), int(round(u0 * w)), int(round(u1 * w))


def _bilinear(img, sx, sy):
    """Sample img (h, w, 4) at float pixel coordinates (sx, sy) with edge clamping."""
    h, w = img.shape[:2]
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    f = img.astype(float)
    top = f[y0, x0] * (1 - fx) + f[y0, x1] * fx
    bot = f[y1, x0] * (1 - fx) + f[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _warp(out, spec):
    """Scale the picture about a centre inside an ellipse (see the module docstring); returns the new image."""
    h, w = out.shape[:2]
    cu, cv = spec["centre"]
    ru, rv = spec["radius"]
    sx, sy = float(spec.get("sx", 1.0)), float(spec.get("sy", 1.0))
    cx, cy = cu * w, cv * h
    rx, ry = max(ru * w, 1.0), max(rv * h, 1.0)
    x0, x1 = max(int(cx - rx) - 1, 0), min(int(cx + rx) + 2, w)
    y0, y1 = max(int(cy - ry) - 1, 0), min(int(cy + ry) + 2, h)
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.0, np.arange(y0, y1) + 0.0)
    rho = np.hypot((gx - cx) / rx, (gy - cy) / ry)
    k = 1.0 - _smooth(rho)                                     # 1 at the centre, 0 at the rim
    scx = 1.0 + (sx - 1.0) * k                                  # the local scale of the picture
    scy = 1.0 + (sy - 1.0) * k
    src = _bilinear(out, cx + (gx - cx) / scx, cy + (gy - cy) / scy)
    res = np.array(out, copy=True)
    res[y0:y1, x0:x1] = np.clip(src + 0.5, 0, 255).astype(np.uint8)
    return res


def _almond(out, spec, rect=None):
    """Multiply alpha by a soft vertical almond (see the module docstring); only inside `rect` when it is given."""
    h, w = out.shape[:2]
    cu, cv = spec["centre"]
    hu, hv = spec["half"]
    soft = float(spec.get("soft", 0.3))
    tip = float(spec.get("tip", 1.7))
    gx, gy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    u = np.abs((gy - cv * h) / max(hv * h, 1.0))
    half = np.maximum(hu * w * (1.0 - np.minimum(u, 1.0) ** tip), 1e-3)
    rho = np.abs(gx - cu * w) / half
    rho = np.where(u >= 1.0, 2.0, rho)
    m = 1.0 - _smooth((rho - (1.0 - soft)) / max(soft, 1e-3))
    if rect is not None:
        y0, y1, x0, x1 = _rect({"rect": rect}, out.shape)
        keep = np.ones((h, w), bool)
        keep[y0:y1, x0:x1] = False
        m = np.where(keep, 1.0, m)
    res = np.array(out, copy=True)
    res[..., 3] = np.clip(res[..., 3] * m + 0.5, 0, 255).astype(np.uint8)
    return res


def recolor(img, rules):
    """Apply the rules to a copy of the RGBA uint8 image (h, w, 4); returns the edited copy."""
    out = np.array(img, copy=True)
    for r in rules:
        out = _recolor_region(out, r)
        if "warp" in r:
            out = _warp(out, r["warp"])
        if "almond" in r:
            out = _almond(out, r["almond"], r.get("rect"))
    return out


def _recolor_region(out, r):
    """The colour ops of one rule (in place on `out`, which is returned)."""
    if not any(k in r for k in ("hue", "sat", "val")):
        return out
    y0, y1, x0, x1 = _rect(r, out.shape)
    reg = out[y0:y1, x0:x1]
    if reg.size == 0:
        return out
    rgb = reg[..., :3].astype(float) / 255.0
    alpha = reg[..., 3]
    h, s, v = rgb_to_hsv(rgb)
    h2, s2, v2 = h, s, v
    if "hue" in r:
        h2 = hue_map(h, r["hue"])
    if "sat" in r:
        s2 = np.clip(curve(s, r["sat"]), 0.0, 1.0)
    if "val" in r:
        v2 = np.clip(curve(v, r["val"]), 0.0, 1.0)
    new = hsv_to_rgb(h2, s2, v2)
    w = (alpha > 0).astype(float)
    if "protect" in r:
        s0, s1 = r["protect"]
        w = w * _smooth((s - s0) / max(s1 - s0, 1e-6))
    f = float(r.get("feather", 0.0))
    if f > 0:
        hh, ww = reg.shape[:2]
        yy = np.minimum(np.arange(hh), hh - 1 - np.arange(hh))[:, None]
        xx = np.minimum(np.arange(ww), ww - 1 - np.arange(ww))[None, :]
        w = w * _smooth(np.minimum(yy, xx) / f)
    mixed = rgb * (1.0 - w[..., None]) + new * w[..., None]
    reg[..., :3] = np.clip(mixed * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
