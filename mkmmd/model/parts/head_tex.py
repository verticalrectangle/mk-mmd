"""Generated textures of the head part (pure numpy, colours from the spec): iris gradient, sclera, blush, flat colour
tiles and the white sphere map the highlights use to stay bright in shadow."""
import numpy as np


# colour defaults (the spec's [colors.eyes] overrides them)
EYE_DEFAULTS = dict(sclera="#f6f1f2", sclera_shadow="#cfc8d6", iris_top="#2e050c", iris_upper="#7a0c1c", iris_mid="#c1121f",
                    iris_lower="#ff4a3d", iris_rim="#ffc2a0", iris_ring="#8c0f1d", highlight_glow="#ffd0c0")


def hex_rgb(h):
    h = str(h).lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ramp(stops, t):
    """Piecewise-linear colour ramp: stops [(position, rgb)], t array -> (..., 3)."""
    pos = np.array([p for p, _ in stops])
    col = np.array([c for _, c in stops])
    return np.stack([np.interp(t, pos, col[:, k]) for k in range(3)], -1)


def flat(rgb, size=8):
    """A uniform opaque tile."""
    a = np.empty((size, size, 4), np.uint8)
    a[..., :3] = np.clip(np.asarray(rgb) * 255 + 0.5, 0, 255).astype(np.uint8)
    a[..., 3] = 255
    return a


def white_sphere(size=16):
    return flat((1.0, 1.0, 1.0), size)


def iris(colors, size=256, seed=3):
    """The iris: a vertical gradient (dark under the lid, crimson in the middle, a coral glow and a pale rim at the bottom),
    a dark limbal ring, a darker ring around the pupil and faint radial fibres. The image is the iris's bounding box
    (row 0 = top); the iris is the inscribed ellipse."""
    c = {k: hex_rgb(v) for k, v in {**EYE_DEFAULTS, **colors}.items()}
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size * 2 - 1                      # -1..1 left to right
    y = 1 - (yy + 0.5) / size * 2                      # +1 top .. -1 bottom
    r = np.hypot(x, y)
    t = (1 - y) / 2                                    # 0 top .. 1 bottom
    base = _ramp([(0.00, c["iris_top"]), (0.26, c["iris_upper"]), (0.55, c["iris_mid"]), (0.82, c["iris_lower"]),
                  (1.00, c["iris_rim"])], t)
    img = base
    # dark limbal ring at the edge
    limb = _smooth((r - 0.80) / 0.20)
    img = img * (1 - 0.55 * limb[..., None]) + c["iris_top"] * (0.55 * limb[..., None])
    # darker ring around the pupil
    ring = np.exp(-(((r - 0.40) / 0.10) ** 2))
    img = img * (1 - 0.55 * ring[..., None]) + c["iris_ring"] * (0.55 * ring[..., None])
    # soft glow crescent along the lower inside of the rim
    ang = np.arctan2(y, x)
    low = np.exp(-(((ang + np.pi / 2) / 0.9) ** 2))
    glow = np.exp(-(((r - 0.66) / 0.17) ** 2)) * low
    img = img * (1 - 0.55 * glow[..., None]) + c["highlight_glow"] * (0.55 * glow[..., None])
    img = img * (1.0 - 0.58 * _smooth((y - 0.05) / 0.60))[..., None]                 # the lash shadow: the upper third is darker
    cres = _smooth((r - 0.50) / 0.05) * (1 - _smooth((r - 0.88) / 0.06)) * np.exp(-(((ang + np.pi / 2) / 0.75) ** 2))
    img = img * (1 - 0.65 * cres[..., None]) + c["iris_rim"] * (0.65 * cres[..., None])      # a brighter lower crescent
    # faint radial fibres
    rng = np.random.default_rng(seed)
    n = 36
    phase = rng.uniform(0, 2 * np.pi, n)
    amp = rng.uniform(0.3, 1.0, n)
    fib = np.zeros_like(r)
    for k in range(n):
        a = 2 * np.pi * k / n + phase[k] * 0.02
        d = np.angle(np.exp(1j * (ang - a)))
        fib += amp[k] * np.exp(-((d / 0.035) ** 2))
    fib = np.clip(fib, 0, 1) * _smooth((r - 0.30) / 0.15) * (1 - _smooth((r - 0.92) / 0.08))
    img = img * (1 - 0.14 * fib[..., None]) + c["iris_top"] * (0.14 * fib[..., None])
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def sclera(colors, frame, size=128, skin=None, shade=0.22, shade_k=0.62):
    """Eyeball white painted in eye coordinates (`frame` from head_eye.sclera_frame): the lash's shadow hangs under the upper
    lid margin (`shade` of the opening height, soft lower edge), the corners are a little darker, and the lower edge warms
    toward the skin so the sclera meets it softly."""
    c = {k: hex_rgb(v) for k, v in {**EYE_DEFAULTS, **colors}.items()}
    r, cc = np.mgrid[0:size, 0:size]
    u = frame["umin"] + (cc + 0.5) / size * (frame["umax"] - frame["umin"])
    v = frame["vmin"] + (1.0 - (r + 0.5) / size) * (frame["vmax"] - frame["vmin"])
    top, bot = frame["fu"](u), frame["fl"](u)
    h = np.maximum(top - bot, 1e-4)
    t = (top - v) / h                                  # 0 on the upper margin .. 1 on the lower margin
    cap = shade_k * (1.0 - _smooth((t - 0.03) / max(shade, 0.05)))
    uc = 0.5 * (frame["u_in"] + frame["u_out"])
    side = 0.30 * _smooth((np.abs(u - uc) / (0.5 * (frame["u_out"] - frame["u_in"])) - 0.55) / 0.45)
    k = np.clip(np.maximum(cap, 0.0) + side, 0, 1)[..., None]
    img = c["sclera"] * (1 - k) + c["sclera_shadow"] * k
    if skin is not None:
        low = (0.55 * _smooth((t - 0.80) / 0.28))[..., None]
        img = img * (1 - low) + hex_rgb(skin) * low
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def blush(color, size=128):
    """Soft round cheek blush: colour with an alpha that fades to nothing at the edge (uv = bounding square)."""
    col = hex_rgb(color)
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size * 2 - 1
    y = 1 - (yy + 0.5) / size * 2
    r = np.hypot(x, y)
    a = 1.0 - _smooth((r - 0.35) / 0.65)             # a plateau to r = 0.35, then a soft fall to nothing at the edge
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def toon(mult, width=32, height=128, edge=0.30, soft=0.45):
    """A very soft face toon ramp (rows from the lit white at the top to `mult` per channel at the bottom, as MMD reads
    it): the face stays almost flat-lit."""
    v = np.linspace(1.0, 0.0, height)
    t = _smooth((v - (edge - soft / 2)) / soft)
    col = np.asarray(mult, float)[None] * (1 - t[:, None]) + t[:, None]
    img = np.repeat(np.clip(col * 255 + 0.5, 0, 255)[:, None, :], width, axis=1)
    return np.dstack([img, np.full((height, width), 255.0)]).astype(np.uint8)


# cylindrical face map: u = 0.5 + azimuth / 2 pi (0 = straight ahead, the seam is at the back), v = (z - UV_Z0) / (UV_Z1 - UV_Z0)
UV_Z0, UV_Z1 = -0.045, 0.21


def face_skin(base, cheek="#f09a96", nose="#d79a8c", w=512, h=256, cheek_k=0.34, nose_k=0.40, jaw=None, band="#e3aa95",
              band_w=(0.9, 1.0, 0.35)):
    """The face skin texture on the cylindrical map: the base colour with a soft warm blush on each cheek and a faint
    warm shade under the nose tip (permanent; the 照れ morph adds a stronger overlay). Row 0 is the top (v = 1)."""
    b, ck, nk = hex_rgb(base), hex_rgb(cheek), hex_rgb(nose)
    r, c = np.mgrid[0:h, 0:w]
    th = ((c + 0.5) / w - 0.5) * 2 * np.pi
    z = UV_Z0 + (1.0 - (r + 0.5) / h) * (UV_Z1 - UV_Z0)

    def blob(th0, z0, sth, sz):
        return np.exp(-0.5 * (((th - th0) / sth) ** 2 + ((z - z0) / sz) ** 2))
    a = np.maximum(blob(0.50, 0.016, 0.14, 0.0085), blob(-0.50, 0.016, 0.14, 0.0085))
    img = np.empty((h, w, 3))
    img[:] = b
    img = img * (1 - cheek_k * a[..., None]) + ck * (cheek_k * a[..., None])
    a2 = blob(0.0, 0.0258, 0.030, 0.0032)
    img = img * (1 - nose_k * a2[..., None]) + nk * (nose_k * a2[..., None])
    if jaw is not None:
        # the neck under the jaw is in shadow: a crisp top edge along the jaw silhouette, then the weight the body's baked neck
        # shadow has at its seam ring (strength x (back + (front - back) (0.5 + 0.5 cos theta)^1.5)), held down to the ring
        d = np.asarray(jaw, float)[None, :] - z
        strength, front, back = band_w
        wcol = strength * (back + (front - back) * (0.5 + 0.5 * np.cos(th[0:1, :])) ** 1.5)
        a3 = np.where(d > 0, wcol * _smooth(d / 0.0010), 0.0)
        img = img * (1 - a3[..., None]) + hex_rgb(band) * a3[..., None]
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out

