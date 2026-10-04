"""Textures of the cat ears and tails (numpy + PIL): black fur with a faint cool sheen, the dark-red -> pink inner ear with
soft fur lines, pale tufts. Every function returns an (h, w, 4) uint8 sRGB image, v UP (row 0 = v 1).

Layouts (matching the UVs of `cat_ear_geo` and `tails`):
  ear fur, ear inner   u across the ear (0 = inner edge .. 1 = outer edge), v = (t + 0.5) / 1.5 with t the height above
                       the hair hull as a fraction of the visible height: v = 1/3 at the hull, 1 at the tip
  tuft                 u across the lock, v = 0 at the root .. 1 at the tip
  tail fur             u = 0 along the sheen line on top of the tail .. 1 along the underside (mirrored round), v = 0 at the
                       root .. 1 at the tip"""
import numpy as np
from PIL import Image, ImageDraw


def _grad(t, stops):
    pos = np.array([p for p, _ in stops], float)
    cols = np.array([c for _, c in stops], float)
    t = np.asarray(t, float)
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mix(a, b, k):
    k = np.asarray(k, float)[..., None] if np.ndim(k) else k
    return np.asarray(a, float) * (1.0 - k) + np.asarray(b, float) * k


def _noise1(n, rng, scale):
    k = max(int(n / scale), 2)
    pts = rng.uniform(-1, 1, k + 3)
    x = np.linspace(0, k, n, endpoint=False)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def _fibres(rng, w, h, strength=1.0):
    """(h, w) multiplicative fibre pattern ~1: thin streaks running along v (fur lying from root to tip)."""
    coarse = _noise1(w, rng, 14.0)
    fine = rng.uniform(-1, 1, w)
    drift = 1.0 + 0.5 * _noise1(h, rng, 60.0)[:, None] * 0.12
    base = 1.0 + strength * (0.05 * coarse + 0.04 * fine)
    return base[None, :] * drift


def _to_u8(img):
    out = np.empty(img.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.rint(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def _strokes(img, rng, n, fn, width, color, alpha, ss=4):
    """Soft antialiased strokes of one colour over `img` (h, w, 3 float): `fn(rng)` -> polyline points (u, v) in 0..1 (v UP);
    each stroke has an opacity drawn from `alpha` = (lo, hi). Drawn as a mask at `ss` x resolution, box filtered."""
    h, w = img.shape[:2]
    big = Image.new("L", (w * ss, h * ss), 0)
    dr = ImageDraw.Draw(big)
    for _ in range(n):
        xy = [(u * w * ss, (1.0 - v) * h * ss) for u, v in fn(rng)]
        a = float(rng.uniform(*alpha))
        dr.line(xy, fill=int(round(255 * a)), width=max(1, int(round(width * ss))), joint="curve")
    m = np.asarray(big.resize((w, h), Image.BOX), float)[..., None] / 255.0
    return img * (1.0 - m) + np.asarray(color, float)[None, None, :] * m


# ---------------------------------------------------------------- ears
def ear_fur(pal, rng, w=128, h=256):
    """Black fur of the ear shell and rim: warm black, darker toward the root (hidden in the hair), a soft cool sheen
    band across the upper half, lighter fur at the very tip and along both edges."""
    v = 1.0 - (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    base, sheen = pal["outer"], _mix(pal["outer_sheen"], pal["cool"], 0.55)
    col = _grad(v, [(0.0, base * 0.55), (0.30, base * 0.78), (0.42, base), (0.56, base), (1.0, base)])
    band = np.exp(-(((v - 0.70) / 0.10) ** 2)) * 0.62 + np.exp(-(((v - 0.84) / 0.045) ** 2)) * 0.22
    tip = _smooth((v - 0.90) / 0.10) * 0.35
    k = np.clip(band + tip, 0.0, 0.85)
    img = col[:, None, :] * (1 - k)[:, None, None] + sheen[None, None, :] * k[:, None, None]
    edge = np.clip((np.abs(2 * u - 1) - 0.62) / 0.38, 0.0, 1.0)
    img = img + (sheen * 0.30)[None, None, :] * (edge ** 1.5)[None, :, None] * (0.35 + 0.65 * _smooth((v - 0.2) / 0.6))[:, None, None]
    img = img * _fibres(rng, w, h)[..., None]
    return _to_u8(img)


def ear_inner(pal, rng, w=128, h=256):
    """Inside of the ear: dark red at the root through the base red to a soft rose pink toward the tip, darker toward the
    walls of the hollow (the rim), with a few soft fur lines sweeping up and outward from the base."""
    v = 1.0 - (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    deep, mid = pal["inner_deep"], pal["inner"]
    pink = np.array([0.78, 0.32, 0.42])
    hi = np.array([0.84, 0.42, 0.50])
    col = _grad(v, [(0.0, deep * 0.75), (0.30, deep), (0.46, mid), (0.70, _mix(mid, pink, 0.6)), (0.88, pink), (1.0, hi)])
    img = np.repeat(col[:, None, :], w, axis=1)
    x = np.abs(2 * u - 1)
    wall = _smooth((x - 0.40) / 0.45)
    img = img * (1.0 - 0.30 * wall)[None, :, None] + (deep * 0.7)[None, None, :] * (0.30 * wall)[None, :, None]
    lip = _smooth((x - 0.62) / 0.16)
    img = img * (1.0 - 0.40 * lip)[None, :, None]
    glow = np.exp(-((x / 0.38) ** 2))[None, :] * np.exp(-(((v - 0.62) / 0.24) ** 2))[:, None]
    img = img + np.array([0.06, 0.03, 0.035])[None, None, :] * glow[..., None]

    def fur(rng):
        u0 = rng.uniform(0.2, 0.8)
        v0 = rng.uniform(0.30, 0.70)
        ln = rng.uniform(0.18, 0.34)
        lean = (u0 - 0.5) * rng.uniform(0.3, 0.7)
        curve = rng.normal(0.0, 0.05)
        return [(u0 + lean * s * ln + curve * s * s * ln, v0 + ln * s) for s in np.linspace(0, 1, 7)]
    img = _strokes(img, rng, 26, fur, 1.2, deep * 0.9, (0.06, 0.12))
    img = _strokes(img, rng, 18, fur, 1.1, _mix(mid, pink, 0.55), (0.07, 0.13))
    img = img * _fibres(rng, w, h, 0.5)[..., None]
    return _to_u8(img)


def tuft(pal, rng, w=64, h=128):
    """Pale cream fur lock: warm shade at the root rising to the cream tip."""
    v = 1.0 - (np.arange(h) + 0.5) / h
    root = _mix(pal["tuft_shadow"], pal["inner"], 0.18)
    col = _grad(v, [(0.0, root * 0.9), (0.22, pal["tuft_shadow"]), (0.62, pal["tuft"]), (1.0, np.minimum(pal["tuft"] * 1.03, 1.0))])
    img = np.repeat(col[:, None, :], w, axis=1)
    u = (np.arange(w) + 0.5) / w
    edge = _smooth((np.abs(2 * u - 1) - 0.55) / 0.45)
    img = img * (1.0 - 0.10 * edge)[None, :, None]
    img = img * _fibres(rng, w, h, 0.7)[..., None]
    return _to_u8(img)


# ---------------------------------------------------------------- tail
def tail_fur(pal, rng, w=64, h=256):
    """Tail fur in the ear fur's look (colors.ears outer + outer_sheen + a cool tint): warm black, darker toward the buried
    root, a soft cool sheen line along the top (u ~ 0.12) that grows toward the tip, a faint band across the upper half and a
    lighter tip, a slightly darker underside."""
    v = 1.0 - (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    base, sheen = pal["outer"], _mix(pal["outer_sheen"], pal["cool"], 0.55)
    col = _grad(v, [(0.0, base * 0.55), (0.12, base * 0.78), (0.28, base), (1.0, base)])
    line = np.exp(-(((u - 0.12) / 0.11) ** 2))
    along = _grad(v, [(0.0, np.zeros(3)), (0.10, np.full(3, 0.10)), (0.45, np.full(3, 0.34)), (1.0, np.full(3, 0.55))])[:, 0]
    band = np.exp(-(((v - 0.70) / 0.10) ** 2)) * 0.30
    tip = _smooth((v - 0.88) / 0.12) * 0.30
    k = np.clip(line[None, :] * along[:, None] + (band + tip)[:, None], 0.0, 0.8)
    img = col[:, None, :] * (1 - k)[..., None] + sheen[None, None, :] * k[..., None]
    under = _smooth((u - 0.55) / 0.45)
    img = img * (1.0 - 0.22 * under)[None, :, None]
    img = img * _fibres(rng, w, h)[..., None]
    return _to_u8(img)


def sheen_sphere(pal, strength=0.16, size=128):
    """Additive sphere map (MMD indexes it by the view-space normal, row 0 = normals facing up): black everywhere except a
    faint cool glow toward the upper front-left and a hint of rim light, so black fur picks up a sheen without a hard
    highlight. `strength` is the peak added brightness."""
    c = (np.arange(size) + 0.5) / size
    x = (2.0 * c[None, :] - 1.0) * np.ones((size, 1))
    y = (1.0 - 2.0 * c[:, None]) * np.ones((1, size))
    r2 = x * x + y * y
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    n = np.stack([x, y, z], -1)
    n0 = np.array([-0.30, 0.52, 0.80])
    n0 = n0 / np.linalg.norm(n0)
    glow = np.exp(-((np.arccos(np.clip(n @ n0, -1.0, 1.0)) / 0.38) ** 2))
    rim = (1.0 - z) ** 3 * 0.5
    tint = _mix(pal["outer_sheen"], pal["cool"], 0.5)
    tint = tint / max(float(tint.max()), 1e-6)
    img = tint[None, None, :] * (strength * (glow + rim * 0.5))[..., None] * (r2 <= 1.0)[..., None]
    return _to_u8(img)
