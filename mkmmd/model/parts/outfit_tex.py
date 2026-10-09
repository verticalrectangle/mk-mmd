"""Procedural outfit textures: dress print, frill cloth, satin ribbon, leather / sole, toon ramps, sphere maps.

numpy only (the same code runs in Blender's Python 3.11 / numpy 1.24 and with numpy 2; the output is bit-identical):
every pixel comes from code, nothing is sampled from any existing asset. All images are `(H, W, 4)` uint8 RGBA, alpha
255, image row 0 = TOP of the image (v = 1 in model UV space; the assembler flips for PMX), and are deterministic from
`seed`.

Real-world tile sizes (metres of cloth covered by ONE tile along u, and along v for the dress), for the UV layout:

    TILE_M = {"dress": 0.30, "frill_u": 0.08, "satin_u": 0.04, "leather": 0.10}

  dress_pattern   0.30 m x 0.30 m, tileable in u AND v (every motif is drawn with wrap-around indexing).
  frill_texture   0.08 m along u (tileable in u only); v runs from the sewn-on edge (row 0) to the free hem
                  (last row), about 0.075 m for the square image.
  satin_texture   0.04 m along the ribbon (u, tileable); v spans the whole ribbon width (rows 0 / last = selvedge).
  leather_texture 0.10 m in u and v (tileable both ways); also used for the sole.

Public API (colours: a dict of '#rrggbb' strings overriding any DEFAULT_COLORS key, unknown keys are ignored):

  dress_pattern(colors=None, size=2048, seed=11, print=True)   dark fabric with a leaf / vine / five-petal flower print
                                                               (print=False: the plain cloth)
  frill_texture(colors=None, size=1024, seed=3, inner=False)
  satin_texture(colors=None, size=512, seed=5, print=False)
  leather_texture(colors=None, size=512, seed=7, sole=False)
  toon_ramp(kind="dress", size=32)                       'dress' | 'frill' | 'satin' | 'leather'
  sphere_map(kind="satin", size=128)                     'satin' | 'leather'
  build_textures(ctx, colors=None, sizes=None, dress_print=True, leg_ribbon=True)   renders + saves everything, returns
                                                               {key: file name}

The dress print is laid out in centimetres on a 30 x 30 cm torus (independent of the render size) and is meant to read
as a floral from a distance: big motifs, sparsely scattered. Leaves are 4.5-7 cm long, the five-petal flowers 3.5-4.5 cm
across, leafy twigs 9-13 cm; a tile holds 4-5 flowers (all teal), one berry twig, at most 12 seed dots and about 45
motifs in all, covering ~21 % of the cloth. Sprigs (flower rosettes, leafy twigs, a berry twig, leaf pairs) are placed
one at a time where they overlap what is there the least (best-candidate blue noise with a halo, wrapped). The print
is deliberately low in contrast on a near-black ground: leaf bodies are dress_leaf taken 55 % of the way from
dress_base, their light halves dress_leaf_hi at 65 %, veins / shadow sides dress_leaf_dark at 45 % (the old
body-to-shadow ratio), every teal colour is dress_accent (and its tints) at 70 %. A very faint tone-on-tone underlayer
of large leaf silhouettes (dress_leaf at 7 %) sits beneath. Motifs are drawn from signed distance fields at full
resolution, each inside its own window (cost ~ print area), with ~1 px antialiasing; 2048 px takes ~2 s.

Toon multipliers (the exact numbers are documented in `toon_ramp`): dress shadow (0.42, 0.57, 0.56), frill shadow
(0.52, 0.62, 0.55), satin (0.70, 0.70, 0.78), leather (0.62, 0.66, 0.80).
"""
import functools
import math
from dataclasses import dataclass, replace

import numpy as np

DEFAULT_COLORS = {
    "dress_base": "#141b17",
    "dress_leaf": "#2f5a3a",
    "dress_leaf_hi": "#3d6e48",
    "dress_leaf_dark": "#18291d",
    "dress_accent": "#2d5a6e",
    "frill": "#3f8f4f",
    "frill_inner": "#9ccb8f",
    "satin": "#17151a",
    "leather": "#1c1a1f",
    "sole": "#0f0e11",
}
# the black items follow the project palette (colors.toml [colors.black]: ribbon, shoe, shoe_sole) when the spec has it;
# `colors.outfit` and the `colors` argument still win over it
PALETTE_BLACK = {"satin": "ribbon", "leather": "shoe", "sole": "shoe_sole"}


TILE_M = {"dress": 0.30, "frill_u": 0.08, "satin_u": 0.04, "leather": 0.10}

_TAU = 2.0 * math.pi


# --------------------------------------------------------------------------------------------- small helpers
def _hex(value):
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"bad colour {value!r} (expected '#rrggbb')")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _palette(colors=None):
    """Float RGB (0..1, sRGB values) per colour key; unknown keys in `colors` are ignored."""
    pal = dict(DEFAULT_COLORS)
    for key, val in (colors or {}).items():
        if key in pal:
            pal[key] = val
    return {k: _hex(v) for k, v in pal.items()}


def _mix(a, b, t):
    return a + (b - a) * t


def _sel(cond, a, b):
    """np.where with float32 branches (Python float branches would give float64 arrays)."""
    return np.where(cond, np.float32(a), np.float32(b))


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rng(seed, salt):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(salt)])


def _smooth_noise(rng, size, kmin, kmax, slope=1.0):
    """Periodic zero-mean unit-std float32 noise (size x size) from random Fourier coefficients with wavenumbers
    kmin <= |k| <= kmax (cycles per tile) and amplitude ~ |k|^-slope. Exactly tileable in both axes."""
    kmax = int(min(kmax, size // 2 - 1))
    ky = np.arange(-kmax, kmax + 1)
    kx = np.arange(0, kmax + 1)
    KX, KY = np.meshgrid(kx, ky)
    kk = np.hypot(KX, KY)
    amp = np.where((kk >= kmin) & (kk <= kmax), np.maximum(kk, 1.0) ** (-slope), 0.0)
    coef = amp * np.exp(1j * _TAU * rng.random(KX.shape))
    spec = np.zeros((size, size // 2 + 1), np.complex128)
    spec[np.ix_(ky % size, kx)] = coef
    out = np.fft.irfft2(spec, s=(size, size))
    return (out / (out.std() + 1e-12)).astype(np.float32)


def _to_rgba8(rgb, rng):
    """Float RGB (h, w, 3) in 0..1 -> opaque uint8 RGBA, with stochastic rounding so smooth dark ramps do not band."""
    h, w = rgb.shape[:2]
    v = rgb * 255.0 + rng.random((h, w, 1), dtype=np.float32)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.clip(v, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


# --------------------------------------------------------------------------------------------- dress: motifs
# The print is described in centimetres on a 30 x 30 cm torus (a layout independent of the render size), then drawn
# at full resolution from signed distance fields, each motif inside its own wrap-around window.

_T = TILE_M["dress"] * 100.0            # tile edge, cm
_G = 256                                # layout (coverage) grid, cells per tile edge


@dataclass
class _Leaf:
    x: float                            # base point, cm
    y: float
    ang: float                          # direction base -> tip (rad; canvas frame, x right, y down)
    length: float                       # cm
    w_up: float                         # half width / length on the +v side and on the -v side
    w_dn: float
    bend: float                         # midline offset at mid length / length (signed)
    tm: float                           # profile: where it is widest, exponent at the base side / the tip side
    bb: float
    bt: float
    veins: int = 0
    flip: int = 1                       # which half is the light one
    tone: float = 1.0
    hue: float = 0.0                    # slight red/blue tint of the green (print variety)
    kind: str = "leaf"                  # leaf | bud | sepal | ghost (tone-on-tone underlayer)
    accent: bool = False


@dataclass
class _Flower:
    x: float
    y: float
    rot: float
    radius: float                       # cm (diameter = 2 radius)
    pw: float                           # petal half width / radius
    tm: float
    bb: float
    bt: float
    tone: float = 1.0
    accent: bool = True


@dataclass
class _Dot:
    x: float
    y: float
    r: float                            # cm
    tone: float = 1.0
    accent: bool = False


@dataclass
class _Stem:
    pts: np.ndarray                     # (n, 2) cm
    w0: float                           # half width at the first point / the last point, cm
    w1: float


def _xf(p, x, y, rot):
    """Primitive `p` (local frame) rotated by `rot` about the origin, then moved to (x, y)."""
    c, s = math.cos(rot), math.sin(rot)
    if isinstance(p, _Stem):
        q = p.pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        return replace(p, pts=q)
    nx = x + p.x * c - p.y * s
    ny = y + p.x * s + p.y * c
    if isinstance(p, _Leaf):
        return replace(p, x=nx, y=ny, ang=p.ang + rot)
    if isinstance(p, _Flower):
        return replace(p, x=nx, y=ny, rot=p.rot + rot)
    return replace(p, x=nx, y=ny)


def _profile(t, tm, bb, bt):
    """Width factor f(t) in [0, 1] (1 at t = tm) and df/dt of a leaf outline, t in [0, 1]: sin(pi x)^beta with a
    different exponent on each side of the widest point (beta < 1 rounds the end, beta > 1 draws it out)."""
    up = t >= tm
    x = np.where(up, 0.5 + 0.5 * (t - tm) / (1.0 - tm), (0.5 / tm) * t)
    k = _sel(up, 0.5 / (1.0 - tm), 0.5 / tm)
    beta = _sel(up, bt, bb)
    ang = math.pi * x
    sn = np.maximum(np.sin(ang), 1e-3)
    f = np.exp(beta * np.log(sn))
    df = beta * f / sn * np.cos(ang) * (math.pi * k)
    return f, df


def _win(S, x0, x1, y0, y1):
    """Index arrays (wrap-around) and pixel-centre coordinates of the canvas window [x0, x1) x [y0, y1)."""
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    return ((ys % S)[:, None], (xs % S)[None, :], (xs + 0.5).astype(np.float32)[None, :],
            (ys + 0.5).astype(np.float32)[:, None])


def _outline_d(du_out, dv, g):
    """Distance estimate to a width-profile outline: first order inside the span, exact beyond its ends."""
    d_in = dv / np.sqrt(1.0 + g * g)
    return np.where(du_out > 0.0, np.hypot(du_out, np.maximum(dv, 0.0)), d_in)


def _leaf_field(lf, s, S):
    """Window indices, signed distance (px, negative inside) and the local fields of leaf `lf` at s px/cm."""
    L = lf.length * s
    wmax = max(lf.w_up, lf.w_dn) * L
    bd = lf.bend * L
    hv = wmax + abs(bd) + 2.0
    cx, cy = lf.x * s, lf.y * s
    c, sn = math.cos(lf.ang), math.sin(lf.ang)
    hu = 0.5 * L + 2.0
    mx, my = cx + c * 0.5 * L, cy + sn * 0.5 * L
    ex = abs(c) * hu + abs(sn) * hv + 2.0
    ey = abs(sn) * hu + abs(c) * hv + 2.0
    x0, y0 = int(math.floor(mx - ex)), int(math.floor(my - ey))
    x1, y1 = int(math.ceil(mx + ex)), int(math.ceil(my + ey))
    x1, y1 = min(x1, x0 + S), min(y1, y0 + S)
    yi, xi, X, Y = _win(S, x0, x1, y0, y1)
    dx, dy = X - cx, Y - cy
    u = dx * c + dy * sn
    v = dy * c - dx * sn
    t = u / L
    tc = np.clip(t, 0.0, 1.0)
    m = (4.0 * bd) * tc * (1.0 - tc)
    mp = (4.0 * bd / L) * (1.0 - 2.0 * tc)
    vp = v - m
    f, df = _profile(tc, lf.tm, lf.bb, lf.bt)
    up = vp >= 0.0
    W = _sel(up, lf.w_up * L, lf.w_dn * L)
    w = W * f
    g = mp + _sel(up, 1.0, -1.0) * (W * df / L)
    dv = np.abs(vp) - w
    du_out = np.maximum(np.maximum(-u, u - L), 0.0)
    d = _outline_d(du_out, dv, g)
    return yi, xi, d, (u, t, tc, vp, w, up, wmax, L)


def _petal_field(fl, k, s, X, Y):
    """Signed distance (px) and local coordinates of petal `k` of flower `fl` on the window coordinates X, Y (px,
    relative to the flower centre)."""
    R = fl.radius * s
    a = fl.rot + k * _TAU / 5.0
    c, sn = math.cos(a), math.sin(a)
    u = X * c + Y * sn
    v = Y * c - X * sn
    tc = np.clip(u / R, 0.0, 1.0)
    f, df = _profile(tc, fl.tm, fl.bb, fl.bt)
    w = (fl.pw * R) * f
    g = fl.pw * df
    du_out = np.maximum(np.maximum(-u, u - R), 0.0)
    d = _outline_d(du_out, np.abs(v) - w, g)
    return d, tc, v


def _flower_win(fl, s, S):
    R = fl.radius * s
    cx, cy = fl.x * s, fl.y * s
    h = R * 1.05 + 3.0
    x0, y0 = int(math.floor(cx - h)), int(math.floor(cy - h))
    x1, y1 = int(math.ceil(cx + h)), int(math.ceil(cy + h))
    x1, y1 = min(x1, x0 + S), min(y1, y0 + S)
    yi, xi, X, Y = _win(S, x0, x1, y0, y1)
    return yi, xi, X - cx, Y - cy


def _dot_win(dt, s, S):
    r = dt.r * s
    cx, cy = dt.x * s, dt.y * s
    h = r + 3.0
    x0, y0 = int(math.floor(cx - h)), int(math.floor(cy - h))
    x1, y1 = int(math.ceil(cx + h)), int(math.ceil(cy + h))
    yi, xi, X, Y = _win(S, x0, x1, y0, y1)
    return yi, xi, X - cx, Y - cy


def _stem_chunks(st, s, S):
    """(window, signed distance) per run of three segments of a stem; half widths taper linearly along the points."""
    pts = st.pts * s
    n = len(pts)
    rad = np.linspace(st.w0, st.w1, n) * s
    rad = np.maximum(rad, 0.6)
    i = 0
    while i < n - 1:
        j = min(i + 3, n - 1)
        seg = pts[i:j + 1]
        pad = float(rad[i:j + 1].max()) + 2.0
        x0, y0 = int(math.floor(seg[:, 0].min() - pad)), int(math.floor(seg[:, 1].min() - pad))
        x1, y1 = int(math.ceil(seg[:, 0].max() + pad)), int(math.ceil(seg[:, 1].max() + pad))
        x1, y1 = min(x1, x0 + S), min(y1, y0 + S)
        yi, xi, X, Y = _win(S, x0, x1, y0, y1)
        d = None
        for k in range(i, j):
            ax, ay = float(pts[k, 0]), float(pts[k, 1])
            bx, by = float(pts[k + 1, 0]), float(pts[k + 1, 1])
            ex, ey = bx - ax, by - ay
            ll = max(ex * ex + ey * ey, 1e-9)
            px, py = X - ax, Y - ay
            h = np.clip((px * ex + py * ey) / ll, 0.0, 1.0)
            r = float(rad[k]) + (float(rad[k + 1]) - float(rad[k])) * h
            dk = np.hypot(px - h * ex, py - h * ey) - r
            d = dk if d is None else np.minimum(d, dk)
        yield yi, xi, d
        i = j


# ------------------------------------------------------------------------------------ dress: coverage tracking
def _cover(cov, p, s):
    """Add the coverage (alpha, max-combined) of primitive `p` to the grid `cov` (s px per cm)."""
    S = cov.shape[0]
    if isinstance(p, _Leaf):
        yi, xi, d, _ = _leaf_field(p, s, S)
        cov[yi, xi] = np.maximum(cov[yi, xi], np.clip(0.5 - d, 0.0, 1.0))
    elif isinstance(p, _Flower):
        yi, xi, X, Y = _flower_win(p, s, S)
        a = None
        for k in range(5):
            d, _, _ = _petal_field(p, k, s, X, Y)
            ak = np.clip(0.5 - d, 0.0, 1.0)
            a = ak if a is None else np.maximum(a, ak)
        cov[yi, xi] = np.maximum(cov[yi, xi], a)
    elif isinstance(p, _Dot):
        yi, xi, X, Y = _dot_win(p, s, S)
        d = np.hypot(X, Y) - p.r * s
        cov[yi, xi] = np.maximum(cov[yi, xi], np.clip(0.5 - d, 0.0, 1.0))
    else:
        for yi, xi, d in _stem_chunks(p, s, S):
            cov[yi, xi] = np.maximum(cov[yi, xi], np.clip(0.5 - d, 0.0, 1.0))


def _box(a, r):
    """Periodic box blur of radius r (cells)."""
    n = 2 * r + 1
    h = np.zeros_like(a)
    for k in range(-r, r + 1):
        h += np.roll(a, k, axis=1)
    out = np.zeros_like(a)
    for k in range(-r, r + 1):
        out += np.roll(h, k, axis=0)
    return out / float(n * n)


def _samples(prims):
    """Local sample points (x, y, weight) of a sprig, used to score a placement against the coverage grid."""
    pts = []
    for p in prims:
        if isinstance(p, _Leaf):
            if p.kind == "sepal":
                continue
            c, sn = math.cos(p.ang), math.sin(p.ang)
            wt = p.length * p.length * (p.w_up + p.w_dn) * 0.5
            for t in (0.25, 0.5, 0.75):
                pts.append((p.x + c * p.length * t, p.y + sn * p.length * t, wt / 3.0))
        elif isinstance(p, _Flower):
            wt = p.radius * p.radius * 2.2
            pts.append((p.x, p.y, wt / 2.0))
            for k in range(5):
                a = p.rot + k * _TAU / 5.0
                pts.append((p.x + 0.6 * p.radius * math.cos(a), p.y + 0.6 * p.radius * math.sin(a), wt / 10.0))
        elif isinstance(p, _Dot):
            pts.append((p.x, p.y, p.r * p.r * 3.0))
    a = np.array(pts, np.float64)
    return a[:, 0], a[:, 1], a[:, 2]


def _place(rng, proto, cov, k_cand, halo=2, far=0.0):
    """Best of `k_cand` random placements of the sprig `proto` (local frame): the one overlapping the least of what is
    already on the torus (blurred by `halo` cells, so that pieces also keep a little apart). With `far` > 0 the score
    also counts, with that weight, the coverage within ~3 cm (a coarse 8-cell grid), which spreads the pieces evenly
    over the torus. Draws the winner into `cov` and returns the placed primitives."""
    s0 = _G / _T
    lx, ly, wt = _samples(proto)
    blur = _box(cov, halo)
    X = rng.random(k_cand) * _T
    Y = rng.random(k_cand) * _T
    R = rng.random(k_cand) * _TAU
    c, sn = np.cos(R)[:, None], np.sin(R)[:, None]
    px = X[:, None] + c * lx[None, :] - sn * ly[None, :]
    py = Y[:, None] + sn * lx[None, :] + c * ly[None, :]
    ix = np.floor(px * s0).astype(np.int64) % _G
    iy = np.floor(py * s0).astype(np.int64) % _G
    score = (blur[iy, ix] * wt[None, :]).sum(axis=1) / wt.sum() + 0.02 * rng.random(k_cand)
    if far > 0.0:
        f = 8
        coarse = _box(cov.reshape(_G // f, f, _G // f, f).mean(axis=(1, 3)), 3)
        score = score + far * (coarse[iy // f, ix // f] * wt[None, :]).sum(axis=1) / wt.sum()
    j = int(np.argmin(score))
    placed = [_xf(p, float(X[j]), float(Y[j]), float(R[j])) for p in proto]
    for p in placed:
        _cover(cov, p, s0)
    return placed


# ------------------------------------------------------------------------------------ dress: sprig generators
def _leaf(rng, x, y, ang, length, bend_sign=0.0, kind="leaf", veins=None):
    wf = rng.uniform(0.155, 0.225)
    if veins is None:
        veins = 4 if length < 5.4 else 5
        veins = max(3, veins - int(rng.random() < 0.3))
    return _Leaf(x, y, ang, length,
                 w_up=wf * rng.uniform(0.92, 1.1), w_dn=wf * rng.uniform(0.88, 1.06),
                 bend=bend_sign * rng.uniform(0.03, 0.1),
                 tm=rng.uniform(0.34, 0.5), bb=rng.uniform(0.62, 0.82), bt=rng.uniform(0.98, 1.45),
                 veins=veins, flip=1 if rng.random() < 0.5 else -1, tone=rng.uniform(0.92, 1.08),
                 hue=rng.uniform(-0.05, 0.05), kind=kind)


def _bud(rng, x, y, ang, length):
    """Egg-shaped bud with two small sepals under it (three primitives; only the first is a motif)."""
    bud = _Leaf(x, y, ang, length, rng.uniform(0.36, 0.42), rng.uniform(0.34, 0.4), 0.0,
                tm=rng.uniform(0.38, 0.46), bb=rng.uniform(0.55, 0.65), bt=rng.uniform(0.75, 0.95), veins=0,
                flip=1 if rng.random() < 0.5 else -1, tone=rng.uniform(0.95, 1.08), kind="bud")
    out = [bud]
    for sgn in (-1.0, 1.0):
        out.append(_Leaf(x, y, ang + sgn * rng.uniform(0.55, 0.8), length * rng.uniform(0.5, 0.62), 0.3, 0.3, 0.0,
                         tm=0.4, bb=0.7, bt=1.0, veins=0, flip=1, tone=0.9, kind="sepal"))
    return out


def _flower(rng, x, y):
    pointed = rng.random() < 0.35
    return _Flower(x, y, rng.uniform(0.0, _TAU), rng.uniform(1.75, 2.25), pw=rng.uniform(0.47, 0.55),
                   tm=0.58 if pointed else rng.uniform(0.6, 0.66), bb=0.9 if pointed else rng.uniform(0.8, 0.95),
                   bt=rng.uniform(0.8, 0.9) if pointed else rng.uniform(0.5, 0.6), tone=rng.uniform(0.95, 1.06))


def _curve(length, k1, k2, step=0.4):
    """Smooth stem along +x from the origin: heading k1 s + k2 s^2. Returns points, headings and arclengths."""
    n = max(3, int(round(length / step)) + 1)
    s = np.linspace(0.0, length, n)
    th = k1 * s + k2 * s * s
    ds = s[1] - s[0]
    mid = 0.5 * (th[1:] + th[:-1])
    pts = np.zeros((n, 2))
    pts[1:, 0] = np.cumsum(np.cos(mid)) * ds
    pts[1:, 1] = np.cumsum(np.sin(mid)) * ds
    return pts, th, s


def _on(pts, th, s, q):
    return float(np.interp(q, s, pts[:, 0])), float(np.interp(q, s, pts[:, 1])), float(np.interp(q, s, th))


def _sprig_branch(rng, end="flower"):
    """Leafy twig 9-13 cm long that ends in a flower, a bud or a leaf (`end`)."""
    length = rng.uniform(9.0, 13.0)
    pts, th, s = _curve(length, rng.uniform(-0.065, 0.065), rng.uniform(-0.0035, 0.0035), step=0.5)
    out = [_Stem(pts, 0.10, 0.05)]
    q = rng.uniform(1.4, 2.2)
    side = 1.0 if rng.random() < 0.5 else -1.0
    step = rng.uniform(3.4, 4.3)
    while q < length - 2.4:
        x, y, a = _on(pts, th, s, q)
        ln = float(np.clip(rng.uniform(5.2, 7.0) * (1.0 - 0.25 * q / length), 4.6, 7.0))
        out.append(_leaf(rng, x, y, a + side * rng.uniform(0.62, 0.98), ln, bend_sign=-side))
        side = -side
        q += step * rng.uniform(0.9, 1.1)
    x, y, a = _on(pts, th, s, length)
    if end == "flower":
        out.append(_flower(rng, x, y))
    elif end == "bud":
        out.extend(_bud(rng, x, y, a, rng.uniform(1.5, 2.0)))
    else:
        out.append(_leaf(rng, x, y, a + rng.uniform(-0.15, 0.15), rng.uniform(4.6, 5.6), bend_sign=side))
    return out


def _sprig_flower(rng):
    """Flower rosette (3.5-4.5 cm across) with 3-4 leaves around it and sometimes a budded stalk."""
    fl = _flower(rng, 0.0, 0.0)
    out = []
    m = 3 if rng.random() < 0.8 else 4
    for _ in range(40):                          # leaf directions: irregular, but never bunched or one-sided
        ang = np.sort(rng.random(m)) * _TAU
        gaps = np.diff(np.append(ang, ang[0] + _TAU))
        if gaps.min() > 0.9 and gaps.max() < 3.0:
            break
    for a in ang:
        a = float(a)
        out.append(_leaf(rng, 0.35 * fl.radius * math.cos(a), 0.35 * fl.radius * math.sin(a), a,
                         rng.uniform(4.6, 6.6), bend_sign=1.0 if rng.random() < 0.5 else -1.0))
    if rng.random() < 0.5:
        j = int(np.argmax(gaps))
        a = float(ang[j] + 0.5 * gaps[j])
        pts, th, s = _curve(rng.uniform(2.4, 3.6), rng.uniform(-0.15, 0.15), 0.0)
        c, sn = math.cos(a), math.sin(a)
        pts = pts @ np.array([[c, sn], [-sn, c]]) + 0.4 * fl.radius * np.array([c, sn])
        out.insert(0, _Stem(pts, 0.075, 0.05))
        out.extend(_bud(rng, float(pts[-1, 0]), float(pts[-1, 1]), a + th[-1], rng.uniform(1.4, 1.9)))
    out.append(fl)
    return out


def _sprig_berry(rng):
    length = rng.uniform(9.0, 12.0)
    pts, th, s = _curve(length, rng.uniform(-0.07, 0.07), rng.uniform(-0.0035, 0.0035), step=0.5)
    out = [_Stem(pts, 0.09, 0.045)]
    side = 1.0 if rng.random() < 0.5 else -1.0
    nb = int(rng.integers(2, 4))
    for k in range(nb):
        q = length * (0.28 + 0.62 * k / max(nb - 1, 1))
        x, y, a = _on(pts, th, s, q)
        ba = a + side * rng.uniform(0.7, 1.1)
        bl = rng.uniform(1.3, 2.4)
        sp, sth, ss = _curve(bl, side * rng.uniform(0.05, 0.25), 0.0, step=0.4)
        c, sn = math.cos(ba), math.sin(ba)
        sp = sp @ np.array([[c, sn], [-sn, c]]) + np.array([x, y])
        out.append(_Stem(sp, 0.06, 0.045))
        out.append(_Dot(float(sp[-1, 0] + c * 0.16), float(sp[-1, 1] + sn * 0.16), rng.uniform(0.30, 0.45),
                        tone=rng.uniform(0.95, 1.1)))
        side = -side
    x, y, a = _on(pts, th, s, length)
    out.append(_Dot(x, y, rng.uniform(0.28, 0.40)))
    for q, sd in ((length * 0.16, side), (length * 0.55, -side)):
        x, y, a = _on(pts, th, s, q)
        out.append(_leaf(rng, x, y, a + sd * rng.uniform(0.7, 1.0), rng.uniform(4.6, 5.8), bend_sign=-sd))
    return out


def _sprig_pair(rng):
    out = []
    spread = rng.uniform(0.45, 0.8)
    a0 = rng.uniform(-0.2, 0.2)
    stem_len = rng.uniform(1.3, 2.4)
    pts, th, s = _curve(stem_len, rng.uniform(-0.1, 0.1), 0.0, step=0.4)
    out.append(_Stem(pts - pts[-1], 0.075, 0.05))
    n = 2 if rng.random() < 0.8 else 1
    for k in range(n):
        sgn = 1.0 if (k == 0) else -1.0
        out.append(_leaf(rng, 0.0, 0.0, a0 + (sgn * spread if n == 2 else 0.0), rng.uniform(4.6, 6.2),
                         bend_sign=-sgn))
    return out


def _filler_leaf(rng):
    ln = rng.uniform(4.6, 5.4)
    pts, th, s = _curve(rng.uniform(0.8, 1.4), rng.uniform(-0.1, 0.1), 0.0, step=0.35)
    return [_Stem(pts - pts[-1], 0.075, 0.05), _leaf(rng, 0.0, 0.0, rng.uniform(-0.3, 0.3), ln,
                                                      bend_sign=1.0 if rng.random() < 0.5 else -1.0)]


def _filler_bud(rng):
    pts, th, s = _curve(rng.uniform(0.9, 1.5), rng.uniform(-0.25, 0.25), 0.0, step=0.35)
    return [_Stem(pts - pts[-1], 0.07, 0.045)] + _bud(rng, 0.0, 0.0, 0.0, rng.uniform(1.4, 1.9))


def _filler_dots(rng):
    return [_Dot(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0.12, 0.2),
                 tone=rng.uniform(0.95, 1.1)) for _ in range(2)]


@dataclass
class _Layout:
    prims: list
    coverage: float                     # fraction of the torus covered by a motif (layout grid estimate)


def _centre(p):
    """Middle of a motif, cm."""
    if isinstance(p, _Leaf):
        return p.x + 0.5 * p.length * math.cos(p.ang), p.y + 0.5 * p.length * math.sin(p.ang)
    return p.x, p.y


def _spread(chosen, cands, count):
    """Mark `count` of `cands` as accent, each time the one farthest (on the torus) from everything chosen so far."""
    if not cands or count <= 0:
        return chosen
    pts = np.array([_centre(p) for p in cands])
    dmin = np.full(len(cands), 1e9)

    def update(q):
        d = np.abs(pts - np.asarray(q))
        d = np.minimum(d, _T - d)
        np.minimum(dmin, np.hypot(d[:, 0], d[:, 1]), out=dmin)

    for q in chosen:
        update(q)
    for _ in range(min(count, len(cands))):
        j = int(np.argmax(dmin))
        cands[j].accent = True
        chosen.append(_centre(cands[j]))
        update(chosen[-1])
        dmin[j] = -1.0
    return chosen


def _assign_accent(prims):
    """Colour families. Every flower is teal; when 8 % of the motifs (leaves, buds, flowers, dots) is more than the
    flowers, a few leaves, spread evenly over the cloth, join them."""
    motifs = [p for p in prims if isinstance(p, (_Flower, _Dot))
              or (isinstance(p, _Leaf) and p.kind in ("leaf", "bud"))]
    flowers = [p for p in motifs if isinstance(p, _Flower)]
    leaves = [p for p in motifs if isinstance(p, _Leaf) and p.kind == "leaf"]
    _spread([_centre(p) for p in flowers], leaves, int(round(0.08 * len(motifs))) - len(flowers))


_COVER_TARGET = 0.21                    # layout grid coverage aimed at, and the range a layout must fall into
_COVER_RANGE = (0.200, 0.225)


def _motif_layout(seed, attempt):
    """One try at the dress motifs: (coverage, rng, primitives, coverage grid). 4-5 flowers per tile (two on leafy
    twigs, the others on rosettes with leaves around them), a third twig, one berry twig, 5-7 buds, then leaf pairs /
    single leaves up to ~21 % of the torus, then 4 pairs of seeds (at most 12 dots with the berries). Each piece goes
    where it overlaps what is there the least and where the neighbourhood is emptiest (halo + far term), so that the
    pieces are spread evenly; the flowers are the teal motifs."""
    rng = _rng(seed, 101 + 1000 * attempt)
    cov = np.zeros((_G, _G), np.float32)
    prims = []
    n_flowers = int(rng.integers(4, 6))
    makers = ([_sprig_flower] * (n_flowers - 2)
              + [lambda r: _sprig_branch(r, "flower")] * 2
              + [lambda r: _sprig_branch(r, "bud" if r.random() < 0.5 else "leaf")]
              + [_sprig_berry])
    for j in rng.permutation(len(makers)):
        prims.extend(_place(rng, makers[int(j)](rng), cov, 96, halo=6, far=1.0))
    for _ in range(int(rng.integers(6, 9))):
        prims.extend(_place(rng, _filler_bud(rng), cov, 64, halo=4, far=0.6))
    for _ in range(40):
        if cov.mean() >= 0.19:
            break
        prims.extend(_place(rng, _sprig_pair(rng), cov, 64, halo=5, far=0.7))
    for _ in range(40):
        if cov.mean() >= 0.205:
            break
        prims.extend(_place(rng, _filler_leaf(rng), cov, 48, halo=4, far=0.6))
    for _ in range(4):
        prims.extend(_place(rng, _filler_dots(rng), cov, 48, halo=3, far=0.3))
    _assign_accent(prims)
    return float(cov.mean()), rng, prims, cov


@functools.lru_cache(maxsize=8)
def _dress_layout(seed):
    """Motif list of the dress print for `seed` (cached, independent of the render size): a floral print of big, sparse
    motifs (see `_motif_layout`). The first of up to 8 tries whose coverage lands in 0.200-0.225 is taken (else the one
    closest to 0.21), so every seed gives about the same density. Underneath go large leaf silhouettes in a tone barely
    lighter than the cloth (`kind == "ghost"`, not motifs)."""
    best = None
    for attempt in range(8):
        cand = _motif_layout(seed, attempt)
        if best is None or abs(cand[0] - _COVER_TARGET) < abs(best[0] - _COVER_TARGET):
            best = cand
        if _COVER_RANGE[0] <= cand[0] <= _COVER_RANGE[1]:
            best = cand
            break
    coverage, rng, prims, cov = best
    ghost = cov.copy()
    ghosts = []
    while ghost.mean() < 0.62 and len(ghosts) < 60:
        ln = rng.uniform(5.0, 7.0)
        proto = [_leaf(rng, 0.0, 0.0, rng.uniform(-0.4, 0.4), ln, bend_sign=1.0 if rng.random() < 0.5 else -1.0,
                       kind="ghost", veins=4)]
        ghosts.extend(_place(rng, proto, ghost, 36, far=0.5))
    return _Layout(ghosts + prims, coverage)


# ------------------------------------------------------------------------------------------- dress: drawing
# Colour mixes of the print, each as the share of the way from the cloth colour (dress_base) to the palette colour.
# Low shares keep the contrast low: the print reads as dark leaves on near-black cloth.
_LEAF_MIX = 0.55           # leaf body        <- dress_leaf
_LEAF_HI_MIX = 0.65        # light half, glints <- dress_leaf_hi
_LEAF_DARK_MIX = 0.45      # veins / shadow side <- dress_leaf_dark (keeps the former shadow-to-body ratio)
_ACCENT_MIX = 0.70         # every colour of the teal family <- its full-strength colour
_GHOST_MIX = 0.07          # tone-on-tone underlayer <- dress_leaf


def _families(pal):
    """Colour sets of the print: green leaves, teal accent motifs and the tone-on-tone ghost layer, all built from the
    cloth colour by the `_*_MIX` shares above (leaf body 55 %, light half 65 %, shadow side 45 %, teal 70 %,
    underlayer 7 %), so the print is dark and low in contrast."""
    base, leaf, hi, dk = pal["dress_base"], pal["dress_leaf"], pal["dress_leaf_hi"], pal["dress_leaf_dark"]
    acc = pal["dress_accent"]
    pale = np.array([0.80, 0.95, 0.93], np.float32)
    body = _mix(base, leaf, _LEAF_MIX)
    dark = _mix(base, dk, _LEAF_DARK_MIX)
    acc_hi = _mix(acc, np.array([0.62, 0.86, 0.86], np.float32), 0.30)
    acc_tip = _mix(acc, np.array([0.70, 0.92, 0.90], np.float32), 0.45)         # lighter petal tips

    def dim(c, k):
        return _mix(base, c, k)

    return {
        "leaf": {"body": body, "hi": dim(hi, _LEAF_HI_MIX), "dark": dark,
                 "tip": dim(_mix(hi, pale, 0.4), _LEAF_HI_MIX), "pale": dim(_mix(hi, pale, 0.45), _LEAF_HI_MIX),
                 "stem": _mix(body, dark, 0.35)},
        "accent": {"body": dim(acc, _ACCENT_MIX), "hi": dim(acc_hi, _ACCENT_MIX),
                   "dark": dim(_mix(acc, dk, 0.6), _ACCENT_MIX), "tip": dim(acc_tip, _ACCENT_MIX),
                   "pale": dim(_mix(acc_tip, pale, 0.55), _ACCENT_MIX)},
        "ghost": {"body": dim(leaf, _GHOST_MIX), "hi": dim(leaf, 1.45 * _GHOST_MIX), "dark": base * 0.94},
    }


def _family(p):
    return "ghost" if getattr(p, "kind", "") == "ghost" else ("accent" if p.accent else "leaf")


def _leaf_color(lf, d, g, s, fam):
    u, t, tc, vp, w, up, wmax, L = g
    tint = np.array([1.0 + lf.hue, 1.0, 1.0 - lf.hue], np.float32) * lf.tone
    body, hi, dark = fam["body"] * tint, fam["hi"] * tint, fam["dark"] * tint
    if lf.kind == "sepal":
        return np.broadcast_to(_mix(body, dark, 0.35), d.shape + (3,))
    hb = _smooth(-0.9, 0.9, vp * float(lf.flip))
    col = body + (hi - body) * hb[..., None]
    edge_w = max(2.0, 0.28 * wmax)
    e = 1.0 - np.clip(-d / edge_w, 0.0, 1.0)
    k = e * e * (0.60 - 0.38 * hb)
    k = np.maximum(k, (1.0 - _smooth(0.0, 0.30, t)) * 0.5)
    col = col + (dark - col) * k[..., None]
    if lf.kind in ("leaf", "ghost") and L > 28.0:
        rw = np.maximum(0.6, (0.0075 * L) * (1.0 - 0.85 * tc))
        mr = np.clip(0.5 - (np.abs(vp) - rw), 0.0, 1.0) * (1.0 - _smooth(0.78, 0.97, t)) * (t > 0.0)
        col = col + (dark - col) * (0.85 * mr)[..., None]
        if lf.veins > 0:
            nv = lf.veins
            theta = 0.72
            cot, sth = math.cos(theta) / math.sin(theta), math.sin(theta)
            delta = 0.62 * L / nv
            q = u - cot * np.abs(vp) - 0.35 * vp * vp / wmax
            ph = (q - 0.15 * L - _sel(up, 0.5 * delta, 0.0)) / delta
            kk = np.rint(ph)
            valid = (kk >= 0) & (kk <= nv - 1) & (t < 0.94)
            dist = np.abs(ph - kk) * (delta * sth)
            vw = max(0.55, 0.0035 * L)
            fade = np.clip((0.84 * w - np.abs(vp)) / (0.3 * w + 1e-3), 0.0, 1.0)
            va = np.clip(0.5 - (dist - vw), 0.0, 1.0) * valid * fade * (np.abs(vp) > rw)
            col = col + (dark - col) * (0.42 * va)[..., None]
    return col


def _draw_leaf(img, lf, s, fam):
    yi, xi, d, g = _leaf_field(lf, s, img.shape[0])
    a = np.clip(0.5 - d, 0.0, 1.0)
    col = _leaf_color(lf, d, g, s, fam)
    win = img[yi, xi]
    img[yi, xi] = win + (col - win) * a[..., None]


def _draw_flower(img, fl, s, fam):
    S = img.shape[0]
    R = fl.radius * s
    yi, xi, X, Y = _flower_win(fl, s, S)
    win = img[yi, xi]
    body, tip, dark, pale = fam["body"] * fl.tone, fam["tip"] * fl.tone, fam["dark"], fam["pale"]
    for k in range(5):
        d, tc, v = _petal_field(fl, k, s, X, Y)
        a = np.clip(0.5 - d, 0.0, 1.0)
        col = dark + (body - dark) * _smooth(0.05, 0.5, tc)[..., None]
        col = col + (tip - col) * (0.9 * _smooth(0.58, 0.97, tc))[..., None]
        cr = np.clip(0.5 - (np.abs(v) - max(0.7, 0.0096 * R)), 0.0, 1.0)
        cr = cr * _smooth(0.10, 0.22, tc) * (1.0 - _smooth(0.45, 0.75, tc)) * 0.45
        rim = (1.0 - _smooth(0.0, max(2.2, 0.024 * R), -d)) * 0.45
        col = col + (dark - col) * np.maximum(cr, rim)[..., None]
        win = win + (col - win) * a[..., None]
    rc = 0.17 * R
    a = np.clip(0.5 - (np.hypot(X, Y) - rc), 0.0, 1.0)
    win = win + (dark - win) * a[..., None]
    n = 7
    for k in range(n):
        ang = fl.rot + k * _TAU / n + 0.3
        sx, sy = 0.27 * R * math.cos(ang), 0.27 * R * math.sin(ang)
        a = np.clip(0.5 - (np.hypot(X - sx, Y - sy) - max(1.0, 0.05 * R)), 0.0, 1.0)
        win = win + (pale - win) * (0.9 * a)[..., None]
    img[yi, xi] = win


def _draw_dot(img, dt, s, fam):
    S = img.shape[0]
    yi, xi, X, Y = _dot_win(dt, s, S)
    r = dt.r * s
    d = np.hypot(X, Y) - r
    a = np.clip(0.5 - d, 0.0, 1.0)
    body, hi, dark = fam["body"] * dt.tone, fam["tip"] * dt.tone, fam["dark"]
    col = body + (dark - body) * (_smooth(-0.35 * r, 0.0, d) * 0.5)[..., None]
    hl = np.clip(0.5 - (np.hypot(X + 0.3 * r, Y + 0.3 * r) - 0.32 * r), 0.0, 1.0) * 0.9
    col = col + (hi - col) * hl[..., None]
    win = img[yi, xi]
    img[yi, xi] = win + (col - win) * a[..., None]


def _draw_stem(img, st, s, fam):
    S = img.shape[0]
    col = fam["stem"]
    for yi, xi, d in _stem_chunks(st, s, S):
        a = np.clip(0.5 - d, 0.0, 1.0)
        win = img[yi, xi]
        img[yi, xi] = win + (col - win) * a[..., None]


def _dress_cloth(S, rng):
    """Multiplicative fabric modulation (S, S): low-frequency mottling, fine twill, thread noise. Within +-3 %."""
    mott = np.clip(_smooth_noise(rng, S, 2, 14, 1.2) / 2.5, -1.0, 1.0)
    n = max(1, int(round(S / 6.0)))
    idx = np.arange(S, dtype=np.float32)
    ph = (idx[None, :] + idx[:, None]) * (n / S)
    w = np.sin(_TAU * ph) + 0.5 * np.sin(2.0 * _TAU * ph + 0.6)
    thread = rng.random(S, dtype=np.float32)[None, :] + rng.random(S, dtype=np.float32)[:, None] - 1.0
    return (1.0 + 0.012 * mott + 0.010 * (w / 1.5) + 0.008 * thread).astype(np.float32)


def dress_pattern(colors=None, size=2048, seed=11, print=True):
    """Dark dress fabric with a leaf / vine / five-petal flower print, tileable in both axes (0.30 m per tile); with
    print=False the plain cloth: the same weave and mottling, no motifs."""
    pal = _palette(colors)
    S = int(size)
    s = S / _T
    prims = _dress_layout(int(seed)).prims if print else ()
    fam = _families(pal)
    img = np.empty((S, S, 3), np.float32)
    img[:] = pal["dress_base"]
    for p in prims:
        if isinstance(p, _Leaf) and p.kind == "ghost":
            _draw_leaf(img, p, s, fam["ghost"])
    for p in prims:
        if isinstance(p, _Stem):
            _draw_stem(img, p, s, fam["leaf"])
    for p in prims:
        if isinstance(p, _Leaf) and p.kind in ("leaf", "sepal"):
            _draw_leaf(img, p, s, fam[_family(p)])
    for p in prims:
        if isinstance(p, _Leaf) and p.kind == "bud":
            _draw_leaf(img, p, s, fam[_family(p)])
    for p in prims:
        if isinstance(p, _Flower):
            _draw_flower(img, p, s, fam[_family(p)])
    for p in prims:
        if isinstance(p, _Dot):
            _draw_dot(img, p, s, fam[_family(p)])
    rng = _rng(seed, 202)
    img *= _dress_cloth(S, rng)[..., None]
    return _to_rgba8(img, rng)


# ------------------------------------------------------------------------------------------- small fabrics
def _centres(S):
    """Pixel-centre coordinates: (S, 1) rows and (1, S) columns."""
    idx = np.arange(S, dtype=np.float32) + 0.5
    return idx[:, None], idx[None, :]


def _weave(S, rng, pitch=3.0):
    """Plain weave in [-1, 1] (product of two sines, integer periods per tile) plus per-thread noise, (S, S)."""
    ys, xs = _centres(S)
    n = max(1, int(round(S / (2.0 * pitch))))
    w = np.sin(_TAU * n * xs / S) * np.sin(_TAU * n * ys / S)
    th = rng.random(S, dtype=np.float32)[None, :] + rng.random(S, dtype=np.float32)[:, None] - 1.0
    return w, th


def _blend(rgb, y0, y1, col, a):
    """In-place blend of `col` over the rows y0:y1 of `rgb` with coverage `a` (rows, S)."""
    rgb[y0:y1] += (col - rgb[y0:y1]) * a[..., None]


def _stitch(rgb, yline, n, duty, thick, lighten, shadow):
    """A dashed stitch line: n round-ended dashes per tile (duty = dash length / period), `thick` px thick, the cloth
    lightened by `lighten` towards a pale thread tint, with a pucker line (darker by `shadow`) just under it."""
    S = rgb.shape[1]
    P = S / float(n)
    y0, y1 = max(0, int(yline - thick - 4)), min(S, int(yline + 2 * thick + 5))
    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
    xs = (np.arange(S, dtype=np.float32) + 0.5)[None, :]
    px = np.abs((xs % P) - 0.5 * P)
    ex = np.maximum(px - (0.5 * duty * P - 0.5 * thick), 0.0)
    a = np.clip(0.5 - (np.hypot(ex, ys - yline) - 0.5 * thick), 0.0, 1.0)
    sh = np.clip(0.5 - (np.hypot(ex, ys - yline - 0.9 * thick) - 0.5 * thick), 0.0, 1.0)
    rgb[y0:y1] *= (1.0 - shadow * sh)[..., None]
    _blend(rgb, y0, y1, np.array([1.0, 1.0, 0.96], np.float32), lighten * a)


def _scallops(rgb, y0f, hf, n, thick, alpha):
    """Faint white lace-like scallop line: n circular arcs per tile hanging from the line y0f (arc height = hf x arc
    width), a small eyelet inside every arc."""
    S = rgb.shape[1]
    P, y0 = S / float(n), y0f * S
    h = hf * P
    Rc = (0.25 * P * P + h * h) / (2.0 * h)
    r0, r1 = max(0, int(y0 - 0.5 * P)), min(S, int(y0 + h + thick + 6))
    ys = (np.arange(r0, r1, dtype=np.float32) + 0.5)[:, None]
    xs = (np.arange(S, dtype=np.float32) + 0.5)[None, :]
    xl = (xs % P) - 0.5 * P
    yl = ys - y0
    d = np.where(yl >= 0.0, np.abs(np.hypot(xl, yl - (h - Rc)) - Rc), np.hypot(np.abs(xl) - 0.5 * P, yl))
    a = np.clip(0.5 + 0.5 * thick - d, 0.0, 1.0)
    de = np.hypot(xl, yl - 0.42 * h) - max(1.5, 0.0035 * S)
    a2 = np.clip(0.5 - de, 0.0, 1.0)
    white = np.array([1.0, 1.0, 0.97], np.float32)
    _blend(rgb, r0, r1, white, np.maximum(a * alpha, a2 * alpha * 0.85))


def frill_texture(colors=None, size=1024, seed=3, inner=False):
    """Ruffle cloth, tileable along x (8 cm). Row 0 = where the frill is sewn on to the dress, last row = free hem.

    Vertical profile: the underside next to the dress is in shade. Over the top 22 % of the rows a smooth gradient takes
    the cloth from x0.50 of the frill colour at row 0 up to x1.0 (`inner=True`: gentler, x0.72), and a thin darker seam
    line, centred at 1.25 % of the height inside the first 2.5 % of the rows, dips to x0.40 (inner: x0.65); below that
    the cloth is flat, +7 % towards the hem; the last 3 % of the rows are the hem band (30 % towards a pale tint of the
    cloth, with a thin fold shadow above it); dashed stitch lines (19 dashes per tile, thread = cloth lightened by
    20 %) at 6 % and 93 % of the height, the upper one shaded with the cloth; woven noise +-2 %. `inner=True` uses the
    `frill_inner` colour, a hint warmer towards the hem, and adds a very faint white scalloped lace line (10 arcs per
    tile, alpha 0.30) with small eyelets, hanging from 80 % of the height."""
    pal = _palette(colors)
    S = int(size)
    rng = _rng(seed, 303 + (1 if inner else 0))
    base = pal["frill_inner"] if inner else pal["frill"]
    ys, xs = _centres(S)
    yy = ys / S
    lum = 1.0 + 0.07 * _smooth(0.35, 1.0, yy)
    w, th = _weave(S, rng)
    slub = np.clip(_smooth_noise(rng, S, 3, 24, 1.0) / 2.5, -1.0, 1.0)
    mod = 1.0 + 0.010 * w + 0.008 * th + 0.012 * slub
    tint = np.ones((S, 1, 3), np.float32)
    if inner:
        warm = _smooth(0.25, 1.0, yy)[..., None]
        tint = tint + warm * np.array([0.02, 0.0, -0.035], np.float32)
    rgb = base[None, None, :] * (lum * mod)[..., None] * tint
    light = _mix(base, np.array([1.0, 1.0, 0.96], np.float32), 0.55)
    yb = ys - 0.97 * S
    hb = np.clip(0.5 + yb, 0.0, 1.0)
    fold = np.exp(-0.5 * ((yb + 1.6 * S / 1024.0) / (1.1 * S / 1024.0)) ** 2)
    rgb = rgb * (1.0 - 0.16 * fold)[..., None]
    rgb = rgb + (light - rgb) * (0.30 * hb)[..., None]
    thick = max(1.5, 0.0055 * S)
    _stitch(rgb, 0.06 * S, 19, 0.6, thick, 0.20, 0.20)
    _stitch(rgb, 0.93 * S, 19, 0.6, thick, 0.20, 0.20)
    if inner:
        _scallops(rgb, 0.80, 0.42, 10, max(1.5, 0.0045 * S), 0.30)
    top0, seam_k = (0.72, 0.65) if inner else (0.50, 0.40)
    grad = top0 + (1.0 - top0) * _smooth(0.0, 0.22, yy)                  # shaded underside: x top0 at row 0 -> 1 at 22 %
    seam = np.exp(-0.5 * ((yy - 0.0125) / 0.005) ** 2)                   # thin seam line inside the first 2.5 % of rows
    rgb = rgb * (grad + (seam_k - grad) * seam)[..., None]               # after the stitches: they sit in the shade too
    return _to_rgba8(np.clip(rgb, 0.0, 1.0), rng)


def satin_texture(colors=None, size=512, seed=5, print=False):
    """Black satin ribbon: u along the ribbon (tile 4 cm, tileable in x), v across its width (rows 0 / last = selvedge).

    Luminance modulation: fine diagonal twill +-1.5 %, broad diagonal sheen bands +-2 %, warp streaks +-0.5 % (max
    +-4 %); the outer 6 % of the width on each side is a slightly lighter selvedge line (+0.05 in sRGB value). With
    `print=True` a sparse pale grey (0.78, 0.78, 0.80) mark pattern at alpha 0.22 is added, repeating every 1 cm of
    ribbon: a small diamond outline on the centre line, a dot between diamonds, a short dash near each selvedge."""
    pal = _palette(colors)
    S = int(size)
    rng = _rng(seed, 404)
    base = pal["satin"]
    ys, xs = _centres(S)
    n1 = max(1, int(round(S / 5.0)))
    fine = np.sin(_TAU * n1 * (xs + ys) / S)
    broad = np.sin(_TAU * (3.0 * xs + 2.0 * ys) / S)
    warp = (rng.random(S, dtype=np.float32)[:, None] - 0.5) * 2.0
    mod = 1.0 + 0.015 * fine + 0.020 * broad + 0.005 * warp
    rgb = base[None, None, :] * mod[..., None]
    de = np.minimum(ys, S - ys) / S
    edge = 1.0 - _smooth(0.045, 0.062, de)
    outer = 1.0 - _smooth(0.0, 0.012, de)
    rgb = rgb + (0.05 * edge + 0.03 * outer)[..., None]
    if print:
        mark = np.array([0.78, 0.78, 0.80], np.float32)
        P = S / 4.0
        px = (xs % P) - 0.5 * P                              # across one 1 cm period, 0 at its middle
        py = ys - 0.5 * S
        hx, hy = 0.28 * P, 0.085 * S
        thick = max(1.5, 0.010 * S)
        n = np.abs(px) / hx + np.abs(py) / hy
        a = np.clip(0.5 + 0.5 * thick - np.abs(n - 1.0) / math.sqrt(1.0 / hx ** 2 + 1.0 / hy ** 2), 0.0, 1.0)
        a = np.maximum(a, np.clip(0.5 - (np.hypot(np.abs(px) - 0.5 * P, py) - max(1.5, 0.011 * S)), 0.0, 1.0))
        for sgn in (-1.0, 1.0):
            ex = np.maximum(np.abs(px) - 0.17 * P, 0.0)
            a = np.maximum(a, np.clip(0.5 - (np.hypot(ex, py - sgn * 0.30 * S) - 0.5 * max(1.5, 0.008 * S)),
                                      0.0, 1.0))
        rgb = rgb + (mark - rgb) * (0.22 * a)[..., None]
    return _to_rgba8(np.clip(rgb, 0.0, 1.0), rng)


def leather_texture(colors=None, size=512, seed=7, sole=False):
    """Near-black shoe leather (the `sole` colour with sole=True), tileable in u and v (10 cm per tile): very fine
    pebbled grain (+-2.5 % luminance grain, -4.5 % creases between the pebbles, +-1.5 % soft mottling) and a barely
    visible lighter wear patch at the centre (+0.02 in sRGB value, ~11 % of the tile wide)."""
    pal = _palette(colors)
    S = int(size)
    rng = _rng(seed, 505 + (1 if sole else 0))
    base = pal["sole" if sole else "leather"]
    fine = _smooth_noise(rng, S, S / 10.0, S / 3.2, 0.0)
    crease = np.exp(-((fine / 0.30) ** 2))
    soft = np.clip(_smooth_noise(rng, S, 1, 6, 1.0) / 2.5, -1.0, 1.0)
    ys, xs = _centres(S)
    wear = np.exp(-2.2 * (1.0 - np.cos(_TAU * (ys / S - 0.5)))) * np.exp(-2.2 * (1.0 - np.cos(_TAU * (xs / S - 0.5))))
    lum = 1.0 + 0.025 * np.clip(fine / 2.5, -1.0, 1.0) - 0.045 * crease + 0.015 * soft
    rgb = base[None, None, :] * lum[..., None] + (0.02 * wear)[..., None]
    return _to_rgba8(np.clip(rgb, 0.0, 1.0), rng)


# --------------------------------------------------------------------------------------------- toon, sphere
# (edge start, edge end, multiplier reached) per step: smoothstep edges between plateaus, starting from white.
_TOON = {
    "dress": ((0.40, 0.48, (0.80, 0.89, 0.89)), (0.62, 0.72, (0.42, 0.57, 0.56))),
    "frill": ((0.40, 0.54, (0.80, 0.86, 0.82)), (0.66, 0.78, (0.52, 0.62, 0.55))),
    "satin": ((0.30, 0.95, (0.70, 0.70, 0.78)),),
    "leather": ((0.38, 0.80, (0.62, 0.66, 0.80)),),
}


def toon_ramp(kind="dress", size=32):
    """MMD toon image (size, size, 4): a vertical gradient (row 0 = fully lit white, last row = shaded), every row
    constant along x. The colour is a per-channel multiplier of the lit colour; plateaus joined by smoothstep edges
    (rows are sampled at v = (row + 0.5) / size):

      dress    1.00 up to v 0.40, edge to (0.80, 0.89, 0.89) by 0.48, edge to (0.42, 0.57, 0.56) over 0.62 .. 0.72
      frill    1.00 up to v 0.40, edge to (0.80, 0.86, 0.82) by 0.54, edge to (0.52, 0.62, 0.55) over 0.66 .. 0.78
      satin    1.00 up to v 0.30, one long soft edge to (0.70, 0.70, 0.78) by 0.95
      leather  1.00 up to v 0.38, soft edge to (0.62, 0.66, 0.80) by 0.80

    (dress shadow = deep teal-green, frill shadow multiplies green by 0.62 with red / blue lower, leather shadow is
    slightly bluish). Alpha is 255."""
    if kind not in _TOON:
        raise ValueError(f"unknown toon kind {kind!r}; expected one of {sorted(_TOON)}")
    v = (np.arange(size, dtype=np.float32) + 0.5) / size
    col = np.ones((size, 3), np.float32)
    prev = np.ones(3, np.float32)
    for a, b, c in _TOON[kind]:
        c = np.array(c, np.float32)
        col = col + (c - prev) * _smooth(a, b, v)[:, None]
        prev = c
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, None, :]
    out[..., 3] = 255
    return out


def sphere_map(kind="satin", size=128):
    """Spherical environment map (size, size, 4) for ADD mode (black = no change), alpha 255. The image is the unit
    disc seen from the front (top = normal up, left = normal to the left), black outside radius 0.97.

      satin    a broad soft sheen blob high left: centre (-0.36, -0.42), Gaussian sigma 0.50 along the diagonal
               (lower left -> upper right) and 0.22 across it, peak 0.15 of white (blue x1.04: at most 40 / 255)
      leather  a small sharper highlight: centre (-0.30, -0.36), radius ~0.13 (core 0.27, flat-topped falloff) plus a
               faint glow around it (sigma 0.24, 0.03); peak 0.30 (blue x1.02: at most 78 / 255); the rest stays black

    Both are deliberately weak: the map is ADDED to the lit colour, and the satin / leather things it is used on are
    near-black, so a stronger blob turns them grey."""
    if kind not in ("satin", "leather"):
        raise ValueError(f"unknown sphere kind {kind!r}; expected 'satin' or 'leather'")
    c = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    X, Y = c[None, :], c[:, None]
    if kind == "satin":
        px, py = X + 0.36, Y + 0.42
        along = (px - py) / math.sqrt(2.0)
        across = (px + py) / math.sqrt(2.0)
        inten = 0.15 * np.exp(-0.5 * (along / 0.50) ** 2 - 0.5 * (across / 0.22) ** 2)
        tint = np.array([1.0, 1.0, 1.04], np.float32)
    else:
        r = np.hypot(X + 0.30, Y + 0.36)
        inten = 0.27 * np.exp(-((r / 0.13) ** 3)) + 0.03 * np.exp(-0.5 * (r / 0.24) ** 2)
        tint = np.array([1.0, 1.0, 1.02], np.float32)
    inten = inten * (1.0 - _smooth(0.80, 0.97, np.hypot(X, Y)))
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.clip(inten[..., None] * tint * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


# ------------------------------------------------------------------------------------------------ build step
_SIZES = {"dress": 2048, "frill": 1024, "satin": 512, "leather": 512, "toon": 32, "sphere": 128}


def build_textures(ctx, colors=None, sizes=None, dress_print=True, leg_ribbon=True):
    """Render and save every outfit image with `ctx.save_png`; returns {key: file name}. Colours: DEFAULT_COLORS, then the
    project palette's black items (`ctx.spec["colors"]["black"]`: ribbon -> satin, shoe -> leather, shoe_sole -> sole), then
    `ctx.spec["colors"]["outfit"]`, then `colors`. `sizes` (optional) overrides the default edge lengths per group
    (dress 2048, frill 1024, satin 512, leather 512, toon 32, sphere 128); dress_print=False makes the dress plain cloth,
    leg_ribbon=False leaves out the calf ribbon's images (satin_print, sphere_satin)."""
    cols = dict(DEFAULT_COLORS)
    black = (ctx.spec.get("colors") or {}).get("black") or {}
    cols.update({k: black[v] for k, v in PALETTE_BLACK.items() if v in black})
    cols.update(((ctx.spec.get("colors") or {}).get("outfit")) or {})
    cols.update(colors or {})
    sz = dict(_SIZES)
    sz.update(sizes or {})
    images = {
        "dress": ("outfit_dress.png", lambda: dress_pattern(cols, sz["dress"], print=dress_print)),
        "frill": ("outfit_frill.png", lambda: frill_texture(cols, sz["frill"])),
        "frill_inner": ("outfit_frill_inner.png", lambda: frill_texture(cols, sz["frill"], inner=True)),
        "satin": ("outfit_satin.png", lambda: satin_texture(cols, sz["satin"])),
        "satin_print": ("outfit_satin_print.png", lambda: satin_texture(cols, sz["satin"], print=True)),
        "leather": ("outfit_leather.png", lambda: leather_texture(cols, sz["leather"])),
        "sole": ("outfit_sole.png", lambda: leather_texture(cols, sz["leather"], sole=True)),
        "toon_dress": ("outfit_toon_dress.png", lambda: toon_ramp("dress", sz["toon"])),
        "toon_frill": ("outfit_toon_frill.png", lambda: toon_ramp("frill", sz["toon"])),
        "toon_satin": ("outfit_toon_satin.png", lambda: toon_ramp("satin", sz["toon"])),
        "toon_leather": ("outfit_toon_leather.png", lambda: toon_ramp("leather", sz["toon"])),
        "sphere_satin": ("outfit_sphere_satin.png", lambda: sphere_map("satin", sz["sphere"])),
        "sphere_leather": ("outfit_sphere_leather.png", lambda: sphere_map("leather", sz["sphere"])),
    }
    if not leg_ribbon:                                          # the calf ribbon's own images: no ribbon, no images
        del images["satin_print"], images["sphere_satin"]
    return {key: ctx.save_png(name, make()) for key, (name, make) in images.items()}
