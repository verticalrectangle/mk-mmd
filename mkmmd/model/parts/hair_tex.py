"""Textures of the hair part (numpy only): the hair atlas, the warm toon ramp, the palette and the braids' sphere ring.

Hair atlas (512 x 1024, v UP: v = 1 is the top row). Two regions, both made of clump tiles (96 px wide each: tiles 0-1 the
base shade for outer layers, tiles 2-3 a shade darker for inner layers; every tile is a convex ribbon, lighter along the
middle with a darker rim, and has one or two sparse strand lines) and a cap tile (the right 128 px, plain, for the scalp cap):
  HEAD    rows 0..511    v in (0.5, 1]: a vertical gradient by HEIGHT, f = 0 at the top of the hair to f = 1 at the lowest hair:
                         dark root, base red, slightly lighter tips; plus the baked angel ring: per clump tile a thin soft patch
                         of the sheen colour (`ring`, `ring_core` of the palette) at f ~ ring_f (the upper crown), never across a
                         whole clump, never on the cap tile, never down at the fringe. v = head_v(z, z_top, z_bot)
  STRAND  rows 512..1023 v in [0.5, 0]: root-to-tip gradient along a strand, t = 0 root .. 1 tip.   v = strand_v(t)
u in 0..1 across a clump: `clump_u(rng, dark)` picks a tile for a clump, `cap_u(theta)` maps the cap."""
import numpy as np

from ..colour import follow

W, H = 512, 1024
HEAD_ROWS = 512
BAND_F = 0.28


def srgb(h):
    """'#rrggbb' -> float rgb in 0..1 (sRGB encoded values)."""
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


DEFAULTS = {
    "base": "#dc2a2e", "shadow": "#8e1418", "deep": "#5a0c12", "light": "#f0503f", "highlight": "#ff9b7c",
    "highlight_core": "#ffd6c0", "rim": "#ff7a62",
}
TOON_SHADOW = (0.42, 0.44, 0.54)          # colors.hair.toon_shadow_multiplier: shadow = lit colour x this


def palette(colors=None):
    """name -> float rgb (sRGB values) for the hair, from the `[colors.hair]` table (DEFAULTS fill the gaps). Extra
    names derived here: `root` (deep), `tip` (light shifted towards the highlight), and the crown sheen `ring` and
    `ring_core`: given as keys of their own, else Rin's light pink-coral sheen (SHEEN, SHEEN_CORE) following the family's
    highlight (`follow`), so a whole colour family brings its own sheen."""
    colors = colors or {}
    pal = dict(DEFAULTS)
    pal.update({k: colors[k] for k in DEFAULTS if isinstance(colors.get(k), str)})   # unknown keys stay unread
    out = {k: srgb(v) for k, v in pal.items()}
    out["root"] = out["deep"]
    out["tip"] = out["light"]
    for key, sheen, ref in (("ring", SHEEN, "highlight"), ("ring_core", SHEEN_CORE, "highlight_core")):
        given = colors.get(key)
        out[key] = srgb(given) if isinstance(given, str) else follow(sheen, out[ref], srgb(DEFAULTS[ref]))
    return out


def toon_multiplier(colors=None):
    m = (colors or {}).get("toon_shadow_multiplier")
    return tuple(float(x) for x in m) if m else TOON_SHADOW


def head_v(z, z_top, z_bot):
    """Atlas v of a head-hair vertex at height z (z_top: crown, z_bot: lowest head hair)."""
    f = np.clip((z_top - np.asarray(z, float)) / max(z_top - z_bot, 1e-6), 0.0, 1.0)
    return 1.0 - 0.5 * (0.002 + 0.996 * f)


def strand_v(t):
    """Atlas v along a strand, t = 0 at the root to 1 at the tip."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return 0.5 - 0.5 * (0.002 + 0.996 * t)


TILE_PX = 96
N_TILES = 4
CAP_U0 = TILE_PX * N_TILES / W


TILE_TONE = (1.0, 1.0, 0.90, 0.88)          # tiles 2 and 3 (inner layers) are only a little darker
CAP_TONE = 0.80                           # the scalp cap under everything


def clump_u(rng, dark=False):
    """(u0, u1) of a clump tile (the clump's q = -1 .. +1 maps to u0 .. u1): tiles 0-1 are the base shade (outer layers),
    tiles 2-3 a shade darker (inner layers)."""
    k = int(rng.integers(2, 4)) if dark else int(rng.integers(0, 2))
    return (k * TILE_PX + 1.0) / W, ((k + 1) * TILE_PX - 1.0) / W


def cap_u(theta):
    """u for the scalp cap at azimuth theta: mirrored left/right, streaks radiate from the crown."""
    return CAP_U0 + 0.02 + (1.0 - CAP_U0 - 0.04) * np.abs(np.asarray(theta, float)) / np.pi


def _grad(t, stops):
    """Piecewise-linear colour gradient: stops [(pos, rgb)] sorted; t any shape -> (..., 3)."""
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops])
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)


def _noise1(n, rng, scale):
    """Smooth 1-D noise in [-1, 1] with ~`scale` features across n samples."""
    k = max(int(n / scale), 2)
    pts = rng.uniform(-1, 1, k + 3)
    x = np.linspace(0, k, n, endpoint=False)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def streaks(rng, w, h, strength=1.0):
    """(h, w) multiplicative fibre pattern ~1: vertical streaks (coarse locks + fine fibres), slowly varying down."""
    coarse = _noise1(w, rng, 36.0)
    mid = _noise1(w, rng, 9.0)
    fine = rng.uniform(-1, 1, w)
    dark_lines = (rng.uniform(0, 1, w) > 0.965).astype(float)
    base = 1.0 + strength * (0.05 * coarse + 0.035 * mid + 0.025 * fine - 0.07 * dark_lines)
    drift = 1.0 + 0.25 * strength * _noise1(h, rng, 70.0)[:, None] * mid[None, :]
    return base[None, :] * drift


def _sm(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _tiles(rng, h, strength=1.0):
    """(S, RIM), both (h, W): S the gentle multiplicative shading of a clump tile (a touch lighter along the middle), RIM the
    soft contact shadow towards a clump's edges (0 along the middle, 1 at the very edge, smooth falloff). No strand lines, no
    random blocks: every tone change is a long soft gradient."""
    S = np.ones((h, W))
    RIM = np.zeros((h, W))
    x = (np.arange(TILE_PX) + 0.5) / TILE_PX
    a = np.abs(2 * x - 1)
    rim = _sm((a - 0.30) / 0.70)
    prof = 1.0 + 0.03 * (1 - a * a)
    for k in range(N_TILES):
        S[:, k * TILE_PX:(k + 1) * TILE_PX] = prof[None, :]
        RIM[:, k * TILE_PX:(k + 1) * TILE_PX] = rim[None, :]
    return S, RIM


SHEEN = np.array([0.94, 0.58, 0.58])         # Rin's crown sheen: a light, less saturated warm pink-coral (never peach or white)
SHEEN_CORE = np.array([0.97, 0.72, 0.70])
# per clump tile: (centre offset in ring half-widths, width scale, strength, u where it fades in, u where it fades out, wave
# phase). The outer-layer tiles (0, 1) carry the sheen, the inner ones (2, 3) only a little: where clumps part, it is
# interrupted, and no two clumps agree on height, so the band undulates softly around the head instead of being a stripe.
TILE_SHEEN = ((-0.15, 1.00, 0.88, 0.00, 0.95, 0.10), (0.30, 0.90, 0.80, 0.05, 1.00, 0.45),
              (-0.05, 0.80, 0.40, 0.05, 0.85, 0.75), (0.20, 0.75, 0.34, 0.15, 0.96, 0.20))


def sheen_alpha(f, x, params, fc, h):
    """(alpha, core), both (len(f), len(x)): one tile's crown sheen, a band centred on height fraction `fc` (half width `h` x
    1.6 x the tile's width scale, smoothstep falloff: feathered, no hard edge), a gently wavy centre line along the clump,
    fading in and out along it; `core` the narrower brighter heart of it."""
    off, wsc, amp, u0, u1, ph = params
    c = fc + off * h + 0.30 * h * np.sin(2 * np.pi * (0.9 * x + ph))
    hw = 1.6 * h * wsc
    d = np.abs(f[:, None] - c[None, :]) / hw
    band = _sm(1.0 - d)
    core = _sm(1.0 - d / 0.45)
    win = _sm((x - u0) / 0.16) * _sm((u1 - x) / 0.16)
    return amp * win[None, :] * band, amp * win[None, :] * core


def make_atlas(pal, rng, ring_f=0.15, ring_w=0.045, ring=True):
    """The hair atlas as (1024, 512, 4) uint8 (sRGB, alpha 255).

    Head region: soft gradients only. Outer tiles (0-1): a slightly darker root at the top of the hair -> base -> a lighter tip
    over the whole height; inner tiles (2-3) and the cap just a little darker (x 0.90 / 0.88 / 0.80), clump edges a touch darker
    with a soft falloff. The baked crown sheen (`ring` false: none): per outer tile one wide feathered band of `pal["ring"]`
    (centre at height fraction `ring_f`, half width `ring_w` as a fraction of the height range) with a slightly brighter soft
    core of `pal["ring_core"]`, a wavy centre line, interrupted at clump edges, faint on the inner tiles, none on the cap tile;
    it follows the head and the lighting instead of crossing the bangs like a camera-relative sphere ring does."""
    img = np.zeros((H, W, 3))
    shadow, base, light = pal["shadow"], pal["base"], pal["light"]
    tone = np.concatenate([np.repeat(np.asarray(TILE_TONE), TILE_PX), np.full(W - N_TILES * TILE_PX, CAP_TONE)])
    f = (np.arange(HEAD_ROWS) + 0.5) / HEAD_ROWS
    outer = _grad(f, [(0.0, shadow * 0.5 + base * 0.5), (0.32, base), (0.72, base * 0.55 + light * 0.45), (1.0, light)])
    S, RIM = _tiles(rng, HEAD_ROWS)
    img[:HEAD_ROWS] = np.clip(outer[:, None, :] * tone[None, :, None] * (1.0 - 0.16 * RIM[..., None]) * S[..., None], 0, 1)
    x = (np.arange(TILE_PX) + 0.5) / TILE_PX
    for k, params in enumerate(TILE_SHEEN if ring else ()):
        a, c = sheen_alpha(f, x, params, ring_f, ring_w)
        sl = slice(k * TILE_PX, (k + 1) * TILE_PX)
        blk = img[:HEAD_ROWS, sl]
        blk = blk * (1 - a[..., None]) + pal["ring"] * a[..., None]
        blk = blk * (1 - 0.45 * c[..., None]) + pal["ring_core"] * (0.45 * c[..., None])
        img[:HEAD_ROWS, sl] = np.clip(blk, 0, 1)
    # ---- strand region: root to tip, the same soft tones, no sheen
    t = (np.arange(H - HEAD_ROWS) + 0.5) / (H - HEAD_ROWS)
    strand = _grad(t, [(0.0, shadow * 0.5 + base * 0.5), (0.25, base), (1.0, light)])
    S2, RIM2 = _tiles(rng, H - HEAD_ROWS)
    img[HEAD_ROWS:] = np.clip(strand[:, None, :] * tone[None, :, None] * (1.0 - 0.16 * RIM2[..., None]) * S2[..., None], 0, 1)
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.rint(img * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def toon_ramp(shadow=TOON_SHADOW, lit=(1.0, 1.0, 1.0), size=32, edge=0.42, soft=0.16):
    """A toon ramp (size x size uint8). MMD multiplies the diffuse texture by it, indexed by the VIEW-space normal's
    vertical component (mmd_tools: v = 0.5 + 0.5 N_y): the TOP row is for surfaces facing up (lit, `lit` = white) and the
    BOTTOM row for surfaces facing down (`shadow` = the per-channel multiplier, hair: [0.42, 0.44, 0.54] = a deep warm
    red, not grey). One soft step at `edge` (fraction down the ramp: 0.7 = surfaces facing down more than ~25 deg are shaded)."""
    t = (np.arange(size) + 0.5) / size
    s = np.clip((t - (edge - soft)) / (2 * soft), 0, 1)
    s = s * s * (3 - 2 * s)
    col = np.array(lit)[None, :] * (1 - s)[:, None] + np.array(shadow)[None, :] * s[:, None]
    img = np.repeat(col[:, None, :], size, axis=1)
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def sphere_ring(pal, size=256, ny=0.56, half=0.085, soft=0.06, strength=0.75):
    """The additive sphere map (angel ring): mmd_tools samples it at the view-space normal (u = 0.5 + 0.5 N_x, v = 0.5 +
    0.5 N_y), so a horizontal band at N_y = `ny` lights the surfaces that face the camera and tilt up by asin(ny): ONE
    crisp, curved band over the crown whatever the head does (flat top, soft edge), plus a thin faint second line below.
    The colour ADDED is the difference between the pale highlight and a lit red (R is already near 1). Returns
    (size, size, 4) uint8 (alpha 255)."""
    r = (np.arange(size) + 0.5) / size
    y = 1.0 - 2.0 * r                                       # row 0 = top = N_y +1
    x = 2.0 * r - 1.0

    def edge(d, s):
        t = np.clip((d + s) / (2 * s), 0, 1)
        return t * t * (3 - 2 * t)

    band = edge(half - np.abs(y - ny), soft)
    second = edge(0.012 - np.abs(y - (ny - 0.2)), 0.01) * 0.3
    fall = np.clip(1.0 - np.abs(x) ** 3, 0, 1)              # the ring fades towards the left/right silhouette
    tint = np.clip(pal["highlight"] - pal["base"], 0.0, 1.0) * strength          # what a lit base needs to reach the highlight
    core = np.clip(pal.get("highlight_core", pal["highlight"]) - pal["highlight"], 0.0, 1.0) * strength
    line = edge(0.012 - np.abs(y - ny), 0.012)                                   # a thin pale line in the middle of the band
    img = ((band + second)[:, None] * fall[None, :])[..., None] * tint[None, None, :] + \
          (line[:, None] * fall[None, :])[..., None] * core[None, None, :]
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out
