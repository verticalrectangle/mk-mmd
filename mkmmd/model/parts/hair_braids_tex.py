"""Textures of the braids (numpy only): the braid atlas (red plait strands + the hair strand region) and the black satin
ribbon.

Braid atlas, same size and strand region as the head-hair atlas (hair_tex.make_atlas): the upper half (the head's height
field, unused here) is replaced by
  STRAND tiles columns k * TILE_PX .. (k + 1) * TILE_PX, k = 0..2: one plait strand each. u = |angle| / pi around the
               strand (0 = the crest facing out, 1 = the underside), v = 0..1 along it. The shading follows the weave:
               a highlight on the crest where the strand lies on top, dark where it dives under a neighbour, a dark
               crease along both side edges, fibre streaks along v.
The lower half is hair_tex's strand region (root-to-tip, 4 clump tiles) for the gather clumps and the tuft locks.

All uv helpers return texture coordinates with v UP (v = 1 is the top row of the PNG)."""
import numpy as np

from . import hair_tex as TEX

W, H, HEAD_ROWS = TEX.W, TEX.H, TEX.HEAD_ROWS
TILE = TEX.TILE_PX
N_TILES = TEX.N_TILES
PAD = 1.0                                           # px kept off the tile borders (no bleeding between tiles)


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a * (1.0 - t)[..., None] + b * t[..., None]


# ---------------------------------------------------------------- uv helpers
def plait_uv(tile, u, v):
    """UV of strand vertices (u, v arrays) in tile `tile`."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    v = np.clip(np.asarray(v, float), 0.0, 1.0)
    U = (tile * TILE + PAD + u * (TILE - 2 * PAD)) / W
    row = PAD + v * (HEAD_ROWS - 2 * PAD)
    return np.stack([U, 1.0 - (row + 0.5) / H], -1)



def strand_uv(tile, q, t):
    """UV of lock / clump vertices: q across the clump (-1 .. 1), t along it (0 = root .. 1 = tip) in strand tile `tile`.
    The tiles of the strand region are the head-hair tiles (hair_tex.clump_u)."""
    q = np.asarray(q, float)
    u0, u1 = (tile * TILE + 1.0) / W, ((tile + 1) * TILE - 1.0) / W
    return np.stack([u0 + (q + 1.0) * 0.5 * (u1 - u0), TEX.strand_v(t) + 0 * q], -1)


# ---------------------------------------------------------------- the strand tile
def _strand_tile(pal, rng, w, h, i, weave):
    """(h, w, 3) shading of plait strand i: crest highlight where it lies on top, dark where it passes under, creases along
    both edges, fibres along v. `weave` = dict(K crossings, phase deg, dz deg) as used by the geometry."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    ramp = TEX._grad(U, [(0.0, pal["light"]), (0.22, pal["base"]), (0.50, pal["base"] * 0.80 + pal["shadow"] * 0.20),
                         (0.80, pal["shadow"]), (1.0, pal["shadow"] * 0.7 + pal["deep"] * 0.3)])
    theta = np.radians(weave["phase"]) + np.pi / 3.0 * weave["K"] * V + 2.0 * np.pi * i / 3.0
    top = 0.5 * (1.0 + np.sin(2.0 * theta + np.radians(weave["dz"])) * -1.0)      # 1 = on top, 0 = right underneath
    under = _smooth((0.55 - top) / 0.55)
    col = _lerp(ramp, np.broadcast_to(pal["deep"], ramp.shape), 0.62 * under)
    edge = np.exp(-(((U - 0.50) / 0.075) ** 2))
    col = _lerp(col, np.broadcast_to(pal["deep"], col.shape), 0.62 * edge)
    ends = 1.0 - _smooth(np.minimum(V, 1.0 - V) / 0.035)
    col = _lerp(col, np.broadcast_to(pal["deep"], col.shape), 0.85 * ends)
    win = 1.0 - under
    hi = np.exp(-(((U - 0.10) / 0.085) ** 2)) * win
    col = _lerp(col, np.broadcast_to(pal["highlight"], col.shape), 0.55 * hi)
    hot = pal["highlight_core"] if "highlight_core" in pal else pal["highlight"]
    core = np.exp(-(((U - 0.06) / 0.03) ** 2)) * win ** 2
    col = _lerp(col, np.broadcast_to(hot, col.shape), 0.32 * core)
    S = TEX.streaks(rng, w, h, 1.1)
    return np.clip(col * S[..., None], 0.0, 1.0)




def braid_atlas(pal, rng, weave):
    """The braid atlas as (1024, 512, 4) uint8: the three strand tiles over hair_tex's strand region."""
    base = TEX.make_atlas(pal, rng)
    img = base[..., :3].astype(float) / 255.0
    rows = HEAD_ROWS
    for k in range(N_TILES):
        img[:rows, k * TILE:(k + 1) * TILE] = _strand_tile(pal, rng, TILE, rows, k % 3, weave)
    img[:rows, N_TILES * TILE:] = img[:rows, (N_TILES - 1) * TILE:(N_TILES - 1) * TILE + (W - N_TILES * TILE)]
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.rint(img * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


# ---------------------------------------------------------------- ribbon
RIBBON = {"base": "#17151a", "shade": "#0d0c0f", "sheen": "#3b363d"}


def ribbon_texture(colors=None, size=128, rng=None):
    """Black satin: near-black blue-grey with a soft sheen band along the middle of the loops and a darker, tighter
    gather towards the knot. u (columns) = along the loop / tail from the knot, v (rows, UP) = across it."""
    pal = dict(RIBBON)
    pal.update({k: v for k, v in (colors or {}).items() if k in RIBBON and isinstance(v, str)})
    base, shade, sheen = (TEX.srgb(pal[k]) for k in ("base", "shade", "sheen"))
    u = (np.arange(size) + 0.5) / size
    v = 1.0 - (np.arange(size) + 0.5) / size                      # row 0 = top = v 1
    U, V = np.meshgrid(u, v)
    col = np.broadcast_to(base, (size, size, 3)).copy()
    gather = 1.0 - _smooth(U / 0.26)                              # folds pinch together at the knot
    col = _lerp(col, np.broadcast_to(shade, col.shape), 0.78 * gather)
    band = np.exp(-(((V - 0.5) / 0.15) ** 2)) * _smooth((U - 0.10) / 0.28) * (1.0 - 0.45 * _smooth((U - 0.72) / 0.28))
    col = _lerp(col, np.broadcast_to(sheen, col.shape), 0.92 * band)
    hem = np.exp(-(((V - 0.06) / 0.022) ** 2)) + np.exp(-(((V - 0.94) / 0.022) ** 2))
    col = _lerp(col, np.broadcast_to(base * 1.55 + 0.012, col.shape), 0.38 * np.clip(hem, 0, 1) * _smooth((U - 0.05) / 0.2))
    edge = _smooth((np.abs(2 * V - 1) - 0.82) / 0.18)             # selvage falls into shade
    col = _lerp(col, np.broadcast_to(shade, col.shape), 0.55 * edge)
    if rng is not None:
        col = col * (1.0 + 0.012 * rng.normal(size=col.shape[:2]))[..., None]
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def ribbon_uv(u, v):
    """UV in the ribbon texture from loop coordinates u (0 = knot .. 1 = tip) and v (0 .. 1 across, 0.5 = middle)."""
    return np.stack([np.clip(u, 0.0, 1.0) * 0.96 + 0.02, np.clip(v, 0.0, 1.0) * 0.96 + 0.02], -1)
