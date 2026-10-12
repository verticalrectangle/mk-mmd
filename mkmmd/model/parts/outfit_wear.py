"""What the garment parts cut from the skin share (outfit_shell): their colours, materials and tileable cloth textures
(numpy + PIL only). A garment part reads its colours from `[colors.<part>]` over its own defaults and draws its cloth as
flat colour with a little texture, outlined dark like a drawing: the toon ramp shades by a per-channel multiplier.

    colours(ctx, "sweater", {"knit": "#e7a7eb", "rib": "#9f75b0"})  -> {"knit": rgb, "rib": rgb}
    material(ctx, "セーター", "sweater", rgb, texture=knit(rgb), shadow=(0.78, 0.70, 0.86))
    knit(rgb), rib(rgb, ribs=8), twill(rgb)                            # (h, w, 4) uint8 tiles

Texture coordinates of the cloth are in tiles: a garment sets how many tiles go round it (`u_rep`) and per metre down
(`v_scale`), so a tile is about TILE metres either way."""
import numpy as np

from ..part import Material
from .hair_tex import srgb, toon_ramp

TILE = 0.06                       # metres of cloth per texture tile
SIZE = 256


def colours(ctx, table, defaults):
    """{key: rgb 0..1} from `[colors.<table>]` over `defaults` (hex strings); unknown keys are an error."""
    got = ctx.get(f"colors.{table}", {}) or {}
    bad = sorted(set(got) - set(defaults))
    if bad:
        raise KeyError(f"[colors.{table}]: unknown {', '.join(bad)} (have {', '.join(defaults)})")
    return {k: np.array(srgb(got.get(k, v)), float) for k, v in defaults.items()}


def _rgba(rgb):
    a = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.rint(a * 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


def _grid(size=SIZE):
    t = (np.arange(size) + 0.5) / size
    return t[None, :], t[:, None]                              # u across (x), v down (y)


def knit(rgb, cols=10, rows=14, depth=0.05, size=SIZE, seed=3):
    """Stockinette: columns of small V stitches, each a slightly brighter centre and darker sides (+-depth of the
    value), a faint per-stitch variation. Tileable."""
    u, v = _grid(size)
    x = (u * cols) % 1.0
    y = (v * rows + 0.5 * ((u * cols) // 1 % 2)) % 1.0
    vee = np.abs(x - 0.5) * 2.0                                # 0 at the column centre, 1 at its sides
    stitch = 1.0 - depth * (vee ** 2) - 0.5 * depth * np.abs(y - 0.5) * 2.0
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.012, (rows, cols))
    jj = jitter[(v * rows).astype(int) % rows, (u * cols).astype(int) % cols]
    return _rgba(np.asarray(rgb, float)[None, None, :] * (stitch + jj)[..., None])


def rib(rgb, ribs=8, depth=0.16, size=SIZE):
    """Vertical ribs (constant along v): ribs per tile, each a rounded wale (lit middle) and a dark groove between."""
    u, _ = _grid(size)
    x = (u * ribs) % 1.0
    wale = 0.5 + 0.5 * np.cos((x - 0.5) * 2.0 * np.pi)          # 1 at the wale's middle, 0 at the groove
    shade = 1.0 - depth + depth * wale ** 0.6
    shade = np.broadcast_to(shade, (size, size))
    return _rgba(np.asarray(rgb, float)[None, None, :] * shade[..., None])


def twill(rgb, lines=18, depth=0.035, size=SIZE, seed=5):
    """Plain woven cloth: faint diagonal twill lines and a little noise. Tileable."""
    u, v = _grid(size)
    d = ((u + v) * lines) % 1.0
    shade = 1.0 - depth * (0.5 + 0.5 * np.cos(d * 2.0 * np.pi))
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 0.01, (size, size))
    return _rgba(np.asarray(rgb, float)[None, None, :] * (shade + n)[..., None])


def outline(rgb, k=0.22):
    """The outline colour of a cloth: its own colour, very dark (a drawing's ink, never pure black)."""
    c = np.clip(np.asarray(rgb, float) * k, 0.03, 1.0)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def material(ctx, name, name_en, rgb, texture=None, shadow=(0.80, 0.74, 0.86), edge_size=0.9, specular=(0.0, 0.0, 0.0),
             shininess=4.0, ambient=(0.5, 0.5, 0.5), double_sided=True):
    """A cloth Material: white diffuse under the texture, or the colour itself without one (then the ambient is that
    share of the colour, so a flat dark violet lights like a textured cloth instead of washing out to lilac), a toon ramp
    whose shaded side is `shadow` (per-channel multiplier) and a dark outline in the cloth's own hue. Files are written
    with ctx.save_png (prefixed by the part's name)."""
    tex = ctx.save_png(name_en, texture) if texture is not None else ""
    toon = ctx.save_png(f"{name_en}_toon", toon_ramp(shadow=tuple(shadow)))
    diffuse = (1.0, 1.0, 1.0, 1.0) if tex else (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    amb = tuple(float(a) for a in ambient) if tex else tuple(float(a * c) for a, c in zip(ambient, rgb))
    return Material(name, name_en=name_en, diffuse=diffuse, specular=tuple(specular), shininess=float(shininess),
                    ambient=amb, texture=tex, toon=toon, double_sided=double_sided, edge=True,
                    edge_color=outline(rgb), edge_size=float(edge_size))
