"""What the garment parts share (pants, socks, the sweater): their colours, materials and tileable cloth textures
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


def knit(rgb, cols=10, rows=14, depth=0.16, size=SIZE, seed=3):
    """Stockinette: `cols` wales across a tile and `rows` courses down it. Each stitch is a V of two rounded legs, ovals
    in either half of the stitch leaning out at the top (`LEAN`), reaching a little into the stitches above and below so
    the courses interlock: lit along a leg's middle (+depth/2), dark between the legs and between the wales (-depth/2),
    a faint per-stitch variation. Tileable."""
    u, v = _grid(size)
    x = (u * cols) % 1.0
    y = (v * rows) % 1.0
    aspect = rows / cols                                       # a stitch's height over its width, in the tile
    light = np.zeros(np.broadcast(u, v).shape)
    lean = np.radians(LEAN)
    for dy in (-1.0, 0.0, 1.0):
        for cx, sg in ((0.28, 1.0), (0.72, -1.0)):             # left leg leans left at the top, right leg right
            px, py = x - cx, (y - (0.5 + dy)) / aspect           # in stitch widths, y down
            c, s = np.cos(sg * lean), np.sin(sg * lean)
            a, b = px * c - py * s, px * s + py * c              # across and along the leg
            rho = np.hypot(a / 0.23, b / (0.62 / aspect))
            light = np.maximum(light, np.sqrt(np.clip(1.0 - rho ** 2, 0.0, 1.0)))
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.008, (rows, cols))
    jj = jitter[(v * rows).astype(int) % rows, (u * cols).astype(int) % cols]
    shade = 1.0 - 0.5 * depth + depth * light + jj
    return _rgba(np.asarray(rgb, float)[None, None, :] * shade[..., None])


LEAN = 28.0                       # degrees a stockinette stitch's legs lean out from upright


def rib(rgb, ribs=8, depth=0.16, size=SIZE):
    """Vertical ribs (constant along v): ribs per tile, each a rounded wale (lit middle) and a dark groove between."""
    u, _ = _grid(size)
    x = (u * ribs) % 1.0
    wale = 0.5 + 0.5 * np.cos((x - 0.5) * 2.0 * np.pi)          # 1 at the wale's middle, 0 at the groove
    shade = 1.0 - depth + depth * wale ** 0.6
    shade = np.broadcast_to(shade, (size, size))
    return _rgba(np.asarray(rgb, float)[None, None, :] * shade[..., None])


def ink_edges(mask, rx=2, ry=1):
    """(h, w) bool: the pixels within the ellipse (rx, ry) pixels of the edge of the bool image `mask`, where it changes
    from in to out: its outline drawn as an ink line on a painted texture. x wraps round (a face painted round a
    shoe), y does not."""
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2 > 1.0:
                continue
            m = np.roll(mask, dx, axis=1)
            if dy:
                m = np.concatenate([m[:1].repeat(dy, 0), m[:-dy]]) if dy > 0 else \
                    np.concatenate([m[-dy:], m[-1:].repeat(-dy, 0)])
            out |= m != mask
    return out


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


def ink_material(ctx, name, name_en, rgb):
    """The material of a garment's drawn lines (outfit_ink strokes): white diffuse under a small texture of the ink
    colour (a flat colour without one washes out to a light grey under the studio's light, as the cloth would; through
    the texture it stays the dark ink it is), unshaded (no toon), no outline, casting no shadow, seen from both sides.
    The texture is written with ctx.save_png."""
    tex = ctx.save_png(name_en, _rgba(np.broadcast_to(np.asarray(rgb, float), (8, 8, 3))))
    return Material(name, name_en=name_en, diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=1.0,
                    ambient=(0.5, 0.5, 0.5), texture=tex, double_sided=True, edge=False, drop_shadow=False,
                    self_shadow=False, self_shadow_map=False)
