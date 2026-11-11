"""What the garment parts share (pants, socks, the sweater): their colours, materials and tileable cloth textures
(numpy + PIL only). A garment part reads its colours from `[colors.<part>]` over its own defaults and draws its cloth as
flat colour with a little texture, outlined dark like a drawing: the toon ramp shades by a per-channel multiplier.

    colours(ctx, "sweater", {"knit": "#e7a7eb", "rib": "#9f75b0"})  -> {"knit": rgb, "rib": rgb}
    stops(ctx, "sweater", "knit")                                      # [(share, rgb)] when the colour is a gradient
    material(ctx, "セーター", "sweater", rgb, texture=knit(rgb), shadow=(0.78, 0.70, 0.86))
    knit(rgb), rib(rgb, ribs=8), twill(rgb), lace(rgb)                 # (h, w, 4) uint8 tiles
    knit_ramp(stops, tiles)                                            # a knit coloured down a gradient
    rib_bands([rgb, ...]), band_v(v, k, n)                             # ribs of several colours in one picture

Texture coordinates of the cloth are in tiles: a garment sets how many tiles go round it (`u_rep`) and per metre down
(`v_scale`), so a tile is about TILE metres either way. A colour may be a gradient down the garment: stops
`[[share, "#rrggbb"], ...]` (0 its top .. 1 its hem) instead of one "#rrggbb"; `colours` then gives its middle colour
(what the outline and the shading follow) and `stops` the gradient. Texture coordinates run v up: v 1 is a picture's
top row."""
import numpy as np

from ..part import Material
from .hair_tex import srgb, toon_ramp

TILE = 0.06                       # metres of cloth per texture tile
SIZE = 256


def gradient(value):
    """[(share, rgb)] of a colour given as stops [[share, "#rrggbb"], ...], sorted; None for a plain "#rrggbb"."""
    if not isinstance(value, (list, tuple)):
        return None
    try:
        out = sorted((float(at), np.array(srgb(c), float)) for at, c in value)
    except (TypeError, ValueError):
        out = []
    if len(out) < 2:
        raise ValueError(f"a gradient is [[share, \"#rrggbb\"], ...] with two stops or more, got {value!r}")
    return out


def at(stops, t):
    """The colour (.., 3) of the gradient `stops` at shares `t` (..)."""
    t = np.asarray(t, float)
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops])
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)


def colours(ctx, table, defaults):
    """{key: rgb 0..1} from `[colors.<table>]` over `defaults` (hex strings; None: a key that may be left out, then
    None); unknown keys are an error. A gradient's colour is the one at its middle."""
    got = ctx.get(f"colors.{table}", {}) or {}
    bad = sorted(set(got) - set(defaults))
    if bad:
        raise KeyError(f"[colors.{table}]: unknown {', '.join(bad)} (have {', '.join(defaults)})")
    out = {}
    for k, v in defaults.items():
        v = got.get(k, v)
        ramp = gradient(v)
        out[k] = None if v is None else at(ramp, 0.5) if ramp else np.array(srgb(v), float)
    return out


def stops(ctx, table, key):
    """The gradient [(share, rgb)] `[colors.<table>] <key>` is given as, or None (one colour, or none)."""
    return gradient((ctx.get(f"colors.{table}", {}) or {}).get(key))


def _rgba(rgb):
    a = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.rint(a * 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


def _grid(size=SIZE):
    t = (np.arange(size) + 0.5) / size
    return t[None, :], t[:, None]                              # u across (x), v down (y)


def knit_shade(cols=10, rows=14, depth=0.16, size=SIZE, seed=3):
    """The light and shade (size, size) of a tile of stockinette, about 1: `cols` wales across a tile and `rows`
    courses down it. Each stitch is a V of two rounded legs, ovals in either half of the stitch leaning out at the top
    (`LEAN`), reaching a little into the stitches above and below so the courses interlock: lit along a leg's middle
    (+depth/2), dark between the legs and between the wales (-depth/2), a faint per-stitch variation. Tileable."""
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
    return 1.0 - 0.5 * depth + depth * light + jj


def knit(rgb, cols=10, rows=14, depth=0.16, size=SIZE, seed=3):
    """A tile of stockinette (`knit_shade`) in one colour."""
    return _rgba(np.asarray(rgb, float)[None, None, :] * knit_shade(cols, rows, depth, size, seed)[..., None])


def knit_ramp(ramp, tiles, size=128, **kw):
    """Stockinette coloured down a gradient: `tiles` tiles of knit one over the other (tileable across), the gradient
    `ramp` [(share, rgb)] running from v 0 (the picture's bottom row: the garment's top, whose v starts at 0) to v 1."""
    shade = np.tile(knit_shade(size=size, **kw), (max(int(tiles), 1), 1))
    share = 1.0 - (np.arange(len(shade)) + 0.5) / len(shade)
    return _rgba(at(ramp, share)[:, None, :] * shade[..., None])


LEAN = 28.0                       # degrees a stockinette stitch's legs lean out from upright


def rib_bands(colours, lines=(), ribs=8, depth=0.16, size=SIZE, rows=48):
    """Vertical ribs (constant along v: ribs per tile, each a rounded wale, lit in its middle, and a dark groove between)
    in several colours: a band of `rows` rows for each of `colours`, the first at v 0 (the picture's bottom), so the rib
    pieces of a garment share one material, each keeping its v inside its band (`band_v`). `lines`: [(v0, v1, rgb)]
    lines drawn across the ribs between those v (a collar's dark edge)."""
    u = (np.arange(size) + 0.5) / size
    wale = 0.5 + 0.5 * np.cos(((u * ribs) % 1.0 - 0.5) * 2.0 * np.pi)   # 1 at the wale's middle, 0 at the groove
    shade = (1.0 - depth + depth * wale ** 0.6)[None, :, None]
    n = len(colours)
    v = 1.0 - (np.arange(n * rows) + 0.5) / (n * rows)
    rgb = np.array([np.asarray(colours[min(int(x * n), n - 1)], float) for x in v])[:, None, :] * shade
    for v0, v1, col in lines:
        rgb[(v >= v0) & (v <= v1)] = np.asarray(col, float)[None, None, :] * shade
    return _rgba(rgb)


def band_v(v, k, n):
    """A rib piece's v (any range, per ring) laid inside band `k` of `n` of a `rib_bands` picture, clear of its
    neighbours."""
    v = np.asarray(v, float)
    return (k + 0.12 + 0.76 * (v - v.min()) / max(float(np.ptp(v)), 1e-9)) / n


def rib(rgb, ribs=8, depth=0.16, size=SIZE):
    """A tile of vertical ribs in one colour (`rib_bands`)."""
    return rib_bands([rgb], ribs=ribs, depth=depth, size=size, rows=size)


def lace(rgb, scallops=4, size=SIZE):
    """A tile of lace trim, v 1 (the top row) where it is sewn on and v 0 its hem: a fine net, a row of eyelet rings and
    a scalloped border drawn along the hem, all a shade of the one colour (opaque). `scallops` to a tile."""
    u, y = _grid(size)                                         # y down the tile: 0 where it is sewn on, 1 the hem
    net = np.minimum(np.abs(np.sin(np.pi * 14.0 * (u + y))), np.abs(np.sin(np.pi * 14.0 * (u - y)))) < 0.16
    x = (u * scallops) % 1.0 - 0.5                             # across one scallop, -0.5 .. 0.5
    arc = 0.97 - 0.30 * np.sqrt(np.clip(1.0 - (2.0 * x) ** 2, 0.0, 1.0))    # the scallop's edge: lowest between two
    border = np.abs(y - arc + 0.10) < 0.022
    ring = np.abs(np.hypot(x * 1.0, (y - 0.60) * scallops * 0.5) - 0.19) < 0.035
    dots = np.hypot(((u * scallops * 3.0) % 1.0) - 0.5, ((y * 6.0) % 1.0) - 0.5) < 0.16
    shade = 1.0 - 0.07 * (net & (y < 0.45)) - 0.16 * border - 0.14 * ring - 0.08 * (dots & (y < 0.4))
    shade = shade - 0.10 * np.clip((y - arc) / 0.03, 0.0, 1.0)             # under the scallops: the shade between them
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
