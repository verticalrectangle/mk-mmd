"""The lock hairdo's texture (numpy only): a cell of the atlas for every lock, painted along its length.

Every lock, and the scalp cap, owns a rectangle of the atlas: across it the lock's width (u), down it the lock from
root to tip (v). What is drawn on it follows the lock, as a painter of hair textures draws each lock on its own, but
whatever belongs to the head as a whole (the colour by height, the sheen) is told by where each point of the lock
is, so locks lying side by side carry one picture:

    colour    by height, down a ramp (`ramp`: [[share of the way down the hair, colour], ...], a colour a name of the
              hair's family, a mix "base:tip:0.4" or "#rrggbb"). By default the root's shade at the top of the hair,
              the base, the tip's colour at its lowest (`[colors.hair] tip` makes hair of two colours); light at
              the top and the shadow colour below draws hair lit from above
    form      where the lock hangs free: a little lighter along the middle, its edges shaded (the left more than the
              right) and a darker line along them: what parts a lock from its neighbours. Where the locks lie together
              on the head there is none of it, and they read as one surface. `root` shades where the hair comes out of
              the head, `tip` its point
    strands   streaks running down the lock, coarse and fine, wavering a little, and a few fine dark lines in its free
              part
    sheen     the crown ring (`Sheen`): one band round the head, toothed above and below, `tooth` metres a tooth,
              brighter at its heart; each lock paints its piece of it; faint on the rows under. `bands` are more
              such bands of any colour: `[{drop, width, colour, strength, tooth}, ...]`, `drop` metres below the
              hair's top (a second sheen, a toothed shade under the first: how a cel shades hair)
    tips      `{colour, length, strength}`: every lock's last `length` metres go to `colour` (hair that ends in
              another colour, each lock on its own, not by its height)
    under     the rows under (`under`) a shade darker

`[hair.paint]` sets how strong each is (PAINT), `hard` how crisp (0 airbrushed, 1 cut like a cel). `Atlas` packs the
cells and holds the picture; a lock's UVs come from its cell (`Cell.uv`)."""
from dataclasses import dataclass

import numpy as np

from .hair_tex import _grad, _noise1, _sm

PAINT = {"strands": 0.10, "lines": 5, "line": 0.22, "edge": 0.30, "edge_width": 0.34, "rim": 0.30, "form": 0.05,
         "root": 0.12, "tip": 0.14, "sheen": 0.85, "sheen_width": 1.0, "glow": 0.30, "tooth": 0.012, "hard": 0.0,
         "under": 0.84, "cap": 0.80, "bands": [], "tips": {"colour": None, "length": 0.06, "strength": 0.85},
         "ramp": [[0.0, "shadow:base:0.5"], [0.22, "base"], [0.62, "base:tip:0.4"], [1.0, "tip"]]}
BAND = {"drop": 0.10, "width": 0.015, "colour": "shadow", "strength": 0.6, "tooth": 0.012}
ROOT_SHADE = 0.02               # metres from the root over which `root` fades
TIP_SHADE = 0.04                # metres before the point over which `tip` grows
HEAD_R = 0.15                   # metres: the radius the sheen's teeth are counted round
PX_ACROSS = 0.0008              # metres a pixel is across a lock, at scale 1
PX_ALONG = 0.0011               # ... and along it
PAD = 3                         # pixels round a cell, filled with its own edge
W_PX = (20, 256)                # a cell is no narrower or wider
H_PX = (48, 2000)               # ... no shorter or longer
SCALE = 3.0                     # the cells start this large and shrink together until they fit the atlas


@dataclass
class Cell:
    """A lock's rectangle in the atlas: its top left pixel, its size, the atlas's side."""
    x: int
    y: int
    w: int
    h: int
    side: int

    def uv(self, q, t):
        """UVs (v up) of points `q` across the lock (-1 its left edge .. 1 its right) and `t` along it (0 the root .. 1
        the tip)."""
        u = (self.x + 0.5 + (np.asarray(q, float) + 1.0) * 0.5 * (self.w - 1)) / self.side
        v = (self.y + 0.5 + np.clip(np.asarray(t, float), 0.0, 1.0) * (self.h - 1)) / self.side
        return np.stack([u, 1.0 - v], -1)


def pack(sizes, side):
    """[(x, y)] top left pixels of rectangles `sizes` [(w, h)] (each padded by PAD) shelved into a square of `side`
    pixels, tallest first; None when they do not fit."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    out = [None] * len(sizes)
    x, y, row = 0, 0, 0
    for i in order:
        w, h = sizes[i][0] + 2 * PAD, sizes[i][1] + 2 * PAD
        if w > side:
            return None
        if x + w > side:
            x, y, row = 0, y + row, 0
        if y + h > side:
            return None
        out[i] = (x + PAD, y + PAD)
        x, row = x + w, max(row, h)
    return out


class Atlas:
    """The picture every cell is painted into. `cells(sizes, fixed)` takes the locks' sizes in metres [(across, along)]
    and cells of a given size in pixels [(w, h)] and gives each its Cell: (the locks', the fixed ones'), the lot scaled
    down together when it would not fit `side` pixels."""

    def __init__(self, side=2048):
        self.side = int(side)
        self.img = np.zeros((self.side, self.side, 3))
        self.used = np.zeros((self.side, self.side), bool)

    def cells(self, sizes, fixed=()):
        scale = SCALE
        for _ in range(40):
            px = [(int(np.clip(round(a / PX_ACROSS * scale), *W_PX)), int(np.clip(round(b / PX_ALONG * scale), *H_PX)))
                  for a, b in sizes]
            px += [(max(int(round(w * scale)), 8), max(int(round(h * scale)), 8)) for w, h in fixed]
            at = pack(px, self.side)
            if at is not None:
                made = [Cell(x, y, w, h, self.side) for (x, y), (w, h) in zip(at, px)]
                return made[:len(sizes)], made[len(sizes):]
            scale *= 0.92
        raise ValueError(f"hair atlas: {len(sizes) + len(fixed)} cells do not fit {self.side} px a side")

    def put(self, cell, rgb):
        """`rgb` (h, w, 3) into the cell, its edge pixels drawn on PAD pixels round it (no neighbour bleeds in)."""
        big = np.pad(np.clip(rgb, 0.0, 1.0), ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
        ys, xs = slice(cell.y - PAD, cell.y + cell.h + PAD), slice(cell.x - PAD, cell.x + cell.w + PAD)
        self.img[ys, xs] = big
        self.used[ys, xs] = True

    def image(self, fill):
        """(side, side, 4) uint8: the cells, `fill` (rgb 0..1) between them."""
        out = np.empty((self.side, self.side, 4), np.uint8)
        out[..., :3] = np.rint(np.where(self.used[..., None], self.img, np.asarray(fill, float)) * 255).astype(np.uint8)
        out[..., 3] = 255
        return out


def colour(pal, name):
    """A ramp colour: a name of the hair's family (`pal`), "a:b:t" (t of the way from a to b, 0.5 without it) or
    "#rrggbb"."""
    name = str(name)
    if name.startswith("#") and len(name) == 7:
        return np.array([int(name[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    parts = name.split(":")
    if len(parts) in (2, 3) and parts[0] in pal and parts[1] in pal:
        t = float(parts[2]) if len(parts) == 3 else 0.5
        return pal[parts[0]] * (1.0 - t) + pal[parts[1]] * t
    if name in pal:
        return pal[name]
    raise ValueError(f"[hair.paint] ramp: colour {name!r} is no colour of the hair's family ({', '.join(sorted(pal))}), "
                     f"no mix \"a:b:t\" of two and no \"#rrggbb\"")


def height_colour(pal, f, ramp):
    """The hair's colour at height fractions `f` (0 the top of the hair .. 1 its lowest) down `ramp`: (n, 3)."""
    stops = sorted((float(at), colour(pal, c)) for at, c in ramp)
    return _grad(f, stops)


def _tone(col, shade, pal):
    """`col` (.., 3) shaded by `shade` (..): under 1 toward the family's shadow, over 1 toward its light."""
    base = np.maximum(pal["base"], 1e-3)
    dark = col * np.clip(pal["shadow"] / base, 0.0, 1.0)
    lite = np.clip(col * np.clip(pal["light"] / base, 1.0, 3.0), 0.0, 1.0)
    k = np.clip(1.0 - shade, 0.0, 1.0)[..., None] * 1.6
    j = np.clip(shade - 1.0, 0.0, 1.0)[..., None] * 3.0
    return col + (dark - col) * np.clip(k, 0.0, 1.0) + (lite - col) * np.clip(j, 0.0, 1.0)


def _edge(t, hard):
    """A soft step of t (0 .. 1) that `hard` (0 .. 1) makes crisp."""
    w = 0.5 * (1.0 - 0.9 * hard)
    return _sm((t - (0.5 - w)) / (2.0 * w))


class Sheen:
    """A band round the head at height `z`, `width` its half thickness in metres, toothed above and below (a tooth
    `tooth` metres wide round a head of HEAD_R), its height and its strength wandering as it goes round; `glow` a soft
    halo round it. The crown's sheen is one, a toothed shade under it another. Every lock that crosses it paints its
    piece of the same band (`on`), so the band runs unbroken over locks that lie side by side."""

    def __init__(self, z, width, tooth, glow, rng):
        self.z, self.hw, self.glow = float(z), float(width), float(glow)
        self.grid = np.linspace(-np.pi, np.pi, 2049)

        def table(waves):                                   # smooth noise in -1 .. 1 that closes round the head
            k = np.arange(1, waves + 1)
            amp, ph = rng.uniform(0.3, 1.0, waves) / np.sqrt(k), rng.uniform(0.0, 2.0 * np.pi, waves)
            return np.sin(np.multiply.outer(self.grid, k) + ph) @ amp / amp.sum()
        self.wander, self.uneven, self.ragged = table(7), table(11), table(23)
        self.teeth = max(int(round(2.0 * np.pi * HEAD_R / max(float(tooth), 1e-3))), 3)
        self.shift = rng.uniform(0.0, 1.0, 2)

    def on(self, z, around, hard):
        """(the band, its heart, its glow, its strength there), each 0 .. 1 and shaped as `z`: at heights `z` and
        angles round the head `around` (radians)."""
        th = (np.asarray(around, float) + np.pi) % (2.0 * np.pi) - np.pi
        turn = th / (2.0 * np.pi)
        ragged = np.interp(th, self.grid, self.ragged)
        up = np.abs(2.0 * ((turn * self.teeth + 0.8 * ragged + self.shift[0]) % 1.0) - 1.0)
        down = np.abs(2.0 * ((turn * round(1.7 * self.teeth) + 1.3 * ragged + self.shift[1]) % 1.0) - 1.0)
        dz = np.asarray(z, float) - (self.z + 0.35 * self.hw * np.interp(th, self.grid, self.wander))
        d = np.where(dz >= 0.0, dz / (self.hw * (0.35 + 0.65 * up)), -dz / (self.hw * (0.25 + 0.45 * down)))
        glow = self.glow * np.exp(-(dz / (1.6 * self.hw)) ** 2) * (1.0 - hard)
        return _edge(1.0 - d, hard), _edge(1.0 - d / 0.5, hard), glow, 0.65 + 0.35 * np.interp(th, self.grid, self.uneven)


def paint_lock(cell, z, around, arc, free, pal, look, rng, z_top, z_bot, sheen=None, under=False, bands=(), ends=True):
    """The picture (h, w, 3) of a lock's cell. `z` and `around` (h, w): each point's height and its angle round the
    head (radians: 0 the front, her left positive); `arc` (h,) the metres from the root down the lock and `free` (h,)
    how free it hangs there (0 lying with its neighbours .. 1 on its own); `sheen` the crown's Sheen (None: none);
    `bands` [(a Sheen, its colour, its strength)] drawn under it; `ends` False for a lock that ends in a tie: it has
    no point of its own (no `tip` shade, no `tips` colour)."""
    h, w = cell.h, cell.w
    x = np.linspace(-1.0, 1.0, w)
    hard = float(look["hard"])
    col = height_colour(pal, np.clip((z_top - np.asarray(z, float)) / max(z_top - z_bot, 1e-6), 0.0, 1.0), look["ramp"])
    arc = np.asarray(arc, float)
    tips = look["tips"]
    if tips["colour"] and ends:
        t = _sm((arc - (arc[-1] - float(tips["length"]))) / max(float(tips["length"]), 1e-6))[:, None, None]
        col = col + (colour(pal, tips["colour"]) - col) * (float(tips["strength"]) * t)
    free = np.asarray(free, float)[:, None]
    shade = 1.0 + look["form"] * free * (1.0 - x * x)[None, :]
    shade = shade - look["root"] * (1.0 - _sm(arc / ROOT_SHADE))[:, None]
    shade = shade - (look["tip"] if ends else 0.0) * _sm((arc - (arc[-1] - TIP_SHADE)) / TIP_SHADE)[:, None]
    # streaks: two draws of noise across, one wavering into the other down the lock
    a = 0.6 * _noise1(w, rng, 14.0) + 0.4 * _noise1(w, rng, 4.0)
    b = 0.6 * _noise1(w, rng, 14.0) + 0.4 * _noise1(w, rng, 4.0)
    mix = (0.5 + 0.5 * _noise1(h, rng, 160.0))[:, None]
    shade = shade + look["strands"] * (a[None, :] * (1.0 - mix) + b[None, :] * mix)
    # the free lock's edges: a shade (the left side deeper) and a line along them
    ax = np.abs(x)
    side = np.where(x < 0.0, 1.0, 0.55)
    wide = _edge((ax - (1.0 - look["edge_width"])) / max(look["edge_width"], 1e-6), hard * 0.6)
    shade = shade - look["edge"] * free * (wide * side)[None, :]
    rim_px = max(1.5, 0.035 * w)
    rim = np.clip(1.0 - (1.0 - ax) * 0.5 * w / rim_px, 0.0, 1.0)
    shade = shade - look["rim"] * free * rim[None, :]
    rows = (np.arange(h) + 0.5) / h
    for _ in range(int(look["lines"])):
        x0, r0 = rng.uniform(-0.75, 0.75), rng.uniform(0.25, 0.75)
        across = np.clip(1.0 - np.abs(x - x0) * 0.5 * w / max(1.3, 0.02 * w), 0.0, 1.0)
        along = _sm((rows - r0) / 0.15)[:, None] * np.maximum(free, 0.25)
        shade = shade - look["line"] * along * across[None, :]
    rgb = _tone(col, shade, pal)
    for band, tint, k in bands:
        on = band.on(z, around, hard)[0] * (0.4 if under else 1.0)
        rgb = rgb + (np.asarray(tint, float) - rgb) * np.clip(float(k) * on, 0.0, 1.0)[..., None]
    if sheen is not None and look["sheen"] > 0.0:
        band, core, glow, strength = sheen.on(z, around, hard)
        amp = look["sheen"] * (0.3 if under else 1.0) * strength * (1.0 - 0.7 * free * rim[None, :])
        rgb = rgb + (pal["ring"] - rgb) * np.clip(amp * np.maximum(band, glow), 0.0, 1.0)[..., None]
        rgb = rgb + (pal["ring_core"] - rgb) * np.clip(0.55 * amp * core, 0.0, 1.0)[..., None]
    if under:
        rgb = _tone(rgb, np.full((h, w), look["under"]), pal)
    return rgb


def paint_cap(cell, pal, look, rng):
    """The scalp cap's cell: the hair's colour by height down it (row 0 the top of the hair), a shade darker, with
    streaks that radiate from the crown (the cap's u is the way round the head)."""
    h, w = cell.h, cell.w
    f = (np.arange(h) + 0.5) / h
    col = height_colour(pal, f, look["ramp"])[:, None, :]
    shade = look["cap"] * (1.0 + look["strands"] * (0.6 * _noise1(w, rng, 9.0) + 0.4 * _noise1(w, rng, 3.0)))[None, :]
    return _tone(np.broadcast_to(col, (h, w, 3)), shade * np.ones((h, 1)), pal)
