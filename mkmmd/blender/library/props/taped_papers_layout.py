"""Layout of the taped papers (pure Python + numpy, no bpy: importable from tests): the pieces, their sheets, curls,
tape and prints, the options and the card.

A loose cluster of flat paper things taped to a wall: printed posters, instant photos, a paper crescent moon, a five-point
paper star, a plain coloured card or a heart, each held by one or two strips of masking tape at a jaunty angle, one
corner curling off the wall. Real sizes: a poster 300 x 400 mm (a little under A3), an instant photo of the Polaroid 600
kind 88 x 107 mm with a 79 x 79 mm picture 6 mm under its top edge and 4.5 mm in from its sides, masking tape 19 mm
(3/4 inch) wide torn to 45-65 mm, card stock A6 (148 x 105 mm); the moon and the star cut from construction paper
150-220 mm across (`mk ref photos 'polaroid photos wall'`, `'masking tape poster wall'`).

Frame (metres, "wall_center", as `cafe_poster`'s): the wall is the plane y = 0, the room at -Y, +Z up, +X to the right
as seen from the room. A piece lies in the prop's x-z plane at `at` = [x, z], turned `rot` degrees anticlockwise (as
seen from the room); piece i lies LAYER + i STACK in front of the wall (later pieces over earlier ones), its tape TAPE_LIFT
in front of what it covers, so nothing z-fights. The default layout spreads seven pieces over about 1.4 x 0.8 m round an
empty gap at the origin (put the prop so the gap is behind a face); the card's `use.look` `gap` is the middle of the
biggest empty circle among the pieces, whatever the layout.

`pieces = [{kind, at = [x, z], size, rot, color, ...}, ...]`:
- `poster`: size [w, h] ([0.30, 0.40]); `design` "sun" (a big sun with slits over sunset stripes, an ink frame), "heart",
  "stars" (a star field round a big star), "rainbow"; `colors` (that design's roles, in its order: DESIGNS); `color` the
  paper (surface).
- `polaroid`: size the frame's width (0.088; it keeps its 88 x 107 shape); `color` the picture's field ("foam"), `ink`
  its shape ("love"), `photo` the shape "heart", "sun", "moon", "star".
- `moon`: a paper crescent, size its disc's diameter (0.20); `color` ("gold").
- `star`: a five-point paper star, size tip to tip across (0.16); `color` ("love").
- `card`: size [w, h] ([0.148, 0.105]) or with `shape` "heart" its width (0.15); `color` ("rose").
- every piece: `rot` (0), `tape` (strips: 0, 1 or 2; posters 2, the rest 1), `curl` (m the curling corner lifts:
  posters 0.010, the rest 0.004-0.006), `corner` (which corner curls: "TL", "TR", "BL", "BR"; default a bottom one).

Options (`slots`): pieces (the default layout), seed (1: the tape's tears and angles), tone (0.04: a little tone between
pieces). Colour roles: tape ("overlay:gold:0.25"), paper ("surface": the ground of a poster that names none, the
polaroids' frames), line ("text": the posters' frame lines).

Card: origin "wall_center", front "-Y", blocks false; `use.look` `gap` (with `radius`) and `centre` (the middle of the
cluster), `use.surface` `poster<i>` for each poster (its sheet, for [[text]]), `pieces` (kind, at, size, rot of each)."""
import math
from collections import namedtuple

import numpy as np

from ....core import burst as BU
from ....core import hearts as HR
from ....core import shell as S
from . import papercraft as PC
from . import showkit as SK

KINDS = ("poster", "polaroid", "moon", "star", "card")
PHOTOS = ("heart", "sun", "moon", "star")
DESIGNS = {"sun": ("gold", "love", "rose", "iris"), "heart": ("rose", "love", "surface", "gold"),
           "stars": ("pine", "gold", "surface", "foam"), "rainbow": ("foam", "love", "gold", "pine", "iris", "surface")}
ROLES = {"tape": "overlay:gold:0.25", "paper": "surface", "line": "text"}
MATS = ("paper", "tape")                    # the printed pieces take materials of their own
POLAROID = (0.088, 0.107, 0.079, 0.006)     # frame width, height, the picture's side, its top margin
LAYER, STACK = 0.0006, 0.0009               # piece i lies LAYER + i * STACK in front of the wall
CURL = {"poster": 0.010, "polaroid": 0.004, "moon": 0.006, "star": 0.005, "card": 0.006}
SIZE = {"poster": [0.30, 0.40], "polaroid": 0.088, "moon": 0.20, "star": 0.16, "card": [0.148, 0.105]}
COLOR = {"polaroid": "foam", "moon": "gold", "star": "love", "card": "rose"}
PPM = {"poster": 2400.0, "polaroid": 4000.0}      # print pixels per metre
DEFAULT = [
    {"kind": "poster", "design": "sun", "at": [-0.53, 0.03], "rot": -2.5},
    {"kind": "polaroid", "photo": "heart", "color": "foam", "ink": "love", "at": [-0.25, 0.27], "rot": 7.0},
    {"kind": "moon", "color": "gold", "at": [0.04, 0.31], "rot": -14.0},
    {"kind": "star", "color": "love", "at": [0.29, 0.22], "rot": 10.0},
    {"kind": "poster", "design": "stars", "size": [0.24, 0.32], "at": [0.56, -0.02], "rot": 3.0, "corner": "BL"},
    {"kind": "card", "shape": "heart", "color": "rose", "at": [0.27, -0.25], "rot": -9.0, "size": 0.15},
    {"kind": "polaroid", "photo": "sun", "color": "iris", "ink": "gold", "at": [-0.26, -0.25], "rot": -6.0},
]
Piece = namedtuple("Piece", "i kind at rot size color ink design colors photo shape tape curl corner")


# ----------------------------------------------------------------------------------------------------------- options
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def piece(i, spec, slots):
    """One checked piece (a bad entry is a ValueError naming it and what to change)."""
    P = f"taped_papers: pieces[{i}]"
    if not isinstance(spec, dict):
        raise ValueError(f"{P} is a table {{kind, at = [x, z], ...}} (got {spec!r})")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{P}: kind is {' or '.join(repr(k) for k in KINDS)} (got {kind!r})")
    at = spec.get("at")
    if not (isinstance(at, (list, tuple)) and len(at) == 2 and all(_num(c) for c in at)):
        raise ValueError(f"{P}: at is [x, z], metres in the wall's plane (got {at!r})")
    shape = spec.get("shape", "rect")
    if kind == "card" and shape not in ("rect", "heart"):
        raise ValueError(f"{P}: a card's shape is 'rect' or 'heart' (got {shape!r})")
    size = spec.get("size", 0.15 if kind == "card" and shape == "heart" else SIZE[kind])
    pair = kind == "poster" or (kind == "card" and shape == "rect")
    if pair:
        if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(_num(c) and 0.03 <= c <= 2.0 for c in size)):
            raise ValueError(f"{P}: size is [width, height] in metres, 0.03 .. 2 (got {size!r})")
        size = (float(size[0]), float(size[1]))
    elif not (_num(size) and 0.03 <= size <= 2.0):
        raise ValueError(f"{P}: size is a width in metres, 0.03 .. 2 (got {size!r})")
    design = spec.get("design", "sun")
    if kind == "poster" and design not in DESIGNS:
        raise ValueError(f"{P}: design is {' or '.join(repr(d) for d in DESIGNS)} (got {design!r})")
    photo = spec.get("photo", "heart")
    if kind == "polaroid" and photo not in PHOTOS:
        raise ValueError(f"{P}: photo is {' or '.join(repr(d) for d in PHOTOS)} (got {photo!r})")
    colors = list(spec.get("colors", DESIGNS.get(design, ()))) if kind == "poster" else []
    if kind == "poster" and len(colors) != len(DESIGNS[design]):
        raise ValueError(f"{P}: colors of the design {design!r} are {len(DESIGNS[design])} roles "
                         f"({', '.join(DESIGNS[design])}) (got {colors!r})")
    color = spec.get("color", None if kind == "poster" else COLOR[kind])        # a poster without one: the paper role
    ink = spec.get("ink", "love")
    for what, c in [("color", color), ("ink", ink)] + [(f"colors[{k}]", c) for k, c in enumerate(colors)]:
        if c is None and what == "color":
            continue
        if not isinstance(c, str):
            raise ValueError(f"{P}: {what} is a palette slot, #hex or 'a:b:t' mix (got {c!r})")
        SK.rgb(c, slots, f"{P}: {what}")
    tape = spec.get("tape", 2 if kind == "poster" else 1)
    if tape not in (0, 1, 2) or isinstance(tape, bool):
        raise ValueError(f"{P}: tape is 0, 1 or 2 strips (got {tape!r})")
    curl = spec.get("curl", CURL[kind])
    if not (_num(curl) and 0.0 <= curl <= 0.05):
        raise ValueError(f"{P}: curl is metres a corner lifts off the wall, 0 .. 0.05 (got {curl!r})")
    corner = spec.get("corner", "BR" if i % 2 == 0 else "BL")
    if corner not in ("TL", "TR", "BL", "BR"):
        raise ValueError(f"{P}: corner is 'TL', 'TR', 'BL' or 'BR' (got {corner!r})")
    rot = spec.get("rot", 0.0)
    if not _num(rot):
        raise ValueError(f"{P}: rot is degrees (got {rot!r})")
    return Piece(i, kind, (float(at[0]), float(at[1])), float(rot), size, color, ink, design, colors, photo, shape,
                 int(tape), float(curl), corner)


def options(slots):
    """(pieces, seed, tone) from the `slots` of the [[prop]], checked."""
    slots = slots or {}
    specs = slots.get("pieces", DEFAULT)
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError(f"taped_papers: pieces is a list of tables {{kind, at = [x, z], ...}} (got {specs!r})")
    return ([piece(i, s, slots) for i, s in enumerate(specs)], SK.number(slots, "seed", 1, "taped_papers", 0, 1e9, whole=True),
            SK.number(slots, "tone", 0.04, "taped_papers", 0.0, 0.5))


# ----------------------------------------------------------------------------------------------------------- shapes
def wh(p):
    """A piece's width and height (its unturned box)."""
    if p.kind == "poster" or (p.kind == "card" and p.shape == "rect"):
        return p.size
    if p.kind == "polaroid":
        return p.size, p.size * POLAROID[1] / POLAROID[0]
    if p.kind == "card":                                         # a heart
        return p.size, p.size * float(np.ptp(HR.heart_outline(64)[:, 1]))
    return p.size, p.size


def crescent(d, n=48):
    """The moon's outline (closed, counter-clockwise; x, z about the disc's centre) and its two arcs (outer, inner) from
    horn to horn: a disc of diameter d less a disc 0.88 as big, set toward the upper right."""
    R, r = 0.5 * d, 0.44 * d
    c = np.array([0.24 * d, 0.12 * d])
    L = float(np.linalg.norm(c))
    x = (R * R - r * r + L * L) / (2.0 * L)                      # where the circles cross, along c from the origin
    hh = math.sqrt(max(R * R - x * x, 0.0))
    u = c / L
    v = np.array([-u[1], u[0]])
    h1, h2 = x * u + hh * v, x * u - hh * v                      # the horns
    a1, a2 = math.atan2(h1[1], h1[0]), math.atan2(h2[1], h2[0])
    if a2 < a1:
        a2 += 2.0 * math.pi                                      # the outer arc the long way round, away from c
    t = np.linspace(a1, a2, n)
    outer = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    b1, b2 = math.atan2(*(h1 - c)[::-1]), math.atan2(*(h2 - c)[::-1])
    if b2 < b1:
        b2 += 2.0 * math.pi
    s = np.linspace(b1, b2, n)
    inner = c + np.stack([r * np.cos(s), r * np.sin(s)], 1)     # the inner arc the long way round inside the disc
    return np.vstack([outer, inner[::-1][1:-1]]), outer, inner


def outline(p):
    """The piece's outline in its own frame (closed, counter-clockwise, about its middle)."""
    w, h = wh(p)
    if p.kind == "moon":
        return crescent(p.size)[0]
    if p.kind == "star":
        return BU.star_outline(60, rounding=16.0) * p.size
    if p.kind == "card" and p.shape == "heart":
        return HR.heart_outline(64) * p.size
    return S.rrect(w, h, min(0.0012, 0.25 * min(w, h)), 2)


def inside(P, poly):
    """Which points P (n, 2) lie inside the closed polygon `poly` (crossing number)."""
    x, z = P[:, 0:1], P[:, 1:2]
    a, b = poly, np.roll(poly, -1, axis=0)
    cross = ((a[None, :, 1] > z) != (b[None, :, 1] > z))
    xs = a[None, :, 0] + (z - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / np.where(
        np.abs(b[None, :, 1] - a[None, :, 1]) < 1e-15, 1e-15, b[None, :, 1] - a[None, :, 1])
    return (np.sum(cross & (x < xs), axis=1) % 2) == 1


def corner_point(p):
    """The point of the outline that curls: the corner of `corner`, or the outline's point furthest that way."""
    w, h = wh(p)
    sx = -1.0 if p.corner[1] == "L" else 1.0
    sz = 1.0 if p.corner[0] == "T" else -1.0
    poly = outline(p)
    return poly[int(np.argmax(poly @ np.array([sx * h, sz * w])))]


def lift(p, X, Z):
    """How far the sheet stands off its layer at (X, Z) in the piece's frame (m, + toward the room): a faint bow over a
    big sheet and the curling corner (a quadratic rise over the corner's last 45 %)."""
    w, h = wh(p)
    c = corner_point(p)
    R = 0.45 * min(w, h) if p.kind in ("poster", "card", "polaroid") else 0.42 * p.size
    d = np.hypot(X - c[0], Z - c[1])
    out = p.curl * np.clip(1.0 - d / R, 0.0, 1.0) ** 2
    if p.kind == "poster":
        out = out + 0.0008 * np.clip(1.0 - (2 * X / w) ** 2, 0, 1) * np.clip(1.0 - (2 * Z / h) ** 2, 0, 1)
    return out


def layer_y(p):
    return -(LAYER + p.i * STACK)


def to_wall(p, X, Z):
    """Points of the piece's frame (X, Z) on the wall (x, z) of the prop."""
    a = math.radians(p.rot)
    return p.at[0] + X * math.cos(a) - Z * math.sin(a), p.at[1] + X * math.sin(a) + Z * math.cos(a)


def from_wall(p, x, z):
    a = math.radians(p.rot)
    dx, dz = x - p.at[0], z - p.at[1]
    return dx * math.cos(a) + dz * math.sin(a), -dx * math.sin(a) + dz * math.cos(a)


def sheet_rows(p):
    """The piece's sheet as a grid of rows in its own frame (x, z): (rows (ns, m, 2), uv (ns, m, 2) or None), or for a
    radial sheet (the star, the heart) None and the outline."""
    w, h = wh(p)
    if p.kind == "moon":
        _, outer, inner = crescent(p.size)
        n = len(outer)
        t = np.linspace(0.02, 0.98, n)
        k = np.linspace(0.0, 1.0, n)
        o = np.stack([np.interp(t, k, outer[:, j]) for j in (0, 1)], 1)
        i = np.stack([np.interp(t, k, inner[:, j]) for j in (0, 1)], 1)
        s = np.linspace(0.0, 1.0, 7)
        return o[None] + (i - o)[None] * s[:, None, None], None
    if p.kind in ("star",) or (p.kind == "card" and p.shape == "heart"):
        return None, outline(p)                                  # filled from its middle: radial_sheet
    nx = 24 if p.kind == "poster" else 12
    nz = max(4, int(round(nx * h / w)))
    xs, zs = np.linspace(-0.5 * w, 0.5 * w, nx + 1), np.linspace(-0.5 * h, 0.5 * h, nz + 1)
    X, Z = np.meshgrid(xs, zs)                                   # rows along x, up z
    uv = np.stack([(X + 0.5 * w) / w, (Z + 0.5 * h) / h], -1)
    return np.stack([X, Z], -1), uv


def piece_mesh(p):
    """The piece's sheet in the prop's frame (material paper; uv on a printed piece)."""
    rows, uv = sheet_rows(p)
    if rows is None:
        m = PC.radial_sheet(outline(p), rings=7, mat=MATS.index("paper"))
        X, Z = m.V[:, 0], m.V[:, 2]
    else:
        X, Z = rows[..., 0].ravel(), rows[..., 1].ravel()
        m = PC.sheet(np.stack([rows[..., 0], np.zeros(rows.shape[:2]), rows[..., 1]], -1), mat=MATS.index("paper"),
                     uv=uv)
    x, z = to_wall(p, X, Z)
    V = np.stack([x, layer_y(p) - lift(p, X, Z), z], 1)
    return S.Mesh(V, m.Q, m.T, m.Qm, m.Tm, UV=m.UV)


def ground_fn(pieces):
    """y of the nearest paper toward the room over each wall point (x, z) (vectorised), the wall's 0 where there is
    none."""
    polys = [outline(p) for p in pieces]

    def ground(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        out = np.zeros_like(x)
        for p, poly in zip(pieces, polys):
            X, Z = from_wall(p, x, z)
            hit = inside(np.stack([X, Z], 1), poly)
            y = layer_y(p) - lift(p, X, Z)
            out = np.where(hit, np.minimum(out, y), out)
        return out
    return ground


def tape_specs(p, rng):
    """[(centre (x, z) in the piece's frame, angle in its frame, length)] of the piece's strips."""
    if p.tape == 0:
        return []
    w, h = wh(p)
    if p.kind == "poster" and p.tape == 2:
        out = []
        for sx in (-1.0, 1.0):
            a = 90.0 - sx * rng.uniform(38.0, 55.0)
            out.append(((sx * (0.5 * w - 0.008), 0.5 * h - 0.008), a, rng.uniform(0.058, 0.07)))
        return out
    poly = outline(p)
    top = poly[int(np.argmax(poly[:, 1]))]
    if p.kind in ("moon", "star") or (p.kind == "card" and p.shape == "heart"):
        tops = [top]
    elif p.tape == 2:
        tops = [np.array([-0.3 * w, 0.5 * h]), np.array([0.3 * w, 0.5 * h])]
    else:
        tops = [np.array([rng.uniform(-0.12, 0.12) * w, 0.5 * h])]
    if p.kind != "poster" and p.tape == 2 and len(tops) == 1:
        tops = [top + np.array([-0.012, 0.0]), top + np.array([0.012, 0.0])]
    out = []
    for t in tops:
        along = rng.random() < 0.3 and p.kind in ("card", "polaroid", "poster")
        a = rng.uniform(-14.0, 14.0) if along else 90.0 + rng.uniform(-24.0, 24.0)
        c = t + (np.array([0.0, -0.002]) if along else np.array([0.0, 0.004]))
        out.append(((float(c[0]), float(c[1])), a, rng.uniform(0.045, 0.06)))
    return out


def tapes_mesh(pieces, seed):
    """Every strip of tape over the pieces (material tape), in the prop's frame."""
    rng = np.random.default_rng(int(seed) + 11)
    ground = ground_fn(pieces)
    strips = []
    for p in pieces:
        for (cx, cz), ang, length in tape_specs(p, rng):
            x, z = to_wall(p, cx, cz)
            strips.append(PC.tape_strip((float(x), float(z)), p.rot + ang, length, ground, rng, mat=MATS.index("tape")))
    return S.merge(strips)


# ------------------------------------------------------------------------------------------------------------ prints
def to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


class Canvas:
    """A picture in sRGB (row 0 its top) over a sheet w x h metres at `ppm` pixels a metre, its coordinates x right
    and z up from the sheet's middle; shapes are laid on with antialiased coverage from their signed distance."""

    def __init__(self, w, h, ppm, ground):
        self.w, self.h, self.ppm = w, h, ppm
        nx, nz = int(round(w * ppm)), int(round(h * ppm))
        self.X = ((np.arange(nx) + 0.5) / ppm - 0.5 * w)[None, :].repeat(nz, 0)
        self.Z = (0.5 * h - (np.arange(nz) + 0.5) / ppm)[:, None].repeat(nx, 1)
        self.img = np.empty((nz, nx, 3))
        self.img[:] = ground

    def cover(self, d):
        return np.clip(0.5 - d * self.ppm, 0.0, 1.0)

    def lay(self, rgb, d):
        a = self.cover(d)
        m = a > 0.0
        a = a[m][:, None]
        self.img[m] = self.img[m] * (1.0 - a) + np.asarray(rgb)[None, :] * a

    def disc(self, c, r):
        return np.hypot(self.X - c[0], self.Z - c[1]) - r

    def box(self, c, hw, hh):
        dx, dz = np.abs(self.X - c[0]) - hw, np.abs(self.Z - c[1]) - hh
        return np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dx, dz), 0.0)

    def poly(self, P):
        """The signed distance to a closed polygon (negative inside), over its box only (elsewhere 1)."""
        lo, hi = P.min(0) - 3.0 / self.ppm, P.max(0) + 3.0 / self.ppm
        m = (self.X >= lo[0]) & (self.X <= hi[0]) & (self.Z >= lo[1]) & (self.Z <= hi[1])
        x, z = self.X[m], self.Z[m]
        d2 = np.full(x.shape, np.inf)
        wn = np.zeros(x.shape, bool)
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            e = b - a
            t = np.clip(((x - a[0]) * e[0] + (z - a[1]) * e[1]) / max(e @ e, 1e-18), 0.0, 1.0)
            d2 = np.minimum(d2, (x - a[0] - t * e[0]) ** 2 + (z - a[1] - t * e[1]) ** 2)
            c = (a[1] > z) != (b[1] > z)
            xs = a[0] + (z - a[1]) * e[0] / (e[1] if abs(e[1]) > 1e-18 else 1e-18)
            wn ^= c & (x < xs)
        out = np.ones(self.X.shape)
        out[m] = np.where(wn, -1.0, 1.0) * np.sqrt(d2)
        return out


def star_poly(c, size, rot=0.0):
    a = math.radians(rot)
    P = BU.star_outline(40, rounding=16.0) * size
    return c + P @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])


def heart_poly(c, size):
    return c + HR.heart_outline(64) * size


def poster_print(p, cols, ink, seed):
    """The poster's picture: (h, w, 3) sRGB. `cols` the design's roles (sRGB, DESIGNS' order), `ink` the frame line."""
    w, h = p.size
    C = Canvas(w, h, PPM["poster"], cols["paper"])
    k = cols["design"]
    if p.design == "sun":
        sun, a, b, c = k
        cz, r = 0.10 * h, 0.31 * w
        C.lay(sun, C.disc((0.0, cz), r))
        for f, t in ((0.10, 0.018), (0.30, 0.026), (0.50, 0.034), (0.70, 0.042)):     # slits across its lower half
            C.lay(cols["paper"], C.box((0.0, cz - f * r), 0.5 * w, 0.5 * t * r * 2.2))
        base = -0.5 * h + 0.03
        for j, col in enumerate((c, b, a)):                                           # sunset stripes under it
            C.lay(col, C.box((0.0, base + 0.018 + j * 0.054), 0.5 * w - 0.03, 0.018))
    elif p.design == "heart":
        ground, heart, rim, dots = k
        C.lay(ground, C.box((0.0, 0.0), 0.5 * w, 0.5 * h))
        C.lay(rim, C.poly(heart_poly((0.0, 0.02 * h), 0.72 * w)))
        C.lay(heart, C.poly(heart_poly((0.0, 0.022 * h), 0.60 * w)))
        for x, z in ((-0.36, 0.40), (0.36, 0.40), (-0.36, -0.40), (0.36, -0.40)):
            C.lay(dots, C.poly(star_poly((x * w, z * h), 0.07 * w, 8.0)))
    elif p.design == "stars":
        ground, big, small, accent = k
        C.lay(ground, C.box((0.0, 0.0), 0.5 * w, 0.5 * h))
        C.lay(big, C.poly(star_poly((-0.08 * w, 0.12 * h), 0.52 * w, -8.0)))
        rng = np.random.default_rng(int(seed) + 3 + p.i)
        placed = [(-0.08 * w, 0.12 * h, 0.30 * w)]
        for j in range(60):
            x, z = rng.uniform(-0.42, 0.42) * w, rng.uniform(-0.44, 0.44) * h
            s = rng.uniform(0.06, 0.14) * w
            if all(math.hypot(x - a, z - b) > 0.5 * s + q for a, b, q in placed) and len(placed) < 14:
                placed.append((x, z, 0.5 * s))
                C.lay((small, accent, big)[j % 3], C.poly(star_poly((x, z), s, rng.uniform(-30, 30))))
    else:                                                                             # rainbow
        sky, r1, r2, r3, r4, cloud = k
        C.lay(sky, C.box((0.0, 0.0), 0.5 * w, 0.5 * h))
        cz = -0.16 * h
        band = 0.075 * w
        for j, col in enumerate((r1, r2, r3, r4)):
            ro = 0.44 * w - j * band
            d = np.maximum(C.disc((0.0, cz), ro), -C.disc((0.0, cz), ro - band))
            C.lay(col, np.maximum(d, cz - C.Z))
        for sx in (-1.0, 1.0):
            x0 = sx * (0.44 * w - 2.0 * band)
            d = np.minimum.reduce([C.disc((x0 - 0.07 * w, cz), 0.075 * w), C.disc((x0 + 0.06 * w, cz + 0.005), 0.08 * w),
                                   C.disc((x0, cz + 0.05 * w), 0.09 * w)])
            C.lay(cloud, np.maximum(d, (cz - 0.035 * w) - C.Z))
    C.lay(ink, np.abs(C.box((0.0, 0.0), 0.5 * w - 0.012, 0.5 * h - 0.012)) - 0.0009)  # the ink frame
    return C.img


def polaroid_print(p, frame, field, ink):
    """The instant photo's picture: (h, w, 3) sRGB: the white frame, the picture's field (lighter at its top) and its
    shape."""
    w, h = wh(p)
    k = w / POLAROID[0]
    side, top = POLAROID[2] * k, POLAROID[3] * k
    C = Canvas(w, h, PPM["polaroid"], frame)
    cz = 0.5 * h - top - 0.5 * side
    win = C.box((0.0, cz), 0.5 * side, 0.5 * side)
    g = np.clip((C.Z - (cz - 0.5 * side)) / side, 0.0, 1.0)[..., None]
    sky = np.asarray(field) * (0.86 + 0.20 * g)
    a = C.cover(win)[..., None]
    C.img = C.img * (1.0 - a) + np.clip(sky, 0.0, 1.0) * a
    c = (0.0, cz + 0.02 * side)
    if p.photo == "heart":
        d = C.poly(heart_poly(c, 0.50 * side))
    elif p.photo == "sun":
        d = C.disc(c, 0.24 * side)
    elif p.photo == "moon":
        d = C.poly(c + crescent(0.52 * side)[0])
    else:
        d = C.poly(star_poly(c, 0.56 * side))
    C.lay(ink, np.maximum(d, win))
    return C.img


def pictures(pieces, slots, roles, seed):
    """{piece index: (h, w, 3) sRGB} of the printed pieces; `roles` the prop's resolved colour roles."""
    def srgb(spec, what):
        return to_srgb(SK.rgb(spec, slots, f"taped_papers: {what}"))
    out = {}
    for p in pieces:
        if p.kind == "poster":
            cols = {"paper": srgb(p.color or roles["paper"], "color"), "design": [srgb(c, "colors") for c in p.colors]}
            out[p.i] = poster_print(p, cols, srgb(roles["line"], "line"), seed)
        elif p.kind == "polaroid":
            out[p.i] = polaroid_print(p, srgb(roles["paper"], "paper"), srgb(p.color, "color"), srgb(p.ink, "ink"))
    return out


def photo_window(p):
    """(u0, v0, u1, v1) of an instant photo's picture in its UV square."""
    w, h = wh(p)
    k = w / POLAROID[0]
    side, top = POLAROID[2] * k, POLAROID[3] * k
    u0 = 0.5 - 0.5 * side / w
    v1 = 1.0 - top / h
    return u0, v1 - side / h, 1.0 - u0, v1


# --------------------------------------------------------------------------------------------------------------- card
def gap(pieces, cell=0.01):
    """(centre (x, z), radius) of the biggest empty circle among the pieces, inside their box: the free cell of a
    `cell` grid furthest from every outline (resampled every 5 mm) and from the box's edge."""
    polys = [outline(p) for p in pieces]
    rims = []
    for p, P in zip(pieces, polys):
        n = max(8, int(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum() / 0.005))
        Q = S.resample_loop(P, n)
        rims.append(np.stack(to_wall(p, Q[:, 0], Q[:, 1]), 1))
    occ = np.concatenate(rims)
    lo, hi = occ.min(0), occ.max(0)
    X, Z = np.meshgrid(np.arange(lo[0], hi[0], cell), np.arange(lo[1], hi[1], cell))
    G = np.stack([X.ravel(), Z.ravel()], 1)
    full = np.zeros(len(G), bool)
    for p, P in zip(pieces, polys):
        full |= inside(np.stack(from_wall(p, G[:, 0], G[:, 1]), 1), P)
    free = G[~full]
    if not len(free):
        return (float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))), 0.0
    d = np.full(len(free), np.inf)
    for chunk in np.array_split(occ, max(1, len(occ) // 256)):
        d = np.minimum(d, np.sqrt(((free[:, None, :] - chunk[None, :, :]) ** 2).sum(-1)).min(1))
    d = np.minimum(d, np.minimum(free - lo, hi - free).min(1))
    j = int(np.argmax(d))
    return (float(free[j, 0]), float(free[j, 1])), float(d[j])


def card(name, pieces, roles=None):
    """The prop card (docs/design.md: Prop card)."""
    (gx, gz), gr = gap(pieces)
    polys = [np.stack(to_wall(p, *outline(p).T), 1) for p in pieces]
    allp = np.concatenate(polys)
    mid = 0.5 * (allp.min(0) + allp.max(0))
    r = lambda v: round(float(v), 5)                                                            # noqa: E731
    surf = []
    for p in pieces:
        if p.kind == "poster":
            a = math.radians(p.rot)
            surf.append({"name": f"poster{p.i}", "center": [r(p.at[0]), r(layer_y(p)), r(p.at[1])], "normal": [0.0, -1.0, 0.0],
                         "up": [r(-math.sin(a)), 0.0, r(math.cos(a))], "size": [r(p.size[0]), r(p.size[1])]})
    use = {"look": [{"name": "gap", "point": [r(gx), 0.0, r(gz)], "radius": r(gr)},
                    {"name": "centre", "point": [r(mid[0]), 0.0, r(mid[1])]}]}
    if surf:
        use["surface"] = surf
    return {"origin": "wall_center", "front": "-Y", "blocks": False, "slots": dict(roles or {}), "use": use,
            "colliders": [], "pieces": [{"kind": p.kind, "at": [r(c) for c in p.at], "size": wh(p), "rot": p.rot}
                                        for p in pieces],
            "objects": [f"{name}_{p.kind}{p.i}" for p in pieces] + [f"{name}_tape"]}
