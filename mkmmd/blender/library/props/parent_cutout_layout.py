"""Layout of the picture-book parent cut-outs (`library:parent_cutout`; pure Python + numpy, no bpy: importable from tests):
the paper pieces' meshes, their white die-cut borders and soft drop shadows, the brads, the colour roles, the wag maths and
the card. The drawings (control points) are parent_cutout_parts.

Frame (metres): the card stands on its foot line, the bottom of its white border at z = 0 under the figure's centre line
(x = 0, +x the figure's left), and faces -Y; y = 0 is the front of the figure's paper at its feet. A giant flat adult
silhouette (`figure` "father", 3.28 m to the top of the border, or "mother", 3.16 m) in construction paper, built as a
stack of plates (`electric_guitar_geo.slab`: a rolled rim of offset rings, the faces a Delaunay fill):

  * the white card (`rim`, 16 mm, rolled 5 mm both sides): the figure's outline closed over gaps narrower than 2 CLOSE (a
    die-cutter bridges them) and grown by RIM, smoothed: a die-cut border;
  * the figure's paper (`paper`, 5 mm proud of the card, its edge rolled 2.2 mm): the outline cut with scissors
    (`scissor`: straight snips that turn at most a few degrees, their joints a hair off the line);
  * its details (hair, collar, shirt, tie, lapels, dress, hem band, buttons, shoes): 3 mm paper layers, `level` 1 .. 4
    stacked 2.5 mm apart, each scissor-cut too;
  * the whole card bows forward by BOW at mid-height (paper is never flat; the flat pieces on brads do not bow);
  * a soft drop shadow: the border's outline offset by SHADOW in the card's plane on a sheet 6 mm behind the card (a
    blurred alpha image: a graphic layer).

Pieces on paper-fastener brads lie flat in front of the body: the father's left forearm with the scolding hand (`hand`,
turning in the card's plane about the brad at his elbow), the mother's arms (`arm_L`, `arm_R`, static, on brads at her
shoulders). Each carries its own border and casts its own shadow (half the body's offset: it is lifted less).

The wag (the father): the hand turns counter-clockwise seen from the front (about -Y) by
    angle = wag + wagging * sin(2 pi wag_hz t)        degrees, t the clip second
(`wag` keyed, `wagging` the amplitude of an automatic tick-tock, `wag_hz` its rate); EXPR is the driver on the hand's
rotation_euler[1] (= -radians(angle)) and the hand's shadow's."""
import functools
import math
import zlib
from collections import namedtuple

import numpy as np

from ....core import palette as PAL
from ....core import shell as S
from . import convertible_palette as CP
from . import electric_guitar_geo as GEO
from .parent_cutout_parts import FIGURES

NAMES = tuple(FIGURES)                  # the `figure` option
RIM = 0.035                             # the white border round every piece
CLOSE = 0.055                           # gaps narrower than twice this are bridged by the border
BOW = 0.015                             # the card bows forward by this at mid-height
STEP = 0.016                            # ring spacing of the big outlines (m); details and small pieces: STEP_FINE
STEP_FINE = 0.008
CARD_Y = (0.005, 0.021)                 # the white card: front, back (y relative to the bowed surface)
PAPER_Y = (0.0, 0.006)                  # the figure's paper (its back 1 mm inside the card)
DETAIL_T, DETAIL_DY = 0.003, 0.0025     # a detail layer's thickness and the step from one level to the next
LEVELS = 4
R_CARD, R_PAPER, R_DETAIL = 0.005, 0.0022, 0.0012       # the rolled edges
TOP_Y = 0.0005 - (LEVELS - 1) * DETAIL_DY - DETAIL_T   # the front of the highest detail level (-0.010)
GAP = 0.006                             # a piece on a brad lies this far in front of what it covers
PIECE_CARD = 0.010                      # its white card's thickness
SHADOW = (0.09, -0.07)                  # the drop shadow's offset in the card's plane (x, z): light from the top left
SHADOW_BACK = 0.006                     # its sheet this far behind the card's back (a piece's: halfway to the body)
SHADOW_SOFT, SHADOW_PX = 0.018, 0.01    # its blur (sigma) and image pixel, m
SHADOW_ALPHA = 0.6                      # its darkness at the core
BRAD = (0.034, 0.012, 0.0055)           # a brad's head radius, dome height, shank radius
PRONG = (0.085, 0.016, 0.0018)          # its two legs, flattened on the back: length, width, thickness

ROLES = {
    "paper": {"father": "pine", "mother": "rose"},  # the figure's construction paper
    "shade": "paper:3,black:1",         # the paper a quarter darker (the father's lapels)
    "hair": "ink",                      # hair and shoes
    "detail": "love",                   # his tie, her dress
    "trim": "chalk",                    # white paper: shirt, collars, cuffs, pocket square, buttons
    "accent": "gold",                   # her bow and hem band
    "rim": "chalk",                     # the white card of the die-cut borders
    "shadow": "ink",                    # the drop shadows
    "brad": "gold",                     # the brass paper fasteners
}
MATS = ("paper", "rim", "hair", "detail", "trim", "accent", "shade", "brad")    # material index -> role
PARAMS = {"wag": (0.0, (-120.0, 120.0), "degrees the hand is turned about the elbow's brad, counter-clockwise seen "
                                        "from the front (key it)"),
          "wagging": (0.0, (0.0, 60.0), "degrees of an automatic tick-tock of the hand about `wag`"),
          "wag_hz": (2.5, (0.1, 12.0), "the tick-tock's rate, wags (there and back) a second")}
EXPR = "-(wag + wagging * sin(6.283185307 * wag_hz * (frame - f0) / fps)) * 0.017453293"


# ===================================================================================================================
# colour roles
# ===================================================================================================================
class Pal(CP.Pal):
    """The cut-out's colour roles over the palette: each a slot, a hex, a role or a 'name:weight,...' blend. Three slots
    of its own: `ink` (the darker of the palette's base and text), `chalk` (the lighter: white card in a light palette as
    in a dark one) and `black` (to darken a blend)."""

    def __init__(self, slots=None, figure="father"):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d[figure] if isinstance(d, dict) else d)) for r, d in ROLES.items()}

    def _ends(self):
        a, b = (PAL.linear(self.palette[k]) for k in ("base", "text"))
        lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]      # noqa: E731
        return (a, b) if lum(a) < lum(b) else (b, a)

    def slot_lin(self, name):
        if name in ("ink", "chalk"):
            return self._ends()[name == "chalk"]
        if name == "black":
            return (0.0, 0.0, 0.0)
        return super().slot_lin(name)

    def lin(self, spec, depth=0):
        """Linear RGB of a role, a slot (`ink`, `chalk` too), a hex or a 'name:weight,...' blend of them."""
        if depth > 8:
            raise ValueError(f"parent_cutout: the colour roles refer to each other in a loop ({spec!r})")
        if spec in self.specs:
            return self.lin(self.specs[spec], depth + 1)
        out, tot = [0.0, 0.0, 0.0], 0.0
        for part in spec.split(","):
            nm, w = CP.parse(part)
            c = self.lin(nm, depth + 1) if nm in self.specs else self.slot_lin(nm)
            out = [o + w * x for o, x in zip(out, c)]
            tot += w
        return tuple(o / tot for o in out)

    def check(self):
        """Every role resolves to a colour, or a ValueError that names it."""
        for role, spec in self.specs.items():
            try:
                self.lin(role)
            except (KeyError, ValueError) as e:
                raise ValueError(f"parent_cutout: colour role {role} = {spec!r}: give a palette slot, a hex or a blend "
                                 f"'slot:weight,...' ({e})") from None


def options(slots):
    """(figure, the starting values of the live root properties) from the prop's slots, checked."""
    slots = dict(slots or {})
    figure = slots.get("figure", "father")
    if figure not in NAMES:
        raise ValueError(f"parent_cutout: figure is {' or '.join(repr(n) for n in NAMES)}, got {figure!r}")
    start = {}
    for k, (d, (lo, hi), _doc) in PARAMS.items():
        v = slots.get(k, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"parent_cutout: {k} must be a number, got {v!r}")
        if not lo <= v <= hi:
            raise ValueError(f"parent_cutout: {k} is {lo:g} .. {hi:g}, got {v:g}")
        start[k] = float(v)
    return figure, start


# ===================================================================================================================
# outlines (closed polygons (n, 2) of (x, z))
# ===================================================================================================================
def normals(P):
    """Outward unit normals at the vertices of the counter-clockwise closed polygon P."""
    a = np.roll(P, -1, 0) - P
    b = P - np.roll(P, 1, 0)
    t = a / np.maximum(np.linalg.norm(a, axis=1), 1e-12)[:, None] + b / np.maximum(np.linalg.norm(b, axis=1), 1e-12)[:, None]
    t = t / np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
    return np.stack([t[:, 1], -t[:, 0]], 1)


def crossings(P):
    """(i, j, t) for every pair of edges i < j of the closed polygon P that are not neighbours and cross (edge k runs from
    P[k] to P[k + 1]; t is where along edge i): an outline that crosses itself has some."""
    A = np.asarray(P, float)
    B = np.roll(A, -1, 0)
    n = len(A)
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    out = []
    for k in range(0, n, 512):
        sl = slice(k, k + 512)
        box = ((lo[sl, None, 0] <= hi[None, :, 0]) & (lo[None, :, 0] <= hi[sl, None, 0]) &
               (lo[sl, None, 1] <= hi[None, :, 1]) & (lo[None, :, 1] <= hi[sl, None, 1]))
        I, J = np.nonzero(box)
        I = I + k
        keep = (J > I + 1) & ~((I == 0) & (J == n - 1))
        I, J = I[keep], J[keep]
        if not len(I):
            continue
        r, s, q = B[I] - A[I], B[J] - A[J], A[J] - A[I]
        den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        ok = np.abs(den) > 1e-18
        den = np.where(ok, den, 1.0)
        t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / den
        u = (q[:, 0] * r[:, 1] - q[:, 1] * r[:, 0]) / den
        hit = ok & (t > 0.0) & (t < 1.0) & (u > 0.0) & (u < 1.0)
        out += [(int(i), int(j), float(tt)) for i, j, tt in zip(I[hit], J[hit], t[hit])]
    return out


def untangle(P):
    """The outer loop of a closed polygon that crosses itself: an offset outline's swallowtails at concave turns and the
    loops of gaps it closes are cut away (the outline is started on its outer loop, at an extreme point)."""
    P = np.asarray(P, float)
    for _ in range(64):
        P = np.roll(P, -int(np.argmax(P[:, 0] + 0.37 * P[:, 1])), 0)
        cr = sorted(crossings(P), key=lambda c: (c[0], -c[1]))
        if not cr:
            return P
        keep, last = [], -1
        for i, j, t in cr:                                   # the outermost loops that do not overlap, all at once
            if i > last:
                keep.append((i, j, t))
                last = j
        parts, start = [], 0
        for i, j, t in keep:
            parts += [P[start:i + 1], (P[i] + t * (P[(i + 1) % len(P)] - P[i]))[None]]
            start = j + 1
        P = np.concatenate(parts + [P[start:]])
    raise ValueError("parent_cutout: an outline does not come apart into one loop")


def resample(P, step):
    """Points every `step` metres along the closed polygon P (from its first point)."""
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    n = max(int(round(s[-1] / step)), 8)
    q = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.stack([np.interp(q, s, Q[:, 0]), np.interp(q, s, Q[:, 1])], 1)


def subdivide(P, step):
    """The closed polygon P with every edge cut into pieces no longer than `step` (its vertices kept)."""
    P = np.asarray(P, float)
    E = np.roll(P, -1, 0) - P
    k = np.maximum(np.ceil(np.linalg.norm(E, axis=1) / step).astype(int), 1)
    return np.concatenate([P[i] + E[i] * (np.arange(k[i])[:, None] / k[i]) for i in range(len(P))])


def smooth(P, passes=6, k=0.5):
    """A closed polygon relaxed toward its neighbours' midpoints: corners round off."""
    P = np.asarray(P, float).copy()
    for _ in range(passes):
        P = (1.0 - k) * P + k * 0.5 * (np.roll(P, 1, 0) + np.roll(P, -1, 0))
    return P


def offset(P, d, join=12.0):
    """The closed polygon P grown by d (negative: shrunk) along its vertex normals, a round join (an arc every `join`
    degrees) where it turns sharply on the side it grows to, its loops cut away. Counter-clockwise."""
    P = GEO.ccw(P)
    a, b = P - np.roll(P, 1, 0), np.roll(P, -1, 0) - P                  # the edges into and out of each vertex
    phi0, phi1 = np.arctan2(-a[:, 0], a[:, 1]), np.arctan2(-b[:, 0], b[:, 1])     # their outward normals' angles
    turn = (phi1 - phi0 + math.pi) % (2.0 * math.pi) - math.pi                  # > 0 where the outline bends left
    avg = P + d * normals(P)
    out = []
    for i in range(len(P)):
        if math.copysign(1.0, d) * turn[i] > math.radians(join):
            t = phi0[i] + turn[i] * np.linspace(0.0, 1.0, int(math.ceil(abs(turn[i]) / math.radians(join))) + 1)
            out.append(P[i] + d * np.stack([np.cos(t), np.sin(t)], 1))
        else:
            out.append(avg[i][None])
    return GEO.ccw(untangle(np.concatenate(out)))


def border(P):
    """The die-cut border round the outline P: closed over gaps narrower than 2 CLOSE, grown by RIM, smoothed."""
    dense = resample(P, 0.01)
    return resample(smooth(offset(offset(dense, CLOSE), RIM - CLOSE), passes=8), STEP)


def scissor(P, seed, step, snip=0.08, turn=7.0, wobble=0.0016, least=0.004, sag=0.00025):
    """The smooth closed outline P as scissors cut it: straight snips at most `snip` long (varying) that turn at most `turn`
    degrees of the curve (none shorter than `least`), their joints up to `wobble` off the line (less on short snips), cut
    into pieces no longer than `step`; the points inside a snip sag in by `sag` (no three ring points in a line: the face's
    Delaunay fill keeps every edge of a slightly concave run)."""
    P = GEO.ccw(resample(P, 0.003))
    e = np.roll(P, -1, 0) - P
    ang = np.arctan2(e[:, 1], e[:, 0])
    bend = np.degrees(np.abs((np.diff(np.concatenate([ang, ang[:1]])) + math.pi) % (2.0 * math.pi) - math.pi))
    seg = np.linalg.norm(e, axis=1)
    rng = np.random.default_rng(seed)
    keep, s, b = [0], 0.0, 0.0
    lim = snip * rng.uniform(0.6, 1.3)
    for i in range(1, len(P)):
        s += seg[i - 1]
        b += bend[i - 1]
        if s >= least and (s >= lim or b >= turn):
            keep.append(i)
            s, b = 0.0, 0.0
            lim = snip * rng.uniform(0.6, 1.3)
    if len(keep) > 3 and np.linalg.norm(P[keep[-1]] - P[0]) < least:
        keep.pop()
    Q = P[keep]
    ln = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
    short = np.minimum(ln, np.roll(ln, 1))
    Q = Q + normals(Q) * (rng.uniform(-wobble, wobble, len(Q)) * np.clip(short / 0.02, 0.0, 1.0))[:, None]
    k = np.maximum(np.ceil(np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1) / step).astype(int), 1)
    out = subdivide(Q, step)
    inner = np.ones(len(out), bool)
    inner[np.concatenate([[0], np.cumsum(k)[:-1]])] = False             # the snips' joints stay where they are
    out[inner] -= sag * normals(out)[inner]
    return out


def outline(points, corners=(), spacing=0.005):
    """The closed spline through the control points (sharp at `corners`), counter-clockwise."""
    return GEO.ccw(GEO.spline(np.asarray(points, float), corners, spacing))


def stroke(points, widths, spacing=0.004):
    """A limb drawn along a centre line: the smooth curve through `points` (pchip on the chord length), its half width
    interpolated from `widths` (one per point), round ends; the inside of a bend cut clean and eased. Counter-clockwise."""
    P = np.asarray(points, float)
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = np.linspace(0.0, u[-1], max(int(u[-1] / spacing), 8) + 1)
    C = S.pchip(u, P, q)
    w = S.pchip(u, np.asarray(widths, float)[:, None], q)[:, 0]
    T = np.gradient(C, axis=0)
    T = T / np.maximum(np.linalg.norm(T, axis=1), 1e-12)[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 15)[1:-1]
    end = C[-1] + w[-1] * (np.cos(a)[:, None] * T[-1] - np.sin(a)[:, None] * N[-1])       # +N round the front to -N
    start = C[0] + w[0] * (-np.cos(a)[:, None] * T[0] + np.sin(a)[:, None] * N[0])         # -N round the back to +N
    ring = np.concatenate([C + w[:, None] * N, end, (C - w[:, None] * N)[::-1], start])
    return GEO.ccw(smooth(resample(untangle(GEO.ccw(ring)), spacing), passes=6))


# ===================================================================================================================
# raster (the shadows)
# ===================================================================================================================
def raster(P, x0, z0, px, w, h, ss=4):
    """Coverage 0..1 (h, w) of the closed polygon P on a grid of `px` pixels from (x0, z0), row 0 the TOP (z0 + h px):
    even-odd scanlines, ss x ss samples a pixel."""
    P = np.asarray(P, float)
    A, B = P, np.roll(P, -1, 0)
    W, H = w * ss, h * ss
    zq = (z0 + (np.arange(H)[::-1] + 0.5) * px / ss)[:, None]
    xs = x0 + (np.arange(W) + 0.5) * px / ss
    za, zb = A[None, :, 1], B[None, :, 1]
    cross = (za > zq) != (zb > zq)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = A[None, :, 0] + (zq - za) * (B[:, 0] - A[:, 0])[None] / (zb - za)
    xi = np.sort(np.where(cross, xi, np.inf), axis=1)
    n = cross.sum(1)
    cov = np.zeros((H, W), np.float32)
    for r in np.nonzero(n >= 2)[0]:
        cov[r] = np.searchsorted(xi[r, :n[r]], xs) % 2 == 1
    return cov.reshape(h, ss, w, ss).mean((1, 3))


def blur(a, sigma):
    """A Gaussian blur of `sigma` pixels (separable, zero outside)."""
    r = max(int(math.ceil(3.0 * sigma)), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a).astype(np.float32)


Shadow = namedtuple("Shadow", "alpha x0 z0 x1 z1")


def shadow(P, dx, dz):
    """The soft shadow of the outline P offset by (dx, dz): an alpha image (row 0 the top) over the box x0..x1, z0..z1."""
    pad = 4.0 * SHADOW_SOFT
    lo, hi = np.asarray(P).min(0) + (dx, dz) - pad, np.asarray(P).max(0) + (dx, dz) + pad
    w, h = (int(math.ceil((b - a) / SHADOW_PX)) for a, b in zip(lo, hi))
    a = blur(raster(np.asarray(P) + (dx, dz), lo[0], lo[1], SHADOW_PX, w, h), SHADOW_SOFT / SHADOW_PX)
    return Shadow(a, float(lo[0]), float(lo[1]), float(lo[0] + w * SHADOW_PX), float(lo[1] + h * SHADOW_PX))


def sheet(sh, y, rows=24):
    """The sheet a shadow is drawn on: a grid of quads in the x-z plane over its box, at y(z) (a function: the card's
    bow), facing -Y, its uv the image's (v up)."""
    xs = np.array([sh.x0, sh.x1])
    zs = np.linspace(sh.z0, sh.z1, rows + 1)
    X, Z = np.meshgrid(xs, zs)
    V = np.stack([X, y(Z), Z], -1).reshape(-1, 3)
    UV = np.stack([(X - sh.x0) / (sh.x1 - sh.x0), (Z - sh.z0) / (sh.z1 - sh.z0)], -1).reshape(-1, 2)
    i = np.arange(rows)[:, None] * 2
    Q = np.concatenate([i, i + 1, i + 3, i + 2], 1)                 # counter-clockwise seen from -Y: it faces -Y
    return S.Mesh(V, Q, UV=UV)


# ===================================================================================================================
# plates
# ===================================================================================================================
def bow(z, height):
    """y of the card's bowed surface at height z: 0 at its foot and top, -BOW at mid-height."""
    u = np.clip(np.asarray(z, float) / height, 0.0, 1.0)
    return -4.0 * BOW * u * (1.0 - u)


def _plate(ring, front, back, r, spacing, role):
    """A plate cut to `ring`: its front face at y = front(x, z), its back at back(x, z), both edges rolled by r."""
    return GEO.slab(ring, front, back, r=r, n=3, wall=1, spacing=spacing, mat=MATS.index(role))


def _seed(*key):
    return zlib.crc32(":".join(str(k) for k in key).encode())


def base_outline(piece):
    """The smooth outline of a piece as drawn (figure frame)."""
    return stroke(*piece.stroke) if piece.stroke is not None else outline(piece.points, piece.corners)


Cut = namedtuple("Cut", "mesh paper rim")
WIDE = 0.15                            # a detail bigger than this (m^2, a dress) is cut and filled at the body's spacing


def detail_ring(figure, key, k, detail, shift):
    """(the scissor-cut outline of detail k of a piece in the card frame, whether it is wide)."""
    drawn = outline(detail.points, detail.corners, 0.003) + shift
    wide = S.polygon_area(drawn) > WIDE
    return scissor(drawn, _seed(figure, key, k), STEP if wide else STEP_FINE, snip=0.08 if wide else 0.05,
                   wobble=0.0016 if wide else 0.0008), wide


def cut(figure, key, piece, base, y0, card_t, shift):
    """A piece as plates, one mesh in the card frame: its white card (`card_t` thick) cut to the border of `base` (its
    outline in the card frame), the paper cut with scissors along `base`, the details on it; `y0(x, z)` is the y of the
    paper's front, `shift` (x, z) takes the drawing to the card frame. Cut(mesh, paper outline, border outline)."""
    big = S.polygon_area(base) > 0.5
    paper = scissor(base, _seed(figure, key), STEP if big else STEP_FINE)
    rim = border(base)
    at = lambda dy: (lambda x, z: y0(x, z) + dy)                        # noqa: E731
    parts = [_plate(rim, at(CARD_Y[0]), at(CARD_Y[0] + card_t), min(R_CARD, 0.4 * card_t), 0.09 if big else 0.04,
                    "rim"),
             _plate(paper, at(PAPER_Y[0]), at(PAPER_Y[1]), R_PAPER, 0.08 if big else 0.04, "paper")]
    for k, d in enumerate(piece.details):
        ring, wide = detail_ring(figure, key, k, d, shift)
        back = 0.0005 - (d.level - 1) * DETAIL_DY
        parts.append(_plate(ring, at(back - DETAIL_T), at(back), R_DETAIL, 0.08 if wide else 0.05, d.role))
    return Cut(S.merge(parts), paper, rim)


def flat_y(zs, height):
    """y of the paper's front of a flat piece (on a brad) that covers the heights `zs`: its card's back GAP in front of
    the body's highest layer there."""
    return float(bow(zs, height).min()) + TOP_Y - GAP - CARD_Y[0] - PIECE_CARD


def brad(at, y_head, y_back):
    """A paper fastener in the card frame: its domed head on the paper at `y_head` at `at` (x, z), the shank through to
    the card's back at `y_back`, the two legs spread flat on the back (up and down). Material `brad`."""
    R, H, rs = BRAD
    depth = y_back - y_head
    dome = [(R * math.cos(a), 0.0015 + (H - 0.0015) * math.sin(a)) for a in np.radians(np.linspace(0.0, 90.0, 9))]
    prof = [(0.0, -depth), (rs, -depth), (rs, -0.0005), (R - 0.003, -0.0005), (R - 0.0005, 0.0003)] + dome
    head = S.lathe(np.array(prof), seg=32, mat=MATS.index("brad")).rot(rx=90.0)       # its axis +z -> -y (to the front)
    L, w, t = PRONG
    legs = [S.rounded_panel(S.rrect(w, L, 0.4 * w, 3), t, r_edge=0.0008, mat=MATS.index("brad"))
            .rot(rx=-90.0).moved((0.0, depth - 0.0004, s * (0.5 * L + rs))) for s in (1.0, -1.0)]
    return S.merge([head] + legs).moved((at[0], y_head, at[1]))


Layout = namedtuple("Layout", "figure height lift body sheets hand hand_sheet hand_at brads pivot tip head size cuts")


def _sheet(rim, d, y):
    sh = shadow(rim, *d)
    return sh, sheet(sh, y)


@functools.lru_cache(maxsize=4)
def build(figure):
    """Everything of a figure that is not Blender (cached), a Layout:
      body        Mesh (card frame): the body and the static pieces (white cards, papers, details)
      sheets      [(Shadow, Mesh)]: the drop shadows of the body (behind it) and of the static pieces (on it)
      hand        the father's moving piece, Mesh in the frame of its brad (the pivot at the origin); else None
      hand_sheet  its shadow (Shadow, Mesh) in the frame `hand_at` (the pivot + half the offset), turning with it
      brads       Mesh (card frame)
      pivot, tip  the hand's brad and the index fingertip at angle 0 (card frame, on the paper's front); else None
      head        the middle of the face on the paper's front
      size        [x, y, z] extent of the card (without the shadows)
      cuts        {piece: Cut} (the hand's outlines in its brad's frame)"""
    fig = FIGURES[figure]
    bases = {k: base_outline(p) for k, p in fig.pieces.items()}
    first = border(bases["body"])
    lift = -float(first[:, 1].min())
    height = float(first[:, 1].max()) + lift
    shift = np.array([0.0, lift])
    cuts = {"body": cut(figure, "body", fig.pieces["body"], bases["body"] + shift, lambda x, z: bow(z, height),
                        CARD_Y[1] - CARD_Y[0], shift)}
    body = [cuts["body"].mesh]
    sheets = [_sheet(cuts["body"].rim, SHADOW, lambda z: bow(z, height) + CARD_Y[1] + SHADOW_BACK)]
    half = (0.5 * SHADOW[0], 0.5 * SHADOW[1])
    brads, hand, hand_sheet, hand_at, pivot, tip = [], None, None, None, None, None
    for key, piece in fig.pieces.items():
        if piece.joint is None:
            continue
        j = np.asarray(piece.joint, float) + shift
        base = bases[key] + shift
        if key == "hand":                                # it sweeps a disc about its brad
            reach = float(np.linalg.norm(base - j, axis=1).max()) + RIM
            zs = np.linspace(j[1] - reach, j[1] + reach, 65)
        else:
            zs = np.linspace(base[:, 1].min() - RIM, base[:, 1].max() + RIM, 65)
        yp = flat_y(zs, height)
        c = cut(figure, key, piece, base, lambda x, z, y=yp: np.full_like(x, y), PIECE_CARD, shift)
        y_sheet = yp + CARD_Y[0] + PIECE_CARD + 0.5 * GAP
        brads.append(brad(j, yp, float(bow(j[1], height)) + CARD_Y[1]))
        if key == "hand":
            top = c.paper[np.argmax(c.paper[:, 1])]
            pivot, tip = (float(j[0]), yp, float(j[1])), (float(top[0]), yp, float(top[1]))
            hand = c.mesh.moved((-j[0], -yp, -j[1]))
            hand_at = (float(j[0] + half[0]), y_sheet, float(j[1] + half[1]))
            sh = shadow(c.rim - j, 0.0, 0.0)
            hand_sheet = (sh, sheet(sh, lambda z: np.zeros_like(z)))
            cuts[key] = Cut(hand, c.paper - j, c.rim - j)
        else:
            body.append(c.mesh)
            sheets.append(_sheet(c.rim, half, lambda z, y=y_sheet: np.full_like(z, y)))
            cuts[key] = c
    body, brads = S.merge(body), S.merge(brads)
    every = [body.V, brads.V] + ([hand.V + pivot] if hand is not None else [])
    lo, hi = np.concatenate(every).min(0), np.concatenate(every).max(0)
    hx, hz = fig.head[0], fig.head[1] + lift
    head = (float(hx), float(bow(hz, height)), float(hz))
    return Layout(figure, height, lift, body, sheets, hand, hand_sheet, hand_at, brads, pivot, tip, head,
                  [float(v) for v in hi - lo], cuts)


# ===================================================================================================================
# the wag
# ===================================================================================================================
def angle(t, wag=0.0, wagging=0.0, wag_hz=PARAMS["wag_hz"][0]):
    """Degrees the hand is turned at clip second t, counter-clockwise seen from the front: what EXPR drives."""
    return wag + wagging * math.sin(2.0 * math.pi * wag_hz * t)


def turn(deg):
    """The 3x3 rotation of the hand by `deg` about the card's normal toward the viewer (-Y): counter-clockwise seen from
    the front, i.e. Blender's rotation_euler[1] = -radians(deg)."""
    return S.axis_angle((0.0, -1.0, 0.0), deg)


def tip_at(figure, deg):
    """The index fingertip (card frame) with the hand turned by `deg`."""
    L = build(figure)
    p, t = np.asarray(L.pivot), np.asarray(L.tip)
    return p + turn(deg) @ (t - p)


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=4):
    return [round(float(x), nd) for x in v]


def objects(name, figure):
    """The objects a figure's builder makes."""
    out = [f"{name}_figure", f"{name}_shadow", f"{name}_brad"]
    return out + ([f"{name}_hand", f"{name}_hand_shadow"] if FIGURES[figure].hand is not None else [])


def card(name, figure, roles=None, start=None):
    """The prop card (docs/design.md: Prop card): `use.look` `head`; the father's `use.hand` `wag` (the hand piece, its
    brad, the turn axis toward the viewer, the index fingertip at angle 0, the root property it turns by) and `params`."""
    L = build(figure)
    use = {"look": [{"name": "head", "point": _r(L.head)}]}
    params = {}
    if L.pivot is not None:
        use["hand"] = [{"name": "wag", "object": f"{name}_hand", "pivot": _r(L.pivot), "axis": [0.0, -1.0, 0.0],
                        "tip": _r(L.tip), "prop": "wag"}]
        params = {k: float((start or {}).get(k, d)) for k, (d, _r2, _doc) in PARAMS.items()}
    return {"size": _r(L.size), "origin": "floor_center", "front": "-Y", "figure": figure, "slots": dict(roles or {}),
            "use": use, "colliders": [], "params": params, "objects": objects(name, figure),
            "form_exempt": [o for o in objects(name, figure) if o.endswith("_shadow")]}
