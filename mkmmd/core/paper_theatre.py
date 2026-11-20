"""The paper theatre (`[[set]] kind = "paper_theatre"`; docs/design.md: Sets), numpy only: the options, the drawing of
every paper piece (outlines as drawn, before the scissors), where everything stands, the motion of what moves, the
stop-motion holds and the card. The Blender side is mkmmd/blender/library/sets/paper_theatre.py; the meshes are made
in paper_theatre_parts.py beside it (numpy, with the cut-out kit's scissors and plates).

A toy theatre of construction paper and card, life-size round a figure 1.5 m tall. The set's frame (metres): she stands
at the origin facing -Y (+X her left) on the stage floor, whose top is z = 0. From the front:

  * the floor: corrugated card under `floor` paper, its cut front edge showing the flutes (FLOOR_FRONT); under it the
    stage's base (card in the proscenium's paper, a scalloped `trim` pelmet with gilt dots) down to BASE_BOTTOM;
  * the proscenium (card, its face at YP): two pillars with gilt rosettes round the opening (`width` x `height`, its top
    a shallow arch), a header, a scalloped crest with a medallion and two finials, the opening trimmed; behind it the
    scalloped valance with tassels and the two curtains, accordion-folded paper tied back with gilt bands and tassels.
    The three drop in together at `arrive` (a carrier, `<name>_frame`, falling the opening's height in DROP_FALL s, then
    a hop, a damped rock and flex) and are hidden before;
  * a strip of grass and flowers at the front of the floor (two layers, each standing on a flap folded back);
  * three clouds, five stars and a crescent moon with a rabbit cut out of it, each on a thread from a batten across the
    stage (three rows between her and the backdrop, outside KEEP_OUT, where the marionette and her strings are). From
    `arrive` (staggered) each falls from its batten, the thread catches and throws it back up two or three times like a
    yo-yo; then it swings as a pendulum of its own length and phase and twists slowly on its thread;
  * two rows of hills on folded card stands, and the backdrop: card bent round the stage (its flat middle at `depth`,
    its sides curving forward, the battens taped across between them), pencil guide lines on it, and the sun pinned to
    it by a brass split pin (the brad is its nose), its rays turning on the pin at `sun_rpm`.

Flat paper is flat on purpose (`mk_form_exempt`); it has thickness, its cuts are scissor snips (the parts module), and the
hanging pieces curl a little. Everything that moves is keyed from `motion`, held `step` times a second when `step` > 0."""
import functools
import math
import zlib
from collections import namedtuple

import numpy as np

from . import shotstyle as SS

G = 9.81
PLACE = ("name", "kind", "at", "yaw")
DEFAULTS = {"arrive": None, "stagger": 0.08, "step": 0.0, "step_from": 0.0, "sun_rpm": 1.0, "swing": 3.0, "twist": 8.0,
            "seed": 1, "colors": None, "width": 2.2, "height": 3.2, "depth": 2.35}
RANGES = {"stagger": (0.0, 1.0), "step": (0.0, 240.0), "sun_rpm": (-30.0, 30.0), "swing": (0.0, 12.0),
          "twist": (0.0, 40.0), "width": (1.8, 3.2), "height": (2.8, 3.8), "depth": (2.0, 4.0)}

# colour roles: (default colour, what it is). Materials are named <set name>_<role>.
ROLES = {
    "sky": ("surface", "the sky sheet"),
    "pencil": ("text:surface:0.55", "pencil guide lines on the sky and the floor"),
    "sun": ("gold:surface:0.3", "the sun's face"),
    "sun_rays": ("gold", "the sun's rays, turning on the split pin"),
    "ink": ("text", "the sun's eyes and smile"),
    "cheek": ("rose", "the sun's cheeks"),
    "cloud": ("#ffffff", "the clouds"),
    "cloud_shade": ("hl_med", "the clouds' bellies"),
    "star": ("gold", "stars 1, 3 and 5"),
    "star2": ("rose", "stars 2 and 4"),
    "moon": ("gold", "the crescent moon (the rabbit is cut out of it)"),
    "hill_back": ("foam", "the back row of hills"),
    "hill_front": ("pine", "the front row of hills"),
    "grass": ("pine", "the front strip of grass and the flowers' stems"),
    "grass_back": ("foam", "the strip of grass behind it"),
    "flower": ("love", "flowers"),
    "flower2": ("gold", "flowers"),
    "flower3": ("iris", "flowers"),
    "flower_eye": ("surface", "the flowers' middles"),
    "floor": ("overlay", "the stage floor's paper"),
    "cardboard": ("gold:overlay:0.62", "corrugated card: the floor's edges, the stands, the battens"),
    "cardboard_dark": ("gold:text:0.55", "inside the corrugated card, between the flutes"),
    "proscenium": ("rose", "the proscenium and the stage's base"),
    "trim": ("iris", "the trim round the opening, the base's band, the valance's second row of scallops"),
    "valance": ("love", "the scalloped valance"),
    "curtain": ("love", "the curtains"),
    "gilt": ("gold", "the tassels, tie-backs, finials and the crest's medallion"),
    "thread": ("text", "the threads"),
    "tape": ("overlay:gold:0.3", "masking tape"),
    "brad": ("gold", "the brass split pins"),
}
MATS = tuple(ROLES)                     # material index -> role

# --- the frame (fixed parts of the layout; the opening's size, the depth and the seed are options)
YP = -0.95                              # the proscenium's face
CARD = 0.025                            # the proscenium's card
PAPER = 0.004                           # construction paper
PILLAR = 0.33                           # a pillar's width
HEADER = 0.70                           # the header over the opening's sides
CREST = (0.95, 0.55)                    # the crest's half width and height over the header
ARCH = 0.10                             # the opening's top rises this much to its middle
FLOOR_FRONT = -1.08                     # the floor's front edge (its corrugated cut)
FLOOR_T = 0.05                          # the floor's card
FLUTE = 0.07                            # the corrugation's pitch
BASE_BOTTOM = -0.85                     # the stage's base hangs down to here
VALANCE = (0.28, 0.15, 7)               # the scallops' lowest point and their cusps under the opening's top; how many
CURTAIN_Y = -0.83                       # the curtains' middle plane
TIE_Z = 1.30                            # where the curtains are tied back
PLEATS = 6                              # accordion folds a curtain
GRASS_Y = (-0.64, -0.58)                # the grass strips' faces (front, back)
KEEP_OUT = (0.85, 0.6)                  # nothing hangs in |x| < 0.85, |y| < 0.6 (the marionette and her strings)
THREAD_R = 0.001                        # a thread's radius (2 mm thick)
TAPE = (0.036, 0.055, 0.0012)           # a tape tab's width, length, thickness
BATTEN_R = 0.012
SKY_BEND = 0.5                          # the backdrop bends forward on this radius at each side
SKY_FRONT = 0.2                         # its sides run forward to this y
SKY_CARD = 0.012                        # its card
SKY_VIEW = 6.5                          # the backdrop's sides hide behind the pillars seen from this far in front of her

# --- the motion
DROP_FALL = 0.30                        # s the frame takes to fall in
FRAME_HOP = 0.12                        # the frame's bounce: share of its landing speed
ROCK = (2.2, 2.0, 1.0)                  # the frame's rock about its foot: degrees, Hz, s to settle
FLEX = (1.0, 3.2, 0.9)                  # its flex (shape key value at 1: FLEX_TOP m at the crest), Hz, s to settle
FLEX_TOP = 0.08
CURTAIN_SWAY = (4.0, 1.4, 1.2)          # the curtains' swing about their top: degrees, Hz, s to settle
CURTAIN_SIDE = (2.5, 1.1, 1.3)          # and in their own plane, the swags swinging out and back
CURTAIN_FLEX = (1.0, 2.3, 1.1)          # their billow (shape key value at 1: CURTAIN_BILLOW m), Hz, s to settle
CURTAIN_BILLOW = 0.10
PIECE_DELAY = 0.12                      # s after `arrive` the first hanging piece is let go
YOYO = {"g": 2.2 * G, "hz": 4.0, "e": 0.45, "catches": 3, "zeta": 0.25, "wound": 0.02}
SWING_RAMP = 1.2                        # s the swing takes to grow after the first catch
TWIST_PERIOD = (3.5, 6.5)


class TheatreError(ValueError):
    """A [[set]] paper_theatre table the builder cannot use."""


# ===================================================================================================================
# options and colours
# ===================================================================================================================
def _number(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise TheatreError(f"paper_theatre: {what} = {v!r} is not a number")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise TheatreError(f"paper_theatre: {what} = {v:g} is out of range ({lo:g} .. {hi:g})")
    return v


def options(spec):
    """The [[set]] table checked and filled with DEFAULTS. An unknown key is an error that lists the known ones."""
    name = spec.get("name", "?")
    unknown = sorted(set(spec) - set(DEFAULTS) - set(PLACE))
    if unknown:
        raise TheatreError(f"paper_theatre {name!r}: unknown keys {unknown} (known: {sorted(DEFAULTS)})")
    o = {}
    for k, d in DEFAULTS.items():
        v = spec.get(k, d)
        if k == "arrive":
            o[k] = None if v is None else _number(v, k)
        elif k == "seed":
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise TheatreError(f"paper_theatre: seed = {v!r}: a whole number, 0 or more")
            o[k] = v
        elif k == "colors":
            v = {} if v is None else v
            if not isinstance(v, dict):
                raise TheatreError(f"paper_theatre: colors = {v!r}: a table of role = colour")
            bad = sorted(set(v) - set(ROLES))
            if bad:
                raise TheatreError(f"paper_theatre: colors: unknown roles {bad} (roles: {', '.join(ROLES)})")
            o[k] = dict(v)
        elif k == "step_from":
            o[k] = _number(v, k)
        else:
            o[k] = _number(v, k, *RANGES[k])
    return o


def colours(opts, palette):
    """{role: (r, g, b)} display-space sRGB floats of every role: ROLES' defaults with `colors` over them."""
    out = {}
    for role, (default, _what) in ROLES.items():
        spec = opts["colors"].get(role, default)
        try:
            out[role] = tuple(float(c) for c in SS.resolve_colour(spec, palette))
        except SS.StyleError as e:
            raise TheatreError(f"paper_theatre: colors.{role} = {spec!r}: {e}") from None
    return out


# ===================================================================================================================
# 2D drawing: shapes as fields (positive inside), outlines by marching squares
# ===================================================================================================================
def polygon_area(P):
    P = np.asarray(P, float)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def ccw(P):
    P = np.asarray(P, float)
    return P if polygon_area(P) > 0 else P[::-1].copy()


def centroid(P):
    """The area centroid of the closed polygon P."""
    P = np.asarray(P, float)
    x, z = P[:, 0], P[:, 1]
    c = x * np.roll(z, -1) - np.roll(x, -1) * z
    a = 0.5 * c.sum()
    return np.array([((x + np.roll(x, -1)) * c).sum(), ((z + np.roll(z, -1)) * c).sum()]) / (6.0 * a)


def resample(P, step):
    """Points every `step` along the closed polygon P (from its first point)."""
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    n = max(int(round(s[-1] / step)), 8)
    q = np.linspace(0.0, s[-1], n + 1)[:-1]
    return np.stack([np.interp(q, s, Q[:, 0]), np.interp(q, s, Q[:, 1])], 1)


def densify(P, step):
    """The closed polygon P with every edge cut into pieces no longer than `step` (its corners kept)."""
    P = np.asarray(P, float)
    E = np.roll(P, -1, 0) - P
    k = np.maximum(np.ceil(np.linalg.norm(E, axis=1) / step).astype(int), 1)
    return np.concatenate([P[i] + E[i] * (np.arange(k[i])[:, None] / k[i]) for i in range(len(P))])


def inside(Q, P):
    """Even-odd: are the points Q (q, 2) inside the closed polygon P."""
    Q, P = np.asarray(Q, float), np.asarray(P, float)
    A, B = P, np.roll(P, -1, 0)
    out = np.zeros(len(Q), bool)
    for lo in range(0, len(Q), 4096):
        x, z = Q[lo:lo + 4096, :1], Q[lo:lo + 4096, 1:]
        cond = (A[None, :, 1] > z) != (B[None, :, 1] > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = A[None, :, 0] + (z - A[None, :, 1]) * (B[None, :, 0] - A[None, :, 0]) / (B[None, :, 1] - A[None, :, 1])
        out[lo:lo + 4096] = (cond & (x < xi)).sum(1) % 2 == 1
    return out


def edge_distance(Q, P):
    """Distance from the points Q (q, 2) to the edges of the closed polygon P."""
    Q, P = np.asarray(Q, float), np.asarray(P, float)
    A, AB = P, np.roll(P, -1, 0) - P
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    out = np.empty(len(Q))
    for lo in range(0, len(Q), 2048):
        q = Q[lo:lo + 2048]
        t = np.clip(((q[:, None] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        d = q[:, None] - (A[None] + t[..., None] * AB[None])
        out[lo:lo + 2048] = np.sqrt((d * d).sum(2).min(1))
    return out


def disc(cx, cz, r):
    return lambda X, Z: r - np.hypot(X - cx, Z - cz)


def oval(cx, cz, a, b, deg=0.0):
    """An ellipse, semi-axes a and b along its own x and z, turned `deg` counter-clockwise."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)

    def f(X, Z):
        dx, dz = X - cx, Z - cz
        return (1.0 - np.hypot((c * dx + s * dz) / a, (-s * dx + c * dz) / b)) * min(a, b)
    return f


def box(x0, z0, x1, z1):
    return lambda X, Z: np.minimum(np.minimum(X - x0, x1 - X), np.minimum(Z - z0, z1 - Z))


def below(z0):
    return lambda X, Z: z0 - Z


def above(z0):
    return lambda X, Z: Z - z0


def shape(P):
    """The closed polygon P as a field (signed distance, positive inside)."""
    P = np.asarray(P, float)

    def f(X, Z):
        Q = np.stack([X.ravel(), Z.ravel()], 1)
        d = edge_distance(Q, P)
        return np.where(inside(Q, P), d, -d).reshape(X.shape)
    return f


def union(*fs):
    return lambda X, Z: np.max(np.stack([f(X, Z) for f in fs]), 0)


def inter(*fs):
    return lambda X, Z: np.min(np.stack([f(X, Z) for f in fs]), 0)


def minus(a, *bs):
    return lambda X, Z: np.min(np.stack([a(X, Z)] + [-b(X, Z) for b in bs]), 0)


def grow(f, d):
    return lambda X, Z: f(X, Z) + d


# marching squares: the segments of each cell case between its edges B(ottom) R(ight) T(op) L(eft); 5 and 10 are saddles
_CASES = {1: ("LB",), 2: ("BR",), 3: ("LR",), 4: ("RT",), 6: ("BT",), 7: ("LT",), 8: ("TL",), 9: ("BT",), 11: ("RT",),
          12: ("LR",), 13: ("BR",), 14: ("LB",)}


def contours(F, xs, zs):
    """Closed loops (each (n, 2)) where the field F (len(zs), len(xs)) crosses 0, its border taken as outside."""
    F = np.array(F, float)
    F[0], F[-1], F[:, 0], F[:, -1] = -1.0, -1.0, -1.0, -1.0
    F[np.abs(F) < 1e-12] = -1e-12
    nx = len(xs)
    ins = F > 0
    case = ins[:-1, :-1] * 1 + ins[:-1, 1:] * 2 + ins[1:, 1:] * 4 + ins[1:, :-1] * 8
    I, J = np.nonzero((case > 0) & (case < 15))

    def key(edge, i, j):
        return {"B": 2 * (i * nx + j), "T": 2 * ((i + 1) * nx + j), "L": 2 * (i * nx + j) + 1,
                "R": 2 * (i * nx + j + 1) + 1}[edge]

    segs = []
    for i, j in zip(I.tolist(), J.tolist()):
        c = int(case[i, j])
        if c in (5, 10):
            mid = F[i, j] + F[i, j + 1] + F[i + 1, j + 1] + F[i + 1, j] > 0
            pairs = ("BR", "TL") if (c == 5) == mid else ("LB", "RT")
        else:
            pairs = _CASES[c]
        segs += [(key(p[0], i, j), key(p[1], i, j)) for p in pairs]
    at = {}
    for s, (a, b) in enumerate(segs):
        at.setdefault(a, []).append(s)
        at.setdefault(b, []).append(s)

    def point(k):
        e, v = divmod(k, 2)
        i, j = divmod(e, nx)
        a = F[i, j]
        if v == 0:                                            # along x from (i, j) to (i, j + 1)
            t = a / (a - F[i, j + 1])
            return (xs[j] + t * (xs[j + 1] - xs[j]), zs[i])
        t = a / (a - F[i + 1, j])                             # along z from (i, j) to (i + 1, j)
        return (xs[j], zs[i] + t * (zs[i + 1] - zs[i]))

    used = np.zeros(len(segs), bool)
    loops = []
    for s0 in range(len(segs)):
        if used[s0]:
            continue
        used[s0] = True
        start, cur = segs[s0]
        ring = [start]
        while cur != start:
            ring.append(cur)
            nxt = [s for s in at[cur] if not used[s]]
            if not nxt:
                break
            used[nxt[0]] = True
            a, b = segs[nxt[0]]
            cur = b if a == cur else a
        if len(ring) > 2:
            loops.append(np.array([point(k) for k in ring]))
    return loops


def outline(field, lo, hi, px):
    """The outline of {field > 0} inside the box lo..hi (x, z): the biggest loop marching squares finds on a grid of `px`,
    counter-clockwise."""
    xs = np.arange(lo[0] - 3 * px, hi[0] + 3 * px, px)
    zs = np.arange(lo[1] - 3 * px, hi[1] + 3 * px, px)
    X, Z = np.meshgrid(xs, zs)
    loops = contours(field(X, Z), xs, zs)
    if not loops:
        raise TheatreError("paper_theatre: a drawing came out empty")
    return ccw(max(loops, key=lambda L: abs(polygon_area(L))))


def drawn(field, lo, hi, px, step=0.005):
    return resample(outline(field, lo, hi, px), step)


# ===================================================================================================================
# the pieces as drawn (each in its own frame, metres)
# ===================================================================================================================
def _rng(seed, *key):
    return np.random.default_rng([int(seed), zlib.crc32(":".join(str(k) for k in key).encode())])


def cloud_fields(w, h, rng):
    """A cloud w wide and h tall, its flat bottom on z = 0, centred on x = 0: puffs along a dome over a wide oval, cut
    flat underneath; and its belly (a band along the bottom, inset). (cloud field, belly field)."""
    n = int(rng.integers(4, 6))
    u = np.linspace(-1.0, 1.0, n)
    xs = (0.34 * u + rng.uniform(-0.025, 0.025, n)) * w
    r = w * (0.19 + 0.06 * (1.0 - np.abs(u))) * rng.uniform(0.9, 1.1, n)
    top = h * (1.0 - 0.32 * (xs / (0.5 * w)) ** 2)
    puffs = [disc(x, t - rr, rr) for x, t, rr in zip(xs, top, r)]
    body = inter(union(*puffs, oval(0.0, 0.32 * h, 0.47 * w, 0.30 * h)), above(0.012 * h))
    wave = 0.36 * h + 0.035 * h * np.sin(np.linspace(0.0, 2.0 * math.pi, 9) + rng.uniform(0, 6.28))
    edge = [(-w, -1.0), (w, -1.0)] + [(x, z) for x, z in zip(np.linspace(w, -w, 9), wave)]
    belly = inter(grow(body, -0.022), shape(edge))
    return body, belly


def star_outline(R, rng, points=5, ratio=0.46):
    """A five-pointed star of radius R about the origin, one point up, cut by hand (every point a little off)."""
    k = np.arange(2 * points)
    a = 0.5 * math.pi + k * math.pi / points + rng.uniform(-0.045, 0.045, 2 * points)
    rad = np.where(k % 2 == 0, R * rng.uniform(0.93, 1.05, 2 * points), R * ratio * rng.uniform(0.94, 1.06, 2 * points))
    return ccw(densify(np.stack([rad * np.cos(a), rad * np.sin(a)], 1), 0.006))


def rabbit_field(cx, cz, s):
    """A rabbit sitting in profile, facing -x, `s` tall from its feet to its ear tips, its feet's middle at (cx, cz)."""
    parts = [disc(0.14, 0.24, 0.235), oval(-0.05, 0.33, 0.23, 0.21, -25.0), oval(-0.21, 0.58, 0.17, 0.135, -15.0),
             oval(-0.05, 0.85, 0.055, 0.21, -20.0), oval(-0.15, 0.83, 0.05, 0.19, -7.0), disc(0.37, 0.30, 0.07),
             oval(-0.23, 0.06, 0.085, 0.055), oval(0.10, 0.045, 0.22, 0.045)]
    k = s / 1.06
    return lambda X, Z: union(*parts)((X - cx) / k, (Z - cz) / k) * k


def moon_fields(R):
    """A crescent moon (the outer disc of radius R less a disc set off up and to the left: its horns point up-left) and
    the sitting rabbit cut out of its fat part. (moon field, rabbit field)."""
    moon = minus(disc(0.0, 0.0, R), disc(-0.50 * R, 0.20 * R, 0.84 * R))
    rabbit = rabbit_field(0.60 * R, -0.50 * R, 0.52 * R)
    return moon, rabbit


def sun_fields(r):
    """The sun's face (a disc of r) and its features about its middle: (face, rays, eyes, smile, cheeks), fields but for
    the rays, a 12-pointed star outline."""
    k = np.arange(24)
    a = k * math.pi / 12
    rad = np.where(k % 2 == 0, 1.70 * r, 1.04 * r)
    rays = ccw(densify(np.stack([rad * np.cos(a), rad * np.sin(a)], 1), 0.01))
    eyes = union(oval(-0.30 * r, 0.20 * r, 0.065 * r, 0.095 * r), oval(0.30 * r, 0.20 * r, 0.065 * r, 0.095 * r))
    smile = inter(minus(disc(0.0, 0.06 * r, 0.40 * r), disc(0.0, 0.13 * r, 0.40 * r)), below(-0.10 * r))
    cheeks = union(oval(-0.52 * r, -0.12 * r, 0.14 * r, 0.10 * r), oval(0.52 * r, -0.12 * r, 0.14 * r, 0.10 * r))
    return disc(0.0, 0.0, r), rays, eyes, smile, cheeks


def ridge(x, humps, base):
    """Heights of a row of hills at x: `base` plus cosine humps (centre, height, half width)."""
    z = np.full_like(np.asarray(x, float), base)
    for c, h, w in humps:
        u = np.clip((x - c) / w, -1.0, 1.0)
        z = z + h * np.cos(0.5 * math.pi * u) ** 2
    return z


def hills_outline(x0, x1, humps, base, foot=0.35):
    """A row of hills from x0 to x1 standing on z = 0: `ridge` over it, sloping down to the floor over `foot` m at each
    end (so an end never shows as a cut edge)."""
    x = np.linspace(x1, x0, max(int((x1 - x0) / 0.01), 8))
    z = ridge(x, humps, base) * (0.06 + 0.94 * smoothstep(np.minimum(x - x0, x1 - x) / foot))
    return ccw(np.vstack([[[x0, 0.0], [x1, 0.0]], np.stack([x, z], 1)]))


def grass_outline(x0, x1, rng, band, blade):
    """A strip of grass from x0 to x1 standing on z = 0: a band `band` high and blades along its top (a zigzag, their
    heights in `blade`, some leaning, each tip left blunt by the scissors)."""
    tips, x = [], x1
    while x > x0 + 0.02:
        w = float(rng.uniform(0.028, 0.055))
        xl = max(x - w, x0)
        lean = float(rng.uniform(-0.45, 0.45)) * w
        tip, top = 0.5 * (x + xl) + lean, band + float(rng.uniform(*blade))
        tips += [(x, band), (tip + 0.0025, top), (tip - 0.0025, top - 0.002)]
        x = xl
    pts = [(x0, 0.0), (x1, 0.0)] + tips + [(x0, band)]
    return ccw(densify(np.array(pts), 0.01))


def flower_fields(r, petals, rng):
    """A daisy of radius r about the origin: (petals, middle)."""
    a0 = float(rng.uniform(0.0, math.pi))
    ps = [oval(0.58 * r * math.cos(a0 + 2 * math.pi * i / petals), 0.58 * r * math.sin(a0 + 2 * math.pi * i / petals),
               0.44 * r, 0.27 * r, math.degrees(a0 + 2 * math.pi * i / petals)) for i in range(petals)]
    return union(*ps, disc(0.0, 0.0, 0.4 * r)), disc(0.0, 0.0, 0.33 * r)


def tape_outline(w, h, rng):
    """A tab of masking tape w wide and h long about the origin, its two ends torn (a zigzag across)."""
    n = 5
    xs = np.linspace(-0.5 * w, 0.5 * w, n + 1)
    top = [(x, 0.5 * h + float(rng.uniform(-0.12, 0.12)) * w) for x in xs]
    bottom = [(x, -0.5 * h + float(rng.uniform(-0.12, 0.12)) * w) for x in xs[::-1]]
    return ccw(np.array(bottom[::-1][:1] + bottom[::-1][1:] + top[::-1]))


# ===================================================================================================================
# the layout
# ===================================================================================================================
# a paper plate: role (material), outline (n, 2) CCW (x, z) as drawn, holes, y of its front face (its back at y + thick),
# bow (the face bulges toward -y by `amount` across x0..x1, 0 at the sides) or None, cut: scissor (snip, wobble, step)
# or None (an uncut edge), frame: None or (R (3, 3), t (3,)) taking the plate's own (x, y, z) to the object's, edge: the
# role of its rim and back when they are another material (paper glued on card) or None
Plate = namedtuple("Plate", "role outline holes y thick bow cut frame edge", defaults=(None,))
Part = namedtuple("Part", "name plates tubes brads special at", defaults=((0.0, 0.0, 0.0),))   # one object (origin `at`)
Hang = namedtuple("Hang", "name kind row anchor length size part period phase twist_period twist_phase swing order")
Curtain = namedtuple("Curtain", "name side hinge outer inner_z inner_x tie_z pleats paper part")
Layout = namedtuple("Layout", "width height depth seed rows statics frame curtains sun hangs anchor sky floor")

SNIP_BIG = (0.16, 0.0025, 0.03)          # long knife-and-scissor cuts on card and big sheets
SNIP = (0.08, 0.0016, 0.012)             # paper pieces
SNIP_FINE = (0.04, 0.0008, 0.006)        # small details


def _plate(role, P, y, thick, cut=SNIP, bow=None, holes=(), frame=None, edge=None):
    return Plate(role, ccw(P), tuple(ccw(h) for h in holes), float(y), float(thick), bow, cut, frame, edge)


def bow_y(bow, x):
    """How far a plate's bow moves its face along y at x: -amount (1 - u^2), u from -1 at x0 to 1 at x1."""
    x = np.asarray(x, float)
    if bow is None:
        return np.zeros_like(x)
    amount, x0, x1 = bow
    u = np.clip((2.0 * x - x0 - x1) / (x1 - x0), -1.0, 1.0)
    return -amount * (1.0 - u * u)


def _layers(role_fields, lo, hi, px, y0, step=0.01, layer=PAPER * 0.75, thick=PAPER * 0.6, cut=SNIP_FINE):
    """Paper details glued on in front of a piece whose face is at y0: each (role, field) a layer `layer` further out,
    its outline sampled every `step`."""
    out = []
    for k, (role, f) in enumerate(role_fields):
        out.append(_plate(role, drawn(f, lo, hi, px, step), y0 - (k + 1) * layer - thick, thick, cut))
    return out


# where the hanging pieces hang: name, kind, row (0 front .. 2 back), x as a share of the opening's half width, the
# thread's end as a share of the opening's height, size (a cloud's width, a star's radius, the moon's radius)
HANGS = (("cloud1", "cloud", 1, -0.636, 0.875, 0.74), ("cloud2", "cloud", 2, 0.727, 1.0, 0.68),
         ("cloud3", "cloud", 2, -1.0, 0.672, 0.62), ("star1", "star", 0, -0.864, 0.9375, 0.115),
         ("star2", "star", 1, 1.0, 0.766, 0.095), ("star3", "star", 1, 0.409, 0.719, 0.085),
         ("star4", "star", 2, 1.227, 0.78, 0.105), ("star5", "star", 2, -0.409, 0.734, 0.09),
         ("moon", "moon", 0, 0.773, 0.875, 0.31))


def rows(depth):
    """The y of the three rows of battens the pieces hang from (front to back)."""
    return (max(0.80, 0.34 * depth), 0.575 * depth, 0.83 * depth)


def _hang_part(name, kind, size, rng):
    """(plates in the piece's own frame, its size (w, h)): the thread's end at the origin, the paper's face just behind
    it (the piece hanging below, its middle right under the thread) and a tab of tape over it in front."""
    if kind == "cloud":
        w, h = size, 0.50 * size
        body, belly = cloud_fields(w, h, rng)
        P = drawn(body, (-0.6 * w, 0.0), (0.6 * w, h), 0.004)
        lo, hi = (-0.6 * w, 0.0), (0.6 * w, h)
        layers = [("cloud_shade", belly)]
        role, holes, cut = "cloud", (), SNIP
    elif kind == "star":
        P = star_outline(size, rng)
        role, holes, layers, cut = ("star" if name in ("star1", "star3", "star5") else "star2"), (), [], SNIP_FINE
    else:
        moon, rabbit = moon_fields(size)
        lo, hi = (-1.1 * size, -1.1 * size), (1.1 * size, 1.1 * size)
        P = drawn(moon, lo, hi, 0.0025, 0.004)
        holes = (drawn(rabbit, lo, hi, 0.0012, 0.003),)
        role, layers, cut = "moon", [], SNIP_FINE
    c = centroid(P)
    near = np.abs(P[:, 0] - c[0]) < 0.02
    top = float(P[near, 1].max()) if near.any() else float(P[:, 1].max())
    attach = np.array([c[0], top - 0.032])
    bow = (0.006 * (P[:, 0].max() - P[:, 0].min()), P[:, 0].min() - attach[0], P[:, 0].max() - attach[0])
    face = THREAD_R + 0.0002 - float(bow_y(bow, np.array([0.0]))[0])    # the paper's face just behind the thread there
    plates = [_plate(role, P - attach, face, PAPER, cut, bow, [h - attach for h in holes])]
    if layers:
        plates += [_plate(r, drawn(f, lo, hi, 0.003, 0.003) - attach, face - 0.0026, 0.0024, SNIP_FINE, bow)
                   for r, f in layers]
    trng = np.random.default_rng(int(rng.integers(1 << 30)))
    tape = tape_outline(TAPE[0], TAPE[1], trng)
    a = math.radians(float(trng.uniform(-8.0, 8.0)))
    tape = tape @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    plates.append(_plate("tape", tape + (0.0, 0.006), -THREAD_R - 0.0001 - TAPE[2], TAPE[2], None))
    lo2, hi2 = P.min(0) - attach, P.max(0) - attach
    return plates, (float(hi2[0] - lo2[0]), float(hi2[1] - lo2[1]))


def _proscenium(hw, H):
    """The proscenium's plates in the frame carrier's frame (its foot's middle at the origin, its face at y = 0)."""
    outer = hw + PILLAR
    top = H + HEADER
    cw, ch = CREST
    n = 11
    a = np.linspace(math.pi * 0.06, math.pi * 0.94, n)
    scal = [disc(cw * math.cos(t), top + ch * math.sin(t) * 0.94, 0.075) for t in a]
    crest = union(inter(oval(0.0, top, cw, ch), above(top - 0.05)), *scal)
    finials = [union(disc(s * (outer - 0.13), top + 0.15, 0.075), box(s * (outer - 0.13) - 0.025, top - 0.05,
                                                                     s * (outer - 0.13) + 0.025, top + 0.12))
               for s in (-1.0, 1.0)]
    def opening(X, Z):                                    # the opening, its top a shallow arch; open at the foot
        return np.minimum(hw - np.abs(X), H + ARCH * (1.0 - (X / hw) ** 2) - Z)
    frame = minus(union(box(-outer, 0.0, outer, top), crest, *finials), opening)
    lo, hi = (-outer - 0.05, -0.05), (outer + 0.05, top + ch + 0.15)
    P = drawn(frame, lo, hi, 0.006, 0.012)
    trim = inter(minus(grow(opening, 0.085), grow(opening, 0.035)), above(0.03), frame)
    medal = disc(0.0, top + 0.24, 0.17)
    star = star_outline(0.11, np.random.default_rng(3)) + (0.0, top + 0.24)
    plates = [_plate("proscenium", P, 0.0, CARD, SNIP_BIG)]
    plates += _layers([("trim", trim)], lo, hi, 0.005, 0.0, step=0.012, cut=SNIP)
    plates += _layers([("gilt", medal)], (-0.3, top), (0.3, top + 0.5), 0.003, 0.0, step=0.004)
    plates.append(_plate("trim", star, -2 * PAPER * 0.75 - PAPER * 0.6, PAPER * 0.6, SNIP_FINE))
    xm = 0.5 * (hw + 0.085 + outer)                        # down the middle of each pillar beside its trim
    for s in (-1.0, 1.0):
        for z in np.linspace(0.45, top - 0.35, 4):
            dot = union(disc(s * xm, z, 0.05), *[disc(s * xm + 0.062 * math.cos(a), z + 0.062 * math.sin(a), 0.022)
                                                 for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)])
            plates.append(_plate("gilt", drawn(dot, (s * xm - 0.1, z - 0.1), (s * xm + 0.1, z + 0.1), 0.0015, 0.003),
                                 -PAPER * 0.75 - PAPER * 0.6, PAPER * 0.6, SNIP_FINE))
    return plates, top + ch


def _valance(hw, H):
    """The valance's plates (behind the proscenium, in the carrier's frame) and the x of its inner cusps."""
    low, cusp, n = VALANCE
    x0, x1 = -hw - 0.09, hw + 0.09
    sw = (x1 - x0) / n
    zc = H - cusp
    cx = x0 + sw * (np.arange(n) + 0.5)
    top = H + 0.22
    body = union(box(x0, zc, x1, top), *[oval(x, zc, 0.5 * sw + 0.002, low - cusp) for x in cx])
    under = union(box(x0, zc, x1, top - 0.1), *[oval(x, zc - 0.035, 0.5 * sw + 0.002, low - cusp) for x in cx])
    lo, hi = (x0 - 0.05, zc - low), (x1 + 0.05, top + 0.05)
    y = CARD + 0.008
    cusps = x0 + sw * np.arange(1, n)
    plates = [_plate("trim", drawn(under, lo, hi, 0.004, 0.008), y + PAPER + 0.003, PAPER, SNIP),
              _plate("valance", drawn(body, lo, hi, 0.004, 0.008), y, PAPER, SNIP)]
    for x in cusps:
        f = union(oval(x, zc - 0.075, 0.021, 0.036), box(x - 0.004, zc - 0.05, x + 0.004, zc + 0.01))
        plates.append(_plate("gilt", drawn(f, (x - 0.05, zc - 0.13), (x + 0.05, zc + 0.03), 0.0015, 0.003),
                             y - 0.004, 0.003, SNIP_FINE))
    return plates


def _curtain(side, hw, H, seed):
    """One curtain (side -1 her right, at -x; +1 her left), in the carrier's frame: its hinge (the middle of its top),
    its outer edge's x, its inner edge (z, x) from the floor to the top, the tie, the pleats, the paper's width, and its
    `part`: the gilt tassel hanging on the inner side of the tie (a Plate in the hinge's frame)."""
    top = H + 0.15
    xo = side * (hw + 0.10)
    x_top = side * (0.62 * hw / 1.1)
    p0, c1, p1 = np.array([x_top, top]), np.array([side * 0.66 * hw / 1.1, 2.1]), np.array([side * (hw - 0.08), TIE_Z])
    c2, p2 = np.array([side * (hw - 0.03), 0.6]), np.array([side * (hw - 0.22), 0.03])
    s = np.linspace(0.0, 1.0, 40)[:, None]
    upper = (1 - s) ** 2 * p1 + 2 * (1 - s) * s * c1 + s ** 2 * p0          # the tie up to the top
    lower = (1 - s) ** 2 * p2 + 2 * (1 - s) * s * c2 + s ** 2 * p1          # the floor up to the tie
    inner = np.vstack([lower, upper[1:]])
    width = np.abs(inner[:, 0] - xo)
    paper = float(width.max()) * 1.06
    hinge = (0.5 * (xo + x_top), CURTAIN_Y - YP, top)
    name = "curtain_R" if side < 0 else "curtain_L"
    c = Curtain(name, side, hinge, xo, inner[:, 1], inner[:, 0], TIE_Z, PLEATS, paper, None)
    _x, ys = pleat(c, np.linspace(0.0, 1.0, 8 * PLEATS + 1), np.full(8 * PLEATS + 1, TIE_Z))
    fringe = [box(-0.026 + 0.0105 * k, -0.16, -0.019 + 0.0105 * k, -0.08) for k in range(5)]
    tassel = union(oval(0.0, -0.075, 0.026, 0.040), box(-0.006, -0.04, 0.006, 0.004), *fringe)
    P = drawn(tassel, (-0.05, -0.18), (0.05, 0.02), 0.0012, 0.003)
    at = np.array([p1[0] - side * 0.02 - hinge[0], TIE_Z - 0.025 - top])
    y = -(float(np.abs(ys).max()) + 0.016)
    return c._replace(part=_plate("gilt", P + at, y, 0.003, SNIP_FINE))


def pleat(curtain, u, z):
    """The accordion fold: (x, y offset from the curtain's middle plane) at the share u (0 outer edge .. 1 inner edge)
    of its width at height z (arrays). The paper keeps its width: where the curtain is gathered the folds are deeper."""
    xi = np.interp(z, curtain.inner_z, curtain.inner_x)
    x = curtain.outer + u * (xi - curtain.outer)
    w = np.abs(xi - curtain.outer)
    k = 2 * curtain.pleats
    depth = np.sqrt(np.maximum((curtain.paper / k) ** 2 - (w / k) ** 2, 0.0))
    tri = 1.0 - 2.0 * np.abs((u * k) % 2.0 - 1.0)                      # -1 .. 1, a fold every 1/k of the width
    return x, 0.5 * depth * tri


def curtain_billow(curtain, z):
    """The curtains' flex shape at value 1 (m along -y): the free skirt under the tie swings, the swag over it bellies."""
    z = np.asarray(z, float)
    top = curtain.inner_z.max()
    skirt = np.clip((curtain.tie_z - z) / curtain.tie_z, 0.0, 1.0) ** 2
    swag = np.sin(math.pi * np.clip((z - curtain.tie_z) / (top - curtain.tie_z), 0.0, 1.0)) * 0.45
    return CURTAIN_BILLOW * np.where(z < curtain.tie_z, skirt, swag)


def proscenium_flex(z, height):
    """The proscenium's flex shape at value 1 (m along -y): the top leans out, the foot stays."""
    return FLEX_TOP * np.clip(np.asarray(z, float) / height, 0.0, 1.0) ** 2


def sky_shape(width, depth):
    """The backdrop's plan: (a, R, D, y_front, S): its flat middle's half width a at y = D, the radius R it bends forward
    on at each side, the y its straight sides run forward to, and its half length S along the plan. It is as wide as
    can be while its sides stay behind the pillars seen from SKY_VIEW m in front of her (the pulled-back view); a 3/4
    view from 2 m out at 35 degrees then sees its curved side instead of what is beyond it."""
    R, yf, t = SKY_BEND, SKY_FRONT, SKY_CARD
    a = (0.5 * width + PILLAR) * (yf + SKY_VIEW) / (YP + SKY_VIEW) - R - t - 0.03
    return a, R, float(depth), yf, a + 0.5 * math.pi * R + (depth - R - yf)


def sky_plan(s, sky):
    """(x, y, nx, ny) of the backdrop's face at arc lengths s from its middle (s < 0 toward -x) and its unit normal
    toward the stage."""
    a, R, D, _yf, _S = sky
    s = np.asarray(s, float)
    sg, u = np.where(s < 0.0, -1.0, 1.0), np.abs(s)
    phi = np.clip((u - a) / R, 0.0, 0.5 * math.pi)
    along = np.maximum(u - a - 0.5 * math.pi * R, 0.0)
    x = np.where(u <= a, s, sg * (a + R * np.sin(phi)))
    y = np.where(u <= a, D, D - R + R * np.cos(phi)) - along
    nx = np.where(u <= a, 0.0, -sg * np.sin(phi))
    ny = np.where(u <= a, -1.0, -np.cos(phi))
    return x, y, nx, ny


def sky_side(y, sky):
    """The x (> 0) of the backdrop's face at the depth y (its sides are mirror images)."""
    a, R, D, _yf, _S = sky
    dy = np.clip(np.asarray(y, float) - (D - R), 0.0, R)
    return a + np.sqrt(np.maximum(R * R - dy * dy, 0.0))


def sky_bend(V, sky):
    """Points (n, 3) of the unrolled backdrop (x the arc length along its plan, y behind its face, z up) bent onto its
    plan in the set's frame: paper bends without stretching."""
    V = np.asarray(V, float)
    x, y, nx, ny = sky_plan(V[:, 0], sky)
    return np.stack([x - V[:, 1] * nx, y - V[:, 1] * ny, V[:, 2]], 1)


@functools.lru_cache(maxsize=8)
def layout(width=2.2, height=3.2, depth=2.35, seed=1):
    """Everything that is not Blender: a Layout. Objects' plates are in their own frames:
      statics   [Part]: floor (with its corrugated edge and pencil lines), base, sky (unrolled: its x is the arc length
                along its plan, the parts module bends it; pencil lines, stands), sun and sun_rays (about the pin, `at`),
                hills_back, hills_front, grass, grass_back, flies (the battens between the backdrop's sides)
      frame     (Part proscenium (with the valance), the carrier's place (0, YP, 0)): the carrier's frame has its origin on
                the proscenium's foot, its face at y = 0
      curtains  [Curtain] (their hinge in the carrier's frame)
      sun       (x, y, z) of its pin
      hangs     [Hang]: anchor (x, y, z) on its batten, the thread's length at rest, the piece's plates (its frame: the
                thread's end at the origin), its pendulum's period and phases
      anchor    the battens' height; rows the battens' y; sky `sky_shape`; floor (x0, x1, y0, y1)"""
    hw, H, D = 0.5 * width, float(height), float(depth)
    rng = lambda *k: _rng(seed, *k)        # noqa: E731
    sky = sky_shape(width, D)
    a, R, _D, _yf, S = sky
    side = a + R
    fx, fy0, fy1 = side + 0.05, FLOOR_FRONT, D + 0.12
    ff = hw + PILLAR + 0.07                                # the floor's front edge: a little wider than the proscenium
    za = H + 1.0
    sky_top = H + 1.25
    statics = []

    # the floor: corrugated card on its side, `floor` paper on top, the flutes in front, narrowing to the proscenium's
    # width at its front edge; under it the stage's base
    outline = np.array([[-ff, fy0], [ff, fy0], [fx, SKY_FRONT], [fx, fy1], [-fx, fy1], [-fx, SKY_FRONT]])
    floor = _plate("floor", outline, 0.0, FLOOR_T, (0.16, 0.0012, 0.03),
                   frame=(((1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0, 0.0, 0.0)), edge="cardboard")
    yb, yf = 0.745 * D, 0.45 * D
    pencil = [((-0.75 * side, yb - 0.03), (0.75 * side, yb - 0.03)), ((-0.85 * side, yf - 0.03), (0.85 * side, yf - 0.03)),
              ((-0.85 * hw, fy0 + 0.10), (0.85 * hw, fy0 + 0.10)), ((0.0, fy0 + 0.12), (0.0, -0.15)),
              ((-0.06, -0.06), (0.06, 0.06)), ((-0.06, 0.06), (0.06, -0.06))]
    statics.append(Part("floor", [floor], [], [], {"corrugated": (-ff, ff, fy0, -FLOOR_T, 0.0),
                                                     "pencil_floor": pencil}))
    base = np.array([[-ff, BASE_BOTTOM], [ff, BASE_BOTTOM], [ff, -FLOOR_T], [-ff, -FLOOR_T]])
    n_sc = int(round(2.0 * ff / 0.26))
    sw = 2.0 * ff / n_sc
    cx = -ff + sw * (np.arange(n_sc) + 0.5)
    pelmet = union(box(-ff + 0.02, -0.15, ff - 0.02, -FLOOR_T - 0.012),
                   *[oval(x, -0.15, 0.5 * sw + 0.002, 0.075) for x in cx])
    lo, hi = (-ff - 0.05, -0.3), (ff + 0.05, 0.0)
    dots = [_plate("gilt", drawn(disc(x, -0.165, 0.022), (x - 0.04, -0.2), (x + 0.04, -0.13), 0.0012, 0.003),
                   fy0 - 0.0035 - 0.0003 - 0.0025, 0.0025, SNIP_FINE) for x in cx]
    statics.append(Part("base", [_plate("proscenium", densify(base, 0.2), fy0, 0.02, SNIP_BIG),
                                 _plate("trim", drawn(pelmet, lo, hi, 0.004, 0.008), fy0 - 0.0035, 0.003, SNIP)] + dots,
                        [], [], {}))

    # the backdrop, unrolled (x = arc length along its plan), its stands behind the flat middle and pencil lines; the
    # sun pinned to its flat middle
    sheet = np.array([[-S, 0.0], [S, 0.0], [S, sky_top], [-S, sky_top]])
    stands = [_stand(x, D + SKY_CARD, 0.5 * sky_top, 0.55) for x in (-0.6 * a, 0.0, 0.6 * a)]
    r_sun = 0.17
    sun_at = (-(a - 1.7 * r_sun - 0.02), D, 0.70 * H)    # upper left, clear of her ears; under a cloud
    sky_lines = {"lines": [((-S + 0.08, 0.62), (S - 0.08, 0.62)), ((-S + 0.12, sky_top - 0.12), (S - 0.12, sky_top - 0.12))],
                 "circles": [((sun_at[0], sun_at[2]), 1.70 * r_sun + 0.06)],
                 "crosses": [((x, sky_top - 0.12), 0.03) for x in np.linspace(-S + 0.3, S - 0.3, 9)]}
    statics.append(Part("sky", [_plate("sky", densify(sheet, 0.25), 0.0, SKY_CARD, SNIP_BIG)] + stands, [], [],
                        {"bend": sky, "pencil_sky": sky_lines}))
    face, rays, eyes, smile, cheeks = sun_fields(r_sun)
    slo, shi = (-2.0 * r_sun, -2.0 * r_sun), (2.0 * r_sun, 2.0 * r_sun)
    ry = -0.0015 - PAPER                                  # the rays' face, about the pin (its frame: y 0 on the sky)
    fy = ry - 0.0005 - PAPER                              # the face's
    dy = fy - 0.001 - 0.0025                              # the eyes', the smile's and the cheeks'
    sun = [_plate("sun", drawn(face, slo, shi, 0.003, 0.006), fy, PAPER, SNIP)]
    sun += [_plate(role, drawn(f, slo, shi, 0.0015, 0.003), dy, 0.0025, SNIP_FINE)
            for role, f in (("ink", eyes), ("ink", smile), ("cheek", cheeks))]
    pin = (0.0, 0.0, fy - 0.0006, SKY_CARD + 0.001)        # the brad (its head the sun's nose) through to the sky's back
    statics.append(Part("sun", sun, [], [pin], {}, sun_at))
    statics.append(Part("sun_rays", [_plate("sun_rays", resample(rays, 0.006), ry, PAPER, SNIP)], [], [], {}, sun_at))

    # the hills: two rows standing on the floor on folded stands, inside the backdrop's sides
    back = [(-1.05 * hw / 1.1, 0.62, 1.0), (0.42 * hw / 1.1, 0.55, 1.1), (1.45, 0.40, 0.8)]
    front = [(-0.40 * hw / 1.1, 0.30, 0.9), (0.95 * hw / 1.1, 0.38, 0.9), (-1.45, 0.28, 0.7)]
    for name, y, humps, base_z, role in (("hills_back", yb, back, 0.30, "hill_back"),
                                         ("hills_front", yf, front, 0.17, "hill_front")):
        hx = float(sky_side(y + 0.05, sky)) - 0.04
        P = hills_outline(-hx, hx, humps, base_z)
        bow = (0.012, -hx, hx)
        plates = [_plate(role, P, y, 0.010, SNIP_BIG, bow=bow)]
        for x in (-0.55 * hx, 0.55 * hx):
            plates.append(_stand(x, y + 0.010 + float(bow_y(bow, np.array([x]))[0]),
                                 0.7 * float(ridge(np.array([x]), humps, base_z)[0]), 0.28))
        statics.append(Part(name, plates, [], [], {}))

    # the grass and its flowers
    plates = []
    spots = [(-1.15, 0.27, 0.06), (-0.92, 0.20, 0.05), (-0.62, 0.23, 0.055), (0.58, 0.22, 0.055), (0.86, 0.28, 0.06),
             (1.12, 0.19, 0.05), (-0.30, 0.15, 0.04), (0.30, 0.16, 0.042)]
    gx = hw + 0.15
    for k, (x, z, r) in enumerate(spots):
        x = x * hw / 1.1
        petals, eye = flower_fields(r, 6, rng("flower", k))
        lo, hi = (x - 1.3 * r, z - 1.3 * r), (x + 1.3 * r, z + 1.3 * r)
        shift = lambda f, x=x, z=z: (lambda X, Z: f(X - x, Z - z))           # noqa: E731
        plates.append(_plate(("flower", "flower2", "flower3")[k % 3], drawn(shift(petals), lo, hi, 0.0015, 0.003),
                             GRASS_Y[0] - 0.006, PAPER * 0.75, SNIP_FINE))
        plates.append(_plate("flower_eye", drawn(shift(eye), lo, hi, 0.0012, 0.003), GRASS_Y[0] - 0.0095, 0.0024,
                             SNIP_FINE))
        plates.append(_plate("grass", np.array([[x - 0.005, 0.035], [x + 0.005, 0.035], [x + 0.0028, z],
                                                [x - 0.0028, z]]), GRASS_Y[0] - 0.0032, 0.0025, SNIP_FINE))
    plates.insert(0, _plate("grass", grass_outline(-gx, gx, rng("grass"), 0.045, (0.05, 0.11)), GRASS_Y[0], PAPER,
                            SNIP_FINE))
    flat = (((1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0, 0.0, PAPER))       # lying on the floor

    def flap(y, role):
        """The foot of a strip folded back flat along the floor: it stands on it."""
        return _plate(role, densify(np.array([[-gx, y], [gx, y], [gx, y + 0.03], [-gx, y + 0.03]]), 0.3), 0.0, PAPER,
                      SNIP, frame=flat)
    plates.append(flap(GRASS_Y[0] + PAPER, "grass"))
    statics.append(Part("grass", plates, [], [], {}))
    back_strip = grass_outline(-gx, gx, rng("grass_back"), 0.06, (0.07, 0.14))
    statics.append(Part("grass_back", [_plate("grass_back", back_strip, GRASS_Y[1], PAPER, SNIP_FINE),
                                       flap(GRASS_Y[1] + PAPER, "grass_back")], [], [], {}))

    # the battens the pieces hang from: dowels across between the backdrop's sides, taped to them at both ends
    ys = rows(D)
    tubes, tapes = [], []
    for k, y in enumerate(ys):
        end = float(sky_side(y, sky)) - 0.002
        tubes.append(((-end, y, za), (end, y, za), BATTEN_R, "cardboard"))
        for sgn in (-1.0, 1.0):
            s = _arc_at(sgn * end, y, sky)
            x_, y_, nx, ny = (float(v[0]) for v in sky_plan(np.array([s]), sky))
            t = tape_outline(0.045, 0.085, rng("batten tape", k, sgn))
            Rm = ((-ny, -nx, 0.0), (nx, -ny, 0.0), (0.0, 0.0, 1.0))           # its x along the wall, its face to the stage
            tapes.append(_plate("tape", t, -0.0015 - TAPE[2], TAPE[2], None, frame=(Rm, (x_, y_, za))))
    flies = Part("flies", tapes, tubes, [], {})
    statics.append(flies)

    # the proscenium and the valance, the curtains: in the carrier's frame (origin on the proscenium's foot)
    pro, pro_top = _proscenium(hw, H)
    frame = (Part("proscenium", pro + _valance(hw, H), [], [], {"flex_height": pro_top}), (0.0, YP, 0.0))
    curtains = [_curtain(s, hw, H, seed) for s in (-1.0, 1.0)]

    # the hanging pieces, a band of tape round the batten at each thread
    hangs = []
    order = _order(seed)
    for (name, kind, row, xs_, zs_, size) in HANGS:
        r = rng("hang", name)
        plates, (pw, ph) = _hang_part(name, kind, size, r)
        x, y, end = xs_ * hw, ys[row], zs_ * H
        length = za - end
        period = 2.0 * math.pi * math.sqrt((length + 0.5 * ph) / G)
        hangs.append(Hang(name, kind, row, (float(x), float(y), float(za)), float(length), (pw, ph),
                          Part(name, plates, [], [], {}), float(period), float(r.uniform(0, 2 * math.pi)),
                          float(r.uniform(*TWIST_PERIOD)), float(r.uniform(0, 2 * math.pi)), float(r.uniform(0.8, 1.2)),
                          order.index(name)))
        tubes.append(((x - 0.018, y, za), (x + 0.018, y, za), BATTEN_R + 0.0015, "tape"))
    return Layout(float(width), float(height), float(depth), int(seed), ys, statics, frame, curtains,
                  tuple(float(v) for v in sun_at), hangs, float(za), sky, (-fx, fx, fy0, fy1))


def _arc_at(x, y, sky):
    """The backdrop's arc length at its point nearest (x, y) (sampled every 2 mm)."""
    s = np.linspace(-sky[4], sky[4], int(2 * sky[4] / 0.002) + 1)
    px, py, _nx, _ny = sky_plan(s, sky)
    return float(s[np.argmin((px - x) ** 2 + (py - y) ** 2)])


def _order(seed):
    """The order the hanging pieces drop in: shuffled by the seed, the moon last."""
    names = [h[0] for h in HANGS if h[0] != "moon"]
    r = _rng(seed, "order")
    return [names[i] for i in r.permutation(len(names))] + ["moon"]


def _stand(x, y_back, height, depth):
    """A folded card stand behind a flat: a right triangle in the y-z plane at x, its upright edge against the flat's
    back (y_back) from the floor up to `height`, its foot `depth` along the floor, its tip cut off square; four straight
    knife cuts (it stands behind its flat)."""
    depth = min(depth, 0.8 * height)
    P = ccw(np.array([[0.0, 0.0], [depth, 0.0], [0.3 * depth, height], [0.0, height]]))
    R = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))         # its x along +y, its thickness along -x
    return _plate("cardboard", P, -0.004, 0.008, None, frame=(R, (float(x), float(y_back), 0.0)))


# ===================================================================================================================
# motion (clip seconds t, arrays)
# ===================================================================================================================
def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def settle(t, t0, amp, hz, secs, phase=0.0):
    """amp e^(-k u) sin(2 pi hz u + phase) from t0 (u = t - t0), 0 before; the envelope is down to 3 % after `secs`."""
    u = np.asarray(t, float) - t0
    k = math.log(1.0 / 0.03) / secs
    v = amp * np.exp(-k * np.maximum(u, 0.0)) * np.sin(2.0 * math.pi * hz * u + phase)
    return np.where(u >= 0.0, v, 0.0)


def drop(t, t0, height, fall, hop=FRAME_HOP, hops=2):
    """Height over its place of something let go at t0 from `height` that lands `fall` s later (falling evenly faster)
    and bounces `hops` times, each with `hop` of the speed it hit with; `height` before t0."""
    t = np.asarray(t, float)
    g = 2.0 * height / fall ** 2
    out = np.where(t < t0, height, 0.0)
    u = t - t0
    m = (u >= 0.0) & (u < fall)
    out[m] = height - 0.5 * g * u[m] ** 2
    v, start = g * fall * hop, t0 + fall
    for _ in range(hops):
        span = 2.0 * v / g
        m = (t >= start) & (t < start + span)
        w = t[m] - start
        out[m] = v * w - 0.5 * g * w ** 2
        start, v = start + span, v * hop
    return out


def landing(t0, height, fall=DROP_FALL, hop=FRAME_HOP, hops=2):
    """When something `drop` lets go at t0 is down for good."""
    g = 2.0 * height / fall ** 2
    v, end = g * fall * hop, t0 + fall
    for _ in range(hops):
        end += 2.0 * v / g
        v *= hop
    return end


def yoyo(t, t0, length, p=None):
    """The thread's length over time for a piece let go at t0 with `wound` m of it out (YOYO): it falls, the thread
    catches at `length` and stretches like a spring of `hz`, throwing the piece back up with `e` of its speed (slack
    while it flies), and catches it again; the last of `catches` catches rings out (damping `zeta`). `wound` before t0.
    Returns (lengths, the time of the first catch)."""
    p = dict(YOYO, **(p or {}))
    t = np.asarray(t, float)
    g, w0, e = p["g"], 2.0 * math.pi * p["hz"], p["e"]
    out = np.full(t.shape, p["wound"])
    fall = math.sqrt(2.0 * max(length - p["wound"], 1e-6) / g)
    u = t - t0
    m = (u >= 0.0) & (u < fall)
    out[m] = p["wound"] + 0.5 * g * u[m] ** 2
    v, start = g * fall, t0 + fall
    first = start
    for k in range(p["catches"]):
        if k == p["catches"] - 1:                              # the last catch rings out
            z = p["zeta"]
            wd = w0 * math.sqrt(1.0 - z * z)
            m = t >= start
            w = t[m] - start
            out[m] = length + v / wd * np.exp(-z * w0 * w) * np.sin(wd * w)
            break
        span = math.pi / w0
        m = (t >= start) & (t < start + span)
        out[m] = length + v / w0 * np.sin(w0 * (t[m] - start))
        start += span
        v *= e
        span = 2.0 * v / g                                     # thrown back up: slack until it comes down again
        m = (t >= start) & (t < start + span)
        w = t[m] - start
        out[m] = length - (v * w - 0.5 * g * w ** 2)
        start += span
    return out, first


Channel = namedtuple("Channel", "object path index values")


def release(opts, hang):
    """The clip second a hanging piece is let go (None: it hangs there all along)."""
    if opts["arrive"] is None:
        return None
    return opts["arrive"] + PIECE_DELAY + hang.order * opts["stagger"]


def motion(opts, lay, t):
    """Every keyed value at the clip seconds t: ([Channel], {object: hidden (bool array)}). Objects are named without
    the set's prefix: frame (the carrier), proscenium, curtain_L / curtain_R, sun_rays, <piece>_hang (the pivot on its
    batten), <piece>, <piece>_thread. Paths are Blender's (`location`, `rotation_euler`, `scale`) or `shape:<key>` for a
    shape key's value."""
    t = np.asarray(t, float)
    ch, hidden = [], {}
    arrive = opts["arrive"]
    if arrive is not None:
        H = lay.height
        z = drop(t, arrive, H, DROP_FALL)
        land = arrive + DROP_FALL
        ch.append(Channel("frame", "location", 2, z))
        ch.append(Channel("frame", "rotation_euler", 0, np.radians(settle(t, land, ROCK[0], ROCK[1], ROCK[2]))))
        ch.append(Channel("proscenium", "shape:flex", 0, settle(t, land, FLEX[0], FLEX[1], FLEX[2], math.pi)))
        for c in lay.curtains:
            lag = 0.03 if c.side > 0 else 0.0
            ch.append(Channel(c.name, "rotation_euler", 0, np.radians(
                settle(t, land + lag, -CURTAIN_SWAY[0], CURTAIN_SWAY[1], CURTAIN_SWAY[2]))))
            ch.append(Channel(c.name, "rotation_euler", 1, np.radians(
                settle(t, land + lag, c.side * CURTAIN_SIDE[0], CURTAIN_SIDE[1], CURTAIN_SIDE[2]))))
            ch.append(Channel(c.name, "shape:flex", 0, settle(t, land + 0.05 + lag, CURTAIN_FLEX[0], CURTAIN_FLEX[1],
                                                               CURTAIN_FLEX[2])))
        for o in ("proscenium", "curtain_L", "curtain_R"):
            hidden[o] = t < arrive
    ch.append(Channel("sun_rays", "rotation_euler", 1, 2.0 * math.pi * opts["sun_rpm"] / 60.0 * t))
    for h in lay.hangs:
        t0 = release(opts, h)
        if t0 is None:
            length, env = np.full(t.shape, h.length), np.ones(t.shape)
        else:
            length, first = yoyo(t, t0, h.length)
            env = smoothstep((t - first) / SWING_RAMP)
            hidden[h.name] = t < t0
            hidden[h.name + "_thread"] = t < t0
        side, fb, twist = amplitudes(opts, h)
        w = 2.0 * math.pi / h.period
        ch.append(Channel(h.name + "_hang", "rotation_euler", 1, side * env * np.sin(w * t + h.phase)))
        ch.append(Channel(h.name + "_hang", "rotation_euler", 0, fb * env * np.sin(w * t + h.phase + 1.9)))
        ch.append(Channel(h.name, "location", 2, -length))
        tw = twist * env * np.sin(2.0 * math.pi * t / h.twist_period + h.twist_phase)
        ch.append(Channel(h.name, "rotation_euler", 2, tw))
        ch.append(Channel(h.name + "_thread", "scale", 2, length))
    return ch, hidden


def amplitudes(opts, h):
    """Radians a hanging piece swings side to side, front to back, and twists: its own share of `swing` (a third of it
    front to back) and `twist`, cut down for a piece that could reach into KEEP_OUT: what is left of its distance in
    front of the keep-out goes half to the swing, the rest to the twist (its sides come forward as it turns)."""
    kx, ky = KEEP_OUT
    ax, ay, _az = h.anchor
    pts = np.concatenate([p.outline for p in h.part.plates])
    rx = float(np.abs(pts[:, 0]).max())                  # how far its edge is from its thread
    reach = h.length + h.size[1]                           # the batten to its lowest point
    side = math.radians(opts["swing"]) * h.swing
    fb, twist = side / 3.0, math.radians(opts["twist"])
    if abs(ax) - rx - reach * math.sin(side) >= kx + 0.03:
        return side, fb, twist
    room = abs(ay) - ky - 0.03
    fb = min(fb, math.asin(min(1.0, max(0.0, 0.5 * room / reach))))
    twist = min(twist, math.asin(min(1.0, max(0.0, (room - reach * math.sin(fb) - 0.01) / rx))))
    return side, fb, twist


def key_times(frame_start, frame_end, frame0, fps, step=0.0, step_from=0.0):
    """(frames to key, the clip second each shows, interpolation): every frame, LINEAR; or with `step` poses a second
    the steps of a grid that starts on clip second `step_from` (frame0 + round(step_from x fps + k x fps / step)), each
    keyed CONSTANT half a frame before its frame (so no shutter spans a step, as docs/design.md: Stepped motion holds)
    with the time of its own frame; the first key holds the step the scene starts in."""
    if step <= 0.0:
        f = np.arange(frame_start, frame_end + 1, dtype=float)
        return f, (f - frame0) / fps, "LINEAR"
    if step > fps:
        raise TheatreError(f"paper_theatre: step = {step:g} poses a second is more than the scene's {fps:g} frames")
    period = fps / step
    origin = frame0 + step_from * fps
    k0 = math.floor((frame_start - origin) / period) - 1
    k1 = math.ceil((frame_end - origin) / period) + 1
    g = np.unique(np.round(origin + np.arange(k0, k1 + 1) * period))
    first = g[g <= frame_start].max()
    g = np.concatenate([[first], g[(g > frame_start) & (g <= frame_end)]])
    keys = g - 0.5
    keys[0] = frame_start - 0.5
    return keys, (g - frame0) / fps, "CONSTANT"


# ===================================================================================================================
# the card
# ===================================================================================================================
def objects(name, lay):
    """Every object the builder makes, by name."""
    out = [f"{name}_{p.name}" for p in lay.statics] + [f"{name}_frame", f"{name}_proscenium"]
    out += [f"{name}_{c.name}" for c in lay.curtains]
    for h in lay.hangs:
        out += [f"{name}_{h.name}_hang", f"{name}_{h.name}", f"{name}_{h.name}_thread"]
    return out


def card(name, opts, lay):
    """The set's card (docs/design.md: Sets): use.look `stage` (her spot, at mid height), `sun`, `moon`, `cloud<i>`,
    `star<i>` (where they hang at rest; `object` the piece), a floor collider, no lights, and `theatre`: the drop's and
    each piece's times, lengths and periods."""
    look = [{"name": "stage", "point": [0.0, 0.0, 0.85]},
            {"name": "sun", "point": [round(v, 4) for v in lay.sun]}]
    pieces = []
    for h in lay.hangs:
        x, y, za = h.anchor
        look.append({"name": h.name, "point": [round(x, 4), round(y, 4), round(za - h.length - 0.5 * h.size[1], 4)],
                     "object": f"{name}_{h.name}"})
        t0 = release(opts, h)
        pieces.append({"name": h.name, "anchor": [round(v, 4) for v in h.anchor], "length": round(h.length, 4),
                       "period": round(h.period, 3), "drops": None if t0 is None else round(t0, 3)})
    arrive = opts["arrive"]
    info = {"opening": [lay.width, lay.height], "depth": lay.depth, "arrive": arrive,
            "lands": None if arrive is None else round(arrive + DROP_FALL, 3),
            "settled": None if arrive is None else round(max(landing(arrive, lay.height) + ROCK[2],
                                                             max(release(opts, h) for h in lay.hangs) + 1.6), 3),
            "step": opts["step"], "pieces": pieces, "roles": list(ROLES)}
    return {"kind": "paper_theatre", "paths": {},
            "use": {"look": look, "surface": []},
            "colliders": [{"type": "floor", "z": 0.0, "tag": name}], "lights": [],
            "objects": objects(name, lay), "materials": [f"{name}_{r}" for r in ROLES], "theatre": info}
