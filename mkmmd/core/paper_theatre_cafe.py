"""The paper theatre's café (`[[set]] kind = "paper_theatre"` key `cafe`; docs/design.md: Sets), numpy only: the option,
the drawing of every piece, where everything stands, how the hole, the tape and the lamps move, and what it adds to the
card. mkmmd/core/paper_theatre.py hands a theatre with `cafe` over to `scene` and `motion` here; a theatre without it
never comes here and is built as it always was.

The theatre dressed as a small café of construction paper and card (frame as in paper_theatre: she sits at the origin
facing -Y, the floor's top is z = 0; her chair and her table are props of their own, `library:paper_cafe_chair` and
`library:paper_cafe_table`, mkmmd/core/paper_cafe.py). The hills, the grass, the sun and the hanging pieces are gone:

  * the café's back wall, a flat of card across the stage at WALL_Y behind her chair: its paper (`wall`) over a dado
    (`wall_low`) with a rail, a tall window with an arched head in a card frame and a sill (`window`) right behind her
    (from close in front of her face it is all that is behind her head: sky and a little of its frame), a shelf of
    paper cups on two brackets on her left, a chalkboard with a chalked doodle (a cup, steam, a heart; no lettering) on
    her right, bunting across the wall's top on a cord between two brass pins, a potted plant of two crossed flats;
  * outside the window the backdrop is pale teal paper sky (the `sky` role's default in a café, CAFE_SKY) with an old
    rip taped shut with an X of masking tape upper right of her head as a close camera in front of her face sees it
    (SKY_RIP), and a low paper skyline stands just outside the window (low enough that from close in front of her it
    is behind her shoulders): plain blocks in two tones, a few with cut windows or a water tank, round a lattice tower
    (legs in an arch, two decks, an antenna, its lattice cut out) and a stepped spire (setbacks, a mast, a needle,
    pencilled piers);
  * over the room a sky border: a big sheet of the same sky paper resting on the wall's top and hung from a batten at
    its front edge (cut in cloud scallops, behind the valance), rising toward the stage's front; pencil guide lines on
    it, two old rips taped shut with masking tape (a dark crack, its torn white fibres, strips of tape across), and two
    paper pendant lamps (a rolled cone shade, a paper bulb) hanging from it on threads beside her, outside the
    keep-out of her strings, swinging a little as the theatre's pieces do (pendulums of their own lengths);
  * the hole: the third old rip is right above her and a little behind (HOLE); on `cafe.hole` it bursts open into a
    ragged rip (jagged torn runs, a few deeper bites) round a white torn rim, its paper left hanging on straight folds
    in flaps of different sizes, long tongues and short ones folded down into the café, and a card of pale clear sky
    (`clear`, a light on it) stands over it; on `cafe.slap` a strip of masking tape is slapped across it from below,
    flying in from under the border and landing ON its time, its torn ends well onto the sky paper past the rim where
    no flap hangs (the object `<name>_slap`, whose origin is the strip's middle on the border, is a hand's point:
    obj('<name>_slap').loc).

Everything moving is held on the set's `step` grid; what lands on a moment is there on the step held on its frame
(paper_theatre_pond.land), what travels sets off early."""
import functools
import math
from collections import namedtuple

import numpy as np

from . import paper_theatre as PT
from . import paper_theatre_pond as PP
from . import shell as S

# --- where everything stands (the set's frame, metres)
WALL_Y = 0.92                   # the café wall's face, behind her chair; the room is in front of it
WALL_T = 0.02                   # its card
WALL_X = 1.62                   # its half width: just inside the backdrop's straight sides
WALL_TOP = 2.72                 # its top edge, where the border over the room rests
DADO = (0.32, 0.03)             # the dado's top under the sill, its rail's height
WINDOW = (0.58, 0.36, 2.0, 0.26)   # half width, sill, the arch's springing and its rise: right behind her head
FRAME = (0.055, 0.008)          # the window frame's width and card
SILL = (0.075, 0.016)           # the sill's depth and card
SKYLINE_Y = 1.25                # the skyline's flat, just outside the window
BORDER_FRONT = (-0.5, 3.36)     # the border's front edge (y, z), behind the valance; its back edge rests on the wall
BORDER_X = 1.6                  # its half width
BORDER_T = 0.008                # its card
FLY_Z = 3.8                     # the batten its front edge hangs from (under the proscenium's header seen from in front)
HOLE = (0.06, 0.12, 0.5, 0.42)  # the hole's middle (x, y) under the border, its half sizes across and along the border
TEAR = (0.005, 0.012, 0.015, 0.03)   # its torn stretches' jag: how far a tooth stands off (least, most), how long
BITES = (3, 0.06, 0.1, 0.022, 0.03)  # how many deeper bites, how far each runs on into the paper (least, most) and its
#                                       mouth's half width (least, most): a round tip
HINGE = (0.07, 0.22)            # a flap's hinge: the straight fold it hangs on (least, most)
GAP = (0.03, 0.17)              # the torn stretch between two flaps (least, most)
TONGUE = (0.16, 0.3, 0.016, 0.03)   # a long tongue's length (least, most) and its ragged tip's width (least, most)
SHORT = (0.045, 0.1)            # a short flap's length (least, most)
FLAP_HANG = {"tongue": (0.85, 1.0), "short": (0.55, 0.85)}   # how far a flap folds down into the room: this share of
#                                                               the fold that would hang it plumb (limp tongues, stiff stubs)
FLAP_CLEAR = 0.07               # no flap hangs within this of the slapped tape's edges
RIM = (0.01, 0.019, 0.005, 0.026)   # the white torn rim's width round the hole (least, most); the sheet's hole is cut
#                                     and the patch over the rip drawn this far round the outline
CLEAR_UP = 0.5                  # the clear sky's card stands this far over the border's face
LAMPS = ((-0.56, 0.75, 0.52, "shade"), (0.56, 0.75, 0.46, "shade2"))   # x, y, thread length, the shade's role
SHADE = (0.03, 0.11, 0.15, 0.003)   # a shade's top and rim radii, its height, its paper
BULB_R = 0.038
LAMP_SWING = 0.6                # the lamps' share of the theatre's `swing`: they swing a little
BUNTING = (1.55, 2.62, 0.07, 0.165, 0.03)   # its half span, the cord's height at the pins, its sag, the flags' spacing,
#                                             how far in front of the wall
FLAG = (0.12, 0.15)             # a flag's width and length
PLANT_AT = (0.92, 0.64)         # the potted plant's foot (x, y): on her left, outside the keep-out of her strings
SLAP = (1.7, 0.11, -13.0, 0.0012)   # the tape strip's length, width, its turn on the border (degrees), its card
SLAP_AT = (0.02, -0.03, 0.01)   # its middle from the hole's middle (u, v), and how far under the border's face it lies
SLAP_TRAVEL = 0.2               # s it flies in from the hand
SLAP_FROM = 0.45                # m under the border it flies in from (along the border's normal)
BURST = 0.12                    # s the hole takes to burst open (it is open ON `hole`)
TAPE_W = 0.045                  # an old rip's strips of tape
CLOSE_EYE = (0.0, -1.1, 1.22)   # the close camera in front of her face the window and its old rip are laid out for
SKY_RIP = (0.23, 1.48, 0.075, 0.26, 14.0)   # the old taped rip on the window's sky: where it is seen from CLOSE_EYE
#                                             (x, z at her depth: upper right of her head, over her left ear), how far
#                                             it stands in front of the backdrop, its length and turn on it (degrees)
SKY_RIP_W = (0.009, 0.007, 0.055)  # its crack's width, the white fibres' (more or less) each side, its tape's width
CAFE = {"hole": None, "slap": None}

Flap = namedtuple("Flap", "plate hinge angle")      # a torn flap: its Plate on the border's face (u, v), its hinge
#                                                     ((u, v), (u, v)), the degrees it folds down
Cafe = namedtuple("Cafe", "frame hole flaps rim patch slap slap_rest slap_R slap_start")   # rim: the white torn rim's
#                                                     Plate on the border's face; slap_R: the strip's turn


def _err(msg):
    return PT.TheatreError(f"paper_theatre: {msg}")


def check(v):
    """`cafe`: true or a table {hole, slap} (clip seconds; absent: the rip stays taped shut, no tape is slapped);
    None or false: no café."""
    if v is None or v is False:
        return None
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise _err(f"cafe = {v!r}: true or a table of {sorted(CAFE)}")
    bad = sorted(set(v) - set(CAFE))
    if bad:
        raise _err(f"cafe: unknown keys {bad} (known: {sorted(CAFE)})")
    o = dict(CAFE, **v)
    for k in CAFE:
        o[k] = None if o[k] is None else PT._number(o[k], f"cafe.{k}")
    if o["hole"] is not None and o["slap"] is not None and o["slap"] <= o["hole"]:
        raise _err(f"cafe: the tape is slapped across the hole after it bursts open (slap {o['slap']:g} <= hole "
                   f"{o['hole']:g})")
    return o


# ===================================================================================================================
# the border over the room: its frame
# ===================================================================================================================
def border_angle():
    """Radians the border rises toward the stage's front."""
    y1, z1 = BORDER_FRONT
    return math.atan2(z1 - WALL_TOP, WALL_Y - y1)


def border_frame():
    """(R, t) taking a plate on the border (its own x = u across the stage, its z = v from the back edge toward the
    front, its y up into the card) to the set's frame, x' = R x + t: its face (a plate's front, toward -y) looks down
    and forward into the café."""
    a = border_angle()
    s, c = math.sin(a), math.cos(a)
    return ((1.0, 0.0, 0.0), (0.0, s, -c), (0.0, c, s)), (0.0, WALL_Y, WALL_TOP)


def border_length():
    y1, z1 = BORDER_FRONT
    return math.hypot(WALL_Y - y1, z1 - WALL_TOP)


def border_uv(x, y):
    """(u, v) on the border over the set's (x, y)."""
    return float(x), float((WALL_Y - y) / math.cos(border_angle()))


def border_point(u, v, below=0.0):
    """The set's point at (u, v) on the border's face, `below` m under it (into the room)."""
    R, t = border_frame()
    return np.asarray(R) @ np.array([u, -below, v]) + np.asarray(t)


def border_normal():
    """The border face's normal, into the room (down and toward the stage's front)."""
    return np.asarray(border_frame()[0]) @ np.array([0.0, -1.0, 0.0])


def on_border(u, v, turn=0.0):
    """A plate frame lying on the border at (u, v), its own x turned `turn` degrees toward +v (its y still the
    border's)."""
    R, t = border_frame()
    b = math.radians(turn)
    Rb = np.array([[math.cos(b), 0.0, -math.sin(b)], [0.0, 1.0, 0.0], [math.sin(b), 0.0, math.cos(b)]])
    M = np.asarray(R) @ Rb
    return tuple(map(tuple, M)), tuple(np.asarray(R) @ np.array([u, 0.0, v]) + np.asarray(t))


# ===================================================================================================================
# drawings (each in its own frame, metres)
# ===================================================================================================================
def tear(a, b, rng, amp, step=0.012):
    """A torn line from a to b: a point every `step` or so, wandering off the straight line by up to `amp` (a slow
    wander and a jag at every point), its ends exactly a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.hypot(*d))
    n = max(int(L / step), 3)
    s = np.linspace(0.0, 1.0, n + 1)
    nrm = np.array([-d[1], d[0]]) / L
    walk = np.cumsum(rng.normal(0.0, 1.0, n + 1))
    walk -= walk[0] + s * (walk[-1] - walk[0])
    walk *= 0.6 * amp / max(float(np.abs(walk).max()), 1e-9)
    off = walk + rng.uniform(-0.4, 0.4, n + 1) * amp
    off[0] = off[-1] = 0.0
    return a + s[:, None] * d + off[:, None] * nrm


def window_outline(hw, bottom, spring, rise, step=0.01):
    """The window's opening (or its frame grown round it): straight sides from `bottom` to `spring`, a segmental arch
    `rise` over them, counter-clockwise from its bottom left corner."""
    R = (hw * hw + rise * rise) / (2.0 * rise)
    cz = spring + rise - R
    a = math.asin(hw / R)
    arc = np.linspace(-a, a, max(int(2 * a * R / step), 8))
    top = np.stack([R * np.sin(arc[::-1]), cz + R * np.cos(arc[::-1])], 1)       # right to left
    return PT.ccw(PT.densify(np.vstack([[[-hw, bottom], [hw, bottom], [hw, spring]], top[1:-1], [[-hw, spring]]]),
                             0.05))


def window_frame_outline():
    """The window frame's outer edge: the opening grown by the frame's width all round but at the sill."""
    hw, sill, spring, rise = WINDOW
    w = FRAME[0]
    R = (hw * hw + rise * rise) / (2.0 * rise)
    cz = spring + rise - R
    R2, hw2 = R + w, hw + w
    spring2 = cz + math.sqrt(R2 * R2 - hw2 * hw2)
    return window_outline(hw2, sill - 0.03, spring2, cz + R2 - spring2)      # its foot hidden under the sill


@functools.lru_cache(maxsize=4)
def hole_drawing(seed):
    """The hole's torn outline on the border (u, v) and its flaps [(outline, (hinge a, hinge b), degrees)], drawn about
    the hole's middle so that every ray from there meets the outline once (it cannot cross itself): an uneven ring of
    long jagged stretches with a few deeper bites where a tear ran on into the paper. Along some stretches a flap of the
    paper is left hanging on a straight fold (its hinge): long tongues and short flaps of different sizes, each inside
    its own wedge of the hole (no two overlap), none where the tape slapped across the hole meets its edge."""
    u0, v0 = border_uv(*HOLE[:2])
    c = np.array([u0, v0])
    rng = PT._rng(seed, "hole")
    n = 1440
    phi = 2.0 * math.pi * np.arange(n) / n
    k = np.arange(2, 5)[:, None]
    rho = 1.0 + (rng.uniform(0.02, 0.05, (3, 1)) * np.cos(k * phi + rng.uniform(0.0, 2.0 * math.pi, (3, 1)))).sum(0)
    D = np.stack([HOLE[2] * np.cos(phi), HOLE[3] * np.sin(phi)], 1)       # the way out from the middle, per unit of rho
    base = c + rho[:, None] * D
    b = math.radians(SLAP[2])                              # the tape's centre line: start where it meets the edge
    off = (base - c - np.asarray(SLAP_AT[:2])) @ np.array([-math.sin(b), math.cos(b)])
    first = int(np.flatnonzero(np.sign(off) != np.sign(np.roll(off, -1)))[0])
    rho, D, base, off = (np.roll(a, -first, 0) for a in (rho, D, base, off))
    size = np.linalg.norm(D, axis=1)
    seg = np.linalg.norm(np.roll(base, -1, 0) - base, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])      # arc length from the first sample
    free = np.abs(off) >= 0.5 * SLAP[1] + FLAP_CLEAR
    ends = np.flatnonzero(np.diff(np.concatenate([[0], free.astype(int), [0]])))
    hinges = []
    for a, z in zip(ends[::2], ends[1::2]):                # the stretches the tape leaves free: flaps and gaps
        pos = s[a] + 0.5 * rng.uniform(*GAP)
        while True:
            w = rng.uniform(*HINGE)
            if pos + w > s[z - 1] - 0.01:
                break
            hinges.append((int(np.searchsorted(s, pos)), int(np.searchsorted(s, pos + w))))
            pos += w + rng.uniform(*GAP)
    stretches = []                                        # the torn stretches between them (ends on their hinges)
    for j, (_ia, ib) in enumerate(hinges):
        ia2 = hinges[(j + 1) % len(hinges)][0]
        stretches.append(np.arange(ib, ia2 + 1 if ia2 > ib else ia2 + 1 + n) % n)
    rho = rho.copy()
    arcs = [np.concatenate([[0.0], np.cumsum(seg[idx[:-1]])]) for idx in stretches]
    bitten = {}
    for j in rng.permutation(len(stretches)):              # deeper bites, where the tape will not cover them
        idx, arc = stretches[j], arcs[j]
        w, depth, u = rng.uniform(*BITES[3:]), rng.uniform(*BITES[1:3]), rng.uniform(0.0, 1.0)
        if len(bitten) >= BITES[0] or arc[-1] < 2.0 * w + 0.05:
            continue
        at = (w + 0.02 + u * (arc[-1] - 2.0 * w - 0.04)) / arc[-1]          # its mouth 2 cm clear of the stretch's ends
        if (np.abs(off[idx[np.abs(arc - at * arc[-1]) < w + 0.01]]) >= 0.5 * SLAP[1] + 0.03).all():
            bitten[int(j)] = (at, w, depth)
    for j, (idx, arc) in enumerate(zip(stretches, arcs)):  # a jag along each: teeth of straight torn runs
        knots, vals, x, sign = [0.0], [0.0], rng.uniform(*TEAR[2:]), float(rng.choice([-1.0, 1.0]))
        while x < arc[-1] - 0.008:
            knots.append(x)
            vals.append(sign * rng.uniform(*TEAR[:2]))
            sign = -sign if rng.random() < 0.8 else sign
            x += rng.uniform(*TEAR[2:])
        jag = np.interp(arc, knots + [arc[-1]], vals + [0.0]) * np.clip(np.minimum(arc, arc[-1] - arc) / 0.012, 0, 1)
        if j in bitten:                                    # a deeper bite: a tear ran on into the paper, its tip round
            at, w, depth = bitten[j]
            x = (arc - at * arc[-1]) / w
            jag = jag * np.clip(np.abs(x) - 1.0, 0.0, 1.0) + depth * np.clip(1.0 - x * x, 0.0, None)
        rho[idx[1:-1]] += jag[1:-1] / size[idx[1:-1]]
    P = c + rho[:, None] * D
    for ia, ib in hinges:                                  # a hinge is a straight fold: the rays meet its chord
        A, B = P[ia], P[ib]
        i = np.arange(ia + 1, ib)
        e, r = B - A, A - c
        P[i] = c + ((e[0] * r[1] - r[0] * e[1]) / (e[0] * D[i, 1] - D[i, 0] * e[1]))[:, None] * D[i]
    kinds = ["tongue"] * max(2, round(0.45 * len(hinges))) + ["short"] * len(hinges)
    kinds = [kinds[i] for i in rng.permutation(len(hinges))]
    a = border_angle()
    flaps = []
    for (ia, ib), kind in zip(hinges, kinds):
        e = (P[ib] - P[ia]) / np.linalg.norm(P[ib] - P[ia])
        nin = np.array([-e[1], e[0]]) if (c - P[ia]) @ np.array([-e[1], e[0]]) > 0.0 else np.array([e[1], -e[0]])
        plumb = math.degrees(math.atan2(math.cos(a), -math.sin(a) * nin[1]))   # the fold that hangs it straight down
        flaps.append((_flap(P[ia], P[ib], c, kind, rng), (P[ia].copy(), P[ib].copy()),
                      plumb * float(rng.uniform(*FLAP_HANG[kind]))))
    return PT.ccw(PT.resample(P, 0.004)), flaps


def _flap(A, B, c, kind, rng):
    """A flap left hanging on the fold A..B of the hole's edge (c the hole's middle), as drawn on the border before it
    folds down: from the fold in toward the middle inside the wedge c, A, B; a long tongue narrowing to a ragged tip, or
    a short flap torn off across. Its sides are graphs over the way in and its torn end one across, so it cannot cross
    itself; nothing is sharp enough to fold a rolled edge."""
    W = float(np.linalg.norm(B - A))
    e = (B - A) / W
    nin = np.array([-e[1], e[0]])
    if (c - A) @ nin < 0.0:
        nin = -nin
    xc, h = float((c - A) @ e), float((c - A) @ nin)
    m = 0.006                                              # kept this far inside the wedge

    def wedge(y):
        return xc * y / h + m, W + (xc - W) * y / h - m
    if kind == "tongue":
        tip = rng.uniform(*TONGUE[2:])
        L = min(rng.uniform(*TONGUE[:2]), 0.95 * h * (1.0 - (tip + 2.0 * m) / W))
        lo, hi = wedge(L)
        xt = rng.uniform(lo + 0.5 * tip, hi - 0.5 * tip) if hi - lo > tip else 0.5 * (lo + hi)
        x0, x1, Ll, Lr = xt - 0.5 * tip, xt + 0.5 * tip, L, L
    else:
        L = rng.uniform(*SHORT)
        lo, hi = wedge(L)
        x0, x1 = max(lo, rng.uniform(0.06, 0.22) * W), min(hi, W - rng.uniform(0.06, 0.22) * W)
        Ll, Lr = L * (1.0 + rng.uniform(-0.25, 0.25)), L * (1.0 + rng.uniform(-0.25, 0.25))

    def side(xa, xb, Ls, other):
        """A torn side from (xa, 0) on the fold to (xb, Ls), jagged by at most a fifth of the flap's width there."""
        ys = [0.0]
        while ys[-1] < Ls - 0.02:
            ys.append(ys[-1] + rng.uniform(0.012, 0.022))
        ys = np.array(ys[:-1] + [Ls]) if len(ys) > 1 else np.array([0.0, Ls])
        xs = xa + (xb - xa) * ys / Ls
        a = np.minimum(0.005, 0.2 * np.abs(other(ys) - xs)) * np.clip(ys / 0.01, 0.0, 1.0)
        jag = rng.uniform(-1.0, 1.0, len(ys)) * a
        jag[0] = jag[-1] = 0.0
        return np.stack([xs + jag, ys], 1)
    left = side(0.0, x0, Ll, lambda y: W + (x1 - W) * y / Lr)
    right = side(W, x1, Lr, lambda y: x0 * y / Ll)
    span = x1 - x0
    if kind == "tongue":                                   # a ragged tip, its teeth blunt
        xs = x1 - span * (np.arange(1, 3) + rng.uniform(-0.15, 0.15, 2)) / 3.0
        ys = L + rng.uniform(-0.2, 0.6, 2) * span / 3.0
    else:                                                  # torn off across: a zigzag, one end longer
        xs = [x1 - rng.uniform(0.01, 0.018)]
        while xs[-1] - x0 > 0.016:
            xs.append(xs[-1] - rng.uniform(0.01, 0.018))
        xs = np.array(xs[:-1]) if len(xs) > 1 else np.array([0.5 * (x0 + x1)])
        tooth = np.where(np.arange(len(xs)) % 2 == 0, 1.0, -1.0) * rng.uniform(0.003, 0.008, len(xs))
        ys = Lr + (Ll - Lr) * (x1 - xs) / span + tooth * np.clip(np.minimum(x1 - xs, xs - x0) / 0.006, 0.0, 1.0)
    F = np.vstack([[[0.0, 0.0]], right, np.stack([xs, ys], 1), left[::-1][:-1]])
    return PT.ccw(A + F[:, :1] * e + F[:, 1:] * nin)


def _dilate(P, c, d, rays=1440):
    """The region inside the closed polygon P (every ray from c meets it once) grown by discs of radius d (one, or one
    per point of P) round its points: the polygon every ray from c meets once, so it cannot fold where P is jagged,
    counter-clockwise."""
    Q = np.asarray(P, float) - c
    d = np.broadcast_to(np.asarray(d, float), (len(Q),))
    th = 2.0 * math.pi * np.arange(rays) / rays
    U = np.stack([np.cos(th), np.sin(th)], 1)
    a = U @ Q.T                                            # each point along each ray, and its distance off it
    q2 = (Q * Q).sum(1)[None, :] - a * a
    far = np.where(q2 < d * d, a + np.sqrt(np.maximum(d * d - q2, 0.0)), 0.0)
    return c + far.max(1)[:, None] * U


@functools.lru_cache(maxsize=4)
def rim_drawing(seed):
    """The outer edge of the white torn rim round the hole: its outline grown by a fibrous width, a slow wander between
    RIM's least and most and tufts of fibres standing out of it."""
    outline, _f = hole_drawing(seed)
    rng = PT._rng(seed, "rim")
    n = len(outline)
    x = rng.normal(0.0, 1.0, n)
    x = sum(np.roll(x, j) for j in range(-3, 4))
    w = RIM[0] + 0.45 * (RIM[1] - RIM[0]) * (x - x.min()) / max(float(np.ptp(x)), 1e-9)
    k = np.arange(-3, 4)
    for i in rng.integers(0, n, 16):
        w[(i + k) % n] += (RIM[1] - RIM[0]) * rng.uniform(0.35, 0.6) * np.cos(0.5 * math.pi * k / 3.5)
    return PT.resample(_dilate(outline, np.array(border_uv(*HOLE[:2])), np.clip(w, RIM[0], RIM[1])), 0.004)


def strip_outline(length, width, rng, teeth=5):
    """A strip of masking tape about the origin, its length along x: its long edges straight as the roll cut them, its
    two ends torn across aslant, a zigzag of `teeth` teeth bold enough to read from a few metres off."""
    hl, hw = 0.5 * length, 0.5 * width
    k = 2 * teeth + 1

    def end(sign):
        z = np.linspace(-hw, hw, k)
        z[1:-1] += rng.uniform(-0.2, 0.2, k - 2) * (z[1] - z[0])
        x = hl + rng.uniform(-0.35, 0.35) * z + np.where(np.arange(k) % 2 == 0, 1.0, -1.0) * rng.uniform(0.004, 0.01, k)
        return np.stack([sign * x, z], 1)
    right, left = end(1.0), end(-1.0)[::-1]
    t = np.linspace(0.0, 1.0, 40)[1:-1, None]
    return PT.ccw(np.vstack([right, right[-1] + t * (left[0] - right[-1]), left, left[-1] + t * (right[0] - left[-1])]))


def crack(u, v, length, turn, seed, key):
    """An old rip on the border: its torn line (u, v) about (u, v), `length` long, turned `turn` degrees."""
    rng = PT._rng(seed, "rip", key)
    b = math.radians(turn)
    d = 0.5 * length * np.array([math.cos(b), math.sin(b)])
    return tear(np.array([u, v]) - d, np.array([u, v]) + d, rng, 0.03, 0.012)


def tower_drawing(h=0.84):
    """A lattice tower `h` tall about x = 0 standing on z = 0: its legs splayed into an arch at the foot, two decks,
    an antenna; and its lattice (the triangles cut out between the braces of each panel). (outline, holes)."""
    zb = 0.66 * h

    def w(z):
        return 0.010 + 0.085 * (1.0 - min(z, zb) / zb) ** 1.6
    decks = ((0.30 * h, 0.34 * h, 0.018), (0.55 * h, 0.575 * h, 0.011))
    marks = sorted([0.0, zb] + [z for z0, z1, _d in decks for z in (z0, z1)])
    right = []
    for a, b in zip(marks[:-1], marks[1:]):
        deck = [d for z0, z1, d in decks if abs(a - z0) < 1e-9 and abs(b - z1) < 1e-9]
        if deck:                                           # a deck: square shoulders out and back
            right += [(w(a), a), (w(a) + deck[0], a), (w(b) + deck[0], b), (w(b), b)]
        else:
            right += [(w(z), z) for z in np.linspace(a, b, max(int((b - a) / 0.025), 2) + 1)]
    right += [(0.006, zb + 0.004), (0.0022, h)]
    clean = [right[0]]
    for p in right[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
            clean.append(p)
    w0 = w(0.0)
    arch = [(0.55 * w0 * math.cos(t), 0.09 * h * math.sin(t)) for t in np.linspace(math.pi, 0.0, 14)]
    P = np.array(clean + [(-x, z) for x, z in clean[::-1]] + arch)     # up the right, down the left, over the arch
    outline = PT.ccw(PT.densify(P, 0.02))
    holes = []
    leg = 0.012
    for z0, z1, n in ((0.10 * h, 0.29 * h, 3), (0.35 * h, 0.54 * h, 3)):
        zs = np.linspace(z0, z1, n + 1)
        for za, zc in zip(zs[:-1], zs[1:]):
            wa, wc = w(za) - leg, w(zc) - leg
            if wc < 0.008:
                continue
            c = np.array([0.0, 0.5 * (za + zc)])
            quad = [np.array(p) for p in ((-wa, za + 0.004), (wa, za + 0.004), (wc, zc - 0.004), (-wc, zc - 0.004))]
            for q0, q1 in zip(quad, quad[1:] + quad[:1]):
                tri = np.array([q0, q1, c])
                m = tri.mean(0)
                tri = m + (tri - m) * 0.56                   # inset: the braces and the legs stay
                edges = [float(np.linalg.norm(tri[i] - tri[i - 1])) for i in range(3)]
                if min(edges) > 0.014:                       # its corners rounded (a sharp one folds the rolled rim)
                    holes.append(PT.ccw(PT.densify(S.fillet(PT.ccw(tri), 0.0035, 3, closed=True), 0.004)))
    return outline, holes


def spire_drawing():
    """A stepped spire about x = 0 standing on z = 0: a wide base, the shaft, setbacks, a mast with a rounded cap and
    a needle. (outline, its piers: pencil lines (x, z0, z1))."""
    steps = [(0.12, 0.0), (0.12, 0.17), (0.095, 0.17), (0.095, 0.55), (0.075, 0.55), (0.075, 0.6), (0.058, 0.6),
             (0.058, 0.635), (0.04, 0.635), (0.04, 0.665), (0.022, 0.665), (0.019, 0.725), (0.012, 0.742),
             (0.0045, 0.75), (0.0025, 0.8)]
    P = np.array(steps + [(-x, z) for x, z in steps[::-1]])
    return PT.ccw(PT.densify(P, 0.02)), [(x, 0.19, 0.53) for x in (-0.06, -0.03, 0.0, 0.03, 0.06)]


# the skyline's blocks: x0, x1, height, role, what is on the roof (None, "tank", "mast", "step"), cut windows (cols, rows)
BLOCKS = ((-0.98, -0.78, 0.46, "skyline2", "tank", None), (-0.8, -0.62, 0.56, "skyline", "step", (2, 3)),
          (-0.44, -0.3, 0.40, "skyline2", None, None), (-0.32, -0.12, 0.52, "skyline", "mast", None),
          (-0.14, 0.1, 0.44, "skyline2", None, (3, 2)), (0.08, 0.28, 0.56, "skyline", "tank", None),
          (0.26, 0.44, 0.42, "skyline2", None, None), (0.64, 0.82, 0.50, "skyline2", "step", (2, 2)),
          (0.8, 0.98, 0.40, "skyline", None, None))
LANDMARKS = (("tower", -0.52), ("spire", 0.54))   # out to the window's sides: from close in front of her they stand
#                                                   beside her shoulders, not behind her head


def block_drawing(x0, x1, h, roof):
    """A plain block from x0 to x1, `h` tall, standing on z = 0, and what is on its roof: a water tank on legs, a mast,
    or a setback."""
    pts = [(x0, 0.0), (x1, 0.0)]
    xm, w = 0.5 * (x0 + x1), x1 - x0
    if roof == "step":
        pts += [(x1, h - 0.06), (x1 - 0.22 * w, h - 0.06), (x1 - 0.22 * w, h), (x0 + 0.22 * w, h),
                (x0 + 0.22 * w, h - 0.06), (x0, h - 0.06)]
    elif roof == "tank":
        t = xm + 0.18 * w
        pts += [(x1, h), (t + 0.032, h), (t + 0.028, h + 0.035), (t + 0.026, h + 0.035), (t + 0.026, h + 0.085),
                (t, h + 0.11), (t - 0.026, h + 0.085), (t - 0.026, h + 0.035), (t - 0.028, h + 0.035),
                (t - 0.032, h), (x0, h)]
    elif roof == "mast":
        pts += [(x1, h), (xm + 0.004, h), (xm + 0.0025, h + 0.16), (xm - 0.0025, h + 0.16), (xm - 0.004, h), (x0, h)]
    else:
        pts += [(x1, h), (x0, h)]
    return PT.ccw(PT.densify(np.array(pts), 0.03))


def block_windows(x0, x1, h, grid):
    """Little windows cut out of a block: `grid` (columns, rows) of them in its upper part."""
    if grid is None:
        return []
    cols, rows = grid
    w = x1 - x0
    out = []
    ww, wh = 0.22 * w / cols * 1.6, 0.035
    for i in range(cols):
        for j in range(rows):
            cx = x0 + w * (i + 0.5) / cols
            cz = h - 0.1 - j * 0.07
            out.append(PT.ccw(PT.densify(np.array([[cx - 0.5 * ww, cz - 0.5 * wh], [cx + 0.5 * ww, cz - 0.5 * wh],
                                                   [cx + 0.5 * ww, cz + 0.5 * wh], [cx - 0.5 * ww, cz + 0.5 * wh]]),
                                         0.006)))
    return out


def mug_drawing(w, h, saucer=False):
    """A cup seen from the side standing on z = 0 about x = 0, with a saucer under it or not: (its outline, its
    handle: a C of paper on its right whose ends tuck in behind the cup)."""
    body = PT.union(PT.box(-0.5 * w, 0.0, 0.5 * w, h), PT.oval(0.0, 0.0, 0.5 * w, 0.18 * h))
    f = PT.inter(body, PT.above(0.0))
    if saucer:
        f = PT.union(f, PT.box(-0.85 * w, -0.012, 0.85 * w, 0.004))
    outline = PT.drawn(f, (-0.9 * w, -0.03), (0.9 * w, 1.2 * h), 0.0015, 0.004)
    a = np.radians(np.linspace(-120.0, 120.0, 25))
    cx, cz, r1, r0 = 0.47 * w, 0.55 * h, 0.27 * h, 0.15 * h
    outer = np.stack([cx + r1 * np.cos(a), cz + r1 * np.sin(a)], 1)
    inner = np.stack([cx + r0 * np.cos(a[::-1]), cz + r0 * np.sin(a[::-1])], 1)
    return outline, PT.ccw(np.vstack([outer, inner]))


def leaf_outline(length, width, turn, at):
    """A leaf `length` long and `width` wide from its stalk end at `at`, pointing `turn` degrees from up (+ to the
    right): an oval drawn out to a point."""
    s = np.linspace(0.0, 1.0, 24)
    half = width * 0.5 * np.sin(math.pi * s) ** 0.8 * (1.0 - 0.35 * s)
    side = np.stack([half, s * length], 1)
    P = np.vstack([side, (side * (-1.0, 1.0))[::-1][1:-1]])
    b = math.radians(-turn)
    Rm = np.array([[math.cos(b), -math.sin(b)], [math.sin(b), math.cos(b)]])
    return PT.ccw(P @ Rm.T + np.asarray(at, float))


# ===================================================================================================================
# the layout
# ===================================================================================================================
def scene(opts):
    """(Layout, Cafe) of a theatre with `cafe`: paper_theatre's layout with the café's statics in place of the hills,
    the grass, the sun and the battens, the lamps for its hanging pieces (no sun: `sun` None), and the Cafe (what moves:
    the hole's patch, flaps and rim, the tape strip)."""
    return _scene(opts["width"], opts["height"], opts["depth"], opts["seed"])


@functools.lru_cache(maxsize=4)
def _scene(width, height, depth, seed):
    base = PT.layout(width, height, depth, seed)
    statics = []
    for p in base.statics:
        if p.name == "sky":                                # the backdrop: no compass circle round a sun now
            pen = dict(p.special["pencil_sky"], circles=[])
            p = p._replace(special=dict(p.special, pencil_sky=pen))
        if p.name in ("floor", "base", "sky"):
            statics.append(p)
    statics += [_wall(seed), _skyline(seed), _border(seed), _clear(seed), _bunting(seed), _plant(seed), _flies(height),
                _sky_rip(seed, depth)]
    hangs = _lamps(seed)
    lay = base._replace(statics=statics, hangs=hangs, sun=None)
    frame = border_frame()
    outline, flaps = hole_drawing(seed)
    flap_list = [Flap(PT._plate("flap", P, 0.0005, 0.6 * BORDER_T, None), hinge, deg)   # torn, not cut
                 for P, hinge, deg in flaps]
    rim = PT._plate("torn", rim_drawing(seed), -0.002, 0.0012, None, holes=(outline,))  # a hair under the face
    patch = _patch(outline, seed)
    tape = strip_outline(SLAP[0], SLAP[1], PT._rng(seed, "slap"))
    slap = PT.Part("slap", [PT._plate("tape", tape, -SLAP[3], SLAP[3], None)], [], [], {})
    u0, v0 = border_uv(*HOLE[:2])
    um, vm = u0 + SLAP_AT[0], v0 + SLAP_AT[1]
    rest = border_point(um, vm, SLAP_AT[2])                # under the border, under the torn rim too
    start = rest + SLAP_FROM * border_normal() + np.array([0.0, -0.12, 0.0])
    R, _t = on_border(um, vm, SLAP[2])
    return lay, Cafe(frame, outline, flap_list, rim, patch, slap, tuple(float(v) for v in rest), R,
                     tuple(float(v) for v in start))


def _strip(role, path_uv, width, below, frame=None):
    """A strip lying on the border's face along `path_uv` (u, v), `below` m under it: ('strip', role, the path in the
    set's frame, the normal, width, below) for the parts module's bands."""
    pts = np.array([border_point(u, v) for u, v in path_uv])
    return (role, pts, tuple(border_normal()), float(width), float(below))


def _rip_parts(path, seed, key, tapes=3):
    """An old rip taped shut on the border: (strips, plates): its crack, the white fibres torn along one side of it,
    and strips of masking tape across it."""
    strips = [_strip("crack", path, 0.006, 0.0004)]
    d = np.gradient(path, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    strips.append(_strip("torn", path + 0.006 * n, 0.005, 0.0004))
    rng = PT._rng(seed, "rip tape", key)
    plates = []
    for k in range(tapes):
        s = (k + 0.5 + rng.uniform(-0.15, 0.15)) / tapes
        i = min(int(s * (len(path) - 1)), len(path) - 2)
        tang = path[i + 1] - path[i]
        along = math.degrees(math.atan2(tang[1], tang[0]))
        R, t = on_border(path[i][0], path[i][1], along + 90.0 + float(rng.uniform(-14.0, 14.0)))
        tab = PT.tape_outline(TAPE_W, float(rng.uniform(0.15, 0.2)), rng)
        plates.append(PT._plate("tape", tab @ np.array([[0.0, 1.0], [1.0, 0.0]]), -0.0016 - PT.TAPE[2], PT.TAPE[2],
                                None, frame=(R, t)))
    return strips, plates


def _outside(path, region):
    """The runs of the polyline `path` that stay outside the closed polygon `region` (two points or more each)."""
    runs, cur = [], []
    for p, out in zip(path, ~PT.inside(path, region)):
        if out:
            cur.append(p)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    return [r for r in runs + [np.array(cur)] if len(r) > 1]


def _border(seed):
    """The sky border over the room: its sheet with the hole cut in it (a little wider than the hole's outline, under
    its torn rim), its cloud-scalloped front edge, pencil lines (stopping short of the hole), two old rips taped shut,
    tape tabs where the lamps' threads meet it."""
    L = border_length()
    rng = PT._rng(seed, "border")
    us = np.linspace(BORDER_X, -BORDER_X, 240)
    bumps = 0.06 * np.abs(np.sin(math.pi * (us + BORDER_X) / 0.27)) ** 0.6
    front = np.stack([us, L - 0.06 + bumps], 1)
    P = np.vstack([[[-BORDER_X, -0.04], [BORDER_X, -0.04]], front])
    outline, _flaps = hole_drawing(seed)
    c = np.array(border_uv(*HOLE[:2]))
    R, t = border_frame()
    sheet = PT._plate("sky", PT.densify(P, 0.2), 0.0, BORDER_T, PT.SNIP_BIG,
                      holes=(PT.resample(_dilate(outline, c, RIM[2]), 0.004),), frame=(R, t))
    sheet = sheet._replace(spacing=0.06)
    plates, strips = [sheet], []
    keep = _dilate(outline, c, RIM[3] + 0.02)
    for path in ([(-1.5, 0.3), (1.5, 0.3)], [(-1.5, L - 0.25), (1.5, L - 0.25)]):
        p = np.array(path, float)
        line = p[0] + np.linspace(0.0, 1.0, 240)[:, None] * (p[1] - p[0])
        line[:, 1] += 0.004 * np.sin(np.linspace(0.0, float(rng.uniform(2, 4)) * math.pi, 240))
        strips += [_strip("pencil", run, 0.005, 0.0004) for run in _outside(line, keep)]
    for key, (u, v, length, turn) in (("rip2", (-0.98, 0.62, 0.5, 18.0)), ("rip3", (1.0, 1.02, 0.46, -24.0))):
        st, pl = _rip_parts(crack(u, v, length, turn, seed, key), seed, key)
        strips += st
        plates += pl
    for x, y, _length, _role in LAMPS:                      # a tab of tape over each lamp's thread
        u, v = border_uv(x, y)
        R2, t2 = on_border(u, v, float(rng.uniform(-10.0, 10.0)))
        tab = PT.tape_outline(PT.TAPE[0], PT.TAPE[1], rng)
        plates.append(PT._plate("tape", tab, -PT.THREAD_R - 0.0002 - PT.TAPE[2], PT.TAPE[2], None, frame=(R2, t2)))
    return PT.Part("border", plates, [], [], {"strips": strips})


def _patch(outline, seed):
    """The paper over the hole until it bursts: the hole's outline grown over its torn rim, a hair under the border's
    face, with the rip it was and its tape."""
    R, t = border_frame()
    u0, v0 = border_uv(*HOLE[:2])
    P = PT.resample(_dilate(outline, np.array([u0, v0]), RIM[3]), 0.01)
    plate = PT._plate("sky", P, -0.001 - PT.PAPER, PT.PAPER, PT.SNIP, frame=(R, t))
    strips, tapes = _rip_parts(crack(u0, v0, 0.62, 8.0, seed, "rip1"), seed, "rip1", 3)
    tapes = [p._replace(y=p.y - PT.PAPER - 0.001) for p in tapes]     # on the patch's face, which is a hair down
    return PT.Part("patch", [plate] + tapes, [], [], {"strips": [(r, p, n, w, b + PT.PAPER + 0.001)
                                                                 for r, p, n, w, b in strips]})


def _clear(seed):
    """The card of pale clear sky over the hole, facing down through it, and the light on it."""
    outline, _f = hole_drawing(seed)
    lo, hi = outline.min(0) - 0.3, outline.max(0) + 0.3
    R, t = border_frame()
    card = PT.densify(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]), 0.15)
    plates = [PT._plate("clear", card, CLEAR_UP, PT.SKY_CARD, None, frame=(R, t))]
    c = 0.5 * (lo + hi)
    for (dx, dz, a, b, deg) in ((0.0, 0.0, 0.42, 0.33, 10.0), (-0.22, 0.2, 0.12, 0.08, -20.0),
                                (0.3, -0.18, 0.1, 0.07, 30.0)):
        f = PT.oval(c[0] + dx, c[1] + dz, a, b, deg)
        P = PT.drawn(f, (c[0] - 0.8, c[1] - 0.8), (c[0] + 0.8, c[1] + 0.8), 0.006, 0.02)
        plates.append(PT._plate("clear_glow", P, CLEAR_UP - 0.002 - PT.PAPER, PT.PAPER, PT.SNIP, frame=(R, t)))
    return PT.Part("clear", plates, [], [], {})


def _sky_rip(seed, depth):
    """An old rip in the window's sky taped shut with an X of masking tape, upper right of her head as the close camera
    sees it (SKY_RIP, SKY_RIP_W): its crack, the white fibres torn along both sides of it, two strips of tape crossed
    over it, flat pieces facing -y. It stands SKY_RIP's offset in front of the backdrop: a cut-out lays its pieces far to
    near by the median depth of each, and that high on the backdrop the close camera sees it farther off than most of
    the sky round her head; in front of the backdrop by this much it is nearer, so the sky is laid under it."""
    x, z, off, length, turn = SKY_RIP
    crack_w, fibres, tape_w = SKY_RIP_W
    eye = np.asarray(CLOSE_EYE, float)
    y = depth - off
    t = (y - eye[1]) / -eye[1]
    mid = np.array([eye[0] + t * (x - eye[0]), eye[2] + t * (z - eye[2])])     # seen where (x, z) at her depth is
    rng = PT._rng(seed, "sky rip")
    b = math.radians(turn)
    d, nrm = np.array([math.cos(b), math.sin(b)]), np.array([-math.sin(b), math.cos(b)])
    line = tear(mid - 0.5 * length * d, mid + 0.5 * length * d, rng, 0.012, 0.014)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    q = np.linspace(0.0, s[-1], int(s[-1] / 0.004) + 1)
    fine = np.stack([np.interp(q, s, line[:, 0]), np.interp(q, s, line[:, 1])], 1)      # straight torn runs, finer
    half = 0.5 * crack_w * (0.35 + 0.65 * np.sin(math.pi * q / s[-1]) ** 0.5)          # its two lips a gap apart
    crack = np.vstack([fine + half[:, None] * nrm, (fine - half[:, None] * nrm)[::-1]])
    f = rng.uniform(0.5, 1.5, (2, len(fine)))
    f = fibres * (f + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 3.0
    lens = np.vstack([[fine[0] - 0.01 * d], fine + (half + f[0])[:, None] * nrm, [fine[-1] + 0.01 * d],
                      (fine - (half + f[1])[:, None] * nrm)[::-1]])
    # its layers 5 mm apart: their medians never tie, so a cut-out lays the fibres, the crack, the X in that order
    plates = [PT._plate("torn", lens, y - 0.001, 0.001, None), PT._plate("crack", crack, y - 0.006, 0.001, None)]
    for k, sign in enumerate((1.0, -1.0)):                 # the X: each strip across the rip, the two crossed
        a = b + math.radians(sign * float(rng.uniform(36.0, 46.0)))
        tab = PT.tape_outline(tape_w, float(rng.uniform(0.19, 0.23)), rng)      # its length along z: turned to a
        Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        at = mid + float(rng.uniform(-0.012, 0.012)) * d
        plates.append(PT._plate("tape", tab @ Rm.T + at, y - 0.011 - 0.0025 * k - PT.TAPE[2], PT.TAPE[2], None))
    return PT.Part("sky_rip", plates, [], [], {})


def _wall(seed):
    """The café's back wall: its card with the window cut in it, the dado and its rail, the window's frame and sill,
    the shelf of cups on its brackets, the chalkboard with its doodle hung on a nail, and its stands behind."""
    rng = PT._rng(seed, "wall")
    hw, sill, spring, rise = WINDOW
    win = window_outline(hw, sill, spring, rise)
    y = WALL_Y
    plates = [PT._plate("wall", PT.densify(np.array([[-WALL_X, 0.0], [WALL_X, 0.0], [WALL_X, WALL_TOP],
                                                     [-WALL_X, WALL_TOP]]), 0.25), y, WALL_T, PT.SNIP_BIG,
                        holes=(win,))._replace(spacing=0.08)]
    dado, rail = DADO
    plates.append(PT._plate("wall_low", PT.densify(np.array([[-WALL_X, 0.0], [WALL_X, 0.0], [WALL_X, dado],
                                                             [-WALL_X, dado]]), 0.2), y - PT.PAPER - 0.001,
                            PT.PAPER, PT.SNIP))
    plates.append(PT._plate("window", PT.densify(np.array([[-WALL_X, dado - 0.5 * rail], [WALL_X, dado - 0.5 * rail],
                                                           [WALL_X, dado + 0.5 * rail], [-WALL_X, dado + 0.5 * rail]]),
                                                 0.2), y - 2 * PT.PAPER - 0.002, PT.PAPER, PT.SNIP))
    fw, ft = FRAME
    plates.append(PT._plate("window", window_frame_outline(), y - ft - 0.001, ft, PT.SNIP, holes=(win,)))
    lying = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))           # a plate lying flat, its face up
    sd, st = SILL
    sx = hw + fw + 0.03
    plates.append(PT._plate("window", PT.densify(np.array([[-sx, y - sd], [sx, y - sd], [sx, y + 0.002],
                                                           [-sx, y + 0.002]]), 0.1), 0.0, st, PT.SNIP,
                            frame=(lying, (0.0, 0.0, sill))))
    # the shelf on her left, its brackets and its cups
    x0, x1, zs, depth = 0.72, 1.08, 1.36, 0.09
    plates.append(PT._plate("cardboard", PT.densify(np.array([[x0, y - depth], [x1, y - depth], [x1, y], [x0, y]]),
                                                    0.1), 0.0, 0.012, PT.SNIP, frame=(lying, (0.0, 0.0, zs))))
    side = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))             # standing across the wall, its x along +y
    for xb in (x0 + 0.06, x1 - 0.06):
        tri = PT.ccw(PT.densify(np.array([[-0.075, zs - 0.012], [0.0, zs - 0.012], [0.0, zs - 0.11]]), 0.01))
        plates.append(PT._plate("cardboard", tri, -0.004, 0.008, None, frame=(side, (xb, y, 0.0))))     # knife-cut
    roles = ("cup", "cup2", "cup3", "cup")
    for k, (dx, w, h, saucer) in enumerate(((0.05, 0.06, 0.07, False), (0.15, 0.07, 0.055, True),
                                            (0.25, 0.055, 0.075, False), (0.32, 0.05, 0.06, False))):
        xc = x0 + dx
        P, handle = mug_drawing(w, h, saucer)
        yc = y - 0.05 - 0.004 * (k % 2)
        plates.append(PT._plate(roles[k], handle + (xc, zs + 0.012), yc + 0.0015, PT.PAPER * 0.6, PT.SNIP_FINE))
        plates.append(PT._plate(roles[k], P + (xc, zs + 0.012), yc, PT.PAPER * 0.6, PT.SNIP_FINE))
    # the chalkboard on her right: its frame, the board, the doodle, the string to its nail
    bx0, bx1, bz0, bz1 = -1.08, -0.72, 1.0, 1.52
    bw = 0.035
    outer = PT.densify(np.array([[bx0, bz0], [bx1, bz0], [bx1, bz1], [bx0, bz1]]), 0.05)
    inner = PT.densify(np.array([[bx0 + bw, bz0 + bw], [bx1 - bw, bz0 + bw], [bx1 - bw, bz1 - bw],
                                 [bx0 + bw, bz1 - bw]]), 0.04)
    yb = y - 0.012
    plates.append(PT._plate("board", PT.densify(np.array([[bx0 + 0.02, bz0 + 0.02], [bx1 - 0.02, bz0 + 0.02],
                                                          [bx1 - 0.02, bz1 - 0.02], [bx0 + 0.02, bz1 - 0.02]]), 0.05),
                            yb, 0.006, PT.SNIP))
    plates.append(PT._plate("cardboard", outer, yb - 0.01, 0.01, PT.SNIP, holes=(inner,)))
    strips = _doodle(0.5 * (bx0 + bx1), 0.5 * (bz0 + bz1), yb, rng)
    nail = (0.5 * (bx0 + bx1), bz1 + 0.12)
    tubes = [((bx0 + 0.05, yb - 0.006, bz1), (nail[0], y - 0.008, nail[1]), PT.THREAD_R, "thread"),
             ((bx1 - 0.05, yb - 0.006, bz1), (nail[0], y - 0.008, nail[1]), PT.THREAD_R, "thread")]
    brads = [(nail[0], nail[1], y - 0.012, y + WALL_T)]
    for x in (-1.4, -0.95, 0.95, 1.4):                      # its stands behind, none where the window shows them
        plates.append(PT._stand(x, y + WALL_T, 0.55 * WALL_TOP, 0.5))
    return PT.Part("wall", plates, tubes, brads, {"strips": strips})


def _doodle(cx, cz, y, rng):
    """The chalk doodle on the board about (cx, cz): a cup on its saucer, three curls of steam, a heart over them."""
    face = y - 0.0005

    def strip(pts):
        p = np.asarray(pts, float) + (cx, cz)
        p = p + rng.normal(0.0, 0.0012, p.shape)              # a chalk hand
        return ("board_chalk", np.stack([p[:, 0], np.full(len(p), face), p[:, 1]], 1), (0.0, -1.0, 0.0), 0.009,
                0.0003)
    t = np.linspace(math.pi, 2.0 * math.pi, 20)
    cup = np.vstack([[[-0.075, 0.0]], np.stack([0.075 * np.cos(t), -0.02 + 0.09 * np.sin(t) * 0.9], 1)[1:-1],
                     [[0.075, 0.0]], [[-0.075, 0.0]]])
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 14)
    handle = np.stack([0.075 + 0.03 * np.cos(a), -0.04 + 0.028 * np.sin(a)], 1)
    saucer = np.array([[-0.11, -0.11], [0.11, -0.11]])
    out = [strip(cup), strip(handle), strip(saucer)]
    for k, x in enumerate((-0.035, 0.0, 0.035)):
        s = np.linspace(0.0, 1.0, 16)
        out.append(strip(np.stack([x + 0.012 * np.sin(2.0 * math.pi * (s + 0.25 * k)), 0.03 + 0.08 * s], 1)))
    from .mandala import _heart
    h = _heart(40) * 0.075 * (1.0, -1.0) + (0.0, 0.17)
    out.append(strip(np.vstack([h, h[:1]])))
    return out


def _skyline(seed):
    """The low skyline just outside the window: plain blocks in two tones (some with cut windows, a water tank, a
    mast or a setback), a lattice tower and a stepped spire among them, and its stands."""
    y = SKYLINE_Y
    plates, strips = [], []
    for k, (x0, x1, h, role, roof, grid) in enumerate(BLOCKS):
        windows = tuple(block_windows(x0, x1, h, grid))
        plates.append(PT._plate(role, block_drawing(x0, x1, h, roof), y + 0.003 * (k % 2), PT.PAPER, PT.SNIP_FINE,
                                holes=windows)._replace(spacing=0.012 if windows else None))
    P, holes = tower_drawing(0.9)                          # the landmarks stand out of the low blocks: beside her
    x = dict(LANDMARKS)["tower"]                           # shoulders seen from close in front, over the sill from afar
    plates.append(PT._plate("tower", P + (x, 0.0), y - 0.004, PT.PAPER, None,      # a lattice cut with a knife
                            holes=tuple(hh + (x, 0.0) for hh in holes))._replace(spacing=0.008))
    P, piers = spire_drawing()
    x = dict(LANDMARKS)["spire"]
    k = (1.05, 1.08)                                       # drawn 0.8 m tall: standing 0.86
    plates.append(PT._plate("spire", P * k + (x, 0.0), y - 0.004, PT.PAPER, PT.SNIP_FINE))
    for px, z0, z1 in piers:
        p = np.stack([np.full(12, x + k[0] * px), np.full(12, y - 0.0045), np.linspace(k[1] * z0, k[1] * z1, 12)], 1)
        strips.append(("pencil", p, (0.0, -1.0, 0.0), 0.004, 0.0003))
    for sx in (-0.6, 0.6):
        plates.append(PT._stand(sx, y + 0.006, 0.3, 0.25))
    return PT.Part("skyline", plates, [], [], {"strips": strips})


def _bunting(seed):
    """Bunting across the wall's top: a cord sagging between two brass pins and paper flags folded over it."""
    hs, z, sag, gap, off = BUNTING
    y = WALL_Y - off
    x = np.linspace(-hs, hs, 121)
    cz = z - sag * (1.0 - (x / hs) ** 2)
    cord = np.stack([x, np.full(len(x), y), cz], 1)
    rng = PT._rng(seed, "bunting")
    plates = []
    roles = ("bunting", "bunting2", "bunting3", "bunting4")
    n = int(2 * hs / gap)
    for k in range(n):
        xc = -hs + (k + 0.5) * 2 * hs / n
        zc = z - sag * (1.0 - (xc / hs) ** 2)
        slope = math.atan(2.0 * sag * xc / hs ** 2)
        w, L = FLAG
        tri = PT.ccw(PT.densify(np.array([[-0.5 * w, 0.0], [0.5 * w, 0.0], [0.0, -L]]), 0.03))
        g = slope + math.radians(float(rng.uniform(-4.0, 4.0)))
        R = ((math.cos(g), 0.0, -math.sin(g)), (0.0, 1.0, 0.0), (math.sin(g), 0.0, math.cos(g)))
        plates.append(PT._plate(roles[k % 4], tri, 0.0012 + 0.0006 * (k % 2), PT.PAPER * 0.6, PT.SNIP_FINE,
                                frame=(R, (xc, y, zc - 0.004))))
    brads = [(sx * hs, z, WALL_Y - 0.004, WALL_Y + WALL_T) for sx in (-1.0, 1.0)]
    return PT.Part("bunting", plates, [], brads, {"cords": [(cord, 0.0022, "cardboard")]})


def _plant(seed):
    """A potted plant of two crossed flats standing in front of the wall on her left: a pot with its rim, leaves on
    stems; the second flat turned a quarter round through the first."""
    rng = PT._rng(seed, "plant")
    x, y = PLANT_AT
    plates = []
    for flat in range(2):
        R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) if flat == 0 else \
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        t = (x, y, 0.0)
        pot = PT.ccw(np.array([[-0.075, 0.0], [0.075, 0.0], [0.1, 0.17], [-0.1, 0.17]]))
        rim = PT.ccw(np.array([[-0.11, 0.16], [0.11, 0.16], [0.11, 0.21], [-0.11, 0.21]]))
        dy = 0.003 * flat
        plates.append(PT._plate("pot", PT.densify(pot, 0.03), -0.002 + dy, PT.PAPER, PT.SNIP, frame=(R, t)))
        plates.append(PT._plate("pot", PT.densify(rim, 0.03), -0.003 - PT.PAPER + dy, PT.PAPER, PT.SNIP,
                                frame=(R, t)))
        for k in range(6):
            turn = -62.0 + 124.0 * k / 5.0 + float(rng.uniform(-8.0, 8.0)) + 9.0 * flat
            L = float(rng.uniform(0.2, 0.28)) * (1.0 - 0.25 * abs(turn) / 62.0)
            b = math.radians(turn)
            stem_top = np.array([0.0, 0.2]) + float(rng.uniform(0.12, 0.3)) * np.array([math.sin(b), math.cos(b)])
            d = (stem_top - (0.0, 0.18)) / np.linalg.norm(stem_top - (0.0, 0.18))
            nrm = np.array([-d[1], d[0]])
            stem = PT.ccw(PT.densify(np.array([(0.0, 0.18) - 0.006 * nrm, (0.0, 0.18) + 0.006 * nrm,
                                               stem_top + 0.004 * nrm, stem_top - 0.004 * nrm]), 0.01))
            role = "leaf" if (k + flat) % 2 == 0 else "leaf2"
            plates.append(PT._plate("leaf", stem, 0.003 + dy, PT.PAPER * 0.6, PT.SNIP_FINE, frame=(R, t)))
            plates.append(PT._plate(role, leaf_outline(L, 0.42 * L, turn, stem_top), 0.001 + dy - 0.0015 * (k % 2),
                                    PT.PAPER * 0.6, PT.SNIP_FINE, frame=(R, t)))
    return PT.Part("plant", plates, [], [], {})


def _flies(height):
    """The batten the border's front edge hangs from on two threads."""
    L = border_length()
    y = BORDER_FRONT[0]
    tubes = [((-BORDER_X, y, FLY_Z), (BORDER_X, y, FLY_Z), PT.BATTEN_R, "cardboard")]
    for x in (-1.35, 1.35):
        p = border_point(x, L - 0.08, 0.0)
        tubes.append(((x, y, FLY_Z), tuple(float(v) for v in p), PT.THREAD_R, "thread"))
    return PT.Part("flies", [], tubes, [], {})


def _lamps(seed):
    """The pendant lamps hung from the border beside her: Hangs (paper_theatre's: their pendulums, their drop with
    `arrive`), each a lathed paper shade with a bulb under it (part special `lamp`)."""
    out = []
    r0, r1, h, _t = SHADE
    for k, (x, y, length, role) in enumerate(LAMPS):
        rng = PT._rng(seed, "lamp", k)
        u, v = border_uv(x, y)
        anchor = tuple(float(c) for c in border_point(u, v, 0.001))
        size = (2.0 * r1, h + BULB_R)
        part = PT.Part(f"lamp{k + 1}", [], [], [], {"lamp": role})
        period = 2.0 * math.pi * math.sqrt((length + 0.5 * size[1]) / PT.G)
        out.append(PT.Hang(f"lamp{k + 1}", "lamp", -1, anchor, float(length), size, part, float(period),
                           float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(*PT.TWIST_PERIOD)),
                           float(rng.uniform(0, 2 * math.pi)), LAMP_SWING, k))
    return out


def lamp_profiles():
    """The lathe profiles (r, z) of a lamp in its piece's frame (the thread's end at the origin, hanging down -z): its
    shade (a rolled paper cone with a turned-out rim, closed over the top round the thread) and its bulb."""
    r0, r1, h, t = SHADE
    z = np.linspace(0.0, -h, 12)
    r = r0 + (r1 - r0) * (1.0 - (1.0 + z / h) ** 1.3)        # a cone swelling a little toward its rim
    outer = list(zip(r, z))
    inner = list(zip(r[::-1] - t, z[::-1] + 0.5 * t))
    shade = [(0.0, 0.006)] + [(r0 * 0.6, 0.006)] + outer + [(r1 + 0.004, -h - 0.004), (r1 - 0.002, -h - 0.006)] + \
        inner[1:] + [(0.0, -0.4 * t)]
    a = np.linspace(0.5 * math.pi, -0.5 * math.pi, 14)
    zb = -h + 0.006
    bulb = [(0.0, zb + 0.022)] + [(max(BULB_R * math.cos(q), 0.0), zb - 0.012 + BULB_R * math.sin(q))
                                   for q in a[1:-1]] + [(0.0, zb - 0.012 - BULB_R)]
    return np.array(shade), np.array(bulb)


# ===================================================================================================================
# motion (clip seconds t, arrays)
# ===================================================================================================================
def hole_open(opts, t):
    """How open the hole is (0 the rip taped shut; 1 open, its flaps folded down; past 1 they spring): bursting over
    BURST, open ON `cafe.hole` (on the set's step held on its frame), springing and settling."""
    t = np.asarray(t, float)
    when = opts["cafe"]["hole"]
    if when is None:
        return np.zeros(t.shape)
    t1 = PP.land(opts, when)
    t0 = t1 - BURST
    u = np.clip((t - t0) / BURST, 0.0, 1.0)
    o = np.where(t < t0, 0.0, 0.25 + 0.75 * u ** 1.5)
    return np.where(t >= t1, 1.0 + PT.settle(t, t1, 0.12, 2.6, 0.7, 0.5 * math.pi), o)


def slap_place(opts, cafe, t):
    """Where the tape strip's middle is at t (n, 3) and whether it is hidden: in the hand from SLAP_TRAVEL before
    `cafe.slap`, flying up at the border faster and faster, flat on it ON its time; hidden before (always without
    `slap`)."""
    t = np.asarray(t, float)
    rest, start = np.asarray(cafe.slap_rest), np.asarray(cafe.slap_start)
    when = opts["cafe"]["slap"]
    if when is None:
        return np.broadcast_to(rest, (len(t), 3)).copy(), np.ones(t.shape, bool)
    t1 = PP.land(opts, when)
    t0 = t1 - SLAP_TRAVEL
    u = np.clip((t - t0) / SLAP_TRAVEL, 0.0, 1.0) ** 1.6
    return start + u[:, None] * (rest - start), t < t0


def motion(opts, lay, cafe, t):
    """Every keyed value at the clip seconds t of a theatre with `cafe`: ([Channel], {object: hidden}, {object: (states,
    index)}): the frame, the curtains and the lamps as paper_theatre.motion keys them (there is no sun), the hole's
    patch and flaps (a flipbook of how open it is) and the tape strip."""
    t = np.asarray(t, float)
    ch, hidden = PT.motion(opts, lay, t)
    ch = [c for c in ch if c.object != "sun_rays"]
    o = hole_open(opts, t)
    hidden["patch"] = o > 0.0
    hidden["flaps"] = o <= 0.0
    flips = {"flaps": PP._flip(o, 3)}
    p, gone = slap_place(opts, cafe, t)
    for i in range(3):
        ch.append(PT.Channel("slap", "location", i, p[:, i]))
    hidden["slap"] = gone
    return ch, hidden, flips


# ===================================================================================================================
# the card
# ===================================================================================================================
def objects(name, lay, cafe):
    return PT.objects(name, lay) + [f"{name}_patch", f"{name}_flaps", f"{name}_slap"]


def card_extra(name, opts, lay, cafe):
    """What the café adds to the card: use.look `window` (the window's middle), `hole` (the hole's middle on the
    border's face) and `slap` (the tape strip's middle, riding `<name>_slap`: a hand's point), and the `theatre`
    entry `cafe` (its times and where its pieces stand)."""
    hw, sill, spring, rise = WINDOW
    u0, v0 = border_uv(*HOLE[:2])
    hole = border_point(u0, v0)
    look = [{"name": "window", "point": [0.0, WALL_Y, round(0.5 * (sill + spring + rise), 4)]},
            {"name": "hole", "point": [round(float(v), 4) for v in hole]},
            {"name": "slap", "point": [round(v, 4) for v in cafe.slap_rest], "object": f"{name}_slap"}]
    c = opts["cafe"]
    info = {"cafe": {"hole": c["hole"], "slap": c["slap"], "wall": WALL_Y,
                     "window": {"half_width": hw, "sill": sill, "top": spring + rise},
                     "border": {"back": [WALL_Y, WALL_TOP], "front": list(BORDER_FRONT),
                                "over_her": round(WALL_TOP + WALL_Y * math.tan(border_angle()), 3)},
                     "hole_at": [round(float(v), 4) for v in hole], "hole_size": [2 * HOLE[2], 2 * HOLE[3]],
                     "lamps": [{"name": h.name, "anchor": [round(v, 4) for v in h.anchor], "length": h.length}
                               for h in lay.hangs]}}
    return look, info
