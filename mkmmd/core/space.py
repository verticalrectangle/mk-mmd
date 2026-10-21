"""The space kit, bpy-free (docs/design.md: Props, the space kit): the options, meshes and motion of the library props
`line`, `hole`, `round_board`, `chess`, `stars`, `floor`, `starfield` and `disc`, which
mkmmd/blender/library/props/space.py builds.

  line         a thin white line along +X made of short segments, with tick marks: a ruler, a tightrope, the edge-on disc of
               a black hole. Its root property `extent` (0..1, keyable) is how much of it there is, measured from its
               `anchor` end: the line retracts into that end as `extent` falls (each segment shrinks toward its
               anchor-side end, a driver on its scale), and `ripples` [[t, x, amp], ...] run a damped pulse out both ways
               from x along it (each segment tilted to meet its neighbours, keyed).
  hole         a black sphere: the shadow of a black hole (the light it bends is drawn by the `lens` effect). Root
               property `breathe` scales it (a kick's inhale).
  round_board  a round chessboard: `rings` rings of `sectors` cells round a hole, light and dark alternating (Byzantine
               chess has 4 x 16). Each ring is an object that turns about the centre by `omega_i * phase`, `phase` the
               root's keyable property and omega_i = omega * (r_i / r_0) ** kepler: with kepler -1.5 the inner rings
               orbit faster and the board winds into a spiral as `phase` grows. Every cell has a hidden `hot` overlay
               (an attacked square) that [[key]] shows.
  chess        a Staunton-like piece turned on a lathe (pawn, rook, bishop, queen, king) or a knight (a lathed base under
               a horse's head cut as a thick rounded plate).
  stars        flat stars at points of the picture's plane, each flashing at its time, joined in order by thin lines that
               draw themselves as the next star flashes: a constellation, a melody as a piano roll.
  floor        a floor to the horizon at z = 0: a black `mirror` (EEVEE's ray tracing, which the floor switches on, reflects
               what the camera sees in it), a `chess` board of `cell` m squares, or a glowing `grid` of lines every `cell`
               m. The pattern softens to its average as it recedes (no moire) and fades into `dark` between the two
               `fade` distances from the camera.
  starfield    stars streaming past along -X at `speed` m/s (and `surge` keys on top: a jump to light speed), drawn as
               streaks `streak` m long: two tiles of `length` m of them leapfrog round the root, so they never run out;
               none within `clear` m of the X axis, where the character walks.
  disc         an accretion disc: a flat ring from `inner` to `outer` that glows white-hot inside and `color` further
               out, its streaks orbiting, the inner ones faster (Kepler: angular speed ~ r ** kepler), additive over
               what is behind it."""
import math

import numpy as np

from . import shell as CS

LINE = {"length": 40.0, "radius": 0.012, "seg": 0.25, "anchor": "end", "ticks": 0.0, "bar": 4, "tick": [0.05, 0.12],
        "tick_width": 0.006, "color": "text", "ripples": [], "ripple": {}}
RIPPLE = {"speed": 7.0, "decay": 0.3, "width": 0.5, "life": 1.0}
HOLE = {"radius": 1.5, "color": "#000000"}
BOARD = {"inner": 2.2, "width": 0.8, "rings": 6, "sectors": 16, "thick": 0.05, "gap": 0.03, "omega": 1.0,
         "kepler": -1.5, "light": "text", "dark": "#000000", "hot": "love"}
PIECES = ("pawn", "rook", "knight", "bishop", "queen", "king")
CHESS = {"piece": "pawn", "height": 1.0, "color": "text", "rough": 0.12, "metal": 0.0}
STARS = {"points": [], "kind": 4, "inner": 0.3, "size": 0.12, "depth": 0.004, "lines": True, "line_width": 0.01,
         "flash": 0.35, "gain": 1.8, "draw": 0.12, "color": "text"}
FLOOR_KINDS = ("mirror", "chess", "grid")
FLOOR = {"kind": "mirror", "size": 400.0, "cell": 2.0, "line": 0.04, "color": "text", "dark": "#000000", "glow": 3.0,
         "rough": None, "metal": 0.0, "fade": [40.0, 220.0]}
FLOOR_ROUGH = {"mirror": 0.03, "chess": 0.12, "grid": 0.08}   # its roughness unless `rough` says
STARFIELD = {"count": 900, "length": 160.0, "width": 90.0, "height": [-1.0, 45.0], "clear": 5.0, "size": 0.035,
             "streak": 0.0, "speed": 12.0, "surge": [], "color": "text", "glow": 12.0, "seed": 1}
DISC = {"inner": 2.0, "outer": 7.0, "color": "love", "hot": "#FFFFFF", "glow": 5.0, "spin": 0.3, "kepler": -1.5,
        "streaks": 3.0, "rings": 22.0, "seed": 1}
MIN_SCALE = 0.0005           # never exactly 0: a singular matrix upsets everything that inverts one
FUNCS = frozenset({"clamp", "min", "max", "abs"})      # what the drivers call (Blender simple expressions)


class SpaceError(ValueError):
    """A bad option of a space-kit prop: says what to change."""


def options(defaults, slots, what):
    """The prop's options: `defaults` with the `slots` given laid over them. A slot that is no option and no palette
    colour override (`#hex` under a palette slot's name) is a SpaceError listing the options."""
    from .palette import PALETTES
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in defaults.items()}
    colours = set(PALETTES["rose-pine"])
    for k, v in (slots or {}).items():
        if k in out:
            out[k] = v
        elif not (k in colours and isinstance(v, str)):
            raise SpaceError(f"{what}: unknown option {k!r} (options: {', '.join(sorted(defaults))})")
    return out


def _positive(o, keys, what):
    for k in keys:
        if not isinstance(o[k], (int, float)) or o[k] <= 0:
            raise SpaceError(f"{what}: {k} must be a positive number, not {o[k]!r}")


# ================================================================================================================ line
def line_options(slots):
    o = options(LINE, slots, "line")
    _positive(o, ("length", "radius", "seg"), "line")
    if o["anchor"] not in ("end", "start"):
        raise SpaceError(f"line: anchor is \"end\" (the line retracts into its +X end) or \"start\", not {o['anchor']!r}")
    if not isinstance(o["ticks"], (int, float)) or o["ticks"] < 0:
        raise SpaceError("line: ticks is the spacing of the tick marks in metres (0: none)")
    o["ripple"] = options(RIPPLE, o["ripple"], "line ripple")
    _positive(o["ripple"], ("speed", "decay", "width", "life"), "line ripple")
    for i, r in enumerate(o["ripples"]):
        if not (isinstance(r, (list, tuple)) and len(r) == 3 and all(isinstance(v, (int, float)) for v in r)):
            raise SpaceError(f"line: ripples[{i}] is [t, x, amp] (clip seconds, metres along the line, metres up)")
    return o


def segments(length, seg, anchor):
    """[(x0, x1, a)]: the line's segments along +X, x0 < x1 in the line's frame, and `a`, how far the segment's
    anchor-side end lies from the anchor. Counted from the anchor; the last may be shorter."""
    n = max(1, int(math.ceil(length / seg - 1e-9)))
    out = []
    for k in range(n):
        a, b = k * seg, min((k + 1) * seg, length)
        out.append((length - b, length - a, a) if anchor == "end" else (a, b, a))
    return out


def segment_scale(extent, a, s, length):
    """How much of a segment shows (its scale along the line, MIN_SCALE..1) when `extent` of the line does: the numpy
    twin of `segment_expr`."""
    return np.clip((np.asarray(extent, float) * length - a) / s, MIN_SCALE, 1.0)


def segment_expr(a, s, length):
    """The driver expression of a segment's scale along the line, of the root's `e` (its `extent`)."""
    return f"clamp((e * {length:.6g} - {a:.6g}) / {s:.6g}, {MIN_SCALE}, 1.0)"


def tick_expr(d, length):
    """The driver expression of a tick's scale: shown while the line reaches it (it pops in over 2 cm)."""
    return f"clamp((e * {length:.6g} - {d:.6g}) / 0.02, {MIN_SCALE}, 1.0)"


def ticks(length, spacing, bar):
    """[(x, long)]: tick marks every `spacing` m from x = 0 to the length, every `bar`-th one long (the first is)."""
    if spacing <= 0:
        return []
    n = int(math.floor(length / spacing + 1e-9))
    return [(round(k * spacing, 6), bar > 0 and k % int(bar) == 0) for k in range(n + 1)]


def ripple(ts, xs, events, speed, decay, width, life):
    """Heights (F, X) of the line at clip seconds `ts` and positions `xs` under `events` [(t, x, amp)]: each sends a
    pulse both ways from x at `speed` m/s, a Ricker wavelet of `width` m ((1 - 2u^2) exp(-u^2), u its distance from the
    front over `width`: a dip with a lift on either side for a negative amp), fading as exp(-age / decay) and gone after
    `life` s."""
    ts, xs = np.asarray(ts, float), np.asarray(xs, float)
    z = np.zeros((len(ts), len(xs)))
    for t0, x0, amp in events:
        age = ts - float(t0)
        live = (age >= 0.0) & (age <= life)
        if not live.any():
            continue
        u = (np.abs(xs[None, :] - float(x0)) - speed * age[live, None]) / width
        z[live] += float(amp) * np.exp(-age[live, None] / decay) * (1.0 - 2.0 * u * u) * np.exp(-u * u)
    return z


def segment_pose(z0, z1, s):
    """(lift, tilt): a segment of length s whose anchor-side end must be at height z0 and other end at z1 is lifted z0
    and turned by tilt radians about its anchor-side end (positive: its far end rises)."""
    return z0, math.atan2(z1 - z0, s)


def segment_mesh(s, radius, toward):
    """A segment's tube from its origin (its anchor-side end) `s` m along -X (toward = -1) or +X (+1), capped."""
    path = np.array([[0.0, 0.0, 0.0], [toward * s, 0.0, 0.0]])
    return CS.tube(path, radius, sides=10)


def tick_mesh(height, width):
    """A tick: a thin vertical rod across the line, centred on it."""
    return CS.tube(np.array([[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]), width / 2.0, sides=8,
                   up=(1.0, 0.0, 0.0))


# ================================================================================================================ hole
def hole_options(slots):
    o = options(HOLE, slots, "hole")
    _positive(o, ("radius",), "hole")
    return o


def sphere(radius, seg=48, rings=24):
    """A UV sphere about the origin (a lathe of a half circle)."""
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, rings + 1)
    return CS.lathe(np.stack([radius * np.cos(a), radius * np.sin(a)], 1), seg=seg)


# ================================================================================================================ board
def board_options(slots):
    o = options(BOARD, slots, "round_board")
    _positive(o, ("inner", "width", "thick"), "round_board")
    for k in ("rings", "sectors"):
        if not isinstance(o[k], int) or o[k] < 1:
            raise SpaceError(f"round_board: {k} is a whole number of at least 1, not {o[k]!r}")
    if o["sectors"] % 2:
        raise SpaceError("round_board: sectors must be even, or the colours do not alternate round a ring")
    if not 0 <= o["gap"] < o["width"]:
        raise SpaceError("round_board: gap is metres between cells, less than a ring's width")
    return o


def radii(inner, width, rings):
    """[(r0, r1)] of each ring from the centre out."""
    return [(inner + i * width, inner + (i + 1) * width) for i in range(rings)]


def omegas(o):
    """Each ring's turn per unit of the root's `phase`: omega * (r_mid / r_mid_0) ** kepler."""
    mids = [0.5 * (a + b) for a, b in radii(o["inner"], o["width"], o["rings"])]
    return [float(o["omega"]) * (m / mids[0]) ** float(o["kepler"]) for m in mids]


def light(ring, sector):
    """True for a light cell: they alternate round a ring and from ring to ring."""
    return (ring + sector) % 2 == 0


def cell_mesh(r0, r1, a0, a1, thick, gap, mat=0, steps=None):
    """An annular-sector slab from radius r0 to r1 and angle a0 to a1 (radians), its top at z = 0 and `thick` deep,
    inset by gap / 2 on every side (radially and along the arc, in metres)."""
    g = gap / 2.0
    r0, r1 = r0 + g, r1 - g
    da = g / max(0.5 * (r0 + r1), 1e-6)
    a0, a1 = a0 + da, a1 - da
    n = steps or max(2, int(math.ceil((a1 - a0) * r1 / 0.15)))
    ang = np.linspace(a0, a1, n + 1)
    c, s = np.cos(ang), np.sin(ang)
    V = []
    for z in (0.0, -thick):
        for r in (r0, r1):
            V.append(np.stack([r * c, r * s, np.full(n + 1, z)], 1))
    V = np.concatenate(V)                         # blocks: top inner, top outer, bottom inner, bottom outer
    ti, to, bi, bo = (np.arange(n + 1) + k * (n + 1) for k in range(4))
    j, j1 = np.arange(n), np.arange(1, n + 1)
    Q = [np.stack([ti[j], to[j], to[j1], ti[j1]], 1),             # top (normals up)
         np.stack([bi[j], bi[j1], bo[j1], bo[j]], 1),             # bottom
         np.stack([to[j], bo[j], bo[j1], to[j1]], 1),             # outer wall
         np.stack([ti[j], ti[j1], bi[j1], bi[j]], 1),             # inner wall
         np.array([[ti[0], bi[0], bo[0], to[0]], [ti[n], to[n], bo[n], bi[n]]])]   # the two ends
    return CS.Mesh(V, np.concatenate(Q), mat=mat)


def ring_mesh(o, ring):
    """One ring's cells as one mesh: material 0 the light cells, 1 the dark."""
    r0, r1 = radii(o["inner"], o["width"], o["rings"])[ring]
    step = 2.0 * math.pi / o["sectors"]
    return CS.merge([cell_mesh(r0, r1, k * step, (k + 1) * step, o["thick"], o["gap"], 0 if light(ring, k) else 1)
                     for k in range(o["sectors"])])


def hot_mesh(o, ring, sector, lift=0.004):
    """The overlay of one cell: a thin slab just above it."""
    r0, r1 = radii(o["inner"], o["width"], o["rings"])[ring]
    step = 2.0 * math.pi / o["sectors"]
    return cell_mesh(r0, r1, sector * step, (sector + 1) * step, 0.004, o["gap"]).moved((0.0, 0.0, lift))


def cell_centre(o, ring, sector, phase=0.0):
    """(x, y) of a cell's centre in the board's frame after its ring has turned for `phase`."""
    r0, r1 = radii(o["inner"], o["width"], o["rings"])[ring]
    a = (sector + 0.5) * 2.0 * math.pi / o["sectors"] + omegas(o)[ring] * phase
    r = 0.5 * (r0 + r1)
    return r * math.cos(a), r * math.sin(a)


# ================================================================================================================ chess
# Profiles (r, z) as shares of the piece's height, bottom up, outside on the right (core.shell.lathe).
_BASE = [(0.0, 0.0), (0.33, 0.0), (0.33, 0.045), (0.30, 0.07), (0.27, 0.085), (0.27, 0.10), (0.22, 0.125)]
_PROFILES = {
    "pawn": _BASE + [(0.15, 0.19), (0.11, 0.33), (0.10, 0.44), (0.17, 0.48), (0.17, 0.51), (0.10, 0.54)]
    + [(0.19 * math.cos(a), 0.74 + 0.19 * math.sin(a)) for a in np.radians(np.linspace(-60, 90, 9))],
    "rook": _BASE + [(0.19, 0.18), (0.17, 0.60), (0.23, 0.66), (0.25, 0.70), (0.25, 1.0), (0.17, 1.0), (0.17, 0.93),
                     (0.0, 0.93)],
    "bishop": _BASE + [(0.15, 0.19), (0.10, 0.45), (0.17, 0.50), (0.17, 0.53), (0.09, 0.56)]
    + [(0.16 * math.cos(a), 0.70 + 0.21 * math.sin(a)) for a in np.radians(np.linspace(-55, 75, 8))]
    + [(0.035, 0.91), (0.06 * math.cos(math.radians(-40)), 0.95 + 0.06 * math.sin(math.radians(-40))), (0.06, 0.95),
       (0.04, 0.99), (0.0, 1.0)],
    "queen": _BASE + [(0.16, 0.19), (0.10, 0.52), (0.18, 0.57), (0.18, 0.60), (0.10, 0.63), (0.12, 0.70), (0.24, 0.86),
                      (0.20, 0.88), (0.12, 0.88), (0.07, 0.90)]
    + [(0.06 * math.cos(a), 0.94 + 0.06 * math.sin(a)) for a in np.radians(np.linspace(-45, 90, 6))],
    "king": _BASE + [(0.16, 0.19), (0.10, 0.52), (0.18, 0.57), (0.18, 0.60), (0.10, 0.63), (0.12, 0.70), (0.21, 0.82),
                     (0.21, 0.84), (0.12, 0.86), (0.0, 0.86)],
}
# The knight's head, side on, facing +x: (x, z) as shares of the height, from z = 0.10 up (it stands on the base).
_KNIGHT = [(-0.24, 0.10), (0.22, 0.10), (0.20, 0.30), (0.26, 0.42), (0.38, 0.53), (0.44, 0.60), (0.42, 0.67),
           (0.30, 0.66), (0.20, 0.70), (0.17, 0.79), (0.10, 0.88), (0.04, 0.99), (-0.02, 0.92), (-0.10, 0.86),
           (-0.20, 0.72), (-0.26, 0.52), (-0.25, 0.30)]


def chess_options(slots):
    o = options(CHESS, slots, "chess")
    if o["piece"] not in PIECES:
        raise SpaceError(f"chess: piece is one of {', '.join(PIECES)}, not {o['piece']!r}")
    _positive(o, ("height",), "chess")
    return o


def profile(piece, height=1.0):
    """The lathe profile (r, z) of a piece other than the knight, in metres for a piece `height` tall."""
    if piece not in _PROFILES:
        raise SpaceError(f"chess: no lathe profile for {piece!r} (have {', '.join(_PROFILES)})")
    return np.asarray(_PROFILES[piece], float) * height


def knight_outline(height=1.0):
    """The knight's head outline (x, z), counter-clockwise, in metres."""
    P = np.asarray(_KNIGHT, float)
    return (P if CS.polygon_area(P) > 0 else P[::-1]) * height


def piece_mesh(piece, height=1.0, seg=40):
    """The piece's mesh standing on z = 0 at the origin, `height` tall (the knight faces +X)."""
    if piece == "knight":
        base = CS.lathe(np.asarray(_BASE + [(0.20, 0.13), (0.0, 0.13)], float) * height, seg=seg)
        half = CS.rounded_panel(knight_outline(height), 0.15 * height, r_edge=0.05 * height, n=4, dome=0.05 * height)
        plate = CS.merge([half, half.apply(np.diag([1.0, 1.0, -1.0]))])               # domed on both faces
        return CS.merge([base, plate.rot(90.0, 0.0, 0.0)])                             # stood up, on the base
    body = CS.lathe(profile(piece, height), seg=seg)
    if piece != "king":
        return body
    w, t, top = 0.05 * height, 0.06 * height, 0.86 * height                     # the cross on the crown
    upright = CS.rounded_box((t, t, 0.20 * height), center=(0.0, 0.0, top + 0.10 * height), r=0.012 * height)
    arm = CS.rounded_box((0.16 * height, t, w), center=(0.0, 0.0, top + 0.13 * height), r=0.012 * height)
    return CS.merge([body, upright, arm])


# ================================================================================================================ stars
def stars_options(slots):
    o = options(STARS, slots, "stars")
    if not isinstance(o["kind"], int) or o["kind"] < 3:
        raise SpaceError(f"stars: kind is the number of points (3 or more), not {o['kind']!r}")
    if not 0 < o["inner"] < 1:
        raise SpaceError("stars: inner is the inner radius as a share of the outer, between 0 and 1")
    _positive(o, ("size", "depth", "flash", "gain", "line_width", "draw"), "stars")
    pts = []
    for i, p in enumerate(o["points"]):
        if not (isinstance(p, (list, tuple)) and len(p) in (3, 4) and all(isinstance(v, (int, float)) for v in p)):
            raise SpaceError(f"stars: points[{i}] is [x, z, size] or [x, z, size, t] (metres in the picture's plane, "
                             f"a size factor, the clip second it flashes)")
        pts.append([float(v) for v in p] + ([None] if len(p) == 3 else []))
    o["points"] = pts
    return o


def star_outline(n, inner):
    """A star of n points (outer radius 1, inner `inner`), the first point up, counter-clockwise."""
    k = np.arange(2 * n)
    a = 0.5 * math.pi + k * math.pi / n
    r = np.where(k % 2 == 0, 1.0, inner)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def star_mesh(n, inner, size, depth):
    """A flat star `size` m to its points, `depth` thick, facing -Y (its points in the x-z plane)."""
    P = star_outline(n, inner) * size
    return CS.rounded_panel(P, depth, r_edge=min(depth * 0.4, size * 0.05), n=1).rot(90.0, 0.0, 0.0)


def flash(ts, t0, length, gain):
    """A star's scale over clip seconds `ts`: 1, up to `gain` at t0 (a fast rise over a tenth of `length`), back to 1 by
    t0 + length (a smooth fall)."""
    ts = np.asarray(ts, float)
    up = np.clip((ts - (t0 - 0.1 * length)) / (0.1 * length), 0.0, 1.0)
    down = np.clip((ts - t0) / (0.9 * length), 0.0, 1.0)
    s = np.where(ts < t0, up, 1.0 - down * down * (3 - 2 * down))
    return 1.0 + (gain - 1.0) * s


def bar_mesh(length, width):
    """A thin rod along +X from the origin: a constellation's line, scaled along X as it draws itself."""
    return CS.tube(np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]), width / 2.0, sides=6)


# ================================================================================================================ floor
def floor_options(slots):
    o = options(FLOOR, slots, "floor")
    if o["kind"] not in FLOOR_KINDS:
        raise SpaceError(f"floor: kind is one of {', '.join(FLOOR_KINDS)}, not {o['kind']!r}")
    _positive(o, ("size", "cell", "line"), "floor")
    if o["rough"] is None:
        o["rough"] = FLOOR_ROUGH[o["kind"]]
    for k in ("rough", "metal"):
        if not isinstance(o[k], (int, float)) or not 0.0 <= o[k] <= 1.0:
            raise SpaceError(f"floor: {k} is 0 .. 1, not {o[k]!r}")
    if not isinstance(o["glow"], (int, float)) or o["glow"] < 0:
        raise SpaceError(f"floor: glow is a number, at least 0, not {o['glow']!r}")
    f = o["fade"]
    if not (isinstance(f, (list, tuple)) and len(f) == 2 and all(isinstance(v, (int, float)) for v in f) and 0 <= f[0] < f[1]):
        raise SpaceError(f"floor: fade is [near, far] metres from the camera, near < far, not {f!r}")
    if o["line"] >= o["cell"]:
        raise SpaceError("floor: a grid's line must be thinner than its cell")
    return o


def soften(dist):
    """The width (m) over which a pattern's edge blends at `dist` metres from the camera: a few pixels' footprint of a 35
    mm lens at 1080 lines, so a checker turns into its own grey and a line into its own glow before it can shimmer."""
    return np.maximum(0.004, 0.0025 * np.asarray(dist, float))


def checker(x, y, cell, soft):
    """0..1 of the light squares of a checker of `cell` m at floor points (x, y), its edges blended over `soft` m: 1 on a
    light square well inside it, 0.5 everywhere once `soft` is near the cell (the board's own grey)."""
    s = np.sin(np.pi * np.asarray(x, float) / cell) * np.sin(np.pi * np.asarray(y, float) / cell)
    return 0.5 + 0.5 * np.tanh(s * cell / (np.pi * np.asarray(soft, float)))


def grid_lines(x, y, cell, line, soft):
    """0..1 of a grid's lines (every `cell` m, `line` m wide) at floor points (x, y), each line box-filtered over `soft` m
    each side (a trapezoid: the line's own width of light, spread): the share of light a pixel there sees stays
    `line / cell` per line however far, so the grid dims rather than shimmers."""
    def one(v):
        d = np.abs(np.asarray(v, float) / cell - np.round(np.asarray(v, float) / cell)) * cell   # to the nearest line
        w = np.maximum(np.asarray(soft, float), 1e-6)
        return np.maximum(0.0, np.minimum(line / 2.0, d + w) - np.maximum(-line / 2.0, d - w)) / (2.0 * w)
    a, b = one(x), one(y)
    return a + b - a * b


# ================================================================================================================ starfield
def starfield_options(slots):
    o = options(STARFIELD, slots, "starfield")
    if not isinstance(o["count"], int) or not 1 <= o["count"] <= 20000:
        raise SpaceError(f"starfield: count is a whole number of stars per tile, 1 .. 20000, not {o['count']!r}")
    _positive(o, ("length", "width", "size"), "starfield")
    for k in ("streak", "clear", "glow"):
        if not isinstance(o[k], (int, float)) or o[k] < 0:
            raise SpaceError(f"starfield: {k} is a number, at least 0, not {o[k]!r}")
    if not isinstance(o["speed"], (int, float)):
        raise SpaceError(f"starfield: speed is m/s along -X, not {o['speed']!r}")
    h = o["height"]
    if not (isinstance(h, (list, tuple)) and len(h) == 2 and h[0] < h[1]):
        raise SpaceError(f"starfield: height is [low, high] metres, low < high, not {h!r}")
    s = o["surge"]
    if not isinstance(s, list) or not all(isinstance(k, (list, tuple)) and len(k) == 2 for k in s):
        raise SpaceError("starfield: surge is [[t, metres], ...]: extra way travelled, eased between keys")
    o["surge"] = sorted([float(t), float(v)] for t, v in s)
    if not isinstance(o["seed"], int):
        raise SpaceError(f"starfield: seed is a whole number, not {o['seed']!r}")
    return o


def star_points(o):
    """(points (n, 3), sizes (n,)): the stars of one tile, x in [0, length), y across, z in `height`, none within
    `clear` m of the X axis; sizes are shares of `size` (most stars small, a few bright). From the seed."""
    rng = np.random.default_rng(o["seed"])
    out = []
    lo, hi = float(o["height"][0]), float(o["height"][1])
    while len(out) < o["count"]:
        n = 2 * (o["count"] - len(out)) + 16
        P = np.stack([rng.uniform(0.0, o["length"], n), rng.uniform(-o["width"] / 2, o["width"] / 2, n),
                      rng.uniform(lo, hi, n)], 1)
        out += [p for p in P if np.hypot(p[1], p[2]) >= o["clear"]][:o["count"] - len(out)]
    sizes = 0.35 + 0.65 * rng.power(4.0, o["count"])[::-1] ** 3
    return np.asarray(out), np.round(sizes, 4)


def starfield_mesh(o):
    """The stars of one tile as one mesh: each a small box `size` across, `streak` m long along X (at least its own
    width: a dot), scaled by its share."""
    P, sizes = star_points(o)
    return CS.merge([_box((max(o["streak"] * s, o["size"] * s), o["size"] * s, o["size"] * s), p)
                     for p, s in zip(P, sizes)])


def _box(size, centre):
    """An axis-aligned box mesh (core.shell.Mesh) `size` (x, y, z) about `centre`."""
    sx, sy, sz = (0.5 * float(v) for v in size)
    c = np.asarray(centre, float)
    V = c + np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                      [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    Q = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return CS.Mesh(V, Q)


def travel(ts, o):
    """Metres the stars have streamed along -X by clip seconds `ts`: `speed` x t plus the `surge` keys (eased between
    them, held before the first and after the last)."""
    ts = np.asarray(ts, float)
    out = float(o["speed"]) * ts
    keys = o["surge"]
    if keys:
        add = np.full(ts.shape, keys[0][1])
        for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
            u = np.clip((ts - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
            seg = (ts >= t0) & (ts <= t1)
            add[seg] = v0 + (v1 - v0) * (u * u * (3.0 - 2.0 * u))[seg]
        add[ts > keys[-1][0]] = keys[-1][1]
        out = out + add
    return out


def tile_x(s, k, length):
    """The X of tile k (0 or 1; each `length` long, its stars at x in [0, length)) when the field has streamed `s` m:
    the two leapfrog, so together they always cover [-length, length) about the root."""
    return np.mod(k * length - np.asarray(s, float), 2.0 * length) - length


# ================================================================================================================ disc
def disc_options(slots):
    o = options(DISC, slots, "disc")
    _positive(o, ("inner", "outer", "streaks", "rings"), "disc")
    if o["outer"] <= o["inner"]:
        raise SpaceError("disc: outer must be more than inner")
    for k in ("glow", "spin"):
        if not isinstance(o[k], (int, float)) or (k == "glow" and o[k] < 0):
            raise SpaceError(f"disc: {k} is a number{', at least 0' if k == 'glow' else ''}, not {o[k]!r}")
    if not isinstance(o["kepler"], (int, float)):
        raise SpaceError(f"disc: kepler is a number (the exponent of the orbit speed), not {o['kepler']!r}")
    return o


def disc_turn(o, r, t):
    """Radians the disc's stuff at radius `r` has orbited by clip second `t`: `spin` turns a second at `inner`, faster
    inside and slower out as (r / inner) ** kepler."""
    return 2.0 * math.pi * float(o["spin"]) * (np.asarray(r, float) / o["inner"]) ** float(o["kepler"]) * t


def disc_mesh(o, sectors=96, rings=12):
    """The flat ring from `inner` to `outer` at z = 0, its normal up."""
    r = np.linspace(o["inner"], o["outer"], rings + 1)
    a = np.linspace(0.0, 2.0 * math.pi, sectors, endpoint=False)
    V = np.array([[ri * math.cos(aj), ri * math.sin(aj), 0.0] for ri in r for aj in a])
    Q = [[i * sectors + j, (i + 1) * sectors + j, (i + 1) * sectors + (j + 1) % sectors, i * sectors + (j + 1) % sectors]
         for i in range(rings) for j in range(sectors)]
    return CS.Mesh(V, np.array(Q))
