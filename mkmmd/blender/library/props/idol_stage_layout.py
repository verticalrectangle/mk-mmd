"""Layout of the idol stage (pure Python + numpy, no bpy: importable from tests): every number the parts agree on, the
options and colour roles, the meshes, the LED chase written once against an ops interface (`NumpyOps` here is the
reference; idol_stage.py runs the same functions on shader nodes, one node per op) and the prop card.

Frame (metres; everything in the card is in it): ORIGIN on the floor under the middle of the deck, Z up, the front (the
audience) toward -Y (card `front` = "-Y"), X to the right seen from the front. A pastel idol stage:

  riser    ONE cage of offset rings round a superellipse plan (`size` [w, d], 6.0 x 3.6, exponent PLAN_P: generous
           rounded corners): a foot rolled in under it, a padded fascia (the skirt) bulging BULGE out between the foot and
           the deck edge, a rolled bullnose (radius RB) overhanging the skirt by OVER, the flat deck at `height` (0.6),
           closed by fan caps top and bottom
  trim     a piping tube round the bullnose's outermost line
  steps    soft cushion steps (rounded panels of a superellipse plan, both edges rolled, the tread domed), rise = height
           / n with n = ceil(height / RISE_MAX): `steps` "front" (one flight centred on the front), "sides" (one flight on
           each side by the front corners, climbing toward the middle) or "none"; their backs run TUCK into the riser
  panels   LED panels in heart and star shapes at the back of the deck: a rounded housing DEPTH deep (rolled rim, a
           piping bead round its front edge) on slim posts, its face an LED dot matrix drawn by its material; bigger
           panels stand further back (each smaller size STAGGER forward), every post foot on the flat deck
  bars     pixel bars along the front fascia under the bullnose: swept rounded housings with a lens strip of LED pixels

The LED chase. Every LED face and lens is built in the stage frame and its object stands at identity under the root, so a
material's Object coordinates ARE the stage's frame: mkr draws no Texture Coordinate node with an object set (mkr-shade
marks "TEX_COORD with an object" unsupported), so the field reads each object's own Object coordinates instead. The dots
sit on a grid of `pitch` in the stage's x-z plane: round (radius DOT_R pitches) on the dark `ground`. Each dot cell has a
coordinate u from its centre, by the panel's `pattern`:

  sweep   x / wave                      left to right in the STAGE's frame: one sweep runs on across panels and bars
  rise    n_rise CDF(z)                 up the panel; CDF is the share of the face's area under z, so every band holds the
                                        same number of dots (n_rise = round(face height / wave), at least 1)
  ring    n_ring rho^2                  out from the panel's centre in scaled copies of its outline: rho = r / R(theta),
                                        R the face's radius at that angle (n_ring = round(size / (2 wave)), at least 1)
  spin    arms theta / 2 pi + rho^2     spiral arms (ARMS: 2 on a heart, 5 on a star) turning counter-clockwise seen
                                        from the front
  blink   (i + j mod 2) / 2             alternate dots (a checkerboard of the stage's dot grid)

and its lit level is a band travelling with the root property `chase` (cycles): level = band(fract(chase - u)), the band
rising over [0, BAND[0]] of a cycle and falling over [BAND[1], BAND[2]] (blink: BLINK, lit for half a cycle). As `chase`
grows the bands move toward larger u. A dot glows `alt` dimly and its colour when lit, the two lights added: color *
level + alt * ALT_K (1 - level), times glow * on. Flash-safe by construction: ring, spin, rise and blink spread
u evenly over a panel's dots (rho^2 and the CDF are shares of its area; blink lights one half of the checkerboard while
the other is dim), so as the bands move a panel's light stays steady (within a few per cent, the dots being discrete); a
sweep holds several bands across a panel (keep `wave` under half a panel's width). The bars' pixels run the same sweep
at level `bars`. R(theta) and the CDF are tables of TABLE_N points (a Float Curve node in each panel's material).

Live root properties (for [[key]]; their starting values are options): `chase` (the phase in cycles; keyable directly;
unkeyed the builder keys it linear at the starting `speed` from clip second 0), `speed` (cycles a second: a card `rates`
entry the keys stage integrates into `chase` once `speed` is keyed), `on` (0..1, the panels' level), `bars` (0..1, the
light bars' level).

Colour roles (`slots` of the prop: a palette slot, `ink` / `paper`, a '#hex' or an 'a:b:t' mix): deck (rose: the deck
and its bullnose), skirt (iris), trim (gold: the piping), treads (the steps), housing (overlay: the panels' and bars'
housings), heart (love), star (gold), heart_small (rose) (the lit dots of the default big heart, stars and small hearts),
alt (foam: the dim unlit dots and pixels), pixels (love: the bars' lit pixels), metal (subtle: the posts), ground (ink:
the LED faces between the dots). A panel's `color` / `alt` is a colour or a role name."""
import math

import numpy as np

from ....core import burst as BU
from ....core import hearts as HR
from ....core import shell as S
from . import showkit as SK

WHO = "idol_stage"

# ===================================================================================================================
# colour roles and material indices
# ===================================================================================================================
ROLES = {"deck": "rose", "skirt": "iris", "trim": "gold", "treads": "rose:paper:0.4", "housing": "overlay",
         "heart": "love", "star": "gold", "heart_small": "rose", "alt": "foam", "pixels": "love", "metal": "subtle",
         "ground": "ink"}
MATS = {name: i for i, name in enumerate(("deck", "skirt", "trim", "treads", "housing", "metal", "pixels"))}
LED0 = len(MATS)                    # the LED face of panel i has material index LED0 + i

# ===================================================================================================================
# the riser
# ===================================================================================================================
SIZE = (6.0, 3.6)                   # plan [w, d]
HEIGHT = 0.6                        # the deck's top
PLAN_P = 4.5                        # superellipse exponent of the plan
PLAN_N = 256                        # points round it
RB = 0.10                           # the bullnose's radius
OVER = 0.05                         # how far it overhangs the skirt line
RB_SEG = 10                         # segments from where it meets the skirt round to the deck
RF = 0.03                           # the foot's roll
BULGE = 0.03                        # the padded skirt's bulge at mid-height
TOP_SUPPORT = 0.04                  # a flat support ring this far in from the bullnose's top tangent
TRIM_R = 0.013                      # the piping round the bullnose: its radius, its centre TRIM_OUT radii out of it
TRIM_OUT = 0.6

# ===================================================================================================================
# steps
# ===================================================================================================================
STEPS = ("front", "sides", "none")
RISE_MAX = 0.18                     # n = ceil(height / RISE_MAX) risers, the last onto the deck
GOING = 0.30                        # tread depth
STEP_W = (1.5, 0.4)                 # front flight: width, at most this share of the stage's width
SIDE_W = (1.2, 0.45)                # side flights: width, at most this share of the stage's depth
SIDE_AT = 0.7                       # side flights: their front edge this share of the half depth in front of the middle
STEP_P = 4.0                        # superellipse exponent of a step's plan
STEP_DOME = 0.008                   # the tread's puff
TUCK = 0.20                         # a step's back runs this far inside the skirt line

# ===================================================================================================================
# panels
# ===================================================================================================================
SHAPES = ("heart", "star")
STAR_ROUND = 3.5                    # the star's tips rounded (core.burst.star_outline's rounding: smaller is rounder)
PATTERNS = ("sweep", "rise", "ring", "spin", "blink")
DEPTH = 0.10                        # housing depth
R_RIM, R_BACK = 0.03, 0.02          # its rolled rim (front) and back edge; the rim at most RIM_TIP of the tips' radius
RIM_TIP = 0.7
BEAD_R = 0.011                      # the piping bead round the front edge, at most BEAD_TIP of the tips' radius
BEAD_TIP = 0.6
POST_R, FOOT_R, FOOT_H = 0.022, 0.075, 0.02
INSERT = 0.06                       # a post runs this far up into the housing
POST_X = 0.22                       # two posts at +-POST_X * size (one centred under a panel smaller than TWO_POSTS)
TWO_POSTS = 1.2
MIN_LIFT = 0.15                     # a panel's lowest point at least this far over the deck
BACK_IN = 0.35                      # the biggest panels' back this far in from the deck's back edge
STAGGER = 0.16                      # each smaller size stands this much further forward
ARMS = {"heart": 2, "star": 5}
SHAPE_SIZE = {"heart": 1.6, "star": 1.2}
DEFAULT_PANELS = (
    {"shape": "heart", "at": [0.0, 2.25], "size": 2.2, "pattern": "ring", "color": "heart"},
    {"shape": "star", "at": [-1.72, 2.62], "size": 1.3, "pattern": "spin", "color": "star"},
    {"shape": "star", "at": [1.72, 2.62], "size": 1.3, "pattern": "spin", "color": "star"},
    {"shape": "heart", "at": [-2.42, 1.42], "size": 0.9, "pattern": "blink", "color": "heart_small"},
    {"shape": "heart", "at": [2.42, 1.42], "size": 0.9, "pattern": "blink", "color": "heart_small"},
)
PANEL_KEYS = ("shape", "at", "size", "pattern", "color", "alt")

# ===================================================================================================================
# light bars
# ===================================================================================================================
BAR_H, BAR_D, BAR_R = 0.065, 0.06, 0.018        # housing section (height, depth, corner radius)
LENS_H, LENS_D, LENS_R = 0.034, 0.012, 0.005    # the lens strip on its front (sunk LENS_SINK into the housing)
LENS_SINK, LENS_INSET = 0.005, 0.035            # ... and LENS_INSET short of the housing's ends
BAR_LEN, BAR_GAP = 1.05, 0.06                    # a bar's length (about) and the gap between bars
BAR_DROP = 0.02                                  # the bars' top this far under where the bullnose meets the skirt
BAR_SINK = 0.006                                 # their backs sunk into the skirt
FRONT_DEG = 35.0                                 # the front: where the skirt's outward normal is this close to -Y
STEP_CLEAR = 0.15                                # bars keep this far from a front flight
END_SEG = 6                                      # rings in a bar's rounded end

# ===================================================================================================================
# the LED field
# ===================================================================================================================
PITCH = 0.05                        # dot pitch (option `pitch`)
WAVE = 0.47                         # band spacing of sweep and rise, the rings' scale (option `wave`)
GLOW = 2.0                          # emission of a lit dot at on = 1 (option `glow`)
DOT_R, DOT_SOFT = 0.31, 0.07        # a dot's radius and its soft edge, in pitches
BAND = (0.12, 0.12, 0.72)           # the chase band: rises over [0, a], falls over [b, c] of a cycle
BLINK = (0.08, 0.5, 0.58)           # blink: half a cycle lit, so the two halves of the checkerboard always add to one
ALT_K = 0.1                         # the dim dots' share of a lit dot's light
TWIST = 1                           # spin: the arms turn a whole cycle from the centre to the rim (keeps u even)
TABLE_N = 65                        # points of a panel's radius and area tables (Float Curve nodes)
STANCE = 0.1                        # use.feet: the ankles this far either side of a mark

PARAMS = {                          # live root properties for [[key]]: (default, (lo, hi), description)
    "speed": (0.5, (-8.0, 8.0), "the chase's speed, cycles a second (the keys stage integrates it into `chase`)"),
    "on": (1.0, (0.0, 1.0), "the LED panels' level, 0 .. 1"),
    "bars": (1.0, (0.0, 1.0), "the light bars' level, 0 .. 1"),
}
CHASE_DOC = "the chase's phase, cycles since clip second 0 (from `speed`; key it to step the chase directly)"
NUMBERS = {                         # options: (default, lo, hi)
    "height": (HEIGHT, 0.35, 1.6),
    "pitch": (PITCH, 0.02, 0.25),
    "wave": (WAVE, 0.1, 10.0),
    "glow": (GLOW, 0.0, 100.0),
}


# ===================================================================================================================
# options
# ===================================================================================================================
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def roles(slots):
    """{role: spec} of every colour role (the card's `slots`), checked."""
    return SK.roles(slots, ROLES, WHO)


def _colour(spec, slots, rl, what):
    """A panel's colour: a role name (its spec) or a colour spec, checked."""
    if isinstance(spec, str) and spec in rl:
        return rl[spec]
    SK.rgb(spec, slots, what)
    return spec


def options(slots):
    """The prop's options from the `slots` of the [[prop]], checked (a bad value is a ValueError that says what to
    change): {size, height, steps, pitch, wave, glow, panels (normalised and placed), start {speed, on, bars}}."""
    slots = dict(slots or {})
    O = {k: SK.number(slots, k, d, WHO, lo, hi) for k, (d, lo, hi) in NUMBERS.items()}
    size = slots.get("size", list(SIZE))
    if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(_num(v) for v in size)
            and 2.0 <= size[0] <= 20.0 and 1.5 <= size[1] <= 12.0):
        raise ValueError(f"{WHO}: size is [width, depth] in metres, width 2 .. 20 and depth 1.5 .. 12 (got {size!r})")
    O["size"] = (float(size[0]), float(size[1]))
    if O["height"] > 0.5 * O["size"][1]:
        raise ValueError(f"{WHO}: height {O['height']:g} is more than half the depth {O['size'][1]:g}: lower it or "
                         f"deepen the stage")
    steps = slots.get("steps", "front")
    if steps not in STEPS:
        raise ValueError(f"{WHO}: steps is one of {', '.join(map(repr, STEPS))} (got {steps!r})")
    O["steps"] = steps
    O["start"] = {k: SK.number(slots, k, d, WHO, lo, hi) for k, (d, (lo, hi), _doc) in PARAMS.items()}
    O["panels"] = _panels(slots, O)
    return O


def _panels(slots, O):
    rl = roles(slots)
    raw = slots.get("panels", [dict(p) for p in DEFAULT_PANELS])
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{WHO}: panels is a list of {{shape, at, size, pattern, color, alt}} tables (got {raw!r})")
    out = []
    for i, p in enumerate(raw):
        who = f"{WHO}: panels[{i}]"
        if not isinstance(p, dict):
            raise ValueError(f"{who} is a table {{shape, at, size, pattern, color, alt}} (got {p!r})")
        bad = sorted(set(p) - set(PANEL_KEYS))
        if bad:
            raise ValueError(f"{who}: unknown keys {bad} (a panel has {', '.join(PANEL_KEYS)})")
        shape = p.get("shape")
        if shape not in SHAPES:
            raise ValueError(f"{who}: shape is \"heart\" or \"star\" (got {shape!r})")
        at = p.get("at")
        if not (isinstance(at, (list, tuple)) and len(at) == 2 and all(_num(v) for v in at)):
            raise ValueError(f"{who}: at is [x, z], the panel's middle in metres (got {at!r})")
        size = p.get("size", SHAPE_SIZE[shape])
        if not (_num(size) and 0.3 <= size <= 6.0):
            raise ValueError(f"{who}: size is the panel's width, 0.3 .. 6 m (got {size!r})")
        pattern = p.get("pattern", "ring" if shape == "heart" else "spin")
        if pattern not in PATTERNS:
            raise ValueError(f"{who}: pattern is one of {', '.join(PATTERNS)} (got {pattern!r})")
        out.append({"shape": shape, "at": (float(at[0]), float(at[1])), "size": float(size), "pattern": pattern,
                    "color": _colour(p.get("color", shape), slots, rl, f"{who}: color"),
                    "alt": _colour(p.get("alt", "alt"), slots, rl, f"{who}: alt")})
    _place_panels(out, O)
    return out


# ===================================================================================================================
# plan and profile
# ===================================================================================================================
def plan(O):
    """The riser's plan (PLAN_N, 2): the skirt line, a superellipse counter-clockwise from +X, evenly spaced."""
    w, d = O["size"]
    return S.resample_loop(S.superellipse(w, d, PLAN_P, 2048), PLAN_N)


def junction(O):
    """The height where the bullnose's underside meets the skirt line."""
    return O["height"] - RB - math.sqrt(RB * RB - (RB - OVER) ** 2)


def bulge(O, h):
    """The skirt's offset (negative: out) at height h: a sine bulge from the foot up to the bullnose."""
    j = junction(O)
    return -BULGE * math.sin(math.pi * min(max((h - RF) / (j - RF), 0.0), 1.0))


def profile(O):
    """[(d, h)] of the riser's rings, d the inset from the skirt line (negative: out), from the floor up: the foot
    rolling in, the bulging skirt, the bullnose over to the deck, a flat support ring."""
    H, j = O["height"], junction(O)
    prof = [(RF * (1.0 - math.sin(a)), RF * (1.0 - math.cos(a))) for a in np.linspace(0.0, 0.5 * math.pi, 4)]
    for h in (RF + 0.02, RF + 0.25 * (j - RF), RF + 0.5 * (j - RF), RF + 0.75 * (j - RF), j - 0.02):
        prof.append((bulge(O, h), h))
    prof.append((0.0, j))
    c_out, c_up = OVER - RB, H - RB
    a0 = math.atan2(j - c_up, -c_out)
    for a in np.linspace(a0, 0.5 * math.pi, RB_SEG + 1)[1:]:
        prof.append((-(c_out + RB * math.cos(a)), c_up + RB * math.sin(a)))
    prof.append((RB - OVER + TOP_SUPPORT, H))
    return prof


def flat_outline(O):
    """The deck's flat top: the plan inset to where the bullnose's top meets it."""
    return S.offset_rings(plan(O), [(RB - OVER, 0.0)])[0][:, :2]


def _crossings(P, x):
    """y of the crossings of the vertical line at x with the closed polygon P (k, 2), sorted (empty when none)."""
    A, B = P, np.roll(P, -1, 0)
    lo, hi = np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
    hit = (lo <= x) & (x < hi)
    t = (x - A[hit, 0]) / (B[hit, 0] - A[hit, 0])
    return np.sort(A[hit, 1] + t * (B[hit, 1] - A[hit, 1]))


def edge(P, x, back=True):
    """The back (largest y) or front (smallest y) edge of the closed polygon P at x, or None off it."""
    c = _crossings(P, x)
    return None if len(c) == 0 else float(c[-1] if back else c[0])


def _span_edge(P, x0, x1, back=True, n=9):
    """The edge of P over x0 .. x1 at its innermost (the back edge's lowest, the front's highest), or None."""
    ys = [edge(P, x, back) for x in np.linspace(x0, x1, n)]
    if any(y is None for y in ys):
        return None
    return min(ys) if back else max(ys)


# ===================================================================================================================
# panels: placement, outlines, the LED face
# ===================================================================================================================
def outline(shape, n=192):
    """The unit-width outline (n, 2) (x, z) of a panel shape, counter-clockwise seen from the front, round its area
    centroid."""
    return HR.heart_outline(n) if shape == "heart" else BU.star_outline(n, rounding=STAR_ROUND)


def _tip_radius(shape):
    """The smallest radius of curvature of a shape's convex turns (its tips), at unit width."""
    P = outline(shape)
    a, c = np.roll(P, 1, 0), np.roll(P, -1, 0)
    ab, bc, ca = (np.linalg.norm(u, axis=1) for u in (P - a, c - P, a - c))
    cross = (P - a)[:, 0] * (c - P)[:, 1] - (P - a)[:, 1] * (c - P)[:, 0]
    return float((ab * bc * ca / (2.0 * np.abs(cross) + 1e-15))[cross > 0].min())


def rim(p):
    """(rim roll, bead radius) of a panel: R_RIM and BEAD_R, smaller on a small panel whose tips are tighter."""
    tip = _tip_radius(p["shape"]) * p["size"]
    return min(R_RIM, RIM_TIP * tip), min(BEAD_R, BEAD_TIP * tip)


def post_xs(p):
    """x of a panel's posts (stage frame)."""
    x, s = p["at"][0], p["size"]
    return [x] if s < TWO_POSTS else [x - POST_X * s, x + POST_X * s]


def _place_panels(panels, O):
    """Each panel's back plane y (bigger panels further back, each smaller size STAGGER forward, every post's foot on the
    flat deck), its face plane and its posts [(x, top z)]; a panel that does not fit is a ValueError."""
    w, d = O["size"]
    H = O["height"]
    flat = flat_outline(O)
    sizes = sorted({p["size"] for p in panels}, reverse=True)
    for i, p in enumerate(panels):
        who = f"{WHO}: panels[{i}]"
        P = outline(p["shape"]) * p["size"] + p["at"]
        z_low = float(P[:, 1].min())
        if z_low < H + MIN_LIFT:
            raise ValueError(f"{who}: its bottom is at z {z_low:.2f}, under the deck's top + {MIN_LIFT:g}: raise at's z "
                             f"by {H + MIN_LIFT - z_low:.2f}")
        y = d / 2.0 - BACK_IN - STAGGER * sizes.index(p["size"])
        posts = []
        for x in post_xs(p):
            back = _span_edge(flat, x - FOOT_R - 0.02, x + FOOT_R + 0.02)
            if back is None:
                raise ValueError(f"{who}: its post at x {x:.2f} is off the deck (it spans x {-w / 2:.2f} .. "
                                 f"{w / 2:.2f}): move at's x in")
            y = min(y, back - FOOT_R - 0.02 + DEPTH / 2.0)
            posts.append((x, _lowest(P, x)))
        xs = [x for x, _z in posts]
        front = _span_edge(flat, min(xs) - FOOT_R, max(xs) + FOOT_R, back=False)
        if y - DEPTH < front + 0.4:
            raise ValueError(f"{who}: the deck is too shallow for it at x {p['at'][0]:.2f}: move it in or deepen the "
                             f"stage")
        p["y"] = round(y, 4)                            # the housing's back plane
        p["face_y"] = round(y - DEPTH, 4)               # its LED face
        p["posts"] = [(x, round(z + INSERT, 4)) for x, z in posts]


def _lowest(P, x):
    """The lowest z where the vertical line at x crosses the closed polygon P (x, z)."""
    c = _crossings(P, x)
    return float(c[0])


def face(p):
    """A panel's LED face: its outline (k, 2) in the stage's x-z plane (the housing's outline inset by its rim's roll,
    where the roll meets the flat face; the face itself lies in the plane y = p["face_y"])."""
    r, _bead = rim(p)
    L = outline(p["shape"]) * p["size"]
    return L + r * S.mitre_offsets(L) + np.asarray(p["at"])


def _polar(P, c, ang):
    """Distance from c to the closed polygon P (k, 2) along the directions `ang` (radians): the nearest crossing."""
    A = np.asarray(P, float) - np.asarray(c, float)
    e = np.roll(A, -1, 0) - A
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    det = d[:, None, 1] * e[None, :, 0] - d[:, None, 0] * e[None, :, 1]
    ok = np.abs(det) > 1e-15
    det = np.where(ok, det, 1.0)
    t = (A[None, :, 1] * e[None, :, 0] - A[None, :, 0] * e[None, :, 1]) / det
    s = (d[:, None, 0] * A[None, :, 1] - d[:, None, 1] * A[None, :, 0]) / det
    t = np.where(ok & (s >= -1e-12) & (s <= 1.0 + 1e-12) & (t > 0.0), t, np.inf)
    return t.min(1)


def radius_table(p):
    """(xs, ys, rmax): the face's radius R about the panel's middle at the folded angle theta' = atan2(dz, |dx|) (the
    shapes are mirror symmetric), tabled at xs = theta' / pi + 1/2 (TABLE_N points on 0..1), ys = R / rmax."""
    ang = np.linspace(-0.5 * math.pi, 0.5 * math.pi, TABLE_N)
    R = _polar(face(p), p["at"], ang)
    rmax = float(R.max())
    return np.linspace(0.0, 1.0, TABLE_N), R / rmax, rmax


def _widths(F, zs):
    """The length inside the closed polygon F (x, z) of the horizontal lines at heights zs."""
    out = []
    for z in zs:
        c = _crossings(F[:, ::-1], z)
        out.append(float((c[1::2] - c[0::2]).sum()) if len(c) >= 2 else 0.0)
    return np.array(out)


def area_table(p):
    """(xs, ys, z0, z1): the share of the face's area under z (its dots' heights spread evenly), z from z0 (the face's
    bottom) to z1 (its top) tabled at xs = (z - z0) / (z1 - z0) (TABLE_N points on 0..1)."""
    F = face(p)
    z0, z1 = float(F[:, 1].min()), float(F[:, 1].max())
    zs = np.linspace(z0, z1, 8 * TABLE_N)
    wd = _widths(F, zs)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (wd[1:] + wd[:-1]) * np.diff(zs))])
    xs = np.linspace(0.0, 1.0, TABLE_N)
    return xs, np.interp(z0 + xs * (z1 - z0), zs, cum / cum[-1]), z0, z1


def field(p, O):
    """The constants of a panel's LED field: what its material is built from and the reference reads (`led`)."""
    f = {"pattern": p["pattern"], "pitch": O["pitch"], "wave": O["wave"], "cx": p["at"][0], "cz": p["at"][1]}
    if p["pattern"] in ("ring", "spin"):
        xs, ys, rmax = radius_table(p)
        f.update(table=(xs, ys), rmax=rmax, n=max(1, int(round(p["size"] / (2.0 * O["wave"])))),
                 arms=ARMS[p["shape"]])
    elif p["pattern"] == "rise":
        xs, ys, z0, z1 = area_table(p)
        f.update(table=(xs, ys), z0=z0, z1=z1, n=max(1, int(round((z1 - z0) / O["wave"]))))
    return f


# ===================================================================================================================
# the LED field, written once against the ops `o` (NumpyOps below; shader nodes in idol_stage.py): floats with + - *,
# Python numbers staying constants, and floor, fract, fmod, abs, sqrt, atan2, div, smoothstep, curve
# ===================================================================================================================
def cell(o, x, z, pitch):
    """(i, j, fx, fz): the dot cell of stage points (x, z) on the grid of `pitch`, and the offset from its centre in
    pitches."""
    gx, gz = x * (1.0 / pitch), z * (1.0 / pitch)
    i, j = o.floor(gx), o.floor(gz)
    return i, j, gx - i - 0.5, gz - j - 0.5


def dot(o, fx, fz):
    """1 on a round dot (DOT_R pitches), 0 on the ground, a soft edge between."""
    return 1.0 - o.smoothstep(DOT_R - DOT_SOFT, DOT_R + DOT_SOFT, o.sqrt(fx * fx + fz * fz))


def u_of(o, f, i, j):
    """The chase coordinate u of the dot cell (i, j) by the panel's pattern (see the module docstring)."""
    pitch, pattern = f["pitch"], f["pattern"]
    cx, cz = (i + 0.5) * pitch, (j + 0.5) * pitch
    if pattern == "sweep":
        return cx * (1.0 / f["wave"])
    if pattern == "blink":
        return 0.5 * o.fmod(i + j, 2.0)
    if pattern == "rise":
        return f["n"] * o.curve((cz - f["z0"]) * (1.0 / (f["z1"] - f["z0"])), f["table"])
    dx, dz = cx - f["cx"], cz - f["cz"]
    c = o.curve(o.atan2(dz, o.abs(dx)) * (1.0 / math.pi) + 0.5, f["table"])
    rho2 = o.div((dx * dx + dz * dz) * (1.0 / (f["rmax"] * f["rmax"])), c * c)
    if pattern == "ring":
        return f["n"] * rho2
    return o.atan2(dz, dx) * (f["arms"] / (2.0 * math.pi)) + TWIST * rho2


def band(o, p, pattern):
    """The lit level at phase p = fract(chase - u): rising over [0, a] of a cycle, falling over [b, c]."""
    a, b, c = BLINK if pattern == "blink" else BAND
    return o.smoothstep(0.0, a, p) - o.smoothstep(b, c, p)


def led(o, f, x, z, chase):
    """(dot, level) of a panel's LED face (field f) at stage points (x, z) at the phase `chase`."""
    i, j, fx, fz = cell(o, x, z, f["pitch"])
    return dot(o, fx, fz), band(o, o.fract(chase - u_of(o, f, i, j)), f["pattern"])


def pixel(o, x, z, chase, pitch, wave, zc):
    """(dot, level) of the bars' lens at stage points (x, z): one row of pixels along x at the height zc, chased by the
    sweep (the panels' sweep: u = x / wave of the pixel's centre)."""
    gx = x * (1.0 / pitch)
    i = o.floor(gx)
    return (dot(o, gx - i - 0.5, (z - zc) * (1.0 / pitch)),
            band(o, o.fract(chase - (i + 0.5) * (pitch / wave)), "sweep"))


class NumpyOps:
    """The ops on numpy arrays, as Blender's nodes compute them: division by zero gives 0, the square root of a negative
    number 0, fmod is floored, smoothstep is Map Range's (clamped), curve a Float Curve through the table's points."""
    floor = staticmethod(np.floor)
    abs = staticmethod(np.abs)
    atan2 = staticmethod(np.arctan2)

    @staticmethod
    def fract(x):
        x = np.asarray(x, float)
        return x - np.floor(x)

    @staticmethod
    def fmod(a, b):
        a = np.asarray(a, float)
        return a - np.floor(a / b) * b

    @staticmethod
    def sqrt(x):
        return np.sqrt(np.maximum(x, 0.0))

    @staticmethod
    def div(a, b):
        b = np.asarray(b, float)
        ok = b != 0.0
        return np.where(ok, np.asarray(a, float) / np.where(ok, b, 1.0), 0.0)

    @staticmethod
    def smoothstep(a, b, x):
        f = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
        return f * f * (3.0 - 2.0 * f)

    @staticmethod
    def curve(t, table):
        return np.interp(t, table[0], table[1])


NP = NumpyOps()


def inside(P, x, y):
    """Even-odd test of points (x, y) against the closed polygon P (k, 2): a bool array."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A, B = P, np.roll(P, -1, 0)
    out = np.zeros(x.shape, bool)
    for (ax, ay), (bx, by) in zip(A, B):
        cross = (ay > y) != (by > y)
        xc = ax + (y - ay) * (bx - ax) / np.where(by != ay, by - ay, 1.0)
        out ^= cross & (x < xc)
    return out


def dots(p, pitch):
    """(i, j, x, z): the dot cells of a panel whose centres lie on its LED face, and those centres (stage frame)."""
    F = face(p)
    lo, hi = F.min(0), F.max(0)
    I, J = np.meshgrid(np.arange(math.floor(lo[0] / pitch), math.ceil(hi[0] / pitch) + 1),
                       np.arange(math.floor(lo[1] / pitch), math.ceil(hi[1] / pitch) + 1))
    I, J = I.ravel().astype(float), J.ravel().astype(float)
    x, z = (I + 0.5) * pitch, (J + 0.5) * pitch
    k = inside(F, x, z)
    return I[k], J[k], x[k], z[k]


def levels(f, i, j, chase):
    """The reference: the lit level of the dot cells (i, j) of a panel with field f at the phase `chase`."""
    return band(NP, NP.fract(chase - u_of(NP, f, i, j)), f["pattern"])


# ===================================================================================================================
# meshes (numpy; the builder makes them objects, every one standing at identity under the root)
# ===================================================================================================================
def riser_mesh(O):
    """The riser: one closed cage of offset rings (the profile round the plan) with fan caps; the skirt (and the
    bottom) `skirt`, the bullnose and the deck `deck`."""
    j = junction(O)
    R = S.offset_rings(plan(O), profile(O))
    m = S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), cap_rings=2,
               mat=MATS["skirt"])
    return m.assign(MATS["deck"], lambda c, _n: c[:, 2] > j + 1e-6)


def trim_mesh(O):
    """The piping round the bullnose's outermost line."""
    path = S.offset_rings(plan(O), [(-(OVER + TRIM_OUT * TRIM_R), O["height"] - RB)])[0]
    return _closed(S.tube(path, TRIM_R, sides=10, closed=True, mat=MATS["trim"]))


def _closed(m):
    """A swept loop welded at its seam, faces outward."""
    m = m.weld(1e-9)
    return m if m.volume() > 0 else m.flip()


def flight(O):
    """(risers, rise): the steps' risers, the last one onto the deck, and their height."""
    n = max(2, int(math.ceil(O["height"] / RISE_MAX - 1e-9)))
    return n, O["height"] / n


def steps_mesh(O):
    """The steps (None for steps = "none"): cushions, step k (1 .. n-1) from z (k - 1) rise to k rise reaching
    (n - k) GOING out of the skirt, its back TUCK inside it."""
    if O["steps"] == "none":
        return None
    (w, d), P = O["size"], plan(O)
    n, rise = flight(O)
    r_top, r_bot = min(0.05, 0.35 * rise), min(0.035, 0.25 * rise)

    def block(a, b, k):
        return S.rounded_panel(S.superellipse(a, b, STEP_P, 96), rise, r_edge=r_top, n=4, dome=STEP_DOME, r_back=r_bot,
                               mat=MATS["treads"]).moved((0.0, 0.0, (k - 1) * rise))

    out = []
    if O["steps"] == "front":
        sw = min(STEP_W[0], STEP_W[1] * w)
        y_out, y_back = edge(P, 0.0, back=False), _span_edge(P, -sw / 2.0, sw / 2.0, back=False) + TUCK
        for k in range(1, n):
            y_f = y_out - (n - k) * GOING
            out.append(block(sw, y_back - y_f, k).moved((0.0, 0.5 * (y_f + y_back), 0.0)))
    else:
        sw = min(SIDE_W[0], SIDE_W[1] * d)
        y_s = -SIDE_AT * d / 2.0 + sw / 2.0
        Pt = P[:, ::-1]                                 # (y, x): crossings of a horizontal line give x
        x_out, x_back = edge(Pt, y_s), _span_edge(Pt, y_s - sw / 2.0, y_s + sw / 2.0) - TUCK
        for k in range(1, n):
            x_f = x_out + (n - k) * GOING
            b = block(x_f - x_back, sw, k).moved((0.5 * (x_f + x_back), y_s, 0.0))
            out += [b, b.mirrored_x()]
    return S.merge(out)


def post_mesh(length):
    """A post standing `length` tall on the deck: a round foot with a rolled rim, a slim rod (its top hidden in the
    housing)."""
    prof = np.array([(0.0, 0.0), (FOOT_R - 0.012, 0.0), (FOOT_R, 0.008), (FOOT_R - 0.005, FOOT_H),
                     (POST_R + 0.01, FOOT_H + 0.004), (POST_R, FOOT_H + 0.03), (POST_R, length), (0.0, length)])
    return S.lathe(S.fillet(prof, [0.0, 0.006, 0.004, 0.004, 0.006, 0.01, 0.0, 0.0], 3), seg=32, mat=MATS["metal"])


def panel_mesh(O, i):
    """Panel i: the housing (a rounded panel of its outline, DEPTH deep, its face toward -Y in the LED material
    LED0 + i), the piping bead round its front edge and its posts, in the stage frame."""
    p, H = O["panels"][i], O["height"]
    r, bead_r = rim(p)
    L = outline(p["shape"]) * p["size"]
    hous = S.rounded_panel(L, DEPTH, r_edge=r, n=3, r_back=R_BACK, mat=MATS["housing"])
    hous.assign(LED0 + i, lambda c, nrm: (c[:, 2] > DEPTH - 1e-6) & (nrm[:, 2] > 0.999))
    bead = _closed(S.tube(np.column_stack([L, np.full(len(L), DEPTH - r)]), bead_r, sides=8, closed=True,
                          mat=MATS["trim"]))
    body = S.merge([hous, bead]).rot(90.0, 0.0, 0.0).moved((p["at"][0], p["y"], p["at"][1]))
    return S.merge([body] + [post_mesh(top - H).moved((x, p["y"] - DEPTH / 2.0, H)) for x, top in p["posts"]])


def bars_z(O):
    """The bars' middle height, or None when the stage is too low for them under the bullnose."""
    zc = junction(O) - BAR_DROP - BAR_H / 2.0
    return zc if zc - BAR_H / 2.0 >= RF + 0.02 else None


def _front_line(O, d):
    """The front stretch (where the skirt's outward normal is within FRONT_DEG of -Y) of the plan offset by d (negative:
    out), left to right: (xy (k, 2), arc length (k,))."""
    P = plan(O)
    M = S.mitre_offsets(P)
    out = -M / np.linalg.norm(M, axis=1)[:, None]
    k = np.nonzero(out[:, 1] < -math.cos(math.radians(FRONT_DEG)))[0]
    xy = (P + d * M)[k]
    return xy, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def bar_runs(O):
    """[(x0, x1)] of the bars along the front (each about BAR_LEN, BAR_GAP apart, clear of a front flight)."""
    zc = bars_z(O)
    if zc is None:
        return []
    xy, s = _front_line(O, bulge(O, zc) - (BAR_D / 2.0 - BAR_SINK))
    spans = [(xy[0, 0], xy[-1, 0])]
    if O["steps"] == "front":
        g = 0.5 * min(STEP_W[0], STEP_W[1] * O["size"][0]) + STEP_CLEAR
        spans = [(xy[0, 0], -g), (g, xy[-1, 0])]
    out = []
    for x0, x1 in spans:
        s0, s1 = np.interp([x0, x1], xy[:, 0], s)
        k = max(1, int(round((s1 - s0 + BAR_GAP) / (BAR_LEN + BAR_GAP))))
        ln = (s1 - s0 - (k - 1) * BAR_GAP) / k
        if ln < 0.3:
            continue
        for q in range(k):
            a = s0 + q * (ln + BAR_GAP)
            out.append(tuple(float(v) for v in np.interp([a, a + ln], s, xy[:, 0])))
    return out


def _strip(O, d, x0, x1, zc, section, mat):
    """A strip swept along the front line offset by d from x0 to x1 at the height zc: `section` (a centred rrect),
    its ends rounded over half its height (the section shrinking to a point)."""
    xy, s = _front_line(O, d)
    s0, s1 = np.interp([x0, x1], xy[:, 0], s)
    e = min(0.5 * float(np.ptp(section[:, 1])), 0.45 * (s1 - s0))
    phi = np.linspace(0.0, 0.5 * math.pi, END_SEG + 1)
    cap = e * (1.0 - np.cos(phi))
    mid = np.linspace(s0 + e, s1 - e, max(3, int((s1 - s0) / 0.05)))[1:-1]
    ss = np.concatenate([s0 + cap, mid, (s1 - cap)[::-1]])
    sc = np.concatenate([np.sin(phi), np.ones(len(mid)), np.sin(phi)[::-1]])
    path = np.column_stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1]), np.full(len(ss), zc)])
    return _closed(S.sweep(path, section, scale=sc, up=(0.0, 0.0, 1.0), mat=mat))


def bars_mesh(O):
    """The light bars (None when there are none): each a rounded housing on the skirt under the bullnose and the lens
    strip on its front (material `pixels`)."""
    zc, runs = bars_z(O), bar_runs(O)
    if not runs:
        return None
    d0 = bulge(O, zc) - (BAR_D / 2.0 - BAR_SINK)
    d1 = d0 - (BAR_D / 2.0 + LENS_D / 2.0 - LENS_SINK)
    out = []
    for x0, x1 in runs:
        out.append(_strip(O, d0, x0, x1, zc, S.rrect(BAR_D, BAR_H, BAR_R, 4), MATS["housing"]))
        out.append(_strip(O, d1, x0 + LENS_INSET, x1 - LENS_INSET, zc, S.rrect(LENS_D, LENS_H, LENS_R, 3),
                          MATS["pixels"]))
    return S.merge(out)


def meshes(O):
    """{part: Mesh} of the stage's objects (`<name>_<part>`), every one in the stage frame."""
    out = {"riser": riser_mesh(O), "trim": trim_mesh(O), "steps": steps_mesh(O), "bars": bars_mesh(O)}
    out.update({f"panel{i}": panel_mesh(O, i) for i in range(len(O["panels"]))})
    return {k: m for k, m in out.items() if m is not None}


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=4):
    return [round(float(x), nd) for x in v]


def deck_rect(O):
    """((x, y), (w, d)): the largest rectangle on the flat deck in front of the panels' posts (`use.rest` `deck`)."""
    flat, (w, _d) = flat_outline(O), O["size"]
    backs = [p["y"] - DEPTH / 2.0 - FOOT_R - 0.05 for p in O["panels"]]
    y_back = min(backs) if backs else edge(flat, 0.0) - 0.05
    best = (0.0, (0.0, 0.0), (0.0, 0.0))
    for X in np.linspace(0.1, 0.5 * w - 0.06, 160):
        f, b = _span_edge(flat, -X, X, back=False), _span_edge(flat, -X, X)
        if f is None or b is None:
            continue
        b = min(b, y_back)
        if b > f and 2.0 * X * (b - f) > best[0]:
            best = (2.0 * X * (b - f), (0.0, 0.5 * (f + b)), (2.0 * X, b - f))
    return best[1], best[2]


def marks(O):
    """{name: (x, y)}: where a cast member stands on the deck: `centre`, `left` and `right` (hers, facing the front:
    left is +X)."""
    w, d = O["size"]
    y = -0.08 * d
    return {"centre": (0.0, y), "left": (0.25 * w, y), "right": (-0.25 * w, y)}


def big_heart(O):
    """The index of the biggest heart panel (the first of the biggest), or None."""
    hearts = [i for i, p in enumerate(O["panels"]) if p["shape"] == "heart"]
    return max(hearts, key=lambda i: (O["panels"][i]["size"], -i)) if hearts else None


def text_panel(p):
    """(centre z, width, height) of the largest rectangle centred on the panel's middle line inside its LED face."""
    F, cx = face(p), p["at"][0]
    zs = np.linspace(F[:, 1].min(), F[:, 1].max(), 162)[1:-1]
    half = []
    for z in zs:
        c = _crossings(F[:, ::-1], z)
        spans = [(a, b) for a, b in zip(c[0::2], c[1::2]) if a <= cx <= b]
        half.append(min(cx - spans[0][0], spans[0][1] - cx) if spans else 0.0)
    best = (0.0, 0.0, 0.0, 0.0)
    for a in range(len(zs)):
        m = math.inf
        for b in range(a + 1, len(zs)):
            m = min(m, half[b], half[a])
            area = 2.0 * m * (zs[b] - zs[a])
            if area > best[0]:
                best = (area, 0.5 * (zs[a] + zs[b]), 2.0 * m, zs[b] - zs[a])
    return best[1:]


def card(name, O, rl, parts):
    """The prop card (docs/design.md: Prop card) of a stage with options O, roles rl and meshes `parts` ({part: Mesh}:
    their bounds give the size)."""
    H, (w, d) = O["height"], O["size"]
    lo = np.min([m.V.min(0) for m in parts.values()], axis=0)
    hi = np.max([m.V.max(0) for m in parts.values()], axis=0)
    (rx, ry), (rw, rd) = deck_rect(O)
    use = {"rest": [{"name": "deck", "type": "plane", "center": _r((rx, ry, H)), "normal": [0.0, 0.0, 1.0],
                     "size": _r((rw, rd))}],
           "feet": [{"name": k, "L": [round(x + STANCE, 4), round(y, 4), None],
                     "R": [round(x - STANCE, 4), round(y, 4), None], "floor_z": round(H, 4)}
                    for k, (x, y) in marks(O).items()],
           "look": [{"name": "centre", "point": _r((0.0, 0.0, H))}]}
    k = big_heart(O)
    if k is not None:
        p = O["panels"][k]
        zc, tw, th = text_panel(p)
        use["surface"] = [{"name": "heart", "center": _r((p["at"][0], p["face_y"], zc)), "normal": [0.0, -1.0, 0.0],
                           "up": [0.0, 0.0, 1.0], "size": _r((tw, th)), "object": f"{name}_panel{k}"}]
        use["look"].append({"name": "heart", "point": _r((p["at"][0], p["face_y"], p["at"][1]))})
    n, rise = flight(O)
    info = {"size": [w, d], "height": H, "steps": O["steps"], "risers": n, "rise": round(rise, 4), "going": GOING,
            "pitch": O["pitch"], "wave": O["wave"], "glow": O["glow"], "bars": [_r(r) for r in bar_runs(O)],
            "bars_z": None if bars_z(O) is None else round(bars_z(O), 4),
            "panels": [{"shape": p["shape"], "at": _r(p["at"]), "size": p["size"], "pattern": p["pattern"],
                        "color": p["color"], "alt": p["alt"], "face_y": p["face_y"], "object": f"{name}_panel{i}",
                        "material": f"{name}_led{i}"} for i, p in enumerate(O["panels"])]}
    return {"size": _r(hi - lo), "origin": "floor_center", "front": "-Y", "slots": dict(rl), "use": use,
            "colliders": [], "params": dict(O["start"]), "rates": [{"rate": "speed", "into": "chase", "per": 1.0}],
            "objects": [f"{name}_{k}" for k in parts], "idol_stage": info}
