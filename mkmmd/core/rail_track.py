"""Railroad track maths (docs/design.md: Props, `rail_track`): the plan of a standard-gauge track that lays itself, the
laying front's schedule and the driver expressions that play it. numpy only (no bpy): tested without Blender.

Plan. The track runs from its root along +X, straight (`radius` 0) or along a circular arc bending left (`radius` > 0,
toward +Y) or right (< 0). A station s is metres along the centre line from the start; `Plan.pose(s)` is the point and
the heading there. The length is a whole number of tie spacings (what was asked, rounded). Ties lie every SPACING (2 ft,
2640 to the mile) with their middles at (i + 1/2) SPACING; rails come in RAIL (30 ft) lengths from s = 0, the joints
suspended between two ties and the last rail cut to the length; the ballast bed comes in pieces one SPACING long.

The laying front. The live root property `laid` (metres along the track, keyable) is how far the crews have come; nothing
exists past it. As it passes:

  bed piece k    grows from its start, the toe of its front ramp at `laid`, while `laid` crosses [k, k + 1] SPACING
  tie i          appears when `laid` reaches its near edge (NEAR before its middle), DROP_H over its seat, tipped, and drops
                 onto the bed with a small bounce over the next DROP_W metres of `laid`, landing before the next tie
                 starts
  rail span j    appears once span j - 1 has landed, lifted LIFT (riding on the last rail), its nose LAG behind `laid`
                 (behind every tie still dropping), and slides along the track with the front until its nose reaches its
                 end e_j; it lands with a small bounce as `laid` goes from e_j + LAG to e_j + SPACING. The first rail
                 drops in at its place (nothing lies behind the start to slide in from)
  spikes         of every tie under span j pop up out of the tie once the span has landed, a wave along the rail as
                 `laid` goes from e_j + SPACING to e_j + 2 SPACING

At every whole number of SPACINGs nothing is caught halfway (no tie hangs in the air, no rail is landing, no spike is
popping, no bed piece is growing; a rail may rest lifted, sliding), so `laid` keyed in steps of SPACING (a tie per hit)
or of RAIL (a rail per hit) from 0, each step gliding over part of the time between two hits ([[key]] `step` and `pulse`,
mkmmd.core.lanes.steps), lays the track cleanly. The whole track is in place from `done` = length + 2 SPACING on.

Drivers. Every animated channel is a Blender simple expression of the root's `laid` (the variable `l`), at most MAX_EXPR
characters: `bed_expr`, `tie_expr`, `rail_expr`, `spike_expr`, `front_expr`. Their numpy twins (`bed_scale`, `tie_state`,
`rail_state`, `spike_scale`, `Plan.pose`) are the reference the tests hold them to."""
import math

import numpy as np

SPACING = 0.6096             # m between tie middles: 2 ft
PER_RAIL = 15                # tie spacings to a rail: 30 ft
RAIL = PER_RAIL * SPACING    # 9.144 m
GAUGE = 1.435                # standard gauge, between the inner faces of the rail heads
RAIL_H = 0.1016              # a 56 lb/yd iron T rail: 4 in high
HEAD_W = 0.0572              # its head, 2 1/4 in wide
FOOT_W = 0.1016              # its foot, 4 in wide
TIE_L, TIE_W, TIE_T = 2.591, 0.21, 0.178       # a tie: 8.5 ft long, its hewn top face about 8 in wide, 7 in thick
BED_H = 0.20                 # the ballast bed's top over the ground
BED_SHOULDER = 0.30          # the bed's top runs this far past the tie ends, then slopes 1 : SLOPE to the ground
SLOPE = 1.5
SINK = 0.02                  # a tie lies this far down in the bed
NEAR = 0.10                  # a tie appears when `laid` is this far short of its middle: its near edge
DROP_W = 0.40                # metres of `laid` a tie takes to drop and settle (< SPACING / 2 + NEAR: done before the next)
DROP_H = 0.45                # how high over its seat a tie appears
FALL = 0.6                   # share of DROP_W the fall takes; the bounce takes the rest
BOUNCE = 0.08                # the bounce's height, share of DROP_H
TILT = 6.0                   # degrees a tie is tipped (about the track's direction) when it appears; levelled by the fall
LAG = 0.45                   # a sliding rail's nose stays this far behind `laid` (DROP_W + margin: clear of dropping ties)
LIFT = RAIL_H + 0.004        # a sliding rail rides this far over its seat
LAND_BOUNCE = 0.012          # m: a landing rail's bounce
SPIKE_WAVE = 0.30            # metres of `laid` the spikes' wave takes to run along a rail
SPIKE_W = 0.28               # metres of `laid` one tie's spikes take to pop
SPIKE_POP = 1.25             # their overshoot (scale along z)
MIN_SCALE = 0.0005           # never exactly 0: a singular matrix upsets everything that inverts one
EDGE = 0.001                 # m: a bed piece or a rail shows from this far past its threshold, so at a rest on a whole
                             # number of spacings (a keyed float a hair either side) the next one stays hidden
MAX_EXPR = 255               # Blender keeps a driver expression of at most 255 characters
MIN_RADIUS = 50.0            # a gentle curve: no tighter
FUNCS = frozenset({"clamp", "min", "max", "pow", "abs", "sin", "cos"})      # what the drivers call


class TrackError(ValueError):
    """A track that cannot be laid as asked (a bad length or radius)."""


def _n(v):
    """A number as a driver expression writes it: fixed-point (no exponent), 12 significant digits."""
    s = np.format_float_positional(float(v), precision=12, unique=True, fractional=False, trim="-")
    return "0" if s == "-0" else s


def _clamp01(x):
    return np.clip(np.asarray(x, float), 0.0, 1.0)


class Plan:
    """Where everything of a track `length` metres long (rounded to whole tie spacings) lies, along a circle of `radius`
    (0: straight; > 0 bends left, < 0 right)."""

    def __init__(self, length=30.0, radius=0.0):
        try:
            length, radius = float(length), float(radius)
        except (TypeError, ValueError):
            raise TrackError(f"length = {length!r}, radius = {radius!r}: numbers (metres)") from None
        if not (math.isfinite(length) and length >= SPACING / 2):
            raise TrackError(f"length = {length!r}: at least one tie spacing ({SPACING} m)")
        if not math.isfinite(radius) or (radius != 0.0 and abs(radius) < MIN_RADIUS):
            raise TrackError(f"radius = {radius!r}: 0 (straight) or a gentle curve, at least {MIN_RADIUS:g} m either way")
        self.n = max(1, round(length / SPACING))
        self.length = self.n * SPACING
        self.radius = radius
        self.k = 1.0 / radius if radius else 0.0                       # signed curvature: + bends toward +Y
        if abs(self.k) * self.length > 2.0 * math.pi:
            raise TrackError(f"length {self.length:.2f} m on radius {radius:g} m: more than a full circle")
        self.ties = (np.arange(self.n) + 0.5) * SPACING
        m = -(-self.n // PER_RAIL)
        self.spans = [(j * PER_RAIL * SPACING, min((j + 1) * PER_RAIL, self.n) * SPACING) for j in range(m)]
        self.done = self.length + 2.0 * SPACING

    # ---------------------------------------------------------------------------------------------------- geometry
    def pose(self, s):
        """(x, y, heading) of the centre line at stations s (radians, from +X toward +Y)."""
        s = np.asarray(s, float)
        if self.k == 0.0:
            return s, np.zeros_like(s), np.zeros_like(s)
        a = self.k * s
        return np.sin(a) / self.k, (1.0 - np.cos(a)) / self.k, a

    def point(self, s, y=0.0, z=0.0):
        """Points (n, 3) at stations s, `y` to the left of the centre line (along the normal), `z` up."""
        x0, y0, h = self.pose(np.atleast_1d(np.asarray(s, float)))
        y = np.broadcast_to(np.asarray(y, float), x0.shape)
        return np.stack([x0 - np.sin(h) * y, y0 + np.cos(h) * y, np.broadcast_to(np.asarray(z, float), x0.shape)], 1)

    @property
    def centre(self):
        """The circle's centre (x, y), None when straight."""
        return None if self.k == 0.0 else (0.0, 1.0 / self.k)

    def span_of(self, i):
        """The rail span over tie i."""
        return min(int(i) // PER_RAIL, len(self.spans) - 1)

    def shown_from(self, j):
        """The `laid` from which rail span j shows: the first drops in at its place, the others once the last has landed."""
        return self.spans[0][1] + LAG if j == 0 else self.spans[j - 1][1] + SPACING

    def spike_start(self, i):
        """The `laid` at which tie i's spikes start to pop: its span landed, then a wave along the rail."""
        r, e = self.spans[self.span_of(i)]
        return e + SPACING + (self.ties[i] - r) / (e - r) * SPIKE_WAVE


# ======================================================================================================== numpy twins
def bed_scale(plan, k, laid):
    """(scale along the track, scale across and up) of bed piece k at `laid`: it grows from its start."""
    a = k * SPACING
    laid = np.asarray(laid, float)
    return np.clip((laid - a) / SPACING, MIN_SCALE, 1.0), np.where(laid > a + EDGE, 1.0, MIN_SCALE)


def _drop(q):
    """The tie's height over its seat as a share of DROP_H at progress q: a fall, then a small bounce."""
    q = _clamp01(q)
    return np.maximum(1.0 - (q / FALL) ** 2, 0.0) + BOUNCE * np.maximum(1.0 - ((q - 0.8) / 0.2) ** 2, 0.0)


def _tip(q):
    """The tie's tip as a share of its tilt at progress q: level by the end of the fall, rocking back a little."""
    q = _clamp01(q)
    return np.maximum(1.0 - q / FALL, 0.0) - 0.35 * np.maximum(1.0 - np.abs(q - 0.8) / 0.2, 0.0)


def tie_state(plan, i, laid, tilt=TILT):
    """(height over its seat, tip in radians, shown) of tie i at `laid` (tipped `tilt` degrees when it appears)."""
    a = plan.ties[i] - NEAR
    laid = np.asarray(laid, float)
    q = (laid - a) / DROP_W
    return DROP_H * _drop(q), math.radians(tilt) * _tip(q), laid >= a


def _land(q):
    q = _clamp01(q)
    return LIFT * np.maximum(1.0 - (q / 0.75) ** 2, 0.0) + LAND_BOUNCE * np.maximum(1.0 - ((q - 0.875) / 0.125) ** 2, 0.0)


def rail_state(plan, j, laid):
    """(station of its start, lift over its seat, shown) of rail span j at `laid`: its nose LAG behind the front."""
    r, e = plan.spans[j]
    laid = np.asarray(laid, float)
    start = r + np.minimum(laid - LAG - e, 0.0)
    return start, _land((laid - e - LAG) / (SPACING - LAG)), laid > plan.shown_from(j) + EDGE


def _pop(q):
    q = _clamp01(q)
    rise = 0.55
    return np.maximum(MIN_SCALE, np.minimum(SPIKE_POP * q / rise,
                                            SPIKE_POP - (SPIKE_POP - 1.0) * _clamp01((q - rise) / (1.0 - rise))))


def spike_scale(plan, i, laid):
    """The scale along z of tie i's spikes at `laid` (MIN_SCALE: not yet; up to SPIKE_POP; 1 in place)."""
    return _pop((np.asarray(laid, float) - plan.spike_start(i)) / SPIKE_W)


# ======================================================================================================== drivers
def _q(a, w):
    return f"clamp((l-{_n(a)})/{_n(w)},0,1)"


def _check(exprs):
    for k, v in exprs.items():
        if len(v) > MAX_EXPR or "**" in v:
            raise TrackError(f"the {k} driver is not a Blender simple expression of {MAX_EXPR} characters: {v}")
    return exprs


def bed_expr(plan, k):
    """{"sx": along the track, "syz": across and up} of bed piece k."""
    a = k * SPACING
    return _check({"sx": f"clamp((l-{_n(a)})/{_n(SPACING)},{_n(MIN_SCALE)},1)",
                   "syz": f"1 if l>{_n(a + EDGE)} else {_n(MIN_SCALE)}"})


def tie_expr(plan, i, z0, tilt=TILT):
    """{"z": the tie's height (its seat at z0), "rx": its tip (radians, `tilt` degrees at the start), "s": its scale}."""
    a = plan.ties[i] - NEAR
    q = _q(a, DROP_W)
    z = (f"{_n(z0)}+{_n(DROP_H)}*(max(1-pow({q}/{_n(FALL)},2),0)"
         f"+{_n(BOUNCE)}*max(1-pow(({q}-0.8)/0.2,2),0))")
    rx = f"{_n(math.radians(tilt))}*(max(1-{q}/{_n(FALL)},0)-0.35*max(1-abs({q}-0.8)/0.2,0))"
    return _check({"z": z, "rx": rx, "s": f"1 if l>={_n(a)} else {_n(MIN_SCALE)}"})


def rail_expr(plan, j, z0=0.0):
    """{"slide": the span's location x (straight) or its turn about the circle's centre (radians, curved), "z": its height
    (its seat at z0, lifted while it slides), "s": its scale}. The span's mesh is built at stations [0, e_j - r_j]
    (curved: about the centre), its seat at z = 0."""
    r, e = plan.spans[j]
    off = f"min(l-{_n(e + LAG)},0)"
    slide = f"{_n(r)}+{off}" if plan.k == 0.0 else f"{_n(plan.k)}*({_n(r)}+{off})"
    q = _q(e + LAG, SPACING - LAG)
    z = f"{_n(z0)}+{_n(LIFT)}*max(1-pow({q}/0.75,2),0)+{_n(LAND_BOUNCE)}*max(1-pow(({q}-0.875)/0.125,2),0)"
    return _check({"slide": slide, "z": z, "s": f"1 if l>{_n(plan.shown_from(j) + EDGE)} else {_n(MIN_SCALE)}"})


def spike_expr(plan, i):
    """{"sz": the scale along z of tie i's spikes}."""
    q = _q(plan.spike_start(i), SPIKE_W)
    sz = (f"max({_n(MIN_SCALE)},min({_n(SPIKE_POP)}*{q}/0.55,"
          f"{_n(SPIKE_POP)}-{_n(SPIKE_POP - 1.0)}*clamp(({q}-0.55)/0.45,0,1)))")
    return _check({"sz": sz})


def front_expr(plan):
    """{channel: expression} of the front's empty, on the centre line where `laid` is (held at the ends): "x" and, on a
    curve, "y" and "rz" (its heading). Its height is the builder's."""
    s = f"clamp(l,0,{_n(plan.length)})"
    if plan.k == 0.0:
        return _check({"x": s})
    k = _n(plan.k)
    return _check({"x": f"sin({k}*{s})/{k}", "y": f"(1-cos({k}*{s}))/{k}", "rz": f"{k}*{s}"})


# ======================================================================================================== variety
def tie_variety(plan, seed=1, variants=6, plates=False):
    """Per tie: (variant, along-track shift m, sideways shift m, yaw jitter degrees, tip sign and size): hewn ties do not
    lie in a perfect row. With tie plates under the rails a tie keeps its line across the track (its plates under the
    rails)."""
    rng = np.random.default_rng([int(seed), 7, plan.n])
    n = plan.n
    variant = rng.integers(0, variants, n)
    dx = rng.uniform(-0.018, 0.018, n)
    dy = rng.uniform(-0.03, 0.03, n)
    yaw = rng.uniform(-0.8, 0.8, n)
    if plates:
        dy, yaw = np.zeros(n), np.zeros(n)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0) * rng.uniform(0.6, 1.0, n)
    return [(int(v), float(a), float(b), float(c), float(d)) for v, a, b, c, d in zip(variant, dx, dy, yaw, sign)]
