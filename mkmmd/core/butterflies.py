"""A swarm of butterflies, bpy-free (the library prop `butterflies`, docs/design.md: the library table and its section):
many butterflies round the prop's root, each with its own phase, wing-beat rate, size, style and colourway, flapping by
the flap maths of mkmmd.core.butterfly, in formations blended by keyable weights:

  rings   a mandala: concentric rings in the root's x-z plane facing -Y (a camera in front), ring k at radius
          inner + k * ring_gap and `cup` further back (+y) at the outer ring (a bowl opening toward the camera), each ring
          offset by half a slot from the one inside it, turning in alternate directions (the inner rings faster) by `turns`
  orbit   tilted circular orbits round `orbit_at`, all the same way round, heads along the motion, backs toward the orbit's
          normal and banked toward the centre
  drift   a loose flock wandering inside an ellipsoid round the root on slow sines of the clip clock

and on top of whatever formation `burst` (they fly out and away, gone at 1, back along the same paths when keyed down) and
`bloom` (the rings grow in from the centre on a short spiral, the inner first).

The maths is written once, `swarm(o, A, P, C)`, against a small ops interface `o`: numbers and values with `+ - *` (a
vector times a float scales it, two vectors multiply component by component), `/` by Python numbers only (`o.div` for
anything else: division by zero gives 0, as in Blender), and vec, xyz (the components), sin, cos (of floats or of vectors
component by component), sqrt, abs, min, max, clamp, smoothstep, mix, dot, cross, length, normalize, scale, div. `NumpyOps`
below is the numpy backend (the reference `pose` the tests use); mkmmd/blender/library/props/butterflies_nodes.py is a
geometry-nodes backend that turns the same function into nodes (every op one node), so the tree computes what numpy
computes by construction.

Frames: the prop's frame is a character's (x right seen from the front, -Y toward the camera, z up); a butterfly's own is
mkmmd.core.butterfly's (head -Y, back +Z, its left +X, unit span). Times are clip seconds; `beats` and `turns` are the clocks
the keys stage integrates from the rates `flap` and `spin` (the card's `rates`).

Every number of the layout comes from `seed` (blake2b, the same on every platform) and is rounded to float32, which is what
the tree stores, so the reference sees exactly the numbers the tree does."""
import math

import numpy as np

from . import butterfly as BF

FORMATIONS = ("rings", "orbit", "drift")
PARAMS = {  # live root properties for [[key]] (also starting values in `slots`): (default, unit, doc)
    "rings": (1.0, "", "weight of the ring mandala (the build option `formation` sets the starting weights)"),
    "orbit": (0.0, "", "weight of the orbits round `orbit_at`"),
    "drift": (0.0, "", "weight of the wandering flock"),
    "burst": (0.0, "", "0..1: they fly out and away flapping fast, gone at 1; keyed back down they fly back in"),
    "bloom": (1.0, "", "0..1: the rings grow in from the centre on a short spiral, the inner first (0 hides them all)"),
    "spin": (0.05, "turns/s", "how fast the inner ring turns (a rate: the keys stage integrates it into `turns`)"),
    "radial": (0.0, "", "0..1: every butterfly turned about its head axis by radial x 90 degrees (1: wings radial)"),
    "size": (0.16, "m", "the mean wingspan"),
    "ripple": (0.25, "s", "a hit's extra wing beat runs out from the inner ring to the outer over this long"),
    "orbit_r": (0.35, "m", "the orbits' radius (each butterfly's within 15 % of it)"),
    "burst_reach": (3.0, "m", "how far a whole burst carries them (each 0.7 .. 1.3 of it)"),
    "burst_flaps": (6.0, "beats", "the extra wing beats over a whole burst"),
}
PARAMS.update(BF.FLAP_PARAMS)
RATES = ({"rate": "flap", "into": "beats", "per": 1.0}, {"rate": "spin", "into": "turns", "per": 1.0})
CLOCKS = {"beats": "wing beats since clip second 0 (from `flap`)",
          "turns": "the inner ring's turns since clip second 0 (from `spin`)"}
INPUTS = ("t", "beats", "turns") + tuple(k for k in PARAMS if k not in ("flap", "spin"))   # what `swarm` reads

DEFAULTS = {  # build options (the `slots` of the [[prop]])
    "ring_counts": [8, 16, 24], "inner": 0.5, "ring_gap": 0.28, "cup": 0.3, "heading": 0.0, "formation": "rings",
    "styles": {s: 1.0 for s in BF.STYLES}, "seed": 1, "size_var": 0.1, "phase_spread": 1.0, "orbit_at": [0.0, 0.0, 0.0],
    "orbit_tilt": 30.0, "drift_r": 0.9, "level": False,
}
MAX_COUNT = 500
MAX_RINGS = 12

RING_SPEED = 0.5             # ring k turns (inner / r_k) ** RING_SPEED as fast as the inner ring
RATE_VAR = 0.15              # wing-beat rate factors within 1 +- RATE_VAR (every variety times `phase_spread`)
ORBIT_VAR = 0.15             # orbit radii within orbit_r * (1 +- ORBIT_VAR)
ORBIT_GAIN = 3.0             # an orbit of radius orbit_r goes round this often per turn of the inner ring
ORBIT_KEPLER = 1.5           # and a tighter one faster: rate ~ (r / orbit_r) ** -ORBIT_KEPLER
BANK = 25.0                  # degrees an orbiting butterfly banks toward the centre
DRIFT_AXES = (1.0, 0.5, 0.8)  # the drift ellipsoid's semi-axes, shares of drift_r (flattened in y)
DRIFT_HZ = (0.06, 0.25)      # the frequencies of the drift's sines
DRIFT_CORE = 0.5             # anchors lie within this share of the ellipsoid; the sines wander the rest of the way
DRIFT_HOLLOW = 0.4           # and no nearer the centre than this share of DRIFT_CORE (a figure stands there)
BURST_SPEED = (0.7, 1.3)     # a butterfly's share of burst_reach
BURST_RISE = 0.35            # how much a burst carries them up
BURST_JITTER = 0.35          # the spread of the burst's directions about straight out
BURST_EPS = 0.05             # m: a butterfly at the root still has a way out
BURST_TURN = 0.15            # the heads turn out over this much of the burst
BURST_FADE = (0.85, 1.0)     # and they shrink to nothing over this stretch of it
BLOOM_W = 0.4                # each ring grows in over this much of the bloom
SPIRAL = math.radians(100.0)  # the turn of the bloom's spiral, back against the ring's motion
W_EPS = 1e-3                 # weights adding up to less than this mean rings
MIN_SCALE = 1e-4             # m: a butterfly is never scale 0 (a singular matrix upsets everything that inverts one)
PARTS = ("body", "wing_L", "wing_R")

ATTRS = {  # the per-butterfly constants (point attributes of the swarm's points): name -> kind
    "ring_f": "FLOAT", "theta0": "FLOAT", "rad": "FLOAT", "depth": "FLOAT", "rdir": "FLOAT", "sgn": "FLOAT",
    "blo": "FLOAT", "rate": "FLOAT", "poff": "FLOAT", "span": "FLOAT", "speed": "FLOAT",
    "orho": "FLOAT", "ophi0": "FLOAT", "orate": "FLOAT",
    "oe1": "FLOAT_VECTOR", "oe2": "FLOAT_VECTOR", "onrm": "FLOAT_VECTOR",
    "anchor": "FLOAT_VECTOR", "da1": "FLOAT_VECTOR", "da2": "FLOAT_VECTOR", "dw1": "FLOAT_VECTOR",
    "dw2": "FLOAT_VECTOR", "dp1": "FLOAT_VECTOR", "dp2": "FLOAT_VECTOR", "dv1": "FLOAT_VECTOR", "dv2": "FLOAT_VECTOR",
    "eps": "FLOAT_VECTOR", "jit": "FLOAT_VECTOR",
}


# ===================================================================================================================
# options
# ===================================================================================================================
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _number(slots, key, default, ok, need):
    v = slots.get(key, default)
    if not _is_num(v) or not ok(float(v)):
        raise ValueError(f"butterflies: {key} {need} (got {v!r})")
    return float(v)


LIVE_CHECKS = {  # root property: (accepts, what it must be)
    "rings": (lambda v: v >= 0, "is a weight, a number >= 0"),
    "orbit": (lambda v: v >= 0, "is a weight, a number >= 0"),
    "drift": (lambda v: v >= 0, "is a weight, a number >= 0"),
    "burst": (lambda v: 0 <= v <= 1, "is 0 (the formation) .. 1 (flown away)"),
    "bloom": (lambda v: 0 <= v <= 1, "is 0 (none) .. 1 (every ring grown in)"),
    "radial": (lambda v: 0 <= v <= 1, "is 0 (backs to the camera) .. 1 (wings radial)"),
    "size": (lambda v: v > 0, "is the mean wingspan in m, a number > 0"),
    "flap_amp": (lambda v: v >= 0, "is the full stroke in degrees, a number >= 0"),
    "hit_time": (lambda v: v > 0, "is the seconds a hit's wing beat takes, a number > 0"),
    "ripple": (lambda v: v >= 0, "is seconds, a number >= 0"),
    "orbit_r": (lambda v: v > 0, "is the orbits' radius in m, a number > 0"),
    "burst_reach": (lambda v: v >= 0, "is metres, a number >= 0"),
    "burst_flaps": (lambda v: v >= 0, "is wing beats, a number >= 0"),
}


def counts_for(count):
    """Ring counts for `count` butterflies: K rings (the fewest with 4 K (K + 1) >= count), ring k's share growing as
    k + 1 (as the circumference), largest remainder; a count divisible by 4 gives counts divisible by 4 (a four-fold
    mandala): 48 is [8, 16, 24], 100 is [8, 12, 20, 28, 32]."""
    K = 1
    while 4 * K * (K + 1) < count:
        K += 1
    unit = 4 if count % 4 == 0 and count // 4 >= K else 1
    total = count // unit
    want = [total * (k + 1) / (K * (K + 1) / 2.0) for k in range(K)]
    n = [max(1, int(math.floor(w))) for w in want]
    for k in sorted(range(K), key=lambda k: (n[k] - want[k], k))[:max(0, total - sum(n))]:
        n[k] += 1
    while sum(n) > total:                       # the max(1, ..) above can overshoot a tiny swarm
        n[max(range(K), key=lambda k: n[k])] -= 1
    return [unit * c for c in n]


def options(slots):
    """(opts, P): the build options and the live root properties' starting values from the `slots` of the [[prop]],
    checked; a bad value is a ValueError that says what to change. Keys that are neither (the palette's slots, colour
    roles of the butterflies) are left to their owners."""
    s = slots or {}
    opts = {}
    rc, count = s.get("ring_counts"), s.get("count")
    if rc is not None:
        if not isinstance(rc, (list, tuple)) or not rc or len(rc) > MAX_RINGS or \
                not all(isinstance(c, int) and not isinstance(c, bool) and c >= 1 for c in rc):
            raise ValueError(f"butterflies: ring_counts is a list of 1 .. {MAX_RINGS} whole numbers >= 1, the butterflies "
                             f"of each ring from the inner one out (got {rc!r})")
        rc = [int(c) for c in rc]
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_COUNT:
            raise ValueError(f"butterflies: count is a whole number of butterflies, 1 .. {MAX_COUNT} (got {count!r})")
        if rc is not None and sum(rc) != count:
            raise ValueError(f"butterflies: count {count} disagrees with ring_counts {rc} (sum {sum(rc)}): give one")
    rc = rc if rc is not None else counts_for(count) if count is not None else list(DEFAULTS["ring_counts"])
    if sum(rc) > MAX_COUNT:
        raise ValueError(f"butterflies: ring_counts {rc} make {sum(rc)} butterflies, at most {MAX_COUNT}")
    opts["ring_counts"] = rc
    opts["inner"] = _number(s, "inner", DEFAULTS["inner"], lambda v: v > 0, "is the inner ring's radius in m, > 0")
    opts["ring_gap"] = _number(s, "ring_gap", DEFAULTS["ring_gap"], lambda v: v >= 0, "is m between rings, >= 0")
    opts["cup"] = _number(s, "cup", DEFAULTS["cup"], lambda v: True, "is m (the outer ring that much further back)")
    opts["heading"] = _number(s, "heading", DEFAULTS["heading"], lambda v: True,
                              "is degrees from outward (0) toward the direction of motion (90)")
    formation = s.get("formation", DEFAULTS["formation"])
    if formation not in FORMATIONS:
        raise ValueError(f"butterflies: formation is one of {', '.join(FORMATIONS)} (got {formation!r})")
    opts["formation"] = formation
    styles = s.get("styles", DEFAULTS["styles"])
    if not isinstance(styles, dict) or not styles or set(styles) - set(BF.STYLES) or \
            not all(_is_num(v) and v >= 0 for v in styles.values()) or sum(styles.values()) <= 0:
        raise ValueError(f"butterflies: styles is {{style: share}} of {', '.join(BF.STYLES)}, shares >= 0 and not all 0 "
                         f"(got {styles!r})")
    opts["styles"] = {k: float(v) for k, v in styles.items() if v > 0}
    seed = s.get("seed", DEFAULTS["seed"])
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"butterflies: seed is a whole number (got {seed!r})")
    opts["seed"] = seed
    opts["size_var"] = _number(s, "size_var", DEFAULTS["size_var"], lambda v: 0 <= v < 1,
                               "is the spread of the spans, 0 .. below 1 (0.2: within 20 % of the mean)")
    opts["phase_spread"] = _number(s, "phase_spread", DEFAULTS["phase_spread"], lambda v: 0 <= v <= 1,
                                   "is 0 (all flap in unison) .. 1 (phases all over the beat)")
    at = s.get("orbit_at", DEFAULTS["orbit_at"])
    if not isinstance(at, (list, tuple)) or len(at) != 3 or not all(_is_num(v) for v in at):
        raise ValueError(f"butterflies: orbit_at is [x, y, z], metres from the root (got {at!r})")
    opts["orbit_at"] = [float(v) for v in at]
    opts["orbit_tilt"] = _number(s, "orbit_tilt", DEFAULTS["orbit_tilt"], lambda v: 0 <= v <= 90,
                                 "is degrees, 0 .. 90 (the orbits' inclinations lie within +- it)")
    opts["drift_r"] = _number(s, "drift_r", DEFAULTS["drift_r"], lambda v: v > 0,
                              "is the drift ellipsoid's radius in m, > 0")
    level = s.get("level", DEFAULTS["level"])
    if not isinstance(level, bool):
        raise ValueError(f"butterflies: level is true or false (got {level!r})")
    opts["level"] = level
    P = {}
    for k, (default, unit, _doc) in PARAMS.items():
        if k in FORMATIONS:
            default = 1.0 if k == formation else 0.0
        ok, need = LIVE_CHECKS.get(k, (lambda v: True, f"is a number{' (' + unit + ')' if unit else ''}"))
        P[k] = _number(s, k, default, ok, need)
    return opts, P


# ===================================================================================================================
# layout
# ===================================================================================================================
def _u(seed, *key):
    return BF.unit(seed, "swarm", *key)


def _sphere(u, v):
    """A unit vector from two numbers in [0, 1): uniform over the sphere."""
    z = 2.0 * u - 1.0
    r = math.sqrt(max(0.0, 1.0 - z * z))
    a = 2.0 * math.pi * v
    return np.array([r * math.cos(a), r * math.sin(a), z])


# steps of three Kronecker sequences frac(0.5 + i a): each spreads evenly round its circle on its own (the golden ratio's
# gaps are the evenest; sqrt 2 and sqrt 3 nearly so) and, the three being independent, they spread evenly together
EVEN = (2.0 / (1.0 + math.sqrt(5.0)), math.sqrt(2.0) - 1.0, math.sqrt(3.0) - 1.0)


class Layout:
    """The seeded per-butterfly constants of a swarm from its build options: `N` butterflies on `K` rings (`counts`), ring
    `ring` (N,) and slot `slot` (N,), `style` (N names), colourway `way` (N,), `combos` (the (style, way) pairs in use,
    in order) and `combo` (N,) indices into them, `attrs` {name: (N,) or (N, 3)} (ATTRS) and the build constants `consts`.

    The mandala is `sym`-fold symmetric (the greatest common divisor of the ring counts): one sector's styles and
    colourways, drawn in proportion to the mix, repeat round every ring; phases, rates and spans are each butterfly's own."""

    def __init__(self, opts):
        counts, seed = list(opts["ring_counts"]), opts["seed"]
        self.counts, self.K, self.N = counts, len(counts), sum(counts)
        K, N = self.K, self.N
        self.ring = np.repeat(np.arange(K), counts)
        self.slot = np.concatenate([np.arange(c) for c in counts])
        self.sym = int(np.gcd.reduce(np.array(counts)))
        self._styles(opts)
        A = {}
        r = opts["inner"] + opts["ring_gap"] * self.ring
        r_in, r_out = opts["inner"], opts["inner"] + opts["ring_gap"] * (K - 1)
        k, n = self.ring.astype(float), np.repeat(np.array(counts, float), counts)
        A["ring_f"] = k / (K - 1) if K > 1 else np.zeros(N)
        A["theta0"] = 0.5 * math.pi + 2.0 * math.pi * (self.slot + 0.5 * (self.ring % 2)) / n
        A["rad"] = r
        A["depth"] = opts["cup"] * (r * r - r_in * r_in) / (r_out * r_out - r_in * r_in) if r_out > r_in else np.zeros(N)
        A["sgn"] = np.where(self.ring % 2 == 0, 1.0, -1.0)
        A["rdir"] = 2.0 * math.pi * A["sgn"] * (r_in / r) ** RING_SPEED
        A["blo"] = (1.0 - BLOOM_W) * A["ring_f"]
        spread = opts["phase_spread"]
        rate, poff, span, speed = (np.zeros(N) for _ in range(4))
        for i in range(N):
            st = self.style[i]
            rate[i] = 1.0 + (BF.RATE_FACTOR[st] * (1.0 + RATE_VAR * (2.0 * _u(seed, "rate", i) - 1.0)) - 1.0) * spread
            poff[i] = spread * _u(seed, "phase", i)
            span[i] = BF.SPAN_FACTOR[st] * (1.0 + opts["size_var"] * (2.0 * _u(seed, "span", i) - 1.0))
            speed[i] = BURST_SPEED[0] + (BURST_SPEED[1] - BURST_SPEED[0]) * _u(seed, "speed", i)
        A.update(rate=rate, poff=poff, span=span, speed=speed)
        A.update(self._orbits(opts, seed))
        A.update(self._drift(opts, seed))
        A["eps"] = np.array([BURST_EPS * _sphere(_u(seed, "ez", i), _u(seed, "ea", i)) for i in range(N)])
        A["jit"] = np.array([BURST_JITTER * (0.5 + 0.5 * _u(seed, "jr", i)) * _sphere(_u(seed, "jz", i), _u(seed, "ja", i))
                             for i in range(N)])
        self.attrs = {k: np.asarray(v, np.float32).astype(float) for k, v in A.items()}
        h = math.radians(opts["heading"])
        self.consts = {"hc": math.cos(h), "hs": math.sin(h), "orbit_at": tuple(opts["orbit_at"]),
                       "bank_c": math.cos(math.radians(BANK)), "bank_s": math.sin(math.radians(BANK))}

    def _styles(self, opts):
        """Styles and colourways: one sector's slots (each ring's count / sym) take the styles in proportion to the mix
        (largest remainder) in a seeded order and a seeded colourway each; every sector repeats them."""
        seed, mix = opts["seed"], opts["styles"]
        per = [c // self.sym for c in self.counts]
        S = sum(per)
        total = sum(mix.values())
        want = {st: S * v / total for st, v in mix.items()}
        n = {st: int(math.floor(w)) for st, w in want.items()}
        for st in sorted(want, key=lambda st: (n[st] - want[st], BF.STYLES.index(st)))[:S - sum(n.values())]:
            n[st] += 1
        kinds = [st for st in BF.STYLES for _ in range(n.get(st, 0))]
        order = sorted(range(S), key=lambda i: _u(seed, "style", i))
        tstyle = [kinds[order[i]] for i in range(S)]
        tway = [int(_u(seed, "way", i) * len(BF.WAYS[st])) for i, st in enumerate(tstyle)]
        off = np.concatenate([[0], np.cumsum(per)[:-1]]).astype(int)
        tmpl = off[self.ring] + self.slot % np.array(per)[self.ring]
        self.style = [tstyle[j] for j in tmpl]
        self.way = np.array([tway[j] for j in tmpl])
        self.combos = sorted({(st, int(w)) for st, w in zip(self.style, self.way)},
                             key=lambda c: (BF.STYLES.index(c[0]), c[1]))
        index = {c: i for i, c in enumerate(self.combos)}
        self.combo = np.array([index[(st, int(w))] for st, w in zip(self.style, self.way)])

    def _orbits(self, opts, seed):
        """Per butterfly an orbit: its node, start and inclination spread evenly (EVEN over a seeded order), its radius
        share and its rate (a tighter orbit goes round faster)."""
        N = self.N
        order = sorted(range(N), key=lambda i: _u(seed, "orbit", i))
        alpha = np.array(EVEN)
        off = np.array([_u(seed, "o0"), _u(seed, "o1"), _u(seed, "o2")])
        tilt = math.radians(opts["orbit_tilt"])
        out = {k: np.zeros(N) for k in ("orho", "ophi0", "orate")}
        out.update({k: np.zeros((N, 3)) for k in ("oe1", "oe2", "onrm")})
        for q, i in enumerate(order):
            node, start, lean = np.mod(0.5 + q * alpha + off, 1.0)
            om, ph, inc = 2.0 * math.pi * node, 2.0 * math.pi * start, tilt * (2.0 * lean - 1.0)
            rho = 1.0 + ORBIT_VAR * (2.0 * _u(seed, "rho", i) - 1.0)
            e1 = np.array([math.cos(om), math.sin(om), 0.0])                # the line of nodes
            e2 = np.array([-math.sin(om) * math.cos(inc), math.cos(om) * math.cos(inc), math.sin(inc)])
            out["oe1"][i], out["oe2"][i], out["onrm"][i] = e1, e2, np.cross(e1, e2)
            out["orho"][i], out["ophi0"][i] = rho, ph
            out["orate"][i] = 2.0 * math.pi * ORBIT_GAIN * rho ** -ORBIT_KEPLER
        return out

    def _drift(self, opts, seed):
        """Per butterfly an anchor in the ellipsoid (a hollow ball of DRIFT_CORE, evenly spread) and two sines per axis
        whose amplitudes add up to what is left of it, so no butterfly ever leaves the ellipsoid."""
        N = self.N
        ax = opts["drift_r"] * np.array(DRIFT_AXES)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        order = sorted(range(N), key=lambda i: _u(seed, "drift", i))
        wander = (1.0 - DRIFT_CORE) / math.sqrt(3.0)          # per axis: the sines' amplitudes add up to this share
        out = {k: np.zeros((N, 3)) for k in ("anchor", "da1", "da2", "dw1", "dw2", "dp1", "dp2", "dv1", "dv2")}
        h3 = DRIFT_HOLLOW ** 3
        for q, i in enumerate(order):
            z = 1.0 - 2.0 * (q + 0.5) / N
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            a = golden * q + 2.0 * math.pi * _u(seed, "dturn")
            d = np.array([rr * math.cos(a), rr * math.sin(a), z])
            rad = DRIFT_CORE * (h3 + (1.0 - h3) * _u(seed, "dr", i)) ** (1.0 / 3.0)
            out["anchor"][i] = ax * d * rad
            for j in range(3):
                f = 0.35 + 0.3 * _u(seed, "dsplit", i, j)
                for m, share in ((1, f), (2, 1.0 - f)):
                    hz = DRIFT_HZ[0] + (DRIFT_HZ[1] - DRIFT_HZ[0]) * _u(seed, "dhz", i, j, m)
                    out[f"da{m}"][i, j] = ax[j] * wander * share
                    out[f"dw{m}"][i, j] = 2.0 * math.pi * hz
                    out[f"dp{m}"][i, j] = 2.0 * math.pi * _u(seed, "dph", i, j, m)
        for m in (1, 2):
            out[f"dv{m}"] = out[f"da{m}"] * out[f"dw{m}"]
        return out

    def ops_attrs(self):
        """The attributes as NumpyOps values: floats (N,), vectors (3, N)."""
        return {k: (v.T if v.ndim == 2 else v) for k, v in self.attrs.items()}

    def rest(self):
        """(N, 3): the butterflies on their rings at turns 0 (where the swarm's points sit)."""
        A = self.attrs
        return np.stack([A["rad"] * np.cos(A["theta0"]), A["depth"], A["rad"] * np.sin(A["theta0"])], 1)


def layout(opts):
    return Layout(opts)


# ===================================================================================================================
# the maths, written once
# ===================================================================================================================
def _rings(o, A, P, C):
    th = A["theta0"] + A["rdir"] * P["turns"]
    c, s = o.cos(th), o.sin(th)
    pos = o.vec(A["rad"] * c, A["depth"], A["rad"] * s)
    out = o.vec(c, 0.0, s)
    along = o.vec(-1.0 * A["sgn"] * s, 0.0, A["sgn"] * c)          # the ring's direction of motion
    head = C["hc"] * out + C["hs"] * along
    a = P["radial"] * (0.5 * math.pi)                               # turned about the head axis: radial x 90 degrees
    b0 = o.vec(0.0, -1.0, 0.0)
    back = o.cos(a) * b0 + o.sin(a) * o.cross(head, b0)
    return pos, head, back


def _orbit(o, A, P, C):
    ph = A["ophi0"] + A["orate"] * P["turns"]
    c, s = o.cos(ph), o.sin(ph)
    rad = c * A["oe1"] + s * A["oe2"]                               # out from the centre, in the orbit's plane
    pos = o.vec(*C["orbit_at"]) + (P["orbit_r"] * A["orho"]) * rad
    head = c * A["oe2"] - s * A["oe1"]
    back = C["bank_c"] * A["onrm"] - C["bank_s"] * rad              # banked toward the centre
    return pos, head, back


def _drift(o, A, P, C):
    a1 = A["dw1"] * P["t"] + A["dp1"]
    a2 = A["dw2"] * P["t"] + A["dp2"]
    pos = A["anchor"] + A["da1"] * o.sin(a1) + A["da2"] * o.sin(a2)
    vel = A["dv1"] * o.cos(a1) + A["dv2"] * o.cos(a2)
    return pos, o.normalize(vel), o.vec(0.0, 0.0, 1.0)


def _blend(o, F, P):
    """The formations by their weights (each >= 0, normalised; a sum near 0 means rings): position, head, back."""
    w = {f: o.max(P[f], 0.0) for f in FORMATIONS}
    total = w["rings"] + w["orbit"] + w["drift"]
    lack = o.max(W_EPS - total, 0.0)
    inv = o.div(1.0, total + lack)
    k = {"rings": (w["rings"] + lack) * inv, "orbit": w["orbit"] * inv, "drift": w["drift"] * inv}
    return tuple(k["rings"] * F["rings"][j] + k["orbit"] * F["orbit"][j] + k["drift"] * F["drift"][j] for j in range(3))


def _square(o, head, back):
    """Head made unit, back made perpendicular to it and unit."""
    h = o.normalize(head)
    return h, o.normalize(back - o.dot(back, h) * h)


def _bloom(o, A, P, pos, head, back):
    """Ring k grows in over its own window of `bloom`, coming out from the root on a short spiral about the mandala's
    axis (Y), back against its ring's motion. Returns (grow, pos, head, back)."""
    grow = o.smoothstep(A["blo"], A["blo"] + BLOOM_W, P["bloom"])
    beta = (-SPIRAL * A["sgn"]) * (1.0 - grow)
    c, s = o.cos(beta), o.sin(beta)

    def turn(v):
        x, y, z = o.xyz(v)
        return o.vec(c * x - s * z, y, s * x + c * z)
    return grow, grow * turn(pos), turn(head), turn(back)


def _burst(o, A, P, pos, head, back):
    """Out along o_i = normalize(normalize(p + eps_i) + jitter_i + rise Z) by burst_reach v_i s^2, the head turning to
    o_i (the back carried round by the same turn) over the first BURST_TURN of s. Returns (s, fade, pos, head, back)."""
    s = o.clamp(P["burst"], 0.0, 1.0)
    way = o.normalize(o.normalize(pos + A["eps"]) + A["jit"] + o.vec(0.0, 0.0, BURST_RISE))
    pos = pos + (P["burst_reach"] * A["speed"] * s * s) * way
    c = o.dot(head, way)
    k = o.cross(head, way)
    back_out = c * back + o.cross(k, back) + o.div(o.dot(k, back), 1.0 + c) * k      # the turn head -> way (Rodrigues)
    lam = o.smoothstep(0.0, BURST_TURN, s)
    fade = 1.0 - o.smoothstep(BURST_FADE[0], BURST_FADE[1], s)
    return s, fade, pos, o.mix(head, way, lam), o.mix(back, back_out, lam)


def _phase(o, A, P, s):
    """Each butterfly's wing-beat phase (cycles): its rate on the beats clock, the prop's phase, its own offset, the extra
    beat of the last hit (rippling out ring by ring) and the burst's extra beats."""
    u = o.div(P["t"] - P["hit"] - P["ripple"] * A["ring_f"], P["hit_time"])
    return P["beats"] * A["rate"] + P["phase"] + A["poff"] + BF.hit_flap(u, o) + P["burst_flaps"] * s


def swarm(o, A, P, C):
    """The swarm at one instant, against the ops backend `o`: A the per-butterfly attributes by name (ATTRS), P the
    inputs by name (INPUTS: the clip second `t`, the clocks `beats` and `turns`, the live root properties), C the build
    constants (`Layout.consts`). Returns {"pos", "head", "back"} (vectors in the prop's frame; the head exact, the back to
    be made perpendicular), "span" (the wingspan in m, never below MIN_SCALE), "phase" (cycles), "angle" (the wings'
    lift, radians) and "bob" (the body's rise along its back, a share of the span)."""
    F = {"rings": _rings(o, A, P, C), "orbit": _orbit(o, A, P, C), "drift": _drift(o, A, P, C)}
    pos, head, back = _blend(o, F, P)
    head, back = _square(o, head, back)
    grow, pos, head, back = _bloom(o, A, P, pos, head, back)
    s, fade, pos, head, back = _burst(o, A, P, pos, head, back)
    phase = _phase(o, A, P, s)
    span = o.max(P["size"] * A["span"] * grow * fade, MIN_SCALE)
    return {"pos": pos, "head": head, "back": back, "span": span, "phase": phase,
            "angle": BF.wing_angle(phase, P["flap_amp"], P["flap_mid"], o), "bob": BF.body_bob(phase, o)}


# ===================================================================================================================
# numpy: the reference
# ===================================================================================================================
class NumpyOps:
    """The ops on numpy: a float is an array or a number, a vector an array of shape (3, ...) (components first, so a
    float scales it by broadcasting). As in Blender, division by zero gives 0, the square root of a negative number 0 and
    the normal of a zero vector the zero vector; smoothstep is Map Range's (a factor clamped to 0..1)."""

    @staticmethod
    def vec(x, y, z):
        if all(isinstance(c, (int, float)) for c in (x, y, z)):
            return np.array([[x], [y], [z]], float)
        return np.stack(np.broadcast_arrays(*(np.asarray(c, float) for c in (x, y, z))))

    @staticmethod
    def xyz(v):
        return v[0], v[1], v[2]

    sin, cos, abs = staticmethod(np.sin), staticmethod(np.cos), staticmethod(np.abs)
    min, max = staticmethod(np.minimum), staticmethod(np.maximum)

    @staticmethod
    def sqrt(x):
        return np.sqrt(np.maximum(x, 0.0))

    @staticmethod
    def div(a, b):
        b = np.asarray(b, float)
        ok = b != 0.0
        return np.where(ok, np.asarray(a, float) / np.where(ok, b, 1.0), 0.0)

    @staticmethod
    def clamp(x, lo, hi):
        return np.minimum(np.maximum(x, lo), hi)

    def smoothstep(self, a, b, x):
        f = np.clip(self.div(np.asarray(x, float) - a, np.asarray(b, float) - a), 0.0, 1.0)
        return f * f * (3.0 - 2.0 * f)

    @staticmethod
    def mix(a, b, f):
        return a + (b - a) * f

    @staticmethod
    def dot(a, b):
        return np.sum(a * b, axis=0)

    def cross(self, a, b):
        return self.vec(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def length(self, v):
        return np.sqrt(self.dot(v, v))

    def normalize(self, v):
        return v * self.div(1.0, self.length(v))

    @staticmethod
    def scale(v, f):
        return v * f


NP = NumpyOps()


def frame(head, back):
    """(N, 3, 3) rotations whose columns are a butterfly's own X, Y, Z in the prop's frame: Y along -head exactly, Z the
    back made perpendicular to it, X = Y x Z (geometry nodes' Axes to Rotation with primary Y, secondary Z). head, back:
    (3, N)."""
    Y = NP.normalize(-head)
    X = NP.normalize(NP.cross(Y, back))
    Z = NP.cross(X, Y)
    return np.stack([X.T, Y.T, Z.T], axis=-1)


def _rot_y(a):
    """(N, 4, 4) rotations about Y by angles a (N,)."""
    c, s = np.cos(a), np.sin(a)
    M = np.zeros((len(a), 4, 4))
    M[:, 0, 0], M[:, 0, 2], M[:, 1, 1], M[:, 2, 0], M[:, 2, 2], M[:, 3, 3] = c, s, 1.0, -s, c, 1.0
    return M


def _translate(v):
    M = np.tile(np.eye(4), (len(v), 1, 1))
    M[:, :3, 3] = v
    return M


def inputs(P, t, beats, turns):
    """The inputs `swarm` reads (INPUTS) from the live root properties P, the clip second t and the clocks."""
    out = {k: float(P[k]) for k in INPUTS if k not in ("t", "beats", "turns")}
    out.update(t=float(t), beats=float(beats), turns=float(turns))
    return out


def pose(L, X):
    """The swarm at one instant in numpy, as the tree has it: L a Layout, X the inputs (`inputs`). Returns the `swarm`
    values as arrays (vectors (N, 3)) plus "rot" (N, 3, 3) and each part's instance matrix (N, 4, 4) in the prop's frame:
    body = T(pos) R S(span) T(0, 0, bob), wing_L = T(pos) R S(span) T(hinge_L + (0, 0, bob)) R_Y(-angle), wing_R the
    same with hinge_R and +angle."""
    R = swarm(NP, L.ops_attrs(), X, L.consts)
    N = L.N
    out = {k: np.broadcast_to(np.asarray(v, float), (3, N)).T.copy() if k in ("pos", "head", "back") else
           np.broadcast_to(np.asarray(v, float), (N,)).copy() for k, v in R.items()}
    rot = frame(R["head"], R["back"])
    M = np.tile(np.eye(4), (N, 1, 1))
    M[:, :3, :3] = rot * out["span"][:, None, None]
    M[:, :3, 3] = out["pos"]
    bob = np.zeros((N, 3))
    bob[:, 2] = out["bob"]
    out["rot"] = rot
    out["body"] = M @ _translate(bob)
    for part, side, sign in (("wing_L", "L", -1.0), ("wing_R", "R", 1.0)):
        out[part] = M @ _translate(bob + np.array(BF.hinge(side))) @ _rot_y(sign * out["angle"])
    return out


def bounds(opts, P, L):
    """(lo, hi): the box round every formation the swarm can take (rings, orbits at the starting `orbit_r`, drift),
    padded by half the widest span at the starting `size`; a burst flies out of it."""
    pad = 0.5 * P["size"] * float(L.attrs["span"].max())
    r = opts["inner"] + opts["ring_gap"] * (L.K - 1)
    boxes = [([-r, min(0.0, opts["cup"]), -r], [r, max(0.0, opts["cup"]), r])]
    c, ro = np.array(opts["orbit_at"]), P["orbit_r"] * (1.0 + ORBIT_VAR)
    rz = ro * math.sin(math.radians(opts["orbit_tilt"]))
    boxes.append((c - [ro, ro, rz], c + [ro, ro, rz]))
    ax = opts["drift_r"] * np.array(DRIFT_AXES)
    boxes.append((-ax, ax))
    lo = np.min([b[0] for b in boxes], axis=0) - pad
    hi = np.max([b[1] for b in boxes], axis=0) + pad
    return [float(v) for v in lo], [float(v) for v in hi]
