"""Confetti, bpy-free (the library prop `confetti`, docs/design.md: the library table): paper squares, hearts, stars and
curly streamers fired from the root in volleys, flying out in a cone, slowed by the air, tumbling and fluttering as they
rain down, then shrinking away. Driver expressions on the frame do it (mkmmd/blender/library/props/confetti.py lays them on
the pieces), so a freeze (mkmmd.core.freeze), which holds the scene on a frame, holds them mid-air.

A piece of volley k is a = clip second - start<k> - delay seconds old (each volley reads its own root property
`start<k>`; the keys stage deals the keys of `start` round the volleys: docs/design.md, Keys). Its flight is the closed
form of linear air drag: with tau = drag * kd and g = 1 - exp(-max(a, 0) / tau) (0 at the mouth, 1 once the air has it),

    position = reach * v * g + (0, 0, -fall * kf * (max(a, 0) - tau * g)) + flutter * g * sin(w * sway * a + phase) * d

in the prop's frame: v is the piece's direction (in a cone of half-angle `cone` about `aim`) times its share of the push,
kf its share of the falling speed, d a horizontal direction whose length is its share of the flutter, w 2 pi times its
share of the sway's rate. It tumbles: Euler angles rot0 + spin * omega * max(a, 0), omega 2 pi times a signed share per
axis. Its scale is size * share * amount, popped in from tiny at the mouth over POP seconds, shrinking away from FADE of its
life to its end, and as its centre comes down to `floor` (over a band FLOOR_SOFT times its own radius, so no part of it
ever passes the floor). A streamer (a paper strip curled into a helix along its local Z, shot out along its flight) also
unrolls: its Z scale goes from UNROLL to 1 with g. Every per-piece number comes from the `seed` (the same on every
machine) and is rounded as the expressions print it, so `pose` is what the drivers do."""
import collections
import functools
import math

import numpy as np

from . import burst as BU
from . import hearts as HR
from . import shell as CS

KINDS = ("square", "ribbon", "heart", "star")
PARAMS = {  # live root property: (default, unit, doc); [[key]] keys them, the `slots` give their starting values
    "start": (0.0, "s", "the clip second the first volley goes off; its keys are dealt round the volleys (start0 ..)"),
    "reach": (3.8, "m", "how far the push carries the fastest piece before the air has it (gravity aside)"),
    "drag": (0.5, "s", "how soon the air slows a piece: the time constant of its drag"),
    "fall": (0.7, "m/s", "how fast a piece falls once the air has it"),
    "flutter": (0.14, "m", "how far a piece sways from side to side as it falls"),
    "sway": (0.8, "Hz", "how often a piece sways"),
    "spin": (1.2, "rev/s", "how fast a piece tumbles"),
    "size": (0.05, "m", "the largest piece's width"),
    "life": (6.5, "s", "how long a piece lives, out, up and down, before it shrinks away"),
    "floor": (-1e4, "m", "the floor's height in the prop's frame: pieces shrink away as they come down to it (-1e4: none)"),
    "amount": (1.0, "", "overall scale of every piece (0 hides them)"),
}
MIX = {"square": 0.55, "ribbon": 0.15, "heart": 0.15, "star": 0.15}    # default share of each kind
COLORS = ("love", "gold", "foam", "iris", "rose", "pine", "text")
AIM = (0.0, -0.25, 1.0)          # up, a little toward the camera in front (-Y)
CONE = 24.0                      # degrees: half-angle of the cone the pieces fly out in
RIBBON = 0.5                     # m: a streamer's length at the starting size
GLOW = 0.25                      # emission strength: the colours read at night
COUNT, VOLLEYS, SEED = 60, 3, 1
MAX_COUNT, MAX_VOLLEYS, MAX_COLORS = 200, 4, 16
MAX_RIBBON = 80.0                # a streamer is at most this many times `size` long
NEVER = 1e4                      # the start of a volley that never goes off
POP = 0.06                       # seconds a piece takes to pop in at the mouth
FADE = 0.8                       # share of its life after which a piece shrinks away
FADE_EPS = 0.001                 # s: the fade ends that much after `life` (a keyed life of 0 never divides by zero)
DELAY = 0.12                     # seconds over which a volley's pieces leave the mouth
UNROLL = 0.2                     # a streamer's length at the mouth, as a share of its full length
MIN_DRAG = 0.01                  # s: the drag the maths never goes below (a keyed 0 would divide by zero)
FLOOR_SOFT = 1.15                # a piece starts shrinking this many of its radii above the floor
FLOOR_EPS = 0.001                # m: the floor band's least width
THICK = 0.012                    # paper thickness, as a share of a piece's width
CURL = 1.6                       # radius of a square's gentle curl, in widths
RIB_W, RIB_R, RIB_PITCH, RIB_STEPS = 0.3, 0.3, 1.2, 16    # a streamer: strip width, curl radius, pitch, points per turn
MIN_SCALE = HR.MIN_SCALE
HELPERS = ("a", "g", "z")        # custom properties of each piece, driven: its age, the drag's ease, its height
FUNCS = frozenset({"exp", "max", "sin", "smoothstep"})
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))
_unit = HR._unit

Options = collections.namedtuple("Options", "count volleys seed mix colors aim cone ribbon glow params")


def _num(slots, key, default, lo=-math.inf, hi=math.inf, whole=False, positive=False, unit=""):
    """The option `key` of the slots (or `default`): a finite number in lo .. hi (> 0 with `positive`, whole with
    `whole`); anything else is a ValueError that says what to change."""
    v = slots.get(key, default)
    ok = (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi
          and (v > 0 or not positive) and (not whole or float(v) == int(v)))
    if not ok:
        rng = (" > 0" if positive else f" {lo:g} .. {hi:g}" if math.isfinite(lo) and math.isfinite(hi) else
               f" >= {lo:g}" if math.isfinite(lo) else "")
        raise ValueError(f"confetti: {key} is {'a whole number' if whole else 'a number'}{rng}"
                         f"{' (' + unit + ')' if unit else ''} (got {v!r})")
    return int(v) if whole else float(v)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def options(slots):
    """The `Options` of a [[prop]]'s `slots` (count, volleys, seed, mix {kind: share}, colors [spec, ...], aim (3,),
    cone, ribbon, glow, params {live root property: starting value}), checked: a bad value is a ValueError that says what
    to change. The colour specs are resolved (and checked) by the builder, from the palette."""
    slots = slots or {}
    count = _num(slots, "count", COUNT, 1, MAX_COUNT, whole=True)
    volleys = _num(slots, "volleys", VOLLEYS, 1, MAX_VOLLEYS, whole=True)
    seed = _num(slots, "seed", SEED, whole=True)
    mix = slots.get("mix", MIX)
    if not isinstance(mix, dict) or not mix or set(mix) - set(KINDS) \
            or not all(_finite(v) and v >= 0 for v in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError(f"confetti: mix is {{kind = share}} of {', '.join(KINDS)}, shares >= 0 and not all 0 "
                         f"(got {mix!r})")
    colors = slots.get("colors", list(COLORS))
    if not isinstance(colors, (list, tuple)) or not 1 <= len(colors) <= MAX_COLORS or not all(
            (isinstance(c, str) and c) or (isinstance(c, (list, tuple)) and len(c) == 3) for c in colors):
        raise ValueError(f"confetti: colors is a list of 1 .. {MAX_COLORS} colours, each a palette slot, '#hex' or an "
                         f"'a:b:t' mix (got {colors!r})")
    aim = slots.get("aim", list(AIM))
    if not isinstance(aim, (list, tuple)) or len(aim) != 3 or not all(_finite(c) for c in aim) or math.hypot(*aim) < 1e-6:
        raise ValueError(f"confetti: aim is the direction [x, y, z] the volleys fly out in, not zero (got {aim!r})")
    cone = _num(slots, "cone", CONE, 0.0, 180.0, unit="degrees")
    glow = _num(slots, "glow", GLOW, 0.0, 5.0)
    P = {}
    for k, (default, unit, _) in PARAMS.items():
        P[k] = _num(slots, k, default, lo=0.0 if k in ("fall", "flutter", "sway", "amount") else -math.inf,
                    positive=k in ("reach", "drag", "size", "life"), unit=unit)
    ribbon = _num(slots, "ribbon", RIBBON, positive=True, unit="m")
    if ribbon > MAX_RIBBON * P["size"]:
        raise ValueError(f"confetti: ribbon is at most {MAX_RIBBON:g} times size (a {ribbon:g} m streamer at size "
                         f"{P['size']:g} m is {ribbon / P['size']:.0f}): make it shorter or size larger")
    return Options(count, volleys, seed, {k: float(v) for k, v in mix.items()}, list(colors),
                   tuple(float(c) for c in aim), cone, ribbon, glow, P)


def root_values(O):
    """{root property: starting value} of everything the drivers read but the clock (f0, fps): the live parameters and
    each volley's start (`start0` the `start` slot, the others NEVER: unkeyed, only the first volley goes off)."""
    P = dict(O.params)
    for k in range(O.volleys):
        P[f"start{k}"] = P["start"] if k == 0 else NEVER
    return P


def deals(volleys):
    """The card's `deals`: the keys of `start` dealt round the volleys' starts (the keys stage, docs/design.md: Keys)."""
    return [{"from": "start", "into": [f"start{k}" for k in range(volleys)]}]


def length(O):
    """A streamer's length in widths (its mesh's units): `ribbon` at the starting `size`."""
    return O.ribbon / O.params["size"]


# ===================================================================================================================
# the pieces
# ===================================================================================================================
def _flat(outline, rings=2):
    """A die-cut of paper: the outline (x, z) as a thin closed plate THICK thick, both edges rolled, centred on the origin,
    its face looking along -Y."""
    m = CS.rounded_panel(outline, THICK, r_edge=0.4 * THICK, n=1, rings=rings, r_back=0.4 * THICK)
    return m.moved((0.0, 0.0, -0.5 * THICK)).rot(90.0, 0.0, 0.0)


def _curled_square():
    """A paper square bent round a cylinder of radius CURL (widths) along its z: x runs round the arc and the thickness
    along its normal, so the paper keeps its width and thickness; centred on its box, facing -Y."""
    m = CS.rounded_panel(CS.resample_loop(CS.rrect(1.0, 1.0, 0.06, 4), 48), THICK, r_edge=0.4 * THICK, n=1, rings=3,
                         r_back=0.4 * THICK).moved((0.0, 0.0, -0.5 * THICK))
    V = m.V.copy()
    th, z = V[:, 0] / CURL, V[:, 2]
    V[:, 0], V[:, 2] = (CURL - z) * np.sin(th), CURL - (CURL - z) * np.cos(th)
    V[:, 2] -= 0.5 * (V[:, 2].min() + V[:, 2].max())
    m.V = V
    return m.rot(90.0, 0.0, 0.0)


def _streamer(length):
    """A paper streamer curled into a helix round Z (RIB_R round, RIB_PITCH per turn), `length` widths long and centred on
    the origin: a strip RIB_W wide and THICK thick whose face looks away from the axis."""
    turns = max(1.0, length / RIB_PITCH)
    n = math.ceil(turns * RIB_STEPS) + 1
    th = np.linspace(0.0, 2.0 * math.pi * turns, n)
    path = np.stack([RIB_R * np.cos(th), RIB_R * np.sin(th), np.linspace(-0.5 * length, 0.5 * length, n)], 1)
    m = CS.sweep(path, CS.rrect(THICK, RIB_W, 0.45 * THICK, 1), up=(0.0, 0.0, 1.0))
    return m if m.volume() > 0 else m.flip()


def mesh(kind, length=RIBBON / PARAMS["size"][0]):
    """The unit piece of a kind (`core.shell.Mesh`, closed, outward, about one unit wide): a gently curled square, a heart
    and a five-point star (flat paper die-cuts facing -Y) or a streamer `length` units long along Z."""
    if kind == "square":
        return _curled_square()
    if kind == "heart":
        return _flat(HR.heart_outline(48))
    if kind == "star":
        return _flat(BU.star_outline(50))
    if kind == "ribbon":
        return _streamer(length)
    raise ValueError(f"confetti: no piece kind {kind!r} (have {', '.join(KINDS)})")


@functools.lru_cache(maxsize=64)
def radius(kind, length=RIBBON / PARAMS["size"][0]):
    """How far the unit piece of a kind reaches from its origin (its widths)."""
    return float(np.linalg.norm(mesh(kind, length).V, axis=1).max())


# ===================================================================================================================
# the layout
# ===================================================================================================================
def _frame(aim):
    """A rotation (3, 3) that turns +Z onto the direction `aim`."""
    a = np.asarray(aim, float) / np.linalg.norm(aim)
    e1 = np.cross([1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0], a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(a, e1), a], 1)


def cone(n, aim, half, turn=0.0):
    """n unit directions spread evenly over the cap of half-angle `half` (degrees) round `aim`: a golden-angle spiral
    (equal solid angle per direction, no clumps, no gaps) turned by `turn` radians."""
    c = math.cos(math.radians(half))
    i = np.arange(n)
    z = 1.0 - (1.0 - c) * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = GOLDEN * i + turn
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1) @ _frame(aim).T


def _kinds(count, mix):
    """The kinds of a volley's pieces in proportion to `mix` (largest remainder), in KINDS order."""
    total = sum(mix.values())
    want = {k: count * v / total for k, v in mix.items()}
    n = {k: math.floor(w) for k, w in want.items()}
    for k in sorted(want, key=lambda k: (n[k] - want[k], KINDS.index(k)))[:count - sum(n.values())]:
        n[k] += 1
    return [k for k in KINDS for _ in range(n.get(k, 0))]


def euler_xyz(M):
    """Blender XYZ Euler angles (radians) of a rotation matrix: M = Rz Ry Rx (core.shell.rot_matrix)."""
    return (math.atan2(M[2, 1], M[2, 2]), math.atan2(-M[2, 0], math.hypot(M[0, 0], M[1, 0])), math.atan2(M[1, 0], M[0, 0]))


def _r(x):
    """A number rounded as the expressions print it."""
    return round(float(x), HR.NDIGITS)


def layout(O):
    """The pieces of every volley: [{volley, kind, colour (index into O.colors), v (3,) (direction times push), delay (s),
    kd, kf (shares of drag and fall), sway (2,) (the flutter's horizontal direction times its share), freq (rad per Hz
    second), phase, rot0 (3,) (radians), omega (3,) (radians per turn: the tumble), share (of size), soft (the floor band,
    per metre of size)}]. Each volley takes the kinds in proportion to `mix` and the colours shared out evenly, spread over
    the cone; flat pieces start at any angle and tumble fast, a streamer leaves along its flight (its local Z) and turns
    slowly. Every number comes from the seed and is rounded as the expressions print it."""
    L = length(O)
    out = []
    for k in range(O.volleys):
        kinds = _kinds(O.count, O.mix)
        slot = sorted(range(O.count), key=lambda i: _unit(O.seed, "confetti", k, "slot", i))
        hue = sorted(range(O.count), key=lambda i: _unit(O.seed, "confetti", k, "hue", i))
        D = cone(O.count, O.aim, O.cone, 2.0 * math.pi * _unit(O.seed, "confetti", k, "turn"))
        for i, kind in enumerate(kinds):
            u = functools.partial(_unit, O.seed, "confetti", k, i)
            rib = kind == "ribbon"
            d = D[slot[i]]
            push = 0.75 + 0.25 * u("push") if rib else 0.55 + 0.45 * u("push")
            share = 0.8 + 0.2 * u("share") if rib else 0.6 + 0.4 * u("share")
            flutter = 0.3 + 0.3 * u("flutter") if rib else 0.6 + 0.4 * u("flutter")
            rate = 0.5 + 0.3 * u("rate") if rib else 0.75 + 0.5 * u("rate")
            psi = 2.0 * math.pi * u("psi")
            if rib:                                            # along its flight, rolled about itself at random
                M = _frame(d) @ CS.rot_matrix(0.0, 0.0, 360.0 * u("roll"))
                rot0 = euler_xyz(M)
                omega = [(1 if u("so", j) < 0.5 else -1) * (0.06 + 0.19 * u("w", j)) for j in range(3)]
            else:
                rot0 = [2.0 * math.pi * u("r0", j) for j in range(3)]
                omega = [(1 if u("so", j) < 0.5 else -1) * (0.35 + 0.65 * u("w", j)) for j in range(3)]
            out.append({
                "volley": k, "kind": kind, "colour": hue[i] % len(O.colors),
                "v": np.round(push * d, HR.NDIGITS), "delay": _r(DELAY * u("delay")),
                "kd": _r(1.0 + 0.3 * u("kd") if rib else 0.75 + 0.5 * u("kd")),
                "kf": _r(1.1 + 0.3 * u("kf") if rib else 0.8 + 0.4 * u("kf")),
                "sway": np.round([flutter * math.cos(psi), flutter * math.sin(psi)], HR.NDIGITS),
                "freq": _r(2.0 * math.pi * rate), "phase": _r(2.0 * math.pi * u("phase")),
                "rot0": np.round(rot0, HR.NDIGITS), "omega": np.round(2.0 * math.pi * np.array(omega), HR.NDIGITS),
                "share": _r(share), "soft": _r(FLOOR_SOFT * radius(kind, L) * share)})
    return out


# ===================================================================================================================
# the motion (numpy) and the driver expressions (the same maths)
# ===================================================================================================================
def age(t, P, piece):
    """Seconds since piece `piece` left the mouth at clip second(s) t (negative before)."""
    return np.asarray(t, float) - P[f"start{piece['volley']}"] - piece["delay"]


def motion(a, P, piece):
    """A piece at age(s) a: (position (..., 3) in the prop's frame, Euler XYZ (..., 3) radians, scale (..., 3)) and its
    helpers {a, g, z}, as the drivers have them. P: the root's values (root_values)."""
    a = np.asarray(a, float)
    ap = np.maximum(a, 0.0)
    tau = max(P["drag"], MIN_DRAG) * piece["kd"]
    g = 1.0 - np.exp(-ap / tau)
    v = piece["v"]
    z = v[2] * P["reach"] * g - piece["kf"] * P["fall"] * (ap - tau * g)
    wob = P["flutter"] * g * np.sin(piece["freq"] * P["sway"] * a + piece["phase"])
    pos = np.stack([v[0] * P["reach"] * g + piece["sway"][0] * wob, v[1] * P["reach"] * g + piece["sway"][1] * wob, z], -1)
    rot = piece["rot0"] + np.multiply.outer(P["spin"] * ap, piece["omega"])
    body = (piece["share"] * P["size"] * P["amount"] * HR._smoothstep(0.0, POP, a)
            * (1.0 - HR._smoothstep(FADE * P["life"], P["life"] + FADE_EPS, a))
            * HR._smoothstep(P["floor"], P["floor"] + FLOOR_EPS + piece["soft"] * P["size"], z))
    s = np.maximum(MIN_SCALE, body)
    sz = np.maximum(MIN_SCALE, body * (UNROLL + (1.0 - UNROLL) * g)) if piece["kind"] == "ribbon" else s
    return pos, rot, np.stack([s, s, sz], -1), {"a": a, "g": g, "z": z}


def pose(t, P, piece):
    """Where piece `piece` is at clip second(s) t: (position (..., 3) in the prop's frame, Euler XYZ (..., 3) radians,
    scale (..., 3)), as the drivers have them. P: the root's values (root_values, or as keyed)."""
    return motion(age(t, P, piece), P, piece)[:3]


def expressions(piece):
    """The driver expressions of a piece: {channel: text} for its helpers a, g, z (in that order: each reads those before
    it), x, y (its location; its z is the helper `z`), rx, ry, rz (Euler XYZ), s (its scale; on a streamer x and y) and,
    on a streamer, sz (its scale along its curl). They read the root's properties by their names (PARAMS, start<k>, f0
    the frame of clip second 0, fps), `frame` and the piece's helpers."""
    n = HR._n
    v = piece["v"]
    tau = f"max(drag, {n(MIN_DRAG)}) * {n(piece['kd'])}"
    out = {"a": f"(frame - f0) / fps - start{piece['volley']} - {n(piece['delay'])}",
           "g": f"1 - exp(-max(a, 0) / ({tau}))",
           "z": f"{n(v[2])} * reach * g - {n(piece['kf'])} * fall * (max(a, 0) - {tau} * g)"}
    wob = f"flutter * g * sin({n(piece['freq'])} * sway * a + {n(piece['phase'])})"
    for ch, c, d in (("x", v[0], piece["sway"][0]), ("y", v[1], piece["sway"][1])):
        out[ch] = f"{n(c)} * reach * g + {n(d)} * {wob}"
    for ch, r0, w in zip(("rx", "ry", "rz"), piece["rot0"], piece["omega"]):
        out[ch] = f"{n(r0)} + {n(w)} * spin * max(a, 0)"
    body = (f"{n(piece['share'])} * size * amount * smoothstep(0, {n(POP)}, a) * "
            f"(1 - smoothstep({n(FADE)} * life, life + {n(FADE_EPS)}, a)) * "
            f"smoothstep(floor, floor + {n(FLOOR_EPS)} + {n(piece['soft'])} * size, z)")
    out["s"] = f"max({n(MIN_SCALE)}, {body})"
    if piece["kind"] == "ribbon":
        out["sz"] = f"max({n(MIN_SCALE)}, {body} * ({n(UNROLL)} + {n(1.0 - UNROLL)} * g))"
    for ch, text in out.items():
        if len(text) > HR.MAX_EXPR:
            raise ValueError(f"confetti: the {ch} expression is {len(text)} characters (Blender keeps {HR.MAX_EXPR})")
    return out


def bounds(O, pieces=None, samples=200):
    """(lo, hi) (3,) each: the box in the prop's frame that every piece stays in over its whole life at the starting
    values (the floor cuts it: a piece is gone there)."""
    P = root_values(O)
    L = length(O)
    a = np.linspace(0.0, P["life"], samples)
    lo, hi = np.zeros(3), np.zeros(3)
    for p in layout(O) if pieces is None else pieces:
        pos = motion(a, P, p)[0]
        r = radius(p["kind"], L) * p["share"] * P["size"] * P["amount"]
        lo, hi = np.minimum(lo, pos.min(0) - r), np.maximum(hi, pos.max(0) + r)
    lo[2] = min(max(lo[2], P["floor"]), hi[2])
    return lo, hi
