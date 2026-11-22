"""The paper theatre's pond and its machinery (`[[set]] kind = "paper_theatre"` keys `pond`, `panorama`, `rise`, `night`,
`moon`, `loose`, `sun`, `day`; docs/design.md: Sets), numpy only: the options, the drawings, where everything stands and
how it moves. mkmmd/core/paper_theatre.py hands a theatre with any of these keys over to `scene` and `motion` here; a
theatre without them never comes here and is built as it always was.

All of it is construction paper and card, worked by hand as a toy theatre's machinery is (frame as in paper_theatre:
she stands or lies at the origin, the floor's top is z = 0 without a pond):

  * `pond`: the floor's paper is the water, W0 under z = 0, so the lily pads that float on it have their tops at
    z = 0 (the main pad round the origin carries her torso as she lies along X); five rows of scalloped waves stand
    across the stage, two in front of the pads' lane and three behind it, sliding against each other as a wave machine
    does; reeds and cattails stand in the wings just in front of the lane and just behind it; paper water lilies ride the
    lane, shut until `lilies` and then all popping open as a gold tab tucked behind the stage's pelmet is pulled down
    (its end is the hand's `use` point);
  * `panorama`: the lane (the pads, the lilies, a few reed clumps) slides past her the other way while she hops in
    place, eased from pad to pad so a pad's middle is under her feet ON each of `hops`; the rows go by faster in front
    and slower behind, the hills and the backdrop slower still (parallax);
  * `rise`: the ground (the floor and everything standing on it: the rows, the lane, the wings, the hills, the sun on
    its rod) sinks out of the bottom of the opening and the clouds are let down past her, so she seems to climb; the
    backdrop reaches down to Z_LOW under the floor so the opening shows sky under her once the floor has gone;
  * the backdrop is a roll when it moves (`panorama` or `night`): the day sky is two halves of paper meeting behind her,
    each winding onto a spool behind its pillar, turned by a crank on the right pillar (its knob the hand's `use`
    point); with `night` the backdrop card behind them is plum night paper pricked with pinholes over a card lit behind
    it, a milky way of torn paper across it, and constellations (a W, a rabbit, a dragonfly) pencilled in from star to
    star, a stroke on each of `strokes`; the halves part from the middle between `from` and `to`, and the hanging
    stars drop in on their threads;
  * `moon = {shape = "croissant", drops}`: a paper croissant for the crescent, wound up until its thread catches on
    `drops`; `loose`: every hanging piece's thread cut at once, the pieces falling out of the bottom of the opening;
  * `sun = {pops}`: the sun stands on a card rod behind the back hills (always so when the backdrop rolls), sinks with
    the ground and pops up over the hills on its split pin ON `pops`, its rays spinning fast and settling;
  * `day`: everything back as it was by day (the backdrop rolled back, the ground up, the lane home, the pieces
    re-hung, the stars and the moon wound up again, the sun down behind the hills until it pops).

Everything is held on the set's `step` grid as the rest of the set is; whatever travels sets off early so it lands ON
its moment. What changes shape (the day halves sliding round the backdrop's bends, a lily opening, a stroke being
drawn) is a flipbook: one exact shape per held pose (the parts module makes them), keyed as shape keys."""
import functools
import math
from collections import namedtuple

import numpy as np

from . import paper_theatre as PT

W0 = 0.008                      # with a pond the water (the floor's top) is this far under the pads' tops, z = 0
PAD_T = 0.004                   # a pad's card
PAD = (0.6, 0.5)                # the main pad's half lengths along X and Y: under her head, the roots of her ears and both
#                                 shoulders with pad to spare (she lies along X, hips over the origin, eyes near x -0.62)
PAD_X = -0.48                   # its middle; her legs trail off it into the water
RIDE_GIVE = 0.005               # riding (`pond.ride`) the main pad keeps within this of its rest on z: deeper its top would
#                                 go under the water's paper (W0), and a pad cannot leave the water
R_MAIN = PAD[0]
PRE = 0.4                       # s the main pad slides to put its middle under her feet before her first landing
COVER_R = 2.6                   # with the house away the water runs this far round the origin
RIPPLE = (2.0, 0.07, 0.9, 18)   # a ring's life (s), its paper's width as it leaves the rim, how far it runs, wedge repeats
RIPPLE_GAP = 0.22               # s between the touch's three rings
LANE = 0.5                      # the pads ride the lane |y| < LANE
FRONT_ROWS = (-0.76, -0.67)     # the waves in front of the lane, low (the pads at her feet show over them from a front
#                                 camera a metre up); the curtains' hems are at y -0.86 .. -0.80
WING_Y = (-0.535, 0.545)        # the reed wings: just in front of the lane, just behind it
WING_X = 0.92                   # their inner foot
HIDE_X = 1.1                    # a lane piece is drawn while its inner edge is inside |x| < HIDE_X (behind a wing beyond)
CELLAR = 0.25                   # with `rise` the stage's base is a box this much deeper than the sink: the cellar
LILY_POP = 0.16                 # s the petals take to open
TAB_PULL = (0.15, 0.22)         # the tab slides down this far (m) in this long (s)
TAB = (0.10, 0.40)              # its width and length
CRANK_M = 1.8                   # m of backdrop a turn of the crank winds
CRANK_R = 0.19                  # the knob's radius on the crank
CRANK_Z = 0.97                  # the crank's axis above the stage
DRAW = 0.2                      # s a pencil stroke takes
EARLY = 0.02                    # what lands on a moment is there this much before its held step starts (land)
DWELL = 0.05                    # s the lane holds a pad under her as she lands on it
SUN_POP = 0.14                  # s the sun takes to come up
SUN_SPIN = (1.25, 0.32)         # turns the rays spin after the pop, and the time constant they settle in
PANO_FAR = {"hills_front": 0.06, "hills_back": 0.035, "sun": 0.035, "sky": 0.05}    # their share of the lane's travel
HALF_Y = (-0.006, -0.0075)      # the day halves' faces in front of the backdrop card (left, right: the right laps over)
HALF_T = 0.0015                 # their paper
SPOOL_R = 0.045
SKY_TOP = 2.2                   # a backdrop that rolls stands this far over the opening (1.25 as it was): a low camera
#                                 looking up under the valance sees night up to 5.2 m behind her, never over it
LILY = 1.35                     # the lilies' size (the petals of a size 1 lily are 0.15 m): a flower to see on a phone
LOOSE_HIDE = -1.2               # a cut-loose piece is gone once its top is under this

POND = {"lilies": None, "tab": 0.42, "ripples": None, "ride": None, "ride_when": None}
RIPPLES = {"touch": None, "then": (), "contrast": 0.3}   # contrast: how far apart the rings' four tones are (1: the roles')
PANORAMA = {"hops": None, "toward": 1}
RISE = {"from": None, "to": None, "sink": 1.8, "lower": 2.3}
NIGHT = {"from": None, "to": None, "strokes": (), "stars": None}
MOON = {"shape": "crescent", "drops": None}
SUN = {"pops": None}
KEYS = ("pond", "panorama", "rise", "night", "moon", "loose", "sun", "day", "house")


def _err(msg):
    return PT.TheatreError(f"paper_theatre: {msg}")


def _times(v, what):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, (list, tuple)) or not v:
        raise _err(f"{what} = {v!r}: a clip second or a list of them")
    out = [PT._number(x, what) for x in v]
    if any(b <= a for a, b in zip(out, out[1:])):
        raise _err(f"{what} = {v!r}: the times must rise")
    return tuple(out)


def _table(key, v, known, required=()):
    if v is True and not required:
        v = {}
    if not isinstance(v, dict):
        raise _err(f"{key} = {v!r}: a table of {sorted(known)}")
    bad = sorted(set(v) - set(known))
    if bad:
        raise _err(f"{key}: unknown keys {bad} (known: {sorted(known)})")
    for r in required:
        if v.get(r) is None:
            raise _err(f"{key}: `{r}` is required")
    return dict(known, **v)


def check(key, v):
    """One of KEYS checked and filled with its defaults (None: the feature is off)."""
    if v is None or v is False:
        return None
    if key == "pond":
        o = _table(key, v, POND)
        o["lilies"] = None if o["lilies"] is None else PT._number(o["lilies"], "pond.lilies")
        o["tab"] = PT._number(o["tab"], "pond.tab", -0.8, 0.8)
        if o["ripples"] is not None:
            rp = _table("pond.ripples", o["ripples"], RIPPLES, ("touch",))
            o["ripples"] = {"touch": PT._number(rp["touch"], "pond.ripples.touch"),
                            "then": () if not rp["then"] else _times(rp["then"], "pond.ripples.then"),
                            "contrast": PT._number(rp["contrast"], "pond.ripples.contrast", 0.0, 1.0)}
        if o["ride"] is not None and not isinstance(o["ride"], str):
            raise _err(f"pond.ride = {o['ride']!r}: the name of the object the main pad rides")
        o["ride_when"] = None if o["ride_when"] is None else _windows(o["ride_when"], "pond.ride_when")
        return o
    if key == "panorama":
        o = _table(key, v, PANORAMA, ("hops",))
        o["hops"] = _times(o["hops"], "panorama.hops")
        if len(o["hops"]) < 2 or len(o["hops"]) > 9:
            raise _err(f"panorama.hops: 2 to 9 landings, not {len(o['hops'])}")
        if isinstance(o["toward"], bool) or o["toward"] not in (1, -1):
            raise _err(f"panorama.toward = {o['toward']!r}: 1 (she travels toward +x) or -1")
        return o
    if key == "rise":
        o = _table(key, v, RISE, ("from", "to"))
        for k in ("from", "to"):
            o[k] = PT._number(o[k], f"rise.{k}")
        if o["to"] <= o["from"]:
            raise _err("rise: `to` must come after `from`")
        o["sink"] = PT._number(o["sink"], "rise.sink", 0.0, 4.0)
        o["lower"] = PT._number(o["lower"], "rise.lower", 0.0, 4.0)
        return o
    if key == "night":
        o = _table(key, v, NIGHT, ("from", "to"))
        for k in ("from", "to"):
            o[k] = PT._number(o[k], f"night.{k}")
        if o["to"] <= o["from"]:
            raise _err("night: `to` must come after `from`")
        o["strokes"] = () if not o["strokes"] else _times(o["strokes"], "night.strokes")
        if len(o["strokes"]) > len(STROKES):
            raise _err(f"night.strokes: at most {len(STROKES)} strokes, not {len(o['strokes'])}")
        o["stars"] = None if o["stars"] is None else _times(o["stars"], "night.stars")
        return o
    if key == "moon":
        o = _table(key, v, MOON)
        if o["shape"] not in ("crescent", "croissant"):
            raise _err(f"moon.shape = {o['shape']!r}: \"crescent\" or \"croissant\"")
        o["drops"] = None if o["drops"] is None else PT._number(o["drops"], "moon.drops")
        return o
    if key == "sun":
        o = _table(key, v, SUN, ("pops",))
        o["pops"] = PT._number(o["pops"], "sun.pops")
        return o
    if key == "house":
        return _windows(v, "house")
    return PT._number(v, key)                                # loose, day: a clip second


def _windows(v, what):
    """[[from, to], ...] clip seconds, rising and apart."""
    if not isinstance(v, (list, tuple)) or not v or not all(isinstance(w, (list, tuple)) and len(w) == 2 for w in v):
        raise _err(f"{what} = {v!r}: a list of [from, to] clip seconds")
    out = [(PT._number(a, what), PT._number(b, what)) for a, b in v]
    if any(b <= a for a, b in out) or any(c < b for (_a, b), (c, _d) in zip(out, out[1:])):
        raise _err(f"{what} = {v!r}: each window from before to, the windows in order and apart")
    return tuple(out)


def on(opts):
    """Is any of the pond's keys given."""
    return any(opts.get(k) is not None for k in KEYS)


def rolls(opts):
    return opts["panorama"] is not None or opts["night"] is not None


def rod_sun(opts):
    return rolls(opts) or opts["sun"] is not None


# ===================================================================================================================
# drawings (each in its own frame, metres)
# ===================================================================================================================
def scallops(x, period, body, crest, skew):
    """The top edge of a row of waves at x: rounded scallops `crest` high over a band `body` high, a cusp every
    `period` (exactly periodic, so a row slid by a period looks the same), each leaning by `skew`."""
    v = np.mod(np.asarray(x, float) / period, 1.0)
    v = v + skew * np.sin(2.0 * math.pi * v) / (2.0 * math.pi)
    return body + crest * np.sin(math.pi * np.clip(v, 0.0, 1.0)) ** 0.7


def wave_outline(x0, x1, period, body, crest, skew):
    xs = np.arange(x1, x0 - 1e-9, -0.006)
    top = np.stack([xs, scallops(xs, period, body, crest, skew)], 1)
    return PT.ccw(np.vstack([[[x0, 0.0], [x1, 0.0]], top]))


def pad_outline(R, notch, rng, gap=0.21):
    """A lily pad of radius R about the origin (x, y), its slit opening at the angle `notch` (radians, from +x toward
    +y), `gap` radians wide at the rim, down to near the middle; the rim gently wavy."""
    a = np.linspace(notch + 0.5 * gap, notch + 2.0 * math.pi - 0.5 * gap, max(int(2 * math.pi * R / 0.008), 48))
    p1, p2 = rng.uniform(0, 2 * math.pi, 2)
    r = R * (1.0 + 0.018 * np.sin(5 * a + p1) + 0.011 * np.sin(9 * a + p2))
    rim = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    tip = 0.09 * R * np.array([math.cos(notch), math.sin(notch)])
    return PT.ccw(np.vstack([[tip], rim]))


def vein_outlines(R, notch, rng, n=7):
    """Thin tapering strips from near a pad's middle out toward its rim, the slit left out."""
    out = []
    for k in range(n):
        a = notch + (k + 0.5 + rng.uniform(-0.15, 0.15)) * 2.0 * math.pi / n
        d, nrm = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        r0, r1, w = 0.13 * R, 0.86 * R, max(0.0035, min(0.0075, 0.016 * R))
        bend = rng.uniform(-0.04, 0.04) * R
        u = np.linspace(0.0, 1.0, 12)
        c = (r0 + u[:, None] * (r1 - r0)) * d + (bend * np.sin(math.pi * u))[:, None] * nrm
        half = (0.5 * w * (1.0 - 0.8 * u))[:, None] * nrm
        out.append(PT.ccw(np.vstack([c + half, (c - half)[::-1]])))
    return out


def petal_outline(L, W):
    """A water-lily petal in its own (x, z): from its base at the origin up to a point at z = L, W wide."""
    z = np.linspace(0.0, L, 24)
    u = z / L
    w = 0.5 * W * (0.16 + 0.84 * np.sin(math.pi * np.minimum(u, 0.999) ** 0.8) ** 0.85)
    w[-1] = 0.0
    return PT.ccw(np.vstack([np.stack([w, z], 1), np.stack([-w, z], 1)[::-1][1:-1]]))


def heart_outline(r, n=14):
    a = np.arange(2 * n) * math.pi / n
    rad = np.where(np.arange(2 * n) % 2 == 0, r, 0.74 * r)
    return PT.ccw(PT.densify(np.stack([rad * np.cos(a), rad * np.sin(a)], 1), 0.004))


def reed_plates(role, width, heights, rng, blades, lean, cattails, y, both=False):
    """A clump of reeds cut from paper, `width` wide, standing on z = 0 from x = 0 (its inner foot) outward along +x,
    its face at y: a band of leaves at the water, `blades` blades glued on it one over another rising to `heights`
    [lo, hi], leaning toward -x by up to `lean` of their height (or either way), and `cattails` of them stalks with a
    brown head. (plates in (x, z), the tallest)."""
    xb = np.sort(rng.uniform(0.02, width - 0.02, blades))
    hs = rng.uniform(*heights, blades)
    lean = rng.uniform(0.35, 1.0, blades) * lean * (np.where(rng.random(blades) < 0.5, -1.0, 1.0) if both else 1.0)
    w0 = rng.uniform(0.024, 0.042, blades)
    stalks = set(int(k) for k in rng.choice(blades, size=min(cattails, blades), replace=False)) if cattails else set()
    xs = np.linspace(0.0, width, max(int(width / 0.02), 4))
    top = 0.10 + 0.02 * np.sin(xs * 31.0 + rng.uniform(0, 6.28))
    band = PT.ccw(np.vstack([[[0.0, 0.0], [width, 0.0]], np.stack([xs, top], 1)[::-1]]))
    plates = [PT._plate(role, band, y, 0.003, PT.SNIP)]
    for j, k in enumerate(rng.permutation(blades)):
        v = np.linspace(0.0, 1.0, max(int(hs[k] / 0.03), 8))
        stalk = k in stalks
        xc = xb[k] - (0.25 if stalk else 1.0) * lean[k] * hs[k] * v * v
        half = np.full(len(v), 0.0045) if stalk else 0.5 * w0[k] * np.maximum((1.0 - v) ** 0.75, 0.05)
        z = v * hs[k] - 0.02
        P = np.vstack([np.stack([xc + half, z], 1), np.stack([xc - half, z], 1)[::-1]])
        plates.append(PT._plate(role, P, y - 0.0007 * (j + 1), 0.0006, PT.SNIP_FINE))
        if stalk:
            xt = xc[-1]
            head = PT.drawn(PT.union(PT.oval(xt, hs[k] - 0.10, 0.017, 0.062),
                                     PT.box(xt - 0.0025, hs[k] - 0.05, xt + 0.0025, hs[k] + 0.045)),
                            (xt - 0.05, hs[k] - 0.2), (xt + 0.05, hs[k] + 0.07), 0.0015, 0.004)
            plates.append(PT._plate("cattail", head, y - 0.0007 * (blades + 2), 0.002, PT.SNIP_FINE))
    return plates, float(hs.max())


def gear_outline(R, teeth):
    a = np.linspace(0.0, 2.0 * math.pi, teeth * 16, endpoint=False)
    r = R * (0.9 + 0.1 * np.clip(1.6 * np.sin(teeth * a), -1.0, 1.0))
    return PT.ccw(np.stack([r * np.cos(a), r * np.sin(a)], 1))


def circle(cx, cz, r, step=0.004):
    n = max(int(2 * math.pi * r / step), 10)
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return PT.ccw(np.stack([cx + r * np.cos(a), cz + r * np.sin(a)], 1))


def croissant_drawing(R, turn=45.0):
    """A paper croissant for the moon, about R in radius: seven pillow lobes along a crescent arc, fattest in the middle
    and tapering into two horns, its hollow toward the upper left as the crescent's (`turn` degrees from a croissant
    lying hollow up); a crease between each two lobes, a sheen on each, a sleepy face on the middle one. (body field,
    creases [polygons], sheens field, eyes field, smile field, cheek field)."""
    t = math.radians(turn)
    rc = 0.66 * R
    th = [math.radians(270.0 + 26.0 * k) + t for k in range(-3, 4)]
    half = [0.28 * R * (1.0 - 0.2 * abs(k)) for k in range(-3, 4)]          # radial half size
    wide = [0.22 * R * (1.0 - 0.12 * abs(k)) for k in range(-3, 4)]         # along the arc: each laps on the next
    lobes, sheens = [], []
    for a, h, w in zip(th, half, wide):
        cx, cz = rc * math.cos(a), rc * math.sin(a)
        lobes.append(PT.oval(cx, cz, w, h, math.degrees(a) + 90.0))
        sx, sz = cx + 0.45 * h * math.cos(a), cz + 0.45 * h * math.sin(a)
        sheens.append(PT.oval(sx, sz, 0.5 * w, 0.12 * h, math.degrees(a) + 90.0))
    for s in (-1.0, 1.0):                                  # the horns' points
        a = math.radians(270.0 + s * 26.0 * 3.6) + t
        lobes.append(PT.oval(rc * math.cos(a), rc * math.sin(a), 0.07 * R, 0.08 * R, math.degrees(a) + 90.0))
    creases = []
    for k in range(6):
        a = 0.5 * (th[k] + th[k + 1])
        h = 0.7 * min(half[k], half[k + 1])
        u = np.linspace(-1.0, 1.0, 9)
        d, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        lean = 0.035 * R * (1.0 if k < 3 else -1.0)        # each crease bowed toward the nearer horn
        c = (rc + u[:, None] * h) * d + (lean * (1.0 - u * u))[:, None] * n
        wd = (0.014 * R * (1.0 - 0.5 * np.abs(u)))[:, None] * n
        creases.append(PT.ccw(np.vstack([c + wd, (c - wd)[::-1]])))
    a0 = th[3]
    fx, fz = rc * math.cos(a0), rc * math.sin(a0)
    ex, ez = -math.sin(a0), math.cos(a0)                   # along the arc at the middle lobe
    nx, nz = math.cos(a0), math.sin(a0)                    # out of the hollow (the face's down)
    eyes = []
    for s in (-1.0, 1.0):
        px, pz = fx + s * 0.085 * R * ex - 0.05 * R * nx, fz + s * 0.085 * R * ez - 0.05 * R * nz
        eyes.append(PT.inter(PT.minus(PT.disc(px, pz, 0.045 * R),
                                      PT.disc(px - 0.02 * R * nx, pz - 0.02 * R * nz, 0.045 * R)),
                             PT.disc(px + 0.03 * R * nx, pz + 0.03 * R * nz, 0.055 * R)))
    mx, mz = fx + 0.04 * R * nx, fz + 0.04 * R * nz
    smile = PT.inter(PT.minus(PT.disc(mx, mz, 0.05 * R), PT.disc(mx - 0.018 * R * nx, mz - 0.018 * R * nz, 0.05 * R)),
                     PT.disc(mx + 0.04 * R * nx, mz + 0.04 * R * nz, 0.06 * R))
    cheek = PT.oval(fx + 0.15 * R * ex + 0.01 * R * nx, fz + 0.15 * R * ez + 0.01 * R * nz, 0.045 * R, 0.03 * R,
                    math.degrees(a0) + 90.0)
    return PT.union(*lobes), creases, PT.union(*sheens), PT.union(*eyes), smile, cheek


# the constellations on the night card (s along the backdrop's plan, z up), pencilled in this order, a stroke a time: a W
# over her, then the rabbit (left) and the dragonfly (right) a stroke each in turn
STARS = {"w0": (-0.44, 3.46), "w1": (-0.22, 3.29), "w2": (0.0, 3.47), "w3": (0.22, 3.31), "w4": (0.44, 3.50),
         "r_ear1": (-1.24, 3.44), "r_ear2": (-1.02, 3.40), "r_head": (-1.18, 2.86), "r_back": (-0.86, 2.80),
         "r_tail": (-0.64, 2.56), "r_foot": (-0.78, 2.40), "r_chest": (-1.16, 2.50),
         "d_head": (1.12, 2.96), "d_thorax": (0.92, 2.73), "d_tail": (0.62, 2.39), "d_tip": (0.37, 2.07),
         "d_wing1": (0.51, 3.07), "d_wing2": (1.37, 2.47), "d_wing3": (0.57, 2.88), "d_wing4": (1.29, 2.27)}
STROKES = (("w0", "w1", "w2", "w3", "w4"), ("r_ear1", "r_head", "r_ear2"), ("d_head", "d_thorax", "d_tail", "d_tip"),
           ("r_head", "r_back", "r_tail"), ("d_wing1", "d_thorax", "d_wing2"), ("r_tail", "r_foot", "r_chest", "r_head"),
           ("d_wing3", "d_thorax", "d_wing4"))
STAR_R = 0.042                  # the constellations' stars (five-pointed holes)


# ===================================================================================================================
# the layout
# ===================================================================================================================
Row = namedtuple("Row", "name y part period amp period_s phase speed")
LanePiece = namedtuple("LanePiece", "name kind part x y r")
Petal = namedtuple("Petal", "plate phi r0 zb tilt")       # tilt (shut, open) degrees from upright, outward
Lily = namedtuple("Lily", "name x y z petals heart")
Half = namedtuple("Half", "name side plates pencil")       # a day half, unrolled (x along the backdrop's plan)
Stroke = namedtuple("Stroke", "name segments")             # [(start (s, z), end (s, z))] on the night card
Pond = namedtuple("Pond", "rows wings lane lilies landings tab tab_at crank knob crank_at halves strokes sun_up "
                          "sun_down sky z_low roll lift ripples", defaults=((),))
Ripple = namedtuple("Ripple", "name t0 rx ry reach width reverse")


def _flat(role, P, top, thick, cut=PT.SNIP):
    """A plate lying flat, its top face at z = -top (the plate's frame has its y pointing down)."""
    return PT._plate(role, P, top, thick, cut, frame=(((1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0, 0.0, 0.0)))


def pad_part(name, R, rng, notch=None, aspect=1.0):
    """A lily pad of radius R (its Y `aspect` times its X), its slit at `notch` (radians from +x; random without it),
    cut veins of darker paper on it, its top at z = 0 of its frame."""
    notch = rng.uniform(0.0, 2.0 * math.pi) if notch is None else notch
    sq = np.array([1.0, aspect])
    plates = [_flat("pad", PT.ccw(pad_outline(R, notch, rng) * sq), 0.0, PAD_T, PT.SNIP)]
    plates += [_flat("pad_vein", PT.ccw(v * sq), -0.0009, 0.0006, None) for v in vein_outlines(R, notch, rng)]
    return PT.Part(name, plates, [], [], {})


def lily_record(name, x, y, z, rng, size=1.0):
    """A water lily (its frame: the base's middle at the origin, z up): two rings of eight petals and a gold heart."""
    petals = []
    a0 = rng.uniform(0.0, 2.0 * math.pi)
    for ring, (n, L, W, r0, zb, tilt) in enumerate(((8, 0.15, 0.064, 0.030, 0.008, (-6.0, 68.0)),
                                                    (8, 0.112, 0.05, 0.020, 0.014, (-13.0, 44.0)))):
        role = "lily" if ring == 0 else "lily2"
        for k in range(n):
            phi = a0 + (k + 0.5 * ring) * 2.0 * math.pi / n + rng.uniform(-0.06, 0.06)
            w = W * size
            plate = PT._plate(role, petal_outline(L * size * rng.uniform(0.94, 1.06), w), 0.0, 0.003, PT.SNIP_FINE,
                              bow=(-0.004 * size, -0.5 * w, 0.5 * w))      # cupped: its middle further out
            petals.append(Petal(plate, float(phi), r0 * size, zb * size,
                                (tilt[0] + rng.uniform(-3, 3), tilt[1] + rng.uniform(-5, 5))))
    heart = _flat("lily_heart", heart_outline(0.034 * size), -0.026 * size, 0.004, PT.SNIP_FINE)
    return Lily(name, float(x), float(y), float(z), petals, heart)


def wing_part(name, side, y, rng, tall):
    """A reed wing standing in the water at depth y on `side` (-1 her right, +1 her left), its inner foot at WING_X,
    the blades leaning in over the stage; cattails among them."""
    plates, _top = reed_plates("reed" if y < 0 else "reed2", 0.95, tall, rng, 15, 0.22, 3, y)
    flip = np.array([side, 1.0])
    return PT.Part(name, [p._replace(outline=PT.ccw((p.outline + (WING_X, 0.0)) * flip)) for p in plates], [], [], {})


def clump_part(name, y, rng, tall):
    """A clump of reeds riding the lane (leaning either way), its middle at the origin."""
    plates, _top = reed_plates("reed2", 0.42, tall, rng, 7, 0.2, 0, y, both=True)
    return PT.Part(name, [p._replace(outline=PT.ccw(p.outline - (0.21, 0.0))) for p in plates], [], [], {})


def tab_part(name):
    """The pull tab in its own frame: its end (where the hand pinches it) at the origin, the strip up from there; a
    finger hole and an arrow pointing down glued on."""
    w, L = TAB
    strip = PT.union(PT.box(-0.5 * w, 0.5 * w, 0.5 * w, L), PT.disc(0.0, 0.5 * w, 0.5 * w))
    P = PT.drawn(strip, (-0.1, -0.02), (0.1, L + 0.02), 0.002, 0.004)
    hole = circle(0.0, 0.048, 0.017, 0.003)
    arrow = PT.union(PT.box(-0.011, 0.13, 0.011, 0.20), PT.shape(np.array([[-0.032, 0.132], [0.032, 0.132],
                                                                           [0.0, 0.088]])))
    A = PT.drawn(arrow, (-0.05, 0.07), (0.05, 0.22), 0.0015, 0.003)
    return PT.Part(name, [PT._plate("tab", P, 0.0, 0.0022, PT.SNIP, holes=[hole]),
                          PT._plate("ink", A, -0.0014, 0.0012, PT.SNIP_FINE)], [], [], {})


def crank_parts():
    """The crank (its axis at the origin, along y; the face toward -y): a gilt gear with four holes and a card arm with
    a brass split pin, turning; and the knob, a red bead at the arm's end (its own object, its middle the origin)."""
    holes = [circle(0.08 * math.cos(a), 0.08 * math.sin(a), 0.025, 0.003)
             for a in np.linspace(0.25 * math.pi, 2.25 * math.pi, 4, endpoint=False)]
    arm = PT.union(PT.box(-0.024, 0.0, 0.024, CRANK_R), PT.disc(0.0, 0.0, 0.038), PT.disc(0.0, CRANK_R, 0.034))
    A = PT.drawn(arm, (-0.08, -0.06), (0.08, CRANK_R + 0.06), 0.0015, 0.004)
    plates = [PT._plate("gilt", gear_outline(0.15, 14), -0.003, 0.003, PT.SNIP_FINE, holes=holes),
              PT._plate("cardboard", A, -0.0075, 0.0045, PT.SNIP_FINE)]
    crank = PT.Part("crank", plates, [], [(0.0, 0.0, -0.0081, 0.0)], {})
    knob = PT.Part("crank_knob", [], [((0.0, 0.0275, 0.0), (0.0, -0.05, 0.0), 0.032, "knob")], [], {},
                   (0.0, -0.035, CRANK_R))
    return crank, knob


def _night_holes(sky, z0, z1, seed):
    """The pinholes of the night card (unrolled (s, z)): the constellations' five-pointed ones and small round ones
    scattered round them, none on the milky way; and the milky way's band (its outline, torn along both edges)."""
    a, R, D, _yf, S = sky
    rng = PT._rng(seed, "night holes")
    s_max = min(S - 0.15, a + 0.5 * math.pi * R + 1.0)
    band = lambda s: 1.55 + 0.42 * s                                    # noqa: E731  the milky way's middle line
    fixed = [np.array(p) for p in STARS.values()]
    pts = list(fixed)
    holes = [PT.star_outline(STAR_R * float(rng.uniform(0.9, 1.15)), rng) + p for p in fixed]
    tries = 0
    while len(holes) < len(fixed) + 100 and tries < 6000:
        tries += 1
        p = np.array([rng.uniform(-s_max, s_max), rng.uniform(z0 + 0.2, z1 - 0.25)])
        if abs(p[1] - band(p[0])) < 0.26 or min(np.hypot(*(p - q)) for q in pts) < 0.19:
            continue
        r = float(np.clip(rng.lognormal(math.log(0.012), 0.35), 0.008, 0.022))
        pts.append(p)
        holes.append(circle(p[0], p[1], r, 0.0028))
    s = np.linspace(-S + 0.05, S - 0.05, 120)
    top = np.clip(band(s) + 0.2 + np.cumsum(rng.normal(0, 0.006, len(s))), z0 + 0.05, z1 - 0.05)
    bot = np.clip(band(s) - 0.2 + np.cumsum(rng.normal(0, 0.006, len(s))), z0 + 0.03, z1 - 0.08)
    milky = PT.ccw(np.vstack([np.stack([s, bot], 1), np.stack([s, top], 1)[::-1]]))
    return tuple(holes), milky


def _sky_night(lay, z_low, seed):
    """With `night`: the backdrop card is plum, pricked, the milky way glued on (bent with it), its stands further back;
    and the card lit behind it, its own Part."""
    a, R, D, yf, S = lay.sky
    sky_top = lay.height + SKY_TOP
    holes, milky = _night_holes(lay.sky, z_low, sky_top, seed)
    sheet = PT.densify(np.array([[-S, z_low], [S, z_low], [S, sky_top], [-S, sky_top]]), 0.25)
    card = PT._plate("night", sheet, 0.0, PT.SKY_CARD, PT.SNIP_BIG, holes=holes)._replace(spacing=0.1)
    stands = [PT._stand(x, D + PT.SKY_CARD + 0.012, 0.5 * sky_top, 0.55) for x in (-0.6 * a, 0.0, 0.6 * a)]
    sky = PT.Part("sky", [card, PT._plate("night2", milky, -0.0015, 0.0012, PT.SNIP)] + stands, [], [],
                  {"bend": lay.sky, "bend_all": True})
    light = PT.Part("starlight", [PT._plate("starlight", sheet, PT.SKY_CARD + 0.003, 0.004, PT.SNIP_BIG)], [], [],
                    {"bend": lay.sky})
    return sky, light


def _sky_day(lay, z_low):
    """The backdrop card the day halves roll over without `night`: as it was, without its pencil (the halves carry it)."""
    a, R, D, yf, S = lay.sky
    sky_top = lay.height + SKY_TOP
    sheet = PT.densify(np.array([[-S, z_low], [S, z_low], [S, sky_top], [-S, sky_top]]), 0.25)
    stands = [PT._stand(x, D + PT.SKY_CARD, 0.5 * sky_top, 0.55) for x in (-0.6 * a, 0.0, 0.6 * a)]
    return PT.Part("sky", [PT._plate("sky", sheet, 0.0, PT.SKY_CARD, PT.SNIP_BIG)] + stands, [], [], {"bend": lay.sky})


def _halves(lay, z_low, seed, extra):
    """The day sky's two halves, unrolled (x along the backdrop's plan, y behind the card's face, z up): the left from
    `extra` beyond its spool to the middle, the right from the middle (lapping 2 cm over the left, in front of it) out
    past its spool; pencil marks on each (ruled lines, compass circles, crosses), as the backdrop had."""
    a, R, D, yf, S = lay.sky
    sky_top = lay.height + SKY_TOP
    out = []
    for side in (-1, 1):
        rng = PT._rng(seed, "half", side)
        u0, u1 = (-S - extra, 0.0) if side < 0 else (-0.02, S + extra)
        yface = HALF_Y[0] if side < 0 else HALF_Y[1]
        sheet = PT.densify(np.array([[u0, z_low], [u1, z_low], [u1, sky_top], [u0, sky_top]]), 0.1)
        plates = [PT._plate("sky", sheet, yface, HALF_T, PT.SNIP_BIG)]
        far = (u0 + extra + 0.3, u1 - 0.25) if side < 0 else (u0 + 0.25, u1 - extra - 0.3)
        lines = [((far[0], 0.62), (far[1], 0.62)), ((far[0], sky_top - 0.12), (far[1], sky_top - 0.12))]
        crosses = [((x, sky_top - 0.12), 0.03) for x in np.arange(far[0] + 0.2, far[1], 0.42)]
        crosses += [((x, z), 0.024) for x, z in zip(rng.uniform(far[0] + 0.2, far[1] - 0.2, 6), rng.uniform(1.2, 3.4, 6))]
        circles = [((x, z), r) for x, z, r in zip(rng.uniform(far[0] + 0.4, far[1] - 0.4, 2), rng.uniform(1.8, 3.2, 2),
                                                  rng.uniform(0.14, 0.3, 2))]
        out.append(Half("day_" + ("L" if side < 0 else "R"), side, plates,
                        {"lines": lines, "crosses": crosses, "circles": circles, "y": yface}))
    return out


def _strokes():
    """Each stroke's segments star to star (s, z), each stopped short of the stars' holes at both ends."""
    out = []
    for k, names in enumerate(STROKES):
        segs = []
        for a, b in zip(names, names[1:]):
            p, q = np.array(STARS[a]), np.array(STARS[b])
            d = (q - p) / np.linalg.norm(q - p)
            gap = STAR_R + 0.012
            segs.append((tuple(p + gap * d), tuple(q - gap * d)))
        out.append(Stroke(f"stroke{k + 1}", segs))
    return out


def _lane(lay, seed, n, sgn):
    """The pieces that ride the lane (rest x: the main pad at PAD_X under her head and shoulders, a landing pad for each
    later hop on along her travel), the lilies, and the landing pads' rest x."""
    hw = 0.5 * lay.width
    rng = PT._rng(seed, "lane")
    xs = [0.0]                                             # along her travel from the main pad
    for _ in range(n - 1):
        xs.append(xs[-1] + float(rng.uniform(1.06, 1.2)))
    span = xs[-1] + hw + 0.8
    at = lambda x: PAD_X + sgn * x                          # noqa: E731
    pieces = [LanePiece("pad1", "pad", pad_part("pad1", PAD[0], rng, notch=0.0 if sgn > 0 else math.pi,
                                                aspect=PAD[1] / PAD[0]), PAD_X, 0.0, PAD[0])]
    rs = [PAD[0]]
    for k, x in enumerate(xs[1:], 2):
        r = float(rng.uniform(0.34, 0.42))
        rs.append(r)
        pieces.append(LanePiece(f"pad{k}", "pad", pad_part(f"pad{k}", r, rng), at(x), float(rng.uniform(-0.04, 0.04)), r))
    x, k = -hw - 0.5, 0                                    # small pads round them, never over a landing pad
    while x < span:
        x += float(rng.uniform(0.32, 0.55))
        y = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.24, 0.36))
        r = min(float(rng.uniform(0.11, 0.2)), LANE - 0.02 - abs(y))
        if r < 0.1 or any(math.hypot(x - c, y) < r + rc + 0.04 for c, rc in zip(xs, rs)):
            continue
        k += 1
        pieces.append(LanePiece(f"leaf{k}", "pad", pad_part(f"leaf{k}", r, rng), at(x), y, r))
    clumps = [-hw - 0.5] + list(np.arange(hw + 0.6, span, 1.85) + rng.uniform(0.0, 0.3))
    for k, x in enumerate(clumps, 1):                      # reed clumps in the water at the lane's back edge
        pieces.append(LanePiece(f"reeds{k}", "reeds", clump_part(f"reeds{k}", 0.47, rng, (0.38, 0.72)),
                                at(float(x)), 0.0, 0.3))
    # the lilies: round her, never between her and the front, never over her head (she lies along X, head toward -X),
    # then along the lane
    spots = [(0.42 * sgn, 0.42, 0.85), (0.95 * sgn, -0.34, 1.0), (-1.34, -0.3, 0.85), (-1.02, 0.42, 0.8),
             (0.08 * sgn, 0.47, 0.75)]
    x = 1.9
    while x < span:
        spots.append((at(x), float(rng.choice([-1.0, 1.0]) * rng.uniform(0.26, 0.38)), float(rng.uniform(0.8, 1.0))))
        x += float(rng.uniform(0.75, 1.15))
    lilies = []
    for k, (x, y, size) in enumerate(spots, 1):
        on_pad = any(math.hypot(x - p.x, (y - p.y) * (PAD[0] / PAD[1] if p.name == "pad1" else 1.0)) < p.r - 0.06
                     for p in pieces if p.kind == "pad")
        lilies.append(lily_record(f"lily{k}", x, y, 0.0 if on_pad else -W0, rng, LILY * size))
    return pieces, lilies, [at(x) for x in xs]


def _rows(lay, seed):
    """The rows of waves: two in front of the lane, three behind it up to the front hills; each as long as a camera 4 m
    out sees through the opening at its depth, and a period and a swell more, within the backdrop's sides."""
    rng = PT._rng(seed, "rows")
    backs = np.linspace(0.585, 0.45 * lay.depth - 0.075, 3)
    specs = [(FRONT_ROWS[0], 0.014, 0.036, 0.22), (FRONT_ROWS[1], 0.018, 0.044, 0.25)]
    specs += [(float(y), b, c, p) for y, b, c, p in zip(backs, (0.04, 0.055, 0.075), (0.085, 0.10, 0.12),
                                                         (0.27, 0.29, 0.31))]
    a, R, _D, _yf, _S = lay.sky
    rows = []
    for k, (y, body, crest, period) in enumerate(specs, 1):
        amp = 0.32 * period
        vis = 0.5 * lay.width * (y + 4.0) / (PT.YP + 4.0) + 0.5 * period + amp + 0.06
        wall = float(PT.sky_side(y, lay.sky)) - 0.03 if y > PT.SKY_FRONT else a + R + 0.03
        half = period * (math.floor(min(vis, wall) / period - 0.5) + 0.5)     # ending on a crest, never at a cusp
        P = wave_outline(-half, half, period, body, crest, float(rng.uniform(0.1, 0.35)))
        part = PT.Part(f"wave{k}", [PT._plate(f"wave{k}", P, y, 0.006, PT.SNIP)], [], [], {})
        rows.append(Row(f"wave{k}", y, part, period, amp, float(rng.uniform(2.3, 3.4)),
                        float(rng.uniform(0, 2 * math.pi)), (5.0 / (5.0 + y)) ** 2))   # nearer rows go by faster
    return rows


def _sun_on_rod(lay):
    """The sun on a card rod behind the back hills: (sun Part with the rod, rays Part, its pin up (x, y, z), z down)."""
    hw = 0.5 * lay.width
    r_sun = 0.17
    x = -0.62 * hw
    yb = 0.745 * lay.depth
    humps = [(-1.05 * hw / 1.1, 0.62, 1.0), (0.42 * hw / 1.1, 0.55, 1.1), (1.45, 0.40, 0.8)]   # as the back hills
    ridge = float(PT.ridge(np.array([x]), humps, 0.30)[0])
    up, down = ridge + 1.7 * r_sun + 0.1, ridge - 1.7 * r_sun - 0.05     # its rays clear of the ridge on its rod
    face, rays, eyes, smile, cheeks = PT.sun_fields(r_sun)
    slo, shi = (-2.0 * r_sun, -2.0 * r_sun), (2.0 * r_sun, 2.0 * r_sun)
    ry = -0.0015 - PT.PAPER
    fy = ry - 0.0005 - PT.PAPER
    dy = fy - 0.001 - 0.0025
    sun = [PT._plate("sun", PT.drawn(face, slo, shi, 0.003, 0.006), fy, PT.PAPER, PT.SNIP)]
    sun += [PT._plate(role, L, dy, 0.0025, PT.SNIP_FINE)
            for role, f in (("ink", eyes), ("ink", smile), ("cheek", cheeks)) for L in drawn_all(f, slo, shi, 0.0015)]
    rod = np.array([[-0.022, -up - 0.6], [0.022, -up - 0.6], [0.022, 0.0], [-0.022, 0.0]])
    sun.append(PT._plate("cardboard", PT.densify(rod, 0.1), 0.0, 0.006, None))
    at = (float(x), float(yb + 0.06), float(up))
    return (PT.Part("sun", sun, [], [(0.0, 0.0, fy - 0.0006, 0.0065)], {}, at),
            PT.Part("sun_rays", [PT._plate("sun_rays", PT.resample(rays, 0.006), ry, PT.PAPER, PT.SNIP)], [], [], {}, at),
            float(down))


def _base_for_tab(lay):
    """The stage's base with its pelmet and gilt dots 4 mm further out, so the tab slides between it and the base."""
    old = next(p for p in lay.statics if p.name == "base")
    return old._replace(plates=[old.plates[0]] + [p._replace(y=p.y - 0.004) for p in old.plates[1:]])


def tab_rest(lay, share):
    """Where the tab's end (its object's origin) is at rest: between the base and its pelmet at the middle of the
    pelmet's scallop nearest `share` of the half opening, its top behind the scallop, hanging 0.25 m under it."""
    hw = 0.5 * lay.width
    ff = hw + PT.PILLAR + 0.07
    n_sc = int(round(2.0 * ff / 0.26))
    sw = 2.0 * ff / n_sc
    cx = -ff + sw * (np.arange(n_sc) + 0.5)
    return float(cx[np.argmin(np.abs(cx - share * hw))]), PT.FLOOR_FRONT - 0.0028, -0.07 - TAB[1] - W0


def croissant_hang(h, seed):
    """The moon's Hang with a croissant for its paper (the same thread and batten; the paper's top just behind the
    thread's end and a tab of tape over it, as every piece's)."""
    rng = PT._rng(seed, "croissant")
    R = 0.36
    body, creases, sheens, eyes, smile, cheek = croissant_drawing(R)
    lo, hi = (-1.2 * R, -1.2 * R), (1.2 * R, 1.2 * R)
    P = PT.drawn(body, lo, hi, 0.003, 0.005)
    c = PT.centroid(P)
    near = np.abs(P[:, 0] - c[0]) < 0.03
    top = float(P[near, 1].max()) if near.any() else float(P[:, 1].max())
    attach = np.array([c[0], top - 0.032])
    bow = (0.006 * (P[:, 0].max() - P[:, 0].min()), P[:, 0].min() - attach[0], P[:, 0].max() - attach[0])
    face = PT.THREAD_R + 0.0002 - float(PT.bow_y(bow, np.array([0.0]))[0])
    plates = [PT._plate("croissant", P - attach, face, PT.PAPER, PT.SNIP, bow)]
    plates += [PT._plate("croissant2", q - attach, face - 0.0014, 0.0011, PT.SNIP_FINE, bow) for q in creases]
    plates += [PT._plate("croissant_hi", PT.resample(L, 0.003) - attach, face - 0.0014, 0.0011, PT.SNIP_FINE, bow)
               for L in PT.contours(*_grid(sheens, lo, hi, 0.0015)) if abs(PT.polygon_area(L)) > 1e-4]
    plates += [PT._plate(role, L - attach, face - 0.0028, 0.0009, PT.SNIP_FINE, bow)
               for role, f in (("ink", eyes), ("ink", smile), ("cheek", cheek)) for L in drawn_all(f, lo, hi, 0.0012)]
    trng = np.random.default_rng(int(rng.integers(1 << 30)))
    plates.append(PT._plate("tape", PT.tape_outline(PT.TAPE[0], PT.TAPE[1], trng) + (0.0, 0.006),
                            -PT.THREAD_R - 0.0001 - PT.TAPE[2], PT.TAPE[2], None))
    lo2, hi2 = P.min(0) - attach, P.max(0) - attach
    size = (float(hi2[0] - lo2[0]), float(hi2[1] - lo2[1]))
    period = 2.0 * math.pi * math.sqrt((h.length + 0.5 * size[1]) / PT.G)
    return h._replace(kind="croissant", size=size, part=PT.Part(h.name, plates, [], [], {}), period=float(period))


def _grid(field, lo, hi, px):
    """(the field on a grid, xs, zs) over lo..hi for PT.contours."""
    xs, zs = np.arange(lo[0], hi[0], px), np.arange(lo[1], hi[1], px)
    return field(*np.meshgrid(xs, zs)), xs, zs


def drawn_all(field, lo, hi, px, step=0.003):
    """Every loop of a drawing that has several (both eyes, both cheeks: PT.drawn keeps the biggest alone), each
    counter-clockwise, sampled every `step`."""
    return [PT.resample(PT.ccw(L), step) for L in PT.contours(*_grid(field, lo, hi, px))
            if abs(PT.polygon_area(L)) > 4.0 * px * px]


def scene(opts):
    """(Layout, Pond) of a theatre with any of the pond's keys: paper_theatre's layout with what they change in it (with
    a pond the floor's paper is the water and every static part W0 lower, and there is no grass; the backdrop is the
    night card or the plain card the halves roll over; the sun is on its rod; the croissant), and the Pond."""
    pano, night = opts["panorama"], opts["night"]
    lay, pond = _scene(opts["width"], opts["height"], opts["depth"], opts["seed"],
                       None if opts["pond"] is None else opts["pond"]["tab"],
                       0 if pano is None else len(pano["hops"]), 1 if pano is None else pano["toward"],
                       None if opts["rise"] is None else opts["rise"]["sink"], night is not None,
                       opts["moon"] is not None and opts["moon"]["shape"] == "croissant", rod_sun(opts),
                       opts["house"] is not None)
    return lay, pond._replace(ripples=ripples(opts))


def ripples(opts):
    """The rings on the water: three running out from the main pad's rim on the touch, a smaller one on each later
    time; each ring's wedges turn the other way from the one before it."""
    if opts["pond"] is None or opts["pond"]["ripples"] is None:
        return ()
    rp = opts["pond"]["ripples"]
    _life, width, reach, _n = RIPPLE
    times = [(rp["touch"] + k * RIPPLE_GAP, 1.0) for k in range(3)] + [(t, 0.6) for t in rp["then"]]
    return tuple(Ripple(f"ripple{k + 1}", float(t), PAD[0], PAD[1], reach * s, width * s, k % 2 == 1)
                 for k, (t, s) in enumerate(times))


@functools.lru_cache(maxsize=4)
def _scene(width, height, depth, seed, tab, hops, toward, sink, night_on, croissant, rod, house):
    lay = PT.layout(width, height, depth, seed)
    rise_on = sink is not None
    pond_on, roll_on = tab is not None, hops > 0 or night_on
    z_low = -(sink + CELLAR - 0.05) if rise_on else 0.0
    lift = -W0 if pond_on else 0.0
    statics = []
    for p in lay.statics:
        if pond_on and p.name in ("grass", "grass_back"):
            continue
        if rod and p.name in ("sun", "sun_rays"):
            continue
        if p.name == "floor" and (pond_on or rise_on):
            fl = p.plates[0]._replace(role="water") if pond_on else p.plates[0]
            special = {k: v for k, v in p.special.items() if not pond_on or k == "corrugated"}
            if rise_on:                                    # its cut lip stays with the base; the sheet sinks behind it
                special.pop("corrugated", None)
                Q = fl.outline.copy()
                Q[np.isclose(Q[:, 1], PT.FLOOR_FRONT), 1] = PT.FLOOR_FRONT + 0.025
                fl = fl._replace(outline=Q)
            p = p._replace(plates=[fl], special=special)
        elif p.name == "base" and (pond_on or rise_on):
            if pond_on:
                p = _base_for_tab(lay)
            if rise_on:
                p, cellar = _cellar(lay, p, sink + CELLAR)
                statics.append(cellar._replace(at=(0.0, 0.0, lift)))
        elif p.name == "sky" and (roll_on or rise_on):
            if night_on:
                p, light = _sky_night(lay, z_low, seed)
                statics.append(light._replace(at=(0.0, 0.0, lift)))
            else:
                p = _sky_day(lay, z_low)
        statics.append(p._replace(at=(p.at[0], p.at[1], p.at[2] + lift)))
    if house:                                              # the water alone when the house is away
        statics.append(PT.Part("cover", [_flat("water", circle(0.0, 0.0, COVER_R, 0.05), 0.0, 0.004, None)], [], [], {},
                               (0.0, 0.0, -W0)))
    sun_at, sun_up, sun_down = lay.sun, None, None
    if rod:
        sun, rays, sun_down = _sun_on_rod(lay)
        sun_up, sun_down = sun.at[2] + lift, sun_down + lift
        sun_at = (sun.at[0], sun.at[1], sun_up)
        statics += [sun._replace(at=sun_at), rays._replace(at=sun_at)]
    rows, wings, lane, lilies, landings, tab_p, tab_at = [], [], [], [], [0.0], None, None
    if pond_on:
        rows = _rows(lay, seed)
        for side in (-1, 1):
            for j, (y, tall) in enumerate(zip(WING_Y, ((0.75, 1.15), (0.95, 1.45)))):
                nm = f"wing_{'R' if side < 0 else 'L'}{j + 1}"
                wings.append(wing_part(nm, side, y, PT._rng(seed, nm), tall))
        lane, lilies, landings = _lane(lay, seed, max(hops, 1), float(toward))
        tab_p, tab_at = tab_part("tab"), tab_rest(lay, tab)
    crank = knob = crank_at = None
    halves, roll = [], (0.0, 0.0)
    a, R, D, yf, S = lay.sky
    if roll_on:
        crank, knob = crank_parts()
        hw = 0.5 * width
        crank_at = (0.5 * (hw + 0.085 + hw + PT.PILLAR), -0.0035, CRANK_Z - lift)   # the carrier's frame
        halves = _halves(lay, z_low, seed, extra=0.6)
        top = height + SKY_TOP
        for side in (-1, 1):
            cx = side * spool_x(lay.sky)
            statics.append(PT.Part(f"spool_{'L' if side < 0 else 'R'}", [], [((cx, yf, z_low - 0.02), (cx, yf, top + 0.03),
                                                                             SPOOL_R, "cardboard")], [], {},
                                   (0.0, 0.0, lift)))
        drift = -toward * PANO_FAR["sky"] * abs(landings[-1] - landings[0])
        roll = S + 0.12 + abs(drift)                       # how far each half winds out from the seam to clear it
        roll = (roll, roll)
    strokes = _strokes() if night_on else []
    hangs = [croissant_hang(h, seed) if (croissant and h.name == "moon") else h for h in lay.hangs]
    frame_part, frame_at = lay.frame
    lay2 = lay._replace(statics=statics, hangs=hangs, frame=(frame_part, (frame_at[0], frame_at[1], frame_at[2] + lift)),
                        sun=tuple(float(v) for v in sun_at))
    return lay2, Pond(rows, wings, lane, lilies, landings, tab_p, tab_at, crank, knob, crank_at, halves, strokes,
                      sun_up, sun_down, lay.sky, z_low, roll, lift)


def _cellar(lay, base, depth):
    """With `rise`: the stage's base runs down `depth`, a box of its card (the base its front, two sides along the
    floor's edges and a back), the cellar the ground sinks into, so nothing that went down shows under the stage."""
    _x0, fx, fy0, fy1 = lay.floor
    ff = 0.5 * lay.width + PT.PILLAR + 0.07
    front = base.plates[0]
    P = PT.densify(np.array([[-ff, -depth], [ff, -depth], [ff, -PT.FLOOR_T], [-ff, -PT.FLOOR_T]]), 0.2)
    base = base._replace(plates=[front._replace(outline=PT.ccw(P))] + list(base.plates[1:]),
                         special=dict(base.special, corrugated=(-ff, ff, fy0, -PT.FLOOR_T, 0.0)))
    corners = [(ff, fy0), (fx, PT.SKY_FRONT), (fx, fy1), (-fx, fy1), (-fx, PT.SKY_FRONT), (-ff, fy0)]
    walls = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        u, n = ((x1 - x0) / L, (y1 - y0) / L), ((y1 - y0) / L, -(x1 - x0) / L)
        W = PT.densify(np.array([[0.0, -depth], [L, -depth], [L, -PT.FLOOR_T], [0.0, -PT.FLOOR_T]]), 0.25)
        walls.append(PT._plate("proscenium", W, -0.01, 0.02, None,
                               frame=(((u[0], n[0], 0.0), (u[1], n[1], 0.0), (0.0, 0.0, 1.0)), (x0, y0, 0.0))))
    return base, PT.Part("cellar", walls, [], [], {})


def spool_x(sky):
    """The x (> 0) of a spool's axis: just inside the backdrop's side, the halves' faces wrapping round it."""
    a, R, _D, _yf, _S = sky
    return a + R + min(HALF_Y) - SPOOL_R - 0.001


def wind(V, delta, sky):
    """Points (n, 3) of a day half, unrolled (x along the plan, y behind the card's face, z up), slid `delta` along the
    plan, in the set's frame: on the plan as the backdrop bends, and past either end wound round that side's spool
    (half a turn on it, then inside its core)."""
    V = np.asarray(V, float)
    a, R, D, yf, S = sky
    s = V[:, 0] + delta
    out = PT.sky_bend(np.stack([np.clip(s, -S, S), V[:, 1], V[:, 2]], 1), sky)
    past = np.abs(s) > S
    if past.any():
        sg = np.sign(s[past])
        cx = spool_x(sky)
        rho = SPOOL_R + 0.001 + (V[past, 1] - min(HALF_Y))
        ang = np.minimum((np.abs(s[past]) - S) / rho, math.pi)
        x = cx + rho * np.cos(ang)
        y = yf - rho * np.sin(ang)
        inside = ang >= math.pi
        x = np.where(inside, cx, x)
        y = np.where(inside, yf, y)
        out[past, 0] = sg * x
        out[past, 1] = y
    return out


# ===================================================================================================================
# motion (clip seconds t, arrays)
# ===================================================================================================================
def land(opts, t):
    """When what lands on clip second t must be there: with `step`, a hair (EARLY, more than half a frame) before the
    step of the set's grid that is held on t's frame (the last step at or before t: up to 1 / `step` s early, which is
    ON the moment in a picture that changes `step` times a second); without it, a hair before t."""
    step = opts["step"]
    if step > 0.0:
        t = opts["step_from"] + math.floor((t - opts["step_from"]) * step + 1e-6) / step
    return t - EARLY


def ease_hops(t, landings, xs):
    """The lane's offset along x at t: 0 until PRE before the first landing, then the main pad eased under her feet
    (its middle at the origin) on landings[0]; on each landing the pad xs[k] still under her for DWELL, then eased off
    and into the next."""
    t = np.asarray(t, float)
    L = -np.asarray(xs, float)
    out = np.zeros(t.shape)
    m = (t >= landings[0] - PRE) & (t < landings[0])
    out[m] = L[0] * PT.smoothstep((t[m] - landings[0] + PRE) / PRE)
    for k in range(len(landings) - 1):
        a = min(landings[k] + DWELL, 0.5 * (landings[k] + landings[k + 1]))
        m = (t >= a) & (t < landings[k + 1])
        out[m] = L[k] + (L[k + 1] - L[k]) * PT.smoothstep((t[m] - a) / (landings[k + 1] - a))
        out[(t >= landings[k]) & (t < a)] = L[k]
    out[t >= landings[-1]] = L[-1]
    return out


def lily_open(opts, t):
    """How open the lilies are (0 shut, 1 open; past 1 they overshoot as they spring): shut until LILY_POP before they
    land open on `pond.lilies`, then a springy settle, shut again from `day`; open all along without it."""
    t = np.asarray(t, float)
    t_open = opts["pond"]["lilies"]
    if t_open is None:
        return np.ones(t.shape)
    t1 = land(opts, t_open)
    t0 = t1 - LILY_POP
    o = np.clip((t - t0) / LILY_POP, 0.0, 1.0) ** 1.6
    return np.where(_back(opts, t, t_open), 0.0, np.where(t >= t1, 1.0 + PT.settle(t, t1, 0.16, 3.0, 0.7), o))


def _window(t, a, b):
    return PT.smoothstep((np.asarray(t, float) - a) / (b - a))


def _back(opts, t, since):
    """True from `day` on, when the day comes back after `since`."""
    day = opts["day"]
    t = np.asarray(t, float)
    if day is None or day <= since:
        return np.zeros(t.shape, bool)
    return t >= day


def ground(opts, t):
    """How far the ground has sunk at t (m), and whether it is gone (sunk, hidden)."""
    t = np.asarray(t, float)
    rise = opts["rise"]
    if rise is None:
        return np.zeros(t.shape), np.zeros(t.shape, bool)
    back = _back(opts, t, rise["from"])
    g = np.where(back, 0.0, rise["sink"] * _window(t, rise["from"], rise["to"]))
    return g, (t >= rise["to"]) & ~back


def lane_offset(opts, pond, t):
    """(the lane's offset along x, how far it has gone by since her first landing (m, >= 0)), home again at `day`."""
    t = np.asarray(t, float)
    pano = opts["panorama"]
    if pano is None:
        return np.zeros(t.shape), np.zeros(t.shape)
    landings = [land(opts, h) for h in pano["hops"]]
    back = _back(opts, t, pano["hops"][0])
    off = np.where(back, 0.0, ease_hops(t, landings, pond.landings))
    gone_by = np.where((t >= landings[0]) & ~back, np.abs(off + pond.landings[0]), 0.0)
    return off, gone_by


def roll_open(opts, t):
    """How far the night roll has gone (0 .. 1): fast off the mark, as a hand yanks a crank round, slowing to a stop."""
    t = np.asarray(t, float)
    night = opts["night"]
    if night is None:
        return np.zeros(t.shape)
    u = np.clip((t - night["from"]) / (night["to"] - night["from"]), 0.0, 1.0)
    return np.where(_back(opts, t, night["from"]), 0.0, 1.0 - (1.0 - u) ** 2)


def halves_offset(opts, pond, t):
    """How far each day half has slid along the plan (m, + toward +x): (left, right)."""
    toward = 1.0 if opts["panorama"] is None else float(opts["panorama"]["toward"])
    drift = -toward * PANO_FAR["sky"] * lane_offset(opts, pond, t)[1]
    e = roll_open(opts, t)
    return drift - pond.roll[0] * e, drift + pond.roll[1] * e      # evenly apart from the seam where the drift left it


def crank_angle(opts, pond, t):
    """The crank's turn (radians; clockwise seen from the front as the paper winds onto the left spool): a turn for
    every CRANK_M of paper wound."""
    left, _right = halves_offset(opts, pond, t)
    return -2.0 * math.pi * left / CRANK_M


def stroke_progress(opts, k, t):
    """How much of stroke k is drawn (0 .. 1): DRAW long, whole as it lands on its time; none after `day`."""
    t = np.asarray(t, float)
    night = opts["night"]
    if night is None or k >= len(night["strokes"]):
        return np.zeros(t.shape)
    t1 = land(opts, night["strokes"][k])
    p = np.clip((t - (t1 - DRAW)) / DRAW, 0.0, 1.0)
    return np.where(_back(opts, t, night["from"]), 0.0, p)


def sun_height(opts, pond, t):
    """The sun's pin on its rod (z) and the rays' extra spin (radians) at t: up from the start (down from the start when
    it pops and nothing else brings it up), down behind the hills from `day`, popped up ON `pops`: SUN_POP to come up,
    a bounce, the rays spinning fast and settling. (Sinking with the ground is the caller's.)"""
    t = np.asarray(t, float)
    up, down = pond.sun_up, pond.sun_down
    z, spin = np.full(t.shape, up), np.zeros(t.shape)
    sun = opts["sun"]
    if sun is None:
        return z, spin
    t1 = land(opts, sun["pops"])
    t0 = t1 - SUN_POP
    day = opts["day"]
    rest = day if (day is not None and day < t0) else None
    if rest is None and opts["rise"] is None and opts["night"] is None:
        rest = -1e9
    if rest is not None:
        z = np.where((t >= rest) & (t < t0), down, z)
    u = np.clip((t - t0) / SUN_POP, 0.0, 1.0)
    z = np.where((t >= t0) & (t < t1), down + (up - down) * u ** 0.7, z)
    after = t >= t1
    z = np.where(after, up + 0.5 * PT.settle(t, t1, 0.12, 2.6, 0.9), z)
    spin = np.where(after, 2.0 * math.pi * SUN_SPIN[0] * (1.0 - np.exp(-np.maximum(t - t1, 0.0) / SUN_SPIN[1])), 0.0)
    return z, spin


def piece_catch(opts, lay, h):
    """The clip second a hanging piece's thread catches when it drops in on its own (the stars with `night`, the moon
    with `drops`), or None."""
    if h.kind == "star" and opts["night"] is not None:
        stars = opts["night"]["stars"]
        i = sorted(x.name for x in lay.hangs if x.kind == "star").index(h.name)
        gap = 2.0 * max(opts["stagger"], 0.08)
        if stars is None:
            return opts["night"]["from"] + 0.45 + i * gap
        return stars[i] if i < len(stars) else stars[-1] + (i - len(stars) + 1) * gap
    if h.kind in ("moon", "croissant") and opts["moon"] is not None and opts["moon"]["drops"] is not None:
        return opts["moon"]["drops"]
    return None


def catch_lead(length):
    """Seconds from a piece's release to its thread's first catch (paper_theatre.yoyo's fall)."""
    return math.sqrt(2.0 * max(length - PT.YOYO["wound"], 1e-6) / PT.YOYO["g"])


def motion(opts, lay, pond, t):
    """Every keyed value at the clip seconds t of a theatre with the pond's keys: ([Channel], {object: hidden},
    {object: (states, index)}). Objects as paper_theatre.motion names them (without the set's prefix); a flipbook is
    its distinct states (one number each: a half's slide, a lily's openness, a stroke's share drawn) and, for each t,
    the index of the state shown."""
    t = np.asarray(t, float)
    Ch = PT.Channel
    ch, hidden, flips = [], {}, {}
    lift = pond.lift
    arrive = opts["arrive"]
    if arrive is not None:                                 # the frame drops in (as paper_theatre.motion, on the water)
        down = arrive + PT.DROP_FALL
        ch.append(Ch("frame", "location", 2, PT.drop(t, arrive, lay.height, PT.DROP_FALL) + lift))
        ch.append(Ch("frame", "rotation_euler", 0, np.radians(PT.settle(t, down, PT.ROCK[0], PT.ROCK[1], PT.ROCK[2]))))
        ch.append(Ch("proscenium", "shape:flex", 0, PT.settle(t, down, PT.FLEX[0], PT.FLEX[1], PT.FLEX[2], math.pi)))
        for c in lay.curtains:
            lag = 0.03 if c.side > 0 else 0.0
            ch.append(Ch(c.name, "rotation_euler", 0, np.radians(
                PT.settle(t, down + lag, -PT.CURTAIN_SWAY[0], PT.CURTAIN_SWAY[1], PT.CURTAIN_SWAY[2]))))
            ch.append(Ch(c.name, "rotation_euler", 1, np.radians(
                PT.settle(t, down + lag, c.side * PT.CURTAIN_SIDE[0], PT.CURTAIN_SIDE[1], PT.CURTAIN_SIDE[2]))))
            ch.append(Ch(c.name, "shape:flex", 0, PT.settle(t, down + 0.05 + lag, PT.CURTAIN_FLEX[0], PT.CURTAIN_FLEX[1],
                                                             PT.CURTAIN_FLEX[2])))
        for o in ("proscenium", "curtain_L", "curtain_R"):
            hidden[o] = t < arrive
    g, gone = ground(opts, t)
    lane_x, lam = lane_offset(opts, pond, t)
    toward = 1.0 if opts["panorama"] is None else float(opts["panorama"]["toward"])
    rising = opts["rise"] is not None

    # the sun turning on its pin; on its rod it rides the back hills, sinks with them and pops
    turn = 2.0 * math.pi * opts["sun_rpm"] / 60.0 * t
    if pond.sun_up is not None:
        z, spin = sun_height(opts, pond, t)
        sx = lay.sun[0] - toward * PANO_FAR["sun"] * lam
        for nm in ("sun", "sun_rays"):
            ch.append(Ch(nm, "location", 0, sx))
            ch.append(Ch(nm, "location", 2, z - g))
            if rising:
                hidden[nm] = gone
        turn = turn + spin
    ch.append(Ch("sun_rays", "rotation_euler", 1, turn))

    # the ground: what stands on the floor sinks with it; the hills slide slowly in the panorama
    for p in lay.statics:
        if rising and p.name in ("floor", "hills_back", "hills_front", "grass", "grass_back"):
            ch.append(Ch(p.name, "location", 2, p.at[2] - g))
            hidden[p.name] = gone
        if opts["panorama"] is not None and p.name in ("hills_back", "hills_front"):
            ch.append(Ch(p.name, "location", 0, -toward * PANO_FAR[p.name] * lam))
    for w in pond.wings:
        if rising:
            ch.append(Ch(w.name, "location", 2, lift - g))
            hidden[w.name] = gone
    for r in pond.rows:
        off = r.amp * np.sin(2.0 * math.pi * t / r.period_s + r.phase) - toward * r.speed * lam
        ch.append(Ch(r.name, "location", 0, np.mod(off + 0.5 * r.period, r.period) - 0.5 * r.period))
        if rising:
            ch.append(Ch(r.name, "location", 2, lift - g))
            hidden[r.name] = gone
    if pond.lane:
        ch.append(Ch("lane", "location", 0, lane_x))
        if rising:
            ch.append(Ch("lane", "location", 2, -g))
        for p in pond.lane:
            hidden[p.name] = (np.abs(p.x + lane_x) - p.r > HIDE_X) | gone
        o = lily_open(opts, t)
        for li in pond.lilies:
            hidden[li.name] = (np.abs(li.x + lane_x) - 0.12 > HIDE_X) | gone
            flips[li.name] = _flip(o, 4)
        for rp in pond.ripples:
            u = (t - land(opts, rp.t0 + EARLY)) / RIPPLE[0]
            hidden[rp.name] = (u < 0.0) | (u >= 1.0) | gone
            flips[rp.name] = _flip(np.clip(u, 0.0, 1.0), 3)
    if pond.tab is not None:
        t_open = opts["pond"]["lilies"]
        pull = np.zeros(t.shape)
        if t_open is not None:
            t1 = land(opts, t_open)
            u = np.clip((t - (t1 - TAB_PULL[1])) / TAB_PULL[1], 0.0, 1.0)
            pull = TAB_PULL[0] * (PT.smoothstep(u) + np.where(t >= t1, PT.settle(t, t1, 0.05, 4.0, 0.4), 0.0))
            pull = np.where(_back(opts, t, t_open), 0.0, pull)
        ch.append(Ch("tab", "location", 2, pond.tab_at[2] - pull))

    # the roll: the day halves (flipbooks) and the crank; the night's strokes
    if pond.halves:
        left, right = halves_offset(opts, pond, t)
        flips["day_L"], flips["day_R"] = _flip(left, 4), _flip(right, 4)
        ch.append(Ch("crank", "rotation_euler", 1, crank_angle(opts, pond, t)))
    for k, s in enumerate(pond.strokes):
        p = stroke_progress(opts, k, t)
        flips[s.name] = _flip(p, 4)
        hidden[s.name] = p <= 0.0

    # the hanging pieces: their own drops, the clouds let down, cut loose, back at day
    rise, loose, day = opts["rise"], opts["loose"], opts["day"]
    for h in lay.hangs:
        catch = piece_catch(opts, lay, h)
        t0 = land(opts, catch) - catch_lead(h.length) if catch is not None else (
            PT.release(opts, h) if arrive is not None else None)
        if t0 is None:
            length, env, hid = np.full(t.shape, h.length), np.ones(t.shape), np.zeros(t.shape, bool)
        else:
            length, first = PT.yoyo(t, t0, h.length)
            env, hid = PT.smoothstep((t - first) / PT.SWING_RAMP), t < t0
        extra = np.zeros(t.shape)
        if rise is not None and h.kind == "cloud":
            extra = rise["lower"] * _window(t, rise["from"], rise["to"])
        fall, flutter = np.zeros(t.shape), np.zeros(t.shape)
        if loose is not None:
            w = np.maximum(t - loose, 0.0)
            fall = np.where(t >= loose, 0.5 * PT.G * w * w, 0.0)
            flutter = np.where(t >= loose, np.sin(0.5 * math.pi * np.clip(w / 0.5, 0.0, 1.0)), 0.0)
            hid = hid | ((t >= loose) & (h.anchor[2] - fall - length - extra < LOOSE_HIDE))
        if day is not None:
            back = t >= day
            extra, fall, flutter = (np.where(back, 0.0, v) for v in (extra, fall, flutter))
            if catch is not None:
                hid, length = hid | back, np.where(back, PT.YOYO["wound"], length)
            elif t0 is None:
                hid = hid & ~back
        if hid.any():
            hidden[h.name] = hid
            hidden[h.name + "_thread"] = hid
        side, fb, twist = PT.amplitudes(opts, h)
        wv = 2.0 * math.pi / h.period
        scale = h.length / (h.length + extra)              # let down, it swings as far as it did, not further
        ch.append(Ch(h.name + "_hang", "rotation_euler", 1, side * env * scale * np.sin(wv * t + h.phase)))
        ch.append(Ch(h.name + "_hang", "rotation_euler", 0, fb * env * scale * np.sin(wv * t + h.phase + 1.9)))
        if loose is not None:
            ch.append(Ch(h.name + "_hang", "location", 2, h.anchor[2] - fall))
        ch.append(Ch(h.name, "location", 2, -(length + extra)))
        tw = twist * env * np.sin(2.0 * math.pi * t / h.twist_period + h.twist_phase)
        if loose is not None:                              # cut loose, the paper tips and turns as it falls
            r = PT._rng(opts["seed"], "flutter", h.name)
            tip, roll = r.uniform(0.35, 0.6) * r.choice([-1, 1]), r.uniform(0.25, 0.5) * r.choice([-1, 1])
            ch.append(Ch(h.name, "rotation_euler", 0, tip * flutter))
            ch.append(Ch(h.name, "rotation_euler", 1, roll * flutter))
            tw = tw + r.uniform(0.6, 1.2) * r.choice([-1, 1]) * flutter
        ch.append(Ch(h.name, "rotation_euler", 2, tw))
        ch.append(Ch(h.name + "_thread", "scale", 2, length + extra))
    return ch, hidden, flips


def _flip(values, digits):
    """A flipbook of a state that is one number: (distinct states, index per t), the values rounded to `digits`."""
    states, index = np.unique(np.round(np.asarray(values, float), digits), return_inverse=True)
    return states, index


def objects(name, lay, pond):
    out = [f"{name}_{p.name}" for p in lay.statics] + [f"{name}_frame", f"{name}_proscenium"]
    out += [f"{name}_{c.name}" for c in lay.curtains]
    for h in lay.hangs:
        out += [f"{name}_{h.name}_hang", f"{name}_{h.name}", f"{name}_{h.name}_thread"]
    out += [f"{name}_{r.name}" for r in pond.rows] + [f"{name}_{w.name}" for w in pond.wings]
    if pond.lane:
        out += [f"{name}_lane"] + [f"{name}_{p.name}" for p in pond.lane] + [f"{name}_{li.name}" for li in pond.lilies]
    if pond.tab is not None:
        out.append(f"{name}_tab")
    if pond.crank is not None:
        out += [f"{name}_crank", f"{name}_crank_knob"] + [f"{name}_{h.name}" for h in pond.halves]
    out += [f"{name}_{s.name}" for s in pond.strokes] + [f"{name}_{rp.name}" for rp in pond.ripples]
    return out


def card_extra(name, opts, lay, pond):
    """What the pond adds to the card: use.look `tab` and `crank` riding their objects (a hand's point is
    obj('<name>_tab').loc, the tab's end, and obj('<name>_crank_knob').loc, the knob's middle), use.rest `pad` (the main
    pad's top, her bed), and the `theatre` entries of what it keys when."""
    look, rest, info = [], [], {}
    if pond.tab is not None:
        look.append({"name": "tab", "point": [round(v, 4) for v in pond.tab_at], "object": f"{name}_tab"})
        rest.append({"name": "pad", "point": [PAD_X, 0.0, 0.0], "half": list(PAD)})
        info["pond"] = {"lilies": opts["pond"]["lilies"], "water": -W0, "pad": {"x": PAD_X, "half": list(PAD)},
                        "rows": [round(r.y, 3) for r in pond.rows], "lilies_n": len(pond.lilies)}
    if pond.crank is not None:
        cx, cy, cz = pond.crank_at
        look.append({"name": "crank", "point": [round(cx, 4), round(PT.YP + cy - 0.024, 4),
                                                round(cz + lift_of(opts) + CRANK_R, 4)], "object": f"{name}_crank_knob"})
    if opts["panorama"] is not None:
        info["panorama"] = {"hops": list(opts["panorama"]["hops"]), "pads_x": [round(x, 3) for x in pond.landings],
                            "toward": opts["panorama"]["toward"]}
    if opts["rise"] is not None:
        info["rise"] = dict(opts["rise"])
    if opts["night"] is not None:
        info["night"] = {"from": opts["night"]["from"], "to": opts["night"]["to"],
                         "strokes": list(opts["night"]["strokes"]), "roll_m": [round(v, 3) for v in pond.roll]}
    if opts["sun"] is not None:
        info["sun"] = {"pops": opts["sun"]["pops"], "up": round(pond.sun_up, 3), "down": round(pond.sun_down, 3)}
    for k in ("loose", "day"):
        if opts[k] is not None:
            info[k] = opts[k]
    return look, rest, info


def lift_of(opts):
    return -W0 if opts["pond"] is not None else 0.0


def house_keys(opts, frame0, fps, start):
    """(key frames, gone) of the house: away from `start` until each `house` window's cut frame and from its end's;
    keyed half a frame before each cut frame (frame0 + round(t x fps)), so it switches ON the cut."""
    ks, vs = [start - 0.5], [True]
    for a, b in opts["house"]:
        ks += [frame0 + round(a * fps) - 0.5, frame0 + round(b * fps) - 0.5]
        vs += [False, True]
    return np.array(ks, float), np.array(vs, bool)


def keeps(pond):
    """What stays when the house is away: the cover water, the pads, the lilies, the ripples (and the lane, an empty)."""
    return ({"cover", "lane"} | {p.name for p in pond.lane if p.kind == "pad"} | {li.name for li in pond.lilies}
            | {rp.name for rp in pond.ripples})
