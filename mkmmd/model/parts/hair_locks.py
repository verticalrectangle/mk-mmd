"""The lock hairdo of the hair part (bpy-free, numpy only): `[hair] style = "locks"`.

A hairdo is written down the way it is combed: groups of locks (`[hair.locks.<name>]`), each a few guides and the
number of locks laid between them. A guide is one lock's course:

    root = [theta, phi]   where it is rooted on the scalp, in degrees: theta round the head from the front toward her
                          left (her right is negative, the back 180), phi down from the crown (her eyes are near 90).
                          Or a point of a line: `["hairline", theta]` is on the hairline at theta, and `["part", 0.3]`
                          is 0.3 of the way along the parting `part` of `[hair.partings]` (`part = [[theta, phi],
                          ...]`, a curve through those points). Locks between two guides rooted on one line are rooted
                          on it too, so the two sides of a parting meet in one clean line
    to = [theta, phi]     where it leaves the head. From the root to here the hair lies on the head (on its hull,
                          `[hair.volume]`), combed the short way over it, through `via = [[theta, phi], ...]` if given
    length = metres       how far it falls once it has left the head (0: it ends there)

plus any of the group's own numbers (`width`, `tuck`, `lean`, `flare`, `stiff`, `thick`, `cup`, `tip_len`, `free`,
`finger` ...): the locks near that guide take them. A group's `count` locks lie between its guides in turn, evenly, so
two guides make a fan and five a fringe that sweeps across the forehead; each lock is a little off its place (`jitter`).

Hair that is tied back is written with a tie, a named place on the head (`[hair.ties] tail = {at = [theta, phi]}`):

    to = "tail"           the lock is gathered to the tie: it lies on the head all its way and ends in the tie (no
                          `length`, no free end). Every guide of a gathered group names the one tie
    root = "tail"         the lock hangs from the tie, a tail: it leaves the head there the way `out = [theta, phi]`
                          points (the guide's, else the tie's; by default out from the head and down in equal parts)
                          and falls `length` metres. A tail has a `width` and no `to`; with `plait` it is a braid

Gathered hair lies close at the nape (`[hair.volume] nape`: the hull comes in to it behind the ears).

The locks of a group overlap, lie in one surface and are shaded and painted as one where they lie together: each is as
wide as the room between its neighbours all the way from the root, so a fan of locks starts as narrow as its roots are
close and the group is one sheet without holes. Over their last `free` metres they part: each narrows to `finger` of
its width, gets its own thickness, its edges their shade (hair_paint), and ends in its point. So a group reads as a
sheet of hair cut into locks at its end, not as a pile of leaves.

A group (defaults in GROUP):
    family      what the locks are: `bangs`, `side`, `back`, `top`, `strand` or `braid`. It names their chains (前髪,
                横髪左 / 横髪右 and 三つ編左 / 三つ編右 by the side they hang on, 後髪, 髪) and so the physics family
                mk's solver and MMD move them by
    count       locks between the guides (a single lock lies on the first guide)
    layer       what lies over what: a higher layer stands further off the head (LAYER_STEP each) and is drawn later
    width       metres across a lock; without it the room between neighbours times `overlap`, all along its way
    rhythm      [1.0, 0.6, ...]: the locks' widths in turn, as shares (wide and narrow locks side by side); a narrower
                lock rides over its neighbours (RIDE)
    free        metres at a lock's end where it hangs on its own; `finger` its width there over its full width
    thick       a free lock's thickness over its width (never thinner than THICK's least, nor thicker than
                `thick_max` metres); `cup` how far its edges bend toward the head (1 / m)
    root_w      a lock with a `width` of its own: its width at the root over `width`, easing to full over the first
                `root_len` metres
    tip         how a lock ends: "point" (it tapers into its point over `tip_len` metres, `tip_power` the point's
                sharpness: 1 a straight wedge) or "round" (rounded off over `tip_len`: a blunt cut with soft corners)
    cut         a height (metres): every lock of the group ends there, level, as hair cut straight across: it lies
                on the head, or falls, until it comes down to that height (it needs no `length`)
    lift        metres off the hull on its way over the head (negative: tucked under what lies over it): numbers
                spread evenly from the root to where it leaves, or `[[metres from the root, metres off], ...]`,
                level after the last. Hair that rises out of a parting: `[[0, -0.012], [0.025, 0.014], [0.1, 0.006]]`
    swell       times the `lift`; a guide's own makes the hair rise more at one end of a row than at the other
    stiff       how soon a falling lock turns to hang straight down (0 .. 1 per step of STEP)
    tuck        pulls a falling lock in toward the neck; `flare` pushes it out and `lean` bends it toward her back
                (negative: forward) until it is past her shoulders, below them it hangs (SETTLE)
    hug         metres: a falling lock is drawn in to this far off her face and head (its middle line) while it
                passes them, down to her chin (HUG_FADE), as a side lock follows the cheek line; without it a lock
                is only kept off them
    mass        {enabled, thin, over}: the group hangs as one heavy mass (down to the shoulder joints never coming
                closer to the neck's axis than where it left the head: hair hangs straight from the head's widest
                part onto the shoulders): its locks have their own thickness and cup
                from where they leave the head (not only at their free ends), thinning to `thin` of it at their ends
                (over their last `over` metres; all the way down without it),
                and each is as wide as the room to its neighbours all down its way, so the mass has no holes
    wave        {amp, length, phase, drift, grow, depth, start}: waves along the fall, `amp` metres to either side,
                one wave every `length` metres, starting `start` below where the lock leaves the head, `grow` times
                wider at the tip; neighbours are `drift` radians apart, so a group waves together; `depth` is the share
                that goes in and out instead of sideways (1: a corkscrew)
    flip        {len, out}: the last `len` metres turn outward by `out` metres (negative: curl under)
    jitter      {root, to (degrees), length, width (shares), lift, cut (metres), phase (radians)}: how far a lock
                strays (`cut`: how far above or below the group's cut it ends)
    under       {enabled, length, width, lift}: a second, darker row half a place aside under this one, its locks
                `length` and `width` times these and `lift` metres closer to the head: what shows between the locks
    chain       {bones, every, radius}: every `every`-th lock is a chain of `bones` bones from where it leaves the
                head (0: the group is rigid on the head); the locks between ride the nearest. A capsule is as thin as
                the room to the body's colliders at rest asks (MIN_RADIUS at least)
    plait       {enabled, pitch, start, ties}: the group's locks are plaits, three strands woven along each lock's
                path (hair_braids' weave) from where it leaves the head (or `start` metres from its root), a crossing
                every `pitch` metres (PITCH of the plait's width unless given: a plait reads as one by its crossings
                being close), to a band and a tuft of loose locks at its end (`ties`: "end", "both" bands its top
                too, "none"; `ribbon` the bands' colour). A plait is its lock's `width` wide (give one);
                hair_braids' weave keys may be given here too (over PLAIT). Hair that ends in another colour
                (`[hair.paint] tips`) plaits into it over the plait's last `tip_len` metres. A long braid is one
                lock: `count = 1`, `family = "braid"`, a chain of its own
    bow         {enabled, above, span, ribbon, shade, pinch, fan, tails}: each lock is tied `above` metres over its
                end with a satin bow `span` metres wide (a band round the lock, a knot on its face, two loops across
                it, two short tails `tails` long down its face; BOW_SHAPE its proportions; LOCK_BOW the defaults) in
                `ribbon`, shaded to
                `shade` (a darker `ribbon` without it). The lock is gathered to `pinch` of its width and thickness
                at the tie and fans out to `fan` of its width below it; the bow rides the lock's chain

Falling locks slide over the body and the garments built before the hair and face the way those face under them
(hair_clear); `[hair] cloth` (metres) is the room they leave over skin that no garment built before the hair covers,
for clothes she wears that are built after it (or not at all, in a build of the hair alone). `build_lock_hair` lays
the scalp cap under it all (hair_head's) and returns the Piece hair.py merges; the texture is hair_paint's atlas, a
cell per lock; the shading normals are hair_head's own."""
import numpy as np

from . import cat_common as CC
from . import hair_braids as BR
from . import hair_braids_bow as BOW
from . import hair_braids_plait as PL
from . import hair_braids_tex as TX
from . import hair_head as HH
from . import hair_paint as PT
from . import hair_tex as TEX
from .. import spec as SP
from ...core import bonemap
from ..part import Material
from .hair_braids_path import Plan
from .hair_clear import Surround
from .hair_fit import angles, dirs, unit
from .hair_geo import (MeshAccum, Piece, arclen, frames, resample, shell_normals, smooth_polyline, smoothstep, spline,
                       split_shell, sweep)

STEP = 0.006                    # metres between a path's points
ROOM = 12                       # points along a lock's way over the head where the room to its neighbours is measured
NARROW = 0.12                   # a lock is never narrower than this share of its full width where its roots crowd
HAIRLINE_IN = np.radians(2.0)   # a root "on the hairline" is this far inside it
LAYER_STEP = 0.0025             # metres a layer stands off the one under it
CTRL = 9                        # control directions a guide's course on the head is kept as
LEAVE = 0.2                     # radians a lock rooted on a line is combed straight away from it before it turns
ROOT = (0.010, 0.004)           # a lock comes out of the scalp cap over its first 10 mm, from 4 mm under it
MIN_OFF = 0.004                 # a lock never lies closer to the skin than this
REST = 0.010                    # a falling lock's middle line rests this far off the body and the garments it lies on
NEAR = 0.035                    # within this of them a lock turns to face as they do (all the way at REST)
TURN = 0.03                     # metres over which a lock's facing turns
MIN_RADIUS = 0.003              # a chain's capsules are never thinner than this
CLEAR = 0.005                   # ... and keep this clear of every static collider at rest where they can
RIDE = 0.004                    # metres a lock of half the width rides over its full-width neighbours (`rhythm`)
THICK = (0.0025, 0.010)         # a free lock is never thinner than this, nor thicker than `thick_max` (this by default)
FLAT = 0.04                     # a lock's thickness where it lies with its neighbours, over its free thickness
HAIR = 0.0002                   # metres between neighbours of a group where they lie in one surface (who is on top)
OWN = 0.6                       # where a lock hangs free it is shaded this much less like the head (its own form shows)
CAP_PX = (512, 256)             # the scalp cap's cell in the atlas
PITCH = 0.28                    # a plait's crossings are this share of its width apart (`plait.pitch` says otherwise)
PLAIT = {"strand_wid": 0.60, "strand_k": 7}     # hair_braids' weave keys a lock's plait has of its own: full, round
TIE = {"at": None, "out": None, "off": 0.006, "stem": 0.012}    # a tie of [hair.ties]: where it is, the way a tail
#                                 leaves it, how far over the hull a tail's root stands, how far it goes before it falls
THROAT = 0.010                  # hair gathered to a tie runs this far past where a tail's band sits, into it
LOCK_BOW = {"enabled": False, "above": 0.050, "span": 0.055, "ribbon": "#d8344a", "shade": None, "pinch": 0.6,
            "fan": 1.15, "tails": 0.024}
BOW_SHAPE = {"knot": (0.15, 0.12, 0.10), "wing_h": 0.21, "wing_th": 0.12, "droop": 10.0, "band_h": 0.16,
             "band_thick": 0.037, "tail_width": 0.14, "notch": 0.10, "spread": 12.0}   # a bow's parts over its span
#                                 (hair_braids' bow: knot half sizes, a loop's half height and puff, ...), droop degrees
FALL_MAX = 3.0                  # metres a lock of a group with a `cut` falls at most to come down to it
SETTLE = 0.06                   # metres over the shoulder joints over which a falling lock's `lean` and `flare` fade
TIE_EASE = 0.025                # metres over which a lock gathers into a bow's tie
HANG = 0.04                     # metres after it leaves the head over which a lock of a `mass` takes its own thickness
ROOM_ALONG = 24                 # points along a whole lock of a `mass` where the room to its neighbours is measured
HUG_FADE = 0.03                 # metres above her chin over which a falling lock's `hug` lets go of her face
GROUP = {
    "enabled": True, "family": "back", "count": 5, "layer": 0, "guides": [],
    "width": None, "overlap": 1.9, "rhythm": [1.0], "free": 0.08, "finger": 0.7, "thick": 0.12, "cup": 6.0,
    "thick_max": THICK[1], "root_w": 0.3, "root_len": 0.03, "tip": "point", "tip_len": 0.07, "tip_power": 1.3,
    "cut": None, "lift": [-0.012, 0.0], "stiff": 0.22, "tuck": 0.15, "flare": 0.0, "lean": 0.0, "swell": 1.0,
    "hug": None, "mass": {"enabled": False, "thin": 1.0, "over": None},
    "wave": {"amp": 0.0, "length": 0.12, "phase": 0.0, "drift": 0.6, "grow": 0.5, "depth": 0.3, "start": 0.03},
    "flip": {"len": 0.0, "out": 0.0},
    "jitter": {"root": 2.0, "to": 2.0, "length": 0.08, "width": 0.12, "lift": 0.001, "phase": 0.3, "cut": 0.0},
    "under": {"enabled": False, "length": 1.06, "width": 1.15, "lift": -0.006},
    "chain": {"bones": 0, "every": 1, "radius": 0.010},
    "plait": {"enabled": False, "pitch": None, "start": None, "ties": "end"},
    "bow": LOCK_BOW,
}
NUMBERS = ("width", "free", "finger", "thick", "cup", "root_w", "root_len", "tip_len", "tip_power", "stiff", "tuck",
           "flare", "lean", "swell")
FAMILIES = {"bangs": ("前髪", "bangs"), "side": ("横髪", "hair"), "back": ("後髪", "hair"), "top": ("髪", "hair"),
            "strand": ("髪", "hair"), "braid": ("三つ編", "braid")}
SIDED = ("side", "braid")       # the families whose chains are named by the side they hang on (左 / 右)
DEFAULTS = {**{k: HH.DEFAULTS[k] for k in ("volume", "ring", "cap", "k_outer", "bulge", "toon", "normal_mix", "edge_mix",
                                           "shade", "dynamic")},
            "paint": PT.PAINT, "atlas": 2048, "partings": {}, "ties": {}, "locks": {}, "cloth": 0.0}


class HairdoError(ValueError):
    pass


def direction(a, what):
    """The unit vector of `[theta, phi]` in degrees."""
    try:
        th, ph = float(a[0]), float(a[1])
    except (TypeError, ValueError, IndexError):
        raise HairdoError(f"{what} = {a!r}: expected [theta, phi] in degrees") from None
    return dirs(np.radians(th), np.radians(ph))


def eased(u, knots, values):
    """The curve through (knots, values) at `u`: eased from each to the next, level at every one (no overshoot)."""
    knots, values = np.asarray(knots, float), np.asarray(values, float)
    if len(knots) < 2:
        return np.full(np.shape(u), float(values[0]))
    k = np.clip(np.searchsorted(knots, u, side="right") - 1, 0, len(knots) - 2)
    t = np.clip((u - knots[k]) / np.maximum(knots[k + 1] - knots[k], 1e-9), 0.0, 1.0)
    return values[k] + (values[k + 1] - values[k]) * smoothstep(t)


def profile(lift, what):
    """`lift` as a function (metres along the way over the head (n,), the whole way) -> metres off the hull (n,):
    plain numbers are spread evenly from the root to where the lock leaves the head; pairs are [metres from the root,
    metres off the hull], level after the last (locks of different lengths then agree wherever they lie side by side)."""
    try:
        a = np.asarray(lift, float)
    except (TypeError, ValueError):
        a = np.zeros(0)
    if a.ndim == 1 and len(a) >= 1:
        return lambda s, whole: eased(s / max(whole, 1e-9), np.linspace(0.0, 1.0, len(a)), a)
    if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2 and (np.diff(a[:, 0]) > 0.0).all():
        return lambda s, whole: eased(s, a[:, 0], a[:, 1])
    raise HairdoError(f"{what} lift = {lift!r}: expected metres off the hull from the root to where the lock leaves "
                      f"the head, or [metres from the root, metres off the hull] pairs with the first rising")


def stray(d, deg, rng):
    """`d` (unit) turned up to `deg` degrees any way."""
    if deg <= 0.0:
        return d
    a = unit(np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0]))
    b = np.cross(d, a)
    r, w = np.tan(np.radians(deg)) * np.sqrt(rng.uniform(0.0, 1.0)), rng.uniform(0.0, 2.0 * np.pi)
    return unit(d + r * (np.cos(w) * a + np.sin(w) * b))


def freedom(s, length, free):
    """How free a lock of `length` hangs at arc positions `s`: 0 where it lies with its neighbours, 1 over its last
    `free` metres, easing between over half of `free` above them."""
    if free <= 0.0:
        return np.zeros_like(np.asarray(s, float))
    return smoothstep((np.asarray(s, float) - (length - 1.5 * free)) / (0.5 * free))


def arm_bones(bones):
    """The names of her arms' bones among `bones` (part Bones): the upper arms' and all that hang from them."""
    sem = bonemap.build_map({b.name: b.name for b in bones})
    out, kids = {sem[k] for k in ("arm.L", "arm.R") if k in sem}, {}
    for b in bones:
        kids.setdefault(b.parent, []).append(b.name)
    stack = list(out)
    while stack:
        for k in kids.get(stack.pop(), []):
            if k not in out:
                out.add(k)
                stack.append(k)
    return out


def cut_at(P, z):
    """The path `P` up to where it first comes down to height `z`, ending exactly there (all of it when it never
    does)."""
    below = np.flatnonzero(P[:, 2] <= z)
    if not len(below) or below[0] == 0:
        return P
    k = int(below[0])
    t = (P[k - 1, 2] - z) / max(P[k - 1, 2] - P[k, 2], 1e-12)
    return np.vstack([P[:k], P[k - 1] + t * (P[k] - P[k - 1])])


class LockHair(HH.HeadHair):
    """The lock hairdo: `plan` lays every group's locks out (paths, shapes, chains), `build` meshes and paints them."""
    DEFAULTS = DEFAULTS

    def __init__(self, ctx, fit, cfg, rig, pal):
        super().__init__(ctx, fit, cfg, rig, pal)
        self.surround = self.drape = None                   # the body and the garments the falling locks keep off
        arms = [np.asarray(ctx.land[k], float) for k in ("arm.L", "arm.R") if k in getattr(ctx, "land", {})]
        self.shoulder_z = float(np.mean([a[2] for a in arms])) if arms else None     # the shoulder joints' height
        self._statics = []                                  # the body's static colliders the chains must not overlap
        if self._body is not None:
            bones = [b for p in getattr(ctx, "parts", {}).values() for b in p.bones]
            self.surround = Surround(self._body[0].meshes, self._wear, self._body[1], cloth=float(self.cfg["cloth"]),
                                     arms=arm_bones(bones))
            self.drape = self.surround.drape()             # what a mass of hair lies on
            self._statics = [rb for p in getattr(ctx, "parts", {}).values() for rb in p.bodies if rb.mode == "static"]
        self.atlas = PT.Atlas(int(self.cfg["atlas"]))
        self._cap_cell = None
        self.more = ([], [])                                # meshes and materials beside the head hair's (plaits, bows)
        self.partings = {}                                  # [hair.partings]: name -> (directions along it, their arc)
        for name, pts in (self.cfg["partings"] or {}).items():
            if name == "hairline" or not isinstance(pts, list) or len(pts) < 2:
                raise HairdoError(f"[hair.partings] {name}: expected two or more [theta, phi] points (`hairline` is "
                                  f"the head's own line)")
            D = unit(spline(np.array([direction(p, f"[hair.partings] {name}[{k}]") for k, p in enumerate(pts)]), 64))
            self.partings[name] = (D, arclen(D))
        self.ties = {}                                      # [hair.ties]: name -> {dir, out, off, stem}
        for name, raw in (self.cfg["ties"] or {}).items():
            if not isinstance(raw, dict) or raw.get("at") is None:
                raise HairdoError(f"[hair.ties] {name}: expected a table with `at = [theta, phi]`")
            t = SP.merge(TIE, raw)
            d = direction(t["at"], f"[hair.ties] {name} at")
            out = unit(d + [0.0, 0.0, -1.0]) if t["out"] is None else direction(t["out"], f"[hair.ties] {name} out")
            self.ties[name] = {"dir": d, "out": out, "off": float(t["off"]), "stem": float(t["stem"])}

    def lowest(self):
        return float(self.fit.chin[2]) - 0.05              # until the locks are planned (`plan` spans to their tips)

    # ---------------------------------------------------------------- where a lock is rooted and combed
    def on_line(self, name, x, what):
        """The direction of point `x` of line `name`: on the hairline at `x` degrees round the head (HAIRLINE_IN inside
        it), or `x` of the way along a parting of `[hair.partings]` (0 its start .. 1 its end)."""
        if name == "hairline":
            th = np.radians(float(x))
            return dirs(th, float(self.fit.cap_phi(th)) - HAIRLINE_IN)
        if name not in self.partings:
            raise HairdoError(f"{what}: no parting {name!r} in [hair.partings] (it has: "
                              f"{', '.join(self.partings) or 'none'})")
        D, s = self.partings[name]
        at = float(np.clip(x, 0.0, 1.0)) * s[-1]
        return unit(np.array([np.interp(at, s, D[:, k]) for k in range(3)]))

    def rooted(self, a, what):
        """A guide's `root` -> (its direction, (line, place on it) or None for plain angles)."""
        if isinstance(a, (list, tuple)) and len(a) == 2 and isinstance(a[0], str):
            try:
                x = float(a[1])
            except (TypeError, ValueError):
                raise HairdoError(f"{what} = {a!r}: expected [line, place on it]") from None
            return self.on_line(a[0], x, what), (a[0], x)
        return direction(a, what), None

    def tie(self, name, what):
        """The tie `name` of `[hair.ties]`."""
        if name not in self.ties:
            raise HairdoError(f"{what}: no tie {name!r} in [hair.ties] (it has: {', '.join(self.ties) or 'none'})")
        return self.ties[name]

    def tied(self, guides, key, what):
        """The tie every guide's `key` (`root` or `to`) names, None when none names one: a group is tied whole."""
        names = {x.get(key) for x in guides if isinstance(x.get(key), str)}
        if not names:
            return None
        if len(names) > 1 or not all(isinstance(x.get(key), str) for x in guides):
            raise HairdoError(f"{what}: every guide's `{key}` names the group's one tie, or none names a tie")
        return self.tie(names.pop(), f"{what} {key}")

    def course(self, guide, root, what):
        """CTRL unit vectors along a guide's way over the head: from its root through its `via` points to its `to`
        (a place, or the tie it is gathered to), evenly spaced along the curve through them."""
        pts = [direction(v, f"{what} via[{k}]") for k, v in enumerate(guide.get("via") or [])]
        to = guide.get("to")
        pts.append(self.tie(to, f"{what} to")["dir"] if isinstance(to, str) else direction(to, f"{what} to"))
        return unit(spline(np.array([root, *pts]), CTRL))

    def combed(self, C, line):
        """The course `C` (CTRL unit vectors) of a lock whose group is rooted on `line` (points of the hull), leaving
        its root as hair is combed from a parting: of the ways it can go first (LEAVE, or 0.4 of its way) it takes the
        one that leads it away from the line and toward where it is going, each way weighed by both. So it leaves the
        middle of a parting across, leaning toward its goal, and the parting's end straight toward its goal."""
        root = C[0]
        reach = min(LEAVE, 0.4 * float(np.arccos(np.clip(np.einsum("ij,ij->i", C[:-1], C[1:]), -1.0, 1.0)).sum()))
        e1 = unit(np.cross(root, [0.0, 0.0, 1.0] if abs(root[2]) < 0.9 else [1.0, 0.0, 0.0]))
        e2 = np.cross(root, e1)
        w = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        ways = unit(root + np.tan(reach) * (np.cos(w)[:, None] * e1 + np.sin(w)[:, None] * e2))
        aim = C[CTRL // 4]
        goal = np.clip((ways - root) @ unit(aim - root * (root @ aim)), 0.0, None)
        far = self.away(ways, line)
        first = unit(((far / far.max()) ** 2 * goal) @ ways)
        rest = [p for p in C[1:] if np.arccos(np.clip(root @ p, -1.0, 1.0)) > 1.5 * reach] or [C[-1]]
        return unit(spline(np.array([root, first, *rest]), CTRL))

    # ---------------------------------------------------------------- one lock's path
    def hint(self, P, mass=False):
        """The way a lock faces along its path P: out from the head's centre above it, level out from the axis through it
        below (a lock hanging down a back faces away from the back), and where it lies on the body or a garment the way
        that surface faces (flat on a shoulder, it faces up). The body's say fades in over TURN below the neck seam and
        the facing is averaged along the path, so a lock turns over a few centimetres, never between two points."""
        c = self.skull.center
        axis = np.stack([np.full(len(P), c[0]), np.full(len(P), c[1]), np.minimum(P[:, 2], c[2])], -1)
        n = unit(P - axis)
        if self.surround is not None:                       # a mass faces as the drape does: no fold turns it
            d, out = (self.drape.over(P), self.drape.normal(P)) if mass else self.surround.away(P)
            w = smoothstep((NEAR - d) / (NEAR - REST)) * smoothstep((self.surround.z_seam - P[:, 2]) / TURN)
            n = unit(n * (1.0 - w)[:, None] + out * w[:, None])
        if len(n) >= 5:
            k = np.hanning(2 * int(round(TURN / STEP)) + 3)[1:-1]
            pad = len(k) // 2
            m = np.concatenate([np.repeat(n[:1], pad, 0), n, np.repeat(n[-1:], pad, 0)], 0)
            n = unit(np.stack([np.convolve(m[:, a], k / k.sum(), mode="valid") for a in range(3)], -1))
        return n

    def on_hull(self, D):
        """The hull's points in the directions `D` (n, 3)."""
        th, ph = angles(D)
        return self.skull.point(D, self.vol(th, ph))

    def away(self, D, line):
        """Metres from the hull's points in the directions `D` (n, 3) to a root line (points of the hull)."""
        return np.linalg.norm(self.on_hull(D)[:, None] - line[None], axis=2).min(1)

    def lying(self, D, x, lie):
        """The surface a group's locks lie in on the head: metres off the skin in the directions `D` (n, 3), `x` (n,)
        the metres from the root there (along the lock, or from the line the group is rooted on: `lie["line"]`). It is
        the hull, `lie`'s `lift` off it less `lower`, `layer` steps and `over` further. At the root it lies under the
        scalp cap and comes out of it over ROOT: a row of roots shows as one clean line, whatever way each lock's end
        is cut."""
        th, ph = angles(D)
        off = (self.vol(th, ph) + LAYER_STEP * lie["layer"] + lie["swell"] * lie["lift"](x, lie["whole"]) -
               lie["lower"] + lie["over"])
        out = smoothstep(x / ROOT[0])
        return np.maximum((1.0 - out) * (self.cap_off(th, ph) - ROOT[1]) + out * off, MIN_OFF)

    def on_head(self, C, lie):
        """The path over the head through the control directions C, in the surface `lie` (`lying`; its `whole`, the
        length of this way over the head, is set here)."""
        turn = float(np.arccos(np.clip(np.einsum("ij,ij->i", C[:-1], C[1:]), -1.0, 1.0)).sum())
        n = max(int(turn * 0.15 / STEP) + 2, 6)               # a point every STEP or so (the hull is about 0.15 m out)
        D = unit(spline(C, n))
        s = arclen(self.on_hull(D))
        lie["whole"] = float(s[-1])
        return self.skull.point(D, self.lying(D, s if lie["line"] is None else self.away(D, lie["line"]), lie))

    def fall(self, P, length, g, cut=None):
        """`P` gone on for `length` metres as hair falls: its heading turns to straight down (`stiff` of the way each
        step), pulled in by `tuck`, out by `flare` and back by `lean` (those two until it is past her shoulders: they
        fade out over SETTLE down to the shoulder joints, below them the hair hangs); `hug` draws it in to her face
        while it passes it; it stays off the head, and what it meets of the body and the garments (`rest` off them:
        REST, more for a thick plait) it slides over. With `cut` (a height) it falls until it comes down to it and
        ends there, whatever its `length` (`P` itself is cut when it does)."""
        if cut is not None:
            if P[:, 2].min() <= cut:
                return cut_at(P, cut)
            length = FALL_MAX
        rest = float(g.get("rest", REST))
        n = int(round(length / STEP))
        if n < 1:
            return P
        c = self.skull.center
        p, h = P[-1], unit(P[-1] - P[-2])
        out = []
        wall = None if self.drape is None else float(np.hypot(p[0] - self.drape.ax, p[1] - self.drape.ay))
        for _ in range(n):
            rxy = unit(np.array([p[0] - c[0], p[1] - c[1], 0.0]))
            past = 1.0 if self.shoulder_z is None else smoothstep((p[2] - self.shoulder_z) / SETTLE)
            down = unit(np.array([0.0, g["lean"] * past, -1.0]) + (g["flare"] * past - g["tuck"]) * rxy)
            h = unit(h + g["stiff"] * (down - h))
            q = self.skull.push_out(p + STEP * h, 0.008)
            if g["hug"] is not None:                        # drawn in to the face as it passes it, down to the chin
                near = float(smoothstep((q[2] - self.fit.chin[2]) / HUG_FADE))
                off = float(self.skull.height_dist(q)) - float(g["hug"])
                if near > 0.0 and off > 0.0:
                    q = q - unit(q - c) * off * near
            if self.drape is not None and g["mass"]["enabled"]:     # a mass lies on the drape, a column at a time
                lay = self.drape.clear(q[None], rest)[0]
                r = float(np.hypot(lay[0] - self.drape.ax, lay[1] - self.drape.ay))
                if self.shoulder_z is not None and lay[2] > self.shoulder_z and r < wall:   # under the head it
                    axis = np.array([self.drape.ax, self.drape.ay])
                    lay[:2] = axis + (lay[:2] - axis) * (wall / max(r, 1e-9))
                wall = max(wall, float(np.hypot(lay[0] - self.drape.ax, lay[1] - self.drape.ay)))   # hangs straight
                if np.linalg.norm(lay - q) > 1e-9:
                    q, h = lay, unit(lay - p)
            elif self.surround is not None:
                d, way = self.surround.away(q[None])
                if d[0] < rest:
                    q = q + way[0] * (rest - d[0])
                    h = unit(q - p)                         # it goes on the way the surface turned it
            p = q
            out.append(p)
            if cut is not None and p[2] <= cut:
                break
        P = np.vstack([P, np.array(out)])
        return P if cut is None else cut_at(P, cut)

    def style(self, P, n_head, wave, flip, phase, mass=False, rest=REST):
        """The fallen part of `P` (its points from `n_head` on) waved and its end flipped (a mass's no closer than
        `rest` to the drape)."""
        if len(P) - n_head < 3 or (wave["amp"] <= 0.0 and flip["out"] == 0.0):
            return P
        Q = P.copy()
        F = P[n_head - 1:]
        s = arclen(F)
        L = float(s[-1])
        T = unit(np.gradient(F, axis=0))
        out = self.hint(F, mass)
        side = unit(np.cross(T, out))
        d = np.zeros_like(F)
        if wave["amp"] > 0.0:
            a = wave["amp"] * smoothstep(s / max(wave["start"], 1e-6)) * (1.0 + wave["grow"] * s / max(L, 1e-6))
            w = 2.0 * np.pi * s / max(wave["length"], 1e-3) + phase
            d += side * (a * np.sin(w))[:, None] + out * (wave["depth"] * a * np.cos(w))[:, None]
        if flip["out"] != 0.0 and flip["len"] > 0.0:
            u = np.clip((s - (L - flip["len"])) / flip["len"], 0.0, 1.0)
            d += out * (flip["out"] * u * u)[:, None]
        Q[n_head - 1:] = self.skull.push_out(F + d, 0.006)
        if mass and self.drape is not None:
            Q[n_head - 1:] = self.drape.clear(Q[n_head - 1:], rest)
        return Q

    # ---------------------------------------------------------------- a group's locks
    def group(self, name, g):
        """The locks of group `name`, as HH.Lock with `extra` {s_leave (where it leaves the head, metres along it),
        free, finger, root_len, wide (its width over its full width at ROOM points from the root to s_leave; None with
        a `width` of its own), lie (the surface it lies in on the head: `lying`; None for a tail, which hangs from its
        tie), group, under, gathered (it ends in a tie)}."""
        what = f"[hair.locks.{name}]"
        guides = g["guides"]
        if not isinstance(guides, list) or not guides or not all(isinstance(x, dict) for x in guides):
            raise HairdoError(f"{what}: guides = [{{root, to, length}}, ...] is required")
        if g["family"] not in FAMILIES:
            raise HairdoError(f"{what}: family {g['family']!r} is none of {', '.join(FAMILIES)}")
        if g["tip"] not in ("point", "round"):
            raise HairdoError(f"{what}: tip {g['tip']!r} is none of point, round")
        cut = None if g["cut"] is None else float(g["cut"])
        mass = bool(g["mass"]["enabled"])
        bow = g["bow"] if g["bow"]["enabled"] else None
        if bow is not None and (g["plait"]["enabled"] or not 0.0 < float(bow["above"]) < 0.5):
            raise HairdoError(f"{what}: a bow is tied round a lock (not a plait) `above` metres over its end, "
                              f"0 .. 0.5")
        tail, gathered = self.tied(guides, "root", what), self.tied(guides, "to", what)
        lengths = np.array([float(x.get("length", 0.0)) for x in guides])
        own = {k: np.array([float(x.get(k, g[k])) for x in guides]) for k in NUMBERS if g[k] is not None or
               all(k in x for x in guides)}
        if g["width"] is None and any("width" in x for x in guides) and "width" not in own:
            raise HairdoError(f"{what}: give `width` in the group or in every guide")
        line = None
        if tail is not None:                                # it leaves the head at the tie, the way `out` points
            if "width" not in own or any("to" in x or "via" in x for x in guides):
                raise HairdoError(f"{what}: a tail hung from a tie has a `width` and no `to`: it leaves the head "
                                  f"there, `out = [theta, phi]` the way it goes")
            aims = np.array([tail["out"] if x.get("out") is None else direction(x["out"], f"{what} guides[{k}] out")
                             for k, x in enumerate(guides)])
            D = tail["dir"][None]
            stem = self.on_hull(D)[0] + (tail["off"] + LAYER_STEP * int(g["layer"])) * D[0]
        else:
            if gathered is not None and lengths.any():
                raise HairdoError(f"{what}: hair gathered to a tie ends in it: its guides have no `length`")
            roots = [self.rooted(x.get("root"), f"{what} guides[{k}] root") for k, x in enumerate(guides)]
            keys = [r[1] for r in roots]
            if all(k is not None for k in keys) and len({k[0] for k in keys}) == 1:
                # every guide rooted on one line: the group's surface is told by the distance from that stretch of it
                at = np.linspace(min(k[1] for k in keys), max(k[1] for k in keys), 64)
                line = self.on_hull(np.array([self.on_line(keys[0][0], x, what) for x in at]))
            courses = np.array([self.course(x, roots[k][0], f"{what} guides[{k}]") for k, x in enumerate(guides)])
            lift = profile(g["lift"], what)
            if gathered is not None:                        # where its locks run into the tie
                D = gathered["dir"][None]
                mouth = self.on_hull(D)[0] + gathered["off"] * D[0]
        weave = g["plait"]                                  # a plait's thickness over its width: the weave's, or its own
        flat = float(weave.get("thick", BR.DEFAULTS["thick"])) / float(weave.get("width", BR.DEFAULTS["width"]))
        rng = self.ctx.rng_for(f"hair.locks.{name}")
        jit, count = g["jitter"], int(g["count"])
        rows = [(False, count, 0.0)]
        if g["under"]["enabled"] and count > 1:
            rows.insert(0, (True, count - 1, 0.5))
        out = []
        last = len(guides) - 1
        rhythm = [float(x) for x in g["rhythm"]] or [1.0]
        for under, n, shift in rows:
            places = [0.0] if n == 1 else [(i + shift) / (count - 1) * last for i in range(n)]
            planned = []
            for i, t in enumerate(places):
                k = min(int(t), max(last - 1, 0))
                f = t - k if last else 0.0
                j = min(k + 1, last)
                num = {key: float((1.0 - f) * v[k] + f * v[j]) for key, v in own.items()}
                if tail is not None:
                    aim = stray(unit((1.0 - f) * aims[k] + f * aims[j]), jit["to"], rng)
                    d_to, lie = tail["dir"], None
                    head = stem + np.outer(np.linspace(0.0, tail["stem"], 3), aim)
                else:
                    C = unit((1.0 - f) * courses[k] + f * courses[j])
                    a, b = roots[k][1], roots[j][1]
                    if a is not None and b is not None and a[0] == b[0]:    # both on one line: so is this lock
                        exact = self.on_line(a[0], (1.0 - f) * a[1] + f * b[1], what)
                        C = unit(C + np.outer(np.linspace(1.0, 0.0, CTRL) ** 2, exact - C[0]))
                    else:
                        C[0] = stray(C[0], jit["root"], rng)
                    if gathered is None:
                        C[-1] = stray(C[-1], jit["to"], rng)
                    else:
                        num["free"] = num["tip_len"] = 0.0  # it ends in the tie with the others, no point of its own
                    if line is not None:
                        C = self.combed(C, line)
                    d_to = C[-1]
                beat = 1.0 if under else rhythm[i % len(rhythm)]
                length = float((1.0 - f) * lengths[k] + f * lengths[j]) * (1.0 + jit["length"] * rng.uniform(-1.0, 1.0))
                rise = jit["lift"] * rng.uniform(-1.0, 1.0) + RIDE * 2.0 * max(1.0 - beat, 0.0)
                if under:
                    length *= g["under"]["length"]
                if tail is None:
                    lie = {"lift": lift, "swell": num["swell"], "layer": int(g["layer"]), "line": line,
                           "lower": -g["under"]["lift"] if under else 0.0, "over": HAIR * (i % 3)}
                    head = self.on_head(C, lie)
                rest = REST - (g["under"]["lift"] if g["under"]["enabled"] and not under else 0.0)  # over its row
                if g["plait"]["enabled"]:               # a plait is thick: its middle line rests further off her
                    if "width" not in num:
                        raise HairdoError(f"{what}: a plait needs a `width`")
                    rest = max(REST, 0.5 * flat * num["width"] + 0.004)
                own_cut = cut
                if cut is not None and jit["cut"] > 0.0:            # a cut whose locks end a little apart
                    own_cut = cut + float(jit["cut"]) * rng.uniform(-1.0, 1.0)
                if gathered is None:
                    P = self.fall(head, length, {**g, **num, "rest": rest}, cut=own_cut)
                else:                                   # it runs on into the tie, the way a tail leaves it
                    P = np.vstack([head, mouth + np.outer(np.linspace(0.0, gathered["stem"] + THROAT, 4)[1:],
                                                          gathered["out"])])
                place = (i + shift) if n > 1 else 0.0       # the rows of a group wave together
                phase = g["wave"]["phase"] + g["wave"]["drift"] * place + (0.0 if under else jit["phase"] * rng.uniform(-1.0, 1.0))
                P = self.style(P, len(head), g["wave"], g["flip"], phase, mass, rest)
                s_leave = float(arclen(P[:len(head)])[-1])
                P = smooth_polyline(resample(P, max(int(arclen(P)[-1] / STEP) + 1, 6)), 1)
                along = resample(P, ROOM_ALONG) if mass else resample(head, ROOM)      # where the room is measured
                planned.append((P, s_leave, num, d_to, along, beat, rise, lie))
            for i, (P, s_leave, num, d_to, along, beat, rise, lie) in enumerate(planned):
                wide = None
                if "width" in num:
                    w = num["width"]
                else:                                       # the room to its neighbours, all along its way on the head
                    #                                         (on its whole way down, for a mass):
                    near = [np.linalg.norm(planned[q][0][None] - along[:, None], axis=2).min(1)  # to the nearest of theirs
                            for q in (i - 1, i + 1) if 0 <= q < len(planned)]
                    if not near:
                        raise HairdoError(f"{what}: a single lock needs a `width`")
                    room = np.mean(near, axis=0)
                    w = float(room.max()) * g["overlap"]
                    wide = np.maximum(room / max(float(room.max()), 1e-9), NARROW)
                w *= beat * (g["under"]["width"] if under else 1.0) * (1.0 + jit["width"] * rng.uniform(-1.0, 1.0))
                L = float(arclen(P)[-1])
                free = min(num["free"], 0.6 * L)
                s = arclen(P)                               # where it hangs free it rises off the shared surface
                P = P + self.hint(P, mass) * (rise * freedom(s, L, free))[:, None]
                thick = float(np.clip(num["thick"] * w * num["finger"], THICK[0], max(float(g["thick_max"]), THICK[0])))
                if g["plait"]["enabled"]:
                    thick = flat * w
                tip_start = float(np.clip(1.0 - num["tip_len"] / max(L, 1e-6), 0.15, 0.97 if g["tip"] == "point" else
                                          0.999))
                if self.surround is not None and gathered is None and L - s_leave > 0.02 and not mass:
                    P = self.surround.keep_off(P, self.hint, w, thick, num["cup"], tip_start, HH.BODY_GAP,
                                               first=max(int(s_leave / STEP) - 1, 1), tip=g["tip"])
                out.append(HH.Lock(f"{name}{'_u' if under else ''}{i}", P, w, thick, tip=g["tip"], tip_start=tip_start,
                                   tip_power=num["tip_power"], root_w=num["root_w"], curv=num["cup"],
                                   layer=int(g["layer"]), theta=float(angles(d_to)[0]), dark=under,
                                   extra={"s_leave": s_leave, "free": free, "finger": num["finger"], "lie": lie,
                                          "root_len": num["root_len"], "wide": wide, "group": name, "under": under,
                                          "gathered": gathered is not None, "bow": None if under else bow,
                                          "mass": dict(g["mass"]) if mass else None}))
        return out

    def clump(self, lk):
        """The lock's mesh: as wide as the room to its neighbours and paper-flat where it lies with them, every point
        of it in the group's surface (`lying`); `finger` of the width, its own thickness and its own `cup` where it
        hangs free. A lock of a `mass` has its own thickness and cup from where it leaves the head (taking it over
        HANG), thinning toward its end, and is as wide as the room to its neighbours all down its way."""
        P = np.asarray(lk.path, float)
        s = arclen(P)
        L = float(s[-1])
        e = lk.extra
        hint = self.hint(P, e.get("mass") is not None)
        lie, c = e["lie"], self.skull.center
        loose = freedom(s, L, e["free"])
        on = (1.0 - loose) * (1.0 - smoothstep((s - (e["s_leave"] - 0.01)) / 0.03))    # 1 where it lies on the head
        if lie is None:                                     # a tail hangs from its tie: none of it lies on the head
            on = np.zeros_like(s)

        def place(mid, sp):
            m = float(np.interp(sp * L, s, on))
            if m <= 0.0:
                return mid
            D = unit(mid - c)
            x = np.full(len(mid), sp * L) if lie["line"] is None else self.away(D, lie["line"])
            return mid + m * (self.skull.point(D, self.lying(D, x, lie)) - mid)
        wide, bow = e.get("wide"), e.get("bow")
        thin = None if e.get("mass") is None else float(e["mass"]["thin"])
        over = None if e.get("mass") is None or e["mass"]["over"] is None else float(e["mass"]["over"])
        tie = None if bow is None else L - float(bow["above"])

        def free(sp):
            return freedom(np.asarray(sp, float) * L, L, e["free"])

        def own(sp):
            """How much of its own form (thickness, cup) the lock has: where it hangs free, or (a mass) off the head."""
            if thin is None:
                return free(sp)
            return np.maximum(free(sp), smoothstep((np.asarray(sp, float) * L - e["s_leave"]) / HANG))

        def tied(sp, fan):
            """What a bow's tie leaves of the lock's width (or thickness): drawn in to `pinch` over TIE_EASE above the
            tie, out to `fan` over the first third below it (the tuft's round end tapers it after that)."""
            if tie is None:
                return 1.0
            x, pinch = np.asarray(sp, float) * L, float(bow["pinch"])
            above = 1.0 - (1.0 - pinch) * smoothstep((x - (tie - TIE_EASE)) / TIE_EASE)
            below = pinch + (fan - pinch) * smoothstep((x - tie) / max(0.3 * (L - tie), 1e-6))
            return np.where(x <= tie, above, below)

        def across(sp):
            f = (1.0 - (1.0 - e["finger"]) * free(sp)) * tied(sp, float(bow["fan"]) if bow else 1.0)
            if wide is None:
                return f
            reach = L if thin is not None else e["s_leave"]
            return f * np.interp(np.asarray(sp, float) * L, np.linspace(0.0, reach, len(wide)), wide)

        def deep(sp):
            f = (FLAT + (1.0 - FLAT) * own(sp)) * tied(sp, float(bow["pinch"]) if bow else 1.0)
            if thin is None:
                return f
            x = np.asarray(sp, float) * L
            down = np.clip((x - e["s_leave"]) / max(L - e["s_leave"], 1e-6), 0.0, 1.0) if over is None else \
                np.clip((x - (L - over)) / max(over, 1e-6), 0.0, 1.0)
            return f * (1.0 + (thin - 1.0) * down)
        return sweep(P, hint, lk.width, lk.thick, tip=lk.tip, tip_start=lk.tip_start, tip_power=lk.tip_power,
                     root_w=lk.root_w if wide is None else 1.0,
                     root_len=float(np.clip(e["root_len"] / max(L, 1e-6), 0.01, 0.5)), curv=lk.curv,
                     k_outer=int(self.cfg["k_outer"]) if lk.width >= 0.018 else 5, bulge=self.cfg["bulge"],
                     ring_step=0.010 if tie is None else 0.005, width_fn=across, thick_fn=deep, curv_fn=own,
                     place=place)

    def proxy_normals(self, P):
        """HeadHair's shading proxy with its axis going on down to the lowest hair: what hangs below the head is shaded
        as a column round her, not as the underside of a ball."""
        c = self.fit.center
        return HH.shade_proxy(P, c, c[2] + self.cfg["shade"]["z_hi"], min(c[2] + self.cfg["shade"]["z_lo"], self.z_bot))

    def radii(self, lk, bones, s_root, radius):
        """A chain's capsule radius per bone: `radius`, or what the room to the static colliders of the parts built
        before the hair leaves at rest, CLEAR kept (MIN_RADIUS at least): a capsule that overlaps one at rest is thrown
        out by physics."""
        if not self._statics:
            return radius
        s = arclen(lk.path)
        sh = np.linspace(s_root, s[-1], bones + 1)
        pts = np.stack([np.interp(sh, s, lk.path[:, k]) for k in range(3)], -1)
        room = [CC.capsule_clearance(self._statics, pts[k], pts[k + 1], 0.0)[0] for k in range(bones)]
        return np.clip(np.asarray(room) - CLEAR, MIN_RADIUS, radius)

    # ---------------------------------------------------------------- the whole hairdo
    def plan(self):
        """Every enabled group's locks, the chains made: [(group name, group, [Lock])] in layer order."""
        groups = []
        for name, raw in (self.cfg["locks"] or {}).items():
            if not isinstance(raw, dict):
                raise HairdoError(f"[hair.locks] {name}: expected a table")
            g = SP.merge(GROUP, raw)
            if g["enabled"]:
                groups.append((name, g))
        if not groups:
            raise HairdoError("[hair] style = \"locks\" needs groups of locks: [hair.locks.<name>] (docs/hair_part.md)")
        groups.sort(key=lambda x: (int(x[1]["layer"]), x[0]))
        planned = [(name, g, self.group(name, g)) for name, g in groups]
        low = min(float(lk.path[:, 2].min()) for _, _, locks in planned for lk in locks)
        self.span_to(min(low - 0.01, self.z_bot))
        count = {}
        for name, g, locks in planned:
            ch = g["chain"]
            bones, every = int(ch["bones"]), max(int(ch["every"]), 1)
            if bones < 1:
                continue
            stem, kind = FAMILIES[g["family"]]

            def hangs(lk):
                return not lk.extra["gathered"] and arclen(lk.path)[-1] - lk.extra["s_leave"] > 0.02
            main = [lk for lk in locks if not lk.extra["under"] and hangs(lk)]
            made = []
            for lk in main[(len(main) - 1) % every // 2::every]:
                tag = stem
                if g["family"] in SIDED:
                    tag += "左" if lk.path[-1, 0] >= 0.0 else "右"
                count[tag] = count.get(tag, 0) + 1
                base = f"{tag}{count[tag]}_"
                s_root = max(lk.extra["s_leave"] - 0.01, 0.0)
                c = self.make_chain(base, lk, kind, bones, s_root, self.radii(lk, bones, s_root, float(ch["radius"])))
                self.chains[base.rstrip("_")] = c
                lk.chain = base.rstrip("_")
                made.append(lk)
            for lk in locks:
                if not lk.chain and made and hangs(lk):
                    lk.chain = min(made, key=lambda m: float(np.linalg.norm(m.path[-1] - lk.path[-1]))).chain
        return planned

    def cap_uv(self, theta, z):
        """The scalp cap in its cell: across it the way round the head (mirrored left and right), down it the height."""
        c = self._cap_cell
        f = np.clip((self.z_top - np.asarray(z, float)) / max(self.z_top - self.z_bot, 1e-6), 0.0, 1.0)
        return c.uv(2.0 * np.abs(np.asarray(theta, float)) / np.pi - 1.0, f)

    def emit(self, acc, lk, chain=None):
        """The lock meshed into `acc`: a thin shell whose two sides share no vertex (hair_geo.split_shell), each shaded
        `normal_mix` of the way like the head-shaped proxy where it lies with its neighbours and OWN less where it
        hangs free (its own form shows), its UVs its cell of the atlas. Its two edges are kept for the painter
        (`extra["edges"]`: the metres from the root, the left edge's points, the right edge's)."""
        st = self.clump(lk)
        n, R = st.rings, st.R
        ring, tip = st.verts[:n * R].reshape(n, R, 3), st.verts[n * R:n * R + 1]
        lk.extra["edges"] = (np.append(st.arc[:n * R:R], [st.length] * len(tip)),
                             np.vstack([ring[:, int(np.argmin(st.q[:R]))], tip]),
                             np.vstack([ring[:, int(np.argmax(st.q[:R]))], tip]))
        V, faces, q, arc, side = split_shell(st)
        uv = lk.extra["cell"].uv(q, arc / max(st.length, 1e-9))
        w = {"頭": np.ones(len(V))} if chain is None else chain.weights(V)
        own = freedom(arc, st.length, lk.extra["free"])
        if lk.extra.get("mass") is not None:                # a mass shows its strands' own form where it hangs
            own = np.maximum(own, smoothstep((arc - lk.extra["s_leave"]) / HANG))
        mix = self.cfg["normal_mix"] * (1.0 - OWN * own)
        acc.add(V, faces, uv, 0, weights=w, normals=shell_normals(V, faces, side, self.proxy_normals(V), mix))
        return st

    def paint(self, locks):
        """Every lock's cell and the cap's painted: the atlas picture. A point of a cell is painted by where it is on
        the head (its height, its angle round the head), so locks that lie side by side carry one picture."""
        look = self.cfg["paint"]
        ring = self.cfg["ring"]
        sheen = None
        if ring["enabled"]:
            sheen = PT.Sheen(self.z_top - ring["drop"], ring["width"] * look["sheen_width"], look["tooth"], look["glow"],
                             self.ctx.rng_for("hair.paint.sheen"), fall=ring["fall"], turn=ring["turn"])
        bands = []
        for k, raw in enumerate(look["bands"]):
            bd = SP.merge(PT.BAND, raw)
            bands.append((PT.Sheen(self.z_top - bd["drop"], bd["width"], bd["tooth"], 0.0,
                                   self.ctx.rng_for(f"hair.paint.band{k}"), fall=bd["fall"], turn=bd["turn"]),
                          PT.colour(self.pal, bd["colour"]), bd["strength"]))
        c = self.skull.center
        for lk in locks:
            cell = lk.extra["cell"]
            arc, left, right = lk.extra["edges"]
            at = (np.arange(cell.h) + 0.5) / cell.h * float(arc[-1])
            a, b = (np.stack([np.interp(at, arc, e[:, k]) for k in range(3)], -1)[:, None, :] for e in (left, right))
            u = np.linspace(0.0, 1.0, cell.w)[None, :, None]
            X = a * (1.0 - u) + b * u
            rgb = PT.paint_lock(cell, X[..., 2], np.arctan2(X[..., 0] - c[0], c[1] - X[..., 1]), at,
                                freedom(at, float(arc[-1]), lk.extra["free"]), self.pal, look,
                                self.ctx.rng_for(f"hair.paint.{lk.name}"), self.z_top, self.z_bot, sheen,
                                under=lk.extra["under"], bands=bands, ends=not lk.extra["gathered"])
            self.atlas.put(cell, rgb)
        self.atlas.put(self._cap_cell, PT.paint_cap(self._cap_cell, self.pal, look, self.ctx.rng_for("hair.paint.cap")))
        return self.atlas.image(self.pal["base"] * look["cap"])

    def plait(self, lk, g, chain):
        """The lock as a three-strand plait (hair_braids' weave along the lock's path): woven from where the lock
        leaves the head (or `plait.start` metres from its root) to a tie a tuft's length short of its end, a tuft of
        loose locks below the tie (`plait.ties`: "end", "both" ties its top too, "none"). It is the lock's `width`
        wide, one crossing every `plait.pitch` metres; hair_braids' own weave keys given in the `plait` table (taper,
        strand_wid, tuft_len ... for a plait 0.034 wide) scale with the width. Hair that ends in another colour
        (`[hair.paint] tips`) plaits into it: the plait's last `plait.tip_len` metres (the tips' `length` unless
        given) and its tuft. Returns ([(block, material 0 hair or 1 the ties' ribbon, weights)], the weave's keys)."""
        p = g["plait"]
        what = f"[hair.locks.{lk.extra['group']}] plait"
        c = SP.merge(BR.DEFAULTS, {**PLAIT, **{k: v for k, v in p.items() if k in BR.DEFAULTS}})
        scale = lk.width / c["width"]
        pitch = PITCH * lk.width if p["pitch"] is None else float(p["pitch"])
        ties = {"end": (False, True), "both": (True, True), "none": (False, False)}.get(p["ties"])
        if ties is None:
            raise HairdoError(f"{what}: ties = {p['ties']!r} is none of end, both, none")
        P = np.asarray(lk.path, float)
        s = arclen(P)
        start = lk.extra["s_leave"] if p["start"] is None else float(p["start"])
        woven = float(s[-1]) - c["tuft_len"] * scale - start
        if woven < 3.0 * pitch:
            raise HairdoError(f"{what}: {woven * 100:.0f} cm from its start to its tuft is too short to plait "
                              f"(a crossing every {pitch * 100:.1f} cm)")
        c["lobes"] = int(round(woven / pitch))
        tips = self.cfg["paint"]["tips"]
        if tips["colour"]:
            reach = float(tips["length"] if p.get("tip_len") is None else p["tip_len"])
            c["tips"] = (PT.colour(self.pal, tips["colour"]), min(reach / woven, 1.0), float(tips["strength"]))
        T, B, N = frames(P, self.hint(P))
        plan = Plan(1.0 if P[-1, 0] >= 0.0 else -1.0, P, s - start, T, B, N, (0.0, woven), float(s[-1]) - start)
        rng = self.ctx.rng_for(f"hair.plait.{lk.name}")
        half = 0.5 * c["band_h"] * scale
        blocks = [(b, 0) for b in PL.weave_blocks(PL.weave_strands(plan, c, scale, rng, half), c, TX.plait_uv, rng)]
        blocks += [(b, 0) for b in PL.tuft_blocks(plan, c, scale, rng, TX.strand_uv)]
        for sigma, tied in zip((0.0, woven), ties):
            if tied:
                O, Tq, Bq, Nq = plan.at(sigma)
                f = float(PL.envelope(c, scale, plan, sigma, half))
                a = 0.5 * lk.width * f + 0.003 * scale
                b = 0.5 * c["thick"] * scale * (2.0 * c["weave_depth"] + c["strand_thick"]) * f + 0.003 * scale
                blocks.append((BOW.band_block(O, Tq, Bq, Nq, a, b, c["band_h"] * scale, c["band_thick"] * scale), 1))
        return [(b, mat, {"頭": np.ones(len(b.verts))} if chain is None else chain.weights(b.verts))
                for b, mat in blocks], c

    def bow(self, lk, st, chain):
        """The bow tied round lock `lk` (`st` its mesh, from `clump`) `above` metres over its end: a band round the
        gathered lock, a knot on its face, two loops across it drooping a little and two short tails down its face
        (hair_braids_bow's pieces in BOW_SHAPE's proportions of `span`), every piece riding the lock's chain where it
        is tied. Returns ([Block], {bone: weight} at the tie, info)."""
        b, sh = lk.extra["bow"], BOW_SHAPE
        span = float(b["span"])
        R, n = st.R, st.rings
        arcs = st.arc[:n * R:R]
        P = np.asarray(lk.path, float)
        s = arclen(P)
        T, B, N = frames(P, self.hint(P, lk.extra.get("mass") is not None))

        def at(x):
            """(the middle of the lock's face-to-back, down it, across it, out of its face, its half width, half its
            depth) where it is `x` metres from its root: its mesh's nearest ring measured in the path's frame."""
            i, k = int(np.argmin(np.abs(arcs - x))), int(np.argmin(np.abs(s - x)))
            d = st.verts[i * R:(i + 1) * R] - st.ctr[i * R]
            t, a, f = T[k], B[k], N[k]
            front, back = float((d @ f).max()), float((d @ f).min())
            return st.ctr[i * R] + f * 0.5 * (front + back), t, a, f, float(np.abs(d @ a).max()), 0.5 * (front - back)
        tie = st.length - float(b["above"])
        O, t, a, f, half_w, half_d = at(tie)
        kt, kw, kd = (x * span for x in sh["knot"])
        blocks = [BOW.band_block(O, t, a, f, half_w + 0.0012, half_d + 0.0012, sh["band_h"] * span,
                                 sh["band_thick"] * span)]
        kc = O + f * (half_d + 0.0012 + 0.55 * kd)
        blocks.append(BOW.knot_block(kc, t, f, kt, kw, kd))
        droop = np.radians(sh["droop"])
        for side in (1.0, -1.0):                            # the loops, across the lock, drooping toward its end
            way = unit(np.cos(droop) * side * a + np.sin(droop) * t)
            blocks.append(BOW.wing_block(kc + side * 0.6 * kw * a, way, unit(t - (t @ way) * way), f,
                                         0.5 * span - 0.6 * kw, sh["wing_h"] * span, sh["wing_th"] * span))
        length, notch = float(b["tails"]), sh["notch"] * span
        rows = max(3, int(length / (1.8 * notch)) + 1)      # rows further apart than the V-cut, or the tail folds
        for side in (1.0, -1.0):                            # the tails, down its face, parting a little
            path = []
            for u in np.linspace(0.0, 1.0, rows):
                Q, _, qa, qf, _, qd = at(tie + 0.6 * kt + u * length)
                lat = side * (0.25 * kw + np.tan(np.radians(sh["spread"])) * u * length)
                path.append(Q + qa * lat + qf * (qd + 0.0015 + 0.001 * u))
            blocks.append(BOW.tail_block(np.array(path), f, sh["tail_width"] * span, 0.03 * span, notch, w_end=1.1,
                                         edge=0.4))
        weights = {"頭": 1.0} if chain is None else {k: float(v[0]) for k, v in chain.weights(O[None]).items()}
        V = np.vstack([bk.verts for bk in blocks])
        info = {"group": lk.extra["group"], "lock": lk.name, "knot": [float(x) for x in kc],
                "span": float(np.ptp(V @ a)), "above_end": float(b["above"]), "end_z": float(P[-1, 2])}
        return blocks, weights, info

    def tie_bows(self, bows):
        """The bows' mesh (`hair_bow`, a material per ribbon colour: 結びリボン1, 2 ...) into `more`; `bows` is
        [(lock, bow's (blocks, weights, info))]."""
        colours, mats = {}, []
        for lk, _ in bows:
            b = lk.extra["bow"]
            key = (str(b["ribbon"]), None if b["shade"] is None else str(b["shade"]))
            if key in colours:
                continue
            rgb = TEX.srgb(key[0])
            shade = 0.62 * rgb if key[1] is None else TEX.srgb(key[1])
            k = len(colours) + 1
            look = {"base": key[0], "shade": TEX.to_hex(shade), "sheen": TEX.to_hex(rgb + 0.45 * (1.0 - rgb))}
            png = self.ctx.save_png(f"tie_ribbon{k}", TX.ribbon_texture(look, rng=self.ctx.rng_for(f"hair.bow{k}")))
            toon = self.ctx.save_png(f"toon_tie_ribbon{k}", TEX.toon_ramp(tuple(np.clip(shade / np.maximum(rgb, 1e-3),
                                                                                        0.3, 0.95))))
            mats.append(Material(f"結びリボン{k}", name_en=f"tie ribbon {k}", diffuse=(1.0, 1.0, 1.0, 1.0),
                                 specular=(0.26, 0.25, 0.31), shininess=24.0, ambient=(0.45, 0.45, 0.5), texture=png,
                                 toon=toon, edge=True, edge_size=0.7,
                                 edge_color=(*(float(x) for x in np.clip(0.8 * shade, 0.0, 1.0)), 1.0)))
            colours[key] = len(mats) - 1
        acc = MeshAccum("hair_bow", [m.name for m in mats])
        for lk, (blocks, weights, info) in bows:
            b = lk.extra["bow"]
            mat = colours[(str(b["ribbon"]), None if b["shade"] is None else str(b["shade"]))]
            for bk in blocks:
                acc.add(bk.verts, bk.faces, bk.uv, mat, weights={k: np.full(len(bk.verts), v)
                                                                   for k, v in weights.items()})
        self.more = (self.more[0] + [acc.to_mesh()], self.more[1] + mats)
        self.stats["bows"] = [info for _, (_, _, info) in bows]

    def build(self, acc):
        """The cap and every lock meshed into `acc`, the plaits and the bows into meshes of their own (`more`: meshes
        and materials beside the head hair's). Returns the atlas picture."""
        planned = self.plan()
        locks = [lk for _, g, group in planned if not g["plait"]["enabled"] for lk in group]
        cells, (self._cap_cell,) = self.atlas.cells([(lk.width, float(arclen(lk.path)[-1])) for lk in locks], [CAP_PX])
        for lk, cell in zip(locks, cells):
            lk.extra["cell"] = cell
        self.cap(acc)
        tips, woven, bows = [], [], []
        for name, g, group in planned:
            for lk in group:
                chain = self.chains[lk.chain] if lk.chain else None
                if g["plait"]["enabled"]:
                    woven.append(self.plait(lk, g, chain))
                    continue
                st = self.emit(acc, lk, chain)
                if lk.extra["bow"] is not None:
                    bows.append((lk, self.bow(lk, st, chain)))
                if g["family"] == "bangs" and not lk.extra["under"]:
                    tips.append(float(lk.path[-1, 2]))
            self.stats[f"locks_{name}"] = len(group)
        if woven:
            if len({c["lobes"] for _, c in woven}) > 1:
                raise HairdoError("the plaits of a hairdo are painted with one picture, so they need as many crossings: "
                                  f"got {sorted(c['lobes'] for _, c in woven)} (their woven length over `plait.pitch`)")
            tex = BR.Textures(self.ctx, woven[0][1], self.pal)
            braid = MeshAccum("hair_braid", [tex.hair.name, tex.ribbon.name])
            for blocks, _ in woven:
                for b, mat, w in blocks:
                    braid.add(b.verts, b.faces, b.uv, mat, weights=w)
            self.more = ([braid.to_mesh()], [tex.hair, tex.ribbon])
        if bows:
            self.tie_bows(bows)
        self.stats["locks"] = len(locks) + len(woven)
        self.stats["low_z"] = float(self.z_bot)
        if tips:
            self.stats["bangs_bottom_z"] = min(tips)
        return self.paint(locks)


def build_lock_hair(ctx, fit, cfg, rig, pal):
    """The lock hairdo -> Piece (mesh `hair_head` with material 髪, `hair_braid` with 三つ編 and 髪リボン when a group
    is plaited, `hair_bow` with 結びリボン1.. when locks are tied with bows), its bones, bodies and joints in `rig`. It
    has no shadow decal under its fringe (the clumps' straight fringe has one): a fringe that sweeps casts its own
    under the lights."""
    try:
        hh = LockHair(ctx, fit, cfg, rig, pal)
        acc = MeshAccum("hair_head", [hh.mat])
        atlas = hh.build(acc)
    except HairdoError as e:
        raise ValueError(str(e)) from None
    return Piece(meshes=[acc.to_mesh(), *hh.more[0]], materials=[HH.hair_material(ctx, hh, cfg, pal, atlas), *hh.more[1]],
                 info=HH.hair_info(hh))
