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
    thick       a free lock's thickness over its width; `cup` how far its edges bend toward the head (1 / m)
    root_w      a lock with a `width` of its own: its width at the root over `width`, easing to full over the first
                `root_len` metres
    tip_len     metres over which it tapers into its point; `tip_power` the point's sharpness (1 a straight wedge)
    lift        metres off the hull on its way over the head (negative: tucked under what lies over it): numbers
                spread evenly from the root to where it leaves, or `[[metres from the root, metres off], ...]`,
                level after the last. Hair that rises out of a parting: `[[0, -0.012], [0.025, 0.014], [0.1, 0.006]]`
    swell       times the `lift`; a guide's own makes the hair rise more at one end of a row than at the other
    stiff       how soon a falling lock turns to hang straight down (0 .. 1 per step of STEP)
    tuck        pulls a falling lock in toward the neck; `flare` pushes it out; `lean` bends it toward her back
    wave        {amp, length, phase, drift, grow, depth, start}: waves along the fall, `amp` metres to either side,
                one wave every `length` metres, starting `start` below where the lock leaves the head, `grow` times
                wider at the tip; neighbours are `drift` radians apart, so a group waves together; `depth` is the share
                that goes in and out instead of sideways (1: a corkscrew)
    flip        {len, out}: the last `len` metres turn outward by `out` metres (negative: curl under)
    jitter      {root, to (degrees), length, width (shares), lift (metres), phase (radians)}: how far a lock strays
    under       {enabled, length, width, lift}: a second, darker row half a place aside under this one, its locks
                `length` and `width` times these and `lift` metres closer to the head: what shows between the locks
    chain       {bones, every, radius}: every `every`-th lock is a chain of `bones` bones from where it leaves the
                head (0: the group is rigid on the head); the locks between ride the nearest. A capsule is as thin as
                the room to the body's colliders at rest asks (MIN_RADIUS at least)
    plait       {enabled, pitch, start, ties}: the group's locks are plaits, three strands woven along each lock's
                path (hair_braids' weave) from where it leaves the head (or `start` metres from its root), a crossing
                every `pitch` metres, to a tie and a tuft of loose locks at its end (`ties`: "end", "both", "none").
                A plait is its lock's `width` wide (give one); hair_braids' weave keys may be given here too. A long
                braid is one lock: `count = 1`, `family = "braid"`, a chain of its own

Falling locks slide over the body and the garments built before the hair and face the way those face under them
(hair_clear). `build_lock_hair` lays the scalp cap under it all (hair_head's) and returns the Piece hair.py merges;
the texture is hair_paint's atlas, a cell per lock; the shading normals are hair_head's own."""
import numpy as np

from . import cat_common as CC
from . import hair_braids as BR
from . import hair_braids_bow as BOW
from . import hair_braids_plait as PL
from . import hair_braids_tex as TX
from . import hair_head as HH
from . import hair_paint as PT
from .. import spec as SP
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
RIDE = 0.004                    # metres a lock of half the width rides over its full-width neighbours (`rhythm`)
THICK = (0.0025, 0.010)         # a free lock is never thinner or thicker than this
FLAT = 0.04                     # a lock's thickness where it lies with its neighbours, over its free thickness
HAIR = 0.0002                   # metres between neighbours of a group where they lie in one surface (who is on top)
OWN = 0.6                       # where a lock hangs free it is shaded this much less like the head (its own form shows)
CAP_PX = (512, 256)             # the scalp cap's cell in the atlas
GROUP = {
    "enabled": True, "family": "back", "count": 5, "layer": 0, "guides": [],
    "width": None, "overlap": 1.9, "rhythm": [1.0], "free": 0.08, "finger": 0.7, "thick": 0.12, "cup": 6.0,
    "root_w": 0.3, "root_len": 0.03, "tip_len": 0.07, "tip_power": 1.3, "lift": [-0.012, 0.0], "stiff": 0.22,
    "tuck": 0.15, "flare": 0.0, "lean": 0.0, "swell": 1.0,
    "wave": {"amp": 0.0, "length": 0.12, "phase": 0.0, "drift": 0.6, "grow": 0.5, "depth": 0.3, "start": 0.03},
    "flip": {"len": 0.0, "out": 0.0},
    "jitter": {"root": 2.0, "to": 2.0, "length": 0.08, "width": 0.12, "lift": 0.001, "phase": 0.3},
    "under": {"enabled": False, "length": 1.06, "width": 1.15, "lift": -0.006},
    "chain": {"bones": 0, "every": 1, "radius": 0.010},
    "plait": {"enabled": False, "pitch": 0.034, "start": None, "ties": "end"},
}
NUMBERS = ("width", "free", "finger", "thick", "cup", "root_w", "root_len", "tip_len", "tip_power", "stiff", "tuck",
           "flare", "lean", "swell")
FAMILIES = {"bangs": ("前髪", "bangs"), "side": ("横髪", "hair"), "back": ("後髪", "hair"), "top": ("髪", "hair"),
            "strand": ("髪", "hair"), "braid": ("三つ編", "braid")}
SIDED = ("side", "braid")       # the families whose chains are named by the side they hang on (左 / 右)
DEFAULTS = {**{k: HH.DEFAULTS[k] for k in ("volume", "ring", "cap", "k_outer", "bulge", "toon", "normal_mix", "edge_mix",
                                           "shade", "dynamic")},
            "paint": PT.PAINT, "atlas": 2048, "partings": {}, "locks": {}}


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


class LockHair(HH.HeadHair):
    """The lock hairdo: `plan` lays every group's locks out (paths, shapes, chains), `build` meshes and paints them."""
    DEFAULTS = DEFAULTS

    def __init__(self, ctx, fit, cfg, rig, pal):
        super().__init__(ctx, fit, cfg, rig, pal)
        self.surround = None                                # the body and the garments the falling locks keep off
        self._statics = []                                  # the body's static colliders the chains must not overlap
        if self._body is not None:
            self.surround = Surround(self._body[0].meshes, self._wear, self._body[1])
            self._statics = [rb for rb in self._body[0].bodies if rb.mode == "static"]
        self.atlas = PT.Atlas(int(self.cfg["atlas"]))
        self._cap_cell = None
        self.more = ([], [])                                # meshes and materials beside the head hair's (the plaits')
        self.partings = {}                                  # [hair.partings]: name -> (directions along it, their arc)
        for name, pts in (self.cfg["partings"] or {}).items():
            if name == "hairline" or not isinstance(pts, list) or len(pts) < 2:
                raise HairdoError(f"[hair.partings] {name}: expected two or more [theta, phi] points (`hairline` is "
                                  f"the head's own line)")
            D = unit(spline(np.array([direction(p, f"[hair.partings] {name}[{k}]") for k, p in enumerate(pts)]), 64))
            self.partings[name] = (D, arclen(D))

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

    def course(self, guide, root, what):
        """CTRL unit vectors along a guide's way over the head: from its root through its `via` points to its `to`,
        evenly spaced along the curve through them."""
        pts = [direction(v, f"{what} via[{k}]") for k, v in enumerate(guide.get("via") or [])]
        pts.append(direction(guide.get("to"), f"{what} to"))
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
    def hint(self, P):
        """The way a lock faces along its path P: out from the head's centre above it, level out from the axis through it
        below (a lock hanging down a back faces away from the back), and where it lies on the body or a garment the way
        that surface faces (flat on a shoulder, it faces up). The body's say fades in over TURN below the neck seam and
        the facing is averaged along the path, so a lock turns over a few centimetres, never between two points."""
        c = self.skull.center
        axis = np.stack([np.full(len(P), c[0]), np.full(len(P), c[1]), np.minimum(P[:, 2], c[2])], -1)
        n = unit(P - axis)
        if self.surround is not None:
            d, out = self.surround.away(P)
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

    def fall(self, P, length, g):
        """`P` gone on for `length` metres as hair falls: its heading turns to straight down (`stiff` of the way each
        step), pulled in by `tuck`, out by `flare` and back by `lean`; it stays off the head, and what it meets of the
        body and the garments (`rest` off them: REST, more for a thick plait) it slides over."""
        rest = float(g.get("rest", REST))
        n = int(round(length / STEP))
        if n < 1:
            return P
        c = self.skull.center
        p, h = P[-1], unit(P[-1] - P[-2])
        out = []
        for _ in range(n):
            rxy = unit(np.array([p[0] - c[0], p[1] - c[1], 0.0]))
            down = unit(np.array([0.0, g["lean"], -1.0]) + (g["flare"] - g["tuck"]) * rxy)
            h = unit(h + g["stiff"] * (down - h))
            q = self.skull.push_out(p + STEP * h, 0.008)
            if self.surround is not None:
                d, way = self.surround.away(q[None])
                if d[0] < rest:
                    q = q + way[0] * (rest - d[0])
                    h = unit(q - p)                         # it goes on the way the surface turned it
            p = q
            out.append(p)
        return np.vstack([P, np.array(out)])

    def style(self, P, n_head, wave, flip, phase):
        """The fallen part of `P` (its points from `n_head` on) waved and its end flipped."""
        if len(P) - n_head < 3 or (wave["amp"] <= 0.0 and flip["out"] == 0.0):
            return P
        Q = P.copy()
        F = P[n_head - 1:]
        s = arclen(F)
        L = float(s[-1])
        T = unit(np.gradient(F, axis=0))
        out = self.hint(F)
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
        return Q

    # ---------------------------------------------------------------- a group's locks
    def group(self, name, g):
        """The locks of group `name`, as HH.Lock with `extra` {s_leave (where it leaves the head, metres along it),
        free, finger, root_len, wide (its width over its full width at ROOM points from the root to s_leave; None with
        a `width` of its own), lie (the surface it lies in on the head: `lying`), group, under}."""
        what = f"[hair.locks.{name}]"
        guides = g["guides"]
        if not isinstance(guides, list) or not guides or not all(isinstance(x, dict) for x in guides):
            raise HairdoError(f"{what}: guides = [{{root, to, length}}, ...] is required")
        if g["family"] not in FAMILIES:
            raise HairdoError(f"{what}: family {g['family']!r} is none of {', '.join(FAMILIES)}")
        roots = [self.rooted(x.get("root"), f"{what} guides[{k}] root") for k, x in enumerate(guides)]
        line, keys = None, [r[1] for r in roots]
        if all(k is not None for k in keys) and len({k[0] for k in keys}) == 1:
            # every guide rooted on one line: the group's surface is told by the distance from that stretch of it
            at = np.linspace(min(k[1] for k in keys), max(k[1] for k in keys), 64)
            line = self.on_hull(np.array([self.on_line(keys[0][0], x, what) for x in at]))
        courses = np.array([self.course(x, roots[k][0], f"{what} guides[{k}]") for k, x in enumerate(guides)])
        lift = profile(g["lift"], what)
        lengths = np.array([float(x.get("length", 0.0)) for x in guides])
        own = {k: np.array([float(x.get(k, g[k])) for x in guides]) for k in NUMBERS if g[k] is not None or
               all(k in x for x in guides)}
        if g["width"] is None and any("width" in x for x in guides) and "width" not in own:
            raise HairdoError(f"{what}: give `width` in the group or in every guide")
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
                C = unit((1.0 - f) * courses[k] + f * courses[j])
                a, b = roots[k][1], roots[j][1]
                if a is not None and b is not None and a[0] == b[0]:        # both on one line: so is this lock
                    exact = self.on_line(a[0], (1.0 - f) * a[1] + f * b[1], what)
                    C = unit(C + np.outer(np.linspace(1.0, 0.0, CTRL) ** 2, exact - C[0]))
                else:
                    C[0] = stray(C[0], jit["root"], rng)
                C[-1] = stray(C[-1], jit["to"], rng)
                if line is not None:
                    C = self.combed(C, line)
                num = {key: float((1.0 - f) * v[k] + f * v[j]) for key, v in own.items()}
                beat = 1.0 if under else rhythm[i % len(rhythm)]
                length = float((1.0 - f) * lengths[k] + f * lengths[j]) * (1.0 + jit["length"] * rng.uniform(-1.0, 1.0))
                rise = jit["lift"] * rng.uniform(-1.0, 1.0) + RIDE * 2.0 * max(1.0 - beat, 0.0)
                if under:
                    length *= g["under"]["length"]
                lie = {"lift": lift, "swell": num["swell"], "layer": int(g["layer"]), "line": line,
                       "lower": -g["under"]["lift"] if under else 0.0, "over": HAIR * (i % 3)}
                head = self.on_head(C, lie)
                rest = REST
                if g["plait"]["enabled"]:               # a plait is thick: its middle line rests further off her
                    if "width" not in num:
                        raise HairdoError(f"{what}: a plait needs a `width`")
                    rest = max(REST, 0.5 * BR.DEFAULTS["thick"] / BR.DEFAULTS["width"] * num["width"] + 0.004)
                P = self.fall(head, length, {**g, **num, "rest": rest})
                place = (i + shift) if n > 1 else 0.0       # the rows of a group wave together
                phase = g["wave"]["phase"] + g["wave"]["drift"] * place + (0.0 if under else jit["phase"] * rng.uniform(-1.0, 1.0))
                P = self.style(P, len(head), g["wave"], g["flip"], phase)
                s_leave = float(arclen(P[:len(head)])[-1])
                P = smooth_polyline(resample(P, max(int(arclen(P)[-1] / STEP) + 1, 6)), 1)
                planned.append((P, s_leave, num, C[-1], resample(head, ROOM), beat, rise, lie))
            for i, (P, s_leave, num, d_to, along, beat, rise, lie) in enumerate(planned):
                wide = None
                if "width" in num:
                    w = num["width"]
                else:                                       # the room to its neighbours, all along its way on the head:
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
                P = P + self.hint(P) * (rise * freedom(s, L, free))[:, None]
                thick = float(np.clip(num["thick"] * w * num["finger"], *THICK))
                if g["plait"]["enabled"]:
                    thick = BR.DEFAULTS["thick"] / BR.DEFAULTS["width"] * w
                tip_start = float(np.clip(1.0 - num["tip_len"] / max(L, 1e-6), 0.15, 0.97))
                if self.surround is not None and L - s_leave > 0.02:
                    P = self.surround.keep_off(P, self.hint, w, thick, num["cup"], tip_start, HH.BODY_GAP,
                                               first=max(int(s_leave / STEP) - 1, 1))
                out.append(HH.Lock(f"{name}{'_u' if under else ''}{i}", P, w, thick, tip="point", tip_start=tip_start,
                                   tip_power=num["tip_power"], root_w=num["root_w"], curv=num["cup"],
                                   layer=int(g["layer"]), theta=float(angles(d_to)[0]), dark=under,
                                   extra={"s_leave": s_leave, "free": free, "finger": num["finger"], "lie": lie,
                                          "root_len": num["root_len"], "wide": wide, "group": name, "under": under}))
        return out

    def clump(self, lk):
        """The lock's mesh: as wide as the room to its neighbours and paper-flat where it lies with them, every point
        of it in the group's surface (`lying`); `finger` of the width, its own thickness and its own `cup` where it
        hangs free."""
        P = np.asarray(lk.path, float)
        s = arclen(P)
        L = float(s[-1])
        e = lk.extra
        hint = self.hint(P)
        lie, c = e["lie"], self.skull.center
        loose = freedom(s, L, e["free"])
        on = (1.0 - loose) * (1.0 - smoothstep((s - (e["s_leave"] - 0.01)) / 0.03))    # 1 where it lies on the head

        def place(mid, sp):
            m = float(np.interp(sp * L, s, on))
            if m <= 0.0:
                return mid
            D = unit(mid - c)
            x = np.full(len(mid), sp * L) if lie["line"] is None else self.away(D, lie["line"])
            return mid + m * (self.skull.point(D, self.lying(D, x, lie)) - mid)

        def free(sp):
            return freedom(np.asarray(sp, float) * L, L, e["free"])
        wide = e.get("wide")

        def across(sp):
            f = 1.0 - (1.0 - e["finger"]) * free(sp)
            if wide is None:
                return f
            return f * np.interp(np.asarray(sp, float) * L, np.linspace(0.0, e["s_leave"], len(wide)), wide)
        return sweep(P, hint, lk.width, lk.thick, tip=lk.tip, tip_start=lk.tip_start, tip_power=lk.tip_power,
                     root_w=lk.root_w if wide is None else 1.0,
                     root_len=float(np.clip(e["root_len"] / max(L, 1e-6), 0.01, 0.5)), curv=lk.curv,
                     k_outer=int(self.cfg["k_outer"]) if lk.width >= 0.018 else 5, bulge=self.cfg["bulge"],
                     ring_step=0.010, width_fn=across, thick_fn=lambda sp: FLAT + (1.0 - FLAT) * free(sp),
                     curv_fn=free, place=place)

    def proxy_normals(self, P):
        """HeadHair's shading proxy with its axis going on down to the lowest hair: what hangs below the head is shaded
        as a column round her, not as the underside of a ball."""
        c = self.fit.center
        return HH.shade_proxy(P, c, c[2] + self.cfg["shade"]["z_hi"], min(c[2] + self.cfg["shade"]["z_lo"], self.z_bot))

    def radii(self, lk, bones, s_root, radius):
        """A chain's capsule radius per bone: `radius`, or what the room to the body's static colliders leaves at rest
        (MIN_RADIUS at least): a capsule that overlaps one at rest is thrown out by physics."""
        if not self._statics:
            return radius
        s = arclen(lk.path)
        sh = np.linspace(s_root, s[-1], bones + 1)
        pts = np.stack([np.interp(sh, s, lk.path[:, k]) for k in range(3)], -1)
        room = [CC.capsule_clearance(self._statics, pts[k], pts[k + 1], 0.0)[0] for k in range(bones)]
        return np.clip(np.asarray(room) - 0.001, MIN_RADIUS, radius)

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
                return arclen(lk.path)[-1] - lk.extra["s_leave"] > 0.02
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
        mix = self.cfg["normal_mix"] * (1.0 - OWN * freedom(arc, st.length, lk.extra["free"]))
        acc.add(V, faces, uv, 0, weights=w, normals=shell_normals(V, faces, side, self.proxy_normals(V), mix))
        return st

    def paint(self, locks):
        """Every lock's cell and the cap's painted: the atlas picture. A point of a cell is painted by where it is on
        the head (its height, its angle round the head), so locks that lie side by side carry one picture."""
        look = self.cfg["paint"]
        ring = self.cfg["ring"]
        sheen = None
        if ring["enabled"]:
            sheen = PT.Sheen(self.z_top - ring["drop"], ring["width"], look, self.ctx.rng_for("hair.paint.sheen"))
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
                                under=lk.extra["under"])
            self.atlas.put(cell, rgb)
        self.atlas.put(self._cap_cell, PT.paint_cap(self._cap_cell, self.pal, look, self.ctx.rng_for("hair.paint.cap")))
        return self.atlas.image(self.pal["base"] * look["cap"])

    def plait(self, lk, g, chain):
        """The lock as a three-strand plait (hair_braids' weave along the lock's path): woven from where the lock
        leaves the head (or `plait.start` metres from its root) to a tie a tuft's length short of its end, a tuft of
        loose locks below the tie (`plait.ties`: "end", "both" ties its top too, "none"). It is the lock's `width`
        wide, one crossing every `plait.pitch` metres; hair_braids' own weave keys given in the `plait` table (taper,
        strand_wid, tuft_len ... for a plait 0.034 wide) scale with the width. Returns ([(block, material 0 hair or 1
        the ties' ribbon, weights)], the weave's keys)."""
        p = g["plait"]
        what = f"[hair.locks.{lk.extra['group']}] plait"
        c = SP.merge(BR.DEFAULTS, {k: v for k, v in p.items() if k in BR.DEFAULTS})
        scale = lk.width / c["width"]
        ties = {"end": (False, True), "both": (True, True), "none": (False, False)}.get(p["ties"])
        if ties is None:
            raise HairdoError(f"{what}: ties = {p['ties']!r} is none of end, both, none")
        P = np.asarray(lk.path, float)
        s = arclen(P)
        start = lk.extra["s_leave"] if p["start"] is None else float(p["start"])
        woven = float(s[-1]) - c["tuft_len"] * scale - start
        if woven < 3.0 * float(p["pitch"]):
            raise HairdoError(f"{what}: {woven * 100:.0f} cm from its start to its tuft is too short to plait "
                              f"(a crossing every {float(p['pitch']) * 100:.1f} cm)")
        c["lobes"] = int(round(woven / float(p["pitch"])))
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

    def build(self, acc):
        """The cap and every lock meshed into `acc`, the plaits into a mesh of their own (`more`: meshes and materials
        beside the head hair's). Returns the atlas picture."""
        planned = self.plan()
        locks = [lk for _, g, group in planned if not g["plait"]["enabled"] for lk in group]
        cells, (self._cap_cell,) = self.atlas.cells([(lk.width, float(arclen(lk.path)[-1])) for lk in locks], [CAP_PX])
        for lk, cell in zip(locks, cells):
            lk.extra["cell"] = cell
        self.cap(acc)
        tips, woven = [], []
        for name, g, group in planned:
            for lk in group:
                chain = self.chains[lk.chain] if lk.chain else None
                if g["plait"]["enabled"]:
                    woven.append(self.plait(lk, g, chain))
                    continue
                self.emit(acc, lk, chain)
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
        self.stats["locks"] = len(locks) + len(woven)
        self.stats["low_z"] = float(self.z_bot)
        if tips:
            self.stats["bangs_bottom_z"] = min(tips)
        return self.paint(locks)


def build_lock_hair(ctx, fit, cfg, rig, pal):
    """The lock hairdo -> Piece (mesh `hair_head` with material 髪, and `hair_braid` with 三つ編 and 髪リボン when a
    group is plaited), its bones, bodies and joints in `rig`. It has no shadow decal under its fringe (the clumps'
    straight fringe has one): a fringe that sweeps casts its own under the lights."""
    try:
        hh = LockHair(ctx, fit, cfg, rig, pal)
        acc = MeshAccum("hair_head", [hh.mat])
        atlas = hh.build(acc)
    except HairdoError as e:
        raise ValueError(str(e)) from None
    return Piece(meshes=[acc.to_mesh(), *hh.more[0]], materials=[HH.hair_material(ctx, hh, cfg, pal, atlas), *hh.more[1]],
                 info=HH.hair_info(hh))
