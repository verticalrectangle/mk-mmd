"""One big bushy tail (part `tail`, numpy only): a fat soft tube swept along a soft S from the small of her back, a few
broad tufts breaking its outline, painted in bands whose edges follow the tufts, ending in a brush of parted lobes like
a soft flame; it hangs on a chain that is stiff at the root and softer toward the tip from an aim bone a show can key.

Path: the centreline leaves the body's `tail_root` landmark in the plane through it that holds the vertical and the
level part of the body's `tail_normal` (straight back), that plane turned `turn` degrees toward her left about the
upright through the root; `lean` moves the centre line further toward her left along its length. Its tangent's
elevation above level is given against the arc length by knots (`[tail.path] s` fractions of `length`, `elev`
degrees, a monotone cubic through them); above 90 the tail turns forward again. Each meridian of the tube starts where
it leaves the skin, so no vertex lies inside her.

Thickness: the body's half-width (`radius`, stations [[fraction of the length, metres], ...], held past the last; its
half-depth in the plane of the path is `flat` of it) runs through the stations on a monotone cubic.

Brush: `[tail.brush] length` before the end the body parts into a middle lobe (`middle` x its radius across where it
parts) and `lobes` lobes round it, the first parting `phase` degrees round from the inside of the curve. Where it parts
a lobe round the middle is the slice of the body's section between two radii and outside the middle one's circle (an
arc of the surface, the radii and an arc of that circle), so the surface runs on without a seam and the lobes share
the vertices where they meet; over `round` of its length the slice rounds into an ellipse `size` x the body's radius
across (out from the axis, round it) that rides on the middle one, its inner side on the middle one's surface and a
further `part` metres x (its share of the way)^2 off it: the lobes close round the middle one as it narrows, part
toward their ends and curl a little outward. The middle lobe runs to the tail's end, each lobe round it `steps`
metres short of it; every lobe narrows by `taper` along its length and ends in a round point. The lobes are the last
band's colour, lighter toward the tail's end.

Fur: the body is smooth; tufts in two places only: one ring of `ring.count` broad tufts at each band's far edge (that
ring is the band's edge: a soft zigzag of fur) and `single` tufts lifting off the outside of the curve between the
edges. A tuft lies along the tail toward the tip: rooted under the body `cover` times its visible length back, it comes
out at `rise` of its height, stands its full height off the smooth body (`height`; on the inside of the curve `inside`
of it) and ends in a rounded point (`point`, rounded over `lip`). The surface is the smooth body pushed out along its
normal by the highest tuft at every point.

Bands: `[tail.bands] pattern` (dark and tan, root and tip dark) laid from `from` metres (where the fur starts to show,
before it all is the first colour) to the end, their widths in the ratio `widths`. A tuft takes the colour of the band
it belongs to, so a band's edge is its ring's rounded free ends: a soft zigzag. Painted as the hair is: a soft light
along the side that faces up, a shade under, a little shade where a tuft comes out of the body, each tuft and lobe
lighter toward its free end. One material 尻尾 (the bands live in the picture, so their scalloped edges never follow
the faces); its toon shade is the mean of the colours' own (`[colors.tail]` `<colour>_shadow` over `<colour>`), its
outline the dark's shadow. The normals are the smooth body's with `[tail.tufts] shade` of the tufts' own: the light
rounds the whole tail softly and the tufts show in its outline and its bands, not as lumps.

Rig: the aim bone 尻尾 (parent 下半身, its head at the root: keyed, it turns the whole tail about the root up or to a
side; mk's `twitch` with family "tail" flicks it; a tiny static body on it is the chain's joint partner) and the chain
尻尾1..N (kind "tail", `[tail.rig] bones`) from the first point of the path where the first capsule clears the body's
static colliders by `clear`, to the end. Mass, spring and swing limit run from the root's values to the tip's (stiff at
the root, softer toward the tip), the damping is high so a swing settles within about a second in MMD; mk's own solver
uses the show's `[sim.<cast>.params.tail]`. Every capsule's gap to the static colliders is checked
(`info["clearance"]`). The surface is skinned to the chain by its arc length, the part inside the clothes on the aim
bone; over its first `base` metres it passes from the lower body (at the skin) to the aim bone, so a turned aim bone
bends the tail at its root instead of swinging its base into her.

Checks: the tail's surface against every part built before it (its skin, its garments, its hair), for the part of the
tail that is outside the clothes: the smallest gap, and the vertices inside each closed part (an odd number of
crossings along two rays) or, for a sheet such as a garment, behind its surface (`info["gaps"]`).

Config `[tail]`: DEFAULTS below (metres, degrees); colours `[colors.tail]` (COLORS)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Bone, Material, Mesh, Part, RigidBody
from . import cat_common as CC
from . import hair_tex as TEX
from . import outfit_wear as WR
from .hair_geo import arclen, smoothstep, unit
from .hair_rig import Rig

DEFAULTS = {
    "length": 0.78,                 # m, centreline from the skin to the end of the longest lobe
    "path": {                       # tangent elevation above level (deg) against the arc length (fractions of length)
        "s": [0.0, 0.07, 0.16, 0.26, 0.36, 0.46, 0.6, 0.9, 1.0],
        "elev": [60.0, 16.0, 16.0, 24.0, 58.5, 89.0, 97.0, 97.0, 121.0],
    },
    "radius": [[0.0, 0.035], [0.25, 0.075], [0.45, 0.100], [0.62, 0.115], [0.80, 0.090], [0.89, 0.060]],   # half-width
    "turn": 0.0,                    # deg, the path's plane turned toward her left about the upright through the root
    "lean": [],                     # [[fraction of length, m], ...] the centre line moved toward her left (+x), off its
                                    # plane (a monotone cubic through them; none: the tail stays in its plane)
    "flat": 0.9,                    # half-depth in the path's plane over the half-width
    "embed": 0.03,                  # m, how far the tube is drawn back into the body before it is cut at the skin
    "skin_gap": 0.001,              # m, the cut lies this far off the skin
    "ring_step": 0.0045,            # m between rings along the surface
    "around": 96,                   # vertices round a ring
    "brush": {                      # the end parts into lobes like a soft flame
        "lobes": 4,                 # round the middle one
        "length": 0.09,             # m from where the body parts to the end of the middle lobe (the tail's end)
        "steps": [-0.035, -0.045, -0.03, -0.04],   # m each lobe round the middle is shorter than it
        "phase": 45.0,              # deg round from the inside of the curve to the first parting
        "middle": 0.5,              # the middle lobe's radius where the body parts, x the body's
        "size": [0.27, 0.45],       # a lobe round it: half-depth (out from the axis), half-width (round it), x that
        "part": 0.012,              # m its end stands off the middle one (x its share of the way squared)
        "round": 0.35,              # share of its length over which the slice rounds into the ellipse
        "taper": 0.25,              # every lobe narrows by this share along its length, then ends in a round point
    },
    "tufts": {                      # the body is smooth; tufts only at the band edges and outside the curve
        "start": 0.08,              # m along the path before which there is no tuft (the root, inside the clothes)
        "ring": {"count": [7, 9],   # one ring at each band's far edge: tufts round the tail, `width` m wide each,
                 "width": 0.055,    # `length` m of them showing, standing `height` m off (root -> tip), `overlap`
                 "length": 0.06, "height": [0.015, 0.020], "overlap": 1.15},   # their width over their spacing
        "single": {"count": 6, "width": 0.05, "length": 0.06, "height": [0.015, 0.022]},   # lifting off outside the curve
        "cover": 1.7,               # a tuft's whole length over the part that shows (the rest lies in the body)
        "rise": 0.45,               # share of its height a tuft stands at a third of the way along it
        "lip": 0.006,               # m over which a tuft's edge rounds off (and its point)
        "point": 0.3,               # how pointed a tuft's free end is (0 a half circle, 1 a leaf), rounded by `lip`
        "inside": 0.6,              # share of the height on the inside of the curve (phi 0, the path's inner side)
        "max": 0.025,               # m, no tuft stands higher
        "jitter": 0.12,             # random turn, shift, width and height of each tuft (shares)
        "shade": 0.3,               # how much of the tufts' own shape the normals (the light) follow; the rest the body's
    },
    "bands": {
        "from": 0.06,               # m along the path where the bands start being seen (before it: the first colour)
        "pattern": ["dark", "tan", "dark", "tan", "dark"],
        "widths": [1.0, 0.8, 1.0, 0.8, 1.0],
    },
    "paint": {
        "size": [512, 1024],        # texture pixels (round the tail, along it)
        "light": 0.55,              # how far the side that faces up goes to the light colour
        "under": 0.30,              # how far the underside goes to the shadow colour
        "tuck": 0.08,               # shade where a tuft comes out of the body
        "tips": 0.35,               # how far each tuft and lobe goes to the light colour toward its free end
        "toon": {"edge": 0.52, "soft": 0.20},
        "edge_size": 0.8,
    },
    "rig": {
        "bones": 8,
        "radius": 0.5,              # chain capsule radius x the tube's half-width there
        "clear": 0.005,             # m, every capsule this far off the body's static colliders
        "hold": 0.3,                # share of the first chain bone that stays rigid with the aim bone
        "base": 0.05,               # m from the skin over which the base passes from the lower body to the aim bone
        "mass": [0.30, 0.12],       # root -> tip
        "spring": [60.0, 10.0],     # joint rotation spring, root -> tip
        "swing": [0.30, 0.60],      # rad, joint swing limit, root -> tip
        "twist": 0.15,              # rad
        "damping": [0.95, 0.99],    # linear, angular
    },
    "seed": "tail",
}
COLORS = {"dark": "#5a2f2b", "dark_shadow": "#371b19", "dark_light": "#7a4a40",
          "tan": "#c9a878", "tan_shadow": "#9a7a54", "tan_light": "#e8d2aa"}
MAT = "尻尾"
AIM = "尻尾"
CHAIN = "尻尾"
UP = unit(np.array([0.0, 0.35, 1.0]))   # the side lit "from the top": up and a little back


# ---------------------------------------------------------------- path and thickness
def centreline(cfg, root, back, n=1200):
    """(P (n, 3), s (n,)) the centreline from the root: its tangent's elevation against the arc length through the knots
    of cfg["path"], heading along the level unit vector `back`."""
    pa = cfg["path"]
    L = float(cfg["length"])
    el = CC.pchip(np.asarray(pa["s"], float), np.radians(np.asarray(pa["elev"], float)))
    s = np.linspace(0.0, L, n)
    E = el(s / L)
    b = unit(np.array([back[0], back[1], 0.0]))
    d = np.cos(E)[:, None] * b[None] + np.sin(E)[:, None] * np.array([0.0, 0.0, 1.0])[None]
    P = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(s)[:, None], axis=0)], 0)
    lean = np.asarray(cfg["lean"], float).reshape(-1, 2)
    if len(lean) >= 2:                                 # the tail leans toward her left (+) off its plane
        P = P + CC.pchip(lean[:, 0], lean[:, 1])(s / L)[:, None] * np.cross(b, [0.0, 0.0, 1.0])[None]
    return np.asarray(root, float)[None] + P, s


class Profile:
    """The body's half-width r(s) on [0, L]: a monotone cubic through the stations, held past the first and the
    last."""

    def __init__(self, cfg):
        L = float(cfg["length"])
        st = np.asarray(cfg["radius"], float).reshape(-1, 2)
        if len(st) < 2 or (np.diff(st[:, 0]) <= 0).any():
            raise ValueError("[tail] radius: give two or more stations [fraction, metres] with rising fractions")
        self.L = L
        self.f = CC.pchip(st[:, 0] * L, st[:, 1])

    def __call__(self, s):
        return self.f(np.clip(np.asarray(s, float), 0.0, self.L))


class Frames:
    """Rotation-minimising frames along the centreline (N in the path's plane, toward the inside of its first bend;
    B = T x N, level, her left for a tail pointing back), extended straight back by `embed` (s < 0 inside the body);
    `at(s)` interpolates centre, T, N, B."""

    def __init__(self, P, s, embed):
        t0 = unit(P[1] - P[0])
        k = max(2, int(np.ceil(embed / 0.004)))
        back = np.linspace(-embed, 0.0, k + 1)[:-1]
        self.P = np.vstack([P[0][None] + back[:, None] * t0[None], P])
        self.s = np.concatenate([back, s])
        n0 = np.cross(np.cross(t0, [0.0, 0.0, 1.0]), t0)     # in the path's plane, the upper side of the start
        if np.linalg.norm(n0) < 1e-6:
            n0 = np.array([0.0, -1.0, 0.0])
        self.T, self.N, self.B = CC.rmf(self.P, n0)

    def at(self, s):
        s = np.atleast_1d(np.asarray(s, float))

        def f(A):
            return np.stack([np.interp(s, self.s, A[:, k]) for k in range(3)], -1)
        C, T, N = f(self.P), unit(f(self.T)), unit(f(self.N))
        N = unit(N - (N * T).sum(-1, keepdims=True) * T)
        return C, T, N, np.cross(T, N)


# ---------------------------------------------------------------- the fur
class Fur:
    """The tufts (see the module docstring). `rows` holds, per ring of tufts or single tuft: root s0, free end e,
    visible length sp, tuft count n, phase, angular half-width hw (rad), height H, band index; `jit` the tufts' turns.
    `field(s, phi)` gives the displacement h, the band of the tuft on top (-1: none), and the tuft's `along` (0 root ..
    1 free end) and `tuck` (1 where it comes out of the body, 0 well clear of it). No tuft reaches past `end` (where
    the brush parts)."""

    def __init__(self, cfg, L, bounds, pattern, radius_at, rng, end):
        tc = cfg["tufts"]
        self.L = L
        self.cfg = tc
        self.flat = float(cfg["flat"])
        start = float(tc["start"])
        ring, single = tc["ring"], tc["single"]
        cover, jit = float(tc["cover"]), float(tc["jitter"])
        r_len = float(ring["length"])
        # where nothing covers it the picture's band edge lies a ring's length before the edge, so between the ring's
        # tufts the next band shows: the tufts' free ends are the edge
        self.band_bounds = np.asarray(bounds, float).copy()
        self.band_bounds[1:-1] -= r_len
        rows = [(float(bounds[k + 1]), r_len, k, "ring") for k in range(len(pattern) - 1)]
        n1, s1 = int(single["count"]), float(single["length"])
        lo, hi = start + 0.5 * s1, float(end) - s1
        for i in range(n1):                             # singles spread along the tail, clear of the band edges
            e = lo + (hi - lo) * (i + 0.5) / max(n1, 1)
            near = np.abs(np.asarray(bounds[1:-1]) - e)
            if len(near) and near.min() < r_len:
                e = float(bounds[1 + int(near.argmin())]) + r_len * (1.0 if e > bounds[1 + int(near.argmin())] else -0.8)
            rows.append((e, s1, int(_band_of(np.array([e - 0.5 * s1]), bounds)[0]), "single"))
        if any(e + 0.012 > end for e, _, _, _ in rows):
            raise ValueError(f"[tail] a tuft reaches past where the brush parts ({end:.3f} m): shorten [tail.brush] "
                             f"length or widen the last band")
        rows.sort(key=lambda r: r[0])
        self.rows, self.jit = [], []
        for e, sp, band, kind in rows:
            g = np.clip((e - start) / max(L - start, 1e-6), 0.0, 1.0)
            r_mid = max(float(radius_at(e - 0.5 * sp)), 1e-3) * 0.5 * (1.0 + self.flat)
            if kind == "ring":
                n = int(np.clip(round(2.0 * np.pi * r_mid / float(ring["width"])), *ring["count"]))
                dphi = 2.0 * np.pi / n
                phase, hw = rng.uniform(0.0, dphi), 0.5 * float(ring["overlap"]) * dphi
                H = float(ring["height"][0]) + (float(ring["height"][1]) - float(ring["height"][0])) * g
            else:
                n, dphi = 1, 2.0 * np.pi
                phase, hw = np.pi + rng.normal(0.0, 0.35), 0.5 * float(single["width"]) / r_mid
                H = rng.uniform(*(float(v) for v in single["height"]))
            self.rows.append(dict(s0=e - cover * sp, e=e, sp=sp, n=n, phase=phase, hw=hw, H=H, band=band,
                                  edge=kind == "ring", kind=kind))
            self.jit.append(dict(turn=rng.normal(0.0, jit * 0.5 * min(dphi, np.pi), n),
                                 shift=np.clip(rng.normal(0.0, jit * sp * 0.5, n), -jit * sp, jit * sp),
                                 width=np.clip(1.0 + rng.normal(0.0, jit, n), 1.0 - 1.5 * jit, 1.0 + 1.5 * jit),
                                 height=np.clip(1.0 + rng.normal(0.0, 0.7 * jit, n), 1.0 - jit, 1.0 + jit)))
        self.first = min((r["s0"] + (cover - 1.0) * r["sp"] for r in self.rows), default=L)

    def field(self, s, phi, radius_at):
        """(h, band, along, tuck) at points (s, phi) of the smooth body (arrays of one shape)."""
        s, phi = np.broadcast_arrays(np.asarray(s, float), np.asarray(phi, float))
        shape = s.shape
        s, phi = s.reshape(-1), phi.reshape(-1)
        h = np.zeros(s.shape)
        band = np.full(s.shape, -1, int)
        along = np.zeros(s.shape)
        tuck = np.zeros(s.shape)
        tc = self.cfg
        lip, rise, cover = float(tc["lip"]), float(tc["rise"]), float(tc["cover"])
        a_vis = 1.0 - 1.0 / cover                       # where a tuft comes out from under the row before it
        r_local = np.maximum(radius_at(s), 1e-4)
        for row, jt in zip(self.rows, self.jit):
            ell = row["e"] - row["s0"]
            idx = np.where((s >= row["s0"] - 0.012) & (s <= row["e"] + 0.012))[0]
            if not len(idx):
                continue
            ss, pp, rl = s[idx], phi[idx], r_local[idx]
            n = row["n"]
            dphi = 2.0 * np.pi / n
            centre = row["phase"] + dphi * np.arange(n) + jt["turn"]
            d = (pp[:, None] - centre[None] + np.pi) % (2.0 * np.pi) - np.pi      # (m, n) angle from each tuft
            a = (ss[:, None] - row["s0"] - jt["shift"][None]) / ell                 # 0 root .. 1 free end
            hw = row["hw"] * jt["width"][None]                                    # angular half-width of each tuft
            hw_m = hw * rl[:, None] * 0.5 * (1.0 + self.flat)                     # metres across half a tuft
            a0 = np.clip(1.0 - hw_m / ell, 0.25, 0.9)                             # the round end starts here
            c = d / hw
            e = np.clip((a - a0) / (1.0 - a0), 0.0, None)
            rho = np.sqrt((c / np.maximum(1.0 - float(tc["point"]) * e, 0.05)) ** 2 + e * e)   # a leaf's rounded point
            inside = (a >= 0.0) & (a <= 1.0) & (rho < 1.0)
            dist = (1.0 - rho) * hw_m                                             # metres to the tuft's edge
            lift = np.sin(0.5 * np.pi * np.clip(dist / lip, 0.0, 1.0))
            t = np.clip((a - a_vis) / max(1.0 - a_vis, 1e-6), 0.0, 1.0)
            up = np.where(a < a_vis, rise * np.clip(a / a_vis, 0.0, 1.0),
                          rise + (1.0 - rise) * np.sin(0.5 * np.pi * np.clip(t / 0.75, 0.0, 1.0)))
            out = float(tc["inside"]) + (1.0 - float(tc["inside"])) * 0.5 * (1.0 - np.cos(centre))   # 1 outside the curve
            Hk = np.minimum(row["H"] * jt["height"] * out, float(tc["max"]))
            hk = np.where(inside, Hk[None] * up * lift, -1.0)
            j = np.argmax(hk, axis=1)
            k = np.arange(len(j))
            best, a_top = hk[k, j], a[k, j]
            take = best > h[idx]
            sel = idx[take]
            h[sel] = best[take]
            band[sel] = row["band"]
            along[sel] = a_top[take]
            tuck[sel] = np.clip(1.0 - (a_top[take] - a_vis) / 0.12, 0.0, 1.0) * (a_top[take] >= a_vis)
        return h.reshape(shape), band.reshape(shape), along.reshape(shape), tuck.reshape(shape)


# ---------------------------------------------------------------- the surface
def _skin_cut(base_fn, skin, n, s_lo, s_hi, gap, iters=16):
    """Per meridian (base_fn(s) -> its n points, s an arc length per meridian), the arc length where it leaves the skin
    by `gap` (bisection between s_lo, inside the body, and s_hi, outside)."""
    lo = np.full(n, float(s_lo))
    hi = np.full(n, float(s_hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = skin(base_fn(mid))
        out = d >= gap
        hi = np.where(out, mid, hi)
        lo = np.where(out, lo, mid)
    return hi


class Brush:
    """The flame at the end (see the module docstring). Lobe k < n runs from `s0`, where the body parts, to `ends[k]`;
    round the tail it holds the body's ring vertices `cuts[k]` .. `cuts[k + 1]` (its arc where the body parts) and is
    centred on the angle `phi[k]`; lobe n is the middle one, to the tail's end. Where the body parts a lobe round the
    middle is the slice of the section between its two cuts and outside the middle one's circle (`middle` x the body's
    radius); its ring runs along its arc, down the second cut, back along the middle one's circle and up the first
    cut: round the lobe the same way as the body's rings go round the tail. The middle one's ring is that circle.
    `template(k)` describes a ring, `lobe(k)` gives the rings."""

    def __init__(self, cfg, frames, prof, na):
        bc = cfg["brush"]
        self.cfg, self.frames, self.flat = bc, frames, float(cfg["flat"])
        L = prof.L
        n = int(bc["lobes"])
        if n < 2 or na < 8 * n:
            raise ValueError(f"[tail.brush] lobes: {n} round the middle one: give 2 or more, and [tail] around at "
                             f"least 8 x as many")
        steps = np.asarray(bc["steps"], float).reshape(-1)
        self.n, self.na = n, na
        self.s0 = L - float(bc["length"])
        self.ends = np.append(L + (steps[np.arange(n) % len(steps)] if len(steps) else np.zeros(n)), L)
        if self.s0 <= 0.0 or (self.ends <= self.s0 + 0.02).any() or (self.ends > L + 1e-9).any():
            raise ValueError("[tail.brush] length, steps: every lobe 2 cm long or more, none past the end (steps <= 0)")
        first = float(bc["phase"]) / 360.0 * na
        self.cuts = np.round(first + na * np.arange(n + 1) / n).astype(int)
        self.phi = np.pi * (self.cuts[:-1] + self.cuts[1:]) / na
        self.mid = float(bc["middle"])
        self.r0 = float(prof(self.s0))
        self.m = max(2, int(round((1.0 - self.mid) * na / (2.0 * np.pi))))      # points down a cut
        self.n_in = [max(2, 2 * int(round(0.5 * (self.cuts[k + 1] - self.cuts[k]) * self.mid))) for k in range(n)]
        self.step = float(cfg["ring_step"])

    def template(self, k):
        """Per vertex of lobe k's ring: the body ring index of its arc (-1 elsewhere), its angle round the tail, how far
        out from the axis it lies where the body parts (1 on the arc), and its angle round the lobe's ellipse."""
        if k == self.n:                                 # the middle one: its circle, along the lobes' inner arcs
            ang = np.concatenate([2.0 * np.pi * (self.cuts[j] + (self.cuts[j + 1] - self.cuts[j]) *
                                                 np.arange(self.n_in[j]) / self.n_in[j]) / self.na
                                  for j in range(self.n)])
            return np.full(len(ang), -1), ang, np.full(len(ang), self.mid), ang
        c0, c1, m, ni, mid = int(self.cuts[k]), int(self.cuts[k + 1]), self.m, self.n_in[k], self.mid
        J = np.arange(c0, c1 + 1)
        a0, a1 = 2.0 * np.pi * c0 / self.na, 2.0 * np.pi * c1 / self.na
        down = 1.0 - (1.0 - mid) * np.arange(1, m + 1) / m             # down the second cut, to the circle
        back = a1 - (a1 - a0) * np.arange(1, ni) / ni                   # back along the circle
        up = mid + (1.0 - mid) * np.arange(m) / m                       # up the first cut
        idx = np.concatenate([J % self.na, np.full(m + ni - 1 + m, -1)])
        ang = np.concatenate([2.0 * np.pi * J / self.na, np.full(m, a1), back, np.full(m, a0)])
        rho = np.concatenate([np.ones(len(J)), down, np.full(ni - 1, mid), up])
        span = a1 - a0                                 # round the slice's edge (the body's radius as 1)
        pos = np.concatenate([ang[:len(J)] - a0, span + (1.0 - down), span + (1.0 - mid) + mid * (a1 - back),
                              span * (1.0 + mid) + (1.0 - mid) + (up - mid)])
        per = span * (1.0 + mid) + 2.0 * (1.0 - mid)
        theta = 2.0 * np.pi * pos / per - np.pi * span / per
        return idx, ang, rho, theta

    def _taper(self, k, s):
        """Lobe k's size at arc lengths s as a share of its size where the body parts: it narrows by `taper` along its
        length and ends in a round point; and its share of the way t."""
        t = np.clip((np.asarray(s, float) - self.s0) / (self.ends[k] - self.s0), 0.0, 1.0)
        return (1.0 - float(self.cfg["taper"]) * t) * np.sqrt(1.0 - t * t), t

    def _shape(self, k, s):
        """(centre (n, 3), out R, round Q, half-depth a, half-width b) of lobe k's ellipse at arc lengths s (for the
        middle one R is the frame's N with its flattening and Q its B). A lobe round the middle rides on it: its inner
        side on the middle one's surface, `part` x t^2 off it toward its end."""
        C, T, N, B = self.frames.at(s)
        g_mid, _ = self._taper(self.n, s)
        r_mid = self.mid * self.r0 * g_mid
        if k == self.n:
            return C, self.flat * N, B, r_mid, r_mid
        g, t = self._taper(k, s)
        d = self.flat * np.cos(self.phi[k]) * N + np.sin(self.phi[k]) * B
        dl = np.linalg.norm(d, axis=-1)
        R = d / dl[:, None]
        size = self.cfg["size"]
        a = float(size[0]) * self.r0 * dl * g
        cen = C + (r_mid * dl + a + float(self.cfg["part"]) * t * t)[:, None] * R
        return cen, R, np.cross(T, R), a, float(size[1]) * self.r0 * g

    def lobe(self, k):
        """(s per ring (nr,), points (nr, rs, 3), end point (3,)) of lobe k: rings from where the body parts toward its
        end, bunched where it rounds off; the end point closes it."""
        idx, ang, rho, theta = self.template(k)
        e_k = float(self.ends[k])
        sd = np.linspace(self.s0, e_k, 600)
        _, _, _, a, b = self._shape(k, sd)
        ml = arclen(np.stack([sd, 0.5 * (a + b)], -1))
        nb = max(6, int(np.ceil(ml[-1] / self.step)))
        s_r = np.interp(np.linspace(0.0, ml[-1], nb + 1), ml, sd)[:-1]
        C, T, N, B = self.frames.at(s_r)
        pie = C[:, None] + (rho * self.r0)[None, :, None] * (
            self.flat * np.cos(ang)[None, :, None] * N[:, None] + np.sin(ang)[None, :, None] * B[:, None])
        cen, R, Q, a, b = self._shape(k, s_r)
        ell = cen[:, None] + (a[:, None] * np.cos(theta)[None])[..., None] * R[:, None] + \
            (b[:, None] * np.sin(theta)[None])[..., None] * Q[:, None]
        t = (s_r - self.s0) / (e_k - self.s0)
        w = smoothstep(t / float(self.cfg["round"]))[:, None, None] if k < self.n else 1.0
        return s_r, (1.0 - w) * pie + w * ell, self._shape(k, np.array([e_k]))[0][0]


class Surface:
    """The tail's closed surface: the body, rings of `around` vertices from the cut at the skin to where the brush parts
    (its last ring is where the lobes start) and one vertex closing the cut; then each lobe of the brush, its rings and
    one vertex at its end (where the body parts the lobes share the vertices of their cuts and of the middle one's
    circle). Holds the smooth body (`base`, `normal`), per body vertex the arc length `s`, the fur's displacement `h`,
    the body's finished vertices `V` and the lobes."""

    def __init__(self, cfg, frames, prof, fur, brush, skin):
        self.cfg = cfg
        na = int(cfg["around"])
        flat = float(cfg["flat"])
        step = float(cfg["ring_step"])
        self.na = na
        phi = 2.0 * np.pi * np.arange(na) / na

        def meridian(s):                                # smooth body points round a ring, an arc length per meridian
            C, T, N, B = frames.at(s)
            r = np.where(s >= 0, prof(np.maximum(s, 0.0)), prof(0.0))
            return C + flat * r[:, None] * np.cos(phi)[:, None] * N + r[:, None] * np.sin(phi)[:, None] * B

        # rings by arc length along the surface (the meridian of the widest side), the last where the brush parts
        s_end = brush.s0
        sd = np.linspace(0.0, s_end, 4000)
        ml = arclen(np.stack([sd, prof(sd)], -1))
        nb = max(8, int(np.ceil(ml[-1] / step)))
        s_rings = np.interp(np.linspace(0.0, ml[-1], nb + 1), ml, sd)
        # the cut at the skin: each meridian starts where it leaves the skin; the first `k_cut` rings share out the room
        embed = float(cfg["embed"])
        if skin is not None:
            s_cut = _skin_cut(meridian, skin, na, -embed, 0.06, float(cfg["skin_gap"]))
        else:
            s_cut = np.zeros(na)
        k_cut = int(np.searchsorted(s_rings, s_cut.max() + 2.0 * step)) + 1
        S = np.repeat(s_rings[:, None], na, axis=1)
        for i in range(k_cut):
            S[i] = s_cut + (s_rings[k_cut] - s_cut) * (i / k_cut)
        self.s = S                                      # (rings, na) arc length per vertex
        self.k_cut, self.s_cut = k_cut, s_cut
        nr = len(s_rings)
        base = np.stack([meridian(S[i]) for i in range(nr)], 0)               # (nr, na, 3)
        self.C_r = np.stack([frames.at(S[i])[0] for i in range(nr)], 0)
        self.base, self.normal = base, self._grid_normals(base)
        self.h = fur.field(S, np.repeat(phi[None], nr, 0), prof)[0]
        self.V = base + self.h[..., None] * self.normal
        self.cen = base[0].mean(axis=0)
        if skin is not None:                            # the middle of the cut, off the skin like its rim
            t0 = frames.at([float(s_cut.mean())])[1][0]
            lo, hi = 0.0, 0.05
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if skin((self.cen + mid * t0)[None])[0] >= float(cfg["skin_gap"]) else (mid, hi)
            self.cen = self.cen + hi * t0
        self.t_start = frames.at([float(s_cut.mean())])[1][0]
        self.nr = nr
        self.brush = brush
        self.lobes = []
        for k in range(brush.n + 1):                    # a lobe's first ring takes its arc from the body's last
            s_k, P_k, end = brush.lobe(k)
            idx = brush.template(k)[0]
            P_k[0, idx >= 0] = self.V[-1, idx[idx >= 0]]
            self.lobes.append((s_k, P_k, end))

    def _grid_normals(self, G):
        """Normals of a body grid (rings, na, 3): round the ring x along it (one-sided at its first and last ring),
        turned away from the centre line."""
        dphi = np.roll(G, -1, axis=1) - np.roll(G, 1, axis=1)
        ds = np.empty_like(G)
        ds[1:-1] = G[2:] - G[:-2]
        ds[0], ds[-1] = G[1] - G[0], G[-1] - G[-2]
        n = unit(np.cross(dphi, ds))
        flip = (n * (G - self.C_r)).sum(-1) < 0
        n[flip] = -n[flip]
        return n

    @staticmethod
    def _lobe_normals(P, end):
        """Normals of a lobe's rings (nr, rs, 3) closed by its end point: round the ring x along it, turned away
        from each ring's middle."""
        ext = np.concatenate([P, end[None, None].repeat(P.shape[1], 1)], 0)
        dth = np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)
        ds = np.empty_like(P)
        ds[1:] = ext[2:] - ext[:-2]
        ds[0] = P[1] - P[0]
        n = unit(np.cross(dth, ds))
        flip = (n * (P - P.mean(axis=1, keepdims=True))).sum(-1) < 0
        n[flip] = -n[flip]
        return n

    def arrays(self, shade):
        """(verts (n, 3), faces, corner uv, vertex s, vertex normals) of the closed surface: the body's grid, the centre
        of the cut, then the lobes' vertices (those where the body parts once, shared by the lobes that meet there),
        each lobe's end last. u = phi / 2 pi (a face across the seam takes u + 1 on its far side), v = s / length.
        The body's normals are its smooth ones with `shade` of its tufts' own; the lobes' their own grids'."""
        nr, na, L = self.nr, self.na, float(self.cfg["length"])
        body_n = unit((1.0 - shade) * self.normal + shade * self._grid_normals(self.V))
        Vs, s_v, nrm = [self.V.reshape(-1, 3), self.cen[None]], [self.s.reshape(-1), [float(self.s[0].mean())]], \
            [body_n.reshape(-1, 3), -self.t_start[None]]
        cen_i = nr * na
        u_ring = np.arange(na + 1) / na
        v_rows = np.clip(self.s / L, 0.0, 1.0)
        faces, uv = [], []
        for i in range(nr - 1):
            for j in range(na):
                k = j + 1
                faces.append([i * na + j, i * na + k % na, (i + 1) * na + k % na, (i + 1) * na + j])
                uv += [(u_ring[j], v_rows[i, j]), (u_ring[k], v_rows[i, k % na]), (u_ring[k], v_rows[i + 1, k % na]),
                       (u_ring[j], v_rows[i + 1, j])]
        for j in range(na):
            k = j + 1
            faces.append([cen_i, k % na, j])
            uv += [((j + 0.5) / na, 0.0), (u_ring[k], v_rows[0, k % na]), (u_ring[j], v_rows[0, j])]
        n_next = cen_i + 1
        shared, acc = {}, {}                            # where the body parts: position -> vertex, summed normals

        def corners(ids, us, vs):
            us = np.asarray(us, float)
            if us.max() - us.min() > 0.5:               # across the seam: the far side's u + 1
                us = np.where(us < 0.5, us + 1.0, us)
            faces.append([int(x) for x in ids])
            uv.extend(zip(us, vs))

        for k, (s_k, P_k, end) in enumerate(self.lobes):
            idx, ang, _, _ = self.brush.template(k)
            nk, rs = P_k.shape[:2]
            n_k = self._lobe_normals(P_k, end)
            ids = np.empty((nk, rs), int)
            for j in range(rs):                         # the first ring: the body's arc, the shared cuts and circle
                if idx[j] >= 0:
                    ids[0, j] = (nr - 1) * na + idx[j]
                    continue
                key = tuple(np.round(P_k[0, j], 7))
                if key not in shared:
                    shared[key] = n_next
                    n_next += 1
                    Vs.append(P_k[0, j][None])
                    s_v.append([s_k[0]])
                    nrm.append(np.zeros((1, 3)))
                    acc[shared[key]] = np.zeros(3)
                ids[0, j] = shared[key]
                acc[shared[key]] += n_k[0, j]
            ids[1:] = n_next + np.arange((nk - 1) * rs).reshape(nk - 1, rs)
            n_next += (nk - 1) * rs
            Vs.append(P_k[1:].reshape(-1, 3))
            s_v.append(np.repeat(s_k[1:], rs))
            nrm.append(n_k[1:].reshape(-1, 3))
            end_i = n_next
            n_next += 1
            Vs.append(end[None])
            s_v.append([float(self.brush.ends[k])])
            nrm.append(unit(end - P_k[-1].mean(axis=0))[None])
            u = (ang / (2.0 * np.pi)) % 1.0
            v_k = np.clip(s_k / L, 0.0, 1.0)
            v_end = min(float(self.brush.ends[k]) / L, 1.0)
            for i in range(nk - 1):
                for j in range(rs):
                    q = (j + 1) % rs
                    corners([ids[i, j], ids[i, q], ids[i + 1, q], ids[i + 1, j]], [u[j], u[q], u[q], u[j]],
                            [v_k[i], v_k[i], v_k[i + 1], v_k[i + 1]])
            for j in range(rs):
                q = (j + 1) % rs
                uj, uq = u[j], u[q]
                if abs(uq - uj) > 0.5:                  # across the seam: the smaller side's u + 1
                    uj, uq = (uj + 1.0, uq) if uj < uq else (uj, uq + 1.0)
                corners([ids[-1, j], ids[-1, q], end_i], [uj, uq, 0.5 * (uj + uq)], [v_k[-1], v_k[-1], v_end])
        V, N = np.concatenate(Vs, 0), np.concatenate(nrm, 0)
        for i, a in acc.items():
            N[i] = unit(a)
        return V, faces, np.asarray(uv, float), np.concatenate([np.ravel(x) for x in s_v]), N


# ---------------------------------------------------------------- the picture
def paint(cfg, frames, prof, fur, brush, pal, pattern):
    """The texture (H, W, 4) uint8: u round the tail (phi), v along it (v up, s / length): each tuft in its band's
    colour, lit along the side that faces up, shaded under and where it comes out of the body, lighter toward its free
    end; past where the brush parts the last band's colour, lighter toward the tail's end."""
    pc = cfg["paint"]
    W, H = (int(v) for v in pc["size"])
    L, flat = prof.L, float(cfg["flat"])
    phi = 2.0 * np.pi * (np.arange(W) + 0.5) / W
    s = (1.0 - (np.arange(H) + 0.5) / H) * L
    S, PHI = np.meshgrid(s, phi, indexing="ij")
    h, band, along, tuck = fur.field(S, PHI, prof)
    band = np.where(band < 0, _band_of(S, fur.band_bounds), band)
    toward = smoothstep((along - 0.4) / 0.6)            # toward a free end
    past = S >= brush.s0
    band[past] = len(pattern) - 1
    toward[past] = (S[past] - brush.s0) / (L - brush.s0)
    tuck[past] = 0.0
    C, T, N, B = frames.at(s)
    r = np.maximum(prof(s), 1e-4)
    # normal of the elliptic section (the slope along the tail left out: the light is soft)
    nN = np.cos(PHI) / (flat * r[:, None])
    nB = np.sin(PHI) / r[:, None]
    n = unit(nN[..., None] * N[:, None, :] + nB[..., None] * B[:, None, :])
    up = (n * UP).sum(-1)
    lit = smoothstep((up - 0.15) / 0.75)
    under = smoothstep((-up - 0.05) / 0.85)
    img = np.zeros((H, W, 3))
    for k, name in enumerate(pattern):
        m = band == k
        if not m.any():
            continue
        base, light, shadow = pal[name], pal[f"{name}_light"], pal[f"{name}_shadow"]
        col = base[None] + (light - base)[None] * (float(pc["light"]) * lit[m])[:, None]
        col = col + (shadow - col) * (float(pc["under"]) * under[m])[:, None]
        col = col + (shadow - col) * (float(pc["tuck"]) * tuck[m])[:, None]
        col = col + (light - col) * (float(pc["tips"]) * toward[m])[:, None]
        img[m] = col
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


def _band_of(s, bounds):
    """Band index of arc lengths s by the band boundaries (s before the first boundary: band 0)."""
    return np.clip(np.searchsorted(np.asarray(bounds)[1:-1], s, side="right"), 0, len(bounds) - 2)


def band_bounds(cfg, L):
    """Arc lengths of the band boundaries [from, ..., L] (len(pattern) + 1)."""
    bc = cfg["bands"]
    w = np.asarray(bc["widths"], float)
    if len(w) != len(bc["pattern"]):
        raise ValueError("[tail.bands] widths needs one number per colour of pattern")
    a = float(bc["from"])
    return a + (L - a) * np.concatenate([[0.0], np.cumsum(w) / w.sum()])


# ---------------------------------------------------------------- rig
def _ramp(pair, f):
    a, b = (float(v) for v in pair)
    return a + (b - a) * np.asarray(f, float)


def build_rig(ctx, cfg, P, s, prof, root):
    """The aim bone, its static body and the chain (see the module docstring) -> (Rig, Chain, info)."""
    rc = cfg["rig"]
    L = prof.L
    nb = int(rc["bones"])
    rig = Rig()
    colliders = CC.static_colliders(ctx)
    k_rad = float(rc["radius"])

    def at(x):
        return np.stack([np.interp(x, s, P[:, k]) for k in range(3)], -1)
    s0 = 0.0
    for x in np.arange(0.0, 0.25, 0.002):              # the chain starts where its first capsule clears the colliders
        s0 = float(x)
        q0, q1 = at(x), at(x + (L - x) / nb)
        if not colliders:
            break
        gap, _ = CC.capsule_clearance(colliders, q0, q1, k_rad * float(prof(x + 0.5 * (L - x) / nb)), step=0.004)
        if gap >= float(rc["clear"]) + 0.001:
            break
    s_pts = np.linspace(s0, L, nb + 1)
    pts = at(s_pts)
    radius = k_rad * prof(0.5 * (s_pts[:-1] + s_pts[1:]))
    low = ctx.find_body("下半身")
    rig.bones.append(Bone(name=AIM, head=tuple(float(x) for x in root), tail=tuple(float(x) for x in pts[0]),
                          parent="下半身", name_en="tail aim", tail_bone=f"{CHAIN}1"))
    rig.bodies.append(RigidBody(name=AIM, bone=AIM, shape="sphere", size=(0.008, 0.0, 0.0),
                                location=tuple(float(x) for x in pts[0]), mode="static", group=15,
                                no_collide=tuple(range(16)), mass=1.0))
    f = np.linspace(0.0, 1.0, nb)
    ch = rig.chain(CHAIN, pts, anchor=AIM, kind="tail", radius=radius, anchor_body=AIM, hold=float(rc["hold"]),
                   name_en="tail ")
    mass, spring, swing = _ramp(rc["mass"], f), _ramp(rc["spring"], f), _ramp(rc["swing"], f)
    lin, ang = (float(v) for v in rc["damping"])
    tw = float(rc["twist"])
    for k, rb in enumerate(rig.bodies[1:]):
        rb.mass, rb.damping = float(mass[k]), (lin, ang)
    for k, jt in enumerate(rig.joints):
        sw, sp = float(swing[k]), float(spring[k])
        jt.rot_lo, jt.rot_hi, jt.spring_rot = (-sw, -sw, -tw), (sw, sw, tw), (sp, sp, 0.5 * sp)
    gaps = CC.chain_clearance(colliders, pts, radius) if colliders else []
    info = {"chain_start": s0, "s_pts": s_pts, "points": pts, "radius": radius, "clearance": [g for g, _ in gaps],
            "clearance_body": [w for _, w in gaps], "mass": mass, "spring": spring, "swing": swing,
            "anchor_body": low.name if low is not None else None}
    return rig, ch, info


def chain_weights(ch, s_pts, s_vert, held=None, body="下半身"):
    """Weights of vertices by their arc length on the centreline (`s_pts`: the centreline's arc length at the chain's
    points), mapped onto the chain's own arc, so a vertex on the inside of a bend never takes a bone of another part
    of the tail; vertices before the chain's start are the aim bone's. `held` (per vertex, 0..1): the share `body`
    keeps (the base, where it meets the skin), so a turned aim bone bends the tail at its root instead of swinging
    its base into her."""
    s_vert = np.asarray(s_vert, float)
    w = ch.weights(s=np.interp(s_vert, s_pts, ch.s) + np.minimum(s_vert - s_pts[0], 0.0))
    if held is not None:
        held = np.clip(np.asarray(held, float), 0.0, 1.0)
        w = {k: v * (1.0 - held) for k, v in w.items()}
        w[body] = w.get(body, 0.0) + held
    return w


# ---------------------------------------------------------------- checks
def _skin_distance(ctx, centre, reach, back):
    """A signed distance (< 0 inside) to the body's skin near `centre`, or None without a body mesh. The sign is told by
    a ray toward `back` (an odd number of crossings: inside), not by the nearest face, whose normal misleads in a
    crease such as the cleft under the tail's root."""
    body = ctx.parts.get("body")
    if body is None:
        return None
    V, F = _part_tris(body, keep=lambda m: m.name == "body")
    if V is None:
        return None
    near = np.linalg.norm(V[F].mean(axis=1) - centre[None], axis=1) < reach
    if not near.any():
        return None
    Fn = F[near]
    dirn = np.asarray(back, float)

    def signed(P):
        P = np.atleast_2d(np.asarray(P, float))
        d = np.abs(CC.mesh_distance(P, V, Fn))
        return np.where(_odd_crossings(P, V, Fn, dirn), -d, d)
    return signed


def _part_tris(part, keep=None):
    """(V, triangles) of a part's meshes (those `keep` accepts), or (None, None)."""
    Vs, Fs, off = [], [], 0
    for m in part.meshes:
        if keep is not None and not keep(m):
            continue
        V = np.asarray(m.verts, float)
        tris = [[f[0], f[k], f[k + 1]] for f in m.faces for k in range(1, len(f) - 1)]
        if not tris:
            continue
        Vs.append(V)
        Fs.append(np.asarray(tris, int) + off)
        off += len(V)
    if not Vs:
        return None, None
    return np.concatenate(Vs, 0), np.concatenate(Fs, 0)


def _closed(V, F):
    """True when every edge of the triangles F is shared by exactly two of them (positions welded)."""
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    E = np.sort(inv.reshape(-1)[np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])], axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return bool((cnt == 2).all())


def _odd_crossings(P, V, F, d):
    """Per point of P, whether a ray from it along `d` crosses the triangles (V, F) an odd number of times."""
    A = V[F[:, 0]]
    e1, e2 = V[F[:, 1]] - A, V[F[:, 2]] - A
    hv = np.cross(d[None], e2)
    det = (e1 * hv).sum(1)
    ok = np.abs(det) > 1e-14
    A, e1, e2, hv, inv = A[ok], e1[ok], e2[ok], hv[ok], 1.0 / det[ok]
    out = np.zeros(len(P), bool)
    for a in range(0, len(P), 64):
        sv = P[a:a + 64, None, :] - A[None]
        u = inv[None] * (sv * hv[None]).sum(-1)
        q = np.cross(sv, e1[None])
        v = inv[None] * (q @ d)
        t = inv[None] * (q * e2[None]).sum(-1)
        out[a:a + 64] = ((u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum(1) % 2 == 1
    return out


RAYS = (unit(np.array([0.31, 0.17, 0.93])), unit(np.array([-0.53, 0.71, -0.45])))


def part_gaps(ctx, V, s_v, s_out, name, reach=0.03):
    """{part: {"gap": smallest distance (m, < 0 inside or behind it) of the tail's vertices past `s_out` that come
    within `reach` of the part (None: none do), "behind": how many of them lie inside it (a closed part: an odd number
    of crossings along each of two rays) or behind its surface (a sheet such as a garment: by its nearest face)}} for
    every part built before the tail."""
    out = {}
    pts = V[s_v >= s_out]
    lo, hi = pts.min(axis=0) - reach, pts.max(axis=0) + reach
    for pname, part in ctx.parts.items():
        if pname == name:
            continue
        PV, PF = _part_tris(part)
        if PV is None:
            continue
        out[pname] = {"gap": None, "behind": 0}
        cen = PV[PF].mean(axis=1)
        near_t = np.all((cen >= lo) & (cen <= hi), axis=1)
        if not near_t.any():
            continue
        corners = PV[PF[near_t]].reshape(-1, 3)
        near_p = np.all((pts >= corners.min(axis=0) - reach) & (pts <= corners.max(axis=0) + reach), axis=1)
        if not near_p.any():
            continue
        Q = pts[near_p]
        d = CC.mesh_distance(Q, PV, PF[near_t])
        keep = np.abs(d) <= reach
        if not keep.any():
            continue
        Q, d = Q[keep], d[keep]
        if _closed(PV, PF):
            inside = _odd_crossings(Q, PV, PF, RAYS[0]) & _odd_crossings(Q, PV, PF, RAYS[1])
        else:
            inside = d < 0
        g = np.where(inside, -np.abs(d), np.abs(d))
        out[pname] = {"gap": float(g.min()), "behind": int(inside.sum())}
    return out


# ---------------------------------------------------------------- the part
@builder("tail", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    pal = WR.colours(ctx, "tail", COLORS)
    pattern = list(cfg["bands"]["pattern"])
    for name in set(pattern):
        if name not in ("dark", "tan"):
            raise ValueError(f"[tail.bands] pattern: {name!r} is not a colour of the tail (dark, tan)")
    root = ctx.land_point("tail_root")
    body = ctx.parts.get("body")
    nrm = np.asarray(body.info.get("tail_normal", [0.0, 0.94, 0.35]), float) if body is not None else \
        np.array([0.0, 0.94, 0.35])
    back = CC.rotate(unit(np.array([nrm[0], nrm[1], 0.0])), [0.0, 0.0, 1.0], -np.radians(float(cfg["turn"])))
    P, s = centreline(cfg, root, back)
    prof = Profile(cfg)
    L = prof.L
    frames = Frames(P, s, float(cfg["embed"]))
    bounds = band_bounds(cfg, L)
    brush = Brush(cfg, frames, prof, int(cfg["around"]))
    fur = Fur(cfg, L, bounds, pattern, prof, ctx.rng_for(cfg["seed"] + ".tufts"), brush.s0)
    skin = _skin_distance(ctx, np.asarray(root, float), 0.15, back)
    surf = Surface(cfg, frames, prof, fur, brush, skin)
    V, F, UV, s_v, normals = surf.arrays(float(cfg["tufts"]["shade"]))
    # ---- one material: the bands are in the picture, so their scalloped edges never follow the faces' stair-steps; the
    # toon shade is the mean of the two colours' own (each band's `_shadow` over its colour)
    tex = ctx.save_png("fur", paint(cfg, frames, prof, fur, brush, pal, pattern))
    pc = cfg["paint"]
    ratios = [np.clip(pal[f"{n}_shadow"] / np.maximum(pal[n], 1e-3), 0.0, 1.0) for n in sorted(set(pattern))]
    toon = ctx.save_png("toon", TEX.toon_ramp(tuple(np.exp(np.mean(np.log(ratios), axis=0))),
                                              edge=float(pc["toon"]["edge"]), soft=float(pc["toon"]["soft"])))
    edge = np.maximum(pal["dark_shadow"] * 0.85, TEX.srgb("#2a1714"))
    mats = [Material(MAT, name_en="tail", diffuse=(0.9, 0.9, 0.9, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                     ambient=(0.5, 0.5, 0.5), texture=tex, toon=toon, edge=True,
                     edge_color=(*(float(x) for x in edge), 1.0), edge_size=float(pc["edge_size"]),
                     comment="tail fur in bands (generated)")]
    # (normals: the smooth body's with `shade` of the tufts' own, so the light rounds the whole tail softly and the
    # tufts show in its outline and its bands rather than as lumps; the lobes' their own)
    # ---- rig and weights
    rig, ch, rinfo = build_rig(ctx, cfg, P, s, prof, root)
    s_skin = s_v - np.concatenate([np.tile(surf.s_cut, surf.nr), [float(surf.s_cut.mean())],
                                   np.full(len(s_v) - surf.nr * surf.na - 1, float(surf.s_cut.mean()))])
    w = chain_weights(ch, rinfo["s_pts"], s_v, held=1.0 - smoothstep(s_skin / float(cfg["rig"]["base"])))
    # ---- numbers
    s_out = fur.first
    got = part_gaps(ctx, V, s_v, s_out, "tail")
    for pname, g in got.items():
        if g["behind"]:
            ctx.log(f"WARNING {g['behind']} vertices of the tail lie inside {pname} (deepest "
                    f"{-1000 * g['gap']:.1f} mm)")
    bad = [k for k, g in enumerate(rinfo["clearance"]) if g < float(cfg["rig"]["clear"]) - 1e-4]
    if bad:
        ctx.log(f"WARNING chain bones {bad} come within {min(rinfo['clearance']):.4f} m of "
                f"{rinfo['clearance_body'][bad[0]]} (want {cfg['rig']['clear']})")
    i_far = int(np.argmax(((P - P[0]) * back[None]).sum(-1)))
    info = {
        "length": L, "root": np.asarray(root, float), "back": back,
        "path": {"far": P[i_far], "far_s": float(s[i_far]), "tip": P[-1], "points": P[::40].copy()},
        "radius": {"stations": [[float(f), float(prof(f * L))] for f in (0.0, 0.2, 0.5, 0.6, 0.65, 0.8, 1.0)],
                   "flat": float(cfg["flat"])},
        "brush": {"parts": float(brush.s0), "ends": [float(e) for e in brush.ends],
                  "points": np.stack([lobe[2] for lobe in surf.lobes]), "phi": [float(p) for p in brush.phi]},
        "fur": {"start": float(fur.first), "rows": [{k: (v if isinstance(v, (bool, np.bool_, str)) else float(v))
                                                     for k, v in r.items()} for r in fur.rows]},
        "bands": [[pattern[k], float(bounds[k]), float(bounds[k + 1])] for k in range(len(pattern))],
        "chain": {"bones": list(ch.names), "start": rinfo["chain_start"], "points": rinfo["points"],
                  "radius": rinfo["radius"], "clearance": rinfo["clearance"], "mass": rinfo["mass"],
                  "spring": rinfo["spring"], "swing": rinfo["swing"]},
        "aim": AIM, "gaps": got, "cut": {"rings": int(surf.k_cut), "s": surf.s_cut},
        "vertices": int(len(V)),
        "centreline": P, "s": s,
    }
    gap_txt = ", ".join(f"{k} " + ("clear (>30 mm)" if g["gap"] is None else
                                   f"{1000 * g['gap']:.1f} mm, {g['behind']} behind") for k, g in got.items())
    ctx.log(f"tail: gaps to {gap_txt}")
    ctx.log(f"tail: {L:.2f} m, far {np.round(P[i_far], 3).tolist()}, tip {np.round(P[-1], 3).tolist()}, "
            f"{len(fur.rows)} rows of tufts, {brush.n} lobes from {brush.s0:.3f} m, chain from "
            f"{rinfo['chain_start']:.3f} m, {len(V)} vertices")
    mesh = Mesh(name="tail", verts=V, faces=F, uv=UV, mats=[MAT], weights=w, normals=normals)
    return Part("tail", meshes=[mesh], materials=mats, bones=rig.bones, bodies=rig.bodies, joints=rig.joints,
                frames={"尻尾": [b.name for b in rig.bones]}, info=info)
