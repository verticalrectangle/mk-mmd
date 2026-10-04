"""Grip style `neck`: a fretting hand on a guitar-like neck, solved on the model's own skin (numpy + scipy, no bpy). Part of
mkmmd.solvers.grip (`solve("neck", hand, prop, ...)`); see docs/design.md: Grips.

The hand wraps the neck from below: the thumb pad on the back of the neck, the pressing fingers ARCHED onto the board with
their pads on the strings they play (each between the wires it plays and the next one toward the nut), the other fingers
hovering above the strings, relaxed. Everything is in the NECK FRAME N (`mkmmd.core.fretting.solver_prop` makes the prop
dict): origin on the board's centre line under the position fret's wire, x toward the nut, z out of the board (the board
surface is z = 0), y = z cross x (toward the low E side). The neck solid is the part below z = 0, a lower half superellipse
whose width and depth vary linearly along x; the skin of the hand stays out of it.

prop (metres, in N)
    targets   [{finger, point [x, y, z], x [lo, hi], radius}]   pressing fingers: the pad's contact point on the string,
              and the range along x the pad may sit in (behind the wire it plays)
    section   {x [xa, xb], width [wa, wb], depth [da, db], p}
    frets     (optional) x of the fret wires

Unknowns: the 21 joint angles (DOF order) and the frame N in the hand's rest frame (translation mm, rotation vector rad, from a
design start whose axes follow the hand's anatomy: the thumb side toward the nut, the fingers across the neck, the palm toward
the neck's back). Residuals: each pad's lowest skin on its string's height (a softmin) and its contact point on the string
within a few millimetres across and inside the range along it, the distal phalanx steep onto the board (an arch, not a
flat finger), the thumb pad on the neck's back at the middle of the pressing fingers, no skin inside the neck, the middle
and proximal phalanges above the strings, the hovering fingers above them, no finger inside another, joint priors of a
relaxed arched hand."""
import math

import numpy as np
from scipy.optimize import least_squares

from . import grip as G

FOUR, DOF = G.FOUR, G.DOF
# joint priors (mcp pip dip spread) of a pressing finger (arched: PIP bent past a right angle, DIP almost straight) and of
# one hovering over the strings, then the thumb (palmar abduction, radial abduction, roll, mcp, ip)
PRIOR_PRESS = np.array([42.0, 72.0, 26.0, 0.0])
PRIOR_HOVER = np.array([30.0, 44.0, 18.0, 0.0])
PRIOR_THUMB = np.array([30.0, 8.0, -10.0, 14.0, 8.0])
SIGMA_FINGER = np.array([14.0, 16.0, 12.0, 6.0])
SIGMA_THUMB = np.array([14.0, 12.0, 25.0, 12.0, 12.0])
LO_FINGER = np.array([0.0, 5.0, -5.0, -12.0])
HI_FINGER = np.array([92.0, 105.0, 75.0, 12.0])
LO_THUMB, HI_THUMB = G.PINCH_LO[8:13], G.PINCH_HI[8:13]
PLACE_MM, PLACE_RAD = 70.0, 1.3                  # the frame's translation (mm) and rotation (rad) bounds around the start
PRIOR_W = 0.75
PAD_TOL_Y = 0.0025                               # a pad's contact point may be this far (m) across from its string
STEEP = math.cos(math.radians(38.0))             # distal phalanx within 38 degrees of straight into the board
THUMB_FACE = math.cos(math.radians(40.0))
THUMB_X = (-0.012, 0.014)                        # the thumb pad's place along x relative to the pressing fingers' mean
THUMB_Y = 0.010                                  # ... and across (about the board's centre line)
CLEAR_PRESS, CLEAR_HOVER = 0.004, 0.007          # m above the strings: proximal / middle phalanges of pressing fingers, hovering fingers
NFEV = (350, 500, 250)


def _hinge(v):
    return np.maximum(v, 0.0)


class _Neck:
    """The neck problem on one hand (see the module doc); x = 21 joint angles then the frame's translation (mm) and rotation
    vector (rad) relative to the design start."""

    X_SCALE = np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB, np.full(3, 8.0), np.full(3, 0.15)]

    def __init__(self, hand, targets, section, strings=None, p_exp=None):
        self.hand = hand
        hd, F = hand, hand.F
        if not targets:
            raise ValueError("a neck grip needs at least one pressing finger (targets)")
        self.targets = {}
        for t in targets:
            f = t["finger"]
            if f not in FOUR:
                raise ValueError(f"neck target finger {f!r} must be one of {FOUR}")
            self.targets[f] = {"point": np.asarray(t["point"], float), "x": tuple(float(v) for v in t["x"]),
                               "radius": float(t.get("radius", 0.0006))}
        sec = section
        self.xa, self.xb = (float(v) for v in sec["x"])
        self.wa, self.wb = (float(v) for v in sec["width"])
        self.da, self.db = (float(v) for v in sec["depth"])
        self.p = float(sec.get("p", 2.6))
        self.pad = {f: hd.surface(F[f][2], hd.ventral, 0.0, 1.0, 0.3) for f in FOUR}
        self.pad_t = hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3)
        if min([len(v) for v in self.pad.values()] + [len(self.pad_t)]) < 3:
            raise ValueError("the hand model has too little palmar skin on a finger or thumb pad for a neck grip")
        self.press = [f for f in FOUR if f in self.targets]
        self.hover = [f for f in FOUR if f not in self.targets]
        self.skin = np.where(hd.dom < hd.nb)[0]                           # skin that follows the hand's own bones
        self.skin_mask = np.zeros(len(hd.V), bool)
        self.skin_mask[self.skin] = True
        up = lambda fs, js: np.where(np.isin(hd.dom, [F[f][j] for f in fs for j in js]))[0]     # noqa: E731
        self.low_press = up(self.press, (0, 1))
        self.hover_all = up(self.hover, (0, 1, 2))
        self.clash = G._Clash(hd)
        self.prior = np.r_[np.concatenate([PRIOR_PRESS if f in self.targets else PRIOR_HOVER for f in FOUR]), PRIOR_THUMB,
                           np.zeros(6)]
        self.lo = np.r_[np.tile(LO_FINGER, 4), LO_THUMB, np.full(3, -PLACE_MM), np.full(3, -PLACE_RAD)]
        self.hi = np.r_[np.tile(HI_FINGER, 4), HI_THUMB, np.full(3, PLACE_MM), np.full(3, PLACE_RAD)]
        self.xmean = float(np.mean([self.targets[f]["point"][0] for f in self.press]))
        self.R0 = self.p0 = None

    # ---- the start frame
    def start_frame(self, deg):
        """Design start: the hand's anatomy against N (x toward the nut = the thumb side, the fingers across the neck, the palm
        toward the back of the neck), the origin placing the pressing fingers' pads on their targets on average."""
        hd = self.hand
        y = -hd.chir * hd.along                      # fingers across the neck; the thumb behind it: z = -ventral
        z = -hd.ventral
        R0 = np.stack([np.cross(y, z), y, z], 1)
        P = hd.pose(deg)[3]
        off = [R0 @ self.targets[f]["point"] - P[self.pad[f]].mean(0) for f in self.press]    # frame origin that puts pad f on its target
        return R0, -np.mean(off, axis=0)

    def frame(self, x):
        return G._rotvec(x[24:27]) @ self.R0, self.p0 + x[21:24] * 1e-3

    # ---- the neck solid
    def section_at(self, xs):
        t = np.clip((xs - self.xa) / (self.xb - self.xa), 0.0, 1.0)
        return self.wa + t * (self.wb - self.wa), self.da + t * (self.db - self.da)

    def sdf(self, L):
        """Signed distance (m, + outside) of points in N to the neck solid and the board above it: over the board the distance
        to its top face, inside the section the nearer of the top face and the (ray-scaled) distance to the back, outside it
        the ray-scaled distance along the line from the section's centre."""
        w, d = self.section_at(L[:, 0])
        y, z = L[:, 1], L[:, 2]
        a = np.abs(y) - 0.5 * w
        over = np.where(a <= 0.0, z, np.hypot(np.maximum(a, 0.0), np.maximum(z, 0.0)))
        s = (np.abs(2.0 * y / w) ** self.p + (np.abs(z) / d) ** self.p) ** (1.0 / self.p)
        ray = (1.0 - s) * np.hypot(y, z) / np.maximum(s, 1e-9)
        under = np.where(s < 1.0, -np.minimum(np.abs(z), ray), -ray)
        return np.where(z <= 0.0, under, over)

    # ---- analysis
    def analyse(self, x, stage="all"):
        hd = self.hand
        deg = x[:21]
        Q, D, H, P = hd.pose(deg)
        R, p = self.frame(x)
        L = (P - p) @ R
        A = dict(Q=Q, D=D, H=H, P=P, R=R, p=p, L=L, deg=deg)
        A["dist_dir"] = {f: R.T @ (D[hd.F[f][2]] @ hd.bone_dir[hd.F[f][2]]) for f in FOUR}
        A["thumb_dir"] = R.T @ (D[hd.F["thumb"][2]] @ hd.pad_t)
        if stage != "reach":
            A["sdf"] = self.sdf(L)
            A["clash"] = self.clash.depth(P, H)
        return A

    @staticmethod
    def _contact(L, idx, tau, gap=None):
        """(softmin of the set's gap, the weighted mean contact point): for a pad pressing a string the gap is the skin's
        height z (its lowest part touches); for the thumb it is the distance `gap` to the neck's back."""
        g = L[idx, 2] if gap is None else gap[idx]
        m = float(g.min())
        w = np.exp(-(g - m) / 0.0012)
        return m - tau * math.log(float(np.sum(np.exp(-(g - m) / tau)))), (w / w.sum()) @ L[idx]

    def residuals(self, x, stage="all"):
        A = self.analyse(x, stage)
        L = A["L"]
        tau = 0.0001 if stage == "polish" else 0.0004
        r = []
        xs = []
        for f in self.press:
            t = self.targets[f]
            low, c = self._contact(L, self.pad[f], tau)
            xs.append(c[0])
            r += [(low - (t["point"][2] - 0.0004)) / 0.0005,
                  _hinge(abs(c[1] - t["point"][1]) - PAD_TOL_Y) / 0.002,
                  (_hinge(t["x"][0] - c[0]) + _hinge(c[0] - t["x"][1])) / 0.002,
                  _hinge(STEEP + A["dist_dir"][f][2]) / 0.12]                # the distal phalanx points down onto the board
        sdf_t = self.sdf(L) if stage == "reach" else A["sdf"]
        gap_t, ct = self._contact(L, self.pad_t, tau, sdf_t)
        xm = float(np.mean(xs))
        r += [(gap_t - 0.0003) / 0.0006, _hinge(xm + THUMB_X[0] - ct[0]) / 0.004, _hinge(ct[0] - xm - THUMB_X[1]) / 0.004,
              _hinge(abs(ct[1]) - THUMB_Y) / 0.004, _hinge(THUMB_FACE - A["thumb_dir"][2]) / 0.15]
        r = np.array(r)
        pri = PRIOR_W * (x[:21] - self.prior[:21]) / np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB]
        out = [r, pri]
        if stage != "reach":
            sk = A["sdf"][self.skin]
            z = L[:, 2]
            over = np.abs(L[:, 1]) < 0.5 * self.section_at(L[:, 0])[0] + 0.006       # above the board (strings): clearances
            out += [_hinge(0.0002 - sk) / 0.0002, A["clash"] / 0.0004,
                    _hinge(CLEAR_PRESS - np.where(over[self.low_press], z[self.low_press], 1.0)) / 0.002,
                    _hinge(CLEAR_HOVER - np.where(over[self.hover_all], z[self.hover_all], 1.0)) / 0.003]
        return np.concatenate(out)

    # ---- solving
    def seed(self, k, nfev=NFEV):
        """Reach, grip, polish from the prior pose (k = 0) or a perturbed pose and frame (k > 0)."""
        rng = np.random.default_rng(k)
        deg = np.clip(self.prior[:21] + (rng.normal(0.0, 0.5, 21) * np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB] if k else 0.0),
                      self.lo[:21] + 1e-6, self.hi[:21] - 1e-6)
        self.R0, self.p0 = self.start_frame(deg)
        if k:
            self.R0 = G._rotvec(rng.normal(0.0, 0.25, 3)) @ self.R0
            self.p0 = self.p0 + rng.normal(0.0, 0.005, 3)
        z = np.r_[deg, np.zeros(6)]
        for stage, n, ftol in zip(("reach", "grip", "polish"), nfev, (1e-6, 1e-6, 1e-5)):
            res = least_squares(lambda v, s=stage: self.residuals(v, s), z, bounds=(self.lo, self.hi), x_scale=self.X_SCALE,
                                max_nfev=n, xtol=1e-12, ftol=ftol)
            z = res.x
        return float(res.cost), z, self.R0, self.p0

    def report(self, A):
        """Numbers measured on the final pose: hard minima over the skin."""
        hd, L = self.hand, A["L"]
        sdf = A["sdf"]
        con = {}
        for f in self.press:
            t = self.targets[f]
            idx = self.pad[f]
            low, c = self._contact(L, idx, 0.0001)
            con[f] = {"gap_mm": round(float(L[idx, 2].min() - t["point"][2]) * 1e3, 2),
                      "across_mm": round(float(c[1] - t["point"][1]) * 1e3, 2),
                      "along_mm": round(float(max(t["x"][0] - c[0], c[0] - t["x"][1], 0.0)) * 1e3, 2),
                      "distal_deg": round(math.degrees(math.acos(float(np.clip(-A["dist_dir"][f][2], -1, 1)))), 1)}
        _l, ct = self._contact(L, self.pad_t, 0.0001, sdf)
        con["thumb"] = {"gap_mm": round(float(sdf[self.pad_t].min()) * 1e3, 2), "x_mm": round(float(ct[0]) * 1e3, 1),
                        "y_mm": round(float(ct[1]) * 1e3, 1),
                        "facing_deg": round(math.degrees(math.acos(float(np.clip(A["thumb_dir"][2], -1, 1)))), 1)}
        z = L[:, 2]
        over = np.abs(L[:, 1]) < 0.5 * self.section_at(L[:, 0])[0] + 0.006
        hov = z[self.hover_all][over[self.hover_all]]
        wrist = A["R"].T @ (hd.h[0] - A["p"])
        return {"contacts": con, "penetration_mm": round(max(0.0, -float(sdf[self.skin].min())) * 1e3, 3),
                "finger_clash_mm": round(float(A["clash"].max()) * 1e3, 2),
                "hovering_clear_mm": round(float(hov.min()) * 1e3, 1) if len(hov) else None,
                "wrist_in_N_mm": [round(float(v) * 1e3, 1) for v in wrist],
                "pressing": self.press, "angles_deg": G._angles(A["deg"])}


def _neck_seed(job):
    hand_args, kw, k, nfev = job
    return _Neck(G.HandModel(**hand_args), **kw).seed(k, nfev)


def solve_neck(hand, prop, seeds=6, workers=None, **tuning):
    """Fretting hand on a neck (see the module doc).

    prop    {"targets": [...], "section": {...}} in the neck frame N (mkmmd.core.fretting.solver_prop builds it from a card's
            `use.grip` neck entry, a chord and a position fret); other keys are ignored.
            The frame (`target_in_wrist` is this frame): origin on the board's centre line under the position fret's wire, x
            along the strings toward the nut, z out of the board, y = z cross x.
    seeds   parallel starts (the lowest cost wins); workers: processes (default: all cores, 1 = here)
    tuning  nfev (evaluations of the reach / grip / polish stages)
    report  contacts {finger: gap_mm (the pad's lowest skin against the pressed string's height; 0 = touching, + above), across_mm
            (contact point from the string), along_mm (outside its range behind the wire), distal_deg (the distal phalanx from
            straight into the board)}, thumb {gap_mm, x_mm, y_mm, facing_deg}, penetration_mm (deepest skin in the neck),
            finger_clash_mm, hovering_clear_mm (the free fingers over the strings), wrist_in_N_mm, angles_deg."""
    unknown = set(tuning) - {"nfev"}
    if unknown:
        raise ValueError(f"unknown neck tuning {sorted(unknown)} (have nfev)")
    if int(seeds) < 1:
        raise ValueError("neck needs at least one seed")
    if not hasattr(prop, "get") or not prop.get("targets") or not prop.get("section"):
        raise ValueError("a neck prop is a dict with `targets` (pressing fingers) and `section` (the neck's solid)")
    nfev = tuple(int(n) for n in tuning.get("nfev", NFEV))
    kw = dict(targets=prop["targets"], section=prop["section"])
    prob = _Neck(hand, **kw)
    runs = G._run_seeds(_neck_seed, [(G._hand_args(hand), kw, s, nfev) for s in range(int(seeds))], workers)
    cost, z, prob.R0, prob.p0 = min(runs, key=lambda c: c[0])
    A = prob.analyse(z, "polish")
    return G._result("neck", hand, A["Q"], A["R"], A["p"], prob.report(A),
                     solver={"version": G.VERSION, "cost": cost, "seeds": int(seeds), "x": z})
