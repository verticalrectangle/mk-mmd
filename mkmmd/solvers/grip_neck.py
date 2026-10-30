"""Grip style `neck`: a fretting hand on a guitar-like neck, solved on the model's own skin (numpy + scipy, no bpy). Part of
mkmmd.solvers.grip (`solve("neck", hand, prop, ...)`); see docs/design.md: Grips.

The hand wraps the neck from below: the thumb pad on the back of the neck, the pressing fingers ARCHED onto the board with
their pads on the strings they play (each just behind the wire it plays), the other fingers hovering just above the strings,
relaxed. Everything is in the NECK FRAME N (`mkmmd.core.fretting.solver_prop` makes the prop dict): origin on the board's
centre line under the position fret's wire, x toward the nut, z out of the board (the board surface is z = 0), y = z cross x
(toward the low E side). The neck solid is the part below z = 0, a lower half superellipse whose width and depth vary
linearly along x; the skin of the hand stays out of it.

prop (metres, in N)
    targets   [{finger, point [x, y, z], x [lo, hi], radius, string, fret}]   the fingers with a place on the strings: the
              fingertip's contact point (on the pressed string `press` of the gap behind its wire, or just over a string it
              hovers over), the range along x it may sit in, and the place's string and fret (what a reference compares, and
              which fingers share a fret on neighbouring strings)
    section   {x [xa, xb], width [wa, wb], depth [da, db], p}
    slots     (optional) x of the four fingers' frets at the position (index .. little, one finger per fret; None past the
              neck): a free finger hovers over its own, and the thumb's place reaches to the index and middle fingers'
    position  (optional) the position fret (a reference at another position is a shift)
    frets     (optional) x of the fret wires

Unknowns: the 21 joint angles (DOF order) and the frame N in the hand's rest frame (translation mm, rotation vector rad, from a
start: a design start whose axes follow the hand's anatomy, the thumb side toward the nut, the fingers across the neck, the
palm toward the neck's back; or a reference). Residuals: each target finger's fingertip (all the skin of its distal phalanx:
what is seen to touch) with its lowest skin on its string's height (a softmin) and the centroid of that lowest skin ON the
string: within PAD_TOL_Y across it and PAD_TOL_X along it from the target point (PAD_TOL_CROWD beside a finger on the same
fret of the next string: two fingertips side by side stagger), inside the range behind the wire, touching with the pad (its
ventral skin within PAD_TOUCH of the lowest), the distal phalanx steep onto the board (an arch, not a flat finger), the thumb
pad on the neck's back behind the position's index and middle fingers (THUMB_X about their slots' midpoint, the same place
for every shape there; about the target fingers' mean without slots), no skin inside the neck, the middle and proximal
phalanges above the strings, the hovering fingers' pads over the board between CLEAR_HOVER and HOVER_MAX above it and over
their own fret (HOME_TOL), the knuckles (the fingers' base joints) at most KNUCKLE_Z out of the board, no finger inside
another, joint priors of a relaxed arched hand. The knuckles are what keeps the hand UNDER the neck: contacts alone are also
met by a hand that stands in front of the board, its knuckles a few centimetres out and its fingers curling back onto the
strings, which reads from the front as a claw over the fretboard. Given the arm (`arm` in N, mkmmd.solvers.grip._Arm: the
elbow the arm IK makes for its pole) the wrist stays within the arm's reach and its bend against that forearm within
NECK_ARM_FREE degrees (a fretting wrist is flexed: the free bend is wide).

A REFERENCE (`ref`: the solved shape the hand comes from, its joints and `target_in_wrist`) makes consecutive shapes one
movement: the solve also starts from it (at another position the hand slid along the neck: the same pose relative to the
board), and it becomes the prior: the joints of the fingers that keep their place (free in both, or on the same string and
fret, pressing or hovering) and of the thumb within REF_TIGHT of their sigmas of the reference's, the other fingers' about
the reference's with the full sigmas, the palm centre within REF_MM (along, across, out of the board: it slides along for a
stretch) and the hand's turn within REF_RAD of the reference's relative to the board (SHIFT_MM and SHIFT_RAD after a shift,
where the arm turns the hand). So between two notes only the fingers that play them move, and the free fingers stay where
they hovered.

Each start first lets the pads reach their targets with the neck solid seen softly (REACH_PEN: a reach that ignored it left
the hand inside the neck, and the full problem then pushed it out to the front of the board, the claw above), then solves
the full problem and polishes it (a reference start skips the reach). The best start wins by its misses of GATES, then its
cost; when even that one misses a gate, RETRY times as many design starts are run again."""
import math

import numpy as np
from scipy.optimize import least_squares

from . import geom
from . import grip as G

FOUR, DOF = G.FOUR, G.DOF
# joint priors (mcp pip dip spread) of a pressing finger (arched: PIP bent past a right angle, DIP almost straight) and of
# one hovering over the strings (relaxed, curled toward them), then the thumb (palmar abduction, radial abduction, roll,
# mcp, ip)
PRIOR_PRESS = np.array([42.0, 72.0, 26.0, 0.0])
PRIOR_HOVER = np.array([38.0, 62.0, 24.0, 0.0])
PRIOR_THUMB = np.array([30.0, 8.0, -10.0, 14.0, 8.0])
SIGMA_FINGER = np.array([14.0, 16.0, 12.0, 6.0])
SIGMA_THUMB = np.array([14.0, 12.0, 25.0, 12.0, 12.0])
LO_FINGER = np.array([0.0, 5.0, -5.0, -12.0])
HI_FINGER = np.array([92.0, 105.0, 75.0, 12.0])
LO_THUMB, HI_THUMB = G.PINCH_LO[8:13], G.PINCH_HI[8:13]
PLACE_MM, PLACE_RAD = 70.0, 1.3                  # the frame's translation (mm) and rotation (rad) bounds around the start
START_ROLL = 75.0                                # deg the design start's hand is turned about the neck: under it, pointing out
PRIOR_W = 0.75
PAD_TOL_Y = 0.001                                # a fingertip's contact point may be this far (m) across from its string ...
PAD_TOL_X = 0.0015                               # ... and this far along the neck from its target point
PAD_TOL_CROWD = 0.005                            # ... or this far beside a finger on the same fret of the next string
PAD_SCALE_Y, PAD_SCALE_X = 0.0005, 0.0005        # m past those per residual unit
PAD_TOUCH = 0.0008                               # m: the pad's lowest skin at most this above the fingertip's (the pad touches)
STEEP = math.cos(math.radians(38.0))             # distal phalanx within 38 degrees of straight into the board ...
STEEP_SCALE = 0.06                               # ... then per residual unit of the cosine (a tight placement flattens fingers)
THUMB_FACE = math.cos(math.radians(40.0))
THUMB_X = (-0.012, 0.014)                        # the thumb pad's place along x about the index and middle slots' midpoint
THUMB_Y = 0.010                                  # ... and across (about the board's centre line)
CLEAR_PRESS, CLEAR_HOVER = 0.004, 0.007          # m above the strings: proximal / middle phalanges of pressing fingers, hovering fingers
HOVER_MAX = 0.012                                # m: a hovering finger's pad at most this far above the board ...
HOVER_IN = 0.006                                 # ... and this far inside the board's edges (it hovers over the strings)
HOVER_STEEP = math.cos(math.radians(55.0))      # ... its distal phalanx within 55 degrees of straight into the board
HOME_TOL, HOME_SCALE = 0.006, 0.006              # m: ... and over its own fret (`slots`) within this, then per residual unit
KNUCKLE_Z = 0.012                                # m: the knuckles (MCP heads) at most this far out of the board
KNUCKLE_SCALE = 0.003
NECK_ARM_FREE, NECK_ARM_SIGMA = 35.0, 8.0        # deg: the wrist's free bend against the forearm, then per residual unit
REACH_PEN = 0.003                                # m of skin inside the neck per residual unit while the pads reach
REF_MM = np.array([0.006, 0.004, 0.004])        # a reference at the same position: the palm centre (m along, across, out of
REF_RAD = 0.07                                  # the board: it may slide along for a stretch) and the hand's turn (rad) ...
SHIFT_MM, SHIFT_RAD = 0.012, 0.20                # ... at another position (per residual unit)
REF_TIGHT = 0.4                                  # the share of the joint sigmas a kept finger and the thumb get about the reference
GATES = {"gap_mm": 3.0, "across_mm": 1.5, "along_mm": 1.0, "penetration_mm": 1.0, "finger_clash_mm": 1.0}   # across: past the string's radius
RETRY = 2                                        # a best start that misses a gate: RETRY times as many design starts again
NFEV = (350, 500, 250)


def _hinge(v):
    return np.maximum(v, 0.0)


def _place(t):
    """What a target finger holds: (string, fret) of its place (None, None when the problem does not name it)."""
    return (None if t.get("string") is None else int(t["string"]), None if t.get("fret") is None else int(t["fret"]))


class _Neck:
    """The neck problem on one hand (see the module doc); x = 21 joint angles then the frame's translation (mm) and rotation
    vector (rad) relative to the start."""

    X_SCALE = np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB, np.full(3, 8.0), np.full(3, 0.15)]

    def __init__(self, hand, targets, section, arm=None, slots=None, position=None, ref=None):
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
                               "radius": float(t.get("radius", 0.0006)), "place": _place(t)}
        for f, t in self.targets.items():                                  # two fingers on one fret of neighbouring strings
            s, k = t["place"]                                              # sit side by side: they stagger along it
            crowd = s is not None and any(g != f and u["place"][1] == k and abs(u["place"][0] - s) == 1
                                          for g, u in self.targets.items() if u["place"][0] is not None)
            t["tol_x"] = PAD_TOL_CROWD if crowd else PAD_TOL_X
        sec = section
        self.xa, self.xb = (float(v) for v in sec["x"])
        self.wa, self.wb = (float(v) for v in sec["width"])
        self.da, self.db = (float(v) for v in sec["depth"])
        self.p = float(sec.get("p", 2.6))
        self.pad = {f: hd.surface(F[f][2], hd.ventral, 0.0, 1.0, 0.3) for f in FOUR}
        self.tip = {f: np.where(hd.dom == F[f][2])[0] for f in FOUR}       # all skin of the distal phalanx: what touches
        self.pad_t = hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3)
        if min([len(v) for v in self.pad.values()] + [len(self.pad_t)]) < 3:
            raise ValueError("the hand model has too little palmar skin on a finger or thumb pad for a neck grip")
        self.press = [f for f in FOUR if f in self.targets]
        self.hover = [f for f in FOUR if f not in self.targets]
        self.skin = np.where(hd.dom < hd.nb)[0]                           # skin that follows the hand's own bones
        up = lambda fs, js: np.where(np.isin(hd.dom, [F[f][j] for f in fs for j in js]))[0]     # noqa: E731
        self.low_press = up(self.press, (0, 1))
        self.hover_all = up(self.hover, (0, 1, 2))
        self.clash = G._Clash(hd)
        self.lo = np.r_[np.tile(LO_FINGER, 4), LO_THUMB, np.full(3, -PLACE_MM), np.full(3, -PLACE_RAD)]
        self.hi = np.r_[np.tile(HI_FINGER, 4), HI_THUMB, np.full(3, PLACE_MM), np.full(3, PLACE_RAD)]
        self.design = np.r_[np.concatenate([PRIOR_PRESS if f in self.targets else PRIOR_HOVER for f in FOUR]), PRIOR_THUMB]
        self.prior, self.sigma = self.design.copy(), np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB]
        slots = list(slots) if slots is not None else [None] * 4
        if len(slots) != 4:
            raise ValueError(f"neck slots are the x of the four fingers' frets (index .. little), got {slots!r}")
        self.home = {f: float(s) for f, s in zip(FOUR, slots) if s is not None and f in self.hover}
        if slots[0] is None:                       # the thumb behind the target fingers ...
            xh = float(np.mean([self.targets[f]["point"][0] for f in self.press]))
        else:                                      # ... or behind the position's index and middle: the same for every shape
            xh = 0.5 * (float(slots[0]) + float(slots[1] if slots[1] is not None else slots[0]))
        self.thumb_lo, self.thumb_hi = xh + THUMB_X[0], xh + THUMB_X[1]
        self.knuckles = np.array([F[f][0] for f in FOUR])                  # the fingers' base joints (MCP heads)
        self.arm = None if arm is None else G._Arm(G.arm_spec(arm), NECK_ARM_FREE, NECK_ARM_SIGMA)
        self.ref = None if ref is None else self._reference(ref, position)
        self.R0 = self.p0 = None

    def _reference(self, ref, position):
        """The reference's frame in the hand's rest frame (from its `target_in_wrist`), its joints, its palm centre in N,
        and the priors about it (see the module doc)."""
        hd = self.hand
        try:
            T = np.asarray(ref["target_in_wrist"], float).reshape(4, 4)
            ang = ref["angles"]
            deg = np.array([float(ang[k]) for k in DOF]) if hasattr(ang, "items") else np.asarray(ang, float).reshape(21)
            places = {f: _place({"string": v[0], "fret": v[1]}) for f, v in (ref.get("targets") or {}).items()}
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ValueError(f"neck ref is {{angles, target_in_wrist, targets, position}} of a solved neck grip ({e})") from None
        Bw = hd.rest_rot[0]
        R, p = Bw @ T[:3, :3], hd.h[0] + Bw @ T[:3, 3]
        deg = np.clip(deg, self.lo[:21] + 1e-6, self.hi[:21] - 1e-6)
        same = position is not None and ref.get("position") is not None and int(ref["position"]) == int(position)
        scale = np.ones(21)
        for i, f in enumerate(FOUR):
            mine = self.targets[f]["place"] if f in self.targets else None
            if mine == places.get(f):                          # free in both, or on the same string and fret
                scale[4 * i:4 * i + 4] = REF_TIGHT
        scale[16:] = REF_TIGHT
        self.prior, self.sigma = deg.copy(), self.sigma * scale
        return {"R": R, "p": p, "deg": deg, "palm": R.T @ (hd.palm_c - p),
                "mm": REF_MM if same else SHIFT_MM, "rad": REF_RAD if same else SHIFT_RAD}

    # ---- the start frame
    def start_frame(self, deg):
        """Design start: the hand's anatomy against N, the hand under the neck: x toward the nut = the thumb side, the
        fingers across the neck, then the hand turned about the neck by START_ROLL so that it comes from behind and below
        the board pointing out of it, its palm toward the neck's treble edge, its knuckles at the board's edge; the origin
        places the pressing fingers' pads on their targets on average."""
        hd = self.hand
        y = -hd.chir * hd.along                      # fingers across the neck, the palm toward the back of the neck ...
        z = -hd.ventral
        R0 = np.stack([np.cross(y, z), y, z], 1) @ G._rotm([1.0, 0.0, 0.0], hd.chir * START_ROLL)   # ... then rolled under it
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
        A = dict(Q=Q, D=D, H=H, P=P, R=R, p=p, L=L, deg=deg, sdf=self.sdf(L))
        A["dist_dir"] = {f: R.T @ (D[hd.F[f][2]] @ hd.bone_dir[hd.F[f][2]]) for f in FOUR}
        A["thumb_dir"] = R.T @ (D[hd.F["thumb"][2]] @ hd.pad_t)
        if stage != "reach":
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

    def wrist(self, A):
        """(wrist head, hand direction) in N: what the arm terms read."""
        R, p, hd = A["R"], A["p"], self.hand
        return R.T @ (hd.h[0] - p), R.T @ hd.along

    def _ref_terms(self, A):
        """The palm centre and the hand's turn relative to the board against the reference's, in residual units."""
        ref = self.ref
        q = geom.mat_to_quat((ref["R"].T @ A["R"])[None])[0]
        turn = 2.0 * q[1:] * (1.0 if q[0] >= 0.0 else -1.0)             # ~ the rotation vector for small turns
        return np.r_[(A["R"].T @ (self.hand.palm_c - A["p"]) - ref["palm"]) / ref["mm"], turn / ref["rad"]]

    def residuals(self, x, stage="all"):
        A = self.analyse(x, stage)
        L, sdf = A["L"], A["sdf"]
        tau = 0.0001 if stage == "polish" else 0.0004
        r = []
        for f in self.press:
            t = self.targets[f]
            low, c = self._contact(L, self.tip[f], tau)                       # the fingertip's lowest skin ...
            low_pad = self._contact(L, self.pad[f], tau)[0]
            r += [(low - (t["point"][2] - 0.0004)) / 0.0005,
                  _hinge(abs(c[1] - t["point"][1]) - PAD_TOL_Y) / PAD_SCALE_Y,              # ... on the string ...
                  _hinge(abs(c[0] - t["point"][0]) - t["tol_x"]) / PAD_SCALE_X,            # ... where it is pressed
                  (_hinge(t["x"][0] - c[0]) + _hinge(c[0] - t["x"][1])) / PAD_SCALE_X,      # ... behind its wire
                  _hinge(low_pad - low - PAD_TOUCH) / 0.0005,                               # ... with the pad
                  _hinge(STEEP + A["dist_dir"][f][2]) / STEEP_SCALE]         # the distal phalanx points down onto the board
        gap_t, ct = self._contact(L, self.pad_t, tau, sdf)
        r += [(gap_t - 0.0003) / 0.0006, _hinge(self.thumb_lo - ct[0]) / 0.004, _hinge(ct[0] - self.thumb_hi) / 0.004,
              _hinge(abs(ct[1]) - THUMB_Y) / 0.004, _hinge(THUMB_FACE - A["thumb_dir"][2]) / 0.15]
        r = np.array(r)
        pri = PRIOR_W * (x[:21] - self.prior) / self.sigma
        knuckles = _hinge(((A["H"][self.knuckles] - A["p"]) @ A["R"])[:, 2] - KNUCKLE_Z) / KNUCKLE_SCALE
        out = [r, pri, knuckles]
        if self.ref is not None:
            out.append(self._ref_terms(A))
        if self.arm is not None:
            out.append(self.arm.residuals(*self.wrist(A)))
        sk = sdf[self.skin]
        if stage == "reach":                                                   # the neck seen softly while the pads reach
            out.append(_hinge(0.0002 - sk) / REACH_PEN)
        else:
            z = L[:, 2]
            over = np.abs(L[:, 1]) < 0.5 * self.section_at(L[:, 0])[0] + 0.006       # above the board (strings): clearances
            out += [_hinge(0.0002 - sk) / 0.0002, A["clash"] / 0.0004,
                    _hinge(CLEAR_PRESS - np.where(over[self.low_press], z[self.low_press], 1.0)) / 0.002,
                    _hinge(CLEAR_HOVER - np.where(over[self.hover_all], z[self.hover_all], 1.0)) / 0.003]
            hov = []
            for f in self.hover:                                               # hovering just over the strings
                low, c = self._contact(L, self.pad[f], tau)
                half = 0.5 * float(self.section_at(np.array([c[0]]))[0][0]) - HOVER_IN
                hov += [_hinge(low - HOVER_MAX) / 0.003, _hinge(abs(c[1]) - half) / 0.003,
                        _hinge(HOVER_STEEP + A["dist_dir"][f][2]) / 0.12,              # curled toward the strings
                        _hinge(abs(c[0] - self.home[f]) - HOME_TOL) / HOME_SCALE if f in self.home else 0.0]   # over its fret
            out.append(np.array(hov))
        return np.concatenate(out)

    # ---- solving
    def seed(self, k, nfev=NFEV):
        """Reach, grip, polish from the design start (k = 0) or a perturbed one (k > 0); k = "ref": grip and polish from the
        reference (its joints and its frame). Returns (cost, x, R0, p0, the gates missed)."""
        if k == "ref":
            deg, (self.R0, self.p0) = self.ref["deg"], (self.ref["R"], self.ref["p"])
            stages = zip(("grip", "polish"), nfev[1:], (1e-6, 1e-5))
        else:
            rng = np.random.default_rng(k)
            deg = np.clip(self.design + (rng.normal(0.0, 0.5, 21) * np.r_[np.tile(SIGMA_FINGER, 4), SIGMA_THUMB] if k else 0.0),
                          self.lo[:21] + 1e-6, self.hi[:21] - 1e-6)
            self.R0, self.p0 = self.start_frame(deg)
            if k:
                self.R0 = G._rotvec(rng.normal(0.0, 0.25, 3)) @ self.R0
                self.p0 = self.p0 + rng.normal(0.0, 0.005, 3)
            stages = zip(("reach", "grip", "polish"), nfev, (1e-6, 1e-6, 1e-5))
        z = np.r_[deg, np.zeros(6)]
        for stage, n, ftol in stages:
            res = least_squares(lambda v, s=stage: self.residuals(v, s), z, bounds=(self.lo, self.hi), x_scale=self.X_SCALE,
                                max_nfev=n, xtol=1e-12, ftol=ftol)
            z = res.x
        return float(res.cost), z, self.R0, self.p0, self.misses(self.report(self.analyse(z, "polish")))

    def misses(self, rep):
        """The GATES a solution's report misses (and a wrist bent past NECK_ARM_FREE): a list of short texts."""
        out = []
        for f in self.press:
            c = rep["contacts"][f]
            tol = dict(GATES, across_mm=GATES["across_mm"] + self.targets[f]["radius"] * 1e3)
            out += [f"{f} {k} {c[k]}" for k in ("gap_mm", "across_mm", "along_mm") if abs(c[k]) > tol[k]]
        if rep["contacts"]["thumb"]["gap_mm"] > GATES["gap_mm"]:
            out.append(f"thumb gap_mm {rep['contacts']['thumb']['gap_mm']}")
        out += [f"{k} {rep[k]}" for k in ("penetration_mm", "finger_clash_mm") if rep[k] > GATES[k]]
        if (rep.get("wrist_bend_deg") or 0.0) > NECK_ARM_FREE:
            out.append(f"wrist_bend_deg {rep['wrist_bend_deg']}")
        return out

    def report(self, A):
        """Numbers measured on the final pose: hard minima over the skin."""
        L, sdf = A["L"], A["sdf"]
        con = {}
        for f in self.press:
            t = self.targets[f]
            idx = self.tip[f]
            _l, c = self._contact(L, idx, 0.0001)
            con[f] = {"gap_mm": round(float(L[idx, 2].min() - t["point"][2]) * 1e3, 2),
                      "across_mm": round(float(c[1] - t["point"][1]) * 1e3, 2),
                      "dx_mm": round(float(c[0] - t["point"][0]) * 1e3, 2),
                      "along_mm": round(float(max(t["x"][0] - c[0], c[0] - t["x"][1], 0.0)) * 1e3, 2),
                      "pad_mm": round(float(L[self.pad[f], 2].min() - L[idx, 2].min()) * 1e3, 2),
                      "distal_deg": round(math.degrees(math.acos(float(np.clip(-A["dist_dir"][f][2], -1, 1)))), 1)}
        _l, ct = self._contact(L, self.pad_t, 0.0001, sdf)
        con["thumb"] = {"gap_mm": round(float(sdf[self.pad_t].min()) * 1e3, 2), "x_mm": round(float(ct[0]) * 1e3, 1),
                        "y_mm": round(float(ct[1]) * 1e3, 1),
                        "facing_deg": round(math.degrees(math.acos(float(np.clip(A["thumb_dir"][2], -1, 1)))), 1)}
        z = L[:, 2]
        over = np.abs(L[:, 1]) < 0.5 * self.section_at(L[:, 0])[0] + 0.006
        hov = z[self.hover_all][over[self.hover_all]]
        hovering = {}
        for f in self.hover:
            _l, c = self._contact(L, self.pad[f], 0.0001)
            hovering[f] = {"gap_mm": round(float(L[self.pad[f], 2].min()) * 1e3, 1), "x_mm": round(float(c[0]) * 1e3, 1),
                           "y_mm": round(float(c[1]) * 1e3, 1),
                           "distal_deg": round(math.degrees(math.acos(float(np.clip(-A["dist_dir"][f][2], -1, 1)))), 1)}
        W, hand_dir = self.wrist(A)
        out = {"contacts": con, "penetration_mm": round(max(0.0, -float(sdf[self.skin].min())) * 1e3, 3),
               "finger_clash_mm": round(float(A["clash"].max()) * 1e3, 2),
               "hovering_clear_mm": round(float(hov.min()) * 1e3, 1) if len(hov) else None, "hovering": hovering,
               "knuckles_mm": round(float(((A["H"][self.knuckles] - A["p"]) @ A["R"])[:, 2].max()) * 1e3, 1),
               "wrist_in_N_mm": [round(float(v) * 1e3, 1) for v in W],
               "pressing": self.press, "angles_deg": G._angles(A["deg"])}
        if self.arm is not None:
            out.update(self.arm.report(W, hand_dir))
        if self.ref is not None:                                               # how far the hand moved from the reference
            d = A["R"].T @ (self.hand.palm_c - A["p"]) - self.ref["palm"]
            out["from_ref"] = {"palm_mm": [round(float(v) * 1e3, 1) for v in d],
                               "turn_deg": round(math.degrees(float(np.linalg.norm(self._ref_terms(A)[3:])) * self.ref["rad"]), 1)}
        return out


def _neck_seed(job):
    hand_args, kw, k, nfev = job
    return (k,) + _Neck(G.HandModel(**hand_args), **kw).seed(k, nfev)


def solve_neck(hand, prop, seeds=6, workers=None, **tuning):
    """Fretting hand on a neck (see the module doc).

    prop    {"targets": [...], "section": {...}, "slots", "position"} in the neck frame N (mkmmd.core.fretting.solver_prop
            builds it from a card's `use.grip` neck entry, a chord and a position fret); other keys are ignored.
            The frame (`target_in_wrist` is this frame): origin on the board's centre line under the position fret's wire, x
            along the strings toward the nut, z out of the board, y = z cross x.
    seeds   parallel design starts (the best wins: the fewest GATES missed, then the lowest cost; a reference adds its own
            start; when the best misses a gate, RETRY * seeds more are run); workers: processes (default: all cores, 1 = here)
    tuning  nfev (evaluations of the reach / grip / polish stages), arm (the arm that carries the hand, in N: see
            mkmmd.solvers.grip.arm_spec; its bend is free up to NECK_ARM_FREE degrees), ref (the solved shape the hand comes
            from: {angles: {joint: deg} (its report's angles_deg), target_in_wrist, targets: {finger: [string, fret]},
            position}; see the module doc)
    report  contacts {finger: gap_mm (the fingertip's lowest skin against the pressed string's height; 0 = touching, + above),
            across_mm (the centroid of that lowest skin from the string), dx_mm (from the target point along the neck, + toward
            the nut), along_mm (outside its range behind the wire), pad_mm (the pad's lowest skin above the fingertip's: 0 = the
            pad touches), distal_deg (the distal phalanx from straight into the board)}, thumb {gap_mm,
            x_mm, y_mm, facing_deg}, penetration_mm (deepest skin in the neck), finger_clash_mm, hovering_clear_mm (the free
            fingers over the strings), hovering {finger: gap_mm (its pad's lowest skin above the board), x_mm, y_mm (its place)},
            knuckles_mm (the highest knuckle out of the board), wrist_in_N_mm, angles_deg, start (the winning start: "ref" or
            a design seed), starts (how many ran), misses (the GATES the winner misses); with `arm` wrist_bend_deg, reach, elbow
            (in N) and elbow_from; with `ref` from_ref {palm_mm (the palm centre's move in N), turn_deg} (how far the hand moved)."""
    unknown = set(tuning) - {"nfev", "arm", "ref"}
    if unknown:
        raise ValueError(f"unknown neck tuning {sorted(unknown)} (have nfev, arm, ref)")
    if int(seeds) < 1:
        raise ValueError("neck needs at least one seed")
    if not hasattr(prop, "get") or not prop.get("targets") or not prop.get("section"):
        raise ValueError("a neck prop is a dict with `targets` (pressing fingers) and `section` (the neck's solid)")
    nfev = tuple(int(n) for n in tuning.get("nfev", NFEV))
    kw = dict(targets=prop["targets"], section=prop["section"], slots=prop.get("slots"), position=prop.get("position"))
    if tuning.get("arm") is not None:
        kw["arm"] = G.arm_spec(tuning["arm"])
    if tuning.get("ref") is not None:
        kw["ref"] = tuning["ref"]
    prob = _Neck(hand, **kw)                                         # checks the problem here before the starts run
    args = G._hand_args(hand)
    starts = (["ref"] if prob.ref is not None else []) + list(range(int(seeds)))
    runs = G._run_seeds(_neck_seed, [(args, kw, s, nfev) for s in starts], workers)
    rank = lambda run: (len(run[5]), run[1])                         # noqa: E731
    if min(runs, key=rank)[5]:
        more = list(range(int(seeds), int(seeds) * (1 + RETRY)))
        runs += G._run_seeds(_neck_seed, [(args, kw, s, nfev) for s in more], workers)
        starts += more
    k, cost, z, prob.R0, prob.p0, missed = min(runs, key=rank)
    A = prob.analyse(z, "polish")
    rep = dict(prob.report(A), start=k, starts=len(starts), misses=missed)
    return G._result("neck", hand, A["Q"], A["R"], A["p"], rep,
                     solver={"version": G.VERSION, "cost": cost, "seeds": int(seeds), "x": z})
