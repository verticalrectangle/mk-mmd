"""Grip style `pluck`: a bass player's plucking hand (fingerstyle), solved on the model's own skin (numpy + scipy, no bpy).
Part of mkmmd.solvers.grip (`solve("pluck", hand, prop, ...)`); see docs/design.md: Playing a worn guitar, Bass lines and
picking.

One STATION of the hand: the string it plays and where its thumb rests (`anchor`): on the pickup cover's top when the lowest
string rings, on the lowest string when a higher one does. The hand hangs over the strings with its palm toward the face, the
index and middle fingers curled down onto the played string with their pads turned toward the lower strings (the way a rest
stroke pulls the string), the thumb pad on its anchor, its distal phalanx pointing toward the nut, the ring and little fingers
curled clear of the strings. Everything is in the STATION FRAME (mkmmd.core.strum.pluck_prop builds the prop): origin on the
played string's axis in the middle of the plucking zone, x across the strings from the lowest to the highest, y along them
toward the nut, z out of the face.

prop (metres, in the station frame)
    strings   [{a, b, radius}]: every string's axis (two points) and radius, the lowest first
    string    the played string's index in `strings`
    zone      [y0, y1]: where along the strings the pads touch them (and the thumb rests on the lowest string)
    anchor    {type: "plane", center, normal, size: [x, y]}: a patch the thumb pad rests on, or {type: "string", string}
    face      {point, normal}: the surface under the strings; blocks [{center, axes (3x3 columns), size}]: parts standing
              on it (the pickup covers). No skin goes into either.

A rest stroke is one swing of a finger from HOVER (its fingertip HOVER_U toward the higher strings from the played string and
HOVER_GAP over it, ready) through the CROSS (the fingertip on top of the string, straight over its axis: where the string is
let go) to REST (the pad against the next lower string's side toward the played one, or toward the thumb when the lowest string
rings). The fingertip is the skin of the distal phalanx, and its place across the string is that of its skin nearest the
string: what the `strum` check measures (mkmmd.checks.props), so the report's `travel` times the stroke exactly.

Unknowns: the frame (translation mm, rotation vector rad, from a design start: the fingers along +x, the palm toward -z), the
thumb's five joints, the ring and little fingers' curls, and the index and middle fingers' joints at the crossing. Residuals:
the thumb pad on its anchor, both crossings in the zone with the pads turned toward the lower strings (PAD_FACE), no skin
inside a string, the face or a block, no plucking finger deeper than DIP under the strings, the ring and little fingers
RING_CLEAR off the strings and the face, the plucking fingers' first two phalanges CLEAR off the strings, no finger in
another, joint priors, and the arm (mkmmd.solvers.grip._Arm: the wrist within reach, its bend against the forearm free to
ARM_FREE degrees, the forearm clear of the `avoid` points). Then, with the hand held, each finger's hover and rest are solved
on its own three joints. The result's `bones` are the station's hand with both fingers hovering, `poses` the plucking fingers'
joints per pose ({"index.cross": {bone: quaternion}, ...})."""
import math

import numpy as np
from scipy.optimize import least_squares

from . import grip as G

FOUR, DOF = G.FOUR, G.DOF
PLUCKERS = ("index", "middle")
POSES = ("hover", "cross", "rest")
# joint priors (mcp pip dip spread) of a plucking finger at the crossing (curled down onto the string), the thumb (palmar
# abduction, radial abduction, roll, mcp, ip) and the ring / little curls (1 = CURL)
PRIOR_PLUCK = np.array([38.0, 45.0, 18.0, 0.0])
SIGMA_PLUCK = np.array([18.0, 20.0, 14.0, 6.0])
LO_PLUCK = np.array([-10.0, 0.0, -5.0, -12.0])
HI_PLUCK = np.array([90.0, 100.0, 70.0, 12.0])
PRIOR_THUMB = np.array([25.0, 35.0, -20.0, 10.0, 5.0])
SIGMA_THUMB = np.array([15.0, 15.0, 30.0, 12.0, 12.0])
LO_THUMB, HI_THUMB = G.PINCH_LO[8:13], G.PINCH_HI[8:13]
CURL = {"ring": (55.0, 72.0, 38.0), "little": (65.0, 82.0, 42.0)}     # mcp pip dip at curl 1
PRIOR_CURL, SIGMA_CURL, LO_CURL, HI_CURL = 0.9, 0.3, 0.4, 1.5
PRIOR_W = 0.6
PLACE_MM, PLACE_RAD = 80.0, 1.4  # the frame's translation (mm) and rotation (rad) bounds around the start
PAD_FACE = 0.3                   # a plucking pad faces the lower strings: its normal at least this much against x
HOVER_U = (0.006, 0.010)         # m: a hovering fingertip's place across from the played string (toward the higher ones) ...
HOVER_GAP = (0.003, 0.006)       # m: ... and its height over the string's surface
CLEAR = 0.0015                   # m: the plucking fingers' proximal and middle phalanges off the strings
RING_CLEAR = 0.004               # m: the ring and little fingers' skin off the strings and the face
TOUCH_GAP = 0.0002               # m: a fingertip touching a string keeps this far off its surface
SIDE = 0.6                       # a pad against a string's side: its contact at least this share of the way across
THUMB_ALONG = 0.6                # the thumb's distal phalanx points toward the nut: its direction at least this much along y
THUMB_REST = 0.0008              # m: a pad resting against the thumb (the lowest string's rest stroke) stays this far off it
PASS = 0.004                     # m: a rest stroke's fingertip ends at least this far past the played string's surface
DIP = 0.004                      # m: no skin of a plucking finger goes deeper than this below the played string's axis
SHARP = 0.0002                   # m: the fingertip's place is a softmin-weighted mean of its skin within about this of the nearest
ARM_FREE, ARM_SIGMA = 25.0, 8.0  # deg: the wrist's free bend against the forearm, then per residual unit
NFEV = (60, 150, 80)             # evaluations of the station's reach / grip / polish stages
SUB_NFEV = 200                   # ... and of each finger pose
SPACING = 0.002                  # m: the skin is solved on one vertex per cell this big
CURVE = 21                       # blend weights at which a stroke segment's fingertip place is measured (`curves`)


def _hinge(v):
    return np.maximum(v, 0.0)


def _slerp(q0, q1, w):
    """The shortest-path spherical blend of unit quaternions [w, x, y, z] (what mathutils' slerp does)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + w * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1.0 - w) * th) * q0 + math.sin(w * th) * q1) / math.sin(th)


def _soft(g, tau, spread=0.0012):
    """(softmin of g, the weights of the points that make it: exp(-(g - min) / spread), normalised)."""
    m = float(g.min())
    w = np.exp(-(g - m) / spread)
    return m - tau * math.log(float(np.sum(np.exp(-(g - m) / tau)))), w / w.sum()


class _Strings:
    """The strings in the station frame: axis points a (k, 3), unit directions e, radii r; x and z made perpendicular to each
    string give an offset's place across the string (toward the higher strings) and out of the face."""

    def __init__(self, strings):
        self.a = np.array([s["a"] for s in strings], float).reshape(-1, 3)
        b = np.array([s["b"] for s in strings], float).reshape(-1, 3)
        self.e = (b - self.a) / np.linalg.norm(b - self.a, axis=1, keepdims=True)
        self.r = np.array([float(s.get("radius", 0.0006)) for s in strings])
        x = np.array([1.0, 0.0, 0.0])
        self.ux = x - (self.e @ x)[:, None] * self.e
        self.ux /= np.linalg.norm(self.ux, axis=1, keepdims=True)
        self.uz = np.cross(self.ux, self.e)
        self.uz *= np.sign(self.uz[:, 2:3])

    def offset(self, P, k):
        """(n, 3) offsets of points P from string k's axis, perpendicular to it."""
        v = np.atleast_2d(P) - self.a[k]
        return v - (v @ self.e[k])[:, None] * self.e[k]

    def gaps(self, P):
        """(n, k) every point's gap to every string's surface (- inside)."""
        v = P[:, None, :] - self.a[None]
        v = v - np.einsum("nki,ki->nk", v, self.e)[..., None] * self.e[None]
        return np.linalg.norm(v, axis=2) - self.r[None]

    def u(self, P, k):
        """Across offsets of points P from string k's axis (+ toward the higher strings)."""
        return self.offset(P, k) @ self.ux[k]


class _Body:
    """The face (a plane) and the blocks on it (oriented boxes): the signed distance of points, + in front of them."""

    def __init__(self, face, blocks):
        self.p, self.n = np.asarray(face["point"], float), np.asarray(face["normal"], float)
        self.n = self.n / np.linalg.norm(self.n)
        self.c = np.array([b["center"] for b in blocks], float).reshape(-1, 3)
        self.ax = np.array([np.asarray(b.get("axes", np.eye(3)), float) for b in blocks]).reshape(-1, 3, 3)
        self.h = np.array([np.asarray(b["size"], float) / 2.0 for b in blocks]).reshape(-1, 3)

    def sdf(self, P):
        d = (P - self.p) @ self.n
        for c, ax, h in zip(self.c, self.ax, self.h):
            q = np.abs((P - c) @ ax) - h
            d = np.minimum(d, np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0))
        return d


class _Pluck:
    """The station problem on one hand (see the module doc); x = the thumb's 5 joints, the ring and little curls, the index and
    middle fingers' crossing joints (mcp pip dip spread each), then the frame's translation (mm) and rotation (rad)."""

    X_SCALE = np.r_[SIGMA_THUMB, SIGMA_CURL, SIGMA_CURL, SIGMA_PLUCK, SIGMA_PLUCK, np.full(3, 8.0), np.full(3, 0.15)]

    def __init__(self, hand, strings, string, zone, anchor, face, blocks=(), arm=None):
        self.hand = hd = hand
        F = hd.F
        self.S = _Strings(strings)
        self.k = int(string)
        if not 0 <= self.k < len(self.S.r):
            raise ValueError(f"pluck: string {string} is not one of the {len(self.S.r)} strings")
        self.zone = (float(min(zone)), float(max(zone)))
        self.anchor = dict(anchor)
        if self.anchor.get("type") == "plane":
            self.a_c = np.asarray(self.anchor["center"], float)
            self.a_n = np.asarray(self.anchor["normal"], float) / np.linalg.norm(self.anchor["normal"])
            self.a_half = np.asarray(self.anchor["size"], float) / 2.0
        elif self.anchor.get("type") == "string":
            self.a_k = int(self.anchor["string"])
        else:
            raise ValueError("pluck anchor: {type: plane, center, normal, size} or {type: string, string}")
        self.body = _Body(face, blocks)
        dom = lambda bones: np.where(np.isin(hd.dom, bones))[0]                         # noqa: E731
        self.tip = {f: dom([F[f][2]]) for f in PLUCKERS}                                 # the fingertip: the distal phalanx
        self.pad = {f: hd.surface(F[f][2], hd.ventral, 0.25, 1.0, 0.3) for f in PLUCKERS}
        self.pad_t = hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3)
        if min([len(v) for v in self.pad.values()] + [len(v) for v in self.tip.values()] + [len(self.pad_t)]) < 3:
            raise ValueError("the hand model has too little skin on the index, middle or thumb pad for a pluck")
        self.skin = np.where(hd.dom < hd.nb)[0]                                          # skin that follows the hand's bones
        self.finger = {f: dom(F[f][:3]) for f in FOUR}
        self.low = {f: dom(F[f][:2]) for f in PLUCKERS}                                  # proximal and middle phalanges
        self.curled = np.r_[self.finger["ring"], self.finger["little"]]
        self.clash = G._Clash(hd)
        self.arm = None if arm is None else G._Arm(G.arm_spec(arm), ARM_FREE, ARM_SIGMA)
        self.lo = np.r_[LO_THUMB, LO_CURL, LO_CURL, LO_PLUCK, LO_PLUCK, np.full(3, -PLACE_MM), np.full(3, -PLACE_RAD)]
        self.hi = np.r_[HI_THUMB, HI_CURL, HI_CURL, HI_PLUCK, HI_PLUCK, np.full(3, PLACE_MM), np.full(3, PLACE_RAD)]
        self.prior = np.r_[PRIOR_THUMB, PRIOR_CURL, PRIOR_CURL, PRIOR_PLUCK, PRIOR_PLUCK]
        self.R0 = self.p0 = None

    # ---- parameters
    def degrees(self, x):
        """The 21 joint angles (DOF order) of x: index and middle from x[7:15], ring and little at their curls x[5:7]
        (fanned out 3 and 6 degrees), the thumb from x[0:5]."""
        deg = np.zeros(21)
        deg[0:4], deg[4:8] = x[7:11], x[11:15]
        for k, f in enumerate(("ring", "little")):
            c = CURL[f]
            deg[8 + 4 * k:12 + 4 * k] = [c[0] * x[5 + k], c[1] * x[5 + k], c[2] * x[5 + k], -3.0 * (k + 1)]
        deg[16:21] = x[0:5]
        return deg

    def start_frame(self, deg):
        """Design start: the fingers across the strings (+x), the palm toward the face (-z), the plucking pads' mean just
        over the played string in the middle of the zone."""
        hd = self.hand
        x, z = hd.along, -hd.ventral
        R0 = np.stack([x, np.cross(z, x), z], 1)
        P = hd.pose(deg)[3]
        m = np.mean([P[self.pad[f]].mean(0) for f in PLUCKERS], axis=0)
        return R0, m - R0 @ np.array([0.0, 0.5 * sum(self.zone), self.S.r[self.k] + 0.006])

    def frame(self, x):
        return G._rotvec(x[18:21]) @ self.R0, self.p0 + x[15:18] * 1e-3

    def wrist(self, A):
        """(wrist head, hand direction) in the station frame: what the arm terms read."""
        R, p, hd = A["R"], A["p"], self.hand
        return R.T @ (hd.h[0] - p), R.T @ hd.along

    # ---- pieces
    def pad_facing(self, D, R, f):
        """Finger f's pad normal (its distal phalanx's palmar side) in the station frame."""
        return R.T @ (D[self.hand.F[f][2]] @ self.hand.ventral)

    def touch(self, L, g, idx, k, tau):
        """(softmin gap of the skin `idx` to string k, the skin's place nearest the string (SHARP-weighted))."""
        low, w = _soft(g[idx, k], tau, SHARP)
        return low, w @ L[idx]

    def anchor_terms(self, L, g, D, R, tau):
        """The thumb pad on its anchor: [gap, outside the patch or zone, on top of the string, facing, pointing toward the
        nut] in residual units."""
        hd = self.hand
        tb = hd.F["thumb"][2]
        pad_dir = R.T @ (D[tb] @ hd.pad_t)
        along = _hinge(THUMB_ALONG - float((R.T @ (D[tb] @ hd.bone_dir[tb]))[1])) / 0.1
        if self.anchor["type"] == "plane":
            low, w = _soft((L[self.pad_t] - self.a_c) @ self.a_n, tau)
            q = w @ L[self.pad_t] - self.a_c
            return [(low - 0.0003) / 0.0006, _hinge(abs(q[0]) - self.a_half[0]) / 0.002,
                    _hinge(abs(q[1]) - self.a_half[1]) / 0.002, _hinge(math.cos(math.radians(45.0)) + pad_dir @ self.a_n) / 0.15,
                    along]
        k = self.a_k
        low, w = _soft(g[self.pad_t, k], tau)
        q = w @ L[self.pad_t]
        off = self.S.offset(q, k)[0]
        return [(low - 0.0003) / 0.0006, _hinge(abs(off @ self.S.ux[k]) - 0.5 * (off @ self.S.uz[k])) / 0.002,
                (_hinge(self.zone[0] - q[1]) + _hinge(q[1] - self.zone[1])) / 0.003,
                _hinge(math.cos(math.radians(45.0)) + pad_dir @ self.S.uz[k]) / 0.15, along]

    def cross_terms(self, L, g, D, R, f, tau):
        """Finger f's fingertip on top of the played string (its skin nearest the string straight over the axis, touching
        it), in the zone, its pad turned toward the lower strings."""
        k = self.k
        low, q = self.touch(L, g, self.tip[f], k, tau)
        nx = float(self.pad_facing(D, R, f) @ self.S.ux[k])
        return [(low - TOUCH_GAP) / 0.0004, float(self.S.u(q, k)[0]) / 0.0005,
                (_hinge(self.zone[0] - q[1]) + _hinge(q[1] - self.zone[1])) / 0.003, _hinge(nx + PAD_FACE) / 0.1]

    # ---- the station
    def analyse(self, x, stage="all"):
        hd = self.hand
        deg = self.degrees(x)
        Q, D, H, P = hd.pose(deg)
        R, p = self.frame(x)
        L = (P - p) @ R
        A = dict(Q=Q, D=D, H=H, P=P, R=R, p=p, L=L, deg=deg, gaps=self.S.gaps(L))
        if stage != "reach":
            A["sdf"] = self.body.sdf(L)
            A["clash"] = self.clash.depth(P, H)
        return A

    def residuals(self, x, stage="all"):
        A = self.analyse(x, stage)
        L, g, D, R = A["L"], A["gaps"], A["D"], A["R"]
        tau = 0.0001 if stage == "polish" else 0.0004
        r = list(self.anchor_terms(L, g, D, R, tau))
        for f in PLUCKERS:
            r += self.cross_terms(L, g, D, R, f, tau)
        out = [np.array(r), PRIOR_W * (x[:15] - self.prior) / self.X_SCALE[:15]]
        if self.arm is not None:
            out.append(self.arm.residuals(*self.wrist(A)))
        if stage != "reach":
            sdf = A["sdf"]
            out += [_hinge(0.0002 - g[self.skin]).ravel() / 0.0003, _hinge(0.0003 - sdf[self.skin]) / 0.0003,
                    A["clash"] / 0.0004, _hinge(RING_CLEAR - g[self.curled]).ravel() / 0.002,
                    _hinge(RING_CLEAR - sdf[self.curled]) / 0.002]
            out += [_hinge(CLEAR - g[self.low[f]]).ravel() / 0.001 for f in PLUCKERS]
            out += [_hinge(-DIP - L[self.finger[f], 2]) / 0.001 for f in PLUCKERS]
        return np.concatenate(out)

    def seed(self, k, nfev=NFEV):
        """Reach, grip, polish from the prior pose (k = 0) or a perturbed pose and frame (k > 0)."""
        rng = np.random.default_rng(k)
        x0 = np.r_[self.prior, np.zeros(6)]
        if k:
            x0[:15] = x0[:15] + rng.normal(0.0, 0.5, 15) * self.X_SCALE[:15]
        x0 = np.clip(x0, self.lo + 1e-6, self.hi - 1e-6)
        self.R0, self.p0 = self.start_frame(self.degrees(x0))
        if k:
            self.R0 = G._rotvec(rng.normal(0.0, 0.25, 3)) @ self.R0
            self.p0 = self.p0 + rng.normal(0.0, 0.006, 3)
        z = x0
        for stage, n, ftol in zip(("reach", "grip", "polish"), nfev, (1e-6, 1e-6, 1e-5)):
            res = least_squares(lambda v, s=stage: self.residuals(v, s), z, bounds=(self.lo, self.hi), x_scale=self.X_SCALE,
                                max_nfev=n, xtol=1e-12, ftol=ftol)
            z = res.x
        return float(res.cost), z, self.R0, self.p0

    # ---- one finger's other poses, the rest of the hand held
    def finger_state(self, deg, f, v, frame):
        d = deg.copy()
        i = 4 * FOUR.index(f)
        d[i:i + 3] = v
        Q, D, H, P = self.hand.pose(d)
        R, p = frame
        L = (P - p) @ R
        return dict(Q=Q, D=D, H=H, P=P, L=L, R=R, p=p, deg=d, gaps=self.S.gaps(L))

    def thumb_gap(self, A, idx, tau):
        """The softmin distance of the skin `idx` to the thumb's last two phalanges (capsules of their skin radii)."""
        hd = self.hand
        Pw = A["P"][idx]
        d = []
        for tb in hd.F["thumb"][1:3]:
            a = A["H"][tb]
            ab = A["D"][tb] @ (hd.tails[tb] - hd.h[tb])
            t = np.clip((Pw - a) @ ab / (ab @ ab), 0.0, 1.0)
            d.append(np.linalg.norm(Pw - (a + t[:, None] * ab), axis=1) - float(hd.radius[tb]))
        return _soft(np.minimum(*d), tau)[0]

    def finger_terms(self, A, f, pose, tau):
        """The residuals of finger f's `hover` or `rest`."""
        k, g, L = self.k, A["gaps"], A["L"]
        nx = float(self.pad_facing(A["D"], A["R"], f) @ self.S.ux[k])
        mine = self.finger[f]
        common = [_hinge(nx + PAD_FACE) / 0.1, *(_hinge(-DIP - L[mine, 2]) / 0.001)]
        low, q = self.touch(L, g, self.tip[f], k, tau)
        u = float(self.S.u(q, k)[0])
        if pose == "hover":
            return [_hinge(HOVER_GAP[0] - low) / 0.0005, _hinge(low - HOVER_GAP[1]) / 0.0005, _hinge(HOVER_U[0] - u) / 0.001,
                    _hinge(u - HOVER_U[1]) / 0.001, *(_hinge(HOVER_GAP[0] - g[mine]).ravel() / 0.0005), *common]
        r = [_hinge(u + self.S.r[k] + PASS) / 0.0005, *(_hinge(-g[mine]).ravel() / 0.0003)]
        if k > 0:                                       # rest: against the next lower string's side toward the played one
            low, q = self.touch(L, g, self.pad[f], k - 1, tau)
            off = self.S.offset(q, k - 1)[0]
            return r + [(low - TOUCH_GAP) / 0.0004, _hinge(SIDE * np.linalg.norm(off) - off @ self.S.ux[k - 1]) / 0.001,
                        *common]
        h = float(self.S.offset(q, k)[0] @ self.S.uz[k])  # the lowest string: toward the thumb, in the strings' plane
        return r + [0.25 * _hinge(self.thumb_gap(A, self.pad[f], tau) - THUMB_REST) / 0.002, _hinge(abs(h) - 0.004) / 0.001,
                    *common]

    def finger_residuals(self, v, deg, f, pose, frame, tau=0.0002):
        A = self.finger_state(deg, f, v, frame)
        mine = self.finger[f]
        return np.concatenate([np.asarray(self.finger_terms(A, f, pose, tau), float),
                               _hinge(0.0003 - self.body.sdf(A["L"][mine])) / 0.0003, self.clash.depth(A["P"], A["H"]) / 0.0004,
                               0.3 * (v - PRIOR_PLUCK[:3]) / SIGMA_PLUCK[:3]])

    def finger_pose(self, deg, f, pose, frame, start, nfev=SUB_NFEV):
        res = least_squares(lambda v: self.finger_residuals(v, deg, f, pose, frame), np.clip(start, LO_PLUCK[:3] + 1e-6,
                            HI_PLUCK[:3] - 1e-6), bounds=(LO_PLUCK[:3], HI_PLUCK[:3]), x_scale=SIGMA_PLUCK[:3], max_nfev=nfev,
                            xtol=1e-12, ftol=1e-8)
        return res.x

    def poses(self, z, nfev=SUB_NFEV):
        """({finger: {pose: (3,) joints}}, the base joints (both fingers hovering)) for the station solve z: the crossing from
        it, each finger's hover beside the other's crossing, then each rest beside the other's hover."""
        frame = self.frame(z)
        deg = self.degrees(z)
        out = {f: {"cross": deg[4 * FOUR.index(f):4 * FOUR.index(f) + 3].copy()} for f in PLUCKERS}
        base = deg.copy()
        for f in PLUCKERS:
            out[f]["hover"] = self.finger_pose(base, f, "hover", frame, out[f]["cross"] - np.array([10.0, 10.0, 0.0]), nfev)
            i = 4 * FOUR.index(f)
            base[i:i + 3] = out[f]["hover"]
        for f in PLUCKERS:
            out[f]["rest"] = self.finger_pose(base, f, "rest", frame, out[f]["cross"] + np.array([12.0, 8.0, 0.0]), nfev)
        return out, base

    # ---- report
    def report(self, z, joints, base):
        """Numbers measured on the solved station, every pose: hard minima over the skin."""
        frame = self.frame(z)
        A = self.analyse(z, "polish")
        k, sk = self.k, self.skin
        g, L = A["gaps"], A["L"]
        con = {"thumb": ({"gap_mm": round(float(((L[self.pad_t] - self.a_c) @ self.a_n).min()) * 1e3, 2), "on": "plane"}
                         if self.anchor["type"] == "plane" else
                         {"gap_mm": round(float(g[self.pad_t, self.a_k].min()) * 1e3, 2), "on": "string"})}
        rows, travel, pen, clash = {}, {}, [], []
        for f in PLUCKERS:
            rows[f], travel[f] = {}, {}
            for pose in POSES:
                st = self.finger_state(base, f, joints[f][pose], frame)
                gp, Lp = st["gaps"], st["L"]
                pen.append(max(0.0, -float(gp[sk].min()), -float(self.body.sdf(Lp[sk]).min())))
                clash.append(float(self.clash.depth(st["P"], st["H"]).max()))
                j = self.tip[f][int(np.argmin(gp[self.tip[f], k]))]                  # the fingertip's skin nearest the string
                u = float(self.S.u(Lp[j], k)[0])
                travel[f][pose] = round(u, 5)
                rows[f][pose] = {"u_mm": round(u * 1e3, 1), "gap_mm": round(float(gp[j, k]) * 1e3, 2),
                                 "facing": round(float(self.pad_facing(st["D"], st["R"], f) @ self.S.ux[k]), 2)}
                if pose == "rest":
                    rows[f][pose]["rest_gap_mm"] = round((float(gp[self.pad[f], k - 1].min()) if k > 0 else
                                                          float(self.thumb_gap(st, self.pad[f], 0.0001))) * 1e3, 2)
            con[f] = {"gap_mm": rows[f]["cross"]["gap_mm"]}
        W, hand_dir = self.wrist(A)
        out = {"contacts": con, "penetration_mm": round(max(pen) * 1e3, 3), "finger_clash_mm": round(max(clash) * 1e3, 2),
               "poses": rows, "travel": travel,
               "curled_clear_mm": round(float(min(g[self.curled].min(), A["sdf"][self.curled].min())) * 1e3, 1),
               "wrist_mm": [round(float(v) * 1e3, 1) for v in W], "angles_deg": G._angles(A["deg"])}
        if self.arm is not None:
            out.update(self.arm.report(W, hand_dir))
        return out

    def curves(self, frame, base_local, poses):
        """{finger: {"hover": [u ...], "cross": [u ...]}}: the fingertip's place across the played string (its skin nearest
        the string, m) at CURVE blend weights 0..1 from each pose to the next (hover -> cross, cross -> rest), the finger's
        bones turned by the slerp of their local rotations, as perform blends them."""
        R, p = frame
        hd, k = self.hand, self.k
        out = {}
        for f in PLUCKERS:
            out[f] = {}
            for a, b in (("hover", "cross"), ("cross", "rest")):
                us = []
                for w in np.linspace(0.0, 1.0, CURVE):
                    bones = dict(base_local)
                    for bn, q in poses[f"{f}.{a}"].items():
                        bones[bn] = _slerp(q, poses[f"{f}.{b}"][bn], w)
                    D, H = hd.fk(hd.rotations_from_local(bones))
                    L = (hd.skin(D, H)[self.tip[f]] - p) @ R
                    j = int(np.argmin(self.S.gaps(L)[:, k]))
                    us.append(round(float(self.S.u(L[j], k)[0]), 6))
                out[f][a] = us
        return out


def _pluck_seed(job):
    hand_args, kw, k, nfev = job
    return _Pluck(G.HandModel(**hand_args), **kw).seed(k, nfev)


def solve_pluck(hand, prop, seeds=6, workers=None, **tuning):
    """A plucking hand's station (see the module doc).

    prop    the station problem (mkmmd.core.strum.pluck_prop builds it from a card's strum entry and a string); other keys are
            ignored. The frame (`target_in_wrist` is this frame): origin on the played string's axis in the middle of the
            zone, x across the strings toward the higher ones, y along them toward the nut, z out of the face.
    seeds   parallel starts (the lowest cost wins); workers: processes (default: all cores, 1 = here)
    tuning  nfev (evaluations of the reach / grip / polish stages), arm (the arm that carries the hand, in the frame: see
            mkmmd.solvers.grip.arm_spec; its bend is free up to ARM_FREE degrees)
    report  contacts {thumb: gap_mm (its pad over the anchor), index, middle: gap_mm (the fingertip over the string at the
            crossing)}, penetration_mm (the deepest skin in a string, the face or a block over every pose), finger_clash_mm,
            poses {finger: {hover, cross, rest: {u_mm (the fingertip's skin nearest the played string, across it: + toward
            the higher strings), gap_mm, facing (the pad's normal along x), rest_gap_mm}}}, travel {finger: {pose: u (m)}}
            (what times a stroke: mkmmd.core.strum.pluck_keys), curled_clear_mm (ring and little), wrist_mm, angles_deg;
            with `arm` wrist_bend_deg, reach, elbow, elbow_from, forearm_clear_mm.
    Returns the result of every style plus `poses` {"index.hover": {bone: [w, x, y, z]}, ...} (the finger's three joints)."""
    unknown = set(tuning) - {"nfev", "arm"}
    if unknown:
        raise ValueError(f"unknown pluck tuning {sorted(unknown)} (have nfev, arm)")
    if int(seeds) < 1:
        raise ValueError("pluck needs at least one seed")
    need = ("strings", "string", "zone", "anchor", "face")
    if not hasattr(prop, "get") or any(prop.get(k) is None for k in need):
        raise ValueError(f"a pluck prop is a dict with {', '.join(need)} (and blocks): mkmmd.core.strum.pluck_prop")
    nfev = tuple(int(n) for n in tuning.get("nfev", NFEV))
    kw = dict(strings=prop["strings"], string=prop["string"], zone=prop["zone"], anchor=prop["anchor"], face=prop["face"],
              blocks=prop.get("blocks") or ())
    if tuning.get("arm") is not None:
        kw["arm"] = G.arm_spec(tuning["arm"])
    full, hand = hand, hand.thinned(SPACING)
    prob = _Pluck(hand, **kw)
    runs = G._run_seeds(_pluck_seed, [(G._hand_args(hand), kw, s, nfev) for s in range(int(seeds))], workers)
    cost, z, prob.R0, prob.p0 = min(runs, key=lambda c: c[0])
    joints, base = prob.poses(z)
    R, p = prob.frame(z)
    meas = _Pluck(full, **kw)                    # measured on the whole skin, as the `strum` check measures the fingertip
    meas.R0, meas.p0 = prob.R0, prob.p0
    res = G._result("pluck", hand, hand.rotations(base), R, p, meas.report(z, joints, base),
                    solver={"version": G.VERSION, "cost": cost, "seeds": int(seeds), "x": z})
    poses = {}
    for f in PLUCKERS:
        i = 4 * FOUR.index(f)
        bones = [hand.names[b] for b in hand.F[f][:3] if hand.names[b]]
        for pose in POSES:
            d = base.copy()
            d[i:i + 3] = joints[f][pose]
            local = hand.local_quats(hand.rotations(d))
            poses[f"{f}.{pose}"] = {b: local[b] for b in bones}
    res["poses"] = poses
    res["report"]["curves"] = meas.curves((R, p), hand.local_quats(hand.rotations(base)), poses)
    return res
