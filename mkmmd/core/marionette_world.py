"""Marionettes in the world, bpy-free (docs/design.md: Marionette): the keys of `[perform.<cast>] marionette` that move
her root (mkmmd.core.marionette.WORLD_KEYS): she lies on her back and floats on the water, is jerked up onto her feet,
hops landing on given times, is flown flat and swims, loses her threads one at a time, falls when the control is let
go, lands on a point, is yanked away by her last thread and is knotted on again. mkmmd.core.marionette has the standing
puppet, the cross's own channels and the threads; mkmmd/blender/build/marionette.py keys what this plans.

Frames. Her LEVEL frame is her root at rest (x her left, y behind her, z up, metres from where she stands, z the
world's vertical); her BODY frame is the root as it moves (the armature's axes). `lie` turns the body about her left
axis through her hips (+ head forward: 90 lies on her front, face down; -90 on her back, face up), and a body hung in
the air swings under the control as a pendulum, tilting with its swing. The plan gives the root as a rotation R and a
translation per frame (a body point q goes to pos + R q in the level frame), her hands as wrist points, directions,
palms and elbow poles in the level frame, and the control's bars as level-frame poses.

The model. As the standing puppet's, every part gets where its threads (or the ground, or gravity) want it late,
through damped springs: her attitude follows the control's (`LIE`), her weight its lift (`BODY`), a hung body swings
under it with the period of its length, a hand after where its thread holds it (`HANDS`), a free leg after gravity
(`FEET`). What holds her decides what she does: lying on the water she rests on her back on the surface (the outline
of her body turned, `lowest`: the back of her head, her shoulder blades and her seat on it), the surface bobbing
(`float`); standing she is on her feet; lifted she hangs; a body held flat whose knee threads are cut sinks at the
hips until she hangs from her shoulders; a hand or a leg whose thread is cut drops and swings under gravity. Let go
(`release`) she falls with air's drag toward a terminal speed, turning heels over head `tumble` degrees, steered onto
the point she lands on (`land`), the control falling after her on her head thread; yanked by that thread (`yank`)
she hangs from where it is pulled; knotted on again (`knot`) she hangs upright over her place, swinging to rest."""
import math

import numpy as np

from . import armtwist as AT
from . import marionette as MR
from . import perform as PF

GRAVITY = MR.GRAVITY
LIE = (0.50, 0.45)          # her attitude after the control's: late, a swing past (about a fifth) and back
HANG_SHARE = 0.3            # a body held flat whose knee threads are cut hangs from its shoulders: this share of the angle
PEND_ZETA = 0.18            # a hung body's swing under the control dies in about two swings
KNOT_ZETA = 0.35            # knotted on, her first swing dies sooner (gone in about four seconds)
PRESS = 0.6                # lying on the water, the pad gives this share of how far she comes down past it
GIVE_MAX = 0.05            # rad: the most her head gives into the pad as she lands on her back (about 3 degrees)
HANDS = (0.42, 0.42)        # a hand after where its thread (or gravity) holds it
FEET = (0.45, 0.30)         # a leg after gravity, swinging, when its thread is cut while she is in the air
GIVE_HEAD = (0.30, 0.35)    # the head after the body when she comes down on her back on the water: a give into the pad
NECK = 0.12                 # m: the head's pivot to where it presses, for the head's give
REGIME = (0.25, 1.0)        # the standing puppet's channels fade out (let go) and in (knotted on) so
LIFT_OFF = 0.03             # m: lifted this far off the surface she starts to hang (aloft), fully at LIFT_OFF + 0.08
WRIST_LIFT = 0.03           # m: taut wrist threads lift her lying hands this far off the surface
WRIST_R = 0.03              # m: a wrist's thickness, lying on the surface
CLATTER = 0.35              # restitution of the control on the ground it lands on
TREAD = (0.9, 0.05)         # s, m: the hands' treading circles while she falls
FALL_WAIT = 0.8             # s after the control is let go she falls straight; then she is steered to where she lands
YANK_RAMP = 0.06            # s: the yank reaches its speed
YANKED = (0.15, 0.4)        # her body after her crown as it is pulled away (a snatch)
YANK_STOP = 0.5             # s after the yank its pull slows to a stop (she is out of the picture), then she is
CARRY = 0.45                # ... carried back to her place, unseen, arriving this long before the knot
LEG_PRONE = 0.6             # m: the leg bar is held this far behind her head's end over her legs when she lies face down
LEG_SUPINE = 0.15           # m: lying on her back, the leg bar is held this far on past her heels (the knee threads
#                             rise away from her body, never across it seen from above)
CLATTER_SPOT = (-0.25, 0.02)    # m, her frame: where on the ground beside her (her right) the control lands after her

FLAT = math.sin(math.radians(76.0))     # tipped this far back she is flat enough for her hands' lying pose
# where her wrists are held while she lies on her back (`arms`): from the shoulder, x outward, y behind her (None: on
# the surface), z up her body; the elbow's pole from the shoulder; the forearm's twist (degrees, mkmmd.core.armtwist).
# "shoulders": lifted by their threads in front of her shoulders, the elbows down by her sides, the fingers toward her
# head and the palms open to her front (a forearm lying flat toward her head would turn the upper arm past a human one)
LIE_HANDS = {"shoulders": ((0.07, -0.12, 0.03), (0.2, 0.25, -0.5), -55.0),
             "chest": ((-0.03, -0.13, -0.13), (0.40, 0.20, -0.30), 20.0)}


# ---- small maths


def rot_x(a):
    """(n, 3, 3) rotations about +x by `a` (n,) rad (+: +z toward -y, her head forward)."""
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0] = 1.0
    R[..., 1, 1], R[..., 1, 2], R[..., 2, 1], R[..., 2, 2] = c, -s, s, c
    return R


def rot_y(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 1, 1] = 1.0
    R[..., 0, 0], R[..., 0, 2], R[..., 2, 0], R[..., 2, 2] = c, s, -s, c
    return R


def rot_z(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 2, 2] = 1.0
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1] = c, -s, s, c
    return R


def rot_axis(axis, a):
    """(n, 3, 3) rotations by `a` (n,) rad about the unit `axis` (3,)."""
    k = np.asarray(axis, float) / max(float(np.linalg.norm(axis)), 1e-12)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    a = np.asarray(a, float)[..., None, None]
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def window(ts, a, b, ease):
    """0 -> 1 over `ease` s from a, back to 0 over the last `ease` s before b."""
    return PF.smooth((ts - a) / ease) * (1.0 - PF.smooth((ts - b + ease) / ease))


def hull2d(pts):
    """The convex hull (k, 2) of points (m, 2), counter-clockwise (Andrew's monotone chain)."""
    P = np.unique(np.round(np.asarray(pts, float), 5), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0.0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(P), half(P[::-1])
    return np.array(lower[:-1] + upper[:-1])


def lowest(theta, hull, pivot):
    """(n,) how far below her hips (`pivot` (3,), her frame) the lowest point of her outline is when she is turned
    `theta` (n,) rad about her left axis: `hull` (k, 2) is her body's outline seen from her side (y, z; no arms, no
    hair): her soles standing, the back of her head, her shoulder blades and her seat lying on her back."""
    q = np.asarray(hull, float) - np.asarray(pivot, float)[1:3]
    th = np.atleast_1d(np.asarray(theta, float))
    z = np.sin(th)[:, None] * q[None, :, 0] + np.cos(th)[:, None] * q[None, :, 1]
    return -z.min(axis=1)


def float_curve(ts, f):
    """(n,) m: the water's surface under her (`float`): a slow bob `bob` m either way over `period` s, eased in and
    out over `ease` s at the window's ends, and a quick dip on each of `dips` (`dip` m, back up with a small bounce)."""
    ts = np.asarray(ts, float)
    z = np.zeros(len(ts))
    if not f:
        return z
    z = -f["bob"] * np.sin(2.0 * math.pi * (ts - f["from"]) / f["period"]) * window(ts, f["from"], f["to"], f["ease"])
    for t in f["dips"]:
        tau = np.maximum(ts - t, 0.0)
        z = z - f["dip"] * 1.25 * (ts >= t) * (1.0 - np.exp(-tau / 0.04)) * np.exp(-tau / 0.3) * np.cos(
            2.0 * math.pi * tau / 0.8)
    return z


def kick_lifts(ts, K):
    """{side: (n,) m}: her own lazy kicks lying on her back (`kicks`): a knee comes up `height` and down in about half
    a `period`, the legs in turn, each kick a little different in size and pace (`wander`, `seed`)."""
    ts = np.asarray(ts, float)
    out = {s: np.zeros(len(ts)) for s in MR.SIDES}
    if not K:
        return out
    rng = np.random.default_rng(K["seed"])
    win = window(ts, K["from"], K["to"], 0.4)
    other = "R" if K["first"] == "L" else "L"
    for i, s in enumerate((K["first"], other)):
        t = K["from"] + 0.15 + i * 0.5 * K["period"]
        while t < K["to"]:
            size = K["height"] * (1.0 + K["wander"] * rng.uniform(-1.0, 1.0))
            dur = 0.55 * K["period"] * (1.0 + 0.5 * K["wander"] * rng.uniform(-1.0, 1.0))
            u = np.clip((ts - t) / dur, 0.0, 1.0)
            out[s] = np.maximum(out[s], size * np.sin(math.pi * u) ** 2)
            t += K["period"] * (1.0 + K["wander"] * rng.uniform(-0.5, 0.5))
        out[s] = out[s] * win
    return out


def slide(h, thigh, shin):
    """m a foot on the ground slides toward the hip when its knee rises `h` above the straight leg."""
    h = np.clip(np.asarray(h, float), 0.0, 0.95 * min(thigh, shin))
    return thigh + shin - np.sqrt(thigh ** 2 - h ** 2) - np.sqrt(shin ** 2 - h ** 2)


def two_bone(S, W, P, a, b):
    """(n, 3) the elbow of an arm from shoulder S over an upper arm `a` and a forearm `b` m long to wrist W, bent toward
    the pole P (all (n, 3)): the arm IK's answer, as the pose stage's rig gives it."""
    d = W - S
    L = np.clip(np.linalg.norm(d, axis=-1), abs(a - b) + 1e-3, a + b - 1e-4)
    u = _unit(d)
    x = (a * a - b * b + L * L) / (2.0 * L)
    h = np.sqrt(np.maximum(a * a - x * x, 0.0))
    v = P - S
    v = v - np.sum(v * u, -1, keepdims=True) * u
    return S + x[:, None] * u + h[:, None] * _unit(v)


def hand_frame(rest, side, S, E, W, supinate):
    """(dir, palm) (n, 3): the hand of an arm posed S, E, W (her frame) going on along its forearm, turned
    `supinate` (n,) degrees from the forearm's neutral (mid-prone: the palm as the rest pose's, carried round by the
    elbow's hinge, mkmmd.core.armtwist), + palm up. `rest` the left arm's rest directions {upper, fore, dir, palm};
    the right arm is mirrored onto it."""
    m = np.array([1.0 if side == "L" else -1.0, 1.0, 1.0])
    S, E, W = S * m, E * m, W * m
    a0, h0 = AT._unit(rest["fore"]), AT.rest_hinge(rest)
    n = len(S)
    d, p = np.empty((n, 3)), np.empty((n, 3))
    B0 = AT.basis(a0, h0)
    dir0, palm0 = np.asarray(rest["dir"], float), np.asarray(rest["palm"], float)
    for i in range(n):
        h = AT.hinge(S[i], E[i], W[i])
        Re = AT.basis(W[i] - E[i], h) @ B0.T
        R = AT.rotation(W[i] - E[i], AT.SIGN["L"] * float(supinate[i])) @ Re
        d[i], p[i] = R @ dir0, R @ palm0
    return d * m, p * m


# ---- the plan


def regimes(ts, P):
    """({pond, fall, landed, yanked, carry} (n,) bool, {rel, land, yank, knot, carry} clip seconds, inf where the plan
    has no such event): at her place (held over it, lying or standing on it), falling, standing where she landed,
    hung from the pulled head thread, carried back to her place unseen to be knotted on."""
    W = P["world"]
    ts = np.asarray(ts, float)
    rel, land, yank, knot = (W[k]["t"] if W.get(k) else math.inf for k in ("release", "land", "yank", "knot"))
    carry = math.inf
    if knot < math.inf:
        carry = min(yank + YANK_STOP + 0.25 if yank < math.inf else knot - CARRY - 0.6, knot - CARRY - 0.05)
    out = {"pond": (ts < rel) | (ts >= knot), "fall": (ts >= rel) & (ts < min(land, knot)),
           "landed": (ts >= land) & (ts < min(yank, knot)), "yanked": (ts >= yank) & (ts < knot),
           "carry": (ts >= carry) & (ts < knot)}
    return out, {"rel": rel, "land": land, "yank": yank, "knot": knot, "carry": carry}


def held(ts, P, part):
    """mkmmd.core.marionette.held, a thread knotted back on from the start of the carry (unseen): she hangs as she
    will once she is seen again."""
    h = MR.held(ts, P, part)
    _r, Tm = regimes(ts, P)
    if Tm["knot"] < math.inf:
        h = np.maximum(h, (np.asarray(ts) >= Tm["carry"]).astype(float))
    return h


def taut(ts, P, cross):
    """(n,) bool: frames her threads pull her (she is at her place, held, the control not let down below her rest):
    the frames the threads' lengths are taken from (the shortest span each has then: straight whenever it pulls)."""
    Rg, _ = regimes(ts, P)
    return (Rg["pond"] | Rg["carry"]) & (np.asarray(cross["lift"]) >= -0.005) & (np.asarray(ts) >= P["from"])


def _at(ts, t, X):
    return np.array([np.interp(t, ts, X[:, j]) for j in range(X.shape[1])])


def tethered(ts, anchor, p0, v0, length, terminal, ball=None, spot=None, steer=None, loose=math.inf, h=1.0 / 240.0):
    """(n, 3): a light thing let go at ts[0] from p0 with velocity v0, falling with air's drag toward `terminal` m/s,
    tied `length` m from `anchor` (n, 3) (pulled after it while the thread is taut; from `loose` on the thread pays out
    and no longer holds it), landing and bouncing on `ball` (center, radius) when given and drawn toward `spot` (3,)
    over the times `steer` (t0, t1)."""
    ts = np.asarray(ts, float)
    grid = np.arange(ts[0], ts[-1] + h, h)
    A = np.stack([np.interp(grid, ts, anchor[:, j]) for j in range(3)], 1)
    VA = np.gradient(A, h, axis=0) if len(grid) > 1 else np.zeros_like(A)
    p, v = np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    k = GRAVITY / terminal
    out = np.empty((len(grid), 3))
    resting = False
    for i, t in enumerate(grid):
        if resting:                                             # lying where it came to rest (the thread pays out)
            out[i] = p
            continue
        a = np.array([0.0, 0.0, -GRAVITY]) - k * v
        if spot is not None and steer[0] <= t < steer[1]:
            w = (t - steer[0]) / max(steer[1] - steer[0], 1e-3)
            a[:2] += 40.0 * w * (np.asarray(spot, float)[:2] - p[:2]) - 9.0 * w * v[:2]
        v = v + h * a
        p = p + h * v
        d = p - A[i]
        r = float(np.linalg.norm(d))
        if r > length and t < loose:                            # the thread is taut: it pulls the thing after her
            nrm = d / r
            p = A[i] + nrm * length
            vr = float((v - VA[i]) @ nrm)
            if vr > 0.0:
                v = v - vr * nrm
        if ball is not None:
            c, rad = np.asarray(ball[0], float), float(ball[1])
            d = p - c
            r = float(np.linalg.norm(d))
            if r < rad:                                         # on the ground: it bounces, rubs, comes to rest
                nrm = d / max(r, 1e-9)
                p = c + nrm * rad
                vn = float(v @ nrm)
                if vn < 0.0:
                    v = v - (1.0 + CLATTER) * vn * nrm
                v = v - (v - (v @ nrm) * nrm) * min(1.0, 30.0 * h)
                resting = t >= loose and float(np.linalg.norm(v)) < 0.25
        out[i] = p
    return np.stack([np.interp(ts, grid, out[:, j]) for j in range(3)], 1)


def world_plan(ts, P, cross, rig, aims=None):
    """Her root, her body's channels, her hands and the control in the world over `ts` (clip seconds) from the plan P
    (mkmmd.core.marionette.normalize: its `world`), the cross's own plan (`cross_plan`) and `rig`, her measures in her
    frame at the settled pose: `pivot` (her hips), `crown` (where the head thread meets her), `hull` (k, 2) (her
    outline seen from her side, `lowest`), `shoulders`, `hips` and `ankles` {side: (3,)} (the legs' joints and the
    ankles' goals), `arms` {side: {upper, fore}}, `legs` {side: {thigh, shin, foot}}, `rest` (the left arm's rest
    directions, mkmmd.core.armtwist), and the card's `anchors` {part: (3,) in its bar's frame}, `leg_home` (3,).
    `aims` {"reach": [(n, 3)], "land": (n, 3)}: the targets of `reach` and `land` in her frame on every frame; a reach
    target flown on her is {"on": (n, 3) where it hangs on her at rest, "rel": (n, 3) its offset from there}: it goes
    where her body takes that point.

    Returns {"root": {pos (n, 3), rot (n, 3, 3)}, the channels mkmmd.core.marionette.body_plan returns (rise 0: the
    root carries her), "hands" {side: {wrist, dir, palm, pole} (n, 3)}, "cross" and "leg" {pos (n, 3), rot (n, 3, 3),
    shown (n,)} (the bar's origin and turn), "grab" (n, 3) (where the head thread is taken on a yank), "rest" (n, 3)
    and "resting" (n,) (the surface under her while she is at her place), "theta" (n,) rad, "aloft" (n,), "hips" (n,
    3), "regimes", "times"}."""
    ts = np.asarray(ts, float)
    n = len(ts)
    lag = P["lag"]
    W = P["world"]
    if W.get("free"):                                           # seated, gathered, snipped, set free (marionette_free)
        from . import marionette_free
        return marionette_free.free_plan(ts, P, cross, rig, aims or {})
    aims = aims or {}
    pivot = np.asarray(rig["pivot"], float)
    crown = np.asarray(rig["crown"], float)
    hull = np.asarray(rig["hull"], float)
    Rg, Tm = regimes(ts, P)
    pond, fall, landed, yanked, carry = (Rg[k] for k in ("pond", "fall", "landed", "yanked", "carry"))
    pondw = np.clip(MR.follow(ts, pond.astype(float), REGIME, lag), 0.0, 1.0)
    hd, bk = held(ts, P, "head"), held(ts, P, "back")
    hung = np.maximum(hd, bk) * (pond | carry)
    knees_held = 0.5 * (held(ts, P, "knee_L") + held(ts, P, "knee_R"))
    # ---- how high her threads hold her over her surface: the control's lift (a thread pulls, it cannot push)
    th_key = np.radians(PF.eased_keys(ts, W["lie"]))
    lift = hung * (np.maximum(cross["lift"], 0.0) + P["hang"] * np.cos(th_key) ** 2 * np.minimum(2.0 * knees_held, 1.0))
    aloft = PF.smooth((MR.follow(ts, lift, MR.BODY, lag) - LIFT_OFF) / 0.08) * hung
    # ---- her attitude: the control's, late; a body held flat sinks at the hips when its knee threads go
    tgt = th_key * (1.0 - aloft * (1.0 - knees_held) * (1.0 - HANG_SHARE))
    total = 0.0
    if Tm["rel"] < math.inf:                                    # falling she comes upright, turning `tumble` on the way
        total = math.radians(W["release"]["tumble"])
        t_end = Tm["land"] if Tm["land"] < math.inf else ts[-1]
        th0 = float(np.interp(Tm["rel"], ts, tgt))
        e = PF.smooth((ts - Tm["rel"] - 0.15) / max(t_end - Tm["rel"] - 0.5, 0.3))
        tgt = np.where((ts >= Tm["rel"]) & (ts < Tm["knot"]), th0 * (1.0 - e) + total * e, tgt)
        tgt = np.where(ts >= Tm["knot"], tgt + total, tgt)       # the turns she made are kept, let go below
    theta = MR.follow(ts, tgt, LIE, lag) - total * (ts >= Tm["knot"])
    s = np.sin(theta)
    upright, supine = np.cos(theta) ** 2, np.maximum(-s, 0.0) ** 2
    # ---- at her place: she hangs from where her back thread meets her (`hang`), which follows the control late; the
    #      surface stops her (her back lying, her soles standing, her heels as she is pulled up); hung, she swings
    surf = float_curve(ts, W["float"])
    low, low_k = lowest(theta, hull, pivot), lowest(th_key, hull, pivot)
    hop, hop_dip, _ = MR.landing_hops(ts, P.get("landings"))
    L_eff = P["height"] + max(float(crown[2] - pivot[2]), 0.3)
    T_pend = 2.0 * math.pi * math.sqrt(L_eff / GRAVITY)
    ctrl = np.stack([cross["pos"][:, 0], np.zeros(n)], 1)
    pend = MR.follow(ts, ctrl, (T_pend, PEND_ZETA), lag) * aloft[:, None]
    if Tm["knot"] < math.inf:                                   # knotted on, she swings to rest; a looping clip
        w = 2.0 * math.pi / T_pend                              # (`loop`, its length) swings so from `loop` s
        amp = L_eff * math.tan(math.radians(W["knot"]["swing"]))    # earlier too: its start goes on as its end
        loop = W["knot"].get("loop")
        for t0 in [Tm["knot"]] + ([Tm["knot"] - loop] if loop else []):
            tau = np.maximum(ts - t0, 0.0)
            pend[:, 1] += (ts >= t0) * amp * np.exp(-KNOT_ZETA * w * tau) * np.cos(w * tau)
    q = np.asarray(rig["hang"], float) - pivot                  # her hang point, from her hips
    lead = np.zeros((n, 3))
    lead[:, :2] = pivot[:2] + pend
    lead[:, 2] = surf + low_k + lift
    Q = MR.follow(ts, lead + np.einsum("nij,j->ni", rot_x(th_key), q), MR.BODY, lag)
    H = Q - np.einsum("nij,j->ni", rot_x(theta), q)
    gap = H[:, 2] - (surf + low)                                # how far her lowest point is over the surface
    H[:, 2] = np.where(gap >= 0.0, H[:, 2], surf + low + PRESS * gap * (1.0 - upright)) + hop * upright
    dip = (0.8 * np.maximum(-gap, 0.0) + hop_dip) * upright      # standing, her knees take a landing; lying, the water
    H_pond, dip_pond = H.copy(), dip.copy()
    swing = np.arctan2(pend - ctrl * aloft[:, None], L_eff)     # the swing's tilt (about y from x, about x from y)
    # ---- let go: she falls, comes upright and lands on her point
    if Tm["rel"] < math.inf:
        i0 = int(np.clip(np.searchsorted(ts, Tm["rel"]), 1, n - 1))
        dt = max(ts[i0] - ts[i0 - 1], 1e-3)
        p0, v0 = H[i0 - 1].copy(), (H[i0] - H[i0 - 1]) / dt
        vt = W["release"]["terminal"]
        k = vt / GRAVITY
        tau = np.maximum(ts - ts[i0 - 1], 0.0)[:, None]
        vterm = np.array([0.0, 0.0, -vt])
        Hf = p0 + vterm * tau + (v0 - vterm) * k * (1.0 - np.exp(-tau / k))
        if Tm["land"] < math.inf:
            g = W["land"]
            at = np.asarray(aims["land"], float)
            Hl = at + np.array([0.0, 0.0, pivot[2]])            # she stands on the point (it may move: a ball bobs)
            q = _at(ts, Tm["land"], Hl) + np.outer(np.maximum(Tm["land"] - ts, 0.0), (0.0, 0.0, g["speed"]))
            ta = min(Tm["rel"] + FALL_WAIT, Tm["land"] - g["approach"] - 0.3)
            w = PF.smooth((ts - ta) / max(Tm["land"] - g["approach"] - ta, 0.2))[:, None]
            Hf = Hf * (1.0 - w) + q * w
            H = np.where(landed[:, None], Hl, H)
            dip = np.where(landed, g["give"] * MR.follow(ts, MR.pulse(ts, Tm["land"], 0.05, 0.08, 0.45), MR.GIVE, lag),
                           dip)
        H = np.where(fall[:, None], Hf, H)
    R = rot_y(-swing[:, 0]) @ rot_x(theta + swing[:, 1])
    pos = H - np.einsum("nij,j->ni", R, pivot)
    # ---- yanked by her head thread, from where the control lay beside her
    grab = pos + np.einsum("nij,j->ni", R, crown)
    if Tm["yank"] < math.inf:
        y = W["yank"]
        ctl = _control(ts, P, cross, rig, th_key, H, R, pos, Rg, Tm, None)
        k = int(np.clip(np.searchsorted(ts, Tm["yank"]) - 1, 0, n - 1))     # where she stands as the hand takes it
        cr0 = grab[k].copy()
        eye = ctl["cross"]["pos"][k] + ctl["cross"]["rot"][k] @ np.asarray(rig["anchors"]["head"], float)
        G0 = eye + y["at"] * (cr0 - eye)
        tau = np.maximum(ts - Tm["yank"], 0.0)
        stop = YANK_STOP - YANK_RAMP * (1.0 - math.exp(-YANK_STOP / YANK_RAMP))
        ramp = np.where(tau > YANK_STOP, stop + 0.35 * (1.0 - np.exp(-(tau - YANK_STOP) / 0.35)),
                        tau - YANK_RAMP * (1.0 - np.exp(-tau / YANK_RAMP)))
        G = G0 + np.outer(ramp, y["v"])
        slack = 1.05 * max(y["at"] * P["height"], float(np.linalg.norm(G0 - cr0)))   # the thread to her, taken up first
        pull = np.maximum(np.linalg.norm(G - cr0, axis=1) - slack, 0.0)
        cr = cr0 + _unit(G - cr0) * pull[:, None]
        Hy = MR.follow(ts, cr - (crown - pivot), YANKED, lag)
        H = np.where(yanked[:, None], Hy, H)
        grab = np.where((ts < Tm["yank"])[:, None], G0, G)
    # ---- carried back unseen and knotted on: she hangs over her place again
    if Tm["knot"] < math.inf:
        w = PF.smooth((ts - Tm["carry"]) / max(Tm["knot"] - CARRY - Tm["carry"], 0.2))[:, None]
        on = carry | (ts >= Tm["knot"])
        H = np.where(on[:, None], H * (1.0 - w) + H_pond * w, H)
        dip = np.where(ts >= Tm["knot"], dip_pond, dip)
    R = rot_y(-swing[:, 0]) @ rot_x(theta + swing[:, 1])
    pos = H - np.einsum("nij,j->ni", R, pivot)
    ctl = _control(ts, P, cross, rig, th_key, H, R, pos, Rg, Tm, grab if Tm["yank"] < math.inf else None)
    # ---- her body: the standing puppet's channels where she stands at her place; lying, flying, falling they rest
    stand = upright * (1.0 - aloft) * pond
    body = MR.body_plan(ts, P, cross, rig["legs"], stand=stand)
    for key in ("lean", "tilt", "turn", "head_tilt", "head_nod", "head_turn", "guitar"):
        body[key] = body[key] * pondw
    body["head_nod"] = body["head_nod"] - MR.HEAD_HELD * MR.follow(ts, hd, MR.BODY, lag) * (1.0 - upright) * pondw
    drop = MR.follow(ts, H[:, 2], GIVE_HEAD, lag) - H[:, 2]     # the head after her body: as she comes down on her
    body["head_nod"] = body["head_nod"] + np.clip(drop / NECK, -GIVE_MAX, 0.0) * supine * (1.0 - aloft) * pondw   # the
    #  press only: coming down her head is not nodded on; once her body stops on the pad it gives, chin up a little
    if Tm["land"] < math.inf:                                   # landing: the chest goes forward with the knees, back
        body["lean"] = body["lean"] + math.radians(9.0) * MR.follow(
            ts, MR.pulse(ts, Tm["land"], 0.06, 0.1, 0.5), MR.GIVE, lag)
    if Tm["yank"] < math.inf:                                   # pulled up by the crown: her chin comes up
        body["head_nod"] = body["head_nod"] - math.radians(12.0) * MR.follow(
            ts, MR.pulse(ts, Tm["yank"], 0.06, 0.4, 0.5), MR.DROP, lag)
    body["rise"] = np.zeros(n)
    body["dip"] = dip + body["cut_dip"]
    body["side"] = body["side"] * stand
    body["behind"] = body["behind"] * stand
    # ---- her legs: the leg bar's knees standing, her own kicks lying, gravity when they are free, kicks swimming
    kicks = kick_lifts(ts, W["kicks"])
    sw = W["swim"]
    swim_w = window(ts, sw["from"], sw["to"], 0.3) * pond if sw else np.zeros(n)
    phase = 2.0 * math.pi * (ts - sw["from"]) / sw["period"] if sw else np.zeros(n)
    down = np.stack([np.zeros(n), -np.sin(theta + swing[:, 1]), -np.cos(theta + swing[:, 1])], 1)   # gravity, her frame
    feet, point = {}, {}
    for side in MR.SIDES:
        lg = rig["legs"][side]
        thigh, shin = float(lg["thigh"]), float(lg["shin"])
        hipj, ank = np.asarray(rig["hips"][side], float), np.asarray(rig["ankles"][side], float)
        f = body["feet"][side] * (upright * pondw)[:, None]
        f[:, 2] += slide(body["knee"][side] * pondw + kicks[side], thigh, shin) * supine   # lying: the foot slides up
        if sw:                                                  # swimming: the legs flutter, in turn
            f[:, 1] += sw["kick"] * np.sin(2.0 * phase + (0.0 if side == sw["first"] else math.pi)) * swim_w
        free = np.maximum(aloft * (1.0 - held(ts, P, f"knee_{side}")) * pond, yanked.astype(float))
        hang = MR.follow(ts, hipj + 0.97 * np.linalg.norm(ank - hipj) * down - ank, FEET, lag)
        feet[side] = f * (1.0 - free[:, None]) + hang * free[:, None]
        point[side] = (body["point"][side] * upright * (1.0 - free) + 0.45 * supine * (1.0 - free) + 0.6 * free
                       + 0.25 * swim_w) * (1.0 - fall)
    body["feet"], body["point"] = feet, point
    hands = _hands(ts, P, rig, aims, theta, swing, R, pos, surf, Rg, Tm, swim_w, phase, lift)
    # the surface she rests on (a collider the sim lays her hair and ears on), under her while she is at her place
    resting = pond & (H[:, 2] - low - surf < 0.6)
    rest = np.stack([np.full(n, pivot[0]), np.full(n, pivot[1]), surf], 1)
    rest = np.where(resting[:, None], rest, np.array([0.0, 0.0, -60.0]))
    body["report"].update({
        "lie_deg": [round(float(np.degrees(theta.min())), 1), round(float(np.degrees(theta.max())), 1)],
        "aloft_s": round(float(aloft.sum() * (ts[1] - ts[0])), 2) if n > 1 else 0.0,
        "height_mm": round(float((H[:, 2] - low - surf)[pond].max()) * 1000, 1) if pond.any() else 0.0,
        "times": {k: (round(float(v), 3) if v < math.inf else None) for k, v in Tm.items()}})
    return {"root": {"pos": pos, "rot": R}, **body, "hands": hands, "cross": ctl["cross"], "leg": ctl["leg"],
            "grab": grab, "rest": rest, "resting": resting, "theta": theta, "aloft": aloft, "hips": H,
            "regimes": Rg, "times": Tm, "rig": rig}


def _to_level(R, pos, q):
    """(n, 3) level-frame points of body-frame points q (n, 3) or (3,)."""
    return pos + np.einsum("nij,nj->ni", R, np.broadcast_to(np.asarray(q, float), pos.shape))


def _to_body(R, pos, x):
    return np.einsum("nji,nj->ni", R, np.asarray(x, float) - pos)


def _hands(ts, P, rig, aims, theta, swing, R, pos, surf, Rg, Tm, swim_w, phase, lift):
    """{side: {wrist, dir, palm, pole}} (n, 3) in her frame: where her threads, the surface and gravity put her hands,
    late; the hand going on along its forearm, turned from the arm's own neutral as each pose asks (mkmmd.core.
    armtwist), so neither the forearm nor the upper arm turns past a human one."""
    n = len(ts)
    W = P["world"]
    lag = P["lag"]
    pond, fall, landed, yanked, carry = (Rg[k] for k in ("pond", "fall", "landed", "yanked", "carry"))
    lie_at, lie_pole, lie_sup = LIE_HANDS[W["arms"]]
    s = np.sin(theta)
    prone = np.maximum(s, 0.0) ** 2
    supine = PF.smooth((np.maximum(-s, 0.0) - FLAT) / (1.0 - FLAT))     # her hands hang as she is tipped onto her back
    upright = np.maximum(1.0 - prone - supine, 0.0)                    # and take the lying pose once she is flat
    down = np.stack([np.zeros(n), -np.sin(theta + swing[:, 1]), -np.cos(theta + swing[:, 1])], 1)
    out = {}
    for side in MR.SIDES:
        sg = 1.0 if side == "L" else -1.0
        S = np.asarray(rig["shoulders"][side], float)
        a, b = float(rig["arms"][side]["upper"]), float(rig["arms"][side]["fore"])
        reach = a + b
        hold = held(ts, P, f"wrist_{side}") * (pond | carry)
        tg, wt, poles, sup = [], [], [], []

        def add(target, weight, pole, supin):
            tg.append(np.broadcast_to(np.asarray(target, float), (n, 3)))
            wt.append(np.asarray(weight, float) * np.ones(n))
            poles.append(np.broadcast_to(np.asarray(pole, float), (n, 3)))
            sup.append(float(supin))

        hull = np.asarray(rig["hull"], float)
        near = np.abs(hull[:, 1] - (S[2] - 0.1)) < 0.18             # her outline round her shoulder blades
        back = float(hull[near, 0].max() - S[1]) if near.any() else 0.1
        hang = S + reach * 0.93 * _unit(np.array([sg * 0.10, -0.20, -1.0]))
        pole0 = S + np.array([sg * 0.45, 0.25, -0.30])
        stay = hold * (1.0 - swim_w)
        add(hang, stay * upright, pole0, 0.0)                    # standing or hung upright: her arms hang
        lie = np.tile(S + np.array([sg * lie_at[0], lie_at[1] or 0.0, lie_at[2]]), (n, 1))
        if lie_at[1] is None:                                    # lying on her back: on the plane of her back (the
            lie[:, 1] = S[1] + back - WRIST_R - WRIST_LIFT * (np.asarray(lift) >= -0.005)   # surface), up by her face
        add(lie, stay * supine, S + np.array([sg * lie_pole[0], lie_pole[1], lie_pole[2]]), lie_sup)
        add(S + reach * 0.9 * _unit(np.array([sg * 0.15, -0.35, 0.9])), stay * prone,      # held flat face down: the
            S + np.array([sg * 0.45, -0.1, -0.2]), -30.0)                                   # threads take them ahead
        if W["swim"]:                                           # a dog paddle: forward high, down, back, up
            ph = phase + (0.0 if side == W["swim"]["first"] else math.pi)
            st = W["swim"]["stroke"]
            c0 = S + np.array([sg * 0.12, -0.28, 0.22])
            loop = np.stack([np.full(n, c0[0]), c0[1] - st / 3.0 * np.sin(ph), c0[2] + st / 2.0 * np.cos(ph)], 1)
            add(loop, swim_w * hold, S + np.array([sg * 0.45, -0.05, -0.25]), -60.0)
        for j, r in enumerate(W["reach"]):                       # the hand bar carries her hands after a target
            T = aims.get("reach", [])[j] if j < len(aims.get("reach", [])) else None
            if T is None:
                continue
            if isinstance(T, dict):                              # flown on her: it hangs where her body takes it
                T = pos + np.einsum("nij,nj->ni", R, np.asarray(T["on"], float)) + np.asarray(T["rel"], float)
            d = _to_body(R, pos, MR.follow(ts, np.asarray(T, float), (0.5, 0.5), lag)) - S
            dist = np.linalg.norm(d, axis=1)
            u = d / np.maximum(dist, 1e-6)[:, None]
            across = _unit(np.cross(u, np.array([0.0, 0.0, 1.0])))
            to = S + u * np.clip(dist - 0.05, 0.15, 0.9 * reach)[:, None] - across * sg * 0.06
            ww = window(ts, r["from"], r["to"], r["ease"]) * hold
            if side == "R":
                ww = np.interp(ts - 0.12, ts, ww)                # the other hand a little behind
            add(to, ww * 1.5, S + np.array([sg * 0.45, 0.15, -0.35]), -20.0)
        free = S + reach * 0.95 * _unit(down + np.array([sg * 0.12, -0.12, 0.0]))   # cut free: it hangs and swings
        add(free, (1.0 - hold) * (pond | yanked | carry), pole0, 0.0)
        if Tm["rel"] < math.inf:                                # falling: treading the air, out at her sides
            ph = 2.0 * math.pi * (ts - Tm["rel"]) / TREAD[0] + (0.0 if side == "L" else 1.7)
            tread = np.stack([S[0] + sg * (0.30 + TREAD[1] * np.cos(ph)), np.full(n, S[1] - 0.12),
                              S[2] + 0.02 + TREAD[1] * np.sin(ph)], 1)
            add(tread, fall.astype(float), S + np.array([sg * 0.4, 0.2, -0.35]), -40.0)
        if Tm["land"] < math.inf:                               # landing: arms out for her balance, then down
            g = window(ts, Tm["land"] - 0.05, Tm["land"] + 1.4, 0.5)
            wob = 0.05 * np.sin(2.0 * math.pi * 2.2 * (ts - Tm["land"])) * np.exp(-np.maximum(ts - Tm["land"], 0.0) / 0.6)
            out_at = np.stack([np.full(n, S[0] + sg * 0.40), np.full(n, S[1] - 0.10), S[2] + 0.02 + sg * wob], 1)
            add(out_at, g * landed, S + np.array([sg * 0.3, 0.3, -0.3]), -45.0)
            add(hang, (1.0 - g) * landed, pole0, 0.0)
        Wt = np.stack(wt, 0)
        Wt = np.concatenate([Wt, np.maximum(1.0 - Wt.sum(0), 0.0)[None]], 0)    # nothing asks: her arms hang
        tg.append(np.broadcast_to(hang, (n, 3)))
        poles.append(np.broadcast_to(pole0, (n, 3)))
        sup.append(0.0)
        Wt = Wt / np.maximum(Wt.sum(0), 1e-9)
        target = np.einsum("kn,knj->nj", Wt, np.stack(tg, 0))
        pole = np.einsum("kn,knj->nj", Wt, np.stack(poles, 0))
        supin = np.einsum("kn,k->n", Wt, np.array(sup))
        # late and loose, about the shoulder: the hand swings on its arm, never through her
        rel = target - S
        u = _unit(MR.follow(ts, _unit(rel), HANDS, lag))
        rad = np.clip(MR.follow(ts, np.clip(np.linalg.norm(rel, axis=1), 0.12, 0.97 * reach), HANDS, lag), 0.12,
                      0.97 * reach)
        wrist = _clear_torso(S + u * rad[:, None], rig, sg)
        E = two_bone(np.tile(S, (n, 1)), wrist, pole, a, b)
        d, p = hand_frame(rig["rest"], side, np.tile(S, (n, 1)), E, wrist, MR.follow(ts, supin, HANDS, lag))
        out[side] = {"wrist": _to_level(R, pos, wrist), "dir": np.einsum("nij,nj->ni", R, d),
                     "palm": np.einsum("nij,nj->ni", R, p), "pole": _to_level(R, pos, pole), "body": wrist,
                     "elbow": E}
    return out


def _clear_torso(wrist, rig, sg):
    """Wrist points (n, 3) in her frame pushed out sideways from her torso (a box from her hips to her shoulders, as
    wide as her shoulders less a hand, as deep as her chest)."""
    S = np.asarray(rig["shoulders"]["L"], float)
    half = abs(S[0]) * 0.85
    lo, hi = float(np.asarray(rig["pivot"], float)[2]) - 0.05, float(S[2])
    inside = (wrist[:, 2] > lo) & (wrist[:, 2] < hi) & (np.abs(wrist[:, 1] - S[1]) < 0.12) & (sg * wrist[:, 0] < half)
    w = wrist.copy()
    w[:, 0] = np.where(inside, sg * half, w[:, 0])
    return w


def _control(ts, P, cross, rig, th_key, H, R, pos, Rg, Tm, grab):
    """The cross (its origin) and the leg bar in her frame: held over her as she stands (the standing puppet's layout),
    over her head's end and her legs as she lies (`over` m on past her head, so the bars do not hang over her face
    seen from above), let go after her on her head thread, landing beside her, dangling from a yank (`grab` the hand),
    back in hand when she is knotted on."""
    n = len(ts)
    W = P["world"]
    pivot, crown = np.asarray(rig["pivot"], float), np.asarray(rig["crown"], float)
    a_head = np.asarray(rig["anchors"]["head"], float)
    leg_home = np.asarray(rig["leg_home"], float)
    sk = np.sin(th_key)
    lead = np.tile(pivot, (n, 1))                                # where her crown would be at the control's attitude
    lead[:, 2] = lowest(th_key, np.asarray(rig["hull"], float), pivot)
    crown_k = lead + np.einsum("nij,j->ni", rot_x(th_key), crown - pivot)
    home = crown_k + np.array([0.0, 0.0, P["height"]]) - a_head
    over = PF.eased_keys(ts, P["over"]) if isinstance(P["over"], list) else P["over"]
    home[:, 1] -= over * sk                                      # on past her head's end as she lies
    held_rot = np.einsum("nij,njk,nkl->nil", rot_z(cross["turn"]), rot_y(cross["tilt"]), rot_x(cross["rock"]))
    lh = np.tile(leg_home, (n, 1))
    lh[:, 1] = leg_home[1] + (LEG_PRONE - leg_home[1]) * np.maximum(sk, 0.0)   # over her legs lying face down
    sup = np.maximum(-sk, 0.0)                                   # on her back: on past her heels (her crown's height
    lh[:, 1] = lh[:, 1] * (1.0 - sup) - (float(crown[2]) + LEG_SUPINE + over) * sup     # standing is her length)
    held_pos, held_leg = home + cross["pos"], home + lh + cross["leg_pos"]
    leg_rot = np.einsum("nij,njk->nik", rot_z(0.5 * cross["turn"]), rot_y(-cross["leg_roll"]))
    cpos, crot, lpos, lrot = held_pos.copy(), held_rot.copy(), held_leg.copy(), leg_rot.copy()
    cshown, lshown = cross["shown"].copy(), cross["shown"].copy()
    if Tm["rel"] < math.inf:
        r = W["release"]
        i0 = int(np.clip(np.searchsorted(ts, Tm["rel"]), 1, n - 1))
        dt = max(ts[i0] - ts[i0 - 1], 1e-3)
        top = _to_level(R, pos, crown)                          # her crown, where the head thread is tied
        L = float(np.linalg.norm(cpos[i0 - 1] + a_head - top[i0 - 1]))
        v0 = (cpos[i0] - cpos[i0 - 1]) / dt
        sel = ts >= ts[i0 - 1]
        fallen = tethered(ts[sel], top[sel], cpos[i0 - 1] + a_head, v0, L, 0.6 * r["terminal"]) - a_head
        tau = np.maximum(ts[sel] - Tm["rel"], 0.0)
        rot = np.einsum("nij,jk->nik", rot_axis((0.35, 1.0, 0.25), math.radians(r["spin"]) * tau), crot[i0 - 1])
        if Tm["land"] < math.inf:                               # she has landed: its turns slip off the stirrer and it
            g = W["land"]                                       # clatters down on the ground beside her, `clatter` s on
            tl, T = ts[sel], g["clatter"]
            k = int(np.clip(np.searchsorted(tl, Tm["land"]), 1, len(tl) - 1))
            pL, vL = fallen[k], (fallen[k] - fallen[k - 1]) / max(tl[k] - tl[k - 1], 1e-3)
            at = _at(ts, Tm["land"], H) - np.array([0.0, 0.0, pivot[2]])
            d = min(math.hypot(*CLATTER_SPOT), 0.8 * g["radius"])
            u = np.array([CLATTER_SPOT[0], CLATTER_SPOT[1], 0.0]) / max(math.hypot(*CLATTER_SPOT), 1e-6)
            spot = at + u * d - np.array([0.0, 0.0, g["radius"] - math.sqrt(g["radius"] ** 2 - d * d)])
            spot = spot + np.array([0.0, 0.0, 0.008])           # its thickness on the ground
            acc = 2.0 * (spot - pL - vL * T) / (T * T)          # the arc that lands it there on time
            s_ = np.clip(tl - tl[k], 0.0, None)[:, None]
            arc = np.where(s_ <= T, pL + vL * s_ + 0.5 * acc * s_ * s_, spot)
            b = np.maximum(tl - tl[k] - T, 0.0)                 # two bounces, then it lies there
            hop = np.where(b < 0.18, 0.05 * 4 * (b / 0.18) * (1 - b / 0.18),
                           np.where(b < 0.28, 0.02 * 4 * ((b - 0.18) / 0.1) * (1 - (b - 0.18) / 0.1), 0.0))
            arc[:, 2] += np.where(s_[:, 0] > T, hop, 0.0)
            fallen = np.where((tl >= tl[k])[:, None], arc, fallen)
            kk = int(np.searchsorted(tl, tl[k] + T))
            if kk < len(rot):                                   # flat on the ground, a wobble as it bounces
                flat = _flatten(rot[kk], spot - (at - np.array([0.0, 0.0, g["radius"]])))
                e = PF.smooth((tl[kk:] - tl[kk]) / 0.12)
                rot[kk:] = np.array([_slerp_m(rot[kk], flat, float(x)) for x in e])
        live = sel & (ts < Tm["knot"]) & ~Rg["carry"]
        cpos = np.where(live[:, None], _pad(fallen, sel), cpos)
        crot = np.where(live[:, None, None], _pad(rot, sel), crot)
        lv = (lpos[i0] - lpos[i0 - 1]) / dt                     # the leg bar, on no thread, drops out of the picture
        lt = np.maximum(ts - ts[i0 - 1], 0.0)
        ldrop = lpos[i0 - 1] + np.outer(lt, lv) + np.outer(0.5 * lt * lt, (0.0, 0.0, -GRAVITY))
        lpos = np.where(live[:, None], ldrop, lpos)
        lrot = np.where(live[:, None, None], np.einsum("nij,jk->nik", rot_axis((1.0, 0.3, 0.2), 4.0 * lt), lrot[i0 - 1]),
                        lrot)
        lshown = lshown & ~(live & (ts >= Tm["rel"] + 1.2))
    if grab is not None:                                        # the control dangles under the hand that pulls
        y = W["yank"]
        k = int(np.clip(np.searchsorted(ts, Tm["yank"]) - 1, 0, n - 1))
        drop = max((1.0 - y["at"]) * P["height"], float(np.linalg.norm(cpos[k] + a_head - grab[k + 1 if k + 1 < n else k])))
        hang = MR.follow(ts, grab - a_head - np.array([0.0, 0.0, drop]), (0.45, 0.3), P["lag"])
        on = Rg["yanked"] & ~Rg["carry"]
        w = (PF.smooth((hang[:, 2] - cpos[:, 2]) / 0.15) * on)[:, None]       # it leaves the ground as its thread
        cpos = cpos * (1.0 - w) + hang * w                                     # comes taut under the hand
    if Tm["knot"] < math.inf:                                   # back in hand: carried there unseen
        k0 = Rg["carry"]
        w = PF.smooth((ts - Tm["carry"]) / max(Tm["knot"] - CARRY - Tm["carry"], 0.2))[:, None]
        i_c = int(np.clip(np.searchsorted(ts, Tm["carry"]), 0, n - 1))
        cpos = np.where(k0[:, None], cpos[i_c] * (1.0 - w) + held_pos * w, cpos)
        crot = np.where(k0[:, None, None], held_rot, crot)
        lpos = np.where(k0[:, None], held_leg, lpos)
        lrot = np.where(k0[:, None, None], leg_rot, lrot)
        lshown = lshown | k0 | (ts >= Tm["knot"])
    return {"cross": {"pos": cpos, "rot": crot, "shown": cshown}, "leg": {"pos": lpos, "rot": lrot, "shown": lshown}}


def _pad(x, sel):
    """x (m, ...) given on the frames `sel` (n,) bool, as (n, ...) with its first value before."""
    out = np.empty((len(sel),) + x.shape[1:])
    out[sel] = x
    out[~sel] = x[0]
    return out


def _flatten(Rm, up=(0.0, 0.0, 1.0)):
    """The rotation nearest Rm whose bars lie flat on a ground of normal `up` (its z along it)."""
    z = _unit(np.asarray(up, float))
    x = Rm[:, 0] - (Rm[:, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def _slerp_m(A, B, t):
    """The rotation a share t of the way from A to B (3, 3)."""
    D = B @ A.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(D) - 1.0) / 2.0)))
    if ang < 1e-6:
        return B.copy() if t >= 1.0 else A.copy()
    ax = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    if np.linalg.norm(ax) < 1e-9:
        return B.copy() if t >= 0.5 else A.copy()
    return rot_axis(ax, ang * t) @ A
