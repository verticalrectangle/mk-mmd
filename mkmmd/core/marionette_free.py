"""A marionette set free, bpy-free (docs/design.md: Marionette, set free): the keys `seat`, `gather`, `scissors`, `snip`,
`ascend`, `haul` and `crossed` (mkmmd.core.marionette.FREE_KEYS) with the knot and the clip's `loop`.

She sits in her seat on her threads (her hips on its point, her thighs over its front, her shins hanging), the control
over her head, alive under it (its own keys move her chest and head as the standing puppet's do). Her fist gathers the
six threads into one bundle over her head (`gather`), her other hand takes the scissors from where they lie and brings
them to it (`scissors`); on the `snip` the blades shut, every thread is cut just above her fist, the control is dropped
and falls past her with the upper pieces, and the pieces left on her are her tails (mkmmd.core.marionette_tails),
held in her fist a moment, then let go with the scissors. Weightless she floats up out of her seat (`ascend`): her hips
first, her legs trailing, her arms opening, her head steady, faster and faster, then (unseen) easing to a hover where
she turns slowly. A hand comes up under her with the ends of her tails (`haul`): ON its time it shuts and pulls, they
snap taut, she jolts down, her limbs flung up, and she settles facing the lens; she is hauled down, in tugs ON their
times (she drops with each, late, and swings after it), between them at an even speed, her arms crossed (`crossed`).
Let go, she falls into her seat (a ballistic arc from how the haul left her) and lands ON the knot, which ties every
thread on her again as the control swings in over her head. With `loop` every moment plays `loop` s earlier (and
later) too, so the pre-roll replays the clip's end and its last frame goes on into its first.

Frames as mkmmd.core.marionette_world's: her LEVEL frame (her rest frame), her BODY frame (her root as it moves). The
plan is what world_plan returns (her root, her channels, her hands, the bars, the grab) and `free`: the phases, the
moments, how far each thread is gathered into her fist, the scissors' hold and blades, the hand's point and the tails'
runs, for the build to draw the threads (mkmmd/blender/build/marionette.py)."""
import math

import numpy as np

from . import marionette as MR
from . import marionette_world as MW
from . import perform as PF

SEATED, FREE, HAUL, FALL = 0, 1, 2, 3
PHASES = ("seated", "free", "haul", "fall")
GRAVITY = MR.GRAVITY
HANDS = (0.22, 0.55)        # her hands after their keys: a little late, a little past (nothing arrives at once)
FEET = (0.42, 0.30)         # her legs trailing when nothing holds them
FLING = (0.2, 0.4)          # her legs flung up at a jolt: quick, and quickly back
CLOSE = (0.15, 0.4)         # s: her knees come together after a let-go (the seat's `close`), and part again
HAUL_SPRING = (0.55, 0.28)  # her hips after the hand's pulls: late, a swing past and back
SWAY = (1.6, 0.18)          # her sideways swing under a hauling hand (a long pendulum)
SWING_IN = (0.09, 0.36)     # the control swinging in at a knot: its swing dies in this many s, with this period
SWING_FROM = (-0.22, 0.25, -0.28)   # m, her frame: where it swings in from (her right, behind, below: slack)
BOUNCE = (0.035, 0.1, 0.26)  # m, s, s: landing in her seat, the hips go into it and bounce: depth, decay, period
CUT_ABOVE = 0.08            # m: the bundle is cut this far above her fist (just over it, the scissors' hand beside and
#                             a little above the fist, both hands in plain view)
APPROACH = 1.2              # s before a haul's catch the hand comes up under her with her tails' ends (unseen)
SLACK = 0.12                # m of slack in her tails the hand takes up just before the catch
DROP_SPIN = 200.0           # deg/s: the dropped control tumbles
LEG_TOSS = (0.0, 0.2, 9.0)  # m/s, her frame: the leg bar's hand recoils at the snip, up out of the picture
LEG_HOME = (0.0, -0.45, 0.5)   # m from her crown, her frame: the leg bar over her knees' threads, above the picture
SHOWN_AFTER_SNIP = 0.6      # s the control, whipped up at the snip, is still drawn (it is out of the picture by then)
TAILS_SHOWN = 1.8           # s a snip's tails are drawn after it (her take is over by then: she is out of the picture)
HAND_TO_FIST = 0.07         # m from her wrist to the middle of her fist (the bundle runs through it)
FIST_REACH = 0.9            # the bundle is held as high as her arm reaches with its elbow still bent: this share of it
SCISSORS_REACH = 0.135      # m from her wrist to the scissors' cut point as she holds them, along her forearm
SCISSORS_UP = 0.8           # of `scissors.dur`: her other hand is up beside her head with the scissors (its `raise` key
#                             at 0.7, a little late); its own thread joins the bundle from then
CATCH_TURN = 70.0           # degrees off the lens a hovering turn has her when a haul's hand catches her tails
SHRUG = 14.0                # degrees a shoulder rises as its hand goes up over it (an arm raised high lifts it)
ROLL = [[0.0, 0.0], [0.5, -3.0], [1.3, -8.0], [2.6, -4.0], [4.0, 0.0]]   # deg her body rolls as she rises (- her left
#                             side up), from the rise's start
TRAIL = {"L": ((0.03, 0.25, -1.0), 0.92), "R": ((-0.02, 0.12, -1.0), 0.95)}   # her legs hanging free: the way from each
#                             hip to its ankle (her frame: the feet a little behind, so the knees bend forward) and the
#                             share of the leg's length it spans (soft knees, the left one more)
OPEN = {True: (0.45, -0.15, 0.80), False: (0.62, -0.20, 0.62)}   # the rise's soft Y: each wrist from its shoulder in
#                             her arm's reach (out, forward -, up; 0.9 of it: the elbows bent 40 to 50 degrees), her
#                             fist's hand (True) the higher
EASED = {True: (0.52, -0.10, 0.74), False: (0.68, -0.15, 0.55)}  # and drifting a little lower as she floats on


def _keys_track(ts, keys):
    """(n, k) values eased from key to key [(t, (k,))] (a smoothstep between neighbours), held outside them."""
    ts = np.asarray(ts, float)
    keys = sorted(keys, key=lambda kv: kv[0])
    T = np.array([k[0] for k in keys])
    V = np.array([np.asarray(k[1], float) for k in keys])
    if len(T) == 1:
        return np.repeat(V[:1], len(ts), 0)
    i = np.clip(np.searchsorted(T, ts, side="right"), 1, len(T) - 1)
    a, b = T[i - 1], T[i]
    w = PF.smooth((ts - a) / np.maximum(b - a, 1e-6))
    w = np.where(ts <= T[0], 0.0, np.where(ts >= T[-1], 1.0, w))
    return V[i - 1] * (1.0 - w)[:, None] + V[i] * w[:, None]


def moments(P, lo=None, hi=None):
    """{kind: sorted times} of the free keys' moments, with their replays `loop` s earlier and later (the pre-roll
    replays the end; the end is followed by the start): knot, snip, let (her hand opening after the snip), haul, to (the
    haul's let-go), gather, scissors."""
    W = P["world"]
    L = W.get("loop")

    def rep(t):
        return [t - L, t, t + L] if L else [t]

    out = {k: [] for k in ("knot", "snip", "let", "haul", "to", "gather", "scissors")}
    if W.get("knot"):
        out["knot"] = rep(W["knot"]["t"])
    if W.get("snip"):
        out["snip"] = rep(W["snip"]["t"])
        out["let"] = rep(W["snip"]["t"] + W["snip"]["let"])
    if W.get("haul"):
        out["haul"], out["to"] = rep(W["haul"]["t"]), rep(W["haul"]["to"])
    for k in ("gather", "scissors"):
        if W.get(k):
            out[k] = rep(W[k]["t"])
    return out


def phases(ts, P):
    """(phase (n,) of SEATED, FREE, HAUL, FALL, the moment it began (n,), seconds since (n,)): seated from a knot (and
    before any moment: at her place), free from a snip, hauled from a haul's catch, falling from its let-go."""
    ts = np.asarray(ts, float)
    M = moments(P, float(ts[0]), float(ts[-1]))
    marks = sorted([(t, SEATED) for t in M["knot"]] + [(t, FREE) for t in M["snip"]] + [(t, HAUL) for t in M["haul"]]
                   + [(t, FALL) for t in M["to"]])
    if not marks:
        return np.full(len(ts), SEATED), np.full(len(ts), -math.inf), np.full(len(ts), math.inf)
    mt = np.array([m[0] for m in marks])
    mk = np.array([m[1] for m in marks])
    i = np.searchsorted(mt, ts, side="right") - 1
    ph = np.where(i >= 0, mk[np.maximum(i, 0)], SEATED)
    t0 = np.where(i >= 0, mt[np.maximum(i, 0)], -math.inf)
    return ph, t0, ts - t0


def seated_ankles(rig, seat, H_seat, knees=0.0):
    """{side: (3,)} her ankles' goals in her body frame sitting upright: each thigh forward from its hip joint (turned
    `knees` degrees toward her left, 8 degrees down), the shin hanging from the knee, the foot on the floor if it
    reaches it (then the shin slants forward)."""
    pivot = np.asarray(rig["pivot"], float)
    floor_b = float(seat["floor"]) - (float(H_seat[2]) - float(pivot[2]))    # the floor in her body frame
    k, dn = math.radians(knees), math.radians(8.0)
    d = np.array([math.sin(k) * math.cos(dn), -math.cos(k) * math.cos(dn), -math.sin(dn)])
    out = {}
    for s in MR.SIDES:
        J = np.asarray(rig["hips"][s], float)
        T, S = float(rig["legs"][s]["thigh"]), float(rig["legs"][s]["shin"])
        K = J + T * d
        A = K + S * MW._unit(np.array([0.0, 0.10, -1.0]))
        low = floor_b + float(np.asarray(rig["ankles"][s], float)[2])
        if A[2] < low:
            A[2] = low
            A[1] = K[1] - math.sqrt(max(S * S - (K[2] - A[2]) ** 2, 0.0))
        out[s] = A
    return out


def shoulder(rig, side, a):
    """(n, 3) the joint of her arm (her frame) with its shoulder raised `a` (n,) rad: turned up about the root of her
    shoulder bone (`clavicles`), about her front-to-back axis."""
    S = np.asarray(rig["shoulders"][side], float)
    C = np.asarray(rig["clavicles"][side], float)
    sg = 1.0 if side == "L" else -1.0
    return C + np.einsum("nij,j->ni", MW.rot_y(-sg * np.atleast_1d(np.asarray(a, float))), S - C)


def bundle_point(rig, at, side_g, side_s):
    """(bundle, cut point, the scissors arm's elbow) (3,) in her body frame: where her fist holds the gathered threads
    (her crown + `at`: over her head, in front of her ears), lowered a centimetre at a time until the fist reaches it
    (HAND_TO_FIST past its wrist) and her scissors reach the cut point CUT_ABOVE over it (SCISSORS_REACH past the other
    wrist), each arm with its elbow still bent (FIST_REACH of the arm, its shoulder raised: SHRUG); and the other arm's
    elbow out at its side and up, so that arm comes over the top of her head, not across her face."""
    crown = np.asarray(rig["crown"], float)
    Sg, Ss_up = (shoulder(rig, s, [math.radians(SHRUG)])[0] for s in (side_g, side_s))
    Ss = np.asarray(rig["shoulders"][side_s], float)
    rg, rs = (float(rig["arms"][s]["upper"]) + float(rig["arms"][s]["fore"]) for s in (side_g, side_s))
    up_s = float(rig["arms"][side_s]["upper"])
    sgs = 1.0 if side_s == "L" else -1.0
    E = Ss + up_s * MW._unit(np.array([sgs * 0.8, -0.35, 0.5]))
    B = crown + np.asarray(at, float)
    cut = np.array([0.0, 0.0, CUT_ABOVE])
    for _ in range(40):
        if (np.linalg.norm(B - Sg) <= FIST_REACH * rg + HAND_TO_FIST
                and np.linalg.norm(B + cut - Ss_up) <= FIST_REACH * rs + SCISSORS_REACH):
            break
        B = B - np.array([0.0, 0.0, 0.01])
    return B, B + cut, E


def _yaw_of(facing):
    f = np.asarray(facing, float)
    return math.atan2(f[0], -f[1])


def _nearest(a, target):
    """target + 2 pi k nearest to a."""
    return target + 2.0 * math.pi * round((a - target) / (2.0 * math.pi))


def _windows(wins, L):
    out = []
    for a, b in wins or []:
        out += [(a, b)] + ([(a - L, b - L), (a + L, b + L)] if L else [])
    return out


def hover_height(P):
    """m her hips rise over her seat before she hovers: where the haul catches her (planned back from the seat: the
    fall into it, the haul's pulls), else the ascend's `stop` (3 m)."""
    W = P["world"]
    haul, asc = W.get("haul"), W.get("ascend")
    if haul:
        Tf = haul["fall"]
        z_let = Tf * haul["speed"] + 0.5 * GRAVITY * Tf * Tf
        z_haul = (haul["jolt"] + haul["speed"] * max(haul["to"] - haul["t"] - 0.15, 0.0)
                  + haul["drop"] * len(haul["tugs"]))
        return z_let + z_haul
    return asc["stop"] if asc and asc["stop"] else 3.0


def free_plan(ts, P, cross, rig, aims):
    """The plan (see the module's docstring) over `ts` from the normalized table P (its `world`: seat, gather, scissors,
    snip, ascend, haul, crossed, knot, loop), the cross's own plan (`cross_plan`: its sway, turns and `raise` move the
    control and, through mkmmd.core.marionette.body_plan, her chest and head while she sits on her threads), `rig` (as
    world_plan's) and `aims`: {"seat": {hip (3,), facing (3,), floor}, "scissors": {grip (3,)} or None} in her level
    frame (the seat's hip point, the way it faces, the floor's height; where the scissors' grip lies)."""
    ts = np.asarray(ts, float)
    n = len(ts)
    P = snapped(P, ts)                                            # the replay lands on the end's frames
    W = P["world"]
    lag = P["lag"]
    pivot, crown = np.asarray(rig["pivot"], float), np.asarray(rig["crown"], float)
    hipmid = 0.5 * (np.asarray(rig["hips"]["L"], float) + np.asarray(rig["hips"]["R"], float))
    seat = aims["seat"]
    yaw_seat = _yaw_of(seat["facing"])
    Ry = MW.rot_z(np.array([yaw_seat]))[0]
    H_seat = np.asarray(seat["hip"], float) + Ry @ (pivot - hipmid)
    ph, t_mark, since = phases(ts, P)
    M = moments(P, float(ts[0]), float(ts[-1]))
    haul, asc = W.get("haul"), W.get("ascend")
    Z = hover_height(P)
    yaw_face = yaw_seat + math.radians(haul["face"]) if haul else yaw_seat
    G_rel = MW.rot_z(np.array([yaw_face]))[0] @ np.asarray(haul["at"], float) if haul else np.zeros(3)
    c_acc = c_T1 = None
    if asc:
        c_acc = asc["height"] / max(asc["by"] - asc["t"], 0.2) ** 2
        c_T1 = math.sqrt(max(Z, 0.05) / (2.0 * c_acc))
    # ---- her hips, attitude and turn, phase by phase in time order (each from where the one before left her)
    H = np.zeros((n, 3))
    theta = np.zeros(n)
    roll = np.zeros(n)
    yaw = np.full(n, yaw_seat)
    G = np.zeros((n, 3))
    pinned = np.zeros(n, bool)
    land_v = {}
    starts = np.r_[0, np.nonzero(np.diff(np.where(np.isfinite(t_mark), t_mark, -1e18)) != 0)[0] + 1]
    ends = np.r_[starts[1:], n]
    prevH, prevV, prevYaw = None, np.zeros(3), yaw_seat
    for a, b in zip(starts, ends):
        sl = slice(a, b)
        tau = since[sl]
        p = int(ph[a])
        if p == SEATED:
            Hs = np.tile(H_seat, (b - a, 1))
            v = float(-prevV[2]) if prevH is not None else 0.0
            depth = min(BOUNCE[0], 0.005 * max(v, 0.0))
            if depth > 0.0:                                       # she lands: into the seat and a bounce
                land_v[f"{float(t_mark[a]):.3f}"] = round(v, 2)
                tt = np.maximum(tau, 0.0)
                Hs[:, 2] -= depth / 0.57 * np.exp(-tt / BOUNCE[1]) * np.sin(2.0 * math.pi * tt / BOUNCE[2])
            H[sl] = Hs
            yaw[sl] = _nearest(prevYaw, yaw_seat)
        elif p == FREE:
            Hs = np.tile(H_seat, (b - a, 1))
            if asc:
                u = np.maximum(tau - (asc["t"] - W["snip"]["t"]), 0.0)
                z = np.where(u <= c_T1, c_acc * u * u, np.where(u <= 2.0 * c_T1, Z - c_acc * (2.0 * c_T1 - u) ** 2, Z))
                hov = u - 2.0 * c_T1
                bob = 0.03 * np.sin(2.0 * math.pi * hov / 3.2) * PF.smooth(hov / 1.5)
                Hs[:, 2] += z + bob
                theta[sl] = np.radians(PF.eased_keys(u, [[0.0, 0.0], [0.35, -12.0], [1.0, -4.0], [2.2, 0.0]]))
                roll[sl] = np.radians(PF.eased_keys(u, ROLL))
                rate = np.radians(asc["turn"])
                nxt = [h for h in M["haul"] if h > t_mark[a]]
                if nxt:                                           # turning so the catch finds her CATCH_TURN off the lens
                    T_h = nxt[0] - (t_mark[a] + (asc["t"] - W["snip"]["t"]) + 2.0 * c_T1)
                    if T_h > 1.0:
                        need = (yaw_face - math.radians(CATCH_TURN)) - _nearest(prevYaw, yaw_seat)
                        k = round((rate * T_h - need) / (2.0 * math.pi))
                        rate = (need + 2.0 * math.pi * k) / T_h
                yaw[sl] = _nearest(prevYaw, yaw_seat) + rate * np.maximum(hov, 0.0) * PF.smooth(hov)
            else:
                yaw[sl] = prevYaw
            H[sl] = Hs
        elif p == HAUL:
            entry = prevH if prevH is not None else H_seat + np.array([0.0, 0.0, Z])
            pull = _haul_pull(tau, haul)
            sw = MR.follow(ts[sl], pull[:, :2], SWAY, lag)
            hz = MR.follow(ts[sl], pull[:, 2], HAUL_SPRING, lag)
            H[sl] = entry + np.column_stack([sw, hz])
            G[sl] = entry + G_rel + pull
            pinned[sl] = True
            yt = _nearest(prevYaw + 0.6, yaw_face)                # on round the way she turns, to face the lens
            yaw[sl] = prevYaw + (yt - prevYaw) * PF.smooth(tau / 0.9)
            theta[sl] = np.radians(PF.eased_keys(tau, [[0.0, 0.0], [0.08, -6.0], [0.5, 5.0], [0.9, 4.0]]))
        else:                                                     # FALL: into her seat, landing ON the next knot
            T = haul["fall"] if haul else 0.5
            H0 = prevH if prevH is not None else H_seat + np.array([0.0, 0.0, 2.0])
            te = ts[a - 1] if a > 0 else ts[a]                   # from her last frame before it, at its speed
            Te = float(t_mark[a]) + T - te
            acc = 2.0 * (H_seat - H0 - prevV * Te) / (Te * Te)
            u = ts[sl] - te
            H[sl] = H0 + np.outer(u, prevV) + 0.5 * np.outer(u * u, acc)
            yaw[sl] = prevYaw + (_nearest(prevYaw, yaw_seat) - prevYaw) * PF.smooth(tau / T)
            theta[sl] = np.radians(PF.eased_keys(tau, [[0.0, 4.0], [0.3, -5.0], [T, -3.0]]))
        prevH = H[b - 1].copy()
        prevV = (H[b - 1] - H[b - 2]) / max(ts[b - 1] - ts[b - 2], 1e-6) if b - a > 1 else np.zeros(3)
        prevYaw = float(yaw[b - 1])
    for th in M["haul"]:                                          # the hand comes up under her with the ends of her
        i0 = int(np.searchsorted(ts, th))                         # tails before each catch (unseen), the slack taken
        if i0 >= n:                                               # up just before it
            continue
        k = (ts >= th - APPROACH) & (ts < th)
        u = (ts[k] - (th - APPROACH)) / APPROACH
        rise = -0.9 * (1.0 - PF.smooth(u / 0.8)) + SLACK * PF.smooth(u / 0.8) * (1.0 - PF.smooth((ts[k] - th + 0.06) / 0.06))
        G[k] = H[i0] + G_rel + np.outer(rise, [0.0, 0.0, 1.0])
        pinned[k] = True
    R = np.einsum("nij,njk,nkl->nil", MW.rot_z(yaw), MW.rot_x(theta), MW.rot_y(roll))
    pos = H - np.einsum("nij,j->ni", R, pivot)
    # ---- her body: the control's channels while she sits on her threads (no snap: the threads are on her from a
    #      swing-in, the same at the clip's end and in its replay), and what each phase does to her
    held_any = np.max(np.stack([MR.held(ts, P, p) for p in MR.PARTS]), axis=0) * (ph == SEATED)
    lift = PF.eased_keys(ts, P["raise"])
    for t, h in P["hops"]:
        lift = lift + h * MR.pulse(ts, t - MR.HOP[0] - 0.06, *MR.HOP)
    ctl_in = {"lift": lift, "sway": cross["sway"], "tilt": cross["tilt"], "rock": cross["rock"], "turn": cross["turn"],
              "knee": {s: np.zeros(n) for s in MR.SIDES}}
    body = MR.body_plan(ts, dict(P, **{"from": -1e9}), ctl_in, rig["legs"], stand=np.zeros(n))
    on = np.clip(MR.follow(ts, held_any, MW.REGIME, lag), 0.0, 1.0)
    for key in ("lean", "tilt", "turn", "head_tilt", "head_nod", "head_turn"):
        body[key] = body[key] * on
    for k in M["knot"]:                                           # she lands: her chest and head give and come back
        body["lean"] = body["lean"] + math.radians(7.0) * MR.follow(ts, MR.pulse(ts, k, 0.05, 0.05, 0.35), MR.GIVE, lag)
        body["head_nod"] = body["head_nod"] + math.radians(13.0) * MR.follow(ts, MR.pulse(ts, k + 0.02, 0.06, 0.04, 0.3),
                                                                             MR.DROP, lag)
    for th in M["haul"]:                                          # the jolt flings her head back
        body["head_nod"] = body["head_nod"] - math.radians(15.0) * MR.follow(ts, MR.pulse(ts, th, 0.05, 0.03, 0.4),
                                                                             MR.DROP, lag)
        body["lean"] = body["lean"] - math.radians(7.0) * MR.follow(ts, MR.pulse(ts, th, 0.05, 0.03, 0.35), MR.GIVE, lag)
    for a_, b_ in _windows(W.get("crossed"), W.get("loop")):     # sulking: chin down, her head on one side
        w = MW.window(ts, a_, b_, 0.4)
        body["lean"] = body["lean"] + math.radians(4.0) * w
        body["head_nod"] = body["head_nod"] + math.radians(6.0) * w
        body["head_tilt"] = body["head_tilt"] + math.radians(5.0) * w
    for key in ("rise", "side", "behind", "dip", "cut_dip", "guitar"):
        body[key] = np.zeros(n)
    # ---- her legs: sitting, hanging free with soft knees, flung up for a moment at the jolt, reaching for the seat as
    #      she falls into it; the seat's `close` draws her knees together from a let-go until they part after the knot
    sit_a = seated_ankles(rig, seat, H_seat, W["seat"]["knees"])
    seated_w = _seated_weight(ts, ph, since, M)
    closed = W["seat"]["close"] * _closed_weight(ts, M, W["seat"]["part"])
    feet, point = {}, {}
    for s in MR.SIDES:
        J = np.asarray(rig["hips"][s], float)
        ank = np.asarray(rig["ankles"][s], float)
        L = float(rig["legs"][s]["thigh"]) + float(rig["legs"][s]["shin"])
        way, share = TRAIL[s]
        tgt = np.tile(J + share * L * MW._unit(np.array(way)) - ank, (n, 1))
        fling = np.zeros((n, 3))
        for th in M["haul"]:                                      # jerked down by her threads: her legs fly up, back
            fling = fling + np.outer(MR.pulse(ts, th, 0.04, 0.02, 0.16), [0.0, -0.08, 0.10])
        free_leg = MR.follow(ts, tgt, FEET, lag) + MR.follow(ts, fling, FLING, lag)
        feet[s] = np.tile(sit_a[s] - ank, (n, 1)) * seated_w[:, None] + free_leg * (1.0 - seated_w)[:, None]
        if closed.any():
            # each knee toward the other (x is her left): a seated knee goes as the ankle under it, a hanging leg
            # turns about its hip, so its ankle goes the farther
            reach = share * L / float(rig["legs"][s]["thigh"])
            feet[s][:, 0] -= (1.0 if s == "L" else -1.0) * closed * (seated_w + (1.0 - seated_w) * reach)
        point[s] = 0.25 * seated_w + 0.65 * (1.0 - seated_w)
    body["feet"], body["point"] = feet, point
    body["knee"] = {s: np.zeros(n) for s in MR.SIDES}
    # ---- the control, her hands, the threads gathered into her fist, the scissors
    (cpos, crot), (lpos, lrot), shown = _control(ts, P, ctl_in, rig, H_seat, Ry, ph, M, float(seat["floor"]))
    hands, sci = _hands(ts, P, rig, R, pos, H_seat, Ry, M, aims.get("scissors"))
    grab = np.where(pinned[:, None], G, pos + np.einsum("nij,j->ni", R, crown))
    body["report"].update({"phases": {PHASES[k]: round(float((ph == k).mean()), 3) for k in range(4)},
                           "hover_m": round(float(Z), 2), "landings_mps": land_v,
                           "times": {k: [round(t, 3) for t in v] for k, v in M.items() if v}})
    runs = tail_runs(M)
    free = {"phase": ph, "since": since, "moments": M, "gathered": _gathered(ts, P, M), "scissors": sci, "grab": G,
            "pinned": pinned, "runs": runs, "seated": seated_w, "H_seat": H_seat, "cut_above": CUT_ABOVE,
            "side": W["gather"]["side"] if W.get("gather") else "L"}
    return {"root": {"pos": pos, "rot": R}, **body, "hands": hands, "cross": {"pos": cpos, "rot": crot, "shown": shown},
            "leg": {"pos": lpos, "rot": lrot, "shown": shown}, "grab": grab, "rest": np.tile([0.0, 0.0, -60.0], (n, 1)),
            "resting": np.zeros(n, bool), "theta": theta, "aloft": (ph != SEATED).astype(float), "hips": H,
            "regimes": {"pond": ph == SEATED}, "times": {}, "rig": rig, "free": free}


def tail_runs(M):
    """[(kind, from, to)]: the clip seconds her tails are run and drawn over: from each snip for TAILS_SHOWN s (to the
    hand coming up with their ends before a catch, or a knot, if sooner); from before each catch (the hand bringing their
    ends up under her, unseen) to the knot after its let-go."""
    out = []
    for s in M["snip"]:
        nxt = [h - APPROACH - 0.3 for h in M["haul"] if h > s] + [k for k in M["knot"] if k > s]
        out.append(("snip", s, min([s + TAILS_SHOWN] + nxt)))
    for h_, to_ in zip(M["haul"], M["to"]):
        nxt = [k for k in M["knot"] if k > to_]
        out.append(("haul", h_ - APPROACH - 0.3, nxt[0] if nxt else to_ + 1.0))
    return out


def _haul_pull(tau, haul):
    """(n, 3) the hauling hand's pull from where it caught her, `tau` s after the catch at `t0`: the jolt ON the catch,
    then hand over hand: an even pull at `speed` and a tug ON each of `tugs` (set off early, so she drops on it), the
    hands taking turns a little to each side."""
    tau = np.asarray(tau, float)
    z = -haul["jolt"] * PF.smooth((tau + 0.02) / 0.09)
    z = z - haul["speed"] * np.maximum(tau - 0.15, 0.0) * PF.smooth((tau - 0.15) / 0.5)
    x = np.zeros(len(tau))
    for k, tg in enumerate(haul["tugs"]):
        u = tau - (tg - haul["t"])
        z = z - haul["drop"] * PF.smooth((u + 0.10) / 0.14)
        x = x + 0.05 * (1.0 if k % 2 == 0 else -1.0) * (PF.smooth((u + 0.10) / 0.14) - PF.smooth((u - 0.3) / 0.5))
    return np.column_stack([x, np.zeros(len(tau)), z])


def _seated_weight(ts, ph, since, M):
    """(n,) how far she sits (1) or hangs free (0): her legs come up and trail as she rises, reach for the seat over the
    last 0.35 s of a fall into it."""
    w = (ph == SEATED).astype(float)
    w = np.where((ph == FREE) & (since < 0.9), 1.0 - PF.smooth(since / 0.6), w)
    for k in M["knot"]:
        near = (ts >= k - 0.35) & (ts < k) & (ph == FALL)
        w = np.where(near, PF.smooth((ts - (k - 0.35)) / 0.33), w)
    return w


def _closed_weight(ts, M, part):
    """(n,) how far her knees are drawn together (the seat's `close`): from each let-go of a haul (in over CLOSE[0] s) as
    she falls into her seat and lands ON the knot, until `part` s after the knot (out over CLOSE[1] s, or over `part`
    when that is shorter: 0 parts them on the knot's own frame, a landing on a cut)."""
    w = np.zeros(len(ts))
    out = min(CLOSE[1], part)
    for to in M["to"]:
        knots = [k for k in M["knot"] if k > to]
        if knots:
            end = knots[0] + part
            gone = PF.smooth((ts - end) / out) if out > 0.0 else (ts > end - 1e-6).astype(float)
            w = np.maximum(w, PF.smooth((ts - to) / CLOSE[0]) * (1.0 - gone))
    return w


def _gathered(ts, P, M, wait=True):
    """{part: (n,) 0..1}: how far each thread is gathered into her fist: from the gather's start her fist sweeps them in
    (her own wrist's first, the other hand's last), until the snip cuts them. The other hand's thread waits for that hand
    to come up beside her head with the scissors (SCISSORS_UP of their `dur`): it joins the bundle from the side, never
    from the table across her face. `wait` False: the sweep alone, each thread taken when her fist comes to it."""
    W = P["world"]
    out = {p: np.zeros(len(ts)) for p in MR.PARTS}
    g, sc = W.get("gather"), W.get("scissors")
    if not g:
        return out
    other = "R" if g["side"] == "L" else "L"
    order = {f"wrist_{g['side']}": 0, "knee_L": 1, "knee_R": 2, "back": 3, "head": 4, f"wrist_{other}": 5}
    for t in M["gather"]:
        prev = max([s for s in M["snip"] if s <= t], default=-math.inf)
        nxt = [s for s in M["snip"] if s > t]
        end = nxt[0] if nxt else math.inf
        up = [s + SCISSORS_UP * sc["dur"] for s in M["scissors"] if prev < s < end] if (sc and wait) else []
        for p, k in order.items():
            a = t + g["dur"] * (0.3 + 0.12 * k)
            if up and p == f"wrist_{other}":
                a = max(a, min(up))
            out[p] = np.maximum(out[p], PF.smooth((ts - a) / (0.18 * g["dur"])) * (ts < end))
    return out


def _control(ts, P, ctl, rig, H_seat, Ry, ph, M, floor=0.0):
    """((pos (n, 3), rot (n, 3, 3)) of the cross and of the leg bar in her level frame, shown (n,)): held over her head
    while she sits on her threads (`height` over her crown, its own sway, lift and turns), swinging in at a knot (a
    pendulum from SWING_FROM: slack, then taut), let down a little while her fist gathers the threads, dropped at the
    snip (the cross falling past her, tumbling, behind her and to her right; the leg bar's hand recoiling up over her
    head), out of the picture SHOWN_AFTER_SNIP s later until the next knot."""
    n = len(ts)
    W = P["world"]
    pivot, crown = np.asarray(rig["pivot"], float), np.asarray(rig["crown"], float)
    a_head = np.asarray(rig["anchors"]["head"], float)
    crown_l = H_seat + Ry @ (crown - pivot)
    home = crown_l + np.array([0.0, 0.0, P["height"]]) - Ry @ a_head
    off = np.zeros((n, 3))                                        # her frame
    off[:, 0] = ctl["sway"]
    off[:, 2] = ctl["lift"]
    g = W.get("gather")
    if g:
        for t in M["gather"]:
            nxt = [s for s in M["snip"] if s > t]
            off[:, 2] -= 0.08 * PF.smooth((ts - t) / g["dur"]) * (ts < (nxt[0] if nxt else math.inf))
    for t in M["knot"]:
        tau = np.maximum(ts - t, 0.0)
        e = np.exp(-tau / SWING_IN[0]) * np.cos(2.0 * math.pi * tau / SWING_IN[1]) * (ts >= t)
        off = off + np.outer(e, SWING_FROM)
    pos = home + off @ Ry.T
    rot = np.einsum("ij,njk,nkl,nlm->nim", Ry, MW.rot_z(ctl["turn"]), MW.rot_y(ctl["tilt"]), MW.rot_x(ctl["rock"]))
    lpos = crown_l + (np.asarray(LEG_HOME) + off * np.array([0.6, 1.0, 1.0])) @ Ry.T
    lrot = np.tile(Ry, (n, 1, 1))
    shown = ph == SEATED
    sn = W.get("snip")
    for s in M["snip"]:                                           # dropped
        i0 = int(np.clip(np.searchsorted(ts, s), 1, n - 1))
        k = (ts >= s) & (ts < s + SHOWN_AFTER_SNIP)
        if not k.any():
            continue
        tau = ts[k] - s
        P0, R0, L0 = pos[i0 - 1].copy(), rot[i0 - 1].copy(), lpos[i0 - 1].copy()
        fall = np.outer(0.5 * tau * tau, (0.0, 0.0, -GRAVITY))
        pos[k] = P0 + np.outer(tau, Ry @ np.asarray(sn["away"], float)) + fall
        rot[k] = np.einsum("nij,jk->nik", MW.rot_axis((0.35, 1.0, 0.25), math.radians(DROP_SPIN) * tau), R0)
        lpos[k] = L0 + np.outer(tau, Ry @ np.asarray(LEG_TOSS)) + fall
        lrot[k] = np.einsum("nij,jk->nik", MW.rot_axis((1.0, 0.3, 0.2), 4.0 * tau), Ry)
        for X, Q in ((pos, rot), (lpos, lrot)):                   # what reaches the floor lies there
            idx = np.nonzero(k)[0]
            low = idx[X[idx, 2] < floor + 0.012]
            if len(low):
                X[low[0]:idx[-1] + 1] = X[low[0]] * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, floor + 0.012])
                Q[low[0]:idx[-1] + 1] = MW._flatten(Q[low[0]])
        shown = shown | k
    return (pos, rot), (lpos, lrot), shown


def _hands(ts, P, rig, R, pos, H_seat, Ry, M, sc_aim):
    """({side: {wrist, dir, palm, pole, body, elbow, shrug, shoulder}} as mkmmd.core.marionette_world's hands, the
    scissors' hold): keys of where each hand is through the moments, in her body frame: on her lap while she sits on her
    threads; her fist up her side to the bundle over her head; her other hand to the scissors on the table, taking them,
    up her side and over to the bundle with its cut point between the blades; after the snip's let-go both arms opening
    up and out in a soft Y (her fist's hand higher), elbows bent, palms up, the scissors carried out to her side and
    dropped there (`snip.drop`); flung up at the jolt and as she falls; crossed over her chest in `crossed`. Eased from
    key to key and a little late (HANDS), each hand going on along its forearm turned as each pose asks
    (mkmmd.core.armtwist), its shoulder raised as the hand goes up over it (SHRUG: `shrug` (n,) rad, `shoulder` (n, 3)
    the arm's joint so raised). The scissors' hold: {held (n,) bool (from her taking them to her dropping them), open
    (n,) 0..1 (opening as she brings them up, shut ON the snip), side, prop}."""
    n = len(ts)
    W = P["world"]
    g, sc, sn = W.get("gather"), W.get("scissors"), W.get("snip")
    pivot = np.asarray(rig["pivot"], float)
    sh = {s: np.asarray(rig["shoulders"][s], float) for s in MR.SIDES}
    hipz = float(np.mean([np.asarray(rig["hips"][s], float)[2] for s in MR.SIDES]))
    seat_pos = H_seat - Ry @ pivot
    reach = {s: float(rig["arms"][s]["upper"]) + float(rig["arms"][s]["fore"]) for s in MR.SIDES}
    side_g = g["side"] if g else "L"
    side_s = sc["side"] if sc else ("R" if side_g == "L" else "L")
    bundle, cut, elbow_s = bundle_point(rig, g["at"] if g else (0.14, -0.04, 0.14), side_g, side_s)

    def aimed(s, P_, d, pole):
        """The wrist `d` m short of P_ along the forearm the arm IK gives it (from her shoulder, raised, toward `pole`):
        her hand goes on along its forearm, so her fist closes round P_ (the bundle) or her scissors' cut point reaches
        it (the cut)."""
        a, b = float(rig["arms"][s]["upper"]), float(rig["arms"][s]["fore"])
        S_up = shoulder(rig, s, [math.radians(SHRUG)])
        W = P_ - d * MW._unit(P_ - S_up[0])
        for _ in range(12):
            W = P_ - d * MW._unit(P_ - MW.two_bone(S_up, W[None], pole[None], a, b)[0])
        return W

    def pose(s, name):
        """(wrist target, elbow pole, supination degrees) of a named pose, her body frame, the reach of her arm `r`."""
        sg = 1.0 if s == "L" else -1.0
        S, r = sh[s], reach[s]
        if name == "lap":
            return np.array([sg * 0.11, -0.19, hipz + 0.10]), S + r * np.array([sg * 1.0, 0.8, -0.7]), 0.0
        if name == "fist":
            pole = S + r * np.array([sg * 1.2, 0.3, -0.1])
            return aimed(s, bundle, HAND_TO_FIST, pole), pole, 30.0
        if name == "side_up":
            return S + r * np.array([sg * 0.75, -0.12, 0.35]), S + r * np.array([sg * 1.2, 0.4, -0.5]), 10.0
        if name == "table":
            if sc_aim is None:
                return S + r * np.array([sg * 0.6, -0.5, -0.6]), S + r * np.array([sg * 1.0, 0.6, -0.5]), -10.0
            p = Ry.T @ (np.asarray(sc_aim["grip"], float) - seat_pos) + np.array([0.0, 0.03, 0.07])
            return p, S + r * np.array([sg * 1.0, 0.6, -0.5]), -10.0
        if name == "raise":
            return S + r * np.array([sg * 0.55, 0.15, 0.6]), elbow_s + np.array([sg * 0.1, 0.1, 0.0]), 20.0
        if name == "snips":                                       # up over her head, the blades at the bundle
            pole = elbow_s + np.array([sg * 0.1, 0.1, 0.0])
            return aimed(s, cut, SCISSORS_REACH, pole), pole, 40.0
        if name == "open":                                        # delight: up and out in a soft Y, her fist's hand
            u = OPEN[s == side_g]                                 # higher, the elbows bent out and down, palms up
            return S + r * np.array([sg * u[0], u[1], u[2]]), S + r * np.array([sg * 1.1, 0.35, -0.3]), 70.0
        if name == "eased":                                       # floating so, the arms drifting a little
            u = EASED[s == side_g]
            return S + r * np.array([sg * u[0], u[1], u[2]]), S + r * np.array([sg * 1.1, 0.4, -0.4]), 60.0
        if name == "float":
            return S + r * np.array([sg * 0.62, -0.25, -0.55]), S + r * np.array([sg * 1.2, 0.5, -0.8]), 20.0
        if name == "flung":
            return S + r * np.array([sg * 0.55, -0.10, 0.80]), S + r * np.array([sg * 1.3, 0.0, -0.3]), 30.0
        if name == "falling":
            return S + r * np.array([sg * 0.62, -0.05, 0.70]), S + r * np.array([sg * 1.3, 0.1, -0.3]), 30.0
        if name == "crossed":                                     # forearms across her chest under her bust, the right
            z = float(S[2]) - 0.36 * r                            # one over, held a little out from her
            p = np.array([-0.07, -0.20, z]) if s == "L" else np.array([0.07, -0.235, z + 0.03])
            return p, S + r * np.array([sg * 1.0, 0.25, -0.9]), 75.0
        raise ValueError(name)
    keys = {s: [] for s in MR.SIDES}
    for s in MR.SIDES:
        K = keys[s]
        for k in M["knot"]:
            K += [(k - 0.25, "falling"), (k + 0.32, "lap")]
        if g and s == side_g:
            for t in M["gather"]:
                K += [(t, "lap"), (t + 0.45 * g["dur"], "side_up"), (t + g["dur"], "fist")]
        if sc and s == side_s:
            for t in M["scissors"]:
                d = sc["dur"]
                K += [(t, "lap"), (t + 0.3 * d, "table"), (t + 0.45 * d, "table"), (t + 0.7 * d, "raise"),
                      (t + d, "snips")]
        for t, lt in zip(M["snip"], M["let"]):
            K += [(lt, "fist" if (g and s == side_g) else "snips" if (sc and s == side_s) else "lap"), (lt + 0.45, "open"),
                  (lt + 1.1, "eased"), (t + 3.0, "float")]
        for th, to in zip(M["haul"], M["to"]):
            K += [(th - 0.4, "float"), (th + 0.08, "flung"), (th + 0.22, "flung"), (to + 0.15, "falling")]
        for a_, b_ in _windows(W.get("crossed"), W.get("loop")):
            K += [(a_ + 0.38, "crossed"), (b_, "crossed")]
        if not K:
            K.append((float(ts[0]), "lap"))
    out = {}
    for s in MR.SIDES:
        sg = 1.0 if s == "L" else -1.0
        S = sh[s]
        a, b = float(rig["arms"][s]["upper"]), float(rig["arms"][s]["fore"])
        vals = []
        for t, nm in keys[s]:
            p, q, sup = pose(s, nm)
            vals.append((t, np.concatenate([p, q, [sup]])))
        tr = MR.follow(ts, _keys_track(ts, vals), HANDS, P["lag"])
        target, pole, supin = tr[:, :3], tr[:, 3:6], tr[:, 6]
        lift = math.radians(SHRUG) * PF.smooth((target[:, 2] - S[2]) / (0.7 * (a + b)))
        Sn = shoulder(rig, s, lift)
        rel = target - Sn
        dist = np.clip(np.linalg.norm(rel, axis=1), 0.12, 0.97 * (a + b))
        wrist = MW._clear_torso(Sn + MW._unit(rel) * dist[:, None], rig, sg)
        E = MW.two_bone(Sn, wrist, pole, a, b)
        d, p = MW.hand_frame(rig["rest"], s, Sn, E, wrist, supin)
        out[s] = {"wrist": MW._to_level(R, pos, wrist), "dir": np.einsum("nij,nj->ni", R, d),
                  "palm": np.einsum("nij,nj->ni", R, p), "pole": MW._to_level(R, pos, pole), "body": wrist, "elbow": E,
                  "shrug": lift, "shoulder": Sn}
    return out, scissors_hold(ts, P, M)


def scissors_hold(ts, P, M):
    """{held (n,) bool, open (n,) 0..1, falling [(t, until)], side, prop}: the scissors in her hand from her taking them
    (part way through `scissors.dur`) until she drops them (`snip.drop` after the snip, out at her side), their blades
    opening as she brings them up and shut ON the snip; dropped, they fall for 1.2 s, then lie where they lay (her rise
    has taken her out of the picture)."""
    ts = np.asarray(ts, float)
    W = P["world"]
    g, sc, sn = W.get("gather"), W.get("scissors"), W.get("snip")
    side_g = g["side"] if g else "L"
    out = {"side": sc["side"] if sc else ("R" if side_g == "L" else "L"), "prop": sc["prop"] if sc else None,
           "held": np.zeros(len(ts), bool), "open": np.zeros(len(ts)), "falling": []}
    if not (sc and sn):
        return out
    d = sc["dur"]
    for t in M["scissors"]:
        nxt = [x for x in M["snip"] if x > t]
        if not nxt:
            continue
        out["held"] |= (ts >= t + 0.42 * d) & (ts < nxt[0] + sn["drop"])
        out["open"] = np.maximum(out["open"], PF.smooth((ts - (t + 0.55 * d)) / (0.4 * d)) * (ts < nxt[0]))
        out["falling"].append((nxt[0] + sn["drop"], nxt[0] + sn["drop"] + 1.2))
    return out


def snapped(P, ts):
    """P with its `loop` a whole number of the frames `ts` are on, so the replay of the end lands on the end's frames."""
    ts = np.asarray(ts, float)
    if not P["world"].get("loop") or len(ts) < 2:
        return P
    fps = 1.0 / float(ts[1] - ts[0])
    return dict(P, world=dict(P["world"], loop=round(P["world"]["loop"] * fps) / fps))
