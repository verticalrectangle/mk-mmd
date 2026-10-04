"""Where a braid goes (bpy-free, numpy only): the top bow beside the jaw, how the plait drapes over the shoulder and
hangs in front of the chest, and the frames along it.

`plan_side` builds one side: a guide curve (heading forward over the shoulder, then straight down) is bent as little as
possible so that it keeps `need` metres away from the body mesh, the head skin and the static colliders, and is returned
as a dense arc-length polyline (`Plan`) with frames. Arc coordinate sigma is 0 at the top bow knot, increases down the
braid, and is negative on the short stretch above the bow where the gather clumps arrive."""
from dataclasses import dataclass, field

import numpy as np

from .hair_braids_geo import collider_distance
from .hair_geo import arclen, frames, unit

STEP = 0.002                                          # sample spacing of the dense centreline (m)


def facing(az, sx):
    """Horizontal unit vector at azimuth az (rad) from the front (-Y) towards the side sx (+1 = her left, +X)."""
    return np.array([sx * np.sin(az), -np.cos(az), 0.0])


@dataclass
class Plan:
    """One braid's centreline: dense points P (m, 3) at arc coordinates sig (m,), frames T (down the braid), B (across,
    pointing away from the midline), N (the face of the plait, away from the body), and the bow positions."""
    sx: float
    P: np.ndarray
    sig: np.ndarray
    T: np.ndarray
    B: np.ndarray
    N: np.ndarray
    s_bow: tuple                                       # sigma of the top and bottom bow knots (0, Lb)
    s_end: float                                       # sigma of the tuft tip
    info: dict = field(default_factory=dict)

    def at(self, s):
        """Point and unit frame (P, T, B, N) at arc coordinates s (scalar or array); extrapolated along the ends."""
        s = np.asarray(s, float)
        out = []
        for A in (self.P, self.T, self.B, self.N):
            out.append(np.stack([np.interp(s, self.sig, A[:, k]) for k in range(3)], -1))
        p, t, b, n = out
        return p, unit(t), unit(b), unit(n)


# ---------------------------------------------------------------- references from the head and body parts
def head_refs(ctx, fit, cfg):
    """Numbers the layout hangs on, taken from the head fit and the body part: scale, chin and shoulder heights, ears,
    the bust line."""
    body = ctx.parts["body"]
    land = {k: np.asarray(v, float) for k, v in (body.info.get("landmarks") or {}).items()}
    land.update({k: np.asarray(v, float) for k, v in ctx.land.items()})
    scale = float((fit.top[2] - fit.chin[2]) / cfg["head_ref"])
    refs = {"scale": scale, "chin_z": float(fit.chin[2]), "top_z": float(fit.top[2]), "center": np.asarray(fit.center, float)}
    for side in ("L", "R"):
        e = fit.ears.get(side) if fit.ears else None
        if e is not None:
            refs["lobe" + side] = np.asarray(e["lobe"], float)
            refs["ear" + side] = np.asarray(fit.ear_centre(side), float)
        else:
            sg = 1.0 if side == "L" else -1.0
            refs["lobe" + side] = fit.center + np.array([sg * 0.09 * scale, 0.0, -0.03 * scale])
            refs["ear" + side] = fit.center + np.array([sg * 0.09 * scale, 0.0, 0.0])
        a = land.get("arm." + side)
        refs["arm" + side] = a if a is not None else np.array([(1.0 if side == "L" else -1.0) * 0.087, 0.006, 1.12])
    z_bust = None
    if "bust_front" in land:
        z_bust = float(land["bust_front"][2])
    elif "torso_profile" in body.info:
        tp = body.info["torso_profile"]
        z = np.asarray(tp["rows"], float)
        yf = np.asarray(tp["y_front"], float)
        sel = (z > refs["chin_z"] - 0.30 * scale) & (z < refs["chin_z"] - 0.05 * scale)
        if sel.any():
            z_bust = float(z[sel][np.argmin(yf[sel])])                 # the deepest chest cut below the shoulders
    if z_bust is None:
        z_bust = float(refs["chin_z"] - 0.14 * scale)
    refs["bust_z"] = z_bust
    return refs


# ---------------------------------------------------------------- constraint fields
def _grad_dir(fn, P, h=1e-3):
    """Unit gradient of a scalar field fn(P) -> (n,) by central differences (the direction of increasing distance)."""
    g = np.zeros_like(P)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[:, k] = fn(P + e) - fn(P - e)
    return unit(g)


def make_constraints(body_mesh, skin_mesh, cols, need_mesh, need_col):
    """[(required distance, fn(P) -> (signed distance (n,), push direction (n,3)))] for the body mesh, the head skin and
    the static colliders."""
    def mesh_field(tm):
        def fn(P):
            sd, q, n = tm.signed(P)
            away = np.where(sd[:, None] >= 0, unit(P - q), n)           # inside: along the face normal, out of the shell
            return sd, away
        return fn

    def col_field(P):
        d = np.min([collider_distance(P, rb) for rb in cols], axis=0)
        g = _grad_dir(lambda Q: np.min([collider_distance(Q, rb) for rb in cols], axis=0), P)
        return d, g

    out = [(need_mesh, mesh_field(body_mesh)), (need_mesh, mesh_field(skin_mesh))]
    if cols:
        out.append((need_col, col_field))
    return out


def _lift(Q, cons):
    """Per point the largest lift (vector) any constraint asks for, and the worst shortfall (m)."""
    push = np.zeros_like(Q)
    mag = np.zeros(len(Q))
    for need, fn in cons:
        sd, away = fn(Q)
        v = np.maximum(need - sd, 0.0)
        take = v > mag
        push[take] = away[take] * v[take, None]
        mag = np.maximum(mag, v)
    return push, float(mag.max())


def relax(P, cons, iters=240, ksm=9, tol=3e-4):
    """P displaced by the smooth field D that lifts it off every constraint: the lift accumulates in D, which is blurred
    each round so the braid stays a smooth curve (a few unblurred rounds at the end make the clearance exact)."""
    D = np.zeros_like(P)
    w = np.hanning(ksm + 2)[1:-1]
    w = w / w.sum()
    pad = ksm // 2
    for _ in range(iters):
        push, worst = _lift(P + D, cons)
        if worst < tol:
            break
        D = D + push
        Dp = np.concatenate([np.repeat(D[:1], pad, 0), D, np.repeat(D[-1:], pad, 0)], 0)
        D = np.stack([np.convolve(Dp[:, k], w, mode="valid") for k in range(3)], -1)
    for _ in range(12):
        push, worst = _lift(P + D, cons)
        if worst < 1e-5:
            break
        D = D + push
    return P + D


# ---------------------------------------------------------------- the guide
def guide_curve(b1, sx, cfg, scale, length):
    """Dense polyline from the bow knot b1 heading forward over the shoulder (slope alpha0 from the vertical, forward and
    a little outward) and relaxing to a hang of alpha_end: the shape a braid takes resting on the shoulder."""
    a0, a1 = np.radians(cfg["drape_deg"]), np.radians(cfg["hang_deg"])
    sc = cfg["drape_len"] * scale
    h = unit(np.array([sx * cfg["heading_out"], -1.0, 0.0]))
    n = int(length / STEP) + 1
    P = np.zeros((n, 3))
    P[0] = b1
    for i in range(1, n):
        s = (i - 1) * STEP
        a = a1 + (a0 - a1) * np.exp(-s / sc)
        P[i] = P[i - 1] + STEP * unit(h * np.sin(a) + np.array([0.0, 0.0, -np.cos(a)]))
    return P


def plan_side(cfg, refs, sx, cons_builder, rng):
    """Plan one side (sx = +1 left, -1 right). `cons_builder(P)` returns the constraint list for the points P."""
    side = "L" if sx > 0 else "R"
    sc = refs["scale"]
    lobe = refs["lobe" + side]
    b1 = np.array([lobe[0] + sx * cfg["bow_dx"] * sc, lobe[1] + cfg["bow_dy"] * sc, refs["chin_z"] - cfg["bow_drop"] * sc])
    b1 = b1 + rng.normal(0.0, cfg["jitter"] * sc, 3) * np.array([1.0, 1.0, 0.6])
    reach = cfg["tuft_from_knot"] * sc
    if cfg.get("length"):
        need_len = float(cfg["length"]) * sc + reach
    else:                                                             # until the guide is well below the bust line
        need_len = None
    P0 = guide_curve(b1, sx, cfg, sc, 0.34 * sc)
    if need_len is None:
        z_t = refs["bust_z"] + cfg["bottom_dz"] * sc
        k = np.flatnonzero(P0[:, 2] <= z_t - reach - 0.02 * sc)
        P0 = P0[:k[0] + 1] if len(k) else P0
    else:
        P0 = P0[:int(need_len * 1.1 / STEP) + 1]
    P1 = relax(P0, cons_builder(P0))
    s = arclen(P1)
    if cfg.get("length"):
        Lb = float(cfg["length"]) * sc
    else:
        below = np.flatnonzero(P1[:, 2] <= z_t)
        Lb = float(s[below[0]]) if len(below) else float(s[-1] - reach)
    s_end = Lb + reach
    if s[-1] < s_end + 4 * STEP:
        raise ValueError(f"braid {side}: guide curve too short ({s[-1]:.3f} m for {s_end:.3f} m)")
    keep = s <= s_end + 4 * STEP
    P1, s = P1[keep], s[keep]
    t0 = unit(P1[4] - P1[0])                                          # extend above the bow along the incoming tangent
    nb = int(np.ceil(0.032 * sc / STEP))
    back = P1[0][None] - t0[None] * (np.arange(nb, 0, -1) * STEP)[:, None]
    P = np.vstack([back, P1])
    sig = np.concatenate([-np.arange(nb, 0, -1) * STEP, s])
    n0 = facing(np.radians(cfg["face_yaw"]), sx)
    T, B, N = frames(P, np.broadcast_to(n0, P.shape))
    if float((B @ np.array([sx, 0.0, 0.0])).mean()) < 0:               # B points away from the midline
        B = -B
    return Plan(sx, P, sig, T, B, N, (0.0, Lb), s_end, {"b1": P1[0].copy(), "side": side, "guide": P0})
