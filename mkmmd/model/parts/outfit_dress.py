"""Bodice, collar, sash, sleeves and cuffs of Rin's dress (numpy only).

Every piece is an offset shell of the body: rings are cut through the body surface (`outfit_fit.Fit`), pushed out by an
ease profile and lofted. Skin weights are the body's own, transferred from the nearest surface point, so the dress
deforms exactly like the skin it covers (arms raised, bent, seated). Frills hang from rings of the pieces they trim and
copy the weights of the ring they hang from."""
import numpy as np

from . import outfit_geo as G
from .outfit_fit import HAND_STEMS
from .outfit_rig import ring_weights

TILE = {"dress": 0.30, "frill_u": 0.08, "satin_u": 0.04}


def _vsmooth(R, passes=1):
    """Smooth ring radii along the loft direction (K, M), end rings kept."""
    R = np.array(R, float)
    for _ in range(passes):
        R[1:-1] = 0.25 * R[:-2] + 0.5 * R[1:-1] + 0.25 * R[2:]
    return R


def _meridian_v(rings, tile):
    """Texture v (K,) from the mean 3D distance between consecutive rings (decreasing downwards from 1)."""
    d = np.linalg.norm(np.diff(rings, axis=0), axis=2).mean(1)
    return 1.0 - np.concatenate([[0.0], np.cumsum(d)]) / tile


def frill_uv(base, rows, tile_u=None, reverse=False):
    """(u (n+1,), v (rows,)) for a frill hanging from the closed ring `base`: u counts whole tiles around (seamless),
    v runs 1 (attachment) -> 0 (hem)."""
    tile_u = tile_u or TILE["frill_u"]
    per = np.linalg.norm(np.roll(base, -1, 0) - base, axis=1)
    n = max(1, int(round(per.sum() / tile_u)))
    u = np.concatenate([[0.0], np.cumsum(per)]) / per.sum() * n
    return u, np.linspace(1.0, 0.0, rows)


def hang_frill(base, out, along, profile, pleats, amp, mat, tag, centers, tile_u=None, seed=0, fit=None, margin=0.0, **kw):
    """Frill patch hanging from `base` (M, 3): rings via frill_rings (kept `margin` off the body when `fit` is given, the
    attachment ring pinned), UV seamless, faces away from `centers`."""
    prof = np.asarray(profile, float)
    rings = G.frill_rings(base, out, along, prof, pleats=pleats, amp=amp, seed=seed, **kw)
    if fit is not None and margin > 0.0:
        rings = fit.conform(rings, margin, pin_first=True)
    u, v = frill_uv(base, len(prof), tile_u)
    p = G.loft(rings, True, u=u, v=v, mat=mat, tag=tag)
    cen = np.broadcast_to(np.asarray(centers, float), (len(rings) * rings.shape[1], 3))
    return G.orient_outward(p, cen), rings


def frill_base(base, center, pleats, per_pleat=6):
    """The attachment ring of a frill at the resolution its pleats need (`per_pleat` samples each, a multiple of 4):
    (points (m, 3), out (m, 3) unit directions away from `center`, source index (m,) into `base`)."""
    m = int(np.ceil(per_pleat * pleats / 4.0) * 4)
    pts, src = G.resample_ring(base, m)
    return pts, G.unit(pts - np.asarray(center, float)), src


def copy_down(w_ring, rows):
    """Weights of a (K=rows, M) frill grid copied from the ring it hangs from: dict of (M,) -> dict of (rows*M,)."""
    return {b: np.tile(np.asarray(a, float), rows) for b, a in w_ring.items()}


# ---------------------------------------------------------------- bodice + collar
def bodice(fit, cfg, soup_dress, soup_trim, rig=None, find_body=None):
    """Fitted bodice from the waist seam up over the shoulders into the stand-up collar, plus the collar frill.
    Returns an info dict (rings, heights, collar ring, weights) for the pieces that attach to it."""
    S, L = fit.S, fit.L
    M = int(cfg.get("segments", 64))
    z_w = float(cfg["waist_z"]) if cfg.get("waist_z") else fit.waist_z()
    z_neck = float(L["neck"][2])
    z_arm = float(L["arm.L"][2])
    z_c1 = float(cfg["collar_top_z"]) if cfg.get("collar_top_z") else z_neck + 0.004 * S   # collar rim: just under the jaw
    z_c0 = z_c1 - cfg.get("collar_h", 0.034) * S
    z_b0 = float(cfg["seam_z"]) if cfg.get("seam_z") else z_w - cfg.get("seam_drop", 0.004) * S    # bodice bottom = skirt top
    # (under the sash: its half width is 0.0125 * S)
    # levels top -> bottom
    collar = np.linspace(z_c1, z_c0, 5)
    shoulder = np.linspace(z_c0, z_arm - 0.045 * S, max(6, int(round((z_c0 - z_arm + 0.045 * S) / (cfg.get('shoulder_step', 0.003) * S)))))[1:]
    body = np.arange(shoulder[-1] - 0.02 * S, z_b0, -0.02 * S)
    zs = np.concatenate([collar, shoulder, body, [z_b0]])
    zs = np.unique(np.round(zs, 6))[::-1]
    theta = G.ring_theta(M)
    ease = cfg.get("ease", {})
    off_z = np.array([z_b0, z_w, (z_w + z_arm) / 2, z_arm, z_neck, z_c1])
    off_v = np.array([ease.get("waist", 0.0045), ease.get("waist", 0.0045), ease.get("chest", 0.0070),
                      ease.get("shoulder", 0.0090), ease.get("neck", 0.0075), ease.get("neck", 0.0075)]) * S
    ctrs, rads = [], []
    last = None
    r_shoulder = float(L["arm.L"][0]) + cfg.get("shoulder_reach", 0.03) * S    # lateral reach of the shoulder ball
    for z in zs:
        c, r = fit.torso_section(float(z), M)
        if z >= z_arm - 0.004 * S and z <= z_c0:
            # the shoulder tops: the torso region alone leaves the shoulder ball (a shell of the arm) bare, so also take the
            # outermost crossing of the whole body within the shoulder's reach
            c2, r2 = fit.torso_section(float(z), M, skin=fit.skin, mode="far", rmax=r_shoulder)
            if r2 is not None:
                r = r2 if r is None else np.maximum(r, r2)
                c = c if c is not None else c2
        if r is None:                                           # body surface missing here: reuse the nearest ring
            c, r = (fit.spine(float(z)), last[1]) if last is not None else (fit.spine(float(z)), np.full(M, 0.05 * S))
        last = (c, r)
        ctrs.append(c)
        rads.append(r)
    rads = np.array(rads)
    rads = np.array([G.dilate_circular(r, 2) for r in rads])
    rads = np.array([G.smooth_circular(r, 2, 2) for r in rads])
    rads = _vsmooth(rads, 1)
    off = np.interp(zs, off_z, off_v)
    # stand-up collar: slightly flared, never narrower than the neck + ease
    rings = np.array([ctrs[k] + G.hdir(theta) * (rads[k] + off[k])[:, None] for k in range(len(zs))])
    n_c = len(collar)
    flare = np.linspace(0.0, 1.0, n_c) ** 2 * cfg.get("collar_flare", 0.003) * S
    for k in range(n_c):                                        # collar rings: flare outwards towards the top
        rings[k] += G.hdir(theta) * (flare[::-1][k])
    rings = fit.conform(rings, cfg.get("min_offset", 0.006))     # nothing inside the skin, nowhere skin showing through
    nrep = int(cfg.get("u_repeat", 2))
    u = np.linspace(0.0, nrep, M + 1)
    v = _meridian_v(rings, TILE["dress"])
    p = G.loft(rings, True, u=u, v=v, mat=0, tag="bodice")
    p = G.orient_outward(p, np.repeat(np.array(ctrs), M, axis=0))
    p.w = fit.weights_for(p.v)
    soup_dress.add(p)
    # ---- collar frill
    top = rings[0]
    cc = np.array(ctrs[0])
    pl = int(cfg.get("collar_pleats", 30))
    top_f, out, src = frill_base(top, cc * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, top[:, 2].mean()]), pl)
    prof = np.array(cfg.get("collar_frill_profile", [(0.0, 0.0), (0.0030, 0.0030), (0.0080, 0.0022), (0.0140, -0.0020),
                                                    (0.0190, -0.0075), (0.0225, -0.0140)])) * S
    # frill rows follow the profile (outward, up)
    fp, frings = hang_frill(top_f, out, np.array([0.0, 0.0, 1.0]), prof, pl,
                            cfg.get("collar_amp", 0.0032) * S, 0, "collar_frill", cc, wobble=0.35,
                            seed=3, hem_wobble=0.0016 * S, fit=fit, margin=cfg.get("collar_frill_margin", 0.0105))
    wtop0 = fit.weights_for(top)
    wtop = {b: a[src] for b, a in wtop0.items()}
    fp.w = copy_down(wtop, len(prof))
    soup_trim.add(fp, 0)
    collar_chains = None
    if rig is not None and cfg.get("collar_physics", True):
        nc = int(cfg.get("collar_columns", 6))
        names = [[f"襟フリル{c}_1" for c in range(nc)]]
        ab = find_body("首") if find_body else None
        collar_chains = frill_chains(rig, fp, frings, wtop, names, [1.2], "首", ab.name if ab else None, "coat",
                                     cfg.get("collar_body_radius", 0.010), nc,
                                     lambda c, pts: np.array([np.cos(c * G.TAU / nc), np.sin(c * G.TAU / nc), 0.0]),
                                     name_en=[[f"CollarFrill{c}" for c in range(nc)]])
    return {"collar_chains": collar_chains, "rings": rings, "centers": np.array(ctrs), "zs": zs, "theta": theta, "z_waist": z_w, "z_collar": (z_c0, z_c1),
            "collar_top": top, "collar_center": cc, "bodice_bottom": rings[-1], "bottom_center": np.array(ctrs)[-1],
            "weights_bottom": fit.weights_for(rings[-1]), "frill_top_rings": frings}


# ---------------------------------------------------------------- sash
def sash(fit, cfg, soup_satin, info):
    """The thin black sash at the waist: a pillowed band hugging the bodice/skirt seam."""
    S = fit.S
    M = int(cfg.get("segments", 64))
    z_w = info["z_waist"]
    h = cfg.get("width", 0.0125) * S                            # half width
    zs = np.array([z_w + 1.18 * h, z_w + h, z_w + 0.5 * h, z_w, z_w - 0.5 * h, z_w - h, z_w - 1.18 * h])
    lift = np.array([0.0006, 0.0030, 0.0046, 0.0052, 0.0046, 0.0030, 0.0006]) * S + cfg.get("gap", 0.0050) * S
    theta = G.ring_theta(M)
    rings = []
    for z, lf in zip(zs, lift):
        c, r = fit.torso_section(float(z), M)
        if r is None:
            c, r = fit.spine(float(z)), np.full(M, 0.1 * S)
        pts, rr = fit.garment_ring(c, r, theta, cfg.get("ease", 0.0045) * S, 2, (2, 2))
        rings.append(c + G.hdir(theta) * (rr + lf - cfg.get("ease", 0.0045) * S)[:, None])
    rings = np.array(rings)
    u = np.linspace(0.0, max(1, int(round(np.linalg.norm(np.roll(rings[3], -1, 0) - rings[3], axis=1).sum() / TILE["satin_u"]))), M + 1)
    v = np.linspace(1.0, 0.0, len(zs))
    p = G.loft(rings, True, u=u, v=v, mat=0, tag="sash")
    cen = np.repeat(np.array([fit.spine(float(z)) for z in zs]), M, axis=0)
    p = G.orient_outward(p, cen)
    p.w = fit.weights_for(p.v)
    soup_satin.add(p)
    return {"sash_rings": rings, "z_waist": z_w}


# ---------------------------------------------------------------- sleeves + cuffs
def arm_path(fit, side, n=36, ext=0.0):
    """Centre line of an arm: shoulder joint -> elbow -> wrist as an arc-length parametrised Catmull path; returns
    points (n, 3) for t in [0, 1] (t = 0 shoulder, 1 wrist) and the unit tangents."""
    L = fit.L
    a, e, w = L[f"arm.{side}"], L[f"elbow.{side}"], L[f"wrist.{side}"]
    ctrl = np.array([a - (e - a) * 0.5, a, e, w, w + (w - e) * 0.3])
    dense = G.catmull(ctrl, 200)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # arc length from the shoulder (index of the landmark a) to the wrist
    ia = int(np.argmin(np.linalg.norm(dense - a, axis=1)))
    iw = int(np.argmin(np.linalg.norm(dense - w, axis=1)))
    s0, s1 = s[ia], s[iw]
    ts = np.linspace(0.0, 1.0, n)
    pts = np.stack([np.interp(s0 + ts * (s1 - s0), s, dense[:, k]) for k in range(3)], 1)
    T = G.unit(np.gradient(pts, axis=0))
    return ts, pts, T, (s1 - s0)


def sleeve(fit, side, cfg, soup_dress, soup_trim, info, rig=None, find_body=None):
    """One long sleeve (puffed at the shoulder) and its green cuff frill. `side` is "L" or "R"."""
    S = fit.S
    M = int(cfg.get("segments", 32))
    n = int(cfg.get("rings", 30))
    ts, pts, T, length = arm_path(fit, side, n)
    t_end = cfg.get("t_end", 0.93)
    ts = np.linspace(0.0, t_end, n)
    pts_all, T_all, _, length = (lambda a: (np.stack([np.interp(ts, a[0], a[1][:, k]) for k in range(3)], 1),
                                            np.stack([np.interp(ts, a[0], a[2][:, k]) for k in range(3)], 1), None, a[3]))(
        arm_path(fit, side, 80))
    T_all = G.unit(T_all)
    front = np.array([0.0, -1.0, 0.0])
    ease = cfg.get("ease", {})
    off_t = np.array([0.0, 0.12, 0.35, 0.55, 0.80, 1.0])
    off_v = np.array([ease.get("shoulder", 0.0120), ease.get("puff", 0.0185), ease.get("upper", 0.0115), ease.get("elbow", 0.0085),
                      ease.get("bell", 0.0125), ease.get("cuff", 0.0080)]) * S
    rings, centers, radii = [], [], []
    for k, t in enumerate(ts):
        ax = T_all[k]
        u = front - ax * np.dot(front, ax)
        u = G.unit(u)
        v = np.cross(ax, u)
        rmax = 0.09 * S if not radii else 1.8 * max(float(radii[-1].max()), 0.02 * S)
        c, r = fit.skin.section(pts_all[k], u, v, M, "near", rmin=0.004, rmax=rmax, iters=1)
        if r is None:
            c, r = pts_all[k], np.full(M, 0.04 * S)
        r = G.smooth_circular(G.dilate_circular(r, 1), 2, 2)
        o = float(np.interp(t / t_end, off_t, off_v))
        th = G.ring_theta(M)
        ring = c + (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v) * (r + o)[:, None]
        rings.append(ring)
        centers.append(c)
        radii.append(r + o)
    rings = fit.conform(np.array(rings), cfg.get("min_offset", 0.006))
    centers = np.array(centers)
    # shoulder cap: shrink the first ring into a dome inside the torso
    cap = []
    ax0 = T_all[0]
    r0 = rings[0] - centers[0]
    for tau, sc in ((0.010, 0.93), (0.020, 0.80), (0.029, 0.60), (0.036, 0.34), (0.040, 0.0)):
        cap.append(centers[0] - ax0 * tau * S + r0 * sc)
    cap = np.array(cap[::-1])                                   # innermost first
    allr = np.concatenate([cap, rings], 0)
    allc = np.concatenate([np.repeat(centers[:1], len(cap), 0), centers], 0)
    allr = np.concatenate([allr[1:]], 0)                        # the innermost (scale 0) ring is the apex
    apex = cap[0].mean(0)
    nrep = int(cfg.get("u_repeat", 1))
    u = np.linspace(0.0, nrep, M + 1)
    v = _meridian_v(allr, TILE["dress"])
    p = G.loft(allr, True, u=u, v=v, mat=0, tag=f"sleeve_{side}")
    p = G.orient_outward(p, np.repeat(allc[1:], M, axis=0))
    cp = G.fan(apex, allr[0], mat=0, tag=f"sleeve_{side}_cap")
    cp = G.orient_outward(cp, allc[1] + ax0 * 0.02)
    # weights: nearest body surface, hands excluded (fingers would drag a cuff)
    donors = [b for b in fit.skin.weights if not any(b[1:].startswith(h) for h in HAND_STEMS)]
    pre = "左" if side == "L" else "右"
    for q in (p, cp):
        q.w = fit.weights_for(q.v, donors, default=f"{pre}手首")
    soup_dress.add(p)
    soup_dress.add(cp)
    # ---- cuff frill hanging off the last ring towards the hand
    base = rings[-1]
    cen = centers[-1]
    along = T_all[-1]
    cp = int(cfg.get("cuff_pleats", 15))
    base_f, out, src = frill_base(base, cen, cp)
    prof = np.array(cfg.get("cuff_profile", [(0.0, 0.0), (0.0030, 0.0090), (0.0090, 0.0190), (0.0155, 0.0290),
                                              (0.0215, 0.0390), (0.0260, 0.0480)])) * S
    fp, frings = hang_frill(base_f, out, along, prof, cp, cfg.get("cuff_amp", 0.0045) * S, 0,
                            f"cuff_{side}", cen, wobble=0.30, seed=11 if side == "L" else 17, hem_wobble=0.0016 * S,
                            fit=fit, margin=cfg.get("cuff_margin", 0.0045))
    wend0 = fit.weights_for(base, donors, default=f"{pre}手首")
    wend = {b: a[src] for b, a in wend0.items()}
    fp.w = copy_down(wend, len(prof))
    soup_trim.add(fp, 0)
    chains = None
    if rig is not None and cfg.get("cuff_physics", True):
        nc = int(cfg.get("cuff_columns", 4))
        names = [[f"袖{pre}{c}_{r + 1}" for c in range(nc)] for r in range(2)]
        # the cuff hangs from the sleeve end, which rides the forearm: anchor the chains on the twist bone (not the hand, whose
        # flexion would swing them into the forearm)
        bones = fit.bone_names
        anchor = f"{pre}手捩" if f"{pre}手捩" in bones else f"{pre}ひじ"
        ab = find_body(f"{pre}ひじ") if find_body else None
        chains = frill_chains(rig, fp, frings, wend, names, [1.4, 3.4], anchor, ab.name if ab else None, "sleeve",
                              cfg.get("cuff_body_radius", 0.012), nc, lambda c, pts: np.cross(T_all[-1], pts[0] - cen),
                              name_en=[[f"Cuff{side}{c}_{r + 1}" for c in range(nc)] for r in range(2)])
    return {"rings": rings, "centers": centers, "ts": ts, "axis": T_all, "cuff_base": base, "cuff_center": cen,
            "cuff_rings": frings, "length": length, "cuff_chains": chains}


# ---------------------------------------------------------------- chains on trims (bow tails, cuffs, collar frill)
def row_point(rings, j, row):
    """Point of column j at fractional row `row` of frill rings (R, M, 3)."""
    r0 = int(np.clip(np.floor(row), 0, len(rings) - 2))
    t = float(np.clip(row - r0, 0.0, 1.0))
    return rings[r0, j] * (1.0 - t) + rings[r0 + 1, j] * t


def frill_chains(rig, patch, rings, w_base, names, nodes, anchor, anchor_body, kind, radius, columns, x_hint_of,
                 name_en=None):
    """Chains down a frill hung from a ring: `columns` evenly spaced columns, each with len(nodes) bones that start at the
    fractional rows `nodes` and end at the frill's last ring. Sets the weights of `patch` (the frill's vertices: row-major
    (R * M): its attachment row keeps `w_base` (a dict of (M,) weights), blending into the first bone, linear between bone
    heads, angular blend between the two nearest columns). names[r][c]; returns the chain points (columns, len(nodes) + 1, 3)."""
    R, M = rings.shape[:2]
    pts = np.zeros((columns, len(nodes) + 1, 3))
    for c in range(columns):
        j = int(round(c * M / columns)) % M
        for i, row in enumerate(nodes):
            pts[c, i] = row_point(rings, j, row)
        pts[c, len(nodes)] = rings[-1, j]
        rig.chain([names[r][c] for r in range(len(nodes))], pts[c], anchor, kind, radius, anchor_body=anchor_body,
                  x_hint=x_hint_of(c, pts[c]), name_en=None if name_en is None else [name_en[r][c] for r in range(len(nodes))])
    th = np.tile(G.ring_theta(M), R)
    rows = np.repeat(np.arange(R, dtype=float), M)
    patch.w = ring_weights(th, -rows, names, -np.asarray(nodes, float), 0.0, copy_down(w_base, R), columns)
    return pts


def strip_chain_weights(t, w_top, names, t_nodes):
    """Skin weights for a hanging strip: vertex i is at parameter t[i] in 0..1 along the strip. The body's own weights
    `w_top` (dict bone -> (n,)) blend (smoothstep) into bone names[0] at t_nodes[0]; linear between bone heads; the last
    bone owns everything below its head. Returns dict bone -> (n,)."""
    t = np.asarray(t, float)
    R = len(names)
    nodes = np.concatenate([[0.0], np.asarray(t_nodes, float)])
    p = np.interp(t, nodes, np.arange(R + 1.0))                    # 0 at the top, k at the head of bone k - 1
    i = np.minimum(np.floor(p).astype(int), R - 1)
    f = np.where(p >= R, 1.0, p - i)
    s = np.where(i == 0, G.smoothstep(f), f)
    out = {b: np.asarray(a, float) * np.where(i == 0, 1.0 - s, 0.0) for b, a in w_top.items()}
    for k in range(R):
        w = np.where(i == k, s, 0.0)
        if k + 1 < R:
            w = w + np.where(i == k + 1, 1.0 - s, 0.0)
        out[names[k]] = out.get(names[k], 0.0) + w
    return out


def throat_bow(fit, cfg, soup_satin, rig, info, anchor="首", anchor_body=None):
    """The small black bow at the throat: sits on the collar front, two tails hang over the chest (a 2-bone ribbon chain
    each: リボン首A / リボン首B)."""
    S = fit.S
    rings, zs = info["rings"], info["zs"]
    z_c0, z_c1 = info["z_collar"]
    z_b = z_c0 + cfg.get("z_frac", 0.32) * (z_c1 - z_c0)
    front = rings[:, 0, :]                                         # theta = 0 points, top -> bottom
    O = np.array([np.interp(z_b, zs[::-1], front[::-1, k]) for k in range(3)])
    fwd = np.array([0.0, -1.0, 0.0])
    O = O + fwd * cfg.get("forward", 0.004) * S
    size = cfg.get("size", 0.0245) * S
    cols = int(cfg.get("cols", 5))
    bw = G.bow_ex(O, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), fwd, size=size, loop_len=cfg.get("loop_len", 1.0),
                  loop_w=cfg.get("loop_w", 0.62), ribbon_w=cfg.get("ribbon_w", 0.56), tail_len=cfg.get("tail_len", 1.9),
                  tail_splay=cfg.get("tail_splay", 0.30), droop=0.10, knot=0.24, tail_rows=8, notch=0.25, seed=5,
                  cup=cfg.get("cup", 0.30), cols=cols)
    margin = cfg.get("margin", 0.0105)
    tails = {id(p) for _, p in bw["tails"]}
    for p in bw["parts"]:
        p.v = fit.push_out(p.v, margin)
        if id(p) in tails:
            continue
        p.w = fit.weights_for(p.v, default="首")
    names = [["リボン首A1", "リボン首B1"], ["リボン首A2", "リボン首B2"]]
    rad = cfg.get("body_radius", 0.010)
    for c, (path, p) in enumerate(bw["tails"]):
        mid = p.v[cols // 2::cols]                                 # the centre line after the push
        tt = np.linspace(0.0, 1.0, len(mid))
        t_nodes = np.array([0.12, 0.55])
        pts = np.array([[np.interp(t, tt, mid[:, k]) for k in range(3)] for t in (0.12, 0.55, 1.0)])
        nm = [names[r][c] for r in range(2)]
        rig.chain(nm, pts, anchor, "ribbon", rad, anchor_body=anchor_body, x_hint=np.array([1.0, 0.0, 0.0]),
                  name_en=[f"NeckRibbon{'AB'[c]}{r + 1}" for r in range(2)])
        w_top = fit.weights_for(p.v[:3], default="首")             # the tail's top: the neck/chest skin
        w_top = {b: np.full(len(p.v), float(a.mean())) for b, a in w_top.items()}
        p.w = strip_chain_weights(np.repeat(tt, cols), w_top, nm, t_nodes)
    soup_satin.extend(bw["parts"], 0)
    return {"origin": O, "size": size, "tails": [p.v[cols // 2::cols] for _, p in bw["tails"]], "names": names}
