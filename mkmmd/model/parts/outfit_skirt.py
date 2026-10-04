"""The skirt: a full A-line envelope with soft folds that always clears the hips and thighs, ending in a deep pleated
green ruffle with a lighter inner frill beneath; skirt chains (スカート) with dynamic bodies, and the skin weights that
tie it all together (numpy only).

Geometry: horizontal rings at levels z_k, radii rho[k, j] at M = columns * 15 angles around a vertical axis through the
waist. rho = bell envelope (waist outline -> hem ellipse) * soft folds, floored by the body (hips and thighs plus a clearance
that exceeds the collision capsules of the chains). Chains hang from the lower body (下半身): `columns` columns at
theta_c = c * 2pi / columns (c = 0 at the front, counter-clockwise from above), `rows` bones each; row heads sit at the
node heights z_1 > ... > z_rows on the skirt surface; the last bone ends at the bottom of the inner frill. Skin weights:
the body's own at the seam (blending into row 1), linear between row nodes, angular blend between the two nearest
columns."""
import numpy as np

from . import outfit_geo as G
from .outfit_dress import TILE, frill_base, hang_frill
from .outfit_rig import ring_weights


def resample_radii(ring, center, m):
    """Radii of a closed ring about a vertical axis at `center`, resampled to m angles (theta 0 = front)."""
    rel = np.asarray(ring, float) - np.asarray(center, float)
    th = np.arctan2(rel[:, 0], -rel[:, 1]) % G.TAU
    r = np.hypot(rel[:, 0], rel[:, 1])
    o = np.argsort(th)
    th, r = th[o], r[o]
    x = G.ring_theta(m)
    return np.interp(x, np.concatenate([th - G.TAU, th, th + G.TAU]), np.tile(r, 3))


def ellipse_radius(theta, rx, ry):
    return 1.0 / np.sqrt((np.sin(theta) / rx) ** 2 + (np.cos(theta) / ry) ** 2)


def required_radii(skin, center, zs, m, clearance, pad=0.012):
    """Smallest radius (K, m) at each level so the skirt stays `clearance` (a scalar or one value per level) outside every
    skin point in its slab."""
    clearance = np.broadcast_to(np.asarray(clearance, float), (len(zs),))
    V = skin.verts
    used = np.unique(skin.tris)
    cen = V[skin.tris].mean(1)
    P = np.vstack([V[used], cen])
    rel = P[:, :2] - np.asarray(center[:2], float)
    rho = np.hypot(rel[:, 0], rel[:, 1])
    th = np.arctan2(rel[:, 0], -rel[:, 1]) % G.TAU
    j = np.rint(th / (G.TAU / m)).astype(int) % m
    req = np.zeros((len(zs), m))
    for k, z in enumerate(zs):
        lo = z - (0.5 * (z - zs[k + 1]) if k + 1 < len(zs) else 0.02) - pad
        hi = z + (0.5 * (zs[k - 1] - z) if k > 0 else 0.0) + pad
        sel = (P[:, 2] >= lo) & (P[:, 2] <= hi)
        r = np.full(m, np.nan)
        if sel.any():
            vals = np.zeros(m)
            np.maximum.at(vals, j[sel], rho[sel])
            r = np.where(vals > 0, vals, np.nan)
        if np.isfinite(r).any():
            x = np.arange(m)
            ok = np.isfinite(r)
            r = np.interp(x, np.concatenate([x[ok] - m, x[ok], x[ok] + m]), np.tile(r[ok], 3))
            r = G.dilate_circular(r, max(2, m // 120))
            r = G.smooth_circular(r, 1, 2)
            req[k] = r + clearance[k]
    return req


class SkirtShape:
    """Radii rho[k, j] of the skirt rings: rings at z[k] (decreasing), angles theta[j] = j * 2pi / m."""

    def __init__(self, fit, cfg, ring0, c0, m, columns):
        S = fit.S
        self.S, self.m, self.columns = S, m, columns
        self.theta = G.ring_theta(m)
        self.center = np.array([c0[0], c0[1]])
        self.z_top = float(c0[2])
        L = fit.L
        knee = 0.5 * (L["knee.L"][2] + L["knee.R"][2])
        # canon: the skirt ends just below the knee. Absolute metres: the ruffle's lower edge is hem_below_knee under the
        # knee joint, the inner frill peeks inner_below further down (so the lowest edge is about 5.6 cm under the knee)
        self.z_hem = float(cfg["hem_z"]) if cfg.get("hem_z") else float(knee - cfg.get("hem_below_knee", 0.030))
        self.h_ruffle = cfg.get("ruffle_h", 0.115) * S
        self.z_rt = self.z_hem + self.h_ruffle                         # ruffle top = end of the main skirt
        self.z_in = self.z_hem - cfg.get("inner_below", 0.026)         # bottom of the inner frill
        hem = np.array(cfg.get("hem_radius", [0.345, 0.325])) * S
        xs = np.array(cfg.get("levels", [0.0, 0.015, 0.045, 0.085, 0.135, 0.195, 0.265, 0.345, 0.43, 0.52, 0.61, 0.70,
                                          0.79, 0.88, 1.0]))
        self.xs = xs
        self.z = self.z_top - xs * (self.z_top - self.z_rt)
        r_w = resample_radii(ring0, c0, m)
        E = ellipse_radius(self.theta, hem[0], hem[1])
        a = float(cfg.get("flare_mix", 0.30))
        p = float(cfg.get("flare_pow", 1.9))
        f = a * (1.0 - (1.0 - xs) ** p) + (1.0 - a) * xs
        env = r_w[None, :] * (1.0 - f)[:, None] + E[None, :] * f[:, None]
        # soft folds: crests on the chain columns, deeper towards the hem, a little irregular. The cloth's innermost level
        # (the fold troughs, `rho_in`) is what must clear the body and where the chain lines run; crests stand out of it
        fold = np.cos(columns * self.theta) + 0.22 * np.cos(2 * columns * self.theta + 0.9)
        fold = fold / 1.22 + 0.25 * G.periodic_noise(self.theta, 5, 7, 0.7)
        fold = np.clip(fold, -1.0, 1.0)
        amp = cfg.get("fold_amp", 0.035) * G.smoothstep(xs / 0.9) ** 1.2
        # the clearance grows from a snug waistband to the full margin over the top quarter, so the cloth leaves the waist
        # in one smooth flare instead of a ledge over the hips
        c_full = cfg.get("clearance", 0.045) * S
        c_top = cfg.get("clearance_top", 0.010) * S
        clearance = c_top + (c_full - c_top) * G.smoothstep(xs / cfg.get("clearance_ease", 0.25))
        req = required_radii(fit.core, self.center, self.z, m, clearance)
        self.req = req
        rho_in = np.maximum(env * (1.0 - amp)[:, None], req)
        # soften kinks left by the floor, keep the floor
        rho_in[1:-1] = 0.25 * rho_in[:-2] + 0.5 * rho_in[1:-1] + 0.25 * rho_in[2:]
        for k in range(len(rho_in)):
            rho_in[k] = G.smooth_circular(rho_in[k], 1, 1)
        rho_in = np.maximum(rho_in, req)
        rho_in[0] = r_w                                               # the seam ring is the bodice's own outline
        self.rho_in = rho_in
        self.rho = rho_in + env * amp[:, None] * (1.0 + fold[None, :])
        self.rho[0] = r_w

    def node_heights(self, fit, cfg, rows):
        """Heights z_1 > ... > z_rows of the chain row heads: the first just below the hip joints (`first_node_above_hip`,
        -2.5 cm; above it the skirt follows the body's own skin weights, so hips and thighs never collapse into the cloth),
        the last just above the ruffle top (the last bone carries the ruffle and the inner frill), the others evenly
        between; `first_segment` (m) makes the first bone that long instead. Tried and dropped: roots at or above the hip
        joints with a short first bone. Seated, the thigh sweeps through the cloth in front of the pelvis there, the roots
        end inside the thigh collider and the chains fold into the lap; the front tips then stop 0.25 m short of the
        knees whatever the roots, bone lengths, radii or sag, so the limit is the solver's missing cloth coupling."""
        S = fit.S
        z_hip = 0.5 * (fit.L["leg.L"][2] + fit.L["leg.R"][2])
        z1 = float(cfg["first_node_z"]) if cfg.get("first_node_z") else z_hip + cfg.get("first_node_above_hip", -0.025)
        zR = self.z_rt + cfg.get("last_node_above_ruffle", 0.012) * S
        seg = cfg.get("first_segment")
        if rows <= 2 or seg is None:
            return np.linspace(z1, zR, rows)
        return np.concatenate([[z1], np.linspace(z1 - seg, zR, rows - 1)])

    def radius_at(self, z, theta, inner=False):
        """Interpolated radius at height z (any) and angles theta (m,) or scalar; clamps beyond the rings. `inner`: the
        cloth's innermost level (where the chain lines run) instead of its visible surface."""
        zz = self.z[::-1]
        tab = (self.rho_in if inner else self.rho)[::-1]
        k = np.clip(np.searchsorted(zz, z) - 1, 0, len(zz) - 2)
        t = np.clip((z - zz[k]) / (zz[k + 1] - zz[k]), 0.0, 1.0)
        rr = tab[k] * (1 - t) + tab[k + 1] * t
        return np.interp(np.asarray(theta) % G.TAU, np.concatenate([self.theta, [G.TAU]]), np.concatenate([rr, rr[:1]]))

    def point(self, z, theta, inner=False):
        th = np.asarray(theta, float)
        r = self.radius_at(z, th, inner)
        return np.stack([self.center[0] + r * np.sin(th), self.center[1] - r * np.cos(th), np.full(np.shape(th), z)], -1)


# collision radius of the chain bodies per row (absolute metres; mk's strand solver collides a particle of half this size,
# at most 3 cm). Small near the waist, where the cloth is tied to the body and has no room (a root-near point that is
# held next to the seat cushion can never reach a big radius from it), bulky towards the hem to keep it off the thighs.
DEFAULT_RADII = (0.020, 0.030, 0.045, 0.060)


def bone_names(columns, rows, prefix="スカート"):
    return [[f"{prefix}{c:02d}_{r + 1}" for c in range(columns)] for r in range(rows)]


def skirt(fit, cfg, info, soup_dress, soup_trim, rig, lower_body_body=None, anchor="下半身"):
    """Build the skirt meshes and its chains. Returns an info dict for tests and the other pieces."""
    S = fit.S
    C = int(cfg.get("columns", 16))
    R = int(cfg.get("rows", 4))
    per = int(cfg.get("per_column", 20))
    M = C * per
    ring0, c0 = info["bodice_bottom"], info["bottom_center"]
    shape = SkirtShape(fit, cfg, ring0, c0, M, C)
    theta = shape.theta
    K = len(shape.z)
    rings = np.array([np.stack([shape.center[0] + shape.rho[k] * np.sin(theta), shape.center[1] - shape.rho[k] * np.cos(theta),
                                np.full(M, shape.z[k])], -1) for k in range(K)])
    z_all = shape.node_heights(fit, cfg, R)
    names = bone_names(C, R)
    # ---- main skirt
    nrep = int(cfg.get("u_repeat", 4))
    u = np.linspace(0.0, nrep, M + 1)
    v = 1.0 - np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rings, axis=0), axis=2).mean(1))]) / TILE["dress"]
    p = G.loft(rings, True, u=u, v=v, mat=0, tag="skirt")
    cen = np.repeat(np.array([[shape.center[0], shape.center[1], z] for z in shape.z]), M, axis=0)
    p = G.orient_outward(p, cen)
    zz = np.repeat(shape.z, M)
    th_v = np.tile(theta, K)
    donors = [b for b in ("下半身", "上半身", "上半身2", "腰", "センター", "グルーブ") if b in fit.skin.weights]
    w_top = fit.weights_for(p.v, donors or None, default=anchor)
    p.w = ring_weights(th_v, zz, names, z_all, shape.z_top, w_top, C)
    soup_dress.add(p)
    # ---- outer ruffle (deep green tier): its own finer ring, six samples per pleat
    pl = int(cfg.get("pleats", 48))
    axis_pt = np.array([shape.center[0], shape.center[1], shape.z_rt])
    base, out, _ = frill_base(rings[-1] + G.hdir(theta) * 0.0015 * S, axis_pt, pl, int(cfg.get("per_pleat", 6)))
    th_f = G.ring_theta(len(base))
    down = np.array([0.0, 0.0, -1.0])
    prof = np.array(cfg.get("ruffle_profile", [(0.0, 0.0), (0.0025, 0.0100), (0.0050, 0.0225), (0.0085, 0.0360), (0.0125, 0.0500),
                                               (0.0165, 0.0635), (0.0200, 0.0760), (0.0230, 0.0850)])) * S
    prof[:, 1] *= shape.h_ruffle / (0.085 * S)
    rp, rrings = hang_frill(base, out, down, prof, pl, cfg.get("ruffle_amp", 0.0110) * S, 0, "ruffle", axis_pt,
                            wobble=0.38, seed=21, hem_wobble=0.0042 * S, amp_pow=1.25, ridge=0.18)
    rp.w = ring_weights(np.tile(th_f, len(prof)), rrings[:, :, 2].reshape(-1), names, z_all, shape.z_top, {}, C)
    soup_trim.add(rp, 2)                                              # the deeper-green hem material
    # ---- inner frill (lighter), hanging under the main skirt inside the ruffle, peeking out below its hem
    z_it = shape.z_rt - 0.012 * S
    ibase = shape.point(z_it, th_f) - out * 0.0030 * S
    drop = z_it - shape.z_in
    iprof = np.array([(0.0, 0.0), (0.0020, 0.18), (0.0050, 0.36), (0.0085, 0.54), (0.0120, 0.72), (0.0145, 0.88), (0.0160, 1.0)])
    iprof[:, 1] *= drop
    iprof[:, 0] *= S
    ip, irings = hang_frill(ibase, out, down, iprof, pl, cfg.get("inner_amp", 0.0070) * S, 1, "inner_frill", axis_pt,
                            wobble=0.30, seed=33, hem_wobble=0.0022 * S, phase=np.pi, amp_pow=1.2, ridge=0.15)
    ip.w = ring_weights(np.tile(th_f, len(iprof)), irings[:, :, 2].reshape(-1), names, z_all, shape.z_top, {}, C)
    soup_trim.add(ip, 1)
    # ---- chains
    pts = np.zeros((C, R + 1, 3))
    for c in range(C):
        th = c * G.TAU / C
        for r in range(R):
            pts[c, r] = shape.point(z_all[r], th, inner=True)
        j = int(round(c * len(th_f) / C)) % len(th_f)
        pts[c, R] = irings[-1, j]                                       # the last bone ends at the inner frill's hem
    rad = np.interp(np.arange(R), np.linspace(0, R - 1, len(DEFAULT_RADII)), DEFAULT_RADII) if "body_radius" not in cfg \
        else np.broadcast_to(np.asarray(cfg["body_radius"], float), (R,))
    for c in range(C):
        th = c * G.TAU / C
        tang = G.tangent_ccw(th)
        rig.chain([names[r][c] for r in range(R)], pts[c], anchor, "skirt", rad,
                  anchor_body=lower_body_body, x_hint=tang,
                  name_en=[f"Skirt{c:02d}_{r + 1}" for r in range(R)])
    return {"shape": shape, "rings": rings, "ruffle_rings": rrings, "inner_rings": irings, "names": names, "z_nodes": z_all,
            "points": pts, "columns": C, "rows": R, "per_column": per, "z_hem": shape.z_hem, "z_in": shape.z_in,
            "z_rt": shape.z_rt, "ruffle_profile": prof}
