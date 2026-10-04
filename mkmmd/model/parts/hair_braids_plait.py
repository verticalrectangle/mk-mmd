"""The plait itself (bpy-free, numpy only): three woven strands, the tuft of loose locks and the gather clumps that run from
the scalp to the top bow.

A three-strand plait is three flat strands that swing from edge to edge and pass over and under each other in the cyclic
order of a real braid (A over B over C over A); every crossing is one lobe of the zigzag silhouette. Each strand is a closed
thin swept shell (outlines are drawn from closed shells); where two strands cross, the one on top lies a strand thickness
above the other, so the creases are real and nothing interpenetrates. All blocks come back as `Block(verts, faces, uv,
kind, ...)` and are weighted by the caller from the chain.

Texture (the braid atlas, see hair_braids_tex): strand tiles map u = |angle| / pi around the strand (0 = the crest facing
out, 1 = the underside, mirrored so there is no seam) and v along the strand."""
from dataclasses import dataclass, field

import numpy as np

from .hair_braids_geo import orient_outward
from .hair_geo import arclen, resample, smooth_polyline, sweep, unit


@dataclass
class Block:
    """A piece of the braid mesh: positions, polygons, per-vertex UV and what it is (for weights and bookkeeping)."""
    kind: str                                   # strand | lock | gather | band | knot | wing | tail
    verts: np.ndarray
    faces: list
    uv: np.ndarray
    mat: int = 0
    tag: dict = field(default_factory=dict)


# ---------------------------------------------------------------- ellipsoid shells (the bow knots)
def ellipsoid_shell(c, e1, e3, a, b, h, n_around=8, n_rings=5, p=0.8):
    """Ellipsoid-like closed shell: centre c, long axis e1 (semi-axis a), crest direction e3 (semi-axis h), third axis
    semi-axis b; the profile along the axis is sin(pi t)^p (p < 1: fuller, more capsule-like ends). Returns (verts, faces,
    u, v) with u = |angle| / pi (0 at the crest) and v = 0..1 along e1."""
    e1 = unit(e1)
    e3 = unit(np.asarray(e3, float) - np.dot(e3, e1) * e1)
    e2 = np.cross(e3, e1)
    t = (np.arange(n_rings) + 1.0) / (n_rings + 1.0)
    x = -a * np.cos(np.pi * t)
    r = np.sin(np.pi * t) ** p
    ang = 2.0 * np.pi * np.arange(n_around) / n_around
    verts = [c - a * e1, c + a * e1]
    u_ = [0.5, 0.5]
    v_ = [0.0, 1.0]
    for i in range(n_rings):
        for j in range(n_around):
            verts.append(c + x[i] * e1 + r[i] * (b * np.sin(ang[j]) * e2 + h * np.cos(ang[j]) * e3))
            u_.append(min(ang[j], 2.0 * np.pi - ang[j]) / np.pi)
            v_.append(0.5 * (x[i] / a + 1.0))
    ring = lambda i, j: 2 + i * n_around + (j % n_around)
    faces = []
    for j in range(n_around):
        faces.append([0, ring(0, j + 1), ring(0, j)])
        faces.append([1, ring(n_rings - 1, j), ring(n_rings - 1, j + 1)])
    for i in range(n_rings - 1):
        for j in range(n_around):
            faces.append([ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)])
    verts = np.array(verts)
    return verts, orient_outward(verts, faces), np.array(u_), np.array(v_)


def envelope(cfg, scale, plan, sigma, bow_half):
    """Width factor of the plait at arc position sigma: a slight taper towards the bottom bow, and a pinch where the
    plait runs into a bow's band (the tie gathers the hair; the plait swells to full width a lobe or two further on)."""
    sigma = np.asarray(sigma, float)
    Lb = plan.s_bow[1]
    taper = 1.0 - cfg["taper"] * np.clip(sigma / Lb, 0.0, 1.0)
    d = np.minimum(sigma - bow_half, Lb - bow_half - sigma)
    t = np.clip(d / (cfg["pinch_len"] * scale), 0.0, 1.0)
    pinch = 1.0 - cfg["pinch"] * (1.0 - t * t * (3.0 - 2.0 * t))
    return taper * pinch


# ---------------------------------------------------------------- the tuft below the bottom bow
def tuft_blocks(plan, cfg, scale, rng, uv_fn):
    """The small loose tuft: pointed flat locks fanning from under the bottom bow (roots inside its band). The middle locks
    are the longest; each curls a little towards the viewer at the tip. `uv_fn(tile, q, t)`."""
    n = int(cfg["tuft_locks"])
    s0 = plan.s_bow[1] + 0.004 * scale
    p0, T, B, N = plan.at(s0)
    fan = np.radians(cfg["tuft_fan"])
    tiles = rng.permutation(4)
    out = []
    for i in range(n):
        x = (i - 0.5 * (n - 1)) / max(0.5 * (n - 1), 1.0)                 # -1 .. 1 across the tuft
        psi = fan * x * rng.uniform(0.85, 1.1) + np.radians(rng.uniform(-2.5, 2.5))
        L = cfg["tuft_len"] * scale * (1.0 - 0.28 * abs(x) ** 1.5) * rng.uniform(0.92, 1.06)
        w = cfg["tuft_width"] * scale * (1.0 - 0.18 * abs(x)) * rng.uniform(0.92, 1.08)
        s = np.linspace(0.0, 1.0, 7)
        root = p0 + x * 0.30 * cfg["width"] * scale * B
        lat = np.tan(psi) * L * s * (0.3 + 0.7 * s)
        P = root + (L * s)[:, None] * T + lat[:, None] * B + (cfg["tuft_curl"] * L * s ** 2)[:, None] * N
        st = sweep(P, N, w, cfg["tuft_thick"] * scale, tip="point", tip_start=0.30, tip_power=1.35, root_w=1.0,
                   curv=6.0, rings=7, k_outer=int(cfg["tuft_k"]))
        uv = uv_fn(int(tiles[i % 4]), st.q, st.arc / max(st.length, 1e-9))
        out.append(Block("lock", st.verts, st.faces, uv, tag=dict(i=i, root=root, length=L)))
    return out


# ---------------------------------------------------------------- gather clumps: scalp -> top bow
def hermite(p0, m0, p1, m1, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def gather_paths(plan, fit, vol, cfg, scale, rng, g):
    """Centrelines of the gather clumps of one side: each starts on the hair hull (skin + hair volume + delta) behind and
    below the ear at azimuth theta / polar angle phi, follows the hull down to where it leaves the head, then converges
    onto the top bow along the incoming braid direction. Returns a list of dicts (theta, phi, root, dir, path, width)."""
    from .hair_fit import dirs
    sk = fit.skull
    sx = plan.sx
    n = int(cfg["gather"])
    p_bow, T1, B1, N1 = plan.at(0.0)
    p_in, _, _, _ = plan.at(-g)                                         # where the chain starts (g above the knot)
    out = []
    th_lo, th_hi = np.radians(cfg["gather_theta"])
    ph_lo, ph_hi = np.radians(cfg["gather_phi"])
    for i in range(n):
        f = i / max(n - 1, 1)
        theta = sx * (th_lo + (th_hi - th_lo) * f + np.radians(rng.uniform(-2.0, 2.0)))
        zig = (-1.0) ** i * 0.5 + 0.5                                         # alternate high / low roots
        phi = ph_lo + (ph_hi - ph_lo) * (0.15 + 0.7 * ((0.5 * f + 0.5 * zig) % 1.0)) + np.radians(rng.uniform(-2.0, 2.0))
        delta = cfg["gather_delta"] + rng.uniform(0.0, 0.004)
        # follow the hull down the meridian until its height is z_leave, then leave towards the bow
        z_leave = p_bow[2] + cfg["gather_leave"] * scale * rng.uniform(0.9, 1.1)
        ph = np.linspace(phi, np.radians(150.0), 60)
        th = theta + sx * np.radians(cfg["gather_drift"]) * np.linspace(0.0, 1.0, len(ph)) ** 1.5
        D = dirs(th, ph)
        ramp = np.clip(np.arange(len(ph)) / 10.0, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        H = sk.point(D, vol(th, ph) + delta - cfg.get("gather_bury", 0.016) * (1.0 - ramp))     # roots start buried in the hair
        below = np.flatnonzero(H[:, 2] <= z_leave)
        k = int(below[0]) if len(below) else len(H) - 1
        k = max(k, 3)
        hull = H[:k + 1]
        L = hull[-1]
        m0 = unit(hull[-1] - hull[-4])
        # the end point: spread across the bow's band, as a fan of clumps
        spread = (f - 0.5) * cfg["gather_spread"] * scale
        E = p_in + 0.55 * g * T1 + spread * B1 + rng.uniform(-0.002, 0.002) * N1
        chord = float(np.linalg.norm(E - L))
        free = hermite(L, m0 * chord * 1.1, E, T1 * chord * 1.1, max(int(chord / 0.006), 6))
        path = np.vstack([hull[:-1], free])
        s = arclen(path)
        path = resample(path, max(int(s[-1] / 0.008), 8))
        path = smooth_polyline(path, 2)
        d0 = unit(path[2] - path[0])
        out.append(dict(theta=float(theta), phi=float(phi), root=path[0].copy(), dir=d0, path=path,
                        width=cfg["gather_width"] * scale * rng.uniform(0.92, 1.08), delta=float(delta)))
    return out


def gather_blocks(plan, items, fit, cfg, scale, rng, uv_fn):
    out = []
    tiles = rng.permutation(4)
    for i, it in enumerate(items):
        path = it["path"]
        hint = unit(path - fit.center)
        w_bow = cfg["gather_end_width"] * scale
        st = sweep(path, hint, it["width"], cfg["gather_thick"] * scale, tip="blunt", root_w=0.55, root_len=0.22, curv=7.0,
                   rings=max(int(arclen(path)[-1] / cfg["gather_ring"]), 6), k_outer=int(cfg["gather_k"]),
                   width_fn=lambda sp, w0=it["width"], w1=w_bow: (1.0 - (1.0 - w1 / w0) * (np.clip((sp - 0.30) / 0.60, 0, 1) ** 1.2)),
                   thick_fn=lambda sp: 0.55 + 0.45 * np.clip(sp / 0.5, 0, 1))
        t = st.arc / max(st.length, 1e-9)
        uv = uv_fn(int(tiles[i % 4]), st.q, 0.93 - 0.88 * t)
        out.append(Block("gather", st.verts, st.faces, uv, tag=dict(i=i, t=t)))
    return out


# ---------------------------------------------------------------- the weave: three strands over / under
def strand_u(q, side):
    """Mirrored texture u of a sweep vertex: 0 on the crest (the middle of the outer arc), 0.5 at the two edges, 1 on the
    underside, so there is no seam."""
    a = np.abs(np.arcsin(np.clip(q, -1.0, 1.0))) / np.pi
    return np.where(side >= 0, a, 1.0 - a)


def weave_strands(plan, cfg, scale, rng, bow_half):
    """Centrelines of the three strands of the plait between the two bands: lists of dicts(P, N, phase, ...). Strand i has
    lateral position x_i = w * sgn(sin t) |sin t|^p and depth z_i = -d sin(2 t + 2 a_i + dz) with t = theta(sigma) + a_i, a_i =
    2 pi i / 3: strands cross every 60 degrees of theta (one crossing = one lobe) and the over / under order is the cyclic
    one of a real plait (A over B over C over A)."""
    K = float(cfg["lobes"])
    W = cfg["width"] * scale
    s0 = bow_half - cfg["weave_run_in"] * scale
    s1 = plan.s_bow[1] - bow_half + cfg["weave_run_in"] * scale
    n = max(int(round((s1 - s0) / cfg["weave_step"])) + 1, 8)
    ss = np.linspace(s0, s1, n)
    P, T, B, N = plan.at(ss)
    env = envelope(cfg, scale, plan, ss, bow_half)
    theta = np.radians(cfg["weave_phase"]) + np.pi / 3.0 * K * (ss - s0) / (s1 - s0)
    ws = cfg["strand_wid"] * W
    w = 0.5 * W - 0.5 * ws
    d = cfg["weave_depth"] * cfg["thick"] * scale
    p = cfg["weave_p"]
    out = []
    for i in range(3):
        a = 2.0 * np.pi * i / 3.0
        t = theta + a
        x = w * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p * env
        z = -d * np.sin(2.0 * t + np.radians(cfg["weave_dz"])) * env
        Pi = P + x[:, None] * B + z[:, None] * N
        out.append(dict(i=i, P=Pi, N=N, x=x, z=z, theta=t, ss=ss, env=env, ws=ws, ts=cfg["strand_thick"] * cfg["thick"] * scale))
    return out


def weave_blocks(strands, cfg, uv_fn, rng):
    """Three closed swept strands -> Blocks. `uv_fn(tile, u, v)`: strand i uses lobe tile i (v along the strand)."""
    out = []
    for S in strands:
        sh = sweep(S["P"], S["N"], S["ws"], S["ts"], tip="blunt", rings=len(S["P"]), k_outer=int(cfg["strand_k"]),
                   outer_frac=0.5, bulge=0.5, width_fn=lambda sp, e=S["env"]: np.interp(sp, np.linspace(0, 1, len(e)), e),
                   thick_fn=lambda sp, e=S["env"]: np.interp(sp, np.linspace(0, 1, len(e)), e))
        u = strand_u(sh.q, sh.side)
        v = sh.arc / max(sh.length, 1e-9)
        out.append(Block("strand", sh.verts, sh.faces, uv_fn(S["i"], u, v), tag=dict(i=S["i"], v=v)))
    return out
