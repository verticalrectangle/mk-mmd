"""Rin's two black cat tails (a slice of the hair part, called by hair.py as `build_tails(ctx, fit, cfg, rig, pal)`).

Each tail is a tapered tube: radius ~2.4 cm at the root, a touch fuller over the first third like fur, thinning to ~6 mm at a
rounded tip. It is swept along a soft S-curve: it leaves the lower back backward and steeply up, bows outward, and its last
third turns up and back in. A few thin pointed clump spikes (12-20 mm) fan from the last 6-8 cm for a fluffy tip. Roots:
`ctx.land["tail_root.L" / ".R"]` (else `tail_root` offset in x by +-root_x); the body's `tail_normal` sets the first part of
the path. The tube runs a few centimetres into the body behind the root, rigid on 下半身.

Rig: `尻尾1_1..尻尾1_9` (left, +X) and `尻尾2_1..` (right), chains of kind "tail" hanging from 下半身 (the first joint partner is
the body's static lower-body collider). The chain starts at the first point of the path where its first capsule clears the
body's static colliders by `clear` (5 mm); every capsule is checked against them (`info["clearance"]`). The tube is skinned to
the chain; the spikes are rigid on the last two bones (every vertex of a spike takes the weights of its root).

Look: the ears' fur family (`colors.ears` outer + outer_sheen, the same toon ramp), plus one faint additive sheen sphere.
The path is set by knots of (elevation, heading): the tangent direction against the arc length, elevation above the horizontal,
heading from straight back (0) toward outward (90 deg; negative = back toward the midline); see DEFAULTS["path"].

Config `[hair.tails]`: DEFAULTS below."""
import numpy as np

from . import cat_common as CC
from .cat_tex import sheen_sphere, tail_fur
from .hair_geo import MeshAccum, Piece, arclen, resample, smoothstep, sweep, unit
from .. import spec as SP
from ..part import Material

DEFAULTS = {
    "enabled": True,
    "length": 0.58,             # m, from the skin at the root to the tip
    "radius_root": 0.024,       # m
    "radius_tip": 0.006,        # m, just before the rounded tip
    "taper": 0.85,              # exponent of the radius falling from root to tip (< 1: thins early, then slowly)
    "fluff": 0.22,              # extra radius (fraction) in the first third
    "bones": 9,                 # chain bones per tail
    "root_x": 0.022,            # m, root offset of each tail from the midline when the body publishes only `tail_root`
    "embed": 0.035,             # m, how far the tube runs into the body behind the root
    "sides": 12,                # vertices around the tube
    "ring_step": 0.022,         # m, ring spacing along the tube
    "path": {                   # tangent direction against the arc length (fractions of `length`), degrees
        "s": [0.0, 0.16, 0.34, 0.5, 0.64, 0.78, 0.9, 1.0],
        "elev": [54, 50, 38, 22, 8, -2, -12, -20],
        "head": [6.0, 22.0, 48.0, 64.0, 50.0, 14.0, -16.0, -30.0],
    },
    "normal_ref_deg": 34.0,     # elevation of `tail_normal` the knots above were drawn for (the real body's 34 deg)
    "normal_fade": 0.32,        # the first knots follow a different `tail_normal` over this fraction of the length
    "spikes": 4,                # pointed clump spikes fanning from the tip region per tail (0 = none)
    "spike_zone": [0.020, 0.072],   # m from the tip: where the first / last spike is rooted
    "spike_len": [0.012, 0.020],    # m, visible length of the shortest / longest spike
    "spike_fan_deg": [22.0, 50.0],  # angle off the tail's direction: the spike nearest the tip / the farthest from it
    "spike_width": 0.0075,      # m, full width of a spike where it leaves the tube
    "spike_thick": 0.0032,      # m
    "sheen": 0.16,              # peak of the additive sheen sphere map (0 = no sphere)
    "radius": 0.5,              # chain capsule radius as a fraction of the tube radius
    "clear": 0.005,             # m, required gap between every chain capsule and the body's static colliders
    "hold": 0.18,               # share of the first chain bone that stays rigid with 下半身
    "seed": "hair.tails",
}
SIDES = {1: ("尻尾1_", 1.0), 2: ("尻尾2_", -1.0)}
TIP_ROUND = 0.9                 # length of the rounded cap in radii of the tip
GOLDEN = np.radians(137.50776)


def _roots(ctx, cfg):
    """[(index, root point, direction hint (backward-up unit vector), sign)] for the two tails."""
    land = ctx.land
    body = ctx.parts.get("body")
    nrm = np.asarray(body.info.get("tail_normal", [0.0, 0.8, 0.5]), float) if body is not None else np.array([0.0, 0.8, 0.5])
    out = []
    for idx, (_, sg) in SIDES.items():
        key = "tail_root.L" if sg > 0 else "tail_root.R"
        if key in land:
            p = np.asarray(land[key], float)
        else:
            c = np.asarray(land["tail_root"], float)
            p = c + np.array([sg * float(cfg["root_x"]), 0.0, 0.0])
        out.append((idx, p, unit(nrm), sg))
    return out


def tail_path(cfg, root, sg, back, n=400, normal=None):
    """Dense centreline (n, 3) from the root, by integrating the tangent given by the knots of cfg["path"]. `normal` (the
    body's tail_normal) shifts the elevation of the first part of the tail by how much its own elevation differs from
    cfg["normal_ref_deg"], so the tail always leaves the skin at the same angle to the surface."""
    pa = cfg["path"]
    s_k = np.asarray(pa["s"], float)
    el = CC.pchip(s_k, np.radians(np.asarray(pa["elev"], float)))
    hd = CC.pchip(s_k, np.radians(np.asarray(pa["head"], float)))
    L = float(cfg["length"])
    s = np.linspace(0.0, 1.0, n)
    E, H = el(s), hd(s)
    if normal is not None:
        d_el = np.degrees(np.arcsin(np.clip(unit(normal)[2], -1.0, 1.0))) - float(cfg["normal_ref_deg"])
        E = E + np.radians(d_el) * (1.0 - smoothstep(s / max(float(cfg["normal_fade"]), 1e-6)))
    b = unit(np.array([back[0], back[1], 0.0]))
    lat = np.array([sg, 0.0, 0.0])
    d = (np.cos(E)[:, None] * (np.cos(H)[:, None] * b[None] + np.sin(H)[:, None] * lat[None]) +
         np.sin(E)[:, None] * np.array([0.0, 0.0, 1.0])[None])
    d = unit(d)
    seg = np.diff(s) * L
    P = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (d[1:] + d[:-1]) * seg[:, None], axis=0)], 0)
    return root[None] + P


def radius_profile(x, L, cfg):
    """Tube radius at arc length x (0 = skin root .. L = tip) and the rounded cap."""
    r0, r1 = float(cfg["radius_root"]), float(cfg["radius_tip"])
    cap = TIP_ROUND * r1
    x = np.asarray(x, float)
    u = np.clip(x, 0.0, L - cap) / max(L - cap, 1e-9)
    r = r0 + (r1 - r0) * u ** float(cfg["taper"])
    r = r * (1.0 + float(cfg["fluff"]) * np.sin(np.pi * np.clip(u / 0.62, 0.0, 1.0)) ** 1.3)
    tcap = np.clip((x - (L - cap)) / cap, 0.0, 1.0)
    return r * np.sqrt(np.clip(1.0 - tcap * tcap, 0.0, 1.0))


class Tube:
    """The centreline P (m, 3) from the root to the tip, extended straight back by `embed` (the stump inside the body), with
    rotation-minimising frames; `at(s)` gives centre, T, N, B at any arc length (s < 0 on the stump)."""

    def __init__(self, P, embed, n0):
        P = np.asarray(P, float)
        sa = arclen(P)
        self.L = float(sa[-1])
        self.embed = float(embed)
        t0 = unit(P[1] - P[0])
        self.P = np.vstack([P[0] - t0 * embed, P])
        self.s = np.concatenate([[-embed], sa])
        self.T, self.N, _ = CC.rmf(self.P, n0)

    def at(self, s):
        s = np.atleast_1d(np.asarray(s, float))

        def f(A):
            return np.stack([np.interp(s, self.s, A[:, k]) for k in range(3)], -1)
        C, T, N = f(self.P), unit(f(self.T)), unit(f(self.N))
        B = unit(np.cross(T, N))
        return C, T, unit(np.cross(B, T)), B


def tube_mesh(tube, cfg):
    """Closed tube: rings every ring_step, a rounded cap ending in one tip vertex, a fan closing the buried end. Returns
    verts, faces, uv (u 0 on the +N line .. 1 opposite, mirrored; v = arc / length)."""
    ns = int(cfg["sides"])
    L, step = tube.L, float(cfg["ring_step"])
    cap = TIP_ROUND * float(cfg["radius_tip"])
    body = np.arange(0.0, L - cap - 0.4 * step, step)
    ends = (L - cap) + cap * np.sin(np.linspace(0.0, 0.5 * np.pi, 5)[1:-1])
    s_rings = np.concatenate([[-tube.embed], body, [L - cap], ends])
    C, T, N, B = tube.at(s_rings)
    rad = np.where(s_rings < 0, float(cfg["radius_root"]), radius_profile(np.maximum(s_rings, 0.0), L, cfg))
    ang = 2.0 * np.pi * np.arange(ns) / ns
    ring = C[:, None, :] + rad[:, None, None] * (np.cos(ang)[None, :, None] * N[:, None, :] +
                                                  np.sin(ang)[None, :, None] * B[:, None, :])
    nr = len(s_rings)
    u = np.abs(((ang + np.pi) % (2.0 * np.pi)) - np.pi) / np.pi
    uv = np.stack([np.tile(u, nr), np.repeat(np.clip(s_rings / L, 0.0, 1.0), ns)], -1)
    tip = tube.at([L])[0][0]
    V = np.concatenate([ring.reshape(-1, 3), tip[None], C[:1]], 0)
    uv = np.concatenate([uv, [[0.5, 1.0]], [[0.5, 0.0]]], 0)
    tip_i, cen_i = nr * ns, nr * ns + 1
    faces = []
    for i in range(nr - 1):
        for j in range(ns):
            k = (j + 1) % ns
            faces.append([i * ns + j, i * ns + k, (i + 1) * ns + k, (i + 1) * ns + j])
    for j in range(ns):
        faces.append([(nr - 1) * ns + j, (nr - 1) * ns + (j + 1) % ns, tip_i])
        faces.append([cen_i, (j + 1) % ns, j])
    return V, faces, uv


def _spike_path(C, T, R, d0, rho, total):
    """Spike centreline: from just off the tube's axis along d0, bending toward the tail's direction T like fur swept back."""
    pts = [C + rho * R]
    for j in range(1, 6):
        t = (j - 0.5) / 5.0
        pts.append(pts[-1] + unit(d0 * (1.0 - 0.55 * t) + T * 0.55 * t) * total / 5.0)
    return np.array(pts)


def _exit_arc(P, tube, cfg):
    """(arc length at which the spike centreline P first leaves the tube, total length)."""
    Q = resample(P, 160)
    d = np.linalg.norm(Q[:, None, :] - tube.P[None], axis=2)
    near = d.argmin(1)
    out = d[np.arange(len(Q)), near] > radius_profile(np.maximum(tube.s[near], 0.0), tube.L, cfg)
    s = arclen(Q)
    return float(s[int(np.argmax(out))] if out.any() else 0.5 * s[-1]), float(s[-1])


def spike_specs(cfg, tube, rng, sg):
    """The tip spikes (a list of dicts), rooted at distances from the tip spread over cfg["spike_zone"] and rolled round the
    tail by the golden angle (mirrored for the right tail: sg = -1). Each starts inside the tube and leaves its surface at
    `fan` off the tail's direction; its total length is solved so that `visible` (the part outside the tube) is the
    designed 12-20 mm."""
    n = int(cfg["spikes"])
    if n <= 0:
        return []
    L = tube.L
    z0, z1 = (float(x) for x in cfg["spike_zone"])
    l0, l1 = (float(x) for x in cfg["spike_len"])
    a0, a1 = (np.radians(float(x)) for x in cfg["spike_fan_deg"])
    phi0 = rng.uniform(0.0, 2.0 * np.pi)
    out = []
    for k in range(n):
        f = k / max(n - 1, 1)
        s = L - (z0 + (z1 - z0) * f)
        C, T, N, B = (a[0] for a in tube.at([s]))
        r = float(radius_profile(s, L, cfg))
        phi = sg * (phi0 + k * GOLDEN + rng.normal(0.0, 0.12))
        R = np.cos(phi) * N + np.sin(phi) * B
        alpha = a0 + (a1 - a0) * f
        d0 = unit(np.cos(alpha) * T + np.sin(alpha) * R)
        vis = float(np.clip(l0 + (l1 - l0) * np.sin(np.pi * (k + 0.5) / n) + rng.normal(0.0, 0.0008), l0, l1))
        rho = 0.25 * r
        total = (r - rho) / max(float(np.sin(alpha)), 0.2) + vis
        for _ in range(4):                                   # the shape scales with `total`: a few rounds settle the exit
            P = _spike_path(C, T, R, d0, rho, total)
            s_exit, s_tot = _exit_arc(P, tube, cfg)
            total = s_exit + vis
        P = _spike_path(C, T, R, d0, rho, total)
        s_exit, s_tot = _exit_arc(P, tube, cfg)
        out.append(dict(P=P, hint=R, s=s, phi=phi, fan=alpha, visible=s_tot - s_exit, f_in=s_exit / s_tot, centre=C))
    return out


def build_tails(ctx, fit, cfg, rig, pal):
    cfg = SP.merge(DEFAULTS, cfg)
    if not cfg["enabled"]:
        return Piece(info={"enabled": False})
    col = CC.palette(ctx)
    tex = ctx.save_png("cat_tail_fur", tail_fur(col, ctx.rng_for(cfg["seed"] + ".fur")))
    toon = CC.toon(ctx, "toon_cat_fur", (0.55, 0.55, 0.62))
    sph = ctx.save_png("sphere_cat_sheen", sheen_sphere(col, float(cfg["sheen"]))) if float(cfg["sheen"]) > 0 else ""
    mat = Material("尻尾", name_en="cat tail", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                   ambient=(0.5, 0.5, 0.52), texture=tex, toon=toon, sphere=sph, sphere_mode="add" if sph else "none",
                   edge=True, edge_color=CC.darker(col["outer"], 0.45) + (1.0,), edge_size=0.8,
                   comment="cat tail fur: the ears' black with a cool sheen (generated)")
    acc = MeshAccum("cat_tails", [mat.name])
    colliders = CC.static_colliders(ctx)
    low = ctx.find_body("下半身")
    anchor_body = low.name if low is not None else None
    L = float(cfg["length"])
    nb = int(cfg["bones"])
    info = {"enabled": True, "bones": [], "chains": {}, "roots": {}, "clearance": {}, "spikes": {}, "length": L,
            "radius_tip": float(cfg["radius_tip"]), "radius_root": float(cfg["radius_root"])}
    frames = {"尻尾": []}
    for idx, root, nrm, sg in _roots(ctx, cfg):
        base, _ = SIDES[idx]
        back = unit(np.array([nrm[0], nrm[1], 0.0]))
        P = tail_path(cfg, root, sg, back, normal=nrm)
        # ---- chain: from the first point that lets the first capsule clear the colliders, to the tip, equal bones
        sa = arclen(P)
        s_chain0 = 0.0
        for s0 in np.arange(0.0, 0.12, 0.002):
            q0 = np.array([np.interp(s0, sa, P[:, k]) for k in range(3)])
            q1 = np.array([np.interp(s0 + (L - s0) / nb, sa, P[:, k]) for k in range(3)])
            gap, _ = CC.capsule_clearance(colliders, q0, q1, float(cfg["radius"]) * float(radius_profile(s0, L, cfg)), step=0.004)
            s_chain0 = float(s0)
            if gap >= float(cfg["clear"]) + 0.001 or not colliders:
                break
        s_pts = np.linspace(s_chain0, L, nb + 1)
        pts = np.stack([np.interp(s_pts, sa, P[:, k]) for k in range(3)], -1)
        radius = float(cfg["radius"]) * radius_profile(0.5 * (s_pts[:-1] + s_pts[1:]), L, cfg)
        ch = rig.chain(base, pts, anchor="下半身", kind="tail", radius=radius, anchor_body=anchor_body, hold=cfg["hold"],
                       name_en=f"tail{idx}_")
        # ---- tube mesh, skinned to the chain
        tube = Tube(P, float(cfg["embed"]), np.array([sg * 0.6, 0.0, 1.0]))
        V, F, UV = tube_mesh(tube, cfg)
        acc.add(V, F, UV, mat=0, weights=ch.weights(V))
        # ---- tip spikes: thin pointed shells, rigid on the bones their roots sit on (the last two)
        sp_info = []
        for sp in spike_specs(cfg, tube, ctx.rng_for(cfg["seed"] + ".spikes"), sg):
            st = sweep(sp["P"], sp["hint"], float(cfg["spike_width"]), float(cfg["spike_thick"]), tip="point",
                       tip_start=sp["f_in"], tip_power=1.15, rings=5, k_outer=3, outer_frac=0.5, bulge=0.8,
                       close_root=True)
            n = len(st.verts)
            w = {b: np.full(n, float(a[0])) for b, a in ch.weights(sp["centre"][None]).items()}
            u = abs(((sp["phi"] + np.pi) % (2.0 * np.pi)) - np.pi) / np.pi
            uv = np.stack([np.full(n, u), np.clip((sp["s"] + st.arc) / L, 0.0, 1.0)], -1)
            off = acc.add(st.verts, st.faces, uv, mat=0, weights=w)
            tip = st.verts[-1]
            near = int(np.argmin(np.linalg.norm(tube.P - tip, axis=1)))
            out = float(np.linalg.norm(tube.P[near] - tip) - radius_profile(max(tube.s[near], 0.0), L, cfg))
            sp_info.append({"verts": (off, n), "root": sp["P"][0], "tip": tip, "visible": sp["visible"],
                            "tip_out": out, "s": sp["s"], "fan_deg": float(np.degrees(sp["fan"]))})
        gaps = CC.chain_clearance(colliders, pts, radius) if colliders else []
        info["bones"] += list(ch.names)
        info["chains"][idx] = pts.copy()
        info["roots"][idx] = {"root": root, "normal": nrm, "chain_start": float(s_chain0), "side": sg}
        info["clearance"][idx] = [g for g, _ in gaps]
        info["spikes"][idx] = sp_info
        frames["尻尾"] += list(ch.names)
    return Piece(meshes=[acc.to_mesh()], materials=[mat], info=info, frames=frames)
