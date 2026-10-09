"""Legs of Rin's outfit: the black ribbon wound round the left calf (bow near the ankle, two dynamic tails) and the black
Mary-Jane shoes (both feet: upper, strap with buckle and bow, rolled collar, platform sole). numpy only (Blender-safe).

    leg_ribbon(fit, cfg, soup_satin, rig, anchor_bodies=None, colliders=None) -> dict
    shoes(fit, cfg, soup_shoes, soup_satin, rig=None) -> dict

Everything is fitted to the body SURFACE (`fit.skin`: sections by rays, exact closest-point queries) and a few landmarks
(`knee.L`, `ankle.L`, `toe.L`, `toe_end.L`), never to a particular foot shape. The shoe is a grid of meridians round the ankle
axis laid on the foot (welt row at the underside .. top line = collar round the ankle + throat tongue on the instep), offset
by `ease` along the smooth skin normal, then pushed out until vertices AND quad centres/edges are >= `min_gap` from the skin.
The right shoe is the same code run on the mirrored right foot, so asymmetric bodies work too. The sole stands on the floor
plane (the `toe_end` landmark, z = 0), flat, so it is flat whenever the foot is.

`cfg` is the `[outfit.legs]` table; DEFAULTS below is merged under `cfg["ribbon"]` / `cfg["shoe"]` (nested tables merge
key by key). Sizes are metres at S = 1 and are multiplied by `fit.S` (S = head_tip.z / 1.7 = 0.793 for Rin); angles are degrees.

ribbon  enabled true (false: the outfit has no ribbon), width 0.0145 (band, 1.15 cm at S), turns 4.5, span [0.0935, 0.877]
        (fractions of the way ankle -> knee, = the guides
        calf_ribbon_z [0.124, 0.379] for Rin; `z` = [z0, z1] overrides), offset 0.004 (height above the skin, 3.2 mm) + cup
        0.0005 (edges), handedness +1 (counter-clockwise going up), segments 26 per turn, cols 5; bow_angle 40 (azimuth of the
        knot, 0 = front, +90 = her left), bow_size/loop/loop_w/ribbon_w/droop/lift = `outfit_geo.bow`; tail_len 0.040 (2 chains
        A, B of tail_bones 3 bones named リボン脚A1.., parent 左ひざ), tail_splay 13, tail_out 0.0065 (lift of the tips),
        tail_radius, tail_width, tail_notch, tail_margin 0.0005 (clearance of the tail bodies from the body's colliders).
        With `colliders` (the body part's static RigidBodies) every tail is lifted along the skin normal until its
        non-root bodies clear all of them by tail_margin, so they keep colliding with the body group (a body that still
        overlaps, or no `colliders` at all, gets group 0 added to its no_collide: it cannot be proven clear).
shoe    columns 96 x rows 11 (the upper), ease 0.0038 (leather to skin) / min_gap 0.0032 (smallest allowed, x S), welt_h 0.0055 (welt
        row above the floor of the foot), footprint_h 0.030 (samples this low on the foot make the footprint the welt hangs
        from), close 0.004 (closing radius: the first-hit radius is the farthest over the neighbouring meridian planes, so the
        leather bridges the gaps between modelled toes), relax 12 (smooth + project iterations) / normal_smooth 8, toe_room
        0.012 over toe_len 0.055, collar {az, h} (height of the top line above the ankle joint by azimuth), collar_flare,
        collar_r (roll radius) / sides / stride, throat {start, end (fractions of the ankle -> toe distance), w_start, w_end
        (half widths)}, top_smooth; sole: lip (+ lip_toe, lip_heel, lip_drop, lip_t) = ledge out from the welt, band/band_step
        = outsole band, edge_r, spring (toe spring), sole_z (None = toe_end z); strap {pos (fraction of ankle -> toe), width,
        thick, arc, points, profile}, buckle {at, width, height, wire}, bow {size, loop, loop_w, ribbon_w, tail_len, droop, lift}.

Weights (shoes): upper, collar, sole and strap take the skin weights of the nearest foot surface point from the donors 左/右
ひざ, 足首, 足つま先 (the forefoot skin is weighted on the toe bone, so the toe box follows bending toes and the sole flexes at
the ball); buckle and bows are rigid with the weights under their middle. Leg ribbon: ひざ / 足首 (the shin's own).

Conventions: theta = azimuth round the foot, 0 = towards the toe, + towards her left (outer side of the left foot); weights are
the body's own (donors restricted to the shin and foot bones of that side); texture: satin u = metres / 0.04 along the ribbon,
v 0..1 across; leather and sole about metres / 0.10 (tileable)."""
import numpy as np

from .. import skin as SK
from . import outfit_geo as G
from . import outfit_rig as RG

TAU = G.TAU

DEFAULTS = {
    "ribbon": {
        "enabled": True,
        "width": 0.0145, "turns": 4.5, "z": None, "span": [0.0935, 0.8770], "offset": 0.0040, "cup": 0.0005,
        "handedness": 1.0, "segments": 26, "cols": 5,
        "bow_angle": 40.0, "bow_size": 0.0190, "bow_loop": 1.05, "bow_loop_w": 0.62, "bow_ribbon_w": 0.62,
        "bow_droop": 0.10, "bow_lift": 0.0040,
        "tail_len": 0.040, "tail_splay": 13.0, "tail_out": 0.0065, "tail_bones": 3, "tail_radius": 0.0042,
        "tail_width": 0.0105, "tail_notch": 0.22, "tail_margin": 0.0005,
    },
    "shoe": {
        "columns": 96, "rows": 11, "pivot": 0.50, "top_smooth": 6, "relax": 12, "normal_smooth": 8, "close": 0.0040, "footprint_h": 0.0300,
        "ease": 0.0038, "min_gap": 0.0032, "welt_h": 0.0055,
        "toe_room": 0.0120, "toe_len": 0.055,
        "collar": {"az": [0.0, 52.0, 70.0, 90.0, 110.0, 135.0, 180.0],
                   "h": [-0.012, -0.012, -0.006, 0.000, 0.004, 0.008, 0.010]},
        "collar_flare": 0.0020, "collar_r": 0.0030, "collar_sides": 6, "collar_stride": 2,
        "throat": {"start": 0.17, "end": 0.54, "w_start": 0.026, "w_end": 0.0175},
        "lip": 0.0030, "lip_toe": 0.0050, "lip_heel": 0.0030, "lip_drop": 0.0018, "lip_t": 0.0018, "band": 0.0070, "band_step": 0.0007, "edge_r": 0.0030,
        "sole_z": None, "spring": 0.0045,
        "strap": {"pos": 0.36, "width": 0.0140, "thick": 0.0020, "arc": [25.0, 155.0], "points": 24, "profile": 10},
        "buckle": {"at": 48.0, "width": 0.0125, "height": 0.0115, "wire": 0.0012},
        "bow": {"size": 0.0190, "loop": 1.0, "loop_w": 0.62, "ribbon_w": 0.60, "tail_len": 1.0, "droop": 0.10, "lift": 0.0005},
    },
}


def merge(base, over):
    """Recursive dict merge (over wins)."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _u(v):
    return G.unit(v)


# ---------------------------------------------------------------- body regions and ray casting
def leg_skin(fit, side="L", z_max=0.6, name=None):
    """The skin triangles of one leg: centroid on that side of x = 0 and below z_max (shares the vertex array)."""
    sk = fit.skin
    cen = sk.verts[sk.tris].mean(1)
    sx = 1.0 if side == "L" else -1.0
    ok = (cen[:, 0] * sx > 0.0) & (cen[:, 2] < z_max)
    return sk.where(None, tri_mask=ok, name=name or f"leg_{side}")


def mirror_skin(sk):
    """The x-mirrored surface of a Skin (windings reversed so that the outside stays outside)."""
    from .outfit_fit import Skin
    out = Skin.__new__(Skin)
    out.verts = sk.verts * np.array([-1.0, 1.0, 1.0])
    out.tris = sk.tris[:, ::-1].copy()
    out.src = sk.src
    out.weights = {}
    out.name = sk.name + "_mirror"
    out._flip = sk._flip
    return out


def vertex_normals(sk):
    """Outward unit normals (nv, 3) of a Skin, area-weighted over its triangles (zero for unused vertices)."""
    V, T = sk.verts, sk.tris
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]) * (-1.0 if sk._flip else 1.0)
    out = np.zeros_like(V)
    for i in range(3):
        np.add.at(out, T[:, i], fn)
    return G.unit(out)


def _tri_closest(p, a, b, c):
    """Closest points of p (m, 3) on the triangles (a, b, c) (pairwise): (q (m, 3), barycentric (m, 3), squared distance (m,))."""
    ab, ac, ap = b - a, c - a, p - a
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
    den = d00 * d11 - d01 * d01
    good = den > 1e-15 * d00 * d11
    safe = np.where(good, den, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    bary = np.stack([1.0 - v - w, v, w], 1)
    out = np.flatnonzero(~(good & (bary >= 0.0).all(1)))
    if out.size:
        pa, aa, bb = p[out], a[out], b[out]
        e = (ab[out], ac[out], c[out] - bb)
        org = (aa, aa, bb)
        ts, fs = [], []
        for o, ev in zip(org, e):
            l2 = (ev * ev).sum(1)
            t = np.clip(((pa - o) * ev).sum(1) / np.where(l2 > 0.0, l2, 1.0), 0.0, 1.0)
            r = pa - o - t[:, None] * ev
            ts.append(t)
            fs.append((r * r).sum(1))
        which = np.argmin(np.stack(fs), axis=0)
        eb = np.zeros((out.size, 3))
        eb[:, 0] = np.where(which == 0, 1.0 - ts[0], np.where(which == 1, 1.0 - ts[1], 0.0))
        eb[:, 1] = np.where(which == 0, ts[0], np.where(which == 2, 1.0 - ts[2], 0.0))
        eb[:, 2] = np.where(which == 1, ts[1], np.where(which == 2, ts[2], 0.0))
        bary[out] = eb
    q = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    r = p - q
    return q, bary, (r * r).sum(1)


class Surface:
    """Exact closest-point queries against the triangles of a (small) Skin, numpy only: triangles are pruned with their
    centroid and bounding radius, so a query costs about its distance to the surface / the triangle size, not the number
    of triangles. Also holds the smooth (area weighted) vertex normals of the surface."""

    def __init__(self, sk, chunk=256):
        self.sk, self.chunk = sk, chunk
        V, T = sk.verts, sk.tris
        self.A, self.B, self.C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        self.cen = (self.A + self.B + self.C) / 3.0
        self.rho = np.sqrt(np.max([((x - self.cen) ** 2).sum(1) for x in (self.A, self.B, self.C)], axis=0))
        self.nrm = _u(np.cross(self.B - self.A, self.C - self.A)) * (-1.0 if sk._flip else 1.0)
        self.NV = vertex_normals(sk)
        self.tris = T

    def closest(self, pts):
        """(signed distance (n,), closest point (n, 3), triangle (n,), barycentric (n, 3)); negative = behind the surface
        (as `Skin.closest`)."""
        P = np.asarray(pts, float).reshape(-1, 3)
        n = len(P)
        dist, qs, tri, bary = np.empty(n), np.empty((n, 3)), np.empty(n, int), np.empty((n, 3))
        c2 = (self.cen ** 2).sum(1)
        for s in range(0, n, self.chunk):
            Pc = P[s:s + self.chunk]
            dc = np.sqrt(np.maximum((Pc ** 2).sum(1)[:, None] - 2.0 * Pc @ self.cen.T + c2[None, :], 0.0))
            mask = dc - self.rho[None, :] <= dc.min(1)[:, None] + 1e-9
            pi, ti = np.nonzero(mask)
            q, b, d2 = _tri_closest(Pc[pi], self.A[ti], self.B[ti], self.C[ti])
            counts = mask.sum(1)
            start = np.cumsum(counts) - counts
            low = np.minimum.reduceat(d2, start)
            hit = np.flatnonzero(d2 == np.repeat(low, counts))
            gid = np.repeat(np.arange(len(counts)), counts)[hit]
            first = hit[np.concatenate([[True], gid[1:] != gid[:-1]])]
            sl = slice(s, s + len(Pc))
            qs[sl], tri[sl], bary[sl] = q[first], ti[first], b[first]
            side = ((Pc - q[first]) * self.nrm[ti[first]]).sum(1)
            dist[sl] = np.sqrt(d2[first]) * np.where(side < 0.0, -1.0, 1.0)
        return dist, qs, tri, bary

    def normal_at(self, tri, bary):
        """Interpolated smooth outward normals at points given as (triangle, barycentric)."""
        return _u((self.NV[self.tris[tri]] * bary[..., None]).sum(1))


def push_out(srf, pts, dmin, iters=3):
    """Move points that are closer than `dmin` to the surface (or inside it) outwards along the line from the closest
    surface point until their signed distance is dmin. Returns (points, signed distances before)."""
    pts = np.array(pts, float)
    d0 = None
    for _ in range(iters):
        d, q, tri, _ = srf.closest(pts)
        d0 = d if d0 is None else d0
        bad = d < dmin - 1e-6
        if not bad.any():
            break
        direction = np.where((np.abs(d) > 1e-4)[:, None], _u(pts - q) * np.sign(d)[:, None], srf.nrm[tri])
        pts[bad] += direction[bad] * (dmin - d[bad])[:, None]
    return pts, d0


# ---------------------------------------------------------------- cylinder-like table of a limb
class LimbTable:
    """Radial profile table of a limb round an axis: sections of the skin by planes perpendicular to `axis` at the
    stations s (arc length from `origin` up the axis), `m` rays from the section centre; bilinear lookups."""

    def __init__(self, sk, origin, axis, ref, s_lo, s_hi, n=48, m=72, rmin=0.004, rmax=0.09):
        self.axis = _u(axis)
        u0 = np.asarray(ref, float) - self.axis * np.dot(ref, self.axis)
        self.u = _u(u0)
        self.v = np.cross(self.axis, self.u)                   # u x v = axis (counter-clockwise seen from above)
        self.m = m
        self.s = np.linspace(s_lo, s_hi, n)
        cen, rad = [], []
        last = None
        for s in self.s:
            c0 = np.asarray(origin, float) + self.axis * s
            if last is not None:
                c0 = c0 + (last - (np.asarray(origin, float) + self.axis * (s - (self.s[1] - self.s[0]))))
            c, r = sk.section(c0, self.u, self.v, m, "near", rmin=rmin, rmax=rmax, iters=2)
            if r is None:
                raise ValueError("limb section failed: no surface near the axis")
            last = c
            cen.append(c)
            rad.append(r)
        self.c = np.array(cen)
        self.r = np.array(rad)

    def _locate(self, s, alpha):
        ds = self.s[1] - self.s[0]
        f = np.clip((np.asarray(s, float) - self.s[0]) / ds, 0.0, len(self.s) - 1.000001)
        i = f.astype(int)
        a = (np.asarray(alpha, float) % TAU) / TAU * self.m
        j = a.astype(int) % self.m
        return i, f - i, j, (j + 1) % self.m, a - np.floor(a)

    def radius(self, s, alpha):
        i, fs, j0, j1, fa = self._locate(s, alpha)
        r0 = self.r[i, j0] * (1 - fa) + self.r[i, j1] * fa
        r1 = self.r[i + 1, j0] * (1 - fa) + self.r[i + 1, j1] * fa
        return r0 * (1 - fs) + r1 * fs

    def centre(self, s):
        i, fs, _, _, _ = self._locate(s, np.zeros_like(np.asarray(s, float)))
        return self.c[i] * (1 - fs)[..., None] + self.c[i + 1] * fs[..., None]

    def dir(self, alpha):
        a = np.asarray(alpha, float)
        return np.cos(a)[..., None] * self.u + np.sin(a)[..., None] * self.v

    def point(self, s, alpha, off=0.0):
        """Point at height off above the skin (measured along the radial direction from the centre line)."""
        return self.centre(s) + self.dir(alpha) * (self.radius(s, alpha) + off)[..., None]

    def normal(self, s, alpha, ds=2e-3, da=0.03):
        """Outward unit normal of the surface at (s, alpha), from finite differences of the table."""
        ps = self.point(s + ds, alpha) - self.point(s - ds, alpha)
        pa = self.point(s, alpha + da) - self.point(s, alpha - da)
        n = np.cross(pa, ps)
        n = _u(n)
        out = self.dir(alpha)
        return np.where((np.einsum("...k,...k->...", n, out) < 0)[..., None], -n, n)


# ---------------------------------------------------------------- leg ribbon
def uv_length(path, tile=0.04):
    """Texture coordinate along a polyline counting tiles of `tile` metres (satin: 0.04)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]) / tile


def skin_weights(fit, pts, side="L", faces=None, smooth=0):
    """Weights of the shin and foot bones of one side, transferred from the nearest skin surface."""
    pre = "左" if side == "L" else "右"
    donors = (f"{pre}ひざ", f"{pre}足首")
    return fit.weights_for(pts, donors, default=f"{pre}ひざ", faces=faces, smooth=smooth)


def _band_grid(tab, s_c, alpha, hw, fr, off, cup):
    """Vertices (N, ncols, 3) of a band wound round a limb: centre line (s_c, alpha) on the surface offset `off`, the
    edges at +-hw measured across the path in the tangent plane (fr = -1..1 across the band), the borders lifted by
    `cup` (quadratically). Returns (grid, centres, offsets (N, ncols))."""
    P = tab.point(s_c, alpha, off)
    T = _u(np.gradient(P, axis=0))
    n = tab.normal(s_c, alpha)
    across = _u(np.cross(n, T))
    across = across * np.where(np.einsum("ij,j->i", across, tab.axis) < 0, -1.0, 1.0)[:, None]
    e_a = _u(np.cross(tab.axis, tab.dir(alpha)))                              # round the limb, counter-clockwise
    R = tab.radius(s_c, alpha) + off
    ds = np.einsum("ij,j->i", across, tab.axis)[:, None] * hw * fr[None, :]
    da = (np.einsum("ij,ij->i", across, e_a) / R)[:, None] * hw * fr[None, :]
    S_g = s_c[:, None] + ds
    A_g = alpha[:, None] + da
    O_g = off + cup * fr[None, :] ** 2 + 0.0 * S_g
    return tab.point(S_g, A_g, O_g), tab.centre(S_g), O_g


def leg_ribbon(fit, cfg, soup_satin, rig, anchor_bodies=None, colliders=None):
    """The ribbon wound in a spiral round the left calf, tied in a bow near the ankle with two dynamic tails.

    Adds the spiral (satin material 1, weights of the shin), the bow's loops and knot (material 0, weights of the shin)
    and the two tails (material 0, weights of their chain bones, fading into the shin at the knot) to `soup_satin`; the
    tail chains (リボン脚A1.., リボン脚B1.., parent 左ひざ, kind "ribbon") to `rig` (skipped when `rig` is None).
    `colliders`: the body's static RigidBodies; the tails are lifted off them (see the module docstring).

    Returns a dict: band (Patch), grid (N, cols, 3), offsets (measured distance of every band vertex to the skin),
    min_offset, max_offset, z_range (measured z of the band vertices), turns, path, axis, table, bow (origin, normal,
    right, up, size), tails (patches), tail_bones {"A": [...], "B": [...]}, tail_points {"A": (n+1, 3), "B": ...},
    tail_lift {"A": m, "B": m} (extra lift along the skin normal that clears the colliders) and tail_clearance (smallest
    clearance of a non-root tail body from the colliders, m; None without `colliders`)."""
    S, L = fit.S, fit.L
    c = merge(DEFAULTS["ribbon"], (cfg or {}).get("ribbon"))
    knee, ankle = L["knee.L"], L["ankle.L"]
    axis = _u(knee - ankle)
    az = max(float(axis[2]), 1e-6)
    leg = leg_skin(fit, "L", z_max=float(knee[2]) + 0.12, name="shin_L")
    leg_srf = Surface(leg)
    if c.get("z"):
        z0, z1 = float(c["z"][0]), float(c["z"][1])
    else:
        z0, z1 = (float(ankle[2] + f * (knee[2] - ankle[2])) for f in c["span"])
    s_of_z = lambda z: (z - float(ankle[2])) / az
    width, off, cup = c["width"] * S, c["offset"] * S, c["cup"] * S
    hw = 0.5 * width
    s_tab = 0.004                                              # the sections stay above the ankle joint (the foot begins)
    tab = LimbTable(leg, ankle, axis, np.array([0.0, -1.0, 0.0]), s_tab, s_of_z(z1) + hw + 0.02, n=64, m=72, rmin=0.004,
                    rmax=0.075)
    # ---- spiral band, bottom end at the bow
    turns = float(c["turns"])
    hand = 1.0 if c["handedness"] >= 0 else -1.0
    a_bow = np.radians(float(c["bow_angle"]))
    s_lo, s_hi = s_of_z(z0) + hw / az, s_of_z(z1) - hw / az
    tail_len = min(c["tail_len"] * S, s_lo - s_tab - 0.006)
    nseg = int(round(turns * c["segments"]))
    t = np.linspace(0.0, 1.0, nseg + 1)
    s_c = s_lo + (s_hi - s_lo) * t
    alpha = a_bow + hand * TAU * turns * t
    ncols = int(c["cols"])
    fr = np.linspace(-1.0, 1.0, ncols)
    grid, ctr, O_g = _band_grid(tab, s_c, alpha, hw, fr, off, cup)
    flat, cflat, tgt = grid.reshape(-1, 3), ctr.reshape(-1, 3), O_g.reshape(-1)
    for _ in range(3):                                           # make the measured height above the real skin exact
        d, _, _, _ = leg_srf.closest(flat)
        flat = flat + _u(flat - cflat) * (tgt - d)[:, None]
    grid = flat.reshape(grid.shape)
    d_fin = leg_srf.closest(flat)[0].reshape(grid.shape[:2])
    path = grid[:, ncols // 2, :]
    u_arc = uv_length(path)
    vv = np.linspace(0.0, 1.0, ncols)
    quads, uv = [], []
    for k in range(nseg):
        for j in range(ncols - 1):
            quads.append((k * ncols + j, (k + 1) * ncols + j, (k + 1) * ncols + j + 1, k * ncols + j + 1))
            uv += [(u_arc[k], vv[j]), (u_arc[k + 1], vv[j]), (u_arc[k + 1], vv[j + 1]), (u_arc[k], vv[j + 1])]
    band = G.Patch(flat.copy(), quads, np.array(uv), None, 1, "leg_ribbon")
    band = G.orient_outward(band, cflat)
    band.w = skin_weights(fit, band.v, "L")
    soup_satin.add(band)
    # ---- the bow: knot and loops sit on the lower end of the band, the tails are separate soft chains
    n_b = tab.normal(s_lo, alpha[0])
    up_b = _u(axis - n_b * np.dot(axis, n_b))
    right = _u(np.cross(up_b, n_b))
    size = c["bow_size"] * S
    lift = c["bow_lift"] * S
    O_bow = tab.point(s_lo, alpha[0], off + lift)
    parts = G.bow(O_bow, right, up_b, n_b, size=size, loop_len=c["bow_loop"], loop_w=c["bow_loop_w"],
                  ribbon_w=c["bow_ribbon_w"], tail_len=1.0, droop=c["bow_droop"], mat=0, tag="leg_bow", seed=3)
    for q in metric_bow_uv(parts[:3]):                           # two loops and the knot (the tails are rebuilt below)
        q.w = skin_weights(fit, q.v, "L")
        soup_satin.add(q)
    # ---- tails: strips along the chain curve; the chain lifts off the skin a little towards the tips
    nb = int(c["tail_bones"])
    beta = np.radians(float(c["tail_splay"]))
    d0, d1 = off + lift, max(c["tail_out"] * S, off + lift)
    shin = "左ひざ"
    r_tail, tail_margin = c["tail_radius"] * S, c["tail_margin"] * S
    tails, bones, pts_all, lifts, clearance = [], {}, {}, {}, []
    for ch, sg in (("A", 1.0), ("B", -1.0)):
        def curve(tt, extra=0.0, sg=sg):
            tt = np.asarray(tt, float)
            s_t = s_lo - tail_len * tt * np.cos(beta) + 0.0 * tt
            lat = sg * tail_len * tt * np.sin(beta)
            a_t = alpha[0] + lat / (tab.radius(s_t, alpha[0]) + d0)
            d_t = d0 + extra + (d1 - d0) * tt ** 1.3
            return tab.point(s_t, a_t, d_t), s_t, a_t
        extra = 0.0
        if colliders:                                  # lift the tail off the body's colliders until its bodies clear them
            for _ in range(8):
                kp, _, _ = curve(np.linspace(0.0, 1.0, nb + 1), extra)
                short = tail_margin - float(RG.chain_clearance(kp, r_tail, colliders)[1:].min())   # the root body ignores group 0
                if short <= 0.0:
                    break
                extra += short + 0.0003
        rows = 12
        tt = np.linspace(0.0, 1.0, rows)
        path_t, s_t, a_t = curve(tt, extra)
        nrm = np.array([tab.normal(float(s_), float(a_)) for s_, a_ in zip(s_t, a_t)])
        tang = _u(np.gradient(path_t, axis=0))
        side = _u(np.cross(nrm, tang))
        hwt = 0.5 * c["tail_width"] * S * (1.0 - 0.15 * tt)
        p = G.strip(path_t, hwt, side, ncols=3, mat=0, tag=f"leg_bow_tail_{ch}", u=uv_length(path_t))
        mid = (rows - 1) * 3 + 1
        p.v[mid] -= tang[-1] * (c["tail_notch"] * c["tail_width"] * S * 1.2)
        p = G.orient_outward(p, np.repeat(tab.centre(s_t), 3, axis=0))
        names = [f"リボン脚{ch}{k + 1}" for k in range(nb)]
        kp, _, _ = curve(np.linspace(0.0, 1.0, nb + 1), extra)
        seg = float(np.linalg.norm(np.diff(kp, axis=0), axis=1).mean())
        wch = SK.chain_weights(p.v, names, kp, blend=0.6 * seg)
        par, _ = SK.chain_param(p.v, kp)
        root = 1.0 - G.smoothstep(par / 0.7)
        ws = skin_weights(fit, p.v, "L")
        w = {b: a * (1.0 - root) for b, a in wch.items()}
        for b, a in ws.items():
            w[b] = w.get(b, 0.0) + a * root
        p.w = G.cap_weights(w, len(p.v), 4)
        soup_satin.add(p)
        tails.append(p)
        bones[ch] = names
        pts_all[ch] = kp
        lifts[ch] = extra
        if colliders:
            clear = RG.chain_clearance(kp, r_tail, colliders)
            clearance.append(float(clear[1:].min()))
        if rig is not None:
            ab = (anchor_bodies or {}).get(shin)
            rig.chain(names, kp, shin, "ribbon", r_tail, anchor_body=ab, x_hint=right,
                      name_en=[f"leg_ribbon_{ch}{k + 1}" for k in range(nb)])
            for k, body in enumerate(rig.bodies[-nb:]):
                if not colliders or (k and clear[k] < 0.0):    # not provably clear of the body: ignore its group
                    body.no_collide = tuple(sorted(set(body.no_collide) | {0}))
    return {"band": band, "grid": grid, "offsets": d_fin, "min_offset": float(d_fin.min()), "max_offset": float(d_fin.max()),
            "z_range": (float(grid[..., 2].min()), float(grid[..., 2].max())), "turns": turns, "path": path, "axis": axis,
            "table": tab, "bow": {"origin": O_bow, "normal": n_b, "right": right, "up": up_b, "size": size},
            "tails": tails, "tail_bones": bones, "tail_points": pts_all, "tail_lift": lifts,
            "tail_clearance": min(clearance) if clearance else None}

# ---------------------------------------------------------------- shoes
def _poly_len(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _first_cross(vals, thresh, below):
    """Fractional index of the first sample (from index 0) with vals <= thresh (below) or >= thresh; None if never."""
    idx = np.nonzero(vals <= thresh if below else vals >= thresh)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return 0.0
    a, b = vals[i - 1], vals[i]
    return (i - 1) + ((thresh - a) / (b - a) if b != a else 0.0)


def _resample_closed(P, n):
    """n points spaced equally by arc length along the closed polyline P (k, d)."""
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    Pc = np.vstack([P, P[:1]])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, Pc[:, k]) for k in range(P.shape[1])], 1)


def smooth_grid(P, iters=2, lam=0.5, closed_j=True):
    """Laplacian smoothing of a (J, K, 3) grid along j (periodic when closed_j) and k (end rows stay in k)."""
    P = np.array(P, float)
    for _ in range(iters):
        Pj = 0.5 * (np.roll(P, 1, 0) + np.roll(P, -1, 0)) if closed_j else P
        Q = P + lam * (Pj - P)
        R = Q.copy()
        R[:, 1:-1] = Q[:, 1:-1] + lam * (0.5 * (Q[:, :-2] + Q[:, 2:]) - Q[:, 1:-1])
        P = R
    return P


def grid_normals(P, inside):
    """Outward unit normals (J, K1, 3) of a grid closed in j, from its own tangents; `inside` (3,) is a point inside."""
    dj = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    dk = np.gradient(P, axis=1)
    n = _u(np.cross(dj, dk))
    if np.einsum("jki,jki->", n, P - inside) < 0:
        n = -n
    return n


def relax_grid(srf, P, dmin, iters=12, lam=0.5):
    """Alternate Laplacian smoothing of the grid P (J, K1, 3) and the projection out of the surface (points closer than
    `dmin` go back out). The shell bridges concavities of the skin (the gaps between modelled toes, the web of the foot)
    instead of following them, and keeps `dmin` from the skin on every convex part."""
    P = np.array(P, float)
    shape = P.shape
    for _ in range(int(iters)):
        P = smooth_grid(P, 1, lam)
        P, _ = push_out(srf, P.reshape(-1, 3), dmin, iters=2)
        P = P.reshape(shape)
    return P


def surface_offset(srf, pts, dist):
    """Points moved `dist` (scalar or per point) along the smooth surface normal of their closest point on a `Surface`.
    Returns (new points, normals, closest points)."""
    d, q, tri, bary = srf.closest(pts)
    n = srf.normal_at(tri, bary)
    return q + n * np.asarray(dist, float).reshape(-1, 1) if np.ndim(dist) else q + n * dist, n, q


class Foot:
    """One foot in left-like space (the right foot is mirrored to x > 0): the surface of the leg and foot below the shoe's
    collar, its landmarks and a few measurements. Sections are taken with rays against the skin triangles."""

    def __init__(self, fit, side, cut=0.16):
        L, S = fit.L, fit.S
        self.side, self.S = side, S
        self.sg = 1.0 if side == "L" else -1.0

        def lm(name):
            p = np.array(L[f"{name}.{side}"], float)
            p[0] *= self.sg
            return p

        self.ankle, self.toe, self.toe_end, self.knee = (lm(n) for n in ("ankle", "toe", "toe_end", "knee"))
        sk = leg_skin(fit, side, z_max=float(self.ankle[2]) + cut)
        self.skin = mirror_skin(sk) if side == "R" else sk
        self.srf = Surface(self.skin)
        V = self.skin.verts[self.skin.tris].reshape(-1, 3)
        self.floor = float(V[:, 2].min())
        low = V[V[:, 2] < float(self.ankle[2])]
        self.y_tip = float(low[:, 1].min())
        self.y_heel = float(low[:, 1].max())
        cen, _ = self.skin.section(np.array([self.ankle[0], self.ankle[1], self.ankle[2] + 0.03 * S]),
                                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 48, "near", rmin=0.004,
                                   rmax=0.06, iters=2)
        self.c_a = np.array(cen[:2] if cen is not None else self.ankle[:2], float)
        self.r_toe = float(self.c_a[1] - self.y_tip)
        lowx = V[V[:, 2] < self.floor + 0.022 * S][:, 0]
        self.width = float(lowx.max() - lowx.min())             # plan width of the foot near the floor
        self.length = float(self.y_heel - self.y_tip)
        self.name = "左" if side == "L" else "右"

    def to_world(self, P):
        """Left-like points (n, 3) -> model space of this foot."""
        P = np.array(P, float)
        P[..., 0] *= self.sg
        return P

    def patch_world(self, p):
        """Mirror a left-like Patch back to the right foot (x flipped, winding reversed)."""
        if self.side == "R":
            p.v = p.v * np.array([-1.0, 1.0, 1.0])
            G.flip_patch(p)
        return p


def hull2d(P):
    """Convex hull (k, 2), counter-clockwise, of the points P (n, 2) (Andrew's monotone chain)."""
    pts = sorted(set((round(float(x), 7), round(float(y), 7)) for x, y in np.asarray(P, float)))
    if len(pts) < 3:
        return np.array(pts, float)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, up = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + up[:-1], float)


def meridian(F, theta, z_p, m=1440, eps=(-70.0, 82.0), rmax=0.30, close=0.0):
    """The foot's surface in the vertical half plane through the ankle axis at azimuth theta (0 = toe, + = outer side),
    as an (n, 2) polyline (plan distance from the axis, z) ordered from the top (steepest ray) downwards, found with
    rays from the pivot (axis, z_p) at elevations eps (deg).

    `close` (radians): the radius of every ray is the farthest first hit over the planes within +-close of the azimuth
    (a morphological closing across narrow gaps: the gaps between modelled toes, the web) - the envelope covers the gaps
    instead of dipping into them."""
    d0 = np.array([F.c_a[0], F.c_a[1], z_p])
    up = np.array([0.0, 0.0, 1.0])
    offs = np.linspace(-close, close, 5) if close > 0 else [0.0]
    best = np.full(m, -np.inf)
    for dl in offs:
        d = np.array([np.sin(theta + dl), -np.cos(theta + dl), 0.0])
        r, got = F.skin.outline(d0, d, up, m, "near", rmin=0.003, rmax=rmax, return_hits=True, despike=0)
        if r is not None:
            best = np.maximum(best, np.where(got, r, -np.inf))
    ok = np.isfinite(best)
    if not ok.any():
        return None
    phi = np.arange(m) * (TAU / m)
    phi = np.where(phi > np.pi, phi - TAU, phi)
    sel = ok & (phi >= np.radians(eps[0])) & (phi <= np.radians(eps[1]))
    o = np.argsort(-phi[sel])
    ph, R = phi[sel][o], best[sel][o]
    return np.stack([R * np.cos(ph), z_p + R * np.sin(ph)], 1)


def _interp_periodic_abs(theta, knots_deg, vals):
    a = np.abs((np.asarray(theta, float) + np.pi) % TAU - np.pi)             # 0 = toe .. pi = heel
    return np.interp(np.degrees(a), knots_deg, vals)


def _throat_radius(theta, d0, d1, w0, w1, step=0.0005):
    """Plan distance at which the ray at azimuth theta leaves the throat opening, 0 for rays that miss it. The opening lies
    on the foot axis from forward distance d0 to d1, its half width tapering from w0 (at d0) to w1 and ending in a round
    cap (an ellipse with semi-axes w1 and 1.2 w1 closes it)."""
    s, c = np.sin(theta), np.cos(theta)
    if c <= 0.0:
        return 0.0
    r = np.arange(step, min(d1 / c, max(w0, w1) / max(abs(s), 1e-9)) + step, step)
    d = r * c
    a = 1.2 * w1
    dc = d1 - a
    t = np.clip((d - d0) / max(dc - d0, 1e-6), 0.0, 1.0)
    w = w0 + (w1 - w0) * (t * t * (3.0 - 2.0 * t))
    cap = np.sqrt(np.clip(1.0 - ((d - dc) / a) ** 2, 0.0, 1.0))
    w = np.where(d > dc, w1 * cap, w)
    inside = (d >= d0) & (np.abs(r * s) < w)
    return float(r[np.nonzero(inside)[0][-1]]) if inside.any() else 0.0


def upper_grid(F, c):
    """The leather upper as a grid of points ON the foot surface, (M, K + 1, 3): column j follows the meridian at azimuth
    theta_j (equal steps of the welt outline), row 0 is the welt (just above the underside of the foot), row K the top
    line: the collar round the ankle (height by azimuth) and the throat tongue on the instep. Returns the grid and the
    azimuths, the pivot height and the welt outline."""
    S = F.S
    M, K = int(c["columns"]), int(c["rows"])
    z_w = F.floor + c["welt_h"] * S
    seg = F.skin.slice(np.array([0.0, 0.0, z_w]), np.array([0.0, 0.0, 1.0]))
    out = hull2d(seg.reshape(-1, 3)[:, :2])                        # plan hull of the foot at the welt: smooth over the toe gaps
    pts = _resample_closed(out, M)
    d = pts - F.c_a
    theta = np.mod(np.arctan2(d[:, 0], -d[:, 1]), TAU)
    order = np.argsort(theta)
    theta, rho = theta[order], np.linalg.norm(d, axis=1)[order]
    z_p = F.floor + c["pivot"] * (float(F.ankle[2]) - F.floor)
    cap = c["collar"]
    knots = np.array(cap["az"], float)
    hs = np.array(cap["h"], float) * S
    d0 = c["throat"]["start"] * F.r_toe
    d1 = c["throat"]["end"] * F.r_toe
    w0, w1 = c["throat"]["w_start"] * S, c["throat"]["w_end"] * S
    mers, s_wel, s_top = [], np.zeros(M), np.zeros(M)
    for j, th in enumerate(theta):
        P2 = meridian(F, th, z_p, close=c["close"] * S / max(rho[j], 0.03))
        if P2 is None or len(P2) < 4:
            raise ValueError("shoe: the foot surface is not star-shaped round the ankle axis (no section found)")
        zs, rs = P2[:, 1], P2[:, 0]
        # the footprint: the widest sample low enough on the foot; the welt hangs straight down from it to the welt height
        # (the underside of a real foot is not flat: the arch, the rounded rim - the leather bridges it)
        low = zs <= F.floor + c["footprint_h"] * S
        if low.any():
            im = int(np.argmax(np.where(low, rs, -np.inf)))
            P2 = P2[:im + 1]
            if P2[-1, 1] > z_w + 1e-5:
                P2 = np.vstack([P2, [P2[-1, 0], z_w]])
            zs, rs = P2[:, 1], P2[:, 0]
        fw = len(zs) - 1.0
        fz = _first_cross(zs, float(F.ankle[2]) + _interp_periodic_abs(th, knots, hs), True)
        rk = _throat_radius(th, d0, d1, w0, w1)
        fr = _first_cross(rs, rk, False) if rk > 0 else 0.0
        ft = max(0.0 if fz is None else fz, 0.0 if fr is None else fr)
        ft = min(ft, fw - 0.5)
        s_all = _poly_len(P2)
        s_of = lambda f: float(np.interp(f, np.arange(len(P2)), s_all))
        mers.append((P2, s_all))
        s_wel[j], s_top[j] = s_of(fw), s_of(ft)
    # the top line: its distance from the welt along the meridian, smoothed round the shoe
    reach = s_wel - s_top
    for _ in range(int(c["top_smooth"])):
        reach = 0.25 * np.roll(reach, 1) + 0.5 * reach + 0.25 * np.roll(reach, -1)
    grid = np.zeros((M, K + 1, 3))
    for j, th in enumerate(theta):
        P2, s_all = mers[j]
        ss = np.linspace(s_wel[j], max(s_wel[j] - reach[j], s_all[0]), K + 1)
        r_, z_ = np.interp(ss, s_all, P2[:, 0]), np.interp(ss, s_all, P2[:, 1])
        grid[j, :, 0] = F.c_a[0] + r_ * np.sin(th)
        grid[j, :, 1] = F.c_a[1] - r_ * np.cos(th)
        grid[j, :, 2] = z_
    return grid, theta, z_p, out


def clear_quads(srf, P, dmin, iters=3):
    """Grid points P (J, K1, 3), closed in j: push vertices outwards until the centres and the edge midpoints of the quads are
    at least dmin from the surface too (a thin spike of the skin can pass between vertices). Returns the new grid."""
    J, K1 = P.shape[:2]
    idx = np.arange(J * K1).reshape(J, K1)
    nxt = np.roll(idx, -1, 0)
    quads = np.stack([idx[:, :-1], idx[:, 1:], nxt[:, 1:], nxt[:, :-1]], -1).reshape(-1, 4)
    owner = np.concatenate([quads] * 3)
    P = P.reshape(-1, 3).copy()
    for _ in range(iters):
        a, b, c_, d = (P[quads[:, i]] for i in range(4))
        samples = np.concatenate([0.25 * (a + b + c_ + d), 0.5 * (a + d), 0.5 * (b + c_)])
        dist, q, tri, _ = srf.closest(samples)
        bad = np.flatnonzero(dist < dmin - 1e-6)
        if not len(bad):
            break
        dirv = np.where((np.abs(dist[bad]) > 1e-4)[:, None], _u(samples[bad] - q[bad]) * np.sign(dist[bad])[:, None],
                        srf.nrm[tri[bad]])
        push = dirv * (dmin - dist[bad])[:, None]
        acc, cnt = np.zeros_like(P), np.zeros(len(P))
        for k in range(4):
            np.add.at(acc, owner[bad, k], push)
            np.add.at(cnt, owner[bad, k], 1.0)
        P += acc / np.maximum(cnt, 1.0)[:, None] * 1.25
    return P.reshape(J, K1, 3)


def build_upper(F, c):
    """Offset shell: returns dict(P (M, K+1, 3) final grid, raw, theta, z_p, dist (per vertex), n (normals), welt_out).

    The raw grid lies ON the foot (first hits of rays from the ankle axis). It is moved out by `dist` (ease + collar flare +
    toe room) along the normals of a heavily smoothed copy of itself - the skin's own normals are noise on a modelled foot
    with separate toes - and then relaxed: smoothing and projection out of the skin alternate, so the leather closes over
    the toes and the web instead of dipping between them, and stays `min_gap` away from every skin triangle."""
    S = F.S
    grid, theta, z_p, out = upper_grid(F, c)
    M, K1 = grid.shape[:2]
    raw = smooth_grid(grid, 2, 0.5)
    kf = np.linspace(0.0, 1.0, K1)
    flare = c["collar_flare"] * S * G.smoothstep((kf - 0.70) / 0.30)
    tipd = np.clip((raw[..., 1] - F.y_tip) / (c["toe_len"] * S), 0.0, 1.0)              # 0 at the toe tip .. 1 beyond toe_len
    toe = c["toe_room"] * S * (1.0 - G.smoothstep(tipd))
    dist = c["ease"] * S + flare[None, :] + toe
    inside = np.array([F.c_a[0], F.c_a[1], z_p])
    n = grid_normals(smooth_grid(grid, int(c["normal_smooth"]), 0.5), inside)
    P = raw + n * dist[..., None]
    dmin = c["min_gap"] * S
    P = relax_grid(F.srf, P, dmin, int(c["relax"]))
    P, _ = push_out(F.srf, P.reshape(-1, 3), dmin, iters=4)
    P = clear_quads(F.srf, P.reshape(grid.shape), dmin * 0.85, iters=6)
    for _ in range(3):                                               # quads (centres, edge middles) and vertices together
        P, _ = push_out(F.srf, P.reshape(-1, 3), dmin * 0.9, iters=3)
        P = clear_quads(F.srf, P.reshape(grid.shape), dmin * 0.85, iters=4)
    P, _ = push_out(F.srf, P.reshape(-1, 3), dmin * 0.9, iters=4)
    return {"P": P.reshape(grid.shape), "raw": raw, "theta": theta, "z_p": z_p, "dist": dist, "n": n, "welt_out": out}


def grid_patch(P, u, v, mat=0, tag="", closed_j=True):
    """Quad patch of a (J, K1, 3) grid: vertex (j, k) = j * K1 + k, faces (j,k) (j,k+1) (j+1,k+1) (j+1,k). `u` (J + closed,)
    and `v` (J, K1) are texture coordinates (the extra u closes the seam). Winding is NOT fixed: use orient_outward."""
    J, K1 = P.shape[:2]
    quads, uvs = [], []
    for j in range(J if closed_j else J - 1):
        j2 = (j + 1) % J
        for k in range(K1 - 1):
            quads.append((j * K1 + k, j * K1 + k + 1, j2 * K1 + k + 1, j2 * K1 + k))
            uvs += [(u[j], v[j, k]), (u[j], v[j, k + 1]), (u[j + 1], v[j2, k + 1]), (u[j + 1], v[j2, k])]
    return G.Patch(P.reshape(-1, 3).copy(), quads, np.array(uvs), None, mat, tag)


def loop_u(row, tile=0.10):
    """u (J + 1,) along a closed loop of points (J, 3) counted in whole tiles of `tile` metres (a seamless texture)."""
    seg = np.linalg.norm(np.roll(row, -1, 0) - row, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum / cum[-1] * max(1, int(round(cum[-1] / tile)))


def row_v(P, tile=0.10):
    """v (J, K1) = distance along each column of a (J, K1, 3) grid in tiles."""
    d = np.linalg.norm(np.diff(P, axis=1), axis=2)
    return np.concatenate([np.zeros((len(P), 1)), np.cumsum(d, axis=1)], axis=1) / tile


def planar_uv(p, tile=0.10, axes=(0, 1)):
    """Planar texture coordinates (metres / tile) for every face corner of a patch."""
    p.uv = np.array([(p.v[i][axes[0]] / tile, p.v[i][axes[1]] / tile) for f in p.f for i in f])
    return p


def smooth_rows(a, passes=1):
    """Circular 1-2-1 smoothing of the rows of a (J, d) array."""
    a = np.array(a, float)
    for _ in range(passes):
        a = 0.25 * np.roll(a, 1, 0) + 0.5 * a + 0.25 * np.roll(a, -1, 0)
    return a


def smooth_rows_open(a, passes=1):
    """1-2-1 smoothing of the rows of a (J, d) array, the end rows stay."""
    a = np.array(a, float)
    for _ in range(passes):
        b = a.copy()
        b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
        a = b
    return a


def unit_sign(a, ref):
    """Vectors `a` flipped where they point against `ref`."""
    return np.where((np.einsum("...k,...k->...", a, ref) < 0)[..., None], -a, a)


def floor_z(F, c):
    """z of the floor plane the sole stands on: `sole_z` if given, else the toe_end landmark (the sole of the foot) - but at
    least a sole's thickness (leather ease, ledge and rounded edge) below the foot's underside, so that a foot which
    already stands on z = 0 (the mannequin) gets a shoe that is merely lower than the floor, never a broken one."""
    S = F.S
    z = float(c["sole_z"]) if c.get("sole_z") is not None else float(F.toe_end[2])
    room = (c["ease"] + c["lip_drop"] + c["lip_t"] + c["edge_r"]) * S + 0.001
    return min(z, F.floor - room)


def bottom_z(F, c, xy):
    """Height of the sole's bottom at plan positions xy (..., 2): the floor plane, curling up in front of the ball (toe
    spring)."""
    S = F.S
    y0 = 0.5 * (float(F.toe[1]) + F.y_tip)
    t = np.clip((y0 - np.asarray(xy)[..., 1]) / max(y0 - F.y_tip, 1e-6), 0.0, 1.0)
    return floor_z(F, c) + c["spring"] * S * t * t


def build_sole(F, c, U):
    """Sole: a ledge out from the welt row, the side wall (an outsole band set out a little) down to the floor plane with a
    rounded bottom edge, and the flat bottom (rings and a centre fan). Returns dict(patches (material 1), min_z, size)."""
    S = F.S
    W = U["P"][:, 0, :]
    M = len(W)
    xy = W[:, :2]
    cen = xy.mean(0)
    tan = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    n2 = _u(np.stack([tan[:, 1], -tan[:, 0]], 1))
    if np.sum(np.einsum("ij,ij->i", n2, xy - cen)) < 0:
        n2 = -n2
    n2 = _u(smooth_rows(n2, 2))
    drop0, lip_t0, r_e0 = c["lip_drop"] * S, c["lip_t"] * S, c["edge_r"] * S
    band0, step = c["band"] * S, c["band_step"] * S
    zw = W[:, 2]
    # the ledge is wider at the toe and the heel (toe box and counter overhang the foot more than the sides)
    w_toe = 1.0 - G.smoothstep((xy[:, 1] - F.y_tip) / (0.050 * S))
    w_heel = G.smoothstep((xy[:, 1] - (F.y_heel - 0.040 * S)) / (0.040 * S))
    lip = (c["lip"] + c["lip_toe"] * w_toe + c["lip_heel"] * w_heel) * S

    def at(a):
        return xy + n2 * np.broadcast_to(a, (M,))[:, None]

    zb = bottom_z(F, c, at(lip + step))
    k = np.clip((zw - zb) / (drop0 + lip_t0 + band0 + r_e0 + 1e-3), 0.05, 1.0)  # squeeze the profile where the wall is low
    drop, lip_t, band, r_e = drop0 * k, lip_t0 * k, band0 * k, r_e0 * k
    z_band = zb + r_e + band
    rows = [(0.0, zw), (lip, zw - drop), (lip, zw - drop - lip_t), (lip, z_band), (lip + step, z_band), (lip + step, zb + r_e)]
    for ang in (35.0, 70.0, 90.0):
        a_ = np.radians(ang)
        a = lip + step - r_e + r_e * np.cos(a_)
        rows.append((a, bottom_z(F, c, at(a)) + r_e * (1.0 - np.sin(a_))))
    pts = np.stack([np.concatenate([at(a), np.broadcast_to(z, (M,))[:, None]], 1) for a, z in rows], 1)     # (M, R, 3)
    wall = grid_patch(pts, loop_u(W), row_v(pts), 1, "sole_wall")
    last = pts[:, -1, :]
    cxy = last[:, :2].mean(0)
    rings = [last]
    for s in (0.62, 0.28):
        r_xy = cxy + (last[:, :2] - cxy) * s
        rings.append(np.concatenate([r_xy, bottom_z(F, c, r_xy)[:, None]], 1))
    ring_pts = np.stack(rings, 1)
    bottom = planar_uv(grid_patch(ring_pts, loop_u(last), row_v(ring_pts), 1, "sole_bottom"))
    apex = np.array([cxy[0], cxy[1], float(bottom_z(F, c, cxy))])
    fan = planar_uv(G.fan(apex, ring_pts[:, -1, :], mat=1, tag="sole_bottom"))
    mid = np.array([cxy[0], cxy[1], 0.5 * float(zw.mean())])
    wall = G.orient_outward(wall, np.repeat(mid[None], len(wall.v), 0))
    bottom = G.orient_outward(bottom, np.repeat(mid[None], len(bottom.v), 0))
    fan = G.orient_outward(fan, mid)
    allv = np.concatenate([p.v for p in (wall, bottom, fan)])
    return {"patches": [wall, bottom, fan], "min_z": float(allv[:, 2].min()),
            "size": (float(allv[:, 0].max() - allv[:, 0].min()), float(allv[:, 1].max() - allv[:, 1].min())),
            "height": float(zw.mean() - allv[:, 2].min())}


def closed_sweep(path, n_out, up, rho, n=8, tag="", mat=0):
    """Tube along a closed 3-D loop (J, 3): a circle of radius rho in the plane of the outward normals `n_out` and `up`
    (J, 3). Texture: u round the tube, v along the loop in tiles of 10 cm. Outward faces."""
    ang = G.ring_theta(n)
    ring = path[:, None, :] + rho * (np.cos(ang)[None, :, None] * n_out[:, None, :] + np.sin(ang)[None, :, None] * up[:, None, :])
    ring = np.concatenate([ring, ring[:1]], 0)
    u = np.linspace(0.0, TAU * rho / 0.10, n + 1)
    p = G.loft(ring, True, u=u, v=loop_u(path), mat=mat, tag=tag)
    return G.orient_outward(p, np.repeat(np.concatenate([path, path[:1]], 0), n, axis=0))


def build_collar(F, c, U):
    """The rolled edge: a tube along the top line of the upper, centred on the edge (it covers the raw edge)."""
    st = int(c["collar_stride"])                                     # the roll needs fewer samples than the upper
    P, N = U["P"][::st], U["n"][::st]
    up = _u(P[:, -1, :] - P[:, -2, :])
    out = _u(N[:, -1, :] - up * np.einsum("ij,ij->i", N[:, -1, :], up)[:, None])
    p = closed_sweep(P[:, -1, :], out, up, c["collar_r"] * F.S, int(c["collar_sides"]), tag="collar")
    p.v, _ = push_out(F.srf, p.v, c["min_gap"] * F.S)
    return p


def build_strap(F, c, U):
    """The instep strap: a flat rounded bar across the foot at `pos` of the way from the ankle axis to the toe, over the
    throat, its ends resting on the quarters. Returns dict(patches, path, normals, across, tangent, phi, half_w, half_t)."""
    S = F.S
    sc = c["strap"]
    y_s = float(F.c_a[1] - sc["pos"] * F.r_toe)
    x_c = float(F.c_a[0])
    z_lo = F.floor + 0.012 * S
    r_up = F.skin.outline(np.array([x_c, y_s, z_lo]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 720, "near",
                          rmin=0.004, rmax=0.20, despike=0)
    piv = np.array([x_c, y_s, F.floor + 0.5 * (z_lo + r_up[0] - F.floor)])
    n_p, m = int(sc["points"]), 720
    phi = np.linspace(np.radians(sc["arc"][0]), np.radians(sc["arc"][1]), n_p)
    r = F.skin.outline(piv, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), m, "near", rmin=0.004, rmax=0.20, despike=0)
    R = np.interp(phi, np.arange(m + 1) * (TAU / m), np.append(r, r[0]))
    base = piv + np.stack([R * np.cos(phi), np.zeros(n_p), R * np.sin(phi)], 1)
    ht, hw = 0.5 * sc["thick"] * S, 0.5 * sc["width"] * S
    ctr, nrm, _ = surface_offset(F.srf, base, c["ease"] * S + ht)
    ctr, _ = push_out(F.srf, ctr, c["ease"] * S * 0.9 + ht, iters=2)
    ctr = smooth_rows_open(ctr, 2)
    T = _u(np.gradient(ctr, axis=0))
    nrm = _u(nrm - T * np.einsum("ij,ij->i", nrm, T)[:, None])
    a = unit_sign(_u(np.cross(nrm, T)), np.array([0.0, 1.0, 0.0]))                  # across the strap, towards the heel
    ang = G.ring_theta(int(sc["profile"]))
    pa = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** 0.5
    ph = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** 0.5
    rings = ctr[:, None, :] + hw * pa[None, :, None] * a[:, None, :] + ht * ph[None, :, None] * nrm[:, None, :]
    rings, _ = push_out(F.srf, rings.reshape(-1, 3), c["min_gap"] * S * 0.9, iters=4)       # the rounded edges too
    rings = rings.reshape(len(ctr), -1, 3)
    u = np.linspace(0.0, 2.0 * (hw + ht) / 0.10 * 2.0, len(ang) + 1)
    body = G.loft(rings, True, u=u, v=1.0 - _poly_len(ctr) / 0.10, mat=0, tag="strap")
    body = G.orient_outward(body, np.repeat(ctr, len(ang), axis=0))
    parts = [body]
    for e, sgn in ((0, -1.0), (n_p - 1, 1.0)):
        cap = G.fan(ctr[e] + T[e] * (sgn * ht * 0.8), rings[e], mat=0, tag="strap")
        parts.append(G.orient_outward(cap, ctr[e] - T[e] * sgn))
    return {"patches": parts, "path": ctr, "normals": nrm, "across": a, "tangent": T, "phi": phi, "half_w": hw, "half_t": ht}


def build_buckle(c, strap, S):
    """A small rounded-rectangle frame with a centre bar lying on the strap near its outer end."""
    bk = c["buckle"]
    i = int(np.argmin(np.abs(strap["phi"] - np.radians(bk["at"]))))
    p0, T, n, a = strap["path"][i], strap["tangent"][i], strap["normals"][i], strap["across"][i]
    w, h, r = 0.5 * bk["width"] * S, 0.5 * bk["height"] * S, bk["wire"] * S
    cx = p0 + n * (strap["half_t"] + 0.8 * r)
    t = G.ring_theta(24)
    ex = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.3
    ey = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.3
    loop = cx + w * ex[:, None] * T + h * ey[:, None] * a
    radial = _u(loop - cx - np.einsum("ij,j->i", loop - cx, n)[:, None] * n)
    frame = closed_sweep(loop, np.broadcast_to(n, loop.shape), radial, r, 6, tag="buckle")
    bar = G.tube(np.linspace(cx - T * w, cx + T * w, 5), 0.8 * r, n=6, mat=0, tag="buckle")
    return [frame] + bar


def metric_bow_uv(parts, tile=0.04):
    """`outfit_geo.bow` strips run u from 0 to 1 along their ribbon: make u count tiles of `tile` metres (the knot, an
    ellipsoid, keeps its own)."""
    for p in parts:
        if len(p.v) % 3 == 0 and len(p.v) // 3 in (18, 8):                          # loop (closed, 18 rows) / tail (8 rows)
            mid = p.v[1::3]
            seg = np.linalg.norm(np.diff(mid, axis=0), axis=1).sum()
            if len(mid) == 18:
                seg += np.linalg.norm(mid[0] - mid[-1])
            p.uv = p.uv * np.array([seg / tile, 1.0])
    return parts


def build_shoe_bow(c, strap, S):
    """The small satin bow on the middle of the strap; its tails run down the instep towards the toe."""
    bw = c["bow"]
    i = len(strap["path"]) // 2
    p0, T, n, a = strap["path"][i], strap["tangent"][i], strap["normals"][i], strap["across"][i]
    O = p0 + n * (strap["half_t"] + bw["lift"] * S)
    parts = G.bow(O, T, a, n, size=bw["size"] * S, loop_len=bw["loop"], loop_w=bw["loop_w"], ribbon_w=bw["ribbon_w"],
                  tail_len=bw["tail_len"], droop=bw["droop"], mat=0, tag="shoe_bow", seed=5)
    return metric_bow_uv(parts)


def foot_donors(F):
    """Bones whose skin weights the shoe takes: the shin, the ankle and the toe bone of its side (the forefoot skin is
    weighted on the toe bone: a toe box that is rigid on the ankle would let bending toes poke through)."""
    return (f"{F.name}ひざ", f"{F.name}足首", f"{F.name}足つま先")


def rigid_weights(fit, F, patches):
    """Weights shared by all vertices of a rigid decoration (buckle, bow): the skin weights under the middle of the piece."""
    bone = f"{F.name}足首"
    pts = np.concatenate([p.v for p in patches])
    w = fit.weights_for(pts.mean(0)[None], foot_donors(F), default=bone)
    for p in patches:
        p.w = {b: np.full(len(p.v), float(a[0])) for b, a in w.items()}


def shoes(fit, cfg, soup_shoes, soup_satin, rig=None):
    """Black Mary-Jane shoes on both feet: leather upper with a rolled collar and a throat opening over the instep, strap
    with a buckle and a small bow, sole with a flat bottom on the floor plane.

    Adds to `soup_shoes` (material 0 = 靴: upper, collar, strap, buckle; 1 = 靴底: sole) and `soup_satin` (0 = 黒リボン: the
    bows). Upper, collar, sole and strap take the weights of the nearest skin point (donors: 左/右 ひざ, 足首, 足つま先: the
    toe box follows the toes and the sole flexes at the ball); buckle and bows are rigid with the weights under their middle.
    Returns {"L": info, "R": info, "min_dist", "sole_min_z"}; info has the Patches (model space), the grid of the upper, the
    measured distances of the leather to the skin, the sole size and height, the Foot (landmarks, floor, length, width)."""
    S = fit.S
    c = merge(DEFAULTS["shoe"], (cfg or {}).get("shoe"))
    res = {}
    for side in ("L", "R"):
        F = Foot(fit, side)
        U = build_upper(F, c)
        W = U["P"]
        upper = grid_patch(W, loop_u(W[:, 0, :]), row_v(W), 0, "upper")
        upper = G.orient_outward(upper, np.repeat(np.array([F.c_a[0], F.c_a[1], U["z_p"]])[None], len(upper.v), 0))
        collar = build_collar(F, c, U)
        sole = build_sole(F, c, U)
        strap = build_strap(F, c, U)
        buckle = build_buckle(c, strap, S)
        bows = build_shoe_bow(c, strap, S)
        leather_left = [upper, collar] + strap["patches"] + buckle
        d, _, _, _ = F.srf.closest(np.concatenate([p.v for p in leather_left]))
        bone = f"{F.name}足首"
        donors = foot_donors(F)
        world = {"upper": F.patch_world(upper), "collar": F.patch_world(collar)}
        for p in (upper, collar):
            p.w = fit.weights_for(p.v, donors, default=bone)
            soup_shoes.add(p, 0)
        for p in sole["patches"]:                                    # the sole flexes with the foot at the ball
            F.patch_world(p)
            p.w = fit.weights_for(p.v, donors, default=bone)
            soup_shoes.add(p, 1)
        rigid = []
        for p in strap["patches"]:
            F.patch_world(p)
            p.w = fit.weights_for(p.v, donors, default=bone)
            soup_shoes.add(p, 0)
            rigid.append(p)
        for p in buckle:
            F.patch_world(p)
        rigid_weights(fit, F, buckle)                                # (before `add`: the soup keeps its own Patch)
        for p in buckle:
            soup_shoes.add(p, 0)
            rigid.append(p)
        for p in bows:
            F.patch_world(p)
        rigid_weights(fit, F, bows)
        for p in bows:
            soup_satin.add(p, 0)
        pts = np.concatenate([p.v for p in world.values()] + [p.v for p in rigid])
        res[side] = {"upper": world["upper"], "collar": world["collar"], "sole": sole, "strap": strap, "foot": F, "grid": W,
                     "leather_points": pts, "min_dist": float(d.min()), "max_dist": float(d.max()),
                     "sole_min_z": sole["min_z"], "sole_size": sole["size"], "sole_height": sole["height"],
                     "floor_z": floor_z(F, c),
                     "collar_z": float(W[:, -1, 2].max()), "welt_z": (float(W[:, 0, 2].min()), float(W[:, 0, 2].max()))}
    res["min_dist"] = min(res[s]["min_dist"] for s in ("L", "R"))
    res["sole_min_z"] = min(res[s]["sole_min_z"] for s in ("L", "R"))
    return res
