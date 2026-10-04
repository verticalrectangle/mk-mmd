"""Fitting the outfit to a body surface (numpy only; Blender-safe: no scipy).

`Skin` is a triangulated body surface with optional per-vertex weights. It answers the questions a tailor asks:
 - `outline`: the radial profile of a cross-section (a plane through a centre, rays from the centre, nearest or
   farthest crossing), the base of every offset shell (bodice rings, sleeves, ribbon turns, shoe stations);
 - `closest`: signed distance to the surface (negative inside), closest point and barycentrics, in chunks;
 - `transfer`: skin weights of the nearest body surface points for garment vertices, optionally restricted to bones.
`Land` wraps the landmark dict (semantic names such as `arm.L`, `knee.R`) with fallbacks; `provisional_skin` builds a
simple mannequin from landmarks so the outfit can be developed and tested before (or without) the real body."""
import warnings

import numpy as np

from .. import skin as SK
from . import outfit_geo as G

# PMX bone-name prefixes of the body regions (side-independent stems; the side prefix 左/右 is added where needed)
ARM_STEMS = ("肩", "腕", "腕捩", "ひじ", "手捩", "手首", "親指", "人指", "中指", "薬指", "小指", "手")
LEG_STEMS = ("足", "ひざ", "足首", "つま先", "足ＩＫ", "足先EX")


def side_bones(weights, side):
    """Names in `weights` that belong to one side's arm ('左'|'右' prefix, or .L/.R suffix) and an arm stem."""
    out = []
    for n in weights:
        if n.startswith(side) and any(n[1:].startswith(s) for s in ARM_STEMS):
            out.append(n)
    return out


def _stem(n):
    """The bone name without its 左/右 side prefix ('' when it has none)."""
    return n[1:] if n[:1] in ("左", "右") else ""


def is_arm(n):
    """An arm or hand bone (腕, 腕捩, ひじ, 手捩, 手首, the fingers, split twist bones); the shoulder (肩) is not one."""
    s = _stem(n)
    return bool(s) and any(s.startswith(a) for a in ARM_STEMS if a != "肩")


def is_leg(n):
    """A leg or foot bone (足, ひざ, 足首, つま先, 足つま先, the IK bones)."""
    s = _stem(n)
    return bool(s) and any(s.startswith(a) for a in LEG_STEMS)


def is_head(n):
    """The head and the eyes."""
    return n in ("頭", "両目", "目") or (n[1:] == "目" and n[:1] in ("左", "右"))


class Skin:
    """Triangulated surface + optional weights {bone: (n,)}."""

    def __init__(self, verts, faces, weights=None, name="skin"):
        self.verts = np.asarray(verts, float)
        self.tris, self.src = G.tri_arrays(list(faces)) if not isinstance(faces, np.ndarray) else (faces, np.arange(len(faces)))
        self.weights = {k: np.asarray(v, float) for k, v in (weights or {}).items()}
        self.name = name
        self._flip = False
        vol = self.signed_volume()
        if vol < 0:                                  # inward winding: keep the geometry, fix the sign convention
            self._flip = True

    # ---- construction
    @classmethod
    def from_meshes(cls, meshes, name="skin"):
        """Merge Part meshes (verts, faces, weights) into one Skin."""
        V, T, W, off = [], [], [], 0
        keys = sorted({k for m in meshes for k in m.weights})
        for m in meshes:
            v = np.asarray(m.verts, float)
            t, _ = G.tri_arrays(list(m.faces))
            V.append(v)
            T.append(t + off)
            W.append({k: np.asarray(m.weights.get(k, np.zeros(len(v))), float) for k in keys})
            off += len(v)
        verts = np.vstack(V)
        tris = np.vstack(T)
        weights = {k: np.concatenate([w[k] for w in W]) for k in keys}
        s = cls.__new__(cls)
        s.verts, s.tris, s.src, s.weights, s.name, s._flip = verts, tris, np.arange(len(tris)), weights, name, False
        if s.signed_volume() < 0:
            s._flip = True
        return s

    def signed_volume(self):
        a, b, c = (self.verts[self.tris[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def where(self, bones=None, share=0.5, tri_mask=None, name=None):
        """The sub-surface of triangles whose three vertices carry at least `share` of their weight on `bones`."""
        if bones is not None:
            tot = np.zeros(len(self.verts))
            for b in bones:
                if b in self.weights:
                    tot += self.weights[b]
            ok = (tot[self.tris] >= share).all(1)
        else:
            ok = np.ones(len(self.tris), bool)
        if tri_mask is not None:
            ok &= tri_mask
        s = Skin.__new__(Skin)
        s.verts, s.tris, s.src, s.weights, s.name, s._flip = self.verts, self.tris[ok], self.src[ok], self.weights, name or self.name, self._flip
        return s

    def bounds(self):
        return self.verts.min(0), self.verts.max(0)

    def to_mesh(self, name=None, mat="skin"):
        """The triangles as a Part `Mesh` (preview and debugging)."""
        from ..part import Mesh
        return Mesh(name or self.name, self.verts, [tuple(int(i) for i in t) for t in self.tris],
                    None, None, [mat], {k: v for k, v in self.weights.items()})

    # ---- slicing
    def slice(self, origin, normal):
        """Segments (S, 2, 3) where the plane through `origin` with `normal` cuts the triangles."""
        n = G.unit(normal)
        d = (self.verts - np.asarray(origin, float)) @ n
        dt = d[self.tris]
        pos = dt > 0
        mixed = pos.any(1) & (~pos).any(1)
        if not mixed.any():
            return np.zeros((0, 2, 3))
        T = self.tris[mixed]
        pt = pos[mixed]
        dd = dt[mixed]
        # lone vertex = the one whose sign differs from the other two
        s = pt.sum(1)
        lone = np.where(s == 1, np.argmax(pt, 1), np.argmax(~pt, 1))
        idx = np.arange(len(T))
        i0 = T[idx, lone]
        i1 = T[idx, (lone + 1) % 3]
        i2 = T[idx, (lone + 2) % 3]
        d0, d1, d2 = dd[idx, lone], dd[idx, (lone + 1) % 3], dd[idx, (lone + 2) % 3]
        p0, p1, p2 = self.verts[i0], self.verts[i1], self.verts[i2]
        t1 = (d0 / (d0 - d1))[:, None]
        t2 = (d0 / (d0 - d2))[:, None]
        return np.stack([p0 + (p1 - p0) * t1, p0 + (p2 - p0) * t2], 1)

    def outline(self, center, u, v, m, mode="near", rmin=0.0, rmax=np.inf, return_hits=False, despike=0.6):
        """Radii (m,) of the section in the plane through `center` spanned by unit u, v: for the ray at angle
        phi_j = j * 2pi / m (from u towards v) the nearest (or farthest) crossing with distance in [rmin, rmax].
        Rays without a crossing are filled by circular interpolation (all missing: None)."""
        c = np.asarray(center, float)
        n = np.cross(u, v)
        segs = self.slice(c, n)
        r = np.full(m, np.nan)
        if len(segs):
            A = np.stack([(segs[:, 0] - c) @ u, (segs[:, 0] - c) @ v], 1)
            E = np.stack([(segs[:, 1] - c) @ u, (segs[:, 1] - c) @ v], 1) - A
            phi = G.ring_theta(m)
            D = np.stack([np.cos(phi), np.sin(phi)], 1)
            den = D[:, 0:1] * E[None, :, 1] - D[:, 1:2] * E[None, :, 0]                  # d x e   (m, S)
            ok = np.abs(den) > 1e-14
            den = np.where(ok, den, 1.0)
            t = (A[None, :, 0] * E[None, :, 1] - A[None, :, 1] * E[None, :, 0]) / den
            s = (A[None, :, 0] * D[:, 1:2] - A[None, :, 1] * D[:, 0:1]) / den
            hit = ok & (s >= -1e-9) & (s <= 1 + 1e-9) & (t >= rmin) & (t <= rmax)
            tt = np.where(hit, t, np.inf if mode == "near" else -np.inf)
            r = tt.min(1) if mode == "near" else tt.max(1)
            r = np.where(np.isfinite(r), r, np.nan)
            if despike and np.isfinite(r).sum() > 6:                  # isolated runs on far surfaces are not the outline
                win = np.stack([np.roll(r, k) for k in range(-3, 4)])
                with warnings.catch_warnings():                       # windows without any crossing are all-NaN: fine
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(win, axis=0)
                bad = np.abs(r - med) > despike * med
                r = np.where(bad, np.nan, r)
        got = np.isfinite(r)
        if not got.any():
            return (None, got) if return_hits else None
        if not got.all():
            x = np.arange(m)
            r = np.interp(x, np.concatenate([x[got] - m, x[got], x[got] + m]), np.tile(r[got], 3))
        return (r, got) if return_hits else r

    def section(self, center, u, v, m=64, mode="near", rmin=0.0, rmax=np.inf, iters=2, despike=0.6):
        """Section centre and radii. The centre is moved to the outline's centroid (`iters` times) so that tilted
        or off-centre cross-sections stay star-shaped. Returns (centre, radii) or (None, None)."""
        c = np.asarray(center, float)
        r = self.outline(c, u, v, m, mode, rmin, rmax, despike=despike)
        if r is None:
            return None, None
        for _ in range(iters):
            phi = G.ring_theta(m)
            P = np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
            Q = np.roll(P, -1, 0)
            cr = P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]
            area = cr.sum() * 0.5
            if abs(area) < 1e-12:
                break
            cx = ((P[:, 0] + Q[:, 0]) * cr).sum() / (6 * area)
            cy = ((P[:, 1] + Q[:, 1]) * cr).sum() / (6 * area)
            c = c + cx * u + cy * v
            r2 = self.outline(c, u, v, m, mode, rmin, rmax, despike=despike)
            if r2 is None:
                break
            r = r2
        return c, r

    # ---- distances and weights (ModelKit's skin.py does the nearest-surface search and the weight transfer)
    def closest(self, pts):
        """(signed distance (q,), closest point (q, 3), tri index (q,), barycentric (q, 3)) of points to the surface;
        negative = behind the surface (the winding decides)."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        q, tri, bary, _ = SK.closest_on_mesh(pts, self.verts, self.tris)
        A, B, C = (self.verts[self.tris[tri, i]] for i in range(3))
        nrm = G.unit(np.cross(B - A, C - A)) * (-1.0 if self._flip else 1.0)
        dist = np.linalg.norm(pts - q, axis=1)
        sgn = np.where(np.einsum("ij,ij->i", pts - q, nrm) < 0, -1.0, 1.0)
        return dist * sgn, q, tri, bary

    def transfer(self, pts, bones=None, fallback=None, faces=None, smooth=0):
        """Skin weights {bone: (q,)} of the nearest surface points for `pts`; `bones` restricts the donor bones (the rest
        are dropped before interpolating); `faces` + `smooth` Laplacian-smooth the result over the garment mesh."""
        w = self.weights if bones is None else {b: a for b, a in self.weights.items() if b in bones}
        if not w:
            return {fallback: np.ones(len(pts))} if fallback else {}
        return SK.transfer(self.verts, self.tris, w, np.asarray(pts, float).reshape(-1, 3), fallback=fallback,
                           smooth_faces=faces, smooth_iters=smooth if faces is not None else 0)


# ---------------------------------------------------------------- landmarks
class Land:
    """Landmark lookup: semantic names -> (3,) arrays (`arm.L`, `knee.R`, `neck` ...), with fallbacks and mirroring."""

    def __init__(self, d):
        self.d = {k: np.asarray(v, float) for k, v in (d or {}).items()}

    def has(self, k):
        return k in self.d

    def __getitem__(self, k):
        if k in self.d:
            return self.d[k]
        if k.endswith(".L") or k.endswith(".R"):
            o = k[:-1] + ("R" if k.endswith("L") else "L")
            if o in self.d:
                p = self.d[o].copy()
                p[0] = -p[0]
                return p
        raise KeyError(f"landmark {k!r} missing")

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return None if default is None else np.asarray(default, float)

    def side(self, name, s):
        return self[f"{name}.{s}"]


# Reisen-class standing A-pose landmarks in metres (rounded high-level measurements; Rin is scaled from these when the
# body part does not publish its own)
BASE_LANDMARKS = {
    "root": (0.0, 0.0, 0.0), "center": (0.0, -0.018, 0.886), "groove": (0.0, -0.018, 0.901),
    "lower_body": (0.0, -0.053, 0.923), "upper_body": (0.0, -0.053, 0.935), "upper_body2": (0.0, -0.050, 1.019),
    "neck": (0.0, 0.002, 1.217), "head": (0.0, -0.005, 1.267), "head_tip": (0.0, -0.014, 1.414),
    "shoulder.L": (0.046, 0.009, 1.186), "arm.L": (0.091, 0.006, 1.180), "elbow.L": (0.267, 0.014, 1.067),
    "wrist.L": (0.414, 0.007, 0.991),
    "leg.L": (0.080, -0.038, 0.781), "knee.L": (0.055, -0.032, 0.441), "ankle.L": (0.063, -0.001, 0.098),
    "toe.L": (0.060, -0.084, 0.041), "toe_end.L": (0.057, -0.165, 0.0),
}


def default_landmarks(scale=1.0):
    d = {k: np.array(v) * scale for k, v in BASE_LANDMARKS.items()}
    for k in list(d):
        if k.endswith(".L"):
            p = d[k].copy()
            p[0] = -p[0]
            d[k[:-1] + "R"] = p
    return d


# ---------------------------------------------------------------- provisional mannequin
def _tube_skin(a, b, ra, rb, n=14, segs=6, ratio=1.0, ref=(0, -1, 0)):
    """Elliptical tapered tube a -> b (radii ra -> rb across the ref direction; `ratio` squashes the other axis)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = G.unit(b - a)
    u, v = G.frame_from_axis(ax, ref)
    t = np.linspace(0, 1, segs + 1)
    th = G.ring_theta(n)
    rr = (ra + (rb - ra) * t)[:, None]
    rings = (a + (b - a) * t[:, None])[:, None, :] + rr[:, :, None] * (np.cos(th)[None, :, None] * u + ratio * np.sin(th)[None, :, None] * v)
    return rings, t


def _poly_limb(points, radii, joint_bones, tail_bone=None, n=14, dense=48, caps=(False, True)):
    """A tube through `points` (Catmull path) whose radius is interpolated between the joints, with weights: joint_bones[i]
    owns the stretch from point i to i+1, blended over +-12 % of the path around each interior joint. Returns patches."""
    pts = np.asarray(points, float)
    P = G.catmull(pts, dense)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    sarc = np.concatenate([[0.0], np.cumsum(seg)])
    # arc parameter of each control point (nearest dense sample)
    js = np.array([sarc[int(np.argmin(np.linalg.norm(P - q, axis=1)))] for q in pts])
    rad = np.interp(sarc, js, radii)
    parts = G.tube(P, rad, n=n, caps=caps)
    tt = sarc
    names = list(joint_bones)
    out = []
    for q in parts:
        nv = len(q.v)
        if nv == dense * n:
            t = np.repeat(tt, n)
        else:                                                   # a cap: the weights of its end ring
            t = np.full(nv, tt[-1] if q.v[0] @ (P[-1] - P[0]) > P[0] @ (P[-1] - P[0]) else tt[0])
        w = {b: np.zeros(nv) for b in names}
        for i, b in enumerate(names):
            lo = js[i] + (0.12 * (js[i] - js[i - 1]) if i else -1e9) * (1 if i else 0)
            hi = js[i + 1] - 0.12 * (js[i + 1] - js[i]) if i + 1 < len(js) - 1 else 1e9
            w[b] = np.where((t >= lo) & (t <= hi), 1.0, 0.0)
        # smooth blends: recompute with smoothstep around interior joints
        w = {b: np.zeros(nv) for b in names}
        for i, b in enumerate(names):
            left = 1.0 if i == 0 else G.smoothstep((t - (js[i] - 0.12 * (js[i] - js[i - 1]))) / (0.24 * (js[i] - js[i - 1]) + 1e-9))
            right = 1.0 if i + 1 >= len(names) else 1.0 - G.smoothstep((t - (js[i + 1] - 0.12 * (js[i + 1] - js[i]))) / (0.24 * (js[i + 1] - js[i]) + 1e-9))
            w[b] = left * right
        q.w = w
        out.append(q)
    return out


def provisional_skin(land, scale=1.0, prop=None):
    """A simple mannequin Skin (torso, neck, arms, legs, hands, feet) with weights, built from landmarks (and, when
    `prop` is the spec's [proportions] table, from its torso cuts and limb diameters). Used when the spec has no body
    part, and by the tests. Standing A-pose."""
    L = land if isinstance(land, Land) else Land(land)
    s = scale
    prop = prop or {}
    sec = (prop.get("sections") or {}).get("torso")
    limb = (prop.get("sections") or {}).get("limb") or {}
    parts = []

    def add(patch, weights):
        patch.w = weights
        parts.append(patch)

    z_leg = (L["leg.L"][2] + L["leg.R"][2]) / 2
    z_ub, z_ub2, z_nk = L["upper_body"][2], L["upper_body2"][2], L["neck"][2]
    zsh = L["arm.L"][2]
    if sec:
        z = np.array(sec["z"], float)
        w = np.array(sec["width"], float) / 2
        yf, yb = np.array(sec["y_front"], float), np.array(sec["y_back"], float)
        prof = np.stack([z, w, (yb - yf) / 2, (yb + yf) / 2], 1)
        top = z_nk + 0.045 * s
        prof = np.vstack([[z[0] - 0.03 * s, w[0] * 0.8, (yb[0] - yf[0]) / 2 * 0.8, (yb[0] + yf[0]) / 2], prof,
                          [[z_nk + 0.01 * s, 0.040 * s, 0.040 * s, 0.003], [top, 0.038 * s, 0.038 * s, 0.003]]])
        prof = prof[np.argsort(prof[:, 0])]
        # the table ends at the neck cut: keep a neck above it
        prof = prof[np.unique(prof[:, 0], return_index=True)[1]]
    else:
        prof = np.array([  # z, rx, ry, cy
            [z_leg - 0.045 * s, 0.135 * s, 0.092 * s, -0.045 * s],
            [z_leg + 0.02 * s, 0.150 * s, 0.098 * s, -0.045 * s],
            [z_leg + 0.095 * s, 0.138 * s, 0.092 * s, -0.050 * s],
            [z_ub + 0.0 * s, 0.108 * s, 0.078 * s, -0.052 * s],
            [z_ub2 - 0.03 * s, 0.118 * s, 0.086 * s, -0.050 * s],
            [z_ub2 + 0.035 * s, 0.128 * s, 0.092 * s, -0.045 * s],
            [zsh - 0.05 * s, 0.134 * s, 0.084 * s, -0.030 * s],
            [zsh + 0.015 * s, 0.128 * s, 0.075 * s, -0.020 * s],
            [z_nk - 0.012 * s, 0.075 * s, 0.062 * s, -0.005 * s],
            [z_nk + 0.01 * s, 0.050 * s, 0.050 * s, 0.002 * s],
            [z_nk + 0.05 * s, 0.047 * s, 0.047 * s, 0.002 * s],
        ])
    zs = np.linspace(prof[0, 0], prof[-1, 0], 40)
    th = G.ring_theta(40)
    rx = np.interp(zs, prof[:, 0], prof[:, 1])
    ry = np.interp(zs, prof[:, 0], prof[:, 2])
    cy = np.interp(zs, prof[:, 0], prof[:, 3])
    rings = np.stack([np.stack([rx[k] * np.sin(th), cy[k] - ry[k] * np.cos(th), np.full(40, zs[k])], -1) for k in range(len(zs))])
    rings = rings[::-1]                                        # top to bottom
    p = G.loft(rings)
    p = G.orient_outward(p, np.stack([np.zeros(len(rings) * 40) + 0.0, np.repeat(cy[::-1], 40), np.repeat(zs[::-1], 40)], 1))
    zz = np.repeat(zs[::-1], 40)

    def smooth_band(z, z0, z1):
        return G.smoothstep((z - z0) / max(z1 - z0, 1e-9))

    lw = np.clip(smooth_band(zz, z_ub - 0.05 * s, z_ub + 0.05 * s), 0, 1)
    u2 = smooth_band(zz, z_ub2 - 0.04 * s, z_ub2 + 0.04 * s)
    nk = smooth_band(zz, z_nk - 0.02 * s, z_nk + 0.02 * s)
    w = {"下半身": 1 - lw, "上半身": lw * (1 - u2), "上半身2": u2 * (1 - nk), "首": nk}
    add(p, w)

    def dia(name, default):
        v = limb.get(name)
        return (0.25 * (v[0] + v[1]) if v else default * s)          # mean radius from [major, minor] diameters

    arm_r = [dia("upper_arm_top", 0.050), dia("elbow", 0.040), dia("wrist", 0.029)]
    leg_r = [dia("thigh_top", 0.082), dia("knee", 0.056), dia("ankle", 0.038)]
    for sd, pre, sg in (("L", "左", 1.0), ("R", "右", -1.0)):
        sh, el, wr = L[f"arm.{sd}"], L[f"elbow.{sd}"], L[f"wrist.{sd}"]
        hand_tip = wr + G.unit(wr - el) * 0.16 * s
        for q in _poly_limb([sh, el, wr, hand_tip], [arm_r[0], arm_r[1], arm_r[2], 0.020 * s],
                            [f"{pre}腕", f"{pre}ひじ", f"{pre}手首"], caps=(False, True)):
            if f"{pre}腕" in q.w:
                top = G.smoothstep((0.25 - np.clip((q.v - sh) @ G.unit(el - sh) / np.linalg.norm(el - sh), 0, 1)) / 0.25)
                q.w[f"{pre}腕"] = q.w[f"{pre}腕"] * (1 - top * 0.5)
                q.w["上半身2"] = top * 0.5
            add(q, q.w)
        sph = G.ellipsoid(sh, (arm_r[0] * 1.1,) * 3, None, 8, 14)
        add(sph, {f"{pre}腕": np.full(len(sph.v), 0.6), "上半身2": np.full(len(sph.v), 0.4)})
        hip, kn, an = L[f"leg.{sd}"], L[f"knee.{sd}"], L[f"ankle.{sd}"]
        for q in _poly_limb([hip, kn, an], [leg_r[0], leg_r[1], leg_r[2]], [f"{pre}足", f"{pre}ひざ"], caps=(False, False)):
            if f"{pre}足" in q.w:
                top = G.smoothstep((0.15 - np.clip((q.v - hip) @ G.unit(kn - hip) / np.linalg.norm(kn - hip), 0, 1)) / 0.15)
                q.w[f"{pre}足"] = q.w[f"{pre}足"] * (1 - top)
                q.w["下半身"] = top
            add(q, q.w)
        toe = L[f"toe.{sd}"]
        foot_rings = []
        ys = np.linspace(an[1] + 0.05 * s, toe[1] - 0.075 * s, 8)
        for k, y in enumerate(ys):
            tt = k / (len(ys) - 1)
            h = (0.075 * (1 - tt) ** 0.6 + 0.022 * tt) * s
            wdt = (0.036 + 0.014 * tt - 0.012 * tt * tt) * s
            ang = G.ring_theta(12)
            foot_rings.append(np.stack([an[0] + wdt * np.sin(ang), np.full(12, y), h / 2 + (h / 2) * np.cos(ang)], -1))
        pf = G.loft(np.array(foot_rings))
        pf = G.orient_outward(pf, np.array([[an[0], y, 0.03 * s] for y in ys for _ in range(12)]))
        add(pf, {f"{pre}足首": np.ones(len(pf.v))})
    out = G.join(parts)
    w = G.cap_weights(out.w, len(out.v), 4)
    return Skin(out.v, out.f, w, name="provisional")


# ---------------------------------------------------------------- the tailor's view of a body
HAND_STEMS = ("親指", "人指", "中指", "薬指", "小指")


class Fit:
    """Landmarks, body regions and cross-sections for the garment builders.

    `skin` is the whole body surface (with weights when the body has them). `torso` is the torso + shoulders, `core`
    torso + legs (no arms: the skirt must clear thighs and hips, never the hands)."""

    def __init__(self, land, skin, scale=None, log=None, bone_names=()):
        self.bone_names = set(bone_names)
        self.L = land if isinstance(land, Land) else Land(land)
        self.skin = skin
        self.log = log or (lambda *a: None)
        tip = self.L.get("head_tip")
        self.S = float(scale) if scale else (float(tip[2]) / 1.7 if tip is not None else float(self.L["neck"][2]) / 1.217)
        names = list(skin.weights)
        # regions by exclusion: whatever bone a body adds (胸, 腰, helper bones ...) belongs to the torso
        self.torso = self._region([n for n in names if not (is_arm(n) or is_leg(n) or is_head(n))], "torso")
        self.core = self._region([n for n in names if not (is_arm(n) or is_head(n))], "core")

    def _region(self, bones, what):
        if not bones:
            self.log(f"WARNING body has no weights for the {what} region: using the whole surface")
            return self.skin
        sub = self.skin.where(bones, 0.5, name=what)
        if len(sub.tris) < 0.05 * len(self.skin.tris):
            self.log(f"WARNING {what} region has only {len(sub.tris)} triangles: using the whole surface")
            return self.skin
        return sub

    # ---- landmarks
    def lm(self, name):
        return self.L[name]

    def spine(self, z):
        """Approximate torso axis point (x, y) at height z from the spine landmarks."""
        pts = [self.L[k] for k in ("lower_body", "upper_body", "upper_body2", "neck", "head") if self.L.has(k)]
        pts = sorted(pts, key=lambda p: p[2])
        zs = np.array([p[2] for p in pts])
        return np.array([np.interp(z, zs, [p[0] for p in pts]), np.interp(z, zs, [p[1] for p in pts]), z])

    def ring_dirs(self, m, theta0=0.0):
        th = theta0 + G.ring_theta(m)
        return th, G.hdir(th)

    # ---- sections
    def torso_section(self, z, m=64, skin=None, mode="near", rmin=0.02, rmax=None):
        """(centre, radii (m,)) of the torso at height z (rays from the spine axis; angle 0 = front, +pi/2 = her left)."""
        sk = skin or self.torso
        c0 = self.spine(z)
        u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        c, r = sk.section(c0, u, v, m, mode, rmin=rmin, rmax=0.30 * self.S if rmax is None else rmax)
        if r is None:
            return None, None
        return c, r

    def waist_z(self, lo=None, hi=None, m=48):
        """Height of the narrowest torso section between the hip joints and the chest (the natural waist)."""
        ub, ub2 = self.L["upper_body"][2], self.L["upper_body2"][2]
        lo = ub - 0.06 * self.S if lo is None else lo
        hi = ub2 + 0.0 if hi is None else hi
        zs = np.linspace(lo, hi, 15)
        per = []
        for z in zs:
            c, r = self.torso_section(z, m)
            if r is None:
                per.append(np.inf)
                continue
            P = np.stack([r * np.sin(G.ring_theta(m)), -r * np.cos(G.ring_theta(m))], 1)
            per.append(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())
        per = np.array(per)
        if not np.isfinite(per).any():
            return float(ub + 0.01 * self.S)
        per = G.smooth_circular(np.where(np.isfinite(per), per, np.nanmax(per[np.isfinite(per)])), 1, 1) if False else per
        k = int(np.argmin(np.where(np.isfinite(per), per, np.inf)))
        return float(zs[k])

    def garment_ring(self, c, r, theta, off=0.0, dilate=2, smooth=(2, 2)):
        """Garment ring around a section: dilated + smoothed radii plus `off`, as (m, 3) points (angle theta)."""
        rr = G.dilate_circular(r, dilate) if dilate else np.asarray(r, float)
        rr = G.smooth_circular(rr, *smooth) if smooth else rr
        rr = np.maximum(rr, r) + off
        return c + G.hdir(theta) * rr[:, None], rr

    def weights_for(self, pts, bones=None, default="上半身", faces=None, smooth=0):
        """Body weights of the nearest body surface points (dict bone -> (n,)); `bones` restricts the donors; `faces`
        (the garment's faces) with `smooth` Laplacian-smooths them over the garment."""
        names = None if bones is None else set(bones)
        return self.skin.transfer(pts, names, default, faces, smooth)

    def push_out(self, pts, margin, iters=3):
        """Move points closer than `margin` to the body surface (or behind it) out to `margin` from the nearest surface
        point: along the line from that point when in front, along the surface normal when behind. Returns (n, 3)."""
        P = np.array(pts, float).reshape(-1, 3)
        for _ in range(iters):
            d, q, tri, _ = self.skin.closest(P)
            bad = d < margin - 1e-6
            if not bad.any():
                break
            A, B, C = (self.skin.verts[self.skin.tris[tri[bad], i]] for i in range(3))
            fn = G.unit(np.cross(B - A, C - A)) * (-1.0 if self.skin._flip else 1.0)
            n = P[bad] - q[bad]
            nl = np.linalg.norm(n, axis=1, keepdims=True)
            front = (d[bad] >= 0)[:, None] & (nl > 1e-6)
            P[bad] = q[bad] + np.where(front, n / np.maximum(nl, 1e-12), fn) * margin
        return P

    def conform(self, rings, margin, passes=3, closed=True, pin_first=False):
        """Rings (K, M, 3) pushed out of the body (`push_out`, so no skin shows through and every point keeps `margin`),
        with the displacement spread over the neighbouring vertices first so the cloth stays smooth; the margin is
        enforced again at the end. `pin_first` keeps ring 0 where it is (a frill's attachment)."""
        R = np.asarray(rings, float)
        K, M = R.shape[:2]
        P = R.reshape(-1, 3)
        D = (self.push_out(P, margin) - P).reshape(K, M, 3)
        if pin_first:
            D[0] = 0.0
        if np.abs(D).max() < 1e-9:
            return R
        for _ in range(passes):
            up = np.concatenate([D[:1], D[:-1]], 0)
            dn = np.concatenate([D[1:], D[-1:]], 0)
            lf = np.roll(D, 1, 1) if closed else np.concatenate([D[:, :1], D[:, :-1]], 1)
            rt = np.roll(D, -1, 1) if closed else np.concatenate([D[:, 1:], D[:, -1:]], 1)
            D = (2.0 * D + up + dn + lf + rt) / 6.0 * 1.15              # diffuse (slightly amplified: the margin is re-enforced)
            if pin_first:
                D[0] = 0.0
        out = self.push_out((P + D.reshape(-1, 3)), margin).reshape(K, M, 3)
        if pin_first:
            out[0] = R[0]
        return out
