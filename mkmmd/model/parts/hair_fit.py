"""Head fit for the hair builders (bpy-free, numpy only; runs on numpy 1.24 and 2.x).

The hair is built around the head part's published data (`ctx.parts["head"].info`, keys documented by the head part:
head_center, skin {verts, faces}, hairline, hairline_side, nape, ears, cat_ear_anchors, eyes, brows ...).
`HeadFit` reads them; `Skull` is the star-shaped skin surface seen from the head centre.

Angles used everywhere in the hair code (model space, Z up, the character faces -Y, her LEFT is +X):
  theta  azimuth from the front (-Y) towards the character's left (+X): front 0, left +90 deg, back 180, right -90
  phi    polar angle from +Z: crown 0, horizon (ear-hole height) 90 deg, below the chin 180
`dirs(theta, phi)` and `angles(d)` convert; `Skull.point(d, off)` is the skin point in direction d pushed `off` metres
radially outwards."""
import numpy as np

TAU = 2.0 * np.pi


def dirs(theta, phi):
    """Unit vectors from azimuth theta and polar angle phi (radians, broadcasting)."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    sp = np.sin(phi)
    return np.stack([sp * np.sin(theta), -sp * np.cos(theta), np.cos(phi)], -1)


def angles(d):
    """(theta, phi) of direction vectors (need not be unit)."""
    d = np.asarray(d, float)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return np.arctan2(d[..., 0], -d[..., 1]), np.arccos(np.clip(d[..., 2], -1.0, 1.0))


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def raycast_hits(origin, rays, verts, tris, chunk=48):
    """First triangle hit by each unit ray from `origin`: (t (inf when none), triangle index (-1), barycentric u, v) with the
    hit point = v0 + u (v1 - v0) + v (v2 - v0). verts (n,3), tris (m,3)."""
    verts = np.asarray(verts, float)
    tris = np.asarray(tris, int)
    v0 = verts[tris[:, 0]]
    e1 = verts[tris[:, 1]] - v0
    e2 = verts[tris[:, 2]] - v0
    rel = np.asarray(origin, float) - v0
    n = len(rays)
    out_t, out_i = np.full(n, np.inf), np.full(n, -1)
    out_u, out_v = np.zeros(n), np.zeros(n)
    for a in range(0, n, chunk):
        d = rays[a:a + chunk]
        p = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("tj,ctj->ct", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("tj,ctj->ct", rel, p) * inv
        q = np.cross(rel, e1)                                  # (t,3)
        v = np.einsum("cj,tj->ct", d, q) * inv
        t = np.einsum("tj,tj->t", e2, q)[None, :] * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
        tt = np.where(hit, t, np.inf)
        k = tt.argmin(axis=1)
        rows = np.arange(len(d))
        best = tt[rows, k]
        out_t[a:a + chunk] = best
        out_i[a:a + chunk] = np.where(np.isfinite(best), k, -1)
        out_u[a:a + chunk] = u[rows, k]
        out_v[a:a + chunk] = v[rows, k]
    return out_t, out_i, out_u, out_v


def raycast_first(origin, rays, verts, tris, chunk=48):
    """Distance along each unit ray from `origin` to the first triangle hit (inf when none). verts (n,3), tris (m,3)."""
    return raycast_hits(origin, rays, verts, tris, chunk)[0]


class Skull:
    """The skin surface as a radial function of direction about `center`: R[i, j] at phi = i * dphi (i = 0..nphi-1,
    the last row is phi = pi) and theta = -pi + j * dtheta (periodic). Everything is metres in model space."""

    def __init__(self, center, R):
        self.center = np.asarray(center, float)
        self.R = np.asarray(R, float)
        self.nphi, self.nth = self.R.shape
        self.dphi = np.pi / (self.nphi - 1)
        self.dth = TAU / self.nth

    # ---------------------------------------------------------------- construction
    @staticmethod
    def grid_dirs(res_deg=2.0):
        nphi = int(round(180.0 / res_deg)) + 1
        nth = int(round(360.0 / res_deg))
        th = -np.pi + np.arange(nth) * TAU / nth
        ph = np.arange(nphi) * np.pi / (nphi - 1)
        T, P = np.meshgrid(th, ph)
        return dirs(T, P), nphi, nth

    @classmethod
    def from_mesh(cls, center, verts, faces, res_deg=2.0, phi_open_deg=145.0):
        """Radius per direction = first skin hit of a ray from `center`. The cranium and face must be closed; rays
        below `phi_open_deg` (down the neck, whose shell may be open at the bottom) that miss take the radius of the
        nearest hit above them."""
        D, nphi, nth = cls.grid_dirs(res_deg)
        tris = np.asarray([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)], int)
        R = raycast_first(center, D.reshape(-1, 3), verts, tris).reshape(nphi, nth)
        phi = np.arange(nphi) * np.pi / (nphi - 1)
        miss = ~np.isfinite(R)
        bad = miss & (phi[:, None] < np.radians(phi_open_deg))
        if bad.any():
            raise ValueError(f"Skull.from_mesh: {int(bad.sum())} rays above {phi_open_deg:.0f} deg from the head centre "
                             f"leave the skin shell open")
        for i in range(1, nphi):                                   # fill the open neck end from the row above
            R[i] = np.where(np.isfinite(R[i]), R[i], R[i - 1])
        return cls(center, R)

    @classmethod
    def from_function(cls, center, fn, res_deg=2.0):
        D, nphi, nth = cls.grid_dirs(res_deg)
        return cls(center, fn(D))

    # ---------------------------------------------------------------- queries
    def radius(self, d):
        """Bilinear radius for directions d (..., 3)."""
        th, ph = angles(d)
        fi = np.clip(ph / self.dphi, 0.0, self.nphi - 1 - 1e-9)
        fj = ((th + np.pi) / self.dth) % self.nth
        i0 = fi.astype(int)
        j0 = fj.astype(int) % self.nth
        j1 = (j0 + 1) % self.nth
        a, b = fi - i0, fj - np.floor(fj)
        R = self.R
        return ((1 - a) * ((1 - b) * R[i0, j0] + b * R[i0, j1]) + a * ((1 - b) * R[i0 + 1, j0] + b * R[i0 + 1, j1]))

    def point(self, d, off=0.0):
        """Skin point in direction d, `off` metres further out along d (broadcasting)."""
        d = unit(d)
        r = self.radius(d) + off
        return self.center + d * np.asarray(r)[..., None]

    def height_dist(self, p):
        """Signed radial distance of points above the skin (> 0 outside), measured along the ray from the centre."""
        v = np.asarray(p, float) - self.center
        n = np.linalg.norm(v, axis=-1)
        return n - self.radius(v)

    def normal(self, d, eps=0.02):
        """Outward unit normal of the skin in direction d, from central differences of the radial surface."""
        d = unit(d)
        flat = d.reshape(-1, 3)
        up = np.where(np.abs(flat[:, 2:3]) > 0.95, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
        e1 = unit(np.cross(up, flat))
        e2 = unit(np.cross(flat, e1))
        t1 = self.point(unit(flat + eps * e1)) - self.point(unit(flat - eps * e1))
        t2 = self.point(unit(flat + eps * e2)) - self.point(unit(flat - eps * e2))
        n = unit(np.cross(t1, t2))
        n = np.where((n * flat).sum(-1, keepdims=True) < 0, -n, n)
        return n.reshape(d.shape)

    def push_out(self, p, margin):
        """Points moved radially outwards so that they clear the skin by `margin` (those already clear are unchanged)."""
        v = np.asarray(p, float) - self.center
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        d = v / np.maximum(n, 1e-12)
        r = np.maximum(n[..., 0], self.radius(d) + margin)
        return self.center + d * r[..., None]

    # ---------------------------------------------------------------- shaping
    def puff(self, centre, axes, semi, k=0.008):
        """Smooth-max the radial surface with an ellipsoid (centre, orthonormal axes (3,3) rows, semi-axes (3,)): the
        hair hull over an ear. Directions whose ray misses the ellipsoid are unchanged."""
        D, _, _ = self.grid_dirs(180.0 / (self.nphi - 1))
        D = D.reshape(-1, 3)
        o = (self.center - np.asarray(centre, float)) @ np.asarray(axes, float).T / np.asarray(semi, float)
        dd = D @ np.asarray(axes, float).T / np.asarray(semi, float)
        a = (dd * dd).sum(1)
        b = 2.0 * (o * dd).sum(1)
        c = (o * o).sum() - 1.0
        disc = b * b - 4 * a * c
        far = np.where(disc > 0, (-b + np.sqrt(np.maximum(disc, 0.0))) / (2 * a), 0.0)   # exit distance
        far = far.reshape(self.R.shape)
        m = far > 0
        hi = np.maximum(self.R, far)
        soft = hi + k * np.log1p(np.exp(-np.abs(self.R - far) / k))
        self.R = np.where(m, np.minimum(soft, hi + k), self.R)
        return self


def _poly_angles(center, pts):
    pts = np.asarray(pts, float)
    th, ph = angles(pts - center)
    return th, ph


class HeadFit:
    """The head part's published data, digested for the hair builders. `info` is `ctx.parts["head"].info`."""

    REQUIRED = ("head_center", "skin", "hairline", "eyes")

    def __init__(self, info, res_deg=2.0, puff_ears=False, meshes=None):
        miss = [k for k in self.REQUIRED if k not in info]
        if miss:
            raise KeyError(f"head part info lacks {miss}")
        self.info = info
        self.center = np.asarray(info["head_center"], float)
        self.face = None                       # the head part's `face` mesh, for decals that must lie exactly on it
        for m in meshes or ():
            if getattr(m, "name", "") == "face":
                v = np.asarray(m.verts, float)
                tris, corners, tri_face, c0 = [], [], [], 0
                for fi, f in enumerate(m.faces):
                    for k in range(1, len(f) - 1):
                        tris.append((f[0], f[k], f[k + 1]))
                        corners.append((c0, c0 + k, c0 + k + 1))
                        tri_face.append(fi)
                    c0 += len(f)
                self.face = {"verts": v, "tris": np.asarray(tris, int), "corners": np.asarray(corners, int),
                             "tri_face": np.asarray(tri_face, int),
                             "uv": None if m.uv is None else np.asarray(m.uv, float),
                             "face_mat": None if m.face_mat is None else np.asarray(m.face_mat, int), "mats": list(m.mats),
                             "morphs": {k: np.asarray(d, float) for k, d in m.morphs.items()},
                             "normals": None if m.normals is None else np.asarray(m.normals, float)}
        skin = info["skin"]
        self.skull = Skull.from_mesh(self.center, skin["verts"], skin["faces"], res_deg=res_deg)
        self.top = np.asarray(info.get("skull_top", self.skull.point([0, 0, 1])), float)
        self.chin = np.asarray(info.get("chin", self.skull.point([0, -0.3, -1])), float)
        self.brow_front = np.asarray(info.get("brow_front", self.skull.point(dirs(0.0, np.radians(80)))), float)
        self.hairline = np.asarray(info["hairline"], float)
        self.hairline_side = {k: np.asarray(v, float) for k, v in (info.get("hairline_side") or {}).items()}
        self.nape = np.asarray(info["nape"], float) if info.get("nape") is not None else None
        self.eyes = info["eyes"]
        self.brows = info.get("brows") or {}
        self.ears = info.get("ears") or {}
        self.cat = info.get("cat_ear_anchors") or {}
        self.neck_center = np.asarray(info["neck_center"], float) if info.get("neck_center") is not None else None
        if puff_ears:                                           # lift the hull over human ears that stick out of the hair
            for side, e in self.ears.items():
                self._puff_ear(e)
        self._cap = self._cap_curve()
        self.vol = Volume(self.skull, taper_from=self._ear_taper_phi())      # hair.py may replace it (from [hair.volume])

    # ---------------------------------------------------------------- ears under the hair
    def _ear_taper_phi(self):
        """Polar angle below which the hair volume may taper: just under the human ears (they must stay hidden)."""
        phis = []
        for e in self.ears.values():
            _, p = _poly_angles(self.center, [e["lobe"]])
            phis.append(float(p[0]))
        return max(phis) + 0.12 if phis else np.radians(100.0)

    def _puff_ear(self, e, margin=0.006):
        top, lobe = np.asarray(e["top"], float), np.asarray(e["lobe"], float)
        front, back = np.asarray(e["front"], float), np.asarray(e["back"], float)
        out = unit(e["out"])
        centre = (top + lobe + front + back) / 4.0 + out * 0.008
        up = unit(top - lobe)
        fb = unit(back - front)
        fb = unit(fb - up * float(fb @ up))
        out = unit(np.cross(up, fb))
        if out @ np.asarray(e["out"], float) < 0:
            out = -out
        axes = np.stack([up, fb, out])
        semi = np.array([np.linalg.norm(top - lobe) / 2 + margin, np.linalg.norm(back - front) / 2 + margin,
                         0.020 + margin])
        self.skull.puff(centre, axes, semi)

    # ---------------------------------------------------------------- the region the scalp cap must cover
    def _cap_curve(self):
        """Sorted (theta, phi) lower boundary of the hair region as a function of azimuth: the hairline across the front; from
        each temple a smooth S-ramp down to the jaw end of the sideburn line (the line runs at its own azimuth, often 10-20 deg
        behind the temple: a step there would give the cap quads that cut chords through the head); behind it straight on to
        the nape polyline, which runs round the back (periodic in theta)."""
        th_h, ph_h = _poly_angles(self.center, self.hairline)
        pts_t, pts_p = list(th_h), list(ph_h)
        for sg in (1.0, -1.0):
            k = int(np.argmax(sg * th_h))
            theta_t, phi_t = float(th_h[k]), float(ph_h[k])
            side = self.hairline_side.get("L" if sg > 0 else "R")
            if side is not None and len(side):
                ts, ps = _poly_angles(self.center, side)
                theta_s, phi_b = float(ts[-1]), max(float(ps[-1]), phi_t)
            else:
                theta_s, phi_b = theta_t + sg * np.radians(12.0), np.radians(100.0)
            if sg * (theta_s - theta_t) < np.radians(8.0):         # a sideburn line right at the temple: ramp over >= 8 deg
                theta_s = theta_t + sg * np.radians(8.0)
            for u in np.linspace(0.0, 1.0, 9)[1:]:
                sm = u * u * (3.0 - 2.0 * u)
                pts_t.append(theta_t + (theta_s - theta_t) * u)
                pts_p.append(phi_t + (phi_b - phi_t) * sm)
        if self.nape is not None and len(self.nape):
            tn, pn = _poly_angles(self.center, self.nape)
            pts_t += list(tn)
            pts_p += list(pn)
        else:                                                  # no nape polyline: the back of the head ends at 125 deg
            pts_t += [np.radians(110), np.radians(180), np.radians(-110)]
            pts_p += [np.radians(125)] * 3
        th = np.array(pts_t)
        ph = np.array(pts_p)
        o = np.argsort(th)
        return th[o], ph[o]

    def cap_phi(self, theta):
        """Polar angle (rad) down to which the scalp cap reaches at azimuth theta (the lower hairline)."""
        th, ph = self._cap
        return np.interp(np.asarray(theta, float), th, ph, period=TAU)

    def hair_region_dirs(self, step_deg=6.0, inset_deg=4.0, phi_min_deg=8.0):
        """Unit directions sampling the region the hair must cover: from phi_min to the lower hairline minus an
        inset, every step_deg (used by the coverage test and published for other parts)."""
        out = []
        for th in np.radians(np.arange(-180.0, 180.0, step_deg)):
            top = np.radians(phi_min_deg)
            bot = float(self.cap_phi(th)) - np.radians(inset_deg)
            if bot > top:
                ph = np.arange(top, bot, np.radians(step_deg))
                out.append(dirs(np.full(len(ph), th), ph))
        return np.concatenate(out, 0)

    def on_face(self, points, tol=0.006):
        """Points moved radially (from the head centre) onto the head part's `face` mesh where it lies within `tol` metres of
        them (the others stay), with the UV there, the name of the face's material and the hit (triangle index, barycentric u,
        v, hit mask) so that morph offsets can be carried over. Returns (points, uv (n,2) or None, material name or None,
        hit or None); the last three are None without a face mesh or UVs."""
        points = np.asarray(points, float)
        if self.face is None:
            return points, None, None, None
        F = self.face
        d = points - self.center
        r = np.linalg.norm(d, axis=-1)
        u = d / np.maximum(r[..., None], 1e-12)
        t, idx, bu, bv = raycast_hits(self.center, u.reshape(-1, 3), F["verts"], F["tris"])
        t, idx, bu, bv = (a.reshape(r.shape) for a in (t, idx, bu, bv))
        ok = np.isfinite(t) & (np.abs(t - r) < tol)
        out = np.where(ok[..., None], self.center + u * np.where(ok, t, r)[..., None], points)
        uv, mat = None, None
        if F["uv"] is not None:
            c = F["corners"][np.maximum(idx, 0)]
            uv = (1 - bu - bv)[..., None] * F["uv"][c[..., 0]] + bu[..., None] * F["uv"][c[..., 1]] + bv[..., None] * F["uv"][c[..., 2]]
            if F["face_mat"] is not None and ok.any():
                names = [F["mats"][F["face_mat"][F["tri_face"][i]]] for i in idx[ok]]
                mat = max(set(names), key=names.count)
        return out, uv, mat, (idx, bu, bv, ok)

    def face_normals(self, hit, fallback):
        """Unit normals for points bound to the face mesh by `on_face`'s hit: the barycentric blend of the face mesh's vertex
        normals (its custom anime-face normals when it has them), so a decal shades exactly like the skin under it; `fallback`
        (n,3) where there is no face mesh or the point is not on it."""
        fallback = np.asarray(fallback, float)
        if self.face is None or self.face["normals"] is None or hit is None:
            return fallback
        idx, bu, bv, ok = hit
        tri = self.face["tris"][np.maximum(idx, 0)]
        N = self.face["normals"]
        n = (1 - bu - bv)[:, None] * N[tri[:, 0]] + bu[:, None] * N[tri[:, 1]] + bv[:, None] * N[tri[:, 2]]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return np.where(ok[:, None], n, fallback)

    def face_morphs(self, hit, ok_only=True):
        """Morph offsets {name: (n,3)} for points bound to the face mesh by `on_face`'s hit: the barycentric blend of the
        offsets of the triangle's vertices (zero where the point is not on the face mesh); morphs that do not move them are
        left out."""
        if self.face is None or hit is None:
            return {}
        idx, bu, bv, ok = hit
        tri = self.face["tris"][np.maximum(idx, 0)]
        out = {}
        for name, D in self.face["morphs"].items():
            o = (1 - bu - bv)[:, None] * D[tri[:, 0]] + bu[:, None] * D[tri[:, 1]] + bv[:, None] * D[tri[:, 2]]
            o = np.where(ok[:, None], o, 0.0)
            if np.abs(o).max() > 1e-7:
                out[name] = o
        return out

    # ---------------------------------------------------------------- face landmarks
    def lid_top(self, side):
        """Upper-lid top at the pupil: the head part's `lid_top` when it publishes one, else a third of the eyeball
        radius above its centre (the opening of these big eyes is about that high)."""
        e = self.eyes[side]
        if "lid_top" in e:
            return np.asarray(e["lid_top"], float)
        return np.asarray(e["center"], float) + np.array([0.0, 0.0, 0.33 * float(e["radius"])])

    def eye_z(self):
        """Mean height of the upper lids at the pupil (the bangs stop above it)."""
        return float(np.mean([self.lid_top(s)[2] for s in self.eyes]))

    def brow_z(self):
        zs = [np.asarray(p, float)[:, 2].mean() for p in self.brows.values()]
        return float(np.mean(zs)) if zs else self.eye_z() + 0.025

    def ear_centre(self, side):
        e = self.ears.get(side)
        if e is None:
            return None
        return (np.asarray(e["top"], float) + np.asarray(e["lobe"], float)) / 2


class Volume:
    """How far the hair stands off the skin: radial offset `vol(theta, phi)` of the hair hull over `skull`. Built from
    the head's own shape: per azimuth the polar angle phi_eq of the widest part (max of R sin(phi)); the offset grows from
    `top` at the crown to the azimuthal value `v_eq(theta)` there (front, side and back values interpolated by
    a + b cos(theta) + c cos(2 theta)), stays there down to `taper_from` (below the human ears, which the hair must
    hide) and then tapers off towards the neck. `lift` adds a crown puff: a Gaussian bump in the polar angle around
    `lift_phi` (width `lift_width`), stronger on the back of the crown (`lift_back`: 0 = all round, 1 = back only), so the
    hair rises over the skull instead of lying on it like a smooth dome."""

    def __init__(self, skull, top=0.028, front=0.020, side=0.034, back=0.027, taper=0.55, taper_from=None,
                 lift=0.0, lift_phi=np.radians(32.0), lift_width=np.radians(24.0), lift_back=0.5):
        self.skull = skull
        self.top = top
        a_plus_c = 0.5 * (front + back)
        b = 0.5 * (front - back)
        c_ = 0.5 * (a_plus_c - side)
        a = side + c_
        self.coef = (a, b, c_)                    # v_eq(theta) = a + b cos(theta) + c cos(2 theta)
        self.taper = taper
        self.taper_from = np.radians(100.0) if taper_from is None else float(taper_from)
        self.lift, self.lift_phi = float(lift), float(lift_phi)
        self.lift_width, self.lift_back = float(lift_width), float(lift_back)
        th = -np.pi + np.arange(skull.nth) * skull.dth
        ph = np.arange(skull.nphi) * skull.dphi
        h = skull.R * np.sin(ph)[:, None]                         # horizontal reach per (phi, theta)
        lo = int(np.radians(40) / skull.dphi)
        hi = int(np.radians(125) / skull.dphi)
        k = lo + np.argmax(h[lo:hi], axis=0)
        self._th = th
        self._peq = ph[k]

    def v_eq(self, theta):
        a, b, c = self.coef
        theta = np.asarray(theta, float)
        return a + b * np.cos(theta) + c * np.cos(2 * theta)

    def phi_eq(self, theta):
        return np.interp(np.asarray(theta, float), self._th, self._peq, period=TAU)

    def __call__(self, theta, phi):
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        peq = self.phi_eq(theta)
        t = phi / np.maximum(peq, 1e-6)
        veq = self.v_eq(theta)
        v = np.where(t <= 1.0, self.top + (veq - self.top) * smoothstep(t) ** 1.2, veq)
        if self.lift:
            g = np.exp(-(((phi - self.lift_phi) / self.lift_width) ** 2))
            v = v + self.lift * g * (1.0 + self.lift_back * np.cos(theta - np.pi)) / (1.0 + self.lift_back)
        p0 = np.maximum(peq, self.taper_from)
        return v * (1.0 - self.taper * smoothstep((phi - p0) / 0.8))

    def max_z(self, step_deg=3.0):
        """Height (m) of the highest point of the hull (skin + volume)."""
        D, _, _ = Skull.grid_dirs(step_deg)
        th, ph = angles(D)
        return float(self.skull.point(D, self(th, ph))[..., 2].max())
