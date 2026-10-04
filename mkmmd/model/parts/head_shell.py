"""A stand-in for `head_shape.HeadShape` built from a closed triangle shell (bpy-free, numpy only).

An imported face has no implicit formula, but everything the head part publishes (hairline, ear placement, ear anchors,
face outline, the star-shaped skin shell for the hair) is computed against `shape.phi / surface / normal / centre / prof`.
`RadialShape` answers the same calls from a radial map of the shell: R(theta, phi) = distance from `centre` to the first
triangle hit along each direction (theta the azimuth, 0 straight ahead (-y) and positive towards +x; phi the polar
angle from +z). Head-local coordinates, like the shape it replaces."""
import numpy as np

from . import head_shape as HS

TAU = 2.0 * np.pi


def ray_first_hit(origin, dirs, V, T, chunk=160):
    """Distance along each unit ray from `origin` to the first triangle hit (inf for a miss). V (n, 3), T (m, 3)."""
    V = np.asarray(V, float)
    T = np.asarray(T, int)
    dirs = np.asarray(dirs, float)
    A = V[T[:, 0]]
    e1 = V[T[:, 1]] - A
    e2 = V[T[:, 2]] - A
    tv = np.asarray(origin, float) - A
    qv = np.cross(tv, e1)
    t_num = np.einsum("tj,tj->t", e2, qv)
    out = np.full(len(dirs), np.inf)
    for s in range(0, len(dirs), chunk):
        d = dirs[s:s + chunk]
        pv = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("tj,rtj->rt", e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("tj,rtj->rt", tv, pv) * inv
        v = np.einsum("rj,tj->rt", d, qv) * inv
        t = t_num[None, :] * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
        out[s:s + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return out


def slice_profile(V, T, zs):
    """Half width, front and back of a closed shell at the heights `zs`: (hw, yf, yb), each (len(zs),). Front and back are
    the extreme y of the cross-section on the midline (|x| < 3 mm); NaN where the plane misses the shell."""
    V = np.asarray(V, float)
    tri = V[np.asarray(T, int)]
    zmin, zmax = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
    hw = np.full(len(zs), np.nan)
    yf = np.full(len(zs), np.nan)
    yb = np.full(len(zs), np.nan)
    for k, z in enumerate(zs):
        sel = (zmin <= z) & (zmax >= z)
        if not sel.any():
            continue
        tt = tri[sel]
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tt[:, i], tt[:, j]
            da, db = a[:, 2] - z, b[:, 2] - z
            m = (da * db <= 0) & (da != db)
            t = da[m] / (da[m] - db[m])
            pts.append(a[m] + (b[m] - a[m]) * t[:, None])
        P = np.concatenate(pts, 0)
        if not len(P):
            continue
        hw[k] = np.abs(P[:, 0]).max()
        mid = np.abs(P[:, 0]) < 0.003
        if mid.any():
            yf[k], yb[k] = P[mid, 1].min(), P[mid, 1].max()
    return hw, yf, yb


class _Profile:
    """The same interface as head_shape.Profiles: `at(z) -> (hw, yf, yb)`, the anchors `a` and the z grid."""

    def __init__(self, z, hw, yf, yb, anchors):
        self.z, self.hw, self.yf, self.yb, self.a = z, hw, yf, yb, anchors
        self.ztop, self.zbot = float(z[-1]), float(z[0])

    def at(self, z):
        z = np.asarray(z, float)
        return (np.interp(z, self.z, self.hw), np.interp(z, self.z, self.yf), np.interp(z, self.z, self.yb))


def _fill(a):
    """Replace NaNs by linear interpolation over the index."""
    a = np.array(a, float)
    bad = ~np.isfinite(a)
    if bad.any() and (~bad).any():
        a[bad] = np.interp(np.nonzero(bad)[0], np.nonzero(~bad)[0], a[~bad])
    return a


class RadialShape:
    """`phi(P)` negative inside, `surface(dirs)`, `normal(P)`, `centre`, `prof`, `ztop`, `zbot`: see head_shape.HeadShape."""

    def __init__(self, centre, R, prof=None):
        self.centre = np.asarray(centre, float)
        self.R = np.asarray(R, float)
        self.nphi, self.nth = self.R.shape
        self.dphi = np.pi / (self.nphi - 1)
        self.dth = TAU / self.nth
        self.prof = prof
        self.ztop = prof.ztop if prof is not None else None
        self.zbot = prof.zbot if prof is not None else None
        self.p = {}

    @classmethod
    def from_shell(cls, centre, V, T, res_deg=2.0, phi_open_deg=150.0, anchors=None):
        """Build the radial map by casting rays from `centre` at the shell (V (n, 3), T (m, 3)), head-local coordinates.
        Rays below `phi_open_deg` that miss (the open end of the neck) take the radius of the row above."""
        nphi = int(round(180.0 / res_deg)) + 1
        nth = int(round(360.0 / res_deg))
        th = -np.pi + np.arange(nth) * TAU / nth
        ph = np.arange(nphi) * np.pi / (nphi - 1)
        TH, PH = np.meshgrid(th, ph)
        D = np.stack([np.sin(PH) * np.sin(TH), -np.sin(PH) * np.cos(TH), np.cos(PH)], -1).reshape(-1, 3)
        R = ray_first_hit(centre, D, V, T).reshape(nphi, nth)
        miss = ~np.isfinite(R)
        bad = miss & (ph[:, None] < np.radians(phi_open_deg))
        if bad.any():
            raise ValueError(f"RadialShape: {int(bad.sum())} rays above {phi_open_deg:.0f} deg from the centre leave the shell open")
        for i in range(1, nphi):
            R[i] = np.where(np.isfinite(R[i]), R[i], R[i - 1])
        V = np.asarray(V, float)
        zs = np.arange(float(V[:, 2].min()) + 1e-3, float(V[:, 2].max()) - 1e-4, 0.0015)
        hw, yf, yb = slice_profile(V, T, zs)
        hw, yf, yb = _fill(hw), _fill(yf), _fill(yb)
        A = dict(HS.ANCHORS)
        A.update(anchors or {})
        top = int(np.argmax(V[:, 2]))
        A["crown"] = (float(V[top, 1]), float(V[top, 2]))
        sh = cls(centre, R)
        sh.prof = _Profile(zs, hw, yf, yb, A)
        sh.ztop, sh.zbot = sh.prof.ztop, sh.prof.zbot
        return sh

    # ---- the radial function
    def radius(self, d):
        d = np.asarray(d, float)
        th = np.arctan2(d[..., 0], -d[..., 1])
        ph = np.arccos(np.clip(d[..., 2] / np.maximum(np.linalg.norm(d, axis=-1), 1e-12), -1.0, 1.0))
        fi = np.clip(ph / self.dphi, 0.0, self.nphi - 1 - 1e-9)
        fj = ((th + np.pi) / self.dth) % self.nth
        i0 = fi.astype(int)
        j0 = fj.astype(int) % self.nth
        j1 = (j0 + 1) % self.nth
        a, b = fi - i0, fj - np.floor(fj)
        R = self.R
        return (1 - a) * ((1 - b) * R[i0, j0] + b * R[i0, j1]) + a * ((1 - b) * R[i0 + 1, j0] + b * R[i0 + 1, j1])

    def phi(self, P, neck=True):
        d = np.asarray(P, float) - self.centre
        r = np.linalg.norm(d, axis=-1)
        return r - self.radius(d)

    def surface(self, dirs, **_):
        dirs = np.asarray(dirs, float)
        u = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
        return self.centre + u * self.radius(u)[..., None]

    def normal(self, P, h=2.5e-3):
        P = np.asarray(P, float)
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            g[..., i] = self.phi(P + e) - self.phi(P - e)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
