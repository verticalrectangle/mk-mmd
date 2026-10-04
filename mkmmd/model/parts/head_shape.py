"""The head's base surface: an implicit shape sampled along rays from the skull centre.

Everything here is in HEAD-LOCAL coordinates: origin at the head bone (the neck/skull joint), +x = the character's left,
-y = forward, +z = up, metres (Rin-sized: the skull top 0.195 m above the bone). `head.py` places the result in model space.

The shape is a superellipse loft: at every height the horizontal section has the half width, front plane and back of
`Profiles`, which are smooth interpolants (monotone cubics, elliptical caps) of a few named proportions (`ANCHORS`: the
skull top, widths at the temples / eye level / cheeks / mouth / chin, the chin point, forehead, face-plane depths, the
back of the skull). Soft blobs add the muzzle and a hint of nose; a neck column blends in under the chin. The surface is
star-shaped about `centre`, so a grid of directions (theta around the vertical axis, phi elevation) gives one vertex per
direction. Signed values are pseudo-distances (metres, negative inside)."""
import numpy as np

# --------------------------------------------------------------------------------------------- small implicit helpers


def smin(a, b, k):
    """Polynomial smooth minimum (the union of two fields blended over a distance k)."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def round_intersect(a, b, r):
    """Intersection of two half-spaces whose corner is rounded by the radius r (the result stays inside the corner)."""
    ux = np.maximum(a + r, 0.0)
    uy = np.maximum(b + r, 0.0)
    return np.minimum(-r, np.maximum(a, b)) + np.hypot(ux, uy)


def ellipsoid(P, c, r, p=2.0):
    """Pseudo-distance to a (super)ellipsoid with centre c and radii r (exponent p, 2 = ellipsoid)."""
    q = np.abs((P - np.asarray(c)) / np.asarray(r))
    if p == 2.0:
        n = np.sqrt((q * q).sum(-1))
    else:
        n = (q ** p).sum(-1) ** (1.0 / p)
    return (n - 1.0) * float(np.min(r))


def pchip(x, y, d_end=(None, None)):
    """Monotone piecewise-cubic Hermite interpolant through (x, y) (x increasing); returns f(xq). `d_end`: optional end
    slopes (None = one-sided estimate)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = np.diff(x)
    dl = np.diff(y) / h
    d = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        if dl[i - 1] * dl[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])
    for end, (i, j, hh, hh2) in enumerate(((0, 1, 0, 1), (-1, -2, -1, -2))):
        if d_end[end] is not None:
            d[i] = d_end[end]
        elif len(x) > 2:
            k = 0 if end == 0 else len(x) - 1
            h0, h1 = (h[0], h[1]) if end == 0 else (h[-1], h[-2])
            m0, m1 = (dl[0], dl[1]) if end == 0 else (dl[-1], dl[-2])
            e = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            if e * m0 <= 0:
                e = 0.0
            elif m0 * m1 <= 0 and abs(e) > 3 * abs(m0):
                e = 3 * m0
            d[i] = e
        else:
            d[i] = dl[0]

    def f(xq):
        xq = np.clip(np.asarray(xq, float), x[0], x[-1])
        k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
        t = (xq - x[k]) / h[k]
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t * t * (3 - 2 * t)
        h11 = t * t * (t - 1)
        return h00 * y[k] + h10 * h[k] * d[k] + h01 * y[k + 1] + h11 * h[k] * d[k + 1]
    return f


# --------------------------------------------------------------------------------------------------- the head profiles
# The head is described by a few named proportions (head-local metres: origin at the head bone, +z up, -y forward):
# the skull top, widths at five named heights (temples, eye level, cheeks, mouth, chin), the chin point, the front-most
# forehead point, the face-plane depth at the eyes, the mouth and the nose, and the back of the skull. All profiles
# between them are smooth interpolants (monotone cubics and elliptical caps) of these numbers.
ANCHORS = dict(
    crown=(-0.0088, 0.1947),                  # (y, z) of the skull top
    temple=(0.1209, 0.0948), eye=(0.0563, 0.0827), cheek=(0.0219, 0.0711), mouth=(0.0029, 0.0599),
    chin=(-0.0171, 0.0294),                   # (z, half width) of the five named widths
    chin_point=(-0.0766, -0.0263),            # (y, z): front-lowest point of the chin
    forehead=(-0.1090, 0.1165),               # (y, z): front-most point of the forehead
    brow_y=-0.1070, brow_z=0.0963,            # face plane at the brows
    eye_plane_y=-0.1010,                      # face plane on the midline at eye height
    philtrum=(-0.0990, 0.0172),               # (y, z) face plane between the nose and the mouth
    mouth_plane=(-0.0945, 0.0010),            # (y, z) face plane at the mouth line
    chin_plane=(-0.0850, -0.0100),            # (y, z) between mouth and chin point
    back=(0.0884, 0.0950),                    # (y, z): back-most point of the skull
    nape=((0.0668, 0.0460), (0.0400, 0.0210), (0.0120, 0.0), (-0.0250, -0.0120)),   # back/underside points (y, z)
)


class Profiles:
    """Half width hw(z), front plane yf(z) and back yb(z) of the head as functions of the height z (tables at 0.05 mm)."""

    def __init__(self, a=None):
        A = dict(ANCHORS)
        A.update(a or {})
        self.a = A
        cy, cz = A["crown"]
        zs = np.arange(A["chin_point"][1] - 1e-3, cz + 1e-3, 5e-5)
        self.z = zs
        # half width: elliptical cap above the temples, monotone cubic between the named widths, a V chin below
        zt, wt = A["temple"]
        names = ["chin", "mouth", "cheek", "eye", "temple"]
        pts_w = sorted([(float(A[n][0]), float(A[n][1])) for n in names] + [(float(z), float(w)) for z, w in A.get("hw_extra", ())])
        zk = np.array([p[0] for p in pts_w])
        wk = np.array([p[1] for p in pts_w])
        mid = pchip(zk, wk, d_end=(None, 0.0))
        zc, wc = A["chin"]
        zp = A["chin_point"][1]
        hw = np.zeros_like(zs)
        up = zs >= zt
        hw[up] = wt * np.sqrt(np.clip(1.0 - ((zs[up] - zt) / (cz - zt)) ** 2, 0.0, 1.0))
        md = (zs < zt) & (zs >= zc)
        hw[md] = mid(zs[md])
        lo = zs < zc
        hw[lo] = wc * np.clip((zs[lo] - zp) / (zc - zp), 0.0, 1.0) ** 0.6
        self.hw = hw
        # front face plane: monotone cubic through the named depths, an elliptical cap above the forehead
        yf_pts = sorted([(A["chin_point"][1], A["chin_point"][0]), (A["chin_plane"][1], A["chin_plane"][0]),
                         (A["mouth_plane"][1], A["mouth_plane"][0]), (A["philtrum"][1], A["philtrum"][0]),
                         (A["eye"][0], A["eye_plane_y"]), (A["brow_z"], A["brow_y"]), (A["forehead"][1], A["forehead"][0])]
                       + [(float(z), float(y)) for z, y in A.get("front_extra", ())])
        zf, yfk = zip(*yf_pts)
        mid_f = pchip(zf, yfk, d_end=(None, 0.0))
        yf = np.zeros_like(zs)
        z_f = A["forehead"][1]
        up = zs >= z_f
        yf[up] = cy + (A["forehead"][0] - cy) * np.sqrt(np.clip(1.0 - ((zs[up] - z_f) / (cz - z_f)) ** 2, 0.0, 1.0))
        yf[~up] = mid_f(zs[~up])
        self.yf = yf
        # back: elliptical cap above the widest point, then monotone cubic down the nape and under the jaw
        by, bz = A["back"]
        yb = np.zeros_like(zs)
        up = zs >= bz
        yb[up] = cy + (by - cy) * np.sqrt(np.clip(1.0 - ((zs[up] - bz) / (cz - bz)) ** 2, 0.0, 1.0))
        pts = [(bz, by)] + [(z, y) for y, z in A["nape"]] + [(A["chin_point"][1], A["chin_point"][0])]
        pts = sorted(set(pts))
        zk2, yk2 = zip(*pts)
        mid_b = pchip(zk2, yk2, d_end=(None, 0.0))
        yb[~up] = mid_b(zs[~up])
        self.yb = yb
        self.ztop, self.zbot = float(cz), float(A["chin_point"][1])

    def at(self, z):
        z = np.asarray(z, float)
        return (np.interp(z, self.z, self.hw), np.interp(z, self.z, self.yf), np.interp(z, self.z, self.yb))


# --------------------------------------------------------------------------------------------------- the head shape
CENTRE = np.array([0.0, 0.0, 0.055])             # skull centre of the sampling rays (on the neck axis, eye height)


class HeadShape:
    """Implicit head. `phi(P)` is negative inside; `surface(dirs)` returns the surface point along each direction
    from `centre`; `normal(P)` the unit gradient. `anchors` overrides ANCHORS (see above)."""

    def __init__(self, anchors=None, **kw):
        self.p = dict(
            n_front=2.35, n_back=2.1,                     # section exponents (2 = ellipse); the face side is squarer
            muzzle=dict(c=(0.0, -0.088, 0.008), r=(0.017, 0.012, 0.013), k=0.012),
            cheek=dict(c=(0.050, -0.078, 0.014), r=(0.024, 0.019, 0.016), k=0.016, on=False),     # optional cheek apples (they made lumps)
            nose=dict(c=(0.0, -0.1000, 0.0290), r=(0.0030, 0.0027, 0.0040), k=0.005),
            neck=dict(c=(0.0, 0.0, 0.0), rx=0.0265, ry=0.0290, k=0.014, top=0.02),
        )
        self.p.update(kw)
        self.centre = np.array([0.0, float(self.p["neck"]["c"][1]), CENTRE[2]])
        self.prof = Profiles(anchors)
        self.ztop, self.zbot = self.prof.ztop, self.prof.zbot

    def hull(self, P):
        """Superellipse loft: at each height the section has the half width, front plane and back of the profiles
        (pseudo-distance, negative inside)."""
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        hw, yf, yb = self.prof.at(z)
        hw = np.maximum(hw, 1e-4)
        yc = 0.5 * (yf + yb)
        fr = y < yc
        hd = np.maximum(np.where(fr, yc - yf, yb - yc), 1e-4)
        n = np.where(fr, self.p["n_front"], self.p["n_back"])
        u = np.abs(x) / hw
        v = np.abs(y - yc) / hd
        S = (u ** n + v ** n) ** (1.0 / n)
        d = (S - 1.0) * np.minimum(hw, hd)
        return np.maximum(d, np.maximum(z - self.ztop, self.zbot - z))

    def phi(self, P, neck=True):
        P = np.asarray(P, float)
        d = self.hull(P)
        b = self.p["muzzle"]
        d = smin(d, ellipsoid(P, b["c"], b["r"]), b["k"])
        b = self.p["cheek"]
        if b.get("on"):
            for sx in (1.0, -1.0):
                d = smin(d, ellipsoid(P, (sx * b["c"][0], b["c"][1], b["c"][2]), b["r"]), b["k"])
        n = self.p["nose"]
        d = smin(d, ellipsoid(P, n["c"], n["r"]), n["k"])
        if neck:
            nk = self.p["neck"]
            r = np.hypot((P[..., 0] - nk["c"][0]) / nk["rx"], (P[..., 1] - nk["c"][1]) / nk["ry"])
            dn = np.maximum((r - 1.0) * min(nk["rx"], nk["ry"]), P[..., 2] - nk["top"])    # column stops inside the skull
            d = smin(d, dn, nk["k"])
        return d

    def normal(self, P, h=4e-4):
        P = np.asarray(P, float)
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            g[..., i] = self.phi(P + e) - self.phi(P - e)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)

    def surface(self, dirs, tmax=0.35, iters=44):
        """Points where rays from `centre` along `dirs` (n, 3) leave the shape (bisection; the shape is star-shaped)."""
        dirs = np.asarray(dirs, float)
        lo = np.zeros(len(dirs))
        hi = np.full(len(dirs), tmax)
        c = self.centre
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            inside = self.phi(c + dirs * mid[:, None]) < 0.0
            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)
        return c + dirs * (0.5 * (lo + hi))[:, None]


def directions(theta, phi):
    """Unit direction(s) from the centre: theta around the vertical axis (0 = forward, -y; +x is positive theta),
    phi elevation (radians, +pi/2 = straight up)."""
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)
    cp = np.cos(phi)
    return np.stack([np.sin(theta) * cp, -np.cos(theta) * cp, np.sin(phi) + 0.0 * theta], -1)
