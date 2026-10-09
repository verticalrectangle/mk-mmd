"""The head's base surface: an implicit shape sampled along rays from the skull centre.

Everything here is in HEAD-LOCAL coordinates: origin at the head bone (the neck/skull joint), +x = the character's left,
-y = forward, +z = up, metres (Rin-sized: the skull top 0.195 m above the bone). `head.py` places the result in model space.

The shape is a superellipse loft: at every height the horizontal section has the half width, front plane and back of
`Profiles`, which are smooth interpolants (monotone cubics, elliptical caps) of a few named proportions (`ANCHORS`: the
skull top, widths at the temples / eye level / cheeks / mouth / chin, the chin point, forehead, face-plane depths, the
back of the skull). The plain skull has soft blobs for a muzzle and a hint of nose; a generated face (`FACE`, `Face`) has
planes instead (each half of the face a plane receding from a midline ridge), with a nose, a bridge and lips on the ridge
and a jaw; a neck column blends in under the chin. The surface is star-shaped about `centre`, so a grid of directions
(theta around the vertical axis, phi elevation) gives one vertex per direction. Signed values are pseudo-distances
(metres, negative inside)."""
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
    """Half width hw(z), front plane yf(z) and back yb(z) of the head as functions of the height z (tables at 0.05 mm).
    The anchor `loft_bottom` (metres, default 0) lets the loft go on that far below the chin point, its narrowest end and the
    chin's depth with it: a generated face cuts the chin with its jaw, and the loft must still be full where the cut is."""

    def __init__(self, a=None):
        A = dict(ANCHORS)
        A.update(a or {})
        self.a = A
        cy, cz = A["crown"]
        zp = float(A["chin_point"][1]) - float(A.get("loft_bottom", 0.0))       # the loft's lowest point
        yp = float(A["chin_point"][0])
        zs = np.arange(zp - 1e-3, cz + 1e-3, 5e-5)
        self.z = zs
        # half width: elliptical cap above the temples, monotone cubic between the named widths, a V chin below
        zt, wt = A["temple"]
        names = ["chin", "mouth", "cheek", "eye", "temple"]
        pts_w = sorted([(float(A[n][0]), float(A[n][1])) for n in names] + [(float(z), float(w)) for z, w in A.get("hw_extra", ())])
        zk = np.array([p[0] for p in pts_w])
        wk = np.array([p[1] for p in pts_w])
        mid = pchip(zk, wk, d_end=(None, 0.0))
        zc, wc = A["chin"]
        hw = np.zeros_like(zs)
        up = zs >= zt
        hw[up] = wt * np.sqrt(np.clip(1.0 - ((zs[up] - zt) / (cz - zt)) ** 2, 0.0, 1.0))
        md = (zs < zt) & (zs >= zc)
        hw[md] = mid(zs[md])
        lo = zs < zc
        hw[lo] = wc * np.clip((zs[lo] - zp) / (zc - zp), 0.0, 1.0) ** 0.6
        self.hw = hw
        # front face plane: monotone cubic through the named depths, an elliptical cap above the forehead
        yf_pts = sorted([(zp, yp), (A["chin_plane"][1], A["chin_plane"][0]),
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
        pts = [(bz, by)] + [(z, y) for y, z in A["nape"]] + [(zp, yp)]
        pts = sorted(set(pts))
        zk2, yk2 = zip(*pts)
        mid_b = pchip(zk2, yk2, d_end=(None, 0.0))
        yb[~up] = mid_b(zs[~up])
        self.yb = yb
        self.ztop, self.zbot = float(cz), zp

    def at(self, z):
        z = np.asarray(z, float)
        return (np.interp(z, self.z, self.hw), np.interp(z, self.z, self.yf), np.interp(z, self.z, self.yb))


# --------------------------------------------------------------------------------------------------------- the face
# The generated face (head source "param") reshapes the front of the skull loft with a few named features. Without them
# (face=None) the shape is the plain skull, the prior a capped imported face relaxes to (head_cap). Head-local metres,
# mirror-symmetric about x = 0; every key can be overridden from the spec's [head.face] table (tables merge key by key).
FACE = dict(
    # midline depths of the face's planes (Profiles anchors, see ANCHORS; `front_extra` more (z, y) knots): the ridge the
    # two planes meet at, before the features. Forward at the muzzle and under the nose, back where the eyes sit, forward
    # again at the brow and the forehead; the loft goes on below the chin point (`loft_bottom`) so the jaw, not the loft's
    # narrow end, shapes the chin's underside
    profile=dict(brow_y=-0.1025, eye_plane_y=-0.0947, philtrum=(-0.1062, 0.0172), mouth_plane=(-0.0953, 0.0010),
                 chin_plane=(-0.0918, -0.0115), forehead=(-0.1105, 0.1165), loft_bottom=0.004,
                 front_extra=((0.0100, -0.1025), (0.0250, -0.1062), (0.0300, -0.1031), (0.0350, -0.0973),
                              (0.0450, -0.0949), (0.0700, -0.0966), (0.0850, -0.0990))),
    # exponent of the box the planes cut (by height, (z, n) knots; 2 is an ellipse): square, so the sides stay full out to
    # where the planes turn back into them (the cheekbones); squarest at the eyes, so the temples stay forward past the
    # eyes' outer corners and the face turns onto the side of the head beyond them (the eyes' outer ends sit on the face)
    front_n=((-0.030, 3.0), (0.000, 3.4), (0.030, 3.8), (0.045, 6.0), (0.090, 6.0), (0.120, 3.2), (0.160, 2.6)),
    # the face's planes: each half of the face is a plane receding from the midline ridge to the depth `side` (y) at
    # x = x_ref, the two meeting at the ridge rounded over `round` either side of it; `knots` (z, side, round) by height,
    # faded flat over `fade` above the top knot. A plane lights evenly in a toon render, so the cheeks, the temples and
    # the sides of the muzzle read as clean drawn areas; the ridge carries the nose; the eyes sit on the planes, their
    # outer corners further back than the inner ones. The side line comes forward over the cheeks, goes back where the
    # eyes sit and forward again at the brow, smooth in z whatever the ridge does (the slope is derived from both).
    # `corner`: the rounding where the planes turn back into the box's sides
    planes=dict(x_ref=0.045, fade=0.040, corner=0.006,
                knots=((-0.015, -0.0700, 0.007), (-0.008, -0.0770, 0.007), (0.000, -0.0832, 0.006), (0.006, -0.0870, 0.006),
                       (0.012, -0.0893, 0.006), (0.018, -0.0898, 0.006), (0.025, -0.0874, 0.006), (0.032, -0.0830, 0.006),
                       (0.040, -0.0811, 0.006), (0.056, -0.0804, 0.006), (0.070, -0.0831, 0.007), (0.085, -0.0865, 0.008),
                       (0.100, -0.0905, 0.009), (0.112, -0.0940, 0.010), (0.122, -0.0960, 0.012))),
    # the nose: a small ridge on the planes' ridge, which already carries most of its height. `tip` (y, z) is its point;
    # it rises out of the bridge below root_z and its underside slopes back into the upper lip at base_z. Across it falls
    # off as a gaussian of half `width` at the root, the tip and the base. `round` softens the tip, `bridge` > 1 makes the
    # bridge line concave (a ski slope), `under` < 1 fills the underside out
    nose=dict(tip=(-0.1120, 0.0285), root_z=0.0460, base_z=0.0120, width=(0.0080, 0.0090, 0.0110), round=0.45,
              bridge=1.4, under=0.90),
    # the lips: bumps along the midline, `bumps` = (height over the mouth line `z` (default: the mouth_plane anchor), half
    # height, forward amount), tapering sideways as exp(-(x / width)^2.5). From under the nose to the lip's edge the profile
    # is one straight slope (a little in front of the line from the nose's tip to the chin) that ends at the mouth: no
    # pout; the upper lip, the space under the nose filled up to that slope. Under the mouth line (folded in, head_mouth)
    # the lower lip comes out again, with a little fullness under it before the chin
    lips=dict(z=None, width=0.015, bumps=((0.0041, 0.0032, 0.0035), (0.0097, 0.0029, 0.0020), (-0.0039, 0.0027, 0.0028),
                                          (-0.0072, 0.0022, 0.0013))),
    # the bridge: the planes' ridge stands `height` proud of them, `width` half wide, from bottom_z (where it runs into the
    # nose's tip) up the brow, fading out over `fade` below top_z as the forehead comes forward (no dip between them)
    bridge=dict(bottom_z=0.024, top_z=0.130, fade=0.035, width=0.024, height=0.0060),
    # the jaw: its underside runs from the chin's lowest point (`chin` (y, z); default: the chin_point anchor) back to the
    # jaw angle at angle_y (below the ear), rising rise_deg on the midline: a low, almost flat underside under the chin, so
    # the neck meets it low; across, it rises `lateral` x^2 (metres per metre^2) towards the sides, so seen from the side the
    # jaw line climbs from the chin towards the ear (a keel, not a box). Behind the angle the underside rises `behind`
    # (dy, dz) more, then falls away at fall_deg until it passes under the back of the skull, whose own underside curves
    # into the nape (steeper than about 50 deg it would face the head centre the skin rays start from); the side rise fades
    # out behind the angle too. `round` is the width of the rounded edge where the face meets the underside (the underside
    # passes round/4 below `chin`, so the rounded edge still reaches the chin point)
    jaw=dict(chin=None, rise_deg=9.0, lateral=3.0, angle_y=-0.0040, behind=(0.0120, 0.0040), fall_deg=35.0, round=0.007),
    neck_k=0.008,          # the throat: the neck column meets the jaw in a fillet this wide (the plain skull: neck.k)
)


def face_config(over=None):
    """FACE with `over` merged into it (nested tables key by key, everything else replaced)."""
    def merge(a, b):
        out = dict(a)
        for k, v in (b or {}).items():
            out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return out
    return merge(FACE, over)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Face:
    """The features of a face configuration (see FACE) over the skull profiles `prof`."""

    def __init__(self, cfg, prof):
        self.c = cfg
        kn = sorted((float(z), float(n)) for z, n in cfg["front_n"])
        self.n_z, self.n_v = np.array([k[0] for k in kn]), np.array([k[1] for k in kn])
        # the planes: the slope per height that takes the ridge (the front plane) to the side depth at x_ref
        pl = cfg["planes"]
        kp = sorted((float(z), float(s), float(r)) for z, s, r in pl["knots"])
        kz, ks, kr = (np.array(v) for v in zip(*kp))
        self.pl_z = prof.z
        self.pl_r = pchip(kz, kr)(self.pl_z)
        xr = float(pl["x_ref"])
        fall = np.maximum(pchip(kz, ks)(self.pl_z) - prof.yf, 0.0)
        fade = 1.0 - _smoothstep((self.pl_z - kz[-1]) / float(pl["fade"]))
        self.pl_s = fall / (np.sqrt(xr * xr + self.pl_r ** 2) - self.pl_r) * fade
        self.corner = float(pl["corner"])
        jaw = cfg["jaw"]
        yc, zc = (float(v) for v in (jaw["chin"] if jaw.get("chin") is not None else prof.a["chin_point"]))
        zc -= 0.25 * float(jaw["round"])
        t = np.tan(np.radians(float(jaw["rise_deg"])))
        yg = float(jaw["angle_y"])
        zg = zc + t * (yg - yc)
        by, bz = (float(v) for v in jaw["behind"])
        tf = np.tan(np.radians(float(jaw["fall_deg"])))
        self.floor_y = pchip([yc - 0.04, yc, yg, yg + by, yg + by + 0.08],
                             [zc - 0.04 * t, zc, zg, zg + bz, zg + bz - 0.08 * tf], d_end=(t, -tf))
        self.lateral, self.lat_end = float(jaw["lateral"]), yg + by
        self.round = float(jaw["round"])
        lips = cfg["lips"]
        self.mouth_z = float(lips["z"]) if lips.get("z") is not None else float(prof.a["mouth_plane"][1])
        # the nose profile m(z) peaks at the tip; scale it so the point lands on `tip` (over the face and its other features)
        nz = cfg["nose"]
        zs = np.linspace(nz["base_z"], nz["root_z"], 801)
        m = self._nose_m(zs)
        yt, zt = float(nz["tip"][0]), float(nz["tip"][1])
        under = float(prof.at(zt)[1]) - float(self._others(np.array([0.0, yt, zt])))
        self.nose_scale = (under - yt) / max(float(m.max()), 1e-6)

    def front_n(self, z):
        return np.interp(z, self.n_z, self.n_v)

    def planes(self, x, z):
        """(how far behind the midline ridge the face's planes lie at (x, z), their slope): slope * |x|, rounded over
        `round` at the ridge."""
        s = np.interp(z, self.pl_z, self.pl_s)
        r = np.interp(z, self.pl_z, self.pl_r)
        return s * (np.sqrt(x * x + r * r) - r), s

    def _nose_m(self, z):
        c = self.c["nose"]
        zt, zr, zb = float(c["tip"][1]), float(c["root_z"]), float(c["base_z"])
        ub = np.clip((zr - z) / (zr - zt), 0.0, None)              # 0 at the root .. 1 at the tip
        uu = np.clip((z - zb) / (zt - zb), 0.0, None)              # 0 at the base .. 1 at the tip
        return np.clip(smin(ub ** c["bridge"], uu ** c["under"], c["round"]), 0.0, None)

    def nose(self, P):
        """Forward displacement of the face by the nose at the points P (metres)."""
        x, z = P[..., 0], P[..., 2]
        c = self.c["nose"]
        zt, zr, zb = float(c["tip"][1]), float(c["root_z"]), float(c["base_z"])
        wr, wt, wb = (float(v) for v in c["width"])
        w = np.where(z >= zt, wt + (wr - wt) * np.clip((z - zt) / (zr - zt), 0.0, 1.0),
                     wt + (wb - wt) * np.clip((zt - z) / (zt - zb), 0.0, 1.0))
        g = np.exp(-(x / w) ** 2)
        return self.nose_scale * self._nose_m(z) * g

    def _others(self, P):
        """The features besides the nose (forward displacement, metres): the lips and the bridge."""
        x, z = P[..., 0], P[..., 2]
        lp = self.c["lips"]
        gl = np.exp(-(np.abs(x) / float(lp["width"])) ** 2.5)
        d = np.zeros_like(x)
        for dz, s, a in lp["bumps"]:
            d = d + float(a) * np.exp(-((z - self.mouth_z - float(dz)) / float(s)) ** 2) * gl
        b = self.c["bridge"]
        win = _smoothstep((z - float(b["bottom_z"])) / 0.010) * _smoothstep((float(b["top_z"]) - z) / float(b["fade"]))
        return d + float(b["height"]) * np.clip(1.0 - (x / float(b["width"])) ** 2, 0.0, None) ** 2 * win

    def relief(self, P):
        """Forward displacement (metres along -y) of the skull's front surface at the points P: the nose and the others."""
        return (self.nose(P) + self._others(P)) * _smoothstep((-0.02 - P[..., 1]) / 0.03)    # only the front of the head

    def floor_z(self, P):
        """Height of the jaw's underside under the points P (its midline line plus the rise towards the sides)."""
        x, y = P[..., 0], P[..., 1]
        return self.floor_y(y) + self.lateral * x * x * (1.0 - _smoothstep((y - self.lat_end) / 0.03))

    def floor(self, P):
        """Pseudo-distance to the jaw's underside (negative above it, inside the head)."""
        h = 1e-4
        dy = (self.floor_y(P[..., 1] + h) - self.floor_y(P[..., 1] - h)) / (2 * h)
        dx = 2.0 * self.lateral * P[..., 0] * (1.0 - _smoothstep((P[..., 1] - self.lat_end) / 0.03))
        return (self.floor_z(P) - P[..., 2]) / np.sqrt(1.0 + dx * dx + dy * dy)


# --------------------------------------------------------------------------------------------------- the head shape
CENTRE = np.array([0.0, 0.0, 0.055])             # skull centre of the sampling rays (on the neck axis, eye height)


class HeadShape:
    """Implicit head. `phi(P)` is negative inside; `surface(dirs)` returns the surface point along each direction
    from `centre`; `normal(P)` the unit gradient. `anchors` overrides ANCHORS (see above); `face` (a dict merged into
    FACE, or None for the plain skull) adds the generated face's features."""

    def __init__(self, anchors=None, face=None, **kw):
        self.p = dict(
            n_front=2.35, n_back=2.1,                     # section exponents (2 = ellipse); the face side is squarer
            muzzle=dict(c=(0.0, -0.088, 0.008), r=(0.017, 0.012, 0.013), k=0.012),
            cheek=dict(c=(0.050, -0.078, 0.014), r=(0.024, 0.019, 0.016), k=0.016, on=False),     # optional cheek apples (they made lumps)
            nose=dict(c=(0.0, -0.1000, 0.0290), r=(0.0030, 0.0027, 0.0040), k=0.005),
            neck=dict(c=(0.0, 0.0, 0.0), rx=0.0265, ry=0.0290, k=0.014, top=0.02),
        )
        self.p.update(kw)
        self.centre = np.array([0.0, float(self.p["neck"]["c"][1]), CENTRE[2]])
        cfg = None if face is None else face_config(face)
        A = dict(anchors or {})
        if cfg is not None:
            A = {**cfg["profile"], **A}
        self.prof = Profiles(A)
        self.face = None if cfg is None else Face(cfg, self.prof)
        self.ztop, self.zbot = self.prof.ztop, self.prof.zbot

    def hull(self, P):
        """Superellipse loft: at each height the section has the half width, front plane and back of the profiles
        (pseudo-distance, negative inside). A generated face cuts the front of a squarer box (standing `corner` ahead of
        the front plane, so the cut decides the midline) with its planes (Face.planes), rounded `corner` into the sides."""
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        hw, yf, yb = self.prof.at(z)
        hw = np.maximum(hw, 1e-4)
        y_box = yf if self.face is None else yf - self.face.corner        # the box's front
        yc = 0.5 * (y_box + yb)
        fr = y < yc
        hd = np.maximum(np.where(fr, yc - y_box, yb - yc), 1e-4)
        n_front = self.p["n_front"] if self.face is None else self.face.front_n(z)
        n = np.where(fr, n_front, self.p["n_back"])
        u = np.abs(x) / hw
        v = np.abs(y - yc) / hd
        S = (u ** n + v ** n) ** (1.0 / n)
        d = (S - 1.0) * np.minimum(hw, hd)
        if self.face is not None:
            off, s = self.face.planes(x, z)
            dp = (yf + off - y) / np.sqrt(1.0 + s * s)
            d = np.where(fr, smax(d, dp, self.face.corner), d)
        return np.maximum(d, np.maximum(z - self.ztop, self.zbot - z))

    def phi(self, P, neck=True):
        P = np.asarray(P, float)
        nk = self.p["neck"]
        if self.face is None:
            d = self.hull(P)
            b = self.p["muzzle"]
            d = smin(d, ellipsoid(P, b["c"], b["r"]), b["k"])
            b = self.p["cheek"]
            if b.get("on"):
                for sx in (1.0, -1.0):
                    d = smin(d, ellipsoid(P, (sx * b["c"][0], b["c"][1], b["c"][2]), b["r"]), b["k"])
            n = self.p["nose"]
            d = smin(d, ellipsoid(P, n["c"], n["r"]), n["k"])
            k_neck, top = nk["k"], nk["top"]
        else:
            F = self.face
            Q = np.array(P, float)
            Q[..., 1] += F.relief(P)
            d = smax(self.hull(Q), F.floor(P), float(F.c["jaw"]["round"]))
            k_neck, top = float(F.c["neck_k"]), 0.06                  # the column reaches up into the raised jaw
        if neck:
            r = np.hypot((P[..., 0] - nk["c"][0]) / nk["rx"], (P[..., 1] - nk["c"][1]) / nk["ry"])
            dn = np.maximum((r - 1.0) * min(nk["rx"], nk["ry"]), P[..., 2] - top)    # column stops inside the skull
            d = smin(d, dn, k_neck)
        return d

    def jaw_side(self, P):
        """Which side of the jaw's edge the points P of a generated face lie on (metres): the face's distance inside the
        underside minus its distance inside the face's front and sides. Negative on the underside, positive on the face,
        zero along the middle of the rounded edge where the face turns under (where a drawn jaw shadow begins)."""
        F = self.face
        Q = np.array(P, float)
        Q[..., 1] += F.relief(P)
        return self.hull(Q) - F.floor(P)

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
