"""Eyes: the lid opening (almond), the eyeball placement, and lid motion.

Head-local metres (see head_shape), Rin-sized head. Eye shape coordinates (u, v) are relative to the iris centre:
u = outward (towards the temple, away from the nose), v = up. The left eye (character's left, +x) is built; the right
eye is its x-mirror.

Lid motion is a rotation about the x axis through the eyeball centre E (the eye sphere): every vertex of the lid rings
rotates by `weight * angle` (angle per ring column), so upper and lower lids meet on a common curve of the shell
without gaps."""
import numpy as np


# -------------------------------------------------------------------------------------------------- the opening
# The opening is two cubic Beziers between the corners: the upper lid (arched) and the lower lid (a deep round bowl that
# follows the bottom of the iris). (u, v) relative to the iris centre.
CORNER_IN = (-0.0240, -0.0036)
CORNER_OUT = (0.0308, 0.0100)
UPPER = ((-0.0112, 0.0186), (0.0085, 0.0216))                    # inner / outer handle of the upper lid
LOWER = ((-0.0092, -0.0272), (0.0172, -0.0350))                  # inner / outer handle of the lower lid
OPEN_CENTRE = (0.0035, -0.0040)                  # a point inside the opening (polar centre for the dip field)


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def lid_curves(n=400):
    """(upper, lower) lid polylines (n, 2), both running from the inner to the outer corner."""
    return (bezier(CORNER_IN, UPPER[0], UPPER[1], CORNER_OUT, n), bezier(CORNER_IN, LOWER[0], LOWER[1], CORNER_OUT, n))


def opening_polygon(n=400):
    """Closed polygon (2n-2, 2): inner corner, the upper lid, the outer corner, back along the lower lid (clockwise)."""
    up, lo = lid_curves(n)
    return np.vstack([up, lo[-2:0:-1]])


class Almond:
    """Polar lookup of the opening outline about OPEN_CENTRE: radius(angle); `poly` and the corner indices."""

    def __init__(self, scale=1.0, n=400):
        self.poly = opening_polygon(n) * scale
        self.i_inner, self.i_outer = 0, n - 1
        self.c = np.array(OPEN_CENTRE) * scale
        q = self.poly - self.c
        a = np.arctan2(q[:, 1], q[:, 0])
        r = np.hypot(q[:, 0], q[:, 1])
        o = np.argsort(a)
        a, r = a[o], r[o]
        if np.any(np.diff(a) <= 0):
            raise ValueError("eye opening is not star-shaped about its centre")
        self.a = np.concatenate([a - 2 * np.pi, a, a + 2 * np.pi])
        self.r = np.tile(r, 3)

    def point(self, ang):
        r = np.interp(ang, self.a, self.r)
        return self.c + np.stack([r * np.cos(ang), r * np.sin(ang)], -1)

# ------------------------------------------------------------------------------------------------------ the eye ring
DEFAULTS = dict(
    iris=(0.0428, 0.0572),               # iris centre (x, z) of the left eye; the right eye is the mirror
    radius=0.040,                        # eyeball radius R_e at the iris (the eyeball is a barrel about the x axis through E)
    centre_depth=0.0030,                 # the eyeball front pole lies this far behind the face surface at the iris
    centre_y=None,                       # fix the eyeball centre's y (the eye bone's head): R_e is then fitted to it
    margin_gap=0.0022,                   # lid margin height above the eyeball surface (the lid thickness)
    rim_gap=0.0005,                      # inner lid rim above the eyeball surface
    block_margins=(0.0030, 0.0030, 0.0040, 0.0070),     # grid margin around the opening: left, right, bottom, top
    ring_t=(0.0, 0.36, 0.70, 1.0),       # position of rings 0..3 between the grid block and the opening
    dip_length=0.013,                    # the socket dip fades out this far outside the opening
)


def arc_targets(ring_xz, w, h, poly, left_idx, right_idx):
    """Map the rect ring onto the opening polygon by arc length. `ring_xz`: ring vertex positions (n, 2) in the front
    plane relative to the eye centre (x to the right), the ring running CCW from the bottom-left corner of a w x h
    block. `poly`: the opening polygon in the same plane with the index of its left and right corner. The lower half of
    the rect (left column middle -> bottom row -> right column middle) maps onto the lower arc between the corners, the
    upper half onto the upper arc. Returns ((n, 2) targets, bool lower-arc flags)."""
    n = len(ring_xz)
    seg = np.hypot(*(np.roll(ring_xz, -1, 0) - ring_xz).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    def s_at(idx):
        i0 = int(np.floor(idx)) % n
        f = idx - np.floor(idx)
        return s[i0] + f * seg[i0]
    sR = s_at(w + h / 2.0)
    sL = s_at(2 * w + 1.5 * h)
    lower = np.array([(s[k] <= sR) or (s[k] >= sL) for k in range(n)])
    t = np.empty(n)
    for k in range(n):
        sk = s[k]
        if sk <= sR:
            t[k] = (sk + (total - sL)) / (sR + total - sL)
        elif sk >= sL:
            t[k] = (sk - sL) / (sR + total - sL)
        else:
            t[k] = (sk - sR) / (sL - sR)
    t = np.clip(t, 0.0, 1.0)
    # the polygon CCW starting at the left corner: lower arc to the right corner, then the upper arc back
    m = len(poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
        left_idx, right_idx = m - 1 - left_idx, m - 1 - right_idx
    poly = poly[np.roll(np.arange(m), -left_idx)]
    io = (right_idx - left_idx) % m
    lower_arc = poly[: io + 1]
    upper_arc = np.vstack([poly[io:], poly[:1]])

    def along(arc, f):
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arc, axis=0).T))])
        q = f * d[-1]
        return np.stack([np.interp(q, d, arc[:, 0]), np.interp(q, d, arc[:, 1])], -1)
    out = np.empty((n, 2))
    out[lower] = along(lower_arc, t[lower])
    out[~lower] = along(upper_arc, t[~lower])
    return out, lower


class Dip:
    """Smooth socket depression around one eye opening: the face surface sinks (+y) by D(psi) at the lid margin and
    fades to nothing `length` outside it (psi = polar angle about the opening's centre)."""

    def __init__(self, centre, sign, alm, psi, depth, length):
        o = np.argsort(psi)
        self.psi = np.concatenate([psi[o] - 2 * np.pi, psi[o], psi[o] + 2 * np.pi])
        self.depth = np.tile(depth[o], 3)
        self.c, self.sign, self.alm, self.length = centre, sign, alm, length

    def __call__(self, x, z):
        u = self.sign * (np.asarray(x) - self.c[0]) - self.alm.c[0]
        v = np.asarray(z) - self.c[1] - self.alm.c[1]
        psi = np.arctan2(v, u)
        r = np.hypot(u, v)
        rho = np.maximum(r - np.interp(psi, self.alm.a, self.alm.r), 0.0) / self.length
        t = np.clip(1.0 - rho, 0.0, 1.0)
        return np.interp(psi, self.psi, self.depth) * t * t * (3.0 - 2.0 * t)


class Barrel:
    """The eyeball surface of one eye: a barrel (surface of revolution about the x axis through E) whose radius follows
    the face's recession sideways, so it sits a constant depth behind the skin across the opening. A point at (x, z) with
    a layer gap g is (x, E.y - sqrt((rho(x) + g)^2 - (z - E.z)^2), z): lids and eyeball layers move on it by rotating
    about the x axis through E, with no gaps by construction."""

    def __init__(self, shape, E, sign, radius, depth, span=0.075):
        from .head_skin import lift
        self.E, self.sign = np.asarray(E, float), sign
        self.radius0 = float(radius)
        self.x = self.E[0] + np.linspace(-span, span, 301)
        y_skin = lift(shape, self.x, np.full_like(self.x, self.E[2]))[:, 1]
        y0 = float(np.interp(self.E[0], self.x, y_skin))
        self.rho = radius - (y_skin - y0)             # recession of the face shrinks the radius
        self.rho = np.maximum(self.rho, 0.012)

    def radius(self, x):
        return np.interp(x, self.x, self.rho)

    def point(self, x, z, gap=0.0):
        """A point of the barrel (the lid shell) at (x, z), `gap` above it."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        r = self.radius(x) + gap
        dz = z - self.E[2]
        y = self.E[1] - np.sqrt(np.maximum(r * r - dz * dz, 1e-12))
        return np.stack([x, y, z], -1)

    def layer(self, x, z, gap=0.0):
        """An eyeball layer point: on the sphere, but never in front of the lid shell where the face recedes faster than
        the sphere (so the lids and sheets always cover the eye graphics)."""
        P = self.sphere(x, z, gap)
        Pb = self.point(x, z, 0.0003 - gap * 0.0 + gap)
        P[:, 1] = np.maximum(P[:, 1], Pb[:, 1])
        return P

    def sphere(self, x, z, gap=0.0):
        """A point of the eyeball sphere (centre E, radius `radius0` + gap) seen from the front at (x, z). The eyeball
        layers live on it, so rotating them about E moves them rigidly over the sclera."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        R = self.radius0 + gap
        dx, dz = x - self.E[0], z - self.E[2]
        y = self.E[1] - np.sqrt(np.maximum(R * R - dx * dx - dz * dz, 1e-12))
        return np.stack([x, y, z], -1)


def add_eye(sb, side, cfg=None):
    """Cut the eye hole of one side into the SkinBuilder: rings 0 (the grid block) .. 3 (the lid margin) and 4 (the inner
    lid rim). Returns a dict describing the eye: centre E, radius, ring vertex ids, upper-lid flags, corner columns, and
    `dip` (the socket field, to be applied to the other grid vertices with apply_dips)."""
    from .head_skin import lift

    c = dict(DEFAULTS)
    c.update(cfg or {})
    sg = 1.0 if side == "L" else -1.0
    cx, cz = sg * c["iris"][0], c["iris"][1]
    alm = Almond()
    xs = cx + sg * alm.poly[:, 0]
    zs = cz + alm.poly[:, 1]
    g = sb.g
    nf = int(np.sum((g.theta > 0) & (g.theta < np.radians(75.0))))        # face columns on one side (x grows with the index)
    cols = range(1, nf + 1) if side == "L" else range(-nf, 0)
    block = sb.find_block(xs.min(), xs.max(), zs.min(), zs.max(), c["block_margins"], cols)
    ring0 = sb.carve(*block)
    n = len(ring0)
    P0 = np.array([sb.V[i] for i in ring0])
    w, h = block[1] - block[0], block[3] - block[2]
    rect_xz = np.stack([P0[:, 0] - cx, P0[:, 2] - cz], -1)
    poly_real = alm.poly * np.array([sg, 1.0])           # the opening in the front plane (x to the right)
    left, right = (alm.i_inner, alm.i_outer) if side == "L" else (alm.i_outer, alm.i_inner)
    treal, lower_half = arc_targets(rect_xz, w, h, poly_real, left, right)
    tuv = treal * np.array([sg, 1.0])                    # (u outward, v up)
    tx, tz = cx + treal[:, 0], cz + treal[:, 1]
    ang = np.arctan2(tuv[:, 1] - alm.c[1], tuv[:, 0] - alm.c[0])
    # eyeball: a barrel about the x axis through E; its radius follows the skin's recession so the margin (on the barrel,
    # margin_gap in front of the eyeball) stays about the same depth behind the face surface along the whole opening
    y_iris = float(lift(sb.shape, np.array([cx]), np.array([cz]))[0, 1])
    if c["centre_y"] is not None:
        R_e = float(np.clip(c["centre_y"] - y_iris - c["centre_depth"], 0.034, 0.060))
        E = np.array([cx, c["centre_y"], cz])
    else:
        R_e = c["radius"]
        E = np.array([cx, y_iris + c["centre_depth"] + R_e, cz])
    bar = Barrel(sb.shape, E, sg, R_e, c["centre_depth"])
    ym = lift(sb.shape, tx, tz)[:, 1]
    Pm = bar.point(tx, tz, c["margin_gap"])
    ysh = Pm[:, 1]
    R_sh = R_e + c["margin_gap"]
    dip = Dip((cx, cz), sg, alm, ang, ysh - ym, c["dip_length"])
    rings = [ring0]
    pos = [P0]
    for k in (1, 2, 3):
        t = c["ring_t"][k]
        x = P0[:, 0] + t * (tx - P0[:, 0])
        z = P0[:, 2] + t * (tz - P0[:, 2])
        P = lift(sb.shape, x, z)
        P[:, 1] += dip(x, z)
        if k == 3:
            P[:, 1] = ysh
        rings.append(sb.add_verts(P))
        pos.append(P)
    # inner lid rim: straight down from the margin to just above the eyeball sphere (rotates with the margin)
    d = pos[3] - E
    rho_sh = np.hypot(d[:, 1], d[:, 2])
    rho_rim = np.sqrt(np.maximum((R_e + c["rim_gap"]) ** 2 - d[:, 0] ** 2, 1e-8))
    P4 = pos[3].copy()
    P4[:, 1] = E[1] + d[:, 1] * rho_rim / rho_sh
    P4[:, 2] = E[2] + d[:, 2] * rho_rim / rho_sh
    P4[:, 1] = np.maximum(P4[:, 1], pos[3][:, 1] + 0.0003)          # never in front of the margin
    rings.append(sb.add_verts(P4))
    pos.append(P4)
    for k in range(3):
        sb.strip(rings[k], rings[k + 1], sb.SKIN)
    sb.strip(rings[3], rings[4], sb.RIM)
    ic, oc = np.array(CORNER_IN), np.array(CORNER_OUT)
    inner = int(np.argmin(np.hypot(*(tuv - ic).T)))
    outer = int(np.argmin(np.hypot(*(tuv - oc).T)))
    upper = ~lower_half
    upper[[inner, outer]] = False
    return dict(side=side, E=E, radius=R_e, shell=R_sh, barrel=bar, rings=rings, upper=upper, inner=inner, outer=outer,
                block=block, centre=np.array([cx, cz]), sign=sg, n=n, dip=dip, alm=alm, tuv=tuv)


def apply_dips(sb, eyes):
    """Sink the grid vertices around the eyes into the sockets (ring vertices already include their dip)."""
    n0 = len(sb.g.V)
    for i in range(n0):
        if i in sb.dead:
            continue
        p = sb.V[i]
        if abs(p[0]) < 0.02 or abs(p[0]) > 0.11 or p[2] < 0.02 or p[2] > 0.10 or p[1] > -0.02:
            continue
        for e in eyes:
            dy = float(e["dip"](p[0], p[2]))
            if dy != 0.0 and e["sign"] * p[0] > 0:
                sb.V[i] = np.array([p[0], p[1] + dy, p[2]])


# ---------------------------------------------------------------------------------------------------- eyeball parts
IRIS = (0.0204, 0.0228)                  # iris semi-axes (x: width, z: height) on the eyeball
PUPIL = (0.0078, 0.0116)
HIGHLIGHTS = (((-0.0080, 0.0070), 0.0050), ((0.0080, -0.0112), 0.0021))      # ((dx, dz) from the iris centre, radius)
LAYER_GAP = 0.0003                       # spacing between the eyeball layers (sclera, iris, pupil, highlights)


def barrel_disc(bar, cx, cz, centre, semi, n_rings, n_spokes, gap, rot=0.0):
    """Elliptical disc on the eyeball: centre vertex, then rings of n_spokes vertices. Real offsets (x right, z up) from
    (cx, cz); faces CCW seen from the front; uv in 0..1 over the bounding box of the ellipse (v up)."""
    a, b = semi
    xs, zs, uv = [cx + centre[0]], [cz + centre[1]], [(0.5, 0.5)]
    for k in range(1, n_rings + 1):
        r = k / n_rings
        for m in range(n_spokes):
            t = 2 * np.pi * m / n_spokes + rot
            xs.append(cx + centre[0] + r * a * np.cos(t))
            zs.append(cz + centre[1] + r * b * np.sin(t))
            uv.append((0.5 + 0.5 * r * np.cos(t), 0.5 + 0.5 * r * np.sin(t)))
    P = bar.layer(np.array(xs), np.array(zs), gap)
    ring = lambda k: 1 + (k - 1) * n_spokes
    F = [(0, ring(1) + m, ring(1) + (m + 1) % n_spokes) for m in range(n_spokes)]
    for k in range(1, n_rings):
        for m in range(n_spokes):
            m1 = (m + 1) % n_spokes
            F.append((ring(k) + m, ring(k + 1) + m, ring(k + 1) + m1, ring(k) + m1))
    return P, F, np.array(uv)


SCLERA_SCALES = (0.38, 0.66, 0.90, 1.08, 1.30)


def _sclera_points(alm, sg, nA=56):
    """Eye coordinates (u outward, v up) of the sclera patch vertices: the centre, then rings of the opening polygon
    scaled about its centre. Returns (u, v, ring size, scales)."""
    phi = 2 * np.pi * np.arange(nA) / nA
    psi = phi if sg > 0 else np.pi - phi
    rim = alm.point(psi)
    uu, vv = [alm.c[0]], [alm.c[1]]
    for sc in SCLERA_SCALES:
        uu.extend(alm.c[0] + sc * (rim[:, 0] - alm.c[0]))
        vv.extend(alm.c[1] + sc * (rim[:, 1] - alm.c[1]))
    return np.array(uu), np.array(vv), nA, SCLERA_SCALES


def sclera_frame():
    """The sclera's uv frame in eye coordinates: bounds (umin, umax, vmin, vmax) of its uv square, the lid curves fu(u),
    fl(u) of the opening and its centre; the texture is painted from this, so shading can follow the lids."""
    alm = Almond()
    uu, vv, _, _ = _sclera_points(alm, 1.0)
    up, lo = lid_curves(800)
    fu = lambda u: np.interp(u, up[:, 0], up[:, 1])
    fl = lambda u: np.interp(u, lo[:, 0], lo[:, 1])
    return dict(umin=float(uu.min()), umax=float(uu.max()), vmin=float(vv.min()), vmax=float(vv.max()), fu=fu, fl=fl,
                u_in=float(CORNER_IN[0]), u_out=float(CORNER_OUT[0]))


def eyeball_parts(eye, n_spokes=48):
    """The eye layers of one side as dict name -> (verts, faces, uv): sclera (a patch of the barrel), iris, pupil, two
    highlights."""
    bar = eye["barrel"]
    cx, cz = eye["centre"]
    out = {}
    # sclera: rings of the opening polygon scaled from its centre (0 .. 1.3): it covers the opening and a margin that stays
    # behind the lid zone; it never reaches the cheek or the temple where the face recedes faster than the eyeball
    sg = eye["sign"]
    alm = eye["alm"]
    uu, vv, nA, scales = _sclera_points(alm, sg)
    x, z = cx + sg * uu, cz + vv
    P = bar.layer(x, z, 0.0)
    fr = sclera_frame()
    uv = np.stack([(uu - fr["umin"]) / (fr["umax"] - fr["umin"]), (vv - fr["vmin"]) / (fr["vmax"] - fr["vmin"])], -1)
    ring = lambda k: 1 + (k - 1) * nA
    F = [(0, ring(1) + m, ring(1) + (m + 1) % nA) for m in range(nA)]
    for k in range(1, len(scales)):
        for m in range(nA):
            m1 = (m + 1) % nA
            F.append((ring(k) + m, ring(k + 1) + m, ring(k + 1) + m1, ring(k) + m1))
    out["sclera"] = (P, F, uv)          # CCW in the front plane for both eyes (the mirrored angle is mirrored too)
    out["iris"] = barrel_disc(bar, cx, cz, (0.0, 0.0), IRIS, 5, n_spokes, LAYER_GAP)
    out["pupil"] = barrel_disc(bar, cx, cz, (0.0, 0.0), PUPIL, 2, 32, 2 * LAYER_GAP)
    for i, (c, r) in enumerate(HIGHLIGHTS):
        out[f"highlight{i + 1}"] = barrel_disc(bar, cx, cz, c, (r, r), 1, 24, 3 * LAYER_GAP)
    return out


# ------------------------------------------------------------------------------------------------------ lash lines
UPPER_LASH = dict(width=0.0126, profile=((0.0, 0.14), (0.15, 0.34), (0.35, 0.62), (0.55, 0.92), (0.78, 1.0), (1.0, 0.96)),
                  wing_tip=(0.0128, 0.0090), wing_ctrl_up=(0.0078, 0.0064), wing_ctrl_lo=(0.0058, -0.0004), wing_n=7)
LOWER_LASH = dict(gap=0.0010, width=0.0009, s0=0.70, s1=0.97, n=10)
CREASE = dict(gap=0.0022, width=0.0008, s0=0.22, s1=0.94, n=16)
LASH_HEIGHT = 0.0004


def _sorted_arc(eye, which):
    """Margin vertices of one lid ordered from the inner to the outer corner: (ring-3 indices into the ring, (u, v))."""
    tuv = eye["tuv"]
    idx = [eye["inner"]] + [i for i in np.nonzero(eye["upper"] if which == "upper" else ~eye["upper"])[0]
                            if i not in (eye["inner"], eye["outer"])] + [eye["outer"]]
    if which == "lower":
        idx = [i for i in idx if True]
    idx = np.array(idx)
    order = np.argsort(tuv[idx, 0])
    idx = idx[order]
    return idx, tuv[idx]


def _smooth_poly(P, passes=7):
    Q = np.array(P, float)
    for _ in range(passes):
        Q[1:-1] = 0.25 * Q[:-2] + 0.5 * Q[1:-1] + 0.25 * Q[2:]
    return Q


def _normals(P, up_sign, bias=0.0):
    """Unit offset directions of a polyline running inner -> outer, pointing away from the opening (up for the upper lid).
    They come from a smoothed copy of the polyline and lean `bias` toward straight up/down, so a thick band offset by
    them does not fold over where the lid curves sharply (the offset must stay below the radius of curvature)."""
    t = np.gradient(_smooth_poly(P), axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = up_sign * np.stack([-t[:, 1], t[:, 0]], -1)
    if bias:
        n = n + bias * np.array([0.0, up_sign])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def _bez2(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def upper_lash(eye, cfg=None):
    """The heavy upper lash line with its wing: cross-sections (lo, hi) in (u, v) running from the inner corner along
    the margin to the outer corner and out along the wing; the last cross-section is the single tip. Also the margin
    ring-3 indices the lower edge sits on."""
    c = dict(UPPER_LASH)
    c.update(cfg or {})
    idx, M = _sorted_arc(eye, "upper")
    n = _normals(M, 1.0, bias=0.8)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))])
    s = d / d[-1]
    prof = np.interp(s, [p for p, _ in c["profile"]], [v for _, v in c["profile"]])
    hi = M + n * (c["width"] * prof)[:, None]
    lo = M.copy()
    # the wing: from the last cross-section (corner, band top) out to the tip
    O, U = lo[-1], hi[-1]
    T = O + np.array(c["wing_tip"])
    k = c["wing_n"]
    lo_w = _bez2(O, O + np.array(c["wing_ctrl_lo"]), T, k)[1:]
    hi_w = _bez2(U, U + np.array(c["wing_ctrl_up"]), T, k)[1:]
    nn = np.vstack([n, np.zeros((len(lo_w), 2))])
    lo = np.vstack([lo, lo_w])
    hi = np.vstack([hi, hi_w])
    return dict(lo=lo, hi=hi, tip=True, margin=idx, normal=nn)


def _tapered(M, off, width, s0, s1, n, up_sign, taper=0.7, bias=0.0):
    """A thin lens-shaped strip along a margin polyline: M (k, 2), offset `off` from it (away from the opening) with the
    width profile sin(pi t)^taper between the arc fractions s0..s1; pointed at both ends."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))])
    s = d / d[-1]
    q = np.linspace(s0, s1, n)
    P = np.stack([np.interp(q, s, M[:, 0]), np.interp(q, s, M[:, 1])], -1)
    N = _normals(M, up_sign, bias=bias)
    Nq = np.stack([np.interp(q, s, N[:, 0]), np.interp(q, s, N[:, 1])], -1)
    Nq /= np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-12)
    t = np.linspace(0.0, 1.0, n)
    w = width * np.sin(np.pi * t) ** taper
    offv = off(q) if callable(off) else off
    lo = P + Nq * np.broadcast_to(offv, (n,))[:, None]
    hi = lo + Nq * w[:, None]
    return lo, hi


def lower_lash(eye, cfg=None):
    c = dict(LOWER_LASH)
    c.update(cfg or {})
    idx, M = _sorted_arc(eye, "lower")
    lo, hi = _tapered(M, c["gap"], c["width"], c["s0"], c["s1"], c["n"], -1.0)
    return dict(lo=lo, hi=hi, tip=False, ends_pointed=True)


def crease(eye, cfg=None):
    """The double-eyelid line: a thin curve above the lash band, following it."""
    c = dict(CREASE)
    c.update(cfg or {})
    up = UPPER_LASH
    idx, M = _sorted_arc(eye, "upper")
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(M, axis=0), axis=1))])
    s = d / d[-1]
    wband = up["width"] * np.interp(s, [p for p, _ in up["profile"]], [v for _, v in up["profile"]])
    off = lambda q: np.interp(q, s, wband) + c["gap"]
    lo, hi = _tapered(M, off, c["width"], c["s0"], c["s1"], c["n"], 1.0, bias=0.8)
    return dict(lo=lo, hi=hi, tip=False, ends_pointed=True)
