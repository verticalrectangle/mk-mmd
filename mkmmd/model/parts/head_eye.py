"""Eyes: the lid opening, the lid shell, the eye behind the opening, and the lash lines.

Head-local metres (see head_shape), Rin-sized head. Eye shape coordinates (u, v) are relative to the iris centre:
u = outward (towards the temple, away from the nose), v = up. The left eye (character's left, +x) is built; the right
eye is its x-mirror.

The lid margins and the lid sheets ride on the lid SHELL (LidShell: just behind the face along the upper lid and at the
corners, curving back under the eye). Behind the opening the eye is an anime eye, not an eyeball: the white is a POCKET,
a deep rounded bowl that starts at the inner lid rim and opens out behind the skin (so the iris can move inside it and
slide behind the lids), and the iris is a FLAT disc inside the pocket on the IRIS PLANE, a few millimetres behind the
opening and turned outward like the face: from the front it reads as a drawn circle, from the side it is seen edge on, a
sliver deep in the eye. The highlights float in front of the iris (they shift against it as the head turns, a glassy
depth) on their own bones that follow the gaze only partly (a reflection does not move with the eye)."""
import numpy as np


# -------------------------------------------------------------------------------------------------- the opening
# The opening is two cubic Beziers between the corners: the upper lid a high, flat-topped arch over the iris that comes
# down an almost vertical outer side into the outer corner, the lower lid a round bowl under the iris that rises into it.
# The outer corner sits at mid-height: seen from the side the lower lid runs straight back and the eye's outer end tucks
# into the head instead of pointing out of it. Each lid moves steadily outward (u never decreases along it), so the lid
# sheets and the closed-eye drawings can be functions of u. (u, v) relative to the iris centre.
CORNER_IN = (-0.0188, -0.0035)
CORNER_OUT = (0.0320, 0.0030)
UPPER = ((-0.0181, 0.0259), (0.0320, 0.0269))                    # inner / outer handle of the upper lid
LOWER = ((-0.0183, -0.0301), (0.0226, -0.0302))                  # inner / outer handle of the lower lid
OPEN_CENTRE = (0.0045, -0.0010)                  # a point inside the opening (polar centre for the dip field)


def lid_s(u):
    """The lid parameter of points at u: 0 at the inner corner, 1 at the outer corner. The lash strips, the lid sheets and
    the closed-eye drawings all place things by it."""
    return (np.asarray(u, float) - CORNER_IN[0]) / (CORNER_OUT[0] - CORNER_IN[0])


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
SHEET_GAP = 0.0020                      # the lid sheets (closed lids) ride this far in front of the lid shell (head_lid)


DEFAULTS = dict(
    iris=(0.0428, 0.0572),               # iris centre (x, z) of the left eye; the right eye is the mirror
    radius=0.040,                        # the eye bone E lies this far behind the lid shell at the iris (the gaze's pivot)
    centre_depth=0.0026,                 # the lid shell lies this far behind the face above the corners' chord (LidShell)
    lower_depth=0.0055,                  # ... and this far at the lower lid: the lower lid tucks in behind the cheek
    tuck=(0.008, 0.012),                 # the tuck starts this far below the chord and is complete this much further down
    centre_y=None,                       # fix the eye bone's y (E): R_e is then fitted to it
    margin_gap=0.0022,                   # lid margin height above the lid shell (the lid thickness)
    rim_gap=0.0005,                      # inner lid rim above the lid shell (the pocket starts there)
    block_margins=(0.0030, 0.0030, 0.0040, 0.0070),     # grid margin around the opening: left, right, bottom, top
    ring_t=(0.0, 0.36, 0.70, 1.0),       # position of rings 0..3 between the grid block and the opening
    dip_length=0.013,                    # the socket dip fades out this far outside the opening
    # the iris plane (iris_plane): iris_depth behind the lid margin at the iris centre (about as far behind the face as
    # drawn eyes sit), facing forward turned iris_yaw_deg outward and iris_pitch_deg up; pushed further back only as far
    # as the skin round the opening (every gaze in `gaze`) and the closed lids (every gaze in `blink_gaze`) need to cover
    # the iris, the pupil and the highlights by iris_clear. Gazes are (outward yaw, up pitch) degrees of the eye bone; the
    # highlights turn highlight_follow of it
    iris_depth=0.0038, iris_yaw_deg=18.0, iris_pitch_deg=0.0, iris_clear=0.0006,
    gaze=((15.0, 0.0), (-25.0, 0.0), (0.0, 15.0), (0.0, -15.0), (15.0, 15.0), (-25.0, -15.0), (15.0, -15.0), (-25.0, 15.0)),
    blink_gaze=((7.0, 0.0), (-7.0, 0.0), (0.0, 6.0), (0.0, -6.0)),
    highlight_gap=0.0028,                # the highlights float this far in front of the iris
    highlight_follow=0.6,                # share of the eyes' (両目) rotation the highlight bones take
    # the pocket (the white): from the inner lid rim back to `back` behind the iris plane, opening out by `undercut`
    # behind the skin on the way; every point stays `clear` inside the head. `t`: the rings between the rim (0) and the
    # bottom point (1, not a ring)
    pocket=dict(back=0.016, undercut=0.006, t=(0.12, 0.3, 0.5, 0.65, 0.8, 0.92), clear=0.0012),
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


class LidShell:
    """The lid shell of one eye: the surface the lid margins and the lid sheets ride on (`point(x, z, gap)`, `gap` towards
    the viewer). It follows the face itself, `upper` behind it over the upper part of the eye (at both corners and along
    the upper lid), so the upper lid's edge sits on the face as drawn eyes do (a shell curving away above the eye while the
    brow comes forward would sink the skin round it into a crater), and `lower` behind it at the lower lid, so the lower
    lid tucks in behind the cheek: the tuck starts tuck[0] below the chord between the corners and is complete tuck[1]
    further down (the middle of the eye stays shallow, so the closed lids there cover a shallow iris)."""

    def __init__(self, shape, E, sign, upper, lower, tuck, span=0.075):
        from .head_skin import lift
        self.E, self.sign = np.asarray(E, float), sign
        self.upper, self.lower, self.tuck = float(upper), float(lower), (float(tuck[0]), float(tuck[1]))
        # the face over the eye on a 1 mm grid
        self.gx = self.E[0] + np.linspace(-span, span, 151)
        self.gz = self.E[2] + np.linspace(-0.035, 0.035, 71)
        X, Z = np.meshgrid(self.gx, self.gz, indexing="ij")
        self.gy = lift(shape, X.ravel(), Z.ravel())[:, 1].reshape(X.shape)

    def face_y(self, x, z):
        """The face's depth at (x, z) (bilinear on the grid)."""
        fx = np.clip((x - self.gx[0]) / (self.gx[1] - self.gx[0]), 0.0, len(self.gx) - 1.000001)
        fz = np.clip((z - self.gz[0]) / (self.gz[1] - self.gz[0]), 0.0, len(self.gz) - 1.000001)
        i, j = fx.astype(int), fz.astype(int)
        a, b = fx - i, fz - j
        G = self.gy
        return (1 - a) * (1 - b) * G[i, j] + a * (1 - b) * G[i + 1, j] + (1 - a) * b * G[i, j + 1] + a * b * G[i + 1, j + 1]

    def depth(self, x, z):
        """How far behind the face the shell lies at (x, z)."""
        u, v = self.sign * (x - self.E[0]), z - self.E[2]
        chord = CORNER_IN[1] + lid_s(u) * (CORNER_OUT[1] - CORNER_IN[1])
        t = np.clip((chord - v - self.tuck[0]) / self.tuck[1], 0.0, 1.0)
        return self.upper + (self.lower - self.upper) * t * t * (3.0 - 2.0 * t)

    def point(self, x, z, gap=0.0):
        """The point of the shell at (x, z), `gap` towards the viewer."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        return np.stack([x, self.face_y(x, z) + self.depth(x, z) - gap, z], -1)


def add_eye(sb, side, cfg=None):
    """Cut the eye hole of one side into the SkinBuilder: rings 0 (the grid block) .. 3 (the lid margin) and 4 (the inner
    lid rim, where the pocket starts). Returns a dict describing the eye: the eye bone E, the shell radius, ring vertex ids,
    upper-lid flags, corner columns, `dip` (the socket field, to be applied to the other grid vertices with apply_dips),
    `layers` (the IrisPlane), `rim` (the inner rim's positions) and `cfg`."""
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
    # the lid shell (LidShell) follows the face, centre_depth behind it above the corners' chord and lower_depth under the
    # eye; the margin rides margin_gap in front of it. The eye bone E (the gaze's pivot) lies R_e behind the shell at the iris
    y_iris = float(lift(sb.shape, np.array([cx]), np.array([cz]))[0, 1])
    if c["centre_y"] is not None:
        R_e = float(np.clip(c["centre_y"] - y_iris - c["centre_depth"], 0.034, 0.060))
        E = np.array([cx, c["centre_y"], cz])
    else:
        R_e = c["radius"]
        E = np.array([cx, y_iris + c["centre_depth"] + R_e, cz])
    shell = LidShell(sb.shape, E, sg, c["centre_depth"], c["lower_depth"], c["tuck"])
    ym = lift(sb.shape, tx, tz)[:, 1]
    ysh = shell.point(tx, tz, c["margin_gap"])[:, 1]
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
    # inner lid rim: straight back from the margin (the lid's thickness) to just over the shell; the pocket starts there and
    # the rim rotates with the margin in a blink. Straight back: at the corners the shell meets the skin at a slant
    P4 = pos[3].copy()
    P4[:, 1] = np.clip(shell.point(pos[3][:, 0], pos[3][:, 2], c["rim_gap"])[:, 1],
                       pos[3][:, 1] + 0.0008, pos[3][:, 1] + 0.0030)         # 0.8 .. 3 mm behind the margin
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
    plane = iris_plane(shell, E, cx, cz, sg, c, lambda x, z: shell.face_y(x, z) + dip(x, z), alm)
    return dict(side=side, E=E, radius=R_e, lid_shell=shell, rings=rings, upper=upper, inner=inner, outer=outer,
                block=block, centre=np.array([cx, cz]), sign=sg, n=n, dip=dip, alm=alm, tuv=tuv, layers=plane, rim=P4, cfg=c)


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


# ---------------------------------------------------------------------------------------------------------- the eye
IRIS = (0.0204, 0.0228)                  # iris semi-axes (x: width, z: height) seen from the front
PUPIL = (0.0078, 0.0116)
HIGHLIGHTS = (((-0.0080, 0.0070), 0.0050), ((0.0080, -0.0112), 0.0021))      # ((dx, dz) from the iris centre, radius)
LAYER_GAP = 0.0003                       # the pupil over the iris


class IrisPlane:
    """The flat iris plane of one eye: through C, unit normal n towards the viewer. `layer(x, z, gap)` is its point seen
    from the front at (x, z), moved `gap` towards the viewer (so a disc keeps its drawn outline in the front view)."""

    def __init__(self, C, n):
        self.C = np.asarray(C, float)
        self.n = np.asarray(n, float) / np.linalg.norm(n)

    def layer(self, x, z, gap=0.0):
        x, z = np.asarray(x, float), np.asarray(z, float)
        C, n = self.C, self.n
        y = C[1] + (gap - (x - C[0]) * n[0] - (z - C[2]) * n[2]) / n[1]
        return np.stack([x, y, z], -1)

    def moved(self, depth):
        """The plane `depth` further back (away from the viewer)."""
        return IrisPlane(self.C - depth * self.n, self.n)


def gaze_rotation(E, yaw, pitch):
    """Rotation of points about the eye bone E: yaw about z (+ turns the front towards +x), then pitch about x."""
    ay, ap = np.radians(yaw), np.radians(pitch)
    Rz = np.array([[np.cos(ay), -np.sin(ay), 0.0], [np.sin(ay), np.cos(ay), 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ap), -np.sin(ap)], [0.0, np.sin(ap), np.cos(ap)]])
    R = Rz @ Rx
    return lambda P: (np.asarray(P, float) - E) @ R.T + E


def iris_plane(shell, E, cx, cz, sg, c, skin_y, alm):
    """The iris plane of one eye: facing forward turned iris_yaw_deg outward and iris_pitch_deg up, iris_depth behind the
    lid margin at the iris centre, then pushed back until every layer (iris, pupil, highlights; the highlights turn
    highlight_follow of the gaze) is covered by iris_clear: where it lies outside the opening, by the skin (`skin_y(x,
    z)`) for every gaze in `gaze`; inside the opening, by the closed lids (SHEET_GAP over the lid shell) for every gaze in
    `blink_gaze` (a blink while looking a little aside). Inside the open eye nothing covers the layers, so they sit just
    behind the opening as drawn eyes do: from the side the iris shows, not a deep white bowl."""
    yaw, pitch = np.radians(c["iris_yaw_deg"]), np.radians(c["iris_pitch_deg"])
    n = np.array([sg * np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    C0 = shell.point(np.array([cx]), np.array([cz]), c["margin_gap"])[0]
    plane = IrisPlane(C0, n).moved(c["iris_depth"])
    t = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    layers = [((0.0, 0.0), IRIS, 0.0, 1.0), ((0.0, 0.0), PUPIL, LAYER_GAP, 1.0)]
    layers += [(o, (r, r), c["highlight_gap"], c["highlight_follow"]) for o, r in HIGHLIGHTS]
    clear = c["iris_clear"]
    gazes = [((0.0, 0.0), True)] + [(g, True) for g in c["blink_gaze"]] + [(g, False) for g in c["gaze"]]
    for _ in range(8):
        need = 0.0
        for (dx, dz), (a, b), gap, follow in layers:
            for k in (1.0, 0.5):
                P = plane.layer(cx + dx + k * a * np.cos(t), cz + dz + k * b * np.sin(t), gap)
                for (yaw_out, pit), blink in gazes:
                    Q = gaze_rotation(E, sg * yaw_out * follow, pit * follow)(P)
                    u, v = sg * (Q[:, 0] - cx) - alm.c[0], Q[:, 2] - cz - alm.c[1]
                    out = np.hypot(u, v) > np.interp(np.arctan2(v, u), alm.a, alm.r)
                    if out.any():                                     # hidden by the skin round the opening
                        ys = skin_y(Q[out, 0], Q[out, 2])
                        need = max(need, float((ys + clear - Q[out, 1]).max()))
                    if blink and (~out).any():                        # hidden by the closed lids
                        ys = shell.point(Q[~out, 0], Q[~out, 2], SHEET_GAP)[:, 1]
                        need = max(need, float((ys + clear - Q[~out, 1]).max()))
        if need <= 1e-6:
            break
        plane = plane.moved(need * abs(plane.n[1]) + 2e-5)
    return plane


def layer_disc(plane, cx, cz, centre, semi, n_rings, n_spokes, gap, rot=0.0):
    """Elliptical disc on the iris plane: centre vertex, then rings of n_spokes vertices. Real offsets (x right, z up) from
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
    P = plane.layer(np.array(xs), np.array(zs), gap)
    ring = lambda k: 1 + (k - 1) * n_spokes
    F = [(0, ring(1) + m, ring(1) + (m + 1) % n_spokes) for m in range(n_spokes)]
    for k in range(1, n_rings):
        for m in range(n_spokes):
            m1 = (m + 1) % n_spokes
            F.append((ring(k) + m, ring(k + 1) + m, ring(k + 1) + m1, ring(k) + m1))
    return P, F, np.array(uv)


def pocket(eye, shape):
    """The white of one eye: a rounded bowl from the inner lid rim (`eye["rim"]`) back to `back` behind the iris plane.
    On the way down each ring opens out by up to `undercut` behind the lids (widest a little behind the iris, where it
    moves), then the bottom rounds off onto a centre point; every point stays `clear` behind the skin (the socket's dip
    included). Returns (verts, faces, (u, v) eye coordinates of the vertices); the faces face the viewer."""
    c = eye["cfg"]["pocket"]
    plane, sg, dip = eye["layers"], eye["sign"], eye["dip"]
    cx, cz = eye["centre"]
    n = plane.n
    e2 = np.array([0.0, 0.0, 1.0]) - n * n[2]                   # up, in the plane
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)                                        # sideways, in the plane
    R = np.asarray(eye["rim"], float)
    a, b, w = (R - plane.C) @ e1, (R - plane.C) @ e2, (R - plane.C) @ n
    oa, ob = float(a.mean()), float(b.mean())
    r = np.maximum(np.hypot(a - oa, b - ob), 1e-6)
    da, db = (a - oa) / r, (b - ob) / r
    w_back = -float(c["back"])
    w_mid = 0.5 * (w + w_back)
    under = float(c["undercut"])

    def inside(P):
        Q = np.array(P, float)
        Q[:, 1] -= dip(Q[:, 0], Q[:, 2])                  # the skin around the opening is sunk by the dip
        return shape.phi(Q, neck=False) <= -float(c["clear"])

    def ring_at(t, f):
        """Ring t (0 the rim .. 1 the bottom) with the opening-out factor f per vertex: down the side wall to w_mid while
        opening out, then a quarter ellipse round to the bottom."""
        if t <= 0.5:
            rad = r + f * under * np.sin(0.5 * np.pi * t / 0.5)
            ww = w + (w_mid - w) * (t / 0.5)
        else:
            ph = 0.5 * np.pi * (t - 0.5) / 0.5
            rad = (r + f * under) * np.cos(ph) ** 0.8
            ww = w_mid + (w_back - w_mid) * np.sin(ph)
        return plane.C + (oa + rad * da)[:, None] * e1 + (ob + rad * db)[:, None] * e2 + ww[:, None] * n

    rings = [R]
    for t in c["t"]:
        f = np.ones(len(R))
        P = ring_at(t, f)
        ok = inside(P)
        lo = np.zeros(len(R))                             # bisect the opening-out factor where the ring would leave the head
        hi = f.copy()
        for _ in range(12):
            if ok.all():
                break
            hi = np.where(ok, hi, 0.5 * (lo + hi))
            P = ring_at(t, hi)
            ok = inside(P)
        if not ok.all():                                  # still outside at the rim's own width: pull towards the axis
            for _ in range(10):
                bad = ~ok
                P[bad] = plane.C + oa * e1 + ob * e2 + 0.85 * (P[bad] - (plane.C + oa * e1 + ob * e2))
                ok = inside(P)
                if ok.all():
                    break
        rings.append(P)
    m = len(R)
    V = np.vstack(rings)
    centre = plane.C + oa * e1 + ob * e2 + w_back * n
    V = np.vstack([V, centre[None]])
    F = []
    for k in range(len(rings) - 1):
        for i in range(m):
            i1 = (i + 1) % m
            F.append((k * m + i, k * m + i1, (k + 1) * m + i1, (k + 1) * m + i))
    last, apex = (len(rings) - 1) * m, len(V) - 1
    for i in range(m):
        F.append((last + i, last + (i + 1) % m, apex))
    view = plane.C + oa * e1 + ob * e2 + 0.06 * n                # the faces face this point in front of the opening
    out = []
    for f in F:
        p = V[list(f)]
        nf = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(tuple(f) if nf @ (view - p.mean(0)) >= 0 else tuple(reversed(f)))
    uv = np.stack([sg * (V[:, 0] - cx), V[:, 2] - cz], -1)
    return V, out, uv


def sclera_frame(scale=1.3):
    """The white's uv frame in eye coordinates: bounds (umin, umax, vmin, vmax) of its uv square (the opening grown by
    `scale` about its centre), the lid curves fu(u), fl(u) of the opening and its corners; the texture is painted from
    this, so its shading follows the lids."""
    alm = Almond()
    q = alm.c + scale * (alm.poly - alm.c)
    up, lo = lid_curves(800)
    fu = lambda u: np.interp(u, up[:, 0], up[:, 1])
    fl = lambda u: np.interp(u, lo[:, 0], lo[:, 1])
    return dict(umin=float(q[:, 0].min()), umax=float(q[:, 0].max()), vmin=float(q[:, 1].min()), vmax=float(q[:, 1].max()),
                fu=fu, fl=fl, u_in=float(CORNER_IN[0]), u_out=float(CORNER_OUT[0]))


def eye_gaps(eye):
    """How far each eye layer floats in front of the iris plane."""
    return dict(iris=0.0, pupil=LAYER_GAP, highlight=float(eye["cfg"]["highlight_gap"]))


def eyeball_parts(eye, shape, n_spokes=48):
    """The eye of one side as dict name -> (verts, faces, uv, normals): the white (the pocket), iris, pupil, two
    highlights. The normals face along the iris plane's normal (flat, evenly lit layers)."""
    plane = eye["layers"]
    cx, cz = eye["centre"]
    gaps = eye_gaps(eye)
    fr = sclera_frame()
    P, F, euv = pocket(eye, shape)
    uv = np.stack([(euv[:, 0] - fr["umin"]) / (fr["umax"] - fr["umin"]), (euv[:, 1] - fr["vmin"]) / (fr["vmax"] - fr["vmin"])], -1)
    out = {"sclera": (P, F, np.clip(uv, 0.0, 1.0))}
    out["iris"] = layer_disc(plane, cx, cz, (0.0, 0.0), IRIS, 5, n_spokes, gaps["iris"])
    out["pupil"] = layer_disc(plane, cx, cz, (0.0, 0.0), PUPIL, 2, 32, gaps["pupil"])
    for i, (c, r) in enumerate(HIGHLIGHTS):
        out[f"highlight{i + 1}"] = layer_disc(plane, cx, cz, c, (r, r), 1, 24, gaps["highlight"])
    return {k: (p, f, u, np.broadcast_to(plane.n, (len(p), 3)).copy()) for k, (p, f, u) in out.items()}


# ------------------------------------------------------------------------------------------------------ lash lines
# Strips along the lid margins, placed by the lid parameter (lid_s). The upper lash band starts in a point `tail` from the inner
# corner and meets the upper lid where the lid rises through `peel_v`; it runs along the lid with the thickness `profile` (by
# lid parameter), its top edge rising from a `notch` over the meeting point to the full band over `ramp`; at `turn` its top
# edge leaves for the top-outer corner (`elbow` from the outer corner), arriving there heading `elbow_in` degrees, and the
# band's outer edge leaves it heading `elbow_out` back down the eye's outer side, the band lying on the lower lid there, to
# a point of `out_tip` degrees where the lid falls through `out_v`. Its lid parameters run on past 1 down the outer side
# (1 .. 1 + OUTER_RUN) and below 0 along the tail. Two blades lie over it: the fork, a second point at the inner end (its top
# edge leaves the band's top edge at lid parameter `base`, its point `length` from the notch's apex heading `angle`, `point`
# degrees sharp), and the wing on the top-outer corner (`root` and `base` on the band's top edge between the turn (0) and
# the elbow (1), its point `length` from their middle heading `angle`; its edges sag `sag` = (upper, lower) below their
# chords, so it curls up). The crease is the double-eyelid line: a stroke over the inner half, `gap` above the band's top
# edge from lid parameter s0 to s1 and on straight past s0 by `ext`, `width` at its widest, its ends tapering `slope` (width
# per length) to points. Angles are degrees counter-clockwise from the outward direction (+u).
UPPER_LASH = dict(width=0.0080, profile=((0.0, 0.45), (0.25, 0.76), (0.55, 0.95), (0.80, 1.0), (1.0, 1.0)),
                  tail=(-0.0062, 0.0002), peel_v=0.0060, notch=0.0026, ramp=0.0040, turn=0.86, elbow=(0.0072, 0.0098),
                  elbow_in=-25.0, elbow_out=-108.0, out_v=-0.0045, out_tip=18.0, step=0.0012)
LASH_FORK = dict(base=0.16, length=0.0068, angle=170.0, point=20.0, n=12)
LASH_WING = dict(root=0.40, base=0.85, length=0.0112, angle=52.0, sag=(0.0010, 0.0005), n=10)
CREASE = dict(s0=0.12, s1=0.58, ext=0.0060, gap=((0.0, 0.0042), (0.5, 0.0036), (1.0, 0.0032)), width=0.0021,
              slope=(0.20, 0.17), n=30)
OUTER_RUN = 0.30
LOWER_LASH = dict(gap=0.0010, width=0.0012, s0=0.12, s1=0.70, n=16, skew=0.6)
LASH_HEIGHT = 0.0004


def _sorted_arc(eye, which):
    """Margin vertices of one lid in ring order from the inner to the outer corner: (ring-3 indices into the ring, (u, v)).
    Along a lid u never decreases, but on the eye's outer side it barely changes, so ring order, not u, orders them."""
    tuv, n = eye["tuv"], len(eye["tuv"])
    i0, i1 = eye["inner"], eye["outer"]
    side = eye["upper"] if which == "upper" else ~eye["upper"]
    nxt = (i0 + 1) % n
    step = 1 if (side[nxt] and nxt != i1) else -1
    idx, k = [i0], i0
    for _ in range(n):
        k = (k + step) % n
        idx.append(k)
        if k == i1:
            break
    idx = np.array(idx)
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


def _at_s(M, s, A=None):
    """Values of A (rows along the margin polyline M, inner -> outer corner; default M itself) at the lid parameters s."""
    sm = np.maximum.accumulate(lid_s(M[:, 0]))
    A = M if A is None else A
    return np.stack([np.interp(s, sm, A[:, k]) for k in range(A.shape[1])], -1)


def _arc_len(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _along(P, d, A=None):
    """Values of A (rows along the polyline P; default P itself) at the arc lengths d along P."""
    a, A = _arc_len(P), (P if A is None else A)
    return np.stack([np.interp(d, a, A[:, k]) for k in range(A.shape[1])], -1)


def _resample(P, n):
    return _along(P, np.linspace(0.0, _arc_len(P)[-1], n))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _dir(deg):
    """Unit vector `deg` degrees counter-clockwise from +u."""
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def _turned(v, deg):
    """v turned `deg` degrees counter-clockwise."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _hermite(p0, d0, p1, d1, n):
    """A cubic from p0 to p1 that leaves along the unit direction d0 and arrives along d1, its handles a third of the chord."""
    h = np.linalg.norm(np.asarray(p1) - np.asarray(p0)) / 3.0
    return bezier(p0, p0 + h * d0, p1 - h * d1, p1, n)


def upper_lines(eye, cfg=None):
    """The lines above one eye, dict piece -> strip: lo, hi = the two ends of its cross-sections (n, 2) in (u, v); s = their
    lid parameters (where the closed-eye drawings put them); `tip` (the last cross-section is a single point) or
    `ends_pointed` (both ends are); `normal` = the direction that nudges the lo edge off the margin into the skin.
      lash      the band; `hold` = (lid parameter where it meets the lid, `turn`): a lowered lid moves what lies beyond
                them rigidly with the band there (head_lid); `rim` = the margin columns (ring 3) it lies on
      lashfork  the second point of the band's inner end
      lashwing  the wing on the band's top-outer corner
      crease    the double-eyelid stroke; `gap` = its height over the band's top edge per cross-section
    cfg: the [head] table; its tables lash, lashfork, lashwing, crease override UPPER_LASH, LASH_FORK, LASH_WING, CREASE."""
    cfg = cfg or {}
    c = dict(UPPER_LASH, **(cfg.get("lash") or {}))
    iu, M = _sorted_arc(eye, "upper")
    il, ML = _sorted_arc(eye, "lower")
    dM = _arc_len(M)
    sM = np.maximum.accumulate(lid_s(M[:, 0]))
    nM = _normals(M, 1.0, bias=0.8)
    kx, ky = np.array(c["profile"], float).T
    width = lambda d: c["width"] * np.interp(np.interp(d, dM, sM), kx, ky)
    on = lambda d: _along(M, d)
    top = lambda d: on(d) + _unit(_along(M, d, nM)) * np.asarray(width(d))[..., None]      # the full band's top edge
    at_s = lambda s: np.interp(s, sM, dM)                                                     # arc length at lid parameters
    count = lambda length, dens=1.0: max(int(np.ceil(length / c["step"] * dens)), 4)

    # ---- the band: where it meets the lid, its inner point, the notch's apex, the turn, the elbow, its outer point
    k = int(np.argmax(M[:, 1] >= c["peel_v"]))
    d_peel = dM[k - 1] + (c["peel_v"] - M[k - 1, 1]) / (M[k, 1] - M[k - 1, 1]) * (dM[k] - dM[k - 1])
    d_ramp, d_turn = d_peel + c["ramp"], float(at_s(c["turn"]))
    p_peel = on(d_peel)
    t_in = M[0] + np.array(c["tail"])
    apex = p_peel + _unit(_along(M, d_peel, nM)) * c["notch"]
    h_ramp, h_turn = top(d_ramp), top(d_turn)
    elbow = M[-1] + np.array(c["elbow"])
    R = ML[::-1]                                                  # the lower lid from the outer corner
    k = int(np.argmax(R[:, 1] <= c["out_v"]))
    t_out = R[k - 1] + (c["out_v"] - R[k - 1, 1]) / (R[k, 1] - R[k - 1, 1]) * (R[k] - R[k - 1])
    side = np.vstack([R[:k], [t_out]])
    # 1. the tail: straight from the inner point to the meeting point (lo) and to the notch's apex (hi)
    f = np.linspace(0.0, 1.0, count(np.linalg.norm(p_peel - t_in)) + 1)[:-1, None]
    lo, hi = [t_in + f * (p_peel - t_in)], [t_in + f * (apex - t_in)]
    s = [lid_s(lo[0][:, 0])]
    # 2. along the lid to the turn, the top edge rising straight from the apex to the full band over `ramp`
    q = np.linspace(d_peel, d_turn, count(d_turn - d_peel) + 1)[:-1]
    r = np.clip((q - d_peel) / c["ramp"], 0.0, 1.0)[:, None]
    lo.append(on(q))
    hi.append(np.where(q[:, None] <= d_ramp, apex + r * (h_ramp - apex), top(q)))
    s.append(np.interp(q, dM, sM))
    # 3. the turn to the outer corner: the top edge curves away to the elbow
    lead = _unit(h_turn - top(d_turn - 0.001))
    up_edge = _hermite(h_turn, lead, elbow, _dir(c["elbow_in"]), 200)
    n3 = count(dM[-1] - d_turn, 1.6)
    q = np.linspace(d_turn, dM[-1], n3 + 1)[:-1]
    lo.append(on(q))
    hi.append(_resample(up_edge, n3 + 1)[:-1])
    s.append(np.interp(q, dM, sM))
    # 4. down the eye's outer side: the lower lid's margin (lo) and the outer edge from the elbow to the point (hi), which
    # meets the lid `out_tip` degrees outside it
    n4 = count(_arc_len(side)[-1], 1.4)
    lo.append(_resample(side, n4 + 1))
    hi.append(_resample(_hermite(elbow, _dir(c["elbow_out"]), t_out, _turned(_unit(t_out - side[-2]), -c["out_tip"]), 200),
                        n4 + 1))
    s.append(1.0 + OUTER_RUN * np.linspace(0.0, 1.0, n4 + 1))
    lo, hi, s = np.vstack(lo), np.vstack(hi), np.concatenate(s)
    out = {"lash": dict(lo=lo, hi=hi, s=s, ends_pointed=True, normal=_unit(hi - lo),
                        hold=(float(np.interp(d_peel, dM, sM)), float(c["turn"])),
                        rim=np.concatenate([iu[dM >= d_peel], il[::-1][1:k]]))}

    # ---- the fork: its top edge leaves the band's top edge at `base` and curves out to its point; its lower edge is the
    # notch's upper side, from the apex
    fk = dict(LASH_FORK, **(cfg.get("lashfork") or {}))
    t_up = apex + fk["length"] * _dir(fk["angle"])
    d0 = float(at_s(fk["base"]))
    h0 = top(d0)
    f_hi = _hermite(h0, _unit(h0 - top(d0 + 0.001)), t_up, _turned(_unit(t_up - apex), fk["point"]), fk["n"])
    f_lo = np.linspace(apex, t_up, fk["n"])
    out["lashfork"] = dict(lo=f_lo, hi=f_hi, s=np.maximum(lid_s(0.5 * (f_lo[:, 0] + f_hi[:, 0])), s[0]), tip=True)

    # ---- the wing: two edges from the band's top edge to the point, each sagging below its chord
    wg = dict(LASH_WING, **(cfg.get("lashwing") or {}))
    w_root, w_base = (_along(up_edge, wg[key] * _arc_len(up_edge)[-1]) for key in ("root", "base"))
    w_tip = 0.5 * (w_root + w_base) + wg["length"] * _dir(wg["angle"])

    def sagging(a, z, sag):
        d = z - a
        return _bez2(a, 0.5 * (a + z) + 2.0 * sag * _unit(np.array([d[1], -d[0]])), z, wg["n"])
    out["lashwing"] = dict(lo=sagging(w_base, w_tip, wg["sag"][1]), hi=sagging(w_root, w_tip, wg["sag"][0]),
                           s=np.linspace(float(lid_s(w_root[0])), 1.0 + OUTER_RUN, wg["n"]), tip=True)

    # ---- the crease: a stroke over the band's top edge
    ck = dict(CREASE, **(cfg.get("crease") or {}))
    ref = top(np.linspace(at_s(ck["s0"]), at_s(ck["s1"]), 40))
    ref = _resample(np.vstack([ref[0] + ck["ext"] * _unit(ref[0] - ref[2]), ref]), ck["n"])
    tg = _unit(np.gradient(ref, axis=0))
    up = np.stack([-tg[:, 1], tg[:, 0]], -1)
    d = _arc_len(ref)
    gx, gy = np.array(ck["gap"], float).T
    gap = np.interp(d / d[-1], gx, gy)
    w = np.minimum(ck["width"], np.minimum(ck["slope"][0] * d, ck["slope"][1] * (d[-1] - d)))
    c_lo = ref + up * gap[:, None]
    out["crease"] = dict(lo=c_lo, hi=c_lo + up * w[:, None], s=lid_s(c_lo[:, 0]), gap=gap, ends_pointed=True)
    return out


def _tapered(M, off, width, s0, s1, n, up_sign, taper=0.7, bias=0.0, skew=0.0):
    """A thin lens-shaped strip along a margin polyline M (k, 2; inner -> outer corner) between the lid parameters s0..s1,
    offset `off` from it (away from the opening; a number or a function of the lid parameter) with the width profile
    sin(pi t)^taper (heavier towards s1 by `skew`); pointed at both ends. Returns (lo, hi, lid parameters)."""
    q = np.linspace(s0, s1, n)
    P = _at_s(M, q)
    Nq = _at_s(M, q, _normals(M, up_sign, bias=bias))
    Nq /= np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-12)
    t = np.linspace(0.0, 1.0, n)
    w = width * np.sin(np.pi * t) ** taper * (1.0 + skew * (t - 0.5))
    offv = off(q) if callable(off) else off
    lo = P + Nq * np.broadcast_to(offv, (n,))[:, None]
    return lo, lo + Nq * w[:, None], q


def lower_lash(eye, cfg=None):
    """The lower lash: a thin line under the bottom of the lower lid, heaviest towards its outer end."""
    c = dict(LOWER_LASH)
    c.update(cfg or {})
    _, M = _sorted_arc(eye, "lower")
    lo, hi, s = _tapered(M, c["gap"], c["width"], c["s0"], c["s1"], c["n"], -1.0, skew=c.get("skew", 0.0))
    return dict(lo=lo, hi=hi, tip=False, ends_pointed=True, s=s)
