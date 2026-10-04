"""Cat ear geometry (bpy-free, numpy only): one closed shell with a recessed inner face inside a black rim, and the pale
tufts of fur at its lower inner edge.

Local frame of an ear: x OUTWARD (away from the head's midline), y the FACING direction (the inner face looks along +y),
z UP along the ear's axis; z = 0 at the base centre (the axis point sunk into the skin). t is the height above the hair
hull as a fraction of the visible height H: t = 0 at the hull crossing of the axis (z = z_h), t = 1 at the tip.

The shell is a loft of closed rings (one per height). A ring is a crescent: a convex BACK arc from the inner edge to the
outer edge, then the FRONT arc back again: a flat rim crest, the lip, a steep wall and a shallow bowl floor. Faces inside the
lip are the inner material (index 1), everything else the black fur (index 0). Below t = 0 the ear keeps its width and runs
down into the hair; its first ring is lowered onto the skin (`clip(x, y)` gives the local height of the skin there, sunk).
The tip is a single vertex."""
from dataclasses import dataclass

import numpy as np

from .cat_common import pchip
from .hair_geo import sweep, unit

# planform: half widths (units of W / 2) of the inner (-x) and outer (+x) edge against t (a fat triangle, straight in the
# middle, a touch concave near the needle tip)
KEYS_T = [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.94, 1.0]
KEYS_OUT = [1.00, 1.00, 0.94, 0.82, 0.66, 0.48, 0.28, 0.13, 0.0]
KEYS_IN = [1.00, 0.98, 0.90, 0.77, 0.61, 0.43, 0.25, 0.12, 0.0]

XI_BACK = np.array([-1.0, -0.80, -0.45, 0.0, 0.45, 0.80, 1.0])                  # inner edge -> outer edge
T_ROWS = [0.0, 0.07, 0.15, 0.24, 0.33, 0.42, 0.52, 0.62, 0.72, 0.81, 0.89, 0.95]
T_CUP_TOP = 0.90                                                                 # faces above this ring stay black


@dataclass
class EarParams:
    W: float = 0.0739                # width where the ear leaves the hair (t = 0)
    H: float = 0.0936                # visible height above the hair hull
    z_h: float = 0.036               # height of the hair hull above the base centre along the axis
    thick: float = 0.011             # bulge of the back behind the median at the root
    rim_front: float = 0.0025        # rim crest height in front of the median
    rim_back: float = 0.0015         # edge height behind the median
    cup_depth: float = 0.0075        # depth of the bowl below the rim crest
    lip: float = 0.72                # half width of the bowl as a fraction of the ear's half width
    lean: float = 0.14               # the tip leans along +y by lean * H
    curl: float = 0.26               # the edges stand curl * half width in front of the middle
    apex_shift: float = 0.10         # the tip sits this fraction of W outward of the axis
    t_first: float = -0.16           # lowest ring above the clip ring
    base_taper: float = 0.5          # below the hull the half width shrinks by this x |t|


class EarShape:
    """The analytic ear: planform, median surface and thickness as functions of (xi, t)."""

    def __init__(self, p: EarParams):
        self.p = p
        a = 0.5 * p.W
        sh = p.apex_shift * p.W
        fo, fi = pchip(KEYS_T, KEYS_OUT), pchip(KEYS_T, KEYS_IN)
        self._xo = lambda t: a * fo(t) + sh * np.clip(t, 0, 1) ** 1.6
        self._xi = lambda t: -a * fi(t) + sh * np.clip(t, 0, 1) ** 1.6
        lp = p.lip
        # front arc from the outer side: crest, lip, wall, floor ... back to the inner side
        self.xf = np.array([lp + 0.12, lp, 0.82 * lp, 0.47 * lp, 0.0, -0.47 * lp, -0.82 * lp, -lp, -(lp + 0.12)])
        # the axis (x = y = 0) runs through the middle of the thickness at the root of the ear
        self.y0 = 0.5 * (-p.rim_back - p.thick + p.rim_front - p.cup_depth)

    def edges(self, t):
        t = np.maximum(np.asarray(t, float), 0.0)
        return self._xi(t), self._xo(t)

    def centre_half(self, t):
        """Centre x and half width at height t; below the hull (t < 0) the ear narrows toward its sunk base."""
        xi, xo = self.edges(t)
        k = 1.0 + self.p.base_taper * np.minimum(np.asarray(t, float), 0.0)
        return 0.5 * (xi + xo), 0.5 * (xo - xi) * k

    def z_of(self, t):
        return self.p.z_h + np.asarray(t, float) * self.p.H

    def median(self, xi, t):
        """y of the median surface at (xi, t): the lean of the whole ear plus the curl of its edges."""
        p = self.p
        tp = np.clip(np.asarray(t, float), 0.0, 1.0)
        _, hw = self.centre_half(t)
        return p.lean * p.H * tp ** 1.8 + p.curl * hw * np.asarray(xi, float) ** 2 - self.y0

    def back_thick(self, t):
        t = np.asarray(t, float)
        s = np.clip((t - 0.12) / 0.85, 0, 1)
        s = s * s * (3 - 2 * s)
        return np.maximum(self.p.thick * (1.0 - 0.86 * s), 0.0013)

    def crest(self, t):
        s = np.clip(np.asarray(t, float), 0, 1)
        return self.p.rim_front * (1.0 - 0.5 * s)

    def depth(self, t):
        """Depth of the bowl: full below t = 0.3, gone at t = 0.92."""
        t = np.asarray(t, float)
        s = np.clip((t - 0.30) / 0.62, 0, 1)
        s = s * s * (3 - 2 * s)
        return self.p.cup_depth * (1.0 - s)

    def front(self, xi, t):
        """y of the front surface at (xi, t) for |xi| <= crest (rim crest, lip, wall, floor)."""
        xi = np.asarray(xi, float)
        q = np.clip(1.0 - (xi / self.p.lip) ** 2, 0.0, 1.0) ** 0.6
        return self.median(xi, t) + self.crest(t) - self.depth(t) * q

    def back(self, xi, t):
        xi = np.asarray(xi, float)
        return (self.median(xi, t) - self.p.rim_back - self.back_thick(t) * np.clip(1.0 - xi * xi, 0, 1) ** 0.7)

    def ring(self, t):
        """(16, 3) loop vertices (x, y, z local) of the ring at height t and the xi of every vertex."""
        c, hw = self.centre_half(t)
        xb, xf = XI_BACK, self.xf
        yb = self.back(xb, t)
        yf = self.front(xf, t)
        # edge vertices (xi = +-1) are shared by the back arc: y of the median minus the edge offset
        yb[0] = self.median(-1.0, t) - self.p.rim_back
        yb[-1] = self.median(1.0, t) - self.p.rim_back
        x = np.concatenate([c + xb * hw, c + xf * hw])
        y = np.concatenate([yb, yf])
        z = np.full(len(x), float(self.z_of(t)))
        return np.stack([x, y, z], -1), np.concatenate([xb, xf])

    def apex(self):
        c, _ = self.centre_half(1.0)
        return np.array([float(c), float(self.median(0.0, 1.0)) + 0.0, float(self.z_of(1.0))])


def build_shell(p: EarParams, clip):
    """The shell in local coordinates. `clip(x, y) -> z`: local height of the sunk skin below the footprint.
    Returns a dict: verts (n,3), faces, uv (n,2), face_mat (0 black / 1 inner), t (n,), xi (n,), R (loop size), rings,
    base_centre (index of the vertex closing the bottom), apex (index of the tip vertex), ring_t, shape (EarShape)."""
    sh = EarShape(p)
    R = len(XI_BACK) + len(sh.xf)
    loop0, xi_loop = sh.ring(-p.z_h / p.H)                  # the base ring: the planform where the axis meets the skin
    zc = np.asarray(clip(loop0[:, 0], loop0[:, 1]), float)
    # the first planar ring must sit above the highest clipped base point
    t_lo = max(p.t_first, (float(zc.max()) + 0.004 - p.z_h) / p.H)
    ring_t = [t_lo] + list(T_ROWS)
    base = loop0.copy()
    base[:, 2] = zc
    verts, xis, ts = [base], [xi_loop], [(zc - p.z_h) / p.H]
    for t in ring_t:
        v, xi = sh.ring(t)
        verts.append(v)
        xis.append(xi)
        ts.append(np.full(R, t))
    nr = len(verts)
    apex_i = nr * R
    centre_i = apex_i + 1
    cen = base.mean(0)
    cen[2] = float(zc.min()) - 0.002
    V = np.concatenate(verts + [sh.apex()[None], cen[None]], 0)
    XI = np.concatenate(xis + [np.zeros(2)])
    T = np.concatenate(ts + [np.ones(1), np.full(1, float((cen[2] - p.z_h) / p.H))])
    ok = [abs(sh.xf[k]) <= p.lip + 1e-9 and abs(sh.xf[k + 1]) <= p.lip + 1e-9 for k in range(len(sh.xf) - 1)]
    faces, fmat = [], []
    for i in range(nr - 1):                              # quad row i joins ring i (lower) to ring i + 1 (upper)
        t_up = ring_t[i]                                 # ring i + 1 is ring_t[i]
        for j in range(R):
            k = (j + 1) % R
            faces.append([i * R + j, i * R + k, (i + 1) * R + k, (i + 1) * R + j])
            f = j - len(XI_BACK)                         # index of the front edge, if it is one
            fmat.append(1 if (0 <= f < len(ok) and ok[f] and t_up <= T_CUP_TOP + 1e-9) else 0)
    for j in range(R):                                   # fan to the tip
        faces.append([(nr - 1) * R + j, (nr - 1) * R + (j + 1) % R, apex_i])
        fmat.append(0)
    for j in range(R):                                   # the bottom, closed with a fan (it lies inside the head)
        faces.append([centre_i, (j + 1) % R, j])
        fmat.append(0)
    uv = np.stack([0.5 * (XI + 1.0), np.clip((T + 0.5) / 1.5, 0.0, 1.0)], -1)
    return dict(verts=V, faces=faces, uv=uv, face_mat=np.array(fmat, int), t=T, xi=XI, R=R, rings=nr, shape=sh,
                apex=apex_i, base_centre=centre_i, ring_t=ring_t)


# pale locks at the lower inner edge: (t of the root, xi of the root, length, sweep angle from the ear's up direction toward
# the outer side (deg), width); each lock rises off the floor of the bowl and sweeps across it like fur parted sideways
TUFTS = [(0.34, -0.60, 0.029, 14.0, 0.0072), (0.22, -0.66, 0.034, 32.0, 0.0080), (0.10, -0.68, 0.033, 52.0, 0.0080),
         (-0.01, -0.64, 0.028, 72.0, 0.0072), (0.43, -0.50, 0.022, 6.0, 0.0060)]


def tuft_specs(sh: EarShape, n, W, rng):
    """`n` pale locks rooted at the foot of the inner wall of the bowl, rising off its floor and curling across it toward the
    ear's axis, tips lifting a little off the surface (local frame): list of dict(P (7,3) centreline, width, thick)."""
    k = W / 0.0739
    p = sh.p
    out = []
    for i in range(n):
        t0, xi0, L, th, wd = TUFTS[i % len(TUFTS)]
        th = np.radians(th + rng.normal(0, 2.0))
        L = (L + rng.normal(0, 0.0012)) * k
        c, hw = sh.centre_half(max(t0, 0.0))
        x0 = float(c + xi0 * hw)
        z0 = float(sh.z_of(t0))
        P = []
        for s in np.linspace(0.0, 1.0, 7):
            x = x0 + L * np.sin(th) * s ** 1.15
            z = z0 + L * np.cos(th) * s
            t = (z - p.z_h) / p.H
            cc, hh = sh.centre_half(max(t, 0.0))
            xi = float(np.clip((x - cc) / hh, -0.62 * p.lip / 0.72, 0.62 * p.lip / 0.72))
            sm = np.clip(s / 0.5, 0, 1)
            lift = -0.0014 * k + (0.0085 * (sm * sm * (3 - 2 * sm)) + 0.0035 * s * s) * k
            P.append([x, float(sh.front(xi, t)) + lift, z])
        out.append(dict(P=np.array(P), width=wd * k, thick=0.0034 * k))
    return out


def _flame(sp):
    """Width of a fur lock against s' (0 root .. 1 tip): a little fuller just above the root, then a convex taper."""
    sp = np.clip(sp, 0.0, 1.0)
    return (0.82 + 0.38 * np.sin(np.pi * np.minimum(sp / 0.55, 1.0) * 0.5)) * (1.0 - sp ** 1.5) ** 0.9


def build_tufts(specs):
    """Closed pointed strips for the tuft specs (local frame): list of dict(verts, faces, uv)."""
    out = []
    for sp in specs:
        P = sp["P"]
        st = sweep(P, np.array([0.0, 1.0, 0.0]), sp["width"], sp["thick"], tip="point", tip_start=0.999, tip_power=1.0,
                   rings=7, k_outer=3, outer_frac=0.5, bulge=0.8, close_root=True, width_fn=_flame)
        s = np.clip(st.arc / max(st.length, 1e-9), 0.0, 1.0)
        uv = np.stack([0.5 * (st.q + 1.0), s], -1)
        out.append(dict(verts=st.verts, faces=st.faces, uv=uv, arc=st.arc, length=st.length))
    return out
