"""Geometry of the bedroom's soft furnishings (the single bed and the rug): pure numpy, no bpy, importable from tests.

Everything is in the prop's own frame (metres, Z up, floor z = 0; a bed's head is at +Y, its foot at -Y, the rug's front
is -Y). The Blender side (`bedroom_soft.py`) only turns the arrays made here into meshes.

Cloth. A quilt or a sheet is a *flat sheet* with coordinates (s, t) in metres, wrapped over the bed in four steps:
  1. corners: the sheet's outline is a rectangle whose corners are quarter ellipses (the corner squares are pulled onto
     them), so the hem keeps one height all the way round instead of drooping where the diagonal is longer;
  2. fold: the part of the sheet beyond `Lc` is turned back over itself (a rolled hinge, then a returned band lying on
     the quilt, its lining facing up), expressed as an effective position `y` and a lift above the base surface;
  3. wrap: the flat top region [-A, A] x [Y0, Y1] stays horizontal, the rest bends over the mattress edge and hangs
     (`Profile` curves for the sides, the foot and the head, blended round the corners by the direction seen from the
     region's corner, in units of the distance to the outline so that every profile ends exactly at the hem);
  4. displacement: deterministic smooth noise (sag on the field, pleats on the hang), then a push out of the solid
     furniture (`Support`, a signed distance field) so the cloth never cuts into the frame.
The mid-surface is then thickened into a closed shell (`Shell`): pattern side, lining side and a rounded rim.

Determinism. The noise is a lattice hash in uint32 arithmetic, identical on every platform and numpy version; there is
no random state anywhere."""
import math

import numpy as np

TAU = 2.0 * math.pi

# ===================================================================================================== the numbers
# --- the frame (outer dimensions; centred on the footprint, the head is at +Y)
FRAME_HX, FRAME_HY = 0.50, 1.00             # half footprint: 1.00 x 2.00
POST_W = 0.055                              # square posts at the four corners
HEAD_TOP, FOOT_TOP = 0.95, 0.55             # top of the head posts / of the foot posts and the footboard
RAIL_T, RAIL_Z0, RAIL_Z1 = 0.025, 0.18, 0.30    # side rails: thickness, bottom, top (the mattress sits on them)
BOARD_T = 0.030                             # thickness (along y) of the head and foot panels
# --- the mattress
MAT_HX, MAT_HY = 0.48, 0.945                # half width, half length
MAT_Z0, MAT_Z1 = 0.30, 0.50                 # bottom, top
MAT_R = 0.035                               # edge radius
SHEET_T = 0.004                             # fitted sheet thickness over the top
SHEET_Z = MAT_Z1 + SHEET_T                  # top of the sheet
SHEET_HANG = 0.06                           # how far the fitted sheet reaches down the mattress side
# --- the quilt
QUILT_T, QUILT_T_EDGE = 0.030, 0.020        # thickness over the field / at the binding
QUILT_GAP = 0.002                           # air between the sheet and the quilt
HANG_SIDE, HANG_FOOT = 0.20, 0.12           # how far the outer surface hangs below the mattress top (z = MAT_Z1)
FOLD_FROM_HEAD = 0.45                       # the quilt ends (rolls back) this far from the head end of the mattress
BAND = 0.27                                 # length of the returned band
ROLL = 0.012                                # extra thickness of the rolled hinge
FAR_CORNER = 0.06                           # corner radius at the quilt's rolled end
# --- the rug
RUG_W, RUG_D, RUG_T = 1.80, 1.20, 0.012
RUG_FRINGE = 0.055                          # length of the fringe beyond the two short edges
RUG_CORNER = 0.02


# ===================================================================================================== noise
def _lattice(ix, iy, iz, seed):
    with np.errstate(over="ignore"):
        h = (ix.astype(np.uint32) * np.uint32(0x9E3779B1) + iy.astype(np.uint32) * np.uint32(0x85EBCA77)
             + iz.astype(np.uint32) * np.uint32(0xC2B2AE3D) + np.uint32((int(seed) * 0x27D4EB2F) & 0xFFFFFFFF))
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    return h.astype(np.float64) / 4294967296.0 * 2.0 - 1.0


def _fade(f):
    return f * f * f * (f * (f * 6.0 - 15.0) + 10.0)


def vnoise3(x, y, z=0.0, seed=0):
    """Smooth value noise in [-1, 1] (quintic interpolation of a hashed lattice); x, y, z broadcast."""
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    u, v, w = _fade(x - x0), _fade(y - y0), _fade(z - z0)
    ix, iy, iz = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)

    def c(dx, dy, dz):
        return _lattice(ix + dx, iy + dy, iz + dz, seed)

    x00 = c(0, 0, 0) * (1 - u) + c(1, 0, 0) * u
    x10 = c(0, 1, 0) * (1 - u) + c(1, 1, 0) * u
    x01 = c(0, 0, 1) * (1 - u) + c(1, 0, 1) * u
    x11 = c(0, 1, 1) * (1 - u) + c(1, 1, 1) * u
    return (x00 * (1 - v) + x10 * v) * (1 - w) + (x01 * (1 - v) + x11 * v) * w


def fbm3(x, y, z=0.0, seed=0, octaves=3, gain=0.5):
    """Sum of `octaves` value-noise octaves (frequency x2 each), normalised to about [-1, 1]."""
    tot, amp, norm, f = 0.0, 1.0, 0.0, 1.0
    for o in range(octaves):
        tot = tot + amp * vnoise3(x * f, y * f, z * f, seed + 17 * o)
        norm += amp
        amp *= gain
        f *= 2.0
    return tot / norm


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ===================================================================================================== sampling
def sample_line(lo, hi, zones=(), base=0.04):
    """Increasing positions from `lo` to `hi` (both included) with spacing `base`, finer inside the zones (a, b, step)."""
    xs = [float(lo)]
    x = float(lo)
    while x < hi - 1e-9:
        step = base
        for a, b, st in zones:
            if a - 1e-9 <= x < b:
                step = min(step, st)
        x = min(x + step, float(hi))
        xs.append(x)
    if len(xs) > 2 and xs[-1] - xs[-2] < 0.35 * min(base, 0.02):
        del xs[-2]
    return np.array(xs)


# ===================================================================================================== profiles
def fillet_path(pts, radii, step=0.003):
    """Dense polyline along `pts` (2D) with a circular fillet of radius radii[i] at each interior vertex."""
    P = [np.asarray(p, float) for p in pts]
    n = len(P)
    seg = [P[i + 1] - P[i] for i in range(n - 1)]
    ln = [float(np.hypot(*s)) for s in seg]
    unit = [s / l for s, l in zip(seg, ln)]
    T, ang = np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        u0, u1 = -unit[i - 1], unit[i]
        ang[i] = math.acos(max(-1.0, min(1.0, float(np.dot(u0, u1)))))          # interior angle between the segments
        if radii[i] > 0 and 1e-6 < ang[i] < math.pi - 1e-6:
            T[i] = radii[i] / math.tan(ang[i] / 2)
    for i in range(n - 1):                                                         # fillets must fit on the segment
        if T[i] + T[i + 1] > ln[i]:
            k = ln[i] / (T[i] + T[i + 1])
            T[i] *= k
            T[i + 1] *= k
    out = [P[0]]
    for i in range(1, n - 1):
        if T[i] <= 0:
            out.append(P[i])
            continue
        u0, u1 = -unit[i - 1], unit[i]
        r = T[i] * math.tan(ang[i] / 2)
        A, B = P[i] + u0 * T[i], P[i] + u1 * T[i]
        bis = (u0 + u1) / np.hypot(*(u0 + u1))
        C = P[i] + bis * (r / math.sin(ang[i] / 2))
        a0, a1 = math.atan2(*(A - C)[::-1]), math.atan2(*(B - C)[::-1])
        sweep = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        m = max(2, int(math.ceil(abs(sweep) * r / step)))
        for k in range(m + 1):
            a = a0 + sweep * k / m
            out.append(C + r * np.array([math.cos(a), math.sin(a)]))
    out.append(P[-1])
    return np.array(out)


class Profile:
    """A curve (a, b) of a cloth's cross-section, parametrised by the arc length d from its start; beyond both ends it
    continues straight along the end tangents. `at(d)` -> a, b, da, db (unit tangent). `hang` is the arc length where
    the last, nearly straight run begins (the cloth hangs from there)."""

    def __init__(self, pts, radii, step=0.003):
        Q = fillet_path(pts, radii, step)
        ln = np.hypot(*np.diff(Q, axis=0).T)
        Q = Q[np.concatenate([[True], ln > 1e-9])]
        self.d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
        self.a, self.b = Q[:, 0], Q[:, 1]
        tg = np.gradient(Q, self.d, axis=0)
        tg /= np.hypot(tg[:, 0], tg[:, 1])[:, None]
        self.ta, self.tb = tg[:, 0], tg[:, 1]
        for i, (p0, p1) in ((0, (pts[0], pts[1])), (-1, (pts[-2], pts[-1]))):     # straight end runs: exact tangents
            v = np.asarray(p1, float) - np.asarray(p0, float)
            self.ta[i], self.tb[i] = v / np.hypot(*v)
        self.length = float(self.d[-1])
        low = np.nonzero(np.abs(self.tb) < 0.985)[0]
        self.hang = float(self.d[low[-1]]) if len(low) else 0.0

    def at(self, d):
        d = np.asarray(d, float)
        dc = np.clip(d, 0.0, self.length)
        ex = d - dc
        a = np.interp(dc, self.d, self.a) + ex * np.where(ex < 0, self.ta[0], self.ta[-1])
        b = np.interp(dc, self.d, self.b) + ex * np.where(ex < 0, self.tb[0], self.tb[-1])
        ta, tb = np.interp(dc, self.d, self.ta), np.interp(dc, self.d, self.tb)
        nrm = np.hypot(ta, tb)
        return a, b, ta / nrm, tb / nrm


# ===================================================================================================== solid support
class RoundBox:
    """Axis-aligned box with rounded edges (signed distance: negative inside)."""

    def __init__(self, lo, hi, r=0.0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.c, self.h, self.r = (lo + hi) / 2, (hi - lo) / 2, min(float(r), float(((hi - lo) / 2).min()))

    def sdf(self, P):
        q = np.abs(P - self.c) - (self.h - self.r)
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0) - self.r


class Support:
    """A union of rounded boxes: the furniture a cloth must stay out of."""

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def sdf(self, P):
        P = np.asarray(P, float)
        return np.min([b.sdf(P) for b in self.boxes], axis=0)

    def grad(self, P, eps=5e-4):
        g = np.zeros_like(P)
        for k in range(3):
            e = np.zeros(3)
            e[k] = eps
            g[..., k] = (self.sdf(P + e) - self.sdf(P - e)) / (2 * eps)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        return g / np.maximum(n, 1e-9)


def _softpos(v, s):
    """max(v, 0) smoothed over [-s, s] (C1): 0 below -s, (v + s)^2 / 4s inside, v above s."""
    v = np.asarray(v, float)
    return np.where(v >= s, v, np.where(v <= -s, 0.0, (v + s) ** 2 / (4 * s)))


def push_out(P, clearance, support, iters=4, soft=0.006):
    """Move the points of P (..., 3) out of `support` until their distance to it is at least `clearance` (array or
    number), along the field's gradient, smoothly (no crease where the push starts)."""
    P = np.array(P, float)
    flat = P.reshape(-1, 3)
    clr = np.broadcast_to(np.asarray(clearance, float), P.shape[:-1]).reshape(-1)
    for _ in range(iters):
        amt = _softpos(clr - support.sdf(flat), soft)
        m = amt > 1e-7
        if not m.any():
            break
        flat[m] += support.grad(flat[m]) * amt[m][:, None]
    return flat.reshape(P.shape)


def dilate(mask, cells):
    """Boolean mask grown by `cells` grid cells (4-neighbourhood, edges clamped)."""
    m = np.asarray(mask, bool)
    for _ in range(int(cells)):
        p = np.pad(m, 1, mode="edge")
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def blur(x, passes=2):
    """[1 2 1] / 4 smoothing of a 2D array along both grid axes (edges clamped)."""
    x = np.asarray(x, float)
    for _ in range(passes):
        p = np.pad(x, ((1, 1), (0, 0)), mode="edge")
        x = (p[:-2] + 2 * x + p[2:]) / 4.0
        p = np.pad(x, ((0, 0), (1, 1)), mode="edge")
        x = (p[:, :-2] + 2 * x + p[:, 2:]) / 4.0
    return x


def relax_clear(P, clearance, support, active, iters=40, step=0.5, soft=0.002):
    """Smooth the cloth grid P (Nt, Ns, 3) where `active` (0..1, same grid) says so, keeping it `clearance` away from
    `support`: Laplacian steps over the four grid neighbours alternate with push-outs, so a cloth that was shoved out of a
    post by an abrupt push settles into a smooth bulge (the way a stiff cloth bridges an obstacle) with no folds."""
    P = np.array(P, float)
    w = (step * np.asarray(active, float))[..., None]
    for _ in range(iters):
        p = np.pad(P, ((1, 1), (1, 1), (0, 0)), mode="edge")
        P += w * ((p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0 - P)
        P = push_out(P, clearance, support, iters=2, soft=soft)
    return P


def mattress_box(grow=0.0):
    return RoundBox((-MAT_HX - grow, -MAT_HY - grow, MAT_Z0), (MAT_HX + grow, MAT_HY + grow, MAT_Z1 + grow),
                    MAT_R + grow)


def bed_support():
    """The solids the quilt must stay out of: the mattress under its sheet, the side rails, the foot posts and the
    footboard (the quilt ends before the head)."""
    g = SHEET_T * 0.8
    boxes = [RoundBox((-MAT_HX - g, -MAT_HY - g, MAT_Z0), (MAT_HX + g, MAT_HY + g, MAT_Z1 + SHEET_T), MAT_R + g)]
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (FRAME_HX - RAIL_T), sx * FRAME_HX))
        boxes.append(RoundBox((x0, -FRAME_HY, RAIL_Z0), (x1, FRAME_HY, RAIL_Z1), 0.003))
        p0, p1 = sorted((sx * (FRAME_HX - POST_W), sx * FRAME_HX))
        boxes.append(RoundBox((p0, -FRAME_HY, 0.0), (p1, -FRAME_HY + POST_W, FOOT_TOP), 0.004))
    boxes.append(RoundBox((-FRAME_HX, -FRAME_HY, 0.20), (FRAME_HX, -FRAME_HY + BOARD_T, FOOT_TOP), 0.004))
    return Support(boxes)


# ===================================================================================================== outlines
def round_corners(S, T, hw, t_lo, t_hi, lo_r, hi_r):
    """Pull the points of the four corner rectangles of [-hw, hw] x [t_lo, t_hi] onto quarter ellipses (semi-axes
    (rx, ry) = lo_r at t_lo, hi_r at t_hi) by the radial map L-inf -> L2 in the corner's own units: the outline becomes
    a rectangle with elliptical corners and lines through the corner centres stay straight."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    S2, T2 = S.copy(), T.copy()
    for tc, sg, (rx, ry) in ((t_lo + lo_r[1], 1.0, lo_r), (t_hi - hi_r[1], -1.0, hi_r)):
        sc = hw - rx
        qx = np.maximum(np.abs(S) - sc, 0.0)
        qy = np.maximum((tc - T) * sg, 0.0)
        nx, ny = qx / rx, qy / ry
        m2 = np.hypot(nx, ny)
        k = np.where(m2 > 1e-12, np.maximum(nx, ny) / np.maximum(m2, 1e-12), 1.0)
        inside = (qx > 0) & (qy > 0)
        S2 = np.where(inside, np.sign(S) * (sc + qx * k), S2)
        T2 = np.where(inside, tc - sg * qy * k, T2)
    return S2, T2


def loop_distance(points, loop):
    """Distance from `points` (..., 2) to the closed polyline `loop` (n, 2)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    a, b = loop, np.roll(loop, -1, axis=0)
    ab = b - a
    l2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
    best = np.full(len(pts), 1e9)
    for lo in range(0, len(pts), 2048):
        p = pts[lo:lo + 2048][:, None, :]
        t = np.clip(((p - a) * ab).sum(axis=2) / l2, 0.0, 1.0)
        c = a + t[..., None] * ab
        best[lo:lo + 2048] = np.sqrt(((p - c) ** 2).sum(axis=2)).min(axis=1)
    return best.reshape(np.asarray(points).shape[:-1])


# ===================================================================================================== the drape
class Drape:
    """Wraps a flat sheet's points over a horizontal top region [-A, A] x [Y0, Y1] and the profiles `side` (both long
    sides), `foot` (y < Y0) and optionally `head` (y > Y1). A profile is a curve (h, z): h the distance outward from the
    region's edge, z the absolute height of the cloth's mid-surface; each ends at the hem."""

    def __init__(self, A, Y0, Y1, side, foot, head=None):
        self.A, self.Y0, self.Y1, self.side, self.foot, self.head = A, Y0, Y1, side, foot, head

    def wrap(self, S, Y):
        """dict with P (..., 3) base points, nu (..., 3) outward-up unit normals, flat (1 on the horizontal region,
        fading over the first 4 cm of the bend) and hang (0 on the field, 1 once the cloth hangs)."""
        S, Y = np.asarray(S, float), np.asarray(Y, float)
        sx = np.where(S < 0, -1.0, 1.0)
        dx = np.maximum(np.abs(S) - self.A, 0.0)
        df = np.maximum(self.Y0 - Y, 0.0)
        dh = np.maximum(Y - self.Y1, 0.0) if self.head is not None else np.zeros_like(Y)
        dl = df + dh
        d = np.hypot(dx, dl)
        ang = np.arctan2(dl, dx)
        w = smoothstep(ang / (math.pi / 2))
        Ls = self.side.length
        Ll = np.where(dh > 0, self.head.length, self.foot.length) if self.head is not None else self.foot.length
        # distance to the outline along this direction: radius of the quarter ellipse with semi-axes (Ls, Ll)
        c, s_ = np.cos(ang), np.sin(ang)
        dout = 1.0 / np.sqrt((c / Ls) ** 2 + (s_ / Ll) ** 2)
        tau = np.minimum(d / dout, 1.0 + 1e-9)
        ds, dlg = tau * Ls, tau * Ll
        hs, zs, tas, tbs = self.side.at(ds)
        hf, zf, taf, tbf = self.foot.at(dlg)
        hang_l = smoothstep((dlg - self.foot.hang) / 0.09)
        if self.head is not None:
            hh, zh, tah, tbh = self.head.at(dlg)
            ih = dh > 0
            hf, zf, taf, tbf = np.where(ih, hh, hf), np.where(ih, zh, zf), np.where(ih, tah, taf), np.where(ih, tbh, tbf)
            hang_l = np.where(ih, smoothstep((dlg - self.head.hang) / 0.09), hang_l)
        h, z = (1 - w) * hs + w * hf, (1 - w) * zs + w * zf
        ta, tb = (1 - w) * tas + w * taf, (1 - w) * tbs + w * tbf
        inv = 1.0 / np.maximum(d, 1e-9)
        dirx, diry = sx * dx * inv, (-df + dh) * inv
        cx = np.clip(S, -self.A, self.A)
        cy = np.clip(Y, self.Y0, self.Y1) if self.head is not None else np.maximum(Y, self.Y0)
        P = np.stack([cx + dirx * h, cy + diry * h, z], axis=-1)
        nrm = np.hypot(ta, tb)
        nu = np.stack([-tb * dirx, -tb * diry, ta], axis=-1) / nrm[..., None]
        hang = (1 - w) * smoothstep((ds - self.side.hang) / 0.09) + w * hang_l
        flat = 1.0 - smoothstep(d / 0.04)
        return {"P": P, "nu": nu, "flat": flat, "hang": hang, "d": d, "w": w}


# ===================================================================================================== the shell
class Shell:
    """A closed mesh made of quads: vertices V (n, 3), faces F (m, 4), per-face material index `mat`, per-vertex UVs
    `uv0` (the cloth's flat coordinates in metres) and `uv1` (distance to the outline in metres, 0)."""

    def __init__(self, V, F, mat, uv0, uv1):
        self.V, self.F, self.mat, self.uv0, self.uv1 = V, F, mat, uv0, uv1

    @property
    def tris(self):
        return 2 * len(self.F)

    def bounds(self):
        return self.V.min(axis=0), self.V.max(axis=0)

    def volume(self):
        """Signed volume (positive for outward-wound faces)."""
        a, b, c, d = (self.V[self.F[:, i]] for i in range(4))
        vol = 0.0
        for p, q, r in ((a, b, c), (a, c, d)):
            vol += float(np.einsum("ij,ij->i", p, np.cross(q, r)).sum()) / 6.0
        return vol

    def open_edges(self):
        """Number of edges that are not shared by exactly two faces (0 for a closed surface)."""
        e = np.concatenate([np.stack([self.F[:, i], self.F[:, (i + 1) % 4]], axis=1) for i in range(4)])
        e = np.sort(e, axis=1)
        e = e[e[:, 0] != e[:, 1]]
        _, cnt = np.unique(e, axis=0, return_counts=True)
        return int((cnt != 2).sum())

    def unused_vertices(self):
        return int(len(self.V) - len(np.unique(self.F)))


def compact(V, F, *attrs):
    """Drop the vertices no face uses; returns V, F and the attribute arrays, reindexed."""
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (V[used], remap[F]) + tuple(a[used] for a in attrs)


def grid_normals(P):
    """Unit normals of a grid of points P (Nt, Ns, 3): tangent_s x tangent_t (central differences by chord)."""
    def tangent(axis):
        t = np.empty_like(P)
        sl = [slice(None)] * 3
        a, b, mid = list(sl), list(sl), list(sl)
        a[axis], b[axis], mid[axis] = slice(2, None), slice(0, -2), slice(1, -1)
        t[tuple(mid)] = P[tuple(a)] - P[tuple(b)]
        lo, hi, lo1, hi1 = list(sl), list(sl), list(sl), list(sl)
        lo[axis], lo1[axis], hi[axis], hi1[axis] = 0, 1, -1, -2
        t[tuple(lo)] = P[tuple(lo1)] - P[tuple(lo)]
        t[tuple(hi)] = P[tuple(hi)] - P[tuple(hi1)]
        return t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)

    n = np.cross(tangent(1), tangent(0))
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def boundary_loop(Nt, Ns):
    """Grid indices (row, col) around the boundary, counter-clockwise seen from the top side (s right, t up)."""
    return ([(0, j) for j in range(Ns - 1)] + [(i, Ns - 1) for i in range(Nt - 1)]
            + [(Nt - 1, j) for j in range(Ns - 1, 0, -1)] + [(i, 0) for i in range(Nt - 1, 0, -1)])


def shell_from_grids(mid, half, uv0, uv1, rim_k=3):
    """Thicken the mid-surface grid `mid` (Nt, Ns, 3) by +-half (Nt, Ns) along its normals into a closed shell: top
    faces (material 0), bottom faces (1) and a rim (2) of `rim_k` points between the two skins round the boundary (a half
    circle of radius half). The boundary vertices of both skins belong to the rim, so shading runs smoothly round it."""
    Nt, Ns, _ = mid.shape
    n = grid_normals(mid)
    top, bot = mid + n * half[..., None], mid - n * half[..., None]
    nT = Nt * Ns
    ids_top = np.arange(nT).reshape(Nt, Ns)
    ids_bot = ids_top + nT
    V = [top.reshape(-1, 3), bot.reshape(-1, 3)]
    UV0 = [uv0.reshape(-1, 2)] * 2
    UV1 = [uv1.reshape(-1, 2)] * 2
    li = np.array(boundary_loop(Nt, Ns))
    nb = len(li)
    Pb, nb_n, hb = mid[li[:, 0], li[:, 1]], n[li[:, 0], li[:, 1]], half[li[:, 0], li[:, 1]]
    tb = np.roll(Pb, -1, axis=0) - np.roll(Pb, 1, axis=0)
    e_out = np.cross(tb, nb_n)
    e_out /= np.maximum(np.linalg.norm(e_out, axis=-1, keepdims=True), 1e-12)
    faces, mats = [], []
    for i in range(Nt - 1):
        for j in range(Ns - 1):
            faces.append((ids_top[i, j], ids_top[i, j + 1], ids_top[i + 1, j + 1], ids_top[i + 1, j]))
            mats.append(0)
    for i in range(Nt - 1):
        for j in range(Ns - 1):
            faces.append((ids_bot[i, j], ids_bot[i + 1, j], ids_bot[i + 1, j + 1], ids_bot[i, j + 1]))
            mats.append(1)
    ring = np.zeros((nb, rim_k + 2), dtype=np.int64)
    ring[:, 0] = ids_top[li[:, 0], li[:, 1]]
    ring[:, -1] = ids_bot[li[:, 0], li[:, 1]]
    extra = []
    for k in range(1, rim_k + 1):
        phi = math.pi * k / (rim_k + 1)
        extra.append(Pb + nb_n * (hb * math.cos(phi))[:, None] + e_out * (hb * math.sin(phi))[:, None])
        ring[:, k] = 2 * nT + (k - 1) * nb + np.arange(nb)
    if extra:
        V.append(np.concatenate(extra))
        UV0.append(np.tile(uv0[li[:, 0], li[:, 1]], (rim_k, 1)))
        UV1.append(np.tile(uv1[li[:, 0], li[:, 1]], (rim_k, 1)))
    for m in range(nb):
        m2 = (m + 1) % nb
        for k in range(rim_k + 1):
            faces.append((ring[m, k], ring[m, k + 1], ring[m2, k + 1], ring[m2, k]))
            mats.append(2)
    return Shell(np.concatenate(V), np.array(faces, dtype=np.int64), np.array(mats, dtype=np.int64),
                 np.concatenate(UV0), np.concatenate(UV1))


# ===================================================================================================== the quilt
SIDE_BEND = MAT_R + SHEET_T * 0.8 + QUILT_GAP + QUILT_T / 2          # the quilt's mid-surface radius over the mattress edge
Z_MID = MAT_Z1 + SHEET_T + QUILT_GAP + QUILT_T / 2                    # height of the quilt's mid-surface over the field
X_MID = MAT_HX + SHEET_T * 0.8 + QUILT_GAP + QUILT_T / 2              # its x on the hanging side
Y_FOLD = MAT_HY - FOLD_FROM_HEAD                                       # where the quilt rolls back
QUILT_Y0 = -MAT_HY + 0.045                                             # the foot edge of the flat region


def quilt_side_profile():
    """The quilt's mid-surface over the mattress edge and down the long sides; starts at A = X_MID - SIDE_BEND."""
    z_hem = MAT_Z1 - HANG_SIDE + QUILT_T_EDGE / 2 + 0.002
    return Profile([(0.0, Z_MID), (SIDE_BEND, Z_MID), (SIDE_BEND, z_hem)], [0, SIDE_BEND, 0])


def quilt_foot_profile(y0=QUILT_Y0):
    """The foot profile (h = y0 - y): it climbs from the field over the footboard, crosses it and hangs outside it."""
    inner = y0 + FRAME_HY - BOARD_T                    # the board's inner face (h)
    outer = y0 + FRAME_HY                              # and its outer face
    z_board = FOOT_TOP + QUILT_GAP + QUILT_T / 2 + 0.012
    out_mid = outer + QUILT_GAP + QUILT_T / 2 + 0.006
    z_hem = MAT_Z1 - HANG_FOOT + QUILT_T_EDGE / 2 + 0.002
    pts = [(0.0, Z_MID), (inner - 0.035, Z_MID), (inner + 0.012, z_board), (out_mid, z_board), (out_mid, z_hem)]
    return Profile(pts, [0, 0.03, 0.03, 0.03, 0])


def fold_map(T, Lc, y0, sep, bulge):
    """Effective y and lift of the sheet coordinate t: for t <= Lc the sheet lies flat (y = y0 + t); then a half circle of
    radius (sep + bulge) / 2 rolls it back; then it returns along -y, `sep` (+ a bulge dying away from the hinge) above
    the layer below."""
    T = np.asarray(T, float)
    y1 = y0 + Lc
    r = (sep + bulge) / 2
    th = np.clip((T - Lc) / r, 0.0, math.pi)
    back = T - Lc - math.pi * r
    y = np.where(T <= Lc, y0 + T, np.where(back < 0, y1 + r * np.sin(th), y1 - back))
    lift = np.where(T <= Lc, 0.0, np.where(back < 0, r * (1 - np.cos(th)), sep + bulge * np.exp(-((back / 0.07) ** 2))))
    return y, lift


def pleat_field(a, d, seed):
    """Pleats of a hanging edge, 0..1: two waves along the edge (coordinate `a`, metres) wandering with the distance d from
    the field. The sides and the foot each use their own coordinate (y, x) and the corner blends the VALUES: blending
    the coordinates would sweep the phase through metres in a few centimetres and fold the cloth."""
    ph1 = TAU * a / 0.19 + 2.6 * vnoise3(a * 1.9, d * 1.0, 0.0, seed + 2)
    ph2 = TAU * a / 0.31 + 3.4 * vnoise3(a * 1.3 + 5.0, d * 0.7, 1.0, seed + 9)
    return 0.5 + 0.32 * np.sin(ph1) + 0.18 * np.sin(ph2)


def quilt_shell(seed=3):
    """The quilt: a closed shell, pattern side outward (material 0), lining side (1), binding rim (2).
    uv0 = flat sheet coordinates (u across from the left edge, v along from the foot hem, metres); uv1 = (distance to the
    hem in metres, 0). Returns (Shell, info)."""
    side, foot = quilt_side_profile(), quilt_foot_profile()
    a_edge, y0 = X_MID - SIDE_BEND, QUILT_Y0
    Lc = Y_FOLD - y0
    sep = QUILT_T + 0.004
    r_f = (sep + ROLL) / 2
    hw = a_edge + side.length
    t_lo, t_hi = -foot.length, Lc + math.pi * r_f + BAND
    drape = Drape(a_edge, y0, Y_FOLD, side, foot)
    sup = bed_support()
    s = sample_line(-hw, hw, [(-hw, -a_edge + 0.04, 0.016), (a_edge - 0.04, hw, 0.016), (-hw, -hw + 0.06, 0.010),
                              (hw - 0.06, hw, 0.010)], 0.045)
    t = sample_line(t_lo, t_hi, [(t_lo, 0.03, 0.014), (Lc - 0.05, Lc + math.pi * r_f + 0.05, 0.010),
                                 (Lc + math.pi * r_f, t_hi - 0.06, 0.020), (t_hi - 0.06, t_hi, 0.010)], 0.045)
    S, T = np.meshgrid(s, t)
    S, T = round_corners(S, T, hw, t_lo, t_hi, (side.length, foot.length), (side.length, FAR_CORNER))
    Nt, Ns = S.shape
    li = np.array(boundary_loop(Nt, Ns))
    e = loop_distance(np.stack([S, T], axis=-1), np.stack([S[li[:, 0], li[:, 1]], T[li[:, 0], li[:, 1]]], axis=-1))
    Ye, lift = fold_map(T, Lc, y0, sep, ROLL)
    W = drape.wrap(S, Ye)
    P, nu = W["P"], W["nu"]
    # --- displacement: sag and soft ridges on the field; on the hang pleats (outward along the normal, never inward)
    # that bunch the hem up where they bulge. The pleats' phase runs along the edge (y on the sides, x on the foot),
    # the same for both layers of the fold so they stay stacked.
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    sag = (0.010 * (0.5 + 0.5 * vnoise3(x * 1.9, y * 1.5, 0.0, seed))
           + 0.004 * (0.5 + 0.5 * vnoise3(x * 5.2 + 3.0, y * 4.4 - 1.0, 0.0, seed + 1)))
    patch = smoothstep(0.5 + 1.4 * vnoise3(x * 1.5, y * 1.3, 7.0, seed + 6))
    ridge = 0.005 * patch * np.sin(TAU * (y + 0.55 * x) / 0.42 + 2.5 * vnoise3(x * 1.3, y * 1.1, 2.0, seed + 7))
    fold = (1 - W["w"]) * pleat_field(Ye, W["d"], seed) + W["w"] * pleat_field(S, W["d"], seed)
    corner = 1.0 - 2.0 * W["w"] * (1.0 - W["w"])                      # calmer where the two profiles blend
    pleat = 0.016 * W["hang"] * fold * corner * (0.7 + 0.3 * vnoise3(x * 4.0, y * 4.0, z * 2.0, seed + 8))
    flare = 0.003 * W["hang"]
    disp = W["flat"] * (sag + ridge) + pleat + flare
    mid = P + nu * (lift + disp)[..., None]
    mid[..., 2] += 0.018 * W["hang"] * (fold - 0.5)
    half = (QUILT_T_EDGE + (QUILT_T - QUILT_T_EDGE) * smoothstep(e / 0.05)) / 2
    clr = half + QUILT_GAP
    pushed = push_out(mid, clr, sup)
    moved = (np.linalg.norm(pushed - mid, axis=-1) > 0.0005) & (mid[..., 1] < -0.5)       # the foot posts shoved the cloth
    active = np.minimum(1.0, blur(dilate(moved, 4).astype(float), 3) * 1.6)
    mid = relax_clear(pushed, clr, sup, active)
    uv0 = np.stack([S + hw, T - t_lo], axis=-1)
    uv1 = np.stack([e, np.zeros_like(e)], axis=-1)
    shell = shell_from_grids(mid, half, uv0, uv1, rim_k=4)
    field = (np.abs(S) < 0.3) & (T > 0.2) & (T < Lc - 0.2)
    info = {"hw": hw, "t_lo": t_lo, "t_hi": t_hi, "Lc": Lc, "y0": y0, "y_fold": Y_FOLD, "a_edge": a_edge,
            "grid": mid.shape[:2], "field_top_z": float((mid[..., 2] + half)[field].mean()),
            "side_len": side.length, "foot_len": foot.length}
    return shell, info


# ===================================================================================================== the sheet
def sheet_shell(seed=11):
    """The fitted sheet: a thin shell over the mattress top wrapping down its four sides (SHEET_HANG), a little wrinkled.
    Material 0 outside, 1 inside, 2 hem. Returns (Shell, info)."""
    bend = MAT_R + SHEET_T * 0.5 + 0.0008
    z_top = MAT_Z1 + SHEET_T / 2
    prof = Profile([(0.0, z_top), (bend, z_top), (bend, MAT_Z1 - SHEET_HANG)], [0, bend, 0])
    a, y1 = MAT_HX - MAT_R, MAT_HY - MAT_R
    drape = Drape(a, -y1, y1, prof, prof, prof)
    L = prof.length
    hw = a + L
    t_lo, t_hi = -L, 2 * y1 + L
    s = sample_line(-hw, hw, [(-hw, -a + 0.03, 0.02), (a - 0.03, hw, 0.02)], 0.08)
    t = sample_line(t_lo, t_hi, [(t_lo, 0.03, 0.02), (t_hi - 0.03 - L, t_hi, 0.02)], 0.08)
    S, T = np.meshgrid(s, t)
    S, T = round_corners(S, T, hw, t_lo, t_hi, (L, L), (L, L))
    Nt, Ns = S.shape
    li = np.array(boundary_loop(Nt, Ns))
    e = loop_distance(np.stack([S, T], axis=-1), np.stack([S[li[:, 0], li[:, 1]], T[li[:, 0], li[:, 1]]], axis=-1))
    W = drape.wrap(S, -y1 + T)
    P, nu = W["P"], W["nu"]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    wr = (0.0010 * (0.5 + 0.5 * vnoise3(x * 5.0, y * 4.0, 0.0, seed))
          + 0.0004 * (0.5 + 0.5 * vnoise3(x * 14.0, y * 11.0, 3.0, seed + 1)))
    gather = 0.003 * W["hang"] * (0.5 + 0.5 * vnoise3(x * 24, y * 24, z * 3, seed + 2))
    half = np.full(e.shape, SHEET_T / 2 * 0.8)
    mid = P + nu * (W["flat"] * wr + gather)[..., None]
    mid = push_out(mid, half + 0.0004, Support([mattress_box()]), soft=0.002)
    uv0 = np.stack([S + hw, T - t_lo], axis=-1)
    uv1 = np.stack([e, np.zeros_like(e)], axis=-1)
    return shell_from_grids(mid, half, uv0, uv1, rim_k=4), {"top_z": float(MAT_Z1 + SHEET_T)}


# ===================================================================================================== pillows
PILLOWS = (
    # name, half width, half length, height, centre (x, y), z above the sheet's top, rotation (x, y, z) in degrees, dent
    ("pillow_a", 0.30, 0.21, 0.17, (-0.02, 0.735), 0.0, (0.0, 0.0, 2.0), 0.03),
    ("pillow_b", 0.26, 0.18, 0.15, (0.07, 0.775), 0.10, (18.0, 0.0, -5.0), 0.02),
)


def euler_matrix(deg):
    """3x3 rotation matrix of Blender's XYZ Euler angles (degrees): R = Rz @ Ry @ Rx."""
    ax, ay, az = (math.radians(a) for a in deg)
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pillow_location(spec):
    (_, _, _, _, (cx, cy), dz, _, _) = spec
    return np.array([cx, cy, SHEET_Z - 0.002 + dz])


def pillow_in_bed(spec, seed=5):
    """The pillow `spec` (a PILLOWS row): (the pillow's own Shell, its vertices in the bed's frame, rotation matrix,
    location of its object)."""
    _, hw, hl, h, _, _, rot, dent = spec
    sh = pillow_shell(hw, hl, h, seed=seed, dent=dent)
    R, loc = euler_matrix(rot), pillow_location(spec)
    return sh, sh.V @ R.T + loc, R, loc


def pillow_shell(hw, hl, height, seed=5, nu=32, nv=24, dent=0.0):
    """A plump pillow lying on a plane: half sizes hw (x) x hl (y), `height` tall, its underside near z = 0. The top and
    bottom skins meet in a seam round the outline (shared vertices: the edge shades round but pinched, the corners come to
    points). uv0 = (x + hw, y + hl) metres, uv1 = (distance to the seam in the plane, 0). Material 0 only."""
    th_u, th_v = np.linspace(-math.pi / 2, math.pi / 2, nu + 1), np.linspace(-math.pi / 2, math.pi / 2, nv + 1)
    p, q = np.meshgrid(np.sin(th_u), np.sin(th_v))
    F = (1 - p ** 2) * (1 - q ** 2)
    pull = 1.0 - 0.05 * (np.abs(p) * np.abs(q)) ** 2                   # the corners draw in a little
    x, y = p * hw * pull, q * hl * pull
    z_seam = 0.34 * height
    ripple = 0.004 * vnoise3(x * 9.0, y * 9.0, 0.0, seed) + 0.010 * vnoise3(x * 3.2, y * 3.2, 5.0, seed + 1)
    lean = 1.0 + 0.16 * q                                              # fuller toward the head end (+y)
    crease = 0.006 * np.sin(5.0 * np.arctan2(q * hl, p * hw) + 2.0 * vnoise3(x * 4.0, y * 4.0, 1.0, seed + 3))
    zt = (z_seam + (height - z_seam) * lean * F ** 0.46 + (ripple + crease * (1 - F)) * F ** 0.5
          - dent * np.exp(-((x / (0.5 * hw)) ** 2 + (y / (0.5 * hl)) ** 2)) * F)
    zb = z_seam * (1 - F ** 0.28)
    Nt, Ns = F.shape
    ids_top = np.arange(Nt * Ns).reshape(Nt, Ns)
    ids_bot = ids_top + Nt * Ns
    li = np.array(boundary_loop(Nt, Ns))
    ids_bot[li[:, 0], li[:, 1]] = ids_top[li[:, 0], li[:, 1]]           # welded seam
    faces = []
    for i in range(Nt - 1):
        for j in range(Ns - 1):
            faces.append((ids_top[i, j], ids_top[i, j + 1], ids_top[i + 1, j + 1], ids_top[i + 1, j]))
    for i in range(Nt - 1):
        for j in range(Ns - 1):
            faces.append((ids_bot[i, j], ids_bot[i + 1, j], ids_bot[i + 1, j + 1], ids_bot[i, j + 1]))
    V = np.concatenate([np.stack([x, y, zt], axis=-1).reshape(-1, 3), np.stack([x, y, zb], axis=-1).reshape(-1, 3)])
    e = np.minimum(hw - np.abs(x), hl - np.abs(y)).reshape(-1)
    uv0 = np.stack([x.reshape(-1) + hw, y.reshape(-1) + hl], axis=-1)
    uv1 = np.stack([e, np.zeros_like(e)], axis=-1)
    V, Fa, uv0, uv1 = compact(V, np.array(faces, dtype=np.int64), np.concatenate([uv0, uv0]), np.concatenate([uv1, uv1]))
    return Shell(V, Fa, np.zeros(len(Fa), dtype=np.int64), uv0, uv1)


# ===================================================================================================== the rug
def rug_shell(seed=9):
    """The rug's woven body: a closed slab RUG_W x RUG_D x RUG_T with a rounded edge (a half circle of radius RUG_T / 2
    round the outline) and softly rounded plan corners; the top is very slightly undulating. Material 0 top, 1 underside,
    2 edge. uv0 = (x + RUG_W / 2, y + RUG_D / 2) metres, uv1 = (distance to the outline in metres, 0)."""
    r = RUG_T / 2
    hw, hl = RUG_W / 2 - r, RUG_D / 2 - r
    s = sample_line(-hw, hw, [(-hw, -hw + 0.06, 0.012), (hw - 0.06, hw, 0.012)], 0.06)
    t = sample_line(-hl, hl, [(-hl, -hl + 0.06, 0.012), (hl - 0.06, hl, 0.012)], 0.06)
    S, T = np.meshgrid(s, t)
    S, T = round_corners(S, T, hw, -hl, hl, (RUG_CORNER, RUG_CORNER), (RUG_CORNER, RUG_CORNER))
    Nt, Ns = S.shape
    li = np.array(boundary_loop(Nt, Ns))
    e = loop_distance(np.stack([S, T], axis=-1), np.stack([S[li[:, 0], li[:, 1]], T[li[:, 0], li[:, 1]]], axis=-1)) + r
    e[li[:, 0], li[:, 1]] = 0.0
    mid = np.stack([S, T, np.full(S.shape, r)], axis=-1)
    uv0 = np.stack([S + RUG_W / 2, T + RUG_D / 2], axis=-1)
    uv1 = np.stack([e, np.zeros_like(e)], axis=-1)
    sh = shell_from_grids(mid, np.full(S.shape, r), uv0, uv1, rim_k=4)
    n = Nt * Ns
    x, y = sh.V[:n, 0], sh.V[:n, 1]
    inner = np.ones(n)
    inner.reshape(Nt, Ns)[li[:, 0], li[:, 1]] = 0.0                    # the boundary stays on the rim's circle
    sh.V[:n, 2] += 0.0007 * inner * vnoise3(x * 3.1, y * 2.7, 0.0, seed)
    return sh


def tube_rings(P, radii, sides=5, up=(0.0, 0.0, 1.0), phase=0.0):
    """Quad tube along the polyline P (n, 3) with radius radii (n,): rings of `sides` points (frame: `up` projected off the
    tangent), faces wound outward, open at both ends. Returns V (n * sides, 3), F ((n - 1) * sides, 4)."""
    P = np.asarray(P, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.asarray(up, float)
    N = up - (T @ up)[:, None] * T
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    a = phase + TAU * np.arange(sides) / sides
    ring = P[:, None, :] + np.asarray(radii, float)[:, None, None] * (
        np.cos(a)[None, :, None] * N[:, None, :] + np.sin(a)[None, :, None] * B[:, None, :])
    j = np.arange(sides)
    F = np.concatenate([np.stack([i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides,
                                  (i + 1) * sides + j], axis=1) for i in range(n - 1)])
    return ring.reshape(-1, 3), F[:, ::-1] if _tube_flipped(P, ring, F) else F


def _tube_flipped(P, ring, F):
    """True when the faces of a tube built by tube_rings wind inward (decided on its first face)."""
    V = ring.reshape(-1, 3)
    a, b, c, d = (V[F[0, i]] for i in range(4))
    n = np.cross(c - a, d - b)
    return float(n @ (V[F[0]].mean(axis=0) - P[0])) < 0.0


def quad_sphere(centre, radii):
    """A lump of 24 quads: the 2 x 2 x 2 subdivided cube projected onto the ellipsoid with semi-axes `radii`. Returns V
    (26, 3), F (24, 4) wound outward."""
    pts = {}
    V = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                if (ix, iy, iz) != (0, 0, 0):
                    pts[(ix, iy, iz)] = len(V)
                    V.append((ix, iy, iz))
    V = np.array(V, float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = []
    for axis in range(3):
        for sgn in (-1, 1):
            u, v = [k for k in range(3) if k != axis]
            for a in (-1, 0):
                for b in (-1, 0):
                    quad = []
                    for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        idx = [0, 0, 0]
                        idx[axis], idx[u], idx[v] = sgn, a + da, b + db
                        quad.append(pts[tuple(idx)])
                    f = np.array(quad)
                    n = np.cross(V[f[2]] - V[f[0]], V[f[3]] - V[f[1]])
                    F.append(f if n @ V[f].mean(axis=0) > 0 else f[::-1])
    return V * np.asarray(radii, float) + np.asarray(centre, float), np.array(F, dtype=np.int64)


def rug_fringe(seed=13, pitch=0.016, strands=2, segments=3, sides=8):
    """The fringe of the two short edges (x = +-RUG_W / 2): a tassel every `pitch` m along y, each a soft knot (a lumpy
    ellipsoid) and `strands` round yarn strands (8-sided tubes) of 5-5.8 cm that fan out, wobble and taper to a frayed
    tip, lying on the floor; nothing reaches past RUG_FRINGE + 3 mm. UV u = distance along the strand (metres), v = 0.
    One material (yarn). Returns (Shell, tassels per side)."""
    n = int(round((RUG_D - 0.032) / pitch)) + 1
    ys = np.linspace(-RUG_D / 2 + 0.016, RUG_D / 2 - 0.016, n)
    s_ = np.linspace(0.0, 1.0, segments + 1)
    Vs, Fs, Us = [], [], []
    base = 0
    for sgn in (-1.0, 1.0):
        for j, y0 in enumerate(ys):
            r = [float(_lattice(np.array([j * 11 + c]), np.array([int(sgn)]), np.array([c]), seed)[0]) for c in range(1, 9)]
            lump, lf = quad_sphere((sgn * (RUG_W / 2 + 0.0105 + 0.0006 * r[0]), y0 + 0.0004 * r[1], 0.0032),
                                   (0.0035 + 0.0002 * r[2], 0.0046 + 0.0003 * r[3], 0.0031))
            Vs.append(lump)
            Fs.append(lf + base)
            Us.append(np.zeros((len(lump), 2)))
            base += len(lump)
            for k in range(strands):
                side = (k - (strands - 1) / 2.0) * 2.0 if strands > 1 else 0.0
                reach = 0.0495 + 0.0085 * (0.5 + 0.5 * r[4 + k])          # to x = W / 2 + 5.0 .. 5.8 cm
                r0, r1 = 0.0021 + 0.0002 * r[5], 0.0009
                x = sgn * (RUG_W / 2 - 0.008 + (0.008 + reach) * s_)
                fan = 0.0032 * side + 0.0042 * r[6 + k % 2] * s_
                wob = 0.0034 * (0.5 + 0.5 * r[7]) * np.sin(TAU * 0.7 * s_ + math.pi * r[3]) * s_
                y = y0 + 0.0014 * side * (1 - s_) + fan * s_ + wob
                rad = r0 + (r1 - r0) * s_ ** 0.8
                P = np.stack([x, y, rad + 0.0001], axis=-1)
                V, F = tube_rings(P, rad, sides, phase=math.pi * r[2])
                Vs.append(V)
                Fs.append(F + base)
                u = np.repeat(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]), sides)
                Us.append(np.stack([u, np.zeros_like(u)], axis=-1))
                base += len(V)
    V, F, UV = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Us)
    return Shell(V, F, np.zeros(len(F), dtype=np.int64), UV, np.zeros_like(UV)), len(ys)
