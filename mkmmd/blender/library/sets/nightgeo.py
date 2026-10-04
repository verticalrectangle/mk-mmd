"""Pure-numpy geometry for the night sets (sky, highway, skyline): merged meshes, primitives, instancing, sweeps along
a path and baked lamp light. No bpy here, so everything is unit-testable; `nightkit` turns a `Mesh` into an object.

Conventions: metres, Z up. A path frame is (T, L, U): along the road, to its left, up. Instances are placed with
`frame_R(T, L, U)`: local +X runs along the road, +Y to the left, +Z up. Quad winding is counter-clockwise seen from
the outside (normals outward); `Mesh.add_grid` makes the normal e_i x e_j of its two grid axes (flip=True reverses)."""
import math

import numpy as np

# ---------------------------------------------------------------------------------------------------- geometry


class Geom:
    """Vertices V (n, 3), quads Q (m, 4), triangles T (k, 3), per-vertex uv (n, 2), `inst` (n,) instance index."""
    __slots__ = ("V", "Q", "T", "UV", "inst")

    def __init__(self, V, Q=None, T=None, UV=None, inst=None):
        self.V = np.asarray(V, float).reshape(-1, 3)
        self.Q = np.zeros((0, 4), np.int64) if Q is None else np.asarray(Q, np.int64).reshape(-1, 4)
        self.T = np.zeros((0, 3), np.int64) if T is None else np.asarray(T, np.int64).reshape(-1, 3)
        self.UV = np.zeros((len(self.V), 2)) if UV is None else np.asarray(UV, float).reshape(-1, 2)
        self.inst = np.zeros(len(self.V), np.int64) if inst is None else np.asarray(inst, np.int64)


_CUBE_FACES = np.array([
    [[+.5, -.5, -.5], [+.5, +.5, -.5], [+.5, +.5, +.5], [+.5, -.5, +.5]],      # +X
    [[-.5, +.5, -.5], [-.5, -.5, -.5], [-.5, -.5, +.5], [-.5, +.5, +.5]],      # -X
    [[+.5, +.5, -.5], [-.5, +.5, -.5], [-.5, +.5, +.5], [+.5, +.5, +.5]],      # +Y
    [[-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, -.5, +.5], [-.5, -.5, +.5]],      # -Y
    [[-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5], [-.5, +.5, +.5]],      # +Z
    [[-.5, +.5, -.5], [+.5, +.5, -.5], [+.5, -.5, -.5], [-.5, -.5, -.5]],      # -Z
]).reshape(24, 3)
_CUBE_UV = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (6, 1))
# which size axis gives the u and v extent of each face (so uv are metres: u along the face's first edge, v up)
_CUBE_UAX = np.repeat([1, 1, 0, 0, 0, 0], 4)
_CUBE_VAX = np.repeat([2, 2, 2, 2, 1, 1], 4)
FACE_SIDES = (0, 1, 2, 3)        # the four vertical faces of a box (indices into its 6 faces)


def boxes(sizes, base=False, skip=()):
    """I boxes in their own frames: V (I*24, 3) centred at the origin (base=True: bottom face at z = 0), flat-shaded
    quads, uv in metres (sides: u along the face, v up; top and bottom: x, y). Faces are ordered +X -X +Y -Y +Z -Z;
    vertex k of box i is i*24 + k; `skip` leaves faces out (indices 0..5) so a quad can take the place of one.
    `inst` holds the box index."""
    sizes = np.atleast_2d(np.asarray(sizes, float))
    if sizes.shape[1] == 1:
        sizes = np.repeat(sizes, 3, axis=1)
    I = len(sizes)
    V = _CUBE_FACES[None] * sizes[:, None, :]
    if base:
        V[:, :, 2] += sizes[:, None, 2] / 2
    uv = _CUBE_UV[None] * np.stack([sizes[:, _CUBE_UAX], sizes[:, _CUBE_VAX]], -1)
    faces = [f for f in range(6) if f not in set(skip)]
    Q = (np.arange(I)[:, None, None] * 24 + np.arange(24).reshape(6, 4)[faces][None]).reshape(-1, 4)
    return Geom(V.reshape(-1, 3), Q, None, uv.reshape(-1, 2), np.repeat(np.arange(I), 24))


def box(size=(1.0, 1.0, 1.0), base=False):
    return boxes([size], base)


def cylinder(r0=0.5, r1=None, h=1.0, n=8, caps=(True, True), base=True, twist=0.0):
    """A (tapered) cylinder along +Z: bottom radius r0, top radius r1 (default r0). Side quads share vertices
    (smooth shading); caps are triangle fans with their own vertices. uv: u = round the girth in metres, v = z."""
    r1 = r0 if r1 is None else r1
    a = 2 * math.pi * np.arange(n) / n + twist
    c, s = np.cos(a), np.sin(a)
    z0, z1 = (0.0, h) if base else (-h / 2, h / 2)
    V = [np.stack([r0 * c, r0 * s, np.full(n, z0)], 1), np.stack([r1 * c, r1 * s, np.full(n, z1)], 1)]
    girth = 2 * math.pi * max(r0, r1)
    u = girth * np.arange(n) / n
    UV = [np.stack([u, np.full(n, 0.0)], 1), np.stack([u, np.full(n, h)], 1)]
    i = np.arange(n)
    Q = np.stack([i, (i + 1) % n, n + (i + 1) % n, n + i], 1)
    T = []
    off = 2 * n
    if r1 <= 1e-9:                                   # a cone: the sides are triangles meeting at one apex
        V[1] = np.array([[0.0, 0.0, z1]])
        UV[1] = np.array([[0.0, h]])
        Q, T = None, [np.stack([i, (i + 1) % n, np.full(n, n)], 1)]
        off = n + 1
    for top, r, z in ((False, r0, z0), (True, r1, z1)):
        if not caps[1 if top else 0] or r <= 0:
            continue
        ring = np.stack([r * c, r * s, np.full(n, z)], 1)
        V += [ring, np.array([[0.0, 0.0, z]])]
        UV += [ring[:, :2], np.zeros((1, 2))]
        k = np.arange(n)
        ctr = off + n
        T.append(np.stack([off + k, off + (k + 1) % n, np.full(n, ctr)], 1) if top else
                 np.stack([off + (k + 1) % n, off + k, np.full(n, ctr)], 1))
        off += n + 1
    return Geom(np.vstack(V), Q, np.vstack(T) if T else None, np.vstack(UV))


def sphere(r=1.0, nu=10, nv=6):
    """A UV sphere (nv rings between the poles), outward normals, smooth-shading friendly."""
    V = [[0.0, 0.0, -r]]
    for j in range(1, nv):
        phi = -math.pi / 2 + math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            V.append([r * math.cos(phi) * math.cos(th), r * math.cos(phi) * math.sin(th), r * math.sin(phi)])
    V.append([0.0, 0.0, r])
    V = np.array(V)
    top = len(V) - 1
    T, Q = [], []
    for i in range(nu):                                                  # south cap
        T.append([0, 1 + (i + 1) % nu, 1 + i])
    for j in range(nv - 2):
        a, b = 1 + j * nu, 1 + (j + 1) * nu
        for i in range(nu):
            Q.append([a + i, a + (i + 1) % nu, b + (i + 1) % nu, b + i])
    last = 1 + (nv - 2) * nu
    for i in range(nu):                                                  # north cap
        T.append([last + i, last + (i + 1) % nu, top])
    return Geom(V, np.array(Q, np.int64).reshape(-1, 4), np.array(T, np.int64), np.zeros((len(V), 2)))


def quad(w, h, plane="XZ", base=False, flip=False):
    """One quad of w x h in the XZ (normal -Y, default) / YZ (normal +X) / XY (normal +Z) plane, centred (XZ and YZ:
    base=True puts the bottom edge at z = 0). flip=True faces the other way and keeps the uv reading left to right
    for a viewer in front. uv in metres: (0, 0) bottom left, (w, h) top right."""
    hw, hh = w / 2, h / 2
    z0, z1 = (0.0, h) if base else (-hh, hh)
    if plane == "XZ":
        V = [[-hw, 0, z0], [hw, 0, z0], [hw, 0, z1], [-hw, 0, z1]]
    elif plane == "YZ":
        V = [[0, -hw, z0], [0, hw, z0], [0, hw, z1], [0, -hw, z1]]
    else:
        V = [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]]
    V = np.array(V, float)
    if flip:
        V[:, 1 if plane == "YZ" else 0] *= -1.0
    return Geom(V, [[0, 1, 2, 3]], None, [[0, 0], [w, 0], [w, h], [0, h]])


def rot_z(yaw):
    """(I, 3, 3) rotations about Z by yaw (radians)."""
    yaw = np.atleast_1d(np.asarray(yaw, float))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(yaw), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    return R


def frame_R(T, L, U):
    """(I, 3, 3) rotations whose columns are T, L, U: local +X along the road, +Y left, +Z up."""
    return np.stack([np.asarray(T, float), np.asarray(L, float), np.asarray(U, float)], axis=-1)


def frame_from_z(d, up=(0.0, 0.0, 1.0)):
    """(I, 3, 3) rotations taking local +Z to the unit vectors d (local X stays as horizontal as possible)."""
    d = np.atleast_2d(np.asarray(d, float))
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    up = np.broadcast_to(np.asarray(up, float), d.shape).copy()
    near = np.abs(np.einsum("ij,ij->i", d, up)) > 0.999
    up[near] = (1.0, 0.0, 0.0)
    x = np.cross(up, d)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.cross(d, x)
    return np.stack([x, y, d], axis=-1)


def instance(g, R=None, t=None, scale=None):
    """Copies of `g`, one per row of R (I, 3, 3) and/or t (I, 3): V' = R (scale * V) + t. scale (I,) or (I, 3)."""
    n_inst = 1
    for a in (R, t, scale):
        if a is not None:
            n_inst = max(n_inst, len(a))
    V = np.broadcast_to(g.V, (n_inst,) + g.V.shape).copy()
    if scale is not None:
        sc = np.asarray(scale, float)
        V = V * (sc[:, None, None] if sc.ndim == 1 else sc[:, None, :])
    if R is not None:
        V = np.einsum("ikj,inj->ink", np.broadcast_to(R, (n_inst, 3, 3)), V)
    if t is not None:
        V = V + np.broadcast_to(np.asarray(t, float), (n_inst, 3))[:, None, :]
    n = len(g.V)
    off = (np.arange(n_inst) * n)[:, None, None]
    Q = (g.Q[None] + off).reshape(-1, 4) if len(g.Q) else None
    T = (g.T[None] + off).reshape(-1, 3) if len(g.T) else None
    return Geom(V.reshape(-1, 3), Q, T, np.tile(g.UV, (n_inst, 1)), np.repeat(np.arange(n_inst), n))


def place(g, R=None, t=None):
    """Move the i-th of `len(R)` equal blocks of `g` (e.g. the boxes of `boxes(sizes)`) with R[i] (3, 3) and t[i]."""
    n_blk = len(R) if R is not None else len(t)
    V = g.V.reshape(n_blk, -1, 3)
    if R is not None:
        V = np.einsum("ikj,inj->ink", np.asarray(R, float), V)
    if t is not None:
        V = V + np.asarray(t, float)[:, None, :]
    return Geom(V.reshape(-1, 3), g.Q, g.T, g.UV, g.inst)


def bars(p0, p1, w, h, up=(0.0, 0.0, 1.0)):
    """Rectangular bars from p0 to p1 (I, 3) of section w x h (h along `up`-ish, w across)."""
    p0, p1 = np.atleast_2d(np.asarray(p0, float)), np.atleast_2d(np.asarray(p1, float))
    d = p1 - p0
    ln = np.linalg.norm(d, axis=1)
    R = frame_from_z(d, up)
    sizes = np.stack([np.broadcast_to(w, ln.shape), np.broadcast_to(h, ln.shape), ln], 1)
    return place(boxes(sizes, base=True), R, p0)


# ---------------------------------------------------------------------------------------------------- merged mesh


class Mesh:
    """Accumulates pieces into one polygon list: `add(geom, mat, smooth, **attrs)`, `add_grid(V, ...)`. Attributes are
    per-vertex (n,) or (n, c) arrays by name, zero where a piece has none. `arrays()` returns what `nightkit` needs."""

    def __init__(self):
        self._V, self._UV, self._Q, self._T = [], [], [], []
        self._QM, self._TM, self._QS, self._TS = [], [], [], []
        self._attr = {}
        self.n = 0

    def __len__(self):
        return self.n

    def _push(self, V, UV, Q, T, qm, tm, smooth, attrs):
        n = len(V)
        self._V.append(V)
        self._UV.append(UV)
        if len(Q):
            self._Q.append(Q + self.n)
            self._QM.append(np.broadcast_to(np.asarray(qm), (len(Q),)).astype(np.int32))
            self._QS.append(np.full(len(Q), bool(smooth)))
        if len(T):
            self._T.append(T + self.n)
            self._TM.append(np.broadcast_to(np.asarray(tm), (len(T),)).astype(np.int32))
            self._TS.append(np.full(len(T), bool(smooth)))
        for k, v in attrs.items():
            v = np.asarray(v, float)
            if v.ndim == 0:
                v = np.full(n, float(v))
            elif v.shape[0] != n:
                raise ValueError(f"attribute {k!r}: {v.shape[0]} values for {n} vertices")
            self._attr.setdefault(k, []).append((self.n, v))
        self.n += n

    def add(self, g, mat=0, smooth=False, **attrs):
        """A Geom. `mat` is one material index or one per quad then triangle (when both exist, a pair (mq, mt))."""
        if not len(g.V):
            return self
        qm, tm = mat if isinstance(mat, tuple) else (mat, mat)
        self._push(g.V, g.UV, g.Q, g.T, qm, tm, smooth, attrs)
        return self

    def add_grid(self, V, mat=0, uv=None, flip=False, smooth=True, **attrs):
        """A (A, B, 3) grid of vertices as (A-1)*(B-1) quads (i, j) (i+1, j) (i+1, j+1) (i, j+1): normal e_i x e_j.
        mat: int or (A-1, B-1) indices. uv (A, B, 2). Per-vertex attrs are (A, B[, c]) arrays."""
        V = np.asarray(V, float)
        A, B = V.shape[:2]
        idx = np.arange(A * B).reshape(A, B)
        Q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        if flip:
            Q = Q[:, ::-1]
        UV = np.zeros((A * B, 2)) if uv is None else np.asarray(uv, float).reshape(-1, 2)
        m = np.asarray(mat)
        qm = m.reshape(-1) if m.ndim == 2 else m
        flat = {k: np.asarray(v, float).reshape(A * B, *np.shape(v)[2:]) if np.ndim(v) else v
                for k, v in attrs.items()}
        self._push(V.reshape(-1, 3), UV, Q, np.zeros((0, 3), np.int64), qm, 0, smooth, flat)
        return self

    def arrays(self):
        """dict: V (n,3), Q (m,4), T (k,3), UV (n,2), QM, TM (material per face), QS, TS (smooth flags), attrs {name:
        (n,) or (n,c) float32}."""
        def cat(parts, shape):
            return np.concatenate(parts) if parts else np.zeros(shape)
        out = {"V": cat(self._V, (0, 3)), "UV": cat(self._UV, (0, 2)),
               "Q": cat(self._Q, (0, 4)).astype(np.int64), "T": cat(self._T, (0, 3)).astype(np.int64),
               "QM": cat(self._QM, (0,)).astype(np.int32), "TM": cat(self._TM, (0,)).astype(np.int32),
               "QS": cat(self._QS, (0,)).astype(bool), "TS": cat(self._TS, (0,)).astype(bool)}
        attrs = {}
        for k, pieces in self._attr.items():
            c = max((p.shape[1] if p.ndim > 1 else 1) for _, p in pieces)
            a = np.zeros((self.n, c), np.float32)
            for off, p in pieces:
                a[off:off + len(p)] = p.reshape(len(p), -1) if p.ndim > 1 else p[:, None]
            attrs[k] = a[:, 0] if c == 1 else a
        out["attrs"] = attrs
        return out


# ---------------------------------------------------------------------------------------------------- sampling


def stations(length, ds, extra=(), start=0.0, min_gap=0.05):
    """Sorted arc lengths from `start` to `length` at most `ds` apart, plus every value in `extra` (clipped)."""
    n = max(2, int(math.ceil((length - start) / ds)) + 1)
    s = np.concatenate([np.linspace(start, length, n), np.clip(np.asarray(list(extra), float), start, length)])
    s = np.unique(np.round(s, 4))
    keep = np.concatenate([[True], np.diff(s) > min_gap])
    return s[keep]


def dash_edges(s0, s1, on, period, phase=0.0):
    """Start and end arc lengths of every dash (length `on`, repeating each `period`, shifted by `phase`) in [s0, s1]."""
    k0 = int(math.floor((s0 - phase) / period)) - 1
    k1 = int(math.ceil((s1 - phase) / period)) + 1
    starts = phase + period * np.arange(k0, k1 + 1)
    return starts, starts + on


def in_dash(s_mid, on, period, phase=0.0):
    """True where the arc length (array) falls inside a dash."""
    return np.mod(np.asarray(s_mid, float) - phase, period) < on


def in_ranges(s, ranges, margin=0.0):
    """Boolean mask: s inside any (a, b) of `ranges`, each grown by `margin`."""
    s = np.asarray(s, float)
    m = np.zeros(s.shape, bool)
    for a, b in ranges:
        m |= (s >= a - margin) & (s <= b + margin)
    return m


# ---------------------------------------------------------------------------------------------------- lamp light


def spot_light(points, normals, lamps, aim, power, cos_half=math.cos(math.radians(70)), blend=0.8, reach=None,
               chunk=1024):
    """Light a set of points would get from spot lamps, in the scale a Blender spot light of `power` watts gives a
    white diffuse surface: sum of P/(4 pi^2) * spotmask * max(0, n.l) / d^2 (the render adds albedo).
    points (n, 3), normals (n, 3) or None (facing the lamp), lamps (m, 3), aim (m, 3) unit axes (or one), power (m,)
    or scalar. `reach` ignores lamps farther than that (m) and makes long roads cheap: points are taken in chunks
    along their longest axis and each chunk only meets the lamps near its bounding box. cos_half = -1, blend = 0 is an
    omnidirectional lamp. Returns (n,)."""
    points = np.asarray(points, float).reshape(-1, 3)
    lamps = np.asarray(lamps, float).reshape(-1, 3)
    out = np.zeros(len(points))
    if not len(lamps) or not len(points):
        return out
    aim = np.broadcast_to(np.asarray(aim, float), lamps.shape)
    power = np.broadcast_to(np.asarray(power, float), (len(lamps),))
    nrm = None if normals is None else np.broadcast_to(np.asarray(normals, float), points.shape)
    bl = max((1.0 - cos_half) * blend, 1e-6)
    axis = 0 if np.ptp(points[:, 0]) >= np.ptp(points[:, 1]) else 1
    order = np.argsort(points[:, axis], kind="stable")
    for a in range(0, len(points), chunk):
        idx = order[a:a + chunk]
        p = points[idx]
        sel = slice(None)
        if reach is not None:
            near = np.all((lamps >= p.min(0) - reach) & (lamps <= p.max(0) + reach), axis=1)
            if not near.any():
                continue
            sel = near
        L, A, P = lamps[sel], aim[sel], power[sel]
        d = L[None, :, :] - p[:, None, :]                                   # point -> lamp
        r2 = np.maximum(np.einsum("nmk,nmk->nm", d, d), 0.04)
        r = np.sqrt(r2)
        l = d / r[..., None]
        z = -np.einsum("nmk,mk->nm", l, A)                                  # cos of the angle off the lamp axis
        t = np.clip((z - cos_half) / bl, 0.0, 1.0)
        mask = t * t * (3 - 2 * t)
        ndl = 1.0 if nrm is None else np.maximum(np.einsum("nmk,nk->nm", l, nrm[idx]), 0.0)
        e = P[None, :] / (4 * math.pi ** 2) * mask * ndl / r2
        if reach is not None:
            e = np.where(r < reach, e, 0.0)
        out[idx] = e.sum(1)
    return out


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------------------------------- distribution


def city_layout(n, seed, arc_deg, distance, depth, width=(30.0, 90.0), height=(40.0, 220.0), core=0.5, az0=0.0,
                band=False, band_len=None):
    """Place n buildings for a skyline band: arrays centre (n, 3: x, y, 0), yaw (n,), size (n, 3: w, d, h), rank (n,)
    0..1 (taller and nearer the middle when `core` > 0). arc: spread over `arc_deg` around heading `az0` (degrees, 0 =
    +X, counter-clockwise) at `distance` m +- depth/2; band: along a straight line `band_len` m long, perpendicular
    to the heading, at `distance`. Seeded."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.uniform(0.1, 0.9, n)) / n - 0.5                   # -0.5 .. 0.5 along the band
    r = distance + rng.uniform(-0.5, 0.5, n) * depth
    a0 = math.radians(az0)
    if band:
        L = band_len if band_len else arc_deg
        d = np.array([math.cos(a0), math.sin(a0)])
        side = np.array([-math.sin(a0), math.cos(a0)])
        xy = d[None, :] * r[:, None] + side[None, :] * (u * L)[:, None]
        facing = np.full(n, a0 + math.pi)
    else:
        a = a0 + u * math.radians(arc_deg)
        xy = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        facing = a + math.pi
    mid = 1.0 - np.abs(2 * u) ** 1.5                                          # 1 in the middle, 0 at the ends
    w = rng.uniform(width[0], width[1], n)
    dep = rng.uniform(width[0], width[1], n)
    tall = rng.random(n) ** 2.2
    tall = (1 - core) * tall + core * np.clip(tall * (0.35 + 0.9 * mid) + 0.1 * mid, 0, 1)
    h = height[0] + (height[1] - height[0]) * np.clip(tall, 0, 1)
    return {"center": np.column_stack([xy, np.zeros(n)]), "yaw": facing, "size": np.column_stack([w, dep, h]),
            "rank": np.clip(tall, 0, 1)}
