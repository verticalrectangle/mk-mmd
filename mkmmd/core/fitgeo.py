"""fitgeo: screen type kept as shapes and the way they move (numpy only). The bake (build/bake_lit.py) evaluates every
word of screen type on each frame it shows; frame to frame, almost all of a word's change is a transform of each letter
(and of each copy of a letter: an outline is the letters grown by copies on a ring) of a fixed shape. So a word is held
as geometries, each the prototype shapes of its islands, every island one of them placed by a map, and per frame either
one map more for the whole word or a map per island, with the values its islands' attributes take. mkr's decoder is
mkr/mkr/src/screen_geo.rs; `decode` here is the reference it follows. The record format is bake_lit.py's (SCREEN).

Islands: the connected triangles of one material slot of a frame's drawn mesh, joined where they share one of Blender's
vertices (a vertex split for its corner data still joins them), numbered by their first triangle. Materialised order: the
islands' vertices island by island (each island's in drawn order), their triangles island by island (so grouped by slot).

Prototypes: islands with the same vertex count, triangles (each island's own vertex numbers) and per-vertex attributes,
whose base positions one map takes onto each other within TOL (derived normals within NORMAL_TOL), share one prototype:
the first of them as the base frame has it.

A map is a 4x4 M (row-major, glTF axes) on q = (x, y, z, 1): x' = (M0 . q) / (M3 . q), y' = M1 . q, z' = (M2 . q) /
(M3 . q); x and z are screen type's plane (Blender's x and -y), y its depth (Blender's z), which a tilt seen in
perspective keeps; an affine map has M3 = (0, 0, 0, 1). A bent map adds Q (3x3, row-major) . (u^2, u v, v^2) to (x', y',
z'), u = x - ox and v = z - oz, (ox, oy, oz) the island's prototype's first vertex: a letter that turns or moves before
the word is bent round a ring moves by no affine map, but by one with quadratic terms to 1e-5 of a frame height. A
frame's normals are the prototype's turned by the cofactor of the map's Jacobian at each vertex (det(J) J^-T: Blender's
normals follow the winding), then normalised.

Fitting a frame of the geometry's topology, in float64: one map of the whole word from the last islands pose's decoded
positions (affine, then projective), else a map per island from its prototype (affine; bent where that misses; then
projective); every map is rounded to float32 and decoded as mkr decodes it before its error is measured. A frame whose
islands miss TOL anywhere, or whose decoded normals miss Blender's by more than NORMAL_TOL (but on sliver triangles,
whose normals are float32 noise), or whose per-vertex attributes changed, or whose island attributes are not one value
per island, starts a new geometry (the same topology, this frame its base).

    seg = Segment(P, N, attrs, tris, slots, src)     drawn arrays of the base frame (glTF world positions, normals,
                                                     {key: (n, comp) float32}, triangles grouped by slot, slots
                                                     [[slot, first, count]], Blender's vertex of each drawn vertex)
    pose = seg.pose(P, N, attrs)                     a later frame of its topology: Pose or None (start a new one)
    word = Word(n, put)                              a word's frames: word.add(k, P, N, attrs, topo=(tris, slots, src))
                                                     (topo None: the last frame's topology); word.record(base) its
                                                     by_frame record, offsets moved by `base`; arrays go to put(a, dt)
    decode(lit, by_frame, k)                         a frame from lit.bin (uint8 array): positions, normals, attrs,
                                                     triangles and slots in materialised order (None: hidden)
"""
import numpy as np

TOL = 1e-5            # world units (screen type: frame heights; 0.0108 px at 1080 lines, an edge pixel off by about two
                      # levels at most): 200 times the float32 rounding of Blender's own evaluated positions (5e-8)
NORMAL_TOL = 0.02     # rad (1.1 deg): derived normals further from Blender's start a new geometry; 12 times the most
                      # Blender's own normals of well-formed triangles stray off a flat word's plane (1.7e-3, float32)
SLIVER = 256 * float(np.finfo(np.float32).eps)   # a triangle whose height is under this share of the word's extent:
                      # its normal is float32 noise (any direction), so a vertex only slivers use is not held to it
RCOND = 1e-10         # of a normal matrix: an island thinner than 1e-5 of its size is flat (its depth column ignored)
PROBE = 4096          # points of a whole word above which a projective fit is first tried on a sample
ISLANDS, WORD, BENT = 0, 1, 2   # pose kinds: a map per island, one of the whole word after another pose, bent maps
DT = {"f32": "<f4", "u32": "<u4", "u8": "u1"}


# ================================================================================================= islands and layout
def _components(edges, n):
    """Connected components of `n` nodes joined by `edges` (m, 2): a label per node, one per component."""
    lab = np.arange(n)
    e0, e1 = edges[:, 0], edges[:, 1]
    while len(edges):
        a, b = lab[e0], lab[e1]
        apart = a != b
        if not apart.any():
            break
        np.minimum.at(lab, np.maximum(a[apart], b[apart]), np.minimum(a[apart], b[apart]))   # hook roots under roots
        while True:
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt
    return lab


def _slot_of(slots, nt):
    out = np.empty(nt, np.int64)
    for s, a, c in slots:
        out[a:a + c] = s
    return out


class Layout:
    """The islands of a drawn mesh and its materialised order (module docstring): `count` islands, `src` (each
    materialised vertex's drawn vertex), `isl` (each materialised vertex's island), `v0` / `t0` (each island's first
    vertex and triangle, count + 1), `tris` (materialised triangles, materialised vertex numbers), `slot` (per island),
    `slots` [[slot, first, count]] of the materialised triangles."""

    def __init__(self, tris, slots, src):
        tris = np.asarray(tris, np.int64).reshape(-1, 3)
        src = np.asarray(src, np.int64)
        nt, nv = len(tris), len(src)
        tri_slot = _slot_of(slots, nt)
        node = tri_slot[:, None] * (int(src.max()) + 1) + src[tris]       # (slot, Blender vertex) of each corner
        u, inv = np.unique(node.ravel(), return_inverse=True)
        inv = inv.reshape(nt, 3)
        lab = _components(np.concatenate([inv[:, :2], inv[:, 1:]]), len(u))[inv[:, 0]]
        _, first, isl = np.unique(lab, return_index=True, return_inverse=True)
        rank = np.empty(len(first), np.int64)
        rank[np.argsort(first)] = np.arange(len(first))
        isl = rank[isl.ravel()]                                             # islands by first triangle
        n = len(first)
        pairs, corner = np.unique((isl[:, None] * nv + tris).ravel(), return_inverse=True)
        self.count = n
        self.isl = pairs // nv
        self.src = pairs % nv
        self.v0 = np.searchsorted(self.isl, np.arange(n + 1))
        order = np.argsort(isl, kind="stable")
        self.tris = corner.reshape(nt, 3)[order]
        t_isl = isl[order]
        self.t0 = np.searchsorted(t_isl, np.arange(n + 1))
        self.slot = tri_slot[order][self.t0[:-1]]
        self.slots = []
        for i in range(n):
            s, c = int(self.slot[i]), int(self.t0[i + 1] - self.t0[i])
            if self.slots and self.slots[-1][0] == s:
                self.slots[-1][2] += c
            else:
                self.slots.append([s, int(self.t0[i]), c])


# ================================================================================================= maps
def _f32(m):
    return None if m is None else np.asarray(m, np.float32).astype(np.float64)


def _cof(A):
    """Cofactor matrices det(A) A^-T of (..., 3, 3)."""
    return np.stack([np.cross(A[..., 1, :], A[..., 2, :]), np.cross(A[..., 2, :], A[..., 0, :]),
                     np.cross(A[..., 0, :], A[..., 1, :])], -2)


def _is_affine(M):
    return bool(np.all(M[:, 3, :3] == 0.0) and np.all(M[:, 3, 3] == 1.0))


def _take(A, grp):
    """A[grp] of per-group arrays, or A[0] broadcast when `grp` is None."""
    return A[0][None] if grp is None else np.take(A, grp, 0)


def _h(M, q, grp):
    """Homogeneous images (v, 4) of points q under the maps M (g, 4, 4), point v under M[grp[v]] (all under M[0] when
    `grp` is None)."""
    if grp is None:
        return q @ M[0, :, :3].T + M[0, :, 3]
    return (np.take(M[:, :, 0], grp, 0) * q[:, 0:1] + np.take(M[:, :, 1], grp, 0) * q[:, 1:2]
            + np.take(M[:, :, 2], grp, 0) * q[:, 2:3] + np.take(M[:, :, 3], grp, 0))


def _uv(q, grp, O):
    o = _take(O, grp)
    return q[:, 0] - o[:, 0], q[:, 2] - o[:, 2]


def apply(M, q, grp=None, Q=None, O=None):
    """Points q (v, 3) mapped as the decoder maps them: point v by map grp[v] of the maps M (g, 4, 4) (map 0 when `grp` is
    None), plus Q (g, 3, 3) . (u^2, u v, v^2) about the origins O (g, 3) when Q is given."""
    h = _h(M, q, grp)
    out = np.stack([h[:, 0] / h[:, 3], h[:, 1], h[:, 2] / h[:, 3]], 1)
    if Q is not None:
        u, v = _uv(q, grp, O)
        m = np.stack([u * u, u * v, v * v], 1)
        out += np.einsum("...ij,...j->...i", _take(Q, grp), m)
    return out


def _cross(a, b):
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], 1)


def turn(M, q, n, grp=None, Q=None, O=None):
    """Normals n (v, 3) at points q turned by the maps (as `apply`): the cofactor of the map's Jacobian at the point,
    unnormalised. The cofactor of J with columns c0, c1, c2 has columns c1 x c2, c2 x c0, c0 x c1."""
    if Q is None and _is_affine(M):
        C = _cof(M[:, :3, :3])
        if grp is None:
            return n @ C[0].T
        return (np.take(C[:, :, 0], grp, 0) * n[:, 0:1] + np.take(C[:, :, 1], grp, 0) * n[:, 1:2]
                + np.take(C[:, :, 2], grp, 0) * n[:, 2:3])
    h = _h(M, q, grp)
    w = h[:, 3]
    x, z = h[:, 0] / w, h[:, 2] / w
    m3 = _take(M[:, 3, :3], grp)
    cols = []
    for c in range(3):
        mc = _take(M[:, :3, c], grp)                                    # column c of the rows x', y', z'
        cols.append(np.stack([(mc[:, 0] - x * m3[:, c]) / w, np.broadcast_to(mc[:, 1], w.shape),
                              (mc[:, 2] - z * m3[:, c]) / w], 1))
    if Q is not None:
        u, v = _uv(q, grp, O)
        Qv = _take(Q, grp)
        cols[0] = cols[0] + Qv[:, :, 0] * (2.0 * u)[:, None] + Qv[:, :, 1] * v[:, None]
        cols[2] = cols[2] + Qv[:, :, 1] * u[:, None] + Qv[:, :, 2] * (2.0 * v)[:, None]
    c0, c1, c2 = cols
    return n[:, 0:1] * _cross(c1, c2) + n[:, 1:2] * _cross(c2, c0) + n[:, 2:3] * _cross(c0, c1)


def unit(n):
    return n / np.maximum(np.sqrt(np.einsum("vi,vi->v", n, n))[:, None], 1e-30)


def _dist(a, b):
    d = a - b
    return np.sqrt(np.einsum("vi,vi->v", d, d))


def _worst(err, v0):
    """Per group the largest of err (v,)."""
    return np.maximum.reduceat(err, v0[:-1])


class _Fit:
    """Least-squares maps of groups of points q (v, 3; group g holds rows v0[g]:v0[g + 1]) onto targets, vectorised
    over the groups, in coordinates centred and scaled per group."""

    def __init__(self, q, v0):
        self.q, self.v0 = q, v0
        cnt = np.diff(v0)
        self.grp = np.repeat(np.arange(len(cnt)), cnt)
        self.c = np.add.reduceat(q, v0[:-1]) / cnt[:, None]
        d = q - self.c[self.grp]
        s = np.sqrt(np.add.reduceat((d * d).sum(1), v0[:-1]) / cnt)
        self.s = np.where(s > 0.0, s, 1.0)
        self.x = np.concatenate([d / self.s[self.grp, None], np.ones((len(q), 1))], 1)
        self.inv = np.linalg.pinv(self._sum(self.x, self.x), rcond=RCOND, hermitian=True)
        self.bent = None                        # (origins, features, their inverse normal matrices), made when needed

    def _sum(self, F, P):
        """Per group sums of the per-point outer products F_v P_v^T: (g, F's columns, P's columns)."""
        if len(self.v0) == 2:
            return (F.T @ P)[None]
        return np.add.reduceat(F[:, :, None] * P[:, None, :], self.v0[:-1])

    def _raw(self, H, g):
        """Maps H (k, 4, 4) on the scaled coordinates of groups g -> maps on q itself."""
        T = np.zeros((len(g), 4, 4))
        T[:, :3, :3] = np.eye(3) / self.s[g, None, None]
        T[:, :3, 3] = -self.c[g] / self.s[g, None]
        T[:, 3, 3] = 1.0
        return H @ T

    def affine(self, P):
        """Affine maps of every group onto P (v, 3)."""
        xtp = self._sum(self.x, P)
        sol = self.inv @ xtp                                                   # (g, 4, 3)
        H = np.zeros((len(sol), 4, 4))
        H[:, :3, :] = np.swapaxes(sol, 1, 2)
        H[:, 3, 3] = 1.0
        return self._raw(H, np.arange(len(sol)))

    def quadratic(self, P, O):
        """Bent maps of every group onto P (module docstring): (M affine (g, 4, 4), Q (g, 3, 3)), the quadratic terms
        about the origins O (g, 3), fitted on the group's own scale."""
        if self.bent is None or self.bent[0] is not O:
            s = self.s[self.grp, None]
            d = (self.q - O[self.grp]) / s
            F = np.concatenate([d, np.ones((len(d), 1)), d[:, 0:1] * d[:, 0:1], d[:, 0:1] * d[:, 2:3],
                                d[:, 2:3] * d[:, 2:3]], 1)
            self.bent = (O, F, np.linalg.pinv(self._sum(F, F), rcond=1e-14, hermitian=True))
        _, F, inv = self.bent
        a = inv @ self._sum(F, P)                                              # (g, 7, 3)
        s = self.s[:, None]
        M = np.zeros((len(a), 4, 4))
        lin = np.swapaxes(a[:, 0:3, :], 1, 2) / s[:, :, None]                  # (g, 3 rows, 3 columns)
        M[:, :3, :3] = lin
        M[:, :3, 3] = a[:, 3, :] - np.einsum("grc,gc->gr", lin, O)
        M[:, 3, 3] = 1.0
        Q = np.swapaxes(a[:, 4:7, :], 1, 2) / (s * s)[:, :, None]
        return M, Q

    def projective(self, P, groups, A):
        """Maps of the groups `groups` onto P (v, 3), x and z projective, y affine (module docstring): the algebraic
        fit in scaled coordinates, then once reweighted toward the geometric one; y is the affine maps' `A` (g, 4, 4)."""
        H = np.zeros((len(groups), 4, 4))
        for j, g in enumerate(groups):
            x, p = self.x[self.v0[g]:self.v0[g + 1]], P[self.v0[g]:self.v0[g + 1], [0, 2]]
            co = p.mean(0)
            e = p - co
            so = float(np.sqrt((e * e).sum(1).mean())) or 1.0
            X, Z = e[:, 0] / so, e[:, 1] / so
            r1 = np.zeros((len(x), 11))
            r2 = np.zeros((len(x), 11))
            r1[:, 0:4], r1[:, 8:11] = x, -X[:, None] * x[:, :3]
            r2[:, 4:8], r2[:, 8:11] = x, -Z[:, None] * x[:, :3]
            w = np.ones(len(x))
            for _ in range(2):
                ww = 1.0 / (w * w)
                N = (r1 * ww[:, None]).T @ r1 + (r2 * ww[:, None]).T @ r2
                u = np.linalg.pinv(N, rcond=RCOND, hermitian=True) @ ((r1.T @ (X * ww)) + (r2.T @ (Z * ww)))
                m3 = np.r_[u[8:11], 1.0]
                w = x @ m3
            H[j, 0] = so * u[0:4] + co[0] * m3
            H[j, 2] = so * u[4:8] + co[1] * m3
            H[j, 3] = m3
        M = self._raw(H, np.asarray(groups))
        M[:, 1] = A[groups, 1]
        return M


# ================================================================================================= one geometry
class Pose:
    """A frame of a Segment: `kind` ISLANDS or BENT (maps (islands, 4, 4) from the prototypes, a BENT one with Q
    (islands, 3, 3)) or WORD (maps (1, 4, 4) after the islands pose of frame `ref`), `attrs` {key: (islands, comp)} or
    None (the reference's), `err` its worst error, `projective` how many of its maps are."""

    def __init__(self, kind, maps, attrs, err, projective, ref=-1, Q=None):
        self.kind, self.maps, self.attrs, self.err, self.projective, self.ref, self.Q = (kind, maps, attrs, err,
                                                                                       projective, ref, Q)
        self.decoded = None             # (positions, turned normals) of an islands pose, as its fit decoded them

    def block(self, keys):
        """The pose as lit.bin holds it (bake_lit.py's docstring: SCREEN)."""
        head = np.array([self.kind, self.attrs is not None, max(self.ref, 0), 0], "<u4").tobytes()
        body = np.asarray(self.maps, "<f4").tobytes()
        if self.kind == BENT:
            body += np.asarray(self.Q, "<f4").tobytes()
        if self.attrs is not None:
            body += b"".join(np.asarray(self.attrs[key], "<f4").tobytes() for key in keys)
        return head + body


class Segment:
    """One geometry of a word (module docstring), made from its base frame's drawn arrays."""

    def __init__(self, P, N, attrs, tris, slots, src, tol=TOL):
        self.tol = tol
        self.topo = (tris, slots, src)
        L = self.layout = Layout(tris, slots, src)
        n, v0 = L.count, L.v0
        Pm = np.asarray(P, np.float64)[L.src]
        Nm = np.asarray(N, np.float64)[L.src].copy()
        # vertices only sliver triangles use: their normals are float32 noise; they take their island's
        a, b, c = Pm[L.tris[:, 0]], Pm[L.tris[:, 1]], Pm[L.tris[:, 2]]
        area2 = np.sqrt(np.einsum("ti,ti->t", np.cross(b - a, c - a), np.cross(b - a, c - a)))
        edge = np.maximum(np.maximum(_dist(b, a), _dist(c, a)), _dist(c, b))
        sliver = area2 < SLIVER * max(float(np.abs(Pm).max()), 1e-6) * np.maximum(edge, 1e-30)
        self.solid = np.zeros(len(Pm), bool)
        self.solid[L.tris[~sliver].ravel()] = True
        if not self.solid.all():
            mean = unit(np.add.reduceat(Nm * self.solid[:, None], v0[:-1]))
            fix = ~self.solid & (np.einsum("vi,vi->v", mean[L.isl], mean[L.isl]) > 0.5)
            Nm[fix] = mean[L.isl[fix]]
        self.island_keys, self.vertex_keys, base_vals, vattr = [], [], {}, {}
        for key, arr in attrs.items():
            am = np.asarray(arr, np.float32).reshape(len(P), -1)[L.src]
            lo, hi = np.minimum.reduceat(am, v0[:-1]), np.maximum.reduceat(am, v0[:-1])
            if np.array_equal(lo, hi):
                self.island_keys.append(key)
                base_vals[key] = lo
            else:
                self.vertex_keys.append(key)
                vattr[key] = am
        self.vattr = vattr
        # prototypes: islands alike in shape, triangles and per-vertex attributes, one map apart
        groups = {}
        for i in range(n):
            lo_v, hi_v = v0[i], v0[i + 1]
            sig = (int(hi_v - lo_v), (L.tris[L.t0[i]:L.t0[i + 1]] - lo_v).tobytes(),
                   b"".join(vattr[key][lo_v:hi_v].tobytes() for key in self.vertex_keys))
            groups.setdefault(sig, []).append(i)
        proto_of = np.arange(n)
        base = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()               # the base frame: each island from its prototype
        err = 0.0
        for members in groups.values():
            rest = np.array(members, np.int64)
            while len(rest) > 1:
                p, cand = rest[0], rest[1:]
                ok, M, e = self._alike(Pm, Nm, p, cand)
                proto_of[cand[ok]] = p
                base[cand[ok]] = M[ok]
                err = max(err, float(e[ok].max(initial=0.0)))
                rest = cand[~ok]
        protos = np.flatnonzero(proto_of == np.arange(n))
        index = np.full(n, -1)
        index[protos] = np.arange(len(protos))
        self.proto = index[proto_of]                                        # per island its prototype
        self.pv0 = np.r_[0, np.cumsum(np.diff(v0)[protos])]                 # prototypes' first vertices
        self.pt0 = np.r_[0, np.cumsum(np.diff(L.t0)[protos])]               # and first triangles
        take = np.concatenate([np.arange(v0[p], v0[p + 1]) for p in protos])
        self.p_pos = Pm[take].astype(np.float32)
        self.p_nrm = Nm[take].astype(np.float32)
        self.p_attr = {key: vattr[key][take] for key in self.vertex_keys}
        self.p_tris = np.concatenate([L.tris[L.t0[p]:L.t0[p + 1]] - v0[p] for p in protos]).astype(np.uint32)
        self.qv = self.pv0[self.proto[L.isl]] + (np.arange(len(L.src)) - v0[L.isl])   # each vertex's prototype's
        self.q = self.p_pos.astype(np.float64)[self.qv]
        self.qn = self.p_nrm.astype(np.float64)[self.qv]
        self.org = self.p_pos.astype(np.float64)[self.pv0[:-1]][self.proto]  # each island's bent maps' origin
        self.fit = _Fit(self.q, v0)
        self.ref = None                 # the islands pose word poses follow: (frame, decoded positions, normals, fit)
        self.ref_vals = None            # and its island attribute values
        self.base_vals = base_vals
        self.base = Pose(ISLANDS, base, base_vals, err, 0)

    # ----------------------------------------------------------------------------------------------- building
    def _alike(self, Pm, Nm, p, cand):
        """Which islands `cand` one map takes island p onto (positions within TOL, derived normals within NORMAL_TOL):
        (ok, the maps (float32 values), their errors)."""
        v0 = self.layout.v0
        size = v0[p + 1] - v0[p]
        q = np.tile(Pm[v0[p]:v0[p + 1]].astype(np.float32).astype(np.float64), (len(cand), 1))
        qn = np.tile(Nm[v0[p]:v0[p + 1]].astype(np.float32).astype(np.float64), (len(cand), 1))
        rows = np.concatenate([np.arange(v0[i], v0[i + 1]) for i in cand])
        g0 = np.arange(len(cand) + 1) * size
        fit = _Fit(q, g0)
        M, _, err, _ = self._maps(fit, q, Pm[rows], g0)
        live = np.tile(self.solid[v0[p]:v0[p + 1]], len(cand)) & self.solid[rows]
        ok = (err <= self.tol) & self._normals_ok(unit(turn(M, q, qn, fit.grp)), Nm[rows], g0, live)
        return ok, M, err

    def _maps(self, fit, q, P, v0, O=None):
        """Maps of fit's groups onto P, affine where that is within TOL, else bent (with origins O), else projective:
        (M float32 values, Q float32 values or None, the error of each decoded as mkr decodes it, the decoded points).
        A whole word tries projective only when a sample of about 512 of its points fits one within 10 TOL (its
        letters moving apart do not)."""
        A = fit.affine(P)
        M, Q = _f32(A), None
        D = apply(M, q, fit.grp)
        err = _worst(_dist(D, P), v0)
        bad = err > self.tol
        if not bad.any():
            return M, Q, err, D
        if O is not None:
            Mb, Qb = fit.quadratic(P, O)
            Mb, Qb = _f32(Mb), _f32(Qb)
            Mt, Qt = M.copy(), np.zeros((len(M), 3, 3))
            Mt[bad], Qt[bad] = Mb[bad], Qb[bad]
            Dt = apply(Mt, q, fit.grp, Qt, O)
            e2 = _worst(_dist(Dt, P), v0)
            take = bad & (e2 <= self.tol)
            if take.any():
                M[take], Q = Mt[take], np.where(take[:, None, None], Qt, 0.0)
                D = np.where(take[fit.grp][:, None], Dt, D)
                err = np.where(take, e2, err)
                bad = err > self.tol
        if not bad.any():
            return M, Q, err, D
        if len(v0) == 2 and v0[-1] > PROBE:
            s = slice(0, None, int(v0[-1]) // 512)
            probe = _Fit(q[s], np.array([0, len(q[s])]))
            Ms = _f32(probe.projective(P[s], [0], probe.affine(P[s])))
            if _dist(apply(Ms, q[s]), P[s]).max() > 10.0 * self.tol:
                return M, Q, err, D
        idx = np.flatnonzero(bad)
        M[idx] = _f32(fit.projective(P, idx, A))
        if Q is not None:
            Q[idx] = 0.0
        D = apply(M, q, fit.grp, Q, O)
        return M, Q, _worst(_dist(D, P), v0), D

    @staticmethod
    def _normals_ok(n, truth, v0, live):
        """Per group: normals n match `truth` within NORMAL_TOL where `live` (a zero normal of Blender's stays zero)."""
        tl = np.sqrt(np.einsum("vi,vi->v", truth, truth))
        cos = np.einsum("vi,vi->v", n, truth) / np.maximum(tl, 1e-30)
        good = np.where(tl > 0.5, cos >= np.cos(NORMAL_TOL), np.einsum("vi,vi->v", n, n) < 1e-12) | ~live
        return np.minimum.reduceat(good.astype(np.int8), v0[:-1]).astype(bool)

    # ----------------------------------------------------------------------------------------------- frames
    def _vals(self, P, attrs):
        """Island attribute values of a frame, or None (per-vertex ones changed, island ones not one per island)."""
        L = self.layout
        for key in self.vertex_keys:
            a = attrs.get(key)
            if a is None or not np.array_equal(np.asarray(a, np.float32).reshape(len(P), -1)[L.src], self.vattr[key]):
                return None
        vals = {}
        for key in self.island_keys:
            a = attrs.get(key)
            if a is None:
                return None
            am = np.asarray(a, np.float32).reshape(len(P), -1)[L.src]
            lo, hi = np.minimum.reduceat(am, L.v0[:-1]), np.maximum.reduceat(am, L.v0[:-1])
            if not np.array_equal(lo, hi):
                return None
            vals[key] = lo
        return vals

    def _islands(self, Pm, Nm, vals):
        fit, v0 = self.fit, self.layout.v0
        M, Q, err, D = self._maps(fit, self.q, Pm, v0, self.org)
        if (err > self.tol).any():
            return None
        Dn = turn(M, self.q, self.qn, fit.grp, Q, self.org)
        if not self._normals_ok(unit(Dn), Nm, v0, self.solid).all():
            return None
        projective = int((~np.all(M[:, 3] == (0.0, 0.0, 0.0, 1.0), 1)).sum())
        pose = Pose(BENT if Q is not None else ISLANDS, M, vals, float(err.max()), projective, Q=Q)
        pose.decoded = (D, Dn)
        return pose

    def _word(self, Pm, Nm, vals):
        k_ref, R, Rn, fit = self.ref
        one = np.array([0, len(R)])
        G, _, err, _ = self._maps(fit, R, Pm, one)
        if err[0] > self.tol or not self._normals_ok(unit(turn(G, R, Rn)), Nm, one, self.solid)[0]:
            return None
        same = all(np.array_equal(vals[key], self.ref_vals[key]) for key in vals)
        return Pose(WORD, G, None if same else vals, float(err[0]), int(not _is_affine(G)), ref=k_ref)

    def pose(self, P, N, attrs):
        """The pose of a later frame of this topology (module docstring), or None."""
        L = self.layout
        vals = self._vals(P, attrs)
        if vals is None:
            return None
        Pm = np.asarray(P, np.float64)[L.src]
        Nm = np.asarray(N, np.float64)[L.src]
        if self.ref is not None:
            pose = self._word(Pm, Nm, vals)
            if pose is not None:
                return pose
        return self._islands(Pm, Nm, vals)

    def held(self, k, pose):
        """Frame k is stored with `pose`: an islands pose becomes the one word poses follow."""
        if pose.kind != WORD:
            R, Rn = pose.decoded or (apply(pose.maps, self.q, self.fit.grp, pose.Q, self.org),
                                     turn(pose.maps, self.q, self.qn, self.fit.grp, pose.Q, self.org))
            self.ref = (k, R, Rn, _Fit(R, np.array([0, len(R)])))        # Rn unnormalised, as the decoder turns it
            self.ref_vals = pose.attrs

    # ----------------------------------------------------------------------------------------------- writing
    def write(self, put):
        """Write the geometry's tables (`put(array, dtype) -> offset`); answers its FIT record (bake_lit.py SCREEN)."""
        L = self.layout
        ptab = np.stack([self.pv0[:-1], np.diff(self.pv0), self.pt0[:-1], np.diff(self.pt0)], 1)
        itab = np.stack([self.proto, L.slot, L.v0[:-1], L.t0[:-1]], 1)
        at = put(itab, "u32")
        pattr = {"position": {"offset": put(self.p_pos, "f32"), "comp": 3, "type": "f32"},
                 "normal": {"offset": put(self.p_nrm, "f32"), "comp": 3, "type": "f32"}}
        seen = {}
        for key in self.vertex_keys:
            a = self.p_attr[key]
            h = (a.shape, a.tobytes())
            if h not in seen:
                seen[h] = put(a, "f32")
            pattr[key] = {"offset": seen[h], "comp": int(a.shape[1]), "type": "f32"}
        return {"vertices": int(len(L.src)), "triangles": {"offset": at, "count": int(len(L.tris))},
                "slots": [list(s) for s in L.slots], "attributes": {},
                "fit": {"protos": {"count": int(len(self.pv0) - 1), "table": put(ptab, "u32"),
                                   "vertices": int(self.pv0[-1]), "attributes": pattr,
                                   "triangles": {"offset": put(self.p_tris, "u32"), "count": int(len(self.p_tris))}},
                        "islands": {"count": int(L.count), "table": at},
                        "attrs": [[key, int(self.base_vals[key].shape[1])] for key in self.island_keys]}}


# ================================================================================================= a word
class Word:
    """A word's frames as fitted geometries (module docstring). `put(array, dtype) -> offset` stores arrays (in the
    word's own part of lit.bin); `stats` counts what each frame became."""

    def __init__(self, n, put, tol=TOL):
        self.put, self.tol = put, tol
        self.frames = np.full(n, -1, np.int64)
        self.poses = np.full(n, -1, np.int64)
        self.geoms, self.seg, self.last = [], None, None
        self.stats = {"frames": 0, "word": 0, "islands": 0, "bent": 0, "projective": 0, "geoms": 0, "refit": 0,
                      "err": 0.0}

    def add(self, k, P, N, attrs, topo=None):
        """Frame k's drawn arrays; `topo` (tris, slots, src) when its topology is not the last frame's held (the first
        frame, the typewriter, a change of material). Answers what it became: "word", "islands", "bent" or "geom"."""
        st = self.stats
        st["frames"] += 1
        if topo is None and self.seg is not None:
            pose = self.seg.pose(P, N, attrs)
            if pose is not None:
                what = {WORD: "word", ISLANDS: "islands", BENT: "bent"}[pose.kind]
                self._hold(k, pose)
                st[what] += 1
                return what
            topo = self.seg.topo
            st["refit"] += 1
        self.seg = Segment(P, N, attrs, *topo, tol=self.tol)
        self.geoms.append(self.seg.write(self.put))
        self.last = None
        st["geoms"] += 1
        self._hold(k, self.seg.base)
        return "geom"

    def _hold(self, k, pose):
        block = pose.block(self.seg.island_keys)
        if self.last is not None and self.last[0] == block:
            at = self.last[1]                           # a frame that did not move: the same pose
        else:
            at = self.put(np.frombuffer(block, np.uint8), "u8")
            self.last = (block, at)
        self.frames[k] = len(self.geoms) - 1
        self.poses[k] = at
        self.seg.held(k, pose)
        self.stats["projective"] += pose.projective
        self.stats["err"] = max(self.stats["err"], pose.err)

    def done(self):
        """No frame comes after the last one added: drop the fitting state (the geometry being fitted, its maps and
        the last pose), keeping the records. A long clip holds hundreds of words, each done when its window ends."""
        self.seg = self.last = None

    def record(self, base=0):
        """The word's `by_frame` record, its BYTE offsets moved by `base`."""
        geoms = [relocate(g, base) for g in self.geoms]
        return {"geoms": geoms, "frames": [int(f) for f in self.frames],
                "poses": [int(p) + base if p >= 0 else -1 for p in self.poses]}


def relocate(geom, base):
    """A FIT record with its BYTE offsets moved by `base`."""
    f = geom["fit"]
    pattr = {key: dict(spec, offset=spec["offset"] + base) for key, spec in f["protos"]["attributes"].items()}
    return dict(geom, triangles=dict(geom["triangles"], offset=geom["triangles"]["offset"] + base),
                fit=dict(f, protos=dict(f["protos"], table=f["protos"]["table"] + base, attributes=pattr,
                                        triangles=dict(f["protos"]["triangles"],
                                                       offset=f["protos"]["triangles"]["offset"] + base)),
                         islands=dict(f["islands"], table=f["islands"]["table"] + base)))


# ================================================================================================= decoding
def _arr(lit, at, n, dt, comp=1):
    a = np.frombuffer(lit, np.dtype(DT[dt]), n * comp, at)
    return a.reshape(n, comp) if comp > 1 else a


def _pose(lit, at, n_isl, attrs):
    """(kind, ref, maps, Q, island values) of the pose at BYTE `at`."""
    kind, has, ref, _ = (int(v) for v in _arr(lit, at, 4, "u32"))
    nm = 1 if kind == WORD else n_isl
    o = at + 16
    maps = _arr(lit, o, nm * 16, "f32").reshape(nm, 4, 4).astype(np.float64)
    o += 64 * nm
    Q = None
    if kind == BENT:
        Q = _arr(lit, o, n_isl * 9, "f32").reshape(n_isl, 3, 3).astype(np.float64)
        o += 36 * n_isl
    vals = None
    if has:
        vals = {}
        for key, comp in attrs:
            vals[key] = _arr(lit, o, n_isl * comp, "f32").reshape(n_isl, comp)
            o += 4 * n_isl * comp
    return kind, ref, maps, Q, vals


def decode(lit, by_frame, k):
    """Frame k of a fitted by_frame record from lit.bin (`lit` bytes or uint8 array): {position (v, 3) float32, normal
    (v, 3) float32 unit, attrs {key: (v, comp) float32}, tris (t, 3) uint32, slots}, or None (hidden): mkr's decoder,
    in numpy (mkr/mkr/src/screen_geo.rs follows it)."""
    gi, at = by_frame["frames"][k], by_frame["poses"][k]
    if gi < 0 or at < 0:
        return None
    g = by_frame["geoms"][gi]
    f = g["fit"]
    pr, isl = f["protos"], f["islands"]
    n = isl["count"]
    itab = _arr(lit, isl["table"], n, "u32", 4).astype(np.int64)
    ptab = _arr(lit, pr["table"], pr["count"], "u32", 4).astype(np.int64)
    pa = pr["attributes"]
    ppos = _arr(lit, pa["position"]["offset"], pr["vertices"], "f32", 3).astype(np.float64)
    pnrm = _arr(lit, pa["normal"]["offset"], pr["vertices"], "f32", 3).astype(np.float64)
    ptri = _arr(lit, pr["triangles"]["offset"], pr["triangles"]["count"], "u32", 3).astype(np.int64)
    cnt = ptab[itab[:, 0], 1]
    isl_v = np.repeat(np.arange(n), cnt)
    qv = ptab[itab[isl_v, 0], 0] + (np.arange(len(isl_v)) - itab[isl_v, 2])
    q, qn = ppos[qv], pnrm[qv]
    org = ppos[ptab[itab[:, 0], 0]]
    kind, ref, maps, Q, vals = _pose(lit, at, n, f["attrs"])
    if kind == WORD:
        _, _, rmaps, rQ, rvals = _pose(lit, by_frame["poses"][ref], n, f["attrs"])
        p1, n1 = apply(rmaps, q, isl_v, rQ, org), turn(rmaps, q, qn, isl_v, rQ, org)
        pos, nrm = apply(maps, p1), turn(maps, p1, n1)
        vals = vals if vals is not None else rvals
    else:
        pos, nrm = apply(maps, q, isl_v, Q, org), turn(maps, q, qn, isl_v, Q, org)
    attrs = {key: _arr(lit, spec["offset"], pr["vertices"], "f32", spec["comp"]).reshape(-1, spec["comp"])[qv]
             for key, spec in pa.items() if key not in ("position", "normal")}
    for key, comp in f["attrs"]:
        attrs[key] = np.asarray(vals[key], np.float32).reshape(n, comp)[isl_v]
    tris = np.concatenate([ptri[ptab[p, 2]:ptab[p, 2] + ptab[p, 3]] + v for p, v in zip(itab[:, 0], itab[:, 2])])
    return {"position": pos.astype(np.float32), "normal": unit(nrm).astype(np.float32), "attrs": attrs,
            "tris": tris.astype(np.uint32), "slots": g["slots"]}
