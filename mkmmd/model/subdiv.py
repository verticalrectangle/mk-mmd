"""Catmull-Clark subdivision with semi-sharp creases as sparse linear operators (bpy-free: numpy + scipy.sparse).

`part.Mesh.subsurf` asks the assembler to apply N Catmull-Clark levels to a cage before export, and the cage carries
vertex weights and morph offsets that must follow. For a fixed topology and crease set every Catmull-Clark rule is a
fixed linear combination of the cage vertices (positions never enter the rules), so one level is a sparse matrix and N
levels are their product. `subdivide` returns those matrices next to the fine mesh:

    sd = subdivide(m.verts, m.faces, m.subsurf, creases=m.crease, uv=m.uv)
    verts    = sd.verts                                                 # == sd.S @ m.verts
    weights  = {b: sd.apply(w, attr=True) for b, w in m.weights.items()}   # L: rows sum to 1, values stay in 0..1
    morphs   = {k: sd.apply(o) for k, o in m.morphs.items()}                # S: offsets (n, 3) -> (m, 3)
    face_mat = m.face_mat[sd.face_parent]                                   # material of every fine face
    sharp    = sd.map_edges(m.sharp)                                        # sharp-normal edges on the fine mesh
    # custom vertex normals can be carried with sd.apply(normals, attr=True) and renormalised.

Public API
----------
    subdivide(verts, faces, levels=1, creases=None, uv=None, boundary='crease', corners='sharp',
              raw_sharpness=False) -> Subdivided
    Subdivided                      .verts (m,3) .faces [[a,b,c,d]..] .uv (corners',2)|None .S (m,n) .L (m,n) .levels
                                    .origin (m,) .face_parent (nfaces',)  .apply(a, attr=False)  .map_edges(edges)
    subdivide_uv(faces, uv, levels=1) -> (corners', 2) array   (the face-varying part alone, no positions needed)
    crease_to_sharpness(c) -> 10 * c**2

Conventions (shared by the whole model toolkit)
-----------------------------------------------
Model space is metres, Z up, the character faces -Y, its left is +X. `verts` (n, 3) float64; `faces` is a list of
vertex-index lists (triangles, quads and n-gons); the winding is counter-clockwise seen from OUTSIDE and stays so. UVs
are per face corner, an array (sum(len(f) for f in faces), 2) in face order (exactly `part.Mesh.uv`), v points UP.

Fine mesh numbering (documented because weights / morphs / edges are keyed by it)
    vertices: the n cage vertices first (same indices, moved by the vertex rules; `origin[i] == i` for them, -1 for the
              rest), then for every level l = 1..levels the edge points of the level-(l-1) mesh, ordered by the sorted
              pair (lower vertex index, higher vertex index) of the edge they split, then its face points in face order.
    faces:    always quads from level 1 on. Level-(l-1) face f with corners v_0..v_(k-1) gives k consecutive quads,
              quad j being [v_j, edge point(v_j, v_j+1), face point(f), edge point(v_j-1, v_j)] (so an n-gon -> n
              quads, a quad -> 4); the next level repeats this on that quad list. `face_parent[q]` is the CAGE face
              of fine face q.
    uv:       4 corners per fine quad in the same order as the quad's vertices above (`Subdivided.uv`).

Rules, per level, evaluated on the current mesh (a crease c in 0..1 has sharpness s = 10 * c**2, Blender/OpenSubdiv)
    face point    mean of the face's vertices.
    edge point    smooth (a + b + F1 + F2) / 4; sharp (s >= 1) midpoint (a + b) / 2; 0 < s < 1 blends smooth -> midpoint
                  by s. Boundary edges (one face) are infinitely sharp (`boundary='crease'`: the boundary stays a smooth
                  curve and does not shrink); so are non-manifold edges (three or more faces).
    vertex point  K = the incident edges that are sharp now (s > 0; boundary and non-manifold edges have s = inf) and
                  K' = those still sharp after this level (s > 1). Fewer than 2 edges select the smooth rule
                  (F_avg + 2 R_avg + (n - 3) P) / n (mean face point, mean edge midpoint, valence n), exactly 2 the
                  crease rule (A + 6 P + B) / 8 along the two sharp edges, 3 or more the corner rule (P stays).
                  The vertex uses rule(K) when rule(K) == rule(K'); otherwise edges are fading out (0 < s <= 1) and it
                  blends w * rule(K) + (1 - w) * rule(K'), w = mean s of the fading edges. So two creases below 1 give
                  the classic crease / smooth blend by their average sharpness, a crease line of s = 10 meeting an
                  edge of s = 0.1 is 10% crease / 90% smooth, and three or more creases of sharpness s < 1 blend
                  smooth -> corner by s.
                  `corners='sharp'` additionally keeps every vertex that belongs to exactly one face (the corner of an
                  open patch); `corners='smooth'` applies the crease rule there (the corner rounds off; as in
                  OpenSubdiv a corner whose two edges also carry user creases s_a, s_b is pulled back towards "stay"
                  with weight min(1, s_a, s_b), which only matters if boundary edges were creased on purpose).
    sharpness     splitting an edge gives two child edges with max(0, s - 1); infinite stays infinite. A crease with
                  c = 1 (s = 10) is therefore sharp for any practical number of levels, c ~ 0.3 (s ~ 0.9) is a slightly
                  softened edge for the first level only.
    attributes    L (weights etc.): cage vertices keep their value, an edge point is the mean of its two end vertices, a
                  face point the mean of the face's vertices (rows sum to 1, values stay inside the cage range).
    uv            face-varying LINEAR: the corner UV of a cage vertex is kept, an edge point's corner UV is the mean
                  of the two corner UVs along that edge IN THAT FACE, a face point's the mean of the face's corner
                  UVs, so UV seams never move or blur (two faces sharing an edge may give the edge point two UVs).

Blender compatibility: positions, UVs, vertex weights and material (face parent) assignment equal Blender's Subdivision
Surface modifier (OpenSubdiv, "use creases", UV smooth "none" = linear, "use limit surface" OFF) to float32 precision
for tris / quads / n-gons, closed and open cages, mixed crease values and levels 1-3 (checked against Blender 4.2.3 and
5.2.2 on about 375 random cages). Blender's Boundary Smooth "Keep corners" is `corners='sharp'`, "All" is
`corners='smooth'`. Differences: the modifier's `use_limit_surface` defaults to True and then moves the last level onto
the limit surface (a cube cage of half-size 1 at 2 levels: by up to 0.05), so set it False whenever S has to agree with
a Blender-evaluated mesh; Blender's default UV smoothing ("preserve boundaries") differs from the linear face-varying
rule used here; its vertex order differs.

Limits and decisions
    * `boundary='smooth'` (boundary treated like the interior, so it shrinks) is not implemented: NotImplementedError.
    * Input validation raises ValueError: indices out of range, faces with fewer than 3 corners or a repeated vertex,
      creases on pairs that are not an edge of the cage, a `uv` of the wrong size. Crease values are clamped to 0..1; a
      pair given in both orders keeps the larger value.
    * `levels=0` returns the cage itself (faces unchanged, identity operators).
    * Vertices that belong to no face are kept as they are. Mixed winding is not repaired (rules need adjacency only).
    * Weld the cage: coincident vertices duplicated along a seam make an open boundary there (boundary edges are
      creases, so the seam stays a curve but the surface is not smooth across it). UV seams belong in the per-corner
      `uv`, hard-shading seams in `Mesh.sharp` (carry them with `map_edges`).
    * Size and speed: S and L have one row per fine vertex (3458 / 13826 / 55298 for an 864-quad cage at 1 / 2 / 3
      levels) and about 6 / 10 / 13 non-zeros per row in S (2-4 in L); that cage takes 0.1 s at 3 levels, 0.4 s at 4.
    * Needs scipy (S and L are scipy CSR matrices); numpy use is plain 1.x-compatible. Blender's bundled Python has no
      scipy: run `subdivide` in the CLI python and hand the fine meshes to the Blender side.
"""
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import scipy.sparse as sp

__all__ = ["Subdivided", "subdivide", "subdivide_uv", "crease_to_sharpness"]


def crease_to_sharpness(c):
    """Blender's edge crease (0..1) -> OpenSubdiv sharpness: s = 10 * c**2 (c is clamped to 0..1).

    c = 1 gives s = 10 (sharp for any practical number of levels), c ~ 0.316 gives s = 1 (one fully sharp level),
    c = 0.3 gives 0.9 (slightly softened). Accepts a scalar (returns a float) or an array (returns a float64 array)."""
    a = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    s = 10.0 * a * a
    return float(s) if s.ndim == 0 else s


# ---------------------------------------------------------------------------------------------------- result type
@dataclass(eq=False)
class Subdivided:
    """Result of `subdivide`: the fine mesh plus the linear operators that produced it.

    verts         (m, 3) float64 fine vertex positions, == S @ verts_in.
    faces         list of quads [a, b, c, d] (counter-clockwise from outside); the cage faces themselves when levels=0.
    uv            (corners', 2) float64 per-corner UVs of `faces` (4 per quad, in face order) or None without input UVs.
    S             scipy CSR (m, n): position operator. Valid for any (n, k) data, in particular morph OFFSETS
                  (offsets_out = S @ offsets); rows sum to 1.
    L             scipy CSR (m, n): linear attribute operator for vertex weights etc.: cage vertices keep their value,
                  an edge point is the mean of its two end vertices, a face point the mean of the face's vertices.
                  Rows sum to 1 and all entries are >= 0, so weights stay in 0..1 and keep summing to 1 over the bones.
    levels        number of Catmull-Clark levels applied.
    origin        (m,) int: cage vertex index of the fine vertices that come from the cage (the first n), else -1.
    face_parent   (len(faces),) int: the cage face each fine face came from (`face_mat_out = face_mat_in[face_parent]`).

    Fine vertex / face numbering is documented in the module docstring."""
    verts: np.ndarray
    faces: list
    uv: np.ndarray
    S: sp.csr_matrix
    L: sp.csr_matrix
    levels: int
    origin: np.ndarray
    face_parent: np.ndarray
    _levels: list = field(default_factory=list, repr=False)       # per level: (vertex count, sorted edge keys)
    _cage_keys: np.ndarray = field(default=None, repr=False)      # sorted edge keys of the cage (lo * n + hi)

    def __repr__(self):
        return (f"Subdivided(levels={self.levels}, cage_verts={self.S.shape[1]}, verts={self.S.shape[0]}, "
                f"faces={len(self.faces)}, uv={'yes' if self.uv is not None else 'no'})")

    @property
    def n_cage(self):
        """Number of cage vertices (columns of S and L)."""
        return self.S.shape[1]

    def apply(self, a, attr=False):
        """Carry per-cage-vertex data to the fine mesh.

        a     (n,) or (n, k) array-like with one row per CAGE vertex (vertex weights, morph offsets, colours ...).
        attr  False: positional operator S (positions and morph OFFSETS, it follows the Catmull-Clark smoothing);
              True: linear attribute operator L (use for vertex weights and other attributes that must not overshoot).
        Returns an ndarray with the same rank as `a` and `len(self.verts)` rows (float64)."""
        a = np.asarray(a, dtype=np.float64)
        if a.ndim not in (1, 2) or a.shape[0] != self.S.shape[1]:
            raise ValueError(f"expected an array with {self.S.shape[1]} rows (one per cage vertex), "
                             f"got shape {a.shape}")
        return (self.L if attr else self.S) @ a

    def map_edges(self, edges):
        """Fine-mesh edges that lie along the given cage edges (to carry `Mesh.sharp`).

        edges  iterable of (i, j) cage vertex pairs, either order; every pair must be an edge of the cage, duplicates
               (also reversed) are merged keeping the first occurrence.
        Returns a list of (a, b) fine vertex-index tuples, 2**levels per cage edge, in the order of the input edges and
        along each edge from its first vertex i to its second vertex j (a is nearer to i). levels=0 returns the edges
        unchanged. Raises ValueError for a pair that is not a cage edge."""
        n = self.S.shape[1]
        e = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64).reshape(-1, 2)
        if e.shape[0] == 0:
            return []
        if e.min() < 0 or e.max() >= n or (e[:, 0] == e[:, 1]).any():
            raise ValueError("map_edges: vertex index out of range or zero-length edge")
        key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
        pos = np.minimum(np.searchsorted(self._cage_keys, key), max(self._cage_keys.size - 1, 0))
        bad = (self._cage_keys[pos] != key) if self._cage_keys.size else np.ones(key.shape, dtype=bool)
        if bad.any():
            raise ValueError(f"map_edges: not an edge of the cage: {[tuple(p) for p in e[bad][:5].tolist()]}")
        _, first = np.unique(key, return_index=True)
        chain_ = e[np.sort(first)]
        for nv, uk in self._levels:
            lo = np.minimum(chain_[:, :-1], chain_[:, 1:])
            hi = np.maximum(chain_[:, :-1], chain_[:, 1:])
            mids = nv + np.searchsorted(uk, lo * nv + hi)
            nxt = np.empty((chain_.shape[0], 2 * chain_.shape[1] - 1), dtype=np.int64)
            nxt[:, 0::2] = chain_
            nxt[:, 1::2] = mids
            chain_ = nxt
        pairs = np.stack([chain_[:, :-1], chain_[:, 1:]], axis=2).reshape(-1, 2)
        return [tuple(p) for p in pairs.tolist()]


# ---------------------------------------------------------------------------------------------------- topology
def _corner_arrays(fsize):
    """Per-corner helpers for faces of the given sizes: (fstart, cface, cnext, cprev), corners numbered face-major."""
    fstart = np.cumsum(fsize) - fsize
    cface = np.repeat(np.arange(fsize.size, dtype=np.int64), fsize)
    local = np.arange(cface.size, dtype=np.int64) - fstart[cface]
    size = fsize[cface]
    cnext = fstart[cface] + (local + 1) % size
    cprev = fstart[cface] + (local - 1) % size
    return fstart, cface, cnext, cprev


class _Topo:
    """Edge / corner bookkeeping of one polygon mesh (one subdivision level); nothing here depends on positions.

    Corners are numbered face-major; `eid[c]` is the unique-edge id of the edge leaving corner c towards the next
    corner of its face. Unique edges are sorted by key = lo * nv + hi (lo < hi vertex ids): `uk`, `ea` (lo), `eb` (hi);
    `ecount` counts the face corners using an edge (1 boundary, 2 manifold interior, more non-manifold); `f1`, `f2`
    are the two faces of a manifold edge (-1 otherwise)."""

    def __init__(self, nv, cv, fsize):
        self.nv = int(nv)
        self.cv = cv
        self.fsize = fsize
        self.nf = int(fsize.size)
        self.nc = int(cv.size)
        self.fstart, self.cface, self.cnext, self.cprev = _corner_arrays(fsize)
        other = cv[self.cnext]
        key = np.minimum(cv, other) * self.nv + np.maximum(cv, other)
        self.uk, self.eid, self.ecount = np.unique(key, return_inverse=True, return_counts=True)
        self.ne = int(self.uk.size)
        self.ea = self.uk // self.nv
        self.eb = self.uk % self.nv
        self.f1 = np.full(self.ne, -1, dtype=np.int64)
        self.f2 = np.full(self.ne, -1, dtype=np.int64)
        man = np.flatnonzero(self.ecount == 2)
        if man.size:
            order = np.argsort(self.eid, kind="stable")
            first = (np.cumsum(self.ecount) - self.ecount)[man]
            self.f1[man] = self.cface[order[first]]
            self.f2[man] = self.cface[order[first + 1]]


def _flatten_faces(faces, nv):
    """faces (list of index lists) -> (cv corner vertex ids (C,), fsize (F,)); validates sizes, ranges, repeats."""
    if not isinstance(faces, (list, tuple, np.ndarray)):
        faces = list(faces)
    fsize = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    cv = np.fromiter(chain.from_iterable(faces), dtype=np.int64, count=int(fsize.sum()))
    if fsize.size and fsize.min() < 3:
        bad = int(np.flatnonzero(fsize < 3)[0])
        raise ValueError(f"face {bad} has {int(fsize[bad])} vertices (need at least 3)")
    if cv.size and (cv.min() < 0 or cv.max() >= nv):
        cface = np.repeat(np.arange(fsize.size), fsize)
        bad = int(cface[np.flatnonzero((cv < 0) | (cv >= nv))[0]])
        raise ValueError(f"face {bad} uses a vertex index outside 0..{nv - 1}: {list(faces[bad])}")
    if cv.size:
        cface = np.repeat(np.arange(fsize.size), fsize)
        order = np.lexsort((cv, cface))
        sv, sf = cv[order], cface[order]
        dup = (sv[1:] == sv[:-1]) & (sf[1:] == sf[:-1])
        if dup.any():
            bad = int(sf[1:][np.flatnonzero(dup)[0]])
            raise ValueError(f"face {bad} repeats a vertex: {list(faces[bad])}")
    return cv, fsize


def _edge_sharpness(creases, topo, raw):
    """User creases {(i, j): c} -> (ne,) sharpness aligned with topo's unique edges (0 where uncreased)."""
    s = np.zeros(topo.ne)
    if not creases:
        return s
    try:
        items = list(creases.items())
        ij = np.array([k for k, _ in items], dtype=np.int64).reshape(-1, 2)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"creases must be a dict {{(i, j): value}}: {exc}") from None
    val = np.array([v for _, v in items], dtype=np.float64)
    if not np.isfinite(val).all():
        raise ValueError("crease values must be finite")
    inside = (ij >= 0).all(axis=1) & (ij < topo.nv).all(axis=1) & (ij[:, 0] != ij[:, 1])
    key = np.minimum(ij[:, 0], ij[:, 1]) * topo.nv + np.maximum(ij[:, 0], ij[:, 1])
    pos = np.minimum(np.searchsorted(topo.uk, key), max(topo.ne - 1, 0))
    ok = inside & (topo.uk[pos] == key) if topo.ne else np.zeros(len(items), dtype=bool)
    if not ok.all():
        bad = [tuple(p) for p in ij[~ok][:5].tolist()]
        raise ValueError(f"crease on a pair that is not an edge of the mesh: {bad}")
    sharp = np.maximum(val, 0.0) if raw else crease_to_sharpness(val)
    np.maximum.at(s, pos, sharp)
    return s


# ---------------------------------------------------------------------------------------------------- one level
def _operators(t, s_user, keep_corners):
    """The two sparse operators of one Catmull-Clark level of mesh `t` with finite crease sharpness `s_user` (ne,).

    Returns (P, Q), both CSR of shape (nv + ne + nf, nv): P the position operator (rows: vertex points, edge points,
    face points), Q the linear attribute operator (vertices kept, edge midpoints, face means)."""
    nv, ne, nf, nc = t.nv, t.ne, t.nf, t.nc
    ea, eb = t.ea, t.eb
    s = np.where(t.ecount == 2, s_user, np.inf)                      # boundary / non-manifold edges: infinitely sharp

    # face points: mean of the face's vertices
    Mf = sp.csr_matrix((1.0 / t.fsize[t.cface], (t.cface, t.cv)), shape=(nf, nv))

    # edge points: ((1 - te) / 2 + te) * midpoint + (1 - te) / 4 * (F1 + F2), te = min(1, s)
    AB = sp.csr_matrix((np.full(2 * ne, 0.5), (np.repeat(np.arange(ne), 2), np.stack([ea, eb], axis=1).ravel())),
                       shape=(ne, nv))
    te = np.minimum(s, 1.0)
    man = np.flatnonzero(t.ecount == 2)
    EF = sp.csr_matrix((np.ones(2 * man.size), (np.repeat(man, 2), np.stack([t.f1[man], t.f2[man]], axis=1).ravel())),
                       shape=(ne, nf))
    Me = sp.diags((1.0 - te) * 0.5 + te) @ AB + sp.diags((1.0 - te) * 0.25) @ (EF @ Mf)

    # vertex points: per vertex a convex mix of the smooth / crease / corner point (w_s + w_c + w_k = 1). The rule is
    # chosen by the number of sharp edges now (K_p: s > 0) and after this level (K_c: s > 1, the edges that survive
    # the decay s -> s - 1); while edges fade out (0 < s <= 1) the two rules are blended by their mean sharpness.
    n_e = np.bincount(ea, minlength=nv) + np.bincount(eb, minlength=nv)      # valence (incident edges)
    n_f = np.bincount(t.cv, minlength=nv)                                    # incident faces (corners)
    sh = np.flatnonzero(s > 0.0)
    v_inc = np.concatenate([ea[sh], eb[sh]])                                 # (vertex, other end, sharpness) incidences
    o_inc = np.concatenate([eb[sh], ea[sh]])
    s_inc = np.concatenate([s[sh], s[sh]])
    alive = s_inc > 1.0                                                      # still sharp at the next level
    k_p = np.bincount(v_inc, minlength=nv)
    k_c = np.bincount(v_inc[alive], minlength=nv)
    n_fade = np.bincount(v_inc[~alive], minlength=nv)
    sum_fade = np.bincount(v_inc[~alive], weights=s_inc[~alive], minlength=nv)
    r_p = np.clip(k_p - 1, 0, 2)                                             # 0 smooth (k<=1), 1 crease (k=2), 2 corner
    r_c = np.clip(k_c - 1, 0, 2)
    w_fade = np.where(n_fade > 0, sum_fade / np.maximum(n_fade, 1), 1.0)     # weight of rule(K) while edges fade
    same = r_p == r_c
    W = np.zeros((nv, 3))
    rows_v = np.arange(nv)
    W[rows_v, r_p] = np.where(same, 1.0, w_fade)
    W[rows_v, r_c] += np.where(same, 0.0, 1.0 - w_fade)
    force = n_e == 0                                                         # isolated vertices stay
    if keep_corners:
        force |= n_f == 1                                                    # corner of an open patch
    else:
        # patch corners round off with the crease rule, except that OpenSubdiv also pulls a corner towards "stay" by
        # min(1, creases of its two edges) when both of its edges carry a user crease
        corner = np.flatnonzero(n_f == 1)
        if corner.size:
            v_all = np.concatenate([ea, eb])
            e_all = np.concatenate([np.arange(ne), np.arange(ne)])
            pick = n_f[v_all] == 1
            s_min = np.full(nv, np.inf)
            np.minimum.at(s_min, v_all[pick], s_user[e_all[pick]])
            w_cor = np.minimum(1.0, s_min[corner])
            W[corner, 0], W[corner, 1], W[corner, 2] = 0.0, 1.0 - w_cor, w_cor
    W[force] = (0.0, 0.0, 1.0)
    w_s, w_c, w_k = W[:, 0], W[:, 1], W[:, 2]
    # the crease rule runs along the two edges that are sharp after the level (when that is a crease) else now
    thr = np.where(r_c == 1, 1.0, 0.0)
    use = (w_c[v_inc] > 0.0) & (s_inc > thr[v_inc])
    Cr = sp.csr_matrix((np.full(int(use.sum()), 0.125) * w_c[v_inc[use]], (v_inc[use], o_inc[use])), shape=(nv, nv))

    n = np.maximum(n_e, 1).astype(np.float64)
    nfv = np.maximum(n_f, 1).astype(np.float64)
    VF = sp.csr_matrix((np.ones(nc), (t.cv, t.cface)), shape=(nv, nf))
    Adj = sp.csr_matrix((np.ones(2 * ne), (np.concatenate([ea, eb]), np.concatenate([eb, ea]))), shape=(nv, nv))
    # smooth rule (F_avg + 2 R_avg + (n - 3) P) / n with R_avg = (n P + sum of neighbours) / (2 n)
    diag = w_s * (n - 2.0) / n + 0.75 * w_c + w_k
    Mv = sp.diags(diag) + sp.diags(w_s / (n * nfv)) @ (VF @ Mf) + sp.diags(w_s / (n * n)) @ Adj + Cr

    P = sp.vstack([Mv, Me, Mf], format="csr")
    P.eliminate_zeros()
    P.sort_indices()
    Q = sp.vstack([sp.identity(nv, format="csr"), AB, Mf], format="csr")
    Q.sort_indices()
    return P, Q


def _child_corners(t):
    """Corner vertex ids (4 * nc,) of the quads that replace the faces of `t` (one quad per corner)."""
    c4 = np.empty((t.nc, 4), dtype=np.int64)
    c4[:, 0] = t.cv
    c4[:, 1] = t.nv + t.eid
    c4[:, 2] = t.nv + t.ne + t.cface
    c4[:, 3] = t.nv + t.eid[t.cprev]
    return c4.reshape(-1)


def _child_sharpness(t, nxt, s_user):
    """Crease sharpness of the unique edges of the refined mesh `nxt`: the two halves of a creased edge get s - 1."""
    s_next = np.zeros(nxt.ne)
    mid = t.nv + np.arange(t.ne, dtype=np.int64)
    dec = np.maximum(s_user - 1.0, 0.0)
    for end in (t.ea, t.eb):
        s_next[np.searchsorted(nxt.uk, end * nxt.nv + mid)] = dec
    return s_next


def _uv_level(uv, fsize, fstart, cface, cnext, cprev):
    """One level of face-varying linear subdivision of corner data `uv` (C, k) -> (4 C, k), quad order as the faces."""
    fmean = np.add.reduceat(uv, fstart, axis=0) / fsize[:, None]
    out = np.empty((uv.shape[0], 4, uv.shape[1]))
    out[:, 0] = uv
    out[:, 1] = 0.5 * (uv + uv[cnext])
    out[:, 2] = fmean[cface]
    out[:, 3] = 0.5 * (uv[cprev] + uv)
    return out.reshape(-1, uv.shape[1])


def _check_uv(uv, ncorners):
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] != ncorners:
        raise ValueError(f"uv must have one row per face corner ({ncorners}), got shape {uv.shape}")
    return uv


def _check_levels(levels):
    try:
        n = int(levels)
    except (TypeError, ValueError):
        raise ValueError(f"levels must be a non-negative integer, got {levels!r}") from None
    if n != levels or n < 0:
        raise ValueError(f"levels must be a non-negative integer, got {levels!r}")
    return n


# ---------------------------------------------------------------------------------------------------- public API
def subdivide(verts, faces, levels=1, creases=None, uv=None, boundary="crease", corners="sharp", raw_sharpness=False):
    """Catmull-Clark subdivision of a polygon cage with semi-sharp creases; returns a `Subdivided`.

    verts     (n, 3) cage positions. faces: list of vertex-index lists (triangles, quads, n-gons; CCW from outside).
              Every face must have at least 3 distinct vertices.
    levels    number of subdivision levels (0 returns the cage with identity operators); after level 1 every face is
              a quad.
    creases   {(i, j): c} semi-sharp creases on cage edges (either vertex order), c in 0..1 (clamped), sharpness
              s = 10 * c**2 (`crease_to_sharpness`); None/empty = none. Each level halves an edge into two children with
              sharpness s - 1 (clamped at 0). A pair that is not an edge raises ValueError.
    uv        (sum of face sizes, 2) per-corner UVs in face order (v up) or None; subdivided face-varying linearly
              (corner UV kept at cage vertices, edge point = mean of the two corner UVs in that face, face point =
              mean of the face's corners).
    boundary  'crease': open boundary edges are infinitely sharp creases (the boundary stays a smooth curve, it does not
              shrink); 'smooth' is not implemented (NotImplementedError).
    corners   'sharp': a boundary vertex used by exactly one face (the corner of an open patch) stays; 'smooth': it gets
              the plain crease rule (and rounds off).
    raw_sharpness  True: `creases` values are already sharpness (>= 0, no 10*c*c mapping and no 0..1 clamp).

    Returns Subdivided(verts, faces, uv, S, L, levels, origin, face_parent) - see its docstring and the module docstring
    for the exact vertex / face numbering and the rules. The operators make the call reusable on other data:
    `sd.apply(weights, attr=True)` for vertex weights, `sd.apply(offsets)` for morph offsets, `sd.map_edges(sharp)`,
    `face_mat[sd.face_parent]`. Vectorised numpy / scipy.sparse; cost is dominated by building S (levels <= 3 fine)."""
    if boundary == "smooth":
        raise NotImplementedError("boundary='smooth' is not implemented; only boundary='crease'")
    if boundary != "crease":
        raise ValueError(f"boundary must be 'crease' or 'smooth', got {boundary!r}")
    if corners not in ("sharp", "smooth"):
        raise ValueError(f"corners must be 'sharp' or 'smooth', got {corners!r}")
    levels = _check_levels(levels)
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 2:
        raise ValueError(f"verts must be an (n, 3) array, got shape {verts.shape}")
    n = verts.shape[0]
    cv, fsize = _flatten_faces(faces, n)
    if levels > 0 and fsize.size == 0:
        raise ValueError("cannot subdivide a mesh without faces")
    uv_cur = _check_uv(uv, cv.size)
    topo = _Topo(n, cv, fsize) if fsize.size else None
    s_user = _edge_sharpness(creases, topo, raw_sharpness) if topo is not None else None
    if creases and topo is None:
        raise ValueError("creases given for a mesh without faces")
    cage_keys = topo.uk if topo is not None else np.zeros(0, dtype=np.int64)

    S = L = None
    snapshots = []
    parent = np.arange(fsize.size, dtype=np.int64)
    cv_out = cv
    for lev in range(levels):
        if lev > 0:
            nxt = _Topo(topo.nv + topo.ne + topo.nf, cv_out, np.full(cv_out.size // 4, 4, dtype=np.int64))
            s_user = _child_sharpness(topo, nxt, s_user)
            topo = nxt
        snapshots.append((topo.nv, topo.uk))
        P, Q = _operators(topo, s_user, corners == "sharp")
        S = P if S is None else P @ S
        L = Q if L is None else Q @ L
        parent = parent[topo.cface]
        if uv_cur is not None:
            uv_cur = _uv_level(uv_cur, topo.fsize, topo.fstart, topo.cface, topo.cnext, topo.cprev)
        cv_out = _child_corners(topo)

    if levels == 0:
        S = sp.identity(n, format="csr")
        L = sp.identity(n, format="csr")
        out_faces = [[int(v) for v in f] for f in faces]
        uv_out = None if uv_cur is None else uv_cur.copy()
    else:
        out_faces = cv_out.reshape(-1, 4).tolist()
        uv_out = uv_cur
    S.sort_indices()
    L.sort_indices()
    verts_out = S @ verts
    origin = np.full(S.shape[0], -1, dtype=np.int64)
    origin[:n] = np.arange(n)
    return Subdivided(verts=np.ascontiguousarray(verts_out), faces=out_faces, uv=uv_out, S=S, L=L, levels=levels,
                      origin=origin, face_parent=parent, _levels=snapshots, _cage_keys=cage_keys)


def subdivide_uv(faces, uv, levels=1):
    """Face-varying linear subdivision of per-corner data alone (no positions or edge topology needed).

    faces   list of vertex-index lists (only their sizes are used: tris, quads, n-gons).
    uv      (sum of face sizes, 2) per-corner UVs in face order, v up (any trailing size works: colours etc.).
    levels  number of Catmull-Clark levels.
    Returns the (corners', 2) float64 array that `subdivide(..., uv=uv).uv` gives for the same faces and levels: the
    corner UV of a cage vertex is kept, the edge point's corner UV is the mean of the two corner UVs along that edge in
    that face, the face point's the mean of the face's corner UVs (seams never move or blur). levels=0: a copy."""
    levels = _check_levels(levels)
    if not isinstance(faces, (list, tuple, np.ndarray)):
        faces = list(faces)
    fsize = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    if fsize.size and fsize.min() < 3:
        raise ValueError(f"face {int(np.flatnonzero(fsize < 3)[0])} has fewer than 3 vertices")
    cur = _check_uv(uv, int(fsize.sum()))
    if cur is None:
        raise ValueError("uv is required")
    cur = cur.copy() if levels == 0 else cur
    for _ in range(levels):
        fstart, cface, cnext, cprev = _corner_arrays(fsize)
        cur = _uv_level(cur, fsize, fstart, cface, cnext, cprev)
        fsize = np.full(cur.shape[0] // 4, 4, dtype=np.int64)
    return cur
