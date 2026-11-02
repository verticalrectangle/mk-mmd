"""semi_truck_body_kit: cage builders the semi truck's body shells share (pure numpy, no bpy: the tests import it).

The cab and the sleeper are stacks of rings: horizontal plan outlines (rounded rectangles, `plan_point`) at rising heights,
every ring with the same columns, so column j is one line running up the body (a corner, a pillar's edge, the side of an
opening). `ring_shell` turns a stack into a closed cage:

    ends      capped (`Cap`: a few shrunken rings and a fan, domed: a roof, a ceiling, a vaulted underside) or bridged to the
              ring of a second stack inside the first (`"rim"`: the bottom edge of a hollow shell's walls)
    inner     a second stack with the same columns inside the first: a hollow shell (the outer skin faces out, the inner
              skin faces the cavity)
    holes     wall quads left out of both skins: openings whose sides are bridged through the wall (window reveals); in a
              solid stack a rectangular block of holes becomes a pocket (`Pocket`: walls pushed in and a back)

Faces carry material indices, edges creases (the rims of openings), and `ring_shell` also returns the vertex index grids so
a caller can crease or recolour by column and row.
"""
import math
from collections import namedtuple

import numpy as np

from ....core import shell as S

Cap = namedtuple("Cap", "ts bulge_out bulge_in")      # shrunken ring scales (1 > t > 0), dome heights (outer, inner skin)
Pocket = namedtuple("Pocket", "rows cols offset steps ts bulge mat")


# =================================================================================================== plan outlines
def plan_point(P, spec):
    """A point (x, y) on the plan outline P = (hw, yf, yb, rf, rb): a rectangle x in [-hw, hw], y from the front yf (the
    smaller y: forward is -y) to the back yb, its front corners rounded by rf and its back corners by rb.
    spec = (segment, value, side) with side +1 for the +x half and -1 for the mirrored one:
      "F"  front face at x = value (onto the corner arc past the straight part)
      "CF" front corner, value = 0 at the front tangent .. 1 at the side tangent
      "S"  side at y = value (onto the corner arcs past the straight part)
      "CB" back corner, value = 0 at the side tangent .. 1 at the back tangent
      "B"  back face at x = value (onto the corner arc past the straight part)"""
    hw, yf, yb, rf, rb = P
    seg, v, s = spec
    if seg == "F":
        a = abs(v) - (hw - rf)
        if a <= 0:
            return (v, yf)
        a = min(a, rf)
        return (math.copysign(hw - rf + a, v), yf + rf - math.sqrt(max(rf * rf - a * a, 0.0)))
    if seg == "CF":
        t = 0.5 * math.pi * v
        return (s * (hw - rf + rf * math.sin(t)), yf + rf - rf * math.cos(t))
    if seg == "S":
        if v < yf + rf:
            a = min(yf + rf - v, rf)
            return (s * (hw - rf + math.sqrt(max(rf * rf - a * a, 0.0))), v)
        if v > yb - rb:
            a = min(v - (yb - rb), rb)
            return (s * (hw - rb + math.sqrt(max(rb * rb - a * a, 0.0))), v)
        return (s * hw, v)
    if seg == "CB":
        t = 0.5 * math.pi * v
        return (s * (hw - rb + rb * math.cos(t)), yb - rb + rb * math.sin(t))
    if seg == "B":
        a = abs(v) - (hw - rb)
        if a <= 0:
            return (v, yb)
        a = min(a, rb)
        return (math.copysign(hw - rb + a, v), yb - rb + math.sqrt(max(rb * rb - a * a, 0.0)))
    raise ValueError(f"plan_point: segment {seg!r}")


def mirrored_columns(front, side, back_corner, back):
    """The column specs of a whole ring, counter-clockwise seen from above, starting on the front face's +x half: `front`
    the front face's x values >= 0 (ascending) and front corner fractions as ("CF", t), `side` the side's specs (y values
    ascending, or ("CF", t) / ("CB", t) tuples), `back_corner` its back corner fractions, `back` the back face's x values >= 0
    (descending). The -x half is the mirror image (a front value 0 and a back value 0 are shared)."""
    def spec(item, seg, s):
        if isinstance(item, tuple):
            return (item[0], item[1], s)
        return (seg, s * item if seg in ("F", "B") else item, s)

    plus = ([spec(v, "F", 1) for v in front] + [spec(v, "S", 1) for v in side]
            + [("CB", t, 1) for t in back_corner] + [spec(v, "B", 1) for v in back])
    minus = [(seg, -v if seg in ("F", "B") else v, -1) for seg, v, _ in reversed(plus)]
    if plus[0][0] == "F" and plus[0][1] == 0.0:
        minus = minus[:-1]                          # the shared centre column of the front face
    if plus[-1][0] == "B" and plus[-1][1] == 0.0:
        minus = minus[1:]                           # and of the back face
    return plus + minus


def ring(P, specs, z):
    """(C, 3) points of the plan outline P at height z (z a number or one per column)."""
    xy = np.array([plan_point(P, sp) for sp in specs], float)
    zz = np.broadcast_to(np.asarray(z, float), (len(specs),))
    return np.column_stack([xy, zz])


# =================================================================================================== caps and bridges
def fan_cap(ring_pts, axis, ts=(0.9, 0.5), bulge=0.0, centre=None):
    """Faces closing a ring (C, 3) that looks along the unit `axis`: shrunken copies at the scales `ts` toward the centre
    (the ring's mean unless given), the last one fanned to the centre; the rings are lifted along `axis` by
    bulge * (1 - t^2) and the centre by `bulge` (a dome; negative: a vault). Returns (new points, quads, triangles) with
    indices counting from the ring's own (0..C-1) and new points from C; faces look along `axis`."""
    R = np.asarray(ring_pts, float)
    C = len(R)
    ax = np.asarray(axis, float)
    c = R.mean(0) if centre is None else np.asarray(centre, float)
    new, Q = [], []
    prev = np.arange(C)
    for k, t in enumerate(ts):
        ids = C + k * C + np.arange(C)
        new.append(c + (R - c) * t + ax * (bulge * (1.0 - t * t)))
        Q.append(np.stack([prev, np.roll(prev, -1), np.roll(ids, -1), ids], 1))
        prev = ids
    ci = C + len(ts) * C
    new.append((c + ax * bulge)[None])
    T = np.stack([prev, np.roll(prev, -1), np.full(C, ci)], 1)
    P = np.concatenate([R] + new)
    Qa = np.concatenate(Q) if Q else np.zeros((0, 4), np.int64)
    look = sum(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, -1]] - P[F[:, 0]]).sum(0) @ ax for F in (T, Qa) if len(F))
    if look < 0:
        T, Qa = T[:, ::-1], Qa[:, ::-1]
    return np.concatenate(new), Qa, T


class Builder:
    """Accumulates points and faces (quads, triangles) with material indices and creases into one `Mesh`."""

    def __init__(self):
        self.V, self.Q, self.T, self.Qm, self.Tm, self.Ce, self.Cw = [], [], [], [], [], [], []
        self.n = 0

    def add(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        ids = self.n + np.arange(len(P))
        self.V.append(P)
        self.n += len(P)
        return ids

    def quads(self, F, mat):
        F = np.asarray(F, np.int64).reshape(-1, 4)
        self.Q.append(F)
        self.Qm.append(np.broadcast_to(np.asarray(mat, np.int32), (len(F),)).copy())

    def tris(self, F, mat):
        F = np.asarray(F, np.int64).reshape(-1, 3)
        self.T.append(F)
        self.Tm.append(np.broadcast_to(np.asarray(mat, np.int32), (len(F),)).copy())

    def crease(self, pairs, w=1.0):
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if len(pairs):
            self.Ce.append(pairs)
            self.Cw.append(np.broadcast_to(np.asarray(w, float), (len(pairs),)).copy())

    def cap(self, ring_ids, axis, ts, bulge, mat, centre=None):
        pts, Q, T = fan_cap(np.concatenate(self.V)[ring_ids], axis, ts, bulge, centre)
        ids = self.add(pts)
        lut = np.concatenate([np.asarray(ring_ids), ids])
        self.quads(lut[Q], mat)
        self.tris(lut[T], mat)
        return ids

    def mesh(self):
        m = S.Mesh(np.concatenate(self.V), np.concatenate(self.Q) if self.Q else None,
                   np.concatenate(self.T) if self.T else None,
                   np.concatenate(self.Qm) if self.Qm else None, np.concatenate(self.Tm) if self.Tm else None)
        if self.Ce:
            m.crease(np.concatenate(self.Ce), np.concatenate(self.Cw))
        return m


def _per_quad(mat, shape):
    return np.broadcast_to(np.asarray(mat, np.int32), shape)


def hole_edges(holes):
    """The boundary edges of a block of hole quads in a ring grid (R - 1, C) (closed along the columns): [(row, col, edge)]
    with edge 0 the quad's bottom (ring row to col + 1), 1 its right side, 2 its top, 3 its left side, for every edge whose
    neighbouring quad is no hole."""
    H = np.asarray(holes, bool)
    R1, C = H.shape
    out = []
    for k, j in zip(*np.nonzero(H)):
        nb = ((k - 1, j), (k, (j + 1) % C), (k + 1, j), (k, (j - 1) % C))
        for e, (a, b) in enumerate(nb):
            if not (0 <= a < R1) or not H[a, b]:
                out.append((int(k), int(j), e))
    return out


def _edge_ids(G, k, j, e):
    """The two vertex ids (a, b) of edge e of quad (k, j) in the grid G (R, C), in the quad's own winding (outer skin)."""
    C = G.shape[1]
    j1 = (j + 1) % C
    return ((G[k, j], G[k, j1]), (G[k, j1], G[k + 1, j1]), (G[k + 1, j1], G[k + 1, j]), (G[k + 1, j], G[k, j]))[e]


def _by_label(v, label):
    return v.get(label, v.get(0, 0)) if isinstance(v, dict) else v


def ring_shell(outer, inner=None, holes=None, top=Cap((0.9, 0.5), 0.0, 0.0), bottom=Cap((0.9, 0.5), 0.0, 0.0),
               mat_out=0, mat_in=0, mat_rim=0, mat_top=None, mat_bottom=None, mat_top_in=None, mat_bottom_in=None,
               crease_out=1.0, crease_in=1.0, pockets=(), mat_end=None):
    """A closed cage from a stack of rings (see the module docstring). outer (R, C, 3): rings from the lowest up, each
    counter-clockwise seen from above. inner: the same for the inner skin (None: a solid). holes (R - 1, C): wall quads
    to leave out (never in the first or last row of quads), bool or int labels (0 = wall, 1, 2 ... = openings).
    top / bottom: a `Cap` or "rim" (needs `inner`). mat_out / mat_in: the walls' material (a number or one per wall
    quad), mat_rim the openings' reveals (a number or {label: material}), mat_end the end rims (default mat_rim's),
    mat_top / mat_bottom (default mat_out) the outer caps, mat_top_in / mat_bottom_in (default mat_in) the inner ones.
    crease_out / crease_in: crease weight of the edges round each opening on the outer / inner skin (a number or
    {label: weight}). pockets: `Pocket`s (solid only): a rectangular block of holes (rows (k0, k1), cols (j0, j1) quad
    ranges) whose boundary is pushed in by `offset` in `steps` rings (the pocket's walls) and closed by a fan (`ts`,
    `bulge` along -offset's direction: its back), in material `mat`.
    Returns (Mesh, {"out": (R, C) ids, "in": (R, C) ids or None})."""
    O = np.asarray(outer, float)
    R, C = O.shape[:2]
    HL = np.zeros((R - 1, C), np.int64) if holes is None else np.asarray(holes).astype(np.int64)
    if HL.shape != (R - 1, C):
        raise ValueError(f"ring_shell: holes {HL.shape}, want {(R - 1, C)}")
    H = HL > 0
    if H[0].any() or H[-1].any():
        raise ValueError("ring_shell: an opening may not touch the end rows")
    m_end = _by_label(mat_rim, 0) if mat_end is None else mat_end
    b = Builder()
    go = b.add(O.reshape(-1, 3)).reshape(R, C)
    gi = None
    if inner is not None:
        I = np.asarray(inner, float)
        if I.shape != O.shape:
            raise ValueError("ring_shell: the inner stack must have the outer's shape")
        gi = b.add(I.reshape(-1, 3)).reshape(R, C)
    k, j = np.meshgrid(np.arange(R - 1), np.arange(C), indexing="ij")
    j1 = (j + 1) % C
    keep = ~H
    mo = _per_quad(mat_out, (R - 1, C))
    Fo = np.stack([go[k, j], go[k, j1], go[k + 1, j1], go[k + 1, j]], -1)
    b.quads(Fo[keep], mo[keep])
    if gi is not None:
        mi = _per_quad(mat_in, (R - 1, C))
        Fi = np.stack([gi[k, j], gi[k + 1, j], gi[k + 1, j1], gi[k, j1]], -1)
        b.quads(Fi[keep], mi[keep])
    up, down = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
    m_top = mat_out if mat_top is None else mat_top
    m_bot = mat_out if mat_bottom is None else mat_bottom
    for end, spec, axis, m_o, m_i in ((R - 1, top, up, m_top, mat_top_in), (0, bottom, down, m_bot, mat_bottom_in)):
        if isinstance(spec, Cap):
            b.cap(go[end], axis, spec.ts, spec.bulge_out, int(np.asarray(m_o).flat[0]))
            if gi is not None:
                mm = mat_in if m_i is None else m_i
                b.cap(gi[end], tuple(-a for a in axis), spec.ts, spec.bulge_in, int(np.asarray(mm).flat[0]))
        elif spec == "rim":
            if gi is None:
                raise ValueError("ring_shell: a rim end needs the inner stack")
            a, bb = go[end], gi[end]
            jj = np.arange(C)
            jn = (jj + 1) % C
            if end == 0:
                F = np.stack([a[jn], a[jj], bb[jj], bb[jn]], 1)
            else:
                F = np.stack([a[jj], a[jn], bb[jn], bb[jj]], 1)
            b.quads(F, m_end)
        else:
            raise ValueError(f"ring_shell: end {spec!r}")
    edges = hole_edges(H)
    pocket_cells = set()
    for pk in pockets:
        if gi is not None:
            raise ValueError("ring_shell: pockets are for solid stacks")
        (k0, k1), (j0, j1) = pk.rows, pk.cols
        cells = {(kk, jj % C) for kk in range(k0, k1) for jj in range(j0, j1)}
        if not all(H[c] for c in cells):
            raise ValueError("ring_shell: a pocket's block must be holes")
        pocket_cells |= cells
        loop = ([go[k0, jj % C] for jj in range(j0, j1)] + [go[kk, j1 % C] for kk in range(k0, k1)]
                + [go[k1, jj % C] for jj in range(j1, j0, -1)] + [go[kk, j0 % C] for kk in range(k1, k0, -1)])
        loop = np.array(loop)
        P0 = np.concatenate(b.V)[loop]
        prev = loop
        off = np.asarray(pk.offset, float)
        for s in range(1, pk.steps + 1):
            ids = b.add(P0 + off * (s / pk.steps))
            n = len(loop)
            ii = np.arange(n)
            b.quads(np.stack([prev[ii], prev[(ii + 1) % n], ids[(ii + 1) % n], ids[ii]], 1), pk.mat)
            prev = ids
        axis = -off / np.linalg.norm(off)
        b.cap(prev, axis, pk.ts, -pk.bulge, pk.mat)
        b.crease(np.stack([loop, np.roll(loop, -1)], 1), _by_label(crease_out, int(HL[k0, j0 % C])))
    for kk, jj, e in edges:
        if (kk, jj) in pocket_cells:
            continue
        if gi is None:
            raise ValueError("ring_shell: an opening in a solid stack must be a pocket")
        lab = int(HL[kk, jj])
        a, c = _edge_ids(go, kk, jj, e)
        a2, c2 = _edge_ids(gi, kk, jj, e)
        b.quads([a, c, c2, a2], _by_label(mat_rim, lab))
        b.crease([(a, c)], _by_label(crease_out, lab))
        b.crease([(a2, c2)], _by_label(crease_in, lab))
    return b.mesh(), {"out": go, "in": gi}


def grid_slab(outer, inner, keep=None, mat_out=0, mat_in=0, mat_rim=0, crease_out=1.0, crease_in=1.0, crease_rim=0.0):
    """A closed plate from an open grid: outer (R, C, 3) one skin (its quads (k, j) -> (k, j + 1) -> (k + 1, j + 1) face
    out), inner (R, C, 3) the other, keep (R - 1, C - 1) bool the quads that exist (default all: a window is a block of
    False). Every boundary edge of the kept quads (the plate's edge and the openings') is bridged through the plate (the
    rim, `mat_rim`) and creased on both skins (crease_out / crease_in); crease_rim creases the rim's corner edges across
    the plate where the boundary turns (keeps the plate's corners square). Returns (Mesh, {"out": ids, "in": ids})."""
    O, I = np.asarray(outer, float), np.asarray(inner, float)
    R, C = O.shape[:2]
    K = np.ones((R - 1, C - 1), bool) if keep is None else np.asarray(keep, bool)
    b = Builder()
    go = b.add(O.reshape(-1, 3)).reshape(R, C)
    gi = b.add(I.reshape(-1, 3)).reshape(R, C)
    k, j = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    b.quads(np.stack([go[k, j], go[k, j + 1], go[k + 1, j + 1], go[k + 1, j]], -1)[K], _per_quad(mat_out, K.shape)[K])
    b.quads(np.stack([gi[k, j], gi[k + 1, j], gi[k + 1, j + 1], gi[k, j + 1]], -1)[K], _per_quad(mat_in, K.shape)[K])
    turn = {}
    for kk, jj in zip(*np.nonzero(K)):
        nb = ((kk - 1, jj), (kk, jj + 1), (kk + 1, jj), (kk, jj - 1))
        for e, (a, c) in enumerate(nb):
            if 0 <= a < R - 1 and 0 <= c < C - 1 and K[a, c]:
                continue
            p, q = _edge_ids(go, kk, jj, e)
            p2, q2 = _edge_ids(gi, kk, jj, e)
            b.quads([q, p, p2, q2], mat_rim)                 # the kept quad runs p -> q: the rim runs q -> p
            b.crease([(p, q)], crease_out)
            b.crease([(p2, q2)], crease_in)
            for v, v2, d in ((p, p2, e), (q, q2, e)):
                turn.setdefault((int(v), int(v2)), set()).add(d % 2)
    if crease_rim > 0:
        b.crease([key for key, dirs in turn.items() if len(dirs) > 1], crease_rim)
    return b.mesh(), {"out": go, "in": gi}


# =================================================================================================== panels on planes
def frame_in_plane(origin, normal, up):
    """(origin, u, v, n): an orthonormal frame on a plane, u to the right seen from the front (along up x normal), v up."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    v = np.asarray(up, float)
    v = v - n * (v @ n)
    v = v / np.linalg.norm(v)
    return np.asarray(origin, float), np.cross(v, n), v, n


def to_plane(P2, frame, depth=0.0):
    """Points (m, 2) of a plane's (u, v) coordinates (plus `depth` along its normal) in 3D."""
    o, u, v, n = frame
    P2 = np.asarray(P2, float)
    return o + P2[:, :1] * u + P2[:, 1:2] * v + np.asarray(depth, float).reshape(-1, 1) * n


def seal(outline3, normal, width, height, mat, n_sec=8, lift=0.0):
    """A rubber seal round an opening: a rounded section `width` across (in the surface) and `height` proud of it swept
    along the closed outline (m, 3) whose surface normal is `normal` (one vector or one per point)."""
    P = np.asarray(outline3, float)
    N = np.broadcast_to(np.asarray(normal, float), P.shape)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    sec = S.superellipse(width, 2.0 * height, 2.6, n_sec * 2)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    U = np.cross(N, T)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    rings = P[:, None, :] + N[:, None, :] * lift + sec[None, :, 0:1] * U[:, None, :] + sec[None, :, 1:2] * N[:, None, :]
    m = S.cage(rings, closed=True, loop=True, mat=mat)
    return m if m.volume() > 0 else m.flip()
