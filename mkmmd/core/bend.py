"""Normals that bend with a deforming mesh as Blender's do (numpy only; the lit bake, mkmmd/blender/build/bake_lit.py,
uses it and its tests check it, and mk's GPU renderer does the same on every frame).

Blender gives each corner of a mesh the normal of its fan: the corners round one vertex joined across the smooth edges
between them (an edge of exactly two smooth faces, wound alike, not marked sharp). On any shape of the mesh a fan has
  lnor   the normalised sum, over the fan's corners, of each face's unit normal (its vector area's direction) times the
         face's angle at the corner
  ra     the mean angle between lnor and the fan's edges (unit vectors from its vertex)
  ref    a reference edge of the fan, flattened onto the plane square to lnor; ortho = lnor x ref
  span   the angle round lnor from ref to another edge of the fan (an open fan's other boundary edge), a whole turn when
         the two are one edge (a closed fan)
and a corner's custom normal keeps its angle from lnor in units of ra (alpha) and its angle round lnor from ref in
units of span (beta) as the shape changes. `fit` finds alpha and beta from the rest shape and picks each fan's
reference edge as the one that lands on Blender's own corner normals on a few deformed shapes (measured on an MMD
model: the 99th percentile of the angle to Blender's normals drops from 18 deg, rest normals turned by the skin, to
0.02 deg); `normals` rebuilds the normals of any shape from them."""
import numpy as np

TURN = 2 * np.pi


def fans(cv, cpoly, ce, start, tot, sharp, smooth):
    """Blender's corner fans from a mesh's arrays: corner vertex `cv`, corner face `cpoly`, corner edge `ce`, face
    `start` corner and corner count `tot`, per edge `sharp`, per face `smooth`. (fan of each corner, how many, smooth
    per edge, each corner's next and previous corner in its face)."""
    nl, ne = len(cv), len(sharp)
    at = np.arange(nl) - start[cpoly]
    cnext = start[cpoly] + (at + 1) % tot[cpoly]
    cprev = start[cpoly] + (at - 1) % tot[cpoly]
    order = np.argsort(ce, kind="stable")
    s = ce[order]
    pair = np.flatnonzero((s[:-1] == s[1:]) & (np.bincount(ce, minlength=ne)[s[:-1]] == 2))
    c1, c2 = order[pair], order[pair + 1]                # the two corners whose edge it is, one in each face
    ok = ~sharp[ce[c1]] & smooth[cpoly[c1]] & smooth[cpoly[c2]] & (cv[c1] == cv[cnext[c2]])
    c1, c2 = c1[ok], c2[ok]
    smooth_e = np.zeros(ne, bool)
    smooth_e[ce[c1]] = True
    links = np.concatenate([np.stack([c1, cnext[c2]], 1), np.stack([cnext[c1], c2], 1)])
    label = np.arange(nl)
    while len(links):                                    # each fan labelled by its least corner
        m = np.minimum(label[links[:, 0]], label[links[:, 1]])
        new = label.copy()
        np.minimum.at(new, links[:, 0], m)
        np.minimum.at(new, links[:, 1], m)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    fan = np.unique(label, return_inverse=True)[1].ravel()
    return fan, int(fan.max()) + 1, smooth_e, cnext, cprev


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def _angle(a, b):
    return np.arccos(np.clip(np.einsum("ij,ij->i", _unit(a), _unit(b)), -1.0, 1.0))


def lnor(P, face_n, corner_face, corner_v, corner_prev, corner_next, corner_fan, nfan):
    """Per fan, the normalised sum over its corners (face, vertex, previous and next vertex in the face) of the face's
    unit normal `face_n` times the face's angle at the corner; positions P."""
    w = face_n[corner_face] * _angle(P[corner_prev] - P[corner_v], P[corner_next] - P[corner_v])[:, None]
    return _unit(np.stack([np.bincount(corner_fan, w[:, j], nfan) for j in range(3)], 1))


def edge_angle(P, ln, v, edge_fan, edge_far, nfan):
    """Per fan, the mean angle between its lnor and its edges (from point v[fan] to P[edge_far])."""
    ang = _angle(P[edge_far] - P[v[edge_fan]], ln[edge_fan])
    return np.bincount(edge_fan, ang, nfan) / np.maximum(np.bincount(edge_fan, minlength=nfan), 1)


def frames(P, ln, v, ref, other):
    """Frames at points v, one per row: (ref, ortho = lnor x ref, span: the angle round lnor from ref to other)."""
    def flat(w):
        d = P[w] - P[v]
        return _unit(d - ln * np.einsum("ij,ij->i", d, ln)[:, None])
    r, oth = flat(ref), flat(other)
    o = np.cross(ln, r)
    d = np.clip(np.einsum("ij,ij->i", r, oth), -1.0, 1.0)
    span = np.where(d >= 1.0 - 1e-6, TURN, np.where(np.einsum("ij,ij->i", o, oth) < 0.0, TURN - np.arccos(d), np.arccos(d)))
    return r, o, span


def custom(ln, ra, r, o, span, alpha, beta):
    """Custom normals rebuilt in their frames: alpha in units of ra (negative: past it, counted from a whole turn) and
    beta in units of span."""
    a = np.where(alpha >= 0.0, alpha * ra, TURN + alpha * (TURN - ra))
    b = beta * span
    return _unit(np.cos(a)[:, None] * ln + np.sin(a)[:, None] * (np.cos(b)[:, None] * r + np.sin(b)[:, None] * o))


def _face_normals(P, cv, cnext, cpoly, nface):
    """Faces' unit normals: their vector area (sum over corners of p x p_next) normalised."""
    w = np.cross(P[cv], P[cv[cnext]])
    return _unit(np.stack([np.bincount(cpoly, w[:, j], nface) for j in range(3)], 1))


def fit(rest, rest_cn, cv, ce, cpoly, nface, fan, nfan, smooth_e, cnext, cprev, shapes):
    """Alpha and beta per corner and each fan's frame for a mesh at rest (positions `rest` per vertex, Blender's corner
    normals `rest_cn`; each corner's vertex, edge and face; fans from `fans`). The reference edge is, of the fan's edges
    (an open fan's: its two boundary edges), the one whose rebuild lands nearest Blender's corner normals on the
    deformed `shapes` [(positions per vertex, corner normals)]; on a tie (or none), the edge from the fan's first corner
    to the next vertex (an open fan's: the boundary edge before its first corner). A dict: per fan `vertex`, `ref`,
    `other` (vertices), `edge_fan` and `edge_far` (its edges, by fan); per corner `alpha` and `beta`."""
    nl = len(cv)
    c0 = np.full(nfan, nl)
    np.minimum.at(c0, fan, np.arange(nl))
    v = cv[c0]
    fan2, far2, e2 = np.r_[fan, fan], np.r_[cv[cnext], cv[cprev]], np.r_[ce, ce[cprev]]
    _, at = np.unique(fan2.astype(np.int64) * len(rest) + far2, return_index=True)     # each fan's edges, by far end
    edge_fan, edge_far, bnd = fan2[at], far2[at], ~smooth_e[e2[at]]
    open_ = np.bincount(edge_fan[bnd], minlength=nfan) == 2              # an open fan: two boundary edges
    whole, two = ~open_[edge_fan], bnd & open_[edge_fan]
    bf, bv = edge_fan[two], edge_far[two]
    cf = np.concatenate([edge_fan[whole], bf[0::2], bf[0::2]])          # candidates: fan, ref, other
    cref = np.concatenate([edge_far[whole], bv[0::2], bv[1::2]])
    coth = np.concatenate([edge_far[whole], bv[1::2], bv[0::2]])
    inc = ~smooth_e[ce[cprev]]
    c_in = np.full(nfan, nl)
    np.minimum.at(c_in, fan[inc], np.flatnonzero(inc))
    first_ref = np.where(open_, cv[cprev[np.minimum(c_in, nl - 1)]], cv[cnext[c0]])

    def shape(P):
        ln = lnor(P, _face_normals(P, cv, cnext, cpoly, nface), cpoly, cv, cv[cprev], cv[cnext], fan, nfan)
        return ln, edge_angle(P, ln, v, edge_fan, edge_far, nfan)
    ln0, ra0 = shape(rest)
    cust = rest_cn.astype(float)
    alpha0 = _angle(cust, ln0[fan])
    alpha = np.where(alpha0 <= ra0[fan], alpha0 / np.maximum(ra0[fan], 1e-12), -(TURN - alpha0) / (TURN - ra0[fan]))
    r, o, span = frames(rest, ln0[cf], v[cf], cref, coth)
    first = cust[c0[cf]]
    beta_c = (np.arctan2(np.einsum("ij,ij->i", o, first), np.einsum("ij,ij->i", r, first)) % TURN) / span
    err = np.zeros(len(cf))
    for co, cn in shapes:
        ln, ra = shape(co)
        rk, ok, sk = frames(co, ln[cf], v[cf], cref, coth)
        n = custom(ln[cf], ra[cf], rk, ok, sk, alpha[c0[cf]], beta_c)
        err += np.nan_to_num(_angle(n, cn[c0[cf]]), nan=np.pi)
    pick = np.lexsort((cref != first_ref[cf], np.round(err, 6), cf))     # per fan: least error, then the first edge
    pick = pick[np.r_[True, cf[pick][1:] != cf[pick][:-1]]]
    ref, other = cref[pick], coth[pick]
    r, o, span = frames(rest, ln0, v, ref, other)                       # each corner's beta in its fan's frame
    beta = (np.arctan2(np.einsum("ij,ij->i", o[fan], cust), np.einsum("ij,ij->i", r[fan], cust)) % TURN) / span[fan]
    return {"vertex": v, "ref": ref, "other": other, "edge_fan": edge_fan, "edge_far": edge_far, "alpha": alpha,
            "beta": beta}


def normals(P, tris, b):
    """The rebuild a renderer does on every frame: the normal of each vertex of a mesh whose vertices are at P (n, 3)
    with triangles `tris` (m, 3), from the bake's bend arrays (bake_lit's MESH `bend`): per vertex `fan`, `alpha`,
    `beta`; per fan `center`, `ref`, `other` (vertices), `corners` (face, previous vertex, next vertex) and `edges`
    (far-end vertices) as (offsets, items); per face its triangles `faces` as (offsets, items)."""
    nf = len(b["center"])
    tri_n = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    off, items = b["faces"]
    nface = len(off) - 1
    face_of = np.repeat(np.arange(nface), np.diff(off))
    face_n = _unit(np.stack([np.bincount(face_of, tri_n[items, j], nface) for j in range(3)], 1))
    c = b["corners"][1].reshape(-1, 3)
    corner_fan = np.repeat(np.arange(nf), np.diff(b["corners"][0]))
    ln = lnor(P, face_n, c[:, 0], b["center"][corner_fan], c[:, 1], c[:, 2], corner_fan, nf)
    edge_fan = np.repeat(np.arange(nf), np.diff(b["edges"][0]))
    ra = edge_angle(P, ln, b["center"], edge_fan, b["edges"][1], nf)
    r, o, span = frames(P, ln, b["center"], b["ref"], b["other"])
    f = b["fan"]
    return custom(ln[f], ra[f], r[f], o[f], span[f], b["alpha"], b["beta"])
