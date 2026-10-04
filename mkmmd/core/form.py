"""Form: how blocky is a piece of modelled geometry? (docs/design.md: Checks, `form`; docs/AGENTS.md: Modelling props.)

Pure numpy on triangles in world space: no bpy, so the maths runs and is tested without Blender (and `core` also loads
inside Blender, which has no scipy). The pipeline:

  weld        vertices closer than TOL are one (never across objects); degenerate triangles are dropped
  adjacency   triangles that share an edge; every such edge gets the angle between the two normals
  parts       loose parts: triangles connected through shared vertices
  patches     planar regions grown over shared edges while the normal stays within PATCH_DEG of the patch's mean
  numbers     per part and overall, area weighted:
                cuboid_share     boxes: a part whose surface is (nearly) all FLAT faces on three orthogonal axes, with real
                                 surface on at least two of them (a thin plate is a slab, not a box). Faces within
                                 AXIS_FULL degrees of an axis count fully, from AXIS_ZERO degrees away not at all, so a
                                 crowned panel or a big fillet is not a box and a bevelled cuboid still is one
                flat_share       area in large planar patches; a patch is large from (BIG_FRAC x the diameter)^2 up,
                                 i.e. a flat panel at the size the prop is looked at
                hard_edge_share  length share of the edges that border a large patch and turn more than HARD_DEG: no
                                 bevel strip between the faces (a chamfer turns 45 degrees per edge, a cube corner 90)
                flat_hard_share  area of large flat patches weighted by the hard share of their own rim: flat panels with
                                 sharp edges, which is what reads as blocky (a table top with a rounded rim does not)
  score       per part 1 - (1 - cuboid)(1 - flat_hard); overall their area-weighted mean. 0 = nothing blocky, 1 = every
              surface is a box or a sharp-edged slab

Sizes are metres. `analyse` is the entry point; the steps are public for the tests."""
import math

import numpy as np

TOL = 1e-5                  # weld distance (m)
MIN_AREA = 1e-9             # triangles smaller than this (m^2) are degenerate
PATCH_DEG = 3.0             # planar patch: neighbour normals within this angle of the patch's mean
HARD_DEG = 65.0             # an edge turning more than this is hard (bevels turn 45 and less per edge)
BIG_FRAC = 0.12             # a patch is large from (BIG_FRAC x the prop's diameter)^2
AXIS_FULL, AXIS_ZERO = 3.0, 10.0     # a face within AXIS_FULL degrees of an axis counts fully, from AXIS_ZERO not at all
BOX_SHARE = (0.60, 0.85)    # share of a part's surface on its three axes: not a box below, a full box above
BOX_SECOND = (0.10, 0.22)   # share on the second strongest axis: below it the part is a plate or a rod, not a box
MIN_PART = 1e-4             # parts below this share of diameter^2 are too small to be tested for a box
EXEMPT_WEIGHT = 0.0         # weight of exempt parts (1 = like every other part)
NEAR_PERP = 0.26            # a part's second axis comes from patches within this |cos| of perpendicular to the first


def _ramp(x, lo, hi):
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def diameter(P):
    """Longest distance between points (n,3), by farthest-point passes: within a few per cent, and independent of how
    the prop is turned in the world."""
    if len(P) == 0:
        return 0.0
    a = P[np.argmax(np.linalg.norm(P - (P.min(0) + P.max(0)) / 2, axis=1))]
    b = P[np.argmax(np.linalg.norm(P - a, axis=1))]
    return float(np.linalg.norm(P[np.argmax(np.linalg.norm(P - b, axis=1))] - b))


# ---------------------------------------------------------------- mesh
class Mesh:
    """Welded triangles: corners P (m,3,3), unit normals n (m,3), areas (m,), welded vertex ids W (m,3), owner (m,)
    (object index), `index` (m,) the position of each kept triangle in the input."""

    def __init__(self, V, T, owner=None, tol=TOL, min_area=MIN_AREA):
        V = np.asarray(V, float).reshape(-1, 3)
        T = np.asarray(T, np.int64).reshape(-1, 3)
        owner = np.zeros(len(T), np.int64) if owner is None else np.asarray(owner, np.int64)
        P = V[T]
        cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        a2 = np.linalg.norm(cr, axis=1)
        vown = np.zeros(len(V), np.int64)
        vown[T[:, 0]] = owner
        key = np.concatenate([np.round(V / tol).astype(np.int64), vown[:, None]], 1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        W = np.asarray(inv).reshape(-1)[T]
        ok = (a2 / 2 > min_area) & (W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 2] != W[:, 0])
        self.index = np.flatnonzero(ok)
        self.P, self.W, self.owner = P[ok], W[ok], owner[ok]
        self.n = cr[ok] / a2[ok, None]
        self.area = a2[ok] / 2
        self.nw = int(W.max()) + 1 if len(W) else 0

    def __len__(self):
        return len(self.area)


def adjacency(mesh):
    """Edges shared by exactly two triangles, and what is left over. A dict: t0, t1 (the two triangles), length (m),
    angle (degrees between their normals: 0 flat, 90 a box corner), nbr (m,3: the triangle across each edge or -1),
    border (length of edges with one triangle), nonmanifold (number of edges with three or more)."""
    W, P = mesh.W, mesh.P
    m = len(W)
    if m == 0:
        z = np.zeros(0)
        return {"t0": z.astype(int), "t1": z.astype(int), "length": z, "angle": z, "nbr": np.full((0, 3), -1),
                "border": 0.0, "nonmanifold": 0}
    a, b = W.reshape(-1), np.roll(W, -1, axis=1).reshape(-1)
    key = np.minimum(a, b) * mesh.nw + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[first, len(ks)])
    tri, slot = order // 3, order % 3

    def edge_len(at):
        return np.linalg.norm(P[tri[at], (slot[at] + 1) % 3] - P[tri[at], slot[at]], axis=1)

    two = first[count == 2]
    t0, t1 = tri[two], tri[two + 1]
    nbr = np.full((m, 3), -1, np.int64)
    nbr[t0, slot[two]] = t1
    nbr[t1, slot[two + 1]] = t0
    dot = np.clip(np.einsum("ij,ij->i", mesh.n[t0], mesh.n[t1]), -1.0, 1.0)
    return {"t0": t0, "t1": t1, "length": edge_len(two), "angle": np.degrees(np.arccos(dot)), "nbr": nbr,
            "border": float(edge_len(first[count == 1]).sum()), "nonmanifold": int((count > 2).sum())}


def components(mesh):
    """Loose parts: a label (m,) per triangle; triangles are connected through shared welded vertices."""
    if len(mesh) == 0:
        return np.zeros(0, np.int64)
    parent = np.arange(mesh.nw)
    e = np.concatenate([mesh.W[:, [0, 1]], mesh.W[:, [1, 2]]])
    while True:
        pu, pv = parent[e[:, 0]], parent[e[:, 1]]
        lo, hi = np.minimum(pu, pv), np.maximum(pu, pv)
        live = lo != hi
        if not live.any():
            break
        lo, hi = lo[live], hi[live]
        o = np.lexsort((lo, hi))
        hi, lo = hi[o], lo[o]
        head = np.r_[True, hi[1:] != hi[:-1]]
        parent[hi[head]] = np.minimum(parent[hi[head]], lo[head])
        while True:                                     # pointer jumping
            up = parent[parent]
            if (up == parent).all():
                break
            parent = up
    return np.unique(parent[mesh.W[:, 0]], return_inverse=True)[1].reshape(-1)


def patches(mesh, nbr, tol_deg=PATCH_DEG):
    """Planar patches: (label (m,), count). Region growing from the biggest unlabelled triangle over shared edges; a
    neighbour joins while its normal is within `tol_deg` of the patch's running mean (so a finely tessellated curve does
    not chain into one patch)."""
    m = len(mesh)
    cos_t = math.cos(math.radians(tol_deg))
    nx, ny, nz = (mesh.n[:, k].tolist() for k in range(3))
    area, nb = mesh.area.tolist(), nbr.tolist()
    label = [-1] * m
    k = 0
    for seed in np.argsort(-mesh.area, kind="stable").tolist():
        if label[seed] >= 0:
            continue
        label[seed] = k
        sx, sy, sz = nx[seed] * area[seed], ny[seed] * area[seed], nz[seed] * area[seed]
        mag = math.sqrt(sx * sx + sy * sy + sz * sz)
        mx, my, mz = sx / mag, sy / mag, sz / mag
        stack = [seed]
        while stack:
            for u in nb[stack.pop()]:
                if u < 0 or label[u] >= 0 or nx[u] * mx + ny[u] * my + nz[u] * mz < cos_t:
                    continue
                label[u] = k
                stack.append(u)
                sx, sy, sz = sx + nx[u] * area[u], sy + ny[u] * area[u], sz + nz[u] * area[u]
                mag = math.sqrt(sx * sx + sy * sy + sz * sz)
                mx, my, mz = sx / mag, sy / mag, sz / mag
        k += 1
    return np.array(label, np.int64), k


def patch_table(mesh, label, count):
    """Area (count,), mean unit normal (count,3) and centre (count,3) of every patch."""
    area = np.bincount(label, mesh.area, count)
    nrm = np.stack([np.bincount(label, mesh.n[:, d] * mesh.area, count) for d in range(3)], 1)
    mid = mesh.P.mean(1)
    cen = np.stack([np.bincount(label, mid[:, d] * mesh.area, count) for d in range(3)], 1)
    return area, nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12), cen / area[:, None]


# ---------------------------------------------------------------- boxes
def axis_frame(n, w, pn, pa, refine=2):
    """The orthonormal frame (3,3) a box-like part's faces line up with, or None: the biggest patch's normal, the biggest
    patch perpendicular to it, their cross product; refined by the area-weighted mean of the normals near each axis."""
    order = np.argsort(-pa, kind="stable")
    e1 = pn[order[0]]
    perp = order[np.abs(pn[order] @ e1) < NEAR_PERP]
    if len(perp) == 0:
        return None
    e2 = pn[perp[0]] - e1 * (e1 @ pn[perp[0]])
    e2 = e2 / np.linalg.norm(e2)
    E = np.stack([e1, e2, np.cross(e1, e2)])
    cos_near = math.cos(math.radians(AXIS_ZERO))
    for _ in range(refine):
        d = n @ E.T
        new = E.copy()
        for k in range(3):
            near = np.abs(d[:, k]) >= cos_near
            if near.any():
                v = (w[near, None] * np.sign(d[near, k])[:, None] * n[near]).sum(0)
                if np.linalg.norm(v) > 1e-12:
                    new[k] = v / np.linalg.norm(v)
        e1 = new[0]
        e2 = new[1] - e1 * (e1 @ new[1])
        e2 = e2 / max(np.linalg.norm(e2), 1e-12)
        E = np.stack([e1, e2, np.cross(e1, e2)])
    return E


def box_weight(n, w, pn, pa):
    """(weight 0..1, axis shares (3,) strongest first, frame or None): how much a part is a box. Its surface facing along
    three orthogonal axes (full within AXIS_FULL degrees of an axis, none from AXIS_ZERO) must make up most of it, and
    the second strongest axis must carry real surface: a plate has one dominant axis, a beam two, a box three."""
    E = axis_frame(n, w, pn, pa)
    if E is None:
        return 0.0, np.zeros(3), None
    d = np.abs(n @ E.T)
    which = d.argmax(1)
    ang = np.degrees(np.arccos(np.clip(d.max(1), 0.0, 1.0)))
    g = np.clip((AXIS_ZERO - ang) / (AXIS_ZERO - AXIS_FULL), 0.0, 1.0) * w
    share = np.sort(np.array([g[which == k].sum() for k in range(3)]) / w.sum())[::-1]
    return _ramp(share.sum(), *BOX_SHARE) * _ramp(share[1], *BOX_SECOND), share, E


# ---------------------------------------------------------------- analysis
def rim_hardness(adj, label, count, hard_deg):
    """Per patch, the share of the length of its edges to other patches that turns more than `hard_deg`; and the masks
    (per shared edge) of all patch-to-patch edges and of the hard ones."""
    p0, p1 = label[adj["t0"]], label[adj["t1"]]
    cross = p0 != p1
    hard = cross & (adj["angle"] > hard_deg)
    ln = adj["length"]
    perim = np.bincount(p0[cross], ln[cross], count) + np.bincount(p1[cross], ln[cross], count)
    hlen = np.bincount(p0[hard], ln[hard], count) + np.bincount(p1[hard], ln[hard], count)
    return np.divide(hlen, perim, out=np.zeros(count), where=perim > 0), cross, hard


def analyse(V, T, owner=None, names=None, exempt=(), exempt_weight=EXEMPT_WEIGHT, big_frac=BIG_FRAC,
            hard_deg=HARD_DEG, patch_deg=PATCH_DEG, worst=8):
    """Blockiness of triangles in world space. V (n,3), T (m,3), owner (m,) index into `names` (default one object),
    `exempt` the names counted at `exempt_weight` (default 0: measured and listed, not scored). A dict:
      score, cuboid_share, flat_share, hard_edge_share, flat_hard_share   the numbers (see the module doc)
      area, size (axis-aligned extents), diameter, big_area, triangles, ignored (degenerate), parts, objects
      components   [{object, part, area, share, cuboid, flat_hard, score, axes, size (box extents), at}] worst first
      patches      [{object, area, share, rim, normal, at}] the biggest flat patches with sharp rims
      by_object    {name: {area, share, parts, cuboid, flat_hard, score}} ranked by what each adds to the score
      exempt       {name: {area, parts, cuboid, flat_hard, score}} for the exempt objects"""
    mesh = Mesh(V, T, owner)
    owner = mesh.owner
    n_obj = int(max(len(names or []), (owner.max() + 1) if len(owner) else 0, 1))
    names = list(names) if names else [f"object{i}" for i in range(n_obj)]
    out = {"score": 0.0, "cuboid_share": 0.0, "flat_share": 0.0, "hard_edge_share": 0.0, "flat_hard_share": 0.0,
           "area": 0.0, "size": [0.0, 0.0, 0.0], "diameter": 0.0, "big_area": 0.0, "triangles": int(len(mesh)),
           "ignored": int(len(np.asarray(T).reshape(-1, 3)) - len(mesh)), "parts": 0, "objects": len(names),
           "components": [], "patches": [], "by_object": {}, "exempt": {}}
    if len(mesh) == 0:
        return out
    exempt = set(exempt)
    w_obj = np.array([exempt_weight if nm in exempt else 1.0 for nm in names])
    if not (w_obj[np.unique(owner)] > 0).any():
        w_obj = np.ones(len(names))                      # nothing but exempt parts: measure them all
    wt = w_obj[owner]                                    # per triangle
    diam = diameter(mesh.P[wt > 0].reshape(-1, 3))
    big_area = (big_frac * diam) ** 2
    adj = adjacency(mesh)
    comp = components(mesh)
    label, npatch = patches(mesh, adj["nbr"], patch_deg)
    pa, pn, pc = patch_table(mesh, label, npatch)
    large = pa >= big_area
    rim, cross, hard = rim_hardness(adj, label, npatch, hard_deg)
    ln, ew = adj["length"], np.minimum(wt[adj["t0"]], wt[adj["t1"]])
    touch = cross & (large[label[adj["t0"]]] | large[label[adj["t1"]]])             # edges bordering a large patch
    hard_edge = float((ln * ew)[touch & hard].sum() / max((ln * ew)[touch].sum(), 1e-12))

    # parts
    order = np.argsort(comp, kind="stable")
    bounds = np.searchsorted(comp[order], np.arange(comp.max() + 2))
    nc = len(bounds) - 1
    ca = np.bincount(comp, mesh.area, nc)
    flat_area, flat_hard_area, kappa = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    size_c, at_c, axes_c = np.zeros((nc, 3)), np.zeros((nc, 3)), np.zeros((nc, 3))
    obj_c, part_c = np.zeros(nc, np.int64), np.zeros(nc, np.int64)
    seen = {}
    min_part = MIN_PART * diam ** 2
    for c in range(nc):
        tris = order[bounds[c]:bounds[c + 1]]
        obj_c[c] = owner[tris[0]]
        part_c[c] = seen[obj_c[c]] = seen.get(obj_c[c], -1) + 1
        pid = np.unique(label[tris])
        big = pid[large[pid]]
        flat_area[c], flat_hard_area[c] = pa[big].sum(), (pa[big] * rim[big]).sum()
        pts_c = mesh.P[tris].reshape(-1, 3)
        lo, hi = pts_c.min(0), pts_c.max(0)
        size_c[c], at_c[c] = np.sort(hi - lo)[::-1], (lo + hi) / 2
        if ca[c] >= min_part and len(tris) >= 4:
            kappa[c], axes_c[c], E = box_weight(mesh.n[tris], mesh.area[tris], pn[pid], pa[pid])
            if E is not None:
                size_c[c] = np.sort(np.ptp(pts_c @ E.T, axis=0))[::-1]
    cw = w_obj[obj_c]
    flat_hard = np.minimum(np.divide(flat_hard_area, ca, out=np.zeros(nc), where=ca > 0), 1.0)
    score_c = 1.0 - (1.0 - kappa) * (1.0 - flat_hard)
    total = float((ca * cw).sum())
    out.update(area=float(ca.sum()), size=[float(x) for x in np.ptp(mesh.P.reshape(-1, 3), axis=0)], diameter=diam,
               big_area=float(big_area), parts=int(nc), hard_edge_share=hard_edge,
               cuboid_share=float((ca * cw * kappa).sum() / total), flat_share=float((flat_area * cw).sum() / total),
               flat_hard_share=float((flat_hard_area * cw).sum() / total),
               score=float((ca * cw * score_c).sum() / total))
    for c in np.argsort(-(ca * cw * score_c), kind="stable")[:worst]:
        if ca[c] * cw[c] * score_c[c] <= 0:
            break
        out["components"].append({"object": names[obj_c[c]], "part": int(part_c[c]), "area": float(ca[c]),
                                  "share": float(ca[c] * cw[c] / total), "cuboid": float(kappa[c]),
                                  "flat_hard": float(flat_hard[c]), "score": float(score_c[c]),
                                  "axes": [float(x) for x in axes_c[c]], "size": [float(x) for x in size_c[c]],
                                  "at": [float(x) for x in at_c[c]]})
    own_p = np.zeros(npatch, np.int64)
    own_p[label] = owner
    for p in np.argsort(-(pa * rim * w_obj[own_p]), kind="stable")[:worst]:
        if not (large[p] and rim[p] > 0 and w_obj[own_p[p]] > 0):
            break
        out["patches"].append({"object": names[own_p[p]], "area": float(pa[p]), "share": float(pa[p] / total),
                               "rim": float(rim[p]), "normal": [float(x) for x in pn[p]],
                               "at": [float(x) for x in pc[p]]})
    rows = []
    for j, nm in enumerate(names):
        mine = obj_c == j
        if not mine.any():
            continue
        a = float(ca[mine].sum())
        row = {"area": a, "share": float(a * w_obj[j] / total), "parts": int(mine.sum()),
               "cuboid": float((ca[mine] * kappa[mine]).sum() / a), "flat_hard": float(flat_hard_area[mine].sum() / a),
               "score": float((ca[mine] * score_c[mine]).sum() / a)}
        if nm in exempt:
            out["exempt"][nm] = {k: v for k, v in row.items() if k != "share"}
        else:
            rows.append((row["share"] * row["score"], nm, row))
    out["by_object"] = {nm: row for _, nm, row in sorted(rows, key=lambda r: -r[0])}
    return out
