"""A garment of a model mk did not build, made into a cloth copy for mk build's cloth stage (numpy only, Blender-safe;
docs/design.md: Cloth). The stage reads the garment's faces off the cast member's own mesh and hands them over as arrays
in the armature's space at rest: metres, z up, the model facing -y, her left +x.

A garment modelled with thickness (an outside and a lining, a closed shell: most MMD outfits) is simulated by its
outside. `lining` finds the inside: the faces wound toward the body (their normal toward the nearest bone) that another
face of the same piece covers, within a few millimetres outward. A face wound outward stays even where something lies on
it (a pocket flap, a lapped closure, the top of a sleeve tucked into the body), so the cloth has no holes under the
details. Of what stays, the largest sheet (the largest welded piece's largest connected part) is the cloth; the rest
(a collar laid on the lapels, a flap, every lining) follows it through the stage's binding. `make` builds the copy's
arrays in the format of mkmmd.model.cloth, its faces triangles."""
import numpy as np


def weld(V, tol=1e-5):
    """(index (n,) of each vertex's welded vertex, the welded positions (m, 3)): vertices at one place (to `tol` m) are one."""
    V = np.asarray(V, float)
    q = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return inv.reshape(-1), V[first]


def collapse(V, T, short):
    """(index (n,) of each vertex's vertex, positions (m, 3), triangles (f', 3)): the vertices that edges of T shorter
    than `short` metres join are one, at their mean (slivers a modeller left make a cloth stiff and jittery there); the
    faces that collapse or repeat another are dropped."""
    V, T = np.asarray(V, float), np.asarray(T, np.int64)
    E = tri_edges(T)
    E = E[np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) < short]
    root = np.arange(len(V))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)
    ids, idx = np.unique([find(i) for i in range(len(V))], return_inverse=True)
    P = np.zeros((len(ids), 3))
    np.add.at(P, idx, V)
    P /= np.bincount(idx, minlength=len(ids))[:, None]
    Tn = idx[T]
    Tn = Tn[(Tn[:, 0] != Tn[:, 1]) & (Tn[:, 1] != Tn[:, 2]) & (Tn[:, 0] != Tn[:, 2])]
    _, first = np.unique(np.sort(Tn, 1), axis=0, return_index=True)
    return idx, P, Tn[np.sort(first)]


def face_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def vertex_normals(V, T):
    """Unit area-weighted vertex normals of triangles (as wound)."""
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        np.add.at(N, T[:, i], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-15)


def tri_edges(T):
    """(e, 2) the unique edges of triangles (f, 3), each sorted."""
    T = np.asarray(T)
    return np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)


def pieces(T, n):
    """(n,) a label per vertex: vertices joined by the triangles share one (the smallest index of the piece)."""
    lab = np.arange(n)
    E = tri_edges(T)
    if not len(E):
        return lab
    while True:
        m = np.minimum(lab[E[:, 0]], lab[E[:, 1]])
        new = lab.copy()
        np.minimum.at(new, E[:, 0], m)
        np.minimum.at(new, E[:, 1], m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def seg_dist(P, A, B):
    """(n, k) distances of points P (n, 3) to the segments A_k B_k."""
    P, A, B = (np.asarray(x, float) for x in (P, A, B))
    AB = B - A
    L2 = np.maximum(np.einsum("kj,kj->k", AB, AB), 1e-18)
    t = np.clip(np.einsum("nkj,kj->nk", P[:, None] - A[None], AB) / L2, 0.0, 1.0)
    return np.linalg.norm(P[:, None] - (A[None] + t[..., None] * AB[None]), axis=2)


def closest_on_seg(P, A, B):
    """(n, 3) the point of segment A_i B_i nearest each point P_i (elementwise: all three (n, 3))."""
    AB = B - A
    t = np.clip(np.einsum("nj,nj->n", P - A, AB) / np.maximum(np.einsum("nj,nj->n", AB, AB), 1e-18), 0.0, 1.0)
    return A + t[:, None] * AB


def _ray_t(O, D, V0, E1, E2):
    """(b, f) the distance along D (b, 3) from O (b, 3) at which each ray meets each triangle (V0, V0 + E1, V0 + E2),
    NaN where it misses (Moller-Trumbore, both directions)."""
    p = np.cross(D[:, None, :], E2[None])
    det = np.einsum("bfj,fj->bf", p, E1)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    s = O[:, None, :] - V0[None]
    u = np.einsum("bfj,bfj->bf", s, p) * inv
    q = np.cross(s, E1[None])
    v = np.einsum("bfj,bj->bf", q, D) * inv
    t = np.einsum("bfj,fj->bf", q, E2) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
    return np.where(hit, t, np.nan)


def outward(P, A, B):
    """(n, 3) unit directions away from the nearest of the bones (segments A..B) at the points P."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    k = np.argmin(seg_dist(P, A, B), 1)
    R = P - closest_on_seg(P, A[k], B[k])
    return R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-15)


def lining(V, T, A, B, labels=None, thick=0.003, chunk=96):
    """(f,) True for a garment's inside: a face wound toward the body (its normal toward the nearest bone, segments A..B)
    that a face of the same piece (`labels`, one per face) covers within `thick` metres outward."""
    V, T = np.asarray(V, float), np.asarray(T)
    f = len(T)
    lab = np.zeros(f, int) if labels is None else np.asarray(labels)
    C = V[T].mean(1)
    R = outward(C, A, B)
    inward = np.einsum("ij,ij->i", face_normals(V, T), R) < 0.0
    V0, E1, E2 = V[T[:, 0]], V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    covered = np.zeros(f, bool)
    ids = np.flatnonzero(inward)
    for s in range(0, len(ids), chunk):
        i = ids[s:s + chunk]
        t = _ray_t(C[i], R[i], V0, E1, E2)
        t[lab[i][:, None] != lab[None]] = np.nan                        # another piece lying on it is not its outside
        covered[i] = ((t > 1e-5) & (t <= thick)).any(1)
    return inward & covered


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _area(V, T):
    return 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)


def _largest(V, T, keep):
    """The faces `keep` of T restricted to their largest connected part (by area)."""
    lab = pieces(T[keep], len(V))[T[keep][:, 0]]
    ids, inv = np.unique(lab, return_inverse=True)
    best = ids[np.argmax(np.bincount(inv.reshape(-1), weights=_area(V, T[keep])))]
    out = np.zeros(len(T), bool)
    out[np.flatnonzero(keep)[lab == best]] = True
    return out


def manifold(V, T, keep):
    """`keep` (f,) less the faces that make an edge of T[keep] carry more than two: on each such edge the two largest
    stay (what Surface Deform binds to and Blender's cloth expects; the others still follow through the binding)."""
    keep = np.asarray(keep, bool).copy()
    area = _area(V, T)
    while True:
        ids = np.flatnonzero(keep)
        E = np.sort(np.stack([T[ids], np.roll(T[ids], -1, axis=1)], 2).reshape(-1, 2), 1)
        owner = np.repeat(ids, 3)
        _, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        bad = np.flatnonzero(cnt[inv] > 2)
        if not len(bad):
            return keep
        for e in np.unique(inv[bad]):
            faces = owner[inv == e]
            keep[faces[np.argsort(-area[faces])][2:]] = False


def open_front(S, F, free, deg, below=None, fade=0.06, band=0.02, full=20.0, side=90.0):
    """(n, 3) the cloth copy's points S (faces F) as an unbuttoned jacket hangs: each point of a front panel slides round
    the torso toward its panel's side, about the vertical line through the middle of the free cloth at its height
    (within `band` metres), by `deg` degrees times its weight, so it stays as far from the body and nothing comes
    forward: `free` (n,) 0..1 (0: a pinned shoulder, a sleeve), the whole of it within `full` degrees of straight ahead
    and none from `side` degrees round, and with `below` (rest height, m) eased in over `fade` metres under it
    (unbuttoned below the top button: the lapels stay). A panel is a part of the free cloth in front of its middle that
    the faces join, on the side of its mean x: a lapped closure crosses x = 0 and its overlap goes with its own panel."""
    S, F = np.asarray(S, float), np.asarray(F)
    free = np.asarray(free, float)
    out = S.copy()
    torso = free > 0.0
    if below is not None:
        free = free * smoothstep((float(below) - S[:, 2]) / max(float(fade), 1e-9))
    if not torso.any():
        return out
    front = torso & (S[:, 1] < np.median(S[torso, 1]))
    lab = pieces(F[front[F].all(1)], len(S))
    sign = np.where(S[:, 0] >= 0.0, 1.0, -1.0)
    for p in np.unique(lab[front]):
        m = front & (lab == p)
        sign[m] = 1.0 if S[m, 0].mean() >= 0.0 else -1.0
    for i in np.flatnonzero(torso & (free > 0.0)):
        near = torso & (np.abs(S[:, 2] - S[i, 2]) < band)
        cx = 0.5 * (S[near, 0].min() + S[near, 0].max())
        cy = 0.5 * (S[near, 1].min() + S[near, 1].max())
        dx, dy = S[i, 0] - cx, S[i, 1] - cy
        round_ = abs(np.degrees(np.arctan2(dx, -dy)))                   # 0 straight ahead, 90 at her side
        w = 1.0 - smoothstep((round_ - float(full)) / max(float(side) - float(full), 1e-9))
        a = np.radians(float(deg)) * free[i] * w * sign[i]               # her left panel toward +x, her right toward -x
        out[i, 0] = cx + dx * np.cos(a) - dy * np.sin(a)
        out[i, 1] = cy + dx * np.sin(a) + dy * np.cos(a)
    return out


def orient(V, T, A, B):
    """T (f, 3) wound outward: each sheet's faces made to agree with their neighbours (two faces that share an edge cross
    it in opposite directions; an edge of more than two faces joins nothing), then each sheet turned as most of its area
    says (its normals away from the nearest of the bones A..B). A face whose nearest bone is the wrong one (the side of
    the torso just under an arm's bone) goes with its sheet, and a model wound either way comes out one way."""
    V, T = np.asarray(V, float), np.asarray(T, np.int64).copy()
    if not len(T):
        return T
    E = np.stack([T, np.roll(T, -1, axis=1)], -1).reshape(-1, 2)        # the faces' directed edges, three per face
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    pair = {}
    for i in np.flatnonzero(cnt[inv] == 2).tolist():
        pair.setdefault(int(inv[i]), []).append(i)
    nb = [[] for _ in range(len(T))]
    for i, j in pair.values():
        same = bool((E[i] == E[j]).all())                               # crossed the same way: one of them turns
        nb[i // 3].append((j // 3, same))
        nb[j // 3].append((i // 3, same))
    flip = np.zeros(len(T), bool)
    comp = np.full(len(T), -1)
    for s in range(len(T)):
        if comp[s] >= 0:
            continue
        comp[s] = c = comp.max() + 1
        stack = [s]
        while stack:
            f = stack.pop()
            for g, same in nb[f]:
                if comp[g] < 0:
                    comp[g], flip[g] = c, flip[f] ^ same
                    stack.append(g)
    T[flip] = T[flip][:, ::-1]
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])      # area-weighted
    vote = np.bincount(comp, weights=np.einsum("nj,nj->n", n, outward(V[T].mean(1), A, B)))
    turn = vote[comp] < 0.0
    T[turn] = T[turn][:, ::-1]
    return T


def make(V, T, W, A, B, arm, collider, pin=None, sleeves=None, inset=0.003, reach=0.15, thick=0.003, short=0.003):
    """The cloth copy's arrays (mkmmd.model.cloth's format, triangles for faces) of a garment: V (n, 3), T (f, 3) its
    vertices and triangles, W (n, k) their weights over the stage's bone list, A, B (j, 3) the bones' rest segments (what
    `lining` measures outward from), `arm` (k,) True for the bones of the arms, `collider` (Vc, Tc, Wc) the body under it.

    pin       {above, fade = 0.06}: carried by the armature above `above` metres (rest height), free below `above - fade`
    sleeves   {hold = 0.9, collides = false}: the vertices the arm bones carry most of hold that share toward their arm
              (a sleeve on an arm that moves faster than cloth) and collide (or not) with the body
    inset     the collider set in along its normals (m), so the cloth's rest shape is clear of it
    reach     the body further than this from the garment is left out of the collider (m)
    short     edges shorter than this (m) are collapsed (`collapse`)
    Returns {sim, sim_faces, pins, holds, collides, sim_weights, collide, collide_faces, collide_weights}."""
    V, T, W = np.asarray(V, float), np.asarray(T, np.int64), np.asarray(W, float)
    A, B = np.asarray(A, float), np.asarray(B, float)
    idx, Vw = weld(V)
    Tw = idx[T]
    Tw = Tw[(Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 0] != Tw[:, 2])]
    ci_, Vw, Tw = collapse(Vw, Tw, short)
    idx = ci_[idx]                                                     # a raw vertex -> its collapsed one
    flab = pieces(Tw, len(Vw))[Tw[:, 0]]
    F = Tw[_largest(Vw, Tw, manifold(Vw, Tw, ~lining(Vw, Tw, A, B, labels=flab, thick=thick)))]
    used, Fi = np.unique(F, return_inverse=True)
    Fi = Fi.reshape(F.shape)
    S = Vw[used]
    Fi = orient(S, Fi, A, B)                                           # every face wound outward
    Ww = np.zeros((len(Vw), W.shape[1]))
    np.add.at(Ww, idx, W)                                              # a welded vertex: the raw ones at that place
    Ws = Ww[used]
    Ws /= np.where(Ws.sum(1, keepdims=True) > 0.0, Ws.sum(1, keepdims=True), 1.0)
    n = len(S)
    pins = np.zeros(n)
    if pin:
        above, fade = float(pin["above"]), float(pin.get("fade", 0.06))
        pins = smoothstep((S[:, 2] - (above - fade)) / max(fade, 1e-9))
    holds, collides = np.zeros(n), np.ones(n, np.uint8)
    if sleeves:
        on = Ws[:, np.asarray(arm, bool)].sum(1) > 0.5
        holds[on] = float(sleeves.get("hold", 0.9))
        collides[on] = 1 if sleeves.get("collides", False) else 0
    # the body under it: welded, the faces near the garment wound outward (the cloth found inside a face goes back out
    # along its normal; models wind a double-sided skirt or a lined shirt either way), set in
    Vc, Tc, Wc = collider
    ci, Vcw = weld(Vc)
    Tcw = ci[np.asarray(Tc, np.int64)]
    Tcw = Tcw[(Tcw[:, 0] != Tcw[:, 1]) & (Tcw[:, 1] != Tcw[:, 2]) & (Tcw[:, 0] != Tcw[:, 2])]
    near = ((Vcw >= S.min(0) - reach) & (Vcw <= S.max(0) + reach)).all(1)
    cand = np.flatnonzero(near)
    for s in range(0, len(cand), 1024):
        c = cand[s:s + 1024]
        d = np.full(len(c), np.inf)
        for u in range(0, n, 1024):
            d = np.minimum(d, np.linalg.norm(Vcw[c][:, None] - S[None, u:u + 1024], axis=2).min(1))
        near[c] = d < reach
    Tcw = Tcw[near[Tcw].all(1)]
    cu, Tci = np.unique(Tcw, return_inverse=True)
    Tci = Tci.reshape(Tcw.shape)
    Vcu = Vcw[cu]
    Tci = orient(Vcu, Tci, A, B)
    Wcw = np.zeros((len(Vcw), W.shape[1]))
    np.add.at(Wcw, ci, np.asarray(Wc, float))
    Wcw = Wcw[cu]
    Wcw /= np.where(Wcw.sum(1, keepdims=True) > 0.0, Wcw.sum(1, keepdims=True), 1.0)
    Vcu = Vcu - vertex_normals(Vcu, Tci) * float(inset)
    return {"sim": S.astype(np.float32), "sim_faces": Fi.astype(np.int32), "pins": pins.astype(np.float32),
            "holds": holds.astype(np.float32), "collides": collides, "sim_weights": Ws.astype(np.float32),
            "collide": Vcu.astype(np.float32), "collide_faces": Tci.astype(np.int32),
            "collide_weights": Wcw.astype(np.float32)}


# ---------------------------------------------------------------------------------------------------------- binding

def face_edges(F):
    """(e, 2) the unique edges of faces F (f, k), each sorted: triangles, quads, or quads whose last index is -1."""
    F = np.asarray(F, np.int64)
    if not F.size:
        return np.zeros((0, 2), np.int64)
    nxt = np.roll(F, -1, axis=1)
    if F.shape[1] > 3:
        tri = F[:, -1] < 0
        nxt[tri, -2] = F[tri, 0]                                       # a padded triangle closes on its first corner
    E = np.stack([F, nxt], -1).reshape(-1, 2)
    E = E[(E >= 0).all(1)]
    return np.unique(np.sort(E, 1), axis=0)


def _adjacency(V, E):
    """Per vertex of V, its (neighbour, edge length) pairs along the edges E."""
    nb = [[] for _ in range(len(V))]
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) if len(E) else np.zeros(0)
    for (a, b), w in zip(np.asarray(E).tolist(), L.tolist()):
        nb[a].append((b, w))
        nb[b].append((a, w))
    return nb


def _grow(nb, seeds, labels, limit=np.inf):
    """(distance (n,), label (n,)): from the vertices `seeds`, each with its label, the shortest way along the edges
    (adjacency `nb`) to every vertex within `limit` metres, and the label of the seed it comes from (inf, -1 beyond)."""
    import heapq
    dist, lab = [np.inf] * len(nb), [-1] * len(nb)
    heap = []
    for s, label in zip(seeds, labels):
        if dist[s] > 0.0:
            dist[s], lab[s] = 0.0, int(label)
            heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j, w in nb[i]:
            dj = d + w
            if dj < dist[j] and dj <= limit:
                dist[j], lab[j] = dj, lab[i]
                heapq.heappush(heap, (dj, j))
    return np.array(dist), np.array(lab, np.int64)


def laps(S, F, free, near=0.02, far=0.15):
    """(m, 2) pairs (i < j) of the copy's free vertices (S (n, 3), faces F, `free` (n,) bool) that lie within `near`
    metres of each other but more than `far` metres apart along its edges: one part of the cloth lapped over another (a
    jacket's closure, a wrap skirt), where a point bound to the nearest of the cloth may follow the wrong part."""
    S = np.asarray(S, float)
    ids = np.flatnonzero(np.asarray(free, bool))
    if len(ids) < 2:
        return np.zeros((0, 2), np.int64)
    cand = []
    for s in range(0, len(ids), 256):
        a = ids[s:s + 256]
        d = np.linalg.norm(S[a][:, None] - S[ids][None], axis=2)
        r, c = np.nonzero((d < near) & (a[:, None] < ids[None]))
        cand.append(np.stack([a[r], ids[c]], 1))
    nb = _adjacency(S, face_edges(F))
    ring = [{j for j, _ in x} for x in nb]
    by = {}
    for i, j in np.concatenate(cand).tolist():
        if j not in ring[i] and not ring[i] & ring[j]:                  # not within two edges of each other
            by.setdefault(i, []).append(j)
    out = []
    for i, js in by.items():
        dist, _ = _grow(nb, [i], [0], limit=far)
        out += [(i, j) for j in js if dist[j] > far]
    return np.array(out, np.int64).reshape(-1, 2)


def lap_colours(S, F, pairs):
    """(n,) a colour per vertex of the copy (S (n, 3), faces F) such that the two ends of every lap (`laps`) differ: the
    ends the copy's edges join are one cluster, clusters that lap over each other take different colours (two where
    that is enough, else as many as it takes), and every other vertex the colour of the nearest end along the edges (0
    where none is). A pair whose ends are one cluster cannot be told apart and is left as it is."""
    S = np.asarray(S, float)
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    col = np.zeros(len(S), np.int64)
    if not len(pairs):
        return col
    E = face_edges(F)
    ends = np.unique(pairs)
    on = np.zeros(len(S), bool)
    on[ends] = True
    root = list(range(len(S)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b in E[on[E[:, 0]] & on[E[:, 1]]].tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)
    cl = {e: find(e) for e in ends.tolist()}
    nbr = {}
    for i, j in pairs.tolist():
        a, b = cl[i], cl[j]
        if a != b:
            nbr.setdefault(a, set()).add(b)
            nbr.setdefault(b, set()).add(a)
    colour = {}
    for c0 in sorted(nbr):                                              # breadth first: two colours when they do
        if c0 in colour:
            continue
        colour[c0], queue = 0, [c0]
        while queue:
            c = queue.pop(0)
            for d in sorted(nbr[c]):
                if d not in colour:
                    used = {colour[x] for x in nbr[d] if x in colour}
                    colour[d] = min(k for k in range(len(used) + 1) if k not in used)
                    queue.append(d)
    seeds = [e for e in ends.tolist() if cl[e] in colour]
    if not seeds:
        return col
    _, lab = _grow(_adjacency(S, E), seeds, [colour[cl[e]] for e in seeds])
    return np.maximum(lab, 0)


def bound_colours(R, E, on, at):
    """(r,) the colour of the cloth each vertex of a mesh bound to it follows (R (r, 3), its edges E): a vertex lying on
    the copy (`on` (r,) bool) takes the colour where it lies (`at` (r,)); any other, along the mesh's own edges, that of
    the nearest vertex lying on it (a lining its own side's, not that of the part lapped over it), or `at` where no
    edge leads to one."""
    on, at = np.asarray(on, bool), np.asarray(at, np.int64)
    seeds = np.flatnonzero(on)
    if not len(seeds):
        return at.copy()
    _, lab = _grow(_adjacency(np.asarray(R, float), np.asarray(E, np.int64).reshape(-1, 2)), seeds.tolist(),
                   at[seeds].tolist())
    return np.where(lab >= 0, lab, at)
