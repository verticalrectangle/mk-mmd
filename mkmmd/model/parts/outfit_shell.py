"""Garments cut out of the body's own surface (numpy only: Blender-safe, no scipy): a `Shell` is a piece of the skin
mesh (its triangles, normals and skin weights) chosen by fields on it and cut exactly along their level lines, then moved
out along the skin's normals, relaxed into cloth and kept off the skin. Skinny pants, socks, a knit sweater and its rib
bands are such shells: they share the skin's topology and weights, so they deform with it in every pose.

    sh = Shell.from_part(body)                       # the body part's skin, triangulated, with its weights
    sh = sh.clip(z - z_hem)                          # keep field >= 0, cut along field == 0 (new vertices on the edges)
    P = sh.offset(dist, relax=6, fit=fit, gap=0.002) # (n, 3): out along the normals, smoothed, at least `gap` off the skin
    patch = sh.patch(P, uv_u, uv_v, rim=...)         # an outfit_geo.Patch (faces, uv per corner, weights)

Fields are per-vertex arrays computed on `sh.verts` (heights, a limb's length coordinate `limb_coord`, bone shares
`share`); clipping interpolates every field, the normals and the weights along the cut edges, so a garment's edge is a
clean level line however coarse the skin is there. `rim` closes an open edge of a shell back towards the skin (a band's
rolled top, a sock's cuff) so the cloth reads thick instead of paper thin."""
import numpy as np

from . import outfit_fit as F
from . import outfit_geo as G


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def vertex_normals(V, T):
    """Area-weighted vertex normals (n, 3) of the triangles T."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return _unit(N)


def triangulate(faces):
    """Fan-split polygons (lists of vertex ids) into an (m, 3) int array, keeping the winding."""
    out = [(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)]
    return np.array(out, np.int64).reshape(-1, 3)


def soft_max(a, b, k):
    """A smooth max of two fields (a corner of radius about k where they cross)."""
    m = np.maximum(a, b)
    return m + k * np.log(np.exp((a - m) / k) + np.exp((b - m) / k))


class Shell:
    """A triangle mesh with per-vertex data carried through cuts: `verts` (n, 3), `tris` (m, 3), `normals` (n, 3) (the
    skin's, interpolated), `W` (n, B) skin weights over `bones`, `fields` {name: (n,)}."""

    def __init__(self, verts, tris, normals, bones, W, fields=None):
        self.verts = np.asarray(verts, float)
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.normals = np.asarray(normals, float)
        self.bones = list(bones)
        self.W = np.asarray(W, float).reshape(len(self.verts), len(self.bones))
        self.fields = {k: np.asarray(v, float) for k, v in (fields or {}).items()}

    @classmethod
    def from_part(cls, part, mesh=None):
        """The skin of a body Part: its first mesh with vertices (or the one named `mesh`), triangulated."""
        ms = [m for m in part.meshes if len(m.verts) and (mesh is None or m.name == mesh)]
        if not ms:
            raise ValueError(f"part {part.name!r} has no mesh {mesh or ''} to cut a garment from")
        m = ms[0]
        V = np.asarray(m.verts, float)
        T = triangulate(m.faces)
        bones = sorted(m.weights)
        W = np.stack([np.asarray(m.weights[b], float) for b in bones], 1) if bones else np.zeros((len(V), 0))
        return cls(V, T, vertex_normals(V, T), bones, W)

    def weights(self):
        """{bone: (n,)} with the zero columns left out."""
        return {b: self.W[:, i].copy() for i, b in enumerate(self.bones) if self.W[:, i].max() > 1e-6}

    def share(self, pred):
        """(n,) summed weight of the bones `pred(name)` accepts (e.g. outfit_fit.is_arm)."""
        cols = [i for i, b in enumerate(self.bones) if pred(b)]
        return self.W[:, cols].sum(1) if cols else np.zeros(len(self.verts))

    def clip(self, f):
        """The part of the shell where the field `f` (n,) is >= 0, cut along f == 0: triangles across the line are split
        and their new vertices interpolate positions, normals, weights and every field. Unused vertices are dropped."""
        f = np.asarray(f, float)
        inside = f >= 0.0
        V, N, W = [self.verts], [self.normals], [self.W]
        X = {k: [v] for k, v in self.fields.items()}
        n0 = len(self.verts)
        new = {}
        extra_v, extra_n, extra_w, extra_x = [], [], [], {k: [] for k in self.fields}

        def cut(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in new:
                t = f[i] / (f[i] - f[j])
                new[key] = n0 + len(extra_v)
                extra_v.append(self.verts[i] + t * (self.verts[j] - self.verts[i]))
                extra_n.append(self.normals[i] + t * (self.normals[j] - self.normals[i]))
                extra_w.append(self.W[i] + t * (self.W[j] - self.W[i]))
                for k, v in self.fields.items():
                    extra_x[k].append(v[i] + t * (v[j] - v[i]))
            return new[key]

        cnt = inside[self.tris].sum(1)
        keep = [self.tris[cnt == 3]]
        out = []
        for tri in self.tris[(cnt == 1) | (cnt == 2)]:
            ins = inside[tri]
            if ins.sum() == 1:
                r = int(np.flatnonzero(ins)[0])                     # the lone inside corner first, winding kept
                a, b, c = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
                out.append((a, cut(a, b), cut(a, c)))
            else:
                r = int(np.flatnonzero(~ins)[0])                    # the lone outside corner first
                c, a, b = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
                ab, ca = cut(b, c), cut(c, a)
                out.append((a, b, ab))
                out.append((a, ab, ca))
        if extra_v:
            V.append(np.array(extra_v))
            N.append(np.array(extra_n))
            W.append(np.array(extra_w).reshape(-1, self.W.shape[1]))
            for k, xs in X.items():
                xs.append(np.array(extra_x[k]))
        tris = np.concatenate(keep + ([np.array(out, np.int64)] if out else []), 0)
        V, N, W = np.concatenate(V), np.concatenate(N), np.concatenate(W)
        X = {k: np.concatenate(v) for k, v in X.items()}
        used = np.unique(tris)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        return Shell(V[used], remap[tris], _unit(N[used]), self.bones, W[used], {k: v[used] for k, v in X.items()})

    def band(self, f, lo, hi):
        """The strip lo <= f <= hi of the field f (n,), cut along both level lines."""
        sh = Shell(self.verts, self.tris, self.normals, self.bones, self.W, dict(self.fields, _band=f))
        sh = sh.clip(sh.fields["_band"] - lo)
        sh = sh.clip(hi - sh.fields["_band"])
        sh.fields.pop("_band")
        return sh

    # ---- topology
    def edges(self):
        """Unique undirected edges (e, 2)."""
        e = np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        return np.unique(np.sort(e, 1), axis=0)

    def boundary(self):
        """Directed boundary edges (b, 2), each in its triangle's winding."""
        e = np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        key = np.sort(e, 1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return e[cnt[np.asarray(inv).reshape(-1)] == 1]

    def loops(self):
        """The boundary as closed loops of vertex ids (lists), each in its triangles' winding."""
        nxt = {}
        for a, b in self.boundary():
            nxt.setdefault(int(a), []).append(int(b))
        seen, out = set(), []
        for start in list(nxt):
            if start in seen:
                continue
            loop, v = [], start
            while v not in seen and v in nxt:
                seen.add(v)
                loop.append(v)
                v = nxt[v][0]
            if len(loop) >= 3:
                out.append(loop)
        return out

    # ---- cloth
    def relax(self, P, iters, lam=0.5):
        """Laplacian smoothing of positions P (n, 3) over the shell's edges, `lam` of the way to the neighbours' mean per
        pass; boundary vertices move only along their boundary neighbours, so a cut edge stays a smooth line where it
        was."""
        P = np.array(P, float)
        if iters <= 0:
            return P
        E = self.edges()
        E = np.concatenate([E, E[:, ::-1]])
        B = self.boundary()
        on_b = np.zeros(len(P), bool)
        on_b[B.ravel()] = True
        pairs = np.concatenate([E[~on_b[E[:, 0]]], B, B[:, ::-1]])       # interior: every neighbour; boundary: its own
        for _ in range(int(iters)):
            acc, cnt = np.zeros_like(P), np.zeros(len(P))
            np.add.at(acc, pairs[:, 0], P[pairs[:, 1]])
            np.add.at(cnt, pairs[:, 0], 1.0)
            ok = cnt > 0
            P[ok] += lam * (acc[ok] / cnt[ok, None] - P[ok])
        return P

    def offset(self, dist, relax=0, lam=0.5, fit=None, gap=None, iters=2):
        """Positions (n, 3): each vertex moved out by `dist` (scalar or (n,)) along the skin's normal, smoothed `relax`
        times (cloth bridges the skin's small dips) and, with `fit` and `gap`, pushed back out to at least `gap` off the
        body's skin (`fit.push_out`), `iters` rounds of smoothing and pushing; then `clear` keeps the triangles off it."""
        d = np.broadcast_to(np.asarray(dist, float), (len(self.verts),))
        P = self.verts + self.normals * d[:, None]
        rounds = max(int(iters), 1) if relax else 1
        for _ in range(rounds):
            P = self.relax(P, int(relax) // rounds if relax else 0, lam)
            if fit is not None and gap is not None:
                P = fit.push_out(P, float(gap))
        if fit is not None and gap is not None:
            P = self.clear(P, fit, float(gap))
        return P

    def clear(self, P, fit, gap, iters=6):
        """Move vertices out until the triangles' centres and edge middles are at least `gap` off the body's skin too (a
        bump of the skin can pass between the vertices of a smoothed shell): each sample too close pushes its triangle's
        corners out along its way out of the skin. Returns the new positions."""
        P = np.array(P, float)
        T = self.tris
        owner = np.concatenate([T, T[:, [0, 1, 1]], T[:, [1, 2, 2]], T[:, [2, 0, 0]]])
        for _ in range(int(iters)):
            a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
            smp = np.concatenate([(a + b + c) / 3.0, 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)])
            dist, q, tri, bary = fit.skin.closest(smp)
            bad = np.flatnonzero(dist < gap - 1e-6)
            if not len(bad):
                break
            out = np.where((np.abs(dist[bad]) > 1e-4)[:, None], _unit(smp[bad] - q[bad]) * np.sign(dist[bad])[:, None],
                           fit.skin.outward(tri[bad], bary[bad]))
            push = out * (gap - dist[bad])[:, None]
            acc, cnt = np.zeros_like(P), np.zeros(len(P))
            for k in range(3):
                np.add.at(acc, owner[bad, k], push)
                np.add.at(cnt, owner[bad, k], 1.0)
            P += acc / np.maximum(cnt, 1.0)[:, None] * 1.25
        return P

    def gap(self, P, fit):
        """The closest the shell at positions P comes to the body's skin: the least signed distance of its vertices and
        of its triangles' centres (negative: something is inside)."""
        P = np.asarray(P, float)
        return float(min(fit.skin.closest(P)[0].min(), fit.skin.closest(P[self.tris].mean(1))[0].min()))

    def patch(self, P, u, v, tris=None, u_rep=1.0, v_scale=1.0, mat=0, tag="", rims=(), rim_depth=0.004):
        """An outfit_geo.Patch of the shell's triangles `tris` (a bool mask or ids; default all) at positions P, with
        texture coordinates from the per-vertex u (periodic, a fraction of a turn: a triangle across the seam gets 1 added
        to its small u's) times `u_rep` (whole repeats round, so the seam matches) and v times `v_scale`. Only the
        vertices the triangles use are kept. `rims`: boundary loops (vertex id lists from `loops()`) closed back towards
        the skin by `rim_depth` (scalar or one per loop) along the skin normal; a rim takes its loop's weights and u."""
        P = np.asarray(P, float)
        u, v = np.asarray(u, float), np.asarray(v, float)
        T = self.tris if tris is None else self.tris[np.asarray(tris)]
        tu = u[T]
        lo, hi = tu.min(1, keepdims=True), tu.max(1, keepdims=True)
        tu = np.where((hi - lo > 0.5) & (tu < 0.5 * (hi + lo)), tu + 1.0, tu)
        uvs = [np.stack([tu.ravel() * u_rep, v[T].ravel() * v_scale], 1)]
        used = np.unique(np.concatenate([T.ravel()] + [np.asarray(lp, np.int64) for lp in rims]))
        remap = np.full(len(P), -1, np.int64)
        remap[used] = np.arange(len(used))
        faces = [tuple(int(x) for x in t) for t in remap[T]]
        verts, W = [P[used]], [self.W[used]]
        nv = len(used)
        depths = np.broadcast_to(np.asarray(rim_depth, float), (len(rims),))
        for loop, depth in zip(rims, depths):
            loop = np.asarray(loop, np.int64)
            verts.append(P[loop] - self.normals[loop] * depth)
            W.append(self.W[loop])
            idx = nv + np.arange(len(loop))
            nv += len(loop)
            ru = []
            for k in range(len(loop)):
                a, b = loop[k], loop[(k + 1) % len(loop)]
                faces.append((int(remap[b]), int(remap[a]), int(idx[k]), int(idx[(k + 1) % len(loop)])))
                ua, ub = u[a], u[b]
                if abs(ub - ua) > 0.5:
                    ub += np.sign(ua - ub)
                dv = depth * v_scale
                ru += [(ub * u_rep, v[b] * v_scale), (ua * u_rep, v[a] * v_scale), (ua * u_rep, v[a] * v_scale - dv),
                       (ub * u_rep, v[b] * v_scale - dv)]
            uvs.append(np.array(ru))
        Wm = np.concatenate(W)
        w = {b: Wm[:, i] for i, b in enumerate(self.bones) if Wm[:, i].max() > 1e-6}
        return G.Patch(np.concatenate(verts), faces, np.concatenate(uvs), w, mat, tag)


ARM_CHAIN = ("arm", "elbow", "wrist")
LEG_CHAIN = ("leg", "knee", "ankle")


def limb_coord(V, land, side, names=ARM_CHAIN):
    """(n,) a limb's length coordinate of points V: the arc length from its first joint (landmark `<names[0]>.<side>`)
    along the bone line through the other two of the nearest point on it (before the first joint the first bone's line
    goes on backwards, so for an arm the shoulder and the chest get small or negative values; past the last joint the
    second bone's line goes on)."""
    a, e, w = (np.asarray(land[f"{n}.{side}"], float) for n in names)
    V = np.asarray(V, float)
    best, s, s0 = np.full(len(V), np.inf), np.zeros(len(V)), 0.0
    for i, (p, q) in enumerate(((a, e), (e, w))):
        d = q - p
        ln = float(np.linalg.norm(d))
        u = d / ln
        t = (V - p) @ u
        tc = np.clip(t, -np.inf if i == 0 else 0.0, ln if i == 0 else np.inf)
        dist = np.linalg.norm(V - (p + tc[:, None] * u), axis=1)
        better = dist < best
        best = np.where(better, dist, best)
        s = np.where(better, s0 + tc, s)
        s0 += ln
    return s


def limb_length(land, side, names=ARM_CHAIN):
    """First joint -> second -> third, metres."""
    a, e, w = (np.asarray(land[f"{n}.{side}"], float) for n in names)
    return float(np.linalg.norm(e - a) + np.linalg.norm(w - e))


def both_sides(V, land, names=ARM_CHAIN):
    """(n,) limb_coord of the limb on each vertex's side (x > 0: the left one)."""
    V = np.asarray(V, float)
    return np.where(V[:, 0] >= 0.0, limb_coord(V, land, "L", names), limb_coord(V, land, "R", names))


def limb_angle(V, land, names=ARM_CHAIN, ref=(0.0, 0.0, 1.0)):
    """(n,) the angle round the axis first joint -> last joint of the limb on each vertex's side, as a fraction of a
    turn 0..1 (0 = towards `ref`: up for an arm, the front (0, -1, 0) for a leg)."""
    V = np.asarray(V, float)
    out = np.zeros(len(V))
    for side, sel in (("L", V[:, 0] >= 0.0), ("R", V[:, 0] < 0.0)):
        a, w = (np.asarray(land[f"{n}.{side}"], float) for n in (names[0], names[-1]))
        ax = _unit(w - a)
        up = _unit(np.asarray(ref, float) - (np.asarray(ref, float) @ ax) * ax)
        fw = np.cross(ax, up)
        d = V[sel] - a
        out[sel] = (np.arctan2(d @ fw, d @ up) / (2 * np.pi)) % 1.0
    return out


def azimuth(V, centre=(0.0, 0.0)):
    """(n,) the angle round a vertical axis through `centre` (x, y) as a fraction of a turn 0..1 (0 = front, -y)."""
    V = np.asarray(V, float)
    return (np.arctan2(V[:, 0] - centre[0], -(V[:, 1] - centre[1])) / (2 * np.pi)) % 1.0


ARM, LEG, HEAD = F.is_arm, F.is_leg, F.is_head
