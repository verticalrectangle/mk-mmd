"""Shared geometry for solvers and checks: vectorised rotations, kinematic collision shapes riding on bones or objects,
and signed penetration of points into them. numpy only (works in Blender's Python as well).

Shapes (docs/design.md: Colliders) live on a source: ("bone", armature, bone) | ("object", "", name) | ("world", "", "").
Their geometry is stored in the source's REST frame for bones (world = posed bone matrix @ rest-local) and in the
object's frame for objects (scale baked into the geometry, the object's scale is ignored when posing)."""
import numpy as np

SAMPLES = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0])     # points along each chain segment (fraction head -> end)
ANCHOR_FREE = 0.25                                  # near a chain's root (m): no collision with the body it hangs
                                                    # from, and the PMX collision masks apply


# ---------------------------------------------------------------- rotations (vectorised, quaternions w x y z)
def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def min_rot(a, b):
    """(n, 3, 3) rotations taking unit vectors a -> b with the least turn."""
    v = np.cross(a, b)
    c = np.einsum("ij,ij->i", a, b)
    K = skew(v)
    R = np.eye(3)[None] + K + K @ K * (1.0 / np.maximum(1.0 + c, 1e-9))[:, None, None]
    bad = c < -0.99999
    if bad.any():                                    # opposite: half turn about any axis perpendicular to a
        ab = a[bad]
        e = np.where(np.abs(ab[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        ax = np.cross(ab, e)
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        R[bad] = 2.0 * ax[:, :, None] * ax[:, None, :] - np.eye(3)[None]
    return R


def quat_to_mat(q):
    """(n, 4) unit quaternions w x y z -> (n, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def mat_to_quat(R):
    """(n, 3, 3) rotations -> (n, 4) unit quaternions w x y z (Shepperd: branch on the largest diagonal term)."""
    t = np.stack([R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1)
    k = np.argmax(t, 1)
    q = np.empty((len(R), 4))
    for case in range(4):
        m = k == case
        if not m.any():
            continue
        r = R[m]
        if case == 0:
            s = 2.0 * np.sqrt(np.maximum(1.0 + t[m, 0], 1e-12))
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif case == 1:
            s = 2.0 * np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-12))
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif case == 2:
            s = 2.0 * np.sqrt(np.maximum(1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2], 1e-12))
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = 2.0 * np.sqrt(np.maximum(1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2], 1e-12))
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def unscaled(M):
    """(…, 4, 4) matrices with the scale removed from their 3x3 part (columns normalised)."""
    out = np.array(M, float, copy=True)
    out[..., :3, :3] /= np.maximum(np.linalg.norm(out[..., :3, :3], axis=-2, keepdims=True), 1e-12)
    return out


def apply(M, p):
    """Transform points p (…, 3) by 4x4 matrices M (…, 4, 4) (broadcasting)."""
    return np.einsum("...ij,...j->...i", M[..., :3, :3], p) + M[..., :3, 3]


# ---------------------------------------------------------------- shapes
KINDS = ("sphere", "capsule", "box", "cylinder")


class Shapes:
    """Kinematic shapes, each riding on a source. `items[kind]` are dicts with src (index into sources), tag, model
    (index of the model body it came from or None) and geometry: sphere c R; capsule a b R; box M half rnd;
    cylinder M R hh rnd (M: 4x4 placement in the source frame; rnd: rounded edges, m)."""

    def __init__(self):
        self.sources, self._si = [], {}
        self.items = {k: [] for k in KINDS}
        self.floor_z = None

    def src(self, kind, owner="", name=""):
        key = (kind, owner, name)
        if key not in self._si:
            self._si[key] = len(self.sources)
            self.sources.append(key)
        return self._si[key]

    def add(self, kind, source, tag, model=None, **geo):
        self.items[kind].append(dict(src=self.src(*source), tag=tag, model=model,
                                     **{k: (np.array(v, float) if isinstance(v, (list, tuple, np.ndarray)) else v)
                                        for k, v in geo.items()}))

    def add_item(self, item):
        """Add a resolved collider dict (as returned by the Blender `sample` op)."""
        it = dict(item)
        kind, source, tag = it.pop("kind"), tuple(it.pop("source")), it.pop("tag")
        if kind == "floor":
            self.floor_z = float(it["z"])
            return
        self.add(kind, source, tag, model=it.pop("model", None), **it)

    def body(self, armature, body, model):
        """A model collision body from rig.json (`bone`, `geom` in the bone's rest frame)."""
        g = body["geom"]
        src = ("bone", armature, body["bone"])
        if g["kind"] == "sphere":
            self.add("sphere", src, "body", model=model, c=g["c"], R=g["R"])
        elif g["kind"] == "capsule":
            self.add("capsule", src, "body", model=model, a=g["a"], b=g["b"], R=g["R"])
        else:
            self.add("box", src, "body", model=model, M=g["M"], half=g["half"], rnd=0.6 * float(np.min(g["half"])))

    def labels(self):
        out = []
        for kind in KINDS:
            out += [it["tag"] for it in self.items[kind]]
        return out + ["floor"]

    def pack(self):
        P = {}
        for kind, items in self.items.items():
            n = len(items)
            P[kind] = dict(n=n, src=np.array([it["src"] for it in items], int), tag=[it["tag"] for it in items],
                           model=[it["model"] for it in items])
            if kind == "sphere":
                P[kind].update(c=np.array([it["c"] for it in items]).reshape(n, 3),
                               R=np.array([it["R"] for it in items], float))
            elif kind == "capsule":
                P[kind].update(a=np.array([it["a"] for it in items]).reshape(n, 3),
                               b=np.array([it["b"] for it in items]).reshape(n, 3),
                               R=np.array([it["R"] for it in items], float))
            elif kind == "box":
                P[kind].update(M=np.array([it["M"] for it in items]).reshape(n, 4, 4),
                               half=np.array([it["half"] for it in items]).reshape(n, 3),
                               rnd=np.array([it["rnd"] for it in items], float))
            else:
                P[kind].update(M=np.array([it["M"] for it in items]).reshape(n, 4, 4),
                               R=np.array([it["R"] for it in items], float),
                               hh=np.array([it["hh"] for it in items], float),
                               rnd=np.array([it["rnd"] for it in items], float))
        return P


def world(P, Rs, ps):
    """World geometry of every packed shape for source rotations Rs (S,3,3) and positions ps (S,3)."""
    W = {}
    s = P["sphere"]
    if s["n"]:
        W["sphere"] = (ps[s["src"]] + np.einsum("kij,kj->ki", Rs[s["src"]], s["c"]), s["R"])
    c = P["capsule"]
    if c["n"]:
        Rk, pk = Rs[c["src"]], ps[c["src"]]
        W["capsule"] = (pk + np.einsum("kij,kj->ki", Rk, c["a"]), pk + np.einsum("kij,kj->ki", Rk, c["b"]), c["R"])
    for kind in ("box", "cylinder"):
        b = P[kind]
        if b["n"]:
            Rk = Rs[b["src"]] @ b["M"][:, :3, :3]
            ck = ps[b["src"]] + np.einsum("kij,kj->ki", Rs[b["src"]], b["M"][:, :3, 3])
            W[kind] = (ck, Rk, b["half"] if kind == "box" else (b["R"], b["hh"]), b["rnd"])
    return W


def penetration(X, r, W, floor_z):
    """Per point and shape: penetration depth (> 0 = inside) and push direction, for points X (M,3) of radius r (M,).
    Returns {kind: (pen (M,K), normal (M,K,3))} for the kinds that have shapes (+ 'floor')."""
    out = {}
    if "sphere" in W:
        C, R = W["sphere"]
        d = X[:, None, :] - C[None]
        dist = np.linalg.norm(d, axis=2)
        out["sphere"] = (R[None] + r[:, None] - dist, d / np.maximum(dist, 1e-9)[..., None])
    if "capsule" in W:
        A, Bc, R = W["capsule"]
        ab = Bc - A
        t = np.clip(np.einsum("mkj,kj->mk", X[:, None, :] - A[None], ab) / np.maximum((ab * ab).sum(1), 1e-12)[None],
                    0.0, 1.0)
        d = X[:, None, :] - (A[None] + t[..., None] * ab[None])
        dist = np.linalg.norm(d, axis=2)
        out["capsule"] = (R[None] + r[:, None] - dist, d / np.maximum(dist, 1e-9)[..., None])
    if "box" in W:                                       # box with edges rounded by rnd: a shrunk box + rnd
        C, Rk, half, rnd = W["box"]
        hb = np.maximum(half - rnd[:, None], 1e-6)
        clear = r[:, None] + rnd[None]
        loc = np.einsum("kji,mkj->mki", Rk, X[:, None, :] - C[None])          # (M,K,3) in box axes
        d = loc - np.clip(loc, -hb[None], hb[None])
        dist = np.linalg.norm(d, axis=2)
        outside = dist > 1e-9
        gap = hb[None] - np.abs(loc)                                            # inside: leave by the nearest face
        ax = np.argmin(gap, axis=2)
        gmin = np.take_along_axis(gap, ax[..., None], 2)[..., 0]
        sgn = np.sign(np.take_along_axis(loc, ax[..., None], 2)[..., 0])
        sgn[sgn == 0] = 1.0
        n_in = np.zeros_like(loc)
        np.put_along_axis(n_in, ax[..., None], sgn[..., None], 2)
        pen = np.where(outside, clear - dist, gmin + clear)
        nl = np.where(outside[..., None], d / np.maximum(dist, 1e-9)[..., None], n_in)
        out["box"] = (pen, np.einsum("kij,mkj->mki", Rk, nl))
    if "cylinder" in W:                                  # cylinder with its rims rounded by rnd
        C, Rk, (Rc, hh), rnd = W["cylinder"]
        Rc, hh = np.maximum(Rc - rnd, 1e-6), np.maximum(hh - rnd, 1e-6)
        clear = r[:, None] + rnd[None]
        loc = np.einsum("kji,mkj->mki", Rk, X[:, None, :] - C[None])
        rad = np.linalg.norm(loc[..., :2], axis=2)
        radial = loc[..., :2] / np.maximum(rad, 1e-9)[..., None]
        q = np.concatenate([radial * np.minimum(rad, Rc[None])[..., None],
                            np.clip(loc[..., 2], -hh[None], hh[None])[..., None]], 2)
        d = loc - q
        dist = np.linalg.norm(d, axis=2)
        outside = dist > 1e-9
        side_gap, top_gap = Rc[None] - rad, hh[None] - np.abs(loc[..., 2])
        n_side = np.concatenate([radial, np.zeros(rad.shape + (1,))], 2)
        n_cap = np.zeros_like(loc)
        n_cap[..., 2] = np.where(loc[..., 2] >= 0, 1.0, -1.0)
        n_in = np.where((side_gap < top_gap)[..., None], n_side, n_cap)
        pen = np.where(outside, clear - dist, np.minimum(side_gap, top_gap) + clear)
        nl = np.where(outside[..., None], d / np.maximum(dist, 1e-9)[..., None], n_in)
        out["cylinder"] = (pen, np.einsum("kij,mkj->mki", Rk, nl))
    if floor_z is not None:
        n = np.zeros((len(X), 1, 3))
        n[..., 2] = 1.0
        out["floor"] = ((floor_z + r - X[:, 2])[:, None], n)
    return out


# ---------------------------------------------------------------- chains (from rig.json) against shapes
class Chains:
    """Simulated / measured chain bones of a model, roots first. Built from rig.json chains of the chosen families:
    per bone the parent index (-1 = root), anchor bone, family, rest head and segment end (armature space), body
    radius, collision group and mask, and the arc length from the root."""

    def __init__(self, rig, families=None):
        bodies = {b["bone"]: b for b in rig["bodies"] if b.get("bone")}
        self.bones, self.parent, self.anchor, self.family, self.head, self.end, self.radius = ([] for _ in range(7))
        self.groups = []
        chains = [c for c in rig["chains"] if families is None or c["family"] in families]
        for c in chains:
            base = len(self.bones)
            for i, name in enumerate(c["bones"]):
                p = c["parents"][i]
                self.bones.append(name)
                self.parent.append(base + p if p >= 0 else -1)
                self.anchor.append(c["anchor"])
                self.family.append(c["family"])
                self.head.append(rig["bones"][name]["head"])
                self.end.append(c["ends"][i])
                self.radius.append(c["body_radius"][i])
                b = bodies.get(name, {})
                self.groups.append((b.get("group", -1), set(b.get("no_collide", []))))
        self.parent = np.array(self.parent, int)
        self.head, self.end = np.array(self.head, float).reshape(-1, 3), np.array(self.end, float).reshape(-1, 3)
        self.radius = np.array(self.radius, float)
        self.chain_bones = set(self.bones)
        L = np.linalg.norm(self.end - self.head, axis=1)
        self.arc = np.zeros(len(self.bones))
        for i in range(len(self.bones)):                      # parents come before children within each chain
            self.arc[i] = L[i] + (0.0 if self.parent[i] < 0 else self.arc[self.parent[i]])

    def __len__(self):
        return len(self.bones)

    def points(self, samples=SAMPLES):
        """Rest sample points (M,3) along every segment and the owning bone index (M,)."""
        own = np.repeat(np.arange(len(self.bones)), len(samples))
        frac = np.tile(samples, len(self.bones))
        seg = self.end - self.head
        return self.head[own] + frac[:, None] * seg[own], own, frac


def model_shapes(shapes, rig, armature, skip_bones=()):
    """Add a model's own collision bodies (rig.json) riding on its bones, except the bodies of skip_bones."""
    out = {}
    for k, b in enumerate(rig["bodies"]):
        if not b.get("bone") or b["bone"] in skip_bones or "geom" not in b:
            continue
        shapes.body(armature, b, model=k)
        out[k] = b
    return out


def enable_matrix(chains, P, rig, rest_W, samples=SAMPLES, radius=None, use_masks=False, anchor_free=ANCHOR_FREE):
    """(N, K) collision enables per kind for chains vs packed shapes P. Skipped pairs: a model body the chain bone
    already overlaps in the rest pose (the author's intended overlaps, e.g. hair roots inside the head sphere); within
    anchor_free metres of the root, the body the chain hangs from and every pair the PMX collision masks exclude;
    further down every body collides unless use_masks. Scene shapes (model None) always collide."""
    N = len(chains)
    X, own, _ = chains.points(samples)
    rr = (chains.radius if radius is None else radius)[own]
    pen_rest = penetration(X, rr, rest_W, None)
    near_root = chains.arc < anchor_free
    anchors = np.array(chains.anchor)
    enable = {}
    for kind in KINDS:
        n = P[kind]["n"]
        if not n:
            continue
        E = np.ones((N, n), bool)
        for k in range(n):
            m = P[kind]["model"][k]
            if m is None:
                continue
            body = rig["bodies"][m]
            E[(pen_rest[kind][0][:, k] > 0).reshape(N, len(samples)).any(1), k] = False
            E[near_root & (anchors == body["bone"]), k] = False
            bmask = set(body.get("no_collide", []))
            for i, (grp, mask) in enumerate(chains.groups):
                if (near_root[i] or use_masks) and (body.get("group") in mask or grp in bmask):
                    E[i, k] = False
        enable[kind] = E
    return enable


def measure(X, rr, W, floor_z, P, enable=None, own=None):
    """Hard penetration per point (M,) (> 0 = inside) and the index of the deepest shape in Shapes.labels() order
    (the floor is the last label). Disabled pairs (enable[kind][bone, k] False, own = bone index per point) count as
    -1 m. Points with no shape at all get -inf and index -1."""
    res = penetration(X, rr, W, floor_z)
    Ss, ids, off = [], [], 0
    for kind in KINDS:
        n = P[kind]["n"]
        if kind in res:
            pen = res[kind][0]
            if enable is not None and kind in enable:
                pen = np.where(enable[kind][own], pen, -1.0)
            Ss.append(pen)
            ids.append(off + np.arange(n))
        off += n
    if "floor" in res:
        Ss.append(res["floor"][0])
        ids.append(np.array([off]))
    if not Ss:
        return np.full(len(X), -np.inf), np.full(len(X), -1)
    S, I = np.concatenate(Ss, 1), np.concatenate(ids)
    k = np.argmax(S, 1)
    return S[np.arange(len(X)), k], I[k]
