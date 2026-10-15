"""Cloth copies (numpy only): the simulation copy of a garment that a part builder declares in its rig layers
(`Part.rig["cloth"]`, the rig description's cloth layer, mkmmd.core.rig) for mk build's cloth stage (docs/design.md:
Cloth). Model space, metres, the rest pose.

A copy (`copy`) is one dict:

    name         the garment, unique in the model; fold shapes whose `piece` is this name give way to the cloth
    materials    the render materials whose faces follow the simulated cloth (the stage binds them to it)
    bones        the bones the weights index (PMX names here, Blender's in rig.json)
    points       {"sim": (n, 3), "collide": (m, 3)}: the simulated cloth's vertices and the body it collides with
                 (they scale with the part, Part.scaled)
    sim_faces    (f, 4) quads of the simulated cloth, facing out
    pins         (n,) 0..1: where the garment is attached: 1 holds a vertex where its weights carry it (cloth under a
                 band, at a seam), and its faces keep their skin's motion there; 0 leaves it free
    holds        (n,) 0..1: the least the free cloth is held toward where its weights carry it (a sleeve on an arm
                 that moves faster than cloth can follow), over which the cloth's `hold` may go
    collides     (n,) 0 or 1: whether the vertex collides with the body (a sleeve rides its arm on its hold: caught
                 between an arm and her chest, its cloth would be pushed off the arm)
    sim_weights  (n, len(bones)) the garment's skin weights at the cloth's vertices: the armature carries the cloth to
                 where it would be with no physics, which pinned vertices follow
    collide_faces (g, 3), collide_weights (m, len(bones)): the skin under the garment, set in a little so the cloth's
                 rest shape is clear of it
    settings     the garment's cloth (mass, stiffness ...: mkmmd.blender.build.cloth), over which a scene's
                 [cloth.<cast>] has the last word

`ring_grid` takes a coarse grid off lofted rings (a piece knit in rows), `skin_collider` the skin under a garment."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    verts: np.ndarray        # (k * m, 3)
    quads: np.ndarray        # (f, 4)
    index: np.ndarray        # (k * m,) the flat index of each vertex in the rings (K * M)


def ring_grid(rings, every=2):
    """Every `every`-th ring and column of closed rings (K, M, 3) (the first and last ring always), as a quad grid
    stepping (k, j) -> (k+1, j) -> (k+1, j+1) -> (k, j+1) like outfit_geo.loft."""
    R = np.asarray(rings, float)
    K, M = R.shape[:2]
    rows = np.unique(np.r_[np.arange(0, K, every), K - 1])
    cols = np.arange(0, M, every)
    k, m = len(rows), len(cols)
    a = np.arange(k * m).reshape(k, m)
    q = np.stack([a[:-1], a[1:], np.roll(a[1:], -1, 1), np.roll(a[:-1], -1, 1)], -1).reshape(-1, 4)
    index = (rows[:, None] * M + cols[None, :]).reshape(-1)
    return Grid(R.reshape(-1, 3)[index], q, index)


def facing_out(verts, quads, inside):
    """`quads` turned round where most of them face towards the points `inside` (n, 3) (one per vertex)."""
    V = np.asarray(verts, float)
    Q = np.asarray(quads)
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    out = V[Q].mean(1) - np.asarray(inside, float)[Q].mean(1)
    return Q[:, ::-1].copy() if np.einsum("ij,ij->i", n, out).sum() < 0 else Q


def vertex_normals(verts, tris):
    """Unit area-weighted vertex normals of triangles (as wound)."""
    V = np.asarray(verts, float)
    T = np.asarray(tris)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        np.add.at(N, T[:, i], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def skin_collider(skin, tri_mask, inset):
    """(verts, outward triangles, weights {bone: (n,)}) of the triangles `tri_mask` of a Skin (outfit_fit.Skin): its
    used vertices only, each moved in along its normal by `inset` metres."""
    T = skin.tris[np.asarray(tri_mask, bool)]
    if skin._flip:
        T = T[:, ::-1]
    used, inv = np.unique(T, return_inverse=True)
    T = inv.reshape(T.shape)
    V = skin.verts[used]
    V = V - vertex_normals(V, T) * float(inset)
    return V, T, {b: w[used] for b, w in skin.weights.items() if w[used].any()}


def _dense(W, bones, n):
    out = np.zeros((n, len(bones)), np.float32)
    for i, b in enumerate(bones):
        if b in W:
            out[:, i] = W[b]
    s = out.sum(1, keepdims=True)
    return out / np.where(s > 0.0, s, 1.0)


def copy(name, materials, pieces, collider, settings):
    """The cloth copy (module docstring) of a garment from its `pieces`, one simulated cloth of separate islands, each
    {"verts": (n, 3), "quads": (f, 4), "pins": (n,), "weights": {bone: (n,)}, "holds": (n,) or a number (default 0),
    "collides": (n,) or a bool (default true)}, and the `collider` skin_collider made."""
    off, V, Q, P, H, K, Ws = 0, [], [], [], [], [], []
    for pc in pieces:
        v = np.asarray(pc["verts"], float)
        V.append(v)
        Q.append(np.asarray(pc["quads"], np.int64) + off)
        P.append(np.clip(np.asarray(pc["pins"], float), 0.0, 1.0))
        H.append(np.clip(np.broadcast_to(np.asarray(pc.get("holds", 0.0), float), len(v)), 0.0, 1.0))
        K.append(np.broadcast_to(np.asarray(pc.get("collides", True), bool), len(v)))
        Ws.append(pc["weights"])
        off += len(v)
    cv, ct, cw = collider
    bones = sorted({b for w in Ws for b, a in w.items() if np.any(a)} | set(cw))
    sim_w = np.concatenate([_dense(w, bones, len(v)) for w, v in zip(Ws, V)])
    return {"name": name, "materials": list(materials), "bones": bones,
            "points": {"sim": np.concatenate(V), "collide": np.asarray(cv, float)},
            "sim_faces": np.concatenate(Q).astype(np.int32), "pins": np.concatenate(P).astype(np.float32),
            "holds": np.concatenate(H).astype(np.float32), "collides": np.concatenate(K).astype(np.uint8),
            "sim_weights": sim_w, "collide_faces": np.asarray(ct, np.int32),
            "collide_weights": _dense(cw, bones, len(cv)), "settings": dict(settings)}
