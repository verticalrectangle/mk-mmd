"""Hands of the procedural body made from a mesh asset, `[body.hand] mesh = "path/to/hand.npz"`, instead of the designed
cage of body_hand: a left hand drawn by someone else, in its own placement and scale, open at the wrist. It is turned into
the hand frame (body_hand.frame) by its own joints, scaled so its middle finger, straight, reaches `length` from the
wrist joint, subdivided, and its wrist end eased onto the forearm's last ring over BLEND along the hand. Its joints become
the skeleton's finger joints (each tip moved out to the end of the skin its last bone drives), its weights the hand's
skin weights (body_weights.hand_weights), and nail plates are cast on its fingertips (body_hand.nail_plate with
`[body.hand] nail`).

The asset, an .npz of:
  verts (n, 3), face_flat, face_sizes   the Catmull-Clark cage: faces counter-clockwise seen from outside, one open
                                        boundary (the seam at the wrist)
  levels                                Catmull-Clark levels to apply; the seam then has (its cage points) * 2**levels
                                        points, and the forearm is built with as many per ring
  joint_names, joints (k, 3)            wrist, thumb0 thumb1 thumb2 thumb_tip, index1 index2 index3 index_tip, middle..,
                                        ring.., little.. (mkmmd.model.skeleton's semantic names without the side)
  bones, weights (n, b)                 per cage vertex, rows summing to one, over names of BONES
  license, source                       optional: where the mesh came from (recorded, not read)
A missing asset with a maker beside it (make_<stem>.py, a module of mkmmd with make(out): the girl base's make_hand) is
rebuilt by it on first use.
Its frame comes from its joints: along = wrist -> middle1, across = index1 -> little1 (towards the little finger), out of
the palm = -(along x across); the fingers and the thumb rest a little curled towards the palm side (a left hand).

Published in the hand shell's info, as by body_hand: frame, chains, lengths, member (each part's share of every vertex,
from the weights), chain_s, along, a_mid, tiles, palm_u, palm_v; plus bone_names and bone_weights. UVs: the palm tile
takes the back of the hand from the thumb's edge (u 0) to the little finger's edge (0.5) and the palm back to the thumb's
edge (1), v along the hand from the seam; a finger tile u round its finger (the back at 0.125, the pad at 0.625), v along
its chain to the tip."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..spec import make_missing
from ..subdiv import subdivide
from . import body_hand as BH
from .body_geom import Path as Chain, Shell, smoothstep, unit, vertex_normals

FINGERS = BH.FINGERS
PARTS = BH.PARTS
BONES = ("forearm", "wrist", "thumb0", "thumb1", "thumb2") + tuple(f"{f}{k}" for f in FINGERS for k in (1, 2, 3))
CHAIN = {"thumb": ("thumb0", "thumb1", "thumb2", "thumb_tip")}
CHAIN.update({f: (f"{f}1", f"{f}2", f"{f}3", f"{f}_tip") for f in FINGERS})
PART_BONES = {"palm": ("forearm", "wrist"), "thumb": ("thumb0", "thumb1", "thumb2")}
PART_BONES.update({f: (f"{f}1", f"{f}2", f"{f}3") for f in FINGERS})
BLEND = 0.030                     # the wrist end eases onto the forearm's last ring over this distance along the hand
BACK_U, FRONT_U = 0.125, 0.625    # u of the back and of the pad on a finger tile


@lru_cache(maxsize=4)
def _load(path, mtime):
    with np.load(path, allow_pickle=False) as z:
        cage = np.asarray(z["verts"], float)
        sizes, flat = np.asarray(z["face_sizes"], int), np.asarray(z["face_flat"], int)
        levels = int(z["levels"])
        joints = {str(n): np.asarray(p, float) for n, p in zip(z["joint_names"], z["joints"])}
        bones = [str(b) for b in z["bones"]]
        W = np.asarray(z["weights"], float)
    missing = [n for n in ("wrist",) + tuple(x for c in CHAIN.values() for x in c) if n not in joints]
    if missing:
        raise ValueError(f"hand mesh {path}: joints missing: {', '.join(missing)}")
    bad = [b for b in bones if b not in BONES]
    if bad:
        raise ValueError(f"hand mesh {path}: unknown bones {', '.join(bad)} (bones: {', '.join(BONES)})")
    if W.shape != (len(cage), len(bones)) or not np.allclose(W.sum(1), 1.0, atol=1e-4):
        raise ValueError(f"hand mesh {path}: weights must be (verts, bones), every row summing to one")
    faces = [[int(v) for v in f] for f in np.split(flat, np.cumsum(sizes)[:-1])]
    sd = subdivide(cage, faces, levels)
    F = [[int(v) for v in f] for f in sd.faces]
    Wd = np.zeros((len(sd.verts), len(BONES)))
    Wd[:, [BONES.index(b) for b in bones]] = np.asarray(sd.L @ W)
    return dict(verts=np.asarray(sd.verts, float), faces=F, weights=Wd, joints=joints, seam=open_loop(F))


def load(D):
    """The subdivided asset named by the design's `mesh` (in its own frame and scale; shared, not to be changed)."""
    p = Path(str(D["mesh"])).expanduser()
    if not p.is_file() and not make_missing(p):
        raise ValueError(f"[body.hand] mesh: {p} does not exist")
    return _load(str(p), p.stat().st_mtime)


def open_loop(faces):
    """The vertices of the one open boundary of `faces`, in order round it (along the faces' own edge direction)."""
    edges = {(f[i], f[(i + 1) % len(f)]) for f in faces for i in range(len(f))}
    nxt = {}
    for a, b in edges:
        if (b, a) not in edges:
            if a in nxt:
                raise ValueError("the mesh's open boundary is not a simple loop")
            nxt[a] = b
    if not nxt:
        raise ValueError("the mesh is closed: it needs one open boundary, its seam")
    loop = [min(nxt)]
    while nxt[loop[-1]] != loop[0]:
        loop.append(nxt[loop[-1]])
        if len(loop) > len(nxt):
            raise ValueError("the mesh's open boundary is not a simple loop")
    if len(loop) != len(nxt):
        raise ValueError("the mesh has more than one open boundary: it may only be open at its seam")
    return np.array(loop)


def seam_size(D):
    """Points of the asset's seam after subdivision, which the forearm's last ring must have too."""
    return len(load(D)["seam"])


def _frame(J):
    a = unit(J["middle1"] - J["wrist"])
    r = J["little1"] - J["index1"]
    r = unit(r - (r @ a) * a)
    return a, r, -np.cross(a, r)


def _tips(V, W, J):
    """Every chain's tip joint moved along its last bone to the farthest point of the skin that bone drives."""
    dom = np.argmax(W, axis=1)
    out = {}
    for p, names in CHAIN.items():
        j3 = J[names[2]]
        d = unit(J[names[3]] - j3)
        mine = dom == BONES.index(names[2])
        if not mine.any():
            raise ValueError(f"hand mesh: no skin follows {names[2]}")
        out[names[3]] = j3 + d * float(((V[mine] - j3) @ d).max())
    return out


def place(land, D):
    """The asset in model space at the left wrist: dict(verts, faces, weights (n, len(BONES)), joints, seam (vertex ids),
    frame, wrist). Fresh arrays: the caller may change them."""
    A = load(D)
    J = dict(A["joints"])
    J.update(_tips(A["verts"], A["weights"], J))
    a_d, r_d, n_d = _frame(J)
    lean = sum(float((J[c[3]] - J[c[0]]) @ n_d) for c in CHAIN.values())
    if lean <= 0:
        raise ValueError("the hand mesh is not a left hand resting with its digits a little curled: they lean away "
                         "from its palm")
    mid = [J["wrist"]] + [J[n] for n in CHAIN["middle"]]
    s = float(D["length"]) / float(sum(np.linalg.norm(q - p) for p, q in zip(mid[:-1], mid[1:])))
    frame = BH.frame(land, D)
    R = np.stack(frame, 1) @ np.stack([a_d, r_d, n_d], 1).T
    W0 = np.asarray(land["wrist.L"], float)

    def put(P):
        return W0 + ((np.asarray(P, float) - J["wrist"]) @ R.T) * s
    return dict(verts=put(A["verts"]), faces=A["faces"], weights=A["weights"].copy(), seam=A["seam"].copy(),
                joints={k: put(p) for k, p in J.items()}, frame=frame, wrist=W0)


def joints(land, D):
    """The finger joints of the left hand from the asset ({semantic name.L: (3,)}: thumb0..2, index1..3, ... and the
    tips)."""
    J = place(land, D)["joints"]
    return {f"{n}.L": J[n] for names in CHAIN.values() for n in names}


def _ease_onto(V, seam, along, arm):
    """Ease the asset's wrist end onto the forearm's last ring: the seam's points move onto the ring's points (paired in
    order round the forearm, the turn that moves them least) and every vertex by the displacement of the seam at its
    angle round the forearm, fading out over BLEND along the hand. Changes V; returns (seam, ring vertex ids) paired."""
    ring_idx = np.asarray(arm.ring_index[-1])
    M = len(ring_idx)
    if len(seam) != M:
        raise ValueError(f"the hand mesh's seam has {len(seam)} points, the forearm's last ring {M}")
    ring = arm.verts[ring_idx]
    th = BH._angle(arm, V[seam])
    if np.sin(np.diff(np.unwrap(th))).sum() < 0:           # ring point k sits at 2 pi k / M: turn the same way
        seam = seam[::-1]
    k0 = min(range(M), key=lambda k: float(np.linalg.norm(ring - V[np.roll(seam, -k)], axis=1).sum()))
    seam = np.roll(seam, -k0)
    gap = float(np.linalg.norm(ring - V[seam], axis=1).max())
    if gap > 0.015:
        raise ValueError(f"the hand mesh's wrist is {gap * 1000:.0f} mm off the forearm's last ring: check its joints "
                         "and [body.hand] length")
    th = BH._angle(arm, V[seam])
    o = np.argsort(th)
    knots = np.concatenate([th[o][-1:] - 2 * np.pi, th[o], th[o][:1] + 2 * np.pi])

    def round_arm(vals, at):
        v = vals[o]
        return np.interp(at, knots, np.concatenate([v[-1:], v, v[:1]]))
    tv = BH._angle(arm, V)
    move = np.stack([round_arm(ring[:, c] - V[seam, c], tv) for c in range(3)], 1)
    fade = 1.0 - smoothstep((along - round_arm(along[seam], tv)) / BLEND)
    V += fade[:, None] * move
    V[seam] = ring
    return seam, ring_idx


def _palm_uv(L, part, seam_along):
    """(u, v) of every vertex on the palm tile, palm_v, and the u columns of a point (along, across): the back of the
    hand from the thumb's edge (u 0) to the little finger's edge (0.5), the palm back to the thumb's edge (1), each side
    spread evenly over the palm's width at that point; v from the seam to the palm's end."""
    A, R, N = L[:, 0], L[:, 1], L[:, 2]
    pv = dict(seam=float(seam_along), end=float(A[part == 0].max()))
    at, lo, hi, mid = [], [], [], []
    for e in np.arange(pv["seam"], pv["end"], 0.002):
        sl = (part == 0) & (A >= e) & (A < e + 0.002)
        if sl.sum() >= 4:
            at.append(e + 0.001)
            lo.append(R[sl].min())
            hi.append(R[sl].max())
            mid.append(0.5 * (N[sl].min() + N[sl].max()))

    def cols(a, r):
        t = np.clip((r - np.interp(a, at, lo)) / np.maximum(np.interp(a, at, hi) - np.interp(a, at, lo), 1e-6), 0, 1)
        return 0.5 * t, 1.0 - 0.5 * t
    back, front = cols(A, R)
    u = np.where(N < np.interp(A, at, mid), back, front)
    v = np.clip((A - pv["seam"]) / (pv["end"] - pv["seam"]), 0.0, 1.0)
    return u, v, pv, cols


def _digit_uv(V, chain, s, frame, thumb):
    """u of every vertex on a finger's (or the thumb's) tile, round the chain (BACK_U on the back, FRONT_U on the pad), at
    its chain coordinate s."""
    a, r, n = frame
    path = Chain(chain, blend=0.004)
    tg = unit(path.tangent(s))
    q = V - path.point(s)
    if thumb:                                                   # the thumb's pad faces the palm and the fingers
        pad = unit(r + n)
        ev = unit(pad[None] - (tg @ pad)[:, None] * tg)
        eu = np.cross(ev, tg)
    else:
        eu = unit(r[None] - (tg @ r)[:, None] * tg)
        ev = unit(n[None] - (tg @ n)[:, None] * tg - (eu @ n)[:, None] * eu)
    phi = np.arctan2(-np.einsum("ij,ij->i", q, ev), -np.einsum("ij,ij->i", q, eu))
    return np.mod((phi - 0.25 * np.pi) / (2 * np.pi), 1.0)


def _uvs(F, member, tiles_uv, atlas):
    """Per-face corner UVs in the atlas: a face takes the tile of the part that owns most of it; a face across the tile's
    u seam holds its far corners at the seam."""
    out = []
    for f in F:
        p = int(np.argmax(member[f].sum(0)))
        u, v = tiles_uv[p][0][f], tiles_uv[p][1][f]
        if u.max() - u.min() > 0.5:
            u = np.where(u < 0.5, 1.0, u)
        u0, v0, u1, v1 = atlas[PARTS[p]]
        out.append(np.stack([u0 + (u1 - u0) * u, v0 + (v1 - v0) * v], 1))
    return out


def hand_shells(shape, arm, atlas, nails=True):
    """(hand_L Shell, [nail_<part>_L Shells]) from the asset on the forearm `arm` (body_mesh.arm_shell, as many points per
    ring as the asset's seam): the asset's seam vertices are welded to the arm's last ring."""
    D = shape.hand
    P = place(shape.land, D)
    V, F, Wd, J = P["verts"], P["faces"], P["weights"], P["joints"]
    frame, W0 = P["frame"], P["wrist"]
    a, r, n = frame
    seam, ring_idx = _ease_onto(V, P["seam"], (V - W0) @ a, arm)
    L = np.stack([(V - W0) @ a, (V - W0) @ r, (V - W0) @ n], 1)
    member = np.stack([Wd[:, [BONES.index(b) for b in PART_BONES[p]]].sum(1) for p in PARTS], 1)
    part = np.argmax(member, axis=1)
    chains = {p: np.array([J[x] for x in CHAIN[p]]) for p in PARTS[1:]}
    lengths = {p: np.linalg.norm(np.diff(chains[p], axis=0), axis=1) for p in PARTS[1:]}
    chain_s = np.stack([BH.chain_coordinate(chains[p], V) for p in PARTS[1:]], 1)
    u, v, palm_v, cols = _palm_uv(L, part, L[seam, 0].mean())
    tiles_uv, tiles = {0: (u, v)}, {}
    for k, p in enumerate(PARTS[1:]):
        length = float(chain_s[part == k + 1, k].max())
        uk = _digit_uv(V, chains[p], chain_s[:, k], frame, p == "thumb")
        tiles_uv[k + 1] = (uk, np.clip(chain_s[:, k] / length, 0.0, 1.0))
        tiles[p] = dict(back_u=BACK_U, front_u=FRONT_U, length=length)
    uvs = _uvs(F, member, tiles_uv, atlas)
    palm_u = {}
    knuckles = {f: J[f + "1"] - W0 for f in FINGERS}
    a_k = min(float(q @ a) for q in knuckles.values()) - 0.004       # the palm is whole just before the knuckles
    for f, q in knuckles.items():
        back, front = cols(np.array([a_k]), np.array([q @ r]))
        palm_u[f] = (float(back[0]), float(front[0]))
    sh = Shell("hand_L")
    sh.verts = V
    sh.faces = [list(f) for f in F]
    sh.uv = uvs
    sh.s = L[:, 0].copy()
    sh.theta = np.full(len(V), np.nan)
    sh.ring = np.full(len(V), -1)
    sh.face_mat = [0] * len(F)
    sh.ext = {int(i): ("arm_L", int(j)) for i, j in zip(seam, ring_idx)}
    sh.info = dict(kind="hand", wrist=W0, frame=frame, chains=chains, lengths=lengths, member=member, chain_s=chain_s,
                   along=L[:, 0].copy(), a_mid=float((J["middle1"] - W0) @ a), la=arm.info["la"], lf=arm.info["lf"],
                   tiles=tiles, palm_u=palm_u, palm_v=palm_v, bone_names=list(BONES), bone_weights=Wd)
    out = []
    if nails:
        N = vertex_normals(V, F)
        dom = np.argmax(Wd, axis=1)
        for k, p in enumerate(PARTS[1:]):
            last = BONES.index(CHAIN[p][2])
            tri = np.array([[f[0], f[i], f[i + 1]] for f in F if (dom[f] == last).any() for i in range(1, len(f) - 1)])
            path = Chain(chains[p], blend=0.004)
            s3, L3 = float(path.cum[2]), float(path.seg[2])
            if p == "thumb":
                pad = unit(r + n)

                def axes(tg, pad=pad):
                    ev = unit(pad - (pad @ tg) * tg)
                    return np.cross(ev, tg), ev
            else:
                def axes(tg):
                    return BH._across_out(tg, r, n)

            def width(s, path=path, axes=axes, tri=tri, p=p):
                c = path.point(np.array([s]))[0]
                eu = axes(unit(path.tangent(np.array([s]))[0]))[0]
                t, _, _ = BH._raycast(np.array([c, c]), np.array([eu, -eu]), V, tri)
                if not np.isfinite(t).all():
                    raise ValueError(f"hand mesh: the {p}'s last bone does not run inside its skin")
                return float(t.sum())
            behind = 0.03 * float(D["length"]) / BH.REF_LENGTH        # the rays start this far off the chain, as body_hand's
            Pn, faces = BH.nail_plate(p, path, s3, L3, axes, width, V, N, tri, D["nail"], behind)
            ns = Shell(f"nail_{p}_L")
            ns.verts = Pn
            ns.faces = faces
            ns.uv = [np.full((4, 2), 0.5) for _ in faces]
            ns.s = np.zeros(len(Pn))
            ns.theta = np.full(len(Pn), np.nan)
            ns.ring = np.full(len(Pn), -1)
            ns.face_mat = [1] * len(faces)
            ns.info = dict(kind="nail", part=p, hand="hand_L")
            out.append(ns)
    return sh, out
