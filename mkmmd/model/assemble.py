"""Assemble the Parts of a character into one PMX model (bpy-free).

`assemble(parts, name, scale)` merges bones (parents first), materials (draw order = declaration order), meshes (one
PMX vertex per unique (position, UV, normal group), Catmull-Clark applied where `Mesh.subsurf` asks, ngons
triangulated, weights capped at 4 bones), vertex morphs, display frames, rigid bodies and joints into a
`pmx_io.PmxModel`, converting model space (metres, Z up, facing -Y) to PMX space exactly as mmd_tools reads it back:
position (x, y, z) -> (x, z, y) / scale; directions swap y and z; rigid-body and joint rotations go through the
rotation matrix to mmd_tools' Euler YXZ and are negated and swapped as its importer expects. Triangles are written
with their winding reversed (mmd_tools' pmx.load/save reverse the face order: Blender CCW <-> file order).

`Assembled.expected` keeps arrays in model space (vertex positions, weights, morph offsets by PMX vertex) so the
Blender-side verification can compare an imported PMX with what was meant."""
import math
from dataclasses import dataclass, field

import numpy as np

from . import pmx_io as X
from .part import Bone

PANEL = {"brow": 1, "eye": 2, "mouth": 3, "other": 4}
TEX_DIR = "tex"                      # PMX texture paths are relative to the file and use backslashes (Windows convention)


@dataclass
class Assembled:
    pmx: X.PmxModel
    textures: list = field(default_factory=list)       # texture file names (in tex_dir) the PMX references
    bone_index: dict = field(default_factory=dict)
    morph_index: dict = field(default_factory=dict)
    material_index: dict = field(default_factory=dict)
    body_index: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)       # numpy arrays in model space for verification
    meta: dict = field(default_factory=dict)           # JSON-able description (bones, bodies, joints, names)
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    scale: float = 0.08

    def save_expected(self, prefix):
        """Write `<prefix>.npz` (arrays) and `<prefix>.json` (names, bones, bodies, joints) for the Blender-side
        verification (`mkmmd.blender.model.verify.load_expected`)."""
        import json
        arrays = {k: np.asarray(self.expected[k]) for k in ("pos", "normal", "uv", "bone_idx", "bone_w", "tri_mat", "tris")}
        names = list(self.expected["morphs"])
        for k, name in enumerate(names):
            ids, vec = self.expected["morphs"][name]
            arrays[f"morph{k}_ids"], arrays[f"morph{k}_vec"] = ids, vec
        np.savez_compressed(prefix + ".npz", **arrays)
        meta = dict(self.meta, morphs=names)
        with open(prefix + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)


# ---------------------------------------------------------------- coordinate helpers

def _swap(a):
    a = np.asarray(a, float)
    return np.stack([a[..., 0], a[..., 2], a[..., 1]], axis=-1)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix_xyz(e):
    """Rotation matrix of a Blender 'XYZ' Euler (radians): Rz @ Ry @ Rx."""
    return _rz(e[2]) @ _ry(e[1]) @ _rx(e[0])


def euler_yxz(R):
    """(ex, ey, ez) with R = Rz(ez) @ Rx(ex) @ Ry(ey): Blender's 'YXZ' Euler (Y applied first), the mode mmd_tools
    gives rigid bodies and joints."""
    R = np.asarray(R, float)
    sa = R[2, 1]
    ex = math.asin(max(-1.0, min(1.0, sa)))
    if abs(sa) < 1.0 - 1e-9:
        ey = math.atan2(-R[2, 0], R[2, 2])
        ez = math.atan2(-R[0, 1], R[1, 1])
    else:                                    # gimbal lock: fold the twist into ey
        ey = math.atan2(R[0, 2], R[0, 0])
        ez = 0.0
    return ex, ey, ez


def pmx_euler(rot_xyz):
    """PMX rotation (rx, ry, rz) that mmd_tools turns back into the model-space XYZ Euler `rot_xyz`."""
    ex, ey, ez = euler_yxz(matrix_xyz(rot_xyz))
    return (-ex, -ez, -ey)


# ---------------------------------------------------------------- bones

def order_bones(bones):
    """Bones with parents (and grant parents) before children, keeping the given order otherwise."""
    by = {b.name: b for b in bones}
    out, state = [], {}

    def emit(b):
        st = state.get(b.name)
        if st == 2:
            return
        if st == 1:
            raise ValueError(f"bone dependency cycle through {b.name!r}")
        state[b.name] = 1
        deps = [b.parent] if b.parent else []
        if b.grant and b.grant.get("parent"):
            deps.append(b.grant["parent"])
        for d in deps:
            if d in by:
                emit(by[d])
        state[b.name] = 2
        out.append(b)

    for b in bones:
        emit(b)
    return out


def _bone_segments(bones):
    """name -> (head, tail) used to find the nearest bone for unweighted vertices."""
    by = {b.name: b for b in bones}
    seg = {}
    for b in bones:
        h = np.asarray(b.head, float)
        if b.tail_bone and b.tail_bone in by:
            t = np.asarray(by[b.tail_bone].head, float)
        elif b.tail is not None:
            t = np.asarray(b.tail, float)
        else:
            t = h
        seg[b.name] = (h, t)
    return seg


def convert_bones(bones, index, scale):
    inv = 1.0 / scale
    out = []
    for b in bones:
        h = np.asarray(b.head, float)
        pb = X.PmxBone(name=b.name, name_en=b.name_en, pos=tuple(_swap(h) * inv), layer=int(b.layer),
                       rotatable=bool(b.rotatable), movable=bool(b.movable), visible=bool(b.visible),
                       controllable=bool(b.visible or b.movable), after_physics=bool(b.after_physics))
        pb.parent = index[b.parent] if b.parent else -1
        if b.tail_bone:
            pb.tail_bone, pb.tail_offset = index[b.tail_bone], None
        elif b.tail is not None:
            pb.tail_bone, pb.tail_offset = -1, tuple(_swap(np.asarray(b.tail, float) - h) * inv)
        else:
            pb.tail_bone, pb.tail_offset = -1, None
        if b.grant:
            pb.grant_rotate = bool(b.grant.get("rotate", True))
            pb.grant_move = bool(b.grant.get("move", False))
            pb.grant_parent = index[b.grant["parent"]]
            pb.grant_ratio = float(b.grant.get("ratio", 1.0))
        if b.fixed_axis is not None:
            a = np.asarray(b.fixed_axis, float)
            pb.fixed_axis = tuple(_swap(a / (np.linalg.norm(a) or 1.0)))
        if b.local_x is not None and b.local_z is not None:
            pb.local_x = tuple(_swap(np.asarray(b.local_x, float)))
            pb.local_z = tuple(_swap(np.asarray(b.local_z, float)))
        if b.ik:
            links = []
            for c in b.ik.get("chain", []):
                lim = c.get("limit")
                if lim:
                    lo = tuple(math.radians(float(x)) for x in lim[0])
                    hi = tuple(math.radians(float(x)) for x in lim[1])
                    links.append((index[c["bone"]], True, lo, hi))
                else:
                    links.append((index[c["bone"]], False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            pb.ik = X.PmxIK(target=index[b.ik["target"]], loops=int(b.ik.get("iterations", 40)),
                            angle=math.radians(float(b.ik.get("angle", 114.5916))), links=links)
        out.append(pb)
    return out


# ---------------------------------------------------------------- mesh helpers

def _corner_layout(faces):
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]]) if len(faces) else np.zeros(0, np.int64)
    vi = np.fromiter((i for f in faces for i in f), dtype=np.int64, count=int(sizes.sum()))
    face_of = np.repeat(np.arange(len(faces)), sizes)
    local = np.arange(len(vi)) - start[face_of]
    nxt = start[face_of] + (local + 1) % sizes[face_of]
    prv = start[face_of] + (local - 1) % sizes[face_of]
    return sizes, start, vi, face_of, nxt, prv


def _ear_clip(poly):
    """Triangles (local index triples) of a simple polygon given as (k, 3) points; handles concave polygons."""
    k = len(poly)
    if k == 3:
        return [(0, 1, 2)]
    nrm = np.zeros(3)
    for i in range(k):
        nrm += np.cross(poly[i], poly[(i + 1) % k])
    ln = np.linalg.norm(nrm)
    if ln < 1e-14:
        return [(0, i, i + 1) for i in range(1, k - 1)]
    nrm /= ln
    ax = np.argmax(np.abs(nrm))
    keep = [a for a in range(3) if a != ax]
    p2 = poly[:, keep]
    sign = 1.0 if nrm[ax] > 0 else -1.0
    if ax == 1:
        sign = -sign
    idx = list(range(k))
    tris = []

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * k:
        guard += 1
        n = len(idx)
        for j in range(n):
            a, b, c = idx[(j - 1) % n], idx[j], idx[(j + 1) % n]
            if sign * cross2(p2[a], p2[b], p2[c]) <= 1e-14:
                continue
            ok = True
            for q in idx:
                if q in (a, b, c):
                    continue
                s1 = sign * cross2(p2[a], p2[b], p2[q])
                s2 = sign * cross2(p2[b], p2[c], p2[q])
                s3 = sign * cross2(p2[c], p2[a], p2[q])
                if s1 >= -1e-14 and s2 >= -1e-14 and s3 >= -1e-14:
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                idx.pop(j)
                break
        else:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    else:                                    # degenerate input: fall back to a fan of what is left
        tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return tris


def triangulate(verts, faces):
    """(tri_corners (T,3) global corner indices, tri_face (T,)). Quads split along the shorter diagonal, larger ngons
    by ear clipping."""
    sizes, start, vi, _, _, _ = _corner_layout(faces)
    tc, tf = [], []
    all_tri = bool((sizes == 3).all()) if len(sizes) else True
    if all_tri:
        c = start[:, None] + np.arange(3)[None, :]
        return c.astype(np.int64), np.arange(len(faces))
    quad = np.nonzero(sizes == 4)[0]
    tri = np.nonzero(sizes == 3)[0]
    parts_c, parts_f = [], []
    if len(tri):
        parts_c.append(start[tri][:, None] + np.arange(3)[None, :])
        parts_f.append(tri)
    if len(quad):
        s = start[quad]
        p = verts[vi[s[:, None] + np.arange(4)[None, :]]]               # (q, 4, 3)
        d02 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        d13 = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
        use02 = d02 <= d13
        a = np.where(use02[:, None], np.array([[0, 1, 2]]), np.array([[1, 2, 3]]))
        b = np.where(use02[:, None], np.array([[0, 2, 3]]), np.array([[1, 3, 0]]))
        parts_c += [s[:, None] + a, s[:, None] + b]
        parts_f += [quad, quad]
    for f in np.nonzero(sizes > 4)[0]:
        s, k = int(start[f]), int(sizes[f])
        for t in _ear_clip(verts[vi[s:s + k]]):
            parts_c.append(np.array([[s + t[0], s + t[1], s + t[2]]]))
            parts_f.append(np.array([f]))
    tc = np.concatenate(parts_c).astype(np.int64)
    tf = np.concatenate(parts_f).astype(np.int64)
    return tc, tf


def corner_normals(verts, faces, smooth=True, sharp=(), custom=None):
    """Per-corner unit normals (C,3) and smoothing group ids (C,). `custom`: (n,3) per-vertex normals (used as given).
    Otherwise angle-weighted vertex normals, split along `sharp` edges (a vertex gets one normal per smoothing group);
    `smooth=False` gives flat faces (one group per face)."""
    sizes, start, vi, face_of, nxt, prv = _corner_layout(faces)
    C = len(vi)
    if custom is not None:
        n = np.asarray(custom, float)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(ln > 1e-12, ln, 1.0)
        return n[vi], np.zeros(C, np.int64)
    p = verts[vi]
    fn_raw = np.zeros((len(faces), 3))
    cr = np.cross(p, p[nxt])
    np.add.at(fn_raw, face_of, cr)
    fl = np.linalg.norm(fn_raw, axis=1, keepdims=True)
    fn = fn_raw / np.where(fl > 1e-14, fl, 1.0)
    if not smooth:
        return fn[face_of], face_of.astype(np.int64)
    a = p[nxt] - p
    b = p[prv] - p
    ang = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
    contrib = fn[face_of] * ang[:, None]
    nv = len(verts)
    vn = np.zeros((nv, 3))
    np.add.at(vn, vi, contrib)
    group = np.zeros(C, np.int64)
    cn = vn[vi].copy()
    sharp_set = {(min(i, j), max(i, j)) for i, j in sharp}
    if sharp_set:
        touched = {v for e in sharp_set for v in e}
        by_vertex = {}
        for c in range(C):
            v = int(vi[c])
            if v in touched:
                by_vertex.setdefault(v, []).append(c)
        for v, corners in by_vertex.items():
            parent = {c: c for c in corners}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            edge_corners = {}
            for c in corners:
                for nb in (int(vi[nxt[c]]), int(vi[prv[c]])):
                    e = (min(v, nb), max(v, nb))
                    if e not in sharp_set:
                        edge_corners.setdefault(e, []).append(c)
            for cs in edge_corners.values():
                for c in cs[1:]:
                    parent[find(c)] = find(cs[0])
            roots = {}
            for c in corners:
                roots.setdefault(find(c), []).append(c)
            for gi, cs in enumerate(roots.values()):
                acc = np.zeros(3)
                for c in cs:
                    acc += contrib[c]
                for c in cs:
                    cn[c] = acc
                    group[c] = gi
    ln = np.linalg.norm(cn, axis=1, keepdims=True)
    cn = np.where(ln > 1e-12, cn / np.where(ln > 1e-12, ln, 1.0), fn[face_of])
    return cn, group


def top4(W, floor=1e-5):
    """(idx (n,4), w (n,4)) of the four largest weights per row, renormalised; rows with no weight give zeros."""
    n, k = W.shape
    if k == 0:
        return np.zeros((n, 4), np.int64), np.zeros((n, 4))
    kk = min(4, k)
    part = np.argsort(-W, axis=1)[:, :kk]
    w = np.take_along_axis(W, part, axis=1)
    w = np.where(w < floor, 0.0, w)
    if kk < 4:
        part = np.concatenate([part, np.zeros((n, 4 - kk), np.int64)], axis=1)
        w = np.concatenate([w, np.zeros((n, 4 - kk))], axis=1)
    tot = w.sum(axis=1, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.0)
    return part.astype(np.int64), w


def _nearest_bone(points, seg_names, seg_a, seg_b):
    """Index of the nearest bone segment for each point (brute force in chunks)."""
    ab = seg_b - seg_a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    out = np.zeros(len(points), np.int64)
    for s in range(0, len(points), 2048):
        P = points[s:s + 2048]
        ap = P[:, None, :] - seg_a[None, :, :]
        t = np.clip((ap * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
        d = np.linalg.norm(ap - t[..., None] * ab[None], axis=2)
        out[s:s + 2048] = np.argmin(d, axis=1)
    return out


# ---------------------------------------------------------------- the assembler

def assemble(parts, name="model", scale=0.08, comment="", name_en="", subdivide=True):
    """Merge `parts` into one `Assembled` model. See the module docstring."""
    inv = 1.0 / scale
    warnings = []
    # ---- bones
    all_bones = [b for p in parts for b in p.bones]
    bones = order_bones(all_bones)
    bone_index = {b.name: i for i, b in enumerate(bones)}
    if len(bone_index) != len(bones):
        raise ValueError("duplicate bone names across parts")
    pmx_bones = convert_bones(bones, bone_index, scale)
    segs = _bone_segments(bones)
    deform_names = [b.name for b in bones if b.deform and b.name in segs]
    seg_a = np.array([segs[n][0] for n in deform_names]) if deform_names else np.zeros((0, 3))
    seg_b = np.array([segs[n][1] for n in deform_names]) if deform_names else np.zeros((0, 3))
    seg_idx = np.array([bone_index[n] for n in deform_names], np.int64)

    # ---- materials
    mats, mat_index = [], {}
    for p in parts:
        for m in p.materials:
            mat_index[m.name] = len(mats)
            mats.append(m)
    tex_names = []

    def tex_ref(f):
        if not f:
            return -1
        if f not in tex_names:
            tex_names.append(f)
        return tex_names.index(f)

    # ---- meshes -> vertices, triangles
    verts_out = {k: [] for k in ("pos", "nrm", "uv", "bidx", "bw", "mesh", "src")}
    tri_chunks, mat_chunks = [], []
    morph_order, morph_decl = [], {}
    for p in parts:
        for mo in p.morphs:
            if mo.name not in morph_decl:
                morph_decl[mo.name] = mo
                morph_order.append(mo.name)
    morph_offsets = {}                                       # name -> list of (global_vertex_index_array, (k,3) model offsets)
    base = 0
    mesh_names = []
    for p in parts:
        for mesh in p.meshes:
            v = np.asarray(mesh.verts, float)
            faces = [list(map(int, f)) for f in mesh.faces]
            n0 = len(v)
            if n0 == 0 or not faces:
                warnings.append(f"{p.name}/{mesh.name}: empty, skipped")
                continue
            uv = None if mesh.uv is None else np.asarray(mesh.uv, float)
            fmat = np.zeros(len(faces), np.int64) if mesh.face_mat is None else np.asarray(mesh.face_mat, np.int64)
            names = list(mesh.weights)
            W = np.stack([np.asarray(mesh.weights[k], float) for k in names], axis=1) if names else np.zeros((n0, 0))
            morphs = {k: np.asarray(d, float) for k, d in mesh.morphs.items()}
            normals = None if mesh.normals is None else np.asarray(mesh.normals, float)
            sharp = [tuple(e) for e in mesh.sharp]
            if subdivide and mesh.subsurf > 0:
                from . import subdiv
                # same surface as Blender's Subdivision modifier (boundary 'smooth': checked to 1e-7 against it)
                sd = subdiv.subdivide(v, faces, levels=int(mesh.subsurf), creases=dict(mesh.crease) or None, uv=uv,
                                      corners="smooth")
                v, faces, uv = sd.verts, [list(map(int, f)) for f in sd.faces], sd.uv
                fmat = fmat[sd.face_parent]
                if W.shape[1]:
                    W = np.asarray(sd.apply(W, attr=True))
                morphs = {k: np.asarray(sd.apply(d)) for k, d in morphs.items()}
                if normals is not None:
                    nn = np.asarray(sd.apply(normals, attr=True))
                    ln = np.linalg.norm(nn, axis=1, keepdims=True)
                    normals = nn / np.where(ln > 1e-12, ln, 1.0)
                sharp = sd.map_edges(sharp) if sharp else []
            sizes, start, vi, face_of, _, _ = _corner_layout(faces)
            C = len(vi)
            if uv is None:
                uv = np.zeros((C, 2))
            if uv.shape != (C, 2):
                raise ValueError(f"{p.name}/{mesh.name}: uv has {uv.shape}, expected ({C}, 2)")
            cn, _group = corner_normals(v, faces, smooth=mesh.smooth, sharp=sharp, custom=normals)
            nkey = np.zeros((C, 3), np.int64) if normals is not None else np.rint(cn * 1e4).astype(np.int64)
            key = np.stack([vi, np.rint(uv[:, 0] * 1e5).astype(np.int64), np.rint(uv[:, 1] * 1e5).astype(np.int64)],
                           axis=1)
            key = np.concatenate([key, nkey], axis=1)
            uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            src = uniq[:, 0]
            nv = len(uniq)
            pos_model = v[src]
            nrm_model = cn[first]
            uv_u = uv[first]
            # weights
            if names:
                gidx = np.array([bone_index[k] for k in names], np.int64)
                part_i, part_w = top4(W)
                bi_src = np.where(part_w > 0, gidx[part_i], 0)
            else:
                bi_src = np.zeros((n0 if not (subdivide and mesh.subsurf > 0) else len(v), 4), np.int64)
                part_w = np.zeros(bi_src.shape)
            bidx, bw = bi_src[src], part_w[src]
            unweighted = bw.sum(axis=1) <= 0
            if unweighted.any():
                cnt = int(unweighted.any() and unweighted.sum())
                if len(seg_idx):
                    near = _nearest_bone(pos_model[unweighted], deform_names, seg_a, seg_b)
                    bidx[unweighted, 0] = seg_idx[near]
                bw[unweighted, 0] = 1.0
                warnings.append(f"{p.name}/{mesh.name}: {cnt} of {nv} vertices had no weights; attached to the nearest bone")
            for k, arr in (("pos", pos_model), ("nrm", nrm_model), ("uv", uv_u), ("bidx", bidx), ("bw", bw)):
                verts_out[k].append(arr)
            verts_out["mesh"].append(np.full(nv, len(mesh_names), np.int64))
            verts_out["src"].append(src)
            mesh_names.append(f"{p.name}/{mesh.name}")
            # triangles
            tc, tf = triangulate(v, faces)
            tri_v = inverse[tc] + base
            pm = pos_model[inverse[tc]]
            deg = int((np.linalg.norm(np.cross(pm[:, 1] - pm[:, 0], pm[:, 2] - pm[:, 0]), axis=1) < 1e-12).sum())
            if deg:
                warnings.append(f"{p.name}/{mesh.name}: {deg} degenerate (zero-area) triangles")
            tri_chunks.append(tri_v)
            if mesh.mats:
                gm = np.array([mat_index[m] for m in mesh.mats], np.int64)
            elif mats:
                gm = np.zeros(1, np.int64)
            else:
                gm = np.zeros(1, np.int64)
            fm = np.clip(fmat[tf], 0, len(gm) - 1)
            mat_chunks.append(gm[fm])
            # morphs
            for mk, off in morphs.items():
                if mk not in morph_decl:
                    morph_decl[mk] = None
                    morph_order.append(mk)
                nz = np.abs(off).max(axis=1) > 1e-7
                sel = nz[src]
                if sel.any():
                    morph_offsets.setdefault(mk, []).append((np.nonzero(sel)[0] + base, off[src[sel]]))
            base += nv

    pos = np.concatenate(verts_out["pos"]) if verts_out["pos"] else np.zeros((0, 3))
    nrm = np.concatenate(verts_out["nrm"]) if verts_out["nrm"] else np.zeros((0, 3))
    uvs = np.concatenate(verts_out["uv"]) if verts_out["uv"] else np.zeros((0, 2))
    bidx = np.concatenate(verts_out["bidx"]) if verts_out["bidx"] else np.zeros((0, 4), np.int64)
    bw = np.concatenate(verts_out["bw"]) if verts_out["bw"] else np.zeros((0, 4))
    tris = np.concatenate(tri_chunks) if tri_chunks else np.zeros((0, 3), np.int64)
    tri_mat = np.concatenate(mat_chunks) if mat_chunks else np.zeros(0, np.int64)
    order = np.argsort(tri_mat, kind="stable")
    tris, tri_mat = tris[order], tri_mat[order]

    # ---- PMX vertices
    pp, nn = _swap(pos) * inv, _swap(nrm)
    vertices = []
    for i in range(len(pos)):
        nb = int((bw[i] > 0).sum())
        if nb <= 1:
            bones_t, w_t = (int(bidx[i, 0]),), (1.0,)
        elif nb == 2:
            o = np.argsort(-bw[i])[:2]
            bones_t, w_t = (int(bidx[i, o[0]]), int(bidx[i, o[1]])), (float(bw[i, o[0]]), float(bw[i, o[1]]))
        else:
            o = np.argsort(-bw[i])[:4]
            bones_t = tuple(int(bidx[i, j]) for j in o)
            w_t = tuple(float(bw[i, j]) for j in o)
        vertices.append(X.PmxVertex(pos=tuple(pp[i]), normal=tuple(nn[i]), uv=(float(uvs[i, 0]), float(1.0 - uvs[i, 1])),
                                    bones=bones_t, weights=w_t))

    # ---- materials -> PMX
    pmx_mats = []
    counts = np.bincount(tri_mat, minlength=len(mats)) if len(mats) else np.zeros(0, np.int64)
    for i, m in enumerate(mats):
        mode = {"none": 0, "mul": 1, "add": 2}.get(m.sphere_mode, 0) if m.sphere else 0
        pmx_mats.append(X.PmxMaterial(
            name=m.name, name_en=m.name_en, diffuse=tuple(m.diffuse), specular=tuple(m.specular),
            shininess=float(m.shininess), ambient=tuple(m.ambient), double_sided=m.double_sided,
            ground_shadow=m.drop_shadow, self_shadow_map=m.self_shadow_map, self_shadow=m.self_shadow, edge=m.edge,
            edge_color=tuple(m.edge_color), edge_size=float(m.edge_size), texture=tex_ref(m.texture),
            sphere_texture=tex_ref(m.sphere), sphere_mode=mode, toon_shared=False, toon=tex_ref(m.toon),
            memo=m.comment, index_count=int(counts[i]) * 3))
    if tris.size and len(mats) == 0:
        raise ValueError("meshes have faces but no part declared a material")
    unused = [m.name for m, c in zip(mats, counts) if c == 0]
    if unused:
        warnings.append(f"materials without faces: {', '.join(unused[:6])}")

    # ---- morphs
    pmx_morphs, morph_index = [], {}
    panel_rank = {"eye": 0, "brow": 1, "mouth": 2, "other": 3}
    ordered = sorted(morph_order, key=lambda k: (panel_rank.get(morph_decl[k].panel if morph_decl.get(k) else "other", 3),
                                                  morph_order.index(k)))
    for mk in ordered:
        decl = morph_decl.get(mk)
        entries = morph_offsets.get(mk, [])
        if not entries:
            warnings.append(f"morph {mk!r} has no offsets (declared but no mesh moves)")
            continue
        idx = np.concatenate([e[0] for e in entries])
        off = np.concatenate([e[1] for e in entries])
        order_i = np.argsort(idx, kind="stable")
        idx, off = idx[order_i], _swap(off[order_i]) * inv
        morph_index[mk] = len(pmx_morphs)
        pmx_morphs.append(X.PmxMorph(name=mk, name_en=(decl.name_en if decl else ""), panel=PANEL[decl.panel if decl else "other"],
                                     kind="vertex", offsets=[(int(i), tuple(o)) for i, o in zip(idx, off)]))

    # ---- rigid bodies and joints
    pmx_bodies, body_index = [], {}
    for p in parts:
        for rb in p.bodies:
            body_index[rb.name] = len(pmx_bodies)
            if rb.shape == "box":
                size = tuple(_swap(np.asarray(rb.size, float)) * inv)
            elif rb.shape == "sphere":
                size = (float(rb.size[0]) * inv, 0.0, 0.0)
            else:
                size = (float(rb.size[0]) * inv, float(rb.size[1]) * inv, 0.0)
            mask = 0xFFFF
            for g in rb.no_collide:
                mask &= ~(1 << int(g))
            pmx_bodies.append(X.PmxBody(
                name=rb.name, name_en="", bone=bone_index[rb.bone] if rb.bone else -1, group=int(rb.group), mask=mask,
                shape={"sphere": 0, "box": 1, "capsule": 2}[rb.shape], size=size,
                pos=tuple(_swap(np.asarray(rb.location, float)) * inv), rot=pmx_euler(rb.rotation), mass=float(rb.mass),
                linear_damping=float(rb.damping[0]), angular_damping=float(rb.damping[1]), restitution=float(rb.bounce),
                friction=float(rb.friction), mode={"static": 0, "dynamic": 1, "dynamic_bone": 2}[rb.mode]))
    pmx_joints = []
    for p in parts:
        for j in p.joints:
            pmx_joints.append(X.PmxJoint(
                name=j.name, kind=0, body_a=body_index[j.a], body_b=body_index[j.b],
                pos=tuple(_swap(np.asarray(j.location, float)) * inv), rot=pmx_euler(j.rotation),
                move_lo=tuple(_swap(np.asarray(j.move_lo, float)) * inv),
                move_hi=tuple(_swap(np.asarray(j.move_hi, float)) * inv),
                rot_lo=tuple(-_swap(np.asarray(j.rot_hi, float))), rot_hi=tuple(-_swap(np.asarray(j.rot_lo, float))),
                spring_move=tuple(_swap(np.asarray(j.spring_move, float))),
                spring_rot=tuple(_swap(np.asarray(j.spring_rot, float)))))

    # ---- display frames
    frames = [X.PmxFrame(name="Root", name_en="Root", special=True,
                         items=[("bone", bone_index["全ての親"])] if "全ての親" in bone_index else []),
              X.PmxFrame(name="表情", name_en="Exp", special=True,
                         items=[("morph", i) for i in range(len(pmx_morphs))])]
    framed = {"全ての親"}
    merged = {}
    for p in parts:
        for fname, members in p.frames.items():
            merged.setdefault(fname, []).extend(members)
    for fname, members in merged.items():
        items = []
        for bn in members:
            if bn not in bone_index:
                warnings.append(f"display frame {fname!r}: unknown bone {bn!r}")
            elif bn not in framed:
                framed.add(bn)
                items.append(("bone", bone_index[bn]))
        if items:
            frames.append(X.PmxFrame(name=fname, name_en=fname, items=items))
    rest = [("bone", bone_index[b.name]) for b in bones if b.name not in framed and b.visible]
    if rest:
        frames.append(X.PmxFrame(name="その他", name_en="Other", items=rest))

    model = X.PmxModel(name=name, name_en=name_en or name, comment=comment, comment_en=comment,
                       vertices=vertices, faces=[int(i) for i in tris[:, [0, 2, 1]].reshape(-1)],
                       textures=[f"{TEX_DIR}\\{f}" for f in tex_names], materials=pmx_mats, bones=pmx_bones,
                       morphs=pmx_morphs, frames=frames, bodies=pmx_bodies, joints=pmx_joints)
    expected = {
        "pos": pos, "normal": nrm, "uv": uvs, "bone_idx": bidx, "bone_w": bw, "mesh": np.concatenate(verts_out["mesh"])
        if verts_out["mesh"] else np.zeros(0, np.int64),
        "tris": tris, "tri_mat": tri_mat,
        "bone_head": np.array([b.head for b in bones], float),
    }
    morph_arrays = {}
    for mk, mi in morph_index.items():
        m = pmx_morphs[mi]
        ids = np.array([o[0] for o in m.offsets], np.int64)
        vec = _swap(np.array([o[1] for o in m.offsets], float)) * scale
        morph_arrays[mk] = (ids, vec)
    stats = {"vertices": len(vertices), "triangles": int(len(tris)), "bones": len(bones), "materials": len(mats),
             "morphs": len(pmx_morphs), "bodies": len(pmx_bodies), "joints": len(pmx_joints), "textures": len(tex_names),
             "dynamic_bodies": sum(1 for b in pmx_bodies if b.mode != 0), "meshes": len(mesh_names)}
    expected["morphs"] = morph_arrays
    def _f(x):
        return None if x is None else [float(c) for c in x]
    meta = {
        "scale": scale, "bone_names": [b.name for b in bones], "materials": [m.name for m in mats],
        "bones": [{"name": b.name, "name_en": b.name_en, "parent": b.parent, "head": _f(b.head),
                   "ik": b.ik if b.ik else None, "grant": dict(b.grant) if b.grant else None,
                   "fixed_axis": _f(b.fixed_axis), "local_x": _f(b.local_x), "local_z": _f(b.local_z),
                   "visible": bool(b.visible), "movable": bool(b.movable), "rotatable": bool(b.rotatable),
                   "layer": int(b.layer), "after_physics": bool(b.after_physics)} for b in bones],
        "material_data": [{"name": m.name, "diffuse": _f(m.diffuse), "specular": _f(m.specular),
                           "ambient": _f(m.ambient), "shininess": float(m.shininess), "double_sided": bool(m.double_sided),
                           "edge": bool(m.edge), "edge_color": _f(m.edge_color), "edge_size": float(m.edge_size),
                           "texture": m.texture, "toon": m.toon, "sphere": m.sphere,
                           "sphere_type": {"none": 0, "mul": 1, "add": 2}.get(m.sphere_mode, 0) if m.sphere else 0}
                          for m in mats],
        "morph_data": [{"name": mk, "panel": (morph_decl[mk].panel if morph_decl.get(mk) else "other"),
                        "name_en": (morph_decl[mk].name_en if morph_decl.get(mk) else "")} for mk in morph_index],
        "frames": [{"name": f.name, "special": bool(f.special),
                    "items": [(k, (pmx_bones[i].name if k == "bone" else pmx_morphs[i].name)) for k, i in f.items]}
                   for f in frames],
        "bodies": [{"name": rb.name, "bone": rb.bone, "shape": rb.shape, "size": [float(x) for x in rb.size],
                    "location": [float(x) for x in rb.location], "rotation": [float(x) for x in rb.rotation],
                    "mode": rb.mode, "group": int(rb.group), "no_collide": [int(g) for g in rb.no_collide]}
                   for p in parts for rb in p.bodies],
        "joints": [{"name": j.name, "a": j.a, "b": j.b, "location": [float(x) for x in j.location],
                    "rotation": [float(x) for x in j.rotation], "move_lo": [float(x) for x in j.move_lo],
                    "move_hi": [float(x) for x in j.move_hi], "rot_lo": [float(x) for x in j.rot_lo],
                    "rot_hi": [float(x) for x in j.rot_hi], "spring_move": _f(j.spring_move),
                    "spring_rot": _f(j.spring_rot)} for p in parts for j in p.joints],
    }
    return Assembled(pmx=model, textures=list(tex_names), bone_index=bone_index, morph_index=morph_index,
                     material_index=mat_index, body_index=body_index, expected=expected, meta=meta, stats=stats,
                     warnings=warnings, scale=scale)
