"""Take parts of a PMX model into mk's model space (bpy-free: pmx_io, numpy, PIL).

`take(path, materials, ...)` cuts the named materials out of a PMX file and hands them over in the form the part builders
use: one shared vertex array (so vertices that two kept materials share, such as a neck ring, stay shared), faces per
material, weights by bone NAME, the bones the kept parts need (pruned, parents first), the vertex / material / group morphs
that still make sense, and optionally the rigid bodies and joints on the kept bones. Positions are scaled to the character
by a uniform scale about the model origin and an optional head enlargement blended by weight (see `take`), so the neck
blends smoothly and the morph offsets of head vertices follow.

Conventions: the PMX is read with `pmx_io` (PMX space, Y up, the face looks to -Z, `unit` metres per PMX unit); mk model
space is metres, Z up, the character faces -Y, her left is +X: (x, y, z)_pmx -> (x, z, y) * unit. PMX triangles are
clockwise, `Take.faces` are counter-clockwise seen from outside (the Part convention). uv are v UP (the Part convention;
`1 - v` of the PMX). Normals are unit vectors in model axes.

Nothing about a particular model lives here: which materials to keep, the scale, the head bone and the colour edits come from
the caller (the project's spec).

    t = take(path, ["skin", "face"], scale=0.95, head={"bone": "頭", "factor": 0.985 / 0.95}, skip_morphs=["x"])
    t.verts, t.faces["skin"], t.weights["頭"], t.bones, t.morphs        # everything the builders need
    piece = t.piece(["face"])                                            # arrays restricted to one or more materials
    mats = part_materials(t, ctx, ["face"], recolor=[...])               # part.Material objects with edited texture copies
"""
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from . import pmx_io
from .part import Bone, Joint, Material, RigidBody

PANELS = {0: "other", 1: "brow", 2: "eye", 3: "mouth", 4: "other"}      # PMX morph panel -> part panel
_SPHERE = {0: "none", 1: "mul", 2: "add", 3: "sub"}


def to_model(P, unit=1.0):
    """PMX points / vectors (n, 3) -> model axes (x, z, y) times `unit`."""
    P = np.asarray(P, float).reshape(-1, 3)
    return np.stack([P[:, 0], P[:, 2], P[:, 1]], -1) * float(unit)


# ------------------------------------------------------------------------------------------------------------ data
@dataclass
class TakeMaterial:
    """A kept material: MMD settings as in the file, texture / sphere / toon images as absolute paths (None = none).
    `toon_shared`: the toon is one of MMD's shared toons (`toon_index` 0..9) rather than an image."""
    name: str
    name_en: str
    diffuse: tuple
    specular: tuple
    shininess: float
    ambient: tuple
    double_sided: bool
    drop_shadow: bool
    self_shadow_map: bool
    self_shadow: bool
    edge: bool
    edge_color: tuple
    edge_size: float
    texture: Optional[Path] = None
    sphere: Optional[Path] = None
    sphere_mode: str = "none"
    toon: Optional[Path] = None
    toon_shared: bool = False
    toon_index: int = -1
    memo: str = ""


class MaterialOffset(NamedTuple):
    """One entry of a material morph: `material` is a kept material's name or None for "all materials"; `op` 0 multiplies,
    1 adds; the other fields are as in pmx_io.MaterialMorphOffset."""
    material: Optional[str]
    op: int
    diffuse: tuple
    specular: tuple
    shininess: float
    ambient: tuple
    edge_color: tuple
    edge_size: float
    texture: tuple
    sphere: tuple
    toon: tuple


@dataclass
class TakeMorph:
    """A morph that survived. kind 'vertex': `ids` (k,) indices into Take.verts and `offsets` (k, 3) model-space metres (scaled,
    head-blended). kind 'material': `material_offsets`. kind 'group': `members` [(morph name, ratio)], only the members that
    survived. `panel`: eye | brow | mouth | other."""
    name: str
    name_en: str
    panel: str
    kind: str
    ids: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None
    material_offsets: list = field(default_factory=list)
    members: list = field(default_factory=list)


@dataclass
class Piece:
    """Arrays for a subset of the kept materials (`Take.piece`): the vertices those materials use, renumbered. `faces` are
    tuples of local indices (CCW), `uv` one (u, v) per face corner (v up) in face order, `face_mat` the index into `mats`
    per face, `ids` the index of every vertex in Take.verts."""
    mats: list
    verts: np.ndarray
    faces: list
    uv: np.ndarray
    normals: np.ndarray
    face_mat: np.ndarray
    weights: dict
    morphs: dict
    ids: np.ndarray
    vert_uv: np.ndarray


@dataclass
class Take:
    verts: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    edge: np.ndarray
    src_ids: np.ndarray
    faces: dict
    weights: dict
    materials: dict
    bones: list
    morphs: list
    bodies: list
    joints: list
    frames: dict
    head_weight: np.ndarray
    info: dict
    root: Path

    def bone(self, name):
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(f"bone {name!r} was not taken")

    def has_bone(self, name):
        return any(b.name == name for b in self.bones)

    def piece(self, names):
        """The vertices, faces, uv, normals, weights and vertex morphs of the materials `names` (kept material names)."""
        names = [names] if isinstance(names, str) else list(names)
        for n in names:
            if n not in self.faces:
                raise KeyError(f"material {n!r} was not taken (have {list(self.faces)})")
        tris = [self.faces[n] for n in names]
        used = np.unique(np.concatenate([t.ravel() for t in tris])) if tris else np.zeros(0, int)
        local = np.full(len(self.verts), -1, int)
        local[used] = np.arange(len(used))
        faces, fmat = [], []
        for k, t in enumerate(tris):
            faces += [tuple(int(local[i]) for i in tri) for tri in t]
            fmat += [k] * len(t)
        vuv = self.uv[used]
        corner = np.array([vuv[i] for f in faces for i in f], float).reshape(-1, 2)
        weights = {b: w[used] for b, w in self.weights.items() if np.any(w[used] > 0)}
        morphs = {}
        for m in self.morphs:
            if m.kind != "vertex" or m.ids is None:
                continue
            sel = np.isin(m.ids, used)
            if not sel.any():
                continue
            arr = np.zeros((len(used), 3))
            np.add.at(arr, local[m.ids[sel]], m.offsets[sel])
            morphs[m.name] = arr
        return Piece(names, self.verts[used], faces, corner, self.normals[used], np.array(fmat, int), weights, morphs, used, vuv)


# ------------------------------------------------------------------------------------------------------------ helpers
def _find_file(root, rel):
    """Case- and separator-insensitive lookup of a path relative to the PMX's folder; None when absent."""
    rel = str(rel).replace("\\", "/")
    p = Path(root) / rel
    if p.is_file():
        return p
    cur = Path(root)
    for part in [x for x in rel.split("/") if x]:
        if not cur.is_dir():
            return None
        hit = [c for c in cur.iterdir() if c.name.lower() == part.lower()]
        if not hit:
            return None
        cur = hit[0]
    return cur if cur.is_file() else None


def _tex(model, root, idx):
    if idx is None or idx < 0 or idx >= len(model.textures):
        return None
    f = _find_file(root, model.textures[idx])
    return None if f is None else f.resolve()


def _euler_xyz(R):
    """Blender 'XYZ' Euler (R = Rz Ry Rx) of a rotation matrix."""
    sy = -R[2, 0]
    e1 = math.asin(max(-1.0, min(1.0, sy)))
    if abs(sy) < 1.0 - 1e-9:
        return (math.atan2(R[2, 1], R[2, 2]), e1, math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(-R[1, 2], R[1, 1]), e1, 0.0)


def _model_euler(pmx_rot):
    """Model-space XYZ Euler (rad) of a PMX rigid body / joint rotation vector: the inverse of assemble.pmx_euler."""
    from .assemble import _rx, _rz, _ry                      # noqa: F401  (the same rotation helpers the assembler uses)
    a, b, c = pmx_rot
    ex, ez, ey = -a, -b, -c                                  # pmx_euler returns (-ex, -ez, -ey)
    R = _rz(ez) @ _rx(ex) @ _ry(ey)                          # Blender YXZ
    return _euler_xyz(R)


def _order(names, parents):
    """Indices ordered parents first (stable)."""
    out, seen = [], set()

    def emit(i):
        if i in seen:
            return
        seen.add(i)
        p = parents[i]
        if p >= 0:
            emit(p)
        out.append(i)
    for i in range(len(names)):
        emit(i)
    return out


# ------------------------------------------------------------------------------------------------------------- take
def take(path, materials, *, unit=0.08, scale=1.0, head=None, skip_morphs=(), bodies=False, keep_bones=(), drop_bones=(),
         base_dir=None):
    """Cut `materials` (names) out of the PMX at `path`.

    unit        metres per PMX unit (0.08 for MMD models)
    scale       uniform scale about the model origin (the floor), applied to everything
    head        None, or {"bone": name, "factor": f}: after the scale, every vertex is blended towards a uniform scale `f` about
                the head of that bone by its weight on the bone's SUBTREE (the bone and its descendants: eyes, tongue, ...):
                p' = pivot + (p - pivot) * (1 + w (f - 1)); the bones of the subtree get the full factor, the vertex morph
                offsets the same per-vertex factor (so a head vertex's offset is scaled by f, a neck vertex's by less)
    skip_morphs names of morphs to drop; material morphs are restricted to the kept materials, group morphs to the members
                that survive (a morph left empty is dropped)
    bodies      True: also take the rigid bodies and joints that sit on kept bones (static colliders, ...)
    keep_bones / drop_bones   names to keep (with their ancestors) / drop (with their subtrees) regardless of the rule

    Bones kept: those weighted by kept vertices and their ancestors, plus every bone that no vertex uses (control, IK, twist,
    D, tip bones), minus any bone whose subtree only serves vertices of dropped materials, and any bone below a dropped one;
    IK targets / links and grant parents of kept bones are kept too."""
    path = Path(str(path)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    model = pmx_io.read(str(path))
    root = path.parent
    mnames = [m.name for m in model.materials]
    wanted = list(materials)
    missing = [w for w in wanted if w not in mnames]
    if missing:
        raise ValueError(f"take: materials {missing} are not in {path.name} (it has {mnames})")
    start, off = [], 0
    for m in model.materials:
        start.append(off // 3)
        off += m.index_count
    tri_all = np.asarray(model.faces, int).reshape(-1, 3)

    def tris_of(mi):
        t = tri_all[start[mi]:start[mi] + model.materials[mi].index_count // 3][:, ::-1]           # CW -> CCW
        return t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]               # no repeated vertices

    kept_idx = [mnames.index(w) for w in wanted]
    kept_tris = {w: tris_of(mnames.index(w)) for w in wanted}
    ids = np.unique(np.concatenate([t.ravel() for t in kept_tris.values()]))
    local = np.full(len(model.vertices), -1, int)
    local[ids] = np.arange(len(ids))
    faces = {w: local[t] for w, t in kept_tris.items()}
    dropped_ids = set()
    for mi, m in enumerate(model.materials):
        if mi not in kept_idx:
            dropped_ids |= set(tris_of(mi).ravel().tolist())
    bone_list = model.bones
    bnames = [b.name for b in bone_list]
    if len(set(bnames)) != len(bnames):
        raise ValueError(f"take: {path.name} has duplicate bone names")
    parents = [b.parent for b in bone_list]

    # ---- bones the kept vertices use, the bones of dropped parts, the head subtree
    kept_w, drop_w = set(), set()
    for vi in ids:
        v = model.vertices[int(vi)]
        kept_w.update(b for b, w in zip(v.bones, v.weights) if b >= 0 and w > 0)
    for vi in dropped_ids - set(ids.tolist()):
        v = model.vertices[int(vi)]
        drop_w.update(b for b, w in zip(v.bones, v.weights) if b >= 0 and w > 0)
    children = {}
    for i, p in enumerate(parents):
        children.setdefault(p, []).append(i)

    def subtree(i):
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack += children.get(j, [])
        return out
    sub = {i: subtree(i) for i in range(len(bone_list))}
    forced_drop = set()
    for n in drop_bones:
        if n not in bnames:
            raise ValueError(f"take: drop_bones {n!r} is not a bone of {path.name}")
        forced_drop.update(sub[bnames.index(n)])

    def with_ancestors(j, into):
        while j >= 0 and j not in into:
            into.add(j)
            j = parents[j]
    required = set()                                                      # weighted by kept vertices, and their ancestors
    for j in kept_w:
        with_ancestors(j, required)
    for n in keep_bones:
        if n not in bnames:
            raise ValueError(f"take: keep_bones {n!r} is not a bone of {path.name}")
        with_ancestors(bnames.index(n), required)
    kept = set(required)
    changed = True
    while changed:                                                       # to a fixpoint: closing over IK / grants can free more control bones
        changed = False
        for i in _order(bnames, parents):                                # control / IK / twist / D / tip bones that no vertex uses
            if i in kept or i in forced_drop:
                continue
            if not any(j in drop_w for j in sub[i]) and (parents[i] < 0 or parents[i] in kept):
                kept.add(i)
                changed = True
        for i in sorted(kept):                                           # IK links / targets and grant parents must exist
            b = bone_list[i]
            need = []
            if (b.grant_rotate or b.grant_move) and b.grant_parent >= 0:
                need.append(b.grant_parent)
            if b.ik is not None:
                need += [b.ik.target] + [lk[0] for lk in b.ik.links]
            for j in need:
                if j not in kept:
                    with_ancestors(j, kept)
                    changed = True
    kept -= forced_drop - required
    order = [i for i in _order(bnames, parents) if i in kept]
    dropped_bone_names = [bnames[i] for i in range(len(bnames)) if i not in kept]

    # ---- the transform: uniform scale about the origin, then the head blend
    P = to_model([v.pos for v in model.vertices], unit)
    N = to_model([v.normal for v in model.vertices], 1.0)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    UV = np.array([v.uv for v in model.vertices], float).reshape(-1, 2)
    EDGE = np.array([v.edge_scale for v in model.vertices], float)
    scale = float(scale)
    head_set, pivot, factor = set(), None, 1.0
    if head:
        hb = head.get("bone")
        if hb not in bnames:
            raise ValueError(f"take: head bone {hb!r} is not a bone of {path.name}")
        head_set = set(sub[bnames.index(hb)])
        pivot = to_model(bone_list[bnames.index(hb)].pos, unit)[0] * scale
        factor = float(head.get("factor", 1.0))
    verts = P[ids] * scale
    normals, uv, edge = N[ids], UV[ids], EDGE[ids]
    uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)
    n = len(ids)
    weights = {}
    for k, vi in enumerate(ids):
        v = model.vertices[int(vi)]
        for b, w in zip(v.bones, v.weights):
            if b < 0 or w <= 0:
                continue
            name = bnames[b]
            if name not in weights:
                weights[name] = np.zeros(n)
            weights[name][k] += float(w)
    tot = sum(weights.values()) if weights else np.ones(n)
    tot = np.where(tot > 0, tot, 1.0)
    weights = {b: w / tot for b, w in weights.items()}
    head_w = np.zeros(n)
    for b in head_set:
        if bnames[b] in weights:
            head_w += weights[bnames[b]]
    sv = 1.0 + head_w * (factor - 1.0)
    if head:
        verts = pivot + (verts - pivot) * sv[:, None]

    def bone_point(p, owner):
        q = to_model(p, unit)[0] * scale
        return pivot + factor * (q - pivot) if (head and owner in head_set) else q

    # ---- bones
    bones = []
    kept_names = {bnames[i] for i in kept}
    for i in order:
        b = bone_list[i]
        h = bone_point(b.pos, i)
        tail, tail_bone = None, ""
        if b.tail_offset is not None:
            vec = to_model(b.tail_offset, unit)[0] * scale * (factor if (head and i in head_set) else 1.0)
            tail = tuple(float(x) for x in (h + vec)) if np.linalg.norm(vec) > 1e-9 else None
        elif b.tail_bone >= 0:
            if bnames[b.tail_bone] in kept_names:
                tail_bone = bnames[b.tail_bone]
            else:
                tail = tuple(float(x) for x in bone_point(bone_list[b.tail_bone].pos, b.tail_bone))
        grant = None
        if (b.grant_rotate or b.grant_move) and b.grant_parent >= 0:
            grant = {"parent": bnames[b.grant_parent], "rotate": bool(b.grant_rotate), "move": bool(b.grant_move),
                     "ratio": float(b.grant_ratio)}
        ik = None
        if b.ik is not None:
            chain = []
            for link in b.ik.links:
                lim = None
                if link[1]:
                    lim = [[math.degrees(x) for x in link[2]], [math.degrees(x) for x in link[3]]]
                chain.append({"bone": bnames[link[0]], "limit": lim})
            ik = {"target": bnames[b.ik.target], "chain": chain, "iterations": int(b.ik.loops),
                  "angle": math.degrees(b.ik.angle)}
        fixed = None if b.fixed_axis is None else tuple(float(x) for x in to_model(b.fixed_axis, 1.0)[0])
        lx = lz = None
        if b.local_x is not None and b.local_z is not None:
            lx = tuple(float(x) for x in to_model(b.local_x, 1.0)[0])
            lz = tuple(float(x) for x in to_model(b.local_z, 1.0)[0])
        bones.append(Bone(name=b.name, head=tuple(float(x) for x in h), tail=tail,
                          parent=bnames[b.parent] if b.parent >= 0 else "", name_en=b.name_en or "", tail_bone=tail_bone,
                          visible=bool(b.visible), movable=bool(b.movable), rotatable=bool(b.rotatable), deform=i in kept_w,
                          layer=int(b.layer), after_physics=bool(b.after_physics), fixed_axis=fixed, local_x=lx, local_z=lz,
                          grant=grant, ik=ik))

    # ---- materials
    mats = {}
    for w in wanted:
        m = model.materials[mnames.index(w)]
        mats[w] = TakeMaterial(
            name=m.name, name_en=m.name_en or "", diffuse=tuple(float(x) for x in m.diffuse),
            specular=tuple(float(x) for x in m.specular), shininess=float(m.shininess), ambient=tuple(float(x) for x in m.ambient),
            double_sided=bool(m.double_sided), drop_shadow=bool(m.ground_shadow), self_shadow_map=bool(m.self_shadow_map),
            self_shadow=bool(m.self_shadow), edge=bool(m.edge), edge_color=tuple(float(x) for x in m.edge_color),
            edge_size=float(m.edge_size), texture=_tex(model, root, m.texture),
            sphere=_tex(model, root, m.sphere_texture) if m.sphere_mode in (1, 2, 3) else None,
            sphere_mode=_SPHERE.get(m.sphere_mode, "none"),
            toon=None if m.toon_shared else _tex(model, root, m.toon), toon_shared=bool(m.toon_shared),
            toon_index=int(m.toon) if m.toon_shared else -1, memo=m.memo or "")

    # ---- morphs
    skip = set(skip_morphs)
    morphs_src = [mo for mo in model.morphs if mo.name not in skip]
    kept_mat_name = {mi: mnames[mi] for mi in kept_idx}
    vsel = {}                                            # vertex morph name -> (ids, offsets)
    out_morphs = []
    for mo in morphs_src:
        panel = PANELS.get(mo.panel, "other")
        if mo.kind == "vertex":
            if not mo.offsets:
                continue
            vi = np.array([o[0] for o in mo.offsets], int)
            sel = local[vi] >= 0
            if not sel.any():
                continue
            li = local[vi[sel]]
            d = to_model([o[1] for o, s in zip(mo.offsets, sel) if s], unit) * scale * sv[li][:, None]
            o = np.argsort(li, kind="stable")
            vsel[mo.name] = True
            out_morphs.append(TakeMorph(mo.name, mo.name_en or "", panel, "vertex", li[o], d[o]))
        elif mo.kind == "material":
            offs = []
            for o in mo.offsets:
                if o.index >= 0 and o.index not in kept_mat_name:
                    continue
                offs.append(MaterialOffset(None if o.index < 0 else kept_mat_name[o.index], o.op, tuple(o.diffuse),
                                           tuple(o.specular), float(o.shininess), tuple(o.ambient), tuple(o.edge_color),
                                           float(o.edge_size), tuple(o.texture), tuple(o.sphere), tuple(o.toon)))
            if offs:
                out_morphs.append(TakeMorph(mo.name, mo.name_en or "", panel, "material", material_offsets=offs))
    survivors = {m.name for m in out_morphs}
    for mo in morphs_src:
        if mo.kind != "group":
            continue
        members = [(model.morphs[i].name, float(r)) for i, r in mo.offsets if model.morphs[i].name in survivors]
        if members:
            out_morphs.append(TakeMorph(mo.name, mo.name_en or "", PANELS.get(mo.panel, "other"), "group", members=members))
    order_m = {mo.name: k for k, mo in enumerate(model.morphs)}
    out_morphs.sort(key=lambda m: order_m.get(m.name, 1e9))

    # ---- rigid bodies and joints on kept bones
    rbodies, rjoints = [], []
    if bodies:
        body_names = {}
        for k, rb in enumerate(model.bodies):
            if rb.bone < 0 or rb.bone not in kept:
                continue
            f = factor if (head and rb.bone in head_set) else 1.0
            pos = to_model([rb.pos], unit)[0] * scale
            if f != 1.0:
                pos = pivot + f * (pos - pivot)
            if rb.shape == 1:
                size = tuple(float(x) for x in to_model([rb.size], unit)[0] * scale * f)
            elif rb.shape == 0:
                size = (float(rb.size[0]) * unit * scale * f, 0.0, 0.0)
            else:
                size = (float(rb.size[0]) * unit * scale * f, float(rb.size[1]) * unit * scale * f, 0.0)
            rbodies.append(RigidBody(
                name=rb.name, bone=bnames[rb.bone], shape={0: "sphere", 1: "box", 2: "capsule"}[rb.shape], size=size,
                location=tuple(float(x) for x in pos), rotation=tuple(float(x) for x in _model_euler(rb.rot)),
                mode={0: "static", 1: "dynamic", 2: "dynamic_bone"}.get(rb.mode, "static"), group=int(rb.group),
                no_collide=tuple(g for g in range(16) if not (rb.mask >> g) & 1), mass=float(rb.mass),
                damping=(float(rb.linear_damping), float(rb.angular_damping)), friction=float(rb.friction),
                bounce=float(rb.restitution)))
            body_names[k] = rb.name
        for j in model.joints:
            if j.body_a in body_names and j.body_b in body_names:
                vec = lambda v, u=1.0: tuple(float(x) for x in to_model([v], u)[0])
                rjoints.append(Joint(
                    name=j.name, a=body_names[j.body_a], b=body_names[j.body_b],
                    location=tuple(float(x) for x in to_model([j.pos], unit)[0] * scale), rotation=tuple(float(x) for x in _model_euler(j.rot)),
                    move_lo=tuple(float(x) for x in to_model([j.move_lo], unit)[0] * scale),
                    move_hi=tuple(float(x) for x in to_model([j.move_hi], unit)[0] * scale),
                    rot_lo=tuple(-x for x in vec(j.rot_hi)), rot_hi=tuple(-x for x in vec(j.rot_lo)),
                    spring_move=vec(j.spring_move), spring_rot=vec(j.spring_rot)))

    # ---- display frames restricted to kept bones
    frames = {}
    for fr in model.frames:
        if fr.special:
            continue
        members = [bnames[i] for kind, i in fr.items if kind == "bone" and bnames[i] in kept_names]
        if members:
            frames[fr.name] = members

    info = dict(path=str(path), unit=unit, scale=scale, head=dict(head) if head else None, pivot=None if pivot is None else pivot,
                vertices=int(n), kept_bones=len(bones), dropped_bones=len(dropped_bone_names), dropped_bone_names=dropped_bone_names,
                kept_materials=wanted, dropped_materials=[m for m in mnames if m not in wanted],
                morphs=[m.name for m in out_morphs], skipped_morphs=sorted(skip), sdef_vertices=int(sum(
                    1 for vi in ids if model.vertices[int(vi)].kind == "SDEF")))
    return Take(verts=verts, normals=normals, uv=uv, edge=edge, src_ids=ids, faces=faces, weights=weights, materials=mats,
                bones=bones, morphs=out_morphs, bodies=rbodies, joints=rjoints, frames=frames, head_weight=head_w, info=info,
                root=root)


# --------------------------------------------------------------------------------------------------------- materials
def _image(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def _safe_stem(path):
    s = re.sub(r"[^0-9A-Za-z]+", "_", Path(str(path)).stem).strip("_").lower()
    return s or "tex"


def part_materials(tk, ctx, names=None, recolor=None, hooks=None, prefix=""):
    """Part Materials for the kept materials `names` (default all): the texture, sphere and toon images are read from the PMX's
    folder, edited by the `recolor` rules (pmx_recolor.recolor; rules and hooks are matched by the CONTENT of the source
    file, since a PMX often lists one image twice), passed through `hooks` ({source path: fn(img) -> img}), written with
    `ctx.save_png` (identical results are shared across calls and across parts) and referenced by file name.
    Returns {name: part.Material}."""
    from . import pmx_recolor as RC
    names = list(tk.materials) if names is None else list(names)
    reg = ctx.__dict__.setdefault("_pmx_files", {"digest": {}, "names": set()})

    def digest_of(p):
        f = _find_file(tk.root, p) if not Path(str(p)).is_absolute() else Path(p)
        return None if f is None or not Path(f).is_file() else hashlib.sha1(Path(f).read_bytes()).hexdigest()

    rules, hooked = {}, {}
    for r in recolor or ():
        d = digest_of(r["texture"])
        if d is None:
            raise ValueError(f"recolor rule texture {r['texture']!r} is not in {tk.root}")
        rules.setdefault(d, []).append(r)
    for k, v in (hooks or {}).items():
        d = digest_of(k)
        if d is not None:
            hooked[d] = v
    out, edited = {}, {}

    def put(path):
        if path is None:
            return "", False
        d0 = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        if d0 not in edited:                                             # one edit (and one hook call) per distinct source image
            img = _image(path)
            if d0 in rules:
                img = RC.recolor(img, rules[d0])
            if d0 in hooked:
                img = hooked[d0](img)
            edited[d0] = img
        img = edited[d0]
        dig = hashlib.sha1(img.tobytes() + str(img.shape).encode()).hexdigest()
        if dig in reg["digest"]:
            return reg["digest"][dig]
        stem = f"{prefix}{_safe_stem(path)}"
        k = 2
        while stem in reg["names"]:
            stem = f"{prefix}{_safe_stem(path)}_{k}"
            k += 1
        reg["names"].add(stem)
        res = (ctx.save_png(stem, img), bool((img[..., 3] < 255).any()))
        reg["digest"][dig] = res
        return res

    for n in names:
        m = tk.materials[n]
        tex, alpha = put(m.texture)
        sph, _ = put(m.sphere)
        toon, _ = put(m.toon)
        out[n] = Material(name=m.name, name_en=m.name_en, diffuse=m.diffuse, specular=m.specular, shininess=m.shininess,
                          ambient=m.ambient, texture=tex, toon=toon, sphere=sph, sphere_mode=m.sphere_mode if sph else "none",
                          double_sided=m.double_sided, edge=m.edge, edge_color=m.edge_color, edge_size=m.edge_size,
                          drop_shadow=m.drop_shadow, self_shadow_map=m.self_shadow_map, self_shadow=m.self_shadow,
                          alpha_blend=bool((alpha or m.diffuse[3] < 1.0) and tex), comment=m.memo)
    return out
