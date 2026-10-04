"""pmxprop: an MMD accessory or prop model (PMX / PMD) as a prop (docs/design.md: PMX props).

The model is imported with mmd_tools (scale 0.08, rigid bodies included), measured, and turned into a plain prop under
the prop's root: static when the model has no moving parts (the armature, rigid body and joint objects and the mmd root
are removed, the meshes keep their own materials and shape keys), otherwise with its armature kept (named
`<prop>_arm`, parented to the root) so the project can pose the parts. The origin is moved to the bottom centre of the
bounds (`origin = "floor_center"`; `"center"` and `"keep"` also work), `scale` resizes the model first. The card comes
from mkmmd.core.propcard: size and bounds, a `look` point at the centre, a `rest` plane `top` when the top is flat,
colliders from the model's static rigid bodies (spheres and capsules ride on the prop root, boxes become hidden box
objects `<prop>_col<i>`) or one box over the bounds. The project's `card_extra` is merged over it by the props stage."""
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...core import propcard as PC
from ..library.mesh import box as make_box
from . import BuildError, link
from .cast import import_model, relink_textures

PHYSICS_TYPES = ("RIGID_BODY", "JOINT", "RIGID_GRP_OBJ", "JOINT_GRP_OBJ", "TEMPORARY", "TEMPORARY_GRP_OBJ")
EXTENSIONS = (".pmx", ".pmd")
KEYS = {"scale", "origin"}                     # [[prop]] keys of a PMX prop (besides name, card, at, yaw, parent, place...)


def mesh_arrays(o):
    """World-space vertices (n, 3) and triangles (m, 3) of a mesh object (rest pose, shape keys at their values)."""
    me = o.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    M = np.array(o.matrix_world, float)
    verts = co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    idx = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", idx)
    return verts, idx.reshape(-1, 3)


def bone_vertex_counts(meshes):
    """{vertex group name: vertices it moves most} over the meshes (mmd_tools' own groups `mmd_*` left out)."""
    counts = {}
    for o in meshes:
        names = {g.index: g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            best, w = None, 1e-4
            for g in v.groups:
                if g.weight > w and not names.get(g.group, "mmd_").startswith("mmd_"):
                    best, w = names[g.group], g.weight
            if best is not None:
                counts[best] = counts.get(best, 0) + 1
    return counts


def rigid_bodies(objs, shift):
    """PMX rigid bodies in the prop frame (after `shift`): {shape, mode, center, rot, half} from the body objects."""
    out = []
    for o in objs:
        if getattr(o, "mmd_type", "") != "RIGID_BODY":
            continue
        M = o.matrix_world
        _t, q, s = M.decompose()
        bb = np.array([list(v) for v in o.bound_box]) * np.array(s)
        lo, hi = bb.min(0), bb.max(0)
        centre = np.array(M @ Vector(((lo + hi) / 2.0 / np.array(s)).tolist())) + shift
        out.append({"shape": o.mmd_rigid.shape, "mode": int(o.mmd_rigid.type), "center": centre.tolist(),
                    "rot": np.array(q.to_matrix()), "half": ((hi - lo) / 2.0).tolist()})
    return out


def collider_specs(name, coll, root, shapes):
    """Hidden objects and card specs for collider shapes (frame = the prop's root)."""
    specs = []
    for i, s in enumerate(shapes):
        if s["kind"] == "box":
            o = make_box(f"{name}_col{i}", s["center"], [2 * h for h in s["half"]], coll, root, collider=True)
            o.rotation_euler = Matrix(np.array(s["rot"]).tolist()).to_euler()
            specs.append({"type": "box", "object": o.name, "rnd": 0.0, "tag": name})
        elif s["kind"] == "sphere":
            specs.append({"type": "sphere", "object": root.name, "c": [round(v, 5) for v in s["c"]],
                          "R": round(s["R"], 5), "tag": name})
        else:
            specs.append({"type": "capsule", "object": root.name, "a": [round(v, 5) for v in s["a"]],
                          "b": [round(v, 5) for v in s["b"]], "R": round(s["R"], 5), "tag": name})
    return specs


def resolve_path(ctx, ref):
    """The model file of a `pmx:PATH` reference (relative paths and ~ as everywhere in the project)."""
    path = ctx.path(ref)
    if os.path.isdir(path):
        found = sorted(f for f in os.listdir(path) if f.lower().endswith(EXTENSIONS))
        if len(found) != 1:
            raise BuildError(f"pmx prop {ref!r}: a folder needs exactly one {'/'.join(EXTENSIONS)} file, "
                             f"found {found or 'none'}")
        path = os.path.join(path, found[0])
    if not os.path.isfile(path) or not path.lower().endswith(EXTENSIONS):
        raise BuildError(f"pmx prop {ref!r}: {path} is not a {'/'.join(EXTENSIONS)} file")
    return path


def load(ctx, name, coll, root, path, spec):
    """Import `path` as the prop `name` under `root`; returns the automatic card (without the project's extras)."""
    before_collections = {c.name for c in bpy.data.collections}
    had_world = bpy.context.scene.rigidbody_world is not None
    scale = float(spec.get("scale", 1.0))
    origin = spec.get("origin", "floor_center")
    if origin not in PC.ORIGINS:
        raise BuildError(f"prop {name!r}: origin must be one of {PC.ORIGINS}, got {origin!r}")
    try:
        mmd_root, arm, meshes, new = import_model(path, "bullet")
    except (StopIteration, RuntimeError) as e:
        raise BuildError(f"prop {name!r}: mmd_tools could not import {path}: {e}")
    if not meshes:
        raise BuildError(f"prop {name!r}: {path} has no meshes")
    if scale != 1.0:
        mmd_root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    # ---- measure (world space = the model as imported, before the prop's own transform is set)
    tris = []
    for o in meshes:
        verts, idx = mesh_arrays(o)
        if len(idx):
            tris.append(verts[idx])
    if not tris:
        raise BuildError(f"prop {name!r}: {path} has no triangles")
    tris = np.concatenate(tris)
    lo, hi = PC.bounds(tris)
    shift = PC.origin_shift(lo, hi, origin)
    tris = tris + shift
    bodies = rigid_bodies(new, shift)
    counts = bone_vertex_counts(meshes)
    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    part_bones = PC.moving_parts(parents, counts)
    T = Matrix.Translation(Vector(shift.tolist()))
    parts = []
    for b in part_bones:
        bone = arm.data.bones[b]
        head = np.array(arm.matrix_world @ bone.head_local) + shift
        tail = np.array(arm.matrix_world @ bone.tail_local) + shift
        parts.append({"bone": b, "parent": parents[b], "head": [round(float(v), 4) for v in head],
                      "tail": [round(float(v), 4) for v in tail], "vertices": counts[b]})

    # ---- remove what a prop does not need, move the rest under the root
    for o in new:
        if getattr(o, "mmd_type", "") in PHYSICS_TYPES:
            bpy.data.objects.remove(o, do_unlink=True)
    if not had_world and bpy.context.scene.rigidbody_world is not None:
        bpy.ops.rigidbody.world_remove()
    for c in list(bpy.data.collections):
        if c.name not in before_collections and c.name.startswith("RigidBody") and not c.objects:
            bpy.data.collections.remove(c)
    top_objects = [arm] if parts else list(meshes)
    for o in top_objects:
        mw = T @ o.matrix_world
        o.parent = root
        o.matrix_parent_inverse.identity()
        o.matrix_world = mw
    if parts:
        arm.name = f"{name}_arm"
        keep = [arm, *meshes]
    else:
        for o in meshes:
            for md in [m for m in o.modifiers if m.type == "ARMATURE"]:
                o.modifiers.remove(md)
        bpy.data.objects.remove(arm, do_unlink=True)
        keep = list(meshes)
    for i, o in enumerate(meshes):
        o.name = f"{name}_mesh{i}"
    bpy.data.objects.remove(mmd_root, do_unlink=True)
    for o in keep:
        link(o, coll)
    for o in meshes:
        o.hide_render = False
    bpy.context.view_layer.update()

    # ---- textures, card, colliders
    fixed, missing = relink_textures(meshes, os.path.dirname(path))
    card, shapes = PC.make_card(name, path, tris, bodies, parts, arm.name if parts else None, origin=origin)
    card["colliders"] = collider_specs(name, coll, root, shapes)
    card["materials"] = sorted({s.material.name for o in meshes for s in o.material_slots if s.material})
    if missing:
        card["stats"]["textures_missing"] = missing
    if fixed:
        card["stats"]["textures_relinked"] = fixed
    card["stats"]["scale"] = scale
    ctx.log("pmx prop", name, f"{len(tris)} triangles, {len(shapes)} colliders ({card['stats']['colliders_from']}), "
            f"{len(parts)} moving parts, top face: {'rest plane' if 'rest' in card['use'] else 'none'}")
    return card
