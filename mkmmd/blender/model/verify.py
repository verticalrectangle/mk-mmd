"""Compare an imported PMX with what `mkmmd.model.assemble` meant to write (positions, weights, morph offsets, UVs,
normals, winding, materials, bones, rigid bodies, joints). The import must not clean the model (vertex order is the
file's). Returns numbers (worst errors) and a list of problems beyond the tolerances."""
import json
import os

import numpy as np
from mathutils import Euler, Matrix

from .. import scene as S

TOL = {"pos_mm": 0.1, "bone_mm": 0.1, "weight": 2e-3, "morph_mm": 0.1, "uv": 1e-4, "normal_dot": 0.999, "body_mm": 0.1,
       "body_deg": 0.05, "limit": 1e-3}


def load_expected(prefix):
    """(arrays, meta) written by `Assembled.save_expected(prefix)`."""
    arrays = dict(np.load(prefix + ".npz"))
    with open(prefix + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return arrays, meta


def _mesh_objects(root):
    return [o for o in root.children_recursive if o.type == "MESH" and getattr(o, "mmd_type", "") == "NONE"]


def verify(root, arm, exp, meta, tol=None):
    t = dict(TOL)
    t.update(tol or {})
    problems, out = [], {}
    mw = arm.matrix_world
    pmx_to_blender = {S.name_j(pb): pb.name for pb in arm.pose.bones}
    bones = arm.data.bones

    def bad(msg):
        problems.append(msg)

    # ---- bones
    heads, wrong_parent, missing = [], [], []
    for b in meta["bones"]:
        bn = pmx_to_blender.get(b["name"])
        if bn is None:
            missing.append(b["name"])
            continue
        bb = bones[bn]
        heads.append(np.linalg.norm(np.array(mw @ bb.head_local) - np.array(b["head"])))
        par = S.name_j(arm.pose.bones[bb.parent.name]) if bb.parent else ""
        if par != (b["parent"] or ""):
            wrong_parent.append(b["name"])
    out["bones_expected"] = len(meta["bones"])
    out["bones_missing"] = missing
    out["bone_head_err_mm"] = round(float(max(heads, default=0.0)) * 1000, 4)
    if missing:
        bad(f"bones missing after import: {missing[:6]}")
    if wrong_parent:
        bad(f"bones with a different parent: {wrong_parent[:6]}")
    if out["bone_head_err_mm"] > t["bone_mm"]:
        bad(f"bone heads off by {out['bone_head_err_mm']} mm")
    ik_issues = []
    for b in meta["bones"]:
        if not b.get("ik") or b["name"] not in pmx_to_blender:
            continue
        ikb = pmx_to_blender[b["name"]]
        hits = [c for pb in arm.pose.bones for c in pb.constraints if c.type == "IK" and c.subtarget == ikb]
        if not hits:
            ik_issues.append(f"{b['name']}: no IK constraint")
        elif hits[0].chain_count != len(b["ik"]["chain"]):
            ik_issues.append(f"{b['name']}: chain_count {hits[0].chain_count} != {len(b['ik']['chain'])}")
    out["ik_issues"] = ik_issues
    problems += ik_issues

    # ---- the mesh
    meshes = _mesh_objects(root)
    if len(meshes) != 1:
        bad(f"expected one mesh object, found {len(meshes)}")
        return {"numbers": out, "problems": problems}
    ob = meshes[0]
    me = ob.data
    n = len(me.vertices)
    out["vertices"] = n
    if n != len(exp["pos"]):
        bad(f"vertex count {n} != expected {len(exp['pos'])}")
        return {"numbers": out, "problems": problems}
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    world = np.array(ob.matrix_world)
    co = co @ world[:3, :3].T + world[:3, 3]
    err = np.linalg.norm(co - exp["pos"], axis=1)
    out["pos_err_mm"] = round(float(err.max()) * 1000, 4)
    if out["pos_err_mm"] > t["pos_mm"]:
        bad(f"vertex positions off by up to {out['pos_err_mm']} mm")
    # weights
    group_bone = {}
    for g in ob.vertex_groups:
        if g.name in bones:
            group_bone[g.index] = S.name_j(arm.pose.bones[g.name])
    exp_idx, exp_w = exp["bone_idx"], exp["bone_w"]
    names = meta["bone_names"]
    worst, bad_vertices = 0.0, 0
    for i, v in enumerate(me.vertices):
        got = {}
        for g in v.groups:
            nm = group_bone.get(g.group)
            if nm is not None and g.weight > 1e-6:
                got[nm] = got.get(nm, 0.0) + g.weight
        want = {}
        for j in range(4):
            if exp_w[i, j] > 0:
                nm = names[int(exp_idx[i, j])]
                want[nm] = want.get(nm, 0.0) + float(exp_w[i, j])
        d = 0.0
        for k in set(got) | set(want):
            d = max(d, abs(got.get(k, 0.0) - want.get(k, 0.0)))
        if d > t["weight"]:
            bad_vertices += 1
        worst = max(worst, d)
    out["weight_err"] = round(worst, 6)
    out["weight_bad_vertices"] = bad_vertices
    if bad_vertices:
        bad(f"{bad_vertices} vertices have weights off by more than {t['weight']} (worst {worst:.4f})")
    # loops: uv, normals
    nl = len(me.loops)
    lv = np.empty(nl, np.int64)
    me.loops.foreach_get("vertex_index", lv)
    if me.uv_layers:
        uvl = me.uv_layers[0]
        uvv = np.empty(nl * 2)
        uvl.data.foreach_get("uv", uvv)
        uvv = uvv.reshape(nl, 2)
        uerr = np.abs(uvv - exp["uv"][lv]).max()
        out["uv_err"] = round(float(uerr), 6)
        if uerr > t["uv"]:
            bad(f"UVs off by {uerr:.5f}")
    else:
        bad("mesh has no UV layer")
    try:
        cn = np.empty(nl * 3)
        me.corner_normals.foreach_get("vector", cn)
        cn = cn.reshape(nl, 3)
        dots = np.einsum("ij,ij->i", cn, exp["normal"][lv])
        area = np.empty(len(me.polygons))
        me.polygons.foreach_get("area", area)
        live = np.repeat(area > 1e-12, 3)[:nl] if nl == 3 * len(area) else np.ones(nl, bool)
        out["degenerate_faces"] = int((area <= 1e-12).sum())      # Blender has no normal for a zero-area face
        dots = dots[live]
        out["normal_min_dot"] = round(float(dots.min()), 5) if len(dots) else 1.0
        out["normal_bad_corners"] = int((dots < t["normal_dot"]).sum())
        if out["normal_bad_corners"]:
            bad(f"{out['normal_bad_corners']} corner normals deviate (min dot {dots.min():.4f})")
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        out["normal_check"] = f"skipped: {type(e).__name__}"
    # winding: Blender's polygon normals against the assembled triangles' own (CCW from outside) normals
    npoly = len(me.polygons)
    pn = np.empty(npoly * 3)
    me.polygons.foreach_get("normal", pn)
    pn = pn.reshape(npoly, 3)
    tri = exp["tris"]
    P = exp["pos"][tri]
    want_n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    solid = np.linalg.norm(want_n, axis=1) > 1e-12                  # skip degenerate triangles
    flipped = int(((np.einsum("ij,ij->i", pn, want_n) < 0) & solid).sum()) if len(tri) == npoly else npoly
    out["faces"] = npoly
    out["faces_flipped"] = flipped
    if flipped:
        bad(f"{flipped} of {npoly} faces wind the wrong way after import")
    mi = np.empty(npoly, np.int64)
    me.polygons.foreach_get("material_index", mi)
    want_counts = np.bincount(exp["tri_mat"], minlength=len(meta["materials"]))
    got_counts = np.bincount(mi, minlength=len(meta["materials"]))
    out["material_faces_ok"] = bool((want_counts == got_counts[:len(want_counts)]).all())
    if not out["material_faces_ok"]:
        bad("faces per material differ after import")
    out["materials"] = len(me.materials)
    # ---- morphs
    sk = me.shape_keys
    basis = None
    morph_err, morph_missing = 0.0, []
    if sk:
        basis = np.empty(n * 3)
        sk.key_blocks[0].data.foreach_get("co", basis)
        basis = basis.reshape(n, 3)
    for k, name in enumerate(meta["morphs"]):
        ids, vec = exp[f"morph{k}_ids"], exp[f"morph{k}_vec"]
        if not sk or name not in sk.key_blocks:
            morph_missing.append(name)
            continue
        cur = np.empty(n * 3)
        sk.key_blocks[name].data.foreach_get("co", cur)
        off = cur.reshape(n, 3) - basis
        want = np.zeros((n, 3))
        want[ids] = vec
        morph_err = max(morph_err, float(np.abs(off - want).max()))
    out["morphs_expected"] = len(meta["morphs"])
    out["morph_err_mm"] = round(morph_err * 1000, 4)
    if morph_missing:
        bad(f"morphs missing as shape keys: {morph_missing[:6]}")
    if out["morph_err_mm"] > t["morph_mm"]:
        bad(f"morph offsets off by {out['morph_err_mm']} mm")
    return {"numbers": out, "problems": problems}


def _swap(v):
    return [v[0], v[2], v[1]]


def _close(a, b, tol=1e-5):
    return bool(np.allclose(np.asarray(list(a), float), np.asarray(list(b), float), atol=tol))


def verify_metadata(root, arm, meta, tol=None):
    """Bone flags, grants, axes, materials, morph panels and names, display frames of the imported model."""
    import bpy
    problems, out = [], {}
    pb_by_j = {S.name_j(pb): pb for pb in arm.pose.bones}
    bone_index = {n: i for i, n in enumerate(meta["bone_names"])}
    bad_flags, bad_grant, bad_axes, bad_names = [], [], [], []
    for b in meta["bones"]:
        pb = pb_by_j.get(b["name"])
        if pb is None:
            continue
        mb, bone = pb.mmd_bone, arm.data.bones[pb.name]
        g = b["grant"]
        if g:
            if not (mb.has_additional_rotation == bool(g.get("rotate", True)) and
                    mb.has_additional_location == bool(g.get("move", False)) and
                    abs(mb.additional_transform_influence - float(g.get("ratio", 1.0))) < 1e-5 and
                    mb.additional_transform_bone_id == bone_index[g["parent"]]):
                bad_grant.append(b["name"])
        elif mb.has_additional_rotation or mb.has_additional_location:
            bad_grant.append(b["name"])
        if b["fixed_axis"] is not None:
            ax = np.asarray(b["fixed_axis"], float)
            ax = ax / (np.linalg.norm(ax) or 1.0)
            if not (mb.enabled_fixed_axis and _close(mb.fixed_axis, _swap(ax))):
                bad_axes.append(b["name"])
        elif mb.enabled_fixed_axis:
            bad_axes.append(b["name"])
        if b["local_x"] is not None and b["local_z"] is not None:
            if not (mb.enabled_local_axes and _close(mb.local_axis_x, _swap(b["local_x"])) and
                    _close(mb.local_axis_z, _swap(b["local_z"]))):
                bad_axes.append(b["name"])
        elif mb.enabled_local_axes:
            bad_axes.append(b["name"])
        if (bone.hide == b["visible"] or mb.transform_order != b["layer"] or
                mb.transform_after_dynamics != b["after_physics"] or
                (not b["movable"]) != all(pb.lock_location) or (not b["rotatable"]) != all(pb.lock_rotation)):
            bad_flags.append(b["name"])
        if mb.name_e != (b["name_en"] or ""):
            bad_names.append(b["name"])
    out["bone_flag_issues"], out["bone_grant_issues"] = bad_flags, bad_grant
    out["bone_axis_issues"], out["bone_name_en_issues"] = bad_axes, bad_names
    for label, lst in (("bone flags (visible/layer/after physics/locks)", bad_flags), ("grants", bad_grant),
                       ("fixed or local axes", bad_axes), ("English names", bad_names)):
        if lst:
            problems.append(f"{label} differ after import: {lst[:5]}")
    # materials
    by_j = {m.mmd_material.name_j: m for m in bpy.data.materials if getattr(m, "mmd_material", None)}
    mat_bad = []
    for m in meta["material_data"]:
        bm = by_j.get(m["name"])
        if bm is None:
            mat_bad.append(f"{m['name']}: missing")
            continue
        mm = bm.mmd_material
        ok = (_close(mm.diffuse_color, m["diffuse"][:3]) and abs(mm.alpha - m["diffuse"][3]) < 1e-5 and
              _close(mm.specular_color, m["specular"]) and _close(mm.ambient_color, m["ambient"]) and
              abs(mm.shininess - m["shininess"]) < 1e-4 and bool(mm.is_double_sided) == m["double_sided"] and
              bool(mm.enabled_toon_edge) == m["edge"] and _close(mm.edge_color, m["edge_color"]) and
              abs(mm.edge_weight - m["edge_size"]) < 1e-5 and int(mm.sphere_texture_type) == m["sphere_type"])
        if not ok:
            mat_bad.append(f"{m['name']}: settings differ")
        for key, rel in (("texture", mm.texture_rel_path), ("toon", mm.toon_texture_rel_path),
                         ("sphere", mm.sphere_texture_rel_path)):
            if m[key] and not str(rel).replace("\\", "/").endswith(m[key]):
                mat_bad.append(f"{m['name']}: {key} {rel!r} != {m[key]!r}")
    out["material_issues"] = mat_bad
    problems += [f"material {s}" for s in mat_bad[:6]]
    # morph panels and names
    cat = {"eye": "EYE", "brow": "EYEBROW", "mouth": "MOUTH", "other": "OTHER"}
    have = {mm.name: mm for mm in root.mmd_root.vertex_morphs}
    morph_bad = []
    for m in meta["morph_data"]:
        mm = have.get(m["name"])
        if mm is None:
            morph_bad.append(f"{m['name']}: not registered as a vertex morph")
        elif mm.category != cat[m["panel"]] or mm.name_e != (m["name_en"] or ""):
            morph_bad.append(f"{m['name']}: panel/name differ ({mm.category}, {mm.name_e!r})")
    out["morph_issues"] = morph_bad
    problems += morph_bad[:6]
    # display frames
    frames = root.mmd_root.display_item_frames
    want = meta["frames"]
    frame_bad = []
    if [f.name for f in frames] != [f["name"] for f in want]:
        frame_bad.append(f"frame names {[f.name for f in frames][:6]} != {[f['name'] for f in want][:6]}")
    else:
        for f, w in zip(frames, want):
            if len(f.data) != len(w["items"]) or bool(f.is_special) != w["special"]:
                frame_bad.append(f"{w['name']}: {len(f.data)} items, expected {len(w['items'])}")
    out["frame_issues"] = frame_bad
    problems += [f"display frame {s}" for s in frame_bad[:6]]
    return {"numbers": out, "problems": problems}


def verify_physics(root, arm, meta, tol=None):
    """Rigid bodies and joints (needs the model imported with PHYSICS)."""
    t = dict(TOL)
    t.update(tol or {})
    problems, out = [], {}
    rigid = {o.mmd_rigid.name_j or o.name: o for o in bpy_objects(root) if getattr(o, "mmd_type", "") == "RIGID_BODY"}
    joints = {o.mmd_joint.name_j or o.name: o for o in bpy_objects(root) if getattr(o, "mmd_type", "") == "JOINT"}
    out["bodies_expected"], out["bodies"] = len(meta["bodies"]), len(rigid)
    out["joints_expected"], out["joints"] = len(meta["joints"]), len(joints)
    pos_err, rot_err, size_err, mode_bad, bone_bad, group_bad = 0.0, 0.0, 0.0, [], [], []
    pmx_to_blender = {S.name_j(pb): pb.name for pb in arm.pose.bones}
    for b in meta["bodies"]:
        o = rigid.get(b["name"])
        if o is None:
            problems.append(f"rigid body {b['name']!r} missing")
            continue
        M = o.matrix_world
        pos_err = max(pos_err, (np.array(M.translation) - np.array(b["location"])).__abs__().max())
        want = Euler(b["rotation"], "XYZ").to_matrix()
        diff = (M.to_3x3().normalized().inverted() @ want).to_quaternion().angle
        rot_err = max(rot_err, np.degrees(diff))
        size = np.array(list(o.mmd_rigid.size))
        want_size = np.array(b["size"], float)
        if b["shape"] == "sphere":
            size_err = max(size_err, abs(size[0] - want_size[0]))
        elif b["shape"] == "capsule":
            size_err = max(size_err, abs(size[0] - want_size[0]), abs(size[1] - want_size[1]))
        else:
            size_err = max(size_err, np.abs(size - want_size).max())
        if int(o.mmd_rigid.type) != {"static": 0, "dynamic": 1, "dynamic_bone": 2}[b["mode"]]:
            mode_bad.append(b["name"])
        if b["bone"] and o.mmd_rigid.bone != pmx_to_blender.get(b["bone"]):
            bone_bad.append(b["name"])
        mask_now = [i for i, v in enumerate(o.mmd_rigid.collision_group_mask) if v]
        if int(o.mmd_rigid.collision_group_number) != b["group"] or mask_now != sorted(set(b["no_collide"])):
            group_bad.append(b["name"])
    out["body_pos_err_mm"] = round(float(pos_err) * 1000, 4)
    out["body_rot_err_deg"] = round(float(rot_err), 4)
    out["body_size_err_mm"] = round(float(size_err) * 1000, 4)
    if out["body_pos_err_mm"] > t["body_mm"]:
        problems.append(f"rigid body positions off by {out['body_pos_err_mm']} mm")
    if out["body_rot_err_deg"] > t["body_deg"]:
        problems.append(f"rigid body rotations off by {out['body_rot_err_deg']} deg")
    if out["body_size_err_mm"] > t["body_mm"]:
        problems.append(f"rigid body sizes off by {out['body_size_err_mm']} mm")
    if mode_bad:
        problems.append(f"rigid body mode differs: {mode_bad[:5]}")
    if bone_bad:
        problems.append(f"rigid body bone differs: {bone_bad[:5]}")
    if group_bad:
        problems.append(f"rigid body collision group/mask differ: {group_bad[:5]}")
    jp, jr, jl, jbad, spring_bad = 0.0, 0.0, 0.0, [], []
    for j in meta["joints"]:
        o = joints.get(j["name"])
        if o is None:
            problems.append(f"joint {j['name']!r} missing")
            continue
        M = o.matrix_world
        jp = max(jp, np.abs(np.array(M.translation) - np.array(j["location"])).max())
        want = Euler(j["rotation"], "XYZ").to_matrix()
        jr = max(jr, np.degrees((M.to_3x3().normalized().inverted() @ want).to_quaternion().angle))
        c = o.rigid_body_constraint
        if c is None:
            jbad.append(j["name"])
            continue
        if not (_close(o.mmd_joint.spring_linear, j["spring_move"]) and _close(o.mmd_joint.spring_angular, j["spring_rot"])):
            spring_bad.append(j["name"])
        for ax, i in (("x", 0), ("y", 1), ("z", 2)):
            jl = max(jl, abs(getattr(c, f"limit_lin_{ax}_lower") - j["move_lo"][i]),
                     abs(getattr(c, f"limit_lin_{ax}_upper") - j["move_hi"][i]),
                     abs(getattr(c, f"limit_ang_{ax}_lower") - j["rot_lo"][i]),
                     abs(getattr(c, f"limit_ang_{ax}_upper") - j["rot_hi"][i]))
    out["joint_pos_err_mm"] = round(float(jp) * 1000, 4)
    out["joint_rot_err_deg"] = round(float(jr), 4)
    out["joint_limit_err"] = round(float(jl), 6)
    if out["joint_pos_err_mm"] > t["body_mm"]:
        problems.append(f"joint positions off by {out['joint_pos_err_mm']} mm")
    if out["joint_rot_err_deg"] > t["body_deg"]:
        problems.append(f"joint rotations off by {out['joint_rot_err_deg']} deg")
    if out["joint_limit_err"] > t["limit"]:
        problems.append(f"joint limits off by {out['joint_limit_err']}")
    if jbad:
        problems.append(f"joints without a constraint: {jbad[:5]}")
    if spring_bad:
        problems.append(f"joint springs differ: {spring_bad[:5]}")
    return {"numbers": out, "problems": problems}


def bpy_objects(root):
    import bpy
    return list(bpy.data.objects)
