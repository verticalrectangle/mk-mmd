"""inspect_model: import a PMX into an empty scene and describe it (docs/design.md: Model description)."""
import hashlib
import os

import bpy
import numpy as np
from mathutils import Matrix

from ..core import bonemap, families
from . import scene as S
from .runtime import op

SCHEMA = 1

# semantic expression name -> candidate morph names (PMX Japanese first, then common English names)
MORPHS = {
    "blink": ["まばたき", "瞬き", "blink"],
    "wink_l": ["ウィンク", "ウインク", "wink"],
    "wink_r": ["ウィンク右", "ウインク右", "wink_r", "wink right"],
    "wink2_l": ["ウィンク２", "ウィンク2", "ウインク２"],
    "wink2_r": ["ｳｨﾝｸ２右", "ウィンク２右", "ウィンク2右"],
    "smile_eyes": ["笑い", "smile"],
    "a": ["あ", "a"], "i": ["い", "i"], "u": ["う", "u"], "e": ["え", "e"], "o": ["お", "o"],
    "a2": ["あ２", "あ2"], "n": ["ん"],
    "cheerful": ["にこり", "にっこり", "cheerful"],
    "serious": ["真面目", "serious"],
    "troubled": ["困る", "troubled"],
    "angry": ["怒り", "angry"],
    "sad": ["悲しみ", "悲しい", "sad"],
    "surprised": ["びっくり", "surprised"],
    "brow_up": ["上", "眉上"], "brow_down": ["下", "眉下"],
    "jito": ["じと目", "ジト目"],
    "calm": ["なごみ"],
    "hau": ["はぅ"],
    "star_eyes": ["星目"],
    "heart_eyes": ["はぁと", "ハート目", "ハート"],
    "pupils_small": ["瞳小"],
    "mouth_smile": ["口角上げ", "にやり", "∧"],
    "mouth_down": ["口角下げ"],
    "mouth_wide": ["口横広げ"],
    "omega": ["ω", "ω□"],
    "tongue": ["ぺろっ", "てへぺろ"],
    "blush": ["照れ", "赤面", "頬染め"],
    "tears": ["涙"],
    "pale": ["青ざめ", "青ざめる"],
}

MEASURE_PAIRS = {
    "upper_arm": ("arm", "elbow"),
    "forearm": ("elbow", "wrist"),
    "thigh": ("leg", "knee"),
    "shin": ("knee", "ankle"),
}


def _clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.actions):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_pmx(path, scale=0.08, physics=True, morphs=True):
    types = {"MESH", "ARMATURE", "DISPLAY"}
    if physics:
        types.add("PHYSICS")
    if morphs:
        types.add("MORPHS")
    before = set(bpy.data.objects)
    bpy.ops.mmd_tools.import_model(filepath=path, types=types, scale=scale, log_level="ERROR")
    new = [o for o in bpy.data.objects if o not in before]
    root = next(o for o in new if getattr(o, "mmd_type", "") == "ROOT")
    arm = next(o for o in root.children_recursive if o.type == "ARMATURE")
    return root, arm, new


def is_helper(name):
    """Bones mmd_tools adds for additional-transform and IK plumbing; not part of the PMX."""
    return name.startswith(("_dummy_", "_shadow_"))


def body_geometry(o, inv_arm, bone_rest):
    """A rigid body's shape in its bone's REST frame (armature space rest matrix `bone_rest`): sphere c R, capsule
    a b R, box M half. Size comes from the body's bound box times its scale (the axis convention of mmd_tools'
    capsule meshes does not matter this way)."""
    t, q, s = (inv_arm @ o.matrix_world).decompose()
    frame = Matrix.Translation(t) @ q.to_matrix().to_4x4()
    L = np.array(bone_rest.inverted() @ frame)
    bb = np.array([list(v) for v in o.bound_box]) * np.array(s)
    lo, hi = bb.min(0), bb.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    R3, T = L[:3, :3], L[:3, 3]
    shape = o.mmd_rigid.shape
    if shape == "SPHERE":
        return {"kind": "sphere", "c": _v(R3 @ ctr + T), "R": round(float(half.max()), 6)}
    if shape == "CAPSULE":
        ax = int(np.argmax(half))
        rad = float(np.delete(half, ax).max())
        e = np.zeros(3)
        e[ax] = max(half[ax] - rad, 0.0)
        return {"kind": "capsule", "a": _v(R3 @ (ctr - e) + T), "b": _v(R3 @ (ctr + e) + T), "R": round(rad, 6)}
    M = L.copy()
    M[:3, 3] = R3 @ ctr + T
    return {"kind": "box", "M": [_v(row) for row in M], "half": _v(half)}


def body_radius(g):
    return round(g["R"] if g["kind"] in ("sphere", "capsule") else float(min(g["half"])), 6)


def segment_end(B, name, dyn):
    """Rest end of a chain bone's segment (armature space): the head of its simulated child; a leaf ends at its tail
    when its chain's bones are connected (or it is alone), else it continues its chain's last direction."""
    b = B[name]
    kids = sorted((c for c in b.children if c.name in dyn), key=lambda c: c.name)
    if kids:
        return kids[0].head_local
    chain = [b]
    while chain[-1].parent is not None and chain[-1].parent.name in dyn:
        chain.append(chain[-1].parent)
    connected = all((c.head_local - p.tail_local).length < 1e-3 for c, p in zip(chain[:-1], chain[1:]))
    if connected or len(chain) == 1:
        return b.tail_local
    return b.head_local + (b.head_local - chain[1].head_local)


def _v(x):
    return [round(float(c), 6) for c in x]


def describe(root, arm, path=None, scale=0.08):
    mw = arm.matrix_world
    B = arm.data.bones
    rig = {"schema": SCHEMA, "scale": scale}
    mr = getattr(root, "mmd_root", None)
    rig["source"] = {"path": path, "sha1": sha1(path) if path and os.path.exists(path) else None,
                     "name_j": getattr(mr, "name", "") if mr else "", "name_e": getattr(mr, "name_e", "") if mr else ""}
    comment = ""
    if mr is not None and getattr(mr, "comment_text", "") and mr.comment_text in bpy.data.texts:
        comment = bpy.data.texts[mr.comment_text].as_string()
    rig["comment"] = comment[:4000]
    names = {pb.name: S.name_j(pb) for pb in arm.pose.bones}
    smap = bonemap.build_map(names)
    rig["map"] = smap
    rig["missing"] = [s for s in bonemap.all_semantic() if s not in smap]
    rig["missing_required"] = [s for s in bonemap.REQUIRED if s not in smap]
    rig["kind"] = "character" if all(s in smap for s in ("head", "arm.L", "arm.R", "leg.L", "leg.R")) else "prop"
    rig["bones"] = {b.name: {"jp": names.get(b.name, ""), "parent": b.parent.name if b.parent else None,
                             "head": _v(b.head_local), "tail": _v(b.tail_local), "deform": b.use_deform,
                             "helper": is_helper(b.name)}
                    for b in B}
    rig["fingers"] = {}
    for side in ("L", "R"):
        for fing, sems in bonemap.FINGERS.items():
            got = [smap[f"{s}.{side}"] for s in sems if f"{s}.{side}" in smap]
            if got:
                rig["fingers"][f"{fing}.{side}"] = got
    # ---- physics: bodies (armature space; geometry in the bone's rest frame), chains
    bodies, dyn = [], {}
    inv = mw.inverted()
    rigid = [o for o in bpy.data.objects if getattr(o, "mmd_type", "") == "RIGID_BODY"]
    for o in rigid:
        r = o.mmd_rigid
        t, q, s = (inv @ o.matrix_world).decompose()
        entry = {"name": o.name, "bone": r.bone if r.bone in B else None, "shape": r.shape.lower(),
                 "size": [float(x) for x in r.size], "type": int(r.type), "group": int(r.collision_group_number),
                 "no_collide": [i for i, v in enumerate(r.collision_group_mask) if v], "loc": _v(t), "quat": _v(q)}
        if entry["bone"]:
            entry["geom"] = body_geometry(o, inv, B[r.bone].matrix_local)
        bodies.append(entry)
        if r.type in ("1", "2") and entry["bone"]:
            dyn[r.bone] = entry
    rig["bodies"] = bodies
    joints = [o for o in bpy.data.objects if getattr(o, "mmd_type", "") == "JOINT" and o.rigid_body_constraint]
    locked = {}
    for j in joints:
        c = j.rigid_body_constraint
        lim = [abs(getattr(c, f"limit_ang_{a}_{s}", 0.0)) for a in "xyz" for s in ("lower", "upper")]
        b2 = None
        for side in ("object2", "object1"):
            ob = getattr(c, side, None)
            if ob is not None and getattr(ob, "mmd_type", "") == "RIGID_BODY":
                b2 = ob.mmd_rigid.bone
                break
        if b2:
            locked[b2] = max(lim) < 1e-6
    chains = []
    roots = [b for b in dyn if not (B[b].parent and B[b].parent.name in dyn)]
    for r in sorted(roots):
        order, parent_idx, stack = [], [], [(B[r], -1)]
        while stack:
            b, pi = stack.pop()
            idx = len(order)
            order.append(b.name)
            parent_idx.append(pi)
            for c in sorted(b.children, key=lambda c: c.name, reverse=True):
                if c.name in dyn:
                    stack.append((c, idx))
        anc = B[r].parent
        while anc is not None and anc.name in dyn:
            anc = anc.parent
        fam = families.classify(r, *[names.get(n, "") for n in order[:3]], *order[:3])
        branching = len(set(parent_idx)) < len(parent_idx) and any(parent_idx.count(i) > 1 for i in set(parent_idx))
        radii = [body_radius(dyn[n]["geom"]) for n in order]
        heads = [B[n].head_local for n in order]
        span = sum((heads[i] - heads[p]).length for i, p in enumerate(parent_idx) if p >= 0)
        ends = [_v(segment_end(B, n, dyn)) for n in order]
        chains.append({"family": fam, "root": r, "anchor": anc.name if anc else None, "bones": order,
                       "parents": parent_idx, "ends": ends, "branching": bool(branching), "body_radius": radii,
                       "locked_joints": sum(1 for n in order if locked.get(n)), "length": round(span, 4)})
    rig["chains"] = chains
    # ---- morphs
    keys = []
    for o in S.model_meshes(arm) or [c for c in root.children_recursive if c.type == "MESH"]:
        sk = o.data.shape_keys
        if sk:
            keys += [kb.name for kb in sk.key_blocks[1:]]
    panel = {}
    if mr is not None:
        for coll in ("vertex_morphs", "bone_morphs", "material_morphs", "group_morphs", "uv_morphs"):
            for m in getattr(mr, coll, []):
                panel[m.name] = {"panel": getattr(m, "category", "OTHER").lower(), "kind": coll[:-7]}
    all_morphs = list(dict.fromkeys(keys + list(panel)))
    norm_index = {bonemap.norm(n).lower(): n for n in all_morphs}
    rig["morphs"] = {}
    for sem, cands in MORPHS.items():
        for c in cands:
            hit = norm_index.get(bonemap.norm(c).lower())
            if hit:
                rig["morphs"][sem] = hit
                break
    rig["morph_list"] = [{"name": n, **panel.get(n, {"panel": "other", "kind": "vertex"}), "shape_key": n in keys}
                         for n in all_morphs]
    # ---- measurements (rest pose, metres)
    pts = []
    for o in [c for c in root.children_recursive if c.type == "MESH" and getattr(c, "mmd_type", "") == "NONE"]:
        m = o.matrix_world
        pts += [m @ v.co for v in o.data.vertices]
    meas = {}
    if pts:
        zs = [p.z for p in pts]
        meas["top"] = round(max(zs), 4)
        meas["bottom"] = round(min(zs), 4)
        meas["width"] = round(max(p.x for p in pts) - min(p.x for p in pts), 4)
        meas["depth"] = round(max(p.y for p in pts) - min(p.y for p in pts), 4)

    def head(sem):
        n = smap.get(sem)
        return (mw @ B[n].head_local) if n else None

    for key, (a, b) in MEASURE_PAIRS.items():
        vals = []
        for side in ("L", "R"):
            pa, pb = head(f"{a}.{side}"), head(f"{b}.{side}")
            if pa is not None and pb is not None:
                vals.append((pa - pb).length)
        if vals:
            meas[key] = round(sum(vals) / len(vals), 4)
    for side in ("L", "R"):
        w, tip = head(f"wrist.{side}"), smap.get(f"middle_tip.{side}") or smap.get(f"middle3.{side}")
        if w is not None and tip:
            end = mw @ (B[tip].head_local if smap.get(f"middle_tip.{side}") else B[tip].tail_local)
            meas.setdefault("hand", round((end - w).length, 4))
    eyes = [head(f"eye.{s}") for s in ("L", "R")]
    eyes = [e for e in eyes if e is not None]
    if eyes:
        meas["eye_height"] = round(sum(e.z for e in eyes) / len(eyes), 4)
    for key, sem in (("hip_height", "leg"), ("knee_height", "knee"), ("ankle_height", "ankle"),
                     ("shoulder_height", "arm")):
        vals = [head(f"{sem}.{s}") for s in ("L", "R")]
        vals = [v for v in vals if v is not None]
        if vals:
            meas[key] = round(sum(v.z for v in vals) / len(vals), 4)
    if head("arm.L") is not None and head("arm.R") is not None:
        meas["shoulder_width"] = round((head("arm.L") - head("arm.R")).length, 4)
    if head("head") is not None:
        meas["head_z"] = round(head("head").z, 4)
    rig["measure"] = meas
    # ---- quirks
    quirks = []
    bone_names = set(B.keys())
    for o in [c for c in root.children_recursive if c.type == "MESH"]:
        odd = [g.name for g in o.vertex_groups if g.name not in bone_names]
        if odd:
            quirks.append(f"{o.name}: {len(odd)} vertex groups are not bones ({', '.join(odd[:4])}); skip them when "
                          f"reading skin weights")
    if rig["missing_required"] and rig["kind"] == "character":
        quirks.append(f"missing standard bones: {', '.join(rig['missing_required'])}")
    nameless = [b for b, j in names.items() if not j and not is_helper(b)]
    if nameless:
        quirks.append(f"{len(nameless)} bones have no PMX name ({', '.join(nameless[:4])})")
    lockc = sum(c["locked_joints"] for c in chains)
    if lockc:
        fams = sorted({c["family"] for c in chains if c["locked_joints"]})
        quirks.append(f"{lockc} physics joints have zero angular range (fully locked) in: {', '.join(fams)}; "
                      f"Bullet treats those chains as near-rigid sticks")
    if any(c["branching"] for c in chains):
        quirks.append("some physics chains branch: " + ", ".join(c["root"] for c in chains if c["branching"]))
    missing_img = [i.name for i in bpy.data.images if i.source == "FILE" and i.filepath and
                   not os.path.exists(bpy.path.abspath(i.filepath))]
    if missing_img:
        quirks.append(f"{len(missing_img)} texture files are missing ({', '.join(missing_img[:4])})")
    if not keys:
        quirks.append("no shape keys: facial expressions are bone or material morphs only")
    rig["quirks"] = quirks
    rig["stats"] = {"bones": len(B), "bodies": len(bodies), "dynamic_bodies": len(dyn), "joints": len(joints),
                    "chains": len(chains), "morphs": len(all_morphs), "shape_keys": len(keys),
                    "vertices": sum(len(o.data.vertices) for o in root.children_recursive if o.type == "MESH")}
    return rig


@op("inspect_model")
def inspect_model(args):
    path = os.path.abspath(os.path.expanduser(args["path"]))
    scale = float(args.get("scale", 0.08))
    _clear_scene()
    root, arm, _ = import_pmx(path, scale=scale, physics=True, morphs=True)
    return describe(root, arm, path, scale)
