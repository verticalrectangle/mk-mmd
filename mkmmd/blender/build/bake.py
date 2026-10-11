"""bake: the built scene for the site's scene viewer (docs/design.md: The site), written beside the .blend as
<name>.bake/ (`mk build --bake` bakes a saved scene again without building it):

  scene.glb   every mesh the clip shows (one hidden on every frame of the clip is left out) as Blender's glTF exporter
              writes it, with its skin (its armature's bones are the joints) and its shape keys as morph targets, MMD
              materials as their base texture and alpha; no animation
  bake.json   the clip (fps, frame0, frames, outputs, the cut per output) and what bake.bin holds where (offsets in
              floats)
  bake.bin    float32: the clip on each of its frames (in a freeze the world on the frame it holds and the cameras on
              their own keys, as mk render draws it), in glTF's axes (+Y up: Blender's (x, y, z) is glTF's (x, z, -y)):
              joints   per armature, frame and bone, the 3x4 matrix (12 floats, column by column) that takes a point of
                       the armature's rest pose, where it is in the world on that frame, to where the bone carries it
              nodes    per mesh, the 3x4 matrix that places its glTF vertices on each frame (one when it never moves):
                       its world matrix when the exporter wrote them in the mesh's own space ("vertices": "local"),
                       its motion since the export frame (frame0) when it wrote them where they were in the world
                       then ("world": skinned meshes)
              shown    per mesh whose visibility changes, 1 or 0 on each frame
              morphs   per mesh with shape keys, their values on each frame (glTF's morph target order)
              cameras  per shot camera, on each frame its 3x4 world matrix, lens (mm), shift x and shift y (fractions of
                       the larger side of the picture; the sensor is 36 mm on that side)

A vertex v of a mesh (its glTF positions, morph targets added) is at N v on a frame, N the mesh's node matrix; a skinned
one at sum_i w_i J_i N v, J_i its joints' matrices on that frame."""
import json
import os
import shutil
import struct
import time

import bpy
import numpy as np

from .. import freeze as FRZ
from . import BuildError

AXES = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def gl(m):
    """Blender matrices (..., 4, 4) in glTF's axes."""
    return AXES @ np.asarray(m, float) @ AXES.T


def affine(m):
    """(..., 4, 4) -> (..., 12): the top three rows, column by column (what a shader's mat4 needs, less its last row)."""
    m = np.asarray(m, float)
    return np.swapaxes(m[..., :3, :], -1, -2).reshape(*m.shape[:-2], 12)


def pose_matrices(arm, buf):
    """The armature's pose bones' matrices (armature space), (n, 4, 4). foreach_get gives each one column by column."""
    arm.pose.bones.foreach_get("matrix", buf)
    return buf.reshape(-1, 4, 4).transpose(0, 2, 1)


def plain_materials(objs):
    """Give every MMD material of `objs` (mmd_tools' shader group) a Principled BSDF the glTF exporter reads: its base
    texture (`mmd_base_tex`) as base colour, and as alpha when the material blends; without one, its diffuse colour."""
    seen = set()
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is None or m.name in seen or m.node_tree is None:
                continue
            seen.add(m.name)
            nt = m.node_tree
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if out is None or not out.inputs["Surface"].links or out.inputs["Surface"].links[0].from_node.type != "GROUP":
                continue
            group = out.inputs["Surface"].links[0].from_node
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            tex = nt.nodes.get("mmd_base_tex")
            if tex is not None and getattr(tex, "image", None) is not None:
                nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                if m.blend_method != "OPAQUE":
                    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            else:
                col = group.inputs.get("Diffuse Color")
                rgb = col.default_value[:3] if col is not None and hasattr(col, "default_value") else m.diffuse_color[:3]
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])


def _cut(sc, ctx, outputs):
    """Per output, the cut: [{shot, from, to (clip seconds), camera}] from the shots stage's table."""
    try:
        table = json.loads(sc.get("mk_shots", "[]"))
    except ValueError:
        table = []
    shots = sorted((e for e in table if not e.get("plate")), key=lambda e: e["from"])
    return {o["name"]: [{"shot": e["name"], "from": round(ctx.time(e["from"]), 4), "to": round(ctx.time(e["to"]), 4),
                         "camera": e["cameras"].get(o["name"])} for e in shots if e["cameras"].get(o["name"])]
            for o in outputs}


def bake(ctx, out_dir):
    sc = bpy.context.scene
    t0 = time.time()
    n = int(round(ctx.duration * ctx.fps))
    frames = [ctx.frame0 + k for k in range(n)]
    wins = FRZ.windows(sc)
    outputs = [{"name": o["name"], "size": list(o["size"])} for o in ctx.data.get("output", [])] or \
              [{"name": "main", "size": [sc.render.resolution_x, sc.render.resolution_y]}]
    cut = _cut(sc, ctx, outputs)
    cams = sorted({c["camera"] for shots in cut.values() for c in shots if c["camera"] in bpy.data.objects})
    meshes = [o for o in sc.objects if o.type == "MESH"]
    arms = {}
    for o in meshes:
        a = o.find_armature()
        if a is not None:
            arms[a.name] = a
    arms = [arms[k] for k in sorted(arms)]
    keyed = [o for o in meshes if o.data.shape_keys is not None and len(o.data.shape_keys.key_blocks) > 1]

    joints = {a.name: np.empty((n, len(a.pose.bones), 12), np.float32) for a in arms}
    bufs = {a.name: np.empty(16 * len(a.pose.bones), np.float32) for a in arms}
    rest = {a.name: np.linalg.inv(np.array([np.array(b.matrix_local) for b in a.data.bones])) for a in arms}
    order_ok = {a.name: [b.name for b in a.data.bones] == [p.name for p in a.pose.bones] for a in arms}
    if not all(order_ok.values()):
        raise BuildError(f"bake: an armature's pose bones are not in its bones' order ({[k for k, v in order_ok.items() if not v]})")
    nodes = {o.name: np.empty((n, 4, 4)) for o in meshes}
    shown = {o.name: np.empty(n, bool) for o in meshes}
    morphs = {o.name: np.empty((n, len(o.data.shape_keys.key_blocks) - 1), np.float32) for o in keyed}
    kbuf = {o.name: np.empty(len(o.data.shape_keys.key_blocks), np.float32) for o in keyed}
    cam_data = {c: np.empty((n, 15), np.float32) for c in cams}
    try:
        for k, f in enumerate(frames):
            FRZ.frame_set(sc, f, wins)
            for a in arms:
                W = np.array(a.matrix_world)
                T = W @ pose_matrices(a, bufs[a.name]) @ rest[a.name] @ np.linalg.inv(W)
                joints[a.name][k] = affine(gl(T))
            for o in meshes:
                nodes[o.name][k] = np.array(o.matrix_world)
                shown[o.name][k] = not o.hide_render
            for o in keyed:
                o.data.shape_keys.key_blocks.foreach_get("value", kbuf[o.name])
                morphs[o.name][k] = kbuf[o.name][1:]
            for c in cams:
                ob = bpy.data.objects[c]
                cam_data[c][k, :12] = affine(gl(np.array(ob.matrix_world)))
                cam_data[c][k, 12:] = (ob.data.lens, ob.data.shift_x, ob.data.shift_y)
    finally:
        FRZ.release()

    keep = [o for o in meshes if shown[o.name].any()]
    if not keep:                                   # a scene built only so far (`--until scene`): nothing to look at
        shutil.rmtree(out_dir, ignore_errors=True)
        ctx.log("bake: no mesh shows on any frame of the clip, nothing baked")
        return {"out": None, "skipped": "no mesh shows on any frame of the clip"}
    part = out_dir + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    used_arms = sorted({o.find_armature().name for o in keep if o.find_armature() is not None})

    sc.frame_set(ctx.frame0)                       # the export frame: each node's matrix on it is nodes[...][0]
    plain_materials(keep)
    bpy.ops.object.select_all(action="DESELECT")
    chain = {o.name: o for o in keep + [bpy.data.objects[a] for a in used_arms]}
    for o in list(chain.values()):                 # with their parents: the exporter leaves out a child of one it skips
        p = o.parent
        while p is not None:
            chain.setdefault(p.name, p)
            p = p.parent
    for o in chain.values():
        try:
            o.hide_set(False)
            o.select_set(True)
        except RuntimeError:                       # not in the view layer: the exporter cannot see it either
            ctx.log("bake: not in the view layer, left out:", o.name)
    t_glb = time.time()
    glb = os.path.join(part, "scene.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_apply=False,
                              export_animations=False, export_skins=True, export_morph=True, export_morph_normal=False,
                              export_cameras=False, export_lights=False, export_yup=True)
    t_glb = time.time() - t_glb
    spaces = vertex_spaces(glb, keep, nodes)

    blob, index, size = [], {"armatures": [], "nodes": [], "shown": [], "morphs": [], "cameras": []}, 0

    def put(arr):
        nonlocal size
        a = np.ascontiguousarray(arr, np.float32).ravel()
        blob.append(a)
        size += a.size
        return size - a.size

    for a in arms:
        if a.name in used_arms:
            index["armatures"].append({"name": a.name, "bones": [b.name for b in a.pose.bones], "offset": put(joints[a.name])})
    for o in keep:
        if o.name not in spaces:
            continue
        N = gl(nodes[o.name])
        if spaces[o.name] == "world":              # its vertices are where they were on the export frame: move them on
            N = N @ np.linalg.inv(N[0])
        m = affine(N)
        still = np.allclose(m, m[:1], atol=1e-7)
        index["nodes"].append({"name": o.name, "vertices": spaces[o.name], "frames": 1 if still else n,
                               "offset": put(m[:1] if still else m),
                               "armature": o.find_armature().name if o.find_armature() is not None else None})
        if not shown[o.name].all():
            index["shown"].append({"name": o.name, "offset": put(shown[o.name].astype(np.float32))})
        if o.name in morphs:
            index["morphs"].append({"name": o.name, "count": int(morphs[o.name].shape[1]), "offset": put(morphs[o.name])})
    for c in cams:
        cd = bpy.data.objects[c].data
        index["cameras"].append({"name": c, "sensor": float(cd.sensor_width), "fit": cd.sensor_fit, "offset": put(cam_data[c])})
    np.concatenate(blob).astype("<f4").tofile(os.path.join(part, "bake.bin"))
    doc = {"version": 1, "fps": ctx.fps, "frame0": ctx.frame0, "frames": n, "duration": ctx.duration, "glb": "scene.glb",
           "outputs": outputs, "cut": cut, "freezes": wins, **index}
    with open(os.path.join(part, "bake.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(part, out_dir)
    report = {"out": out_dir, "frames": n, "meshes": len(index["nodes"]), "armatures": len(index["armatures"]),
              "cameras": len(cams), "glb_mb": round(os.path.getsize(os.path.join(out_dir, "scene.glb")) / 1e6, 1),
              "bin_mb": round(size * 4 / 1e6, 1), "export_seconds": round(t_glb, 1), "seconds": round(time.time() - t0, 1)}
    ctx.log("bake", report)
    return report


def vertex_spaces(glb, objs, nodes):
    """Where the glTF exporter put each mesh's vertices: "local" (the mesh's own space: its node matrix places it) or
    "world" (where they were in the world on the export frame, as it writes skinned meshes), told by the POSITION bounds
    glTF keeps against the mesh's own bounds and its world bounds on that frame (nodes[name][0]). A mesh it did not
    export is left out; one it put elsewhere is a BuildError."""
    with open(glb, "rb") as fh:
        head = fh.read(20)
        doc = json.loads(fh.read(struct.unpack_from("<I", head, 12)[0]))
    wanted, out = {o.name: o for o in objs}, {}
    for node in doc.get("nodes", []):
        o = wanted.get(node.get("name"))
        if o is None or "mesh" not in node:
            continue
        acc = [doc["accessors"][p["attributes"]["POSITION"]] for p in doc["meshes"][node["mesh"]]["primitives"]]
        lo, hi = np.min([a["min"] for a in acc], 0), np.max([a["max"] for a in acc], 0)
        co = np.empty(3 * len(o.data.vertices))
        o.data.vertices.foreach_get("co", co)
        local = co.reshape(-1, 3) @ AXES[:3, :3].T
        W = gl(nodes[o.name][0])
        world = local @ W[:3, :3].T + W[:3, 3]
        tol = 1e-3 + 1e-4 * float(np.abs(hi - lo).max())
        fits = lambda p: np.allclose(p.min(0), lo, atol=tol) and np.allclose(p.max(0), hi, atol=tol)   # noqa: E731
        if fits(local):
            out[o.name] = "local"
        elif fits(world):
            out[o.name] = "world"
        else:
            raise BuildError(f"bake: the glTF exporter put {o.name}'s vertices neither in its own space nor in the world")
    return out


def run(ctx):
    blend = bpy.data.filepath
    if not blend:
        raise BuildError("bake: the scene is not saved (the bake goes beside the .blend)")
    return bake(ctx, os.path.splitext(blend)[0] + ".bake")
