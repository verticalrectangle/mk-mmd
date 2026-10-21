"""bake: the built scene for the site's scene viewer and mk's own renderer (docs/design.md: The site, Rendering), written
beside the .blend as <name>.bake/ (`mk build --bake` bakes a saved scene again without building it):

  scene.glb   every mesh the clip shows as Blender's glTF exporter writes it, with its skin (its armature's bones are the
              joints) and its shape keys as morph targets, MMD materials as their base texture and alpha; no animation.
              Left out: a mesh hidden on every frame of the clip, and one baked frame by frame (`frames` below)
  bake.json   the clip (fps, frame0, frames, outputs, the cut per output, the shot table, the freezes), what bake.bin holds
              where (offsets in floats), and what the vector look needs to be drawn without Blender:
              objects    every object a look sorts (mkmmd.blender.styles.look_objects), in the scene's order: its pattern
                         record (collections, truthy properties), whether it is type (the text stage's), its materials as
                         a look numbers them, whether it casts Workbench's shadow, and `baked`: "glb", "by_frame" or null
                         (`why`), with the frame ranges [k0, k1) it renders in
              blocks     what Workbench draws that no look hides and the bake cannot hold (collection instances, volumes,
                         grease pencil, particle instances), with their frame ranges
              materials  per material of the objects: its viewport alpha and colour (linear) and the texture Workbench's
                         Texture colour draws for it: "glb" (the glb's base texture), "none" (the viewport colour) or
                         "other", with that image's alpha cutoff, premultiplied alpha and colour space and the node's
                         filtering
              by_frame   the objects the glb cannot carry (curves, meshes with modifiers besides the armature) as Blender
                         evaluates them: geometries [vertices offset, count, triangles offset, count], and the geometry
                         on each frame (-1: hidden); an unchanged geometry is not stored again
              rings      the shots stage's ring table, and per ring with a point centre the offset of that point on each
                         frame (NaN where the ring is not live or its expression failed: `errors`)
              workbench  the studio light of the scene's shading (four lights as stored: use, direction, diffuse colour,
                         wrap), its ambient and rotation, and whether back faces are culled
              screen     how many screen type objects the scene has (their layers are Blender's to draw)
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
                       the larger side of the picture); its sensor, fit and clip range are in bake.json
              by_frame per geometry its vertices in the world (position x y z, normal x y z) and its triangles (three
                       vertex indices and a material slot each)
              rings    per point ring, x y z on each frame

A vertex v of a mesh (its glTF positions, morph targets added) is at N v on a frame, N the mesh's node matrix; a skinned
one at sum_i w_i J_i N v, J_i its joints' matrices on that frame."""
import json
import os
import shutil
import struct
import time

import bpy
import numpy as np

from ...core import rings as RG
from .. import freeze as FRZ
from .. import styles as ST
from . import BuildError

AXES = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
FLEX_TYPES = ("CURVE", "SURFACE", "FONT", "META")           # evaluated into a mesh on every frame
FLEX_CAP = 20_000_000                                        # floats one object may add to bake.bin (80 MB)
IMAGE_NODES = ("TEX_IMAGE", "TEX_ENVIRONMENT")


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
    shots = sorted((e for e in _table(sc) if not e.get("plate")), key=lambda e: e["from"])
    return {o["name"]: [{"shot": e["name"], "from": round(ctx.time(e["from"]), 4), "to": round(ctx.time(e["to"]), 4),
                         "camera": e["cameras"].get(o["name"])} for e in shots if e["cameras"].get(o["name"])]
            for o in outputs}


def _table(sc):
    try:
        return json.loads(sc.get("mk_shots", "[]"))
    except ValueError:
        return []


# ================================================================================================= what a look draws
def _texture_node(nt, seen):
    """Blender's active texture of a node tree as Workbench finds it: the active node when it is a texture, else the
    first texture node, else one in a group (Python cannot see the active-texture flag of an inactive node)."""
    act = nt.nodes.active
    if act is not None and act.type.startswith("TEX_") and act.type != "TEX_COORD":
        return act
    for node in nt.nodes:
        if node.type.startswith("TEX_") and node.type != "TEX_COORD":
            return node
    for node in nt.nodes:
        if node.type == "GROUP" and node.node_tree is not None and node.node_tree.name not in seen:
            seen.add(node.node_tree.name)
            found = _texture_node(node.node_tree, seen)
            if found is not None:
                return found
    return None


def _workbench_texture(m):
    """The image texture node Workbench's Texture colour draws `m` with, or None (it draws the viewport colour)."""
    if not m.use_nodes or m.node_tree is None:
        return None
    node = _texture_node(m.node_tree, set())
    return node if node is not None and node.type in IMAGE_NODES and getattr(node, "image", None) is not None else None


def _glb_image(m):
    """The image the glTF exporter takes as `m`'s base texture: the one on the Principled BSDF's Base Color that the
    active output shows (after plain_materials), or None."""
    if not m.use_nodes or m.node_tree is None:
        return None
    out = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    links = out.inputs["Surface"].links if out is not None else ()
    bsdf = links[0].from_node if links else None
    if bsdf is None or bsdf.type != "BSDF_PRINCIPLED" or not bsdf.inputs["Base Color"].links:
        return None
    src = bsdf.inputs["Base Color"].links[0].from_node
    return src.image if src.type == "TEX_IMAGE" else None


def _material(m, node):
    """bake.json's record of a material (`node`: its Workbench texture node, found before plain_materials ran)."""
    out = {"alpha": round(float(m.diffuse_color[3]), 6), "color": [round(float(c), 6) for c in m.diffuse_color[:3]],
           "texture": "none"}
    if node is not None:
        img = node.image
        out.update(texture="glb" if _glb_image(m) == img else "other",
                   cutoff=img.alpha_mode not in ("NONE", "CHANNEL_PACKED"), premul=img.alpha_mode == "PREMUL",
                   srgb=img.colorspace_settings.name == "sRGB", closest=getattr(node, "interpolation", "Linear") == "Closest",
                   extension=getattr(node, "extension", "REPEAT"))
    return out


def _baked(o):
    """How the bake holds an object a look sorts: ("glb" | "by_frame" | None, why)."""
    if ST.is_type(o):
        return None, "type (the text stage's): drawn by Blender"
    if o.type == "MESH":
        for ps in o.particle_systems:
            if ps.settings.render_type in ("OBJECT", "COLLECTION", "PATH"):
                return None, f"particle system {ps.name!r}"
        if any(md.show_render and md.type not in ("ARMATURE", "PARTICLE_SYSTEM") for md in o.modifiers):
            return "by_frame", None
        return "glb", None
    if o.type in FLEX_TYPES:
        return "by_frame", None
    return None, f"a {o.type.lower()} object"


def _blocks(sc):
    """Objects no look sorts or hides that Workbench draws, and the bake cannot hold: (object, why)."""
    out = []
    for o in sc.objects:
        if not ST.renders(o):
            continue
        if o.type == "EMPTY" and o.instance_type == "COLLECTION" and o.instance_collection is not None:
            out.append((o, f"instances collection {o.instance_collection.name!r}"))
        elif o.instance_type in ("VERTS", "FACES") and o.children:
            out.append((o, "instances its children"))
        elif o.type in ("VOLUME", "GPENCIL", "GREASEPENCIL"):
            out.append((o, f"a {o.type.lower()} object"))
    return out


def _ranges(flags):
    """[[k0, k1), ...] where `flags` is true."""
    out, start = [], None
    for k, v in enumerate(flags):
        if v and start is None:
            start = k
        elif not v and start is not None:
            out.append([start, k])
            start = None
    if start is not None:
        out.append([start, len(flags)])
    return out


def _workbench(sc):
    sh = sc.display.shading
    sl = bpy.context.preferences.studio_lights.get(sh.studio_light)
    lights = [{"use": bool(L.use), "direction": [round(float(v), 6) for v in L.direction],
               "diffuse": [round(float(v), 6) for v in L.diffuse_color], "wrap": round(float(L.smooth), 6)}
              for L in (sl.solid_lights if sl is not None else [])]
    return {"studio": sh.studio_light, "lights": lights,
            "ambient": [round(float(v), 6) for v in sl.light_ambient] if sl is not None else [1.0, 1.0, 1.0],
            "rotate_z": round(float(sh.studiolight_rotate_z), 6), "cull": bool(sh.show_backface_culling)}


class Flex:
    """An object baked frame by frame: its evaluated mesh in the world on each frame it shows (see the docstring)."""

    def __init__(self, ob, n):
        self.ob = ob
        self.frames = [-1] * n
        self.geoms = []            # [vertex array index, count, triangle array index, count]
        self.verts, self.tris = [], []
        self.size = 0
        self.why = None

    def add(self, k, dg):
        if self.why is not None or self.ob.hide_render:
            return
        ev = self.ob.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            vn, tri = _mesh_arrays(ev, me)
        finally:
            ev.to_mesh_clear()
        if self.geoms and np.array_equal(vn, self.verts[self.geoms[-1][0]]):
            if np.array_equal(tri, self.tris[self.geoms[-1][2]]):
                self.frames[k] = len(self.geoms) - 1
                return
        ti = len(self.tris) - 1 if self.tris and np.array_equal(tri, self.tris[-1]) else None
        if ti is None:
            self.tris.append(tri)
            ti = len(self.tris) - 1
            self.size += tri.size
        self.verts.append(vn)
        self.size += vn.size
        self.geoms.append([len(self.verts) - 1, len(vn), ti, len(tri)])
        self.frames[k] = len(self.geoms) - 1
        if self.size > FLEX_CAP:
            self.why = f"too big to bake on every frame (over {FLEX_CAP * 4 // 1000000} MB)"
            self.verts, self.tris, self.geoms = [], [], []


def _mesh_arrays(ob, me):
    """An evaluated mesh in the world, glTF's axes: ((n, 6) float32 position and normal per vertex, (m, 4) float32
    triangles: three vertex indices and a material slot). Faces drawn flat or with split normals get vertices of their
    own (Blender's normals domain)."""
    me.calc_loop_triangles()
    nv, nt = len(me.vertices), len(me.loop_triangles)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    tv = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tv)
    mat = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get("material_index", mat)
    domain = me.normals_domain
    if domain == "POINT":
        nor = np.empty(nv * 3, np.float32)
        me.vertex_normals.foreach_get("vector", nor)
        pos, nor, tris = co, nor.reshape(-1, 3), tv.reshape(-1, 3)
    else:
        pos = co[tv]
        if domain == "FACE":
            fn = np.empty(len(me.polygons) * 3, np.float32)
            me.polygon_normals.foreach_get("vector", fn)
            pi = np.empty(nt, np.int32)
            me.loop_triangles.foreach_get("polygon_index", pi)
            nor = np.repeat(fn.reshape(-1, 3)[pi], 3, axis=0)
        else:
            cn = np.empty(len(me.loops) * 3, np.float32)
            me.corner_normals.foreach_get("vector", cn)
            tl = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("loops", tl)
            nor = cn.reshape(-1, 3)[tl]
        tris = np.arange(nt * 3, dtype=np.int32).reshape(-1, 3)
    M = np.array(ob.matrix_world)
    p = (pos @ M[:3, :3].T + M[:3, 3]) @ AXES[:3, :3].T
    n = nor @ np.linalg.inv(M[:3, :3]) @ AXES[:3, :3].T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    vn = np.concatenate([p, n], 1).astype(np.float32)
    return vn, np.concatenate([tris, mat[:, None]], 1).astype(np.float32)


# ================================================================================================= the bake
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
    looks = ST.look_objects(sc)
    how = {o.name: _baked(o) for o in looks}
    blocks = _blocks(sc)
    flex = [Flex(o, n) for o in looks if how[o.name][0] == "by_frame"]
    meshes = [o for o in sc.objects if o.type == "MESH" and not ST.is_type(o) and
              (o.name not in how or how[o.name][0] == "glb")]
    arms = {}
    for o in meshes:
        a = o.find_armature()
        if a is not None:
            arms[a.name] = a
    arms = [arms[k] for k in sorted(arms)]
    keyed = [o for o in meshes if o.data.shape_keys is not None and len(o.data.shape_keys.key_blocks) > 1]
    ring_tab = json.loads(sc.get("mk_rings") or "{}")
    rings = ring_tab.get("rings") or []
    point_rings = [r for r in rings if r["center"]["mode"] != "frame"]

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
    vis = {o.name: np.zeros(n, bool) for o in looks + [b for b, _ in blocks]}
    centres = {r["index"]: np.full((n, 3), np.nan, np.float32) for r in point_rings}
    ring_errors = {}
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
            for name in vis:
                vis[name][k] = not bpy.data.objects[name].hide_render
            if flex:
                dg = bpy.context.evaluated_depsgraph_get()
                for fx in flex:
                    fx.add(k, dg)
            if point_rings:
                live = {r["index"] for r in RG.live(rings, (f - ring_tab["frame0"]) / ring_tab["fps"])}
                for r in point_rings:
                    if r["index"] in live and r["index"] not in ring_errors:
                        try:
                            centres[r["index"]][k] = AXES[:3, :3] @ np.array(ST.ring_point(r, f))
                        except RuntimeError as e:
                            ring_errors[r["index"]] = str(e)
    finally:
        FRZ.release()

    keep = [o for o in meshes if shown[o.name].any()]
    if not keep and not any(fx.geoms for fx in flex):    # a scene built only so far (`--until scene`): nothing to look at
        shutil.rmtree(out_dir, ignore_errors=True)
        ctx.log("bake: no mesh shows on any frame of the clip, nothing baked")
        return {"out": None, "skipped": "no mesh shows on any frame of the clip"}
    part = out_dir + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    used_arms = sorted({o.find_armature().name for o in keep if o.find_armature() is not None})
    mat_names = []
    for o in looks:
        for m in ST.look_materials(o):
            if m not in mat_names:
                mat_names.append(m)
    wb_nodes = {m: _workbench_texture(bpy.data.materials[m]) for m in mat_names}   # before plain_materials relinks

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
    if keep:
        bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_apply=False,
                                  export_animations=False, export_skins=True, export_morph=True,
                                  export_morph_normal=False, export_cameras=False, export_lights=False, export_yup=True,
                                  export_current_frame=True)    # frame0, where nodes[...][0] is; not frame 0, where a
        spaces = vertex_spaces(glb, keep, nodes)                # root keyed by a walk or a vehicle stands elsewhere
    else:
        _empty_glb(glb)
        spaces = {}
    t_glb = time.time() - t_glb

    blob, size = [], 0
    index = {"armatures": [], "nodes": [], "shown": [], "morphs": [], "cameras": [], "by_frame": []}

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
        index["cameras"].append({"name": c, "sensor": float(cd.sensor_width), "sensor_height": float(cd.sensor_height),
                                 "fit": cd.sensor_fit, "clip": [float(cd.clip_start), float(cd.clip_end)],
                                 "offset": put(cam_data[c])})
    for fx in flex:
        if fx.why is None and fx.geoms:
            v_at = [put(v) for v in fx.verts]
            t_at = [put(t) for t in fx.tris]
            index["by_frame"].append({"name": fx.ob.name, "materials": [s.material.name if s.material else None
                                                                      for s in fx.ob.material_slots],
                                    "geoms": [[v_at[vi], nv, t_at[ti], nt] for vi, nv, ti, nt in fx.geoms],
                                    "frames": fx.frames})
    baked = {e["name"] for e in index["nodes"]} | {e["name"] for e in index["by_frame"]}
    flex_why = {fx.ob.name: fx.why for fx in flex if fx.why is not None}
    objects = []
    for o in looks:
        kind, why = how[o.name]
        ranges = _ranges(vis[o.name])
        if kind is not None and o.name not in baked:
            kind, why = None, flex_why.get(o.name) or ("never shows in the clip" if not ranges else "not exported")
        rec = ST.record(o)
        objects.append({"name": o.name, "type": o.type, "collections": list(rec.collections), "props": sorted(rec.props),
                        "text": ST.is_type(o), "materials": ST.look_materials(o), "shadow": bool(o.display.show_shadows),
                        "baked": kind, **({"why": why} if why else {}), "shows": ranges})
    ring_doc = {"table": ring_tab or None, "points": {}, "errors": {str(i): e for i, e in ring_errors.items()}}
    for i, c in centres.items():
        ring_doc["points"][str(i)] = put(c)
    np.concatenate(blob).astype("<f4").tofile(os.path.join(part, "bake.bin"))
    doc = {"version": 3, "fps": ctx.fps, "frame0": ctx.frame0, "frames": n, "duration": ctx.duration, "glb": "scene.glb",
           "outputs": outputs, "cut": cut, "shots": _table(sc), "freezes": wins, **index, "objects": objects,
           "blocks": [{"name": b.name, "why": why, "shows": _ranges(vis[b.name])} for b, why in blocks],
           "materials": {m: _material(bpy.data.materials[m], wb_nodes[m]) for m in mat_names},
           "rings": ring_doc, "workbench": _workbench(sc), "screen": sum(1 for o in sc.objects if "mk_screen" in o.keys()),
           "medium": json.loads(sc.get("mk_medium") or "[]")}
    with open(os.path.join(part, "bake.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(part, out_dir)
    report = {"out": out_dir, "frames": n, "meshes": len(index["nodes"]), "by_frame": len(index["by_frame"]),
              "armatures": len(index["armatures"]), "cameras": len(cams),
              "glb_mb": round(os.path.getsize(os.path.join(out_dir, "scene.glb")) / 1e6, 1),
              "bin_mb": round(size * 4 / 1e6, 1), "export_seconds": round(t_glb, 1), "seconds": round(time.time() - t0, 1)}
    if flex_why:
        report["not_by_frame"] = flex_why
    ctx.log("bake", report)
    return report


def _empty_glb(path):
    """A glb with an empty scene (every mesh the clip shows is baked frame by frame)."""
    doc = json.dumps({"asset": {"version": "2.0", "generator": "mk bake"}, "scenes": [{"nodes": []}], "scene": 0,
                      "nodes": [], "meshes": [], "accessors": [], "bufferViews": [], "buffers": []}).encode("utf-8")
    doc += b" " * (-len(doc) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(doc)) + struct.pack("<II", len(doc), 0x4E4F534A) + doc)


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
