"""cast: import every [[cast]] model with mmd_tools and place it.

[[cast]] keys: name, asset (registry slug) or pmx (path), rig (path; default the registry's), armature (name to give
the armature; default "<Name>_arm"), at [x, y, z], yaw (deg; 0 = facing -Y), parent (object the model rides on),
physics: "mk" (default: no Bullet; chains come from rig.json and the sim stage solves them), "bullet" (keep the
author's rigid bodies), "none" (no secondary motion: no Bullet, and the sim stage leaves the member out),
materials: {"<PMX material name>" = {color | tint | hide}} (`paint`: a colour of the model changed in the scene, the
model file untouched).

A model with group, bone, material or UV morphs (vowels made of a mouth shape and a tongue bone, a blush that is a
material) has its morph sliders bound (`bind_morphs`), so every morph is keyed the same way, on the root's
`.placeholder` (mkmmd.blender.scene.morph_holders)."""
import json
import math
import os
import re
import sys

import bpy
from mathutils import Euler, Vector

from ... import cache as CA
from ...core import cast as CN
from ...core import palette as PAL
from . import BuildError, Member, collection, link

SKIP_TYPES = ("RIGID_BODY", "JOINT", "TEMPORARY", "RIGID_GRP_OBJ", "JOINT_GRP_OBJ", "TEMPORARY_GRP_OBJ")


def _registry(ctx):
    path = os.path.join(os.path.expanduser(ctx.assets or "~/mk-assets"), "registry.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {e["slug"]: e for e in json.load(fh)}


def resolve(ctx, spec):
    """(pmx path, rig dict or None) for a cast entry."""
    reg = None
    pmx, rig_path = spec.get("pmx"), spec.get("rig")
    if spec.get("asset"):
        reg = _registry(ctx)
        e = reg.get(spec["asset"])
        if e is None:
            raise BuildError(f"cast {spec.get('name')!r}: asset {spec['asset']!r} is not in the registry")
        pmx = pmx or e["path"]
        if not rig_path and e.get("rig"):
            rig_path = e["rig"] if os.path.isabs(e["rig"]) else os.path.join(os.path.expanduser(ctx.assets), e["rig"])
    if not pmx:
        raise BuildError(f"cast {spec.get('name')!r}: give asset or pmx")
    pmx = ctx.path(pmx)
    rig = None
    if rig_path:
        with open(ctx.path(rig_path), encoding="utf-8") as fh:
            rig = json.load(fh)
    return pmx, rig


def import_model(path, physics, cache=None):
    """Import a PMX with mmd_tools (no Bullet unless physics is "bullet"): (root, armature, meshes, every new object).
    With a cache folder the import is kept there as a .blend keyed by the model file, the import's settings, mmd_tools'
    version and this function's code (`_import_key`), and a later build appends it instead of importing again."""
    types = {"MESH", "ARMATURE", "DISPLAY", "MORPHS"}
    if physics == "bullet":
        types.add("PHYSICS")
    blend = os.path.join(cache, "cast", _import_key(path, types) + ".blend") if cache else None
    if blend is not None and os.path.exists(blend):
        new = _append(blend)
    else:
        before = set(bpy.data.objects)
        bpy.ops.mmd_tools.import_model(filepath=path, types=types, scale=0.08, log_level="ERROR")
        new = [o for o in bpy.data.objects if o not in before]
        if blend is not None:
            os.makedirs(os.path.dirname(blend), exist_ok=True)
            tmp = f"{blend}.{os.getpid()}.tmp"
            bpy.data.libraries.write(tmp, set(new), compress=True)
            os.replace(tmp, blend)                     # whole or not at all for a build that reads it meanwhile
    root = next(o for o in new if getattr(o, "mmd_type", "") == "ROOT")
    arm = next(o for o in root.children_recursive if o.type == "ARMATURE")
    meshes = [o for o in root.children_recursive if o.type == "MESH" and getattr(o, "mmd_type", "") == "NONE"]
    for o in new:
        if getattr(o, "mmd_type", "") in SKIP_TYPES:
            o.hide_render = True
    return root, arm, meshes, new


def _import_key(path, types):
    """What an import makes depends on: the model file, the import's settings, mmd_tools' version and the code that
    asks for it (mkmmd.cache.code)."""
    mod = next((m for name, m in sys.modules.items() if name.split(".")[-1] == "mmd_tools" and getattr(m, "__file__", None)),
               None)
    manifest = os.path.join(os.path.dirname(mod.__file__), "blender_manifest.toml") if mod is not None else None
    version = CA.Path(manifest) if manifest and os.path.exists(manifest) else getattr(mod, "bl_info", {}).get("version")
    return CA.key("cast import", CA.Path(path), sorted(types), 0.08, str(version), bpy.app.version_string,
                  CA.code(sys.modules[__name__], "import_model", "_append"))


def _append(blend):
    """The objects of a kept import, appended and linked to the scene; the node groups and images it shares with a model
    imported before it are the ones already there, as mmd_tools' own import reuses them, and its materials' mmd ids
    follow the ones already taken, as mmd_tools numbers a new model's."""
    groups, images, mats = set(bpy.data.node_groups), set(bpy.data.images), set(bpy.data.materials)
    taken = max([m.mmd_material.material_id for m in mats], default=-1)
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = list(src.objects)
    new = [o for o in dst.objects if o is not None]
    for o in new:
        bpy.context.scene.collection.objects.link(o)
    for coll, before, same in ((bpy.data.node_groups, groups, lambda a, b: a.bl_idname == b.bl_idname),
                               (bpy.data.images, images, lambda a, b: a.filepath == b.filepath)):
        for idb in [x for x in coll if x not in before]:
            m = re.fullmatch(r"(.*)\.\d{3}", idb.name)
            kept = coll.get(m.group(1)) if m else None
            if kept is not None and kept in before and same(idb, kept):
                idb.user_remap(kept)
                coll.remove(idb)
    added = sorted((m for m in bpy.data.materials if m not in mats and m.mmd_material.material_id >= 0),
                   key=lambda m: m.mmd_material.material_id)
    for i, m in enumerate(added):
        m.mmd_material.material_id = taken + 1 + i
    return new


def bind_morphs(root, coll):
    """Bind the model's morph sliders when it has morphs that are not plain shape keys (group, bone, material, UV):
    mmd_tools makes every morph a shape key of a `.placeholder` under the root, whose drivers sum each morph into the
    meshes' shape keys, the bone morphs' constraints and the material morphs' nodes. Returns how many such morphs the
    model has (0: left unbound, its shape keys keyed on the meshes)."""
    mr = root.mmd_root
    n = sum(len(getattr(mr, c)) for c in ("group_morphs", "bone_morphs", "material_morphs", "uv_morphs"))
    if not n:
        return 0
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.mmd_tools.morph_slider_setup(type="BIND")
    for o in root.children_recursive:
        if getattr(o, "mmd_type", "") == "PLACEHOLDER":
            link(o, coll)
            o.hide_render = True
    return n


def relink_textures(meshes, folder):
    """Images whose files are missing: find a file with the same name (any case) under the model's folder and point
    the image at it (packs often move textures into subfolders the PMX does not mention). Returns (relinked, still
    missing) names."""
    index = {}
    for d, _dirs, files in os.walk(folder):
        for f in files:
            index.setdefault(f.lower(), os.path.join(d, f))
    fixed, missing = [], []
    seen = set()
    for m in meshes:
        for slot in m.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                img = getattr(node, "image", None)
                if img is None or img.name in seen or img.source != "FILE":
                    continue
                seen.add(img.name)
                path = bpy.path.abspath(img.filepath)
                if os.path.exists(path):
                    continue
                hit = index.get(os.path.basename(path.replace("\\", "/")).lower())
                if hit:
                    img.filepath = hit
                    img.reload()
                    fixed.append(img.name)
                else:
                    missing.append(img.name)
    return fixed, missing


def material_name(mat):
    """A model material's PMX name (mmd_tools keeps it as `name_j`; Blender's own name may carry a .001 suffix)."""
    mm = getattr(mat, "mmd_material", None)
    return getattr(mm, "name_j", "") or mat.name


def paint(name, meshes, spec, palette):
    """`[[cast]] materials`: {PMX material name: {color | tint | hide}} applied to the model's own materials in the scene.
    `color` (a palette slot or #hex) makes the material that flat colour (its texture dropped; its toon and sphere kept:
    MMD lights it as ambient + 0.6 x diffuse, so diffuse is the colour and ambient 0.4 of it), `tint` multiplies its
    diffuse and ambient colours (the texture's painted detail stays), `hide = true` draws it no more (alpha 0). Returns
    {name: what was done}; a name the model lacks or an unknown key is an error that lists what there is."""
    if not spec:
        return {}
    mats = {}
    for ob in meshes:
        for slot in ob.material_slots:
            if slot.material is not None:
                mats.setdefault(material_name(slot.material), slot.material)
    out = {}
    for pmx_name, ch in spec.items():
        mat = mats.get(pmx_name)
        if mat is None:
            raise BuildError(f"cast {name!r}: materials: no material {pmx_name!r} (the model has {sorted(mats)})")
        if not isinstance(ch, dict) or not ch or set(ch) - {"color", "tint", "hide"}:
            raise BuildError(f"cast {name!r}: materials.{pmx_name} = {ch!r}: give color, tint or hide")
        if sum(k in ch for k in ("color", "tint")) > 1:
            raise BuildError(f"cast {name!r}: materials.{pmx_name}: color or tint, not both")
        mm = mat.mmd_material
        done = []
        try:
            if "color" in ch:
                c = PAL.linear(PAL.resolve(str(ch["color"]), palette))
                mm.diffuse_color, mm.ambient_color = c, tuple(0.4 * x for x in c)
                shader = mat.node_tree.nodes.get("mmd_shader") if mat.use_nodes else None
                if shader is not None and "Base Tex Fac" in shader.inputs:
                    shader.inputs["Base Tex Fac"].default_value = 0.0
                done.append(f"color {ch['color']}")
            if "tint" in ch:
                c = PAL.linear(PAL.resolve(str(ch["tint"]), palette))
                mm.diffuse_color = tuple(a * b for a, b in zip(mm.diffuse_color, c))
                mm.ambient_color = tuple(a * b for a, b in zip(mm.ambient_color, c))
                done.append(f"tint {ch['tint']}")
        except KeyError as e:
            raise BuildError(f"cast {name!r}: materials.{pmx_name}: {e.args[0]}") from None
        if ch.get("hide"):
            mm.alpha = 0.0
            done.append("hidden")
        out[pmx_name] = ", ".join(done) or "unchanged"
    return out


def run(ctx):
    coll = collection("Cast")
    out = {}
    for spec in ctx.data.get("cast", []):
        name = spec["name"]
        try:
            physics = CN.physics(spec)
        except ValueError as e:
            raise BuildError(f"cast {name!r}: {e}") from None
        pmx, rig = resolve(ctx, spec)
        root, arm, meshes, new = import_model(pmx, physics, ctx.cache)
        title = CN.root_name(spec)
        root.name = title
        arm.name = CN.armature_name(spec)
        for i, m in enumerate(meshes):
            m.name = f"{title}_mesh{i}"
        for o in new:
            link(o, coll)
        if spec.get("parent"):
            par = bpy.data.objects.get(spec["parent"])
            if par is None:
                raise BuildError(f"cast {name!r}: parent object {spec['parent']!r} not found")
            root.parent = par
        root.location = Vector(spec.get("at", (0.0, 0.0, 0.0)))
        root.rotation_euler = Euler((0.0, 0.0, math.radians(float(spec.get("yaw", 0.0)))))
        m = Member(name, spec)
        m.root, m.arm, m.meshes, m.rig, m.pmx = root, arm, meshes, rig, pmx
        ctx.cast[name] = m
        if rig and rig.get("source", {}).get("sha1") and rig["source"].get("path") and \
                os.path.abspath(rig["source"]["path"]) != os.path.abspath(pmx):
            ctx.log(f"cast {name}: rig.json was made from {rig['source']['path']}, the model is {pmx}")
        fixed, missing = relink_textures(meshes, os.path.dirname(pmx))
        painted = paint(name, meshes, spec.get("materials"), ctx.palette)
        bound = bind_morphs(root, coll)
        out[name] = {"armature": arm.name, "meshes": len(meshes), "physics": physics, "rig": bool(rig),
                     "textures_relinked": fixed, "textures_missing": missing, "bound_morphs": bound}
        if painted:
            out[name]["materials"] = painted
        ctx.log("cast", name, arm.name)
    return out
