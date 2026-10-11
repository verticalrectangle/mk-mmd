"""Blender ops the site asks for (docs/design.md: The site): a model file of another format as one .glb."""
from pathlib import Path

from .runtime import op

IMPORT = {".gltf": ("import_scene", "gltf"), ".glb": ("import_scene", "gltf"), ".obj": ("wm", "obj_import"),
          ".stl": ("wm", "stl_import"), ".ply": ("wm", "ply_import"), ".fbx": ("import_scene", "fbx"),
          ".usd": ("wm", "usd_import"), ".usdz": ("wm", "usd_import"), ".usda": ("wm", "usd_import"),
          ".usdc": ("wm", "usd_import")}


@op("model_to_glb")
def model_to_glb(args):
    """args: src (a model file Blender imports), out (.glb). The model as it shows, modifiers applied, no animation."""
    import bpy
    src, out = Path(args["src"]), Path(args["out"])
    kind = IMPORT.get(src.suffix.lower())
    if kind is None:
        raise ValueError(f"{src.name}: Blender has no importer for {src.suffix} here ({', '.join(sorted(IMPORT))})")
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    getattr(getattr(bpy.ops, kind[0]), kind[1])(filepath=str(src))
    meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
    if not meshes:
        raise ValueError(f"{src.name}: no mesh in it")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_apply=True, export_animations=False,
                              export_skins=False, export_morph=False, export_cameras=False, export_lights=False,
                              export_yup=True)
    return {"out": str(out), "meshes": len(meshes)}
