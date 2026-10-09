"""Run inside Blender (make_hand does): blender -b bchan_edited.blend --python export_glove.py -- RAW.npz

B-chan's gloves at rest as RAW.npz: the cage (the mirror applied, solidify and subdivision off), the rig's left and right
deform-group weights per vertex, and its deform bones (heads and tails, rest pose, world space). Blender 4.2: numpy only."""
import sys


def main():
    import bpy
    import numpy as np

    out = sys.argv[sys.argv.index("--") + 1]
    rig = bpy.data.objects["BChan Rig_rig"]
    rig.data.pose_position = "REST"
    obj = bpy.data.objects["Gloves"]
    for m in obj.modifiers:
        if m.type in ("SOLIDIFY", "SUBSURF"):
            m.show_viewport = m.show_render = False
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    mw = np.array(obj.matrix_world)
    V = np.array([tuple(v.co) for v in me.vertices], float)
    V = (np.c_[V, np.ones(len(V))] @ mw.T)[:, :3]
    sizes = np.array([p.loop_total for p in me.polygons], int)
    flat = np.array([me.loops[i].vertex_index for p in me.polygons
                     for i in range(p.loop_start, p.loop_start + p.loop_total)], int)
    groups = [g.name for g in obj.vertex_groups]
    keep = [g for g in groups if g.endswith(".l") or g.endswith(".r")]
    W = np.zeros((len(V), len(keep)))
    col = {groups.index(g): k for k, g in enumerate(keep)}
    for i, v in enumerate(me.vertices):
        for g in v.groups:
            if g.group in col:
                W[i, col[g.group]] = g.weight
    deform = [b for b in rig.data.bones if b.use_deform]
    rmw = rig.matrix_world
    np.savez(out, verts=V, face_flat=flat, face_sizes=sizes, weight_names=np.array(keep), weights=W,
             bone_names=np.array([b.name for b in deform]), bone_heads=np.array([tuple(rmw @ b.head_local) for b in deform]),
             bone_tails=np.array([tuple(rmw @ b.tail_local) for b in deform]))
    ev.to_mesh_clear()
    print("EXPORTED", len(V), "vertices", len(sizes), "faces", len(keep), "groups")


if __name__ == "__main__":
    main()
