"""Run inside Blender (make_body does): blender -b human_base_meshes_bundle.blend --python export_body.py -- RAW.npz

The stylized female body of Blender Studio's Human Base Meshes as RAW.npz: its vertices (world space, metres, Z up, facing
-Y), its faces (face_flat, face_sizes) and its UV map per face corner (uv, in the order of face_flat). Blender 4.2: numpy
only."""
import sys

OBJECT = "GEO-body_female_stylized"


def main():
    import bpy
    import numpy as np

    out = sys.argv[sys.argv.index("--") + 1]
    obj = bpy.data.objects[OBJECT]
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    mw = np.array(obj.matrix_world)
    V = np.array([tuple(v.co) for v in me.vertices], float)
    V = (np.c_[V, np.ones(len(V))] @ mw.T)[:, :3]
    sizes = np.array([p.loop_total for p in me.polygons], int)
    loops = [i for p in me.polygons for i in range(p.loop_start, p.loop_start + p.loop_total)]
    flat = np.array([me.loops[i].vertex_index for i in loops], int)
    uv = np.array([tuple(me.uv_layers.active.data[i].uv) for i in loops], float) if me.uv_layers.active else np.zeros((0, 2))
    np.savez(out, verts=V, face_flat=flat, face_sizes=sizes, uv=uv)
    ev.to_mesh_clear()
    print("EXPORTED", len(V), "vertices", len(sizes), "faces", len(uv), "uv corners")


if __name__ == "__main__":
    main()
