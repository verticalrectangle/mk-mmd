"""Blender side of the shell toolkit (the geometry is `mkmmd.core.shell`, the rules are in docs/modelling.md).

    obj = mesh_object(name, mesh, coll, parent, mat_of)            a plain mesh part (sweeps, lathes, rounded boxes ...):
                                                                   custom normals from the area-weighted vertex normals
    obj = shell_object(name, cage, coll, parent, mat_of, levels=2, bevel=(0.003, 2), cutters=[...])
                                                                   a cage with edge creases, Subdivision Surface, Boolean
                                                                   cutters (wheel arches) and a Bevel for what stays hard
    cut = cutter_object(name, mesh, coll, parent, mat)             a hidden solid for a Boolean cutter

`mat_of(role)` gives the Blender material of a face's material index (`Mesh.Qm / Tm`); an object carries only the roles its
faces use. Everything is parented to `parent` (the prop's root) and named by the caller (`<prop>_<part>`); building a name
again replaces the old object.
"""
import math

import bpy
import numpy as np

from .mesh import as_collider  # noqa: F401  (re-exported: hidden collider boxes live next to the cutters)


def purge(name):
    """Remove the object `name` (a rebuild in the same session) with its mesh or light data."""
    o = bpy.data.objects.get(name)
    if o is not None:
        data = o.data
        bpy.data.objects.remove(o)
        if data is not None and data.users == 0:
            for coll in (bpy.data.meshes, bpy.data.lights):
                if coll.get(data.name) is data:
                    coll.remove(data)
                    break
    old = bpy.data.meshes.get(name)
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)


def as_cutter(o):
    """Hidden from render, drawn as wire: a solid only a Boolean modifier uses."""
    o.hide_render = True
    o.display_type = "WIRE"
    o["mk_cutter"] = True


def _edge_index(me):
    """Sorted edge keys of a mesh and their order (to find an edge from its two vertices)."""
    ev = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = np.sort(ev.reshape(-1, 2), axis=1)
    key = ev[:, 0] * max(len(me.vertices), 1) + ev[:, 1]
    order = np.argsort(key)
    return key[order], order


def mesh_data(name, mesh, mat_of, smooth=True, normals=True):
    """A Blender mesh from a `core.shell.Mesh`: faces smooth, per-vertex uv, edge creases (`crease_edge`) and, when
    `normals`, custom split normals from the area-weighted vertex normals."""
    V, Q, T = mesh.V, mesh.Q, mesh.T
    nq, nt = len(Q), len(T)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    loops = np.concatenate([Q.ravel(), T.ravel()]).astype(np.int32)
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    me.polygons.add(nq + nt)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.concatenate([np.full(nq, 4), np.full(nt, 3)]).astype(np.int32))
    roles = np.concatenate([mesh.Qm, mesh.Tm]).astype(np.int64)
    used = np.unique(roles)
    me.polygons.foreach_set("material_index", np.searchsorted(used, roles).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.full(nq + nt, bool(smooth)))
    me.update(calc_edges=True)
    if mesh.UV is not None:
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", mesh.UV[loops].astype(np.float32).ravel())
    if len(mesh.Ce):
        key, order = _edge_index(me)
        ce = np.sort(mesh.Ce, axis=1)
        want = ce[:, 0] * max(len(V), 1) + ce[:, 1]
        pos = np.clip(np.searchsorted(key, want), 0, len(key) - 1)
        hit = key[pos] == want
        w = np.zeros(len(me.edges), np.float32)
        w[order[pos[hit]]] = mesh.Cw[hit]
        me.attributes.new("crease_edge", "FLOAT", "EDGE").data.foreach_set("value", w)
    if normals and len(V):
        me.normals_split_custom_set_from_vertices(mesh.vertex_normals())
    for r in used:
        me.materials.append(mat_of(int(r)))
    return me, used


def _object(name, me, coll, parent, loc, rot):
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_euler = [math.radians(a) for a in rot]
    return o


def mesh_object(name, mesh, coll, parent, mat_of, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    """A `core.shell.Mesh` as an object under `parent` (see the module docstring)."""
    purge(name)
    me, _ = mesh_data(name, mesh, mat_of)
    return _object(name, me, coll, parent, loc, rot)


def cutter_object(name, mesh, coll, parent, mat_of, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    """A hidden solid for a Boolean cutter. The faces it leaves in the cut body take its material (`mat_of(role)`)."""
    purge(name)
    me, _ = mesh_data(name, mesh, mat_of, smooth=True, normals=False)
    o = _object(name, me, coll, parent, loc, rot)
    as_cutter(o)
    return o


def shell_object(name, cage, coll, parent, mat_of, levels=2, render_levels=None, bevel=None, cutters=(),
                 cutter_role=None, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), solver="EXACT"):
    """A cage (`core.shell.Mesh` with creases) as a smooth shell: a Subdivision Surface of `levels` (creases respected;
    `render_levels` for renders, default levels + 1), then a Boolean DIFFERENCE for every cutter object (the faces the cut
    creates take the material `cutter_role` through the cutter's own material: build the cutter with it), then a Bevel
    `bevel = (width m, segments)` on the edges sharper than 30 degrees (what the cuts leave hard). The modifiers stay live
    (no apply): a saved .blend carries the cage and renders the smooth surface."""
    purge(name)
    me, used = mesh_data(name, cage, mat_of, normals=False)
    slot = None
    if cutters:                                              # the cut faces take this slot of the body's materials
        role = 0 if cutter_role is None else int(cutter_role)
        if role in used.tolist():
            slot = used.tolist().index(role)
        else:
            me.materials.append(mat_of(role))
            slot = len(me.materials) - 1
        for cut in cutters:
            cut.data.polygons.foreach_set("material_index", np.full(len(cut.data.polygons), slot, np.int32))
    o = _object(name, me, coll, parent, loc, rot)
    md = o.modifiers.new("Subdivision", "SUBSURF")
    md.levels = int(levels)
    md.render_levels = int(levels + 1 if render_levels is None else render_levels)
    md.use_creases = True
    md.use_custom_normals = False
    for k, cut in enumerate(cutters):
        b = o.modifiers.new(f"Cut{k}", "BOOLEAN")
        b.operation = "DIFFERENCE"
        b.object = cut
        b.solver = solver
    if bevel:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width, bv.segments = float(bevel[0]), int(bevel[1])
        bv.limit_method = "ANGLE"
        bv.angle_limit = math.radians(30.0)
        bv.harden_normals = False
    return o


def evaluated_bounds(o):
    """(lo, hi) of the object's evaluated mesh (modifiers applied) in its parent's frame: evaluates the dependency graph."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg)
    pts = np.array([list(v) for v in ev.bound_box])
    return pts.min(0), pts.max(0)


def evaluated_mesh(o):
    """(V (n, 3), T (m, 3)) of the object's evaluated mesh (every modifier applied: the smooth surface a render shows) in
    the object's own frame, as float64 / int arrays. Evaluates the dependency graph."""
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    try:
        V = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", V)
        me.calc_loop_triangles()
        T = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", T)
    finally:
        ev.to_mesh_clear()
    return V.reshape(-1, 3).astype(np.float64), T.reshape(-1, 3).astype(np.int64)


def probe(o):
    """A `core.shell.Probe` on the object's evaluated surface: ray casts that put trims, lamps and plates exactly on a
    subdivided shell (whose limit surface lies inside its control cage)."""
    from ...core.shell import Probe
    return Probe(*evaluated_mesh(o))


def evaluate_cage(cage, levels=2):
    """A `core.shell.Probe` on the Subdivision Surface of a cage (`Mesh` with creases), before any object exists: lets a
    part module place its panels and lamps on the limit surface of the cage it is about to return. Leaves nothing behind."""
    from ...core.shell import Probe
    me, _ = mesh_data("_mk_probe", cage, lambda role: None, normals=False)
    o = bpy.data.objects.new("_mk_probe", me)
    bpy.context.scene.collection.objects.link(o)
    try:
        md = o.modifiers.new("Subdivision", "SUBSURF")
        md.levels = md.render_levels = int(levels)
        md.use_creases = True
        return Probe(*evaluated_mesh(o))
    finally:
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)


def exempt(o, why="graphic layer"):
    """Tag an object as not a form (a label, a stripe, a display segment lifted a fraction of a millimetre off a surface):
    the `form` check (no cuboids, no unbevelled hard edges) skips it. Use it for decals only, never to hide a form."""
    o["mk_form_exempt"] = why
    return o
