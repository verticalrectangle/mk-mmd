"""Mesh and material helpers shared by library prop and set builders."""
import math

import bmesh
import bpy
from mathutils import Vector

from ...core.palette import linear


def mesh_obj(name, bm, coll, parent, mat=None):
    """Turn a bmesh into an object linked to `coll`, parented to `parent`, with one material."""
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    if mat is not None:
        me.materials.append(mat)
    return o


def as_collider(o):
    """Hidden from render, drawn as wire: a shape only the solvers and checks use."""
    o.hide_render = True
    o.display_type = "WIRE"
    o["mk_collider"] = True


def steady_order(bm, faces=None, edges=True):
    """Put `faces` of `bm` (default: all of them) and their edges in the order of their vertices, among the places
    they hold: the other faces and edges stay where they are. Blender hands over the faces and the edges that
    create_uvsphere and extrude_face_region make, and the faces of an exact Boolean, in an order that differs from run
    to run (each face is the same, corner for corner, and so are the vertices; Blender 4.2): a scene built twice then
    holds two different meshes (after a bevel, which follows the edges, not even the same points), and the second
    one's bake differs from the first's, so a render cannot tell a scene that was built again from one that changed
    (mkmmd.render.reuse). Call it after the operator; `edges` False leaves the edges alone (a finished mesh: nothing
    follows them). The sort moves what it reorders: a BMFace or BMEdge of theirs held from before names another one
    afterwards (bmesh's sort), so take them again from their vertices (`v.link_faces`, `bm.faces.get(verts)`), which
    do not move."""
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    fs = list(bm.faces if faces is None else faces)
    es = [] if not edges else list(bm.edges) if faces is None else list({e for f in fs for e in f.edges})
    for seq, items, key in ((bm.edges, es, lambda e: tuple(sorted(v.index for v in e.verts))),
                            (bm.faces, fs, lambda f: tuple(v.index for v in f.verts))):
        keys = {x.index: key(x) for x in items}
        to = dict(zip(sorted(keys, key=keys.__getitem__), sorted(keys)))     # an index -> the place it takes
        if any(a != b for a, b in to.items()):
            seq.sort(key=lambda x: to.get(x.index, x.index))
            seq.index_update()


def box(name, center, size, coll, parent, rot=(0.0, 0.0, 0.0), mat=None, collider=False):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    o = mesh_obj(name, bm, coll, parent, mat)
    o.location = Vector(center)
    o.rotation_euler = tuple(math.radians(a) for a in rot)
    if collider:
        as_collider(o)
    return o


def cylinder(name, center, radius, depth, coll, parent, rot=(0.0, 0.0, 0.0), mat=None, collider=False, segments=32):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth)
    o = mesh_obj(name, bm, coll, parent, mat)
    o.location = Vector(center)
    o.rotation_euler = tuple(math.radians(a) for a in rot)
    if collider:
        as_collider(o)
    return o


def torus(name, center, axis, radius, tube, coll, parent, mat=None, seg=48, ring=12):
    """A torus of `radius` (centre of the tube) about `axis`."""
    bm = bmesh.new()
    verts = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        row = []
        for j in range(ring):
            b = 2 * math.pi * j / ring
            r = radius + tube * math.cos(b)
            row.append(bm.verts.new((r * math.cos(a), r * math.sin(a), tube * math.sin(b))))
        verts.append(row)
    for i in range(seg):
        for j in range(ring):
            bm.faces.new((verts[i][j], verts[(i + 1) % seg][j], verts[(i + 1) % seg][(j + 1) % ring],
                          verts[i][(j + 1) % ring]))
    o = mesh_obj(name, bm, coll, parent, mat)
    o.location = Vector(center)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(axis).normalized())
    return o


def material(name, colour, rough=0.5, metal=0.0, emission=None, strength=0.0):
    """Principled material from a hex colour; optional emission (hex) at `strength`."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*linear(colour), 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*linear(emission), 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m
