"""Rounded geometry of the bedroom_80s set: nothing visible has a razor-sharp edge. A helper module: it registers no
builder.

Every hard edge (two faces meeting at 75 degrees or more, the angle the blockiness audit looks for) of an object gets a
bevel of a few millimetres, with at least 3 segments, and the bevel strips are shaded smooth; long runs (skirting,
picture rail, headrail, sill board) are swept from a section with rounded corners instead of being stacked boxes
(bedroom_maths: `sweep`, `fillet_polygon`). Things that stand on a wall are built 4 mm into it, so their rounded back
edges are buried and no groove shows where they meet the plaster."""
import math

import bmesh
import bpy

from ..props.cafe_kit import bm_append
from . import bedroom_maths as BM
from .cafe_nodes import box_geo

HARD = math.radians(75.0)        # the dihedral angle the blockiness audit counts as a hard edge


def exempt(*objects):
    """Tag objects `mk_form_exempt`: the `form` check lists them but does not score them. Used for architecture (walls,
    floor, ceiling, trim: boards are straight by nature) and for what is not modelled form at all (the far city, the
    roof-shadow plane)."""
    for o in objects:
        o["mk_form_exempt"] = 1


def bevel_hard(bm, width, segments=3, min_angle=HARD, smooth=True):
    """Bevel every edge of `bm` whose two faces meet at `min_angle` or more; the strips are shaded smooth. Returns the
    new faces."""
    bm.normal_update()
    edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) >= min_angle - 1e-6]
    if not edges:
        return []
    res = bmesh.ops.bevel(bm, geom=edges, offset=width, segments=segments, affect="EDGES", profile=0.5,
                          clamp_overlap=True, loop_slide=True)
    if smooth:
        for f in res["faces"]:
            f.smooth = True
    return res["faces"]


def box_bm(groups, face_slot=None):
    """A bmesh of boxes (x0, x1, y0, y1, z0, z1) in set coordinates, every arris bevelled. `groups` is a list of
    (boxes, bevel width, segments); each group is bevelled on its own, so a small moulding keeps its own radius. With
    `face_slot(normal) -> material index` the faces get their slot before the bevel (the strips inherit it)."""
    bm = bmesh.new()
    for boxes, width, segments in groups:
        part = bmesh.new()
        for b in boxes:
            box_geo(part, *b[:6])
        if face_slot is not None:
            part.normal_update()
            for f in part.faces:
                f.material_index = face_slot(f.normal)
        bevel_hard(part, width, segments)
        bm_append(bm, part)
        part.free()
    return bm


def to_object(R, base, bm, mats, group="room", smooth=False):
    """bmesh -> object "<set>_<base>" with the materials in slot order (the bmesh is freed)."""
    bm.normal_update()
    if smooth:
        for f in bm.faces:
            f.smooth = True
    me = bpy.data.meshes.new(R.oname(base))
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return R.obj(base, me, group)


def boxes_obj(R, base, groups, mats, group="room", face_slot=None):
    """One object from rounded boxes (see `box_bm`); `groups` may also be a plain list of boxes (3 mm, 3 segments)."""
    if groups and not isinstance(groups[0][0], (list, tuple)):
        groups = [(groups, 0.003, 3)]
    return to_object(R, base, box_bm(groups, face_slot), mats, group)


def sweep_bm(bm, points, profile, closed=False, mat=0):
    """Sweep `profile` along `points` into `bm` (bedroom_maths.sweep). Flat shading on the end caps, smooth on the rest.
    Returns the new faces."""
    V, faces, _ = BM.sweep(points, profile, closed)
    verts = [bm.verts.new((float(v[0]), float(v[1]), float(v[2]))) for v in V]
    new = []
    for f in faces:
        face = bm.faces.new([verts[i] for i in f])
        face.material_index = mat
        face.smooth = len(f) == 4
        new.append(face)
    return new


def moulded_obj(R, base, runs, profile, mats, group="room", cap_bevel=0.0012):
    """One object from sweeps of one section along several runs [(points, closed)] (e.g. `BM.wall_runs`)."""
    bm = bmesh.new()
    for points, closed in runs:
        part = bmesh.new()
        sweep_bm(part, points, profile, closed)
        bevel_hard(part, cap_bevel, 2)                   # the rims of the end caps and the folds at the mitres
        bm_append(bm, part)
        part.free()
    return to_object(R, base, bm, mats, group)
