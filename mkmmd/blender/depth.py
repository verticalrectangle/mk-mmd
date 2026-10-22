"""The geometry a camera sees in a Blender scene, for mkmmd.core.clip (docs/design.md: Shots: Clip ranges): `Depth()`
keeps every rendered object's triangles in chunks (made once, from the first frame it is read on), `frame()` takes the
scene's current frame (call it after a frame change), and `least(camera, size)` is the depth of the nearest geometry
that camera sees on it for a picture of `size` (w, h) pixels; `near(camera, size)` the clip_start that makes."""
import bpy
import numpy as np

from ..core import clip as CL

GEOMETRY = ("MESH", "CURVE", "SURFACE", "FONT", "META")          # read as triangles
BOXES = ("CURVES", "POINTCLOUD", "VOLUME", "GPENCIL")            # counted as solid boxes


def _in_drawn_collection(ob):
    """Whether one of the collections an object is in is drawn by a render (an object in none is drawn)."""
    return not ob.users_collection or any(not c.hide_render for c in ob.users_collection)


class Depth:
    def __init__(self):
        self.pieces = {}              # object name -> (Piece, (vertices, triangles) it was made for)
        self.drawn = set()            # objects found in a drawn collection (asked again while they are not)
        self.items, self.read = [], {}

    def frame(self):
        """Take what the scene renders on its current frame: every drawn object's box, frame and (when it is wanted)
        triangles; instances of geometry made by other objects count as solid boxes."""
        dg = bpy.context.evaluated_depsgraph_get()
        vl = bpy.context.view_layer.objects
        self.items, self.read = [], {}
        for inst in dg.object_instances:
            ob = inst.object
            if ob.type not in GEOMETRY and ob.type not in BOXES:
                continue
            orig = ob.original
            if not inst.is_instance and orig.hide_render:
                continue
            if inst.is_instance or ob.type in BOXES:
                get = _none                                  # an instance is drawn by its instancer's switch
            else:
                name = orig.name
                if name not in self.drawn:
                    if name not in vl or not _in_drawn_collection(orig):
                        continue
                    self.drawn.add(name)
                get = (lambda ob=ob: self._piece(ob))
            box = np.array([tuple(c) for c in ob.bound_box], float)
            self.items.append(((box.min(0), box.max(0)), np.array(inst.matrix_world, float), get))

    def _piece(self, ob):
        name = ob.original.name
        if name in self.read:
            return self.read[name]
        me = ob.data if ob.type == "MESH" else ob.to_mesh()
        try:
            n, m = len(me.vertices), len(me.loop_triangles)
            V = np.empty(3 * n, np.float32)
            me.vertices.foreach_get("co", V)
            V = V.reshape(-1, 3).astype(float)
            kept = self.pieces.get(name)
            if kept is None or kept[1] != (n, m):
                F = np.empty(3 * m, np.int32)
                me.loop_triangles.foreach_get("vertices", F)
                kept = self.pieces[name] = (CL.Piece(V, F.reshape(-1, 3)), (n, m))
        finally:
            if ob.type != "MESH":
                ob.to_mesh_clear()
        self.read[name] = got = (kept[0], V)
        return got

    def least(self, cam, size):
        """The depth (m) of the nearest geometry the camera object `cam` sees on the current frame, for a picture of
        `size` (w, h) pixels (inf: none; None for a camera that is not a perspective one)."""
        cd = cam.data
        if cd.type != "PERSP":
            return None
        sensor = cd.sensor_height if cd.sensor_fit == "VERTICAL" else cd.sensor_width
        view = CL.View(cd.lens, sensor, size, (cd.shift_x, cd.shift_y), cd.sensor_fit)
        to_cam = np.linalg.inv(np.array(cam.matrix_world.normalized(), float))
        return CL.least_depth([(box, to_cam @ M, get) for box, M, get in self.items], view)

    def near(self, cam, size):
        """The clip_start `cam` wants on the current frame for a picture of `size` (None: leave it as it is)."""
        d = self.least(cam, size)
        return None if d is None else min(CL.near(d), 0.5 * cam.data.clip_end)


def _none():
    return None
