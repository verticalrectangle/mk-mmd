"""Layers of the cut effects, rendered next to the cut's frames (docs/design.md: Shots: Transitions and inserts).

    layers = Layers(scene, looks, aspect, frames_dir, demands)   # demands: mkmmd.core.transition.demands for this render
    for f in frames:
        scene.frame_set(f)
        layers.run(f)                       # writes what frame f's effects need and is not on disk yet

An item is one of the plan's demands: a `plate` (a shot seen through its own camera and look, as the cut would show it),
a `matte` (a silhouette shot's figure alone as coverage, and its frame without the figure), a `point` (a world point, an
expression of the `mk q` language, seen through a shot's camera: its place in the frame, `p`, and `m`, how many frame heights a
metre spans at its depth, which turns a size in metres into pixels). Each is claimed with an empty file before it is drawn,
so a stopped render resumes and several Blender processes share the frames; the file that marks an item finished is
written last. Cameras: the shot table's camera of the shot for the output aspect, which `mk build` keyed over the frames
the plan needs (`keyed` in the table)."""
import contextlib
import json
import math
import os

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from ..core import transition as TR
from . import styles as ST
from .ops_core import _namespace


class Layers:
    def __init__(self, scene, looks, aspect, frames_dir, demands):
        self.sc, self.looks, self.aspect, self.dir = scene, looks, aspect, str(frames_dir)
        self.demands = {int(k): v for k, v in (demands or {}).items()}
        self.entries = {e["name"]: e for e in looks.table}
        self.frame = None
        self.done = 0

    def wants(self, frame):
        return int(frame) in self.demands

    def pending(self, frame):
        """Whether `frame` has an item that no process has claimed yet (nothing on disk for it)."""
        frame = int(frame)
        return any(not os.path.exists(os.path.join(self.dir, TR.rel_paths(item, frame)[-1]))
                   for item in self.demands.get(frame, []))

    # ---------------------------------------------------------------- cameras and files
    def _camera(self, shot, frame):
        entry = self.entries.get(shot)
        cam = (entry or {}).get("cameras", {}).get(self.aspect)
        if cam is None or cam not in bpy.data.objects:
            raise RuntimeError(f"shot {shot!r} has no camera for output {self.aspect!r} (run mk build)")
        keyed = entry.get("keyed")
        if keyed and not keyed[0] <= frame <= keyed[1]:
            raise RuntimeError(f"shot {shot!r}'s camera is keyed over frames {keyed[0]}..{keyed[1]}, not {frame}: a transition "
                               f"or insert in mk.toml needs it, run mk build again")
        return bpy.data.objects[cam]

    @contextlib.contextmanager
    def _through(self, shot, frame):
        """Look through `shot` at `frame`: the scene camera is that shot's (the markers that switch cameras on a frame change
        are set aside meanwhile) and the look is that shot's."""
        sc = self.sc
        cam = self._camera(shot, frame)
        markers = [(m.name, m.frame, m.camera) for m in sc.timeline_markers]
        old = sc.camera
        for m in list(sc.timeline_markers):
            sc.timeline_markers.remove(m)
        sc.camera = cam
        rs = ST.Restore()
        ims = sc.render.image_settings
        for k, v in (("file_format", "PNG"), ("color_mode", "RGB"), ("color_depth", "8"), ("compression", 15)):
            rs.attr(ims, k, v)
        try:
            self.looks.prepare(frame, self.aspect, shot=shot)
            yield
        finally:
            rs.run()
            for name, f, c in markers:
                sc.timeline_markers.new(name, frame=f).camera = c
            sc.camera = old

    def _claim(self, item, frame):
        """The item's absolute paths when this process may draw it, None when it is done or another process has it."""
        paths = [os.path.join(self.dir, p) for p in TR.rel_paths(item, frame)]
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            os.close(os.open(paths[-1], os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return None
        return paths

    # ---------------------------------------------------------------- items
    def run(self, frame):
        """Draw the missing items of `frame` (the scene is at that frame). Returns how many were drawn."""
        frame = int(frame)
        self.frame = frame
        todo = {}
        for item in self.demands.get(frame, []):
            paths = self._claim(item, frame)
            if paths:
                todo.setdefault(item["shot"], []).append((item, paths))
        n = 0
        for shot, items in todo.items():
            with self._through(shot, frame):
                for item, paths in items:
                    getattr(self, "_" + item["kind"])(item, paths)
                    n += 1
        self.done += n
        return n

    def _plate(self, item, paths):
        sc = self.sc
        if self.looks.kind:
            self.looks.render(paths[0])
        else:
            sc.render.filepath = paths[0]
            bpy.ops.render.render(write_still=True)

    def _matte(self, item, paths):
        lk = self.looks
        if lk.kind != "silhouette":
            raise RuntimeError(f"shot {item['shot']!r} is not a silhouette in output {self.aspect!r}: its figure cannot be a matte")
        p = lk.passes()
        ST._save(paths[0], lk.compose(p, subject=False))
        a = lk.matte()
        ST._save(paths[1], np.repeat(a[..., None], 3, axis=2))          # the coverage as grey: the CLI reads it back as such
        if float(a.max()) <= 0.0:
            print(f"WARNING shot {item['shot']!r} frame {self.frame}: no figure in the matte (is the subject out of frame, or "
                  f"hidden by `hide`?)", flush=True)

    def _point(self, item, paths):
        sc = self.sc
        ns = _namespace(None)
        ns["frame"] = self.frame
        try:
            co = Vector(eval(compile(item["expr"], "<mk transition>", "eval"), ns))     # noqa: S307 - the project's own expression
        except Exception as e:                                               # a bad expression is the project's to fix
            raise RuntimeError(f"point {item['key']}: {item['expr']!r} failed at frame {self.frame}: {e}") from None
        p = world_to_camera_view(sc, sc.camera, co)
        up = sc.camera.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
        q = world_to_camera_view(sc, sc.camera, co + up)                    # one metre up in the camera's plane
        r = sc.render
        m = math.hypot((q.x - p.x) * r.resolution_x / r.resolution_y, q.y - p.y)
        with open(paths[0], "w", encoding="utf-8") as fh:
            json.dump({"p": [round(p.x, 6), round(1.0 - p.y, 6)], "depth": round(p.z, 4), "m": round(m, 6)}, fh)
