"""mk build: assemble a scene from the project's mk.toml in one Blender session (docs/design.md: Building).

Stages run in order; each reads its own section of mk.toml and records what later stages need on the context:
  scene    fps, frame range (pre-roll before frame0), render size, collections
  sets     library set builders (sky, roads, tunnels, skylines, rooms) with their paths, surfaces and lights
  props    library or imported props placed in the world; their use points and colliders
  vehicles props driving along a set's path: lane, speed, body roll / pitch, spinning wheels, steering
  cast     models imported (no Bullet), named, placed
  pose     sitting / standing base pose, feet and hands on targets (leg IK, arm IK), fingers
  motion   VMD motions on NLA strips, retimed to the song's beats
  perform  gaze, blinks, breathing, sway, beat bob, startles, expressions, lip sync, twitches
  shots    the cut: baked cameras per shot and output aspect (mounts, aim, lag, shake, framing, focus), markers
  lights   [[light]] rigs in palette colours (mounted, aimed, keyed) and the [look] colour pipeline
  text     [[text]] kinetic type on set and prop surfaces: fitted, palette-coloured, typewriter and keyed numbers
  keys     [[key]] keys on set, prop and object properties (fog, flashes, steam, visibility)
  sim      secondary motion (hair, ears, tails, skirts) solved outside Blender, baked to keys
  save     the .blend
Solvers run as `python -m <module> IN.npz OUT.npz` on the CLI's Python, cached by a hash of their inputs."""
import hashlib
import json
import os
import subprocess
import time

import bpy
import numpy as np

from ...core import tables as TB
from ..runtime import CTX, op

TIMELINE = "audio/timeline.json"            # where a table's `timeline` defaults to (`mk timeline analyze` writes it)
STAGES = ["scene", "sets", "props", "vehicles", "cast", "pose", "motion", "perform", "shots", "lights", "text", "keys",
          "sim", "save"]


class BuildError(RuntimeError):
    pass


class Member:
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec
        self.root = self.arm = None
        self.meshes = []
        self.rig = None
        self.base = {}              # world-axis base rotations (bone -> Quaternion) the pose stage keyed
        self.ik = {}                # side -> (target empty, pole empty)
        self.seat = None            # world seat frame used by the pose stage


class Ctx:
    def __init__(self, job):
        self.project = job["project"] or {}
        self.data = self.project.get("data", {})
        self.root = self.project.get("root") or os.getcwd()
        self.fps = float(self.project.get("fps", 30))
        self.frame0 = int(self.project.get("frame0", 1))
        self.duration = float(self.project.get("duration", 10.0))
        sc = self.data.get("scene", {})
        self.end = int(sc.get("end", self.frame0 + round(self.duration * self.fps) - 1))
        self.start = int(sc.get("start", max(1, self.frame0 - int(3 * self.fps))))
        self.settle = int(sc.get("settle_frames", 24))           # rest -> base pose over these pre-roll frames
        self.cache = job["args"].get("cache") or os.path.join(self.root, ".mk", "cache")
        self.python = job["config"].get("python")
        self.assets = job["config"].get("assets")
        self.cast = {}
        self.props = {}
        self.sets = {}
        from ...core import palette as PAL
        look = self.data.get("look", {})
        self.palette = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
        self.log_lines = []
        self.t0 = time.time()

    # time
    def frame(self, t):
        return self.frame0 + t * self.fps

    def time(self, f):
        return (f - self.frame0) / self.fps

    @property
    def frames(self):
        return np.arange(self.start, self.end + 1)

    def path(self, rel):
        rel = os.path.expanduser(str(rel))
        return rel if os.path.isabs(rel) else os.path.join(self.root, rel)

    def log(self, *a):
        line = f"[{time.time() - self.t0:6.1f}s] " + " ".join(str(x) for x in a)
        self.log_lines.append(line)
        print("BUILD", line, flush=True)

    def section(self, name, cast_name=None):
        s = self.data.get(name, {})
        return s.get(cast_name, {}) if cast_name is not None else s

    def check_tables(self, section):
        """Refuse the project's `[<section>.<name>]` tables that name no cast member (the project's [[cast]], so a build that
        leaves out the cast stage checks the same) or hold a key the stage does not read (mkmmd.core.tables)."""
        try:
            TB.check_section(section, self.data.get(section), [c["name"] for c in self.data.get("cast", [])])
        except TB.TableError as e:
            raise BuildError(str(e)) from None

    def timeline(self, table, where):
        """The timeline JSON a table names with `timeline = "path"` (project-relative); without the key the project's own
        `audio/timeline.json`, where `mk timeline analyze` writes it. `where` names the table in the error."""
        ref = table.get("timeline", TIMELINE)
        try:
            with open(self.path(ref), encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            hint = "" if "timeline" in table else " (no `timeline` key: the default; run `mk timeline analyze` or set `timeline = \"path\"`)"
            raise BuildError(f"{where}: no timeline file {ref!r}{hint}") from None
        except (OSError, ValueError) as e:
            raise BuildError(f"{where}: cannot read the timeline {ref!r}: {e}") from None

    # solvers
    def solve(self, module, arrays, spec, tag):
        """Run `python -m module IN.npz OUT.npz` on the CLI's Python. IN holds `arrays` plus `spec` (JSON text);
        cached by a hash of all of it and of the solver package's source in <cache>/<solver>/<tag>-<key>.npz (editing
        a solver invalidates its results). Returns (output arrays, report dict) where report is the solver's `report`
        JSON text when it writes one."""
        arrays = dict(arrays, spec=np.array(json.dumps(spec, sort_keys=True, default=str)))
        h = hashlib.sha256()
        h.update(module.encode())
        pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           *module.split(".")[1:-1])
        for fn in sorted(os.listdir(pkg)):
            if fn.endswith(".py"):
                with open(os.path.join(pkg, fn), "rb") as fh:
                    h.update(fn.encode() + fh.read())
        for k in sorted(arrays):
            a = np.ascontiguousarray(arrays[k])
            h.update(k.encode())
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
        key = h.hexdigest()[:24]
        d = os.path.join(self.cache, module.rsplit(".", 1)[-1])
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"{tag}-{key}.npz")
        if os.path.exists(out):
            self.log(f"{module}: cached {os.path.basename(out)}")
        else:
            if not self.python:
                raise BuildError("no solver Python configured (the CLI passes its own interpreter)")
            inp = os.path.join(d, f"{tag}-{key}.in.npz")
            np.savez(inp, **arrays)
            t0 = time.time()
            env = dict(os.environ)
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            r = subprocess.run([self.python, "-m", module, inp, out], capture_output=True, text=True, env=env)
            os.unlink(inp)
            if r.returncode != 0 or not os.path.exists(out):
                raise BuildError(f"{module} failed ({r.returncode}):\n{r.stderr[-3000:]}")
            self.log(f"{module}: solved in {time.time() - t0:.1f}s")
        with np.load(out, allow_pickle=False) as z:
            res = {k: z[k] for k in z.files}
        report = json.loads(str(res.pop("report"))) if "report" in res else {}
        return res, report


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.actions, bpy.data.cameras,
                 bpy.data.lights, bpy.data.curves, bpy.data.images):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)


def collection(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def link(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


@op("build")
def build(args):
    import importlib
    job = {"project": CTX["project"], "config": CTX["config"], "args": args}
    ctx = Ctx(job)
    stages = args.get("stages") or STAGES
    report = {}
    for st in STAGES:
        if st not in stages:
            continue
        t0 = time.time()
        if st == "save":
            out = ctx.path(args["out"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for p in ctx.props.values():               # checks resolve {type = "prop", prop = name} and read use points from these
                p.root["mk_colliders"] = json.dumps(p.colliders)
                p.root["mk_use"] = json.dumps(p.card.get("use", {}))
            bpy.context.scene.frame_set(ctx.frame0)
            bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
            report["save"] = {"out": out}
        else:
            report[st] = importlib.import_module(f"{__name__}.{st}").run(ctx) or {}
        report[st]["seconds"] = round(time.time() - t0, 2)
        ctx.log(st, "done")
    return {"stages": report, "log": ctx.log_lines, "frames": [ctx.start, ctx.end], "frame0": ctx.frame0}
