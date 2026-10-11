"""Checks: named metrics with thresholds (docs/design.md: Checks).

A metric declares what it needs from the scene (bones, families, objects, expressions, the camera, colliders); the
runner merges the needs of every check into ONE Blender `sample` call over the union of their frames, then each
metric computes its numbers outside Blender. Render-based metrics read image files instead; ray-based ones run their
own Blender `visibility` call."""
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from .. import bridge
from .. import config as CFG
from ..assets import AssetError, Registry
from ..core import frames as FR
from ..core.cast import armature_name
from ..project import ProjectError

METRICS = {}


class CheckError(ValueError):
    pass


def metric(cls):
    METRICS[cls.name] = cls()
    return cls


class Metric:
    name = ""
    doc = ""
    args = {}               # arg -> help
    sampled = True          # needs the Blender sample op
    uses_frames = True      # evaluated over scene frames (image metrics read files instead)
    default_max = None      # the limit when a check gives neither min nor max
    default_min = None      # ... or the lower one

    def needs(self, args, ctx, need):
        return None

    def compute(self, args, ctx, data, frames, state):
        raise NotImplementedError


# ---------------------------------------------------------------- what to sample
class Need:
    def __init__(self):
        self.bones, self.families, self.objects, self.exprs, self.colliders = {}, {}, [], [], []
        self.meshes = []
        self.props = []
        self.camera = False

    def bone(self, arm, *names):
        lst = self.bones.setdefault(arm, [])
        lst += [n for n in names if n not in lst]

    def family(self, arm, *fams):
        lst = self.families.setdefault(arm, [])
        lst += [f for f in fams if f not in lst]

    def obj(self, *names):
        self.objects += [n for n in names if n not in self.objects]

    def expr(self, e):
        if e not in self.exprs:
            self.exprs.append(e)
        return self.exprs.index(e)

    def collider_specs(self, specs):
        idx = []
        for s in specs:
            key = json.dumps(s, sort_keys=True)
            keys = [json.dumps(c, sort_keys=True) for c in self.colliders]
            if key not in keys:
                self.colliders.append(s)
                keys.append(key)
            idx.append(keys.index(key))
        return idx

    def mesh(self, objects=(), prop=None, exclude=(), frame=None):
        """A mesh request: the evaluated geometry (world space) of the named `objects` and of everything a render shows
        under the prop root `prop`, minus `exclude` (names or fnmatch patterns), at one `frame` (None: as the scene
        stands). Returns its index for Data.mesh."""
        spec = {"objects": list(objects), "prop": prop, "exclude": list(exclude),
                "frame": None if frame is None else int(frame)}
        if spec not in self.meshes:
            self.meshes.append(spec)
        return self.meshes.index(spec)

    def prop_use(self, *names):
        """The use points (the card's `use`, stored on the prop's root when the scene was built) of these props."""
        self.props += [n for n in names if n not in self.props]

    def empty(self):
        return not (self.bones or self.families or self.objects or self.exprs or self.colliders or self.meshes
                    or self.props or self.camera)

    def job(self, frames, out):
        return {"frames": frames, "out": str(out), "bones": self.bones, "families": self.families,
                "objects": self.objects, "exprs": self.exprs, "camera": self.camera, "colliders": self.colliders,
                "meshes": self.meshes, "props": self.props}


# ---------------------------------------------------------------- sampled data
class Data:
    def __init__(self, npz_path, reply):
        with np.load(npz_path) as z:
            self.z = {k: z[k] for k in z.files}
        self.meta = reply
        self.frames = [int(f) for f in self.z["frames"]]
        self._fi = {f: i for i, f in enumerate(self.frames)}

    def rows(self, frames):
        return np.array([self._fi[int(f)] for f in frames], int)

    def _arm(self, arm):
        if arm not in self.meta["bones"]:
            raise CheckError(f"armature {arm!r} was not sampled")
        return self.meta["bones"][arm]

    def resolve(self, arm, names):
        """Blender names for requested names (as requested, with or without the optional '?' prefix, or already a
        Blender name). Missing optional bones are dropped."""
        m = self._arm(arm)
        out = []
        for n in names:
            bare = n[1:] if n.startswith("?") else n
            for key in (n, bare, "?" + bare):
                if key in m["resolved"]:
                    out.append(m["resolved"][key])
                    break
            else:
                if bare in m["names"]:
                    out.append(bare)
                elif not n.startswith("?"):
                    raise CheckError(f"bone {bare!r} was not sampled on {m['armature']}")
        return out

    def has(self, arm, name):
        bare = name[1:] if name.startswith("?") else name
        m = self._arm(arm)
        return bare in m["resolved"] or "?" + bare in m["resolved"] or bare in m["names"]

    def bones(self, arm, names, frames=None):
        """World matrices (F, n, 4, 4), rest matrices in armature space (n, 4, 4), resolved names."""
        m = self._arm(arm)
        res = self.resolve(arm, names)
        cols = [m["names"].index(r) for r in res]
        W = self.z[f"bone_world_{m['index']}"][:, cols]
        if frames is not None:
            W = W[self.rows(frames)]
        return W, self.z[f"bone_rest_{m['index']}"][cols], res

    def arm_world(self, arm, frames=None):
        A = self.z[f"arm_world_{self._arm(arm)['index']}"]
        return A if frames is None else A[self.rows(frames)]

    def family(self, arm, fams):
        sel = self._arm(arm)["families"]
        return [n for f in fams for n in sel.get(f, [])]

    def info(self, arm, name):
        m = self._arm(arm)
        i = m["names"].index(self.resolve(arm, [name])[0])
        return {k: m[k][i] for k in ("jp", "parent", "length", "children")}

    def expr(self, i, frames=None):
        v = self.z[f"expr_{i}"]
        return v if frames is None else v[self.rows(frames)]

    def obj(self, name, frames=None):
        W = self.z["obj_world"][:, self.meta["objects"].index(name)]
        return W if frames is None else W[self.rows(frames)]

    def colliders(self, idx):
        return [it for i in idx for it in self.meta["colliders"][i]]

    def use(self, prop):
        """A prop's use points (its card's `use`, as built; ask for them with Need.prop_use)."""
        try:
            return self.meta["props"][prop]
        except KeyError:
            raise CheckError(f"prop {prop!r}: no use points in the sample (rebuild the scene: they are stored when it is saved)") from None

    def mesh(self, i):
        """Mesh request i: {"vertices" (n,3), "triangles" (m,3), "object" (m,) index into "names", "names", "exempt"
        (names tagged `mk_form_exempt`), "roots" ({prop: world matrix (4,4)}), "frame"} in world space. A request whose
        names do not exist raises its CheckError."""
        meta = self.meta["meshes"][i]
        if meta.get("error"):
            raise CheckError(meta["error"])
        return {"vertices": self.z[f"mesh_v_{i}"], "triangles": self.z[f"mesh_t_{i}"], "object": self.z[f"mesh_o_{i}"],
                "names": meta["objects"], "exempt": meta.get("exempt", []), "frame": meta["frame"],
                "roots": {p: np.array(M, float) for p, M in meta.get("roots", {}).items()}}

    def camera(self, frames):
        r = self.rows(frames)
        c = self.meta["camera"]
        return {"names": [c["names"][i] for i in r], "world": self.z["cam_world"][r], "lens": self.z["cam_lens"][r],
                "sensor": self.z["cam_sensor"][r], "shift": self.z["cam_shift"][r], "clip": self.z["cam_clip"][r],
                "sensor_fit": c["sensor_fit"], "resolution": c["resolution"], "pixel_aspect": c["pixel_aspect"]}

    def source_frames(self, sources, frames):
        """Posed (unscaled) rotation (F,S,3,3) and position (F,S,3) of shape sources; and their rest frames (S,3,3),
        (S,3): bones at rest in armature space, objects and the world at identity."""
        from ..solvers import geom
        F, S = len(frames), len(sources)
        Rs, ps = np.tile(np.eye(3), (F, S, 1, 1)), np.zeros((F, S, 3))
        R0, p0 = np.tile(np.eye(3), (S, 1, 1)), np.zeros((S, 3))
        for j, (kind, owner, name) in enumerate(sources):
            if kind == "bone":
                W, rest, _ = self.bones(owner, [name], frames)
                M = geom.unscaled(W[:, 0])
                rest = geom.unscaled(rest[0])
                R0[j], p0[j] = rest[:3, :3], rest[:3, 3]
            elif kind == "object":
                M = geom.unscaled(self.obj(name, frames))
            else:
                continue
            Rs[:, j], ps[:, j] = M[:, :3, :3], M[:, :3, 3]
        return Rs, ps, R0, p0


# ---------------------------------------------------------------- context: project, scene, cast, tracks
class Context:
    def __init__(self, scene, project=None):
        self.scene = scene
        self.project = project
        self._rigs = {}

    def cast(self, args):
        """(armature key for sampling, rig dict or None) from args: cast | armature | rig."""
        rig_path, arm = args.get("rig"), args.get("armature", "")
        if args.get("cast") or (self.project and self.project.cast and not arm and not rig_path):
            if not self.project:
                raise CheckError("cast needs a project (mk.toml)")
            try:
                c = self.project.cast_member(args.get("cast"))
            except ProjectError as e:
                raise CheckError(str(e))
            arm = armature_name(c)
            if c.get("rig"):
                rig_path = self.project.path(c["rig"])
            elif c.get("asset"):
                try:
                    rig_path = Registry().rig_path(c["asset"])
                except AssetError as e:
                    raise CheckError(str(e))
        rig = None
        if rig_path:
            p = Path(rig_path).expanduser()
            if self.project and not p.is_absolute():
                p = self.project.path(p)
            if str(p) not in self._rigs:
                self._rigs[str(p)] = json.loads(p.read_text(encoding="utf-8"))
            rig = self._rigs[str(p)]
        return arm, rig

    def colliders(self, spec):
        if isinstance(spec, str):
            if not self.project:
                raise CheckError(f"collider set {spec!r} needs a project")
            try:
                return self.project.colliders(spec)
            except ProjectError as e:
                raise CheckError(str(e))
        return list(spec or [])

    def track(self, ref, frames):
        """'track:NAME.channel' -> values for frames (float array; NaN where the track has no value)."""
        name, _, chan = ref[len("track:"):].partition(".")
        if not self.project:
            raise CheckError(f"{ref}: tracks need a project")
        try:
            t = self.project.track(name)
        except ProjectError as e:
            raise CheckError(str(e))
        if chan not in t:
            raise CheckError(f"{ref}: channel {chan!r} not in track {name!r} (have {sorted(k for k in t if k != 'frames')})")
        idx = {int(f): i for i, f in enumerate(t["frames"])}
        vals = t[chan]
        sample = np.array(vals[0], float)
        out = np.full((len(frames),) + sample.shape, np.nan)
        for k, f in enumerate(frames):
            i = idx.get(int(f))
            if i is not None and vals[i] is not None:
                out[k] = vals[i]
        return out

    def images(self, pattern):
        base = self.project.root if self.project else Path.cwd()
        p = Path(pattern).expanduser()
        files = sorted(p.parent.glob(p.name)) if p.is_absolute() else sorted(base.glob(pattern))
        if not files:
            raise CheckError(f"no images match {pattern!r}")
        return files


def is_expr(s):
    return any(ch in s for ch in "()[]+-*/")


# ---------------------------------------------------------------- running
def check_frames(check, ctx, override=None):
    spec = override or check.get("frames")
    p = ctx.project
    if spec is None:
        if p is None:
            raise CheckError(f"check {check.get('name')!r}: give frames (no project to default to)")
        return list(range(p.frame0, p.last_frame + 1))
    try:
        return FR.parse(str(spec), fps=p.fps if p else None, frame0=p.frame0 if p else None,
                        duration=p.duration if p else None)
    except FR.FrameSpecError as e:
        raise CheckError(str(e))


def judge(check, value):
    lo, hi = check.get("min"), check.get("max")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def load():
    """Import the metric modules (they register themselves in METRICS)."""
    from . import camera, form, hands, image, motion, props  # noqa: F401


def run(checks, ctx, frames_override=None, keep_sample=False):
    """Run checks; returns a list of results (name, metric, value, min/max, ok, detail | error)."""
    load()
    plan, need, union = [], Need(), set()
    results = [None] * len(checks)
    for i, c in enumerate(checks):
        m = METRICS.get(c.get("metric"))
        if m is None:
            results[i] = {"name": c.get("name"), "metric": c.get("metric"), "ok": False,
                          "error": f"unknown metric (have {', '.join(sorted(METRICS))})"}
            continue
        try:
            frames = check_frames(c, ctx, frames_override) if m.uses_frames else None
            state = m.needs(c.get("args", {}), ctx, need)
        except (CheckError, KeyError, ValueError) as e:
            results[i] = {"name": c.get("name"), "metric": m.name, "ok": False, "error": str(e)}
            continue
        if m.sampled and frames:
            union |= set(frames)
        plan.append((i, c, m, frames, state))
    data = None
    if (union or need.meshes) and not need.empty():
        out = CFG.cache_dir() / "samples" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        reply = bridge.run("sample", need.job(sorted(union), out), blend=ctx.scene, project=ctx.project,
                           timeout=3600)
        data = Data(out, reply)
        data.seconds = round(time.time() - t0, 2)
        if keep_sample:
            out.with_suffix(".json").write_text(json.dumps(reply, ensure_ascii=False), encoding="utf-8")
        else:
            out.unlink(missing_ok=True)
    for i, c, m, frames, state in plan:
        base = {"name": c.get("name") or m.name, "metric": m.name, "args": c.get("args", {})}
        try:
            value, detail = m.compute(c.get("args", {}), ctx, data, frames, state)
        except (CheckError, KeyError, ValueError, IndexError) as e:
            results[i] = {**base, "ok": False, "error": f"{type(e).__name__}: {e}"}
            continue
        lim = {k: c.get(k) for k in ("min", "max")}
        if lim["min"] is None and lim["max"] is None:
            lim = {"min": m.default_min, "max": m.default_max}
        res = {**base, "value": None if value is None else round(float(value), 4)}
        for k in ("min", "max"):
            if lim[k] is not None:
                res[k] = lim[k]
        res["ok"] = judge(lim, value)
        res["detail"] = detail
        results[i] = res
    return results


# ---------------------------------------------------------------- the last results
def results_path(project):
    return Path(project.mk_dir) / "checks.json"


def keep_results(project, results, scene):
    """Keep a run of the project's checks in .mk/checks.json, merged by name into the last runs' (an --only run updates
    its checks and keeps the others' results), in mk.toml's order: {"scene", "results": [{..., "ran"}]}."""
    path = results_path(project)
    try:
        old = {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8")).get("results", [])}
    except (OSError, ValueError, KeyError, TypeError):
        old = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    old.update({r["name"]: dict(r, ran=now) for r in results})
    order = [c.get("name") for c in project.checks]
    merged = [old[n] for n in order if n in old] + [r for n, r in old.items() if n not in order]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scene": scene, "results": merged}, ensure_ascii=False, indent=1), encoding="utf-8")


def last_results(project):
    """The project's [[check]] entries, each with its last result ({"scene", "results"}; one never run has no `ran`)."""
    try:
        doc = json.loads(results_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {"scene": None, "results": []}
    by = {r.get("name"): r for r in doc.get("results", [])}
    return {"scene": doc.get("scene"),
            "results": [by.get(c.get("name")) or {"name": c.get("name"), "metric": c.get("metric")} for c in project.checks]}
