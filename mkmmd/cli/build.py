"""mk build: assemble the project's scene from mk.toml (docs/design.md: Building)."""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

from .. import bridge
from .. import memory as MEM
from .common import add_project_arg, emit, get_project, UsageError

STAGES = ["scene", "sets", "props", "vehicles", "cast", "pose", "motion", "perform", "ribbons", "shots", "lights", "text",
          "keys", "breathe", "sim", "folds", "cloth", "stepped", "clip", "save", "bake"]
BUILDING = STAGES[:-3]                     # what --until names; clip, save and bake follow
PEAK_FILE = "build_memory.json"            # in the project's cache: the last build's peak (one Blender's resident memory)
MARGIN = 2e9                               # bytes to spare beyond two Blenders at that peak

HELP = """Build the project's scene from mk.toml in one Blender session and save it to [project] blend.

Stages, in order: """ + " ".join(STAGES) + """. Each reads its own sections of mk.toml:
  scene     [scene], [[output]]       fps, frame range with the pre-roll before frame0, render size
  sets      [[set]]                   library sets: sky, roads, rooms
  props     [[prop]], [[scatter]]     library, card-file and PMX props, placement rules, use points, colliders
  vehicles  [[vehicle]]               props driving along a set's path
  cast      [[cast]]                  models imported (no Bullet), named, placed; group/material morphs bound
  pose      [pose.<cast>], [[move.<cast>]]   sitting or standing, feet, arm IK, grips, worn props, named moves
  motion    [[motion.<cast>]]         VMD motions on NLA strips
  perform   [perform.<cast>]          gaze, blinks, breathing, sway, lip sync, strumming
  ribbons   [[ribbon]]                3D paint strokes (crayon, toothpaste, neon) drawn on in the scene, kept clear of
                                      the cast's bodies
  shots     [[shot]], [[transition]], [[insert]], [[glitch]], [vector], [[ring]], [[freeze]], [[medium]]   cameras, cut,
                                      looks, medium shifts
  lights    [[light]], [look]         lights in palette colours, colour management
  text      [[text]]                  type on surfaces, lyric type, handwriting, screen type
  keys      [[key]]                   keys on set, prop and object properties
  breathe   [look] breathe            every colour the build took from the palette driven between it and the other
                                      palette over clip time (a probe Blender builds the stages above with that palette:
                                      beside the build when the memory holds both, else first)
  sim       [sim.<cast>]              hair, ears, tails, skirts solved outside Blender and baked to keys
  folds     [folds.<cast>]            fold shapes a joint angle drives (rig.json folds), keyed from the final pose
  cloth     [cloth.<cast>]            garments' cloth copies (rig.json cloth) simulated with Blender's cloth, cached
  stepped   [[stepped]]               the cast on twos: poses held 1 / fps s at a time, the cameras on every frame
  clip                                every shot camera starts at half the depth of the nearest geometry it sees, on
                                      every frame (fine depth: what a model hides behind the skin stays hidden)
  save                                the .blend
  bake                                <blend>.bake/: the clip for the site's scene viewer and mk's renderers
                                      (docs/design.md: The bake), every frame's bones, morphs, objects, visibility and
                                      shot cameras, the meshes, and the lit scene mkr draws (lights, materials, view LUT)
Solvers (secondary motion, grips) run outside Blender and are cached in <project>/.mk/cache by a hash of their
inputs, so a rebuild after a small change is quick. Prints the per-stage report and the build log as JSON; WARNING
lines in the log are worth reading (a hand short of its goal, a prop over its form limit, a pattern matching nothing).

Examples:
  mk build                         # everything, save to [project] blend
  mk build --until pose --out build/pose_test.blend
  mk build --skip sim              # no secondary motion (fast look at poses and performance)
  mk build --bake                  # bake the saved scene again (after an mk upgrade), without building it
"""


def add(sub):
    p = sub.add_parser("build", help="build the scene from mk.toml", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--until", choices=BUILDING, help="stop after this stage (then clip, save and bake)")
    p.add_argument("--skip", action="append", default=[], choices=BUILDING + ["clip", "bake"],
                   help="skip a stage (repeatable); later stages need what sets, props and cast make")
    p.add_argument("--bake", action="store_true", help="only bake the saved scene ([project] blend or --out) again")
    p.add_argument("--out", help="where to save (default: [project] blend)")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args, required=True)
    out = args.out or (str(proj.blend) if proj.blend else None)
    if not out:
        raise UsageError("set [project] blend in mk.toml or pass --out")
    if args.bake:
        if not Path(out).is_file():
            raise UsageError(f"no saved scene at {out}: run mk build first")
        stages, blend = ["bake"], out
    else:
        stages = STAGES[:STAGES.index(args.until) + 1] + ["clip", "save", "bake"] if args.until else list(STAGES)
        stages, blend = [s for s in stages if s not in args.skip], None
    t0 = time.time()
    cache = str(proj.mk_dir / "cache")
    job = {"stages": stages, "out": out, "cache": cache}
    memory = {}
    if "breathe" in stages and (proj.data.get("look") or {}).get("breathe"):
        peak, avail = _peak(proj), MEM.available()
        if _beside(peak, avail):
            memory["probe"] = "beside"
        else:                              # the probe builds first, so the two Blenders never hold memory at once
            memory["probe"] = "first"
            job["probe_ready"] = _probe_first(proj, stages, cache)
    try:
        res = bridge.run("build", job, blend=blend, project=proj, timeout=3600 * 3)
    finally:
        ready = job.get("probe_ready")
        if ready and not os.environ.get("MK_KEEP_JOBS"):   # the build's breathe stage removes it, unless it failed first
            shutil.rmtree(ready["folder"], ignore_errors=True)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024     # the biggest Blender of the run
    _record(proj, peak)
    res["memory"] = {"peak_gb": round(peak / 1e9, 1), **memory}
    res["seconds"] = round(time.time() - t0, 1)
    emit(res)
    return 0


def _peak(proj):
    """The last build's peak (bytes of one Blender's resident memory), or None."""
    try:
        return float(json.loads((proj.mk_dir / "cache" / PEAK_FILE).read_text(encoding="utf-8"))["peak"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _record(proj, peak):
    path = proj.mk_dir / "cache" / PEAK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"peak": peak, "at": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")


def _beside(peak, avail):
    """Whether the breathe probe may build beside the build: the memory available now holds two Blenders at the last
    build's peak with MARGIN to spare. Without a record (or a reading) it may not: the probe builds first."""
    return peak is not None and avail is not None and avail >= 2 * peak + MARGIN


def _probe_first(proj, stages, cache):
    """The breathe probe built now, before the build: the stages before `breathe` with the other palette, saved in the
    project's cache, where the build's breathe stage takes it (build/breathe.py: Ready)."""
    folder = Path(cache) / "breathe" / f"{time.strftime('%Y%m%d-%H%M%S')}-first"
    folder.mkdir(parents=True, exist_ok=True)
    blend = folder / "probe.blend"
    probe_stages = [s for s in STAGES[:STAGES.index("breathe")] if s in stages]
    t0 = time.time()
    try:
        bridge.run("build", {"stages": probe_stages, "cache": cache, "probe": str(blend)}, project=proj,
                   timeout=3600 * 3)
    except bridge.BlenderError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return {"folder": str(folder), "blend": str(blend), "stages": probe_stages, "seconds": round(time.time() - t0, 1)}
