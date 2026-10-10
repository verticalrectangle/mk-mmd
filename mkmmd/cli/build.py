"""mk build: assemble the project's scene from mk.toml (docs/design.md: Building)."""
import argparse
import time

from .. import bridge
from .common import add_project_arg, emit, get_project, UsageError

STAGES = ["scene", "sets", "props", "vehicles", "cast", "pose", "motion", "perform", "shots", "lights", "text", "keys",
          "sim", "save"]

HELP = """Build the project's scene from mk.toml in one Blender session and save it to [project] blend.

Stages, in order: """ + " ".join(STAGES) + """. Each reads its own sections of mk.toml:
  scene     [scene], [[output]]       fps, frame range with the pre-roll before frame0, render size
  sets      [[set]]                   library sets: sky, roads, rooms
  props     [[prop]], [[scatter]]     library, card-file and PMX props, placement rules, use points, colliders
  vehicles  [[vehicle]]               props driving along a set's path
  cast      [[cast]]                  models imported (no Bullet), named, placed
  pose      [pose.<cast>]             sitting or standing, feet, arm IK, grips, worn props
  motion    [[motion.<cast>]]         VMD motions on NLA strips
  perform   [perform.<cast>]          gaze, blinks, breathing, sway, lip sync, strumming
  shots     [[shot]], [[transition]], [[insert]], [vector], [[ring]], [[freeze]]   the cameras, the cut, the looks' tables
  lights    [[light]], [look]         lights in palette colours, colour management
  text      [[text]]                  type on surfaces, lyric type, handwriting, screen type
  keys      [[key]]                   keys on set, prop and object properties
  sim       [sim.<cast>]              hair, ears, tails, skirts solved outside Blender and baked to keys
  save                                the .blend
Solvers (secondary motion, grips) run outside Blender and are cached in <project>/.mk/cache by a hash of their
inputs, so a rebuild after a small change is quick. Prints the per-stage report and the build log as JSON; WARNING
lines in the log are worth reading (a hand short of its goal, a prop over its form limit, a pattern matching nothing).

Examples:
  mk build                         # everything, save to [project] blend
  mk build --until pose --out build/pose_test.blend
  mk build --skip sim              # no secondary motion (fast look at poses and performance)
"""


def add(sub):
    p = sub.add_parser("build", help="build the scene from mk.toml", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--until", choices=STAGES[:-1], help="stop after this stage (then save)")
    p.add_argument("--skip", action="append", default=[], choices=STAGES[:-1],
                   help="skip a stage (repeatable); later stages need what sets, props and cast make")
    p.add_argument("--out", help="where to save (default: [project] blend)")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args, required=True)
    out = args.out or (str(proj.blend) if proj.blend else None)
    if not out:
        raise UsageError("set [project] blend in mk.toml or pass --out")
    stages = STAGES[:STAGES.index(args.until) + 1] + ["save"] if args.until else list(STAGES)
    stages = [s for s in stages if s not in args.skip]
    t0 = time.time()
    res = bridge.run("build", {"stages": stages, "out": out, "cache": str(proj.mk_dir / "cache")}, project=proj,
                     timeout=3600 * 3)
    res["seconds"] = round(time.time() - t0, 1)
    emit(res)
    return 0
