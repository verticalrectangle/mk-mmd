"""mk build: assemble the project's scene from mk.toml (docs/design.md: Building)."""
import argparse
import time

from .. import bridge
from .common import add_project_arg, emit, get_project, UsageError

STAGES = ["scene", "sets", "props", "vehicles", "cast", "pose", "motion", "perform", "shots", "sim", "save"]

HELP = """Build the project's scene from mk.toml in one Blender session and save it to [project] blend.

Stages, in order: """ + " ".join(STAGES) + """. Solvers (secondary motion, grips) run outside Blender and are
cached in <project>/.mk/cache by a hash of their inputs, so a rebuild after a small change is quick.

Examples:
  mk build                         # everything, save to [project] blend
  mk build --until pose --out build/pose_test.blend
  mk build --skip sim              # no secondary motion (fast look at poses and performance)
"""


def add(sub):
    p = sub.add_parser("build", help="build the scene from mk.toml", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--until", choices=STAGES[:-1], help="stop after this stage (then save)")
    p.add_argument("--skip", action="append", default=[], choices=STAGES[:-1], help="skip a stage (repeatable)")
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
