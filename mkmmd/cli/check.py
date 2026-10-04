"""mk check: run named metrics with thresholds; exit 1 when any fails."""
import argparse
import json
import time

from .. import checks as CH
from .common import CHECK_FAILED, add_project_arg, emit, get_project, scene_path, UsageError

HELP = """Measure a scene and compare against thresholds.

With no METRIC, runs every [[check]] in mk.toml (or the ones named by --only). With a METRIC, runs that one metric
ad hoc. All sampled metrics share one Blender pass over the union of their frames.

Metrics (mk check --list for their arguments):
  jitter         shake of hair / ears / tails relative to the head (ratio; calm < 0.5)
  contact        distance between two tracked points (nib on paper, hand on a wheel)
  penetration    chains sinking into the body and scene colliders (mm)
  foot_slide     planted feet sliding (mm per frame)
  joint_limits   elbows, knees, wrists, neck, spine in human ranges (degrees over)
  framing        subject inside every output aspect's safe area (margin)
  occlusion      subject hidden behind objects (share of points)
  camera_inside  camera inside a closed mesh (frames)
  flicker        temporal noise in rendered frames
  palette        near-black share / distance to a palette in rendered frames

Examples:
  mk check                                         # the project's checks
  mk check --only "back hair is calm" --frames 181:400
  mk check jitter --args '{"family": "back_hair"}' --max 0.5 --frames 181:918
  mk check contact --args '{"a": "obj(\\"Pen\\").loc", "b": "track:nib.target", "when": "track:nib.writing"}' --max 0.1
"""


def add(sub):
    p = sub.add_parser("check", help="measure and compare against thresholds", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args_pos", nargs="*", metavar="[SCENE.blend] [METRIC]")
    p.add_argument("--args", dest="metric_args", default="{}", help="metric arguments as JSON (ad-hoc metric)")
    p.add_argument("--max", type=float, help="fail above this value (ad-hoc metric)")
    p.add_argument("--min", type=float, help="fail below this value (ad-hoc metric)")
    p.add_argument("--name", help="name for the ad-hoc check")
    p.add_argument("--only", action="append", metavar="NAME", help="run only these project checks (repeatable)")
    p.add_argument("--frames", help="override every check's frames")
    p.add_argument("--list", action="store_true", help="list metrics with their arguments")
    p.add_argument("--keep-sample", action="store_true", help="keep the sampled .npz in ~/.cache/mk/samples")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    from ..checks import camera, image, motion  # noqa: F401  (register metrics)
    if args.list:
        emit({name: {"doc": m.doc, "args": m.args} for name, m in sorted(CH.METRICS.items())})
        return 0
    proj = get_project(args)
    pos = list(args.args_pos)
    blend_arg = pos.pop(0) if pos and pos[0].endswith(".blend") else None
    if pos:
        if len(pos) != 1:
            raise UsageError("pass at most one METRIC")
        try:
            margs = json.loads(args.metric_args)
        except json.JSONDecodeError as e:
            raise UsageError(f"--args is not valid JSON: {e}")
        checks = [{"name": args.name or pos[0], "metric": pos[0], "args": margs, "max": args.max, "min": args.min}]
    else:
        if proj is None:
            raise UsageError("no mk.toml here: pass a METRIC (and a scene), or run inside a project")
        checks = [c for c in proj.checks if not args.only or c.get("name") in args.only]
        if args.only:
            missing = set(args.only) - {c.get("name") for c in checks}
            if missing:
                raise UsageError(f"no checks named {sorted(missing)} (have {[c.get('name') for c in proj.checks]})")
        if not checks:
            raise UsageError("the project has no [[check]] entries")
    if pos and pos[0] not in CH.METRICS:
        raise UsageError(f"unknown metric {pos[0]!r} (have {', '.join(sorted(CH.METRICS))})")
    needs_scene = any(CH.METRICS[c["metric"]].uses_frames for c in checks if c.get("metric") in CH.METRICS)
    blend = scene_path(blend_arg, proj) if (needs_scene or blend_arg) else None
    ctx = CH.Context(blend, proj)
    t0 = time.time()
    results = CH.run(checks, ctx, frames_override=args.frames, keep_sample=args.keep_sample)
    ok = all(r["ok"] for r in results)
    emit({"ok": ok, "passed": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results),
          "seconds": round(time.time() - t0, 2), "scene": str(blend) if blend else None, "results": results})
    return 0 if ok else CHECK_FAILED
