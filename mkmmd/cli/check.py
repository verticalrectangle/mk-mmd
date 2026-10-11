"""mk check: run named metrics with thresholds; exit 1 when any fails."""
import argparse
import json
import time

from .. import checks as CH
from .common import CHECK_FAILED, add_project_arg, emit, get_project, scene_path, UsageError

HELP = """Measure a scene and compare against thresholds.

With no METRIC, runs every [[check]] in mk.toml (or the ones named by --only). With a METRIC, runs that one metric
ad hoc. All sampled metrics share one Blender pass over the union of their frames.

A [[check]] entry is  name, metric, args = {...} (the metric's arguments, below), frames (a frame spec, default the
whole clip), min and max. It passes when min <= value <= max; a metric with a default limit (form 0.25, prop_body 8,
strum 10, wrist_bend 60, hand_mirror 5, mic_palm at least 0.3, hand_room at least 0) uses it when the entry gives
neither. A check that cannot be computed fails with an `error`. Exit 1 when any check fails, 0 when all pass. The
project's checks keep their last results in .mk/checks.json (the site's Checks tab reads them; not with --frames).

Metrics (mk check --list for their arguments):
  jitter         shake of hair / ears / tails relative to the head (ratio; calm < 0.5)
  contact        distance between two tracked points (nib on paper, hand on a wheel)
  penetration    chains sinking into the body and scene colliders (mm)
  foot_slide     planted feet sliding (mm per frame)
  joint_limits   elbows, knees, wrists, neck, spine in human ranges (degrees over)
  framing        subject inside every output's safe area (margin)
  occlusion      subject hidden behind objects (share of points)
  camera_inside  camera inside a closed mesh (frames)
  flicker        temporal noise in rendered frames
  palette        near-black share / distance to a palette in rendered frames
  form           how blocky a prop's modelled shapes are (0 smooth .. 1 boxes; hero props stay under 0.25)
  strum          the pick against the strings at every stroke of a strum spec: distance at the strike (mm), timing in detail
  prop_body      a prop's geometry inside the character's collision bodies (mm; a guitar resting on the body)
  wrist_bend     a hand's bend from its forearm while its moves hold it (degrees; the moves library keeps it under 60)
  hand_mirror    a two-hand move's hands, each against the other's mirror image across the body (mm)
  mic_palm       a fist round a mic or a hammer, its palm's share toward the body (the library keeps it over 0.3)
  hand_room      a hand at a place that does not touch the body, its room to the collision bodies (mm)

Examples:
  mk check                                         # the project's checks
  mk check --only "back hair is calm" --frames 181:400
  mk check jitter --args '{"family": "back_hair"}' --max 0.5 --frames 181:918
  mk check contact --args '{"a": "obj(\\"Pen\\").loc", "b": "track:nib.target", "when": "track:nib.writing"}' --max 0.1
  mk check build/scene.blend form --args '{"prop": "car"}'      # a prop's boxiness, worst parts in detail
"""


def add(sub):
    p = sub.add_parser("check", help="measure and compare against thresholds", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args_pos", nargs="*", metavar="[SCENE.blend] [METRIC]",
                   help="an ad-hoc check: an optional scene (default: the project's) and the metric to run")
    p.add_argument("--args", dest="metric_args", default="{}", help="metric arguments as JSON (ad-hoc metric)")
    p.add_argument("--max", type=float, help="fail above this value (ad-hoc metric)")
    p.add_argument("--min", type=float, help="fail below this value (ad-hoc metric)")
    p.add_argument("--name", help="name for the ad-hoc check")
    p.add_argument("--only", action="append", metavar="NAME", help="run only these project checks (repeatable)")
    p.add_argument("--frames", help="override every check's frames (frame spec: 181:280, 181:280:5, 100,140,200, "
                                    "t=1.5:3.0 in clip seconds, all)")
    p.add_argument("--list", action="store_true", help="list metrics with their arguments")
    p.add_argument("--keep-sample", action="store_true", help="keep the sampled .npz in ~/.cache/mk/samples")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    CH.load()
    if args.list:
        emit({name: {"doc": m.doc, "args": m.args, **({"default_max": m.default_max} if m.default_max is not None else {}),
                     **({"default_min": m.default_min} if m.default_min is not None else {})}
              for name, m in sorted(CH.METRICS.items())})
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
    needs_scene = any(CH.METRICS[c["metric"]].uses_frames or CH.METRICS[c["metric"]].sampled
                      for c in checks if c.get("metric") in CH.METRICS)
    blend = scene_path(blend_arg, proj) if (needs_scene or blend_arg) else None
    ctx = CH.Context(blend, proj)
    t0 = time.time()
    results = CH.run(checks, ctx, frames_override=args.frames, keep_sample=args.keep_sample)
    ok = all(r["ok"] for r in results)
    if not pos and not args.frames:
        CH.keep_results(proj, results, str(blend) if blend else None)
    emit({"ok": ok, "passed": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results),
          "seconds": round(time.time() - t0, 2), "scene": str(blend) if blend else None, "results": results})
    return 0 if ok else CHECK_FAILED
