"""mk q: ask any question about a scene, frame by frame."""
import argparse
import math

from .. import bridge
from .common import add_project_arg, emit, get_project, parse_frames, scene_path, UsageError

HELP = """Evaluate a Python expression in a scene at each frame and print the values.

Names available in EXPR:
  bone(NAME[, armature])  world-space view of a posed bone: .head .tail .center .dir .length .matrix .quat
                          NAME is a Blender name, a semantic name (head, wrist.R, index2.L) or a PMX name (右手首)
  obj(NAME)               world-space view of an object: .loc .matrix .quat .euler .dims .bbox .visible
  morph(NAME)             shape-key value on the model's meshes
  cam()                   the active camera (an obj view); screen(point) -> camera-view coordinates (x, y, depth)
  dist(a, b), angle(a, b) helpers; frame, t (clip seconds when in a project), np, math, Vector, bpy, scene

Examples:
  mk q build/stan.blend 'bone("head").head' --frames 181:280
  mk q 'dist(bone("index_tip.R").head, obj("Pen").loc)' --frames all --summary
  mk q build/stan.blend --list bones
"""


def add(sub):
    p = sub.add_parser("q", help="evaluate an expression over frames", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args", nargs="*", metavar="[SCENE.blend] EXPR")
    p.add_argument("--frames", help="frame spec (default: the scene's current frame)")
    p.add_argument("--list", metavar="KIND",
                   help="list names instead: armatures bones semantic morphs cameras markers collections actions "
                        "objects")
    p.add_argument("--armature", help="which armature `bone()` uses when the scene has several")
    p.add_argument("--summary", action="store_true", help="print min/max/mean instead of every value")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args)
    pos = list(args.args)
    blend_arg = pos.pop(0) if pos and pos[0].endswith(".blend") else None
    blend = scene_path(blend_arg, proj)
    if args.list:
        data = bridge.run("list", {"kind": args.list, "armature": args.armature}, blend=blend, project=proj)
        emit({"scene": str(blend), "kind": args.list, "items": data})
        return 0
    if len(pos) != 1:
        raise UsageError("pass exactly one EXPR (quote it)")
    frames = parse_frames(args.frames, proj, default=None) if args.frames else None
    if frames is None:
        frames = [bridge.run("ping", blend=blend)["frame_range"][0]] if proj is None else [proj.frame0]
    data = bridge.run("q", {"expr": pos[0], "frames": frames, "armature": args.armature}, blend=blend, project=proj)
    out = {"scene": str(blend), "expr": pos[0]}
    if args.summary:
        out["summary"] = summarize(data["values"])
        out["frames"] = [frames[0], frames[-1], len(frames)]
    else:
        out["frames"] = data["frames"]
        out["values"] = data["values"]
    emit(out)
    return 0


def summarize(values):
    """min/max/mean of numbers, or of vector lengths (with the per-axis ranges) for vectors."""
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(values) and nums:
        return {"min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums), "n": len(nums)}
    vecs = [v for v in values if isinstance(v, list) and v and all(isinstance(c, (int, float)) for c in v)]
    if len(vecs) == len(values) and vecs:
        n = len(vecs[0])
        lens = [math.sqrt(sum(c * c for c in v)) for v in vecs]
        return {"axis_min": [min(v[i] for v in vecs) for i in range(n)],
                "axis_max": [max(v[i] for v in vecs) for i in range(n)],
                "length_min": min(lens), "length_max": max(lens), "n": len(vecs)}
    return {"n": len(values), "note": "values are not all numbers or vectors"}
