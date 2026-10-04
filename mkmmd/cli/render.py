"""mk render: render the cut of every output aspect to PNG frames (resumable, parallel, disk-guarded)."""
import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import bridge
from ..core import transition as TR
from .common import add_project_arg, cut_plan, emit, get_project, parse_frames, UsageError

PRESETS = {
    "draft": {"percent": 50, "samples": 16, "motion_blur": False},
    "preview": {"percent": 50, "samples": 32, "motion_blur": True},
    "final": {"percent": 100, "samples": 64, "motion_blur": True},
}
EST_MB = {50: 0.9, 100: 3.2}          # rough PNG size per 1080-class frame at that percentage

HELP = """Render the project's cut (shot cameras, per output aspect) to PNG frames in
<project>/renders/<preset>/<output>/<frame>.png. Frames already on disk are skipped, so a stopped render resumes;
--jobs N runs N Blender processes that share the frames. Before starting, mk checks that the disk can hold the frames.

Presets: draft (50 %, 16 samples, no motion blur), preview (50 %, 32 samples, motion blur), final (100 %, 64 samples,
motion blur). [render] in mk.toml can override samples, shutter, engine.

Shots with a render-time look (`style = "silhouette"`, `reflection = {...}` in [[shot]], see docs/design.md: Shots) are
rendered in it, inside the same Blender job: the frame on disk is the finished flat frame, not a pass. --no-styles renders
every shot as it is lit.

[[transition]] and [[insert]] entries (docs/design.md: Shots: Transitions and inserts) need more than the cut's frames: the
other shot's frames (plates), a silhouette's figure (mattes) and projected anchors. They are rendered next to the frames
in <output>/{plate,matte,back,point}/ on the same terms (claimed, resumable, counted in the disk check); mk post
composites them. --no-transitions leaves them out (and --no-styles does too: a figure needs its silhouette look).

Screen type ([[text]] with `screen`, docs/design.md: Text: Screen type) is kept off the frames, in <output>/screen/<frame>.png
(RGBA, only for the frames that have any); mk post lays it over the cut and its effects. --no-styles leaves it out.

Examples:
  mk render --preset draft                     # every output, the whole clip
  mk render --output 9x16 --frames t=0:5 --jobs 2
"""


def add(sub):
    p = sub.add_parser("render", help="render the cut per output aspect (resumable, parallel)", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", action="append", metavar="NAME", help="output aspect(s) (default: all)")
    p.add_argument("--frames", help="frame spec (default: the clip)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="draft")
    p.add_argument("--jobs", type=int, default=1, help="parallel Blender processes per output")
    p.add_argument("--samples", type=int)
    p.add_argument("--percent", type=int)
    p.add_argument("--no-styles", action="store_true", help="ignore the shots' silhouette / reflection looks")
    p.add_argument("--no-transitions", action="store_true", help="do not render the layers of [[transition]] / [[insert]]")
    add_project_arg(p)
    p.set_defaults(func=run)


def frames_dir(proj, preset, output):
    return proj.root / "renders" / preset / output


def run(args):
    proj = get_project(args, required=True)
    if not proj.blend or not proj.blend.exists():
        raise UsageError("no built scene: run `mk build` first")
    outs = [proj.output(n) for n in args.output] if args.output else proj.outputs
    if not outs:
        raise UsageError("no [[output]] in mk.toml")
    frames = parse_frames(args.frames, proj) if args.frames else list(range(proj.frame0, proj.last_frame + 1))
    cfg = dict(PRESETS[args.preset])
    cfg.update({k: v for k, v in proj.data.get("render", {}).items() if k in ("samples", "shutter", "engine")})
    if args.samples:
        cfg["samples"] = args.samples
    if args.percent:
        cfg["percent"] = args.percent
    plan = None if args.no_styles or args.no_transitions else cut_plan(proj)
    demands = TR.demands(plan, frames) if plan and (plan["transitions"] or plan["inserts"]) else {}
    need_mb = 0.0
    for o in outs:
        d = frames_dir(proj, args.preset, o.name)
        have = {p.stem for p in d.glob("*.png") if p.stat().st_size > 0} if d.exists() else set()
        todo = [f for f in frames if f"{f:05d}" not in have]
        scale = (max(o.size) / 1920.0) ** 2
        mb = EST_MB[100 if cfg["percent"] > 50 else 50]
        need_mb += len(todo) * mb * scale
        for _, item in TR.pending_items(d, demands):             # a plate is a frame; a matte and its back are mostly flat colour
            need_mb += (mb if item["kind"] == "plate" else 0.3 if item["kind"] == "matte" else 0.0) * scale
    free_mb = shutil.disk_usage(proj.root).free / 1e6
    if need_mb > free_mb - 1500:
        raise UsageError(f"not enough disk: about {need_mb:.0f} MB of frames, {free_mb:.0f} MB free (keeping 1.5 GB "
                         f"spare); render fewer frames or outputs, or free space")
    t0 = time.time()
    report = {}
    for o in outs:
        d = frames_dir(proj, args.preset, o.name)
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):                        # stale claims from a stopped render
            if p.stat().st_size == 0:
                p.unlink()
        TR.clear_claims(d)
        job = {"frames": frames, "out": str(d), "size": list(o.size), "aspect": o.name, "styles": not args.no_styles,
               "demands": {str(f): items for f, items in demands.items()}, **cfg}
        n = max(1, args.jobs)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = []
            for k in range(n):
                futs.append(ex.submit(bridge.run, "render_frames", job, proj.blend, proj, 3600 * 12, False))
                if k < n - 1:
                    time.sleep(8)                          # stagger loading
            res = [f.result() for f in futs]
        looks = {}
        for r in res:
            for k, v in r.get("looks", {}).items():
                looks[k] = looks.get(k, 0) + v
        report[o.name] = {"dir": str(d), "rendered": sum(r["rendered"] for r in res),
                          "s_per_frame": max(r["s_per_frame"] for r in res), "aspect_bound": res[0]["aspect_bound"],
                          "frames_on_disk": len([p for p in d.glob("*.png") if p.stat().st_size > 0])}
        if looks:
            report[o.name]["looks"] = looks               # frames rendered in a silhouette / reflection look
        if demands:
            report[o.name]["layers"] = {"drawn": sum(r.get("layers", 0) for r in res),
                                        "pending": len(TR.pending_items(d, demands))}
    emit({"preset": args.preset, "seconds": round(time.time() - t0, 1), "outputs": report})
    return 0
