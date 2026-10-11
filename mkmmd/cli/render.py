"""mk render: render the cut of every output aspect to PNG frames (resumable, parallel, disk-guarded)."""
import argparse
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import bridge
from ..core import transition as TR
from .common import RUNTIME, add_project_arg, cut_plan, emit, get_project, parse_frames, UsageError

PRESETS = {
    "draft": {"percent": 50, "samples": 16, "motion_blur": False},
    "preview": {"percent": 50, "samples": 32, "motion_blur": True},
    "final": {"percent": 100, "samples": 64, "motion_blur": True},
}
ENGINES = ("eevee", "cycles", "workbench", "mk")
EST_MB = {50: 0.9, 100: 3.2}          # rough PNG size per 1080-class frame at that percentage

HELP = """Render the project's cut (shot cameras, per output) to PNG frames in
<project>/renders/<preset>/<output>/<frame>.png (<frame> is the Blender frame, five digits). Frames already on disk are
skipped, so a stopped render resumes; --jobs N runs N Blender processes that share the frames. Before starting, mk
checks that the disk can hold the frames.

Presets: draft (50 %, 16 samples, no motion blur), preview (50 %, 32 samples, motion blur), final (100 %, 64 samples,
motion blur). [render] in mk.toml can override samples, shutter (motion blur, 0.35) and engine (eevee, cycles,
workbench, mk); --samples, --percent and --engine override them for one run.

Shots with a render-time look (`style = "silhouette"` or `"vector"`, `reflection = {...}` in [[shot]], see docs/design.md:
Shots) are rendered in it, inside the same Blender job: the frame on disk is the finished flat frame, not a pass.
--no-styles renders every shot as it is lit.

engine = "mk" (docs/design.md: Rendering) draws the vector look itself: its passes on the GPU in headless Chromium from the
scene's bake (`mk build` bakes it), finished by the same composer as Blender's. A shot it cannot draw (another look, type
over the picture, an object the bake cannot hold) renders in Blender with the scene's own engine, and so do the layers
below; the report names each shot's engine and why.

[[transition]], [[insert]] and [[glitch]] entries (docs/design.md: Shots: Transitions and inserts, Glitches) need more than
the cut's frames: the other shot's frames (plates), a silhouette's figure (mattes), a glitch's shot without its objects
(bare frames) and projected anchors. They are rendered next to the frames in <output>/{plate,matte,back,bare,point}/ on the
same terms (claimed, resumable, counted in the disk check); mk post composites them. --no-transitions leaves them out (and
--no-styles does too: a figure needs its silhouette look).

Screen type ([[text]] with `screen`, docs/design.md: Text: Screen type) is kept off the frames, in <output>/screen/<frame>.png
(RGBA, only for the frames that have any); mk post lays it over the cut and its effects. --no-styles leaves it out.

Examples:
  mk render --preset draft                     # every output, the whole clip
  mk render --output 9x16 --frames t=0:5 --jobs 2
"""


def add(sub):
    p = sub.add_parser("render", help="render the cut per output (resumable, parallel)", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", action="append", metavar="NAME", help="output(s) to render (default: all)")
    p.add_argument("--frames", help="frame spec: 181:280, 181:280:5, 100,140,200, t=0:5 (clip seconds), all "
                   "(default: the whole clip)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="draft", help="draft (default), preview or final")
    p.add_argument("--jobs", type=int, default=1, help="parallel Blender processes per output")
    p.add_argument("--samples", type=int, help="render samples (default: the preset's, or [render] samples)")
    p.add_argument("--percent", type=int, help="resolution percentage (default: the preset's, 50 or 100)")
    p.add_argument("--engine", choices=ENGINES, help="the render engine for this run (default: [render] engine, else the "
                   "scene's own)")
    p.add_argument("--no-styles", action="store_true", help="ignore the shots' silhouette / reflection looks")
    p.add_argument("--no-transitions", action="store_true",
                   help="do not render the layers of [[transition]] / [[insert]] / [[glitch]]")
    add_project_arg(p)
    p.set_defaults(func=run)


def frames_dir(proj, preset, output):
    return proj.root / "renders" / preset / output


def on_disk(d):
    return {p.stem for p in d.glob("*.png") if p.stat().st_size > 0} if d.exists() else set()


def mk_plan(proj, outs, frames, percent):
    """mk's renderer's plan for these frames (mkmmd.render.plan) from the scene's bake, which must be as new as the scene."""
    from ..render import plan as PL
    bake = proj.blend.with_suffix(".bake")
    try:
        if (bake / "bake.json").stat().st_mtime < proj.blend.stat().st_mtime:
            raise UsageError(f"the bake {bake} is older than the scene: run `mk build --bake` (engine = \"mk\" draws from it)")
        doc, blob = PL.load(bake)
    except (OSError, PL.BakeError) as e:
        raise UsageError(f"engine = \"mk\" draws from the scene's bake: {e}") from None
    return bake, doc, PL.plan(doc, blob, [(o.name, o.size) for o in outs], frames, percent)


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
    if args.percent:
        cfg["percent"] = args.percent
    if args.engine:
        cfg["engine"] = args.engine
    if args.samples:
        cfg["samples"] = args.samples
    if args.percent:
        cfg["percent"] = args.percent
    mk = cfg.get("engine") == "mk" and not args.no_styles
    if cfg.get("engine") == "mk":
        cfg["engine"] = None                               # Blender's frames: the scene's own engine
    plan = None if args.no_styles or args.no_transitions else cut_plan(proj)
    demands = TR.demands(plan, frames) if TR.has_effects(plan) else {}
    need_mb = 0.0
    for o in outs:
        d = frames_dir(proj, args.preset, o.name)
        todo = [f for f in frames if f"{f:05d}" not in on_disk(d)]
        scale = (max(o.size) / 1920.0) ** 2
        mb = EST_MB[100 if cfg["percent"] > 50 else 50]
        need_mb += len(todo) * mb * scale
        for _, item in TR.pending_items(d, demands):             # plates and bare frames are frames; a matte is flat
            need_mb += (mb if item["kind"] in ("plate", "bare") else 0.3 if item["kind"] == "matte" else 0.0) * scale
    free_mb = shutil.disk_usage(proj.root).free / 1e6
    if need_mb > free_mb - 1500:
        raise UsageError(f"not enough disk: about {need_mb:.0f} MB of frames, {free_mb:.0f} MB free (keeping 1.5 GB "
                         f"spare); render fewer frames or outputs, or free space")
    for o in outs:
        d = frames_dir(proj, args.preset, o.name)
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):                        # stale claims from a stopped render
            if p.stat().st_size == 0:
                p.unlink()
        TR.clear_claims(d)
    t0 = time.time()
    drawn, mk_frames, engine_run = {}, {o.name: [] for o in outs}, None
    if mk:
        from ..render import engine as EN
        bake, doc, drawn = mk_plan(proj, outs, frames, cfg["percent"])
        for it in drawn["items"]:
            mk_frames[it["output"]].append(it["frame"])
        have = {o.name: on_disk(frames_dir(proj, args.preset, o.name)) for o in outs}
        todo = [i for i, it in enumerate(drawn["items"]) if f"{it['frame']:05d}" not in have[it["output"]]]
        sub = dict(drawn, items=[drawn["items"][i] for i in todo])
        paths = [frames_dir(proj, args.preset, it["output"]) / f"{it['frame']:05d}.png" for it in sub["items"]]
        engine_run = {"result": None, "error": None}

        def draw():
            try:
                engine_run["result"] = EN.run(bake, sub, paths)
            except EN.EngineError as e:
                engine_run["error"] = str(e)
        engine_run["thread"] = threading.Thread(target=draw, daemon=True)
        engine_run["thread"].start()

    def blender(o):
        """The Blender jobs of one output (none when mk draws all it needs and it has no layers to draw)."""
        d = frames_dir(proj, args.preset, o.name)
        mine = set(mk_frames[o.name])
        left = [f for f in frames if f not in mine and f"{f:05d}" not in on_disk(d)]
        if mk and not left and not demands and not (doc.get("screen", 0) > 0 and mine):
            return []
        n, mk_list = max(1, args.jobs), sorted(mine)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = []
            for k in range(n):
                job = {"frames": frames, "out": str(d), "size": list(o.size), "aspect": o.name,
                       "styles": not args.no_styles, "demands": {str(f): items for f, items in demands.items()},
                       "mk": mk_list, "mk_layers": mk_list[k::n], **cfg}
                futs.append(ex.submit(bridge.run, "render_frames", job, proj.blend, proj, 3600 * 12, False))
                if k < n - 1:
                    time.sleep(8)                          # stagger loading
            return [f.result() for f in futs]
    if mk:                                                 # Blender only draws what mk leaves: every output at once
        with ThreadPoolExecutor(max_workers=len(outs)) as ex:
            results = dict(zip([o.name for o in outs], ex.map(blender, outs)))
    else:
        results = {o.name: blender(o) for o in outs}
    report = {}
    for o in outs:
        d, res = frames_dir(proj, args.preset, o.name), results[o.name]
        looks = {}
        for r in res:
            for k, v in r.get("looks", {}).items():
                looks[k] = looks.get(k, 0) + v
        report[o.name] = {"dir": str(d), "rendered": sum(r["rendered"] for r in res),
                          "s_per_frame": max((r["s_per_frame"] for r in res), default=0.0),
                          "aspect_bound": res[0]["aspect_bound"] if res else None}
        if looks:
            report[o.name]["looks"] = looks               # frames rendered in a silhouette / reflection look
        if mk:
            report[o.name]["engines"] = {"mk": len(mk_frames[o.name]), "blender": len(frames) - len(mk_frames[o.name])}
            report[o.name]["shots"] = drawn["shots"][o.name]
            report[o.name]["screen_layers"] = sum(r.get("screen_layers", 0) for r in res)
        if demands:
            report[o.name]["layers"] = {"drawn": sum(r.get("layers", 0) for r in res),
                                        "pending": len(TR.pending_items(d, demands))}
    out = {"preset": args.preset, "outputs": report}
    if engine_run is not None:
        engine_run["thread"].join()
        if engine_run["error"]:
            out.update(seconds=round(time.time() - t0, 1), error=f"mk's renderer: {engine_run['error']}")
            emit(out)
            return RUNTIME
        out["mk"] = engine_run["result"]
    for o in outs:
        report[o.name]["frames_on_disk"] = len(on_disk(frames_dir(proj, args.preset, o.name)))
    out["seconds"] = round(time.time() - t0, 1)
    emit(out)
    return 0
