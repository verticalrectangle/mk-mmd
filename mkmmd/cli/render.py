"""mk render: render the cut of every output aspect to PNG frames (resumable, parallel, disk-guarded)."""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import bridge
from .. import gpulock as GL
from ..core import cast as CN
from ..core import keep as KP
from ..core import progress as PR
from ..core import screentype as SRT
from ..core import transition as TR
from .common import RUNTIME, add_project_arg, cut_plan, emit, get_project, parse_frames, UsageError

PRESETS = {
    "draft": {"percent": 50, "samples": 16, "motion_blur": False},
    "preview": {"percent": 50, "samples": 32, "motion_blur": True},
    "final": {"percent": 100, "samples": 64, "motion_blur": True},
}
ENGINES = ("eevee", "cycles", "workbench", "mk", "mkr")
OWN = ("mk", "mkr")                   # engines of mk's own: they draw what they can, Blender the rest and the layers
EST_MB = {50: 0.9, 100: 3.2}          # rough PNG size per 1080-class frame at that percentage

HELP = """Render the project's cut (shot cameras, per output) to PNG frames in
<project>/renders/<preset>/<output>/<frame>.png (<frame> is the Blender frame, five digits). Frames already on disk are
skipped, so a stopped render resumes; --jobs N runs N Blender processes that share the frames. Before starting, mk
checks that the disk can hold the frames.

Presets: draft (50 %, 16 samples, no motion blur), preview (50 %, 32 samples, motion blur), final (100 %, 64 samples,
motion blur; engine mkr draws them as 36 samples of twice the scene's shadow rays, as close to EEVEE's 64, faster). [render]
in mk.toml can override samples, shutter (motion blur, 0.35) and engine (eevee, cycles, workbench, mk); --samples,
--percent and --engine override them for one run.

Shots with a render-time look (`style = "silhouette"` or `"vector"`, `reflection = {...}` in [[shot]], see docs/design.md:
Shots) are rendered in it, inside the same Blender job: the frame on disk is the finished flat frame, not a pass.
--no-styles renders every shot as it is lit.

engine = "mk" (docs/design.md: Rendering) draws the vector look itself: its passes on the GPU in headless Chromium from the
scene's bake (`mk build` bakes it), finished by the same composer as Blender's. A shot it cannot draw (another look, type
over the picture, an object the bake cannot hold) renders in Blender with the scene's own engine, and so do the layers
below; the report names each shot's engine and why.

engine = "mkr" draws the lit frames themselves (what EEVEE draws) with mkr, mk-mmd's renderer in Rust on the GPU, from
the scene's bake (version 5), and the layers beside them: plates, bare frames, figures, points, keep masks and screen
type. Frames and layers in a silhouette or vector look (a shot's or a [[medium]] window's: mattes and backs too) are
drawn as the look's passes, which worker processes compose with the same composer as Blender's frames. `mkr info` says
which frames and layers it draws; the others (a reflection, a fisheye camera, something it cannot draw) render in
Blender, which starts only when something is left for it; the report counts the frames per engine and why, and the
layers left to Blender.

[[transition]], [[insert]] and [[glitch]] entries (docs/design.md: Shots: Transitions and inserts, Glitches) need more than
the cut's frames: the other shot's frames (plates), a silhouette's figure (mattes), a glitch's shot without its objects
(bare frames) and projected anchors. They are rendered next to the frames in <output>/{plate,matte,back,bare,point}/ on the
same terms (claimed, resumable, counted in the disk check); mk post composites them. --no-transitions leaves them out (and
--no-styles does too: a figure needs its silhouette look).

Screen type ([[text]] with `screen`, docs/design.md: Text: Screen type) is kept off the frames, in <output>/screen/<frame>.png
(RGBA, only for the frames that have any); mk post lays it over the cut and its effects. --no-styles leaves it out.

Before it draws, mk render writes each output folder's manifest (render.json: the files every frame gets, the frame,
its layers, its screen type and keep mask, when the engine can say; mkr's can) and marks it done when it stops, so `mk
post --follow` takes each frame as soon as its files are there. --post runs that alongside: `mk post --preset P
--follow` (with --post-args added to its own flags), each output's video finished while the render draws; its report
joins this one's.

Examples:
  mk render --preset draft                     # every output, the whole clip
  mk render --output 9x16 --frames t=0:5 --jobs 2
  mk render --preset final --engine mkr --post --post-args "--gpu --encoder vaapi"
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
    p.add_argument("--post", action="store_true", help="run `mk post --follow` alongside: the videos are finished as "
                   "the frames come")
    p.add_argument("--post-args", default="", metavar="ARGS", help="more flags for that mk post (one string, e.g. "
                   "\"--gpu --encoder vaapi\")")
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


def mkr_bake(proj):
    """The scene's bake for mkr, which must be as new as the scene: (path, bake.json)."""
    bake = proj.blend.with_suffix(".bake")
    try:
        if (bake / "bake.json").stat().st_mtime < proj.blend.stat().st_mtime:
            raise UsageError(f"the bake {bake} is older than the scene: run `mk build --bake` (engine = \"mkr\" draws "
                             f"from it)")
        doc = json.loads((bake / "bake.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise UsageError(f"engine = \"mkr\" draws from the scene's bake: {e}") from None
    if "lit" not in doc:
        raise UsageError(f"the bake {bake} is version {doc.get('version')}: engine = \"mkr\" needs version 5 "
                         f"(`mk build --bake`)")
    return bake, doc


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
    own = cfg.get("engine") if cfg.get("engine") in OWN and not args.no_styles else None
    if cfg.get("engine") in OWN:
        cfg["engine"] = None                               # Blender's frames: the scene's own engine
    plan = None if args.no_styles or args.no_transitions else cut_plan(proj)
    demands = TR.demands(plan, frames) if TR.has_effects(plan) else {}
    try:
        keep = KP.spec(proj.data.get("post"), proj.cast)
    except ValueError as e:
        raise UsageError(str(e)) from None
    if keep:                                               # the mask of where those materials are seen: keep/<frame>.png
        keep = {"armature": CN.armature_name(next(c for c in proj.cast if c.get("name") == keep["cast"])),
                "materials": keep["materials"]}
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
    drawn, own_frames, engine_run, doc = {}, {o.name: [] for o in outs}, None, {}
    promised = {o.name: None for o in outs}               # the files each output's frames get (None: not known)
    if own == "mk":
        from ..render import engine as EN
        bake, doc, drawn = mk_plan(proj, outs, frames, cfg["percent"])
        for it in drawn["items"]:
            own_frames[it["output"]].append(it["frame"])
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
    elif own == "mkr":
        from ..render import mkr as MKR
        from ..render import plan as PL
        bake, doc = mkr_bake(proj)
        blob = PL.load(bake)[1]
        try:
            req = MKR.request(doc, demands, keep, frames, outputs=[o.name for o in outs])
        except (ValueError, MKR.EngineError) as e:
            raise UsageError(str(e)) from None
        try:
            drawn = MKR.plan(bake, [o.name for o in outs], frames, req)
        except MKR.EngineError as e:
            raise UsageError(f"engine = \"mkr\": {e}") from None
        todo = {}
        for o in outs:
            d = frames_dir(proj, args.preset, o.name)
            own_frames[o.name] = drawn["outputs"][o.name]["draw"]
            promised[o.name] = promise(req, drawn["outputs"][o.name], frames)
            mine, drawn["outputs"][o.name]["theirs"] = MKR.split(req, drawn["outputs"][o.name], d, o.name)
            ids = MKR.look_ids(mine, o.name)
            looks = MKR.compose_jobs(doc, blob, o.name, mine["looks"]["outputs"][o.name], o.size, cfg["percent"], d,
                                     ids) if ids else None
            todo[o.name] = ([f for f in own_frames[o.name] if f"{f:05d}" not in on_disk(d)], mine, looks)
        engine_run = {"result": None, "error": None}
        # final frames at the preset's own samples: mkr's sampling for them (MKR.FINAL)
        sampling = MKR.sampling(args.preset, proj.data.get("render"), args.samples)

        def draw():
            try:
                engine_run["result"] = {n: MKR.draw(bake, n, fs, frames_dir(proj, args.preset, n), cfg, layers, looks=looks,
                                                    sampling=sampling)
                                        for n, (fs, layers, looks) in todo.items()}
            except MKR.EngineError as e:
                engine_run["error"] = str(e)
        engine_run["thread"] = threading.Thread(target=draw, daemon=True)

    def blender(o):
        """The Blender jobs of one output (none when mk's engine draws all it needs and every layer)."""
        d = frames_dir(proj, args.preset, o.name)
        mine = set(own_frames[o.name])
        left = [f for f in frames if f not in mine and f"{f:05d}" not in on_disk(d)]
        dem, extra = demands, {}
        if own == "mkr":                                   # Blender draws the layers mkr leaves, and its own frames'
            theirs = drawn["outputs"][o.name]["theirs"]
            dem = theirs["demands"]
            extra = {"mk_screen": theirs["mk_screen"], "mk_keep": theirs["mk_keep"]}
            covered = set(theirs["mk_keep"])
            no_keep = keep is not None and any(f not in covered and not (d / KP.rel(f)).exists() for f in frames)
            if not left and not dem and not theirs["screen"] and not no_keep:
                return []
        else:
            no_keep = keep is not None and any(not (d / KP.rel(f)).exists() for f in frames)
            if own and not left and not demands and not no_keep and not (doc.get("screen", 0) > 0 and mine):
                return []
        n, own_list = max(1, args.jobs), sorted(mine)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = []
            for k in range(n):
                job = {"frames": frames, "out": str(d), "size": list(o.size), "aspect": o.name,
                       "styles": not args.no_styles, "demands": {str(f): items for f, items in dem.items()},
                       "mk": own_list, "mk_layers": own_list[k::n], "keep": keep, **extra, **cfg}
                futs.append(ex.submit(bridge.run, "render_frames", job, proj.blend, proj, 3600 * 12, False))
                if k < n - 1:
                    time.sleep(8)                          # stagger loading
            return [f.result() for f in futs]
    with GL.held("mk render --post" if args.post else "mk render"):    # the post alongside runs inside this hold
        return finish_run(args, proj, outs, frames, t0, own, keep, demands, drawn, own_frames, engine_run, promised,
                          blender)


def finish_run(args, proj, outs, frames, t0, own, keep, demands, drawn, own_frames, engine_run, promised, blender):
    """The drawing of a render (mk's engine's thread `engine_run`, Blender's jobs `blender(o)`) under its manifests,
    with `mk post --follow` alongside when asked; its report emitted, its exit code."""
    for o in outs:                                         # what each frame gets, for mk post --follow
        PR.write(frames_dir(proj, args.preset, o.name), promised[o.name], frames)
    post = start_post(args, proj, outs) if args.post else None
    stopped = None
    try:
        if engine_run is not None:
            engine_run["thread"].start()
        if own:                                            # Blender only draws what mk's engine leaves: every output at once
            with ThreadPoolExecutor(max_workers=len(outs)) as ex:
                results = dict(zip([o.name for o in outs], ex.map(blender, outs)))
        else:
            results = {o.name: blender(o) for o in outs}
        if engine_run is not None:
            engine_run["thread"].join()
            stopped = engine_run["error"]
    except BaseException as e:
        stopped = f"{type(e).__name__}: {e}"
        if post is not None:
            post.terminate()
        raise
    finally:
        for o in outs:
            PR.finish(frames_dir(proj, args.preset, o.name), stopped)
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
        if own:
            report[o.name]["engines"] = {own: len(own_frames[o.name]), "blender": len(frames) - len(own_frames[o.name])}
            if own == "mk":
                report[o.name]["shots"] = drawn["shots"][o.name]
            else:
                why = {}
                for reason in drawn["outputs"][o.name]["blender"].values():
                    why[reason] = why.get(reason, 0) + 1
                report[o.name]["blender_because"] = why
                theirs = drawn["outputs"][o.name]["theirs"]
                report[o.name]["blender_jobs"] = len(res)
                report[o.name]["blender_layers"] = {"items": sum(len(v) for v in theirs["demands"].values()),
                                                    "screen": len(theirs["screen"])}
            report[o.name]["screen_layers"] = sum(r.get("screen_layers", 0) for r in res)
        if keep:
            report[o.name]["keep_layers"] = sum(r.get("keep_layers", 0) for r in res)
        if demands:
            report[o.name]["layers"] = {"drawn": sum(r.get("layers", 0) for r in res),
                                        "pending": len(TR.pending_items(d, demands))}
    out = {"preset": args.preset, "outputs": report}
    code = 0
    if engine_run is not None:
        if engine_run["error"]:
            name = "mk's renderer" if own == "mk" else "mkr"
            out["error"] = f"{name}: {engine_run['error']}"
            code = RUNTIME
        else:
            out[own] = engine_run["result"]
    for o in outs:
        report[o.name]["frames_on_disk"] = len(on_disk(frames_dir(proj, args.preset, o.name)))
    out["render_seconds"] = round(time.time() - t0, 1)
    if post is not None:                                   # the videos, finished as the frames came
        out["post"], ok = posted(post)
        code = code or (0 if ok else RUNTIME)
    out["seconds"] = round(time.time() - t0, 1)
    emit(out)
    return code


def promise(req, out_plan, frames):
    """{frame: [files relative to the output's folder]} an mkr render leaves for each of `frames`, mkr's and Blender's
    alike: the frame, its layers (`req`, MKR.request), its screen type where it has any (`out_plan`, MKR.plan's for the
    output), its negative layer where mkr draws negative type (the plan's "negative") and its keep mask."""
    scr = out_plan.get("screen") or {}
    screen = set(scr.get("draw") or ()) | set(scr.get("blender") or ())
    negative = set(scr.get("negative") or ())
    out = {}
    for f in frames:
        rels = [f"{f:05d}.png"]
        for item in req["items"].get(str(f), ()):
            rels += [r for r in TR.rel_paths(item, f) if r not in rels]
        if f in negative:
            rels.append(SRT.negative_rel(f))
        if f in screen:
            rels.append(SRT.layer_rel(f))
        if req.get("keep"):
            rels.append(KP.rel(f))
        out[f] = rels
    return out


def start_post(args, proj, outs):
    """`mk post --follow` for these outputs of this preset, started alongside the render (its report on its stdout)."""
    cmd = [sys.executable, "-m", "mkmmd.cli.main", "post", "--preset", args.preset, "--follow", "--project",
           str(proj.root)]
    for o in outs:
        cmd += ["--output", o.name]
    return subprocess.Popen(cmd + shlex.split(args.post_args), stdout=subprocess.PIPE, text=True)


def posted(post):
    """The report of the `mk post` that followed the render, once it is done, and whether it succeeded."""
    out, _ = post.communicate()
    try:
        rep = json.loads(out[out.index("{"):])
    except ValueError:
        rep = {"error": f"mk post printed no report: {out[-400:]!r}"}
    return rep, post.returncode == 0
