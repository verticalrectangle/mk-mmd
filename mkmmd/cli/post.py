"""mk post: grade rendered frames and encode each output to MP4 with the song."""
import argparse
import contextlib
import functools
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config as CFG
from .. import footage as FT
from .. import framefx as FF
from .. import gpulock as GL
from .. import master as MS
from .. import memory as MEM
from .. import postframes as PF
from .. import postgpu as PG
from ..core import excerpt as EX
from ..core import framefx as FX
from ..core import keep as KP
from ..core import progress as PR
from ..gpufx import footage as GFT
from .common import add_project_arg, cut_breath, cut_plan, emit, get_project, UsageError

ENCODERS = ("x264", "vaapi")
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo")
SHM = "/dev/shm"            # mkr-post's hand-over folder, in memory where the system has it
RATES = "aformat=sample_rates=44100|48000"     # the song at 44.1 or 48 kHz, the nearer to its own: what players and
#                                                platforms take as AAC (a 96 kHz master is resampled, not passed on)

HELP = """Grade the rendered frames of each output ([post] in mk.toml: contrast, split tone, the palette floor that keeps
shadows off black, halation, vignette, grain) and encode them to <project>/out/<name>_<output>[_<preset>].mp4 (the
preset is left out of the name for `final`) with the audio of [audio] (file, start = song seconds at clip time 0), as
H.264 (yuv420p, even size) and AAC (at 44.1 or 48 kHz, the nearer to the file's own rate: a 96 kHz master is resampled)
with a 60 ms fade-out at the end. With [audio.master] the clip's audio is mastered for
small speakers first (mkmmd.master: the booms of a lane of [audio] hits get overtones a phone can play, a lifted attack
and the rest of the mix ducked under them; the sub is cut; the loudness is set under a limiter), and the report's
`audio_master` says how far the booms stand out through a phone's speakers before and after. Missing frames are an error
unless --allow-gaps (then the previous frame repeats). The report has each video's path, frames, size and the darkest luma
after the grade (`min_luma`: above 0 when the floor keeps shadows off black). A clip of one frame ([project] duration of
one frame) is a still: each output's graded frame is written as <project>/out/<name>_<output>[_<preset>].png, at its
full size and without sound. `[post] screen = "over"` lays the screen type over the finished, graded frame (its colours
exactly as written, no grain or vignette on it: type set on a photograph) instead of under the grade ("graded", the
default). `[post] keep = {cast, materials, saturation}` keeps those materials of a cast member in colour where they are
seen (her eyes in a black-and-white picture) through the mask mk render draws beside each frame (keep/<frame>.png): a
missing mask is an error unless --allow-gaps; inside a cut effect's window the frame is graded whole.

[[transition]], [[insert]] and [[glitch]] entries (docs/design.md: Shots: Transitions and inserts, Glitches) are
composited from the layers mk render drew next to the frames, before the grade: the grade, grain and vignette cover the
whole frame. A missing layer is an error unless --allow-gaps (then the plain cut shows there); --no-transitions leaves the
effects out. [[fx]] frame effects (lens, echo, slitscan, push, scope; docs/design.md: Shots: Frame effects) follow, in
the order of the table, reading the rendered frames around each one. Lens streaks ([post] streaks) are drawn from the
result's brightest lights; then screen type ([[text]] with
`screen`, docs/design.md: Text) is laid over it: <frames>/screen/<frame>.png, drawn by mk render. A page turn at the
clip's end, the loop's wrap, comes after all that: each frame of its window as finished turns over frame 0 as finished
(its own effects and screen type on), held still, and the grade covers the result; screen type laid over the graded
picture and the keep masks turn with the page. With [look] breathe every colour all that takes from the palette (the
effects', the streaks', the grade's floor, split toning and halation) is each frame's mix of the two palettes' colours
(docs/design.md: Post).

[[excerpt]] entries (name, from, to in clip seconds, outputs, loop) are spans written as videos of their own beside each
output, <project>/out/<name>_<output>_<excerpt>[_<preset>].mp4: the frames of the span as the encoder takes them (nothing
is finished twice), with the clip's sound over them, faded out at the end unless `loop = true` (a loop for a feed runs
back into its start).

The frames are finished by worker processes over chunks of consecutive frames and handed to the encoder in order: the
same frames whatever their number, --jobs or [post] jobs (default: up to 4, fewer when half the memory available would
not hold them at a worker's peak, about 1.7 GB at 1920 x 1080; one process for a little work). With --gpu (or [post]
gpu) mkr-post finishes what it can of each frame and writes it into the encoder itself, the workers (default up to 3)
only planning the frames ahead of it. The outputs are finished one after another, sharing the workers; a run that uses
the GPU (--gpu, or the vaapi encoder) holds the GPU lock (mkmmd.gpulock) while it runs. The encoder is [post] encoder
(or --encoder): "x264" (the default: libx264 at [post] x264_preset or --x264-preset, "slow" by default, --crf 15) or
"vaapi" (the GPU's H.264 encoder through VAAPI on the config's vaapi_device, constant QP --crf: many times faster,
larger files).

--follow takes the frames as a running `mk render` writes them: each frame once every file the render promised for it
(its manifest, render.json in the output's folder) is there. The render holds the GPU while it draws, so run them as
one job: `mk render --post` starts this itself, inside its own hold. A render that stops without a promised file is an
error.

Examples:
  mk post --preset draft
  mk post --preset final --to ~/share
  mk post --preset final --encoder vaapi --jobs 3
  mk post --preset final --gpu --encoder vaapi --follow
"""


def add(sub):
    p = sub.add_parser("post", help="grade frames and encode MP4s with audio", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="draft", help="which render folder under renders/ (draft, preview, final; "
                   "default draft)")
    p.add_argument("--output", action="append", metavar="NAME", help="output(s) to encode (default: all)")
    p.add_argument("--to", metavar="DIR", help="also copy the videos here")
    p.add_argument("--no-audio", action="store_true", help="encode without the song")
    p.add_argument("--allow-gaps", action="store_true", help="repeat the previous frame for a missing frame and show "
                   "the plain cut where an effect's layer is missing")
    p.add_argument("--no-transitions", action="store_true",
                   help="composite no [[transition]] / [[insert]] / [[glitch]]: plain cuts")
    p.add_argument("--crf", type=int, default=15, help="quality, lower is better: x264's CRF, VAAPI's constant QP "
                   "(default 15)")
    p.add_argument("--encoder", choices=ENCODERS, help="x264 (default, or [post] encoder) or vaapi (the GPU's H.264)")
    p.add_argument("--x264-preset", choices=X264_PRESETS, help="x264's preset (default [post] x264_preset, else slow)")
    p.add_argument("--jobs", type=int, metavar="N", help="processes that finish the frames of an output (default [post] "
                   "jobs, else up to 4 as the memory available allows; up to 3 with --gpu, which only plan them)")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                   help="finish what mkr-post can of each frame on the GPU, within a level of the CPU's (default "
                   "[post] gpu, else off)")
    p.add_argument("--follow", action="store_true", help="take the frames as a running mk render writes them (its "
                   "manifest in each output's folder)")
    add_project_arg(p)
    p.set_defaults(func=run)


def encode(frames, size, fps, out, audio=None, crf=15, encoder="x264", preset="slow", excerpts=()):
    """ffmpeg reading rgb24 frames of `size` on its stdin and writing `out`: H.264 (yuv420p) by libx264 at `preset` and
    CRF `crf`, or by the GPU through VAAPI at constant QP `crf`, with the song's span `audio` (AAC at one of RATES,
    faded out). Each of `excerpts` (mkmmd.core.excerpt's entries, with the file each goes `out` to) is written beside
    it as the frames pass: its span cut out of the same stream and encoded alike."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if encoder == "vaapi":
        cmd += ["-vaapi_device", CFG.load()["vaapi_device"]]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-ss", f"{audio['start']:.6f}", "-t", f"{audio['duration']:.6f}", "-i", audio["file"]]

    def one(path, picture=None, sound=None):
        args = []
        if audio:
            args += ["-map", "0:v", "-map", "1:a", "-af", f"{sound},{RATES}", "-c:a", "aac", "-b:a", "320k"]
        chain = ([picture] if picture else []) + (["format=nv12", "hwupload"] if encoder == "vaapi" else [])
        if chain:
            args += ["-vf", ",".join(chain)]
        if encoder == "vaapi":
            args += ["-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", str(crf)]
        else:
            args += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-tune", "film"]
        return args + ["-movflags", "+faststart", str(path)]
    cmd += one(out, sound=f"afade=t=out:st={max(0.0, audio['duration'] - 0.06):.4f}:d=0.06" if audio else None)
    for e in excerpts:
        cmd += one(e["out"], *EX.filters(e, fps))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mastered(proj, au, f, start, duration):
    """The clip's audio mastered by [audio.master] into a WAV in the project's cache: (path, report)."""
    if not au.get("hits"):
        raise UsageError("[audio.master] needs [audio] hits, the file whose lane names the booms")
    hp = proj.root / Path(au["hits"]).expanduser()
    if not hp.is_file():
        raise UsageError(f"[audio] hits {hp} not found")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(f), "-ac", "2",
                          "-ar", str(MS.SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(float).reshape(-1, 2)
    try:
        y, rep = MS.master(x, MS.SR, json.loads(hp.read_text(encoding="utf-8")), au["master"])
    except MS.MasterError as e:
        raise UsageError(str(e)) from None
    out = proj.mk_dir / "cache" / "post" / "master.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(MS.SR), "-ac", "2", "-i", "-", "-c:a",
                    "pcm_f32le", str(out)], input=y.astype(np.float32).tobytes(), check=True)
    return out, rep


def encoding(args, post):
    """(encoder, x264 preset) from the flags, else [post] encoder and x264_preset; a UsageError for an unknown one."""
    enc = args.encoder or post.get("encoder", "x264")
    if enc not in ENCODERS:
        raise UsageError(f"[post] encoder = {enc!r}: one of {', '.join(map(repr, ENCODERS))}")
    preset = args.x264_preset or post.get("x264_preset", "slow")
    if preset not in X264_PRESETS:
        raise UsageError(f"[post] x264_preset = {preset!r}: one of {', '.join(X264_PRESETS)}")
    device = CFG.load()["vaapi_device"]
    if enc == "vaapi" and not os.path.exists(device):
        raise UsageError(f"encoder vaapi: no device {device} (the config's vaapi_device, MK_VAAPI_DEVICE)")
    return enc, preset


def run(args):
    proj = get_project(args, required=True)
    outs = [proj.output(n) for n in args.output] if args.output else proj.outputs
    post = proj.data.get("post", {})
    au = proj.data.get("audio", {})
    audio, mastering = None, None
    still = proj.last_frame == proj.frame0                 # a picture: one PNG per output, no sound
    try:
        keep = KP.spec(proj.data.get("post"), proj.cast)
    except ValueError as e:
        raise UsageError(str(e)) from None
    jobs = args.jobs if args.jobs is not None else post.get("jobs")
    if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1):
        raise UsageError(f"[post] jobs / --jobs = {jobs!r}: a whole number of processes, 1 or more")
    encoder, x264_preset = encoding(args, post)
    try:
        excerpts = [] if still else EX.normalize(proj.data.get("excerpt"), proj.duration, proj.fps,
                                                 [o.name for o in proj.outputs])
    except EX.ExcerptError as e:
        raise UsageError(str(e)) from None
    gpu = bool(args.gpu if args.gpu is not None else post.get("gpu", False))
    if not args.no_audio and au.get("file") and not still:
        f = Path(au["file"]).expanduser()
        if not f.exists():
            raise UsageError(f"[audio] file {f} not found")
        audio = {"file": str(f), "start": float(au.get("start", 0.0)), "duration": proj.duration}
        if au.get("master") is not None:
            wav, mastering = mastered(proj, au, f, audio["start"], proj.duration)
            audio = {"file": str(wav), "start": 0.0, "duration": proj.duration}
    frames = list(range(proj.frame0, proj.last_frame + 1))
    plan = None if args.no_transitions else cut_plan(proj)
    breath = cut_breath(proj, plan)                            # [look] breathe: each frame's colours, its mix's
    od = proj.root / "out"
    od.mkdir(exist_ok=True)
    t0 = time.time()
    stems = FF.stem_source(proj.root, proj.mk_dir / "cache" / "timeline")
    for spec in (plan or {}).get("fx") or []:                  # a scope's stems are there before anything is encoded
        if spec["kind"] == "scope":
            try:
                stems(spec["timeline"], spec["stem"])
            except FileNotFoundError as e:
                raise UsageError(f"fx {spec['index']}: {e}") from None
    try:                                                       # and every footage clip opens and plays where it starts
        FF.check_footage((plan or {}).get("fx") or [], FT.source(proj.root))
    except FT.FootageError as e:
        raise UsageError(str(e)) from None
    todo = [frames_of(proj, o, args, frames, keep, still, od, excerpts) for o in outs]
    s = {"plan": plan, "breath": breath, "keep": keep, "frames": frames, "audio": audio, "crf": args.crf,
         "encoder": encoder, "x264_preset": x264_preset, "gpu": gpu, "jobs": jobs, "allow_gaps": args.allow_gaps,
         "still": still}
    post_cache = proj.mk_dir / "cache" / "post"
    MEM.sweep("mkpost", [SHM, tempfile.gettempdir(), post_cache])   # what runs killed outright left in memory, on disk
    pools = {}                                                 # worker processes, shared by the outputs in turn
    on_gpu = (gpu or encoder == "vaapi") and not still
    kept = clip_frames(post_cache, plan, todo) if gpu and not still else None
    encoders = []                                              # (encoder, video) still writing their last frames
    ready = Ahead(prepare, proj, todo[0], s)                   # the first output's setup while the GPU is waited for
    try:
        with GL.held(f"mk post --preset {args.preset}") if on_gpu else contextlib.nullcontext():
            report = {}
            for i, t in enumerate(todo):
                got = ready.result()
                ready = Ahead(prepare, proj, todo[i + 1], s) if i + 1 < len(todo) else None   # the next one's meanwhile
                report[t["name"]], ff = finish(proj, t, s, pools, kept, got)
                if ff is not None:
                    encoders.append((ff, t["out"]))
            while encoders:
                ff, video = encoders.pop(0)
                if ff.wait() != 0:
                    raise UsageError(f"ffmpeg failed for {video}")
    finally:
        for ff, _ in encoders:                                 # an error: the videos before it end as they are
            ff.wait()
        for pool in pools.values():
            pool.close()
        if kept is not None:
            kept.close()
    if args.to:
        for t in todo:
            dest = Path(args.to).expanduser() / t["out"].name
            dest.write_bytes(t["out"].read_bytes())
            for e in t["excerpts"]:
                (dest.parent / e["out"].name).write_bytes(e["out"].read_bytes())
            report[t["name"]]["copied"] = str(dest)
    emit({"preset": args.preset, "seconds": round(time.time() - t0, 1), "audio": bool(audio), "audio_master": mastering,
          "outputs": report})
    return 0


def frames_of(proj, o, args, frames, keep, still, od, excerpts=()):
    """What output `o` is made from, checked: {name, src (its frame folder), files {frame: PNG}, missing (frames not
    there), out (the video or still), excerpts (those of `excerpts` cut out of it, each with its own `out`), follow (a
    mkmmd.core.progress.Follow with --follow)}. With --follow the frames a running render promised count as there."""
    src = proj.root / "renders" / args.preset / o.name
    files = {int(p.stem): p for p in src.glob("*.png") if p.stat().st_size > 0} if src.exists() else {}
    follow = None
    if args.follow:
        follow = PR.Follow(src, frames)
        if follow.promised is not None:
            files.update({f: src / f"{f:05d}.png" for f in frames if f in follow.promised})
    missing = [f for f in frames if f not in files]
    if missing and not args.allow_gaps:
        raise UsageError(f"{o.name}: {len(missing)} frames missing in {src} (first {missing[:5]}); render them or "
                         f"pass --allow-gaps")
    if keep and follow is None:
        masks = [f for f in frames if f in files and not (src / KP.rel(f)).exists()]
        if masks and not args.allow_gaps:
            raise UsageError(f"{o.name}: {len(masks)} keep masks missing in {src / KP.LAYER_DIR} (first {masks[:5]}): "
                             f"mk render draws them for [post] keep; or pass --allow-gaps")
    if not files:
        raise UsageError(f"{o.name}: no frames in {src}")
    suffix = "" if args.preset == "final" else f"_{args.preset}"
    return {"name": o.name, "src": src, "files": files, "missing": len(missing), "follow": follow,
            "out": od / f"{proj.name}_{o.name}{suffix}.{'png' if still else 'mp4'}",
            "excerpts": [{**e, "out": od / f"{proj.name}_{o.name}_{e['name']}{suffix}.mp4"}
                         for e in excerpts if o.name in e["outputs"]]}


def size_of(t, still):
    """The size (w, h) output `t` (frames_of) is graded at: its first frame's, even for a video; following a render,
    once the render has that frame."""
    first = min(t["files"])
    if t["follow"] is not None:
        try:
            t["follow"].wait(first)
        except FileNotFoundError as e:
            raise UsageError(f"{t['name']}: {e}") from None
    with Image.open(t["files"][first]) as im:
        w, h = im.size
    return (w, h) if still else (w // 2 * 2, h // 2 * 2)


def size_now(t):
    """The size output `t` (frames_of) of a video is graded at when its first frame is on disk now (size_of without
    waiting), else None (a followed render that has not drawn it yet)."""
    try:
        with Image.open(t["files"][min(t["files"])]) as im:
            w, h = im.size
    except (OSError, ValueError):
        return None
    return w // 2 * 2, h // 2 * 2


def clip_frames(folder, plan, todo):
    """The footage clips' frames the outputs `todo` share on the GPU way (mkmmd.gpufx.footage.Kept: each decoded once a
    run and fitted for every output that shows it), kept in a folder of their own under `folder` (the project's cache,
    on disk) until the run ends; None for a single output or no footage."""
    fxs = (plan or {}).get("fx") or []
    if len(todo) < 2 or not any(e["kind"] == "footage" for fx in fxs for e in [fx, *(fx.get("aspect") or {}).values()]):
        return None
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [(t["name"], functools.partial(size_now, t),
                {e["index"]: e for e in (FX.for_output(fx, t["name"]) for fx in fxs) if e["kind"] == "footage"})
               for t in todo]
    return GFT.Kept(tempfile.mkdtemp(prefix=MEM.scratch_prefix("mkpost"), dir=folder), outputs)


class Ahead:
    """`fn(*args)` run on a thread of its own; `result()` waits for it and gives what it returned, or raises what it
    raised."""

    def __init__(self, fn, *args):
        self.value, self.error = None, None
        self.thread = threading.Thread(target=self._run, args=(fn, args), name="mkpost-ahead", daemon=True)
        self.thread.start()

    def _run(self, fn, args):
        try:
            self.value = fn(*args)
        except BaseException as e:                              # noqa: BLE001 (raised by result)
            self.error = e

    def result(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.value


def prepare(proj, t, s):
    """What output `t` (frames_of) is finished with, worked out before its turn (while the output before it is
    finished): {size, job (its PF.Output), why_cpu (why mkr-post does not finish it, None when it does), jobs}."""
    plan, frames, follow, still = s["plan"], s["frames"], t["follow"], s["still"]
    size = size_of(t, still)
    try:
        job = PF.Output(proj, plan, t["name"], t["src"], t["files"], frames, size, s["keep"], verify=True,
                        allow_gaps=s["allow_gaps"], follow=follow, breath=s["breath"])
    except PF.SetupError as e:
        raise UsageError(str(e)) from None
    why_cpu = None if s["gpu"] else "not asked (--gpu, [post] gpu)"
    if s["gpu"] and not still:
        why_cpu = PG.unusable(job, files=follow is None)
    n_jobs = s["jobs"] or PF.jobs(len(frames), size)
    if not s["jobs"] and why_cpu is None:                     # mkr-post finishes them: the workers only plan
        n_jobs = min(n_jobs, PF.GPU_JOBS)
    return {"size": size, "job": job, "why_cpu": why_cpu, "jobs": n_jobs}


def finish(proj, t, s, pools, clips, ready):
    """Output `t` (frames_of) finished and encoded, or its still written, with the settings `s` and what `prepare`
    worked out for it (`ready`): (its report, its encoder still writing its last frames, None for a still). Worker
    processes come from `pools` ({count: PF.Planners}, shared by the outputs), made there the first time; the footage
    clips' frames from `clips` (clip_frames, or None) where an output before this one decoded them."""
    t0 = time.time()
    if clips is not None:
        clips.begin(t["name"])
    plan, frames, out, follow, still = s["plan"], s["frames"], t["out"], t["follow"], s["still"]
    waited = follow.waited if follow is not None else 0.0  # what its setup waited, the output before it running
    size, job, why_cpu, n_jobs = ready["size"], ready["job"], ready["why_cpu"], ready["jobs"]
    pool = None
    if n_jobs > 1 and len(frames) > PF.CHUNK:
        pool = pools.get(n_jobs) or pools.setdefault(n_jobs, PF.Planners(n_jobs))
    ff = None if still else encode(frames, size, proj.fps, out, s["audio"], s["crf"], s["encoder"], s["x264_preset"],
                                   t["excerpts"])
    server = None
    if why_cpu is None and not still:
        scratch = tempfile.mkdtemp(prefix=MEM.scratch_prefix("mkpost"), dir=SHM if os.path.isdir(SHM) else None)
        job.gpu, job.args["gpu"] = Path(scratch), scratch
        try:
            server = PG.Server(job, scratch, sink=ff.stdin)      # mkr-post writes the frames into the encoder
        except BaseException:
            ff.kill()
            raise
    lmin, bare, typed, kept = 1.0, 0, 0, 0
    done = PF.finished(job, n_jobs, server=server, pool=pool, kept=clips)
    try:
        for _k, data, (luma, n_typed, n_kept, n_bare) in done:
            lmin, typed, kept, bare = min(lmin, luma), typed + n_typed, kept + n_kept, bare + n_bare
            if ff is None:
                Image.frombytes("RGB", size, data).save(out)
            elif data is not None:
                ff.stdin.write(data)
    except (FT.FootageError, FX.FxError) as e:              # a clip that fails, a pivot with no figure
        if ff is not None:
            ff.kill()
        raise UsageError(str(e)) from None
    except BrokenPipeError:                                 # the encoder stopped: its own error is above
        raise UsageError(f"ffmpeg failed for {out}") from None
    except BrokenProcessPool:                               # a worker died under it (the memory ran out?)
        if ff is not None:
            ff.kill()
        raise UsageError(f"{t['name']}: a worker process died (out of memory?): fewer --jobs (this run: {n_jobs})") \
            from None
    except FileNotFoundError as e:                          # --follow: the render stopped without a file it promised
        if ff is not None:
            ff.kill()
        raise UsageError(f"{t['name']}: {e}") from None
    except RuntimeError as e:                               # mkr-post stopped (or the encoder under it)
        if ff is not None:
            try:
                failed = ff.wait(timeout=2) != 0
            except subprocess.TimeoutExpired:
                failed = False
            ff.kill()
            if failed:
                raise UsageError(f"ffmpeg failed for {out}") from None
        raise UsageError(f"{t['name']}: {e}") from None
    finally:
        done.close()                                        # the workers stop here, whatever happened
        if server is not None:
            server.close()
    if ff is not None:
        ff.stdin.close()                                    # its last frames encoded while the next output starts
    rep = {"image" if still else "video": str(out), "frames": len(frames), "missing": t["missing"], "size": list(size),
           "min_luma": round(lmin, 4), "screen_type_frames": typed}
    if not still:
        rep["encoder"] = s["encoder"] if s["encoder"] == "vaapi" else f"x264 {s['x264_preset']}"
        rep["jobs"] = n_jobs
        rep["gpu"] = True if why_cpu is None else why_cpu
        rep["seconds"] = round(time.time() - t0, 1)
        if follow is not None:
            rep["waited_for_render"] = round(follow.waited - waited, 1)
        if t["excerpts"]:
            rep["excerpts"] = {e["name"]: {"video": str(e["out"]), "frames": e["frames"],
                                           "seconds": round(e["frames"] / proj.fps, 4)} for e in t["excerpts"]}
    if s["keep"]:
        rep["keep_frames"] = kept                          # graded with the colour kept (not inside a cut effect)
    if job.fx:
        rep["effects"] = {"transitions": len(plan["transitions"]), "inserts": len(plan["inserts"]),
                          "glitches": len(plan["glitches"]), "fx": len(plan.get("fx") or []),
                          "frames_without_layers": bare, "matte_scale": job.notes}
    return rep, ff
