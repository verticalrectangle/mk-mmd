"""mk post: grade rendered frames and encode each output to MP4 with the song."""
import argparse
import subprocess
import time
from pathlib import Path

from .. import cutfx as CF
from .. import post as P
from ..core import palette as PAL
from ..core import transition as TR
from .common import add_project_arg, cut_plan, emit, get_project, UsageError

HELP = """Grade the rendered frames of each output ([post] in mk.toml: contrast, split tone, the palette floor that keeps
shadows off black, halation, vignette, grain) and encode them to <project>/out/<name>_<output>[_<preset>].mp4 with
the audio of [audio] (file, start = song seconds at clip time 0). Missing frames are an error unless --allow-gaps
(then the previous frame repeats).

[[transition]] and [[insert]] entries (docs/design.md: Shots: Transitions and inserts) are composited from the layers mk
render drew next to the frames, before the grade: the grade, grain and vignette cover the whole frame. A missing layer is
an error unless --allow-gaps (then the plain cut shows there); --no-transitions leaves the effects out.

Examples:
  mk post --preset draft
  mk post --preset final --to ~/Videos
"""


def add(sub):
    p = sub.add_parser("post", help="grade frames and encode MP4s with audio", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="draft", help="which render folder (draft, preview, final)")
    p.add_argument("--output", action="append", metavar="NAME")
    p.add_argument("--to", metavar="DIR", help="also copy the videos here")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--allow-gaps", action="store_true")
    p.add_argument("--no-transitions", action="store_true", help="composite no [[transition]] / [[insert]]: plain cuts")
    p.add_argument("--crf", type=int, default=15)
    add_project_arg(p)
    p.set_defaults(func=run)


def encode(frames, size, fps, out, audio=None, crf=15):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-ss", f"{audio['start']:.6f}", "-t", f"{audio['duration']:.6f}", "-i", audio["file"], "-map", "0:v",
                "-map", "1:a", "-af", f"afade=t=out:st={max(0.0, audio['duration'] - 0.06):.4f}:d=0.06",
                "-c:a", "aac", "-b:a", "320k"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-tune", "film",
            "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def run(args):
    proj = get_project(args, required=True)
    outs = [proj.output(n) for n in args.output] if args.output else proj.outputs
    look = proj.data.get("look", {})
    pal = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
    au = proj.data.get("audio", {})
    audio = None
    if not args.no_audio and au.get("file"):
        f = Path(au["file"]).expanduser()
        if not f.exists():
            raise UsageError(f"[audio] file {f} not found")
        audio = {"file": str(f), "start": float(au.get("start", 0.0)), "duration": proj.duration}
    frames = list(range(proj.frame0, proj.last_frame + 1))
    plan = None if args.no_transitions else cut_plan(proj)
    od = proj.root / "out"
    od.mkdir(exist_ok=True)
    t0 = time.time()
    report = {}
    for o in outs:
        src = proj.root / "renders" / args.preset / o.name
        files = {int(p.stem): p for p in src.glob("*.png") if p.stat().st_size > 0} if src.exists() else {}
        missing = [f for f in frames if f not in files]
        if missing and not args.allow_gaps:
            raise UsageError(f"{o.name}: {len(missing)} frames missing in {src} (first {missing[:5]}); render them or "
                             f"pass --allow-gaps")
        first = P.read(files[min(files)]) if files else None
        if first is None:
            raise UsageError(f"{o.name}: no frames in {src}")
        size = (first.shape[1] // 2 * 2, first.shape[0] // 2 * 2)
        grade = P.Grade(proj.data.get("post", {}), pal, size)
        fx = CF.CutFx(plan, src, size, o.name) if plan and (plan["transitions"] or plan["inserts"]) else None
        if fx:
            lacking = fx.missing(frames)
            if lacking and not args.allow_gaps:
                raise UsageError(f"{o.name}: {len(lacking)} layer files of transitions / inserts missing in {src} (first "
                                 f"{lacking[:3]}); mk render draws them, or pass --no-transitions or --allow-gaps")
            if not lacking:
                try:
                    fx.check(frames)
                except TR.TransitionError as e:
                    raise UsageError(str(e)) from None
        suffix = "" if args.preset == "final" else f"_{args.preset}"
        out = od / f"{proj.name}_{o.name}{suffix}.mp4"
        ff = encode(frames, size, proj.fps, out, audio, args.crf)
        prev, lmin, bare = None, 1.0, 0
        for k, f in enumerate(frames):
            raw = P.read(files[f], size) if f in files else prev
            img = raw
            if fx and fx.active(f):
                try:
                    img = fx.frame(f, raw)
                except FileNotFoundError:                         # layers missing (allowed): the plain cut shows
                    bare += 1
            g = grade(img, k)
            lmin = min(lmin, P.luma_min(g))
            ff.stdin.write(P.to_u8(g).tobytes())
            prev = raw
        ff.stdin.close()
        if ff.wait() != 0:
            raise UsageError(f"ffmpeg failed for {out}")
        report[o.name] = {"video": str(out), "frames": len(frames), "missing": len(missing), "size": list(size),
                          "min_luma": round(lmin, 4)}
        if fx:
            report[o.name]["effects"] = {"transitions": len(plan["transitions"]), "inserts": len(plan["inserts"]),
                                         "frames_without_layers": bare, "matte_scale": fx.notes}
        if args.to:
            dest = Path(args.to).expanduser() / out.name
            dest.write_bytes(out.read_bytes())
            report[o.name]["copied"] = str(dest)
    emit({"preset": args.preset, "seconds": round(time.time() - t0, 1), "audio": bool(audio), "outputs": report})
    return 0
