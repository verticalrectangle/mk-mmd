"""mk post: grade rendered frames and encode each output to MP4 with the song."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import cutfx as CF
from .. import framefx as FF
from .. import master as MS
from .. import post as P
from ..core import palette as PAL
from ..core import screentype as SRT
from ..core import transition as TR
from .common import add_project_arg, cut_plan, emit, get_project, UsageError

HELP = """Grade the rendered frames of each output ([post] in mk.toml: contrast, split tone, the palette floor that keeps
shadows off black, halation, vignette, grain) and encode them to <project>/out/<name>_<output>[_<preset>].mp4 (the
preset is left out of the name for `final`) with the audio of [audio] (file, start = song seconds at clip time 0), as
H.264 (yuv420p, even size) and AAC with a 60 ms fade-out at the end. With [audio.master] the clip's audio is mastered for
small speakers first (mkmmd.master: the booms of a lane of [audio] hits get overtones a phone can play, a lifted attack
and the rest of the mix ducked under them; the sub is cut; the loudness is set under a limiter), and the report's
`audio_master` says how far the booms stand out through a phone's speakers before and after. Missing frames are an error
unless --allow-gaps (then the previous frame repeats). The report has each video's path, frames, size and the darkest luma
after the grade (`min_luma`: above 0 when the floor keeps shadows off black). A clip of one frame ([project] duration of
one frame) is a still: each output's graded frame is written as <project>/out/<name>_<output>[_<preset>].png, at its
full size and without sound. `[post] screen = "over"` lays the screen type over the finished, graded frame (its colours
exactly as written, no grain or vignette on it: type set on a photograph) instead of under the grade ("graded", the
default).

[[transition]], [[insert]] and [[glitch]] entries (docs/design.md: Shots: Transitions and inserts, Glitches) are
composited from the layers mk render drew next to the frames, before the grade: the grade, grain and vignette cover the
whole frame. A missing layer is an error unless --allow-gaps (then the plain cut shows there); --no-transitions leaves the
effects out. [[fx]] frame effects (lens, echo, slitscan, push, scope; docs/design.md: Shots: Frame effects) follow, in
the order of the table, reading the rendered frames around each one. Lens streaks ([post] streaks) are drawn from the
result's brightest lights; then screen type ([[text]] with
`screen`, docs/design.md: Text) is laid over it: <frames>/screen/<frame>.png, drawn by mk render.

Examples:
  mk post --preset draft
  mk post --preset final --to ~/share
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
    p.add_argument("--crf", type=int, default=15, help="x264 quality, lower is better (default 15)")
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


def run(args):
    proj = get_project(args, required=True)
    outs = [proj.output(n) for n in args.output] if args.output else proj.outputs
    look = proj.data.get("look", {})
    pal = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
    au = proj.data.get("audio", {})
    audio, mastering = None, None
    still = proj.last_frame == proj.frame0                 # a picture: one PNG per output, no sound
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
    od = proj.root / "out"
    od.mkdir(exist_ok=True)
    t0 = time.time()
    report = {}
    stems = FF.stem_source(proj.root, proj.mk_dir / "cache" / "timeline")
    for spec in (plan or {}).get("fx") or []:                  # a scope's stems are there before anything is encoded
        if spec["kind"] == "scope":
            try:
                stems(spec["timeline"], spec["stem"])
            except FileNotFoundError as e:
                raise UsageError(f"fx {spec['index']}: {e}") from None
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
        size = (first.shape[1], first.shape[0]) if still else (first.shape[1] // 2 * 2, first.shape[0] // 2 * 2)
        screen = proj.data.get("post", {}).get("screen", "graded")
        if screen not in ("graded", "over"):
            raise UsageError(f"[post] screen = {screen!r}: \"graded\" (the grade covers the screen type) or \"over\"")
        grade = P.Grade(proj.data.get("post", {}), pal, size)
        st_spec = proj.data.get("post", {}).get("streaks")
        try:
            streaks = P.Streaks({} if st_spec is True else dict(st_spec), pal, size) if st_spec else None
        except (ValueError, KeyError) as e:
            raise UsageError(str(e).strip('"')) from None
        fx = CF.CutFx(plan, src, size, o.name) if TR.has_effects(plan) else None
        if fx:
            lacking = fx.missing(frames)
            if lacking and not args.allow_gaps:
                raise UsageError(f"{o.name}: {len(lacking)} layer files of transitions / inserts / fx missing in {src} "
                                 f"(first {lacking[:3]}); mk render draws them, or pass --no-transitions or --allow-gaps")
            if not lacking:
                try:
                    fx.check(frames)
                except TR.TransitionError as e:
                    raise UsageError(str(e)) from None
        ffx = None
        if (plan or {}).get("fx"):
            ffx = FF.FrameFx(plan["fx"], size, fps=proj.fps, frame0=proj.frame0, stem_of=stems,
                             hits=FF.project_hits(proj.root, proj.data),
                             frame_of=lambda g, files=files, size=size: P.read(files[g], size) if g in files else None,
                             point_of=lambda key, g, src=src, size=size: CF.read_point(src, key, g, size))
        suffix = "" if args.preset == "final" else f"_{args.preset}"
        out = od / f"{proj.name}_{o.name}{suffix}.{'png' if still else 'mp4'}"
        ff = None if still else encode(frames, size, proj.fps, out, audio, args.crf)
        prev, prev_layer, lmin, bare, typed = None, None, 1.0, 0, 0
        for k, f in enumerate(frames):
            raw = P.read(files[f], size) if f in files else prev
            layer = P.read_rgba(src / SRT.layer_rel(f), size) if f in files else prev_layer
            img = raw
            if fx and fx.active(f):
                try:
                    img = fx.frame(f, raw)
                except FileNotFoundError:                         # layers missing (allowed): the plain cut shows
                    bare += 1
            if ffx and ffx.active(f):
                try:
                    img = ffx.frame(f, img)
                except FileNotFoundError:                         # a point or a frame missing (allowed): left out there
                    bare += 1
            if streaks is not None:                               # lens streaks: from the picture's lights, not the type
                img = streaks(img)
            if layer is not None and screen == "graded":         # screen type: over the cut and what its effects made of it
                img = SRT.composite(img, layer)
                typed += 1
            g = grade(img, k)
            if layer is not None and screen == "over":           # or over the finished picture, the grade under it
                g = SRT.composite(g, layer)
                typed += 1
            lmin = min(lmin, P.luma_min(g))
            if ff is None:
                Image.fromarray(P.to_u8(g)).save(out)
            else:
                ff.stdin.write(P.to_u8(g).tobytes())
            prev, prev_layer = raw, layer
        if ff is not None:
            ff.stdin.close()
            if ff.wait() != 0:
                raise UsageError(f"ffmpeg failed for {out}")
        report[o.name] = {"image" if still else "video": str(out), "frames": len(frames), "missing": len(missing),
                          "size": list(size), "min_luma": round(lmin, 4), "screen_type_frames": typed}
        if fx:
            report[o.name]["effects"] = {"transitions": len(plan["transitions"]), "inserts": len(plan["inserts"]),
                                         "glitches": len(plan["glitches"]), "fx": len(plan.get("fx") or []),
                                         "frames_without_layers": bare, "matte_scale": fx.notes}
        if args.to:
            dest = Path(args.to).expanduser() / out.name
            dest.write_bytes(out.read_bytes())
            report[o.name]["copied"] = str(dest)
    emit({"preset": args.preset, "seconds": round(time.time() - t0, 1), "audio": bool(audio), "audio_master": mastering,
          "outputs": report})
    return 0
