"""mk timeline: analyse the song span of a clip (beats, downbeats, loudness, word timings) into a timeline file."""
import argparse
import json
from pathlib import Path

from .. import config as CFG
from .common import add_project_arg, emit, get_project, UsageError

HELP = """Analyse a song span into a timeline JSON (docs/design.md: Timeline): beats and downbeats (Demucs drum stem,
dynamic-programming beat tracker, constant grid when steady), per-frame vocal / mix / drum loudness, and word timings
(Whisper transcription or a lyrics file you provide, aligned on the vocal stem with wav2vec2 CTC and snapped to the
sung onsets). Lines are numbered from 1, words from 1 within a line; the printed summary never contains lyric text.

Needs the timeline extra (torch, demucs, faster-whisper). In a project, defaults come from [audio] (file, start) and
[project] (duration, fps), and the result goes to audio/timeline.json.

Examples:
  mk timeline analyze                                # the project's clip
  mk timeline analyze song.flac --start 12.0 --duration 40 --out /tmp/tl.json
  mk timeline analyze --lyrics lyrics.txt            # your own line breaks
  mk timeline show audio/timeline.json               # numbers only
"""


def add(sub):
    p = sub.add_parser("timeline", help="analyse a song span (beats, loudness, word timings)", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    a = s.add_parser("analyze", help="analyse audio into a timeline")
    a.add_argument("audio", nargs="?")
    a.add_argument("--start", type=float, help="song seconds at clip time 0")
    a.add_argument("--duration", type=float)
    a.add_argument("--fps", type=int)
    a.add_argument("--lyrics", help="text file, one sung line per line (never printed)")
    a.add_argument("--no-words", action="store_true")
    a.add_argument("--model", default="large-v3")
    a.add_argument("--language", default="en")
    a.add_argument("--out")
    add_project_arg(a)
    sh = s.add_parser("show", help="summarise a timeline (numbers only)")
    sh.add_argument("path", nargs="?")
    add_project_arg(sh)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args)
    if args.action == "show":
        path = Path(args.path) if args.path else (proj.root / "audio" / "timeline.json" if proj else None)
        if not path or not path.exists():
            raise UsageError("no timeline file")
        from ..timeline.analyze import summary
        emit(summary(json.loads(path.read_text(encoding="utf-8"))))
        return 0
    au = proj.data.get("audio", {}) if proj else {}
    audio = args.audio or au.get("file")
    if not audio:
        raise UsageError("give AUDIO or set [audio] file in mk.toml")
    audio = Path(audio).expanduser()
    if not audio.exists():
        raise UsageError(f"{audio} not found")
    start = args.start if args.start is not None else float(au.get("start", 0.0))
    duration = args.duration or (proj.duration if proj else None)
    if not duration:
        raise UsageError("give --duration (or run inside a project)")
    fps = args.fps or (int(proj.fps) if proj else 30)
    out = Path(args.out) if args.out else ((proj.root / "audio" / "timeline.json") if proj else None)
    if out is None:
        raise UsageError("give --out (or run inside a project)")
    from ..timeline.analyze import analyse, summary
    cache = (proj.mk_dir / "cache" / "timeline") if proj else CFG.cache_dir() / "timeline"
    tl = analyse(audio, start, duration, fps, str(cache), words=not args.no_words, lyrics=args.lyrics,
                 whisper_model=args.model, language=args.language, log=lambda *a: None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tl, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"out": str(out), **summary(tl)})
    return 0
