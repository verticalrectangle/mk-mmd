"""mk timeline: analyse the song span of a clip (beats, downbeats, loudness, note onsets, word timings) into a timeline
file, and recompute note onsets into an existing one."""
import argparse
import json
from pathlib import Path

from .. import config as CFG
from .common import add_project_arg, emit, get_project, UsageError

HELP = """Analyse a song span into a timeline JSON (docs/design.md: Timeline): beats and downbeats (Demucs drum stem,
dynamic-programming beat tracker, constant grid when steady), per-frame vocal / mix / drum loudness, note onsets (when
each pluck or strum of the `other` stem starts, in clip seconds), and word timings (Whisper transcription or a lyrics
file you provide, aligned on the vocal stem with wav2vec2 CTC and snapped to the sung onsets). Lines are numbered from
1, words from 1 within a line; the printed summary never contains lyric text.

Needs the timeline extra (torch, demucs, faster-whisper). In a project, defaults come from [audio] (file, start) and
[project] (duration, fps), and the result goes to audio/timeline.json.

`onsets` recomputes the note onsets of a stem into an EXISTING timeline from the cached Demucs stems: the song span is
read again the way `analyze` read it (the file's audio block; pass the same --context, default 20), so the stem cache
hits. It sets `onsets` and `onsets_meta` for that stem and nothing else in the file, and prints numbers only: the
count, onsets per beat and how they sit on the 8th-note grid of the beats (median offset and spread in ms, empty 8ths,
onsets between 8ths). --lo/--hi (Hz) is the band whose energy rises are the onsets (default 800-6000: pick noise and
string harmonics).

Examples:
  mk timeline analyze                                # the project's clip
  mk timeline analyze song.flac --start 12.0 --duration 40 --out /tmp/tl.json
  mk timeline analyze --lyrics lyrics.txt            # your own line breaks
  mk timeline show audio/timeline.json               # numbers only
  mk timeline onsets                                 # the project's timeline: stem other, 800-6000 Hz
  mk timeline onsets --timeline /tmp/tl.json --stem other --lo 1000 --hi 5000
"""
STEMS = ("drums", "bass", "other", "vocals")


def add(sub):
    p = sub.add_parser("timeline", help="analyse a song span (beats, loudness, note onsets, word timings)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    a = s.add_parser("analyze", help="analyse audio into a timeline")
    a.add_argument("audio", nargs="?", help="the song file (default: [audio] file of mk.toml)")
    a.add_argument("--start", type=float, help="song seconds at clip time 0 (default: [audio] start, else 0)")
    a.add_argument("--duration", type=float, help="clip seconds to analyse (default: the project's duration)")
    a.add_argument("--fps", type=int, help="frames per second of the per-frame loudness series (default: the "
                   "project's, else 30)")
    a.add_argument("--lyrics", help="text file, one sung line per line (never printed)")
    a.add_argument("--no-words", action="store_true", help="beats and loudness only: no transcription, no word timings")
    a.add_argument("--model", default="large-v3", help="faster-whisper model for the transcription (default large-v3)")
    a.add_argument("--language", default="en", help="language of the words (default en)")
    a.add_argument("--context", type=float, default=20.0,
                   help="seconds of song analysed on each side of the clip for tempo, bar phase and lines (default 20)")
    a.add_argument("--out", help="timeline JSON to write (default: <project>/audio/timeline.json)")
    add_project_arg(a)
    sh = s.add_parser("show", help="summarise a timeline (numbers only)")
    sh.add_argument("path", nargs="?", help="timeline JSON (default: the project's audio/timeline.json)")
    add_project_arg(sh)
    on = s.add_parser("onsets", help="recompute note onsets of a stem into an existing timeline")
    on.add_argument("--stem", action="append", choices=STEMS, help="Demucs stem, repeatable (default other)")
    on.add_argument("--lo", type=float, help="band floor, Hz (default 800)")
    on.add_argument("--hi", type=float, help="band ceiling, Hz (default 6000)")
    on.add_argument("--timeline", help="timeline JSON (default: the project's audio/timeline.json)")
    on.add_argument("--context", type=float, default=20.0,
                    help="the --context given to analyze: the stem cache is keyed by the song span (default 20)")
    add_project_arg(on)
    p.set_defaults(func=run)


def onsets(args, proj):
    """Set onsets / onsets_meta of the chosen stems in an existing timeline file; every other key stays as it was."""
    path = args.timeline or (proj.root / "audio" / "timeline.json" if proj else None)
    path = Path(path).expanduser() if path else None
    if not path or not path.exists():
        raise UsageError("no timeline file: pass --timeline (or run inside a project with audio/timeline.json)")
    tl = json.loads(path.read_text(encoding="utf-8"))
    audio = (tl.get("audio") or {}).get("file")
    if not audio:
        raise UsageError("the timeline has no audio block: run `mk timeline analyze`")
    if not Path(audio).expanduser().exists():
        raise UsageError(f"{audio} (the audio the timeline was analysed from) not found")
    from ..timeline import analyze as A
    from ..timeline import signal as S
    lo = S.ONSET_LO if args.lo is None else args.lo
    hi = S.ONSET_HI if args.hi is None else args.hi
    cache = (proj.mk_dir / "cache" / "timeline") if proj else CFG.cache_dir() / "timeline"
    try:
        new, meta = A.timeline_onsets(tl, str(cache), tuple(args.stem or A.ONSET_STEMS), lo, hi, args.context,
                                      log=lambda *a: None)
    except ValueError as e:
        raise UsageError(str(e))
    for key, add in (("onsets", new), ("onsets_meta", meta)):          # per stem: other stems' onsets stay
        tl[key] = {**(tl[key] if isinstance(tl.get(key), dict) else {}), **add}
    path.write_text(json.dumps(tl, ensure_ascii=False, indent=1), encoding="utf-8")
    beats = tl.get("beats", [])
    emit({"timeline": str(path),
          "onsets": {k: {**A.onset_stats(v, beats), "band": meta[k]["band"], "latency_ms": meta[k]["latency_ms"]}
                     for k, v in new.items()}})
    return 0


def run(args):
    proj = get_project(args)
    if args.action == "show":
        path = Path(args.path) if args.path else (proj.root / "audio" / "timeline.json" if proj else None)
        if not path or not path.exists():
            raise UsageError("no timeline file")
        from ..timeline.analyze import summary
        emit(summary(json.loads(path.read_text(encoding="utf-8"))))
        return 0
    if args.action == "onsets":
        return onsets(args, proj)
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
                 whisper_model=args.model, language=args.language, context=args.context, log=lambda *a: None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tl, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"out": str(out), **summary(tl)})
    return 0
