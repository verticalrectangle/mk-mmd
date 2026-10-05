"""mk play: open the project's encoded cut in the player of your choice (the `player` setting)."""
import argparse
import re
import shlex
import shutil
import subprocess
import sys

from .. import config as CFG
from .common import RUNTIME, add_project_arg, emit, get_project, UsageError

PRESETS = ("final", "preview", "draft")          # the default preset is the first of these that has videos

HELP = """Open the project's encoded videos (out/<name>_<output>[_<preset>].mp4, what mk post writes) in the player of your
choice: the `player` setting (~/.config/mk/config.toml, or MK_PLAYER), a command line run with the videos. Without it,
the system's opener (xdg-open, open) gets each video.

The command line may hold placeholders: {files} (the videos, one argument each; without it they go last), {chapters} (an
FFMETADATA file of the cut's shots as chapters, which mpv's --chapters-file and tern-video-block's --chapters read),
{frame0} (the project's number of the clip's first frame, as mk look --frames counts) and {fps}.

Players:
  player = "tern-video-block --split right --chapters {chapters} --first-frame {frame0}"
      Tern's video block (github.com/verticalrectangle/tern-video-block): every output side by side, beside this pane
  player = "mpv --chapters-file={chapters}"

Examples:
  mk play                          # the newest preset there is (final, else preview, else draft), every output
  mk play --preset draft --output 9x16
"""


def add(sub):
    p = sub.add_parser("play", help="open the encoded cut in the player of your choice (the player setting)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, help="which videos (default: final, else preview, else draft)")
    p.add_argument("--output", action="append", metavar="NAME", help="only these outputs (default: all)")
    add_project_arg(p)
    p.set_defaults(func=run)


def video_name(proj, output, preset):
    """The file `mk post` writes for an output and a preset: the preset is left out of the name for `final`."""
    return f"{proj.name}_{output}{'' if preset == 'final' else '_' + preset}.mp4"


def videos(proj):
    """{preset: [(output, path)]}: the encoded videos in out/, the outputs in the project's order."""
    found = {}
    for preset in PRESETS:
        for o in proj.outputs:
            p = proj.root / "out" / video_name(proj, o.name, preset)
            if p.is_file():
                found.setdefault(preset, []).append((o.name, p))
    return found


def chapters(proj):
    """FFMETADATA text with the shots in the cut (plates left out) as chapters, in time order, millisecond time base."""
    shots = sorted((s for s in proj.data.get("shot", []) if not s.get("plate") and "from" in s and "to" in s),
                   key=lambda s: float(s["from"]))
    lines = [";FFMETADATA1"]
    for s in shots:
        title = re.sub(r"([=;#\\\n])", r"\\\1", str(s["name"]))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(float(s['from']) * 1000)}",
                  f"END={round(float(s['to']) * 1000)}", f"title={title}"]
    return "\n".join(lines) + "\n"


def player_argv(template, files, chapters_path, frame0, fps):
    """The command line the `player` setting `template` (shell words) makes for the videos `files`: {files} becomes the
    videos, one argument each (without it they go last); {chapters}, {frame0} and {fps} are replaced in any word."""
    try:
        words = shlex.split(template)
    except ValueError as e:
        raise UsageError(f"the player setting is not a command line ({e}): {template}") from None
    if not words:
        raise UsageError("the player setting is empty")
    argv, placed = [], False
    for w in words:
        if w == "{files}":
            argv += [str(f) for f in files]
            placed = True
        else:
            argv.append(w.replace("{chapters}", str(chapters_path)).replace("{frame0}", str(frame0))
                        .replace("{fps}", f"{fps:g}"))
    return argv if placed else argv + [str(f) for f in files]


def run(args):
    proj = get_project(args, required=True)
    found = videos(proj)
    preset = args.preset or next((p for p in PRESETS if p in found), None)
    items = found.get(preset or "", [])
    if args.output:
        unknown = sorted(set(args.output) - {o.name for o in proj.outputs})
        if unknown:
            raise UsageError(f"no output {', '.join(unknown)} (have {', '.join(o.name for o in proj.outputs)})")
        items = [it for it in items if it[0] in args.output]
    if not items:
        raise UsageError(f"no encoded video in {proj.root / 'out'}" + (f" for preset {preset}" if preset else "")
                         + ": run mk render and mk post first")
    files = [p for _, p in items]
    template = CFG.load()["player"].strip()
    if template:
        chap = proj.mk_dir / "play" / "shots.ffmeta"
        if "{chapters}" in template:
            chap.parent.mkdir(parents=True, exist_ok=True)
            chap.write_text(chapters(proj), encoding="utf-8")
        commands = [player_argv(template, files, chap, proj.frame0, proj.fps)]
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        commands = [[opener, str(f)] for f in files]
    if shutil.which(commands[0][0]) is None:
        raise UsageError(f"{commands[0][0]} is not on the PATH (the player setting: ~/.config/mk/config.toml or MK_PLAYER)")
    codes = [subprocess.call(c) for c in commands]
    emit({"opened": [str(f) for f in files], "preset": preset, "player": commands, "exit": codes})
    return 0 if not any(codes) else RUNTIME
