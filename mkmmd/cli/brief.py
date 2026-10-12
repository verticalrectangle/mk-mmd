"""mk brief: start a video from a song (`new`: the project and its Brief tab) and read what the person wants (`show`)."""
import argparse

from .. import brief as BR
from ..site import show as SHOW
from .common import UsageError, add_project_arg, emit, get_project

HELP = """A video starts from a brief (docs/design.md: Briefs): the part of the song, what the person wants for each line
or group of lines, and the video as a whole. `new` makes the project (the song copied into audio/, an mk.toml whose
clip is the whole song, an empty brief) and opens its Brief tab beside this pane: there the person chooses the part,
pastes its lyrics, times the words and writes; Send types the brief into the pane of the agent that opened the page.
`show` prints it with each part on the song and on the clip (song and clip seconds, the lines, words and bars it
covers), the stretches no part covers, the lyrics' line count and the timeline's numbers: numbers and the person's
words, never lyric text.

Examples:
  mk brief new ~/videos/blue --song ~/Music/blue.flac             # the project; its Brief tab opens beside this pane
  mk brief new ~/videos/blue --song blue.flac --lyrics blue.txt   # with the lyrics, one sung line per line
  mk brief show                                                   # this project's brief, as the agent reads it
  mk site --tab brief                                             # the Brief tab again
"""


def add(sub):
    p = sub.add_parser("brief", help="start a video from a song and read what the person wants (the Brief tab)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    n = s.add_parser("new", help="a project from a song, its Brief tab opened")
    n.add_argument("dir", help="the project's folder (made when missing; refused when it has an mk.toml)")
    n.add_argument("--song", required=True, help="the song (copied into audio/)")
    n.add_argument("--lyrics", help="its lyrics, one sung line per line (kept as audio/lyrics.txt, never printed)")
    n.add_argument("--name", help="the project's name (default: the folder's)")
    n.add_argument("--where", default="right", choices=["right", "down", "tab"],
                   help="where the page opens in Tern (default: right of this pane)")
    n.add_argument("--no-open", action="store_true", help="make the project, open no page")
    sh = s.add_parser("show", help="the brief as the agent reads it: numbers and the person's words, no lyrics")
    add_project_arg(sh)
    p.set_defaults(func=run)


def run(args):
    if args.action == "new":
        try:
            out = BR.new(args.dir, args.song, name=args.name, lyrics=args.lyrics)
        except BR.BriefError as e:
            raise UsageError(str(e))
        if not args.no_open:
            try:
                url = SHOW.page(project=out["project"]) + "&tab=brief"
                out.update(url=url, **SHOW.show(url, args.where))
            except (RuntimeError, OSError, ValueError) as e:
                out["not_opened"] = f"the page did not open ({e}): `mk site --tab brief` in the project opens it"
        emit(out)
        return 0
    proj = get_project(args, required=True)
    try:
        emit(BR.show(proj.root))
    except BR.BriefError as e:
        raise UsageError(str(e))
    return 0
