"""mk site: the project's page (its reviews, the music, the scene and its checks), served on this machine."""
import argparse

from ..site import server as S
from ..site import show as SHOW
from .common import UsageError, add_project_arg, emit, get_project

HELP = """The project's page, served by mk on this machine (docs/design.md: The site): in Tern beside this pane,
elsewhere in the browser. One server serves every page mk opens; it stops by itself after three hours without a request.

Examples:
  mk site                       # this project's page
  mk site --tab music           # its music: the audio, beats, hits, words, shots and effects
  mk site --tab scene           # the built scene played to the music, through the shot cameras
  mk site --where tab           # in a new Tern tab
  mk site --stop                # stop the server
"""


def add(sub):
    p = sub.add_parser("site", help="the project's page: reviews, music, scene and checks", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--where", default="right", choices=["right", "down", "tab"], help="where it opens in Tern (default: "
                                                                                       "right of this pane)")
    p.add_argument("--tab", choices=["reviews", "music", "scene"], help="the tab it opens on (default: reviews)")
    p.add_argument("--url", action="store_true", help="print the page's URL, open nothing")
    p.add_argument("--stop", action="store_true", help="stop the server")
    add_project_arg(p)
    p.set_defaults(func=run)


def run(args):
    if args.stop:
        emit({"stopped": S.stop()})
        return 0
    proj = get_project(args, required=True)
    try:
        url = SHOW.page(project=str(proj.root)) + (f"&tab={args.tab}" if args.tab else "")
        how = {} if args.url else SHOW.show(url, args.where)
    except (RuntimeError, OSError, ValueError) as e:
        raise UsageError(f"cannot open the site: {e}")
    emit({"project": str(proj.root), "url": url, **how})
    return 0
