"""mk export: the project as files to put elsewhere. `mk export web`: its scene and music as a static page."""
import argparse
import re
from pathlib import Path

from .. import assets as A
from ..site import export as EX
from .assets import credits_text, project_credits
from .common import CHECK_FAILED, UsageError, add_project_arg, emit, get_project

HELP = """Export the project (docs/design.md: Exporting the page).

`web` writes the scene mk site shows (the bake played to the music timeline, through the shot cameras or orbiting
freely) as a folder of static files any web server can host: index.html, the page's scripts, data/ (the bake packed to a
fraction of its size, the scene's .glb with WebP textures, the music timeline, each output's video and its sound),
CREDITS.md and licences/ (each credited asset's readme files). The page shows the timeline and the output's video at
once and loads the 3D scene when asked. It runs in an iframe too: index.html?theme=light|dark|mono&view=3d|draft&maxh=PX,
and it posts its height and its clock to the host page (docs/design.md has the messages).

The scene's meshes and textures leave this machine with it, so every credited asset must have a reviewed licence and
none may forbid redistribution (`mk assets show SLUG --terms`, `mk assets set`); otherwise nothing is written (exit 1).

Examples:
  mk export web                                   # out/web/
  mk export web --out ~/site/mk-mmd/hello --outputs 9x16
  mk export web --video draft --side 960          # mk post's drafts, 960 px on the longer side
"""
NO_REDISTRIBUTION = re.compile(r"\bno redistribut|\bredistribution (is )?(not allowed|forbidden|prohibited)", re.I)


def add(sub):
    p = sub.add_parser("export", help="the project as files to put elsewhere (web: a static page of its scene)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    w = s.add_parser("web", help="the scene and music timeline as a static page", description=HELP,
                     formatter_class=argparse.RawDescriptionHelpFormatter)
    w.add_argument("--out", help="the folder to write (default: out/web in the project)")
    w.add_argument("--outputs", help="comma-separated outputs to export videos of (default: all)")
    w.add_argument("--video", choices=["final", "draft", "none"], default="final",
                   help="mk post's output (its draft when there is none), its draft, or no video (default: final)")
    w.add_argument("--side", type=int, default=1280, help="the videos' longer side in px, at most (default: 1280)")
    w.add_argument("--crf", type=int, default=23, help="the videos' H.264 quality, lower is better (default: 23)")
    w.add_argument("--texture-side", type=int, default=1024, help="the scene's textures' longer side, at most (default: 1024)")
    add_project_arg(w)
    p.set_defaults(func=run)


def run(args):
    proj = get_project(args, required=True)
    reg = A.Registry()
    slugs, lines = project_credits(proj)
    try:
        text, problems = credits_text(reg, slugs, lines)
        entries = [reg.get(s) for s in slugs]
    except A.AssetError as e:
        raise UsageError(str(e))
    problems += [f"{e['slug']}: its restrictions forbid redistribution ({e['restrictions']})" for e in entries
                 if NO_REDISTRIBUTION.search(e.get("restrictions") or "")]
    if problems:
        emit({"written": False, "problems": problems})
        return CHECK_FAILED
    licences = [(e["slug"], r) for e in entries for r in e.get("readmes") or []]
    missing = [r for _, r in licences if not Path(r).is_file()]
    if missing:
        emit({"written": False, "problems": [f"readme not found: {r}" for r in missing]})
        return CHECK_FAILED
    out = args.out or str(proj.root / "out" / "web")
    outputs = [o.strip() for o in args.outputs.split(",") if o.strip()] if args.outputs else None
    try:
        got = EX.export(proj, out, outputs=outputs, video=args.video, side=args.side, crf=args.crf,
                        tex_side=args.texture_side, credits=text + "\n", licences=licences)
    except EX.ExportError as e:
        raise UsageError(str(e))
    emit({"written": True, "out": got["out"], "bytes": got["bytes"], "scene_bytes": got["scene_bytes"],
          "videos": got["videos"], "assets": slugs, "notes": got["notes"]})
    return 0
