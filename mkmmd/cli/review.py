"""mk review: put decisions to a person, with pictures, choices, notes and 3D models, in Tern's review block."""
import argparse
import os
import shutil
import subprocess

from .. import review as RV
from .common import UsageError, emit

HELP = """Put decisions to a person: a review file lists questions, each with pictures, options to choose from and
models to turn in 3D. In Tern (with the mk plugin: `tern plugin link tern/` in this repository) it opens as a review
block; the person picks, writes notes, draws on the pictures in a whiteboard, and presses Send, which writes the
answers and posts them into the chat of the agent that opened it. The file format is in mkmmd/review.py and
docs/design.md (Reviews).

Examples:
  mk review check looks.review.toml           # every problem, one per line
  mk review open looks.review.toml            # beside this pane; Send posts the answers here
  mk review answers looks.review.toml         # the answers, and the marks in mm on lab sheets
"""


def add(sub):
    p = sub.add_parser("review", help="put decisions to a person: pictures, choices, notes and 3D models in Tern",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ss = p.add_subparsers(dest="review_command", metavar="ACTION", required=True)
    c = ss.add_parser("check", help="check a review file")
    c.add_argument("review", metavar="NAME.review.toml")
    c.set_defaults(func=run_check)
    s = ss.add_parser("show", help="the review as the block reads it: checked, paths absolute, answers so far (JSON)")
    s.add_argument("review", metavar="NAME.review.toml")
    s.set_defaults(func=run_show)
    o = ss.add_parser("open", help="open the review in Tern beside this pane; Send posts the answers to it")
    o.add_argument("review", metavar="NAME.review.toml")
    o.add_argument("--where", default="right", choices=["right", "down", "tab"], help="where it opens (default: right)")
    o.add_argument("--reply-to", type=int, metavar="PANE", help="the agent pane Send posts to (default: $TERN_PANE, "
                                                                "the pane this runs in)")
    o.set_defaults(func=run_open)
    a = ss.add_parser("answers", help="the answers so far, and what was drawn on the pictures (mm on lab sheets)")
    a.add_argument("review", metavar="NAME.review.toml")
    a.set_defaults(func=run_answers)


def _load(fn, path, *args):
    try:
        return fn(path, *args)
    except RV.ReviewError as e:
        raise UsageError(f"{path}:\n{e}")


def run_check(args):
    r = _load(RV.load, args.review)
    emit({"ok": True, "review": r["path"], "questions": len(r["questions"]),
          "images": sum(len(q["images"]) + sum(len(o["images"]) for o in q["options"]) for q in r["questions"]),
          "models": sum(len(q["models"]) + sum(len(o["models"]) for o in q["options"]) for q in r["questions"])})
    return 0


def run_show(args):
    emit(_load(RV.show, args.review))
    return 0


def run_open(args):
    pane = os.environ.get("TERN_PANE", "")
    reply = args.reply_to if args.reply_to is not None else (int(pane) if pane.isdigit() else None)
    answers = _load(RV.prepare, args.review, reply)
    tern = shutil.which("tern")
    if tern is None:
        raise UsageError("tern is not on PATH: open the review from Tern (its Files pane, or `tern open FILE` in a pane)")
    where = ["--tab"] if args.where == "tab" else ["--split", args.where]
    r = subprocess.run([tern, "open", *where, os.path.abspath(os.path.expanduser(args.review))], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise UsageError(f"tern open failed: {(r.stderr or r.stdout).strip()} (is the mk plugin loaded? `tern plugin "
                         f"link tern/` in mk-mmd)")
    emit({"review": os.path.abspath(args.review), "answers": str(answers), "reply_to": reply,
          "opened": r.stdout.strip()})
    return 0


def run_answers(args):
    emit(_load(RV.answers, args.review))
    return 0
