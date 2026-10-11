"""mk review: put decisions to a person, with pictures, choices, notes and 3D models, on the review page mk serves."""
import argparse
import os
import time

from .. import review as RV
from ..site import show as SHOW
from .common import UsageError, emit

HELP = """Put decisions to a person: a review file lists questions, each with pictures, options to choose from and
models to turn in 3D. It opens as a page mk serves on this machine: in Tern beside the agent's pane, elsewhere in the
browser. The person picks, writes notes, draws on the pictures and turns the models, and presses Send, which writes the
answers and the message for the agent; `--wait` holds `open` until then and prints them (run it as a background job).
The file format is in mkmmd/review.py and docs/design.md (Reviews).

Examples:
  mk review check looks.review.toml           # every problem, one per line
  mk review open looks.review.toml --wait     # beside this pane; returns the answers when they are sent
  mk review answers looks.review.toml         # the answers, and the marks in mm on lab sheets
"""


def add(sub):
    p = sub.add_parser("review", help="put decisions to a person: pictures, choices, notes and 3D models on a page",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ss = p.add_subparsers(dest="review_command", metavar="ACTION", required=True)
    c = ss.add_parser("check", help="check a review file")
    c.add_argument("review", metavar="NAME.review.toml")
    c.set_defaults(func=run_check)
    s = ss.add_parser("show", help="the review as the page reads it: checked, paths absolute, answers so far (JSON)")
    s.add_argument("review", metavar="NAME.review.toml")
    s.set_defaults(func=run_show)
    o = ss.add_parser("open", help="open the review page (Tern: beside this pane; else the browser)")
    o.add_argument("review", metavar="NAME.review.toml")
    o.add_argument("--where", default="right", choices=["right", "down", "tab"], help="where it opens in Tern (default: "
                                                                                       "right of this pane)")
    o.add_argument("--wait", action="store_true", help="return when the review is sent, with the answers")
    o.add_argument("--timeout", type=float, metavar="S", help="with --wait: give up after S seconds (exit 1)")
    o.add_argument("--replace", type=int, metavar="PANE", help=argparse.SUPPRESS)    # the plugin's launcher pane
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
    path = os.path.abspath(os.path.expanduser(args.review))
    answers_file = _load(RV.prepare, path)
    sent_before = _load(RV.show, path)["sent"]
    try:
        url = SHOW.page(review=path)
        how = SHOW.show(url, args.where, args.replace)
    except (RuntimeError, OSError, ValueError) as e:
        raise UsageError(f"cannot open the review page: {e}")
    out = {"review": path, "answers_file": str(answers_file), "url": url, **how}
    if not args.wait:
        emit(out)
        return 0
    t0 = time.monotonic()
    while True:
        r = _load(RV.show, path)
        if r["sent"] and r["sent"] != sent_before:
            break
        if args.timeout is not None and time.monotonic() - t0 > args.timeout:
            emit({**out, "error": f"not sent within {args.timeout:g} s"})
            return 1
        time.sleep(0.5)
    rec = RV._read_json(answers_file) or {}
    emit({**out, "message": rec.get("message"), **_load(RV.answers, path)})
    return 0


def run_answers(args):
    emit(_load(RV.answers, args.review))
    return 0
