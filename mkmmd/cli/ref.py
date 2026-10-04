"""mk ref: measure how real people move in reference clips and turn it into performance parameters."""
import argparse
from pathlib import Path

from ..core import jsonx
from ..ref import RefError, RefUsage, measure as ME
from ..ref import store
from .common import CHECK_FAILED, RUNTIME, UsageError, add_project_arg, emit, get_project

HELP = """Reference clips -> numbers for the performance stage ([perform.<cast>]).

Find short Pexels clips of people doing what the character does (singing, driving, listening), track them with
MediaPipe (face, blendshapes, head pose, hands, shoulders, camera drift) and read off how people really blink, look
around, nod, sway, breathe and open their mouths. `measure` pools the clips (median and IQR over clips and over
events) and recommends blink / lids / gaze / head_share / eye_max / nod / sway / breath / bob / sing values with a
`why`, `range`, sample count and confidence for each.

Sets live in ~/.cache/mk/ref/<project name | default>/<set>/ (clips.json, clips/, track/). Downloads (at most 150 MB per
clip, the smallest rendition >= 720p) and tracking data are copyrighted and large: they never go into a repository;
`mk ref clean` deletes them. Inside a project, `measure` writes <project>/ref/<set>.json (numbers only, safe to keep).
The Pexels API key is read from the system keyring (secret-tool lookup service pexels key api) and never printed.

Examples:
  mk ref search 'woman singing in car' --n 20
  mk ref add 8042695 6954269 --set singing --note 'front-on, studio light'
  mk ref track --set singing --jobs 3          # roughly 1.5 s of tracking per second of video and job
  mk ref measure --set singing                 # recommend + why on stdout, everything in the .json
  mk ref sheet --set singing                   # landmarks drawn on frames at the measured events
  mk ref clean --set singing                   # delete downloads, tracks and the sheet
"""


def add(sub):
    p = sub.add_parser("ref", help="reference clips: search, track, measure, performance parameters", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)

    se = s.add_parser("search", help="Pexels video search (downloads nothing)")
    se.add_argument("query")
    se.add_argument("--n", type=int, default=30, help="how many results (default 30)")
    se.add_argument("--min-height", type=int, default=720, help="smallest short side of a usable rendition, px (default 720)")
    se.add_argument("--orientation", choices=("landscape", "portrait", "any"), default="landscape")

    ad = s.add_parser("add", help="remember clips in a reference set (id, author, url, license note, why)")
    ad.add_argument("ids", nargs="+", type=int, metavar="ID", help="Pexels video ids")
    ad.add_argument("--set", dest="set_name", metavar="NAME", help=f"reference set (default {store.DEFAULT_SET!r})")
    ad.add_argument("--note", help="why these clips (kept in clips.json and in the measure output)")
    add_project_arg(ad)

    tr = s.add_parser("track", help="download (capped) and track the clips of a set with MediaPipe")
    tr.add_argument("--set", dest="set_name", metavar="NAME")
    tr.add_argument("--jobs", type=int, help="parallel tracking processes (default min(4, cpus / 2))")
    tr.add_argument("--min-height", type=int, default=720, help="smallest short side of the rendition to download (default 720)")
    tr.add_argument("--max-seconds", type=float, default=90.0, help="track at most this much of a clip (default 90)")
    tr.add_argument("--force", action="store_true", help="track again even if a track exists")
    add_project_arg(tr)

    me = s.add_parser("measure", help="per-clip and pooled metrics, recommended perform parameters")
    me.add_argument("--set", dest="set_name", metavar="NAME")
    me.add_argument("--out", metavar="FILE.json", help="where the full result goes (default <project>/ref/<set>.json, else the set's cache folder)")
    me.add_argument("--full", action="store_true", help="print the full result on stdout instead of the digest")
    add_project_arg(me)

    sh = s.add_parser("sheet", help="contact sheet with the landmarks drawn at the measured events")
    sh.add_argument("--set", dest="set_name", metavar="NAME")
    sh.add_argument("--out", metavar="FILE.jpg", help="default: contact_sheet.jpg in the set's cache folder")
    sh.add_argument("--per-row", type=int, default=4, help="frames per clip (default 4)")
    add_project_arg(sh)

    cl = s.add_parser("clean", help="delete downloads, tracks and the contact sheet")
    cl.add_argument("--set", dest="set_name", metavar="NAME")
    cl.add_argument("--all", action="store_true", help="every set of this project (or the default scope) and the model bundles")
    cl.add_argument("--keep-tracks", action="store_true", help="delete the downloads and the sheet only")
    cl.add_argument("--forget", action="store_true", help="also delete the set itself (clips.json, measure.json)")
    add_project_arg(cl)
    p.set_defaults(func=run)


def run(args):
    try:
        return _run(args)
    except RefUsage as e:
        raise UsageError(str(e))
    except RefError as e:
        emit({"error": str(e)})
        return RUNTIME


def _set(args):
    proj = get_project(args)
    return proj, store.RefSet(store.scope_dir(proj), args.set_name)


def _run(args):
    act = args.action
    if act == "search":
        from ..ref import pexels
        res = pexels.search(args.query, args.n, args.min_height, args.orientation)
        emit({"query": args.query, "count": len(res), "min_height": args.min_height, "orientation": args.orientation,
              "note": "`pick` is the rendition `mk ref track` downloads; `mk ref add ID ... --set NAME` remembers clips", "results": res})
    elif act == "add":
        from ..ref import pexels
        proj, rs = _set(args)
        key = pexels.api_key()
        added, updated, failed = [], [], []
        for cid in args.ids:
            try:
                entry = pexels.clip_entry(pexels.video(cid, key))
            except (RefUsage, RefError) as e:
                failed.append({"id": cid, "error": str(e)})
                continue
            (added if rs.put(entry, args.note) == "added" else updated).append(cid)
        clips = rs.clips()
        emit({"set": rs.name, "project": proj.name if proj else None, "path": str(rs.clips_file), "added": added, "updated": updated,
              "failed": failed, "count": len(clips),
              "clips": [{k: c.get(k) for k in ("id", "author", "url", "duration_s", "license", "why")} for c in clips]})
        return CHECK_FAILED if failed else 0
    elif act == "track":
        from ..ref import track
        proj, rs = _set(args)
        rep = track.track_set(rs, args.jobs, args.min_height, args.max_seconds, args.force)
        rep["usage"] = rs.usage()
        emit(rep)
        return CHECK_FAILED if rep["failed"] else 0
    elif act == "measure":
        return _measure(args)
    elif act == "sheet":
        from ..ref import sheet
        proj, rs = _set(args)
        pal = (proj.data.get("look", {}) if proj else {}).get("palette", "rose-pine-moon")
        emit(sheet.make_sheet(rs, args.out or rs.sheet_file, args.per_row, pal))
    elif act == "clean":
        proj = get_project(args)
        scope = store.scope_dir(proj)
        if args.all:
            rep = store.clean_scope(scope, True, not args.keep_tracks, args.forget, models=True)
        else:
            name = store.set_name(args.set_name)
            rep = {name: store.RefSet(scope, name).clean(True, not args.keep_tracks, args.forget)}
        emit({"scope": str(scope), "freed_bytes": store.total_freed(rep), "freed_mb": round(store.total_freed(rep) / 1e6, 1), "sets": rep})
    return 0


def _measure(args):
    import numpy as np
    proj, rs = _set(args)
    clips = rs.require_clips()
    have = [c for c in clips if rs.track_path(c["id"]).exists()]
    if not have:
        raise RefUsage(f"no tracks for set {rs.name!r}: run `mk ref track --set {rs.name}` first")
    untracked = [c["id"] for c in clips if c not in have]
    out, _ = ME.run_measure(have, lambda cid: dict(np.load(rs.track_path(cid))), proj.name if proj else None, rs.name,
                            fps=proj.fps if proj else 30.0)
    out["untracked"] = untracked
    if args.out:
        path = Path(args.out).expanduser()
    elif proj:
        path = proj.root / "ref" / f"{rs.name}.json"
    else:
        path = rs.measure_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonx.dumps(out, precision=6) + "\n", encoding="utf-8")
    out["out"] = str(path)
    emit(out if args.full else ME.digest(out))
    return 0
