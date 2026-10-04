"""mk inspect: describe models (rig.json) and motions (tempo, energy, travel)."""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import bridge
from .. import config as CFG
from ..core import vmd
from .common import CHECK_FAILED, emit, UsageError

HELP = """Describe MMD models and motions.

A model (.pmx/.pmd) is imported into an empty scene and described as rig.json (docs/design.md: Model description):
standard bones mapped to semantic names, fingers, physics chains with their families, collision bodies, expressions,
measurements and known quirks. A motion (.vmd) gets frames, tracks, tempo (bpm, beat phase), energy and travel.

Examples:
  mk inspect model.pmx                         # print a summary, write <assets>/rigs/<slug>.rig.json
  mk inspect model.pmx --full                  # print the whole rig.json
  mk inspect assets/crowd/**/*.pmx --jobs 6    # many models in parallel
  mk inspect dance.vmd
"""


def add(sub):
    p = sub.add_parser("inspect", help="describe models (rig.json) and motions", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", metavar="FILE")
    p.add_argument("--out", metavar="PATH", help="rig.json path (one model) or folder (several)")
    p.add_argument("--jobs", type=int, default=1, help="parallel Blender processes for many models")
    p.add_argument("--scale", type=float, default=0.08, help="import scale (default 0.08: 1 MMD unit = 8 cm)")
    p.add_argument("--full", action="store_true", help="print the full description, not a summary")
    p.add_argument("--no-write", action="store_true", help="do not write rig.json files")
    p.set_defaults(func=run)


def slug(path):
    """Stable file-name slug for a model: parent folder + file stem, ASCII-safe where possible."""
    p = Path(path)
    raw = f"{p.parent.name}__{p.stem}"
    s = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return s[:120] or "model"


def rig_out_path(path, args, many):
    if args.out:
        o = Path(args.out).expanduser()
        return (o / f"{slug(path)}.rig.json") if (many or o.is_dir()) else o
    return Path(CFG.load()["assets"]) / "rigs" / f"{slug(path)}.rig.json"


def summarize_rig(rig):
    fams = {}
    for c in rig["chains"]:
        f = fams.setdefault(c["family"], {"chains": 0, "bones": 0, "locked": 0})
        f["chains"] += 1
        f["bones"] += len(c["bones"])
        f["locked"] += c["locked_joints"]
    return {"name": rig["source"]["name_j"] or rig["source"]["name_e"], "stats": rig["stats"],
            "semantic_bones": len(rig["map"]), "missing_required": rig["missing_required"],
            "families": fams, "expressions": sorted(rig["morphs"]), "measure": rig["measure"],
            "quirks": rig["quirks"], "comment_chars": len(rig.get("comment", ""))}


def inspect_model(path, args, many):
    t0 = time.time()
    rig = bridge.run("inspect_model", {"path": str(path), "scale": args.scale}, timeout=600)
    out = None
    if not args.no_write:
        out = rig_out_path(path, args, many)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rig, ensure_ascii=False, indent=1), encoding="utf-8")
    res = rig if args.full else summarize_rig(rig)
    return {"path": str(path), "rig_json": str(out) if out else None, "seconds": round(time.time() - t0, 2), **res}


def run(args):
    files = []
    for p in args.paths:
        p = Path(p).expanduser()
        if not p.exists():
            raise UsageError(f"{p} does not exist")
        files.append(p.resolve())
    models = [p for p in files if p.suffix.lower() in (".pmx", ".pmd")]
    motions = [p for p in files if p.suffix.lower() == ".vmd"]
    other = [p for p in files if p not in models and p not in motions]
    if other:
        raise UsageError(f"unsupported files: {[str(o) for o in other]}")
    results, failed = [], []
    for m in motions:
        try:
            results.append({"path": str(m), **vmd.analyse(vmd.read(m))})
        except (ValueError, OSError) as e:
            failed.append({"path": str(m), "error": str(e)})
    many = len(models) > 1
    if models:
        with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, os.cpu_count() or 1))) as ex:
            futs = {ex.submit(inspect_model, m, args, many): m for m in models}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except bridge.BlenderError as e:
                    failed.append({"path": str(futs[fut]), "error": str(e).splitlines()[0], "log": e.log})
    if len(results) == 1 and not failed:
        emit(results[0])
    else:
        emit({"inspected": len(results), "failed": failed, "results": results})
    return CHECK_FAILED if failed else 0
