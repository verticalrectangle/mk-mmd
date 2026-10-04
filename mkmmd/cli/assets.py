"""mk assets: the local asset registry (models, motions, props, vehicles, audio, references) and credits."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import assets as A
from .. import bridge
from ..core import vmd
from .common import CHECK_FAILED, add_project_arg, emit, get_project, UsageError

HELP = """Keep track of every asset a project uses: where it lives, who made it and what its license asks for.

The registry is <assets>/registry.json (MK_ASSETS, default ~/mk-assets). Adding a model also inspects it and stores
its rig.json under <assets>/rigs/. mk never guesses a license: new entries are `unreviewed` and keep the paths of
the author's readme files; read them (`mk assets show SLUG --terms` prints the lines about terms of use) and fill in
license, restrictions and credit with `mk assets set`. `mk assets credits` fails while anything used is unreviewed.

Examples:
  mk assets add ~/models/reisen/reisen.pmx --kind model --slug miy_reisen --author Miy
  mk assets scan ~/models --kind model --jobs 6
  mk assets show miy_reisen --terms
  mk assets set miy_reisen license="Miy terms of use (readme)" credit="Reisen model: Miy" restrictions="no political use"
  mk assets credits --out CREDITS.md               # in a project: its cast and [credits] assets
"""


def add(sub):
    p = sub.add_parser("assets", help="asset registry and credits", description=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    a = s.add_parser("add", help="register one file")
    a.add_argument("path")
    a.add_argument("--kind", required=True, choices=A.KINDS)
    for f in ("slug", "name", "author", "url", "license", "restrictions", "credit", "tags"):
        a.add_argument(f"--{f}")
    a.add_argument("--readme-root", help="look for readme files up to this folder (default: two levels up)")
    a.add_argument("--replace", action="store_true", help="replace an entry with the same slug")
    a.add_argument("--no-inspect", action="store_true", help="models: do not build rig.json now")
    sc = s.add_parser("scan", help="register every model (.pmx/.pmd) or motion (.vmd) under a folder")
    sc.add_argument("folder")
    sc.add_argument("--kind", required=True, choices=("model", "motion"))
    sc.add_argument("--tags")
    sc.add_argument("--jobs", type=int, default=4)
    ls = s.add_parser("list", help="list entries")
    ls.add_argument("--kind", choices=A.KINDS)
    ls.add_argument("--tag")
    sh = s.add_parser("show", help="one entry")
    sh.add_argument("slug")
    sh.add_argument("--terms", action="store_true", help="print readme lines about terms of use")
    st = s.add_parser("set", help="update fields: field=value ... (tags as a,b; empty value removes)")
    st.add_argument("slug")
    st.add_argument("fields", nargs="+")
    rm = s.add_parser("rm", help="remove an entry (files are not touched)")
    rm.add_argument("slug")
    cr = s.add_parser("credits", help="credits for a project's assets (cast + [credits] assets) or given slugs")
    cr.add_argument("slugs", nargs="*")
    cr.add_argument("--out", help="write the markdown here")
    add_project_arg(cr)
    p.set_defaults(func=run)


def _tags(s):
    return [t.strip() for t in s.split(",") if t.strip()] if s else None


def _model_entry(path, reg, slug=None):
    rig = bridge.run("inspect_model", {"path": str(path)}, timeout=600)
    slug = slug or A.slugify(rig["source"]["name_e"] or f"{Path(path).parent.name}_{Path(path).stem}")
    rel = Path("rigs") / f"{slug}.rig.json"
    out = reg.root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig, ensure_ascii=False, indent=1), encoding="utf-8")
    return slug, rig, str(rel)


def _entry(path, kind, args_ns=None, reg=None, tags=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise UsageError(f"{path} does not exist")
    g = (lambda k: getattr(args_ns, k, None)) if args_ns else (lambda k: None)
    root = g("readme_root") or path.parents[min(1, len(path.parents) - 1)]
    readmes = [str(r) for r in A.find_readmes(path, stop=root)]
    e = {"kind": kind, "path": str(path), "slug": g("slug"), "name": g("name"), "author": g("author"),
         "source_url": g("url"), "license": g("license") or "unreviewed", "restrictions": g("restrictions"),
         "credit": g("credit"), "tags": _tags(g("tags")) or tags, "readmes": readmes}
    extra = {}
    if kind == "model" and not g("no_inspect"):
        slug, rig, rel = _model_entry(path, reg, e["slug"])
        e["slug"], e["rig"] = slug, rel
        e["name"] = e["name"] or rig["source"]["name_j"] or rig["source"]["name_e"]
        extra = {"height_m": rig["measure"].get("top"), "families": sorted({c["family"] for c in rig["chains"]}),
                 "quirks": len(rig["quirks"])}
    elif kind == "motion" and path.suffix.lower() == ".vmd":
        info = vmd.analyse(vmd.read(path))
        extra = {"frames": info["frames"], "bpm": (info.get("tempo") or {}).get("bpm"),
                 "in_place": info["travel"]["in_place"], "camera_keys": info["camera_keys"]}
    e["extra"] = extra or None
    e["slug"] = e["slug"] or A.slugify(f"{path.parent.name}_{path.stem}")
    return e


def run(args):
    reg = A.Registry()
    try:
        return _run(args, reg)
    except A.AssetError as e:
        raise UsageError(str(e))


def _run(args, reg):
    act = args.action
    if act == "add":
        e = reg.add(_entry(args.path, args.kind, args, reg), replace=args.replace)
        reg.save()
        emit(e)
    elif act == "scan":
        exts = {".pmx", ".pmd"} if args.kind == "model" else {".vmd"}
        files = sorted(p for p in Path(args.folder).expanduser().rglob("*") if p.suffix.lower() in exts)
        new = [f for f in files if reg.find_path(f) is None]
        added, failed = [], []
        with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, os.cpu_count() or 1))) as ex:
            futs = {ex.submit(_entry, f, args.kind, None, reg, _tags(args.tags)): f for f in new}
            for fut in as_completed(futs):
                try:
                    e = fut.result()
                except (bridge.BlenderError, ValueError, OSError) as err:
                    failed.append({"path": str(futs[fut]), "error": str(err).splitlines()[0]})
                    continue
                base, n = e["slug"], 2
                while any(x["slug"] == e["slug"] for x in reg.entries):
                    e["slug"] = f"{base}_{n}"
                    n += 1
                added.append(reg.add(e)["slug"])
        reg.save()
        emit({"found": len(files), "already": len(files) - len(new), "added": len(added), "failed": failed})
        return CHECK_FAILED if failed else 0
    elif act == "list":
        rows = [e for e in reg.entries if (not args.kind or e["kind"] == args.kind) and
                (not args.tag or args.tag in (e.get("tags") or []))]
        emit([{k: e.get(k) for k in ("slug", "kind", "name", "author", "license")} for e in rows])
    elif act == "show":
        e = reg.get(args.slug)
        out = dict(e)
        if args.terms:
            out["terms"] = {r: A.license_lines(r) for r in e.get("readmes", []) if Path(r).exists()}
        emit(out)
    elif act == "set":
        fields = {}
        for kv in args.fields:
            k, sep, v = kv.partition("=")
            if not sep:
                raise UsageError(f"{kv!r}: use field=value")
            fields[k] = (_tags(v) if k == "tags" else v) if v != "" else None
        e = reg.update(args.slug, **fields)
        reg.save()
        emit(e)
    elif act == "rm":
        emit(reg.remove(args.slug))
        reg.save()
    elif act == "credits":
        slugs = list(args.slugs)
        lines = []
        if not slugs:
            proj = get_project(args, required=True)
            slugs = list(dict.fromkeys(c["asset"] for c in proj.cast if c.get("asset")))   # one entry per asset
            slugs += [s for s in proj.data.get("credits", {}).get("assets", []) if s not in slugs]
            lines = proj.data.get("credits", {}).get("lines", [])
        if not slugs and not lines:
            raise UsageError("no assets: give slugs, or list them in the project's cast / [credits] assets / lines")
        text, problems = reg.credits(slugs) if slugs else ("# Credits\n", [])
        text = text.rstrip("\n") + "\n" + extra_credits(lines)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        emit({"assets": slugs, "lines": len(lines), "out": args.out, "problems": problems,
              "text": None if args.out else text})
        return CHECK_FAILED if problems else 0
    return 0


def extra_credits(lines):
    """Markdown for the project's own credit lines (`[credits] lines = [...]`: the song, a print made for the project, the
    palette, tools). A line starting with `#` is a heading and passes through; the rest are bullets, under `## Also`
    until a heading says otherwise. Empty for no lines."""
    if not lines:
        return ""
    out, heading = [], False
    for ln in (str(x).strip() for x in lines):
        if not ln:
            continue
        if ln.startswith("#"):
            out += ["", "## " + ln.lstrip("#").strip(), ""]
            heading = True
            continue
        if not heading:
            out += ["", "## Also", ""]
            heading = True
        out.append(f"- {ln}")
    return "\n".join(out) + "\n"
