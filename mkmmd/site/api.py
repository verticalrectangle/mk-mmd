"""What the site's pages ask its server for (docs/design.md: The site). A route is fn(site, query, body) -> (status,
payload); it raises HttpError with what the page should show. Paths in queries and bodies are absolute; files go back
as `file?path=...` URLs the server serves only from the folders a page was opened for."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

import numpy as np

from .. import __version__
from .. import review as RV
from . import draft as DR
from . import music as MU
from .server import HttpError, body_json, cache


def file_url(path):
    return "file?path=" + urllib.parse.quote(str(path))


def _call(fn, *args):
    try:
        return fn(*args)
    except RV.ReviewError as e:
        raise HttpError(422, str(e))


def _images(r):
    for q in r["questions"]:
        yield from q["images"]
        for o in q["options"]:
            yield from o["images"]


def _models(r):
    """The review's models in page order (question, then its options), each with its id (m1, m2 ...)."""
    out = []
    for q in r["questions"]:
        for m in [*q["models"], *(m for o in q["options"] for m in o["models"])]:
            m["id"] = f"m{len(out) + 1}"
            out.append(m)
    return out


def _review_path(site, query):
    path = query.get("path")
    if not path:
        raise HttpError(400, "no review path")
    p = Path(path).expanduser().resolve()
    site.allowed(p)                        # the page must have been opened for its folder
    return p


# --------------------------------------------------------------------------------------------------------- the server
def ping(site, query, body):
    return 200, {"ok": True, "version": __version__, "pid": os.getpid()}


def stop(site, query, body):
    httpd = site.state.get("httpd")
    if httpd is not None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    return 200, {"ok": True}


def _allow_review(site, r):
    """A review's page may show its pictures, models and music wherever they are: their folders are served."""
    site.allow(Path(r["path"]).parent)
    for img in _images(r):
        site.allow(Path(img["path"]).parent)
    for m in _models(r):
        if m.get("open"):
            site.allow(Path(m["open"]).parent)
    for q in r["questions"]:
        if q.get("music"):
            site.allow(q["music"]["project"])


def open_page(site, query, body):
    """Serve a review (its folder and the folders of the files it names) or a project (its root); answers the page's
    query string."""
    b = body_json(body) or {}
    if b.get("review"):
        r = _call(RV.load, b["review"])
        _allow_review(site, r)
        return 200, {"page": "?review=" + urllib.parse.quote(r["path"])}
    if b.get("project"):
        root = Path(b["project"]).expanduser().resolve()
        root = root.parent if root.is_file() else root
        if not (root / "mk.toml").is_file():
            raise HttpError(422, f"{root} has no mk.toml")
        site.allow(root)
        return 200, {"page": "?project=" + urllib.parse.quote(str(root))}
    raise HttpError(400, "open a review or a project")


# ---------------------------------------------------------------------------------------------------------- reviews
SKIP = {"build", "renders", "out", "node_modules", "__pycache__"}


def project(site, query, body):
    """A project's page: its name, root, review files (each with its title, answers and whether it was sent) and whether
    it has music (an [audio] file or a timeline) and a baked scene."""
    root = Path(query.get("path") or "").expanduser().resolve()
    if not (root / "mk.toml").is_file():
        raise HttpError(422, f"{root} has no mk.toml")
    site.allowed(root / "mk.toml")
    from ..project import Project
    proj = Project.load(root)
    reviews = []
    for p in sorted(root.rglob("*" + RV.SUFFIX)):
        if any(part.startswith(".") or part in SKIP for part in p.relative_to(root).parts[:-1]):
            continue
        item = {"path": str(p), "name": p.name}
        try:
            r = RV.show(p)
            asked = [q for q in r["questions"] if q["options"]]
            item.update(title=r["title"], sent=r["sent"], total=len(asked),
                        answered=sum(1 for q in asked if (r["answers"].get(q["id"]) or {}).get("choice")))
        except RV.ReviewError as e:
            item["error"] = str(e)
        reviews.append(item)
    has_music = bool((proj.data.get("audio") or {}).get("file")) or (root / MU.TIMELINE).is_file()
    return 200, {"name": proj.name, "root": str(root), "reviews": reviews, "music": has_music,
                 "scene": bool(proj.blend) and (bake_dir(proj) / "bake.json").is_file()}


def bake_dir(proj):
    """Where `mk build` bakes the project's scene: beside [project] blend, as <name>.bake (mkmmd/blender/build/bake.py)."""
    return Path(proj.blend).with_suffix(".bake")


def _project(site, query):
    """The project at ?path= (its root), which the page must have been opened for."""
    root = Path(query.get("path") or "").expanduser().resolve()
    if not (root / "mk.toml").is_file():
        raise HttpError(422, f"{root} has no mk.toml")
    site.allowed(root / "mk.toml")
    from ..project import Project, ProjectError
    try:
        return Project.load(root)
    except ProjectError as e:
        raise HttpError(422, str(e))


def scene(site, query, body):
    """A project's baked scene as file URLs (bake.json, bake.bin, scene.glb), whether the .blend is newer than it, and
    the outputs mk post has a draft of."""
    proj = _project(site, query)
    if not proj.blend:
        raise HttpError(422, "the project has no [project] blend, so `mk build` bakes no scene")
    d = bake_dir(proj)
    if not (d / "bake.json").is_file():
        raise HttpError(404, f"no baked scene in {d}: `mk build` bakes it (`mk build --bake` a saved scene)")
    site.allow(d)
    blend = Path(proj.blend)
    return 200, {"json": file_url(d / "bake.json"), "bin": file_url(d / "bake.bin"), "glb": file_url(d / "scene.glb"),
                 "stale": blend.is_file() and blend.stat().st_mtime > (d / "bake.json").stat().st_mtime,
                 "drafts": [o.name for o in proj.outputs if DR.path(proj, o.name).is_file()]}


def draft(site, query, body):
    """mk post's draft of ?output= as the scene viewer decodes it (mkmmd.site.draft): its URL and sample table."""
    proj = _project(site, query)
    f = DR.path(proj, query.get("output") or "")
    if not f.is_file():
        raise HttpError(404, f"no draft {f.name}: `mk post --preset draft` makes it")
    try:
        doc = DR.index(f, cache())
    except ValueError as e:
        raise HttpError(422, str(e))
    site.allow(f.parent)
    return 200, {"url": file_url(f), **doc}


def music(site, query, body):
    """A project's music timeline (mkmmd.site.music) with its audio as a file URL: the page must have been opened for the
    project, or for a review whose question names it."""
    root = Path(query.get("path") or "").expanduser().resolve()
    if not (root / "mk.toml").is_file():
        raise HttpError(422, f"{root} has no mk.toml")
    site.allowed(root / "mk.toml")
    from ..project import ProjectError
    try:
        out = MU.music(root, cache())
    except ProjectError as e:
        raise HttpError(422, str(e))
    a = out["audio"]
    if a and a.get("path"):
        site.allow(Path(a["path"]).parent)               # the [audio] file itself when ffmpeg cannot cut the clip
        a["url"] = file_url(a["path"])
    return 200, out


def review(site, query, body):
    """The review as the page draws it: RV.show, each picture with its URL, each model with its id and kind (glb: a
    file as it is; convert: a model file Blender turns into a .glb; build: posed by `mk model glb`), the marks."""
    r = _call(RV.show, _review_path(site, query))
    _allow_review(site, r)
    for img in _images(r):
        img["url"] = file_url(img["show"])
    for m in _models(r):
        m["kind"] = ("glb" if m["open"].lower().endswith(".glb") else "convert") if m.get("open") else "build"
        m.pop("glb", None)
    r["marks"] = RV._read_json(r["marks_file"]) or {"images": []}
    r["views"] = [{**v, "url": file_url(v["path"])} for v in RV.views(r["path"])]
    if r["views"]:
        site.allow(Path(r["views"][0]["path"]).parent)
    return 200, r


def marks(site, query, body):
    path = _review_path(site, query)
    return 200, _call(RV.write_marks, path, (body_json(body) or {}).get("images"))


def answers(site, query, body):
    path = _review_path(site, query)
    return 200, {"answers": _call(RV.write_answers, path, (body_json(body) or {}).get("answers"))}


def send(site, query, body):
    path = _review_path(site, query)
    text = _call(RV.send, path)
    return 200, {"sent": _call(RV.show, path)["sent"], "message": text}


# ----------------------------------------------------------------------------------------------------------- models
def _source_stamp(m):
    """What a built model depends on besides its command: its source file's size and time, and for a spec the .toml
    files of its folder and every base's files (a spec includes bases)."""
    src = m.get("open") or m["glb"][2]
    files = []
    if src.startswith("base:") or src.endswith(".toml"):
        from ..model import bases
        root = Path(bases.__file__).parent
        files += sorted([*root.glob("*/*.toml"), *root.glob("*/*.npz")])
        if src.endswith(".toml"):
            files += sorted(Path(src).parent.glob("*.toml"))
    else:
        files.append(Path(src))
    return [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.is_file()]


def _model_entry(site, query):
    path = _review_path(site, query)
    r = _call(RV.load, path)
    m = next((x for x in _models(r) if x["id"] == query.get("id")), None)
    if m is None:
        raise HttpError(404, f"no model {query.get('id')!r} in {path.name}")
    pose = query.get("pose")
    if pose:
        from ..model import lab as LAB
        if not m.get("glb"):
            raise HttpError(422, f"{m['label']}: only a .pmx or a spec can be posed")
        if pose not in LAB.POSES:
            raise HttpError(422, f"pose {pose!r}: one of {', '.join(LAB.POSES)}")
        argv = list(m["glb"])
        argv[argv.index("--pose") + 1] = pose
        m = {**m, "glb": argv}
    return path, m


def _model_glb(site, m):
    """The model's .glb: a .glb file as it is, any other model file turned into one by Blender, a .pmx or a spec posed
    by `mk model glb`; built ones are cached by command and source (`<cache>/site/models`)."""
    if m.get("open") and m["open"].lower().endswith(".glb"):
        return Path(m["open"])
    key = hashlib.sha1(json.dumps([m.get("glb") or ["convert", m["open"]], _source_stamp(m)]).encode()).hexdigest()[:20]
    out = cache() / "models" / f"{key}.glb"
    with site.lock:
        lock = site.state.setdefault("model_locks", {}).setdefault(key, threading.Lock())
    with lock:
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".{os.getpid()}.tmp.glb")
            if m.get("open"):
                from .. import bridge
                try:
                    bridge.run("model_to_glb", {"src": m["open"], "out": str(tmp)}, use_serve=False)
                except bridge.BlenderError as e:
                    raise HttpError(500, f"Blender could not turn {Path(m['open']).name} into a .glb: {e}")
            else:
                res = subprocess.run([sys.executable, "-m", "mkmmd.cli.main", *m["glb"], "--out", str(tmp)],
                                     capture_output=True, text=True, timeout=900)
                if res.returncode != 0:
                    try:
                        why = json.loads(res.stdout).get("error")
                    except ValueError:
                        why = (res.stderr or res.stdout)[-800:]
                    raise HttpError(500, f"mk model glb failed: {why}")
            tmp.replace(out)
    return out


def model(site, query, body):
    """A review model (?id, ?pose for a .pmx or a spec) as a .glb URL."""
    _, m = _model_entry(site, query)
    return 200, {"url": file_url(_model_glb(site, m)), "label": m["label"], "posable": bool(m.get("glb"))}


def _mesh(site, glb):
    from ..model import lab as LAB
    from .meshes import read_glb
    key = (str(glb), glb.stat().st_mtime_ns)
    with site.lock:
        hit = site.state.setdefault("meshes", {}).get(key)
    if hit is None:
        V, T = read_glb(glb)
        hit = (V, T, LAB._welded(V))
        with site.lock:
            site.state["meshes"] = {key: hit}            # one mesh kept: the one being cut
    return hit


def section(site, query, body):
    """Where a plane ({p, d, target}: point, normal and the point to measure around, glTF axes) cuts the model: each
    loop's segments, perimeter (mm) and whether it closes, and which loop a girth there takes (the innermost around
    the target, as `mk model lab` measures)."""
    from ..model import lab as LAB
    _, m = _model_entry(site, query)
    b = body_json(body) or {}
    try:
        p, d = np.asarray(b["p"], float), np.asarray(b["d"], float)
        target = np.asarray(b.get("target") or b["p"], float)
    except (KeyError, TypeError, ValueError):
        raise HttpError(400, "a section needs p and d (three numbers each)")
    V, T, weld = _mesh(site, _model_glb(site, m))
    loops = LAB.sections(V, T, p, d, weld)
    pick = LAB._nearest(loops, target, d)
    return 200, {"loops": [{"segments": np.round(np.stack([lp["a"], lp["b"]], 1), 5).reshape(-1).tolist(),
                            "perimeter_mm": round(lp["perimeter"] * 1000.0, 1), "closed": lp["closed"],
                            "centroid": np.round(lp["centroid"], 5).tolist()} for lp in loops],
                 "girth": None if pick is None else next(i for i, lp in enumerate(loops) if lp is pick)}


def measures(site, query, body):
    """A .pmx's or a spec's measures as `mk model lab` takes them (its pose, region and side), each girth with its
    plane in glTF axes so the page can cut there."""
    from ..model import glb as GLB
    from ..model import lab as LAB
    from ..model import partcache as PC
    from ..cli.main import build_parser
    _, m = _model_entry(site, query)
    if not m.get("glb"):
        return 200, {"measures": []}
    key = json.dumps([m["glb"], _source_stamp(m)])
    with site.lock:
        hit = site.state.setdefault("measures", {}).get(key)
    if hit is not None:
        return 200, hit
    args = build_parser().parse_args(m["glb"])
    pmx = args.model.lower().endswith(".pmx")
    parts = None if pmx or not args.parts or args.parts.strip() == "all" else [s.strip() for s in args.parts.split(",")]
    morphs = {n.strip(): float(w) if eq else 1.0 for n, eq, w in (x.partition("=") for x in args.morph)}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mk_site_") as tmp:
        model_ = LAB.load(args.model, overrides=args.overrides, parts=parts, unit=args.unit,
                          workdir=None if pmx else tmp, cache=PC.PartCache())
        V, _N = model_.deform(args.pose, morphs)
        cuts = {}
        values = LAB.measure(model_, args.region, args.side, V, cuts=cuts)
    A = GLB.AXES
    out = {"measures": [{"name": k, "mm": v, **({"p": np.round(A @ cuts[k]["p"], 5).tolist(), "d": np.round(A @ cuts[k]["d"], 5).tolist(),
                                                  "target": np.round(A @ cuts[k]["target"], 5).tolist()} if k in cuts else {})}
                        for k, v in values.items()]}
    with site.lock:
        site.state["measures"][key] = out
    return 200, out


def snapshot(site, query, body):
    """Keep a picture of the 3D viewer as a view of the review ({png: a data URL, model, label, question, camera}), so
    it can be drawn on; answers the picture as the page draws pictures."""
    import base64
    path = _review_path(site, query)
    b = body_json(body) or {}
    head, _, data = str(b.get("png", "")).partition(",")
    if head != "data:image/png;base64" or not data:
        raise HttpError(400, "a view is a PNG data URL")
    meta = {k: b[k] for k in ("model", "label", "question", "camera") if k in b}
    v = _call(RV.add_view, path, base64.b64decode(data), meta)
    site.allow(Path(v["path"]).parent)
    return 200, {**v, "url": file_url(v["path"])}


def routes():
    return {("GET", "ping"): ping, ("POST", "stop"): stop, ("POST", "open"): open_page, ("GET", "project"): project,
            ("GET", "review"): review, ("PUT", "answers"): answers, ("POST", "send"): send, ("PUT", "marks"): marks,
            ("GET", "model"): model, ("POST", "section"): section, ("GET", "measures"): measures,
            ("POST", "snapshot"): snapshot, ("GET", "music"): music, ("GET", "scene"): scene, ("GET", "draft"): draft}
