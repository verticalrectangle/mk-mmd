"""What the site's pages ask its server for (docs/design.md: The site). A route is fn(site, query, body) -> (status,
payload); it raises HttpError with what the page should show. Paths in queries and bodies are absolute; files go back
as `file?path=...` URLs the server serves only from the folders a page was opened for."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import numpy as np

from .. import __version__
from .. import brief as BR
from .. import review as RV
from . import draft as DR
from . import music as MU
from . import show as SHOW
from .server import HttpError, body_json, cache

TELL_MAX = 1500                                    # the longest line a sent review types into the agent's pane


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


def _remember_agent(site, b, *paths):
    """The agent that opened a page ({pane, window} from show.opener), kept for the review or project folder it shows:
    a review sent there is typed into its pane. The last agent to open a page wins; a person opening it changes none."""
    agent = b.get("agent")
    if isinstance(agent, dict) and str(agent.get("pane", "")).isdigit():
        with site.lock:
            for p in paths:
                site.state.setdefault("agents", {})[str(p)] = {"pane": str(agent["pane"]),
                                                               "window": str(agent.get("window") or "")}


def open_page(site, query, body):
    """Serve a review (its folder and the folders of the files it names) or a project (its root); answers the page's
    query string. A review inside a project (an mk.toml above it that loads) opens on the project's page, in its
    Reviews tab, so the page's tabs stay over it."""
    b = body_json(body) or {}
    if b.get("review"):
        r = _call(RV.load, b["review"])
        _allow_review(site, r)
        from ..project import Project, ProjectError
        try:
            proj = Project.find(Path(r["path"]).parent)
        except (ProjectError, ValueError, KeyError, TypeError):
            proj = None                          # a review in a project whose mk.toml does not load still opens, alone
        if proj is None:
            _remember_agent(site, b, r["path"])
            return 200, {"page": "?review=" + urllib.parse.quote(r["path"])}
        site.allow(proj.root)
        _remember_agent(site, b, r["path"], proj.root)
        return 200, {"page": "?project=" + urllib.parse.quote(str(proj.root)) + "&review=" + urllib.parse.quote(r["path"])}
    if b.get("project"):
        root = Path(b["project"]).expanduser().resolve()
        root = root.parent if root.is_file() else root
        if not (root / "mk.toml").is_file():
            raise HttpError(422, f"{root} has no mk.toml")
        site.allow(root)
        _remember_agent(site, b, root)
        return 200, {"page": "?project=" + urllib.parse.quote(str(root))}
    raise HttpError(400, "open a review or a project")


# ---------------------------------------------------------------------------------------------------------- reviews
SKIP = {"build", "renders", "out", "node_modules", "__pycache__"}


def project(site, query, body):
    """A project's page: its name, root, its brief (whether it is written and when it was sent), review files (each
    with its title, answers and whether it was sent) and whether it has music (an [audio] file or a timeline), a baked
    scene and checks."""
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
    written = (root / BR.BRIEF).is_file()
    try:
        sent = BR.read(root)["sent"] if written else None
    except BR.BriefError:
        sent = None
    return 200, {"name": proj.name, "root": str(root), "brief": {"written": written, "sent": sent}, "reviews": reviews,
                 "music": has_music, "scene": bool(proj.blend) and (bake_dir(proj) / "bake.json").is_file(),
                 "checks": bool(proj.checks)}


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
    from ..project import Project, ProjectError
    try:
        out = MU.music(root, cache())
        out["checks"] = check_marks(Project.load(root))
    except ProjectError as e:
        raise HttpError(422, str(e))
    a = out["audio"]
    if a and a.get("path"):
        site.allow(Path(a["path"]).parent)               # the [audio] file itself when ffmpeg cannot cut the clip
        a["url"] = file_url(a["path"])
    return 200, out


# ---------------------------------------------------------------------------------------------------------- checks
def _checks_of(proj):
    """The project's checks with their last results (mkmmd.checks.last_results), each with its metric's words (`doc`)
    and its worst moment in clip seconds (`t`, from the detail's at_frame)."""
    from .. import checks as CH
    CH.load()
    doc = CH.last_results(proj)
    for r in doc["results"]:
        m = CH.METRICS.get(r.get("metric"))
        r["doc"] = m.doc if m else ""
        at = r["detail"].get("at_frame") if isinstance(r.get("detail"), dict) else None
        r["t"] = None if at is None else round(proj.time(at), 4)
    return doc


def check_marks(proj):
    """The checks the music timeline marks: [{name, ok, t}] for each check whose last result has a worst moment."""
    return [{"name": r["name"], "ok": r["ok"], "t": r["t"]} for r in _checks_of(proj)["results"]
            if r.get("t") is not None and "ok" in r]


def checks(site, query, body):
    """The project's checks with their last results, and whether a run is going (POST checks starts one)."""
    proj = _project(site, query)
    run = site.state.setdefault("check_runs", {}).get(str(proj.root))
    running = run is not None and run.poll() is None
    return 200, {**_checks_of(proj), "running": running, "exit": None if run is None or running else run.returncode}


def run_checks(site, query, body):
    """Run the project's checks (`mk check`, in the background: it keeps the results in .mk/checks.json)."""
    proj = _project(site, query)
    runs = site.state.setdefault("check_runs", {})
    run = runs.get(str(proj.root))
    if run is None or run.poll() is not None:
        with open(cache() / "check-run.log", "w", encoding="utf-8") as log:
            runs[str(proj.root)] = subprocess.Popen([sys.executable, "-m", "mkmmd.cli.main", "check", "--project",
                                                     str(proj.root)], stdout=log, stderr=subprocess.STDOUT,
                                                    cwd=str(proj.root))
    return 200, {"running": True}


def review(site, query, body):
    """The review as the page draws it: RV.show, each picture with its URL, each model with its id and kind (glb: a
    binary glTF, .glb or .vrm, as it is; convert: a model file Blender turns into a .glb; build: posed by `mk model
    glb`), the marks."""
    r = _call(RV.show, _review_path(site, query))
    _allow_review(site, r)
    for img in _images(r):
        img["url"] = file_url(img["show"])
    for m in _models(r):
        m["kind"] = ("glb" if m["open"].lower().endswith(RV.AS_IS) else "convert") if m.get("open") else "build"
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


def _agent_of(site, path):
    """The agent that opened this review's page, or its project's (the nearest folder above it with one)."""
    agents = site.state.get("agents", {})
    for p in (Path(path), *Path(path).parents):
        if str(p) in agents:
            return agents[str(p)]
    return None


def _tell(site, path, text, more):
    """Type `text` as one line into the Tern pane of the agent that opened the page of `path` (or of a folder above
    it): {delivered: "pane"}, else {delivered: None} (no agent; `why` when telling it failed). Past TELL_MAX the line
    ends with where the rest is (`more`)."""
    agent = _agent_of(site, path)
    if agent is None:
        return {"delivered": None}
    line = " | ".join(s.strip() for s in text.splitlines() if s.strip())
    if len(line) > TELL_MAX:
        line = line[:TELL_MAX - 60 - len(more)].rstrip() + f" ... (all of it: {more})"
    try:
        SHOW.tell(agent, line)
        return {"delivered": "pane"}
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return {"delivered": None, "why": str(e)}


def send(site, query, body):
    """Send the review (RV.send) and get it to the agent: a waiting `mk review open --wait` has it; else the answers
    are typed into the pane of the agent that opened the page. `delivered` says which ("waiting", "pane"), or None
    (no agent: they are saved for `mk review answers`; `why` when telling the pane failed)."""
    path = _review_path(site, query)
    text = _call(RV.send, path)
    out = {"sent": _call(RV.show, path)["sent"], "message": text}
    if SHOW.is_waiting(path):
        return 200, {**out, "delivered": "waiting"}
    return 200, {**out, **_tell(site, path, text, f"mk review answers {path}")}


# ----------------------------------------------------------------------------------------------------------- briefs
def _brief(fn, *args):
    try:
        return fn(*args)
    except BR.BriefError as e:
        raise HttpError(422, str(e))


def brief(site, query, body):
    """The Brief tab (mkmmd.brief): the brief with each picture's URL, the song (a WAV of the whole of it in the site
    cache) and its length, the clip, how many lines the lyrics have (never their text), the timeline in song seconds
    (numbers only), the run timing the words, how many shots the clip carries, the sizes a video can have."""
    proj = _project(site, query)
    doc = _brief(BR.read, proj.root)
    s = _brief(BR.song, proj)
    song = None
    if s:
        a = MU.wav(s[0], 0.0, None, cache())
        site.allow(Path(a["path"]).parent)
        song = {"file": str(s[0]), "seconds": round(s[1], 3), "url": file_url(a["path"]), "offset": a["offset"],
                "problem": a.get("problem")}
    c0, c1 = BR.clip(proj)
    pictures = [*doc["video"]["pictures"], *(x for p in doc["parts"] for x in p["pictures"])]
    return 200, {"brief": doc, "pictures": {x: file_url(proj.root / x) for x in pictures}, "song": song,
                 "clip": {"start": round(c0, 3), "end": round(c1, 3)}, "lyrics": BR.lyrics_lines(proj.root),
                 "timeline": BR.timeline(proj), "words": _words_state(site, proj.root), "shots": BR.shots(proj),
                 "sizes": list(BR.SIZES)}


def save_brief(site, query, body):
    """Keep the brief the page gives ({brief: {language, video, parts}}, checked: one that does not fit is refused
    whole, naming each problem)."""
    proj = _project(site, query)
    return 200, {"brief": _brief(BR.save, proj.root, (body_json(body) or {}).get("brief"))}


def set_clip(site, query, body):
    """Make a stretch of the song the project's clip ({start, end} in song seconds; `force` when shots are on the
    clip, which would then move against the music). Not while the words are being timed."""
    proj = _project(site, query)
    b = body_json(body) or {}
    if _words_state(site, proj.root)["running"]:
        raise HttpError(409, "the words are being timed: the part can change when they are")
    n = BR.shots(proj)
    if n and not b.get("force"):
        raise HttpError(409, f"the project has {n} shot{'s' if n > 1 else ''} on its clip: moving the part moves the "
                             "music under them")
    start, end = _brief(BR.set_clip, proj.root, b.get("start"), b.get("end"))
    return 200, {"clip": {"start": start, "end": end}}


def lyrics(site, query, body):
    """Keep the lyrics of the part ({text}, one sung line per line; empty removes them): answers how many lines they
    have, never the text."""
    proj = _project(site, query)
    return 200, {"lines": _brief(BR.set_lyrics, proj.root, (body_json(body) or {}).get("text"))}


def picture(site, query, body):
    """Keep a picture for the brief ({data: a data URL}): its path in the project and its URL."""
    import base64
    proj = _project(site, query)
    head, _, data = str((body_json(body) or {}).get("data", "")).partition(",")
    if not (head.startswith("data:") and head.endswith(";base64") and data):
        raise HttpError(400, "a picture is a data URL (data:image/...;base64,...)")
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        raise HttpError(400, "the picture's data is not base64")
    rel = _brief(BR.add_picture, proj.root, raw)
    return 200, {"path": rel, "url": file_url(proj.root / rel)}


def _words_state(site, root):
    """The run timing the words of the project at `root`: {running, stage (the analysis's last `timeline: NAME`
    line), seconds, exit, error}."""
    with site.lock:
        run = site.state.get("word_runs", {}).get(str(root))
    if run is None:
        return {"running": False, "stage": None, "seconds": None, "exit": None, "error": None}
    code = run["proc"].poll()
    if code is not None and run["t1"] is None:
        run["t1"] = time.monotonic()
    try:
        log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    stages = [ln.split(":", 1)[1].strip() for ln in log.splitlines() if ln.startswith("timeline:")]
    out = {"running": code is None, "stage": stages[-1] if stages else None,
           "seconds": round((run["t1"] or time.monotonic()) - run["t0"], 1), "exit": code, "error": None}
    if code:
        try:
            out["error"] = json.loads(Path(run["out"]).read_text(encoding="utf-8")).get("error")
        except (OSError, ValueError, AttributeError):
            pass
        out["error"] = out["error"] or log.strip()[-600:] or f"exit {code}"
    return out


def words(site, query, body):
    """The run timing the words (`_words_state`)."""
    return 200, _words_state(site, _project(site, query).root)


def time_words(site, query, body):
    """Time the words of the clip: `mk timeline analyze` in the background (into audio/timeline.json), with the
    lyrics when the brief's language is English and it has them, the beats only (--no-words) for another language."""
    proj = _project(site, query)
    if _brief(BR.song, proj) is None:
        raise HttpError(422, "the project has no [audio] file: there is no song to time")
    doc = _brief(BR.read, proj.root)
    argv = [sys.executable, "-m", "mkmmd.cli.main", "timeline", "analyze", "--project", str(proj.root)]
    if doc["language"] != "en":
        argv.append("--no-words")
    elif BR.lyrics_lines(proj.root):
        argv += ["--lyrics", str(proj.root / BR.LYRICS)]
    with site.lock:
        runs = site.state.setdefault("word_runs", {})
        run = runs.get(str(proj.root))
        if run is not None and run["proc"].poll() is None:
            raise HttpError(409, "the words are being timed already")
        out, log = proj.mk_dir / "words.json", proj.mk_dir / "words.log"
        with open(out, "w", encoding="utf-8") as fo, open(log, "w", encoding="utf-8") as fl:
            proc = subprocess.Popen(argv, stdout=fo, stderr=fl, stdin=subprocess.DEVNULL, cwd=str(proj.root))
        runs[str(proj.root)] = {"proc": proc, "t0": time.monotonic(), "t1": None, "out": out, "log": log}
    return 200, _words_state(site, proj.root)


def send_brief(site, query, body):
    """Send the brief (BR.send) and type it into the pane of the agent that opened the project's page."""
    proj = _project(site, query)
    text = _brief(BR.send, proj.root)
    return 200, {"sent": BR.read(proj.root)["sent"], "message": text,
                 **_tell(site, proj.root, text, f"mk brief show --project {proj.root}")}


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
    """The model's .glb: a binary glTF (.glb, .vrm) as it is, any other model file turned into one by Blender, a .pmx
    or a spec posed by `mk model glb`; built ones are cached by command and source (`<cache>/site/models`)."""
    if m.get("open") and m["open"].lower().endswith(RV.AS_IS):
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
            ("POST", "snapshot"): snapshot, ("GET", "music"): music, ("GET", "scene"): scene, ("GET", "draft"): draft,
            ("GET", "checks"): checks, ("POST", "checks"): run_checks, ("GET", "brief"): brief,
            ("PUT", "brief"): save_brief, ("PUT", "clip"): set_clip, ("PUT", "lyrics"): lyrics,
            ("POST", "picture"): picture, ("GET", "words"): words, ("POST", "words"): time_words,
            ("POST", "send_brief"): send_brief}
