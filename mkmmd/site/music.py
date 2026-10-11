"""A project's music as the site's timeline draws it (docs/design.md: The site), all in clip seconds: the clip's audio
(the [audio] file cut to the clip, cached as WAV in the site cache), the timeline's beats, bars and onsets, its words
by (line, word) and their times (never their text), the hits the project keeps ([audio] hits: named lists of hit
times, `lanes` picks and orders them), and the cut with its effects (transitions, inserts, glitches, freezes, rings,
texts). What cannot be read is named in `problems` and the rest is still shown."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from ..core import freeze as FRZ
from ..core import palette as PAL
from ..core import rings as RG
from ..core import screentype as ST
from ..core import transition as TR
from ..project import Project

TIMELINE = "audio/timeline.json"            # where `mk timeline analyze` writes a project's timeline


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _hits(raw):
    """A list of hits as [{t, db}] sorted by time (db None for a bare time), or None when `raw` is not one."""
    if not isinstance(raw, list):
        return None
    out = []
    for x in raw:
        if _num(x):
            out.append({"t": round(float(x), 4), "db": None})
        elif isinstance(x, dict) and _num(x.get("t")):
            out.append({"t": round(float(x["t"]), 4), "db": round(float(x["db"]), 1) if _num(x.get("db")) else None})
        else:
            return None
    return sorted(out, key=lambda h: h["t"])


def _words(tl):
    """Every word of the timeline as {line, word, start, end}: numbered from 1, no text."""
    out = []
    for i, line in enumerate(tl.get("lines") or [], 1):
        for j, w in enumerate((line or {}).get("words") or [], 1):
            if _num(w.get("start")) and _num(w.get("end")):
                out.append({"line": i, "word": j, "start": round(float(w["start"]), 4), "end": round(float(w["end"]), 4)})
    return out


def _text_span(e, words):
    """Where a resolved [[text]] entry is on screen, (from, to) in clip seconds, or None when it always is."""
    if isinstance(e.get("lyrics"), dict):
        ly = e["lyrics"]
        sel = [w for w in words if w["line"] == ly.get("line")]
        if isinstance(ly.get("words"), list) and len(ly["words"]) == 2:
            sel = [w for w in sel if ly["words"][0] <= w["word"] <= ly["words"][1]]
        a = float(ly["from"]) if _num(ly.get("from")) else (sel[0]["start"] if sel else None)
        b = float(ly["to"]) if _num(ly.get("to")) else (sel[-1]["end"] if sel else None)
        return (a, b) if a is not None and b is not None and b > a else None
    ink = e["ink"] if isinstance(e.get("ink"), dict) else {}
    show = e["kinetic"].get("show") if isinstance(e.get("kinetic"), dict) else None
    for a, b in ((ink.get("from"), ink.get("to")), tuple(show) if isinstance(show, list) and len(show) == 2 else (None, None)):
        if _num(a) and _num(b) and b > a:
            return float(a), float(b)
    return None


def audio(proj, cache_dir):
    """The clip's audio for the page: {file, path, offset} (`path` the file to play, `offset` the seconds of it at clip
    time 0), {file, problem} when it cannot be played, None without an [audio] file. With ffmpeg the clip is cut out as
    WAV (cached by the file's size and time, start and duration), else the whole file is played from its start."""
    au = proj.data.get("audio") or {}
    if not au.get("file"):
        return None
    src = proj.path(au["file"])
    if not src.is_file():
        return {"file": str(src), "problem": f"[audio] file {au['file']} not found"}
    start, dur = float(au.get("start", 0.0)), float(proj.duration)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"file": str(src), "path": str(src), "offset": start}
    st = src.stat()
    key = hashlib.sha1(json.dumps([str(src), st.st_size, st.st_mtime_ns, start, dur]).encode()).hexdigest()[:20]
    out = Path(cache_dir) / "audio" / f"{key}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.wav")
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(src),
                            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(tmp)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not tmp.is_file():
            return {"file": str(src), "path": str(src), "offset": start,
                    "problem": "ffmpeg could not cut the clip: " + (r.stderr.strip()[-300:] or f"exit {r.returncode}")}
        tmp.replace(out)
    return {"file": str(src), "path": str(out), "offset": 0.0}


def music(root, cache_dir):
    """The timeline of the project at `root` (see the module docstring)."""
    proj = Project.load(root)
    data, fps, frame0, duration = proj.data, float(proj.fps), int(proj.frame0), float(proj.duration)
    problems = []
    secs = lambda f: round((f - frame0) / fps, 4)                       # noqa: E731 - a Blender frame in clip seconds
    look = data.get("look") or {}
    try:
        palette = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
    except (KeyError, ValueError) as e:
        palette = None
        problems.append(f"[look] palette: {e}")

    tl = _json(proj.path(TIMELINE)) if proj.path(TIMELINE).is_file() else None
    if tl is None:
        problems.append(f"no timeline ({TIMELINE}: `mk timeline analyze` writes it)" if not proj.path(TIMELINE).is_file()
                        else f"{TIMELINE} cannot be read")
        tl = {}
    beats = [round(float(t), 4) for t in tl.get("beats") or [] if _num(t)]
    downbeats = [round(float(t), 4) for t in tl.get("downbeats") or [] if _num(t)]
    words = _words(tl)
    lanes = []
    au = data.get("audio") or {}
    if au.get("hits"):
        doc = _json(proj.path(au["hits"]))
        if not isinstance(doc, dict):
            problems.append(f"[audio] hits {au['hits']}: not a JSON object of hit lists")
        else:
            names = au.get("lanes") or [k for k in doc if k not in ("beats", "downbeats")]
            for name in names:
                hits = _hits(doc.get(name))
                if hits is None:
                    problems.append(f"[audio] hits {au['hits']}: {name!r} is not a list of hit times")
                else:
                    lanes.append({"name": str(name), "source": "hits", "hits": hits})
    for stem, ts in (tl.get("onsets") or {}).items():
        hits = _hits(ts)
        if hits is not None:
            lanes.append({"name": str(stem), "source": "onsets", "hits": hits})

    shots, effects = [], []
    if palette is not None:
        try:
            plan = TR.plan(data, fps, frame0, palette)
            shots = [{"name": c["name"], "from": secs(c["from"]), "to": secs(c["to"])} for c in plan["cuts"]]
            for tr in plan["transitions"]:
                effects.append({"kind": "transition", "label": tr["kind"], "from": secs(tr["first"]), "to": secs(tr["last"] + 1)})
            for ins in plan["inserts"]:
                effects.append({"kind": "insert", "label": str(ins["shot"]), "from": secs(ins["f0"]), "to": secs(ins["f1"])})
            for g in plan["glitches"]:
                effects.append({"kind": "glitch", "label": "glitch", "from": secs(g["f0"]), "to": secs(g["f1"])})
        except TR.TransitionError as e:
            problems.append(f"the cut: {e}")
        try:
            for r in RG.normalize(data.get("ring"), palette):
                effects.append({"kind": "ring", "label": "held ring" if r["hold"] else "ring", "from": round(r["at"], 4),
                                "to": round(min(r["until"], duration), 4)})
        except RG.RingError as e:
            problems.append(f"rings: {e}")
    try:
        for f0, f1 in FRZ.normalize(data.get("freeze"), fps, frame0):
            effects.append({"kind": "freeze", "label": "freeze", "from": secs(f0), "to": secs(f1)})
    except FRZ.FreezeError as e:
        problems.append(f"freezes: {e}")
    try:
        for k, e in enumerate(ST.resolve_entries([t for t in data.get("text") or [] if isinstance(t, dict)])):
            span = _text_span(e, words)
            if span:
                effects.append({"kind": "text", "label": str(e.get("name") or f"text {k + 1}"),
                                "from": round(span[0], 4), "to": round(span[1], 4)})
    except ST.TextError as e:
        problems.append(f"texts: {e}")
    effects.sort(key=lambda e: (e["from"], e["to"]))

    return {"name": proj.name, "root": str(proj.root), "fps": fps, "frame0": frame0, "duration": duration,
            "audio": audio(proj, cache_dir), "bpm": float(tl["bpm"]) if _num(tl.get("bpm")) else None,
            "beats": beats, "downbeats": downbeats, "words": words, "lanes": lanes, "shots": shots, "effects": effects,
            "problems": problems}
