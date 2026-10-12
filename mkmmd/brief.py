"""Briefs (docs/design.md: Briefs): what a person wants from a video, written on the Brief tab of the project's page
before anything is built, for the agent to answer with a treatment (a review). The part of the song is the project's
clip ([audio] start and [project] duration, which the tab rewrites in place, every other line of mk.toml as it was);
the brief keeps the rest:

brief/brief.json
  version      1
  language     "en" (the words are timed: mk's aligner hears English) or "other" (the beats only)
  video        story, cast, place, look, posting (the person's words); sizes (of SIZES); pictures
  parts        [{id, from, to, note, pictures, echo}]: spans of song seconds inside the clip that do not overlap (a line
               or a group of lines once the words are timed, or any stretch: an intro, a solo), what the person wants
               there, their pictures, and `echo` {of: an earlier part's id, how: "same" | "bigger"} for a part that
               comes back
  saved, sent  when the page last saved it, when Send was pressed
brief/pictures/   the pictures the page was given (PNG, JPEG, WebP, GIF), named by a hash of their bytes
audio/lyrics.txt  the lyrics of the part, one sung line per line, as `mk timeline analyze --lyrics` reads them: written
                  by the page, never sent back to it, never printed

Lines and words are numbers and times here, never text."""
import datetime
import filecmp
import functools
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

from .project import FILENAME, Project
from .site.music import TIMELINE                    # where `mk timeline analyze` writes a project's timeline

BRIEF = "brief/brief.json"
PICTURES = "brief/pictures"
LYRICS = "audio/lyrics.txt"
SIZES = {"9x16": (1080, 1920), "1x1": (1080, 1080), "16x9": (1920, 1080)}
LANGUAGES = ("en", "other")
HOW = ("same", "bigger")
VIDEO_TEXT = ("story", "cast", "place", "look", "posting")
TEXT_MAX = 20000                            # characters in one field
LYRICS_MAX = 200_000                        # bytes
PICTURE_MAX = 25 << 20                      # bytes
PICTURE_KINDS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif"}
PART_ID = re.compile(r"p[1-9][0-9]{0,3}")
MIN_PART_S = 0.1
MIN_CLIP_S = 1.0
FPS, PREROLL_S = 30, 3

MK_TOML = """# {comment}: a video made from its brief (the Brief tab: `mk site --tab brief`; brief/brief.json).
# No lyric text anywhere: words are (line, word) of audio/timeline.json.

[project]
name = {name}
fps = {fps}
{frame0:<30}# Blender frame of clip time 0, after {preroll} s of pre-roll
{duration:<30}# clip length, seconds: the part of the song (the Brief tab sets it)

[audio]
file = {file}
start = 0.0                   # song seconds at clip time 0 (the Brief tab sets it)
"""


class BriefError(ValueError):
    """A brief, a song or a project the brief cannot use; the message names every problem, one per line."""


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v):
    """A number as mk.toml writes it: at most three decimals, always a float."""
    s = f"{float(v):.3f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def mmss(t):
    """Song seconds as m:ss.s."""
    m, s = divmod(max(0.0, float(t)), 60.0)
    return f"{int(m)}:{s:04.1f}"


# ----------------------------------------------------------------------------------------------------------- the song
@functools.lru_cache(maxsize=64)
def _seconds(path, size, mtime_ns):
    exe = shutil.which("ffprobe")
    if exe is None:
        raise BriefError("ffprobe is not on PATH (it comes with ffmpeg)")
    r = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=60)
    try:
        s = float(r.stdout.strip())
    except ValueError:
        s = 0.0
    if not s > 0:
        why = r.stderr.strip()[-200:]
        raise BriefError(f"{Path(path).name} cannot be read as audio" + (f": {why}" if why else ""))
    return s


def song_seconds(path):
    """The length of an audio file in seconds (ffprobe; cached by the file's size and time)."""
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return _seconds(str(p), st.st_size, st.st_mtime_ns)


def song(proj):
    """(path, seconds) of the project's [audio] file, or None when it has none or the file is missing."""
    f = (proj.data.get("audio") or {}).get("file")
    if not f or not proj.path(f).is_file():
        return None
    return proj.path(f), song_seconds(proj.path(f))


def clip(proj):
    """The part of the song the video is made of: (start, end) in song seconds."""
    start = float((proj.data.get("audio") or {}).get("start", 0.0))
    return start, start + float(proj.duration)


# ------------------------------------------------------------------------------------------------------- a new video
def new(root, song_file, name=None, lyrics=None):
    """Start a video at `root` from `song_file` (copied into audio/ unless it is that file already) and, when given,
    its lyrics: an mk.toml whose clip is the whole song, and an empty brief. Refuses a folder that is a project."""
    root = Path(root).expanduser().resolve()
    src = Path(song_file).expanduser().resolve()
    if (root / FILENAME).exists():
        raise BriefError(f"{root} is a project already: `mk site --tab brief` opens its brief")
    if not src.is_file():
        raise BriefError(f"no song {src}")
    seconds = song_seconds(src)
    text = None
    if lyrics:
        lp = Path(lyrics).expanduser()
        if not lp.is_file():
            raise BriefError(f"no lyrics file {lp}")
        text = lp.read_text(encoding="utf-8")
    dest = root / "audio" / src.name
    if dest.exists() and dest.resolve() != src and not filecmp.cmp(dest, src, shallow=False):
        raise BriefError(f"{dest} is there already and is another file")
    name = " ".join(str(name or root.name).split())
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    rel = dest.relative_to(root).as_posix()
    (root / FILENAME).write_text(MK_TOML.format(
        comment=name, name=json.dumps(name, ensure_ascii=False), fps=FPS, frame0=f"frame0 = {FPS * PREROLL_S + 1}",
        preroll=PREROLL_S, duration=f"duration = {_num(math.floor(seconds * 1000) / 1000)}",
        file=json.dumps(rel, ensure_ascii=False)), encoding="utf-8")
    write(root, empty())
    lines = set_lyrics(root, text) if text is not None else None
    return {"project": str(root), "mk_toml": str(root / FILENAME), "song": str(dest), "seconds": round(seconds, 3),
            "lyrics_lines": lines, "brief": str(root / BRIEF)}


# --------------------------------------------------------------------------------------------------------- the clip
_HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def _set_value(text, table, key, value):
    """`text` (a TOML file) with `key = value` in `[table]`: the value of its line replaced (its comment kept), or a
    line added under the table's header. Raises BriefError when the file has no such table."""
    lines = text.splitlines(keepends=True)
    head = next((i for i, ln in enumerate(lines) if (m := _HEADER.match(ln)) and m.group(1).strip() == table), None)
    if head is None:
        raise BriefError(f"{FILENAME} has no [{table}] table: set [{table}] {key} = {_num(value)} by hand")
    end = next((j for j in range(head + 1, len(lines)) if lines[j].lstrip().startswith("[")), len(lines))
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\r\n]*?)(\s*(?:#[^\r\n]*)?)(\r?\n)?$")
    for j in range(head + 1, end):
        m = pat.match(lines[j])
        if m:
            lines[j] = m.group(1) + _num(value) + m.group(3) + (m.group(4) or "")
            return "".join(lines)
    if not lines[head].endswith("\n"):
        lines[head] += "\n"
    lines.insert(head + 1, f"{key} = {_num(value)}\n")
    return "".join(lines)


def set_clip(root, start, end):
    """Make the stretch of the song from `start` to `end` (song seconds, at least MIN_CLIP_S long) the project's clip:
    [audio] start and [project] duration rewritten in mk.toml, every other line as it was. Returns (start, end)."""
    proj = Project.load(root)
    s = song(proj)
    if s is None:
        raise BriefError("the project has no [audio] file, so it has no song to take a part of")
    seconds = s[1]
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        raise BriefError("a part is two numbers, start and end, in song seconds")
    if not (0.0 <= start < end <= seconds + 0.05):
        raise BriefError(f"the part {start:.2f}-{end:.2f} s is not inside the song (0-{seconds:.2f} s)")
    end = min(end, math.floor(seconds * 1000) / 1000)
    if end - start < MIN_CLIP_S:
        raise BriefError(f"a part is at least {MIN_CLIP_S:g} s long")
    start, dur = round(start, 3), round(end - start, 3)
    path = proj.root / FILENAME
    old = path.read_text(encoding="utf-8")
    text = _set_value(_set_value(old, "audio", "start", start), "project", "duration", dur)
    try:
        data = tomllib.loads(text)
        ok = abs(float(data["audio"]["start"]) - start) < 1e-9 and abs(float(data["project"]["duration"]) - dur) < 1e-9
    except (tomllib.TOMLDecodeError, KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        raise BriefError(f"{FILENAME} cannot be rewritten in place: set [audio] start = {_num(start)} and [project] "
                         f"duration = {_num(dur)} by hand")
    path.write_text(text, encoding="utf-8")
    return start, round(start + dur, 3)


def shots(proj):
    """How many [[shot]] entries the project has: the clip carries them."""
    return len(proj.data.get("shot") or [])


# ------------------------------------------------------------------------------------------- lyrics and pictures
def lyrics_lines(root):
    """How many sung lines the project's lyrics file has, or None without one."""
    p = Path(root) / LYRICS
    if not p.is_file():
        return None
    return sum(1 for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip())


def set_lyrics(root, text):
    """Keep the lyrics of the part (one sung line per line; blank lines dropped) for `mk timeline analyze --lyrics`;
    empty text removes them. Returns how many lines there are, never the text."""
    if not isinstance(text, str):
        raise BriefError("lyrics are text, one sung line per line")
    if len(text.encode("utf-8")) > LYRICS_MAX:
        raise BriefError(f"lyrics are at most {LYRICS_MAX // 1000} kB")
    lines = [ln for ln in (x.strip() for x in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")) if ln]
    p = Path(root) / LYRICS
    if not lines:
        p.unlink(missing_ok=True)
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def add_picture(root, data):
    """Keep a picture the page was given (its bytes) in brief/pictures/, named by a hash of them; returns its path in
    the project. Anything but a PNG, JPEG, WebP or GIF image is refused."""
    from PIL import Image
    if not data:
        raise BriefError("an empty picture")
    if len(data) > PICTURE_MAX:
        raise BriefError(f"a picture is at most {PICTURE_MAX >> 20} MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            kind = im.format
            im.verify()
    except (OSError, SyntaxError, ValueError) as e:
        raise BriefError(f"not a picture: {e}")
    if kind not in PICTURE_KINDS:
        raise BriefError(f"a {kind} picture: PNG, JPEG, WebP or GIF")
    rel = f"{PICTURES}/{hashlib.sha1(data).hexdigest()[:16]}{PICTURE_KINDS[kind]}"
    out = Path(root) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return rel


# ------------------------------------------------------------------------------------------------------- the brief
def empty():
    return {"version": 1, "language": "en", "video": {**{k: "" for k in VIDEO_TEXT}, "sizes": [], "pictures": []},
            "parts": [], "saved": None, "sent": None}


def read(root):
    """The brief of the project at `root` (an empty one when it has none yet)."""
    p = Path(root) / BRIEF
    doc = empty()
    if not p.is_file():
        return doc
    try:
        got = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BriefError(f"{BRIEF} cannot be read: {e}")
    if not isinstance(got, dict):
        raise BriefError(f"{BRIEF} is not a brief")
    doc.update({k: v for k, v in got.items() if k in doc})
    doc["video"] = {**empty()["video"], **(got.get("video") if isinstance(got.get("video"), dict) else {})}
    return doc


def write(root, doc):
    p = Path(root) / BRIEF
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def _pictures(root, given, where, problems):
    if not isinstance(given, list):
        problems.append(f"{where}: pictures are a list of paths in {PICTURES}/")
        return []
    out, base = [], (Path(root) / PICTURES).resolve()
    for x in given:
        p = (Path(root) / str(x)).resolve() if isinstance(x, str) else None
        if p is None or p.parent != base or not p.is_file():
            problems.append(f"{where}: {x!r} is not a picture in {PICTURES}/")
        elif str(x) not in out:
            out.append(str(x))
    return out


def _text(v, where, problems):
    if not isinstance(v, str):
        problems.append(f"{where} is text")
        return ""
    if len(v) > TEXT_MAX:
        problems.append(f"{where} is at most {TEXT_MAX} characters")
    return v


def check(given, root, seconds):
    """The editable part of a brief ({language, video, parts}) checked against the song (`seconds` long, None without
    one): parts sorted by time. A part may lie outside the clip (the part of the song changed after it was written):
    `show` says so. Raises BriefError naming every problem."""
    if not isinstance(given, dict):
        raise BriefError("a brief is {language, video, parts}")
    problems = [f"unknown key {k!r} (a brief has language, video and parts)" for k in given
                if k not in ("language", "video", "parts")]
    language = given.get("language", "en")
    if language not in LANGUAGES:
        problems.append(f"language {language!r}: one of {', '.join(LANGUAGES)}")
    v = given.get("video", {})
    video = {**{k: "" for k in VIDEO_TEXT}, "sizes": [], "pictures": []}
    if not isinstance(v, dict):
        problems.append("video is {" + ", ".join(VIDEO_TEXT) + ", sizes, pictures}")
        v = {}
    for k, x in v.items():
        if k in VIDEO_TEXT:
            video[k] = _text(x, f"video {k}", problems)
        elif k == "sizes":
            if not isinstance(x, list) or not all(isinstance(s, str) and s in SIZES for s in x) or len(set(x)) < len(x):
                problems.append(f"video sizes: each one of {', '.join(SIZES)}, once")
            else:
                video["sizes"] = list(x)
        elif k == "pictures":
            video["pictures"] = _pictures(root, x, "video", problems)
        else:
            problems.append(f"video: unknown key {k!r}")
    raw = given.get("parts", [])
    if not isinstance(raw, list):
        problems.append("parts are a list")
        raw = []
    if raw and seconds is None:
        problems.append("the project has no song, so it has no parts")
        raw = []
    parts, ids = [], set()
    for k, p in enumerate(raw, 1):
        where = f"part {k}"
        if not isinstance(p, dict):
            problems.append(f"{where} is {{id, from, to, note, pictures, echo}}")
            continue
        extra = set(p) - {"id", "from", "to", "note", "pictures", "echo"}
        if extra:
            problems.append(f"{where}: unknown {', '.join(sorted(map(repr, extra)))}")
        pid = p.get("id")
        if not isinstance(pid, str) or not PART_ID.fullmatch(pid) or pid in ids:
            problems.append(f"{where}: id {pid!r} (p1, p2 ...: once each)")
        if isinstance(pid, str):
            ids.add(pid)
        a, b = p.get("from"), p.get("to")
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (a, b)):
            problems.append(f"{where}: from and to are song seconds")
            continue
        if b - a < MIN_PART_S or a < -0.01 or b > seconds + 0.05:
            problems.append(f"{where}: {mmss(a)}-{mmss(b)} is not a stretch of the song (0:00.0-{mmss(seconds)})")
        echo = p.get("echo")
        if echo is not None and (not isinstance(echo, dict) or set(echo) != {"of", "how"} or echo["how"] not in HOW
                                 or not isinstance(echo["of"], str)):
            problems.append(f"{where}: echo is {{of: an earlier part's id, how: {' or '.join(map(repr, HOW))}}}")
            echo = None
        parts.append({"id": pid, "from": round(float(a), 3), "to": round(float(b), 3),
                      "note": _text(p.get("note", ""), f"{where} note", problems),
                      "pictures": _pictures(root, p.get("pictures", []), where, problems),
                      "echo": {"of": echo["of"], "how": echo["how"]} if echo else None})
    parts.sort(key=lambda p: p["from"])
    for prev, p in zip(parts, parts[1:]):
        if p["from"] < prev["to"] - 1e-6:
            problems.append(f"{p['id']} ({mmss(p['from'])}) overlaps {prev['id']} (until {mmss(prev['to'])})")
    for i, p in enumerate(parts):
        if p["echo"] and p["echo"]["of"] not in {q["id"] for q in parts[:i]}:
            problems.append(f"{p['id']}: it comes back from {p['echo']['of']!r}, which is no part before it")
    if problems:
        raise BriefError("\n".join(problems))
    return {"language": language, "video": video, "parts": parts}


def save(root, given):
    """Keep what the page gives (see `check`); returns the brief as written."""
    proj = Project.load(root)
    s = song(proj)
    clean = check(given, proj.root, s[1] if s else None)
    doc = read(proj.root)
    doc.update(clean, saved=_now())
    write(proj.root, doc)
    return doc


# ---------------------------------------------------------------------------------------------- what the agent reads
def timeline(proj):
    """The project's timeline in song seconds, numbers only: {start, end (the stretch it was analysed on), bpm, beats,
    downbeats, lines: [{line, start, end, words: [[start, end], ...]}] (numbered as the file numbers them),
    words_source}; None without one."""
    try:
        tl = json.loads(proj.path(TIMELINE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(tl, dict):
        return None
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)          # noqa: E731
    au = tl.get("audio") if isinstance(tl.get("audio"), dict) else {}
    t0 = float(au["start"]) if num(au.get("start")) else clip(proj)[0]
    dur = float(au["duration"]) if num(au.get("duration")) else float(proj.duration)
    lines = []
    for i, ln in enumerate(tl.get("lines") or [], 1):
        words = [[round(t0 + float(w["start"]), 3), round(t0 + float(w["end"]), 3)] for w in ((ln or {}).get("words") or [])
                 if isinstance(w, dict) and num(w.get("start")) and num(w.get("end"))]
        if words:
            lines.append({"line": i, "start": words[0][0], "end": words[-1][1], "words": words})
    return {"start": round(t0, 3), "end": round(t0 + dur, 3), "bpm": float(tl["bpm"]) if num(tl.get("bpm")) else None,
            "beats": [round(t0 + float(b), 3) for b in tl.get("beats") or [] if num(b)],
            "downbeats": [round(t0 + float(b), 3) for b in tl.get("downbeats") or [] if num(b)],
            "lines": lines, "words_source": tl.get("words_source")}


def _covered(tl, a, b):
    """The lines, the first and last words (line, word) and the bars (from the downbeats) of song seconds a..b."""
    mid = lambda s, e: a <= (s + e) / 2.0 <= b                                       # noqa: E731
    lines = [ln["line"] for ln in tl["lines"] if mid(ln["start"], ln["end"])]
    words = [[ln["line"], j] for ln in tl["lines"] for j, (s, e) in enumerate(ln["words"], 1) if mid(s, e)]
    d = tl["downbeats"]
    bars = None
    if d:
        first = sum(1 for x in d if x <= a + 1e-3)                    # bar k starts at downbeat k; 0 is the pickup
        last = sum(1 for x in d if x < b - 1e-3)
        bars = [first, max(first, last)]
    return {"lines": [lines[0], lines[-1]] if lines else None, "words": [words[0], words[-1]] if words else None,
            "bars": bars}


def show(root):
    """What the agent reads (`mk brief show`): the brief with each part placed on the song and on the clip (song and
    clip seconds, `in_clip` false for a part the clip no longer holds) with the lines, words and bars it covers,
    pictures as paths, `open` (the stretches of the clip no part covers: the agent's to fill), the lyrics' line count
    and the timeline's numbers. Never lyric text."""
    proj = Project.load(Path(root).expanduser().resolve())
    root = proj.root
    doc = read(root)
    s = song(proj)
    doc.update(check({k: doc[k] for k in ("language", "video", "parts")}, root, s[1] if s else None))
    c0, c1 = clip(proj)
    tl = timeline(proj)
    parts = []
    index = {p["id"]: k for k, p in enumerate(doc["parts"], 1)}
    for k, p in enumerate(doc["parts"], 1):
        a, b = p["from"], p["to"]
        item = {"n": k, "id": p["id"], "song": [a, b], "clip": [round(a - c0, 3), round(b - c0, 3)],
                "in_clip": a >= c0 - 0.01 and b <= c1 + 0.01, "seconds": round(b - a, 3),
                **(_covered(tl, a, b) if tl else {"lines": None, "words": None, "bars": None}),
                "note": p["note"], "pictures": [str(root / x) for x in p["pictures"]], "echo": None}
        if p["echo"]:
            item["echo"] = {"part": index.get(p["echo"]["of"]), "id": p["echo"]["of"], "how": p["echo"]["how"]}
        parts.append(item)
    gaps, t = [], c0
    held = [(max(p["from"], c0), min(p["to"], c1)) for p in doc["parts"] if p["to"] > c0 and p["from"] < c1]
    for a, b in held + [(c1, c1)]:
        if a - t >= 0.5:
            gaps.append({"song": [round(t, 3), round(a, 3)], "clip": [round(t - c0, 3), round(a - c0, 3)]})
        t = max(t, b)
    n_lyrics = lyrics_lines(root)
    return {
        "project": str(root), "name": proj.name, "file": str(root / BRIEF), "saved": doc["saved"], "sent": doc["sent"],
        "song": {"file": str(s[0]), "seconds": round(s[1], 3)} if s else None,
        "clip": {"start": round(c0, 3), "end": round(c1, 3), "seconds": round(c1 - c0, 3),
                 "whole_song": bool(s) and c0 < 0.01 and c1 > s[1] - 0.05},
        "lyrics": {"file": str(root / LYRICS), "lines": n_lyrics} if n_lyrics is not None else None,
        "language": doc["language"],
        "timeline": None if tl is None else {
            "file": str(proj.path(TIMELINE)),
            "matches_clip": abs(tl["start"] - c0) < 0.01 and abs(tl["end"] - c1) < 0.01,
            "bpm": tl["bpm"], "beats": len(tl["beats"]), "bars": len(tl["downbeats"]), "lines": len(tl["lines"]),
            "words": sum(len(ln["words"]) for ln in tl["lines"]), "words_source": tl["words_source"]},
        "video": {**{k: doc["video"][k] for k in VIDEO_TEXT}, "sizes": list(doc["video"]["sizes"]),
                  "pictures": [str(root / x) for x in doc["video"]["pictures"]]},
        "parts": parts,
        "open": gaps,
    }


def message(root):
    """What the agent reads when the brief is sent: the part of the song, the video, each part with its lines and
    the person's note."""
    b = show(root)
    c, tl = b["clip"], b["timeline"]
    lines = [f"Brief sent: {b['name']} ({b['file']})."]
    head = f"The part: {mmss(c['start'])}-{mmss(c['end'])} of the song ({c['seconds']:.1f} s" + \
        (", the whole song)" if c["whole_song"] else ")")
    if tl and tl["lines"]:
        head += f", {tl['lines']} lines timed" + ("" if tl["matches_clip"] else " before the part changed")
    if b["video"]["sizes"]:
        head += "; sizes " + ", ".join(b["video"]["sizes"])
    lines.append(head + ".")
    v = b["video"]
    bits = [f"{label}: {' '.join(v[k].split())}" for k, label in
            (("story", "the story"), ("cast", "who's in it"), ("place", "where"), ("look", "the look"),
             ("posting", "posted on"))
            if v[k].strip()]
    if v["pictures"]:
        bits.append(f"{len(v['pictures'])} picture" + ("s" if len(v["pictures"]) > 1 else ""))
    if bits:
        lines.append("Video: " + "; ".join(bits) + ".")
    for p in b["parts"]:
        where = ""
        if p["lines"]:
            a, z = p["lines"]
            where = (f"line {a}" if a == z else f"lines {a}-{z}") + ", "
        where += f"{mmss(p['song'][0])}-{mmss(p['song'][1])}"
        if p["echo"]:
            how = "the same" if p["echo"]["how"] == "same" else "bigger"
            where += f", comes back from part {p['echo']['part']}, {how}"
        if not p["in_clip"]:
            where += ", outside the part of the song"
        pics = f" [{len(p['pictures'])} picture" + ("s]" if len(p["pictures"]) > 1 else "]") if p["pictures"] else ""
        lines.append(f"Part {p['n']} ({where}): {' '.join(p['note'].split()) or '(no note)'}{pics}")
    lines.append(f"`mk brief show --project {b['project']}` reads it all (times, lines, bars, pictures).")
    return "\n".join(lines)


def send(root):
    """Send the brief: it gets the time; returns the message the agent reads."""
    root = Project.load(root).root
    text = message(root)
    doc = read(root)
    doc["sent"] = _now()
    write(root, doc)
    return text
