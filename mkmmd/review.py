"""Review files: decisions put to a person with pictures, choices and 3D models, answered on the review page mk serves
(mkmmd/site: beside the agent in Tern, in any browser elsewhere) or anywhere that edits the answers file.

An agent writes NAME.review.toml beside its pictures and opens it (`mk review open`, with `--wait` to get the answers
back when they are sent): the page shows each question with its pictures, a choice, a notes field and models to turn
in its 3D viewer, and the person draws on the pictures. Send writes the answers and the message the agent reads;
`mk review answers` reads them back, with what was drawn over the pictures (in millimetres on the model when the
picture is a lab sheet).

NAME.review.toml
  [review]             title (required), text (Markdown)
  [[question]]         id (required: letters, digits, _ and -), ask (required), text, images, models, music,
                       recommended (an option id)
  [[question.option]]  id (required, as a question's, unique in its question), label (required), text, images, models
  images  paths relative to the review file (PNG, JPEG, WebP, GIF; others are shown as PNG previews). A lab sheet
          (`mk model lab`: its .json beside it) is measured: lines drawn over it come back in mm.
  models  {label, file}: a model file (MODEL_FILES: a .glb as it is, the others turned into one) or a .pmx, posed by
          `mk model glb`; or {label, spec, parts, set}: a spec (model.toml, base:NAME) built in-process. pose (default
          tpose), morph (a list of NAME or NAME=WEIGHT) and region (default body; head, hand, foot, arm, leg with side L
          or R: only that part, framed up close) apply to .pmx and spec models.
  music   a project folder (or its mk.toml), or {project, from, to}: the project's music timeline under the question
          (its audio, beats, hits, words, shots and effects; `from` and `to` in clip seconds show that span)

NAME.answers.json   the page's record: answers {question: {choice, notes}}, sent (when Send was pressed), message (what
                    the agent reads)
NAME.marks.json     what was drawn over the pictures, in each picture's own pixels: lines and arrows {from, to}, boxes
                    and ellipses {box: x, y, w, h}, strokes {points}, notes {box, text}, each with its colour
"""
import datetime
import json
import re
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from .model import lab as LAB

SUFFIX = ".review.toml"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
SHOWN = {".png", ".jpg", ".jpeg", ".webp", ".gif"}      # what a browser draws
MAX_SIDE = 4096                                         # ... at most this many pixels a side
PREVIEW_SIDE = 2048
MODEL_FILES = {".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx", ".usd", ".usdz", ".usda", ".usdc"}
# a .glb is viewed as it is; the others are turned into one by Blender's own importers (mkmmd.site.models)


class ReviewError(ValueError):
    """A review file that cannot be shown; the message lists every problem, one per line."""


def stem(path):
    name = Path(path).name
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else Path(path).stem


def answers_path(path):
    return Path(path).with_name(stem(path) + ".answers.json")


def marks_path(path):
    return Path(path).with_name(stem(path) + ".marks.json")


# ------------------------------------------------------------------------------------------------------------ reading
def _image(base, rel, where, problems, previews):
    p = (base / str(rel)).expanduser().resolve()
    if not p.is_file():
        problems.append(f"{where}: no image {rel}")
        return None
    try:
        with Image.open(p) as im:
            w, h = im.size
            shown = p
            if p.suffix.lower() not in SHOWN or max(w, h) > MAX_SIDE:
                previews.mkdir(parents=True, exist_ok=True)
                shown = previews / (p.stem + ".png")
                im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
                im.save(shown)
    except OSError as e:
        problems.append(f"{where}: {rel} is not an image ({e})")
        return None
    return {"path": str(p), "name": p.name, "w": w, "h": h, "show": str(shown),
            "sheet": p.with_suffix(".json").is_file()}


def _model(base, m, where, problems):
    if not isinstance(m, dict) or not m.get("label") or ("file" in m) == ("spec" in m):
        problems.append(f"{where}: a model is {{label, file}} or {{label, spec, ...}}")
        return None
    pose, region, side = m.get("pose", "tpose"), m.get("region", "body"), m.get("side", "L")
    if pose not in LAB.POSES or region not in LAB.REGIONS or side not in ("L", "R"):
        problems.append(f"{where}: pose {pose!r}, region {region!r}, side {side!r}: poses {', '.join(LAB.POSES)}; "
                        f"regions {', '.join(LAB.REGIONS)}; sides L, R")
        return None
    morphs = [str(x) for x in m.get("morph", [])]
    out = {"label": str(m["label"])}
    if "file" in m:
        p = (base / str(m["file"])).expanduser().resolve()
        if not p.is_file():
            problems.append(f"{where}: no model file {m['file']}")
            return None
        if p.suffix.lower() in MODEL_FILES:
            return {**out, "open": str(p)}
        if p.suffix.lower() != ".pmx":
            problems.append(f"{where}: {p.name}: not a .pmx nor a model file ({' '.join(sorted(MODEL_FILES))})")
            return None
        src, extra = str(p), []
    else:
        spec = str(m["spec"])
        src = spec if spec.startswith("base:") else str((base / spec).expanduser().resolve())
        if not spec.startswith("base:") and not Path(src).is_file():
            problems.append(f"{where}: no spec {spec}")
            return None
        extra = (["--parts", ",".join(m["parts"]) if isinstance(m["parts"], list) else str(m["parts"])]
                 if m.get("parts") else [])
        for kv in m.get("set", []):
            extra += ["--set", str(kv)]
    argv = ["model", "glb", src, "--pose", pose, "--region", region, "--side", side, *extra]
    for mo in morphs:
        argv += ["--morph", mo]
    return {**out, "glb": argv}


def _music(base, m, where, problems):
    """A question's music, checked: {project (absolute folder), from, to (clip seconds, when given)}, or None."""
    spec = {"project": m} if isinstance(m, str) else m
    if not isinstance(spec, dict) or not isinstance(spec.get("project"), str) or set(spec) - {"project", "from", "to"}:
        problems.append(f"{where}: music is a project folder or {{project, from, to}}")
        return None
    root = (base / spec["project"]).expanduser().resolve()
    if root.name == "mk.toml":
        root = root.parent
    if not (root / "mk.toml").is_file():
        problems.append(f"{where}: music: no mk.toml in {spec['project']}")
        return None
    out = {"project": str(root)}
    for k in ("from", "to"):
        if k in spec:
            if isinstance(spec[k], bool) or not isinstance(spec[k], (int, float)):
                problems.append(f"{where}: music `{k}` is clip seconds")
                return None
            out[k] = float(spec[k])
    if out.get("from", 0.0) < 0 or ("from" in out and "to" in out and out["to"] <= out["from"]):
        problems.append(f"{where}: music `from` and `to` must be a span of the clip (0 <= from < to)")
        return None
    return out


def load(path):
    """The review at `path`, checked and with every path absolute: what the page shows. Raises ReviewError naming
    every problem."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ReviewError(f"{path} does not exist")
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ReviewError(f"{path.name}: {e}")
    base, problems = path.parent, []
    previews = base / f".{stem(path)}.previews"
    head = doc.get("review") or {}
    if not head.get("title"):
        problems.append("[review] needs a title")
    unknown = set(doc) - {"review", "question"}
    if unknown:
        problems.append(f"unknown tables: {', '.join(sorted(unknown))} (a review has [review] and [[question]])")
    qs, seen = [], set()
    for qi, q in enumerate(doc.get("question") or [], 1):
        where = f"question {qi}"
        qid = str(q.get("id", ""))
        if not ID.fullmatch(qid):
            problems.append(f"{where}: id {qid!r}: letters, digits, _ and -, starting with a letter or digit")
        elif qid in seen:
            problems.append(f"{where}: id {qid!r} is used twice")
        seen.add(qid)
        if not q.get("ask"):
            problems.append(f"{where}: needs `ask`, the question itself")
        opts, oseen = [], set()
        for oi, o in enumerate(q.get("option") or [], 1):
            ow = f"{where} ({qid}), option {oi}"
            oid = str(o.get("id", ""))
            if not ID.fullmatch(oid) or oid in oseen:
                problems.append(f"{ow}: id {oid!r}: letters, digits, _ and -, unique in its question")
            oseen.add(oid)
            if not o.get("label"):
                problems.append(f"{ow}: needs a label")
            opts.append({"id": oid, "label": str(o.get("label", "")), "text": str(o.get("text", "")),
                         "images": [x for x in (_image(base, i, ow, problems, previews) for i in o.get("images", [])) if x],
                         "models": [x for x in (_model(base, m, ow, problems) for m in o.get("models", [])) if x]})
        rec = q.get("recommended")
        if rec is not None and str(rec) not in oseen:
            problems.append(f"{where} ({qid}): recommended {rec!r} is not one of its options")
        qs.append({"id": qid, "ask": str(q.get("ask", "")), "text": str(q.get("text", "")),
                   "recommended": None if rec is None else str(rec), "options": opts,
                   "images": [x for x in (_image(base, i, where, problems, previews) for i in q.get("images", [])) if x],
                   "models": [x for x in (_model(base, m, where, problems) for m in q.get("models", [])) if x],
                   "music": _music(base, q["music"], where, problems) if "music" in q else None})
    if not qs:
        problems.append("no [[question]]")
    if problems:
        raise ReviewError("\n".join(problems))
    return {"path": str(path), "title": str(head["title"]), "text": str(head.get("text", "")), "questions": qs,
            "answers_file": str(answers_path(path)), "marks_file": str(marks_path(path))}


def _read_json(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def show(path):
    """`load` plus the answers given so far: what the page starts from."""
    r = load(path)
    rec = _read_json(r["answers_file"]) or {}
    r["answers"] = rec.get("answers") or {}
    r["sent"] = rec.get("sent")
    return r


def prepare(path):
    """Check the review and start (or keep) its answers file. Returns the answers file's path."""
    r = load(path)
    p = Path(r["answers_file"])
    if _read_json(p) is None:
        p.write_text(json.dumps({"review": r["path"], "answers": {}, "sent": None, "message": None}, ensure_ascii=False,
                                indent=1), encoding="utf-8")
    return p


def _record(r):
    return _read_json(r["answers_file"]) or {"review": r["path"], "answers": {}, "sent": None, "message": None}


def _write_record(r, rec):
    Path(r["answers_file"]).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def write_answers(path, given):
    """Keep the answers the page gives ({question: {choice, notes}}) for the review's questions: a choice is one of the
    question's options or none. Raises ReviewError naming what does not fit."""
    r = load(path)
    if not isinstance(given, dict):
        raise ReviewError("answers are {question id: {choice, notes}}")
    questions, out, problems = {q["id"]: q for q in r["questions"]}, {}, []
    for qid, a in given.items():
        q = questions.get(qid)
        if q is None or not isinstance(a, dict):
            problems.append(f"{qid!r}: not a question of this review" if q is None else f"{qid!r}: an answer is {{choice, notes}}")
            continue
        choice, notes = a.get("choice"), a.get("notes", "")
        if choice is not None and choice not in {o["id"] for o in q["options"]}:
            problems.append(f"{qid!r}: {choice!r} is not one of its options")
            continue
        if not isinstance(notes, str):
            problems.append(f"{qid!r}: notes are text")
            continue
        out[qid] = {"choice": choice, "notes": notes}
    if problems:
        raise ReviewError("\n".join(problems))
    rec = _record(r)
    rec["answers"] = out
    _write_record(r, rec)
    return out


def mark_count(path):
    doc = _read_json(marks_path(Path(path).expanduser().resolve())) or {}
    return sum(len(img.get("marks") or []) for img in doc.get("images") or [])


def message(path):
    """What the agent reads when the review is sent: each question with the choice made and the notes."""
    r = show(path)
    lines = [f"Review answered: {r['title']} ({r['path']})"]
    for i, q in enumerate(r["questions"], 1):
        a = r["answers"].get(q["id"]) or {}
        pick = "no choice"
        if a.get("choice"):
            pick = f"{a['choice']}: " + next((o["label"] for o in q["options"] if o["id"] == a["choice"]), "?")
        lines.append(f"{i}. {q['ask']} → {pick}")
        if a.get("notes"):
            lines.append("   notes: " + a["notes"].replace("\n", "\n   "))
    n = mark_count(path)
    lines.append(f"Marks drawn on the pictures: {n}. `mk review answers {r['path']}` reads them (in mm on lab sheets)."
                 if n else f"`mk review answers {r['path']}` reads them back.")
    return "\n".join(lines)


def send(path):
    """Send the review: its record gets the time and the message. Returns the message."""
    r = load(path)
    text = message(path)
    rec = _record(r)
    rec["sent"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["message"] = text
    _write_record(r, rec)
    return text


# -------------------------------------------------------------------------------------------------------------- marks
MARK_KINDS = ("line", "arrow", "box", "ellipse", "ink", "text")
COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def _check_mark(m, where):
    """A mark as the page draws it, checked: {kind, color, from/to | box | points, text}, in the picture's pixels."""
    if not isinstance(m, dict) or m.get("kind") not in MARK_KINDS:
        raise ReviewError(f"{where}: a mark's kind is one of {', '.join(MARK_KINDS)}")
    color = m.get("color") or "#e5383b"
    if not COLOR.fullmatch(str(color)):
        raise ReviewError(f"{where}: colour {color!r} is not #rrggbb")
    out = {"kind": m["kind"], "color": str(color).lower()}
    try:
        if m["kind"] in ("line", "arrow"):
            out["from"] = [round(float(v), 1) for v in m["from"]][:2]
            out["to"] = [round(float(v), 1) for v in m["to"]][:2]
        elif m["kind"] == "ink":
            out["points"] = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in m["points"]]
        else:
            x, y, bw, bh = (float(v) for v in m["box"])
            out["box"] = [round(min(x, x + bw), 1), round(min(y, y + bh), 1), round(abs(bw), 1), round(abs(bh), 1)]
    except (KeyError, TypeError, ValueError, IndexError):
        need = {"ink": "a stroke needs points", "line": "a line needs from and to", "arrow": "an arrow needs from and to"}
        raise ReviewError(f"{where}: " + need.get(m["kind"], f"a {m['kind']} needs a box [x, y, w, h]"))
    if m["kind"] == "ink" and len(out["points"]) < 2:
        raise ReviewError(f"{where}: a stroke has at least two points")
    if m.get("text") or m["kind"] == "text":
        out["text"] = str(m.get("text") or "")
    return out


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_marks(src, marks, out):
    """The picture `src` with `marks` drawn over it, written to `out` (PNG): what the agent looks at."""
    from PIL import ImageDraw
    with Image.open(src) as im:
        im = im.convert("RGB")
    W, H = im.size
    lw = max(2, round(max(W, H) / 380))
    draw = ImageDraw.Draw(im)
    font = _font(max(14, round(H / 38)))
    for m in marks:
        c = m["color"]
        if m["kind"] in ("line", "arrow"):
            a, b = np.asarray(m["from"], float), np.asarray(m["to"], float)
            draw.line([tuple(a), tuple(b)], fill=c, width=lw)
            d = b - a
            if m["kind"] == "arrow" and np.linalg.norm(d) > 0:
                d /= np.linalg.norm(d)
                size = 4 * lw + 8
                for s in (1.0, -1.0):
                    ang = s * np.radians(28)
                    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ -d
                    draw.line([tuple(b), tuple(b + r * size)], fill=c, width=lw)
        elif m["kind"] == "ink":
            draw.line([tuple(p) for p in m["points"]], fill=c, width=lw, joint="curve")
        elif m["kind"] in ("box", "ellipse"):
            x, y, bw, bh = m["box"]
            (draw.rectangle if m["kind"] == "box" else draw.ellipse)([x, y, x + bw, y + bh], outline=c, width=lw)
        if m.get("text"):
            x, y = (m["box"][:2] if "box" in m else m.get("to") or m["points"][-1])
            draw.text((x, y), m["text"], fill=c, font=font, stroke_width=max(2, lw // 2), stroke_fill="white")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def views_dir(path):
    return Path(path).expanduser().resolve().parent / f".{stem(path)}.views"


def views(path):
    """The views marked in the 3D viewer (`.<name>.views/`): each a picture of what was on screen, with the model and
    the camera that drew it."""
    out = []
    for png in sorted(views_dir(path).glob("view-*.png"), key=lambda p: int(p.stem.split("-")[1])):
        meta = _read_json(png.with_suffix(".json")) or {}
        with Image.open(png) as im:
            w, h = im.size
        out.append({"path": str(png), "name": png.name, "w": w, "h": h, "show": str(png), "sheet": False, **meta})
    return out


def add_view(path, png_bytes, meta):
    """Keep a picture of the 3D viewer (`meta`: model, label, question, camera) as the next view; returns it."""
    d = views_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.stem.split("-")[1]) for p in d.glob("view-*.png")), default=0)
    png = d / f"view-{n}.png"
    png.write_bytes(png_bytes)
    with Image.open(png) as im:
        im.verify()
    png.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return next(v for v in views(path) if v["path"] == str(png))


def write_marks(path, images):
    """Keep what was drawn over the review's pictures (`images`: [{path, marks}], a picture or a 3D view of the review
    each) and draw each marked picture for the agent (`.<name>.marks/`). Returns the marks file's contents."""
    r = load(path)
    known = {v["path"]: v for v in views(path)}
    for q in r["questions"]:
        for img in [*q["images"], *(i for o in q["options"] for i in o["images"])]:
            known[img["path"]] = img
    if not isinstance(images, list):
        raise ReviewError("marks are [{path, marks}]")
    out_dir = Path(r["path"]).parent / f".{stem(r['path'])}.marks"
    doc = {"saved": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "images": []}
    for k, item in enumerate(images):
        img = known.get(str(Path(str((item or {}).get("path", ""))).expanduser().resolve()))
        if img is None:
            raise ReviewError(f"image {k + 1}: {(item or {}).get('path')!r} is not a picture of this review")
        marks = [_check_mark(m, f"{img['name']}, mark {j + 1}") for j, m in enumerate(item.get("marks") or [])]
        if not marks:
            continue
        marked = out_dir / f"{len(doc['images']) + 1}-{Path(img['path']).stem}.png"
        render_marks(img["path"], marks, marked)
        doc["images"].append({"path": img["path"], "w": img["w"], "h": img["h"], "marks": marks, "marked": str(marked)})
    Path(r["marks_file"]).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def _along(points):
    """A point per pixel along a polyline."""
    pts = np.asarray(points, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a))))
        out.extend(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    return np.asarray(out)


def _resolve_mark(mark, sheet_png, layout):
    """A mark drawn over a lab sheet, in model space: a line, arrow or stroke traced (inside or outside the outline,
    mm), a box's or an ellipse's cell and size in mm, a note's cell and point."""
    out = {"kind": mark.get("kind"), "text": mark.get("text", "")}
    if mark.get("kind") in ("line", "arrow", "ink"):
        pts = mark["points"] if mark["kind"] == "ink" else [mark["from"], mark["to"]]
        try:
            out.update(LAB.trace_points(_along(pts), sheet_png, layout))
        except ValueError as e:
            out["problem"] = str(e)
        return out
    x, y, w, h = (float(v) for v in mark["box"])
    cell, inside = LAB.sheet_cell(layout, [[x + w / 2.0, y + h / 2.0]])
    if cell is None or not inside.any():
        out["problem"] = "not on a cell of the sheet"
        return out
    ppm = float(layout["px_per_m"])
    centre = LAB.cell_world(cell, layout, [[x + w / 2.0, y + h / 2.0]])[0]
    out.update(cell=cell["label"], model=cell["model"], view=cell["view"], centre=[round(float(v), 5) for v in centre])
    if mark.get("kind") != "text":
        out.update(width_mm=round(w / ppm * 1000.0, 1), height_mm=round(h / ppm * 1000.0, 1),
                   region_mm=round(float((centre - np.asarray(cell["origin"], float)) @ np.asarray(cell["axis"], float))
                                   * 1000.0, 1))
    return out


def marks(path):
    """The marks drawn over the review's pictures: in each picture's pixels, and in model space (mm) on lab sheets; each
    names its marked-up picture (`marked`)."""
    doc = _read_json(marks_path(Path(path).expanduser().resolve())) or {}
    out = []
    for img in doc.get("images") or []:
        sheet = Path(img["path"])
        layout = _read_json(sheet.with_suffix(".json"))
        for m in img.get("marks") or []:
            item = {"image": sheet.name, **m, "marked": img.get("marked")}
            if layout and "cells" in layout:
                item["model_space"] = _resolve_mark(m, sheet, layout)
            out.append(item)
    return out


def answers(path):
    """The answers so far, each with its option's label, the marks and the views kept from the 3D viewer: what the
    agent reads back."""
    r = show(path)
    got, missing = [], []
    for q in r["questions"]:
        a = r["answers"].get(q["id"]) or {}
        label = next((o["label"] for o in q["options"] if o["id"] == a.get("choice")), None)
        if q["options"] and not a.get("choice"):
            missing.append(q["id"])
        got.append({"id": q["id"], "ask": q["ask"], "choice": a.get("choice"), "label": label,
                    "recommended": q["recommended"], "notes": a.get("notes", "")})
    return {"review": r["path"], "title": r["title"], "sent": r["sent"], "complete": not missing, "unanswered": missing,
            "answers": got, "marks": marks(path),
            "views": [{k: v[k] for k in ("path", "model", "label", "question", "camera") if k in v} for v in views(path)]}
