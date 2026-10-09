"""Review files: decisions put to a person with pictures, choices and 3D models, answered in Tern's review block (the
mk plugin, tern/ in this repository) or anywhere that edits the answers file.

An agent writes NAME.review.toml beside its pictures and opens it beside its chat (`mk review open`): the block shows
each question with its pictures, a choice, a notes field and models to turn in Tern's 3D block. Send writes the
answers and posts them into the agent's chat; `mk review answers` reads them back, with the lines, boxes and notes drawn
over the pictures in a whiteboard (in millimetres on the model when the picture is a lab sheet).

NAME.review.toml
  [review]             title (required), text (Markdown)
  [[question]]         id (required: letters, digits, _ and -), ask (required), text, images, models, recommended (an
                       option id)
  [[question.option]]  id (required, as a question's, unique in its question), label (required), text, images, models
  images  paths relative to the review file (PNG, JPEG, WebP, GIF; others are shown as PNG previews). A lab sheet
          (`mk model lab`: its .json beside it) is measured: lines drawn over it come back in mm.
  models  {label, file}: a file Tern's 3D block opens (MODEL_FILES) or a .pmx, posed by `mk model glb`; or {label,
          spec, parts, set}: a spec (model.toml, base:NAME) built in-process. pose (default tpose), morph (a list of
          NAME or NAME=WEIGHT) and region (default body; head, hand, foot, arm, leg with side L or R: only that part,
          framed up close) apply to .pmx and spec models.

NAME.answers.json   the block's record: reply_to (the agent pane Send posts to), answers {question: {choice, notes}},
                    sent (when Send was pressed), message (what it posted)
NAME.marks.json     what was drawn over the pictures, in each picture's own pixels (written by the plugin from the
                    whiteboards it opened): lines {from, to}, boxes {box: x, y, w, h} and notes, each with its text
"""
import json
import re
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from .model import lab as LAB

SUFFIX = ".review.toml"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
SHOWN = {".png", ".jpg", ".jpeg", ".webp", ".gif"}      # what a Tern image node draws
MAX_SIDE = 4096                                         # ... at most this many pixels a side
PREVIEW_SIDE = 2048
MODEL_FILES = {".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx", ".usd", ".usdz", ".usda", ".usdc", ".3ds", ".wrl",
               ".x3d"}                                  # what Tern's 3D block opens


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
            problems.append(f"{where}: {p.name}: not a .pmx nor a file the 3D block opens ({' '.join(sorted(MODEL_FILES))})")
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


def load(path):
    """The review at `path`, checked and with every path absolute: what the block draws. Raises ReviewError naming
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
                   "models": [x for x in (_model(base, m, where, problems) for m in q.get("models", [])) if x]})
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
    """`load` plus the answers given so far: what the block starts from."""
    r = load(path)
    rec = _read_json(r["answers_file"]) or {}
    r["reply_to"] = rec.get("reply_to")
    r["answers"] = rec.get("answers") or {}
    r["sent"] = rec.get("sent")
    return r


def prepare(path, reply_to=None):
    """Check the review and start (or keep) its answers file, naming the agent pane Send posts to. Returns the answers
    file's path."""
    r = load(path)
    p = Path(r["answers_file"])
    rec = _read_json(p) or {"review": r["path"], "answers": {}, "sent": None, "message": None}
    if reply_to is not None:
        rec["reply_to"] = int(reply_to)
    rec.setdefault("reply_to", None)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------------------ answers
def _resolve_mark(mark, sheet_png, layout):
    """A mark drawn over a lab sheet, in model space: a line traced (inside or outside the outline, mm), a box's cell
    and size in mm, a note's cell and point."""
    out = {"kind": mark.get("kind"), "text": mark.get("text", "")}
    if mark.get("kind") == "line":
        a, b = np.asarray(mark["from"], float), np.asarray(mark["to"], float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a))) + 1)            # a point per pixel along it
        try:
            out.update(LAB.trace_points(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a), sheet_png, layout))
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
    """The marks drawn over the review's pictures: in each picture's pixels, and in model space (mm) on lab sheets."""
    doc = _read_json(marks_path(Path(path).expanduser().resolve())) or {}
    out = []
    for img in doc.get("images") or []:
        sheet = Path(img["path"])
        layout = _read_json(sheet.with_suffix(".json"))
        for m in img.get("marks") or []:
            item = {"image": sheet.name, **m}
            if layout and "cells" in layout:
                item["model_space"] = _resolve_mark(m, sheet, layout)
            out.append(item)
    return out


def answers(path):
    """The answers so far, each with its option's label, and the marks: what the agent reads back."""
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
            "answers": got, "marks": marks(path)}
