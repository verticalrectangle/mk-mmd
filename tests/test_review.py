"""Review files (mkmmd.review): a review is checked before the block shows it (every problem named at once), answers
come back with their options' labels and what is still open, reopening keeps them, model entries become commands mk
accepts, and what was drawn over a lab sheet comes back in millimetres on the model."""
import json

import pytest
from PIL import Image

from mkmmd import review as RV
from mkmmd.cli.main import build_parser

pytest.importorskip("cv2")


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def two_questions(tmp_path):
    return write(tmp_path / "pick.review.toml", """
[review]
title = "Pick"
[[question]]
id = "eyes"
ask = "Eyes?"
recommended = "B"
[[question.option]]
id = "A"
label = "Rin's"
[[question.option]]
id = "B"
label = "Soft"
[[question]]
id = "body"
ask = "Body?"
[[question.option]]
id = "fix"
label = "Fix it"
""")


def answer(path, answers):
    p = RV.answers_path(path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["answers"] = answers
    p.write_text(json.dumps(rec), encoding="utf-8")


def test_a_review_names_every_problem_at_once(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "ok.png")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")
    f = write(tmp_path / "bad.review.toml", """
[review]
text = "no title"
[[question]]
id = "two words"
ask = "?"
[[question]]
id = "dup"
ask = "first"
images = ["missing.png", "ok.png"]
recommended = "Z"
[[question.option]]
id = "A"
[[question]]
id = "dup"
ask = "second"
models = [{ label = "m", file = "notes.txt" }, { label = "p", spec = "base:girl", pose = "dance" }]
""")
    with pytest.raises(RV.ReviewError) as e:
        RV.load(f)
    problems = str(e.value).splitlines()
    for needle in ("needs a title", "'two words'", "'dup' is used twice", "no image missing.png", "needs a label",
                   "recommended 'Z'", "notes.txt: not a .pmx", "pose 'dance'"):
        assert sum(needle in p for p in problems) == 1, needle


def test_answers_name_the_chosen_option_and_what_is_left(tmp_path):
    f = two_questions(tmp_path)
    RV.prepare(f, reply_to=7)
    answer(f, {"eyes": {"choice": "B", "notes": "a small wing"}})
    got = RV.answers(f)
    assert not got["complete"] and got["unanswered"] == ["body"]
    eyes = got["answers"][0]
    assert (eyes["choice"], eyes["label"], eyes["notes"], eyes["recommended"]) == ("B", "Soft", "a small wing", "B")


def test_reopening_keeps_the_answers_and_names_the_new_agent_pane(tmp_path):
    f = two_questions(tmp_path)
    RV.prepare(f, reply_to=7)
    answer(f, {"eyes": {"choice": "A", "notes": ""}})
    RV.prepare(f, reply_to=12)
    shown = RV.show(f)
    assert shown["reply_to"] == 12 and shown["answers"]["eyes"]["choice"] == "A"


def test_model_entries_become_commands_mk_accepts(tmp_path):
    (tmp_path / "m.pmx").write_bytes(b"")
    (tmp_path / "ready.glb").write_bytes(b"")
    f = write(tmp_path / "models.review.toml", """
[review]
title = "Models"
[[question]]
id = "q"
ask = "Which?"
models = [{ label = "pmx", file = "m.pmx", pose = "arms_down", morph = ["笑い=0.5"] },
          { label = "glb", file = "ready.glb" },
          { label = "spec", spec = "base:girl", parts = ["head"], set = ["head.lash.width=0.007"], pose = "rest", region = "head" }]
""")
    pmx, ready, spec = RV.load(f)["questions"][0]["models"]
    assert ready == {"label": "glb", "open": str((tmp_path / "ready.glb").resolve())}     # the 3D block opens it as is
    parser = build_parser()
    a = parser.parse_args(pmx["glb"])
    assert (a.func.__name__, a.pose, a.morph) == ("run_glb", "arms_down", ["笑い=0.5"])
    a = parser.parse_args(spec["glb"])
    assert (a.pose, a.region, a.parts, a.overrides) == ("rest", "head", "head", ["head.lash.width=0.007"])


def lab_sheet(tmp_path):
    """A 400 x 300 lab sheet of one cell (x 0..400, y 20..300; 1000 px a metre; its centre is the model-space point
    (0, 0, 1)) whose drawn shape covers x 100..300, y 100..260."""
    Image.new("RGB", (400, 300), (222, 224, 230)).save(tmp_path / "sheet.png")
    mask = Image.new("L", (400, 300), 0)
    mask.paste(255, (100, 100, 300, 260))
    mask.save(tmp_path / "sheet.mask.png")
    cell = {"label": "front", "model": "m", "pose": "rest", "view": "front", "box": [0, 20, 400, 280],
            "centre": [0.0, 0.0, 1.0], "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0]}
    (tmp_path / "sheet.json").write_text(json.dumps({"px_per_m": 1000.0, "cells": [cell]}), encoding="utf-8")
    return tmp_path / "sheet.png"


def test_marks_drawn_over_a_lab_sheet_come_back_in_millimetres(tmp_path):
    sheet = lab_sheet(tmp_path)
    f = write(tmp_path / "m.review.toml", '[review]\ntitle = "Marks"\n[[question]]\nid = "q"\nask = "?"\n'
                                         'images = ["sheet.png"]\n')
    RV.marks_path(f).write_text(json.dumps({"images": [{"path": str(sheet), "w": 400, "h": 300, "marks": [
        {"kind": "line", "from": [50.0, 180.0], "to": [250.0, 180.0], "text": "too wide"},
        {"kind": "box", "shape": "rect", "box": [120.0, 110.0, 60.0, 30.0], "text": ""},
        {"kind": "text", "box": [200.0, 200.0, 40.0, 10.0], "text": "here"}]}]}), encoding="utf-8")
    line, box, note = (m["model_space"] for m in RV.answers(f)["marks"])
    assert line["cell"] == "front" and line["length_mm"] == pytest.approx(200.0, abs=2.0)
    assert line["inset_mm"]["min"] < -40.0 and line["inset_mm"]["max"] > 40.0   # starts 50 mm outside, runs inside
    assert (box["width_mm"], box["height_mm"]) == (60.0, 30.0)
    assert box["region_mm"] == pytest.approx(1035.0, abs=0.5)                 # 35 mm above the cell's centre at 1 m
    assert note["cell"] == "front" and note["text"] == "here"
