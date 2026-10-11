"""Review files (mkmmd.review): a review is checked before the page shows it (every problem named at once), the answers
the page gives are checked (a bad one keeps nothing) and come back with their options' labels and what is still open,
reopening keeps them, model entries become commands mk accepts, and marks drawn over a lab sheet come back in
millimetres on the model."""
import json
from pathlib import Path

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


def test_a_review_names_every_problem_at_once(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "ok.png")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")
    f = write(tmp_path / "bad.review.toml", """
[review]
text = "no title"
[[question]]
id = "two words"
ask = "?"
music = "nowhere"
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
music = { project = ".", from = 3.0, to = 1.0 }
""")
    (tmp_path / "mk.toml").write_text('[project]\nfps = 30\nframe0 = 1\nduration = 4.0\n', encoding="utf-8")
    with pytest.raises(RV.ReviewError) as e:
        RV.load(f)
    problems = str(e.value).splitlines()
    for needle in ("needs a title", "'two words'", "'dup' is used twice", "no image missing.png", "needs a label",
                   "recommended 'Z'", "notes.txt: not a .pmx", "pose 'dance'", "music: no mk.toml in nowhere",
                   "music `from` and `to` must be a span"):
        assert sum(needle in p for p in problems) == 1, needle


def test_answers_name_the_chosen_option_and_what_is_left(tmp_path):
    f = two_questions(tmp_path)
    RV.prepare(f)
    RV.write_answers(f, {"eyes": {"choice": "B", "notes": "a small wing"}})
    got = RV.answers(f)
    assert not got["complete"] and got["unanswered"] == ["body"]
    eyes = got["answers"][0]
    assert (eyes["choice"], eyes["label"], eyes["notes"], eyes["recommended"]) == ("B", "Soft", "a small wing", "B")


def test_answers_that_do_not_fit_the_review_are_refused_and_nothing_is_kept(tmp_path):
    f = two_questions(tmp_path)
    RV.prepare(f)
    RV.write_answers(f, {"eyes": {"choice": "A", "notes": ""}})
    with pytest.raises(RV.ReviewError) as e:
        RV.write_answers(f, {"eyes": {"choice": "B"}, "body": {"choice": "Z"}, "hair": {"choice": None}})
    problems = str(e.value).splitlines()
    assert len(problems) == 2 and "'Z' is not one of its options" in problems[0] and "'hair'" in problems[1]
    assert RV.show(f)["answers"] == {"eyes": {"choice": "A", "notes": ""}}


def test_reopening_keeps_the_answers(tmp_path):
    f = two_questions(tmp_path)
    RV.prepare(f)
    RV.write_answers(f, {"eyes": {"choice": "A", "notes": "keep"}})
    RV.prepare(f)
    assert RV.show(f)["answers"]["eyes"] == {"choice": "A", "notes": "keep"}


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
    assert ready == {"label": "glb", "open": str((tmp_path / "ready.glb").resolve())}     # the viewer opens it as is
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
    RV.write_marks(f, [{"path": str(sheet), "marks": [
        {"kind": "line", "color": "#E5383B", "from": [50.0, 180.0], "to": [250.0, 180.0], "text": "too wide"},
        {"kind": "box", "box": [180.0, 140.0, -60.0, -30.0]},            # dragged up and left from (180, 140)
        {"kind": "ink", "points": [[150.0, 50.0], [150.0, 150.0]]},
        {"kind": "text", "box": [200.0, 200.0, 40.0, 10.0], "text": "here"}]}])
    got = RV.answers(f)["marks"]
    assert Path(got[0]["marked"]).is_file() and got[0]["color"] == "#e5383b"
    line, box, ink, note = (m["model_space"] for m in got)
    assert line["cell"] == "front" and line["length_mm"] == pytest.approx(200.0, abs=2.0)
    assert line["inset_mm"]["min"] < -40.0 and line["inset_mm"]["max"] > 40.0   # starts 50 mm outside, runs inside
    assert (box["width_mm"], box["height_mm"]) == (60.0, 30.0)
    assert box["region_mm"] == pytest.approx(1035.0, abs=0.5)                 # 35 mm above the cell's centre at 1 m
    assert ink["length_mm"] == pytest.approx(100.0, abs=2.0)
    assert note["cell"] == "front" and note["text"] == "here"


def test_a_mark_the_page_could_not_have_drawn_is_refused(tmp_path):
    sheet = lab_sheet(tmp_path)
    f = write(tmp_path / "m.review.toml", '[review]\ntitle = "Marks"\n[[question]]\nid = "q"\nask = "?"\n'
                                         'images = ["sheet.png"]\n')
    for marks, needle in (([{"kind": "ink", "points": [[1, 2]]}], "at least two points"),
                          ([{"kind": "box", "color": "red", "box": [0, 0, 1, 1]}], "not #rrggbb"),
                          ([{"kind": "star"}], "kind is one of")):
        with pytest.raises(RV.ReviewError, match=needle):
            RV.write_marks(f, [{"path": str(sheet), "marks": marks}])
    with pytest.raises(RV.ReviewError, match="not a picture of this review"):
        RV.write_marks(f, [{"path": str(tmp_path / "sheet.mask.png"), "marks": [{"kind": "text", "box": [0, 0, 1, 1]}]}])
    assert not RV.marks_path(f).exists()
