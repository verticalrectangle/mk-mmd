"""mk look --no-styles: the flag reaches the Blender `look` op as `styles: false` (the op's default is true), for the
scene and for the --ab scene alike."""
import json
from pathlib import Path

import pytest
from PIL import Image

from mkmmd import bridge
from mkmmd.cli import main as MAIN


def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_look(op, args, **kw):
        seen.append((op, args, kw.get("blend")))
        Path(args["out"]).mkdir(parents=True, exist_ok=True)                  # the Blender op makes the folder
        path = Path(args["out"]) / "cut_sq_00007.jpg"
        Image.new("RGB", (24, 24), (200, 100, 50)).save(path)
        return [{"path": str(path), "view": "cut", "size": "sq", "frame": 7}]

    monkeypatch.setattr(bridge, "run", fake_look)
    return seen


def test_looks_are_on_by_default(tmp_path, capsys, calls):
    code, _ = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--out", str(tmp_path / "o")], capsys)
    assert code == 0 and [c[0] for c in calls] == ["look"]
    assert calls[0][1]["styles"] is True


def test_no_styles_reaches_the_op(tmp_path, capsys, calls):
    code, _ = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--no-styles", "--out", str(tmp_path / "o")],
                      capsys)
    assert code == 0 and calls[0][1]["styles"] is False


def test_no_styles_applies_to_both_scenes_of_an_ab_pair(tmp_path, capsys, calls):
    code, _ = run_cli(["look", str(tmp_path / "a.blend"), "--frames", "7", "--no-styles", "--ab", str(tmp_path / "b.blend"),
                       "--out", str(tmp_path / "o")], capsys)
    assert code == 0 and len(calls) == 2
    assert [c[1]["styles"] for c in calls] == [False, False]
    assert [Path(c[1]["out"]).name for c in calls] == ["a", "b"]


def test_the_flag_is_documented(capsys):
    with pytest.raises(SystemExit):
        MAIN.main(["look", "--help"])
    text = capsys.readouterr().out
    assert "--no-styles" in text and "silhouette" in text
