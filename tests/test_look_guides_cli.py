"""mk look --guides: thirds and the 5 % safe area are drawn on every rendered image (both scenes of an --ab pair), and
only when asked for."""
import json
from pathlib import Path

import pytest
from PIL import Image

from mkmmd import bridge
from mkmmd.cli import main as MAIN

W, H = 120, 90
FLAT = (40, 40, 40)
THIRD, SAFE = (156, 207, 216), (246, 193, 119)


def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(autouse=True)
def flat_renders(monkeypatch):
    def fake_look(op, args, **kw):
        Path(args["out"]).mkdir(parents=True, exist_ok=True)                  # the Blender op makes the folder
        path = Path(args["out"]) / "cut_sq_00007.png"
        Image.new("RGB", (W, H), FLAT).save(path)
        return [{"path": str(path), "view": "cut", "size": "sq", "frame": 7}]

    monkeypatch.setattr(bridge, "run", fake_look)


def near(px, colour, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(px, colour))


def drawn(path):
    """(on a thirds line, on the safe-area edge, in the open) pixels of a rendered image."""
    im = Image.open(path).convert("RGB")
    return im.getpixel((W // 3, H // 2)), im.getpixel((W // 2, round(H * 0.05))), im.getpixel((W // 2 + 7, H // 2 + 5))


def test_guides_draw_thirds_and_the_safe_area(tmp_path, capsys):
    code, res = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--guides", "--out", str(tmp_path / "o")],
                        capsys)
    assert code == 0
    third, safe, open_ = drawn(res["images"][0])
    assert near(third, THIRD) and near(safe, SAFE) and near(open_, FLAT)


def test_no_guides_leaves_the_render_alone(tmp_path, capsys):
    code, res = run_cli(["look", str(tmp_path / "scene.blend"), "--frames", "7", "--out", str(tmp_path / "o")], capsys)
    assert code == 0
    assert all(near(px, FLAT) for px in drawn(res["images"][0]))


def test_guides_reach_both_scenes_of_an_ab_pair(tmp_path, capsys):
    code, res = run_cli(["look", str(tmp_path / "a.blend"), "--frames", "7", "--guides", "--ab", str(tmp_path / "b.blend"),
                         "--out", str(tmp_path / "o")], capsys)
    assert code == 0
    for path in res["images"] + res["images_b"]:
        assert near(drawn(path)[0], THIRD)
