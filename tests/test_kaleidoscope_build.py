"""The kaleidoscope toy in a real `mk build` (the Blender half of the prop): its use points where the card puts them on a
moved, turned and scaled prop, the eyehole's empty an insert's anchor can name, the chamber and the view turning with a
keyed `turn`, the print packed into the .blend at its size, and a form the guard passes. Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import kaleidoscope as K


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, YAW, SCALE = (0.3, -0.2, 0.74), 30.0, 1.5
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[look]
palette = "rose-pine-dawn"

[[prop]]
name = "ks"
card = "library:kaleidoscope"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
yaw = {YAW}
scale = {SCALE}
slots = {{ pattern = "marbled", seed = 4 }}

[[key]]
target = "ks"
prop = "turn"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 90.0]]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaleidoscope")
    (root / "mk.toml").write_text(PROJECT, encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(root), "--until", "keys"])
    assert e.value.code == 0
    return root


def query(root, expr, frames, capsys):
    capsys.readouterr()
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", ",".join(map(str, frames)), "--project",
                   str(root)], capsys)
    assert code == 0, q
    return q["values"]


def world(p):
    a = math.radians(YAW)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return np.array(AT) + SCALE * (R @ np.asarray(p, float))


def test_the_eyehole_empty_is_where_the_card_puts_the_eyepiece_and_faces_out(built, capsys):
    card = K.card("ks")
    (loc, z), = query(built, '[list(obj("ks_eyehole").loc), list(obj("ks_eyehole").matrix.col[2][:3])]', [31], capsys)
    assert loc == pytest.approx(list(world(card["use"]["anchor"][0]["point"])), abs=2e-6)
    a = math.radians(YAW)
    out = np.array([math.sin(a), -math.cos(a), 0.0])                              # -Y of the prop, turned by the yaw
    assert np.asarray(z) / np.linalg.norm(z) == pytest.approx(list(out), abs=1e-6)


def test_the_toy_lies_on_its_rims_on_the_surface_it_was_put_on(built, capsys):
    (lo,), = query(built, '[min((o.matrix_world @ Vector(c)).z for o in bpy.data.objects if o.name.startswith("ks_") '
                          'and o.type == "MESH" and not o.hide_render for c in o.bound_box)]', [31], capsys)
    assert lo == pytest.approx(AT[2], abs=2e-4)


def test_a_keyed_turn_turns_the_chamber_and_the_view_about_the_axis(built, capsys):
    expr = ('[list(obj("ks_beads").loc), obj("ks_chamber").matrix.to_euler().y, '
            'bpy.data.objects["ks_view"].rotation_euler.y, bpy.data.objects["ks_chamber"].rotation_euler.y]')
    (b0, _, v0, c0), (b1, _, v1, c1) = query(built, expr, [31, 61], capsys)
    assert (c0, v0) == pytest.approx((0.0, 0.0)) and (c1, v1) == pytest.approx((math.pi / 2, math.pi / 2))
    axis = world(K.point(0.5 * (K.CHAMBER[0] + K.CHAMBER[1])))
    d0, d1 = np.asarray(b0) - axis, np.asarray(b1) - axis
    assert np.linalg.norm(d0) == pytest.approx(np.linalg.norm(d1), abs=1e-6)     # the beads swing round the axis


def test_the_print_is_packed_at_its_size_and_the_paper_shows_it(built, capsys):
    expr = ('[list(bpy.data.images["ks_print"].size), bpy.data.images["ks_print"].packed_file is not None, '
            '[n.image.name for n in bpy.data.materials["ks_paper"].node_tree.nodes if n.type == "TEX_IMAGE"]]')
    (size, packed, used), = query(built, expr, [31], capsys)
    w = round(2 * math.pi * K.TUBE_R * 1000 * 10)
    h = round((K.PAPER_S[1] - K.PAPER_S[0]) * 1000 * 10)
    assert size == [w, h] and packed and used == ["ks_print"]


def test_the_build_reports_the_card_and_no_form_warning(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT, encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code == 0
    assert not [line for line in out["log"] if "WARNING" in str(line) and "'ks'" in str(line)]   # the form guard passes
    prop = out["stages"]["props"]["ks"]
    assert prop["size"] == pytest.approx([2 * K.AXIS_Z, K.LENGTH, 2 * K.AXIS_Z], abs=1e-4)      # in its own frame
    assert prop["uses"] == {"look": ["eyepiece", "chamber", "grip"], "anchor": ["eyehole"], "rest": ["side"],
                            "grip": ["tube"]}
