"""The pillow cloud in a real `mk build` (the Blender half of the prop): a keyed `puff` drives the shape key of the pillow and
of the cord in its seam and the materials' mix, the evaluated surface is the pillow at 0 and the cloud at 1 (as the
layout's numbers say), the card has its use points and the prop passes the form guard. Skipped without Blender."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import pillow_cloud_layout as LAY
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

PROJECT = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "pl"
card = "library:pillow_cloud"
at = [0.0, 0.0, 1.0]

[[prop]]
name = "cl"
card = "library:pillow_cloud"
at = [2.0, 0.0, 1.0]
slots = { puff = 1.0, case = "love" }

[[key]]
target = "pl"
prop = "puff"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 1.0], [1.5, 1.1]]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("pillow")
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


def frame(t):
    return round(31 + 30 * t)


EVAL_BOUNDS = ('(lambda vs: [list(map(min, *vs)), list(map(max, *vs))])([v.co for v in bpy.data.objects["{}"]'
               '.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])')


def test_a_keyed_puff_drives_both_shape_keys_and_the_materials(built, capsys):
    ts = (0.0, 0.5, 1.0, 1.5)
    expr = ('[bpy.data.objects["pl_pillow"].data.shape_keys.key_blocks["cloud"].value, '
            'bpy.data.objects["pl_piping"].data.shape_keys.key_blocks["cloud"].value, '
            'bpy.data.materials["pl_case"].node_tree.nodes["mk_puff"].outputs[0].default_value, '
            'bpy.data.materials["pl_piping"].node_tree.nodes["mk_puff"].outputs[0].default_value]')
    got = query(built, expr, [frame(t) for t in ts], capsys)
    for t, vals in zip(ts, got):
        want = t if t <= 1.0 else 1.0 + 0.2 * (t - 1.0)
        assert vals == pytest.approx([want] * 4, abs=1e-5), t


def test_the_surface_is_the_pillow_at_zero_and_the_cloud_at_one(built, capsys):
    _, cloud = LAY.meshes()
    (lo0, hi0), (lo1, hi1) = query(built, EVAL_BOUNDS.format("pl_pillow"), [frame(0.0), frame(1.0)], capsys)
    assert np.subtract(hi0, lo0) == pytest.approx([LAY.W, LAY.T, LAY.H], abs=0.012)   # subdivided: a hair inside the cage
    assert lo1 == pytest.approx(cloud.min(0), abs=0.012) and hi1 == pytest.approx(cloud.max(0), abs=0.012)
    ((clo, chi),) = query(built, EVAL_BOUNDS.format("cl_pillow"), [frame(0.0)], capsys)   # a starting puff of 1
    assert clo == pytest.approx(lo1, abs=1e-4) and chi == pytest.approx(hi1, abs=1e-4)


def test_the_card_names_its_use_points_and_the_form_guard_is_quiet(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT, encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code == 0 and not [ln for ln in out["log"] if "WARNING" in str(ln)]
    rep = out["stages"]["props"]["pl"]
    assert rep["uses"] == {"anchor": ["hug"], "rest": ["hold_L", "hold_R"], "look": ["centre"]}
    assert rep["size"] == pytest.approx([LAY.W, LAY.T, LAY.H], abs=0.012)


def test_a_bad_starting_puff_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT.replace('slots = { puff = 1.0, case = "love" }', 'slots = { puff = 3 }'),
                                      encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "puff" in out["error"]
