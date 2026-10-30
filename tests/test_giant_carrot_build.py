"""The giant carrot in a real `mk build` (the Blender half of the prop: its drivers): every bite key follows its bite's clip
second (a starting value in `slots` and a CONSTANT [[key]], as a gag keys them), and a chunk flies where
giant_carrot_layout.chunk_pose puts it on a moved and turned carrot (falling toward the world's floor), tiny before its
bite and after its flight. Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import giant_carrot_layout as LAY
from mkmmd.cli import main as MAIN
from mkmmd.core import shell as S


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, ROT = (0.3, -0.2, 1.0), (-150.0, 20.0, 35.0)
BITES = [0.2, 0.5, 0.9, LAY.NEVER]               # bite3 keyed CONSTANT from 1e4 to 0.9 at 0.9 s
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "c"
card = "library:giant_carrot"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
rot = [{ROT[0]}, {ROT[1]}, {ROT[2]}]
slots = {{ bite1 = {BITES[0]}, bite2 = {BITES[1]} }}

[[key]]
target = "c"
prop = "bite3"
interp = "CONSTANT"
keys = [[0.0, 10000.0], [0.9, 0.9]]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("giant_carrot")
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
    """The frame of clip second t; tests pick t on whole frames (t * 30 an integer)."""
    f = 31 + 30 * t
    assert abs(f - round(f)) < 1e-9, t
    return int(round(f))


def test_the_root_carries_the_clock_and_the_bites(built, capsys):
    (got,) = query(built, '[bpy.data.objects["c"][k] for k in ("f0", "fps", "bite1", "bite2", "bite4")]', [frame(0.0)],
                   capsys)
    assert got == pytest.approx([31.0, 30.0, BITES[0], BITES[1], LAY.NEVER])


def test_each_bite_key_carves_in_over_its_bite_time_from_its_clip_second(built, capsys):
    ts = (0.1, 0.2, 7 / 30, 0.3, 0.5, 16 / 30, 0.6, 0.8, 0.9, 28 / 30, 1.0, 1.9)
    expr = '[bpy.data.objects["c_body"].data.shape_keys.key_blocks[f"bite{k}"].value for k in (1, 2, 3, 4)]'
    got = query(built, expr, [frame(t) for t in ts], capsys)
    for t, vals in zip(ts, got):
        want = [LAY.key_value(k, t, BITES) for k in range(1, 5)]
        assert vals == pytest.approx(want, abs=1e-5), t
    assert got[2][0] == pytest.approx(2.0 / 3.0, abs=1e-5) and got[9][2] == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_a_chunk_flies_its_arc_on_a_moved_turned_carrot_and_falls_toward_the_floor(built, capsys):
    c = next(x for x in LAY.chunks() if x["bite"] == 1 and x["j"] == 0)
    ts = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
    expr = ('[list(bpy.data.objects["c_chunk1_0"].matrix_world.translation), '
            'bpy.data.objects["c_chunk1_0"].scale[0]]')
    got = query(built, expr, [frame(t) for t in ts], capsys)
    R = S.rot_matrix(*ROT)
    rx, ry = math.radians(ROT[0]), math.radians(ROT[1])
    for t, (p, s) in zip(ts, got):
        pos, _, scale = LAY.chunk_pose(c, t, BITES, rx, ry)
        assert p == pytest.approx(list(np.array(AT) + R @ pos), abs=2e-5), t
        assert s == pytest.approx(scale, abs=1e-6), t
    assert got[0][1] == got[-1][1] == pytest.approx(LAY.MIN_SCALE)                   # unseen before and after
    z = [p[2] for p, _ in got[1:6]]
    assert max(z) > z[0] and z[-1] < z[0]                                              # up, then down in the world


def test_a_keyed_bite_throws_its_own_chunks(built, capsys):
    got = query(built, '[bpy.data.objects[f"c_chunk3_{j}"].scale[0] for j in range(4)]', [frame(0.8), frame(1.0)],
                capsys)
    assert got[0] == pytest.approx([LAY.MIN_SCALE] * 4) and min(got[1]) > 0.9
