"""The speaker stack and the penlight in a real `mk build` (the Blender half of the two props): the stack builds with no
form WARNING, its shells cut (holes, port tunnels, horn mouth) and shared per kind; every cone pumps out with the root's
keyed `pump` exactly as speaker_stack_layout.drivers says, on a moved, turned and scaled stack; the penlight's tube
material reads the keyed `color` and `glow` through its driven Value nodes, holds the palette's colours in order and picks
one as penlight_layout.pick does; the cards land on the roots. Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import penlight_layout as PL
from mkmmd.blender.library.props import showkit as SK
from mkmmd.blender.library.props import speaker_stack_layout as SL
from mkmmd.cli import main as MAIN
from mkmmd.core import palette as PAL


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, YAW, SCALE = (0.4, -0.2, 0.0), 30.0, 1.5
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "st"
card = "library:speaker_stack"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
yaw = {YAW}
scale = {SCALE}
slots = {{ stack = ["sub", "sub", "top"], cap = "gold" }}

[[key]]
target = "st"
prop = "pump"
interp = "LINEAR"
keys = [[0.0, 0.0], [0.5, 1.0], [1.0, 0.25]]

[[prop]]
name = "pl"
card = "library:penlight"
at = [-0.5, 0.0, 1.0]
slots = {{ colors = ["love", "foam", "gold"], color = 1 }}

[[key]]
target = "pl"
prop = "color"
interp = "CONSTANT"
keys = [[0.0, 1], [0.5, 2], [1.0, 4]]

[[key]]
target = "pl"
prop = "glow"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 1.0]]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("show_props")
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
    return int(round(31 + 30 * t))


def test_the_stack_builds_clean_with_its_shells_cut_and_shared_per_kind(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT, encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code == 0, out.get("error")
    assert not [ln for ln in out["log"] if "WARNING" in ln]                       # within its form limit
    (got,) = query(tmp_path, '[[bpy.data.objects[f"st_cab{k}_{p}"].data.name for p in ("shell", "drivers")] '
                             'for k in range(3)] + [sorted(m.name for m in bpy.data.objects["st_cab0_shell"].data.materials),'
                             ' len(bpy.data.objects["st_cab0_shell"].data.polygons), bpy.data.objects["st"]["mk_use"]]',
                   [31], capsys)
    names, mats, faces, rest = got[:3], got[3], got[4], json.loads(got[5])["rest"][0]["name"]
    assert names[0] == names[1] != names[2]                                       # the two subs share their meshes
    assert mats == ["st_baffle", "st_cabinet", "st_port"]                         # the cuts' faces are the port's
    assert faces > len(SL.shell_mesh("sub").Q)                                    # cut, not the bare cage
    assert rest == "top"


def test_every_cone_pumps_out_with_the_keyed_pump_on_a_moved_turned_scaled_stack(built, capsys):
    m, d = SL.drivers("sub")
    i = int(np.argmax(np.abs(d[:, 1])))                                           # a vertex on a moving cone
    c = SL.placed(["sub", "sub", "top"])[1]
    expr = ('[bpy.data.objects["st_cab1_drivers"].data.shape_keys.key_blocks["pump"].value, '
            'list(bpy.data.objects["st_cab1_drivers"].matrix_world @ bpy.data.objects["st_cab1_drivers"]'
            f'.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices[{i}].co)]')
    got = query(built, expr, [frame(t) for t in (0.0, 0.5, 1.0)], capsys)
    a = math.radians(YAW)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    for (value, world), pump in zip(got, (0.0, 1.0, 0.25)):
        assert value == pytest.approx(pump, abs=1e-6)
        local = np.array([0.0, c["y0"], c["z0"]]) + m.V[i] + pump * d[i]
        assert world == pytest.approx(np.array(AT) + SCALE * R @ local, abs=2e-5)


def test_the_penlight_s_tube_reads_its_index_and_glow_and_holds_the_palette_s_colours(built, capsys):
    nodes = 'bpy.data.materials["pl_tube"].node_tree.nodes'
    expr = (f'[{nodes}["mk_color"].outputs[0].default_value, {nodes}["mk_glow"].outputs[0].default_value, '
            f'[list({nodes}[f"mk_color{{i}}"].outputs[0].default_value) for i in range(3)], '
            f'sorted((n.operation, round(n.inputs[1].default_value, 3)) for n in {nodes} if n.type == "MATH" '
            f'and n.operation in ("FLOORED_MODULO", "COMPARE")), bpy.data.objects["pl"]["mk_use"]]')
    got = query(built, expr, [frame(t) for t in (0.0, 0.5, 1.0)], capsys)
    pal = PAL.get()
    for (color, glow, rgbs, ops, use), want, g in zip(got, (1.0, 2.0, 4.0), (0.0, 0.5, 1.0)):
        grip = json.loads(use)["grip"]
        assert color == pytest.approx(want) and glow == pytest.approx(g, abs=1e-6)
        assert np.array(rgbs)[:, :3] == pytest.approx(np.array([SK.rgb(c, pal) for c in ("love", "foam", "gold")]),
                                                      abs=1e-5)
        assert ops == [["COMPARE", 1.0], ["COMPARE", 2.0], ["FLOORED_MODULO", 3.0]]   # round(color) mod 3 picks one
        assert grip[0]["name"] == "hold" and grip[0]["radius"] == PL.HANDLE_R
    assert [PL.pick(c, 3) for c in (1.0, 2.0, 4.0)] == [1, 2, 1]                  # foam, gold, foam again
