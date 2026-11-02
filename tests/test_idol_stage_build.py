"""The idol stage in a real `mk build` (the Blender half of the prop: its driven Value nodes and the keys stage's `rates`):
the panels' and bars' materials follow a keyed `on` and the chase, a keyed `speed` is integrated into `chase` (a speed
change never jumps the chase), unkeyed the chase runs at the starting speed both sides of clip second 0, a [[key]] on
`chase` drives it directly, every object stands at identity under the root, and the stage builds with no form WARNING.
Skipped without Blender."""
import json
import os
from pathlib import Path

import pytest

from mkmmd import config as CFG
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
name = "st"
card = "library:idol_stage"
at = [0.3, 0.2, 0.0]
yaw = 30.0

[[prop]]
name = "st2"
card = "library:idol_stage"
at = [9.0, 0.0, 0.0]
slots = { speed = 0.8, steps = "sides" }

[[prop]]
name = "st3"
card = "library:idol_stage"
at = [-9.0, 0.0, 0.0]
slots = { steps = "none", panels = [{ shape = "star", at = [0.0, 1.6], size = 1.0, pattern = "blink" }] }

[[key]]
target = "st"
prop = "on"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 1.0]]

[[key]]
target = "st"
prop = "speed"
interp = "LINEAR"
keys = [[0.0, 0.5], [0.5, 0.5], [1.0, 1.5]]

[[key]]
target = "st3"
prop = "chase"
interp = "CONSTANT"
keys = [[0.0, 0.0], [1.0, 0.5]]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("idol_stage")
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
    return round(f)


def chase_at(t):
    """The integral of the keyed speed: 0.5 until 0.5 s, a linear ramp to 1.5 at 1.0 s, then held."""
    if t <= 0.5:
        return 0.5 * t
    if t <= 1.0:
        return 0.25 + 0.5 * (t - 0.5) + (t - 0.5) ** 2
    return 0.75 + 1.5 * (t - 1.0)


def node(prop_name, mat, value):
    return f'bpy.data.materials["{prop_name}_{mat}"].node_tree.nodes["mk_{value}"].outputs[0].default_value'


def test_the_panels_follow_a_keyed_on_and_the_bars_their_level(built, capsys):
    ts = (0.0, 0.5, 1.0, 1.5)
    got = query(built, f"[{node('st', 'led0', 'on')}, {node('st', 'led4', 'on')}, {node('st', 'pixels', 'bars')}]",
                [frame(t) for t in ts], capsys)
    for t, (a, b, bars) in zip(ts, got):
        assert a == pytest.approx(min(t, 1.0), abs=1e-5) and b == pytest.approx(a) and bars == pytest.approx(1.0), t


def test_a_keyed_speed_is_integrated_into_the_chase(built, capsys):
    ts = (0.0, 0.3, 0.6, 0.8, 1.0, 1.5, 1.9)
    expr = f'[bpy.data.objects["st"]["chase"], {node("st", "led0", "chase")}, {node("st", "pixels", "chase")}]'
    for t, vals in zip(ts, query(built, expr, [frame(t) for t in ts], capsys)):
        assert vals == pytest.approx([chase_at(t)] * 3, abs=1e-4), t


def test_an_unkeyed_chase_runs_at_the_starting_speed_both_sides_of_clip_zero(built, capsys):
    ts = (-0.5, 0.0, 1.0, 1.9)
    got = query(built, f"[{node('st2', 'led0', 'chase')}, {node('st2', 'led2', 'chase')}]", [frame(t) for t in ts],
                capsys)
    for t, (a, b) in zip(ts, got):
        assert a == pytest.approx(0.8 * t, abs=1e-5) and b == pytest.approx(a), t


def test_a_key_on_the_chase_drives_it_directly(built, capsys):
    got = query(built, node("st3", "led0", "chase"), [frame(0.5), frame(1.2)], capsys)
    assert got == pytest.approx([0.0, 0.5])


def test_every_object_stands_at_identity_under_the_root_and_every_driver_is_simple(built, capsys):
    expr = ('[(o.name, o.parent.name, list(o.location), list(o.rotation_euler), list(o.scale)) for o in bpy.data.objects '
            'if o.name.startswith("st_")]')
    (rows,) = query(built, expr, [frame(0.0)], capsys)
    names = sorted(r[0] for r in rows)
    assert names == sorted(["st_riser", "st_trim", "st_steps", "st_bars"] + [f"st_panel{i}" for i in range(5)])
    for name, parent, loc, rot, scale in rows:
        assert parent == "st" and loc == [0.0] * 3 and rot == [0.0] * 3 and scale == [1.0] * 3, name
    expr = ('[(m.name, fc.data_path, fc.driver.is_simple_expression and fc.driver.is_valid) for m in bpy.data.materials '
            'if m.name.startswith("st") and m.node_tree and m.node_tree.animation_data '
            'for fc in m.node_tree.animation_data.drivers]')
    (drivers,) = query(built, expr, [frame(0.0)], capsys)
    assert len(drivers) == 2 * (5 + 5 + 1) + 2 * 3 and all(ok for _m, _p, ok in drivers)   # chase + on / bars


def test_the_stage_builds_with_no_form_warning_and_reports_its_card(built, capsys):
    code, out = cli(["build", "--project", str(built), "--until", "props", "--out",
                     str(built / "build" / "props.blend")], capsys)
    assert code == 0 and not [ln for ln in out["log"] if "WARNING" in str(ln)]
    rep = out["stages"]["props"]
    assert rep["st"]["uses"] == {"rest": ["deck"], "feet": ["centre", "left", "right"], "look": ["centre", "heart"],
                                 "surface": ["heart"]}
    assert rep["st3"]["uses"] == {"rest": ["deck"], "feet": ["centre", "left", "right"], "look": ["centre"]}


def test_a_bad_option_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT.replace('steps = "sides"', 'steps = "back"'), encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "idol_stage: steps is one of" in out["error"]
