"""The turntable in a real `mk build` (the Blender half of the prop: drivers, the keys stage's `rates`, the label's text):
a keyed `rpm` turns the platter by its integral, so a spin-up never jumps the record; the stylus is where the layout's
kinematics put it for the keyed `arm` and `cue` on a moved, turned and scaled prop; text on `"<prop>:label"` rides the
record; the speed lamps follow `rpm`. Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import turntable_layout as L
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, YAW, SCALE = (0.3, 0.2, 0.5), 30.0, 2.0
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "tt"
card = "library:turntable"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
yaw = {YAW}
scale = {SCALE}

[[prop]]
name = "tt2"
card = "library:turntable"
at = [2.0, 0.0, 0.0]
slots = {{ rpm = 33.3 }}

[[key]]
target = "tt"
prop = "rpm"
interp = "LINEAR"
keys = [[0.0, 0.0], [0.5, 0.0], [1.0, 45.0]]

[[key]]
target = "tt"
prop = "arm"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 1.0], [2.0, 1.2]]

[[key]]
target = "tt"
prop = "cue"
interp = "LINEAR"
keys = [[0.0, 1.0], [1.0, 1.0], [1.5, 0.0]]

[[text]]
name = "lbl"
on = "tt:label"
text = "A"
lift = 0.0003
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("turntable")
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


def turns_at(t):
    """The integral of the keyed rpm / 60: 0 until 0.5 s, a linear ramp to 45 at 1.0 s, then held."""
    if t <= 0.5:
        return 0.0
    if t <= 1.0:
        return 0.75 * (t - 0.5) ** 2
    return 0.1875 + 0.75 * (t - 1.0)


def frame(t):
    """The frame of clip second t; tests pick t on whole frames (t * 30 an integer)."""
    f = 31 + 30 * t
    assert abs(f - round(f)) < 1e-9, t
    return int(round(f))


def test_the_platter_turns_by_the_integral_of_the_keyed_rpm(built, capsys):
    ts = (0.0, 0.6, 0.7, 1.0, 1.5, 1.9)
    got = query(built, 'bpy.data.objects["tt_platter"].rotation_euler.z', [frame(t) for t in ts], capsys)
    for t, z in zip(ts, got):
        assert z == pytest.approx(-2.0 * math.pi * turns_at(t), abs=1e-4), t


def test_an_unkeyed_platter_turns_steadily_at_its_starting_rpm_both_sides_of_clip_zero(built, capsys):
    ts = (-0.5, 0.0, 1.0, 1.9)
    got = query(built, 'bpy.data.objects["tt2_platter"].rotation_euler.z', [frame(t) for t in ts], capsys)
    for t, z in zip(ts, got):
        assert z == pytest.approx(-2.0 * math.pi * 33.3 / 60.0 * t, abs=1e-4), t


def test_the_stylus_follows_arm_and_cue_on_a_moved_turned_scaled_prop(built, capsys):
    ts = (0.0, 0.5, 1.0, 1.2, 1.5, 1.9)
    got = query(built, 'list(bpy.data.objects["tt_stylus"].matrix_world.translation)', [frame(t) for t in ts], capsys)
    a = math.radians(YAW)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    for t, p in zip(ts, got):
        arm = t if t <= 1.0 else 1.0 + 0.2 * (t - 1.0)
        cue = 1.0 if t <= 1.0 else max(0.0, 1.0 - (t - 1.0) / 0.5)
        want = np.array(AT) + SCALE * (R @ L.stylus(arm, cue))
        assert p == pytest.approx(list(want), abs=2e-5), t


def test_the_label_s_text_rides_the_record(built, capsys):
    got = query(built, '[bpy.data.objects["lbl"].parent.name, list(bpy.data.objects["lbl"].matrix_world.translation)]',
                [frame(1.0), frame(1.5)], capsys)
    (par0, p0), (par1, p1) = got
    assert par0 == par1 == "tt_platter"
    a = math.radians(YAW)
    centre = np.array(AT) + SCALE * np.array([math.cos(a) * L.SPINDLE[0] - math.sin(a) * L.SPINDLE[1],
                                              math.sin(a) * L.SPINDLE[0] + math.cos(a) * L.SPINDLE[1], 0.0])
    d0, d1 = (np.array(p)[:2] - centre[:2] for p in (p0, p1))
    turned = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    want = -2.0 * math.pi * (turns_at(1.5) - turns_at(1.0))                            # clockwise: 0.375 of a turn
    assert math.remainder(turned - want, 2.0 * math.pi) == pytest.approx(0.0, abs=1e-3)
    assert np.linalg.norm(d0) == pytest.approx(np.linalg.norm(d1), abs=1e-6)          # round the spindle


def test_the_speed_lamps_follow_rpm(built, capsys):
    expr = ('[bpy.data.materials["tt_led33"].node_tree.nodes["mk_led33"].outputs[0].default_value, '
            'bpy.data.materials["tt_led45"].node_tree.nodes["mk_led45"].outputs[0].default_value]')
    (at0, at2) = query(built, expr, [frame(0.0), frame(1.5)], capsys)
    assert at0 == [1.0, 0.0] and at2 == [0.0, 1.0]                                     # stopped: the deck wakes at 33


def test_a_bad_starting_value_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT.replace('scale = 2.0', 'scale = 2.0\nslots = { arm = 3 }'), encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "arm is 0" in out["error"]
