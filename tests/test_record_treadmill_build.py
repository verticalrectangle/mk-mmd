"""The record treadmill in a real `mk build` (the Blender half of the prop: the spin driver and the keys stage's `rates`):
a keyed `rpm` turns the platter by its integral, so a spin-up never jumps the record; unkeyed it turns steadily at its
starting rpm both sides of clip second 0; the record and its label ride the platter; text on `"<prop>:label"` rides it
too; a bad starting value stops the build with the prop named. Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, YAW = (0.3, 0.2, 0.0), 30.0
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[prop]]
name = "rec"
card = "library:record_treadmill"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
yaw = {YAW}

[[prop]]
name = "rec2"
card = "library:record_treadmill"
at = [3.0, 0.0, 0.0]

[[prop]]
name = "rec3"
card = "library:record_treadmill"
at = [-3.0, 0.0, 0.0]
slots = {{ rpm = 45 }}

[[key]]
target = "rec"
prop = "rpm"
interp = "LINEAR"
keys = [[0.0, 0.0], [0.5, 0.0], [1.0, 33.3]]

[[text]]
name = "lbl"
on = "rec:label"
text = "A"
lift = 0.0003
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("record_treadmill")
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
    """The integral of the keyed rpm / 60: 0 until 0.5 s, a linear ramp to 33.3 at 1.0 s, then held."""
    k = 33.3 / 60.0 / 0.5                       # turns per second, per second, on the ramp
    if t <= 0.5:
        return 0.0
    if t <= 1.0:
        return 0.5 * k * (t - 0.5) ** 2
    return 0.5 * k * 0.25 + 33.3 / 60.0 * (t - 1.0)


def frame(t):
    """The frame of clip second t; tests pick t on whole frames (t * 30 an integer)."""
    f = 31 + 30 * t
    assert abs(f - round(f)) < 1e-9, t
    return round(f)


def test_the_platter_turns_clockwise_by_the_integral_of_the_keyed_rpm(built, capsys):
    ts = (0.0, 0.6, 0.7, 1.0, 1.5, 1.9)
    got = query(built, 'bpy.data.objects["rec_platter"].rotation_euler.z', [frame(t) for t in ts], capsys)
    for t, z in zip(ts, got):
        assert z == pytest.approx(-2.0 * math.pi * turns_at(t), abs=1e-4), t


@pytest.mark.parametrize("name, rpm", [("rec2", 33.3), ("rec3", 45.0)])
def test_an_unkeyed_platter_turns_steadily_at_its_starting_rpm_both_sides_of_clip_zero(built, capsys, name, rpm):
    ts = (-0.5, 0.0, 1.0, 1.9)
    got = query(built, f'bpy.data.objects["{name}_platter"].rotation_euler.z', [frame(t) for t in ts], capsys)
    for t, z in zip(ts, got):
        assert z == pytest.approx(-2.0 * math.pi * rpm / 60.0 * t, abs=1e-4), t


def test_the_record_and_its_label_ride_the_platter(built, capsys):
    expr = '[bpy.data.objects[n].parent.name for n in ("rec_record", "rec_label")]'
    assert query(built, expr, [frame(1.0)], capsys) == [["rec_platter", "rec_platter"]]


def test_the_label_s_text_rides_the_record(built, capsys):
    got = query(built, '[bpy.data.objects["lbl"].parent.name, list(bpy.data.objects["lbl"].matrix_world.translation)]',
                [frame(1.0), frame(1.5)], capsys)
    (par0, p0), (par1, p1) = got
    assert par0 == par1 == "rec_platter"
    d0, d1 = (np.array(p)[:2] - np.array(AT[:2]) for p in (p0, p1))                  # about the spindle
    turned = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    want = -2.0 * math.pi * (turns_at(1.5) - turns_at(1.0))                            # clockwise
    assert math.remainder(turned - want, 2.0 * math.pi) == pytest.approx(0.0, abs=1e-3)
    assert np.linalg.norm(d0) == pytest.approx(np.linalg.norm(d1), abs=1e-6)


def test_a_bad_starting_value_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT.replace('slots = { rpm = 45 }', 'slots = { rpm = "fast" }'),
                                      encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "record_treadmill: rpm must be a number" in out["error"]
