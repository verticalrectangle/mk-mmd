"""library:butterflies in a real `mk build` (the Blender half: the swarm's points, the geometry-nodes tree, the drivers on
its inputs, the keys stage's rates): the evaluated instance transforms of the swarm equal the numpy pose
(mkmmd.core.butterflies.pose) on several frames, with the rates (`spin`, `flap`), the formation weights, `bloom` and
`radial` keyed, `burst` pulsing on a kick lane and `hit` on a snare lane of an `[audio] hits` file; the clocks are the
integrals of the keyed rates; a `level` swarm stays level on a tilted root. Build and per-frame evaluation times are
printed (`pytest -s`). Skipped without Blender."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import butterflies as BS


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

SLOTS = {"count": 100, "seed": 5}
PROJECT = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 3.0
blend = "build/t.blend"

[audio]
hits = "audio/hits.json"

[[prop]]
name = "sw"
card = "library:butterflies"
at = [0.3, -0.2, 1.1]
yaw = 30
slots = { count = 100, seed = 5 }

[[prop]]
name = "lv"
card = "library:butterflies"
at = [3.0, 0.0, 1.0]
rot = [25.0, -15.0, 40.0]
slots = { formation = "orbit", count = 12, level = true }

[[key]]
target = "sw"
prop = "spin"
interp = "LINEAR"
keys = [[0.0, 0.05], [1.0, 0.4], [2.0, -0.1]]

[[key]]
target = "sw"
prop = "flap"
interp = "LINEAR"
keys = [[0.0, 3.0], [1.5, 7.0]]

[[key]]
target = "sw"
prop = "orbit"
interp = "LINEAR"
keys = [[0.0, 0.0], [1.0, 0.8]]

[[key]]
target = "sw"
prop = "drift"
interp = "LINEAR"
keys = [[1.0, 0.0], [2.0, 0.6]]

[[key]]
target = "sw"
prop = "bloom"
interp = "LINEAR"
keys = [[0.0, 0.2], [0.8, 1.0]]

[[key]]
target = "sw"
prop = "radial"
interp = "LINEAR"
keys = [[0.0, 0.0], [2.0, 0.7]]

[[key]]
target = "sw"
prop = "burst"
lane = "kick"
pulse = [[0.0, 0.0], [0.4, 0.9]]

[[key]]
target = "sw"
prop = "hit"
lane = "snare"
"""
HITS = {"kick": [0.5, 1.6], "snare": [0.3, 1.2]}
BAD = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 1.0
blend = "build/t.blend"

[[prop]]
name = "sw"
card = "library:butterflies"
slots = { orbit_tilt = 200 }
"""
TS = (0.2, 0.5, 0.6, 0.9, 1.3, 1.7, 2.2)          # clip seconds on whole frames: bloom, bursts, hits, every blend
NAMES = [k for k in BS.INPUTS if k != "t"]
# every instance of the swarm: its object, position, unit axes and scale (mk q prints 6 decimals: a hidden butterfly's
# 0.1 mm matrix would lose its axes)
SWARM = ('{"P": {k: float(bpy.data.objects["sw"][k]) for k in %r}, '
         '"W": [list(r) for r in bpy.data.objects["sw_swarm"].matrix_world], '
         '"I": [(i.instance_object.original.name, list(i.matrix_world.translation), '
         '[list(r) for r in i.matrix_world.to_3x3().normalized()], i.matrix_world.to_scale()[0]) '
         'for i in bpy.context.evaluated_depsgraph_get().object_instances '
         'if i.is_instance and i.parent and i.parent.original.name == "sw_swarm"]}') % NAMES


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("butterflies")
    (root / "mk.toml").write_text(PROJECT, encoding="utf-8")
    (root / "audio").mkdir()
    (root / "audio" / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
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
    f = 31 + 30 * t
    assert abs(f - round(f)) < 1e-9, t
    return int(round(f))


def test_the_instances_are_where_the_numpy_pose_puts_them(built, capsys):
    opts, _ = BS.options(SLOTS)
    L = BS.layout(opts)
    combos = {}
    for st, w in L.combos:
        for part in BS.PARTS:
            combos[f"sw_{st}{w}_{part}"] = ((st, w), part)
    for t, got in zip(TS, query(built, SWARM, [frame(t) for t in TS], capsys)):
        P = got["P"]
        ref = BS.pose(L, BS.inputs(P, t, P["beats"], P["turns"]))
        W = np.array(got["W"])
        seen = {part: 0 for part in BS.PARTS}
        have = {part: [None] * L.N for part in BS.PARTS}
        for name, loc, axes, scale in got["I"]:
            combo, part = combos[name]
            i = seen[part]
            seen[part] += 1
            assert L.combos[L.combo[i]] == combo, (t, name, i)               # its own style and colourway
            have[part][i] = (np.array(loc), np.array(axes), scale)
        assert seen == {part: L.N for part in BS.PARTS}, t
        for part in BS.PARTS:
            want = W @ ref[part]
            loc = np.array([h[0] for h in have[part]])
            axes = np.array([h[1] for h in have[part]])
            scale = np.array([h[2] for h in have[part]])
            w_scale = np.linalg.norm(want[:, :3, 0], axis=1)
            assert np.abs(loc - want[:, :3, 3]).max() < 1e-4, (t, part)                       # positions, m
            assert np.abs(axes - want[:, :3, :3] / w_scale[:, None, None]).max() < 1e-4, (t, part)   # axes
            assert scale == pytest.approx(w_scale, rel=1e-4, abs=1e-7), (t, part)
        if t == 0.2:
            assert 0.2 < P["bloom"] < 1.0 and P["burst"] == 0.0               # half grown, still
        if t == 0.6:
            assert P["burst"] == pytest.approx(0.225, abs=1e-6)               # a quarter of the way into the pulse


def test_the_clocks_are_the_integrals_of_the_keyed_rates(built, capsys):
    def beats(t):
        return 3.0 * t + 4.0 / 3.0 * t * t if t <= 1.5 else 7.5 + 7.0 * (t - 1.5)

    def turns(t):
        if t <= 1.0:
            return 0.05 * t + 0.175 * t * t
        if t <= 2.0:
            return 0.225 + 0.4 * (t - 1.0) - 0.25 * (t - 1.0) ** 2
        return 0.375 - 0.1 * (t - 2.0)

    ts = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    got = query(built, '[bpy.data.objects["sw"]["beats"], bpy.data.objects["sw"]["turns"], '
                       'bpy.data.objects["lv"]["beats"], bpy.data.objects["lv"]["turns"]]', [frame(t) for t in ts], capsys)
    for t, (b, tn, lb, ltn) in zip(ts, got):
        assert b == pytest.approx(beats(t), abs=1e-4) and tn == pytest.approx(turns(t), abs=1e-4), t
        assert lb == pytest.approx(3.0 * t, abs=1e-5) and ltn == pytest.approx(0.05 * t, abs=1e-5), t     # unkeyed


def test_a_hit_lane_keys_the_last_hit_s_time(built, capsys):
    ts = (0.5, 1.0, 1.3, 2.0)
    got = query(built, 'bpy.data.objects["sw"]["hit"]', [frame(t) for t in ts], capsys)
    assert got == pytest.approx([0.3, 0.3, 1.2, 1.2])


def test_a_level_swarm_keeps_level_and_turns_with_its_root(built, capsys):
    (M,) = query(built, '[list(r) for r in bpy.data.objects["lv_swarm"].matrix_world]', [frame(0.5)], capsys)
    R = np.array(M)[:3, :3]
    a = math.radians(40.0)
    assert R == pytest.approx(np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]),
                              abs=1e-5)


def test_a_frame_of_a_100_butterfly_swarm_evaluates_quickly(built, capsys):
    expr = '(lambda T: [(scene.frame_set(f), T.perf_counter())[1] for f in range(31, 122)])(__import__("time"))'
    (stamps,) = query(built, expr, [31], capsys)
    per = np.diff(stamps)
    print(f"\nbutterflies: a frame of 100 + 12 butterflies evaluates in {1000 * np.median(per):.1f} ms "
          f"(median of {len(per)}, max {1000 * per.max():.1f} ms)")
    assert np.median(per) < 0.020


def test_the_build_report(built, capsys):
    out_blend = built / "build" / "props.blend"
    code, out = cli(["build", "--project", str(built), "--until", "props", "--out", str(out_blend)], capsys)
    assert code == 0
    log = [line for line in out["log"] if "library:butterflies" in line or "props done" in line]
    print("\nbutterflies build:", "; ".join(log))
    assert out["stages"]["props"]["sw"]["uses"]["look"] == ["centre"]


def test_a_bad_option_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(BAD, encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "orbit_tilt is degrees" in out["error"]
