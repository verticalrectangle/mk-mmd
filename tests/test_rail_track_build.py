"""library:rail_track in a real `mk build` (the Blender half: the shared meshes, the drivers on `laid`, the keys stage's
counting key): on several frames every bed piece, tie, spike set and rail span stands where the numpy schedule
(mkmmd.core.rail_track) puts it, `laid` counts the hits of a lane with a pulse shaping each step (mkmmd.core.lanes.steps),
and the front's empty rides `laid` round a curve. Skipped without Blender."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import rail_track_layout as LAY
from mkmmd.cli import main as MAIN
from mkmmd.core import lanes as LN
from mkmmd.core import rail_track as RT


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
duration = 3.0
blend = "build/t.blend"

[audio]
hits = "audio/hits.json"

[[prop]]
name = "tr"
card = "library:rail_track"
slots = { length = 10, seed = 2 }

[[prop]]
name = "cv"
card = "library:rail_track"
at = [0.0, 6.0, 0.0]
slots = { length = 7, radius = -80, plates = true, laid = 0 }

[[key]]
target = "tr"
prop = "laid"
lane = "chug"
step = 1.2192
start = 0
pulse = [[0.0, 0.0], [0.2, 1.0]]

[[key]]
target = "cv"
prop = "laid"
interp = "LINEAR"
keys = [[0.0, 0.0], [2.0, 8.5344]]
"""
HITS = {"chug": [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9]}
STEP, PULSE = 1.2192, [[0.0, 0.0], [0.2, 1.0]]
TS = (0.0, 0.2, 0.3, 0.5, 0.6, 1.0, 1.4, 1.5, 2.0, 2.4, 2.9)
STATE = ('{p: {"laid": bpy.data.objects[p]["laid"], "o": {o.name: [list(o.matrix_world.translation), '
         'list(o.rotation_euler), list(o.scale)] for o in bpy.data.objects if o.parent and o.parent.name == p}} '
         'for p in ("tr", "cv")}')


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("rail_track")
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
    return round(f)


def laid_at(keys, t):
    ts, vs = zip(*keys)
    return float(np.interp(t, ts, vs))


def check(plan, opts, got, at=(0.0, 0.0, 0.0)):
    """Every part of a built track against the numpy schedule at its `laid`."""
    laid, o = got["laid"], got["o"]
    name = next(iter(o)).split("_")[0]
    at = np.array(at)
    variety = RT.tie_variety(plan, opts["seed"], LAY.VARIANTS, opts["plates"])
    assert sum(k.startswith(f"{name}_tie") for k in o) == plan.n == sum(k.startswith(f"{name}_spikes") for k in o)
    for k in range(plan.n):
        sx, syz = RT.bed_scale(plan, k, laid)
        loc, rot, scale = o[f"{name}_bed{k}"]
        assert scale == pytest.approx([float(sx), float(syz), float(syz)], abs=1e-6), (laid, k)
        assert loc == pytest.approx(list(plan.point([k * RT.SPACING])[0] + at), abs=1e-5)
    for i in range(plan.n):
        h, tip, shown = RT.tie_state(plan, i, laid, RT.TILT * variety[i][4])
        loc, rot, scale = o[f"{name}_tie{i}"]
        assert loc[2] == pytest.approx(LAY.TIE_TOP + float(h), abs=1e-5) and rot[0] == pytest.approx(float(tip), abs=1e-5)
        assert scale == pytest.approx([1.0 if shown else RT.MIN_SCALE] * 3, abs=1e-6), (laid, i)
        loc, rot, scale = o[f"{name}_spikes{i}"]
        assert scale[2] == pytest.approx(float(RT.spike_scale(plan, i, laid)), abs=1e-5)
    for j, (r, e) in enumerate(plan.spans):
        start, lift, shown = RT.rail_state(plan, j, laid)
        loc, rot, scale = o[f"{name}_rail{j}"]
        assert loc[2] == pytest.approx(LAY.seat_z(opts["plates"]) + float(lift), abs=1e-5)
        if plan.k == 0.0:
            assert loc[0] == pytest.approx(float(start) + at[0], abs=1e-5)
        else:
            assert rot[2] == pytest.approx(plan.k * float(start), abs=1e-6)
            assert loc[:2] == pytest.approx(list(np.array(plan.centre) + at[:2]), abs=1e-5)
        assert scale == pytest.approx([1.0 if shown else RT.MIN_SCALE] * 3, abs=1e-6)
    x, y, h = (float(v) for v in plan.pose(min(max(laid, 0.0), plan.length)))
    loc, rot, _ = o[f"{name}_front"]
    assert loc == pytest.approx([x + at[0], y + at[1], LAY.seat_z(opts["plates"]) + RT.RAIL_H + at[2]], abs=1e-5)
    assert rot[2] == pytest.approx(h, abs=1e-6)


def test_every_part_follows_the_schedule_on_laid(built, capsys):
    tr_opts, tr = LAY.options({"length": 10, "seed": 2})
    cv_opts, cv = LAY.options({"length": 7, "radius": -80, "plates": True, "laid": 0})
    keys = LN.steps(HITS["chug"], 0.0, STEP, PULSE)
    seen = set()
    for t, got in zip(TS, query(built, STATE, [frame(t) for t in TS], capsys)):
        assert got["tr"]["laid"] == pytest.approx(laid_at(keys, t), abs=1e-6), t           # the counting key
        assert got["cv"]["laid"] == pytest.approx(min(t / 2.0, 1.0) * 8.5344, abs=1e-6), t
        check(tr, tr_opts, got["tr"])
        check(cv, cv_opts, got["cv"], at=(0.0, 6.0, 0.0))
        seen.add(round(got["tr"]["laid"] / RT.SPACING, 3))
    assert {0.0, 2.0} <= seen                                                             # rests on whole spacings


def test_a_counting_key_needs_a_lane_and_numbers(tmp_path, capsys):
    bad = PROJECT.replace('lane = "chug"\nstep = 1.2192', 'step = "one tie"\nlane = "chug"')
    (tmp_path / "mk.toml").write_text(bad, encoding="utf-8")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "keys"], capsys)
    assert code != 0 and "step" in json.dumps(out)
