"""The parent cut-outs built for real (`mk build` of a tiny project, then `mk q`): the father's hand turns about its brad
by angle = wag + wagging * sin(2 pi wag_hz t) for a keyed `wag` and for the automatic `wagging`, its shadow with it, the
index fingertip landing where parent_cutout_layout.tip_at says; both figures build without a warning. Skipped without
Blender."""
import contextlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.blender.library.props import parent_cutout_layout as LAY
from mkmmd.cli import main as MAIN


def have_blender():
    try:
        return Path(CFG.load()["blender"]).exists() and not os.environ.get("MK_SKIP_BLENDER_TESTS")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not have_blender(), reason="Blender not available")

AT, YAW = (0.4, 0.3, 0.0), 15.0
WAG = [[0.0, -20.0], [1.0, 0.0], [2.0, 20.0], [2.2, 0.0]]
PROJECT = f"""[project]
name = "t"
fps = 30
frame0 = 31
duration = 3.0
blend = "build/t.blend"

[[prop]]
name = "dad"
card = "library:parent_cutout"
at = [{AT[0]}, {AT[1]}, {AT[2]}]
yaw = {YAW}

[[prop]]
name = "mom"
card = "library:parent_cutout"
at = [-1.5, 0.3, 0.0]
slots = {{ figure = "mother" }}

[[key]]
target = "dad"
prop = "wag"
interp = "LINEAR"
keys = {json.dumps(WAG)}

[[key]]
target = "dad"
prop = "wagging"
interp = "CONSTANT"
keys = [[0.0, 0.0], [2.25, 18.0]]
"""


def _cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def _wag(t):
    """The keyed `wag` at clip second t (LINEAR between the keys, held past the ends)."""
    ts, vs = zip(*WAG)
    return float(np.interp(t, ts, vs))


def _expected(frame):
    t = (frame - 31) / 30.0
    return LAY.angle(t, _wag(t), 18.0 if t >= 2.25 else 0.0, 2.5)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """(project folder, exit code, the build's JSON report)."""
    d = tmp_path_factory.mktemp("parents")
    (d / "mk.toml").write_text(PROJECT, encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(d), "--until", "keys"])
    return d, e.value.code, json.loads(buf.getvalue())


def test_the_hand_turns_by_the_keyed_wag_and_the_automatic_tick_tock(built, capsys):
    d, code, out = built
    capsys.readouterr()
    assert code == 0, out.get("error")
    L = LAY.build("father")
    local = [float(v) for v in np.asarray(L.tip) - np.asarray(L.pivot)]
    expr = ("(bpy.data.objects['dad_hand'].rotation_euler[1], bpy.data.objects['dad_hand_shadow'].rotation_euler[1], "
            f"list(bpy.data.objects['dad_hand'].matrix_world @ Vector({tuple(local)})), "
            "[bpy.data.objects['dad'][k] for k in ('wag_hz', 'f0', 'fps')])")
    frames = [31, 46, 61, 91, 101, 106, 112]
    code, q = _cli(["q", str(d / "build" / "t.blend"), expr, "--frames", ",".join(map(str, frames)),
                    "--project", str(d)], capsys)
    assert code == 0, q
    c, s = math.cos(math.radians(YAW)), math.sin(math.radians(YAW))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for f, (rot, ghost, tip, props) in zip(frames, q["values"]):
        a = _expected(f)
        assert rot == pytest.approx(-math.radians(a), abs=1e-5), f            # rotation_euler[1] = -radians(angle)
        assert ghost == pytest.approx(rot, abs=1e-6)                           # the shadow turns with it
        assert tip == pytest.approx(R @ LAY.tip_at("father", a) + np.array(AT), abs=2e-4), f   # the fingertip
        assert props == pytest.approx([2.5, 31.0, 30.0])
    assert [round(_expected(f), 3) for f in (31, 61, 91, 106, 112)] == [-20.0, 0.0, 20.0, 18.0, -18.0]


def test_both_figures_build_with_their_cards_and_no_warning(built, capsys):
    d, code, out = built
    capsys.readouterr()
    assert code == 0, out.get("error")
    assert not [ln for ln in out.get("log", []) if "WARNING" in ln]           # the form guard passes both
    code, q = _cli(["q", str(d / "build" / "t.blend"),
                    "sorted(o.name for o in bpy.data.objects if o.name.startswith(('dad_', 'mom_')))", "--frames", "31",
                    "--project", str(d)], capsys)
    assert code == 0, q
    assert q["values"][0] == sorted(LAY.objects("dad", "father") + LAY.objects("mom", "mother"))
