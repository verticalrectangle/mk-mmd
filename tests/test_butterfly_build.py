"""The butterfly in a real `mk build` (the Blender half of the prop): the wings turn as the core maths says for a keyed `flap`
(the keys stage's rate integral) and for one beat on every hit of a lane, the body bobs, the card has its use points and
rates, the pattern atlas is packed into the .blend, and the prop passes the form guard. Skipped without Blender."""
import json
import os
from pathlib import Path

import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.core import butterfly as BF


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

[audio]
hits = "hits.json"

[[prop]]
name = "b1"
card = "library:butterfly"
at = [0.3, 0.0, 1.0]
yaw = 30
slots = { style = "morpho", flap = 2.0, flap_amp = 90, flap_mid = 15, tile = 128 }

[[prop]]
name = "b2"
card = "library:butterfly"
at = [-0.3, 0.0, 1.0]
slots = { style = "swallowtail", flap = 0, phase = 0.25, tile = 128 }

[[key]]
target = "b1"
prop = "flap"
interp = "LINEAR"
keys = [[0.0, 2.0], [1.0, 6.0]]

[[key]]
target = "b2"
prop = "hit"
lane = "kick"
"""
HITS = {"kick": [0.5, 1.2]}


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("butterfly")
    (root / "mk.toml").write_text(PROJECT, encoding="utf-8")
    (root / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
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


def beats_b1(t):
    """The integral of the keyed flap: 2 beats/s rising linearly to 6 at 1 s, then held."""
    if t <= 0.0:
        return 2.0 * t
    if t <= 1.0:
        return 2.0 * t + 2.0 * t * t
    return 4.0 + 6.0 * (t - 1.0)


def test_the_wings_follow_the_integral_of_the_keyed_flap(built, capsys):
    frames = [31, 37, 46, 52, 61, 76]
    got = query(built, 'bpy.data.objects["b1_wing_R"].rotation_euler.y', frames, capsys)
    for f, y in zip(frames, got):
        t = (f - 31) / 30.0
        want = float(BF.wing_angle(beats_b1(t), 90.0, 15.0))
        assert y == pytest.approx(want, abs=2e-3), f
    left = query(built, 'bpy.data.objects["b1_wing_L"].rotation_euler.y', frames, capsys)
    assert left == pytest.approx([-v for v in got], abs=1e-6)


def test_one_beat_on_every_hit_and_rest_in_between(built, capsys):
    frames = [40, 46, 49, 52, 55, 62, 67, 70]
    got = query(built, 'bpy.data.objects["b2_wing_R"].rotation_euler.y', frames, capsys)
    rest = float(BF.wing_angle(0.25, 100.0, 20.0))
    P = {k: v[0] for k, v in BF.FLAP_PARAMS.items()}
    P.update(phase=0.25)
    for f, y in zip(frames, got):
        t = (f - 31) / 30.0
        last = max([h for h in HITS["kick"] if h <= t + 1e-9], default=-1000.0)
        want = float(BF.wing_angle(BF.flap_phase(0.0, dict(P, hit=last), t), 100.0, 20.0))
        assert y == pytest.approx(want, abs=2e-3), f
    assert got[0] == pytest.approx(rest, abs=1e-3) and got[1] == pytest.approx(rest, abs=1e-3)   # before the hit
    assert abs(got[3] - rest) > 0.3                                                              # mid-beat


def test_the_body_bobs_with_the_beat(built, capsys):
    zs = query(built, 'bpy.data.objects["b1_body"].location.z', [31, 34, 37], capsys)
    for f, z in zip([31, 34, 37], zs):
        t = (f - 31) / 30.0
        assert z == pytest.approx(float(BF.body_bob(beats_b1(t))) * 0.09, abs=2e-5)


def test_the_card_the_atlas_and_the_form_guard(built, capsys):
    got = query(built, '[bpy.data.images["mk_butterfly_morpho_128"].packed_file is not None, '
                       'len(bpy.data.objects["b1_wing_L"].data.attributes["mk_dorsal"].data), '
                       'bpy.data.objects["b1_body"].scale.x]', [31], capsys)[0]
    assert got[0] is True and got[1] > 1000 and got[2] == pytest.approx(0.09)
    code, out = cli(["build", "--project", str(built), "--until", "props"], capsys)
    assert code == 0 and not [line for line in out["log"] if "WARNING" in str(line) and "b1" in str(line)]
    card = out["stages"]["props"]["b1"]
    assert card["uses"]["look"] == ["head", "body", "tip_L", "tip_R"] and card["uses"]["anchor"] == ["perch"]


def test_a_bad_style_stops_the_build_with_the_prop_named(tmp_path, capsys):
    (tmp_path / "mk.toml").write_text(PROJECT.replace('style = "morpho"', 'style = "moth"'), encoding="utf-8")
    (tmp_path / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "props"], capsys)
    assert code != 0 and "moth" in out["error"]
