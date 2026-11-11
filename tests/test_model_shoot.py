"""bpy-free invariants of `mk model shoot`'s studio (mkmmd/model/shoot.py): the views a list asks for, the regions a
model's own measures frame, the cameras, and the project it writes."""
import math
import tomllib

import pytest

from mkmmd.model import shoot as SH

MEASURE = {"top": 1.62, "width": 1.30, "eye_height": 1.40, "head_z": 1.34, "hip_height": 0.82}


def test_views_are_named_or_come_in_sets_once_each():
    assert SH.views_of("turn,face,front") == [*SH.SETS["turn"], "face"]
    assert SH.views_of(None) == SH.SETS["all"]
    with pytest.raises(SH.ShootError, match="view 'nose'"):
        SH.views_of("face,nose")


def test_regions_follow_the_models_own_measures():
    reg = SH.regions(MEASURE)
    small = SH.regions({k: 0.5 * v for k, v in MEASURE.items()})
    for name in ("full", "bust", "face"):                                  # half the model: half the field, so she
        assert small[name][0] == pytest.approx(0.5 * reg[name][0])        # fills her pictures as a tall one does
        assert small[name][1] == pytest.approx(0.5 * reg[name][1])
    assert reg["full"][1] >= MEASURE["top"] and reg["full"][1] >= MEASURE["width"]     # arms out: nothing cut off
    mid, field = reg["face"]
    assert mid - 0.5 * field < MEASURE["head_z"] and mid + 0.5 * field > MEASURE["top"]      # neck to the horns' tips


def test_a_camera_stands_where_its_region_fills_the_frame():
    plan = SH.shots(MEASURE, ["face", "face_high", "back"], expressions=["笑い"])
    assert [s["name"] for s in plan] == ["face", "face_high", "back", "morph:笑い"]
    reg = SH.regions(MEASURE)
    face, high, back, smile = plan
    dist = math.dist(face["at"], face["look"])
    assert dist == pytest.approx(reg["face"][1] * face["lens"] / SH.SENSOR, abs=1e-3)
    assert high["at"][2] > high["look"][2] and math.dist(high["at"], high["look"]) == pytest.approx(dist, abs=1e-3)
    assert back["yaw"] == 180.0 and back["look"][2] == pytest.approx(reg["full"][0], abs=1e-3)
    assert smile["morph"] == "笑い" and smile["at"] == face["at"]


def test_the_project_is_a_turntable_lit_alike_for_every_view():
    plan = SH.shots(MEASURE, ["front", "side", "face"], expressions=["あ"])
    text = SH.project("/m/her.pmx", "/m/her.rig.json", plan, size=900, view="AgX")
    d = tomllib.loads(text)
    assert d["look"]["view"] == "AgX" and d["output"][0]["size"] == [900, 900]
    assert d["cast"][0] == {"name": "her", "pmx": "/m/her.pmx", "rig": "/m/her.rig.json"}
    assert len(d["shot"]) == 4 and [x["name"] for x in d["light"]] == [x[0] for x in SH.LIGHTS]
    turn = d["key"][0]                                                     # she turns, the lights and the camera stay
    assert turn["target"] == "Her" and turn["interp"] == "CONSTANT" and len(turn["keys"]) == 4
    assert [k[1] for k in turn["keys"]] == pytest.approx([0.0, math.radians(-90.0), 0.0, 0.0], abs=1e-4)
    for i, shot in enumerate(d["shot"]):                                   # a view's picture is taken inside its shot
        t = (SH.frame_of(i) - 1) / SH.FPS
        assert shot["from"] < t < shot["to"]
    said = d["perform"]["her"]["expressions"]
    assert [e["morph"] for e in said] == ["あ"] and said[0]["keys"][1][1] == 1.0
    assert "rig" not in tomllib.loads(SH.project("/m/her.pmx", None, plan[:1]))["cast"][0]
    assert tomllib.loads(SH.project("/m/her.pmx", None, plan[:1]))["look"]["view"] == "Standard"
