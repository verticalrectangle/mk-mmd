"""Real `mk build` runs of tiny projects (the stages in mkmmd/blender/build, docs/design.md: Building): what a project that
asks for something the stage cannot do now gets, a BuildError that names the table and the fix instead of a bare TypeError /
KeyError / AttributeError, and the behaviours that were wrong (a car's pitch, `floor = true`, `physics = "none"`, a shot's
per-output tables). The cast is a mannequin made by `mk model build` (about a second). Skipped without Blender."""
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

HEAD = """[project]
name = "t"
fps = 30
frame0 = 31
duration = 2.0
blend = "build/t.blend"

[[output]]
name = "16x9"
size = [640, 360]

[[output]]
name = "9x16"
size = [360, 640]
"""


def cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def mannequin(tmp_path_factory):
    d = tmp_path_factory.mktemp("mannequin")
    (d / "m.toml").write_text('[model]\nname = "mq"\nparts = ["mannequin", "mannequin_hair"]\n'
                              f'out = "{d / "out"}"\n[model.needs]\nmannequin = []\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["model", "build", str(d / "m.toml")])
    assert e.value.code == 0
    return d / "out" / "mq.pmx", d / "out" / "mq.rig.json"


@pytest.fixture
def make(tmp_path, mannequin, capsys):
    """make(body, until=None, cast="", **) builds HEAD + a cast member `mq` + `body`; returns (exit code, JSON)."""
    capsys.readouterr()

    def build(body, until=None, cast="", with_cast=True, skip=()):
        text = HEAD
        if with_cast:
            text += f'\n[[cast]]\nname = "mq"\npmx = "{mannequin[0]}"\nrig = "{mannequin[1]}"\n{cast}\n'
        (tmp_path / "mk.toml").write_text(text + "\n" + body, encoding="utf-8")
        argv = ["build", "--project", str(tmp_path)] + (["--until", until] if until else [])
        for s in skip:
            argv += ["--skip", s]
        return cli(argv, capsys)
    build.root = tmp_path
    return build


def warnings(out):
    return [ln for ln in out.get("log", []) if "WARNING" in ln]


# ---------------------------------------------------------------- pose
def test_standing_on_the_floor_builds_and_a_seat_without_sit_names_the_cast_and_the_fix(make):
    code, out = make('[pose.mq]\nfeet = "floor"\nlean = 3\n', until="pose")
    assert code == 0, out.get("error")                                      # was a TypeError at seat["hip"]
    code, out = make('[pose.mq]\nfeet = "seat"\n', until="pose")
    assert code == 3 and "BuildError" in out["error"] and "pose.mq.feet" in out["error"]
    assert 'feet = "floor"' in out["error"] and "`sit`" in out["error"]


def test_an_empty_pose_table_is_reported_not_skipped_in_silence(make):
    code, out = make("[pose.mq]\n", until="pose")
    assert code == 0 and any("pose.mq: the table is empty" in w for w in warnings(out))


# ---------------------------------------------------------------- the per-member tables
def test_tables_of_no_cast_member_and_unknown_keys_are_build_errors(make):
    for body, until, frags in [("[pose.nobody]\nlean = 1\n", "pose", ["[pose.nobody]", "no cast member 'nobody'", "cast: mq"]),
                               ("[pose.mq]\nlean = 1\nsiting = 2\n", "pose", ["[pose.mq]", "unknown key 'siting'", "sit_offset"]),
                               ("[pose.mq.hands.left]\nat = [0, 0, 1]\n", "pose", ["hands", "unknown key 'left'", "known: L, R"]),
                               ("[perform.mq]\nblinks = {per_min = 5}\n", "perform", ["[perform.mq]", "unknown key 'blinks'"]),
                               ("[perform.mq]\nblink = {per_minute = 5}\n", "perform", ["blink", "unknown key 'per_minute'"]),
                               ("[sim.mq]\nflor = 0.3\n", "sim", ["[sim.mq]", "unknown key 'flor'"])]:
        code, out = make(body, until=until)
        assert code == 3 and "BuildError" in out["error"], (body, out.get("error"))
        assert all(f in out["error"] for f in frags), (frags, out["error"])


# ---------------------------------------------------------------- targets and timelines
def test_the_camera_target_before_the_shots_stage_is_a_build_error_not_an_attribute_error(make):
    code, out = make('[perform.mq]\nlook = "camera"\n', until="perform")
    assert code == 3 and "BuildError" in out["error"] and 'target "camera"' in out["error"] and "[[light]]" in out["error"]


def test_a_table_without_a_timeline_key_reads_the_projects_and_says_which_table_when_there_is_none(make):
    body = "[perform.mq]\nbob = {deg = 1.0}\n"
    code, out = make(body, until="perform")
    assert code == 3 and "BuildError" in out["error"] and "perform.mq.bob" in out["error"] and "audio/timeline.json" in out["error"]
    (make.root / "audio").mkdir()
    (make.root / "audio" / "timeline.json").write_text(json.dumps({"beats": [0.5, 1.0, 1.5], "downbeats": [0.5]}), encoding="utf-8")
    code, out = make(body, until="perform")
    assert code == 0, out.get("error")                                      # the default file is read


# ---------------------------------------------------------------- cast physics and the sim floor
def test_physics_none_leaves_the_member_out_of_the_sim_and_an_unknown_word_is_refused(make):
    code, out = make("[sim.mq]\nfloor = true\n", cast='physics = "none"')
    assert code == 0, out.get("error")
    assert out["stages"]["sim"]["mq"] == {"skipped": 'physics = "none"'} and out["stages"]["cast"]["mq"]["physics"] == "none"
    code, out = make("", until="cast", cast='physics = "bulet"')
    assert code == 3 and "physics = 'bulet'" in out["error"] and "mk, none, bullet" in out["error"]


def test_sim_floor_true_is_the_ground_at_zero_not_one_metre(make):
    code, out = make("[sim.mq]\nfloor = true\nfamilies = [\"bangs\"]\n")
    assert code == 0, out.get("error")
    assert out["stages"]["sim"]["mq"]["floor_z"] == 0.0                     # float(True) was 1.0
    code, out = make("[sim.mq]\nfloor = 0.25\nfamilies = [\"bangs\"]\n")
    assert code == 0 and out["stages"]["sim"]["mq"]["floor_z"] == 0.25
    code, out = make("[sim.mq]\nfloor = \"ground\"\n")
    assert code == 3 and "[sim.mq] floor = 'ground'" in out["error"]


# ---------------------------------------------------------------- vehicles
VEHICLE = """
[[set]]
name = "road"
kind = "test_road"

[[prop]]
name = "car"
card = "library:car_mockup"
{extra}
[[vehicle]]
prop = "car"
path = "road:road"
lane = "R1"
speed = [[0, 8], [0.4, 8], [1.0, 20], [1.4, 20], [1.9, 6]]
at = 40.0
pitch = 5.0
"""


def test_a_car_noses_up_under_acceleration_and_dives_under_braking(make, capsys):
    code, out = make(VEHICLE.format(extra=""), with_cast=False)
    assert code == 0, out.get("error")
    # the body's Euler X (degrees) is its pitch: positive tips the nose down (checked on the car's own geometry), so a nose-up car < 0
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "obj('car').euler[0]", "--frames", "52,82", "--project",
                   str(make.root)], capsys)
    accelerating, braking = q["values"]
    assert accelerating < -1.0 and braking > 1.0                            # nose up under acceleration, down under braking


def test_a_card_whose_steering_object_is_missing_is_a_build_error(make):
    code, out = make(VEHICLE.format(extra='card_extra = { steering = { object = "no_such_wheel" } }\n'), with_cast=False,
                     until="vehicles")
    assert code == 3 and "BuildError" in out["error"] and "steering object 'no_such_wheel' missing" in out["error"]


# ---------------------------------------------------------------- shots
SHOTS = """
[[prop]]
name = "car"
card = "library:car_mockup"

[[shot]]
name = "a"
from = 0.0
to = 1.0
at = {{ prop = "car", point = [0, -3, 1] }}
look = [0, 0, 0.5]
keys = [{{ t = 0.0 }}, {{ t = 1.0, lens = 50 }}]
frame = {{ subject = [[0, 0, 0.5]], fill = 0.5 }}
{extra}"""


def test_a_shot_with_a_dict_at_and_keys_and_an_aspect_that_changes_frame_builds(make):
    code, out = make(SHOTS.format(extra="[shot.aspect.9x16]\nframe = { fill = 0.7 }\n"), with_cast=False, until="shots")
    assert code == 0, out.get("error")                                      # a TypeError, then a KeyError: 'subject'
    assert set(out["stages"]["shots"]["a"]["cameras"]) == {"16x9", "9x16"}


def test_shot_mistakes_are_build_errors_that_list_what_there_is(make):
    for extra, frags in [("[shot.aspect.1x1]\nlens = 40\n", ["[shot.aspect.1x1]", "outputs: 16x9, 9x16"]),
                         ("lenss = 40\n", ["shot 'a'", "unknown key 'lenss'", "known: name, from, to"]),
                         ("dof = { focus = [0, 0, 0], fstopp = 2 }\n", ["shot 'a' dof", "unknown key 'fstopp'"]),
                         ("\n[[shot]]\nname = \"p\"\nplate = true\nfrom = 1.0\n", ["shot 'p'", "`from` and `to` together"])]:
        code, out = make(SHOTS.format(extra=extra), with_cast=False, until="shots")
        assert code == 3 and "BuildError" in out["error"], (extra, out.get("error"))
        assert all(f in out["error"] for f in frags), (frags, out["error"])
