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
        MAIN.main(["model", "build", str(d / "m.toml"), "--no-cache"])
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

ELBOWS = """[pose.mq]
feet = "floor"
[pose.mq.hands.L]
at = {{ cast = "mq", point = [0.08, -0.24, 0.95] }}
pole = {{ cast = "mq", point = [{x}, {y}, {z}] }}
[pose.mq.hands.R]
at = {{ cast = "mq", point = [-0.08, -0.24, 0.95] }}
pole = {{ cast = "mq", point = [-{x}, {y}, {z}] }}
"""


def _bend(shoulder, elbow, wrist, pole):
    """Degrees between where the elbow bends and where the pole is, both seen across the shoulder-wrist line."""
    import numpy as np
    s, e, w, p = (np.asarray(v, float) for v in (shoulder, elbow, wrist, pole))
    axis = (w - s) / np.linalg.norm(w - s)
    across = [(v - s) - axis * float((v - s) @ axis) for v in (e, p)]
    a, b = (v / np.linalg.norm(v) for v in across)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def test_both_elbows_point_at_their_own_poles_whatever_the_rig_rolls(make, capsys):
    """The IK pole angle depends on the roll of the arm bones, which MMD rigs mirror between the sides: one fixed angle
    sent one elbow the wrong way (a driver's elbows met in front of her chest). Mirrored poles give mirrored elbows."""
    for pole in ([0.5, 0.05, 0.75], [0.45, 0.0, 1.35]):                       # out and down; out and up (chicken wings)
        code, out = make(ELBOWS.format(x=pole[0], y=pole[1], z=pole[2]), until="pose")
        assert code == 0, out.get("error")
        expr = "[list(bone(b).head) for b in ('arm.L', 'elbow.L', 'wrist.L', 'arm.R', 'elbow.R', 'wrist.R')]"
        code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "80", "--project", str(make.root)], capsys)
        assert code == 0, q
        sl, el, wl, sr, er, wr = q["values"][0]
        left = _bend(sl, el, wl, pole)
        right = _bend(sr, er, wr, [-pole[0], pole[1], pole[2]])
        assert left < 20.0 and right < 20.0, (pole, left, right)

DRIVE = """
[[prop]]
name = "car"
card = "library:car_mockup"
[pose.mq]
sit = "car:driver"
feet = "car:driver"
lean = 12
sit_offset = [0.0, -0.14, 0.0]
[pose.mq.hands.L]
grip = "car:wheel"
clock = {left}
[pose.mq.hands.R]
grip = "car:wheel"
clock = {right}
"""


def _angle(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


def test_a_driver_with_no_poles_holds_the_wheel_with_straight_wrists_and_the_elbows_out_and_down(make, capsys):
    """The wheel grip was placed for the rim alone: the forearms came in from the elbow's side and the hands bent at the
    wrist through the cuff (107 degrees on this seat). The grip and the elbow are now chosen together for the arm."""
    for left, right in ((9, 3), (10, 2)):
        code, out = make(DRIVE.format(left=left, right=right), until="pose")
        assert code == 0, out.get("error")
        names = [f"{b}.{s}" for s in ("L", "R") for b in ("arm", "elbow", "wrist", "middle1")]
        code, q = cli(["q", str(make.root / "build" / "t.blend"), f"[list(bone(b).head) for b in {names!r}]",
                       "--frames", "80", "--project", str(make.root)], capsys)
        assert code == 0, q
        pts = q["values"][0]
        for k, side in enumerate(("L", "R")):
            sh, el, wr, mid = pts[4 * k: 4 * k + 4]
            bend = _angle([w - e for w, e in zip(wr, el)], [m - w for m, w in zip(mid, wr)])
            assert bend < 45.0, (left, side, bend)                       # the hand in line with the forearm
            out_x = (el[0] - sh[0]) * (1 if side == "L" else -1)          # the mannequin faces -Y: its left is +X
            assert out_x > -0.01 and el[2] < sh[2] - 0.04, (left, side, sh, el)   # not tucked in, not winged up


SPREAD = """[pose.mq]
feet = "floor"
[pose.mq.hands.L]
at = {{ cast = "mq", point = [0.3, -0.2, 1.0] }}
fingers = {{ index = [5, 5, 5], spread = {spread} }}
"""


def _fanned(wrist, base, tip, m1, m2, l1):
    """Signed degrees a finger (base -> tip) turns away from the middle finger (m1 -> m2) in the plane of the palm: + away
    from it, toward the finger's own side."""
    import numpy as np
    w, b, t, a, c, l = (np.asarray(v, float) for v in (wrist, base, tip, m1, m2, l1))
    n = np.cross(a - w, l - w)
    n /= np.linalg.norm(n)
    flat = lambda v: v - (v @ n) * n                                  # noqa: E731
    mid = flat(c - a) / np.linalg.norm(flat(c - a))
    side = flat(b - a) - (flat(b - a) @ mid) * mid
    side /= np.linalg.norm(side)
    f = flat(t - b)
    return float(np.degrees(np.arctan2(f @ side, f @ mid)))


def test_spread_fans_the_fingers_apart_from_the_middle_one_and_a_negative_spread_closes_them(make, capsys):
    fan = {}
    for spread in (0, 12, -6):
        code, out = make(SPREAD.format(spread=spread), until="pose")
        assert code == 0, out.get("error")
        names = ["wrist.L", "index1.L", "index2.L", "middle1.L", "middle2.L", "little1.L", "little2.L"]
        code, q = cli(["q", str(make.root / "build" / "t.blend"), f"[list(bone(b).head) for b in {names!r}]",
                       "--frames", "80", "--project", str(make.root)], capsys)
        assert code == 0, q
        w, i1, i2, m1, m2, l1, l2 = q["values"][0]
        fan[spread] = (_fanned(w, i1, i2, m1, m2, l1), _fanned(w, l1, l2, m1, m2, l1))
    assert fan[12][0] - fan[0][0] == pytest.approx(12.0, abs=2.0)    # the index turns `spread` away from the middle,
    assert fan[12][1] - fan[0][1] == pytest.approx(14.4, abs=2.5)    # the little finger 1.2 times as far
    assert fan[-6][0] - fan[0][0] == pytest.approx(-6.0, abs=2.0)    # a negative spread closes them together

GUITAR = """[[prop]]
name = "guitar"
card = "library:electric_guitar"
wear = "mq"
[pose.mq]
feet = "floor"
[perform.mq]
bounce = { depth = 0.05, beats = [0.5, 1.0, 1.5, 2.0], decay = 0.12 }
"""


def test_a_worn_guitars_cord_stays_in_its_jack_and_swings_as_the_player_bounces(make, capsys):
    """The cord was a fixed curve whose top 35 cm a hook dragged along: below that it stood still whatever the player did.
    The sim stage now swings all of it from the jack."""
    import numpy as np
    code, out = make(GUITAR)
    assert code == 0, out.get("error")
    expr = ('[[list(bpy.data.objects["guitar_cable"].data.splines[0].points[i].co[:3]) for i in (0, 15)], '
            'list(bpy.data.objects["guitar_jack"].matrix_world.translation)]')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "40,46,52,58", "--project", str(make.root)],
                  capsys)                                         # around the beat at t 0.5 (frame 46)
    assert code == 0, q
    top = np.array([r[0][0] for r in q["values"]])
    mid = np.array([r[0][1] for r in q["values"]])
    jack = np.array([r[1] for r in q["values"]])
    assert np.abs(top - jack).max() < 1e-3                        # the plug end is in the jack on every frame
    assert np.ptp(mid[:, 2]) > 0.01                               # 45 cm down the cord, the bounce shows


BODY = """[pose.mq]
feet = "floor"
[perform.mq]
bounce = { depth = 0.05, beats = [1.0], decay = 0.1 }
kick = { foot = "L", height = 0.2, back = 0.0, beats = [1.0], decay = 0.1 }
rise = [[1.4, 0.0], [1.9, 0.1]]
"""


def test_bounce_dips_the_hips_on_the_beat_with_the_feet_planted_kick_lifts_one_foot_rise_lifts_all(make, capsys):
    code, out = make(BODY, until="perform")
    assert code == 0, out.get("error")
    expr = "[bone('center').head.z, bone('ankle.L').head.z, bone('ankle.R').head.z]"
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "49,61,73,88", "--project", str(make.root)],
                  capsys)                                         # t 0.6 (still), 1.0 (the beat), 1.4, 1.9 (risen)
    assert code == 0, q
    still, beat, before, risen = q["values"]
    assert beat[0] - still[0] == pytest.approx(-0.05, abs=0.004)   # the hips dip on the beat ...
    assert beat[2] - still[2] == pytest.approx(0.0, abs=0.002)     # ... the planted foot stays put ...
    assert beat[1] - still[1] == pytest.approx(0.2, abs=0.01)      # ... the kicking foot goes up
    for k in range(3):
        assert risen[k] - before[k] == pytest.approx(0.1, abs=0.006)    # rise lifts hips and feet alike


TRAFFIC = """
[[set]]
name = "road"
kind = "test_road"
length = 120.0
curve = 0.0

[[prop]]
name = "car"
card = "library:car_mockup"

[[prop]]
name = "other"
card = "library:car_mockup"

[[vehicle]]
prop = "car"
path = "road:road"
lane = "R1"
speed = 20.0
at = 40.0

[[vehicle]]
prop = "other"
path = "road:road"
lane = "L1"
speed = 25.0
meet = { vehicle = "car", t = 1.5 }
leave = true
"""


def test_an_oncoming_car_meets_ours_when_asked_faces_the_other_way_and_is_hidden_off_the_road(make, capsys):
    code, out = make(TRAFFIC, with_cast=False)
    assert code == 0, out.get("error")
    expr = ("[list(obj('car').loc), list(obj('other').loc), list(bpy.data.objects['car'].matrix_world.col[1])[:3], "
            "list(bpy.data.objects['other'].matrix_world.col[1])[:3], bpy.data.objects['other'].hide_render]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "1,76", "--project", str(make.root)], capsys)
    assert code == 0, q
    early, meet = q["values"]
    assert early[4] is True                                       # t -1: 132.5 m along a 120 m road: not on it yet, hidden
    car, other, y_car, y_other, hidden = meet
    assert hidden is False
    import numpy as np
    gap = np.asarray(other) - np.asarray(car)
    assert np.linalg.norm(gap) == pytest.approx(3.6, abs=0.05)    # alongside: the width of a lane apart, side by side
    assert float(np.dot(y_car, y_other)) == pytest.approx(-1.0, abs=0.01)    # nose to nose: it drives the other way


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


# ---------------------------------------------------------------- props
SCALED = """[[prop]]
name = "box"
card = "library:cassette_player"
{where}
scale = 4.0
"""


def test_a_scaled_library_prop_grows_its_colliders_and_use_points_and_place_is_refused(make, capsys):
    code, out = make(SCALED.format(where="at = [0.0, 1.0, 0.0]"), with_cast=False, until="props")
    assert code == 0, out.get("error")
    blend = str(make.root / "build" / "t.blend")
    code, q = cli(["q", blend, 'obj("box_col_body").dims', "--frames", "31"], capsys)
    assert code == 0 and q["values"][0][2] == pytest.approx(4 * 0.16, rel=0.02)           # the player is 0.16 m tall
    code, q = cli(["q", blend, 'obj("box").matrix @ Vector((0.1145, -0.06, 0.0845))', "--frames", "31"], capsys)
    assert q["values"][0] == pytest.approx([4 * 0.1145, 1.0 - 4 * 0.06, 4 * 0.0845], abs=1e-4)   # a speaker's centre
    code, out = make(SCALED.format(where='place = { on = "floor" }'), with_cast=False, until="props")
    assert code == 3 and "BuildError" in out["error"] and "give `at` with `scale`" in out["error"]


MIC = """[pose.mq]
feet = "floor"

[[prop]]
name = "mic"
card = "library:handheld_mic"
{attach}
cable = true
"""


def test_an_attached_mic_hangs_its_cord_from_its_jack_to_the_floor_and_a_cable_needs_attach(make, capsys):
    code, out = make(MIC.format(attach='attach = "mq:wrist.R"'), until="pose")
    assert code == 0, out.get("error")
    cable = out["stages"]["pose"]["cables"]["mic"]
    code, q = cli(["q", str(make.root / "build" / "t.blend"), 'obj("mic_jack").loc', "--frames", "61"], capsys)
    assert cable["top"] == pytest.approx(q["values"][0], abs=2e-3)                        # it leaves the mic's tail
    assert cable["floor_z"] == 0.0 and cable["length_m"] > cable["top"][2]                  # and reaches the floor
    code, out = make(MIC.format(attach=""), until="pose")
    assert code == 3 and "`cable` goes with `attach`" in out["error"]
