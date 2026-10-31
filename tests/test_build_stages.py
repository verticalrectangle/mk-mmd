"""Real `mk build` runs of tiny projects (the stages in mkmmd/blender/build, docs/design.md: Building): what a project that
asks for something the stage cannot do now gets, a BuildError that names the table and the fix instead of a bare TypeError /
KeyError / AttributeError, and the behaviours that were wrong (a car's pitch, `floor = true`, `physics = "none"`, a shot's
per-output tables). The cast is a mannequin made by `mk model build` (about a second). Skipped without Blender."""
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from mkmmd import config as CFG
from mkmmd.cli import main as MAIN
from mkmmd.model import pmx_io as X


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


def test_a_worn_cord_lies_on_the_floor_it_was_hung_on_when_the_wearer_drops_onto_it(make, capsys):
    """The sim took its floor from the wearer's root on whatever frame it ran (the last), so with a player keyed off the
    floor (falling in from the sky, jumping out at the end) it swung a cord it saw a metre under its floor: stretched,
    and left high in the air after she landed."""
    code, out = make(GUITAR + '[[key]]\ntarget = "Mq"\nprop = "location"\nindex = 2\n'
                              'keys = [[0.0, 1.0], [0.4, 0.0], [1.8, 0.0], [1.95, 1.0]]\n')
    assert code == 0, out.get("error")
    assert out["stages"]["sim"]["cables"]["guitar"]["floor_pen_mm_max"] < 5.0  # was 1000: a floor where she stood last
    expr = 'min(p.co[2] for p in bpy.data.objects["guitar_cable"].data.splines[0].points)'
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "61,76", "--project", str(make.root)],
                  capsys)                                         # t 1.0, 1.5: she landed at 0.4, jumps at 1.8
    assert code == 0, q
    assert max(q["values"]) < 0.02                               # the cord's end lies on the floor, not a metre up


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


SWAY = """[pose.mq]
feet = "floor"
[perform.mq]
crouch = [[1.0, 0.0], [1.3, 0.2]]
rock = { deg = 0.0, period = 1.6, hips = 0.04 }
rise = [[1.7, 0.0], [1.9, 0.1]]
"""


def test_rock_hips_shift_the_weight_from_foot_to_foot_and_crouch_lowers_the_hips_with_the_feet_planted(make, capsys):
    import numpy as np
    code, out = make(SWAY, until="perform")
    assert code == 0, out.get("error")
    expr = "[list(bone('center').head), list(bone('ankle.L').head), list(bone('ankle.R').head)]"
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "43,55,79", "--project", str(make.root)],
                  capsys)                                         # t 0.4 (swung left), 0.8 (centred), 1.6 (crouched)
    assert code == 0, q
    left, mid, low = (np.array(v) for v in q["values"])
    assert left[0] - mid[0] == pytest.approx([0.04, 0.0, 0.0], abs=0.004)    # the hips go to the model's left (+x) ...
    assert low[0] - mid[0] == pytest.approx([0.0, 0.0, -0.2], abs=0.006)     # ... and down in the crouch
    for k in (1, 2):
        assert np.abs(left[k] - mid[k]).max() < 0.002 and np.abs(low[k] - mid[k]).max() < 0.002    # the feet stay put


HINGE = """[pose.mq]
feet = "floor"
hips = { shift = [0.0, 0.08, -0.05], pitch = 30 }
lean = 40
"""


def test_hips_pitch_tips_the_pelvis_forward_about_the_waist_with_the_feet_planted(make, capsys):
    import numpy as np
    code, out = make(HINGE, until="perform")
    assert code == 0, out.get("error")
    expr = ("[list(bone('lower_body').head), list((bone('leg.L').head + bone('leg.R').head) / 2), "
            "list(bone('ankle.L').head), list(bone('ankle.R').head)]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "1,40", "--project", str(make.root)],
                  capsys)                                         # the rest pose, and the base pose after the settle
    assert code == 0, q
    rest, posed = ([np.array(p) for p in v] for v in q["values"])
    a0, a1 = rest[1] - rest[0], posed[1] - posed[0]               # waist -> hip joints, translation taken out
    tilt = np.degrees(np.arctan2(a1[1], -a1[2]) - np.arctan2(a0[1], -a0[2]))
    assert tilt == pytest.approx(30.0, abs=2.0)                    # the hip joints swing back: the pelvis tips forward
    for k in (2, 3):
        assert np.abs(posed[k] - rest[k]).max() < 0.003            # the feet stay planted


SWAY_SPANS = """[pose.mq]
feet = "floor"
[perform.mq]
rock = [{ deg = 0.0, period = 1.6, hips = 0.04, to = 0.6, ease = 0.2 },
        { deg = 0.0, period = 1.6, hips = -0.02, from = 1.4, ease = 0.2 },
        { deg = 8.0, period = 1.6, from = 1.4, ease = 0.2 }]
"""


def test_rocks_sway_only_inside_their_spans_and_a_list_of_them_adds_up(make, capsys):
    import numpy as np
    code, out = make(SWAY_SPANS, until="perform")
    assert code == 0, out.get("error")
    expr = "[bone('center').head.x, bone('head').head.x]"
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "43,55,61,88", "--project", str(make.root)],
                  capsys)                                         # t 0.4, 0.8 (the sine's zero), 1.0, 1.9
    assert code == 0, q
    early, zero, between, late = (np.array(v) for v in q["values"])
    assert early[0] - zero[0] == pytest.approx(0.04, abs=0.004)        # the first span's hips, at the sine's peak
    assert between[0] - zero[0] == pytest.approx(0.0, abs=0.002)       # both spans shut, though the sine is at -0.71
    assert between[1] - between[0] == pytest.approx(zero[1] - zero[0], abs=0.004)    # ... and no lean either
    assert late[0] - zero[0] == pytest.approx(-0.02 * np.sin(2 * np.pi * 1.9 / 1.6), abs=0.003)   # the second span's hips
    assert (late[1] - late[0]) - (zero[1] - zero[0]) > 0.02            # the third leans the body to her left over them


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



BANGS = ("前髪1", "前髪2", "前髪3")


def _bangs_and_head(make, capsys):
    """Per frame (40, 60, 90): the bangs' points (the three heads and the tip) in the world, and the head's motion from
    rest (4x4: a point of the model's rest space -> where it is now)."""
    expr = (f"[[list(bone(n).head) for n in {BANGS!r}] + [list(bone({BANGS[-1]!r}).tail)], "
            "[list(r) for r in bone('head').matrix @ arm().data.bones[bone('head').name].matrix_local.inverted()]]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "40,60,90", "--project", str(make.root)],
                  capsys)
    assert code == 0, q
    return [(np.array(v[0]), np.array(v[1])) for v in q["values"]]


def test_a_capsule_given_by_its_points_rides_its_bone_and_holds_the_bangs_out(make, mannequin, capsys):
    """[sim.<cast>] colliders `{type = "capsule", bone, a, b, R}`: the segment a-b of the model's rest space, riding on
    the bone (a shape where no two bones span it: a chest under a lapel). One laid across the forehead through the
    bangs' middle joint, on the head, pushes the bangs out of it on every frame, wherever the head has turned; a and b
    read in another space would put it elsewhere and leave them where they hang."""
    code, out = make('[sim.mq]\nfamilies = ["bangs"]\n')
    assert code == 0, out.get("error")
    shapes = out["stages"]["sim"]["mq"]["colliders"]
    free = _bangs_and_head(make, capsys)
    rig = json.loads(mannequin[1].read_text(encoding="utf-8"))
    mid = np.array(next(b["head"] for b in rig["bones"].values() if b.get("jp") == BANGS[1]))
    a, b, R = mid - [0.04, 0.0, 0.0], mid + [0.04, 0.0, 0.0], 0.01
    code, out = make('[sim.mq]\nfamilies = ["bangs"]\ncolliders = [{ type = "capsule", bone = "頭", '
                     f'a = {a.round(5).tolist()}, b = {b.round(5).tolist()}, R = {R}, tag = "brow" }}]\n')
    assert code == 0, out.get("error")
    assert out["stages"]["sim"]["mq"]["colliders"] == shapes + 1
    held = _bangs_and_head(make, capsys)
    for (P0, _), (P, M) in zip(free, held):
        A, B = (M @ np.append(a, 1.0))[:3], (M @ np.append(b, 1.0))[:3]
        t = np.clip((P - A) @ (B - A) / ((B - A) @ (B - A)), 0.0, 1.0)
        gap = np.linalg.norm(P - (A + t[:, None] * (B - A)), axis=1)
        assert gap.min() > R - 5e-4, gap                                   # all of the bangs outside it
        assert np.linalg.norm(P[1] - P0[1]) > 0.004, (P[1], P0[1])         # the middle joint was pushed out



KEPT = """[pose.mq]
feet = "floor"

[[shot]]
name = "face"
from = 0.0
to = 2.0
at = [0.0, -0.9, 1.36]
look = [0.0, 0.0, 1.36]
lens = 50

[[light]]
name = "sun"
kind = "sun"
power = 3.0
at = [1.0, -3.0, 3.0]
look = [0.0, 0.0, 1.3]

[post]
saturation = 0.0
"""


def test_the_keep_mask_is_where_the_kept_material_is_seen_and_the_frame_renders_as_it_did(make, capsys):
    """[post] keep: `mk render` draws `keep/<frame>.png` beside each frame, the share of each pixel the kept materials
    show, linear. The eyes' mask and the skin's never add up to more than the whole of a pixel (the eyes stand in front
    of the face and every other material hides what it covers; a rim is shared between them), the frame rendered right
    after the pass is the one rendered without keep (the scene is left as it was), and a material the model does not
    have is refused with the ones it has."""
    from PIL import Image
    d = make.root / "renders" / "draft" / "16x9"

    def render(keep):
        code, out = make(KEPT + (f'keep = {{ cast = "mq", materials = ["{keep}"] }}\n' if keep else ""))
        assert code == 0, out.get("error")
        for p in sorted(d.rglob("*.png"), reverse=True) if d.exists() else ():
            p.unlink()
        code, rep = cli(["render", "--project", str(make.root), "--output", "16x9", "--frames", "40"], capsys)
        if code:
            return code, rep
        frame = np.asarray(Image.open(d / "00040.png").convert("RGB"), int)
        mask = np.asarray(Image.open(d / "keep" / "00040.png").convert("L"), float) / 255 if keep else None
        return frame, mask

    seen, eyes = render("eye")
    _, skin = render("skin")
    assert (eyes > 0.5).sum() > 20 and (skin > 0.5).sum() > 20 * (eyes > 0.5).sum()
    assert (eyes + skin).max() <= 1.03                       # never more of a pixel between them than all of it
    plain, _ = render(None)
    assert plain[skin > 0.5].mean() > 50 and np.abs(seen - plain).max() <= 2     # a lit face, as it was
    code, rep = render("eyes")
    assert code == 3 and "has no material eyes" in rep["error"] and "eye" in rep["error"].split("it has")[1]



def test_a_sweep_is_a_floor_that_curves_up_into_a_wall_both_facing_the_subject(make, capsys):
    """`kind = "sweep"`, a photo studio's paper: the floor lies on z = 0 from y = -depth to the curve and faces up, the
    curve rises into the wall at y = radius, which faces the subject (toward -y); the set stands at `at`. An unknown key
    is an error that lists the known ones."""
    code, out = make('[[set]]\nname = "paper"\nkind = "sweep"\nat = [0.0, 1.0, 0.0]\nwidth = 4.0\ndepth = 3.0\n'
                     'height = 2.5\nradius = 0.8\n', until="sets", with_cast=False)
    assert code == 0, out.get("error")
    expr = ('[[list(o.matrix_world @ p.center) + list((o.matrix_world.to_3x3() @ p.normal).normalized()) '
            'for p in o.data.polygons] for o in [bpy.data.objects["paper_sweep"]]][0]')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "31", "--project", str(make.root)], capsys)
    assert code == 0, q
    P = np.array(q["values"][0])
    c, n = P[:, :3], P[:, 3:]
    floor, wall = c[:, 2] < 1e-6, c[:, 2] > 0.8 + 1e-3
    assert floor.any() and wall.any()
    assert np.allclose(n[floor], [0, 0, 1], atol=1e-6) and np.allclose(n[wall], [0, -1, 0], atol=1e-6)
    assert np.allclose(c[floor, 1], 1.0 - 3.0 / 2)                           # the floor: y from at - depth to the curve
    assert np.allclose(c[wall, 1], 1.0 + 0.8) and c[:, 2].max() < 2.5
    assert ((n[:, 1] <= 1e-6) & (n[:, 2] >= -1e-6)).all()                    # every face turned to the subject
    code, out = make('[[set]]\nname = "paper"\nkind = "sweep"\nradius = 0.8\ncurve = 1\n', until="sets", with_cast=False)
    assert code == 3 and "unknown keys ['curve']" in out["error"] and "'radius'" in out["error"]


def test_cast_materials_colour_tint_and_hide_the_model_s_own_materials(make, capsys):
    """`[[cast]] materials = {<PMX name> = {color | tint | hide}}`: `color` makes the material that flat colour (its
    texture off, ambient 0.4 of it), `tint` multiplies its colours, `hide` draws it no more; a name the model does not
    have is an error that lists the ones it has."""
    code, out = make("", until="cast", cast='materials = { skin = { color = "#ff0000" }, eye = { tint = "#808080" }, '
                                             'shirt = { hide = true } }')
    assert code == 0, out.get("error")
    expr = ('{m.mmd_material.name_j: [list(m.mmd_material.diffuse_color), list(m.mmd_material.ambient_color), '
            'm.mmd_material.alpha, m.node_tree.nodes["mmd_shader"].inputs["Base Tex Fac"].default_value] '
            'for m in bpy.data.materials if m.mmd_material.name_j in ("skin", "eye", "shirt", "mouth")}')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "31", "--project", str(make.root)], capsys)
    assert code == 0, q
    got = q["values"][0]
    assert got["skin"][0] == pytest.approx([1.0, 0.0, 0.0]) and got["skin"][1] == pytest.approx([0.4, 0.0, 0.0])
    assert got["skin"][3] == 0.0 and got["eye"][3] == 1.0                         # the colour's texture is off
    grey = ((0x80 / 255 + 0.055) / 1.055) ** 2.4                                   # #808080 in linear
    assert got["eye"][0] == pytest.approx([grey] * 3, abs=1e-4)                    # the eye's white diffuse, tinted
    assert got["shirt"][2] == 0.0 and got["mouth"][2] == 1.0
    code, out = make("", until="cast", cast='materials = { skn = { hide = true } }')
    assert code == 3 and "no material 'skn'" in out["error"] and "'skin'" in out["error"]


def test_lift_raises_an_ankle_above_its_rest_height_and_leaves_the_other_foot_down(make, capsys):
    """`[pose.<cast>] lift = {L = m}` (a foot kicked up in a jump): the left ankle ends that far above where it stands at
    rest, eased in over the settle; the right foot, with no lift, stays where it was."""
    code, out = make('[pose.mq]\nfeet = "floor"\nlift = { L = 0.15 }\n', until="pose")
    assert code == 0, out.get("error")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "[list(bone(b).head) for b in ('ankle.L', 'ankle.R')]",
                   "--frames", "1,40", "--project", str(make.root)], capsys)
    assert code == 0, q
    (l0, r0), (l1, r1) = (np.array(v) for v in q["values"][0]), (np.array(v) for v in q["values"][1])
    assert l1[2] - l0[2] == pytest.approx(0.15, abs=0.005) and np.abs(r1 - r0).max() < 0.005
    code, out = make('[pose.mq]\nlift = { left = 0.15 }\n', until="pose")
    assert code == 3 and "'left'" in out["error"] and "L, R" in out["error"]


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


def test_a_boombox_pump_swells_both_speaker_grilles_about_their_centres_and_pushes_them_out(make, capsys):
    import numpy as np
    keyed = SCALED.format(where="at = [0.0, 1.0, 0.0]") + ('[[key]]\ntarget = "box"\nprop = "pump"\n'
                                                           'keys = [[0.0, 0.0], [1.0, 1.0]]\n')
    code, out = make(keyed, with_cast=False, until="keys")
    assert code == 0, out.get("error")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), '[obj(f"box_grille_{s}").dims for s in "LR"] + '
                   '[obj(f"box_grille_{s}").loc for s in "LR"]', "--frames", "31,61"], capsys)   # pump 0, then 1
    assert code == 0, q
    (dl0, dr0, l0, r0), (dl1, dr1, l1, r1) = ([np.array(v) for v in f] for f in q["values"])
    assert dl0[0] == pytest.approx(4 * 2 * 0.0405, rel=0.01) and dl1[0] == pytest.approx(1.12 * dl0[0], rel=1e-3)
    assert dr1[2] == pytest.approx(1.12 * dr0[2], rel=1e-3)                  # both grilles, across and up
    for p0, p1, x in ((l0, l1, 4 * 0.1145), (r0, r1, -4 * 0.1145)):
        assert p0[0] == pytest.approx(x, abs=1e-4) and p1[0] == pytest.approx(x, abs=1e-4)   # about the speaker's centre
        assert p1[1] - p0[1] == pytest.approx(-4 * 0.004, abs=1e-4)                        # out of the front (-Y)


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


MOVES = """[pose.mq]
feet = "floor"

[[move.mq]]
name = "point"
t = 0.5
dur = 0.8
hand = "L"
{extra}"""


def test_a_move_holds_its_place_to_its_end_then_the_hand_goes_home_and_a_posed_hand_is_not_moved(make, capsys):
    code, out = make(MOVES.format(extra=""), until="pose")
    assert code == 0, out.get("error")
    assert not [w for w in warnings(out) if "short of its goal" in w]                     # the arm out is within reach
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "list(bone('wrist.L').head)", "--frames", "34,58,68,85",
                   "--project", str(make.root)], capsys)                                  # t 0.1, 0.9, 1.23, 1.8
    import numpy as np
    before, mid, end, after = (np.array(v) for v in q["values"])
    assert np.linalg.norm(mid - before) > 0.15                                           # it pointed
    assert np.linalg.norm(end - mid) < 3e-3                                              # and held to the end
    assert np.linalg.norm(after - before) < 0.01                                         # then went home
    code, out = make(MOVES.format(extra="\n[pose.mq.hands.L]\nat = [0.3, -0.2, 1.0]\n"), until="pose")
    assert code == 3 and "[pose.mq.hands.L] places that hand and [[move.mq]] moves it" in out["error"]


FREE = """[pose.mq]
feet = "floor"

[[motion.mq]]
vmd = "{vmd}"
bones = "arms"
{moves}"""


def _vmd(path, keys):
    """Write a VMD of bone keys {PMX bone name: [(frame, quaternion x y z w in MMD's axes)]}, linear between them."""
    interp = bytes([20] * 8 + [107] * 8) * 4
    recs = [(n, f, q) for n, ks in keys.items() for f, q in ks]
    out = b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"mq".ljust(20, b"\0") + struct.pack("<I", len(recs))
    for n, f, q in recs:
        out += n.encode("shift_jis").ljust(15, b"\0") + struct.pack("<I7f", f, 0.0, 0.0, 0.0, *q) + interp
    path.write_bytes(out + struct.pack("<4I", 0, 0, 0, 0))


def test_a_hand_let_go_follows_the_arm_strip_and_the_ik_lets_go_and_takes_over_where_the_strip_has_it(make, capsys):
    """A `free` move eases the arm IK off over its span, so a VMD strip of the arms (a walk's swing) drives the arm there,
    the pose stage's own forearm key composing with the strip; before and after it the IK holds the hand at its rest
    place though the strip goes on (it holds its pose over the whole clip). The hand travels to where the strip has it
    and back at the human pace, and while the IK eases off and on its goal and pole are on the strip's hand and elbow:
    the arm is the strip's there too (the goal used to wait at the rest place, so the arm swept between the two)."""
    vmd = make.root / "arm.vmd"
    up, low = (0.0, 0.0, 0.4226, 0.9063), (0.0, 0.0, 0.2588, 0.9659)          # the arm raised 50 deg, then 30
    fold = (0.0, 0.6428, 0.0, 0.766)                                            # the elbow bent 80 deg
    _vmd(vmd, {"左腕": [(0, up), (60, low)], "左ひじ": [(0, fold), (60, fold)]})
    moves = ('\n[[move.mq]]\nname = "rest"\nhand = "L"\nplace = "rest"\n'
             '\n[[move.mq]]\nname = "free"\nt = 0.5\ndur = 0.8\nhand = "L"\n')
    got = {}
    for run, extra in (("strip", ""), ("free", moves)):
        code, out = make(FREE.format(vmd=vmd, moves=extra), until="motion")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), "[list(bone(b).head) for b in ('wrist.L', 'elbow.L')]",
                       "--frames", "34,49,61,67,88", "--project", str(make.root)], capsys)
        got[run] = [[np.array(p) for p in v] for v in q["values"]]       # t 0.1, 0.6 and 1.2 (the IK half on), 1.0, 1.9
    (before, _), (off_w, off_e), (free, _), (on_w, on_e), (after, _) = got["free"]
    strip = got["strip"]
    assert np.linalg.norm(free - strip[2][0]) < 2e-3                            # let go: the strip's arm, elbow and all
    for (w, e), (sw, se) in (((off_w, off_e), strip[1]), ((on_w, on_e), strip[3])):
        assert np.linalg.norm(w - sw) < 5e-3 and np.linalg.norm(e - se) < 0.01   # the IK half on: still the strip's arm
    assert np.linalg.norm(after - before) < 5e-3                                # held at the rest place on both sides ...
    assert np.linalg.norm(before - strip[0][0]) > 0.1                           # ... though the strip is raising the arm


def test_a_hand_let_go_to_a_strip_that_plays_its_fingers_curls_them_as_the_strip_does_not_more(make, capsys):
    """The action of a member with a hand let go blends with Combine, so the hand's own finger keys (a rest place's
    relaxed curl) used to compose with the strip's fingers: a strip's curl of 40 deg became 54. While the hand is let
    go its finger keys give way by its IK share, as its arm does; outside the span the hand's own curl is back."""
    vmd = make.root / "hand.vmd"
    curl = (0.0, 0.0, 0.3420, 0.9397)                                           # the index's base joint, 40 deg
    _vmd(vmd, {"左腕": [(0, (0.0, 0.0, 0.4226, 0.9063)), (60, (0.0, 0.0, 0.4226, 0.9063))], "左人指１": [(0, curl), (60, curl)]})
    strip = ('[pose.mq]\nfeet = "floor"\n\n[[motion.mq]]\nvmd = "{vmd}"\nbones = ["arms", "index1.L"]\nstart = 0.2\n'
             'from = 0\nto = 42\nextend = "none"\nblend_in = 0.25\nblend_out = 0.25\n{moves}')   # 0.2-1.6 s, full 0.45-1.35
    moves = ('\n[[move.mq]]\nname = "rest"\nhand = "L"\nplace = "rest"\n'
             '\n[[move.mq]]\nname = "free"\nt = 0.5\ndur = 0.8\nhand = "L"\n')
    got = {}
    for run, extra in (("strip", ""), ("free", moves)):
        code, out = make(strip.format(vmd=vmd, moves=extra), until="motion")
        assert code == 0, out.get("error")
        assert not warnings(out), warnings(out)                                 # the strip covers the span fully
        code, q = cli(["q", str(make.root / "build" / "t.blend"), "math.degrees(bone('index1.L').local_quat.angle)",
                       "--frames", "34,61", "--project", str(make.root)], capsys)   # t 0.1 (held), 1.0 (let go)
        got[run] = q["values"]
    assert got["strip"][1] == pytest.approx(40.0, abs=0.5)
    assert got["free"][1] == pytest.approx(got["strip"][1], abs=0.5)            # the strip's curl, not 40 + 14
    assert got["free"][0] == pytest.approx(14.0, abs=1.0)                        # the rest place's relaxed curl outside


def test_a_strip_plays_a_share_of_its_motion_and_a_mirrored_one_plays_it_on_the_other_side(make, capsys):
    """`influence = 0.5` plays half the motion (the arm raised half as far, eased in and out over the blends);
    `mirror = true` plays it left for right: the right arm raised as the motion raises the left."""
    vmd = make.root / "arm.vmd"
    _vmd(vmd, {"左腕": [(0, (0.0, 0.0, 0.4226, 0.9063)), (60, (0.0, 0.0, 0.4226, 0.9063))]})     # raised 50 deg
    expr = "[list(bone(b).head) for b in ('arm.L', 'wrist.L', 'arm.R', 'wrist.R')]"
    got = {}
    for run, extra in (("rest", None), ("full", ""), ("half", "influence = 0.5\n"), ("mirror", "mirror = true\n")):
        code, out = make('[pose.mq]\nfeet = "floor"\n' if extra is None else FREE.format(vmd=vmd, moves=extra),
                         until="motion")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "61", "--project", str(make.root)],
                      capsys)
        got[run] = [np.array(p) for p in q["values"][0]]

    def raised(run, side):
        i = 0 if side == "L" else 2
        a, b = got[run][i + 1] - got[run][i], got["rest"][i + 1] - got["rest"][i]
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))
    assert raised("full", "L") == pytest.approx(50.0, abs=5.0) and raised("full", "R") < 1.0
    assert raised("half", "L") == pytest.approx(raised("full", "L") / 2, abs=3.0)
    assert raised("mirror", "R") == pytest.approx(raised("full", "L"), abs=1.0) and raised("mirror", "L") < 1.0
    code, out = make(FREE.format(vmd=vmd, moves="influence = 0\n"), until="motion")
    assert code == 3 and "influence = 0" in out["error"]


def test_a_strip_eases_in_and_out_over_its_blends(make, capsys):
    """`blend_in` / `blend_out` ramp a strip's influence: halfway through each the arm is partway between where it hangs
    and the motion's pose, and between them it is the motion's; without them it snaps in and out (a dance that stops for
    a pose must not jump)."""
    vmd = make.root / "arm.vmd"
    up = (0.0, 0.0, 0.4226, 0.9063)                                            # the arm raised 50 deg
    _vmd(vmd, {"左腕": [(0, up), (60, up)]})
    strip = 'start = 0.5\nfrom = 0\nto = 30\nextend = "none"\n'               # frames 46-76: its own frames only
    wrist = {}
    for run, blends in (("snap", ""), ("ease", "blend_in = 0.4\nblend_out = 0.4\n")):
        code, out = make(FREE.format(vmd=vmd, moves=strip + blends), until="motion")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), "list(bone('wrist.L').head)", "--frames",
                       "40,52,61,70,82", "--project", str(make.root)], capsys)   # before, in, held, out, after
        wrist[run] = [np.array(v) for v in q["values"]]
    rest, full = wrist["snap"][0], wrist["snap"][2]
    assert np.linalg.norm(full - rest) > 0.1
    assert np.linalg.norm(wrist["snap"][1] - full) < 2e-3 and np.linalg.norm(wrist["snap"][3] - full) < 2e-3
    before, ease_in, held, ease_out, after = wrist["ease"]
    assert np.linalg.norm(before - rest) < 2e-3 and np.linalg.norm(after - rest) < 2e-3
    assert np.linalg.norm(held - full) < 2e-3
    for w in (ease_in, ease_out):
        assert 0.2 < np.linalg.norm(w - rest) / np.linalg.norm(full - rest) < 0.8


def test_a_strip_that_extends_forward_holds_its_last_pose_after_it_and_nothing_before_it(make, capsys):
    """`extend = "forward"` keeps a strip's last pose after it and leaves the frames before it alone ("hold" keeps its
    first pose before it too, "none" neither), and a later entry plays over an earlier one: a dance paused between two
    strips, the later picking up where the earlier stopped."""
    vmd = make.root / "arm.vmd"
    down, up, half = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.4226, 0.9063), (0.0, 0.0, 0.2164, 0.9763)    # 0, 50, 25 deg
    _vmd(vmd, {"左腕": [(0, up), (30, half), (60, down)]})
    strip = 'start = 0.5\nfrom = 0\nto = 30\nextend = "{}"\n'                  # frames 46-76: raised, then half way
    got = {}
    for extend in ("none", "hold", "forward"):
        code, out = make(FREE.format(vmd=vmd, moves=strip.format(extend)), until="motion")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), "list(bone('wrist.L').head)", "--frames", "40,76,85",
                       "--project", str(make.root)], capsys)                    # before it, its last frame, after it
        got[extend] = [np.array(v) for v in q["values"]]
    rest = got["none"][0]
    assert np.linalg.norm(got["none"][2] - rest) < 2e-3                        # "none": nothing before or after
    assert np.linalg.norm(got["hold"][0] - rest) > 0.05                        # "hold": raised before it
    before, end, after = got["forward"]
    assert np.linalg.norm(end - rest) > 0.05
    assert np.linalg.norm(before - rest) < 2e-3 and np.linalg.norm(after - end) < 2e-3
    later = strip.format("forward") + f'\n[[motion.mq]]\nvmd = "{vmd}"\nbones = "arms"\nstart = 1.5\nfrom = 60\nto = 61\n' \
                                      'extend = "forward"\n'                    # frames 76-77: down, held after
    code, out = make(FREE.format(vmd=vmd, moves=later), until="motion")
    assert code == 0, out.get("error")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "list(bone('wrist.L').head)", "--frames", "85",
                   "--project", str(make.root)], capsys)
    assert np.linalg.norm(np.array(q["values"][0]) - rest) < 2e-3            # the later entry wins


def test_a_motion_that_switches_the_leg_ik_off_leaves_the_member_s_ik_on(make, capsys):
    """A VMD whose first frame switches the leg IK off (FK legs, as a BVH conversion's are) made mmd_tools switch the
    member's IK off for the whole scene, so every later stage's legs lost their IK: the stage keeps it on, and the thigh
    the motion swings by FK gives way to it (the ankle stays on its IK bone)."""
    vmd = make.root / "fk.vmd"
    _vmd(vmd, {"左足": [(0, (0.5, 0.0, 0.0, 0.866)), (60, (0.5, 0.0, 0.0, 0.866))]})      # the thigh swung 60 deg
    ik_off = b"".join(n.encode("shift_jis").ljust(20, b"\0") + b"\0" for n in ("左足ＩＫ", "右足ＩＫ"))
    vmd.write_bytes(vmd.read_bytes() + struct.pack("<IIBI", 1, 0, 1, 2) + ik_off)          # frame 0: both IKs off
    code, out = make(f'[[motion.mq]]\nvmd = "{vmd}"\nstart = 0.0\n', until="motion")
    assert code == 0, out.get("error")
    expr = ("[[c.influence for o in bpy.data.objects if o.type == 'ARMATURE' for pb in o.pose.bones "
            "for c in pb.constraints if c.type == 'IK' and c.subtarget], list(bone('ankle.L').head), "
            "list(bone('leg_ik.L').head)]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "61", "--project", str(make.root)],
                  capsys)
    influence, ankle, ik = q["values"][0]
    assert influence and min(influence) == 1.0                                  # the IK still on
    assert np.linalg.norm(np.subtract(ankle, ik)) < 0.02                        # the swung thigh gave way to it


def test_a_track_keyed_on_the_far_side_of_the_rest_blends_in_the_short_way(make, capsys):
    """q and -q are one rotation, but Blender blends quaternions component by component: a track keyed at w < 0 (as BVH
    conversions write whole tracks) swung the arm half a turn round on its way in over a blend. Turned to the near side
    at the import, it eases in as the same motion keyed at w > 0 does."""
    vmd = make.root / "arm.vmd"
    strip = 'start = 0.5\nfrom = 0\nto = 30\nextend = "none"\nblend_in = 0.4\nblend_out = 0.4\n'   # frames 46-76
    wrist = {}
    for run, up in (("near", (0.0, 0.0, 0.4226, 0.9063)), ("far", (0.0, 0.0, -0.4226, -0.9063))):   # raised 50 deg
        _vmd(vmd, {"左腕": [(0, up), (60, up)]})
        code, out = make(FREE.format(vmd=vmd, moves=strip), until="motion")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), "list(bone('wrist.L').head)", "--frames", "52,61",
                       "--project", str(make.root)], capsys)                    # halfway in, held
        wrist[run] = [np.array(v) for v in q["values"]]
    for near, far in zip(wrist["near"], wrist["far"]):
        assert np.linalg.norm(near - far) < 2e-3


def test_a_camera_riding_her_head_bone_stays_where_it_is_on_her_head_as_the_head_turns(make, capsys):
    """`at = {cast, bone, point}` is a point given where it is at rest, carried by that bone's move: a close-up in front
    of her face keeps its place on her head while a motion turns the head 40 deg (without the bone the camera stays in
    the member's frame and the head turns away from it)."""
    vmd = make.root / "head.vmd"
    _vmd(vmd, {"頭": [(0, (0.0, 0.0, 0.0, 1.0)), (30, (0.0, 0.342, 0.0, 0.9397))]})
    body = (f'[[motion.mq]]\nvmd = "{vmd}"\nbones = ["head"]\n\n[[shot]]\nname = "eye"\nfrom = 0.0\nto = 2.0\n'
            'at = { cast = "mq", bone = "head", point = [0.0, -0.35, 1.45] }\nlook = "cast:mq.head"\nlens = 50\n')
    code, out = make(body, until="shots")
    assert code == 0, out.get("error")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "[list(cam().loc), list(bone('head').matrix.inverted() @ "
                   "cam().loc)]", "--frames", "31,61", "--project", str(make.root)], capsys)
    (world0, on_head0), (world1, on_head1) = q["values"]
    assert np.linalg.norm(np.subtract(world1, world0)) > 0.1                     # the head turned and took it along
    assert np.linalg.norm(np.subtract(on_head1, on_head0)) < 1e-3


CLIPPED = """[pose.mq]
feet = "floor"

[[prop]]
name = "ground"
card = "library:floor"

[[shot]]
name = "far"
from = 0.0
to = 1.0
at = [0.0, -4.0, 1.0]
look = [0.0, 0.0, 1.0]
lens = 50

[[shot]]
name = "low"
from = 1.0
to = 2.0
at = [0.0, -4.0, 0.15]
look = [0.0, 0.0, 0.15]
lens = 50
"""


def test_a_shot_camera_starts_half_way_to_the_nearest_thing_it_sees_and_cuts_nothing_off(make, capsys):
    """On every frame each shot camera starts at half the depth of the nearest geometry in its view (widened by a tenth
    on each side): 4 m in front of her at chest height that is her front, a few centimetres short of 4 m (it started at
    2 cm, and a render that starts there sees what a model hides behind its skin from a couple of metres); 15 cm over
    the floor it is where the bottom of the view meets the floor, 0.15 m over that side's tangent, per output."""
    code, out = make(CLIPPED, until="shots")
    assert code == 0, out.get("error")
    assert out["stages"]["clip"]["cameras"] == 4
    names = ["far@16x9", "low@16x9", "low@9x16"]
    code, q = cli(["q", str(make.root / "build" / "t.blend"), f"[bpy.data.objects[n].data.clip_start for n in {names!r}]",
                   "--frames", "46,76", "--project", str(make.root)], capsys)
    (far, _, _), (_, low, tall) = q["values"]
    assert 1.8 < far < 2.0, far
    assert low == pytest.approx(0.5 * 0.15 / (1.1 * 180 * 36.0 / 50.0 / 640), rel=1e-4)       # the 640 x 360 picture
    assert tall == pytest.approx(0.5 * 0.15 / (1.1 * 320 * 36.0 / 50.0 / 640), rel=1e-4)      # 360 x 640


MIC_LEAN = """[pose.mq]
feet = "floor"

[[move.mq]]
name = "rest"
hand = "R"
place = "mic"

[[move.mq]]
name = "chest_pat"
t = 0.3
dur = 0.4
hand = "L"

[[move.mq]]
name = "lean_back"
t = 1.0
dur = 0.8
"""


def test_moved_hands_ride_the_chest_through_a_lean_and_come_in_to_the_body_from_outside(make, capsys):
    (make.root / "audio").mkdir(exist_ok=True)
    (make.root / "audio" / "timeline.json").write_text(json.dumps({"beats": [0.3, 0.5]}), encoding="utf-8")
    code, out = make(MIC_LEAN, until="perform")
    assert code == 0, out.get("error")
    places = out["stages"]["pose"]["moves"]["mq"]["places"]
    assert places["L"]["chest"]["moved_mm"] > 0                   # written inside the chest: brought out onto it
    assert places["L"]["rest"]["moved_mm"] < 30.0                 # a hanging hand is nudged off the hip at most, not
    #                                                               thrown off as if inside its own arm (excluded)
    expr = "[list(bone('upper_body2').matrix.inverted() @ bone('wrist.R').head), list(bone('wrist.R').head)]"
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "55,73", "--project", str(make.root)],
                  capsys)                                         # t 0.8 (before the lean), 1.4 (leaning back)
    assert code == 0, q
    import numpy as np
    (chest0, world0), (chest1, world1) = ([np.array(v) for v in f] for f in q["values"])
    assert np.linalg.norm(world1 - world0) > 0.02                 # the lean moved the mic hand in the world ...
    assert np.linalg.norm(chest1 - chest0) < 3e-3                 # ... with the chest: the mic stays at the mouth
    code, q = cli(["q", str(make.root / "build" / "t.blend"),
                   "[list(bone(b).head) for b in ('arm.L', 'elbow.L', 'wrist.L')] + [list(obj('mq_elbow.L').loc)]",
                   "--frames", "40", "--project", str(make.root)], capsys)                # t 0.3: on the chest
    S, E, W, P = (np.array(v) for v in q["values"][0])
    from mkmmd.core import armreach as AR
    predicted = AR.pole_elbow(S, W, np.linalg.norm(E - S), np.linalg.norm(W - E), P)   # its pole: the stage's own, or
    assert np.linalg.norm(predicted - E) < 5e-3                   # the one the move library turned the elbow to
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "[list(bone(b).head) for b in ('arm.R', 'elbow.R', 'wrist.R')]",
                   "--frames", "40", "--project", str(make.root)], capsys)                # the mic hand
    S, E, W = (np.array(v) for v in q["values"][0])
    planned = AR.pole_elbow(S, W, np.linalg.norm(E - S), np.linalg.norm(W - E), S + np.array(AR.POLE) * [-1, 1, -1])
    assert np.linalg.norm(planned - E) < 5e-3 and S[2] - E[2] > 0.12  # its elbow hangs under the shoulder, as planned


PEEK_ROCK = """[pose.mq]
feet = "floor"

[[move.mq]]
name = "peek"
t = 0.5
dur = 1.0
hand = "L"

[perform.mq]
head_rock = { deg = 6.0, period = 0.8 }
"""


def test_a_peek_s_hand_goes_with_the_head_over_its_hold_as_the_head_rocks(tmp_path, mannequin, capsys):
    """The ring at the eye (a place seen through, mkmmd.core.moves.SIGHTED) is held with the head from its arrival to the
    end of its hold, not with the chest the moved hands ride: the move's own lean and head tilt and a head rocking on top
    (perform's head_rock) carry the hand with the head, so the ring stays on the eye."""
    rig = json.loads(Path(mannequin[1]).read_text(encoding="utf-8"))
    rig["morphs"]["mouth_smile"] = rig["morphs"]["smile_eyes"]           # the mannequin's one smile, for the peek's face
    (tmp_path / "mq.rig.json").write_text(json.dumps(rig), encoding="utf-8")
    (tmp_path / "mk.toml").write_text(HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{mannequin[0]}"\n'
                                      f'rig = "{tmp_path / "mq.rig.json"}"\n\n' + PEEK_ROCK, encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "perform"], capsys)
    assert code == 0, out.get("error")
    assert out["stages"]["pose"]["mq"]["sight"] == {"L": [[46.0, 76.0]]}          # t 0.5 to 1.5
    expr = ("[list(bone('head').matrix.inverted() @ bone('wrist.L').head), "
            "list(bone('upper_body2').matrix.inverted() @ bone('wrist.L').head)]")
    code, q = cli(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "46,52,58,64,70,76",
                   "--project", str(tmp_path)], capsys)
    assert code == 0, q
    import numpy as np
    head, chest = (np.array([v[i] for v in q["values"]]) for i in (0, 1))
    assert np.linalg.norm(head - head[0], axis=1).max() < 2e-3          # still against the head from its arrival on ...
    assert np.linalg.norm(chest - chest[0], axis=1).max() > 5e-3        # ... as the head rocks it about over the chest



@pytest.fixture(scope="module")
def grouped(mannequin):
    """The mannequin with a group morph `口`: half its `あ` and all its `笑い`."""
    m = X.read(str(mannequin[0]))
    names = [x.name for x in m.morphs]
    m.morphs.append(X.PmxMorph("口", "mouth", 3, "group", [(names.index("あ"), 0.5), (names.index("笑い"), 1.0)]))
    path = mannequin[0].with_name("mq_group.pmx")
    X.write(m, str(path))
    return path


def test_a_group_morph_is_keyed_through_the_bound_sliders_and_sums_with_its_members(tmp_path, mannequin, grouped, capsys):
    (tmp_path / "mk.toml").write_text(
        HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{grouped}"\nrig = "{mannequin[1]}"\n\n[perform.mq]\n'
        'expressions = [{ morph = "口", keys = [[0.0, 0.0], [1.0, 0.8]] }, { morph = "a", keys = [[0.0, 0.2], [1.0, 0.2]] }]\n',
        encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--until", "perform"], capsys)
    assert code == 0, out.get("error")                                    # was: morph '口' is not on the model
    assert out["stages"]["cast"]["mq"]["bound_morphs"] == 1
    expr = ('[[round(o.data.shape_keys.key_blocks[k].value, 4) for k in ("あ", "笑い")] for o in arm().parent.'
            'children_recursive if o.type == "MESH" and o.mmd_type == "NONE" and o.data.shape_keys]')    # not the sliders
    code, q = cli(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "61", "--project", str(tmp_path)], capsys)
    assert code == 0, q                                                   # a new Blender: the drivers survive the reload
    assert q["values"][0] == [[pytest.approx(0.2 + 0.5 * 0.8), pytest.approx(0.8)]]


FUSES = """[[prop]]
name = "fa"
card = "library:fuse"
at = [0.3, 0.2, 0.5]

[[prop]]
name = "fb"
card = "library:fuse"
at = [-0.4, 0.1, 0.2]
scale = 2.0
slots = { lit = 1.0 }

[[key]]
target = "fa"
prop = "lit"
keys = [[0.0, 1.0]]

[[key]]
target = "fb"
prop = "burn"
keys = [[1.0, 0.0], [1.5, 0.97]]
"""


def test_a_fuse_s_spark_rides_the_end_of_its_rope_wherever_the_prop_stands_and_whatever_its_size(make, capsys):
    import numpy as np
    from mkmmd.blender.library.props import fuse_layout as FL
    code, out = make(FUSES, with_cast=False, until="keys")
    assert code == 0, out.get("error")
    assert not warnings(out)                                              # built grown: the rope is a round form
    expr = ("[list(obj(n + '_spark').matrix.translation) + [obj(n + '_spark').matrix.to_scale()[0]] for n in ('fa', 'fb')]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "61,76", "--project", str(make.root)],
                  capsys)                                                 # t 1.0 (grown), 1.5 (fb burned down)
    assert code == 0, q
    (fa1, fb1), (_, fb2) = ([np.array(v) for v in f] for f in q["values"])
    tip = np.array(FL.CURL[-1])
    assert np.linalg.norm(fa1[:3] - (np.array([0.3, 0.2, 0.5]) + tip)) < 2e-3          # at the tip, counted once
    assert np.linalg.norm(fb1[:3] - (np.array([-0.4, 0.1, 0.2]) + 2.0 * tip)) < 4e-3   # the scaled rope's tip
    assert fb1[3] / fa1[3] == pytest.approx(2.0, rel=1e-3)                # the spark grows with the prop
    assert np.linalg.norm(fb2[:3] - np.array([-0.4, 0.1, 0.2])) < 0.03   # burned down to the socket


BURST = """[[prop]]
name = "bz"
card = "library:burst"
at = [0.2, 0.5, 1.0]
scale = 2.0
slots = { start = 0.6, count = 9, seed = 3, reach = 0.8, life = 1.0 }

[[key]]
target = "bz"
prop = "reach"
keys = [[0.0, 0.5]]
"""


def test_a_burst_s_pieces_are_where_the_core_puts_them_on_a_moved_scaled_prop_following_its_keys(make, capsys):
    import numpy as np
    from mkmmd.core import burst as BU
    code, out = make(BURST, with_cast=False, until="keys")
    assert code == 0, out.get("error")
    assert not warnings(out)
    count, seed, mix, P, _, _ = BU.options({"start": 0.6, "count": 9, "seed": 3, "reach": 0.8, "life": 1.0})
    P["reach"] = 0.5                                                       # keyed: the root's property is live
    names, at = [], {k: 0 for k in BU.KINDS}
    pieces = BU.layout(count, seed, mix)
    for p in pieces:
        names.append(f"bz_{p['kind']}{at[p['kind']]}")
        at[p["kind"]] += 1
    expr = (f"[[list(m.translation) + list(m.to_scale()) + list(m.to_euler()) for m in "
            f"(bpy.data.objects[n].matrix_world for n in {names!r})], "
            f"sorted(o.name for o in bpy.data.objects if o.get('mk_heart'))]")
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "46,58,88", "--project", str(make.root)],
                  capsys)                                                 # t 0.5 (before), 0.9 (flying), 1.9 (gone)
    assert code == 0, q
    root = np.array([0.2, 0.5, 1.0])
    for (got, hearts), t in zip(q["values"], (0.5, 0.9, 1.9)):
        assert hearts == sorted(n for n in names if "_heart" in n)       # the hearts are tagged as accents
        for g, p in zip(np.array(got), pieces):
            pos, rx, ry, s = BU.pose(t, P, p)
            assert g[:3] == pytest.approx(root + 2.0 * pos, abs=1e-5)     # in the prop's frame, scaled with it
            assert g[3:6] == pytest.approx([2.0 * s] * 3, rel=1e-4)
            if t == 0.9:
                assert g[6:] == pytest.approx([rx, ry, 0.0], abs=1e-4)    # tumbling a little, spinning in the picture


HAMMER = """[[prop]]
name = "hm"
card = "library:squeaky_hammer"
at = [0.3, 0.2, 0.5]
scale = 2.0

[[key]]
target = "hm"
prop = "squash"
keys = [[0.0, 0.0], [1.0, 1.0]]
"""


def test_a_squeaky_hammer_s_head_squashes_about_its_middle_as_its_squash_is_keyed(make, capsys):
    from mkmmd.blender.library.props import squeaky_hammer_layout as LAY
    code, out = make(HAMMER, with_cast=False, until="keys")
    assert code == 0, out.get("error")
    assert not warnings(out)
    expr = "[list(obj('hm_head').matrix.translation), list(obj('hm_head').matrix.to_scale())]"
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "31,61", "--project", str(make.root)],
                  capsys)                                                 # t 0 (squash 0), 1.0 (squash 1)
    assert code == 0, q
    middle = [0.3, 0.2, 0.5 + 2.0 * LAY.HEAD_Z]
    (at0, s0), (at1, s1) = q["values"]
    assert at0 == pytest.approx(middle, abs=1e-5) and at1 == pytest.approx(middle, abs=1e-5)   # it stays in place
    assert s0 == pytest.approx([2.0] * 3, rel=1e-5)
    along, round_ = LAY.squash(1.0)
    assert s1 == pytest.approx([2.0 * along, 2.0 * round_, 2.0 * round_], rel=1e-5)          # shorter along X, rounder


APPEARS = """[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }

[vector.tones]
dark = "#1B2A6B"

[[vector.materials]]
match = "*"
tone = "dark"

[[prop]]
name = "hm"
card = "library:squeaky_hammer"
at = [0.0, 0.0, 0.3]

[[key]]
target = "hm_head"
prop = "hide_render"
interp = "CONSTANT"
keys = [[0.0, 1.0], [1.0, 0.0]]

[[key]]
target = "hm_handle"
prop = "hide_render"
interp = "CONSTANT"
keys = [[0.0, 1.0], [1.0, 0.0]]

[[shot]]
name = "a"
from = 0.0
to = 1.2
at = [0.0, -1.0, 0.45]
look = [0.0, 0.0, 0.45]
lens = 35
style = "vector"

[[shot]]
name = "b"
from = 1.2
to = 2.0
at = [0.0, -1.0, 0.45]
look = [0.0, 0.0, 0.45]
lens = 35
style = "vector"
hide = ["hm_*"]
"""


def test_a_flat_look_draws_what_keys_show_after_its_shot_starts_and_hides_what_it_is_told_to_whatever_the_keys(make, capsys,
                                                                                                            tmp_path):
    """A look sorts the objects when its shot starts: one hidden there by its keys and shown later must be drawn in its
    tone (it was drawn with no id, and the look failed), and one the shot hides must stay hidden though its keys show it."""
    import numpy as np
    from PIL import Image
    code, out = make(APPEARS, with_cast=False)
    assert code == 0, out.get("error")
    code, look = cli(["look", str(make.root / "build" / "t.blend"), "--frames", "46,64,76", "--output", "16x9", "--size",
                      "320", "--out", str(tmp_path / "look"), "--project", str(make.root)], capsys)
    assert code == 0, look.get("error")                           # t 0.5 keyed hidden, 1.1 keyed shown, 1.5 shot b hides it
    navy = [int((np.abs(np.asarray(Image.open(p).convert("RGB"), int) - (27, 42, 107)).max(-1) < 40).sum())
            for p in look["images"]]
    assert navy[0] == 0 and navy[1] > 500 and navy[2] == 0


GLITCHED = """[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }
[vector.tones]
dark = "#1B2A6B"
[[vector.materials]]
match = "*"
tone = "dark"
[pose.mq]
feet = "floor"
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
style = "vector"
[[glitch]]
from = 0.5
to = 1.0
objects = ["Mq_*"]
"""


def test_a_glitch_breaks_up_the_objects_it_names_and_they_stay_gone_after_it(make, capsys, tmp_path):
    """`mk render` / `mk look` draw the shot without the glitch's objects (its bare frames); the composite breaks the
    figure up over the window and leaves the bare frame on its last frame, where the scene itself still shows it; from
    the frame the window ends on the build has them hidden."""
    import numpy as np
    from PIL import Image
    code, out = make(GLITCHED)
    assert code == 0, out.get("error")
    assert out["stages"]["shots"]["_cut_effects"]["glitches"][0]["frames"] == [46, 60]
    assert out["stages"]["keys"]["_glitch_hidden"] == {"0": 1}         # the mannequin's one mesh

    def navy(*extra):
        code, look = cli(["look", str(make.root / "build" / "t.blend"), "--frames", "40,53,60,70", "--output", "16x9",
                          "--size", "320", "--out", str(tmp_path / ("look" + "".join(extra))), "--project", str(make.root),
                          *extra], capsys)
        assert code == 0, look.get("error")
        return [int((np.abs(np.asarray(Image.open(p).convert("RGB"), int) - (27, 42, 107)).max(-1) < 40).sum())
                for p in look["images"]]
    plain, glitched = navy("--no-transitions"), navy()
    assert min(plain[:3]) > 500 and plain[3] == 0                 # in the scene up to the window's end, hidden after it
    assert glitched[0] == plain[0] and glitched[2] == 0           # before the window as it is; gone on its last frame


BAKED = """[pose.mq]
feet = "floor"
[[move.mq]]
name = "hands_up"
t = 0.2
dur = 1.2
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
keys = [{ t = 0.0 }, { t = 2.0, at = [0.6, -2.6, 0.9] }]
[[freeze]]
from = 1.5
to = 1.8
"""


def read_glb(path):
    """(document, accessor(i) -> float array) of a .glb, sparse accessors filled in."""
    import struct
    import numpy as np
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 12)[0]
    doc, binary = json.loads(data[20:20 + n]), data[20 + n + 8:]
    comp = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    size = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def view(bv_i, offset, count, dt, k):
        bv = doc["bufferViews"][bv_i]
        return np.ndarray((count, k), dt, binary, bv.get("byteOffset", 0) + offset,
                          (bv.get("byteStride") or dt.itemsize * k, dt.itemsize))

    def accessor(i):
        a = doc["accessors"][i]
        dt, k = np.dtype(comp[a["componentType"]]), size[a["type"]]
        out = view(a["bufferView"], a.get("byteOffset", 0), a["count"], dt, k).astype(float) if "bufferView" in a \
            else np.zeros((a["count"], k))
        if "sparse" in a:
            s = a["sparse"]
            idx = view(s["indices"]["bufferView"], s["indices"].get("byteOffset", 0), s["count"],
                       np.dtype(comp[s["indices"]["componentType"]]), 1)[:, 0]
            out[idx] = view(s["values"]["bufferView"], s["values"].get("byteOffset", 0), s["count"], dt, k)
        return out / np.iinfo(dt).max if a.get("normalized") else out
    return doc, accessor


def posed(bake_dir, name, frame):
    """The mesh `name` on Blender frame `frame` from scene.glb and the bake alone, as mkmmd/blender/build/bake.py says a
    viewer places it (morphs, node matrix, skin), in the world in glTF's axes."""
    import numpy as np
    doc, acc = read_glb(bake_dir / "scene.glb")
    bake = json.loads((bake_dir / "bake.json").read_text(encoding="utf-8"))
    blob = np.fromfile(bake_dir / "bake.bin", "<f4")
    k = frame - bake["frame0"]
    node = next(n for n in doc["nodes"] if n.get("name") == name)
    entry = next(e for e in bake["nodes"] if e["name"] == name)
    o = entry["offset"] + 12 * (k if entry["frames"] > 1 else 0)
    N = blob[o:o + 12].reshape(4, 3).T
    morph = next((e for e in bake["morphs"] if e["name"] == name), None)
    weights = blob[morph["offset"] + morph["count"] * k:][:morph["count"]] if morph else []
    J = None
    if "skin" in node:
        arm = next(a for a in bake["armatures"] if a["name"] == entry["armature"])
        nb, at = len(arm["bones"]), {b: i for i, b in enumerate(arm["bones"])}
        every = blob[arm["offset"] + nb * 12 * k:][:nb * 12].reshape(nb, 4, 3).transpose(0, 2, 1)
        J = every[[at[doc["nodes"][j]["name"]] for j in doc["skins"][node["skin"]]["joints"]]]
    out = []
    for p in doc["meshes"][node["mesh"]]["primitives"]:
        v = acc(p["attributes"]["POSITION"])
        for w, t in zip(weights, p.get("targets", [])):
            v = v + w * acc(t["POSITION"])
        v = v @ N[:, :3].T + N[:, 3]
        if J is not None:
            v = np.einsum("vk,vkij,vj->vi", acc(p["attributes"]["WEIGHTS_0"]),
                          J[acc(p["attributes"]["JOINTS_0"]).astype(int)], np.c_[v, np.ones(len(v))])
        out.append(v)
    return np.concatenate(out)


def test_the_bake_puts_the_mannequin_and_the_camera_where_blender_has_them_moving_and_in_a_freeze(make, capsys):
    """The scene viewer and mk's renderer draw a frame from build/<name>.bake alone: the skinned mannequin rebuilt from it
    lands on Blender's own evaluation of that frame (in the freeze, of the frame it holds), and the shot's camera is where
    its keys put it on that frame, in the freeze too (it goes on moving over the held world)."""
    import numpy as np
    from scipy.spatial import cKDTree
    code, out = make(BAKED, cast="at = [0.5, 0.3, 0.0]\nyaw = 30")       # off the origin, turned: her vertices move with her
    assert code == 0, out.get("error")
    d = make.root / "build" / "t.bake"
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    assert (bake["frames"], bake["freezes"]) == (60, [[76, 85]])
    assert {o: [c["shot"] for c in cut] for o, cut in bake["cut"].items()} == {"16x9": ["a"], "9x16": ["a"]}
    cam = next(c for c in bake["cameras"] if c["name"] == bake["cut"]["16x9"][0]["camera"])
    blob = np.fromfile(d / "bake.bin", "<f4")
    first = blob[cam["offset"]:cam["offset"] + 15]
    assert np.allclose(first[9:12], [0.0, 0.9, 2.6], atol=1e-4) and first[12] == 35.0      # at = [0, -2.6, 0.9], +Y up
    for frame in (50, 80):                                                       # moving; in the freeze, still moving
        code, q = cli(["q", str(make.root / "build" / "t.blend"), f"obj({cam['name']!r}).loc", "--frames", str(frame),
                       "--project", str(make.root)], capsys)
        x, y, z = q["values"][0]
        o = cam["offset"] + 15 * (frame - bake["frame0"])
        assert np.allclose(blob[o + 9:o + 12], [x, z, -y], atol=1e-4) and x > 0.1, (frame, blob[o + 9:o + 12], (x, y, z))
    name = next(n["name"] for n in bake["nodes"] if n["armature"])
    expr = ("(lambda o, dg: np.array([(o.matrix_world @ v.co)[:] for v in o.evaluated_get(dg).data.vertices]).ravel()"
            f".tolist())(bpy.data.objects[{name!r}], bpy.context.evaluated_depsgraph_get())")
    assert np.abs(posed(d, name, 34) - posed(d, name, 40)).max() > 0.05          # the hands on their way up, and up
    for frame, world in ((34, 34), (80, 76)):                                    # moving; in the freeze, held on 76
        code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", str(world), "--project",
                       str(make.root)], capsys)
        assert code == 0, q.get("error")
        truth = np.array(q["values"][0]).reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1]
        dist, _ = cKDTree(truth).query(posed(d, name, frame))
        assert dist.max() < 5e-5, (frame, dist.max())


POPPED = """[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }
[pose.mq]
feet = "floor"
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
style = "vector"
[[ribbon]]
name = "loop"
style = "neon"
swoop = { points = [[0.7, -0.7, 0.8], [0.0, -0.9, 1.2], [-0.7, -0.7, 0.8]] }
draw = [0.0, 0.6]
[[key]]
target = "loop_ribbon"
prop = "scale"
index = 0
keys = [[0.0, 1.0], [0.9, 1.0], [1.0, 0.0]]
[[key]]
target = "loop_ribbon"
prop = "scale"
index = 1
keys = [[0.0, 1.0], [0.9, 1.0], [1.0, 0.0]]
[[key]]
target = "loop_ribbon"
prop = "scale"
index = 2
keys = [[0.0, 1.0], [0.9, 1.0], [1.0, 0.0]]
"""


def test_the_bake_holds_a_mesh_scaled_to_nothing_as_no_geometry(make):
    """A gag prop shown only round its move is popped out by keying its scale to 0, so its world matrix is singular on
    those frames: the bake holds it there as a geometry of no vertices (it shows nothing), where it used to fail
    inverting the matrix for the normals."""
    code, out = make(POPPED)
    assert code == 0, out.get("error")
    bake = json.loads((make.root / "build" / "t.bake" / "bake.json").read_text(encoding="utf-8"))
    entry = next(e for e in bake["by_frame"] if e["name"] == "loop_ribbon")
    sizes = [entry["geoms"][g][1] if g >= 0 else None for g in entry["frames"]]
    assert sizes[20] and sizes[-1] == 0, sizes


LIT = """[pose.mq]
feet = "floor"
[[move.mq]]
name = "hands_up"
t = 0.2
dur = 1.2
[perform.mq]
expressions = [{ morph = "a", keys = [[0.0, 0.0], [1.0, 1.0]] }, { morph = "赤", keys = [[0.0, 0.0], [1.0, 0.8]] }]
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
keys = [{ t = 0.0 }, { t = 2.0, at = [0.6, -2.6, 0.9] }]
[[freeze]]
from = 1.5
to = 1.8
[[light]]
name = "key"
kind = "area"
power = 200
at = [1.0, -2.0, 2.5]
look = [0.0, 0.0, 1.0]
size = [1.0, 0.5]
keys = [[0.0, 0.25], [1.0, 1.0]]
[look]
exposure = 0.7
contrast = "Medium High Contrast"
"""


@pytest.fixture(scope="module")
def lit(tmp_path_factory, mannequin):
    """One build of LIT (the mannequin, given a material morph `赤` that reddens its skin, off the origin and turned,
    raising its hands with its mouth opening and its skin reddening, a keyed area light, a freeze, a contrast look):
    (project folder, its bake folder)."""
    import contextlib
    import io
    root = tmp_path_factory.mktemp("lit")
    m = X.read(str(mannequin[0]))
    m.morphs.append(X.PmxMorph("赤", "red", 4, "material", [X.MaterialMorphOffset.of(0, 1, diffuse=(0.5, -0.3, -0.3, 0.0))]))
    X.write(m, str(root / "mq_lit.pmx"))
    (root / "mk.toml").write_text(HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{root / "mq_lit.pmx"}"\nrig = "{mannequin[1]}"\n'
                                  'at = [0.5, 0.3, 0.0]\nyaw = 30\n\n' + LIT, encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(root)])
    assert e.value.code == 0, json.loads(buf.getvalue()).get("error")
    return root, root / "build" / "t.bake"


def lit_posed(bake_dir, name, frame):
    """The lit mesh `name` on Blender frame `frame` from lit.bin and bake.bin alone, as bake_lit's docstring says a
    renderer places it (shape keys, world matrix, skin), in the world in glTF's axes."""
    bake = json.loads((bake_dir / "bake.json").read_text(encoding="utf-8"))
    lit = np.fromfile(bake_dir / "lit.bin", np.uint8)
    blob = np.fromfile(bake_dir / "bake.bin", "<f4")
    m = next(x for x in bake["lit"]["meshes"] if x["name"] == name)
    k, nv, a = frame - bake["frame0"], m["vertices"], m["attributes"]

    def attr(key):
        spec = a[key]
        dt = {"f32": "<f4", "u16": "<u2", "u32": "<u4"}[spec["type"]]
        return np.frombuffer(lit, dt, nv * spec["comp"], spec["offset"]).reshape(nv, spec["comp"]).astype(float)
    v = attr("position")
    if m["morphs"]:
        nk = len(m["morphs"]["names"])
        deltas = np.frombuffer(lit, "<f4", nk * nv * 3, m["morphs"]["deltas"]).reshape(nk, nv, 3)
        v = v + np.tensordot(blob[m["morphs"]["values"] + nk * k:][:nk], deltas, 1)
    o = m["matrix"]["offset"] + 12 * (k if m["matrix"]["frames"] > 1 else 0)
    M = blob[o:o + 12].reshape(4, 3).T
    v = v @ M[:, :3].T + M[:, 3]
    if m["kind"] == "skinned":
        arm = next(x for x in bake["armatures"] if x["name"] == m["armature"])
        nb = len(arm["bones"])
        J = blob[arm["offset"] + nb * 12 * k:][:nb * 12].reshape(nb, 4, 3).transpose(0, 2, 1)
        joints = np.concatenate([attr(j) for j in ("joints", "joints2") if j in a], 1).astype(int)
        w = np.concatenate([attr(j) for j in ("weights", "weights2") if j in a], 1)
        moved = np.einsum("vk,vkij,vj->vi", w, J[joints], np.c_[v, np.ones(nv)])
        v = np.where(w.sum(1, keepdims=True) > 0.0, moved, v)              # weights all 0: not deformed
    return v


def lit_normals(bake_dir, name, P):
    """The normals of the lit mesh `name` whose vertices are at P, rebuilt from its `bend` arrays as a renderer does on
    every frame (mkmmd.core.bend.normals)."""
    from mkmmd.core import bend as BD
    bake = json.loads((bake_dir / "bake.json").read_text(encoding="utf-8"))
    lit = np.fromfile(bake_dir / "lit.bin", np.uint8)
    m = next(x for x in bake["lit"]["meshes"] if x["name"] == name)
    b, nv, a = m["bend"], m["vertices"], m["attributes"]

    def u32(at, n):
        return np.frombuffer(lit, "<u4", n, at).astype(np.int64)
    ab = np.frombuffer(lit, "<f4", nv * 2, a["bend"]["offset"]).reshape(nv, 2).astype(float)
    arrays = {"fan": u32(a["fan"]["offset"], nv), "alpha": ab[:, 0], "beta": ab[:, 1],
              **{k: u32(b[k], b["fans"]) for k in ("center", "ref", "other")},
              **{k: (u32(b[k]["offsets"], b[k]["count"] + 1), u32(b[k]["items"], b[k]["size"]))
                 for k in ("corners", "edges", "faces")}}
    tris = u32(m["triangles"]["offset"], 3 * m["triangles"]["count"]).reshape(-1, 3)
    return BD.normals(P, tris, arrays)


def test_the_lit_mesh_rebuilt_from_the_bake_alone_lands_on_blenders_posed_mesh(lit, capsys):
    """mkr draws the lit frames from the bake's `lit` section alone: the skinned mannequin rebuilt from lit.bin and
    bake.bin (rest vertices split by Blender's corner data, the mouth's shape key, its world matrix, up to 8 bones per
    vertex) lands on Blender's own evaluation of the frame, moving, with the mouth half and fully open, and in the freeze
    (the frame it holds), and its normals, rebuilt from its fans on the posed vertices, are Blender's corner normals
    (where turning the rest normals with the bones misses them by tens of degrees); the bake's own check says so too."""
    from scipy.spatial import cKDTree
    root, d = lit
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    assert bake["version"] == 5
    mesh = next(m for m in bake["lit"]["meshes"] if m["kind"] == "skinned")
    check = bake["lit"]["check"]["meshes"][mesh["name"]]
    assert "あ" in mesh["morphs"]["names"] and check["mm"] < 0.1 and check["normal_deg"][0] < 1.0, check
    expr = ("(lambda o, dg: (lambda me, M: [np.array([(M @ v.co)[:] for v in me.vertices]).ravel().tolist(), "
            "[c.vertex_index for c in me.loops], np.array([(M.to_3x3().inverted_safe().transposed() @ n.vector)"
            ".normalized()[:] for n in me.corner_normals]).ravel().tolist()])(o.evaluated_get(dg).data, o.matrix_world))"
            f"(bpy.data.objects[{mesh['name']!r}], bpy.context.evaluated_depsgraph_get())")
    assert np.abs(lit_posed(d, mesh["name"], 34) - lit_posed(d, mesh["name"], 46)).max() > 0.05    # the hands go up
    for frame, world in ((34, 34), (46, 46), (61, 61), (80, 76)):                # moving; in the freeze, held on 76
        code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", str(world), "--project", str(root)], capsys)
        assert code == 0, q.get("error")
        pos, corner_v, nor = q["values"][0]
        truth = np.array(pos).reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1]
        ours = lit_posed(d, mesh["name"], frame)
        assert cKDTree(truth).query(ours)[0].max() < 5e-5, frame
        assert cKDTree(ours).query(truth)[0].max() < 5e-5, frame               # and none of Blender's is missing
        normals = lit_normals(d, mesh["name"], ours)
        dist, near = cKDTree(ours).query(truth[corner_v], k=8, distance_upper_bound=5e-5)
        cos = np.where(np.isfinite(dist), np.einsum("ckj,cj->ck", np.r_[normals, [[0.0, 0.0, 0.0]]][near],
                                                    np.array(nor).reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1]), -1.0)
        worst = np.degrees(np.arccos(np.clip(cos.max(1), -1.0, 1.0))).max()     # each corner: its vertex at that point
        assert worst < 1.0, (frame, worst)


def test_a_keyed_light_and_a_material_morphs_node_input_carry_their_values_on_every_frame(lit, capsys):
    """The area light's keyed power and the node input mmd_tools drives with the skin's material morph come out of the
    bake as Blender has them on each frame (in the freeze, on the frame it holds), the light where it stands and
    shining the way it points (its local -Z), its size as given."""
    root, d = lit
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    blob = np.fromfile(d / "bake.bin", "<f4")
    light = next(x for x in bake["lit"]["lights"] if x["name"] == "key")
    anim = next(a for a in bake["lit"]["materials"]["skin"]["anim"] if a["node"].startswith("mmd_bind"))
    assert (light["type"], light["shape"], light["frames"], anim["frames"]) == ("AREA", "RECTANGLE", 60, 60)
    expr = ("(lambda L, nt: [list(L.matrix_world.translation), list(L.matrix_world.col[2])[:3], L.data.energy, "
            f"next(s for s in nt.nodes[{anim['node']!r}].inputs if s.identifier == {anim['socket']!r}).default_value])"
            "(bpy.data.objects['key'], bpy.data.materials['skin'].node_tree)")
    seen = []
    for frame, world in ((31, 31), (40, 40), (61, 61), (80, 76)):
        code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", str(world), "--project", str(root)], capsys)
        assert code == 0, q.get("error")
        at, z, energy, value = q["values"][0]
        row = blob[light["offset"] + 22 * (frame - bake["frame0"]):][:22]
        assert np.allclose(row[0:3], [at[0], at[2], -at[1]], atol=1e-5) and np.allclose(row[3:6], [-z[0], -z[2], z[1]], atol=1e-5)
        assert row[15] == pytest.approx(energy, rel=1e-6) and np.allclose(row[16:18], [1.0, 0.5], atol=1e-6) and row[21] == 1.0
        assert blob[anim["offset"] + frame - bake["frame0"]] == pytest.approx(value, abs=1e-6)
        seen.append((energy, value))
    assert seen[0] == (pytest.approx(50.0), 0.0) and seen[2] == (pytest.approx(200.0), pytest.approx(0.8))


ON_OBJECTS = """[[prop]]
name = "page"
card = "library:cafe_page"
at = [0.0, -0.4, 0.75]
[[prop]]
name = "rug"
card = "library:rug_80s"
at = [0.0, 0.6, 0.0]
[[text]]
name = "ink"
ink = { strokes = "ink.json", on = "page:page" }
[[text]]
name = "sign"
text = "ON THE MOVE"
at = [0.0, -0.6, 1.0]
facing = [0.0, -1.0, 0.0]
size = 0.05
[[key]]
target = "sign"
prop = "location"
index = 0
keys = [[0.0, -0.3], [2.0, 0.3]]
interp = "LINEAR"
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -1.6, 1.1]
look = [0.0, -0.4, 0.85]
"""


@pytest.fixture(scope="module")
def on_objects(tmp_path_factory):
    """One build of ON_OBJECTS (handwriting ink on a page, whose material reads the ink object's keyed clock; a line of
    type, frame by frame for its geometry-nodes modifier, sliding across the shot; a rug whose three material slots read
    different things): (project folder, its bake folder)."""
    import contextlib
    import io
    root = tmp_path_factory.mktemp("on_objects")
    line = {"t": [0.2 + 0.01 * i for i in range(120)], "p": [[20.0 + 0.15 * i, 30.0] for i in range(120)]}
    (root / "ink.json").write_text(json.dumps({"unit": "mm", "strokes": [line]}), encoding="utf-8")
    (root / "mk.toml").write_text(HEAD + "\n" + ON_OBJECTS, encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(root)])
    assert e.value.code == 0, json.loads(buf.getvalue()).get("error")
    return root, root / "build" / "t.bake"


def test_a_mesh_carries_what_any_of_its_materials_reads(on_objects):
    """The rug's top is printed from its UV map in the first of its three slots, the other two read none: its lit mesh
    carries that map (every material's own node tree is named "Shader Nodetree": told apart by name, only one of a
    mesh's materials used to count, and the print came out flat in mkr)."""
    root, d = on_objects
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    body = next(m for m in bake["lit"]["meshes"] if m["name"] == "rug_body")
    assert len(set(body["materials"])) > 1 and "uv:UVMap" in body["attributes"], (body["materials"], sorted(body["attributes"]))


def test_what_a_material_reads_off_its_object_is_in_the_bake_on_every_frame(on_objects, capsys):
    """Handwriting ink shows behind the nib by its object's keyed clock (the custom property clip_t), which its material
    reads with an Attribute node of type Object: the ink's lit mesh carries that value on every frame (object_attrs,
    RGBA as the node reads a number: (v, v, v, 1)), as Blender has it, so mkr can draw the ink."""
    root, d = on_objects
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    blob = np.fromfile(d / "bake.bin", "<f4")
    ink = next(m for m in bake["lit"]["meshes"] if m["name"] == "ink")
    rec = ink["object_attrs"]["OBJECT:clip_t"]
    assert rec["frames"] == bake["frames"], rec
    clocks = []
    for frame in (31, 40, 55, 90):
        code, q = cli(["q", str(root / "build" / "t.blend"), 'bpy.data.objects["ink"]["clip_t"]', "--frames",
                       str(frame), "--project", str(root)], capsys)
        assert code == 0, q.get("error")
        v = blob[rec["offset"] + 4 * (frame - bake["frame0"]):][:4]
        assert np.allclose(v, [q["values"][0]] * 3 + [1.0], atol=1e-6), (frame, v, q["values"][0])
        clocks.append(float(v[0]))
    assert clocks == sorted(clocks) and clocks[-1] > clocks[0], clocks         # the clock runs


def test_an_object_baked_frame_by_frame_that_only_moves_holds_one_geometry_placed_by_its_matrix(on_objects, capsys):
    """The sliding type is evaluated on every frame (its geometry-nodes modifier), but only its matrix changes: the bake
    holds its geometry once, in its own space, and its world matrix per frame, which puts it where Blender draws it
    (what kept a car's 38 000-vertex label under the bake's cap); a world-space geometry per frame would be 60."""
    from scipy.spatial import cKDTree
    root, d = on_objects
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    lit = np.fromfile(d / "lit.bin", np.uint8)
    blob = np.fromfile(d / "bake.bin", "<f4")
    sign = next(m for m in bake["lit"]["meshes"] if m["name"] == "sign")
    bf, mat = sign["by_frame"], sign["matrix"]
    assert sign["kind"] == "by_frame" and len(bf["geoms"]) == 1 and mat["frames"] == bake["frames"], (len(bf["geoms"]), mat)
    g = bf["geoms"][0]
    pos = np.frombuffer(lit, "<f4", 3 * g["vertices"], g["attributes"]["position"]["offset"]).reshape(-1, 3).astype(float)
    expr = ("(lambda ev: [(ev.matrix_world @ v.co)[:] for v in ev.to_mesh().vertices])"
            "(bpy.data.objects['sign'].evaluated_get(bpy.context.evaluated_depsgraph_get()))")
    xs = []
    for frame in (31, 60, 90):
        code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", str(frame), "--project", str(root)], capsys)
        assert code == 0, q.get("error")
        truth = np.array(q["values"][0]).reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1]
        o = mat["offset"] + 12 * (frame - bake["frame0"])
        M = blob[o:o + 12].reshape(4, 3).T
        ours = pos @ M[:, :3].T + M[:, 3]
        assert cKDTree(truth).query(ours)[0].max() < 1e-5 and cKDTree(ours).query(truth)[0].max() < 1e-5, frame
        xs.append(float(ours[:, 0].mean()))
    assert xs[0] < xs[1] < xs[2] and xs[2] - xs[0] > 0.5, xs                    # it slides 0.6 m


def test_the_view_lut_the_bake_captures_is_blenders_display_transform(lit, capsys):
    """bake.json's view LUT (view_lut.bin: the scene's view transform, its contrast look and exposure, sampled on a log2
    grid) gives what Blender writes to an 8-bit file for scene-linear colours off the grid, to a level."""
    from PIL import Image
    root, d = lit
    view = json.loads((d / "bake.json").read_text(encoding="utf-8"))["lit"]["view"]
    assert (view["transform"], view["look"], view["exposure"]) == ("AgX", "AgX - Medium High Contrast", pytest.approx(0.7))
    n, lo, hi = view["lut"]["n"], view["lut"]["lo"], view["lut"]["hi"]
    lut = np.fromfile(d / view["lut"]["file"], "<f4").reshape(n, n, n, 3)          # [b][g][r]
    rgb = np.array([[0.18, 0.18, 0.18], [0.9, 0.2, 0.05], [2.0, 1.0, 0.5], [0.01, 0.03, 0.002], [12.0, 4.0, 30.0],
                    [0.05, 0.6, 0.3], [0.0, 0.0, 0.0]])
    png = root / "view_probe.png"
    flat = np.c_[rgb, np.ones(len(rgb))].ravel().tolist()
    expr = ("(lambda sc, img: (img.pixels.foreach_set(" + repr(flat) + "), [setattr(sc.render.image_settings, k, v) "
            "for k, v in (('file_format', 'PNG'), ('color_mode', 'RGB'), ('color_depth', '8'))], "
            "setattr(sc.render, 'dither_intensity', 0.0), img.save_render(" + repr(str(png)) + ", scene=sc), 1)[-1])"
            f"(scene, bpy.data.images.new('probe', {len(rgb)}, 1, float_buffer=True))")
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", "31", "--project", str(root)], capsys)
    assert code == 0, q.get("error")
    blender = np.asarray(Image.open(png).convert("RGB"), float)[0]
    x = np.clip((np.log2(np.maximum(rgb, 2.0 ** lo)) - lo) / (hi - lo) * (n - 1), 0, n - 1)
    i = np.minimum(x.astype(int), n - 2)
    f = x - i
    ours = np.zeros_like(rgb)
    for c in np.ndindex(2, 2, 2):                                     # trilinear in log2: what a renderer does with it
        w = np.prod([f[:, a] if c[a] else 1 - f[:, a] for a in range(3)], 0)
        ours += w[:, None] * lut[i[:, 2] + c[2], i[:, 1] + c[1], i[:, 0] + c[0]]
    ours[-1] = view["zero"]
    assert np.abs(ours * 255 - blender).max() <= 1.0, np.c_[ours * 255, blender]


LAYERS = """[pose.mq]
feet = "floor"
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
[[shot]]
name = "p"
plate = true
at = [1.5, -2.0, 1.2]
look = [0.0, 0.0, 1.0]
lens = 35
[[insert]]
from = 0.5
to = 1.5
shot = "p"
shape = "iris"
anchor = "bone('head').head"
[[text]]
name = "title"
screen = { anchor = "bottom", side = "left", height = 0.2 }
text = "HI"
size = 0.07
kinetic = { show = [0.2, 1.6] }
[[prop]]
name = "pc"
card = "library:pillow_cloud"
at = [1.0, 0.5, 0.4]
"""


@pytest.fixture(scope="module")
def layers(tmp_path_factory, mannequin):
    """One build of LAYERS: the mannequin, a screen-type title shown 0.2 .. 1.6 s, an iris insert whose anchor is her
    head (a point layer), a pillow whose subdivision renders a level finer than the viewport shows: (folder, bake)."""
    import contextlib
    import io
    root = tmp_path_factory.mktemp("layers")
    (root / "mk.toml").write_text(HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{mannequin[0]}"\nrig = "{mannequin[1]}"\n\n'
                                  + LAYERS, encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        MAIN.main(["build", "--project", str(root)])
    assert e.value.code == 0, json.loads(buf.getvalue()).get("error")
    return root, root / "build" / "t.bake"


def test_screen_type_is_baked_as_blender_evaluates_it_with_its_layers_camera_and_view_transform(layers, capsys):
    """mkr draws the screen layer from the bake: each output's copy of the title as Blender evaluates it on the frames
    it shows (none outside them), a frame of it landing on Blender's own evaluation, the orthographic camera one frame
    height tall per output, and the layer's Standard view transform as a LUT (sRGB's curve)."""
    from mkmmd.core import screentype as SR
    root, d = layers
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    scr = bake["lit"]["screen"]
    lit = np.fromfile(d / "lit.bin", np.uint8)
    title = {m["aspect"]: m for m in scr["meshes"]}
    assert set(title) == {"16x9", "9x16"} and all(m["show"] == [37, 79] for m in title.values())
    frames = title["16x9"]["by_frame"]["frames"]
    assert all((f >= 0) == (37 <= k + 31 < 79) for k, f in enumerate(frames)), frames
    assert all(v["frames"] and not v["problems"] and v["px"] < 0.05 and v["normal_deg"] < 1.1
               for v in bake["lit"]["check"]["screen"].values())                  # decoded words land on Blender's
    for out, size in (("16x9", (640, 360)), ("9x16", (360, 640))):
        cam = scr["cameras"][out]
        assert cam["type"] == "ORTHO" and cam["ortho_scale"] == pytest.approx(SR.ortho_scale(size))
        assert np.allclose(cam["matrix"][9:], [0.0, 10.0, 0.0])                         # z = 10 up, in glTF's axes
    from mkmmd.core import fitgeo as FG
    ours = FG.decode(lit, title["16x9"]["by_frame"], 50 - 31)["position"]           # as mkr decodes the fitted word
    expr = ("(lambda o: (setattr(o, 'hide_viewport', False), scene.frame_set(scene.frame_current), (lambda ev: "
            "[(ev.matrix_world @ v.co)[:] for v in ev.to_mesh().vertices])(o.evaluated_get("
            "bpy.context.evaluated_depsgraph_get())))[-1])(bpy.data.objects['title@16x9'])")
    code, q = cli(["q", str(root / "build" / "t.blend"), expr, "--frames", "50", "--project", str(root)], capsys)
    assert code == 0, q.get("error")
    truth = np.array(q["values"][0]).reshape(-1, 3)[:, [0, 2, 1]] * [1, 1, -1]
    assert len(truth) > 10 and np.linalg.norm(truth[:, None] - ours[None], axis=2).min(1).max() < 2e-5     # fit: 1e-5
    n, lo, hi = scr["lut"]["n"], scr["lut"]["lo"], scr["lut"]["hi"]
    lut = np.fromfile(d / scr["lut"]["file"], "<f4").reshape(n, n, n, 3)
    v = 2.0 ** (lo + (hi - lo) * np.arange(n) / (n - 1))
    srgb = np.clip(np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055), 0.0, 1.0)
    assert np.abs(lut[np.arange(n), np.arange(n), np.arange(n), 1] - srgb).max() < 1e-3


def test_a_layer_point_and_render_level_geometry_come_from_the_bake_and_every_lit_mesh_has_a_look_record(layers, capsys):
    """The iris's anchor (her head, through the cut's camera) is in the bake on the insert's frames and nowhere else, where
    Blender's expression puts it; the pillow's subdivided mesh is baked at its render level (the viewport shows fewer
    faces); every lit mesh names the look record its patterns match (screen type included)."""
    root, d = layers
    bake = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    blob = np.fromfile(d / "bake.bin", "<f4")
    pt = bake["lit"]["points"]["i0"]
    assert pt["frames"] == [[15, 45, "a"]] and not pt["errors"]
    xyz = blob[pt["offset"]:pt["offset"] + 3 * bake["frames"]].reshape(-1, 3)
    assert np.isnan(xyz[:15]).all() and np.isnan(xyz[45:]).all()
    for frame in (46, 70):
        code, q = cli(["q", str(root / "build" / "t.blend"), "list(bone('head').head)", "--frames", str(frame),
                       "--project", str(root)], capsys)
        assert code == 0, q.get("error")
        x, y, z = q["values"][0]
        assert np.allclose(xyz[frame - 31], [x, z, -y], atol=1e-5), frame
    assert "pc_pillow" in bake["lit"]["render_settings"]
    mesh = next(m for m in bake["lit"]["meshes"] if m["name"] == "pc_pillow")
    ours = mesh["by_frame"]["geoms"][mesh["by_frame"]["frames"][0]]["triangles"]["count"]
    tris = ("(lambda o, render: ([setattr(m, 'levels', m.render_levels) for m in o.modifiers if render and "
            "m.type == 'SUBSURF'], bpy.context.view_layer.update(), (lambda me: (me.calc_loop_triangles(), "
            "len(me.loop_triangles))[-1])(o.evaluated_get(bpy.context.evaluated_depsgraph_get()).data))[-1])"
            "(bpy.data.objects['pc_pillow'], {})")
    counts = []
    for render in (True, False):
        code, q = cli(["q", str(root / "build" / "t.blend"), tris.format(render), "--frames", "31", "--project",
                       str(root)], capsys)
        assert code == 0, q.get("error")
        counts.append(q["values"][0])
    assert ours == counts[0] > counts[1], (ours, counts)
    looks = {o["name"] for o in bake["objects"]} | set(bake["lit"]["records"])
    names = [m["record"] for m in bake["lit"]["meshes"] + bake["lit"]["screen"]["meshes"]]
    assert names and set(names) <= looks and "title@16x9" in bake["lit"]["records"]


FROZEN_CUT = """[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }
[vector.tones]
dark = "#1B2A6B"
[[vector.materials]]
match = "*"
tone = "dark"
[pose.mq]
feet = "floor"
[[shot]]
name = "a"
from = 0.0
to = 1.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
style = "vector"
[[shot]]
name = "b"
from = 1.0
to = 2.0
at = [2.0, -1.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35
style = "vector"
[[freeze]]
from = 0.8
to = 1.4
"""


def test_a_cut_inside_a_freeze_shows_the_next_shot_through_its_own_camera(make, capsys):
    """In a freeze the world holds on its first frame but the cut goes on: after a cut inside it the frame is the next
    shot's, through its camera, as after the freeze (Blender's render bound the camera of the world's frame instead,
    the shot before the cut)."""
    import numpy as np
    from PIL import Image
    code, out = make(FROZEN_CUT)
    assert code == 0, out.get("error")
    code, rep = cli(["render", "--project", str(make.root), "--output", "16x9", "--frames", "49,67,82"], capsys)
    assert code == 0, rep.get("error")
    d = make.root / "renders" / "draft" / "16x9"
    a, frozen, b = (np.asarray(Image.open(d / f"{f:05d}.png").convert("RGB"), int) for f in (49, 67, 82))
    off = lambda x, y: float((np.abs(x - y).max(-1) > 8).mean())   # noqa: E731
    assert off(frozen, b) < 0.01 and off(frozen, a) > 0.02, (off(frozen, b), off(frozen, a))


HANDS = """[pose.mq]
feet = "floor"
[[move.mq]]
name = "hands_up"
t = 0.1
dur = 0.6
[[move.mq]]
name = "lean_back"
t = 0.1
dur = 0.7
[[move.mq]]
name = "point"
hand = "R"
t = 0.75
dur = 0.3
[[move.mq]]
name = "mic_up"
hand = "L"
t = 1.3
dur = 0.5
"""


def test_the_hand_checks_measure_the_moves_holds_in_the_frame_of_the_body_they_ride(make, capsys):
    """hands_up under lean_back's sideways tilt is mirrored only in the upper body's frame (the hands ride it); the right
    hand leaves it early for the point, a move the mirror check must not count; mic_up keeps the palm in."""
    code, out = make(HANDS)
    assert code == 0, out.get("error")
    checks = {m: {"cast": "mq"} for m in ("hand_mirror", "wrist_bend", "mic_palm", "hand_room")}
    got = {}
    for m, a in checks.items():
        code, res = cli(["check", m, "--args", json.dumps(a), "--project", str(make.root)], capsys)
        got[m] = res["results"][0]
    assert all(r["ok"] for r in got.values()), {m: (r.get("value"), r.get("error")) for m, r in got.items()}
    mirror = got["hand_mirror"]
    assert mirror["value"] < 2.0 and mirror["detail"]["move"] == "hands_up"
    assert got["mic_palm"]["detail"]["side"] == "L" and got["mic_palm"]["value"] > 0.5


def _turn(bone, deg):
    """A VMD rotation (x y z w, MMD's axes) of `deg` degrees about a PMX twist bone's fixed axis."""
    ax = np.asarray(bone.fixed_axis, float)
    s = np.sin(np.radians(deg) / 2.0) * ax / np.linalg.norm(ax)
    return (float(s[0]), float(s[1]), float(s[2]), float(np.cos(np.radians(deg) / 2.0)))


def test_the_twist_checks_flag_an_arm_turned_past_a_human_one_and_its_skin_pulled_in(make, mannequin, capsys):
    """A motion turns the left forearm's twist bone 150 deg about its axis, then 150 the other way, the right one 40,
    and the right upper arm's 130: joint_limits flags both of the left forearm's turns (a palm turned up where the thumb
    rises: supination, + on either side), the right upper arm's and not the right forearm's; skin_collapse finds the
    left forearm and the right upper arm pulled in where the twist bones' weights meet the elbow's and the arm's, and
    the right forearm's skin within a few per cent."""
    bones = {b.name: b for b in X.read(str(mannequin[0])).bones}
    vmd = make.root / "twist.vmd"
    _vmd(vmd, {"左手捩": [(0, _turn(bones["左手捩"], 150)), (25, _turn(bones["左手捩"], 150)),
                          (35, _turn(bones["左手捩"], -150)), (60, _turn(bones["左手捩"], -150))],
               "右手捩": [(0, _turn(bones["右手捩"], 40)), (60, _turn(bones["右手捩"], 40))],
               "右腕捩": [(0, _turn(bones["右腕捩"], 130)), (60, _turn(bones["右腕捩"], 130))]})
    code, out = make(f'[[motion.mq]]\nvmd = "{vmd}"\nstart = 0.0\n', until="motion")
    assert code == 0, out.get("error")
    root = str(make.root)
    code, q = cli(["q", str(make.root / "build" / "t.blend"), "bone('thumb2.L').tail.z - bone('wrist.L').head.z",
                   "--frames", "46,76", "--project", root], capsys)
    thumb_up = dict(zip((46, 76), q["values"]))
    turned = {}
    for f in (46, 76):
        code, res = cli(["check", "joint_limits", "--args", '{"cast": "mq"}', "--max", "0", "--frames", str(f),
                         "--project", root], capsys)
        turned[f] = res["results"][0]["detail"]
    for f in (46, 76):
        fa = turned[f]["forearm_twist.L"]
        assert abs(fa["max"]) == pytest.approx(150.0, abs=3.0) and fa["over_deg"] > 45.0
        assert (fa["max"] > 0) == (thumb_up[f] > thumb_up[76 if f == 46 else 46])   # the thumb rises: supination, +
        assert "over_deg" not in turned[f]["forearm_twist.R"]
        assert abs(turned[f]["forearm_twist.R"]["max"]) == pytest.approx(40.0, abs=3.0)
        assert abs(turned[f]["arm_twist.R"]["max"]) == pytest.approx(130.0, abs=3.0)
        assert turned[f]["arm_twist.R"]["over_deg"] > 30.0 and "over_deg" not in turned[f]["arm_twist.L"]
    assert turned[46]["forearm_twist.L"]["max"] * turned[76]["forearm_twist.L"]["max"] < 0     # one each way
    code, res = cli(["check", "skin_collapse", "--args", '{"cast": "mq"}', "--project", root], capsys)
    got = res["results"][0]
    assert code == 1 and not got["ok"] and got["value"] > 0.3, got
    regions = got["detail"]["regions"]
    assert regions["forearm.L"]["loss"] > 0.3 and regions["upper_arm.R"]["loss"] > 0.3
    assert regions["forearm.R"]["loss"] < 0.1 and regions["upper_arm.L"]["loss"] < 0.05
    assert got["detail"]["region"] in ("forearm.L", "upper_arm.R") and got["detail"]["kind"] == "collapse"


DRAWN = """[vector]
colors = { background = "#FFD21F", line = "#1B2A6B" }
shadow = 0.3

[vector.opposite]
colors = { background = "#1B2A6B", line = "#FFD21F", inner = "#1B2A6B" }

[vector.tones]
white = { lit = "#FFFFFF", shade = "#8FB4F0" }

[[vector.materials]]
match = "*"
tone = "white"

[pose.mq]
feet = "floor"

[[move.mq]]
name = "hands_up"
t = 0.2
dur = 1.2

[[prop]]
name = "mic"
card = "library:handheld_mic"
attach = "mq:wrist.R"
cable = true

[[prop]]
name = "bb"
card = "library:cassette_player"
at = [0.0, 0.45, 0.0]
scale = 5.0

[[prop]]
name = "fz"
card = "library:fuse"
at = [0.45, 0.1, 0.0]
slots = { lit = 1.0 }

[[key]]
target = "fz"
prop = "burn"
keys = [[0.0, 0.0], [1.2, 0.9]]

[[shot]]
name = "a"
from = 0.0
to = 1.3
at = [0.9, -2.2, 1.3]
look = [0.1, 0.0, 0.8]
lens = 35
style = "vector"

[[shot]]
name = "b"
from = 1.3
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35

[[ring]]
at = 0.3
center = [0.3, 0.6]
dur = 0.6
"""


def test_mk_s_own_renderer_draws_the_vector_look_as_blender_does_and_leaves_blender_the_rest(make, capsys):
    """engine = "mk" draws shot a from the bake (the skinned mannequin with the shadow she casts on the boombox behind her,
    the mic's swinging cable and the burning fuse's rope baked frame by frame, a ring flipping the palette) as Blender
    draws it, but for slivers where light turns to shadow; shot b, lit by the scene, is Blender's, and the report says so."""
    import shutil
    import numpy as np
    from PIL import Image
    from mkmmd.render import engine as EN
    if EN.browser() is None:
        pytest.skip("Chromium not available")
    code, out = make(DRAWN)
    assert code == 0, out.get("error")
    d = make.root / "renders" / "draft" / "16x9"

    def render(*extra):
        code, rep = cli(["render", "--project", str(make.root), "--output", "16x9", "--frames", "36,46,52,70", *extra],
                        capsys)
        assert code == 0, rep.get("error")
        return rep
    rep = render("--engine", "mk")                               # the ring is up from frame 40 to 58
    got = rep["outputs"]["16x9"]
    assert got["shots"] == {"a": {"engine": "mk", "frames": 3},
                            "b": {"engine": "blender", "frames": 1, "why": "lit by the scene (no flat look)"}}
    assert got["engines"] == {"mk": 3, "blender": 1} and rep["mk"]["frames"] == 3 and got["frames_on_disk"] == 4
    mine = {f: np.asarray(Image.open(d / f"{f:05d}.png").convert("RGB"), int) for f in (36, 46, 52)}
    shutil.rmtree(d)
    render()
    for f, img in mine.items():                                  # measured: 0.05 to 0.11 % off by more than 8 levels
        off = np.abs(img - np.asarray(Image.open(d / f"{f:05d}.png").convert("RGB"), int)).max(-1) > 8
        assert off.mean() < 0.003, (f, off.mean())


EXPANDED = """[pose.mq]
feet = "floor"

[[shot]]
name = "a"
from = 0.0
to = 1.3
at = [0.4, -2.4, 1.1]
look = [0.0, 0.0, 0.9]
lens = 35
style = "vector"

[[shot]]
name = "b"
from = 1.3
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35

[[transition]]
at = 1.3
kind = "expand"
dur = 0.3
"""


def test_mkr_draws_a_matte_through_a_vector_look_as_blender_does(make, capsys):
    """An expand out of a shot in the vector look needs that shot's figure as a matte and its frame without her (back/)
    on the transition's frames: mkr draws them (the look's passes and her coverage at twice the frame's size, as for a
    silhouette) and they match Blender's files; no Blender job is left."""
    import shutil
    from PIL import Image
    code, out = make(EXPANDED)
    assert code == 0, out.get("error")
    d = make.root / "renders" / "draft" / "16x9"

    def render(*extra):
        code, rep = cli(["render", "--project", str(make.root), "--output", "16x9", "--frames", "56:72", *extra], capsys)
        assert code == 0, rep.get("error")
        return rep
    rep = render("--engine", "mkr")
    got = rep["outputs"]["16x9"]
    assert got["blender_jobs"] == 0 and got["engines"]["blender"] == 0, got
    files = {p.relative_to(d): np.asarray(Image.open(p).convert("RGB"), int)
             for kind in ("matte", "back") for p in (d / kind).rglob("*.png")}
    assert {p.parts[0] for p in files} == {"matte", "back"}, sorted(files)
    shutil.rmtree(d)
    render()
    for rel, img in files.items():                               # Blender's own files of the same layers
        diff = np.abs(img - np.asarray(Image.open(d / rel).convert("RGB"), int)).max(-1)
        assert diff.mean() < 0.5 and (diff > 8).mean() < 0.005, (str(rel), diff.mean(), (diff > 8).mean())


# ---------------------------------------------------------------- cloth
@pytest.fixture(scope="module")
def skirted(tmp_path_factory):
    """The mannequin in its skirt, whose cloth copy (rig.json `cloth`, mq.cloth.npz) the cloth stage simulates."""
    d = tmp_path_factory.mktemp("skirted")
    (d / "m.toml").write_text('[model]\nname = "mq"\nparts = ["mannequin", "mannequin_skirt"]\n'
                              f'out = "{d / "out"}"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        MAIN.main(["model", "build", str(d / "m.toml"), "--no-cache"])
    assert e.value.code == 0
    return d / "out" / "mq.pmx", d / "out" / "mq.rig.json"


SPIN = """[pose.mq]
feet = "floor"
[perform.mq]
step_turn = [{{ t = 0.2, deg = {deg} }}]
[cloth.mq]
"""


def test_a_simulated_skirt_lags_a_spin_its_faces_on_the_cloth_and_a_rebuild_reuses_it(tmp_path, skirted, capsys):
    """The cloth stage simulates the skirt's cloth copy in the world: as she spins, its hem trails her hips (it turns
    less than the waist ring held on them), and the skirt's faces follow the cloth on every frame. A rebuild with the
    same motion reuses the simulation; another spin runs it again."""
    import numpy as np
    from scipy.spatial import cKDTree

    def build(deg):
        (tmp_path / "mk.toml").write_text(HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{skirted[0]}"\nrig = "{skirted[1]}"\n\n'
                                          + SPIN.format(deg=deg), encoding="utf-8")
        code, out = cli(["build", "--project", str(tmp_path), "--skip", "sim"], capsys)
        assert code == 0, out.get("error")
        return out["stages"]["cloth"]["mq"]["skirt"]
    rep = build(180)
    assert not rep["cached"] and rep["pen_mm"] < 5.0 and rep["stretch_pct"] < 60.0, rep
    (skirt, n), = rep["follows"].items()
    expr = ('[[list(v.co) for v in o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh().vertices] '
            f'for o in (bpy.data.objects["{skirt}"], bpy.data.objects["{skirt}_cloth"])]')
    code, q = cli(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "36:50:2", "--project", str(tmp_path)],
                  capsys)                                         # the turn starts at t 0.2 (frame 37)
    assert code == 0, q
    faces, cloth = (np.array([v[i] for v in q["values"]]) for i in (0, 1))        # her armature's space
    assert faces.shape[1] == n
    assert max(cKDTree(c).query(f)[0].max() for f, c in zip(faces, cloth)) < 5e-4  # the skirt's faces are on the cloth
    z = cloth[0, :, 2]
    waist, hem = np.abs(z - z.max()) < 1e-3, np.abs(z - z.min()) < 1e-3
    centre = cloth[:, waist].mean(1, keepdims=True)
    a = np.unwrap(np.arctan2(*(cloth - centre)[..., [1, 0]].transpose(2, 0, 1)), axis=0)
    turned = np.degrees(a - a[0])                                                    # since just before the turn
    trail = np.abs(turned[:, hem].mean(1) - turned[:, waist].mean(1))
    assert trail[0] == 0.0 and trail[1:].max() > 3.0, trail                       # measured: 5.3 degrees at frame 40
    again = build(180)
    assert again["cached"] and again["key"] == rep["key"] and again["pen_mm"] == rep["pen_mm"]
    assert not build(90)["cached"]



MADE = """[cloth.mq]
pieces = ["made"]
hold = 0.05
[[cloth.mq.make]]
name = "made"
materials = ["skirt"]
collide = ["skin", "shirt"]                 # the body under it: the shirt covers her hips
pin = {{ above = 0.87, fade = 0.03 }}
"""


def test_a_cloth_copy_made_from_the_skirt_s_own_faces_is_pinned_at_the_waist_and_lags_a_spin(tmp_path, skirted, capsys):
    """`[[cloth.<cast>.make]]` makes the cloth copy from the model's own faces (a model that brings none, mkmmd.core.garment):
    here the skirt's, its rig copy left out. The copy is pinned above `pin.above` (the waist ring), its rendered faces lie
    on the cloth on every frame, and as she spins the hem trails the waist."""
    from scipy.spatial import cKDTree
    (tmp_path / "mk.toml").write_text(HEAD + f'\n[[cast]]\nname = "mq"\npmx = "{skirted[0]}"\nrig = "{skirted[1]}"\n\n'
                                      + SPIN.format(deg=180).replace("[cloth.mq]\n", "") + MADE.format(), encoding="utf-8")
    code, out = cli(["build", "--project", str(tmp_path), "--skip", "sim"], capsys)
    assert code == 0, out.get("error")
    rep = out["stages"]["cloth"]["mq"]["made"]
    assert set(out["stages"]["cloth"]["mq"]) == {"made"} and rep["made"]["pinned"] > 0 and rep["pen_mm"] < 5.0, rep
    (skirt, n), = rep["follows"].items()
    expr = ('[[list(v.co) for v in o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh().vertices] '
            f'for o in (bpy.data.objects["{skirt}"], bpy.data.objects["{skirt}_cloth"])]')
    code, q = cli(["q", str(tmp_path / "build" / "t.blend"), expr, "--frames", "36:50:2", "--project", str(tmp_path)],
                  capsys)
    assert code == 0, q
    faces, cloth = (np.array([v[i] for v in q["values"]]) for i in (0, 1))
    assert faces.shape[1] == n
    assert max(cKDTree(c).query(f)[0].max() for f, c in zip(faces, cloth)) < 5e-4      # its faces are on the cloth
    z = cloth[0, :, 2]
    waist, hem = np.abs(z - z.max()) < 1e-3, np.abs(z - z.min()) < 1e-3
    centre = cloth[:, waist].mean(1, keepdims=True)
    a = np.unwrap(np.arctan2(*(cloth - centre)[..., [1, 0]].transpose(2, 0, 1)), axis=0)
    turned = np.degrees(a - a[0])
    trail = np.abs(turned[:, hem].mean(1) - turned[:, waist].mean(1))
    assert trail[1:].max() > 3.0, trail


WALK = """[pose.mq]
feet = "floor"
[perform.mq]
walk = {{ steps = [0.3, 0.7, 1.1], hops = [{{ from = 1.3, to = 1.8 }}], stride = 0.5 }}
{extra}"""


def test_a_walk_carries_the_root_plants_the_feet_and_a_tumble_turns_about_the_hips(make, capsys):
    """The root goes where the steps and the hop carry the body; a foot that is down does not move in the world; a 360
    degree flip in the air ends upright with the hips where the walk put them and keeps them there all through; a
    [[key]] that keys the root's location outright is refused (a relative one is a push on top)."""
    import numpy as np
    code, out = make(WALK.format(extra='tumble = [{ from = 1.35, to = 1.75, deg = 360 }]'), until="perform",
                     cast="yaw = 90")
    assert code == 0, out.get("error")
    rep = out["stages"]["perform"]["mq"]
    assert rep["walk"]["steps"] == 3 and rep["walk"]["hops"] == 1 and rep["tumble"] == {"flip": 360.0}
    expr = '[list(obj("Mq").loc), list(bone("leg_ik.L").head), list(bone("center").head), list(obj("Mq").euler)]'
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "31:91", "--project", str(make.root)],
                  capsys)
    assert code == 0, q
    root, foot, hips, euler = (np.array([v[i] for v in q["values"]]) for i in range(4))
    t = (np.arange(31, 92) - 31) / 30.0
    assert abs(root[0, 0]) < 0.05 and root[-1, 0] > 1.0                                # walked toward +X (yaw 90)
    down = (t > 0.31) & (t < 0.7)                                                      # L landed at 0.3, lifts after 0.7
    assert np.ptp(foot[down], axis=0).max() < 1e-3
    flip = (t >= 1.35) & (t <= 1.75)
    path = root[flip] - root[flip][0]                                                  # the body's own travel ...
    sway = (hips[flip] - hips[flip][0]) - path * np.array([1.0, 1.0, 0.0])            # ... aside, the hips stay put
    assert np.abs(sway[:, 1]).max() < 0.02 and abs(euler[-1, 0] - euler[0, 0]) % 360 == pytest.approx(0.0, abs=0.5)
    code, out = make(WALK.format(extra="") + '\n[[key]]\ntarget = "Mq"\nprop = "location"\nkeys = [[0.0, 0.0], '
                     '[1.0, 0.3]]\n', until="perform", cast="yaw = 90")
    assert code == 3 and "relative = true" in out["error"]


STRUT = """[pose.mq]
feet = "floor"
[perform.mq]
walk = { steps = [0.3, 0.8, 1.3], stride = 0.5, cross = 1.0, sway = 0.04 }
head_turn = [[1.6, 0.0], [1.8, 50.0]]
"""


def test_a_strut_sets_the_feet_on_one_line_sways_the_hips_over_the_planted_foot_and_a_head_turn_turns_the_head(make, capsys):
    """Facing +X (yaw 90), the character's left is +Y: each landing foot comes down on the line under the root, the
    hips are over the foot that is down halfway between two landings (the left one after the first step), and the head
    turns toward her left by the head_turn keys."""
    import numpy as np
    code, out = make(STRUT, until="perform", cast="yaw = 90")
    assert code == 0, out.get("error")
    assert out["stages"]["perform"]["mq"]["sway_mm"] == pytest.approx(40.0, abs=1.0)  # its peak, as the frames sample it
    expr = ('[list(obj("Mq").loc), list(bone("leg_ik.L").head), list(bone("leg_ik.R").head), list(bone("center").head), '
            'list(bone("head").matrix.to_euler())]')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "31:91", "--project", str(make.root)],
                  capsys)
    assert code == 0, q
    root, left, right, hips, head = (np.array([v[i] for v in q["values"]]) for i in range(5))
    t = (np.arange(31, 92) - 31) / 30.0
    on = lambda s: int(np.argmin(np.abs(t - s)))                                     # noqa: E731
    assert abs(left[on(0.35), 1] - root[on(0.35), 1]) < 0.01                          # L down on the line (y of the root)
    assert abs(right[on(0.85), 1] - root[on(0.85), 1]) < 0.01                         # then R
    mid = on(0.55)                                                                    # L carries her from 0.3 to 0.8
    assert hips[mid, 1] - root[mid, 1] == pytest.approx(0.04, abs=0.005)              # over it: toward her left
    assert hips[on(1.05), 1] - root[on(1.05), 1] == pytest.approx(-0.04, abs=0.005)   # then over R
    assert np.degrees(head[on(1.85), 2] - head[on(1.55), 2]) > 30                     # the head turned toward her left


TWIRL = """[pose.mq]
feet = "floor"
[perform.mq]
{extra}spin = {{ from = 0.25, to = 1.95, deg = 360 }}
"""
FACING = ('[math.degrees(math.atan2(d.x, -d.y)) for d in [bone(n).matrix.to_3x3() @ arm("Mq_arm").data.bones[bone(n).name]'
          '.matrix_local.to_3x3().inverted() @ Vector((0, -1, 0)) for n in ("head", "upper_body2", "lower_body")]] '
          '+ [bpy.data.objects["Mq"].rotation_euler[2]] + list(bone("toe_ik.L").head) + list(bone("ankle.R").head)')


def test_a_spin_turns_her_round_chest_first_hips_last_the_head_spotting_the_front_in_the_air_and_on_her_toes(make, capsys):
    """A twirl was a [[key]] turning the root on a turntable. A spin turns the root (so the hair and skirt flare from the
    real turn) once round from a wind-up the other way to an overshoot settling back; the chest leads the hips, and the
    head holds the front until the neck runs out of turn, then whips round ahead of the chest. Floating (`rise`, `air`)
    the legs go round with her; standing, she turns on her left toes (they stay put) with the right foot lifted. A
    [[key]] that also turns the root is refused."""
    import numpy as np
    t = (np.arange(31, 92) - 31) / 30.0
    got = {}
    for where, extra in (("air", 'rise = [[0.0, 0.0], [0.15, 0.3]]\nair = { from = 0.0, to = 2.0 }\n'), ("ground", "")):
        code, out = make(TWIRL.format(extra=extra), until="perform")
        assert code == 0, out.get("error")
        (sp,) = out["stages"]["perform"]["mq"]["spin"]["spins"]
        assert sp["air"] == (where == "air") and len(sp["whips"]) == 1 and sp["neck_deg"] <= 70.0 and sp["twist_deg"] <= 35
        code, q = cli(["q", str(make.root / "build" / "t.blend"), FACING, "--frames", "31:91", "--project",
                       str(make.root)], capsys)
        assert code == 0, q
        v = np.array(q["values"])
        head, chest, hips = (np.degrees(np.unwrap(np.radians(v[:, i] - v[0, i]))) for i in range(3))
        root = np.degrees(v[:, 3])
        assert root.min() < -10.0 and root.max() > 365.0 and root[-1] == pytest.approx(360.0, abs=0.01)
        whip = sp["whips"][0]
        held = (t > 0.25) & (t < whip)
        assert np.abs(head[held]).max() < 3.0 and chest[held].max() > 40.0              # the head spots the front
        turning = (hips > 30.0) & (hips < 330.0)
        assert (chest[turning] > hips[turning]).all()                                     # the chest leads, the hips trail
        later = (t > whip + 0.25) & (head < 355.0)
        assert later.any() and (head[later] - chest[later]).min() > 30.0                  # the head ahead of the chest
        got[where] = (t, v, hips)
    t, v, hips = got["ground"]
    on = (hips > 20.0) & (hips < 340.0)
    assert np.ptp(v[on, 4:7], axis=0).max() < 1e-3                                       # on her left toes ...
    assert (v[on, 9] - v[0, 9]).min() > 0.05                                              # ... the right foot lifted
    code, out = make(TWIRL.format(extra="") + '\n[[key]]\ntarget = "Mq"\nprop = "rotation_euler"\nindex = 2\n'
                     'keys = [[0.0, 0.0], [1.0, 1.0]]\n', until="perform")
    assert code == 3 and "a [[key]] also keys 'Mq' rotation_euler[2]" in out["error"]


# ---------------------------------------------------------------- the camera: follow, dolly, keyed roll
FOLLOWED = """
[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -5.0, 1.0]
look = [0.0, 0.0, 0.9]
follow = { target = "cast:mq.center", axes = "x" }

[shot.aspect.9x16]
follow = { lag = 0.3 }
"""


def test_a_followed_camera_moves_with_her_along_x_only_and_never_turns(make, capsys):
    """She walks along +X and hops. The camera and its aim move by her x displacement since the shot's `from`, so the hop
    rises in the frame and the camera never turns; the 9x16 output eases the same move through its lag."""
    import numpy as np
    from mkmmd.core import perform as PF
    code, out = make(WALK.format(extra="") + FOLLOWED, until="shots", cast="yaw = 90")
    assert code == 0, out.get("error")
    assert out["stages"]["shots"]["a"]["moves"]["16x9"]["follow_m"] > 1.0
    expr = ('[list(bone("center").head), list(obj("a@16x9").loc), list(obj("a@16x9").matrix.col[2])[:3], '
            'list(obj("a@9x16").loc)]')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "29:90", "--project", str(make.root)], capsys)
    assert code == 0, q
    hips, cam, back, tall = (np.array([v[i] for v in q["values"]]) for i in range(4))
    moved = hips[:, 0] - hips[2, 0]                                                    # frame 31: the shot's `from`
    assert moved.max() > 1.0 and np.ptp(hips[:, 2]) > 0.2                              # she walked and hopped
    assert np.allclose(cam[:, 0], moved, atol=1e-4)                                    # x: hers, exactly
    assert np.allclose(cam[:, 1:], [-5.0, 1.0], atol=1e-4)                             # y and z stay: the hop rises
    assert np.ptp(back, axis=0).max() < 1e-4                                           # the aim went along: no turn
    eased = PF.lowpass(moved, 0.3 * 30)
    assert np.allclose(tall[:, 0], eased, atol=1e-4) and (moved - eased).max() > 0.05


DOLLY_ROLL = """
[[shot]]
name = "d"
from = 0.0
to = 1.0
at = [0.0, -5.0, 0.85]
look = [0.0, 0.0, 0.85]
keys = [{ t = 0.0 }, { t = 0.8, lens = 105 }]
frame = { subject = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]], fill = 0.5, solve = "dolly" }

[[shot]]
name = "r"
from = 1.0
to = 2.0
at = [0.0, -4.0, 1.0]
look = [0.0, 0.0, 1.0]
roll = 5
keys = [{ t = 1.0 }, { t = 1.5, roll = 40 }, { t = 1.9, roll = -20 }]
"""


def test_a_dolly_keeps_the_subject_its_size_as_the_lens_runs_and_a_keyed_roll_lands_on_its_keys(make, capsys):
    """The dolly: a 1.7 m subject on the aim point (with 0.15 m headroom, 1.85 m) fills half the frame height on every frame
    while the lens runs from 35 to 105 mm, so the camera stands 1.85 x lens / (0.5 x sensor height) away: 20.25 mm high on
    the 640 x 360 output, 36 mm on the 360 x 640 one. The roll: a key without `roll` takes the shot's (5), keys ease."""
    import math
    import numpy as np
    from mkmmd.core import perform as PF
    code, out = make(DOLLY_ROLL, with_cast=False, until="shots")
    assert code == 0, out.get("error")
    expr = ('[bpy.data.cameras["d@16x9"].lens, list(obj("d@16x9").loc), bpy.data.cameras["d@9x16"].lens, '
            'list(obj("d@9x16").loc)]')
    code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "29:63", "--project", str(make.root)], capsys)
    assert code == 0, q
    lens, cam, lens_t, cam_t = (np.array([v[i] for v in q["values"]]) for i in range(4))
    assert lens.min() == pytest.approx(35.0) and lens.max() == pytest.approx(105.0)
    away, away_t = np.linalg.norm(cam - [0, 0, 0.85], axis=1), np.linalg.norm(cam_t - [0, 0, 0.85], axis=1)
    assert np.allclose(away / lens, 1.85 / (0.5 * 20.25), rtol=1e-4)                   # three times as far at 105 mm
    assert np.allclose(away_t / lens_t, 1.85 / (0.5 * 36.0), rtol=1e-4)                # solved per output
    tilt = 'math.degrees(math.atan2(obj("r@16x9").matrix.col[0][2], obj("r@16x9").matrix.col[1][2]))'
    code, q = cli(["q", str(make.root / "build" / "t.blend"), tilt, "--frames", "61,67,76,88", "--project",
                   str(make.root)], capsys)
    assert code == 0, q
    got = np.array(q["values"], float)
    sign = math.copysign(1.0, got[0])                                                  # the shot's own 5 degrees sets it
    assert got * sign == pytest.approx([5.0, 5.0 + 35.0 * float(PF.smooth(0.4)), 40.0, -20.0], abs=0.01)


TYPED = """
[[text]]
name = "sign"
at = [0.0, 0.0, 1.0]
facing = [0.0, -1.0, 0.0]
box = [1.0, 0.3]
text = "mk"
fit = 0.8

[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -4.0, 1.0]
look = [0.0, 0.0, 1.0]
style = "silhouette"
hide = ["*"]
keep = ["sign*", "nothing_here*"]
"""


def test_a_look_pattern_counts_what_later_stages_make_and_a_typo_still_warns(make):
    """The text stage makes its objects after the shots stage: a look that keeps them is not warned about, a pattern that
    matches nothing in the finished scene is."""
    code, out = make(TYPED, with_cast=False)
    assert code == 0, out.get("error")
    warned = [ln for ln in warnings(out) if "pattern" in ln]
    assert len(warned) == 1 and "'nothing_here*' matches no object" in warned[0], warned


# ---------------------------------------------------------------- medium shifts and the cast on twos
MEDIUM = """[pose.mq]
feet = "floor"

[[shot]]
name = "a"
from = 0.0
to = 2.0
at = [0.0, -2.6, 0.9]
look = [0.0, 0.0, 0.9]
lens = 35

[[medium]]
from = 0.5
to = 1.0
style = "vector"
colors = { background = "#FFD21F", line = "#1B2A6B" }
tones = { dark = "#1B2A6B" }
materials = [{ match = "*", tone = "dark" }]

[medium.aspect.9x16]
style = "none"
"""


def test_a_medium_window_draws_the_lit_shot_in_its_own_look_over_its_frames_and_leaves_the_output_it_is_told_to(make, capsys,
                                                                                                               tmp_path):
    """The shot is lit; over the window's frames (0.5 .. 1.0 s: 46 .. 60) the 16x9 output is drawn flat in the window's
    colours, before and after it is lit again, and the 9x16 output, which the window leaves alone, stays lit."""
    import numpy as np
    from PIL import Image
    code, out = make(MEDIUM)
    assert code == 0, out.get("error")
    assert out["stages"]["shots"]["_medium"] == [{"index": 0, "frames": [46, 60], "looks": {"16x9": "vector", "9x16": None}}]

    def yellow(output, frames):
        code, look = cli(["look", str(make.root / "build" / "t.blend"), "--frames", frames, "--output", output, "--size",
                          "200", "--out", str(tmp_path / output), "--project", str(make.root)], capsys)
        assert code == 0, look.get("error")
        return [float((np.abs(np.asarray(Image.open(p).convert("RGB"), int) - (255, 210, 31)).max(-1) < 30).mean())
                for p in look["images"]]
    wide = yellow("16x9", "45,46,60,61")
    assert wide[0] == 0.0 and wide[1] > 0.5 and wide[2] > 0.5 and wide[3] == 0.0
    assert yellow("9x16", "53") == [0.0]


COLLAGE = MEDIUM.split("[[medium]]")[0] + """[[fx]]
kind = "mirror"
from = 0.5
to = 1.0
objects = "Mq_*"
back = ["#FFD21F", "#1B2A6B"]
fill = ["picture", "#FFFFFF"]
place = [0.3, 0.55]
"""


def test_a_collage_cuts_her_figure_out_of_the_frame_and_folds_it_onto_flat_colour(make, capsys, tmp_path):
    """mk look (as mk render) draws her coverage beside the frame, and the mirror lays her picture on yellow with its
    centroid at 0.3 of the width and its reflection, white on navy, across the middle: the shape her coverage has."""
    import numpy as np
    from PIL import Image
    code, out = make(COLLAGE)
    assert code == 0, out.get("error")
    code, look = cli(["look", str(make.root / "build" / "t.blend"), "--frames", "50", "--output", "16x9", "--size", "320",
                      "--out", str(tmp_path / "look"), "--project", str(make.root)], capsys)
    assert code == 0, look.get("error")
    img = np.asarray(Image.open(look["images"][0]).convert("RGB"), int)                     # 320 x 180
    [layer] = (tmp_path / "look" / "layers" / "16x9" / "figure").rglob("00050.png")
    cover = np.asarray(Image.open(layer).convert("L"), float) / 255.0 > 0.5
    assert cover.shape == (180, 320) and 0.01 < cover.mean() < 0.3                           # her, in the shot's camera
    yellow, white = np.abs(img - (255, 210, 31)).max(-1) < 40, (img > 225).all(-1)
    navy = np.abs(img - (27, 42, 107)).max(-1) < 40
    assert yellow[:, :160].mean() > 0.7 and (navy | white)[:, 160:].mean() > 0.9
    shown = np.abs(img[:, :159] - (255, 210, 31)).max(-1) > 120     # her picture; the JPEG smears colour a pixel
    folded = white[:, 161:][:, ::-1]                                # across the fold, so its two columns are left out
    assert np.nonzero(shown)[1].mean() == pytest.approx(0.3 * 320 - 0.5, abs=6.0)
    assert (shown & folded).sum() / (shown | folded).sum() > 0.85                            # her reflection
    assert abs(int(cover.sum()) - int(shown.sum())) < 0.05 * cover.sum()                    # her coverage, moved


STEPPED = """
[[stepped]]
from = 0.4
to = 1.0
cast = "mq"
fps = 6
"""


def test_on_twos_holds_her_whole_pose_between_steps_and_the_camera_still_moves_on_every_frame(make, capsys):
    """At 6 poses a second at 30 fps a pose is drawn every 5th frame from 0.4 s (frame 43) and held until the next: her
    root, hips and planted IK foot on every frame of the window are those of the plain walk at its last step, outside
    the window they are the plain walk's own, and the camera that follows her is the plain build's on every frame."""
    import numpy as np
    expr = ('[list(obj("Mq").loc), list(bone("center").head), list(bone("leg_ik.R").head), list(bone("head").head), '
            'list(obj("a@16x9").loc)]')
    got = {}
    for name, extra in (("plain", ""), ("stepped", STEPPED)):
        code, out = make(WALK.format(extra="") + FOLLOWED + extra, until="stepped", cast="yaw = 90")
        assert code == 0, out.get("error")
        code, q = cli(["q", str(make.root / "build" / "t.blend"), expr, "--frames", "38:66", "--project", str(make.root)],
                      capsys)
        assert code == 0, q
        got[name] = np.array([np.concatenate(v) for v in q["values"]])
    assert out["stages"]["stepped"]["mq"]["windows"] == [[43, 60]] and out["stages"]["stepped"]["mq"]["poses"] == 4
    frames = np.arange(38, 67)
    held = np.where((frames >= 43) & (frames < 61), 43 + (frames - 43) // 5 * 5, frames)
    body, cam = slice(0, 12), slice(12, 15)
    assert np.allclose(got["stepped"][:, body], got["plain"][held - 38, body], atol=1e-5)
    assert np.ptp(got["plain"][5:10, body], axis=0).max() > 0.01                      # she does move over a hold
    assert np.allclose(got["stepped"][:, cam], got["plain"][:, cam], atol=1e-6)
    assert np.all(np.abs(np.diff(got["stepped"][5:23, 12])) > 1e-4)                  # the camera, on every frame
