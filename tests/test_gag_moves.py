"""The gag moves, bpy-free (mkmmd.core.gags through mkmmd.core.moves.compile): every gag place natural and in reach, a
held prop's part brought where its place wants it with the fist round its grip, the legs' perform keys of a bounce and a
run, a chomp's bites numbered on across its entries, the mouth spans lip sync gives way in, and perform's squash keeping
the volume. The Blender half (the props bone-parented, the keys laid) is proved on scratch builds."""
import numpy as np
import pytest

from mkmmd.core import gags as GG
from mkmmd.core import moves as MV
from mkmmd.core import perform as PF

MARKS = {"arm.L": [0.0914, 0.006, 1.1795], "arm.R": [-0.0914, 0.006, 1.1795], "chest": [0.0, -0.0504, 1.019],
         "mouth": [0.0, -0.0647, 1.2795], "eye": [0.0, -0.044, 1.3229], "reach": 0.374, "upper": 0.209, "fore": 0.166,
         "hand": 0.166}
BEATS = [round(0.72 * k, 3) for k in range(40)]
DOWN = BEATS[::4]
CARROT = {"scale": 1.0, "bounds": {"min": [-0.15, -0.13, 0.0], "max": [0.15, 0.13, 0.93]},
          "use": {"grip": [{"name": "hold", "type": "hold", "center": [0, 0, 0.33], "axis": [0, 0, -1], "length": 0.5,
                            "radius": 0.0497, "radius_tip": 0.0162,
                            "profile": [[0.0, 0.0162], [0.25, 0.036], [0.5, 0.0497]]}],
                  "bite": [{"name": f"bite{k}", "point": [0, 0, 0.06 * k], "dir": [0, 0, -1], "prop": f"bite{k}"}
                           for k in (1, 2, 3, 4)]}}
BRUSH = {"scale": 1.0, "bounds": {"min": [-0.02, -0.08, 0.0], "max": [0.02, 0.02, 0.74]},
         "use": {"grip": [{"name": "hold", "type": "hold", "center": [0, 0, 0.26], "axis": [0, 0, 1], "length": 0.48,
                           "radius": 0.0184}],
                 "anchor": [{"name": "brush", "point": [0.0, -0.064, 0.675], "dir": [0, -1, 0]}]}}
PILLOW = {"scale": 1.0, "bounds": {"min": [-0.28, -0.085, -0.19], "max": [0.28, 0.085, 0.19]},
          "use": {"anchor": [{"name": "hug", "point": [0, 0.085, 0], "dir": [0, -1, 0]}],
                  "rest": [{"name": "hold_L", "type": "plane", "center": [0.2604, 0, 0], "normal": [1, 0, 0]},
                           {"name": "hold_R", "type": "plane", "center": [-0.2604, 0, 0], "normal": [-1, 0, 0]}]}}
PROPS = {"carrot": CARROT, "brush": BRUSH, "pillow": PILLOW}


def compile(entries):
    return MV.compile(entries, MARKS, BEATS, downbeats=DOWN, props=PROPS)


@pytest.mark.parametrize("name", GG.PLACES)
def test_every_gag_place_keeps_the_wrist_natural_the_arm_untwisted_and_in_reach(name):
    for side in ("L", "R"):
        pt, d, p, _, _, bend, pole = MV.resolve(name, side, MARKS)
        assert bend <= MV.HELD + 1e-6
        assert MV.overtwist(MV.twisted(side, MARKS, pt, d, p, pole)) == 0.0
        assert np.linalg.norm(pt - np.array(MARKS[f"arm.{side}"])) <= MARKS["reach"] + 1e-6


@pytest.mark.parametrize("name", GG.PLACES)
def test_the_right_hand_s_gag_places_mirror_the_left_s(name):
    (pl, dl, ql, *_), (pr, dr, qr, *_) = (MV.resolve(name, s, MARKS) for s in ("L", "R"))
    M = np.array([-1.0, 1.0, 1.0])
    assert pl * M == pytest.approx(pr) and dl * M == pytest.approx(dr) and ql * M == pytest.approx(qr)


def test_a_chomp_brings_each_bite_to_the_lips_on_its_hit_and_fires_the_bite_keys_counted_across_chomps():
    es = [{"name": "chomp", "t": 1.0, "dur": 1.5, "prop": "carrot", "hits": [1.3, 2.0]},
          {"name": "chomp", "t": 3.0, "dur": 1.0, "prop": "carrot", "hits": [3.4]}]
    res = compile(es)
    assert res["props"]["carrot"] == {"bite1": 1.3, "bite2": 2.0, "bite3": 3.4}
    held = res["held"]["carrot"]
    assert held["bone"] == "wrist" and held["side"] == "R"
    R, T = np.array(held["rel"][0]), np.array(held["rel"][1])
    keys = {k["t"]: k for k in res["hands"]["R"]}
    lips = GG.lips(MARKS, None) + np.array([0.0, GG.LIP_INSET, 0.0])
    for t, k in ((1.3, 1), (2.0, 2), (3.4, 3)):
        key = keys[t]
        d, p = np.array(key["dir"]), np.array(key["palm"])
        H = GG._frame(d, p)
        at = np.array(key["at"]) + H @ T + H @ R @ np.array([0.0, 0.0, 0.06 * k])     # bite k's point in her frame
        assert at == pytest.approx(lips, abs=2e-3)
        assert np.linalg.norm(np.array(key["at"]) - np.array(MARKS["arm.R"])) <= MARKS["reach"]
    assert (H @ R @ np.array([0.0, 0.0, -1.0]))[2] > 0.3          # its tip up to the mouth out of the fist's thumb side
    mouth = res["perform"]["mouth"]
    assert mouth == [[1.0, 2.5], [3.0, 4.0]]
    a = next(e for e in res["expressions"] if e["morph"] == "a")["keys"]
    shut = {round(t, 2): v for t, v in a}
    assert shut[1.32] == 0.0 and shut[1.26] == 1.0                # wide open as it comes, shut on the hit


def test_a_chomp_refuses_more_bites_than_the_prop_has_and_a_prop_held_by_two_hands():
    with pytest.raises(MV.MoveError, match="has 4 bites"):
        compile([{"name": "chomp", "t": 1.0, "dur": 3.0, "prop": "carrot", "hits": [1.3, 1.7, 2.1, 2.5, 2.9]}])
    with pytest.raises(MV.MoveError, match="held"):
        compile([{"name": "chomp", "t": 1.0, "dur": 1.0, "prop": "carrot", "hits": [1.4]},
                 {"name": "chomp", "t": 3.0, "dur": 1.0, "prop": "carrot", "hits": [3.4], "hand": "L"}])
    with pytest.raises(MV.MoveError, match="needs `prop`"):
        compile([{"name": "chomp", "t": 1.0, "dur": 1.0, "hits": [1.4]}])


def test_a_brush_puts_the_bristles_on_the_teeth_facing_them_and_scrubs_on_the_beats():
    res = compile([{"name": "brush", "t": 1.0, "dur": 1.5, "hand": "L", "prop": "brush"}])
    held = res["held"]["brush"]
    R, T = np.array(held["rel"][0]), np.array(held["rel"][1])
    keys = [k for k in res["hands"]["L"] if 1.0 <= k["t"] <= 2.5 and "fingers" in k]
    lips = GG.lips(MARKS, None) + np.array([0.0, GG.BRUSH_INSET, 0.0])
    xs = []
    for k in keys:
        H = GG._frame(np.array(k["dir"]), np.array(k["palm"]))
        tip = np.array(k["at"]) + H @ T + H @ R @ np.array([0.0, -0.064, 0.675])
        assert tip[1:] == pytest.approx(lips[1:], abs=3e-3)       # on the teeth, at their height
        xs.append(tip[0])
        assert (H @ R @ np.array([0.0, -1.0, 0.0]))[1] > 0.8      # the bristles face her
    assert max(xs) - min(xs) == pytest.approx(2 * GG.SCRUB, abs=2e-3)
    assert res["places"]["L"]["brush@brush"]["wrist_deg"] < MV.HELD


def test_a_yawn_hugging_a_pillow_rides_it_on_the_chest_its_back_on_the_body_a_palm_on_each_end():
    clear = lambda P: float(max((np.linalg.norm(np.asarray(P).reshape(-1, 3) - [0.0, -0.02, 1.05], axis=1)  # noqa: E731
                                 - 0.13).min(), 0.0))
    res = MV.compile([{"name": "yawn", "t": 1.0, "dur": 2.0, "prop": "pillow"}], MARKS, BEATS, clear=clear,
                     props=PROPS)
    held = res["held"]["pillow"]
    assert held["bone"] == "chest"
    R, origin = np.array(held["pose"][0]), np.array(held["pose"][1])
    back = origin + R @ np.array([0.0, 0.085, 0.0])
    assert clear(back) == pytest.approx(GG.HUG_GAP, abs=0.006)     # against her, a little off
    assert origin[2] + 0.19 == pytest.approx(MARKS["mouth"][2] - GG.HUG_DROP, abs=1e-6)   # its top under the chin
    pl, pr = (res["places"][s]["pillow@pillow"] for s in ("L", "R"))
    assert pl["wrist_deg"] == pytest.approx(pr["wrist_deg"])     # a mirror image: the hand_mirror check holds it so
    assert [e["morph"] for e in res["expressions"]] == ["a", "o", "calm"] and res["perform"]["mouth"] == [[1.0, 3.0]]


def test_a_bounce_pumps_the_knees_on_its_hits_and_jumps_with_the_fists_up_on_its_accents():
    res = compile([{"name": "bounce", "t": 0.0, "dur": 4.0, "accents": [2.16]}])
    legs = res["perform"]
    assert legs["bounce"]["beats"] == [b for b in BEATS if b < 4.0]
    assert max(v for _, v in legs["rise"]) == GG.JUMP["height"] and legs["air"][0]["from"] == pytest.approx(2.17)
    up = [k["t"] for k in res["hands"]["L"] if k["at"] == res["hands"]["L"][0]["at"]]
    assert up                                                      # the fists dip and come back on the beats
    ups = {k["t"] for s in ("L", "R") for k in res["hands"][s]
           if k["at"][2] > MARKS["arm.L"][2] + 0.2}
    assert 2.16 in ups                                             # both fists punched up on the accent
    assert res["hands"]["L"][3]["at"] == pytest.approx(np.array(res["hands"]["R"][3]["at"]) * [-1, 1, 1])


def test_a_run_in_place_steps_on_the_spot_knees_on_each_beat_and_halfway_the_fists_pumping_opposite():
    res = compile([{"name": "run_in_place", "t": 1.44, "dur": 2.88}])
    walk = res["perform"]["walk"]
    assert walk["stride"] == 0.0 and walk["slide"] == 1.0 and walk["first"] == "L"
    assert len(walk["steps"]) % 2 == 0 and np.diff(walk["steps"]) == pytest.approx(0.36, abs=1e-3)
    L = {k["t"]: k for k in res["hands"]["L"]}
    R = {k["t"]: k for k in res["hands"]["R"]}
    for t in (1.44, 1.8):                                          # opposite arms: one forward when the other is back
        assert (L[t]["at"][1] - R[t]["at"][1]) * (1 if t == 1.44 else -1) > 0.2


def test_a_wag_wags_the_raised_finger_on_the_beats_with_the_other_hand_on_the_hip():
    res = compile([{"name": "wag", "t": 1.44, "dur": 1.44, "hand": "L"}])
    out, inn = (np.array(MV.resolve(n, "L", MARKS)[1]) for n in ("wag_out", "wag_in"))
    dirs = {k["t"]: np.array(k["dir"]) for k in res["hands"]["L"] if "fingers" in k}
    assert dirs[1.44] == pytest.approx(out, abs=1e-3) and dirs[1.8] == pytest.approx(inn, abs=1e-3)
    assert any(k["at"] == pytest.approx(list(MV.resolve("hip", "R", MARKS)[0]), abs=1e-4) for k in res["hands"]["R"])
    assert {e["morph"] for e in res["expressions"]} >= {"jito", "mouth_down"}


def test_chores_loop_on_the_beats_and_a_stir_circles_once_a_beat():
    res = compile([{"name": "chores", "t": 1.44, "dur": 1.44, "chore": "stir"}])
    ring = [np.array(k["at"]) for k in res["hands"]["R"] if 1.44 <= k["t"] < 2.16]
    centre = np.array(MV.resolve("stir", "R", MARKS)[0])
    assert [np.linalg.norm(p - centre) for p in ring[1:]] == pytest.approx([GG.STIR] * (len(ring) - 1), abs=1e-3)
    with pytest.raises(MV.MoveError, match="chore"):
        compile([{"name": "chores", "t": 1.0, "chore": "dust"}])
    with pytest.raises(MV.MoveError, match="is for a gag move"):
        compile([{"name": "point", "t": 1.0, "chore": "stir"}])


def test_the_hand_checks_see_the_gag_moves_hands_at_their_places():
    rows = GG.rows({"name": "wag", "hand": "L"}, {"L": False, "R": False})
    assert rows == [("L", ["wag_out", "wag_in"], False), ("R", ["hip"], False)]
    assert all(m for _, _, m in GG.rows({"name": "bounce"}, {"L": False, "R": False}))
    assert not any(m for _, _, m in GG.rows({"name": "run_in_place"}, {"L": False, "R": False}))
    assert GG.rows({"name": "bounce"}, {"L": False, "R": True}) == [("L", ["club", "pump_up"], True)]


def test_a_squash_keeps_the_volume_squashes_on_the_hit_and_springs_back_past_its_height():
    ts = np.arange(0.0, 2.0, 1.0 / 240.0)
    s = PF.squash(ts, [0.5], depth=0.1, stretch=0.05, attack=0.04, rebound=0.14)
    assert s[0] == 1.0 and s[np.argmin(abs(ts - 0.5))] == pytest.approx(0.9, abs=1e-3)
    peak = ts[np.argmax(s)]
    assert 0.6 < peak < 0.7 and s.max() == pytest.approx(1.05, abs=0.01)
    assert abs(s[-1] - 1.0) < 1e-3
    with pytest.raises(ValueError):
        PF.squash(ts, [0.5], depth=0.1, stretch=0.2)
