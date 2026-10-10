"""Moves, bpy-free (mkmmd.core.moves): `[[move.<cast>]]` entries compiled to hand keys, lean / tilt keys, twitches and
expressions. The Blender half (the keys laid into the pose and perform tables, a move held on a real arm) is checked in
tests/test_build_stages.py."""
import math

import pytest

from mkmmd.core import moves as MV

MARKS = {"arm.L": [0.09, 0.0, 1.25], "arm.R": [-0.09, 0.0, 1.25], "chest": [0.0, -0.06, 1.12],
         "mouth": [0.0, -0.07, 1.35], "eye": [0.0, -0.05, 1.41], "reach": 0.37}
BEATS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def near(p):
    return pytest.approx(p, abs=1e-4)                                   # the keys are rounded to 0.1 mm


@pytest.mark.parametrize("reach", [0.30, 0.37, 0.55])
@pytest.mark.parametrize("name", sorted(MV.OUT))
def test_a_place_with_the_arm_out_is_within_reach_of_any_arm(name, reach):
    marks = dict(MARKS, reach=reach)
    for side in ("L", "R"):
        d = math.dist(MV.place(name, side, marks)[0], marks[f"arm.{side}"])
        assert 0.8 * reach < d < reach                                    # out, and never past the arm's length


def test_the_hands_mirror_each_other():
    for name in ("point", "heart", "sparkle", "up"):
        (l, dl, pl, _), (r, dr, pr, _) = MV.place(name, "L", MARKS), MV.place(name, "R", MARKS)
        assert l == pytest.approx([-r[0], r[1], r[2]]) and dl[0] == pytest.approx(-dr[0])


def test_a_move_holds_its_last_place_to_its_end_and_the_hand_goes_home_after_it():
    keys = MV.compile([{"name": "point", "t": 1.0, "dur": 0.8, "hand": "L"}], MARKS)["hands"]["L"]
    home = MV.place("rest", "L", MARKS)[0]
    point = MV.place("point", "L", MARKS)[0]
    assert [k["t"] for k in keys] == pytest.approx([1.0 - MV.EASE, 1.0, 1.8, 1.8 + MV.EASE])
    assert [k["at"] for k in keys] == [near(p) for p in (home, point, point, home)]
    assert keys[1]["fingers"] == "point" and keys[0]["fingers"] == "relaxed"


def test_moves_close_together_go_straight_on_and_far_apart_go_home_between():
    def ats(gap):
        es = [{"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"},
              {"name": "heart_wink", "t": 1.5 + gap, "dur": 0.5, "hand": "L"}]
        return [k["at"] for k in MV.compile(es, MARKS)["hands"]["L"]]
    home = near(MV.place("rest", "L", MARKS)[0])
    assert ats(0.2).count(home) == 2                                      # in before the first, out after the last
    assert ats(1.0).count(home) == 4                                      # and home between them


def test_one_hand_cannot_play_two_moves_at_once():
    with pytest.raises(MV.MoveError, match="the L hand plays two moves at once"):
        MV.compile([{"name": "point", "t": 1.0, "dur": 1.0, "hand": "L"},
                    {"name": "heart_wink", "t": 1.5, "dur": 1.0, "hand": "L"}], MARKS)
    out = MV.compile([{"name": "point", "t": 1.0, "dur": 1.0, "hand": "L"},
                      {"name": "point", "t": 1.5, "dur": 1.0, "hand": "R"}], MARKS)
    assert out["hands"]["L"] and out["hands"]["R"]                        # the other hand may


def test_a_beat_move_pats_on_each_beat_inside_it_and_needs_one():
    keys = MV.compile([{"name": "chest_pat", "t": 0.9, "dur": 1.0, "hand": "R"}], MARKS, BEATS)["hands"]["R"]
    on = MV.place("chest", "R", MARKS)[0]
    pats = [k["t"] for k in keys if k["at"] == near(on)]
    assert pats == pytest.approx([1.0, 1.5])                              # 0.5 and 2.0 are outside it
    with pytest.raises(MV.MoveError, match="no beat between"):
        MV.compile([{"name": "chest_pat", "t": 1.1, "dur": 0.3}], MARKS, BEATS)


def test_a_both_hands_move_keys_both_and_a_resting_mic_hand_waits_at_the_mouth():
    out = MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "sparkle", "t": 1.0, "dur": 0.5}], MARKS)
    mic = near(MV.place("mic", "R", MARKS)[0])
    assert out["hands"]["L"] and out["hands"]["R"]
    assert out["hands"]["R"][0]["at"] == mic and out["hands"]["R"][-1]["at"] == mic     # from the mic and back to it
    only = MV.compile([{"name": "rest", "hand": "R", "place": "mic"}], MARKS)["hands"]
    assert only["L"] == [] and len(only["R"]) == 1 and only["R"][0]["t"] == 0.0 and only["R"][0]["at"] == mic
    with pytest.raises(MV.MoveError, match="rest"):
        MV.compile([{"name": "rest", "hand": "R", "place": "chest"}], MARKS)


def test_the_member_s_yaw_turns_the_hand_directions_into_the_world():
    k0 = MV.compile([{"name": "point", "t": 1.0, "hand": "L"}], MARKS)["hands"]["L"][1]
    k90 = MV.compile([{"name": "point", "t": 1.0, "hand": "L"}], MARKS, yaw=90.0)["hands"]["L"][1]
    assert k0["at"] == k90["at"]                                          # places stay in the member's own frame
    assert k90["dir"] == pytest.approx([-k0["dir"][1], k0["dir"][0], k0["dir"][2]], abs=1e-4)


def test_faces_close_together_hold_through_and_a_face_needs_a_morph():
    out = MV.compile([{"name": "face", "t": 1.0, "dur": 0.5, "morph": "smile_eyes", "value": 0.5},
                      {"name": "face", "t": 1.55, "dur": 0.5, "morph": "smile_eyes"}], MARKS)
    (ex,) = out["expressions"]
    assert ex["morph"] == "smile_eyes" and len(ex["keys"]) == 4           # one held span, not two
    assert max(v for _, v in ex["keys"]) == 1.0
    with pytest.raises(MV.MoveError, match="needs `morph`"):
        MV.compile([{"name": "face", "t": 1.0}], MARKS)


def test_body_moves_ease_in_and_out_and_two_at_once_are_refused():
    out = MV.compile([{"name": "lean_back", "t": 1.0, "dur": 1.0}], MARKS)
    assert [v for _, v in out["lean"]] == [0.0, -11.0, -11.0, 0.0] and out["lean"][-1][0] == 2.0
    with pytest.raises(MV.MoveError, match="both lean the body"):
        MV.compile([{"name": "lean_back", "t": 1.0, "dur": 1.0}, {"name": "into_lens", "t": 1.5, "dur": 1.0}], MARKS)


@pytest.mark.parametrize("entries, marks, frag", [
    ([{"name": "moonwalk", "t": 1.0}], MARKS, "no such move"),
    ([{"name": "point", "t": 1.0, "speed": 2}], MARKS, "expected {name, t"),
    ([{"name": "point", "t": 1.0, "dur": 0}], MARKS, "dur must be positive"),
    ([{"name": "point", "t": 1.0}], {k: v for k, v in MARKS.items() if k != "reach"}, "landmarks missing"),
])
def test_bad_moves_are_refused_naming_the_fault(entries, marks, frag):
    with pytest.raises(MV.MoveError, match=frag):
        MV.compile(entries, marks)
