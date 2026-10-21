"""Stepped motion (mkmmd.core.stepped): the frames a [[stepped]] window draws a new pose on, the pose each frame shows, and
the entries' errors. The stage that keys the holds in Blender is checked in test_build_stages.py."""
import pytest

from mkmmd.core import stepped as STP

CAST = ["rin", "len"]


def test_on_twos_at_60_fps_draws_a_pose_every_fifth_frame_and_holds_it_until_the_next():
    w = STP.normalize([{"from": 1.0, "to": 2.0, "cast": "rin"}], CAST, 60, 181)[0]
    assert (w["cast"], w["from"], w["to"], w["fps"]) == (["rin"], 241, 301, 12.0)
    assert w["steps"] == list(range(241, 301, 5))
    assert [STP.held(w, f) for f in (240, 241, 245, 246, 299, 300, 301)] == [240, 241, 241, 246, 296, 296, 301]


def test_a_rate_that_does_not_divide_the_clips_draws_on_the_nearest_frames():
    assert STP.steps(0, 12, 30, 12) == [0, 2, 5, 8, 10]                        # 2.5 frames apart
    assert STP.steps(10, 13, 60, 60) == [10, 11, 12]                           # on ones: every frame


def test_a_window_without_a_cast_takes_everyone_and_windows_may_meet_but_not_overlap_for_a_member():
    ws = STP.normalize([{"from": 1.0, "to": 2.0, "cast": "len"}, {"from": 0.0, "to": 1.0}, {"from": 1.5, "to": 2.5,
                        "cast": ["rin"]}], CAST, 30, 1)
    assert [(w["index"], w["cast"]) for w in ws] == [(1, CAST), (0, ["len"]), (2, ["rin"])]
    with pytest.raises(STP.StepError, match="overlap for len"):
        STP.normalize([{"from": 0.0, "to": 1.2}, {"from": 1.0, "to": 2.0, "cast": "len"}], CAST, 30, 1)


@pytest.mark.parametrize("entry, frag", [
    ({"from": 1.0, "to": 1.005}, "under a frame"),
    ({"from": 0.0, "to": 50.0}, "outside the clip"),
    ({"from": 0.0, "to": 1.0, "fps": 90}, "at most the clip's 60"),
    ({"from": 0.0, "to": 1.0, "fps": 0}, "above 0"),
    ({"from": 0.0, "to": 1.0, "cast": "nobody"}, "no cast member 'nobody'"),
    ({"from": 0.0, "to": 1.0, "cast": []}, "a member's name or a list"),
    ({"from": 0.0, "to": 1.0, "speed": 2}, "unknown keys")])
def test_bad_entries_say_what_is_wrong(entry, frag):
    with pytest.raises(STP.StepError, match=frag):
        STP.normalize([entry], CAST, 60, 181, last_frame=1000)
