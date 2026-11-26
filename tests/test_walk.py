"""Walking, hopping and tumbling (mkmmd.core.perform.walk_plan, tumble_plan): feet that do not slide while down, steps a
stride apart with the body midway, the treadmill, hops on their parabola with the feet carried, and turns about the
hips that add up."""
import math

import numpy as np
import pytest

from mkmmd.core import armreach as AR
from mkmmd.core import perform as PF

FEET = {"L": (0.09, 0.0), "R": (-0.09, 0.0)}
TS = np.arange(0.0, 6.0, 1.0 / 120.0)
BEAT = 0.375                                   # 45 samples at 120 per second: every event on a sample


def world_feet(plan):
    """Each foot's floor position in the walk's frame (the root's travel plus the foot relative to the root)."""
    return {s: plan["feet"][s] + plan["shift"] for s in plan["feet"]}


def planted(plan, side):
    """Frames where the foot is on the ground and still (not lifting, not in flight)."""
    w = world_feet(plan)[side]
    still = np.r_[True, np.linalg.norm(np.diff(w, axis=0), axis=1) < 1e-9]
    return still & (plan["lift"][side] == 0.0) & (plan["height"] == 0.0)


def test_steps_land_a_stride_past_the_other_foot_with_the_body_midway():
    steps = [1.0 + k * BEAT for k in range(6)]
    p = PF.walk_plan(TS, {"steps": steps, "stride": 0.6, "first": "L"}, FEET)
    w = world_feet(p)
    for k, t in enumerate(steps):
        i = int(round(t * 120))
        s = "L" if k % 2 == 0 else "R"
        o = "R" if s == "L" else "L"
        assert -w[s][i, 1] == pytest.approx(0.6 * (k + 1), abs=1e-6)        # forward is -y, a stride per step
        assert w[s][i, 0] == pytest.approx(FEET[s][0])                         # on its own side
        assert -p["shift"][i, 1] == pytest.approx(0.5 * (-w[s][i, 1] - w[o][i, 1]), abs=1e-6)   # the body midway
        assert p["lift"][s][i] == 0.0 and p["dip"][i] == pytest.approx(0.015)
    assert p["steps"] == 6 and p["hops"] == 0 and np.allclose(p["shift"][:, 0], 0.0)


def test_a_foot_on_the_ground_never_slides_and_swings_through_its_lift():
    steps = [1.0 + k * BEAT for k in range(8)]
    p = PF.walk_plan(TS, {"steps": steps, "lift": 0.08}, FEET)
    w = world_feet(p)
    for s in ("L", "R"):
        down = planted(p, s)
        runs = np.split(np.flatnonzero(down), np.flatnonzero(np.diff(np.flatnonzero(down)) > 1) + 1)
        assert len(runs) >= 4
        for r in runs:
            assert np.ptp(w[s][r], axis=0).max() < 1e-9                     # planted: not a millimetre of skating
        assert p["lift"][s].max() == pytest.approx(0.08, abs=0.002)


def test_on_a_treadmill_the_body_stays_and_the_planted_foot_slides_back():
    steps = [1.0 + k * BEAT for k in range(6)]
    a = PF.walk_plan(TS, {"steps": steps}, FEET)
    b = PF.walk_plan(TS, {"steps": steps, "slide": 1.0}, FEET)
    assert np.allclose(b["shift"], 0.0) and np.allclose(a["feet"]["L"], b["feet"]["L"])    # relative to the body: same
    i, j = int(round(steps[2] * 120)), int(round((steps[3] - 0.05) * 120))                 # L down between its steps
    assert b["feet"]["L"][j, 1] > b["feet"]["L"][i, 1] + 0.1                               # sliding back under her


def test_a_hop_flies_its_parabola_carries_the_feet_and_the_back_foot_steps_next():
    steps = [1.0, 1.0 + BEAT, 1.0 + 2 * BEAT]
    hop = {"from": 1.0 + 2 * BEAT + 0.25, "to": 1.0 + 2 * BEAT + 0.75}
    after = [hop["to"] + BEAT]
    p = PF.walk_plan(TS, {"steps": steps + after, "hops": [hop], "stride": 0.5}, FEET)
    T = hop["to"] - hop["from"]
    mid = int(round((hop["from"] + T / 2) * 120))
    assert p["height"][mid] == pytest.approx(PF.GRAVITY * T * T / 8, rel=1e-3) and p["air"] == [(hop["from"], hop["to"])]
    land = int(round(hop["to"] * 120))
    w = world_feet(p)
    body = -p["shift"][land, 1]
    ahead = max(("L", "R"), key=lambda s: -w[s][land, 1])
    assert -w[ahead][land, 1] == pytest.approx(body + 0.125) and p["height"][land] == 0.0
    back = "R" if ahead == "L" else "L"
    k = int(round(after[0] * 120))
    assert -w[back][k, 1] == pytest.approx(body + 0.125 + 0.5)              # the back foot steps past the front one
    high = PF.walk_plan(TS, {"steps": steps, "hops": [{**hop, "height": 0.4}]}, FEET)
    assert high["height"].max() == pytest.approx(0.4, abs=1e-3)
    side = PF.walk_plan(TS, {"hops": [{**hop, "move": [0.8, 0.0]}]}, FEET)  # a knight's hop: sideways
    assert side["shift"][-1] == pytest.approx([0.8, 0.0])


def test_a_catwalk_sets_each_foot_on_the_centre_line_and_carries_the_hips_over_the_foot_that_is_down():
    steps = [1.0 + k * 2 * BEAT for k in range(6)]
    p = PF.walk_plan(TS, {"steps": steps, "cross": 1.0, "sway": 0.03}, FEET)
    w = world_feet(p)
    for k, t in enumerate(steps):
        i = int(round(t * 120))
        s = "L" if k % 2 == 0 else "R"
        assert w[s][i, 0] == pytest.approx(0.0, abs=1e-9)                 # on the line, not on its own side
        assert p["sway"][i] == pytest.approx(0.0, abs=1e-9)               # the hips pass the middle as it lands
    for k, (a, b) in enumerate(zip(steps, steps[1:])):
        down = "L" if k % 2 == 0 else "R"                                  # the foot that landed on a carries her to b
        mid = int(round(0.5 * (a + b) * 120))
        assert p["sway"][mid] == pytest.approx(0.03 if down == "L" else -0.03, abs=1e-4)
    half = PF.walk_plan(TS, {"steps": steps, "cross": 0.5}, FEET)
    assert world_feet(half)["L"][int(round(steps[0] * 120)), 0] == pytest.approx(0.045)


def test_the_arms_swing_while_she_walks_and_settle_when_she_stops_in_stride():
    steps = [1.0 + k * BEAT for k in range(4)]
    p = PF.walk_plan(TS, {"steps": steps, "stride": 0.6, "arms": 0.12}, FEET)
    walking = (TS > steps[0]) & (TS < steps[-1])
    assert np.abs(p["swing"]["L"][walking]).max() == pytest.approx(0.12, abs=0.01)
    settled = TS > steps[-1] + PF.ARMS_SETTLE
    assert np.abs(p["swing"]["L"][settled]).max() < 1e-9                # her feet a stride apart, her hands at rest
    assert np.abs(np.diff(p["swing"]["L"])).max() < 0.01                 # a pendulum: no jump, no whip (a slide's
    #                                                                      return was 0.02 a sample)
    for k, t in enumerate(steps[1:-1], 1):                               # L lands first: on the odd steps R does
        lands = "L" if k % 2 == 0 else "R"
        i = int(round(t * 120))
        assert p["swing"]["R" if lands == "L" else "L"][i] == pytest.approx(0.12, abs=1e-6)     # the other hand ahead
        assert p["swing"][lands][i] == pytest.approx(-0.12, abs=1e-6)                          # its own hand behind


def test_the_pelvis_goes_with_the_legs_and_the_chest_against_it():
    steps = [1.0 + k * BEAT for k in range(6)]                           # L, R, L, R, L, R
    p = PF.walk_plan(TS, {"steps": steps, "twist": 8, "list": 4}, FEET)
    right, left = int(round(steps[3] * 120)), int(round(steps[2] * 120))
    assert math.degrees(p["twist"][right]) == pytest.approx(8.0, abs=1e-6)        # turned left: the right hip forward
    assert math.degrees(p["twist"][left]) == pytest.approx(-8.0, abs=1e-6)
    mid = int(round(0.5 * (steps[2] + steps[3]) * 120))                  # the left foot down, the right in the air
    assert math.degrees(p["list"][mid]) == pytest.approx(-4.0, abs=0.01)  # the left hip up, the right one hanging
    assert abs(p["list"][right]) < 1e-9                                   # level as both feet are down
    assert math.degrees(p["chest"]["turn"][right]) == pytest.approx(-4.8) and p["chest"]["tilt"][mid] > 0
    still = TS > steps[-1] + PF.ARMS_SETTLE
    assert not p["twist"][still].any() and not p["list"][still].any()     # and square again once she has stopped


def test_a_walking_arm_swings_from_the_shoulder_and_a_posed_hand_stays_put():
    ts = np.arange(0.0, 1.0, 1.0 / 60.0)
    S, upper, fore = np.array([0.15, 0.0, 1.30]), 0.22, 0.20
    hang = np.tile(S + 0.97 * (upper + fore) * np.array([0.13, 0.03, -1.0]) / np.linalg.norm([0.13, 0.03, -1.0]), (60, 1))
    pole = np.tile(S + np.array([AR.POLE[0], AR.POLE[1], -AR.POLE[2]]), (60, 1))   # the pose stage's: the elbow out
    swing = 0.12 * np.sin(2 * np.pi * ts)                              # forward, then back
    fwd, left = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    walks = np.ones(60)
    at, poles, turns = PF.swing_arm(ts, S, hang, pole, swing, upper, fore, fwd, left, 1.0, walks)
    front, back = int(np.argmax(swing)), int(np.argmin(swing))
    assert (at[front] - at[back]) @ fwd == pytest.approx(0.24, abs=0.03)         # as far each way as the plan says ...
    assert at[front][2] > hang[front][2] + 0.02 and at[front][0] < hang[front][0] - 0.01   # ... up and in, on an arc
    reach = np.linalg.norm(at - S, axis=1)
    assert reach[front] < reach[0] - 0.01 < reach[back]                 # the elbow folds in front and opens behind
    assert reach.max() < upper + fore
    elbows = np.array([AR.pole_elbow(S, at[i], upper, fore, poles[i]) for i in range(60)])
    fore_arm = at - elbows                                              # seen from the front, where the lens is: the
    lean = np.degrees(np.arctan2(fore_arm[:, 0], -fore_arm[:, 2]))      # forearm's lean out from straight down
    assert np.ptp(lean) < 12.0                                          # it does not flap across her and back (the
    #                                                                     pose's own pole, its elbow out: 37 degrees)
    assert ((elbows - S) @ fwd).max() < 0.01                            # the elbow stays behind her shoulder: the fold
    assert (fore_arm[front] @ fwd) > 0.1                                 # lifts the forearm forward
    lag = int(round(PF.SWING_LAG * 60))                                 # the hand turns as the forearm does, a moment
    for i in (front, back):                                             # late: the wrist stays as straight as it hung
        was = hang[i] - AR.pole_elbow(S, hang[i], upper, fore, pole[i])
        now = at[i] - AR.pole_elbow(S, at[i], upper, fore, poles[i])
        q = turns[i + lag]
        got = was + 2.0 * np.cross(q[1:], np.cross(q[1:], was) + q[0] * was)
        assert math.degrees(math.acos(got @ now / (np.linalg.norm(got) * np.linalg.norm(now)))) < 1.0
    assert math.degrees(2.0 * math.acos(turns[front + lag][0])) > 20.0  # it has turned with the swing, not stayed down
    assert np.allclose(np.linalg.norm(poles - S, axis=1), np.linalg.norm(pole - S, axis=1))    # the pole goes round too
    chest = np.tile(S + np.array([0.02, -0.25, -0.05]), (60, 1))        # a hand held before her chest does not hang
    held, kept, still = PF.swing_arm(ts, S, chest, pole, swing, upper, fore, fwd, left, 1.0, walks)
    assert np.allclose(held, chest) and np.allclose(kept, pole) and np.allclose(still, [1.0, 0.0, 0.0, 0.0])
    stood, same, _ = PF.swing_arm(ts, S, hang, pole, 0.0 * swing, upper, fore, fwd, left, 1.0, 0.0 * walks)
    assert np.allclose(stood, hang) and np.allclose(same, pole)         # and one that hangs while she stands keeps its pose


def test_a_step_after_a_pause_starts_from_a_stand_so_she_holds_still_between_two_walks():
    steps = [1.0, 1.0 + 2 * BEAT, 4.0, 4.0 + 2 * BEAT]               # 2.25 s after the second step: a pause
    p = PF.walk_plan(TS, {"steps": steps, "sway": 0.03}, FEET)
    still = (TS > steps[1] + 0.01) & (TS < 4.0 - PF.FIRST_SWING - 0.01)
    assert np.ptp(p["shift"][still], axis=0).max() < 1e-9             # the body waits where the last step put it
    for s in ("L", "R"):
        assert np.ptp(p["feet"][s][still], axis=0).max() < 1e-9 and p["lift"][s][still].max() == 0.0
    assert np.abs(p["sway"][still]).max() < 1e-9                       # the hips are not swaying either
    lifted = TS[(p["lift"]["L"] > 0) & (TS > 2.0)]                     # L steps third: lifted only for its swing
    assert lifted.min() == pytest.approx(4.0 - PF.FIRST_SWING, abs=1.5 / 120) and lifted.max() < 4.0


@pytest.mark.parametrize("spec, frag", [
    ({"steps": [1.0, 1.5], "hops": [{"from": 1.2, "to": 1.8}]}, "inside the hop"),
    ({"hops": [{"from": 1.0, "to": 2.0}, {"from": 1.5, "to": 2.5}]}, "before the hop before it lands"),
    ({"steps": [1.0, 1.0]}, "two steps at one time"),
    ({"steps": [1.0], "strides": 0.5}, "unknown"),
    ({"hops": [{"from": 2.0, "to": 1.0}]}, "from before to"),
])
def test_a_walk_that_cannot_be_says_why(spec, frag):
    with pytest.raises(ValueError, match=frag):
        PF.walk_plan(TS, spec, FEET)


def test_tumbles_turn_about_their_axis_add_up_and_hold():
    t = PF.tumble_plan(TS, [{"from": 1.0, "to": 1.5, "deg": 360}, {"from": 2.0, "to": 2.5, "deg": -90, "axis": "cartwheel"},
                            {"from": 3.0, "to": 3.4, "deg": 180}])
    assert t["flip"][int(1.5 * 120)] == pytest.approx(2 * math.pi) and t["flip"][-1] == pytest.approx(3 * math.pi)
    assert t["cartwheel"][-1] == pytest.approx(-math.pi / 2) and t["flip"][0] == 0.0
    with pytest.raises(ValueError, match="before the flip before it ends"):
        PF.tumble_plan(TS, [{"from": 1.0, "to": 2.0, "deg": 90}, {"from": 1.5, "to": 2.5, "deg": 90}])
    with pytest.raises(ValueError, match="axis"):
        PF.tumble_plan(TS, [{"from": 1.0, "to": 2.0, "deg": 90, "axis": "barrel"}])


def test_a_twist_rolls_her_over_about_her_own_long_axis_after_a_flip_onto_her_back():
    """Flipped onto her back she lies face up; a twist of 180 turns her over about her long axis (face down, her head
    where it was), and a cartwheel then turns her flat about the vertical whatever the twist did."""
    t = PF.tumble_plan(TS, [{"from": 0.5, "to": 1.0, "deg": -90}, {"from": 1.5, "to": 2.0, "deg": 180, "axis": "twist"},
                            {"from": 2.5, "to": 3.0, "deg": 90, "axis": "cartwheel"}])
    R = PF.tumble_rotations(t)
    front, head = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    at = {s: R[int(s * 120)] for s in (1.2, 2.2, 3.2)}
    assert at[1.2] @ front == pytest.approx([0, 0, 1], abs=1e-9) and at[1.2] @ head == pytest.approx([0, 1, 0], abs=1e-9)
    assert at[2.2] @ front == pytest.approx([0, 0, -1], abs=1e-9) and at[2.2] @ head == pytest.approx([0, 1, 0], abs=1e-9)
    assert at[3.2] @ front == pytest.approx([0, 0, -1], abs=1e-9) and at[3.2] @ head == pytest.approx([1, 0, 0], abs=1e-9)
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3))
