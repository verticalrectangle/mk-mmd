"""Walking, hopping and tumbling (mkmmd.core.perform.walk_plan, tumble_plan): feet that do not slide while down, steps a
stride apart with the body midway, the treadmill, hops on their parabola with the feet carried, and turns about the
hips that add up."""
import math

import numpy as np
import pytest

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
    assert np.abs(np.diff(p["swing"]["L"])).max() < 0.03                 # no jump on the way (a jump is ~0.12 a sample)


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
