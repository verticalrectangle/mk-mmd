"""A marionette set free (mkmmd.core.marionette_free) and tails (mkmmd.core.marionette_tails): across the loop's wrap the
pre-roll replays the end and the blades shut ON frame 0, the scissors hand's thread joins her fist only once that hand
is up beside her head, the weightless rise reaches its height on its time still speeding up while the dropped control
falls past her (behind her, never through her), the haul's tugs drop her as they land and the fall lands her in her seat
ON the knot, a tail streams opposite to her motion at its own length and is settled on the first frame it is seen, and
the tables the build cannot play are refused."""
import math

import numpy as np
import pytest

from mkmmd.core import marionette as MR
from mkmmd.core import marionette_free as MF
from mkmmd.core import marionette_tails as MT
from mkmmd.core import marionette_world as MW
from test_marionette_world import RIG

FPS = 60.0
TS = (np.arange(1, 2228) - 361) / FPS            # a 31.12 s clip (1867 frames) at 60 fps, six seconds of pre-roll
LOOP = 1867 / FPS
SPEC = {"prop": "strings", "from": -0.9667, "loop": 31.12, "seat": {"at": "chair:seat", "knees": 14},
        "knot": {"t": 30.15}, "gather": {"t": 30.55, "dur": 0.4}, "scissors": {"prop": "snips", "t": 30.56, "dur": 0.45},
        "snip": {"t": 0.0}, "ascend": {"t": 0.0, "height": 1.2, "by": 1.02},
        "haul": {"t": 25.98, "to": 29.45, "tugs": [26.68, 27.37, 28.07, 28.76]}, "crossed": [[26.05, 29.3]], "tails": {}}
SEAT = {"hip": np.array([0.0, 0.035, 0.515]), "facing": np.array([0.0, -1.0, 0.0]), "floor": 0.0}


@pytest.fixture(scope="module")
def free():
    P = MR.normalize(SPEC, [])
    cross = MR.cross_plan(TS, P, MR.STIRRER["leg"] / 2.0)
    return P, MW.world_plan(TS, P, cross, RIG, {"seat": SEAT, "scissors": {"grip": np.array([-0.4, -0.1, 0.63])}})


def _i(t):
    return int(np.argmin(np.abs(TS - t)))


def test_across_the_wrap_the_pre_roll_replays_the_end_and_the_blades_shut_on_frame_0(free):
    P, W = free
    i0 = _i(0.0)
    end, pre = slice(_i(30.2), len(TS)), slice(_i(30.2 - LOOP), i0)
    assert np.abs(W["hips"][end] - W["hips"][pre]).max() < 1e-6                     # her body: the end, replayed
    for s in "LR":
        assert np.abs(W["hands"][s]["wrist"][end] - W["hands"][s]["wrist"][pre]).max() < 1e-4
    sci = W["free"]["scissors"]
    assert sci["held"][i0 - 1] and sci["held"][-1] and sci["held"][i0]          # in her hand across the wrap
    assert sci["open"][-1] > 0.95 and sci["open"][i0 - 1] > 0.95 and sci["open"][i0] == 0.0   # shut ON frame 0
    seated = W["hips"][_i(30.6)]
    assert seated[2] == pytest.approx(SEAT["hip"][2] + RIG["pivot"][2] - RIG["hips"]["L"][2], abs=0.002)


def test_the_scissors_hands_thread_joins_her_fist_only_once_that_hand_is_up_beside_her_head():
    P = MR.normalize(dict(SPEC, gather={"t": 30.55, "dur": 0.3}, scissors={"prop": "snips", "t": 30.6, "dur": 0.42}), [])
    W = MW.world_plan(TS, P, MR.cross_plan(TS, P, MR.STIRRER["leg"] / 2.0), RIG,
                      {"seat": SEAT, "scissors": {"grip": np.array([-0.4, -0.1, 0.63])}})
    Ps = MF.snapped(P, TS)
    swept = MF._gathered(TS, Ps, MF.moments(Ps), wait=False)["wrist_R"]
    taken = W["free"]["gathered"]["wrist_R"]
    z = W["hands"]["R"]["body"][:, 2]
    end = (TS > 30.5) & (TS < LOOP)
    assert (end & (swept > 0.01) & (z < RIG["shoulders"]["R"][2])).any()   # her fist comes to it while it is at the table
    assert taken[_i(LOOP - 0.05)] == pytest.approx(1.0)
    assert z[end & (taken > 0.01)].min() > RIG["crown"][2] - 0.15           # taken in only with that hand up by her head


def test_she_rises_weightless_to_its_height_on_time_still_speeding_up_and_the_control_whips_up_away(free):
    P, W = free
    i0, i1 = _i(0.0), _i(1.02)
    z = W["hips"][:, 2]
    assert z[i1] - z[i0] == pytest.approx(1.2, abs=0.03)
    v = np.diff(z) * FPS
    assert v[i1 - 1] > v[i1 - 20] > v[i0 + 10] > 0.0                             # faster and faster
    crown = W["root"]["pos"] + np.einsum("nij,j->ni", W["root"]["rot"], RIG["crown"])
    up = W["cross"]["pos"][i0:_i(0.6), 2] - crown[i0:_i(0.6), 2]
    assert up.min() > 0.5                                                         # never down past her face
    assert W["cross"]["pos"][_i(0.15), 2] > W["cross"]["pos"][i0, 2] + 1.0       # out of the picture, upward


def test_the_tugs_drop_her_as_they_land_and_the_fall_lands_her_in_her_seat_on_the_knot(free):
    P, W = free
    v = np.diff(W["hips"][:, 2]) * FPS
    for t in (26.68, 27.37, 28.07, 28.76):
        k = int(np.argmin(v[_i(t - 0.3):_i(t + 0.4)])) + _i(t - 0.3)
        assert t - 0.05 <= TS[k] <= t + 0.15                                      # she drops with it, a little late
    k = _i(30.15)
    assert W["hips"][k, 2] == pytest.approx(W["free"]["H_seat"][2], abs=0.004)
    assert W["hips"][k - 1, 2] > W["hips"][k, 2] + 0.05                           # coming down onto it, fast
    assert 5.0 < W["report"]["landings_mps"]["30.150"] < 10.0


def test_a_tail_streams_opposite_to_her_motion_at_its_length_and_is_settled_on_its_first_frame():
    ts = np.arange(0.0, 1.5, 1.0 / FPS)
    roots = np.zeros((len(ts), 1, 3))
    roots[:, 0, 0] = 2.0 * ts                                                     # she flies along +x at 2 m/s
    roots[:, 0, 2] = 1.0
    out = MT.simulate(ts, roots, [0.75], gravity=0.0, settle=1.5, n=30)
    for i in (0, len(ts) - 1):                                                    # settled from the first frame on
        tip = out[i, 0, -1] - roots[i, 0]
        assert tip[0] < -0.65 and abs(tip[2]) < 0.15
    seg = np.linalg.norm(np.diff(out, axis=2), axis=3).sum(axis=2)
    assert np.abs(seg - 0.75).max() < 0.02
    held = MT.simulate(ts, roots, [0.75], gravity=1.0, n=30,
                       pins=[(0, 0.75, roots[:, 0] + np.array([0.0, 0.0, -0.75]), np.ones(len(ts), bool))])
    assert np.allclose(held[:, 0, -1], roots[:, 0] + np.array([0.0, 0.0, -0.75]), atol=1e-6)


def test_tables_the_build_cannot_play_are_refused():
    assert MR.normalize({"prop": "tails", "tails": {"length": 0.75}}, [])["control"] is False
    with pytest.raises(MR.MarionetteError, match="no `from`"):
        MR.normalize({"prop": "tails", "tails": {}, "sway": [[0.0, 0.1]]}, [])
    base = {"prop": "strings", "from": 1.0, "seat": "chair:seat", "snip": {"t": 2.0}}
    with pytest.raises(MR.MarionetteError, match="one or the other"):
        MR.normalize(dict(base, lie=[[0.0, -90]]), [])
    with pytest.raises(MR.MarionetteError, match="knot"):
        MR.normalize(dict(base, haul={"t": 3.0, "to": 5.0}), [])
    with pytest.raises(MR.MarionetteError, match="before the snip"):
        MR.normalize(dict(base, gather={"t": 1.8, "dur": 0.4}), [])
    with pytest.raises(MR.MarionetteError, match="seat"):
        MR.normalize({"prop": "strings", "from": 1.0, "snip": {"t": 2.0}}, [])
