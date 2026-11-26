"""The marionette in the world (mkmmd.core.marionette_world): she lies on the water and touches it on the frame the
keys say, the loop's last frame is the frame before the touch, her hips sink (never rise) when her knee threads are cut
in the air, hops land on their times, let go she lands upright on her point with the control beside her, her hands
stay out of her body and her arms inside a human arm's turn, the world keys are refused out of order."""
import math

import numpy as np
import pytest

from mkmmd.core import armtwist as AT
from mkmmd.core import marionette as MR
from mkmmd.core import marionette_world as MW

FPS = 60.0
TS = (np.arange(1, 1723) - 181) / FPS          # a 25.7 s clip at 60 fps with 3 s of pre-roll: -3.0 .. 25.683
LEGS = {s: {"thigh": 0.40, "shin": 0.40, "foot": 0.14} for s in "LR"}
# a body 1.7 m tall standing at her root (x her left, y behind her, z up): her outline seen from her side
HULL = np.array([[0.05, 0.0], [-0.12, 0.0], [-0.12, 0.08], [0.06, 0.45], [-0.08, 0.5], [0.12, 0.95], [-0.1, 1.0],
                 [0.10, 1.30], [-0.12, 1.30], [0.12, 1.55], [-0.10, 1.55], [0.10, 1.70], [-0.06, 1.72]])
RIG = {"pivot": np.array([0.0, 0.0, 0.92]), "crown": np.array([0.0, 0.0, 1.70]), "hang": np.array([0.0, 0.075, 1.29]),
       "hull": HULL, "shoulders": {"L": np.array([0.16, 0.0, 1.33]), "R": np.array([-0.16, 0.0, 1.33])},
       "clavicles": {"L": np.array([0.03, 0.01, 1.36]), "R": np.array([-0.03, 0.01, 1.36])},
       "hips": {"L": np.array([0.09, 0.0, 0.85]), "R": np.array([-0.09, 0.0, 0.85])},
       "ankles": {"L": np.array([0.09, 0.0, 0.08]), "R": np.array([-0.09, 0.0, 0.08])},
       "arms": {s: {"upper": 0.26, "fore": 0.23} for s in "LR"}, "legs": LEGS,
       "rest": {"upper": np.array([0.24, 0.0, -0.14]), "fore": np.array([0.2, 0.0, -0.12]),
                "dir": np.array([0.08, 0.0, -0.05]), "palm": np.array([-0.5, 0.0, -0.866])},
       "anchors": {p: MR.ANCHORS[p][1] for p in MR.PARTS}, "leg_home": MR.LEG_BAR["home"]}
LOWER = [[-3.02, 0.10], [-0.17, 0.08], [-0.03, -0.06], [0.5, -0.10]]   # lowered, the last drop onto the water at 0
PART = {"prop": "strings", "from": -3.5,
        "lie": [[-3.5, -90], [6.68, -90], [6.98, 0], [9.18, 0], [9.9, 85], [12.08, 85], [20.0, 0], [21.19, 0],
                [21.95, -90]],
        "raise": LOWER + [[6.62, -0.10], [6.78, 0.14], [7.0, 0.0], [9.12, 0.0], [9.75, 1.2], [12.08, 1.25], [19.9, 0.3],
                          [21.19, 0.3], [22.68, 0.10]] + [[t + 25.7, v] for t, v in LOWER if t + 25.7 > 22.7],
        "float": {"from": 0.0, "to": 6.7, "dips": [1.53]}, "kicks": {"from": 3.1, "to": 6.0},
        "hops": {"land": [7.19, 7.58, 7.95], "height": 0.10}, "swim": {"from": 9.6, "to": 12.1, "period": 0.74},
        "sway": [[9.3, 0.0], [9.9, 0.15], [10.6, -0.12], [11.3, 0.1], [12.0, -0.05], [12.6, 0.0]],
        "reach": [{"from": 6.9, "to": 10.4, "at": [1.3, -0.1, 1.6]}],
        "cuts": [{"t": 10.61, "threads": ["knee_L"]}, {"t": 10.99, "threads": ["knee_R"]},
                 {"t": 11.34, "threads": ["wrist_L"]}, {"t": 11.73, "threads": ["wrist_R"]},
                 {"t": 12.08, "threads": ["back"]}],
        "release": {"t": 12.08, "tumble": 360}, "land": {"t": 17.26, "at": [0.0, 0.0, -7.6]}, "yank": {"t": 18.18},
        "knot": {"t": 20.1, "loop": 25.7}}


@pytest.fixture(scope="module")
def part():
    P = MR.normalize(PART, [])
    cross = MR.cross_plan(TS, P, MR.LEG_BAR["half"])
    aims = {"reach": [np.tile([1.3, -0.1, 1.6], (len(TS), 1))], "land": np.tile([0.0, 0.0, -7.6], (len(TS), 1))}
    return P, MW.world_plan(TS, P, cross, RIG, aims)


def _i(t):
    return int(np.argmin(np.abs(TS - t)))


def _over(W, P):
    """(n,) m her lowest point is over the water (her place's surface)."""
    return W["hips"][:, 2] - MW.lowest(W["theta"], HULL, RIG["pivot"]) - MW.float_curve(TS, P["world"]["float"])


def test_lowered_on_her_back_she_touches_the_water_on_frame_0_and_the_last_frame_is_the_one_before(part):
    P, W = part
    h = _over(W, P)
    i0 = _i(0.0)
    assert math.degrees(W["theta"][i0]) == pytest.approx(-90.0, abs=0.5)           # flat on her back
    assert abs(h[i0]) < 0.0015 and 0.006 < h[i0 - 1] < 0.014                       # touches on frame 0, 1 cm before
    assert np.all(np.diff(h[i0 - 60:i0 + 1]) < 1e-6)                               # coming down all the way to it
    assert -0.015 < h[i0:i0 + 60].min() < -0.003                                   # the water gives, a little
    assert abs(h[i0 + 120]) < 0.004                                                # and she rests on it, no hovering
    assert h[-1] == pytest.approx(h[i0 - 1], abs=0.0015)                           # the loop: the last frame is frame -1
    assert np.allclose(W["hips"][-1, :2], W["hips"][i0 - 1, :2], atol=0.003)
    nod = np.degrees(np.abs(W["head_nod"][i0:i0 + 60]))
    assert 1.0 < nod.max() < 3.5                                                   # her head gives a little, no nod


def test_the_seconds_before_0_replay_the_end_so_the_loop_has_no_jump(part):
    P, W = part
    pre, end = slice(_i(-0.5), _i(0.0)), slice(_i(25.2), len(TS))                 # the pre-roll's last half second
    assert np.abs(W["hips"][pre] - W["hips"][end]).max() < 0.001                  # is the end's: the knot's swing too
    assert np.abs(W["theta"][pre] - W["theta"][end]).max() < math.radians(0.05)


def test_lying_on_her_back_the_control_is_past_her_head_and_the_leg_bar_past_her_heels(part):
    P, W = part
    for t in (1.0, 25.5):                                                          # seen from above, no stick over her
        k = _i(t)
        R, pos = W["root"]["rot"][k], W["root"]["pos"][k]
        crown, heel = pos + R @ RIG["crown"], pos + R @ np.array([0.0, 0.0, 0.0])
        assert W["cross"]["pos"][k][1] - crown[1] > P["over"] - 0.02
        assert heel[1] - W["leg"]["pos"][k][1] > MW.LEG_SUPINE - 0.02
    P2 = MR.normalize(dict(PART, over=[[0.5, 0.4], [1.0, 1.5]]), [])                # `over` keyed: held further off
    aims = {"reach": [np.tile([1.3, -0.1, 1.6], (len(TS), 1))], "land": np.tile([0.0, 0.0, -7.6], (len(TS), 1))}
    W2 = MW.world_plan(TS, P2, MR.cross_plan(TS, P2, MR.LEG_BAR["half"]), RIG, aims)
    assert W2["cross"]["pos"][_i(2.0)][1] - W["cross"]["pos"][_i(2.0)][1] == pytest.approx(1.1, abs=0.01)


def test_cut_free_in_the_air_her_knees_let_her_hips_sink_from_her_shoulders(part):
    P, W = part
    a, b = _i(10.55), _i(12.0)
    assert W["aloft"][a] > 0.99 and W["aloft"][b] > 0.99                           # flown, still hung
    assert math.degrees(W["theta"][a]) > 75.0 and math.degrees(W["theta"][b]) < 45.0   # flat, then hanging
    assert W["hips"][b, 2] < W["hips"][a, 2] + 0.02                                # the hips sink, they never rise
    for s in "LR":                                                                 # the cut legs hang down
        assert W["feet"][s][b, 1] < -0.15


def test_hops_land_on_their_times():
    rise, dip, lead = MR.landing_hops(TS, [(7.19, 0.1), (7.58, 0.1), (7.95, 0.1)])
    for t in (7.19, 7.58, 7.95):
        on = int(np.searchsorted(TS, t - 1e-9))                                    # the first frame at or after it
        assert rise[on] < 1e-9 and rise[on - 1] > 0.004 and rise[_i(t - 0.17)] > 0.07   # down on it, up just before
        assert dip[_i(t + 0.04)] > 0.02                                            # into the knees
    assert lead[_i(7.0)] > 0.05                                                    # the bars ahead of her


def test_let_go_she_lands_upright_on_her_point_and_the_control_after_her_beside_her(part):
    P, W = part
    i = _i(17.26)
    assert W["hips"][i] == pytest.approx([0.0, 0.0, -7.6 + RIG["pivot"][2]], abs=0.01)
    assert abs(math.remainder(W["theta"][i], 2.0 * math.pi)) < math.radians(4.0)    # on her feet
    assert W["dip"][_i(17.4)] > 0.05                                               # her knees give
    j = _i(17.26 + 0.9)
    eye = W["cross"]["pos"][j] + W["cross"]["rot"][j] @ np.asarray(RIG["anchors"]["head"])
    centre = np.array([0.0, 0.0, -7.6 - 0.4])
    assert np.linalg.norm(eye - centre) == pytest.approx(0.4 + 0.012, abs=0.03)    # on the ball, beside her
    k = _i(18.18) - 1
    assert W["hips"][_i(18.78), 2] > W["hips"][k, 2] + 1.0                        # yanked away by her head thread


def test_her_hands_stay_out_of_her_body_and_her_arms_turn_as_arms_do(part):
    P, W = part
    S = RIG["shoulders"]
    for s, sg in (("L", 1.0), ("R", -1.0)):
        H = W["hands"][s]
        w = H["body"]
        torso = (w[:, 2] > 0.88) & (w[:, 2] < 1.33) & (np.abs(w[:, 1]) < 0.11) & (sg * w[:, 0] < 0.13)
        assert not torso.any()
        R, pos = W["root"]["rot"], W["root"]["pos"]
        d = np.einsum("nji,nj->ni", R, H["dir"])
        p = np.einsum("nji,nj->ni", R, H["palm"])
        E = H["elbow"]
        clip = TS >= 0.0
        fore, upper = [], []
        for i in np.where(clip)[0][::6]:
            f, u = AT.twists(RIG["rest"], s, S[s], E[i], w[i], d[i], p[i])
            fore.append(f)
            upper.append(u)
        assert min(fore) > AT.FOREARM[0] + 10 and max(fore) < AT.FOREARM[1] - 10
        assert min(upper) > AT.UPPER[0] + 5 and max(upper) < AT.UPPER[1] + 30


def test_knotted_on_every_thread_holds_her_again(part):
    P, W = part
    for th in MR.PARTS:
        h = MR.held(TS, P, th)
        assert h[_i(19.0)] == (1.0 if th == "head" else 0.0) and h[_i(20.2)] == 1.0


def test_world_keys_out_of_order_are_refused():
    base = {"prop": "strings", "from": 1.0}
    with pytest.raises(MR.MarionetteError, match="from a fall"):
        MR.normalize(dict(base, land={"t": 5.0, "at": [0, 0, 0]}), [])
    with pytest.raises(MR.MarionetteError, match="after `land`"):
        MR.normalize(dict(base, release={"t": 3.0}, land={"t": 5.0, "at": [0, 0, 0]}, yank={"t": 4.0}), [])
    with pytest.raises(MR.MarionetteError, match="knotted on again after"):
        MR.normalize(dict(base, release={"t": 3.0}, land={"t": 5.0, "at": [0, 0, 0]}, yank={"t": 6.0},
                          knot={"t": 6.5}), [])
    with pytest.raises(MR.MarionetteError, match="give one of them"):
        MR.normalize(dict(base, release={"t": 3.0}, throw={"t": 3.1}), [])
    with pytest.raises(MR.MarionetteError, match="one of"):
        MR.normalize(dict(base, arms="hips"), [])


def test_the_stirrers_control_ties_every_thread_to_its_stick():
    lay = MR.layout("stirrers")
    M = MR.cross_meshes("stirrers")
    card = MR.card("strings", MR.ROLES, "stirrers")
    assert set(card["marionette"]["anchors"]) == set(MR.PARTS) and card["marionette"]["control"] == "stirrers"
    half = {"cross": MR.STIRRER["length"] / 2.0, "hand": MR.STIRRER["length"] / 2.0, "leg": MR.STIRRER["leg"] / 2.0}
    for p, (bar, pt) in lay["anchors"].items():
        mesh = M[bar].V
        assert abs(pt[0]) <= half[bar] and abs(pt[1]) <= half[bar]          # on its stick
        assert np.min(np.linalg.norm(mesh - np.asarray(pt), axis=1)) < 0.01  # where its knot is
    assert "strings_rest" in card["objects"] and "strings_grab" in card["objects"]
    with pytest.raises(MR.MarionetteError):
        MR.layout("spoons")
