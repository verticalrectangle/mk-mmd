import math

import numpy as np
import pytest

from mkmmd.core import marionette as MR

BEATS = [14.66 + 0.502 * k for k in range(12)]
LEGS = {s: {"thigh": 0.34, "shin": 0.34, "foot": 0.14} for s in "LR"}


def _plan(**kw):
    spec = {"prop": "strings", "from": 13.64}
    spec.update(kw)
    return MR.normalize(spec, BEATS)


def _arc(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def test_thread_is_straight_when_taut_and_keeps_its_length_when_slack():
    A, B = np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.0, 1.0])
    d = float(np.linalg.norm(B - A))
    taut = MR.sag_points(A, B, d * 0.98, 24)
    assert np.allclose(taut[0], A) and np.allclose(taut[-1], B)
    assert _arc(taut) == pytest.approx(d, rel=1e-6)
    slack = MR.sag_points(A, B, d * 1.1, 24)
    assert _arc(slack) == pytest.approx(d * 1.1, rel=0.03)
    mid = 0.5 * (A + B)
    assert slack[12][2] < mid[2]                           # it sags down, not up
    vertical = MR.sag_points(A, np.array([0.0, 0.0, 1.0]), 1.2, 24, side=(1.0, 0.0, 0.0))
    assert _arc(vertical) == pytest.approx(1.2, rel=0.03) and vertical[12][0] > 0.05   # bows to its side


def test_follow_lags_and_settles_on_its_target():
    ts = np.arange(0.0, 3.0, 1 / 60)
    x = MR.follow(ts, MR.step(ts, 1.0), (0.4, 0.3))
    assert abs(x[ts < 1.0]).max() < 1e-9
    assert x[np.searchsorted(ts, 1.03)] < 0.3               # late
    assert x.max() > 1.1                                    # and loose: it swings past
    assert x[-1] == pytest.approx(1.0, abs=0.02)


def test_normalize_refuses_what_the_build_cannot_play():
    with pytest.raises(MR.MarionetteError, match="unknown keys"):
        _plan(wobble=1)
    with pytest.raises(MR.MarionetteError, match="after it is on her"):
        _plan(cuts=[{"t": 13.0, "threads": ["head"]}])
    with pytest.raises(MR.MarionetteError, match="cut already"):
        _plan(cuts=[{"t": 15.0, "threads": ["head"]}, {"t": 16.0, "threads": ["head", "back"]}])
    with pytest.raises(MR.MarionetteError, match="no thread"):
        _plan(cuts=[{"t": 15.0, "threads": ["tail"]}])


def test_beat_tables_swing_both_ways_and_knees_take_turns():
    P = _plan(tilt={"deg": 7, "from": 14.66, "to": 16.6}, knees={"from": 14.66, "to": 16.6, "every": 2})
    vals = [v for _t, v in P["tilt"][1:-1]]
    assert vals == [7.0, -7.0, 7.0, -7.0]
    assert P["tilt"][0][1] == 0.0 and P["tilt"][-1][1] == 0.0
    assert [(round(t, 2), s) for t, s, _h in P["knees"]] == [(14.66, "L"), (15.66, "R")]


def test_the_cross_falls_in_and_the_leg_bar_lifts_the_knee_it_names():
    ts = np.arange(12.0, 17.0, 1 / 60)
    P = _plan(knees=[{"t": 15.0, "side": "R", "height": 0.12}])
    c = MR.cross_plan(ts, P, 0.2)
    assert not c["shown"][ts < 13.64 - P["drop"] - 0.02].any()
    assert c["pos"][0, 2] == pytest.approx(MR.DROP_HEIGHT)
    assert abs(c["pos"][np.searchsorted(ts, 13.64), 2]) < 0.08     # home when the threads land
    i = np.searchsorted(ts, 14.98)                                    # the bar is up just before the beat
    assert c["knee"]["R"][i] > 0.1 and c["knee"]["L"][i] == 0.0
    assert c["leg_roll"][i] < 0.0                                     # its right end up


def test_her_body_is_still_before_the_threads_and_the_cut_head_drops_until_she_takes_it():
    ts = np.arange(12.0, 21.0, 1 / 60)
    P = _plan(tilt={"deg": 7, "from": 14.66, "to": 18.69}, knees={"from": 14.66, "to": 18.69},
              cuts=[{"t": 19.19, "threads": ["head"], "own": [19.45, 20.2]},
                    {"t": 19.69, "threads": ["knee_L", "knee_R"], "give": 0.08}])
    c = MR.cross_plan(ts, P, 0.2)
    b = MR.body_plan(ts, P, c, LEGS)
    before = ts < 13.5
    for k in ("lean", "tilt", "head_tilt", "head_nod", "rise", "side", "dip"):
        assert np.abs(b[k][before]).max() < 1e-6, k
    assert b["rise"].min() >= 0.0 and b["feet"]["L"][:, 2].min() >= 0.0
    nod = np.degrees(b["head_nod"])
    assert nod[np.searchsorted(ts, 19.42)] > 30.0                    # chin to chest
    assert nod[np.searchsorted(ts, 20.6)] < 10.0                     # she lifted it herself
    dip = b["dip"]
    assert dip[np.searchsorted(ts, 19.69):np.searchsorted(ts, 20.0)].max() > 0.06   # the knees give
    assert dip[np.searchsorted(ts, 20.8)] < 0.04                     # and she catches her weight
    assert np.abs(np.degrees(b["head_tilt"][np.searchsorted(ts, 15.0):np.searchsorted(ts, 18.6)])).max() > 1.0


def test_a_cut_piece_on_the_cross_leaves_upward_and_the_one_on_her_lands_on_the_floor():
    A, C, B = np.array([0.0, 0.0, 2.3]), np.array([0.0, -0.05, 1.7]), np.array([0.0, -0.1, 1.0])
    up = MR.cut_upper(A, C, 0.0)
    assert np.allclose(up[0], A) and np.allclose(up[-1], C, atol=1e-6)
    gone = MR.cut_upper(A, C, MR.YANK)
    assert gone[:, 2].min() > A[2] + 0.3
    low0 = MR.cut_lower(B, C, 0.0, (0.0, 0.5, -1.0))
    assert np.allclose(low0[0], B) and np.allclose(low0[-1], C, atol=1e-6)
    late = MR.cut_lower(B, C, 2.0, (0.0, 0.5, -1.0), floor_z=0.0, B_slide=B)
    assert late[:, 2].min() >= 0.0 and late[:, 2].max() < 0.01


def test_a_cut_hand_swings_on_its_arm_into_its_hang():
    ts = np.arange(18.0, 21.0, 1 / 60)
    sh, start, hang = np.array([0.0, 0.0, 1.2]), np.array([-0.1, -0.25, 1.0]), np.array([-0.12, -0.3, 0.88])
    p = MR.hang_path(ts, 18.69, start, sh, hang)
    assert np.allclose(p[ts < 18.69], start)
    assert np.allclose(p[-1], hang, atol=0.005)
    r = np.linalg.norm(p - sh, axis=1)
    assert r.min() > min(np.linalg.norm(start - sh), np.linalg.norm(hang - sh)) - 0.02     # on its arm, not shrinking


def test_the_cross_meshes_and_card_name_every_thread():
    M = MR.cross_meshes()
    assert set(M) == {"cross", "hand", "leg"} and all(len(m.V) for m in M.values())
    card = MR.card("strings", MR.ROLES)
    assert set(card["marionette"]["anchors"]) == set(MR.PARTS)
    assert all(f"strings_thread_{p}" in card["objects"] for p in MR.PARTS)
    assert math.isclose(card["marionette"]["leg_bar"]["half"], MR.LEG_BAR["half"])
