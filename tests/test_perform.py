import shutil

import numpy as np
import pytest

from mkmmd.core import lipsync as LS
from mkmmd.core import perform as PF
from mkmmd.core import timeline as TL


def test_gaze_weights_rise_hold_return_and_never_exceed_one():
    g = PF.GazeEvents([(1.0, 2.0, 2.5, 1, 0.2), (1.9, 3.0, 3.4, 2, 0.2)])
    ts = np.array([0.5, 1.3, 1.95, 2.4, 3.6])
    W = g.weights(ts)
    assert W[:, 0].sum() == 0 and W[:, -1].sum() == 0
    assert W[0, 1] == pytest.approx(1.0)
    assert W.sum(0).max() <= 1.0 + 1e-9                       # overlapping events share the gaze
    assert W[1, 3] > W[0, 3]                                  # the later event takes over


def test_glance_weights_are_the_micro_eye_lift_of_the_reference_video():
    ts = np.linspace(0.0, 4.0, 401)
    W = PF.glance_weights(ts, [(1.0, 0.75, 0.12, 0.25)])
    assert W.shape == (1, 401) and W[0, 0] == 0 and W[0, -1] == 0
    assert W[0, 150] == pytest.approx(1.0)                                # inside the hold (t = 1.5)
    t = 1.9                                                               # in the fall: the reference's own formula
    ref = PF.smooth((t - 1.0) / 0.12) * (1.0 - PF.smooth((t - 1.0 - 0.75) / 0.25))
    assert W[0, 190] == pytest.approx(ref)
    assert 0 < W[0, 190] < 1


def test_glance_weights_that_overlap_share_the_eyes():
    ts = np.linspace(0.0, 3.0, 301)
    W = PF.glance_weights(ts, [(0.5, 1.0, 0.1, 0.4), (1.2, 1.0, 0.1, 0.4)])
    assert W.sum(0).max() <= 1.0 + 1e-9
    assert PF.glance_weights(ts, []).shape == (0, 301)


def test_blink_schedule_spacing_extras_and_avoid():
    extra = [(2.0, 0.3)]
    b = PF.blink_schedule(0.0, 60.0, per_min=20, seed=1, extra=extra, avoid=[(10.0, 20.0)], min_gap=1.2)
    times = [t for t, _ in b]
    assert (2.0, 0.3) in b
    assert all(not (10.0 <= t <= 20.0) for t, d in b if (t, d) != (2.0, 0.3))
    assert min(np.diff(sorted(times))) >= 1.2 - 1e-9
    assert 12 <= len(b) <= 24                                 # ~20 per minute, minus the avoided 10 s
    assert PF.blink_schedule(0, 60, 20, seed=1) == PF.blink_schedule(0, 60, 20, seed=1)


def test_blink_curve_closes_fully_and_reopens():
    ts = np.linspace(0, 1, 301)
    c = PF.blink_curve(ts, [(0.2, 0.157)])
    assert c.max() == pytest.approx(1.0) and c[0] == 0 and c[-1] == 0


def test_beat_bob_peaks_just_after_beats_with_accent():
    ts = np.linspace(0, 2, 601)
    bob = PF.beat_bob(ts, [0.5, 1.5], deg=2.0, accent=[1.0, 2.0])
    k1, k2 = np.argmax(bob[:300]), 300 + np.argmax(bob[300:])
    assert 0.49 <= ts[k1] <= 0.55 and bob[k2] == pytest.approx(2 * bob[k1], rel=1e-6)


def test_a_tap_lifts_before_the_beat_falls_fastest_onto_it_and_rests_after():
    ts = np.linspace(0.0, 2.0, 2001)
    e = PF.tap(ts, [0.5, 1.5], lift=0.16, accent=[1.0, 1.5])
    k = int(np.argmax(e[:1000]))
    assert 0.34 < ts[k] < 0.5 and e[k] == pytest.approx(1.0, abs=1e-3)       # up before the beat, not after it
    assert e[500] == pytest.approx(0.0, abs=1e-9) and e[501:1339].max() == 0.0   # down on the beat, rests to the next lift
    speed = np.abs(np.diff(e[:501]))
    assert int(np.argmax(speed)) >= 495                                    # the strike is the fastest moment
    assert e[1000:].max() == pytest.approx(1.5, abs=2e-3)                   # an accented beat lifts higher


def test_timeline_reads_both_layouts():
    assert TL.beats({"beats": [1, 2], "downbeats": [1]}) == ([1, 2], [1])
    assert TL.beats({"tempo": {"beats": [0.5], "downbeats": [0.5]}}) == ([0.5], [0.5])


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_lipsync_closes_for_m_and_between_distant_words():
    tl = {"fps": 30, "lines": [{"words": [{"text": "mama", "start": 1.0, "voiced_end": 1.6}]},
                               {"words": [{"text": "papa", "start": 3.0, "voiced_end": 3.5}]}]}
    k = LS.keyframes(tl, mouth=0.8)
    a = k["a"]
    assert max(v for _, v in a) > 0.4                         # open vowels
    gap = [v for t, v in a if 1.62 < t < 2.97]                # rest keys 0.06 s after / 0.05 s before the words
    assert len(gap) == 2 and max(gap) == 0.0
    k1 = LS.keyframes(tl, lines=[1, 1])
    assert max(t for t, _ in k1["a"]) < 2.0                    # only the first line


WORDS = {"fps": 30, "lines": [{"words": [{"text": f"w{i}", "start": 1.0 + i, "voiced_end": 1.5 + i} for i in range(3)]},
                              {"words": [{"text": f"x{i}", "start": 5.0 + i, "voiced_end": 5.5 + i} for i in range(2)]}]}


def test_chosen_words_are_picked_by_line_and_word_from_either_end_in_time_order():
    pick = LS.select_words(WORDS, words=[[2, -1], [1, 2], [1, -2]])          # [1, 2] and [1, -2] are the same word
    assert [w["start"] for w in pick] == [2.0, 6.0]
    assert [w["start"] for w in LS.select_words(WORDS, words=[[1, -1], [2, -1]])] == [3.0, 6.0]   # the line endings


@pytest.mark.parametrize("kw, frag", [
    (dict(words=[[3, 1]]), "no line 3"),
    (dict(words=[[1, 4]]), "line 1 has 3 words"),
    (dict(words=[[1, 0]]), "line 1 has 3 words"),
    (dict(words=[[1, -4]]), "line 1 has 3 words"),
    (dict(words=[2]), "is not \\[line, word\\]"),
    (dict(words=[[1, 1]], lines=[1, 1]), "not both"),
])
def test_a_word_choice_the_timeline_does_not_have_is_refused(kw, frag):
    with pytest.raises(ValueError, match=frag):
        LS.select_words(WORDS, **kw)


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_lipsync_of_chosen_words_moves_the_mouth_only_around_them():
    k = LS.keyframes(WORDS, words=[[2, -1]], mouth=0.9)
    keys = [(t, v) for vs in k.values() for t, v in vs]
    assert all(5.8 < t < 6.7 for t, _ in keys)                               # the word spans 6.0 to 6.5
    assert max(v for _, v in keys) > 0.3


def test_head_aims_inside_its_range_and_leaves_a_look_in_range_alone():
    f, u = (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)                  # a model facing -Y, Z up
    yaw, elev = PF.yaw_elevation(f, u, (1.0, 0.0, 0.0))        # straight to her left
    assert yaw == pytest.approx(np.pi / 2) and elev == pytest.approx(0.0)
    assert PF.clamp_look(f, u, (1.0, 0.0, 0.0), 0.7) is None   # 0.7 x 90 = 63 deg < 75: the look is used as it is
    yaw, elev = PF.yaw_elevation(f, u, PF.clamp_look(f, u, (0.0, -0.2, 1.0), 0.7))   # a sign almost overhead
    assert yaw == pytest.approx(0.0, abs=1e-9) and 0.7 * elev == pytest.approx(np.radians(35.0))
    yaw, elev = PF.yaw_elevation(f, u, PF.clamp_look(f, u, (-1.0, 1.0, 0.0), 0.7, {"yaw": 60}))   # behind, right
    assert 0.7 * yaw == pytest.approx(np.radians(-60.0)) and elev == pytest.approx(0.0, abs=1e-9)
    assert PF.clamp_look(f, u, (0.0, -0.2, 1.0), 0.0) is None  # a head that does not turn needs no limit


def test_yaw_elevation_does_not_depend_on_how_the_head_is_turned():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=3), rng.normal(size=3)
    q, _ = np.linalg.qr(np.stack([a, b, np.cross(a, b)], 1))
    q *= np.sign(np.linalg.det(q))                              # a rotation, not a reflection
    f, u, d = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.3, -0.8, 0.5])
    assert PF.yaw_elevation(q @ f, q @ u, q @ d) == pytest.approx(PF.yaw_elevation(f, u, d))


def test_rock_is_a_periodic_sine_of_clip_time_toward_the_models_left():
    ts = np.arange(-2.0, 31.0, 1 / 30)
    r = PF.rock(ts, period=4.0, deg=3.0)
    assert r.max() == pytest.approx(np.radians(3.0), abs=1e-3) and r.min() == pytest.approx(-np.radians(3.0), abs=1e-3)
    assert PF.rock(ts + 4.0, 4.0, 3.0) == pytest.approx(r)                      # the same sway one period later: any cut works
    assert PF.rock(np.array([1.0]), 4.0, 3.0)[0] == pytest.approx(np.radians(3.0))     # a quarter period in: the peak, + = her left
    assert PF.rock(ts, 4.0, 3.0, phase=np.pi) == pytest.approx(-r)              # the phase turns the lean around
    assert not PF.rock(ts, 4.0, 0.0).any()                                      # no amplitude, no motion


FEET = {"L": (0.09, 0.03), "R": (-0.09, 0.03)}         # ankles on the floor about the point turned about (x left, y back)


def _turned(p, deg):
    a = np.radians(deg)
    return np.cos(a) * p[0] - np.sin(a) * p[1], np.sin(a) * p[0] + np.cos(a) * p[1]


def test_a_turn_on_the_spot_steps_one_foot_then_the_other_into_the_turned_stance():
    """Rin's 145 degree turn to the boombox and back spun her on a turntable in four frames: a turn steps the feet round
    one at a time (one always down, and a foot down never slides, even as the body travels) into the turned and moved
    stance and back, never further than TWIST_MAX from the hips and never onto each other (her ankles stand behind the
    point she turns about: stepping with the inside foot first set it down where the other still stood), and the chest
    leads the hips, by at most CHEST_MAX, in each direction."""
    ts = np.arange(5.5, 7.2, 1 / 120)
    go, back = [0.0, 0.03], [0.0, -0.03]
    plan = PF.turn_plan(ts, [{"t": 5.86, "deg": 145, "move": go}, {"t": 6.52, "deg": -145, "move": back}], FEET)
    yaw, feet, lift = plan["yaw"], plan["feet"], plan["lift"]
    i = int(np.searchsorted(ts, 6.4))                                            # between the turns
    assert np.degrees(yaw[i]) == pytest.approx(145.0) and plan["shift"][i] == pytest.approx(go)
    for s in "LR":
        assert feet[s][i] == pytest.approx(np.add(_turned(FEET[s], 145.0), go))
        assert feet[s][-1] == pytest.approx(FEET[s]) and yaw[-1] == pytest.approx(0.0, abs=1e-12)
        moved = np.linalg.norm(np.diff(feet[s], axis=0), axis=1) > 1e-12
        up = (lift[s][1:] > 0) | (lift[s][:-1] > 0)
        assert moved.any() and not (moved & ~up).any()                           # it moves only off the ground
        assert np.abs(plan["foot_yaw"][s] - yaw).max() <= PF.TWIST_MAX + 1e-9
    assert not ((lift["L"] > 0) & (lift["R"] > 0)).any()
    assert plan["gap"] == pytest.approx(np.linalg.norm(feet["L"] - feet["R"], axis=1).min()) and plan["gap"] > 0.08
    there, home = (ts > 5.86) & (ts < 6.3), (ts > 6.52) & (ts < 7.0)
    assert plan["chest"][there].max() > np.radians(10) and plan["chest"][home].min() < -np.radians(10)
    assert np.abs(plan["chest"]).max() <= PF.CHEST_MAX and plan["chest"][i] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="starts before the turn before it ends"):
        PF.turn_plan(ts, [{"t": 5.86, "deg": 145}, {"t": 6.0, "deg": -145}], FEET)


TWIRL = {"from": 18.25, "to": 20.95, "deg": 360}         # Reisen's twirl in the air, line 7
TOES = {"L": (0.01, -0.14, 0.07), "R": (-0.01, -0.14, 0.07)}   # toe points 14 cm ahead of the ankles, 7 cm under them


def _twirl(**kw):
    ts = np.arange(17.8, 21.4, 0.01)
    cur = PF.spin_curves(ts, dict(TWIRL, **kw))
    return ts, {k: np.degrees(cur[k]) for k in ("hips", "chest", "head", "eye")}, cur["phases"]


def test_a_spin_winds_up_turns_chest_first_hips_last_and_settles_from_an_overshoot():
    """Reisen's twirl in the air turned her root once round on a turntable: a spin winds up the other way first, the
    chest setting off before the hips both ways, turns with the chest ahead of the hips (the spine twisting by the
    speed), overshoots and settles back by its end; the head lands on the turn before the chest and the chest before
    the hips. Nothing moves before it and all of it ends where it turned to."""
    ts, a, ph = _twirl()
    hips, chest, head = a["hips"], a["chest"], a["head"]
    before, after = ts < TWIRL["from"], ts >= TWIRL["to"]
    assert not any(a[k][before].any() for k in a)
    assert all(a[k][after] == pytest.approx(360.0, abs=1e-6) for k in ("hips", "chest", "head"))
    assert hips.min() == pytest.approx(-PF.SPIN["windup"], abs=0.01) and chest.min() < -0.8 * PF.SPIN["windup"]
    off = [int(np.argmax(np.abs(v) > 1.0)) for v in (chest, hips)]
    assert off[0] < off[1] and chest[off[0]] < 0 and hips[off[1]] < 0           # the other way, the chest first
    back = [int(np.argmax(np.diff(v) > 0)) for v in (chest, hips)]               # and turning back the chest first
    assert back[0] < back[1]
    turning = (hips > 30) & (hips < 330)
    assert (chest[turning] > hips[turning]).all() and (chest - hips).max() > 20.0
    assert hips.max() == pytest.approx(360.0 + PF.SPIN["overshoot"], abs=0.01) and chest.max() > 365.0
    assert ts[np.argmax(chest)] < ts[np.argmax(hips)] < TWIRL["to"] - 0.3        # each overshoots, then settles back
    home = [ts[np.argmax(v >= 359.0)] for v in (head, chest, hips)]
    assert home[0] < home[1] < home[2]


def test_the_head_spots_where_she_faced_then_whips_round_ahead_of_the_chest_and_the_eyes_lead_it():
    """Like a dancer's: the head holds where she faced (the lens) while the body winds up and turns under it, until the
    neck has turned well round, then whips round in `whip` seconds, past the chest and ahead of it as far as the neck
    lets it, lands on the spot again before the chest comes round, and holds it while the chest and hips arrive under
    it; the eyes lead the head by EYE_LEAD."""
    ts, a, ph = _twirl()
    head, chest = a["head"], a["chest"]
    (whip,) = ph["whips"]
    held = ts < whip
    assert np.abs(head[held]).max() < 2.0 and (chest - head)[held].max() > 55.0   # on the spot, the neck turned round
    free = chest - head < PF.SPIN["neck"] - np.degrees(PF.SPIN_SOFT)
    assert np.abs(head[held & free]).max() < 1e-9                                # exactly, until the neck nears its end
    during = (ts >= whip) & (ts <= whip + PF.SPIN["whip"])
    assert np.ptp(head[during]) > 180.0                                          # a whip, not a drift
    ahead = (ts > whip + PF.SPIN["whip"]) & (head < 359.0)
    assert ahead.sum() > 20 and (head[ahead] - chest[ahead]).min() > 50.0       # leading the chest
    landed = ts >= ts[np.argmax(head >= 360.0 - 1e-6)]
    assert chest[landed][0] < 330.0 and np.abs(head[landed] - 360.0).max() < 1e-6
    lead = int(round(PF.EYE_LEAD / 0.01))
    assert a["eye"][:-lead] == pytest.approx(head[lead:], abs=1e-6)
    _, b, ph2 = _twirl(spot=False)                                               # without spotting it just leads
    turning = (b["chest"] > 30) & (b["chest"] < 330)
    assert ph2["whips"] == [] and (b["head"][turning] > b["chest"][turning]).all()


@pytest.mark.parametrize("spin", [TWIRL, {"from": 0.0, "to": 1.0, "deg": 360}, {"from": 0.0, "to": 3.0, "deg": -720},
                                  {"from": 0.0, "to": 1.2, "deg": 180, "twist": 50, "neck": 80, "windup": 40},
                                  {"from": 0.0, "to": 1.0, "deg": 90, "spot": False}])
def test_a_spin_keeps_the_spine_and_the_neck_inside_their_twist_ranges(spin):
    """joint_limits holds a spine to 60 degrees of twist and a neck to 85: a spin's chest leads its hips by `twist` at
    most and its head turns from the chest by `neck` at most, fast or slow, one turn or two, either way; a spin of two
    turns spots twice."""
    ts = np.arange(spin["from"] - 0.2, spin["to"] + 0.2, 1 / 240)
    cur = PF.spin_curves(ts, spin)
    twist, neck = (np.degrees(np.abs(cur[a] - cur[b])).max() for a, b in (("chest", "hips"), ("head", "chest")))
    assert twist <= spin.get("twist", PF.SPIN["twist"]) + 1e-6 < 60.0 and neck <= spin.get("neck", PF.SPIN["neck"]) + 1e-6 < 85.0
    assert np.degrees(cur["hips"][-1]) == pytest.approx(spin["deg"])
    assert len(cur["phases"]["whips"]) == (2 if abs(spin["deg"]) == 720 else 0 if spin.get("spot") is False else 1)
    if spin["deg"] < 0:                                                          # the same spin, mirrored
        assert -cur["head"] == pytest.approx(PF.spin_curves(ts, dict(spin, deg=-spin["deg"]))["head"])


def test_without_a_spin_nothing_turns_and_a_spin_gives_the_head_back_to_the_gaze_around_it():
    """No spin, nothing: the plan of a body that does not turn keys no turn and leaves the head to the gaze. A spin takes
    the head from the gaze only around its own span (over SPIN_OWN on either side), its look turned from the hips only
    inside it; bad spins are refused by name."""
    ts = np.arange(17.0, 22.5, 1 / 30)
    still = PF.turn_plan(ts, [], FEET)
    assert still["spins"] == [] and not any(still[k].any() for k in ("yaw", "chest", "look", "eye", "own", "hips", "dip"))
    plan = PF.turn_plan(ts, [], FEET, [TWIRL], np.ones(len(ts), bool))
    t0, t1 = TWIRL["from"], TWIRL["to"]
    assert (plan["own"][(ts >= t0) & (ts <= t1)] == 1.0).all()
    assert not plan["own"][(ts < t0 - PF.SPIN_OWN) | (ts > t1 + PF.SPIN_OWN)].any()
    assert not plan["look"][(ts < t0) | (ts > t1)].any() and np.degrees(plan["yaw"][-1]) == pytest.approx(360.0)
    (rep,) = plan["spins"]
    assert rep["air"] and rep["foot"] is None and rep["twist_deg"] <= PF.SPIN["twist"] and rep["neck_deg"] <= PF.SPIN["neck"]
    for s in "LR":                                                               # in the air the legs go round with her
        assert plan["feet"][s] == pytest.approx(np.stack(_turned(FEET[s], np.degrees(plan["yaw"])), 1))
        assert not plan["lift"][s].any()
    for bad, frag in [({"from": 18.0, "to": 20.0}, "with from, to and deg"), (dict(TWIRL, whirl=1), "with from, to"),
                      (dict(TWIRL, deg=0), "deg not 0"), (dict(TWIRL, twist=70), "twist within"),
                      (dict(TWIRL, spot="yes"), "spot is true or false"), (dict(TWIRL, to=18.6), "too short")]:
        with pytest.raises(ValueError, match=frag):
            PF.turn_plan(ts, [], FEET, [bad], np.ones(len(ts), bool))
    with pytest.raises(ValueError, match="starts before the turn before it ends"):
        PF.turn_plan(ts, [{"t": 18.0, "deg": 90, "dur": 0.5}], FEET, [TWIRL], np.ones(len(ts), bool))
    with pytest.raises(ValueError, match="partly in the air"):
        PF.turn_plan(ts, [], FEET, [TWIRL], ts > 19.0)
    with pytest.raises(ValueError, match="no foot on the ground"):
        PF.turn_plan(ts, [], FEET, [dict(TWIRL, foot="L")], np.ones(len(ts), bool))


def test_on_the_ground_she_spins_on_the_toes_of_one_foot_the_other_lifted_and_comes_down_in_her_stance():
    """On the ground a spin turns about the toe point of the foot on her turning side: the weight goes over it as she
    winds up (both feet flat and still), the heel rises as the hips set off (the body with it) and the other foot lifts
    and goes round with her, never moving while it is down; both come down in her stance as she ends, and the toe
    point turned on never moves. After a step turn the spin goes on from where it left her."""
    ts = np.arange(0.5, 3.6, 1 / 120)
    spin = {"from": 1.0, "to": 3.2, "deg": 360}
    plan = PF.turn_plan(ts, [], FEET, [spin], np.zeros(len(ts), bool), TOES)
    (rep,) = plan["spins"]
    assert rep["foot"] == "L" and not rep["air"]
    toe = np.add(FEET["L"], TOES["L"][:2])
    pivot = plan["shift"] + np.stack(_turned(toe, np.degrees(plan["yaw"])), 1)
    assert np.ptp(pivot, axis=0).max() < 1e-9                                    # the toe point she turns on stays put
    windup = (ts > spin["from"]) & (plan["yaw"] < 0) & (np.diff(plan["yaw"], prepend=0.0) < 0)
    assert windup.sum() > 20 and not plan["lift"]["L"][windup].any() and not plan["lift"]["R"][windup].any()
    assert all(np.ptp(plan["feet"][s][windup], axis=0).max() < 1e-9 for s in "LR")   # both flat and still
    turning = (np.degrees(plan["yaw"]) > 30) & (np.degrees(plan["yaw"]) < 330)
    assert plan["lift"]["L"][turning].min() > 0.03 and plan["pitch"]["L"][turning] == pytest.approx(PF.RELEVE)
    assert plan["dip"][turning] == pytest.approx(-plan["lift"]["L"][turning])   # the body rises with the heel
    assert plan["lift"]["R"][turning] == pytest.approx(PF.SPIN["lift"])
    assert plan["hips"][turning] == pytest.approx(np.tile(toe, (turning.sum(), 1)))   # the hips over the toes she turns on
    moved = np.linalg.norm(np.diff(plan["feet"]["R"], axis=0), axis=1) > 1e-9
    up = (plan["lift"]["R"][1:] > 0) | (plan["lift"]["R"][:-1] > 0)
    assert moved.any() and not (moved & ~up).any()                              # the other foot moves only off the floor
    end = ts >= spin["to"]
    for s in "LR":
        assert plan["feet"][s][end] == pytest.approx(np.tile(FEET[s], (end.sum(), 1)), abs=1e-9)
        assert not plan["lift"][s][end].any() and not plan["hips"][end].any()
    right = PF.turn_plan(ts, [], FEET, [dict(spin, deg=-360)], np.zeros(len(ts), bool), TOES)
    assert right["spins"][0]["foot"] == "R" and right["lift"]["L"].max() == pytest.approx(PF.SPIN["lift"])
    both = PF.turn_plan(ts, [{"t": 0.55, "deg": 90}], FEET, [spin], np.zeros(len(ts), bool), TOES)
    assert np.degrees(both["yaw"][-1]) == pytest.approx(450.0)
    pivot = both["shift"] + np.stack(_turned(toe, np.degrees(both["yaw"])), 1)
    assert np.ptp(pivot[ts >= spin["from"]], axis=0).max() < 1e-9               # turning on her toes where she now stands


def test_aloft_is_inside_an_air_span_or_risen_off_the_ground():
    ts = np.arange(0.0, 10.0, 0.5)
    up = PF.aloft(ts, {"from": 1.0, "to": 2.0}, [[5.0, 0.0], [6.0, 0.4], [8.0, 0.4], [9.0, 0.0]])
    assert list(ts[up]) == [1.0, 1.5, 2.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    assert not PF.aloft(ts).any() and PF.aloft(ts, [{"from": 0.0, "to": 9.5}]).all()


def test_in_the_air_the_legs_tuck_one_after_the_other_hold_still_and_reach_down_to_land():
    """Blown up by the blast, Len hung in the air with straight legs: in the air the feet come up and back, the first
    ahead of the other, the toes pointed; the pose holds still in between (a freeze shows one frame of it while the world
    stops, so it must not have moved on when the world starts again) and the feet are under the hips at the landing."""
    ts = np.arange(9.3, 10.6, 1 / 120)
    off, pitch, kicks = PF.air_plan(ts, [{"from": 9.392, "to": 10.455}])
    assert kicks == []
    at = lambda t: int(np.argmin(np.abs(ts - t)))                              # noqa: E731
    for s in "LR":
        assert not off[s][: at(9.39)].any() and not off[s][at(10.455) + 1:].any() and not pitch[s][at(10.455) + 1:].any()
    mid = at(9.9)
    assert off["L"][mid] == pytest.approx([0.0, PF.AIR["back"], PF.AIR["tuck"]])
    assert off["R"][mid] == pytest.approx([0.0, PF.AIR["share"] * PF.AIR["back"], PF.AIR["share"] * PF.AIR["tuck"]])
    assert pitch["L"][mid] == pytest.approx(np.radians(PF.AIR["point"]))
    early = at(9.392 + 0.07)
    assert off["R"][early][2] < 0.5 * PF.AIR["share"] * off["L"][early][2]     # the second foot follows
    hold = (ts > 9.392 + PF.AIR["rise"] + PF.AIR["lag"]) & (ts < 10.455 - PF.AIR["extend"])
    assert hold.sum() > 30 and all(np.ptp(off[s][hold], axis=0).max() == 0 for s in "LR")
    with pytest.raises(ValueError, match="from before to"):
        PF.air_plan(ts, [{"from": 10.0, "to": 9.5}])


LEGS = {s: {"hip": (x, -0.035, 0.68), "thigh": 0.34, "shin": 0.345} for s, x in (("L", 0.08), ("R", -0.08))}


def _leg(off, pitch, s, legs=LEGS):
    """(thigh forward of straight down, knee bend, the foot's pitch against the shin) over time, rad, of a leg whose
    ankle the air offsets move (from its place under the hip joint `legs[s]["hip"]`)."""
    hip, lg = legs[s]["hip"], legs[s]
    ang = np.array([PF.leg_angles((hip[1] - o[1], o[2] - hip[2]), lg["thigh"], lg["shin"]) for o in off[s]])
    return ang[:, 0], ang[:, 1], pitch[s] - (ang[:, 1] - ang[:, 0])


def test_a_leg_s_angles_and_its_ankle_are_each_other_s_inverse():
    for phi, knee in ((0.0, 0.0), (0.4, 1.2), (-0.2, 0.3), (0.5, 2.0)):
        f, u = PF.leg_ankle(phi, knee, 0.34, 0.345)
        assert PF.leg_angles((f, u), 0.34, 0.345) == pytest.approx((phi, knee), abs=1e-9)
    assert PF.leg_angles((0.0, -0.9), 0.34, 0.345) == pytest.approx((0.0, 0.0))          # out of reach: straight


def test_floating_legs_kick_in_turns_about_their_tuck_from_the_tuck_until_they_reach_down_to_land():
    """Floating weightless through four lines of a song, a girl held her tucked legs still for twelve seconds, a stiff
    pose: `kicks` bends and straightens each knee in turn about its tuck, the thigh swinging with it and the foot pointing
    as the knee straightens and flexing as it bends; they ease in once both legs are tucked and out before the legs
    reach down to land, and without `kicks` the plan is what it was."""
    ts = np.arange(0.0, 14.0, 1 / 60)
    span = {"from": 1.0, "to": 13.0, "tuck": 0.12, "back": 0.06, "rise": 0.6, "point": 35}
    still, still_pitch, none = PF.air_plan(ts, [span], LEGS)
    off, pitch, (rep,) = PF.air_plan(ts, [dict(span, kicks={"wander": 0})], LEGS)
    assert none == [] and PF.air_plan(ts, [dict(span, kicks=False)], LEGS)[2] == []
    t0, t1 = 1.0 + 0.6 + PF.AIR["lag"], 13.0 - PF.AIR["extend"]
    assert (rep["from"], rep["to"]) == (pytest.approx(t0, abs=1e-3), pytest.approx(t1, abs=1e-3))
    calm = (ts <= t0) | (ts >= t1)
    for s in "LR":                                                   # the tuck, the take-off and the landing untouched
        assert np.array_equal(off[s][calm], still[s][calm]) and np.array_equal(pitch[s][calm], still_pitch[s][calm])
    full = (ts > t0 + PF.KICKS["ease"]) & (ts < t1 - PF.KICKS["ease"])
    moved = {s: [a - b for a, b in zip(_leg(off, pitch, s), _leg(still, still_pitch, s))] for s in "LR"}
    for s in "LR":
        hip, knee, foot = (np.degrees(v) for v in moved[s])
        assert knee[full].max() == pytest.approx(PF.KICKS["knee"], abs=0.2)           # bends ...
        assert knee[full].min() == pytest.approx(-PF.KICKS["knee"], abs=0.2)          # ... and straightens
        assert np.abs(hip[full]).max() == pytest.approx(PF.KICKS["hip"], abs=0.2)
        assert np.abs(foot[full]).max() == pytest.approx(PF.KICKS["foot"], abs=0.2)
        assert np.corrcoef(knee[full], foot[full])[0, 1] < -0.7          # pointing as it straightens
        for edge in ((ts > t0) & (ts < t0 + 0.1), (ts > t1 - 0.1) & (ts < t1)):        # eased in and out
            assert np.abs(knee[edge]).max() < 0.5 and np.abs(foot[edge]).max() < 0.5
    assert np.corrcoef(moved["L"][1][full], moved["R"][1][full])[0, 1] < -0.99           # in turns
    assert rep["on"] == "period" and rep["cycle_s"] == PF.KICKS["period"]
    assert all(n >= 3 for n in rep["kicks"].values()) and rep["landings"]["L"][0] < rep["landings"]["R"][0]


def test_floating_kicks_land_on_every_other_beat_from_the_downbeat_in_turns():
    beats = [0.25 + 0.75 * k for k in range(20)]
    lands = PF.kick_beats(beats, beats[3::4], 2)                     # bars of four from 2.5 s: beats 1 and 3
    assert lands[:4] == pytest.approx([1.0, 2.5, 4.0, 5.5])
    ts = np.arange(0.0, 14.0, 1 / 60)
    span = {"from": 1.0, "to": 13.0, "tuck": 0.12, "back": 0.06, "rise": 0.6, "first": "R"}
    still, still_pitch, _ = PF.air_plan(ts, [span], LEGS)
    off, pitch, (rep,) = PF.air_plan(ts, [dict(span, kicks={"beats": lands, "wander": 0})], LEGS)
    on = [b for b in lands if 1.66 <= b <= 12.78]
    assert rep["on"] == "beats" and rep["landings"]["R"] == pytest.approx(on[0::2], abs=0.02)   # the first foot first
    assert rep["landings"]["L"] == pytest.approx(on[1::2], abs=0.02)
    knee = _leg(off, pitch, "R")[1] - _leg(still, still_pitch, "R")[1]
    for b in on[2:-2:2]:                                             # straightest on its beats
        near = np.abs(ts - b) < 0.5
        assert ts[near][np.argmin(knee[near])] == pytest.approx(b, abs=1 / 60)
    with pytest.raises(ValueError, match="fewer than two beats"):
        PF.air_plan(ts, [dict(span, kicks={"beats": [5.0]})], LEGS)
    with pytest.raises(ValueError, match="needs the beats"):
        PF.air_plan(ts, [dict(span, kicks={"every": 2})], LEGS)


def test_floating_kicks_wander_seeded_and_never_lock_a_knee():
    """Not a metronome: each kick comes at its own pace and size, the same for a seed; from a tuck too straight for its
    kick (no tuck at all) the knee bends further on average instead of straightening past KICK_KNEE_MIN."""
    ts = np.arange(0.0, 20.0, 1 / 60)
    span = {"from": 1.0, "to": 19.0, "tuck": 0.12, "back": 0.06, "rise": 0.6}
    _, _, (a,) = PF.air_plan(ts, [dict(span, kicks=True)], LEGS)
    _, _, (b,) = PF.air_plan(ts, [dict(span, kicks=True)], LEGS)
    _, _, (c,) = PF.air_plan(ts, [dict(span, kicks={"seed": 4})], LEGS)
    assert a == b and a["landings"] != c["landings"]
    gaps = np.diff(a["landings"]["L"])
    assert gaps.std() > 0.05 and abs(gaps.mean() - PF.KICKS["period"]) < 0.5
    assert a["knee_deg"]["L"][1] - a["knee_deg"]["L"][0] > 2 * PF.KICKS["knee"] * 1.05    # some kicks bigger
    straight = {s: dict(LEGS[s], hip=(LEGS[s]["hip"][0], 0.0, 0.684)) for s in "LR"}      # a bend of 6 degrees
    off, pitch, (rep,) = PF.air_plan(ts, [dict(span, tuck=0.0, back=0.0, kicks=True)], straight)
    knee = np.degrees(_leg(off, pitch, "L", straight)[1])
    full = (ts > rep["from"] + PF.KICKS["ease"]) & (ts < rep["to"] - PF.KICKS["ease"])
    assert knee[full].min() >= np.degrees(PF.KICK_KNEE_MIN) - 1e-6 and knee.min() >= knee[0] - 1e-6
    assert knee[full].mean() > knee[0] + 5.0                         # bent further on average
    with pytest.raises(ValueError, match="legs"):
        PF.air_plan(ts, [dict(span, kicks=True)])
    with pytest.raises(ValueError, match="kicks"):
        PF.air_plan(ts, [dict(span, kicks={"amount": 1})], LEGS)
    with pytest.raises(ValueError, match="wander within"):
        PF.air_plan(ts, [dict(span, kicks={"wander": 0.9})], LEGS)


# ---------------------------------------------------------------- groove: a body standing in time
G_BEATS = [0.5 * k for k in range(1, 33)]                 # 120 BPM


def test_a_groove_is_one_change_of_leg_and_the_rest_of_her_hangs_on_it():
    """Her weight goes to the other leg on every second beat, arriving on it and running a little past; settled on a leg
    her hips are over it with that hip up, her chest leans back the other way and only the free heel is off the floor.
    Outside its window she stands as posed."""
    ts = np.arange(0.0, 16.5, 1 / 60)
    g = PF.groove(ts, G_BEATS, None, spine=0.45, every=2, seed=3)
    at = lambda x, t: float(np.interp(t, ts, x))            # noqa: E731
    land = G_BEATS[0::2]
    on = [at(g["w"], t + 0.12) for t in land[1:-1]]
    assert all(a * b < 0 for a, b in zip(on, on[1:])) and min(abs(v) for v in on) > 0.85     # a leg, then the other
    assert max(abs(v) for v in on) > 1.0                                                   # a little past it
    for t in (b + 0.3 for b in land[2:-2]):
        s = np.sign(at(g["w"], t))                                                         # +: on her left leg
        free, stand = ("R", "L") if s > 0 else ("L", "R")
        assert at(g["side"], t) * s > 0.015 and at(g["tilt"], t) * s < 0 and at(g["roll"], t) * s < 0
        assert at(g["heel"][free], t) > np.radians(4.0) and at(g["heel"][stand], t) == pytest.approx(0.0, abs=1e-9)
    w = PF.groove(ts, G_BEATS, None, **{"from": 4.0, "to": 8.0})
    assert not w["dip"][ts < 3.5].any() and not w["side"][ts > 8.5].any() and w["win"].max() == pytest.approx(1.0)
    assert w["soft"].max() == pytest.approx(np.radians(PF.GROOVE["soft"]))


def test_a_groove_lets_her_down_onto_each_beat_under_where_she_stands_deepest_where_her_weight_lands():
    ts = np.arange(0.0, 16.5, 1 / 240)
    d = PF.groove(ts, G_BEATS, None, every=2, dip=0.02, vary=0)["dip"]
    assert d.min() >= 0.0 and np.abs(np.diff(d, 2)).max() < 1e-4            # never above her stance; a wave, no corner
    for k in range(2, len(G_BEATS) - 2):
        near = (ts > G_BEATS[k] - 0.2) & (ts < G_BEATS[k] + 0.2)
        assert ts[near][np.argmax(d[near])] == pytest.approx(G_BEATS[k], abs=0.006)      # lowest on the beat
        assert d[near].max() == pytest.approx(0.02 * (1.0 if k % 2 == 0 else PF.GROOVE["light"]), rel=0.01)
        assert d[(ts > G_BEATS[k] + 0.28) & (ts < G_BEATS[k] + 0.29)].max() < 1e-4       # up again between beats


def test_a_leg_reaches_farthest_as_it_was_modelled_and_less_the_more_its_knee_bends():
    hip, knee, ankle = [0.08, -0.04, 0.78], [0.055, -0.03, 0.44], [0.063, 0.0, 0.10]      # a little knock-kneed, as models are
    bends, reach = PF.leg_reach(hip, knee, ankle)
    limbs = np.linalg.norm(np.subtract(knee, hip)) + np.linalg.norm(np.subtract(ankle, knee))
    assert reach[0] == pytest.approx(np.linalg.norm(np.subtract(ankle, hip))) and reach[0] < limbs - 1e-4
    assert (np.diff(reach) <= 0).all() and reach[-1] < 0.6 * reach[0] and bends[-1] == pytest.approx(np.radians(150.0))
    straight = PF.leg_reach([0, 0, 0.8], [0, 0, 0.4], [0, 0, 0.0])[1]
    assert straight[0] == pytest.approx(0.8) and straight[60] == pytest.approx(0.8 * np.cos(np.radians(15.0)))
