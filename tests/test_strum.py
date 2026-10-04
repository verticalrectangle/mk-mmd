"""Strumming maths (mkmmd.core.strum): which strokes a rhythm makes, which way each goes, and where the pick tip is over time."""
import math

import numpy as np
import pytest

from mkmmd.core import strum as SM

BEAT = 0.4723
BEATS = [round(i * BEAT, 4) for i in range(40)]
DOWNBEATS = BEATS[::4]
TL = {"beats": BEATS, "downbeats": DOWNBEATS}
SIGMA = 0.0225


def eighths(t0=2 * BEAT, n=16, shift=0.0):
    return [round(t0 + i * BEAT / 2 + shift, 4) for i in range(n)]


# ------------------------------------------------------------------------------------------------ rhythm and strokes
def test_parse_rhythm_forms_and_errors():
    assert SM.parse_rhythm("onsets:other") == ("onsets", "other")
    assert SM.parse_rhythm("beats:8") == ("beats", 8)
    assert SM.parse_rhythm([1.0, 2.5]) == ("times", [1.0, 2.5])
    for bad in ("onsets:", "beats:7", "beats:", "sixteenths", "beats:0"):
        with pytest.raises(SM.StrumError):
            SM.parse_rhythm(bad)


def test_beat_grid_makes_the_subdivisions_inside_the_window():
    t, sub = SM.beat_grid(BEATS, 2, BEATS[2], BEATS[6])
    assert len(t) == 9 and t[0] == pytest.approx(BEATS[2]) and t[-1] == pytest.approx(BEATS[6])
    assert sub.tolist() == [0, 1] * 4 + [0]
    assert np.diff(t) == pytest.approx(BEAT / 2, abs=1e-3)
    with pytest.raises(SM.StrumError):
        SM.beat_grid([1.0], 2)


def test_nearest_beat_gives_the_signed_distance_in_beats():
    k, ph = SM.nearest_beat(BEATS[5] + 0.1 * BEAT, BEATS)
    assert k == 5 and ph == pytest.approx(0.1, abs=1e-3)
    k, ph = SM.nearest_beat(BEATS[5] - 0.2 * BEAT, BEATS)
    assert k == 5 and ph == pytest.approx(-0.2, abs=1e-3)
    assert SM.nearest_beat(-1.0, BEATS)[0] == 0 and SM.nearest_beat(99.0, BEATS)[0] == len(BEATS) - 1


def test_beats_rhythm_alternates_down_on_the_beat_and_up_between():
    st = SM.plan({"rhythm": "beats:8", "from": BEATS[4], "to": BEATS[8]}, TL)
    assert len(st) == 9
    assert [s.dir for s in st] == [1, -1] * 4 + [1]
    assert st[0].t == pytest.approx(BEATS[4]) and st[1].t == pytest.approx(BEATS[4] + BEAT / 2, abs=1e-3)
    q = SM.plan({"rhythm": "beats:4", "from": BEATS[4], "to": BEATS[8]}, TL)
    assert len(q) == 5 and all(s.dir == 1 for s in q)
    s16 = SM.plan({"rhythm": "beats:16", "from": BEATS[4], "to": BEATS[5]}, TL)
    assert [s.dir for s in s16] == [1, -1, 1, -1, 1]


def test_onsets_take_their_direction_from_the_beat_they_fall_on():
    on = eighths(shift=0.012)                                       # eighths a little late, as a detector sees them
    st = SM.plan({"rhythm": "onsets:other"}, dict(TL, onsets={"other": on}))
    assert [s.t for s in st] == pytest.approx(on)
    assert [s.dir for s in st] == [1, -1] * 8


def test_off_grid_onsets_alternate_and_doubled_onsets_are_one_stroke():
    on = [2.0, 2.0 + 0.236 * 0.5 * 0.0 + 0.04, 2.1 * 1.0 + 0.2, 3.0]               # a ghost 40 ms after the first
    tl = dict(TL, onsets={"other": on})
    st = SM.plan({"rhythm": "onsets:other"}, tl)
    assert len(st) == 3 and st[1].t == pytest.approx(2.3)
    sixteenth = [BEATS[8], BEATS[8] + BEAT * 0.25, BEATS[8] + BEAT * 0.5]                 # on the beat, a sixteenth, the offbeat
    s = SM.plan({"rhythm": "onsets:o"}, dict(TL, onsets={"o": sixteenth}))
    assert [x.dir for x in s] == [1, -1, -1]                                               # the sixteenth alternates; the offbeat is up


def test_window_and_accent_select_and_weight_strikes():
    on = eighths()
    tl = dict(TL, onsets={"other": on})
    st = SM.plan({"rhythm": "onsets:other", "from": on[4] - 0.01, "to": on[11] + 0.01, "accent": "downbeats"}, tl)
    assert [s.t for s in st] == pytest.approx(on[4:12])
    accented = [s for s in st if s.accent]
    assert accented and all(min(abs(s.t - d) for d in DOWNBEATS) <= 0.06 for s in accented)
    g, up = SM.DEFAULTS["accent_gain"], SM.DEFAULTS["up_scale"]
    for s in st:
        assert s.amp == pytest.approx((g if s.accent else 1.0) * (up if s.dir < 0 else 1.0))
    assert not any(s.accent for s in SM.plan({"rhythm": "onsets:other"}, tl))
    assert all(s.accent for s in SM.plan({"rhythm": "beats:4", "accent": "beats"}, TL)[:5])
    assert SM.plan({"rhythm": "times".split()[0] and [1.0, 2.0], "accent": [2.0]}, TL)[1].accent
    with pytest.raises(SM.StrumError, match="accent"):
        SM.plan({"rhythm": "beats:8", "accent": "loudest"}, TL)


def test_plan_reports_what_is_missing():
    with pytest.raises(SM.StrumError, match="onsets"):
        SM.plan({"rhythm": "onsets:other"}, TL)
    with pytest.raises(SM.StrumError, match="at least two beats"):
        SM.plan({"rhythm": "beats:8"}, {})
    assert SM.plan({"rhythm": "onsets:o", "from": 50.0}, dict(TL, onsets={"o": [1.0]})) == []


# ------------------------------------------------------------------------------------------------ the pick's path
def stroke_run(on, **kw):
    st = SM.plan({"rhythm": on, "accent": "downbeats"}, TL)
    ts = np.arange(0.0, max(on) + 2.0, 0.001)
    return st, ts, SM.path(st, ts, SIGMA, **kw)


def at(P, ts, t):
    i = int(round(t / (ts[1] - ts[0])))
    return P["u"][i], P["h"][i]


def test_the_pick_meets_the_first_string_at_the_strike_and_leaves_the_last_after_the_attack():
    on = eighths(n=8)
    st, ts, P = stroke_run(on)
    assert SM.DEFAULTS["attack"] == pytest.approx(0.075)
    for s in st:
        u, h = at(P, ts, s.t)
        assert u == pytest.approx(-s.dir * SIGMA, abs=4e-4)                     # on the first string (the low E on a down stroke)
        dep = SM.DEFAULTS["depth"] * (SM.DEFAULTS["accent_gain"] if s.accent else 1.0) * (SM.DEFAULTS["up_scale"] if s.dir < 0 else 1.0)
        assert h == pytest.approx(-dep, abs=1e-4)                                # pressed into the strings
        u2, h2 = at(P, ts, s.t + s.attack)
        assert u2 == pytest.approx(s.dir * SIGMA, abs=4e-4)                      # on the last
        mid = (ts >= s.t + 0.002) & (ts <= s.t + s.attack - 0.002)
        assert P["contact"][mid].all() and np.allclose(P["h"][mid], -dep, atol=1e-6)
        v = np.diff(P["u"][mid]) / 0.001
        assert v == pytest.approx(s.dir * 2 * SIGMA / s.attack, rel=0.02)         # constant speed across the strings: the strum's spread


def test_down_strokes_move_down_and_up_strokes_up_and_alternating_ones_are_continuous():
    on = eighths(n=12)
    st, ts, P = stroke_run(on)
    for s in st:
        i, j = int(round(s.t / 0.001)), int(round((s.t + s.attack) / 0.001))
        assert np.sign(P["u"][j] - P["u"][i]) == s.dir
    run = (ts >= on[0] - 0.05) & (ts <= on[-1] + 0.2)
    assert np.abs(np.diff(P["u"][run])).max() < 0.0012                           # <= 1.2 m/s everywhere (no jumps: alternating strokes share their turn)
    assert np.abs(np.diff(P["h"][run])).max() < 0.0012
    assert np.abs(P["u"]).max() == pytest.approx(0.5 * 0.09 * SM.DEFAULTS["accent_gain"], abs=1e-3)   # the accented stroke's far end


def test_the_travel_is_the_span_and_the_ends_are_lifted_off_the_strings():
    on = eighths(n=6)
    st, ts, P = stroke_run(on, span=0.10)
    plain = [s for s in st if not s.accent and s.dir > 0][0]
    i = int(round((plain.t - plain.lead) / 0.001))
    assert abs(P["u"][i]) == pytest.approx(0.05, abs=2e-3) and P["h"][i] == pytest.approx(SM.DEFAULTS["lift"], abs=5e-4)
    assert (P["h"][P["stroke"] >= 0] <= SM.DEFAULTS["lift"] + 1e-9).all()


def test_the_hand_rests_outside_its_window_and_comes_back_to_rest():
    on = eighths(n=4)
    st, ts, P = stroke_run(on)
    win = SM.window(st)
    assert win[0] == pytest.approx(on[0] - SM.DEFAULTS["lead"] - SM.DEFAULTS["approach"], abs=1e-6)
    before, after = ts < win[0] - 1e-9, ts > win[1] + 1e-9
    assert (P["u"][before] == 0).all() and (P["h"][before] == 0).all() and (P["u"][after] == 0).all()
    assert abs(at(P, ts, win[1])[0]) < 1e-3 and abs(at(P, ts, win[1])[1]) < 1e-3
    assert abs(at(P, ts, on[0] - SM.DEFAULTS["lead"] - SM.DEFAULTS["approach"] + 0.001)[0]) < 5e-4         # eased in from rest
    assert SM.window([]) is None
    assert (SM.path([], ts, SIGMA)["u"] == 0).all()


def test_a_pause_sends_the_hand_back_to_rest_and_a_short_gap_does_not():
    a, b = eighths(n=4), eighths(t0=10.0, n=4)
    st, ts, P = stroke_run(a + b)
    gap = (ts > a[-1] + 0.6) & (ts < b[0] - 0.7)
    assert (P["u"][gap] == 0).all() and (P["h"][gap] == 0).all()                  # at rest in between
    close, ts2, P2 = stroke_run(a + eighths(t0=a[-1] + 0.9, n=4))
    mid = (ts2 > a[-1] + 0.3) & (ts2 < a[-1] + 0.6)
    assert np.abs(P2["u"][mid]).max() > 0                                          # still moving across the gap
    assert np.abs(np.diff(P2["u"])).max() < 0.0012


def test_a_stroke_in_the_same_direction_returns_over_the_strings_lifted():
    st = [SM.Stroke(2.0, 1), SM.Stroke(2.24, 1)]                                   # two down strokes in a row
    ts = np.arange(1.0, 3.5, 0.001)
    P = SM.path(st, ts, SIGMA)
    ret = (ts > 2.0 + st[0].attack + st[0].follow) & (ts < 2.24 - st[1].lead)
    assert ret.any() and P["h"][ret].max() > SM.DEFAULTS["return_lift"] * 0.9       # the lift over the strings on the way back
    assert (P["stroke"][ret] == -1).all()
    assert np.abs(np.diff(P["u"])).max() < 0.0035


def test_amplitude_never_drops_the_ends_inside_the_strings():
    weak = SM.Stroke(2.0, 1, amp=0.1)
    P = SM.path([weak], np.arange(1.0, 3.0, 0.001), SIGMA)
    assert np.abs(P["u"]).max() >= SIGMA + 0.007


def test_accent_strokes_are_wider_and_deeper_than_plain_ones():
    on = eighths(n=8)
    st, ts, P = stroke_run(on)
    acc = [s for s in st if s.accent][0]
    plain = [s for s in st if not s.accent and s.dir == acc.dir][0]
    assert abs(at(P, ts, acc.t - acc.lead)[0]) > abs(at(P, ts, plain.t - plain.lead)[0]) + 0.01
    assert at(P, ts, acc.t)[1] < at(P, ts, plain.t)[1]
    with_depth = SM.plan({"rhythm": "beats:4", "depth": 0.006}, TL)
    assert all(s.depth == 0.006 for s in with_depth)
    P6 = SM.path(with_depth[:2], np.arange(0.0, 3.0, 0.001), SIGMA)
    assert P6["h"].min() == pytest.approx(-0.006, abs=1e-4)


def test_strike_times_are_the_first_string_contact_of_each_direction():
    st = SM.plan({"rhythm": "beats:8", "from": BEATS[4], "to": BEATS[6]}, TL)
    down, up = SM.strike_times(st)
    assert down == pytest.approx([BEATS[4], BEATS[5], BEATS[6]], abs=1e-3) and len(up) == 2
