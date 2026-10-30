"""Strings that ring (mkmmd.core.strings): which strings a strum or a picked line sets vibrating, when, from which fret, how
long, and the shape a pressed and a ringing string take; pure maths on the library bass's and guitar's cards."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_bass_layout as BL
from mkmmd.blender.library.props import electric_guitar_layout as GL
from mkmmd.core import fretting as FR
from mkmmd.core import strings as CS
from mkmmd.core import strum as SM


@pytest.fixture(scope="module")
def bass():
    neck, strum = BL.card("bass")["use"]["grip"]
    return neck, strum


@pytest.fixture(scope="module")
def guitar():
    neck, strum = GL.card("guitar")["use"]["grip"]
    return neck, strum


def centres(neck, entry):
    a, c, centre = (np.asarray(entry[k], float) for k in ("along", "across", "center"))
    s_c = float(centre @ a)
    return {s: float((FR.string_point(neck, s, s_c) - centre) @ c) for s in range(1, len(neck["strings"]) + 1)}


def sigma(neck, entry):
    """Half the width of the strings at the strum centre, as the perform stage measures it."""
    cs = centres(neck, entry)
    return abs(cs[1] - cs[len(neck["strings"])]) / 2.0


TL = {"beats": [i * 0.5 for i in range(40)], "downbeats": [i * 2.0 for i in range(10)]}


# ---------------------------------------------------------------- presses
def test_a_finger_holds_its_string_from_its_shapes_landing_until_the_hand_leaves_for_the_next():
    states = [(None, 6, {"index": [4, 0]}, 0.0), (2.8, 6, {"middle": [3, 1]}, 0.12), (6.3, 6, {"middle": [3, 1, "hover"]}, 0.12),
              (9.0, 6, {"little": [4, 3]}, 0.1)]
    pr = CS.presses(states, 4)
    assert [(p["string"], p["fret"]) for p in pr] == [(4, 6), (3, 7), (4, 9)]           # the hovering finger holds nothing
    first, second, third = pr
    assert first["on"] == -math.inf and first["off"] == pytest.approx(2.8 - 0.12)       # until the hand leaves
    assert second["on"] == 2.8 and second["off"] == pytest.approx(6.3 - 0.12)
    assert second["in"] == pytest.approx(0.03) and second["out"] == pytest.approx(0.03)   # a quarter of the 0.12 s moves
    assert third["off"] == math.inf and third["in"] == pytest.approx(0.025)
    v = CS.press_value(second, [2.70, 2.775, 2.8, 6.0, 6.20, 6.3])
    assert v[0] == 0.0 and 0.0 < v[1] < 1.0 and v[2] == v[3] == 1.0 and 0.0 < v[4] < 1.0 and v[5] == 0.0


def test_a_finger_that_stays_on_its_string_and_fret_through_a_change_keeps_holding_it():
    states = [(None, 5, "power", 0.0), (2.0, 5, {"index": [6, 0], "ring": [5, 2]}, 0.12)]
    pr = CS.presses(states, 6)
    held = {(p["string"], p["fret"]): p for p in pr}
    assert set(held) == {(6, 5), (5, 7), (4, 7)}
    assert held[(6, 5)]["off"] == math.inf and held[(5, 7)]["off"] == math.inf        # kept
    assert held[(4, 7)]["off"] == pytest.approx(2.0 - 0.12)                           # the little finger lets go


def test_a_string_sounds_from_the_highest_fret_held_down_on_it():
    pr = [{"string": 5, "fret": 3, "on": 0.0, "off": 9.0, "in": 0.0, "out": 0.0},
          {"string": 5, "fret": 7, "on": 1.0, "off": 2.0, "in": 0.0, "out": 0.0}]
    assert CS.fretted(pr, 5, 0.5) == 3 and CS.fretted(pr, 5, 1.5) == 7 and CS.fretted(pr, 4, 1.5) == 0


# ---------------------------------------------------------------- hits and rings
def test_a_picked_note_sounds_on_its_own_string_as_the_pick_crosses_it(bass):
    neck, entry = bass
    cs = centres(neck, entry)
    picked = [(1.0, cs[4], 0.0018, 4), (1.5, cs[3], 0.0016, 3)]
    strokes = SM.plan({"rhythm": "notes"}, TL, 10.0, picked)
    p = SM.params({"rhythm": "notes"})
    hits = CS.hits(strokes, cs, 0.024, p)
    assert [h[1] for h in hits] == [4, 3]                                              # one string each, its own
    assert hits[0][0] == pytest.approx(1.0 + 0.5 * strokes[0].attack) and hits[1][0] == pytest.approx(1.5 + 0.5 * strokes[1].attack)


def test_a_strum_sets_its_strings_ringing_one_after_another_across_its_attack(guitar):
    neck, entry = guitar
    cs = centres(neck, entry)
    strokes = SM.plan({"rhythm": [1.0, 1.25]}, TL, 10.0)                              # a down stroke on the beat, then up
    p = SM.params({})
    hits = CS.hits(strokes, cs, sigma(neck, entry), p)
    down = sorted((t, s) for t, s, _g in hits if t < 1.2)
    up = sorted((t, s) for t, s, _g in hits if t > 1.2)
    assert [s for _t, s in down] == [6, 5, 4, 3, 2, 1]                                 # low E first on the way down
    assert [s for _t, s in up] == [1, 2, 3, 4, 5, 6]                                   # high e first on the way up
    assert down[0][0] >= 1.0 - 1e-9 and down[-1][0] <= 1.0 + strokes[0].attack + 1e-9
    assert np.all(np.diff([t for t, _s in down]) > 0.005)                              # spread over the attack, not at once


def test_a_power_chord_strum_rings_only_the_strings_it_holds(guitar):
    neck, entry = guitar
    states = [(None, 5, "power", 0.0), (3.0, 5, "E", 0.12)]
    ring = CS.sounding(states, 6)
    assert [s for s in range(1, 7) if ring(s, 1.0)] == [4, 5, 6]                       # the others are muted
    assert [s for s in range(1, 7) if ring(s, 3.5)] == [1, 2, 3, 4, 5, 6]               # open E rings all six
    hits = CS.hits(SM.plan({"rhythm": [1.0]}, TL, 10.0), centres(neck, entry), sigma(neck, entry), SM.params({}), ring)
    assert sorted(s for _t, s, _g in hits) == [4, 5, 6]
    assert CS.sounding([], 6)(1, 0.0)                                                  # no fretting hand: every string open


def test_a_note_rings_until_the_next_one_on_its_string_or_its_finger_lets_go():
    pr = [{"string": 4, "fret": 6, "on": -math.inf, "off": 2.0, "in": 0.0, "out": 0.04}]
    hits = [(0.3, 4, 1.0), (1.0, 3, 1.0), (3.0, 3, 1.0), (3.5, 3, 1.0)]
    poly = CS.rings(pr, hits)
    assert poly[0][:4] == (0.3, 2.0, 4, 6)                                             # until the finger lets go
    assert poly[1][:4] == (1.0, 3.0, 3, 0) and poly[2][:2] == (3.0, 3.5)              # an open string until struck again
    assert poly[3][1] == math.inf                                                      # the last rings out
    mono = CS.rings(pr, hits, mono=True)
    assert mono[0][:2] == (0.3, 1.0) and mono[1][:2] == (1.0, 3.0)                     # a line: the next note mutes it


def test_the_frames_show_the_ring_from_its_fret_decaying_and_damped(bass):
    neck, _entry = bass
    params = CS.ring_params({}, neck["scale"])
    params["shimmer"] = 0.0
    pr = [{"string": 4, "fret": 7, "on": 0.0, "off": 1.0, "in": 0.0, "out": 0.04}]
    rung = CS.rings(pr, [(0.5, 4, 1.0)])
    ts = np.arange(0.0, 1.5, 1 / 30)
    F = CS.frames(rung, pr, ts, params, lambda s, f: CS.fret_u(neck, s, f), 4)
    a = F[4]["amp"]
    assert np.all(a[ts < 0.5] == 0.0) and a[ts >= 0.5][0] == pytest.approx(params["amp"], rel=0.05)
    k = np.where(ts >= 0.5)[0]
    assert np.all(np.diff(a[k][ts[k] < 1.0]) < 0)                                      # decaying
    assert np.all(a[ts > 1.0 + params["damp"]] == 0.0)                                 # damped when the finger lets go
    assert np.all(F[4]["u0"][a > 0] == pytest.approx(CS.fret_u(neck, 4, 7)))
    assert set(F[4]["press"]) == {7} and F[3]["amp"].max() == 0.0                       # only the played string moves
    assert params["amp"] == pytest.approx(CS.AMP * BL.SCALE)


def test_ring_parameters_are_checked():
    assert CS.ring_params({"ring": False}, 0.65) is None
    assert CS.ring_params({"ring": {"amp": 0.003, "decay": 1.5}}, 0.65)["amp"] == 0.003
    with pytest.raises(CS.StringsError):
        CS.ring_params({"ring": {"width": 1}}, 0.65)
    with pytest.raises(CS.StringsError):
        CS.ring_params({"ring": {"decay": 0}}, 0.65)


# ---------------------------------------------------------------- the shape
def test_a_pressed_string_runs_from_the_nut_down_to_the_finger_and_up_to_the_saddle(bass):
    neck, _entry = bass
    for string, fret in ((4, 6), (3, 7), (1, 12)):
        nut, saddle = CS.axis(neck, string)
        target = next(t for t in FR.finger_targets(neck, {"index": [string, 0]}, fret) if t["finger"] == "index")
        along = np.asarray(neck["frame"]["along"], float)
        u_f = CS.u_at(neck, string, float(np.asarray(target["point"]) @ along))       # the finger's place along the neck
        u_k = CS.fret_u(neck, string, fret)
        P = CS.pressed_axis(neck, string, fret, np.array([0.0, u_f, u_k, 1.0]))
        assert P[0] == pytest.approx(nut) and P[3] == pytest.approx(saddle)
        assert P[1] == pytest.approx(np.asarray(target["point"]), abs=1e-6)            # where the pad holds it
        normal = np.asarray(neck["frame"]["normal"], float)
        crown = np.asarray(neck["frets"][fret], float) @ normal + neck["fret_height"]
        r = FR.string_entry(neck, string)["radius"]
        assert float(P[2] @ normal) == pytest.approx(crown + r, abs=1e-9)                # lying on the wire's crown
        open_at = nut + u_k * (saddle - nut)
        assert float((open_at - P[2]) @ normal) > 0.001                                  # pushed down a millimetre or more
        assert CS.pressed_axis(neck, string, fret, np.array([-0.1, 1.2])) == pytest.approx(
            np.array([nut - 0.1 * (saddle - nut), nut + 1.2 * (saddle - nut)]))         # the headstock and tail stay


def test_a_ringing_string_is_a_spindle_widest_midway_along_its_vibrating_length(bass):
    neck, _entry = bass
    nut, saddle = CS.axis(neck, 3)
    d = (saddle - nut) / np.linalg.norm(saddle - nut)
    across = np.asarray(neck["frame"]["across"], float)
    ex = across - (across @ d) * d
    ex /= np.linalg.norm(ex)
    ey = np.cross(d, ex)
    u0 = CS.fret_u(neck, 3, 7)
    us = np.linspace(0.0, 1.0, 41)
    r = 0.0012
    pts = np.array([nut + u * (saddle - nut) + r * e for u in us for e in (ex, ey)])
    off = CS.ring_offsets(neck, 3, pts, u0, 0.004)
    side, up = off[0::2], off[1::2]
    w = np.linalg.norm(side, axis=1)
    assert np.all(w[us < u0 - 1e-9] == 0.0) and w[-1] == pytest.approx(0.0, abs=1e-12)  # still above the fret, at the saddle
    mid = int(np.argmax(w))
    assert us[mid] == pytest.approx(0.5 * (u0 + 1.0), abs=0.03) and w[mid] == pytest.approx(0.004, rel=0.01)
    assert np.abs(side @ d).max() < 1e-12 and np.abs(side @ ex - w).max() < 1e-12      # sideways, square to the string
    assert np.linalg.norm(up, axis=1)[mid] == pytest.approx(CS.ECC * 0.004, rel=0.01)  # and less out of the board


def test_every_named_chord_says_which_open_strings_ring():
    assert set(CS.OPEN) == set(FR.CHORDS)
    for chord, open_ in CS.OPEN.items():
        held = {s for s, _k in FR.CHORDS[chord].values()}
        assert not held & set(open_), chord                                            # an open string is not held
