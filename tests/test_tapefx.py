"""Cassette type, the numbers (mkmmd/core/tapefx.py): the slide, the wow and flutter, the rewind and the row breaks.
Synthetic numbers only."""
import math

import pytest

from mkmmd.core import tapefx as TF


def slide_p(**over):
    p = {k: v for k, v in TF.SLIDE_DEFAULTS.items() if k != "from"}
    p.update(side=1.0)
    p.update(over)
    return p


# ------------------------------------------------------------------------------------------------------ slide
def test_slide_side_alternates_or_is_fixed():
    assert [TF.slide_side("alt", k) for k in range(4)] == [1.0, -1.0, 1.0, -1.0]
    assert TF.slide_side("left", 3) == -1.0 and TF.slide_side("right", 0) == 1.0
    with pytest.raises(ValueError, match="'left', 'right' or 'alt'"):
        TF.slide_side("up", 0)


def test_a_strip_starts_off_to_its_side_tilted_and_arrives_straight_at_its_slot():
    p = slide_p(dist=0.5, tilt=4.0, dur=0.16)
    c = TF.slide_state(0.0, p)
    assert c["dxp"] == pytest.approx(0.5) and c["rot"] == pytest.approx(4.0)
    c = TF.slide_state(0.16, p)
    assert c["dxp"] == pytest.approx(0.0) and c["rot"] == pytest.approx(0.0)
    left = TF.slide_state(0.0, slide_p(side=-1.0))
    assert left["dxp"] == pytest.approx(-0.5) and left["rot"] == pytest.approx(-4.0)


def test_the_slide_decelerates():
    p = slide_p()
    xs = [TF.slide_state(t, p)["dxp"] for t in (0.0, 0.04, 0.08, 0.12, 0.16)]
    steps = [a - b for a, b in zip(xs, xs[1:])]
    assert all(s > 0 for s in steps) and steps == sorted(steps, reverse=True)          # always moving in, ever more slowly


def test_it_clunks_home_with_a_squash_of_its_length_that_dies_away():
    p = slide_p(clunk=0.07)
    assert "sx" not in TF.slide_state(0.1, p)                                          # still travelling: nothing to squash
    c = TF.slide_state(0.16 + 1e-9, p)
    assert c["sx"] == pytest.approx(1.0 - 0.07, abs=1e-6) and c["sy"] > 1.0
    later = TF.slide_state(0.16 + 0.3, p)
    assert abs(later["sx"] - 1.0) < 1e-3 and abs(later["sy"] - 1.0) < 1e-3
    assert TF.slide_state(5.0, p)["dxp"] == 0.0


# ------------------------------------------------------------------------------------------------------ wow
def test_a_note_wows_only_when_it_is_held_long_enough():
    p = dict(TF.WOW_DEFAULTS)
    assert not TF.wow_held(1.0, 1.29, p) and TF.wow_held(1.0, 1.3, p)


def test_wow_fades_in_with_the_note_and_out_after_it():
    p = dict(TF.WOW_DEFAULTS)
    zero = {"dx": 0.0, "dy": 0.0, "rot": 0.0, "sx": 0.0}
    assert TF.wow_at(0.9, 1.0, 2.0, p) == zero and TF.wow_at(1.0, 1.0, 2.0, p) == zero
    assert TF.wow_at(2.0 + p["tail"], 1.0, 2.0, p) == zero and TF.wow_at(3.0, 1.0, 2.0, p) == zero
    assert any(abs(v) > 0 for v in TF.wow_at(1.5, 1.0, 2.0, p).values())


def test_wow_stays_inside_its_amounts_and_phase_keeps_words_apart():
    p = dict(TF.WOW_DEFAULTS)
    dx_max = p["amount"] + p["flutter"] * 1.5
    seen = [TF.wow_at(1.0 + i / 60.0, 1.0, 3.0, p, phase=0.7) for i in range(120)]
    assert max(abs(s["dx"]) for s in seen) <= dx_max + 1e-12
    assert max(abs(s["dy"]) for s in seen) <= 0.8 * p["flutter"] + 1e-12
    assert max(abs(s["rot"]) for s in seen) <= p["tilt"] + 1e-12 and max(abs(s["sx"]) for s in seen) <= p["pitch"] + 1e-12
    assert TF.wow_at(1.7, 1.0, 3.0, p, phase=0.0) != TF.wow_at(1.7, 1.0, 3.0, p, phase=2.0)


def test_wow_is_the_slow_drift_plus_the_fast_flutter():
    p = dict(TF.WOW_DEFAULTS, flutter=0.0, pitch=0.0, tilt=0.0)
    for i in range(10):
        t = 1.5 + i * 0.01
        got = TF.wow_at(t, 1.0, 3.0, p)
        assert got["dx"] == pytest.approx(p["amount"] * math.sin(2 * math.pi * p["rate"] * (t - 1.0)))
        assert got["dy"] == 0.0 and got["rot"] == 0.0 and got["sx"] == 0.0


# ------------------------------------------------------------------------------------------------------ rewind
def test_rewind_ranks_set_who_leaves_first():
    assert TF.rewind_ranks("first_first", 4) == [0, 1, 2, 3]
    assert TF.rewind_ranks("last_first", 4) == [3, 2, 1, 0]
    assert TF.rewind_ranks("none", 3) == [0, 0, 0]
    with pytest.raises(ValueError, match="rewind order"):
        TF.rewind_ranks("sideways", 2)


def test_a_word_shoots_back_smeared_and_fades_out():
    p = dict(TF.REWIND_DEFAULTS)
    c0, c1 = TF.rewind_state(0.0, p), TF.rewind_state(1.0, p)
    assert c0 == {"dxp": 0.0, "sx": 1.0, "alpha": 1.0}
    assert c1["dxp"] == pytest.approx(-p["dist"]) and c1["sx"] == pytest.approx(1.0 + p["stretch"]) and c1["alpha"] == 0.0
    dxs = [TF.rewind_state(u / 10.0, p)["dxp"] for u in range(11)]
    assert dxs == sorted(dxs, reverse=True) and all(b - a <= 1e-12 for a, b in zip(dxs, dxs[1:]))   # only ever further back
    steps = [a - b for a, b in zip(dxs, dxs[1:])]
    assert steps == sorted(steps)                                                                   # ease-in: faster and faster
    assert TF.rewind_state(-1.0, p) == c0 and TF.rewind_state(2.0, p) == c1


def test_alpha_holds_for_the_first_half_of_a_rewind():
    p = dict(TF.REWIND_DEFAULTS)
    assert TF.rewind_state(0.5, p)["alpha"] == 1.0 and 0.0 < TF.rewind_state(0.75, p)["alpha"] < 1.0


# ------------------------------------------------------------------------------------------------------ rows
def test_wrap_rows_balances_the_rows():
    assert TF.wrap_rows([1, 1, 1, 1], 2) == [0, 0, 1, 1]
    assert TF.wrap_rows([1, 1, 1, 1, 1, 1, 1], 2) in ([0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1])
    assert TF.wrap_rows([3, 1, 1, 1], 2) == [0, 1, 1, 1]                      # the long word has a row to itself
    assert TF.wrap_rows([1, 1, 1, 3], 2) == [0, 0, 0, 1]


def test_wrap_rows_edge_cases():
    assert TF.wrap_rows([2, 1], 1) == [0, 0]
    assert TF.wrap_rows([2, 1, 3], 9) == [0, 1, 2]                            # never more rows than words
    assert TF.wrap_rows([2], 3) == [0]
    assert TF.wrap_rows([1, 1, 1, 1], 0) == [0, 0, 0, 0]


def test_wrap_rows_counts_the_gaps_between_words():
    widths = [1, 1, 1, 1, 4]
    assert TF.wrap_rows(widths, 2, gap=0.0) == [0, 0, 0, 0, 1]             # 4 | 4
    assert TF.wrap_rows(widths, 2, gap=1.0) == [0, 0, 0, 1, 1]             # 5 | 6 beats 7 | 4
    rows = TF.wrap_rows([1, 1, 1, 1, 1, 1], 3)
    assert rows == [0, 0, 1, 1, 2, 2]
