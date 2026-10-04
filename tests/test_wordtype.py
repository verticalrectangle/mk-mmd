"""Lyric type (mkmmd/core/wordtype.py): word selection, landing frames, the keyed motion, layouts and the guarantee that
no lyric text leaves the expansion except in the one `text` field of each word's spec. The timelines here are synthetic:
made-up filler words and numbers."""
import ast
import json
import math
import re
from pathlib import Path

import pytest

from mkmmd.core import wordtype as L
from mkmmd.core import typefx as FX

FPS = 30.0
FILL = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll", "mm", "nn", "oo", "pp"]


def timeline(with_db=True):
    """Three lines of filler words: line 1 words 1-4 sit at 1.0-2.6 s, line 2 at 4.0-6.2 s, line 3 at 8.0-9.0 s."""
    spec = [[(1.0, 1.2), (1.25, 1.5), (1.8, 2.1), (2.2, 2.6)],
            [(4.0, 4.3), (4.35, 4.9), (5.0, 5.2), (5.3, 5.6), (5.7, 6.2)],
            [(8.0, 8.5), (8.55, 9.0)]]
    k = 0
    lines = []
    for ws in spec:
        row = []
        for s, e in ws:
            row.append({"text": FILL[k % len(FILL)] * (1 + k // len(FILL)), "start": s, "end": e, "voiced_end": e,
                        "ctc_start": s, "whisper": "zz"})
            k += 1
        lines.append({"words": row, "start": ws[0][0], "end": ws[-1][1]})
    tl = {"lines": lines, "tempo": {"ticks": [1.5 * i for i in range(1, 8)]}}
    if with_db:
        tl["vocal_db"] = [-12.0 - 10.0 * (0.5 + 0.5 * math.sin(i / 9.0)) for i in range(int(10 * FPS))]
    return tl


def words_of(tl):
    return [w["text"] for ln in tl["lines"] for w in ln["words"]]


def no_text_in(obj, tl, skip=()):
    """True when no filler word of `tl` appears anywhere in the JSON of obj (outside the dict keys in `skip`)."""
    blob = json.dumps(obj, default=str)
    return not any(re.search(r"\b" + re.escape(w) + r"\b", blob) for w in words_of(tl))


def entry(**lyr):
    base = {"timeline": "t.json"}
    if "lines" not in lyr:
        base["line"] = 1
    base.update(lyr)
    return {"name": "m", "at": [0, 0, 0], "facing": [0, -1, 0], "box": [1.0, 0.5], "fit": 0.8, "lyrics": base}


def fake_measure(s):
    return {"w": 0.55 * len(s), "y0": -0.2, "y1": 0.7, "cap": 0.7}


# ------------------------------------------------------------------------------------------------------ selection
def test_select_line_lines_and_words_are_one_based_and_inclusive():
    tl = timeline()
    assert [(w.line, w.index) for w in L.select(tl, line=1)] == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert [(w.line, w.index) for w in L.select(tl, lines=[1, 2], words=[2, 3])] == [(1, 2), (1, 3), (2, 2), (2, 3)]
    assert [(w.line, w.index) for w in L.select(tl, line=3, words=2)] == [(3, 2)]
    ws = L.select(tl, lines=[2, 3])
    assert [w.k for w in ws] == list(range(7)) and ws[0].start == 4.0 and ws[-1].voiced_end == 9.0


@pytest.mark.parametrize("kw, frag", [
    ({"line": 4}, "lines 4..4 are out of range (the timeline has 3 lines)"),
    ({"line": 0}, "out of range"),
    ({"lines": [2, 1]}, "out of range"),
    ({"line": 1, "words": [2, 9]}, "words 2..9 are out of range (line 1 has 4 words)"),
    ({"line": 1, "words": [0, 2]}, "out of range"),
    ({"line": 1, "lines": [1, 2]}, "exactly one of"),
    ({}, "exactly one of"),
    ({"lines": [1]}, "[first, last]"),
    ({"line": 1.5}, "whole number"),
])
def test_select_errors_name_numbers_never_words(kw, frag):
    tl = timeline()
    with pytest.raises(L.LyricsError) as e:
        L.select(tl, **kw)
    assert frag in str(e.value) and no_text_in(str(e.value), tl)


def test_select_reports_a_broken_timeline_without_echoing_its_values():
    tl = timeline()
    tl["lines"][0]["words"][1]["start"] = "bb"                      # a word where a number belongs
    with pytest.raises(L.LyricsError) as e:
        L.select(tl, line=1)
    assert "start of word (1, 2) is not a number" in str(e.value) and no_text_in(str(e.value), tl)
    tl = timeline()
    tl["lines"][1]["words"][0]["text"] = 7
    with pytest.raises(L.LyricsError, match=r"word \(2, 1\) has no text"):
        L.select(tl, line=2)
    with pytest.raises(L.LyricsError, match="no lines"):
        L.select({"lines": []}, line=1)


def test_word_repr_and_str_hold_numbers_only():
    tl = timeline()
    w = L.select(tl, line=1)[2]
    assert repr(w) == str(w) == "Word(1,3)" and w.text == "cc" and w.tag == "l1w3"
    assert no_text_in([repr(x) for x in L.select(tl, lines=[1, 3])], tl)


def test_word_db_is_the_mean_level_over_the_note_and_mid_range_without_a_track():
    vdb = [-30.0] * 30 + [-10.0] * 30 + [-30.0] * 30
    assert L.word_db(vdb, 1.0, 2.0, FPS) == pytest.approx(-10.0)
    assert L.word_db(vdb, 1.0, 1.0, FPS) == pytest.approx(-10.0)      # at least one frame
    assert L.word_db(vdb, 50.0, 51.0, FPS) == L.DB_FLOOR + 0.5 * L.DB_RANGE
    assert [w.db for w in L.select(timeline(with_db=False), line=1)] == [L.DB_FLOOR + 0.5 * L.DB_RANGE] * 4


# ------------------------------------------------------------------------------------------------------ timing
@pytest.mark.parametrize("onset, first, want", [
    (1.381, None, 41),            # floor(41.43 + 0.4)
    (1.38, None, 41),             # floor(41.4 + 0.4) = 41 (the 0.4 rounds a word 0.6 frames or more past a frame up)
    (1.37, None, 41),             # floor(41.1 + 0.4)
    (1.36, None, 41),             # 40.8 + 0.4 = 41.2
    (1.35, None, 40),             # 40.5 + 0.4 = 40.9
    (2.595, 78, 78),              # never before the shot's first frame (floor(78.25) = 78 here, but clamps anyway)
    (2.50, 78, 78),
    (0.0, None, 0),
])
def test_landing_frame_rule(onset, first, want):
    assert L.landing_frame(onset, FPS, first) == want


def test_cut_frame_rounds_like_the_shots_stage():
    # shots.py: int(round(frame0 + t * fps)); a clip frame is that minus frame0 (banker's rounding on ties included)
    for frame0 in (0, 1, 181):
        for t in (0.0, 1.3, 2.6, 16.87, 22.76, 24.6, 0.5 / FPS, 2.5 / FPS):
            assert L.cut_frame(t, FPS, frame0) + frame0 == int(round(frame0 + t * FPS))


def test_exit_tick_is_one_frame_ahead_of_the_first_tick_that_leaves_room():
    ticks = [1.44, 2.98, 4.47, 5.97]
    assert L.exit_tick(ticks, 3.89, 5.0, FPS) == pytest.approx((math.floor(4.47 * FPS + 0.4) - 1) / FPS)
    assert L.exit_tick(ticks, 3.89, 5.0, FPS) == pytest.approx(133 / FPS)
    assert L.exit_tick(ticks, 2.592, 2.6, FPS) is None                  # the cut is too close for a drip
    assert L.exit_tick(ticks, 4.5, 4.8, FPS) is None                    # 4.47 is before the last word's end - 0.03 s
    assert L.exit_tick(ticks, 4.49, 5.0, FPS) == pytest.approx(133 / FPS)    # ... but 0.03 s earlier still counts
    assert L.exit_tick([], 1.0, 9.0, FPS) is None


def test_ticks_are_read_from_either_timeline_layout_and_fall_back_to_beats():
    assert L.ticks_of({"ticks": [2.0, 1.0]}) == [1.0, 2.0]
    assert L.ticks_of({"tempo": {"ticks": [3.0]}}) == [3.0]
    assert L.ticks_of({"beats": [0.5, 1.0]}) == [0.5, 1.0]
    assert L.ticks_of({}) == []


# ------------------------------------------------------------------------------------------------------ motion maths
def test_spread_grows_ease_out_while_the_note_is_held_and_relaxes_after():
    t0, t1 = 1.0, 1.6
    amp = L.spread_max(t0, t1)
    assert 0.07 <= amp <= 0.34
    assert L.spread_amount(t0, t1, 0.9) == 0.0
    assert L.spread_amount(t0, t1, 1.3) == pytest.approx(amp * L.eo3(0.5))
    assert L.spread_amount(t0, t1, t1) == pytest.approx(amp)
    assert L.spread_amount(t0, t1, t1 + 0.13) == pytest.approx(amp * math.exp(-1.0))
    assert L.spread_max(0.0, 0.01) == pytest.approx(0.07 + 0.27 * 0.0)          # a very short note: the minimum
    assert L.spread_max(0.0, 2.0) == pytest.approx(0.34)                         # a long one: the maximum


def test_louder_words_are_bolder_and_the_voice_breathes_the_weight_while_the_note_is_held():
    assert L.level(-34.0) == 0.0 and L.level(-5.0) == 1.0 and L.level(-60.0) == 0.0 and L.level(0.0) == 1.0
    assert L.level(-20.0) > L.level(-30.0)
    vdb = [-30.0] * 90 + [-8.0] * 90
    vs = L.smooth_vocal(vdb)
    quiet = L.Word(1, 1, 0, 1.0, 1.5, 1.5, -30.0)
    assert L.weight_level(quiet, vs, 0.5, FPS) == pytest.approx(L.level(-30.0))      # before the note: its base
    held = L.weight_level(quiet, vs, 4.0 / FPS * 10, FPS)
    assert held == pytest.approx(L.level(-30.0))                                     # the voice is as quiet as the word
    loud_voice = L.Word(1, 1, 0, 3.0, 3.6, 3.6, -30.0)                               # the voice is loud at 3 s
    assert L.weight_level(loud_voice, vs, 3.3, FPS) > L.level(-30.0) + 0.1
    assert L.weight_level(loud_voice, vs, 5.0, FPS) == pytest.approx(L.level(-30.0))  # after the note: back to its base


def test_pop_is_a_damped_spring_that_starts_at_one_plus_amplitude():
    assert L.pop_scale(0.0, 0.3) == pytest.approx(1.3)
    assert L.pop_scale(-1.0) == 1.0
    assert abs(L.pop_scale(1.0, 0.3) - 1.0) < 1e-6
    assert L.pop_scale(0.075 / 2, 0.3, 0.075, 0.22) < 1.3


def test_drop_starts_half_a_cell_above_lands_squashed_and_settles():
    dy, sy, sx = L.drop_state(0.0)
    assert (dy, sy, sx) == (0.5, 1.0, 1.0)
    dy, sy, sx = L.drop_state(0.115)
    assert dy == pytest.approx(0.0) and sy < 1.0 and sx > 1.0                        # the landing: squashed and wide
    dy, sy, sx = L.drop_state(2.0)
    assert dy == pytest.approx(0.0, abs=1e-6) and sy == pytest.approx(1.0, abs=1e-6) and sx == pytest.approx(1.0, abs=1e-6)


def test_drip_letter_is_identity_until_its_time_then_stretches_falls_and_fades():
    a = L.drip_letter(0.0, 0, 1.0, 2600.0, 3.0, 0.5)
    assert a == {"dy": 0.0, "sy": 1.0, "sx": 1.0, "dx": 0.0, "alpha": 1.0}
    assert L.drip_letter(-0.2, 3, 1.0, 2600.0, 3.0, 0.5) == a
    b = L.drip_letter(0.2, 0, 1.0, 2600.0, 3.0, 0.5)
    s = 1 - math.exp(-0.2 / L.DRIP_TAU)
    assert b["sy"] == pytest.approx(1 + 3.0 * s) and b["sx"] == pytest.approx(1 / math.sqrt(b["sy"]))
    assert b["dy"] == pytest.approx(0.5 * 2600 / 150 * 0.04) and 0.0 < b["alpha"] < 1.0
    assert L.drip_letter(0.5, 0, 1.0, 2600.0, 3.0, 0.5)["alpha"] == 0.0              # gone after its life


def test_steam_progress_eases_out_and_arrives_when_the_span_ends():
    assert L.steam_progress(0.0, 1.0) == 0.0
    assert L.steam_progress(1.0, 1.0) == pytest.approx(1.0) and L.steam_progress(3.0, 1.0) == pytest.approx(1.0)
    assert L.steam_progress(0.3, 1.0) > 0.3                                         # ease-out: ahead of linear
    assert L.steam_progress(0.5, 0.0) == 0.0                                        # the last word never moves


def test_compress_keeps_linear_interpolation_exact():
    keys = [(i / FPS, 2.0 * i) for i in range(11)]
    assert L.compress(keys) == [[0.0, 0.0], [10 / FPS, 20.0]]
    keys = [(i / FPS, math.sin(i / 3.0)) for i in range(20)]
    out = L.compress(keys)
    assert out[0][0] == 0.0 and out[-1][0] == 19 / FPS and len(out) == len(keys)     # a curve keeps its keys
    assert L.compress([(0.0, 1.0), (0.1, 1.0), (0.2, 1.0)]) == [[0.0, 1.0]]                  # constant: one key
    assert L.compress([(0.0, 0.0), (0.1, 1.0), (0.2, 1.0), (0.3, 1.0)]) == [[0.0, 0.0], [0.1, 1.0]]    # the last value holds


# ------------------------------------------------------------------------------------------------------ expand
def run(tl, **kw):
    e = kw.pop("entry", None) or entry(**kw.pop("lyr"))
    return L.expand(e, tl, fps=FPS, frame0=kw.pop("frame0", 181), panel=kw.pop("panel", (1.0, 0.5)),
                    measure=kw.pop("measure", fake_measure))


def test_expand_makes_one_text_per_word_named_by_line_and_word():
    tl = timeline()
    specs, summ = run(tl, lyr={"lines": [1, 2], "words": [1, 2], "style": "none"})
    assert [s["name"] for s in specs] == ["m_l1w1", "m_l1w2", "m_l2w1", "m_l2w2"]
    assert [s["lyric"] for s in specs] == [[1, 1], [1, 2], [2, 1], [2, 2]]
    assert [s["text"] for s in specs] == ["aa", "bb", "ee", "ff"]
    assert all("lyrics" not in s and s["fit"] == 0.8 and s["box"] == [1.0, 0.5] and s["facing"] == [0, -1, 0] for s in specs)
    assert summ["words"] == 4 and summ["lines"] == [1, 2] and summ["skipped"] == []


def test_a_word_lands_on_floor_onset_times_fps_plus_point_four_and_is_shown_until_the_cut():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "to": 3.0, "style": "none", "spread": False})
    for s, (start, _) in zip(specs, [(1.0, 0), (1.25, 0), (1.8, 0), (2.2, 0)]):
        on, off = s["kinetic"]["show"]
        assert on * FPS == pytest.approx(math.floor(start * FPS + 0.4))
        assert off * FPS == pytest.approx(90)
    assert summ["first_frame"] == 30 and summ["last_frame"] == 66 and summ["cut_frame"] == 90


def test_a_word_never_lands_before_from_and_to_defaults_to_the_last_words_end():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "from": 1.5, "style": "none", "spread": False})
    assert [round(s["kinetic"]["show"][0] * FPS) for s in specs] == [45, 45, 54, 66]      # the first two wait for `from`
    assert all(round(s["kinetic"]["show"][1] * FPS) == 78 for s in specs)                 # 2.6 s: the last word's end
    assert summ["cut_frame"] == 78


def test_words_that_land_after_the_cut_are_left_out_and_reported_by_number():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "to": 2.0, "style": "none"})
    assert [s["lyric"] for s in specs] == [[1, 1], [1, 2], [1, 3]] and summ["skipped"] == [[1, 4]]
    assert summ["last_frame"] == 54


def test_from_and_to_round_like_the_shots_stage_does():
    tl = timeline()
    e = entry(line=1, to=2.6, style="none")
    for frame0 in (0, 1, 181):
        specs, summ = run(tl, entry=e, frame0=frame0)
        assert summ["cut_frame"] + frame0 == int(round(frame0 + 2.6 * FPS))


def test_accent_cycles_with_the_line_number_or_the_word():
    tl = timeline()
    specs, _ = run(tl, lyr={"lines": [1, 3], "colors": ["love", "gold", "pine"], "style": "none"})
    by_line = {s["lyric"][0]: s["color"] for s in specs}
    assert by_line == {1: "love", 2: "gold", 3: "pine"}
    specs, _ = run(tl, lyr={"lines": [2, 3], "colors": ["love", "gold"], "style": "none"})
    assert {s["lyric"][0]: s["color"] for s in specs} == {2: "gold", 3: "love"}
    specs, _ = run(tl, lyr={"line": 1, "colors": ["love", "gold", "pine"], "color_by": "word", "style": "none"})
    assert [s["color"] for s in specs] == ["love", "gold", "pine", "love"]
    e = entry(line=1, style="none")
    e["color"] = "iris"
    specs, _ = run(tl, entry=e)
    assert {s["color"] for s in specs} == {"iris"}                                  # no colors: the entry's colour
    with pytest.raises(L.LyricsError, match="color_by"):
        run(tl, lyr={"line": 1, "colors": ["love"], "color_by": "nope"})


def test_surfaces_cycle_word_by_word():
    tl = timeline()
    e = entry(line=1, style="none")
    e["on"] = ["a:p0", "a:p1", "a:p2"]
    del e["facing"], e["box"]
    specs, _ = run(tl, entry=e, panel=(1.0, 1.0))
    assert [s["on"] for s in specs] == ["a:p0", "a:p1", "a:p2", "a:p0"]
    e["on"] = "a:single"
    specs, _ = run(tl, entry=e, panel=(1.0, 1.0))
    assert {s["on"] for s in specs} == {"a:single"}


def test_per_word_tilt_cycles():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "tilt": [0, 0, 12]})
    assert [s["kinetic"].get("rot", [[0, 0]])[0][1] for s in specs] == [0, 0, 12, 0]
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "tilt": 5})
    assert all(s["kinetic"]["rot"][0][1] == 5 for s in specs)


def test_pop_style_keys_a_spring_on_the_landing_frame_and_settles_to_rest():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "pop", "arrive": {"a": 0.3, "td": 0.075, "tp": 0.22}, "spread": False})
    k = specs[0]["kinetic"]
    land = round(k["show"][0] * FPS)
    assert land == 30
    assert k["scale"][0][0] == pytest.approx(land / FPS) and k["scale"][0][1] == pytest.approx(1.3)
    assert abs(k["scale"][-1][1] - 1.0) < 0.01 and k["scale"][-1][0] < k["show"][0] + 1.0
    assert "opacity" not in k and "dy" not in k and "tracking" not in k


def test_arrival_alpha_ramps_in_over_a_couple_of_frames_when_asked():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "pop", "arrive": {"alpha0": 0.55, "alpha_frames": 2}, "spread": False})
    op = specs[0]["kinetic"]["opacity"]
    assert op[0][1] == pytest.approx(0.55) and op[-1][1] == pytest.approx(1.0)
    assert op[-1][0] - op[0][0] == pytest.approx(2 / FPS)


def test_spread_is_keyed_tracking_that_follows_the_note_and_leaves_headroom_for_the_fit():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none"})
    w = L.select(tl, line=1)[0]
    k = specs[0]["kinetic"]
    keys = dict(k["tracking"])
    land = round(k["show"][0] * FPS)
    for c in (land, land + 3, land + 6):
        t = c / FPS
        if t in keys:
            assert keys[t] == pytest.approx(1.0 + L.spread_amount(w.start, w.voiced_end, t))
    assert k["tracking"][0][1] >= 1.0 and k["headroom"] == pytest.approx(L.spread_max(w.start, w.voiced_end))
    assert max(v for _, v in k["tracking"]) == pytest.approx(1.0 + L.spread_max(w.start, w.voiced_end), rel=0.05)
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False})
    assert "tracking" not in specs[0]["kinetic"] and "headroom" not in specs[0]["kinetic"]
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "spread": {"scale": 0.5}})
    assert specs[0]["kinetic"]["headroom"] == pytest.approx(0.5 * L.spread_max(w.start, w.voiced_end))


def test_weight_is_a_keyed_stroke_radius_between_lo_and_hi():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "weight": {"lo": 0.0, "hi": 0.04}})
    for s in specs:
        vals = [v for _, v in s["kinetic"]["weight"]]
        assert all(0.0 <= v <= 0.04 + 1e-9 for v in vals)
    assert "weight" not in run(tl, lyr={"line": 1, "style": "none"})[0][0]["kinetic"]


def test_type_style_sets_a_typewriter_reveal_over_the_word():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "type"})
    r = specs[1]["reveal"]
    assert r["from"] == pytest.approx(specs[1]["kinetic"]["show"][0]) and r["to"] == pytest.approx(1.5)


def test_drop_style_squashes_about_the_bottom_and_falls_into_the_cell():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "drop", "arrive": {"drop": 1.0}})
    k = specs[0]["kinetic"]
    assert k["pivot"] == "bottom" and k["dyp"][0][1] == pytest.approx(0.5) and abs(k["dyp"][-1][1]) < 1e-3
    assert "sy" in k and "sx" in k


def test_leave_drip_starts_on_the_tick_and_the_word_is_gone_before_the_cut():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "to": 4.0, "leave": "drip", "drip": {"life": 0.5, "g": 1000.0}})
    t_tick = (math.floor(3.0 * FPS + 0.4) - 1) / FPS
    assert summ["drip_from"] == pytest.approx(t_tick, abs=1e-4)
    for s in specs:
        d = s["kinetic"]["drip"]
        assert d["t"] == pytest.approx(t_tick) and d["g"] == 1000.0 and d["life"] == 0.5 and 1 <= d["seed"] <= 997
        assert s["kinetic"]["show"][1] < 4.0                                   # hidden once the last letter is gone
        assert s["kinetic"]["show"][1] >= t_tick + 0.5 + L.DRIP_SPREAD


def test_leave_drip_holds_to_the_cut_when_no_tick_fits():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "to": 2.8, "leave": "drip"})
    assert summ["drip_from"] is None and all("drip" not in s["kinetic"] for s in specs)
    assert all(round(s["kinetic"]["show"][1] * FPS) == 84 for s in specs)
    with pytest.raises(L.LyricsError, match="leave"):
        run(tl, lyr={"line": 1, "leave": "melt"})


def test_drip_rows_leave_the_lowest_first_with_a_gap():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 2, "style": "none", "to": 8.0, "leave": "drip",
                            "layout": {"kind": "stack", "dir": "down"}, "drip": {"gap": 0.05, "order": "bottom_first"}})
    ts = [s["kinetic"]["drip"]["t"] for s in specs]
    assert ts == sorted(ts, reverse=True) and ts[0] - ts[-1] == pytest.approx(0.05 * (len(ts) - 1))   # first on top: last
    specs, _ = run(tl, lyr={"line": 2, "style": "none", "to": 8.0, "leave": "drip",
                            "drip": {"gap": 0.05, "order": "last_first"}})
    ts = [s["kinetic"]["drip"]["t"] for s in specs]
    assert ts == sorted(ts, reverse=True)
    with pytest.raises(L.LyricsError, match="drip order"):
        run(tl, lyr={"line": 2, "to": 8.0, "leave": "drip", "drip": {"order": "sideways"}})


def test_recolor_keys_a_tint_from_the_last_words_landing():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "recolor": {"color": "foam", "over": 0.3}})
    last_land = round(specs[-1]["kinetic"]["show"][0] * FPS)
    for s in specs:
        t = s["kinetic"]["tint"]
        assert t["color"] == "foam" and t["keys"][-1][1] == pytest.approx(1.0)
        assert t["keys"][0][0] >= last_land / FPS - 1e-9 or s is not specs[-1]
    assert specs[-1]["kinetic"]["tint"]["keys"][0] == [last_land / FPS, 0.0]
    with pytest.raises(L.LyricsError, match="recolor needs"):
        run(tl, lyr={"line": 1, "recolor": {"over": 1.0}})


# ------------------------------------------------------------------------------------------------------ layouts
def test_flow_puts_the_words_in_reading_order_centred_on_the_panel():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "layout": "flow"})
    u = [s["offset"][0] for s in specs]
    assert u == sorted(u) and summ["layout"] == "flow"
    assert u[0] == pytest.approx(-u[-1], rel=1e-6) or abs(u[0] + u[-1]) < 0.2        # roughly symmetric about the centre
    assert all(s["align"] == "center" and s["valign"] == "middle" and "fit" not in s for s in specs)
    sizes = {s["size"] for s in specs}
    assert len(sizes) == 1                                                            # one cap height for the whole row
    em = sizes.pop() / 0.7
    widths = [fake_measure(w["text"])["w"] * (1 + L.spread_max(w["start"], w["voiced_end"]))
              for w in tl["lines"][0]["words"]]
    assert (sum(widths) + 3 * 0.28) * em <= 0.8 * 1.0 + 1e-9                          # fits 80 % of the 1 m panel


def test_flow_sets_one_row_per_lyric_line():
    tl = timeline()
    specs, _ = run(tl, lyr={"lines": [1, 2], "style": "none", "layout": "flow"}, panel=(2.0, 1.0))
    v = {s["lyric"][0]: round(s["offset"][1], 6) for s in specs}
    assert v[1] > v[2]                                                                # line 1 above line 2


def test_stack_gives_each_word_its_own_row_and_alternates_the_alignment():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "stack", "align": ["left", "right"]}})
    v = [s["offset"][1] for s in specs]
    u = [s["offset"][0] for s in specs]
    assert v == sorted(v, reverse=True)                                               # first word on top
    assert u[0] < 0 < u[1] and u[2] < 0 < u[3]
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "stack", "dir": "up"}})
    v = [s["offset"][1] for s in specs]
    assert v == sorted(v)                                                             # first word at the bottom
    with pytest.raises(L.LyricsError, match="stack dir"):
        run(tl, lyr={"line": 1, "layout": {"kind": "stack", "dir": "left"}})


def test_stack_rows_recycle_and_tilts_follow_the_row():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 2, "style": "none", "layout": {"kind": "stack", "rows": 3, "tilt": [-2, 1.5, 2.5]},
                            "recycle": True})
    ys = [round(s["offset"][1], 6) for s in specs]
    assert ys[0] == ys[3] and ys[1] == ys[4] and len({ys[0], ys[1], ys[2]}) == 3     # five words, three rows
    assert [s["kinetic"]["rot"][0][1] for s in specs] == [-2, 1.5, 2.5, -2, 1.5]
    first = specs[0]["kinetic"]["show"]
    assert first[1] == pytest.approx(specs[3]["kinetic"]["show"][0])                  # the row is taken over: gone at once


def test_slots_place_words_in_explicit_cells_and_recycle_with_a_fade():
    tl = timeline()
    slots = [{"at": [-0.25, 0.2], "box": [0.4, 0.2], "tilt": 3}, {"at": [0.25, -0.2], "box": [0.4, 0.2], "align": "left"}]
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "slots", "slots": slots},
                               "recycle": {"gap": 1, "fade": 4}})
    assert [s["offset"] for s in specs] == [[-0.25, 0.1], [0.25, -0.1], [-0.25, 0.1], [0.25, -0.1]]
    assert specs[0]["box"] == [0.4, 0.1] and specs[1]["align"] == "left"
    assert specs[0]["kinetic"]["rot"][0][1] == 3 and "rot" not in specs[1]["kinetic"]
    land2 = round(specs[2]["kinetic"]["show"][0] * FPS)
    assert round(specs[0]["kinetic"]["show"][1] * FPS) == land2 - 1                  # gone a frame before the next lands
    op = dict(specs[0]["kinetic"]["opacity"])
    t_exp = (land2 - 1) / FPS
    assert max(op) <= t_exp and op[max(op)] == pytest.approx(1 - L.sstep(t_exp - 4 / FPS, t_exp - 0.5 / FPS, max(op)))
    assert min(op.values()) < 0.2                                                    # faded to (almost) nothing
    assert "opacity" not in specs[2]["kinetic"] or specs[2]["kinetic"]["opacity"][0][1] == 1.0


def test_slots_assign_picks_the_slot_of_every_word():
    tl = timeline()
    slots = [{"at": [0.0, 0.3]}, {"at": [0.0, 0.0]}, {"at": [0.0, -0.3]}]
    specs, _ = run(tl, lyr={"line": 2, "style": "none",
                            "layout": {"kind": "slots", "slots": slots, "assign": [0, 0, 1, 2, 2]},
                            "recycle": True})
    assert [s["offset"][1] for s in specs] == [0.15, 0.15, 0.0, -0.15, -0.15]
    gone = [round(s["kinetic"]["show"][1] * FPS) for s in specs]
    assert gone[0] == round(specs[1]["kinetic"]["show"][0] * FPS) and gone[3] == round(specs[4]["kinetic"]["show"][0] * FPS)
    assert gone[1] == gone[2] == gone[4]                                              # nobody takes those over: the cut


def test_rise_words_start_at_the_base_and_reach_their_slots_when_the_last_word_lands():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "rise", "to": 4.0, "layout": {"kind": "stack", "dir": "down", "rows": 4}},
                   panel=(1.0, 1.0))
    t_last = specs[-1]["kinetic"]["show"][0]
    first = specs[0]["kinetic"]
    assert first["dyp"][0][1] < 0                                                    # starts below its slot (the base)
    assert first["dyp"][-1][1] == pytest.approx(0.0, abs=1e-9) and first["dyp"][-1][0] == pytest.approx(t_last)
    assert "dyp" not in specs[-1]["kinetic"] and "dxp" not in specs[-1]["kinetic"]   # the last word never moves
    assert any("rot" in s["kinetic"] for s in specs[:-1]) and "scale" in first
    with pytest.raises(L.LyricsError, match="rise"):
        run(tl, lyr={"line": 1, "style": "rise"})                                    # no layout: nowhere to rise to


# ------------------------------------------------------------------------------------------------------ validation
def test_unknown_keys_and_bad_values_are_errors_without_text():
    tl = timeline()
    for lyr, frag in [({"line": 1, "stile": "pop"}, "lyrics keys ['stile']"),
                      ({"line": 1, "style": "bounce"}, "style 'bounce'"),
                      ({"line": 1, "arrive": {"speed": 1}}, "arrive keys ['speed']"),
                      ({"line": 1, "layout": "spiral"}, "layout kind 'spiral'"),
                      ({"line": 1, "layout": {"kind": "flow", "pitch": 2}}, "layout 'flow' keys ['pitch']"),
                      ({"line": 1, "drip": {"bounce": 1}}, "drip keys ['bounce']"),
                      ({"line": 1, "from": "soon"}, "from must be a number of seconds"),
                      ({"line": 9}, "out of range"),
                      ({"line": 1, "layout": {"kind": "slots"}}, "needs slots")]:
        with pytest.raises(L.LyricsError) as e:
            run(tl, lyr=lyr)
        assert frag in str(e.value) and "text 'm'" in str(e.value) and no_text_in(str(e.value), tl)
    e = entry(line=1)
    e["text"] = "x"
    with pytest.raises(L.LyricsError, match="drop `text`"):
        run(tl, entry=e)
    with pytest.raises(L.LyricsError, match="needs a name"):
        L.expand({"lyrics": {"line": 1}}, tl, fps=FPS)
    with pytest.raises(L.LyricsError, match="measures the words"):
        run(tl, lyr={"line": 1, "layout": "flow"}, measure=None)


# ------------------------------------------------------------------------------------------------------ no text out
def test_the_text_is_only_in_the_text_field_of_each_spec():
    tl = timeline()
    for lyr in ({"line": 1, "style": "pop", "weight": {}, "recolor": {"color": "foam"}},
                {"line": 2, "style": "rise", "layout": {"kind": "stack", "rows": 5}, "leave": "drip", "to": 8.0},
                {"lines": [1, 3], "style": "drop", "layout": "flow", "colors": ["love"], "recycle": True},
                {"line": 1, "style": "type"}):
        specs, summ = run(tl, lyr=lyr, panel=(2.0, 1.0))
        assert specs
        for s in specs:
            assert s["text"] in words_of(tl)
            rest = {k: v for k, v in s.items() if k != "text"}
            assert no_text_in(rest, tl), rest["name"]
        assert no_text_in(summ, tl)
        assert all(re.fullmatch(r"m_l\d+w\d+", s["name"]) for s in specs)


def test_a_failing_measure_cannot_leak_through_the_layout():
    tl = timeline()
    calls = []

    def measure(s):
        calls.append(s)
        return fake_measure(s)

    run(tl, lyr={"line": 1, "layout": "flow"}, measure=measure)
    assert sorted(calls) == sorted(w["text"] for w in tl["lines"][0]["words"])       # it is the callback that sees the text


def test_the_build_report_has_no_room_for_the_text():
    """The text stage reports `chars` and `lyric` for a lyric word and never `widest`: check the source says so."""
    src = (Path(__file__).resolve().parent.parent / "mkmmd/blender/build/text.py").read_text(encoding="utf-8")
    assert 'info["lyric"] = list(lyric)' in src and 'info["chars"] = len(strings[0])' in src
    assert 'if lyric is None:' in src and 'shown = f"the word ({lyric[0]}, {lyric[1]})"' in src


# ---------------------------------------------------------------------------------------------- the text stage's side
def _literal(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_every_key_the_expansion_emits_is_one_the_text_stage_accepts():
    src = (Path(__file__).resolve().parent.parent / "mkmmd/blender/build/text.py").read_text(encoding="utf-8")
    known_text = _literal(ast.parse(src), "KNOWN")
    tl = timeline()
    slots = [{"at": [0.0, 0.3]}, {"at": [0.0, -0.3]}]
    emitted_text, emitted_kin = set(), set()
    for lyr in ({"lines": [1, 2], "style": "pop", "weight": {}, "recolor": {"color": "foam"}, "leave": "drip",
                 "to": 8.0, "tilt": [0, 3], "layout": {"kind": "slots", "slots": slots}, "recycle": {"fade": 3}},
                {"line": 1, "style": "drop", "backing": [{"color": "rose"}, {"pattern": "dots"}], "sizes": [0.1, 0.2]},
                {"line": 1, "style": "rise", "layout": {"kind": "stack", "rows": 4}},
                {"line": 1, "style": "type", "layout": "flow"}, {"line": 1, "style": "slap", "arrive": {"jitter": 0.05}}):
        specs, _ = run(tl, lyr=lyr, panel=(2.0, 1.0))
        for s in specs:
            emitted_text |= set(s)
            emitted_kin |= set(s["kinetic"])
    assert emitted_text <= known_text, emitted_text - known_text
    assert emitted_kin <= FX.KINETIC_KEYS, emitted_kin - FX.KINETIC_KEYS
    assert {"kinetic", "lyric", "back", "backing", "outline", "lit"} <= known_text
    for s in specs:                                  # and the stage's own validation takes what the expansion made
        assert FX.kinetic_spec(s)["show"]


# ---------------------------------------------------------------------------------------------- spelling, sizes, groups
def test_clean_and_case_set_the_word_without_touching_its_numbers():
    tl = timeline()
    tl["lines"][0]["words"][0]["text"] = "Ab,"
    tl["lines"][0]["words"][1]["text"] = "d\u2019E!"
    tl["lines"][0]["words"][2]["text"] = "--"
    ws = L.select(tl, line=1)
    assert [w.text for w in ws[:3]] == ["Ab,", "d\u2019E!", "--"]                    # kept as written
    ws = L.select(tl, line=1, case="lower", clean=True)
    assert [w.text for w in ws[:3]] == ["ab", "d'e", "--"]                           # nothing left to keep: the word stays
    assert [w.text for w in L.select(tl, line=1, case="upper")[:2]] == ["AB,", "D\u2019E!"]
    assert [w.text for w in L.select(tl, line=1, case="title", clean=True)[:2]] == ["Ab", "D'e"]
    assert [(w.line, w.index, w.start) for w in ws] == [(1, i, w["start"]) for i, w in enumerate(tl["lines"][0]["words"], 1)]
    with pytest.raises(L.LyricsError, match="case is one of"):
        L.select(tl, line=1, case="shout")
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "case": "upper", "clean": True})
    assert [s["text"] for s in specs[:2]] == ["AB", "D'E"]


def test_sizes_cycle_over_the_words_and_a_slots_own_size_wins():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "sizes": [0.1, 0.2]})
    assert [s["size"] for s in specs] == [0.1, 0.2, 0.1, 0.2]
    slots = [{"at": [0, 0.2], "size": 0.5}, {"at": [0, -0.2]}]
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "sizes": [0.1], "layout": {"kind": "slots", "slots": slots}})
    assert [s["size"] for s in specs] == [0.5, 0.1, 0.5, 0.1]
    for bad in ([], [0.0], ["a"], "big"):
        with pytest.raises(L.LyricsError, match="sizes is a list of cap heights"):
            run(tl, lyr={"line": 1, "sizes": bad})


def test_random_tilt_is_seeded_by_the_word_and_bounded():
    tl = timeline()
    a, _ = run(tl, lyr={"line": 1, "style": "none", "tilt": {"random": 5.0}})
    b, _ = run(tl, lyr={"line": 1, "style": "none", "tilt": {"random": 5.0}})
    ta = [s["kinetic"]["rot"][0][1] for s in a]
    assert ta == [s["kinetic"]["rot"][0][1] for s in b] and all(abs(t) <= 5.0 for t in ta) and len(set(ta)) == len(ta)
    with pytest.raises(L.LyricsError, match="tilt is a number"):
        run(tl, lyr={"line": 1, "tilt": {"spin": 3}})


def test_recycle_assign_groups_words_apart_from_where_they_stand():
    tl = timeline()
    slots = [{"at": [0, 0.3]}, {"at": [0, 0.0]}, {"at": [0, -0.3]}]
    lyr = {"line": 2, "style": "none", "to": 8.0, "layout": {"kind": "slots", "slots": slots, "assign": [0, 0, 0, 1, 2]},
           "recycle": {"assign": [0, 0, 0, 0, 0, 1]}}
    specs, _ = run(tl, lyr=lyr)
    land = [round(s["kinetic"]["show"][0] * FPS) for s in specs]
    gone = [round(s["kinetic"]["show"][1] * FPS) for s in specs]
    assert gone[:4] == land[1:5]                    # one at a time: each leaves as the next lands, the 4th as the 5th does
    assert gone[4] == 240                           # nobody follows the last: the cut
    del lyr["recycle"]["assign"]
    specs, _ = run(tl, lyr=lyr)
    gone = [round(s["kinetic"]["show"][1] * FPS) for s in specs]
    assert gone[0] == land[1] and gone[1] == land[2] and gone[2] == 240 and gone[3] == 240     # by slot: 0 0 0 1 2
    with pytest.raises(L.LyricsError, match="recycle assign"):
        run(tl, lyr={"line": 1, "recycle": {"assign": []}})


def test_recolor_is_zero_until_its_moment_and_a_word_gone_before_it_never_changes():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "to": 4.0, "recolor": {"color": "foam", "from": 2.3, "over": 0.3}})
    last = specs[-1]["kinetic"]["tint"]["keys"]
    assert last[0][1] == 0.0 and last[-1][1] == pytest.approx(1.0) and last[-1][0] == pytest.approx(2.6, abs=1 / FPS)
    first = specs[0]["kinetic"]["tint"]["keys"]
    assert first[0] == [30 / FPS, 0.0] and first[-1][1] == pytest.approx(1.0)        # lands at 1.0 s, ramps at 2.3 s
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "recolor": {"color": "foam", "from": 2.3, "over": 0.3}})
    assert specs[-1]["kinetic"]["tint"]["keys"][-1][1] > 0.95                         # the cut ends the ramp a frame short
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "to": 2.0, "recolor": {"color": "foam", "from": 2.3, "over": 0.3}})
    assert all(s["kinetic"]["tint"]["keys"] == [[s["kinetic"]["show"][0], 0.0]] for s in specs)      # a flat 0: no tint


def test_backing_goes_to_every_word_or_cycles_over_them():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "backing": {"color": "rose"}})
    assert all(s["backing"] == {"color": "rose"} for s in specs)
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "backing": [{"pattern": "dots"}, {"pattern": "check"}]})
    assert [s["backing"]["pattern"] for s in specs] == ["dots", "check", "dots", "check"]
    e = entry(line=1, style="none")
    e["backing"] = {"color": "gold"}
    specs, _ = run(tl, entry=e)
    assert all(s["backing"] == {"color": "gold"} for s in specs)                    # an entry-level backing is inherited
    with pytest.raises(L.LyricsError, match="backing is a table"):
        run(tl, lyr={"line": 1, "backing": "stripe"})


def test_recolor_words_limits_the_tint_to_part_of_the_selection():
    tl = timeline()
    rc = {"color": "foam", "from": 1.5, "over": 0.4, "words": [1, 3]}
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "to": 4.0, "recolor": rc})
    assert ["tint" in s["kinetic"] for s in specs] == [True, True, True, False]
    for bad in ([0, 2], [3, 2], [1], "all", [1.5, 2]):
        with pytest.raises(L.LyricsError, match="recolor words is"):
            run(tl, lyr={"line": 1, "recolor": {"color": "foam", "words": bad}})


def test_offsets_nudge_each_word_on_top_of_where_it_stands():
    tl = timeline()
    e = entry(line=1, style="none", offsets=[[0.0, -0.1], [0.05, 0.0]])
    e["offset"] = [0.5, 0.5]
    specs, _ = run(tl, entry=e)
    assert [s["offset"] for s in specs] == [[0.5, 0.4], [0.55, 0.5], [0.5, 0.4], [0.55, 0.5]]
    slots = [{"at": [0.0, 0.2]}]
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "offsets": [[0.01, 0.02]],
                            "layout": {"kind": "slots", "slots": slots}})
    assert specs[0]["offset"] == [0.01, pytest.approx(0.1 + 0.02)]
    for bad in ([], [[1]], [[1, "a"]], "up", [0.1, 0.2]):
        with pytest.raises(L.LyricsError, match="offsets is a list"):
            run(tl, lyr={"line": 1, "offsets": bad})


def test_rise_keeps_its_base_with_offsets_and_a_selection_cut_before_it_lands_is_empty_not_an_error():
    tl = timeline()
    rise = {"line": 1, "style": "rise", "to": 4.0, "layout": {"kind": "stack", "rows": 4}}
    a, _ = run(tl, lyr=rise, panel=(1.0, 1.0))
    b, _ = run(tl, lyr={**rise, "offsets": [[0.0, 0.0]]}, panel=(1.0, 1.0))
    assert [s["kinetic"].get("dyp") for s in a] == [s["kinetic"].get("dyp") for s in b]
    specs, summ = run(tl, lyr={"line": 1, "style": "pop", "to": 0.5})
    assert specs == [] and summ["words"] == 0 and len(summ["skipped"]) == 4 and summ["first_frame"] is None
    specs, summ = run(tl, lyr={"line": 1, "style": "rise", "to": 0.5, "layout": {"kind": "stack"}}, panel=(1.0, 1.0))
    assert specs == [] and summ["words"] == 0
