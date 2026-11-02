"""Calls (mkmmd/core/calltype.py): words of a [[text]]'s own, one on each hit of `times` or a lane, round-robin, through
the lyric machinery: where each lands, when and how it leaves (`hold`, `recycle`, `to`), its name, colour and strip, the
summary's numbers, and every message a bad calls table gets. The words are idol fan calls (never a song's words) on
synthetic hits; the measure is a stand-in, so nothing here needs Blender."""
import json
import math
import re

import numpy as np
import pytest

from mkmmd.core import calltype as CT
from mkmmd.core import wordtype as L

FPS = 30.0
WORDS = ["ハイ！", "オイ！", "タイガー"]
HITS = {"kick": [{"t": 0.5, "db": 0.0}, {"t": 1.0, "db": -1.0}, {"t": 1.5, "db": -6.0}, {"t": 2.0, "db": -2.0},
                 {"t": 2.5, "db": 0.0}, {"t": 3.0, "db": -0.5}, 3.5],
        "snare": [0.75, 1.75]}
SLOTS2 = {"kind": "slots", "slots": [{"at": [-0.25, 0.2]}, {"at": [0.25, -0.2]}]}


def fake_measure(s, font=None, tracking=None):
    return {"w": 0.55 * len(s), "y0": -0.2, "y1": 0.7, "cap": 0.7}


def entry(**calls):
    calls.setdefault("words", WORDS)
    return {"name": "cl", "screen": True, "size": 0.08, "calls": calls}


def run(e, frame0=31, hits=HITS, end=6.0, measure=fake_measure):
    return CT.expand(e, fps=FPS, frame0=frame0, hits=hits, end=end, panel=(1.78, 1.0), measure=measure)


def frames(specs, i):
    """Clip frame of each call's show[i]: 0 the landing, 1 the frame it is gone on."""
    return [round(s["kinetic"]["show"][i] * FPS) for s in specs]


def at(keys, t):
    """A keyed channel's value at t: linear between its keys, held outside them."""
    ts, vs = zip(*keys)
    return float(np.interp(t, ts, vs))


# ------------------------------------------------------------------------------------------------------ hits
def test_a_call_lands_on_floor_t_fps_plus_point_four_whatever_frame0():
    for frame0 in (0, 31, 181):
        specs, summ = run(entry(times=[2.2, 1.0, 0.99, 1.0 - 0.5 / FPS], style="none"), frame0=frame0)
        assert [s["kinetic"]["show"][0] for s in specs] == [29 / FPS, 30 / FPS, 30 / FPS, 66 / FPS]
        assert (summ["first_frame"], summ["last_frame"]) == (29, 66)


def test_hit_k_shows_words_k_mod_len_named_by_its_number_among_the_hits_picked():
    specs, _ = run(entry(times=[3.0, 1.0, 5.0, 2.0, 4.0, 0.5], **{"from": 1.0}))
    assert [s["name"] for s in specs] == ["cl_c1", "cl_c2", "cl_c3", "cl_c4", "cl_c5"]
    assert [s["text"] for s in specs] == [WORDS[0], WORDS[1], WORDS[2], WORDS[0], WORDS[1]]
    assert frames(specs, 0) == [30, 60, 90, 120, 150]
    specs, _ = run(entry(words="ハイ！", times=[1.0, 2.0]))           # one call: a list of one
    assert [s["text"] for s in specs] == ["ハイ！", "ハイ！"]


def test_times_are_sorted_and_kept_from_from_up_to_not_including_to():
    assert CT.hit_times({"words": WORDS, "times": [3.0, 0.5, 1.0, 2.0, 2.999], "from": 1.0, "to": 3.0}) == [1.0, 2.0, 2.999]
    assert CT.hit_times({"words": WORDS, "times": [2.5, 0.5]}) == [0.5, 2.5]
    assert CT.hit_times({"words": WORDS, "times": 2.0}) == [2.0]
    specs, _ = run(entry(times=[3.0, 0.5, 1.0, 2.0], **{"from": 1.0, "to": 3.0}))
    assert frames(specs, 0) == [30, 60] and frames(specs, 1) == [90, 90]


def test_a_lane_gives_its_hits_at_or_above_min_db_every_nth_inside_from_and_to():
    def pick(**kw):
        return CT.hit_times({"words": WORDS, "lane": "kick", **kw}, HITS)

    assert pick() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert pick(min_db=-3) == [0.5, 1.0, 2.0, 2.5, 3.0]               # a hit without a level only when none is asked
    assert pick(min_db=-3, every=2) == [0.5, 2.0, 3.0]
    assert pick(min_db=-3, **{"from": 1.0, "to": 3.0}) == [1.0, 2.0, 2.5]
    assert pick(min_db=-3, every=2, **{"from": 1.0, "to": 3.0}) == [1.0, 2.5]
    specs, summ = run(entry(lane="kick", min_db=-3, every=2))
    assert frames(specs, 0) == [15, 60, 90] and summ["hits"] == 3


# ------------------------------------------------------------------------------------------------------ leaving
def test_hold_takes_a_call_down_round_hold_fps_frames_after_it_lands_and_never_sooner_than_a_frame():
    specs, _ = run(entry(times=[1.0, 2.0], hold=0.5))
    assert frames(specs, 1) == [45, 75]
    specs, _ = run(entry(times=[1.0, 2.0], hold=0.51))               # 15.3 frames: rounded, 15
    assert frames(specs, 1) == [45, 75]
    specs, _ = run(entry(times=[1.0, 2.0], hold=0.01))                # a third of a frame: up for one
    assert frames(specs, 1) == [31, 61]
    specs, _ = run(entry(times=[1.0, 5.5], hold=2.0))                 # a hold past the end: gone at the end
    assert frames(specs, 1) == [90, 180]
    assert all("opacity" not in s["kinetic"] for s in specs)          # without recycle's fade it simply goes


def test_with_recycle_the_next_call_in_a_slot_takes_it_over():
    specs, _ = run(entry(times=[1.0, 1.5, 2.0, 2.5], layout=SLOTS2, recycle=True))
    assert frames(specs, 0) == [30, 45, 60, 75] and frames(specs, 1) == [60, 75, 180, 180]
    specs, _ = run(entry(times=[1.0, 1.5, 2.0, 2.5], layout=SLOTS2, recycle={"gap": 2}))
    assert frames(specs, 1) == [58, 73, 180, 180]
    specs, _ = run(entry(times=[1.0, 1.5, 2.0], recycle=True))       # one place (layout same): each takes the last's
    assert frames(specs, 1) == [45, 60, 180]


def test_the_fade_and_slide_exit_starts_at_the_earlier_of_the_next_landing_and_the_end_of_the_hold():
    specs, _ = run(entry(times=[1.0, 1.4, 3.0], hold=0.5, recycle={"fade": 3, "slide": 0.5, "gap": 1}, style="none"))
    exits = [41, 57, 105]   # c1: the next lands on 42 (less the gap) before its hold ends on 45; c2: its hold ends on 57,
    assert frames(specs, 1) == exits    # before the next lands on 90; c3: nothing follows, its hold ends on 105
    for s, e in zip(specs, exits):
        k = s["kinetic"]
        fade = [1.0 - L.sstep((e - 3) / FPS, (e - 0.5) / FPS, c / FPS) for c in range(e - 3, e)]
        assert [at(k["opacity"], c / FPS) for c in range(e - 3, e)] == pytest.approx(fade)
        assert at(k["opacity"], (e - 4) / FPS) == 1.0 and fade[0] == 1.0 and fade[-1] < 0.2
        slide = [0.5 * L.sstep((e - 3) / FPS, e / FPS, c / FPS) ** 2 for c in range(e - 3, e)]
        assert [at(k["dx"], c / FPS) for c in range(e - 3, e)] == pytest.approx(slide)
        assert at(k["dx"], (e - 4) / FPS) == pytest.approx(0.0, abs=1e-12)


def test_every_call_is_gone_by_to_or_by_the_end_of_the_build_when_to_is_not_given():
    specs, summ = run(entry(times=[1.0, 2.0, 3.5], to=3.0))
    assert frames(specs, 1) == [90, 90] and summ["cut_frame"] == 90 and summ["hits"] == 2     # 3.5 is past `to`
    specs, summ = run(entry(times=[1.0, 2.0]), end=2.5)
    assert frames(specs, 1) == [75, 75] and summ["cut_frame"] == 75
    to = 2.0 + 0.5 / FPS                                              # on a half frame: rounded as the shots stage cuts
    for frame0 in (0, 1, 181):
        specs, summ = run(entry(times=[1.0], to=to), frame0=frame0)
        assert frames(specs, 1) == [int(round(frame0 + to * FPS)) - frame0] == [summ["cut_frame"]]
    assert {run(entry(times=[1.0], to=to), frame0=f)[1]["cut_frame"] for f in (0, 1)} == {60, 61}


# ------------------------------------------------------------------------------------------------------ looks
def test_colors_cycle_with_the_call_or_with_the_hit():
    e = entry(words=["ハイ！", "オイ！"], times=[1.0, 2.0, 3.0, 4.0], colors=["love", "gold", "pine"])
    assert [s["color"] for s in run(e)[0]] == ["love", "gold", "love", "gold"]          # by call (the default)
    e["calls"]["color_by"] = "hit"
    assert [s["color"] for s in run(e)[0]] == ["love", "gold", "pine", "love"]
    e = entry(words=["ハイ！", "オイ！"], times=[1.0, 2.0])
    e["color"] = "rose"
    assert [s["color"] for s in run(e)[0]] == ["rose", "rose"]                           # no colors: the entry's


def test_strips_cycle_over_the_hits_and_a_random_tilt_is_fixed_by_call_and_hit():
    strips = [{"color": "love", "slant": 14}, {"color": "gold", "slant": -12}, {"color": "foam", "slant": 16}]
    e = entry(words=["ハイ！", "オイ！"], times=[1.0, 2.0, 3.0, 4.0], backing=strips, tilt={"random": 10}, style="none")
    specs = run(e)[0]
    assert [s["backing"]["color"] for s in specs] == ["love", "gold", "foam", "love"]
    rot = [s["kinetic"]["rot"][0][1] for s in specs]
    assert rot == [s["kinetic"]["rot"][0][1] for s in run(e)[0]] and all(abs(r) <= 10 for r in rot) and len(set(rot)) == 4


def test_calls_side_by_side_are_set_without_room_for_a_spread():
    specs, _ = run(entry(times=[1.0, 2.0, 3.0], layout={"kind": "flow", "rows": 1}, style="none"))
    em = specs[0]["size"] / 0.7                                       # the common cap height over the cap per em
    w = [0.55 * len(s["text"]) for s in specs]                        # each call as wide as its ink: no headroom
    gaps = [b["offset"][0] - a["offset"][0] for a, b in zip(specs, specs[1:])]
    assert gaps == pytest.approx([em * (0.5 * (w[k] + w[k + 1]) + 0.28) for k in range(2)])


# ------------------------------------------------------------------------------------------------------ what comes out
def test_the_summary_counts_calls_hits_words_and_the_hits_never_on_screen_in_clip_frames():
    specs, summ = run(entry(times=[1.0, 1.01, 1.99, 2.5], to=2.0, recycle=True))
    assert [s["name"] for s in specs] == ["cl_c2"]   # c1 lands with c2 and gives it its place; c3 lands on the cut
    assert summ == {"calls": 1, "hits": 3, "words": 3, "style": "pop", "layout": "same", "skipped": [1, 3],
                    "first_frame": 30, "last_frame": 30, "cut_frame": 60}


def test_a_call_s_string_is_only_in_its_text_and_no_call_is_a_sung_word():
    for calls in ({"times": [1.0, 2.0, 3.0, 4.0]},
                  {"lane": "kick", "min_db": -3, "style": "slam", "hold": 0.4, "tilt": {"random": 10},
                   "colors": ["love", "gold"], "color_by": "hit", "recycle": {"fade": 3, "slide": 6},
                   "backing": [{"color": "love", "slant": 14}, {"color": "gold", "slant": -12}], "layout": SLOTS2},
                  {"times": [1.0, 1.5, 2.0], "layout": {"kind": "flow", "rows": 1}, "style": "drop"}):
        specs, summ = run(entry(**calls))
        assert specs and all(re.fullmatch(r"cl_c\d+", s["name"]) for s in specs)
        for s in specs:
            assert s["text"] in WORDS and "lyric" not in s and "reveal" not in s
            rest = json.dumps({k: v for k, v in s.items() if k != "text"}, ensure_ascii=False)
            assert not any(w in rest for w in WORDS), s["name"]
            assert not {"tracking", "headroom", "weight", "wobble", "drip"} & set(s["kinetic"])   # nothing follows a note
        assert not any(w in json.dumps(summ, ensure_ascii=False) for w in WORDS)


# ------------------------------------------------------------------------------------------------------ refused
def test_every_bad_calls_table_is_refused_with_a_message_that_names_the_entry_and_the_key():
    def bad(calls, frag, **kw):
        with pytest.raises(CT.CallError) as e:
            run({"name": "cl", "screen": True, "calls": calls}, **kw)
        msg = str(e.value)
        assert msg.startswith("text 'cl': ") and frag in msg, msg
        return msg

    t = {"words": WORDS, "times": [1.0, 2.0]}
    bad("ハイ！", "calls is a table {words, times | lane, style, ...}")
    msg = bad({**t, "colour": "love"}, "calls keys ['colour'] are unknown (known: [")
    assert "lyrics" not in msg
    bad({**t, "case": "upper", "spread": False, "colour": 1},
        "calls keys ['case', 'colour', 'spread'] are unknown (known: ['arrive', 'backing', 'color_by', 'colors', 'every', "
        "'from', 'hold', 'lane', 'layout', 'min_db', 'offsets', 'recycle', 'scales', 'sizes', 'style', 'tilt', 'times', "
        "'to', 'words']); ['case', 'spread'] belong to lyrics: calls are set as written and have no note")
    for key in sorted(L.LYRICS_KEYS - CT.CALL_KEYS):
        bad({**t, key: 1}, f"calls keys ['{key}'] are unknown")
        bad({**t, key: 1}, f"; ['{key}'] belong to lyrics: calls are set as written and have no note")
    for words in (None, [], [""], [" "], ["ハイ！", 3], "", 3):
        bad({"times": [1.0]} if words is None else {"words": words, "times": [1.0]},
            "calls: words is the list of calls (strings, taken round-robin by the hits)")
    bad({**t, "style": "type"}, "calls: style 'type' plays over a sung note, and a call has none")
    bad({**t, "color_by": "word"}, "calls: color_by is 'call' or 'hit' (a list of colors cycles with the call, or with "
                                   "the hit)")
    bad({**t, "from": "soon"}, "calls: from = 'soon': a number")
    bad({**t, "to": True}, "calls: to = True: a number")
    bad({**t, "to": math.nan}, "calls: to = nan: a number")
    bad({**t, "from": 2.0, "to": 1.0}, "calls: to = 1 comes before from = 2")
    bad({**t, "from": 1.0, "to": 1.0}, "calls: to = 1 comes before from = 1")
    bad({**t, "hold": 0}, "calls: hold = 0: more than 0")
    bad({**t, "hold": -0.5}, "calls: hold = -0.5: more than 0")
    bad({**t, "hold": "1"}, "calls: hold = '1': a number")
    both = "calls: give `times` (clip seconds) or `lane` (one of the [audio] hits file), one of them"
    bad({**t, "lane": "kick"}, both)
    bad({"words": WORDS}, both)
    bad({**t, "min_db": -3}, "calls: min_db and every pick from a lane; give the times you want")
    bad({**t, "every": 2}, "calls: min_db and every pick from a lane; give the times you want")
    for times in ([], "1.0"):
        bad({"words": WORDS, "times": times}, "calls: times is a list of clip seconds")
    bad({"words": WORDS, "times": [1.0, "x"]}, "calls: times[] = 'x': a number")
    lane = {"words": WORDS, "lane": "kick"}
    bad(lane, "calls: lane 'kick' reads the [audio] hits file, and the project names none (`mk timeline hits` writes it)",
        hits=None)
    bad({**lane, "lane": "tom"}, "calls: no lane 'tom' in the hits file (have kick, snare)")
    bad({**lane, "lane": 3}, "calls: `lane` names a lane of the [audio] hits file")
    bad({**lane, "every": 0}, "calls: every = 0: a whole number, at least 1")
    bad({**lane, "min_db": "loud"}, "calls: min_db = 'loud': a number")
    bad({**lane, "min_db": 1.0}, "calls: no hit to land on (lane 'kick': mind from, to and min_db)")
    bad({"words": WORDS, "times": [5.0], "to": 3.0}, "calls: no hit to land on between from and to")
    bad(t, "calls: give `to`, where the calls are gone", end=None)
    bad({**t, "style": "bounce"}, "calls: style 'bounce' must be one of")      # the lyric machinery's, in calls' terms
    bad({**t, "layout": "spiral"}, "calls: layout kind 'spiral' must be one of")
    bad({**t, "recycle": {"bogus": 1}}, "calls: recycle keys ['bogus'] are unknown")
    bad({**t, "layout": "flow"}, "calls: layout 'flow' measures the words; the build supplies that", measure=None)


def test_a_calls_entry_has_a_name_and_no_words_of_another_kind_and_a_lyrics_entry_no_calls():
    with pytest.raises(CT.CallError, match=r"^\[\[text\]\] with calls needs a name$"):
        CT.expand({"calls": {"words": WORDS, "times": [1.0]}}, fps=FPS, end=6.0)
    for key in ("text", "value", "kinetic", "reveal", "lyric", "lyrics"):
        e = entry(times=[1.0])
        e[key] = {"line": 1} if key == "lyrics" else "x"
        with pytest.raises(CT.CallError) as err:
            run(e)
        assert str(err.value) == f"text 'cl': a calls entry takes its words from `calls.words`: drop `{key}`"
    tl = {"lines": [{"words": [{"text": "aa", "start": 1.0, "end": 1.2, "voiced_end": 1.2}]}]}
    e = {"name": "ly", "screen": True, "lyrics": {"timeline": "t.json", "line": 1}, "calls": {"words": WORDS}}
    with pytest.raises(L.LyricsError, match=r"^text 'ly': a lyrics entry takes its words from the timeline: drop `calls`$"):
        L.expand(e, tl, fps=FPS)
