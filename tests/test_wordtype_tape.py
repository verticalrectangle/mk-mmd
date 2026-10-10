"""Lyric type, the cassette features (docs/design.md: Text, Lyrics): the slide, wow and flutter, the rewind, punch words, row
breaks, carry and zones, on the synthetic timeline of test_wordtype (made-up filler words and numbers)."""
import math
import re
import tomllib
from pathlib import Path

import pytest
from test_wordtype import FPS, entry, fake_measure, no_text_in, timeline, words_of

from mkmmd.core import screentype as SR
from mkmmd.core import tapefx as TF
from mkmmd.core import typefx as FX
from mkmmd.core import wordtype as L

SHOTS = [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}, {"name": "c", "from": 4.0, "to": 6.0}]


def run(tl, shots=None, panel_of=None, **kw):
    e = kw.pop("entry", None) or entry(**kw.pop("lyr"))
    e.update(kw.pop("extra", {}))
    return L.expand(e, tl, fps=FPS, frame0=kw.pop("frame0", 181), panel=kw.pop("panel", (1.0, 0.5)),
                    measure=kw.pop("measure", fake_measure), shots=shots, panel_of=panel_of)


def frames(keys):
    return [round(t * FPS) for t, _ in keys]


# ------------------------------------------------------------------------------------------------------ slide
def test_slide_feeds_each_word_in_from_alternating_sides_and_clunks_home():
    specs, _ = run(timeline(), lyr={"line": 1, "style": "slide", "spread": False})
    k0, k1 = specs[0]["kinetic"], specs[1]["kinetic"]
    land0 = round(k0["show"][0] * FPS)
    assert frames(k0["dxp"])[0] == land0 and k0["dxp"][0][1] == pytest.approx(0.5) and k1["dxp"][0][1] == pytest.approx(-0.5)
    assert k0["dxp"][-1][1] == pytest.approx(0.0) and k0["rot"][0][1] == pytest.approx(4.0)
    assert min(v for _, v in k0["sx"]) < 0.95 and k0["sx"][-1][1] == pytest.approx(1.0, abs=2e-3)      # the clunk dies away
    assert "opacity" not in k0                                                                         # hashed alpha is grainy: none


def test_slide_takes_its_numbers_and_rejects_bad_ones():
    specs, _ = run(timeline(), lyr={"line": 1, "style": "slide", "spread": False,
                                    "arrive": {"from": "left", "dist": 0.3, "dur": 0.3, "tilt": 0.0}})
    assert all(s["kinetic"]["dxp"][0][1] == pytest.approx(-0.3) for s in specs)
    assert "rot" not in specs[0]["kinetic"]
    for arrive, frag in (({"from": "up"}, "'left', 'right' or 'alt'"), ({"dur": 0.0}, "dur must be positive"),
                         ({"speed": 1}, "arrive keys \\['speed'\\]")):
        with pytest.raises(L.LyricsError, match=frag):
            run(timeline(), lyr={"line": 1, "style": "slide", "arrive": arrive})


# ------------------------------------------------------------------------------------------------------ wow
def test_wow_moves_only_the_words_held_long_enough_and_only_while_they_are_held():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "wow": {}})
    held = [s["lyric"][1] for s in specs if "dx" in s["kinetic"]]
    assert held == [3, 4]                                                    # notes of 0.30 s and 0.40 s; the short ones stay put
    k = specs[3]["kinetic"]
    w = L.select(tl, line=1)[3]
    assert frames(k["dx"])[0] >= round(w.start * FPS) and frames(k["dx"])[-1] <= math.ceil((w.voiced_end + 0.15) * FPS) + 1
    assert max(abs(v) for _, v in k["dx"]) <= TF.WOW_DEFAULTS["amount"] + 1.5 * TF.WOW_DEFAULTS["flutter"] + 1e-9
    assert 0.0 < max(abs(v) for _, v in k["rot"]) <= TF.WOW_DEFAULTS["tilt"] + 1e-9
    assert k["sx"][0][1] == pytest.approx(1.0) and any(abs(v - 1.0) > 1e-3 for _, v in k["sx"])


def test_wow_adds_to_the_arrival_instead_of_replacing_it():
    specs, _ = run(timeline(), lyr={"line": 1, "style": "slide", "spread": False, "wow": {}})
    k = specs[3]["kinetic"]
    assert k["dxp"][0][1] != 0.0 and "dx" in k                                                # still arrives from the side
    land = k["show"][0]
    clunk = min(v for t, v in k["sx"] if t < land + 0.35)
    assert clunk < 0.97 and any(abs(v) > 0.005 for t, v in k["dx"] if t > land + 0.35)       # the clunk, then the wow


def test_wow_numbers_can_be_changed_and_a_longer_minimum_hold_leaves_more_words_alone():
    specs, _ = run(timeline(), lyr={"line": 1, "style": "none", "spread": False, "wow": {"min_hold": 0.35, "amount": 0.2}})
    assert [s["lyric"][1] for s in specs if "dx" in s["kinetic"]] == [4]
    assert max(abs(v) for _, v in specs[3]["kinetic"]["dx"]) > 0.1
    with pytest.raises(L.LyricsError, match="wow keys \\['wobble'\\]"):
        run(timeline(), lyr={"line": 1, "wow": {"wobble": 1}})


# ------------------------------------------------------------------------------------------------------ rewind
def test_rewind_starts_on_a_tick_and_every_word_shoots_back_smeared_and_fades():
    specs, summ = run(timeline(), lyr={"line": 1, "style": "none", "spread": False, "to": 4.0, "leave": "rewind"})
    t0 = (math.floor(3.0 * FPS + 0.4) - 1) / FPS                                   # the tick at 3.0 s: one frame ahead of it
    assert summ["rewind_from"] == pytest.approx(round(t0, 4)) and summ["drip_from"] is None
    for k, s in enumerate(specs):
        kin = s["kinetic"]
        start = t0 + 0.02 * k                                                     # first_first: a gap after each
        assert all(abs(v) < 1e-12 for t, v in kin["dxp"] if t <= start + 1e-9)    # nothing moves before its turn ...
        assert any(abs(v) > 0.0 for t, v in kin["dxp"] if t > start + 1e-9)       # ... and it goes the frame after
        assert kin["dxp"][-1][1] == pytest.approx(-0.6) and kin["sx"][-1][1] == pytest.approx(4.0)
        assert kin["opacity"][-1][1] == 0.0
        assert abs(kin["show"][1] - ((start + 0.2) * FPS + 1) / FPS) <= 1.01 / FPS    # gone when the life is out
        assert kin["show"][1] < 4.0                                                # gone well before the cut


def test_rewind_order_none_takes_every_word_at_once_and_last_first_reverses():
    a, _ = run(timeline(), lyr={"line": 1, "style": "none", "spread": False, "to": 4.0, "leave": "rewind",
                                "rewind": {"order": "none"}})
    ends = {s["kinetic"]["show"][1] for s in a}
    assert len(ends) == 1
    b, _ = run(timeline(), lyr={"line": 1, "style": "none", "spread": False, "to": 4.0, "leave": "rewind",
                                "rewind": {"order": "last_first", "gap": 0.05}})
    shows = [s["kinetic"]["show"][1] for s in b]
    assert shows == sorted(shows, reverse=True) and shows[0] > shows[-1]


def test_rewind_holds_to_the_cut_when_no_tick_fits_and_checks_its_numbers():
    specs, summ = run(timeline(), lyr={"line": 1, "style": "none", "spread": False, "to": 2.8, "leave": "rewind"})
    assert summ["rewind_from"] is None and all("dxp" not in s["kinetic"] for s in specs)
    for bad, frag in (({"order": "sideways"}, "rewind order"), ({"life": 0.0}, "life must be positive")):
        with pytest.raises(L.LyricsError, match=frag):
            run(timeline(), lyr={"line": 1, "to": 4.0, "leave": "rewind", "rewind": bad})
    with pytest.raises(L.LyricsError, match="leave is one of"):
        run(timeline(), lyr={"line": 1, "leave": "explode"})


# ------------------------------------------------------------------------------------------------------ punch, scales
def levels(tl):
    return [w.db for w in L.select(tl, line=1)]


def test_punch_picks_the_loud_words_by_level_and_gives_them_their_own_look():
    tl = timeline()
    db = levels(tl)
    above = sorted(db)[len(db) // 2]                       # the louder half
    calls = []

    def measure(s, font=None, tracking=None):
        calls.append(font)
        return fake_measure(s)

    punch = {"above": above, "font": "bold", "color": "gold", "scale": 1.5, "backing": False, "tilt": 3.0}
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "layout": "flow", "punch": punch}, measure=measure)
    flagged = [k for k, d in enumerate(db) if d >= above]
    assert [s["lyric"][1] - 1 for s in specs if s.get("font") == "bold"] == flagged
    assert summ["punch"] == [[1, k + 1] for k in flagged]
    loud, soft = [s for s in specs if s.get("font") == "bold"], [s for s in specs if s.get("font") != "bold"]
    assert all(s["color"] == "gold" and s["backing"] is False for s in loud)
    assert all("backing" not in s and s.get("color") != "gold" for s in soft)
    assert {round(s["size"] / soft[0]["size"], 6) for s in loud} == {1.5}              # cap height 1.5 x the others'
    assert all(s["kinetic"]["rot"][-1][1] == pytest.approx(3.0) for s in loud)
    assert calls.count("bold") == len(loud)                                             # measured in their own font
    assert no_text_in(summ, tl)


def test_punch_by_number_and_scales_cycle_over_the_words():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "layout": "flow", "punch": {"words": [2, 4], "scale": 2.0},
                               "scales": [1.0, 1.0, 0.5]})
    assert summ["punch"] == [[1, 2], [1, 4]]
    sizes = [s["size"] for s in specs]
    assert [round(x / sizes[0], 6) for x in sizes] == [1.0, 2.0, 0.5, 2.0]     # scales cycled (1, 1, 0.5, 1) x punch (1, 2, 1, 2)


def test_punch_and_scales_have_to_make_sense():
    tl = timeline()
    for lyr, frag in (({"punch": {}}, "punch needs `above`"),
                      ({"punch": {"above": -5, "style": "rise"}}, "cannot arrive as steam"),
                      ({"punch": {"words": ["a"]}}, "whole numbers"),
                      ({"punch": {"above": -5, "scale": 0}}, "scale must be positive"),
                      ({"punch": {"above": -5, "colour": "x"}}, "punch keys \\['colour'\\]"),
                      ({"scales": [1, -1]}, "positive numbers")):
        with pytest.raises(L.LyricsError, match=frag):
            run(tl, lyr={"line": 1, **lyr})


def test_a_punch_word_can_arrive_in_a_style_of_its_own():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False,
                            "punch": {"words": [1], "style": "slap", "arrive": {"wobble": 5.0}}})
    assert "rot" in specs[0]["kinetic"] and "rot" not in specs[1]["kinetic"]


# ------------------------------------------------------------------------------------------------------ rows
def test_flow_rows_wraps_the_words_into_that_many_even_rows():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "flow", "rows": 2}}, panel=(2.0, 1.0))
    v = [round(s["offset"][1], 6) for s in specs]
    assert len(set(v)) == 2 and v[0] == v[1] and v[2] == v[3] and v[0] > v[2]
    one, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "flow", "rows": 1}}, panel=(2.0, 1.0))
    assert len({round(s["offset"][1], 6) for s in one}) == 1


def test_flow_rows_auto_takes_the_fewest_rows_that_reach_the_size_or_the_biggest_type():
    tl = timeline()
    wide, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "flow", "rows": "auto"}}, panel=(8.0, 1.0),
                  extra={"size": 0.1})
    assert len({round(s["offset"][1], 6) for s in wide}) == 1                       # room enough: one row at 0.1
    narrow, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "flow", "rows": "auto"}}, panel=(0.6, 1.0),
                    extra={"size": 0.1})
    assert len({round(s["offset"][1], 6) for s in narrow}) >= 2
    big, _ = run(tl, lyr={"line": 1, "style": "none", "layout": {"kind": "flow", "rows": "auto"}}, panel=(1.0, 1.0))
    assert len({round(s["offset"][1], 6) for s in big}) >= 2                        # no size: whatever makes the type biggest
    with pytest.raises(L.LyricsError, match="rows is a whole number"):
        run(tl, lyr={"line": 1, "layout": {"kind": "flow", "rows": 0}})


def test_a_row_holding_a_bigger_word_stands_off_its_neighbour_by_the_plain_gap_so_rows_never_overlap():
    tl = timeline()
    lyr = {"line": 1, "style": "none", "spread": False, "layout": {"kind": "flow", "rows": 2}}   # words 1-2, then 3-4

    def gap(specs):
        """The clear space between the rows' ink (fake_measure: every box from -0.2 to 0.7 em), in the common em."""
        em = min(s["size"] for s in specs) / 0.7
        ink = [(s["offset"][1] - 0.45 * s["size"] / 0.7, s["offset"][1] + 0.45 * s["size"] / 0.7) for s in specs]
        return (min(lo for lo, _ in ink[:2]) - max(hi for _, hi in ink[2:])) / em
    plain, _ = run(tl, lyr=lyr, panel=(2.0, 1.0))
    assert gap(plain) == pytest.approx(1.5 * 0.7 - 0.9)                     # leading 1.5 caps less the ink's height
    for k in (1, 3):                                                         # the big word in the top row, then the bottom
        big, _ = run(tl, lyr=dict(lyr, punch={"words": [k], "scale": 2.0}), panel=(2.0, 1.0))
        assert big[k - 1]["size"] == pytest.approx(2.0 * big[k]["size"])          # twice its row neighbour's size
        assert gap(big) == pytest.approx(gap(plain))


# ------------------------------------------------------------------------------------------------------ carry
def test_a_word_that_began_before_from_lands_again_on_from_and_carry_still_skips_the_arrival():
    tl = timeline()
    land, summ = run(tl, lyr={"line": 1, "style": "pop", "from": 1.5, "spread": False})
    assert summ["carried"] == [[1, 1], [1, 2]]
    assert all("scale" in s["kinetic"] for s in land)
    still, summ = run(tl, lyr={"line": 1, "style": "pop", "from": 1.5, "spread": False, "carry": "still"})
    assert summ["carried"] == [[1, 1], [1, 2]]
    assert [("scale" in s["kinetic"]) for s in still] == [False, False, True, True]
    assert still[0]["kinetic"]["show"][0] == pytest.approx(1.5)
    with pytest.raises(L.LyricsError, match="carry is one of"):
        run(tl, lyr={"line": 1, "carry": "keep"})


# ------------------------------------------------------------------------------------------------------ zones
def zoned(**over):
    lyr = {"line": 1, "style": "none", "spread": False, **over}
    e = entry(**lyr)
    e["screen"] = {"anchor": "top"}
    return e


def test_a_line_is_set_again_in_every_shot_it_is_on_screen_in_with_the_words_that_were_up_at_the_cut():
    tl = timeline()
    specs, summ = run(tl, SHOTS, entry=zoned(zones={"b": {"screen": {"anchor": "bottom"}}}), panel=None,
                      panel_of=lambda e: (1.0, 0.5))
    by = {}
    for s in specs:
        by.setdefault(s["_shot"], []).append(s["lyric"][1])
    assert by == {"a": [1, 2, 3], "b": [1, 2, 3, 4]}                          # 1-3 were up at the cut and come along
    assert [s["name"] for s in specs] == [f"m_l1w{k}_a" for k in (1, 2, 3)] + [f"m_l1w{k}_b" for k in (1, 2, 3, 4)]
    a3 = next(s for s in specs if s["name"] == "m_l1w3_a")["kinetic"]["show"]
    b1 = next(s for s in specs if s["name"] == "m_l1w1_b")["kinetic"]["show"]
    assert round(a3[1] * FPS) == 60 and round(b1[0] * FPS) == 60              # hold to the cut, land again on its first frame
    assert next(s for s in specs if s["_shot"] == "a")["screen"] == {"anchor": "top"}
    assert next(s for s in specs if s["_shot"] == "b")["screen"] == {"anchor": "bottom"}
    assert summ["carried"] == [[1, 1], [1, 2], [1, 3]]
    assert [b["shot"] for b in summ["blocks"]] == ["a", "b"] and [b["words"] for b in summ["blocks"]] == [[1, 3], [1, 4]]
    assert summ["words"] == 7 and summ["cut_frame"] == 78 and no_text_in(summ, tl)


def test_each_zone_is_the_entry_with_its_overlay_laid_over_it_and_the_panel_is_asked_per_block():
    tl = timeline()
    asked = []

    def panel_of(e):
        asked.append((e["screen"]["anchor"], e["lyrics"].get("backing")))
        return (1.0, 0.5)

    e = zoned(zones={"b": {"screen": {"anchor": "bottom"}, "lyrics": {"backing": False}}}, backing={"color": "rose"})
    specs, _ = run(tl, SHOTS, entry=e, panel=None, panel_of=panel_of)
    assert asked == [("top", {"color": "rose"}), ("bottom", False)]
    assert all(s["backing"] == {"color": "rose"} for s in specs if s["_shot"] == "a")
    assert all(s["backing"] is False for s in specs if s["_shot"] == "b")


def test_a_carried_word_can_land_again_still_in_the_new_zone_and_the_last_block_leaves_by_the_lines_own_rule():
    tl = timeline()
    e = zoned(zones={"b": {"lyrics": {"carry": "still"}}}, to=4.0, leave="rewind", style="pop")
    specs, summ = run(tl, SHOTS, entry=e, panel_of=lambda e: (1.0, 0.5))
    a = [s for s in specs if s["_shot"] == "a"]
    b = {s["lyric"][1]: s for s in specs if s["_shot"] == "b"}
    assert all("scale" in s["kinetic"] for s in a)                             # the first shot's words arrive as usual
    assert all("scale" not in b[k]["kinetic"] for k in (1, 2, 3)) and "scale" in b[4]["kinetic"]
    assert all("dxp" not in s["kinetic"] for s in a)                           # no rewind before the line is over
    assert all("dxp" in s["kinetic"] for s in b.values()) and summ["rewind_from"] is not None


def test_zones_that_do_not_hold_together_are_errors_naming_numbers_and_shots():
    tl = timeline()
    with pytest.raises(L.LyricsError, match="the shot 'z', which the project does not cut to"):
        run(tl, SHOTS, entry=zoned(zones={"z": {}}))
    with pytest.raises(L.LyricsError, match="one `line` shot by shot"):
        run(tl, SHOTS, entry={**zoned(zones={"a": {}}), "lyrics": {"timeline": "t.json", "lines": [1, 2], "zones": {"a": {}}}})
    with pytest.raises(L.LyricsError, match="need the project's \\[\\[shot\\]\\]s"):
        run(tl, None, entry=zoned(zones={"a": {}}))
    with pytest.raises(L.LyricsError, match="table of overrides"):
        run(tl, SHOTS, entry=zoned(zones={"a": 3}))
    with pytest.raises(L.LyricsError, match="no word of line 3 is on screen"):
        run(tl, SHOTS[:1], entry=entry(line=3, style="none", zones={"a": {}}))


def test_a_word_that_would_flash_for_one_frame_before_a_cut_is_carried_not_shown_in_the_shot():
    tl = timeline()
    tl["lines"][0]["words"][3]["start"] = 1.99                                # lands on frame 60 - 0 ... i.e. 1 frame before 2.0 s?
    shots = [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}]
    specs, summ = run(tl, shots, entry=zoned(zones={"a": {}}), panel_of=lambda e: (1.0, 0.5))
    last_land = L.landing_frame(1.99, FPS)
    assert 60 - last_land < 2
    assert [s["lyric"][1] for s in specs if s["_shot"] == "a"] == [1, 2, 3]
    assert [s["lyric"][1] for s in specs if s["_shot"] == "b"] == [1, 2, 3, 4]
    assert round(next(s for s in specs if s["name"] == "m_l1w4_b")["kinetic"]["show"][0] * FPS) == 60


def test_zone_objects_carry_the_numbers_of_their_word_and_no_text_but_their_own():
    tl = timeline()
    specs, summ = run(tl, SHOTS, entry=zoned(zones={"b": {"screen": {"anchor": "bottom"}}}), panel_of=lambda e: (1.0, 0.5))
    for s in specs:
        assert s["text"] in words_of(tl)
        rest = {k: v for k, v in s.items() if k != "text"}
        assert no_text_in(rest, tl), rest["name"]
        assert FX.kinetic_spec(s)["show"]


# ------------------------------------------------------------------------------------------------------ screen type
def test_screen_words_are_stacked_in_depth_so_strips_never_fight():
    tl = timeline()
    e = entry(line=1, style="none")
    e["screen"] = True
    specs, _ = L.expand(e, tl, fps=FPS, frame0=181, panel=(1.0, 0.5), measure=fake_measure)
    lifts = [s["lift"] for s in specs]
    assert lifts == sorted(lifts) and len(set(lifts)) == len(lifts) and lifts[0] > 0
    plain, _ = run(tl, lyr={"line": 1, "style": "none"})
    assert all("lift" not in s for s in plain)                              # type on a surface keeps its own lift


# ------------------------------------------------------------------------------------------------------ strips in the layout
def test_the_flow_leaves_room_for_the_strip_of_tape_behind_every_word():
    tl = timeline()
    em = 0.05 / 0.7                                                                   # `size` over the fake font's cap height
    bare, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "layout": "flow"}, panel=(4.0, 2.0), extra={"size": 0.05})
    tape, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "layout": "flow", "backing": {"color": "rose"}},
                  panel=(4.0, 2.0), extra={"size": 0.05})
    step = lambda specs: specs[1]["offset"][0] - specs[0]["offset"][0]            # noqa: E731
    assert step(tape) - step(bare) == pytest.approx(2 * 0.55 * em)                  # the default pad, each side of a word
    punch = {"words": [2], "backing": False}
    mixed, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "layout": "flow", "backing": {"color": "rose"},
                            "punch": punch}, panel=(4.0, 2.0), extra={"size": 0.05})
    assert step(mixed) - step(bare) == pytest.approx(0.55 * em)                     # only the first word's side has a strip
    tall, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "layout": "flow", "backing": {"height": 2.0}},
                  panel=(4.0, 2.0), extra={"size": 0.05})
    assert all(s["backing"] == {"height": 2.0} for s in tall)


def test_rows_of_strips_are_never_closer_than_the_tallest_strip_and_rows_of_bare_words_keep_their_leading():
    tl = timeline()
    em = 0.05 / 0.7
    kw = dict(panel=(4.0, 2.0), extra={"size": 0.05})
    lyr = {"line": 1, "style": "none", "spread": False, "layout": {"kind": "flow", "rows": 2}}
    tape, _ = run(tl, lyr={**lyr, "backing": {"color": "rose"}}, **kw)
    v = sorted({round(s["offset"][1], 9) for s in tape})
    assert v[1] - v[0] == pytest.approx(1.05 * (0.9 + 2 * 0.3) * em)       # the fake font's ink is 0.9 em; the strip adds 0.3 each side
    bare, _ = run(tl, lyr=lyr, **kw)
    v = sorted({round(s["offset"][1], 9) for s in bare})
    assert v[1] - v[0] == pytest.approx(1.5 * 0.7 * em)                    # leading x cap height, as before


# ------------------------------------------------------------------------------------------------------ punch: top
def spiky(tl):
    """Word levels of line 1: word 4 the loudest, then word 2, the rest quiet."""
    db = [-20.0] * int(10 * FPS)
    for a, b, v in ((66, 78, -5.0), (37, 45, -8.0)):
        db[a:b] = [v] * (b - a)
    tl["vocal_db"] = db
    return tl


def test_punch_top_takes_the_loudest_words_of_the_selection():
    tl = spiky(timeline())
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "punch": {"top": 2, "color": "gold"}})
    assert summ["punch"] == [[1, 2], [1, 4]] and [s["color"] for s in specs if s.get("color")] == ["gold", "gold"]
    _, summ = run(tl, lyr={"line": 1, "style": "none", "punch": {"top": 1}})
    assert summ["punch"] == [[1, 4]]
    with pytest.raises(L.LyricsError, match="punch top is a whole number"):
        run(tl, lyr={"line": 1, "punch": {"top": 0}})


def test_a_punch_word_is_the_same_word_in_every_shot_of_a_zoned_line():
    tl = spiky(timeline())
    e = zoned(punch={"top": 2, "color": "gold"}, zones={"b": {}})
    specs, summ = run(tl, SHOTS, entry=e, panel_of=lambda e: (1.0, 0.5))
    gold = {sh: sorted(s["lyric"][1] for s in specs if s["_shot"] == sh and s.get("color") == "gold") for sh in "ab"}
    assert gold == {"a": [2], "b": [2, 4]}                                    # word 4 is not in shot a: it is not punch there either
    assert summ["punch"] == [[1, 2], [1, 4]]


def test_a_zone_lays_out_the_whole_line_so_words_stay_where_they_are_when_the_band_does_not_change():
    tl = timeline()
    e = zoned(layout="flow", zones={"b": {}})
    specs, _ = run(tl, SHOTS, entry=e, panel_of=lambda ee: (1.0, 0.5))
    by = {(s["_shot"], s["lyric"][1]): s["offset"] for s in specs}
    for k in (1, 2, 3):
        assert by[("a", k)] == by[("b", k)]                     # same band, same line: a word does not move at the cut
    assert ("b", 4) in by and ("a", 4) not in by                # word 4 only lands in shot b, where its place was always reserved


def test_flow_rows_can_sit_against_the_left_or_right_margin_of_the_panel():
    tl = timeline()
    em = 0.05 / 0.7
    ws = L.select(tl, line=1)
    half = lambda k: 0.55 * 2 * (1 + L.spread_max(ws[k].start, ws[k].voiced_end)) * em / 2     # noqa: E731 - a word's half width
    panel, extra = (2.0, 1.0), {"size": 0.05, "fit": 1.0}                           # all of it is room: 1.0 each side
    base = {"line": 1, "style": "none", "spread": False}
    left, _ = run(tl, lyr={**base, "layout": {"kind": "flow", "rows": 2, "align": "left"}}, panel=panel, extra=extra)
    for k in (0, 2):                                                                # the first word of each row
        assert left[k]["offset"][0] - half(k) == pytest.approx(-1.0)
    right, _ = run(tl, lyr={**base, "layout": {"kind": "flow", "rows": 2, "align": "right"}}, panel=panel, extra=extra)
    for k in (1, 3):                                                                # the last word of each row
        assert right[k]["offset"][0] + half(k) == pytest.approx(1.0)
    centre, _ = run(tl, lyr={**base, "layout": {"kind": "flow", "rows": 2}}, panel=panel, extra=extra)
    assert centre[0]["offset"][0] < 0.0 < centre[1]["offset"][0]                    # centred rows straddle the middle
    with pytest.raises(L.LyricsError, match="flow align is left, center or right"):
        run(tl, lyr={**base, "layout": {"kind": "flow", "align": "middle"}})


def test_rows_follow_the_side_of_the_screen_band_unless_the_layout_says_otherwise():
    tl = timeline()
    panel, base = (2.0, 1.0), {"line": 1, "style": "none", "spread": False, "layout": {"kind": "flow", "rows": 2}}
    on_left = run(tl, entry={**entry(**base), "size": 0.05, "fit": 1.0, "screen": {"side": "left"}}, panel=panel)[0]
    assert on_left[0]["offset"][0] < -0.9 + 0.1                                      # rows start at the left margin
    on_right = run(tl, entry={**entry(**base), "size": 0.05, "fit": 1.0, "screen": {"side": "right"}}, panel=panel)[0]
    assert on_right[1]["offset"][0] > 0.9 - 0.1
    own = {**base, "layout": {"kind": "flow", "rows": 2, "align": "center"}}
    forced = run(tl, entry={**entry(**own), "size": 0.05, "fit": 1.0, "screen": {"side": "left"}}, panel=panel)[0]
    assert forced[0]["offset"][0] > on_left[0]["offset"][0] + 0.2                      # the layout's own `align` wins


def test_knockout_type_has_no_strip_and_no_ring_whatever_the_entry_or_the_punch_words_ask_for():
    tl = timeline()
    lyr = {"line": 1, "style": "none", "layout": "flow", "backing": {"color": "rose"},
           "punch": {"top": 1, "backing": {"color": "gold"}, "outline": {"color": "base"}}}
    e = entry(**lyr)
    e["outline"] = {"color": "text"}
    e["knockout"] = True
    specs, _ = run(tl, entry=e)
    assert all("backing" not in s and "outline" not in s for s in specs) and all(s["knockout"] for s in specs)
    plain = entry(**lyr)
    specs, _ = run(tl, entry=plain)
    assert any("backing" in s for s in specs)
    bare, _ = run(tl, entry={**e, "knockout": False}, panel=(1.0, 0.5))
    assert any("backing" in s for s in bare)                                     # `knockout = false` is not knockout type
    # and the layout leaves no room for strips that are not there
    narrow = run(tl, entry={**entry(**{**lyr, "punch": {"top": 1}}), "knockout": True, "size": 0.05, "fit": 1.0},
                 panel=(4.0, 2.0))[0]
    wide = run(tl, entry={**entry(**{**lyr, "punch": {"top": 1}}), "size": 0.05, "fit": 1.0}, panel=(4.0, 2.0))[0]
    assert narrow[1]["offset"][0] - narrow[0]["offset"][0] < wide[1]["offset"][0] - wide[0]["offset"][0]


# ------------------------------------------------------------------------------------------------------ write
def test_write_takes_the_words_held_long_and_leaves_the_punch_words_alone():
    tl = timeline()
    specs, summ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "punch": {"words": [4]},
                               "write": {"min_hold": 0.29, "reveal": True, "color": "love"}})
    assert summ["write"] == [[1, 3]] and summ["punch"] == [[1, 4]]                 # word 4 is punch; word 3 (0.30 s) is held
    w3 = next(s for s in specs if s["lyric"] == [1, 3])
    assert w3["color"] == "love" and w3["reveal"]["from"] == pytest.approx(w3["kinetic"]["show"][0])
    assert w3["reveal"]["to"] == pytest.approx(L.select(tl, line=1)[2].voiced_end)       # written over the note
    assert all("reveal" not in s for s in specs if s["lyric"] != [1, 3])


def test_write_and_punch_keep_their_own_looks_and_arrivals():
    tl = timeline()
    specs, _ = run(tl, lyr={"line": 1, "style": "none", "spread": False, "punch": {"words": [1], "style": "slap"},
                            "write": {"min_hold": 0.29, "style": "slide", "scale": 2.0}, "layout": "flow"})
    by = {s["lyric"][1]: s for s in specs}
    assert "scale" in by[1]["kinetic"] and "dxp" not in by[1]["kinetic"]                  # punch: slap
    assert "dxp" in by[3]["kinetic"] and "scale" not in by[3]["kinetic"]                  # write: slide
    assert by[3]["size"] == pytest.approx(2.0 * by[2]["size"])
    for bad, frag in (({"scale": 0}, "write scale must be positive"), ({"style": "rise"}, "write word cannot arrive as steam"),
                      ({"min_hold": "long"}, "write min_hold"), ({"colour": "x"}, "write keys \\['colour'\\]")):
        with pytest.raises(L.LyricsError, match=frag):
            run(tl, lyr={"line": 1, "write": bad})


# ------------------------------------------------------------------------------------------------------ the documented example
def test_the_example_in_the_docs_resolves_and_expands_for_both_outputs():
    """docs/design.md: Cassette type: a base that is never built, a line that extends it, zones per shot, 9:16 touches."""
    doc = (Path(__file__).resolve().parent.parent / "docs/design.md").read_text(encoding="utf-8")
    block = re.search(r"```toml\n(.*?)```", doc[doc.index("### Cassette type"):], re.S).group(1)
    entries = SR.resolve_entries(tomllib.loads(block)["text"])
    assert [e["name"] for e in entries] == ["ly3"]
    shots = [{"name": "g1_hero", "from": 0.0, "to": 7.0}, {"name": "rin_cu", "from": 7.0, "to": 8.7},
             {"name": "tunnel_in", "from": 8.7, "to": 11.57}]
    sizes = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
    tl = timeline()
    seen = {}
    for out, e in SR.per_output(entries[0], list(sizes)):
        e = {**e, "_aspect": out}
        asked = []

        def panel_of(ee, out=out, asked=asked):
            asked.append(ee["screen"])
            return SR.panel(SR.screen_spec(ee), sizes[out], ee.get("at"), ee.get("box"))[1]

        specs, summ = L.expand(e, tl, fps=FPS, frame0=181, panel=None, measure=lambda s, *a: fake_measure(s), shots=shots,
                               panel_of=panel_of)
        seen[out] = asked
        assert {s["_shot"] for s in specs} == {"rin_cu", "tunnel_in"} and all(s["name"].endswith(s["_shot"]) for s in specs)
        assert [b["shot"] for b in summ["blocks"]] == ["rin_cu", "tunnel_in"] and no_text_in(summ, tl)
        assert all(s["kinetic"]["show"] for s in specs) and all(s["knockout"] is not True for s in specs if "knockout" in s)
    assert [a["anchor"] for a in seen["16x9"]] == ["bottom", "top"] and [a["side"] for a in seen["16x9"]] == ["center", "left"]
    assert [a["anchor"] for a in seen["9x16"]] == ["bottom", "bottom"] and seen["9x16"][1]["height"] == 0.3     # aspect, then zone


def test_every_key_screen_and_zone_words_emit_is_one_the_text_stage_accepts():
    src = (Path(__file__).resolve().parent.parent / "mkmmd/blender/build/text.py").read_text(encoding="utf-8")
    known = _literal_known(src)
    tl = timeline()
    e = zoned(punch={"top": 1, "font": "b"}, write={"min_hold": 0.29, "reveal": True}, wow={}, leave="rewind", to=4.0,
              zones={"b": {"knockout": True}}, layout={"kind": "flow", "rows": "auto"}, backing={"color": "rose"})
    specs, _ = run(tl, SHOTS, entry=e, panel_of=lambda ee: (1.0, 0.5), measure=lambda s, *a: fake_measure(s))
    emitted = set().union(*(set(s) for s in specs))
    assert emitted <= known, emitted - known
    assert {"screen", "knockout", "_shot", "reveal", "lift"} <= emitted


def _literal_known(src):
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KNOWN" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("KNOWN")
