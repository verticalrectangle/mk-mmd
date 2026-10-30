"""A bass line on a four-string neck (mkmmd.core.fretting): strings numbered on any neck, hovering fingers, the notes of a
line read from times or a hits lane, each placed on a string and fret with the least hand travel (one finger per fret, open
strings allowed), and the hand's keyed shapes; pure maths on a synthetic short-scale card."""
import pytest

from mkmmd.core import fretting as FR
from mkmmd.core import tables as TB

SCALE = 0.762
TUNING = [28, 33, 38, 43]                                          # E1 A1 D2 G2


def bass_entry(nfrets=19, board_y=-0.0115, tuning=True):
    """A four-string neck entry in the contract's frame: the saddle line at z = 0, +z toward the nut, the board facing -y, the
    low E at -x."""
    z = lambda n: SCALE * 2.0 ** (-n / 12.0)                       # noqa: E731
    nut_x = [(i - 1.5) * 0.0106 for i in range(4)]
    br_x = [(i - 1.5) * 0.0177 for i in range(4)]
    radius = [0.00133, 0.00108, 0.00083, 0.00057]
    e = {"name": "neck", "type": "neck", "frame": {"along": [0, 0, 1], "across": [1, 0, 0], "normal": [0, -1, 0]},
         "thumb": [0, 1, 0], "scale": SCALE, "frets": [[0.0, board_y, z(n)] for n in range(nfrets + 1)],
         "strings": [{"name": "EADG"[i], "nut": [nut_x[i], board_y - 0.003, z(0)], "bridge": [br_x[i], board_y - 0.007, -0.006],
                      "radius": radius[i]} for i in range(4)],
         "section": {"width": [0.0413, 0.057], "depth": [0.021, 0.024], "p": 2.6}, "fret_height": 0.0011}
    if tuning:
        e["tuning"] = list(TUNING)
    return e


# ------------------------------------------------------------------------------------------------ strings on any neck
def test_strings_are_numbered_from_the_highest_on_a_four_string_neck():
    e = bass_entry()
    assert FR.string_entry(e, 4)["name"] == "E" and FR.string_entry(e, 1)["name"] == "G"
    p = FR.string_point(e, 4, 0.5)
    assert p[0] < FR.string_point(e, 1, 0.5)[0]                    # the low E on -x
    for bad in (0, 5):
        with pytest.raises(FR.FrettingError, match="not on the neck"):
            FR.string_point(e, bad, 0.5)


def test_a_six_string_chord_is_refused_on_a_bass_and_tables_are_checked_against_its_strings():
    assert FR.shape({"index": [4, 0]}, strings=4) == {"index": (4, 0)}
    with pytest.raises(FR.FrettingError, match="six strings"):
        FR.shape("power", strings=4)
    with pytest.raises(FR.FrettingError, match="1..4"):
        FR.shape({"index": [5, 0]}, strings=4)
    with pytest.raises(FR.FrettingError):
        FR.finger_targets(bass_entry(), "power", 3)
    assert FR.shape("power") == FR.CHORDS["power"]                  # six strings stay the default


def test_a_hovering_finger_stays_over_its_string_without_pressing_it():
    e = bass_entry()
    assert FR.hovering({"ring": [1, 2, "hover"], "index": [2, 0]}) == {"ring"} and FR.hovering("power") == set()
    assert FR.shape({"ring": [1, 2, "hover"], "index": [1, 0]}, 4) == {"ring": (1, 2), "index": (1, 0)}   # may share a string
    for bad in ({"ring": [1, 2, "press"]}, {"ring": [1, 2, "hover", 1]}, {"index": [1, 0], "ring": [1, 2]}):
        with pytest.raises(FR.FrettingError):
            FR.shape(bad, 4)
    pressed = FR.finger_targets(e, {"ring": [1, 2]}, 2)[0]
    hover = FR.finger_targets(e, {"ring": [1, 2, "hover"]}, 2)[0]
    assert not pressed["hover"] and hover["hover"] and hover["fret"] == pressed["fret"] == 4
    axis = FR.string_point(e, 1, hover["point"][2])
    r = FR.string_entry(e, 1)["radius"]
    assert hover["point"][1] == pytest.approx(axis[1] - (r + FR.HOVER))      # over the string as it lies (-y is out)
    assert hover["point"][1] < pressed["point"][1] - FR.HOVER                # well clear of where a pressed string lies
    assert hover["point"][2] == pytest.approx(pressed["point"][2]) and hover["lo"] == pressed["lo"]
    prop, _frame = FR.solver_prop(e, {"ring": [1, 2, "hover"]}, 2)           # the solver sees a target in the air
    assert prop["targets"][0]["point"][2] > FR.FRET_HEIGHT + r + FR.HOVER


# ------------------------------------------------------------------------------------------------ the notes of a line
def test_notes_come_from_times_or_a_lane_inside_the_window():
    times = {"times": [[2.0, 45.2], [1.0, 43.0], [3.0, 47.6]], "from": 1.5}
    assert FR.notes_of(times) == [(2.0, 45), (3.0, 48)]                       # sorted, windowed, rounded to semitones
    assert FR.notes_of(dict(times, octave=1)) == [(2.0, 57), (3.0, 60)]
    hits = {"bass_notes": [{"t": 1.0, "db": -3.0, "midi": 40.2}, {"t": 2.0, "db": -20.0, "midi": 41.0},
                           {"t": 3.0, "db": -1.0, "midi": 45.8}], "kick": [{"t": 1.0, "db": 0.0}]}
    assert FR.notes_of({"lane": "bass_notes"}, hits) == [(1.0, 40), (2.0, 41), (3.0, 46)]
    assert FR.notes_of({"lane": "bass_notes", "min_db": -6, "to": 2.5}, hits) == [(1.0, 40)]
    assert FR.notes_of({"lane": "bass_notes", "from": 1.0, "to": 1.0}, hits) == [(1.0, 40)]   # the window is inclusive


def test_notes_say_what_is_wrong():
    hits = {"kick": [{"t": 1.0, "db": 0.0}], "bass_notes": [{"t": 1.0, "midi": 40}]}
    cases = [({"lane": "bass_notes", "times": [[1, 40]]}, "one of them"), ({}, "one of them"),
             ({"lane": "bass_notes"}, None), ({"lane": "snare"}, "no lane"), ({"lane": "kick"}, "no pitch"),
             ({"times": [[1.0]]}, "expected"), ({"times": [[1.0, 40]], "octave": 0.5}, "whole number"),
             ({"times": [[1.0, 40]], "to": 0.5}, "no note"), ({"times": [[1.0, 40]], "bogus": 1}, "unknown key")]
    for spec, msg in cases:
        with pytest.raises(FR.FrettingError, match=msg or "hits file"):
            FR.notes_of(spec, None if msg is None else hits)
    with pytest.raises(FR.FrettingError, match="hits file"):
        FR.notes_of({"lane": "bass_notes"}, None)


def test_the_tuning_comes_from_the_table_or_the_card():
    e = bass_entry()
    assert FR.tuning_of({}, e) == TUNING
    assert FR.tuning_of({"tuning": [26, 33, 38, 43]}, e) == [26, 33, 38, 43]           # a drop D
    with pytest.raises(FR.FrettingError, match="give `tuning`"):
        FR.tuning_of({}, bass_entry(tuning=False))
    with pytest.raises(FR.FrettingError, match="the neck 4"):
        FR.tuning_of({"tuning": [40, 45, 50, 55, 59, 64]}, e)


# ------------------------------------------------------------------------------------------------ where each note is played
def frets():
    return FR.fret_z(bass_entry())


def test_a_line_in_one_hand_position_never_shifts():
    """A A B B C B A Bb B C: the G string at the second position (index on fret 2), one finger per fret."""
    line = [45, 45, 47, 47, 48, 47, 45, 46, 47, 48]
    placed = FR.place([(0.2 * i, m) for i, m in enumerate(line)], TUNING, frets())
    assert {n["position"] for n in placed} == {2} and {n["string"] for n in placed} == {1}
    assert [n["fret"] for n in placed] == [2, 2, 4, 4, 5, 4, 2, 3, 4, 5]
    assert [n["finger"] for n in placed][:5] == ["index", "index", "ring", "ring", "little"]
    for n in placed:                                                         # one finger per fret: finger = fret - position
        assert FR.FINGERS.index(n["finger"]) == n["fret"] - n["position"]


def test_the_places_minimise_hand_travel_over_the_whole_line():
    """E1 G1 A1 C2: frets 3, 5 and 8 of the E string would shift; the line stays in one position instead (the low E open,
    frets 3 and 5 of the E string, the C on the A string)."""
    placed = FR.place([(0.0, 28), (0.5, 31), (1.0, 33), (1.5, 36)], TUNING, frets())
    assert len({n["position"] for n in placed}) == 1                         # no shift
    assert [(n["string"], n["fret"]) for n in placed] == [(4, 0), (4, 3), (4, 5), (3, 3)]
    high = FR.place([(0.0, 55), (0.4, 57), (0.8, 58)], TUNING, frets())       # G3 A3 Bb3: high on the G string
    assert all(n["string"] == 1 for n in high) and [n["fret"] for n in high] == [12, 14, 15]
    assert {n["position"] for n in high} == {12} and high[0]["finger"] == "index" and high[2]["finger"] == "little"
    far = FR.place([(0.0, 55), (0.4, 59)], TUNING, frets())                   # G3 B3: 12 -> 16 on the G string would shift ...
    assert far[0]["position"] == far[1]["position"] and [(n["string"], n["fret"]) for n in far] == [(2, 17), (1, 16)]


def test_open_strings_are_used_when_they_save_a_shift_and_fretted_notes_otherwise():
    with_open = FR.place([(0.0, 62), (0.3, 43), (0.6, 62)], TUNING, frets())   # D4 G2 D4: the G open between two fret-19 notes
    assert with_open[1]["fret"] == 0 and with_open[1]["finger"] is None and with_open[1]["string"] == 1
    assert with_open[0]["position"] == with_open[1]["position"] == with_open[2]["position"]
    alone = FR.place([(0.0, 45), (0.3, 43), (0.6, 45)], TUNING, frets())       # A2 G2 A2: G on the D string at the 5th fret
    assert alone[1]["fret"] == 5 and alone[1]["string"] == 2


def test_notes_off_the_neck_are_named():
    with pytest.raises(FR.FrettingError, match=r"t 1\.000 \(MIDI 20\).*octave"):
        FR.place([(0.0, 40), (1.0, 20)], TUNING, frets())
    with pytest.raises(FR.FrettingError, match="MIDI 28..62"):
        FR.place([(0.0, 70)], TUNING, frets())


# ------------------------------------------------------------------------------------------------ the hand's shapes
def test_hand_keys_change_on_every_new_note_and_arrive_before_it():
    placed = FR.place([(1.0, 45), (1.12, 47), (1.34, 47), (2.0, 48)], TUNING, frets())
    keys = FR.hand_keys(placed, move=0.12)
    assert [k["t"] for k in keys] == [1.0, 1.12, 2.0]                        # the repeated B makes no key
    assert keys[0]["chord"] == {"index": [1, 0]} and keys[1]["chord"] == {"ring": [1, 2]} and keys[2]["chord"] == {"little": [1, 3]}
    assert keys[1]["move"] == pytest.approx(FR.MOVE_SHARE * 0.12)             # cut to the time since the last note
    assert keys[2]["move"] == pytest.approx(0.12)                             # (the repeated B counts as that note)
    assert all(k["fret"] == 2 for k in keys)
    for k in keys:                                                           # every key is a shape the solver takes
        FR.finger_targets(bass_entry(), k["chord"], k["fret"])


def test_an_open_string_lifts_the_finger_that_played_the_note_before():
    placed = FR.place([(0.0, 62), (0.3, 43), (0.6, 62)], TUNING, frets())
    keys = FR.hand_keys(placed)
    lifted = keys[1]["chord"]
    (finger, spec), = lifted.items()
    assert spec[2] == "hover" and (spec[0], spec[1] + keys[1]["fret"]) == (placed[0]["string"], placed[0]["fret"])
    assert FR.hovering(lifted) == {finger} and keys[2]["chord"] == keys[0]["chord"]
    targets = FR.finger_targets(bass_entry(), lifted, keys[1]["fret"])
    assert targets[0]["hover"]
    first_open = FR.hand_keys(FR.place([(0.0, 28), (0.3, 33)], TUNING, frets()))     # the low E only sounds open
    assert first_open[0]["chord"] == {"index": [4, 0, "hover"]}               # nothing before it: the index over it


def test_a_change_eases_the_fingers_off_the_strings_and_lands_on_the_note():
    """A keyed change passes through both shapes eased off the strings: every finger of a shape hovering over its place
    (`lifted`), the fingers lifting in the first LIFT of the move and landing in the last, never before the settle ends."""
    assert FR.lifted("power") == {"index": [6, 0, "hover"], "ring": [5, 2, "hover"], "little": [4, 2, "hover"]}
    assert FR.lifted({"middle": [3, 1, "hover"]}, 4) == {"middle": [3, 1, "hover"]}            # already off its string
    press, = FR.finger_targets(bass_entry(), {"middle": [3, 1]}, 7)
    off, = FR.finger_targets(bass_entry(), FR.lifted({"middle": [3, 1]}, 4), 7)
    assert off["hover"] and not press["hover"] and off["fret"] == press["fret"]
    assert -(off["point"][1] - press["point"][1]) > FR.HOVER                                   # over the string, not on it
    change = [(100.0, 4.0, "B", "b", [(FR.LIFT, "A~", "a~"), (1.0 - FR.LIFT, "B~", "b~")])]
    assert FR.change_keys("A", change, 50, 0) == [(50, "A"), (96.0, "A"), (97.0, "A~"), (99.0, "B~"), (100.0, "B")]
    assert [v for _f, v in FR.change_keys("a", change, 50, 1)] == ["a", "a", "a~", "b~", "b"]
    early = FR.change_keys("A", [(52.0, 4.0, "B", "b", [(0.25, "X", "x"), (0.75, "Y", "y")])], 50, 0)
    assert [f for f, _v in early] == [50, 50.5, 50.875, 51.625, 52.0]                          # the move starts after the settle
    assert FR.change_keys("A", [(100.0, 3.0, "B", "b", [])], 50, 0) == [(50, "A"), (97.0, "A"), (100.0, "B")]


def test_line_places_a_lane_on_the_card_and_the_hand_table_takes_notes():
    hits = {"bass_notes": [{"t": 24.455, "db": -5.4, "midi": 45.0}, {"t": 24.638, "db": -2.3, "midi": 47.1},
                           {"t": 25.518, "db": -4.1, "midi": 48.1}, {"t": 30.0, "db": 0.0, "midi": 28.0}]}
    placed, keys = FR.line({"lane": "bass_notes", "from": 24.42, "to": 28.84}, bass_entry(), hits)
    assert [n["midi"] for n in placed] == [45, 47, 48] and len(keys) == 3
    TB.check({"grip": "bass:neck", "notes": {"lane": "bass_notes", "from": 1, "to": 2, "tuning": TUNING, "octave": 0,
                                             "min_db": -12}}, TB.HAND, "[pose.rin.hands.L]")
    with pytest.raises(TB.TableError):
        TB.check({"grip": "bass:neck", "notes": {"lanes": "x"}}, TB.HAND, "[pose.rin.hands.L]")


def test_a_position_keeps_the_line_up_the_neck_with_the_least_travel_inside_it():
    """The riff (A2 B2 C3 Bb2 ...) at `position = 7`: the D string from the 7th fret, one finger per fret, no shift; a note the
    hand cannot reach from there up is named; open strings stay allowed (the hand does not move for them)."""
    riff = [(0.0, 45), (0.2, 47), (0.4, 47), (1.0, 48), (1.2, 47), (1.6, 45), (2.4, 46), (3.0, 47), (3.6, 48)]
    placed = FR.place(riff, TUNING, frets(), position=7)
    assert {n["position"] for n in placed} == {7} and {n["string"] for n in placed} == {2}
    assert [n["fret"] for n in placed[:4]] == [7, 9, 9, 10] and placed[3]["finger"] == "little"
    low = FR.place(riff, TUNING, frets())                                         # without it: the G string at the 2nd fret
    assert {n["position"] for n in low} == {2}
    placed, keys = FR.line({"times": [list(n) for n in riff], "position": 7}, bass_entry())
    assert all(k["fret"] >= 7 for k in keys) and keys[0]["chord"] == {"index": [2, 0]}
    every = FR.place([(0.0, 36), (0.5, 48), (1.0, 43), (1.5, 40)], TUNING, frets(), position=7)    # C2 C3 G2 (open) E2
    assert all(n["fret"] == 0 or n["fret"] >= 7 for n in every) and every[2]["fret"] in (0, 10)
    assert all(n["position"] >= 7 for n in every)
    with pytest.raises(FR.FrettingError, match=r"position 7 up: t 0\.000 \(MIDI 34\)"):
        FR.place([(0.0, 34)], TUNING, frets(), position=7)                       # Bb1: E string fret 6 only
    for bad in (0, 20):
        with pytest.raises(FR.FrettingError, match="a fret of the neck"):
            FR.place(riff, TUNING, frets(), position=bad)
    with pytest.raises(FR.FrettingError, match="whole fret number"):
        FR.line({"times": [[0.0, 45]], "position": 7.5}, bass_entry())
