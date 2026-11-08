"""mkmmd.timeline.tokens on synthetic tokens and vocals: invented words only."""
import numpy as np
import pytest

from mkmmd.timeline import tokens as T

HOP = 0.01


def ev_of(sung_spans, onsets, length=12.0, midi=None, quiet=()):
    """Evidence with the voice sung over `sung_spans` [(a, b)], quiet (-20 dB under, still voiced) over `quiet`."""
    n = int(length / HOP)
    rel = np.full(n, -60.0)
    voiced = np.zeros(n, bool)
    for a, b in sung_spans:
        rel[int(a / HOP):int(b / HOP)] = -5.0
        voiced[int(a / HOP):int(b / HOP)] = True
    for a, b, db in quiet:
        rel[int(a / HOP):int(b / HOP)] = db
        voiced[int(a / HOP):int(b / HOP)] = True
    return T.Evidence(0.0, HOP, rel, voiced, [(t, 4.0) for t in onsets], midi)


def flat(lines):
    return [w["start"] for ln in lines for w in ln["words"]]


def test_match_takes_the_occurrence_that_holds_every_line_in_order():
    words = ["zib", "vor", "kal", "mup"]
    tokens = [("zib", 1.0, 1.3), ("vor", 1.3, 1.6),                       # an earlier repeat of the first line only
              ("dree", 2.0, 2.4),
              ("zib", 4.0, 4.3), ("vor", 4.3, 4.6), ("kal", 4.6, 5.0), ("mup", 5.0, 5.4)]
    m = T.match(tokens, words)
    assert [m[k][0] for k in range(4)] == [3, 4, 5, 6]
    assert all(m[k][1] == 1.0 for k in range(4))


def test_match_prefers_the_repeat_that_is_sung():
    words = ["zib", "vor", "kal"]
    tokens = [("zib", 1.0, 1.3), ("vor", 1.3, 1.6), ("kal", 1.6, 2.0),   # heard in silence (a hallucination)
              ("zib", 6.0, 6.3), ("vor", 6.3, 6.6), ("kal", 6.6, 7.0)]
    ev = ev_of([(5.9, 7.2)], [6.0, 6.3, 6.6])
    m = T.match(tokens, words, sung=ev.sung_share)
    assert [m[k][0] for k in range(3)] == [3, 4, 5]


def test_words_land_on_onsets_in_order_and_apart():
    lines = [["zib", "vor", "kal"], ["mup", "dree"]]
    onsets = [2.0, 2.5, 3.1, 4.0, 4.6]
    tokens = [("zib", 2.05, 2.48), ("vor", 2.48, 3.08), ("kal", 3.08, 3.6), ("mup", 4.1, 4.58), ("dree", 4.58, 5.2)]
    out, stats = T.word_lines(lines, tokens, ev_of([(2.0, 5.5)], onsets), 10.0)
    on = flat(out)
    assert on == pytest.approx(onsets, abs=0.011)
    assert all(b - a >= 0.12 for a, b in zip(on, on[1:]))
    assert stats["anchored"] == 5 and stats["unanchored"] == 0
    assert out[0]["words"][0]["end"] == pytest.approx(on[1], abs=1e-6)              # a word ends where the next starts


def test_a_word_without_token_falls_on_the_onset_between_its_neighbours():
    lines = [["zib", "vor", "kal", "mup"]]
    onsets = [1.0, 1.6, 2.3, 3.0]
    tokens = [("zib", 1.02, 1.58), ("kal", 2.32, 2.98), ("mup", 2.98, 3.5)]       # "vor" never heard
    out, stats = T.word_lines(lines, tokens, ev_of([(1.0, 3.8)], onsets), 10.0)
    assert flat(out) == pytest.approx(onsets, abs=0.011)
    assert stats["unanchored"] == 1


def test_a_word_without_token_or_onset_still_gets_a_place_between_its_neighbours():
    lines = [["zib", "vor", "kal"]]
    onsets = [1.0, 3.0]                                                              # nothing for the middle word
    tokens = [("zib", 1.0, 1.5), ("kal", 3.0, 3.5)]
    on = flat(T.word_lines(lines, tokens, ev_of([(1.0, 3.8)], onsets), 10.0)[0])
    assert on[0] == pytest.approx(1.0, abs=0.011) and on[2] == pytest.approx(3.0, abs=0.011)
    assert 1.1 <= on[1] <= 2.9


def test_boundary_words_a_late_phrase_start_and_a_last_word_held_to_the_end_of_the_voice():
    lines = [["zib", "vor"], ["kal", "mup"]]
    onsets = [1.0, 1.5, 4.0, 4.6]
    # Whisper starts the second phrase 0.6 s late but ends its words on the next onset
    tokens = [("zib", 1.0, 1.5), ("vor", 1.5, 2.0), ("kal", 4.6, 4.6), ("mup", 4.6, 5.0)]
    out, _ = T.word_lines(lines, tokens, ev_of([(1.0, 2.4), (4.0, 6.2)], onsets), 10.0)
    assert flat(out) == pytest.approx(onsets, abs=0.011)
    assert out[0]["end"] == pytest.approx(2.4, abs=0.02)                              # the line ends with the voice
    assert out[1]["words"][-1]["end"] == pytest.approx(6.2, abs=0.02)


def test_no_word_of_a_line_follows_a_pause():
    lines = [["zib", "vor"], ["kal", "mup"]]
    # a weak token for the third word pulls it before the pause; the pause keeps the second line together after it
    onsets = [1.0, 1.5, 2.0, 4.0, 4.5]
    tokens = [("zib", 1.0, 1.5), ("vor", 1.5, 1.9), ("kxl", 2.0, 2.2), ("mup", 4.5, 5.0)]
    out, _ = T.word_lines(lines, tokens, ev_of([(1.0, 2.3), (4.0, 5.2)], onsets), 10.0)
    on = flat(out)
    assert on[2] >= 3.9 and on[3] == pytest.approx(4.5, abs=0.011)


def test_a_quieter_repeat_of_the_last_note_is_an_echo_not_a_word():
    n = int(6.0 / HOP)
    midi = np.full(n, 60.0)
    midi[int(4.0 / HOP):] = 64.0                                                     # the next line on another note
    ev = ev_of([(1.0, 2.0), (4.0, 5.0)], [1.0, 2.4, 4.0], length=6.0, midi=midi, quiet=[(2.4, 2.7, -14.0)])
    assert not ev.sung[int(2.5 / HOP)]                                               # 2.4-2.7: the delay
    assert ev.sung[int(4.5 / HOP)]
    out, _ = T.word_lines([["zib"], ["vor"]], [("zib", 1.0, 1.4), ("vor", 2.5, 4.6)], ev, 6.0)
    assert flat(out) == pytest.approx([1.0, 4.0], abs=0.011)


def test_too_few_sung_moments_is_an_error():
    with pytest.raises(ValueError):
        T.word_lines([["zib", "vor", "kal"]], [], ev_of([(1.0, 1.02)], []), 3.0)
