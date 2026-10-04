"""Lyric type per shot, the numbers (mkmmd/core/typezones.py): which words of a line stand in which shot."""
from mkmmd.core import typezones as TZ

SHOTS = [("a", 0, 60), ("b", 60, 120), ("c", 120, 180)]


def test_a_line_that_stays_in_one_shot_is_one_block_of_all_its_words():
    blocks = TZ.plan_blocks([10, 20, 40], [100, 100, 100], SHOTS[:1])
    assert blocks == [{"shot": "a", "words": [0, 1, 2], "first": 10, "end": 60}]


def test_a_line_across_a_cut_is_a_block_per_shot_and_the_words_up_at_the_cut_are_in_both():
    lands, hides = [30, 37, 54, 66], [78] * 4
    blocks = TZ.plan_blocks(lands, hides, SHOTS)
    assert [(b["shot"], b["words"]) for b in blocks] == [("a", [0, 1, 2]), ("b", [0, 1, 2, 3])]
    assert blocks[0]["first"] == 30 and blocks[0]["end"] == 60                 # up from the first landing, held to the cut
    assert blocks[1]["first"] == 60 and blocks[1]["end"] == 78                 # the shot's first frame (they land again), to the hide


def test_a_shot_nobody_stands_in_is_left_out():
    blocks = TZ.plan_blocks([130, 140], [170, 170], SHOTS)
    assert [b["shot"] for b in blocks] == ["c"]
    assert TZ.plan_blocks([200], [220], SHOTS) == []


def test_a_word_gone_before_a_shot_starts_is_not_in_it():
    blocks = TZ.plan_blocks([10, 20], [50, 100], SHOTS)
    assert [(b["shot"], b["words"]) for b in blocks] == [("a", [0, 1]), ("b", [1])]


def test_a_word_up_for_fewer_than_min_run_frames_in_a_shot_is_left_to_the_next():
    blocks = TZ.plan_blocks([10, 59], [100, 100], SHOTS, min_run=2)
    assert [(b["shot"], b["words"]) for b in blocks] == [("a", [0]), ("b", [0, 1])]
    blocks = TZ.plan_blocks([10, 59], [100, 100], SHOTS, min_run=1)
    assert [(b["shot"], b["words"]) for b in blocks] == [("a", [0, 1]), ("b", [0, 1])]


def test_split_words_lists_the_words_no_block_holds():
    blocks = TZ.plan_blocks([10, 59, 130], [60, 60, 60], SHOTS)         # the last word lands after the line is gone
    assert TZ.split_words(blocks, 3) == [1, 2]
    assert TZ.split_words(TZ.plan_blocks([10], [100], SHOTS), 1) == []
