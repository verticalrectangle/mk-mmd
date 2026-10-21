"""Hit lanes (mkmmd.core.lanes): the hits of a lane picked inside a span, at or above a level, every n-th one; what a lane
that cannot be read says."""
import json

import pytest

from mkmmd.core import lanes as LN

HITS = {"kick": [{"t": 0.0, "db": 0.0}, {"t": 0.5, "db": -9.0}, {"t": 1.0, "db": -1.0}, {"t": 1.5, "db": -0.5},
                 {"t": 2.0, "db": 0.0}],
        "bare": [0.25, 0.75, 1.25]}


def test_a_lane_is_picked_by_span_level_and_every_nth_hit():
    assert LN.pick(HITS, "kick") == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert LN.pick(HITS, "kick", min_db=-2.0) == [0.0, 1.0, 1.5, 2.0]             # the quiet one is left out
    assert LN.pick(HITS, "kick", min_db=-2.0, every=2) == [0.0, 1.5]              # every other of those, the first kept
    assert LN.pick(HITS, "kick", t0=0.5, t1=2.0) == [0.5, 1.0, 1.5]               # from up to (not including) to
    assert LN.pick(HITS, "bare") == [0.25, 0.75, 1.25]
    assert LN.pick(HITS, "bare", min_db=-6.0) == []                               # a level asked of hits that have none


def test_a_lane_that_cannot_be_picked_says_what_is_wrong(tmp_path):
    with pytest.raises(LN.LaneError, match=r"no lane 'snare'.*bare, kick"):
        LN.pick(HITS, "snare")
    with pytest.raises(LN.LaneError, match="every"):
        LN.pick(HITS, "kick", every=0)
    with pytest.raises(LN.LaneError, match="hit 1"):
        LN.pick({"kick": [0.1, "loud"]}, "kick")
    with pytest.raises(LN.LaneError, match="`lane` names a lane"):
        LN.choice({"min_db": -3}, "fx 2")
    (tmp_path / "hits.json").write_text(json.dumps(HITS), encoding="utf-8")
    assert LN.load(tmp_path / "hits.json") == HITS
    with pytest.raises(LN.LaneError, match="hits file"):
        LN.load(tmp_path / "missing.json")
