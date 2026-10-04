import pytest

from mkmmd.core import bonemap
from mkmmd.core import fingers as FG


def test_a_preset_gives_every_finger_its_three_curls():
    c = FG.curls("relaxed")
    assert list(c) == list(bonemap.FINGERS)
    assert c["index"] == c["ring"] == (14, 22, 14) and c["thumb"] == (8, 10, 8)
    assert all(v == (0, 0, 0) for v in FG.curls("flat").values())


def test_point_leaves_only_the_index_straight():
    c = FG.curls("point")
    assert c["index"] == (0.0, 0.0, 0.0) and c["middle"] == (80, 95, 65) and c["thumb"] == (25, 35, 40)


def test_a_table_lists_degrees_and_leaves_the_rest_straight():
    c = FG.curls({"index": [8, 10], "little": [18, 18, 4], "thumb": [0, 8, 0], "middle": 12})
    assert c["index"] == (8.0, 10.0, 0.0)                      # joints left off the end stay straight
    assert c["little"] == (18.0, 18.0, 4.0)
    assert c["thumb"] == (0.0, 8.0, 0.0)
    assert c["middle"] == (12.0, 0.0, 0.0)                     # a bare number is the first joint
    assert c["ring"] == (0.0, 0.0, 0.0)                        # a finger left out is straight


@pytest.mark.parametrize("spec, fragment", [
    ("loose", "preset 'loose'"),
    ({"pinky": [1, 2, 3]}, "unknown"),
    ({"index": [1, 2, 3, 4]}, "up to three"),
    ({"index": ["a"]}, "up to three"),
    (7, "preset name or a table"),
])
def test_bad_specs_name_what_is_wrong(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        FG.curls(spec)
