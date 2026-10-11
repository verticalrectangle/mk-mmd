import numpy as np
import pytest

from mkmmd.core import bonemap
from mkmmd.core import fingers as FG


def test_a_preset_gives_every_finger_its_three_curls():
    c = FG.curls("relaxed")
    assert list(c) == list(bonemap.FINGERS)
    assert c["index"] == c["ring"] == (14, 22, 14) and c["thumb"] == (8, 10, 8)
    assert all(v == (0, 0, 0) for v in FG.curls("flat").values())


def test_point_leaves_only_the_index_straight_and_tucks_its_thumb():
    c = FG.curls("point")
    assert c["index"] == (0.0, 0.0, 0.0) and c["middle"] == (80, 95, 65)
    assert FG.tucked("point") and FG.tucked({"thumb": "tuck"}) and not FG.tucked("fist")


PALM = np.array([0.0, 0.0, -1.0])                                # MMD rest: fingers along +x, palms down, curling to -z
MIDDLE = [[0.075, 0.0, 0.0], [0.100, 0.0, 0.0], [0.117, 0.0, 0.0], [0.131, 0.0, 0.0]]
INDEX = [[0.072, 0.020, 0.0], [0.096, 0.020, 0.0], [0.111, 0.021, 0.0], [0.124, 0.021, 0.0]]
HANDS = {                                    # (thumb, index, middle) at rest, as models hold them
    "forward": ([[0.020, 0.020, -0.005], [0.045, 0.035, -0.012], [0.062, 0.045, -0.018], [0.080, 0.055, -0.024]],
                INDEX, MIDDLE),
    "aside": ([[0.020, 0.022, -0.004], [0.036, 0.046, -0.006], [0.046, 0.063, -0.007], [0.056, 0.080, -0.008]],
              INDEX, MIDDLE),
    # close knuckles and a long thumb along the index: the least turned thumb that reaches the middle finger stands up
    # beside the index
    "close": ([[0.022, 0.017, -0.010], [0.050, 0.029, -0.022], [0.070, 0.034, -0.028], [0.092, 0.047, -0.034]],
              [[0.075, 0.015, -0.004], [0.107, 0.015, -0.005], [0.123, 0.014, -0.008], [0.139, 0.013, -0.007]],
              [[0.075, 0.000, 0.000], [0.111, 0.000, -0.002], [0.130, 0.000, -0.006], [0.148, 0.000, -0.005]]),
}


@pytest.mark.parametrize("hand", list(HANDS))
def test_a_tucked_thumb_rests_across_the_curled_middle_finger_clear_of_the_fingers(hand):
    """Whatever way a model holds its thumb at rest, the pointing hand's thumb ends folded over the curled middle
    finger: its tip resting on the middle and end segments, on the middle finger's side of the index (not standing up
    beside it), nothing of it through the middle or index finger."""
    tp, ix, md = (np.asarray(p, float) for p in HANDS[hand])
    c = FG.curls("point")
    s, th = FG.tuck_thumb(tp, ix, md, PALM, c["index"], c["middle"])
    m, x = FG.posed(md, PALM, c["middle"]), FG.posed(ix, PALM, c["index"])
    away = (tp[0] - ix[0]) * [1, 1, 0]
    t = FG.posed(tp, PALM, th, np.cross(tp[1] - tp[0], away) / np.linalg.norm(np.cross(tp[1] - tp[0], away)), s)
    length = float(np.linalg.norm(np.diff(md, axis=0), axis=1).sum())
    gap = min(FG._seg_dist(t[-1], m[1], m[2]), FG._seg_dist(t[-1], m[2], m[3]))
    rest_of = np.concatenate([t[1:-1], 0.5 * (t[:-1] + t[1:])])
    near = min(FG._seg_dist(rest_of, a, b).min() for a, b in zip(np.r_[m[:-1], x[:-1]], np.r_[m[1:], x[1:]]))
    assert abs(gap - FG.TUCK_GAP * length) <= FG.TUCK_SLACK * length
    assert near >= (FG.TUCK_CLEAR - FG.TUCK_SLACK) * length
    spacing = ix[0][1] - md[0][1]                                # the index lies at +y of the middle finger
    assert t[-1][1] <= 0.5 * (ix[0][1] + md[0][1]) + FG.TUCK_ACROSS * spacing + 1e-9
    untucked = FG.posed(tp, PALM, (25, 35, 40))                  # curled about its own axis alone: it sticks out
    assert min(FG._seg_dist(untucked[-1], m[1], m[2]), FG._seg_dist(untucked[-1], m[2], m[3])) > 2 * gap


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
