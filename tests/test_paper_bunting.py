"""The paper bunting's layout (mkmmd.blender.library.props.paper_bunting_layout, no bpy): the cord hangs `sag` under the
line between its ends, no flag comes nearer the wall than CLEAR or overlaps the next, each flag's fold goes round the
cord, and the options."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import paper_bunting_layout as LAY


@pytest.mark.parametrize("slots", [{}, {"shape": "swallowtail", "mount": "tape", "seed": 4},
                                   {"shape": "round", "span": 2.4, "sag": 0.3, "seed": 9},
                                   {"ends": [[-0.8, 0.2], [0.9, -0.1]], "count": 7, "size": [0.12, 0.2]}])
def test_the_flags_hang_off_the_wall_round_the_cord_and_apart(slots):
    O = LAY.options(slots)
    P, s = LAY.cord(O)
    a, b = np.array(O.a), np.array(O.b)
    mid = P[np.argmin(np.abs(P[:, 0] - 0.5 * (a[0] + b[0])))]
    assert 0.5 * (a[1] + b[1]) - mid[2] == pytest.approx(O.sag, abs=0.004)     # the sag under the middle of the line
    F = LAY.flags(O)
    assert len(F) >= 2
    spans = []
    for f in F:
        W = f["rows"].reshape(-1, 3)
        assert W[:, 1].max() < -LAY.CLEAR                                     # never nearer the wall than CLEAR
        u = (W - f["point"]) @ f["along"]
        spans.append((f["s"] + u.min(), f["s"] + u.max()))
        ring = f["rows"][3:10]                                              # the fold over the cord, at its middle column
        d = np.linalg.norm(np.cross(ring[:, 5] - f["point"], f["along"]), axis=1)
        assert d.min() > LAY.CORD_R                                         # the paper goes round the cord, clear of it
    for (_, e0), (s1, _) in zip(spans, spans[1:]):
        assert s1 > e0                                                      # no flag overlaps the next


def test_too_many_flags_or_a_bad_option_is_refused_saying_what_to_change():
    with pytest.raises(ValueError, match="overlap"):
        LAY.flag_places(LAY.options({"count": 30}), 1.6)
    for slots, frag in (({"shape": "square"}, "shape is"), ({"ends": [[0, 0]]}, "ends is"),
                        ({"count": 5, "spacing": 0.2}, "spacing or count"), ({"colors": [1, 2]}, "colors is"),
                        ({"mount": "glue"}, "mount is")):
        with pytest.raises(ValueError, match=frag):
            LAY.options(slots)


def test_the_card_names_the_ends_and_the_low_point():
    O = LAY.options({"span": 2.0, "sag": 0.25})
    card = LAY.card("b", O)
    json.dumps(card)
    a = {u["name"]: u["point"] for u in card["use"]["anchor"]}
    assert a["end_L"][0] == pytest.approx(-1.0) and a["end_R"][0] == pytest.approx(1.0)
    low = {u["name"]: u["point"] for u in card["use"]["look"]}["middle"]
    assert low[2] == pytest.approx(-0.25, abs=0.004) and card["origin"] == "wall_center"
