"""The taped papers' layout (mkmmd.blender.library.props.taped_papers_layout, no bpy): every sheet and strip in front of
the wall, every strip in front of whatever it covers (no z-fighting), the default layout's gap, an instant photo's picture
where its gloss window is, and the pieces' checks."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import taped_papers_layout as LAY

KINDS = [{"kind": "poster", "design": d, "at": [0.4 * i, 0.0], "rot": 5.0 * i} for i, d in enumerate(LAY.DESIGNS)] + [
    {"kind": "polaroid", "photo": "moon", "at": [0.0, 0.5]}, {"kind": "moon", "at": [0.25, 0.45], "tape": 2},
    {"kind": "star", "at": [0.55, 0.45]}, {"kind": "card", "at": [0.8, 0.45], "tape": 2, "corner": "TR"},
    {"kind": "card", "shape": "heart", "at": [1.0, 0.45], "rot": -30}]


@pytest.mark.parametrize("specs", [None, KINDS])
def test_the_papers_and_tape_lie_in_front_of_the_wall_and_of_what_they_cover(specs):
    pieces, seed, _ = LAY.options({} if specs is None else {"pieces": specs})
    for p in pieces:
        m = LAY.piece_mesh(p)
        assert m.V[:, 1].max() < -0.0004 and not m.is_closed()
    tape = LAY.tapes_mesh(pieces, seed)
    ground = LAY.ground_fn(pieces)
    assert len(tape.V) and np.all(tape.V[:, 1] < ground(tape.V[:, 0], tape.V[:, 2]) - 0.0002)


def test_the_default_layout_leaves_a_gap_for_a_face():
    pieces, _, _ = LAY.options({})
    (x, z), r = LAY.gap(pieces)
    assert r > 0.15 and abs(x) < 0.1 and abs(z) < 0.1
    card = LAY.card("papers", pieces)
    json.dumps(card)
    gap = {u["name"]: u for u in card["use"]["look"]}["gap"]
    assert gap["point"] == [pytest.approx(x, abs=1e-5), 0.0, pytest.approx(z, abs=1e-5)]


def test_an_instant_photo_shows_its_picture_where_its_gloss_window_is():
    p = LAY.piece(0, {"kind": "polaroid", "photo": "sun", "color": "pine", "ink": "gold", "at": [0, 0]}, {})
    img = LAY.pictures([p], {}, LAY.ROLES, 1)[0]
    h, w, _ = img.shape
    u0, v0, u1, v1 = LAY.photo_window(p)
    frame = img[int(h * 0.97), w // 2]                                       # the chin under the picture
    corner = img[int(h * (1 - v1)) + 3, int(w * u0) + 3]                     # just inside the picture's top left
    outside = img[int(h * (1 - v1)) - 3, int(w * u0) - 3]
    assert np.allclose(frame, outside, atol=0.02) and not np.allclose(corner, frame, atol=0.1)


@pytest.mark.parametrize("spec, frag", [({"kind": "sticker", "at": [0, 0]}, "kind is"), ({"kind": "moon"}, "at is"),
                                        ({"kind": "poster", "at": [0, 0], "design": "car"}, "design is"),
                                        ({"kind": "poster", "at": [0, 0], "size": 0.3}, "size is"),
                                        ({"kind": "star", "at": [0, 0], "tape": 3}, "tape is"),
                                        ({"kind": "card", "at": [0, 0], "corner": "MID"}, "corner is")])
def test_a_bad_piece_is_refused_saying_what_to_change(spec, frag):
    with pytest.raises(ValueError, match=frag):
        LAY.options({"pieces": [spec]})
