"""The idol fan penlight's layout (mkmmd.blender.library.props.penlight_layout, no bpy): closed outward parts in the card's
frame, a hold on the handle a fist closes round (the button above it, the tube out of its top), the colour index rounded
and wrapped round the list, the options and the card. The Blender half (the tube's material picking the colour from the
root's `color`) is tests/test_show_props_build.py."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import penlight_layout as LAY
from mkmmd.blender.library.props import showkit as SK


def test_every_part_is_closed_and_faces_out():
    for m in (LAY.handle_mesh(), LAY.button_mesh(), LAY.collar_mesh(), LAY.tube_mesh(), LAY.ring_mesh()):
        assert m.is_closed() and m.volume() > 0
    body = LAY.meshes()["body"]
    used = set(np.concatenate([body.Qm, body.Tm]).tolist())
    assert used == {LAY.MATS.index(r) for r in ("handle", "grip", "button", "collar", "ring")}


def test_the_hold_lies_on_the_handle_with_the_button_above_it_and_the_tube_beyond():
    card = LAY.card("pl")
    hold = card["use"]["grip"][0]
    assert hold["name"] == "hold" and hold["type"] == "hold" and hold["axis"] == [0.0, 0.0, 1.0]
    z0, z1 = hold["center"][2] - hold["length"] / 2, hold["center"][2] + hold["length"] / 2
    h = LAY.handle_mesh()
    span = (h.V[:, 2] > z0) & (h.V[:, 2] < z1)
    r = np.hypot(h.V[span, 0], h.V[span, 1])
    assert r.max() == pytest.approx(hold["radius"], abs=1e-6) and r.min() > hold["radius"] - 0.002   # the handle's girth
    assert 0.0 < z0 and z1 < LAY.HANDLE_TOP                              # on the handle, clear of its ends
    b = LAY.button_mesh()
    assert b.V[:, 2].min() > z1 - 0.001 and b.V[:, 2].max() < LAY.COLLAR[0]      # above the fist, under the collar
    assert b.V[:, 1].max() < 0.0 and b.V[:, 1].min() < -LAY.HANDLE_R - 0.001     # on the front, standing out of it
    t = LAY.tube_mesh()
    assert t.V[:, 2].max() == pytest.approx(LAY.LENGTH) and t.V[:, 2].min() > LAY.HANDLE_TOP
    assert card["use"]["anchor"][0] == {"name": "tip", "point": [0.0, 0.0, LAY.LENGTH], "dir": [0.0, 0.0, 1.0]}


def test_the_colour_index_rounds_and_wraps_round_the_list():
    assert [LAY.pick(c, 7) for c in (0, 0.49, 0.5, 1.2, 6, 6.6, 7, 13, -1, -0.6)] == [0, 0, 1, 1, 6, 0, 0, 6, 6, 6]


def test_the_options_take_any_colours_and_starting_values():
    colors, P, strength = LAY.options({"colors": ["love", "#ff8800", "iris:text:0.3"], "color": 2, "glow": 0.5})
    assert colors == ["love", "#ff8800", "iris:text:0.3"] and P == {"color": 2.0, "glow": 0.5}
    assert strength == LAY.STRENGTH
    assert LAY.options({})[0] == LAY.COLORS and LAY.options({"colors": "foam"})[0] == ["foam"]
    rgb = [SK.rgb(c, {"love": "#eb6f92"}) for c in colors]
    assert rgb[0] == pytest.approx(SK.rgb("#eb6f92", {}))


@pytest.mark.parametrize("slots, frag", [
    ({"colors": []}, "colors is a list of 1 .. 16"),
    ({"colors": ["love", "neon"]}, r"colors\[1\]"),
    ({"colors": 3}, "colors is a list"),
    ({"glow": -1}, "glow is a number 0 .. 10"),
    ({"color": "pink"}, "color is a number"),
    ({"strength": -2}, "strength is a number"),
])
def test_a_bad_penlight_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        LAY.options(slots)


def test_the_card_is_plain_json():
    colors, P, _ = LAY.options({"colors": ["love", "foam"], "color": 1})
    card = LAY.card("pl", {"handle": "base"}, colors, P)
    json.dumps(card)
    assert card["origin"] == "tail" and card["front"] == "-Y" and card["params"] == {"color": 1.0, "glow": 1.0}
    assert card["colors"] == ["love", "foam"] and card["objects"] == ["pl_body", "pl_tube"]
    assert {u["name"] for u in card["use"]["look"]} == {"tube", "grip"}
