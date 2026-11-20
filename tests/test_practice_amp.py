"""The practice amp's layout (mkmmd.blender.library.props.practice_amp_layout, no bpy): closed outward parts, the card's
jack at the jack's mouth, the controls on the panel clear of one another for every knob count, the welt in the gap
round the cloth, and the options."""
import json

import pytest

from mkmmd.blender.library.props import practice_amp_layout as LAY


@pytest.mark.parametrize("knobs", [3, 4, 5])
def test_every_part_is_closed_and_faces_out_but_the_print(knobs):
    for part, m in LAY.part_meshes(LAY.options({"knob_count": knobs})).items():
        if part not in LAY.GRAPHIC:
            assert m.is_closed() and m.volume() > 0, part


def test_the_jack_anchor_is_the_mouth_of_the_jack_looking_out_of_the_front():
    card = LAY.card("amp")
    json.dumps(card)
    jack = card["use"]["anchor"][0]
    assert jack["name"] == "jack" and jack["dir"] == [0.0, -1.0, 0.0]
    m = LAY.on_front(LAY.jack_mesh(), LAY.JACK_X, LAY.row_z(), LAY.Y_BAFFLE - LAY.PANEL_T)
    lo, hi = m.bbox()
    assert jack["point"][1] == pytest.approx(lo[1], abs=1e-4)               # its front-most rim
    assert jack["point"][0] == pytest.approx(0.5 * (lo[0] + hi[0]), abs=1e-6)
    assert jack["point"][2] == pytest.approx(0.5 * (lo[2] + hi[2]), abs=1e-6)
    assert lo[1] > LAY.Y_FRONT - 0.003                                       # not proud of the tolex frame


@pytest.mark.parametrize("knobs", [3, 4, 5])
def test_the_controls_sit_on_the_panel_clear_of_one_another(knobs):
    hw, z0, z1 = LAY.panel_box()
    reach = [(LAY.JACK_X, 0.0080), (LAY.SWITCH_X, 0.0105), (LAY.LAMP_X, 0.0086)]
    reach += [(x, LAY.DOTS[0] + 1.35 * LAY.DOTS[1]) for x in LAY.knob_xs(knobs)]       # a knob with its scale
    reach.sort()
    for x, r in reach:
        assert -hw + 0.003 < x - r and x + r < hw - 0.003
        assert LAY.row_z() - r > z0 + 0.003 and LAY.row_z() + r < z1 - 0.003
    for (xa, ra), (xb, rb) in zip(reach, reach[1:]):
        assert xb - xa > ra + rb + 0.001, (xa, xb)


def test_the_welt_lies_in_the_gap_between_the_cloth_and_the_field():
    hw_f, z0_f, _ = LAY.field()
    hw_g, z0_g, z1_g = LAY.grille_box()
    _, z0_p, _ = LAY.panel_box()
    lo, hi = LAY.piping_mesh().bbox()
    assert hw_g < hi[0] < hw_f and -hw_f < lo[0] < -hw_g
    assert z0_f < lo[2] < z0_g and z1_g < hi[2] < z0_p
    g_lo, _ = LAY.grille_mesh().bbox()
    assert lo[1] < g_lo[1] and hi[1] <= LAY.Y_BAFFLE + 1e-9                  # proud of the cloth, not behind the baffle


@pytest.mark.parametrize("slots, frag", [({"knob_count": 6}, "knob_count is a whole number 3 .. 5"),
                                         ({"knob_count": 3.5}, "knob_count"),
                                         ({"corners": 1}, "corners is the protectors' colour, or false"),
                                         ({"glow": -1}, "glow is a number")])
def test_bad_options_are_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        LAY.options(slots)


def test_corners_is_the_protectors_colour_or_false_for_none():
    assert not LAY.options({"corners": False})["corners"] and LAY.options({"corners": "gold"})["corners"]
    assert LAY.roles({"corners": False})["corners"] == LAY.ROLES["corners"]
    assert LAY.roles({"corners": "gold"})["corners"] == "gold"
    with pytest.raises(ValueError, match="corners"):
        LAY.roles({"corners": "not-a-colour"})
