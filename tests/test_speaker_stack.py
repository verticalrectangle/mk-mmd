"""The festival speaker stack's layout (mkmmd.blender.library.props.speaker_stack_layout, no bpy): closed outward parts in
each cabinet's frame, a shell whose inset rings never turn inside out, the cones' pump (the cone and cap move, the frame
stays, the surround stretches between) clear of the grille, the cutters under the parts that hide their edges, the stack
on its feet, the options and the card. The Blender half (the pump's shape key driven by the root, the Boolean cuts) is
tests/test_show_props_build.py."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import speaker_stack_layout as LAY
from mkmmd.core import shell as S


@pytest.mark.parametrize("kind", sorted(LAY.CABINETS))
def test_every_part_is_closed_and_faces_out(kind):
    for part in LAY.parts(kind):
        m = LAY.part_mesh(kind, part)
        assert m.is_closed() and m.volume() > 0, (kind, part)
    for cut in LAY.cutters(kind):
        assert cut.is_closed() and cut.volume() > 0


@pytest.mark.parametrize("kind", sorted(LAY.CABINETS))
def test_the_shell_is_the_cabinet_s_size_its_rings_never_cross_and_the_baffle_is_recessed(kind):
    c = LAY.CABINETS[kind]
    m = LAY.shell_mesh(kind)
    lo, hi = m.bbox()
    assert lo == pytest.approx([-c["W"] / 2, -c["D"] / 2, 0.0], abs=1e-9)
    assert hi == pytest.approx([c["W"] / 2, c["D"] / 2, c["H"]], abs=1e-9)
    for d, _ in LAY.shell_profile(c["D"]):                              # every inset ring a simple loop
        ring = S.rrect(c["W"] - 2 * d, c["H"] - 2 * d, max(LAY.R_CORNER - d, LAY.R_MIN), 12)
        assert not S.section_crosses_itself(ring) and S.polygon_area(ring) > 0
    baffle = np.concatenate([m.Qm, m.Tm]) == LAY.MATS.index("baffle")
    cq, ct = m.face_centres()
    y = np.concatenate([cq, ct])[baffle, 1]
    assert baffle.sum() > 0 and y == pytest.approx(np.full(len(y), -c["D"] / 2 + LAY.RECESS), abs=1e-9)


def test_a_pump_moves_the_cone_and_cap_out_keeps_the_frame_and_stretches_the_surround():
    P, w = LAY.driver_profile(0.229)                       # the back's apex, the frame, the surround, the cone, the cap
    r_so, r_si, r_dc = LAY.driver_zones(0.229)
    assert w[0] == 0.0 and P[0, 1] < 0.0                                # the back (hidden in the cabinet) stays
    assert w[P[:, 0] > r_so + 1e-6].max() == 0.0                        # so does the frame
    roll = np.nonzero((P[:, 0] < r_so) & (P[:, 0] > r_si) & (P[:, 1] > 0.0))[0]
    assert np.all((w[roll] > 0.0) & (w[roll] < 1.0))
    assert np.all(np.diff(w[roll]) > 0)                                 # more the nearer the cone
    assert np.all(w[roll[-1] + 1:] == 1.0)                              # the cone and the dust cap move as one
    assert P[-1, 0] == 0.0 and P[-1, 1] > P[roll[-1] + 1:, 1].min()     # ... ending on the cap's dome
    for kind in LAY.CABINETS:
        m, d = LAY.drivers(kind)
        assert d.shape == m.V.shape and np.all(d[:, [0, 2]] == 0.0)
        assert d[:, 1].min() == pytest.approx(-LAY.PUSH[kind]) and d[:, 1].max() == 0.0


@pytest.mark.parametrize("kind", sorted(LAY.CABINETS))
def test_at_full_pump_every_cone_stays_behind_the_grille(kind):
    m, d = LAY.drivers(kind)
    g = LAY.grille_mesh(kind)
    assert (m.V + d)[:, 1].min() > g.V[:, 1].max() + 0.004                 # 4 mm of air at least


@pytest.mark.parametrize("kind", sorted(LAY.CABINETS))
def test_the_holes_hide_under_the_driver_frames_and_the_parts_sit_on_the_baffle(kind):
    c = LAY.CABINETS[kind]
    y_baffle = -c["D"] / 2 + LAY.RECESS
    for x, z, Rf in c["drivers"]:
        r_so, _, _ = LAY.driver_zones(Rf)
        assert r_so < LAY.HOLE * Rf < Rf                                   # the frame ring covers the hole's edge
    m, _ = LAY.drivers(kind)
    front = m.V[:, 1].min()
    assert y_baffle - 0.03 < front < y_baffle                             # the frame and surround stand on the baffle
    hw, z0, z1 = LAY.opening(kind)
    lo, hi = m.bbox()
    assert -hw < lo[0] and hi[0] < hw and z0 < lo[2] and hi[2] < z1     # inside the opening
    for part in ("ports", "horn"):
        if part in LAY.parts(kind):
            pm = LAY.part_mesh(kind, part)
            assert pm.V[:, 1].min() > -c["D"] / 2 + LAY.GRILLE["gap"]    # behind the grille


def test_the_cabinets_stand_on_their_feet_on_one_another_with_flush_fronts():
    P = LAY.placed(["sub", "sub", "top"])
    assert [c["kind"] for c in P] == ["sub", "sub", "top"]
    assert P[0]["z0"] == pytest.approx(LAY.FOOT[1])
    for a, b in zip(P, P[1:]):
        assert b["z0"] == pytest.approx(a["z0"] + a["H"] + LAY.FOOT[1])
    fronts = [c["y0"] - c["D"] / 2 for c in P]
    assert fronts == pytest.approx([fronts[0]] * 3)
    feet = LAY.hardware_mesh("sub")
    assert feet.V[:, 2].min() == pytest.approx(-LAY.FOOT[1])           # the feet reach down to the cabinet below


@pytest.mark.parametrize("slots, frag", [
    ({"stack": ["sub", "woofer"]}, "stack is a list"),
    ({"stack": []}, "stack is a list"),
    ({"stack": ["top", "sub"]}, "subs stand under the tops"),
    ({"grille": "cloth"}, "grille is"),
    ({"pump": 2.0}, "pump is a number 0 .. 1"),
    ({"pump": True}, "pump is a number"),
])
def test_a_bad_stack_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        LAY.options(slots)


def test_the_card_is_plain_json_with_its_cones_rest_colliders_and_form_limit():
    stack, grille, P = LAY.options({"stack": ["sub", "top"]})
    card = LAY.card("st", stack, grille, {"cap": "love"}, P)
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y" and card["params"] == {"pump": 0.0}
    assert card["form_max"] == LAY.FORM_MAX and card["stack"] == ["sub", "top"]
    cones = [a for a in card["use"]["anchor"] if a["name"].startswith("cone")]
    assert [a["name"] for a in cones] == ["cone0", "cone1", "cone2", "cone3"]
    assert all(a["dir"] == [0.0, -1.0, 0.0] for a in cones)
    xs = [a["point"][0] for a in cones]
    assert xs[0] < xs[1] and xs[2] < xs[3] and cones[0]["point"][2] < cones[2]["point"][2]   # bottom up, left to right
    top = card["use"]["rest"][0]
    assert top["name"] == "top" and top["center"][2] == pytest.approx(card["bounds"]["max"][2])
    assert [c["object"] for c in card["colliders"]] == ["st_cab0_col", "st_cab1_col"]
    assert "st_cab0_ports" in card["objects"] and "st_cab1_horn" in card["objects"] and "st_cab1_ports" not in card["objects"]
    assert set(LAY.ROLES) == set(LAY.MATS)
