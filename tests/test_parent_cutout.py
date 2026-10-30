"""The picture-book parent cut-outs' layout (mkmmd.blender.library.props.parent_cutout_layout and _parts, no bpy): closed
outward plates, outlines that never cross themselves, the white border round the paper, the card on its foot line, the
father's hand on its brad with the card's tip at the index fingertip, the drop shadows, the wag maths and the driver's
expression, the options and the card. The Blender half (the wag drivers) is tests/test_parent_cutout_drivers.py."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_guitar_geo as GEO
from mkmmd.blender.library.props import parent_cutout_layout as LAY
from mkmmd.blender.library.props import parent_cutout_parts as PARTS
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

DAWN = PAL.PALETTES["rose-pine-dawn"]


@pytest.fixture(scope="module", params=LAY.NAMES)
def built(request):
    return LAY.build(request.param)


def _closed(m):
    return m.is_closed() and m.volume() > 0


def _details(figure):
    """Every detail's scissor-cut ring, as `cut` makes it (card frame)."""
    shift = np.array([0.0, LAY.build(figure).lift])
    return [(key, k, LAY.detail_ring(figure, key, k, d, shift)[0])
            for key, piece in PARTS.FIGURES[figure].pieces.items() for k, d in enumerate(piece.details)]


def test_every_plate_is_closed_and_faces_outward(built):
    assert _closed(built.body) and _closed(built.brads)                      # the cards, papers, details; the brads
    if built.hand is not None:
        assert _closed(built.hand)
    for sh, sheet in built.sheets + ([built.hand_sheet] if built.hand_sheet else []):
        n, _ = sheet.face_vectors()
        assert (n[:, 1] < 0).all()                                            # the shadow sheets face the viewer (-Y)


def test_no_outline_crosses_itself_and_the_white_border_runs_round_the_paper(built):
    for key, c in built.cuts.items():
        assert LAY.crossings(c.paper) == [] and LAY.crossings(c.rim) == [], key
        assert S.polygon_area(c.paper) > 0 and S.polygon_area(c.rim) > 0
        inside = GEO.inside(c.paper, c.rim)
        gap = GEO.distance(c.paper, c.rim)
        assert inside.all() and gap.min() > 0.6 * LAY.RIM, key              # a border all round, never thinner
    for key, k, ring in _details(built.figure):
        assert LAY.crossings(ring) == [], (key, k)


def test_scissor_cuts_are_straight_snips_close_to_the_drawn_line():
    base = LAY.outline(PARTS.FIGURES["father"].pieces["body"].points, [0])
    cut = LAY.scissor(base, 7, LAY.STEP)
    assert GEO.distance(cut, base).max() < 0.006                             # within a few mm of the drawing
    e = np.roll(cut, -1, 0) - cut
    turn = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(np.arctan2(e[:, 1], e[:, 0]))))))
    assert (turn < 1.5).mean() > 0.6 and (turn > 2.0).sum() > 40             # mostly straight, turning at the joints
    assert not np.allclose(LAY.scissor(base, 8, LAY.STEP)[:50], cut[:50])   # each piece its own cut


def test_the_card_stands_on_its_foot_line_and_is_as_tall_as_a_giant_parent(built):
    V = built.body.V
    assert V[:, 2].min() == pytest.approx(0.0, abs=2e-3)                     # the border's foot on the floor
    assert 3.0 <= built.height <= 3.4 and V[:, 2].max() == pytest.approx(built.height, abs=3e-3)
    assert abs(V[:, 0].max() + V[:, 0].min()) < 0.25                         # about the centre line
    feet = V[V[:, 2] < 0.01]
    assert feet[:, 0].min() < -0.2 and feet[:, 0].max() > 0.2                # it stands on both feet
    assert V[:, 1].min() > -0.05 and V[:, 1].max() < 0.04                    # a card, not a slab
    size = np.array(built.size)
    assert size[2] == pytest.approx(built.height, abs=3e-3) and 1.2 < size[0] < 1.6


def test_proportions_follow_an_adult_seven_and_a_half_heads_tall():
    for figure, crown, chin in (("father", 3.21, 2.765), ("mother", 2.98, 2.58)):
        heads = crown / (crown - chin)
        assert 6.8 <= heads <= 7.8, figure
    L = LAY.build("father")
    head = np.array(L.head)
    rim = L.cuts["body"].rim
    assert GEO.inside(head[[0, 2]][None], rim)[0] and 2.8 < head[2] < 3.2   # the look point in the face


def test_the_hand_turns_on_its_brad_at_the_elbow_and_the_tip_is_the_index_fingertip():
    L = LAY.build("father")
    pivot, tip = np.array(L.pivot), np.array(L.tip)
    lift = np.array([0.0, L.lift])
    assert pivot[[0, 2]] == pytest.approx(np.array(PARTS.ELBOW) + lift, abs=1e-9)
    paper = L.cuts["hand"].paper                                             # in the brad's frame
    disc = paper[paper[:, 1] < -0.04]                                        # the round end under the brad
    r = np.hypot(disc[:, 0], disc[:, 1])
    assert r.max() < 0.095 and r.min() > 0.08                               # centred on the pivot
    brad = L.brads.V
    head = brad[(np.hypot(brad[:, 0] - pivot[0], brad[:, 2] - pivot[2]) < LAY.BRAD[0] + 1e-6) & (brad[:, 1] < pivot[1])]
    assert len(head) and head[:, [0, 2]].mean(0) == pytest.approx(pivot[[0, 2]], abs=1e-6)   # the brad's dome on it
    assert head[:, 1].min() == pytest.approx(pivot[1] - LAY.BRAD[1], abs=1e-6)
    top = paper[np.argmax(paper[:, 1])] + pivot[[0, 2]]
    assert tip[[0, 2]] == pytest.approx(top, abs=1e-9) and tip[1] == pivot[1]   # the highest point of the hand
    drawn = np.array(PARTS.FATHER_TIP) + lift
    assert np.hypot(*(tip[[0, 2]] - drawn)) < 0.006                          # the index fingertip as drawn
    assert 0.75 < tip[2] - pivot[2] < 1.0                                    # the finger raised above the elbow
    hand = L.cuts["hand"].paper
    fist = hand[(hand[:, 1] > 0.55) & (hand[:, 1] < 0.72)]                   # across the fist, under the finger
    assert tip[0] - pivot[0] < -0.04 and tip[0] - pivot[0] < fist[:, 0].min() + 0.05   # the index on the thumb side,
    assert fist[:, 0].max() > 0.09                                           # toward his head; the curl on the far side


def test_the_hand_lies_in_front_of_the_body_at_every_turn():
    L = LAY.build("father")
    hand = L.hand.V + np.array(L.pivot)
    assert hand[:, 1].max() < L.body.V[:, 1].min()                           # it can turn round its brad freely
    assert L.hand_at[1] < L.body.V[:, 1].min() and L.hand_at[1] > hand[:, 1].max()   # its shadow between them


def test_the_wag_turns_counter_clockwise_seen_from_the_front_about_the_axis():
    L = LAY.build("father")
    p, t0 = np.array(L.pivot), np.array(L.tip)
    for deg in (-30.0, -20.0, 0.0, 20.0, 45.0):
        t = LAY.tip_at("father", deg)
        assert np.linalg.norm(t - p) == pytest.approx(np.linalg.norm(t0 - p))   # in the card's plane, about the brad
        assert t[1] == pytest.approx(p[1])
    assert LAY.tip_at("father", 20.0)[0] < t0[0] < LAY.tip_at("father", -20.0)[0]   # + leans toward the head (-x)
    R = LAY.turn(30.0)
    assert R @ np.array([0.0, -1.0, 0.0]) == pytest.approx([0.0, -1.0, 0.0])  # the axis toward the viewer
    rot_y = np.array([[math.cos(-math.radians(30)), 0, math.sin(-math.radians(30))], [0, 1, 0],
                      [-math.sin(-math.radians(30)), 0, math.cos(-math.radians(30))]])
    assert R == pytest.approx(rot_y)                                          # = Blender rotation_euler[1] -30 degrees


def test_the_driver_expression_is_the_wag_maths_and_a_simple_expression():
    assert len(LAY.EXPR) <= 255
    names = set(LAY.EXPR.replace("(", " ").replace(")", " ").replace("*", " ").replace("/", " ").replace("+", " ")
                .replace("-", " ").split())
    assert {n for n in names if n[0].isalpha()} <= {"wag", "wagging", "wag_hz", "frame", "f0", "fps", "sin"}
    assert "**" not in LAY.EXPR and "%" not in LAY.EXPR
    for wag, wagging, hz, frame in ((-20.0, 0.0, 2.5, 31), (12.0, 18.0, 2.5, 47), (0.0, 18.0, 1.7, 100)):
        env = {"wag": wag, "wagging": wagging, "wag_hz": hz, "frame": frame, "f0": 31.0, "fps": 30.0, "sin": math.sin}
        t = (frame - 31.0) / 30.0
        assert eval(LAY.EXPR, {"__builtins__": {}}, env) == pytest.approx(-math.radians(LAY.angle(t, wag, wagging, hz)),
                                                                         abs=1e-7)
    assert LAY.angle(0.1, 5.0, 10.0, 2.5) == pytest.approx(15.0)              # a quarter of a tick-tock: + full swing


def test_the_shadows_are_the_borders_offset_down_and_to_the_side(built):
    sh, sheet = built.sheets[0]
    a = sh.alpha
    assert a.max() == pytest.approx(1.0, abs=0.02) and a[0].max() < 0.01 and a[:, 0].max() < 0.01   # soft, padded
    h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = sh.x0 + (xs + 0.5) * LAY.SHADOW_PX
    cz = sh.z1 - (ys + 0.5) * LAY.SHADOW_PX                                  # row 0 is the top
    centre = np.array([(a * cx).sum(), (a * cz).sum()]) / a.sum()
    rim = built.cuts["body"].rim
    cov = LAY.raster(rim, sh.x0, sh.z0, LAY.SHADOW_PX, w, h)
    own = np.array([(cov * cx).sum(), (cov * cz).sum()]) / cov.sum()
    assert centre - own == pytest.approx(LAY.SHADOW, abs=0.004)
    assert sheet.V[:, 1].min() > built.body.V[:, 1].max() - 0.025            # behind the card (on its back plane)


def test_options_and_colour_roles_are_checked():
    assert LAY.options({}) == ("father", {"wag": 0.0, "wagging": 0.0, "wag_hz": 2.5})
    assert LAY.options({"figure": "mother", "wag": 15, "wag_hz": 3})[0] == "mother"
    for bad in ({"figure": "uncle"}, {"wag": "left"}, {"wag_hz": 0.0}, {"wagging": -5.0}, {"wag": True}):
        with pytest.raises(ValueError, match="parent_cutout"):
            LAY.options(bad)
    for figure in LAY.NAMES:
        pal = LAY.Pal(DAWN, figure)
        pal.check()
        for role in LAY.ROLES:
            assert all(0.0 <= c <= 1.0 for c in pal.lin(role))
    dawn = LAY.Pal(DAWN, "father")
    assert dawn.lin("paper") == pytest.approx(PAL.linear(DAWN["pine"]))
    assert dawn.lin("shade") == pytest.approx([0.75 * c for c in PAL.linear(DAWN["pine"])])   # a darker paper
    assert dawn.lin("hair") == pytest.approx(PAL.linear(DAWN["text"]))                       # ink in Dawn
    assert LAY.Pal(DAWN, "mother").lin("paper") == pytest.approx(PAL.linear(DAWN["rose"]))
    moon = LAY.Pal(PAL.PALETTES["rose-pine-moon"], "father")
    assert sum(moon.lin("rim")) > sum(moon.lin("shadow"))                     # white card, dark shadow in a dark palette
    with pytest.raises(ValueError, match="colour role paper"):
        LAY.Pal(dict(DAWN, paper="nope"), "father").check()


def test_the_card_has_the_contracted_use_points():
    c = LAY.card("dad", "father", {"paper": "pine"}, {"wag": 5.0, "wagging": 0.0, "wag_hz": 2.5})
    L = LAY.build("father")
    assert c["origin"] == "floor_center" and c["front"] == "-Y" and c["figure"] == "father"
    (hand,) = c["use"]["hand"]
    assert hand["name"] == "wag" and hand["object"] == "dad_hand" and hand["prop"] == "wag"
    assert hand["axis"] == [0.0, -1.0, 0.0]
    assert hand["pivot"] == pytest.approx(L.pivot, abs=1e-4) and hand["tip"] == pytest.approx(L.tip, abs=1e-4)
    (look,) = c["use"]["look"]
    assert look["name"] == "head" and look["point"] == pytest.approx(L.head, abs=1e-4)
    assert c["params"] == {"wag": 5.0, "wagging": 0.0, "wag_hz": 2.5}
    assert set(c["objects"]) == {"dad_figure", "dad_shadow", "dad_brad", "dad_hand", "dad_hand_shadow"}
    assert c["form_exempt"] == ["dad_shadow", "dad_hand_shadow"]
    m = LAY.card("mom", "mother", {}, {})
    assert "hand" not in m["use"] and m["params"] == {} and [u["name"] for u in m["use"]["look"]] == ["head"]
    assert set(m["objects"]) == {"mom_figure", "mom_shadow", "mom_brad"}
