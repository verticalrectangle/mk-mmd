"""The golden spike's layout (mkmmd.blender.library.props.golden_spike_layout, no bpy): a closed spike standing on its tip
at the origin, the real last spike's size, the chisel point, the head, the beads round the engraved band, a square shank
that is no box, the card against the meshes (the hold on the shank with the shank's radius, the tip, the look points)
and the `length` option."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import golden_spike_layout as LAY
from mkmmd.blender.library.props.cafe_colors import Colors
from mkmmd.core import form as F
from mkmmd.core import palette as PAL


def _ring_at(m, z, tol=1e-7):
    """The vertices of the spike's ring at height z, ordered round the axis."""
    P = m.V[np.abs(m.V[:, 2] - z) < tol]
    return P[np.argsort(np.arctan2(P[:, 1], P[:, 0]))]


def test_the_spike_is_one_closed_form_standing_on_its_tip_at_the_origin():
    m = LAY.spike_mesh()
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-12) and hi[2] == pytest.approx(LAY.LENGTH, abs=1e-9)
    tip = m.V[np.argmin(m.V[:, 2])]
    assert tip == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)                        # the tip's middle is the origin
    assert LAY.LENGTH == pytest.approx(5.625 * 0.0254, abs=5e-4)                  # 5 5/8 in: the 1869 last spike
    assert hi[:2] - lo[:2] == pytest.approx(LAY.HEAD[:2], abs=1e-6)                # the head is the widest
    card = LAY.card("s", LAY.ROLES)
    assert card["size"] == pytest.approx([LAY.HEAD[0], LAY.HEAD[1], LAY.LENGTH], abs=1e-5) and card["origin"] == "tip"
    assert card["objects"] == ["s_spike", "s_beads"]


def test_the_square_shank_and_the_head_are_forms_not_boxes():
    parts = LAY.meshes()
    V, T, owner, off = [], [], [], 0
    for i, m in enumerate(parts.values()):
        V.append(m.V)
        T.append(m.triangles() + off)
        owner.append(np.full(len(m.triangles()), i))
        off += len(m.V)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), list(parts))
    assert r["score"] < 0.05, (r["score"], r["components"][:4])


def test_a_square_shank_tapers_a_little_to_a_chisel_point_whose_edge_runs_across_x():
    m = LAY.spike_mesh()
    z = LAY.stations()
    low, high = z[(z > LAY.POINT_H + 0.005)][0], z[z < LAY.shank_top() - 0.002][-1]
    for zz, w in ((low, LAY.SHANK[0]), (high, LAY.SHANK[1])):
        r = _ring_at(m, zz)
        assert np.ptp(r[:, 0]) == pytest.approx(np.ptp(r[:, 1]), rel=0.02)        # square
        assert np.ptp(r[:, 0]) == pytest.approx(LAY.shank_w(zz), abs=1e-4)
    assert LAY.shank_w(low) < LAY.shank_w(high)                                  # tapering toward the point
    edge = _ring_at(m, z[0])
    assert np.ptp(edge[:, 1]) < 0.003 and np.ptp(edge[:, 0]) > 0.012             # thin along Y, the shank's width in X
    assert np.ptp(_ring_at(m, z[z < LAY.POINT_H / 2][-1])[:, 1]) < 0.6 * LAY.SHANK[0]


def test_the_head_overhangs_the_shank_with_its_top_crowned():
    m = LAY.spike_mesh()
    top = m.V[m.V[:, 2] > LAY.LENGTH - LAY.HEAD[2] + 1e-6]
    assert np.ptp(top[:, 0]) == pytest.approx(LAY.HEAD[0], abs=1e-6)
    assert LAY.HEAD[0] > 1.7 * LAY.SHANK[1] and LAY.HEAD[1] > 1.3 * LAY.SHANK[1]
    crown = m.V[m.V[:, 2] > LAY.LENGTH - LAY.CROWN + 1e-9]
    middle = np.hypot(crown[:, 0], crown[:, 1])
    assert crown[np.argmin(middle), 2] == pytest.approx(LAY.LENGTH)               # highest in the middle
    assert crown[:, 2].min() > LAY.LENGTH - LAY.CROWN


def test_two_closed_beads_bound_the_engraved_band_on_the_shank():
    beads = LAY.bead_mesh()
    assert beads.is_closed() and beads.volume() > 0
    b0, b1 = LAY.band()
    assert b1 - b0 == pytest.approx(LAY.BAND_H)
    assert LAY.POINT_H < b0 and b1 < LAY.shank_top()                              # on the shank
    zs = np.round(beads.V[:, 2], 4)
    assert beads.V[:, 2].min() == pytest.approx(b0 - LAY.BEAD_R, abs=1e-4)
    assert beads.V[:, 2].max() == pytest.approx(b1 + LAY.BEAD_R, abs=1e-4)
    assert len(set(zs)) > 4
    w = LAY.shank_w(b1)
    assert np.abs(beads.V[:, :2]).max() == pytest.approx(0.5 * w + LAY.BEAD_R, abs=3e-4)   # half sunk in the shank


def test_the_card_holds_the_shank_and_points_at_the_tip_and_the_head():
    card = LAY.card("s", LAY.ROLES)
    hold = card["use"]["grip"][0]
    assert hold["name"] == "hold" and hold["type"] == "hold" and hold["axis"] == [0.0, 0.0, 1.0]   # toward the head
    z0, z1 = hold["center"][2] - hold["length"] / 2, hold["center"][2] + hold["length"] / 2
    assert z0 == pytest.approx(LAY.POINT_H, abs=1e-5) and z1 == pytest.approx(LAY.shank_top(), abs=1e-5)
    assert hold["length"] > 0.085                                                 # a fist's width
    m = LAY.spike_mesh()
    z = LAY.stations()
    ring = _ring_at(m, z[np.argmin(np.abs(z - hold["center"][2]))])
    perimeter = float(np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1).sum())
    assert hold["radius"] == pytest.approx(perimeter / (2.0 * math.pi), rel=0.01)   # as long round as the shank
    tip = card["use"]["anchor"][0]
    assert tip == {"name": "tip", "point": [0.0, 0.0, 0.0], "dir": [0.0, 0.0, -1.0]}
    looks = {e["name"]: e["point"] for e in card["use"]["look"]}
    assert looks["head"] == [0.0, 0.0, LAY.LENGTH]
    assert looks["band"][2] == pytest.approx(sum(LAY.band()) / 2, abs=1e-5)


def test_the_length_option_stretches_the_shank_and_keeps_the_head_point_and_section():
    assert LAY.options({}) == {"length": LAY.LENGTH}
    assert LAY.options({"length": 0.2, "gold": "#ea9d34"}) == {"length": 0.2}      # the palette's slots are not options
    for bad in (True, "long", 0.05, 0.5, float("nan")):
        with pytest.raises(LAY.LayoutError):
            LAY.options({"length": bad})
    m = LAY.spike_mesh(0.2)
    assert m.is_closed() and m.bbox()[1][2] == pytest.approx(0.2)
    assert np.ptp(m.V[:, 0]) == pytest.approx(LAY.HEAD[0], abs=1e-6)
    b0, b1 = LAY.band(0.2)
    share = ((b0 + b1) / 2 - LAY.POINT_H) / (LAY.shank_top(0.2) - LAY.POINT_H)
    assert share == pytest.approx(LAY.BAND_AT) and b1 - b0 == pytest.approx(LAY.BAND_H)
    card = LAY.card("s", LAY.ROLES, 0.2)
    assert card["size"][2] == 0.2 and card["use"]["look"][0]["point"] == [0.0, 0.0, 0.2]
    assert card["use"]["grip"][0]["length"] == pytest.approx(LAY.shank_top(0.2) - LAY.POINT_H, abs=1e-5)


@pytest.mark.parametrize("palette", ["rose-pine-moon", "rose-pine-dawn"])
def test_the_gold_is_the_palette_s_and_the_engraving_darker(palette):
    assert LAY.ROLES["metal"] == "gold" and not any("#" in spec for spec in LAY.ROLES.values())
    c = Colors(PAL.PALETTES[palette])
    assert c.hex(LAY.ROLES["metal"]) == PAL.PALETTES[palette]["gold"]
    lum = {role: sum(k * v for k, v in zip((0.2126, 0.7152, 0.0722), PAL.linear(c.hex(spec))))
           for role, spec in LAY.ROLES.items()}
    assert lum["engraving"] < 0.7 * lum["metal"]
