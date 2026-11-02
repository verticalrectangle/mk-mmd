"""The spike mallet's layout (mkmmd.blender.library.props.spike_mallet_layout, no bpy): closed parts in the card's frame,
the barrel head across the top of the handle, the hoops on it, the card against the meshes (the hold on the handle with
the handle's radius, the striking face's anchor on the face), forms with no box in them and colour roles that come from
the palette."""
import numpy as np
import pytest

from mkmmd.blender.library.props import spike_mallet_layout as LAY
from mkmmd.blender.library.props.cafe_colors import Colors
from mkmmd.core import form as F
from mkmmd.core import palette as PAL


def _placed():
    """{part: Mesh} in the prop's frame."""
    return {k: m.moved(loc) for k, (m, loc) in LAY.meshes().items()}


def _radius_along_z(m, z, tol=0.002):
    """The mean distance from the Z axis of the vertices of a lathe within tol of height z."""
    near = np.abs(m.V[:, 2] - z) < tol
    return float(np.hypot(m.V[near, 0], m.V[near, 1]).mean())


def test_every_part_is_closed_and_the_whole_is_the_card_s_size():
    parts = _placed()
    for name, m in parts.items():
        assert m.is_closed() and m.volume() > 0, name
    lo = np.min([m.bbox()[0] for m in parts.values()], axis=0)
    hi = np.max([m.bbox()[1] for m in parts.values()], axis=0)
    card = LAY.card("m", LAY.ROLES)
    assert hi - lo == pytest.approx(card["size"], abs=0.002)
    assert lo[2] == pytest.approx(0.0, abs=1e-9)                                  # the handle's end at the origin
    assert card["size"] == pytest.approx([0.46, 0.24, 1.24], abs=0.005)           # the frame promised to the moves
    assert card["objects"] == ["m_head", "m_bands", "m_handle"] and card["front"] == "-Y"


def test_the_mallet_is_forms_not_boxes():
    parts = _placed()
    V, T, owner, off = [], [], [], 0
    for i, m in enumerate(parts.values()):
        V.append(m.V)
        T.append(m.triangles() + off)
        owner.append(np.full(len(m.triangles()), i))
        off += len(m.V)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), list(parts))
    assert r["score"] < 0.05, (r["score"], r["components"][:4])


def test_the_head_is_a_barrel_across_the_top_with_its_axis_along_x():
    head = LAY.head_mesh()
    lo, hi = head.bbox()
    assert hi - lo == pytest.approx([LAY.HEAD_L, 2 * LAY.R_MID, 2 * LAY.R_MID], abs=1e-3)
    assert (lo + hi) / 2 == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)             # centred on its middle
    assert LAY.HEAD_Z == pytest.approx(1.10, abs=0.03)
    r = np.hypot(head.V[:, 1], head.V[:, 2])
    mid, end = np.abs(head.V[:, 0]) < 0.01, np.abs(np.abs(head.V[:, 0]) - 0.2) < 0.006
    assert r[mid].max() > r[end].max() + 0.004                                     # fuller in the middle: a barrel
    face = head.V[:, 0] > LAY.HEAD_L / 2 - 1e-9
    assert face.sum() == 1 and np.abs(head.V[face, 1:]).max() < 1e-12             # the +X face's crown on the axis
    rim = np.abs(head.V[:, 0]) > LAY.HEAD_L / 2 - LAY.CROWN - 1e-6
    assert r[rim].max() < LAY.R_END - LAY.ROLL / 4                                # the rim rolled over, not a sharp edge


def test_the_hoops_ride_near_the_faces_proud_of_the_wood():
    bands = LAY.band_mesh()
    x = bands.V[:, 0]
    assert set(np.sign(np.round(x, 6))) == {-1.0, 1.0}                             # one near each face
    assert np.abs(x).max() < LAY.HEAD_L / 2 - LAY.CROWN - LAY.ROLL                # inside the rolled rims
    r = np.hypot(bands.V[:, 1], bands.V[:, 2])
    wood = LAY.barrel_r(x)
    assert (r - wood).max() == pytest.approx(LAY.BAND_T, abs=2e-4)                 # standing proud ...
    assert (r - wood).min() < -LAY.BAND_SINK / 2                                   # ... from inside the wood


def test_the_hold_is_the_handle_a_fist_closes_on_with_its_radius_there():
    hold = LAY.card("m", LAY.ROLES)["use"]["grip"][0]
    assert hold["name"] == "hold" and hold["type"] == "hold" and hold["axis"] == [0.0, 0.0, 1.0]   # toward the head
    z0, z1 = hold["center"][2] - hold["length"] / 2, hold["center"][2] + hold["length"] / 2
    assert hold["length"] >= 0.35                                                  # two fists 0.25 m apart
    assert hold["center"][:2] == [0.0, 0.0] and hold["center"][2] == pytest.approx(0.40, abs=0.03)
    knob = float(np.linspace(0.0, 0.1, 201)[np.argmax(LAY.handle_r(np.linspace(0.0, 0.1, 201)))])
    assert knob < z0 and z1 < LAY.HEAD_Z - LAY.R_MID - 0.1                         # over the knob, well under the head
    handle = LAY.handle_mesh()
    centre = hold["center"][2]
    assert _radius_along_z(handle, centre) == pytest.approx(hold["radius"], abs=2e-4)
    for s, r in hold["profile"]:                                                   # s from the head end
        assert _radius_along_z(handle, z1 - s) == pytest.approx(r, abs=2e-4)
    assert hold["profile"][0][0] == 0.0 and hold["profile"][-1][0] == pytest.approx(hold["length"])
    assert 0.018 < hold["radius"] < 0.024


def test_the_face_anchor_is_the_middle_of_the_striking_face():
    card = LAY.card("m", LAY.ROLES)
    face = card["use"]["anchor"][0]
    assert face["name"] == "face" and face["dir"] == [1.0, 0.0, 0.0]
    head = _placed()["head"]
    tip = head.V[np.argmax(head.V[:, 0])]
    assert face["point"] == pytest.approx(tip, abs=1e-6)                           # the +X face's middle, on it
    assert max(m.V[:, 0].max() for m in _placed().values()) == pytest.approx(face["point"][0], abs=1e-6)
    looks = {e["name"]: e["point"] for e in card["use"]["look"]}
    assert looks["face"] == face["point"] and looks["head"] == [0.0, 0.0, LAY.HEAD_Z]
    assert looks["grip"] == card["use"]["grip"][0]["center"]


def test_the_handle_runs_from_its_end_through_the_head_to_the_tenon_on_its_top():
    handle = LAY.handle_mesh()
    assert handle.bbox()[0][2] == pytest.approx(0.0, abs=1e-9)
    top = handle.V[:, 2] > LAY.LENGTH - 0.005
    x, y = handle.V[top, 0], handle.V[top, 1]
    assert handle.V[top, 2] == pytest.approx(LAY.barrel_top(x, y) + LAY.TENON_PROUD, abs=1e-9)
    assert LAY.handle_r(0.026) > LAY.SHAFT_R + 0.004                               # the swell at the end
    assert LAY.handle_r(LAY.HEAD_Z - LAY.R_MID) > LAY.SHAFT_R + 0.003               # the flare into the head


@pytest.mark.parametrize("palette", ["rose-pine-moon", "rose-pine-dawn"])
def test_the_colour_roles_are_palette_specs_and_the_wood_stays_warm_and_lighter_than_the_iron(palette):
    assert not any("#" in spec for spec in LAY.ROLES.values())
    c = Colors(PAL.PALETTES[palette])
    rgb = {role: PAL.linear(c.hex(spec)) for role, spec in LAY.ROLES.items()}
    lum = {role: 0.2126 * r + 0.7152 * g + 0.0722 * b for role, (r, g, b) in rgb.items()}
    for wood in ("wood", "grain", "handle"):
        assert rgb[wood][0] > rgb[wood][2] and lum[wood] > 1.5 * lum["band"]
    assert lum["handle"] > lum["wood"] > lum["grain"]
