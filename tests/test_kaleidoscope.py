"""The cardboard kaleidoscope toy's layout (mkmmd.core.kaleidoscope, no bpy): its size and how it lies, closed outward
meshes, the paper's seam and how its print is mapped, the beads settled in the chamber, the use points against the meshes
they name, and the options. The Blender half is checked in tests/test_kaleidoscope_build.py."""
import numpy as np
import pytest

from mkmmd.core import kaleidoscope as K
from mkmmd.core import shell as S

PARTS = {**K.eyepiece_meshes(), **{f"far_{k}": v for k, v in K.far_meshes().items()}, "tube": K.tube_mesh()}


def prop(mesh):
    return K.to_prop(mesh)


@pytest.mark.parametrize("part", sorted(PARTS))
def test_every_part_is_a_closed_outward_solid(part):
    m = PARTS[part]
    assert m.is_closed() and m.volume() > 0, part


def test_it_is_a_230_mm_tube_49_mm_across_lying_on_its_rims_with_the_eyepiece_toward_minus_y():
    allv = np.concatenate([prop(m).V for m in PARTS.values()])
    lo, hi = allv.min(0), allv.max(0)
    assert lo[2] == pytest.approx(0.0, abs=2e-5) and hi[2] == pytest.approx(2 * K.AXIS_Z, abs=2e-5)   # on the floor
    assert hi[0] - lo[0] == pytest.approx(2 * K.AXIS_Z, abs=2e-5)
    assert K.AXIS_Z == pytest.approx(K.RIM_R, abs=1e-4)                           # on the beads of the rims
    tube = prop(PARTS["tube"])
    assert 2 * K.TUBE_R == pytest.approx(0.049, abs=1e-4)
    assert np.hypot(tube.V[:, 0], tube.V[:, 2] - K.AXIS_Z).max() == pytest.approx(K.TUBE_R, abs=1e-9)
    assert tube.V[:, 2].min() > 0.002                                              # the tube hovers over the floor
    cap = prop(PARTS["cap"])
    assert cap.V[:, 1].min() == pytest.approx(-K.LENGTH / 2, abs=1e-9)            # the eyepiece cap is the -Y end


def test_the_rims_are_turned_with_a_groove_between_the_lip_and_the_bead():
    P = K.eyepiece_rim_profile()
    outer = P[(P[:, 0] > K.TUBE_R + 0.0004) & (P[:, 1] > 0.001) & (P[:, 1] < K.COLLAR_EYE - 0.001)]
    r = outer[np.argsort(outer[:, 1]), 0]
    dips = np.sum((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))
    assert dips >= 1 and r.max() == pytest.approx(K.RIM_R, abs=3e-4)              # a groove; the bead is the widest


def test_the_paper_s_seam_is_a_step_of_its_thickness_at_the_bottom():
    t = prop(PARTS["tube"])
    edge = t.Qm == K.m("paper_edge")
    centres = t.V[t.Q[edge]].mean(1)
    assert edge.sum() > 0 and np.all(centres[:, 2] < K.AXIS_Z - 0.02)            # under the axis: the bottom
    paper = t.Q[t.Qm == K.m("paper")]
    rad = np.hypot(t.V[paper][..., 0], t.V[paper][..., 2] - K.AXIS_Z)
    assert rad.max() - rad.min() == pytest.approx(K.PAPER_T, abs=1e-9)


def test_the_print_reads_unmirrored_from_outside_its_top_row_at_the_eyepiece():
    t = prop(PARTS["tube"])
    q = t.Q[t.Qm == K.m("paper")]
    P, U = t.V[q], t.UV[q]
    dp1, dp3 = P[:, 1] - P[:, 0], P[:, 3] - P[:, 0]
    du1, du3 = U[:, 1] - U[:, 0], U[:, 3] - U[:, 0]
    det = du1[:, 0] * du3[:, 1] - du1[:, 1] * du3[:, 0]
    dPdU = (dp1 * du3[:, 1:2] - dp3 * du1[:, 1:2]) / det[:, None]                 # the surface's tangents along U, V
    dPdV = (dp3 * du1[:, 0:1] - dp1 * du3[:, 0:1]) / det[:, None]
    out = P.mean(1) - np.array([0.0, 0.0, K.AXIS_Z]) * [1, 0, 1]
    out[:, 1] = 0.0
    assert np.all(np.einsum("ij,ij->i", np.cross(dPdU, dPdV), out) > 0)           # right-handed: not mirrored
    v_eye = t.UV[np.argmin(t.V[:, 1])][1]
    assert v_eye > 1.0 and t.UV[np.argmax(t.V[:, 1])][1] < 0.0                    # V = 1 at the eyepiece end


def test_the_cap_s_dish_is_painted_and_the_rest_is_the_rim():
    cap = PARTS["cap"]
    face = cap.Q[cap.Qm == K.m("face")]
    assert len(face) and np.hypot(cap.V[face][..., 0], cap.V[face][..., 1]).max() <= K.FACE_R + 1e-3
    assert (cap.Qm == K.m("rim")).sum() > len(face)


def _apart(a, b, gap=0.0002):
    """Whether two chamber pieces' bounding cylinders about the axis are clear of each other."""
    return (np.hypot(*(a["centre"][:2] - b["centre"][:2])) >= a["size"] + b["size"] + gap - 1e-12
            or abs(a["centre"][2] - b["centre"][2]) >= a["half"] + b["half"] + gap - 1e-12)


def test_beads_settle_in_the_chamber_without_overlapping_and_are_seeded():
    lay = K.chamber_layout(3)
    assert len(lay) >= 60 and {p["kind"] for p in lay} == set(K.KINDS)
    c = np.array([p["centre"] for p in lay])
    size = np.array([p["size"] for p in lay])
    assert all(_apart(a, b) for i, a in enumerate(lay) for b in lay[i + 1:])     # none through another
    assert np.all(np.hypot(c[:, 0], c[:, 1]) + size <= K.BORE_R)                  # inside the bore
    assert all(K.CHAMBER[0] <= p["centre"][2] - p["half"] and p["centre"][2] + p["half"] <= K.CHAMBER[1] for p in lay)
    pile = lay[:-3]                                                              # all but the three strays
    assert all(p["centre"][1] >= K.BORE_R * (1.0 - 2.0 * K.FILL) - 1e-9 for p in pile)   # under the pile's top
    assert np.mean([p["centre"][1] for p in pile]) > 0.1 * K.BORE_R             # mostly below the axis (+y down)
    for p in pile:                                                               # each rests on the bore or a piece
        lower = dict(p, centre=p["centre"] + [0.0, 0.0004, 0.0])
        on_bore = np.hypot(*lower["centre"][:2]) > K.BORE_R - p["size"] - 0.0003
        assert on_bore or not all(_apart(lower, o) for o in lay if o is not p)
    mesh = K.chamber_mesh(3)
    assert mesh.is_closed() and mesh.volume() > 0
    lo, hi = mesh.bbox()
    assert lo[2] >= K.CHAMBER[0] and hi[2] <= K.CHAMBER[1]                        # between the two glasses
    again = K.chamber_layout(3)
    assert all(np.array_equal(a["centre"], b["centre"]) and a["turn"] == b["turn"] for a, b in zip(lay, again))
    other = np.array([p["centre"] for p in K.chamber_layout(4)])
    assert other.shape != c.shape or not np.allclose(other, c)


def test_the_use_points_sit_on_the_parts_they_name():
    card = K.card("ks")
    use = card["use"]
    look = {e["name"]: np.array(e["point"]) for e in use["look"]}
    eye = look["eyepiece"]
    lens = prop(PARTS["lens"])
    apex = lens.V[np.argmin(lens.V[:, 1])]
    assert eye == pytest.approx(apex, abs=2e-5)                                   # the lens's apex, on the axis
    ferrule = prop(PARTS["ferrule"])
    opening = np.hypot(ferrule.V[:, 0], ferrule.V[:, 2] - K.AXIS_Z).min()
    a = use["anchor"][0]
    assert a["name"] == "eyehole" and a["object"] == "ks_eyehole" and a["point"] == pytest.approx(eye, abs=1e-5)
    assert a["radius"] == pytest.approx(opening, abs=1e-5) and a["dir"] == pytest.approx([0.0, -1.0, 0.0])
    assert look["chamber"] == pytest.approx([0.0, K.LENGTH / 2, K.AXIS_Z], abs=1e-5)
    side = use["rest"][0]
    for p in (side["a"], side["b"]):
        assert p[2] == pytest.approx(K.AXIS_Z + K.TUBE_R, abs=2e-4)               # on top of the tube
        assert -K.LENGTH / 2 + K.COLLAR_EYE < p[1] < K.LENGTH / 2 - K.FAR_RIM     # between the rims
    grip = use["grip"][0]
    assert grip["type"] == "cylinder" and grip["radius"] == K.TUBE_R and grip["axis"] == pytest.approx([0, 1, 0])
    assert grip["height"] == pytest.approx(K.LENGTH - K.COLLAR_EYE - K.FAR_RIM)
    assert look["grip"] == pytest.approx(grip["center"])
    assert card["size"] == pytest.approx([2 * K.AXIS_Z, K.LENGTH, 2 * K.AXIS_Z]) and card["front"] == "-Y"


def test_the_view_shows_its_whole_picture_centred_on_the_axis():
    v = PARTS["view"]
    face = v.V[:, 2] < K.VIEW_S + 1e-6
    assert v.UV[face].min() >= -0.03 and v.UV[face].max() <= 1.03
    centre = v.UV[np.argmin(np.hypot(v.V[:, 0], v.V[:, 1]))]
    assert centre == pytest.approx([0.5, 0.5])


def test_options_fill_defaults_and_refuse_bad_values():
    o = K.options({}, ("roses", "memphis"))
    assert (o["pattern"], o["seed"], o["wear"], o["beads"], o["turn"]) == ("roses", 1, 0.6, K.BEADS, 0.0)
    assert K.options({"beads": "love, #ffffff ,pine"})["beads"] == ("love", "#ffffff", "pine")
    for bad, frag in (({"pattern": "plaid"}, "pattern"), ({"wear": 2}, "wear"), ({"seed": "x"}, "seed"),
                      ({"beads": ""}, "beads"), ({"print": "love"}, "print")):
        with pytest.raises(ValueError, match=frag):
            K.options(bad, ("roses", "memphis"))


def test_a_piece_is_closed_whatever_its_kind():
    rng_layout = K.chamber_layout(11)
    for kind in K.KINDS:
        p = next(q for q in rng_layout if q["kind"] == kind)
        m = K.piece_mesh(p, 11, 0)
        assert m.is_closed() and m.volume() > 0, kind
        assert np.linalg.norm(m.V - p["centre"], axis=1).max() <= p["size"] + 1e-6, kind
    assert S.polygon_area(K.bit_outline(np.random.default_rng(5), 0.003)) > 0
