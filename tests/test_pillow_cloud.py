"""The pillow cloud's layout (mkmmd.blender.library.props.pillow_cloud_layout, no bpy): a closed, outward pillow of two
stuffed panels sewn round a seam on the mid-plane, the cloud it puffs into as the same points on a cumulus (no fold, every
point on the cloud), the cord in the seam that sinks into the cloud, the fabric's uv laid flat, and the card's use points
on the pillow they name. The Blender half (the `puff` drivers) is tests/test_pillow_cloud_build.py."""
import numpy as np
import pytest

from mkmmd.blender.library.props import pillow_cloud_layout as LAY
from mkmmd.core import shell as S


@pytest.fixture(scope="module")
def shapes():
    pillow, cloud = LAY.meshes()
    return pillow, cloud


def test_the_pillow_is_two_stuffed_panels_closed_and_outward_with_the_seam_on_the_mid_plane(shapes):
    pillow, _ = shapes
    assert pillow.is_closed() and pillow.volume() > 0
    lo, hi = pillow.bbox()
    assert hi - lo == pytest.approx([LAY.W, LAY.T, LAY.H], abs=1e-6)
    assert (lo + hi) / 2 == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)              # origin at its centre
    seam = pillow.V[LAY.seam_ring()]
    assert np.abs(seam[:, 1]).max() < 1e-12                                        # the panels meet on y = 0
    assert len(np.unique(LAY.seam_ring())) == len(LAY.seam_ring()) == 2 * (LAY.NU + LAY.NV)
    front, back, _ = LAY.topology()
    assert pillow.V[front[1:-1, 1:-1].ravel(), 1] == pytest.approx(-pillow.V[back[1:-1, 1:-1].ravel(), 1])
    mid = pillow.V[front[LAY.NU // 2, LAY.NV // 2]]
    assert mid[1] == pytest.approx(-0.5 * LAY.T)                                   # plumpest in the middle


def test_its_sides_pull_in_and_its_ears_stand_out(shapes):
    pillow, _ = shapes
    front, _, _ = LAY.topology()
    corner = pillow.V[front[-1, -1]]
    side = pillow.V[front[-1, LAY.NV // 2]]
    assert corner[0] == pytest.approx(0.5 * LAY.W) and side[0] < corner[0] - 0.01  # the side bows in at its middle
    ear = pillow.V[front[-3, -3]]
    assert abs(ear[1]) < 0.25 * LAY.T                                              # thin near the corner


def test_the_cloud_is_the_same_points_on_a_cumulus_with_no_fold(shapes):
    pillow, cloud = shapes
    assert cloud.shape == pillow.V.shape
    m = S.Mesh(cloud, pillow.Q)
    assert m.is_closed() and m.volume() > 4.0 * pillow.volume()                    # it puffs up
    assert np.abs(LAY.cloud_sdf(cloud)).max() < 1e-6                               # every point on the cloud
    nq, _ = m.face_vectors()
    cq, _ = m.face_centres()
    cos = (nq / np.linalg.norm(nq, axis=1)[:, None] * LAY.cloud_normals(cq)).sum(1)
    assert cos.min() > 0.5                                                         # every face looks out of the cloud
    lo, hi = cloud.min(0), cloud.max(0)
    assert lo[2] == pytest.approx(LAY.CLOUD_BASE, abs=0.01)                        # a flat underside
    assert hi[0] - lo[0] > 2.0 * LAY.W and hi[2] - lo[2] > LAY.H                   # wider and taller than the pillow


def test_the_cord_follows_the_seam_and_sinks_into_the_cloud(shapes):
    pillow, cloud = shapes
    cord, thread = LAY.piping_meshes(pillow.V, cloud)
    assert cord.is_closed() and cord.volume() > 0 and thread.shape == cord.V.shape
    n = len(LAY.seam_ring())
    centre = cord.V.reshape(n, -1, 3).mean(1)
    ring = pillow.V[LAY.seam_ring()]
    assert np.linalg.norm(centre - ring, axis=1).max() < 0.02                     # on the seam (eased at the ears)
    assert np.median(np.linalg.norm(centre - ring, axis=1)) < 1e-3
    assert (LAY.cloud_sdf(thread) < -0.25 * LAY.PIPING[1]).all()                  # out of sight inside the cloud


def test_the_fabric_lies_flat_so_the_print_keeps_its_size_up_the_walls(shapes):
    pillow, _ = shapes
    front, _, _ = LAY.topology()
    P = pillow.V[front]
    uv = pillow.UV[front]
    du = np.linalg.norm(np.diff(P, axis=0), axis=2)                                # 3D steps along u
    duv = np.abs(np.diff(uv[..., 0], axis=0)) * LAY.W                             # the same steps on the flat fabric
    assert duv == pytest.approx(du, rel=1e-9)
    _, back, _ = LAY.topology()
    assert pillow.UV[back[3, 5]] == pytest.approx(pillow.UV[front[3, 5]])          # one print on both panels


def test_the_card_puts_the_hug_on_its_back_and_the_palms_on_its_ends(shapes):
    pillow, cloud = shapes
    card = LAY.card("pl", {}, None, cloud)
    hug = card["use"]["anchor"][0]
    assert hug["name"] == "hug" and hug["dir"] == [0.0, -1.0, 0.0]
    assert hug["point"] == pytest.approx([0.0, pillow.V[:, 1].max(), 0.0], abs=1e-4)   # the middle of its back
    rest = {r["name"]: r for r in card["use"]["rest"]}
    seam = pillow.V[LAY.seam_ring()]
    end = seam[np.abs(seam[:, 2]) < 1e-9, 0]                                       # the seam at mid-height
    for nm, s in (("hold_L", 1.0), ("hold_R", -1.0)):
        assert rest[nm]["normal"] == [s, 0.0, 0.0]                                 # out of the pillow
        assert rest[nm]["center"][0] == pytest.approx(s * np.abs(end).max(), abs=1e-4)
    assert card["params"] == {"puff": 0.0} and card["origin"] == "center" and card["front"] == "-Y"
    assert card["cloud"]["bounds"]["min"] == pytest.approx(cloud.min(0), abs=1e-4)
    assert card["objects"] == ["pl_pillow", "pl_piping"]


@pytest.mark.parametrize("v", [True, "1", float("nan"), 2.0, -1.0])
def test_a_bad_starting_puff_is_refused(v):
    with pytest.raises(ValueError, match="puff"):
        LAY.options({"puff": v})


def test_the_colour_roles_take_slots_hexes_and_blends():
    pal = LAY.Pal({"case": "#ff0000", "dots": "gold:1,rose:1"})
    assert pal.lin("case") == pytest.approx((1.0, 0.0, 0.0))
    assert set(pal.roles()) == set(LAY.ROLES) and pal.roles()["case"] == "#ff0000"
