"""The carrot plush's layout (mkmmd.blender.library.props.carrot_plush_layout, no bpy): closed outward meshes in the card's
frame (the body's middle at the origin, +Z from the tip to the leaves, the face toward -Y), the gores' seams, the face's
graphic layers lying just off the body on the front, and the card's use points on the body."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import carrot_plush_layout as LAY
from mkmmd.core import shell as S


def _parts(m):
    """Vertex sets of the mesh's connected parts."""
    parent = np.arange(len(m.V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in list(m.Q) + list(m.T):
        r0 = find(int(f[0]))
        for v in f[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    roots = np.array([find(i) for i in range(len(m.V))])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _sub(m, ids):
    keep = np.zeros(len(m.V), bool)
    keep[ids] = True
    remap = np.cumsum(keep) - 1
    return S.Mesh(m.V[keep], remap[m.Q[keep[m.Q].all(1)]], remap[m.T[keep[m.T].all(1)]])


MESHES = {"body": LAY.body_mesh, "leaves": LAY.leaves_mesh, "face": LAY.face_mesh, "cheeks": LAY.cheeks_mesh,
          "stitches": LAY.stitches_mesh}


@pytest.mark.parametrize("which", sorted(MESHES))
def test_every_part_is_closed_and_faces_out(which):
    m = MESHES[which]()
    for p in _parts(m):
        sub = _sub(m, p)
        assert sub.is_closed() and sub.volume() > 0, which


def test_the_body_is_a_plump_cone_0_34_long_about_the_origin_with_leaves_0_12_above_it():
    m = LAY.body_mesh()
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(-0.17) and hi[2] == pytest.approx(0.17) and LAY.BODY_L == pytest.approx(0.34)
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    assert r.max() == pytest.approx(LAY.R_MAX, abs=1e-6) and abs(m.V[np.argmax(r), 2] - LAY.Z_WIDE) < 0.01
    tip = m.V[:, 2] < LAY.Z_TIP + 0.02
    assert r[tip].max() > 0.025                                                     # a soft round tip, not a point
    leaves = LAY.leaves_mesh()
    assert leaves.bbox()[1][2] - LAY.Z_TOP == pytest.approx(0.12, abs=0.012)
    assert len(_parts(leaves)) == 3 and leaves.bbox()[0][2] < LAY.Z_TOP            # three, sewn into the top


def test_the_gores_leave_seam_valleys_with_the_face_on_a_gore_and_a_seam_down_the_back():
    th = np.linspace(-math.pi, math.pi, 3601)[:-1]
    g = LAY.gores(th)
    valleys = (g < np.roll(g, 1)) & (g <= np.roll(g, -1))
    assert valleys.sum() == LAY.GORES and 5 <= LAY.GORES <= 6
    assert g.min() == pytest.approx(1.0 - LAY.SEAM, abs=1e-4) and g.max() == pytest.approx(1.0)
    assert LAY.gores(LAY.FRONT) == pytest.approx(1.0) and LAY.gores(0.5 * math.pi) == pytest.approx(1.0 - LAY.SEAM)
    m = LAY.body_mesh()
    ring = np.abs(m.V[:, 2] - m.V[np.argmin(np.abs(m.V[:, 2]))][2]) < 1e-12
    r = np.hypot(m.V[ring, 0], m.V[ring, 1])
    assert r.max() - r.min() == pytest.approx(LAY.SEAM * LAY.radius(m.V[ring][0, 2]), rel=0.05)   # creased valleys


@pytest.mark.parametrize("which, thick", [("face", LAY.EYE_T + LAY.SHINE_T + 0.0016), ("cheeks", LAY.CHEEK_T)])
def test_the_face_s_layers_lie_just_off_the_front_of_the_body(which, thick):
    m = MESHES[which]()
    assert (m.V[:, 1] < -0.04).all()                                                # on the front
    th = np.arctan2(m.V[:, 1], m.V[:, 0])
    P, N = LAY.surface(th, m.V[:, 2])
    out = np.einsum("ij,ij->i", m.V - P, N)
    assert out.min() > 0.0 and out.max() < LAY.LIFT + thick + 0.0005               # never inside, never far off
    assert m.bbox()[0][0] == pytest.approx(-m.bbox()[1][0], abs=1e-6)              # mirrored about the middle


def test_the_card_s_points_are_on_the_body():
    card = LAY.card("p", LAY.options({}), [0.15, 0.15, 0.46])
    assert card["origin"] == "center" and card["front"] == "-Y"
    (hold,) = card["use"]["anchor"]
    assert hold["name"] == "hold" and hold["dir"] == [0.0, -1.0, 0.0]
    x, y, z = hold["point"]
    assert x == pytest.approx(0.0, abs=1e-6) and z == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(LAY.radius(0.0) * LAY.gores(0.5 * math.pi), abs=1e-5)  # the middle of its back
    (face,) = card["use"]["look"]
    assert face["name"] == "face" and face["point"][1] < 0.0
    assert face["point"] == pytest.approx([0.0, -LAY.radius(LAY.EYE[0][1]), LAY.EYE[0][1]], abs=1e-5)
    assert card["objects"] == ["p_body", "p_leaves", "p_face", "p_cheeks", "p_stitches"]


def test_colour_roles_are_checked():
    assert LAY.options({"cheek": "#ff99aa", "base": "#000000", "seed": 4})["cheek"] == "#ff99aa"
    for slots, frag in (({"body": "orange"}, "body"), ({"face": 3}, "face"), ({"leaf": "#12345"}, "leaf")):
        with pytest.raises(ValueError, match=frag):
            LAY.options(slots)
