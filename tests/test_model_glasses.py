"""bpy-free invariants of the glasses part on the girl base: the lenses are the size asked, their rims' tops sit where
`height` puts them on the eye opening, nothing of them comes nearer the face than `gap` in any morph, they ride the
head bone alone; the lens picture is nearly clear with one small glint at its upper outer edge."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import glasses as GL
from mkmmd.model.skin import closest_on_mesh


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    spec = SP.load("base:girl", ['model.parts=["body", "head", "glasses"]', "glasses.place.height=0.5"])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("glasses"), seed=1)
    BD.run(spec, ctx=ctx)
    return ctx.parts


def test_the_lenses_are_the_size_asked_and_sit_where_height_says(built):
    part = built["glasses"]
    PT.check(part)
    info = part.info
    assert info["size"] == pytest.approx(0.040, abs=1e-6) and info["wire"] == pytest.approx(0.0014, abs=1e-6)
    assert info["eye"]["rims_top_share"] == pytest.approx(0.5, abs=0.01)
    assert info["centres"]["L"][0] == pytest.approx(-info["centres"]["R"][0], abs=1e-9)
    mesh = part.meshes[0]
    assert list(mesh.weights) == ["頭"] and np.allclose(mesh.weights["頭"], 1.0)
    assert [m.name for m in part.materials] == list(GL.MATS) and part.materials[1].alpha_blend
    assert min(info["temples"]["back"]) > 0.010                                     # back along the head, not stuck
    assert not part.materials[0].self_shadow_map and not part.materials[0].drop_shadow


def test_nothing_of_them_comes_nearer_the_face_than_the_gap_in_any_morph(built):
    face = GL.Face(built["head"])
    mesh = built["glasses"].meshes[0]
    V = np.asarray(mesh.verts)
    tris = np.array([t for f in mesh.faces for t in ((f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1))])
    worst = np.inf
    for s in face.states:
        P = face.points(s)
        near = ((P >= V.min(0) - 0.01) & (P <= V.max(0) + 0.01)).all(1)
        if near.any():
            q = closest_on_mesh(P[near], V, tris)[0]
            worst = min(worst, float(np.linalg.norm(P[near] - q, axis=1).min()))
    assert worst >= 0.002 - 2e-4                                        # face vertices to the glasses' surface
    assert built["glasses"].info["gap"]["least"] >= 0.002 - 2e-4        # the glasses' wire and lenses to the face


def test_the_glint_is_one_crisp_arc_about_8_mm_long_at_the_upper_outer_edge_of_a_clear_lens():
    col = {k: np.array(v, float) for k, v in (("lens", (0.95, 0.98, 1.0)), ("glint", (1.0, 1.0, 1.0)))}
    R, size = 0.0193, 256
    img = GL.lens_picture(col, 0.0, SP.merge(GL.DEFAULTS["glint"], {}), R, size=size)
    a = img[..., 3].astype(float) / 255.0
    assert (a == 0.0).mean() > 0.9 and a.max() == 1.0                                # clear glass, a solid glint
    ys, xs = np.nonzero(a > 0.5)
    assert xs.mean() > size / 2 and ys.mean() < size / 2                             # its upper outer edge
    assert np.hypot(np.ptp(xs), np.ptp(ys)) * 2 * R / size == pytest.approx(0.008, abs=0.0015)
