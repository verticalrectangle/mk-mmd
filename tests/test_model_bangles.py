"""bpy-free invariants of the bangles part: a ring sits round its limb where it is asked, off the skin, and moves with
it. Built on the girl base's body."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts import outfit as OU

RINGS = ('bangles.rings=[{on = "ankle.L", toward = "knee.L", at = 0.04}, '
         '{on = "ankle.R", toward = "knee.R", at = 0.04, color = "silver", width = 0.012}]')


def test_a_ring_goes_round_its_limb_off_the_skin_and_moves_with_it(tmp_path):
    spec = SP.load("base:girl", ['model.parts=["body", "bangles"]', RINGS])
    ctx = BD.BuildCtx(spec, tmp_path, seed=1)
    part = next(p for p in BD.run(spec, ctx=ctx) if p.name == "bangles")
    assert [m.name for m in part.materials] == ["バングルgold", "バングルsilver"]
    mesh = part.meshes[0]
    V, fm = np.asarray(mesh.verts), np.asarray(mesh.face_mat)
    ctx.part = "test"
    fit = OU.make_fit(ctx)
    skin, S = fit.skin, fit.S                                                  # sizes are metres x S
    for side, k, width in (("L", 0, 0.008), ("R", 1, 0.012)):
        ring = V[np.unique([i for f, m in zip(mesh.faces, fm) if m == k for i in f])]
        ankle, knee = ctx.land[f"ankle.{side}"], ctx.land[f"knee.{side}"]
        along = (ring - ankle) @ ((knee - ankle) / np.linalg.norm(knee - ankle))
        assert along.min() == pytest.approx((0.04 - 0.5 * width) * S, abs=0.002)     # where and as wide as asked
        assert along.max() == pytest.approx((0.04 + 0.5 * width) * S, abs=0.002)
        d = skin.closest(ring)[0]
        assert 0.001 < d.min() and d.max() < 0.012                             # round the leg, off its skin
        assert np.linalg.norm(ring.mean(0) - (ankle + 0.04 * S * (knee - ankle) / np.linalg.norm(knee - ankle))) < 0.01
        assert np.allclose(part.info["landmarks"][f"bangle.{k}"], ring.mean(0), atol=0.01)
    legs = sum(w for b, w in mesh.weights.items() if "足" in b or "ひざ" in b)
    assert legs.min() > 0.95                                                   # the leg's own weights


def test_a_ring_on_no_landmark_says_so(tmp_path):
    spec = SP.load("base:girl", ['model.parts=["body", "bangles"]', 'bangles.rings=[{on = "ankle.L", toward = "tail"}]'])
    with pytest.raises(BD.BuildError, match="toward = 'tail': no such landmark"):
        BD.run(spec, ctx=BD.BuildCtx(spec, tmp_path, seed=1))
