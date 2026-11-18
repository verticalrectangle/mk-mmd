"""bpy-free invariants of the hangings part on the girl base's body and a stand-in skirt: each thing hangs from its
cord's place, comes to rest `gap` off what it hangs over, faces out, swings on an accessory chain of two bones from the
lower body; the jug carries its brushed mark on its belly's front."""
import numpy as np
import pytest

from mkmmd.core import families
from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import hangings as HG

STAND_IN = ('hangings.stand_in={axis = [0.0, -0.03], profile = [[0.86, 0.13, 0.11], [0.80, 0.15, 0.12], '
            '[0.70, 0.19, 0.17], [0.60, 0.20, 0.18], [0.50, 0.16, 0.15]]}')
ITEMS = ('hangings.items=[{kind = "jug", at = [40.0, 0.84]}, {kind = "book", at = [75.0, 0.84]}]')


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    spec = SP.load("base:girl", ['model.parts=["body", "hangings"]', ITEMS, STAND_IN])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("hangings"), seed=1)
    BD.run(spec, ctx=ctx)
    return ctx


def test_each_thing_rests_off_what_it_hangs_over_and_faces_out(built):
    part = built.parts["hangings"]
    PT.check(part)
    axis = np.mean([built.land["leg.L"], built.land["leg.R"]], 0)
    under = HG.resting_surface({"body": built.parts["body"]}, SP.load("base:girl", [STAND_IN])["hangings"]["stand_in"],
                               axis)
    mesh = part.meshes[0]
    V = np.asarray(mesh.verts)
    for i, it in enumerate(part.info["items"]):
        names = [f"吊り飾り{'AB'[i]}1", f"吊り飾り{'AB'[i]}2"]
        thing = V[mesh.weights[names[1]] > 0.99]
        assert np.abs(under.closest(thing)[0]).min() >= 0.002 - 2e-4                    # resting 2 mm off it
        assert it["cord"]["top"][2] == pytest.approx(0.84, abs=1e-6)
        assert it["cord"]["length"] == pytest.approx(0.045, abs=1e-6)
        theta = np.radians(it["azimuth"])
        out = np.array([np.sin(theta), -np.cos(theta)])
        assert (it["middle"][:2] - it["cord"]["top"][:2]) @ out > 0.0                      # it leans out, not in
    jug, book = part.info["items"]
    assert jug["across"] == pytest.approx(0.062, abs=0.001) and book["across"] == pytest.approx(0.0714, abs=0.001)


def test_a_garment_wound_inward_still_has_its_outside_in_front():
    V, T = HG.stand_in_mesh(np.array([0.0, 0.0]), [[0.9, 0.15, 0.15], [0.6, 0.2, 0.2]])
    skirt = PT.Part("skirt", meshes=[PT.Mesh("skirt", V, [tuple(t[::-1]) for t in T])])
    d = HG.resting_surface({"skirt": skirt}, {}, np.zeros(2)).closest(np.array([[0.3, 0.0, 0.75], [0.0, 0.0, 0.75]]))[0]
    assert d[0] > 0.0 and d[1] < 0.0                                                  # outside in front, inside behind


def test_a_thing_riding_a_cloth_takes_the_weights_where_it_lies_on_it():
    """Two chains' cloth, one at her left, one at her right: a jug lying on the left one moves with the left chain."""
    left = PT.Mesh("a", np.array([[0.20, 0.0, z] for z in (0.80, 0.75, 0.70)]), [(0, 1, 2)],
                   weights={"skirt_L": np.ones(3), "hips": np.full(3, 0.2)})
    right = PT.Mesh("b", np.array([[-0.20, 0.0, z] for z in (0.80, 0.75, 0.70)]), [(0, 1, 2)],
                    weights={"skirt_R": np.ones(3)})
    on = HG.riding_weights(np.vstack([left.verts, right.verts]), HG.merged_weights([left, right]),
                           np.array([0.24, 0.0, 0.74]), k=3)
    assert set(on) == {"skirt_L", "hips"} and on["skirt_L"] == pytest.approx(1.0 / 1.2)
    assert sum(on.values()) == pytest.approx(1.0)


def test_they_swing_on_accessory_chains_from_the_lower_body(built):
    part = built.parts["hangings"]
    bones = {b.name: b for b in part.bones}
    for tag in "AB":
        assert bones[f"吊り飾り{tag}1"].parent == "下半身" and bones[f"吊り飾り{tag}2"].parent == f"吊り飾り{tag}1"
        assert families.classify(f"吊り飾り{tag}1") == "accessory"
    bodies = {b.name: b for b in part.bodies}
    assert bodies["吊り飾りB2"].shape == "box" and bodies["吊り飾りA2"].shape == "capsule"
    assert all(b.mode == "dynamic" for b in part.bodies) and len(part.joints) == 4


def test_the_jug_carries_its_mark_on_the_front_of_its_belly():
    _, _, lay, _ = HG.jug(SP.merge(HG.SIZES["jug"], {}))
    col = {k: np.array(v, float) for k, v in (("glaze", (0.95, 0.95, 0.93)), ("glaze_shadow", (0.77, 0.76, 0.74)),
                                             ("mark", (0.23, 0.16, 0.16)))}
    img = HG.jug_picture(col, lay, 0.030).astype(float) / 255.0
    dark = img[..., 0] < 0.5
    ys, xs = np.nonzero(dark)
    h, w = dark.shape
    assert abs(xs.mean() / w - 0.5) < 0.03                                         # round its front
    assert abs((1.0 - ys.mean() / h) - lay["belly_v"]) < 0.04                      # on its belly
    assert np.ptp(ys) / h == pytest.approx(0.030 / lay["arc"], rel=0.15)           # 30 mm tall
