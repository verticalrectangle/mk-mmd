"""bpy-free invariants of the skirt part: it hangs from her hips off her skin and flares to its hem, the skin under it
carries its top and shares its lower part and its lace with a ring of tied chains, and it is painted as denim with
stitched seams. Built on the girl base's body."""
import numpy as np
import pytest

from mkmmd.core import families
from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts import outfit as OU
from mkmmd.model.parts import outfit_rig as ORIG
from mkmmd.model.parts import skirt


def test_the_skirt_hangs_from_her_hips_and_its_lower_part_from_a_ring_of_chains(tmp_path):
    spec = SP.load("base:girl", ['model.parts=["body", "skirt"]'])
    ctx = BD.BuildCtx(spec, tmp_path, seed=1)
    parts = {p.name: p for p in BD.run(spec, ctx=ctx)}
    part, info = parts["skirt"], parts["skirt"].info
    ctx.part = "test"
    fit = OU.make_fit(ctx)
    S = fit.S                                                              # sizes are metres x S
    assert info["z_top"] == pytest.approx(float(ctx.land["leg.L"][2]) + 0.085 * S, abs=1e-6)
    assert info["z_hem"] == pytest.approx(info["z_top"] - 0.215 * S, abs=1e-6)
    mesh = part.meshes[0]
    V, fm = np.asarray(mesh.verts), np.asarray(mesh.face_mat)
    assert [m.name for m in part.materials] == ["デニム", "スカートレース"]
    denim, lace = (V[np.unique([i for f, m in zip(mesh.faces, fm) if m == k for i in f])] for k in (0, 1))
    assert fit.skin.closest(denim)[0].min() > 0.0005                       # off her skin, its turned edges too
    top, hem = denim[denim[:, 2] > info["z_top"] - 0.01 * S], denim[denim[:, 2] < info["z_hem"] + 0.005 * S]
    assert np.ptp(hem[:, 0]) > np.ptp(top[:, 0]) + 0.04 * S                # an A-line
    assert lace[:, 2].max() < info["z_hem"] + 0.005 * S and lace[:, 2].min() < info["z_hem"] - 0.02 * S    # under the hem
    chains = sum(w for b, w in mesh.weights.items() if b.startswith("スカート"))
    legs = sum(w for b, w in mesh.weights.items() if "足" in b)
    low = V[:, 2] < info["z_hem"] + 0.002
    assert chains[V[:, 2] > info["z_hold"] + 0.002].max() < 1e-6           # the skin under it carries its top
    assert 0.35 < chains[low].min() and chains[low].max() < 0.7            # half of its hem and lace swings on the chains
    assert legs[low].mean() > 0.25                                         # the rest goes with her thighs
    bones = {b.name: b for b in part.bones}
    assert len(bones) == 36 and families.classify("スカート00_1") == "skirt" and bones["スカート00_1"].parent == "下半身"
    long = {r: np.linalg.norm(np.subtract(bones[f"スカート00_{r}"].tail, bones[f"スカート00_{r}"].head)) for r in (1, 2, 3)}
    assert long[1] < 0.5 * long[2] and long[2] == pytest.approx(long[3], rel=0.35)     # a short root, then what swings
    sideways = [j for j in part.joints if j.a.startswith("スカート") and j.b.startswith("スカート") and j.a[-1] == j.b[-1]]
    assert len(sideways) == 36                                             # every bone tied to its neighbour round the ring
    statics = [rb for rb in parts["body"].bodies if rb.mode == "static"]
    radius = {rb.name: rb.size[0] for rb in part.bodies}
    for b in part.bones:                                                   # no capsule past a root starts inside her
        if not b.name.endswith("_1"):
            seg = np.linspace(b.head, b.tail, 7)
            assert ORIG.union_sdf(seg, statics).min() > radius[b.name]


def test_the_denim_is_painted_with_its_seams_and_stitching():
    col = {"denim": np.array([0.2, 0.3, 0.8]), "thread": np.array([0.95, 0.7, 0.2])}
    img = skirt.picture(col, 0.9, 0.2, 0.025, SP.merge(skirt.DEFAULTS, {}), np.random.default_rng(1)).astype(float) / 255.0
    h = img.shape[0]
    gold = (img[..., 0] > 0.7) & (img[..., 2] < 0.5)
    hem = int((0.2 - 0.006) / 0.2 * h)                                    # a row of stitches 6 mm over the hem
    assert 0.5 < gold[hem - 1:hem + 2].any(0).mean() < 0.9                # dashes of thread all the way round
    cloth = img[int(0.5 * h)]
    assert gold[int(0.5 * h)].mean() < 0.1 and cloth[:, 2].mean() > 2.0 * cloth[:, 0].mean()       # denim between them
    assert np.abs(img[:, 0] - img[:, -1]).mean() < 0.06                   # it closes round her without a step
