"""The hair part against the REAL body, head and outfit of a character project (the parts as they are right now, built
from the model.toml that MK_TEST_RIN_MODEL names). Skipped when the variable is unset or the project does not build here.

What the stand-in fixtures of the other hair tests cannot know: the real blouse stands 8-18 mm off the skin around the chest, the
tail roots sit on the real lower back, the real colliders are boxes and capsules that were fitted to the real body."""
import numpy as np
import pytest

from local_project import SPEC, WHY
from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts.hair_braids_geo import euler_matrix


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if SPEC is None:
        pytest.skip(WHY)
    sp = SP.load(SPEC)
    if "outfit" not in SP.model_cfg(sp)["parts"]:
        pytest.skip("the character project has no outfit part")
    ctx = BD.BuildCtx(sp, tmp_path_factory.mktemp("hair_real_all") / "tex", seed=1)
    try:
        BD.run(sp, only=["hair", "outfit"], ctx=ctx)
    except BD.BuildError as e:
        pytest.skip(f"the real parts do not build here: {e}")
    return ctx


def _tris(m):
    return np.array([[f[0], f[k], f[k + 1]] for f in m.faces for k in range(1, len(f) - 1)])


def _surface(meshes):
    """Vertices and area-weighted vertex normals of several meshes, concatenated."""
    V, N = [], []
    for m in meshes:
        v = np.asarray(m.verts, float)
        t = _tris(m)
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        acc = np.zeros_like(v)
        for k in range(3):
            np.add.at(acc, t[:, k], n)
        V.append(v)
        N.append(acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12))
    return np.concatenate(V), np.concatenate(N)


def _nearest(P, V, N):
    """Distance from each point of P to the nearest vertex of the surface (V, N), and the signed height of the point over
    that vertex along its normal (< 0: under the surface). Vertices are ~1 cm apart, which is fine for >= 3 mm margins."""
    d = np.empty(len(P))
    h = np.empty(len(P))
    for i in range(0, len(P), 500):
        D = np.linalg.norm(P[i:i + 500, None, :] - V[None, :, :], axis=2)
        j = D.argmin(1)
        d[i:i + 500] = D[np.arange(len(j)), j]
        h[i:i + 500] = np.einsum("ij,ij->i", P[i:i + 500] - V[j], N[j])
    return d, h


def _mesh(part, name):
    return next(m for m in part.meshes if m.name == name)


def _capsule_points(rb, n=9):
    c = np.asarray(rb.location, float)
    ax = euler_matrix(rb.rotation) @ np.array([0.0, 0.0, 1.0])
    return c[None] + np.linspace(-0.5, 0.5, n)[:, None] * rb.size[1] * ax[None]


def test_braids_hang_clear_of_the_blouse(real):
    """No braid vertex under the dress (the project sets [hair.braids] clearance for it) and none under the skin."""
    hair, out, body = real.parts["hair"], real.parts["outfit"], real.parts["body"]
    Vd, Nd = _surface([m for m in out.meshes if m.name in ("outfit_dress", "outfit_frills")])
    Vb, Nb = _surface(body.meshes)
    for name in ("hair_braid_L", "hair_braid_R"):
        P = np.asarray(_mesh(hair, name).verts, float)
        for what, V, N, margin in (("dress", Vd, Nd, 0.003), ("skin", Vb, Nb, 0.004)):
            sel = (np.abs(V[:, 0]) < 0.35) & (V[:, 2] > 0.9) & (V[:, 2] < 1.45)
            d, h = _nearest(P, V[sel], N[sel])
            assert not ((h < 0) & (d < 0.03)).any(), f"{name} has vertices under the {what}"
            assert d.min() >= margin, f"{name} is only {d.min() * 1e3:.1f} mm from the {what}"


def test_tails_start_inside_the_back_and_are_free_beyond(real):
    """The tail tube is buried in the lower back behind each root (hidden by the dress) and every vertex further than
    8 cm from the roots is above both the body skin and the dress; the chains hang from 下半身."""
    hair, out, body = real.parts["hair"], real.parts["outfit"], real.parts["body"]
    lm = body.info["landmarks"]
    roots = [np.asarray(lm[k], float) for k in ("tail_root.L", "tail_root.R")]
    P = np.asarray(_mesh(hair, "cat_tails").verts, float)
    dr = np.minimum(np.linalg.norm(P - roots[0], axis=1), np.linalg.norm(P - roots[1], axis=1))
    Vd, Nd = _surface([m for m in out.meshes if m.name in ("outfit_dress", "outfit_frills")])
    Vb, Nb = _surface(body.meshes)
    for what, V, N in (("dress", Vd, Nd), ("skin", Vb, Nb)):
        sel = (V[:, 2] > 0.7) & (V[:, 2] < 1.6)
        d, h = _nearest(P, V[sel], N[sel])
        assert not ((h < 0) & (d < 0.05) & (dr > 0.08)).any(), f"a tail vertex further than 8 cm from the root is under the {what}"
    nrm = np.asarray(body.info["tail_normal"], float)
    first = {b.name: b for b in hair.bones if b.name in ("尻尾1_1", "尻尾2_1")}
    assert set(first) == {"尻尾1_1", "尻尾2_1"}
    for (name, b), root in zip(sorted(first.items()), roots):
        assert b.parent == "下半身"
        head = np.asarray(b.head, float)
        assert 0.02 < np.linalg.norm(head - root) < 0.09              # the first free bone starts a few cm off the root ...
        assert (head - root) @ nrm > 0.02                              # ... behind it along the body's tail normal


def test_chains_stay_out_of_the_real_colliders(real):
    """Every dynamic chain capsule of the hair part (head hair, braids, tails, ears) keeps off the body part's static
    colliders (sphere/box/capsule) at rest; the closest approach is published for the report."""
    hair, body = real.parts["hair"], real.parts["body"]
    from mkmmd.model.parts.hair_braids_geo import collider_clearance, colliders
    cols = colliders(body)
    worst = []
    for rb in hair.bodies:
        if rb.mode == "static" or rb.shape != "capsule":
            continue
        pts = _capsule_points(rb)
        worst.append((float(collider_clearance(pts, cols, rb.size[0]).min()), rb.name))
    worst.sort()
    assert worst, "no dynamic capsules"
    assert worst[0][0] >= 0.0, f"{worst[0][1]} starts {-worst[0][0] * 1e3:.1f} mm inside a collider (closest: {worst[:3]})"


def test_head_hair_stays_off_the_neck_and_shoulders(real):
    hair, body = real.parts["hair"], real.parts["body"]
    V, N = _surface(body.meshes)
    sel = (np.abs(V[:, 0]) < 0.3) & (V[:, 2] > 1.0)
    P = np.asarray(_mesh(hair, "hair_head").verts, float)
    P = P[P[:, 2] < 1.3]
    d, h = _nearest(P, V[sel], N[sel])
    assert not ((h < 0) & (d < 0.03)).any()
    assert d.min() >= 0.004, f"head hair is {d.min() * 1e3:.1f} mm from the body skin"
