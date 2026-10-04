"""Twin braids of the hair part (bpy-free): rig and bone families, weights, closed shells, clearance from the body, the head
skin and the static colliders, determinism, published info. Builds on the stand-in head and body of hair_fixtures."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hair_fixtures as HF  # noqa: E402

from mkmmd.core import families  # noqa: E402
from mkmmd.model import part as P  # noqa: E402
from mkmmd.model.parts import hair_braids as HB  # noqa: E402
from mkmmd.model.parts import hair_tex  # noqa: E402
from mkmmd.model.parts.hair_braids_geo import TriMesh, collider_clearance, colliders, signed_volume  # noqa: E402
from mkmmd.model.parts.hair_fit import HeadFit  # noqa: E402
from mkmmd.model.parts.hair_rig import Rig  # noqa: E402


def build(tmp_path, cfg=None, seed=1):
    ctx = HF.make_ctx(tmp_path, seed=seed)
    fit = HeadFit(ctx.parts["head"].info)
    rig = Rig()
    piece = HB.build_braids(ctx, fit, cfg or {}, rig, hair_tex.palette())
    part = P.Part("hair", meshes=piece.meshes, materials=piece.materials, bones=rig.bones, bodies=rig.bodies,
                  joints=rig.joints, info=piece.info, frames=piece.frames)
    return ctx, fit, rig, piece, part


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build(tmp_path_factory.mktemp("braids"))


def chain_names(rig, prefix):
    return [b.name for b in rig.bones if b.name.startswith(prefix)]


def test_bones_classify(built):
    _, _, rig, _, _ = built
    for side in "左右":
        names = chain_names(rig, f"三つ編{side}")
        assert len(names) == HB.DEFAULTS["bones"]
        assert names == [f"三つ編{side}{k}" for k in range(1, len(names) + 1)]
        assert all(families.classify(n) == "braid" for n in names)
        for bow in "上下":
            rib = chain_names(rig, f"リボン{side}{bow}")
            assert rib == [f"リボン{side}{bow}1", f"リボン{side}{bow}2"]
            assert all(families.classify(n) == "ribbon" for n in rib)


def test_part_check_and_weight_bones(built):
    ctx, _, rig, piece, part = built
    P.check(part)
    known = {b.name for b in rig.bones} | {b.name for b in ctx.parts["body"].bones}
    for m in piece.meshes:
        assert set(m.weights) <= known
        assert all(mat in {x.name for x in piece.materials} for mat in m.mats)
        assert m.uv is not None and len(m.uv) == sum(len(f) for f in m.faces)
    assert {m.name for m in piece.materials} == {"三つ編", "髪リボン"}
    for m in piece.materials:
        assert m.edge and m.texture and m.toon and (ctx.tex_dir / m.texture).exists()
        assert min(m.edge_color[:3]) >= 0.0 and max(m.edge_color[:3]) < 0.45


def test_weights_sum_to_one(built):
    _, _, _, piece, _ = built
    for m in piece.meshes:
        tot = np.zeros(len(m.verts))
        for w in m.weights.values():
            tot += w
        assert np.abs(tot - 1.0).max() < 1e-6, m.name
        assert min(w.min() for w in m.weights.values()) >= 0.0


def test_chain_structure(built):
    ctx, _, rig, _, _ = built
    bones = {b.name: b for b in rig.bones}
    bodies = {b.name for b in rig.bodies}
    head = ctx.find_body("頭")
    for side in "左右":
        names = chain_names(rig, f"三つ編{side}")
        assert bones[names[0]].parent == "頭"
        for k, n in enumerate(names):
            b = bones[n]
            if k:
                assert b.parent == names[k - 1]
                assert np.allclose(b.head, bones[names[k - 1]].tail, atol=1e-9)         # connected
            assert b.tail_bone == (names[k + 1] if k + 1 < len(names) else "")
            assert n in bodies and f"J_{n}" in {j.name for j in rig.joints}
        j0 = next(j for j in rig.joints if j.name == f"J_{names[0]}")
        assert j0.a == head.name
        for bow, bone in (("上", names[0]), ("下", names[HB.DEFAULTS["bones"] - HB.DEFAULTS["tuft_bones"] - 1])):
            rib = chain_names(rig, f"リボン{side}{bow}")
            assert bones[rib[0]].parent == bone and bones[rib[1]].parent == rib[0]       # a branch off the braid bone
            assert np.allclose(bones[rib[0]].head, bones[rib[1]].head - (np.array(bones[rib[1]].head) -
                                                                         np.array(bones[rib[0]].head)), atol=1e-9)
            assert np.allclose(bones[rib[0]].tail, bones[rib[1]].head, atol=1e-9)
            j = next(j for j in rig.joints if j.name == f"J_{rib[0]}")
            assert j.a == bone                                                           # the braid bone's own body
    assert len({b.name for b in rig.bones}) == len(rig.bones)


def test_static_tails(tmp_path):
    ctx, _, rig, piece, part = build(tmp_path, {"bow_tails": "static"})
    assert not [b for b in rig.bones if b.name.startswith("リボン")]
    assert not [b for b in rig.bodies if b.name.startswith("リボン")]
    P.check(part)
    for m in piece.meshes:
        tot = sum(w for w in m.weights.values())
        assert np.abs(tot - 1.0).max() < 1e-6


def test_shells_closed_and_outward(built):
    _, _, _, piece, _ = built
    for m in piece.meshes:
        edges = {}
        for f in m.faces:
            for a, b in zip(f, list(f[1:]) + [f[0]]):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        for (a, b), c in edges.items():
            assert c == 1 and edges.get((b, a)) == 1, f"{m.name}: open or non-manifold edge {a}-{b}"
        # connected components, each with positive volume (outward faces)
        parent = list(range(len(m.verts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for f in m.faces:
            for k in f[1:]:
                parent[find(k)] = find(f[0])
        comp = {}
        for f in m.faces:
            comp.setdefault(find(f[0]), []).append(f)
        assert len(comp) > 20
        for faces in comp.values():
            assert signed_volume(m.verts, faces) > 0.0


def test_vertex_budget(built):
    _, _, _, piece, _ = built
    verts = sum(len(m.verts) for m in piece.meshes)
    assert 1500 < verts <= 3000
    assert all(len(m.verts) <= 1500 for m in piece.meshes)


def capsule_points(rb, n=9):
    from mkmmd.model.parts.hair_braids_geo import euler_matrix
    c = np.asarray(rb.location, float)
    ax = euler_matrix(rb.rotation) @ np.array([0.0, 0.0, 1.0])
    h = 0.5 * rb.size[1]
    return c[None] + np.linspace(-h, h, n)[:, None] * ax[None]


def test_clearance(built):
    ctx, _, rig, piece, _ = built
    body = ctx.parts["body"]
    body_tm = TriMesh(body.meshes[0].verts, body.meshes[0].faces)
    sk = ctx.parts["head"].info["skin"]
    skin_tm = TriMesh(sk["verts"], sk["faces"])
    cols = colliders(body)
    radius = {b.name: b.size[0] for b in rig.bodies}
    for side, key in (("左", "L"), ("右", "R")):
        # the centreline (every dense sample) against the body mesh and the head skin
        c = piece.info[key]["centerline"]
        r = HB.DEFAULTS["radius"]
        assert body_tm.signed(c)[0].min() >= r + 0.008, (key, body_tm.signed(c)[0].min())
        assert skin_tm.signed(c)[0].min() >= r + 0.008
        pts = piece.info["chain_pts"][key]
        assert body_tm.signed(pts)[0].min() >= r + 0.008
    for rb in rig.bodies:                                           # every chain capsule: >= 5 mm from colliders and the skin
        pts = capsule_points(rb)
        r = rb.size[0]
        assert collider_clearance(pts, cols, r).min() >= 0.005, rb.name
        assert skin_tm.signed(pts)[0].min() - r >= 0.005, rb.name
        assert body_tm.signed(pts)[0].min() - r >= 0.005, rb.name
    # no vertex of any braid piece inside the body or the head skin
    for m in piece.meshes:
        assert body_tm.signed(m.verts)[0].min() > 0.0015, m.name
        assert skin_tm.signed(m.verts)[0].min() > 0.0015, m.name


def test_hangs_in_front_of_the_shoulder(built):
    ctx, _, _, piece, _ = built
    lm = ctx.parts["body"].info["landmarks"]
    for key, sg in (("L", 1.0), ("R", -1.0)):
        c = piece.info[key]["centerline"]
        assert np.all(sg * c[:, 0] > 0.07) and np.all(sg * c[:, 0] < 0.13)           # beside the neck, over the shoulder
        k = np.argmin(np.abs(c[:, 2] - lm["arm." + key][2]))                          # at shoulder height ...
        assert c[k, 1] < lm["arm." + key][1] - 0.04                                   # ... in front of the shoulder joint
        assert c[-1, 2] < c[0, 2] - 0.10                                              # hangs down
        arm_x = abs(lm["arm." + key][0])
        assert sg * c[:, 0].min() > 0.07 and arm_x < 0.13


def test_info_and_gather(built):
    _, fit, _, piece, _ = built
    info = piece.info
    for key in "LR":
        s = info[key]
        assert len(s["bones"]) == HB.DEFAULTS["bones"] and s["chain_pts"].shape == (HB.DEFAULTS["bones"] + 1, 3)
        assert set(s["bows"]) == {"top", "bottom"} and all(len(b["center"]) == 3 for b in s["bows"].values())
        assert 0.14 < s["length"] < 0.20
        lo, hi = s["bounds"]
        assert (hi - lo).min() > 0.0 and len(s["gather"]) == HB.DEFAULTS["gather"]
        for g in s["gather"]:
            assert 95.0 <= abs(np.degrees(g["theta"])) <= 127.0 and 70.0 <= np.degrees(g["phi"]) <= 120.0
            assert np.sign(g["theta"]) == (1.0 if key == "L" else -1.0) and abs(np.linalg.norm(g["dir"]) - 1.0) < 1e-6
            assert fit.skull.height_dist(g["root"]) > 0.02                              # above the skin, on the hair hull
    assert info["bones"]["L"][0] == "三つ編左1" and info["bones"]["R"][-1] == f"三つ編右{HB.DEFAULTS['bones']}"


def test_mirror_and_variation(built):
    _, _, _, piece, _ = built
    L, R = piece.info["L"]["centerline"], piece.info["R"]["centerline"]
    assert abs(L[0, 0] + R[0, 0]) < 0.006 and abs(L[-1, 2] - R[-1, 2]) < 0.006
    assert not np.array_equal(piece.meshes[0].verts[:, 2], piece.meshes[1].verts[:, 2])   # each side has its own variation


def test_deterministic(tmp_path):
    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    for ma, mb in zip(a[3].meshes, b[3].meshes):
        assert np.array_equal(ma.verts, mb.verts) and ma.faces == mb.faces and np.array_equal(ma.uv, mb.uv)
        for k in ma.weights:
            assert np.array_equal(ma.weights[k], mb.weights[k])
    assert [x.name for x in a[2].bones] == [x.name for x in b[2].bones]
    c = build(tmp_path / "c", seed=7)
    assert not np.array_equal(a[3].meshes[0].verts, c[3].meshes[0].verts)


def test_config_overrides(tmp_path):
    ctx, _, rig, piece, part = build(tmp_path, {"bones": 11, "lobes": 5, "tuft_locks": 3, "gather": 4, "width": 0.03})
    P.check(part)
    assert len(chain_names(rig, "三つ編左")) == 11
    assert len(piece.info["L"]["gather"]) == 4
    off = build(tmp_path / "off", {"enabled": False})
    assert not off[3].meshes and not off[2].bones
