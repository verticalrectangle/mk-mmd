"""The body part (mkmmd.model.parts.body): landmarks, skeleton, weights, welded hands, colliders, symmetry. bpy-free."""
from collections import Counter

import numpy as np
import pytest

from mkmmd.core import bonemap
from mkmmd.model import build
from mkmmd.model import part as P
from mkmmd.model import skeleton as SK
from mkmmd.model.parts import body_geom as G
from mkmmd.model.parts import body_shape as BS
from mkmmd.model.parts import body_weights as BW

FINGERS = ("thumb", "index", "middle", "ring", "little")


def make(tmp_path, **body):
    spec = {"model": {"name": "t", "parts": ["body"], "seed": 1}}
    if body:
        spec["body"] = body
    return build.run(spec, only="body", tex_dir=tmp_path)[0]


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    return make(tmp_path_factory.mktemp("body_tex"))


@pytest.fixture(scope="module")
def mesh(part):
    return part.meshes[0]


def bone_map(part):
    return {b.name: b for b in part.bones}


def test_part_is_valid_and_standard(part):
    P.check(part)
    names = {b.name for b in part.bones}
    smap = bonemap.build_map({n: n for n in names})
    assert not [s for s in bonemap.REQUIRED if s not in smap]
    for must in ("腕捩", "手捩", "つま先ＩＫ"):
        assert any(must in n for n in names)
    assert [m.name for m in part.materials] == ["skin", "nail"]
    assert part.frames and "腕" in part.frames
    assert part.meshes[0].name == "body" and part.info["mesh"] == "body"


def test_landmarks_complete_and_mirrored(part):
    land = part.info["landmarks"]
    need = SK.required_landmarks()
    assert not [n for n in need if n not in land]
    for k, v in land.items():
        if k.endswith(".L"):
            r = land[k[:-2] + ".R"]
            assert np.allclose(r, [-v[0], v[1], v[2]], atol=1e-4), k
    for extra in ("tail_root", "tail_root.L", "tail_root.R", "bust_front", "toe_end.L"):
        assert extra in land
    n = np.asarray(part.info["tail_normal"])
    assert abs(np.linalg.norm(n) - 1) < 1e-6 and n[1] > 0.5          # the tail root faces backwards (+Y)


def test_weights_valid(part, mesh):
    names = list(mesh.weights)
    M = np.stack([mesh.weights[n] for n in names], 1)
    assert (M >= 0).all() and np.allclose(M.sum(1), 1.0, atol=1e-6)
    assert ((M > 0).sum(1) <= 4).all()
    bones = bone_map(part)
    assert all(n in bones and bones[n].deform for n in names)
    for tw in ("左腕捩", "左手捩", "右腕捩", "右手捩"):
        assert tw in mesh.weights and mesh.weights[tw].max() > 0.5, tw


def dominant(part, mesh, mask):
    names = list(mesh.weights)
    M = np.stack([mesh.weights[n] for n in names], 1)
    return Counter(names[i] for i in np.argmax(M[mask], axis=1))


def test_limb_chains_in_order(part, mesh):
    v = mesh.verts
    rg = part.info["regions"]
    a, b = rg["leg_L"]
    z = v[a:b, 2]
    lv = np.arange(a, b)
    land = part.info["landmarks"]
    thigh = dominant(part, mesh, lv[(z > land["knee.L"][2] + 0.05) & (z < land["leg.L"][2] - 0.08)])
    shin = dominant(part, mesh, lv[(z > land["ankle.L"][2] + 0.05) & (z < land["knee.L"][2] - 0.05)])
    foot = dominant(part, mesh, lv[z < land["ankle.L"][2] - 0.04])
    assert thigh.most_common(1)[0][0] == "左足"
    assert shin.most_common(1)[0][0] == "左ひざ"
    assert foot.most_common(1)[0][0] == "左足首"
    a, b = rg["arm_L"]
    av = np.arange(a, b)
    x = v[a:b, 0]
    up = dominant(part, mesh, av[(x > land["arm.L"][0] + 0.03) & (x < land["elbow.L"][0] - 0.03)])
    fore = dominant(part, mesh, av[(x > land["elbow.L"][0] + 0.03) & (x < land["wrist.L"][0] - 0.03)])
    hand = dominant(part, mesh, av[(x > land["wrist.L"][0] + 0.01) & (x < land["middle1.L"][0] - 0.02)])
    assert {k for k, _ in up.most_common(2)} <= {"左腕", "左腕捩", "左ひじ"}
    assert {k for k, _ in fore.most_common(2)} <= {"左ひじ", "左手捩", "左手首"}
    assert hand.most_common(1)[0][0] == "左手首"


def test_finger_chains_ordered(part, mesh):
    bones = bone_map(part)
    land = part.info["landmarks"]
    for side, jp in (("L", "左"), ("R", "右")):
        wrist = np.asarray(land[f"wrist.{side}"])
        for f in FINGERS:
            names = SK.finger_joint_names(f)
            d = [np.linalg.norm(np.asarray(land[f"{n}.{side}"]) - wrist) for n in names]
            assert all(x < y for x, y in zip(d, d[1:])), (side, f, d)
        for f, jpn in (("index", "人指"), ("middle", "中指"), ("ring", "薬指"), ("little", "小指")):
            chain = [f"{jp}{jpn}{i}" for i in "１２３"]
            assert bones[chain[0]].parent == f"{jp}手首"
            assert bones[chain[1]].parent == chain[0] and bones[chain[2]].parent == chain[1]
            for n in chain:
                assert n in mesh.weights and mesh.weights[n].max() > 0.9, n
        for i in "０１２":
            n = f"{jp}親指{i}"
            assert n in mesh.weights and mesh.weights[n].max() > 0.5, n


def test_mesh_symmetric_within_a_tenth_of_a_millimetre(part, mesh):
    v = mesh.verts
    for name in part.info["regions"]:
        if not name.endswith("_L"):
            continue
        a, b = part.info["regions"][name]
        c, d = part.info["regions"][name[:-2] + "_R"]
        assert (b - a) == (d - c), name
        assert np.allclose(v[a:b] * [-1, 1, 1], v[c:d], atol=1e-4), name
    # torso: the vertex set is its own mirror image
    a, b = part.info["regions"]["torso"]
    t = v[a:b]
    key = lambda p: np.lexsort((np.round(p[:, 2], 5), np.round(p[:, 1], 5), np.round(np.abs(p[:, 0]), 5)))
    left, right = t[t[:, 0] > 1e-6], t[t[:, 0] < -1e-6]
    assert len(left) == len(right)
    assert np.allclose(left[key(left)] * [-1, 1, 1], right[key(right)], atol=1e-4)
    # left and right bones carry the same total weight
    ws = sum(mesh.weights[n].sum() for n in mesh.weights if n.startswith("左"))
    wr = sum(mesh.weights[n].sum() for n in mesh.weights if n.startswith("右"))
    assert abs(ws - wr) < 1e-3 * max(ws, 1.0)


def edges(mesh, mat=None):
    """Edge use counts of the faces of one material (default: all)."""
    c = Counter()
    for fi, f in enumerate(mesh.faces):
        if mat is not None and mesh.face_mat[fi] != mat:
            continue
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            c[(min(a, b), max(a, b))] += 1
    return c


def test_surface_closed_except_the_neck_seam(part, mesh):
    c = edges(mesh, mat=0)                                       # the skin; nail plates are open patches on top of it
    assert max(c.values()) == 2
    boundary = [e for e, k in c.items() if k == 1]
    nt = part.info["neck_top"]
    assert len(boundary) == nt["n"]
    bv = np.unique(np.array(boundary).ravel())
    assert np.allclose(mesh.verts[bv][:, 2], nt["z"], atol=1e-6)
    ring = np.asarray(nt["ring"])
    got = mesh.verts[bv]
    for p in ring:
        assert np.linalg.norm(got - p, axis=1).min() < 1e-6
    # the ring is a plain ellipse on the seam: rx, ry about the centre
    cx, cy = nt["center"]
    e = ((ring[:, 0] - cx) / nt["rx"]) ** 2 + ((ring[:, 1] - cy) / nt["ry"]) ** 2
    assert np.allclose(e, 1.0, atol=1e-6)
    # first ring point is the front-centre point (theta 0 = front, towards +X)
    assert abs(ring[0][0] - cx) < 1e-9 and ring[0][1] < cy
    # seam vertices are 100 % 首
    kubi = mesh.weights["首"]
    assert (kubi[bv] > 0.999).all()


def test_faces_sane(mesh):
    v = mesh.verts
    n = 0
    for f in mesh.faces:
        n += len(f)
        for k in range(1, len(f) - 1):
            area = 0.5 * np.linalg.norm(np.cross(v[f[k]] - v[f[0]], v[f[k + 1]] - v[f[0]]))
            assert area > 1e-10, f
    assert mesh.uv.shape == (n, 2) and np.isfinite(mesh.uv).all()
    assert mesh.uv.min() >= -1e-9 and mesh.uv.max() <= 1 + 1e-9
    assert len(mesh.face_mat) == len(mesh.faces) and set(np.unique(mesh.face_mat)) == {0, 1}
    # outward faces: the signed volume of the (nearly closed) body is positive
    vol = 0.0
    for f in mesh.faces:
        for k in range(1, len(f) - 1):
            vol += np.dot(v[f[0]], np.cross(v[f[k]], v[f[k + 1]])) / 6.0
    assert vol > 0.015


def test_hands_have_webs_and_no_overlap(part, mesh):
    """Four fingers continue the palm ring (shared vertices): the finger shells own no first-ring vertices and the
    slits are closed by three quads, so the left hand region is manifold."""
    rg = part.info["regions"]
    v = mesh.verts
    mid = np.asarray(part.info["landmarks"]["middle_tip.L"])
    idx = np.where(np.linalg.norm(v - mid, axis=1) < 0.012)[0]
    assert len(idx) > 10
    c = edges(mesh, mat=0)
    near = set(idx)
    assert all(c[e] == 2 for e in c if e[0] in near and e[1] in near)
    # fingertip of the middle finger is the farthest point of the hand from the wrist
    wr = np.asarray(part.info["landmarks"]["wrist.L"])
    a, b = rg["middle_L"]
    assert np.linalg.norm(v[a:b] - wr, axis=1).max() > 0.150


def test_colliders_hug_and_mirror(part):
    bodies = {b.name: b for b in part.bodies}
    need = ["col_head", "col_neck", "col_upper_body", "col_upper_body2", "col_lower_body"]
    need += [f"col_{r}_{s}" for r in ("shoulder", "arm", "forearm", "hand", "thigh", "shin", "foot") for s in "LR"]
    assert not [n for n in need if n not in bodies]
    bones = {b.name for b in part.bones}
    for b in bodies.values():
        assert b.mode == "static" and b.group == 0 and b.bone in bones and b.no_collide == (0,)
        assert max(b.size) < 0.4
    for r in ("shoulder", "arm", "forearm", "hand", "thigh", "shin", "foot"):
        assert bodies[f"col_{r}_L"].size == bodies[f"col_{r}_R"].size
        a, c = np.asarray(bodies[f"col_{r}_L"].location), np.asarray(bodies[f"col_{r}_R"].location)
        assert np.allclose(c, a * [-1, 1, 1], atol=1e-6)
    assert bodies["col_arm_L"].size[0] < 0.045 and bodies["col_thigh_L"].size[0] < 0.07


def test_skin_is_warm_and_published(part, tmp_path):
    from PIL import Image
    sk = part.info["skin"]
    for k in ("base", "shade", "blush", "blush_knee", "toon", "texture", "material", "toon_mult", "ambient"):
        assert k in sk, k
    m = np.asarray(sk["toon_mult"], float)
    assert m[0] > m[1] > 0.5 and m[0] > m[2] and m[0] - m[1] > 0.05           # pink shadow half, never grey or blue
    a = np.asarray(sk["ambient"], float)
    assert a[0] >= a[1] >= a[2]
    base = np.array([int(sk["base"][i:i + 2], 16) for i in (1, 3, 5)])
    shade = np.array([int(sk["shade"][i:i + 2], 16) for i in (1, 3, 5)])
    assert base[0] >= base[1] >= base[2] and shade[0] > shade[1] and shade[0] > shade[2] and (shade < base).all()   # light pink shadow
    mat = [x for x in part.materials if x.name == "skin"][0]
    assert mat.texture == sk["texture"] and mat.toon == sk["toon"]


def test_neck_shadow_band_in_the_texture(tmp_path):
    from PIL import Image
    p = make(tmp_path)
    ns = p.info["neck_shadow"]
    assert abs(ns["z_top"] - p.info["neck_top"]["z"]) < 1e-9 and ns["strength"] > 0 and ns["height"] > 0.02
    px = np.asarray(Image.open(tmp_path / p.info["skin"]["texture"]).convert("RGB")).astype(float)
    n = px.shape[0]

    def at(u, v):                                     # torso tile: u 0..0.5 of the atlas width, v 0.5..1 of its height
        return px[int((1 - (0.5 + 0.5 * v)) * n), int(u * 0.5 * n)]
    top_front, top_back, mid = at(0.01, 0.995), at(0.5, 0.995), at(0.01, 0.4)
    assert top_front[1] < mid[1] - 40 and top_front[2] < mid[2] - 50           # darker, and warm (peach-brown), at the seam
    assert top_front[0] > top_front[1] > top_front[2]
    assert top_back[1] > top_front[1] + 25                                      # weaker at the nape
    off = make(tmp_path / "off", neck_shadow={"strength": 0.0})
    px0 = np.asarray(Image.open(tmp_path / "off" / off.info["skin"]["texture"]).convert("RGB")).astype(float)
    assert abs(px0[int((1 - 0.9975) * n), 5] - mid).max() < 3


def test_nails_option(tmp_path):
    p = make(tmp_path / "a", nails=False)
    assert [m.name for m in p.materials] == ["skin"] and set(np.unique(p.meshes[0].face_mat)) == {0}
    q = make(tmp_path / "b", nails="#c9262d")
    nail = [m for m in q.materials if m.name == "nail"][0]
    assert np.allclose(nail.diffuse[:3], [0xc9 / 255, 0x26 / 255, 0x2d / 255], atol=1e-6)


def test_chain_weights_hand_over_in_order():
    s = np.array([-0.1, 0.0, 0.1, 0.2, 0.3, 0.5])
    W = BW.chain_weights(s, [0.2, 0.4], [0.05, 0.05])
    assert np.allclose(W.sum(1), 1.0)
    assert W[0, 0] == 1.0 and W[-1, -1] == 1.0 and abs(W[3, 0] - 0.5) < 1e-9


def test_cap_weights_keeps_four_largest():
    rng = np.random.default_rng(0)
    M = rng.random((50, 7))
    M /= M.sum(1, keepdims=True)
    W = BW.cap_weights({f"b{i}": M[:, i] for i in range(7)}, 4)
    A = np.stack(list(W.values()), 1)
    assert ((A > 0).sum(1) <= 4).all() and np.allclose(A.sum(1), 1.0)
    top = np.sort(M, axis=1)[:, -1]
    assert (A.max(1) >= top - 1e-9).all()


def test_shape_resolve_uses_spec_and_defaults():
    base = BS.resolve()
    assert not [n for n in SK.required_landmarks() if n not in base.land]
    custom = BS.resolve({"landmarks": {"wrist.L": [0.40, 0.006, 0.95]}}, {})
    assert np.allclose(custom.land["wrist.L"], [0.40, 0.006, 0.95]) and np.allclose(custom.land["wrist.R"], [-0.40, 0.006, 0.95])
    assert base.notes and not custom.notes


def test_tube_is_closed_and_outward():
    path = G.Path([[0, 0, 0], [0, 0, 0.3], [0.05, 0, 0.6]], blend=0.05)
    tab = G.Table([0, 0.3, 0.6], rx=[0.05, 0.04, 0.03], ry=[0.05, 0.045, 0.03], ox=[0, 0, 0], oy=[0, 0, 0], n=[2, 2, 2])
    sh = G.tube("t", path, np.linspace(0, 0.62, 12), tab, 12, (0, -1, 0))
    assert sh.signed_volume() > 0
    c = Counter()
    for f in sh.faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            c[(min(a, b), max(a, b))] += 1
    assert set(c.values()) == {2}
