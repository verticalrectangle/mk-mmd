"""mkmmd.model.assemble: coordinate conversion, bone order, triangulation, normals, weights, whole-model assembly."""
import math

import numpy as np
import pytest

from mkmmd.model import assemble as AS
from mkmmd.model import part as P
from mkmmd.model import pmx_io as X


def rand_rot(rng):
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    th = rng.uniform(0.1, 3.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def test_euler_yxz_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        R = rand_rot(rng)
        ex, ey, ez = AS.euler_yxz(R)
        assert np.allclose(AS._rz(ez) @ AS._rx(ex) @ AS._ry(ey), R, atol=1e-9)
    R = AS._rz(0.4) @ AS._rx(math.pi / 2) @ AS._ry(0.3)                    # gimbal lock
    ex, ey, ez = AS.euler_yxz(R)
    assert np.allclose(AS._rz(ez) @ AS._rx(ex) @ AS._ry(ey), R, atol=1e-9)


def test_pmx_euler_inverts_what_mmd_tools_does():
    """mmd_tools: blender_euler = (-rx, -rz, -ry) read as YXZ; that must give back the model-space XYZ rotation."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        e = rng.uniform(-3, 3, size=3)
        rx, ry, rz = AS.pmx_euler(e)
        bx, by, bz = -rx, -rz, -ry
        R = AS._rz(bz) @ AS._rx(bx) @ AS._ry(by)          # Blender 'YXZ': Y first, then X, then Z
        assert np.allclose(R, AS.matrix_xyz(e), atol=1e-9)


def test_swap_converts_model_to_pmx_axes():
    p = AS._swap([1.0, 2.0, 3.0])
    assert p.tolist() == [1.0, 3.0, 2.0]                              # x, z, y: model Z (up) becomes PMX Y


def bone(name, parent="", head=(0, 0, 0), **kw):
    return P.Bone(name, head, parent=parent, **kw)


def test_order_bones_puts_parents_and_grant_parents_first_and_keeps_order():
    bs = [bone("c", "b"), bone("a"), bone("b", "a"), bone("d", "a", grant={"parent": "e"}), bone("e", "a")]
    assert [b.name for b in AS.order_bones(bs)] == ["a", "b", "c", "e", "d"]
    assert [b.name for b in AS.order_bones([bone("a"), bone("b", "a")])] == ["a", "b"]
    with pytest.raises(ValueError, match="cycle"):
        AS.order_bones([bone("a", "b"), bone("b", "a")])


def test_convert_bones_flags_tails_ik_and_axes():
    bones = [bone("全ての親", head=(0, 0, 0), tail=(0, 0, 0.08), movable=True),
             bone("a", "全ての親", head=(0.08, 0.16, 0.24), tail_bone="b", local_x=(1, 0, 0), local_z=(0, -1, 0),
                  fixed_axis=(0, 0.6, 0.8)),
             bone("b", "a", head=(0.16, 0.16, 0.24), visible=False, grant={"parent": "a", "ratio": 0.5}),
             bone("ik", "全ての親", head=(0, 0, 0), movable=True,
                  ik={"target": "b", "iterations": 12, "angle": 57.29578,
                      "chain": [{"bone": "a", "limit": [[-180, 0, 0], [-0.5, 0, 0]]}, {"bone": "全ての親", "limit": None}]})]
    idx = {b.name: i for i, b in enumerate(bones)}
    pb = AS.convert_bones(bones, idx, 0.08)
    assert np.allclose(pb[1].pos, (1.0, 3.0, 2.0))                    # (x, z, y) / 0.08
    assert pb[0].tail_offset is not None and np.allclose(pb[0].tail_offset, (0, 1.0, 0))
    assert pb[1].tail_bone == 2 and pb[1].tail_offset is None
    assert np.allclose(pb[1].local_x, (1, 0, 0)) and np.allclose(pb[1].local_z, (0, 0, -1))
    assert np.allclose(pb[1].fixed_axis, (0, 0.8, 0.6))
    assert not pb[2].visible and not pb[2].controllable and pb[2].grant_parent == 1 and pb[2].grant_ratio == 0.5
    assert pb[2].grant_rotate and not pb[2].grant_move
    assert pb[3].movable and pb[3].ik.target == 2 and pb[3].ik.loops == 12 and np.isclose(pb[3].ik.angle, 1.0, atol=1e-4)
    assert pb[3].ik.links[0][0] == 1 and pb[3].ik.links[0][1] is True
    assert np.allclose(pb[3].ik.links[0][2], (-math.pi, 0, 0)) and np.allclose(pb[3].ik.links[0][3], (-0.5 * math.pi / 180, 0, 0))
    assert pb[3].ik.links[1][1] is False


def test_triangulate_quads_ngons_and_concave():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    tc, tf = AS.triangulate(v, [[0, 1, 2, 3]])
    assert tc.shape == (2, 3) and tf.tolist() == [0, 0] and sorted(tc.reshape(-1).tolist()).count(0) >= 1
    # a concave L-shaped hexagon (z = 0): total area 3, every triangle winds CCW (positive area)
    L = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], float)
    tc, tf = AS.triangulate(L, [[0, 1, 2, 3, 4, 5]])
    assert len(tc) == 4 and set(tf.tolist()) == {0}
    area = 0.0
    for t in tc:
        a, b, c = L[t]
        s = np.cross(b - a, c - a)[2] / 2
        assert s > 0
        area += s
    assert math.isclose(area, 3.0)
    # mixed faces keep their face ids
    V = np.vstack([v, [[0, 0, 1]]])
    tc, tf = AS.triangulate(V, [[0, 1, 4], [0, 1, 2, 3]])
    assert tf.tolist().count(0) == 1 and tf.tolist().count(1) == 2


def test_corner_normals_smooth_sharp_and_flat():
    # a cube: smooth shading averages at the corners, sharp edges split the normals, flat gives face normals
    v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) - 0.5
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    n, g = AS.corner_normals(v, f, smooth=True)
    assert np.allclose(np.linalg.norm(n, axis=1), 1)
    assert np.all(np.einsum("ij,ij->i", n, v[np.array(f).reshape(-1)]) > 0.5)        # all outward
    assert abs(n[0] @ np.array([-1, -1, -1]) / np.sqrt(3)) > 0.99                    # corner normal along the diagonal
    ed = [(a, b) for fc in f for a, b in zip(fc, fc[1:] + fc[:1])]
    n2, g2 = AS.corner_normals(v, f, smooth=True, sharp=ed)
    assert len(set(g2.tolist())) > 1
    fl = np.zeros((6, 3))
    for i, fc in enumerate(f):
        fl[i] = np.cross(v[fc[1]] - v[fc[0]], v[fc[2]] - v[fc[1]])
    fl /= np.linalg.norm(fl, axis=1, keepdims=True)
    for c, face in enumerate(np.repeat(np.arange(6), 4)):
        assert np.allclose(n2[c], fl[face], atol=1e-9)                               # fully sharp = flat normals
    n3, g3 = AS.corner_normals(v, f, smooth=False)
    assert np.allclose(n3, n2) and len(set(g3.tolist())) == 6
    n4, _ = AS.corner_normals(v, f, custom=np.tile([0, 0, 1.0], (8, 1)))
    assert np.allclose(n4, [0, 0, 1])


def test_top4_caps_and_normalises():
    W = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]], float)
    idx, w = AS.top4(W)
    assert idx[0].tolist() == [4, 3, 2, 1] and np.isclose(w[0].sum(), 1) and np.isclose(w[0, 0], 0.5 / 1.4)
    assert w[1].sum() == 0 and w[2].tolist() == [1, 0, 0, 0] and np.allclose(w[3], [0.5, 0.5, 0, 0])
    idx2, w2 = AS.top4(np.ones((2, 2)))
    assert idx2.shape == (2, 4) and np.allclose(w2[:, :2], 0.5) and np.allclose(w2[:, 2:], 0)


def tiny_part(**kw):
    bones = [P.Bone("全ての親", (0, 0, 0)), P.Bone("センター", (0, 0, 0.8), parent="全ての親"),
             P.Bone("a", (0.0, 0.0, 1.0), parent="センター"), P.Bone("b", (0.0, 0.0, 1.2), parent="a")]
    v = np.array([[0, 0, 0.9], [0.1, 0, 0.9], [0.1, 0, 1.1], [0, 0, 1.1], [0.1, 0, 1.3], [0, 0, 1.3]], float)
    faces = [[0, 1, 2, 3], [3, 2, 4, 5]]
    uv = np.array([[0, 0], [1, 0], [1, 0.5], [0, 0.5], [0, 0.5], [1, 0.5], [1, 1], [0, 1]], float)
    w = {"a": np.array([1, 1, 0.5, 0.5, 0, 0.0]), "b": np.array([0, 0, 0.5, 0.5, 1, 1.0])}
    mesh = P.Mesh("m", v, faces, uv=uv, face_mat=np.array([0, 1]), mats=["m1", "m2"], weights=w,
                  morphs={"あ": np.array([[0, 0, 0]] * 4 + [[0, -0.01, 0.02]] * 2, float)})
    base = dict(bones=bones, meshes=[mesh], materials=[P.Material("m1", texture="t.png", sphere="s.png", sphere_mode="add"),
                                                   P.Material("m2", toon="toon.png")],
                morphs=[P.Morph("あ", "mouth", "a")], frames={"体(上)": ["a", "b", "nope"]})
    base.update(kw)
    return P.Part("body", **base)


def test_assemble_a_tiny_model():
    part = tiny_part(bodies=[P.RigidBody("rb", "a", shape="capsule", size=(0.04, 0.1, 0), location=(0.0, 0.0, 1.0),
                                         rotation=(math.pi / 2, 0, 0)),
                             P.RigidBody("rd", "b", shape="box", size=(0.1, 0.2, 0.3), mode="dynamic_bone", group=3,
                                         no_collide=(0, 1))],
                     joints=[P.Joint("j", "rb", "rd", location=(0, 0, 1.1), rot_lo=(-0.5, -0.1, 0), rot_hi=(0.3, 0.2, 0))])
    a = AS.assemble([part], name="tiny", scale=0.08)
    m = a.pmx
    m.validate()
    assert a.stats["bones"] == 4 and a.stats["triangles"] == 4 and a.stats["materials"] == 2
    assert len(m.vertices) == 6                                                      # UVs agree along the shared edge
    assert [mat.index_count for mat in m.materials] == [6, 6]                       # one quad each -> 2 triangles
    assert m.textures == ["tex\\t.png", "tex\\s.png", "tex\\toon.png"]        # Windows separators like mmd_tools
    assert m.materials[0].texture == 0 and m.materials[0].sphere_texture == 1 and m.materials[0].sphere_mode == 2
    assert m.materials[1].toon == 2 and not m.materials[1].toon_shared
    assert np.allclose(m.vertices[0].pos, (0.0, 11.25, 0.0))                          # model (0, 0, 0.9) -> (x, z, y)/0.08
    # morph and its offsets (PMX axes, MMD units)
    assert [mo.name for mo in m.morphs] == ["あ"] and m.morphs[0].panel == 3 and len(m.morphs[0].offsets) >= 2
    idx, off = m.morphs[0].offsets[0]
    assert np.allclose(off, np.array([0, 0.02, -0.01]) / 0.08)                      # model (0,-0.01,0.02) -> (x, z, y)
    # frames: Root + 表情 + the part's frame (unknown bone dropped with a warning)
    assert [f.name for f in m.frames][:3] == ["Root", "表情", "体(上)"] and any("nope" in w for w in a.warnings)
    # bodies: capsule axis and sizes in MMD units, mode and masks
    rb, rd = m.bodies
    assert rb.shape == 2 and np.allclose(rb.size, (0.5, 1.25, 0)) and rb.mode == 0 and rb.mask == 0xFFFF
    assert rd.shape == 1 and np.allclose(rd.size, (1.25, 3.75, 2.5)) and rd.mode == 2 and rd.group == 3
    assert rd.mask == 0xFFFF & ~0b11
    assert m.joints[0].body_a == 0 and m.joints[0].body_b == 1
    j = m.joints[0]
    assert np.allclose(j.rot_lo, -np.array([0.3, 0.0, 0.2])) and np.allclose(j.rot_hi, -np.array([-0.5, 0.0, -0.1]))
    # expected arrays line up with the PMX vertices
    e = a.expected
    assert len(e["pos"]) == len(m.vertices) and e["bone_idx"].shape == (len(m.vertices), 4)
    assert np.allclose(e["bone_w"].sum(axis=1), 1)


def test_unweighted_vertices_attach_to_the_nearest_bone_with_a_warning():
    part = tiny_part()
    part.meshes[0].weights = {"a": np.array([1, 1, 0, 0, 0, 0.0])}
    a = AS.assemble([part], name="t")
    assert any("no weights" in w for w in a.warnings)
    assert np.allclose(a.expected["bone_w"].sum(axis=1), 1)
    idx = a.expected["bone_idx"][:, 0]
    names = [b.name for b in part.bones]
    assert names[idx[4]] in ("a", "b")                                          # nearest deforming bone, not the root


def test_faces_keep_index_order_and_uv_v_is_flipped():
    part = tiny_part()
    a = AS.assemble([part], name="t")
    m = a.pmx
    u, v = m.vertices[0].uv
    assert (u, v) == (0.0, 1.0)                                                  # input (0, 0) -> v flipped
    # custom normals are carried per source vertex
    part2 = tiny_part()
    part2.meshes[0].normals = np.tile([0, -1.0, 0], (6, 1))
    a2 = AS.assemble([part2], name="t")
    assert all(np.allclose(v.normal, (0, 0, -1)) for v in a2.pmx.vertices)        # model -Y (front) -> PMX -Z


def test_split_vertices_follow_uv_seams_and_sharp_edges():
    part = tiny_part()
    assert len(AS.assemble([part], name="t").pmx.vertices) == 6                    # UVs agree along the shared edge 2-3
    part.meshes[0].uv[4] = (0.1, 0.5)                                              # a seam: face 1 gets other UVs there
    part.meshes[0].uv[5] = (0.9, 0.5)
    assert len(AS.assemble([part], name="t").pmx.vertices) == 8
    part = tiny_part()
    part.meshes[0].sharp = [(2, 3)]                                                # coplanar faces: normals agree, no split
    assert len(AS.assemble([part], name="t").pmx.vertices) == 6
    part.meshes[0].verts[4:, 1] -= 0.2                                             # now the second face folds away
    assert len(AS.assemble([part], name="t").pmx.vertices) == 8
    part.meshes[0].sharp = []
    assert len(AS.assemble([part], name="t").pmx.vertices) == 6                    # smooth: one averaged normal


def test_assemble_without_material_raises_and_empty_morph_warns():
    part = tiny_part(materials=[])
    with pytest.raises(KeyError):
        AS.assemble([part], name="t")
    part = tiny_part()
    part.morphs.append(P.Morph("ghost", "eye"))
    a = AS.assemble([part], name="t")
    assert any("ghost" in w for w in a.warnings)


def test_pmx_roundtrip_of_an_assembled_model(tmp_path):
    part = tiny_part(bodies=[P.RigidBody("rb", "a")])
    a = AS.assemble([part], name="tiny")
    path = tmp_path / "tiny.pmx"
    X.write(a.pmx, path)
    back = X.read(path)
    assert len(back.vertices) == len(a.pmx.vertices) and back.faces == a.pmx.faces
    assert [b.name for b in back.bones] == [b.name for b in a.pmx.bones]
    assert back.morphs[0].name == "あ" and len(back.morphs[0].offsets) == len(a.pmx.morphs[0].offsets)
    assert back.textures == a.pmx.textures
