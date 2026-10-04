"""The generic PMX importer (mkmmd.model.pmx_take) and the texture recolour (mkmmd.model.pmx_recolor), bpy-free, on a tiny synthetic PMX
(never on a real model: the repo's tests must not depend on one).

The synthetic model: an octahedron head weighted to the head bone (a ring of three neck vertices shared with the body material,
half on the neck bone), eyeballs on the eye bones, a few body triangles, a hidden skin material, a hair chain with its tip, a tie
chain and a material for each of them (dropped), IK / grant / control bones, vertex morphs (one touches the dropped hair only), a
material morph and a group morph, two rigid bodies and a joint."""
import math

import numpy as np
import pytest
from PIL import Image

from mkmmd.model import assemble, pmx_io as X, pmx_recolor as RC, pmx_take as T

UNIT = 0.08


def _p(x, y, z):
    """Model metres (x, y forward -, z up) -> PMX units."""
    return (x / UNIT, z / UNIT, y / UNIT)


BONES = [  # name, parent, model-space head (x, y, z) metres, extra flags
    ("全ての親", -1, (0.0, 0.0, 0.0), {}),
    ("センター", 0, (0.0, 0.0, 0.9), {}),
    ("上半身", 1, (0.0, 0.0, 1.0), {}),
    ("首", 2, (0.0, 0.0, 1.15), {}),
    ("頭", 3, (0.0, 0.0, 1.2), {}),
    ("両目", 4, (0.0, -0.1, 1.4), {}),
    ("左目", 4, (0.03, -0.05, 1.3), dict(grant_rotate=True, grant_parent=5, grant_ratio=1.0, layer=2)),
    ("右目", 4, (-0.03, -0.05, 1.3), dict(grant_rotate=True, grant_parent=5, grant_ratio=1.0, layer=2)),
    ("頭先", 4, (0.0, 0.0, 1.4), dict(visible=False)),
    ("髪1", 4, (0.0, 0.05, 1.35), {}),
    ("髪2", 9, (0.0, 0.08, 1.25), {}),
    ("髪先", 10, (0.0, 0.1, 1.15), dict(visible=False)),
    ("左足ＩＫ", 0, (0.1, 0.0, 0.1), {}),
    ("左足", 1, (0.1, 0.0, 0.85), {}),
    ("左ひざ", 13, (0.1, 0.0, 0.45), {}),
    ("左足首", 14, (0.1, 0.0, 0.1), {}),
    ("ネクタイ1", 2, (0.0, -0.06, 1.1), {}),
    ("ネクタイ先", 16, (0.0, -0.06, 1.0), dict(visible=False)),
    ("操作", 0, (0.3, 0.0, 0.0), {}),
    ("左足D", 14, (0.1, 0.0, 0.45), dict(grant_rotate=True, grant_parent=14, grant_ratio=1.0)),
    ("髪の先の先", 9, (0.0, 0.06, 1.3), dict(visible=False)),
]
B = {n: i for i, (n, *_r) in enumerate(BONES)}


def _octa(c, r):
    c = np.asarray(c, float)
    return [c + (r, 0, 0), c - (r, 0, 0), c + (0, r, 0), c - (0, r, 0), c + (0, 0, r), c - (0, 0, r)]


OCTA = [(4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0), (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3)]      # CCW from outside


class Synth:
    def __init__(self, tmp):
        self.dir = tmp
        (tmp / "tex").mkdir()
        tex = np.zeros((16, 16, 3), np.uint8)
        tex[...] = (240, 220, 200)
        tex[8:, 8:] = (160, 150, 215)
        Image.fromarray(tex).save(tmp / "tex" / "skin.png")
        Image.fromarray(tex).save(tmp / "tex" / "skin2.png")
        Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(tmp / "tex" / "toon.bmp")
        a = np.zeros((8, 8, 4), np.uint8)
        a[..., 0] = 200
        a[:4, :, 3] = 255
        Image.fromarray(a).save(tmp / "tex" / "alpha.png")
        V, W, UV = [], [], []

        def add(p, bones, weights, uv=(0.3, 0.3)):
            V.append(p)
            W.append((tuple(B[b] for b in bones), tuple(weights)))
            UV.append(uv)
            return len(V) - 1

        head = [add(p, ["頭"], [1.0], (0.1 * k, 0.2)) for k, p in enumerate(_octa((0, 0, 1.3), 0.08))]        # 0..5
        self.neck = [add(np.array([0.02 * k, 0.0, 1.17]), ["頭", "首"], [0.5, 0.5], (0.5, 0.1 * k)) for k in range(3)]   # 6..8
        eyes = {s: [add(np.array([sg * 0.03, -0.07, 1.3 + 0.01 * k]), [bone], [1.0]) for k in range(3)]
                for s, sg, bone in (("L", 1, "左目"), ("R", -1, "右目"))}
        body = [add(np.array([0.0, 0.0, 1.17 - 0.05 * k]), ["上半身", "首"], [0.7, 0.3]) for k in range(1, 4)]         # shares nothing
        legs = [add(np.array([0.1, 0.0, 0.3 - 0.1 * k]), ["左ひざ", "左足首"], [0.4, 0.6]) for k in range(3)]
        hair = [add(np.array([0.0, 0.05 + 0.01 * k, 1.35 - 0.05 * k]), ["髪1", "髪2"], [0.5, 0.5]) for k in range(3)]
        cloth = [add(np.array([0.0, -0.06, 1.1 - 0.02 * k]), ["ネクタイ1", "上半身"], [0.8, 0.2]) for k in range(3)]
        hidden = [add(np.array([0.05, 0.0, 1.0 - 0.02 * k]), ["上半身"], [1.0]) for k in range(3)]
        unused = [add(np.array([2.0, 0.0, 0.0]), ["全ての親"], [1.0])]
        self.ids = dict(head=head, eyes=eyes, body=body, legs=legs, hair=hair, cloth=cloth, hidden=hidden, unused=unused)
        pos = np.array(V)
        self.V = pos
        n = np.array([[0, 1, 0]] * len(V), float)
        verts = [X.PmxVertex(_p(*p), tuple(float(c) for c in (0, 0, 1)), (float(u[0]), float(u[1])),
                             bones=tuple(B[b] for b in [BONES[i][0] for i in w[0]]) if False else tuple(w[0]), weights=tuple(w[1]))
                 for p, w, u in zip(pos, W, UV)]
        verts[eyes["L"][0]].edge_scale = 0.5
        tris = {
            "face": [tuple(head[i] for i in t) for t in OCTA] + [(head[4], self.neck[0], self.neck[1]), (self.neck[0], self.neck[1], self.neck[2]),
                                                                (head[0], head[0], head[1])],        # the last one is degenerate
            "eyes": [tuple(eyes["L"]), tuple(eyes["R"])],
            "body": [(self.neck[0], body[0], body[1]), (self.neck[1], body[1], body[2]), tuple(legs)],
            "hidden": [tuple(hidden)],
            "hair": [tuple(hair)],
            "cloth": [tuple(cloth), (cloth[0], cloth[1], hidden[0])],
        }
        self.tris = tris
        order = ["face", "eyes", "body", "hidden", "hair", "cloth"]
        faces = [int(i) for k in order for t in tris[k] for i in t[::-1]]                                # PMX is clockwise
        textures = ["tex\\skin.png", "tex\\skin2.png", "tex\\toon.bmp", "tex\\alpha.png"]
        mats = [X.PmxMaterial("face", texture=0, toon_shared=False, toon=2, edge=True, edge_size=0.8, edge_color=(0.5, 0.3, 0.3, 1.0),
                              diffuse=(1, 1, 1, 1), ambient=(0.5, 0.5, 0.5), index_count=3 * len(tris["face"])),
                X.PmxMaterial("eyes", texture=3, sphere_texture=0, sphere_mode=2, toon_shared=True, toon=1, index_count=3 * len(tris["eyes"])),
                X.PmxMaterial("body", texture=1, double_sided=True, index_count=3 * len(tris["body"])),
                X.PmxMaterial("hidden", texture=1, diffuse=(1, 1, 1, 0.0), index_count=3 * len(tris["hidden"])),
                X.PmxMaterial("hair", index_count=3 * len(tris["hair"])),
                X.PmxMaterial("cloth", index_count=3 * len(tris["cloth"]))]
        self.names = [m.name for m in mats]
        bones = []
        for n_, par, p, kw in BONES:
            bones.append(X.PmxBone(n_, "", _p(*p), par, **kw))
        bones[B["左足ＩＫ"]].ik = X.PmxIK(B["左足首"], 40, 0.5, [(B["左ひざ"], True, (-math.pi, 0.0, 0.0), (-0.01, 0.0, 0.0)), (B["左足"], False, (0, 0, 0), (0, 0, 0))])
        bones[B["頭"]].tail_bone = B["頭先"]
        bones[B["髪先"]].tail_offset = None
        self.d = {k: np.array([0.001 * (i + 1), 0.002, -0.004]) for i, k in enumerate(("head", "neck", "hair", "body"))}
        vm = lambda name, panel, items: X.PmxMorph(name, "", panel, "vertex", [(i, _p(*d)) for i, d in items])
        morphs = [
            vm("m1", 2, [(head[0], self.d["head"]), (self.neck[0], self.d["neck"]), (hair[0], self.d["hair"]), (body[0], self.d["body"])]),
            vm("m2", 3, [(hair[1], self.d["hair"])]),
            vm("skipme", 1, [(head[1], self.d["head"])]),
            X.PmxMorph("mat", "", 4, "material", [
                X.MaterialMorphOffset.of(3, 1, diffuse=(0, 0, 0, 1)), X.MaterialMorphOffset.of(5, 0, diffuse=(0, 0, 0, 0)),
                X.MaterialMorphOffset.of(-1, 0, diffuse=(1, 1, 1, 0.5))]),
            X.PmxMorph("mat_cloth", "", 4, "material", [X.MaterialMorphOffset.of(5, 1, diffuse=(0, 0, 0, 1))]),
            X.PmxMorph("g1", "", 4, "group", [(0, 1.0), (1, 0.5)]),
            X.PmxMorph("g2", "", 4, "group", [(1, 1.0)]),
        ]
        bodies = [X.PmxBody("rb_head", "", B["頭"], 0, 0xFFFF, 0, (0.1, 0, 0), _p(0, 0, 1.3), (0.0, 0.0, 0.0), 1.0),
                  X.PmxBody("rb_box", "", B["上半身"], 1, 0xFFFD, 1, (0.1, 0.2, 0.3), _p(0, 0, 1.0), (0.3, 0.0, 0.0), 2.0),
                  X.PmxBody("rb_hair", "", B["髪1"], 8, 0xFFFF, 2, (0.02, 0.1, 0), _p(0, 0.05, 1.3), (0.0, 0.0, 0.0), 0.1, mode=2)]
        joints = [X.PmxJoint("j_hair", "", 0, 0, 2, _p(0, 0.05, 1.35)), X.PmxJoint("j_kept", "", 0, 0, 1, _p(0, 0, 1.15))]
        frames = [X.PmxFrame("Root", "", True, [("bone", 0)]), X.PmxFrame("表情", "", True, [("morph", 0)]),
                  X.PmxFrame("体", "", False, [("bone", B["上半身"]), ("bone", B["髪1"]), ("bone", B["頭"])]), X.PmxFrame("髪", "", False, [("bone", B["髪1"])])]
        self.model = X.PmxModel(name="synthetic", vertices=verts, faces=faces, textures=textures, materials=mats, bones=bones, morphs=morphs,
                                frames=frames, bodies=bodies, joints=joints)
        self.path = tmp / "m.pmx"
        X.write(self.model, str(self.path))
        self.keep = ["face", "eyes", "body", "hidden"]


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return Synth(tmp_path_factory.mktemp("pmx"))


SCALE, FACTOR = 0.95, 0.985 / 0.95


@pytest.fixture(scope="module")
def tk(syn):
    return T.take(syn.path, syn.keep, scale=SCALE, head={"bone": "頭", "factor": FACTOR}, skip_morphs=["skipme"], bodies=True)


@pytest.fixture(scope="module")
def plain(syn):
    return T.take(syn.path, syn.keep, scale=1.0)


# ------------------------------------------------------------------------------------------------------ vertices, faces
def test_only_the_vertices_of_the_kept_materials_come_along(syn, plain):
    used = sorted(set(i for k in syn.keep for t in syn.tris[k] for i in t))
    assert list(plain.src_ids) == used
    assert len(plain.verts) == len(used) and plain.normals.shape == plain.verts.shape and plain.uv.shape == (len(used), 2)
    assert set(plain.faces) == set(syn.keep)
    assert not set(syn.ids["hair"] + syn.ids["cloth"] + syn.ids["unused"]) & set(plain.src_ids.tolist())


def test_positions_are_model_axes_in_metres(syn, plain):
    assert np.allclose(plain.verts, syn.V[plain.src_ids], atol=2e-6)


def test_faces_are_ccw_outward_and_degenerate_ones_are_dropped(syn, plain):
    f = plain.faces["face"]
    assert len(f) == len(OCTA) + 2                                                    # the repeated-vertex triangle is gone
    c = plain.verts[plain.faces["face"][:8].ravel()].mean(0)
    for t in f[:8]:
        n = np.cross(plain.verts[t[1]] - plain.verts[t[0]], plain.verts[t[2]] - plain.verts[t[0]])
        assert n @ (plain.verts[t].mean(0) - c) > 0
    assert plain.faces["face"].max() < len(plain.verts)


def test_uv_is_v_up_and_edge_scale_and_unit_normals_come_along(syn, plain):
    k = int(np.searchsorted(plain.src_ids, syn.ids["eyes"]["L"][0]))
    assert plain.edge[k] == pytest.approx(0.5) and plain.edge[0] == pytest.approx(1.0)
    assert np.allclose(plain.uv[0], [syn.model.vertices[plain.src_ids[0]].uv[0], 1 - syn.model.vertices[plain.src_ids[0]].uv[1]], atol=1e-6)
    assert np.allclose(np.linalg.norm(plain.normals, axis=1), 1.0)


def test_weights_are_by_bone_name_and_rows_sum_to_one(plain, syn):
    assert set(plain.weights) >= {"頭", "首", "左目", "右目", "上半身", "左ひざ", "左足首"}
    assert "髪1" not in plain.weights and "ネクタイ1" not in plain.weights
    assert np.allclose(sum(plain.weights.values()), 1.0)
    k = int(np.searchsorted(plain.src_ids, syn.neck[0]))
    assert plain.weights["頭"][k] == pytest.approx(0.5) and plain.weights["首"][k] == pytest.approx(0.5)


# -------------------------------------------------------------------------------------------------------------- bones
def test_bone_pruning_keeps_the_needed_ones_and_drops_dropped_chains(tk):
    names = [b.name for b in tk.bones]
    assert {"全ての親", "センター", "上半身", "首", "頭", "両目", "左目", "右目", "頭先", "左足ＩＫ", "左足", "左ひざ", "左足首", "操作", "左足D"} <= set(names)
    for gone in ("髪1", "髪2", "髪先", "ネクタイ1", "ネクタイ先", "髪の先の先"):
        assert gone not in names
    assert tk.info["dropped_bones"] == 6 and tk.info["kept_bones"] == len(names)


def test_bones_come_parents_first_and_keep_their_data(tk, syn):
    names = [b.name for b in tk.bones]
    for b in tk.bones:
        assert b.parent == "" or names.index(b.parent) < names.index(b.name)
    eye = tk.bone("左目")
    assert eye.grant == {"parent": "両目", "rotate": True, "move": False, "ratio": 1.0} and eye.layer == 2
    assert tk.bone("頭先").visible is False and tk.bone("頭").tail_bone == "頭先"
    ik = tk.bone("左足ＩＫ").ik
    assert ik["target"] == "左足首" and ik["iterations"] == 40 and ik["angle"] == pytest.approx(math.degrees(0.5))
    assert [c["bone"] for c in ik["chain"]] == ["左ひざ", "左足"]
    assert ik["chain"][0]["limit"] == [[pytest.approx(-180.0), 0.0, 0.0], [pytest.approx(math.degrees(-0.01)), 0.0, 0.0]] and ik["chain"][1]["limit"] is None
    assert tk.bone("左足D").grant["parent"] == "左ひざ"
    assert tk.bone("頭").deform and not tk.bone("操作").deform


def test_ik_targets_and_grant_parents_are_kept_even_when_nothing_weights_them(syn, tmp_path):
    # keep only the head material: no vertex uses the leg bones, yet an IK bone with leg links is unweighted-control and stays
    t = T.take(syn.path, ["eyes"])
    names = {b.name for b in t.bones}
    assert {"左足ＩＫ", "左足首", "左ひざ", "左足"} <= names                                  # the IK's targets and links come with it
    assert "左足D" in names and "髪1" not in names
    t2 = T.take(syn.path, ["face"], drop_bones=["操作"], keep_bones=["髪1"])
    n2 = {b.name for b in t2.bones}
    assert "操作" not in n2 and {"髪1", "頭"} <= n2


# --------------------------------------------------------------------------------------------------------- the scaling
def test_uniform_scale_about_the_origin(syn):
    t = T.take(syn.path, syn.keep, scale=0.5)
    p = T.take(syn.path, syn.keep, scale=1.0)
    assert np.allclose(t.verts, 0.5 * p.verts, atol=1e-9)
    assert np.allclose(t.bone("左足ＩＫ").head, 0.5 * np.array(p.bone("左足ＩＫ").head))


def test_head_blend_follows_the_weight_on_the_head_subtree(syn, tk, plain):
    pivot = np.asarray(tk.info["pivot"])
    assert np.allclose(pivot, SCALE * np.array(plain.bone("頭").head), atol=1e-9)
    for k, name in ((0, "head"), (1, "neck"), (2, "body")):
        src = {"head": syn.ids["head"][0], "neck": syn.neck[0], "body": syn.ids["body"][0]}[name]
        i = int(np.searchsorted(tk.src_ids, src))
        w = tk.head_weight[i]
        want = pivot + (SCALE * plain.verts[i] - pivot) * (1 + w * (FACTOR - 1))
        assert np.allclose(tk.verts[i], want, atol=1e-9)
    ih = int(np.searchsorted(tk.src_ids, syn.ids["head"][0]))
    inn = int(np.searchsorted(tk.src_ids, syn.neck[0]))
    ib = int(np.searchsorted(tk.src_ids, syn.ids["body"][0]))
    assert tk.head_weight[ih] == pytest.approx(1.0) and tk.head_weight[inn] == pytest.approx(0.5) and tk.head_weight[ib] == pytest.approx(0.0)
    assert np.allclose(tk.verts[ib], SCALE * plain.verts[ib])                            # body: the plain scale
    ie = int(np.searchsorted(tk.src_ids, syn.ids["eyes"]["L"][0]))
    assert tk.head_weight[ie] == pytest.approx(1.0)                                      # the eyes are in the head's subtree


def test_head_bones_get_the_full_factor_and_the_others_do_not(tk, plain):
    pivot = np.asarray(tk.info["pivot"])
    for n in ("左目", "両目", "頭先"):
        assert np.allclose(tk.bone(n).head, pivot + FACTOR * (SCALE * np.array(plain.bone(n).head) - pivot), atol=1e-9)
    assert np.allclose(tk.bone("頭").head, pivot)
    for n in ("首", "上半身", "左足首"):
        assert np.allclose(tk.bone(n).head, SCALE * np.array(plain.bone(n).head), atol=1e-9)


# --------------------------------------------------------------------------------------------------------------- morphs
def test_vertex_morphs_are_restricted_scaled_and_blended(syn, tk):
    names = [m.name for m in tk.morphs]
    assert "skipme" not in names and "m2" not in names                                    # skipped / only on dropped vertices
    m1 = next(m for m in tk.morphs if m.name == "m1")
    assert m1.kind == "vertex" and m1.panel == "eye" and len(m1.ids) == 3                   # the hair offset was dropped
    pos = {int(tk.src_ids[i]): o for i, o in zip(m1.ids, m1.offsets)}
    want_head = SCALE * np.array([0.001, 0.002, -0.004]) * 1.0 * FACTOR
    assert np.allclose(pos[syn.ids["head"][0]], want_head * (1.0 if False else 1.0), atol=1e-7) or np.allclose(
        pos[syn.ids["head"][0]], SCALE * syn.d["head"] * FACTOR, atol=1e-7)
    assert np.allclose(pos[syn.neck[0]], SCALE * syn.d["neck"] * (1 + 0.5 * (FACTOR - 1)), atol=1e-7)
    assert np.allclose(pos[syn.ids["body"][0]], SCALE * syn.d["body"], atol=1e-7)


def test_material_morphs_keep_only_kept_materials_and_names(tk):
    m = next(m for m in tk.morphs if m.name == "mat")
    assert m.kind == "material"
    assert [(o.material, o.op) for o in m.material_offsets] == [("hidden", 1), (None, 0)]    # the offset for the dropped cloth is gone
    assert m.material_offsets[0].diffuse == (0.0, 0.0, 0.0, 1.0)
    assert "mat_cloth" not in [x.name for x in tk.morphs]                                   # nothing left of it


def test_group_morphs_keep_surviving_members(tk):
    g = next(m for m in tk.morphs if m.name == "g1")
    assert g.kind == "group" and g.members == [("m1", 1.0)]
    assert "g2" not in [m.name for m in tk.morphs]
    order = [m.name for m in tk.morphs]
    assert order.index("m1") < order.index("mat") < order.index("g1")                      # the file's order


# ------------------------------------------------------------------------------------------------------ bodies, frames
def test_rigid_bodies_and_joints_on_kept_bones_only(syn, tk, plain):
    assert [b.name for b in tk.bodies] == ["rb_head", "rb_box"] and tk.joints and [j.name for j in tk.joints] == ["j_kept"]
    h, box = tk.bodies
    pivot = np.asarray(tk.info["pivot"])
    assert h.bone == "頭" and h.shape == "sphere" and h.size[0] == pytest.approx(0.1 * UNIT * SCALE * FACTOR)
    assert np.allclose(h.location, pivot + FACTOR * (SCALE * np.array([0, 0, 1.3]) - pivot), atol=1e-7)
    assert box.shape == "box" and np.allclose(box.size, SCALE * np.array([0.1, 0.3, 0.2]) * UNIT, atol=1e-9)       # PMX (x, y, z) -> model (x, z, y)
    assert box.no_collide == (1,) and box.group == 1
    assert T.take(syn.path, syn.keep).bodies == []


def test_rigid_body_rotation_round_trips_through_the_assembler_convention(syn, tk):
    box = tk.bodies[1]
    back = assemble.pmx_euler(box.rotation)
    assert np.allclose(back, syn.model.bodies[1].rot, atol=1e-6)
    rng = np.random.default_rng(1)
    for _ in range(20):
        e = rng.uniform(-1.2, 1.2, 3)
        assert np.allclose(assemble.matrix_xyz(T._model_euler(assemble.pmx_euler(e))), assemble.matrix_xyz(e), atol=1e-7)


def test_display_frames_keep_the_kept_bones(tk):
    assert tk.frames == {"体": ["上半身", "頭"]}


# ---------------------------------------------------------------------------------------------------------------- pieces
def test_piece_renumbers_and_restricts_morphs(syn, tk):
    p = tk.piece(["face"])
    assert len(p.verts) == len(np.unique(tk.faces["face"]))
    assert len(p.faces) == len(tk.faces["face"]) and p.uv.shape == (3 * len(p.faces), 2)
    assert set(p.weights) == {"頭", "首"} and np.allclose(sum(p.weights.values()), 1.0)
    assert set(p.morphs) == {"m1"} and np.abs(p.morphs["m1"]).max(1).astype(bool).sum() == 2             # head + neck, not the body vertex
    assert (p.ids[p.faces[0][0]] == np.unique(tk.faces["face"])[p.faces[0][0]])
    both = tk.piece(["face", "body"])
    assert set(both.face_mat.tolist()) == {0, 1} and both.mats == ["face", "body"]
    with pytest.raises(KeyError):
        tk.piece(["hair"])


def test_shared_vertices_are_in_both_pieces_with_the_same_data(syn, tk):
    a, b = tk.piece(["face"]), tk.piece(["body"])
    shared = sorted(set(a.ids.tolist()) & set(b.ids.tolist()))
    assert len(shared) == 2                                                               # two ring vertices are used by both materials
    for i in shared:
        ia, ib = int(np.nonzero(a.ids == i)[0][0]), int(np.nonzero(b.ids == i)[0][0])
        assert np.allclose(a.verts[ia], b.verts[ib]) and np.allclose(a.normals[ia], b.normals[ib])
        assert a.weights["頭"][ia] == pytest.approx(b.weights["頭"][ib])


def test_errors_name_the_problem(syn):
    with pytest.raises(ValueError, match="not in"):
        T.take(syn.path, ["nope"])
    with pytest.raises(ValueError, match="head bone"):
        T.take(syn.path, ["face"], head={"bone": "nope", "factor": 1.0})
    with pytest.raises(ValueError, match="drop_bones"):
        T.take(syn.path, ["face"], drop_bones=["nope"])


# ----------------------------------------------------------------------------------------------------------- materials
class Ctx:
    """Just enough BuildCtx for part_materials."""

    def __init__(self, d):
        self.dir = d
        self.names = []

    def save_png(self, name, rgba):
        n = f"t_{name}.png"
        Image.fromarray(np.asarray(rgba)).save(self.dir / n)
        self.names.append(n)
        return n


def test_part_materials_carry_the_settings_and_share_identical_images(syn, tk, tmp_path):
    ctx = Ctx(tmp_path)
    m = T.part_materials(tk, ctx)
    f = m["face"]
    assert f.edge and f.edge_size == pytest.approx(0.8) and f.edge_color == pytest.approx((0.5, 0.3, 0.3, 1.0))
    assert f.texture and f.toon and f.sphere == "" and f.sphere_mode == "none" and not f.alpha_blend
    assert m["body"].texture == f.texture                                                 # skin.png and skin2.png are one image
    assert m["body"].double_sided
    e = m["eyes"]
    assert e.alpha_blend and e.sphere and e.sphere_mode == "add" and e.toon == ""         # a shared toon has no image
    assert m["hidden"].diffuse[3] == 0.0 and m["hidden"].alpha_blend
    assert all(n.isascii() for n in ctx.names) and len(ctx.names) == len(set(ctx.names))
    again = T.part_materials(tk, ctx, ["face"])                                          # a second call (another part) writes nothing new
    assert again["face"].texture == f.texture and len(ctx.names) == len(set(ctx.names))


def test_part_materials_recolour_and_hook_by_file_content(syn, tk, tmp_path):
    ctx = Ctx(tmp_path)
    rules = [dict(texture="tex/skin2.png", rect=[0.5, 0.5, 1.0, 1.0], hue=[[0, 0], [200, 200], [225, 352], [290, 352], [330, 340]], protect=[0.05, 0.2])]
    seen = []

    def hook(img):
        seen.append(img.shape)
        out = np.array(img, copy=True)
        out[0, 0] = (1, 2, 3, 255)
        return out
    m = T.part_materials(tk, ctx, ["face", "body"], recolor=rules, hooks={str(syn.dir / "tex" / "skin.png"): hook})
    assert m["face"].texture == m["body"].texture                                         # the rule written for skin2.png edits skin.png too
    img = np.asarray(Image.open(tmp_path / m["face"].texture).convert("RGB")).astype(int)
    assert img[12, 12][0] > img[12, 12][2]                                                # violet -> rosy
    assert tuple(img[0, 0]) == (1, 2, 3) and len(seen) == 1
    assert abs(img[2, 2] - np.array([240, 220, 200])).max() <= 2                           # elsewhere untouched
    with pytest.raises(ValueError, match="not in"):
        T.part_materials(tk, ctx, ["face"], recolor=[dict(texture="tex/missing.png")])


# -------------------------------------------------------------------------------------------------------------- recolour
def test_recolor_hue_map_wraps_and_protects_low_saturation():
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = (160, 20, 100)          # magenta
    img[1, 0, :3] = (250, 245, 248)         # near white
    img[2, 0, :3] = (30, 200, 60)           # green, outside the rule's hue span (identity anchors)
    out = RC.recolor(img, [dict(hue=[[0, 0], [100, 100], [200, 200], [290, 350], [330, 358]], protect=[0.05, 0.2])])
    h, s, v = RC.rgb_to_hsv(out[..., :3].astype(float) / 255)
    assert h[0, 0] > 340 or h[0, 0] < 10
    assert np.abs(out[1, 0, :3].astype(int) - img[1, 0, :3].astype(int)).max() <= 3
    assert np.abs(out[2, 0, :3].astype(int) - img[2, 0, :3].astype(int)).max() <= 3
    assert (out[..., 3] == 255).all()


def test_recolor_rect_curves_and_alpha():
    img = np.zeros((10, 10, 4), np.uint8)
    img[..., :3] = (100, 50, 50)
    img[..., 3] = 255
    img[0, 9, 3] = 0
    out = RC.recolor(img, [dict(rect=[0.5, 0.0, 1.0, 1.0], val=0.5)])
    assert (out[:, :5, :3] == img[:, :5, :3]).all()
    assert (out[1:, 5:, 0] == 50).all() and out[0, 9, 0] == 100                           # transparent texels are skipped
    assert (out[..., 3] == img[..., 3]).all()
    sat = RC.recolor(img, [dict(sat=[[0, 0], [1, 0]])])
    assert np.ptp(sat[1, 1, :3]) == 0                                                     # saturation curve to zero -> grey
