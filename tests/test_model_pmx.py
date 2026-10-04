"""mkmmd.model.pmx_io: PMX 2.0 / 2.1 reader and writer.

Round trips of a model touching every section and record variant (both encodings, both versions), exact header and
section bytes, index-size boundaries, files built by an independent struct writer (other tools' index sizes), the vertex
section against a naive struct reference, validation messages, and truncated / corrupt input. The check against an
independent implementation (mmd_tools in Blender) needs Blender and lives outside the suite."""
import base64
import copy
import math
import random
import struct
from itertools import chain

import numpy as np
import pytest

from mkmmd.model import pmx_io as P
from mkmmd.model.pmx_io import (
    MaterialMorphOffset, PmxBody, PmxBone, PmxFrame, PmxIK, PmxJoint, PmxMaterial, PmxModel, PmxMorph, PmxVertex)

V = PmxVertex
UP = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 0.0)
ONE3, ONE4 = (1.0,) * 3, (1.0,) * 4


def f32(x):
    return float(np.float32(x))


def sizes_of(data):
    """The six index sizes of a file: vertex, texture, material, bone, morph, rigid body."""
    return tuple(data[11:17])


# ------------------------------------------------------------------------------------------------ the full model
def full_model(version=2.0, encoding="utf-16le"):
    """A model that touches every section and record variant; all numbers are float32-exact (dyadic fractions)."""
    v21 = version >= 2.1
    m = PmxModel(name="凛のモデル", name_en="Rin's model", comment="一行目\r\n二行目 😀", comment_en="line one\r\nline two",
                 version=version, encoding=encoding, add_uv_count=4)
    add = tuple((0.25 * i, 0.5, -1.0, 2.0) for i in range(1, 5))
    m.vertices = [
        V((0.0, 0.5, 1.0), UP, (0.25, 0.75), (0,), (1.0,), add, edge_scale=0.5),
        V((1.0, 2.0, -3.0), (0.0, 0.0, 1.0), (1.0, 0.0), (0, 5), (0.75, 0.25), add),
        V((-1.5, 2.0, 0.125), UP, (0.5, 0.5), (0, 1, 5, 2), (0.5, 0.25, 0.125, 0.125), add, edge_scale=2.0),
        V((2.0, 0.0, 0.0), UP, (0.0, 1.0), (6, 7), (0.5, 0.5), add,
          sdef=((0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.5, 1.0, 0.0)), kind="SDEF"),
        V((0.0, 0.0, 2.0), UP, (1.0, 1.0), (-1,), (1.0,), add),
        V((0.0, 1.0, 2.0), UP, (1.0, 0.5), (1, 2, -1, -1), (0.5, 0.5, 0.0, 0.0), add),
    ]
    if v21:
        m.vertices.append(V((0.5, 0.5, 0.5), UP, (0.5, 0.25), (1, 2, 3, 4), (0.25, 0.25, 0.25, 0.25), add, kind="QDEF"))
    m.faces = [0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 5]
    m.textures = ["tex/skin.png", "tex\\sphere.png", "toon_custom.png"]
    m.materials = [
        PmxMaterial("肌", "skin", (1.0, 0.5, 0.25, 1.0), (0.5, 0.25, 0.125), 8.0, (0.25, 0.25, 0.25),
                    edge_color=(0.0, 0.0, 0.5, 1.0), edge_size=1.5, texture=0, sphere_texture=1, sphere_mode=1,
                    toon_shared=True, toon=3, memo="skin memo", index_count=6),
        PmxMaterial("服", "cloth", double_sided=True, ground_shadow=False, self_shadow_map=False, self_shadow=False,
                    edge=False, sphere_texture=1, sphere_mode=2, toon_shared=False, toon=2, index_count=3),
        PmxMaterial("透", "glass", (0.5, 0.5, 0.5, 0.25), texture=0, sphere_mode=3, toon=-1,
                    vertex_color=v21, point_draw=v21, line_draw=v21, index_count=3),
    ]
    B = PmxBone
    m.bones = [
        B("全ての親", "root", (0.0, 0.0, 0.0), movable=True, tail_offset=(0.0, 1.0, 0.0)),                       # 0
        B("センター", "center", (0.0, 8.0, 0.0), 0, movable=True, tail_bone=2),                                  # 1
        B("足", "leg", (1.0, 8.0, 0.5), 1, layer=3, tail_bone=3),                                                # 2
        B("ひざ", "knee", (1.0, 4.5, 0.5), 2, tail_bone=4),                                                      # 3
        B("足首", "ankle", (1.0, 1.0, 0.5), 3, tail_offset=(0.0, -0.5, -1.0)),                                   # 4
        B("足ＩＫ", "legIK", (1.0, 1.0, 0.5), 0, layer=1, movable=True, tail_offset=(0.0, 0.0, 1.0),
          ik=PmxIK(4, 40, 2.0, [(3, True, (-3.0, 0.0, 0.0), (-0.0625, 0.0, 0.0)), (2, False, Z, Z)])),           # 5
        B("左腕", "arm", (2.0, 9.0, 0.0), 1, tail_bone=7, local_x=(1.0, 0.0, 0.0), local_z=(0.0, 0.0, -1.0)),    # 6
        B("左腕捩", "twist", (3.0, 9.0, 0.0), 6, layer=2, tail_offset=(1.0, 0.0, 0.0), fixed_axis=(1.0, 0.0, 0.0),
          grant_rotate=True, grant_parent=6, grant_ratio=0.5),                                                   # 7
        B("移動付与", "grantmove", (0.0, 9.0, 0.0), 1, tail_bone=-1, grant_move=True, grant_parent=1, grant_ratio=-1.0),
        B("両付与", "grantboth", (0.0, 10.0, 0.0), 2, tail_bone=-1, grant_rotate=True, grant_move=True,
          grant_parent=2, grant_ratio=0.25),                                                                     # 9
        B("髪", "hair", (0.0, 12.0, 0.0), 1, tail_offset=(0.0, 1.0, 0.0), after_physics=True, ext_parent=3),     # 10
        B("隠し", "hidden", (0.0, 3.0, 0.0), -1, tail_bone=-1, visible=False, controllable=False, rotatable=False,
          movable=True),                                                                                         # 11
    ]
    m.morphs = [
        PmxMorph("あ", "a", 3, "vertex", [(0, (0.0, 0.5, 0.0)), (1, (0.0, 0.25, 0.0)), (4, (-0.5, 0.0, 0.125))]),
        PmxMorph("まばたき", "blink", 2, "vertex", [(2, (0.0, -0.25, 0.0))]),
        PmxMorph("眉", "brow", 1, "uv", [(0, (0.25, 0.25, 0.0, 0.0))]),
        PmxMorph("uv1", "", 4, "uv1", [(1, (0.5, 0.5, 0.5, 0.5))]),
        PmxMorph("uv2", "", 4, "uv2", [(2, (0.5, 0.0, 0.5, 0.0))]),
        PmxMorph("uv3", "", 4, "uv3", [(3, (0.0, 0.5, 0.0, 0.5))]),
        PmxMorph("uv4", "", 4, "uv4", [(4, (1.0, 1.0, 1.0, 1.0)), (0, (0.0, 0.0, 0.0, 0.25))]),
        PmxMorph("体", "bonemorph", 4, "bone", [(1, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                                              (2, (0.5, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))]),
        PmxMorph("材質", "material", 4, "material", [
            MaterialMorphOffset.of(0, 1, diffuse=(0.25, 0.0, 0.0, 0.0)),
            (-1, 0, (1.0, 1.0, 1.0, 0.5), ONE3, 1.0, ONE3, ONE4, 1.0, ONE4, ONE4, ONE4)]),
        PmxMorph("口", "mouth", 3, "group", [(0, 1.0), (1, 0.5), (7, -0.25)]),
        PmxMorph("空", "empty", 0, "vertex", []),
    ]
    if v21:
        m.morphs += [
            PmxMorph("反転", "flip", 4, "flip", [(0, 1.0), (1, 0.5)]),
            PmxMorph("衝撃", "impulse", 4, "impulse", [(0, True, (0.0, 1.0, 0.0), (0.5, 0.0, 0.0)),
                                                     (1, False, (1.0, 0.0, 0.0), (0.0, 0.0, 0.25))]),
        ]
    m.frames = [
        PmxFrame("Root", "Root", True, [("bone", 0)]),
        PmxFrame("表情", "Exp", True, [("morph", i) for i in range(len(m.morphs))]),
        PmxFrame("体", "Body", False, [("bone", 1), ("bone", 2), ("bone", 5), ("morph", 0)]),
        PmxFrame("空"),
    ]
    m.bodies = [
        PmxBody("体", "body", 1, 0, 0xFFFE, 2, (0.5, 1.0, 0.0), (0.0, 8.0, 0.0), Z, 1.0, 0.5, 0.5, 0.0, 0.5, 0),
        PmxBody("髪", "hair", 10, 15, 0x7FFF, 0, (0.5, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 0.25), 0.5, 0.875, 0.75,
                0.125, 0.25, 1),
        PmxBody("箱", "box", 11, 3, 0xFFFF, 1, (0.5, 0.25, 0.125), (1.0, 2.0, 3.0), (0.5, -0.5, 1.0), 2.0, 0.0, 1.0,
                1.0, 1.0, 2),
        PmxBody("無", "nobone", -1, 1, 0x0000, 0, (1.0, 0.0, 0.0)),
    ]
    m.joints = [
        PmxJoint("髪J", "hairJ", 0, 0, 1, (0.0, 12.0, 0.0), (0.0, 0.0, 0.5), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5),
                 (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        PmxJoint("none", "", 0, -1, 3),
    ]
    if v21:
        m.joints += [PmxJoint(f"kind{k}", "", k, 0, 1, (float(k), 0.0, 0.0)) for k in range(1, 6)]
    return m


def small(version=2.0):
    """A minimal valid model with one of each entity, for the validation cases."""
    m = PmxModel(name="m", version=version)
    m.vertices = [V((0.0, 0.0, 0.0), UP, (0.0, 0.0), (0,)),
                  V((1.0, 0.0, 0.0), UP, (1.0, 0.0), (0, 1), (0.5, 0.5)),
                  V((0.0, 1.0, 0.0), UP, (0.0, 1.0), (0, 1, 2, 0), (0.5, 0.25, 0.25, 0.0))]
    m.faces = [0, 1, 2]
    m.textures = ["a.png"]
    m.materials = [PmxMaterial("mat", texture=0, index_count=3)]
    m.bones = [PmxBone("root", tail_offset=(0.0, 1.0, 0.0)), PmxBone("b1", parent=0, tail_bone=2),
               PmxBone("b2", parent=1, tail_offset=(0.0, 1.0, 0.0))]
    m.morphs = [PmxMorph("vm", offsets=[(0, (0.0, 1.0, 0.0))]), PmxMorph("gm", kind="group", offsets=[(0, 1.0)])]
    m.frames = [PmxFrame("f", items=[("bone", 0), ("morph", 1)])]
    m.bodies = [PmxBody("b0", bone=1), PmxBody("b1", bone=2, mode=1)]
    m.joints = [PmxJoint("j", body_a=0, body_b=1)]
    return m


# ------------------------------------------------------------------------------------------------ round trips
@pytest.mark.parametrize("encoding", ["utf-16le", "utf-8"])
@pytest.mark.parametrize("version", [2.0, 2.1])
def test_full_model_round_trip(version, encoding):
    m = full_model(version, encoding)
    data = P.to_bytes(m)
    m2 = P.from_bytes(data)
    assert m2 == m
    assert m2.version == version and m2.encoding == encoding and m2.add_uv_count == 4
    assert P.to_bytes(m2) == data                       # write -> read -> write is stable


def test_round_trip_through_files(tmp_path):
    m = full_model(2.0)
    path = tmp_path / "x.pmx"
    P.write(m, str(path))
    assert path.read_bytes() == P.to_bytes(m)
    assert P.read(str(path)) == m
    assert P.read(path) == m                            # pathlib paths work too


def test_empty_model_exact_bytes():
    for version in (2.0, 2.1):
        for encoding, code, enc in (("utf-16le", 0, "utf-16-le"), ("utf-8", 1, "utf-8")):
            m = PmxModel(name="a", name_en="bc", comment="日本", comment_en="", version=version, encoding=encoding,
                         add_uv_count=2)
            data = P.to_bytes(m)
            text = lambda s: struct.pack("<i", len(s.encode(enc))) + s.encode(enc)
            expected = (b"PMX " + struct.pack("<f", version) + bytes([8, code, 2, 1, 1, 1, 1, 1, 1])
                        + text("a") + text("bc") + text("日本") + text("") + struct.pack("<9i", *[0] * 9)
                        + (struct.pack("<i", 0) if version >= 2.1 else b""))
            assert data == expected
            assert P.from_bytes(data) == m


def test_unicode_names_and_surrogate_pairs():
    m = PmxModel(name="𝄞 emoji 😀 日本語 ＩＫ", name_en="")
    m.bones = [PmxBone("左足𠮷")]
    for enc in ("utf-16le", "utf-8"):
        m.encoding = enc
        assert P.from_bytes(P.to_bytes(m)) == m
    m.name = "lone \ud800 surrogate"
    with pytest.raises(ValueError, match="cannot encode"):
        P.to_bytes(m)


def test_encoding_aliases_and_version_float_rounding():
    m = PmxModel(name="x", encoding="UTF-16-LE", version=2.1)
    assert P.from_bytes(P.to_bytes(m)).encoding == "utf-16le"
    assert P.from_bytes(P.to_bytes(m)).version == 2.1
    m.encoding = "utf_8"
    assert P.from_bytes(P.to_bytes(m)).encoding == "utf-8"


def test_reader_returns_tuples_and_python_numbers():
    m2 = P.from_bytes(P.to_bytes(full_model(2.1)))
    for v in m2.vertices:
        assert type(v.pos) is tuple and type(v.weights) is tuple and type(v.bones) is tuple and type(v.edge_scale) is float
        assert all(type(x) is float for x in v.pos) and all(type(b) is int for b in v.bones)
    assert type(m2.faces) is list and all(type(i) is int for i in m2.faces)
    assert all(isinstance(o, MaterialMorphOffset) for o in m2.morphs[8].offsets)
    assert m2.morphs[8].offsets[0].diffuse == (0.25, 0.0, 0.0, 0.0) and m2.morphs[8].offsets[1].index == -1


def test_mixed_inputs_are_accepted_and_write_identical_bytes():
    ref = full_model(2.0)
    alt = full_model(2.0)
    for v in alt.vertices:
        v.pos, v.normal, v.uv = np.array(v.pos), list(v.normal), np.array(v.uv, dtype=np.float32)
        v.bones, v.weights = np.array(v.bones, dtype=np.int64), np.array(v.weights)
        v.add_uv = [np.array(a) for a in v.add_uv]
    alt.faces = np.array(alt.faces)
    alt.bones[5].ik.links = [list(link) for link in alt.bones[5].ik.links]
    alt.morphs[0].offsets = [(np.int64(i), np.array(d)) for i, d in alt.morphs[0].offsets]
    alt.morphs[8].offsets[1] = dict(index=-1, op=0, diffuse=(1.0, 1.0, 1.0, 0.5))
    assert P.to_bytes(alt) == P.to_bytes(ref)


def test_kind_explicit_matches_inferred():
    a, b = small(), small()
    for v, kind in zip(b.vertices, ("BDEF1", "BDEF2", "BDEF4")):
        v.kind = kind
    assert P.to_bytes(a) == P.to_bytes(b)
    assert P.from_bytes(P.to_bytes(b)) == a             # the reader reports BDEF vertices as kind ''


def test_ik_link_forms_and_material_morph_forms():
    m = small()
    m.bones[2].ik = PmxIK(0, 8, 0.5, [(1, False), (0, False, None, None)])
    m.morphs.append(PmxMorph("mm", kind="material", offsets=[
        MaterialMorphOffset.of(0, 1, diffuse=(0.5, 0.0, 0.0, 0.0), shininess=2.0),
        (0, 1, (0.5, 0.0, 0.0, 0.0), (0.0,) * 3, 2.0, (0.0,) * 3, (0.0,) * 4, 0.0, (0.0,) * 4, (0.0,) * 4, (0.0,) * 4),
        {"index": 0, "op": 1, "diffuse": (0.5, 0.0, 0.0, 0.0), "shininess": 2.0},
    ]))
    m2 = P.from_bytes(P.to_bytes(m))
    assert m2.bones[2].ik.links == [(1, False, Z, Z), (0, False, Z, Z)]
    offs = m2.morphs[-1].offsets
    assert offs[0] == offs[1] == offs[2] == MaterialMorphOffset.of(0, 1, diffuse=(0.5, 0.0, 0.0, 0.0), shininess=2.0)
    mul = MaterialMorphOffset.of(-1, 0)                 # a multiply defaults to the neutral value 1.0
    assert mul.diffuse == ONE4 and mul.edge_size == 1.0 and MaterialMorphOffset.of(-1, 1).diffuse == (0.0,) * 4
    with pytest.raises(ValueError, match="unknown material morph offset field"):
        MaterialMorphOffset.of(0, 1, colour=(1, 1, 1))


def test_duplicate_names_are_allowed():
    m = small()
    m.bones[1].name = m.bones[2].name = "same"
    m.materials.append(PmxMaterial("mat", index_count=0))
    m.validate()
    assert P.from_bytes(P.to_bytes(m)) == m


# ------------------------------------------------------------------------------------------------ vertex section
def naive_vertex_bytes(vs, bone_fmt):
    """Vertex records packed one by one with struct, straight from the format description."""
    out = b""
    for v in vs:
        kind = v.kind or {1: "BDEF1", 2: "BDEF2", 4: "BDEF4"}[len(v.bones)]
        out += struct.pack("<8f", *v.pos, *v.normal, *v.uv)
        out += b"".join(struct.pack("<4f", *a) for a in v.add_uv)
        out += bytes([("BDEF1", "BDEF2", "BDEF4", "SDEF", "QDEF").index(kind)])
        out += b"".join(struct.pack("<" + bone_fmt, b) for b in v.bones)
        if kind == "BDEF2":
            out += struct.pack("<f", v.weights[0])
        elif kind in ("BDEF4", "QDEF"):
            out += struct.pack("<4f", *v.weights)
        elif kind == "SDEF":
            out += struct.pack("<f9f", v.weights[0], *chain.from_iterable(v.sdef))
        out += struct.pack("<f", v.edge_scale)
    return out


@pytest.mark.parametrize("nuv", [0, 1, 4])
@pytest.mark.parametrize("nbones, bone_fmt", [(100, "b"), (300, "h")])
def test_vertex_section_matches_naive_struct_reference(nuv, nbones, bone_fmt):
    rng = random.Random(nuv * 1000 + nbones)
    q = lambda lo, hi: f32(rng.uniform(lo, hi))
    vs = []
    for _ in range(1500):
        pos, nrm, uv = [tuple(q(-3, 3) for _ in range(k)) for k in (3, 3, 2)]
        add = tuple(tuple(q(0, 1) for _ in range(4)) for _ in range(nuv))
        e = q(0, 2)
        kind = rng.choice(["BDEF1", "BDEF2", "BDEF4", "SDEF", "QDEF", "BDEF4"])
        n = {"BDEF1": 1, "BDEF2": 2, "SDEF": 2}.get(kind, 4)
        bones = tuple(rng.randrange(-1, nbones) for _ in range(n))
        w = tuple(q(0, 1) for _ in range(n))
        if n == 2:                                      # the file stores weight 0 only; the reader restores 1 - w0
            w = (w[0], 1.0 - w[0])
        if n == 1:
            w = (1.0,)
        sdef = tuple(tuple(q(-1, 1) for _ in range(3)) for _ in range(3)) if kind == "SDEF" else None
        vs.append(V(pos, nrm, uv, bones, w, add, sdef, kind if kind in ("SDEF", "QDEF") else "", e))
    m = PmxModel(vertices=vs, add_uv_count=nuv, version=2.1, bones=[PmxBone(f"b{i}") for i in range(nbones)])
    data = P.to_bytes(m)
    start = 17 + 4 * 4 + 4                              # header, four empty texts, vertex count
    expected = naive_vertex_bytes(vs, bone_fmt)
    assert data[start:start + len(expected)] == expected
    assert P.from_bytes(data).vertices == vs


def test_single_kind_and_empty_vertex_lists():
    for kind_bones in ((1,), (2,), (4,)):
        w = (1.0,) if len(kind_bones) == 1 else (0.5, 0.5) if len(kind_bones) == 2 else (0.25,) * 4
        m = PmxModel(vertices=[V((float(i), 0.0, 0.0), UP, (0.0, 0.0), tuple(range(len(kind_bones))), w) for i in range(5)],
                     bones=[PmxBone(f"b{i}") for i in range(4)])
        assert P.from_bytes(P.to_bytes(m)) == m
    assert P.from_bytes(P.to_bytes(PmxModel())).vertices == []


# ------------------------------------------------------------------------------------------------ header and sizes
@pytest.mark.parametrize("encoding, code", [("utf-16le", 0), ("utf-8", 1)])
@pytest.mark.parametrize("version", [2.0, 2.1])
@pytest.mark.parametrize("nuv", [0, 3, 4])
def test_header_bytes(version, encoding, code, nuv):
    data = P.to_bytes(PmxModel(version=version, encoding=encoding, add_uv_count=nuv))
    assert data[:4] == b"PMX "
    assert struct.unpack("<f", data[4:8])[0] == pytest.approx(version, abs=1e-6)
    assert data[8] == 8
    assert list(data[9:17]) == [code, nuv, 1, 1, 1, 1, 1, 1]


def test_index_size_thresholds():
    assert [P._isz(n, True) for n in (0, 1, 127, 128, 32767, 32768, 10 ** 6)] == [1, 1, 1, 2, 2, 4, 4]
    assert [P._isz(n, False) for n in (0, 1, 255, 256, 65535, 65536, 10 ** 6)] == [1, 1, 1, 2, 2, 4, 4]


@pytest.mark.parametrize("n, size", [(1, 1), (255, 1), (256, 2), (65535, 2), (65536, 4)])
def test_vertex_index_size_boundaries(n, size):
    v = V((0.0, 0.0, 0.0), UP, (0.0, 0.0), (-1,))
    last = V((1.0, 2.0, 3.0), UP, (0.5, 0.5), (-1,))
    m = PmxModel(vertices=[v] * (n - 1) + [last], faces=[0, n // 2, n - 1], materials=[PmxMaterial("m", index_count=3)],
                 morphs=[PmxMorph("mo", offsets=[(n - 1, (0.5, 0.0, 0.0))]), PmxMorph("uv", kind="uv", offsets=[(n - 1, ONE4)])])
    data = P.to_bytes(m)
    assert sizes_of(data)[0] == size
    m2 = P.from_bytes(data)
    assert len(m2.vertices) == n and m2.vertices[-1] == last and m2.vertices[0] == (v if n > 1 else last)
    assert m2.faces == [0, n // 2, n - 1]
    assert m2.morphs == m.morphs


def _cheap(kind, n):
    """A model with n entities of one signed-index kind and references to the last one."""
    last = n - 1
    if kind == "texture":
        return PmxModel(textures=[f"t{i}.png" for i in range(n)], materials=[
            PmxMaterial("m", texture=last, sphere_texture=0, toon=last)]), 1
    if kind == "morph":
        ms = [PmxMorph(f"m{i}", kind="group") for i in range(n)]
        if n > 1:
            ms[0] = PmxMorph("g", kind="group", offsets=[(last, 0.5)])
        return PmxModel(morphs=ms, frames=[PmxFrame("f", items=[("morph", last)])]), 4
    if kind == "body":
        return PmxModel(bodies=[PmxBody(f"b{i}") for i in range(n)], joints=[PmxJoint("j", body_a=last, body_b=0)]), 5
    if kind == "material":
        ms = [PmxMaterial(f"m{i}") for i in range(n)]
        return PmxModel(materials=ms, morphs=[PmxMorph("mm", kind="material", offsets=[
            MaterialMorphOffset.of(last, 1, diffuse=(0.5, 0.0, 0.0, 0.0))])]), 2
    bones = [PmxBone(f"b{i}", parent=i - 1) for i in range(n)]
    return PmxModel(bones=bones, vertices=[V((0.0, 0.0, 0.0), UP, (0.0, 0.0), (last,))],
                    bodies=[PmxBody("b", bone=last)], frames=[PmxFrame("f", items=[("bone", last)])]), 3


@pytest.mark.parametrize("kind", ["texture", "morph", "body", "material", "bone"])
@pytest.mark.parametrize("n, size", [(1, 1), (127, 1), (128, 2)])
def test_signed_index_size_boundaries_small(kind, n, size):
    m, slot = _cheap(kind, n)
    data = P.to_bytes(m)
    assert sizes_of(data)[slot] == size
    assert P.from_bytes(data) == m


@pytest.mark.parametrize("kind", ["texture", "morph", "body"])
@pytest.mark.parametrize("n, size", [(32767, 2), (32768, 4)])
def test_signed_index_size_boundaries_large(kind, n, size):
    m, slot = _cheap(kind, n)
    data = P.to_bytes(m)
    assert sizes_of(data)[slot] == size
    m2 = P.from_bytes(data)
    assert m2 == m


@pytest.mark.parametrize("n, size", [(32767, 2), (32768, 4)])
def test_large_bone_counts_around_the_int16_boundary(n, size):
    # the cheapest valid entities: one shared bone object repeated, vertices / bodies referring to the last one
    m = PmxModel(bones=[PmxBone("b")] * n, vertices=[V((0.0, 0.0, 0.0), UP, (0.0, 0.0), (n - 1,))],
                 bodies=[PmxBody("b", bone=n - 1)])
    data = P.to_bytes(m)
    assert sizes_of(data)[3] == size
    m2 = P.from_bytes(data)
    assert len(m2.bones) == n and m2.vertices == m.vertices and m2.bodies[0].bone == n - 1


def test_index_sizes_override_forces_larger_sizes_and_roundtrips():
    m = full_model(2.1)
    ref = P.to_bytes(m)
    assert sizes_of(ref) == (1, 1, 1, 1, 1, 1)
    data = P.to_bytes(m, index_sizes={"vertex": 2, "texture": 4, "material": 2, "bone": 4, "morph": 2, "body": 4})
    assert sizes_of(data) == (2, 4, 2, 4, 2, 4) and len(data) > len(ref)
    assert P.from_bytes(data) == m
    data = P.to_bytes(m, index_sizes={"vertex": 4, "bone": 2})
    assert sizes_of(data) == (4, 1, 1, 2, 1, 1) and P.from_bytes(data) == m
    with pytest.raises(ValueError, match="unknown index_sizes key"):
        P.to_bytes(m, index_sizes={"vertices": 2})
    with pytest.raises(ValueError, match="must be 1, 2 or 4"):
        P.to_bytes(m, index_sizes={"bone": 3})
    big, _ = _cheap("morph", 128)
    with pytest.raises(ValueError, match="at least 2"):
        P.to_bytes(big, index_sizes={"morph": 1})


# ------------------------------------------------------------------------------------------------ other tools' files
def _foreign(version, utf8, sizes):
    """A tiny PMX assembled with plain struct calls from the format description (not with pmx_io), using the given
    index sizes (vertex, texture, material, bone, morph, rigid body)."""
    sv, st, sm, sb, so, sr = sizes
    enc = "utf-8" if utf8 else "utf-16-le"
    s = lambda t: struct.pack("<i", len(t.encode(enc))) + t.encode(enc)
    vi = lambda i: struct.pack("<" + {1: "B", 2: "H", 4: "i"}[sv], i)
    ix = lambda size, i: struct.pack("<" + {1: "b", 2: "h", 4: "i"}[size], i)
    fl = lambda *a: struct.pack("<%df" % len(a), *a)
    i32 = lambda i: struct.pack("<i", i)
    out = b"PMX " + struct.pack("<f", version) + bytes([8, 1 if utf8 else 0, 1, sv, st, sm, sb, so, sr])
    out += s("名前") + s("name") + s("コメント") + s("comment")
    out += i32(3)                                                       # vertices: BDEF1, BDEF2, SDEF, one extra UV
    for pos, kind in (((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 3)):
        out += fl(*pos) + fl(0, 1, 0) + fl(0.5, 0.5) + fl(1, 2, 3, 4) + bytes([kind])
        if kind == 0:
            out += ix(sb, 1)
        elif kind == 1:
            out += ix(sb, 0) + ix(sb, 1) + fl(0.75)
        else:
            out += ix(sb, 0) + ix(sb, 1) + fl(0.5) + fl(0, 1, 0) + fl(0.5, 1, 0) + fl(1.5, 1, 0)
        out += fl(1.0)
    out += i32(3) + vi(0) + vi(1) + vi(2)                               # faces
    out += i32(1) + s("a\\b.png")                                       # textures
    out += i32(1) + s("材質") + s("mat") + fl(1, 0.5, 0.25, 1) + fl(0.5, 0.5, 0.5) + fl(8) + fl(0.25, 0.25, 0.25)
    out += bytes([0b00011011]) + fl(0, 0, 0, 1) + fl(1.0) + ix(st, 0) + ix(st, -1) + bytes([0, 1, 2]) + s("memo") + i32(3)
    out += i32(2)                                                       # bones
    out += s("親") + s("root") + fl(0, 0, 0) + ix(sb, -1) + i32(0) + struct.pack("<H", 0x001F) + ix(sb, 1)
    out += s("子") + s("child") + fl(0, 1, 0) + ix(sb, 0) + i32(1) + struct.pack("<H", 0x003E) + fl(0, 0, 1)
    out += ix(sb, 0) + i32(4) + fl(0.5) + i32(1) + ix(sb, 0) + bytes([1]) + fl(-1, 0, 0) + fl(0, 0, 0)
    out += i32(2)                                                       # morphs
    out += s("頂点") + s("v") + bytes([3, 1]) + i32(1) + vi(2) + fl(0, 0.5, 0)
    out += s("グループ") + s("g") + bytes([4, 0]) + i32(1) + ix(so, 0) + fl(1.0)
    out += i32(1) + s("Root") + s("Root") + bytes([1]) + i32(2) + bytes([0]) + ix(sb, 0) + bytes([1]) + ix(so, 1)
    out += i32(1) + s("剛体") + s("body") + ix(sb, 1) + bytes([2]) + struct.pack("<H", 0xFFFE) + bytes([2])
    out += fl(0.5, 1, 0) + fl(0, 8, 0) + fl(0, 0, 0.25) + fl(1, 0.5, 0.5, 0, 0.5) + bytes([1])
    out += i32(1) + s("J") + s("j") + bytes([0]) + ix(sr, 0) + ix(sr, -1) + fl(*range(24))
    if version >= 2.1:
        out += i32(0)
    return out


def _foreign_expected(version, utf8):
    m = PmxModel("名前", "name", "コメント", "comment", version, "utf-8" if utf8 else "utf-16le", 1)
    ex = ((1.0, 2.0, 3.0, 4.0),)
    m.vertices = [
        V((0.0, 0.0, 0.0), UP, (0.5, 0.5), (1,), (1.0,), ex),
        V((1.0, 0.0, 0.0), UP, (0.5, 0.5), (0, 1), (0.75, 0.25), ex),
        V((0.0, 1.0, 0.0), UP, (0.5, 0.5), (0, 1), (0.5, 0.5), ex,
          ((0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.5, 1.0, 0.0)), "SDEF"),
    ]
    m.faces = [0, 1, 2]
    m.textures = ["a\\b.png"]
    m.materials = [PmxMaterial("材質", "mat", (1.0, 0.5, 0.25, 1.0), (0.5, 0.5, 0.5), 8.0, (0.25, 0.25, 0.25),
                               double_sided=True, ground_shadow=True, self_shadow_map=False, self_shadow=True, edge=True,
                               edge_color=(0.0, 0.0, 0.0, 1.0), edge_size=1.0, texture=0, sphere_texture=-1,
                               sphere_mode=0, toon_shared=True, toon=2, memo="memo", index_count=3)]
    m.bones = [PmxBone("親", "root", (0.0, 0.0, 0.0), -1, 0, tail_bone=1, movable=True),
               PmxBone("子", "child", (0.0, 1.0, 0.0), 0, 1, tail_offset=(0.0, 0.0, 1.0), movable=True,
                       ik=PmxIK(0, 4, 0.5, [(0, True, (-1.0, 0.0, 0.0), Z)]))]
    m.morphs = [PmxMorph("頂点", "v", 3, "vertex", [(2, (0.0, 0.5, 0.0))]),
                PmxMorph("グループ", "g", 4, "group", [(0, 1.0)])]
    m.frames = [PmxFrame("Root", "Root", True, [("bone", 0), ("morph", 1)])]
    m.bodies = [PmxBody("剛体", "body", 1, 2, 0xFFFE, 2, (0.5, 1.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 0.25), 1.0, 0.5,
                        0.5, 0.0, 0.5, 1)]
    floats = [tuple(float(x) for x in range(i, i + 3)) for i in range(0, 24, 3)]
    m.joints = [PmxJoint("J", "j", 0, 0, -1, *floats)]
    return m


@pytest.mark.parametrize("utf8", [False, True])
@pytest.mark.parametrize("version", [2.0, 2.1])
@pytest.mark.parametrize("sizes", [(1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2), (4, 4, 4, 4, 4, 4), (4, 1, 2, 4, 1, 2),
                                   (2, 4, 1, 2, 4, 1)])
def test_reader_accepts_files_with_any_index_sizes_and_encodings(version, utf8, sizes):
    assert P.from_bytes(_foreign(version, utf8, sizes)) == _foreign_expected(version, utf8)


def test_writer_output_equals_the_independent_struct_layout():
    for version in (2.0, 2.1):
        for utf8 in (False, True):
            expected = _foreign_expected(version, utf8)
            assert P.to_bytes(expected) == _foreign(version, utf8, (1, 1, 1, 1, 1, 1))


# A file written by the PMX exporter of mmd_tools 4.5.13 (an independent writer): a Blender import of a model generated
# with this module (BDEF1/2/4 + SDEF vertices, IK with and without limit, grant, local / fixed axes, vertex, group, bone,
# material and UV morphs, frames, rigid bodies, a joint) exported again by mmd_tools. All content is generated.
MMD_TOOLS_EXPORT = (
    "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"
)


def test_file_written_by_the_mmd_tools_exporter_is_read_and_rewritten_identically():
    data = base64.b64decode(MMD_TOOLS_EXPORT)
    m = P.from_bytes(data)
    assert (m.version, m.encoding, m.add_uv_count) == (2.0, "utf-16le", 1)
    assert (m.name, m.name_en, m.comment, m.comment_en) == ("チェック", "check", "line1\r\nline2", "en")
    assert [len(x) for x in (m.vertices, m.faces, m.textures, m.materials, m.bones, m.morphs, m.frames, m.bodies,
                             m.joints)] == [4, 6, 1, 2, 9, 7, 3, 2, 1]
    assert [(v.kind or len(v.bones)) for v in m.vertices] == [4, 2, 1, "SDEF"]
    assert m.vertices[0].weights == (0.5, 0.25, 0.125, 0.125) and m.vertices[0].bones == (0, 2, 3, 4)
    assert m.vertices[1].weights == (0.75, 0.25) and m.vertices[3].sdef == ((0.0, 1.0, 0.0), (0.5, 1.0, 0.0),
                                                                           (1.5, 1.0, 0.0))
    assert m.vertices[1].add_uv == ((1.0, 1.0, 1.0, 1.0),)
    ik = m.bones[5]
    assert (ik.name, ik.layer, ik.movable, ik.tail_offset) == ("足ＩＫ", 1, True, (0.0, 0.0, 1.0))
    assert ik.ik == PmxIK(4, 40, 2.0, [(3, True, (-3.0, 0.0, 0.0), (-0.0625, 0.0, 0.0)), (2, False, Z, Z)])
    assert (m.bones[6].local_x, m.bones[6].local_z, m.bones[6].tail_bone) == ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), 7)
    twist = m.bones[7]
    assert (twist.fixed_axis, twist.grant_rotate, twist.grant_move, twist.grant_parent, twist.grant_ratio,
            twist.after_physics, twist.layer) == ((1.0, 0.0, 0.0), True, False, 6, 0.5, True, 2)
    assert [(x.kind, len(x.offsets)) for x in m.morphs] == [
        ("vertex", 2), ("vertex", 1), ("group", 2), ("bone", 1), ("material", 2), ("uv", 1), ("uv1", 1)]
    assert [(f.name, f.special, len(f.items)) for f in m.frames] == [("Root", True, 1), ("表情", True, 7), ("体", False, 3)]
    assert [(b.group, b.mask, b.shape, b.mode, b.bone) for b in m.bodies] == [(0, 0xFFFE, 2, 0, 1), (1, 0xFFFD, 0, 1, 8)]
    assert (m.joints[0].body_a, m.joints[0].body_b, m.joints[0].spring_rot) == (0, 1, (2.0, 2.0, 2.0))
    assert [(x.toon_shared, x.toon, x.texture, x.double_sided, x.edge) for x in m.materials] == [
        (True, 1, 0, False, True), (False, -1, -1, True, False)]
    m.validate()
    assert P.to_bytes(m) == data                          # our writer reproduces mmd_tools' bytes exactly


def test_version_2_1_soft_bodies():
    base = _foreign(2.1, False, (1,) * 6)
    assert P.from_bytes(base[:-4]) == _foreign_expected(2.1, False)             # a 2.1 file may omit the section
    with pytest.raises(NotImplementedError, match="soft bod"):
        P.from_bytes(base[:-4] + struct.pack("<i", 2) + b"\x00" * 16)
    assert P.from_bytes(_foreign(2.0, False, (1,) * 6) + b"trailing junk") == _foreign_expected(2.0, False)


def test_reader_accepts_extra_global_values():
    data = bytearray(_foreign(2.0, False, (1,) * 6))
    data[8] = 10                                                                # header with two more global bytes
    data[17:17] = b"\x05\x06"
    assert P.from_bytes(bytes(data)) == _foreign_expected(2.0, False)


def test_reader_ignores_material_flag_bits_that_2_0_does_not_define():
    data = bytearray(_foreign(2.0, False, (1,) * 6))
    flag_at = data.index(bytes([0b00011011]))
    data[flag_at] |= 0b11100000
    assert P.from_bytes(bytes(data)).materials[0].vertex_color is False
    data = bytearray(_foreign(2.1, False, (1,) * 6))
    data[flag_at] |= 0b11100000
    mt = P.from_bytes(bytes(data)).materials[0]
    assert (mt.vertex_color, mt.point_draw, mt.line_draw) == (True, True, True)


# ------------------------------------------------------------------------------------------------ corrupt input
@pytest.mark.parametrize("version", [2.0, 2.1])
def test_truncated_data_only_raises_value_error(version):
    data = P.to_bytes(full_model(version))
    last = len(data) - (0 if version < 2.1 else 4)                                 # a 2.1 file may omit the final count
    cuts = sorted(set(range(0, last, 3)) | set(range(64)) | set(range(last - 64, last)))
    for cut in cuts:
        with pytest.raises(ValueError):
            P.from_bytes(data[:cut])
    assert P.from_bytes(data[:last]).name == "凛のモデル"


@pytest.mark.parametrize("version, encoding", [(2.0, "utf-16le"), (2.1, "utf-8")])
def test_corrupted_bytes_raise_value_error_or_parse(version, encoding):
    rng = random.Random(version)
    data = P.to_bytes(full_model(version, encoding))
    for _ in range(400):
        b = bytearray(data)
        for _ in range(rng.choice((1, 1, 2, 4, 8))):
            i = rng.randrange(len(b)) if rng.random() < 0.7 else rng.randrange(400)
            b[i] = rng.choice((0x00, 0xFF, 0x7F, 0x80, rng.randrange(256)))
        try:
            P.from_bytes(bytes(b))
        except (ValueError, NotImplementedError):
            pass                                        # anything else (struct.error, IndexError, ...) fails the test


def test_corrupt_headers_and_counts():
    good = _foreign(2.0, False, (1,) * 6)
    cases = [
        ("not a PMX file", b"PMD " + good[4:]),
        ("unsupported PMX version 3.0", good[:4] + struct.pack("<f", 3.0) + good[8:]),
        ("unsupported PMX version 1.0", good[:4] + struct.pack("<f", 1.0) + good[8:]),
        ("unknown text encoding id 7", good[:9] + b"\x07" + good[10:]),
        ("index size 3", good[:11] + b"\x03" + good[12:]),
        ("5 additional UVs", good[:10] + b"\x05" + good[11:]),
        ("4 global values", good[:8] + b"\x04" + good[9:]),
    ]
    for message, data in cases:
        with pytest.raises(ValueError, match=message):
            P.from_bytes(data)
    texts = ("名前", "name", "コメント", "comment")
    count_at = 17 + sum(4 + len(t.encode("utf-16-le")) for t in texts)             # the vertex count
    assert struct.unpack_from("<i", good, count_at)[0] == 3
    bad_count = bytearray(good)
    bad_count[count_at:count_at + 4] = struct.pack("<i", 2 ** 30)
    with pytest.raises(ValueError, match="implausible item count"):
        P.from_bytes(bytes(bad_count))
    bad_count[count_at:count_at + 4] = struct.pack("<i", -1)
    with pytest.raises(ValueError, match="implausible item count -1"):
        P.from_bytes(bytes(bad_count))
    bad_type = bytearray(good)
    bad_type[count_at + 4 + 32 + 16] = 9                                           # weight type of vertex 0 (1 extra UV)
    with pytest.raises(ValueError, match="vertex 0 has unknown weight type 9"):
        P.from_bytes(bytes(bad_type))
    bad_len = bytearray(good)
    bad_len[17:21] = struct.pack("<i", 10 ** 6)                                    # the model name length
    with pytest.raises(ValueError, match="bad text length"):
        P.from_bytes(bytes(bad_len))


def test_empty_or_garbage_input():
    for data in (b"", b"PMX ", b"PMX \x00\x00\x00@", bytes(40)):
        with pytest.raises(ValueError):
            P.from_bytes(data)
    assert P.from_bytes(bytearray(P.to_bytes(PmxModel(name="a")))).name == "a"
    assert P.from_bytes(memoryview(P.to_bytes(PmxModel(name="a")))).name == "a"
    with pytest.raises(TypeError, match="use read"):
        P.from_bytes("x.pmx")


# ------------------------------------------------------------------------------------------------ validation
def _assign(m, changes):
    """Set attribute paths such as 'vertices.1.bones' (digits index into lists) on the model."""
    for path, value in changes.items():
        obj = m
        *head, last = path.split(".")
        for part in head:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        if last.isdigit():
            obj[int(last)] = value
        else:
            setattr(obj, last, value)


NAN = float("nan")
MAT = {"morphs.0.kind": "material"}
BONE_MORPH = {"morphs.0.kind": "bone"}
# name, {attribute path: new value}, expected message (regex); every case starts from small()
CASES = [
    ("vertex bone index", {"vertices.1.bones": (7, 0)},
     r"vertex 1: bone index 7 is not a valid bone index \(the model has 3 bones"),
    ("vertex bone index below -1", {"vertices.0.bones": (-2,)}, r"bone index -2"),
    ("weights length", {"vertices.0.weights": (0.5, 0.5)}, r"vertex 0: weights has 2 entries but bones has 1"),
    ("three bones", {"vertices.0.bones": (0, 1, 2), "vertices.0.weights": (0.5, 0.25, 0.25)}, r"vertex 0: 3 bones"),
    ("five bones", {"vertices.0.bones": (0,) * 5, "vertices.0.weights": (0.2,) * 5}, r"vertex 0: 5 bones"),
    ("kind needs bones", {"vertices.0.kind": "BDEF2"}, r"vertex 0: kind BDEF2 needs 2 bones, got 1"),
    ("unknown kind", {"vertices.0.kind": "XDEF"}, r"vertex 0: unknown kind 'XDEF'"),
    ("sdef missing", {"vertices.1.kind": "SDEF"}, r"vertex 1: SDEF needs sdef"),
    ("sdef malformed", {"vertices.1.kind": "SDEF", "vertices.1.sdef": ((0, 0, 0), (1, 1, 1))},
     r"vertex 1: SDEF needs sdef"),
    ("sdef without kind", {"vertices.0.sdef": ((0, 0, 0),) * 3}, r"vertex 0: sdef data given but kind is BDEF1"),
    ("qdef in 2.0", {"vertices.2.kind": "QDEF"}, r"vertex 2: weight type QDEF needs PMX version 2.1"),
    ("additional uv count", {"add_uv_count": 1},
     r"vertex 0: has 0 additional UVs but the model declares add_uv_count=1"),
    ("additional uv shape", {"add_uv_count": 1, **{f"vertices.{i}.add_uv": ((0, 0, 0),) for i in range(3)}},
     r"vertex 0: additional uv needs 4 components, got 3"),
    ("additional uv channel", {"add_uv_count": 2, "vertices.0.add_uv": ((0.0,) * 4, (0.0,) * 3),
                               "vertices.1.add_uv": ((0.0,) * 4,) * 2, "vertices.2.add_uv": ((0.0,) * 4,) * 2},
     r"vertex 0: additional uv 1 needs 4 components, got 3"),
    ("additional uv nan", {"add_uv_count": 2, "vertices.0.add_uv": ((0.0,) * 4,) * 2,
                           "vertices.1.add_uv": ((0.0,) * 4, (NAN, 0.0, 0.0, 0.0)), "vertices.2.add_uv": ((0.0,) * 4,) * 2},
     r"vertex 1: additional uv 1 is not finite"),
    ("pos length", {"vertices.0.pos": (0.0, 0.0)}, r"vertex 0: pos needs 3 components, got 2"),
    ("uv length", {"vertices.2.uv": (0.0,)}, r"vertex 2: uv needs 2 components, got 1"),
    ("nan position", {"vertices.1.pos": (NAN, 0.0, 0.0)}, r"vertex 1: pos is not finite"),
    ("inf weight", {"vertices.1.weights": (float("inf"), 0.5)}, r"vertex 1: weight is not finite"),
    ("float bone index", {"vertices.0.bones": (0.5,)}, r"vertex bones must be ints and weights numbers"),
    ("string weight", {"vertices.0.weights": ("x",)}, r"vertex bones must be ints and weights numbers"),
    ("edge scale", {"vertices.0.edge_scale": NAN}, r"vertex 0: edge_scale is not finite"),
    ("faces not a multiple of 3", {"faces": [0, 1]}, r"holds 2 indices, which is not a multiple of 3"),
    ("face index range", {"faces": [0, 1, 9]},
     r"faces\[2\] \(triangle 0\): vertex index 9 out of range \(the model has 3 vertices\)"),
    ("negative face index", {"faces": [0, 1, -1]}, r"vertex index -1 out of range"),
    ("nested faces", {"faces": [[0, 1, 2]]}, r"not nested"),
    ("float faces", {"faces": [0.0, 1.0, 2.0]}, r"integer vertex indices"),
    ("material counts", {"materials.0.index_count": 0},
     r"material index counts sum to 0 but the model has 3 face indices"),
    ("material counts too many", {"materials.0.index_count": 6}, r"sum to 6 but the model has 3 face indices"),
    ("material count multiple of 3", {"materials.0.index_count": 2},
     r"material 0 'mat': index_count 2 must be a non-negative multiple of 3"),
    ("faces without materials", {"materials": []}, r"sum to 0 but the model has 3 face indices"),
    ("material texture", {"materials.0.texture": 1},
     r"material 0 'mat': texture 1 is not a valid texture index \(the model has 1 textures, -1 = none\)"),
    ("material sphere texture", {"materials.0.sphere_texture": -2}, r"sphere_texture -2"),
    ("material toon texture", {"materials.0.toon": 4}, r"toon 4 is not a valid texture index"),
    ("shared toon", {"materials.0.toon_shared": True, "materials.0.toon": 10},
     r"shared toon number 10 out of range 0..9"),
    ("sphere mode", {"materials.0.sphere_mode": 4}, r"sphere_mode 4 out of range 0..3"),
    ("material colour", {"materials.0.diffuse": (1.0, 1.0, 1.0)}, r"diffuse must be 4 finite numbers"),
    ("2.1 material flags", {"materials.0.line_draw": True}, r"need PMX version 2.1"),
    ("non-str texture", {"textures.0": 5}, r"texture 0: path must be a str"),
    ("bone parent", {"bones.1.parent": 9}, r"bone 1 'b1': parent 9 is not a valid bone index"),
    ("bone own parent", {"bones.1.parent": 1}, r"bone 1 'b1': is its own parent"),
    ("bone parent cycle", {"bones.0.parent": 2}, r"parent chain loops back"),
    ("bone tail", {"bones.1.tail_bone": 3}, r"tail_bone 3 is not a valid bone index"),
    ("bone tail offset", {"bones.0.tail_offset": (1.0, 2.0)}, r"tail_offset must be 3 finite numbers"),
    ("grant without parent", {"bones.1.grant_rotate": True}, r"grant_parent -1 is not a valid bone index"),
    ("grant from itself", {"bones.1.grant_move": True, "bones.1.grant_parent": 1}, r"grants from itself"),
    ("fixed axis", {"bones.1.fixed_axis": (1.0, 0.0)}, r"fixed_axis must be 3 finite numbers"),
    ("local axes one only", {"bones.1.local_x": (1.0, 0.0, 0.0)}, r"local_x and local_z must be set together"),
    ("ik target", {"bones.1.ik": PmxIK(9)}, r"IK target 9 is not a valid bone index"),
    ("ik link", {"bones.1.ik": PmxIK(0, links=[(9, False)])}, r"IK link 0: bone 9 is not a valid bone index"),
    ("ik link limit", {"bones.1.ik": PmxIK(0, links=[(1, True, (0.0, 0.0), Z)])},
     r"lower limit must be 3 finite numbers"),
    ("ik link shape", {"bones.1.ik": PmxIK(0, links=[5])}, r"IK link must be"),
    ("ik angle", {"bones.1.ik": PmxIK(0, angle=NAN)}, r"IK angle must be a finite number"),
    ("bone name", {"bones.2.name": None}, r"name must be a str"),
    ("morph kind", {"morphs.0.kind": "shape"}, r"morph 0 'vm': unknown morph kind 'shape'"),
    ("morph panel", {"morphs.0.panel": 5}, r"panel 5 out of range 0..4"),
    ("vertex morph index", {"morphs.0.offsets": [(3, Z)]},
     r"offset 0: vertex index 3 is not a valid vertex index"),
    ("vertex morph shape", {"morphs.0.offsets": [(0, (0.0, 0.0))]}, r"offset 0: value needs 3 components, got 2"),
    ("vertex morph pairs", {"morphs.0.offsets": [5]}, r"must be \(vertex index, values\) pairs"),
    ("vertex morph float index", {"morphs.0.offsets": [(0.5, Z)]}, r"must be \(vertex index, values\) pairs"),
    ("uv morph shape", {"morphs.0.kind": "uv", "morphs.0.offsets": [(0, Z)]}, r"value needs 4 components, got 3"),
    ("group morph index", {"morphs.1.offsets": [(2, 1.0)]},
     r"morph 1 'gm': offset 0: morph 2 is not a valid morph index"),
    ("group morph itself", {"morphs.1.offsets": [(1, 1.0)]}, r"cannot refer to itself"),
    ("group morph ratio", {"morphs.1.offsets": [(0, NAN)]}, r"ratio must be a finite number"),
    ("bone morph bone", {**BONE_MORPH, "morphs.0.offsets": [(3, Z, (0.0, 0.0, 0.0, 1.0))]},
     r"bone 3 is not a valid bone index"),
    ("bone morph quaternion", {**BONE_MORPH, "morphs.0.offsets": [(0, Z, Z)]},
     r"rotation quaternion must be 4 finite numbers"),
    ("material morph index", {**MAT, "morphs.0.offsets": [MaterialMorphOffset.of(1, 1)]},
     r"material 1 is not a valid material index"),
    ("material morph op", {**MAT, "morphs.0.offsets": [MaterialMorphOffset.of(0, 2)]}, r"op 2 out of range 0..1"),
    ("material morph form", {**MAT, "morphs.0.offsets": [(0, 1)]}, r"bad material morph offset"),
    ("material morph dict", {**MAT, "morphs.0.offsets": [{"index": 0}]}, r"needs at least 'index' and 'op'"),
    ("material morph field", {**MAT, "morphs.0.offsets": [{"index": 0, "op": 1, "colour": 1}]},
     r"unknown material morph offset field"),
    ("flip needs 2.1", {"morphs.1.kind": "flip"}, r"morph kind 'flip' needs PMX version 2.1"),
    ("impulse needs 2.1", {"morphs.0.kind": "impulse", "morphs.0.offsets": []},
     r"morph kind 'impulse' needs PMX version 2.1"),
    ("impulse body", {"version": 2.1, "morphs.0.kind": "impulse", "morphs.0.offsets": [(2, False, Z, Z)]},
     r"rigid body 2 is not a valid rigid body index"),
    ("frame item kind", {"frames.0.items": [("joint", 0)]},
     r"frame 0 'f': item 0 kind must be 'bone' or 'morph'"),
    ("frame item shape", {"frames.0.items": [5]}, r"item 0 must be \('bone' \| 'morph', index\)"),
    ("frame bone index", {"frames.0.items": [("bone", 3)]}, r"bone 3 is not a valid bone index"),
    ("frame morph index", {"frames.0.items": [("morph", 2)]}, r"morph 2 is not a valid morph index"),
    ("frame bone -1", {"frames.0.items": [("bone", -1)]}, r"bone -1 is not a valid bone index"),
    ("body bone", {"bodies.0.bone": 3}, r"rigid body 0 'b0': bone 3 is not a valid bone index"),
    ("body group", {"bodies.0.group": 16}, r"group 16 out of range 0..15"),
    ("body mask", {"bodies.1.mask": 0x10000}, r"mask 65536 out of range 0..65535"),
    ("body shape", {"bodies.0.shape": 3}, r"shape 3 out of range 0..2"),
    ("body mode", {"bodies.0.mode": 3}, r"mode 3 out of range 0..2"),
    ("body size", {"bodies.0.size": (1.0, 2.0)}, r"size must be 3 finite numbers"),
    ("body mass", {"bodies.0.mass": NAN}, r"mass must be a finite number"),
    ("joint body", {"joints.0.body_b": 2}, r"joint 0 'j': body_b 2 is not a valid rigid body index"),
    ("joint kind in 2.0", {"joints.0.kind": 2}, r"kind 2 out of range 0..0"),
    ("joint kind range", {"version": 2.1, "joints.0.kind": 6}, r"kind 6 out of range 0..5"),
    ("joint limit", {"joints.0.move_lo": (0.0, 0.0)}, r"move_lo must be 3 finite numbers"),
    ("version", {"version": 3.0}, r"unsupported PMX version 3.0"),
    ("encoding", {"encoding": "latin-1"}, r"unsupported text encoding 'latin-1'"),
    ("add_uv_count range", {"add_uv_count": 5}, r"add_uv_count 5 out of range 0..4"),
    ("model name", {"name": 5}, r"model: name must be a str"),
]


@pytest.mark.parametrize("name, changes, message", CASES, ids=[c[0] for c in CASES])
def test_validate_messages(name, changes, message):
    m = small()
    m.validate()                                        # the base model is valid
    _assign(m, changes)
    with pytest.raises(ValueError, match=message):
        m.validate()
    with pytest.raises(ValueError, match=message):       # writing refuses the same model with the same message
        P.to_bytes(m)


def test_validate_accepts_valid_models_and_special_frame_modes():
    for version in (2.0, 2.1):
        full_model(version).validate()
    m = small()
    m.validate()                                        # special frames are optional by default
    m.validate(require_special_frames=False)
    with pytest.raises(ValueError, match="exactly two special display frames"):
        m.validate(require_special_frames=True)
    m.frames = [PmxFrame("Root", special=True, items=[("bone", 0)]), PmxFrame("表情", special=True,
                items=[("morph", 0)]), PmxFrame("other")]
    m.validate(require_special_frames=True)
    m.frames[0], m.frames[2] = m.frames[2], m.frames[0]
    with pytest.raises(ValueError, match="first two frames"):
        m.validate(require_special_frames=True)


def test_validate_reports_the_first_offender_among_many():
    m = small()
    m.vertices = [copy.copy(v) for _ in range(100) for v in m.vertices]
    m.faces = [0, 1, 2] * 100
    m.materials[0].index_count = 300
    m.vertices[250].bones = (0, 1)
    m.vertices[250].weights = (1.0,)
    with pytest.raises(ValueError, match="vertex 250: weights has 1 entries but bones has 2"):
        m.validate()
    m.vertices[250].weights = (0.5, 0.5)
    m.vertices[17].pos = (0.0, 0.0, math.inf)
    with pytest.raises(ValueError, match="vertex 17: pos is not finite"):
        m.validate()


def test_validate_returns_none_and_does_not_modify():
    m = full_model(2.1)
    before = P.to_bytes(m)
    assert m.validate() is None
    assert P.to_bytes(m) == before
