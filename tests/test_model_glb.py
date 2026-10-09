"""mk model glb (mkmmd.model.glb): a model posed with its own weights, written as one binary glTF with its textures, as
Tern's 3D block and other glTF viewers read it."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd.model import glb as GLB
from mkmmd.model import lab as LAB
from mkmmd.model import pmx_io as X


def read_glb(path):
    """(the JSON document, an accessor reader) of a .glb, its header and both chunks checked."""
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack("<4sII", data[:12])
    assert (magic, version, length) == (b"glTF", 2, len(data))
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    assert jtype == b"JSON" and jlen % 4 == 0
    doc = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack("<I4s", data[20 + jlen:28 + jlen])
    assert btype == b"BIN\0" and 28 + jlen + blen == len(data) and blen == doc["buffers"][0]["byteLength"]
    blob = data[28 + jlen:]

    def acc(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        dtype = {5126: np.float32, 5125: np.uint32}[a["componentType"]]
        return np.frombuffer(blob, dtype, a["count"] * n, v["byteOffset"]).reshape(a["count"], n)
    return doc, acc


def cube(c, h):
    """Corners (model space), outward normals and the 12 triangles of a cube, wound clockwise seen from outside, as PMX
    files keep them."""
    P = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + np.asarray(c, float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = []
    for a, b, cc, d in quads:
        for t in ((a, b, cc), (a, cc, d)):
            n = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
            out = P[list(t)].mean(axis=0) - np.asarray(c, float)
            F.append(t if n @ out < 0 else (t[0], t[2], t[1]))       # clockwise from outside
    return P, F, (P - np.asarray(c, float)) / np.linalg.norm(P - np.asarray(c, float), axis=1, keepdims=True)


def textured_model(tmp_path):
    Image.new("RGB", (4, 4), (200, 120, 90)).save(tmp_path / "opaque.png")
    clear = Image.new("RGBA", (4, 4), (90, 120, 200, 255))
    clear.putpixel((1, 1), (90, 120, 200, 0))                            # one see-through texel
    clear.save(tmp_path / "clear.png")
    mats = [dict(texture=0, diffuse=(1.0, 1.0, 1.0, 1.0)),                # opaque texture
            dict(texture=1, diffuse=(1.0, 1.0, 1.0, 1.0), double_sided=True),   # texture with a hole
            dict(texture=-1, diffuse=(0.2, 0.4, 0.6, 0.5)),               # half see-through colour
            dict(texture=-1, diffuse=(1.0, 0.0, 0.0, 0.0)),               # hidden: diffuse alpha 0
            dict(texture=0, diffuse=(1.0, 1.0, 1.0, 1.0))]                # the first texture again
    verts, faces, pm = [], [], []
    for i, m in enumerate(mats):
        P, F, N = cube((3.0 * i, 0.0, 1.0), 0.5)
        base = len(verts)
        for k, (p, n) in enumerate(zip(P, N)):
            verts.append(X.PmxVertex(pos=(p[0], p[2], p[1]), normal=(n[0], n[2], n[1]), uv=(k / 8.0, 1.0 - k / 8.0),
                                     bones=(0,), weights=(1.0,)))
        faces += [base + j for f in F for j in f]
        pm.append(X.PmxMaterial(name=f"m{i}", index_count=3 * len(F), **m))
    pmx = X.PmxModel(name="cubes", vertices=verts, faces=faces, textures=["opaque.png", "clear.png"], materials=pm,
                     bones=[X.PmxBone("root")])
    return LAB.from_pmx(pmx, root=tmp_path, unit=1.0)


def test_glb_turns_model_space_upright_and_keeps_every_face_facing_out(tmp_path):
    m = textured_model(tmp_path)
    V, N = m.deform()
    GLB.write(tmp_path / "m.glb", m, V, N)
    doc, acc = read_glb(tmp_path / "m.glb")
    prim = doc["meshes"][0]["primitives"]
    P, Nr = acc(prim[0]["attributes"]["POSITION"]), acc(prim[0]["attributes"]["NORMAL"])
    drawn = np.unique(m.T[~m.hidden[m.tri_mat]])                                # the hidden cube is not written
    Vd = V[drawn]
    assert np.allclose(P, np.c_[Vd[:, 0], Vd[:, 2], -Vd[:, 1]], atol=1e-6)     # model Z up -> glTF Y up, -Y fwd -> +Z
    assert np.allclose(acc(prim[0]["attributes"]["TEXCOORD_0"]), m.uv[drawn], atol=1e-6)   # v runs down in both
    for p in prim:
        T = acc(p["indices"]).reshape(-1, 3)
        face = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        assert (np.einsum("ij,ij->i", face, Nr[T].sum(axis=1)) > 0).all()      # counter-clockwise from outside


def test_glb_carries_each_material_textures_once_and_blends_only_what_is_see_through(tmp_path):
    m = textured_model(tmp_path)
    GLB.write(tmp_path / "m.glb", m, *m.deform())
    doc, _ = read_glb(tmp_path / "m.glb")
    mats = {x["name"]: x for x in doc["materials"]}
    assert set(mats) == {"m0", "m1", "m2", "m4"}                                 # the hidden one is left out
    assert len(doc["images"]) == 2 and all(i["mimeType"] == "image/png" for i in doc["images"])
    tex = {n: mats[n]["pbrMetallicRoughness"].get("baseColorTexture", {}).get("index") for n in mats}
    assert tex["m0"] == tex["m4"] != tex["m1"] and tex["m2"] is None
    assert {n: mats[n]["alphaMode"] for n in mats} == {"m0": "OPAQUE", "m1": "BLEND", "m2": "BLEND", "m4": "OPAQUE"}
    assert mats["m1"]["doubleSided"] and not mats["m0"]["doubleSided"]
    assert mats["m2"]["pbrMetallicRoughness"]["baseColorFactor"] == [0.2, 0.4, 0.6, 0.5]


def test_a_region_holds_only_its_own_vertices_so_a_viewer_frames_it(tmp_path):
    m = textured_model(tmp_path)
    GLB.write(tmp_path / "part.glb", m, *m.deform(), keep=m.tri_mat == 1)          # the second cube alone
    doc, _ = read_glb(tmp_path / "part.glb")
    prims = doc["meshes"][0]["primitives"]
    pos = doc["accessors"][prims[0]["attributes"]["POSITION"]]
    assert len(prims) == 1 and doc["materials"][0]["name"] == "m1" and pos["count"] == 8
    assert pos["min"] == pytest.approx([2.5, 0.5, -0.5]) and pos["max"] == pytest.approx([3.5, 1.5, 0.5])
