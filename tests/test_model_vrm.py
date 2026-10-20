"""mk's own model file, VRM 1.0 (mkmmd.model.vrm): its rest pose (VRM's T-pose, the skin bound there), humanoid,
morph targets and expressions, springs and colliders from the PMX's physics, constraints from its helper bones, MToon
from its toon ramps, and what MK_rig keeps (the way back to the PMX's rest, the folds, the cloth copy). On the
mannequin (mkmmd/model/parts/mannequin.py), with a few bones and a fold added where a test needs them."""
import copy
import json
import struct
from itertools import pairwise

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from mkmmd.cli import main as MAIN
from mkmmd.core import rig as RG
from mkmmd.model import assemble as AS
from mkmmd.model import build as BD
from mkmmd.model import lab as LAB
from mkmmd.model import pmx_io as X
from mkmmd.model import spec as SP
from mkmmd.model import vrm as VRM
from mkmmd.model.parts import body_tex

MQ = '[model]\nname = "mq"\nparts = ["mannequin", "mannequin_hair", "mannequin_skirt"]\n'
AXES = VRM.AXES


@pytest.fixture(scope="module")
def mq(tmp_path_factory):
    """(folder, Assembled, unit) of the mannequin with its hair and skirt, as mk model build assembles it."""
    d = tmp_path_factory.mktemp("vrm_mq")
    (d / "model.toml").write_text(MQ, encoding="utf-8")
    spec = SP.load(d / "model.toml")
    unit = SP.model_cfg(spec)["scale"]
    asm = AS.assemble(BD.run(spec, tex_dir=d / AS.TEX_DIR), name="mq", scale=unit)
    return d, asm, unit


def write(mq, path, pmx=None, layers=None, cfg=None):
    """(report, glTF json, binary chunk, lab Model of the PMX) of a VRM written from the mannequin (or `pmx`)."""
    d, asm, unit = mq
    pmx = pmx or asm.pmx
    rep = VRM.write(path, pmx, d, unit, layers=asm.layers if layers is None else layers, cfg=cfg, name="mq")
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[12:16])[0]
    return rep, json.loads(raw[20:20 + n]), raw[28 + n:], LAB.from_pmx(pmx, d, unit)


def array(doc, blob, i):
    """Accessor i as an array (a sparse one with its values in place)."""
    a = doc["accessors"][i]
    comp = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
    k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    out = np.zeros((a["count"], k), comp)
    if "bufferView" in a:
        v = doc["bufferViews"][a["bufferView"]]
        out = np.frombuffer(blob, comp, a["count"] * k, v.get("byteOffset", 0)).reshape(-1, k).copy()
    if "sparse" in a:
        s = a["sparse"]
        iv, vv = (doc["bufferViews"][s[x]["bufferView"]] for x in ("indices", "values"))
        idx = np.frombuffer(blob, np.uint32, s["count"], iv.get("byteOffset", 0))
        out[idx] = np.frombuffer(blob, comp, s["count"] * k, vv.get("byteOffset", 0)).reshape(-1, k)
    return out


def world(doc, rotations=None):
    """{node: (rotation 3x3, position)} of the glTF nodes, with optional local rotations {node: 3x3} on their rest."""
    parent = {c: i for i, nd in enumerate(doc["nodes"]) for c in nd.get("children", [])}
    out = {}

    def at(i):
        if i not in out:
            p = parent.get(i)
            Rp, tp = (np.eye(3), np.zeros(3)) if p is None else at(p)
            out[i] = (Rp @ (rotations or {}).get(i, np.eye(3)), tp + Rp @ np.asarray(doc["nodes"][i].get("translation",
                                                                                                         [0, 0, 0])))
        return out[i]
    for i in range(len(doc["nodes"])):
        at(i)
    return out, parent


def test_the_rest_is_the_t_pose_and_the_skin_is_bound_there(mq, tmp_path):
    """The VRM's rest pose is the model in VRM's T-pose (lab.tpose_rotations, its own weights and grants): its vertices
    and its bones' heads are the posed ones, in glTF's axes (+Y up, facing +Z); no node keeps a rotation; each joint's
    inverse bind matrix undoes its rest place, so skinning at rest leaves the mesh as it is; every vertex's weights
    sum to 1."""
    _, doc, blob, M = write(mq, tmp_path / "mq.vrm")
    G = M.matrices(LAB.tpose_rotations(M))
    V, _ = M.skin(M.V, M.N, G)
    prim = doc["meshes"][0]["primitives"][0]["attributes"]
    assert np.abs(array(doc, blob, prim["POSITION"]) - V @ AXES.T).max() < 2e-6
    nodes = doc["nodes"]
    assert not any("rotation" in nd or "scale" in nd or "matrix" in nd for nd in nodes)
    pos, _ = world(doc)
    heads = np.einsum("bij,bj->bi", G[:, :3, :3], M.heads) + G[:, :3, 3]
    ik = {b for b, bone in enumerate(mq[1].pmx.bones) if bone.ik is not None}
    for b in range(len(M.bones)):
        if b not in ik:
            assert np.abs(pos[b][1] - AXES @ heads[b]).max() < 2e-5, M.bones[b]
    skin = doc["skins"][0]
    ibm = array(doc, blob, skin["inverseBindMatrices"]).reshape(-1, 4, 4)
    for k, j in enumerate(skin["joints"]):
        assert np.abs(ibm[k][3, :3] + pos[j][1]).max() < 2e-5 and np.allclose(ibm[k][:3, :3], np.eye(3))
    W = array(doc, blob, prim["WEIGHTS_0"])
    assert np.abs(W.sum(axis=1) - 1.0).max() < 1e-6


def test_the_humanoid_hangs_as_vrm_asks_from_hips_that_carry_the_spine_and_the_legs(mq, tmp_path):
    """Every humanoid bone VRM requires is there, each hangs from the humanoid bone VRM puts above it (VRM 1.0
    humanoid.md), the hips are the waist (腰: the spine and both legs hang from it), and the semantic bones land under
    VRM's names (the thumb's three bones from its metacarpal, the eyes). A bone that breaks VRM's order is left out,
    with the ones that need it: a thumb's first joint hung from the wrist, past its metacarpal."""
    pmx = copy.deepcopy(mq[1].pmx)
    names = [b.name for b in pmx.bones]
    pmx.bones[names.index("左親指１")].parent = names.index("左手首")
    for path, model in ((tmp_path / "mq.vrm", None), (tmp_path / "thumb.vrm", pmx)):
        rep, doc, _, M = write(mq, path, pmx=model)
        hb = {k: v["node"] for k, v in doc["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"].items()}
        for req in ("hips", "spine", "head", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg",
                    "rightLowerLeg", "rightFoot", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm",
                    "rightLowerArm", "rightHand"):
            assert req in hb, req
        _, parent = world(doc)

        def ancestors(i, parent=parent):
            out = set()
            while i in parent:
                i = parent[i]
                out.add(i)
            return out
        for k, node in hb.items():
            up = next((u for u in VRM._UP.get(k, []) if u is not True and u in hb), None)
            if up is not None:
                assert hb[up] in ancestors(node), k
        assert M.bones[hb["hips"]] == "腰"
        assert {M.bones[hb[k]] for k in ("leftUpperArm", "leftEye", "leftThumbMetacarpal", "rightIndexIntermediate")} \
            == {"左腕", "左目", "左親指０", "右人指２"}
        thumb = {"leftThumbProximal", "leftThumbDistal"} & set(hb)
        assert thumb == ({"leftThumbProximal", "leftThumbDistal"} if model is None else set()), thumb
    assert any("leftThumbProximal" in w for w in rep["warnings"])


def test_each_morph_target_is_its_morph_at_the_t_pose(mq, tmp_path):
    """A vertex morph's target moves each vertex as the morph does with the model in the T-pose: its PMX offset
    turned with the vertex by its weights' bones, so a morph keeps working once the rest is the T-pose (one on the
    forearm, which the T-pose raises, as well as the face's)."""
    pmx = copy.deepcopy(mq[1].pmx)
    M0 = LAB.from_pmx(pmx, mq[0], mq[2])
    fore = np.flatnonzero(M0.dominant == M0.bones.index("左ひじ"))
    assert len(fore) > 20
    pmx.morphs.append(X.PmxMorph(name="腕テスト", kind="vertex", offsets=[(int(i), (0.0, 0.1, 0.05)) for i in fore]))
    _, doc, blob, M = write(mq, tmp_path / "mq.vrm", pmx=pmx)
    names = doc["meshes"][0]["extras"]["targetNames"]
    targets = doc["meshes"][0]["primitives"][0]["targets"]
    rest, _ = M.deform("tpose")
    assert set(names) == {m.name for m in pmx.morphs if m.kind == "vertex"} and "腕テスト" in names
    for name, t in zip(names, targets):
        moved, _ = M.deform("tpose", {name: 1.0})
        assert np.abs(array(doc, blob, t["POSITION"]) - (moved - rest) @ AXES.T).max() < 1e-6, name


def test_expressions_are_mmds_standard_morphs_and_a_fold_shape_is_left_out(mq, tmp_path):
    """VRM's presets bind MMD's standard morphs (blink まばたき, blinkLeft ウィンク: her left eye, blinkRight
    ウィンク右, aa あ, happy 笑い); [vrm.expressions] replaces a preset's morphs; every other vertex morph is a custom
    expression of its own name, but a fold shape (rig layer `folds`) is in none: MK_rig keeps it with its driver."""
    _, asm, _ = mq
    fold = {"morph": "照れ", "piece": "shirt", "driver": {"kind": "angle", "bone": "左ひじ", "ref": "左腕",
                                                          "dir": [0.8, 0.0, -0.6], "from": 25.0, "to": 115.0}}
    cfg = SP.merge(VRM.DEFAULTS, {"expressions": {"happy": [["あ", 0.5]]}})
    _, doc, _, _ = write(mq, tmp_path / "mq.vrm", layers=dict(asm.layers, folds=[fold]), cfg=cfg)
    names = doc["meshes"][0]["extras"]["targetNames"]
    ex = doc["extensions"]["VRMC_vrm"]["expressions"]

    def bound(e):
        return {names[b["index"]]: b["weight"] for b in e["morphTargetBinds"]}
    p = ex["preset"]
    assert bound(p["blink"]) == {"まばたき": 1.0} and bound(p["blinkLeft"]) == {"ウィンク": 1.0}
    assert bound(p["blinkRight"]) == {"ウィンク右": 1.0} and bound(p["aa"]) == {"あ": 1.0}
    assert bound(p["happy"]) == {"あ": 0.5}
    assert set(ex["custom"]) == {"まばたき", "ウィンク", "ウィンク右", "あ", "笑い"}
    assert all("照れ" not in bound(e) for group in ex.values() for e in group.values())
    folds = doc["extensions"]["MK_rig"]["folds"]
    assert [names[f["target"]] for f in folds] == ["照れ"]


def test_springs_and_colliders_come_from_the_physics(mq, tmp_path):
    """Each bone a dynamic rigid body moves is a joint of exactly one spring, root first, the spring ending at a node
    on its last bone's tail (so the last bone swings too), each joint as wide as its body; its stiffness, gravity and
    drag are its family's. Each static body is a collider on its bone, in the T-pose: a capsule's two ends where the
    posed body's are."""
    _, doc, _, M = write(mq, tmp_path / "mq.vrm")
    _, asm, unit = mq
    pmx = asm.pmx
    G = M.matrices(LAB.tpose_rotations(M))
    sb = doc["extensions"]["VRMC_springBone"]
    dyn = {b.bone: b for b in pmx.bodies if b.mode != 0}
    joints = [j["node"] for s in sb["springs"] for j in s["joints"] if j["node"] < len(M.bones)]
    assert sorted(joints) == sorted(dyn)
    pos, parent = world(doc)
    for s in sb["springs"]:
        bones = [j["node"] for j in s["joints"][:-1]]
        assert all(parent[b] == a for a, b in pairwise(bones))
        end = s["joints"][-1]["node"]
        assert parent[end] == bones[-1] and "hitRadius" not in s["joints"][-1]
        last = bones[-1]
        tail = G[last][:3, :3] @ M.tails[last] + G[last][:3, 3]
        assert np.abs(pos[end][1] - AXES @ tail).max() < 2e-5
        fam = VRM.families.classify(*(M.bones[b] for b in bones[:3]))
        want = VRM.DEFAULTS["springs"][fam]
        for j in s["joints"][:-1]:
            assert abs(j["hitRadius"] - dyn[j["node"]].size[0] * unit) < 1e-5
            assert (j["stiffness"], j["gravityPower"], j["dragForce"]) == (want["stiffness"], want["gravity"],
                                                                          want["drag"])
    static = [b for b in pmx.bodies if b.mode == 0]
    assert len(sb["colliders"]) == len(static)
    for c, body in zip(sb["colliders"], static):
        assert c["node"] == body.bone
        if body.shape != 2:
            continue
        R = VRM._body_matrix(body.rot)
        centre = VRM.to_model(np.asarray(body.pos, float), unit)[0]
        ends = [centre + s * R[:, 2] * body.size[1] * unit / 2 for s in (-1.0, 1.0)]
        posed = [G[body.bone][:3, :3] @ e + G[body.bone][:3, 3] for e in ends]
        got = [pos[body.bone][1] + np.asarray(c["shape"]["capsule"][k]) for k in ("offset", "tail")]
        assert np.abs(np.array(got) - np.array(posed) @ AXES.T).max() < 1e-5


def test_helper_bones_follow_by_rotation_constraints(mq, tmp_path):
    """A PMX rotation grant onto a bone the humanoid does not drive is a rotation constraint from its granting bone
    with its ratio (a helper at the elbow); a grant from both eyes comes from the eye on the bone's side, which VRM's
    look-at turns (an eye highlight); none goes on a humanoid bone (the eyes, an elbow granted part of the arm's turn),
    and a moving grant is MK_rig's."""
    _, asm, _ = mq
    pmx = copy.deepcopy(asm.pmx)
    names = [b.name for b in pmx.bones]

    def add(name, parent, grant, ratio, move=False):
        p = pmx.bones[names.index(parent)]
        pmx.bones.append(X.PmxBone(name=name, pos=p.pos, parent=names.index(parent), tail_offset=(0.0, 0.1, 0.0),
                                   grant_rotate=not move, grant_move=move, grant_parent=names.index(grant),
                                   grant_ratio=ratio))
        names.append(name)
    add("左ひじ補助", "左腕", "左ひじ", 0.5)
    add("左ハイライト", "頭", "両目", 0.6)
    add("右ハイライト", "頭", "両目", 0.6)
    add("左袖動", "左腕", "左腕", 0.5, move=True)
    elbow = pmx.bones[names.index("右ひじ")]
    elbow.grant_rotate, elbow.grant_parent, elbow.grant_ratio = True, names.index("右腕"), 0.3
    _, doc, _, _ = write(mq, tmp_path / "mq.vrm", pmx=pmx)
    cons = {names[i]: nd["extensions"]["VRMC_node_constraint"]["constraint"]["rotation"]
            for i, nd in enumerate(doc["nodes"][:len(names)]) if "VRMC_node_constraint" in nd.get("extensions", {})}
    assert {k: (names[v["source"]], v["weight"]) for k, v in cons.items()} == {
        "左ひじ補助": ("左ひじ", 0.5), "左ハイライト": ("左目", 0.6), "右ハイライト": ("右目", 0.6)}
    kept = {names[g["node"]]: g["why"] for g in doc["extensions"]["MK_rig"]["grants"] if g["why"]}
    assert set(kept) == {"左目", "右目", "右ひじ", "左袖動"}


def test_mtoon_takes_the_toon_ramps_step_and_shade(mq, tmp_path):
    """A material's MToon shade is its lit colour times the toon ramp's darkest row, and its step is where the ramp's
    is: MMD reads the ramp at 0.5 + 0.5 N.L, so a step at 0.4 up the ramp, 0.25 wide, lights down to N.L = -0.2
    (shadingShift 0.2) over a soft band (shadingToony 0.81). A material without a toon ramp is flat (its shade is its
    lit colour), and the edge is the outline in its colour, unlit."""
    d, asm, _ = mq
    pmx = copy.deepcopy(asm.pmx)
    ramp = body_tex.toon_ramp(None, (0.8, 0.6, 0.5), edge=0.4, soft=0.25)
    Image.fromarray(ramp).save(d / AS.TEX_DIR / "test_ramp.png")
    pmx.textures.append(f"{AS.TEX_DIR}\\test_ramp.png")
    skin = pmx.materials[0]
    skin.toon, skin.toon_shared, skin.edge, skin.edge_size = len(pmx.textures) - 1, False, True, 0.7
    flat = next(m for m in pmx.materials if not m.toon_shared and m.toon < 0)
    _, doc, _, _ = write(mq, tmp_path / "mq.vrm", pmx=pmx)
    mats = {m["name"]: m for m in doc["materials"]}
    mt = mats[skin.name]["extensions"]["VRMC_materials_mtoon"]
    assert abs(mt["shadingShiftFactor"] - 0.2) < 0.02 and abs(mt["shadingToonyFactor"] - 0.81) < 0.02
    lit = np.clip(np.asarray(skin.ambient) + VRM.MMD_LIGHT * np.asarray(skin.diffuse[:3]), 0, 1)
    assert np.abs(np.asarray(mt["shadeColorFactor"]) - VRM._lin(lit * (0.8, 0.6, 0.5))).max() < 0.01
    assert mt["outlineWidthMode"] == "screenCoordinates" and mt["outlineLightingMixFactor"] == 0.0
    assert abs(mt["outlineWidthFactor"] - 0.7 * VRM.DEFAULTS["outline"]) < 1e-9
    assert np.abs(np.asarray(mt["outlineColorFactor"]) - VRM._lin(skin.edge_color[:3])).max() < 1e-6
    fm = mats[flat.name]
    assert fm["extensions"]["VRMC_materials_mtoon"]["shadeColorFactor"] == fm["pbrMetallicRoughness"][
        "baseColorFactor"][:3]


def test_mk_rig_turns_the_vrm_back_to_the_pmx_rest_and_keeps_the_cloth_on_her(mq, tmp_path):
    """MK_rig's rest turns, set on the VRM's nodes, put every bone back where the PMX's rest pose has it (an IK bone
    aside, which sits on its target); a cloth copy's points are skinned into the T-pose with their own weights, so they
    lie where the posed garment is (the skirt, and a sleeve's on the forearm the T-pose raises); a fold's driver
    direction turns with its reference bone, so the angle it measures at rest is the same."""
    _, asm, _ = mq
    fold = {"morph": "照れ", "piece": "shirt", "driver": {"kind": "angle", "bone": "左ひじ", "ref": "左腕",
                                                          "dir": [0.6, 0.0, -0.8], "from": 25.0, "to": 115.0}}
    _, doc, blob, M = write(mq, tmp_path / "mq.vrm", layers=dict(asm.layers, folds=[fold]))
    mk = doc["extensions"]["MK_rig"]
    rot = {e["node"]: Rotation.from_quat(e["rotation"]).as_matrix() for e in mk["rest"]["pmx_rest"]}
    pos, _ = world(doc, rot)
    ik = {b for b, bone in enumerate(asm.pmx.bones) if bone.ik is not None}
    off = max(float(np.abs(pos[b][1] - AXES @ M.heads[b]).max()) for b in range(len(M.bones)) if b not in ik)
    assert off < 2e-5
    G = M.matrices(LAB.tpose_rotations(M))
    arm = M.bones.index("左ひじ")
    fore = np.flatnonzero(M.dominant == arm)[:40]
    sleeve = {"name": "sleeve", "materials": [asm.pmx.materials[0].name], "bones": ["左ひじ"],
              "points": {"sim": M.V[fore], "collide": M.V[fore]}, "sim_faces": np.zeros((1, 4), np.int32),
              "pins": np.zeros(len(fore), np.float32), "holds": np.zeros(len(fore), np.float32),
              "collides": np.ones(len(fore), np.uint8), "sim_weights": np.ones((len(fore), 1), np.float32),
              "collide_faces": np.zeros((1, 3), np.int32), "collide_weights": np.ones((len(fore), 1), np.float32),
              "settings": {}}
    _, doc, blob, _ = write(mq, tmp_path / "cloth.vrm", layers=dict(asm.layers, cloth=asm.layers["cloth"] + [sleeve]))
    for cl, out in zip(asm.layers["cloth"] + [sleeve], doc["extensions"]["MK_rig"]["cloth"]):
        sim = out["arrays"]["sim"]
        v = doc["bufferViews"][sim["bufferView"]]
        got = np.frombuffer(blob, np.float32, int(np.prod(sim["shape"])), v.get("byteOffset", 0)).reshape(sim["shape"])
        bones = [M.bones.index(b) for b in cl["bones"]]
        W = np.asarray(cl["sim_weights"], float)
        W = W / W.sum(axis=1, keepdims=True)
        want = np.einsum("nb,bij,nj->ni", W, G[bones][:, :3, :3], cl["points"]["sim"]) + W @ G[bones][:, :3, 3]
        assert np.abs(got - want @ AXES.T).max() < 1e-5, cl["name"]
    assert np.abs(G[arm][:3, :3] - np.eye(3)).max() > 0.1                  # the T-pose turns the forearm
    f = mk["folds"][0]["driver"]
    e, a = M.bones.index("左ひじ"), M.bones.index("左腕")
    bone_dir = LAB.unit(M.tails[e] - M.heads[e])
    at_rest = np.degrees(np.arccos(bone_dir @ LAB.unit(fold["driver"]["dir"])))
    t_dir = AXES @ (G[e][:3, :3] @ bone_dir)
    in_vrm = np.degrees(np.arccos(LAB.unit(t_dir) @ LAB.unit(f["dir"])))
    assert f["bone"] == e and f["ref"] == a and abs(in_vrm - at_rest) < 1e-3


def test_vrm_keys_are_checked_and_not_reported_unread(tmp_path):
    """mk model build reads [vrm] itself, so the parts' unread-key report leaves it alone, and a key [vrm] does not
    have fails at once instead of passing for one that works."""
    (tmp_path / "model.toml").write_text(MQ + '\n[vrm.meta]\nauthors = ["me"]\n[vrm.springs.hair]\nstiffness = 2.0\n',
                                         encoding="utf-8")
    spec = SP.load(tmp_path / "model.toml")
    cfg = VRM.config(spec)
    assert cfg["meta"]["authors"] == ["me"] and cfg["springs"]["hair"]["stiffness"] == 2.0
    assert cfg["springs"]["hair"]["drag"] == VRM.DEFAULTS["springs"]["hair"]["drag"]
    ctx = BD.BuildCtx(SP.load(tmp_path / "model.toml"), tmp_path / "tex", seed=1)
    BD.run(ctx.spec, ctx=ctx)
    assert not [w for w in ctx.warnings if "vrm" in w]
    for bad in ('[vrm]\noutlines = 0.002\n', '[vrm.meta]\nauthor = ["me"]\n', '[vrm.springs.hair]\nstifness = 1.0\n',
                '[vrm.expressions]\nsmile = [["笑い", 1.0]]\n'):
        (tmp_path / "bad.toml").write_text(MQ + bad, encoding="utf-8")
        with pytest.raises(SP.SpecError):
            VRM.config(SP.load(tmp_path / "bad.toml"))


def test_mk_model_vrm_writes_beside_a_pmx_with_its_rig_json_layers(mq, tmp_path, capsys):
    """`mk model vrm X.pmx` writes X.vrm beside it and takes X.rig.json's model layers (bones by Blender name, the
    cloth copy's arrays in X.cloth.npz) back to the PMX's names, so the VRM keeps the cloth copy and the folds."""
    d, asm, _ = mq
    pmx_path = d / "mq.pmx"
    X.write(asm.pmx, pmx_path)
    fold = {"morph": "照れ", "piece": "shirt", "driver": {"kind": "angle", "bone": "左ひじ", "ref": "左腕",
                                                          "dir": [0.6, 0.0, -0.8], "from": 25.0, "to": 115.0}}
    blender = {f"{b.name[1:]}.L" if b.name.startswith("左") else f"{b.name[1:]}.R" if b.name.startswith("右")
               else b.name: {"jp": b.name} for b in asm.pmx.bones}
    rig = RG.merge({"bones": blender}, dict(asm.layers, folds=[fold]))
    rig, cloth = RG.cloth_split(rig, "mq.cloth.npz")
    np.savez_compressed(d / "mq.cloth.npz", **cloth)
    (d / "mq.rig.json").write_text(json.dumps(rig, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as code:
        MAIN.main(["model", "vrm", str(pmx_path)])
    rep = json.loads(capsys.readouterr().out)
    assert code.value.code == 0
    raw = (d / "mq.vrm").read_bytes()
    doc = json.loads(raw[20:20 + struct.unpack("<I", raw[12:16])[0]])
    mk = doc["extensions"]["MK_rig"]
    names = [b.name for b in asm.pmx.bones]
    assert rep["vrm"] == str(d / "mq.vrm") and [c["name"] for c in mk["cloth"]] == [asm.layers["cloth"][0]["name"]]
    assert [names[n] for n in mk["cloth"][0]["nodes"]] == asm.layers["cloth"][0]["bones"]
    assert mk["folds"][0]["morph"] == "照れ" and names[mk["folds"][0]["driver"]["bone"]] == "左ひじ"
