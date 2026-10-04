"""mkmmd.model.skeleton: the MMD standard skeleton from landmarks."""
import numpy as np
import pytest

from mkmmd.core import bonemap, families
from mkmmd.model import part as P
from mkmmd.model import skeleton as SK


def landmarks(side_only=True):
    L = {
        "upper_body": (0, 0.0, 0.93), "neck": (0, 0.0, 1.22), "head": (0, 0, 1.27), "head_tip": (0, 0, 1.58),
        "shoulder.L": (0.05, 0, 1.18), "arm.L": (0.09, 0, 1.18), "elbow.L": (0.27, 0.01, 1.07),
        "wrist.L": (0.41, 0.0, 0.99), "leg.L": (0.08, 0, 0.78), "knee.L": (0.055, -0.03, 0.44),
        "ankle.L": (0.063, 0, 0.098), "toe.L": (0.06, -0.15, 0.03),
    }
    base = {"thumb": (0.43, -0.02, 0.965), "index": (0.474, -0.014, 0.947), "middle": (0.478, 0.007, 0.949),
            "ring": (0.475, 0.026, 0.948), "little": (0.467, 0.043, 0.947)}
    for f, b in base.items():
        for i, n in enumerate(SK.finger_joint_names(f)):
            L[f"{n}.L"] = tuple(np.array(b) + np.array([0.02 * i, 0, -0.015 * i]))
    return L


def by_name(bones):
    return {b.name: b for b in bones}


def test_required_landmarks_and_error_message():
    need = SK.required_landmarks()
    assert "head_tip" in need and "thumb0.L" in need and "little_tip.L" in need and "toe.L" in need
    assert "thumb0.L" not in SK.required_landmarks({"fingers": False})
    L = landmarks()
    del L["knee.L"]
    with pytest.raises(KeyError, match="knee"):
        SK.standard_bones(L)
    with pytest.raises(KeyError):
        SK.standard_bones(landmarks(), {"nonsense": 1})


def test_standard_names_hierarchy_and_order():
    bones = SK.standard_bones(landmarks())
    b = by_name(bones)
    names = [x.name for x in bones]
    assert len(names) == len(set(names)) == 73            # 8 trunk + 2*(6 arm + 20 finger) + 9 lower + 4 IK
    for must in ("全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2", "首", "頭", "下半身", "左肩", "右肩", "左腕",
                 "左腕捩", "左ひじ", "左手捩", "左手首", "左親指０", "左親指２", "左親指先", "左人指１", "左中指３", "左薬指２",
                 "左小指１", "右小指先", "左足", "左ひざ", "左足首", "左つま先", "左足ＩＫ", "左つま先ＩＫ", "右つま先ＩＫ"):
        assert must in b, must
    parents = {n: b[n].parent for n in names}
    assert parents["全ての親"] == "" and parents["センター"] == "全ての親" and parents["グルーブ"] == "センター"
    assert parents["腰"] == "グルーブ" and parents["上半身"] == "腰" and parents["下半身"] == "腰"
    assert parents["上半身2"] == "上半身" and parents["首"] == "上半身2" and parents["頭"] == "首"
    assert parents["左肩"] == "上半身2" and parents["左腕"] == "左肩" and parents["左腕捩"] == "左腕"
    assert parents["左ひじ"] == "左腕捩" and parents["左手捩"] == "左ひじ" and parents["左手首"] == "左手捩"
    assert parents["左人指１"] == "左手首" and parents["左人指２"] == "左人指１" and parents["左人指先"] == "左人指３"
    assert parents["左親指０"] == "左手首" and parents["左親指先"] == "左親指２"
    assert parents["左足"] == "下半身" and parents["左ひざ"] == "左足" and parents["左足首"] == "左ひざ"
    assert parents["左足ＩＫ"] == "全ての親" and parents["左つま先ＩＫ"] == "左足ＩＫ"
    pos = {n: i for i, n in enumerate(names)}
    for n in names:                                       # parents (and grant parents) always come first
        if parents[n]:
            assert pos[parents[n]] < pos[n], n
        g = b[n].grant
        if g:
            assert pos[g["parent"]] < pos[n], n
    P.check(P.Part("body", bones=bones))


def test_every_required_semantic_bone_maps_through_bonemap():
    bones = SK.standard_bones(landmarks())
    m = bonemap.build_map({b.name: b.name for b in bones})
    assert [s for s in bonemap.REQUIRED if s not in m] == []
    for s in ("root", "groove", "waist", "upper_body2", "arm_twist.L", "wrist_twist.R", "toe.L", "toe_ik.R", "thumb0.L",
              "index_tip.R", "little3.L", "leg_ik.L"):
        assert s in m, s
    for b in bones:
        assert b.semantic, b.name                            # every standard bone has its semantic name
        assert m[b.semantic] == b.name
    # also through mmd_tools' renamed form: 左腕 -> 腕.L, name_j keeps the PMX name
    renamed = {b.name.replace("左", "") + ".L" if b.name.startswith("左") else b.name: b.name for b in bones}
    assert [s for s in bonemap.REQUIRED if s not in bonemap.build_map(renamed)] == []


def test_right_side_mirrors_left_unless_given():
    L = landmarks()
    b = by_name(SK.standard_bones(L))
    for n in ("腕", "ひじ", "手首", "人指２", "足", "足首", "つま先"):
        l, r = np.array(b["左" + n].head), np.array(b["右" + n].head)
        assert np.allclose(r, l * [-1, 1, 1]), n
    L["wrist.R"] = (-0.45, 0.0, 0.95)
    b2 = by_name(SK.standard_bones(L))
    assert np.allclose(b2["右手首"].head, (-0.45, 0, 0.95)) and np.allclose(b2["左手首"].head, (0.41, 0, 0.99))


def test_ik_chain_and_knee_limit():
    b = by_name(SK.standard_bones(landmarks(), {"ik_loops": 40}))
    ik = b["左足ＩＫ"]
    assert ik.movable and ik.ik["target"] == "左足首" and ik.ik["iterations"] == 40
    assert [c["bone"] for c in ik.ik["chain"]] == ["左ひざ", "左足"]
    assert ik.ik["chain"][0]["limit"] == [[-180.0, 0.0, 0.0], [-0.5, 0.0, 0.0]] and ik.ik["chain"][1]["limit"] is None
    assert np.isclose(ik.ik["angle"], 114.5916)
    assert np.allclose(ik.head, b["左足首"].head)            # IK bone sits on the ankle: zero error at rest
    toe = b["左つま先ＩＫ"]
    assert toe.layer == 1 and toe.ik["target"] == "左つま先" and [c["bone"] for c in toe.ik["chain"]] == ["左足首"]
    # the IK target is a child of the first link, links follow the parent chain (what mmd_tools requires)
    assert b[ik.ik["target"]].parent == ik.ik["chain"][0]["bone"]
    assert b[ik.ik["chain"][0]["bone"]].parent == ik.ik["chain"][1]["bone"]
    assert b[toe.ik["target"]].parent == toe.ik["chain"][0]["bone"]


def test_twist_bones_fixed_axis_and_split_grants():
    b = by_name(SK.standard_bones(landmarks()))
    arm, elbow, wrist = (np.array(b[n].head) for n in ("左腕", "左ひじ", "左手首"))
    ax = np.array(b["左腕捩"].fixed_axis)
    assert np.allclose(ax, (elbow - arm) / np.linalg.norm(elbow - arm)) and np.isclose(np.linalg.norm(ax), 1)
    assert b["左腕捩"].tail_bone == "左ひじ" and b["左手捩"].tail_bone == "左手首"
    t = np.array(b["左腕捩"].head)
    assert 0.5 < np.linalg.norm(t - arm) / np.linalg.norm(elbow - arm) < 0.7
    assert "左腕捩1" not in b
    s = by_name(SK.standard_bones(landmarks(), {"twist": "split"}))
    for i, ratio in ((1, 0.25), (2, 0.5), (3, 0.75)):
        g = s[f"左腕捩{i}"].grant
        assert g == {"parent": "左腕捩", "rotate": True, "move": False, "ratio": ratio}
        assert s[f"右手捩{i}"].grant["parent"] == "右手捩" and not s[f"右手捩{i}"].visible
    n = by_name(SK.standard_bones(landmarks(), {"twist": "none"}))
    assert "左腕捩" not in n and n["左ひじ"].parent == "左腕" and n["左手首"].parent == "左ひじ"


def test_local_axes_and_finger_tips():
    b = by_name(SK.standard_bones(landmarks()))
    for side, sx in (("左", 1), ("右", -1)):
        arm = b[side + "腕"]
        x, z = np.array(arm.local_x), np.array(arm.local_z)
        assert np.isclose(np.linalg.norm(x), 1) and np.isclose(np.linalg.norm(z), 1) and abs(x @ z) < 1e-9
        assert x[0] * sx > 0.5 and z[1] < -0.9                # along the arm (outwards), z towards the front (-Y)
    tip = b["左人指先"]
    assert not tip.visible and tip.tail is not None and tip.local_x is None
    assert b["左人指１"].tail_bone == "左人指２" and b["左手首"].tail_bone == "左中指１"


def test_options_switch_bones_on_and_off():
    full = SK.standard_bones(landmarks())
    minimal = SK.standard_bones(landmarks(), {"groove": False, "waist": False, "upper_body2": False, "fingers": False,
                                              "toes": False, "leg_ik": False, "twist": "none"})
    names = {b.name for b in minimal}
    assert not names & {"グルーブ", "腰", "上半身2", "左人指１", "左つま先", "左足ＩＫ", "左腕捩"}
    b = by_name(minimal)
    assert b["上半身"].parent == "センター" and b["首"].parent == "上半身" and b["下半身"].parent == "センター"
    assert b["左足首"].tail_bone == "" and b["左足首"].tail is not None
    assert len(minimal) < len(full) - 40
    m = bonemap.build_map({x.name: x.name for x in minimal})
    assert [s for s in bonemap.REQUIRED if s not in m and not s.startswith("leg_ik")] == []
    sp = by_name(SK.standard_bones(landmarks(), {"shoulder_p": True}))
    assert sp["左肩"].parent == "左肩P" and sp["左腕"].parent == "左肩C" and sp["左肩C"].grant["ratio"] == -1.0


def test_eyes_only_with_eye_landmarks():
    assert "左目" not in by_name(SK.standard_bones(landmarks()))
    L = landmarks()
    L["eye.L"] = (0.031, -0.044, 1.40)
    b = by_name(SK.standard_bones(L))
    assert b["両目"].parent == "頭" and b["両目"].movable
    assert b["左目"].grant == {"parent": "両目", "rotate": True, "move": False, "ratio": 1.0} and b["左目"].layer == 2
    assert np.allclose(b["右目"].head, (-0.031, -0.044, 1.40))
    assert b["両目"].head[2] > b["左目"].head[2]
    with pytest.raises(KeyError):
        SK.standard_bones(landmarks(), {"eyes": True})


def test_defaults_derive_missing_optional_landmarks():
    c = SK.complete(landmarks())
    for k in ("center", "groove", "lower_body", "waist", "upper_body2", "arm_twist.L", "wrist_twist.R", "leg_ik.L",
              "toe_end.R", "root"):
        assert k in c, k
    assert c["center"][2] < c["lower_body"][2] < 1.0 and c["upper_body"][2] < c["upper_body2"][2] < c["neck"][2]
    L = landmarks()
    L["center"] = (0, 0, 0.8)
    L["upper_body2"] = (0, 0.01, 1.05)
    b = by_name(SK.standard_bones(L))
    assert np.allclose(b["センター"].head, (0, 0, 0.8)) and np.allclose(b["上半身2"].head, (0, 0.01, 1.05))


def test_frames_cover_the_standard_bones():
    bones = SK.standard_bones(landmarks())
    fr = SK.standard_frames(bones)
    assert set(fr) == {"ＩＫ", "体(上)", "体(下)", "腕", "指", "足"}
    flat = [n for v in fr.values() for n in v]
    assert len(flat) == len(set(flat)) and "左足ＩＫ" in fr["ＩＫ"] and "左人指２" in fr["指"] and "左人指先" not in flat
    names = {b.name for b in bones}
    assert set(flat) <= names


def test_chain_names_classify_by_families_rules():
    # the convention in docs: bangs 前髪*, braids 三つ編*, ears 猫耳*, tails 尻尾*, skirt スカート*, ribbons リボン*
    for name, fam in (("前髪1", "bangs"), ("三つ編左1", "braid"), ("猫耳右2", "ears"), ("尻尾1_2", "tail"),
                      ("スカート前1", "skirt"), ("リボン左1", "ribbon"), ("袖左1", "sleeve"), ("襟1", "coat"),
                      ("横髪左1", "side_hair"), ("後髪3", "back_hair")):
        assert families.classify(name) == fam, name
