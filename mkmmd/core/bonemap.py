"""Standard MMD bone names -> semantic names.

MMD models share a standard skeleton named in Japanese (PMX names such as 左腕). mmd_tools renames bones on import
(左腕 -> 腕.L) and keeps the original name in `pose_bone.mmd_bone.name_j`. Matching uses the original name,
NFKC-normalised, so full-width digits and letters match half-width ones (人指１ == 人指1, 足ＩＫ == 足IK)."""
import unicodedata

CENTER = {
    "root": ["全ての親"],
    "view_center": ["操作中心"],
    "center": ["センター"],
    "groove": ["グルーブ"],
    "waist": ["腰"],
    "lower_body": ["下半身"],
    "upper_body": ["上半身"],
    "upper_body2": ["上半身2"],
    "upper_body3": ["上半身3"],
    "neck": ["首"],
    "head": ["頭"],
    "eyes": ["両目"],
}

# stems, expanded with the 左/右 prefix (or suffix) into `<stem>.L` / `<stem>.R`
SIDED = {
    "eye": ["目"],
    "shoulder_p": ["肩P"],
    "shoulder": ["肩"],
    "shoulder_c": ["肩C"],
    "arm": ["腕"],
    "arm_twist": ["腕捩"],
    "arm_twist1": ["腕捩1"],
    "arm_twist2": ["腕捩2"],
    "arm_twist3": ["腕捩3"],
    "elbow": ["ひじ", "肘"],
    "wrist_twist": ["手捩"],
    "wrist_twist1": ["手捩1"],
    "wrist_twist2": ["手捩2"],
    "wrist_twist3": ["手捩3"],
    "wrist": ["手首"],
    "thumb0": ["親指0"],
    "thumb1": ["親指1"],
    "thumb2": ["親指2"],
    "thumb_tip": ["親指先"],
    "index1": ["人指1", "人差指1"],
    "index2": ["人指2", "人差指2"],
    "index3": ["人指3", "人差指3"],
    "index_tip": ["人指先", "人差指先"],
    "middle1": ["中指1"],
    "middle2": ["中指2"],
    "middle3": ["中指3"],
    "middle_tip": ["中指先"],
    "ring1": ["薬指1"],
    "ring2": ["薬指2"],
    "ring3": ["薬指3"],
    "ring_tip": ["薬指先"],
    "little1": ["小指1"],
    "little2": ["小指2"],
    "little3": ["小指3"],
    "little_tip": ["小指先"],
    "leg": ["足"],
    "knee": ["ひざ", "膝"],
    "ankle": ["足首"],
    "toe": ["つま先"],
    "leg_ik": ["足IK"],
    "toe_ik": ["つま先IK"],
    "leg_ik_parent": ["足IK親"],
    "leg_d": ["足D"],
    "knee_d": ["ひざD"],
    "ankle_d": ["足首D"],
    "toe_ex": ["足先EX"],
    "waist_cancel": ["腰キャンセル"],
}

FINGERS = {
    "thumb": ["thumb0", "thumb1", "thumb2", "thumb_tip"],
    "index": ["index1", "index2", "index3", "index_tip"],
    "middle": ["middle1", "middle2", "middle3", "middle_tip"],
    "ring": ["ring1", "ring2", "ring3", "ring_tip"],
    "little": ["little1", "little2", "little3", "little_tip"],
}

# the bones any standard MMD model is expected to have (semi-standard ones such as upper_body2 or the twist bones
# are optional and reported as missing, not as errors)
REQUIRED = ["center", "lower_body", "upper_body", "neck", "head"] + [
    f"{s}.{side}" for side in ("L", "R") for s in ("shoulder", "arm", "elbow", "wrist", "leg", "knee", "ankle",
                                                   "leg_ik")]
SIDES = (("L", "左"), ("R", "右"))


def norm(name):
    return unicodedata.normalize("NFKC", name or "").strip()


def split_side(name_j, blender_name=""):
    """('L'|'R'|None, stem) from a PMX name (左腕, 腰キャンセル左) or an mmd_tools Blender name (腕.L)."""
    n = norm(name_j)
    if n:
        for side, jp in SIDES:
            if n.startswith(jp) and len(n) > 1:
                return side, n[1:]
            if n.endswith(jp) and len(n) > 1:
                return side, n[:-1]
        return None, n
    b = norm(blender_name)
    for side, _ in SIDES:
        if b.endswith("." + side) or b.endswith("_" + side):
            return side, b[:-2]
    return None, b


def build_map(bones):
    """Semantic name -> Blender bone name. `bones` maps Blender bone names to their PMX names (name_j, may be '')."""
    index = {}
    for bname, jname in bones.items():
        side, stem = split_side(jname, bname)
        index.setdefault((side, stem), bname)
    out = {}
    for sem, cands in CENTER.items():
        for c in cands:
            hit = index.get((None, norm(c)))
            if hit:
                out[sem] = hit
                break
    for stem, cands in SIDED.items():
        for side, _ in SIDES:
            for c in cands:
                hit = index.get((side, norm(c)))
                if hit:
                    out[f"{stem}.{side}"] = hit
                    break
    return out


def all_semantic():
    """Every semantic name this map knows, in a stable order."""
    names = list(CENTER)
    for stem in SIDED:
        names += [f"{stem}.L", f"{stem}.R"]
    return names
