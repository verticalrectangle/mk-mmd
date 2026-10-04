"""The MMD standard + semi-standard skeleton, built from landmarks.

    bones = standard_bones(landmarks, opts)        # list[Bone], parents before children, PMX names
    frames = standard_frames(bones)                # display frames: "ＩＫ", "体(上)", "体(下)", "腕", "指", "足"

`landmarks` maps semantic names (the names of `mkmmd/core/bonemap.py`) to the HEAD position of that bone in model space
(metres, Z up, facing -Y, left = +X); values are 3-sequences. Sided names carry `.L` / `.R`; give one side and the other
is mirrored (x -> -x). Missing optional landmarks are derived (see `complete`). Required (per side for sided ones):

    upper_body  neck  head  head_tip          (head_tip = top of the skull: the tail of 頭)
    shoulder.L  arm.L  elbow.L  wrist.L       (arm = the shoulder joint, 腕)
    leg.L  knee.L  ankle.L  toe.L             (leg = the hip joint, 足; toe = where つま先 sits, the foot's front)
    fingers (unless opts fingers=False), head positions of every joint:
    thumb0.L thumb1.L thumb2.L thumb_tip.L  index1.L index2.L index3.L index_tip.L  middle1..3 / middle_tip,
    ring1..3 / ring_tip,  little1..3 / little_tip     (the tips are the fingertip points)

Optional (defaults in brackets): lower_body [upper_body] (下半身 head: the pelvis pivot, the pose stage tilts the pelvis
about it); center [lower_body - 0.04 s]; groove [center + 0.015 s]; waist [upper_body]; upper_body2 [30% from upper_body
to neck]; root [0,0,0]; eye.L/eye.R (creates 両目/左目/右目 only when present or opts eyes=True; then eyes [above and in
front of the eyes]); toe_end.L [toe + 0.03 s forward] (tail of つま先); arm_twist.L [60% from arm to elbow];
wrist_twist.L [60% from elbow to wrist]; leg_ik.L [ankle] (足ＩＫ head: leave at the ankle so the rest pose solves with
zero error); toe_ik.L [toe]; shoulder_p.L [shoulder]. `s` is head_tip.z / 1.7.

Options (`opts` dict): groove=True, waist=True, upper_body2=True, eyes=None (auto), shoulder_p=False (肩P/肩C), twist=
"standard" (腕捩/手捩 with fixed axes) | "split" (+ 腕捩1..3/手捩1..3 granted 25/50/75 %) | "none", fingers=True,
toes=True (つま先, つま先ＩＫ), leg_ik=True, ik_loops=50, toe_ik_loops=10, ik_angle=114.5916 (deg per iteration, 2 rad).

IK limits are written in the PMX/MMD convention (the numbers PMXEditor shows, degrees): the knee bends about x between
-180 and -0.5 degrees. Bone hierarchy follows the modern MMD layout (全ての親 > センター > グルーブ > 腰 > 上半身 >
上半身2 > 首 > 頭, 下半身 under 腰, arms through the twist bones so IK on the wrist can twist them, fingers under
手首, 足ＩＫ under 全ての親 with つま先ＩＫ under it)."""
import numpy as np

from ..core import bonemap
from .part import Bone

SIDES = (("L", "左", 1.0), ("R", "右", -1.0))
FINGERS = ("thumb", "index", "middle", "ring", "little")
FINGER_JP = {"thumb": "親指", "index": "人指", "middle": "中指", "ring": "薬指", "little": "小指"}
FINGER_EN = {"thumb": "thumb", "index": "fore", "middle": "middle", "ring": "third", "little": "little"}
FORWARD = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
_FW = str.maketrans("0123456789", "０１２３４５６７８９")
DEFAULTS = dict(groove=True, waist=True, upper_body2=True, eyes=None, shoulder_p=False, twist="standard", fingers=True,
                toes=True, leg_ik=True, ik_loops=50, toe_ik_loops=10, ik_angle=114.5916)
KNEE_LIMIT = [[-180.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]       # PMX convention, degrees

ARM_JOINTS = ("shoulder", "arm", "elbow", "wrist")
LEG_JOINTS = ("leg", "knee", "ankle", "toe")


def finger_joint_names(finger):
    """Semantic joint names of a finger, root to tip: thumb0 thumb1 thumb2 thumb_tip; index1 index2 index3 index_tip."""
    first = 0 if finger == "thumb" else 1
    n = 3
    return [f"{finger}{i}" for i in range(first, first + n)] + [f"{finger}_tip"]


def required_landmarks(opts=None):
    """Names that `standard_bones` needs (central ones plus the `.L` side; `.R` is mirrored when absent)."""
    o = _opts(opts)
    names = ["upper_body", "neck", "head", "head_tip"]
    names += [f"{j}.L" for j in ARM_JOINTS + LEG_JOINTS]
    if o["fingers"]:
        for f in FINGERS:
            names += [f"{j}.L" for j in finger_joint_names(f)]
    return names


def _opts(opts):
    o = dict(DEFAULTS)
    if opts:
        unknown = set(opts) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown skeleton options {sorted(unknown)} (known: {sorted(DEFAULTS)})")
        o.update(opts)
    if o["twist"] not in ("standard", "split", "none"):
        raise ValueError("twist must be 'standard', 'split' or 'none'")
    return o


def _v(x):
    return np.asarray(x, float).reshape(3)


def _mirror(p):
    q = np.array(p, float)
    q[0] = -q[0]
    return q


def complete(landmarks, opts=None):
    """Landmarks with the mirrored side and every derived optional landmark filled in (a new dict of float arrays).
    Raises KeyError listing the missing required names."""
    o = _opts(opts)
    L = {k: _v(v) for k, v in landmarks.items()}
    for k in list(L):
        if k.endswith(".L") and k[:-2] + ".R" not in L:
            L[k[:-2] + ".R"] = _mirror(L[k])
        elif k.endswith(".R") and k[:-2] + ".L" not in L:
            L[k[:-2] + ".L"] = _mirror(L[k])
    need = [n for n in required_landmarks(o) if n not in L]
    need += [n.replace(".L", ".R") for n in required_landmarks(o) if n.endswith(".L") and n.replace(".L", ".R") not in L]
    if need:
        raise KeyError(f"missing landmarks: {sorted(set(need))}")
    s = float(L["head_tip"][2]) / 1.7
    L.setdefault("root", np.zeros(3))
    L.setdefault("lower_body", L["upper_body"].copy())
    L.setdefault("center", L["lower_body"] - np.array([0.0, 0.0, 0.04 * s]))
    L.setdefault("groove", L["center"] + np.array([0.0, 0.0, 0.015 * s]))
    L.setdefault("waist", L["upper_body"].copy())
    L.setdefault("upper_body2", L["upper_body"] + 0.30 * (L["neck"] - L["upper_body"]))
    for side in ("L", "R"):
        a, e, w = L[f"arm.{side}"], L[f"elbow.{side}"], L[f"wrist.{side}"]
        L.setdefault(f"arm_twist.{side}", a + 0.6 * (e - a))
        L.setdefault(f"wrist_twist.{side}", e + 0.6 * (w - e))
        L.setdefault(f"leg_ik.{side}", L[f"ankle.{side}"].copy())
        L.setdefault(f"toe_ik.{side}", L[f"toe.{side}"].copy())
        L.setdefault(f"shoulder_p.{side}", L[f"shoulder.{side}"].copy())
        t = L[f"toe.{side}"]
        L.setdefault(f"toe_end.{side}", t + np.array([0.0, -0.03 * s, 0.0]))
    want_eyes = o["eyes"] if o["eyes"] is not None else ("eye.L" in L or "eye.R" in L)
    if want_eyes:
        if "eye.L" not in L and "eye.R" not in L:
            raise KeyError("opts eyes=True needs the landmark eye.L (or eye.R)")
        for side in ("L", "R"):
            if f"eye.{side}" not in L:
                L[f"eye.{side}"] = _mirror(L["eye.R" if side == "L" else "eye.L"])
        mid = (L["eye.L"] + L["eye.R"]) / 2
        L.setdefault("eyes", mid + np.array([0.0, -0.06 * s, 0.19 * s]))
    return L


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def local_axes(head, child):
    """MMD local axes of an arm/finger bone: x along the bone (head -> child), z = the model's forward direction made
    perpendicular to x (up when the bone points forward). Same rule on both sides, as standard models do."""
    x = _unit(np.asarray(child, float) - np.asarray(head, float))
    z = FORWARD - float(FORWARD @ x) * x
    if np.linalg.norm(z) < 1e-6:
        z = UP - float(UP @ x) * x
    return tuple(float(v) for v in x), tuple(float(v) for v in _unit(z))


def _jp_num(n):
    return str(n).translate(_FW)


class _Builder:
    def __init__(self, land, opts):
        self.L, self.o = land, opts
        self.bones = []
        self.by = {}
        s = float(land["head_tip"][2]) / 1.7
        self.s = s

    def add(self, name, head, parent="", tail=None, tail_bone="", **kw):
        head = np.asarray(head, float)
        if tail is not None:
            tail = tuple(float(x) for x in tail)
        b = Bone(name=name, head=tuple(float(x) for x in head), tail=tail, parent=parent, tail_bone=tail_bone, **kw)
        self.bones.append(b)
        self.by[name] = b
        return b

    def head_of(self, name):
        return np.asarray(self.by[name].head, float)

    def stub(self, head, direction=UP, length=None):
        """A short leaf tail (end point) so mmd_tools does not have to invent one."""
        length = 0.04 * self.s if length is None else length
        return np.asarray(head, float) + _unit(direction) * length


def _semantic_map(names):
    ident = {n: n for n in names}
    inv = {}
    for sem, b in bonemap.build_map(ident).items():
        inv[b] = sem
    return inv


def standard_bones(landmarks, opts=None):
    """The standard skeleton as a list of `Bone` (parents first). See the module docstring for landmarks and options."""
    o = _opts(opts)
    L = complete(landmarks, o)
    B = _Builder(L, o)
    s = B.s
    add = B.add
    use_groove, use_waist = o["groove"], o["waist"]

    # ---- trunk
    add("全ての親", L["root"], tail=L["root"] + np.array([0, 0, 0.08]), name_en="master", movable=True, deform=False)
    add("センター", L["center"], "全ての親", tail=L["center"] - np.array([0, 0, 0.08]), name_en="center", movable=True,
        deform=False)
    spine_parent = "センター"
    if use_groove:
        add("グルーブ", L["groove"], "センター", tail=L["groove"] + np.array([0, 0, 0.10]), name_en="groove",
            movable=True, deform=False)
        spine_parent = "グルーブ"
    if use_waist:
        wt = L["waist"]
        add("腰", wt, spine_parent, tail=wt + np.array([0, 0, 0.06]), name_en="waist", deform=False)
        spine_parent = "腰"
    ub2 = o["upper_body2"]
    add("上半身", L["upper_body"], spine_parent, tail_bone="上半身2" if ub2 else "首", name_en="upper body")
    if ub2:
        add("上半身2", L["upper_body2"], "上半身", tail_bone="首", name_en="upper body2")
    add("首", L["neck"], "上半身2" if ub2 else "上半身", tail_bone="頭", name_en="neck")
    add("頭", L["head"], "首", tail=L["head_tip"], name_en="head")
    if "eyes" in L:
        eyes_head = L["eyes"]
        mid = (L["eye.L"] + L["eye.R"]) / 2
        add("両目", eyes_head, "頭", tail=mid + np.array([0, -0.04 * s, 0]), name_en="eyes", movable=True,
            deform=False)
        for side, jp, _ in SIDES:
            e = L[f"eye.{side}"]
            add(f"{jp}目", e, "頭", tail=e + np.array([0, -0.05 * s, 0]), name_en=f"eye_{side}", layer=2,
                grant={"parent": "両目", "rotate": True, "move": False, "ratio": 1.0})

    # ---- arms
    for side, jp, sx in SIDES:
        sh, ar, el, wr = (L[f"{j}.{side}"] for j in ARM_JOINTS)
        parent = "上半身2" if ub2 else "上半身"
        if o["shoulder_p"]:
            add(f"{jp}肩P", sh, parent, tail=sh + np.array([0, 0, 0.08 * s]), name_en=f"shoulder P_{side}", visible=False,
                deform=False)
            parent = f"{jp}肩P"
        add(f"{jp}肩", sh, parent, tail_bone=f"{jp}腕", name_en=f"shoulder_{side}")
        parent = f"{jp}肩"
        if o["shoulder_p"]:
            add(f"{jp}肩C", ar, parent, tail=ar + np.array([0, 0, 0.08 * s]), name_en=f"shoulder C_{side}",
                visible=False, deform=False,
                grant={"parent": f"{jp}肩P", "rotate": True, "move": False, "ratio": -1.0})
            parent = f"{jp}肩C"
        lx, lz = local_axes(ar, el)
        add(f"{jp}腕", ar, parent, tail_bone=f"{jp}ひじ", name_en=f"arm_{side}", local_x=lx, local_z=lz)
        elbow_parent = f"{jp}腕"
        if o["twist"] != "none":
            at = L[f"arm_twist.{side}"]
            ax = tuple(float(v) for v in _unit(el - ar))
            add(f"{jp}腕捩", at, f"{jp}腕", tail_bone=f"{jp}ひじ", name_en=f"arm twist_{side}", fixed_axis=ax)
            elbow_parent = f"{jp}腕捩"
            if o["twist"] == "split":
                for i, (fr, ratio) in enumerate(((0.30, 0.25), (0.52, 0.50), (0.74, 0.75)), 1):
                    p = ar + fr * (el - ar)
                    add(f"{jp}腕捩{i}", p, f"{jp}腕", tail=p + np.array([0, 0, 0.05 * s]), name_en=f"arm twist{i}_{side}",
                        visible=False, grant={"parent": f"{jp}腕捩", "rotate": True, "move": False, "ratio": ratio})
        lx, lz = local_axes(el, wr)
        add(f"{jp}ひじ", el, elbow_parent, tail_bone=f"{jp}手首", name_en=f"elbow_{side}", local_x=lx, local_z=lz)
        wrist_parent = f"{jp}ひじ"
        if o["twist"] != "none":
            wt = L[f"wrist_twist.{side}"]
            ax = tuple(float(v) for v in _unit(wr - el))
            add(f"{jp}手捩", wt, f"{jp}ひじ", tail_bone=f"{jp}手首", name_en=f"wrist twist_{side}", fixed_axis=ax)
            wrist_parent = f"{jp}手捩"
            if o["twist"] == "split":
                for i, (fr, ratio) in enumerate(((0.30, 0.25), (0.52, 0.50), (0.74, 0.75)), 1):
                    p = el + fr * (wr - el)
                    add(f"{jp}手捩{i}", p, f"{jp}ひじ", tail=p + np.array([0, 0, 0.05 * s]),
                        name_en=f"wrist twist{i}_{side}", visible=False,
                        grant={"parent": f"{jp}手捩", "rotate": True, "move": False, "ratio": ratio})
        mid_name = f"{jp}中指１"
        if o["fingers"]:
            lx, lz = local_axes(wr, L[f"middle1.{side}"])
            add(f"{jp}手首", wr, wrist_parent, tail_bone=mid_name, name_en=f"wrist_{side}", local_x=lx, local_z=lz)
        else:
            add(f"{jp}手首", wr, wrist_parent, tail=wr + _unit(wr - el) * 0.08 * s, name_en=f"wrist_{side}")
        if o["fingers"]:
            for fing in FINGERS:
                joints = finger_joint_names(fing)
                first = 0 if fing == "thumb" else 1
                names = [f"{jp}{FINGER_JP[fing]}{_jp_num(first + i)}" for i in range(3)] + [f"{jp}{FINGER_JP[fing]}先"]
                en = [f"{FINGER_EN[fing]}{first + i}_{side}" for i in range(3)] + [f"{FINGER_EN[fing]} tip_{side}"]
                parent = f"{jp}手首"
                for i, (j, nm) in enumerate(zip(joints, names)):
                    pos = L[f"{j}.{side}"]
                    if i < 3:
                        nxt = L[f"{joints[i + 1]}.{side}"]
                        lx, lz = local_axes(pos, nxt)
                        add(nm, pos, parent, tail_bone=names[i + 1], name_en=en[i], local_x=lx, local_z=lz)
                    else:
                        prev = L[f"{joints[i - 1]}.{side}"]
                        add(nm, pos, parent, tail=pos + _unit(pos - prev) * 0.015 * s, name_en=en[i], visible=False,
                            deform=False)
                    parent = nm

    # ---- legs
    lower_parent = "腰" if use_waist else spine_parent
    add("下半身", L["lower_body"], lower_parent, tail=(L["leg.L"] + L["leg.R"]) / 2, name_en="lower body")
    for side, jp, sx in SIDES:
        leg, kn, an, toe = (L[f"{j}.{side}"] for j in LEG_JOINTS)
        add(f"{jp}足", leg, "下半身", tail_bone=f"{jp}ひざ", name_en=f"leg_{side}")
        add(f"{jp}ひざ", kn, f"{jp}足", tail_bone=f"{jp}足首", name_en=f"knee_{side}")
        if o["toes"]:
            add(f"{jp}足首", an, f"{jp}ひざ", tail_bone=f"{jp}つま先", name_en=f"ankle_{side}")
            add(f"{jp}つま先", toe, f"{jp}足首", tail=L[f"toe_end.{side}"], name_en=f"toe_{side}", visible=False,
                deform=False)
        else:
            add(f"{jp}足首", an, f"{jp}ひざ", tail=toe, name_en=f"ankle_{side}")
    if o["leg_ik"]:
        for side, jp, sx in SIDES:
            ik, tik = L[f"leg_ik.{side}"], L[f"toe_ik.{side}"]
            add(f"{jp}足ＩＫ", ik, "全ての親", tail_bone=f"{jp}つま先ＩＫ" if o["toes"] else "",
                tail=None if o["toes"] else ik + np.array([0, -0.08 * s, 0]), name_en=f"leg IK_{side}", movable=True,
                deform=False,
                ik={"target": f"{jp}足首", "iterations": int(o["ik_loops"]), "angle": float(o["ik_angle"]),
                    "chain": [{"bone": f"{jp}ひざ", "limit": [list(KNEE_LIMIT[0]), list(KNEE_LIMIT[1])]},
                              {"bone": f"{jp}足", "limit": None}]})
            if o["toes"]:
                add(f"{jp}つま先ＩＫ", tik, f"{jp}足ＩＫ", tail=tik + np.array([0, 0, 0.08]), name_en=f"toe IK_{side}",
                    movable=True, deform=False, layer=1,
                    ik={"target": f"{jp}つま先", "iterations": int(o["toe_ik_loops"]), "angle": float(o["ik_angle"]),
                        "chain": [{"bone": f"{jp}足首", "limit": None}]})

    sem = _semantic_map([b.name for b in B.bones])
    for b in B.bones:
        b.semantic = sem.get(b.name, "")
    return B.bones


def standard_frames(bones):
    """Display frames (name -> bone names) for the standard bones present in `bones` ("Root" and "表情" are added by
    the assembler)."""
    names = [b.name for b in bones]
    have = set(names)

    def pick(*cands):
        return [n for n in cands if n in have]
    sides = [jp for _, jp, _ in SIDES]
    frames = {
        "ＩＫ": pick(*[f"{jp}{k}" for jp in sides for k in ("足ＩＫ", "つま先ＩＫ")]),
        "体(上)": pick("センター", "グルーブ", "腰", "上半身", "上半身2", "首", "頭", "両目", "左目", "右目"),
        "体(下)": pick("下半身"),
        "腕": pick(*[f"{jp}{k}" for jp in sides for k in ("肩P", "肩", "肩C", "腕", "腕捩", "ひじ", "手捩", "手首")]),
        "指": [n for n in names if any(f"{jp}{FINGER_JP[f]}" in n for jp in sides for f in FINGERS) and not n.endswith("先")],
        "足": pick(*[f"{jp}{k}" for jp in sides for k in ("足", "ひざ", "足首", "つま先")]),
    }
    return {k: v for k, v in frames.items() if v}
