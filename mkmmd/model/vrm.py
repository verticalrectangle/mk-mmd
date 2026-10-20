"""mk's own model file: a VRM 1.0 written beside the PMX (docs/design.md: VRM). numpy, scipy and PIL; no Blender.

  config(spec) -> cfg                   the spec's [vrm] table over DEFAULTS (an unknown key raises SpecError)
  write(path, pmx, root, unit=0.08, layers=None, cfg=None, name=None) -> report
        `pmx` a pmx_io.PmxModel whose texture paths are relative to the folder `root`, `unit` metres per PMX unit,
        `layers` mk's model layers ({"folds": [...], "cloth": [...]}: bones by PMX name, a cloth copy's arrays inline, as
        `assemble.Assembled.layers` has them, or `rig_layers` of a written rig.json)
  rig_layers(rig_path) -> layers        the model layers of a rig.json (and its .cloth.npz), bones by PMX name

A .vrm is one binary glTF 2.0 file, its textures inside, with these extensions:

  VRMC_vrm              meta: the licence and permissions, restrictive unless [vrm.meta] says otherwise; humanoid: the
                        semantic bones (core.bonemap) under VRM's names (HUMANOID), the hips the lowest bone both the
                        spine and the legs hang from (MMD's waist), a model's D bones in place of its legs' when it has
                        them; expressions: VRM's presets from MMD's standard morphs (PRESETS, by core.bonemap.MORPHS'
                        semantic names; [vrm.expressions] replaces one), every other vertex and group morph a custom
                        expression under its PMX name (fold shapes stay MK_rig's); lookAt: the eye bones; firstPerson
  VRMC_springBone       a spring per physics chain: the bones dynamic rigid bodies move, from a root hanging from one
                        they do not (a branch is its own spring), and a node at the last bone's tail to end it; its
                        stiffness, gravity and drag by the chain's family (core.families; [vrm.springs]), each joint's
                        hit radius its rigid body's. Colliders: the static rigid bodies (a box is the capsule along its
                        longest side), one collider group per PMX collision group; a spring collides with the groups its
                        bodies do
  VRMC_node_constraint  a rotation constraint per PMX rotation grant onto a bone the humanoid does not drive (a helper
                        bone, an eye highlight); a grant from both eyes (両目) comes from the eye on the bone's side,
                        which VRM's look-at turns
  VRMC_materials_mtoon  MMD's look: the lit colour saturate(ambient + 0.6 x diffuse) (MMD's default light) times the
                        texture; the shade the toon ramp's darkest row times that, the step where the ramp's is (MMD
                        reads the ramp at 0.5 + 0.5 N.L from its bottom: a step at v, w wide between 10 and 90 %, is
                        shadingShift 1 - 2v, shadingToony 1 - 1.25w); flat without a toon; an additive sphere map the
                        matcap; the edge the outline (screen ratio [vrm] outline per unit of edge size, unlit). A
                        viewer without MToon may draw it unlit (MToon's own fallback)
  MK_rig                what VRM cannot hold: the PMX names, the turns back to the PMX's rest pose, IK, the grants VRM
                        did not take, bone, UV and material morphs, the rigid bodies and joints, the chains and their
                        sideways ties, the fold shapes and their drivers, the cloth copies (their arrays in the binary
                        chunk), the display frames, the PMX material numbers (docs/design.md: VRM)

The rest pose is VRM's T-pose (lab.tpose_rotations), posed with the model's own weights and grants: every node keeps no
rotation, the vertices, normals and morph offsets are the posed ones (a morph's offset turned by its vertex's blended
rotation), an IK bone sits where its target went. glTF's axes (glb.AXES): +Y up, the model faces +Z, its left is +X."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ..core import bonemap, families
from ..core import rig as RG
from . import glb as GLB
from . import lab as LAB
from . import spec as SP
from .assemble import _rx, _ry, _rz
from .pmx_take import _find_file, to_model

SPEC_VERSION = "1.0"
LICENCE = "https://vrm.dev/licenses/1.0/"            # the VRM Public License 1.0, which the meta's permissions set
MMD_LIGHT = 154.0 / 255.0                           # MMD's default light colour
SHARED_TOON = (0.80, 0.50, 0.08)                    # a shared toon (toon01..10, MMD's own files): shade, step, width
AXES = GLB.AXES
_FLOAT, _UINT, _USHORT = 5126, 5125, 5123
_ARRAY, _ELEMENTS = 34962, 34963
DOWN = [0.0, -1.0, 0.0]

# chain family -> VRM spring settings: stiffness (pull back to the rest direction), gravity, drag (0..1)
SPRINGS = {"bangs": (2.0, 0.05, 0.6), "side_hair": (1.0, 0.15, 0.45), "back_hair": (0.8, 0.15, 0.45),
           "twintail": (0.8, 0.2, 0.4), "braid": (0.9, 0.2, 0.45), "hair": (0.9, 0.15, 0.45), "ears": (4.0, 0.0, 0.5),
           "tail": (1.2, 0.1, 0.4), "skirt": (1.0, 0.1, 0.5), "ribbon": (0.6, 0.3, 0.4), "breasts": (4.0, 0.0, 0.7),
           "sleeve": (1.2, 0.3, 0.5), "coat": (1.2, 0.2, 0.5), "accessory": (1.0, 0.2, 0.45), "other": (1.0, 0.1, 0.5)}

DEFAULTS = {
    "enabled": True,
    "meta": {"name": "", "version": "", "authors": ["unknown"], "copyrightInformation": "", "contactInformation": "",
             "references": [], "licenseUrl": LICENCE, "avatarPermission": "onlyAuthor",
             "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False,
             "commercialUsage": "personalNonProfit", "allowPoliticalOrReligiousUsage": False,
             "allowAntisocialOrHateUsage": False, "creditNotation": "required", "allowRedistribution": False,
             "modification": "prohibited"},
    "outline": 0.0016,                              # MToon outline width (a ratio of the screen height) per edge size
    "look": {"inner": 10.0, "outer": 10.0, "up": 10.0, "down": 10.0},      # degrees the eyes turn at most
    "springs": {f: {"stiffness": s, "gravity": g, "drag": d} for f, (s, g, d) in SPRINGS.items()},
    "expressions": {},                              # preset -> [[morph, weight], ...] (a PMX or a semantic morph name)
}

# VRM humanoid bone -> semantic bone (core.bonemap); the hips are found (`_hips`), legs prefer their D bones
HUMANOID = {"spine": "upper_body", "chest": "upper_body2", "upperChest": "upper_body3", "neck": "neck", "head": "head"}
for _side, _s in (("left", "L"), ("right", "R")):
    HUMANOID.update({f"{_side}Eye": f"eye.{_s}", f"{_side}UpperLeg": f"leg.{_s}", f"{_side}LowerLeg": f"knee.{_s}",
                     f"{_side}Foot": f"ankle.{_s}", f"{_side}Toes": f"toe_ex.{_s}", f"{_side}Shoulder": f"shoulder.{_s}",
                     f"{_side}UpperArm": f"arm.{_s}", f"{_side}LowerArm": f"elbow.{_s}", f"{_side}Hand": f"wrist.{_s}",
                     f"{_side}ThumbMetacarpal": f"thumb0.{_s}", f"{_side}ThumbProximal": f"thumb1.{_s}",
                     f"{_side}ThumbDistal": f"thumb2.{_s}"})
    for _f, _vf in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("little", "Little")):
        for _k, _part in enumerate(("Proximal", "Intermediate", "Distal")):
            HUMANOID[f"{_side}{_vf}{_part}"] = f"{_f}{_k + 1}.{_s}"
D_BONES = {"leg": "leg_d", "knee": "knee_d", "ankle": "ankle_d"}

# VRM humanoid bone -> the humanoid bones its parent may be, nearest first: the first one the model has must be an
# ancestor of it, and a bone marked True needs that one (VRM 1.0 humanoid.md)
_UP = {"spine": ["hips"], "chest": ["spine"], "upperChest": [True, "chest"], "neck": ["upperChest", "chest", "spine"],
       "head": ["neck", "upperChest", "chest", "spine"], "jaw": ["head"]}
for _side in ("left", "right"):
    _UP.update({f"{_side}Eye": ["head"], f"{_side}UpperLeg": ["hips"], f"{_side}LowerLeg": [f"{_side}UpperLeg"],
                f"{_side}Foot": [f"{_side}LowerLeg"], f"{_side}Toes": [f"{_side}Foot"],
                f"{_side}Shoulder": ["upperChest", "chest", "spine"],
                f"{_side}UpperArm": [f"{_side}Shoulder", "upperChest", "chest", "spine"],
                f"{_side}LowerArm": [f"{_side}UpperArm"], f"{_side}Hand": [f"{_side}LowerArm"],
                f"{_side}ThumbMetacarpal": [f"{_side}Hand"], f"{_side}ThumbProximal": [True, f"{_side}ThumbMetacarpal"],
                f"{_side}ThumbDistal": [True, f"{_side}ThumbProximal"]})
    for _vf in ("Index", "Middle", "Ring", "Little"):
        _UP.update({f"{_side}{_vf}Proximal": [f"{_side}Hand"],
                    f"{_side}{_vf}Intermediate": [True, f"{_side}{_vf}Proximal"],
                    f"{_side}{_vf}Distal": [True, f"{_side}{_vf}Intermediate"]})

# VRM expression preset -> [(semantic morph, weight)] (core.bonemap.MORPHS)
PRESETS = {
    "happy": [("smile_eyes", 1.0), ("cheerful", 1.0), ("mouth_smile", 1.0)],
    "angry": [("angry", 1.0), ("jito", 0.4), ("mouth_down", 1.0)],
    "sad": [("sad", 1.0), ("mouth_down", 1.0)],
    "relaxed": [("calm", 1.0), ("mouth_smile", 0.6)],
    "surprised": [("surprised", 1.0), ("brow_up", 1.0), ("o", 0.35)],
    "aa": [("a", 1.0)], "ih": [("i", 1.0)], "ou": [("u", 1.0)], "ee": [("e", 1.0)], "oh": [("o", 1.0)],
    "blink": [("blink", 1.0)], "blinkLeft": [("wink_l", 1.0)], "blinkRight": [("wink_r", 1.0)],
}
EMOTIONS = ("happy", "angry", "sad", "relaxed", "surprised")
ALL_PRESETS = tuple(PRESETS) + ("lookUp", "lookDown", "lookLeft", "lookRight", "neutral")


def config(spec):
    """The spec's [vrm] table over DEFAULTS. Raises SpecError for a key DEFAULTS does not have (a misspelt one must not
    pass for one that works: mk model build reads this table itself, so no part notes it)."""
    own = (spec.get("vrm") if spec is not None else None) or {}

    def keys(t, d, where):
        for k in t:
            if k not in d:
                raise SP.SpecError(f"[{where}] {k}: unknown key (keys: {', '.join(d)})")
    keys(own, DEFAULTS, "vrm")
    for table in ("meta", "look"):
        keys(own.get(table) or {}, DEFAULTS[table], f"vrm.{table}")
    for fam, s in (own.get("springs") or {}).items():
        keys({fam: 0}, DEFAULTS["springs"], "vrm.springs")
        keys(s, DEFAULTS["springs"][fam], f"vrm.springs.{fam}")
    for p in own.get("expressions") or {}:
        if p not in ALL_PRESETS:
            raise SP.SpecError(f"[vrm.expressions] {p}: not a VRM preset ({', '.join(ALL_PRESETS)})")
    return SP.merge(DEFAULTS, dict(own))


# ------------------------------------------------------------------------------------------------------------ maths
def _lin(c):
    """sRGB (0..1) -> linear, as glTF and MToon take colour factors."""
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _r(v, k=6):
    return [round(float(x), k) for x in np.asarray(v, float).ravel()]


def _gl(v):
    """Model-space vector(s) -> glTF axes."""
    return np.asarray(v, float) @ AXES.T


def _quat(R):
    """glTF (x, y, z, w) of a model-space rotation."""
    return _r(Rotation.from_matrix(AXES @ R @ AXES.T).as_quat())


def _body_matrix(rot):
    """Model-space rotation of a PMX rigid body's or joint's rotation (pmx_take._model_euler's convention)."""
    a, b, c = (float(x) for x in rot)
    return _rz(-b) @ _rx(-a) @ _ry(-c)


def _ramp(path):
    """(lit, shade, v, w) of a toon ramp image: its top and bottom rows (sRGB 0..1), where up from the bottom (0..1) its
    step from the shade to the lit colour is half way, and how wide it is between 10 and 90 % (MMD reads row
    0.5 - 0.5 N.L from the top)."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), float) / 255.0
    col = a[:, a.shape[1] // 2]
    top, bot = col[0], col[-1]
    d = top - bot
    if float(d @ d) < 1e-6:
        return top, bot, 0.5, 0.0
    t = np.clip(((col - bot) @ d) / float(d @ d), 0.0, 1.0)[::-1]         # bottom row first: rising towards lit
    v = (np.arange(len(t)) + 0.5) / len(t)
    t = np.maximum.accumulate(t)
    tie = t + np.arange(len(t)) * 1e-9                       # ties broken upwards
    return top, bot, float(np.interp(0.5, tie, v)), max(float(np.interp(0.9, tie, v) - np.interp(0.1, tie, v)), 0.0)


# ------------------------------------------------------------------------------------------------------- the writer
class _Doc:
    """The glTF document and its binary chunk."""

    def __init__(self):
        self.bin, self.accessors = GLB.Bin(), []

    def acc(self, arr, kind, target=None, ctype=_FLOAT, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self.bin.view(arr.tobytes(), target), "componentType": ctype, "count": len(arr),
             "type": kind}
        if minmax:
            a["min"], a["max"] = [float(x) for x in arr.min(axis=0)], [float(x) for x in arr.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    def sparse(self, n, idx, vals):
        """A VEC3 float accessor of n zeros but `vals` at the (increasing) indices `idx`: a morph target."""
        vals = np.asarray(vals, np.float32)
        lo, hi = np.minimum(vals.min(axis=0), 0.0), np.maximum(vals.max(axis=0), 0.0)
        a = {"componentType": _FLOAT, "count": int(n), "type": "VEC3", "min": [float(x) for x in lo],
             "max": [float(x) for x in hi],
             "sparse": {"count": len(idx),
                        "indices": {"bufferView": self.bin.view(np.asarray(idx, np.uint32).tobytes()),
                                    "componentType": _UINT},
                        "values": {"bufferView": self.bin.view(vals.tobytes())}}}
        self.accessors.append(a)
        return len(self.accessors) - 1


def _unique_names(names):
    seen, out = {}, []
    for n in names:
        k = n or "bone"
        if k in seen:
            seen[k] += 1
            k = f"{k}_{seen[k]}"
        else:
            seen[k] = 0
        out.append(k)
    return out


def _hips(M):
    """The lowest bone both the spine (上半身) and the legs hang from: MMD's waist (腰) where a model has one."""
    want = [M.sem.get(s) for s in ("upper_body", "leg.L", "leg.R")]
    if any(w is None for w in want):
        return None

    def ancestors(b):
        out = []
        b = int(M.parents[b])
        while b >= 0:
            out.append(b)
            b = int(M.parents[b])
        return out
    common = set(ancestors(want[0]))
    for w in want[1:]:
        common &= set(ancestors(w))
    for b in ancestors(want[0]):                    # nearest first
        if b in common:
            return b
    return None


def _humanoid(M, pmx, warn):
    """{VRM humanoid bone: bone index}, checked against VRM's parent rules (a bone that breaks them is left out)."""
    out = {}
    hips = _hips(M)
    if hips is not None:
        out["hips"] = hips
    for vrm, sem in HUMANOID.items():
        stem, side = sem.rsplit(".", 1) if "." in sem else (sem, None)
        b = M.sem.get(sem)
        d = M.sem.get(f"{D_BONES[stem]}.{side}") if stem in D_BONES and side else None
        if d is not None and M.grants[d] is not None and M.grants[d][0] == b and abs(M.grants[d][1] - 1.0) < 1e-6:
            b = d                                       # the D bone carries the leg's weights
        if b is not None:
            out[vrm] = b

    def ancestors(b):
        s = set()
        b = int(M.parents[b])
        while b >= 0:
            s.add(b)
            b = int(M.parents[b])
        return s
    for vrm in [k for k in HUMANOID if k in out]:      # parents before children (HUMANOID's order)
        if vrm not in out:
            continue
        ups = _UP.get(vrm, [])
        need = bool(ups) and ups[0] is True
        cands = [u for u in ups if u is not True]
        have = next((u for u in cands if u in out), None)
        ok = (have is not None and out[have] in ancestors(out[vrm])) if (need or have) else bool(not cands)
        if need and (not cands or cands[0] not in out):
            ok = False
        if not ok and cands:
            warn.append(f"humanoid {vrm} ({M.bones[out[vrm]]}) left out: it does not hang from "
                        f"{have or cands[0]}")
            del out[vrm]
    for req in ("hips", "spine", "head", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg",
                "rightFoot", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand"):
        if req not in out:
            raise ValueError(f"the model has no {req} bone VRM requires (semantic bones: core.bonemap)")
    return out


def _chains(pmx, nb, parents, skip):
    """[(bones, body per bone)] of the physics chains: bones dynamic rigid bodies move, from a root hanging from a bone
    they do not; a branch is its own chain (VRM 1.0 springBone). `skip`: bones no chain may take (the humanoid's)."""
    dyn = {}
    for i, body in enumerate(pmx.bodies):
        if body.mode != 0 and 0 <= body.bone < nb and body.bone not in skip:
            dyn.setdefault(body.bone, i)
    kids = {}
    for b in range(nb):
        kids.setdefault(int(parents[b]), []).append(b)
    out = []
    for root in sorted(b for b in dyn if int(parents[b]) not in dyn):
        stack = [root]
        while stack:
            cur = stack.pop()
            chain = [cur]
            while True:
                nxt = [c for c in kids.get(cur, []) if c in dyn]
                if not nxt:
                    break
                cur = nxt[0]
                chain.append(cur)
                stack += nxt[1:]
            out.append((chain, [dyn[b] for b in chain]))
    return out


def _radius(body, unit):
    s = np.asarray(body.size, float) * unit
    return float(s[0]) if body.shape in (0, 2) else float(min(s))


def write(path, pmx, root, unit=0.08, layers=None, cfg=None, name=None):
    """Write `pmx` as a VRM 1.0 to `path` (module docstring). Returns a report."""
    cfg = cfg if cfg is not None else SP.merge(DEFAULTS, {})
    warn = []
    M = LAB.from_pmx(pmx, Path(root), unit, label=name or pmx.name or "model")
    nb = len(M.bones)
    if nb == 0 or len(M.V) == 0:
        raise ValueError("the model has no bones or no vertices")
    tpose = LAB.tpose_rotations(M)
    G = M.matrices(tpose)
    R3 = G[:, :3, :3]
    V, N = M.skin(M.V, M.N, G)
    heads = np.einsum("bij,bj->bi", R3, M.heads) + G[:, :3, 3]
    tails = np.einsum("bij,bj->bi", R3, M.tails) + G[:, :3, 3]
    for b, bone in enumerate(pmx.bones):                       # an IK bone goes where its target went
        if bone.ik is not None and 0 <= bone.ik.target < nb:
            t = bone.ik.target
            heads[b], tails[b] = heads[t] + (M.heads[b] - M.heads[t]), tails[t] + (M.tails[b] - M.heads[t])
    parents = M.parents
    hum = _humanoid(M, pmx, warn)
    hum_bones = set(hum.values())

    # ---- skin weights: duplicates merged, unweighted vertices on the root bone, each row summing to 1 in float32
    J, W = M.wi.copy(), M.ww.copy()
    for k in range(4):
        for j in range(k):
            same = (J[:, j] == J[:, k]) & (W[:, k] > 0)
            W[same, j] += W[same, k]
            W[same, k] = 0.0
    J[W <= 0] = 0
    free = W.sum(axis=1) <= 0
    J[free, 0], W[free, 0] = 0, 1.0
    if free.any():
        warn.append(f"{int(free.sum())} vertices had no weights: they follow the root bone")
    W = (W / W.sum(axis=1, keepdims=True)).astype(np.float32)
    top = np.argmax(W, axis=1)
    rows = np.arange(len(W))
    W[rows, top] = 0.0
    W[rows, top] = np.float32(1.0) - W.sum(axis=1, dtype=np.float32)
    Mv = np.einsum("nk,nkij->nij", W.astype(float), R3[J])       # each vertex's blended rotation (its morph offsets)

    doc = _Doc()
    names = _unique_names(M.bones)

    # ---- nodes: the bones (node i = bone i), the springs' end nodes, the mesh
    nodes = []
    for b in range(nb):
        p = int(parents[b])
        t = heads[b] - (heads[p] if p >= 0 else 0.0)
        nodes.append({"name": names[b], "translation": _r(_gl(t))})
    for b in range(nb):
        p = int(parents[b])
        if p >= 0:
            nodes[p].setdefault("children", []).append(b)

    # ---- the mesh: one primitive per material over shared attributes
    n = len(V)
    pos = doc.acc(_gl(V).astype(np.float32), "VEC3", _ARRAY, minmax=True)
    Ng = _gl(N)
    Ng /= np.maximum(np.linalg.norm(Ng, axis=1, keepdims=True), 1e-12)
    nrm = doc.acc(Ng.astype(np.float32), "VEC3", _ARRAY)
    tex0 = doc.acc(np.asarray(M.uv, np.float32), "VEC2", _ARRAY)
    joints = doc.acc(J.astype(np.uint16), "VEC4", _ARRAY, ctype=_USHORT)
    weights = doc.acc(W, "VEC4", _ARRAY)
    ibm = np.zeros((nb, 4, 4), np.float32)
    ibm[:] = np.eye(4, dtype=np.float32)
    ibm[:, 3, :3] = -_gl(heads)                                 # column-major: the translation is the last column
    ibm_acc = doc.acc(ibm.reshape(nb, 16), "MAT4")

    vmorph = [i for i, m in enumerate(pmx.morphs) if m.kind == "vertex" and m.offsets]
    target_of, targets, target_names = {}, [], []
    for i in vmorph:
        m = pmx.morphs[i]
        idx = np.array([o[0] for o in m.offsets], np.int64)
        d = to_model(np.array([o[1] for o in m.offsets], float), unit)
        ok = (idx >= 0) & (idx < n)
        acc = np.zeros((n, 3))
        np.add.at(acc, idx[ok], d[ok])
        used = np.flatnonzero(np.abs(acc).max(axis=1) > 0)
        if not len(used):
            continue
        vals = _gl(np.einsum("nij,nj->ni", Mv[used], acc[used]))
        target_of[i] = len(targets)
        targets.append({"POSITION": doc.sparse(n, used, vals)})
        target_names.append(m.name)

    images, textures, tex_of = [], [], {}

    def texture(idx):
        if not 0 <= idx < len(pmx.textures):
            return None, False
        f = _find_file(Path(root), pmx.textures[idx])
        if f is None:
            return None, False
        key = str(Path(f).resolve())
        if key not in tex_of:
            data, see = GLB.png(f)
            images.append({"bufferView": doc.bin.view(data), "mimeType": "image/png", "name": Path(f).name})
            textures.append({"sampler": 0, "source": len(images) - 1})
            tex_of[key] = (len(textures) - 1, see)
        return tex_of[key]

    materials, prims, mk_mats = [], [], []
    ramps = {}
    for mi, m in enumerate(pmx.materials):
        tris = M.T[M.tri_mat == mi]
        lit = np.clip(np.asarray(m.ambient, float) + MMD_LIGHT * np.asarray(m.diffuse[:3], float), 0.0, 1.0)
        top_c, bot_c, step, width, toon = np.ones(3), np.ones(3), 0.5, 0.0, "none"
        if m.toon_shared:
            bot_c, step, width = np.full(3, SHARED_TOON[0]), SHARED_TOON[1], SHARED_TOON[2]
            toon = f"shared {int(m.toon) + 1:02d}"
        elif 0 <= m.toon < len(pmx.textures):
            f = _find_file(Path(root), pmx.textures[m.toon])
            if f is not None:
                key = str(Path(f).resolve())
                if key not in ramps:
                    ramps[key] = _ramp(f)
                top_c, bot_c, step, width = ramps[key]
                toon = Path(f).name
            else:
                warn.append(f"material {m.name}: toon {pmx.textures[m.toon]} not found: drawn flat")
        base = [*_r(_lin(lit * top_c)), round(float(np.clip(m.diffuse[3], 0.0, 1.0)), 6)]
        mt = {"specVersion": SPEC_VERSION, "shadeColorFactor": _r(_lin(lit * bot_c)),
              "shadingShiftFactor": round(1.0 - 2.0 * step, 4) if toon != "none" else 0.0,
              "shadingToonyFactor": round(float(np.clip(1.0 - 1.25 * width, 0.0, 1.0)), 4) if toon != "none" else 0.9,
              "giEqualizationFactor": 0.9}
        pbr = {"baseColorFactor": base, "metallicFactor": 0.0, "roughnessFactor": 1.0}
        ti, see = texture(m.texture)
        if ti is not None:
            pbr["baseColorTexture"] = {"index": ti}
            mt["shadeMultiplyTexture"] = {"index": ti}
        see = see or m.diffuse[3] < 0.999
        if m.sphere_mode == 2:
            si, _ = texture(m.sphere_texture)
            if si is not None:
                mt["matcapTexture"], mt["matcapFactor"] = {"index": si}, [1.0, 1.0, 1.0]
        if m.edge and m.edge_size > 0:
            mt.update(outlineWidthMode="screenCoordinates",
                      outlineWidthFactor=round(float(m.edge_size) * float(cfg["outline"]), 6),
                      outlineColorFactor=_r(_lin(np.asarray(m.edge_color[:3], float))), outlineLightingMixFactor=0.0)
        if see:
            mt["transparentWithZWrite"] = True
        materials.append({"name": m.name, "pbrMetallicRoughness": pbr, "alphaMode": "BLEND" if see else "OPAQUE",
                          "doubleSided": bool(m.double_sided),
                          "extensions": {"VRMC_materials_mtoon": mt}})
        mk_mats.append({"material": mi, "jp": m.name, "en": m.name_en, "diffuse": _r(m.diffuse), "ambient": _r(m.ambient),
                        "specular": _r(m.specular), "shininess": round(float(m.shininess), 4), "toon": toon,
                        "toon_step": round(float(step), 4), "toon_width": round(float(width), 4),
                        "sphere": ({"texture": pmx.textures[m.sphere_texture], "mode": int(m.sphere_mode)}
                                   if 0 <= m.sphere_texture < len(pmx.textures) and m.sphere_mode else None),
                        "edge": {"on": bool(m.edge), "size": round(float(m.edge_size), 4), "color": _r(m.edge_color)},
                        "flags": {"ground_shadow": bool(m.ground_shadow), "self_shadow_map": bool(m.self_shadow_map),
                                  "self_shadow": bool(m.self_shadow)}})
        if not len(tris):
            continue
        idx = doc.acc(np.asarray(tris[:, [0, 2, 1]], np.uint32).reshape(-1), "SCALAR", _ELEMENTS, ctype=_UINT)
        prim = {"attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": tex0, "JOINTS_0": joints,
                               "WEIGHTS_0": weights}, "indices": idx, "material": mi}
        if targets:
            prim["targets"] = targets
        prims.append(prim)
    if not prims:
        raise ValueError("the model has no triangles")

    # ---- spring chains: their end nodes, joints, colliders
    end_nodes = []
    mesh_node = None
    chains = _chains(pmx, nb, parents, hum_bones)
    springs, in_spring = [], {}
    fam_cfg = cfg["springs"]
    for k, (bones, bodies) in enumerate(chains):
        fam = families.classify(*(M.bones[b] for b in bones[:3]))
        p = fam_cfg.get(fam) or fam_cfg["other"]
        last = bones[-1]
        tail = tails[last]
        js = []
        for b, bi in zip(bones, bodies):
            js.append({"node": b, "hitRadius": round(_radius(pmx.bodies[bi], unit), 5),
                       "stiffness": float(p["stiffness"]), "gravityPower": float(p["gravity"]),
                       "gravityDir": DOWN, "dragForce": float(p["drag"])})
            in_spring[b] = k
        if np.linalg.norm(tail - heads[last]) > 1e-6:
            e = nb + len(end_nodes)
            end_nodes.append({"name": f"{names[last]}_end", "translation": _r(_gl(tail - heads[last]))})
            nodes[last].setdefault("children", []).append(e)
            js.append({"node": e})
        springs.append({"name": M.bones[bones[0]], "joints": js, "_bodies": bodies, "_family": fam})
    nodes += end_nodes
    mesh_node = len(nodes)
    nodes.append({"name": M.names[0] or name or "mesh", "mesh": 0, "skin": 0})

    colliders, group_of = [], {}
    dyn_bones = {b for b, _ in in_spring.items()}
    for i, body in enumerate(pmx.bodies):
        if body.mode != 0 or not 0 <= body.bone < nb or body.bone in dyn_bones:
            continue
        b = body.bone
        c = to_model(np.asarray(body.pos, float), unit)[0]
        R = R3[b] @ _body_matrix(body.rot)
        off = (R3[b] @ c + G[b][:3, 3]) - heads[b]
        s = np.asarray(body.size, float) * unit
        if body.shape == 0:
            shape = {"sphere": {"offset": _r(_gl(off)), "radius": round(float(s[0]), 5)}}
        else:
            if body.shape == 2:
                axis, half, rad = R[:, 2], float(s[1]) / 2.0, float(s[0])
            else:
                half3 = np.array([s[0], s[2], s[1]])                # PMX box size -> model axes (assemble._swap)
                k = int(np.argmax(half3))
                others = np.delete(half3, k)
                axis, rad = R[:, k], float(others.mean())
                half = max(float(half3[k]) - rad, 0.0)
            shape = {"capsule": {"offset": _r(_gl(off - axis * half)), "radius": round(rad, 5),
                                 "tail": _r(_gl(off + axis * half))}}
        group_of.setdefault(int(body.group), []).append(len(colliders))
        colliders.append({"node": b, "shape": shape})
    groups = sorted(group_of)
    collider_groups = [{"name": f"group {g}", "colliders": group_of[g]} for g in groups]
    for sp in springs:
        mask = 0
        for bi in sp["_bodies"]:
            mask |= int(pmx.bodies[bi].mask)
        sp["colliderGroups"] = [k for k, g in enumerate(groups) if mask & (1 << g)]
        if not sp["colliderGroups"]:
            del sp["colliderGroups"]

    # ---- constraints: rotation grants onto bones the humanoid does not drive
    constraints, kept_grants = {}, []
    eyes = M.sem.get("eyes")
    for b, bone in enumerate(pmx.bones):
        if not (bone.grant_rotate or bone.grant_move) or not 0 <= bone.grant_parent < nb:
            continue
        src, why = bone.grant_parent, None
        if src == eyes:
            side = bonemap.split_side(bone.name)[0]
            src = M.sem.get(f"eye.{side}") if side else None
        if bone.grant_move:
            why = "a moving grant"
        elif b in hum_bones:
            why = "onto a humanoid bone (the humanoid drives it)"
        elif src is None or src == b:
            why = "from both eyes, on a bone of neither side"
        elif not 0.0 < bone.grant_ratio <= 1.0:
            why = f"ratio {bone.grant_ratio:g} (a constraint takes 0..1)"
        if why is None:
            constraints[b] = (src, float(bone.grant_ratio))
        kept_grants.append({"node": b, "source": int(bone.grant_parent), "ratio": round(float(bone.grant_ratio), 6),
                            "rotate": bool(bone.grant_rotate), "move": bool(bone.grant_move),
                            "constraint": None if why else {"source": int(src)}, "why": why})
    for b in list(constraints):                                 # no source chain may come back to its bone
        seen, s = {b}, constraints[b][0]
        while s in constraints:
            if s in seen:
                warn.append(f"grant cycle through {M.bones[b]}: no constraint")
                del constraints[b]
                break
            seen.add(s)
            s = constraints[s][0]
    for b, (src, w) in constraints.items():
        nodes[b]["extensions"] = {"VRMC_node_constraint": {"specVersion": SPEC_VERSION,
                                                           "constraint": {"rotation": {"source": int(src),
                                                                                       "weight": round(w, 6)}}}}

    # ---- expressions
    folds = list((layers or {}).get("folds") or [])
    fold_morphs = {f["morph"] for f in folds}
    sem_morph = bonemap.morph_map([m.name for m in pmx.morphs])
    by_name = {m.name: i for i, m in enumerate(pmx.morphs)}

    def binds(entries, ratio=1.0, seen=()):
        """(morph target binds, material colour binds) of [(morph index, weight)], groups expanded."""
        mt, mc = {}, []
        for i, w in entries:
            m = pmx.morphs[i]
            if m.kind == "vertex" and i in target_of:
                mt[target_of[i]] = min(1.0, mt.get(target_of[i], 0.0) + w * ratio)
            elif m.kind == "group" and i not in seen:
                a, b = binds([(int(j), float(r)) for j, r in m.offsets if 0 <= int(j) < len(pmx.morphs)],
                             w * ratio, seen + (i,))
                for k_, v in a.items():
                    mt[k_] = min(1.0, mt.get(k_, 0.0) + v)
                mc += b
            elif m.kind == "material":
                mc += _material_binds(pmx, m, materials, w * ratio)
        return mt, mc

    def expression(mt, mc, preset=None):
        e = {"morphTargetBinds": [{"node": mesh_node, "index": k_, "weight": round(v, 6)} for k_, v in sorted(mt.items())
                                  if v > 0], "isBinary": False}
        if mc:
            e["materialColorBinds"] = mc
        if preset in EMOTIONS:
            e.update(overrideBlink="blend", overrideLookAt="none", overrideMouth="blend" if preset == "surprised" else "none")
        return e

    preset_out, custom = {}, {}
    for preset, default in PRESETS.items():
        entries = []
        for nm, w in cfg["expressions"].get(preset) or default:
            mname = sem_morph.get(nm, nm)
            if mname in by_name:
                entries.append((by_name[mname], float(w)))
        mt, mc = binds(entries)
        if mt or mc:
            preset_out[preset] = expression(mt, mc, preset)
    for i, m in enumerate(pmx.morphs):
        if m.name in fold_morphs or m.kind not in ("vertex", "group", "material") or m.name in ALL_PRESETS:
            continue
        mt, mc = binds([(i, 1.0)])
        if (mt or mc) and m.name not in custom:
            custom[m.name] = expression(mt, mc)

    # ---- VRMC_vrm
    meta = {k: v for k, v in dict(cfg["meta"]).items() if v not in ("", [], None)}
    meta["name"] = meta.get("name") or name or pmx.name or "model"
    meta["authors"] = [str(a) for a in (meta.get("authors") or ["unknown"])]
    vrm = {"specVersion": SPEC_VERSION, "meta": meta,
           "humanoid": {"humanBones": {k: {"node": int(b)} for k, b in hum.items()}},
           "firstPerson": {"meshAnnotations": [{"node": mesh_node, "type": "auto"}]}}
    expressions = {}
    if preset_out:
        expressions["preset"] = preset_out
    if custom:
        expressions["custom"] = custom
    if expressions:
        vrm["expressions"] = expressions
    if "leftEye" in hum and "rightEye" in hum:
        look = cfg["look"]
        mid = 0.5 * (heads[hum["leftEye"]] + heads[hum["rightEye"]])
        rm = {k: {"inputMaxValue": 90.0, "outputScale": float(look[k])} for k in ("inner", "outer", "down", "up")}
        vrm["lookAt"] = {"type": "bone", "offsetFromHeadBone": _r(_gl(mid - heads[hum["head"]])),
                         "rangeMapHorizontalInner": rm["inner"], "rangeMapHorizontalOuter": rm["outer"],
                         "rangeMapVerticalDown": rm["down"], "rangeMapVerticalUp": rm["up"]}

    spring_ext = {"specVersion": SPEC_VERSION}
    if colliders:
        spring_ext["colliders"], spring_ext["colliderGroups"] = colliders, collider_groups
    if springs:
        spring_ext["springs"] = [{k: v for k, v in sp.items() if not k.startswith("_")} for sp in springs]

    # ---- MK_rig
    mk = _mk_rig(pmx, M, G, R3, heads, tails, tpose, unit, names, target_of, chains, springs, kept_grants, mk_mats,
                 folds, (layers or {}).get("cloth") or [], doc, warn)

    used_ext = ["VRMC_materials_mtoon", "VRMC_springBone", "VRMC_vrm", "MK_rig"]
    if constraints:
        used_ext.insert(2, "VRMC_node_constraint")
    title = name or pmx.name or "model"
    mesh = {"name": title, "primitives": prims}
    if targets:
        mesh["weights"] = [0.0] * len(targets)
        mesh["extras"] = {"targetNames": target_names}
    roots = [b for b in range(nb) if int(parents[b]) < 0]
    gl = {"asset": {"version": "2.0", "generator": "mk-mmd (mk model vrm)"}, "extensionsUsed": used_ext,
          "scene": 0, "scenes": [{"name": title, "nodes": roots + [mesh_node]}], "nodes": nodes,
          "meshes": [mesh], "skins": [{"inverseBindMatrices": ibm_acc, "joints": list(range(nb)), "skeleton": roots[0]}],
          "materials": materials, "accessors": doc.accessors,
          "extensions": {"VRMC_vrm": vrm, "VRMC_springBone": spring_ext, "MK_rig": mk}}
    if images:
        gl.update(images=images, textures=textures,
                  samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
    blob = doc.bin.bytes()
    gl["buffers"] = [{"byteLength": len(blob)}]
    gl["bufferViews"] = doc.bin.views
    out = GLB.pack(gl, blob)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(out)
    tmp.replace(path)
    return {"vrm": str(path), "mb": round(len(out) / 1e6, 2), "bones": nb, "humanoid": len(hum),
            "vertices": n, "triangles": len(M.T), "materials": len(materials), "morph_targets": len(targets),
            "expressions": {"preset": sorted(preset_out), "custom": len(custom)},
            "springs": len(springs), "spring_joints": sum(len(s["joints"]) for s in springs),
            "colliders": len(colliders), "constraints": len(constraints), "warnings": warn}


def _material_binds(pmx, m, materials, w):
    """VRM material colour binds of a PMX material morph at weight w: its diffuse and ambient as the lit colour (and
    alpha), its edge colour as the outline's."""
    out = []
    for o in m.offsets:
        idx = int(o[0])
        mats = range(len(pmx.materials)) if idx < 0 else ([idx] if idx < len(pmx.materials) else [])
        for mi in mats:
            pm = pmx.materials[mi]
            dif, amb = np.asarray(pm.diffuse, float), np.asarray(pm.ambient, float)
            if int(o[1]) == 0:
                dif2, amb2 = dif * np.asarray(o[2], float), amb * np.asarray(o[5], float)
                edge2 = np.asarray(pm.edge_color, float) * np.asarray(o[6], float)
            else:
                dif2, amb2 = dif + np.asarray(o[2], float), amb + np.asarray(o[5], float)
                edge2 = np.asarray(pm.edge_color, float) + np.asarray(o[6], float)
            base = materials[mi]["pbrMetallicRoughness"]["baseColorFactor"]
            lit = np.clip(amb2 + MMD_LIGHT * dif2[:3], 0.0, 1.0)
            to = np.concatenate([_lin(lit), [float(np.clip(dif2[3], 0.0, 1.0))]])
            to = np.asarray(base) + w * (to - np.asarray(base))
            out.append({"material": mi, "type": "color", "targetValue": _r(to)})
            if pmx.materials[mi].edge:
                oc = materials[mi]["extensions"]["VRMC_materials_mtoon"].get("outlineColorFactor", [0.0, 0.0, 0.0])
                to_e = np.asarray(oc) + w * (_lin(edge2[:3]) - np.asarray(oc))
                out.append({"material": mi, "type": "outlineColor", "targetValue": [*_r(to_e), 1.0]})
    return out


def _mk_rig(pmx, M, G, R3, heads, tails, tpose, unit, names, target_of, chains, springs, grants, mats, folds, cloth,
            doc, warn):
    """The MK_rig extension (module docstring): the T-pose's positions and directions in glTF axes, nodes by index."""
    nb = len(M.bones)
    morph_target = {pmx.morphs[i].name: t for i, t in target_of.items()}
    bones = []
    for b, bone in enumerate(pmx.bones):
        e = {"node": b, "jp": bone.name, "en": bone.name_en, "layer": int(bone.layer),
             "flags": {k: bool(getattr(bone, k)) for k in ("rotatable", "movable", "visible", "controllable",
                                                         "after_physics")}}
        if bone.tail_offset is None and 0 <= bone.tail_bone < nb:
            e["tail_node"] = int(bone.tail_bone)
        else:
            e["tail"] = _r(_gl(tails[b] - heads[b]))
        if bone.fixed_axis is not None:
            e["fixed_axis"] = _r(_gl(R3[b] @ to_model(np.asarray(bone.fixed_axis, float), 1.0)[0]))
        if bone.local_x is not None and bone.local_z is not None:
            e["local_x"] = _r(_gl(R3[b] @ to_model(np.asarray(bone.local_x, float), 1.0)[0]))
            e["local_z"] = _r(_gl(R3[b] @ to_model(np.asarray(bone.local_z, float), 1.0)[0]))
        bones.append(e)
    pmx_rest = []
    for b, Rl in sorted(tpose.items()):
        p = int(M.parents[b])
        Qp = R3[p] if p >= 0 else np.eye(3)
        pmx_rest.append({"node": b, "rotation": _quat(Qp @ Rl.T @ Qp.T)})
    ik = [{"node": b, "target": int(bone.ik.target), "loops": int(bone.ik.loops), "angle": round(float(bone.ik.angle), 6),
           "links": [{"node": int(link[0]), "limit": [_r(link[2]), _r(link[3])] if link[1] else None}
                     for link in bone.ik.links]}
          for b, bone in enumerate(pmx.bones) if bone.ik is not None]
    morphs = []
    for i, m in enumerate(pmx.morphs):
        e = {"jp": m.name, "en": m.name_en, "panel": int(m.panel), "kind": m.kind}
        if i in target_of:
            e["target"] = target_of[i]
        elif m.kind == "group" or m.kind == "flip":
            e["offsets"] = [[int(j), round(float(r), 6)] for j, r in m.offsets]
        elif m.kind == "bone":
            e["offsets"] = [{"node": int(o[0]), "move": _r(_gl(to_model(np.asarray(o[1], float), unit)[0])),
                             "turn": _quat(_pmx_quat(o[2]))} for o in m.offsets]
        elif m.kind == "material":
            e["offsets"] = [[int(o[0]), int(o[1]), *[_r(x) if isinstance(x, (tuple, list)) else round(float(x), 6)
                                                     for x in o[2:]]] for o in m.offsets]
        elif m.kind.startswith("uv"):
            e["offsets"] = [[int(o[0]), _r(o[1])] for o in m.offsets]
        else:
            e["offsets"] = [[int(o[0]), int(o[1]), _r(o[2]), _r(o[3])] for o in m.offsets]
        morphs.append(e)
    bodies = []
    for body in pmx.bodies:
        b = int(body.bone)
        Gb = G[b] if 0 <= b < nb else np.eye(4)
        c = to_model(np.asarray(body.pos, float), unit)[0]
        R = Gb[:3, :3] @ _body_matrix(body.rot)
        bodies.append({"jp": body.name, "en": body.name_en, "node": b if 0 <= b < nb else None,
                       "shape": ("sphere", "box", "capsule")[int(body.shape)],
                       "size": _r(np.asarray(body.size, float) * unit), "center": _r(_gl(Gb[:3, :3] @ c + Gb[:3, 3])),
                       "rotation": _quat(R), "group": int(body.group), "mask": int(body.mask),
                       "mass": round(float(body.mass), 6), "damping": [round(float(body.linear_damping), 6),
                                                                      round(float(body.angular_damping), 6)],
                       "restitution": round(float(body.restitution), 6), "friction": round(float(body.friction), 6),
                       "mode": ("bone", "physics", "physics_bone")[int(body.mode)]})
    joints = []
    for j in pmx.joints:
        a = int(j.body_a)
        b = int(pmx.bodies[a].bone) if 0 <= a < len(pmx.bodies) else -1
        Gb = G[b] if 0 <= b < nb else np.eye(4)
        c = to_model(np.asarray(j.pos, float), unit)[0]
        joints.append({"jp": j.name, "en": j.name_en, "kind": int(j.kind), "bodies": [int(j.body_a), int(j.body_b)],
                       "center": _r(_gl(Gb[:3, :3] @ c + Gb[:3, 3])), "rotation": _quat(Gb[:3, :3] @ _body_matrix(j.rot)),
                       "move": [_r(np.asarray(j.move_lo, float) * unit), _r(np.asarray(j.move_hi, float) * unit)],
                       "turn": [_r(j.rot_lo), _r(j.rot_hi)], "spring_move": _r(j.spring_move),
                       "spring_turn": _r(j.spring_rot)})
    root_of = {}
    for k, (bs, _) in enumerate(chains):
        for b in bs:
            root_of[b] = k
    ties = {}
    for j in pmx.joints:
        ab = [int(pmx.bodies[x].bone) if 0 <= x < len(pmx.bodies) else -1 for x in (j.body_a, j.body_b)]
        ka, kb = root_of.get(ab[0]), root_of.get(ab[1])
        if ka is not None and kb is not None and ka != kb:
            ties.setdefault(ka, set()).add(kb)
            ties.setdefault(kb, set()).add(ka)
    chain_out = [{"spring": k, "family": springs[k]["_family"], "nodes": bs, **({"ties": sorted(ties[k])} if k in ties else {})}
                 for k, (bs, _) in enumerate(chains)]
    by_name = {nm: i for i, nm in enumerate(M.bones)}

    def node_of(nm, where):
        if nm not in by_name:
            raise ValueError(f"{where}: no bone {nm!r} in the model")
        return by_name[nm]
    fold_out = []
    for f in folds:
        d = dict(f["driver"])
        bone, ref = node_of(d["bone"], f"fold {f['morph']}"), node_of(d["ref"], f"fold {f['morph']}")
        key = "dir" if d["kind"] == "angle" else "axis"
        d.update(bone=bone, ref=ref)
        d[key] = _r(_gl(R3[ref] @ np.asarray(d[key], float)))
        e = {"morph": f["morph"], "piece": f.get("piece"), "driver": d}
        if f["morph"] in morph_target:
            e["target"] = morph_target[f["morph"]]
        else:
            warn.append(f"fold {f['morph']}: no such vertex morph")
        fold_out.append(e)
    cloth_out = []
    for c in cloth:
        bones_c = [node_of(b, f"cloth {c['name']}") for b in c["bones"]]
        e = {"name": c["name"], "materials": [next((i for i, m in enumerate(pmx.materials) if m.name == nm), None)
                                              for nm in c["materials"]],
             "nodes": bones_c, "settings": dict(c.get("settings") or {}), "arrays": {}}

        def skinned(P, Wd, Gc):
            P, Wd = np.asarray(P, float), np.asarray(Wd, float)
            s = np.maximum(Wd.sum(axis=1, keepdims=True), 1e-12)
            out = np.einsum("nb,bij,nj->ni", Wd / s, Gc[:, :3, :3], P) + (Wd / s) @ Gc[:, :3, 3]
            return np.where(Wd.sum(axis=1, keepdims=True) > 0, out, P)
        pts = c.get("points") or {}
        arrays = {"sim": _gl(skinned(pts["sim"], c["sim_weights"], G[bones_c])).astype(np.float32),
                  "collide": _gl(skinned(pts["collide"], c["collide_weights"], G[bones_c])).astype(np.float32)}
        arrays.update({k: np.asarray(c[k]) for k in RG.CLOTH_ARRAYS if k in c})
        for k, a in arrays.items():
            a = np.ascontiguousarray(a)
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            if k in ("sim_faces", "collide_faces"):
                a = a.astype(np.int32)
            cols = a.shape[1] if a.ndim == 2 else 1
            e["arrays"][k] = {"bufferView": doc.bin.view(a.tobytes()), "dtype": str(a.dtype), "shape": list(a.shape),
                              "columns": int(cols)}
        cloth_out.append(e)
    frames = [{"jp": f.name, "en": f.name_en, "special": bool(f.special),
               "items": [["bone", int(i)] if kind == "bone" else ["morph", pmx.morphs[int(i)].name]
                         for kind, i in f.items]} for f in pmx.frames]
    return {"schema": 1, "unit": unit, "source": {"jp": pmx.name, "en": pmx.name_en, "comment": pmx.comment,
                                                  "comment_en": pmx.comment_en},
            "rest": {"pose": "tpose", "pmx_rest": pmx_rest}, "bones": bones, "ik": ik, "grants": grants,
            "morphs": morphs, "materials": mats, "physics": {"bodies": bodies, "joints": joints}, "chains": chain_out,
            "folds": fold_out, "cloth": cloth_out, "frames": frames}


def _pmx_quat(q):
    """Model-space rotation of a PMX bone-morph quaternion (x, y, z, w): PMX's left-handed axes swap y and z."""
    R = Rotation.from_quat(np.asarray(q, float)).as_matrix()
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    return P @ R @ P


def rig_layers(rig_path):
    """The model layers of a written rig.json, as `write` takes them: bones by PMX name, a cloth copy's arrays read
    back from the .npz rig.json names (beside it)."""
    rig_path = Path(rig_path)
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    jp = {name: b.get("jp") or name for name, b in rig.get("bones", {}).items()}
    out = {}
    if rig.get("folds"):
        out["folds"] = [dict(f, driver=dict(f["driver"], bone=jp.get(f["driver"]["bone"], f["driver"]["bone"]),
                                            ref=jp.get(f["driver"]["ref"], f["driver"]["ref"]))) for f in rig["folds"]]
    entries = []
    for c in rig.get("cloth") or []:
        f = rig_path.parent / c["file"]
        with np.load(f) as z:
            arr = {k: z[k] for k in c.get("arrays", [])}
        e = {k: v for k, v in c.items() if k not in ("file", "arrays", "sha1")}
        e["bones"] = [jp.get(b, b) for b in c["bones"]]
        e["points"] = {k: arr[f"{c['name']}.{k}"] for k in ("sim", "collide") if f"{c['name']}.{k}" in arr}
        e.update({k: arr[f"{c['name']}.{k}"] for k in RG.CLOTH_ARRAYS if f"{c['name']}.{k}" in arr})
        entries.append(e)
    if entries:
        out["cloth"] = entries
    return out
