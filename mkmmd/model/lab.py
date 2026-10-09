"""The model lab: quick looks at a character in-process, without Blender. Views and pose sheets posed with the model's
own weights, several models side by side at one scale with their numbers, and a red line drawn on a sheet (or on a
screenshot of part of one) read back in millimetres.

A source is a spec (built in-process: the part builders, then the assembler, so the lab sees the PMX `mk model build`
writes) or a .pmx file. Model space throughout: metres, Z up, -Y forward, +X her left.

  load(src, label=None, overrides=(), parts=None, unit=0.08, workdir=None) -> Model
  Model.deform(pose=None, morphs=None) -> (V, N)
        forward kinematics through the bone tree (rotation grants followed: D bones, twist bones) and linear blend
        skinning with the model's weights. Poses are named by semantic bone (POSES: finger curls, spread, arms down,
        T-pose, sit, both sides); a pose name that is one of the model's morphs applies that morph instead.
  region_mask(model, region, side)       the vertices of a semantic bone's subtree (REGIONS), the whole body for "body"
  render(V, N, T, lit, shade, cam, ...)  orthographic toon render: a vectorised z-buffer (triangles bucketed by their
                                         size in pixels, fragments resolved with np.maximum.at), two-tone shading,
                                         outlines where the depth jumps
  measure(model, region, side, V)        lengths, widths and girths in mm from exact cross-sections
  sheet(models, region, ...)             one scale for every model; the layout records each cell's camera
  trace(marked, sheet_png, layout)       place a marked image on the sheet (ORB features, else multi-scale template
                                         matching), then the red stroke in model space and its distance to the outline
  trace_points(pts, sheet_png, layout)   the same for a stroke already in sheet pixels (a line drawn over the sheet in
                                         a Tern whiteboard; mkmmd.review reads those)
"""
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core import bonemap
from . import assemble as AS
from . import build as BD
from . import pmx_io
from . import spec as SP
from .pmx_take import _find_file, to_model

BG = (222, 224, 230)                      # sheet and cell background: cool grey, clear of pale skin
LINE = (120, 80, 75)                      # outline colour
SHADE = np.array([0.90, 0.80, 0.80])      # the toon shadow is the lit colour times this
LABEL_H = 20                              # label strip above each cell (px)

# region -> the semantic bone whose subtree it is (sided regions add .L / .R); "body" is every vertex
REGIONS = {"body": None, "head": "head", "hand": "wrist", "foot": "ankle", "arm": "arm", "leg": "leg"}
SIDED = ("hand", "foot", "arm", "leg")
VIEWS = {"body": ("front", "outer", "back", "3q"), "head": ("front", "3q", "outer", "back"),
         "hand": ("back", "palm", "thumb", "3q"), "foot": ("outer", "top", "front", "sole"),
         "arm": ("front", "outer", "back", "3q"), "leg": ("front", "outer", "back", "3q")}

# finger curls: degrees at the three joints of each finger (+ towards the palm); spread: degrees about the palm normal
# at the root (+ towards the little finger)
HAND_POSES = {
    "relaxed": ({"index": (14, 22, 10), "middle": (16, 24, 12), "ring": (18, 26, 13), "little": (20, 28, 14),
                 "thumb": (4, 10, 12)}, None),
    "curled": ({"index": (40, 55, 35), "middle": (42, 58, 36), "ring": (44, 60, 38), "little": (46, 62, 40),
                "thumb": (8, 25, 30)}, None),
    "fist": ({"index": (80, 95, 60), "middle": (82, 97, 62), "ring": (84, 98, 62), "little": (86, 100, 62),
              "thumb": (20, 40, 45)}, None),
    "spread": ({}, {"index": 10, "middle": 0, "ring": -8, "little": -18}),
}
ARMS_DOWN_DEG = 75.0                      # arms_down: the upper arm this far below horizontal
THUMB_TOWARDS = (1.0, 0.5)                # the thumb flexes towards this mix of (across to the little finger, out of the
                                          # palm): it folds over the fingers in a fist (it is turned ~90 deg from them)
POSES = ("rest",) + tuple(HAND_POSES) + ("arms_down", "tpose", "sit")


# --------------------------------------------------------------------------------------------------------- small maths
def unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def rot(axis, deg):
    """Rotation matrix about `axis` by `deg` (right hand)."""
    a = unit(axis)
    t = math.radians(deg)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def _scaled(R, k):
    """`k` of rotation R (same axis, k times the angle)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec() * float(k)).as_matrix()


def view_basis(towards, up):
    """(right, up, towards the camera) of a camera on the `towards` side looking back at the target, `up` kept up."""
    c = unit(towards)
    r = np.cross(np.asarray(up, float), c)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(np.array([0.0, 1.0, 0.0]), c)
    r = unit(r)
    return r, np.cross(c, r), c


# --------------------------------------------------------------------------------------------------------------- model
@dataclass
class Model:
    """A model as the lab uses it (arrays in model space; see `from_pmx`)."""
    label: str
    source: str
    V: np.ndarray            # (n, 3) rest positions (m)
    N: np.ndarray            # (n, 3) rest normals
    uv: np.ndarray           # (n, 2) texture coordinates, u right and v down (PMX and glTF agree)
    T: np.ndarray            # (t, 3) triangles
    tri_mat: np.ndarray      # (t,) material index
    lit: np.ndarray          # (m, 3) lit colour per material, 0..255 (diffuse times the mean of its texture)
    rgba: np.ndarray         # (m, 4) diffuse colour and alpha per material, 0..1
    textures: list           # per material: its texture's absolute path, or None (none, missing, or a spec built in a
                             # folder that is gone: see `load`)
    double_sided: np.ndarray  # (m,) materials drawn from both sides
    hidden: np.ndarray       # (m,) materials not drawn (diffuse alpha ~ 0)
    mat_names: list
    bones: list              # bone names
    heads: np.ndarray        # (b, 3)
    tails: np.ndarray        # (b, 3)
    parents: np.ndarray      # (b,) -1 = none
    grants: list             # per bone: (grant parent, ratio) for rotation grants, else None
    order: list              # bones with parents and grant parents first
    wi: np.ndarray           # (n, 4) bone index per weight slot
    ww: np.ndarray           # (n, 4) weights (rows sum to 1, or 0 for unweighted vertices)
    morphs: dict             # morph name -> (vertex indices, offsets (k, 3) m); group morphs expanded
    morph_panels: dict       # morph name -> its PMX panel: 1 eyebrow, 2 eye, 3 mouth, 4 other
    sem: dict                # semantic name -> bone index
    names: tuple = ("", "")  # the PMX model name, Japanese and English

    @property
    def visible(self):
        """(n,) vertices used by a drawn material."""
        v = np.zeros(len(self.V), bool)
        v[self.T[~self.hidden[self.tri_mat]].ravel()] = True
        return v

    @property
    def dominant(self):
        """(n,) the bone with the largest weight per vertex."""
        return self.wi[np.arange(len(self.V)), np.argmax(self.ww, axis=1)]

    def head(self, semantic):
        b = self.sem.get(semantic)
        if b is None:
            raise ValueError(f"{self.label}: no {semantic} bone")
        return self.heads[b]

    def subtree(self, b):
        """Bone indices under `b`: its descendants, plus bones that take their rotation from one of them (D bones)."""
        kids, granted = defaultdict(list), defaultdict(list)
        for i, p in enumerate(self.parents):
            kids[int(p)].append(i)
        for i, g in enumerate(self.grants):
            if g is not None:
                granted[g[0]].append(i)
        out, stack = set(), [int(b)]
        while stack:
            i = stack.pop()
            if i not in out:
                out.add(i)
                stack += kids[i] + granted[i]
        return out

    def matrices(self, rots):
        """(b, 4, 4) bone transforms for local rotations `rots` {bone: 3x3} about each bone's rest head; a rotation
        grant adds `ratio` of its parent's rotation."""
        G = np.repeat(np.eye(4)[None], len(self.bones), axis=0)
        eff = {}
        for b in self.order:
            R = rots.get(b)
            g = self.grants[b]
            if g is not None and g[0] in eff:
                Rg = _scaled(eff[g[0]], g[1])
                R = Rg if R is None else Rg @ R
            p = int(self.parents[b])
            base = G[p] if p >= 0 else np.eye(4)
            if R is None:
                G[b] = base
                continue
            eff[b] = R
            L = np.eye(4)
            L[:3, :3] = R
            L[:3, 3] = self.heads[b] - R @ self.heads[b]
            G[b] = base @ L
        return G

    def deform(self, pose=None, morphs=None):
        """(V, N) after the morphs {name: weight} and the pose (a POSES name, None / "rest", or a morph name)."""
        morphs = dict(morphs or {})
        if pose not in (None, "rest") and pose not in POSES:
            if pose not in self.morphs:
                raise ValueError(f"{self.label}: no pose or morph {pose!r}; poses: {', '.join(POSES)}")
            morphs[pose] = morphs.get(pose, 0.0) + 1.0
            pose = None
        V = self.V.copy()
        for name, w in morphs.items():
            if name not in self.morphs:
                raise ValueError(f"{self.label}: no morph {name!r}")
            idx, d = self.morphs[name]
            np.add.at(V, idx, float(w) * d)
        rots = pose_rotations(self, pose) if pose not in (None, "rest") else {}
        if not rots:
            return V, self.N.copy()
        return self.skin(V, self.N, self.matrices(rots))

    def skin(self, V, N, G):
        """Linear blend skinning of positions V and normals N by bone transforms G (from `matrices`); unweighted
        vertices stay."""
        out, nout = np.zeros_like(V), np.zeros_like(N)
        for k in range(4):
            w = self.ww[:, k]
            m = w > 0
            if not m.any():
                continue
            M = G[self.wi[m, k]]
            out[m] += w[m, None] * (np.einsum("nij,nj->ni", M[:, :3, :3], V[m]) + M[:, :3, 3])
            nout[m] += w[m, None] * np.einsum("nij,nj->ni", M[:, :3, :3], N[m])
        free = self.ww.sum(axis=1) <= 0
        out[free], nout[free] = V[free], N[free]
        return out, nout / np.maximum(np.linalg.norm(nout, axis=1, keepdims=True), 1e-12)


def _mean_texture(path):
    """Mean RGB (0..1) of an image's opaque texels (alpha > 0.5; all texels when none is)."""
    return _mean_texture_at(str(path), Path(path).stat().st_mtime)


@lru_cache(maxsize=512)
def _mean_texture_at(path, mtime):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGBA").resize((64, 64), Image.BILINEAR), float) / 255.0
    op = a[..., 3] > 0.5
    return (a[op][:, :3] if op.any() else a[..., :3].reshape(-1, 3)).mean(axis=0)


def _order(parents, grants):
    """Bone indices with parents and grant parents first (stable)."""
    out, state = [], {}

    def visit(i):
        if state.get(i) == 2:
            return
        if state.get(i) == 1:                       # a cycle: leave the rest of it in file order
            return
        state[i] = 1
        for j in (int(parents[i]), grants[i][0] if grants[i] is not None else -1):
            if 0 <= j < len(parents):
                visit(j)
        state[i] = 2
        out.append(i)
    for i in range(len(parents)):
        visit(i)
    return out


def from_pmx(pmx, root, unit=0.08, label="model", source=""):
    """A Model from a PmxModel; `root` is the folder its texture paths are relative to, `unit` metres per PMX unit."""
    n = len(pmx.vertices)
    V = to_model(np.array([v.pos for v in pmx.vertices], float).reshape(-1, 3), unit)
    N = to_model(np.array([v.normal for v in pmx.vertices], float).reshape(-1, 3), 1.0)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    uv = np.array([v.uv for v in pmx.vertices], float).reshape(-1, 2)
    T = np.asarray(pmx.faces, np.int64).reshape(-1, 3)
    counts = [int(m.index_count) // 3 for m in pmx.materials]
    tri_mat = np.repeat(np.arange(len(counts)), counts)
    lit, textures, hidden = [], [], []
    for m in pmx.materials:
        col = np.array(m.diffuse[:3], float)
        f = _find_file(root, pmx.textures[m.texture]) if 0 <= m.texture < len(pmx.textures) else None
        if f is not None:
            col = col * _mean_texture(f)
        lit.append(np.clip(col, 0.0, 1.0) * 255.0)
        textures.append(str(Path(f).resolve()) if f is not None else None)
        hidden.append(float(m.diffuse[3]) < 0.05)
    names = [b.name for b in pmx.bones]
    heads = to_model(np.array([b.pos for b in pmx.bones], float).reshape(-1, 3), unit)
    tails = heads.copy()
    for i, b in enumerate(pmx.bones):
        if b.tail_offset is not None:
            tails[i] = heads[i] + to_model(np.array(b.tail_offset, float), unit)[0]
        elif 0 <= b.tail_bone < len(names):
            tails[i] = heads[b.tail_bone]
    parents = np.array([b.parent if 0 <= b.parent < len(names) else -1 for b in pmx.bones], int)
    grants = [(int(b.grant_parent), float(b.grant_ratio)) if b.grant_rotate and 0 <= b.grant_parent < len(names)
              else None for b in pmx.bones]
    wi, ww = np.zeros((n, 4), np.int64), np.zeros((n, 4))
    for i, v in enumerate(pmx.vertices):
        for k, (b, w) in enumerate(zip(v.bones[:4], v.weights[:4])):
            if b >= 0 and w > 0:
                wi[i, k], ww[i, k] = b, w
    tot = ww.sum(axis=1, keepdims=True)
    ww = np.where(tot > 0, ww / np.maximum(tot, 1e-12), 0.0)
    morphs = {}
    for mi, m in enumerate(pmx.morphs):
        parts, stack = [], [(mi, 1.0, (mi,))]
        while stack:
            j, w, seen = stack.pop()
            mj = pmx.morphs[j]
            if mj.kind == "vertex" and mj.offsets:
                idx = np.array([o[0] for o in mj.offsets], np.int64)
                parts.append((idx, w * to_model(np.array([o[1] for o in mj.offsets], float), unit)))
            elif mj.kind == "group":
                stack += [(int(k), w * float(r), seen + (int(k),)) for k, r in mj.offsets
                          if 0 <= int(k) < len(pmx.morphs) and int(k) not in seen]
        if parts:
            morphs[m.name] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    sem = {s: names.index(b) for s, b in bonemap.build_map({nm: nm for nm in names}).items() if b in names}
    return Model(label=label, source=source, V=V, N=N, uv=uv, T=T, tri_mat=tri_mat, lit=np.array(lit).reshape(-1, 3),
                 rgba=np.array([m.diffuse[:4] for m in pmx.materials], float).reshape(-1, 4), textures=textures,
                 double_sided=np.array([bool(m.double_sided) for m in pmx.materials], bool),
                 hidden=np.array(hidden, bool), mat_names=[m.name for m in pmx.materials], bones=names, heads=heads,
                 tails=tails, parents=parents, grants=grants, order=_order(parents, grants), wi=wi, ww=ww,
                 morphs=morphs, morph_panels={m.name: int(m.panel) for m in pmx.morphs if m.name in morphs}, sem=sem,
                 names=(pmx.name, pmx.name_en))


def load(src, label=None, overrides=(), parts=None, unit=0.08, log=None, workdir=None):
    """A Model from a .pmx file (`unit` metres per PMX unit) or a spec (a path or base:NAME): the spec's `parts`
    (default all, plus what they need) are built and assembled as `mk model build` does, in `workdir` (kept: the
    model's `textures` are the files there) or in a temporary folder removed on return (its `textures` are then None)."""
    s = str(src)
    if s.lower().endswith(".pmx"):
        path = Path(s).expanduser()
        if not path.is_file():
            raise ValueError(f"{path} does not exist")
        return from_pmx(pmx_io.read(path), path.parent, unit, label or path.stem, str(path))
    spec = SP.load(s, list(overrides))
    cfg = SP.model_cfg(spec)

    def build(folder):
        built = BD.run(spec, only=parts, tex_dir=Path(folder) / AS.TEX_DIR, log=log)
        asm = AS.assemble(built, name=cfg["name"], scale=cfg["scale"])
        return from_pmx(asm.pmx, Path(folder), cfg["scale"], label or cfg["name"], s)
    if workdir is not None:
        Path(workdir).expanduser().mkdir(parents=True, exist_ok=True)
        return build(Path(workdir).expanduser())
    with tempfile.TemporaryDirectory(prefix="mk_lab_") as tmp:
        model = build(tmp)
    model.textures = [None] * len(model.textures)
    return model


# ------------------------------------------------------------------------------------------------------------- posing
def hand_axes(model, side):
    """(along the hand, across towards the little finger, out of the palm) from the wrist and the knuckle bones."""
    w, m1 = model.head(f"wrist.{side}"), model.head(f"middle1.{side}")
    i1, l1 = model.head(f"index1.{side}"), model.head(f"little1.{side}")
    a = unit(m1 - w)
    rr = l1 - i1
    r = unit(rr - (rr @ a) * a)
    return a, r, (np.cross(r, a) if side == "L" else np.cross(a, r))


def _finger_rotations(model, side, deg, spread):
    a, r, n = hand_axes(model, side)
    sgn = 1.0 if side == "L" else -1.0                  # a mirrored hand turns the other way about its own normal
    thumb_to = unit(THUMB_TOWARDS[0] * r + THUMB_TOWARDS[1] * n)
    out = {}
    for f, chain in bonemap.FINGERS.items():
        idx = [model.sem.get(f"{c}.{side}") for c in chain]
        if any(i is None for i in idx[:3]):
            continue
        pts = [model.heads[i] for i in idx[:3]] + [model.heads[idx[3]] if idx[3] is not None else model.tails[idx[2]]]
        d_prev = a
        for k in range(3):
            d = pts[k + 1] - pts[k]
            d = unit(d) if np.linalg.norm(d) > 1e-6 else d_prev
            d_prev = d
            R = np.eye(3)
            ang = (deg.get(f) or (0, 0, 0))[k]
            if ang:
                R = rot(np.cross(d, thumb_to if f == "thumb" else n), ang)
            if k == 0 and spread and spread.get(f):
                R = rot(n, sgn * spread[f]) @ R
            if not np.allclose(R, np.eye(3)):
                out[idx[k]] = R
    return out


def _below(d):
    """Degrees a direction points below horizontal, measured in her frontal plane (out to her side, down)."""
    return math.degrees(math.atan2(-d[2], abs(d[0])))


def pose_rotations(model, name):
    """{bone: 3x3} local rotations of a POSES pose, both sides. arms_down lowers the upper arms to ARMS_DOWN_DEG below
    horizontal; tpose raises them to horizontal and straightens the elbows so the forearms run on level with them."""
    out = {}
    for side in ("L", "R"):
        sgn = 1.0 if side == "L" else -1.0
        if name in HAND_POSES:
            if model.sem.get(f"wrist.{side}") is None:
                continue
            out.update(_finger_rotations(model, side, *HAND_POSES[name]))
        elif name in ("arms_down", "tpose"):
            i, e = model.sem.get(f"arm.{side}"), model.sem.get(f"elbow.{side}")
            if i is None or e is None:
                continue
            target = ARMS_DOWN_DEG if name == "arms_down" else 0.0
            below = _below(model.heads[e] - model.heads[i])
            out[i] = rot((0.0, 1.0, 0.0), sgn * (target - below))      # both turn about the forward axis, so the
            w = model.sem.get(f"wrist.{side}")                       # elbow's turn adds to the arm's
            if name == "tpose" and w is not None:
                out[e] = rot((0.0, 1.0, 0.0), sgn * (below - _below(model.heads[w] - model.heads[e])))
        elif name == "sit":
            for b, deg in ((f"leg.{side}", -90.0), (f"knee.{side}", 90.0)):
                if model.sem.get(b) is not None:
                    out[model.sem[b]] = rot((1.0, 0.0, 0.0), deg)
        else:
            raise ValueError(f"no pose {name!r}; poses: {', '.join(POSES)}")
    return out


# ------------------------------------------------------------------------------------------------------------ regions
def region_mask(model, region, side="L"):
    """(n,) the region's drawn vertices: those whose largest weight is on its bone's subtree (every drawn vertex for
    "body")."""
    if region not in REGIONS:
        raise ValueError(f"no region {region!r}; regions: {', '.join(REGIONS)}")
    vis = model.visible
    root = REGIONS[region]
    if root is None:
        return vis
    name = f"{root}.{side}" if region in SIDED else root
    b = model.sem.get(name)
    if b is None:
        raise ValueError(f"{model.label}: no {name} bone for the {region} region")
    return vis & np.isin(model.dominant, sorted(model.subtree(b)))


def view_dirs(model, region, side="L"):
    """{view: (towards the camera, up)} for the region; "outer" / "inner" are her side's outside / inside."""
    sx = 1.0 if side == "L" else -1.0
    Z = np.array([0.0, 0.0, 1.0])
    if region == "hand":
        a, r, n = hand_axes(model, side)
        return {"back": (-n, a), "palm": (n, a), "thumb": (-r, a), "little": (r, a), "3q": (unit(-n - r), a),
                "tip": (a, -n)}
    return {"front": (np.array([0.0, -1.0, 0.0]), Z), "back": (np.array([0.0, 1.0, 0.0]), Z),
            "outer": (np.array([sx, 0.0, 0.0]), Z), "inner": (np.array([-sx, 0.0, 0.0]), Z),
            "3q": (unit((0.71 * sx, -0.71, 0.0)), Z), "top": (Z, np.array([0.0, 1.0, 0.0])),
            "sole": (-Z, np.array([0.0, -1.0, 0.0]))}


def view_label(view, side="L"):
    if view in ("outer", "inner"):
        left = (view == "outer") == (side == "L")
        return "left side" if left else "right side"
    return view


def region_axis(model, region, side="L"):
    """(origin, unit axis) along which a region is measured: the hand from the wrist towards the fingers, a limb from its
    root joint, the foot forwards from the ankle, body and head upwards."""
    if region == "hand":
        return model.head(f"wrist.{side}"), hand_axes(model, side)[0]
    if region == "arm":
        o = model.head(f"arm.{side}")
        return o, unit(model.head(f"wrist.{side}") - o)
    if region == "leg":
        o = model.head(f"leg.{side}")
        return o, unit(model.head(f"ankle.{side}") - o)
    if region == "foot":
        return model.head(f"ankle.{side}"), np.array([0.0, -1.0, 0.0])
    if region == "head":
        return model.head("head"), np.array([0.0, 0.0, 1.0])
    return np.zeros(3), np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------- rendering
def _fragments(tx, ty, tz, i0, j0, i1, j1, det, k, eps=1e-6):
    """Pixels inside triangles whose bounding boxes fit in k x k: (triangle row, pixel x, pixel y, depth, b1, b2)."""
    ox, oy = np.meshgrid(np.arange(k), np.arange(k))
    px = i0[:, None] + ox.ravel()[None]
    py = j0[:, None] + oy.ravel()[None]
    ok = (px <= i1[:, None]) & (py <= j1[:, None])
    gx, gy = px + 0.5, py + 0.5
    X0, X1, X2 = tx[:, 0:1], tx[:, 1:2], tx[:, 2:3]
    Y0, Y1, Y2 = ty[:, 0:1], ty[:, 1:2], ty[:, 2:3]
    b1 = ((gx - X0) * (Y2 - Y0) - (X2 - X0) * (gy - Y0)) / det[:, None]
    b2 = ((X1 - X0) * (gy - Y0) - (gx - X0) * (Y1 - Y0)) / det[:, None]
    b0 = 1.0 - b1 - b2
    inside = ok & (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
    rows, cols = np.nonzero(inside)
    z = (b0 * tz[:, 0:1] + b1 * tz[:, 1:2] + b2 * tz[:, 2:3])[rows, cols]
    return rows, px[rows, cols], py[rows, cols], z, b1[rows, cols], b2[rows, cols]


def render(V, N, T, lit, shade, cam, size=360, ss=2, outline=True, depth=None, bg=BG):
    """Orthographic toon render. V, N (n, 3); T (t, 3) triangles; lit, shade (t, 3) colours 0..255 per triangle;
    cam {centre, right, up, towards, half} (half the frame's width in metres); depth (lo, hi) keeps the triangles whose
    centre lies in that range along `towards` (relative to the centre). Returns (PIL image, (size, size) bool mask)."""
    W = int(size) * int(ss)
    s = W / (2.0 * float(cam["half"]))
    c0 = np.asarray(cam["centre"], float)
    r, u, c = (np.asarray(cam[k], float) for k in ("right", "up", "towards"))
    P = np.asarray(V, float) - c0
    X, Y, Z = W / 2.0 + (P @ r) * s, W / 2.0 - (P @ u) * s, P @ c
    T = np.asarray(T, np.int64)
    tx, ty, tz = X[T], Y[T], Z[T]
    keep = np.ones(len(T), bool)
    if depth is not None:
        zc = tz.mean(axis=1)
        keep &= (zc >= depth[0]) & (zc <= depth[1])
    i0 = np.maximum(np.ceil(tx.min(axis=1) - 0.5), 0).astype(np.int64)
    i1 = np.minimum(np.floor(tx.max(axis=1) - 0.5), W - 1).astype(np.int64)
    j0 = np.maximum(np.ceil(ty.min(axis=1) - 0.5), 0).astype(np.int64)
    j1 = np.minimum(np.floor(ty.max(axis=1) - 0.5), W - 1).astype(np.int64)
    det = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
    keep &= (i0 <= i1) & (j0 <= j1) & (np.abs(det) > 1e-12)
    extent = np.maximum(i1 - i0, j1 - j0) + 1
    zbuf = np.full(W * W, -np.inf)
    tbuf = np.full(W * W, -1, np.int64)
    b1buf, b2buf = np.zeros(W * W), np.zeros(W * W)

    def put(idx, k):
        rows, px, py, z, b1, b2 = _fragments(tx[idx], ty[idx], tz[idx], i0[idx], j0[idx], i1[idx], j1[idx], det[idx], k)
        if not len(rows):
            return
        pix = py * W + px
        np.maximum.at(zbuf, pix, z)
        win = z >= zbuf[pix]
        tbuf[pix[win]], b1buf[pix[win]], b2buf[pix[win]] = idx[rows[win]], b1[win], b2[win]

    lo = 0
    for k in (1, 2, 4, 8, 16, 32, 64):
        idx = np.nonzero(keep & (extent > lo) & (extent <= k))[0]
        step = max(1, (1 << 22) // (k * k))
        for a in range(0, len(idx), step):
            put(idx[a:a + step], k)
        lo = k
    for t in np.nonzero(keep & (extent > lo))[0]:       # big triangles one at a time
        put(np.array([t]), int(extent[t]))

    img = np.empty((W * W, 3))
    img[:] = bg
    cov = tbuf >= 0
    if cov.any():
        t = tbuf[cov]
        b1, b2 = b1buf[cov], b2buf[cov]
        n = (1 - b1 - b2)[:, None] * N[T[t, 0]] + b1[:, None] * N[T[t, 1]] + b2[:, None] * N[T[t, 2]]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[n @ c < 0] *= -1.0                            # the back of an open sheet shades like its front
        light = unit(0.30 * r + 0.55 * u + 0.75 * c)
        d = n @ light
        kk = np.clip((d - 0.05) / 0.12, 0.0, 1.0)
        kk = kk * kk * (3 - 2 * kk)
        col = np.asarray(shade, float)[t] * (1 - kk)[:, None] + np.asarray(lit, float)[t] * kk[:, None]
        img[cov] = col * (0.92 + 0.08 * np.clip(d, 0, 1))[:, None]
    img = img.reshape(W, W, 3)
    if outline:
        z = np.where(np.isfinite(zbuf), zbuf, -1e3).reshape(W, W)
        thr = max(0.004, 4.0 / s)
        e = np.zeros((W, W), bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = z[:W - dy, max(0, -dx):W - max(0, dx)]
            b = z[dy:, max(0, dx):W + min(0, dx)]
            e[:W - dy, max(0, -dx):W - max(0, dx)] |= np.abs(a - b) > thr
        img[e] = LINE
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    mask = cov.reshape(W, W)
    if ss > 1:
        im = im.resize((size, size), Image.LANCZOS)
        mask = mask.reshape(size, ss, size, ss).mean(axis=(1, 3)) >= 0.5
    return im, mask


def scene_colours(model):
    """(T, lit, shade) of the drawn triangles."""
    keep = ~model.hidden[model.tri_mat]
    lit = model.lit[model.tri_mat[keep]]
    return model.T[keep], lit, lit * SHADE[None]


# ---------------------------------------------------------------------------------------------------------- measuring
def _welded(V, tol=1e-5):
    """Canonical vertex index per vertex: vertices split at UV seams (same position) share one."""
    _, inv = np.unique(np.round(np.asarray(V) / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.reshape(-1)


def sections(V, T, p, d, weld=None):
    """Loops where the plane through `p` with normal `d` cuts the triangles: a list of {"a", "b": (k, 3) segment ends,
    "points", "centroid", "perimeter"}. `weld`: canonical vertex ids (see _welded)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    wid = _welded(V) if weld is None else weld
    sd = (V - np.asarray(p, float)) @ unit(d)
    sd = np.where(sd == 0.0, 1e-12, sd)
    st = sd[T]
    cut = (st.min(axis=1) < 0) & (st.max(axis=1) > 0)
    Tc, sc = T[cut], st[cut]
    if not len(Tc):
        return []
    M, P, K = [], [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = Tc[:, i], Tc[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):      # edges that do not cross are dropped by M
            t = sc[:, i] / (sc[:, i] - sc[:, j])
            P.append(V[a] + t[:, None] * (V[b] - V[a]))
        M.append(sc[:, i] * sc[:, j] < 0)
        wa, wb = wid[a], wid[b]
        K.append(np.minimum(wa, wb) * (int(wid.max()) + 1) + np.maximum(wa, wb))
    M, P, K = np.stack(M, 1), np.stack(P, 1), np.stack(K, 1)
    two = M.sum(axis=1) == 2
    M, P, K = M[two], P[two], K[two]
    sel = np.argsort(~M, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(M))
    PA, PB = P[rows, sel[:, 0]], P[rows, sel[:, 1]]
    keys, inv = np.unique(np.concatenate([K[rows, sel[:, 0]], K[rows, sel[:, 1]]]), return_inverse=True)
    ka, kb = inv[:len(M)], inv[len(M):]
    g = coo_matrix((np.ones(len(M)), (ka, kb)), shape=(len(keys), len(keys)))
    _, lab = connected_components(g, directed=False)
    seg_lab = lab[ka]
    out = []
    for c in np.unique(seg_lab):
        m = seg_lab == c
        pts = np.concatenate([PA[m], PB[m]])
        out.append({"a": PA[m], "b": PB[m], "points": pts, "centroid": pts.mean(axis=0),
                    "perimeter": float(np.linalg.norm(PA[m] - PB[m], axis=1).sum())})
    return out


def _nearest(loops, target, d):
    """The loop to measure at `target`: the smallest one that encloses it in the cutting plane (normal d), so a limb
    is measured on its skin and not on the sleeve or skirt around it; else the loop whose centre is nearest."""
    if not loops:
        return None
    d = unit(d)
    e1 = unit(np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]))
    e2 = np.cross(d, e1)
    t = np.asarray(target, float)

    def encloses(lp):                                  # a ray along e1 crosses the loop's segments an odd number of times
        ax, ay = (lp["a"] - t) @ e1, (lp["a"] - t) @ e2
        bx, by = (lp["b"] - t) @ e1, (lp["b"] - t) @ e2
        span = (ay > 0) != (by > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = ax + (0.0 - ay) * (bx - ax) / (by - ay)
        return int(np.count_nonzero(span & (x > 0))) % 2 == 1

    inside = [lp for lp in loops if encloses(lp)]
    if inside:
        return min(inside, key=lambda lp: lp["perimeter"])

    def dist(lp):
        v = lp["centroid"] - t
        return float(np.linalg.norm(v - (v @ d) * d))
    return min(loops, key=dist)


def _extent(pts, axis):
    s = np.asarray(pts) @ unit(axis)
    return float(s.max() - s.min()) if len(s) else 0.0


def measure(model, region="body", side="L", V=None):
    """{name: value}: lengths, widths, depths and girths in mm (ratios are named *_ratio) of the region at `V`
    (default the rest pose). Measures whose bones the model lacks are left out. Cross-sections cut every triangle,
    hidden ones too (skin a dress always covers is often moved to an invisible material), and measure the innermost
    loop around the point: the skin, not the clothes over it."""
    V = model.V if V is None else V
    T = model.T
    weld = _welded(V)
    mask = region_mask(model, region, side)
    P = V[mask]
    out = {}
    mm = 1000.0

    def has(*names):
        return all(model.sem.get(n) is not None for n in names)

    def girth(name, p, d, target, wide=None, deep=None):
        lp = _nearest(sections(V, T, p, d, weld), target, d)
        if lp is None:
            return
        out[f"{name} girth"] = round(lp["perimeter"] * mm, 1)
        if wide is not None:
            out[f"{name} width"] = round(_extent(lp["points"], wide) * mm, 1)
        if deep is not None:
            out[f"{name} depth"] = round(_extent(lp["points"], deep) * mm, 1)

    X, Y, Z = np.eye(3)
    if region == "body":
        out["height"] = round(float(np.ptp(P[:, 2])) * mm, 1)
        if has("arm.L"):
            z = model.head("arm.L")[2]
            near = P[np.abs(P[:, 2] - z) < 0.01]
            if len(near):
                out["shoulder width"] = round(float(np.ptp(near[:, 0])) * mm, 1)
        if has("upper_body", "neck"):
            ub, nk = model.head("upper_body"), model.head("neck")
            for name, pt in (("chest", ub + 0.6 * (nk - ub)), ("waist", ub)):
                girth(name, pt, Z, pt, X, Y)
        if has("leg.L", "leg.R"):
            pt = 0.5 * (model.head("leg.L") + model.head("leg.R"))
            girth("hips", pt, Z, pt, X, Y)
        if has("leg.L", "knee.L", "ankle.L"):
            hip, knee, ank = model.head("leg.L"), model.head("knee.L"), model.head("ankle.L")
            for name, pt in (("thigh", hip + 0.30 * (knee - hip)), ("knee", knee), ("calf", knee + 0.35 * (ank - knee)),
                             ("ankle", ank + np.array([0.0, 0.0, 0.02]))):
                girth(name, pt, Z, pt, X)
        if has("arm.L", "elbow.L", "wrist.L"):
            sh, el, wr = model.head("arm.L"), model.head("elbow.L"), model.head("wrist.L")
            girth("upper arm", sh + 0.4 * (el - sh), el - sh, sh + 0.4 * (el - sh))
            girth("forearm", el + 0.35 * (wr - el), wr - el, el + 0.35 * (wr - el))
    elif region == "hand":
        w = model.head(f"wrist.{side}")
        a, r, n = hand_axes(model, side)
        out["hand length"] = round(float(((P - w) @ a).max()) * mm, 1)
        pl = float((model.head(f"middle1.{side}") - w) @ a)
        out["palm length"] = round(pl * mm, 1)
        out["finger / palm_ratio"] = round(out["hand length"] / max(out["palm length"], 1e-9) - 1.0, 3)
        for name, t in (("wrist", 0.0), ("mid palm", 0.5 * pl), ("knuckle line", pl - 0.002)):
            pt = w + t * a
            lp = _nearest(sections(V, T, pt, a, weld), pt, a)
            if lp is not None:
                out[f"{name} width"] = round(_extent(lp["points"], r) * mm, 1)
                out[f"{name} thickness"] = round(_extent(lp["points"], n) * mm, 1)
        dom = model.dominant
        for f, chain in bonemap.FINGERS.items():
            ids = [model.sem.get(f"{c}.{side}") for c in chain[:3]]
            if any(i is None for i in ids):
                continue
            fv = V[np.isin(dom, ids) & mask]
            if len(fv):
                root = model.heads[ids[0]]
                d = unit(model.heads[ids[1]] - root)
                out[f"{f} length"] = round(float(((fv - root) @ d).max()) * mm, 1)
    elif region == "head":
        for name, ax in (("head height", 2), ("head width", 0), ("head depth", 1)):
            out[name] = round(float(np.ptp(P[:, ax])) * mm, 1)
    elif region == "foot":
        out["foot length"] = round(float(np.ptp(P[:, 1])) * mm, 1)
        out["foot width"] = round(float(np.ptp(P[:, 0])) * mm, 1)
        out["ankle height"] = round(float(model.head(f"ankle.{side}")[2] - P[:, 2].min()) * mm, 1)
    elif region in ("arm", "leg"):
        o, ax = region_axis(model, region, side)
        out[f"{region} length"] = round(float(((P - o) @ ax).max()) * mm, 1)
        for t in (0.15, 0.35, 0.55, 0.75):
            pt = o + t * out[f"{region} length"] / mm * ax
            lp = _nearest(sections(V, T, pt, ax, weld), pt, ax)
            if lp is not None:
                out[f"girth at {int(t * 100)} %"] = round(lp["perimeter"] * mm, 1)
    return out


# -------------------------------------------------------------------------------------------------------------- sheets
@lru_cache(maxsize=4)
def _font(size=13):
    """A font with Japanese glyphs for the labels (morph and bone names): fontconfig's choice for lang=ja, then the usual
    macOS and Windows ones, else Pillow's default (which shows Japanese as boxes)."""
    paths = []
    if shutil.which("fc-match"):
        try:
            paths.append(subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=ja"], capture_output=True,
                                        text=True, timeout=5).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    paths += ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc"]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                  # Pillow < 10.1
        return ImageFont.load_default()


def sheet(models, region="body", side="L", views=None, poses=("rest",), size=360, ss=2):
    """(PIL image, layout dict, PIL "L" mask) of `models` (Model list) in the region's views, one row per (pose, model),
    every cell at one scale (the largest region decides it; for "body" the floor is one line), the rest pose's numbers
    in a table under the cells. The layout has each cell's box and camera; the mask (255 where a model covers the pixel)
    is what `trace` measures the outline on."""
    views = list(views or VIEWS[region])
    poses = list(poses or ["rest"])
    shapes, frames = {}, []
    for mi, m in enumerate(models):
        mask = region_mask(m, region, side)
        dirs = view_dirs(m, region, side)
        bad = [v for v in views if v not in dirs]
        if bad:
            raise ValueError(f"no view {bad[0]!r} for the {region} region; views: {', '.join(dirs)}")
        pts = []
        for p in poses:
            V, N = m.deform(p)
            shapes[mi, p] = (V, N)
            pts.append(V[mask])
        pts = np.concatenate(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        frames.append({"lo": lo, "hi": hi, "dirs": dirs, "mask": mask})
    half = 1.04 * max(0.5 * float(np.linalg.norm(f["hi"] - f["lo"])) for f in frames)
    if region == "body":
        half = 1.04 * max(max(0.5 * float(f["hi"][2]) + 0.02, 0.5 * float(np.linalg.norm((f["hi"] - f["lo"])[:2])))
                          for f in frames)
    rows = [(p, mi) for p in poses for mi in range(len(models))]
    cols = len(views)
    table = {m.label: measure(m, region, side) for m in models}
    keys = []
    for nums in table.values():
        keys += [k for k in nums if k not in keys]
    font = _font(13)
    name_w, val_w, line_h = 190, 96, 17
    grid_w = cols * size
    width = max(grid_w, name_w + val_w * len(models) + 16)
    table_h = (len(keys) + 2) * line_h + 10 if keys else 0
    height = len(rows) * (size + LABEL_H) + table_h
    img = Image.new("RGB", (width, height), (255, 255, 255))
    cover = Image.new("L", (width, height), 0)
    D = ImageDraw.Draw(img)
    layout = {"version": 1, "region": region, "side": side, "size": size, "label_h": LABEL_H, "bg": list(BG),
              "half": half, "px_per_m": size / (2.0 * half), "cells": [], "numbers": table}
    for ri, (p, mi) in enumerate(rows):
        m, f = models[mi], frames[mi]
        V, N = shapes[mi, p]
        T, lit, shade = scene_colours(m)
        centre = 0.5 * (f["lo"] + f["hi"])
        if region == "body":
            centre = np.array([centre[0], centre[1], half - 0.02])
        origin, axis = region_axis(m, region, side)
        for ci, v in enumerate(views):
            towards, up = f["dirs"][v]
            r, u, c = view_basis(towards, up)
            cam = {"centre": centre, "right": r, "up": u, "towards": c, "half": half}
            depth = None
            if region != "body":
                zs = (V[f["mask"]] - centre) @ c
                pad = 0.35 * half
                depth = (float(zs.min()) - pad, float(zs.max()) + pad)
            im, mask = render(V, N, T, lit, shade, cam, size=size, ss=ss, depth=depth)
            x, y = ci * size, ri * (size + LABEL_H)
            img.paste(im, (x, y + LABEL_H))
            cover.paste(Image.fromarray(mask.astype(np.uint8) * 255, "L"), (x, y + LABEL_H))
            bits = ([m.label] if len(models) > 1 else []) + ([p] if len(poses) > 1 else []) + [view_label(v, side)]
            label = " · ".join(bits)
            D.text((x + 5, y + 3), label, fill=(0, 0, 0), font=font)
            layout["cells"].append({"label": label, "model": m.label, "pose": p, "view": v, "box": [x, y + LABEL_H,
                                                                                                    size, size],
                                    "centre": centre.tolist(), "right": r.tolist(), "up": u.tolist(),
                                    "towards": c.tolist(), "origin": np.asarray(origin).tolist(),
                                    "axis": np.asarray(axis).tolist()})
    if keys:
        y0 = len(rows) * (size + LABEL_H) + 8
        D.text((8, y0), f"{region} ({side})" if region in SIDED else region, fill=(0, 0, 0), font=font)
        for j, m in enumerate(models):
            D.text((name_w + j * val_w, y0), m.label[:14], fill=(0, 0, 0), font=font)
        for i, k in enumerate(keys):
            yy = y0 + (i + 1) * line_h
            D.text((8, yy), k.replace("_ratio", ""), fill=(60, 60, 60), font=font)
            for j, m in enumerate(models):
                val = table[m.label].get(k)
                txt = "" if val is None else (f"{val:.2f}" if k.endswith("_ratio") else f"{val:.1f} mm")
                D.text((name_w + j * val_w, yy), txt, fill=(0, 0, 0), font=font)
    return img, layout, cover


# --------------------------------------------------------------------------------------------------------------- trace
def _cv():
    import cv2
    cv2.ocl.setUseOpenCL(False)        # nothing here needs the GPU; probing OpenCL is slow and prints driver warnings
    return cv2


def _register(img, ref):
    """2x3 matrix taking pixels of `img` (BGR) onto `ref` (the sheet, BGR): ORB features and a RANSAC similarity,
    else multi-scale template matching (the image a scaled crop of the sheet). Returns (A, method, score)."""
    cv2 = _cv()
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) >= 10:
            src = np.float32([k1[g.queryIdx].pt for g in good])
            dst = np.float32([k2[g.trainIdx].pt for g in good])
            A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                 maxIters=5000, confidence=0.999)
            if A is not None and inl is not None and int(inl.sum()) >= 12:
                scale = math.hypot(A[0, 0], A[1, 0])
                if abs(math.degrees(math.atan2(A[1, 0], A[0, 0]))) < 3.0 and 0.05 < scale < 20.0:
                    return A, "features", int(inl.sum())
    H1, W1 = g1.shape
    H2, W2 = g2.shape
    best = None

    def tryscale(s):
        nonlocal best
        tw, th = round(W1 * s), round(H1 * s)
        if tw > W2 or th > H2 or tw < 24 or th < 24:
            return
        t = cv2.resize(g1, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        res = cv2.matchTemplate(g2, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (mx, s, loc)
    for s in np.geomspace(0.05, 8.0, 90):
        tryscale(float(s))
    if best is not None:
        s0 = best[1]
        for s in np.linspace(s0 * 0.94, s0 * 1.06, 25):
            tryscale(float(s))
    if best is None or best[0] < 0.6:
        raise ValueError("could not place the marked image on the sheet (is it a screenshot of this sheet?)")
    mx, s, (x, y) = best
    return np.array([[s, 0.0, x], [0.0, s, y]]), "template", round(float(mx), 3)


def red_pixels(img_bgr):
    """(k, 2) pixel coordinates (x, y) of the red marking in a BGR image."""
    b, g, r = (img_bgr[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 150) & (g < 100) & (b < 100) & (r - np.maximum(g, b) > 80))
    return np.stack([xs, ys], -1).astype(float)


def trace(marked, sheet_png, layout, mask_png=None, step_mm=2.0):
    """Read a red line drawn on a lab sheet, or on a screenshot of part of it (scaled, with window borders), back in
    model space: which cell it is on, its points (m), and how far inside (+) or outside (-) the outline each part of
    it lies (mm), every `step_mm` along the stroke, with the position along the region's axis (`region_mm`). The
    outline is the sheet's coverage mask (`mask_png`, default <sheet stem>.mask.png beside it); without one it is
    guessed from the colours (pixels unlike the background, holes filled)."""
    cv2 = _cv()
    S = cv2.imread(str(sheet_png), cv2.IMREAD_COLOR)
    M = cv2.imread(str(marked), cv2.IMREAD_COLOR)
    if S is None or M is None:
        raise ValueError(f"cannot read {sheet_png if S is None else marked}")
    red = red_pixels(M)
    if len(red) < 5:
        raise ValueError("no red line in the marked image (draw it in pure red)")
    hole = np.zeros(M.shape[:2], np.uint8)
    hole[red[:, 1].astype(int), red[:, 0].astype(int)] = 255
    clean = cv2.inpaint(M, cv2.dilate(hole, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    A, method, score = _register(clean, S)
    out = trace_points(red @ A[:, :2].T + A[:, 2], sheet_png, layout, mask_png, step_mm, sheet=S)
    out["registration"] = {"method": method, "score": score, "scale": round(math.hypot(A[0, 0], A[1, 0]), 4)}
    return out


def sheet_cell(layout, pts):
    """(cell, mask): the cell of a sheet's layout holding most of the sheet pixels `pts` (k, 2), and which they are."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    best, inside = None, None
    for cell in layout["cells"]:
        x, y, w, h = cell["box"]
        m = (pts[:, 0] >= x) & (pts[:, 0] < x + w) & (pts[:, 1] >= y) & (pts[:, 1] < y + h)
        if best is None or m.sum() > inside.sum():
            best, inside = cell, m
    return best, inside


def cell_world(cell, layout, pts):
    """(k, 3) model-space points of sheet pixels `pts` (k, 2), on the plane of `cell`'s view through its centre."""
    x, y, w, h = cell["box"]
    loc = np.asarray(pts, float).reshape(-1, 2) - (x, y)
    ppm = float(layout["px_per_m"])
    c0, r, u = (np.asarray(cell[k], float) for k in ("centre", "right", "up"))
    return c0[None] + ((loc[:, 0] - w / 2.0) / ppm)[:, None] * r[None] + ((h / 2.0 - loc[:, 1]) / ppm)[:, None] * u[None]


def trace_points(pts, sheet_png, layout, mask_png=None, step_mm=2.0, sheet=None):
    """A stroke given as sheet pixels `pts` (k, 2), in any order (the red pixels `trace` found, or a line drawn over the
    sheet in a whiteboard), in model space: what `trace` returns, without the registration. `sheet` is the sheet
    image already read (BGR), else it is read from `sheet_png`."""
    cv2 = _cv()
    S = sheet if sheet is not None else cv2.imread(str(sheet_png), cv2.IMREAD_COLOR)
    if S is None:
        raise ValueError(f"cannot read {sheet_png}")
    pts = np.asarray(pts, float).reshape(-1, 2)
    best, inside_cell = sheet_cell(layout, pts)
    if best is None or inside_cell.sum() < 0.5 * len(pts):
        raise ValueError("the line is not on one cell of the sheet")
    x, y, w, h = best["box"]
    mpath = Path(mask_png) if mask_png else Path(sheet_png).with_name(Path(sheet_png).stem + ".mask.png")
    if mpath.is_file():
        sil = (cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w] > 127).astype(np.uint8)
    else:
        crop = S[y:y + h, x:x + w].astype(int)
        sil = (np.abs(crop - np.array(layout.get("bg", BG))[::-1]).max(axis=-1) > 12).astype(np.uint8)
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        flood = sil.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        sil = sil | (1 - flood)                         # pale insides enclosed by the outline
    din = cv2.distanceTransform(sil, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - sil, cv2.DIST_L2, 5)
    loc = pts[inside_cell] - (x, y)
    ix = np.clip(np.round(loc[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.round(loc[:, 1]).astype(int), 0, h - 1)
    inset = np.where(sil[iy, ix] > 0, din[iy, ix], -dout[iy, ix]) / float(layout["px_per_m"]) * 1000.0
    world = cell_world(best, layout, pts[inside_cell])
    q = world - world.mean(axis=0)
    axis = np.linalg.svd(q, full_matrices=False)[2][0]
    along = q @ axis
    origin, rax = np.asarray(best["origin"], float), np.asarray(best["axis"], float)
    if (world[along.argmax()] - world[along.argmin()]) @ rax < 0:
        along, axis = -along, -axis                     # run the stroke the way the region's axis runs
    along -= along.min()
    samples = []
    step = step_mm / 1000.0
    for k in range(int(along.max() / step) + 1):
        m = (along >= k * step) & (along < (k + 1) * step)
        if m.sum() < 2:
            continue
        p = world[m].mean(axis=0)
        samples.append({"along_mm": round((k + 0.5) * step_mm, 1), "region_mm": round(float((p - origin) @ rax) * 1000, 1),
                        "inset_mm": round(float(np.median(inset[m])), 2), "point": [round(float(v), 5) for v in p]})
    return {"cell": best["label"], "model": best["model"], "pose": best["pose"], "view": best["view"],
            "points": len(loc), "length_mm": round(float(along.max()) * 1000, 1),
            "inset_mm": {"max": round(float(inset.max()), 2), "mean": round(float(inset.mean()), 2),
                         "min": round(float(inset.min()), 2)},
            "samples": samples,
            "note": "inset_mm > 0: the line runs inside the outline by that much (trim); < 0: outside it (add)"}
