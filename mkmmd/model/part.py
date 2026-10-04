"""Character parts: the contract between part builders (bpy-free, numpy) and the assembler (Blender + mmd_tools).

A character is built from a spec (TOML) by part builders, in order (body, head, hair, outfit, ...). Each builder is a
plain function `build(ctx) -> Part` that only uses numpy (and PIL for textures), so it is testable without Blender;
the assembler turns the parts into one mmd_tools model and exports a PMX.

Model space: metres, Z up, the character faces -Y, feet on z = 0, centred on x = 0; the character's LEFT is +X.
Names: bones, morphs and materials use their PMX (Japanese) names; English names go in `name_en`. Side prefixes are
左/右 for standard bones (左腕) and chain bones may use either a prefix or a suffix (三つ編左1, 尻尾1_2).

This module is the frozen interface: fields may be ADDED (with defaults), never renamed or removed."""
from dataclasses import dataclass, field

import numpy as np

PANELS = ("eye", "brow", "mouth", "other")


@dataclass
class Material:
    name: str                                   # unique in the model; PMX material name
    name_en: str = ""
    diffuse: tuple = (1.0, 1.0, 1.0, 1.0)       # sRGB 0..1 + alpha
    specular: tuple = (0.0, 0.0, 0.0)
    shininess: float = 5.0
    ambient: tuple = (0.5, 0.5, 0.5)            # sRGB 0..1
    texture: str = ""                           # PNG file name in ctx.tex_dir ("" = none)
    toon: str = ""                              # toon ramp PNG in ctx.tex_dir ("" = none)
    sphere: str = ""                            # sphere map PNG in ctx.tex_dir
    sphere_mode: str = "none"                   # none | mul | add
    double_sided: bool = False
    edge: bool = True                           # PMX outline
    edge_color: tuple = (0.1, 0.05, 0.05, 1.0)
    edge_size: float = 1.0
    drop_shadow: bool = True
    self_shadow_map: bool = True
    self_shadow: bool = True
    alpha_blend: bool = False                   # true for anything with a non-opaque texture (lashes, blush, frills)
    comment: str = ""


@dataclass
class Bone:
    name: str                                   # PMX name, e.g. 左腕, 三つ編左1
    head: tuple                                 # model space (m)
    tail: tuple = None                          # model space end point; None with tail_bone set, or a leaf
    parent: str = ""                            # "" = no parent (only 全ての親)
    name_en: str = ""
    tail_bone: str = ""                         # PMX "connect to bone" instead of an offset tail
    visible: bool = True
    movable: bool = False                       # translatable (センター, IK targets, ...)
    rotatable: bool = True
    deform: bool = True
    layer: int = 0                              # PMX transform order
    after_physics: bool = False
    fixed_axis: tuple = None                    # twist bones: unit axis in model space
    local_x: tuple = None                       # optional local axes (fingers, arms): x and z in model space
    local_z: tuple = None
    grant: dict = None                          # {"parent": bone, "rotate": True, "move": False, "ratio": 1.0}
    ik: dict = None                             # {"target": bone, "chain": [{"bone": b, "limit": [[x0,y0,z0],[x1,y1,z1]] (deg) | None}],
                                                #  "iterations": 40, "angle": 114.6 (deg per iteration)}
    semantic: str = ""                          # mk semantic name when it is a standard bone (arm.L, index1.L)


@dataclass
class Mesh:
    name: str                                   # object name, e.g. "face", "hair_bangs"
    verts: np.ndarray                           # (n, 3) float, model space
    faces: list                                 # list of vertex index lists (tris and quads; ngons triangulated by the assembler)
    uv: np.ndarray = None                       # (sum of face sizes, 2) per face corner, in face order; None = no UV
    face_mat: np.ndarray = None                 # (len(faces),) int index into `mats`; None = all mats[0]
    mats: list = field(default_factory=list)    # material names used by this mesh
    weights: dict = field(default_factory=dict)  # bone name -> (n,) float; normalised and capped at 4 bones by the assembler
    morphs: dict = field(default_factory=dict)  # morph name -> (n, 3) vertex offsets (m); same name across meshes = one morph
    normals: np.ndarray = None                  # (n, 3) custom vertex normals (anime face shading), None = computed
    sharp: list = field(default_factory=list)   # [(i, j)] sharp edges
    subsurf: int = 0                            # Catmull-Clark levels applied by the assembler before export
    crease: dict = field(default_factory=dict)  # {(i, j): 0..1} subdivision creases
    smooth: bool = True


@dataclass
class Morph:
    name: str                                   # PMX name (まばたき, あ, 笑い, 照れ ...)
    panel: str = "other"                        # eye | brow | mouth | other
    name_en: str = ""


@dataclass
class RigidBody:
    name: str
    bone: str
    shape: str = "capsule"                      # sphere | box | capsule
    size: tuple = (0.05, 0.1, 0.0)              # sphere (r,), box (half x, half y, half z), capsule (r, height)
    location: tuple = (0.0, 0.0, 0.0)           # model space centre
    rotation: tuple = (0.0, 0.0, 0.0)           # XYZ Euler (rad), model space
    mode: str = "static"                        # static (follows the bone) | dynamic | dynamic_bone
    group: int = 0                              # 0..15
    no_collide: tuple = ()                      # groups this body ignores
    mass: float = 1.0
    damping: tuple = (0.5, 0.5)                 # linear, angular
    friction: float = 0.5
    bounce: float = 0.0


@dataclass
class Joint:
    name: str
    a: str                                      # rigid body names
    b: str
    location: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)
    move_lo: tuple = (0.0, 0.0, 0.0)
    move_hi: tuple = (0.0, 0.0, 0.0)
    rot_lo: tuple = (0.0, 0.0, 0.0)             # rad
    rot_hi: tuple = (0.0, 0.0, 0.0)
    spring_move: tuple = (0.0, 0.0, 0.0)
    spring_rot: tuple = (0.0, 0.0, 0.0)


@dataclass
class Part:
    name: str                                   # body, head, hair, outfit, tails ...
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    bones: list = field(default_factory=list)   # bones this part adds (the body part adds the standard skeleton)
    morphs: list = field(default_factory=list)  # declarations for morph names used in this part's meshes
    bodies: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    frames: dict = field(default_factory=dict)  # display frame name -> bone names
    info: dict = field(default_factory=dict)    # anything the next builders may query (landmarks, surfaces, sizes)


def check(part):
    """Raise ValueError for a malformed part (shapes, indices, names, weights, morphs)."""
    mats = {m.name for m in part.materials}
    bones = {b.name for b in part.bones}
    for m in part.meshes:
        v = np.asarray(m.verts, float)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
            raise ValueError(f"{part.name}/{m.name}: verts must be finite (n, 3)")
        n = len(v)
        corners = 0
        for f in m.faces:
            if len(f) < 3 or min(f) < 0 or max(f) >= n or len(set(f)) != len(f):
                raise ValueError(f"{part.name}/{m.name}: bad face {list(f)[:6]}")
            corners += len(f)
        if m.uv is not None and np.asarray(m.uv).shape != (corners, 2):
            raise ValueError(f"{part.name}/{m.name}: uv must be (face corners {corners}, 2)")
        if m.face_mat is not None and len(m.face_mat) != len(m.faces):
            raise ValueError(f"{part.name}/{m.name}: face_mat length != faces")
        if m.face_mat is not None and len(m.faces) and (min(m.face_mat) < 0 or max(m.face_mat) >= len(m.mats)):
            raise ValueError(f"{part.name}/{m.name}: face_mat index outside mats")
        unknown = [x for x in m.mats if x not in mats]
        if unknown and part.materials:
            raise ValueError(f"{part.name}/{m.name}: materials {unknown} not declared by the part")
        for b, w in m.weights.items():
            w = np.asarray(w, float)
            if w.shape != (n,) or not np.isfinite(w).all() or w.min() < 0:
                raise ValueError(f"{part.name}/{m.name}: weights for {b} must be (n,) >= 0")
        for k, d in m.morphs.items():
            if np.asarray(d, float).shape != (n, 3):
                raise ValueError(f"{part.name}/{m.name}: morph {k} must be (n, 3)")
    for mo in part.morphs:
        if mo.panel not in PANELS:
            raise ValueError(f"{part.name}: morph {mo.name} panel {mo.panel!r} not in {PANELS}")
    names = [b.name for b in part.bones]
    if len(names) != len(set(names)):
        raise ValueError(f"{part.name}: duplicate bone names")
    return bones
