"""bpy-free PMX 2.0 / 2.1 reader and writer on plain dataclasses (standard library + numpy only).

API (everything else is private)::

    read(path) -> PmxModel                  from_bytes(data) -> PmxModel
    write(model, path)                      to_bytes(model, index_sizes=None) -> bytes
    PmxModel.validate(require_special_frames=False)      # ValueError naming the offending item

    PmxModel (name, name_en, comment, comment_en, version, encoding, add_uv_count, vertices, faces, textures,
              materials, bones, morphs, frames, bodies, joints)
    PmxVertex PmxMaterial PmxBone PmxIK PmxMorph PmxFrame PmxBody PmxJoint     MaterialMorphOffset

Conventions
    * Values are carried exactly as the file stores them: PMX space (MMD units, Y up), radians, no conversion of any
      kind happens here. For reference, mmd_tools maps PMX (x, y, z) to Blender (x, z, y) * scale on import (and back
      on export), flips v (v -> 1 - v) and reverses every triangle, (a, b, c) <-> (c, b, a): PMX triangles are
      clockwise in MMD's left-handed space, so an exporter that starts from counter-clockwise-outward model-space
      faces must write them reversed.
    * Indices are plain ints into the lists of the same model (vertex, texture, material, bone, morph, rigid body);
      -1 means "none". Faces are ONE flat list of vertex indices, 3 per triangle (no quads, no nested lists), in the
      order of the materials: each material owns `index_count` consecutive indices (see PmxMaterial).
    * The file stores float32. Writing rounds to float32, reading returns the float32 value as a Python float, so
      write -> read is exact only for values that are float32-representable (otherwise equal to ~1e-7 relative).
    * Tuples: positions, normals, colours, ... are tuples of floats; the reader always returns tuples (never lists
      or arrays); the writer accepts any sequence (list, tuple, numpy array) of the right length.
    * Index sizes (1, 2 or 4 bytes) are chosen by the writer from the entity counts, like mmd_tools does:
      vertex indices are UNSIGNED for sizes 1 and 2 (size 1 up to 255 vertices, size 2 up to 65535, else a signed
      int32); every other index is SIGNED (size 1 up to 127 entities, size 2 up to 32767, else int32; -1 = none).
      The reader accepts any size combination, so files written by other tools load unchanged.
    * Weights: `kind=''` infers BDEF1 / BDEF2 / BDEF4 from len(bones) (1, 2, 4; anything else needs an explicit kind
      or padding with zero weights). BDEF2 and SDEF store only weights[0] in the file; the reader restores
      weights = (w0, 1 - w0). QDEF (PMX 2.1) has the BDEF4 layout. SDEF needs `sdef = (C, R0, R1)`.
    * Rigid body `mask`: bit i set means the body COLLIDES with bodies of group i (a Bullet-style filter mask,
      0xFFFF = collides with all groups; this is how mmd_tools reads the file field).
    * Versions: 2.0 (default; the only one mmd_tools 4.5 accepts, and it also lacks QDEF) and 2.1 (adds the QDEF
      weight type, flip / impulse morphs, material flag bits 5..7, joint kinds 1..5 and a soft body section). Soft
      bodies are not supported: the writer emits an empty section, the reader raises NotImplementedError for a
      non-empty one. Texts are UTF-16LE (default) or UTF-8; the reader returns 'utf-16le' / 'utf-8'.
    * write() / to_bytes() always run PmxModel.validate() first, so a file is never produced from a model with
      dangling indices; the reader is lenient about what other tools wrote (index sizes, encodings, 2.0 / 2.1,
      a 2.1 file without the soft body count) and raises ValueError (never struct.error) for truncated or
      corrupt data.
    * Speed: vertices, faces and vertex / UV morph offsets (the bulk of a model) are read, validated and written
      with numpy: a 100k-vertex, 200k-triangle model with 60k morph offsets validates in ~0.2 s, writes in ~0.25 s and
      reads in ~0.3-0.4 s; everything else is small and uses struct.

Example::

    m = PmxModel(name="x", bones=[PmxBone("全ての親")], faces=[0, 1, 2], materials=[PmxMaterial("m", index_count=3)],
                 vertices=[PmxVertex((float(i), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0), (0,)) for i in range(3)])
    write(m, "x.pmx")
    assert read("x.pmx").vertices[1].pos == (1.0, 0.0, 0.0)

The PMX file layout (header, text sections, flag bits, record layouts) follows the PMX 2.0 / 2.1 specification; the
bone flag bits are 0 tail is a bone index, 1 rotatable, 2 movable, 3 visible, 4 controllable, 5 IK, 8 grant
rotation, 9 grant translation, 10 fixed axis, 11 local axes, 12 after physics, 13 external parent.
"""
import math
import struct
from dataclasses import dataclass, field
from itertools import chain, repeat
from operator import index as _as_index
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "PmxVertex", "PmxMaterial", "PmxIK", "PmxBone", "PmxMorph", "PmxFrame", "PmxBody", "PmxJoint",
    "PmxModel", "MaterialMorphOffset", "write", "read", "to_bytes", "from_bytes",
]

_Z3 = (0.0, 0.0, 0.0)


# ====================================================================================================== data model
@dataclass
class PmxVertex:
    """One vertex. `pos`/`normal` are (x, y, z), `uv` is (u, v) in the PMX convention (v down; the exporter flips).

    bones/weights: 1, 2 or 4 bone indices (BDEF1/2/4; 2 for SDEF) and as many weights; -1 is "no bone" for unused
    BDEF4 slots. BDEF2 and SDEF store only weights[0] in the file (weights[1] is implied as 1 - weights[0]).
    `add_uv`: one 4-tuple per additional UV channel, len == PmxModel.add_uv_count. `sdef`: (C, R0, R1), three
    3-tuples, only with kind 'SDEF'. `kind`: '' infers BDEF1/BDEF2/BDEF4 from len(bones); or 'BDEF1' 'BDEF2' 'BDEF4'
    'SDEF' 'QDEF' (QDEF needs version 2.1). The reader returns '' for BDEF vertices and 'SDEF' / 'QDEF' otherwise."""
    pos: tuple
    normal: tuple
    uv: tuple
    bones: tuple = (0,)
    weights: tuple = (1.0,)
    add_uv: tuple = ()
    sdef: tuple = None
    kind: str = ""
    edge_scale: float = 1.0


@dataclass
class PmxMaterial:
    """One material. `index_count` is the number of face INDICES (3 per triangle) this material owns; materials
    consume the model's flat `faces` list in order, so the counts must sum to len(faces).
    Colours: diffuse (r, g, b, a), specular (r, g, b), ambient (r, g, b), edge_color (r, g, b, a).
    texture / sphere_texture: index into PmxModel.textures or -1. sphere_mode: 0 none, 1 multiply, 2 add,
    3 sub-texture. toon: with toon_shared a shared toon number 0..9 (toon01..toon10), else a texture index or -1.
    Flags: double_sided, ground_shadow, self_shadow_map, self_shadow, edge (outline); PMX 2.1 only (bits 5..7):
    vertex_color, point_draw, line_draw."""
    name: str
    name_en: str = ""
    diffuse: tuple = (1.0, 1.0, 1.0, 1.0)
    specular: tuple = (0.0, 0.0, 0.0)
    shininess: float = 5.0
    ambient: tuple = (0.5, 0.5, 0.5)
    double_sided: bool = False
    ground_shadow: bool = True
    self_shadow_map: bool = True
    self_shadow: bool = True
    edge: bool = True
    vertex_color: bool = False
    point_draw: bool = False
    line_draw: bool = False
    edge_color: tuple = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 1.0
    texture: int = -1
    sphere_texture: int = -1
    sphere_mode: int = 0
    toon_shared: bool = False
    toon: int = -1
    memo: str = ""
    index_count: int = 0


@dataclass
class PmxIK:
    """IK data of a bone. `angle` is the limit in radians per iteration. `links` is a list of
    (bone, has_limit, lo, hi) with lo / hi (x, y, z) angle limits in radians (ignored, and read back as zeros, when
    has_limit is False); the first link is the one next to the effector, the last is the chain root."""
    target: int
    loops: int = 40
    angle: float = 2.0
    links: list = field(default_factory=list)


@dataclass
class PmxBone:
    """One bone. `pos` is the bone head. Connection (tail): when `tail_offset` is None the tail is the bone
    `tail_bone` (-1 = none), otherwise `tail_offset` is an (x, y, z) offset from `pos` and `tail_bone` is ignored. `layer`
    is the transform order. Grant (inherit): `grant_rotate` / `grant_move` copy `grant_ratio` of the rotation /
    translation of bone `grant_parent` (both are ignored when neither flag is set). `fixed_axis`: unit axis for twist
    bones. `local_x` / `local_z`: local axes, written only when both are set. `ext_parent`: key of the external
    parent (rarely used). `ik`: PmxIK or None. The reader fills the fields the flags select and leaves the rest at
    their defaults."""
    name: str
    name_en: str = ""
    pos: tuple = _Z3
    parent: int = -1
    layer: int = 0
    tail_bone: int = -1
    tail_offset: tuple = None
    rotatable: bool = True
    movable: bool = False
    visible: bool = True
    controllable: bool = True
    grant_rotate: bool = False
    grant_move: bool = False
    grant_parent: int = -1
    grant_ratio: float = 0.0
    fixed_axis: tuple = None
    local_x: tuple = None
    local_z: tuple = None
    after_physics: bool = False
    ik: PmxIK = None
    ext_parent: int = None


@dataclass
class PmxMorph:
    """One morph. `panel`: 0 system, 1 eyebrow, 2 eye, 3 mouth, 4 other. `kind` and the shape of `offsets`:
    'vertex': (vertex, (dx, dy, dz));  'uv' / 'uv1'..'uv4': (vertex, (a, b, c, d));
    'group' / 'flip' (2.1): (morph, ratio);  'bone': (bone, (tx, ty, tz), (qx, qy, qz, qw));
    'material': a MaterialMorphOffset, a plain tuple of its 11 fields or a dict with its field names
    (the reader returns MaterialMorphOffset tuples);  'impulse' (2.1): (body, local_flag, velocity3, torque3)."""
    name: str
    name_en: str = ""
    panel: int = 4
    kind: str = "vertex"
    offsets: list = field(default_factory=list)


@dataclass
class PmxFrame:
    """One display frame. `items` is a list of ('bone', bone_index) or ('morph', morph_index). `special` marks the
    two frames MMD treats specially: the root-bone frame and the morph ("表情") frame, in this order, first."""
    name: str
    name_en: str = ""
    special: bool = False
    items: list = field(default_factory=list)


@dataclass
class PmxBody:
    """One rigid body. `shape`: 0 sphere, 1 box, 2 capsule; `size` (sphere (r,..), box half extents, capsule
    (r, height, ..)); `pos` / `rot` are carried as stored (centre and rotation vector in radians; mmd_tools reads `rot`
    as the Blender YXZ Euler -(rx, rz, ry), i.e. MMD applies yaw-pitch-roll). `group` 0..15; `mask` bit i set = collides
    with group i (see module docstring). `mode`: 0 follows the bone, 1 physics, 2 physics + bone alignment."""
    name: str
    name_en: str = ""
    bone: int = -1
    group: int = 0
    mask: int = 0xFFFF
    shape: int = 0
    size: tuple = _Z3
    pos: tuple = _Z3
    rot: tuple = _Z3
    mass: float = 1.0
    linear_damping: float = 0.5
    angular_damping: float = 0.5
    restitution: float = 0.0
    friction: float = 0.5
    mode: int = 0


@dataclass
class PmxJoint:
    """One joint between rigid bodies `body_a` and `body_b` (indices, -1 = none). `kind` 0 is the spring 6DOF joint
    (the only one in PMX 2.0; 2.1 adds 1 6DOF, 2 P2P, 3 cone twist, 4 slider, 5 hinge). `pos` / `rot` as for PmxBody;
    limits and springs are (x, y, z), rotation limits in radians."""
    name: str
    name_en: str = ""
    kind: int = 0
    body_a: int = -1
    body_b: int = -1
    pos: tuple = _Z3
    rot: tuple = _Z3
    move_lo: tuple = _Z3
    move_hi: tuple = _Z3
    rot_lo: tuple = _Z3
    rot_hi: tuple = _Z3
    spring_move: tuple = _Z3
    spring_rot: tuple = _Z3


class MaterialMorphOffset(NamedTuple):
    """One entry of a 'material' morph: the 2.0 fields in file order. `index` is the material (-1 = all materials),
    `op` 0 multiplies and 1 adds. It is a tuple, so it equals (and can be built from) a plain 11-tuple."""
    index: int
    op: int
    diffuse: tuple
    specular: tuple
    shininess: float
    ambient: tuple
    edge_color: tuple
    edge_size: float
    texture: tuple
    sphere: tuple
    toon: tuple

    @classmethod
    def of(cls, index, op, **fields):
        """Build an offset from keywords named like the fields; omitted ones are neutral for `op` (1.0 for a
        multiply, 0.0 for an add). Raises ValueError for an unknown field name."""
        fill = 1.0 if op == 0 else 0.0
        out = {
            "diffuse": (fill,) * 4, "specular": (fill,) * 3, "shininess": fill, "ambient": (fill,) * 3,
            "edge_color": (fill,) * 4, "edge_size": fill, "texture": (fill,) * 4, "sphere": (fill,) * 4,
            "toon": (fill,) * 4,
        }
        unknown = set(fields) - set(out)
        if unknown:
            raise ValueError(f"unknown material morph offset field(s) {sorted(unknown)}; expected {sorted(out)}")
        out.update(fields)
        return cls(index, op, **out)


@dataclass
class PmxModel:
    """A whole PMX file. `version` 2.0 or 2.1; `encoding` 'utf-16le' or 'utf-8'; `add_uv_count` 0..4 additional UV
    channels per vertex; `faces` the flat list of vertex indices (3 per triangle); `textures` path strings relative
    to the file (written as given, '/' or '\\\\')."""
    name: str = ""
    name_en: str = ""
    comment: str = ""
    comment_en: str = ""
    version: float = 2.0
    encoding: str = "utf-16le"
    add_uv_count: int = 0
    vertices: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    textures: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    bones: list = field(default_factory=list)
    morphs: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    bodies: list = field(default_factory=list)
    joints: list = field(default_factory=list)

    def validate(self, require_special_frames=False):
        """Raise ValueError (message names the entity and the problem) unless the model can be written as PMX.

        Checks: version / encoding / add_uv_count; vertex tuple lengths, weight counts and kinds (1, 2 or 4 bones
        unless an explicit kind), `weights` as long as `bones`, SDEF data, finite numbers, every bone / morph /
        texture / material / vertex / rigid body index in range (-1 allowed where it means none); faces a flat
        integer list whose length is a multiple of 3 with indices < len(vertices); material `index_count` values
        multiples of 3 that sum to len(faces); bone parents (no cycles), grant, IK and axes; morph kinds and
        offsets (flip / impulse need version 2.1); display frame items; rigid body and joint fields.
        Duplicate names are allowed. Special display frames are optional unless `require_special_frames`, which
        demands exactly two, and as the first two frames. Returns None."""
        _validate(self, require_special_frames)


# ====================================================================================================== tables
_S = struct.Struct
_I, _B, _SB, _F = _S("<i"), _S("<B"), _S("<b"), _S("<f")
_F3 = _S("<3f")
_IH = _S("<iH")
_MAT_A = _S("<4f3ff3fB4ff")          # diffuse, specular, shininess, ambient, flags, edge colour, edge size
_BODY = _S("<BHB9f5fB")              # group, mask, shape, size/pos/rot, mass..friction, mode
_JOINT = _S("<24f")
_MM_FLOATS = _S("<4f3ff3f4ff4f4f4f")   # material morph body after (index, op)

_SF = {1: "b", 2: "h", 4: "i"}       # signed index formats
_UDT = {1: "u1", 2: "<u2", 4: "<i4"}
_SDT = {1: "i1", 2: "<i2", 4: "<i4"}
_SIZE_KEYS = ("vertex", "texture", "material", "bone", "morph", "body")

_VERSIONS = (2.0, 2.1)
_CODECS = {
    "utf-16le": (0, "utf-16-le"), "utf-16-le": (0, "utf-16-le"), "utf16le": (0, "utf-16-le"),
    "utf-8": (1, "utf-8"), "utf8": (1, "utf-8"),
}
_CODEC_NAMES = ("utf-16le", "utf-8")

_KIND_NAME = ("BDEF1", "BDEF2", "BDEF4", "SDEF", "QDEF")
_KIND_CODE = {k: i for i, k in enumerate(_KIND_NAME)}
_KIND_NB = np.array([1, 2, 4, 2, 4], dtype=np.int64)         # bones per weight type
_INFER = np.array([255, 0, 1, 255, 2, 255], dtype=np.uint8)   # len(bones) -> weight type (255 = ambiguous)

_MORPH_CODE = {"group": 0, "vertex": 1, "bone": 2, "uv": 3, "uv1": 4, "uv2": 5, "uv3": 6, "uv4": 7,
               "material": 8, "flip": 9, "impulse": 10}
_MORPH_NAME = {v: k for k, v in _MORPH_CODE.items()}
_MORPH_UV_KINDS = ("uv", "uv1", "uv2", "uv3", "uv4")


def _isz(count, signed):
    """Smallest index size in bytes for `count` entities (signed: 1 <= 127, 2 <= 32767; unsigned: 255 / 65535)."""
    if signed:
        return 1 if count <= 127 else 2 if count <= 32767 else 4
    return 1 if count <= 255 else 2 if count <= 65535 else 4


def _version(v):
    """Return 2.0 or 2.1 for a version number (tolerating float32 rounding) or raise ValueError."""
    try:
        r = round(float(v), 2)
    except (TypeError, ValueError):
        r = None
    if r not in _VERSIONS:
        raise ValueError(f"unsupported PMX version {v!r}; this module reads and writes 2.0 and 2.1")
    return r


def _encoding(name):
    """(header id, python codec) for an encoding name or raise ValueError."""
    hit = _CODECS.get(str(name).lower().replace("_", "-"))
    if hit is None:
        raise ValueError(f"unsupported text encoding {name!r}; use 'utf-16le' or 'utf-8'")
    return hit


# ====================================================================================================== validation
def _chk_str(w, what, s):
    if not isinstance(s, str):
        raise ValueError(f"{w}: {what} must be a str, got {s!r}")


def _chk_names(w, obj, *attrs):
    for what in attrs:
        _chk_str(w, what, getattr(obj, what))


def _chk_int(w, what, v):
    try:
        return _as_index(v)
    except TypeError:
        raise ValueError(f"{w}: {what} must be an int, got {v!r}") from None


def _chk_idx(w, what, v, noun, count, none_ok=True):
    """Check that `v` indexes one of `count` entities of kind `noun` (-1 allowed when none_ok)."""
    v = _chk_int(w, what, v)
    if not ((-1 if none_ok else 0) <= v < count):
        raise ValueError(f"{w}: {what} {v} is not a valid {noun} index (the model has {count} {noun}s"
                         f"{', -1 = none' if none_ok else ''})")
    return v


def _chk_range(w, what, v, lo, hi):
    v = _chk_int(w, what, v)
    if not lo <= v <= hi:
        raise ValueError(f"{w}: {what} {v} out of range {lo}..{hi}")
    return v


def _chk_vec(w, what, v, n):
    try:
        ok = len(v) == n and all(map(math.isfinite, v))
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{w}: {what} must be {n} finite numbers, got {v!r}")


def _chk_num(w, what, x):
    try:
        ok = math.isfinite(x)
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{w}: {what} must be a finite number, got {x!r}")


def _lens(seqs):
    return np.fromiter(map(len, seqs), np.int64, len(seqs))


def _col(seqs, k, w, what, dtype=np.float64, per=1):
    """Stack `seqs` (each exactly k numbers) into an (n, k) array; ValueError names the first offender (entity
    `w` number i // per, item i % per when `per` items belong to one entity)."""
    n = len(seqs)
    try:
        lens = _lens(seqs)
    except TypeError:
        raise ValueError(f"{w}: {what} must be a sequence of {k} numbers") from None
    bad = np.flatnonzero(lens != k)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"{w} {i // per}: {what}{f' {i % per}' if per > 1 else ''} needs {k} components, "
                         f"got {int(lens[i])}")
    try:
        out = np.fromiter(chain.from_iterable(seqs), dtype, n * k).reshape(n, k)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{w}: {what} must contain numbers ({e})") from None
    if out.dtype.kind == "f":
        bad = np.flatnonzero(~np.isfinite(out).all(axis=1))
        if bad.size:
            i = int(bad[0])
            raise ValueError(f"{w} {i // per}: {what}{f' {i % per}' if per > 1 else ''} is not finite "
                             f"({tuple(out[i])})")
    return out


class _VertexData:
    """Validated, gathered vertex columns. `bones` / `weights` are flat (sum of nb); `sdef` is (n_sdef, 9)."""
    __slots__ = ("n", "code", "pos", "normal", "uv", "add", "edge", "nb", "bones", "weights", "sdef")


def _gather_vertices(vs, nuv, nbones):
    """Check the vertex list and stack its fields into numpy arrays (see _VertexData)."""
    n = len(vs)
    d = _VertexData()
    d.n = n
    d.pos = _col([v.pos for v in vs], 3, "vertex", "pos")
    d.normal = _col([v.normal for v in vs], 3, "vertex", "normal")
    d.uv = _col([v.uv for v in vs], 2, "vertex", "uv")
    try:
        d.edge = np.fromiter((v.edge_scale for v in vs), np.float64, n)
    except (TypeError, ValueError):
        raise ValueError("vertex edge_scale must be a number") from None
    if not np.isfinite(d.edge).all():
        raise ValueError(f"vertex {int(np.flatnonzero(~np.isfinite(d.edge))[0])}: edge_scale is not finite")

    adds = [v.add_uv for v in vs]
    na = _lens(adds)
    bad = np.flatnonzero(na != nuv)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: has {int(na[i])} additional UVs but the model declares add_uv_count={nuv}")
    d.add = _col(list(chain.from_iterable(adds)), 4, "vertex", "additional uv", per=max(nuv, 1)).reshape(n, nuv, 4)

    bl, wl = [v.bones for v in vs], [v.weights for v in vs]
    nb, nw = _lens(bl), _lens(wl)
    bad = np.flatnonzero(nb != nw)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: weights has {int(nw[i])} entries but bones has {int(nb[i])}")
    kinds = np.array([v.kind or "" for v in vs], dtype=str)
    code = _INFER[np.minimum(nb, 5)]
    for name, c in _KIND_CODE.items():
        code[kinds == name] = c
    bad = np.flatnonzero((kinds != "") & ~np.isin(kinds, _KIND_NAME))
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: unknown kind {str(kinds[i])!r}; use '' or one of {', '.join(_KIND_NAME)}")
    bad = np.flatnonzero(code == 255)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: {int(nb[i])} bones; give 1, 2 or 4 (pad with zero weights) or an explicit kind")
    bad = np.flatnonzero(nb != _KIND_NB[code])
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: kind {_KIND_NAME[code[i]]} needs {int(_KIND_NB[code[i]])} bones, got {int(nb[i])}")
    d.code, d.nb = code, nb
    total = int(nb.sum())
    try:
        d.bones = np.fromiter(map(_as_index, chain.from_iterable(bl)), np.int64, total)
        d.weights = np.fromiter(chain.from_iterable(wl), np.float64, total)
    except (TypeError, ValueError):
        raise ValueError("vertex bones must be ints and weights numbers") from None
    ends = np.cumsum(nb)
    bad = np.flatnonzero((d.bones < -1) | (d.bones >= nbones))
    if bad.size:
        k = int(bad[0])
        raise ValueError(f"vertex {int(np.searchsorted(ends, k, side='right'))}: bone index {int(d.bones[k])} is not a "
                         f"valid bone index (the model has {nbones} bones, -1 = none)")
    bad = np.flatnonzero(~np.isfinite(d.weights))
    if bad.size:
        raise ValueError(f"vertex {int(np.searchsorted(ends, int(bad[0]), side='right'))}: weight is not finite")

    has = np.fromiter((v.sdef is not None for v in vs), bool, n)
    bad = np.flatnonzero(has & (code != 3))
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"vertex {i}: sdef data given but kind is {_KIND_NAME[code[i]]}; set kind='SDEF'")
    rows = np.flatnonzero(code == 3)
    sd = [vs[i].sdef for i in rows.tolist()]
    for i, s in zip(rows.tolist(), sd):
        if s is None or len(s) != 3 or any(len(t) != 3 for t in s):
            raise ValueError(f"vertex {i}: SDEF needs sdef = (C, R0, R1), three 3-tuples, got {s!r}")
    d.sdef = _col([x for s in sd for x in s], 3, "sdef vector", "value").reshape(-1, 9)
    return d


def _faces(faces, nv):
    """Validate the flat face list and return it as an int64 array."""
    try:
        a = np.asarray(faces)
    except ValueError:
        raise ValueError("faces must be a flat list of vertex indices, 3 per triangle") from None
    if a.size == 0:
        return np.zeros(0, np.int64)
    if a.ndim != 1:
        raise ValueError("faces must be ONE flat list of vertex indices (3 per triangle), not nested")
    if a.dtype.kind not in "iu":
        raise ValueError(f"faces must be integer vertex indices, got dtype {a.dtype}")
    if a.size % 3:
        raise ValueError(f"faces holds {a.size} indices, which is not a multiple of 3")
    bad = np.flatnonzero((a < 0) | (a >= nv))
    if bad.size:
        k = int(bad[0])
        raise ValueError(f"faces[{k}] (triangle {k // 3}): vertex index {int(a[k])} out of range "
                         f"(the model has {nv} vertices)")
    return a.astype(np.int64, copy=False)


def _mat_offset(w, o):
    """Normalise a material morph offset (tuple, MaterialMorphOffset or dict) to a MaterialMorphOffset."""
    try:
        if isinstance(o, dict):
            if "index" not in o or "op" not in o:
                raise ValueError("a dict offset needs at least 'index' and 'op'")
            return MaterialMorphOffset.of(**o)
        if len(o) != 11:
            raise ValueError(f"expected 11 fields, got {len(o)}")
        return MaterialMorphOffset(*o)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{w}: bad material morph offset {o!r} ({e})") from None


def _ik_link(w, link):
    """Normalise an IK link to (bone, has_limit, lo, hi) (lo / hi None without limit)."""
    try:
        bone = link[0]
        has = bool(link[1]) if len(link) > 1 else False
        lo, hi = (link[2], link[3]) if has else (None, None)
    except (TypeError, IndexError):
        raise ValueError(f"{w}: IK link must be (bone, has_limit, lo, hi), got {link!r}") from None
    return bone, has, lo, hi


def _check_offsets(w, i, mo, ver, nv, nmo, nb, nmat, nbody):
    """Check one morph; return (index array, data array) for the numpy kinds (vertex / uv), else None."""
    code = _MORPH_CODE.get(mo.kind)
    if code is None:
        raise ValueError(f"{w}: unknown morph kind {mo.kind!r}; expected one of {', '.join(_MORPH_CODE)}")
    if code >= 9 and ver < 2.1:
        raise ValueError(f"{w}: morph kind {mo.kind!r} needs PMX version 2.1 (the model is {ver})")
    _chk_range(w, "panel", mo.panel, 0, 4)
    offs = mo.offsets
    n = len(offs)
    if mo.kind == "vertex" or mo.kind in _MORPH_UV_KINDS:
        k = 3 if mo.kind == "vertex" else 4
        try:
            idx = np.fromiter(map(_as_index, (o[0] for o in offs)), np.int64, n)
            values = [o[1] for o in offs]
        except (TypeError, IndexError, ValueError):
            raise ValueError(f"{w}: the offsets of a {mo.kind!r} morph must be (vertex index, values) pairs") from None
        data = _col(values, k, f"{w}: offset", "value")
        bad = np.flatnonzero((idx < 0) | (idx >= nv))
        if bad.size:
            j = int(bad[0])
            raise ValueError(f"{w}: offset {j}: vertex index {int(idx[j])} is not a valid vertex index "
                             f"(the model has {nv} vertices)")
        return idx, data
    for j, o in enumerate(offs):
        ww = f"{w}: offset {j}"
        try:
            if mo.kind in ("group", "flip"):
                _chk_idx(ww, "morph", o[0], "morph", nmo, none_ok=False)
                if o[0] == i:
                    raise ValueError(f"{ww}: a morph cannot refer to itself")
                _chk_num(ww, "ratio", o[1])
            elif mo.kind == "bone":
                _chk_idx(ww, "bone", o[0], "bone", nb, none_ok=False)
                _chk_vec(ww, "translation", o[1], 3)
                _chk_vec(ww, "rotation quaternion", o[2], 4)
            elif mo.kind == "material":
                m = _mat_offset(ww, o)
                _chk_idx(ww, "material", m.index, "material", nmat)
                _chk_range(ww, "op", m.op, 0, 1)
                for name, size in (("diffuse", 4), ("specular", 3), ("ambient", 3), ("edge_color", 4),
                                   ("texture", 4), ("sphere", 4), ("toon", 4)):
                    _chk_vec(ww, name, getattr(m, name), size)
                _chk_num(ww, "shininess", m.shininess)
                _chk_num(ww, "edge_size", m.edge_size)
            else:   # impulse
                _chk_idx(ww, "rigid body", o[0], "rigid body", nbody, none_ok=False)
                _chk_vec(ww, "velocity", o[2], 3)
                _chk_vec(ww, "torque", o[3], 3)
        except (TypeError, IndexError):
            raise ValueError(f"{ww}: malformed {mo.kind} morph offset {o!r}") from None
    return None


def _validate(m, require_special_frames=False):
    """Implementation of PmxModel.validate; returns the arrays gathered on the way for the writer."""
    ver = _version(m.version)
    _encoding(m.encoding)
    nuv = _chk_range("model", "add_uv_count", m.add_uv_count, 0, 4)
    for what in ("name", "name_en", "comment", "comment_en"):
        _chk_str("model", what, getattr(m, what))
    nv, nt, nmat, nb, nmo, nbody = (len(x) for x in (m.vertices, m.textures, m.materials, m.bones, m.morphs, m.bodies))
    g = {"vertices": _gather_vertices(m.vertices, nuv, nb), "faces": _faces(m.faces, nv), "morphs": {}}
    qdef = np.flatnonzero(g["vertices"].code == 4)
    if ver < 2.1 and qdef.size:
        raise ValueError(f"vertex {int(qdef[0])}: weight type QDEF needs PMX version 2.1 (the model is {ver})")

    for i, t in enumerate(m.textures):
        _chk_str(f"texture {i}", "path", t)

    total = 0
    for i, mt in enumerate(m.materials):
        w = f"material {i} {mt.name!r}"
        _chk_names(w, mt, "name", "name_en", "memo")
        for what, size in (("diffuse", 4), ("specular", 3), ("ambient", 3), ("edge_color", 4)):
            _chk_vec(w, what, getattr(mt, what), size)
        _chk_num(w, "shininess", mt.shininess)
        _chk_num(w, "edge_size", mt.edge_size)
        _chk_idx(w, "texture", mt.texture, "texture", nt)
        _chk_idx(w, "sphere_texture", mt.sphere_texture, "texture", nt)
        _chk_range(w, "sphere_mode", mt.sphere_mode, 0, 3)
        if mt.toon_shared:
            _chk_range(w, "shared toon number", mt.toon, 0, 9)
        else:
            _chk_idx(w, "toon", mt.toon, "texture", nt)
        cnt = _chk_int(w, "index_count", mt.index_count)
        if cnt < 0 or cnt % 3:
            raise ValueError(f"{w}: index_count {cnt} must be a non-negative multiple of 3")
        if ver < 2.1 and (mt.vertex_color or mt.point_draw or mt.line_draw):
            raise ValueError(f"{w}: vertex_color / point_draw / line_draw need PMX version 2.1")
        total += cnt
    if total != len(g["faces"]):
        raise ValueError(f"material index counts sum to {total} but the model has {len(g['faces'])} face indices "
                         f"({nmat} materials)")

    parents = []
    for i, b in enumerate(m.bones):
        w = f"bone {i} {b.name!r}"
        _chk_names(w, b, "name", "name_en")
        _chk_vec(w, "pos", b.pos, 3)
        p = _chk_idx(w, "parent", b.parent, "bone", nb)
        if p == i:
            raise ValueError(f"{w}: is its own parent")
        parents.append(p)
        _chk_int(w, "layer", b.layer)
        if b.tail_offset is None:
            _chk_idx(w, "tail_bone", b.tail_bone, "bone", nb)
        else:
            _chk_vec(w, "tail_offset", b.tail_offset, 3)
        if b.grant_rotate or b.grant_move:
            gp = _chk_idx(w, "grant_parent", b.grant_parent, "bone", nb, none_ok=False)
            if gp == i:
                raise ValueError(f"{w}: grants from itself")
            _chk_num(w, "grant_ratio", b.grant_ratio)
        if b.fixed_axis is not None:
            _chk_vec(w, "fixed_axis", b.fixed_axis, 3)
        if (b.local_x is None) != (b.local_z is None):
            raise ValueError(f"{w}: local_x and local_z must be set together")
        if b.local_x is not None:
            _chk_vec(w, "local_x", b.local_x, 3)
            _chk_vec(w, "local_z", b.local_z, 3)
        if b.ext_parent is not None:
            _chk_int(w, "ext_parent", b.ext_parent)
        if b.ik is not None:
            ik = b.ik
            _chk_idx(w, "IK target", ik.target, "bone", nb, none_ok=False)
            _chk_int(w, "IK loops", ik.loops)
            _chk_num(w, "IK angle", ik.angle)
            for j, link in enumerate(ik.links):
                lw = f"{w}: IK link {j}"
                bone, has, lo, hi = _ik_link(lw, link)
                _chk_idx(lw, "bone", bone, "bone", nb, none_ok=False)
                if has:
                    _chk_vec(lw, "lower limit", lo, 3)
                    _chk_vec(lw, "upper limit", hi, 3)
    state = [0] * nb                     # parent cycle check: 0 new, 1 on the current path, 2 done
    for i in range(nb):
        path, j = [], i
        while j != -1 and state[j] == 0:
            state[j] = 1
            path.append(j)
            j = parents[j]
        if j != -1 and state[j] == 1:
            raise ValueError(f"bone {j} {m.bones[j].name!r}: the parent chain loops back to it")
        for j in path:
            state[j] = 2

    for i, mo in enumerate(m.morphs):
        w = f"morph {i} {mo.name!r}"
        _chk_names(w, mo, "name", "name_en")
        arrays = _check_offsets(w, i, mo, ver, nv, nmo, nb, nmat, nbody)
        if arrays is not None:
            g["morphs"][i] = arrays

    special = []
    for i, f in enumerate(m.frames):
        w = f"frame {i} {f.name!r}"
        _chk_names(w, f, "name", "name_en")
        if f.special:
            special.append(i)
        for j, item in enumerate(f.items):
            try:
                kind, idx = item
            except (TypeError, ValueError):
                raise ValueError(f"{w}: item {j} must be ('bone' | 'morph', index), got {item!r}") from None
            if kind == "bone":
                _chk_idx(f"{w}: item {j}", "bone", idx, "bone", nb, none_ok=False)
            elif kind == "morph":
                _chk_idx(f"{w}: item {j}", "morph", idx, "morph", nmo, none_ok=False)
            else:
                raise ValueError(f"{w}: item {j} kind must be 'bone' or 'morph', got {kind!r}")
    if require_special_frames and special != [0, 1]:
        raise ValueError("expected exactly two special display frames (root bones, morphs) as the first two frames, "
                         f"found special frames at positions {special}")

    for i, b in enumerate(m.bodies):
        w = f"rigid body {i} {b.name!r}"
        _chk_names(w, b, "name", "name_en")
        _chk_idx(w, "bone", b.bone, "bone", nb)
        _chk_range(w, "group", b.group, 0, 15)
        _chk_range(w, "mask", b.mask, 0, 0xFFFF)
        _chk_range(w, "shape", b.shape, 0, 2)
        _chk_range(w, "mode", b.mode, 0, 2)
        for what in ("size", "pos", "rot"):
            _chk_vec(w, what, getattr(b, what), 3)
        for what in ("mass", "linear_damping", "angular_damping", "restitution", "friction"):
            _chk_num(w, what, getattr(b, what))

    for i, j in enumerate(m.joints):
        w = f"joint {i} {j.name!r}"
        _chk_names(w, j, "name", "name_en")
        _chk_range(w, "kind", j.kind, 0, 5 if ver >= 2.1 else 0)
        _chk_idx(w, "body_a", j.body_a, "rigid body", nbody)
        _chk_idx(w, "body_b", j.body_b, "rigid body", nbody)
        for what in ("pos", "rot", "move_lo", "move_hi", "rot_lo", "rot_hi", "spring_move", "spring_rot"):
            _chk_vec(w, what, getattr(j, what), 3)
    return g


# ====================================================================================================== writer
class _Out:
    """Output chunks plus the text codec and index sizes of the file being written."""

    def __init__(self, codec, sizes):
        self.parts = []
        self.codec = codec
        self.sizes = sizes
        self.ft, self.fm, self.fb, self.fo, self.fr = (_SF[s] for s in sizes[1:])
        self._tex, self._bone = _S("<" + self.ft), _S("<" + self.fb)

    def put(self, b):
        self.parts.append(b)

    def text(self, s):
        try:
            b = s.encode(self.codec)
        except UnicodeEncodeError as e:
            raise ValueError(f"cannot encode {s!r} as {self.codec}: {e.reason}") from None
        self.parts.append(_I.pack(len(b)))
        self.parts.append(b)

    def bone(self, i):
        self.parts.append(self._bone.pack(i))

    def tex(self, i):
        self.parts.append(self._tex.pack(i))


def _vdtype(code, nuv, bone_dtype):
    """numpy record dtype of one vertex of weight type `code` (packed, little endian, file layout)."""
    f = [("pos", "<f4", (3,)), ("normal", "<f4", (3,)), ("uv", "<f4", (2,))]
    if nuv:
        f.append(("add", "<f4", (nuv, 4)))
    f += [("type", "u1"), ("bones", bone_dtype, (int(_KIND_NB[code]),))]
    if code in (1, 3):
        f.append(("w", "<f4", (1,)))
    elif code in (2, 4):
        f.append(("w", "<f4", (4,)))
    if code == 3:
        f.append(("sdef", "<f4", (9,)))
    f.append(("edge", "<f4"))
    return np.dtype(f)


def _w_vertices(o, d, nuv):
    o.put(_I.pack(d.n))
    if d.n == 0:
        return
    bdt = _SDT[o.sizes[3]]
    starts = np.cumsum(d.nb) - d.nb
    groups = []
    for c in np.unique(d.code).tolist():
        idx = np.flatnonzero(d.code == c)
        k = int(_KIND_NB[c])
        r = np.zeros(idx.size, _vdtype(c, nuv, bdt))
        r["pos"], r["normal"], r["uv"], r["edge"] = d.pos[idx], d.normal[idx], d.uv[idx], d.edge[idx]
        if nuv:
            r["add"] = d.add[idx]
        r["type"] = c
        flat = starts[idx][:, None] + np.arange(k)
        r["bones"] = d.bones[flat]
        if c in (1, 3):
            r["w"][:, 0] = d.weights[starts[idx]]
        elif c in (2, 4):
            r["w"] = d.weights[flat]
        if c == 3:
            r["sdef"] = d.sdef
        bad = [n for n in r.dtype.names if r.dtype[n].base.kind == "f" and not np.isfinite(r[n]).all()]
        if bad:
            raise ValueError(f"vertex {', '.join(bad)} values do not fit in a float32")
        groups.append((idx, r))
    if len(groups) == 1:
        o.put(groups[0][1].tobytes())
        return
    per = np.zeros(d.n, np.int64)
    for idx, r in groups:
        per[idx] = r.dtype.itemsize
    m = np.zeros((d.n, int(per.max())), np.uint8)
    for idx, r in groups:
        m[idx, :r.dtype.itemsize] = r.view(np.uint8).reshape(idx.size, -1)
    o.put(m[np.arange(m.shape[1]) < per[:, None]].tobytes())


def _w_offsets(o, mo, i, g):
    """Offsets of morph `mo` (index i); numpy kinds come from the arrays gathered by validation."""
    kind = mo.kind
    if i in g["morphs"]:
        idx, data = g["morphs"][i]
        k = data.shape[1]
        r = np.empty(idx.size, np.dtype([("i", _UDT[o.sizes[0]]), ("d", "<f4", (k,))]))
        r["i"], r["d"] = idx, data
        o.put(r.tobytes())
    elif kind in ("group", "flip"):
        st = _S("<" + o.fo + "f")
        for mi, ratio in mo.offsets:
            o.put(st.pack(mi, ratio))
    elif kind == "bone":
        st = _S("<" + o.fb + "3f4f")
        for bi, t, q in mo.offsets:
            o.put(st.pack(bi, *t, *q))
    elif kind == "material":
        st = _S("<" + o.fm + "b")
        for off in mo.offsets:
            m = _mat_offset("", off)
            o.put(st.pack(m.index, m.op))
            o.put(_MM_FLOATS.pack(*m.diffuse, *m.specular, m.shininess, *m.ambient, *m.edge_color, m.edge_size,
                                  *m.texture, *m.sphere, *m.toon))
    else:   # impulse
        st = _S("<" + o.fr + "B3f3f")
        for bi, local, vel, tq in mo.offsets:
            o.put(st.pack(bi, 1 if local else 0, *vel, *tq))


def _w_bones(o, bones):
    o.put(_I.pack(len(bones)))
    for b in bones:
        o.text(b.name)
        o.text(b.name_en)
        o.put(_F3.pack(*b.pos))
        o.bone(b.parent)
        local = b.local_x is not None and b.local_z is not None
        grant = b.grant_rotate or b.grant_move
        flags = (b.tail_offset is None) | (bool(b.rotatable) << 1) | (bool(b.movable) << 2) | (bool(b.visible) << 3) \
            | (bool(b.controllable) << 4) | ((b.ik is not None) << 5) | (bool(b.grant_rotate) << 8) \
            | (bool(b.grant_move) << 9) | ((b.fixed_axis is not None) << 10) | (local << 11) \
            | (bool(b.after_physics) << 12) | ((b.ext_parent is not None) << 13)
        o.put(_IH.pack(b.layer, flags))
        if b.tail_offset is None:
            o.bone(b.tail_bone)
        else:
            o.put(_F3.pack(*b.tail_offset))
        if grant:
            o.bone(b.grant_parent)
            o.put(_F.pack(b.grant_ratio))
        if b.fixed_axis is not None:
            o.put(_F3.pack(*b.fixed_axis))
        if local:
            o.put(_F3.pack(*b.local_x))
            o.put(_F3.pack(*b.local_z))
        if b.ext_parent is not None:
            o.put(_I.pack(b.ext_parent))
        if b.ik is not None:
            ik = b.ik
            o.bone(ik.target)
            o.put(_S("<ifi").pack(ik.loops, ik.angle, len(ik.links)))
            for link in ik.links:
                bone, has, lo, hi = _ik_link("", link)
                o.bone(bone)
                o.put(_B.pack(1 if has else 0))
                if has:
                    o.put(_F3.pack(*lo))
                    o.put(_F3.pack(*hi))


def _index_sizes(model, override):
    """Index sizes (vertex, texture, material, bone, morph, body) of `model`, raised by the optional override dict."""
    need = [_isz(len(model.vertices), False)] + [
        _isz(len(x), True) for x in (model.textures, model.materials, model.bones, model.morphs, model.bodies)]
    if override:
        unknown = set(override) - set(_SIZE_KEYS)
        if unknown:
            raise ValueError(f"unknown index_sizes key(s) {sorted(unknown)}; expected {list(_SIZE_KEYS)}")
        for key, size in override.items():
            i = _SIZE_KEYS.index(key)
            if size not in (1, 2, 4) or size < need[i]:
                raise ValueError(f"index_sizes[{key!r}]={size!r}: must be 1, 2 or 4 and at least {need[i]} for this model")
            need[i] = size
    return tuple(need)


def _w_materials(o, materials):
    o.put(_I.pack(len(materials)))
    st_tex = _S("<" + o.ft + o.ft + "BB")
    for m in materials:
        o.text(m.name)
        o.text(m.name_en)
        flags = bool(m.double_sided) | (bool(m.ground_shadow) << 1) | (bool(m.self_shadow_map) << 2) \
            | (bool(m.self_shadow) << 3) | (bool(m.edge) << 4) | (bool(m.vertex_color) << 5) \
            | (bool(m.point_draw) << 6) | (bool(m.line_draw) << 7)
        o.put(_MAT_A.pack(*m.diffuse, *m.specular, m.shininess, *m.ambient, flags, *m.edge_color, m.edge_size))
        o.put(st_tex.pack(m.texture, m.sphere_texture, m.sphere_mode, 1 if m.toon_shared else 0))
        if m.toon_shared:
            o.put(_SB.pack(m.toon))
        else:
            o.tex(m.toon)
        o.text(m.memo)
        o.put(_I.pack(m.index_count))


def _w_morphs(o, morphs, g):
    o.put(_I.pack(len(morphs)))
    st = _S("<BBi")
    for i, mo in enumerate(morphs):
        o.text(mo.name)
        o.text(mo.name_en)
        o.put(st.pack(mo.panel, _MORPH_CODE[mo.kind], len(mo.offsets)))
        _w_offsets(o, mo, i, g)


def _w_frames(o, frames):
    o.put(_I.pack(len(frames)))
    st = _S("<Bi")
    st_bone, st_morph = _S("<B" + o.fb), _S("<B" + o.fo)
    for f in frames:
        o.text(f.name)
        o.text(f.name_en)
        o.put(st.pack(1 if f.special else 0, len(f.items)))
        for kind, idx in f.items:
            o.put(st_bone.pack(0, idx) if kind == "bone" else st_morph.pack(1, idx))


def _w_bodies(o, bodies):
    o.put(_I.pack(len(bodies)))
    for b in bodies:
        o.text(b.name)
        o.text(b.name_en)
        o.bone(b.bone)
        o.put(_BODY.pack(b.group, b.mask, b.shape, *b.size, *b.pos, *b.rot, b.mass, b.linear_damping,
                         b.angular_damping, b.restitution, b.friction, b.mode))


def _w_joints(o, joints):
    o.put(_I.pack(len(joints)))
    st = _S("<B" + o.fr + o.fr)
    for j in joints:
        o.text(j.name)
        o.text(j.name_en)
        o.put(st.pack(j.kind, j.body_a, j.body_b))
        o.put(_JOINT.pack(*j.pos, *j.rot, *j.move_lo, *j.move_hi, *j.rot_lo, *j.rot_hi, *j.spring_move, *j.spring_rot))


def to_bytes(model, index_sizes=None):
    """Serialise `model` to PMX bytes (version and encoding from the model); runs `model.validate()` first.

    Index sizes are the smallest that fit (module docstring). `index_sizes` optionally forces larger ones, e.g.
    {'bone': 4, 'texture': 2}; keys are vertex, texture, material, bone, morph, body and values 1, 2 or 4 (a size
    smaller than the model needs raises ValueError). Raises ValueError for an invalid model (or numbers that do not
    fit the file format). Returns bytes."""
    g = _validate(model)
    ver = _version(model.version)
    codec_id, codec = _encoding(model.encoding)
    nuv = model.add_uv_count
    sizes = _index_sizes(model, index_sizes)
    o = _Out(codec, sizes)
    try:
        o.put(b"PMX " + _S("<fB8B").pack(ver, 8, codec_id, nuv, *sizes))
        for s in (model.name, model.name_en, model.comment, model.comment_en):
            o.text(s)
        _w_vertices(o, g["vertices"], nuv)
        o.put(_I.pack(len(g["faces"])))
        o.put(g["faces"].astype(_UDT[sizes[0]]).tobytes())
        o.put(_I.pack(len(model.textures)))
        for t in model.textures:
            o.text(t)
        _w_materials(o, model.materials)
        _w_bones(o, model.bones)
        _w_morphs(o, model.morphs, g)
        _w_frames(o, model.frames)
        _w_bodies(o, model.bodies)
        _w_joints(o, model.joints)
        if ver >= 2.1:
            o.put(_I.pack(0))                 # soft bodies: none
    except (struct.error, OverflowError) as e:
        raise ValueError(f"cannot pack the model as PMX: {e}") from None
    return b"".join(o.parts)


def write(model, path):
    """Validate `model` and write it to `path` as a PMX file (see to_bytes). Returns None."""
    data = to_bytes(model)
    with open(path, "wb") as fh:
        fh.write(data)


# ====================================================================================================== reader
class _Reader:
    """Cursor over the file bytes with the header's index sizes and text codec."""

    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, st):
        """Unpack a Struct at the cursor and advance; returns the tuple."""
        v = st.unpack_from(self.d, self.p)
        self.p += st.size
        return v

    def one(self, st):
        v = st.unpack_from(self.d, self.p)[0]
        self.p += st.size
        return v

    def count(self, item_min=1):
        """Read an int32 item count, rejecting values the remaining bytes cannot hold."""
        at = self.p
        n = self.one(_I)
        if n < 0 or n * item_min > len(self.d) - self.p:
            raise ValueError(f"corrupt PMX data: implausible item count {n} at byte {at}")
        return n

    def text(self):
        at = self.p
        n = self.one(_I)
        if n < 0 or n > len(self.d) - self.p:
            raise ValueError(f"corrupt PMX data: bad text length {n} at byte {at}")
        s = self.d[self.p:self.p + n].decode(self.codec, "replace")
        self.p += n
        return s

    def bone(self):
        return self.one(self.st_bone)

    def tex(self):
        return self.one(self.st_tex)


def _r_header(r):
    d = r.d
    if len(d) < 9 or d[:3] != b"PMX":
        raise ValueError("not a PMX file (bad signature)")
    r.p = 4
    r.version = _version(r.one(_F))
    n = r.one(_B)
    if n < 8:
        raise ValueError(f"corrupt PMX header: {n} global values, expected at least 8")
    g = r.take(_S("<8B"))
    r.p += n - 8                                  # tolerate extra globals of future versions
    enc, r.nuv = g[0], g[1]
    if enc not in (0, 1):
        raise ValueError(f"corrupt PMX header: unknown text encoding id {enc}")
    r.encoding = _CODEC_NAMES[enc]
    r.codec = _CODECS[r.encoding][1]
    for size in g[2:]:
        if size not in (1, 2, 4):
            raise ValueError(f"corrupt PMX header: index size {size} (expected 1, 2 or 4)")
    r.sizes = g[2:]
    if r.nuv > 4:
        raise ValueError(f"corrupt PMX header: {r.nuv} additional UVs (maximum 4)")
    r.ft, r.fm, r.fb, r.fo, r.fr = (_SF[s] for s in g[3:8])
    r.st_bone, r.st_tex = _S("<" + r.fb), _S("<" + r.ft)


def _r_vertices(r, n):
    nuv = r.nuv
    d = r.d
    pre = 32 + 16 * nuv
    bdt = _SDT[r.sizes[3]]
    dts = [_vdtype(c, nuv, bdt) for c in range(5)]
    size_of = [dt.itemsize for dt in dts] + [0] * 251
    starts, p = [], r.p
    try:
        for _ in range(n):
            starts.append(p)
            p += size_of[d[p + pre]]
    except IndexError:
        raise ValueError("truncated PMX data in the vertex section") from None
    buf = np.frombuffer(d, np.uint8)
    starts = np.array(starts, np.int64)
    types = buf[starts + pre] if n else np.zeros(0, np.uint8)
    bad = np.flatnonzero(types > 4)
    if bad.size:
        raise ValueError(f"corrupt PMX data: vertex {int(bad[0])} has unknown weight type {int(types[bad[0]])}")
    if p > len(d):
        raise ValueError("truncated PMX data in the vertex section")
    out = [None] * n
    for c in np.unique(types).tolist():
        idx = np.flatnonzero(types == c)
        dt = dts[c]
        rec = sliding_window_view(buf, dt.itemsize)[starts[idx]].view(dt).reshape(-1)
        m = idx.size
        pos = list(map(tuple, rec["pos"].tolist()))
        nrm = list(map(tuple, rec["normal"].tolist()))
        uv = list(map(tuple, rec["uv"].tolist()))
        bones = list(map(tuple, rec["bones"].tolist()))
        add = [tuple(map(tuple, a)) for a in rec["add"].tolist()] if nuv else [()] * m
        kind, sdef = "", [None] * m
        if c == 0:
            weights = [(1.0,)] * m
        elif c in (1, 3):
            weights = [(w, 1.0 - w) for w in rec["w"][:, 0].tolist()]
            if c == 3:
                kind = "SDEF"
                sdef = [(tuple(s[0:3]), tuple(s[3:6]), tuple(s[6:9])) for s in rec["sdef"].tolist()]
        else:
            weights = list(map(tuple, rec["w"].tolist()))
            kind = "QDEF" if c == 4 else ""
        edge = rec["edge"].tolist()
        verts = map(PmxVertex, pos, nrm, uv, bones, weights, add, sdef, repeat(kind), edge)
        for i, v in zip(idx.tolist(), verts):
            out[i] = v
    r.p = p
    return out


def _r_offsets(r, kind, n):
    """Read `n` offsets of a morph of the given kind."""
    if kind == "vertex" or kind in _MORPH_UV_KINDS:
        k = 3 if kind == "vertex" else 4
        rec_dt = np.dtype([("i", _UDT[r.sizes[0]]), ("d", "<f4", (k,))])
        need = n * rec_dt.itemsize
        if need > len(r.d) - r.p:
            raise ValueError("truncated PMX data in a morph")
        rec = np.frombuffer(r.d, rec_dt, n, r.p)
        r.p += need
        return list(zip(rec["i"].tolist(), map(tuple, rec["d"].tolist())))
    out = []
    if kind in ("group", "flip"):
        st = _S("<" + r.fo + "f")
        for _ in range(n):
            out.append(r.take(st))
    elif kind == "bone":
        st = _S("<" + r.fb + "3f4f")
        for _ in range(n):
            t = r.take(st)
            out.append((t[0], tuple(t[1:4]), tuple(t[4:8])))
    elif kind == "material":
        st = _S("<" + r.fm + "b")
        for _ in range(n):
            index, op = r.take(st)
            f = r.take(_MM_FLOATS)
            out.append(MaterialMorphOffset(index, op, f[0:4], f[4:7], f[7], f[8:11], f[11:15], f[15], f[16:20],
                                           f[20:24], f[24:28]))
    else:   # impulse
        st = _S("<" + r.fr + "B3f3f")
        for _ in range(n):
            t = r.take(st)
            out.append((t[0], bool(t[1]), tuple(t[2:5]), tuple(t[5:8])))
    return out


def _r_bone(r):
    name, name_en = r.text(), r.text()
    pos = r.take(_F3)
    parent = r.bone()
    layer, flags = r.take(_IH)
    b = PmxBone(name, name_en, pos, parent, layer)
    if flags & 0x0001:
        b.tail_bone = r.bone()
    else:
        b.tail_offset = r.take(_F3)
    b.rotatable, b.movable = bool(flags & 0x0002), bool(flags & 0x0004)
    b.visible, b.controllable = bool(flags & 0x0008), bool(flags & 0x0010)
    b.grant_rotate, b.grant_move = bool(flags & 0x0100), bool(flags & 0x0200)
    if flags & 0x0300:
        b.grant_parent = r.bone()
        b.grant_ratio = r.one(_F)
    if flags & 0x0400:
        b.fixed_axis = r.take(_F3)
    if flags & 0x0800:
        b.local_x = r.take(_F3)
        b.local_z = r.take(_F3)
    b.after_physics = bool(flags & 0x1000)
    if flags & 0x2000:
        b.ext_parent = r.one(_I)
    if flags & 0x0020:
        target = r.bone()
        loops, angle, nlinks = r.take(_S("<ifi"))
        if nlinks < 0 or nlinks > len(r.d) - r.p:
            raise ValueError(f"corrupt PMX data: bone {name!r} has {nlinks} IK links")
        links = []
        for _ in range(nlinks):
            bone = r.bone()
            if r.one(_B):
                links.append((bone, True, r.take(_F3), r.take(_F3)))
            else:
                links.append((bone, False, _Z3, _Z3))
        b.ik = PmxIK(target, loops, angle, links)
    return b


def from_bytes(data):
    """Parse PMX 2.0 / 2.1 bytes into a PmxModel (any index sizes, UTF-16LE or UTF-8).

    BDEF vertices come back with kind '' (SDEF / QDEF explicit), BDEF2 / SDEF weights as (w0, 1 - w0), texts as str.
    Raises ValueError for a bad signature, an unsupported version, or truncated / corrupt data and
    NotImplementedError when the file contains soft bodies (PMX 2.1). Returns a PmxModel."""
    if isinstance(data, str):
        raise TypeError("from_bytes() needs the file contents as bytes, got str (use read(path) to read a file)")
    r = _Reader(bytes(data))
    try:
        return _parse(r)
    except struct.error as e:
        raise ValueError(f"truncated or corrupt PMX data near byte {r.p}: {e}") from None


def _r_faces(r):
    n = r.count(r.sizes[0])
    arr = np.frombuffer(r.d, _UDT[r.sizes[0]], n, r.p)
    r.p += n * arr.dtype.itemsize
    return arr.tolist()


def _r_materials(r):
    out = []
    st_tex = _S("<" + r.ft + r.ft + "BB")
    for _ in range(r.count(60)):
        mt = PmxMaterial(r.text(), r.text())
        a = r.take(_MAT_A)
        mt.diffuse, mt.specular, mt.shininess, mt.ambient = a[0:4], a[4:7], a[7], a[8:11]
        f = a[11]
        mt.double_sided, mt.ground_shadow, mt.self_shadow_map = bool(f & 1), bool(f & 2), bool(f & 4)
        mt.self_shadow, mt.edge = bool(f & 8), bool(f & 16)
        if r.version >= 2.1:                          # bits 5..7 are undefined in 2.0 files
            mt.vertex_color, mt.point_draw, mt.line_draw = bool(f & 32), bool(f & 64), bool(f & 128)
        mt.edge_color, mt.edge_size = a[12:16], a[16]
        mt.texture, mt.sphere_texture, mt.sphere_mode, shared = r.take(st_tex)
        if shared > 1:
            raise ValueError(f"corrupt PMX data: toon flag {shared} of material {mt.name!r}")
        mt.toon_shared = bool(shared)
        mt.toon = r.one(_SB) if shared else r.tex()
        mt.memo = r.text()
        mt.index_count = r.one(_I)
        out.append(mt)
    return out


def _r_morphs(r):
    out = []
    st = _S("<BBi")
    for _ in range(r.count(14)):
        mo = PmxMorph(r.text(), r.text())
        mo.panel, code, n = r.take(st)
        if code not in _MORPH_NAME:
            raise ValueError(f"corrupt PMX data: unknown morph type {code} in morph {mo.name!r}")
        mo.kind = _MORPH_NAME[code]
        if n < 0 or n > len(r.d) - r.p:
            raise ValueError(f"corrupt PMX data: morph {mo.name!r} has {n} offsets")
        mo.offsets = _r_offsets(r, mo.kind, n)
        out.append(mo)
    return out


def _r_frames(r):
    out = []
    st, st_morph = _S("<Bi"), _S("<" + r.fo)
    for _ in range(r.count(9)):
        fr = PmxFrame(r.text(), r.text())
        special, n = r.take(st)
        fr.special = bool(special)
        if n < 0 or n > len(r.d) - r.p:
            raise ValueError(f"corrupt PMX data: frame {fr.name!r} has {n} items")
        for _item in range(n):
            kind = r.one(_B)
            if kind == 0:
                fr.items.append(("bone", r.bone()))
            elif kind == 1:
                fr.items.append(("morph", r.one(st_morph)))
            else:
                raise ValueError(f"corrupt PMX data: frame {fr.name!r} has an item of kind {kind}")
        out.append(fr)
    return out


def _r_bodies(r):
    out = []
    for _ in range(r.count(60)):
        b = PmxBody(r.text(), r.text(), r.bone())
        t = r.take(_BODY)
        b.group, b.mask, b.shape = t[0], t[1], t[2]
        b.size, b.pos, b.rot = t[3:6], t[6:9], t[9:12]
        b.mass, b.linear_damping, b.angular_damping, b.restitution, b.friction = t[12:17]
        b.mode = t[17]
        out.append(b)
    return out


def _r_joints(r):
    out = []
    st = _S("<B" + r.fr + r.fr)
    for _ in range(r.count(60)):
        j = PmxJoint(r.text(), r.text())
        j.kind, j.body_a, j.body_b = r.take(st)
        t = r.take(_JOINT)
        j.pos, j.rot, j.move_lo, j.move_hi = t[0:3], t[3:6], t[6:9], t[9:12]
        j.rot_lo, j.rot_hi, j.spring_move, j.spring_rot = t[12:15], t[15:18], t[18:21], t[21:24]
        out.append(j)
    return out


def _parse(r):
    _r_header(r)
    m = PmxModel(version=r.version, encoding=r.encoding, add_uv_count=r.nuv)
    m.name, m.name_en, m.comment, m.comment_en = r.text(), r.text(), r.text(), r.text()
    m.vertices = _r_vertices(r, r.count(32 + 16 * r.nuv + 1 + r.sizes[3] + 4))
    m.faces = _r_faces(r)
    m.textures = [r.text() for _ in range(r.count(4))]
    m.materials = _r_materials(r)
    m.bones = [_r_bone(r) for _ in range(r.count(24))]
    m.morphs = _r_morphs(r)
    m.frames = _r_frames(r)
    m.bodies = _r_bodies(r)
    m.joints = _r_joints(r)
    if r.version >= 2.1 and len(r.d) - r.p >= 4 and r.one(_I) != 0:
        raise NotImplementedError("PMX 2.1 soft bodies are not supported")
    return m


def read(path):
    """Read the PMX file at `path` into a PmxModel (see from_bytes for the conventions and exceptions)."""
    with open(path, "rb") as fh:
        return from_bytes(fh.read())
