"""The giant carrot (`card = "library:giant_carrot"`): a 0.60 m carrot standing on its tip along +Z, with leafy greens, that a
character chomps from the tip up. Objects under the root: `<name>_body` (the root and its crown, with the shape keys
`bite1` .. `bite4`), `<name>_greens` (stems and leaflets) and the chunks `<name>_chunk<k>_<j>` each bite throws.

    [[prop]]
    name = "carrot"
    card = "library:giant_carrot"
    slots = { bite1 = 0.2, bite2 = 0.5 }      # clip seconds of the first bites (a gag keys them CONSTANT instead)

Root properties: `bite1` .. `bite4` (the clip second of each bite; 1e4 = never), `f0` (the frame of clip second 0, from the
scene's `mk_frame0`, else its first frame) and `fps`. Drivers (Blender simple expressions, checked here): every shape key's
value, every chunk's age `a` (a custom property), location, rotation and scale; a chunk reads the root's world rotation
(TRANSFORMS variables) so it falls toward the world's floor. The numbers, the maths and the card: giant_carrot_layout.py;
the meshes: giant_carrot_parts.py; the materials: giant_carrot_mats.py."""
import re

import bpy
import numpy as np

from ....core import palette as PAL
from .. import shell as SH
from . import giant_carrot_layout as LAY
from . import giant_carrot_mats as MT
from . import giant_carrot_parts as PT
from . import register
from .cafe_kit import Kit, clear_drivers

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ROT = {"rx": "ROT_X", "ry": "ROT_Y"}


def _drive(owner, path, expr, root, index=-1, obj=None):
    """A scripted driver `expr` on owner.path[index]: its names are the root's properties (LAY.ROOT_VARS), the root's world
    rotation (LAY.TURN_VARS) or the object `obj`'s own helpers (LAY.HELPERS). Refused unless it is a Blender simple
    expression."""
    fc = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for nm in sorted(set(NAME.findall(expr)) - LAY.FUNCS - {"frame"}):
        v = d.variables.new()
        v.name = nm
        t = v.targets[0]
        if nm in LAY.ROOT_VARS:
            v.type = "SINGLE_PROP"
            t.id_type, t.id, t.data_path = "OBJECT", root, f'["{nm}"]'
        elif nm in LAY.TURN_VARS:
            v.type = "TRANSFORMS"
            t.id = root
            t.transform_type, t.transform_space, t.rotation_mode = ROT[nm], "WORLD_SPACE", "XYZ"
        elif nm in LAY.HELPERS and obj is not None:
            v.type = "SINGLE_PROP"
            t.id_type, t.id, t.data_path = "OBJECT", obj, f'["{nm}"]'
        else:
            raise ValueError(f"giant_carrot: the driver on {path} reads {nm!r}, which nothing provides ({expr})")
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"giant_carrot: the driver on {path}[{index}] is not a Blender simple expression ({expr})")
    return fc


def _body(name, coll, root, mat_of):
    """The body with its four non-additive bite keys, each driven by its bite's clip second."""
    mesh = PT.body()[0]
    SH.purge(f"{name}_body")
    me, _ = SH.mesh_data(f"{name}_body", mesh, mat_of, normals=False)         # plain smooth normals: they follow the keys
    o = bpy.data.objects.new(f"{name}_body", me)
    coll.objects.link(o)
    o.parent = root
    o.shape_key_add(name="Basis", from_mix=False)
    for k, delta in enumerate(PT.key_deltas(), 1):
        kb = o.shape_key_add(name=f"bite{k}", from_mix=False)
        kb.data.foreach_set("co", (mesh.V + delta).astype(np.float32).ravel())
        kb.slider_min, kb.slider_max = 0.0, 1.0
        _drive(me.shape_keys, f'key_blocks["bite{k}"].value', LAY.key_expression(k), root)
    return o


def _chunk(name, coll, root, c, mat_of):
    """A bitten-off piece, flying on its bite's clock (drivers)."""
    o = SH.mesh_object(LAY.chunk_name(name, c), PT.chunk_mesh(c["size"], c["seed"]), coll, root, mat_of)
    o.rotation_mode = "XYZ"
    o["a"] = 0.0
    ex = LAY.chunk_expressions(c)
    _drive(o, '["a"]', ex["a"], root, obj=o)
    for i, ch in enumerate("xyz"):
        _drive(o, "location", ex[ch], root, i, obj=o)
    for i, ch in enumerate(("rx", "ry", "rz")):
        _drive(o, "rotation_euler", ex[ch], root, i, obj=o)
    for i in range(3):
        _drive(o, "scale", ex["s"], root, i, obj=o)
    return o


@register("giant_carrot")
def giant_carrot(name, coll, root, slots=None):
    opts = LAY.options(slots)
    K = Kit(name, coll, root, slots)
    sc = bpy.context.scene
    for old in [o.name for o in bpy.data.objects if o.name.startswith(f"{name}_chunk")]:
        SH.purge(old)
    clear_drivers(root)
    for k, t in enumerate(opts["bites"], 1):
        K.prop(f"bite{k}", t, doc=f"the clip second of bite {k} ({LAY.NEVER:g}: never); key it CONSTANT, in order")
    K.prop("f0", float(sc.get("mk_frame0", sc.frame_start)), doc="the frame of clip second 0")
    K.prop("fps", float(K.fps), doc="frames per second")
    col = {role: K.role(role, opts["roles"]) for role in LAY.ROLES}
    mats = MT.materials(name, col)
    mat_of = lambda i: mats[LAY.MATS[i]]                                             # noqa: E731
    _body(name, coll, root, mat_of)
    SH.mesh_object(f"{name}_greens", PT.greens_mesh(), coll, root, mat_of)
    for c in LAY.chunks():
        _chunk(name, coll, root, c, mat_of)
    roles = {role: PAL.resolve(v, K.colors.p) for role, v in opts["roles"].items()}
    return LAY.card(name, roles, opts["bites"], PT.size())
