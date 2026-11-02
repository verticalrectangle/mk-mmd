"""The festival speaker stack (`card = "library:speaker_stack"`): per cabinet k (bottom up) the objects
`<name>_cab<k>_shell`, `_drivers` (the woofers, with the shape key `pump`), `_hardware` (rigging, handles, feet, the
connector plate), `_ports` (a sub's slot ports), `_horn` (a top's horn), `_grille` (the wire mesh), and the hidden
collider box `<name>_cab<k>_col`, under the root. Cabinets of one kind share their meshes, so all their cones pump
together from one driver on the shared shape key.

    [[prop]]
    name = "stack_L"
    card = "library:speaker_stack"
    at = [-3.0, 1.5, 0.0]
    yaw = 15
    slots = { stack = ["sub", "sub", "top"], cabinet = "base", cap = "love" }

    [[key]]                                 # the cones punch out on every hit of the bass lane
    target = "stack_L"
    prop = "pump"
    lane = "bass"
    pulse = [[0.0, 1.0], [0.12, 0.15], [0.3, 0.0]]

The coating and the baffle are a fine noise bump on a rough base (a textured cabinet paint); the cone paper rough, the dust
caps and the horn glossy, the hardware metal. Numbers, roles and the card: speaker_stack_layout.py."""
import math

import bpy
import numpy as np

from .. import mesh as MESH
from .. import shell as SH
from . import register
from . import showkit as SK
from . import speaker_stack_layout as LAY
from .cafe_kit import Kit, clear_drivers, drive

LOOK = {  # role: Principled settings, and the bump of a textured coating (noise scale, strength)
    "cabinet": dict(rough=0.62, spec=0.4, bump=(700.0, 0.35)),
    "baffle": dict(rough=0.8, spec=0.3, bump=(900.0, 0.2)),
    "grille": dict(rough=0.38, metal=0.35, spec=0.5),
    "cone": dict(rough=0.78, spec=0.3, sheen=0.25),
    "surround": dict(rough=0.5, spec=0.45, coat=0.15),
    "cap": dict(rough=0.32, spec=0.5, coat=0.6),
    "horn": dict(rough=0.36, spec=0.5, coat=0.3),
    "hardware": dict(rough=0.3, metal=0.85, spec=0.5),
    "rubber": dict(rough=0.72, spec=0.35),
    "port": dict(rough=0.9, spec=0.2),
}


def _material(K, role, color, rough=0.5, metal=0.0, spec=0.5, coat=0.0, sheen=0.0, bump=None):
    """A Principled material of a colour role; a textured coating gets a fine noise bump (in the object's own space)."""
    m = K.simple_mat(role, color, rough=rough, metal=metal, coat=coat, spec=spec, sheen=sheen)
    m.diffuse_color = color
    if bump is None:
        return m
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, -300)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-700, -300)
    noise.inputs["Scale"].default_value = bump[0]
    noise.inputs["Detail"].default_value = 2.0
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.location = (-450, -300)
    bp.inputs["Strength"].default_value = bump[1]
    bp.inputs["Distance"].default_value = 0.0004
    nt.links.new(noise.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def _pump_key(o, disp, root):
    """The shape key `pump` of a drivers object (its cones pushed out by `disp` at 1), driven by the root's `pump`."""
    if o.data.shape_keys is None:
        o.shape_key_add(name="Basis", from_mix=False)
    kb = o.data.shape_keys.key_blocks.get("pump") or o.shape_key_add(name="pump", from_mix=False)
    co = np.empty(len(o.data.vertices) * 3, np.float32)
    o.data.vertices.foreach_get("co", co)
    kb.data.foreach_set("co", co + disp.astype(np.float32).ravel())
    kb.slider_min, kb.slider_max = 0.0, 1.0
    clear_drivers(o.data.shape_keys)
    drive(o.data.shape_keys, 'key_blocks["pump"].value', "clamp(pump, 0, 1)", var=("pump", root, '["pump"]'))


def _shell(name, kind, coll, root, loc, mat_of):
    """A cabinet's shell object with its holes, tunnels and horn mouth cut (Boolean, baked: a plain mesh), its edges
    sharper than 35 degrees split (the cuts' rims) so the flat faces round them shade flat."""
    cuts = [SH.cutter_object(f"{name}_cut{i}", m, coll, root, mat_of, loc=loc) for i, m in enumerate(LAY.cutters(kind))]
    o = SH.shell_object(name, LAY.shell_mesh(kind), coll, root, mat_of, levels=0, render_levels=0, cutters=cuts,
                        cutter_role=LAY.MATS.index("port"), loc=loc, bake_it=True)
    o.data.set_sharp_from_angle(angle=math.radians(35.0))
    return o


@register("speaker_stack")
def speaker_stack(name, coll, root, slots=None):
    slots = dict(slots or {})
    stack, grille, P = LAY.options(slots)
    roles = SK.roles(slots, LAY.ROLES, "speaker_stack")
    K = Kit(name, coll, root, slots)
    clear_drivers(root)
    K.prop("pump", P["pump"], 0.0, 1.0, LAY.PARAMS["pump"][2])
    mats = {role: _material(K, role, SK.rgba(roles[role], slots, f"speaker_stack: {role}"), **LOOK[role])
            for role in LAY.MATS}
    mat_of = lambda i: mats[LAY.MATS[i]]                                                       # noqa: E731
    data = {}                                       # (kind, part) -> the mesh every cabinet of that kind shares
    for c in LAY.placed(stack):
        kind, k = c["kind"], c["k"]
        loc = (0.0, c["y0"], c["z0"])
        for part in LAY.parts(kind, grille):
            oname = f"{name}_cab{k}_{part}"
            SH.purge(oname)
            if (kind, part) not in data and part == "shell":
                o = _shell(oname, kind, coll, root, loc, mat_of)
                data[(kind, part)] = o.data
                o.data.name = f"{name}_{kind}_{part}"
                continue
            if (kind, part) not in data:
                mname = f"{name}_{kind}_{part}"
                old = bpy.data.meshes.get(mname)
                if old is not None and old.users == 0:
                    bpy.data.meshes.remove(old)
                data[(kind, part)], _ = SH.mesh_data(mname, LAY.part_mesh(kind, part), mat_of)
            o = bpy.data.objects.new(oname, data[(kind, part)])
            coll.objects.link(o)
            o.parent = root
            o.location = loc
            if part == "drivers" and o.data.shape_keys is None:
                _pump_key(o, LAY.drivers(kind)[1], root)
        centre, size = LAY.collider_box(kind)
        SH.purge(f"{name}_cab{k}_col")
        MESH.box(f"{name}_cab{k}_col", (centre[0], c["y0"] + centre[1], c["z0"] + centre[2]), size, coll, root,
                 collider=True)
    return LAY.card(name, stack, grille, roles, P)
