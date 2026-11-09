"""The boom microphone stand with its microphone (`card = "library:mic_stand"`): a round base on the floor, a two-part
pole with its clutch, a boom on a swivel rising toward +Y with a counterweight behind, and at its end a clip holding a
handheld vocal microphone in line with it, its cord run back under the boom, down the pole and away along the floor.
The singer stands on the +Y side.

    [[prop]]
    name = "mic"
    card = "library:mic_stand"
    at = [-0.30, -0.52, 0.0]
    yaw = -40                            # the boom toward her mouth
    slots = { height = 1.24, angle = 28, boom = 0.36 }

Objects `<name>_base`, `_pole`, `_boom` (the swivel, the tube, the counterweight, the clip), `_mic_body`, `_mic_grille`,
`_mic_connector`, `_cable` (a bevelled curve). Numbers, options, roles and the card: mic_stand_layout.py."""
import bpy

from .. import shell as SH
from . import mic_stand_layout as LAY
from . import register
from .cafe_kit import Kit


def _cable(K, mat, opt):
    name = f"{K.name}_cable"
    SH.purge(name)
    old = bpy.data.curves.get(name)
    if old is not None and old.users == 0:
        bpy.data.curves.remove(old)
    pts = LAY.cable_path(opt)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution, cu.resolution_u, cu.use_fill_caps = LAY.CABLE_R, 2, 12, True
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    sp.order_u, sp.use_endpoint_u = 4, True
    for p, q in zip(sp.points, pts):
        p.co = (float(q[0]), float(q[1]), float(q[2]), 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    K.coll.objects.link(o)
    o.parent = K.root
    return o


@register("mic_stand")
def mic_stand(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    opt = LAY.options(K.slots)
    finish = {"stand": (0.32, 1.0), "base": (0.5, 0.3), "body": (0.45, 0.6), "grille": (0.35, 1.0), "hardware": (0.3, 1.0),
              "cable": (0.55, 0.0)}
    mats = {role: K.simple_mat(role, K.role(role, LAY.ROLES), rough=r, metal=m) for role, (r, m) in finish.items()}
    roles = list(LAY.MATS)
    mat_of = lambda i: mats[roles[i]]                                                          # noqa: E731
    for key, mesh in LAY.meshes(opt).items():
        SH.mesh_object(f"{name}_{key}", mesh, coll, root, mat_of)
    _cable(K, mats["cable"], opt)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in LAY.ROLES}, opt)
