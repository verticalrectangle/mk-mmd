"""The handheld vocal microphone (`card = "library:handheld_mic"`): three lathed parts (`<name>_connector`, `<name>_body`:
the handle with its collar, `<name>_grille`: the ball), the empty `<name>_jack` under the tail, the cord `<name>_cable` (a
bevelled NURBS curve straight down from the jack: `attach` with `cable` re-hangs it to the floor and the sim swings it,
docs/design.md: Posing) and the hidden cylinder `<name>_col_body`. Numbers, roles and the card: handheld_mic_layout.py."""
import bpy
import numpy as np

from ....core import shell as S
from .. import shell as SH
from ..mesh import cylinder as _collider_cylinder
from . import handheld_mic_layout as LAY
from . import register
from .cafe_kit import Kit


def _lathe(pts, seg, mat, r=0.0008):
    return S.lathe(S.fillet(np.asarray(pts, float), r, 3), seg=seg, mat=mat)


def _cable(K, mat):
    """`<name>_cable`: a NURBS curve with a round bevel from the jack straight down, 0.6 m (re-hung by the pose stage)."""
    name = f"{K.name}_cable"
    SH.purge(name)
    old = bpy.data.curves.get(name)
    if old is not None and old.users == 0:
        bpy.data.curves.remove(old)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution = LAY.CABLE_R, 2
    sp = cu.splines.new("NURBS")
    sp.points.add(3)
    sp.order_u, sp.use_endpoint_u = 4, True
    for p, z in zip(sp.points, (-0.002, -0.2, -0.4, -0.6)):
        p.co = (0.0, 0.0, z, 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    K.coll.objects.link(o)
    o.parent = K.root
    return o


@register("handheld_mic")
def handheld_mic(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    mats = {"body": K.simple_mat("body", K.role("body", LAY.ROLES), rough=0.45, metal=0.6),
            "grille": K.simple_mat("grille", K.role("grille", LAY.ROLES), rough=0.35, metal=1.0),
            "hardware": K.simple_mat("hardware", K.role("hardware", LAY.ROLES), rough=0.3, metal=1.0),
            "cable": K.simple_mat("cable", K.role("cable", LAY.ROLES), rough=0.55)}
    roles = list(mats)
    mat_of = lambda i: mats[roles[i]]                                                          # noqa: E731
    SH.mesh_object(f"{name}_connector", _lathe(LAY.connector_profile(), 32, roles.index("hardware")), coll, root, mat_of)
    SH.mesh_object(f"{name}_body", _lathe(LAY.body_profile(), 48, roles.index("body")), coll, root, mat_of)
    SH.mesh_object(f"{name}_grille", _lathe(LAY.grille_profile(), 48, roles.index("grille"), r=0.0004), coll, root, mat_of)
    _cable(K, mats["cable"])
    K.empty("jack", loc=(0.0, 0.0, -0.002), kind="ARROWS", size=0.01)
    SH.purge(f"{name}_col_body")
    _collider_cylinder(f"{name}_col_body", (0.0, 0.0, 0.5 * LAY.LENGTH), LAY.BALL_R, LAY.LENGTH, coll, root, collider=True)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in roles})
