"""The giant toothbrush (`card = "library:toothbrush"`): `<name>_body` (handle, neck and head: one smooth cage, the soft
grip panels in their own colour), `<name>_bristles` and `<name>_tufts` (the lathed tufts) under the root, standing along
+Z on the end of its handle, the bristles facing -Y. A hand holds it by the card's `use.grip` `hold`; `use.anchor`
`brush` is the middle of the bristles' tips. Numbers, roles and the card: toothbrush_layout.py."""
from .. import shell as SH
from . import register
from . import toothbrush_layout as LAY
from .cafe_kit import Kit

LOOK = {"handle": dict(rough=0.22, coat=0.6, spec=0.55), "grip": dict(rough=0.62, spec=0.3),
        "bristles": dict(rough=0.45, spec=0.4, sheen=0.4), "tufts": dict(rough=0.45, spec=0.4, sheen=0.4)}


@register("toothbrush")
def toothbrush(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    mats = {}
    for role in LAY.MATS:
        mats[role] = K.simple_mat(role, K.role(role, LAY.ROLES), **LOOK[role])
        mats[role].diffuse_color = K.role(role, LAY.ROLES)
    mat_of = lambda i: mats[LAY.MATS[i]]                                                      # noqa: E731
    SH.mesh_object(f"{name}_body", LAY.body_mesh(), coll, root, mat_of)
    for part, mesh in LAY.bristle_meshes().items():
        SH.mesh_object(f"{name}_{part}", mesh, coll, root, mat_of)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in LAY.MATS})
