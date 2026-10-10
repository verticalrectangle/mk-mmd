"""The squeaky toy hammer (`card = "library:squeaky_hammer"`): `<name>_handle` and `<name>_head` (the pleated bellows and
its two caps, its origin in the head's middle, so `squash` scales it about there), under the root. The root property
`squash` (0 .. 1) drives the head's scale. Put it in a hand with `attach`, `offset` and `attach_rot` (docs/design.md:
Posing) and play `bonk` with that hand. Numbers, roles and the card: squeaky_hammer_layout.py."""
from .. import shell as SH
from . import register
from . import squeaky_hammer_layout as LAY
from .cafe_kit import Kit, clear_drivers, drive


@register("squeaky_hammer")
def squeaky_hammer(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    clear_drivers(root)
    root["squash"] = 0.0
    root.id_properties_ui("squash").update(description="0..1: the head squeezed along its axis and swollen round",
                                           min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
    mats = {role: K.simple_mat(role, K.role(role, LAY.ROLES), rough=0.38, coat=0.3) for role in LAY.MATS}
    mat_of = lambda i: mats[LAY.MATS[i]]                                                       # noqa: E731
    SH.mesh_object(f"{name}_handle", LAY.handle_mesh(), coll, root, mat_of)
    head = SH.mesh_object(f"{name}_head", LAY.head_mesh(), coll, root, mat_of, loc=(0.0, 0.0, LAY.HEAD_Z))
    var = ("squash", root, '["squash"]')
    clear_drivers(head)
    drive(head, "scale", f"1 - {LAY.SQUASH[0]} * min(max(squash, 0), 1)", 0, var)
    for i in (1, 2):
        drive(head, "scale", f"1 + {LAY.SQUASH[1]} * min(max(squash, 0), 1)", i, var)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in LAY.MATS})
