"""The giant toothpaste tube (`card = "library:toothpaste_tube"`): `<name>_body` (crimp, body, shoulder, threaded neck and
the open mouth: one cage, banded in its colour roles) under the root, standing on its crimp along +Z, the nozzle at the
top, and with `capped = true` in the slots `<name>_cap` (a ribbed screw cap). The root property `squeeze` (0 .. 1) drives
the body's shape key "squeeze" (pressed flat round the hold); a [[ribbon]] `from = "prop:<name>.nozzle"` keys it while
the paste comes out. The paste's own colours are the slot `stripes` ("love,gold,foam,iris,rose"). Numbers, roles and
the card: toothpaste_tube_layout.py."""
import bpy
import numpy as np

from .. import shell as SH
from . import register
from . import toothpaste_tube_layout as LAY
from .cafe_kit import Kit, clear_drivers, drive

LOOK = {"body": dict(rough=0.3, coat=0.4, spec=0.5), "shoulder": dict(rough=0.28, coat=0.5, spec=0.55),
        "paste": dict(rough=0.2, coat=0.8, spec=0.55), "cap": dict(rough=0.35, coat=0.3, spec=0.5)}


def _roles(slots):
    """Each colour role's slot or hex: the prop's `slots`, its `stripes` list read into stripe0 .. stripe4."""
    out = {role: str(slots.get(role, LAY.ROLES[role])) for role in LAY.MATS}
    if slots.get("stripes"):
        cols = [c.strip() for c in str(slots["stripes"]).split(",") if c.strip()]
        if len(cols) != len(LAY.STRIPE_ROLES):
            raise ValueError(f"toothpaste_tube: stripes = {slots['stripes']!r}: five colours, comma separated")
        out.update(zip(LAY.STRIPE_ROLES, cols))
    return out


@register("toothpaste_tube")
def toothpaste_tube(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    roles = _roles(K.slots)
    capped = str(K.slots.get("capped", "false")).lower() in ("1", "true", "yes")
    clear_drivers(root)
    root["squeeze"] = 0.0
    root.id_properties_ui("squeeze").update(description="0..1: the body pressed flat round the hold",
                                            min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
    mats = {}
    for role in LAY.MATS:
        m = K.simple_mat(role, K.role(role, roles), **LOOK.get(role, LOOK["body"]))
        m.diffuse_color = K.role(role, roles)
        mats[role] = m
    mat_of = lambda i: mats[LAY.MATS[i]]                                                      # noqa: E731
    SH.purge(f"{name}_body")
    me, _ = SH.mesh_data(f"{name}_body", LAY.body_mesh(0.0), mat_of, normals=False)
    body = bpy.data.objects.new(f"{name}_body", me)
    coll.objects.link(body)
    body.parent = root
    body.shape_key_add(name="Basis", from_mix=False)
    key = body.shape_key_add(name="squeeze", from_mix=False)
    key.data.foreach_set("co", LAY.body_mesh(1.0).V.astype(np.float32).ravel())
    clear_drivers(me.shape_keys)
    drive(me.shape_keys, 'key_blocks["squeeze"].value', "min(max(squeeze, 0), 1)", var=("squeeze", root, '["squeeze"]'))
    if capped:
        SH.mesh_object(f"{name}_cap", LAY.cap_mesh(), coll, root, mat_of)
    return LAY.card(name, roles, capped)
