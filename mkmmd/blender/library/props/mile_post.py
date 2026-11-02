"""The mile post (`card = "library:mile_post"`): a 19th-century railroad mile post (`style = "railroad"`, the default)
or a US interstate mile marker (`style = "interstate"`), its number in numerals built in code (mkmmd.core.numerals).

    [[prop]]
    name = "mp7"
    card = "library:mile_post"
    at = [2.4, 0.0, 0.0]
    slots = { number = 7 }                                    # the white post, its cast plate reading 7

    [[prop]]
    name = "marker"
    card = "library:mile_post"
    slots = { style = "interstate", number = 123 }            # the green D10 sign, 1 2 3 stacked down it

    [[prop]]                                                  # a blank plate the size of two numerals ...
    name = "mp"
    card = "library:mile_post"

    [[text]]                                                  # ... that a text writes on (a keyed count, a word)
    name = "mp_count"
    on = "mp:plate"
    value = { keys = [[0.0, 1], [2.47, 2]], format = "{:.0f}", interp = "CONSTANT" }
    color = "#f4efe4"                                         # a palette slot or #hex: paint on the iron plate

Objects under the root: railroad `<name>_post` (the timber, with the per-vertex `mk_check` its material darkens the
season checks by), `<name>_plate` (the cast plate, its bead and bolts) and `<name>_numerals` (the cast numerals); interstate
`<name>_post` (the U-channel), `<name>_panel` (the sign and its bolts), `<name>_border` and `<name>_numerals` (the sheeting's
legend: graphic layers, exempt from `form`). A blank plate has no `<name>_numerals`. Static: no live properties, no
drivers. Numbers, meshes, options and the card: mile_post_layout.py; materials: mile_post_mats.py."""
import numpy as np

from .. import shell as SH
from . import mile_post_layout as LAY
from . import mile_post_mats as MT
from . import register
from .cafe_kit import Kit


@register("mile_post")
def mile_post(name, coll, root, slots=None):
    opts = LAY.options(slots)
    style = opts["style"]
    K = Kit(name, coll, root, slots)
    col = {role: K.role(role, opts["roles"]) for role in opts["roles"]}
    mats = MT.materials(name, style, col)
    keys = LAY.MATS[style]
    mat_of = lambda i: mats[keys[i]]                                                        # noqa: E731
    meshes = LAY.parts(opts)
    for part in ("post", "plate", "panel", "border", "numerals"):
        if part not in meshes:
            SH.purge(f"{name}_{part}")                    # left by an earlier build of this name (a blank plate now)
    for part, mesh in meshes.items():
        o = SH.mesh_object(f"{name}_{part}", mesh, coll, root, mat_of)
        if style == "interstate" and part in LAY.GRAPHIC:
            SH.exempt(o, "sign sheeting")
        if style == "railroad" and part == "post":
            o.data.attributes.new("mk_check", "FLOAT", "POINT").data.foreach_set(
                "value", LAY.post_check().astype(np.float32))
    V = np.concatenate([m.V for m in meshes.values()])
    return LAY.card(name, opts, V.max(0) - V.min(0))
