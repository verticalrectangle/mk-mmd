"""The carrot plush (`card = "library:carrot_plush"`): a soft toy carrot with an embroidered face, held to a side or a chest.
Objects under the root: `<name>_body` (minky, five gores), `<name>_leaves` (three stuffed felt leaves), and the graphic
layers (tagged `mk_form_exempt`) `<name>_face` (eyes, highlights, smile), `<name>_cheeks` and `<name>_stitches` (the
leaves' midribs).

    [[prop]]
    name = "plush"
    card = "library:carrot_plush"
    slots = { cheek = "love" }

Materials: minky on the body (a soft sheen and a fine pile), felt on the leaves and the cheeks (rougher, a duller sheen),
satin stitches for the face, its highlights and the midribs (fine threads across). The numbers, meshes and card:
carrot_plush_layout.py."""
from ....core import palette as PAL
from .. import shell as SH
from ..sets import nightkit as NK
from . import carrot_plush_layout as LAY
from . import register
from .cafe_kit import Kit

FABRIC = {   # role -> (roughness, sheen weight, sheen roughness, pile bump strength, specular)
    "body": (0.78, 0.8, 0.3, 0.35, 0.25),
    "leaf": (0.92, 0.35, 0.55, 0.5, 0.15),
    "cheek": (0.95, 0.2, 0.6, 0.4, 0.1),
}
# role -> (roughness, sheen weight, shade: the thread's base colour is the role's times this; a dark thread's satin sheen
# lifts it, so the face's is darkened to read as the role's colour)
THREAD = {"face": (0.5, 0.15, 0.5), "shine": (0.4, 0.4, 1.0), "stitch": (0.45, 0.4, 1.0)}


def _fabric(name, role, colour):
    rough, sheen, sheen_rough, bump, spec = FABRIC[role]
    m, nb = NK.new_material(f"{name}_{role}")
    pile = nb.noise(nb.texcoord("Object"), scale=900.0, detail=4.0, rough=0.7)
    tone = nb.remap(nb.noise(nb.texcoord("Object"), scale=35.0, detail=2.0), 0.3, 0.7, 0.9, 1.06)
    col = nb.mixc(1.0, colour, nb.comb(tone, tone, tone), blend="MULTIPLY")
    b = nb.principled(col, rough=rough, spec=spec,
                      normal=nb.node("ShaderNodeBump", {"Strength": bump, "Distance": 0.0004, "Height": pile}))
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Sheen Roughness"].default_value = sheen_rough
    b.inputs["Sheen Tint"].default_value = tuple(0.5 + 0.5 * c for c in colour[:3]) + (1.0,)     # the pile's own hue
    nb.output(b)
    m.diffuse_color = colour
    return m


def _thread(name, role, colour):
    """Satin stitches: fine threads (bands across the object's x), a soft sheen along them."""
    rough, sheen, shade = THREAD[role]
    m, nb = NK.new_material(f"{name}_{role}")
    bands = nb.node("ShaderNodeTexWave", {"Vector": nb.texcoord("Object"), "Scale": 260.0, "Distortion": 1.5,
                                          "Detail": 1.0}, wave_type="BANDS", bands_direction="X").outputs["Fac"]
    b = nb.principled(tuple(shade * c for c in colour[:3]) + (1.0,), rough=rough, spec=0.35,
                      normal=nb.node("ShaderNodeBump", {"Strength": 0.45, "Distance": 0.0003, "Height": bands}))
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Sheen Roughness"].default_value = 0.3
    nb.output(b)
    m.diffuse_color = colour
    return m


@register("carrot_plush")
def carrot_plush(name, coll, root, slots=None):
    roles = LAY.options(slots)
    K = Kit(name, coll, root, slots)
    mats = {}
    for role in LAY.MATS:
        colour = K.role(role, roles)
        mats[role] = _fabric(name, role, colour) if role in FABRIC else _thread(name, role, colour)
    mat_of = lambda i: mats[LAY.MATS[i]]                                             # noqa: E731
    SH.mesh_object(f"{name}_body", LAY.body_mesh(), coll, root, mat_of)
    SH.mesh_object(f"{name}_leaves", LAY.leaves_mesh(), coll, root, mat_of)
    SH.exempt(SH.mesh_object(f"{name}_face", LAY.face_mesh(), coll, root, mat_of), "embroidered face: thread on cloth")
    SH.exempt(SH.mesh_object(f"{name}_cheeks", LAY.cheeks_mesh(), coll, root, mat_of), "felt cheeks: thin patches")
    SH.exempt(SH.mesh_object(f"{name}_stitches", LAY.stitches_mesh(), coll, root, mat_of), "midrib stitches: thread")
    return LAY.card(name, {role: PAL.resolve(v, K.colors.p) for role, v in roles.items()}, K.size())
