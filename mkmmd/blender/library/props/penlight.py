"""The idol fan penlight (`card = "library:penlight"`): `<name>_body` (the handle with its ribbed grip, the button, the
collar and the strap ring) and `<name>_tube` (the glowing tube) under the root, standing along +Z on the end of its handle,
the button toward -Y. A hand holds it by the card's `use.grip` `hold` (a gag move's fist), or `attach` puts it in one.

    [[prop]]
    name = "light_L"
    card = "library:penlight"
    attach = "rin:wrist.L"
    slots = { colors = ["love", "foam", "gold"], color = 0 }

    [[key]]                               # the next colour on every accent, as a fan's thumb switches it
    target = "light_L"
    prop = "color"
    lane = "accent"
    values = [0, 1, 2]

The tube's material picks its light from `colors` by the root property `color` (a driven Value node, rounded and wrapped
round the list by Math nodes: one RGB node per colour, chosen by Compare and Mix) and glows by `glow`. Numbers, roles and
the card: penlight_layout.py."""
from .. import shell as SH
from . import penlight_layout as LAY
from . import register
from . import showkit as SK
from .cafe_kit import Kit, clear_drivers

LOOK = {"handle": dict(rough=0.42, coat=0.25, spec=0.5), "grip": dict(rough=0.72, spec=0.3),
        "button": dict(rough=0.3, coat=0.5, spec=0.5), "collar": dict(rough=0.12, coat=0.8, spec=0.6),
        "ring": dict(rough=0.28, metal=0.9, spec=0.5)}
FROST = (0.8, 0.8, 0.8, 1.0)          # the tube's frosted plastic, unlit (a physical colour, not a palette one)


def _math(nt, op, a, b=None, c=None, loc=(0, 0)):
    """A Math node `op` of a, b, c (numbers or sockets); its output socket."""
    n = nt.nodes.new("ShaderNodeMath")
    n.operation, n.location = op, loc
    for sock, v in zip(n.inputs, (a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            sock.default_value = float(v)
        else:
            nt.links.new(v, sock)
    return n.outputs[0]


def _light(K, name, colors, strength):
    """The tube's material: the colour `colors[round(color) mod n]` (RGB nodes chosen by Compare and Mix) glowing at
    strength * glow, brighter where the tube faces the eye and toward its base."""
    m, nt, out = K.new_mat("tube")
    pick = _math(nt, "FLOORED_MODULO", _math(nt, "ROUND", K.param(nt, "color", loc=(-1400, 300)), loc=(-1200, 300)),
                 float(len(colors)), loc=(-1000, 300))
    col = None
    for i, spec in enumerate(colors):
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.name = rgb.label = f"mk_color{i}"
        rgb.location = (-800, -200 * i)
        rgb.outputs[0].default_value = SK.rgba(spec, K.slots, f"penlight: colors[{i}]")
        if col is None:
            col = rgb.outputs[0]
            continue
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type, mix.location = "RGBA", "MIX", (-500, -200 * i)
        nt.links.new(_math(nt, "COMPARE", pick, float(i), 0.25, loc=(-700, 400 - 60 * i)), mix.inputs[0])
        nt.links.new(col, mix.inputs[6])
        nt.links.new(rgb.outputs[0], mix.inputs[7])
        col = mix.outputs[2]
    facing = nt.nodes.new("ShaderNodeLayerWeight")
    facing.location = (-800, 700)
    facing.inputs["Blend"].default_value = 0.5
    core = _math(nt, "MULTIPLY_ADD", facing.outputs["Facing"], -0.45, 1.0, loc=(-600, 700))    # 1 facing, 0.55 grazing
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-1200, 900)
    xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    xyz.location = (-1000, 900)
    nt.links.new(coord.outputs["Object"], xyz.inputs[0])
    base = nt.nodes.new("ShaderNodeMapRange")                  # 1 at the tube's foot, 0.78 at its tip
    base.location = (-800, 1100)
    nt.links.new(xyz.outputs["Z"], base.inputs["Value"])
    base.inputs["From Min"].default_value, base.inputs["From Max"].default_value = LAY.COLLAR[1], LAY.LENGTH
    base.inputs["To Min"].default_value, base.inputs["To Max"].default_value = 1.0, 0.78
    glow = K.param(nt, "glow", loc=(-800, 1300))
    level = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", core, base.outputs["Result"], loc=(-400, 800)),
                  _math(nt, "MULTIPLY", glow, strength, loc=(-600, 1300)), loc=(-200, 900))
    tint = nt.nodes.new("ShaderNodeMix")                       # frosted plastic when off, the light's colour when lit
    tint.data_type, tint.blend_type, tint.location = "RGBA", "MIX", (0, 500)
    tint.clamp_factor = True
    nt.links.new(glow, tint.inputs[0])
    tint.inputs[6].default_value = FROST
    nt.links.new(col, tint.inputs[7])
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.location = (300, 200)
    b.inputs["Roughness"].default_value = 0.3
    b.inputs["Coat Weight"].default_value = 0.5
    nt.links.new(tint.outputs[2], b.inputs["Base Color"])
    nt.links.new(col, b.inputs["Emission Color"])
    nt.links.new(level, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    m.diffuse_color = SK.rgba(colors[0], K.slots)
    return m


@register("penlight")
def penlight(name, coll, root, slots=None):
    slots = dict(slots or {})
    colors, P, strength = LAY.options(slots)
    roles = SK.roles(slots, LAY.ROLES, "penlight")
    K = Kit(name, coll, root, slots)
    clear_drivers(root)
    for k, (_default, _unit, doc) in LAY.PARAMS.items():
        K.prop(k, P[k], 0.0 if k == "glow" else None, None, doc)
    mats = {}
    for role in LAY.ROLES:
        c = SK.rgba(roles[role], slots, f"penlight: {role}")
        mats[role] = K.simple_mat(role, c, **LOOK[role])
        mats[role].diffuse_color = c
    mats["tube"] = _light(K, name, colors, strength)
    mat_of = lambda i: mats[LAY.MATS[i]]                                                       # noqa: E731
    for part, mesh in LAY.meshes().items():
        SH.mesh_object(f"{name}_{part}", mesh, coll, root, mat_of)
    return LAY.card(name, roles, colors, P)
