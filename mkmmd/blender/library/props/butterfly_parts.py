"""The parts both butterfly props share (a helper module: it registers no builder): a style's pattern atlas as a packed image,
the materials of a style in a colourway (wing, body, eye), and the part meshes at unit span with their materials and the
wings' `mk_dorsal` attribute (mkmmd/core/butterfly.py: the frame, the outlines and the meshes; butterfly_pattern.py: the
patterns).

    meshes = part_meshes(prefix, style, slots, way=0, sheen=None, tile=512)   {"body", "wing_L", "wing_R": bpy Mesh}

The wing material reads the atlas at the mesh's uv (the upper side's tiles; the underside's half a texture lower where
`mk_dorsal` is under 0.5), lays the masks over the ground in the order sheen, accent, ink, spot with the colours of the
roles, and gives the sheen channel a Principled sheen in the `sheen` colour (bright at a glancing view, as the scales of a
morpho are) scaled by `sheen`, with a little light let through (a wing glows against the light)."""
import os

import bpy
import numpy as np

from ....core import butterfly as BF
from ....core import butterfly_pattern as BP
from .. import shell as SH
from .cafe_kit import L, N

TILE = 512


def cache_folder():
    return os.path.join(os.path.expanduser(os.environ.get("MK_CACHE", "~/.cache/mk")), "butterfly")


def atlas_image(style, tile=TILE):
    """The style's pattern atlas as an image (non-colour data, the sheen mask in alpha, packed into the .blend), drawn
    once per style, tile and version of the maths and shared by every butterfly of that style."""
    key = BP.masks_key(style, tile)
    name = f"mk_butterfly_{style}_{tile}"
    img = bpy.data.images.get(name)
    if img is not None and img.get("mk_key") == key:
        return img
    M = BP.cached_masks(style, tile, cache_folder())
    if img is None or tuple(img.size) != (2 * tile, 2 * tile):
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, 2 * tile, 2 * tile, alpha=True, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(np.ascontiguousarray(M[::-1]).astype(np.float32).ravel())     # Blender's rows run up
    img.pack()
    img["mk_key"] = key
    return img


def _mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1400, 0)
    return m, nt, out


def _math(nt, op, a, b=None, loc=(0, 0), c=None):
    """A Math node `op` on a, b (and c for the three-operand ones), each a socket or a number."""
    n = nt.nodes.new("ShaderNodeMath")
    n.operation, n.location = op, loc
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _mix(nt, fac, a, b, loc):
    """Linear mix of two colours (sockets or tuples) by `fac` (a socket or a number)."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type, n.blend_type, n.location = "RGBA", "MIX", loc
    for sock, v in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, (int, float)):
            sock.default_value = float(v)
        elif isinstance(v, tuple):
            sock.default_value = (*v[:3], 1.0)
        else:
            nt.links.new(v, sock)
    return n.outputs[2]


def wing_material(name, style, col, sheen, image, glow=0.0):
    """The wing's material: the atlas' masks over the ground in the roles' colours (`col`: role -> linear RGB), glowing
    `glow` of its own colour (an undersides' ambient light: a wing in a lit scene is never black)."""
    m, nt, out = _mat(name)
    uv = N(nt, "ShaderNodeUVMap", (-1400, 200))
    uv.uv_map = "UVMap"
    sep = N(nt, "ShaderNodeSeparateXYZ", (-1200, 200))
    L(nt, uv.outputs[0], sep.inputs[0])
    att = N(nt, "ShaderNodeAttribute", (-1400, -50))
    att.attribute_type, att.attribute_name = "GEOMETRY", "mk_dorsal"
    side = _math(nt, "GREATER_THAN", att.outputs["Fac"], 0.5, (-1200, -50))          # 1 the upper side, 0 the underside
    v = _math(nt, "ADD", _math(nt, "MULTIPLY", side, 0.5, (-1050, -50)), sep.outputs[1], (-900, 50))
    v = _math(nt, "SUBTRACT", v, 0.5, (-760, 50))
    comb = N(nt, "ShaderNodeCombineXYZ", (-620, 150))
    L(nt, sep.outputs[0], comb.inputs[0])
    L(nt, v, comb.inputs[1])
    tex = N(nt, "ShaderNodeTexImage", (-460, 150))
    tex.image, tex.interpolation, tex.extension = image, "Linear", "EXTEND"
    L(nt, comb.outputs[0], tex.inputs[0])
    rgb = N(nt, "ShaderNodeSeparateColor", (-180, 250))
    L(nt, tex.outputs["Color"], rgb.inputs[0])
    ink, spot, accent = rgb.outputs[0], rgb.outputs[1], rgb.outputs[2]
    sheen_mask = tex.outputs["Alpha"]
    ground = _mix(nt, side, col["under"], col["ground"], (0, 450))
    # the sheen: where the mask lies and nothing is painted over it, scaled by `sheen`
    free = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, ink, (360, 0)),
                 _math(nt, "SUBTRACT", 1.0, spot, (360, -80)), (520, -20))
    free = _math(nt, "MULTIPLY", free, _math(nt, "SUBTRACT", 1.0, accent, (520, -120)), (680, -40))
    glint = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", sheen_mask, free, (820, -40)), min(sheen, 1.0), (960, -40))
    lw = N(nt, "ShaderNodeLayerWeight", (-180, 600))
    lw.inputs["Blend"].default_value = 0.35
    turn = _math(nt, "MULTIPLY", sheen_mask, min(sheen, 1.0), (0, 600))      # face on a hint, at a glance the sheen
    turn = _math(nt, "MULTIPLY", turn, _math(nt, "MULTIPLY_ADD", lw.outputs["Facing"], 0.6, (0, 700), c=0.15),
                 (150, 600))
    c = _mix(nt, turn, ground, col["sheen"], (180, 450))
    c = _mix(nt, accent, c, col["accent"], (360, 450))
    c = _mix(nt, ink, c, col["ink"], (540, 450))
    c = _mix(nt, spot, c, col["spot"], (720, 450))
    b = N(nt, "ShaderNodeBsdfPrincipled", (1000, 350))
    L(nt, c, b.inputs["Base Color"])
    L(nt, _math(nt, "MULTIPLY_ADD", glint, -0.3, (1000, -40), c=0.66), b.inputs["Roughness"])     # scales are matte
    L(nt, _math(nt, "MULTIPLY_ADD", glint, 0.45, (1000, -120), c=0.12), b.inputs["Specular IOR Level"])
    L(nt, _math(nt, "MULTIPLY_ADD", glint, 0.8, (1100, -120), c=0.06), b.inputs["Sheen Weight"])
    b.inputs["Sheen Roughness"].default_value = 0.35
    L(nt, _mix(nt, glint, c, col["sheen"], (860, 120)), b.inputs["Sheen Tint"])
    L(nt, _mix(nt, glint, (1.0, 1.0, 1.0), col["sheen"], (860, 0)), b.inputs["Specular Tint"])
    L(nt, c, b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = float(glow)
    tr = N(nt, "ShaderNodeBsdfTranslucent", (1000, -260))
    L(nt, c, tr.inputs["Color"])
    mix = N(nt, "ShaderNodeMixShader", (1240, 100))
    mix.inputs[0].default_value = 0.12
    L(nt, b.outputs[0], mix.inputs[1])
    L(nt, tr.outputs[0], mix.inputs[2])
    L(nt, mix.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*col["ground"], 1.0)
    return m


def body_material(name, style, col, glow=0.0):
    """The body: velvety, its fur catching the light at a glancing view; a monarch's dotted with white."""
    m, nt, out = _mat(name)
    b = N(nt, "ShaderNodeBsdfPrincipled", (900, 0))
    base = col["body"]
    if style == "monarch":
        tc = N(nt, "ShaderNodeTexCoord", (-600, 0))
        vor = N(nt, "ShaderNodeTexVoronoi", (-400, 0))
        vor.feature = "F1"
        vor.inputs["Scale"].default_value = 45.0
        L(nt, tc.outputs["Object"], vor.inputs["Vector"])
        dots = _math(nt, "LESS_THAN", vor.outputs["Distance"], 0.15, (-200, 0))
        sepz = N(nt, "ShaderNodeSeparateXYZ", (-400, -250))
        L(nt, tc.outputs["Object"], sepz.inputs[0])
        up = _math(nt, "GREATER_THAN", sepz.outputs[2], -0.006, (-200, -200))         # on the upper side only
        L(nt, _mix(nt, _math(nt, "MULTIPLY", dots, up, (0, -100)), base, col["spot"], (200, 0)), b.inputs["Base Color"])
    else:
        b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Roughness"].default_value = 0.6
    b.inputs["Sheen Weight"].default_value = 0.8
    b.inputs["Sheen Roughness"].default_value = 0.5
    b.inputs["Sheen Tint"].default_value = (*[0.5 * c + 0.5 * s for c, s in zip(base, col["spot"])], 1.0)
    b.inputs["Emission Color"].default_value = (*base, 1.0)
    b.inputs["Emission Strength"].default_value = 0.5 * float(glow)
    L(nt, b.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*base, 1.0)
    return m


def eye_material(name, col):
    m, nt, out = _mat(name)
    b = N(nt, "ShaderNodeBsdfPrincipled", (900, 0))
    b.inputs["Base Color"].default_value = (*[0.45 * c for c in col["eye"]], 1.0)
    b.inputs["Roughness"].default_value = 0.3
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.04
    L(nt, b.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*col["eye"], 1.0)
    return m


def materials(prefix, style, slots, way=0, sheen=None, tile=TILE):
    """{"wing", "body", "eye": Material} of a style in colourway `way` (`slots`: the palette, role overrides and `glow`),
    named `<prefix>_wing` etc."""
    roles = BF.roles(style, slots, way)
    col = {k: BF.colour(v, slots) for k, v in roles.items()}
    sheen = BF.SHEEN[style] if sheen is None else float(sheen)
    glow = BF.glow_of(slots)
    return {"wing": wing_material(f"{prefix}_wing", style, col, sheen, atlas_image(style, tile), glow),
            "body": body_material(f"{prefix}_body", style, col, glow),
            "eye": eye_material(f"{prefix}_eye", col)}


def part_meshes(prefix, style, slots, way=0, sheen=None, tile=TILE):
    """{"body", "wing_L", "wing_R": bpy Mesh} of a style at unit span (mkmmd.core.butterfly.parts), named
    `<prefix>_body` etc., their materials in the meshes' own slots and the wings' `mk_dorsal` attribute set."""
    mats = materials(prefix, style, slots, way, sheen, tile)
    P = BF.parts(style)
    out = {}
    for key in ("body", "wing_L", "wing_R"):
        name = f"{prefix}_{key}"
        old = bpy.data.meshes.get(name)
        if old is not None and old.users == 0:
            bpy.data.meshes.remove(old)
        if key == "body":
            me, _ = SH.mesh_data(name, P["body"], lambda r: (mats["body"], mats["eye"])[r])
        else:
            me, _ = SH.mesh_data(name, P[key], lambda _r: mats["wing"])
            me.attributes.new("mk_dorsal", "FLOAT", "POINT").data.foreach_set(
                "value", np.asarray(P["attrs"][key]["mk_dorsal"], np.float32))
        out[key] = me
    return out
