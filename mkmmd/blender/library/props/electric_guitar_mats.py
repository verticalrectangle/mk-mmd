"""Materials of the electric guitar (Blender side): one Principled material per colour role of `electric_guitar_layout.ROLES`,
coloured from the palette (`LAY.Pal`), plus the strap's band `<name>_strap`.

Nothing renders black: every surface glows `FLOOR` of its colour, so a metal that reflects a dark sky is still its colour (and
the darkest colour anywhere is the palette's `base`). Finishes: a glossy clear coat on the body, satin maple and rosewood with a
faint grain along the neck, bright metal for the hardware, soft plastic for the guard, knobs and pick, a woven band for the strap.
"""
import bpy

from . import electric_guitar_layout as LAY
from .convertible_palette import rgba

FLOOR = 0.15                                  # share of its colour a surface always glows

# role -> (roughness, metallic, extra Principled inputs)
FINISH = {
    "body": (0.30, 0.0, {"Coat Weight": 1.0, "Coat Roughness": 0.035}),
    "pickguard": (0.38, 0.0, {"Coat Weight": 0.25, "Coat Roughness": 0.2}),
    "guard_core": (0.5, 0.0, {}),
    "neck": (0.42, 0.0, {"Coat Weight": 0.35, "Coat Roughness": 0.3}),
    "fretboard": (0.50, 0.0, {"Coat Weight": 0.15, "Coat Roughness": 0.35}),
    "hardware": (0.20, 1.0, {}),
    "frets": (0.26, 1.0, {}),
    "strings": (0.30, 0.9, {}),
    "knobs": (0.32, 0.0, {"Coat Weight": 0.4, "Coat Roughness": 0.15}),
    "cable": (0.45, 0.0, {}),
    "pick": (0.30, 0.0, {"Coat Weight": 0.5, "Coat Roughness": 0.1}),
    "inlay": (0.34, 0.0, {}),
    "dark": (0.6, 0.0, {}),
}
GRAINED = ("neck", "fretboard")
METALS = ("hardware", "frets", "strings")


def _metal_glow(nt, bsdf, lo=0.12, hi=0.80):
    """A matcap-like floor for polished metal: how much it glows depends on the angle to the viewer, brightest where it faces
    her and dim toward the rim, so curved metal still shows its form where it reflects nothing (a dark world, a flat sky)."""
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 1.6
    mad = nt.nodes.new("ShaderNodeMath")
    mad.operation = "MULTIPLY_ADD"
    mad.inputs[1].default_value = hi - lo
    mad.inputs[2].default_value = lo
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs[0], pw.inputs[0])
    nt.links.new(pw.outputs[0], mad.inputs[0])
    nt.links.new(mad.outputs[0], bsdf.inputs["Emission Strength"])


def _principled(nt, colour, rough, metal, extra):
    b = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = rgba(colour)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Emission Color"].default_value = rgba(colour)
    b.inputs["Emission Strength"].default_value = FLOOR
    for k, v in extra.items():
        b.inputs[k].default_value = v
    return b


def _grain(nt, bsdf, colour):
    """A faint streaky grain along the neck (object z): the base colour is mixed with a darker copy by a stretched noise."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (120.0, 90.0, 5.0)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.0
    nz.inputs["Detail"].default_value = 4.0
    nz.inputs["Roughness"].default_value = 0.6
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = rgba(colour)
    mix.inputs[7].default_value = rgba(tuple(0.72 * c for c in colour))
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], mix.inputs[0])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(mix.outputs[2], bsdf.inputs["Emission Color"])


class Materials:
    """The guitar's Blender materials, made on first use. `of(i)` is the material of mesh role index `i` (`LAY.MATS`)."""

    def __init__(self, name, pal):
        self.name, self.pal = name, pal
        self.by_role = {}
        self.names = {i: r for r, i in LAY.MATS.items()}
        self.strap = self._strap()

    def role(self, role):
        if role not in self.by_role:
            colour_role = LAY.ROLE_OF_MAT.get(role, role)
            colour = self.pal.lin(colour_role)
            rough, metal, extra = FINISH[role]
            m = bpy.data.materials.get(f"{self.name}_{role}") or bpy.data.materials.new(f"{self.name}_{role}")
            m.use_nodes = True
            b = _principled(m.node_tree, colour, rough, metal, extra)
            if role in GRAINED:
                _grain(m.node_tree, b, colour)
            if role in METALS:
                _metal_glow(m.node_tree, b)
            self.by_role[role] = m
        return self.by_role[role]

    def of(self, index):
        return self.role(self.names[int(index)])

    def _strap(self):
        """`<name>_strap`: a woven band, a matte colour with a fine rib pattern (no texture files)."""
        colour = self.pal.lin("strap")
        m = bpy.data.materials.get(f"{self.name}_strap") or bpy.data.materials.new(f"{self.name}_strap")
        m.use_nodes = True
        nt = m.node_tree
        b = _principled(nt, colour, 0.85, 0.0, {})
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (260.0, 260.0, 260.0)
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.wave_type, wv.bands_direction = "BANDS", "DIAGONAL"
        wv.inputs["Scale"].default_value = 1.0
        wv.inputs["Distortion"].default_value = 0.8
        wv.inputs["Detail"].default_value = 1.0
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs[6].default_value = rgba(colour)
        mix.inputs[7].default_value = rgba(tuple(0.7 * c for c in colour))
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.35
        bump.inputs["Distance"].default_value = 0.0004
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], wv.inputs["Vector"])
        nt.links.new(wv.outputs["Fac"], mix.inputs[0])
        nt.links.new(mix.outputs[2], b.inputs["Base Color"])
        nt.links.new(mix.outputs[2], b.inputs["Emission Color"])
        nt.links.new(wv.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        m.use_fake_user = True                          # the pose stage builds the strap later: keep the material in the file
        return m
