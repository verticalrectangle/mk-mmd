"""Materials of the electric bass (Blender side): one Principled material per colour role of `electric_bass_layout.ROLES`,
coloured from the palette (`LAY.Pal`), plus the strap's band `<name>_strap`; the finishes and node set-ups are the guitar's
(`electric_guitar_mats`): a glossy clear coat on the body, satin maple and rosewood with a faint grain along the neck, bright
metal with a facing-angle glow, soft plastic for the guard, knobs, covers and pick, a woven band for the strap. Nothing renders
black: every surface glows a share of its colour.
"""
import bpy

from . import electric_bass_layout as LAY
from . import electric_guitar_mats as EM

FINISH = {**EM.FINISH,
          "knobs": (0.36, 0.0, {"Coat Weight": 0.3, "Coat Roughness": 0.2}),      # black plastic: knobs, pickup covers
          "stripe": (0.30, 0.0, {"Coat Weight": 1.0, "Coat Roughness": 0.035})}   # painted under the body's clear coat


class Materials:
    """The bass's Blender materials, made on first use. `of(i)` is the material of mesh role index `i` (`LAY.MATS`)."""

    def __init__(self, name, pal):
        self.name, self.pal = name, pal
        self.by_role = {}
        self.names = {i: r for r, i in LAY.MATS.items()}
        self.strap = EM.Materials._strap(self)

    def role(self, role):
        if role not in self.by_role:
            colour = self.pal.lin(LAY.ROLE_OF_MAT.get(role, role))
            rough, metal, extra = FINISH[role]
            m = bpy.data.materials.get(f"{self.name}_{role}") or bpy.data.materials.new(f"{self.name}_{role}")
            m.use_nodes = True
            b = EM._principled(m.node_tree, colour, rough, metal, extra)
            if role in EM.GRAINED:
                EM._grain(m.node_tree, b, colour)
            if role in EM.METALS:
                EM._metal_glow(m.node_tree, b)
            self.by_role[role] = m
        return self.by_role[role]

    def of(self, index):
        return self.role(self.names[int(index)])
