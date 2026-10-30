"""sandbox_piano: a children's wooden sandbox with a toy grand piano and its bench in it, built procedurally (no downloaded
model, no textures): the cartoon homage to a piano kept in a sandbox. Chunky and rounded in candy colours: stacked boards
with rounded edges and triangular corner seats, bumpy sand, a grand on its plan (spine, bentside, tail) with the lid
propped open on its stick over a gold plate and strings, a music desk, two octaves of chunky keys between cheek blocks,
three turned legs and the pedal lyre going into the sand, and a padded bench. Numbers, roles and the card:
sandbox_piano_layout.py; meshes: sandbox_piano_parts.py.

    [[prop]]
    name = "sandbox"
    card = "library:sandbox_piano"
    slots = { piano = "iris", bench = "love" }    # recolour by role

    [pose.rin]
    sit = "sandbox:bench"                          # on the bench, facing the keys (-Y); the feet go to its `feet`

Objects `<name>_<part>`: boards, seats, sand, case (the rim and soundboard, cheeks, key bed, key slip), lid (its origin
on the hinge line, turned LID_DEG about Y), inside (plate and strings), keys, trim (legs, lyre, pedals, stick, hinges,
music desk), bench. Static: no live properties, no drivers.

Materials: one per colour role (`slots`: a palette slot, a hex, a blend or another role), each glowing a FLOOR of its
colour so nothing renders black; the woods carry a faint grain (the boards' along their length, the soundboard's along
the strings), the sand a fine grain and its bump, the lacquers a clear coat, the plate is metal with the facing-angle
glow, the cushion has a fabric sheen."""
from .. import shell as SH
from ..sets import nightkit as NK
from . import register
from . import sandbox_piano_layout as LAY
from . import sandbox_piano_parts as PT
from .convertible_palette import rgba
from .electric_guitar_mats import FLOOR, _metal_glow, _principled

FINISH = {                      # role -> (roughness, metallic, extra Principled inputs): satin toy lacquers
    "wood": (0.55, 0.0, {"Coat Weight": 0.25, "Coat Roughness": 0.3}),
    "seats": (0.38, 0.0, {"Coat Weight": 0.4, "Coat Roughness": 0.2}),
    "sand": (0.95, 0.0, {"Specular IOR Level": 0.15}),
    "piano": (0.30, 0.0, {"Coat Weight": 0.8, "Coat Roughness": 0.16}),
    "lid": (0.30, 0.0, {"Coat Weight": 0.8, "Coat Roughness": 0.16}),
    "inside": (0.50, 0.0, {"Coat Weight": 0.3, "Coat Roughness": 0.2}),
    "plate": (0.30, 1.0, {}),
    "trim": (0.34, 0.0, {"Coat Weight": 0.6, "Coat Roughness": 0.18}),
    "keys_white": (0.28, 0.0, {"Coat Weight": 0.5, "Coat Roughness": 0.16}),
    "keys_black": (0.28, 0.0, {"Coat Weight": 0.6, "Coat Roughness": 0.14}),
    "bench": (0.85, 0.0, {"Sheen Weight": 0.8, "Sheen Roughness": 0.4, "Specular IOR Level": 0.25}),
    "bench_wood": (0.30, 0.0, {"Coat Weight": 0.6, "Coat Roughness": 0.16}),
}
GRAIN = {"wood": (2.5, 2.5, 70.0), "inside": (40.0, 1.2, 40.0)}    # object-space scales: streaks along the boards / strings


def _grain(nb, b, colour, scale, depth=0.16):
    """A faint wood grain: the colour darkened along streaks of a noise stretched by `scale` (object space)."""
    p = nb.node("ShaderNodeMapping", {"Vector": nb.texcoord("Object"), "Scale": scale}).outputs[0]
    g = nb.noise(p, scale=1.0, detail=5.0, rough=0.62)
    col = nb.mixc(nb.smooth(0.35, 0.75, g), rgba(colour), rgba(tuple((1.0 - depth) * c for c in colour)))
    nb.set(b.inputs["Base Color"], col)
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, (FLOOR, FLOOR, FLOOR, 1.0), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = 1.0


def _sand(nb, b, colour):
    """Sand: fine grains of lighter and darker colour, a light grain here and there, a coarser mottle, and the bump the
    grains make."""
    co = nb.texcoord("Object")
    fine = nb.noise(co, scale=900.0, detail=2.0, rough=0.6)
    mottle = nb.noise(co, scale=6.0, detail=3.0, rough=0.5)
    grains = nb.node("ShaderNodeMapping", {"Vector": co, "Scale": (260.0, 260.0, 260.0)}).outputs[0]
    speck = nb.math("GREATER_THAN", nb.white(grains), 0.93)
    tone = nb.math("MULTIPLY_ADD", fine, 0.30, 0.84)                                   # 0.84 .. 1.14
    tone = nb.math("MULTIPLY_ADD", mottle, 0.12, nb.math("SUBTRACT", tone, 0.06))
    tone = nb.math("MULTIPLY_ADD", speck, 0.18, tone)
    col = nb.mixc(1.0, rgba(colour), nb.comb(tone, tone, tone), blend="MULTIPLY")
    nb.set(b.inputs["Base Color"], col)
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, (FLOOR, FLOOR, FLOOR, 1.0), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = 1.0
    nb.set(b.inputs["Normal"], nb.node("ShaderNodeBump", {"Strength": 0.35, "Distance": 0.002,
                                                          "Height": fine}).outputs["Normal"])


class _Materials:
    """One Blender material per colour role, made on first use: `of(i)` for mesh role index i (`LAY.MATS`)."""

    def __init__(self, name, pal):
        self.name, self.pal, self.made = name, pal, {}
        self.roles = {i: r for r, i in LAY.MATS.items()}

    def of(self, index):
        role = self.roles[int(index)]
        if role not in self.made:
            colour = self.pal.lin(role)
            rough, metal, extra = FINISH[role]
            m, nb = NK.new_material(f"{self.name}_{role}")
            b = _principled(nb.nt, colour, rough, metal, extra)
            nb.output(b)
            if role in GRAIN:
                _grain(nb, b, colour, GRAIN[role])
            elif role == "sand":
                _sand(nb, b, colour)
            elif role == "plate":
                _metal_glow(nb.nt, b)
            self.made[role] = m
        return self.made[role]


@register("sandbox_piano")
def sandbox_piano(name, coll, root, slots=None):
    pal = LAY.Pal(dict(slots or {}))
    pal.checked()
    mats = _Materials(name, pal)
    for key, part in PT.parts().items():
        SH.mesh_object(f"{name}_{key}", part.mesh, coll, root, mats.of, loc=part.origin, rot=part.rot)
    return LAY.card(name, pal.roles())
