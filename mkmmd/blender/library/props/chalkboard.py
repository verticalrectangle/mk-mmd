"""chalkboard: a school chalkboard on an A-frame easel, built procedurally (no downloaded model, no textures): rounded
wooden legs splayed like an A, the back leg hinged at the top, a crossbar, the ledge with its chalk tray holding a white
and a pink chalk stick, a stub and a felt eraser, and the green-black slate in a rounded wooden frame leaning back on
the legs, faint chalk dust on it. Option `drawing` = "noughts" draws a game of noughts and crosses on the slate in chalk.
Numbers, meshes, the drawing and the card: chalkboard_layout.py.

    [[prop]]
    name = "board"
    card = "library:chalkboard"
    slots = { drawing = "noughts", slate = "#2c3a33" }   # the option; recolour by role

    [[text]]                                              # write on the slate
    name = "sum"
    on = "board:board"
    text = "2 + 2"

Objects `<name>_<part>`: board (the slate and its frame), easel (legs, hinge, crossbar, ledge), chalk (sticks and
eraser) and, with `drawing = "noughts"`, drawing (the chalk strokes: a graphic layer, exempt from `form`). Static: no
live properties, no drivers.

Materials: one per colour role (`slots`: a palette slot, a hex, a blend or another role), each glowing a FLOOR of its
colour so nothing renders black: the slate matte with soft clouds of chalk dust and wiped streaks in the chalk's colour,
the wood satin with a faint grain, chalk dry and powdery, the felt soft; the strokes are chalk whose dry, broken edges
come from a noise in their alpha (dithered)."""
from .. import shell as SH
from ..sets import nightkit as NK
from . import chalkboard_layout as LAY
from . import register
from .convertible_palette import rgba
from .electric_guitar_mats import FLOOR, _principled

FINISH = {                      # role -> (roughness, metallic, extra Principled inputs)
    "slate": (0.82, 0.0, {"Specular IOR Level": 0.3}),
    "wood": (0.45, 0.0, {"Coat Weight": 0.35, "Coat Roughness": 0.25}),
    "chalk": (0.95, 0.0, {"Specular IOR Level": 0.1}),
    "chalk_pink": (0.95, 0.0, {"Specular IOR Level": 0.1}),
    "felt": (1.0, 0.0, {"Sheen Weight": 0.7, "Sheen Roughness": 0.5, "Specular IOR Level": 0.1}),
    "eraser": (0.35, 0.0, {"Coat Weight": 0.5, "Coat Roughness": 0.2}),
}
DUST = 0.10                     # how far the dust clouds go toward the chalk's colour


def _glow(nb, b, col):
    nb.set(b.inputs["Base Color"], col)
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, (FLOOR, FLOOR, FLOOR, 1.0), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = 1.0


def _slate(nb, b, colour, chalk):
    """The slate: dust clouds (a soft large noise) and wiped streaks (a noise stretched across the board) toward the
    chalk's colour, and a faint mottle."""
    co = nb.texcoord("Object")
    clouds = nb.smooth(0.56, 0.80, nb.noise(co, scale=3.2, detail=4.0, rough=0.55))
    wiped = nb.node("ShaderNodeMapping", {"Vector": co, "Scale": (1.2, 14.0, 14.0)}).outputs[0]
    streaks = nb.smooth(0.55, 0.80, nb.noise(wiped, scale=2.0, detail=3.0, rough=0.6))
    dust = nb.math("MINIMUM", nb.math("ADD", nb.math("MULTIPLY", clouds, DUST), nb.math("MULTIPLY", streaks, 0.6 * DUST)),
                   1.0)
    mottle = nb.math("MULTIPLY_ADD", nb.noise(co, scale=40.0, detail=2.0, rough=0.5), 0.12, 0.94)
    base = nb.mixc(1.0, rgba(colour), nb.comb(mottle, mottle, mottle), blend="MULTIPLY")
    _glow(nb, b, nb.mixc(dust, base, rgba(chalk)))


def _grain(nb, b, colour):
    """A faint wood grain along the parts (a noise stretched along Z and, more weakly, X)."""
    p = nb.node("ShaderNodeMapping", {"Vector": nb.texcoord("Object"), "Scale": (8.0, 60.0, 2.0)}).outputs[0]
    g = nb.smooth(0.35, 0.75, nb.noise(p, scale=1.0, detail=5.0, rough=0.62))
    _glow(nb, b, nb.mixc(g, rgba(colour), rgba(tuple(0.84 * c for c in colour))))


def _powder(nb, b, colour):
    """Dry chalk: a fine speckle of the colour and its bump."""
    co = nb.texcoord("Object")
    fine = nb.noise(co, scale=700.0, detail=2.0, rough=0.6)
    tone = nb.math("MULTIPLY_ADD", fine, 0.16, 0.90)
    _glow(nb, b, nb.mixc(1.0, rgba(colour), nb.comb(tone, tone, tone), blend="MULTIPLY"))
    nb.set(b.inputs["Normal"], nb.node("ShaderNodeBump", {"Strength": 0.3, "Distance": 0.0006,
                                                          "Height": fine}).outputs["Normal"])


def _line(name, colour):
    """The chalk strokes: the chalk's colour, their alpha broken by a fine noise (dry chalk dragged over the slate)."""
    m, nb = NK.new_material(f"{name}_chalk_line")
    m.surface_render_method = "DITHERED"
    grit = nb.noise(nb.texcoord("Object"), scale=420.0, detail=3.0, rough=0.7)
    alpha = nb.math("MULTIPLY", nb.smooth(0.30, 0.55, grit), 0.92)
    b = _principled(nb.nt, colour, 0.95, 0.0, {"Specular IOR Level": 0.1})
    nb.set(b.inputs["Alpha"], alpha)
    b.inputs["Emission Strength"].default_value = 0.35
    nb.output(b)
    return m


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
            if role == "slate":
                _slate(nb, b, colour, self.pal.lin("chalk"))
            elif role in ("wood", "eraser"):
                _grain(nb, b, colour)
            elif role in ("chalk", "chalk_pink"):
                _powder(nb, b, colour)
            self.made[role] = m
        return self.made[role]


@register("chalkboard")
def chalkboard(name, coll, root, slots=None):
    slots = dict(slots or {})
    opts = LAY.options(slots)
    pal = LAY.Pal(slots)
    pal.checked()
    mats = _Materials(name, pal)
    parts = LAY.parts(opts["drawing"])
    for key, part in parts.items():
        if key not in LAY.GRAPHIC:
            SH.mesh_object(f"{name}_{key}", part.mesh, coll, root, mats.of, loc=part.origin, rot=part.rot)
    if "drawing" in parts:
        line = _line(name, pal.lin("chalk"))
        part = parts["drawing"]
        SH.exempt(SH.mesh_object(f"{name}_drawing", part.mesh, coll, root, lambda _i: line, loc=part.origin,
                                 rot=part.rot), "chalk strokes: a graphic layer")
    return LAY.card(name, pal.roles(), opts)
