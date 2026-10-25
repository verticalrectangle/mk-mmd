"""Materials of the turntable (Blender side): one material per mesh role of `turntable_layout.MATS`, coloured from the
palette (`LAY.Pal`), node trees built with the night kit's node builder (`sets/nightkit.NB`). Every surface glows a FLOOR of
its colour (the guitar's rule: nothing renders black) and bright metal gets the guitar's facing-angle glow, so the deck keeps
its form in a flat world that it has nothing to reflect.

  vinyl      the record: glossy dark, the recorded band in grooves of slightly different shine (a noise along the radius)
             with the groove sheen: concentric grooves reflect a light along the radius that points at the half vector
             (a bright bar through the centre that stays put while the record turns: EEVEE has no anisotropy, so it is
             drawn from the view and a key light from the front left, both in the record's own frame), and a soft
             Fresnel lift where glossy vinyl would mirror the room
  label      matte paper with two thin printed rings (the `print` role)
  mat        felt: a fibre noise and a soft sheen
  rim        the platter's dark bevel with four rows of bright strobe dots (DOT_ROWS), drawn in the platter's frame, so
             they turn with it
  plate      brushed aluminium (the roughness streaked along X)
  lamp       the strobe lamp and the target light's window, lit; led33 / led45 the speed lamps, lit by drivers on `rpm`
"""
import math

from ..sets import nightkit as NK
from . import turntable_layout as LAY
from .convertible_palette import rgba
from .electric_guitar_mats import FLOOR, _metal_glow, _principled

FINISH = {                      # role -> (roughness, metallic, extra Principled inputs)
    "plate": (0.34, 1.0, {}),
    "metal": (0.16, 1.0, {}),
    "body": (0.55, 0.0, {"Specular IOR Level": 0.35}),
    "rubber": (0.80, 0.0, {"Specular IOR Level": 0.25}),
    "dark": (0.85, 0.0, {"Specular IOR Level": 0.2}),
    "cartridge": (0.32, 0.0, {"Coat Weight": 0.4, "Coat Roughness": 0.12}),
    "mat": (1.0, 0.0, {"Sheen Weight": 1.0, "Sheen Roughness": 0.45, "Specular IOR Level": 0.15}),
    "label": (0.62, 0.0, {"Specular IOR Level": 0.3}),
    "vinyl": (0.12, 0.0, {"Specular IOR Level": 0.6, "Coat Weight": 0.25, "Coat Roughness": 0.06}),
    "rim": (0.45, 0.0, {}),
    "lamp": (0.4, 0.0, {}),
    "led33": (0.4, 0.0, {}),
    "led45": (0.4, 0.0, {}),
}
METALS = ("plate", "metal")
LAMP_GLOW = 6.0                 # emission of the strobe lamp and the target light
LED_GLOW = 5.0                  # a speed lamp when it is lit
KEY_LIGHT = (-0.35, -0.55, 0.76)    # the world direction toward the light the groove sheen is drawn for (front left, above)
SHEEN = 0.55                    # the groove sheen's strength (share of `paper`)
GLOSS = 0.22                    # the record's soft reflection of the room at grazing angles (share of `paper`)


def _radius(nb):
    """(r from the object's z axis, x, y, z) of the shading point in the object's frame."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    return nb.vlen(nb.comb(x, y, 0.0)), x, y, z


def _floor_glow(nb, b, col):
    """Emission = FLOOR x col (a socket), as `_principled` sets for a plain colour."""
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, (FLOOR, FLOOR, FLOOR, 1.0), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = 1.0


def _vinyl(nb, b, colour, paper):
    """Grooves and the groove sheen on the record (see the module docstring)."""
    r, x, y, _ = _radius(nb)
    grooved = nb.math("MULTIPLY", nb.smooth(LAY.GROOVE_IN - 0.0004, LAY.GROOVE_IN, r),
                      nb.math("SUBTRACT", 1.0, nb.smooth(LAY.GROOVE_OUT, LAY.GROOVE_OUT + 0.0004, r)))
    bands = nb.node("ShaderNodeTexNoise", {"W": nb.math("MULTIPLY", r, 520.0), "Scale": 1.0, "Detail": 3.0,
                                           "Roughness": 0.55}, noise_dimensions="1D").outputs["Fac"]
    tone = nb.math("MULTIPLY_ADD", bands, 0.55, 0.72)                                  # 0.72 .. 1.27 of the colour
    shade = nb.math("ADD", 1.0, nb.math("MULTIPLY", grooved, nb.math("SUBTRACT", tone, 1.0)))
    nb.set(b.inputs["Base Color"], nb.mixc(1.0, rgba(colour), nb.comb(shade, shade, shade), blend="MULTIPLY"))
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", grooved, 0.16, 0.10))
    # the sheen: a bar along the radius that points along the half vector of the view and the key light
    n = math.sqrt(sum(c * c for c in KEY_LIGHT))

    def to_object(v):
        return nb.node("ShaderNodeVectorTransform", {0: v}, vector_type="VECTOR", convert_from="WORLD",
                       convert_to="OBJECT").outputs[0]

    h = nb.vmath("ADD", nb.vmath("NORMALIZE", to_object(nb.geometry("Incoming"))),
                 nb.vmath("NORMALIZE", to_object(tuple(c / n for c in KEY_LIGHT))))
    hx, hy, _ = nb.sep(h)
    hxy = nb.comb(hx, hy, 0.0)
    hnx, hny, _ = nb.sep(nb.vmath("NORMALIZE", hxy))
    rnx, rny, _ = nb.sep(nb.vmath("NORMALIZE", nb.comb(x, y, 0.0)))
    c = nb.math("SUBTRACT", nb.math("MULTIPLY", rnx, hny), nb.math("MULTIPLY", rny, hnx))     # sin(radius, half vector)
    c2 = nb.math("MULTIPLY", c, c)
    bar = nb.math("EXPONENT", nb.math("MULTIPLY", c2, -1.0 / (2 * 0.06 ** 2)))
    soft = nb.math("EXPONENT", nb.math("MULTIPLY", c2, -1.0 / (2 * 0.35 ** 2)))
    lit = nb.math("MINIMUM", nb.math("MULTIPLY", nb.vlen(hxy), 2.5), 1.0)              # none when the half vector is straight up
    s = nb.math("MULTIPLY", nb.math("MULTIPLY", nb.math("MULTIPLY_ADD", bar, SHEEN, nb.math("MULTIPLY", soft, 0.10)), lit),
                nb.math("MULTIPLY_ADD", grooved, 0.85, 0.15))
    fresnel = nb.node("ShaderNodeLayerWeight", {"Blend": 0.3}).outputs["Fresnel"]      # glossy vinyl mirrors the room
    s = nb.math("MULTIPLY_ADD", fresnel, GLOSS, s)                                     # at a grazing angle
    sheen = nb.mixc(1.0, rgba(paper), nb.comb(s, s, s), blend="MULTIPLY")
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, rgba(tuple(FLOOR * v for v in colour)), sheen, blend="ADD",
                                               clamp=False))
    b.inputs["Emission Strength"].default_value = 1.0


def _label(nb, b, colour, ink):
    """The paper label: its colour with two thin printed rings, one inside the edge and one round the hole."""
    r, _, _, _ = _radius(nb)
    e = 0.00012

    def ring(r0, r1):
        return nb.math("MULTIPLY", nb.smooth(r0 - e, r0 + e, r), nb.math("SUBTRACT", 1.0, nb.smooth(r1 - e, r1 + e, r)))

    rings = nb.math("MAXIMUM", ring(LAY.LABEL_R - 0.0040, LAY.LABEL_R - 0.0033), ring(LAY.HOLE_R + 0.0030, LAY.HOLE_R + 0.0036))
    col = nb.mixc(rings, rgba(colour), rgba(ink))
    nb.set(b.inputs["Base Color"], col)
    _floor_glow(nb, b, col)


def _felt(nb, b, colour):
    """Felt: a fine fibre noise in the colour and the bump it makes."""
    fibre = nb.noise(nb.texcoord("Object"), scale=2200.0, detail=6.0, rough=0.7)
    nb.set(b.inputs["Base Color"], nb.mixc(fibre, rgba(tuple(0.86 * c for c in colour)),
                                           rgba(tuple(min(1.0, 1.12 * c) for c in colour))))
    nb.set(b.inputs["Normal"], nb.node("ShaderNodeBump", {"Strength": 0.25, "Distance": 0.0002,
                                                          "Height": fibre}).outputs["Normal"])


def _dots(nb, b, dark, bright):
    """Four rows of strobe dots on the platter's bevel (platter frame: z over the plate, the bevel about 45 degrees)."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    a = nb.math("ARCTAN2", y, x)
    rows = None
    for zk, nk in LAY.DOT_ROWS:
        pitch = 2.0 * math.pi * 0.160 / nk                                             # arc between dots at the bevel
        u = nb.math("SUBTRACT", nb.math("FRACT", nb.math("MULTIPLY_ADD", a, nk / (2.0 * math.pi), 0.5)), 0.5)
        dv = nb.math("MULTIPLY", nb.math("SUBTRACT", z, zk), math.sqrt(2.0))           # along the 45 degree slope
        d = nb.vlen(nb.comb(nb.math("MULTIPLY", u, pitch), dv, 0.0))
        dot = nb.math("SUBTRACT", 1.0, nb.smooth(LAY.DOT_R - 0.00018, LAY.DOT_R + 0.00018, d))
        rows = dot if rows is None else nb.math("MAXIMUM", rows, dot)
    col = nb.mixc(rows, rgba(dark), rgba(bright))
    nb.set(b.inputs["Base Color"], col)
    nb.set(b.inputs["Metallic"], nb.math("MULTIPLY", rows, 0.9))
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", rows, -0.2, 0.45))
    _floor_glow(nb, b, col)
    nb.set(b.inputs["Normal"], nb.node("ShaderNodeBump", {"Strength": 0.6, "Distance": 0.0004,
                                                          "Height": rows}).outputs["Normal"])


def _brushed(nb, b, rough):
    """Brushed aluminium: the roughness streaked along X."""
    p = nb.node("ShaderNodeMapping", {"Vector": nb.texcoord("Object"), "Scale": (6.0, 900.0, 900.0)}).outputs[0]
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", nb.noise(p, scale=1.0, detail=4.0, rough=0.6), 0.14,
                                          rough - 0.07))


class Materials:
    """The turntable's Blender materials, made on first use. `of(i)` is the material of mesh role index `i` (`LAY.MATS`);
    the speed lamps' drivers read the root's `rpm`."""

    def __init__(self, name, pal, root):
        self.name, self.pal, self.root = name, pal, root
        self.by_role = {}
        self.names = {i: r for r, i in LAY.MATS.items()}

    def role(self, role):
        if role in self.by_role:
            return self.by_role[role]
        colour = self.pal.lin(LAY.ROLE_OF_MAT.get(role, role))
        rough, metal, extra = FINISH[role]
        m, nb = NK.new_material(f"{self.name}_{role}")
        b = _principled(nb.nt, colour, rough, metal, extra)
        nb.output(b)
        if role in METALS:
            _metal_glow(nb.nt, b)
            if role == "plate":
                _brushed(nb, b, rough)
        elif role == "vinyl":
            _vinyl(nb, b, colour, self.pal.lin("paper"))
        elif role == "label":
            _label(nb, b, colour, self.pal.lin("print"))
        elif role == "mat":
            _felt(nb, b, colour)
        elif role == "rim":
            _dots(nb, b, colour, self.pal.lin("plate"))
        elif role == "lamp":
            b.inputs["Emission Strength"].default_value = LAMP_GLOW
        elif role in ("led33", "led45"):
            self._led(nb, b, role)
        self.by_role[role] = m
        return m

    def _led(self, nb, b, role):
        """A speed lamp: FLOOR when off, LED_GLOW when its speed is the one set (a driver on the root's `rpm`)."""
        v = nb.node("ShaderNodeValue")
        v.name = v.label = f"mk_{role}"
        nb.set(b.inputs["Emission Strength"], nb.math("MULTIPLY_ADD", v.outputs[0], LED_GLOW - FLOOR, FLOOR))
        d = nb.nt.driver_add(f'nodes["{v.name}"].outputs[0].default_value').driver
        d.type = "SCRIPTED"
        d.expression = LAY.EXPR[role]
        var = d.variables.new()
        var.name, var.type = "rpm", "SINGLE_PROP"
        var.targets[0].id_type, var.targets[0].id = "OBJECT", self.root
        var.targets[0].data_path = '["rpm"]'
        if not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"turntable: the {role} driver is not a Blender simple expression ({d.expression})")

    def of(self, index):
        return self.role(self.names[int(index)])
