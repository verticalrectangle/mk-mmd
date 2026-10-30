"""Materials of the record treadmill (Blender side): one material per mesh role of `record_treadmill_layout.MATS`,
coloured from the palette (`LAY.Pal`), node trees built with the night kit's node builder (`sets/nightkit.NB`), on the
turntable's pattern (turntable_mats.py). Every surface glows a FLOOR of its colour (nothing renders black) and bright metal
gets the guitar's facing-angle glow, so the deck keeps its form in a flat world it has nothing to reflect.

  vinyl      the record: glossy dark, the recorded band in grooves of slightly different shine (a noise along the radius),
             the smooth bands between tracks, and the groove sheen: concentric grooves reflect a light along the radius
             that points at the half vector (a bright bar through the centre that stays put while the record turns:
             EEVEE has no anisotropy, so it is drawn from the view and a key light, both moved into the record's frame),
             and a soft Fresnel lift where glossy vinyl would mirror the room
  label      matte paper printed with `LAY.label_image` (a packed image on the label's uv)
  rim        the platter's dark bevel with four rows of bright strobe dots (DOT_ROWS), drawn in the platter's frame, so
             they turn with it
  deck       brushed aluminium (the roughness streaked along X); base a candy lacquer; plate bright metal; feet rubber
"""
import math

import bpy
import numpy as np

from ..sets import nightkit as NK
from . import record_treadmill_layout as LAY
from .convertible_palette import rgba
from .electric_guitar_mats import FLOOR, _metal_glow, _principled
from .kaleidoscope_mats import packed_image

FINISH = {                      # role -> (roughness, metallic, extra Principled inputs)
    "plinth": (0.30, 0.0, {"Coat Weight": 0.6, "Coat Roughness": 0.10}),
    "deck": (0.34, 1.0, {}),
    "feet": (0.80, 0.0, {"Specular IOR Level": 0.25}),
    "plate": (0.16, 1.0, {}),
    "rim": (0.45, 0.0, {}),
    "vinyl": (0.12, 0.0, {"Specular IOR Level": 0.6, "Coat Weight": 0.25, "Coat Roughness": 0.06}),
    "label": (0.62, 0.0, {"Specular IOR Level": 0.3}),
}
METALS = ("deck", "plate")
KEY_LIGHT = (-0.35, -0.55, 0.76)    # the world direction toward the light the groove sheen is drawn for (front left, above)
SHEEN = 0.55                    # the groove sheen's strength (share of `paper`)
GLOSS = 0.22                    # the record's soft reflection of the room at grazing angles (share of `paper`)
BANDS = 140.0                   # the grooves' shine varies along the radius on this scale (1 / m)
LABEL_PX = 1024                 # the print's pixels across the label


def _radius(nb):
    """(r from the object's z axis, x, y, z) of the shading point in the object's frame."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    return nb.vlen(nb.comb(x, y, 0.0)), x, y, z


def _floor_glow(nb, b, col):
    """Emission = FLOOR x col (a socket), as `_principled` sets for a plain colour."""
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, (FLOOR, FLOOR, FLOOR, 1.0), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = 1.0


def _band(nb, r, r0, r1, e):
    """1 between radii r0 and r1, 0 outside, soft over e."""
    return nb.math("MULTIPLY", nb.smooth(r0 - e, r0 + e, r), nb.math("SUBTRACT", 1.0, nb.smooth(r1 - e, r1 + e, r)))


def _vinyl(nb, b, colour, paper):
    """Grooves, the gaps between tracks and the groove sheen on the record (see the module docstring)."""
    r, x, y, _ = _radius(nb)
    grooved = _band(nb, r, LAY.GROOVE_IN, LAY.GROOVE_OUT, 0.0015)
    for g in LAY.TRACK_GAPS:                                            # smooth, unrecorded bands between the tracks
        grooved = nb.math("MULTIPLY", grooved, nb.math("SUBTRACT", 1.0, _band(nb, r, g - 0.0025, g + 0.0025, 0.0008)))
    bands = nb.node("ShaderNodeTexNoise", {"W": nb.math("MULTIPLY", r, BANDS), "Scale": 1.0, "Detail": 3.0,
                                           "Roughness": 0.55}, noise_dimensions="1D").outputs["Fac"]
    tone = nb.math("MULTIPLY_ADD", bands, 0.55, 0.72)                                  # 0.72 .. 1.27 of the colour
    shade = nb.math("ADD", 1.0, nb.math("MULTIPLY", grooved, nb.math("SUBTRACT", tone, 1.0)))
    nb.set(b.inputs["Base Color"], nb.mixc(1.0, rgba(colour), nb.comb(shade, shade, shade), blend="MULTIPLY"))
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", grooved, 0.16, 0.10))
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


def _dots(nb, b, dark, bright):
    """Four rows of strobe dots on the platter's bevel (platter frame: z over the deck, the bevel BEVEL[0] .. BEVEL[1])."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    a = nb.math("ARCTAN2", y, x)
    (r0, z0), (r1, z1) = (LAY.PLATTER_R - 0.006, LAY.BEVEL[0]), (LAY.TOP_R + 0.006, LAY.BEVEL[1])
    along = math.hypot(r1 - r0, z1 - z0) / (z1 - z0)                                # slope length per unit of height
    rows = None
    for zk, nk in LAY.DOT_ROWS:
        rk = r0 + (r1 - r0) * (zk - z0) / (z1 - z0)
        pitch = 2.0 * math.pi * rk / nk                                                # arc between dots at the row
        u = nb.math("SUBTRACT", nb.math("FRACT", nb.math("MULTIPLY_ADD", a, nk / (2.0 * math.pi), 0.5)), 0.5)
        dv = nb.math("MULTIPLY", nb.math("SUBTRACT", z, zk), along)                    # along the slope
        d = nb.vlen(nb.comb(nb.math("MULTIPLY", u, pitch), dv, 0.0))
        dot = nb.math("SUBTRACT", 1.0, nb.smooth(LAY.DOT_R - 0.0006, LAY.DOT_R + 0.0006, d))
        rows = dot if rows is None else nb.math("MAXIMUM", rows, dot)
    col = nb.mixc(rows, rgba(dark), rgba(bright))
    nb.set(b.inputs["Base Color"], col)
    nb.set(b.inputs["Metallic"], nb.math("MULTIPLY", rows, 0.9))
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", rows, -0.2, 0.45))
    _floor_glow(nb, b, col)
    nb.set(b.inputs["Normal"], nb.node("ShaderNodeBump", {"Strength": 0.6, "Distance": 0.002,
                                                          "Height": rows}).outputs["Normal"])


def _brushed(nb, b, rough):
    """Brushed aluminium: the roughness streaked along X."""
    p = nb.node("ShaderNodeMapping", {"Vector": nb.texcoord("Object"), "Scale": (2.0, 300.0, 300.0)}).outputs[0]
    nb.set(b.inputs["Roughness"], nb.math("MULTIPLY_ADD", nb.noise(p, scale=1.0, detail=4.0, rough=0.6), 0.14,
                                          rough - 0.07))


def _label(nb, b, image):
    """The paper label: the packed print on the label's uv, matte, glowing its FLOOR."""
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", {"Vector": uv}, image=image, interpolation="Cubic", extension="EXTEND")
    col = tex.outputs["Color"]
    nb.set(b.inputs["Base Color"], col)
    _floor_glow(nb, b, col)


class Materials:
    """The treadmill's Blender materials, made on first use: `of(i)` is the material of mesh role index `i` (`LAY.MATS`)."""

    def __init__(self, name, pal):
        self.name, self.pal = name, pal
        self.by_role = {}
        self.names = {i: r for r, i in LAY.MATS.items()}

    def print_image(self):
        """The label's print in the roles' colours, packed (an image of the same name and colours is reused)."""
        cols = {k: tuple(round(float(v), 5) for v in LAY.srgb(self.pal.lin(k)))
                for k in ("label", "rays", "print", "leaf", "paper")}
        key = repr(sorted(cols.items())) + f"|{LABEL_PX}"
        name = f"{self.name}_label_print"
        img = bpy.data.images.get(name)
        if img is not None and img.get("mk_key") == key:
            return img
        rgb = LAY.label_image(cols, LABEL_PX)
        return packed_image(name, np.concatenate([rgb, np.ones_like(rgb[..., :1])], -1), key=key)

    def role(self, role):
        if role in self.by_role:
            return self.by_role[role]
        colour = self.pal.lin(role)
        rough, metal, extra = FINISH[role]
        m, nb = NK.new_material(f"{self.name}_{role}")
        b = _principled(nb.nt, colour, rough, metal, extra)
        nb.output(b)
        if role in METALS:
            _metal_glow(nb.nt, b)
            if role == "deck":
                _brushed(nb, b, rough)
        elif role == "vinyl":
            _vinyl(nb, b, colour, self.pal.lin("paper"))
        elif role == "rim":
            _dots(nb, b, colour, self.pal.lin("dots"))
        elif role == "label":
            _label(nb, b, self.print_image())
        self.by_role[role] = m
        return m

    def of(self, index):
        return self.role(self.names[int(index)])
