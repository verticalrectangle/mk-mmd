"""The golden spike (`card = "library:golden_spike"`): `<name>_spike` (the spike, one cage from the chisel point at the
root's origin up to the crowned head) and `<name>_beads` (the two raised beads bounding the engraved band) under the root,
standing along +Z. Option `length` (slots); `[[prop]] scale` scales it. Static: no live properties. Numbers, roles and the
card: golden_spike_layout.py.

One material, `<name>_gold`: polished gold (the role `metal`, the palette's gold by default) with fine scratches in its
roughness, and a studio's reflection drawn in its emission (the reflected ray's height picks a dark floor, a bright
horizon streak or the sky), so it reads as polished metal in any world, a black one included; on the band between the
beads the engraving, drawn in the spike's own coordinates: two waves interlaced round the shank (REPEATS), a dot in
each loop and a line inside each bead, cut in by a bump and filled with the `engraving` colour."""
import math

from .. import shell as SH
from ..sets import nightkit as NK
from . import golden_spike_layout as LAY
from . import register
from .cafe_kit import Kit

REPEATS = 6                         # the waves' repeats round the shank
WAVE_A = 0.0068                     # the waves' amplitude (m, up and down the band's middle)
LINE_W = 0.00036                    # half the width of a groove (m)
DOT_R = 0.0016                      # the dots' radius (m)
BORDER = 0.0026                     # the border lines this far inside the beads (m)
STUDIO = ((0.0, 0.22), (0.4, 0.3), (0.47, 0.6), (0.5, 1.35), (0.55, 0.6), (0.8, 0.85), (1.0, 1.0))
GLOW = 0.75                         # the studio reflection's strength


def _engraving(nb, length):
    """0 .. 1: the engraving's grooves on the band (0 off it)."""
    x, y, z = nb.sep(nb.texcoord("Object"))
    b0, b1 = LAY.band(length)
    w, d, p = LAY.section_at(0.5 * (b0 + b1), length)
    lam = LAY.perimeter(w, d, p) / REPEATS                                   # one repeat round the shank (m)
    k = 2.0 * math.pi / lam
    a = nb.math("MULTIPLY", nb.math("ARCTAN2", y, x), lam * REPEATS / (2.0 * math.pi))   # m round the shank
    h = nb.math("SUBTRACT", z, 0.5 * (b0 + b1))                              # m from the band's middle
    s = nb.math("SINE", nb.math("MULTIPLY", a, k))
    c = nb.math("COSINE", nb.math("MULTIPLY", a, k))
    # a height off a wave, over g, is the distance to it (the wave's slope is WAVE_A k cos)
    g = nb.math("SQRT", nb.math("MULTIPLY_ADD", nb.math("MULTIPLY", c, c), (WAVE_A * k) ** 2, 1.0))
    d1 = nb.math("DIVIDE", nb.math("ABSOLUTE", nb.math("MULTIPLY_ADD", s, -WAVE_A, h)), g)
    d2 = nb.math("DIVIDE", nb.math("ABSOLUTE", nb.math("MULTIPLY_ADD", s, WAVE_A, h)), g)
    da = nb.math("SUBTRACT", nb.math("FLOORED_MODULO", a, 0.5 * lam), 0.25 * lam)      # from a loop's middle
    dot = nb.math("MAXIMUM", nb.math("SUBTRACT", nb.vlen(nb.comb(da, h, 0.0)), DOT_R - LINE_W), 0.0)   # a filled dot
    edge = nb.math("ABSOLUTE", nb.math("SUBTRACT", nb.math("ABSOLUTE", h), 0.5 * LAY.BAND_H - BORDER))
    near = nb.math("MINIMUM", nb.math("MINIMUM", d1, d2), nb.math("MINIMUM", dot, edge))
    groove = nb.math("SUBTRACT", 1.0, nb.smooth(0.55 * LINE_W, LINE_W, near))
    inside = nb.math("SUBTRACT", 1.0, nb.smooth(0.5 * LAY.BAND_H - LAY.BEAD_R - 0.0004, 0.5 * LAY.BAND_H - LAY.BEAD_R,
                                                 nb.math("ABSOLUTE", h)))
    return nb.math("MULTIPLY", groove, inside)


def _gold_mat(name, metal, engraving, length):
    m, nb = NK.new_material(f"{name}_gold")
    co = nb.texcoord("Object")
    cut = _engraving(nb, length)
    scratch = nb.noise(nb.node("ShaderNodeMapping", {"Vector": co, "Scale": (60.0, 900.0, 260.0)}).outputs[0],
                       scale=1.0, detail=3.0, rough=0.6)
    rough = nb.mixf(cut, nb.math("MULTIPLY_ADD", scratch, 0.12, 0.13), 0.42)
    col = nb.mixc(cut, metal, engraving)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.8, "Distance": 0.00025, "Height": cut},
                   invert=True).outputs["Normal"]
    b = nb.principled(col, rough=rough, metal=1.0, normal=bump, emission=col, strength=0.0)
    _studio(nb, b, col)
    nb.output(b)
    return m


def _studio(nb, b, col):
    """The studio's reflection in `b`'s emission: `col` times STUDIO at the reflected ray's height (world z, -1 .. 1
    mapped to 0 .. 1)."""
    up = nb.remap(nb.sep(nb.texcoord("Reflection"))[2], -1.0, 1.0)
    k = nb.fcurve(up, STUDIO, interp="EASE")
    nb.set(b.inputs["Emission Color"], nb.mixc(1.0, col, nb.comb(k, k, k), blend="MULTIPLY"))
    b.inputs["Emission Strength"].default_value = GLOW


@register("golden_spike")
def golden_spike(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    length = LAY.options(K.slots)["length"]
    mat = _gold_mat(name, K.role("metal", LAY.ROLES), K.role("engraving", LAY.ROLES), length)
    for part, mesh in LAY.meshes(length).items():
        SH.mesh_object(f"{name}_{part}", mesh, coll, root, lambda i: mat)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in LAY.ROLES}, length)
