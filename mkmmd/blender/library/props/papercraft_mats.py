"""Materials of the wall papers (`paper_bunting`, `taped_papers`). A helper module: it registers no builder.

Paper is matte and fibrous: its colour (the mesh's colour attribute `mk_paper`, so one material colours every piece)
mottled in soft clouds and flecked with lighter fibres, a fine tooth in the bump, a sheen at grazing angles, both sides
drawn; a printed sheet is its picture (an sRGB image over the sheet's UV) on the same paper. Masking tape is crepe paper:
a cream ground with fine streaks across the strip, slightly see-through (blended). Twine is two plies twisted (stripes
round the cord along its UV, u in metres), fuzzy. Every pattern is noise in the object's own space or the UV, so it
holds still as the prop moves."""
import math

import bpy
import numpy as np

from ..sets import nightkit as NK


def _rgba(c, a=1.0):
    return (float(c[0]), float(c[1]), float(c[2]), a)


def _fibre(nb, P, colour, mottle=0.10, fleck=0.07):
    """`colour` mottled in soft clouds and flecked with lighter fibres (noise in the object's space)."""
    cloud = nb.math("SUBTRACT", nb.noise(P, scale=11.0, detail=3.0, rough=0.55), 0.5)
    fibre = nb.smooth(0.62, 0.72, nb.noise(P, scale=140.0, detail=3.0, rough=0.55))
    k = nb.math("ADD", nb.math("MULTIPLY_ADD", cloud, mottle, 1.0), nb.math("MULTIPLY", fibre, fleck))
    return nb.vmath("SCALE", colour, scale=k)


def _paper_bsdf(nb, P, colour, rough):
    bump = nb.node("ShaderNodeBump", {"Strength": 0.12, "Distance": 0.0004,
                                      "Height": nb.noise(P, scale=520.0, detail=3.0, rough=0.6)})
    b = nb.principled(colour, rough=rough, spec=0.25, normal=bump.outputs["Normal"])
    nb.set(b.inputs["Sheen Weight"], 0.3)
    nb.set(b.inputs["Sheen Roughness"], 0.5)
    return b


def paper(name, attr="mk_paper", rough=0.86):
    """Coloured paper: the colour attribute `attr` of the mesh (per piece) on fibrous matte paper."""
    m, nb = NK.new_material(name)
    P = nb.texcoord("Object")
    nb.output(_paper_bsdf(nb, P, _fibre(nb, P, nb.attr(attr, "Color")), rough))
    m.use_backface_culling = False
    m.diffuse_color = (0.8, 0.7, 0.6, 1.0)
    return m


def printed(name, image, gloss=None, rough=0.78):
    """A printed sheet: `image` over its UV on fibrous paper. `gloss` = (u0, v0, u1, v1, rough): a glossy window (an
    instant photo's picture) in the UV square."""
    m, nb = NK.new_material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", {"Vector": uv}, image=image, interpolation="Linear", extension="EXTEND")
    P = nb.texcoord("Object")
    col = _fibre(nb, P, tex.outputs["Color"], mottle=0.05, fleck=0.035)
    r = rough
    if gloss is not None:
        u0, v0, u1, v1, rg = gloss
        u, v, _ = nb.sep(uv)
        inside = nb.math("MULTIPLY", nb.math("MULTIPLY", nb.math("GREATER_THAN", u, u0), nb.math("LESS_THAN", u, u1)),
                         nb.math("MULTIPLY", nb.math("GREATER_THAN", v, v0), nb.math("LESS_THAN", v, v1)))
        r = nb.mixf(inside, rough, rg)
    nb.output(_paper_bsdf(nb, P, col, r))
    m.use_backface_culling = False
    m.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return m


def tape(name, rgb, alpha=0.9):
    """Masking tape: crepe paper (streaks across the strip, u across and v along in metres), a little see-through."""
    m, nb = NK.new_material(name, blended=True)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    u, v, _ = nb.sep(uv)
    crepe = nb.noise(nb.comb(nb.math("MULTIPLY", u, 70.0), nb.math("MULTIPLY", v, 1500.0), 0.0), scale=1.0, detail=4.0,
                     rough=0.6)
    blot = nb.noise(nb.comb(nb.math("MULTIPLY", u, 180.0), nb.math("MULTIPLY", v, 180.0), 3.0), scale=1.0, detail=2.0)
    k = nb.math("MULTIPLY_ADD", crepe, 0.12, nb.math("MULTIPLY_ADD", blot, 0.05, 0.88))
    col = nb.vmath("SCALE", _rgba(rgb), scale=k)
    a = nb.math("MULTIPLY_ADD", crepe, 0.10, alpha - 0.05)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.18, "Distance": 0.0002, "Height": crepe})
    nb.output(nb.principled(col, rough=0.72, spec=0.3, alpha=a, normal=bump.outputs["Normal"]))
    m.show_transparent_back = False
    m.diffuse_color = _rgba(rgb, alpha)
    return m


def twine(name, rgb, pitch=0.0055):
    """Jute twine: two plies twisted round the cord (the tube's UV: u along it in metres, v round it), fuzzy."""
    m, nb = NK.new_material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    u, v, _ = nb.sep(uv)
    ph = nb.math("MULTIPLY", nb.math("ADD", nb.math("MULTIPLY", u, 1.0 / pitch), nb.math("MULTIPLY", v, 2.0)),
                 2.0 * math.pi)
    ply = nb.smooth(-0.2, 0.9, nb.math("SINE", ph))
    fuzz = nb.noise(nb.texcoord("Object"), scale=900.0, detail=2.0)
    k = nb.math("MULTIPLY_ADD", ply, 0.28, nb.math("MULTIPLY_ADD", fuzz, 0.12, 0.70))
    b = nb.principled(nb.vmath("SCALE", _rgba(rgb), scale=k), rough=0.9, spec=0.25)
    nb.set(b.inputs["Sheen Weight"], 0.5)
    nb.set(b.inputs["Sheen Roughness"], 0.4)
    nb.output(b)
    m.diffuse_color = _rgba(rgb)
    return m


def plastic(name, rgb, rough=0.25):
    m, nb = NK.new_material(name)
    nb.output(nb.principled(_rgba(rgb), rough=rough, spec=0.5, coat=0.5, coat_rough=0.1))
    m.diffuse_color = _rgba(rgb)
    return m


def paint(o, rgb):
    """Give the object's mesh the colour attribute `mk_paper`: linear RGB per vertex (n, 3)."""
    me = o.data
    if "mk_paper" in me.attributes:
        me.attributes.remove(me.attributes["mk_paper"])
    a = me.attributes.new("mk_paper", "FLOAT_COLOR", "POINT")
    c = np.ones((len(me.vertices), 4), np.float32)
    c[:, :3] = rgb
    a.data.foreach_set("color", c.ravel())


def image(name, rgb):
    """A packed sRGB byte image from a picture (h, w, 3) whose row 0 is its top."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = "sRGB"
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = rgb[::-1]
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img
