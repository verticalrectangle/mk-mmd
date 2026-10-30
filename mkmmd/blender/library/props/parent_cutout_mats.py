"""Materials of the picture-book parent cut-outs (mkmmd/blender/library/props/parent_cutout.py). A helper module: it
registers no builder.

Construction paper is matte and fibrous: its colour mottled in soft clouds and flecked with lighter fibres, a fine tooth
in the bump, a sheen at grazing angles, its cut edges a little lighter (the dye does not reach the fibres there); the white
card of the borders shows a cream core on its cut edge and a greyer back. Every pattern is noise in the object's own
coordinates (metres as built, so a figure 3.2 m tall shows the flecks of a paper doll blown up to that size, and the
forearm keeps its paper as it turns); the cut edge is told from the faces by the object-space normal (the faces look along
y). The brads are brass. A drop shadow is the palette's ink, its alpha a blurred image of the outline (a graphic layer:
blended, seen from the front only, casting no shadow of its own)."""
import bpy
import numpy as np

from ..sets import nightkit as NK


def _rgba(c, a=1.0):
    return (float(c[0]), float(c[1]), float(c[2]), a)


def paper(name, colour, edge, back=None, rough=0.88, mottle=0.09, fleck=0.10, grain=0.18):
    """A paper or card material: `colour` (linear RGB) on the faces, `edge` on the cut edge, `back` on the face that looks
    along +y (default the colour)."""
    m, nb = NK.new_material(name)
    P = nb.texcoord("Object")
    _nx, ny, _nz = nb.sep(nb.texcoord("Normal"))
    side = nb.math("SUBTRACT", 1.0, nb.smooth(0.35, 0.75, nb.math("ABSOLUTE", ny)))          # 1 on the cut edge
    behind = nb.smooth(0.3, 0.7, ny)                                                          # 1 on the back face
    cloud = nb.math("SUBTRACT", nb.noise(P, scale=7.0, detail=4.0, rough=0.6), 0.5)
    fibre = nb.smooth(0.62, 0.70, nb.noise(P, scale=90.0, detail=3.0, rough=0.55))
    k = nb.math("ADD", nb.math("MULTIPLY", cloud, 2.0 * mottle), nb.math("ADD", nb.math("MULTIPLY", fibre, fleck), 1.0))
    face = nb.mixc(behind, _rgba(colour), _rgba(colour if back is None else back))
    col = nb.mixc(side, nb.vmath("SCALE", face, scale=k), _rgba(edge))
    bump = nb.node("ShaderNodeBump", {"Strength": grain, "Distance": 0.0006,
                                      "Height": nb.noise(P, scale=380.0, detail=4.0, rough=0.6)})
    b = nb.principled(col, rough=rough, spec=0.22, normal=bump.outputs["Normal"])
    nb.set(b.inputs["Sheen Weight"], 0.35)
    nb.set(b.inputs["Sheen Roughness"], 0.5)
    nb.output(b)
    m.diffuse_color = _rgba(colour)
    return m


def brass(name, colour):
    """The paper fasteners: brass, a soft polish, a faint glow of their own so they read in shade."""
    m, nb = NK.new_material(name)
    b = nb.principled(_rgba(colour), rough=0.3, metal=1.0, emission=_rgba(colour), strength=0.12)
    nb.output(b)
    m.diffuse_color = _rgba(colour)
    return m


def alpha_image(name, alpha):
    """A packed byte image (Non-Color) of an alpha mask (h, w) whose row 0 is its top (Blender's rows run up)."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    h, w = alpha.shape
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    px = np.empty((h, w, 4), np.float32)
    px[..., :3] = np.clip(alpha[::-1], 0.0, 1.0)[..., None]
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def shadow(name, image, colour, strength):
    """A drop shadow: flat `colour` (linear RGB) at `strength` of the image's value, the rest see-through, and nothing
    seen from behind (Cycles ignores backface culling: the shader drops its back face); blended."""
    m, nb = NK.new_material(name, blended=True)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap")
    tex = nb.node("ShaderNodeTexImage", {"Vector": uv.outputs[0]}, image=image, interpolation="Linear",
                  extension="CLIP")
    front = nb.math("SUBTRACT", 1.0, nb.geometry("Backfacing"))
    a = nb.math("MULTIPLY", nb.math("MULTIPLY", tex.outputs["Color"], strength), front)
    nb.output(nb.mix_shader(a, nb.transparent(), nb.emission(_rgba(colour), 1.0)))
    m.show_transparent_back = False
    m.diffuse_color = _rgba(colour, strength)
    return m
