"""colorbar_card: a TV colour-bar test card as a printed card (`card = "library:colorbar_card"`).

    [[prop]]
    name = "bars"
    card = "library:colorbar_card"
    at = [0.4, -0.6, 1.5]
    rot = [0.0, 20.0, 10.0]
    slots = { style = "dawn", size = [0.6, 0.45] }

One object `<name>_card`: a rounded 4:3 card (`size` [0.60, 0.45] m, `thick` 0.012), its edges rolled, its face printed
with the bars (EraLooks' mkmmd.core.colorbars `texture` at 1024 x 768, an sRGB byte image: `style` "both" the palette's
bars over the true ones, "dawn", "true"), its back with `colorbars.back`, its cut edge cream paper (role `edge`). The bars
take the palette from the slots (the project's palette and the prop's overrides; rose-pine-dawn without one). No
drivers: a project keys the root (an [[fx]] card tumbles its own). Card: `use.surface` `face` (the picture's rectangle on
the face, normal -Y), `use.look` `centre`. colorbar_card_layout has the geometry and the card."""
import bpy

from ..sets import nightkit as NK
from .. import shell as SH
from . import colorbar_card_layout as LAY
from . import register


def _image(name, rgb):
    """A packed sRGB byte image from a picture whose row 0 is its top (LAY.pixels flips it for Blender)."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(LAY.pixels(rgb))
    img.pack()
    return img


def _print(name, img):
    """A printed face: the picture on a satin card, a faint paper tooth in the bump."""
    m, nb = NK.new_material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap")
    tex = nb.node("ShaderNodeTexImage", {"Vector": uv.outputs[0]}, image=img, interpolation="Linear", extension="EXTEND")
    bump = nb.node("ShaderNodeBump", {"Strength": 0.05, "Distance": 0.0003,
                                      "Height": nb.noise(nb.texcoord("Object"), scale=1500.0, detail=3.0, rough=0.6)})
    nb.output(nb.principled(tex.outputs["Color"], rough=0.42, spec=0.4, normal=bump.outputs["Normal"]))
    m.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return m


def _edge(name, colour):
    """The card's cut edge: cream paper, matte."""
    m, nb = NK.new_material(name)
    nb.output(nb.principled((*colour, 1.0), rough=0.85, spec=0.25))
    m.diffuse_color = (*colour, 1.0)
    return m


@register("colorbar_card")
def colorbar_card(name, coll, root, slots=None):
    slots = dict(slots or {})
    size, thick, style = LAY.options(slots)
    pal = LAY.Pal(slots)
    try:
        edge = pal.lin("edge")
    except KeyError as e:
        raise ValueError(f"colorbar_card: colour role edge = {pal.specs['edge']!r}: give a palette slot, a hex or a blend "
                         f"'slot:weight,...' ({e})") from None
    face, back = LAY.pictures(size, LAY.palette(slots), style)
    mats = {"face": _print(f"{name}_face", _image(f"{name}_face", face)),
            "back": _print(f"{name}_back", _image(f"{name}_back", back)),
            "edge": _edge(f"{name}_edge", edge)}
    SH.mesh_object(f"{name}_card", LAY.mesh(size, thick), coll, root, lambda i: mats[LAY.MATS[i]])
    return LAY.card(name, size, thick, style, pal.roles())
