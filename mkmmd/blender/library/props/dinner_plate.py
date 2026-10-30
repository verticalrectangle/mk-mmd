"""The dinner plate (`card = "library:dinner_plate"`): a glazed ceramic plate, 0.27 m across, with stylised veggies on it.
Objects under the root: `<name>_plate` and, unless `veggies = false`, `<name>_veggies` (florets, carrot coins, peas, a
cherry tomato).

    [[prop]]
    name = "plate"
    card = "library:dinner_plate"
    slots = { band = "love", seed = 3 }          # veggies = false: an empty plate

Materials: the glaze (a clear coat over the `plate` colour; the coloured band and a thin line inside it on the rim's top,
drawn from the radius in the plate's own frame; the foot ring's bottom bare, unglazed), the florets (a dark green with a
granular bud bump), the stalks (pale green, a little light through them), the coins (the carrot's skin, flesh and core
from the coin's uv: giant_carrot_mats.flesh_colour), glossy peas, a glossy tomato, a satin calyx. The numbers, meshes
and card: dinner_plate_layout.py."""
from ....core import palette as PAL
from .. import shell as SH
from ..sets import nightkit as NK
from . import dinner_plate_layout as LAY
from . import giant_carrot_mats as CM
from . import register
from .cafe_kit import Kit

EDGE = 0.0005               # the band's and the line's edges are this soft (m)


def _plate(name, col):
    m, nb = NK.new_material(f"{name}_plate")
    x, y, z = nb.sep(nb.texcoord("Object"))
    r = nb.vlen(nb.comb(x, y, 0.0))
    nz = nb.sep(nb.texcoord("Normal"))[2]
    band = nb.math("SUBTRACT", nb.smooth(LAY.BAND[0] - EDGE, LAY.BAND[0] + EDGE, r),
                   nb.smooth(LAY.BAND[1] - EDGE, LAY.BAND[1] + EDGE, r))
    line = nb.math("SUBTRACT", 1.0, nb.smooth(LAY.LINE[1] - 0.5 * EDGE, LAY.LINE[1] + 0.5 * EDGE,
                                              nb.math("ABSOLUTE", nb.math("SUBTRACT", r, LAY.LINE[0]))))
    paint = nb.math("MULTIPLY", nb.math("MAXIMUM", band, line), nb.smooth(0.3, 0.6, nz))
    colour = nb.mixc(paint, col["plate"], col["band"])
    bare = nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.smooth(0.0008, 0.0016, z)),
                   nb.smooth(0.5, 0.8, nb.math("MULTIPLY", nz, -1.0)))
    colour = nb.mixc(bare, colour, tuple(0.8 * c for c in col["plate"][:3]) + (1.0,))
    b = nb.principled(colour, rough=nb.mixf(bare, 0.12, 0.8), coat=nb.mixf(bare, 0.7, 0.0), coat_rough=0.05, spec=0.5)
    nb.output(b)
    m.diffuse_color = col["plate"]
    return m


def _floret(name, col):
    m, nb = NK.new_material(f"{name}_floret")
    p = nb.texcoord("Object")
    buds = nb.voronoi(p, scale=420.0, feature="F1", which="Distance")
    tone = nb.remap(nb.noise(p, scale=90.0, detail=2.0), 0.3, 0.7, 0.75, 1.15)
    colour = nb.mixc(1.0, col["floret"], nb.comb(tone, tone, tone), blend="MULTIPLY")
    b = nb.principled(colour, rough=0.6, spec=0.35,
                      normal=nb.node("ShaderNodeBump", {"Strength": 0.6, "Distance": 0.0006, "Height": buds}))
    nb.output(b)
    m.diffuse_color = col["floret"]
    return m


def _plain(name, role, colour, rough, coat=0.0, through=0.0):
    m, nb = NK.new_material(f"{name}_{role}")
    tone = nb.remap(nb.noise(nb.texcoord("Object"), scale=160.0, detail=2.0), 0.3, 0.7, 0.92, 1.06)
    c = nb.mixc(1.0, colour, nb.comb(tone, tone, tone), blend="MULTIPLY")
    b = nb.principled(c, rough=rough, coat=coat, coat_rough=0.08, spec=0.5)
    nb.output(nb.mix_shader(through, b, nb.node("ShaderNodeBsdfTranslucent", {"Color": c})) if through else b)
    m.diffuse_color = colour
    return m


def _coin(name, col):
    m, nb = NK.new_material(f"{name}_coin")
    rel = nb.sep(nb.texcoord("UV"))[0]
    b = nb.principled(CM.flesh_colour(nb, rel, col), rough=0.38, coat=0.25, coat_rough=0.2, spec=0.5)
    nb.output(b)
    m.diffuse_color = col["flesh"]
    return m


def materials(name, col):
    """{material: Blender material} for dinner_plate_layout.MATS, from the roles' linear colours `col`."""
    return {"plate": _plate(name, col), "floret": _floret(name, col),
            "stalk": _plain(name, "stalk", col["stalk"], 0.45, through=0.15), "coin": _coin(name, col),
            "pea": _plain(name, "pea", col["pea"], 0.28, coat=0.3), "tomato": _plain(name, "tomato", col["tomato"], 0.2,
                                                                                         coat=0.6),
            "calyx": _plain(name, "calyx", col["calyx"], 0.5)}


@register("dinner_plate")
def dinner_plate(name, coll, root, slots=None):
    opts = LAY.options(slots)
    K = Kit(name, coll, root, slots)
    mats = materials(name, {role: K.role(role, opts["roles"]) for role in LAY.ROLES})
    mat_of = lambda i: mats[LAY.MATS[i]]                                             # noqa: E731
    SH.mesh_object(f"{name}_plate", LAY.plate_mesh(), coll, root, mat_of)
    SH.purge(f"{name}_veggies")
    if opts["veggies"]:
        SH.mesh_object(f"{name}_veggies", LAY.veggies_mesh(opts["seed"]), coll, root, mat_of)
    roles = {role: PAL.resolve(v, K.colors.p) for role, v in opts["roles"].items()}
    return LAY.card(name, roles, K.size(), opts["veggies"])
