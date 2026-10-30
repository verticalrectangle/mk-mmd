"""pillow_cloud: a plump pillow that puffs into a cartoon cloud (`card = "library:pillow_cloud"`).

    [[prop]]
    name = "pillow"
    card = "library:pillow_cloud"
    at = [0.0, -0.25, 1.05]                        # held up before her chest (it is upright: faces -Y, origin its centre)

    [[key]]                                        # it puffs into a cloud on a word, overshooting a little
    target = "pillow"
    prop = "puff"
    keys = [[1.73, 0.0], [1.9, 1.12], [2.05, 1.0]]

Objects `<name>_pillow` (the two fabric panels sewn round their edge, under a Subdivision Surface) and `<name>_piping`
(the cord sewn into the seam), each with the shape key `cloud`: the same points moved onto a cumulus, the cord sinking
into it as a thread (pillow_cloud_layout has the geometry). The root property `puff` drives both keys (0 the pillow, 1 the
cloud; up to 1.25 overshoots) and the materials: the pillowcase's print (polka dots on the case colour, the cotton's
weave; folds at the ears are in the shape) fades into the cloud's fluff (white, a soft glow, its underside shaded) as it
swells. Colour roles (`slots`: a palette slot, a hex or a 'slot:weight,...' blend): case, dots, piping, cloud, shade;
`puff` in `slots` is the starting value. Card: `use.anchor` `hug` (the middle of its back, `dir` the way its front
faces), `use.rest` `hold_L` / `hold_R` (the palms' planes on its ends, normals out), `use.look` `centre`, `params`, `cloud`
(the cloud's size and bounds in the prop's frame)."""
import numpy as np

from .. import shell as SH
from ..sets import nightkit as NK
from . import pillow_cloud_layout as LAY
from . import register
from .cafe_kit import clear_drivers
from .convertible_palette import rgba

GLOW = 0.25                 # the cloud's own light (a cartoon cloud is never grey)
FLOOR = 0.12                # the pillowcase's (nothing renders black)
DOTS = (10.0, 7.0, 0.27)    # the polka dots on the fabric's uv: across, up, radius (share of a cell)


def _drive(owner, path, expr, root, prop, index=-1):
    """A scripted driver `expr` on owner.path reading the root's property `prop`; refused unless it is a Blender simple
    expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    v = d.variables.new()
    v.name, v.type = prop, "SINGLE_PROP"
    v.targets[0].id_type, v.targets[0].id = "OBJECT", root
    v.targets[0].data_path = f'["{prop}"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"pillow_cloud: the driver on {path} is not a Blender simple expression ({expr})")
    return fc


def _dots(nb, uv):
    """1 inside the polka dots of a staggered grid over the panel's uv (DOTS: cells about 56 x 54 mm on the pillow), 0
    between them."""
    u, v, _ = nb.sep(uv)
    x, y = nb.math("MULTIPLY", u, DOTS[0]), nb.math("MULTIPLY", v, DOTS[1])
    row = nb.math("FLOOR", y)
    x = nb.math("ADD", x, nb.math("MULTIPLY", nb.math("FLOORED_MODULO", row, 2.0), 0.5))
    fx, fy = nb.math("SUBTRACT", nb.math("FRACT", x), 0.5), nb.math("SUBTRACT", nb.math("FRACT", y), 0.5)
    d = nb.vlen(nb.comb(fx, fy, 0.0))
    return nb.math("SUBTRACT", 1.0, nb.smooth(DOTS[2] - 0.025, DOTS[2] + 0.025, d))


def _material(name, pal, root, part):
    """The material of the pillowcase (`part` "case": the print) or of the cord in its seam ("piping"), mixed into the
    cloud's fluff by the root's `puff` (a driven Value node)."""
    m, nb = NK.new_material(f"{name}_{part}")
    puff = nb.node("ShaderNodeValue")
    puff.name = puff.label = "mk_puff"
    puff.outputs[0].default_value = float(root["puff"])
    _drive(m.node_tree, f'nodes["{puff.name}"].outputs[0].default_value', "puff", root, "puff")
    k = nb.smooth(0.12, 0.8, puff.outputs[0])                           # 0 the pillowcase .. 1 the cloud
    obj = nb.texcoord("Object")
    if part == "case":                                                   # polka dots on the case colour
        uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
        fabric = nb.mixc(_dots(nb, uv), rgba(pal.lin("case")), rgba(pal.lin("dots")))
    else:
        fabric = rgba(pal.lin("piping"))
    weave = nb.noise(nb.node("ShaderNodeMapping", {"Vector": obj, "Scale": (900.0, 900.0, 900.0)}).outputs[0],
                     scale=1.0, detail=1.0, rough=0.5)
    # the cloud: its fluff from the shaded underside up to the bright crown, a cauliflower bump
    _, _, z = nb.sep(obj)
    up = nb.smooth(LAY.CLOUD_BASE + 0.02, LAY.CLOUD_BASE + 0.32, z)
    cloud = nb.mixc(up, rgba(pal.lin("shade")), rgba(pal.lin("cloud")))
    fluff = nb.noise(obj, scale=9.0, detail=3.0, rough=0.55)
    height = nb.mixf(k, nb.math("MULTIPLY", weave, 0.4), fluff)
    bump = nb.node("ShaderNodeBump", {"Strength": nb.mixf(k, 0.12, 0.35), "Distance": 0.004, "Height": height})
    base = nb.mixc(k, fabric, cloud)
    glow = nb.mixf(k, FLOOR, GLOW)
    b = nb.principled(base, rough=nb.mixf(k, 0.78, 0.95), emission=base, strength=glow, spec=0.3,
                      normal=bump.outputs["Normal"])
    nb.set(b.inputs["Sheen Weight"], nb.mixf(k, 0.35, 0.8))
    nb.set(b.inputs["Sheen Roughness"], 0.4)
    nb.output(b)
    return m


def _cloud_key(o, cloud, root, start):
    """The shape key `cloud` (positions (n, 3)) on the object's mesh, its value driven by the root's `puff`."""
    o.shape_key_add(name="Basis", from_mix=False)
    key = o.shape_key_add(name="cloud", from_mix=False)
    key.data.foreach_set("co", np.asarray(cloud, np.float32).ravel())
    key.slider_min, key.slider_max = -0.25, 1.5
    keys = o.data.shape_keys
    clear_drivers(keys)
    _drive(keys, 'key_blocks["cloud"].value', "puff", root, "puff")
    key.value = start


@register("pillow_cloud")
def pillow_cloud(name, coll, root, slots=None):
    slots = dict(slots or {})
    start = LAY.options(slots)
    pal = LAY.Pal(slots)
    clear_drivers(root)
    for k, (_d, (lo, hi), doc) in LAY.PARAMS.items():
        root[k] = start[k]
        root.id_properties_ui(k).update(description=doc, default=start[k], soft_min=lo, soft_max=hi)
    pillow, cloud = LAY.meshes()
    cord, thread = LAY.piping_meshes(pillow.V, cloud)
    mats = [_material(name, pal, root, part) for part in ("case", "piping")]
    o = SH.shell_object(f"{name}_pillow", pillow, coll, root, mats.__getitem__, levels=1, render_levels=2)
    _cloud_key(o, cloud, root, start["puff"])
    p = SH.shell_object(f"{name}_piping", cord, coll, root, mats.__getitem__, levels=1, render_levels=1)
    _cloud_key(p, thread, root, start["puff"])
    return LAY.card(name, pal.roles(), start, cloud)
