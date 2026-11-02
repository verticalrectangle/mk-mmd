"""idol_stage: a pastel idol stage (card `library:idol_stage`), built procedurally: a riser with a rounded plan, a padded
skirt and a rolled bullnose edge, piping round it, soft cushion steps, LED panels in heart and star shapes on slim posts
at the back of the deck (their faces LED dot matrices lit by a chase) and pixel bars along the front fascia under the
bullnose, chased by the same field. Numbers, meshes, the chase maths and the card: idol_stage_layout.py.

    [[prop]]
    name = "stage"
    card = "library:idol_stage"
    slots = { deck = "rose", steps = "sides", speed = 1.2 }        # roles, options, starting values

    [[cast]]
    name = "rin"
    asset = "my_model"
    at = [0.0, -0.288, 0.6]                        # on the deck's centre mark
    [pose.rin]
    feet = "stage:centre"

    [[key]]                                        # twice as fast from 12 s
    target = "stage"
    prop = "speed"
    keys = [[0.0, 1.2], [12.0, 1.2], [12.5, 2.4]]

Live root properties for [[key]] (docs/design.md: Keys): `chase` (the phase in cycles; the panels' and bars' materials read
it through driven Value nodes; keyable directly, e.g. stepped on hits), `speed` (cycles a second: the keys stage integrates
it into `chase`; unkeyed `chase` runs at the starting speed from clip second 0), `on` (0..1, the panels' level) and `bars`
(0..1, the light bars'). Drivers only: nothing runs at render time.

Objects `<name>_<part>`, every one at identity under the root (its mesh in the stage's frame, so the LED materials read the
stage's frame as their Object coordinates: mkr draws no Texture Coordinate node with an object set): riser, trim, steps
(unless steps = "none"), bars (when the stage is tall enough for them), panel<i> (housing, LED face, piping bead, posts).
Materials `<name>_<role>` (deck, skirt, trim, treads, housing, metal), `<name>_led<i>` (panel i's dot matrix) and
`<name>_pixels` (the bars' lens)."""
import math

import bpy

from ...keys import set_fcurve
from .. import shell as SH
from . import idol_stage_layout as LAY
from . import register
from . import showkit as SK
from .cafe_kit import Kit, clear_drivers

LOOK = {"deck": dict(rough=0.38, coat=0.25, spec=0.5), "skirt": dict(rough=0.6, spec=0.35, sheen=0.35),
        "trim": dict(rough=0.28, metal=0.7, spec=0.5), "treads": dict(rough=0.45, coat=0.2, spec=0.45),
        "housing": dict(rough=0.35, coat=0.4, spec=0.5), "metal": dict(rough=0.3, metal=0.85, spec=0.5)}
LENS_K = 0.12                       # an unlit dot's lens: this far from the ground toward the dim colour
FACE = dict(rough=0.5, spec=0.25)   # the LED face: matte black plastic, so the scene's lights do not wash the dots out


# ===================================================================================================================
# driven values and the shader-node backend of the layout's LED field
# ===================================================================================================================
def _param(nt, prop, root):
    """A Value node `mk_<prop>` in `nt` driven by the root's custom property `prop` (a Blender simple expression,
    checked): its output socket."""
    n = nt.nodes.new("ShaderNodeValue")
    n.name = n.label = f"mk_{prop}"
    n.outputs[0].default_value = float(root.get(prop, 0.0))     # the value before the driver evaluates
    d = nt.driver_add(f'nodes["{n.name}"].outputs[0].default_value').driver
    d.type = "SCRIPTED"
    v = d.variables.new()
    v.name, v.type = prop, "SINGLE_PROP"
    v.targets[0].id_type, v.targets[0].id = "OBJECT", root
    v.targets[0].data_path = f'["{prop}"]'
    d.expression = prop
    if not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"{LAY.WHO}: the driver of {nt.name}'s {n.name} is not a Blender simple expression ({prop})")
    return n.outputs[0]


class Float:
    """A float socket of the node backend: + - * between sockets and Python numbers make Math nodes."""
    __array_ufunc__ = None
    __slots__ = ("g", "sock")

    def __init__(self, g, sock):
        self.g, self.sock = g, sock

    def __add__(self, b):
        return self.g.math("ADD", self, b)

    def __radd__(self, a):
        return self.g.math("ADD", a, self)

    def __sub__(self, b):
        return self.g.math("SUBTRACT", self, b)

    def __rsub__(self, a):
        return self.g.math("SUBTRACT", a, self)

    def __mul__(self, b):
        return self.g.math("MULTIPLY", self, b)

    def __rmul__(self, a):
        return self.g.math("MULTIPLY", a, self)


_PY = {"ADD": lambda a, b: a + b, "SUBTRACT": lambda a, b: a - b, "MULTIPLY": lambda a, b: a * b,
       "DIVIDE": lambda a, b: a / b if b != 0.0 else 0.0, "FLOOR": lambda a, _b: float(math.floor(a)),
       "FRACT": lambda a, _b: a - math.floor(a), "ABSOLUTE": lambda a, _b: abs(a),
       "SQRT": lambda a, _b: math.sqrt(a) if a > 0.0 else 0.0, "ARCTAN2": math.atan2,
       "FLOORED_MODULO": lambda a, b: a - math.floor(a / b) * b if b != 0.0 else 0.0}


class Nodes:
    """The ops of idol_stage_layout's LED field as shader nodes of `nt` (one node per op; Python numbers stay constants,
    folded when both operands are numbers), so a material computes what the layout's numpy reference computes."""

    def __init__(self, nt):
        self.nt, self.count = nt, 0

    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (-2600.0 + 200.0 * (self.count // 20), 1400.0 - 120.0 * (self.count % 20))
        self.count += 1
        return n

    def put(self, sock, v):
        if isinstance(v, Float):
            self.nt.links.new(v.sock, sock)
        else:
            sock.default_value = float(v)

    def math(self, op, a, b=None):
        if not isinstance(a, Float) and not isinstance(b, Float):
            return _PY[op](float(a), None if b is None else float(b))
        n = self.node("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return Float(self, n.outputs[0])

    def floor(self, x):
        return self.math("FLOOR", x)

    def fract(self, x):
        return self.math("FRACT", x)

    def abs(self, x):
        return self.math("ABSOLUTE", x)

    def sqrt(self, x):
        return self.math("SQRT", x)

    def fmod(self, a, b):
        return self.math("FLOORED_MODULO", a, b)

    def atan2(self, y, x):
        return self.math("ARCTAN2", y, x)

    def div(self, a, b):
        return self.math("DIVIDE", a, b)

    def smoothstep(self, a, b, x):
        """Map Range's smoothstep: (3 - 2 f) f^2 of f = (x - a) / (b - a) clamped to 0..1."""
        n = self.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP")
        for i, v in enumerate((x, a, b, 0.0, 1.0)):
            self.put(n.inputs[i], v)
        return Float(self, n.outputs["Result"])

    def curve(self, t, table):
        """A Float Curve through the table's points (straight between them: vector handles) of t on 0..1."""
        n = self.node("ShaderNodeFloatCurve")
        pts = n.mapping.curves[0].points
        xs, ys = table
        pts[0].location, pts[1].location = (float(xs[0]), float(ys[0])), (float(xs[-1]), float(ys[-1]))
        for x, y in zip(xs[1:-1], ys[1:-1]):
            pts.new(float(x), float(y))
        for p in pts:
            p.handle_type = "VECTOR"
        n.mapping.update()
        n.inputs["Factor"].default_value = 1.0
        self.put(n.inputs["Value"], t)
        return Float(self, n.outputs["Value"])

    def xz(self):
        """The stage-frame x and z of the shading point: the object's own coordinates (it stands at identity)."""
        tc = self.node("ShaderNodeTexCoord")
        sep = self.node("ShaderNodeSeparateXYZ")
        self.nt.links.new(tc.outputs["Object"], sep.inputs[0])
        return Float(self, sep.outputs["X"]), Float(self, sep.outputs["Z"])


# ===================================================================================================================
# materials
# ===================================================================================================================
def _rgb(nt, name, spec, slots, what, loc):
    n = nt.nodes.new("ShaderNodeRGB")
    n.name = n.label = name
    n.location = loc
    n.outputs[0].default_value = SK.rgba(spec, slots, what)
    return n.outputs[0]


def _mix(nt, f, a, b, loc):
    """Mix (RGBA) of the colour sockets a and b by f (a number or a socket)."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type, n.blend_type, n.clamp_factor, n.location = "RGBA", "MIX", True, loc
    if isinstance(f, (int, float)):
        n.inputs[0].default_value = float(f)
    else:
        nt.links.new(f, n.inputs[0])
    nt.links.new(a, n.inputs[6])
    nt.links.new(b, n.inputs[7])
    return n.outputs[2]


def plain_material(K, role, spec):
    """`<name>_<role>`: a Principled in the role's colour (LOOK: roughness, coat, sheen, metal)."""
    c = SK.rgba(spec, K.slots, f"{LAY.WHO}: {role}")
    m = K.simple_mat(role, c, **LOOK[role])
    m.diffuse_color = c
    return m


def led_material(K, base, colours, field, level_prop, glow):
    """`<name>_<base>`: an LED face. colours = (ground, alt, lit) specs; field(nodes, x, z, chase) -> (dot, level) is the
    layout's field run on nodes (the stage-frame Object coordinates, the driven `chase`). A Principled whose base colour
    is the dark ground with the dots' lenses, a dot emitting lit * level + alt * ALT_K (1 - level) (the two lights added,
    not their hues mixed) times glow * <level_prop>: as the colour mix(alt, lit, level / w) at the strength dot *
    <level_prop> * glow * w, w = level + ALT_K (1 - level)."""
    m, nt, out = K.new_mat(base)
    g = Nodes(nt)
    ground, alt, lit = (_rgb(nt, f"mk_{k}", spec, K.slots, f"{LAY.WHO}: {base} {k}", (-700, -300 - 160 * i))
                        for i, (k, spec) in enumerate(zip(("ground", "alt", "lit"), colours)))
    x, z = g.xz()
    dot, level = field(g, x, z, Float(g, _param(nt, "chase", K.root)))
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.location = (300, 0)
    b.inputs["Roughness"].default_value = FACE["rough"]
    b.inputs["Specular IOR Level"].default_value = FACE["spec"]
    nt.links.new(_mix(nt, dot.sock, ground, _mix(nt, LENS_K, ground, alt, (-400, -300)), (-200, -300)),
                 b.inputs["Base Color"])
    w = level + (LAY.ALT_K - LAY.ALT_K * level)
    nt.links.new(_mix(nt, g.div(level, w).sock, alt, lit, (-200, -500)), b.inputs["Emission Color"])
    strength = dot * Float(g, _param(nt, level_prop, K.root)) * (glow * w)
    nt.links.new(strength.sock, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    m.diffuse_color = SK.rgba(colours[2], K.slots)
    return m


def _chase(root, speed):
    """`chase` at a steady `speed`: 0 at clip second 0, linear both ways (the keys stage rewrites it when `speed` is
    keyed; a [[key]] on `chase` replaces it)."""
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    fc = set_fcurve(root, '["chase"]', 0, [f0, f0 + fps], [0.0, speed], interp="LINEAR")
    fc.extrapolation = "LINEAR"


@register("idol_stage")
def idol_stage(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = LAY.options(slots)
    rl = LAY.roles(slots)
    K = Kit(name, coll, root, slots)
    clear_drivers(root)
    for k, (_d, (lo, hi), doc) in LAY.PARAMS.items():
        root[k] = O["start"][k]
        root.id_properties_ui(k).update(description=doc, default=O["start"][k], soft_min=lo, soft_max=hi)
    root["chase"] = 0.0
    root.id_properties_ui("chase").update(description=LAY.CHASE_DOC)
    _chase(root, O["start"]["speed"])
    parts = LAY.meshes(O)
    by_index = {LAY.MATS[r]: plain_material(K, r, rl[r]) for r in LOOK}
    for i, p in enumerate(O["panels"]):
        f = LAY.field(p, O)
        by_index[LAY.LED0 + i] = led_material(K, f"led{i}", (rl["ground"], p["alt"], p["color"]),
                                              lambda g, x, z, chase, f=f: LAY.led(g, f, x, z, chase), "on", O["glow"])
    if "bars" in parts:
        zc = LAY.bars_z(O)
        by_index[LAY.MATS["pixels"]] = led_material(
            K, "pixels", (rl["ground"], rl["alt"], rl["pixels"]),
            lambda g, x, z, chase: LAY.pixel(g, x, z, chase, O["pitch"], O["wave"], zc), "bars", O["glow"])
    for part, mesh in parts.items():
        SH.mesh_object(f"{name}_{part}", mesh, coll, root, by_index.__getitem__)
    return LAY.card(name, O, rl, parts)
