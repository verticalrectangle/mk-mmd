"""The geometry nodes and materials of a [[ribbon]] (the ribbons stage, mkmmd/blender/build/ribbons.py, builds the objects
round them; docs/design.md: Ribbons).

The ribbon object's own mesh is its centre line: one vertex per point of mkmmd.core.ribbon's centre line (about a
centimetre apart) joined by edges, with the point attributes `mk_f` (the direction its flat side faces), `mk_w` (a
crayon's width scale) and `mk_s` (the arc length, for the materials). The node group "mk_ribbon_<name>" makes the
surface on every frame, with no Python at render time:

  Mesh to Curve -> Trim Curve (factors Tail .. Grow of its length) -> Resample Curve (Spacing apart, both ends kept)
  -> Subdivide Curve (CAP_CUTS more points in each segment within a cap of either end, so the domes are smooth and
  the body stays light) -> Set Curve Tilt (the curve's normal turned onto `mk_f`: the profile's x lies along it) ->
  Set Curve Radius (Half x mk_w x Thick, rounded off over Cap radii at both ends: a quarter circle, so each end is a
  dome) -> Curve to Mesh with the profile (a circle of RESOLUTION points scaled to Aspect across x, its unit position
  stored as `mk_pv`) -> the materials (a toothpaste's stripes by the angle of `mk_pv` round the profile, stripe 0
  centred on the side `mk_f` faces) -> smooth shading; an empty mesh while Grow - Tail is under a micrometre.

Grow, Tail and Thick are inputs of the modifier, driven by the ribbon root's properties of the same names (lower case).
A neon's glow is a second object with the same mesh and a tree of its own: a round tube `halo.size` half widths in
radius whose additive emission falls off as a Gaussian of the distance from its axis as the camera sees it."""
import math

import bpy

from ..library.sets import nightkit as NK

INPUTS = ("Grow", "Tail", "Thick")            # driven from the root's grow, tail, thick
RESOLUTION = {"crayon": 12, "toothpaste": 20, "neon": 10, "halo": 10}    # profile points (a toothpaste's: 4 a stripe)
CAP_CUTS = 5                                  # points added to each segment near an end


class _G:
    """A geometry node tree under construction: `node` adds one (inputs by name or index, attributes as keywords),
    `math` / `vmath` one operation, Python numbers staying input defaults."""

    def __init__(self, nt):
        self.nt, self.k = nt, 0

    def node(self, kind, ins=None, **attrs):
        n = self.nt.nodes.new(kind)
        for a, v in attrs.items():
            setattr(n, a, v)
        n.location = (220.0 * (self.k % 10), -190.0 * (self.k // 10))
        self.k += 1
        for key, v in (ins or {}).items():
            self.put(n.inputs[key], v)
        return n

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None):
        return self.node("ShaderNodeMath", {0: a, 1: b}, operation=op).outputs[0]

    def vmath(self, op, a, b=None):
        n = self.node("ShaderNodeVectorMath", {0: a, 1: b}, operation=op)
        return n.outputs["Value"] if op == "DOT_PRODUCT" else n.outputs["Vector"]

    def attr(self, name, dtype="FLOAT"):
        return self.node("GeometryNodeInputNamedAttribute", {"Name": name}, data_type=dtype).outputs["Attribute"]


def tree(name, style, half, aspect, spacing, mats, cap=1.0):
    """The node group `name` (an older one of that name is replaced): the surface of one ribbon. `style` picks the
    profile's resolution ("halo" for a neon's glow shell), `half` is the half width (m), `aspect` the thickness over the
    width (the profile's x), `spacing` the resample distance (m), `mats` the materials (a toothpaste's stripes in
    order; one otherwise), `cap` the length of the rounded ends in radii."""
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    nt = bpy.data.node_groups.new(name, "GeometryNodeTree")
    nt.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    for k, dv in zip(INPUTS, (1.0, 0.0, 1.0)):
        nt.interface.new_socket(k, in_out="INPUT", socket_type="NodeSocketFloat").default_value = dv
    g = _G(nt)
    gi, go = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    curve = g.node("GeometryNodeMeshToCurve", {"Mesh": gi.outputs["Geometry"]}).outputs["Curve"]
    trim = g.node("GeometryNodeTrimCurve", mode="FACTOR")
    g.put(trim.inputs["Curve"], curve)
    g.put(trim.inputs[2], gi.outputs["Tail"])                      # Start, End (factor sockets; then the length ones)
    g.put(trim.inputs[3], gi.outputs["Grow"])
    count = g.math("ADD", g.math("CEIL", g.math("DIVIDE", g.node("GeometryNodeSplineLength").outputs["Length"],
                                                 spacing)), 1.0)
    res = g.node("GeometryNodeResampleCurve", {"Curve": trim.outputs["Curve"], "Count": count}, mode="COUNT")
    par = g.node("GeometryNodeSplineParameter").outputs["Length"]
    length = g.node("GeometryNodeSplineLength").outputs["Length"]
    d = g.math("MINIMUM", par, g.math("SUBTRACT", length, par))           # to the nearer end, on whatever curve
    radius = g.math("MULTIPLY", g.math("MULTIPLY", g.attr("mk_w"), gi.outputs["Thick"]), half)
    near = g.math("LESS_THAN", g.math("SUBTRACT", d, g.math("MULTIPLY", radius, cap)), 1.5 * spacing)
    fine = g.node("GeometryNodeSubdivideCurve", {"Curve": res.outputs["Curve"],
                                                 "Cuts": g.math("MULTIPLY", near, float(CAP_CUTS))})
    want = g.attr("mk_f", "FLOAT_VECTOR")
    nrm = g.node("GeometryNodeInputNormal").outputs["Normal"]
    tan = g.node("GeometryNodeInputTangent").outputs["Tangent"]
    tilt = g.math("ARCTAN2", g.vmath("DOT_PRODUCT", g.vmath("CROSS_PRODUCT", nrm, want), tan),
                  g.vmath("DOT_PRODUCT", nrm, want))
    tilted = g.node("GeometryNodeSetCurveTilt", {"Curve": fine.outputs["Curve"], "Tilt": tilt}).outputs["Curve"]
    x = g.math("MINIMUM", g.math("DIVIDE", d, g.math("MAXIMUM", g.math("MULTIPLY", radius, cap), 1e-6)), 1.0)
    one = g.math("SUBTRACT", 1.0, x)
    dome = g.math("SQRT", g.math("MAXIMUM", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", one, one)), 0.0))
    sized = g.node("GeometryNodeSetCurveRadius", {"Curve": tilted, "Radius": g.math("MULTIPLY", radius, dome)})
    circle = g.node("GeometryNodeCurvePrimitiveCircle", {"Resolution": RESOLUTION[style], "Radius": 1.0}, mode="RADIUS")
    unit = g.node("GeometryNodeStoreNamedAttribute", {"Geometry": circle.outputs["Curve"], "Name": "mk_pv",
                                                      "Value": g.node("GeometryNodeInputPosition").outputs["Position"]},
                  data_type="FLOAT_VECTOR", domain="POINT")
    profile = g.node("GeometryNodeTransform", {"Geometry": unit.outputs["Geometry"], "Scale": (aspect, 1.0, 1.0)})
    geo = g.node("GeometryNodeCurveToMesh", {"Curve": sized.outputs["Curve"],
                                             "Profile Curve": profile.outputs["Geometry"]}).outputs["Mesh"]
    geo = g.node("GeometryNodeSetMaterial", {"Geometry": geo, "Material": mats[0]}).outputs["Geometry"]
    if len(mats) > 1:
        pv = g.node("ShaderNodeSeparateXYZ", {0: g.attr("mk_pv", "FLOAT_VECTOR")})
        turn = g.math("FRACT", g.math("ADD", g.math("DIVIDE", g.math("ARCTAN2", pv.outputs[1], pv.outputs[0]),
                                                    2.0 * math.pi), 0.5 / len(mats)))
        band = g.math("FLOOR", g.math("MULTIPLY", turn, float(len(mats))))
        for i, m in enumerate(mats[1:], 1):
            sel = g.node("FunctionNodeCompare", {0: band, 1: float(i), "Epsilon": 0.25}, data_type="FLOAT",
                         operation="EQUAL").outputs[0]
            geo = g.node("GeometryNodeSetMaterial", {"Geometry": geo, "Selection": sel, "Material": m}).outputs["Geometry"]
    geo = g.node("GeometryNodeSetShadeSmooth", {"Geometry": geo}).outputs["Geometry"]
    shown = g.node("FunctionNodeCompare", {0: g.math("SUBTRACT", gi.outputs["Grow"], gi.outputs["Tail"]), 1: 1e-6},
                   data_type="FLOAT", operation="GREATER_THAN").outputs[0]
    sw = g.node("GeometryNodeSwitch", {"Switch": shown, "True": geo}, input_type="GEOMETRY")
    nt.links.new(sw.outputs[0], go.inputs["Geometry"])
    return nt


def inputs(nt):
    """{input name: the modifier's property identifier} of a ribbon tree."""
    return {s.name: s.identifier for s in nt.interface.items_tree
            if s.item_type == "SOCKET" and s.in_out == "INPUT" and s.name in INPUTS}


# ------------------------------------------------------------------------------------------------------ materials
def _uv(nb):
    """(u along the ribbon in metres, v round its profile 0..1) from the mesh's mk_s and mk_pv."""
    x, y, _ = nb.sep(nb.attr("mk_pv", "Vector"))
    v = nb.math("ADD", nb.math("DIVIDE", nb.math("ARCTAN2", y, x), 2.0 * math.pi), 0.5)
    return nb.attr("mk_s", "Fac"), v


def _viewport(m, rgb, alpha=1.0):
    """What Workbench (and the vector look) draws for the material: its colour, and `alpha` (a look leaves a material
    under half opaque out)."""
    m.diffuse_color = (*rgb, alpha)


def crayon(name, rgb, width):
    """Wax crayon: matte with a soft sheen, streaked along the stroke (the wax laid in lines, a darker and a lighter
    one) and pitted where the paper's tooth took no wax, the streaks raised a little (a bump)."""
    m, nb = NK.new_material(name)
    u, v = _uv(nb)
    k = 0.05 / max(width, 1e-3)
    streak = nb.noise(nb.comb(nb.math("MULTIPLY", u, 2.2 * k), nb.math("MULTIPLY", v, 26.0), 0.0), scale=1.0, detail=3.0,
                      rough=0.55)
    tooth = nb.noise(nb.comb(nb.math("MULTIPLY", u, 55.0 * k), nb.math("MULTIPLY", v, 60.0), 3.1), scale=1.0, detail=1.0)
    pits = nb.smooth(0.62, 0.74, tooth)
    shade = nb.math("ADD", nb.math("MULTIPLY", nb.math("SUBTRACT", streak, 0.5), 0.45), 1.0)
    base = nb.vmath("MULTIPLY", rgb, nb.comb(shade, shade, shade))
    base = nb.mixc(nb.math("MULTIPLY", pits, 0.55), base, tuple(min(1.0, 0.6 * c + 0.4) for c in rgb) + (1.0,))
    bump = nb.node("ShaderNodeBump", {"Strength": 0.35, "Distance": 0.002, "Height": nb.math("ADD", streak,
                                                                                           nb.math("MULTIPLY", pits, -0.6))})
    sh = nb.principled(base, rough=0.78, spec=0.32, normal=bump.outputs["Normal"])
    sh.inputs["Sheen Weight"].default_value = 0.35
    sh.inputs["Sheen Roughness"].default_value = 0.4
    nb.output(sh)
    _viewport(m, rgb)
    return m


def paste(name, rgb):
    """One stripe of toothpaste: glossy (a clear coat over a smooth gel) and a little translucent."""
    m, nb = NK.new_material(name)
    sh = nb.principled(rgb + (1.0,), rough=0.22, spec=0.55, coat=0.8, coat_rough=0.06)
    sh.inputs["Subsurface Weight"].default_value = 0.12
    sh.inputs["Subsurface Radius"].default_value = (0.02, 0.012, 0.008)
    nb.output(sh)
    _viewport(m, rgb)
    return m


def neon(name, rgb, glow):
    """A neon marker's ink: its own light, hotter across the middle of the ribbon (toward white) than at its edges."""
    m, nb = NK.new_material(name)
    _, y, _ = nb.sep(nb.attr("mk_pv", "Vector"))
    core = nb.math("POWER", nb.math("SUBTRACT", 1.0, nb.math("ABSOLUTE", y)), 2.0)
    hot = tuple(min(1.0, 0.55 * c + 0.45) for c in rgb) + (1.0,)
    nb.output(nb.emission(nb.mixc(nb.math("MULTIPLY", core, 0.6), rgb + (1.0,), hot), glow))
    _viewport(m, rgb)
    return m


def halo(name, rgb, strength, sigma=0.35):
    """The glow round a neon: additive light on a round shell, a Gaussian of how far from its axis the camera sees a point
    (no rim), left out of the looks (viewport alpha 0)."""
    m, nb = NK.new_material(name, blended=True)
    mu = nb.math("ABSOLUTE", nb.vdot(nb.geometry("Normal"), nb.geometry("Incoming")))
    rho2 = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", mu, mu))
    fac = nb.math("EXPONENT", nb.math("MULTIPLY", rho2, -1.0 / (sigma * sigma)))
    nb.output(nb.add_shader(nb.transparent(), nb.emission(rgb + (1.0,), nb.math("MULTIPLY", fac, strength))))
    m.use_backface_culling = True
    _viewport(m, rgb, 0.0)
    return m
