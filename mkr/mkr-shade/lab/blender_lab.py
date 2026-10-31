"""mkr-shade lab, Blender side: render node setups with EEVEE Next (Blender 4.2) to float EXR and export each setup's
node graph with the bake's own exporter (mkmmd.blender.build.bake_lit.graph), so `cargo run -p mkr-shade --example lab`
can evaluate the same graph on the GPU and compare per pixel.

    ~/blender-portable/blender-4.2.3-linux-x64/blender -b --factory-startup \\
        --python mkr/mkr-shade/lab/blender_lab.py -- OUT_DIR [--only GLOB[,GLOB...]]

Kinds of case:
  field    an N x N grid of unit quads under an orthographic camera, K x K render pixels each; every quad carries
           face attributes lab_a, lab_b, lab_c (vectors) the setup reads through Attribute nodes, and its emission is the
           value under test; the reference is each quad's centre pixel (out of reach of the film filter)
  closure  the same grid lit by a uniform white world: the pixel is the sum of the closure colours (EEVEE's world
           light is 1 for every lobe); each row tilts the shading normal (custom normals) so N.I sweeps 0..1
  shade    the emission of a value that depends on the shading point, the normal tilted per row as for closures
  lit      closures under one sun lamp and a black world (`sun` degrees about X, `slab` thickness mode): how EEVEE
           lights a closure, i.e. the renderer's side of the contract; `examples/lab.rs` skips these. Keep them in a
           folder of their own (`--only "lit_*"`)
Files per case: <name>.json (graph, groups, images, kind, n, rows), <name>.in.bin (float32 n*n*9: a, b, c per quad),
<name>.ref.bin (float32 n*n*4)."""
import fnmatch
import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from mkmmd.blender.build import bake_lit as BL  # noqa: E402

N = 48          # quads per side
K = 5           # render pixels per quad side
RNG = np.random.default_rng(7)


# ================================================================================================= scenes
def reset(kind, sun=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.eevee.taa_render_samples = 1
    sc.eevee.use_shadows = False
    sc.eevee.use_raytracing = False
    sc.render.filter_size = 1.5
    sc.render.dither_intensity = 0.0
    sc.render.resolution_x = sc.render.resolution_y = N * K
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    ims = sc.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_depth = "32"
    ims.color_mode = "RGBA"
    ims.exr_codec = "ZIP"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = float(N)
    cam.data.clip_start, cam.data.clip_end = 0.1, 100.0
    cam.location = (0.0, 0.0, 10.0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0) if kind == "closure" else (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    if sun is not None:
        # one sun (2 W/m², 0.5° wide) turned `sun` degrees about X: 30 lights the planes' front at cos 0.866, 150 their back
        ld = bpy.data.lights.new("sun", "SUN")
        ld.energy = 2.0
        ld.angle = math.radians(0.5)
        lo = bpy.data.objects.new("sun", ld)
        lo.rotation_euler = (math.radians(sun), 0.0, 0.0)
        sc.collection.objects.link(lo)
    return sc


def grid(sc, normals=None):
    """N x N unit quads centred on the origin, quad (i, j) = face j * N + i at x = -N/2 + i, y = -N/2 + j."""
    verts = [(-N / 2 + i, -N / 2 + j, 0.0) for j in range(N + 1) for i in range(N + 1)]
    faces = [(j * (N + 1) + i, j * (N + 1) + i + 1, (j + 1) * (N + 1) + i + 1, (j + 1) * (N + 1) + i)
             for j in range(N) for i in range(N)]
    me = bpy.data.meshes.new("grid")
    me.from_pydata(verts, [], faces)
    me.update()
    for name in ("lab_a", "lab_b", "lab_c"):
        me.attributes.new(name, "FLOAT_VECTOR", "FACE")
    uv = me.uv_layers.new(name="UVMap")
    co = np.array(verts)
    lc = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lc)
    uvs = (co[lc, :2] + N / 2) / N
    uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    if normals is not None:
        per_loop = np.repeat(np.asarray(normals, np.float64), 4, axis=0)
        me.normals_split_custom_set(per_loop.tolist())
    ob = bpy.data.objects.new("grid", me)
    sc.collection.objects.link(ob)
    return ob


def set_inputs(ob, a, b, c):
    for name, arr in (("lab_a", a), ("lab_b", b), ("lab_c", c)):
        ob.data.attributes[name].data.foreach_set("vector", np.asarray(arr, np.float32).ravel())


def material(ob):
    m = bpy.data.materials.new("lab")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    ob.data.materials.append(m)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def render(sc, path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path, check_existing=False)
    px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
    bpy.data.images.remove(im)
    os.remove(path)
    return px


def centres(px):
    """Each quad's centre pixel, (N * N, 4) in face order (Blender's rows run bottom to top)."""
    c = K // 2
    return px[c::K, c::K, :][:N, :N].reshape(N * N, 4)


# ================================================================================================= node helpers
class Socks:
    """A node's sockets by identifier, then name, enabled or not (Blender's string keys only find available ones)."""

    def __init__(self, coll):
        self.coll = coll

    def __getitem__(self, k):
        if isinstance(k, str):
            for s in self.coll:
                if s.identifier == k:
                    return s
            for s in self.coll:
                if s.name == k:
                    return s
            raise KeyError(k)
        return self.coll[k]

    def __iter__(self):
        return iter(self.coll)

    def __len__(self):
        return len(self.coll)


class NodeP:
    def __init__(self, n):
        object.__setattr__(self, "n", n)

    inputs = property(lambda self: Socks(self.n.inputs))
    outputs = property(lambda self: Socks(self.n.outputs))

    def __getattr__(self, k):
        return getattr(self.n, k)

    def __setattr__(self, k, v):
        setattr(self.n, k, v)


class Ctx:
    def __init__(self, nt):
        self.nt = nt
        self.attr = {}

    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        return NodeP(n)

    def link(self, a, b):
        self.nt.links.new(a, b)

    def input(self, name):
        """Attribute node of lab_<name> (outputs Color, Vector, Fac, Alpha)."""
        if name not in self.attr:
            self.attr[name] = self.node("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="lab_" + name)
        return self.attr[name]

    def xyz(self, name):
        sep = self.node("ShaderNodeSeparateXYZ")
        self.link(self.input(name).outputs["Vector"], sep.inputs[0])
        return sep.outputs

    def vec(self, name):
        return self.input(name).outputs["Vector"]

    def col(self, name):
        return self.input(name).outputs["Color"]

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        else:
            sock.default_value = v


# ================================================================================================= cases
CASES = []


def case(name, kind="field", a=None, b=None, c=None, rows=None, blended=False, slab=False, sun=None):
    def deco(fn):
        CASES.append({"name": name, "kind": kind, "build": fn, "a": a, "b": b, "c": c, "rows": rows,
                      "blended": blended, "slab": slab, "sun": sun})
        return fn
    return deco


# Lamp-lit closures (black world, one sun, shadows off): how EEVEE lights a Translucent BSDF in each thickness mode,
# and whether a Mix Shader's weight enters once or squared.
def lit_case(name, sun, slab=False):
    return case(name, kind="lit", sun=sun, slab=slab)


def _translucent(c, colour, weight=1.0):
    t = c.node("ShaderNodeBsdfTranslucent")
    t.inputs["Color"].default_value = (colour, colour, colour, 1.0)
    if weight == 1.0:
        return t.outputs[0]
    e = c.node("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 0.0
    m = c.node("ShaderNodeMixShader")
    m.inputs["Fac"].default_value = weight
    c.link(e.outputs[0], m.inputs[1])
    c.link(t.outputs[0], m.inputs[2])
    return m.outputs[0]


for _name, _sun, _slab, _col, _w in (("lit_translucent_c080", 30.0, False, 0.8, 1.0),
                                     ("lit_translucent_c050", 30.0, False, 0.5, 1.0),
                                     ("lit_translucent_mix050", 30.0, False, 0.8, 0.5),
                                     ("lit_translucent_back_c080", 150.0, False, 0.8, 1.0),
                                     ("lit_slab_back_c080", 150.0, True, 0.8, 1.0),
                                     ("lit_slab_back_c050", 150.0, True, 0.5, 1.0),
                                     ("lit_slab_back_mix050", 150.0, True, 0.8, 0.5)):
    lit_case(_name, _sun, _slab)(lambda c, col=_col, w=_w: _translucent(c, col, w))


@lit_case("lit_diffuse_c080_sun70", 70.0)
def _lit_diffuse70(c):
    d = c.node("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    return d.outputs[0]


def _sheen(c, rough):
    """Sheen alone (black base, no specular): if EEVEE adds it to the diffuse closure, it is lit as Lambert."""
    p = c.node("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    p.inputs["Specular IOR Level"].default_value = 0.0
    p.inputs["Sheen Weight"].default_value = 1.0
    p.inputs["Sheen Roughness"].default_value = rough
    return p.outputs[0]


for _r in (0.5, 1.0):
    for _sun in (30.0, 70.0):
        lit_case(f"lit_sheen_r{round(_r * 100):03d}_sun{int(_sun)}", _sun)(lambda c, r=_r: _sheen(c, r))


@lit_case("lit_diffuse_c080", 30.0)
def _lit_diffuse(c):
    d = c.node("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    return d.outputs[0]


def uniform(lo, hi):
    return lambda: RNG.uniform(lo, hi, (N * N, 3))


def plane(lo, hi, z=0.0):
    """Quad (i, j) -> (lo + (hi - lo) * (i + 0.5) / N, lo + (hi - lo) * (j + 0.5) / N, z)."""
    def f():
        t = lo + (hi - lo) * (np.arange(N) + 0.5) / N
        x, y = np.meshgrid(t, t)
        return np.stack([x.ravel(), y.ravel(), np.full(N * N, z)], axis=1)
    return f


def cube(lo, hi):
    """Like plane, z varying in bands of rows so every axis is exercised."""
    def f():
        p = plane(lo, hi)()
        p[:, 2] = lo + (hi - lo) * ((np.arange(N * N) // N) % 8 + 0.5) / 8
        return p
    return f


def special_values():
    """Edge values for maths: zeros, signs, integers, near-integers, large and small."""
    base = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 0.25, 1e-6, -1e-6, 7.5, -7.5, 1.5, -1.5,
                     0.999, 1.001, 100.0, -100.0, 0.1, 2.5, -2.5])
    def f():
        idx = RNG.integers(0, len(base), (N * N, 3))
        v = base[idx]
        rnd = RNG.uniform(-4.0, 4.0, (N * N, 3))
        mask = RNG.random((N * N, 3)) < 0.5
        return np.where(mask, v, rnd)
    return f


MATH_OPS = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD", "POWER", "LOGARITHM", "SQRT", "INVERSE_SQRT",
            "ABSOLUTE", "EXPONENT", "MINIMUM", "MAXIMUM", "LESS_THAN", "GREATER_THAN", "SIGN", "COMPARE", "SMOOTH_MIN",
            "SMOOTH_MAX", "ROUND", "FLOOR", "CEIL", "TRUNC", "FRACT", "MODULO", "FLOORED_MODULO", "WRAP", "SNAP",
            "PINGPONG", "SINE", "COSINE", "TANGENT", "ARCSINE", "ARCCOSINE", "ARCTANGENT", "ARCTAN2", "SINH", "COSH",
            "TANH", "RADIANS", "DEGREES"]

for _op in MATH_OPS:
    def _build(c, op=_op):
        m = c.node("ShaderNodeMath", operation=op)
        x = c.xyz("a")
        for i in range(3):
            c.link(x[i], m.inputs[i])
        # Encode so negative and large results survive the film: 0.5 + atan(v) / pi in (0, 1).
        enc = c.node("ShaderNodeMath", operation="ARCTANGENT")
        c.link(m.outputs[0], enc.inputs[0])
        s = c.node("ShaderNodeMath", operation="MULTIPLY_ADD")
        c.link(enc.outputs[0], s.inputs[0])
        s.inputs[1].default_value = 1.0 / math.pi
        s.inputs[2].default_value = 0.5
        return s.outputs[0]
    case(f"math_{_op.lower()}", a=special_values())(_build)

for _op in ["ADD", "MULTIPLY"]:
    def _build(c, op=_op):
        m = c.node("ShaderNodeMath", operation=op, use_clamp=True)
        x = c.xyz("a")
        c.link(x[0], m.inputs[0])
        c.link(x[1], m.inputs[1])
        return m.outputs[0]
    case(f"math_clamp_{_op.lower()}", a=uniform(-1.0, 2.0))(_build)

VMATH_OPS = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD", "CROSS_PRODUCT", "PROJECT", "REFLECT", "REFRACT",
             "FACEFORWARD", "DOT_PRODUCT", "DISTANCE", "LENGTH", "SCALE", "NORMALIZE", "SNAP", "FLOOR", "CEIL",
             "MODULO", "WRAP", "FRACTION", "ABSOLUTE", "MINIMUM", "MAXIMUM", "SINE", "COSINE", "TANGENT"]

for _op in VMATH_OPS:
    def _build(c, op=_op):
        m = c.node("ShaderNodeVectorMath", operation=op)
        c.link(c.vec("a"), m.inputs[0])
        c.link(c.vec("b"), m.inputs[1])
        c.link(c.vec("c"), m.inputs[2])
        c.link(c.xyz("c")[0], m.inputs["Scale"])
        out = m.outputs["Value"] if op in ("DOT_PRODUCT", "DISTANCE", "LENGTH") else m.outputs["Vector"]
        enc = c.node("ShaderNodeVectorMath", operation="SCALE")
        # 0.5 + v / (2 (1 + |v|)) per component keeps sign and magnitude in (0, 1)
        absn = c.node("ShaderNodeVectorMath", operation="ABSOLUTE")
        c.link(out, absn.inputs[0])
        add1 = c.node("ShaderNodeVectorMath", operation="ADD")
        c.link(absn.outputs[0], add1.inputs[0])
        add1.inputs[1].default_value = (1.0, 1.0, 1.0)
        div = c.node("ShaderNodeVectorMath", operation="DIVIDE")
        c.link(out, div.inputs[0])
        c.link(add1.outputs[0], div.inputs[1])
        c.link(div.outputs[0], enc.inputs[0])
        enc.inputs["Scale"].default_value = 0.5
        off = c.node("ShaderNodeVectorMath", operation="ADD")
        c.link(enc.outputs[0], off.inputs[0])
        off.inputs[1].default_value = (0.5, 0.5, 0.5)
        return off.outputs[0]
    case(f"vmath_{_op.lower()}", a=uniform(-3.0, 3.0), b=uniform(-3.0, 3.0), c=uniform(-2.0, 2.0))(_build)

BLENDS = ["MIX", "DARKEN", "MULTIPLY", "BURN", "LIGHTEN", "SCREEN", "DODGE", "ADD", "OVERLAY", "SOFT_LIGHT",
          "LINEAR_LIGHT", "DIFFERENCE", "EXCLUSION", "SUBTRACT", "DIVIDE", "HUE", "SATURATION", "COLOR", "VALUE"]

for _bt in BLENDS:
    for _clamp in (False, True):
        def _build(c, bt=_bt, clamp=_clamp):
            m = c.node("ShaderNodeMixRGB", blend_type=bt, use_clamp=clamp)
            c.link(c.xyz("c")[0], m.inputs["Fac"])
            c.link(c.col("a"), m.inputs["Color1"])
            c.link(c.col("b"), m.inputs["Color2"])
            return m.outputs[0]
        case(f"mixrgb_{_bt.lower()}{'_clamp' if _clamp else ''}", a=uniform(0.0, 1.2), b=uniform(0.0, 1.2),
             c=uniform(-0.2, 1.2))(_build)
    def _build(c, bt=_bt):
        m = c.node("ShaderNodeMix", data_type="RGBA", blend_type=bt, clamp_factor=False, clamp_result=False)
        c.link(c.xyz("c")[0], m.inputs["Factor"])
        c.link(c.col("a"), m.inputs[6])
        c.link(c.col("b"), m.inputs[7])
        return m.outputs[2]
    case(f"mix_{_bt.lower()}", a=uniform(0.0, 1.2), b=uniform(0.0, 1.2), c=uniform(-0.2, 1.2))(_build)


@case("mix_float", a=uniform(-2, 2), c=uniform(-0.5, 1.5))
def _mix_float(c):
    m = c.node("ShaderNodeMix", data_type="FLOAT", clamp_factor=True)
    x = c.xyz("a")
    c.link(c.xyz("c")[0], m.inputs["Factor"])
    c.link(x[0], m.inputs[2])
    c.link(x[1], m.inputs[3])
    return m.outputs[0]


@case("mix_vector_nonuniform", a=uniform(-1, 1), b=uniform(-1, 1), c=uniform(-0.5, 1.5))
def _mix_vec(c):
    m = c.node("ShaderNodeMix", data_type="VECTOR", factor_mode="NON_UNIFORM", clamp_factor=False)
    c.link(c.vec("c"), m.inputs[1])
    c.link(c.vec("a"), m.inputs[4])
    c.link(c.vec("b"), m.inputs[5])
    return m.outputs[1]


for _it in ("LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"):
    for _clamp in (False, True):
        def _build(c, it=_it, clamp=_clamp):
            m = c.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type=it, clamp=clamp)
            x = c.xyz("a")
            c.link(x[0], m.inputs["Value"])
            m.inputs["From Min"].default_value = 0.2
            m.inputs["From Max"].default_value = 0.8
            m.inputs["To Min"].default_value = 0.1
            m.inputs["To Max"].default_value = 0.9
            m.inputs["Steps"].default_value = 4.0
            return m.outputs["Result"]
        case(f"maprange_{_it.lower()}{'_clamp' if _clamp else ''}", a=uniform(-0.2, 1.2))(_build)


@case("maprange_vector_smooth", a=uniform(-0.2, 1.2))
def _mr_vec(c):
    m = c.node("ShaderNodeMapRange", data_type="FLOAT_VECTOR", interpolation_type="SMOOTHSTEP")
    c.link(c.vec("a"), m.inputs["Vector"])
    m.inputs[8].default_value = (0.9, 0.1, 0.3)     # From_Max_FLOAT3 (from min > max on x)
    return m.outputs["Vector"]


for _ct in ("MINMAX", "RANGE"):
    def _build(c, ct=_ct):
        m = c.node("ShaderNodeClamp", clamp_type=ct)
        x = c.xyz("a")
        for i in range(3):
            c.link(x[i], m.inputs[i])
        return m.outputs[0]
    case(f"clamp_{_ct.lower()}", a=uniform(-1, 1))(_build)

for _interp in ("LINEAR", "CONSTANT", "EASE", "B_SPLINE", "CARDINAL"):
    for _mode in ("RGB", "HSV", "HSL"):
        for _nstops in (2, 4):
            if _mode != "RGB" and _nstops == 2:
                continue
            def _build(c, interp=_interp, mode=_mode, nstops=_nstops):
                r = c.node("ShaderNodeValToRGB")
                cr = r.color_ramp
                cr.interpolation = interp
                cr.color_mode = mode
                if mode != "RGB":
                    cr.hue_interpolation = "NEAR"
                cols = [(0.9, 0.1, 0.05, 1.0), (0.1, 0.8, 0.2, 0.5), (0.05, 0.2, 0.9, 0.8), (0.95, 0.9, 0.1, 0.2)]
                pos = [0.15, 0.85] if nstops == 2 else [0.1, 0.35, 0.62, 0.9]
                cr.elements[0].position, cr.elements[0].color = pos[0], cols[0]
                cr.elements[1].position, cr.elements[1].color = pos[1], cols[1]
                for k in range(2, nstops):
                    e = cr.elements.new(pos[k])
                    e.color = cols[k]
                c.link(c.xyz("a")[0], r.inputs[0])
                return r.outputs["Color"]
            case(f"ramp_{_interp.lower()}_{_mode.lower()}_{_nstops}", a=plane(-0.1, 1.1))(_build)


@case("ramp_alpha", a=plane(-0.1, 1.1))
def _ramp_alpha(c):
    r = c.node("ShaderNodeValToRGB")
    cr = r.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.2, (1, 0, 0, 0.1)
    cr.elements[1].position, cr.elements[1].color = 0.7, (0, 0, 1, 0.9)
    c.link(c.xyz("a")[0], r.inputs[0])
    return r.outputs["Alpha"]


for _mode in ("RGB", "HSV", "HSL"):
    def _build(c, mode=_mode):
        s = c.node("ShaderNodeSeparateColor", mode=mode)
        c.link(c.col("a"), s.inputs[0])
        comb = c.node("ShaderNodeCombineXYZ")
        for i in range(3):
            c.link(s.outputs[i], comb.inputs[i])
        return comb.outputs[0]
    case(f"separate_{_mode.lower()}", a=uniform(0.0, 1.2))(_build)

    def _build(c, mode=_mode):
        s = c.node("ShaderNodeCombineColor", mode=mode)
        x = c.xyz("a")
        for i in range(3):
            c.link(x[i], s.inputs[i])
        return s.outputs[0]
    case(f"combine_{_mode.lower()}", a=uniform(0.0, 1.0))(_build)


@case("hue_sat", a=uniform(0.0, 1.2), b=uniform(0.0, 1.0), c=uniform(0.0, 1.5))
def _hue_sat(c):
    h = c.node("ShaderNodeHueSaturation")
    b, cc = c.xyz("b"), c.xyz("c")
    c.link(b[0], h.inputs["Hue"])
    c.link(cc[0], h.inputs["Saturation"])
    c.link(cc[1], h.inputs["Value"])
    c.link(b[1], h.inputs["Fac"])
    c.link(c.col("a"), h.inputs["Color"])
    return h.outputs[0]


@case("invert", a=uniform(0.0, 1.2), b=uniform(-0.2, 1.2))
def _invert(c):
    n = c.node("ShaderNodeInvert")
    c.link(c.xyz("b")[0], n.inputs["Fac"])
    c.link(c.col("a"), n.inputs["Color"])
    return n.outputs[0]


@case("gamma", a=uniform(-0.2, 1.5), b=uniform(0.2, 3.0))
def _gamma(c):
    n = c.node("ShaderNodeGamma")
    c.link(c.col("a"), n.inputs["Color"])
    c.link(c.xyz("b")[0], n.inputs["Gamma"])
    return n.outputs[0]


@case("bright_contrast", a=uniform(0.0, 1.2), b=uniform(-0.5, 0.5))
def _bc(c):
    n = c.node("ShaderNodeBrightContrast")
    b = c.xyz("b")
    c.link(c.col("a"), n.inputs["Color"])
    c.link(b[0], n.inputs["Bright"])
    c.link(b[1], n.inputs["Contrast"])
    return n.outputs[0]


@case("rgb_to_bw", a=uniform(0.0, 1.0))
def _bw(c):
    n = c.node("ShaderNodeRGBToBW")
    c.link(c.col("a"), n.inputs[0])
    return n.outputs[0]


@case("color_to_float", a=uniform(0.0, 1.0))
def _c2f(c):
    m = c.node("ShaderNodeMath", operation="ADD")
    c.link(c.col("a"), m.inputs[0])
    return m.outputs[0]


@case("vector_to_float", a=uniform(0.0, 1.0))
def _v2f(c):
    m = c.node("ShaderNodeMath", operation="ADD")
    c.link(c.vec("a"), m.inputs[0])
    return m.outputs[0]


for _vt in ("POINT", "TEXTURE", "VECTOR", "NORMAL"):
    def _build(c, vt=_vt):
        m = c.node("ShaderNodeMapping", vector_type=vt)
        c.link(c.vec("a"), m.inputs["Vector"])
        m.inputs["Location"].default_value = (0.3, -0.2, 0.1)
        m.inputs["Rotation"].default_value = (0.4, -0.7, 1.1)
        m.inputs["Scale"].default_value = (1.5, 0.7, -1.2)
        enc = c.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        c.link(m.outputs[0], enc.inputs[0])
        enc.inputs[1].default_value = (0.25, 0.25, 0.25)
        enc.inputs[2].default_value = (0.5, 0.5, 0.5)
        return enc.outputs[0]
    case(f"mapping_{_vt.lower()}", a=uniform(-1, 1))(_build)

for _rt in ("AXIS_ANGLE", "X_AXIS", "Y_AXIS", "Z_AXIS", "EULER_XYZ"):
    for _inv in (False, True):
        def _build(c, rt=_rt, inv=_inv):
            m = c.node("ShaderNodeVectorRotate", rotation_type=rt, invert=inv)
            c.link(c.vec("a"), m.inputs["Vector"])
            m.inputs["Center"].default_value = (0.1, 0.2, -0.1)
            c.link(c.vec("b"), m.inputs["Axis"])
            c.link(c.xyz("c")[0], m.inputs["Angle"])
            c.link(c.vec("c"), m.inputs["Rotation"])
            enc = c.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
            c.link(m.outputs[0], enc.inputs[0])
            enc.inputs[1].default_value = (0.25, 0.25, 0.25)
            enc.inputs[2].default_value = (0.5, 0.5, 0.5)
            return enc.outputs[0]
        case(f"vrotate_{_rt.lower()}{'_inv' if _inv else ''}", a=uniform(-1, 1), b=uniform(-1, 1),
             c=uniform(-3, 3))(_build)

# ---------------------------------------------------------------------------------------------------- textures
NOISE_TYPES = ["FBM", "MULTIFRACTAL", "RIDGED_MULTIFRACTAL", "HYBRID_MULTIFRACTAL", "HETERO_TERRAIN"]

for _dims in ("1D", "2D", "3D", "4D"):
    for _nt in NOISE_TYPES:
        for _variant in ("plain", "detail", "distort"):
            if _nt != "FBM" and _variant == "distort":
                continue
            def _build(c, dims=_dims, nt=_nt, variant=_variant):
                n = c.node("ShaderNodeTexNoise", noise_dimensions=dims, noise_type=nt)
                c.link(c.vec("a"), n.inputs["Vector"])
                c.link(c.xyz("b")[0], n.inputs["W"])
                n.inputs["Scale"].default_value = 2.3
                n.inputs["Detail"].default_value = {"plain": 0.0, "detail": 3.6, "distort": 2.0}[variant]
                n.inputs["Roughness"].default_value = 0.6
                n.inputs["Lacunarity"].default_value = 2.2
                n.inputs["Offset"].default_value = 0.7
                n.inputs["Gain"].default_value = 1.3
                n.inputs["Distortion"].default_value = 1.7 if variant == "distort" else 0.0
                if nt != "FBM":
                    # unbounded outputs: squash into 0..1
                    m = c.node("ShaderNodeMath", operation="ARCTANGENT")
                    c.link(n.outputs["Fac"], m.inputs[0])
                    s = c.node("ShaderNodeMath", operation="MULTIPLY_ADD")
                    c.link(m.outputs[0], s.inputs[0])
                    s.inputs[1].default_value = 1.0 / math.pi
                    s.inputs[2].default_value = 0.5
                    return s.outputs[0]
                return n.outputs["Color"] if variant != "plain" else n.outputs["Fac"]
            case(f"noise_{_dims.lower()}_{_nt.lower()}_{_variant}", a=cube(-2.0, 2.0), b=uniform(-1.0, 1.0))(_build)


@case("noise_unnormalized", a=cube(-2.0, 2.0))
def _noise_unnorm(c):
    n = c.node("ShaderNodeTexNoise", noise_dimensions="3D", normalize=False)
    c.link(c.vec("a"), n.inputs["Vector"])
    n.inputs["Detail"].default_value = 2.5
    m = c.node("ShaderNodeMath", operation="MULTIPLY_ADD")
    c.link(n.outputs["Fac"], m.inputs[0])
    m.inputs[1].default_value = 0.3
    m.inputs[2].default_value = 0.5
    return m.outputs[0]


VOR_FEATURES = ["F1", "F2", "SMOOTH_F1", "DISTANCE_TO_EDGE", "N_SPHERE_RADIUS"]
VOR_METRICS = ["EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI"]

for _dims in ("1D", "2D", "3D", "4D"):
    for _f in VOR_FEATURES:
        for _metric in (VOR_METRICS if _f in ("F1", "F2", "SMOOTH_F1") else ["EUCLIDEAN"]):
            for _out in (["Distance", "Color", "Position"] if _f in ("F1", "F2", "SMOOTH_F1") else
                         ["Distance" if _f == "DISTANCE_TO_EDGE" else "Radius"]):
                if _metric != "EUCLIDEAN" and _out != "Distance":
                    continue
                def _build(c, dims=_dims, f=_f, metric=_metric, out=_out):
                    n = c.node("ShaderNodeTexVoronoi", voronoi_dimensions=dims, feature=f, distance=metric)
                    c.link(c.vec("a"), n.inputs["Vector"])
                    c.link(c.xyz("b")[0], n.inputs["W"])
                    n.inputs["Scale"].default_value = 3.1
                    n.inputs["Randomness"].default_value = 0.85
                    n.inputs["Smoothness"].default_value = 0.6
                    n.inputs["Exponent"].default_value = 1.7
                    sock = n.outputs["W"] if (out == "Position" and dims == "1D") else n.outputs[out]
                    if out == "Position":
                        m = c.node("ShaderNodeVectorMath", operation="FRACTION")
                        c.link(sock, m.inputs[0])
                        return m.outputs[0]
                    return sock
                case(f"voronoi_{_dims.lower()}_{_f.lower()}_{_metric.lower()}_{_out.lower()}", a=cube(-1.5, 1.5),
                     b=uniform(-1.0, 1.0))(_build)


@case("voronoi_3d_f1_fractal", a=cube(-1.5, 1.5))
def _vor_fractal(c):
    n = c.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1", normalize=True)
    c.link(c.vec("a"), n.inputs["Vector"])
    n.inputs["Scale"].default_value = 2.0
    n.inputs["Detail"].default_value = 2.5
    n.inputs["Roughness"].default_value = 0.6
    return n.outputs["Distance"]


@case("voronoi_3d_edge_fractal", a=cube(-1.5, 1.5))
def _vor_edge_fractal(c):
    n = c.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE", normalize=True)
    c.link(c.vec("a"), n.inputs["Vector"])
    n.inputs["Scale"].default_value = 2.0
    n.inputs["Detail"].default_value = 1.5
    return n.outputs["Distance"]


for _wt in ("BANDS", "RINGS"):
    for _dir in (["X", "Y", "Z", "DIAGONAL"] if _wt == "BANDS" else ["X", "Y", "Z", "SPHERICAL"]):
        for _prof in ("SIN", "SAW", "TRI"):
            for _dist in (0.0, 2.5):
                if _dist and _prof != "SIN":
                    continue
                def _build(c, wt=_wt, d=_dir, prof=_prof, dist=_dist):
                    n = c.node("ShaderNodeTexWave", wave_type=wt, wave_profile=prof)
                    if wt == "BANDS":
                        n.bands_direction = d
                    else:
                        n.rings_direction = d
                    c.link(c.vec("a"), n.inputs["Vector"])
                    n.inputs["Scale"].default_value = 1.3
                    n.inputs["Distortion"].default_value = dist
                    n.inputs["Detail"].default_value = 2.4
                    n.inputs["Detail Scale"].default_value = 1.6
                    n.inputs["Phase Offset"].default_value = 0.4
                    return n.outputs["Fac"]
                case(f"wave_{_wt.lower()}_{_dir.lower()}_{_prof.lower()}{'_distort' if _dist else ''}",
                     a=cube(-1.0, 1.0))(_build)

for _gt in ("LINEAR", "QUADRATIC", "EASING", "DIAGONAL", "RADIAL", "QUADRATIC_SPHERE", "SPHERICAL"):
    def _build(c, gt=_gt):
        n = c.node("ShaderNodeTexGradient", gradient_type=gt)
        c.link(c.vec("a"), n.inputs["Vector"])
        return n.outputs["Fac"]
    case(f"gradient_{_gt.lower()}", a=cube(-1.2, 1.2))(_build)


@case("checker", a=cube(-2.0, 2.0))
def _checker(c):
    n = c.node("ShaderNodeTexChecker")
    c.link(c.vec("a"), n.inputs["Vector"])
    n.inputs["Scale"].default_value = 3.0
    n.inputs["Color1"].default_value = (0.9, 0.2, 0.1, 1.0)
    n.inputs["Color2"].default_value = (0.1, 0.3, 0.8, 1.0)
    return n.outputs["Color"]


for _sm in (0.0, 0.4):
    def _build(c, sm=_sm):
        n = c.node("ShaderNodeTexBrick", offset=0.4, offset_frequency=2, squash=0.8, squash_frequency=3)
        c.link(c.vec("a"), n.inputs["Vector"])
        n.inputs["Scale"].default_value = 3.0
        n.inputs["Mortar Size"].default_value = 0.04
        n.inputs["Mortar Smooth"].default_value = sm
        n.inputs["Bias"].default_value = 0.1
        n.inputs["Color1"].default_value = (0.9, 0.3, 0.1, 1.0)
        n.inputs["Color2"].default_value = (0.4, 0.6, 0.2, 1.0)
        n.inputs["Mortar"].default_value = (0.1, 0.1, 0.1, 1.0)
        return n.outputs["Color"]
    case(f"brick{'_smooth' if _sm else ''}", a=cube(-2.0, 2.0))(_build)

for _depth in (0, 1, 2, 5, 10):
    def _build(c, depth=_depth):
        n = c.node("ShaderNodeTexMagic", turbulence_depth=depth)
        c.link(c.vec("a"), n.inputs["Vector"])
        n.inputs["Scale"].default_value = 2.5
        n.inputs["Distortion"].default_value = 1.3
        return n.outputs["Color"]
    case(f"magic_{_depth}", a=cube(-1.0, 1.0))(_build)

for _dims in ("1D", "2D", "3D", "4D"):
    def _build(c, dims=_dims):
        n = c.node("ShaderNodeTexWhiteNoise", noise_dimensions=dims)
        c.link(c.vec("a"), n.inputs["Vector"])
        c.link(c.xyz("b")[0], n.inputs["W"])
        return n.outputs["Color"]
    case(f"white_noise_{_dims.lower()}", a=cube(-3.0, 3.0), b=uniform(-2.0, 2.0))(_build)

# ---------------------------------------------------------------------------------------------------- closures
ROWS = np.linspace(0.02, 1.0, N)          # N.I per row (cos of the tilt)


def tilted_normals():
    out = []
    for j in range(N):
        cos_t = ROWS[j]
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        out += [(sin_t, 0.0, cos_t)] * N
    return out


def rough_cols():
    """lab_a.x: roughness by column 0..1; lab_a.y, z: 0.5."""
    def f():
        r = (np.arange(N) + 0.5) / N
        a = np.zeros((N * N, 3))
        a[:, 0] = np.tile(r, N)
        a[:, 1:] = 0.5
        return a
    return f


def closure_case(name, **kw):
    return case(name, kind="closure", a=kw.pop("a", rough_cols()), rows=ROWS.tolist(), **kw)


@closure_case("closure_diffuse")
def _cl_diffuse(c):
    n = c.node("ShaderNodeBsdfDiffuse")
    n.inputs["Color"].default_value = (0.8, 0.5, 0.2, 1.0)
    return n.outputs[0]


for _dist in ("GGX", "MULTI_GGX"):
    def _build(c, dist=_dist):
        n = c.node("ShaderNodeBsdfAnisotropic", distribution=dist)
        n.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        c.link(c.xyz("a")[0], n.inputs["Roughness"])
        return n.outputs[0]
    closure_case(f"closure_glossy_{_dist.lower()}")(_build)


def principled(c, **inputs):
    n = c.node("ShaderNodeBsdfPrincipled")
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    c.link(c.xyz("a")[0], n.inputs["Roughness"])
    return n


@closure_case("closure_principled_default")
def _cl_p0(c):
    return principled(c, **{"Base Color": (0.8, 0.8, 0.8, 1.0)}).outputs[0]


@closure_case("closure_principled_black")
def _cl_p1(c):
    return principled(c, **{"Base Color": (0.0, 0.0, 0.0, 1.0)}).outputs[0]


@closure_case("closure_principled_spec1")
def _cl_p2(c):
    return principled(c, **{"Base Color": (0.0, 0.0, 0.0, 1.0), "Specular IOR Level": 1.0}).outputs[0]


@closure_case("closure_principled_metal")
def _cl_p3(c):
    return principled(c, **{"Base Color": (0.9, 0.6, 0.3, 1.0), "Metallic": 1.0}).outputs[0]


@closure_case("closure_principled_metal_tint")
def _cl_p4(c):
    return principled(c, **{"Base Color": (0.9, 0.6, 0.3, 1.0), "Metallic": 1.0,
                            "Specular Tint": (0.5, 0.8, 1.0, 1.0)}).outputs[0]


@closure_case("closure_principled_coat")
def _cl_p5(c):
    return principled(c, **{"Base Color": (0.2, 0.4, 0.8, 1.0), "Coat Weight": 1.0, "Coat Roughness": 0.1}).outputs[0]


@closure_case("closure_principled_coat_tint")
def _cl_p6(c):
    return principled(c, **{"Base Color": (0.8, 0.8, 0.8, 1.0), "Coat Weight": 0.7,
                            "Coat Tint": (0.9, 0.5, 0.3, 1.0)}).outputs[0]


for _sr in (0.0, 0.05, 0.1, 0.175, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
    def _build(c, sr=_sr):
        return principled(c, **{"Base Color": (0.0, 0.0, 0.0, 1.0), "Specular IOR Level": 0.0, "Sheen Weight": 1.0,
                                "Sheen Roughness": sr}).outputs[0]
    closure_case(f"closure_sheen_r{round(_sr * 1000):04d}")(_build)


@closure_case("closure_principled_sheen_mix")
def _cl_p7(c):
    return principled(c, **{"Base Color": (0.6, 0.3, 0.2, 1.0), "Sheen Weight": 0.6, "Sheen Roughness": 0.4,
                            "Sheen Tint": (0.8, 0.9, 1.0, 1.0)}).outputs[0]


@closure_case("closure_principled_transmission")
def _cl_p8(c):
    return principled(c, **{"Base Color": (0.9, 0.9, 0.9, 1.0), "Transmission Weight": 1.0}).outputs[0]


@closure_case("closure_principled_emission")
def _cl_p9(c):
    return principled(c, **{"Base Color": (0.3, 0.3, 0.3, 1.0), "Emission Color": (1.0, 0.5, 0.2, 1.0),
                            "Emission Strength": 0.7}).outputs[0]


@closure_case("closure_principled_alpha", blended=True)
def _cl_p10(c):
    n = principled(c, **{"Base Color": (0.7, 0.2, 0.2, 1.0)})
    c.link(c.xyz("a")[1], n.inputs["Alpha"])
    return n.outputs[0]


@closure_case("closure_translucent")
def _cl_tr(c):
    n = c.node("ShaderNodeBsdfTranslucent")
    n.inputs["Color"].default_value = (0.3, 0.6, 0.9, 1.0)
    return n.outputs[0]


@closure_case("closure_mix_diffuse_glossy")
def _cl_mix(c):
    d = c.node("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.8, 0.4, 0.2, 1.0)
    g = c.node("ShaderNodeBsdfAnisotropic")
    g.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    c.link(c.xyz("a")[0], g.inputs["Roughness"])
    m = c.node("ShaderNodeMixShader")
    m.inputs["Fac"].default_value = 0.3
    c.link(d.outputs[0], m.inputs[1])
    c.link(g.outputs[0], m.inputs[2])
    return m.outputs[0]


@closure_case("closure_add_emission_diffuse")
def _cl_add(c):
    d = c.node("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.2, 0.4, 0.6, 1.0)
    e = c.node("ShaderNodeEmission")
    e.inputs["Color"].default_value = (0.5, 0.2, 0.1, 1.0)
    e.inputs["Strength"].default_value = 1.5
    a = c.node("ShaderNodeAddShader")
    c.link(d.outputs[0], a.inputs[0])
    c.link(e.outputs[0], a.inputs[1])
    return a.outputs[0]


@closure_case("closure_glass")
def _cl_glass(c):
    n = c.node("ShaderNodeBsdfGlass")
    n.inputs["Color"].default_value = (0.9, 0.8, 0.7, 1.0)
    c.link(c.xyz("a")[0], n.inputs["Roughness"])
    return n.outputs[0]


@closure_case("closure_refraction")
def _cl_refr(c):
    n = c.node("ShaderNodeBsdfRefraction")
    n.inputs["Color"].default_value = (0.9, 0.8, 0.7, 1.0)
    c.link(c.xyz("a")[0], n.inputs["Roughness"])
    return n.outputs[0]


@closure_case("closure_mmd_shader_dev")
def _cl_mmd(c):
    """The MMD shader's lobes: diffuse roughness 0 mixed 98 / 2 with a glossy of roughness 1/Reflect."""
    d = c.node("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.9, 0.8, 0.7, 1.0)
    d.inputs["Roughness"].default_value = 0.0
    g = c.node("ShaderNodeBsdfAnisotropic", distribution="MULTI_GGX")
    g.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    c.link(c.xyz("a")[0], g.inputs["Roughness"])
    m = c.node("ShaderNodeMixShader")
    m.inputs["Fac"].default_value = 0.02
    c.link(d.outputs[0], m.inputs[1])
    c.link(g.outputs[0], m.inputs[2])
    return m.outputs[0]


# ---------------------------------------------------------------------------------------------------- shading geometry
def shade_case(name, **kw):
    """Emission of a value that depends on the shading point: the normal tilts per row as for closures."""
    return case(name, kind="shade", a=kw.pop("a", rough_cols()), rows=ROWS.tolist(), **kw)


def encode(c, sock, scale=0.5, offset=0.5):
    """v * scale + offset per component (vectors that may be negative)."""
    m = c.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
    c.link(sock, m.inputs[0])
    m.inputs[1].default_value = (scale, scale, scale)
    m.inputs[2].default_value = (offset, offset, offset)
    return m.outputs[0]


for _out in ("Fresnel", "Facing"):
    def _build(c, out=_out):
        n = c.node("ShaderNodeLayerWeight")
        c.link(c.xyz("a")[0], n.inputs["Blend"])
        return n.outputs[out]
    shade_case(f"layer_weight_{_out.lower()}")(_build)


@shade_case("fresnel")
def _fresnel(c):
    n = c.node("ShaderNodeFresnel")
    m = c.node("ShaderNodeMath", operation="MULTIPLY_ADD")
    c.link(c.xyz("a")[0], m.inputs[0])
    m.inputs[1].default_value = 2.0
    m.inputs[2].default_value = 1.0
    c.link(m.outputs[0], n.inputs["IOR"])
    return n.outputs[0]


for _out in ("Normal", "Incoming", "True Normal", "Position"):
    def _build(c, out=_out):
        g = c.node("ShaderNodeNewGeometry")
        return encode(c, g.outputs[out], 0.02 if out == "Position" else 0.5)
    shade_case(f"geometry_{_out.lower().replace(' ', '_')}")(_build)

for _out in ("Normal", "UV", "Object", "Camera", "Window", "Reflection"):
    def _build(c, out=_out):
        t = c.node("ShaderNodeTexCoord")
        scale = {"Object": 0.02, "Camera": 0.02, "UV": 1.0, "Window": 1.0}.get(out, 0.5)
        offset = 0.0 if out in ("UV", "Window") else 0.5
        return encode(c, t.outputs[out], scale, offset)
    shade_case(f"texcoord_{_out.lower()}")(_build)

for _vt, _frm, _to in (("NORMAL", "OBJECT", "CAMERA"), ("POINT", "WORLD", "CAMERA"), ("VECTOR", "CAMERA", "WORLD"),
                       ("NORMAL", "CAMERA", "OBJECT"), ("POINT", "CAMERA", "WORLD")):
    def _build(c, vt=_vt, frm=_frm, to=_to):
        t = c.node("ShaderNodeTexCoord")
        v = c.node("ShaderNodeVectorTransform", vector_type=vt, convert_from=frm, convert_to=to)
        c.link(t.outputs["Normal"] if vt != "POINT" else t.outputs["Object"], v.inputs[0])
        return encode(c, v.outputs[0], 0.5 if vt != "POINT" else 0.02)
    shade_case(f"vtransform_{_vt.lower()}_{_frm.lower()}_{_to.lower()}")(_build)


@shade_case("mmd_sphere_uv")
def _mmd_sphere(c):
    """MMDTexUV's sphere coordinates: Normal -> Vector Transform (object -> camera) -> Mapping (0.5 + 0.5 xy)."""
    t = c.node("ShaderNodeTexCoord")
    v = c.node("ShaderNodeVectorTransform", vector_type="NORMAL", convert_from="OBJECT", convert_to="CAMERA")
    c.link(t.outputs["Normal"], v.inputs[0])
    m = c.node("ShaderNodeMapping", vector_type="POINT")
    m.inputs["Location"].default_value = (0.5, 0.5, 0.0)
    m.inputs["Scale"].default_value = (0.5, 0.5, 1.0)
    c.link(v.outputs[0], m.inputs["Vector"])
    return m.outputs[0]


for _out in ("View Vector", "View Z Depth", "View Distance"):
    def _build(c, out=_out):
        n = c.node("ShaderNodeCameraData")
        if out == "View Vector":
            return encode(c, n.outputs[out])
        m = c.node("ShaderNodeMath", operation="MULTIPLY")
        c.link(n.outputs[out], m.inputs[0])
        m.inputs[1].default_value = 0.05
        return m.outputs[0]
    shade_case(f"camera_data_{_out.lower().replace(' ', '_')}")(_build)


@shade_case("light_path_camera")
def _light_path(c):
    n = c.node("ShaderNodeLightPath")
    comb = c.node("ShaderNodeCombineXYZ")
    c.link(n.outputs["Is Camera Ray"], comb.inputs[0])
    c.link(n.outputs["Is Shadow Ray"], comb.inputs[1])
    c.link(n.outputs["Is Glossy Ray"], comb.inputs[2])
    return comb.outputs[0]


@shade_case("normal_map_object", b=uniform(0.0, 1.0))
def _nmap_obj(c):
    n = c.node("ShaderNodeNormalMap", space="OBJECT")
    c.link(c.col("b"), n.inputs["Color"])
    c.link(c.xyz("a")[0], n.inputs["Strength"])
    return encode(c, n.outputs[0])


@shade_case("normal_map_world", b=uniform(0.0, 1.0))
def _nmap_world(c):
    n = c.node("ShaderNodeNormalMap", space="WORLD")
    c.link(c.col("b"), n.inputs["Color"])
    n.inputs["Strength"].default_value = 0.7
    return encode(c, n.outputs[0])


# ---------------------------------------------------------------------------------------------------- image textures
IMG = {"w": 16, "h": 12}


def lab_image(name, colorspace="sRGB", alpha_mode="STRAIGHT"):
    """A 16 x 12 RGBA PNG of seeded random texels (alpha too), saved and loaded back as a file image."""
    w, h = IMG["w"], IMG["h"]
    rng = np.random.default_rng(11)
    px = rng.uniform(0.0, 1.0, (h, w, 4)).astype(np.float32)
    px[..., 3] = np.round(px[..., 3] * 4) / 4          # a few exact alpha levels, 0 and 1 among them
    im = bpy.data.images.new(name + "_src", w, h, alpha=True)
    im.pixels.foreach_set(px.ravel())
    path = os.path.join(IMG["dir"], name + ".png")
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
    im = bpy.data.images.load(path, check_existing=False)
    im.name = name
    im.colorspace_settings.name = colorspace
    im.alpha_mode = alpha_mode
    return im


def image_case(name, colorspace="sRGB", alpha_mode="STRAIGHT", out="Color", interp="Closest", ext="REPEAT",
               projection="FLAT", lo=0.0, hi=1.0, alpha_used=False, a=None):
    def build(c):
        t = c.node("ShaderNodeTexImage", interpolation=interp, extension=ext, projection=projection)
        t.image = lab_image("lab_" + name, colorspace, alpha_mode)
        if projection == "BOX":
            t.projection_blend = 0.3
        c.link(c.vec("a"), t.inputs["Vector"])
        if out == "Alpha":
            return t.outputs["Alpha"]
        if alpha_used:
            # the Alpha output feeds the tree (Blender then gives straight colour) without changing the result
            m = c.node("ShaderNodeMixRGB", blend_type="MULTIPLY")
            m.inputs["Fac"].default_value = 0.0
            c.link(t.outputs["Color"], m.inputs["Color1"])
            c.link(t.outputs["Alpha"], m.inputs["Color2"])
            return m.outputs[0]
        return t.outputs["Color"]
    texel = plane(lo, hi)
    case("img_" + name, a=a or texel)(build)


image_case("color_premultiplied")
image_case("color_straight", alpha_used=True)
image_case("alpha", out="Alpha")
image_case("noncolor", colorspace="Non-Color")
image_case("alpha_none", alpha_mode="NONE", alpha_used=True)
image_case("alpha_none_out", alpha_mode="NONE", out="Alpha")
image_case("channel_packed", alpha_mode="CHANNEL_PACKED")
image_case("premul", alpha_mode="PREMUL")
image_case("premul_straight", alpha_mode="PREMUL", alpha_used=True)
for _ext in ("REPEAT", "EXTEND", "CLIP", "MIRROR"):
    image_case(f"ext_{_ext.lower()}", ext=_ext, lo=-1.3, hi=2.1, alpha_used=True)
image_case("linear", interp="Linear", lo=0.02, hi=0.98, alpha_used=True)
image_case("linear_clip", interp="Linear", ext="CLIP", lo=-0.2, hi=1.2, alpha_used=True)
for _proj in ("SPHERE", "TUBE"):
    image_case(f"proj_{_proj.lower()}", projection=_proj, alpha_used=True, a=uniform(-1.0, 2.0))


# Bump: heights from the continuous position, read at each quad's centre pixel (derivatives one render pixel apart).
def bump_height(c, kind):
    g = c.node("ShaderNodeNewGeometry")
    if kind == "noise":
        n = c.node("ShaderNodeTexNoise")
        c.link(g.outputs["Position"], n.inputs["Vector"])
        n.inputs["Scale"].default_value = 0.4
        n.inputs["Detail"].default_value = 3.0
        return n.outputs["Fac"]
    sep = c.node("ShaderNodeSeparateXYZ")
    c.link(g.outputs["Position"], sep.inputs[0])
    s = c.node("ShaderNodeMath", operation="SINE")
    m = c.node("ShaderNodeMath", operation="MULTIPLY")
    c.link(sep.outputs[0], m.inputs[0])
    m.inputs[1].default_value = 0.37
    c.link(m.outputs[0], s.inputs[0])
    co = c.node("ShaderNodeMath", operation="COSINE")
    m2 = c.node("ShaderNodeMath", operation="MULTIPLY")
    c.link(sep.outputs[1], m2.inputs[0])
    m2.inputs[1].default_value = 0.23
    c.link(m2.outputs[0], co.inputs[0])
    p = c.node("ShaderNodeMath", operation="MULTIPLY")
    c.link(s.outputs[0], p.inputs[0])
    c.link(co.outputs[0], p.inputs[1])
    return p.outputs[0]


for _hk in ("waves", "noise"):
    for _inv in (False, True):
        def _build(c, hk=_hk, inv=_inv):
            b = c.node("ShaderNodeBump", invert=inv)
            b.inputs["Strength"].default_value = 0.8
            b.inputs["Distance"].default_value = 0.6
            c.link(bump_height(c, hk), b.inputs["Height"])
            return encode(c, b.outputs[0])
        case(f"bump_{_hk}{'_invert' if _inv else ''}")(_build)


@shade_case("bump_tilted")
def _bump_tilted(c):
    b = c.node("ShaderNodeBump")
    b.inputs["Strength"].default_value = 1.0
    b.inputs["Distance"].default_value = 0.3
    c.link(bump_height(c, "waves"), b.inputs["Height"])
    return encode(c, b.outputs[0])


# ================================================================================================= running
def run_case(cs, out_dir):
    kind = cs["kind"]
    sc = reset(kind, cs["sun"])
    ob = grid(sc, normals=tilted_normals() if kind in ("closure", "shade") else None)
    zero = np.zeros((N * N, 3))
    a = cs["a"]() if cs["a"] else zero
    b = cs["b"]() if cs["b"] else zero
    cc = cs["c"]() if cs["c"] else zero
    set_inputs(ob, a, b, cc)
    m, nt, out = material(ob)
    if cs["blended"]:
        m.surface_render_method = "BLENDED"
    if cs["slab"]:
        m.thickness_mode = "SLAB"
    ctx = Ctx(nt)
    sock = cs["build"](ctx)
    if kind in ("closure", "lit"):
        nt.links.new(sock, out.inputs["Surface"])
    else:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(sock, em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    px = render(sc, os.path.join(out_dir, cs["name"] + ".exr"))
    ref = centres(px)
    used = {"groups": set(), "images": set(), "objects": set()}
    tree = BL.graph(nt, used)
    images = {name: BL._image(bpy.data.images[name], out_dir, i) for i, name in enumerate(sorted(used["images"]))}
    doc = {"name": cs["name"], "kind": kind, "n": N, "blender": bpy.app.version_string,
           "material": {"settings": BL._props(m, skip=BL._skip("id")), "tree": tree, "anim": []},
           "groups": {}, "images": images, "rows": cs["rows"]}
    with open(os.path.join(out_dir, cs["name"] + ".json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    np.concatenate([a, b, cc], axis=1).astype(np.float32).tofile(os.path.join(out_dir, cs["name"] + ".in.bin"))
    ref.astype(np.float32).tofile(os.path.join(out_dir, cs["name"] + ".ref.bin"))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out_dir = argv[0]
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1].split(",")
    os.makedirs(out_dir, exist_ok=True)
    IMG["dir"] = os.path.abspath(out_dir)
    todo = [cs for cs in CASES if only is None or any(fnmatch.fnmatch(cs["name"], p) for p in only)]
    for k, cs in enumerate(todo):
        run_case(cs, out_dir)
        print(f"LAB {k + 1}/{len(todo)} {cs['name']}", flush=True)
    print("LAB DONE", len(todo), flush=True)


main()
