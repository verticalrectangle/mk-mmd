"""Blender side of the night sets: numpy Mesh -> object, a small shader-node builder, distance haze, baked lamp light,
glow sprites and lights. Colours come from the project palette (slot names or '#hex'), never hard-coded."""
import math

import bpy
import numpy as np

from ....core.palette import linear, mix, resolve

# ---------------------------------------------------------------------------------------------------- colours


def hexof(palette, c):
    """Slot name or '#rrggbb' -> '#rrggbb'. A 'slot:slot:t' string mixes two slots (t = share of the second)."""
    if isinstance(c, str) and c.count(":") == 2:
        a, b, t = c.split(":")
        return mix(resolve(a, palette), resolve(b, palette), float(t))
    return resolve(c, palette)


def rgb(palette, c, k=1.0):
    """Linear RGB tuple of a slot or hex, times k."""
    return tuple(x * k for x in linear(hexof(palette, c)))


def haze_hex(palette):
    """The colour far things fade into: the horizon of the night sky (iris over hl_med)."""
    return mix(palette["hl_med"], palette["iris"], 0.30)


def clean(name):
    """Remove datablocks left by an earlier build of the same name (a rebuild in the same session)."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights):
        item = coll.get(name)
        if item is not None:
            coll.remove(item)


# ---------------------------------------------------------------------------------------------------- objects


def to_object(mesh, name, coll, parent, materials=(), shade_smooth=None, shadow=True, glossy=True, camera=True):
    """A `nightgeo.Mesh` as an object in `coll` under `parent`. Attributes become mesh attributes (1 value: FLOAT, 3 or
    4: FLOAT_COLOR); material index i uses materials[i]."""
    a = mesh.arrays()
    V, Q, T = a["V"], a["Q"], a["T"]
    me = bpy.data.meshes.new(name)
    nq, nt = len(Q), len(T)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    loops = np.concatenate([Q.ravel(), T.ravel()]).astype(np.int32)
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    me.polygons.add(nq + nt)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.concatenate([np.full(nq, 4), np.full(nt, 3)]).astype(np.int32))
    me.polygons.foreach_set("material_index", np.concatenate([a["QM"], a["TM"]]).astype(np.int32))
    smooth = np.concatenate([a["QS"], a["TS"]]) if shade_smooth is None else np.full(nq + nt, bool(shade_smooth))
    me.polygons.foreach_set("use_smooth", smooth)
    me.update(calc_edges=True)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", a["UV"][loops].astype(np.float32).ravel())
    for k, v in a["attrs"].items():
        if v.ndim == 1:
            at = me.attributes.new(k, "FLOAT", "POINT")
            at.data.foreach_set("value", v.astype(np.float32))
        else:
            if v.shape[1] == 3:
                v = np.concatenate([v, np.ones((len(v), 1), np.float32)], 1)
            at = me.attributes.new(k, "FLOAT_COLOR", "POINT")
            at.data.foreach_set("color", v.astype(np.float32).ravel())
    for m in materials:
        me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.visible_shadow, o.visible_glossy, o.visible_camera = shadow, glossy, camera
    return o


def empty(name, coll, parent, loc=(0.0, 0.0, 0.0), size=0.5, kind="PLAIN_AXES"):
    o = bpy.data.objects.new(name, None)
    o.empty_display_type, o.empty_display_size = kind, size
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    return o


def make_light(name, coll, parent, kind, loc, power, colour, direction=None, radius=0.2, shadow=False, reach=None,
               spot_deg=140.0, blend=0.8, size=None, specular=1.0):
    """A light object. direction: unit vector the light looks along (spot, area, sun); size (x, y) for AREA."""
    ld = bpy.data.lights.new(name, kind)
    ld.energy = power
    ld.color = colour
    ld.use_shadow = shadow
    ld.specular_factor = specular
    if kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = radius
    if kind == "SPOT":
        ld.spot_size, ld.spot_blend = math.radians(spot_deg), blend
    if kind == "AREA":
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size
    if kind == "SUN":
        ld.angle = radius
    if reach:
        ld.use_custom_distance, ld.cutoff_distance = True, reach
    o = bpy.data.objects.new(name, ld)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    if direction is not None:
        from mathutils import Vector
        o.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return o


# ---------------------------------------------------------------------------------------------------- nodes


def _is_link(v):
    return isinstance(v, (bpy.types.NodeSocket, bpy.types.Node))


class NB:
    """Shader node builder over one node tree. Methods return output sockets (shader nodes: the node), accept floats,
    tuples or sockets for every input."""

    def __init__(self, nt):
        self.nt = nt
        self._x = 0

    # -- plumbing
    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.Node):
            v = v.outputs[0]
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
            return
        dv = getattr(sock, "default_value", None)
        try:
            n = len(dv)
        except TypeError:
            n = 0
        if n:
            v = tuple(v) if hasattr(v, "__len__") else (v,) * n
            v = v + (1.0,) * (n - len(v)) if len(v) < n else v[:n]
        sock.default_value = v

    def node(self, kind, _in=None, **props):
        n = self.nt.nodes.new(kind)
        self._x += 1
        n.location = (200 * (self._x % 12), -140 * (self._x // 12))
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (_in or {}).items():
            if isinstance(k, int):
                self.set(n.inputs[k], v)
            else:
                self.set(n.inputs[k], v)
        return n

    # -- values
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", {0: a, 1: b, 2: c}, operation=op, use_clamp=clamp)
        return n.outputs[0]

    def vmath(self, op, a, b=None, c=None, scale=None):
        n = self.node("ShaderNodeVectorMath", {0: a, 1: b, 2: c, 3: scale}, operation=op)
        return n.outputs[0]

    def vdot(self, a, b):
        return self.node("ShaderNodeVectorMath", {0: a, 1: b}, operation="DOT_PRODUCT").outputs[1]

    def vlen(self, a):
        return self.node("ShaderNodeVectorMath", {0: a}, operation="LENGTH").outputs[1]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ", {0: v})
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        return self.node("ShaderNodeCombineXYZ", {0: x, 1: y, 2: z}).outputs[0]

    def mixf(self, t, a, b):
        return self.node("ShaderNodeMix", {0: t, 2: a, 3: b}, data_type="FLOAT").outputs[0]

    def mixc(self, t, a, b, blend="MIX", clamp=True):
        n = self.node("ShaderNodeMix", {0: t, 6: a, 7: b}, data_type="RGBA", blend_type=blend, clamp_factor=clamp)
        return n.outputs[2]

    def mixv(self, t, a, b):
        return self.node("ShaderNodeMix", {0: t, 4: a, 5: b}, data_type="VECTOR", factor_mode="UNIFORM").outputs[1]

    def remap(self, v, a0, a1, b0=0.0, b1=1.0, clamp=True):
        return self.node("ShaderNodeMapRange", {0: v, 1: a0, 2: a1, 3: b0, 4: b1}, clamp=clamp).outputs[0]

    def smooth(self, a, b, x):
        """smoothstep(a, b, x) for a < b."""
        t = self.remap(x, a, b, 0.0, 1.0)
        return self.math("MULTIPLY", self.math("MULTIPLY", t, t), self.math("SUBTRACT", 3.0, self.math("MULTIPLY", t, 2.0)))

    def ramp(self, fac, stops, interp="LINEAR"):
        """Colour ramp: stops [(pos, (r, g, b[, a])), ...] -> colour socket."""
        n = self.node("ShaderNodeValToRGB", {0: fac})
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (p, c) in zip(cr.elements, stops):
            el.position = p
            el.color = tuple(c) + (1.0,) * (4 - len(c))
        return n.outputs[0]

    def fcurve(self, fac, stops, interp="LINEAR"):
        """Float ramp [(pos, value)] -> float socket (greys through a colour ramp, read back with Separate)."""
        col = self.ramp(fac, [(p, (v, v, v)) for p, v in stops], interp)
        return self.sep(col)[0]

    # -- inputs
    def texcoord(self, which="Generated"):
        return self.node("ShaderNodeTexCoord").outputs[which]

    def attr(self, name, which="Color", kind="GEOMETRY"):
        return self.node("ShaderNodeAttribute", attribute_name=name, attribute_type=kind).outputs[which]

    def geometry(self, which="Position"):
        return self.node("ShaderNodeNewGeometry").outputs[which]

    def cam_distance(self):
        return self.node("ShaderNodeCameraData").outputs["View Distance"]

    def noise(self, vec, scale=5.0, detail=2.0, rough=0.5, dist=0.0, dims="3D", which="Fac", **kw):
        n = self.node("ShaderNodeTexNoise", {0: vec, 2: scale, 3: detail, 4: rough, 8: dist}, noise_dimensions=dims, **kw)
        return n.outputs[which]

    def voronoi(self, vec, scale=5.0, rand=1.0, feature="F1", metric="EUCLIDEAN", which="Distance", dims="3D"):
        n = self.node("ShaderNodeTexVoronoi", {0: vec, 2: scale, 8: rand}, feature=feature, distance=metric,
                      voronoi_dimensions=dims)
        return n.outputs[which]

    def white(self, vec, dims="3D"):
        return self.node("ShaderNodeTexWhiteNoise", {0: vec}, noise_dimensions=dims).outputs["Value"]

    # -- shaders
    def principled(self, base, rough=0.5, metal=0.0, emission=None, strength=0.0, coat=0.0, coat_rough=0.03,
                   spec=0.5, normal=None, alpha=None, coat_ior=None):
        ins = {"Base Color": base, "Roughness": rough, "Metallic": metal, "Coat Weight": coat,
               "Coat Roughness": coat_rough, "Specular IOR Level": spec, "Normal": normal, "Alpha": alpha,
               "Coat IOR": coat_ior}
        if emission is not None:
            ins["Emission Color"] = emission
            ins["Emission Strength"] = strength
        return self.node("ShaderNodeBsdfPrincipled", ins)

    def emission(self, colour, strength=1.0):
        return self.node("ShaderNodeEmission", {"Color": colour, "Strength": strength})

    def add_shader(self, a, b):
        return self.node("ShaderNodeAddShader", {0: a, 1: b})

    def mix_shader(self, fac, a, b):
        return self.node("ShaderNodeMixShader", {0: fac, 1: a, 2: b})

    def transparent(self):
        return self.node("ShaderNodeBsdfTransparent")

    def output(self, shader):
        out = self.node("ShaderNodeOutputMaterial")
        self.nt.links.new(shader.outputs[0] if isinstance(shader, bpy.types.Node) else shader, out.inputs["Surface"])
        return out


def new_material(name, blended=False):
    """A fresh node material (an earlier one of the same name is removed) and its node builder with the default nodes
    cleared."""
    old = bpy.data.materials.get(name)
    if old is not None:
        bpy.data.materials.remove(old)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    if blended:
        m.surface_render_method = "BLENDED"
        m.use_backface_culling = True
    return m, NB(m.node_tree)


def haze(nb, shader, colour, distance, cap=1.0, power=1.0):
    """Aerial perspective: mix `shader` toward flat `colour` with the distance from the camera, 1 - exp(-(d/distance)^
    power), at most `cap`. The same horizon colour as the sky, so far things dissolve into it."""
    d = nb.math("DIVIDE", nb.cam_distance(), distance)
    if power != 1.0:
        d = nb.math("POWER", d, power)
    f = nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.math("EXPONENT", nb.math("MULTIPLY", d, -1.0))), cap)
    return nb.mix_shader(f, shader, nb.emission(colour, 1.0))


def spill(nb, shader, albedo, name="spill", gain=1.0, floor=0.0, neutral=0.0):
    """Add light that is not simulated: Emission(albedo * (gain * attribute `name` + floor)). The attribute is the baked
    lamp light (an RGB per-vertex attribute, see nightgeo.spot_light); `floor` lifts the darkest surfaces so an unlit
    thing still shows its palette colour (floor 1 = exactly albedo) and nothing renders black. `neutral` (0..1) greys
    the albedo seen by the baked light only, so warm lamp light stays warm on a bluish surface. shader=None returns
    the emission alone (a surface lit only by the baked light)."""
    lamp = nb.vmath("MULTIPLY", nb.attr(name), (gain, gain, gain))
    lit_albedo = albedo
    if neutral:
        grey = nb.node("ShaderNodeRGBToBW", {0: albedo}).outputs[0]
        lit_albedo = nb.mixc(neutral, albedo, nb.comb(grey, grey, grey))
    out = nb.vmath("MULTIPLY", lit_albedo, lamp)
    if floor:
        out = nb.vmath("ADD", out, nb.vmath("MULTIPLY", albedo, (floor, floor, floor)))
    em = nb.emission(out, 1.0)
    return em if shader is None else nb.add_shader(shader, em)


def glow_material(name, colour, strength, power=2.0):
    """A soft additive glow for a sphere: radial falloff from the middle as seen from the camera, never occluding."""
    m, nb = new_material(name, blended=True)
    nrm, inc = nb.geometry("Normal"), nb.geometry("Incoming")
    fac = nb.math("ABSOLUTE", nb.vdot(nrm, inc))
    fac = nb.math("POWER", fac, power)
    em = nb.emission(colour, nb.math("MULTIPLY", fac, strength))
    nb.output(nb.add_shader(nb.transparent(), em))
    return m


def glow_solid_material(name, colour, strength, sigma=0.35):
    """A soft glow that is real (dithered) geometry for the renderer, for a sphere: alpha is a gaussian of the distance
    from the middle as the camera sees it (sigma as a share of the sphere's radius), so screen-space reflections and
    probes see it too (a lamp's halo is mirrored in a wet road)."""
    m, nb = new_material(name, blended=False)
    mu = nb.math("ABSOLUTE", nb.vdot(nb.geometry("Normal"), nb.geometry("Incoming")))
    rho2 = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", mu, mu))
    fac = nb.math("EXPONENT", nb.math("MULTIPLY", rho2, -1.0 / (sigma * sigma)))
    nb.output(nb.mix_shader(fac, nb.transparent(), nb.emission(colour, strength)))
    return m


def glow_gauss_material(name, colour, strength, sigma=0.3):
    """An additive glow for a sphere with a gaussian falloff from the middle as the camera sees it (no visible rim;
    sigma is a share of the sphere's radius). Clean and noise-free, but invisible to screen-space reflections."""
    m, nb = new_material(name, blended=True)
    mu = nb.math("ABSOLUTE", nb.vdot(nb.geometry("Normal"), nb.geometry("Incoming")))
    rho2 = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", mu, mu))
    fac = nb.math("EXPONENT", nb.math("MULTIPLY", rho2, -1.0 / (sigma * sigma)))
    nb.output(nb.add_shader(nb.transparent(), nb.emission(colour, nb.math("MULTIPLY", fac, strength))))
    return m
