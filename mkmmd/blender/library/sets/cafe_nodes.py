"""Shader-node and mesh helpers shared by the cafe room modules (cafe.py, cafe_glass.py, cafe_street.py). A helper
module: it registers no builder.

`NG` is a tiny node-graph builder: every method returns an output socket (or a tuple of them); python numbers and
tuples become socket defaults, sockets get linked. `NG.param(name)` reads one of the set's animatable parameters
(custom properties on the set root, see `params_group`)."""
import bmesh
import bpy
from mathutils import Vector

from ..props.cafe_kit import drive

# the set's animatable parameters: custom properties of the set root, read by the shaders through one node group
PARAMS = ("fog", "outside_sat", "sun", "bolt", "bolt_variant", "bolt_rig", "flash", "refl", "clip_t")


def params_group(name, root):
    """Node group "<name>_params" with one float output per parameter; a driver copies each from the root's custom
    property of the same name, so keying the property animates every material that uses the group."""
    g = bpy.data.node_groups.get(f"{name}_params")
    if g is not None:
        bpy.data.node_groups.remove(g)
    g = bpy.data.node_groups.new(f"{name}_params", "ShaderNodeTree")
    out = g.nodes.new("NodeGroupOutput")
    out.location = (300, 0)
    for i, p in enumerate(PARAMS):
        g.interface.new_socket(p, in_out="OUTPUT", socket_type="NodeSocketFloat")
        v = g.nodes.new("ShaderNodeValue")
        v.name = v.label = p
        v.location = (0, -60 * i)
        g.links.new(v.outputs[0], out.inputs[p])
        drive(g, f'nodes["{p}"].outputs[0].default_value', "p", var=("p", root, f'["{p}"]'))
    return g


class NG:
    def __init__(self, nt, params=None):
        self.nt = nt
        self.params = params
        self._grp = None
        self.i = 0

    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (-190 * (self.i % 28), -150 * (self.i // 28))
        self.i += 1
        return n

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
            return
        t = sock.type
        if t == "VECTOR":
            sock.default_value = (v, v, v) if isinstance(v, (int, float)) else tuple(v)[:3]
        elif t == "RGBA":
            v = (v, v, v, 1.0) if isinstance(v, (int, float)) else tuple(v)
            sock.default_value = v if len(v) == 4 else v + (1.0,)
        else:
            sock.default_value = v

    # ---- scalar / vector maths
    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op)
        n.use_clamp = clamp
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        self.put(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b, clamp=False): return self.m("ADD", a, b, clamp=clamp)
    def sub(self, a, b, clamp=False): return self.m("SUBTRACT", a, b, clamp=clamp)
    def mul(self, a, b, clamp=False): return self.m("MULTIPLY", a, b, clamp=clamp)
    def div(self, a, b): return self.m("DIVIDE", a, b)
    def mad(self, a, b, c): return self.m("MULTIPLY_ADD", a, b, c)
    def mn(self, a, b): return self.m("MINIMUM", a, b)
    def mx(self, a, b): return self.m("MAXIMUM", a, b)
    def lt(self, a, b): return self.m("LESS_THAN", a, b)
    def gt(self, a, b): return self.m("GREATER_THAN", a, b)
    def fl(self, a): return self.m("FLOOR", a)
    def fr(self, a): return self.m("FRACT", a)
    def sq(self, a): return self.m("SQRT", a)
    def ab(self, a): return self.m("ABSOLUTE", a)
    def pw(self, a, b): return self.m("POWER", a, b)
    def sat(self, a): return self.m("ADD", a, 0.0, clamp=True)
    def inv(self, a): return self.m("SUBTRACT", 1.0, a)

    def vm(self, op, a, b=None, c=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        self.put(n.inputs[2], c)
        self.put(n.inputs[3], scale)
        return n.outputs[1] if op in ("DOT_PRODUCT", "DISTANCE", "LENGTH") else n.outputs[0]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        self.put(n.inputs[0], x)
        self.put(n.inputs[1], y)
        self.put(n.inputs[2], z)
        return n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.put(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def rgb(self, r, g, b):
        n = self.node("ShaderNodeCombineColor")
        self.put(n.inputs[0], r)
        self.put(n.inputs[1], g)
        self.put(n.inputs[2], b)
        return n.outputs[0]

    def mixf(self, a, b, fac):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self.put(n.inputs[0], fac)
        self.put(n.inputs[2], a)
        self.put(n.inputs[3], b)
        return n.outputs[0]

    def mixv(self, a, b, fac):
        n = self.node("ShaderNodeMix", data_type="VECTOR")
        self.put(n.inputs[0], fac)
        self.put(n.inputs[4], a)
        self.put(n.inputs[5], b)
        return n.outputs[1]

    def mixc(self, a, b, fac, blend="MIX"):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.put(n.inputs[0], fac)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def maprange(self, v, a0, a1, b0=0.0, b1=1.0, interp="LINEAR", clamp=True):
        n = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.put(n.inputs[0], v)
        self.put(n.inputs[1], a0)
        self.put(n.inputs[2], a1)
        self.put(n.inputs[3], b0)
        self.put(n.inputs[4], b1)
        return n.outputs[0]

    def ss(self, v, lo, hi):
        return self.maprange(v, lo, hi, 0.0, 1.0, "SMOOTHSTEP")

    # ---- textures / inputs
    def texco(self, which):
        return self.node("ShaderNodeTexCoord").outputs[which]

    def param(self, name):
        """One of the set's parameters (PARAMS) as a float socket."""
        if self._grp is None:
            self._grp = self.node("ShaderNodeGroup")
            self._grp.node_tree = self.params
        return self._grp.outputs[name]

    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, dims="3D", w=None):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        self.put(n.inputs["Vector"], vec)
        self.put(n.inputs["Scale"], scale)
        self.put(n.inputs["Detail"], detail)
        self.put(n.inputs["Roughness"], rough)
        self.put(n.inputs["Distortion"], distortion)
        if w is not None:
            self.put(n.inputs["W"], w)
        return n.outputs["Fac"]

    def voronoi(self, vec, scale, dims="2D", feature="F1", randomness=1.0):
        n = self.node("ShaderNodeTexVoronoi", voronoi_dimensions=dims, feature=feature)
        self.put(n.inputs["Vector"], vec)
        self.put(n.inputs["Scale"], scale)
        self.put(n.inputs["Randomness"], randomness)
        return n

    def white(self, dims="1D", w=None, vec=None):
        n = self.node("ShaderNodeTexWhiteNoise", noise_dimensions=dims)
        if w is not None:
            self.put(n.inputs["W"], w)
        if vec is not None:
            self.put(n.inputs["Vector"], vec)
        return n.outputs["Value"], n.outputs["Color"]

    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        n = self.node("ShaderNodeMapping")
        self.put(n.inputs[0], vec)
        self.put(n.inputs[1], loc)
        self.put(n.inputs[2], rot)
        self.put(n.inputs[3], scale)
        return n.outputs[0]

    def hsv(self, col, hue=0.5, sat=1.0, val=1.0, fac=1.0):
        n = self.node("ShaderNodeHueSaturation")
        self.put(n.inputs["Hue"], hue)
        self.put(n.inputs["Saturation"], sat)
        self.put(n.inputs["Value"], val)
        self.put(n.inputs["Fac"], fac)
        self.put(n.inputs["Color"], col)
        return n.outputs[0]

    def bump(self, height, strength=0.3, dist=0.01):
        n = self.node("ShaderNodeBump")
        self.put(n.inputs["Strength"], strength)
        self.put(n.inputs["Distance"], dist)
        self.put(n.inputs["Height"], height)
        return n.outputs[0]


def principled(g, base, rough=0.5, metal=0.0, spec=0.5, normal=None, **extra):
    """Principled BSDF node; extra keys are input names ('Transmission Weight' as transmission_weight, ...)."""
    n = g.node("ShaderNodeBsdfPrincipled")
    g.put(n.inputs["Base Color"], base)
    g.put(n.inputs["Roughness"], rough)
    g.put(n.inputs["Metallic"], metal)
    g.put(n.inputs["Specular IOR Level"], spec)
    if normal is not None:
        g.put(n.inputs["Normal"], normal)
    for k, v in extra.items():
        g.put(n.inputs[k if k in n.inputs else k.replace("_", " ").title()], v)
    return n


def finish(g, shader, thickness=None):
    out = g.node("ShaderNodeOutputMaterial")
    g.nt.links.new(shader, out.inputs["Surface"])
    if thickness is not None:
        g.put(out.inputs["Thickness"], thickness)
    return out


def hide_from_rays(o, shadow=False, glossy=True, diffuse=True, transmission=True, volume=False):
    o.visible_shadow = shadow
    o.visible_glossy = glossy
    o.visible_diffuse = diffuse
    o.visible_transmission = transmission
    o.visible_volume_scatter = volume


# ================================================================= meshes
def box_geo(bm, x0, x1, y0, y1, z0, z1):
    ret = bmesh.ops.create_cube(bm, size=1.0)
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    for v in ret["verts"]:
        v.co = Vector((cx + v.co.x * sx, cy + v.co.y * sy, cz + v.co.z * sz))
    return ret["verts"]


def axis_mat(normal, mats):
    """Face material index by dominant axis: mats maps '+x','-x','+y','-y','+z','-z' (or 'any') to a slot index."""
    a = max(range(3), key=lambda i: abs(normal[i]))
    k = ("+" if normal[a] >= 0 else "-") + "xyz"[a]
    return mats.get(k, mats.get("any", 0))


# ================================================================= per-set context
class Ctx:
    """What the cafe room modules share: names, collections, palette colours, the parameter group, time."""

    def __init__(self, name, coll, root, colors, params, spec, frame0, fps):
        self.name, self.coll, self.root = name, coll, root
        self.C, self.params, self.spec = colors, params, spec
        self.frame0, self.fps = frame0, fps
        self._groups = {}

    def group(self, key):
        """Child collection "<set>_<key>" of the set's collection."""
        if key not in self._groups:
            c = bpy.data.collections.new(f"{self.name}_{key}")
            self.coll.children.link(c)
            self._groups[key] = c
        return self._groups[key]

    def oname(self, base):
        return f"{self.name}_{base}"

    def obj(self, base, data, group="room", loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        """Object "<set>_<base>" with `data`, linked to the group collection and parented to the set root."""
        o = bpy.data.objects.new(self.oname(base), data)
        self.group(group).objects.link(o)
        o.parent = self.root
        o.location, o.rotation_euler, o.scale = loc, rot, scale
        return o

    def mat(self, base, blend="DITHERED"):
        """Material "<set>_<base>" with an empty node tree -> (material, NG)."""
        m = bpy.data.materials.get(self.oname(base)) or bpy.data.materials.new(self.oname(base))
        m.use_nodes = True
        for n in list(m.node_tree.nodes):
            m.node_tree.nodes.remove(n)
        m.surface_render_method = blend
        return m, NG(m.node_tree, self.params)

    def mesh_obj(self, base, verts, faces, group, mats=(), uvs=None, slots=None, smooth=False):
        me = bpy.data.meshes.new(self.oname(base))
        me.from_pydata(verts, [], faces)
        if uvs is not None:
            uv = me.uv_layers.new(name="UVMap")
            for poly in me.polygons:
                for li in poly.loop_indices:
                    uv.data[li].uv = uvs[me.loops[li].vertex_index]
        if slots is not None:
            for p, sl in zip(me.polygons, slots):
                p.material_index = sl
        for p in me.polygons:
            p.use_smooth = smooth
        me.update()
        for m in mats:
            me.materials.append(m)
        return self.obj(base, me, group)

    def boxes_obj(self, base, specs, group, mats, face_slot=None, bevel=0.0, shade_smooth=False):
        """One mesh object from axis-aligned boxes (x0, x1, y0, y1, z0, z1). `mats`: list of materials (slots);
        `face_slot(normal) -> slot`. Optional bevel on every edge."""
        bm = bmesh.new()
        for s in specs:
            box_geo(bm, *s[:6])
        if bevel:
            bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, affect="EDGES", profile=0.5)
        bm.normal_update()
        for f in bm.faces:
            f.material_index = face_slot(f.normal) if face_slot else 0
            f.smooth = shade_smooth
        me = bpy.data.meshes.new(self.oname(base))
        bm.to_mesh(me)
        bm.free()
        for m in mats:
            me.materials.append(m)
        return self.obj(base, me, group)
