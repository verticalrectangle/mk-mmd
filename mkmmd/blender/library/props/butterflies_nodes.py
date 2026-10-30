"""The geometry nodes of library:butterflies (butterflies.py builds the prop around them).

`Nodes` is an ops backend of mkmmd.core.butterflies: the swarm's maths, written once there, run on it makes one node per
op (Math, Vector Math, Map Range, Mix, Clamp, Combine / Separate XYZ), Python numbers staying constants (default values of
the inputs). `Float` and `Vector` wrap a node's output: `+ - *` between them as in the core's numpy backend (a vector
times a float scales it, two vectors multiply component by component), `/` only by a Python number (`div` for anything
else), so the tree computes what the reference `pose` computes, by construction.

`swarm_tree` is the swarm's node tree: the maths evaluated once per frame on the swarm's points (one Capture Attribute
holds each butterfly's position, rotation (Axes to Rotation: Y along -head, Z toward the back), span, wing angle and bob),
then for each part (body, wing_L, wing_R) the hidden source objects of every style and colourway joined as instances,
picked per point by its `pick` attribute (Instance on Points with Pick Instance) and placed by Set Instance Transform:
body T(pos) R S(span) T(0, 0, bob), a wing that times T(hinge + (0, 0, bob)) R_Y(-+angle)."""
import math

import bpy

from ....core import butterflies as BS
from ....core import butterfly as BF

GROUP_COLUMN = 36                     # nodes per column of the tree's layout (only for looking at it)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _socket(sockets, identifier):
    """A node's socket by its identifier (a name lookup finds the first socket of that name, of whatever type)."""
    for s in sockets:
        if s.identifier == identifier:
            return s
    raise KeyError(f"butterflies: no socket {identifier!r} ({[s.identifier for s in sockets]})")


class _Value:
    __array_ufunc__ = None            # a numpy number on the left defers to these operators
    __slots__ = ("g", "sock")

    def __init__(self, g, sock):
        self.g, self.sock = g, sock


class Float(_Value):
    __slots__ = ()

    def __add__(self, b):
        return self.g.math("ADD", self, b)

    def __radd__(self, a):
        return self.g.math("ADD", a, self)

    def __sub__(self, b):
        return self.g.math("SUBTRACT", self, b)

    def __rsub__(self, a):
        return self.g.math("SUBTRACT", a, self)

    def __mul__(self, b):
        return self.g.scale(b, self) if isinstance(b, Vector) else self.g.math("MULTIPLY", self, b)

    def __rmul__(self, a):
        return self.g.math("MULTIPLY", a, self)

    def __truediv__(self, b):
        if not _is_number(b):
            raise TypeError("butterflies: `/` takes a Python number; divide by a value with div()")
        return self.g.math("DIVIDE", self, b)

    def __neg__(self):
        return self.g.math("MULTIPLY", self, -1.0)


class Vector(_Value):
    __slots__ = ()

    def __add__(self, b):
        return self.g.vmath("ADD", self, b)

    def __radd__(self, a):
        return self.g.vmath("ADD", a, self)

    def __sub__(self, b):
        return self.g.vmath("SUBTRACT", self, b)

    def __rsub__(self, a):
        return self.g.vmath("SUBTRACT", a, self)

    def __mul__(self, b):
        return self.g.vmath("MULTIPLY", self, b) if isinstance(b, Vector) else self.g.scale(self, b)

    def __rmul__(self, a):
        return self.g.scale(self, a)

    def __truediv__(self, b):
        if not _is_number(b):
            raise TypeError("butterflies: `/` takes a Python number; divide by a value with div()")
        return self.g.scale(self, 1.0 / b)

    def __neg__(self):
        return self.g.scale(self, -1.0)


_PY = {"ADD": lambda a, b: a + b, "SUBTRACT": lambda a, b: a - b, "MULTIPLY": lambda a, b: a * b,
       "DIVIDE": lambda a, b: a / b if b != 0 else 0.0, "MINIMUM": min, "MAXIMUM": max,
       "SINE": lambda a, _b: math.sin(a), "COSINE": lambda a, _b: math.cos(a),
       "SQRT": lambda a, _b: math.sqrt(a) if a > 0 else 0.0, "ABSOLUTE": lambda a, _b: abs(a)}


class Nodes:
    """The ops of mkmmd.core.butterflies as nodes of the geometry node tree `nt`."""

    def __init__(self, nt):
        self.nt, self.count = nt, 0

    # ------------------------------------------------------------------------------------------------- plumbing
    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (240.0 * (self.count // GROUP_COLUMN), -170.0 * (self.count % GROUP_COLUMN))
        self.count += 1
        return n

    def put(self, sock, v):
        if isinstance(v, _Value):
            self.nt.links.new(v.sock, sock)
        elif isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif _is_number(v):
            sock.default_value = float(v)
        else:
            raise TypeError(f"butterflies: cannot feed {v!r} to a node")

    def math(self, op, a, b=None):
        if isinstance(a, Vector) or isinstance(b, Vector):
            raise TypeError(f"butterflies: {op} of a vector and a float (scale a vector by a float with *)")
        if not isinstance(a, _Value) and not isinstance(b, _Value):
            return _PY[op](float(a), None if b is None else float(b))
        n = self.node("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return Float(self, n.outputs[0])

    def vmath(self, op, a, b=None):
        if not isinstance(a, Vector) or not (b is None or isinstance(b, Vector)):
            raise TypeError(f"butterflies: vector {op} takes vectors (got {type(a).__name__}, {type(b).__name__})")
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.nt.links.new(a.sock, n.inputs[0])
        if b is not None:
            self.nt.links.new(b.sock, n.inputs[1])
        if op in ("DOT_PRODUCT", "LENGTH"):
            return Float(self, n.outputs["Value"])
        return Vector(self, n.outputs["Vector"])

    def attr(self, name, kind):
        """A point attribute of the swarm's points by name: kind FLOAT, FLOAT_VECTOR or INT."""
        n = self.node("GeometryNodeInputNamedAttribute", data_type=kind)
        n.inputs["Name"].default_value = name
        return (Vector if kind == "FLOAT_VECTOR" else Float)(self, n.outputs["Attribute"])

    # ------------------------------------------------------------------------------------------------- the ops
    def vec(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for sock, c in zip(n.inputs, (x, y, z)):
            self.put(sock, c)
        return Vector(self, n.outputs[0])

    def xyz(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.nt.links.new(v.sock, n.inputs[0])
        return tuple(Float(self, s) for s in n.outputs)

    def sin(self, x):
        return self.vmath("SINE", x) if isinstance(x, Vector) else self.math("SINE", x)

    def cos(self, x):
        return self.vmath("COSINE", x) if isinstance(x, Vector) else self.math("COSINE", x)

    def sqrt(self, x):
        return self.math("SQRT", x)

    def abs(self, x):
        return self.math("ABSOLUTE", x)

    def min(self, a, b):
        return self.math("MINIMUM", a, b)

    def max(self, a, b):
        return self.math("MAXIMUM", a, b)

    def div(self, a, b):
        return self.math("DIVIDE", a, b)

    def clamp(self, x, lo, hi):
        if not any(isinstance(v, _Value) for v in (x, lo, hi)):
            return min(max(float(x), float(lo)), float(hi))
        n = self.node("ShaderNodeClamp", clamp_type="MINMAX")
        for sock, v in zip(n.inputs, (x, lo, hi)):
            self.put(sock, v)
        return Float(self, n.outputs[0])

    def smoothstep(self, a, b, x):
        """Map Range's smoothstep: (3 - 2 f) f^2 of f = (x - a) / (b - a) clamped to 0..1 (0 when b == a)."""
        n = self.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP")
        for i, v in enumerate((x, a, b, 0.0, 1.0)):
            self.put(n.inputs[i], v)
        return Float(self, n.outputs["Result"])

    def mix(self, a, b, f):
        """a + (b - a) f, unclamped: two floats, or two vectors by a float."""
        vector = isinstance(a, Vector) or isinstance(b, Vector)
        n = self.node("ShaderNodeMix", data_type="VECTOR" if vector else "FLOAT", clamp_factor=False)
        kind = "Vector" if vector else "Float"
        self.put(_socket(n.inputs, "Factor_Float"), f)
        self.put(_socket(n.inputs, f"A_{kind}"), a)
        self.put(_socket(n.inputs, f"B_{kind}"), b)
        return (Vector if vector else Float)(self, _socket(n.outputs, f"Result_{kind}"))

    def dot(self, a, b):
        return self.vmath("DOT_PRODUCT", a, b)

    def cross(self, a, b):
        return self.vmath("CROSS_PRODUCT", a, b)

    def length(self, v):
        return self.vmath("LENGTH", v)

    def normalize(self, v):
        return self.vmath("NORMALIZE", v)

    def scale(self, v, f):
        if not isinstance(v, Vector) or isinstance(f, Vector):
            raise TypeError("butterflies: scale(vector, float)")
        n = self.node("ShaderNodeVectorMath", operation="SCALE")
        self.nt.links.new(v.sock, n.inputs[0])
        self.put(n.inputs["Scale"], f)
        return Vector(self, n.outputs["Vector"])


# ===================================================================================================================
# the swarm's tree
# ===================================================================================================================
CAPTURE = (("pos", "VECTOR"), ("rot", "ROTATION"), ("span", "FLOAT"), ("angle", "FLOAT"), ("bob", "FLOAT"))


def swarm_tree(tree_name, L, sources):
    """The node tree `tree_name` of a swarm (Layout L) whose parts are `sources` {part: [object of each L.combos entry]}.
    Returns (tree, {input name: the modifier's socket identifier}, pick (len(L.combos),): the instance index of each
    combo in the tree's lists, which the points' `pick` attribute must hold)."""
    old = bpy.data.node_groups.get(tree_name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    nt = bpy.data.node_groups.new(tree_name, "GeometryNodeTree")
    nt.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    idents = {k: nt.interface.new_socket(k, in_out="INPUT", socket_type="NodeSocketFloat").identifier for k in BS.INPUTS}
    g = Nodes(nt)
    gin, gout = g.node("NodeGroupInput"), g.node("NodeGroupOutput")
    P = {k: Float(g, gin.outputs[ident]) for k, ident in idents.items()}
    A = {k: g.attr(k, kind) for k, kind in BS.ATTRS.items()}
    R = BS.swarm(g, A, P, L.consts)
    axes = g.node("FunctionNodeAxesToRotation", primary_axis="Y", secondary_axis="Z")
    g.put(axes.inputs["Primary Axis"], -R["head"])
    g.put(axes.inputs["Secondary Axis"], R["back"])
    cap = g.node("GeometryNodeCaptureAttribute", domain="POINT")
    for nm, kind in CAPTURE:
        cap.capture_items.new(kind, nm)
    nt.links.new(gin.outputs["Geometry"], cap.inputs["Geometry"])
    values = dict(R, rot=axes.outputs["Rotation"])
    for nm, _kind in CAPTURE:
        g.put(cap.inputs[nm], values[nm])
    pos, span = Vector(g, cap.outputs["pos"]), Float(g, cap.outputs["span"])
    angle, bob = Float(g, cap.outputs["angle"]), Float(g, cap.outputs["bob"])
    body = g.node("FunctionNodeCombineTransform")
    g.put(body.inputs["Translation"], pos)
    nt.links.new(cap.outputs["rot"], body.inputs["Rotation"])
    g.put(body.inputs["Scale"], g.vec(span, span, span))
    pick = g.attr("pick", "INT")
    out = g.node("GeometryNodeJoinGeometry")
    orders = []
    for part in BS.PARTS:
        join = g.node("GeometryNodeJoinGeometry")
        for ob in sources[part]:
            info = g.node("GeometryNodeObjectInfo", transform_space="ORIGINAL")
            info.inputs["Object"].default_value = ob
            info.inputs["As Instance"].default_value = True
            nt.links.new(info.outputs["Geometry"], join.inputs[0])
        named = [ln.from_node.inputs["Object"].default_value.name for ln in join.inputs[0].links]
        orders.append([sources[part].index(bpy.data.objects[n]) for n in named])
        iop = g.node("GeometryNodeInstanceOnPoints")
        nt.links.new(cap.outputs["Geometry"], iop.inputs["Points"])
        nt.links.new(join.outputs[0], iop.inputs["Instance"])
        iop.inputs["Pick Instance"].default_value = True
        g.put(iop.inputs["Instance Index"], pick)
        local = g.node("FunctionNodeCombineTransform")
        if part == "body":
            g.put(local.inputs["Translation"], g.vec(0.0, 0.0, bob))
        else:
            side = part[-1]
            hx, hy, hz = BF.hinge(side)
            g.put(local.inputs["Translation"], g.vec(hx, hy, hz + bob))
            euler = g.node("FunctionNodeEulerToRotation")
            g.put(euler.inputs[0], g.vec(0.0, -angle if side == "L" else angle, 0.0))
            nt.links.new(euler.outputs[0], local.inputs["Rotation"])
        both = g.node("FunctionNodeMatrixMultiply")
        nt.links.new(body.outputs[0], both.inputs[0])
        nt.links.new(local.outputs[0], both.inputs[1])
        sit = g.node("GeometryNodeSetInstanceTransform")
        nt.links.new(iop.outputs[0], sit.inputs["Instances"])
        nt.links.new(both.outputs[0], sit.inputs["Transform"])
        nt.links.new(sit.outputs[0], out.inputs[0])
    if any(o != orders[0] for o in orders):
        raise RuntimeError(f"butterflies: the parts' instance lists came out in different orders ({orders})")
    if sorted(orders[0]) != list(range(len(L.combos))):
        raise RuntimeError(f"butterflies: the instance list {orders[0]} is not one of each of {len(L.combos)} sources")
    pick_of = [0] * len(orders[0])
    for position, combo in enumerate(orders[0]):
        pick_of[combo] = position
    nt.links.new(out.outputs[0], gout.inputs["Geometry"])
    return nt, idents, pick_of
