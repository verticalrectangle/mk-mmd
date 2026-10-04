"""Shader-node building blocks of the bedroom soft furnishings: a small expression builder over Math nodes and the
Memphis print evaluated on the GPU (the functions of `bedroom_soft_pattern`, node for node).

    nb = NB(nt)
    u, v = nb.uv("UVMap")                       # the UV map's coordinates (metres on the flat cloth) as scalars
    d = nb.hypot(u, v) - 0.3                    # Python operators make Math nodes (constants fold away)
    m = nb.sat(0.5 - d / 0.004)                 # a 4 mm wide anti-aliased edge

Colours are plain sockets or RGBA tuples; `nb.cmix(fac, a, b)` mixes them, `nb.table(stops, fac)` reads a colour-ramp
lookup table (constant interpolation: stop i is read at (i + 0.5) / n)."""
import math

from . import bedroom_soft_pattern as PT

_FOLD = {"ADD": lambda a, b: a + b, "SUBTRACT": lambda a, b: a - b, "MULTIPLY": lambda a, b: a * b,
         "DIVIDE": lambda a, b: a / b if b else 0.0, "MINIMUM": min, "MAXIMUM": max, "ABSOLUTE": abs,
         "SQRT": lambda a: math.sqrt(max(a, 0.0)), "SINE": math.sin, "COSINE": math.cos,
         "FRACT": lambda a: a - math.floor(a), "FLOOR": math.floor, "MULTIPLY_ADD": lambda a, b, c: a * b + c}


class Ex:
    """A scalar of a node tree: a float constant or a socket. Operators build Math nodes."""
    __slots__ = ("nb", "v")

    def __init__(self, nb, v):
        self.nb, self.v = nb, v

    @property
    def const(self):
        return isinstance(self.v, float)

    def _w(self, o):
        return o if isinstance(o, Ex) else Ex(self.nb, float(o))

    def __add__(self, o):
        return self.nb.op("ADD", self, self._w(o))

    __radd__ = __add__

    def __sub__(self, o):
        return self.nb.op("SUBTRACT", self, self._w(o))

    def __rsub__(self, o):
        return self.nb.op("SUBTRACT", self._w(o), self)

    def __mul__(self, o):
        return self.nb.op("MULTIPLY", self, self._w(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.nb.op("DIVIDE", self, self._w(o))

    def __rtruediv__(self, o):
        return self.nb.op("DIVIDE", self._w(o), self)

    def __neg__(self):
        return self.nb.op("MULTIPLY", self, Ex(self.nb, -1.0))

    def __abs__(self):
        return self.nb.op("ABSOLUTE", self)


class NB:
    """Expression builder for one node tree."""

    def __init__(self, nt, x0=-3400, y0=900):
        self.nt, self.n, self.x0, self.y0 = nt, 0, x0, y0

    # ---- placement and generic nodes
    def _loc(self):
        i = self.n
        self.n += 1
        return (self.x0 + (i % 36) * 190, self.y0 - (i // 36) * 150)

    def node(self, kind, **attrs):
        n = self.nt.nodes.new(kind)
        n.location = self._loc()
        for k, v in attrs.items():
            setattr(n, k, v)
        return n

    def c(self, v):
        return Ex(self, float(v))

    def wrap(self, socket):
        return Ex(self, socket)

    def op(self, name, *args, clamp=False):
        args = [a if isinstance(a, Ex) else Ex(self, float(a)) for a in args]
        if not clamp and name in _FOLD and all(a.const for a in args):
            return Ex(self, float(_FOLD[name](*[a.v for a in args])))
        n = self.node("ShaderNodeMath", operation=name, use_clamp=clamp)
        for i, a in enumerate(args):
            if a.const:
                n.inputs[i].default_value = a.v
            else:
                self.nt.links.new(a.v, n.inputs[i])
        return Ex(self, n.outputs[0])

    def link(self, ex, socket):
        """Feed an expression (or a constant) into an input socket."""
        if ex.const:
            socket.default_value = ex.v
        else:
            self.nt.links.new(ex.v, socket)

    # ---- scalar helpers
    def min(self, a, b):
        return self.op("MINIMUM", a, b)

    def max(self, a, b):
        return self.op("MAXIMUM", a, b)

    def abs(self, a):
        return self.op("ABSOLUTE", a)

    def sqrt(self, a):
        return self.op("SQRT", a)

    def sin(self, a):
        return self.op("SINE", a)

    def cos(self, a):
        return self.op("COSINE", a)

    def fract(self, a):
        return self.op("FRACT", a)

    def floor(self, a):
        return self.op("FLOOR", a)

    def madd(self, a, b, c):
        return self.op("MULTIPLY_ADD", a, b, c)

    def sat(self, a):
        """Clamp to [0, 1] (a Math node's own clamp)."""
        if isinstance(a, Ex) and a.const:
            return Ex(self, min(max(a.v, 0.0), 1.0))
        return self.op("ADD", a, 0.0, clamp=True)

    def clamp(self, a, lo, hi):
        return self.min(self.max(a, lo), hi)

    def hypot(self, a, b):
        return self.sqrt(self.madd(a, a, b * b))

    def eq(self, a, k):
        """1.0 where |a - k| < 0.1 (integers carried in floats), else 0.0."""
        return self.op("COMPARE", a, float(k), 0.1)

    def gt(self, a, b):
        return self.op("GREATER_THAN", a, b)

    def lt(self, a, b):
        return self.op("LESS_THAN", a, b)

    def mix(self, a, b, t):
        """a + (b - a) * t"""
        return self.madd(b - a, t, a)

    def smooth(self, x):
        """smoothstep of x already scaled to [0, 1]."""
        x = self.sat(x)
        return x * x * (3.0 - 2.0 * x)

    def edge(self, d, aa):
        """Anti-aliased mask of the negative side of a signed distance d (metres): a linear ramp 2 aa wide."""
        return self.sat(0.5 - d / (2.0 * aa))

    # ---- inputs
    def uv(self, name="UVMap"):
        n = self.node("ShaderNodeUVMap", uv_map=name)
        s = self.node("ShaderNodeSeparateXYZ")
        self.nt.links.new(n.outputs[0], s.inputs[0])
        return Ex(self, s.outputs["X"]), Ex(self, s.outputs["Y"])

    def obj(self):
        """Object coordinates as (x, y, z) scalars."""
        n = self.node("ShaderNodeTexCoord")
        s = self.node("ShaderNodeSeparateXYZ")
        self.nt.links.new(n.outputs["Object"], s.inputs[0])
        return Ex(self, s.outputs["X"]), Ex(self, s.outputs["Y"]), Ex(self, s.outputs["Z"])

    def vec(self, x, y, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        for k, e in zip("XYZ", (x, y, z)):
            self.link(e if isinstance(e, Ex) else Ex(self, float(e)), n.inputs[k])
        return n.outputs[0]

    # ---- colours
    def cmix(self, fac, a, b):
        """Mix colours a, b (sockets or RGBA tuples) by the scalar `fac` -> socket."""
        n = self.node("ShaderNodeMixRGB", blend_type="MIX")
        self.link(fac if isinstance(fac, Ex) else Ex(self, float(fac)), n.inputs["Fac"])
        for k, col in (("Color1", a), ("Color2", b)):
            if isinstance(col, (tuple, list)):
                n.inputs[k].default_value = tuple(col)
            else:
                self.nt.links.new(col, n.inputs[k])
        return n.outputs["Color"]

    def cmul(self, a, b, fac=1.0):
        """Multiply colour a by b, blended in by `fac` (a number or a scalar) -> socket."""
        n = self.node("ShaderNodeMixRGB", blend_type="MULTIPLY")
        self.link(fac if isinstance(fac, Ex) else Ex(self, float(fac)), n.inputs["Fac"])
        for k, col in (("Color1", a), ("Color2", b)):
            if isinstance(col, (tuple, list)):
                n.inputs[k].default_value = tuple(col)
            else:
                self.nt.links.new(col, n.inputs[k])
        return n.outputs["Color"]

    def csplit(self, socket):
        n = self.node("ShaderNodeSeparateColor")
        self.nt.links.new(socket, n.inputs["Color"])
        return Ex(self, n.outputs["Red"]), Ex(self, n.outputs["Green"]), Ex(self, n.outputs["Blue"])

    def table(self, stops, fac, interp="CONSTANT"):
        """ColorRamp node over `stops` [(position, rgba)] read at the scalar `fac`; returns the node."""
        n = self.node("ShaderNodeValToRGB")
        n.color_ramp.interpolation = interp
        cr = n.color_ramp
        cr.elements[0].position, cr.elements[0].color = stops[0]
        cr.elements[1].position, cr.elements[1].color = stops[-1]
        for pos, col in stops[1:-1]:
            e = cr.elements.new(pos)
            e.color = col
        self.link(fac, n.inputs["Fac"])
        return n

    def table4(self, stops, fac):
        """The four channels of a table read as scalars (r, g, b, a)."""
        n = self.table(stops, fac)
        r, g, b = self.csplit(n.outputs["Color"])
        return r, g, b, Ex(self, n.outputs["Alpha"])


# ================================================================================================== the Memphis print
def motif_nodes(nb, kind, x, y, R):
    """Signed distance of motif `kind` at the rotated cell coordinates (x, y), motif radius R: the node twin of
    `bedroom_soft_pattern.sdf` (same dimensions)."""
    if kind == PT.TRIANGLE:
        return nb.max(nb.max(-y, PT.S3 * x + 0.5 * y), -PT.S3 * x + 0.5 * y) - PT.TRI_IN * R
    if kind == PT.DOTS:
        ds = [nb.hypot(x - cx * R, y - cy * R) - r * R for cx, cy, r in PT.DOT_SET]
        return nb.min(nb.min(ds[0], ds[1]), ds[2])
    if kind == PT.RING:
        return nb.abs(nb.hypot(x, y) - PT.RING_R * R) - PT.RING_W * R
    if kind in (PT.SQUIGGLE, PT.ZIGZAG):
        if kind == PT.SQUIGGLE:
            k, A = PT.TAU / (PT.SQ_P * R), PT.SQ_A * R
            kx = k * x
            yc = A * nb.sin(kx)
            slope = A * k * nb.cos(kx)
            g = 1.0 / nb.sqrt(slope * slope + 1.0)
            w, Lh = PT.SQ_W * R, PT.SQ_L * R
        else:
            xp = x / (PT.ZZ_P * R)
            yc = (PT.ZZ_A * R) * (4.0 * nb.abs(nb.fract(xp) - 0.5) - 1.0)
            g = 1.0 / math.sqrt(1.0 + (4.0 * PT.ZZ_A / PT.ZZ_P) ** 2)
            w, Lh = PT.ZZ_W * R, PT.ZZ_L * R
        q1 = nb.abs(x) - (Lh - w)
        q2 = nb.abs(y - yc) * g
        return nb.hypot(nb.max(q1, 0.0), nb.max(q2, 0.0)) + nb.min(nb.max(q1, q2), 0.0) - w
    if kind == PT.STRIPES:
        pitch = PT.ST_PITCH * R
        bars = nb.abs(nb.fract(y / pitch + 0.5) - 0.5) * pitch - PT.ST_W * R
        return nb.max(bars, nb.max(nb.abs(x), nb.abs(y)) - PT.ST_BOX * R)
    if kind == PT.ARCH:
        yy = y + PT.ARCH_SHIFT * R
        return nb.max(nb.abs(nb.hypot(x, yy) - PT.ARCH_R * R) - PT.ARCH_W * R, -yy)
    if kind == PT.CROSS:
        return nb.min(nb.max(nb.abs(x) - PT.CROSS_L * R, nb.abs(y) - PT.CROSS_W * R),
                      nb.max(nb.abs(x) - PT.CROSS_W * R, nb.abs(y) - PT.CROSS_L * R))
    raise ValueError(kind)


def memphis(nb, u, v, nx, ny, sx, sy, tables, aa=0.0022):
    """The lattice print at the flat coordinates (u, v) (metres from the lattice corner): returns (mask, c1, c2) as
    scalars: the anti-aliased coverage of the cell's motif, its two colour indices and the signed distance to the motif
    (metres; large where the cell is empty). `tables`: one (A, B) pair of
    `bedroom_soft_pattern.lut_stops` per 32 cells (at most two); cell index = cx + nx * row within its table, rows
    beyond LUT_SIZE // nx continue in the next table."""
    rows = PT.LUT_SIZE // nx
    cx = nb.clamp(nb.floor(u / sx), 0.0, nx - 1.0)
    cy = nb.clamp(nb.floor(v / sy), 0.0, ny - 1.0)
    px = u - (cx + 0.5) * sx
    py = v - (cy + 0.5) * sy
    tab = nb.floor(cy / rows) if len(tables) > 1 else nb.c(0.0)
    row = cy - tab * rows
    fac = (cx + row * nx + 0.5) / PT.LUT_SIZE
    vals = [list(nb.table4(A, fac)) + list(nb.table4(B, fac)) for A, B in tables]
    if len(vals) == 2:
        vals = [nb.mix(a, b, tab) for a, b in zip(*vals)]
    else:
        vals = vals[0]
    ar, ag, ab, _, br, bg, bb, ba = vals
    kind = nb.floor(ar * 8.0 + 0.5)
    rot = ag * PT.TAU
    scale = ab + PT.SCALE_LO
    px = px - (br - 0.5) * sx
    py = py - (bg - 0.5) * sy
    cs, sn = nb.cos(rot), nb.sin(rot)
    x = cs * px + sn * py
    y = cs * py - sn * px
    R = scale * (PT.MOTIF_R * min(sx, sy))
    d = nb.eq(kind, 0) * 1000.0
    for k in range(1, 9):
        d = d + nb.eq(kind, k) * motif_nodes(nb, k, x, y, R)
    return nb.edge(d, aa), nb.floor(bb * 7.0 + 0.5), nb.floor(ba * 7.0 + 0.5), d
