"""Shader-node back end of `bedroom_decor_field`: writes field and colour expressions as nodes of one material.

`Compiler(nt, palette, x, y)` turns expressions into node sockets (x, y are the sockets that carry the design
coordinates). Every distinct expression becomes at most a few nodes; constants become socket defaults; multi-output ops
(`xf`) and vector packs share one node. The arithmetic is exactly what `Evaluator` computes with numpy."""
from .bedroom_decor_field import topo
from .cafe_kit import L, N, principled, ramp

MATH = {"add": "ADD", "sub": "SUBTRACT", "mul": "MULTIPLY", "div": "DIVIDE", "min": "MINIMUM", "max": "MAXIMUM",
        "mod": "FLOORED_MODULO", "pow": "POWER", "abs": "ABSOLUTE", "sqrt": "SQRT", "sin": "SINE", "cos": "COSINE",
        "fract": "FRACT", "floor": "FLOOR", "atan2": "ARCTAN2", "pingpong": "PINGPONG"}
BLEND = {"add": "ADD", "screen": "SCREEN", "mul": "MULTIPLY", "mix": "MIX"}
MAX_RAMP_STOPS = 32


class Compiler:
    def __init__(self, nt, palette, x, y, origin=(-3200, 800)):
        self.nt, self.pal = nt, palette
        self.val = {}                                   # id(expr) -> float | rgba tuple | output socket
        self.vec = {}                                   # (id(ex), id(ey)) -> Combine XYZ output
        self.xf_nodes = {}                              # xf key -> (Mapping node, Separate XYZ node)
        self.x, self.y = x, y
        self.origin = origin
        self.count = 0

    # ---- placement and sockets
    def _new(self, kind, **attrs):
        n = self.nt.nodes.new(kind)
        n.location = (self.origin[0] + 220 * (self.count % 40), self.origin[1] - 70 * (self.count // 40))
        self.count += 1
        for k, v in attrs.items():
            setattr(n, k, v)
        return n

    def _set(self, inp, v):
        if isinstance(v, (int, float)):
            inp.default_value = float(v)
        elif isinstance(v, tuple):
            inp.default_value = v
        else:
            self.nt.links.new(v, inp)

    def _vec(self, ex, ey):
        """Vector socket (ex, ey, 0): the Mapping output when both are the two outputs of one xf, else a Combine."""
        if (ex.op == "xf" and ey.op == "xf" and ex.args == ey.args and ex.attrs == ey.attrs
                and ex.out == 0 and ey.out == 1):
            return self.xf_nodes[self._xfkey(ex)][0].outputs[0]
        key = (id(ex), id(ey))
        if key not in self.vec:
            n = self._new("ShaderNodeCombineXYZ")
            self._set(n.inputs[0], self.val[id(ex)])
            self._set(n.inputs[1], self.val[id(ey)])
            self.vec[key] = n.outputs[0]
        return self.vec[key]

    @staticmethod
    def _xfkey(e):
        return (tuple(id(a) for a in e.args), e.attrs)

    # ---- the build
    def compile(self, *roots):
        """Sockets (or constants) of the roots, building every node they need."""
        for e in topo(roots):
            if id(e) not in self.val:
                self.val[id(e)] = self._node(e)
        return [self.val[id(r)] for r in roots]

    def socket(self, e):
        """Output socket of an expression (a constant is turned into a Value / RGB node)."""
        v = self.compile(e)[0]
        if isinstance(v, (int, float)):
            n = self._new("ShaderNodeValue")
            n.outputs[0].default_value = float(v)
            return n.outputs[0]
        if isinstance(v, tuple):
            n = self._new("ShaderNodeRGB")
            n.outputs[0].default_value = v
            return n.outputs[0]
        return v

    def _node(self, e):
        if e.kind == "c":
            return self._colour(e)
        op = e.op
        if op == "const":
            return e.attrs[0]
        if op == "x":
            return self.x
        if op == "y":
            return self.y
        a = [self.val[id(x)] for x in e.args]
        if op in MATH:
            n = self._new("ShaderNodeMath", operation=MATH[op])
            for i, v in enumerate(a):
                self._set(n.inputs[i], v)
            if op == "pingpong":
                n.inputs[1].default_value = e.attrs[0]
            return n.outputs[0]
        if op == "remap":
            f0, f1, t0, t1, smooth = e.attrs
            n = self._new("ShaderNodeMapRange", clamp=True, interpolation_type="SMOOTHSTEP" if smooth else "LINEAR")
            self._set(n.inputs[0], a[0])
            for i, v in zip((1, 2, 3, 4), (f0, f1, t0, t1)):
                n.inputs[i].default_value = v
            return n.outputs[0]
        if op == "ramp":
            return self._ramp(e, a[0])
        if op == "xf":
            key = self._xfkey(e)
            if key not in self.xf_nodes:
                cx, cy, rot, sx, sy = e.attrs
                m = self._new("ShaderNodeMapping", vector_type="TEXTURE")
                self.nt.links.new(self._vec(e.args[0], e.args[1]), m.inputs[0])
                m.inputs["Location"].default_value = (cx, cy, 0.0)
                m.inputs["Rotation"].default_value = (0.0, 0.0, rot)
                m.inputs["Scale"].default_value = (sx, sy, 1.0)
                s = self._new("ShaderNodeSeparateXYZ")
                self.nt.links.new(m.outputs[0], s.inputs[0])
                self.xf_nodes[key] = (m, s)
            return self.xf_nodes[key][1].outputs[e.out]
        if op == "len2":
            n = self._new("ShaderNodeVectorMath", operation="LENGTH")
            self.nt.links.new(self._vec(e.args[0], e.args[1]), n.inputs[0])
            return n.outputs["Value"]
        if op == "dot2":
            n = self._new("ShaderNodeVectorMath", operation="DOT_PRODUCT")
            self.nt.links.new(self._vec(e.args[0], e.args[1]), n.inputs[0])
            n.inputs[1].default_value = (e.attrs[0], e.attrs[1], 0.0)
            return n.outputs["Value"]
        raise ValueError(f"cannot compile scalar op {op!r}")

    def _ramp(self, e, fac):
        stops, interp = e.attrs
        if len(stops) > MAX_RAMP_STOPS:
            raise ValueError(f"colour ramp with {len(stops)} stops (Blender allows {MAX_RAMP_STOPS})")
        vals = [v for _, v in stops]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span <= 1e-12:
            return lo
        norm = lambda v: (v - lo) / span
        n = self._new("ShaderNodeValToRGB")
        self._fill_ramp(n, [(p, (norm(v),) * 3 + (norm(v),)) for p, v in stops], interp)
        self._set(n.inputs[0], fac)
        out = n.outputs["Alpha"]
        if span > 1e-12 and (lo != 0.0 or hi != 1.0):
            m = self._new("ShaderNodeMath", operation="MULTIPLY_ADD")
            self.nt.links.new(out, m.inputs[0])
            m.inputs[1].default_value = span
            m.inputs[2].default_value = lo
            out = m.outputs[0]
        return out

    @staticmethod
    def _fill_ramp(n, stops, interp):
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        cr.elements[0].position = stops[0][0]
        cr.elements[0].color = stops[0][1]
        for p, c in stops[1:]:
            el = cr.elements.new(max(p, cr.elements[-1].position + 1e-6))
            el.color = c

    def _colour(self, e):
        op = e.op
        if op == "const":
            return (*self.pal.rgb(e.attrs), 1.0)
        a = [self.val[id(x)] for x in e.args]
        if op in BLEND:
            n = self._new("ShaderNodeMixRGB", blend_type=BLEND[op], use_clamp=False)
            self._set(n.inputs[0], a[2])
            self._set(n.inputs[1], a[0])
            self._set(n.inputs[2], a[1])
            return n.outputs[0]
        if op == "scale":
            n = self._new("ShaderNodeVectorMath", operation="SCALE")
            self._set(n.inputs[0], tuple(a[0][:3]) if isinstance(a[0], tuple) else a[0])
            self._set(n.inputs[3], a[1])
            return n.outputs[0]
        if op == "ramp":
            stops, interp = e.attrs
            if len(stops) > MAX_RAMP_STOPS:
                raise ValueError(f"colour ramp with {len(stops)} stops (Blender allows {MAX_RAMP_STOPS})")
            n = self._new("ShaderNodeValToRGB")
            self._fill_ramp(n, [(p, (*self.pal.rgb(att), 1.0)) for p, att in stops], interp)
            self._set(n.inputs[0], a[0])
            return n.outputs["Color"]
        raise ValueError(f"cannot compile colour op {op!r}")


# ================================================================= materials
# Neutral shading factors (not palette colours): a faint warm cast of the paper and the lightness of its fibre grain.
PAPER_TINT = (1.0, 0.985, 0.955, 1.0)
GRAIN_DARK, GRAIN_LIGHT = (0.955, 0.95, 0.94, 1.0), (1.0, 1.0, 1.0, 1.0)


def poster_material(K, name, pal, fin, w, h, margin):
    """The printed sheet's material. `fin` = the finished design's expressions (`bedroom_decor_prints.finish`):
    albedo (colour), rough, spec, height (-1 in a crease groove, 0 elsewhere) and alpha (torn corners). The design's
    coordinates come from the sheet's UV, mapped so that the printed area (the sheet less `margin` all around) is
    [-aspect/2, aspect/2] x [-0.5, 0.5]. On top of the print: a warm cast, paper fibre grain (colour and bump) and the
    creases' relief."""
    m, nt, out = K.new_mat(name)
    uv = N(nt, "ShaderNodeTexCoord", (-3600, 0))
    s = h - 2.0 * margin
    mp = N(nt, "ShaderNodeMapping", (-3400, 0), vector_type="TEXTURE",
           inputs={"Location": (0.5, 0.5, 0.0), "Scale": (s / w, s / h, 1.0)})
    L(nt, uv.outputs["UV"], mp.inputs["Vector"])
    sp = N(nt, "ShaderNodeSeparateXYZ", (-3200, 0))
    L(nt, mp.outputs[0], sp.inputs[0])
    comp = Compiler(nt, pal, sp.outputs["X"], sp.outputs["Y"])
    keys = ("albedo", "rough", "spec", "height", "alpha")
    sk = dict(zip(keys, (comp.socket(fin[k]) for k in keys)))
    mm = N(nt, "ShaderNodeMapping", (-3400, -700), inputs={"Scale": (w * 1000.0, h * 1000.0, 1.0)})      # 1 unit = 1 mm
    L(nt, uv.outputs["UV"], mm.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", (-3200, -700), inputs={"Scale": 1.3, "Detail": 7.0, "Roughness": 0.65})
    L(nt, mm.outputs[0], nz.inputs["Vector"])
    gr = ramp(nt, (-2950, -700), [(0.30, GRAIN_DARK), (0.70, GRAIN_LIGHT)])
    L(nt, nz.outputs["Fac"], gr.inputs["Fac"])
    tint = N(nt, "ShaderNodeMixRGB", (-2700, -300), blend_type="MULTIPLY", inputs={"Fac": 1.0, "Color2": PAPER_TINT})
    L(nt, sk["albedo"], tint.inputs["Color1"])
    fib = N(nt, "ShaderNodeMixRGB", (-2450, -300), blend_type="MULTIPLY", inputs={"Fac": 1.0})
    L(nt, tint.outputs["Color"], fib.inputs["Color1"])
    L(nt, gr.outputs["Color"], fib.inputs["Color2"])
    hgt = N(nt, "ShaderNodeMath", (-2700, -900), operation="MULTIPLY_ADD", inputs={1: 1.5})       # fibres + creases
    L(nt, sk["height"], hgt.inputs[0])
    L(nt, nz.outputs["Fac"], hgt.inputs[2])
    bp = N(nt, "ShaderNodeBump", (-2450, -900), inputs={"Strength": 0.3, "Distance": 0.0005})
    L(nt, hgt.outputs[0], bp.inputs["Height"])
    b = principled(nt, out, loc=(-2100, 0), **{"Sheen Weight": 0.05, "Sheen Roughness": 0.6})
    L(nt, fib.outputs["Color"], b.inputs["Base Color"])
    L(nt, sk["rough"], b.inputs["Roughness"])
    L(nt, sk["spec"], b.inputs["Specular IOR Level"])
    L(nt, sk["alpha"], b.inputs["Alpha"])
    L(nt, bp.outputs["Normal"], b.inputs["Normal"])
    return m


def pattern_material(K, name, pal, colour, plane="xy", rough=0.4, coat=0.0, spec=0.5, bump=0.0):
    """A Principled material whose base colour is the colour expression `colour` evaluated on the object's own
    coordinates (metres): design (X, Y) = the object's (x, y) | (x, z) | (y, z) for plane "xy" | "xz" | "yz"."""
    m, nt, out = K.new_mat(name)
    tc = N(nt, "ShaderNodeTexCoord", (-3400, 0))
    sp = N(nt, "ShaderNodeSeparateXYZ", (-3200, 0))
    L(nt, tc.outputs["Object"], sp.inputs[0])
    ax = {"x": "X", "y": "Y", "z": "Z"}
    comp = Compiler(nt, pal, sp.outputs[ax[plane[0]]], sp.outputs[ax[plane[1]]])
    b = principled(nt, out, loc=(-2400, 0), **{"Roughness": rough, "Coat Weight": coat, "Coat Roughness": 0.2,
                                               "Specular IOR Level": spec})
    L(nt, comp.socket(colour), b.inputs["Base Color"])
    if bump:
        nz = N(nt, "ShaderNodeTexNoise", (-3000, -600), inputs={"Scale": 700.0, "Detail": 3.0, "Roughness": 0.6})
        L(nt, tc.outputs["Object"], nz.inputs["Vector"])
        bp = N(nt, "ShaderNodeBump", (-2700, -600), inputs={"Strength": bump, "Distance": 0.0004})
        L(nt, nz.outputs["Fac"], bp.inputs["Height"])
        L(nt, bp.outputs["Normal"], b.inputs["Normal"])
    return m
