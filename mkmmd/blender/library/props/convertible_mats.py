"""Materials of the 1980s convertible (Blender side). One Blender material per role of `convertible_layout.MATS`,
coloured from the project's palette by role (see `Pal`), plus `paint_flat` (the body colour without the two-tone and
pinstripe masks, for the parts that move: those have their own object space).

Animated state reaches the shaders through one node group, `<name>_params`, whose Value nodes are driven by the custom
properties on the prop root (`lamps`, `brake`, `tails`, `dash_on`, `bars`): keying the property animates every material
that reads it, with no per-material drivers.

Colour roles are in `convertible_palette` (`[[prop]] slots = {body = "iris"}` recolours).
"""
import bpy
import numpy as np

from . import convertible_dash as DASH
from . import convertible_layout as LAY
from .cafe_kit import drive
from .convertible_palette import rgba

# the two-tone line, the pinstripe and the body-side heights (object space z of the body)
TWO_TONE_Z = LAY.Z_RUB - 0.01               # the paint splits on the ramp of the rub strip crease (the body's lower shelf)
PIN_Z = (LAY.Z_RUB + 0.065, LAY.Z_RUB + 0.078)
PIN_W = 0.0021                              # half width of one pinstripe line
FLOOR = 0.17                                # unlit surfaces glow this share of their colour: nothing renders black
VFD_LIT = 4.5                               # emission of a lit display cell (displays at ~0.9 under AgX)
VFD_OFF, VFD_GHOST = 0.012, 0.030           # an unlit cell: a floor, plus this much while the dash is on. Together 0.042
                                            # of the dim `display`+`subtle` blend: ~9 % of a lit cell's displayed brightness


# ===================================================================================================================
# node helper
# ===================================================================================================================
class NG:
    """Tiny shader-node builder: every method returns an output socket; python numbers and tuples become socket
    defaults, sockets get linked."""

    def __init__(self, nt, params=None):
        self.nt, self.params, self._grp, self.i = nt, params, None, 0

    def node(self, idname, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (-200 * (self.i % 24), -160 * (self.i // 24))
        self.i += 1
        return n

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif sock.type == "VECTOR":
            sock.default_value = (v, v, v) if isinstance(v, (int, float)) else tuple(v)[:3]
        elif sock.type == "RGBA":
            v = (v, v, v, 1.0) if isinstance(v, (int, float)) else tuple(v)
            sock.default_value = v if len(v) == 4 else v + (1.0,)
        else:
            sock.default_value = v

    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op)
        n.use_clamp = clamp
        for s, v in zip(n.inputs, (a, b, c)):
            self.put(s, v)
        return n.outputs[0]

    def add(self, a, b): return self.m("ADD", a, b)
    def sub(self, a, b): return self.m("SUBTRACT", a, b)
    def mul(self, a, b): return self.m("MULTIPLY", a, b)
    def mad(self, a, b, c): return self.m("MULTIPLY_ADD", a, b, c)
    def mx(self, a, b): return self.m("MAXIMUM", a, b)
    def mn(self, a, b): return self.m("MINIMUM", a, b)
    def lt(self, a, b): return self.m("LESS_THAN", a, b)
    def ab(self, a): return self.m("ABSOLUTE", a)
    def inv(self, a): return self.m("SUBTRACT", 1.0, a)
    def sat(self, a): return self.m("ADD", a, 0.0, clamp=True)

    def vm(self, op, a, b=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        return n.outputs[1] if op in ("DOT_PRODUCT", "DISTANCE", "LENGTH") else n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.put(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def mixc(self, a, b, fac):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
        self.put(n.inputs[0], fac)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def maprange(self, v, a0, a1, b0=0.0, b1=1.0, interp="LINEAR", clamp=True):
        n = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        for s, x in zip(n.inputs[1:5], (a0, a1, b0, b1)):
            self.put(s, x)
        self.put(n.inputs[0], v)
        return n.outputs[0]

    def ss(self, v, lo, hi):
        return self.maprange(v, lo, hi, 0.0, 1.0, "SMOOTHSTEP")

    def emission(self, color, strength):
        n = self.node("ShaderNodeEmission")
        self.put(n.inputs["Color"], color)
        self.put(n.inputs["Strength"], strength)
        return n

    def hsv(self, col, hue=0.5, sat=1.0, val=1.0):
        n = self.node("ShaderNodeHueSaturation")
        self.put(n.inputs["Hue"], hue)
        self.put(n.inputs["Saturation"], sat)
        self.put(n.inputs["Value"], val)
        self.put(n.inputs["Color"], col)
        return n.outputs[0]

    def texco(self, which="Object"):
        return self.node("ShaderNodeTexCoord").outputs[which]

    def param(self, name):
        if self._grp is None:
            self._grp = self.node("ShaderNodeGroup")
            self._grp.node_tree = self.params
        return self._grp.outputs[name]

    def principled(self, base, rough=0.5, metal=0.0, **extra):
        n = self.node("ShaderNodeBsdfPrincipled")
        self.put(n.inputs["Base Color"], base)
        self.put(n.inputs["Roughness"], rough)
        self.put(n.inputs["Metallic"], metal)
        for k, v in extra.items():
            self.put(n.inputs[k.replace("_", " ")], v)
        return n

    def finish(self, shader):
        out = self.node("ShaderNodeOutputMaterial")
        self.nt.links.new(shader, out.inputs["Surface"])
        return out


def params_group(name, root, names):
    """Node group `<name>_params` with one float output per custom property of the root; each is driven by it."""
    gname = f"{name}_params"
    old = bpy.data.node_groups.get(gname)
    if old is not None:
        bpy.data.node_groups.remove(old)
    g = bpy.data.node_groups.new(gname, "ShaderNodeTree")
    out = g.nodes.new("NodeGroupOutput")
    out.location = (300, 0)
    for i, p in enumerate(names):
        g.interface.new_socket(p, in_out="OUTPUT", socket_type="NodeSocketFloat")
        v = g.nodes.new("ShaderNodeValue")
        v.name = v.label = p
        v.location = (0, -60 * i)
        v.outputs[0].default_value = float(root.get(p, 0.0))
        g.links.new(v.outputs[0], out.inputs[p])
        drive(g, f'nodes["{p}"].outputs[0].default_value', "p", var=("p", root, f'["{p}"]'))
    return g


# ===================================================================================================================
# materials
# ===================================================================================================================
class Materials:
    """Builds the car's materials. `get(role)` returns the Blender material of a role (made on first use)."""

    def __init__(self, name, pal, root, param_names):
        self.name, self.pal = name, pal
        self.params = params_group(name, root, param_names)
        self.cache = {}

    def _new(self, base, blend="DITHERED"):
        mname = f"{self.name}_{base}"
        old = bpy.data.materials.get(mname)
        if old is not None:
            bpy.data.materials.remove(old)
        m = bpy.data.materials.new(mname)
        m.use_nodes = True
        for n in list(m.node_tree.nodes):
            m.node_tree.nodes.remove(n)
        m.surface_render_method = blend
        return m, NG(m.node_tree, self.params)

    def get(self, role):
        if role not in self.cache:
            self.cache[role] = getattr(self, f"m_{role}")()
        return self.cache[role]

    def for_object(self, role, moving=False):
        """The material for a role on an object; moving objects get the flat paint (no object-space masks)."""
        return self.get("paint_flat") if (moving and role == "paint") else self.get(role)

    # ---- helpers
    def _simple(self, base, col, rough=0.5, metal=0.0, floor=FLOOR, **extra):
        m, g = self._new(base)
        b = g.principled(rgba(col), rough, metal, **extra)
        self._lift(g, b, col, floor)
        g.finish(b.outputs[0])
        return m

    def _lift(self, g, bsdf, col, floor=FLOOR):
        """The palette has no black: a faint emission keeps shadows above it. Its colour is the surface's own colour
        lifted halfway toward the palette's `hl_med`, so even the darkest slot (rubber, tyres, wheel wells) shows."""
        if floor > 0:
            lifted = self.pal.mix(tuple(col)[:3], "hl_med")
            g.put(bsdf.inputs["Emission Color"], rgba(lifted))
            g.put(bsdf.inputs["Emission Strength"], floor)

    def _emissive(self, base, col, strength, rough=0.4, base_col=None, **extra):
        """Principled material that emits `col` at `strength` (a float or a socket made with g)."""
        m, g = self._new(base)
        bsdf = g.principled(rgba(base_col or col), rough, 0.0, Emission_Color=rgba(col), **extra)
        return m, g, bsdf

    # ---- body
    def _paint(self, base, masks):
        p = self.pal
        m, g = self._new(base)
        col = rgba(p.lin("body"))
        if masks:
            z = g.sep(g.texco("Object"))[2]
            low = g.maprange(z, TWO_TONE_Z + 0.0012, TWO_TONE_Z - 0.0012, 0.0, 1.0)
            col = g.mixc(col, rgba(p.lin("lower")), low)
            lines = None
            for zc in PIN_Z:
                s = g.inv(g.ss(g.ab(g.sub(z, zc)), PIN_W - 0.0005, PIN_W + 0.0005))
                lines = s if lines is None else g.mx(lines, s)
            col = g.mixc(col, rgba(p.lin("stripe")), lines)
        b = g.principled(col, 0.38, 0.30, Coat_Weight=1.0, Coat_Roughness=0.03, Specular_IOR_Level=0.6)
        g.put(b.inputs["Emission Color"], col)                       # the floor: shadows keep the paint's colour
        g.put(b.inputs["Emission Strength"], 0.10)
        g.finish(b.outputs[0])
        return m

    def m_paint(self):
        return self._paint("paint", True)

    def m_paint_flat(self):
        return self._paint("paint_flat", False)

    def m_bumper(self):
        p = self.pal
        m, g = self._new("bumper")
        col = rgba(p.lin("lower"))
        b = g.principled(col, 0.46, 0.15, Coat_Weight=0.5, Coat_Roughness=0.1, Specular_IOR_Level=0.5)
        g.put(b.inputs["Emission Color"], col)
        g.put(b.inputs["Emission Strength"], 0.10)
        g.finish(b.outputs[0])
        return m

    def m_underbody(self):
        return self._simple("underbody", self.pal.mix("base", ("overlay", 0.15)), 0.85, 0.0, floor=0.3,
                            Specular_IOR_Level=0.2)

    def m_chrome(self):
        return self._simple("chrome", self.pal.lin("trim"), 0.14, 1.0, floor=0.05)

    def m_rubber(self):
        return self._simple("rubber", self.pal.mix("rubber", ("overlay", 0.22)), 0.58, 0.0, Specular_IOR_Level=0.35)

    def m_seam(self):
        return self._simple("seam", self.pal.mix("rubber", ("overlay", 0.1)), 0.8, 0.0, Specular_IOR_Level=0.15)

    def m_tyre(self):
        return self._simple("tyre", self.pal.mix("rubber", ("overlay", 0.35)), 0.78, 0.0, Specular_IOR_Level=0.3)

    def m_alloy(self):
        return self._simple("alloy", self.pal.mix("wheel", ("text", 0.35)), 0.3, 0.9, floor=0.08)

    def m_mirror_glass(self):
        return self._simple("mirror", self.pal.mix("subtle", ("text", 0.4)), 0.04, 1.0, floor=0.04)

    # ---- glass and lamps
    def m_glass(self):
        m, g = self._new("glass", blend="BLENDED")
        m.use_transparency_overlap = False
        tint = self.pal.mix("glass", ("text", 1.2), ("base", 0.2))
        clear = g.node("ShaderNodeBsdfTransparent")
        g.put(clear.inputs[0], rgba(tint))
        gloss = g.node("ShaderNodeBsdfGlossy")
        g.put(gloss.inputs["Roughness"], 0.04)
        g.put(gloss.inputs["Color"], (1.0, 1.0, 1.0, 1.0))
        fres = g.node("ShaderNodeFresnel")
        g.put(fres.inputs["IOR"], 1.45)
        mix = g.node("ShaderNodeMixShader")
        fac = g.mad(fres.outputs[0], 0.85, 0.12)                    # always a little reflective
        g.put(mix.inputs[0], fac)
        g.nt.links.new(clear.outputs[0], mix.inputs[1])
        g.nt.links.new(gloss.outputs[0], mix.inputs[2])
        g.finish(mix.outputs[0])
        return m

    def m_lens_head(self):
        p = self.pal
        m, g = self._new("lens_head")
        on = g.ss(g.param("lamps"), 0.15, 0.9)                                   # the warm-up flicker, as the lights have it
        flick = g.sub(1.0, g.mul(0.3, g.mul(g.inv(on), g.ab(g.m("SINE", g.mul(g.param("lamps"), 70.0))))))
        strength = g.mad(g.mul(on, flick), 3.2, 0.1)
        b = g.principled(rgba(p.mix("headlamp", ("glass", 0.15))), 0.08, 0.0, Emission_Color=rgba(p.lin("headlamp")),
                         Emission_Strength=strength, Specular_IOR_Level=0.8)
        g.finish(b.outputs[0])
        return m

    def m_lamp_tail(self):
        p = self.pal
        m, g = self._new("lamp_tail")
        strength = g.add(g.mad(g.param("tails"), 2.4, 0.3), g.mul(g.param("brake"), 9.0))
        lens = g.hsv(rgba(p.mix("taillamp", ("base", 0.55))), hue=0.525, sat=1.4, val=0.9)       # a deeper, redder lens
        glow = g.hsv(rgba(p.lin("taillamp")), hue=0.525, sat=1.4, val=1.0)
        b = g.principled(lens, 0.18, 0.0, Emission_Color=glow, Emission_Strength=strength, Specular_IOR_Level=0.8)
        g.finish(b.outputs[0])
        return m

    def m_lamp_amber(self):
        p = self.pal
        m, g = self._new("lamp_amber")
        strength = g.mad(g.param("tails"), 0.9, 0.12)
        b = g.principled(rgba(p.mix("gold", ("base", 0.3))), 0.2, 0.0, Emission_Color=rgba(p.lin("gold")),
                         Emission_Strength=strength, Specular_IOR_Level=0.7)
        g.finish(b.outputs[0])
        return m

    def m_plate(self):
        return self._simple("plate", self.pal.mix("text", ("subtle", 0.25)), 0.5, 0.0)

    # ---- interior
    def m_vinyl(self):
        p = self.pal
        m, g = self._new("vinyl")
        grain = g.node("ShaderNodeTexNoise", noise_dimensions="3D")
        g.put(grain.inputs["Vector"], g.texco("Object"))
        g.put(grain.inputs["Scale"], 380.0)
        g.put(grain.inputs["Detail"], 1.0)
        bump = g.node("ShaderNodeBump")
        g.put(bump.inputs["Strength"], 0.12)
        g.put(bump.inputs["Distance"], 0.002)
        g.nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
        b = g.principled(rgba(p.lin("interior")), 0.52, 0.0, Sheen_Weight=0.35, Specular_IOR_Level=0.45)
        self._lift(g, b, p.lin("interior"), 0.07)
        g.nt.links.new(bump.outputs[0], b.inputs["Normal"])
        g.finish(b.outputs[0])
        return m

    def m_vinyl_dark(self):
        p = self.pal
        return self._simple("vinyl_dark", p.mix("interior", ("base", 0.55)), 0.68, 0.0, Sheen_Weight=0.2)

    def m_carpet(self):
        p = self.pal
        m, g = self._new("carpet")
        grain = g.node("ShaderNodeTexNoise", noise_dimensions="3D")
        g.put(grain.inputs["Vector"], g.texco("Object"))
        g.put(grain.inputs["Scale"], 520.0)
        g.put(grain.inputs["Detail"], 2.0)
        bump = g.node("ShaderNodeBump")
        g.put(bump.inputs["Strength"], 0.2)
        g.put(bump.inputs["Distance"], 0.002)
        g.nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
        carpet = p.mix("interior", ("base", 1.1))
        b = g.principled(rgba(carpet), 1.0, 0.0, Specular_IOR_Level=0.1, Sheen_Weight=0.5)
        self._lift(g, b, carpet, 0.12)
        g.nt.links.new(bump.outputs[0], b.inputs["Normal"])
        g.finish(b.outputs[0])
        return m

    def m_trim_dark(self):
        return self._simple("trim_dark", self.pal.mix("rubber", ("overlay", 0.6)), 0.45, 0.0, Specular_IOR_Level=0.4)

    def m_wood(self):
        p = self.pal
        m, g = self._new("wood")
        wave = g.node("ShaderNodeTexWave", wave_type="RINGS", rings_direction="Z", wave_profile="SIN")
        g.put(wave.inputs["Vector"], g.texco("Object"))
        g.put(wave.inputs["Scale"], 18.0)
        g.put(wave.inputs["Distortion"], 7.0)
        g.put(wave.inputs["Detail"], 2.0)
        g.put(wave.inputs["Detail Scale"], 2.2)
        g.put(wave.inputs["Detail Roughness"], 0.6)
        light = rgba(p.mix("gold", ("rose", 0.35), ("base", 1.2)))
        dark = rgba(p.mix("gold", ("base", 2.6)))
        col = g.mixc(dark, light, g.ss(wave.outputs["Fac"], 0.15, 0.85))
        b = g.principled(col, 0.34, 0.0, Coat_Weight=0.6, Coat_Roughness=0.08, Specular_IOR_Level=0.5)
        g.put(b.inputs["Emission Color"], col)
        g.put(b.inputs["Emission Strength"], 0.10)
        g.finish(b.outputs[0])
        return m

    def m_steering(self):
        return self._simple("steering", self.pal.mix("rubber", ("overlay", 0.7)), 0.5, 0.0, Specular_IOR_Level=0.4)

    def m_deck_face(self):
        return self._simple("deck_face", self.pal.mix("rubber", ("overlay", 0.9), ("subtle", 0.2)), 0.34, 0.65)

    def m_boot(self):
        return self._simple("boot", self.pal.lin("boot"), 0.82, 0.0, Sheen_Weight=0.6, Specular_IOR_Level=0.3)

    def m_cassette(self):
        return self._simple("cassette", self.pal.mix("overlay", ("base", 0.7)), 0.4, 0.0)

    def m_label(self):
        return self._simple("label", self.pal.mix("text", ("rose", 0.12)), 0.6, 0.0)

    def m_stripe_a(self):
        return self._simple("stripe_a", self.pal.lin("rose"), 0.6, 0.0)

    def m_stripe_b(self):
        return self._simple("stripe_b", self.pal.lin("gold"), 0.6, 0.0)

    def m_stripe_c(self):
        return self._simple("stripe_c", self.pal.lin("pine"), 0.6, 0.0)

    # ---- displays
    def _vfd(self, base, bar=False):
        """A display cell: pure emission (no reflection). An unlit cell is a faint hint (VFD_OFF + VFD_GHOST x dash_on of
        a darker, less saturated colour); a lit one is `display` at VFD_LIT x dash_on. `vfd_ghost` cells are always
        unlit; `vfd_lit` cells are lit, except the cluster's bar graph, whose cells light up to the `bars` level."""
        p = self.pal
        m, g = self._new(base)
        on = g.param("dash_on")
        dim_col = rgba(p.mix("display", "subtle"))                       # darker and less saturated than a lit cell
        off = g.add(g.mul(on, VFD_GHOST), VFD_OFF)
        if not bar:
            em = g.emission(dim_col, off)
        else:
            # the cluster's bar graph: cells in a row (convertible_dash.BAR_GRAPH); a cell is lit when it lies to the
            # left of the `bars` level along the row. Everything else that uses this role is always lit.
            bg = DASH.BAR_GRAPH
            first, last = np.array(bg["first"]), np.array(bg["last"])
            axis = np.array(bg["axis"])
            pitch = np.linalg.norm(last - first) / (bg["cells"] - 1)
            length = np.linalg.norm(last - first) + pitch
            centre = (first + last) / 2
            nrm, up, _ = LAY.CLUSTER_FRAME
            q = g.vm("SUBTRACT", g.texco("Object"), tuple(float(v) for v in centre))
            t = g.add(g.mul(g.vm("DOT_PRODUCT", q, tuple(float(v) for v in axis)), 1.0 / float(length)), 0.5)
            dv = g.ab(g.vm("DOT_PRODUCT", q, tuple(float(v) for v in up)))
            dn = g.ab(g.vm("DOT_PRODUCT", q, tuple(float(v) for v in nrm)))
            inside = g.mul(g.mul(g.lt(dv, 0.012), g.lt(dn, 0.012)), g.lt(g.ab(g.sub(t, 0.5)), 0.56))
            lit = g.mx(g.inv(inside), g.lt(t, g.param("bars")))
            on_strength = g.mx(off, g.mul(on, VFD_LIT))
            strength = g.add(off, g.mul(lit, g.sub(on_strength, off)))
            em = g.emission(g.mixc(dim_col, rgba(p.lin("display")), lit), strength)
        # a cell sits on the glass: lit like the glass (a plain diffuse of its colour: no reflection) plus its own glow,
        # so an unlit cell is the glass plus a faint hint in any light
        body = g.node("ShaderNodeBsdfDiffuse")
        g.put(body.inputs["Color"], rgba(p.mix("base", ("display", 0.05))))
        g.put(body.inputs["Roughness"], 1.0)
        both = g.node("ShaderNodeAddShader")
        g.nt.links.new(body.outputs[0], both.inputs[0])
        g.nt.links.new(em.outputs[0], both.inputs[1])
        g.finish(both.outputs[0])
        return m

    def m_vfd_ghost(self):
        return self._vfd("vfd_ghost")

    def m_vfd_lit(self):
        return self._vfd("vfd_lit", bar=True)

    def m_display_glass(self):
        """Smoked, matte-coated glass in front of the cells: dark, with only a faint sheen, so the unlit cells (a faint
        hint) stand out of it instead of being drowned in a mirror image of the sky."""
        p = self.pal
        m, g = self._new("display_glass")
        glow = g.mul(g.param("dash_on"), 0.012)                       # a dim backlight
        b = g.principled(rgba(p.mix("base", ("display", 0.05))), 0.45, 0.0, Emission_Color=rgba(p.lin("display")),
                         Emission_Strength=glow, Specular_IOR_Level=0.2)
        g.finish(b.outputs[0])
        return m

    def _indicator(self, base, slot):
        p = self.pal
        m, g = self._new(base)
        strength = g.mad(g.param("dash_on"), 2.0, 0.03)
        b = g.principled(rgba(p.mix(slot, ("base", 1.2))), 0.4, 0.0, Emission_Color=rgba(p.lin(slot)),
                         Emission_Strength=strength)
        g.finish(b.outputs[0])
        return m

    def m_ind_a(self):
        return self._indicator("ind_a", "gold")

    def m_ind_b(self):
        return self._indicator("ind_b", "pine")

    def m_ind_c(self):
        return self._indicator("ind_c", "love")
