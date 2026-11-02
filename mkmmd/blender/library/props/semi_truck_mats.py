"""Materials of the semi truck (Blender side). One Blender material per role of `semi_truck_layout.MATS`, `<name>_<role>`,
coloured from the palette by the colour roles of `semi_truck_layout.ROLES` (`convertible_palette.Pal`): every colour is a
node socket value resolved from the slots the props stage hands the builder, and the node trees never branch on a colour,
so a breathing palette drives them all.

Animated state reaches the shaders through one node group, `<name>_params`, whose values are driven by the root's custom
properties (`lamps`, `markers`, `brake`, `dash_on`, `dome`; `semi_truck_layout.PARAMS`).

Nothing renders black: every surface glows a small share of its own colour lifted toward `hl_med` (a floor), and polished
metal glows by its angle to the viewer (a matcap-like floor), so chrome keeps its form at night where it reflects nothing.
"""
import bpy

from . import semi_truck_layout as LAY
from .convertible_mats import NG, params_group
from .convertible_palette import rgba

FLOOR = 0.12                      # unlit surfaces glow this share of their colour
SHIELD_PITCH, SHIELD_HOLE_R = 0.022, 0.0062   # the heat shields' holes (metres on the shields' uv)


class Materials:
    """The truck's materials, made on first use. `get(role)` is the material of a role of LAY.MATS; `of(i)` of an index."""

    def __init__(self, name, pal, root):
        self.name, self.pal = name, pal
        self.params = params_group(name, root, list(LAY.PARAMS))
        self.cache = {}

    def of(self, index):
        return self.get(LAY.MATS[int(index)])

    def get(self, role):
        if role not in self.cache:
            self.cache[role] = getattr(self, f"m_{role}")()
        return self.cache[role]

    # ---------------------------------------------------------------------------------------------- helpers
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

    def _lift(self, g, bsdf, col, floor=FLOOR):
        """A faint emission of the surface's own colour lifted halfway toward `hl_med`: shadows keep their colour."""
        if floor > 0:
            g.put(bsdf.inputs["Emission Color"], rgba(self.pal.mix(tuple(col)[:3], "hl_med")))
            g.put(bsdf.inputs["Emission Strength"], floor)

    def _simple(self, base, col, rough=0.5, metal=0.0, floor=FLOOR, **extra):
        m, g = self._new(base)
        b = g.principled(rgba(col), rough, metal, **extra)
        self._lift(g, b, col, floor)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def _metal(self, base, col, rough, lo=0.06, hi=0.55, bump=None):
        """Polished metal: a Principled metal whose emission (its own colour) follows the angle to the viewer, brightest
        face-on: a matcap-like floor so curved chrome reads in a dark world."""
        m, g = self._new(base)
        b = g.principled(rgba(col), rough, 1.0, Specular_IOR_Level=0.6)
        lw = g.node("ShaderNodeLayerWeight")
        g.put(lw.inputs["Blend"], 0.5)
        facing = g.m("POWER", g.inv(lw.outputs["Facing"]), 1.6)
        g.put(b.inputs["Emission Color"], rgba(col))
        g.put(b.inputs["Emission Strength"], g.mad(facing, hi - lo, lo))
        if bump is not None:
            g.nt.links.new(bump, b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def _noise_bump(self, g, scale, strength, distance=0.002, detail=2.0):
        n = g.node("ShaderNodeTexNoise", noise_dimensions="3D")
        g.put(n.inputs["Vector"], g.texco("Object"))
        g.put(n.inputs["Scale"], scale)
        g.put(n.inputs["Detail"], detail)
        bump = g.node("ShaderNodeBump")
        g.put(bump.inputs["Strength"], strength)
        g.put(bump.inputs["Distance"], distance)
        g.nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
        return bump.outputs[0]

    def _glow(self, base, face, glow, strength, rough=0.35, floor=0.03, **extra):
        """A surface lit from behind: `face` in the light, plus `glow` at `strength` (a socket) and a small floor."""
        m, g = self._new(base)
        b = g.principled(rgba(face), rough, 0.0, Emission_Color=rgba(glow), **extra)
        g.put(b.inputs["Emission Strength"], g.add(strength, floor) if not isinstance(strength, (int, float)) else strength + floor)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(face)
        return m

    # ---------------------------------------------------------------------------------------------- paint and metal
    def _paint(self, base, role):
        p = self.pal
        m, g = self._new(base)
        col = rgba(p.lin(role))
        b = g.principled(col, 0.34, 0.25, Coat_Weight=1.0, Coat_Roughness=0.06, Specular_IOR_Level=0.6)
        g.put(b.inputs["Emission Color"], col)
        g.put(b.inputs["Emission Strength"], 0.08)
        g.finish(b.outputs[0])
        m.diffuse_color = col
        return m

    def m_paint(self):
        return self._paint("paint", "body")

    def m_paint_accent(self):
        return self._paint("paint_accent", "accent")

    def m_chrome(self):
        return self._metal("chrome", self.pal.mix("trim", ("text", 0.6)), 0.05)

    def m_alloy(self):
        return self._metal("alloy", self.pal.mix("trim", ("subtle", 0.4)), 0.12, lo=0.05, hi=0.45)

    def m_chassis(self):
        return self._simple("chassis", self.pal.mix("chassis", ("overlay", 0.25)), 0.55, 0.3, floor=0.10,
                            Specular_IOR_Level=0.35)

    def m_iron(self):
        p = self.pal
        m, g = self._new("iron")
        col = p.mix("chassis", ("muted", 0.45))
        b = g.principled(rgba(col), 0.62, 0.55, Specular_IOR_Level=0.3)
        self._lift(g, b, col, 0.08)
        g.nt.links.new(self._noise_bump(g, 260.0, 0.15), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_rubber(self):
        return self._simple("rubber", self.pal.mix("rubber", ("overlay", 0.25)), 0.6, 0.0, Specular_IOR_Level=0.3)

    def m_tyre(self):
        p = self.pal
        m, g = self._new("tyre")
        col = p.mix("rubber", ("overlay", 0.35))
        b = g.principled(rgba(col), 0.78, 0.0, Specular_IOR_Level=0.3, Sheen_Weight=0.2)
        self._lift(g, b, col, 0.10)
        g.nt.links.new(self._noise_bump(g, 420.0, 0.12, 0.001), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_mudflap(self):
        return self._simple("mudflap", self.pal.mix("rubber", ("overlay", 0.3)), 0.45, 0.0, Specular_IOR_Level=0.45)

    # ---------------------------------------------------------------------------------------------- glass and lamps
    def m_glass(self):
        """Clear glass: transparent with a slight tint, a soft reflection rising toward grazing angles (the convertible's
        windshield recipe), so the cab's inside reads through it."""
        m, g = self._new("glass", blend="BLENDED")
        m.use_transparency_overlap = True
        m.use_backface_culling = True
        tint = self.pal.mix("glass", ("text", 5.0))
        clear = g.node("ShaderNodeBsdfTransparent")
        g.put(clear.inputs[0], rgba(tint))
        gloss = g.node("ShaderNodeBsdfGlossy")
        g.put(gloss.inputs["Roughness"], 0.25)
        g.put(gloss.inputs["Color"], (1.0, 1.0, 1.0, 1.0))
        fres = g.node("ShaderNodeFresnel")
        g.put(fres.inputs["IOR"], 1.45)
        mix = g.node("ShaderNodeMixShader")
        g.put(mix.inputs[0], g.maprange(fres.outputs[0], 0.08, 1.0, 0.0, 0.45))
        g.nt.links.new(clear.outputs[0], mix.inputs[1])
        g.nt.links.new(gloss.outputs[0], mix.inputs[2])
        g.finish(mix.outputs[0])
        m.diffuse_color = rgba(tint, 0.3)
        return m

    def m_mirror_glass(self):
        return self._metal("mirror_glass", self.pal.mix("subtle", ("text", 0.5)), 0.03, lo=0.04, hi=0.3)

    def m_lens_head(self):
        p = self.pal
        m, g = self._new("lens_head")
        strength = g.mad(g.param("lamps"), 4.0, 0.15)
        b = g.principled(rgba(p.mix("headlamp", ("glass", 0.15))), 0.06, 0.0, Emission_Color=rgba(p.lin("headlamp")),
                         Emission_Strength=strength, Specular_IOR_Level=0.8, Coat_Weight=1.0, Coat_Roughness=0.03)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("headlamp"))
        return m

    def m_lamp_amber(self):
        p = self.pal
        m, g = self._new("lamp_amber")
        strength = g.mad(g.param("markers"), 2.6, 0.12)
        b = g.principled(rgba(p.mix("marker", ("base", 0.3))), 0.15, 0.0, Emission_Color=rgba(p.lin("marker")),
                         Emission_Strength=strength, Specular_IOR_Level=0.7, Coat_Weight=0.8, Coat_Roughness=0.05)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("marker"))
        return m

    def m_lamp_red(self):
        p = self.pal
        m, g = self._new("lamp_red")
        strength = g.add(g.mad(g.param("markers"), 1.8, 0.15), g.mul(g.param("brake"), 8.0))
        lens = g.hsv(rgba(p.mix("taillamp", ("base", 0.5))), hue=0.525, sat=1.4, val=0.9)
        glow = g.hsv(rgba(p.lin("taillamp")), hue=0.525, sat=1.4, val=1.0)
        b = g.principled(lens, 0.15, 0.0, Emission_Color=glow, Emission_Strength=strength, Specular_IOR_Level=0.8,
                         Coat_Weight=0.8, Coat_Roughness=0.05)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("taillamp"))
        return m

    def m_grille_back(self):
        return self._simple("grille_back", self.pal.mix("chassis", ("overlay", 0.4)), 0.5, 0.6, floor=0.06)

    def m_seam(self):
        return self._simple("seam", self.pal.mix("rubber", ("overlay", 0.1)), 0.8, 0.0, floor=0.05, Specular_IOR_Level=0.15)

    # ---------------------------------------------------------------------------------------------- the trailer
    def m_trailer_skin(self):
        p = self.pal
        m, g = self._new("trailer_skin")
        col = p.lin("van")
        b = g.principled(rgba(col), 0.36, 0.05, Coat_Weight=0.5, Coat_Roughness=0.12, Specular_IOR_Level=0.5)
        self._lift(g, b, col, 0.08)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_trailer_metal(self):
        p = self.pal
        m, g = self._new("trailer_metal")
        col = p.mix("van_trim", ("text", 0.35))
        # brushed: a noise stretched along the object's x (rails and posts are drawn along their length)
        n = g.node("ShaderNodeTexNoise", noise_dimensions="3D")
        mp = g.node("ShaderNodeMapping")
        g.put(mp.inputs["Scale"], (6.0, 6.0, 900.0))
        g.nt.links.new(g.texco("Object"), mp.inputs["Vector"])
        g.nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        g.put(n.inputs["Detail"], 3.0)
        rough = g.maprange(n.outputs["Fac"], 0.3, 0.7, 0.22, 0.36)
        b = g.principled(rgba(col), rough, 0.85, Specular_IOR_Level=0.5)
        lw = g.node("ShaderNodeLayerWeight")
        g.put(lw.inputs["Blend"], 0.5)
        g.put(b.inputs["Emission Color"], rgba(col))
        g.put(b.inputs["Emission Strength"], g.mad(g.m("POWER", g.inv(lw.outputs["Facing"]), 1.6), 0.3, 0.06))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def mural(self, side, path, fit, pivot):
        """`<name>_mural_<side>`: the van's wall finish (trailer_skin's) with the image at `path` laid over its panel
        (LAY.SIDE_PANEL for `left` / `right`, LAY.BACK_PANEL for `back`) as a viewer in front of that face sees it, scaled by
        `fit` (LAY.mural_scale). Where the image is transparent or does not reach, the wall's palette colour shows. The
        mapping is by object position in the trailer's frame (the truck's minus `pivot`), so every face given the role
        shows its own part of one continuous picture."""
        img = bpy.data.images.load(str(path), check_existing=True)
        iw, ih = (float(v) for v in img.size)
        if min(iw, ih) < 1:
            raise ValueError(f"semi_truck: the mural {path} has no pixels Blender can read")
        px, py, pz = (float(v) for v in pivot)
        m, g = self._new(f"mural_{side}")
        pos = g.sep(g.texco("Object"))
        if side == "back":
            hw, z0, z1 = LAY.BACK_PANEL
            w, h = 2.0 * hw, z1 - z0
            u = g.mul(g.sub(hw - px, pos[0]), 1.0 / w)               # a viewer behind the van has +X on the left
        else:
            y0, y1, z0, z1 = LAY.SIDE_PANEL
            w, h = y1 - y0, z1 - z0
            u = (g.mul(g.sub(pos[1], y0 - py), 1.0 / w) if side == "left"   # facing +X the rear is on the viewer's right
                 else g.mul(g.sub(y1 - py, pos[1]), 1.0 / w))
        v = g.mul(g.sub(pos[2], z0 - pz), 1.0 / h)
        su, sv = LAY.mural_scale(fit, w / h, iw / ih)
        xyz = g.node("ShaderNodeCombineXYZ")
        g.put(xyz.inputs[0], g.mad(g.sub(u, 0.5), su, 0.5))
        g.put(xyz.inputs[1], g.mad(g.sub(v, 0.5), sv, 0.5))
        tex = g.node("ShaderNodeTexImage", image=img, extension="CLIP", interpolation="Smart")
        g.nt.links.new(xyz.outputs[0], tex.inputs["Vector"])
        skin = self.pal.lin("van")
        col = g.mixc(rgba(skin), tex.outputs["Color"], tex.outputs["Alpha"])
        b = g.principled(col, 0.36, 0.05, Coat_Weight=0.5, Coat_Roughness=0.12, Specular_IOR_Level=0.5)
        g.put(b.inputs["Emission Color"], col)
        g.put(b.inputs["Emission Strength"], 0.08)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(skin)
        return m

    def m_trailer_under(self):
        return self._simple("trailer_under", self.pal.mix("chassis", ("overlay", 0.35), ("muted", 0.1)), 0.6, 0.35,
                            floor=0.09, Specular_IOR_Level=0.3)

    def m_tape_red(self):
        p = self.pal
        return self._simple("tape_red", p.mix("tape", ("base", 0.15)), 0.25, 0.0, floor=0.18, Specular_IOR_Level=0.9)

    def m_tape_white(self):
        return self._simple("tape_white", self.pal.mix("text", ("subtle", 0.2)), 0.25, 0.0, floor=0.18,
                            Specular_IOR_Level=0.9)

    def m_tread_plate(self):
        """Aluminium diamond plate: two sets of slanted lugs (crossed waves) as a bump on a brushed metal."""
        p = self.pal
        m, g = self._new("tread_plate")
        col = p.mix("trim", ("subtle", 0.8))
        co = g.texco("Object")
        lugs = None
        for ang in (0.785, -0.785):
            mp = g.node("ShaderNodeMapping")
            g.put(mp.inputs["Rotation"], (0.0, 0.0, ang))
            g.nt.links.new(co, mp.inputs["Vector"])
            w = g.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", wave_profile="SIN")
            g.nt.links.new(mp.outputs["Vector"], w.inputs["Vector"])
            g.put(w.inputs["Scale"], 22.0)
            s = g.ss(w.outputs["Fac"], 0.72, 0.9)
            lugs = s if lugs is None else g.mul(lugs, s)
        bump = g.node("ShaderNodeBump")
        g.put(bump.inputs["Strength"], 0.6)
        g.put(bump.inputs["Distance"], 0.002)
        g.nt.links.new(lugs, bump.inputs["Height"])
        b = g.principled(rgba(col), 0.28, 0.9, Specular_IOR_Level=0.5)
        g.nt.links.new(bump.outputs[0], b.inputs["Normal"])
        lw = g.node("ShaderNodeLayerWeight")
        g.put(lw.inputs["Blend"], 0.5)
        g.put(b.inputs["Emission Color"], rgba(col))
        g.put(b.inputs["Emission Strength"], g.mad(g.m("POWER", g.inv(lw.outputs["Facing"]), 1.6), 0.3, 0.05))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    # ---------------------------------------------------------------------------------------------- the cab inside
    def m_vinyl(self):
        p = self.pal
        m, g = self._new("vinyl")
        col = p.lin("interior")
        b = g.principled(rgba(col), 0.5, 0.0, Sheen_Weight=0.35, Specular_IOR_Level=0.45)
        self._lift(g, b, col, 0.07)
        g.nt.links.new(self._noise_bump(g, 380.0, 0.12, 0.002, 1.0), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_vinyl_dark(self):
        p = self.pal
        m, g = self._new("vinyl_dark")
        col = p.lin("dash")
        b = g.principled(rgba(col), 0.58, 0.0, Sheen_Weight=0.25, Specular_IOR_Level=0.4)
        self._lift(g, b, col, 0.08)
        g.nt.links.new(self._noise_bump(g, 300.0, 0.08, 0.002, 1.0), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_carpet(self):
        p = self.pal
        m, g = self._new("carpet")
        col = p.mix("dash", ("interior", 0.4))
        b = g.principled(rgba(col), 1.0, 0.0, Specular_IOR_Level=0.1, Sheen_Weight=0.5)
        self._lift(g, b, col, 0.10)
        g.nt.links.new(self._noise_bump(g, 520.0, 0.2, 0.002), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_headliner(self):
        p = self.pal
        m, g = self._new("headliner")
        col = p.mix("interior", ("text", 0.6))
        b = g.principled(rgba(col), 0.9, 0.0, Sheen_Weight=0.6, Specular_IOR_Level=0.2)
        self._lift(g, b, col, 0.08)
        g.nt.links.new(self._noise_bump(g, 200.0, 0.1, 0.002, 1.0), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_gauge_face(self):
        p = self.pal
        m, g = self._new("gauge_face")
        b = g.principled(rgba(p.mix("gauges", ("dash", 0.4))), 0.4, 0.0, Emission_Color=rgba(p.mix("gauges", ("glow", 1.0))),
                         Specular_IOR_Level=0.3)
        g.put(b.inputs["Emission Strength"], g.mad(g.param("dash_on"), 0.9, 0.04))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("gauges"))
        return m

    def m_gauge_needle(self):
        p = self.pal
        m, g = self._new("gauge_needle")
        b = g.principled(rgba(p.lin("glow")), 0.35, 0.0, Emission_Color=rgba(p.lin("glow")))
        g.put(b.inputs["Emission Strength"], g.mad(g.param("dash_on"), 4.0, 0.1))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("glow"))
        return m

    def m_switch(self):
        return self._simple("switch", self.pal.mix("dash", ("base", 1.5)), 0.35, 0.0, floor=0.08, Specular_IOR_Level=0.5)

    def m_plastic_dark(self):
        return self._simple("plastic_dark", self.pal.mix("chassis", ("overlay", 0.5)), 0.42, 0.0, floor=0.08,
                            Specular_IOR_Level=0.45)

    def m_display(self):
        p = self.pal
        m, g = self._new("display")
        b = g.principled(rgba(p.mix("display", ("base", 1.0))), 0.3, 0.0, Emission_Color=rgba(p.lin("display")))
        g.put(b.inputs["Emission Strength"], g.mad(g.param("dash_on"), 5.0, 0.05))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("display"))
        return m

    def m_steering(self):
        p = self.pal
        m, g = self._new("steering")
        col = p.mix("dash", ("base", 0.8))
        b = g.principled(rgba(col), 0.45, 0.0, Specular_IOR_Level=0.45, Sheen_Weight=0.2)
        self._lift(g, b, col, 0.08)
        g.nt.links.new(self._noise_bump(g, 900.0, 0.18, 0.001, 1.0), b.inputs["Normal"])
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_dome(self):
        p = self.pal
        m, g = self._new("dome")
        b = g.principled(rgba(p.mix("glow", ("text", 2.0))), 0.3, 0.0, Emission_Color=rgba(p.mix("glow", ("text", 2.0))))
        g.put(b.inputs["Emission Strength"], g.mad(g.param("dome"), 6.0, 0.05))
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(p.lin("text"))
        return m

    def m_curtain(self):
        p = self.pal
        m, g = self._new("curtain")
        col = p.lin("curtain")
        b = g.principled(rgba(col), 0.85, 0.0, Sheen_Weight=0.8, Specular_IOR_Level=0.15)
        self._lift(g, b, col, 0.08)
        g.finish(b.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_woodgrain(self):
        p = self.pal
        m, g = self._new("woodgrain")
        wave = g.node("ShaderNodeTexWave", wave_type="RINGS", rings_direction="Z", wave_profile="SIN")
        g.put(wave.inputs["Vector"], g.texco("Object"))
        g.put(wave.inputs["Scale"], 16.0)
        g.put(wave.inputs["Distortion"], 8.0)
        g.put(wave.inputs["Detail"], 2.0)
        g.put(wave.inputs["Detail Scale"], 2.2)
        g.put(wave.inputs["Detail Roughness"], 0.6)
        light = rgba(p.lin("wood"))
        dark = rgba(p.mix("wood", ("base", 1.6)))
        col = g.mixc(dark, light, g.ss(wave.outputs["Fac"], 0.15, 0.85))
        b = g.principled(col, 0.3, 0.0, Coat_Weight=0.8, Coat_Roughness=0.06, Specular_IOR_Level=0.5)
        g.put(b.inputs["Emission Color"], col)
        g.put(b.inputs["Emission Strength"], 0.08)
        g.finish(b.outputs[0])
        m.diffuse_color = light
        return m

    def m_underbody(self):
        return self._simple("underbody", self.pal.mix("chassis", ("overlay", 0.15)), 0.85, 0.0, floor=0.2,
                            Specular_IOR_Level=0.2)

    # ---------------------------------------------------------------------------------------------- the exterior's own
    def m_shield(self):
        """The stacks' perforated heat shields: polished metal of the `trim` tint with a staggered grid of round holes
        drawn from the mesh's uv, which the exterior gives in metres (u along the shield, v round it): holes SHIELD_HOLE_R
        in radius SHIELD_PITCH apart, every other row shifted half a pitch; through a hole the dark pipe shows."""
        p = self.pal
        m, g = self._new("shield")
        col = p.mix("trim", ("text", 0.6))
        u, v, _ = g.sep(g.texco("UV"))
        su = g.mul(u, 1.0 / SHIELD_PITCH)
        shift = g.mul(g.m("MODULO", g.m("FLOOR", su), 2.0), 0.5)
        fu = g.sub(g.m("FRACT", su), 0.5)
        fv = g.sub(g.m("FRACT", g.add(g.mul(v, 1.0 / SHIELD_PITCH), shift)), 0.5)
        d = g.m("SQRT", g.add(g.mul(fu, fu), g.mul(fv, fv)))
        r = SHIELD_HOLE_R / SHIELD_PITCH
        hole = g.inv(g.ss(d, r - 0.04, r + 0.04))
        metal = g.principled(rgba(col), 0.07, 1.0, Specular_IOR_Level=0.6)
        lw = g.node("ShaderNodeLayerWeight")
        g.put(lw.inputs["Blend"], 0.5)
        g.put(metal.inputs["Emission Color"], rgba(col))
        g.put(metal.inputs["Emission Strength"], g.mad(g.m("POWER", g.inv(lw.outputs["Facing"]), 1.6), 0.45, 0.06))
        dark_col = p.mix("chassis", ("overlay", 0.1))
        dark = g.principled(rgba(dark_col), 0.8, 0.3, Specular_IOR_Level=0.2)
        self._lift(g, dark, dark_col, 0.04)
        mix = g.node("ShaderNodeMixShader")
        g.put(mix.inputs[0], hole)
        g.nt.links.new(metal.outputs[0], mix.inputs[1])
        g.nt.links.new(dark.outputs[0], mix.inputs[2])
        g.finish(mix.outputs[0])
        m.diffuse_color = rgba(col)
        return m

    def m_hose_red(self):
        return self._simple("hose_red", self.pal.mix("hose_red", ("base", 0.12)), 0.3, 0.0, floor=0.14,
                            Specular_IOR_Level=0.55, Coat_Weight=0.4, Coat_Roughness=0.1)

    def m_hose_blue(self):
        return self._simple("hose_blue", self.pal.mix("hose_blue", ("base", 0.12)), 0.3, 0.0, floor=0.14,
                            Specular_IOR_Level=0.55, Coat_Weight=0.4, Coat_Roughness=0.1)
