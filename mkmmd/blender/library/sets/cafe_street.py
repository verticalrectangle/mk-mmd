"""What is outside the cafe window: the sky dome, a wet pastel street with row houses, trees and the neighbours' facade,
three layers of falling rain and the lightning rigs. Everything outside is emission-only and runs through one grade
(`Outside.grade`): saturation (`outside_sat`), lightning flash (`flash`), low sun (`sun`), distance haze. Parameters are
read from the set's parameter group (cafe_nodes.PARAMS). A helper module: it registers no builder.

Room frame: the window wall is the plane x = WINDOW_X, outside is x < WINDOW_X; the street runs along y."""
import math
import random

import bmesh
import bpy
from mathutils import Vector

from ..props.cafe_kit import drive, mix
from .cafe_nodes import finish, hide_from_rays, box_geo

WINDOW_X = -0.80
WALL_T = 0.28
STREET_Z = -0.15                            # outside ground level
SKY_RADIUS = 88.0                           # inside the default 100 m camera far clip
SUN_DIR = Vector((-0.86, -0.22, 0.46)).normalized()      # towards the low sun (it breaks through at the window)
SPLAT_DRAWS = 4 * 16                        # random draws the glass splats used in the original; the street continues after them

RAIN_SPEED = 7.0            # m/s
RAIN_LAYERS = (
    # name, x range, y range, pattern height (m), streaks, length, width
    ("RainNear", (-1.45, -2.7), (-7.0, 7.0), 6.5, 520, (0.55, 0.95), (0.0040, 0.0062)),
    ("RainMid", (-2.7, -6.5), (-15.0, 15.0), 7.7, 1500, (1.0, 1.7), (0.0070, 0.0110)),
    ("RainFar", (-6.5, -18.0), (-48.0, 48.0), 9.1, 3600, (1.8, 3.0), (0.0140, 0.0230)),
)

BOLT_RIGS = {
    # name: (a camera position the rig is framed for, a point on the glass it must show through, distance)
    "Front": ((0.0, -2.6, 1.3), (-0.91, 0.30, 2.55), 54.0),
    "Behind": ((0.4, 0.95, 1.4), (-0.91, -0.30, 2.65), 52.0),
    "Side": ((0.2, -0.4, 1.4), (-0.91, -0.25, 2.55), 46.0),
}


def palette_colors(C):
    """Colours of the outside as blends of palette slots. `light(...)` tints are max-normalised light colours."""
    b = C.blend
    L = C.light
    # row-house pastels: the original picked each from this list per building
    pastels = [b(hl_high=.6, rose=.4, k=1.25), b(surface=.5, foam=.5, hue=-30, chroma=2, k=1.1),
               b(overlay=.6, rose=.4, hue=60, chroma=1.5, k=1.2), b(love=.6, surface=.4, hue=-70, k=1.2),
               b(hl_low=.55, gold=.45), b(overlay=.6, iris=.4, hue=-90, chroma=2, k=1.1), b(base=.75, rose=.25),
               b(overlay=.8, gold=.2, hue=60, chroma=1.5), b(surface=.5, rose=.5, hue=-50)]
    warm_glow = mix(mix(b(gold=1), b(rose=1), 0.35), b(surface=1), 0.30)
    return {
        "sky_h": b(surface=.8, love=.2),                                    # horizon haze
        "sky_z": b(iris=.6, hl_low=.4, hue=-20, chroma=1.5, k=1.2),                   # zenith (overcast lavender)
        "sky_c": b(hl_med=.8, rose=.2, hue=-80, chroma=1.25, k=1.2),                   # lighter cloud banks
        "warm_glow": warm_glow,
        "flash_sky": b(rose=.8, foam=.2, hue=-90, chroma=1.5, k=1.2),     # sky during a strike: luminous periwinkle
        "flash_obj": b(rose=.9, muted=.1, hue=-90, chroma=1.25, k=.9),    # everything else outside during a strike
        "sun_tint": L(gold=.5, hl_high=.5, hue=20),                        # low warm sun multiplied onto everything outside
        "sun_disc": L(foam=.5, overlay=.25, gold=.25),
        "facade_rose": mix(b(rose=1), b(base=1), 0.55),
        "pavers": [b(overlay=.6, rose=.3, surface=.1), b(hl_low=.7, love=.3)], "paver_mortar": b(hl_med=.7, love=.3),
        "asphalt": [b(iris=.6, base=.4, hue=-20, chroma=1.25, k=.9), b(overlay=.5, iris=.5, hue=-30, chroma=1.25, k=1.1)],
        "curb": b(hl_low=1),
        "pastels": pastels, "rose": b(rose=1), "cream": b(surface=1),
        "win_glass": [b(iris=.6, base=.4, hue=-40, chroma=1.25, k=1.2), b(iris=.7, surface=.3, hue=-40, chroma=1.5)],
        "win_lit": b(gold=.6, hl_low=.4, hue=20, k=1.2),
        "trunk": b(hl_med=.8, rose=.2, k=.75),
        "blossom": [b(base=.6, love=.4, hue=-20, chroma=2), b(hl_low=.6, love=.4, hue=-10, chroma=1.25, k=1.2),
                    b(hl_low=.6, gold=.4, hue=-60, chroma=1.5, k=.8), b(hl_low=.65, love=.35, k=1.25)],
        "leaves": [b(gold=.8, hl_med=.2, hue=70, chroma=.5, k=1.2), b(foam=.7, overlay=.3, hue=-60, chroma=2, k=1.2),
                   b(hl_low=.6, gold=.4, hue=70, chroma=1.25)],
        "rain": [b(iris=.8, overlay=.2, hue=-20, chroma=1.5, k=1.1), b(hl_low=.8, rose=.2, hue=-90, chroma=.75, k=1.2)],
        "bolt_core": L(surface=.45, overlay=.3, pine=.25), "bolt_halo": L(subtle=.8, surface=.15, foam=.05),
    }


class ColorBM:
    """bmesh with a per-vertex RGBA layer; it ends up as the 'bcol' colour attribute (points) of the mesh."""

    def __init__(self):
        self.bm = bmesh.new()
        self.layer = self.bm.verts.layers.float_color.new("bcol")

    def box(self, x0, x1, y0, y1, z0, z1, rgba):
        for v in box_geo(self.bm, x0, x1, y0, y1, z0, z1):
            v[self.layer] = rgba

    def prim(self, make, place, rgba):
        """make(bm) -> bmesh op result with 'verts'; place(co) -> new coordinate for each of its (live) vertices."""
        for v in [v for v in make(self.bm)["verts"] if v.is_valid]:
            v.co = place(v.co.copy())
            v[self.layer] = rgba

    def to_mesh(self, name, smooth=False):
        self.bm.normal_update()
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        for p in me.polygons:
            p.use_smooth = smooth
        return me


class Outside:
    def __init__(self, R):
        self.R = R
        self.c = palette_colors(R.C)

    # ---------------------------------------------------------------- colour pipeline
    def grade(self, g, col, strength=1.0, mist=None, sky=False):
        """Everything outside runs through this: outside_sat (HSV saturation), flash (periwinkle strike light), sun (warm
        gold wash), optional distance mist (near, far, amount) towards the horizon haze. Returns (colour, strength)."""
        c_ = self.c
        sat = g.param("outside_sat")
        flash = g.param("flash")
        sun = g.param("sun")
        c = g.hsv(col, sat=sat)
        if mist:
            near, far, amt = mist
            dist = g.node("ShaderNodeCameraData").outputs["View Distance"]
            f = g.mul(g.ss(dist, near, far), amt)
            c = g.mixc(c, g.hsv(c_["sky_h"], sat=sat), f)
        c = g.mixc(c, c_["sun_tint"], g.mul(sun, 0.80), blend="MULTIPLY")
        c = g.mixc(c, c_["flash_sky"] if sky else c_["flash_obj"], g.mul(flash, 0.88 if sky else 0.70))
        s = g.mul(g.mul(strength, g.mad(flash, 0.55 if sky else -0.12, 1.0)), g.mad(sun, 0.55, 1.0))
        return c, s

    def mat_out(self, name, fn, strength=1.0, mist=(12.0, 140.0, 0.8)):
        m, g = self.R.mat(name)
        col = fn(g)
        c, s = self.grade(g, col, strength, mist)
        em = g.node("ShaderNodeEmission")
        g.put(em.inputs["Color"], c)
        g.put(em.inputs["Strength"], s)
        finish(g, em.outputs[0])
        return m

    def mat_sky(self):
        c_ = self.c
        m, g = self.R.mat("Sky")
        co = g.texco("Object")
        d = g.vm("NORMALIZE", co)
        _, _, dz = g.sep(d)
        t = g.ss(dz, -0.02, 0.80)
        cl = g.noise(g.mapping(d, scale=(3.0, 3.0, 9.0)), 1.5, detail=5.0, rough=0.62)
        zen = g.mixc(c_["sky_z"], c_["sky_c"], g.ss(cl, 0.32, 0.72))
        col = g.mixc(c_["sky_h"], zen, t)
        c, s = self.grade(g, col, 1.0, sky=True)
        # sunbreak: broad gold lobe towards the sun, strongest low on the horizon, plus the disc. Added after the
        # saturation so a grey outside (outside_sat = 0) is still lit gold.
        cs = g.mx(g.vm("DOT_PRODUCT", d, tuple(SUN_DIR)), 0.0)
        low = g.inv(g.ss(dz, 0.05, 0.75))
        sun = g.param("sun")
        glow = g.mul(g.mul(g.sat(g.mul(g.pw(cs, 2.0), 1.8)), g.mad(low, 0.7, 0.3)), g.mul(sun, 0.95))
        c = g.mixc(c, c_["warm_glow"], glow)
        disc = g.add(g.mul(g.pw(cs, 900.0), 3.0), g.mul(g.pw(cs, 60.0), 0.7))
        c = g.mixc(c, c_["sun_disc"], g.sat(g.mul(disc, sun)))
        s = g.add(s, g.mul(g.mul(g.sat(g.pw(cs, 900.0)), 6.0), sun))
        em = g.node("ShaderNodeEmission")
        g.put(em.inputs["Color"], c)
        g.put(em.inputs["Strength"], s)
        finish(g, em.outputs[0])
        return m

    def mat_ground(self):
        """Wet pastel street: pink-beige pavers on the sidewalks, lavender asphalt, sky sheen at grazing angles."""
        c_ = self.c

        def fn(g):
            pos = g.texco("Object")
            px, py, _ = g.sep(pos)
            # sidewalk (x > -3.6 and x < -10.6), asphalt in between
            side = g.add(g.ss(px, -3.75, -3.45), g.ss(px, -10.45, -10.75))
            br = g.node("ShaderNodeTexBrick", offset=0.5, offset_frequency=1, squash=1.0, squash_frequency=2)
            g.put(br.inputs["Vector"], pos)
            g.put(br.inputs["Color1"], c_["pavers"][0])
            g.put(br.inputs["Color2"], c_["pavers"][1])
            g.put(br.inputs["Mortar"], c_["paver_mortar"])
            g.put(br.inputs["Mortar Size"], 0.012)
            g.put(br.inputs["Mortar Smooth"], 0.15)
            g.put(br.inputs["Brick Width"], 0.6)
            g.put(br.inputs["Row Height"], 0.3)
            patch = g.noise(pos, 0.35, detail=3.0, rough=0.6)
            asph = g.mixc(c_["asphalt"][0], c_["asphalt"][1], patch)
            curb = g.sub(g.ss(px, -3.72, -3.6), g.ss(px, -3.6, -3.45))
            ground = g.mixc(asph, br.outputs["Color"], g.sat(side))
            ground = g.mixc(ground, c_["curb"], g.mul(g.sat(curb), 0.8))
            fres = g.node("ShaderNodeLayerWeight").outputs["Facing"]
            sheen = g.pw(fres, 3.0)
            return g.mixc(ground, c_["sky_h"], g.mul(sheen, 0.85))
        return self.mat_out("Ground", fn, strength=1.05, mist=(15.0, 160.0, 0.85))

    def mat_facade(self):
        """Row-house facades from one material: base colour + random seed per building from the 'bcol' colour attribute;
        window grid from world position: lavender glass, white frames, some warm lit windows, striped awnings."""
        c_ = self.c

        def fn(g):
            vc = g.node("ShaderNodeVertexColor", layer_name="bcol")
            base, seed = vc.outputs["Color"], vc.outputs["Alpha"]
            geo = g.node("ShaderNodeNewGeometry")
            px, py, pz = g.sep(geo.outputs["Position"])
            nx, _, _ = g.sep(geo.outputs["Normal"])
            face = g.ss(nx, 0.45, 0.85)
            hh = g.sub(pz, STREET_Z)
            bay = 2.9
            uu = g.div(g.add(py, g.mul(seed, 40.0)), bay)
            bi = g.fl(uu)
            cu = g.fr(uu)
            gf = 3.2
            fh = 2.6
            ff = g.div(g.sub(hh, gf), fh)
            fi = g.fl(ff)
            cv = g.fr(ff)
            upper = g.ss(hh, gf - 0.05, gf + 0.05)

            def box(v, a, b, w=0.025):
                return g.mul(g.ss(v, a - w, a + w), g.sub(1.0, g.ss(v, b - w, b + w)))

            win_u = box(cu, 0.22, 0.78)
            win_v = box(cv, 0.20, 0.78)
            win_up = g.mul(win_u, win_v)
            frm_up = g.mul(box(cu, 0.17, 0.83), box(cv, 0.14, 0.84))
            # shopfront: wide glazing
            sv = g.div(hh, gf)
            win_sf = g.mul(box(cu, 0.10, 0.90), box(sv, 0.09, 0.66))
            frm_sf = g.mul(box(cu, 0.06, 0.94), box(sv, 0.05, 0.70))
            win = g.mixf(win_sf, win_up, upper)
            frm = g.mixf(frm_sf, frm_up, upper)
            # awning band on the shopfront (stripes), only on buildings with seed > 0.4
            band = g.mul(box(sv, 0.70, 0.84, 0.01), g.sub(1.0, upper))
            stripes = g.ss(g.fr(g.div(py, 0.42)), 0.45, 0.55)
            awn = g.mul(band, g.gt(seed, 0.4))
            awn_col = g.mixc(c_["rose"], c_["cream"], stripes)
            # warm lit windows
            rv, _ = g.white("3D", vec=g.xyz(bi, g.mad(fi, 1.0, g.mul(upper, 0.0)), g.mul(seed, 17.0)))
            lit = g.lt(rv, 0.24)
            glass = g.mixc(c_["win_glass"][0], c_["win_glass"][1], g.ss(g.add(cv, g.mul(rv, 0.5)), 0.4, 1.2))
            glass = g.mixc(glass, c_["win_lit"], lit)
            wall = base
            col = g.mixc(wall, c_["cream"], g.mul(frm, face))
            col = g.mixc(col, glass, g.mul(win, face))
            col = g.mixc(col, awn_col, g.mul(awn, face))
            # plinth + soft vertical shading (darker near the ground, lighter up)
            shade = g.mad(g.ss(hh, 0.0, 9.0), 0.22, 0.78)
            return g.mixc((0.0, 0.0, 0.0), col, shade)
        return self.mat_out("Facade", fn, strength=1.0, mist=(14.0, 150.0, 0.85))

    def mat_foliage(self):
        def fn(g):
            vc = g.node("ShaderNodeVertexColor", layer_name="bcol")
            geo = g.node("ShaderNodeNewGeometry")
            _, _, nz = g.sep(geo.outputs["Normal"])
            clump = g.noise(g.texco("Object"), 7.0, detail=4.0, rough=0.6)
            c = g.mixc(vc.outputs["Color"], g.mixc(vc.outputs["Color"], (1.0, 1.0, 1.0), 0.5), g.ss(clump, 0.45, 0.7))
            shade = g.mad(g.ss(nz, -0.6, 0.9), 0.30, 0.74)
            return g.mixc((0.0, 0.0, 0.0), c, shade)
        return self.mat_out("Foliage", fn, strength=1.0, mist=(10.0, 130.0, 0.8))

    # ---------------------------------------------------------------- geometry
    def build_street(self, rng):
        """Street, facades across the road, trees on the sidewalk, the neighbours' wall."""
        R, c_ = self.R, self.c
        ground_m = self.mat_ground()
        R.mesh_obj("OutGround", [(-86, -86, STREET_Z), (-1.08, -86, STREET_Z), (-1.08, 86, STREET_Z),
                                 (-86, 86, STREET_Z)], [(0, 1, 2, 3)], "outside", [ground_m])
        # row houses across the street
        cb = ColorBM()
        y = -62.0
        while y < 62.0:
            w = rng.uniform(5.6, 9.8)
            h = 5.8 if -45.0 < y < 50.0 else rng.choice([5.8, 5.8, 8.4])        # keep the sky open above the roofs
            xf = -12.6 + rng.uniform(-0.5, 0.5)
            depth = 9.0
            c = rng.choice(c_["pastels"])[:3]
            seed = rng.random()
            cb.box(xf - depth, xf, y, y + w - 0.05, STREET_Z, STREET_Z + h, tuple(c) + (seed,))
            # cornice: slightly wider, lighter
            cb.box(xf - depth, xf + 0.28, y - 0.03, y + w - 0.02, STREET_Z + h - 0.35, STREET_Z + h + 0.12,
                   tuple(mix(c, c_["cream"], 0.6)[:3]) + (0.0,))
            if rng.random() < 0.55:      # chimney
                cx = rng.uniform(y + 1.0, y + w - 1.5)
                cb.box(xf - 5.0, xf - 4.3, cx, cx + 0.7, STREET_Z + h, STREET_Z + h + rng.uniform(1.2, 2.2),
                       tuple(mix(c, c_["rose"], 0.4)[:3]) + (0.0,))
            y += w
        # distant second row (hazy)
        y = -80.0
        while y < 80.0:
            w = rng.uniform(8.0, 14.0)
            h = rng.uniform(6.0, 9.0)
            xf = -30.0 + rng.uniform(-1.5, 1.5)
            cb.box(xf - 12.0, xf, y, y + w - 0.1, STREET_Z, STREET_Z + h, tuple(rng.choice(c_["pastels"])[:3]) + (rng.random(),))
            y += w
        me = cb.to_mesh(R.oname("OutBuildings"), smooth=False)
        me.materials.append(self.mat_facade())
        R.obj("OutBuildings", me, "outside")

        # trees along the sidewalk
        cb = ColorBM()
        for ty in [-13.0, -6.9, 1.0, 8.9, 16.5, 25.0]:      # outside the sight corridors of the lightning rigs
            tx = -3.0 + rng.uniform(-0.25, 0.25)
            trunk_h = rng.uniform(3.0, 3.8)
            cb.prim(lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.16, radius2=0.10,
                                                     depth=trunk_h),
                    lambda co: Vector((tx + co.x, ty + co.y, STREET_Z + trunk_h / 2 + co.z)), tuple(c_["trunk"][:3]) + (0.0,))
            blossom = rng.random() < 0.6
            palette = c_["blossom"] if blossom else c_["leaves"]
            for _ in range(rng.randint(7, 10)):
                r = rng.uniform(0.9, 1.5)
                cx = tx + rng.uniform(-1.3, 1.3)
                cy = ty + rng.uniform(-1.5, 1.5)
                cz = STREET_Z + trunk_h + rng.uniform(0.0, 1.9)
                cb.prim(lambda bm: bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0),
                        lambda co, r=r, cx=cx, cy=cy, cz=cz: Vector(
                            (cx + co.x * r * (1 + rng.uniform(-0.12, 0.12)), cy + co.y * r * (1 + rng.uniform(-0.12, 0.12)),
                             cz + co.z * r * 0.82 * (1 + rng.uniform(-0.12, 0.12)))),
                        tuple(rng.choice(palette)[:3]) + (0.0,))
        me = cb.to_mesh(R.oname("OutTrees"), smooth=True)
        me.materials.append(self.mat_foliage())
        R.obj("OutTrees", me, "outside")

        # the neighbours: our own facade plane continues along the street
        fac_m = self.mat_out("FacadeWall", lambda g: c_["facade_rose"], strength=1.0, mist=(15.0, 150.0, 0.7))
        xw = WINDOW_X - WALL_T
        R.mesh_obj("FacadeFar", [(xw, -60, STREET_Z), (xw, -3.48, STREET_Z), (xw, -3.48, 9.0), (xw, -60, 9.0),
                                 (xw, 1.33, STREET_Z), (xw, 60, STREET_Z), (xw, 60, 9.0), (xw, 1.33, 9.0)],
                   [(0, 3, 2, 1), (4, 7, 6, 5)], "outside", [fac_m])
        sky_me = bpy.data.meshes.new(R.oname("Sky"))
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=SKY_RADIUS)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(sky_me)
        bm.free()
        sky_me.materials.append(self.mat_sky())
        sky = R.obj("Sky", sky_me, "outside")
        hide_from_rays(sky, shadow=False, glossy=True, diffuse=False, transmission=True)

    # ---------------------------------------------------------------- falling rain
    def mat_rain(self):
        m, g = self.R.mat("Rain")
        vc = g.node("ShaderNodeVertexColor", layer_name="bcol")
        k = vc.outputs["Alpha"]
        col = g.mixc(self.c["rain"][0], self.c["rain"][1], k)
        c, s = self.grade(g, col, 1.15, mist=(8.0, 90.0, 0.55))
        em = g.node("ShaderNodeEmission")
        g.put(em.inputs["Color"], c)
        g.put(em.inputs["Strength"], s)
        finish(g, em.outputs[0])
        return m

    def build_rain(self):
        """Three layers of thin vertical streaks between the window and the street. Each layer is one mesh made of two
        crossed quads per streak; the object slides down with a driver (saw-tooth over the pattern height, so the loop is
        seamless) plus a per-frame sideways jitter so the rain never repeats visibly. The rain starts to fall 3 frames
        before clip time 0."""
        R = self.R
        mat = self.mat_rain()
        for li, (name, xr, yr, zh, n, lr, wr) in enumerate(RAIN_LAYERS):
            cb = ColorBM()
            for rep in range(3):
                r2 = random.Random(1000 + li)                 # same pattern in every repeat -> periodic in z
                for _ in range(n):
                    x = r2.uniform(*xr)
                    y = r2.uniform(*yr)
                    z0 = r2.uniform(-1.0, -1.0 + zh) + rep * zh
                    L = r2.uniform(*lr)
                    w = r2.uniform(*wr)
                    slant = 0.07 * L
                    k = r2.random()
                    for (dx, dy) in ((w / 2, 0.0), (0.0, w / 2)):
                        vs = [cb.bm.verts.new((x - dx, y - dy, z0)), cb.bm.verts.new((x + dx, y + dy, z0)),
                              cb.bm.verts.new((x + dx, y + dy + slant, z0 + L)),
                              cb.bm.verts.new((x - dx, y - dy + slant, z0 + L))]
                        for v in vs:
                            v[cb.layer] = (0.0, 0.0, 0.0, k)
                        cb.bm.faces.new(vs)
            me = cb.to_mesh(R.oname(name))
            me.materials.append(mat)
            o = R.obj(name, me, "outside")
            hide_from_rays(o, shadow=False, glossy=True, diffuse=False, transmission=True)
            step = RAIN_SPEED / R.fps
            drive(o, "location", f"-fmod(max(frame - {R.frame0 - 3}, 0) * {step:.5f} + {li * 1.7}, {zh})", index=2)
            drive(o, "location",
                  f"{0.35 + 0.1 * li} * (fmod(abs(sin(frame * {12.9898 + li}) * 43758.5453), 1.0) - 0.5)", index=1)

    # ---------------------------------------------------------------- lightning
    def bolt_paths(self, seed, variant):
        """Branching bolt in a plane: list of polylines [(a, b, radius)], a = horizontal metres, b = vertical metres."""
        rng = random.Random(seed)

        def jagged(p0, p1, jag, levels):
            pts = [p0, p1]
            for _ in range(levels):
                new = [pts[0]]
                for i in range(len(pts) - 1):
                    a, b = pts[i], pts[i + 1]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    ln = math.hypot(dx, dy) or 1.0
                    off = rng.uniform(-1.0, 1.0) * jag * ln * 0.5
                    new.append(((a[0] + b[0]) / 2 - dy / ln * off, (a[1] + b[1]) / 2 + dx / ln * off))
                    new.append(b)
                pts = new
                jag *= 0.62
            return pts

        paths = []

        def stroke(p0, p1, r0, r1, jag=0.55, levels=5, branches=0, depth=0):
            pts = jagged(p0, p1, jag, levels)
            n = len(pts)
            paths.append([(p[0], p[1], r0 + (r1 - r0) * i / (n - 1)) for i, p in enumerate(pts)])
            if depth >= 2:
                return
            for _ in range(branches):
                i = rng.randint(n // 6, n - n // 5)
                q = pts[i]
                side = rng.choice((-1.0, 1.0))
                ang = math.radians(rng.uniform(22, 58)) * side
                length = rng.uniform(0.12, 0.28) * abs(p1[1] - p0[1]) / (1 + depth * 1.4)
                dxn, dyn = math.sin(ang), -math.cos(ang)
                stroke(q, (q[0] + dxn * length, q[1] + dyn * length), r0 * (0.5 - 0.18 * depth), r0 * 0.14, jag * 0.8,
                       max(3, levels - 1 - depth), branches=1 if depth == 0 else 0, depth=depth + 1)

        if variant == 0:
            stroke((2.0, 31.0), (-3.0, -26.0), 1.0, 0.6, branches=6)
        else:
            stroke((-3.0, 32.0), (4.0, -26.0), 1.0, 0.55, branches=4)
            stroke((-3.0, 32.0), (-9.5, -8.0), 0.7, 0.3, jag=0.5, branches=2)
        return paths

    def mat_bolt(self, rig_index, kind):
        """Bolt strength = parameter `bolt` (alpha / brightness), only for the variant selected by `bolt_variant` and
        the rig selected by `bolt_rig`."""
        m, g = self.R.mat(f"Bolt_{rig_index}_{kind}")
        bolt = g.param("bolt")
        var = g.param("bolt_variant")
        sel = g.sub(1.0, g.sat(g.mul(g.ab(g.sub(var, float(kind[-1]))), 2.0)))
        rig = g.param("bolt_rig")
        sel = g.mul(sel, g.sub(1.0, g.sat(g.mul(g.ab(g.sub(rig, float(rig_index))), 2.0))))
        vis = g.mul(g.sat(g.mul(bolt, 3.0)), sel)
        if kind.startswith("core"):
            col, strength, alpha = self.c["bolt_core"], g.mul(22.0, g.mad(bolt, 0.5, 0.5)), vis
        else:
            fac = g.node("ShaderNodeLayerWeight").outputs["Facing"]
            halo = kind.startswith("halo")
            col = self.c["bolt_halo"]
            strength = 5.0 if halo else 3.0
            alpha = g.mul(g.mul(vis, g.pw(g.inv(fac), 2.2 if halo else 3.0)), 0.55 if halo else 0.26)
        em = g.node("ShaderNodeEmission")
        g.put(em.inputs["Color"], col)
        g.put(em.inputs["Strength"], strength)
        tr = g.node("ShaderNodeBsdfTransparent")
        mixs = g.node("ShaderNodeMixShader")
        g.put(mixs.inputs[0], alpha)
        g.nt.links.new(tr.outputs[0], mixs.inputs[1])
        g.nt.links.new(em.outputs[0], mixs.inputs[2])
        finish(g, mixs.outputs[0])
        return m

    def build_lightning(self):
        """Bolt curves for each rig and both variants (+ glow shells). Visible only through the parameters `bolt`
        (strength), `bolt_variant` (0/1 shape) and `bolt_rig` (which framing: 0 Front, 1 Behind, 2 Side)."""
        R = self.R
        for ri, (rig, (origin, through, dist)) in enumerate(BOLT_RIGS.items()):
            mats = {f"{k}{v}": self.mat_bolt(ri, f"{k}{v}") for k in ("core", "halo", "glow") for v in (0, 1)}
            O, T = Vector(origin), Vector(through)
            d = (T - O).normalized()
            centre = O + d * dist
            hd = Vector((d.x, d.y, 0.0)).normalized()
            A = Vector((-hd.y, hd.x, 0.0))
            for variant in (0, 1):
                paths = self.bolt_paths(7 + 31 * ri + 101 * variant, variant)
                for kind, scale in (("core", 0.85), ("halo", 3.0), ("glow", 8.0)):
                    cu = bpy.data.curves.new(R.oname(f"Bolt{rig}{variant}_{kind}"), "CURVE")
                    cu.dimensions = "3D"
                    cu.bevel_depth = scale * 0.5
                    cu.bevel_resolution = 1
                    cu.use_fill_caps = False
                    for pts in paths:
                        sp = cu.splines.new("POLY")
                        sp.points.add(len(pts) - 1)
                        for p, (a, b, r) in zip(sp.points, pts):
                            w = centre + A * (a + (-6.0 if variant == 0 else 6.0)) + Vector((0.0, 0.0, b))
                            p.co = (w.x, w.y, w.z, 1.0)
                            p.radius = r
                    cu.materials.append(mats[f"{kind}{variant}"])
                    o = R.obj(f"Bolt{rig}{variant}_{kind}", cu, "outside")
                    hide_from_rays(o, shadow=False, glossy=True, diffuse=False, transmission=True)
