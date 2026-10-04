"""night_sky: the world of the night drive (background and the sky the wet road reflects) and an optional moonlight.
Nothing spatial: the set root is only an empty. Layers, composited in this order: a gradient (dark indigo dome, thin
iris haze band on the horizon), a local city glow, thin lit clouds, stars, the moon with its halo.

The direction maths and the star density solver import without Blender, so `bpy` and the node kit are imported inside
the build functions."""
import math

import numpy as np

from ....core.palette import mix
from . import register

# ---------------------------------------------------------------------------------------------------- spec

SPEC = {"zenith": "base", "dome": "surface", "horizon": None, "band": 4.0}
STARS = {"density": 4.0, "size": 0.07, "brightness": 1.0, "fade": [3.0, 18.0], "color": "text", "seed": 1}
MOON = {"az": -74.0, "el": 13.0, "size": 4.0, "color": "text", "strength": 3.0, "halo": 1.0, "light": 0.3,
        "shadow": True, "seed": 1}
GLOW = {"az": -90.0, "width": 60.0, "height": 2.8, "core": "gold", "rim": "rose", "strength": 1.0, "haze": 0.1}
CLOUDS = {"cover": 0.4, "softness": 0.5, "opacity": 0.35, "drift": 0.3, "wind": 20.0, "seed": 1}
EEVEE = {"raytracing": True, "trace_scale": 1, "quality": 0.5, "probe": 2048}
TABLES = {"stars": STARS, "moon": MOON, "glow": GLOW, "clouds": CLOUDS, "eevee": EEVEE}
BUILD_KEYS = {"name", "kind", "at", "yaw"}                 # keys of the build stage itself
LIMITS = {                                                 # (table or None, key): (min, max)
    (None, "band"): (0.5, 60.0),
    ("stars", "density"): (0.01, 1000.0), ("stars", "size"): (0.005, 2.0), ("stars", "brightness"): (0.0, 100.0),
    ("moon", "size"): (0.1, 60.0), ("moon", "strength"): (0.0, 1000.0), ("moon", "halo"): (0.0, 100.0),
    ("moon", "light"): (0.0, 1000.0),
    ("glow", "width"): (1.0, 360.0), ("glow", "height"): (0.5, 45.0), ("glow", "strength"): (0.0, 100.0),
    ("glow", "haze"): (0.0, 100.0),
    ("clouds", "cover"): (0.0, 1.0), ("clouds", "softness"): (0.0, 1.0), ("clouds", "opacity"): (0.0, 1.0),
    ("clouds", "drift"): (-90.0, 90.0),
    ("eevee", "quality"): (0.0, 1.0),
}

# ---------------------------------------------------------------------------------------------------- look constants

SPHERE_SQDEG = 41252.96
FAR = 1.0e5                       # distance of the card's look points (m): far enough to be a direction
LN2 = math.log(2.0)
DOME_HALF = 18.0                  # elevation (deg) where the dome has lifted half-way from the zenith to its low colour
LIFT = 2.5                        # degrees over which the city glow rises out of the horizon haze
GLOW_RIM = 0.19                   # radiance of the glow colours at strength 1 (the horizon haze itself is ~0.13)
GLOW_CORE = 0.27
STAR_FLOOR = 0.04                 # faintest peak radiance that still counts as a visible star
STAR_PEAK = 1.8                   # peak radiance of the brightest star at brightness 1
STAR_GAMMA = 2.5                  # brightness = u ** gamma for u uniform: many dim stars, a few bright
PROBE_STAR = 0.12                 # size (deg) of a star as the light probe (reflections) sees it
PROBE_GAIN = 0.4                  # and its brightness relative to the star the camera sees
CLOUD_PLANE_Z = 0.18              # the cloud layer is a flat plane: coordinates = xy / (z + this)
HALO_GAUSS = ((2.4, 0.10), (7.0, 0.009))     # halo = sum of gaussians (radius in moon radii, amplitude / disc radiance)


# ---------------------------------------------------------------------------------------------------- pure maths


def direction(az, el=0.0):
    """Unit vector of a world direction: az in degrees (0 = +X, counter-clockwise seen from above, the convention of
    `Path.heading` and yaw), el above the horizon."""
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def tangent_frame(m):
    """Unit vectors (e1, e2) perpendicular to the unit vector m: e1 horizontal (to the left of m seen from the origin),
    e2 pointing up."""
    e1 = np.cross([0.0, 0.0, 1.0], m)
    n = np.linalg.norm(e1)
    e1 = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else e1 / n
    return e1, np.cross(m, e1)


def to_root(v, yaw):
    """A world-space vector in the frame of a root turned by `yaw` (radians) about Z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def size_factor(b):
    """A star's size relative to the base size, from its brightness b in 0..1 (bright stars are a little bigger)."""
    return 0.85 + 0.7 * np.asarray(b, float)


def star_scale(density, size_deg, brightness=1.0):
    """Voronoi cells per radian that give about `density` stars per 100 square degrees with a peak radiance above
    STAR_FLOOR. A star is the nearest feature point of a 3D cell grid seen through a gaussian of its 3D distance, so it
    shows when its point lies within a few sigma of the unit sphere: N = 8 pi sigma S^3 E[k sqrt(2 ln(peak b / floor))]
    over the whole sphere, with the star's size factor k and brightness b = u ** STAR_GAMMA."""
    sigma = math.radians(size_deg) / 2.3548
    b = ((np.arange(4000) + 0.5) / 4000.0) ** STAR_GAMMA
    reach = np.sqrt(np.maximum(2.0 * np.log(np.maximum(STAR_PEAK * brightness * b, 1e-12) / STAR_FLOOR), 0.0))
    total = density * SPHERE_SQDEG / 100.0
    return (total / (8.0 * math.pi * sigma * np.mean(size_factor(b) * reach))) ** (1.0 / 3.0)


def parse(spec):
    """The spec with every default filled in: sub-tables merged over their defaults (False switches stars, moon, glow,
    clouds and eevee off, True takes the defaults). Unknown keys and values out of range are errors."""
    bad = sorted(set(spec) - set(SPEC) - set(TABLES) - BUILD_KEYS)
    if bad:
        raise ValueError(f"night_sky: unknown keys {bad} (have {sorted(set(SPEC) | set(TABLES))})")
    cfg = {k: spec.get(k, v) for k, v in SPEC.items()}
    for key, default in TABLES.items():
        over = spec.get(key, {})
        if over is False:
            cfg[key] = None
            continue
        over = {} if over is True else over
        if not isinstance(over, dict):
            raise ValueError(f"night_sky: {key} must be a table or false, not {over!r}")
        bad = sorted(set(over) - set(default))
        if bad:
            raise ValueError(f"night_sky: unknown keys {bad} in {key} (have {sorted(default)})")
        cfg[key] = {**default, **over}
    for (table, key), (lo, hi) in LIMITS.items():
        scope = cfg if table is None else cfg[table]
        if scope is not None and not lo <= scope[key] <= hi:
            where = f"{table}.{key}" if table else key
            raise ValueError(f"night_sky: {where} must be in [{lo}, {hi}], not {scope[key]!r}")
    if cfg["stars"] and not (len(cfg["stars"]["fade"]) == 2 and cfg["stars"]["fade"][0] < cfg["stars"]["fade"][1]):
        raise ValueError(f"night_sky: stars.fade must be [gone, full] elevations (deg), not {cfg['stars']['fade']!r}")
    if cfg["eevee"]:
        if cfg["eevee"]["trace_scale"] not in (1, 2, 4, 8, 16):
            raise ValueError(f"night_sky: eevee.trace_scale must be 1, 2, 4, 8 or 16, not "
                             f"{cfg['eevee']['trace_scale']!r}")
        if cfg["eevee"]["probe"] not in (128, 256, 512, 1024, 2048, 4096):
            raise ValueError(f"night_sky: eevee.probe must be 128 .. 4096 (powers of two), not "
                             f"{cfg['eevee']['probe']!r}")
    return cfg


# ---------------------------------------------------------------------------------------------------- the world shader


class _Sky:
    """Builds the world node tree. Angles are world angles: `Generated` in a world shader is the unit view direction."""

    def __init__(self, K, world, cfg, pal):
        self.K, self.cfg, self.pal = K, cfg, pal
        world.use_nodes = True
        world.node_tree.nodes.clear()
        self.nt = world.node_tree
        self.nb = K.NB(self.nt)

    # -- small arithmetic on top of the kit's node builder
    def mul(self, a, b):
        return self.nb.math("MULTIPLY", a, b)

    def add(self, a, b):
        return self.nb.math("ADD", a, b)

    def sub(self, a, b):
        return self.nb.math("SUBTRACT", a, b)

    def div(self, a, b):
        return self.nb.math("DIVIDE", a, b)

    def sq(self, a):
        return self.nb.math("MULTIPLY", a, a)

    def expneg(self, a):
        """exp(-a)."""
        return self.nb.math("EXPONENT", self.nb.math("MULTIPLY", a, -1.0))

    def gauss(self, a, w):
        """exp(-(a / w)^2)."""
        return self.expneg(self.sq(self.div(a, w)))

    def scale(self, col, f):
        return self.nb.vmath("SCALE", col, scale=f)

    def plus(self, a, b):
        return self.nb.vmath("ADD", a, b)

    def rgb(self, c, k=1.0):
        return self.K.rgb(self.pal, c, k)

    def hexof(self, c):
        return self.K.hexof(self.pal, c)

    # -- layers
    def gradient(self):
        """The zenith colour overhead, lifting toward the dome colour lower down (half-way at DOME_HALF degrees), and a
        thin band of horizon colour on and below the horizon (half-way at `band` degrees), blended in gamma space like
        the palette's own mixes."""
        c, nb = self.cfg, self.nb
        horizon = c["horizon"] or self.K.haze_hex(self.pal)
        z_band = math.sin(math.radians(c["band"])) / LN2 ** 0.5
        z_dome = math.sin(math.radians(DOME_HALF)) / LN2 ** (1.0 / 1.5)
        w_band = self.gauss(self.zc, z_band)
        w_dome = self.expneg(nb.math("POWER", self.div(self.zc, z_dome), 1.5))
        lo, mid, hi = (tuple(v ** (1.0 / 2.2) for v in self.rgb(k)) for k in (c["zenith"], c["dome"], horizon))
        mixed = nb.mixc(w_band, nb.mixc(w_dome, lo, mid), hi)
        return nb.node("ShaderNodeGamma", {0: mixed, 1: 2.2}).outputs[0]

    def glow(self, sky):
        """The city glow: a gaussian blob just above the horizon toward the city (rising out of the horizon haze, so
        the horizon colour stays the haze colour) blends the sky toward rose and, at its heart, gold at about the
        brightness of the haze; plus a faint all-round lift of the low sky."""
        g, nb = self.cfg["glow"], self.nb
        az, width = math.radians(g["az"]), math.radians(g["width"])
        height = math.sin(math.radians(g["height"]))
        hl = nb.math("SQRT", nb.math("MAXIMUM", self.add(self.sq(self.x), self.sq(self.y)), 1e-6))
        cos_d = self.div(self.add(self.mul(self.x, math.cos(az)), self.mul(self.y, math.sin(az))), hl)
        sin_d = self.div(self.sub(self.mul(self.y, math.cos(az)), self.mul(self.x, math.sin(az))), hl)
        across = self.expneg(self.mul(self.sq(self.mul(nb.math("ARCTAN2", sin_d, cos_d), 2.0 / width)), LN2))
        rise = nb.smooth(0.0, math.sin(math.radians(LIFT)), self.zc)
        blob = self.mul(self.mul(across, self.expneg(self.mul(self.sq(self.div(self.zc, height)), LN2))), rise)
        k = float(g["strength"])
        target = nb.mixc(nb.remap(blob, 0.45, 1.25), self.rgb(g["rim"], GLOW_RIM * k),
                         self.rgb(g["core"], GLOW_CORE * k))
        lift = self.mul(self.mul(self.gauss(self.zc, 3.2 * height), rise), k * g["haze"])
        return self.plus(nb.mixc(blob, sky, target), self.scale(self.rgb(g["rim"], GLOW_RIM), lift))

    def moon_frame(self):
        """Sockets in the moon's frame: rho (distance from its centre in moon radii), a and b (the two coordinates, in
        moon radii) and gate (1 on the moon's side of the sky and above the horizon)."""
        m, nb = self.cfg["moon"], self.nb
        mv = direction(m["az"], m["el"])
        e1, e2 = tangent_frame(mv)
        a, b, dm = nb.vdot(self.d, tuple(e1)), nb.vdot(self.d, tuple(e2)), nb.vdot(self.d, tuple(mv))
        R = math.sin(math.radians(m["size"]) / 2.0)
        rho = self.div(nb.math("SQRT", self.add(self.sq(a), self.sq(b))), R)
        near = nb.math("GREATER_THAN", dm, 0.3)
        return {"a": self.div(a, R), "b": self.div(b, R), "rho": rho, "near": near}

    def moon(self, mf):
        """(disc mask, disc emission, halo emission): a soft-edged disc with limb falloff and soft maria, and a halo."""
        m, nb = self.cfg["moon"], self.nb
        base = self.hexof(m["color"])
        warm = self.rgb(mix(base, self.pal["gold"], 0.10))
        maria = self.rgb(mix(base, self.pal["iris"], 0.5), 0.75)
        halo_rgb = self.rgb(mix(base, self.pal["iris"], 0.45))
        inside = self.sub(1.0, nb.smooth(0.98, 1.02, mf["rho"]))
        rc = nb.math("MINIMUM", mf["rho"], 1.0)
        limb = self.sub(1.0, self.mul(self.sub(1.0, nb.math("SQRT", self.sub(1.0, self.sq(rc)))), 0.35))
        pv = nb.comb(mf["a"], mf["b"], 7.0 * float(m["seed"]))
        mare = nb.smooth(0.44, 0.60, nb.noise(pv, scale=1.3, detail=3.0, rough=0.55))
        grain = nb.noise(pv, scale=7.0, detail=2.0, rough=0.5)
        refl = self.mul(self.sub(1.0, self.mul(mare, 0.30)), self.add(self.mul(grain, 0.18), 0.91))
        disc = self.scale(nb.mixc(self.mul(mare, 0.6), warm, maria), self.mul(self.mul(refl, limb), m["strength"]))
        halo = self.mul(self.gauss(mf["rho"], HALO_GAUSS[0][0]), HALO_GAUSS[0][1])
        for w, amp in HALO_GAUSS[1:]:
            halo = self.add(halo, self.mul(self.gauss(mf["rho"], w), amp))
        above = nb.smooth(-0.07, 0.0, self.z)                            # the halo fades out below the horizon
        halo = self.scale(halo_rgb, self.mul(self.mul(halo, self.mul(mf["near"], above)), m["strength"] * m["halo"]))
        sharp = self.mul(mf["near"], nb.smooth(-0.002, 0.002, self.z))   # the disc is cut at the horizon
        return self.mul(inside, sharp), disc, halo

    def drift(self, deg_per_s):
        """A Value node: how far (in cloud-plane units) the clouds have slid, driven by the frame number."""
        import bpy
        node = self.nb.node("ShaderNodeValue")
        node.label = "cloud drift"
        if deg_per_s:
            scene = bpy.context.scene
            per_frame = math.radians(float(deg_per_s)) / CLOUD_PLANE_Z / (scene.render.fps / scene.render.fps_base)
            driver = node.outputs[0].driver_add("default_value").driver
            driver.type = "SCRIPTED"
            driver.expression = f"frame * {per_frame!r}"
        return node.outputs[0]

    def clouds(self, sky, mf):
        """(cloud opacity 0..1, the sky with the clouds in it). A flat noise layer stretched along the wind and warped;
        clouds are the local sky colour brightened (and, near the moon, lit by it), never darker than the sky."""
        c, nb = self.cfg["clouds"], self.nb
        inv = self.div(1.0, self.add(self.zc, CLOUD_PLANE_Z))
        u, v = self.mul(self.x, inv), self.mul(self.y, inv)
        psi = math.radians(c["wind"])
        along = self.sub(self.add(self.mul(u, math.cos(psi)), self.mul(v, math.sin(psi))), self.drift(c["drift"]))
        across = self.add(self.mul(u, -math.sin(psi)), self.mul(v, math.cos(psi)))
        n = nb.noise(nb.comb(self.mul(along, 0.7), self.mul(across, 2.2), 3.0 * float(c["seed"])), scale=1.0,
                     detail=4.0, rough=0.55, dist=0.55)
        c0 = 0.70 - 0.30 * float(c["cover"])
        dens = nb.smooth(c0, c0 + 0.04 + 0.30 * float(c["softness"]), n)
        dens = self.mul(self.mul(dens, nb.smooth(0.03, 0.30, self.z)), c["opacity"])
        lit = self.plus(self.scale(sky, 1.9), self.rgb("iris", 0.015))
        if mf is not None:
            glow = self.rgb(mix(self.hexof(self.cfg["moon"]["color"]), self.pal["iris"], 0.3))
            lit = self.plus(lit, self.scale(glow, self.mul(self.mul(self.gauss(mf["rho"], 14.0), 0.10), mf["near"])))
        return dens, nb.mixc(dens, sky, lit)

    def stars(self, dens):
        """Star emission: a field of gaussian dots fading out toward the horizon and behind clouds. The nearest feature
        point of a 3D cell grid is a star when it lies within a few sigma of the unit sphere (radial offset d); its
        profile is exp(-(rho^2 + d^2) / 2 sigma^2) with rho the distance along the sky. Camera rays see that; the
        light probe (what glossy surfaces reflect) sees a wider, dimmer glint of the same stars, which its texels can
        sample without crawling."""
        s, nb = self.cfg["stars"], self.nb
        S = star_scale(s["density"], s["size"], s["brightness"])
        sigma = math.radians(s["size"]) / 2.3548 * S
        shift = tuple(float(v) for v in np.random.default_rng(int(s["seed"])).uniform(0.0, 64.0, 3) / S)
        vor = nb.node("ShaderNodeTexVoronoi", {0: nb.vmath("ADD", self.d, shift), 2: S, 8: 1.0}, feature="F1",
                      distance="EUCLIDEAN", voronoi_dimensions="3D")
        radial = self.mul(self.sub(nb.vlen(nb.vmath("SUBTRACT", vor.outputs["Position"], shift)), 1.0), S)
        along = nb.math("MAXIMUM", self.sub(self.sq(vor.outputs["Distance"]), self.sq(radial)), 0.0)
        r, g, _ = nb.sep(vor.outputs["Color"])
        b = nb.math("POWER", r, STAR_GAMMA)
        sg = self.mul(nb.math("MULTIPLY_ADD", b, 0.7, 0.85), sigma)
        cam = nb.node("ShaderNodeLightPath").outputs["Is Camera Ray"]
        wide = nb.mixf(cam, math.radians(PROBE_STAR) / 2.3548 * S, sg)
        gain = self.mul(self.mul(b, STAR_PEAK * float(s["brightness"])), nb.mixf(cam, PROBE_GAIN, 1.0))
        amp = self.mul(self.expneg(self.add(self.div(self.sq(radial), self.mul(self.sq(sg), 2.0)),
                                            self.div(along, self.mul(self.sq(wide), 2.0)))), gain)
        main = s["color"]
        tint = nb.mixc(nb.math("LESS_THAN", g, 0.11),
                       nb.mixc(nb.math("LESS_THAN", g, 0.20), self.rgb(main), self.rgb(f"{main}:gold:0.5")),
                       self.rgb(f"{main}:foam:0.5"))
        fade = nb.smooth(*(math.sin(math.radians(v)) for v in s["fade"]), self.z)
        if dens is not None:
            fade = self.mul(fade, nb.math("MAXIMUM", self.sub(1.0, self.mul(dens, 1.6)), 0.0))
        return self.scale(tint, self.mul(amp, fade))

    def build(self):
        c, nb = self.cfg, self.nb
        self.d = nb.texcoord("Generated")
        self.x, self.y, self.z = nb.sep(self.d)
        self.zc = nb.math("MAXIMUM", self.z, 0.0)
        col = self.gradient()
        if c["glow"]:
            col = self.glow(col)
        mf = self.moon_frame() if c["moon"] else None
        dens = None
        if c["clouds"]:
            dens, col = self.clouds(col, mf)
        if c["stars"]:
            col = self.plus(col, self.stars(dens))
        if c["moon"]:
            disc, disc_col, halo_col = self.moon(mf)
            if dens is not None:
                disc = self.mul(disc, self.sub(1.0, self.mul(dens, 0.6)))
            col = nb.mixc(disc, self.plus(col, halo_col), disc_col)
        bg = nb.node("ShaderNodeBackground", {0: col, 1: 1.0})
        self.nt.links.new(bg.outputs[0], nb.node("ShaderNodeOutputWorld").inputs[0])


# ---------------------------------------------------------------------------------------------------- the builder


def _eevee(scene, world, e):
    """The renderer settings the look needs (see `night_sky`: eevee)."""
    ee = scene.eevee
    ee.use_raytracing = bool(e["raytracing"])
    ee.ray_tracing_method = "SCREEN"
    ee.ray_tracing_options.resolution_scale = str(int(e["trace_scale"]))
    ee.ray_tracing_options.screen_trace_quality = float(e["quality"])
    world.probe_resolution = str(int(e["probe"]))


@register("night_sky")
def night_sky(name, coll, root, spec, palette):
    """night_sky: the scene's world, i.e. the background and the sky that glossy things reflect, and optionally a
    moonlight. Replaces the scene's world with a fresh one named after the set. The sky is in WORLD angles: the set's
    `at` and `yaw` do not move or turn it (only the look points of the card are expressed in the set's frame). Every
    key is optional; colours are palette slots or '#hex' (and 'slot:slot:t' mixes); angles are degrees.

      zenith = "base"      sky colour overhead
      dome = "surface"     colour of the low dome: the sky lifts from the zenith colour toward it (half-way at 18 deg)
      horizon = (haze)     colour on and below the horizon; default `nightkit.haze_hex(palette)`, the colour the other
                           night sets fade distant things into, so ground and sky meet without a seam
      band = 4             elevation (deg) where the horizon band has faded half-way into the dome (thin: gone by ~10)

      stars = {density = 4        stars per 100 square degrees (a 10 x 10 deg patch) bright enough to see
               size = 0.07        diameter of a typical star (deg, soft gaussian, full width at half maximum); a
                                  few bright ones are up to 1.5x bigger
               brightness = 1.0   multiplier of the star radiance (brightest star: peak radiance 1.8)
               fade = [3, 18]     elevations where stars are gone / at full strength (they fade into the haze)
               color = "text"     most stars; a few lean foam or gold
               seed = 1}          stars = false: none. In reflections stars show as wider, dimmer glints.

      moon = {az = -74, el = 13   direction of the moon: az 0 = +X, counter-clockwise seen from above (the convention
                                  of yaw and `Path.heading`; -90 is -Y, where the default road runs), el above horizon
              size = 4.0          angular diameter (deg)
              color = "text"      the lit surface; maria lean iris, the disc a touch gold
              strength = 3.0      radiance of the disc (it rolls to white-lilac under AgX)
              halo = 1.0          multiplier of the soft glow around the disc
              light = 0.3         strength (W/m2) of a SUN light from the moon, in cool moonlight colour, penumbra as
                                  wide as the disc; 0 = no light object. It has no specular: the moon's reflections
                                  come from the sky itself
              shadow = true       the moonlight casts (soft) shadows
              seed = 1}           moon = false: no moon, no halo, no light

      glow = {az = -90            direction of the city
              width = 60          angular width at half maximum
              height = 2.8        height at half maximum above the horizon; the glow rises out of the haze, so the
                                  horizon itself keeps the horizon colour
              core = "gold", rim = "rose"   colour of its heart and of its edges
              strength = 1.0      1 = the heart is about as bright as the horizon haze, warm and faint
              haze = 0.1}         faint all-round lift of the low sky, as a share of strength. glow = false: none

      clouds = {cover = 0.4       0..1, share of the sky under cloud
                softness = 0.5    0..1, how blurred their edges are
                opacity = 0.35    0..1, how much they hide the sky (thin veils, lit lavender near the moon)
                drift = 0.3       degrees per second the pattern slides along the wind (0 = still)
                wind = 20         direction of the streaks and the drift, in the sky plane (az convention)
                seed = 1}         clouds = false: none

      eevee = {raytracing = true  EEVEE Next screen-space ray tracing: lamps reflected in wet asphalt
               trace_scale = 1    ray tracing resolution divisor (1 full, 2 half, 4 quarter)
               quality = 0.5      screen trace quality, 0..1
               probe = 2048}      world light probe resolution (128 .. 4096): what mirrors reflect; at 1024 the
                                  stars in a mirror turn into diamonds. eevee = false leaves the scene's renderer
                                  settings alone. Neither the view transform nor the compositor is touched.

    The world's sun extraction is switched off (the moon disc must not turn into a second sun on top of the light).
    Card: kind, paths {}, use.look [{name: "moon", point}, {name: "glow", point}] (points 100 km out in that world
    direction, expressed in the set root's frame), use.surface [], colliders [], lights [the moonlight object]."""
    import bpy
    from . import nightkit as K

    cfg = parse(spec)
    old = bpy.data.worlds.get(name)
    if old is not None:
        bpy.data.worlds.remove(old)
    world = bpy.data.worlds.new(name)
    scene = bpy.context.scene
    scene.world = world
    _Sky(K, world, cfg, palette).build()
    world.sun_threshold = 0.0
    if cfg["eevee"]:
        _eevee(scene, world, cfg["eevee"])
    yaw = root.rotation_euler.z
    look, lights = [], []
    m, g = cfg["moon"], cfg["glow"]
    if m:
        look.append({"name": "moon", "point": to_root(direction(m["az"], m["el"]) * FAR, yaw).tolist()})
        if float(m["light"]) > 0.0:
            d = to_root(direction(m["az"], m["el"]), yaw)
            tint = mix(K.hexof(palette, m["color"]), palette["iris"], 0.35)
            K.clean(f"{name}_moonlight")
            sun = K.make_light(f"{name}_moonlight", coll, root, "SUN", tuple(float(v) for v in d * 100.0),
                               float(m["light"]), K.rgb(palette, tint), direction=tuple(float(v) for v in -d),
                               radius=math.radians(float(m["size"])), shadow=bool(m["shadow"]), specular=0.0)
            lights.append(sun.name)
    if g:
        look.append({"name": "glow", "point": to_root(direction(g["az"], 0.8 * g["height"]) * FAR, yaw).tolist()})
    return {"kind": "night_sky", "paths": {}, "use": {"look": look, "surface": []}, "colliders": [], "lights": lights}
