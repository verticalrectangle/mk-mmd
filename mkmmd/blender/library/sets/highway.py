"""highway: a night highway along a centerline, built to scale (3 km and more) from a few merged meshes.

    [[set]]
    name = "road"
    kind = "highway"
    seed = 7                  # no `points`: core.path.gentle_road(length, seed, turn_deg, leg, z)
    length = 3000.0

Everything is in the set root's frame (the road starts at the root origin, heading -Y by default). Unknown keys raise
ValueError. Keys (defaults):

  Centerline  points [[x, y, z], ...] control points (a Catmull-Rom spline through them) or seed (0) / length (2000) /
              turn_deg (12) / leg (250) / z (0) for a gently winding road (core.path.gentle_road).
  Cross-section  lanes (2: lanes running with the path), oncoming (= lanes; 0 = one-way), lane_width (3.6),
              shoulder (2.5), inner_shoulder (0.9, divided and one-way roads), divided (true), median (3.0), drive
              ("right": the lanes with the path are on the right; "left" mirrors it), dash [3.0, 12.0] (paint, period).
              Lane names fwd1.. (with the path) / opp1.. (against it), 1 = next to the median or centre line.
  Roadside    drop (0.6: embankment height), guardrails (true), barrier (true: concrete median barrier), ds (3: station
              spacing, m), trees = {density (9 per 100 m per side), seed, near (11), far (70), height [6, 14],
              pine (0.6: share of pines, the rest round crowns)} or false.
  Lamps       lamps = {spacing (40), first (20), height (9), arm (3.2), layout ("auto": median poles with two arms on
              divided roads, else outer poles; "outer" / "left" / "right" / "stagger"), power (5000: baked watts of
              one lamp), strength (1500: emission of the lens, reflected in the wet road), halo (4) / halo_strength (2)
              / haze (10) / haze_strength (0.035): additive glow spheres (radius m), core (0.42) / core_strength
              (40), soft (1.1) / soft_strength (7): the bulb and the dithered sphere that screen-space reflections
              see}. lamps = false: no lamps. Poles near tunnels are dropped.
  Real lights lights = {every (4: every n-th lamp gets a real spot light; 0 = none), range [s0, s1] (arc lengths a
              real light may stand in; default the whole road), max (16), power (= lamp power), reach (90: influence
              m), shadow (false), specular (0.5)}. Every other lamp is lit by baked light (`spill` vertex attribute on
              road, roadside, rails, trees), so a long road costs nothing in the renderer and the pools of light match
              the real ones in strength and shape (a 126 degree spot, blend 0.7).
  Roadside   (mkmmd/blender/library/sets/roadside.py) studs = true | {spacing (12), size (0.11), strength (2.4)}: road
              studs on the lane lines (white) and the median-side edges (gold); overpasses = [{s, clearance (5.5), width
              (11), deck (1.0), overhang (14), parapet (1.0), name}]: bridges over the road (lamp poles near them are
              dropped); signs = [{s, side ("right"), size [3.4, 1.8], height (2.2: under the board), offset, name}]:
              boards on two posts, each a `use.surface` `<name>_panel`; markers = true | {every (160), first (40), side}:
              mile-marker plates; pylons = true | {side ("left"), offset (60), spacing (200), first (40), height (34),
              light (40)}: a power line with blinking red lights; masts = [{s, offset (-120), height (95), lights (4),
              light (60), name}]: radio masts with blinking red lights.
  Signs       gantries = [{s, panels (2), panel [4.2, 2.4], clearance (5.6), span ("forward"|"full"), name}],
              billboards = [{s, side ("right"|"left"), offset (m from the centerline to the board centre, default
              half the road + 11), size [12, 5], height (6: m under the board), angle (12: degrees turned toward the
              road), name}]. Each sign panel is its own object (`<set>_<name>_panel<j>`) and a card `use.surface`
              entry (`gantry1_panel1`, `billboard1_panel`: centre, normal toward the traffic, up, size). Signs inside
              tunnels are skipped.
  Tunnels     tunnels = [{from, to}] (arc lengths, m), tunnel = {wall (3.9: wall height), crown (6.4), strip_z (3.3),
              strength (1.4: emission of the gold strips), power (55: baked watts per 3 m of strip), floor (0.25),
              hill (0.9: how much the hill swells between the portals), ceiling (false: a centre strip), lights (0:
              real area lights every n metres inside, 0 = none)}. Twin tubes on divided roads, one tube otherwise;
              gold neon rings on the portals.
  Look        wet (0.7: how soaked the asphalt is), gloss (1.3: coat IOR, how mirror-like), floor (0.32: unlit
              surfaces show this share of their palette colour, so nothing renders black), haze {distance (450), cap
              (1)} (aerial perspective toward the sky's horizon colour), slots = {asphalt = "hl_low", asphalt_hi =
              "overlay", paint = "text", yellow = "gold", steel, concrete, ground, sign = "pine", sign_edge = "foam",
              lamp = "gold", spill, tree, billboard, billboard_edge, portal_ring, ...}: palette slot names or '#hex'.

Card: paths.road = {points (control points), width (paved width, m), lanes [{name, offset, dir}], length, tunnels},
use.look (ahead points on the first forward lane), use.surface (every sign panel), lights (real lights), colliders (a
floor when the road is level)."""
import math

import bpy
import numpy as np

from ....core.path import Path, gentle_road
from ....core.palette import mix
from . import nightgeo as G
from . import nightkit as K
from . import register
from . import roadgeo as R
from . import roadside as RS

DEFAULTS = {
    "seed": 0, "length": 2000.0, "turn_deg": 12.0, "leg": 250.0, "z": 0.0, "points": None,
    "lanes": 2, "oncoming": None, "lane_width": 3.6, "shoulder": 2.5, "inner_shoulder": 0.9, "divided": True,
    "median": 3.0, "drive": "right", "dash": [3.0, 12.0],
    "drop": 0.6, "guardrails": True, "barrier": True, "ds": 3.0,
    "lamps": {}, "lights": {}, "gantries": [], "billboards": [], "tunnels": [], "tunnel": {}, "wet": 0.7,
    "haze": {}, "slots": {}, "floor": 0.32, "trees": {}, "gloss": 1.3,
    "studs": False, "overpasses": [], "signs": [], "markers": False, "pylons": False, "masts": [],
}
LAMP = {"spacing": 40.0, "first": 20.0, "height": 9.0, "arm": 3.2, "layout": "auto", "power": 5000.0,
        "strength": 1500.0, "halo": 4.0, "halo_strength": 2.0, "haze": 10.0, "haze_strength": 0.035,
        "core": 0.42, "core_strength": 40.0, "soft": 1.1, "soft_strength": 7.0}
LIGHTS = {"every": 4, "range": None, "max": 16, "power": None, "reach": 90.0, "shadow": False, "specular": 0.5}
TUNNEL = {"wall": 3.9, "crown": 6.4, "strip_z": 3.3, "strength": 1.4, "power": 55.0, "floor": 0.25, "ceiling": False, "hill": 0.9,
          "lights": 0.0}
HAZE = {"distance": 450.0, "cap": 1.0}
TREES = {"density": 9.0, "seed": 1, "near": 11.0, "far": 70.0, "height": [6.0, 14.0], "pine": 0.6}
SLOTS = {"asphalt": "hl_low", "asphalt_hi": "overlay", "paint": "text", "yellow": "gold", "steel": "muted",
         "concrete": "hl_med", "ground": "base", "sign": "pine", "sign_edge": "foam", "lamp": "gold",
         "reflect_right": "text", "reflect_left": "gold", "tunnel": "hl_med", "billboard": "surface",
         "billboard_edge": "iris", "portal_ring": "gold", "spill": "gold:text:0.3", "tree": "base:pine:0.16",
         "trunk": "hl_low"}


def _merge(base, over):
    out = dict(base)
    if isinstance(over, dict):
        out.update(over)
    return out


class Highway:
    def __init__(self, name, coll, root, spec, palette):
        self.name, self.coll, self.root, self.pal = name, coll, root, palette
        unknown = sorted(set(spec) - set(DEFAULTS) - {"name", "kind", "at", "yaw"})
        if unknown:
            raise ValueError(f"highway {name!r}: unknown keys {unknown} (known: {sorted(DEFAULTS)})")
        c = _merge(DEFAULTS, spec)
        self.c = c
        lamps = c["lamps"]
        self.lamp_on = lamps is not False
        self.lamp = _merge(LAMP, lamps if isinstance(lamps, dict) else {})
        self.lights = _merge(LIGHTS, c["lights"])
        self.tun = _merge(TUNNEL, c["tunnel"])
        self.haze = _merge(HAZE, c["haze"])
        self.floor = float(c["floor"])
        self.tree_cfg = None if c["trees"] is False else _merge(TREES, c["trees"])
        self.slots = _merge(SLOTS, c["slots"])
        pts = c["points"]
        if pts is None:
            pts = gentle_road(float(c["length"]), int(c["seed"]), float(c["turn_deg"]), float(c["leg"]), float(c["z"]))
        self.points = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        sps = int(np.clip(math.ceil(seg.max() / 2.0), 8, 300))
        self.path = Path(self.points, samples_per_seg=sps)
        self.length = self.path.length
        self.sec = R.cross_section(c["lanes"], c["oncoming"], c["lane_width"], c["shoulder"], c["inner_shoulder"],
                                   c["median"], c["divided"], c["drive"], dash=c["dash"])
        self.tunnels = sorted(({"from": max(0.0, float(t["from"])), "to": min(self.length, float(t["to"]))}
                               for t in c["tunnels"]), key=lambda t: t["from"])
        self.ranges = [(t["from"], t["to"]) for t in self.tunnels]
        z = self.path.P[:, 2]
        self.ground_z = float(z.min()) - float(c["drop"])
        self.level = bool(np.ptp(z) < 1e-6)
        self.objects, self.lights_out, self.surfaces = [], [], []
        self.mats = {}
        self.sources = []                  # (positions (m,3), aim, power, cos_half, blend, reach m)

    # ------------------------------------------------------------------------------------------ helpers
    def col(self, key, k=1.0):
        return K.rgb(self.pal, self.slots[key], k)

    def haze_rgb(self):
        return K.rgb(self.pal, K.haze_hex(self.pal))

    def tunnel_haze_rgb(self):
        return K.rgb(self.pal, K.mix(self.pal["hl_low"], self.pal["gold"], 0.2))

    def fog(self, nb, shader):
        """Distance haze toward the horizon colour outside, toward warm dark inside tunnels (vertex attribute `tun`)."""
        tun = nb.attr("tun", "Fac")
        colour = nb.mixc(tun, self.haze_rgb(), self.tunnel_haze_rgb())
        dist = nb.mixf(tun, self.haze["distance"], 140.0)
        return K.haze(nb, shader, colour, dist, self.haze["cap"])

    def spill(self, pts, normals=None):
        """RGB baked lamp light at points (n, 3) from every virtual source, as an (n, 3) array."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        e = np.zeros(len(pts))
        for pos, aim, power, cos_half, blend, reach in self.sources:
            if len(pos):
                e += G.spot_light(pts, normals, pos, aim, power, cos_half, blend, reach)
        e = np.minimum(e, 4.0)
        return e[:, None] * np.array(self.col("spill"))[None, :]

    def obj(self, mesh, key, mats, **kw):
        o = K.to_object(mesh, f"{self.name}_{key}", self.coll, self.root, mats, **kw)
        self.objects.append(o)
        return o

    # ------------------------------------------------------------------------------------------ materials
    def material_tracks(self, key, tracks, rough, wet, hi, gain=None):
        gain = 1.8 if gain is None else gain
        """Asphalt-like surface: dark palette mix with grain, polished wheel tracks and wet patches."""
        m, nb = K.new_material(f"{self.name}_{key}")
        pos = nb.geometry("Position")
        lo, hi_c = self.col("asphalt"), self.col(hi)
        grain = nb.noise(pos, scale=0.45, detail=4.0, rough=0.6)
        patch = nb.noise(pos, scale=0.06, detail=3.0, rough=0.55)
        puddle = nb.smooth(0.52, 0.72, patch)
        albedo = nb.mixc(nb.remap(grain, 0.30, 0.70), lo, hi_c)
        wetness = nb.math("MULTIPLY", puddle, 0.55 * wet + 0.2)
        if tracks:
            u = nb.sep(nb.texcoord("UV"))[0]
            d = nb.math("SUBTRACT", nb.math("ABSOLUTE", u), 0.85)
            track = nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", d, d), -1.0 / (0.36 ** 2)))
            wetness = nb.math("ADD", wetness, nb.math("MULTIPLY", track, 0.6 * wet))
        wetness = nb.math("ADD", wetness, 0.3 * wet, clamp=True)
        wetness = nb.math("MINIMUM", wetness, 1.0)
        albedo = nb.mixc(nb.math("MULTIPLY", wetness, 0.35), albedo, self.col("asphalt", 0.55))
        rgh = nb.mixf(wetness, rough, rough * 0.35)
        coat = nb.math("MULTIPLY", wetness, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", nb.attr("tun", "Fac"), 0.7)))
        bsdf = nb.principled(albedo, rough=rgh, coat=coat, coat_rough=nb.mixf(wetness, 0.25, 0.10), spec=0.4,
                             coat_ior=float(self.c["gloss"]))
        sh = K.spill(nb, bsdf, albedo, gain=gain, floor=self.floor, neutral=0.85)
        nb.output(self.fog(nb, sh))
        return m

    def material_paint(self, key, slot, k):
        m, nb = K.new_material(f"{self.name}_{key}")
        alb = self.col(slot, k)
        bsdf = nb.principled(alb, rough=0.55, spec=0.4)
        sh = K.spill(nb, bsdf, alb, gain=1.6, floor=self.floor, neutral=0.6)
        nb.output(self.fog(nb, sh))
        return m

    def material_solid(self, key, slot, rough=0.8, metal=0.0, k=1.0, spill=True, noise=0.0, gain=1.0):
        m, nb = K.new_material(f"{self.name}_{key}")
        alb = self.col(slot, k)
        if noise:
            alb = nb.mixc(nb.remap(nb.noise(nb.geometry("Position"), scale=0.18, detail=3.0), 0.3, 0.7),
                          alb, self.col(slot, k * (1.0 + noise)))
        bsdf = nb.principled(alb, rough=rough, metal=metal, spec=0.4)
        sh = K.spill(nb, bsdf, alb, gain=gain, floor=self.floor, neutral=0.7) if spill else bsdf
        nb.output(self.fog(nb, sh))
        return m

    def material_emit(self, key, slot, strength):
        m, nb = K.new_material(f"{self.name}_{key}")
        nb.output(nb.emission(self.col(slot), strength))
        return m

    def material_tunnel(self):
        """Tunnel concrete lit only by the baked strip light (no world light leaks in): panel seams every 3 m."""
        m, nb = K.new_material(f"{self.name}_tunnel")
        v = nb.sep(nb.texcoord("UV"))[1]
        f = nb.math("FRACT", nb.math("DIVIDE", v, 3.0))
        edge = nb.math("MINIMUM", f, nb.math("SUBTRACT", 1.0, f))
        seam = nb.math("SUBTRACT", 1.0, nb.smooth(0.0, 0.03, edge))
        alb = nb.mixc(nb.math("MULTIPLY", seam, 0.7), self.col("tunnel", 0.9), self.col("tunnel", 0.3))
        sh = K.spill(nb, None, alb, gain=1.4, floor=self.tun["floor"], neutral=0.7)
        nb.output(K.haze(nb, sh, self.tunnel_haze_rgb(), 140.0, 0.9))
        return m

    def build_materials(self):
        w = float(self.c["wet"])
        M = self.mats
        M["asphalt"] = self.material_tracks("asphalt", True, 0.5, w, "asphalt_hi")
        M["shoulder"] = self.material_tracks("shoulder", False, 0.62, w * 0.8, "asphalt_hi")
        M["median"] = self.material_solid("median", "concrete", 0.85, k=0.65, noise=0.25, gain=0.4)
        M["white"] = self.material_paint("paint", "paint", 0.62)
        M["yellow"] = self.material_paint("paint_yellow", "yellow", 0.55)
        M["ground"] = self.material_solid("ground", "ground", 0.95, k=0.85, noise=0.45, gain=1.6)
        M["steel"] = self.material_solid("steel", "steel", 0.42, metal=0.7, k=0.7, gain=0.4)
        M["concrete"] = self.material_solid("concrete", "concrete", 0.85, k=0.7, noise=0.2, gain=0.4)
        M["lens"] = self.material_emit("lens", "lamp", self.lamp["strength"])
        M["reflect_r"] = self.material_emit("reflect_r", "reflect_right", 2.2)
        M["reflect_l"] = self.material_emit("reflect_l", "reflect_left", 2.2)
        M["halo"] = K.glow_gauss_material(f"{self.name}_halo", self.col("lamp"), float(self.lamp["halo_strength"]), 0.30)
        M["haze"] = K.glow_gauss_material(f"{self.name}_hazeglow", self.col("lamp"), float(self.lamp["haze_strength"]), 0.45)
        M["core"] = self.material_emit("core", "lamp", float(self.lamp["core_strength"]))
        M["soft"] = K.glow_solid_material(f"{self.name}_soft", self.col("lamp"), float(self.lamp["soft_strength"]), 0.5)
        M["lamp_small"] = self.material_emit("lamp_small", "lamp", 9.0)
        M["neon"] = self.material_emit("neon", "billboard_edge", 1.3)
        M["neon_gold"] = self.material_emit("neon_gold", "portal_ring", 2.0)
        M["tunnel_light"] = self.material_emit("tunnel_light", "lamp", float(self.tun["strength"]))
        M["tunnel"] = self.material_tunnel()
        M["tree"] = self.material_solid("tree", "tree", 0.9, k=1.0, gain=1.2, noise=0.25)
        M["trunk"] = self.material_solid("trunk", "trunk", 0.9, k=1.0, gain=0.8)

    # ------------------------------------------------------------------------------------------ build
    def build(self):
        self.build_materials()
        self.collect_lamps()
        self.collect_tunnel_sources()
        s = self.stations()
        self.road_ribbon(s)
        self.roadside(s)
        if self.c["guardrails"] or self.c["barrier"]:
            self.rails(s)
        if self.lamp_on:
            self.lamp_objects()
            self.real_lights()
        for i, g in enumerate(self.c["gantries"]):
            if not self.in_tunnel(g["s"]):
                self.gantry(i, g)
        for i, b in enumerate(self.c["billboards"]):
            if not self.in_tunnel(b["s"]):
                self.billboard(i, b)
        if self.c["studs"]:
            RS.studs(self, self.c["studs"], s)
        if self.c["overpasses"]:
            RS.overpasses(self, self.c["overpasses"])
        if self.c["signs"]:
            RS.signs(self, self.c["signs"])
        if self.c["markers"]:
            RS.markers(self, self.c["markers"])
        if self.c["pylons"]:
            RS.pylons(self, self.c["pylons"])
        if self.c["masts"]:
            RS.masts(self, self.c["masts"])
        for i, t in enumerate(self.tunnels):
            self.tunnel(i, t)
        if self.tree_cfg:
            self.trees()
        return self.card()

    def in_tunnel(self, s, margin=12.0):
        return any(a - margin <= float(s) <= b + margin for a, b in self.ranges)

    def stations(self):
        extra = []
        for pat in self.sec.dashed():
            on, period, phase = pat
            st, en = G.dash_edges(0.0, self.length, on, period, phase)
            extra += list(st) + list(en)
        for t in self.tunnels:
            extra += [t["from"] - 0.06, t["from"], t["to"], t["to"] + 0.06]
        for g in self.c["gantries"]:
            extra.append(float(g["s"]))
        return G.stations(self.length, float(self.c["ds"]), extra)

    # ------------------------------------------------------------------------------------------ lamps
    def collect_lamps(self):
        self.arms = None
        if not self.lamp_on:
            return
        L = self.lamp
        lay = R.lamp_layout(self.sec, self.length, L["layout"], L["spacing"], L["first"], self.ranges)
        clear = RS.lamp_exclusion(self.c)                 # no pole and no arm through an overpass deck
        if clear and len(lay["s"]):
            keep = {"": ~G.in_ranges(np.asarray(lay["s"]), clear, 0.0)}
            if "pole_s" in lay:
                keep["pole_"] = ~G.in_ranges(np.asarray(lay["pole_s"]), clear, 0.0)
            for k, v in list(lay.items()):
                m = keep["pole_"] if k.startswith("pole_") and "pole_" in keep else keep[""]
                if hasattr(v, "__len__") and len(v) == len(m):
                    lay[k] = np.asarray(v)[m]
        self.layout = lay
        n = len(lay["s"])
        if not n:
            self.arms = None
            return
        sign, x, s = lay["sign"], lay["x"], lay["s"]
        T, Lf, U = self.path.frame(s)
        head_x = x + sign * L["arm"]
        base = self.path.point(s) + Lf * x[:, None]
        head = self.path.point(s) + Lf * head_x[:, None] + U * (L["height"] + 0.08)
        win = self.lights["range"]
        mask = R.real_light_mask(s, int(self.lights["every"]), win, int(self.lights["max"]))
        self.arms = {"s": s, "x": x, "sign": sign, "T": T, "L": Lf, "U": U, "head": head, "base": base,
                     "real": mask}
        virtual = head[~mask]
        if len(virtual):
            self.sources.append((virtual, np.array([0.0, 0.0, -1.0]), L["power"], math.cos(math.radians(63)), 0.7, 34.0))

    def lamp_objects(self):
        a = self.arms
        if a is None:
            return
        L = self.lamp
        H, A = L["height"], L["arm"]
        lay = self.layout
        ps = lay["pole_s"]
        # poles (one per pole position)
        Tp, Lp, Up = self.path.frame(ps)
        Pp = self.path.point(ps) + Lp * lay["pole_x"][:, None]
        Rp = G.frame_R(Tp, Lp, Up)
        metal = G.Mesh()
        pole = G.cylinder(0.17, 0.09, H + 0.1, 8)
        plate = G.cylinder(0.27, 0.27, 0.18, 8)
        metal.add(G.instance(pole, Rp, Pp), 0, smooth=True)
        metal.add(G.instance(plate, Rp, Pp), 0, smooth=False)
        # arms: bars + luminaire, built pointing along local +Y
        def bar(p0, p1, w=0.11, h=0.11):
            return G.bars([p0], [p1], w, h)
        arm_geo = [bar((0, 0, H - 0.25), (0, 0.30 * A, H + 0.30)), bar((0, 0.30 * A, H + 0.30), (0, 0.72 * A, H + 0.38)),
                   bar((0, 0.72 * A, H + 0.38), (0, A, H + 0.22))]
        head_box = G.place(G.boxes([(0.44, 1.05, 0.16)]), [np.eye(3)], [[0.0, A, H + 0.12]])
        Ra = G.frame_R(a["T"], a["L"], a["U"]) @ G.rot_z(np.where(a["sign"] > 0, 0.0, math.pi))
        for g in arm_geo + [head_box]:
            metal.add(G.instance(g, Ra, a["base"]), 0, smooth=False)
        self.obj(metal, "lamps", [self.mats["steel"]])
        lens = G.quad(0.32, 0.88, "XY")                                     # normal +Z: turn it to face down
        lens = G.Geom(lens.V + np.array([0.0, A, H + 0.03]), lens.Q[:, ::-1], None, lens.UV)
        lm = G.Mesh()
        lm.add(G.instance(lens, Ra, a["base"]), 0)
        self.obj(lm, "lamp_lens", [self.mats["lens"]], shadow=False)
        centres = a["head"] - np.array([0.0, 0.0, 0.3])
        for key, radius, nu, nv, mat in (("lamp_haze", L["haze"], 10, 6, "haze"), ("lamp_halo", L["halo"], 12, 8, "halo"),
                                         ("lamp_soft", L["soft"], 10, 6, "soft"), ("lamp_core", L["core"], 10, 6, "core")):
            if radius > 0:
                hm = G.Mesh()
                hm.add(G.instance(G.sphere(radius, nu, nv), None, centres), 0, smooth=True)
                self.obj(hm, key, [self.mats[mat]], shadow=False, glossy=(key in ("lamp_core", "lamp_soft")))

    def real_lights(self):
        a = self.arms
        if a is None or not a["real"].any():
            return
        for k in np.flatnonzero(a["real"]):
            lt = K.make_light(f"{self.name}_lamp{k:03d}", self.coll, self.root, "SPOT", tuple(a["head"][k] - [0, 0, 0.05]),
                              float(self.lights["power"] or self.lamp["power"]), self.col("spill"), (0.0, 0.0, -1.0), radius=0.22,
                              shadow=bool(self.lights["shadow"]), reach=float(self.lights["reach"]), spot_deg=126.0,
                              blend=0.7, specular=float(self.lights["specular"]))
            self.lights_out.append(lt.name)

    # ------------------------------------------------------------------------------------------ road
    def road_ribbon(self, s):
        P = self.path.point(s)
        T, Lf, U = self.path.frame(s)
        s_mid = 0.5 * (s[:-1] + s[1:])
        mesh = G.Mesh()
        order = ["asphalt", "shoulder", "median", "white", "yellow"]
        for b in self.sec.bands:
            xs = np.array([b["x0"], b["x1"]])
            V = P[:, None, :] + Lf[:, None, :] * xs[None, :, None]
            uv = np.stack([np.broadcast_to(xs - b["uc"], (len(s), 2)), np.broadcast_to(s[:, None], (len(s), 2))], -1)
            mat = R.band_material(b, s_mid)[:, None]
            nrm = np.broadcast_to(U[:, None, :], V.shape).reshape(-1, 3)
            sp = self.spill(V.reshape(-1, 3), nrm).reshape(len(s), 2, 3)
            mesh.add_grid(V, mat=mat, uv=uv, smooth=True, spill=sp,
                          tun=np.broadcast_to(G.in_ranges(s, self.ranges).astype(float)[:, None], (len(s), 2)))
        self.obj(mesh, "road", [self.mats[k] for k in order])
        self.s_stations = s
        self.frames = (P, T, Lf, U)

    def roadside(self, s):
        P, T, Lf, U = self.frames
        c = self.c
        drop = float(c["drop"])
        d = R.edge_offsets(drop)
        mesh = G.Mesh()
        far = d > d[2] + 1e-9
        for side, x_edge in ((+1.0, self.sec.x_max), (-1.0, self.sec.x_min)):
            x = x_edge + side * d
            dz = R.ground_dz(d, drop)
            # far lines settle onto the flat ground plane whatever the road does
            blend = G.smoothstep(d[2], d[-1], d)
            z_rel = dz[None, :] * np.ones((len(s), 1))
            V = P[:, None, :] + Lf[:, None, :] * x[None, :, None] + U[:, None, :] * z_rel[:, :, None]
            V[:, :, 2] = V[:, :, 2] * (1 - blend)[None, :] + self.ground_z * blend[None, :]
            nrm = np.zeros_like(V)
            nrm[..., 2] = 1.0
            sp = self.spill(V.reshape(-1, 3), nrm.reshape(-1, 3)).reshape(len(s), len(d), 3)
            uv = np.stack([np.broadcast_to(d[None, :], (len(s), len(d))), np.broadcast_to(s[:, None], (len(s), len(d)))], -1)
            mesh.add_grid(V, uv=uv, flip=(side > 0), smooth=True, spill=sp,
                          tun=np.broadcast_to(G.in_ranges(s, self.ranges).astype(float)[:, None], (len(s), len(d))))
        self.obj(mesh, "roadside", [self.mats["ground"]])
        # the flat ground out to the horizon (haze takes over long before)
        ctr = self.points[:, :2].mean(0)
        ext = 12000.0
        gm = G.Mesh()
        gq = np.array([[ctr[0] - ext, ctr[1] - ext, self.ground_z - 0.25], [ctr[0] + ext, ctr[1] - ext, self.ground_z - 0.25],
                       [ctr[0] + ext, ctr[1] + ext, self.ground_z - 0.25], [ctr[0] - ext, ctr[1] + ext, self.ground_z - 0.25]])
        gm.add(G.Geom(gq, [[0, 1, 2, 3]], None, np.zeros((4, 2))), 0)
        self.obj(gm, "ground", [self.mats["ground"]])

    # ------------------------------------------------------------------------------------------ rails
    def rails(self, s):
        """Guardrails along both outer edges (W-beam ring, posts, white reflectors) and the concrete median barrier
        (gold reflectors); none inside tunnels."""
        P, T, Lf, U = self.frames
        steel, conc, white, gold = G.Mesh(), G.Mesh(), G.Mesh(), G.Mesh()
        post_s = np.arange(2.0, self.length - 1.0, 4.0)
        post_s = post_s[~G.in_ranges(post_s, self.ranges, 6.0)]
        keep = np.flatnonzero(~G.in_ranges(0.5 * (s[:-1] + s[1:]), self.ranges, 4.0))
        if self.c["guardrails"]:
            prof = np.array([[0.045, 0.53], [0.075, 0.68], [0.045, 0.83], [-0.045, 0.83], [-0.045, 0.53]])
            for side, x_edge in ((+1.0, self.sec.x_max), (-1.0, self.sec.x_min)):
                x = x_edge + side * 0.65
                lat = x - side * prof[:, 0]                                    # the front faces the road
                ring = P[:, None, :] + Lf[:, None, :] * lat[None, :, None] + U[:, None, :] * prof[None, :, 1:2]
                ring = np.concatenate([ring, ring[:, :1]], axis=1)
                for run in _runs(keep):
                    sub = ring[run[0]:run[-1] + 2]
                    sp = self.spill(sub.reshape(-1, 3)).reshape(len(sub), sub.shape[1], 3)
                    steel.add_grid(sub, flip=(side < 0), smooth=False, spill=sp)
                Tp, Lp, Up = self.path.frame(post_s)
                base = self.path.point(post_s) + Lp * (x + side * 0.16) - Up * 0.2
                steel.add(G.instance(G.boxes([(0.10, 0.14, 1.15)], base=True), G.frame_R(Tp, Lp, Up), base), 0)
                self._reflectors(white, post_s[::2], x - side * (prof[1, 0] + 0.03), side, 0.68)
        if self.c["barrier"] and self.sec.divided:
            prof = np.array([[0.30, 0.0], [0.30, 0.10], [0.12, 0.45], [0.10, 0.85], [-0.10, 0.85], [-0.12, 0.45],
                             [-0.30, 0.10], [-0.30, 0.0]])
            ring = P[:, None, :] + Lf[:, None, :] * prof[None, :, 0:1] + U[:, None, :] * prof[None, :, 1:2]
            for run in _runs(keep):
                sub = ring[run[0]:run[-1] + 2]
                sp = self.spill(sub.reshape(-1, 3)).reshape(len(sub), sub.shape[1], 3)
                conc.add_grid(sub, flip=True, smooth=False, spill=sp)
            rs = np.arange(4.0, self.length - 1.0, 8.0)
            rs = rs[~G.in_ranges(rs, self.ranges, 6.0)]
            self._reflectors(gold, rs, 0.14, -1.0, 0.62)
            self._reflectors(gold, rs, -0.14, +1.0, 0.62)
        for mesh, key, mat in ((steel, "rails", "steel"), (conc, "barrier", "concrete"), (white, "reflectors", "reflect_r"),
                               (gold, "reflectors_gold", "reflect_l")):
            if len(mesh):
                self.obj(mesh, key, [self.mats[mat]], shadow=False)

    def _reflectors(self, mesh, rs, x, side, z):
        """Small reflector plates at lateral x, height z, facing the road (side +1: road on its -L side)."""
        Tr, Lr, Ur = self.path.frame(rs)
        rp = self.path.point(rs) + Lr * x + Ur * z
        quad = G.quad(0.16, 0.07, "YZ")
        Rq = G.frame_R(Tr, Lr, Ur) @ G.rot_z(-math.pi / 2 if side > 0 else math.pi / 2)
        mesh.add(G.instance(quad, Rq, rp), 0)

    # ------------------------------------------------------------------------------------------ placing
    def put(self, g, s, lat=0.0, yaw=0.0, dz=0.0):
        """A Geom built in the road frame at arc length s (local +X along the road, +Y left, +Z up), into the root frame."""
        T, Lf, U = self.path.frame(np.array([float(s)]))
        Rm = G.frame_R(T, Lf, U) @ G.rot_z(yaw)
        base = self.path.point(np.array([float(s)])) + Lf * lat + U * dz
        return G.instance(g, Rm, base), Rm[0], base[0]

    def ground_dz(self, x):
        """Height of the verge / slope at lateral x relative to the road surface."""
        d = np.where(x > 0, x - self.sec.x_max, self.sec.x_min - x)
        return float(R.ground_dz(np.maximum(d, 0.0), float(self.c["drop"])))

    # ------------------------------------------------------------------------------------------ signs
    def sign_material(self, w, h):
        """Pine board with a foam border, lit from above by its own lamps (emission only: no light, no shadow acne)."""
        key = f"sign_{w:.2f}x{h:.2f}"
        if key in self.mats:
            return self.mats[key]
        m, nb = K.new_material(f"{self.name}_{key}")
        u, v, _ = nb.sep(nb.texcoord("UV"))
        e = nb.math("MINIMUM", nb.math("MINIMUM", u, nb.math("SUBTRACT", w, u)),
                    nb.math("MINIMUM", v, nb.math("SUBTRACT", h, v)))
        border = nb.math("SUBTRACT", 1.0, nb.smooth(0.10, 0.14, e))
        col = nb.mixc(border, self.col("sign"), self.col("sign_edge"))
        lit = nb.math("ADD", 0.55, nb.math("MULTIPLY", nb.math("DIVIDE", v, h), 0.6))
        sh = nb.emission(nb.vmath("MULTIPLY", col, lit), 1.0)
        nb.output(self.fog(nb, sh))
        self.mats[key] = m
        return m

    def billboard_material(self, w, h):
        """A blank, dim billboard face for type or pictures to go on: a soft light wash falling from the lamps on top."""
        key = f"billboard_{w:.2f}x{h:.2f}"
        if key in self.mats:
            return self.mats[key]
        m, nb = K.new_material(f"{self.name}_{key}")
        col = self.col("billboard")
        v = nb.sep(nb.texcoord("UV"))[1]
        wash = nb.math("ADD", 0.5, nb.math("MULTIPLY", nb.math("POWER", nb.math("DIVIDE", v, h), 3.0), 1.2))
        sh = K.spill(nb, None, col, gain=1.8, floor=0.0, neutral=0.5)
        sh = nb.add_shader(sh, nb.emission(col, wash))
        nb.output(self.fog(nb, sh))
        self.mats[key] = m
        return m

    def gantry(self, i, g):
        """An overhead sign gantry: two posts, a lattice truss, sign panels facing the traffic that comes along +T."""
        sec = self.sec
        gname = g.get("name", f"gantry{i + 1}")
        s = float(g["s"])
        n = int(g.get("panels", 2))
        pw, ph = (float(v) for v in g.get("panel", (4.2, 2.4)))
        clear = float(g.get("clearance", 5.6))
        fwd = next((c for c in sec.carriageways if c["dir"] == 1), None)
        span = g.get("span", "forward" if sec.divided else "full")
        if span == "forward" and fwd is not None and sec.divided:
            xs = sorted([fwd["x_out"] + fwd["side"] * 1.6, fwd["x_in"] - fwd["side"] * 0.25])
        else:
            xs = [sec.x_min - 1.6, sec.x_max + 1.6]
        x_lo, x_hi = xs
        ztop = clear + ph
        zb, zt = ztop + 0.30, ztop + 1.05
        steel = G.Mesh()
        eye = np.eye(3)
        post_h = zt + 0.5
        posts = G.place(G.boxes([(0.55, 0.55, post_h + 0.6)] * 2, base=True), np.broadcast_to(eye, (2, 3, 3)),
                        [[0.0, x_lo, -0.6], [0.0, x_hi, -0.6]])
        steel.add(self.put(posts, s)[0], 0)
        nb_ = max(2, int(round((x_hi - x_lo) / 1.9)))
        xk = np.linspace(x_lo, x_hi, nb_ + 1)
        p0 = [(0.0, x_lo, zb), (0.0, x_lo, zt)] + [(0.0, x, zb) for x in xk[1:-1]]
        p1 = [(0.0, x_hi, zb), (0.0, x_hi, zt)] + [(0.0, x, zt) for x in xk[1:-1]]
        for k in range(nb_):
            p0.append((0.0, xk[k], zb if k % 2 == 0 else zt))
            p1.append((0.0, xk[k + 1], zt if k % 2 == 0 else zb))
        steel.add(self.put(G.bars(p0, p1, 0.15, 0.15), s)[0], 0)
        spacing = (x_hi - x_lo) / n
        pw = min(pw, spacing * 0.92)
        sign_mat = self.sign_material(pw, ph)
        lamps = G.Mesh()
        for j in range(n):
            xc = x_lo + (j + 0.5) * spacing
            zc = clear + ph / 2
            back = G.place(G.boxes([(0.10, pw, ph)], skip=(1,)), [eye], [[-0.32, xc, zc]])
            hang = G.bars([(-0.05, xc - pw * 0.33, zb), (-0.05, xc + pw * 0.33, zb)],
                          [(-0.30, xc - pw * 0.33, ztop - 0.02), (-0.30, xc + pw * 0.33, ztop - 0.02)], 0.08, 0.08)
            face = G.quad(pw, ph, "YZ", flip=True)
            face = G.Geom(face.V + np.array([-0.37, xc, zc]), face.Q, None, face.UV)
            panel = G.Mesh()
            panel.add(self.put(back, s)[0], 0)
            panel.add(self.put(hang, s)[0], 0)
            panel.add(self.put(face, s)[0], 1)
            pname = f"{gname}_panel{j + 1}"
            self.obj(panel, pname, [self.mats["steel"], sign_mat], shadow=False)
            # panel lights: small heads on short arms above the panel
            for dx in (-0.3, 0.3):
                head = G.place(G.boxes([(0.32, 0.30, 0.10)]), [eye], [[-0.62, xc + dx * pw, ztop + 0.36]])
                lamps.add(self.put(head, s)[0], 0)
            T_, Lf_, U_ = self.path.frame(np.array([s]))
            Rm = G.frame_R(T_, Lf_, U_)[0]
            base = self.path.point(np.array([s]))[0]
            centre = base + Rm @ np.array([-0.374, xc, zc])
            self.surfaces.append({"name": pname, "center": centre.tolist(), "normal": (-Rm[:, 0]).tolist(),
                                  "up": Rm[:, 2].tolist(), "size": [pw, ph]})
        self.obj(steel, f"{gname}", [self.mats["steel"]])
        self.obj(lamps, f"{gname}_lamps", [self.mats["lamp_small"]], shadow=False)

    def billboard(self, i, b):
        """A roadside billboard on two posts, turned toward the traffic, with a lit face and lamps on top."""
        sec = self.sec
        bname = b.get("name", f"billboard{i + 1}")
        s = float(b["s"])
        side = -1.0 if b.get("side", "right") == "right" else 1.0
        W, H = (float(v) for v in b.get("size", (12.0, 5.0)))
        hgt = float(b.get("height", 6.0))
        off = float(b.get("offset", sec.half + 11.0))
        ang = math.radians(float(b.get("angle", 12.0)))
        yaw = -ang if side < 0 else ang
        xc = side * off
        dz = self.ground_dz(xc) - 0.3
        eye = np.eye(3)
        steel, lamps, neon = G.Mesh(), G.Mesh(), G.Mesh()
        zc = hgt + H / 2
        # posts, frame, catwalk (local frame of the board: x toward the back, y across, z up; the face looks along -x)
        for sy in (-1.0, 1.0):
            post = G.cylinder(0.24, 0.24, hgt + H * 0.8 + 0.3, 10)
            steel.add(self.put(G.Geom(post.V + np.array([0.75, sy * (W / 2 - 1.8), 0.0]), post.Q, post.T, post.UV),
                               s, xc, yaw, dz)[0], 0, smooth=True)
        fr = G.bars([(0.1, -W / 2, hgt), (0.1, -W / 2, hgt + H), (0.1, -W / 2, hgt), (0.1, W / 2, hgt)] +
                    [(0.35, -W / 2 + 0.5, hgt - 0.15), (0.35, -W / 2 + 0.5, hgt + H + 0.15)],
                    [(0.1, W / 2, hgt), (0.1, W / 2, hgt + H), (0.1, -W / 2, hgt + H), (0.1, W / 2, hgt + H),
                     (0.35, W / 2 - 0.5, hgt - 0.15), (0.35, W / 2 - 0.5, hgt + H + 0.15)], 0.30, 0.30)
        steel.add(self.put(fr, s, xc, yaw, dz)[0], 0)
        back = G.place(G.boxes([(0.22, W, H)], skip=(1,)), [eye], [[0.0, 0.0, zc]])
        steel.add(self.put(back, s, xc, yaw, dz)[0], 0)
        arms = []
        lamp_pos = []
        n_lamps = max(3, int(round(W / 1.6)))
        for k in range(n_lamps):
            y = -W / 2 + (k + 0.5) * W / n_lamps
            arms.append(((0.0, y, hgt + H + 0.1), (-1.7, y, hgt + H + 0.55)))
            lamp_pos.append((-1.7, y, hgt + H + 0.5))
        steel.add(self.put(G.bars([a for a, _ in arms], [b_ for _, b_ in arms], 0.09, 0.09), s, xc, yaw, dz)[0], 0)
        for lp in lamp_pos:
            head = G.place(G.boxes([(0.55, 0.38, 0.12)]), [eye], [[lp[0], lp[1], lp[2]]])
            lamps.add(self.put(head, s, xc, yaw, dz)[0], 0)
        # neon edge
        e = 0.05
        ring_p0 = [(-0.17, -W / 2 - e, hgt - e), (-0.17, W / 2 + e, hgt - e), (-0.17, W / 2 + e, hgt + H + e),
                   (-0.17, -W / 2 - e, hgt + H + e)]
        ring_p1 = [ring_p0[1], ring_p0[2], ring_p0[3], ring_p0[0]]
        neon.add(self.put(G.bars(ring_p0, ring_p1, 0.10, 0.10), s, xc, yaw, dz)[0], 0)
        # face: a grid so the lamp light can fall off down the board
        nx, nz = 24, 10
        yy, zz = np.linspace(-W / 2, W / 2, nx + 1), np.linspace(hgt, hgt + H, nz + 1)
        grid = np.stack([np.full((nx + 1, nz + 1), -0.11), np.broadcast_to(yy[:, None], (nx + 1, nz + 1)),
                         np.broadcast_to(zz[None, :], (nx + 1, nz + 1))], -1)
        uv = np.stack([np.broadcast_to((yy - yy[0])[:, None], (nx + 1, nz + 1)),
                       np.broadcast_to((zz - zz[0])[None, :], (nx + 1, nz + 1))], -1)
        Tl, Ll, Ul = self.path.frame(np.array([s]))
        Rm = (G.frame_R(Tl, Ll, Ul) @ G.rot_z(yaw))[0]
        base = self.path.point(np.array([s]))[0] + Ll[0] * xc + Ul[0] * dz
        Vw = base + np.einsum("ij,abj->abi", Rm, grid)
        lp_w = base + np.einsum("ij,nj->ni", Rm, np.array(lamp_pos))
        face_n = -Rm[:, 0]
        aim = Rm @ np.array([0.45, 0.0, -0.9])
        aim = aim / np.linalg.norm(aim)
        e_f = G.spot_light(Vw.reshape(-1, 3), np.broadcast_to(face_n, (Vw.shape[0] * Vw.shape[1], 3)), lp_w, aim,
                           1500.0, math.cos(math.radians(80)), 0.9).reshape(nx + 1, nz + 1)
        spill_col = e_f[..., None] * np.array(self.col("lamp"))[None, None, :]
        face = G.Mesh()
        face.add_grid(Vw, mat=0, uv=uv, flip=True, smooth=False, spill=spill_col)
        bb_mat = self.billboard_material(W, H)
        self.obj(face, f"{bname}_panel", [bb_mat], shadow=False)
        self.obj(steel, bname, [self.mats["steel"]])
        self.obj(lamps, f"{bname}_lamps", [self.mats["lamp_small"]], shadow=False)
        self.obj(neon, f"{bname}_edge", [self.mats["neon"]], shadow=False)
        centre = base + Rm @ np.array([-0.114, 0.0, zc])
        self.surfaces.append({"name": f"{bname}_panel", "center": centre.tolist(), "normal": face_n.tolist(),
                              "up": Rm[:, 2].tolist(), "size": [W, H]})

    # ------------------------------------------------------------------------------------------ tunnels
    def tube_specs(self):
        """(x_left, x_right) of every tunnel tube: one per carriageway on divided roads, else one for the whole road."""
        sec, walk = self.sec, 0.8
        if sec.divided:
            out = []
            for cw in sec.carriageways:
                a, b = cw["x_in"], cw["x_out"] + cw["side"] * walk
                out.append((max(a, b), min(a, b)))
            return out
        return [(sec.x_max + walk, sec.x_min - walk)]

    def collect_tunnel_sources(self):
        self.tubes = self.tube_specs()
        tun = self.tun
        for t in self.tunnels:
            ss = np.arange(t["from"] + 1.5, t["to"], 3.0)
            if not len(ss):
                continue
            T, Lf, U = self.path.frame(ss)
            Pt = self.path.point(ss)
            pos, aim = [], []
            for xl, xr in self.tubes:
                for xw, inward in ((xl, -1.0), (xr, +1.0)):
                    pos.append(Pt + Lf * (xw + inward * 0.12) + U * tun["strip_z"])
                    aim.append(Lf * inward)
            # a strip lights its own wall, the ceiling, the floor and the wall across: a lamp radiating every way
            self.sources.append((np.vstack(pos), np.vstack(aim), tun["power"], -1.0, 0.0, 30.0))

    def tunnel(self, i, t):
        """One tunnel: tube walls with gold light strips, a hill over it, portal walls with a gold ring."""
        s0, s1 = t["from"], t["to"]
        tun = self.tun
        st = self.s_stations
        sel = (st >= s0 - 1e-6) & (st <= s1 + 1e-6)
        ss = st[sel]
        P, T, Lf, U = (a[sel] for a in self.frames)
        wall, light, conc, earth, ring = G.Mesh(), G.Mesh(), G.Mesh(), G.Mesh(), G.Mesh()
        openings = []
        drop = float(self.c["drop"])
        for xl, xr in self.tubes:
            sz = tun["strip_z"]
            prof = R.tube_profile(xl, xr, tun["wall"], tun["crown"], rungs=(1.1, 2.2, sz - 0.45, sz - 0.14, sz + 0.14, sz + 0.45))
            openings.append(R.tube_profile(xl, xr, tun["wall"], tun["crown"], rungs=()))
            V = P[:, None, :] + Lf[:, None, :] * prof[None, :, 0:1] + U[:, None, :] * prof[None, :, 1:2]
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
            uv = np.stack([np.broadcast_to(arc, V.shape[:2]), np.broadcast_to(ss[:, None], V.shape[:2])], -1)
            sp = self.spill(V.reshape(-1, 3)).reshape(V.shape)
            wall.add_grid(V, uv=uv, smooth=True, spill=sp)
            z0, z1 = tun["strip_z"] - 0.13, tun["strip_z"] + 0.13
            for xw, inward, flip in ((xl, -1.0, False), (xr, +1.0, True)):
                Vs = P[:, None, :] + Lf[:, None, :] * (xw + inward * 0.04) + U[:, None, :] * np.array([z0, z1])[None, :, None]
                light.add_grid(Vs, flip=flip, smooth=False)
            if tun.get("ceiling"):
                xc = 0.5 * (xl + xr)
                Vc = P[:, None, :] + Lf[:, None, :] * np.array([xc + 0.18, xc - 0.18])[None, :, None] \
                    + U[:, None, :] * (tun["crown"] - 0.03)
                light.add_grid(Vc, smooth=False)
        xl_all, xr_all = max(t_[0] for t_ in self.tubes), min(t_[1] for t_ in self.tubes)
        mp = R.mound_profile(xl_all, xr_all, tun["crown"], -drop, spread=26.0)
        t_ = (ss - s0) / max(s1 - s0, 1e-6)
        rise = 1.0 + float(tun["hill"]) * np.sin(np.pi * t_) ** 0.8                    # the hill swells between the portals
        zm = (-drop) + (mp[None, :, 1] + drop) * rise[:, None]
        Vm = P[:, None, :] + Lf[:, None, :] * mp[None, :, 0:1] + U[:, None, :] * zm[:, :, None]
        Vm[:, [0, -1], 2] = self.ground_z
        earth.add_grid(Vm, flip=True, smooth=False)
        quads = R.facade(openings, mp, -drop)
        for end, rev, sgn in ((s0, False, -1.0), (s1, True, +1.0)):
            Pe = self.path.point(np.array([end]))[0]
            Te, Le, Ue = (a[0] for a in self.path.frame(np.array([end])))
            C = Pe + Le * quads[..., 0:1] + Ue * quads[..., 1:2]
            C[..., 2] = np.where(np.isclose(quads[..., 1], -drop), self.ground_z, C[..., 2])
            Q = np.arange(len(quads) * 4).reshape(-1, 4)
            conc.add(G.Geom(C.reshape(-1, 3), Q[:, ::-1] if rev else Q, None, quads.reshape(-1, 2)), 0)
            for prof in openings:
                pts = Pe + Le * prof[:, 0:1] + Ue * prof[:, 1:2] + Te * (0.14 * sgn)
                ring.add(G.bars(pts[:-1], pts[1:], 0.22, 0.16))
        name = f"tunnel{i + 1}"
        self.obj(wall, f"{name}_walls", [self.mats["tunnel"]])
        self.obj(light, f"{name}_lights", [self.mats["tunnel_light"]], shadow=False)
        self.obj(conc, f"{name}_portals", [self.mats["concrete"]])
        self.obj(earth, f"{name}_hill", [self.mats["ground"]])
        self.obj(ring, f"{name}_rings", [self.mats["neon_gold"]], shadow=False)
        every = float(tun["lights"])
        if every > 0:
            for k, sl in enumerate(np.arange(s0 + every / 2, s1, every)):
                for j, (xl, xr) in enumerate(self.tubes):
                    pos = self.path.offset(float(sl), 0.5 * (xl + xr), tun["crown"] - 0.4)
                    lt = K.make_light(f"{self.name}_tunnel{i + 1}_{k:03d}_{j}", self.coll, self.root, "AREA", tuple(pos),
                                      900.0, self.col("lamp"), (0.0, 0.0, -1.0), size=(5.0, 0.6), reach=45.0)
                    self.lights_out.append(lt.name)

    # ------------------------------------------------------------------------------------------ trees
    def trees(self):
        """Dark roadside trees (stylised pines and round crowns) in loose groves behind the verge, seeded."""
        T = self.tree_cfg
        rng = np.random.default_rng(int(T["seed"]))
        n = int(self.length / 100.0 * float(T["density"]))
        if n <= 0:
            return
        pine_g = [G.cylinder(0.14, 0.10, 1.6, 6), G.cylinder(1.0, 0.0, 2.3, 8, caps=(True, False)),
                  G.cylinder(0.78, 0.0, 2.0, 8, caps=(True, False)), G.cylinder(0.52, 0.0, 1.7, 8, caps=(True, False))]
        pine_dz = [0.0, 0.9, 2.3, 3.6]
        round_trunk = G.cylinder(0.16, 0.12, 2.4, 6)
        crown = G.sphere(1.0, 8, 5)
        crown = G.Geom(crown.V * np.array([1.0, 1.0, 1.15]) + np.array([0.0, 0.0, 3.5]), crown.Q, crown.T, crown.UV)
        wood, leaf = G.Mesh(), G.Mesh()
        h_lo, h_hi = (float(v) for v in T["height"])
        for side, x_edge in ((+1.0, self.sec.x_max), (-1.0, self.sec.x_min)):
            s = rng.uniform(0.0, self.length, n)
            grove = 0.5 + 0.5 * np.sin(s * 0.011 + side * 1.9) * np.sin(s * 0.0063 + 0.7)
            s = s[rng.random(n) < 0.2 + 0.8 * grove ** 1.4]
            m = len(s)
            dist = float(T["near"]) + (float(T["far"]) - float(T["near"])) * rng.random(m) ** 1.6
            height = h_lo + (h_hi - h_lo) * rng.random(m) ** 1.3
            is_pine = rng.random(m) < float(T["pine"])
            yaw = rng.uniform(0.0, 2 * math.pi, m)
            ok = ~G.in_ranges(s, self.ranges, 60.0)
            for b in self.c["billboards"]:
                near_b = (np.abs(s - float(b["s"])) < 30.0) & (np.sign(-1.0 if b.get("side", "right") == "right" else 1.0) == side)
                ok &= ~near_b
            for g in self.c["gantries"]:
                ok &= np.abs(s - float(g["s"])) > 8.0
            s, dist, height, is_pine, yaw = s[ok], dist[ok], height[ok], is_pine[ok], yaw[ok]
            if not len(s):
                continue
            Tf, Lf, Uf = self.path.frame(s)
            base = self.path.point(s) + Lf * (x_edge + side * dist)[:, None]
            base[:, 2] = self.ground_z
            R_ = G.rot_z(yaw)
            sc = height / 6.0
            for pick, geoms, dzs in ((is_pine, pine_g, pine_dz), (~is_pine, [round_trunk, crown], [0.0, 0.0])):
                if not pick.any():
                    continue
                for k, (g, dz) in enumerate(zip(geoms, dzs)):
                    gg = G.Geom(g.V + np.array([0.0, 0.0, dz]), g.Q, g.T, g.UV)
                    inst = G.instance(gg, R_[pick], base[pick], scale=sc[pick])
                    (wood if k == 0 else leaf).add(inst, 0, smooth=True, spill=self.spill(inst.V))
        for mesh, key, mat in ((wood, "trunks", "trunk"), (leaf, "trees", "tree")):
            if len(mesh):
                self.obj(mesh, key, [self.mats[mat]])

    # ------------------------------------------------------------------------------------------ card
    def card(self):
        lanes = self.sec.lanes
        fwd = next((ln for ln in lanes if ln["dir"] == 1), lanes[0] if lanes else None)
        look = []
        if fwd is not None:
            for key, s in (("ahead", min(60.0, self.length)), ("far", self.length)):
                look.append({"name": key, "point": [float(v) for v in self.path.offset(s, fwd["offset"], 1.2)]})
        colliders = [{"type": "floor", "z": float(self.points[0, 2])}] if self.level else []
        return {"kind": "highway",
                "paths": {"road": {"points": self.points.tolist(), "width": float(self.sec.width), "lanes": lanes,
                                   "length": float(self.length), "tunnels": self.tunnels}},
                "use": {"look": look, "surface": self.surfaces}, "colliders": colliders, "lights": self.lights_out}


def _runs(idx):
    """Split sorted indices into runs of consecutive values."""
    if not len(idx):
        return []
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, brk)


@register("highway")
def build(name, coll, root, spec, palette):
    return Highway(name, coll, root, spec, palette).build()
