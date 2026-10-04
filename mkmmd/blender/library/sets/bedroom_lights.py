"""Lights, world, render settings and the outside (city and sky) of the bedroom_80s set. A helper module: it registers
no builder. Light strengths scale with the set's animatable parameters through drivers (`window_glow`, `neon`); the
city's brightness (`city_glow`) multiplies the emission of the skyline's materials (bedroom_mats.scale_emission)."""
import math

import bpy
from mathutils import Vector

from ..props.cafe_kit import drive
from .bedroom_mats import scale_emission

KEY_E = 1.1                      # strength (W/m2) of the window's slat-shadow light at window_glow = 1
KEY_AZ, KEY_EL = 22.0, 50.0      # it travels 22 degrees off the window's axis (towards -x) and 50 degrees downwards
KEY_ANGLE = 1.5                  # degrees: soft-edged slat and mullion shadows, no razor-edged patch on the floor
GLOW_E = 22.0                    # W of the soft area light in the opening
FILL_E = 7.0                     # W of the ambient fill under the ceiling
NEON_E = 9.0                     # W of the neon's light at neon = 1

CITY = {"shape": "arc", "az": 90.0, "arc": 100.0, "distance": 380.0, "depth": 240.0, "count": 150,
        "height": [14.0, 95.0], "footprint": [14.0, 40.0], "core": 0.55, "seed": 7, "lit": 0.45, "aviation": 9}
NEAR = {"shape": "arc", "az": 90.0, "arc": 110.0, "distance": 200.0, "depth": 120.0, "count": 70,
        "height": [8.0, 24.0], "footprint": [12.0, 30.0], "core": 0.1, "seed": 11, "lit": 0.22, "aviation": 0,
        "window": {"gold": 6.0, "rose": 1.0, "foam": 1.2, "text": 0.8, "strength": 0.75},
        "backdrop": False, "ground": False, "haze": {"distance": 380.0}}
GROUND_R = 240.0                 # radius (m) of the dark foreground plane under the near buildings
CITY_DROP = 10.0                 # the room is this far above the street (m); city.drop overrides it
GROUND_TONE = 0.3                # the dark land's tone (skyline.py: base .. hl_low), kept by the foreground plane
SKY = {"zenith": "base", "dome": "surface",
       "stars": {"density": 3.0, "fade": [1.5, 14.0]},
       "moon": False,
       "glow": {"az": 90.0, "width": 110.0, "height": 4.5, "strength": 0.8, "core": "rose", "rim": "iris"},
       "clouds": {"cover": 0.35, "opacity": 0.25, "wind": 100.0}}


def merge(base, over):
    """`over` laid over `base`, sub-tables merged key by key (anything that is not a table replaces)."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_light(R, name, kind, loc, direction=None, energy=1.0, color=(1.0, 1.0, 1.0, 1.0), size=(1.0, 1.0), radius=0.1,
               angle=0.0, shadow=True, specular=1.0):
    """Light object "<set>_<name>" in the lights collection. `direction` is where it shines (lights look along -Z)."""
    ld = bpy.data.lights.new(R.oname(name), kind)
    ld.color = tuple(color[:3])
    ld.energy = energy
    ld.use_shadow = shadow
    ld.specular_factor = specular
    if kind == "AREA":
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = size
    elif kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = radius
    elif kind == "SUN":
        ld.angle = math.radians(angle)
    o = R.obj(name, ld, "lights", loc=tuple(loc))
    if direction is not None:
        o.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return o


def key_direction():
    """Unit vector the window's key light travels along (into the room, downwards)."""
    az, el = math.radians(KEY_AZ), math.radians(KEY_EL)
    return Vector((-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), -math.sin(el)))


def build_lights(R, cfg, L, neon_line=None):
    """window_key (a sun: the slat shadows), window_glow (a soft area light in the opening), fill (ambient light under
    the ceiling), neon_light and neon_wash (along the neon tube: one shines down the wall, one up at the frieze and the
    ceiling). Returns the light objects."""
    w, T, H = cfg["window"], cfg["T"], cfg["H"]
    y1 = cfg["y1"]
    lights = []
    glow_k = cfg["params"]["window_glow"]
    key = make_light(R, "window_key", "SUN", (w["x"], y1 + 6.0, H + 3.0), key_direction(), KEY_E * glow_k, L["key"],
                     angle=KEY_ANGLE, specular=0.6)
    drive(key.data, "energy", f"{KEY_E} * glow", var=("glow", R.root, '["window_glow"]'))
    glow = make_light(R, "window_glow", "AREA", (w["x"], y1 + T + 0.12, (w["z0"] + w["z1"]) / 2), (0.0, -1.0, 0.0),
                      GLOW_E * glow_k, L["glow"], size=(w["width"], w["height"]), specular=0.4)
    drive(glow.data, "energy", f"{GLOW_E} * glow", var=("glow", R.root, '["window_glow"]'))
    fill = make_light(R, "fill", "AREA", (0.0, 0.0, H - 0.05), (0.0, 0.0, -1.0), FILL_E * cfg["fill"], L["fill"],
                      size=(cfg["x1"] - cfg["x0"] - 0.5, cfg["y1"] - cfg["y0"] - 0.5), specular=0.15)
    lights += [key, glow, fill]
    if neon_line is not None:
        a, b, n = neon_line
        mid = (a + b) / 2
        size = ((b - a).length, 0.06)
        for tag, d, k in (("neon_light", (n * 0.55 + Vector((0.0, 0.0, -0.83))).normalized(), 1.0),
                          ("neon_wash", (n * 0.5 + Vector((0.0, 0.0, 0.86))).normalized(), 0.8)):
            lt = make_light(R, tag, "AREA", tuple(mid + n * 0.03), d, NEON_E * k * cfg["params"]["neon"],
                            L["neon_light"], size=size, specular=0.3)
            drive(lt.data, "energy", f"{NEON_E * k} * neon", var=("neon", R.root, '["neon"]'))
            lights.append(lt)
    return lights


def build_world(R, L):
    """The plain dark world used when the sky is switched off: a dim blue-violet colour."""
    sc = bpy.context.scene
    w = bpy.data.worlds.get(R.oname("World")) or bpy.data.worlds.new(R.oname("World"))
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = L["world"]
    bg.inputs["Strength"].default_value = 0.12
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    sc.world = w
    return w


def setup_render(sc):
    """EEVEE Next with ray tracing and soft shadows, AgX Base Contrast (what the set's lighting is made for)."""
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    e = sc.eevee
    e.taa_render_samples = 48
    e.use_shadows = True
    e.shadow_ray_count = 2
    e.shadow_step_count = 6
    e.use_raytracing = True
    e.ray_tracing_method = "SCREEN"
    e.ray_tracing_options.resolution_scale = "1"
    e.ray_tracing_options.trace_max_roughness = 0.6
    e.ray_tracing_options.use_denoise = True
    e.fast_gi_method = "GLOBAL_ILLUMINATION"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = "sRGB"
    sc.render.film_transparent = False


def build_ground(R, coll, sub, palette, haze):
    """A dark disc of street level under the window (emission only, so no light of the room or the sun lands on it): the
    skyline's own land starts further out and is pale at its near edge, which would show as fog below the window."""
    import bmesh
    from . import nightkit as K
    name = R.oname("ground")
    K.clean(name)
    m, nb = K.new_material(name)
    shader = nb.emission(K.rgb(palette, f"base:hl_low:{GROUND_TONE}"), 1.0)
    nb.output(K.haze(nb, shader, K.rgb(palette, haze["colour"]), haze["distance"], haze["cap"]))
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=False, segments=72, radius=GROUND_R)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = sub
    o.visible_shadow = False
    return o


def build_outside(R, cfg, palette, params):
    """The city (skyline set), its near low-rise layer and the night sky (night_sky set) behind the window. The
    skylines hang from an empty `<set>_outside` that sits `drop` m below the floor, so the room is on an upper floor;
    the sky is in world angles and follows the set's yaw. Returns {"objects": [...], "lights": [light names],
    "city_point": [x, y, z] in the set frame}."""
    from . import nightkit as K
    from .night_sky import night_sky
    from .skyline import skyline
    out = {"objects": [], "lights": [], "city_point": None}
    coll = R.group("outside")
    root = R.root
    yaw = math.degrees(root.rotation_euler.z)
    if cfg["city"] is not None or cfg["near"] is not None:
        spec = merge(CITY, cfg["city"] or {})
        drop = float((cfg["city"] or {}).get("drop", CITY_DROP))
        spec.pop("drop", None)
        sub = bpy.data.objects.new(R.oname("outside"), None)
        coll.objects.link(sub)
        sub.parent = root
        sub.location = (0.0, 0.0, -drop)
        if cfg["city"] is not None:
            card = skyline(R.oname("skyline"), coll, sub, spec, palette)
            mid = next(p["point"] for p in card["use"]["look"] if p["name"] == "skyline")
            out["city_point"] = [mid[0], mid[1], mid[2] - drop]
        if cfg["near"] is not None:
            skyline(R.oname("near"), coll, sub, merge(NEAR, cfg["near"]), palette)
        build_ground(R, coll, sub, palette, {"colour": K.haze_hex(palette), "distance": spec["distance"], "cap": 0.55})
        mats = {m for ob in sub.children for m in ob.data.materials if m is not None}
        for m in mats:
            scale_emission(m, params)
        out["objects"] = list(sub.children)
    if cfg["sky"] is not None:
        spec = merge(SKY, cfg["sky"])
        for table in ("glow", "moon"):                  # azimuths are given in the set's frame; the sky is the world's
            if isinstance(spec.get(table), dict) and "az" in spec[table]:
                spec[table] = {**spec[table], "az": float(spec[table]["az"]) + yaw}
        card = night_sky(R.oname("sky"), coll, root, spec, palette)
        out["lights"] = list(card["lights"])
    return out
