"""cafe_room: a small corner cafe on a rainy street, built from the room outwards: plastered shell with a wainscot and
oak beams, one big mullioned window with rain-streaked, fogging glass, the wet pastel street behind it (row houses,
trees, sky), falling rain, lightning rigs, a pendant lamp, the room lights and the world. Pastel, matte, warm.

    [[set]]
    name = "cafe"
    kind = "cafe_room"
    at = [0, 0, 0]
    yaw = 0                          # the rest of the layout assumes this frame (the props of cafe.py, cafe_*.py)

Room frame (metres, Z up, floor z = 0). A character sits at the origin facing -Y; +X is her left. The window wall is the
plane x = -0.80 (outside is x < -0.80), the wall behind her is y = 1.05, the front wall y = -3.2, the left wall x = 1.6,
the ceiling z = 3.1. The big window spans y -1.75..0.65, z 0.82..2.85 (glass at x = -0.91, a 4 x 4 grid of panes). A
small round table is expected at (0, -0.48); the pendant lamp hangs over it.

spec keys (all optional)
    frame0     Blender frame of clip time 0 (default: the project's frame0); clip_t counts seconds from it
    duration   clip length in seconds (default: the project's; only used to place the default splats)
    ticks      clip seconds at which a rain splat lands on the glass; timeline = "audio/timeline.json" reads its
               `ticks` (or `tempo.ticks`), else its beats (what `mk timeline analyze` writes) instead; default: every 1.5 s
    seed       random seed of the street and the splat parameters (default 4242)
    pendant    [x, y] of the pendant lamp (default [0, -0.48])
    render     configure EEVEE Next (ray tracing, soft shadows) and AgX "Base Contrast" like the lighting expects
               (default true)
    rain, lightning   build the falling rain / the bolt rigs (default true)
    colors     {role = slot or "#hex"} overrides of the interior roles below (plaster, wainscot, window_paint, oak,
               oak_dark, cream, fog_tint, ...)
    steam, fog, outside_sat, sun, bolt, bolt_variant, bolt_rig, flash, refl    initial values of the parameters below

Animatable parameters: custom properties on the set root (key them with `root["fog"]`, e.g. in a build script or an
action). Every shader and light reads them through the node group "<name>_params" and drivers; nothing in the set is
keyed.
    steam        0..1   steam over the tea (default 0.6): the cafe_mug prop follows it, so the mug needs no key of its
                        own (cafe_mug's own `steam` property is then driven by this one)
    fog          0..1   condensation on the inside of the glass: the share of the glass that is fogged (default 0.12)
    outside_sat  0..1   saturation of everything outside the window (1 colour, 0 grey; default 1)
    sun          0..1   a low warm sun breaking through: gold sky glow and disc, gold sun lamp, warm outside (default 0)
    bolt         0..1   lightning bolt strength (0 = invisible, default 0)
    bolt_variant 0|1    which of the two bolt shapes (default 0)
    bolt_rig     0|1|2  which bolt rig is visible: 0 Front (seen from the front cameras), 1 Behind, 2 Side (default 0);
                        each rig is framed for one camera position and another rig could peek through the panes
    flash        0..1   lightning flash: cool-white light outside the window, sky and world brighten (default 0)
    refl         0..1   extra reflectivity of the window glass (her reflection in the pane; default 0)
    clip_t       s      seconds since clip start, driven by the scene frame (read-only; remove the driver to retime)

Colours. Every colour is a blend of palette slots (props/cafe_kit.py Colors.blend): with `[look] palette =
"rose-pine-dawn"` they reproduce the reference scene they were fitted on. The set is designed for a light palette: a dark palette
darkens the walls, wood and street the same way (lights and world do not follow). Light colours (lamps, sun, flash) are
`light(...)` tints: hue and saturation come from the palette, the strength does not.

Objects are named "<name>_<part>" and grouped in the child collections <name>_room, _window, _outside and _lights.
Needs the Rose Pine slots base surface overlay muted subtle text love gold rose pine foam iris hl_low hl_med hl_high.

Card: paths {}, use.look [window, street, pendant, sky], use.surface [the 16 panes, back_wall], colliders [the floor],
lights [the light objects], params {name: default}."""
import math
import random

import bmesh
import bpy

from ....core.palette import linear
from ...runtime import CTX
from ..props.cafe_kit import Colors, drive, mix
from . import cafe_glass as GL
from . import register
from .cafe_nodes import NG, PARAMS, Ctx, axis_mat, finish, hide_from_rays, params_group, principled
from .cafe_street import SPLAT_DRAWS, SUN_DIR, Outside

# ================================================================= geometry constants (metres, room frame)
WINDOW_X = -0.80            # window wall plane (her right); outside is x < WINDOW_X
BACK_WALL_Y = 1.05          # wall behind her
FRONT_WALL_Y = -3.2         # wall behind the front cameras (rarely seen)
WALL_T = 0.28               # wall thickness (window wall: x in [-1.08, -0.80])
ROOM_X0, ROOM_X1 = WINDOW_X, 1.60           # interior faces: window wall .. left wall
ROOM_Y0, ROOM_Y1 = FRONT_WALL_Y, BACK_WALL_Y
ROOM_H = 3.10
FLOOR_T = 0.20
WIN_Y0, WIN_Y1 = GL.OPEN_Y
WIN_Z0, WIN_Z1 = GL.OPEN_Z
FRAME_W = GL.FRAME_W                        # painted frame member width (visible face)
FRAME_X0, FRAME_X1 = -1.02, -0.785          # frame depth range (x)
GLASS_X = -0.91                             # the glass sheet
BAR_W, BAR_D = GL.BAR_W, 0.075              # mullion bar width / depth (x)
BARS_Y = GL.BARS_Y                          # vertical bar centres (col 0 nearest the front cameras)
BARS_Z = GL.BARS_Z                          # horizontal bar centres (row 0 is the bottom row)
SILL_X0, SILL_X1 = -1.00, WINDOW_X + 0.18
SILL_TOP = WIN_Z0                           # sill board top surface (plants stand here)
SILL_T = 0.04
TABLE_C = (0.0, -0.48)                      # the pendant hangs over the table centre

PROPS = {"steam": (0.6, 0.0, 1.0, "steam over the tea: read by the mug prop (cafe_mug follows it)"),
         "fog": (0.12, 0.0, 1.0, "share of the glass fogged by condensation"),
         "outside_sat": (1.0, 0.0, 1.0, "saturation of everything outside the window"),
         "sun": (0.0, 0.0, 1.0, "low warm sun breaking through"),
         "bolt": (0.0, 0.0, 1.0, "lightning bolt strength"),
         "bolt_variant": (0, 0, 1, "which of the two bolt shapes"),
         "bolt_rig": (0, 0, 2, "which bolt rig is visible (0 front, 1 behind, 2 side)"),
         "flash": (0.0, 0.0, 1.0, "lightning flash outside the window"),
         "refl": (0.0, 0.0, 1.0, "extra reflectivity of the window glass")}

WORLD_STRENGTH = 0.30
KEY_E = 170.0
FILL_E = 70.0
TOP_E = 34.0
PENDANT_E = 45.0
FLASH_E = 4000.0
SUN_E = 14.0


# ================================================================= colours
def interior_colors(C, spec):
    """Interior roles as blends of palette slots (fitted on Rose Pine Dawn); `spec["colors"]` overrides single roles."""
    b, L = C.blend, C.light
    plaster = b(surface=.6, gold=.4)                                    # warm cream
    oak = b(gold=.7, hl_high=.3, hue=10, chroma=.75, k=1.1)                # light oak
    roles = {
        "plaster": plaster,
        "ceiling": mix(plaster, b(surface=1), 0.5),
        "wainscot": mix(b(foam=1), b(base=1), 0.72),
        "window_paint": b(pine=1),
        "cream": b(surface=1),
        "oak": oak,
        "oak_dark": b(gold=.7, subtle=.3, hue=10, chroma=.75, k=1.1),         # darker oak
        "oak_light": b(gold=.5, hl_med=.5, hue=20, k=1.2),                  # lightest oak
        "fog_tint": mix(b(surface=1), b(overlay=.7, love=.3, hue=-90, k=1.2), 0.45),     # condensation: cream, a little lavender
        "ext_wall": mix(b(rose=1), b(base=1), 0.55),
        "bulb": b(surface=1),
        "shade_paper": mix(b(surface=1), b(gold=1), 0.18),
        "shade_glow": mix(b(surface=1), b(gold=1), 0.35),
        "shade_inner": mix(b(gold=1), b(surface=1), 0.5),
        # light tints (fitted: hue and saturation only, the strength is set by the lamp)
        "key": L(muted=.65, hl_med=.2, foam=.15), "fill_warm": L(overlay=.7, gold=.3, hue=10, chroma=.7),
        "fill_front": L(surface=.45, hl_low=.45, gold=.1), "fill_top": L(surface=.8, gold=.1, pine=.1),
        "pendant": L(gold=.4, hl_low=.6, hue=20), "flash": L(pine=.45, hl_low=.45, iris=.1),
        "sun": L(surface=.7, gold=.3, hue=20, chroma=2.0),
        # world
        "world_up_warm": L(hl_med=.55, gold=.25, surface=.2), "world_up_cool": L(subtle=.5, hl_med=.4, iris=.1),
        "world_down": L(gold=.45, foam=.45, hl_high=.1, k=0.863), "world_flash": L(muted=.6, surface=.2, foam=.2),
        "warm_glow": mix(mix(b(gold=1), b(rose=1), 0.35), b(surface=1), 0.30),
    }
    for k, v in (spec.get("colors") or {}).items():
        if k not in roles:
            raise KeyError(f"cafe_room colors: no role {k!r} (have {sorted(roles)})")
        roles[k] = (*linear(v), 1.0) if v.startswith("#") else C.slot(v)
    return roles


# ================================================================= interior materials
def mat_plaster(R, name, color, rough=0.92, bump=0.10, scale=7.0):
    m, g = R.mat(name)
    co = g.texco("Object")
    mott = g.noise(co, scale, detail=6.0, rough=0.62)
    grain = g.noise(co, 160.0, detail=1.0, rough=0.5)
    col = g.mixc(color, tuple(min(1.0, c * 1.08) for c in color), mott)
    h = g.mad(grain, 0.6, g.mul(mott, 0.4))
    b = principled(g, col, rough=rough, spec=0.3, normal=g.bump(h, bump, 0.003))
    finish(g, b.outputs[0])
    return m


def mat_paint(R, name, color, rough=0.4, bump=0.012):
    m, g = R.mat(name)
    brush = g.noise(g.texco("Object"), 24.0, detail=2.0, rough=0.5)
    b = principled(g, color, rough=rough, spec=0.5, normal=g.bump(brush, bump, 0.001))
    finish(g, b.outputs[0])
    return m


def mat_wood(R, name, color, color2, stretch=(1.0, 30.0, 30.0), rough=0.5):
    m, g = R.mat(name)
    co = g.mapping(g.texco("Object"), scale=stretch)
    grain = g.noise(co, 1.6, detail=5.0, rough=0.55, distortion=0.6)
    col = g.mixc(color, color2, g.ss(grain, 0.3, 0.7))
    b = principled(g, col, rough=rough, spec=0.45, normal=g.bump(grain, 0.12, 0.002))
    finish(g, b.outputs[0])
    return m


def mat_floor(R, L):
    """Light oak boards running along y (towards the front cameras): Brick texture planks + grain + grooves."""
    m, g = R.mat("FloorOak")
    co = g.texco("Object")
    p = g.mapping(co, rot=(0, 0, math.pi / 2))
    br = g.node("ShaderNodeTexBrick", offset=0.43, offset_frequency=1, squash=1.0, squash_frequency=2)
    g.put(br.inputs["Vector"], p)
    g.put(br.inputs["Color1"], L["oak"])
    g.put(br.inputs["Color2"], mix(L["oak"], L["oak_light"], 0.7))
    g.put(br.inputs["Mortar"], L["oak_dark"])
    g.put(br.inputs["Scale"], 1.0)
    g.put(br.inputs["Mortar Size"], 0.004)
    g.put(br.inputs["Mortar Smooth"], 0.3)
    g.put(br.inputs["Bias"], 0.0)
    g.put(br.inputs["Brick Width"], 1.5)
    g.put(br.inputs["Row Height"], 0.16)
    grain = g.noise(g.mapping(p, scale=(0.5, 38.0, 1.0)), 1.4, detail=6.0, rough=0.6, distortion=0.5)
    col = g.mixc(br.outputs["Color"], g.mixc(br.outputs["Color"], L["oak_dark"], 0.5), g.ss(grain, 0.35, 0.75))
    b = principled(g, col, rough=0.42, spec=0.5,
                   normal=g.bump(g.mad(br.outputs["Fac"], 1.0, g.mul(grain, 0.25)), 0.35, 0.003))
    finish(g, b.outputs[0])
    return m


def mat_shade(R, L):
    """Pendant lamp shade: warm paper, glowing from the inside. Outside = lit cream paper + glow, inside = gold."""
    m, g = R.mat("PendantShade")
    back = g.node("ShaderNodeNewGeometry").outputs["Backfacing"]
    outer = principled(g, L["shade_paper"], rough=0.8, spec=0.2, emission_color=L["shade_glow"],
                       emission_strength=1.4, transmission_weight=0.25)
    inner = g.node("ShaderNodeEmission")
    g.put(inner.inputs["Color"], L["shade_inner"])
    g.put(inner.inputs["Strength"], 4.0)
    mixs = g.node("ShaderNodeMixShader")
    g.put(mixs.inputs[0], back)
    g.nt.links.new(outer.outputs[0], mixs.inputs[1])
    g.nt.links.new(inner.outputs[0], mixs.inputs[2])
    finish(g, mixs.outputs[0])
    return m


def mat_emit(R, base, color, strength):
    m, g = R.mat(base)
    em = g.node("ShaderNodeEmission")
    g.put(em.inputs["Color"], color)
    g.put(em.inputs["Strength"], strength)
    finish(g, em.outputs[0])
    return m


# ================================================================= the room shell
def build_shell(R, mats):
    """Walls, floor, ceiling, skirting, wainscot, beams. Interior faces sit exactly on the planes of the layout."""
    X0, X1, Y0, Y1, H = ROOM_X0, ROOM_X1, ROOM_Y0, ROOM_Y1, ROOM_H
    T = WALL_T
    OX0, OX1, OY0, OY1 = X0 - T, X1 + T, Y0 - T, Y1 + T
    plaster, ext, ceil_m = mats["plaster"], mats["ext"], mats["ceiling"]
    out = {}

    def walls(base, specs, slot_map):
        return R.boxes_obj(base, specs, "room", [plaster, ext, ceil_m, mats["wainscot"]],
                           face_slot=lambda n: axis_mat(n, slot_map))

    # window wall: four slabs around the opening
    zb, zt = -FLOOR_T, H + 0.2
    ww = [(OX0, X0, OY0, OY1, zb, WIN_Z0),
          (OX0, X0, OY0, OY1, WIN_Z1, zt),
          (OX0, X0, OY0, WIN_Y0, WIN_Z0, WIN_Z1),
          (OX0, X0, WIN_Y1, OY1, WIN_Z0, WIN_Z1)]
    out["WallWindow"] = walls("WallWindow", ww, {"+x": 0, "-x": 1, "any": 0})
    out["WallBack"] = walls("WallBack", [(OX0, OX1, Y1, OY1, zb, zt)], {"-y": 0, "any": 1})
    out["WallFront"] = walls("WallFront", [(OX0, OX1, OY0, Y0, zb, zt)], {"+y": 0, "any": 1})
    out["WallLeft"] = walls("WallLeft", [(X1, OX1, OY0, OY1, zb, zt)], {"-x": 0, "any": 1})
    out["Floor"] = R.boxes_obj("Floor", [(X0, X1, Y0, Y1, -FLOOR_T, 0.0)], "room", [mats["floor"], ext],
                               face_slot=lambda n: axis_mat(n, {"+z": 0, "any": 1}))
    out["Ceiling"] = R.boxes_obj("Ceiling", [(OX0, OX1, OY0, OY1, H, H + 0.2)], "room", [ceil_m, ext],
                                 face_slot=lambda n: axis_mat(n, {"-z": 0, "any": 1}))

    # skirting + picture rail + wainscot panel (lower wall), around the four walls (window wall below the sill)
    sk_h, sk_t = 0.11, 0.02
    wz = 1.0                                            # wainscot height
    pan_t = 0.012
    trims, panels = [], []
    for (x0, x1, y0, y1) in [(X0, X0 + sk_t, Y0, Y1), (X1 - sk_t, X1, Y0, Y1),
                             (X0, X1, Y1 - sk_t, Y1), (X0, X1, Y0, Y0 + sk_t)]:
        trims.append((x0, x1, y0, y1, 0.0, sk_h))
    rail_z = wz
    trims += [(X1 - 0.03, X1, Y0, Y1, rail_z, rail_z + 0.035), (X0, X1, Y1 - 0.03, Y1, rail_z, rail_z + 0.035),
              (X0, X1, Y0, Y0 + 0.03, rail_z, rail_z + 0.035)]
    panels += [(X1 - pan_t, X1, Y0, Y1, sk_h, rail_z), (X0, X1, Y1 - pan_t, Y1, sk_h, rail_z),
               (X0, X1, Y0, Y0 + pan_t, sk_h, rail_z)]
    out["Skirting"] = R.boxes_obj("Skirting", trims, "room", [mats["trim"]], bevel=0.004)
    out["Wainscot"] = R.boxes_obj("Wainscot", panels, "room", [mats["wainscot"]])

    # ceiling beams (light oak) across the room along x
    beams = [(X0, X1, y - 0.07, y + 0.07, H - 0.16, H) for y in (-2.45, -0.95)]
    out["Beams"] = R.boxes_obj("Beams", beams, "room", [mats["oak_beam"]], bevel=0.006)
    return out


def window_panes():
    """Visible glass rectangles between frame and bars: list of dicts (row 0 = bottom, col 0 = nearest the front)."""
    out = []
    for p in GL.pane_rects():
        out.append({"row": p["row"], "col": p["col"], "y0": p["y0"], "y1": p["y1"], "z0": p["z0"], "z1": p["z1"]})
    return out


def build_window(R, mats):
    """WindowFrame (ring + casing + mullion grid, painted), WindowSill (oak), WindowSurround (outside trim)."""
    y0, y1, z0, z1 = WIN_Y0, WIN_Y1, WIN_Z0, WIN_Z1
    fx0, fx1 = FRAME_X0, FRAME_X1
    fw = FRAME_W
    specs = [(fx0, fx1, y0, y0 + fw, z0, z1), (fx0, fx1, y1 - fw, y1, z0, z1),
             (fx0, fx1, y0 + fw, y1 - fw, z0, z0 + fw), (fx0, fx1, y0 + fw, y1 - fw, z1 - fw, z1)]
    bx0, bx1 = GLASS_X - BAR_D / 2, GLASS_X + BAR_D / 2
    specs += [(bx0, bx1, yc - BAR_W / 2, yc + BAR_W / 2, z0 + fw, z1 - fw) for yc in BARS_Y]
    specs += [(bx0, bx1, y0 + fw, y1 - fw, zc - BAR_W / 2, zc + BAR_W / 2) for zc in BARS_Z]
    # interior casing boards on the wall face
    cw, ct = 0.075, 0.014
    specs += [(WINDOW_X - ct, WINDOW_X + 0.0, y0 - cw, y0, z0 - 0.02, z1 + cw),
              (WINDOW_X - ct, WINDOW_X + 0.0, y1, y1 + cw, z0 - 0.02, z1 + cw),
              (WINDOW_X - ct, WINDOW_X + 0.0, y0, y1, z1, z1 + cw)]
    frame = R.boxes_obj("WindowFrame", specs, "window", [mats["pine"]], bevel=0.004)
    # sill board: through the wall, 0.18 m into the room, round front edge
    sill = R.boxes_obj("WindowSill", [(SILL_X0, SILL_X1, y0 - 0.07, y1 + 0.07, SILL_TOP - SILL_T, SILL_TOP)], "window",
                       [mats["oak"]], bevel=0.009)
    # outside trim + ledge
    ox = WINDOW_X - WALL_T
    sur = [(ox - 0.03, ox, y0 - 0.11, y0, z0 - 0.07, z1 + 0.11), (ox - 0.03, ox, y1, y1 + 0.11, z0 - 0.07, z1 + 0.11),
           (ox - 0.03, ox, y0, y1, z1, z1 + 0.11), (ox - 0.12, ox, y0 - 0.13, y1 + 0.13, z0 - 0.09, z0 - 0.02)]
    surround = R.boxes_obj("WindowSurround", sur, "window", [mats["cream_paint"]], bevel=0.005)
    return frame, sill, surround


def build_glass(R, mat):
    """WindowGlass: one quad over the whole opening facing +X (into the room), UV = (u, v) of cafe_glass."""
    verts = [(GLASS_X, WIN_Y0, WIN_Z0), (GLASS_X, WIN_Y1, WIN_Z0), (GLASS_X, WIN_Y1, WIN_Z1), (GLASS_X, WIN_Y0, WIN_Z1)]
    o = R.mesh_obj("WindowGlass", verts, [(0, 1, 2, 3)], "window", [mat], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    hide_from_rays(o, shadow=False, glossy=True, diffuse=True, transmission=True)
    return o


def build_pendant(R, L, mats):
    """Ceiling rose, cord and a glowing paper shade over the table."""
    tx, ty = R.pendant
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=40, radius1=0.21, radius2=0.06, depth=0.2)
    for v in bm.verts:
        v.co.z += 2.22
        v.co.x += tx
        v.co.y += ty
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(R.oname("PendantShade"))
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mats["shade"])
    shade = R.obj("PendantShade", me, "room")
    hide_from_rays(shade, shadow=False)

    def cyl(base, r, depth, loc, mat, verts):
        bmc = bmesh.new()
        bmesh.ops.create_cone(bmc, cap_ends=True, segments=verts, radius1=r, radius2=r, depth=depth)
        mec = bpy.data.meshes.new(R.oname(base))
        bmc.to_mesh(mec)
        bmc.free()
        for p in mec.polygons:
            p.use_smooth = True
        mec.materials.append(mat)
        return R.obj(base, mec, "room", loc=loc)

    cyl("PendantCord", 0.004, ROOM_H - 2.32, (tx, ty, (ROOM_H + 2.32) / 2), mats["cream_paint"], 8)
    cyl("PendantRose", 0.06, 0.03, (tx, ty, ROOM_H - 0.015), mats["cream_paint"], 24)
    bulb = cyl("PendantBulb", 0.035, 0.07, (tx, ty, 2.17), mats["bulb"], 16)
    hide_from_rays(bulb, shadow=False, glossy=False)


# ================================================================= lights, world
def make_light(R, name, kind, loc, rot=(0, 0, 0), energy=100.0, color=(1, 1, 1), size=1.0, size_y=None,
               spot=None, radius=None, angle=None, shadow=True):
    ld = bpy.data.lights.new(R.oname(name), kind)
    ld.color = color[:3]
    ld.energy = energy
    ld.use_shadow = shadow
    if kind == "AREA":
        ld.shape = "RECTANGLE"
        ld.size = size
        ld.size_y = size_y if size_y is not None else size
    elif kind == "SPOT":
        ld.spot_size = spot or math.radians(90)
        ld.spot_blend = 0.6
        ld.shadow_soft_size = radius or 0.1
    elif kind == "POINT":
        ld.shadow_soft_size = radius or 0.1
    elif kind == "SUN":
        ld.angle = angle if angle is not None else math.radians(2.0)
    return R.obj(name, ld, "lights", loc=loc, rot=rot)


def build_lights(R, L):
    tx, ty = R.pendant
    out = []
    key = make_light(R, "Key_Window", "AREA", (WINDOW_X - WALL_T - 0.05, -0.55, 1.85), rot=(0, -math.pi / 2, 0),
                     energy=KEY_E, color=L["key"], size=2.55, size_y=2.15)
    key.data.transmission_factor = 0.0          # a light behind the pane must not show up through the glass
    fill = make_light(R, "Fill_Warm", "AREA", (1.5, -0.7, 1.55), rot=(0, math.pi / 2, 0), energy=FILL_E,
                      color=L["fill_warm"], size=2.6, size_y=2.0)
    front = make_light(R, "Fill_Front", "AREA", (0.6, -3.1, 1.9), rot=(math.pi / 2, 0, 0), energy=FILL_E * 0.8,
                       color=L["fill_front"], size=2.4, size_y=1.8)
    top = make_light(R, "Fill_Ceiling", "AREA", (0.3, -0.8, ROOM_H - 0.12), rot=(0, 0, 0), energy=TOP_E,
                     color=L["fill_top"], size=2.2, size_y=3.4)
    pend = make_light(R, "Pendant_Light", "POINT", (tx, ty, 2.14), energy=PENDANT_E, color=L["pendant"], radius=0.22)
    pend.data.specular_factor = 0.22            # keeps the lamp from showing up as a moon in the window glass
    for ob in (fill, front, top, pend):
        ob.data.transmission_factor = 0.0      # room lights must not show up as bright shapes through the pane
        if ob is not pend:
            ob.data.specular_factor = 0.35
    flash = make_light(R, "Flash_Area", "AREA", (WINDOW_X - 5.5, -0.55, 2.2), rot=(0, -math.pi / 2, 0),
                       energy=0.0, color=L["flash"], size=14.0, size_y=8.0)
    drive(flash.data, "energy", f"{FLASH_E} * flash", var=("flash", R.root, '["flash"]'))
    flash.data.transmission_factor = 0.0
    sun = make_light(R, "Sun_Gold", "SUN", (WINDOW_X - 20, 0, 6), energy=0.0, color=L["sun"],
                     angle=math.radians(2.5))
    # shines towards +x, slightly towards -y, 14 degrees downward
    sun.rotation_euler = (-SUN_DIR).to_track_quat("-Z", "Y").to_euler()
    drive(sun.data, "energy", f"{SUN_E} * sun", var=("sun", R.root, '["sun"]'))
    return [o.name for o in (key, fill, front, top, pend, flash, sun)]


def build_world(R, L):
    sc = bpy.context.scene
    w = bpy.data.worlds.get(R.oname("World")) or bpy.data.worlds.new(R.oname("World"))
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    g = NG(nt, R.params)
    d = g.vm("NORMALIZE", g.texco("Generated"))
    _, _, dz = g.sep(d)
    up = g.mixc(L["world_up_warm"], L["world_up_cool"], g.ss(dz, 0.0, 0.8))      # warm-neutral ambient, lavender above
    down = L["world_down"]                                   # warm floor bounce below the horizon
    col = g.mixc(down, up, g.ss(dz, -0.25, 0.1))
    sun = g.param("sun")
    flash = g.param("flash")
    col = g.mixc(col, L["warm_glow"], g.mul(sun, 0.35))
    col = g.mixc(col, L["world_flash"], g.mul(flash, 0.6))
    strength = g.mul(1.0, g.mad(flash, 3.0, 1.0))
    bg = g.node("ShaderNodeBackground")
    g.put(bg.inputs["Color"], col)
    g.put(bg.inputs["Strength"], g.mul(strength, WORLD_STRENGTH))
    out = g.node("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    sc.world = w
    return w


def setup_render(sc):
    """EEVEE Next with screen-space ray tracing (glass refraction), soft shadows, AgX Base Contrast."""
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


# ================================================================= the builder
@register("cafe_room")
def cafe_room(name, coll, root, spec, palette):
    sc = bpy.context.scene
    proj = CTX.get("project") or {}
    fps = sc.render.fps / sc.render.fps_base
    frame0 = int(spec.get("frame0", proj.get("frame0", sc.frame_start)))
    duration = float(spec.get("duration", proj.get("duration", 24.6)))
    C = Colors(palette)
    L = interior_colors(C, spec)

    # ---- parameters: custom properties on the root, read through a node group and drivers
    root["mk_set"] = "cafe_room"                     # marks the root: props follow the scene-wide controls (steam)
    for k, (default, lo, hi, doc) in PROPS.items():
        root[k] = type(default)(spec.get(k, default))
        root.id_properties_ui(k).update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=doc, default=default)
    root["clip_t"] = 0.0
    root.id_properties_ui("clip_t").update(description="seconds since clip start (driven by the frame)")
    drive(root, '["clip_t"]', f"(frame - {frame0}) / {fps:g}")
    params = params_group(name, root)
    R = Ctx(name, coll, root, C, params, spec, frame0, fps)
    R.pendant = tuple(spec.get("pendant", TABLE_C))
    rng_glass = random.Random(int(spec.get("seed", 4242)))
    rng_street = random.Random(int(spec.get("seed", 4242)))
    for _ in range(SPLAT_DRAWS):
        rng_street.uniform(0.0, 1.0)                 # the street continues the stream after the splat parameters
    if spec.get("render", True):
        setup_render(sc)

    # ---- interior
    mats = {
        "plaster": mat_plaster(R, "Plaster", L["plaster"]),
        "ceiling": mat_plaster(R, "Ceiling", L["ceiling"], scale=5.0),
        "wainscot": mat_plaster(R, "Wainscot", L["wainscot"], rough=0.6, bump=0.04),
        "ext": None,
        "floor": mat_floor(R, L),
        "trim": mat_paint(R, "Trim", L["cream"], 0.42),
        "cream_paint": mat_paint(R, "CreamPaint", L["cream"], 0.42),
        "pine": mat_paint(R, "WindowPaint", L["window_paint"], 0.36),
        "oak": mat_wood(R, "SillOak", L["oak"], L["oak_dark"], stretch=(30.0, 1.0, 30.0)),
        "oak_beam": mat_wood(R, "BeamOak", L["oak"], L["oak_dark"], stretch=(1.0, 30.0, 30.0)),
        "shade": mat_shade(R, L),
        "bulb": mat_emit(R, "Bulb", L["bulb"], 6.0),
    }
    out = Outside(R)
    out.c["facade_rose"] = L["ext_wall"]
    mats["ext"] = out.mat_out("ExtWall", lambda g: L["ext_wall"], strength=1.0, mist=None)
    build_shell(R, mats)
    build_window(R, mats)
    build_pendant(R, L, mats)

    # ---- the glass, and everything behind it
    ticks = GL.splat_ticks(spec, proj.get("root") or ".", duration)
    splats = GL.splat_drops(ticks)
    fog_img = GL.fog_image("mk_cafe_fog_noise")            # one packed copy, shared by every cafe_room in the file
    build_glass(R, GL.mat_glass(R.oname("WindowGlass"), params, splats, rng_glass, L["fog_tint"], fog_img))
    out.build_street(rng_street)
    if spec.get("rain", True):
        out.build_rain()
    if spec.get("lightning", True):
        out.build_lightning()

    # ---- lights, world, the planar probe that shows a reflection in the pane
    lights = build_lights(R, L)
    build_world(R, L)
    pd = bpy.data.lightprobes.new(R.oname("GlassMirror"), "PLANE")
    R.obj("GlassMirror", pd, "window", loc=(GLASS_X + 0.004, -0.55, 1.835), rot=(0, math.pi / 2, 0),
          scale=(1.03, 1.22, 1.0))
    bpy.context.view_layer.update()

    panes = [{"name": f"pane_r{p['row']}c{p['col']}",
              "center": [GLASS_X, (p["y0"] + p["y1"]) / 2, (p["z0"] + p["z1"]) / 2], "normal": [1.0, 0.0, 0.0],
              "up": [0.0, 0.0, 1.0], "size": [round(p["y1"] - p["y0"], 4), round(p["z1"] - p["z0"], 4)]}
             for p in window_panes()]
    back = {"name": "back_wall", "center": [-0.25, BACK_WALL_Y, 1.6], "normal": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0],
            "size": [2.4, 1.2]}
    return {
        "kind": "cafe_room",
        "paths": {},
        "use": {"look": [{"name": "window", "point": [GLASS_X, -0.55, 1.835]},
                         {"name": "street", "point": [-12.0, -0.55, 1.8]},
                         {"name": "pendant", "point": [R.pendant[0], R.pendant[1], 2.17]},
                         {"name": "sky", "point": [-40.0, -0.55, 25.0]}],
                "surface": panes + [back]},
        "colliders": [{"type": "floor", "z": 0.0, "tag": name}],
        "lights": lights,
        "params": {k: root[k] for k in list(PARAMS) + ["steam"] if k in root.keys()},
    }
