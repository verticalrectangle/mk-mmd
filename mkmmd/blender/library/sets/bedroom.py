"""bedroom_80s: an 80s bedroom at night, built from the room outwards: striped wallpaper over a dark herringbone
parquet, a picture rail with a pink neon tube, a teal six-panel door, a double-hung window with a venetian blind half
raised, and behind the glass a lit city under a dark sky. A shell only: furniture, lamp, posters and rug are props the
project places. Dusty violet and blue by night; the 80s is in the pastel trim, the teal door, the chrome and the neon.

    [[set]]
    name = "room"
    kind = "bedroom_80s"
    at = [0, 0, 0]                   # the props' placement rules assume this frame
    yaw = 0

Frame (metres, Z up, floor z = 0, root at the middle of the room). The interior is x -1.8 .. 1.8, y -1.6 .. 1.6,
z 0 .. 2.55 (`size`). The window wall is the plane y = +1.6 ("back"; outside is +Y), the door wall x = +1.8 ("right"),
the left wall x = -1.8 (the bed's), the front wall y = -1.6 (behind the cameras). Walls are 0.2 m thick (`wall`); the
faces on the card sit exactly on the interior planes. Trim stands off the walls: skirting 0.022 m (0.134 m high),
picture rail 0.028 m (its top 0.2 m under the ceiling: z 2.285 .. 2.35), casings 0.018 m, sockets 0.009 m, so props
against a wall should keep about 0.025 m from its plane.

The window. Opening x -0.25 .. 1.05 (centre 0.4, 1.3 m wide), z 0.95 .. 2.15 (1.2 m high) through the 0.2 m wall, so
the reveal is deep. The painted frame sits 0.09 .. 0.15 m behind the wall face and the glass in its middle (y = 1.72);
a meeting rail and a muntin make 2 x 2 panes; casing boards 0.075 m wide, an apron, and a sill board whose top is at
exactly z = 0.95 and which reaches 0.06 m into the room. The venetian blind hangs in front of the casing (outside
mount): a bullnose headrail above the opening (its bottom at z = 2.22), about 56 aluminium slats of 25 mm with ladder
strings and a rounded bottom rail, 1.4 m wide, the tilt wand on the left, the lift cord and its tassel on the right.
Raised half way, the lower part of the window is open and the upper part shows tilted slats. The glass is a clear
night pane with a faint reflection (a planar probe: what it mirrors is the room, not the sky).
The door: x = 1.8, y -1.45 .. -0.55, z 0 .. 2.05, a closed six-panel leaf sunk 0.04 m in a pocket of the wall, with
casing, a lever handle (rose and escutcheon) on the y = -0.55 side and three hinges on the other.

spec keys (all optional)
    size       [3.6, 3.2, 2.55] interior width (x), depth (y) and height (z). The window and the door keep their
               absolute positions, so a smaller room may need `window` / `door` too (the spec checks that they fit)
    wall       0.2  wall thickness (0.16 .. 0.5)
    open       []   walls and parts to leave out: any of "front", "back", "left", "right", "ceiling", "floor". An open
               wall takes its skirting, rail and fittings with it. (`mk look --hide <name>_wall_front --hide
               <name>_ceiling` hides them for one view instead.)
    window     {x = 0.4, width = 1.3, height = 1.2, sill = 0.95, cols = 2, rows = 2}  centre x, size and sill height
               of the opening (its top may not rise above H - 0.38); cols x rows panes (rows are sashes: meeting
               rails; cols: muntins)
    door       {y = -1.0, width = 0.9, height = 2.05}
    floor      "herringbone" (0.07 x 0.28 m planks at 45 degrees) or "boards" (0.11 m boards, staggered joints)
    pendant    true  the ceiling rose with a short flex and a bare bulb (off: a stub, not a lamp)
    neon       true  the neon tube on the picture rail. false builds neither the tube nor its lights (the `neon`
               property then stays 0); a number 0 .. 1 is the initial `neon` level (true: 0.7)
    neon_wall  "left"  whose picture rail carries the tube: "back", "left", "right" or "front"
    fill       1.0   strength of the ambient fill light under the ceiling, relative
    seed       80    random tones of the planks
    render     true  configure EEVEE Next (ray tracing, soft shadows, 48 samples) and AgX "Base Contrast"
    city       true  the skyline. false builds no city at all (and no near layer); a table overrides the defaults of
               the `skyline` set (every key of its docstring) and may add `drop` (default 10): how far below the floor
               the street is, in metres. Defaults: shape "arc", az 90 (+Y), arc 100, distance 380, depth 240,
               count 150, height [14, 95], footprint [14, 40], core 0.55, seed 7, lit 0.45, aviation 9 (red beacons)
    near       follows `city`  a second, near layer of low buildings, so the lower half of the window shows rooftops
               and lit windows instead of fog. false removes it; a table overrides its `skyline` keys (distance 200,
               height [8, 24], dimmer windows, no beacons, no glow sheet)
    sky        true  the night sky (the `night_sky` set, which replaces the scene's world). false leaves a plain dim
               blue world; a table overrides its keys (sub-tables merge key by key). Defaults: zenith base, dome
               surface, stars of density 3 (fading in between 1.5 and 14 degrees), no moon, a faint violet-rose city
               glow toward the window (az 90 in the set's frame: the set's yaw is added, the sky itself is in world
               angles) and thin clouds
    colors     {role = slot or "#hex"}  overrides of single colour roles (bedroom_colors.roles): paper, paper_light,
               paper_pin, paper_motif, frieze, ceiling, plaster, outside, trim, trim_dark, door, door_dark, floor_a,
               floor_b, floor_groove, glass, slat, string, cord, metal, plate, rocker, insert, bulb, neon,
               neon_tube, and the light tints key, glow, fill, neon_light, world
    blinds, slat_angle, window_glow, city_glow, neon     initial values of the parameters below

Animatable parameters: custom properties on the set root (key them with `[[key]] target = "room" prop = "blinds"`, or in
an action). Every shader, light and modifier reads them through drivers; nothing in the set is keyed.
    blinds       0..1     0 = the blind fully lowered over the window, 1 = fully raised (the slats piled under the
                          headrail); the lift cord is pulled out as it rises (default 0.5)
    slat_angle   -85..85  tilt of the slats in degrees: 0 flat; positive lifts the outer (street-side) edge, so the
                          view looks down through the slats and light from above gets in; negative the other way
                          (default 20). The key light shines 50 degrees down, so the slat shadows are strongest with
                          the slats near flat; tilted parallel to the light (about 50) they let all of it through
    window_glow  0..3     strength of the light that comes in through the window: the key and the soft area light
                          (default 1)
    city_glow    0..3     brightness of the city's lit windows, its glow and its beacons; the silhouettes keep their
                          tone (default 1)
    neon         0..1     the neon tube's emission and its two lights (default 0.7; 0 = off)

Shapes. Nothing visible has a razor edge: every arris of a board, a reveal or a plate is bevelled (3 mm on the walls,
casings and frames, 1 - 3.5 mm on the small parts, 3 or more segments), the skirting, the picture rail, the sill board
and the headrail are swept from rounded sections (mitred at the corners), the parquet planks have a 0.8 mm chamfer, the
blind's bottom rail, strings, cord and tassel are round. The blockiness audit finds no hard edge (75 degrees or more) in
any object of the set. `mk_form_exempt` is set on the architecture, which is straight by nature (the four walls, floor,
ceiling, skirting, picture rail, door casing, window frame and trim, the roof-shadow plane) and on everything outside
(the far city is boxes at 150 .. 500 m); the door leaf, blind, headrail, fixtures, hardware, neon, clips and ceiling
rose pass the `form` analysis on their own (scores 0 .. 0.03).

Objects, all named "<name>_<part>", in the child collections <name>_room, _window, _outside and _lights:
    wall_back, wall_left, wall_right, wall_front   one closed mesh each (slots: wallpaper, reveal paint, outside,
        door pocket); wall_back has the window opening, wall_right the door pocket, lined in the trim paint. The back
        and front slabs reach into the side walls, so a corner is two interior planes crossing
    floor, ceiling, roof_shadow (an invisible plane at the ceiling that only the lights see: hiding `<name>_ceiling`
    for a plan view does not let the sun and the sky in over the walls), skirting, picture_rail, door_casing,
    door_leaf, door_hardware, fixtures, fixture_inserts (a light switch and a socket by the door, sockets on the window
    wall), neon, neon_clips (four ring clips on tabs), ceiling_rose, ceiling_flex, ceiling_bulb
    window_frame, window_trim (casing, sill board, apron), window_glass, window_mirror (the planar probe),
    window_blind_head (the bullnose headrail, two brackets, the tilt wand), window_blinds (a geometry-nodes object:
    slats, bottom rail, ladder strings, lift cord and tassel, driven by `blinds` and `slat_angle`),
    window_blind_slat (hidden: the one slat the node tree instances)
    outside (an empty `drop` m below the floor that the skylines hang from), skyline_city, skyline_glow,
    skyline_land, skyline_beacons, skyline_halos (the city), near_city (the near layer), ground (a dark disc of
    street level under the window)
    the world <name>_sky and the node groups <name>_params and <name>_blinds
Walls are single objects so views can hide them. Materials are procedural nodes: wallpaper stripes, pin stripes,
diamonds and paper grain (object coordinates), plank tone, grain, groove and varnish (UV maps "plank" and "rand"),
brush marks on the paint, painted aluminium slats.

Lights (the card's `lights`). window_key: a sun with a 1.5 degree disc that travels 22 degrees off the window's axis
(towards -x) and 50 degrees down, so it falls on the desk zone, the chair and the rug, not on the far floor; cool
violet, 1.1 W/m2 x window_glow; it casts the soft-edged shadows of the mullions and the slats (a wider disc would
wash the slat stripes out: 2 m from the blind the penumbra of a 1.5 degree disc is 5 cm, of a 5 degree one 17 cm,
seven slat pitches). window_glow: a 1.3 x 1.2 m area light just outside the opening, 22 W x window_glow (the glow on
the reveals and the sill). fill: a dim blue-violet area light under the ceiling, 7 W, so that no corner is black.
neon_light and neon_wash: area lights along the tube (one shines down the wall, one up at the frieze and the ceiling),
9 W and 7 W x neon. The desk lamp, a prop, is meant to dominate the desk; the set alone is a dark, moody,
blue-violet room.

Notes. `mk look` orbit views use a camera that clips at 100 m, which cuts the city (150 .. 500 m) away: look through
a [[shot]] camera (clip 5000 m) or `--cam`. The set builds in about 1.2 s, the skylines being the slow part. Every
colour is a blend of palette slots (bedroom_colors.py); the set is made for a dark palette (rose-pine-moon), other
palettes recolour it consistently but the light ones lose the night.

Card: kind "bedroom_80s", paths {}, lights [the light objects], colliders [{"type": "floor", "z": 0.0, "tag": name}],
room {min [-1.8, -1.6, 0], max [1.8, 1.6, 2.55]}, params {name: initial value} (the five above), and
    use.rest      floor: a plane at z = 0, normal +Z, size [3.6, 3.2] (the interior)
    use.surface   wall_back (centre [0, 1.6, 1.275], normal [0, -1, 0], up +Z, size [3.6, 2.55]), wall_left (centre
                  [-1.8, 0, 1.275], normal [1, 0, 0], size [3.2, 2.55]), wall_right (centre [1.8, 0, 1.275], normal
                  [-1, 0, 0]), wall_front (centre [0, -1.6, 1.275], normal [0, 1, 0]); the normals point into the
                  room. window: the glass (centre [0.4, 1.72, 1.55], normal [0, -1, 0], size [1.3, 1.2])
    use.look      window [0.4, 1.6, 1.55] (the opening, interior side), city (a point of the skyline straight ahead
                  of the window, about 380 m out), desk_zone [0.4, 1.0, 0.9], bed_zone [-1.3, 0.6, 0.6],
                  room [0, 0, 1.2]
    use.obstacles axis-aligned boxes in the set frame that props must not overlap: window (x -0.325 .. 1.125,
                  y 1.5 .. 1.6, z 0.89 .. 2.265: casing, sill board, apron, blind and headrail) and door (x 1.7 ..
                  1.8, y -1.525 .. -0.475, z 0 .. 2.125: leaf, handle, casing). The same list is also the card's
                  top-level `obstacles`, which is where the placement stage reads it.
Sizes follow the spec (`size`, `window`, `door`). Walls above z 2.28 (H - 0.27) belong to the picture rail, the frieze
and the neon tube: hang things below that."""
import bpy

from ...runtime import CTX
from ..props.cafe_colors import Colors
from . import bedroom_lights as LT
from . import bedroom_maths as BM
from . import bedroom_mats as MT
from . import bedroom_parts as PT
from . import bedroom_shape as SH
from . import bedroom_window as WN
from . import register
from .bedroom_colors import roles
from .cafe_nodes import Ctx


# architecture: straight boards and slabs by nature; the `form` check lists them but does not score them (everything
# else passes on its own: the door leaf, the blind, the fixtures, the hardware, the neon, the ceiling rose)
EXEMPT = ("wall_back", "wall_left", "wall_right", "wall_front", "floor", "ceiling", "roof_shadow", "skirting",
          "picture_rail", "door_casing", "window_frame", "window_trim")


def make_materials(R, L, cfg):
    """Every material of the set by role name (bedroom_mats)."""
    herring = cfg["floor"] == "herringbone"
    return {
        "paper_x": MT.mat_wallpaper(R, L, "x", cfg["rail_z"], "WallpaperX"),
        "paper_y": MT.mat_wallpaper(R, L, "y", cfg["rail_z"], "WallpaperY"),
        "reveal": MT.mat_plaster(R, "Reveal", L["plaster"], scale=9.0, rough=0.8),
        "outside": MT.mat_plain(R, "Outside", L["outside"], 1.0),
        "door_dark": MT.mat_plain(R, "DoorDark", L["door_dark"], 0.8),
        "ceiling": MT.mat_plaster(R, "Ceiling", L["ceiling"]),
        "trim": MT.mat_paint(R, "Trim", L["trim"]),
        "trim_dark": MT.mat_paint(R, "TrimDark", L["trim_dark"]),
        "door": MT.mat_paint(R, "Door", L["door"], rough=0.36),
        "parquet": MT.mat_parquet(R, L, 0.28 if herring else 1.2, 0.07 if herring else 0.11),
        "groove": MT.mat_plain(R, "Groove", L["floor_groove"], 0.7),
        "glass": MT.mat_glass(R, L),
        "slat": MT.mat_slat(R, L),
        "string": MT.mat_plain(R, "String", L["string"], 0.9),
        "cord": MT.mat_plain(R, "Cord", L["cord"], 0.7),
        "metal": MT.mat_plain(R, "Metal", L["metal"], rough=0.22, metal=1.0),
        "plate": MT.mat_plain(R, "Plate", L["plate"], rough=0.4, spec=0.4),
        "insert": MT.mat_plain(R, "Insert", L["insert"], rough=0.5),
        "rocker": MT.mat_plain(R, "Rocker", L["rocker"], rough=0.45),
        "bulb": MT.mat_plain(R, "Bulb", L["bulb"], rough=0.25, spec=0.6),
        "neon": MT.mat_neon(R, L),
    }


@register("bedroom_80s")
def bedroom_80s(name, coll, root, spec, palette):
    cfg = BM.parse(name, spec)
    sc = bpy.context.scene
    proj = CTX.get("project") or {}
    C = Colors(palette)
    L = roles(C, cfg["colors"])

    # ---- parameters: custom properties on the root, read through drivers
    root["mk_set"] = "bedroom_80s"
    for k, (default, lo, hi, doc) in BM.PARAMS.items():
        root[k] = float(cfg["params"][k])
        root.id_properties_ui(k).update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=doc, default=default)
    params = MT.params_group(name, root)
    frame0, fps = int(proj.get("frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    R = Ctx(name, coll, root, C, params, spec, frame0, fps)
    if cfg["render"]:
        LT.setup_render(sc)

    # ---- the room
    mats = make_materials(R, L, cfg)
    made = {}
    for build in (PT.build_shell, PT.build_trim, PT.build_door, PT.build_fixtures, PT.build_neon, PT.build_pendant,
                  WN.build_window):
        made.update(build(R, cfg, mats))

    # ---- outside, lights, world
    outside = LT.build_outside(R, cfg, palette, params)
    SH.exempt(*(made[k] for k in EXEMPT if k in made), *outside["objects"])
    neon = PT.neon_line(cfg) if cfg["neon"] is not None and cfg["neon_wall"] not in cfg["open"] else None
    lights = LT.build_lights(R, cfg, L, neon)
    if cfg["sky"] is None:
        LT.build_world(R, L)
    bpy.context.view_layer.update()

    w = cfg["window"]
    city = outside["city_point"] or [w["x"], cfg["y1"] + 380.0, 5.0]
    return BM.card(cfg, [o.name for o in lights] + outside["lights"], {k: root[k] for k in BM.PARAMS}, city)
