"""Layout of the semi truck: every number the parts agree on (pure Python, no bpy: importable from tests).

The rig is an American long-nose conventional tractor coupled to a 53 ft dry van, generic and unbadged (no logo, no
lettering). Proportions from a long-hood class tractor's body-builder drawing (131 inch bumper to back of cab, set-forward
steer axle 31 inch behind the bumper, 99.9 inch from that axle to the back of the cab, 81.8 inch from the bottom of the
frame rails to the cab roof, 94 inch across the front fenders, 118.5 inch mirror to mirror) and the 53 ft van's legal
envelope (102 inch wide, 13 ft 6 inch tall, kingpin 36 inch behind the front wall).

Frame: metres, Z up, the ground at z = 0, FORWARD -Y, +X the truck's left: the driver's side (left-hand drive). Origin on
the ground on the centre line, midway between the two drive axles (the tandem's centre): the point a [[vehicle]] keeps on
its lane. The tractor runs from the bumper at y = Y_BUMPER (-7.14) to the frame's end at y = FRAME_Y[1]; the trailer from
TRAILER_Y[0] to TRAILER_Y[1] (+14.94). Everything of the trailer is parented to the empty `<name>_trailer` at the kingpin
(the builder moves those meshes into its frame), so the trailer can pivot there.

Wheels: every tyre is 11R22.5 on a 22.5 x 8.25 polished ten-stud wheel; its lowest point is at z = 0 (WHEEL_Z = TYRE_R).
A wheel object (`<name>_wheel_<axle>_<side>`) is the part that turns: tyre(s), rim(s), the hub and drum, the hub cap and
the lug nuts, origin on the axle's centre line at the tyre's centre (steer) or midway between the two tyres of a dual pair
(where the two discs are clamped on the hub's face), unrotated, spinning about local +X (a positive turn rolls the tyre
forward). The -X wheels are mirror images, so one spin angle rolls every wheel forward. What does not turn (the axle beam,
spindles, brake chambers, springs, air bags, frame) is static and reaches into the hub's bore (SPINDLE_R), so no wheel
floats.
"""
import math

# =================================================================================================== wheels and tyres
TYRE_R = 0.527                 # 11R22.5: 1054 mm overall
TYRE_W = 0.279                 # section width (11 in)
RIM_R = 0.2858                 # bead seat radius (22.5 in)
RIM_W = 0.2096                 # 8.25 in between the flanges
RIM_FLANGE_R = RIM_R + 0.0127  # the flange the bead sits against
WHEEL_Z = TYRE_R               # every axle's centre height: the tyres stand on z = 0
LUGS = 10                      # hub-piloted, ten studs
BOLT_R = 0.1429                # 285.75 mm bolt circle
PILOT_R = 0.1105               # 221 mm centre bore (the hub pilots in it)
DUAL_SPACING = 0.330           # tyre centre to tyre centre of a dual pair (13 in)
WHEEL_OFFSET = 0.142          # a disc's mounting face from its tyre's centre line: a steer wheel's face is that far
                              # outboard of the tyre centre, a dual pair's two discs meet midway between the tyres
HUB_R = 0.135                  # the hub's barrel behind the mounting face (the drum's back is bigger: DRUM_R)
DRUM_R = 0.210                 # brake drum (16.5 in), inside the rim barrel
SPINDLE_R = 0.055              # the axle's spindle / tube end that the hub turns on

# =================================================================================================== axles
# steer axle and the drive tandem (52 in spread), the trailer's tandem (49 in spread); y of each axle's centre line
AXLE_S_Y = -6.35                                   # set-forward steer axle
AXLE_D_Y = (-0.66, 0.66)                           # drive axles D1, D2
WHEELBASE = -AXLE_S_Y                              # steer axle to the tandem centre (250 in)
STEER_TRACK = 2.08                                 # tyre centre to tyre centre on the steer axle
DRIVE_TRACK = 1.84                                 # centre of one dual pair to the other's (72.5 in)
STEER_RATIO = 20.0                                 # steering wheel turns per road-wheel turn (heavy truck)

# =================================================================================================== frame and coupling
FRAME_X = 0.432                # outer face of each rail's web: a 34 in wide ladder frame
RAIL_H, RAIL_FLANGE, RAIL_T = 0.273, 0.089, 0.0095  # C channel: 10.75 in tall, 3.5 in flanges (inward), 3/8 in thick
FRAME_Z = (0.780, 0.780 + RAIL_H)                  # bottom and top of the rails
FRAME_Y = (AXLE_S_Y - 0.498, AXLE_D_Y[1] + 0.72)   # front of frame (19.6 in ahead of the steer axle) and its rear end
KINGPIN_Y = -0.30                                  # fifth wheel's pivot: a foot ahead of the tandem centre
FIFTH_TOP_Z = 1.20                                 # top of the fifth wheel plate = the trailer's coupler plate
FIFTH_HALF_W = 0.46                                # the plate is 36 in across

# =================================================================================================== the tractor's body
Y_BUMPER = AXLE_S_Y - 0.787                        # front face of the bumper (31 in ahead of the steer axle)
BBC = 3.325                                        # bumper to back of cab (130.9 in)
BOC_Y = Y_BUMPER + BBC                             # back of the cab
BUMPER_Y = (Y_BUMPER, Y_BUMPER + 0.22)             # the bumper's face and the back of its end wraps
BUMPER_Z = (0.42, 0.86)
BUMPER_HALF_W = 1.22

GRILLE_Y = Y_BUMPER + 0.10                         # the grille shell's front face (the bumper stands 10 cm proud)
GRILLE_HALF_W = 0.51                               # outer half width of the grille shell
GRILLE_Z = (0.90, 1.99)                            # bottom and top of the shell
HOOD_Z = (1.985, 2.035)                            # hood top on the centre line: behind the grille, at the cowl
HOOD_TOP_HALF_W = (0.60, 0.74)                     # half width of the flat hood top (to its side roll): front, rear
COWL_Y = -5.50                                     # the hood's rear edge (it tilts forward about the front)
FENDER_HALF_W = 1.195                              # 94 in across the front fenders
FENDER_TOP_Z = 1.34                                # crown of the front fender over the tyre
FENDER_ARCH_R = 0.66                               # the fender's opening round the steer tyre, about the axle
HEADLAMP = (0.80, Y_BUMPER + 0.10, 1.17)           # centre of the +X lamp pod's lens, in the grille shell's front plane (a 7 cm
                                                   # pod stands out of the fender behind it); x mirrored for the other

CAB_FRONT_Y = -5.10                                # the cab's front wall behind the cowl: the door's front shut line
CAB_HALF_W = 1.00                                  # outside of the cab at the doors
CAB_BOTTOM_Z = 1.10                                # bottom of the cab's sides (the rocker)
CAB_FLOOR_Z = 1.25                                 # top of the cab floor inside
CAB_ROOF_Z = 2.86                                  # top of the cab roof (81.8 in over the frame's bottom)
CAB_WALL_T = 0.06                                  # side and back wall thickness: the inside face is at CAB_HALF_W - CAB_WALL_T
WS_BASE = (-5.16, 2.06)                            # (y, z) foot of the windshield on the centre line (just over the hood's rear edge)
WS_TOP = (-4.89, 2.68)                             # (y, z) its top at the header: raked 24 degrees back from vertical
WS_HALF_W = (0.88, 0.86)                           # half width of the glass at its foot / at its top (two flat panes, a centre post)
WS_RAKE_DEG = math.degrees(math.atan2(WS_TOP[0] - WS_BASE[0], WS_TOP[1] - WS_BASE[1]))   # from vertical (~24)
DOOR_Y = (-5.06, -3.93)                            # the door's front and rear shut lines
DOOR_Z = (CAB_BOTTOM_Z + 0.02, CAB_ROOF_Z - 0.10)  # its bottom and top
DOOR_GLASS = ((-4.98, 1.99), (-4.02, 2.63))        # (y, z) of the door window's front-bottom and rear-top corners
SLEEPER_Y = (BOC_Y + 0.046, BOC_Y + 0.046 + 1.60)  # a 63 in sleeper, 1.8 in behind the cab (a rubber boot bridges the gap)
SLEEPER_ROOF_Z = 3.22                              # a mid roof: its front cap rises from the cab roof
SLEEPER_HALF_W = 1.00
# the walk-through from the cab into the sleeper: nearly the cab's inside width (68 in) and 49 in tall from the floor (the
# low-roof opening of the drawing's sleeper table); the seats' high backs stand in it, the curtain hangs behind them
WALKTHROUGH = (0.865, CAB_FLOOR_Z, CAB_FLOOR_Z + 1.245)   # half width, bottom z, top z (in the cab's back wall at BOC_Y)
CURTAIN_Y = SLEEPER_Y[0] + 0.12                   # the sleeper curtain's line, behind the seat backs

# where the hardware stands (the parts' modules shape it; these are the places others keep clear of)
STACK_X, STACK_Y, STACK_TOP_Z, STACK_R = 1.07, BOC_Y + 0.023, 4.02, 0.089   # two 7 in chrome stacks in the notch between the
                                                                            # cab's and the sleeper's rounded back corners
AIR_CLEANER = (0.90, -5.30, 1.24, 2.02, 0.18)     # x, y, bottom z, top z, radius: canisters on the cowl's sides
STEP_Y = (-5.05, -4.35)                            # the step box under the door (two treads)
TANK_Y, TANK_X, TANK_Z, TANK_R = (-3.62, -2.22), 0.84, 0.75, 0.33   # a 26 in round fuel tank each side, under the sleeper (the
                                                                    # stack drops between the step box and the tank)
DECK_Y = (SLEEPER_Y[1] + 0.05, KINGPIN_Y - 0.55)   # tread-plate deck between the rails behind the sleeper
MIRROR = (1.40, CAB_FRONT_Y - 0.02, 2.25)          # centre of the +X West Coast mirror head (0.21 x 0.46 m; its outer edge at
                                                   # the front view's 118.5 in mirror to mirror)

# =================================================================================================== the cab inside
SEAT_X = 0.47                                      # the driver's seat centre (+X); the passenger's at -SEAT_X
CUSHION_TOP = CAB_FLOOR_Z + 0.45                   # air-ride seat cushion
CUSHION_Y = (-4.46, -4.00)                         # front edge, back edge of the cushion
BACK_RECLINE_DEG = 12.0                            # backrest from vertical
HIP = (SEAT_X, -4.18, CUSHION_TOP + 0.13)          # where the pose stage puts the hip joints (sit `hip`)
WHEEL_R, WHEEL_TUBE = 0.245, 0.016                 # steering wheel: rim radius to the tube's centre (a 20 in wheel), tube radius
WHEEL_TILT_DEG = 40.0                              # the wheel's plane from horizontal (its column 50 degrees over horizontal)
WHEEL_C = (SEAT_X, HIP[1] - 0.34, HIP[2] + 0.36)   # rim centre: above the knees, the lower rim clear of the thighs
WHEEL_AXIS = (0.0, math.sin(math.radians(WHEEL_TILT_DEG)), math.cos(math.radians(WHEEL_TILT_DEG)))   # toward the driver
WHEEL_ROT = (WHEEL_TILT_DEG, 0.0, 180.0)           # Euler (deg) of the steering wheel's own frame: its rim in the local XY
                                                   # plane, +Z the column axis toward the driver (WHEEL_AXIS), +Y 12 o'clock,
                                                   # +X 3 o'clock as the driver sees it (the world's -X)
DASH_Y = (WS_BASE[0] + 0.04, -4.86)                # the dash from the windshield's foot back to its face
DASH_TOP_Z = WS_BASE[1] - 0.02
GAUGES_C = (SEAT_X, -4.885, 1.93)                  # centre of the driver's gauge cluster (seen through the wheel)
SHIFTER_BASE = (0.19, -4.50, CAB_FLOOR_Z)          # the gear lever's boot on the floor (right of the driver's knee)
DOME_C = (0.0, -4.32, CAB_ROOF_Z - 0.09)           # the dome light on the headliner
FEET = {"L": (SEAT_X + 0.12, -4.80), "R": (SEAT_X - 0.10, -4.84)}   # heels' targets (x, y) on the cab floor

# =================================================================================================== the trailer
KINGPIN_SET = 0.914                                # 36 in from the front wall to the kingpin
TRAILER_LEN = 16.154                               # 53 ft
TRAILER_Y = (KINGPIN_Y - KINGPIN_SET, KINGPIN_Y - KINGPIN_SET + TRAILER_LEN)
TRAILER_HALF_W = 1.295                             # 102 in
TRAILER_TOP_Z = 4.115                              # 13 ft 6 in
TRAILER_FLOOR_Z = 1.335                            # top of the floor (the coupler plate at FIFTH_TOP_Z, 4 in crossmembers, the boards)
TRAILER_RAIL_Z = (1.20, 1.36)                      # the bottom rail along each side: its lower lip and its top
TRAILER_TOP_RAIL_Z = (4.03, TRAILER_TOP_Z)         # the top rail
TRAILER_AXLE_Y = (KINGPIN_Y + 12.0 - 1.245, KINGPIN_Y + 12.0)   # a 49 in tandem slid to 39.4 ft kingpin-to-rear-axle
TRAILER_TRACK = 1.97                               # dual centre to dual centre (77.5 in)
LANDING_Y = TRAILER_Y[0] + 3.0                     # the landing gear's legs
LANDING_X = 0.95
# the sliding tandem's box under the floor (the trailer builds it; the trailer suspension hangs from its rails)
SLIDER_X = 0.48                                    # centre of each slider rail (the rails sit between the inner tyres)
SLIDER_Z = (1.00, 1.20)                            # bottom and top of the slider rails (under the crossmembers)
SLIDER_Y = (TRAILER_AXLE_Y[0] - 1.55, TRAILER_AXLE_Y[1] + 0.75)
# the flat wall areas (between the front corner caps, the rear frame, the bottom rail and the top rail): the `use.surface`
# panels `left`, `right` and `back`, and where a mural image is mapped (mural_left / mural_right / mural_back faces)
SIDE_PANEL = (TRAILER_Y[0] + 0.10, TRAILER_Y[1] - 0.13, TRAILER_RAIL_Z[1], TRAILER_TOP_RAIL_Z[0])   # y0, y1, z0, z1
BACK_PANEL = (1.13, TRAILER_FLOOR_Z + 0.07, 3.90)  # half width, bottom z, top z of the rear doors' faces (at y = TRAILER_Y[1])


# =================================================================================================== materials
# One index per material role; every part's mesh carries these as its per-face material index. The builder
# (semi_truck.py) turns each into a Blender material coloured from the palette (semi_truck_mats.py).
MATS = (
    "paint",          # the body colour: hood, fenders, cab, sleeper
    "paint_accent",   # the second colour: stripes, two-tone panels
    "chrome",         # polished chrome and stainless: bumper, grille shell, stacks, air cleaners, horns, mirror arms, caps
    "alloy",          # polished aluminium: wheels, fuel tanks, steps' trim
    "chassis",        # satin black steel: frame, axles, springs, brackets, the fifth wheel
    "iron",           # dark grey cast iron: brake drums, hubs' barrels, spindles
    "rubber",         # bushings, air bags, hoses, door seals, the cab boot
    "tyre",           # tyre rubber
    "mudflap",        # mud flaps
    "glass",          # windows
    "mirror_glass",   # mirror faces
    "lens_head",      # headlamp lenses (glow with `lamps`)
    "lamp_amber",     # clearance and marker lamps (glow with `markers`)
    "lamp_red",       # tail lamps and rear markers (glow with `markers`, brighter with `brake`)
    "grille_back",    # the dark mesh behind the grille bars
    "seam",           # panel gaps
    "trailer_skin",   # the van's wall panels
    "trailer_metal",  # the van's brushed aluminium rails, posts, front cap, rear frame, doors' hardware
    "trailer_under",  # the van's underside: crossmembers, coupler plate, landing gear
    "tape_red",       # conspicuity tape, red blocks (a graphic layer)
    "tape_white",     # conspicuity tape, white blocks
    "tread_plate",    # diamond plate: steps, deck, the fifth wheel's walk board
    "vinyl",          # upholstery: seats, door inserts (the interior colour)
    "vinyl_dark",     # dash, trim panels, the column
    "carpet",         # the cab floor
    "headliner",      # the cab ceiling
    "gauge_face",     # gauge dials (lit with `dash_on`)
    "gauge_needle",   # gauge needles and the lit markings
    "switch",         # toggle switches, knobs, small dark plastic
    "plastic_dark",   # CB radio body, the mic, consoles
    "display",        # the CB's channel digits (lit)
    "steering",       # the steering wheel's rim
    "dome",           # dome light lens (lit with `dome`)
    "curtain",        # the sleeper curtain
    "woodgrain",      # the dash's wood-look panel
    "underbody",      # generic dark: the cab's underside, inner fender liners
    "mural_left",     # the outer faces of the van's +X side wall inside SIDE_PANEL: trailer_skin, or the `mural` image
    "mural_right",    # the -X side wall's
    "mural_back",     # the rear doors' outer faces inside BACK_PANEL
    "shield",         # the stacks' perforated heat shields (polished, a staggered grid of holes drawn from the uv)
    "hose_red",       # the emergency air line and its glad hand (glossy plastic)
    "hose_blue",      # the service air line and its glad hand
)
M = {name: i for i, name in enumerate(MATS)}

# colour roles (the card's `slots`; `[[prop]] slots = {body = "pine"}` recolours): role -> default colour spec, a palette slot,
# a '#hex', a blend 'slot:w,slot:w' (linear light) whose parts may be 'a:b:t' mixes (convertible_palette.Pal)
ROLES = {
    "body": "love",                         # paint: hood, fenders, cab, sleeper
    "accent": "gold",                       # the second paint: stripes and two-tone panels
    "trim": "text",                         # the tint of bright metal (chrome, polished aluminium)
    "chassis": "base",                      # frame, axles, springs, the fifth wheel
    "rubber": "base",                       # tyres, flaps, hoses, seals: the darkest slot, never pure black
    "glass": "foam",                        # the windows' tint
    "headlamp": "text",                     # headlamp light and its beams
    "marker": "gold",                       # clearance and marker lamps
    "taillamp": "love",                     # tail lamps
    "van": "text",                          # the van's walls
    "van_trim": "subtle",                   # the van's aluminium: rails, posts, doors' frame
    "tape": "love",                         # conspicuity tape's red blocks (the white ones are `van`'s white: text)
    "interior": "gold:5,rose:3,overlay:2",  # upholstery: a warm tan
    "dash": "overlay:3,base:2",             # the dash and the trim panels
    "gauges": "text",                       # gauge faces
    "glow": "gold",                         # the gauges' backlight and needles, the dome light
    "display": "love",                      # the CB radio's channel digits
    "wood": "gold:3,rose:1,base:4",         # the dash's wood-look panel
    "curtain": "iris:2,base:3",             # the sleeper's curtain
    "hose_red": "love",                     # the emergency air line
    "hose_blue": "pine",                    # the service air line
}

# animatable root properties (key with [[key]]): name -> (default, doc)
PARAMS = {
    "lamps": (1.0, "headlamps: 0 off, 1 lit (the lenses and the beams)"),
    "markers": (1.0, "clearance, marker and tail lamps' running level"),
    "brake": (0.0, "brake lamps: 0 off, 1 full"),
    "dash_on": (1.0, "the gauges' backlight and the CB's digits"),
    "dome": (0.0, "the cab's dome light"),
}

# the builder's non-colour options (in the `slots`): number options and their defaults
NUM_OPTIONS = {"beam_power": 1100.0, "beam_angle": 50.0, "beam_tilt": 4.0}


def options(slots):
    """The builder's options from a prop's `slots`: the numbers of NUM_OPTIONS, `trailer` and `beams` (booleans, default
    true), `mural_fit` and `murals` {side: image reference} for the sides that have one (`mural` paints left and right;
    `mural_left`, `mural_right` and `mural_back` give one panel its own)."""
    s = dict(slots or {})
    o = {k: float(s.get(k, v)) for k, v in NUM_OPTIONS.items()}
    for k in ("trailer", "beams"):
        v = s.get(k, True)
        if not isinstance(v, bool):
            raise TypeError(f"semi_truck: {k} = {v!r}: true or false")
        o[k] = v
    o["mural_fit"] = str(s.get("mural_fit", "cover"))
    if o["mural_fit"] not in MURAL_FITS:
        raise ValueError(f"semi_truck: mural_fit = {o['mural_fit']!r}: one of {', '.join(MURAL_FITS)}")
    base = s.get("mural")
    murals = {side: s.get(f"mural_{side}", base if side != "back" else None) for side in ("left", "right", "back")}
    o["murals"] = {k: str(v) for k, v in murals.items() if v}
    if o["murals"] and not o["trailer"]:
        raise ValueError("semi_truck: a mural needs the trailer (trailer = false has no van to paint)")
    return o


def is_trailer(key):
    """Is a part key the trailer's (parented to `<name>_trailer`, left out without the trailer)?"""
    return str(key).startswith("trailer")


# =================================================================================================== helpers
def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def axles():
    """Every axle: (key, y, kind, track) from front to back: kind "steer" (single tyres), "drive" or "trailer" (duals);
    track is centre to centre of the tyres (steer) or of the dual pairs."""
    out = [("S", AXLE_S_Y, "steer", STEER_TRACK)]
    out += [(f"D{k + 1}", y, "drive", DRIVE_TRACK) for k, y in enumerate(AXLE_D_Y)]
    out += [(f"T{k + 1}", y, "trailer", TRAILER_TRACK) for k, y in enumerate(TRAILER_AXLE_Y)]
    return out


def wheel_key(axle, side):
    """Dynamic part key (and object suffix) of the wheel at one end of an axle: `wheel_S_L` ... `trailer_wheel_T2_R`.
    side "L" is +X (the driver's side), "R" is -X."""
    return f"{'trailer_' if axle.startswith('T') else ''}wheel_{axle}_{side}"


def wheel_hubs():
    """{wheel key: (x, y, z) of its origin}: the hub's centre on the axle line, at the middle of the tyre (steer) or
    midway between the two tyres of a dual pair."""
    out = {}
    for key, y, kind, track in axles():
        for side, sx in (("L", 1.0), ("R", -1.0)):
            out[wheel_key(key, side)] = (sx * track / 2, y, WHEEL_Z)
    return out


def tyre_centres(key):
    """x of the tyre centres of the wheel `key` (one for a steer wheel, two for a dual pair), signed."""
    x = wheel_hubs()[key][0]
    if key.startswith("wheel_S"):
        return [x]
    s = 1.0 if x > 0 else -1.0
    return [x - s * DUAL_SPACING / 2, x + s * DUAL_SPACING / 2]


# =================================================================================================== surfaces and murals
DOOR_PANEL = (DOOR_Y[0] + 0.07, DOOR_Y[1] - 0.07, CAB_BOTTOM_Z + 0.14, DOOR_GLASS[0][1] - 0.09)    # y0, y1, z0, z1 below the window
SLEEPER_PANEL = (SLEEPER_Y[0] + 0.16, SLEEPER_Y[1] - 0.76, CAB_BOTTOM_Z + 0.22, 2.30)              # the sleeper's side ahead of
LUGGAGE_DOOR = (SLEEPER_Y[1] - 0.66, SLEEPER_Y[1] - 0.20, CAB_BOTTOM_Z + 0.20, CAB_BOTTOM_Z + 0.82)  # the luggage door (low, at
                                                                                                  # the rear; its window over it)
SURFACE_LIFT = 0.004                               # a surface's centre stands this far out of the face it lies on


def surfaces(trailer=True):
    """The card's `use.surface` panels: {name, center, normal, up, size}. `left` / `right` are the van's side walls (the
    viewer's right runs toward the rear on `left`, toward the front on `right`), `back` its rear doors, `door_L` / `door_R`
    the cab doors below the windows, `sleeper_L` / `sleeper_R` the sleeper's sides. Without the trailer only the tractor's."""
    out = []

    def add(name, c, n, size):
        out.append({"name": name, "center": [round(float(v), 5) for v in c], "normal": [float(v) for v in n],
                    "up": [0.0, 0.0, 1.0], "size": [round(float(v), 4) for v in size]})

    for side, sx in (("L", 1.0), ("R", -1.0)):
        y0, y1, z0, z1 = DOOR_PANEL
        add(f"door_{side}", (sx * (CAB_HALF_W + SURFACE_LIFT), (y0 + y1) / 2, (z0 + z1) / 2), (sx, 0, 0), (y1 - y0, z1 - z0))
        y0, y1, z0, z1 = SLEEPER_PANEL
        add(f"sleeper_{side}", (sx * (SLEEPER_HALF_W + SURFACE_LIFT), (y0 + y1) / 2, (z0 + z1) / 2), (sx, 0, 0),
            (y1 - y0, z1 - z0))
    if trailer:
        y0, y1, z0, z1 = SIDE_PANEL
        for name, sx in (("left", 1.0), ("right", -1.0)):
            add(name, (sx * (TRAILER_HALF_W + SURFACE_LIFT), (y0 + y1) / 2, (z0 + z1) / 2), (sx, 0, 0), (y1 - y0, z1 - z0))
        hw, z0, z1 = BACK_PANEL
        add("back", (0.0, TRAILER_Y[1] + SURFACE_LIFT, (z0 + z1) / 2), (0, 1, 0), (2 * hw, z1 - z0))
    return out


MURAL_FITS = ("cover", "fill", "contain")


def mural_scale(fit, panel_aspect, image_aspect):
    """(su, sv): the image's texture coordinate is 0.5 + (panel coordinate - 0.5) * s along each axis, for a panel of
    aspect w / h showing an image of aspect w / h. "cover" fills the panel with the image uniformly scaled and centred,
    its overflow cropped; "contain" shows all of it inside the panel (outside it the coordinate leaves 0..1: the wall's
    own colour); "fill" stretches it over the panel."""
    if fit not in MURAL_FITS:
        raise ValueError(f"mural_fit = {fit!r}: one of {', '.join(MURAL_FITS)}")
    if fit == "fill":
        return 1.0, 1.0
    r = panel_aspect / image_aspect                   # > 1: the panel is wider than the image
    if (fit == "cover") == (r > 1.0):
        return 1.0, 1.0 / r                           # the image's width spans the panel
    return r, 1.0                                     # its height spans the panel
