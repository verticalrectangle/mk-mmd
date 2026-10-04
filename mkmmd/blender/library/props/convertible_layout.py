"""Layout of the 1980s convertible: every number the parts agree on (pure Python, no bpy, importable from tests).

Frame: the prop's local frame. Z up, the ground is z = 0, the origin is the car's centre on the ground. FORWARD IS -Y
(the nose is at -Y), the DRIVER sits on +X (left-hand drive, so +X is the car's left). Metres, degrees.
Everything that places a part (the body, the seats, the dash, the wheels, the card's use points) reads this module, so a
number changes in one place. The proportions follow a 1984-86 Dodge 600 convertible (measured on side-on photographs
against its 4.59 m length and 2.62 m wheelbase): a long flat hood, a shoulder crease at 0.83 m the whole way along the
side, a door top at 0.915 m, a windshield raked 45 degrees from the cowl up to a header at 1.32 m, a flat deck.
"""
import math

# ------------------------------------------------------------------------------------------------------ overall
LENGTH = 4.59                       # bumper tip to bumper tip
WIDTH = 1.73                        # body, without mirrors
HALF_W = WIDTH / 2
Y_FRONT, Y_REAR = -LENGTH / 2, LENGTH / 2         # bumper tips
Y_NOSE, Y_TAIL = -2.21, 2.21                      # body panels: the face of the nose, the face of the tail panel
WHEELBASE = 2.62
AXLE_F, AXLE_R = -WHEELBASE / 2, WHEELBASE / 2    # y of the axles (overhangs 0.98 front and rear)
TRACK = 1.46                                      # centre to centre of the tyres
TYRE_R, TYRE_W = 0.31, 0.205                      # 195/60R15
RIM_R = 0.1905                                    # 15 inch rim: the tyre's inner radius
WHEEL_Z = TYRE_R                                  # wheel centre height: the tyre stands on z = 0
WHEEL_X = TRACK / 2
ARCH_R = TYRE_R + 0.05                            # wheel opening radius: 5 cm round the tyre, then the flared lip
Z_BOT = 0.24                                      # underside of the body panels
Z_RUB = 0.44                                      # the rub strip crease: the paint splits here (two-tone), the rubber strip sits on it
Z_SHOULDER = 0.83                                 # the shoulder crease: one horizontal line along the whole side
Z_BELT = 0.915                                    # door top (the sill the passenger's arm rests on)
Z_COWL = 0.99                                     # the hood's rear edge, the highest point of the body
Z_GLASS = 0.925                                   # the foot of the windshield glass and the dash pad's top at the glass
Z_HOOD_NOSE = 0.80                                # hood height at the nose (it rises to Z_COWL at the cowl)
Z_DECK = 0.876                                    # deck lid top on the centre line
Y_COWL = -0.65                                    # foot of the windshield glass: where the cockpit opening starts
Y_HOOD_REAR = -0.74                               # the hood's rear edge; the cowl (wiper well) drops from it to the glass
Y_DOOR_FRONT = -0.68                              # the door's front shut line
Y_BOOT = 1.15                                     # where the rear clip (deck, boot compartment) starts
Y_DOOR_REAR = 0.62                                # the door's rear shut line (the body wall continues to Y_BOOT)
X_BELT_OUT = 0.835                                # outer edge of the door top (the sides lean in above the crease)
WALL_T = 0.06                                     # thickness of the door / quarter wall at the belt
X_WALL_IN = X_BELT_OUT - WALL_T                   # inner face of the wall: 0.775
FLOOR_Z = 0.23                                    # carpet height

# windshield: a plane from the cowl up to the header
WS_BASE = (Y_COWL, Z_GLASS)                       # (y, z) at the foot of the glass
WS_TOP = (-0.24, 1.32)                            # (y, z) of the header rail (the A-pillar measured on a side-on photo: 44 deg)
WS_HALF_W_BASE, WS_HALF_W_TOP = 0.76, 0.70        # half width at the cowl / at the header (outer edge of the frame)
WS_FRAME = 0.05                                   # frame member width: the A-pillars and the header, body colour
WS_RAKE_DEG = math.degrees(math.atan2(WS_TOP[1] - WS_BASE[1], WS_TOP[0] - WS_BASE[0]))   # from horizontal (~44)
VISOR_FLIP_DEG = 180.0 - WS_RAKE_DEG              # a sun visor turns this far about its rod from hanging on the glass to level, pointing back (~136)

# hood profile (y, z) on the centre line: the long flat hood, rolling over at the nose
HOOD_PROFILE = ((Y_NOSE, Z_HOOD_NOSE), (-2.11, 0.84), (-1.71, 0.92), (-1.31, 0.956), (-0.9, 0.978), (Y_HOOD_REAR, Z_COWL),
                (-0.695, 0.962), (Y_COWL, Z_GLASS + 0.005))


def hood_z(y):
    """Height of the hood and cowl surface on the centre line at y (piecewise linear through HOOD_PROFILE)."""
    pts = HOOD_PROFILE
    if y <= pts[0][0]:
        return pts[0][1]
    for (y0, z0), (y1, z1) in zip(pts[:-1], pts[1:]):
        if y <= y1:
            return z0 + (z1 - z0) * (y - y0) / (y1 - y0)
    return pts[-1][1]


HOOD_PITCH_DEG = math.degrees(math.atan2(Z_COWL - Z_HOOD_NOSE, Y_HOOD_REAR - Y_NOSE))    # nose down, ~7 deg

# ------------------------------------------------------------------------------------------------------ the cabin
SEAT_X = 0.42                                     # driver +X, passenger -X (a passenger's shoulder is ~0.29 m from the door top)
CUSHION_TOP = 0.49                                # seat surface at the hip (contoured cushion, lowest point)
HIP_Y = 0.15
HIP_Z = CUSHION_TOP + 0.15                        # hip joint 15 cm above the cushion
CUSHION_Y = (-0.12, 0.41)                         # front edge, back edge of the front cushion
CUSHION_W = 0.50
BACK_BASE_Y = 0.21                                # y of the backrest's FRONT surface where it meets the cushion top (z = CUSHION_TOP);
                                                  # at hip height it is ~10 cm behind the hip joint (the sitter's back lies on it)
BACK_RECLINE_DEG = 16.0                           # backrest tilt from vertical (top toward +Y)
BACK_H = 0.60                                     # height of the backrest pad above the cushion
HEADREST_Z = (1.08, 1.25)
REAR_SEAT_Y = (0.58, 1.02)                        # rear bench cushion: front edge, back edge
REAR_SEAT_TOP = 0.46
REAR_SEAT_W = 1.22
REAR_BACK_RECLINE_DEG = 22.0

# steering wheel: ring about C with normal A pointing at the driver and up
WHEEL_C = (SEAT_X, -0.155, 0.99)                  # rim centre (shoulder to rim ~0.41 m: the elbows stay bent)
WHEEL_COL_DEG = 22.0                              # column rake: the wheel plane is 22 deg off vertical
WHEEL_AXIS = (0.0, math.cos(math.radians(WHEEL_COL_DEG)), math.sin(math.radians(WHEEL_COL_DEG)))
WHEEL_R, WHEEL_TUBE = 0.19, 0.017                 # rim radius (to the tube's centre), tube radius
STEER_RATIO = 14.0

# dash (instrument panel): profile in (y, z) pulled across the cabin width
DASH_X = X_WALL_IN                                # fills the cabin between the door panels
DASH_TOP_Y = (Y_COWL, -0.36)                      # the pad: from the cowl back to the lip
DASH_TOP_Z = (Z_GLASS, Z_GLASS + 0.025)
BINNACLE_C = (SEAT_X, -0.335, 1.015)              # centre of the cluster face
BINNACLE_TILT_DEG = 20.0                          # the cluster face leans back (toward +Y) like the wheel
CLUSTER_SIZE = (0.34, 0.11)                       # glass panel (w, h)

# centre stack with the cassette deck
STACK_X = (-0.15, 0.15)
DECK_FACE_Y = -0.30                               # y of the deck's face plate
DECK_Z = 0.80                                     # slot height
SLOT_C = (0.0, DECK_FACE_Y, DECK_Z)               # cassette slot centre; a cassette goes in along -Y
CASSETTE = (0.100, 0.064, 0.012)                  # standard compact cassette (x, y, z)
TAPE_OUT_Y = DECK_FACE_Y + 0.032 + 0.045          # centre y of the cassette held out in front of the slot (tape = 0)
TAPE_IN_Y = DECK_FACE_Y + 0.032 - 0.058           # fully inserted (tape = 1): 6 mm of it shows

# console between the seats
CONSOLE_X = 0.14                                  # half width
CONSOLE_Y = (-0.30, 0.66)
CONSOLE_Z = 0.58                                  # top of the front part (the armrest lid is higher)
SHIFTER_C = (0.0, -0.02, CONSOLE_Z)               # base of the shift lever

# rest edges (door top) and look points
SILL_X = X_BELT_OUT - WALL_T / 2                  # centre of the door top (0.805)
SILL_Y = (-0.55, 0.38)
LOOK_ROAD = (0.0, -30.0, 1.2)
MIRROR_C = (0.0, -0.26, 1.205)                    # rear-view mirror glass centre (its housing clears the glass by ~1 cm)

# fixed flush headlamps: a bezel with two clear lenses on the fascia either side of the grille; the spot lights sit at the lens
HEADLAMP_X = 0.515                                # centre of each lamp (+X driver side, -X passenger side)
HEADLAMP_Z = 0.672

# plates (US size) as surfaces
PLATE_SIZE = (0.305, 0.152)
PLATE_FRONT = (0.0, Y_FRONT - 0.004, 0.40)        # centre; faces -Y (on the bumper face)
PLATE_REAR = (0.0, Y_REAR + 0.004, 0.40)          # centre; faces +Y

# ------------------------------------------------------------------------------------------------------ materials
# One index per material role; every part's mesh carries these as its per-face material index. The builder
# (convertible.py) turns each role into a Blender material coloured from the palette.
MATS = (
    "paint",          # body paint: clear coat, two-tone below a pinstripe (shader, object-space z)
    "underbody",      # dark matte: wheel-arch roofs, underside, anything nobody should see lit
    "chrome",         # bright trim: bumper caps, window frame, mirrors, handles, grille bars, tail pipe
    "rubber",         # black rubber (the darkest slot, never pure black): bumper strips, rub strip, seals, wipers
    "tyre",           # tyre
    "alloy",          # wheel alloy
    "glass",          # windshield (transparent, reflective)
    "lens_head",      # headlamp lens: clear, lights up with `lamps`
    "lamp_tail",      # tail-lamp band: red, glows, brighter with `brake`
    "lamp_amber",     # side marker / turn signal lenses
    "plate",          # blank licence plates
    "vinyl",          # upholstery: seats, door panel tops (warm tan)
    "vinyl_dark",     # dash pad, door-top rolls, darker upholstery
    "carpet",         # floor, lower door panels
    "trim_dark",      # dark plastic: vents, knob bezels, pedals, console body
    "wood",           # simulated woodgrain accents
    "steering",       # steering wheel rim and pad
    "deck_face",      # cassette deck face plate
    "vfd_ghost",      # unlit display segments (dim, driven by dash_on)
    "vfd_lit",        # lit display elements and bar graph cells (driven by dash_on and the bar level)
    "display_glass",  # dark glass behind the displays
    "ind_a",          # warning light: gold
    "ind_b",          # warning light: pine
    "ind_c",          # warning light: love
    "cassette",       # cassette shell
    "label",          # cassette label paper
    "stripe_a",       # label stripes
    "stripe_b",
    "stripe_c",
    "boot",           # padded top boot (folded soft top cover)
    "mirror_glass",   # mirror glass
    "seam",           # panel gap lines
    "bumper",         # the bumper's body: the lower paint colour, a little duller than the body
)
M = {name: i for i, name in enumerate(MATS)}


# ------------------------------------------------------------------------------------------------------ helpers
def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def plane_frame(normal, up_hint=(0.0, 0.0, 1.0)):
    """(normal, up, right) of a surface facing `normal` with its top edge toward `up_hint`: right = up x normal, the
    direction a viewer in front of the surface sees to the right (the card's use.surface convention)."""
    n = unit(normal)
    d = sum(a * b for a, b in zip(up_hint, n))
    up = unit(tuple(a - d * b for a, b in zip(up_hint, n)))
    return n, up, unit(cross(up, n))


def tilt_normal(tilt_deg, toward=(0.0, 1.0, 0.0)):
    """Unit normal facing `toward` (default +Y, the driver) and tipped up by tilt_deg."""
    t = math.radians(tilt_deg)
    return unit((toward[0] * math.cos(t), toward[1] * math.cos(t), math.sin(t)))


def arch_circle(axle_y):
    """(centre y, centre z, radius) of a wheel opening."""
    return axle_y, WHEEL_Z, ARCH_R


CLUSTER_FRAME = plane_frame(tilt_normal(BINNACLE_TILT_DEG))     # normal, up, right of the cluster face
