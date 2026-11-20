"""Layout of the practice amp (pure Python + numpy, no bpy: importable from tests): its numbers, meshes, colour roles,
options and card.

A 10-15 W guitar practice combo, unbranded. Measured: the Fender Frontman 15G (337 x 314 x 181 mm, an 8 inch speaker:
its manual), the Frontman 10G (260 x 279 x 146) and the Orange Crush 12 (305 x 290 x 175) for the class; the front's
layout read off a front photograph of a Frontman 15G at its 337 mm width (`mk ref photos 'guitar practice amplifier'`,
`'Fender Frontman amplifier'`): a tolex frame about 18 mm wide round a recessed field, the control panel across the field's
top about 58 mm tall, the grille cloth under it, knobs 26-40 mm apart in a row with the input jack at the left end and the
power lamp at the right, a strap handle about 190 mm long between end caps 33 mm wide, small corner protectors (a Crate
practice amp's are bigger). This one is 330 x 300 x 180 mm, a little rounder and chunkier than its models.

Frame (metres): it stands on four rubber feet on the floor, origin under the middle of its footprint, its front toward -Y,
+Z up. The cabinet's underside is at FOOT_H, its top at FOOT_H + H; the handle lies on the top.

- cabinet: ONE closed cage of rings round the depth (rounded rectangles inset ring by ring, as the speaker stack's): the
  back rolled into the sides, the front edge rolled over into the tolex frame, its inner roll down a short wall into a
  cove onto the baffle, RECESS behind the front, with support rings beside every roll so the flat faces shade flat.
- grille: the cloth stretched over its board, a plate with rolled edges and a slight pillow, set into the field under the
  panel; the piping: a welt round it in the joint between the cloth and the frame.
- panel: a plate across the field's top with a printed border and a scale of dots round each knob (graphic layers);
  `knobs` skirted knobs with knurled caps and a pointer line (each turned to its own setting), the input jack at the left
  end (a washer, a nut and a threaded sleeve round the bore), a rocker switch and a faceted jewel lamp at the right end.
- the strap handle on top between two end caps with their screws, four rubber feet, and (`corners`) eight corner
  protectors.

Colour roles (`slots`: a palette slot, `ink` / `paper`, a #hex or an `a:b:t` mix): tolex (rose), grille (text: the
cloth; its threads are lighter), piping (surface), panel (surface), print (text: the panel's border and dots), knobs
(text; their pointer lines take the piping's colour), handle (text), metal (hl_high: the jack, the lamp's bezel, the
handle's caps and screws), lamp (love: the jewel), rubber (text: feet, the switch, the jack's bore, the baffle behind the
cloth), corners (text: the corner protectors; false: none). Options: knob_count (4: 3 .. 5), glow (0.8: the lamp's light,
0 off).

Card: `use.anchor` `jack` (the input jack's mouth, `dir` -Y: where a cable's plug goes in), `use.look` `front` (the
middle of the front) and `panel` (the middle of the control panel), a box collider, `form_max` FORM_MAX: a combo amp
is a box by nature (its cabinet reads as a box however round its edges)."""
import math

import numpy as np

from ....core import shell as S
from . import showkit as SK

W, H, D = 0.330, 0.300, 0.180    # the cabinet (the feet and the handle are extra)
FOOT_H, FOOT_R = 0.010, 0.014    # rubber feet
RIM = 0.019                      # the tolex frame's width round the front field, its roll included
R_EDGE = 0.009                   # the front and back edges' roll
R_CORNER = 0.014                 # the corners of the section (the edges along the depth)
R_IN = 0.003                     # the frame's inner roll into the field
R_COVE = 0.003                   # the cove where the field's wall meets the baffle
RECESS = 0.014                   # how far the baffle lies behind the front
SUPPORT = 0.003                  # support rings this far beside a roll
R_MIN = 0.004                    # the least corner radius of an inset ring (the field's corners)
Y_FRONT = -D / 2
Y_BAFFLE = Y_FRONT + RECESS

PANEL_H, PANEL_T = 0.058, 0.005  # the control panel's height and thickness (a plate on the baffle)
PANEL_GAP = 0.0015               # between the panel and the field's walls
GRILLE_T, GRILLE_DOME = 0.0058, 0.0016   # the cloth on its board, and its pillow
GRILLE_GAP = 0.0048              # between the cloth and the field's walls / the panel: the piping lies in it
SINK = 0.001                     # the cloth's and the panel's backs sink this far into the baffle
PIPE = (0.0021, 0.0043)          # the welt's half width across and half depth
KNOB = {"skirt": 0.0125, "cap": 0.0094, "h": 0.0172, "ribs": 24, "knurl": 0.035}
DOTS = (0.0162, 0.00072, 11, 300.0)      # a knob's scale: radius, dot radius, count, arc (degrees)
SETTINGS = (-62.0, 38.0, 75.0, -20.0, 110.0)   # each knob's pointer, degrees clockwise from 12 o'clock
JACK_X, SWITCH_X, LAMP_X = -0.118, 0.086, 0.119
JACK_MOUTH = 0.0061              # the jack's sleeve stands this far out of the panel's face
HANDLE = {"x": 0.078, "cap": (0.036, 0.030, 0.010), "strap": (0.026, 0.0055), "arch": 0.017}
CORNER = (0.026, 0.0095, 0.0022)  # a corner protector's size, its rounding, how far it stands proud of the cabinet
ROLES = {"tolex": "rose", "grille": "text", "piping": "surface", "panel": "surface", "print": "text", "knobs": "text",
         "handle": "text", "metal": "hl_high", "lamp": "love", "rubber": "text", "corners": "text"}
MATS = tuple(ROLES)              # material index -> role
FORM_MAX = 0.7                   # a box by nature, as the boombox: the tolex cabinet (two thirds of the surface) reads
#                                  as a box however its edges roll; every other part is round (form ~0.66 measured)
PARTS = ("cabinet", "grille", "piping", "panel", "print", "knobs", "hardware", "jewel", "handle")
GRAPHIC = ("print",)             # flat graphic layers: tagged form-exempt


def _m(role):
    return MATS.index(role)


def options(slots):
    """{knob_count, corners (whether there are corner protectors), glow} from the `slots` of the [[prop]], checked: a bad
    value is a ValueError that says what to change (`knobs` and `corners` are colour roles too: see `roles`)."""
    slots = slots or {}
    corners = slots.get("corners", True)
    if not isinstance(corners, (bool, str)):
        raise ValueError(f"practice_amp: corners is the protectors' colour, or false for none (got {corners!r})")
    return {"knob_count": SK.number(slots, "knob_count", 4, "practice_amp", 3, 5, whole=True),
            "corners": corners is not False, "glow": SK.number(slots, "glow", 0.8, "practice_amp", 0.0, 20.0)}


def roles(slots):
    """{role: colour spec} of every colour role, checked: the `slots` entry, else ROLES' (`corners` = true / false says
    whether there are protectors and keeps their default colour)."""
    s = dict(slots or {})
    if isinstance(s.get("corners"), bool):
        s.pop("corners")
    return SK.roles(s, ROLES, "practice_amp")


# ------------------------------------------------------------------------------------------------------------- front
def field():
    """(half width, z0, z1) of the front field inside the tolex frame, in the prop's frame."""
    return W / 2 - RIM, FOOT_H + RIM, FOOT_H + H - RIM


def panel_box():
    """(half width, z0, z1) of the control panel."""
    hw, _, z1 = field()
    return hw - PANEL_GAP, z1 - PANEL_GAP - PANEL_H, z1 - PANEL_GAP


def grille_box():
    """(half width, z0, z1) of the grille cloth."""
    hw, z0, _ = field()
    return hw - GRILLE_GAP, z0 + GRILLE_GAP, panel_box()[1] - GRILLE_GAP


def row_z():
    """The height of the knob row (the panel's middle)."""
    _, z0, z1 = panel_box()
    return 0.5 * (z0 + z1)


def knob_xs(n):
    """The knobs' x, left to right as you face the amp."""
    x0, x1 = -0.090, 0.056                                     # between the jack and the switch, scales clear of both
    step = min(0.040, (x1 - x0) / (n - 1))
    mid = 0.5 * (x0 + x1)
    return [round(mid + (i - 0.5 * (n - 1)) * step, 6) for i in range(n)]


def on_front(m, x, z, y_face):
    """A mesh modelled about its own +Z (its face up, its +Y up the picture) stood on a face looking -Y at (x, y_face,
    z)."""
    return S.place(m, (x, y_face, z), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))


# ----------------------------------------------------------------------------------------------------------- cabinet
def shell_profile(n=4):
    """(inset d, depth h) of the cabinet's rings from the back (h 0) to the baffle (h D - RECESS)."""
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)
    back = [(R_EDGE * (1.0 - math.sin(t)), R_EDGE * (1.0 - math.cos(t))) for t in a]
    front = [(R_EDGE * (1.0 - math.cos(t)), D - R_EDGE + R_EDGE * math.sin(t)) for t in a]
    rim = [(RIM - R_IN + R_IN * math.sin(t), D - R_IN + R_IN * math.cos(t)) for t in a]
    cove = [(RIM + R_COVE * (1.0 - math.cos(t)), D - RECESS + R_COVE * (1.0 - math.sin(t))) for t in a]
    prof = ([(R_EDGE + SUPPORT, 0.0)] + back + [(0.0, R_EDGE + SUPPORT), (0.0, D - R_EDGE - SUPPORT)] + front
            + [(R_EDGE + SUPPORT, D), (RIM - R_IN - SUPPORT, D)] + rim + [(RIM, D - RECESS + R_COVE + SUPPORT)] + cove
            + [(RIM + R_COVE + SUPPORT, D - RECESS)])
    return [p for i, p in enumerate(prof) if i == 0 or abs(p[0] - prof[i - 1][0]) + abs(p[1] - prof[i - 1][1]) > 1e-9]


def cabinet_mesh(n=12):
    """The tolex cabinet (material tolex; the baffle's face, hidden behind the cloth and the panel, rubber)."""
    R = np.stack([np.column_stack([S.rrect(W - 2 * d, H - 2 * d, max(R_CORNER - d, R_MIN), n), np.full(4 * (n + 1), h)])
                  for d, h in shell_profile()])
    m = S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), mat=_m("tolex"))
    out = S.Mesh(np.stack([m.V[:, 0], D / 2 - m.V[:, 2], m.V[:, 1] + H / 2 + FOOT_H], 1), m.Q, m.T, m.Qm, m.Tm)
    out.assign(_m("rubber"), lambda ctr, nrm: (np.abs(ctr[:, 1] - Y_BAFFLE) < 1e-6) & (nrm[:, 1] < -0.99))
    return out


# ----------------------------------------------------------------------------------------------- grille, piping, panel
def grille_mesh():
    """The cloth on its board: a plate with rolled edges (the back's too: it sinks SINK into the baffle) and a pillow."""
    hw, z0, z1 = grille_box()
    m = S.rounded_panel(S.rrect(2 * hw, z1 - z0, 0.004, 4), GRILLE_T + SINK, r_edge=0.0022, n=3, dome=GRILLE_DOME, rings=6,
                        r_back=0.0015, mat=_m("grille"))
    return on_front(m, 0.0, 0.5 * (z0 + z1), Y_BAFFLE + SINK)


def piping_mesh():
    """The welt round the cloth: a closed tube of an oval section (deeper than wide), from the baffle to 1.4 mm proud of
    the cloth's face, in the gap between the cloth and the field's walls."""
    hw, z0, z1 = grille_box()
    g = GRILLE_GAP / 2
    ring = S.rrect(2 * (hw + g), z1 - z0 + 2 * g, 0.004 + g, 6)
    y = Y_BAFFLE - PIPE[1]
    path = np.column_stack([ring[:, 0], np.full(len(ring), y), ring[:, 1] + 0.5 * (z0 + z1)])
    m = S.tube(path, PIPE[0], sides=12, closed=True, caps=(False, False), up=(0.0, -1.0, 0.0),
               aspect=PIPE[1] / PIPE[0], mat=_m("piping")).weld(1e-9)
    return m if m.volume() > 0 else m.flip()


def panel_mesh():
    hw, z0, z1 = panel_box()
    m = S.rounded_panel(S.rrect(2 * hw, z1 - z0, 0.0035, 4), PANEL_T + SINK, r_edge=0.0016, n=3, rings=2, r_back=0.0012,
                        mat=_m("panel"))
    return on_front(m, 0.0, 0.5 * (z0 + z1), Y_BAFFLE + SINK)


def _disc(r, n=10):
    """A flat disc about +Z (a printed dot)."""
    a = 2.0 * math.pi * np.arange(n) / n
    V = np.vstack([[0.0, 0.0, 0.0], np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1)])
    T = np.stack([np.zeros(n, np.int64), 1 + np.arange(n), 1 + (np.arange(n) + 1) % n], 1)
    return S.Mesh(V, T=T, mat=_m("print"))


def print_mesh(n_knobs):
    """The panel's print: a border line inside its edge and a scale of dots round each knob (graphic layers 0.2 mm off
    the face)."""
    hw, z0, z1 = panel_box()
    y = Y_BAFFLE - PANEL_T - 0.0002
    zc = 0.5 * (z0 + z1)
    ring = S.rrect(2 * hw - 0.009, z1 - z0 - 0.009, 0.0025, 4)
    border = S.ribbon(np.column_stack([ring[:, 0], np.full(len(ring), y), ring[:, 1] + zc]), 0.0011, (0.0, -1.0, 0.0),
                      closed=True, mat=_m("print"))
    r, rd, count, arc = DOTS
    dots = []
    for x in knob_xs(n_knobs):
        for k in range(count):
            a = math.radians(90.0 + 0.5 * arc - arc * k / (count - 1))       # 7 o'clock round to 5 o'clock, clockwise
            big = 1.35 if k in (0, count - 1, count // 2) else 1.0
            dots.append(on_front(_disc(rd * big), x + r * math.cos(a), zc + r * math.sin(a), y))
    return S.merge([border] + dots)


def knob_mesh():
    """One skirted knob about +Z (out of the panel): a bevelled skirt, a knurled cap, a flat top with a rolled rim."""
    rs, rc, h = KNOB["skirt"], KNOB["cap"], KNOB["h"]
    pts = np.array([(0.0, -0.001), (rs, -0.001), (rs, 0.0026), (rs - 0.0016, 0.0042), (rc + 0.0005, 0.0048),
                    (rc, 0.0062), (rc - 0.0002, h - 0.0016), (rc - 0.0016, h), (0.0, h + 0.0002)])
    m = S.lathe(S.fillet(pts, [0.0, 0.0, 0.0008, 0.0008, 0.0006, 0.0006, 0.0012, 0.0005, 0.0], 3), seg=72,
                mat=_m("knobs"))
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    side = (m.V[:, 2] > 0.0064) & (m.V[:, 2] < h - 0.0018) & (r > 1e-6)
    th = np.arctan2(m.V[:, 1], m.V[:, 0])
    k = 1.0 + KNOB["knurl"] * 0.5 * (1.0 + np.cos(KNOB["ribs"] * th))
    m.V[side, :2] *= k[side, None]
    return m


def pointer_mesh(deg):
    """A knob's pointer line on its top (a graphic layer), `deg` clockwise from 12 o'clock."""
    a = math.radians(90.0 - deg)
    t = np.linspace(0.0018, 0.0079, 4)
    path = np.stack([t * math.cos(a), t * math.sin(a), np.full(4, KNOB["h"] + 0.0004)], 1)
    return S.ribbon(path, 0.0015, (0.0, 0.0, 1.0), mat=_m("piping"))


def knobs_meshes(n_knobs):
    """(knobs, pointers) on the panel's face."""
    y = Y_BAFFLE - PANEL_T
    k = knob_mesh()
    knobs, ptrs = [], []
    for i, x in enumerate(knob_xs(n_knobs)):
        knobs.append(on_front(k, x, row_z(), y))
        ptrs.append(on_front(pointer_mesh(SETTINGS[i]), x, row_z(), y))
    return S.merge(knobs), S.merge(ptrs)


def jack_mesh():
    """The input jack about +Z: a washer, a round nut, the threaded sleeve rolled over at its mouth into the bore (the
    bore's faces rubber: dark)."""
    pts = np.array([(0.0, -0.0005), (0.0080, -0.0005), (0.0080, 0.0010), (0.0068, 0.0012), (0.0068, 0.0034),
                    (0.0049, 0.0036), (0.0047, JACK_MOUTH), (0.0034, JACK_MOUTH - 0.0006), (0.0034, 0.0002),
                    (0.0, 0.0002)])
    m = S.lathe(S.fillet(pts, [0.0, 0.0, 0.0004, 0.0003, 0.0004, 0.0003, 0.0006, 0.0003, 0.0, 0.0], 3), seg=32,
                mat=_m("metal"))
    m.assign(_m("rubber"), lambda c, nrm: np.hypot(c[:, 0], c[:, 1]) < 0.0036)
    return m


def jack_point():
    """The jack's mouth in the prop's frame: the middle of its rim's front-most ring."""
    return (JACK_X, Y_BAFFLE - PANEL_T - float(jack_mesh().V[:, 2].max()), row_z())


def bezel_mesh():
    """The lamp's chrome bezel about +Z: a ring with a flange round the jewel, its section a closed loop (welded)."""
    pts = np.array([(0.0055, -0.0004), (0.0086, -0.0004), (0.0086, 0.0012), (0.0072, 0.0042), (0.0055, 0.0040),
                    (0.0055, -0.0004)])
    m = S.lathe(S.fillet(pts, [0.0, 0.0, 0.0006, 0.0008, 0.0006, 0.0], 3), seg=32, mat=_m("metal")).weld(1e-9)
    return m if m.volume() > 0 else m.flip()


def jewel_mesh():
    """The faceted jewel about +Z (flat shaded by the builder): ten facets round, three rows of them."""
    pts = np.array([(0.0, 0.0010), (0.0058, 0.0010), (0.0057, 0.0052), (0.0040, 0.0084), (0.0, 0.0096)])
    return S.lathe(pts, seg=10, mat=_m("lamp"), phase=math.pi / 10)


def switch_mesh():
    """A rocker switch about +Z: its bezel and the rocker tipped 11 degrees (on)."""
    bez = S.rounded_frame(S.rrect(0.0135, 0.0205, 0.0025, 3), S.rrect(0.0095, 0.0165, 0.0015, 3), 0.0028,
                          r_out=0.0010, r_in=0.0006, floor=0.0026, mat=_m("rubber"))
    rock = S.rounded_panel(S.rrect(0.0088, 0.0158, 0.0014, 3), 0.0050, r_edge=0.0012, n=2, dome=0.0004,
                           mat=_m("rubber")).rot(-11.0, 0.0, 0.0).moved((0.0, 0.0, 0.0008))
    return S.merge([bez, rock])


def handle_meshes():
    """(strap, caps with their screws) on the cabinet's top: the strap's ends tucked in the caps, arching up between."""
    top = FOOT_H + H
    cx = HANDLE["x"]
    cw, cd, ch = HANDLE["cap"]
    sw, st = HANDLE["strap"]
    x = np.linspace(-cx + 0.006, cx - 0.006, 41)
    u = x / (cx - 0.006)
    z = top + 0.0045 + HANDLE["arch"] * (1.0 - u * u) ** 0.85
    strap = S.sweep(np.stack([x, np.zeros_like(x), z], 1), S.rrect(sw, st, 0.0022, 3), up=(0.0, 0.0, 1.0),
                    mat=_m("handle"))
    strap = strap if strap.volume() > 0 else strap.flip()
    cap = S.rounded_panel(S.rrect(cw, cd, 0.009, 4), ch, r_edge=0.0035, n=3, dome=0.0015, rings=3, mat=_m("metal"))
    screw = S.lathe(S.fillet(np.array([(0.0, -0.001), (0.0027, -0.001), (0.0027, 0.0006), (0.0, 0.0016)]),
                             [0.0, 0.0, 0.0008, 0.0], 3), seg=16, mat=_m("metal"))
    parts = []
    for s in (-1.0, 1.0):
        parts.append(cap.moved((s * cx, 0.0, top - 0.0005)))
        parts.append(screw.moved((s * (cx + 0.0095), 0.0, top + ch + 0.0003)))
    return strap, S.merge(parts)


def feet_mesh():
    pts = np.array([(0.0, 0.0), (FOOT_R - 0.003, 0.0), (FOOT_R, 0.0035), (FOOT_R - 0.0012, FOOT_H + 0.001),
                    (0.0, FOOT_H + 0.001)])
    foot = S.lathe(S.fillet(pts, [0.0, 0.0025, 0.002, 0.0008, 0.0], 3), seg=28, mat=_m("rubber"))
    return S.merge([foot.moved((sx * (W / 2 - 0.036), sy * (D / 2 - 0.032), 0.0)) for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)])


def corners_mesh():
    """Eight corner protectors: rounded blocks over the cabinet's corners, CORNER[2] proud of its faces."""
    s, r, out = CORNER
    c = s / 2 - out
    zc = FOOT_H + H / 2
    return S.merge([S.rounded_box((s, s, s), (sx * (W / 2 - c), sy * (D / 2 - c), zc + sz * (H / 2 - c)), r=r, k=2,
                                  mat=_m("corners"))
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def hardware_mesh(n_knobs, corners=True):
    """The jack, the lamp's bezel, the switch, the handle's caps and screws, the feet and (corners) the protectors."""
    y = Y_BAFFLE - PANEL_T
    parts = [on_front(jack_mesh(), JACK_X, row_z(), y), on_front(bezel_mesh(), LAMP_X, row_z(), y),
             on_front(switch_mesh(), SWITCH_X, row_z(), y), handle_meshes()[1], feet_mesh()]
    if corners:
        parts.append(corners_mesh())
    return S.merge(parts)


def jewel_placed():
    return on_front(jewel_mesh(), LAMP_X, row_z(), Y_BAFFLE - PANEL_T)


def part_meshes(O):
    """{part: Mesh} of every object (`<name>_<part>`) in the prop's frame."""
    n = O["knob_count"]
    knobs, ptrs = knobs_meshes(n)
    return {"cabinet": cabinet_mesh(), "grille": grille_mesh(), "piping": piping_mesh(), "panel": panel_mesh(),
            "print": S.merge([print_mesh(n), ptrs]), "knobs": knobs, "hardware": hardware_mesh(n, O["corners"]),
            "jewel": jewel_placed(), "handle": handle_meshes()[0]}


# --------------------------------------------------------------------------------------------------------------- card
def handle_top():
    """The highest point of the handle: the strap's arch."""
    return FOOT_H + H + 0.0045 + HANDLE["arch"] + 0.5 * HANDLE["strap"][1]


def collider_box():
    """(centre, size) of the collider box: the cabinet with its feet and handle."""
    top = handle_top()
    return (0.0, 0.0, top / 2), (W, D, top)


def card(name, roles=None):
    """The prop card (docs/design.md: Prop card); the props stage measures its bounds and size."""
    jack = {"name": "jack", "point": [round(v, 5) for v in jack_point()], "dir": [0.0, -1.0, 0.0]}
    looks = [{"name": "front", "point": [0.0, Y_FRONT, round(FOOT_H + H / 2, 5)]},
             {"name": "panel", "point": [0.0, round(Y_BAFFLE - PANEL_T, 5), round(row_z(), 5)]}]
    return {"origin": "floor_center", "front": "-Y", "slots": dict(roles or {}), "use": {"anchor": [jack], "look": looks},
            "colliders": [{"type": "box", "object": f"{name}_col", "rnd": 0.012}], "form_max": FORM_MAX,
            "objects": [f"{name}_{p}" for p in PARTS]}
