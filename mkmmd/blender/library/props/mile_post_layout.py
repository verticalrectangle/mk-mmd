"""Layout of the mile post (pure Python + numpy, no bpy: importable from tests): its numbers, meshes, colour roles,
options and card. The numerals are mkmmd.core.numerals' (drawn in code, no font).

Frame (metres): the post stands on the floor at the origin (origin floor_center), its number facing -Y, +Z up.

`style = "railroad"` (the default): a 19th-century railroad mile post. A 6 x 6 in timber (0.152 m, rough-sawn full size)
standing 1.30 m to its shoulder, painted white; its four arrises chamfered 1 1/8 in, a hipped top rising 55 mm to a point,
the season checks a boxed-heart post opens down its faces as it weathers (one per face, 6 mm wide, 5 mm deep, tapering
out at both ends). On its front face, 35 mm under the shoulder, a cast-iron number plate: a plate with a rolled edge and a
raised bead round it, held by four domed bolts in its corners, the numerals cast in relief (a condensed gothic 115 mm
tall, 0.61 of that wide, the stroke 0.15: a rounded section 3.5 mm proud). Proportions: the Southern Pacific's common
standard mile post (a 6 x 6 in post; numerals half as wide as they are tall, the stroke 0.16 of their height) and
photographs of old wooden and cast posts.

`style = "interstate"`: a US reference location sign (MUTCD D10, the freeway size, without the word MILE): a panel
12 in wide with 1.5 in corners and a white border (0.5 in, 0.375 in in from the edge), its numerals 10 in tall in the
proportions of FHWA Series E (core.numerals), stacked one per row 3 in apart with 3.5 in over the first and under the
last, bolted 1.75 in from its top and bottom edges to a galvanised U-channel post (3 lb/ft: 3.5 in across the flanges,
1.75 in deep, a 1.625 in bearing face, holes every inch) whose top is 1 in under the panel's; the panel's bottom 4 ft over
the ground (the MUTCD's least).

Options (`slots` of the [[prop]]): `style`; `number` (a whole number 0 .. 999 or a string of 1 to 3 digits; "" leaves the
plate blank, the default); `digits` (1 .. 3: how many numerals the plate is sized for; default the number's own count,
2 for a blank plate). The plate grows with `digits`: the railroad plate is wider, the interstate panel taller.

Colour roles (a palette slot, `paper` / `ink`, a `#hex` or a mix `a:b:t` of two such; cafe_colors.Colors.hex):
railroad: `post` (paper: the white paint), `plate` (ink: the cast iron), `numerals` (paper), `wood` (muted:gold:0.25:
the grey weathered timber where the paint is gone), `grime` (ink:gold:0.3: the dirt splashed up the foot), `foot` (ink:
the tarred foot); interstate: `post` (muted:paper:0.35: the galvanised steel and the panel's back), `plate` (pine: the
green sheeting, as the highway set's signs), `numerals` (paper), `border` (paper).

Card: `use.surface` `plate` (the numeral area on the plate's face: `[[text]] on = "<name>:plate"` writes there, reading
from the front), `use.look` `plate` (its centre) and `top` (the post's top)."""
import math
import re

import numpy as np

from ....core import numerals as NU
from ....core import shell as S
from .cafe_colors import Colors

WHO = "mile_post"
STYLES = ("railroad", "interstate")
MAX_DIGITS = 3
BLANK_DIGITS = 2                    # a blank plate is sized for two numerals unless `digits` says otherwise
INCH = 0.0254

# ------------------------------------------------------------------------------------------------ railroad: the post
POST_W = 0.152                      # 6 in square, rough-sawn full size
POST_H = 1.30                       # the shoulder: where the faces end and the hipped top begins
CHAMFER = 0.0286                    # the four arrises chamfered 1 1/8 in (on each face)
EASE = 0.003                        # every arris of the chamfered section eased (paint and weather soften it)
CAP_RISE = 0.055                    # the hipped top rises this much from the shoulder to its point
SHOULDER_R = 0.006                  # the shoulder's edge rounded
RING_DZ = 0.05                      # ring spacing up the faces (finer where a check tapers)
#          face  across  from   to     width  depth: season checks (face: 0 front -Y, 1 right +X, 2 back +Y, 3 left -X;
#                (m from the face's middle, toward the next face counter-clockwise seen from above))
CHECKS = ((0, 0.011, 0.06, 1.06, 0.006, 0.005),
          (1, -0.007, 0.22, 1.24, 0.006, 0.005),
          (2, 0.016, 0.03, 1.12, 0.006, 0.005),
          (3, -0.013, 0.30, 1.22, 0.006, 0.005))
CHECK_TAPER = 0.16                  # a check closes over this length at either end
FOOT_Z = 0.18                       # the tarred foot reaches this high (material; about a sixth of what stands out)
GRIME_Z = 0.42                      # dirt splashed up to about here (material)

# ------------------------------------------------------------------------------------------------ railroad: the plate
DIGIT_H = 0.115                     # the cast numerals' height (4.5 in)
RR_CONDENSE, RR_STROKE = 0.70, 0.15     # a condensed gothic: 0.62 of the height wide, the stroke 0.15 of it
RR_GAP = 0.15                       # between two numerals, heights
PLATE_PAD = (0.022, 0.025)          # plate round the numerals: at the sides, over and under
PLATE_T = 0.008                     # cast iron 5/16 in
PLATE_R = 0.012                     # the plate's corners
PLATE_EDGE = 0.0025                 # its face's edge rolled
PLATE_BACK = 0.0015                 # its back's edge rolled
PLATE_SINK = 0.001                  # its back sunk into the post's face (no sliver of gap)
PLATE_UNDER = 0.035                 # its top edge this far under the shoulder
BEAD_IN, BEAD_R = 0.0075, 0.0026    # the raised bead round the face: its line in from the edge, its radius
RELIEF_H, RELIEF_SINK = 0.0035, 0.0008  # the numerals stand this far out of the face, their feet sunk in it
BOLT_IN, BOLT_R, BOLT_H = 0.0165, 0.0055, 0.0040    # the domed bolts: in from both edges, radius, height

# ------------------------------------------------------------------------------------------------ interstate
PANEL_W = 12 * INCH
NUMERAL_H = 10 * INCH               # freeway size
PANEL_PAD = 3.5 * INCH              # over the first numeral and under the last
COLUMN_GAP = 3 * INCH               # between two numerals of the column
PANEL_R = 1.5 * INCH                # the panel's corners
MARGIN, BORDER = 0.375 * INCH, 0.5 * INCH   # the white border: in from the edge, its width
PANEL_T = 0.003                     # aluminium sheet (0.080 in) with its sheeting
PANEL_EDGE = 0.0012                 # both edges rolled
PANEL_LOW = 48 * INCH               # the panel's bottom over the ground
LIFT = 0.0003                       # the sheeting's legend over the panel's face
CH_B, CH_A, CH_C, CH_D, CH_T = 3.5 * INCH, 1.75 * INCH, 1.625 * INCH, 0.718 * INCH, 0.16 * INCH   # U-channel 3 lb/ft
CH_BEND = 0.006                     # the channel's bends (radius of the sheet's middle line)
CH_HOLE, CH_PITCH = 0.375 * INCH, 1.0 * INCH  # punched holes down the web (material)
POST_TOP_UNDER = 1 * INCH           # the post's top under the panel's top
SIGN_BOLT_IN, SIGN_BOLT_R, SIGN_BOLT_H = 1.75 * INCH, 0.0095, 0.0035   # button-head bolts on the face

ROLES = {"railroad": {"post": "paper", "plate": "ink", "numerals": "paper", "wood": "muted:gold:0.25",
                      "grime": "ink:gold:0.3", "foot": "ink"},
         "interstate": {"post": "muted:paper:0.35", "plate": "pine", "numerals": "paper", "border": "paper"}}
# material index -> material of each style's meshes (several roles may feed one material)
MATS = {"railroad": ("post", "plate", "numerals"),
        "interstate": ("post", "back", "face", "border", "numerals")}
GRAPHIC = ("border", "numerals")    # interstate parts that are flat sheeting: graphic layers (shell.exempt)
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ================================================================================================= options
def _number(v):
    """The `number` option as a digit string ("" blank), checked."""
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise ValueError(f"{WHO}: number is a whole number 0 .. 999 or a string of 1 to {MAX_DIGITS} digits "
                         f"(\"\" leaves the plate blank), got {v!r}")
    if isinstance(v, int):
        if not 0 <= v < 10 ** MAX_DIGITS:
            raise ValueError(f"{WHO}: number {v} is out of range: a mile post carries 0 .. {10 ** MAX_DIGITS - 1}")
        return str(v)
    if v == "":
        return v
    if len(v) > MAX_DIGITS or not all(c in NU.DIGITS for c in v):
        raise ValueError(f"{WHO}: number {v!r} is not a string of 1 to {MAX_DIGITS} digits 0-9 (\"\" leaves the "
                         f"plate blank)")
    return v


def options(slots=None):
    """{"style", "number" (a digit string, "" blank), "digits" (the plate's size in numerals), "roles" {role: spec}} from
    the `slots` of the [[prop]] (the palette's slots and keys this prop does not know are ignored), checked: a bad value
    is a ValueError that says what to give instead."""
    slots = dict(slots or {})
    style = slots.get("style", "railroad")
    if style not in STYLES:
        raise ValueError(f"{WHO}: style is {' or '.join(repr(s) for s in STYLES)}, got {style!r}")
    number = _number(slots.get("number", ""))
    digits = slots.get("digits", len(number) or BLANK_DIGITS)
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= MAX_DIGITS:
        raise ValueError(f"{WHO}: digits is how many numerals the plate is sized for, 1 .. {MAX_DIGITS}, got {digits!r}")
    if digits < len(number):
        raise ValueError(f"{WHO}: number {number!r} has {len(number)} digits but the plate is sized for digits = {digits}")
    colors = Colors(slots)
    roles = {}
    for role, default in ROLES[style].items():
        spec = slots.get(role, default)
        try:
            ok = isinstance(spec, str) and HEX.match(colors.hex(spec)) is not None
        except (KeyError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{WHO}: colour role {role} = {spec!r}: a palette slot, 'paper', 'ink', a \"#rrggbb\" or a "
                             f"mix \"a:b:t\"")
        roles[role] = spec
    return {"style": style, "number": number, "digits": digits, "roles": roles}


# ================================================================================================= railroad: the post
def _octagon():
    """The chamfered square's corners, counter-clockwise from the front face's left end (seen from above), and the
    index of each face's first corner: front (-Y), right (+X), back (+Y), left (-X)."""
    h, c = 0.5 * POST_W, CHAMFER
    return np.array([(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c),
                     (-h, -h + c)])


def _section():
    """The post's section, counter-clockwise: the chamfered square with its eight corners eased, the faces carrying three
    points per check (its lips and its floor); and per check, the index of its floor point and the inward normal."""
    C = _octagon()
    pts, rad, floors = [], [], []
    for f in range(4):
        a, b = C[2 * f], C[2 * f + 1]
        t = (b - a) / np.linalg.norm(b - a)
        inward = np.array([-t[1], t[0]])
        mid = 0.5 * (a + b)
        pts.append(a)
        rad.append(EASE)
        for k, (face, across, _z0, _z1, w, _d) in enumerate(CHECKS):
            if face != f:
                continue
            for s in (-0.5 * w, 0.0, 0.5 * w):
                pts.append(mid + (across + s) * t)
                rad.append(0.0)
            floors.append((k, mid + across * t, inward))
        pts.append(b)
        rad.append(EASE)
    ring = S.fillet(np.array(pts), rad, 3, closed=True)
    out = []
    for k, p, inward in floors:
        out.append((k, int(np.argmin(np.linalg.norm(ring - p, axis=1))), inward))
    return ring, sorted(out)


def check_depth(k, z):
    """The depth (m) of check k at heights z: full in its middle, closing smoothly over CHECK_TAPER at either end."""
    _f, _a, z0, z1, _w, d = CHECKS[k]
    z = np.asarray(z, float)

    def ss(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    return d * ss((z - z0) / CHECK_TAPER) * ss((z1 - z) / CHECK_TAPER)


def _post_heights():
    zs = set(np.round(np.arange(0.0, POST_H - SHOULDER_R, RING_DZ), 6).tolist())
    for _f, _a, z0, z1, _w, _d in CHECKS:
        for a, b in ((z0, z0 + CHECK_TAPER), (z1 - CHECK_TAPER, z1)):
            zs.update(np.round(np.linspace(a, b, 7), 6).tolist())
    return np.array(sorted(z for z in zs if 0.0 <= z < POST_H - SHOULDER_R - 1e-4))


def _post_rings():
    """The post's rings (k, m, 3) from the floor up the faces (the checks opening and closing along them), round the eased
    shoulder and up the hipped top, and per ring point how open a check is there (0 .. 1: 1 on the floor of a check at
    its full depth)."""
    ring, floors = _section()
    h = 0.5 * POST_W
    arc = S.fillet(np.array([(h, POST_H - 0.2), (h, POST_H), (0.0, POST_H + CAP_RISE)]), [0.0, SHOULDER_R, 0.0], 4)[1:-1]
    r_end = 0.12 * h                                                  # the hips run straight in to here; a rounded point
    cone = np.linspace(arc[-1, 0], r_end, 9)[1:]
    prof = np.vstack([arc, np.column_stack([cone, POST_H + CAP_RISE * (1.0 - cone / h)])])
    rings, open_ = [], []
    for z in _post_heights():
        R, w = ring.copy(), np.zeros(len(ring))
        for k, i, inward in floors:
            d = float(check_depth(k, z))
            R[i] = R[i] + inward * d
            w[i] = d / CHECKS[k][5]
        rings.append(np.column_stack([R, np.full(len(R), z)]))
        open_.append(w)
    for r, z in prof:
        rings.append(np.column_stack([ring * (r / h), np.full(len(ring), z)]))
        open_.append(np.zeros(len(ring)))
    return np.array(rings), np.array(open_)


def post_mesh():
    """The post: ONE closed cage of the section's rings from the floor up the faces, round the eased shoulder and up
    the hipped top to its (slightly rounded) point."""
    rings, _ = _post_rings()
    tip = POST_H + CAP_RISE - rings[-1][0, 2]
    return S.cage(rings, closed=True, caps=("fan", "fan"), cap_bulge=(0.0, tip), mat=0)


def post_check():
    """Per vertex of `post_mesh()` how open a season check is there (0 .. 1; the caps' vertices 0): the post's
    `mk_check` attribute, which its material darkens the checks by."""
    rings, open_ = _post_rings()
    w = open_.ravel()
    return np.concatenate([w, np.zeros(len(post_mesh().V) - len(w))])


# ================================================================================================= the plate
def plate_size(style, digits):
    """(w, h) of the railroad plate or the interstate panel sized for `digits` numerals."""
    if style == "railroad":
        w, h = NU.extent(digits, DIGIT_H, RR_CONDENSE, RR_STROKE, RR_GAP)
        return w + 2 * PLATE_PAD[0], h + 2 * PLATE_PAD[1]
    _w, h = NU.extent(digits, NUMERAL_H, 1.0, NU.STROKE, COLUMN_GAP / NUMERAL_H, column=True)
    return PANEL_W, h + 2 * PANEL_PAD


def numeral_area(style, digits):
    """(w, h) of the box the numerals take on the plate: the card's surface."""
    if style == "railroad":
        return NU.extent(digits, DIGIT_H, RR_CONDENSE, RR_STROKE, RR_GAP)
    return NU.extent(digits, NUMERAL_H, 1.0, NU.STROKE, COLUMN_GAP / NUMERAL_H, column=True)


def plate_frame(style, digits):
    """(centre of the plate's face, its normal, its up) in the prop's frame: where the numerals stand."""
    w, h = plate_size(style, digits)
    if style == "railroad":
        y = -0.5 * POST_W + PLATE_SINK - PLATE_T
        return np.array([0.0, y, POST_H - PLATE_UNDER - 0.5 * h]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    y = -0.5 * CH_A - PANEL_T
    return np.array([0.0, y, PANEL_LOW + 0.5 * h]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])


def _on_plate(mesh, style, digits, out=0.0):
    """A mesh drawn in the plate's face frame (x right, y up, z out of the face, the origin at its centre) put on the
    plate, `out` metres off the face."""
    c, n, up = plate_frame(style, digits)
    return S.place(mesh, c + out * n, n, up)


def _bolt(r, h, mat):
    prof = S.fillet(np.array([(0.0, -0.0006), (r, -0.0006), (r, 0.25 * h), (0.55 * r, 0.92 * h), (0.0, h)]),
                    [0.0, 0.0, 0.0012, 0.004, 0.0], 4)
    return S.lathe(prof, seg=20, mat=mat)


def rr_plate_mesh(digits):
    """The cast plate on the post's face: the plate (edges rolled), the bead round its face, four domed bolts."""
    w, h = plate_size("railroad", digits)
    mat = MATS["railroad"].index("plate")
    plate = S.rounded_panel(S.rrect(w, h, PLATE_R, 5), PLATE_T, r_edge=PLATE_EDGE, n=3, r_back=PLATE_BACK, mat=mat)
    plate = plate.moved((0.0, 0.0, -PLATE_T))                    # its face at z = 0 of the face frame
    path = S.rrect(w - 2 * BEAD_IN, h - 2 * BEAD_IN, PLATE_R - BEAD_IN, 8)
    path = S.resample_loop(path, 160)
    sec = NU.section(2 * BEAD_R, BEAD_R, 0.0006, p=2.0, n=10)
    bead = S.sweep(np.column_stack([path, np.zeros(len(path))]), sec, up=(0.0, 0.0, 1.0), closed=True, mat=mat).weld()
    bead = bead if bead.volume() >= 0 else bead.flip()
    bolts = [_bolt(BOLT_R, BOLT_H, mat).moved((sx * (0.5 * w - BOLT_IN), sy * (0.5 * h - BOLT_IN), -0.0002))
             for sx in (-1, 1) for sy in (-1, 1)]
    return _on_plate(S.merge([plate, bead] + bolts), "railroad", digits)


def rr_numerals_mesh(number, digits):
    """The cast numerals standing out of the plate's face."""
    placed = NU.layout(number, DIGIT_H, RR_CONDENSE, RR_STROKE, RR_GAP)
    m = NU.relief(placed, RR_STROKE * DIGIT_H, RELIEF_H, RELIEF_SINK, mat=MATS["railroad"].index("numerals"))
    return _on_plate(m, "railroad", digits)


# ================================================================================================= interstate
def channel_section():
    """The U-channel's section (x across, y back from its bearing face at y = -CH_A / 2), counter-clockwise: its sheet's
    middle line (flange tip, flange, side, web, side, flange) with the bends rounded, thickened by CH_T. Returns the
    outline (m, 2) and the count of points along the middle line (the outline is that line's outer side, then its inner
    side back)."""
    yw, yf = -0.5 * CH_A + 0.5 * CH_T, 0.5 * CH_A - 0.5 * CH_T
    xw = 0.5 * (CH_C - CH_T)
    xs = 0.5 * CH_B - CH_D
    mid = np.array([(-0.5 * CH_B, yf), (-xs, yf), (-xw, yw), (xw, yw), (xs, yf), (0.5 * CH_B, yf)])
    mid = S.fillet(mid, [0.0, CH_BEND, CH_BEND, CH_BEND, CH_BEND, 0.0], 5)
    T = np.gradient(mid, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.column_stack([T[:, 1], -T[:, 0]])                       # to the right of the way along the line
    a, b = mid + 0.5 * CH_T * N, mid - 0.5 * CH_T * N
    out = np.vstack([a, b[::-1]])
    if S.polygon_area(out) < 0:
        out = np.vstack([b, a[::-1]])
    return out, len(mid)


def channel_mesh(top):
    """The galvanised U-channel post from the floor to `top`: its section swept up Z, the two ends closed by the strip
    between the sheet's two sides (the section is not star-shaped: no fan)."""
    sec, m = channel_section()
    n = len(sec)
    zs = np.linspace(0.0, top, max(2, int(math.ceil(top / 0.25)) + 1))
    rings = np.stack([np.column_stack([sec, np.full(n, z)]) for z in zs])
    mesh = S.cage(rings, closed=True, mat=MATS["interstate"].index("post"))
    V = mesh.V
    caps = []
    for r, look in ((0, -1.0), (len(zs) - 1, 1.0)):
        i = r * n + np.arange(m - 1)
        j = r * n + (n - 1 - np.arange(m - 1))                     # the inner side, walked back
        Q = np.stack([i, i + 1, j - 1, j], 1)
        P = V[Q]
        nz = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])[:, 2]
        caps.append(np.where((nz * look < 0)[:, None], Q[:, ::-1], Q))
    Q = np.concatenate([mesh.Q] + caps)
    return S.Mesh(V, Q, mesh.T, mat=MATS["interstate"].index("post"))


def panel_mesh(digits):
    """The sign: the aluminium panel (its face in the green sheeting, its back and edges bare) and the two button-head
    bolts through it into the post's web."""
    w, h = plate_size("interstate", digits)
    face, back = MATS["interstate"].index("face"), MATS["interstate"].index("back")
    panel = S.rounded_panel(S.rrect(w, h, PANEL_R, 8), PANEL_T, r_edge=PANEL_EDGE, n=2, r_back=PANEL_EDGE, mat=back)
    panel.assign(face, lambda c, n: (n[:, 2] > 0.7) & (c[:, 2] > 0.5 * PANEL_T))
    panel = panel.moved((0.0, 0.0, -PANEL_T))
    bolts = [_bolt(SIGN_BOLT_R, SIGN_BOLT_H, back).moved((0.0, s * (0.5 * h - SIGN_BOLT_IN), LIFT)) for s in (-1, 1)]
    return _on_plate(S.merge([panel] + bolts), "interstate", digits)


def border_mesh(digits):
    """The white border: a flat ribbon round the panel, MARGIN in from its edge (sheeting: a graphic layer)."""
    w, h = plate_size("interstate", digits)
    k = MARGIN + 0.5 * BORDER
    path = S.resample_loop(S.rrect(w - 2 * k, h - 2 * k, PANEL_R - k, 10), 240)
    m = S.ribbon(np.column_stack([path, np.full(len(path), LIFT)]), BORDER, (0.0, 0.0, 1.0), closed=True,
                 mat=MATS["interstate"].index("border"))
    return _on_plate(m, "interstate", digits)


def is_numerals_mesh(number, digits):
    """The numerals: flat ribbons with round caps, stacked one per row (sheeting: a graphic layer)."""
    placed = NU.layout(number, NUMERAL_H, 1.0, NU.STROKE, COLUMN_GAP / NUMERAL_H, column=True)
    m = NU.ribbons(placed, NU.STROKE * NUMERAL_H, LIFT * 1.5, mat=MATS["interstate"].index("numerals"))
    return _on_plate(m, "interstate", digits)


def post_top(style, digits):
    if style == "railroad":
        return POST_H + CAP_RISE
    return PANEL_LOW + plate_size("interstate", digits)[1] - POST_TOP_UNDER


# ================================================================================================= parts and card
def parts(opts):
    """{part: Mesh} in the prop's frame for checked `options`: railroad post, plate (and numerals); interstate post,
    panel, border (and numerals). A blank plate has no numerals."""
    style, number, digits = opts["style"], opts["number"], opts["digits"]
    if style == "railroad":
        out = {"post": post_mesh(), "plate": rr_plate_mesh(digits)}
        if number:
            out["numerals"] = rr_numerals_mesh(number, digits)
        return out
    out = {"post": channel_mesh(post_top(style, digits)), "panel": panel_mesh(digits), "border": border_mesh(digits)}
    if number:
        out["numerals"] = is_numerals_mesh(number, digits)
    return out


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, opts, size=None):
    """The prop card (docs/design.md: Prop card): `use.surface` plate (the numeral area on the plate's face), `use.look`
    plate and top. `size` [x, y, z] the meshes' extent (measured from `parts` when not given)."""
    style, digits = opts["style"], opts["digits"]
    c, n, up = plate_frame(style, digits)
    if size is None:
        V = np.concatenate([m.V for m in parts(opts).values()])
        size = V.max(0) - V.min(0)
    keys = ["post", "plate"] if style == "railroad" else ["post", "panel", "border"]
    if opts["number"]:
        keys.append("numerals")
    return {"size": _r(size, 4), "origin": "floor_center", "front": "-Y", "slots": dict(opts["roles"]),
            "use": {"surface": [{"name": "plate", "center": _r(c), "normal": _r(n), "up": _r(up),
                                 "size": _r(numeral_area(style, digits))}],
                    "look": [{"name": "plate", "point": _r(c)}, {"name": "top", "point": [0.0, 0.0, round(post_top(style, digits), 5)]}]},
            "objects": [f"{name}_{k}" for k in keys]}
