"""Layout of the record treadmill (pure Python + numpy, no bpy: importable from tests): every number the parts agree on,
the colour roles, the meshes, the label's print, the spin and the prop card.

Frame (metres; everything in the card is in it): ORIGIN on the floor under the spindle, Z up, the front toward -Y (card
`front` = "-Y"), X to the right seen from the front. A giant turntable: a 1.68 m platter (an SL-1200's 332 mm platter is
1.10 times an LP, so its strobe bevel shows round the record) on a low rounded-square plinth on four rubber feet, and a
1.60 m record on the platter: an LP scaled 5.3 times (the recorded area from 292 to 120 mm of its 302 mm, a 100 mm label),
the grooved playing surface flat at Z_TOP from the lead-in (LEADIN_R) to the label area, a raised bead at the rim and a
raised label area round the centre hole, the spindle through the hole.

The platter turns about the spindle CLOCKWISE seen from above, as a record does: at the run spot RUN (on the -X side,
inside the grooved band) the record's top moves toward +Y, so a character standing there facing -Y has her feet carried
backward and runs in place on it (`use.feet` `run`, its floor the playing surface). The root property `rpm` (33.3) is a
rate the keys stage integrates into `turns` (docs/design.md: Keys); the platter's driver turns it by -2 pi turns about Z
(EXPR["spin"]). Unkeyed, the builder's `turns` runs at the starting rpm from clip second 0, linear both ways.

The label is an original print, no lettering: a cream centre ringed round the spindle, a sunburst, three stylised beets
with their leaves in it, a hairline ring and a band at the edge (`label_image`, laid on the label paper by its uv); its
`use.surface` `label` (the largest panel above the spindle inside the print) rides the platter.

Colour roles (`slots` of the prop: a palette slot, a hex, a blend "gold:2,rose:1" or another role; `ink` and `paper` are
the palette's darkest and lightest, as the turntable's; no role is named after a palette slot, whose values the slots
carry): plinth (its sides), deck (its top), feet, plate (the platter's bright lips and the spindle), rim (the dark strobe
bevel), dots (the strobe dots), vinyl, label (the print's ground), rays (the sunburst), print (the rings and the beets),
leaf (the beets' leaves)."""
import math

import numpy as np

from ....core import shell as S
from . import convertible_palette as CP
from . import turntable_layout as TL

WHO = "record_treadmill"

# ===================================================================================================================
# colour roles
# ===================================================================================================================
ROLES = {
    "plinth": "foam",               # the plinth's sides: a candy lacquer
    "deck": "paper:2,muted:1",      # the plinth's top: brushed aluminium
    "feet": "ink",                  # rubber feet
    "plate": "paper:3,subtle:1",    # bright metal: the platter's lips, the spindle
    "rim": "ink",                   # the platter's dark bevel
    "dots": "plate",                # the strobe dots on it
    "vinyl": "ink",                 # the record (a slot or hex for coloured vinyl)
    "label": "gold",                # the label's ground
    "rays": "label:1,paper:1",      # the sunburst
    "print": "love",                # the rings and the beets
    "leaf": "foam",                 # the beets' leaves
}
# material index of a role in the meshes (`Mesh.Qm / Tm`); the builder turns them into Blender materials
MATS = {name: i for i, name in enumerate(("plinth", "deck", "feet", "plate", "rim", "vinyl", "label"))}


class Pal(TL.Pal):
    """Colour roles over the palette, as the turntable's (`ink` and `paper` slots, blends, a role naming roles), for
    any table of roles: `Pal(slots, ROLES, "record_treadmill")`."""

    def __init__(self, slots=None, roles=None, who=WHO):
        CP.Pal.__init__(self, slots)
        slots = dict(slots or {})
        self.who = who
        self.specs = {r: str(slots.get(r, d)) for r, d in (ROLES if roles is None else roles).items()}

    def lin(self, spec, depth=0):
        """Linear RGB of a role, a slot (`ink` and `paper` too), a hex or a 'slot:weight,...' blend of them."""
        if depth > 8:
            raise ValueError(f"{self.who}: the colour roles refer to each other in a loop ({spec!r})")
        if spec in self.specs:
            return self.lin(self.specs[spec], depth + 1)
        out, tot = [0.0, 0.0, 0.0], 0.0
        for part in spec.split(","):
            nm, w = CP.parse(part)
            c = self.lin(nm, depth + 1) if nm in self.specs else self.slot_lin(nm)
            out = [o + w * x for o, x in zip(out, c)]
            tot += w
        if tot <= 0.0:
            raise ValueError(f"{self.who}: a colour blend needs a positive weight ({spec!r})")
        return tuple(o / tot for o in out)

    def checked(self):
        """{role: linear RGB} of every role; a role that is no colour raises ValueError naming it and the fix."""
        out = {}
        for role, spec in self.specs.items():
            try:
                out[role] = self.lin(role)
            except (KeyError, ValueError) as e:
                raise ValueError(f"{self.who}: colour role {role} = {spec!r} is not a colour: give a palette slot "
                                 f"({', '.join(sorted(self.palette))}, ink, paper), a '#rrggbb' hex, a blend "
                                 f"'slot:2,slot:1' or another role ({e})") from None
        return out


def srgb(lin):
    """Display (sRGB) values 0..1 of a linear RGB colour."""
    c = np.clip(np.asarray(lin, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# ===================================================================================================================
# the plinth (z up from the floor)
# ===================================================================================================================
PLINTH_W, PLINTH_D, PLINTH_P = 1.96, 1.96, 4.5      # its plan: a superellipse (a rounded square)
FOOT_H = 0.020                                      # the plinth stands this high on its feet
Z_PLINTH = 0.078                                    # its top face: the deck the platter turns over
PLINTH_EDGE = (0.024, 0.014)                        # the rolled top and bottom edges
FOOT_R = 0.075
FEET = ((-0.64, -0.64), (0.64, -0.64), (-0.64, 0.64), (0.64, 0.64))

# ===================================================================================================================
# the platter (its frame: the spindle's axis, z from the deck) and the record on it
# ===================================================================================================================
PLATTER_R = 0.842                                   # 1.68 m across
PLATTER_TOP = 0.070                                 # the face the record lies on, over the deck
TOP_R = 0.800                                       # that face's edge, where the strobe bevel starts
BEVEL = (0.032, 0.064)                              # the bevel's foot and top (platter z); 47 degrees
DOT_ROWS = ((0.0385, 184), (0.0455, 180), (0.0525, 176), (0.0595, 172))   # (platter z, dots) of the strobe rows
DOT_R = 0.0045                                      # a strobe dot's radius
SPINDLE_R = 0.018

REC_R = 0.800                                       # 1.60 m across
HOLE_R = 0.0195
REC_T = 0.012                                       # the playing surface over the platter's face
BEAD_T = 0.0135                                     # the rim's bead
LABEL_AREA_R = 0.282                                # the raised label area
LABEL_AREA_T = 0.0135
LEADIN_R = 0.780                                    # the flat playing surface ends here, at the bead
GROOVE_OUT = 0.772                                  # the first groove (the recorded band: 292 mm of 302)
GROOVE_IN = 0.320                                   # the last groove (120 mm)
LABEL_R = 0.265                                     # 100 mm of 302
LABEL_T = 0.0003                                    # the paper
TRACK_GAPS = (0.676, 0.578, 0.488, 0.398)           # the smooth bands between tracks

Z_REC0 = Z_PLINTH + PLATTER_TOP                     # the record's underside (world)
Z_TOP = Z_REC0 + REC_T                              # 0.160: the playing surface she runs on
Z_LABEL = Z_REC0 + LABEL_AREA_T + LABEL_T           # the label's paper
SPINDLE_TOP = Z_REC0 + LABEL_AREA_T + 0.055         # the spindle's rounded tip

# the label's text panel (x along +X, y along +Y, read from the front): the largest rectangle above the spindle with
# LABEL_MARGIN to spare inside the print and SPINDLE_CLEAR from the spindle's axis
LABEL_MARGIN = 0.010
SPINDLE_CLEAR = 0.040
LABEL_PANEL = {"w": 0.340, "y0": SPINDLE_CLEAR, "y1": 0.190}

# ===================================================================================================================
# where she runs
# ===================================================================================================================
RUN = (-0.45, 0.0)                                  # the spot: inside the grooved band, on the -X side
STRIDE = 0.10                                       # the ankles either side of it (L toward +X: her left)

SEG = 256                                           # segments round the big lathes


def surface_speed(rpm, p=RUN):
    """(vx, vy) m/s of the record's top at point p (x, y) at `rpm`: clockwise seen from above for rpm > 0."""
    w = -2.0 * math.pi * float(rpm) / 60.0          # angular speed about +Z
    return (-w * p[1], w * p[0])


# ===================================================================================================================
# the spin (a Blender simple expression of the root's `turns`; tests evaluate it against `spin`)
# ===================================================================================================================
EXPR = {"spin": "-6.2831853*turns"}


def spin(turns):
    """The platter's rotation about Z (radians) after `turns` revolutions: clockwise seen from above."""
    return -2.0 * math.pi * float(turns)


PARAMS = {                          # root properties for [[key]]: (default, soft range, description)
    "rpm": (33.3, (-80.0, 80.0), "platter speed, revolutions a minute (33.3, 45; 0 stops it, negative turns it back)"),
}


def options(slots):
    """The starting values of the root properties from the prop's `slots` (`rpm`), checked."""
    slots = dict(slots or {})
    out = {}
    for k, (d, (lo, hi), _doc) in PARAMS.items():
        v = slots.get(k, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{WHO}: {k} must be a number (revolutions a minute, e.g. 33.3), got {v!r}")
        if not lo <= v <= hi:
            raise ValueError(f"{WHO}: {k} is {lo:g} .. {hi:g} revolutions a minute, got {v:g}")
        out[k] = float(v)
    return out


# ===================================================================================================================
# meshes (numpy; the builder makes them objects)
# ===================================================================================================================
def _lathe(profile, mat, seg=SEG, fillets=None, n=3, closed=False):
    """A lathe about +Z of a (r, z) profile (filleted first when `fillets` is given). closed: the profile is a loop (an
    annulus), joined back to its first point and welded."""
    P = np.asarray(profile, float)
    if fillets is not None:
        P = S.fillet(P, fillets, n, closed=closed)
    if closed:
        P = np.vstack([P, P[:1]])
    m = S.lathe(P, seg=seg, mat=mat)
    return m.weld(1e-9) if closed else m


def plinth_mesh():
    """The plinth: a rounded square slab from FOOT_H up to the deck, both edges rolled; its flat top is `deck`."""
    m = S.rounded_panel(S.superellipse(PLINTH_W, PLINTH_D, PLINTH_P, 192), Z_PLINTH - FOOT_H, r_edge=PLINTH_EDGE[0],
                        n=4, r_back=PLINTH_EDGE[1], mat=MATS["plinth"]).moved((0.0, 0.0, FOOT_H))
    m.assign(MATS["deck"], lambda c, nrm: (c[:, 2] > Z_PLINTH - 1e-6) & (nrm[:, 2] > 0.999))
    return m


def feet_mesh():
    """Four rubber feet under the plinth's corners, their tops tucked into it."""
    prof = [(0.0, 0.0), (FOOT_R - 0.010, 0.0), (FOOT_R, 0.010), (FOOT_R - 0.006, FOOT_H + 0.004), (0.0, FOOT_H + 0.004)]
    one = _lathe(prof, MATS["feet"], seg=48, fillets=[0.0, 0.006, 0.004, 0.0, 0.0])
    return S.merge([one.moved((x, y, 0.0)) for x, y in FEET])


def platter_mesh():
    """The platter in its frame (origin on the axis at the deck): a bright lower lip, the dark bevel the strobe dots are
    drawn on (`rim`), a bright top lip and the face the record lies on; the spindle."""
    b0, b1 = BEVEL
    prof = [(0.0, 0.004), (PLATTER_R - 0.014, 0.004), (PLATTER_R, 0.014), (PLATTER_R, 0.026), (PLATTER_R - 0.006, b0),
            (TOP_R + 0.006, b1), (TOP_R + 0.003, PLATTER_TOP - 0.002), (TOP_R, PLATTER_TOP), (0.0, PLATTER_TOP)]
    fil = [0.0, 0.006, 0.004, 0.003, 0.003, 0.003, 0.002, 0.001, 0.0]
    p = _lathe(prof, MATS["plate"], fillets=fil)
    p.assign(MATS["rim"], lambda c, nrm: (c[:, 2] > b0 - 0.002) & (c[:, 2] < b1 + 0.002)
             & (np.hypot(c[:, 0], c[:, 1]) > TOP_R + 0.002))
    top = SPINDLE_TOP - Z_PLINTH
    sp = [(0.0, PLATTER_TOP - 0.010), (SPINDLE_R, PLATTER_TOP - 0.010), (SPINDLE_R, top - 0.010), (0.0, top)]
    spindle = _lathe(sp, MATS["plate"], seg=48, fillets=[0.0, 0.0, 0.012, 0.0])
    return S.merge([p, spindle])


def record_mesh():
    """The record in the platter's frame: the flat underside, a rounded bead at the rim, the grooved playing surface
    flat at REC_T over the platter's face from LEADIN_R in to the label area, the raised label area, the centre hole
    (all `vinyl`: the grooves and their sheen are drawn by its material)."""
    z0 = PLATTER_TOP
    loop = [(HOLE_R, z0), (REC_R - 0.004, z0), (REC_R, z0 + 0.005), (REC_R - 0.002, z0 + BEAD_T),
            (REC_R - 0.012, z0 + BEAD_T), (LEADIN_R, z0 + REC_T), (LABEL_AREA_R + 0.008, z0 + REC_T),
            (LABEL_AREA_R, z0 + LABEL_AREA_T), (HOLE_R + 0.004, z0 + LABEL_AREA_T), (HOLE_R, z0 + LABEL_AREA_T - 0.004)]
    fil = [0.001, 0.002, 0.002, 0.0015, 0.004, 0.006, 0.004, 0.003, 0.0015, 0.001]
    return _lathe(loop, MATS["vinyl"], fillets=fil, n=3, closed=True)


def label_mesh():
    """The label's paper in the platter's frame: a flat ring on the label area (a graphic layer), its uv the print's
    square (u along +X, v along +Y, the print's centre on the spindle)."""
    z = PLATTER_TOP + LABEL_AREA_T + LABEL_T
    r0 = HOLE_R + 0.002
    m = _lathe([(LABEL_R, z), (0.5 * (LABEL_R + r0), z), (r0, z)], MATS["label"])
    m.UV = m.V[:, :2] / (2.0 * LABEL_R) + 0.5
    return m


GRAPHIC = ("label",)                                # graphic layers (exempt from `form`)
SPIN = ("record", "label")                          # ride the platter


def parts():
    """{key: Part(mesh, origin, rot)}: the plinth and feet in the prop's frame; the platter at the deck on the spindle's
    axis, and the record and label riding it (their meshes in the platter's frame, origin 0 under it)."""
    deck = (0.0, 0.0, Z_PLINTH)
    zero = (0.0, 0.0, 0.0)
    return {"plinth": S.Part(plinth_mesh(), zero, zero), "feet": S.Part(feet_mesh(), zero, zero),
            "platter": S.Part(platter_mesh(), deck, zero), "record": S.Part(record_mesh(), zero, zero),
            "label": S.Part(label_mesh(), zero, zero)}


# ===================================================================================================================
# the label's print (an original design, no lettering)
# ===================================================================================================================
RAYS = 20                           # sunburst rays
BEETS = (-150.0, -90.0, -30.0)     # the beets' bearings on the print (degrees from +X): the lower half, clear of the text
BEET_AT = 0.555                     # their centres, as a share of LABEL_R from the spindle


def _poly_sd(px, py, P):
    """Signed distance (negative inside; exact on the edges, a little short at the corners) of points to a convex
    counter-clockwise polygon P (k, 2)."""
    d = None
    for (ax, ay), (bx, by) in zip(P, np.roll(P, -1, axis=0)):
        ex, ey = bx - ax, by - ay
        n = math.hypot(ex, ey)
        s = ((px - ax) * ey - (py - ay) * ex) / n          # > 0 on the right of a CCW edge: outside
        d = s if d is None else np.maximum(d, s)
    return d


def _ellipse_sd(px, py, c, a, b, ang):
    """Approximate signed distance to an ellipse (centre c, semi-axes a along `ang` radians and b across)."""
    ca, sa = math.cos(ang), math.sin(ang)
    u = (px - c[0]) * ca + (py - c[1]) * sa
    v = -(px - c[0]) * sa + (py - c[1]) * ca
    return (np.hypot(u / a, v / b) - 1.0) * min(a, b)


def label_image(cols, size=1024):
    """The label's print as an (size, size, 3) float32 array of display (sRGB) values 0..1, row 0 at the top (+Y), the
    square spanning the label's diameter. `cols` = {label, rays, print, leaf, paper: sRGB triples}. Shapes are signed
    distances, anti-aliased over a pixel; each beet is drawn in its own window of the square."""
    px = 2.0 / size                                         # one pixel, in label radii
    c = ((np.arange(size) + 0.5) * px - 1.0).astype(np.float32)
    x, y = np.meshgrid(c, -c)                               # y up: row 0 is the top
    r = np.hypot(x, y)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(cols["label"], np.float32)
    whole = (slice(None), slice(None))

    def lay(sd, col, win=whole):
        k = np.clip(0.5 - sd / px, 0.0, 1.0)[..., None]
        img[win] += (np.asarray(col, np.float32) - img[win]) * k

    def ring(r0, r1):
        return np.maximum(r0 - r, r - r1)

    f = (np.arctan2(y, x) * (RAYS / (2.0 * math.pi))) % 1.0       # the sunburst's wedges, half of them lit
    wedge = (np.abs(f - 0.25) - 0.25) * (2.0 * math.pi / RAYS) * r
    lay(np.maximum(wedge, ring(0.26, 0.86)), cols["rays"])
    lay(r - 0.235, cols["print"])                           # the centre: a cream disc ringed in print
    lay(r - 0.205, cols["paper"])
    lay(ring(0.105, 0.120), cols["print"])                  # a ring round the spindle
    lay(ring(0.875, 0.890), cols["paper"])                  # a hairline, then the band at the edge
    lay(ring(0.920, 0.975), cols["print"])
    rb = 0.125                                              # a beet: leaves out, the body, a root tail in
    for deg in BEETS:
        t = math.radians(deg)
        o = np.array([math.cos(t), math.sin(t)])            # outward
        q = np.array([-o[1], o[0]])
        cb = BEET_AT * o
        c0, c1 = (int(max(0, (cb[0] + 1.0 + e * 0.36) / px)) for e in (-1, 1))
        r0, r1 = (int(max(0, (1.0 - cb[1] + e * 0.36) / px)) for e in (-1, 1))
        win = (slice(r0, r1), slice(c0, c1))
        X, Y = x[win], y[win]
        s = cb + o * (rb * 0.8)                             # where the leaves spring from
        for k in (-1, 0, 1):
            ang = t + k * math.radians(34.0)
            dv = np.array([math.cos(ang), math.sin(ang)])
            lay(_ellipse_sd(X, Y, s + dv * 0.095, 0.095, 0.045, ang), cols["leaf"], win)
            along = (X - s[0]) * dv[0] + (Y - s[1]) * dv[1]
            lay(np.maximum(np.abs((X - s[0]) * dv[1] - (Y - s[1]) * dv[0]) - 0.006, np.abs(along - 0.075) - 0.075),
                cols["print"], win)                         # the stalk and the leaf's midrib
        tail = np.array([cb - o * (rb + 0.085), cb - o * 0.03 + q * (rb * 0.62), cb - o * 0.03 - q * (rb * 0.62)])
        if (tail[1, 0] - tail[0, 0]) * (tail[2, 1] - tail[0, 1]) - (tail[1, 1] - tail[0, 1]) * (tail[2, 0] - tail[0, 0]) < 0:
            tail = tail[::-1]
        lay(np.minimum(np.hypot(X - cb[0], Y - cb[1]) - rb, _poly_sd(X, Y, tail)), cols["print"], win)
        hi = cb + o * (rb * 0.35) + q * (rb * 0.38)
        lay(_ellipse_sd(X, Y, hi, 0.032, 0.018, t + math.radians(60.0)), cols["paper"], win)   # a highlight
    return np.clip(img, 0.0, 1.0)


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def label_panel(name):
    """The label's `use.surface` panel above the spindle: it rides the platter (`object`)."""
    p = LABEL_PANEL
    cy = 0.5 * (p["y0"] + p["y1"])
    return {"name": "label", "center": _r((0.0, cy, Z_LABEL)), "normal": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0],
            "size": _r((p["w"], p["y1"] - p["y0"])), "object": f"{name}_platter"}


def card(name, roles=None, start=None):
    """The prop card (docs/design.md: Prop card): the run spot, the record's top, the label surface, the look point, the
    live root property `rpm` and the `rates` entry that makes the keys stage integrate it into `turns`."""
    start = dict(start or {k: v[0] for k, v in PARAMS.items()})
    use = {"feet": [{"name": "run", "L": [RUN[0] + STRIDE, RUN[1], None], "R": [RUN[0] - STRIDE, RUN[1], None],
                     "floor_z": round(Z_TOP, 5)}],
           "rest": [{"name": "top", "type": "plane", "center": _r((0.0, 0.0, Z_TOP)), "normal": [0.0, 0.0, 1.0],
                     "radius": LEADIN_R}],
           "surface": [label_panel(name)],
           "look": [{"name": "centre", "point": _r((0.0, 0.0, Z_LABEL))}]}
    return {"size": _r((PLINTH_W, PLINTH_D, SPINDLE_TOP), 4), "origin": "floor_center", "front": "-Y",
            "slots": dict(roles or {}), "use": use, "colliders": [], "params": start,
            "rates": [{"rate": "rpm", "into": "turns", "per": 60.0}],
            "objects": [f"{name}_{k}" for k in ("plinth", "feet", "platter", "record", "label")]}
