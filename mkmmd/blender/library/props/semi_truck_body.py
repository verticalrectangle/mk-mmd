"""semi_truck_body: the tractor's smooth body (pure numpy, no bpy): the tilt hood with its fenders, the cab
(`semi_truck_body_cab`), the sleeper, the grille, and the cab's glass and seals.

static_shells()   {"hood": the hood and both front fenders, one loft cut round the steer tyres; "cab": the cab's loose
                   parts; "sleeper": the sleeper box, the walk-through's pocket cut into its front}
static_parts()    {"grille": the chrome shell, its bars and the dark mesh behind them; "glass": the windshield and the door
                   glass; "seals": their rubber}

The hood is one loft along y from its face behind the grille (L.GRILLE_Y + GRILLE_DEPTH) to its rear edge at L.COWL_Y: a
half section from the underside round the fender (out to L.FENDER_HALF_W, its crown at L.FENDER_TOP_Z over the tyre) and up
the hood's side to its flat top (L.HOOD_TOP_HALF_W, L.HOOD_Z), as the tilt hood of a long-nose tractor is one fibreglass
piece. A cylinder of L.FENDER_ARCH_R about the steer axle cuts each wheel opening. The sleeper is a loft too: flat sides
at L.SLEEPER_HALF_W, rounded corners in plan (the stacks stand in the notch the cab's and its own make), a front cap rising
from the cab's roof to L.SLEEPER_ROOF_Z.
"""
import numpy as np

from ....core import shell as S
from . import semi_truck_body_cab as CAB
from . import semi_truck_layout as L

M = L.M

GRILLE_DEPTH = 0.075                 # the grille shell from its face (L.GRILLE_Y) back to the hood's face
HOOD_FRONT_Y = L.GRILLE_Y + GRILLE_DEPTH - 0.003
SKIRT_Z = (0.95, 0.98)               # the fenders' lower edge at the front and at the rear (the arches cut above it)


def _hood_station(y, front=False):
    """The hood's half section at y: underside, fender (skirt, side, crown), the valley where the fender meets the hood's
    side, the side, the roll and the crowned top."""
    u = (y - HOOD_FRONT_Y) / (L.COWL_Y - HOOD_FRONT_Y)             # 0 at the face, 1 at the rear edge
    ztop = L.HOOD_Z[0] + (L.HOOD_Z[1] - L.HOOD_Z[0]) * u
    hw = L.HOOD_TOP_HALF_W[0] + (L.HOOD_TOP_HALF_W[1] - L.HOOD_TOP_HALF_W[0]) * u
    fc = float(np.interp(y, [HOOD_FRONT_Y, -6.80, L.AXLE_S_Y, -5.95, -5.72, L.COWL_Y],
                         [L.FENDER_TOP_Z - 0.08, L.FENDER_TOP_Z - 0.01, L.FENDER_TOP_Z, L.FENDER_TOP_Z - 0.01,
                          L.FENDER_TOP_Z - 0.08, L.FENDER_TOP_Z - 0.20]))
    fo = float(np.interp(y, [HOOD_FRONT_Y, -6.85, -5.80, L.COWL_Y],
                         [L.FENDER_HALF_W - 0.06, L.FENDER_HALF_W, L.FENDER_HALF_W, L.FENDER_HALF_W - 0.07]))
    zb = SKIRT_Z[0] + (SKIRT_Z[1] - SKIRT_Z[0]) * u
    h = fc - zb                                                     # the fender's side, skirt to crown
    xs = hw + 0.06                                                  # the hood's side panel
    xv = xs + 0.03                                                  # the valley between it and the fender
    pts = [S.pt("keel", 0.0, zb + 0.04), S.pt("u1", 0.50, zb + 0.03), S.pt("u2", fo - 0.10, zb),
           S.pt("skirt", fo, zb + 0.12 * h), S.pt("side0", fo, zb + 0.35 * h), S.pt("side1", fo, zb + 0.70 * h),
           S.pt("shoulder", fo - 0.05, fc - 0.03), S.pt("crown", fo - 0.20, fc), S.pt("inner", xv + 0.06, fc - 0.02),
           S.pt("valley", xv, fc + 0.02, crease=0.7), S.pt("hside0", xs, fc + 0.10), S.pt("hside1", xs, ztop - 0.14),
           S.pt("roll", hw + 0.025, ztop - 0.025), S.pt("top", 0.6 * hw, ztop + 0.008), S.pt("tc", 0.0, ztop + 0.012)]
    return S.station(y, pts, hard=front)


def hood():
    """The hood and fenders: (cage, [wheel-opening cutters])."""
    ys = [HOOD_FRONT_Y, HOOD_FRONT_Y + 0.03, -6.86, -6.70, L.AXLE_S_Y, -6.00, -5.78, -5.62, L.COWL_Y]
    lf = S.loft([_hood_station(y, i == 0) for i, y in enumerate(ys)], spacing=0.10, cap_bulge=(0.01, 0.0),
                mat=M["paint"])
    lf.assign(M["underbody"], cols=("keel", "u2"))
    cutters = []
    for sx in (1.0, -1.0):
        path = [[sx * 0.45, L.AXLE_S_Y, L.WHEEL_Z], [sx * (L.FENDER_HALF_W + 0.25), L.AXLE_S_Y, L.WHEEL_Z]]
        c = S.tube(path, L.FENDER_ARCH_R, sides=48, mat=M["underbody"])
        cutters.append(c if c.volume() > 0 else c.flip())
    return lf.mesh, cutters


# ====================================================================================================== the sleeper
def _sleeper_station(y):
    y0, y1 = L.SLEEPER_Y
    hw = float(np.interp(y, [y0, y0 + 0.03, y0 + 0.08, y0 + 0.135, y1 - 0.10, y1 - 0.04, y1],
                         [L.SLEEPER_HALF_W - 0.12, L.SLEEPER_HALF_W - 0.045, L.SLEEPER_HALF_W - 0.01, L.SLEEPER_HALF_W,
                          L.SLEEPER_HALF_W, L.SLEEPER_HALF_W - 0.03, L.SLEEPER_HALF_W - 0.10]))
    zt = float(np.interp(y, [y0, y0 + 0.15, y0 + 0.30, y0 + 0.45, y0 + 0.60, y1 - 0.08, y1],
                         [L.CAB_ROOF_Z - 0.03, L.CAB_ROOF_Z + 0.09, L.SLEEPER_ROOF_Z - 0.12, L.SLEEPER_ROOF_Z - 0.02,
                          L.SLEEPER_ROOF_Z, L.SLEEPER_ROOF_Z - 0.01, L.SLEEPER_ROOF_Z - 0.09]))
    zb = L.CAB_BOTTOM_Z
    pts = [S.pt("keel", 0.0, zb), S.pt("u", hw - 0.08, zb), S.pt("s0", hw, zb + 0.06), S.pt("s1", hw, zb + 0.30),
           S.pt("s2", hw, zt - 0.40), S.pt("s3", hw, zt - 0.20), S.pt("r1", hw - 0.04, zt - 0.05),
           S.pt("r2", hw - 0.18, zt), S.pt("tc", 0.0, zt + 0.01)]
    return S.station(y, pts)


def sleeper():
    """The sleeper: (cage, [the walk-through's pocket]). The pocket opens its front behind the cab's walk-through, as deep
    as the curtain hanging in it needs."""
    y0, y1 = L.SLEEPER_Y
    ys = [y0, y0 + 0.03, y0 + 0.08, y0 + 0.135, y0 + 0.30, y0 + 0.45, y0 + 0.60, y1 - 0.10, y1 - 0.04, y1]
    lf = S.loft([_sleeper_station(y) for y in ys], spacing=0.15, cap_bulge=(0.0, 0.0), mat=M["paint"])
    lf.assign(M["underbody"], cols=("keel", "u"))
    hw, z0, z1 = L.WALKTHROUGH
    depth = L.CURTAIN_Y + 0.10 - (y0 - 0.05)
    pocket = S.rounded_box((2 * hw, depth, z1 - z0), center=(0.0, y0 - 0.05 + depth / 2, (z0 + z1) / 2), r=0.02,
                           mat=M["vinyl_dark"])
    return lf.mesh, [pocket]


# ====================================================================================================== the grille
def grille():
    """The grille: a chrome shell (a rounded tube of a frame round the opening), vertical chrome bars, the dark mesh."""
    w, z0, z1 = 2 * L.GRILLE_HALF_W, L.GRILLE_Z[0], L.GRILLE_Z[1]
    zc, yc = (z0 + z1) / 2, L.GRILLE_Y + GRILLE_DEPTH / 2
    loop = S.rrect(w - 0.06, (z1 - z0) - 0.06, 0.06, 4)
    path = np.column_stack([loop[:, 0], np.full(len(loop), yc), loop[:, 1] + zc])
    frame = S.sweep(path, S.rrect(0.06, GRILLE_DEPTH, 0.014, 2), up=(0.0, 1.0, 0.0), closed=True,
                    mat=M["chrome"]).weld(1e-6)
    parts = [frame if frame.volume() > 0 else frame.flip()]
    parts.append(S.rounded_box((w - 0.08, 0.02, (z1 - z0) - 0.08), center=(0.0, L.GRILLE_Y + GRILLE_DEPTH - 0.01, zc),
                               r=0.005, mat=M["grille_back"]))
    n = 17
    for x in np.linspace(-(w / 2 - 0.075), w / 2 - 0.075, n):
        parts.append(S.rounded_box((0.014, 0.035, (z1 - z0) - 0.10), center=(float(x), L.GRILLE_Y + 0.03, zc), r=0.005,
                                   mat=M["chrome"]))
    return S.merge(parts)


# ====================================================================================================== the module's interface
def static_shells():
    hood_cage, arches = hood()
    sleeper_cage, pocket = sleeper()
    return {"hood": S.smooth(hood_cage, levels=2, bevel=(0.01, 2), cutters=arches, cutter_role=M["underbody"]),
            "cab": S.smooth(CAB.cab_cage(), levels=2),
            "sleeper": S.smooth(sleeper_cage, levels=2, bevel=(0.008, 2), cutters=pocket, cutter_role=M["vinyl_dark"])}


def static_parts():
    return {"grille": grille(), "glass": S.merge(CAB.glass_parts()), "seals": S.merge(CAB.seal_parts())}
