"""convertible_body: the body of the 1980s convertible, ONE smooth lofted shell (proportions of a 1984-86 Dodge 600 convertible).

The shell is a loft of half sections along y (`core.shell.loft`): a few stations per region, interpolated with monotone cubics,
mirrored, subdivided by the builder (a Subdivision Surface with creases) and cut by four wheel-arch solids with a flared lip
(a Boolean), then bevelled where the cut leaves a hard edge. The same point names run through every station:

    keel under rk0 rk1 fa fb fc crease u1 u2 e1 e2 e3 t1 t2 t2b t3 t4 tc
    underside .. rocker .. rub strip shelf .. flank (widest at fc) .. SHOULDER CREASE .. upper side .. top edge .. top surface

closed top (hood, deck): u1 u2 are the upper side leaning in, e1 e2 the top edge rolled over, e3 t1 t2 t2b t3 t4 tc the crowned
top; open top (cockpit): u1 u2 the door shoulder leaning in to the belt, e1 e2 e3 the rounded door-top cap, t1 t2 t2b the inner
wall, t3 (the tight corner) t4 tc the floor. Hard stations at the cowl and behind the cockpit turn the top into the walls of the opening.

Also here: the windshield (glass and frame, a plane raked 45 degrees) and the folded soft top under its padded boot.
Exports: static_shells() {"body", "boot"}, static_parts() {"glass", "trim"}; the geometry is bpy-free.
"""
import math

import numpy as np

from ....core import shell as S
from . import convertible_layout as LAY

M = LAY.M
NAMES = ("keel", "under", "rk0", "rk1", "fa", "fb", "fc", "crease", "u1", "u2", "e1", "e2", "e3", "t1", "t2", "t2b", "t3", "t4", "tc")
C45 = math.cos(math.radians(45.0))

HW = LAY.HALF_W                         # half width at the shoulder crease
Z_SILL = 0.26                           # bottom of the rocker, the lowest edge you see from the side
Z_PAN = 0.19                            # the floor pan under the cockpit, between the sills
Z_FLOOR = LAY.FLOOR_Z - 0.008           # the body's floor surface (the carpet lies on it)
ARCH_X0 = 0.58                          # the wheel opening's cut starts 5 cm inside the tyre's inner face
FLANK_Z = 0.62                          # the widest line of the body side: the side leans in above and below it (a barrel)
CREASE_IN = 0.018                       # how far the shoulder crease stands inside the widest line
DOOR_GAPS = (LAY.Y_DOOR_FRONT, LAY.Y_DOOR_REAR)    # the door's front and rear shut lines


# =============================================================================================== sections
def closed_section(hw, zb, zt, zc, r_e=0.05, tumble=0.014, crown=0.012, dy=0.0, zk=None):
    """Half section of a closed-top body: the underside, the lower side, the shoulder crease at zc, the upper side
    leaning in by `tumble`, the top edge rolled over with radius r_e, the crowned top surface to the centre line.
    hw: half width at the crease; zb: bottom of the sides; zk: centre-line underside; zt: top height on the centre line;
    dy rakes the face (points move +y with height: a nose that leans back)."""
    zk = zb if zk is None else zk
    ze = zt - crown                                # top surface at the edge of the roll
    r = min(r_e, 0.45 * (ze - zb))
    zs = ze - r                                    # where the roll leaves the side
    zce = min(zc, zs - 0.004)                      # the crease never rides above the roll
    xs = hw - tumble
    xc = hw - CREASE_IN
    xr = xs - r                                    # x where the roll meets the top
    top = lambda x: zt - crown * (x / xr) ** 2 if xr > 1e-6 else zt
    ramp = lambda z: dy * (z - zb) / max(zt - zb, 1e-6)
    P = [("keel", 0.0, zk), ("under", 0.55 * hw, zk + 0.004), ("rk0", hw - 0.040, zb), ("rk1", hw - 0.010, zb + 0.032),
         ("fa", hw - 0.014, LAY.Z_RUB - 0.03, 0.9), ("fb", hw - 0.004, LAY.Z_RUB + 0.012, 0.7), ("fc", hw, FLANK_Z),
         ("crease", xc, zce, 1.0), ("u1", 0.5 * (xc + xs), 0.5 * (zce + zs)), ("u2", xs, zs),
         ("e1", xr + r * C45, zs + r * C45), ("e2", xr, ze),
         ("e3", 0.80 * xr, top(0.80 * xr)), ("t1", 0.62 * xr, top(0.62 * xr)), ("t2", 0.45 * xr, top(0.45 * xr)),
         ("t2b", 0.33 * xr, top(0.33 * xr)), ("t3", 0.21 * xr, top(0.21 * xr)), ("t4", 0.10 * xr, top(0.10 * xr)), ("tc", 0.0, zt)]
    return [S.pt(n, x, z, ramp(z), *(c or (0.0,))) for n, x, z, *c in P]


def open_section(hw, zb, belt, zc, r_cap=0.022, xin=LAY.X_WALL_IN, xout=LAY.X_BELT_OUT, floor=Z_FLOOR, zk=Z_PAN):
    """Half section of the cockpit: the same lower side and crease, the upper side leaning in to the belt, the door-top
    cap (rounded: outer edge, crown, inner edge), the inner wall down to the floor and across it to the centre line."""
    zce = min(zc, belt - 0.06)
    xc = hw - CREASE_IN
    cap_w = xout - xin
    # the pan hangs lower than the sills (the floor above it is at the carpet height): the underside climbs steeply from the pan
    # to the rocker's bottom edge, outside the floor corner, so the section never crosses itself
    P = [("keel", 0.0, zk), ("under", 0.55 * hw, zk + 0.004), ("rk0", 0.75, zk + 0.010), ("rk1", hw - 0.010, zb + 0.032),
         ("fa", hw - 0.014, LAY.Z_RUB - 0.03, 0.9), ("fb", hw - 0.004, LAY.Z_RUB + 0.012, 0.7), ("fc", hw, FLANK_Z),
         ("crease", xc, zce, 1.0), ("u1", 0.5 * (xc + xout), 0.5 * (zce + belt - r_cap)), ("u2", xout, belt - r_cap),
         ("e1", xout - r_cap * (1 - C45), belt - r_cap * (1 - C45)), ("e2", xout - 0.5 * cap_w, belt + 0.002),
         ("e3", xin + r_cap * (1 - C45), belt - r_cap * (1 - C45)),
         ("t1", xin, belt - r_cap), ("t2", xin, 0.36), ("t2b", xin - 0.004, 0.28), ("t3", xin - 0.04, floor + 0.0065),
         ("t4", 0.55, floor + 0.0015), ("tc", 0.0, floor)]
    return [S.pt(n, x, z, 0.0, *(c or (0.0,))) for n, x, z, *c in P]


# =============================================================================================== stations
# (y, half width at the crease, shoulder crease height) of the nose, the hood height comes from the layout
NOSE = ((-2.210, 0.770, 0.730), (-2.200, 0.825, 0.755), (-2.180, 0.852, 0.780), (-2.150, 0.862, 0.800),
        (-2.100, 0.865, 0.815), (-1.900, 0.865, LAY.Z_SHOULDER), (-1.600, 0.865, LAY.Z_SHOULDER),
        (-1.310, 0.865, LAY.Z_SHOULDER), (-1.000, 0.865, LAY.Z_SHOULDER), (-0.740, 0.865, LAY.Z_SHOULDER),
        (-0.695, 0.865, LAY.Z_SHOULDER), (LAY.Y_COWL - 0.015, 0.865, LAY.Z_SHOULDER))
# the cockpit: door tops flat to the rear quarters, sloping down behind the rear seat
COCKPIT = ((LAY.Y_COWL, LAY.Z_BELT), (0.0, LAY.Z_BELT), (0.62, LAY.Z_BELT), (1.10, 0.895))
# (y, half width, deck height on the centre line)
DECK = ((1.14, 0.870, 0.876), (1.50, 0.866, 0.877), (1.90, 0.862, 0.873), (2.15, 0.860, 0.868), (2.18, 0.850, 0.863),
        (2.20, 0.825, 0.859), (2.21, 0.770, 0.856))


def _zside(y):
    """Bottom of the sides: the valance steps up a little toward the bumper ends."""
    d = min(abs(y - LAY.Y_NOSE), abs(LAY.Y_TAIL - y))
    return 0.30 - 0.04 * min(1.0, d / 0.3)


def stations():
    st = []
    for i, (y, hw, z_c) in enumerate(NOSE):
        dy = 0.045 if i == 0 else (0.022 if i == 1 else 0.0)
        crown = 0.013
        st.append(S.station(y, closed_section(hw, _zside(y), LAY.hood_z(y), z_c, r_e=0.028, tumble=0.012, dy=dy, crown=crown, zk=_zside(y)),
                            hard=(i == len(NOSE) - 1)))
    for i, (y, belt) in enumerate(COCKPIT):
        st.append(S.station(y, open_section(HW, Z_SILL, belt, LAY.Z_SHOULDER), hard=(i == 0 or i == len(COCKPIT) - 1)))
    for i, (y, hw, zt) in enumerate(DECK):
        dy = {len(DECK) - 1: -0.05, len(DECK) - 2: -0.03, len(DECK) - 3: -0.012}.get(i, 0.0)        # the tail panel leans forward
        st.append(S.station(y, closed_section(hw, _zside(y), zt, LAY.Z_SHOULDER, r_e=0.025, tumble=0.012, crown=0.010, dy=dy, zk=_zside(y)),
                            hard=(i == 0)))
    return st


def gaps():
    """Thin dark grooves: the hood's front edge, the door shut lines (sides, belt to rocker), the deck lid's edges."""
    g = [S.gap(-2.10, 0.006, 0.003, ("e1", "tc"), M["seam"]),
         S.gap(LAY.Y_HOOD_REAR, 0.006, 0.003, ("e1", "tc"), M["seam"]),
         S.gap(1.72, 0.006, 0.003, ("e1", "tc"), M["seam"]),
         S.gap(2.145, 0.006, 0.003, ("e1", "tc"), M["seam"])]
    g += [S.gap(y, 0.006, 0.004, ("rk1", "u2"), M["seam"]) for y in DOOR_GAPS]
    return g


def body_loft():
    lf = S.loft(stations(), spacing=0.12, cap_rings=2, gaps=gaps())
    lf.assign(M["underbody"], cols=("keel", "rk0"))
    lf.assign(M["underbody"], y=(-0.60, 1.145), cols=("t1", "tc"))
    return lf


def arch_cutters():
    """The four wheel openings: cylinders about the axle a hand wider than the tyre, flaring into a rounded lip."""
    out = []
    for ay in (LAY.AXLE_F, LAY.AXLE_R):
        for sx in (1.0, -1.0):
            out.append(S.arch_cutter((ay, LAY.WHEEL_Z), LAY.ARCH_R, sx * ARCH_X0, sx * 0.895, flare=0.028, lip=0.055, mat=M["underbody"]))
    return out


# =============================================================================================== windshield
def ws_frame():
    """(origin, normal, up) of the windshield plane at the middle of its base, and its length up the glass."""
    (y0, z0), (y1, z1) = LAY.WS_BASE, LAY.WS_TOP
    up = np.array([0.0, y1 - y0, z1 - z0])
    L = float(np.linalg.norm(up))
    up = up / L
    normal = np.array([0.0, -up[2], up[1]])                  # out of the glass, forward and up
    return np.array([0.0, y0, z0]), normal, up, L


def _trapezoid(w0, w1, y0, y1, r0, r1):
    """Closed CCW outline of the glass: base width w0 at y0, top width w1 at y1, rounded corners."""
    P = np.array([[-w0 / 2, y0], [w0 / 2, y0], [w1 / 2, y1], [-w1 / 2, y1]])
    return S.fillet(P, [r0, r0, r1, r1], n=4, closed=True)


def windshield():
    """(glass, frame) in car coordinates, both on the plane from the cowl to the header: a thin pane (rolled edge, 4 mm) in a
    body-colour frame that is a closed sweep of a rounded section (the A-pillars and the header are tubes, not slabs)."""
    o, n, up, L = ws_frame()
    bw0, bw1 = 2 * LAY.WS_HALF_W_BASE, 2 * LAY.WS_HALF_W_TOP
    fw = LAY.WS_FRAME
    mid = _trapezoid(bw0 - fw, bw1 - fw, 0.5 * fw, L - 0.5 * fw, 0.020, 0.034)           # the centre line of the frame
    path = np.column_stack([S.resample_loop(mid, 180), np.zeros(180)])
    sec = S.superellipse(fw, 0.042, 2.8, 20)                                                 # across the frame, out of the glass
    frame = S.sweep(path, sec, up=(0.0, 0.0, 1.0), closed=True, mat=M["paint"]).weld(1e-9)
    frame = frame if frame.volume() > 0 else frame.flip()
    glass = S.rounded_panel(_trapezoid(bw0 - 2 * fw + 0.014, bw1 - 2 * fw + 0.014, 0.030, L - 0.054, 0.010, 0.020), 0.006,
                            r_edge=0.0035, r_back=0.0025, n=3, mat=M["glass"])
    frame = S.place(frame, o - n * 0.010, n, up)
    glass = S.place(glass, o + n * 0.003, n, up)
    return glass, frame


# =============================================================================================== folded top and boot
BOOT = dict(y0=1.13, y1=1.68, hw=0.725, rise=0.092, r=0.045)


def boot_cage():
    """The padded boot over the folded soft top: a low pillow, flat underneath (buried in the deck), its front edge against
    the rear seat, the ends and sides rolled over, the top crowned. A cage: the builder subdivides it."""
    b = BOOT
    ys = np.array([b["y0"], b["y0"] + 0.02, b["y0"] + 0.07, 0.5 * (b["y0"] + b["y1"]), b["y1"] - 0.07, b["y1"] - 0.02, b["y1"]])
    sz = np.array([0.58, 0.90, 0.99, 1.0, 0.99, 0.90, 0.60])                     # width factor along the boot
    lift = np.array([0.0, 0.60, 0.93, 1.0, 0.93, 0.60, 0.0])                     # height factor
    z0 = LAY.Z_DECK - 0.012                                                      # the deck, a little buried
    rings = []
    for y, w, h in zip(ys, sz, lift):
        hw, hh = b["hw"] * w, 0.5 * max(b["rise"] * h, 0.012)                    # half width, half height of the ring
        P = S.superellipse(2 * hw, 2 * hh, 3.2, 32)                              # (x, z) around, CCW seen from +y
        rings.append(np.array([[p[0], y, z0 + hh + p[1]] for p in P]))
    R = np.stack(rings)
    return S.cage(R, closed=True, caps=("fan", "fan"), cap_bulge=(0.0, 0.0), cap_rings=1, crease_rings=[0.8, 0, 0, 0, 0, 0, 0.8],
                  mat=M["boot"])


# =============================================================================================== exports
def static_shells():
    lf = body_loft()
    return {"body": S.smooth(lf.mesh, levels=2, bevel=(0.003, 2), cutters=arch_cutters(), cutter_role=M["underbody"], solver="FAST"),
            "boot": S.smooth(boot_cage(), levels=2, bevel=None)}


def static_parts():
    glass, frame = windshield()
    return {"glass": glass, "trim": frame}
