"""The semi truck's wrap-around dash (pure numpy, no bpy): the dash body, the driver's gauge cluster, the centre stack
angled to the driver, the passenger's glove box and vents, the steering column, the CB radio with its handheld mic.

    dash_parts()     {group: [Mesh]}: "cab_dash" forms, "cab_dash_decals" graphic layers (tick marks, display segments)
    GAUGES           [(name, centre (3,), normal (3,), radius)] every gauge's dial: the tests aim the driver's eye at them
    CB_C, MIC_GRIP   the radio's display and the mic's grip (the card's LOOKS `cb` and ANCHORS `cb_mic`)

The dash body is a loft along x of a (y, z) profile: a crowned top from the windshield's foot to a rolled brow at
L.DASH_Y[1], L.DASH_TOP_Z high; under the brow the face, recessed in front of the driver where the cluster sits, the knee
panel, the underside, a front wall sunk into the toe board. The cluster is a pod (a moulded bezel with a woodgrain panel
in its pocket) in the plane through L.GAUGES_C that faces the driver's eye (kit.EYE): the tach and the speedo stand either
side of where the hub and the column hide the middle, the small gauges outside them, all in the wheel's open upper half.
The centre stack is a loft along z standing on the floor, its face turned STACK_TURN toward the driver and leaning back:
small gauges, toggle switches, knobs, and low down the CB radio with the mic on a hook and its coiled cord.

Roles: dash body, bezels, stack `vinyl_dark`, panels `woodgrain`, gauge bezels `chrome`, dials `gauge_face`, needles and
the lit markings `gauge_needle`, toggles / knobs / vent louvres `switch`, radio and mic `plastic_dark`, the channel digits
`display`, the column `vinyl_dark` with a `rubber` boot.
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_cab_kit as K
from . import semi_truck_layout as L

M = L.M
EX, EY, EZ = np.eye(3)

# ====================================================================================================== the dash body
DASH_X = K.X_IN - 0.006                              # the body's ends stand against the side walls
Y_BROW = L.DASH_Y[1]                                 # the brow's back edge
Z_TOP = L.DASH_TOP_Z
Y_FRONT = K.Y_FIREWALL - 0.045                       # the front wall: sunk into the toe board
Y_RECESS = -5.000                                    # the face behind the cluster and the stack

# profile points (y, z, corner radius): the top, the brow, then the face per zone, then the underside and the front wall
_TOP = [(Y_FRONT + 0.010, Z_TOP - 0.045, 0.020), (-4.975, Z_TOP - 0.010, 0.25), (Y_BROW - 0.008, Z_TOP, 0.022),
        (Y_BROW - 0.002, Z_TOP - 0.048, 0.010), (Y_BROW - 0.045, Z_TOP - 0.062, 0.010)]
_FACE = {
    "p": [(-4.920, 1.945, 0.020), (-4.948, 1.800, 0.060), (-4.990, 1.605, 0.050)],     # passenger: a face leaning 10 deg
    "d": [(Y_RECESS, 1.945, 0.020), (-4.888, 1.790, 0.030), (-4.940, 1.605, 0.050)],   # driver: recessed for the cluster
    "s": [(Y_RECESS, 1.945, 0.020), (Y_RECESS + 0.004, 1.800, 0.060), (Y_RECESS, 1.605, 0.050)],   # behind the stack
}
_UNDER = [(-5.022, 1.545, 0.030), (Y_FRONT + 0.012, 1.530, 0.020), (Y_FRONT - 0.035, 1.960, 0.020)]


def dash_profile(zone):
    """The dash body's section (m, 2) of (y, z) in a zone: "p" passenger, "d" driver (the cluster's recess), "s" behind the
    centre stack. Every zone has the same corners, so the sections make one cage."""
    P = np.array(_TOP + _FACE[zone] + _UNDER)
    return K.rounded_poly(P[:, :2], P[:, 2], n=3)


DASH_STATIONS = ((-DASH_X, "p"), (-0.62, "p"), (-0.30, "p"), (-0.215, "p"), (-0.165, "s"), (0.075, "s"), (0.125, "d"),
                 (0.40, "d"), (0.70, "d"), (DASH_X, "d"))


def dash_body():
    rings = [np.column_stack([np.full(len(P), x), P]) for x, zone in DASH_STATIONS for P in [dash_profile(zone)]]
    return K.capped(np.stack(rings), mat=M["vinyl_dark"])


# ====================================================================================================== the cluster
GC = np.array(L.GAUGES_C, float)
CLUSTER_FR = K.frame(K.EYE - GC, EZ)                 # (right = -X as the driver sees it, up, normal toward the eye)
CR, CU, CN = CLUSTER_FR
CLUSTER_U = (-0.430, 0.335)                          # the pod's ends along its right axis (u = 0 at GAUGES_C): x 0.90 .. 0.135
CLUSTER_V = (-0.125, 0.125)
POD_DRAFT = 0.030                                    # the pod's walls lean in this much from its back to its face
POD_FACE = 0.010                                     # the bezel's face stands this far over the woodgrain panel
POD_BACK = 0.120                                     # ... its back this far behind it, inside the dash body
BIG_R, SMALL_R = 0.058, 0.031                        # gauge bezels' outer radii
GAUGE_UV = (("tach", (-0.205, 0.004), BIG_R), ("speedo", (0.205, 0.004), BIG_R),
            ("oil", (-0.312, 0.040), SMALL_R), ("water", (-0.312, -0.046), SMALL_R),
            ("fuel", (0.312, 0.040), SMALL_R), ("volts", (0.312, -0.046), SMALL_R))
NEEDLE = {"tach": 0.30, "speedo": 0.46, "oil": 0.55, "water": 0.48, "fuel": 0.72, "volts": 0.52,
          "air_p": 0.80, "air_s": 0.78, "boost": 0.12, "trans": 0.40, "axle": 0.38, "pyro": 0.20}
VENT_UV, VENT_R = (-0.378, 0.004), 0.026             # the round vent at the cluster's left end


def cluster_point(u, v, w=0.0):
    return GC + u * CR + v * CU + w * CN


def _cluster_pod():
    """The moulded bezel round the cluster (a frame with a pocket and drafted walls) and the woodgrain panel in it."""
    u0, u1 = CLUSTER_U
    v0, v1 = CLUSTER_V
    w, h = u1 - u0, v1 - v0
    uc = 0.5 * (u0 + u1)
    draft = POD_DRAFT
    outer = S.rrect(w + 2 * draft, h + 2 * draft, 0.075, 6)
    inner = S.rrect(w - 0.040, h - 0.040, 0.048, 6)
    pod = S.rounded_frame(outer, inner, POD_FACE + POD_BACK, r_out=0.010, r_in=0.004, n=3, floor=POD_FACE + 0.006,
                          mat=M["vinyl_dark"], draft=draft)
    origin = cluster_point(uc, 0.0, -POD_BACK)
    out = [K.place(pod, origin, CLUSTER_FR)]
    wood = S.rounded_panel(S.rrect(w - 0.046, h - 0.046, 0.045, 6), 0.006, r_edge=0.002, n=2, rings=1, mat=M["woodgrain"])
    out.append(K.place(wood, cluster_point(uc, 0.0, -0.006), CLUSTER_FR))
    return out


# ====================================================================================================== the centre stack
STACK_TURN = 30.0                                    # the stack's face turns this far toward the driver (plan)
STACK_LEAN = 18.0                                    # ... and leans back this far from vertical
STACK_C = np.array([-0.030, -4.800, 1.650])          # the face's centre
STACK_HALF_W = 0.170                                 # half width of the face
STACK_Z = (K.Z_FLOOR - 0.003, 2.005)                 # from just inside the floor to its rolled top
STACK_SPLAY = 15.0                                   # the side walls open out this much toward the back (plan)
_tn = math.radians(STACK_TURN)
STACK_NH = np.array([math.sin(_tn), math.cos(_tn), 0.0])         # the face's horizontal normal
STACK_T = np.array([-math.cos(_tn), math.sin(_tn), 0.0])         # along the face, to the driver's right
STACK_N = K.unit(STACK_NH * math.cos(math.radians(STACK_LEAN)) + EZ * math.sin(math.radians(STACK_LEAN)))
STACK_FR = K.frame(STACK_N, EZ)                       # (right, up, normal) of the face


def stack_point(s, h, w=0.0):
    """A point on the stack's face: s along it (to the driver's right), h up it, w out of it."""
    r, u, n = STACK_FR
    return STACK_C + s * r + h * u + w * n


def _stack_ring(z, inset=0.0):
    """The stack's plan outline at height z (a rounded quadrilateral, (m, 2)): the face's two ends, the side walls splayed
    STACK_SPLAY back into the dash, the back inside the dash body; `inset` pulls it in (the rolled top)."""
    lean = math.tan(math.radians(STACK_LEAN))
    c = STACK_C[:2] - STACK_NH[:2] * lean * (z - STACK_C[2])
    fl = c - STACK_T[:2] * STACK_HALF_W
    fr = c + STACK_T[:2] * STACK_HALF_W
    sp = math.tan(math.radians(STACK_SPLAY))
    yb = Y_RECESS - 0.010
    bl = np.array([fl[0] + (fl[1] - yb) * sp, yb])
    br = np.array([fr[0] - (fr[1] - yb) * sp, yb + 0.030])
    P = np.array([fl, fr, br, bl])                   # counter-clockwise in plan
    if inset:
        P = P + S.mitre_offsets(P) * inset
    return K.rounded_poly(P, [0.022, 0.022, 0.012, 0.012], n=4)


def _stack_body():
    z0, z1 = STACK_Z
    rt = 0.024
    zs = list(np.linspace(z0, z1 - rt, 6))
    rings = [(z, 0.0) for z in zs]
    rings += [(z1 - rt + rt * math.sin(t), rt * (1.0 - math.cos(t))) for t in np.radians([30.0, 60.0, 85.0])]
    R = []
    for z, inset in rings:
        P = _stack_ring(z, inset)
        R.append(np.column_stack([P, np.full(len(P), z)]))
    return K.capped(np.stack(R), mat=M["vinyl_dark"])


STACK_GAUGES = (("air_p", (-0.088, 0.262)), ("air_s", (0.0, 0.262)), ("boost", (0.088, 0.262)),
                ("trans", (-0.088, 0.180)), ("axle", (0.0, 0.180)), ("pyro", (0.088, 0.180)))
STACK_GAUGE_R = 0.033
TOGGLES_H = 0.098                                    # the row of toggle switches
KNOBS_H = 0.010                                      # the heater knobs
WOOD_H = ((0.060, 0.326), (-0.052, 0.050))           # the two woodgrain panels on the face: (bottom, top) along it


def _stack_panels():
    out = []
    for h0, h1 in WOOD_H:
        hh = h1 - h0
        wood = S.rounded_panel(S.rrect(2 * STACK_HALF_W - 0.040, hh, 0.022, 4), 0.005, r_edge=0.002, n=2, rings=1,
                               mat=M["woodgrain"])
        out.append(K.place(wood, stack_point(0.0, 0.5 * (h0 + h1), -0.003), STACK_FR))
    return out


# ====================================================================================================== gauges
def _arc_ticks(fr, origin, r0, r1, a0, a1, n, width, w, mat):
    """Tick marks (flat quads, decals) on a dial: n of them from angle a0 to a1 (radians, CCW from the dial's right),
    radially from r0 to r1, `w` over the dial's face."""
    r, u, nrm = fr
    a = np.linspace(a0, a1, n)
    d = np.cos(a)[:, None] * r + np.sin(a)[:, None] * u
    t = np.cross(nrm, d)
    C = origin + d * (0.5 * (r0 + r1)) + nrm * w
    hl, hw = 0.5 * (r1 - r0), 0.5 * width
    V = np.stack([C - d * hl - t * hw, C + d * hl - t * hw, C + d * hl + t * hw, C - d * hl + t * hw], 1)
    k = len(a)
    return S.Mesh(V.reshape(-1, 3), np.arange(4 * k).reshape(k, 4), mat=mat)


def gauge(origin, fr, r_out, reading, big=False):
    """One gauge on a panel face at `origin` (frame fr): a chrome bezel ring sunk 4 mm into the panel, the dial 2 mm proud
    of the panel inside it, a needle at `reading` (0..1 of the sweep) on a hub cap; tick marks (decals). Returns (forms,
    decals, dial centre)."""
    r, u, n = fr
    r_in = r_out - (0.009 if big else 0.0055)
    seg = 32 if big else 16
    forms = [K.place(K.bezel(r_out, r_in, 0.012 if big else 0.009, seg=seg, mat=M["chrome"]), origin, fr, -0.004)]
    dial = S.lathe(np.array([(0.0, 0.0), (r_in + 0.0008, 0.0), (r_in + 0.0008, 0.004), (0.0, 0.004)]), seg=seg,
                   mat=M["gauge_face"])
    forms.append(K.place(dial, origin, fr, -0.002))
    face_w = 0.002
    sweep = math.radians(270.0 if big else 100.0)
    a_start = math.radians(-135.0 if big else -50.0)   # measured from 12 o'clock, clockwise as the driver sees it
    a = a_start + reading * sweep
    d = math.sin(a) * r + math.cos(a) * u
    length = r_in * (0.86 if big else 0.80)
    sec = np.array([(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)])
    base = origin + n * (face_w + 0.0012)
    tail = base - d * 0.012 * (r_in / 0.05)
    path = np.stack([tail + (base + d * length - tail) * s for s in np.linspace(0.0, 1.0, 5)])
    widths = np.linspace(0.0042 if big else 0.0030, 0.0010, 5)
    forms.append(K.solid(S.sweep(path, sec, scale=np.stack([widths, np.full(5, 0.0009)], 1), up=n, mat=M["gauge_needle"])))
    cap = K.rprof([(0.0, 0.0), (0.0065 if big else 0.0045, 0.0), (0.0065 if big else 0.0045, 0.0022), (0.0, 0.0048)],
                  [0.0, 0.0, 0.0015, 0.0], n=2)
    forms.append(K.lathe_at(cap, origin + n * face_w, n, seg=12 if big else 8, mat=M["switch"]))
    # the markings: majors and minors on the sweep (clockwise as seen: angle in the dial's (r, u) plane = 90 deg - a)
    lo, hi = math.pi / 2 - a_start, math.pi / 2 - (a_start + sweep)
    decals = [_arc_ticks(fr, origin, r_in * 0.80, r_in * 0.95, lo, hi, 11 if big else 5, r_in * 0.045, face_w + 0.0002,
                         M["gauge_needle"]),
              _arc_ticks(fr, origin, r_in * 0.88, r_in * 0.95, lo, hi, 41 if big else 9, r_in * 0.020, face_w + 0.0002,
                         M["gauge_needle"])]
    if big:                                           # the red band at the top of the tach's / speedo's sweep
        decals.append(_arc_ticks(fr, origin, r_in * 0.72, r_in * 0.78, lo + (hi - lo) * 0.82, hi, 14, r_in * 0.11,
                                 face_w + 0.0003, M["display"]))
    return forms, decals, origin + n * face_w


def _cluster_gauges(b):
    for name, (u, v), rr in GAUGE_UV:
        f, d, c = gauge(cluster_point(u, v), CLUSTER_FR, rr, NEEDLE[name], big=rr > 0.05)
        b.forms += f
        b.decals += d
        b.gauges.append((name, c, CN.copy(), rr))
    for name, (s, h) in STACK_GAUGES:
        f, d, c = gauge(stack_point(s, h), STACK_FR, STACK_GAUGE_R, NEEDLE[name])
        b.forms += f
        b.decals += d
        b.gauges.append((name, c, STACK_N.copy(), STACK_GAUGE_R))


def _indicators(b):
    """A strip of round warning lamps over the middle of the cluster: a dark rolled strip, six domed lenses (amber, red)."""
    strip = K.slab(S.stadium(0.170, 0.020, 3), 0.004, r_top=0.0018, draft=0.001, mat=M["plastic_dark"])
    b.forms.append(K.place(strip, cluster_point(0.0, 0.088), CLUSTER_FR, -0.002))
    for k, u in enumerate(np.linspace(-0.066, 0.066, 6)):
        lens = S.rounded_panel(S.ellipse(0.0050, 0.0050, 10), 0.003, r_edge=0.0012, n=1, rings=1, dome=0.0008,
                               mat=M["display"] if k % 3 == 1 else M["gauge_needle"])
        b.forms.append(K.place(lens, cluster_point(u, 0.088), CLUSTER_FR, 0.0005))


# ====================================================================================================== switches, knobs, vents
def _toggle(origin, fr, up_deg=18.0):
    """A toggle switch: a chrome nut and bushing on the panel, a dark bat leaning up."""
    u, n = fr[1], fr[2]
    out = [K.lathe_at(K.rprof([(0.0, 0.0), (0.0068, 0.0), (0.0068, 0.0030), (0.0042, 0.0042), (0.0042, 0.0080),
                                (0.0, 0.0080)], [0, 0, 0.0008, 0, 0.0008, 0], n=1), origin + n * -0.001, n, seg=10,
                      mat=M["chrome"])]
    a = math.radians(up_deg)
    d = K.unit(n * math.cos(a) + u * math.sin(a))
    bat = K.rprof([(0.0, 0.0), (0.0016, 0.0), (0.0022, 0.016), (0.0030, 0.0185), (0.0018, 0.0215), (0.0, 0.022)],
                  [0, 0, 0.001, 0.001, 0.0006, 0], n=1)
    out.append(K.lathe_at(bat, origin + n * 0.006, d, seg=8, mat=M["switch"]))
    return out


def _knob(origin, fr, radius=0.016, depth=0.016):
    """A heater knob: a fluted dark body on a flared skirt with a rolled face, a lit pointer bar across its face."""
    prof = K.rprof([(0.0, -0.002), (radius + 0.004, -0.002), (radius + 0.004, 0.001), (radius, 0.004), (radius * 0.94, depth),
                    (0.0, depth + 0.0015)], [0, 0.0012, 0.0015, 0.002, 0.004, 0], n=1)
    body = S.lathe(prof, seg=20, mat=M["switch"])
    ang = np.arctan2(body.V[:, 1], body.V[:, 0])
    rad = np.linalg.norm(body.V[:, :2], axis=1)
    flute = (rad > 0.6 * radius) & (body.V[:, 2] > 0.006)
    body.V[:, :2] *= (1.0 + 0.05 * np.cos(10 * ang) * flute)[:, None]
    out = [K.place(body, origin, fr, 0.0)]
    bar = K.slab(S.stadium(0.0022, radius * 1.1, 2), 0.0012, r_top=0.0005, n=1, rings=1, mat=M["gauge_needle"])
    out.append(K.place(bar.moved((0.0, radius * 0.40, 0.0)), origin, fr, depth + 0.0004))
    return out


def _eyeball_vent(origin, fr, radius=0.030):
    """A round vent: a chrome bezel, a dark ball with louvres, turned a little toward the driver."""
    out = [K.place(K.bezel(radius, radius - 0.006, 0.010, seg=24, mat=M["chrome"]), origin, fr, -0.003)]
    ball = S.lathe(np.array([(0.0, -0.015), (radius - 0.006, -0.008), (radius - 0.0055, 0.0), (radius - 0.008, 0.004),
                             (0.0, 0.006)]), seg=20, mat=M["switch"])
    out.append(K.place(ball, origin, fr, 0.0))
    for k in range(-2, 3):
        y = k * (radius - 0.009) / 2.6
        half = math.sqrt(max((radius - 0.008) ** 2 - y * y, 1e-6))
        blade = S.sweep(np.array([(-half, y, 0.006), (half, y, 0.006)]), S.rrect(0.0016, 0.007, 0.0007, 1), up=(0, 0, 1),
                        mat=M["switch"])
        out.append(K.place(K.solid(blade).rot(rx=-20.0, about=(0.0, y, 0.006)), origin, fr, 0.0))
    return out


def _louvre_vent(origin, fr, w, h):
    """A rectangular vent: a drafted frame with a pocket and slats of a thin lens section across it."""
    f = 0.008
    frame_m = S.rounded_frame(S.rrect(w, h, 0.010, 3), S.rrect(w - 2 * f, h - 2 * f, 0.004, 3), 0.010, r_out=0.004,
                              r_in=0.002, n=2, floor=0.008, mat=M["vinyl_dark"], draft=0.002)
    out = [K.place(frame_m, origin, fr, -0.006)]
    iw, ih = w - 2 * f, h - 2 * f
    nslat = max(3, round(ih / 0.011))
    sec = S.ellipse(0.0010, 0.0042, 8)
    t = math.radians(25.0)
    sec = np.stack([sec[:, 0] * math.cos(t) - sec[:, 1] * math.sin(t), sec[:, 0] * math.sin(t) + sec[:, 1] * math.cos(t)], 1)
    for j in range(nslat):
        y = -ih / 2 + ih * (j + 0.5) / nslat
        slat = S.sweep(np.array([(-iw / 2 - 0.001, y, 0.0), (iw / 2 + 0.001, y, 0.0)]), sec, up=(0, 0, 1), mat=M["switch"])
        out.append(K.place(K.solid(slat), origin, fr, -0.001))
    return out


def _stack_controls(b):
    fr = STACK_FR
    for k, s in enumerate(np.linspace(-0.120, 0.120, 6)):
        b.forms += _toggle(stack_point(s, TOGGLES_H), fr, up_deg=18.0 if k % 2 else -14.0)
    for s in (-0.095, 0.0, 0.095):
        b.forms += _knob(stack_point(s, KNOBS_H), fr)


# ====================================================================================================== passenger side
_PF = np.array([p[:2] for p in _FACE["p"]])          # the passenger face's corners (y, z), top to bottom
_pd = _PF[-1] - _PF[0]
PASS_FACE = K.frame((0.0, -_pd[1], _pd[0]), EZ)      # its normal looks back at the passenger and a little down


def pass_face(x, z):
    """The point of the passenger face (its control polygon: flat enough where the panels stand) at (x, z)."""
    return np.array([x, float(np.interp(z, _PF[::-1, 1], _PF[::-1, 0])), z])


def _passenger(b):
    """The passenger side of the dash: a woodgrain band with two louvred vents, the glove box lid with a chrome latch."""
    n = PASS_FACE[2]
    band = S.rounded_panel(S.rrect(0.62, 0.080, 0.022, 3), 0.006, r_edge=0.002, n=2, rings=1, mat=M["woodgrain"])
    b.forms.append(K.place(band, pass_face(-0.575, 1.885), PASS_FACE, -0.0035))
    for x in (-0.40, -0.75):
        b.forms += _louvre_vent(pass_face(x, 1.885) + n * 0.004, PASS_FACE, 0.110, 0.056)
    lid = K.slab(S.rrect(0.40, 0.17, 0.040, 4), 0.012, r_top=0.005, r_bot=0.002, n=2, draft=0.003, dome=0.003, rings=1,
                 mat=M["vinyl_dark"])
    b.forms.append(K.place(lid, pass_face(-0.58, 1.715), PASS_FACE, -0.006))
    latch = S.rounded_panel(S.stadium(0.060, 0.016, 3), 0.006, r_edge=0.0025, n=2, rings=1, dome=0.001, mat=M["chrome"])
    b.forms.append(K.place(latch, pass_face(-0.58, 1.770), PASS_FACE, 0.004))


def _defrost(b):
    """The defrost grille along the foot of the windshield: a low drafted strip on the top's slope, dark slots (decals)."""
    a, c = np.array(_TOP[0][:2]), np.array(_TOP[1][:2])  # the straight run of the top from the front edge up to the crown
    y = a[0] + 0.060
    z = a[1] + (y - a[0]) * (c[1] - a[1]) / (c[0] - a[0])
    nrm = K.unit((0.0, -(c[1] - a[1]), c[0] - a[0]))
    fr = K.frame(nrm, (0.0, 1.0, 0.0))
    strip = K.slab(S.stadium(1.40, 0.040, 4), 0.006, r_top=0.003, n=2, draft=0.003, rings=1, mat=M["vinyl_dark"])
    origin = np.array([0.0, y, z])
    b.forms.append(K.place(strip, origin, fr, -0.003))
    C = origin + np.linspace(-0.66, 0.66, 34)[:, None] * fr[0] + fr[2] * 0.0032
    b.decals.append(K.rects(C, 0.026, 0.0045, fr[0], fr[1], M["seam"]))


# ====================================================================================================== the column
WHEEL_C = np.array(L.WHEEL_C, float)
WHEEL_AXIS = np.array(L.WHEEL_AXIS, float)
R_WHEEL = S.rot_matrix(*L.WHEEL_ROT)


def column_entry():
    """Distance s back along the column axis from the wheel's centre to where it meets the cluster's plane."""
    return float(((GC - WHEEL_C) @ CN) / (-WHEEL_AXIS @ CN))


def _column(b):
    """The column: a cup the wheel's collar turns in, a tapered shroud down into the dash, a rubber boot where it enters,
    the turn-signal stalk on the left and the trailer hand valve on the right."""
    sx = column_entry()
    axis = -WHEEL_AXIS
    shroud = K.rprof([(0.0, 0.092), (0.056, 0.092), (0.058, 0.110), (0.050, 0.150), (0.047, sx - 0.10), (0.050, sx - 0.085)],
                     [0, 0.004, 0.006, 0.02, 0.01, 0], n=2)
    b.forms.append(K.lathe_at(shroud, WHEEL_C, axis, seg=28, mat=M["vinyl_dark"], closed_ends=True))
    rid = []
    for k in range(4):
        s = sx - 0.090 + k * 0.026
        rid += [(0.054 + 0.005 * k, s), (0.062 + 0.005 * k, s + 0.013)]
    boot = K.rprof([(0.0, sx - 0.094)] + rid + [(0.078, sx + 0.040), (0.0, sx + 0.040)],
                   [0.0] + [0.004] * len(rid) + [0.0, 0.0], n=1)
    b.forms.append(K.lathe_at(boot, WHEEL_C, axis, seg=24, mat=M["rubber"]))
    # the stalk: from the shroud's left (+X, local -X of the wheel frame), out and a little back toward the driver
    left = R_WHEEL @ np.array([-1.0, 0.0, 0.0])
    up = R_WHEEL @ np.array([0.0, 1.0, 0.0])
    base = WHEEL_C + axis * 0.135 + left * 0.040
    d = K.unit(left * 0.92 + up * 0.10 - axis * 0.35)
    shaft = K.rprof([(0.0, -0.004), (0.0060, -0.004), (0.0050, 0.120), (0.0062, 0.128), (0.0070, 0.160), (0.0, 0.168)],
                    [0, 0.0015, 0.002, 0.002, 0.003, 0], n=2)
    b.forms.append(K.lathe_at(shaft, base, d, seg=10, mat=M["switch"]))
    b.forms.append(K.lathe_at(K.rprof([(0.0, -0.012), (0.012, -0.010), (0.012, 0.012), (0.0, 0.016)], [0, 0.004, 0.004, 0],
                                      n=2), base, d, seg=12, mat=M["vinyl_dark"]))
    # the trailer brake hand valve on the right: a short lever in a chrome bracket
    right = -left
    vb = WHEEL_C + axis * 0.150 + right * 0.046
    dv = K.unit(right * 0.85 + up * 0.25 - axis * 0.30)
    b.forms.append(K.lathe_at(K.rprof([(0.0, -0.01), (0.014, -0.008), (0.015, 0.018), (0.0, 0.024)], [0, 0.004, 0.005, 0],
                                      n=2), vb, dv, seg=12, mat=M["chrome"]))
    lever = K.rprof([(0.0, 0.010), (0.0045, 0.010), (0.0040, 0.075), (0.0085, 0.082), (0.0080, 0.105), (0.0, 0.110)],
                    [0, 0.001, 0.002, 0.003, 0.003, 0], n=1)
    b.forms.append(K.lathe_at(lever, vb, dv, seg=10, mat=M["switch"]))


# ====================================================================================================== the CB radio
CB_SIZE = (0.170, 0.050, 0.200)                      # the radio's body: width, height, depth
CB_AT = -0.135                                       # its centre up the stack's face
CB_C = stack_point(0.0, CB_AT, 0.008)                # the middle of its front (the card's LOOKS `cb`)
MIC_SIZE = (0.056, 0.080, 0.030)                     # the mic: width, height, thickness
MIC_HOOK = stack_point(-0.128, CB_AT + 0.023, 0.0)   # the hook it hangs on, beside the radio on the driver's side
MIC_C = MIC_HOOK + STACK_FR[2] * (0.010 + MIC_SIZE[2] / 2) - STACK_FR[1] * (MIC_SIZE[1] / 2 - 0.006)
MIC_DIR = STACK_FR[1].copy()                         # it lifts up off its hook


def radio():
    """The radio's body (closed): a dark box-ish shell with a rolled front edge and drafted sides, its front 8 mm proud of
    the stack's face, the rest inside the stack."""
    w, h, d = CB_SIZE
    body = K.slab(S.rrect(w, h, 0.009, 3), d, r_top=0.005, r_bot=0.0, n=2, draft=0.004, rings=1, mat=M["plastic_dark"])
    return K.place(body, stack_point(0.0, CB_AT, 0.008 - d), STACK_FR)


def _cb(b):
    """The CB radio low on the stack: the body, a chrome trim round it, a lit channel window (two digits), three knobs;
    the mic on its hook beside it and the coiled cord between them."""
    r, u, n = STACK_FR
    w, h, _ = CB_SIZE
    b.forms.append(radio())
    trim = S.rounded_frame(S.rrect(w + 0.016, h + 0.016, 0.014, 3), S.rrect(w + 0.001, h + 0.001, 0.0095, 3), 0.006,
                           r_out=0.0025, r_in=0.0012, n=2, mat=M["chrome"])
    b.forms.append(K.place(trim, stack_point(0.0, CB_AT, -0.002), STACK_FR))
    wc = stack_point(-0.034, CB_AT + 0.003, 0.0)      # the channel window: a dark lens with lit seven-segment digits
    lens = S.rounded_panel(S.rrect(0.046, 0.024, 0.003, 2), 0.002, r_edge=0.0008, n=1, rings=1, mat=M["plastic_dark"])
    b.forms.append(K.place(lens, wc, STACK_FR, 0.0075))
    b.decals += _digits(wc + n * 0.0097, r, u, "19", 0.011, 0.016)
    for k, s in enumerate((0.012, 0.037, 0.062)):
        rr = 0.0085 if k < 2 else 0.011
        prof = K.rprof([(0.0, 0.0), (rr, 0.0), (rr, 0.010), (0.0, 0.012)], [0, 0.001, 0.003, 0], n=1)
        b.forms.append(K.lathe_at(prof, stack_point(s, CB_AT + 0.002, 0.007), n, seg=14, mat=M["switch"]))
    hook = K.slab(S.stadium(0.016, 0.030, 3), 0.006, r_top=0.002, r_bot=0.001, n=2, rings=1, mat=M["chrome"])
    b.forms.append(K.place(hook, MIC_HOOK, STACK_FR, -0.002))
    b.forms.append(S.tube(np.array([MIC_HOOK + n * 0.003 + u * 0.008, MIC_HOOK + n * 0.012 + u * 0.011]), 0.0032, sides=8,
                          mat=M["chrome"]))
    b.forms.append(mic())
    b.forms.append(mic_cord())


_SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
        "8": "abcdefg", "9": "abcdfg"}


def _digits(origin, r, u, text, w, h):
    """Lit seven-segment digits (decals, role display) centred at origin in the (r, u) plane."""
    out = []
    t = 0.0022
    for k, ch in enumerate(text):
        cx = (k - (len(text) - 1) / 2) * (w * 1.35)
        x0, x1, yt, yb = cx - w / 2, cx + w / 2, h / 2, -h / 2
        seg = {"a": ((x0, yt), (x1, yt)), "b": ((x1, yt), (x1, 0.0)), "c": ((x1, 0.0), (x1, yb)), "d": ((x0, yb), (x1, yb)),
               "e": ((x0, 0.0), (x0, yb)), "f": ((x0, yt), (x0, 0.0)), "g": ((x0, 0.0), (x1, 0.0))}
        for s in _SEG[ch]:
            (ax, ay), (bx, by) = seg[s]
            c = origin + r * (0.5 * (ax + bx)) + u * (0.5 * (ay + by))
            horiz = abs(ay - by) < 1e-9
            out.append(K.rects([c], (abs(bx - ax) - t) if horiz else t, t if horiz else (abs(by - ay) - t), r, u,
                               M["display"]))
    return out


def mic():
    """The handheld mic hanging on the hook (closed parts): a rounded oval body (its back against the hook), a grille, the
    push-to-talk bar on its side, the strain relief at the bottom where the cord leaves."""
    r, u, n = STACK_FR
    w, h, t = MIC_SIZE
    body = K.slab(S.superellipse(w, h, 3.2, 36), t, r_top=0.010, r_bot=0.010, n=3, dome=0.003, rings=1,
                  mat=M["plastic_dark"])
    out = [K.place(body, MIC_C - n * (t / 2), STACK_FR)]
    grille = S.rounded_panel(S.superellipse(w * 0.62, h * 0.42, 3.0, 24), 0.0025, r_edge=0.001, n=1, rings=1, dome=0.001,
                             mat=M["switch"])
    out.append(K.place(grille, MIC_C + u * (h * 0.14) + n * (t / 2 - 0.0005), STACK_FR))
    ptt = K.slab(S.stadium(0.010, 0.040, 3), 0.006, r_top=0.003, r_bot=0.001, n=2, rings=1, mat=M["switch"])
    out.append(K.place(ptt, MIC_C - r * (w / 2 - 0.004) + u * 0.006, K.frame(-r, u)))
    relief = K.rprof([(0.0, 0.0), (0.0060, 0.0), (0.0050, 0.012), (0.0036, 0.022), (0.0, 0.024)], [0, 0, 0.002, 0.001, 0],
                     n=1)
    out.append(K.lathe_at(relief, MIC_C - u * (h / 2 - 0.010), -u, seg=10, mat=M["rubber"]))
    return S.merge(out)


def mic_cord_ends():
    """(start, end) of the coiled cord's centre line: inside the radio's left end, and inside the mic's strain relief."""
    u = STACK_FR[1]
    a = stack_point(-CB_SIZE[0] / 2 + 0.008, CB_AT - 0.014, -0.010)
    z = MIC_C - u * (MIC_SIZE[1] / 2 + 0.006)
    return a, z


def mic_cord_path():
    """The cord's centre line: out of the radio's left end, a coiled loop hanging toward the floor, up into the mic's foot."""
    r, u, n = STACK_FR
    a, z = mic_cord_ends()
    ctrl = np.array([a, a - r * 0.022 + n * 0.014 - u * 0.006, stack_point(-0.106, CB_AT - 0.090, 0.032),
                     stack_point(-0.120, CB_AT - 0.150, 0.042), stack_point(-0.134, CB_AT - 0.110, 0.034), z - u * 0.020, z])
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    return S.pchip(t, ctrl, np.linspace(0.0, t[-1], 64))


def mic_cord():
    return K.helix_tube(mic_cord_path(), 0.0070, 0.0095, 0.0021, turn_pts=6, sides=5, mat=M["plastic_dark"],
                        up=tuple(STACK_FR[2]))


# ====================================================================================================== the module
class Bag:
    """What the dash builds: `forms` (closed parts), `decals` (graphic layers), `gauges` (name, dial centre, normal, r)."""

    def __init__(self):
        self.forms, self.decals, self.gauges = [], [], []


def dash_parts():
    b = Bag()
    b.forms.append(dash_body())
    b.forms += _cluster_pod()
    b.forms.append(_stack_body())
    b.forms += _stack_panels()
    _cluster_gauges(b)
    _indicators(b)
    _stack_controls(b)
    b.forms += _eyeball_vent(cluster_point(*VENT_UV), CLUSTER_FR, VENT_R)
    _passenger(b)
    _defrost(b)
    _column(b)
    _cb(b)
    return b
