"""Running gear of the 1980s convertible (pure numpy, no bpy).

dynamic_parts()   wheel_FL, wheel_FR, wheel_RL, wheel_RR: one mesh each, in the wheel's own frame. Origin = the axle
                  centre (the Part origin), spin axis = local X, the outer face looks toward +X. The -X wheels are the
                  exact mirror images, so the same spin angle about local +X rolls all four wheels forward (-Y). The
                  lowest point of every tyre is at car z = 0. A wheel is: the tyre (195/60R15: rounded shoulders, a
                  slightly bulging blank sidewall, a flat tread with three shallow grooves), the rim barrel with its
                  lip, a recessed face of twelve curved, pitched turbine blades that spiral from a hub ring to the
                  barrel, a chrome hub cap with five lug nuts, and a dark brake rotor and backing plate seen between
                  the blades.

Roles: tyre, alloy (barrel, blades, hub ring), underbody (rotor, backing), chrome (cap, nuts).
"""
import math

import numpy as np

from ....core import shell as S
from . import convertible_layout as L

M = L.M

# ===================================================================================================================
# wheel (local frame: origin on the axle, spin axis X, outer face toward +X)
# ===================================================================================================================
SEG_TYRE, SEG_RIM, SEG_S = 60, 72, 36   # segments around the tyre, the rim barrel and the smaller round parts
R_SEAT = L.RIM_R                     # bead seat = the tyre's inner radius
R_FLANGE = R_SEAT + 0.017            # top of the rim lip
R_DISH = R_SEAT - 0.0045             # inside of the barrel: the wall of the dish the blades stand in
X_LIP, X_FLANGE = 0.088, 0.079       # outer face of the lip, back face of the flange the tyre bead rests on

N_BLADES = 12
R_ROOT, R_TIP = 0.072, R_SEAT - 0.0020          # blade ends: buried in the hub ring / in the barrel wall
X_BLADE = (0.054, 0.058)                        # axial centre of the blade sections: root, tip
BLADE_T = 0.0036                                # plate thickness
SWIRL_DEG = 52.0                                # how far the tip lags behind the root (the spiral)
PITCH_DEG = (50.0, 34.0)                        # chord angle from the wheel plane: root, tip (the twist)
COVER = (0.64, 0.58)                            # tangential cover of the blades at root / tip (the rest is gap)

X_PLATE = 0.058                      # hub plate face: the cap and the lug nuts sit on it
X_RING, X_HUB0 = 0.076, 0.046        # top of the hub ring, back of the hub
R_RING = (0.062, 0.080)              # hub ring inner / outer radius
R_BOLT = 0.0475                      # lug nut circle


def _revolve(profile, seg, mat, closed_ends=False):
    """Surface of revolution about the wheel's X axis: `profile` = (radius, x) points, outside on the right."""
    return S.lathe(profile, seg, mat=mat, closed_ends=closed_ends).rot(ry=90.0)


def _revolve_loop(loop, seg, mat):
    """Closed (radius, x) outline (counter-clockwise, outside on the right) revolved into a closed solid. The first
    point is repeated at the end and that last ring is folded onto ring 0, so doubled points inside the outline
    (creases) stay doubled."""
    P = np.asarray(loop, float)
    assert P[:, 0].min() > 1e-6, "no point on the axis: every ring must have `seg` vertices"
    m = S.lathe(np.vstack([P, P[:1]]), seg, mat=mat)
    n = len(P) * seg
    return S.Mesh(m.V[:n], np.where(m.Q >= n, m.Q - n, m.Q), np.where(m.T >= n, m.T - n, m.T), m.Qm, m.Tm).rot(ry=90.0)


def _tyre():
    """Tyre solid: sidewall knots (radius above the rim, half width) from the tread edge to the bead, mirrored."""
    kr = (L.TYRE_R - L.RIM_R) / 0.1195
    kx = L.TYRE_W / 2 / 0.1025
    K = np.array([(0.1195, 0.0720), (0.1165, 0.0840), (0.1080, 0.0935), (0.0945, 0.0990), (0.0775, 0.1018),
                  (0.0615, 0.1025), (0.0455, 0.1002), (0.0325, 0.0955), (0.0225, 0.0900), (0.0170, 0.0830),
                  (0.0145, 0.0800), (0.0000, 0.0775)])
    K = np.stack([L.RIM_R + K[:, 0] * kr, K[:, 1] * kx], 1)
    R = L.TYRE_R
    xt = K[0, 1]
    tread = [(R, -xt)]
    for g in np.array([-0.047, 0.0, 0.047]) * kx:                       # three shallow circumferential grooves
        a, b = (R, g - 0.0045 * kx), (R, g + 0.0045 * kx)               # doubled edge points: crisp groove edges
        tread += [a, a, (R - 0.004, g), b, b]
    tread.append((R, xt))
    neg = [(r, -x) for r, x in K[:0:-1]]                                # bead heel (-x side) ... shoulder
    pos = [(r, x) for r, x in K[1:]]                                    # shoulder ... bead heel (+x side)
    return _revolve_loop(neg + tread + pos, SEG_TYRE, M["tyre"])


def _rim():
    """Barrel with both lips: a thin closed shell, the visible part is the lip face and the inside of the barrel."""
    xl, xs = X_LIP, X_FLANGE
    P = [(R_FLANGE, xs), (R_FLANGE, xl), (R_DISH, xl), (R_DISH, -xl), (R_FLANGE, -xl), (R_FLANGE, -xs),
         (R_SEAT, -xs), (R_SEAT, xs)]
    loop = S.fillet(P, [0.0, 0.004, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0], n=2, closed=True)   # only the outer lip is seen
    return _revolve_loop(loop, SEG_RIM, M["alloy"])


def _blade():
    """One turbine blade: a thin plate whose sections lie on cylinders about the axle (so both ends sit flush inside the
    hub ring and the barrel wall), pitched against the wheel plane (more at the root: the twist) and spiralling backward
    toward the rim."""
    n = 11
    s = np.linspace(0.0, 1.0, n)
    r = R_ROOT + (R_TIP - R_ROOT) * s
    th = -np.radians(SWIRL_DEG) * s ** 1.3
    phi = np.radians(PITCH_DEG[0] + (PITCH_DEG[1] - PITCH_DEG[0]) * s)
    width = (COVER[0] + (COVER[1] - COVER[0]) * s) * 2 * math.pi * r / N_BLADES          # tangential extent
    chord = (width - BLADE_T * np.sin(phi)) / np.cos(phi)
    x_c = X_BLADE[0] + (X_BLADE[1] - X_BLADE[0]) * s
    secs = []
    for i in range(n):
        sec = S.rrect(chord[i], BLADE_T, 0.0010, n=1)                    # (8, 2): along the chord, across it
        a = sec[:, 0] * math.cos(phi[i]) - sec[:, 1] * math.sin(phi[i])  # tangential arc length
        b = sec[:, 0] * math.sin(phi[i]) + sec[:, 1] * math.cos(phi[i])  # axial
        ang = th[i] + a / r[i]
        secs.append(np.stack([x_c[i] + b, r[i] * np.cos(ang), r[i] * np.sin(ang)], 1))
    return S.skin(np.array(secs), caps=(True, True), closed=True, mat=M["alloy"])


def _blades():
    b = _blade()
    return S.merge([b.rot(rx=360.0 * k / N_BLADES) for k in range(N_BLADES)])


def _hub():
    """Hub plate with the ring the blades grow from (alloy)."""
    P = np.array([(0.0, X_HUB0), (R_RING[1], X_HUB0), (R_RING[1], X_RING), (R_RING[0], X_RING),
                  (R_RING[0], X_PLATE), (0.0, X_PLATE)])
    prof = S.fillet(P, [0.0, 0.002, 0.004, 0.003, 0.003, 0.0], n=2, closed=False)
    return _revolve(prof, SEG_S, M["alloy"])


def _cap():
    """Chrome hub cap (a low dome) and the five lug nuts around it."""
    x = X_PLATE - 0.0005
    cap = _revolve([(0.0, x), (0.0305, x), (0.0305, X_PLATE + 0.0045), (0.0270, X_PLATE + 0.0120),
                    (0.0205, X_PLATE + 0.0170), (0.0120, X_PLATE + 0.0198), (0.0, X_PLATE + 0.0208)], 24, M["chrome"])
    nut = _revolve([(0.0105, 0.0), (0.0105, 0.012), (0.0085, 0.0165), (0.0, 0.0175)], 6, M["chrome"], closed_ends=True)
    nuts = []
    for k in range(5):
        a = 2 * math.pi * k / 5 + math.pi / 2
        nuts.append(nut.moved((x, R_BOLT * math.cos(a), R_BOLT * math.sin(a))))
    return S.merge([cap] + nuts)


def _rotor():
    """Dark backing plate, brake rotor and hat: what shows between the blades (underbody)."""
    P = [(0.0, 0.012), (R_DISH + 0.0015, 0.012), (R_DISH + 0.0015, 0.024), (0.150, 0.024), (0.150, 0.030),
         (0.090, 0.030), (0.090, X_HUB0 + 0.002), (0.0, X_HUB0 + 0.002)]
    return _revolve(P, SEG_S, M["underbody"])


def wheel_mesh():
    """One wheel on the +X side of the car (outer face toward +X), in its own frame."""
    return S.merge([_tyre(), _rim(), _blades(), _hub(), _cap(), _rotor()])


def dynamic_parts():
    left = wheel_mesh()
    right = left.mirrored_x()
    z = L.WHEEL_Z
    return {
        "wheel_FL": S.Part(left, (L.WHEEL_X, L.AXLE_F, z), (0.0, 0.0, 0.0)),
        "wheel_FR": S.Part(right, (-L.WHEEL_X, L.AXLE_F, z), (0.0, 0.0, 0.0)),
        "wheel_RL": S.Part(left.copy(), (L.WHEEL_X, L.AXLE_R, z), (0.0, 0.0, 0.0)),
        "wheel_RR": S.Part(right.copy(), (-L.WHEEL_X, L.AXLE_R, z), (0.0, 0.0, 0.0)),
    }
