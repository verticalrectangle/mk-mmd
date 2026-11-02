"""semi_truck_gear: the running gear and the frame of the semi truck (pure numpy, no bpy).

dynamic_parts()   the ten wheels of L.wheel_hubs() (semi_truck_wheels)
static_parts()    {"frame": the ladder frame (two C-channel rails and their crossmembers) and the fifth wheel; "axles": the
                   steer axle's I-beam on its leaf springs, the drive tandem's housings, air bags and trailing arms, the
                   drive shaft; "flaps": the mud flaps behind the tandem on their chrome hangers}

Everything stands where semi_truck_layout says (FRAME_X, FRAME_Z, FRAME_Y, AXLE_S_Y, AXLE_D_Y, KINGPIN_Y, FIFTH_TOP_Z).
"""
import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L
from . import semi_truck_wheels as W

M = L.M


def dynamic_parts():
    return W.dynamic_parts()


def _box(size, center, mat, r=0.008):
    return S.rounded_box(tuple(float(s) for s in size), center=tuple(float(c) for c in center), r=r, mat=mat)


def _tube(a, b, radius, mat, sides=16):
    m = S.tube(np.array([a, b], float), radius, sides=sides, mat=mat,
               up=(0.0, 0.0, 1.0) if abs(b[2] - a[2]) < 0.9 * np.linalg.norm(np.subtract(b, a)) else (1.0, 0.0, 0.0))
    return m if m.volume() > 0 else m.flip()


def frame():
    """The two C-channel rails (web outboard, flanges in), five crossmembers and the fifth wheel on its brackets."""
    y0, y1 = L.FRAME_Y
    z0, z1 = L.FRAME_Z
    ly, yc = y1 - y0, (y0 + y1) / 2
    t, fl, h = L.RAIL_T, L.RAIL_FLANGE, L.RAIL_H
    parts = []
    for sx in (1.0, -1.0):
        xw = sx * (L.FRAME_X - t / 2)
        parts.append(_box((t, ly, h), (xw, yc, (z0 + z1) / 2), M["chassis"], r=0.002))
        for zf in (z0 + t / 2, z1 - t / 2):
            parts.append(_box((fl, ly, t), (sx * (L.FRAME_X - fl / 2), yc, zf), M["chassis"], r=0.002))
    inner = 2 * (L.FRAME_X - t)
    for y in (y0 + 0.05, -5.60, -3.10, -1.40, y1 - 0.05):
        parts.append(_box((inner, 0.10, h - 0.06), (0.0, y, (z0 + z1) / 2), M["chassis"], r=0.004))
    top = L.FIFTH_TOP_Z
    parts.append(_box((2 * L.FIFTH_HALF_W, 0.86, 0.06), (0.0, L.KINGPIN_Y + 0.08, top - 0.03), M["chassis"], r=0.02))
    for sx in (1.0, -1.0):                                          # the plate's brackets down to the rails
        parts.append(_box((0.10, 0.42, top - 0.06 - z1), (sx * 0.36, L.KINGPIN_Y + 0.08, (top - 0.06 + z1) / 2),
                          M["chassis"], r=0.01))
    return S.merge(parts)


def axles():
    """The steer axle (a drop-centre I-beam) on its leaf springs, each drive axle's housing and centre bowl with an air
    bag and a trailing arm each side, the drive shaft from the transmission back to the tandem."""
    parts = []
    zs = L.WHEEL_Z
    half = L.STEER_TRACK / 2 - 0.16                                  # the beam's ends, inside the hubs' spindles
    parts.append(_box((2 * half, 0.10, 0.12), (0.0, L.AXLE_S_Y, zs - 0.02), M["chassis"], r=0.015))
    z_spring = (zs + 0.04 + L.FRAME_Z[0]) / 2
    for sx in (1.0, -1.0):
        parts.append(_box((0.08, 1.30, L.FRAME_Z[0] - zs - 0.04), (sx * (L.FRAME_X - 0.05), L.AXLE_S_Y, z_spring),
                          M["chassis"], r=0.015))
    for yd in L.AXLE_D_Y:
        hx = L.DRIVE_TRACK / 2 - 0.22
        parts.append(_tube((-hx, yd, zs), (hx, yd, zs), 0.075, M["chassis"]))
        bowl = S.rounded_box((0.40, 0.36, 0.40), center=(0.0, yd, zs), r=0.15, mat=M["chassis"])
        parts.append(bowl)
        for sx in (1.0, -1.0):
            parts.append(_box((0.09, 0.95, 0.09), (sx * (L.FRAME_X - 0.05), yd - 0.30, zs + 0.12), M["chassis"], r=0.02))
            bag = S.tube(np.array([[sx * (L.FRAME_X - 0.05), yd + 0.30, zs + 0.17],
                                   [sx * (L.FRAME_X - 0.05), yd + 0.30, L.FRAME_Z[0] - 0.01]]), 0.12, sides=20,
                         mat=M["rubber"], up=(1.0, 0.0, 0.0))
            parts.append(bag if bag.volume() > 0 else bag.flip())
    parts.append(_tube((0.0, -5.10, L.FRAME_Z[0] - 0.02), (0.0, L.AXLE_D_Y[0] - 0.20, zs + 0.06), 0.055, M["chassis"]))
    return S.merge(parts)


def flaps():
    """A mud flap behind each side of the tandem, hanging from a chrome bar out of the frame."""
    y = L.AXLE_D_Y[1] + 0.62
    xc = L.DRIVE_TRACK / 2
    parts = []
    for sx in (1.0, -1.0):
        parts.append(_box((0.62, 0.012, 0.62), (sx * xc, y, 0.56), M["mudflap"], r=0.004))
        parts.append(_tube((sx * L.FRAME_X, y - 0.02, 0.90), (sx * (xc + 0.32), y - 0.02, 0.90), 0.022, M["chrome"]))
    return S.merge(parts)


def static_parts():
    return {"frame": frame(), "axles": axles(), "flaps": flaps()}
