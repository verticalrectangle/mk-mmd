"""semi_truck_trailer: the 53 ft dry van (pure numpy, no bpy; the builder is semi_truck.py, the numbers
semi_truck_layout.py). A composite plate van with an extruded aluminium frame, swing doors and its underside.

Static groups (one object `<name>_<group>` each, all parented to `<name>_trailer` at the kingpin):
    trailer_box      the composite side walls (outer faces inside SIDE_PANEL in mural_left / mural_right), the front
                     wall and the crowned aluminium roof
    trailer_frame    the top rail round the van (its front part the nose cap), the bottom rail and front sill, the
                     front corner caps, the rear corner posts, header and sill, the joint strips and the rivet heads
    trailer_doors    the two swing doors (outer faces inside BACK_PANEL in mural_back), their hinges, lock rods and
                     seals (semi_truck_trailer_doors)
    trailer_under    the floor's underside, the sliding tandem's rails, axles and springs, the landing gear, the rear
                     (ICC) bumper, the tail lamps and the mud flaps
"""
import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L
from . import semi_truck_trailer_box as BOX
from . import semi_truck_trailer_doors as DOORS

M = L.M
Y0, Y1 = L.TRAILER_Y
HW = L.TRAILER_HALF_W

LOOKS = {
    "trailer_doors": (0.0, Y1, 2.6),
    "trailer_front": (0.0, Y0, 2.6),
    "trailer_side_L": (HW, 0.5 * (Y0 + Y1), 2.7),
    "trailer_side_R": (-HW, 0.5 * (Y0 + Y1), 2.7),
}


def _box(size, center, mat, r=0.006):
    return S.rounded_box(tuple(float(s) for s in size), center=tuple(float(c) for c in center), r=r, mat=mat)


def _tube(a, b, radius, mat, sides=14):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    up = (1.0, 0.0, 0.0) if abs(d[2]) > 0.9 * np.linalg.norm(d) else (0.0, 0.0, 1.0)
    m = S.tube(np.array([a, b]), radius, sides=sides, mat=mat, up=up)
    return m if m.volume() > 0 else m.flip()


def under():
    """The van's underside, every piece where the layout puts its place (SLIDER_*, TRAILER_AXLE_Y, LANDING_*)."""
    zf = L.FIFTH_TOP_Z
    parts = [_box((2 * HW - 0.06, Y1 - Y0 - 0.30, 0.10), (0.0, (Y0 + Y1) / 2 + 0.05, zf + 0.05), M["trailer_under"], r=0.01)]
    s0, s1 = L.SLIDER_Y
    zs0, zs1 = L.SLIDER_Z
    for sx in (1.0, -1.0):
        parts.append(_box((0.10, s1 - s0, zs1 - zs0), (sx * L.SLIDER_X, (s0 + s1) / 2, (zs0 + zs1) / 2), M["trailer_under"]))
    for ya in L.TRAILER_AXLE_Y:
        hx = L.TRAILER_TRACK / 2 - 0.20
        parts.append(_tube((-hx, ya, L.WHEEL_Z), (hx, ya, L.WHEEL_Z), 0.065, M["chassis"]))
        for sx in (1.0, -1.0):
            parts.append(_box((0.08, 0.95, 0.07), (sx * L.SLIDER_X, ya - 0.25, L.WHEEL_Z + 0.10), M["chassis"], r=0.015))
            bag = S.tube(np.array([[sx * L.SLIDER_X, ya + 0.28, L.WHEEL_Z + 0.12], [sx * L.SLIDER_X, ya + 0.28, zs0 - 0.01]]),
                         0.11, sides=18, mat=M["rubber"], up=(1.0, 0.0, 0.0))
            parts.append(bag if bag.volume() > 0 else bag.flip())
    for sx in (1.0, -1.0):                                           # the landing gear: two legs, sand shoes, a brace
        x = sx * L.LANDING_X
        parts.append(_box((0.10, 0.10, zf - 0.45), (x, L.LANDING_Y, (zf + 0.45) / 2), M["trailer_under"], r=0.01))
        parts.append(_box((0.075, 0.075, 0.40), (x, L.LANDING_Y, 0.30), M["trailer_under"], r=0.008))
        parts.append(_box((0.26, 0.30, 0.04), (x, L.LANDING_Y, 0.07), M["trailer_under"], r=0.01))
        parts.append(_tube((x, L.LANDING_Y, 0.85), (sx * 0.15, L.LANDING_Y + 0.55, zf - 0.02), 0.025, M["trailer_under"]))
    parts.append(_tube((-L.LANDING_X, L.LANDING_Y, 0.95), (L.LANDING_X, L.LANDING_Y, 0.95), 0.03, M["trailer_under"]))
    parts.append(_box((0.16, 0.10, 0.20), (L.LANDING_X + 0.13, L.LANDING_Y, 0.95), M["trailer_under"], r=0.02))
    yb = Y1 - 0.30                                                   # the rear (ICC) bumper under the doors
    parts.append(_box((2 * HW - 0.30, 0.10, 0.10), (0.0, yb, 0.56), M["trailer_metal"], r=0.015))
    for sx in (1.0, -1.0):
        parts.append(_box((0.08, 0.08, zf - 0.56), (sx * 0.80, yb + 0.02, (zf + 0.56) / 2), M["trailer_metal"], r=0.01))
        parts.append(_box((0.62, 0.012, 0.62), (sx * L.TRAILER_TRACK / 2, L.TRAILER_AXLE_Y[1] + 0.70, 0.55), M["mudflap"],
                          r=0.004))
        for x in (1.02, 0.80):                                       # two round tail lamps each side, on the rear sill
            lamp = S.tube(np.array([[sx * x, Y1 - 0.02, 1.27], [sx * x, Y1 + 0.012, 1.27]]), 0.055, sides=20,
                          mat=M["lamp_red"], up=(0.0, 0.0, 1.0))
            parts.append(lamp if lamp.volume() > 0 else lamp.flip())
    return S.merge(parts)


def static_parts():
    w = BOX.walls()
    return {"trailer_box": w["box"], "trailer_frame": w["frame"], "trailer_doors": S.merge(DOORS.doors()),
            "trailer_under": under()}


def _merged(groups):
    return {g: S.merge(m) if isinstance(m, list) else m for g, m in groups.items()}
