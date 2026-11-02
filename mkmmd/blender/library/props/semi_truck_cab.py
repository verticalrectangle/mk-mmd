"""The semi truck's cab inside (pure numpy, no bpy): the steering wheel (a moving part) and the cab's furniture, built in
`semi_truck_cab_dash` (the wrap-around dash, gauges, centre stack, column, CB radio and mic), `semi_truck_cab_seats`
(the air-ride seats) and `semi_truck_cab_trim` (doors, headliner, visors, back wall, curtain, floor, pedals, shifter,
horn cord, dome light), on the numbers of `semi_truck_cab_kit` (the hollow `semi_truck_body` leaves, the driver's eye).

    static_parts(probes)   {"cab_dash", "cab_seats", "cab_trim": forms, "cab_decals": graphic layers}; with the cab shell's
                           probe (probes["cab"]) the headliner is hung just under the evaluated ceiling
    dynamic_parts()        {"steering_wheel": Part}: the wheel in its own frame (rim in the local XY plane round the origin,
                           radius L.WHEEL_R to the tube's centre, tube L.WHEEL_TUBE, +Z the column axis toward the driver, +Y
                           12 o'clock, +X 3 o'clock as the driver sees it), origin L.WHEEL_C, base rotation L.WHEEL_ROT
    LOOKS, ANCHORS         `cb` (the radio), `gauges`; `cb_mic` (the mic on its hook, dir: up off the hook), `shifter`
                           (the knob, dir: up the lever), `horn_cord` (the pull handle, dir: the pull, down)

The wheel is a big truck wheel: a round smooth rim (role steering), four chrome spokes all in the lower half (at about
3:45, 5:15, 6:45 and 8:15 as the driver sees it: nothing where the hands ride at 10 and 2, and the gauges stay in view
through the open upper half), a dished hub with a big round horn pad (no mark) in a chrome ring, and the collar that turns
in the column's cup.
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_cab_dash as D
from . import semi_truck_layout as L

M = L.M

# ====================================================================================================== the wheel
WHEEL_C = np.array(L.WHEEL_C, float)
R_WHEEL = S.rot_matrix(*L.WHEEL_ROT)                  # the wheel's own frame -> the truck's
SPOKES_DEG = (-22.0, 202.0, -68.0, 248.0)            # spoke angles from 3 o'clock, counter-clockwise as the driver sees it
HUB_DISH = 0.050                                     # the spokes' roots sit this far behind the rim's plane
HUB_R = 0.072                                        # the hub's widest radius (its shadow on the cluster stays small)


def wheel_rim():
    """The rim: a round smooth tube on the ring (radius WHEEL_R to the tube's centre, tube WHEEL_TUBE), role steering."""
    return S.tube(S.torus_path(L.WHEEL_R, n=80), L.WHEEL_TUBE, sides=14, closed=True, mat=M["steering"]).weld(1e-9)


def _spoke(deg):
    """A chrome blade from the hub up into the rim at `deg`: a little wider and thinner toward the rim, its end flaring
    into the tube and buried in it."""
    n = 12
    t = np.linspace(0.0, 1.0, n)
    sm = t * t * (3.0 - 2.0 * t)
    r = 0.052 + (L.WHEEL_R - 0.052) * t
    z = -HUB_DISH * (1.0 - sm)
    a = math.radians(deg)
    path = np.stack([r * math.cos(a), r * math.sin(a), z], 1)
    width = 0.030 + 0.008 * t + 0.014 * np.clip((t - 0.80) / 0.20, 0.0, 1.0) ** 2
    thick = 0.013 - 0.004 * sm
    blade = S.sweep(path, S.rrect(1.0, 1.0, 0.32, 3), scale=np.stack([width, thick], 1), up=(0.0, 0.0, 1.0),
                    mat=M["chrome"])
    return blade if blade.volume() > 0 else blade.flip()


def wheel_mesh():
    """The steering wheel in its own frame: the rim, the four spokes, the dished hub with its horn pad and chrome ring, the
    collar behind."""
    parts = [wheel_rim()] + [_spoke(d) for d in SPOKES_DEG]
    hub = S.fillet(np.array([(0.0, -0.118), (0.040, -0.118), (0.040, -0.082), (HUB_R, -0.058), (HUB_R + 0.002, -0.030),
                             (HUB_R - 0.004, -0.012), (0.0, -0.012)]), [0.0, 0.004, 0.010, 0.012, 0.008, 0.0, 0.0], n=3)
    parts.append(S.lathe(hub, seg=36, mat=M["vinyl_dark"]))
    pad = S.rounded_panel(S.superellipse(0.120, 0.114, 2.4, 48), 0.022, r_edge=0.009, n=3, dome=0.010, rings=2,
                          mat=M["steering"])
    parts.append(pad.moved((0.0, 0.0, -0.020)))
    ring = S.rounded_frame(S.superellipse(0.132, 0.126, 2.4, 48), S.superellipse(0.118, 0.112, 2.4, 48), 0.006,
                           r_out=0.0022, r_in=0.0018, n=2, mat=M["chrome"])
    parts.append(ring.moved((0.0, 0.0, -0.016)))
    return S.merge(parts)


# ====================================================================================================== the module
def static_parts(probes=None):
    b = D.dash_parts()
    return {"cab_dash": S.merge(b.forms), "cab_decals": S.merge(b.decals)}


def dynamic_parts():
    return {"steering_wheel": S.Part(wheel_mesh(), tuple(float(v) for v in WHEEL_C), tuple(float(v) for v in L.WHEEL_ROT))}


def _lst(v):
    return [float(x) for x in v]


LOOKS = {"cb": _lst(D.CB_C), "gauges": _lst(D.GC)}
ANCHORS = {"cb_mic": {"point": _lst(D.MIC_C), "dir": _lst(D.MIC_DIR)}}
