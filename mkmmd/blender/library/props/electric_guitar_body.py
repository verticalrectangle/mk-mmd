"""The guitar's body and pickguard (pure numpy, no bpy): a contoured double-cutaway slab and the 3-ply guard on it.

The body is one closed plate (`electric_guitar_geo.slab`): the outline is a spline through the control points of
`electric_guitar_layout`, the rim is rolled over (4 mm radius, 4 segments, front and back), and the two flat faces are
height functions: the FRONT face `face(x, z)` is y = 0 except the forearm bevel (a scoop along the bass edge, where the picking
arm rests), the BACK face `back(x, z)` is y = BODY_T except the belly cut (a scoop on the bass side of the back). The
pickguard is a second, thin slab that follows `face`.
"""
import numpy as np

from . import electric_guitar_geo as G
from . import electric_guitar_layout as L

_CACHE = {}


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def outline():
    """The body's outline polygon (m, 2) of (x, z): counter-clockwise, 4 mm between points."""
    if "outline" not in _CACHE:
        pts, corners = L.body_control_points()
        _CACHE["outline"] = G.ccw(G.spline(pts, corners, spacing=0.004))
    return _CACHE["outline"]


def edge_distance(x, z):
    """Distance from points (x, z) to the body's outline (inside or out)."""
    x = np.asarray(x, float)
    return G.distance(np.column_stack([x.ravel(), np.asarray(z, float).ravel()]), outline()).reshape(x.shape)


def forearm(x, z):
    """How far the front face is scooped back (m, >= 0) by the forearm bevel: deepest at the bass edge between the waist and
    the lower bout, fading to nothing `FOREARM['width']` inside it and along z."""
    f = L.FOREARM
    z0, z1, z2, z3 = f["z"]
    gate = smoothstep((z - z0) / (z1 - z0)) * (1.0 - smoothstep((z - z2) / (z3 - z2)))
    side = smoothstep(-x / 0.040)                           # bass side only
    return f["depth"] * smootherstep(1.0 - edge_distance(x, z) / f["width"]) * gate * side


def belly(x, z):
    """How far the back face is scooped toward the front (m, >= 0) by the belly cut."""
    b = L.BELLY
    (cx, cz), (a, c) = b["center"], b["radii"]
    q = np.sqrt(((x - cx) / a) ** 2 + ((z - cz) / c) ** 2)
    return b["depth"] * smootherstep(1.0 - q)


def face(x, z):
    """y of the body's front face at (x, z): 0, deeper (+y) in the forearm bevel."""
    return forearm(np.asarray(x, float), np.asarray(z, float))


def back(x, z):
    """y of the body's back face at (x, z): BODY_T, shallower (-y) in the belly cut."""
    return L.BODY_T - belly(np.asarray(x, float), np.asarray(z, float))


def body_mesh():
    """The body: one closed mesh, role `body` (the gloss coat), in the prop's frame."""
    return G.slab(outline(), face, back, r=L.BODY_EDGE_R, n=4, wall=2, spacing=0.009, mat=L.MATS["body"])


def guard_outline():
    pts = np.array(L.GUARD, float)
    return G.ccw(G.spline(pts, L.GUARD_CORNERS, spacing=0.004))


def guard_mesh():
    """The pickguard: 3 plies (white, a dark core you see on the bevel, white) lying on the body's face."""
    m = L.MATS
    top = lambda x, z: face(x, z) - L.GUARD_T                                     # noqa: E731
    bottom = lambda x, z: face(x, z) + 0.0003                                     # noqa: E731
    return G.slab(guard_outline(), top, bottom, r=0.0009, n=1, wall=3, spacing=0.010, mat=m["pickguard"],
                  mat_wall=(m["pickguard"], m["guard_core"], m["pickguard"]), mat_back=m["pickguard"])
