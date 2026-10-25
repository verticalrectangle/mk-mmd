"""The bass's body, pickguard, control plate and racing stripes (pure numpy, no bpy): an offset slab with rounded horns, the
guard and the chrome plate on it, and the optional Competition stripes as a graphic layer.

The body is one closed plate (`electric_guitar_geo.slab`, the guitar's toolkit): the outline is a spline through the control
points of `electric_bass_layout`, the rim is rolled over (5 mm radius, 4 segments, front and back), and the two flat faces are
height functions: the FRONT face `face(x, z)` is y = 0 except the forearm bevel (a scoop along the bass edge of the lower bout,
where the picking arm rests), the BACK face `back(x, z)` is y = BODY_T except the belly cut (a scoop on the bass side of the
back, under the waist). The pickguard (three plies) and the control plate are thin slabs that follow `face`. The stripes are a
ribbon of three bands clipped to the body's outline, lying a fraction of a millimetre over the face and the rolled edge.
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_bass_layout as L
from . import electric_guitar_geo as G

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
        _CACHE["outline"] = G.ccw(G.spline(L.body_control_points(), (), spacing=0.004))
    return _CACHE["outline"]


def edge_distance(x, z):
    """Distance from points (x, z) to the body's outline (inside or out)."""
    x = np.asarray(x, float)
    return G.distance(np.column_stack([x.ravel(), np.asarray(z, float).ravel()]), outline()).reshape(x.shape)


def forearm(x, z):
    """How far the front face is scooped back (m, >= 0) by the forearm bevel: deepest at the bass edge of the lower bout,
    fading to nothing `FOREARM['width']` inside it and along z."""
    f = L.FOREARM
    z0, z1, z2, z3 = f["z"]
    gate = smoothstep((z - z0) / (z1 - z0)) * (1.0 - smoothstep((z - z2) / (z3 - z2)))
    side = smoothstep((-x - 0.06) / 0.04)                    # the bass edge only
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


# ===================================================================================================================
# the pickguard and the control plate
# ===================================================================================================================
def guard_outline():
    pts, corners = L.guard_control_points()
    return G.ccw(G.spline(pts, corners, spacing=0.004))


def guard_mesh():
    """The pickguard: 3 plies (the guard's colour, a dark core you see on the bevel, the guard's colour) lying on the face."""
    m = L.MATS
    top = lambda x, z: face(x, z) - L.GUARD_T                                     # noqa: E731
    bottom = lambda x, z: face(x, z) + 0.0003                                     # noqa: E731
    return G.slab(guard_outline(), top, bottom, r=0.0008, n=1, wall=3, spacing=0.010, mat=m["pickguard"],
                  mat_wall=(m["pickguard"], m["guard_core"], m["pickguard"]), mat_back=m["pickguard"])


def guard_screw_points(inset=0.0040):
    """(n, 2) where the guard's screws stand: each of `GUARD_SCREWS` (read off the photograph) moved onto the guard, `inset`
    in from its nearest edge point."""
    o = guard_outline()
    M = S.mitre_offsets(o)
    out = []
    for x, z in L.GUARD_SCREWS:
        i = int(np.argmin(np.hypot(o[:, 0] - L.X_SCALE * x, o[:, 1] - z)))
        out.append(o[i] + inset * M[i] / np.linalg.norm(M[i]))
    return np.array(out)


def plate_outline():
    return G.ccw(G.spline(L.plate_control_points(), (), spacing=0.003))


def plate_mesh():
    """The chrome control plate along the lower treble edge: a thin slab with a rolled rim, lying on the face. Hardware."""
    top = lambda x, z: face(x, z) - L.PLATE_T                                     # noqa: E731
    bottom = lambda x, z: face(x, z) + 0.0003                                     # noqa: E731
    return G.slab(plate_outline(), top, bottom, r=0.0006, n=2, wall=1, spacing=0.008, mat=L.MATS["hardware"])


# ===================================================================================================================
# the racing stripes (a graphic layer)
# ===================================================================================================================
STRIPES = {"through": (-0.1520, 0.1120), "to": (-0.0560, -0.0800),       # the bands' centre line, from the bass edge to the tail
           "bands": ((-0.0160, -0.0090), (-0.0060, 0.0060), (0.0090, 0.0160))}   # across it (m): thin, wide, thin


def clip_convex(poly, clip):
    """Sutherland-Hodgman: the polygon `poly` (n, 2) cut to the convex CCW polygon `clip`; an empty array when nothing is left."""
    out = np.asarray(poly, float)
    C = np.asarray(clip, float)
    for i in range(len(C)):
        if not len(out):
            break
        a, b = C[i], C[(i + 1) % len(C)]
        e = b - a
        side = e[0] * (out[:, 1] - a[1]) - e[1] * (out[:, 0] - a[0])            # > 0: inside (left of the edge)
        res = []
        for k in range(len(out)):
            p, q = out[k], out[(k + 1) % len(out)]
            sp, sq = side[k], side[(k + 1) % len(out)]
            if sp >= 0:
                res.append(p)
            if (sp >= 0) != (sq >= 0):
                res.append(p + (q - p) * (sp / (sp - sq)))
        out = np.array(res)
    return out


def surface_y(x, z, lift=0.0):
    """y of the body's front surface at (x, z) inside the outline, over the face and the rolled edge (the slab's quarter
    round of radius BODY_EDGE_R from the face's edge to the wall), `lift` metres out of it (-y)."""
    r = L.BODY_EDGE_R
    d = np.clip(edge_distance(x, z), 0.0, r)                   # distance in from the wall
    cphi = np.clip(1.0 - d / r, 0.0, 1.0)
    drop = r * (1.0 - np.sqrt(1.0 - cphi ** 2))                 # 0 on the face, r at the wall
    return face(x, z) + drop - lift


def stripes_mesh(lift=(0.00005, 0.00035), spacing=0.004):
    """The three racing stripes across the lower bout: each band (a strip along the centre line `STRIPES`) cut to the body's
    outline 0.6 mm inside its wall, triangulated (`electric_guitar_geo.fill`) and laid over the face and the rolled edge as a
    closed film between the two `lift`s (metres out of the surface). Role `stripe`; a graphic layer (the builder tags it
    exempt)."""
    a, b = np.array(STRIPES["through"]), np.array(STRIPES["to"])
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]])                                  # across the bands
    ring = G.inset(outline(), 0.0006)
    out = []
    for lo, hi in STRIPES["bands"]:
        far = 0.30
        band = G.ccw(np.array([a - d * far + n * lo, a + d * far + n * lo, a + d * far + n * hi, a - d * far + n * hi]))
        piece = clip_convex(ring, band)
        if len(piece) < 3:
            continue
        piece = G.ccw(piece)
        piece = piece[np.r_[True, np.linalg.norm(np.diff(piece, axis=0), axis=1) > 1e-5]]
        edge = G.ccw(_resample_closed(piece, 0.002))
        P2, T = G.fill(edge, spacing, margin=0.6)
        out.append(_film(P2, T, len(edge), lift))
    return S.merge(out)


def _film(P2, T, m, lift):
    """A closed film over the front surface from a triangulated region (P2 (n, 2) of (x, z), its first m points the boundary
    ring in order, T its triangles): the region at both lifts, joined along the ring."""
    lo, hi = lift
    n = len(P2)
    top = np.column_stack([P2[:, 0], surface_y(P2[:, 0], P2[:, 1], hi), P2[:, 1]])
    bot = np.column_stack([P2[:, 0], surface_y(P2[:, 0], P2[:, 1], lo), P2[:, 1]])
    V = np.concatenate([top, bot])
    Tt = G._orient(V, T, (0.0, -1.0, 0.0))
    Tb = G._orient(V, T + n, (0.0, 1.0, 0.0))
    j = np.arange(m)
    Q = np.stack([j, j + n, (j + 1) % m + n, (j + 1) % m], 1)       # the ring runs counter-clockwise seen from the front
    mesh = S.Mesh(V, Q, np.concatenate([Tt, Tb]), mat=L.MATS["stripe"])
    return mesh if mesh.volume() > 0 else mesh.flip()


def _resample_closed(P, step):
    """Points every `step` metres along the closed polygon P (its corners kept)."""
    Q = np.vstack([P, P[:1]])
    out = []
    for p, q in zip(Q[:-1], Q[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(q - p) / step)))
        out.append(p + (q - p) * (np.arange(k)[:, None] / k))
    return np.concatenate(out)
