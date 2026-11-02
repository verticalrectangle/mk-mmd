"""The cab inside's shared numbers and small builders (pure numpy, no bpy): what `semi_truck_cab` and its part modules
(`semi_truck_cab_dash`, `semi_truck_cab_seats`, `semi_truck_cab_trim`) agree on.

The hollow they fill is the cab shell `semi_truck_body` builds (agreed with it, checked by ray casts at build time): flat
vertical side walls at x = +-X_IN up to Z_WALL_TOP (above it they roll into the ceiling), a flat floor at Z_FLOOR, the
ceiling at Z_CEIL on the centre line crowned CEIL_CROWN lower at the walls, the toe board at Y_FIREWALL (up to the
windshield's foot), the back wall at Y_BACK with the walk-through (L.WALKTHROUGH) open to the sleeper, whose dark pocket
reaches back to just behind the curtain (L.CURTAIN_Y). The door glass's inner face is at +-X_GLASS.

Frames. A face is (right, up, normal) with right x up = normal; `place` puts a mesh modelled in such a frame (x right, y up,
z out of the face) on it.
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L

M = L.M
EX, EY, EZ = np.eye(3)

# ====================================================================================================== the hollow
X_IN = L.CAB_HALF_W - L.CAB_WALL_T                   # the side walls' inner faces
Y_BACK = L.BOC_Y - L.CAB_WALL_T                      # the back wall's inner face
Y_FIREWALL = L.CAB_FRONT_Y + L.CAB_WALL_T            # the toe board's inner face (up to the windshield's foot)
Z_FLOOR = L.CAB_FLOOR_Z
Z_CEIL = L.CAB_ROOF_Z - 0.08                         # the ceiling on the centre line
CEIL_CROWN = 0.02                                    # ... this much lower at the walls
Z_WALL_TOP = 2.70                                    # the walls are flat up to here, then roll into the ceiling
X_GLASS = L.CAB_HALF_W - 0.025 - 0.006               # the door glass's inner face (a 6 mm pane, its outer face 25 mm in)
WT_X, WT_Z0, WT_Z1 = L.WALKTHROUGH                   # the walk-through: half width, bottom, top
Y_SLEEPER_POCKET = L.CURTAIN_Y + 0.03                # the back of the sleeper's dark pocket behind the curtain

# the driver's eye (the gauges face it, the wheel leaves them in view): above and a little ahead of the hip
EYE = np.array([L.SEAT_X, L.HIP[1] + 0.06, L.HIP[2] + 0.72])


def ceiling_z(x):
    """Height of the ceiling at x (the crown; flat in y)."""
    return Z_CEIL - CEIL_CROWN * (np.asarray(x, float) / X_IN) ** 2


def inside(P, margin=0.0):
    """Which points (n, 3) lie in the cab's hollow (the walk-through and the sleeper pocket behind it included)."""
    P = np.atleast_2d(np.asarray(P, float))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cab = ((np.abs(x) <= X_IN - margin) & (y >= Y_FIREWALL - 0.09) & (y <= Y_BACK - margin) & (z >= Z_FLOOR - 0.004)
           & (z <= ceiling_z(np.clip(x, -X_IN, X_IN)) - margin))
    walk = ((np.abs(x) <= WT_X - margin) & (y > Y_BACK - margin - 1e-9) & (y <= Y_SLEEPER_POCKET - margin)
            & (z >= Z_FLOOR - 0.004) & (z <= WT_Z1 + 0.10 - margin))
    return cab | walk


# ====================================================================================================== small builders
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def solid(m):
    """The mesh with its faces turned outward (a sweep's winding follows its path's turn)."""
    return m if m.volume() >= 0 else m.flip()


def frame(normal, up):
    """(right, up, normal) unit vectors, right x up = normal; `up` is made perpendicular to `normal`."""
    n = unit(normal)
    u = np.asarray(up, float)
    u = unit(u - n * (u @ n))
    return np.cross(u, n), u, n


def place(mesh, origin, fr, z0=0.0):
    """A mesh modelled in a face's frame (x right, y up, z out of the face) put on the face at `origin` (+ z0 along its
    normal: a negative z0 sinks the back of a panel into the surface)."""
    r, u, n = fr
    return mesh.apply(np.stack([r, u, n], 1), np.asarray(origin, float) + n * z0)


def to3(pts, origin, fr, w=0.0):
    """2D points (m, 2) in a frame's (right, up) plane, `w` along its normal, as 3D points."""
    r, u, n = fr
    pts = np.atleast_2d(np.asarray(pts, float))
    return np.asarray(origin, float) + pts[:, :1] * r + pts[:, 1:2] * u + w * n


def lathe_at(profile, origin, axis, seg=24, mat=0, phase=0.0, closed_ends=False):
    """A surface of revolution about `axis` through `origin` (the profile is (radius, distance along the axis))."""
    m = S.lathe(np.asarray(profile, float), seg=seg, mat=mat, phase=phase, closed_ends=closed_ends)
    return m.apply(S.rotation_between((0.0, 0.0, 1.0), axis), origin)


def rprof(pts, radii, n=3):
    """A lathe / section profile: the polyline `pts` with its corners rounded by `radii` (the ends stay)."""
    return S.fillet(np.asarray(pts, float), radii, n=n)


def rounded_poly(P, radii, n=3):
    """A closed polygon (m, 2) with every corner rounded, ALWAYS n + 1 points per corner (a straight or zero-radius corner
    gets its points spread along the two edges), so polygons with the same corners have the same number of points (rings
    of a cage). The tangent length is limited to 45 % of each neighbouring edge."""
    P = np.asarray(P, float)
    m = len(P)
    rad = np.broadcast_to(np.asarray(radii, float), (m,))
    out = []
    for i in range(m):
        p, p0, p1 = P[i], P[i - 1], P[(i + 1) % m]
        d0, d1 = p - p0, p1 - p
        l0, l1 = float(np.linalg.norm(d0)), float(np.linalg.norm(d1))
        d0, d1 = d0 / l0, d1 / l1
        cr = d0[0] * d1[1] - d0[1] * d1[0]
        phi = math.atan2(abs(cr), float(d0 @ d1))
        r = max(float(rad[i]), 1e-4)
        if phi < 1e-3:
            t = min(r, 0.45 * l0, 0.45 * l1) * 0.5
            out.append(p[None] + np.linspace(-t, t, n + 1)[:, None] * d1[None])
            continue
        t = min(r * math.tan(phi / 2), 0.45 * l0, 0.45 * l1)
        re = t / math.tan(phi / 2)
        s = 1.0 if cr > 0 else -1.0
        a = p - d0 * t
        c = a + np.array([-d0[1], d0[0]]) * s * re
        a0 = math.atan2(a[1] - c[1], a[0] - c[0])
        ang = a0 + s * phi * np.arange(n + 1) / n
        out.append(np.stack([c[0] + re * np.cos(ang), c[1] + re * np.sin(ang)], 1))
    return np.concatenate(out)


def slab(outline, height, r_top=0.004, r_bot=0.0, draft=0.0, dome=0.0, n=3, rings=2, mat=0):
    """A closed plate over the 2D `outline` (CCW, (m, 2)): its back at z = 0, its face at z = height, the face's edge rolled
    with radius r_top (n segments), the back's edge with r_bot (0: a sharp hidden back edge), the wall leaning in by
    `draft` toward the face (a moulded part is never a straight wall) and the face domed by `dome`."""
    rt = min(r_top, 0.49 * height)
    rb = min(r_bot, 0.49 * height)
    prof = []
    if rb > 0:
        a = np.linspace(0.0, 0.5 * math.pi, n + 1)[:-1]
        prof += [(rb * (1.0 - math.sin(t)), rb * (1.0 - math.cos(t))) for t in a]
    else:
        prof.append((0.0, 0.0))
    prof.append((draft * (height - rt) / max(height, 1e-9), height - rt))
    d0 = prof[-1][0]
    prof += [(d0 + rt * (1.0 - math.cos(t)), height - rt + rt * math.sin(t)) for t in np.linspace(0.0, 0.5 * math.pi, n + 1)[1:]]
    R = S.offset_rings(outline, prof)
    return S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_bulge=(0.0, dome),
                  cap_rings=rings, mat=mat)


def bezel(r_out, r_in, height, seg=32, mat=0):
    """A round bezel ring (a gauge's, a lamp's): outer radius r_out, opening r_in, `height` tall from z = 0, both top edges
    rolled over (a lathe of a closed profile: closed, outward)."""
    w = r_out - r_in
    ro, ri = min(0.5 * w, 0.9 * height), min(0.35 * w, 0.6 * height)
    a = np.linspace(0.0, 0.5 * math.pi, 4)
    prof = [(r_in, 0.0), (r_out, 0.0), (r_out, height - ro)]
    prof += [(r_out - ro + ro * math.cos(t), height - ro + ro * math.sin(t)) for t in a[1:]]
    prof += [(r_in + ri - ri * math.sin(t), height - ri + ri * math.cos(t)) for t in a]
    prof.append((r_in, 0.0))
    m = S.lathe(np.array(prof), seg=seg, mat=mat).weld(1e-9)
    return m if m.volume() > 0 else m.flip()


def rects(C, w, h, r, u, mat):
    """Flat rectangles (centres C (k, 3), widths w, heights h) in the plane spanned by r and u, facing r x u: graphic
    layers (decals)."""
    C = np.atleast_2d(np.asarray(C, float))
    k = len(C)
    w = np.broadcast_to(np.asarray(w, float), (k,))[:, None]
    h = np.broadcast_to(np.asarray(h, float), (k,))[:, None]
    dr, du = np.asarray(r, float) * 0.5, np.asarray(u, float) * 0.5
    V = np.stack([C - dr * w - du * h, C + dr * w - du * h, C + dr * w + du * h, C - dr * w + du * h], 1)
    return S.Mesh(V.reshape(-1, 3), np.arange(4 * k).reshape(k, 4), mat=mat)


def discs(C, radius, r, u, mat, sides=12):
    """Flat discs (triangle fans) at the centres C (k, 3), facing r x u: graphic layers."""
    C = np.atleast_2d(np.asarray(C, float))
    k = len(C)
    t = 2 * math.pi * np.arange(sides) / sides
    rad = np.broadcast_to(np.asarray(radius, float), (k,))[:, None, None]
    ring = C[:, None, :] + rad * (np.cos(t)[None, :, None] * np.asarray(r) + np.sin(t)[None, :, None] * np.asarray(u))
    V = np.concatenate([C, ring.reshape(-1, 3)])
    i = np.arange(k)[:, None]
    j = np.arange(sides)[None, :]
    T = np.stack([np.broadcast_to(i, (k, sides)), k + i * sides + j, k + i * sides + (j + 1) % sides], -1).reshape(-1, 3)
    return S.Mesh(V, None, T, mat=mat)


def capped(rings, mat=0, centre_fn=None):
    """A closed cage of rings (ns, m, 3) round a closed section, both ends closed by two shrinking rings and a fan toward
    the end ring's centre (`centre_fn(ring)` or its vertex mean): the ends of a loft whose sections are star-shaped about
    that point."""
    rings = np.asarray(rings, float)
    ns, m = rings.shape[:2]
    body = S.cage(rings, closed=True, caps=(None, None), mat=mat)
    Vs, Q, T, nv = [body.V], [body.Q], [], len(body.V)
    for which in (0, ns - 1):
        ring = rings[which]
        c = ring.mean(0) if centre_fn is None else np.asarray(centre_fn(ring), float)
        prev, qs = which * m + np.arange(m), []
        for f in (0.62, 0.28):
            ids = nv + np.arange(m)
            Vs.append(c + (ring - c) * f)
            nv += m
            qs.append(np.stack([prev, np.roll(prev, -1), np.roll(ids, -1), ids], 1))
            prev = ids
        Vs.append(c[None])
        ts = np.stack([prev, np.roll(prev, -1), np.full(m, nv)], 1)
        nv += 1
        V = np.concatenate(Vs)
        axis = c - rings[1 if which == 0 else ns - 2].mean(0)          # the cap looks away from the neighbouring ring
        look = sum(np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, -1] - V[F][:, 0]).sum(0) @ axis for F in qs + [ts])
        if look < 0:
            qs, ts = [q[:, ::-1] for q in qs], ts[:, ::-1]
        Q += qs
        T.append(ts)
    out = S.Mesh(np.concatenate(Vs), np.concatenate(Q), np.concatenate(T), mat=mat)
    return out if out.volume() > 0 else out.flip()


def helix_tube(path, coil_r, pitch, wire_r, turn_pts=8, sides=5, mat=0, up=(0.0, 0.0, 1.0)):
    """A coiled cord: a wire of radius wire_r wound with radius coil_r and `pitch` (m per turn) round the centre line
    `path` (n, 3, resampled by arc length). Closed (capped) tube."""
    P = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    turns = s[-1] / pitch
    k = max(math.ceil(turns * turn_pts), 8)
    t = np.linspace(0.0, s[-1], k + 1)
    C = np.stack([np.interp(t, s, P[:, j]) for j in range(3)], 1)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    U = np.cross(np.asarray(up, float), T)                # the coil's frame: `up` must not run along the path
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    B = np.cross(T, U)
    ang = 2.0 * math.pi * t / pitch
    W = C + coil_r * (np.cos(ang)[:, None] * U + np.sin(ang)[:, None] * B)
    W[0], W[-1] = C[0], C[-1]                        # the wire leaves the coil on the centre line at both ends
    return solid(S.tube(W, wire_r, sides=sides, mat=mat))
