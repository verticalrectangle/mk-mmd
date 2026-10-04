"""The instrument panel and everything the driver touches in the 1980s convertible, built from profiles, lofts, lathes and
sweeps (docs/modelling.md): no visible cuboid, every hard edge a crease on a subdivided cage or a bevel of 3 mm and more.
Pure numpy (no bpy); every number the parts agree on comes from `convertible_layout`.

    static_shells()  {"dash", "binnacle", "stack"}: creased cages the builder subdivides (Subdivision Surface level 2)
    static_parts()   {"interior": forms, "decals": graphic layers}: panels, frames, vents, knobs, column, pedals, mirror,
                     and the flat display quads / stripes / seams lifted a fraction of a millimetre off a surface
    dynamic_parts()  {"steering_wheel": Part, "cassette": Part, "visors": Part}
    SURFACES, GRIDS, ANCHORS, LOOKS, BAR_GRAPH     the card's flat faces, the speed digit grid, the cassette slot and the
                                                   mirror, the points the characters look at, the bar graph's 24 cells

Frames. The dash faces +Y (the occupants), so a viewer sees -X on the right of every face built here: a face is
(right, up, normal) with right x up = normal. The cluster's frame is `CLUSTER_FRAME` of the layout (normal tipped up 20
degrees, right = -X). The wheel part is built in its own frame (ring in XY, +Z toward the driver, spokes along X, local +Y
down in the car); its base rotation takes local +Z onto the column axis. The cassette part has its origin at the centre of
its label (the label plane is local z = 0, the shell lies below it). The visors part is built in the car's frame moved so that
its origin lies on the two pivot rods' common axis (VISOR_ORIGIN; the rods are parallel to X): it turns about its local X.

The dash body is ONE cage: the profile (toe board, knee ledge, lower panel, the shelf, the band under the brow, the rolled
brow, the pad) pulled along x with rolled ends. The pad is one plane: it starts at the foot of the windshield glass
(Z_GLASS at Y_COWL, layout) and rises to DASH_TOP_Z at the brow; its back wall stands 2 cm behind the glass foot (YW), against
the body's cowl lip, and the heights of the face (brow, band, shelf, lower panel) are a table under the pad's top (Z_BROW,
Z_BAND, Z_SHELF, Z_LOW), so a new DASH_TOP_Z moves the whole face. Its flat regions are exactly flat after subdivision, and
every panel (wood, glove lid, vents, speaker, defrost) stands on one of them with its back sunk 2-4 mm. The cluster hood
(the binnacle) is a second cage set into the pad: a pod with a visor lip over the glass, its belly hung from the brow's
underside; the stack's surround is a third, a loft along y. The cluster panel stack, from the glass outward along the cluster
normal: the pod's face (-4 mm), the bezel's back (-6 mm), the glass (0), the display elements (+0.6 mm), the card's text
surface (+1.2 mm), the bezel's top (+3.5 mm), the visor lip (+30 mm). The deck is built the same way on the plane
y = DECK_FACE_Y (glass +0.6 mm, ghost digits +1.2 mm, the text surface +1.8 mm). The cassette slot is centred on the
inserted cassette body (6.6 mm under DECK_Z), so the cassette passes the rim with 2 mm to spare.

Roles: pad, brow, pod `vinyl_dark`, knee panel / bezels / knobs / vent frames / surround `trim_dark`, wood panels `wood`,
vent voids and slots `underbody`, faceplate `deck_face`, rims, slats, buttons, latch `chrome`, display glass
`display_glass`, unlit segments `vfd_ghost`, the 24 bar cells `vfd_lit` (nothing else uses that role), indicators
`ind_a/b/c`, pinstripes, stitching and panel lines `seam`.
"""
import math

import numpy as np

from ....core import shell as S
from . import convertible_layout as L
from .convertible_layout import M

Mesh = S.Mesh
EX, EY, EZ = np.eye(3)

# ======================================================================================================== the numbers
CC = np.array(L.BINNACLE_C, float)                                  # centre of the cluster glass
CN, CU, CR = (np.array(v, float) for v in L.CLUSTER_FRAME)          # its normal, up and right (= -X)
CW, CH = L.CLUSTER_SIZE
LIFT = 0.0006                                                       # display elements above the glass
TEXT_LIFT = 0.0012                                                  # the card's text surfaces above the glass

WHEEL_C = np.array(L.WHEEL_C, float)
WHEEL_AXIS = np.array(L.WHEEL_AXIS, float)
R_WHEEL = S.rotation_between((0.0, 0.0, 1.0), WHEEL_AXIS)           # wheel frame (local +Z = column axis) -> car

DY = L.DECK_FACE_Y                                                  # the deck's face plane
SLOT_Z = L.DECK_Z - 0.0066                                          # the slot is centred on the inserted cassette body
WIN_Z = 0.8535                                                      # radio window centre
DECK_LIFT = 0.0006

YC, ZC = L.Y_COWL, L.Z_GLASS                                         # the pad at the foot of the windshield glass
YL, ZK = L.DASH_TOP_Y[1], L.DASH_TOP_Z[1]                           # the brow's front and the pad's highest point
YW = YC - 0.020                                                     # the dash's back wall: it meets the body's cowl lip, 2 cm behind the glass foot
YB = YL - 0.035                                                     # the band face under the brow
PAD_SLOPE = (ZK - ZC) / (YL - YC)                                   # the pad plane rises toward the driver
HOOD_BACK = 0.004                                                   # the pod's face is this far behind the glass

# the dash face, as heights under the pad's top (ZK): the brow's underside, the band (its foot is a crease), the shelf, the
# three control points of the lower (glove box) panel, which lie on one plane
Z_BROW = ZK - 0.040
Z_BAND = (ZK - 0.130, ZK - 0.085)
Z_SHELF = ZK - 0.160
Z_LOW = (ZK - 0.340, ZK - 0.270, ZK - 0.200)
Z_BAND_C = 0.5 * (Z_BAND[0] + Z_BROW)                               # the middle of the band: the wood panel, the end vents

# the windshield plane (y, z): along the glass and into the cabin
_G0, _G1 = np.array(L.WS_BASE, float), np.array(L.WS_TOP, float)
GLASS_LEN = float(np.hypot(*(_G1 - _G0)))
_GD = (_G1 - _G0) / GLASS_LEN
_GN = np.array([_GD[1], -_GD[0]])


def _euler_xyz(R):
    """Blender XYZ euler (degrees) of a rotation matrix: R = Rz Ry Rx."""
    b = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(b)) > 1e-6:
        a, c = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:
        a, c = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return tuple(round(math.degrees(t), 9) + 0.0 for t in (a, b, c))


WHEEL_ROT = _euler_xyz(R_WHEEL)                                     # (-68, 0, 0)


def _lst(v):
    return [float(x) for x in v]


# ===================================================================================================== small builders
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame(normal, up):
    """(right, up, normal) unit vectors, right x up = normal; `up` is made perpendicular to `normal`."""
    n = _unit(normal)
    u = np.asarray(up, float)
    u = _unit(u - n * (u @ n))
    return np.cross(u, n), u, n


def _place(mesh, origin, fr, z0=0.0):
    """A mesh built in a local frame (x right, y up, z out of the face) put on a surface: its local origin goes to
    `origin + n * z0` (a negative z0 sinks the back of a panel into the surface)."""
    r, u, n = fr
    return mesh.apply(np.stack([r, u, n], 1), np.asarray(origin, float) + n * z0)


def _to3(pts, origin, fr, w=0.0):
    """2D points (m, 2) in a frame's (right, up) plane, `w` along its normal, as 3D points."""
    r, u, n = fr
    pts = np.asarray(pts, float)
    return np.asarray(origin, float) + pts[:, :1] * r + pts[:, 1:2] * u + w * n


def _pillow(outline, height, r, n=3, dome=(0.0, 0.0), rings=2, mat=0):
    """A closed slab over the 2D `outline`: z from 0 to `height`, both edges rolled over with radius r, the back (z = 0)
    and the face (z = height) flat or domed by `dome` = (back, face)."""
    r = min(r, height / 2 - 1e-6)
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)
    prof = [(r * (1 - math.sin(t)), r * (1 - math.cos(t))) for t in a]
    prof += [(r * (1 - math.cos(t)), height - r + r * math.sin(t)) for t in a]
    R = S.offset_rings(outline, prof)
    return S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_bulge=dome, cap_rings=rings,
                  mat=mat)


def _rects(C, w, h, r, u, mat):
    """Flat rectangles (centres C (k, 3), widths w, heights h) in the plane spanned by r and u, facing r x u."""
    C = np.atleast_2d(np.asarray(C, float))
    k = len(C)
    w = np.broadcast_to(np.asarray(w, float), (k,))[:, None]
    h = np.broadcast_to(np.asarray(h, float), (k,))[:, None]
    dr, du = np.asarray(r, float) * 0.5, np.asarray(u, float) * 0.5
    V = np.stack([C - dr * w - du * h, C + dr * w - du * h, C + dr * w + du * h, C - dr * w + du * h], 1)
    return Mesh(V.reshape(-1, 3), np.arange(4 * k).reshape(k, 4), mat=mat)


def _poly(P, n, mat):
    """One flat convex polygon (m, 3) wound to face `n`."""
    P = np.asarray(P, float)
    m = len(P)
    area = sum(np.cross(P[i], P[(i + 1) % m]) for i in range(m))
    if area @ np.asarray(n, float) < 0:
        P = P[::-1]
    if m == 4:
        return Mesh(P, [[0, 1, 2, 3]], mat=mat)
    return Mesh(P, None, np.stack([np.zeros(m - 2, int), np.arange(1, m - 1), np.arange(2, m)], 1), mat=mat)


def _discs(C, radius, r, u, mat, sides=10):
    """Flat discs (triangle fans) at the centres C (k, 3), facing r x u."""
    C = np.atleast_2d(np.asarray(C, float))
    k = len(C)
    t = 2 * math.pi * np.arange(sides) / sides
    ring = C[:, None, :] + radius * (np.cos(t)[None, :, None] * np.asarray(r) + np.sin(t)[None, :, None] * np.asarray(u))
    V = np.concatenate([C, ring.reshape(-1, 3)])
    i = np.arange(k)[:, None]
    j = np.arange(sides)[None, :]
    T = np.stack([np.broadcast_to(i, (k, sides)), k + i * sides + j, k + i * sides + (j + 1) % sides], -1).reshape(-1, 3)
    return Mesh(V, None, T, mat=mat)


def _hexsegs(A, B, thick, d, n, mat):
    """Pointed ("hexagonal") display segments between the nodes A and B (k, 3), each pulled back by d at both ends,
    `thick` wide, in a plane facing n: one quad and two triangles per segment."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    k = len(A)
    e = B - A
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    nn = np.cross(n, e)                                              # e x nn = n
    h = thick / 2
    t1, t2 = A + e * d, B - e * d
    s1, s2 = t1 + e * h, t2 - e * h
    V = np.concatenate([t1, s1 - nn * h, s2 - nn * h, t2, s2 + nn * h, s1 + nn * h])
    i = np.arange(k)
    Q = np.stack([k + i, 2 * k + i, 4 * k + i, 5 * k + i], 1)
    T = np.concatenate([np.stack([i, k + i, 5 * k + i], 1), np.stack([3 * k + i, 4 * k + i, 2 * k + i], 1)])
    return Mesh(V, Q, T, mat=mat)


def _digit_nodes(cx, cy, W, H, t):
    """The seven segments a b c d e f g of a digit cell centred at (cx, cy), as node pairs ((7, 2), (7, 2))."""
    x0, x1 = cx - (W - t) / 2, cx + (W - t) / 2
    yt, yb = cy + (H - t) / 2, cy - (H - t) / 2
    A = [(x0, yt), (x1, cy), (x1, yb), (x0, yb), (x0, yb), (x0, cy), (x0, cy)]
    B = [(x1, yt), (x1, yt), (x1, cy), (x1, yb), (x0, cy), (x0, yt), (x1, cy)]
    return np.array(A), np.array(B)


def _lathe_at(profile, origin, axis, seg=24, mat=0):
    """A surface of revolution about `axis` through `origin` (the profile is (radius, distance along the axis))."""
    return S.lathe(profile, seg=seg, mat=mat).apply(S.rotation_between((0.0, 0.0, 1.0), axis), origin)


def _round_profile(pts, radii, n=4):
    """A lathe profile (radius, height): the polyline `pts` with the corners rounded by `radii`."""
    return S.fillet(np.asarray(pts, float), radii, n=n)


def _vent(w, h, nslat, tilt_deg, frame_mat, slat_mat, depth=0.011, f=0.0065):
    """A louvered vent in a local frame (x right, y up, z out of the face): a rolled frame with a pocket, and `nslat`
    slats (a tilted ellipse swept across the opening, ends buried in the frame's wall)."""
    ro = min(0.012, 0.28 * h)
    outer = S.rrect(w, h, ro, 4)
    inner = S.rrect(w - 2 * f, h - 2 * f, max(ro - f, 0.003), 4)
    frame = S.rounded_frame(outer, inner, depth, r_out=0.55 * f, r_in=0.33 * f, floor=depth - 0.003, mat=frame_mat)
    iw, ih = w - 2 * f, h - 2 * f
    pitch = ih / nslat
    t = math.radians(tilt_deg)
    ell = S.ellipse(0.0011, 0.0050, 14)
    sec = np.stack([ell[:, 0] * math.cos(t) - ell[:, 1] * math.sin(t), ell[:, 0] * math.sin(t) + ell[:, 1] * math.cos(t)], 1)
    slat = S.sweep(np.array([(-iw / 2 - 0.0015, 0.0, 0.0), (iw / 2 + 0.0015, 0.0, 0.0)]), sec, up=(0.0, 0.0, 1.0),
                   mat=slat_mat)
    parts = [frame] + [slat.moved((0.0, -ih / 2 + pitch * (j + 0.5), depth - 0.0045)) for j in range(nslat)]
    return S.merge(parts)


# ================================================================================================= the dash body: cages
def _zp(y):
    """Height of the pad plane at y (from the cowl up to the brow)."""
    return ZC + PAD_SLOPE * (y - YC)


def _glass_y(z):
    """y of the windshield glass's plane at height z."""
    return _G0[0] + (z - _G0[1]) * _GD[0] / _GD[1]


DASH_NAMES = ("wall_b", "ch1", "ch2", "under1", "under2", "ledge", "low0", "low1", "low2", "shelf", "band0", "band1", "brow",
              "lip0", "lip1", "lip2", "lip3", "crest") + tuple(f"pad{k}" for k in range(6)) + ("cowl1", "cowl2", "wall_m")
DASH_IDX = {n: i for i, n in enumerate(DASH_NAMES)}


def dash_polygon():
    """The dash body's section: ((m, 2) control polygon (y, z) counter-clockwise from the back wall's foot up the underside,
    the knee ledge, the lower panel, the shelf, the band, the rolled brow, the pad, down the back wall; (m,) crease weights
    of the edges that run along x). The back wall stands on the cockpit's front line (Y_COWL, the base of the glass); the
    hidden corners there are chamfered (turns of 55 degrees and less), so the flat back wall has no hard rim."""
    zu = Z_BROW
    pts = [(YW, 0.425, 0.0), (YW + 0.006, 0.406, 0.0), (YW + 0.020, 0.395, 0.0),                    # the foot of the wall
           (YL - 0.158, 0.490, 0.0), (YL - 0.125, 0.540, 0.0), (YL - 0.102, 0.585, 0.5),           # underside, knee ledge
           (YL - 0.040, Z_LOW[0], 0.0), (YL - 0.028, Z_LOW[1], 0.0), (YL - 0.016, Z_LOW[2], 0.0),  # the lower (glove) panel
           (YL - 0.023, Z_SHELF, 0.0), (YB, Z_BAND[0], 0.8), (YB, Z_BAND[1], 0.0),                 # the shelf, the band
           (YB, zu, 0.9),                                                                           # under the brow
           (YL - 0.020, zu + 0.002, 0.0), (YL + 0.002, zu + 0.003, 0.0), (YL + 0.005, 0.5 * (zu + _zp(YL)), 0.0),   # the rolled brow
           (YL + 0.004, _zp(YL + 0.004), 0.0), (YL - 0.026, _zp(YL - 0.026), 0.0)]
    pts += [(y, _zp(y), 0.0) for y in np.linspace(YL - 0.026, YC + 0.040, 7)[1:]]                    # the flat pad (6 points)
    pts += [(YW + 0.009, _zp(YW + 0.009), 0.7), (YW, _zp(YW) - 0.009, 0.7), (YW, 0.740, 0.0)]       # the chamfered cowl, the wall
    assert len(pts) == len(DASH_NAMES)
    a = np.array(pts)
    return a[:, :2], a[:, 2]


def hood_polygon():
    """The binnacle's section (a pod with a visor lip over the glass, set into the pad): ((m, 2) control polygon (y, z)
    counter-clockwise, (m,) crease weights). Its lower back is buried in the dash."""
    cu, cn = CU[1:], CN[1:]
    B0 = CC[1:] - HOOD_BACK * cn                                     # on the pod's face plane, at the glass centre
    face = lambda v: B0 + v * cu                                     # noqa: E731
    q0 = np.array([YB - 0.020, Z_BROW - 0.010])
    q1, q2 = np.array([YB + 0.030, Z_BROW - 0.004]), np.array([-0.337, Z_BROW + 0.003])         # the belly: the brow's underside, carried on
    d = q2 - q1
    s, v = np.linalg.solve(np.array([d, -cu]).T, B0 - q1)           # the belly line meets the face line at the chin
    chin = q1 + s * d
    ft = face(0.088)
    vis_v = ft + 0.028 * cn + 0.022 * cu
    crest = vis_v + np.array([-0.040, 0.004])
    by = _glass_y(1.10) + 0.024                                       # the pod's back wall: 2.4 cm inside the glass
    pts = [q0, q1, q2, chin, chin + 0.014 * cu, face(-0.040), face(0.020), face(0.065), ft, ft + 0.020 * cn,
           ft + 0.038 * cn + 0.006 * cu, vis_v, crest, crest + np.array([-0.026, -0.002]), np.array([by + 0.002, 1.098]),
           np.array([by - 0.006, 1.050]), np.array([by - 0.012, _zp(by - 0.012) + 0.004]), np.array([by - 0.032, _zp(by - 0.032) - 0.025])]
    cw = np.zeros(len(pts))
    cw[8], cw[10] = 0.8, 0.3                                         # the visor's underside corner, the lip's nose
    return np.array(pts), cw


def _extruded_rings(P, x0, x1, mids, r):
    """Rings (n, m, 3) of the closed (y, z) polygon P pulled along x from x0 to x1: the ends roll over with radius r (four
    rings each: the profile shrinks inward by r (1 - cos a) while x closes in as r (1 - sin a)), `mids` are plain rings.
    The inward offsets are mitred, but never longer than 1.8 r (a hidden wedge corner does not fly off)."""
    P = np.asarray(P, float)
    ccw = S.polygon_area(P) > 0
    Mi = S.mitre_offsets(P if ccw else P[::-1])
    if not ccw:
        Mi = Mi[::-1]
    ln = np.linalg.norm(Mi, axis=1)
    Mi = Mi * np.minimum(1.0, 1.8 / np.maximum(ln, 1e-9))[:, None]
    t = np.radians([90.0, 60.0, 30.0, 0.0])
    dx, d = r * (1 - np.sin(t)), r * (1 - np.cos(t))
    st = [(x0 + a, b) for a, b in zip(dx, d)] + [(x, 0.0) for x in mids] + [(x1 - a, b) for a, b in zip(dx[::-1], d[::-1])]
    return np.stack([np.column_stack([np.full(len(P), x), P + b * Mi]) for x, b in st])


def _kernel_centre(R):
    """A point well inside the kernel of the simple polygon R (m, 2): the grid point farthest from the line of every edge
    that still has all of them on its inner side (a fan from it covers the polygon); the vertex mean when there is none."""
    R = np.asarray(R, float)
    R = R if S.polygon_area(R) > 0 else R[::-1]
    lo, hi = R.min(0), R.max(0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120)), -1).reshape(-1, 2)
    e = np.roll(R, -1, axis=0) - R
    ln = np.linalg.norm(e, axis=1)
    d = (e[None, :, 0] * (g[:, None, 1] - R[None, :, 1]) - e[None, :, 1] * (g[:, None, 0] - R[None, :, 0])) / ln[None, :]
    worst = d.min(1)
    k = int(np.argmax(worst))
    return g[k] if worst[k] > 0 else R.mean(0)


def _flat_cap(V, ring_ids, centre, axis, layers=2):
    """Faces closing a ring that looks along `axis` with a flat disc: `layers` rings shrunk toward `centre`, then a fan to it.
    Returns (new vertices, triangles, quads); indices count from len(V)."""
    ring = V[ring_ids]
    n0, C = len(V), len(ring_ids)
    new, Q = [], []
    prev = np.asarray(ring_ids)
    for k in range(1, layers + 1):
        t = 1.0 - k / (layers + 1.0)
        ids = n0 + len(new) + np.arange(C)
        new.extend(centre + (ring - centre) * t)
        Q.append(np.stack([prev, np.roll(prev, -1), np.roll(ids, -1), ids], 1))
        prev = ids
    ai = n0 + len(new)
    new.append(centre)
    T = np.stack([prev, np.roll(prev, -1), np.full(C, ai)], 1)
    Qa = np.concatenate(Q)
    Vall = np.concatenate([V, np.array(new)])
    look = 0.0
    for F in (T, Qa):
        P = Vall[F]
        look += np.cross(P[:, 1] - P[:, 0], P[:, -1] - P[:, 0]).sum(0) @ axis
    if look < 0:
        T, Qa = T[:, ::-1], Qa[:, ::-1]
    return np.array(new), T, Qa


def _capped_tube(rings, crease_cols, mat):
    """A cage of the rings (n, m, 3) round a closed section, both ends closed by flat caps fanned from a kernel point of the
    end outline (the sections here are not star-shaped about their vertex mean: a brow overhangs, a visor juts)."""
    m = S.cage(rings, closed=True, caps=(None, None), crease_cols=crease_cols, mat=mat)
    ns, mp = rings.shape[:2]
    V, T, Q = m.V, [m.T], [m.Q]
    for which, ax in ((0, np.array([-1.0, 0.0, 0.0])), (ns - 1, np.array([1.0, 0.0, 0.0]))):
        ids = which * mp + np.arange(mp)
        c = np.array([rings[which][0, 0], *_kernel_centre(rings[which][:, 1:])])
        nv, t, q = _flat_cap(V, ids, c, ax)
        V = np.concatenate([V, nv])
        T.append(t)
        Q.append(q)
    out = Mesh(V, np.concatenate(Q), np.concatenate(T), mat=mat)
    out.crease(m.Ce, m.Cw)
    return out


def dash_cage():
    P, cw = dash_polygon()
    rings = _extruded_rings(P, -L.DASH_X, L.DASH_X, (-0.45, -0.15, 0.15, 0.45), 0.012)
    m = _capped_tube(rings, cw, M["vinyl_dark"])
    nseg, last = len(P), DASH_IDX["ledge"]                              # tube quads run ring by ring: face f is segment f % m
    f = np.arange((len(rings) - 1) * nseg)
    m.Qm[f[(f % nseg) < last]] = M["trim_dark"]                         # the underside, up to the knee ledge
    return m


def hood_cage():
    P, cw = hood_polygon()
    rings = _extruded_rings(P, CC[0] - 0.225, CC[0] + 0.225, (CC[0] - 0.10, CC[0] + 0.10), 0.007)
    return _capped_tube(rings, cw, M["vinyl_dark"])


STACK_RINGS = ((-0.322, 0.236, 0.258, 0.800, 0.030), (-0.334, 0.246, 0.268, 0.798, 0.034),     # y, width, height, centre z, corner
               (-0.356, 0.268, 0.290, 0.7925, 0.042), (-0.384, 0.295, 0.312, 0.784, 0.052),     # radius: the top climbs to the brow
               (-0.425, 0.325, 0.325, 0.7675, 0.060))                                           # and stays under the pad, the bottom sinks


def stack_cage():
    """The centre stack's surround: a loft along y of rounded rectangles that widens from the faceplate back into the dash."""
    rings = []
    for y, w, h, zc, r in STACK_RINGS:
        o = S.rrect(w, h, r, 4)
        rings.append(np.column_stack([o[:, 0], np.full(len(o), y), o[:, 1] + zc]))
    return S.cage(np.stack(rings), closed=True, caps=("fan", "fan"), cap_rings=2, mat=M["trim_dark"])


# ======================================================================================== surfaces panels stand on
def _pad_frame():
    n = _unit((0.0, -PAD_SLOPE, 1.0))
    return _frame(n, (0.0, -1.0, -PAD_SLOPE))


def _pad_at(x, y):
    return np.array([x, y, _zp(y)])


_P, _ = dash_polygon()
_LOW0, _LOW1 = _P[DASH_IDX["low0"]], _P[DASH_IDX["low2"]]               # the lower panel's bottom and top control points
_lw = (_LOW1 - _LOW0) / np.linalg.norm(_LOW1 - _LOW0)


def _lower_frame():
    return _frame((0.0, _lw[1], -_lw[0]), (0.0, _lw[0], _lw[1]))


def _lower_at(x, z):
    """A point on the lower panel's plane at height z."""
    return np.array([x, _LOW0[0] + (z - _LOW0[1]) * _lw[0] / _lw[1], z])


def _band_frame():
    return _frame(EY, EZ)


def _band_at(x, z):
    return np.array([x, YB, z])


class _Bag:
    """The two lists a module builds: `forms` (the object `interior`) and `decals` (graphic layers: the object `decals`,
    which the form check skips)."""

    def __init__(self):
        self.forms, self.decals = [], []


# ========================================================================================================= dash details
def _pad_details(b):
    fr = _pad_frame()
    # defrost vents: two long louvered frames at the windshield base
    vent = _vent(0.46, 0.040, 3, 28.0, M["trim_dark"], M["trim_dark"], depth=0.009, f=0.0100)
    for x in (0.40, -0.40):
        b.forms.append(_place(vent, _pad_at(x, YC + 0.034), fr, -0.0035))
    # speaker: a soft panel with a grid of perforations
    cy = YC + 0.095
    panel = S.rounded_panel(S.rrect(0.176, 0.082, 0.016, 4), 0.006, r_edge=0.003, n=2, mat=M["trim_dark"])
    b.forms.append(_place(panel, _pad_at(0.0, cy), fr, -0.003))
    xs = (np.arange(17) - 8) * 0.0098
    ys = (np.arange(5) - 2) * 0.0098
    pts = np.array([(x, y) for y in ys for x in xs])
    b.decals.append(_discs(_to3(pts, _pad_at(0.0, cy), fr, 0.0033), 0.0016, fr[0], fr[1], M["underbody"]))


def _panel_rim(b, fr, origin, w, h, z_top, inset=0.003, r=0.007):
    """A thin chrome pinstripe round a rectangle on a panel face."""
    path = _to3(S.rrect(w - 2 * inset, h - 2 * inset, r, 3), origin, fr, z_top + 0.0004)
    b.decals.append(S.ribbon(path, 0.0012, fr[2], mat=M["chrome"], closed=True))


def _wood(b):
    """Burl panels (5 mm, rolled rim) with a chrome pinstripe: a band across the passenger side of the dash (between the
    stack and the end vent) and a panel under the cluster on the driver's side (the column's boot stands in front of it)."""
    fr = _band_frame()
    for x0, x1, zc, h in ((-0.625, -0.215, Z_BAND_C, 0.050), ):
        o, w = _band_at(0.5 * (x0 + x1), zc), x1 - x0
        b.forms.append(_place(S.rounded_panel(S.rrect(abs(w), h, 0.008, 3), 0.006, r_edge=0.003, n=2, mat=M["wood"]), o, fr,
                              -0.003))
        _panel_rim(b, fr, o, abs(w), h, 0.003, 0.0035, 0.005)
    fr = _lower_frame()
    for x0, x1, zc, h in ((0.205, 0.700, Z_LOW[2] - 0.042, 0.050), ):
        o, w = _lower_at(0.5 * (x0 + x1), zc), x1 - x0
        b.forms.append(_place(S.rounded_panel(S.rrect(w, h, 0.008, 3), 0.006, r_edge=0.003, n=2, mat=M["wood"]), o, fr, -0.003))
        _panel_rim(b, fr, o, w, h, 0.003, 0.0035, 0.005)


def _glove(b):
    """The glove box lid on the lower panel (rolled rim, a hair of crown), its gap, and a chrome dome latch."""
    fr = _lower_frame()
    cx, zc = -0.46, 0.5 * (Z_LOW[0] + Z_LOW[2]) - 0.002
    o = _lower_at(cx, zc)
    lid_h = 0.105
    lid = S.rounded_panel(S.rrect(0.480, lid_h, 0.020, 4), 0.012, r_edge=0.0048, n=3, dome=0.0012, mat=M["vinyl_dark"])
    b.forms.append(_place(lid, o, fr, -0.004))
    b.decals.append(S.ribbon(_to3(S.rrect(0.492, lid_h + 0.012, 0.026, 4), o, fr, 0.0006), 0.0016, fr[2], mat=M["seam"], closed=True))
    base = _to3(np.array([(-0.185, -0.024)]), o, fr, 0.0)[0]
    prof = [(0.0165, -0.002), (0.0165, 0.0020), (0.0140, 0.0054), (0.0090, 0.0072), (0.0, 0.0078)]
    b.forms.append(_lathe_at(prof, base, fr[2], seg=24, mat=M["chrome"]))
    slot = _rects([base + fr[2] * 0.0082], 0.021, 0.0017, fr[0], fr[1], M["seam"])
    b.decals.append(slot)


def _end_vents(b):
    fr = _band_frame()
    vent = _vent(0.070, 0.046, 4, 24.0, M["trim_dark"], M["chrome"], depth=0.010, f=0.0058)
    for x in (0.700, -0.700):
        b.forms.append(_place(vent, _band_at(x, Z_BAND_C), fr, -0.003))


# ============================================================================================== binnacle and cluster
def _bezel(b):
    fr = (CR, CU, CN)
    frame = S.rounded_frame(S.rrect(CW + 0.026, CH + 0.026, 0.016, 5), S.rrect(CW, CH, 0.008, 5), 0.0095, r_out=0.0070,
                            r_in=0.0045, mat=M["trim_dark"])
    b.forms.append(_place(frame, CC, fr, -0.006))


# panel layout (u right, v up from the panel centre)
SPEED_UV, SPEED_SIZE = (-0.045, 0.008), (0.125, 0.052)
BARS_UV, BARS_SIZE = (0.0, -0.040), (0.230, 0.014)
BAR_CELLS, BAR_GAP = 24, 0.0015
DIGIT_W, DIGIT_H, DIGIT_T = 0.034, 0.052, 0.0062                     # a seven-segment cell: width, height, stroke


def _cluster_display(b):
    P = lambda uv, w=LIFT: CC + uv[0] * CR + uv[1] * CU + w * CN       # noqa: E731
    d = b.decals
    d.append(S.quad_in_plane(CC, CN, CU, (CW, CH), mat=M["display_glass"]))
    ghost, lit = M["vfd_ghost"], M["vfd_lit"]

    # speed readout: three seven-segment digits in three equal columns of the speed surface, two decimal dots
    dw, dh, t = DIGIT_W, DIGIT_H, DIGIT_T
    pitch = SPEED_SIZE[0] / 3
    cx = [SPEED_UV[0] + (k - 1) * pitch for k in range(3)]
    A, B = [], []
    for c in cx:
        a, bb = _digit_nodes(c, SPEED_UV[1], dw, dh, t)
        A.append(a)
        B.append(bb)
    A, B = np.concatenate(A), np.concatenate(B)
    fr = (CR, CU, CN)
    d.append(_hexsegs(_to3(A, CC, fr, LIFT), _to3(B, CC, fr, LIFT), t, 0.0016, CN, ghost))
    dots = [(c + pitch / 2, SPEED_UV[1] - dh / 2 + 0.003) for c in cx[:2]]
    d.append(_rects([P(q) for q in dots], 0.0055, 0.0055, CR, CU, ghost))

    # horizontal bar graph: 24 contiguous cells ordered left to right
    cw = (BARS_SIZE[0] - (BAR_CELLS - 1) * BAR_GAP) / BAR_CELLS
    u = BARS_UV[0] - BARS_SIZE[0] / 2 + cw / 2 + np.arange(BAR_CELLS) * (cw + BAR_GAP)
    d.append(_rects(CC + u[:, None] * CR + BARS_UV[1] * CU + LIFT * CN, cw, BARS_SIZE[1], CR, CU, lit))

    # fuel (left) and temperature (right): five stacked cells each
    v = (np.arange(5) - 2) * 0.0085
    for u0 in (-0.145, 0.145):
        d.append(_rects(CC + u0 * CR + v[:, None] * CU + LIFT * CN, 0.012, 0.007, CR, CU, ghost))

    # six indicators along the top edge: two turn arrows (pine), love, gold, a ghost, gold
    vt = 0.043
    for u0, mat in ((-0.09, M["ind_c"]), (-0.03, M["ind_a"]), (0.03, M["vfd_ghost"]), (0.09, M["ind_a"])):
        d.append(_rects([P((u0, vt))], 0.0065, 0.0065, CR, CU, mat))
    for u0, s in ((-0.15, -1.0), (0.15, 1.0)):                        # arrows point outward: s = side of the tip
        tip, b0, b1 = P((u0 + s * 0.0045, vt)), P((u0 - s * 0.0035, vt + 0.0045)), P((u0 - s * 0.0035, vt - 0.0045))
        d.append(_poly(np.array([tip, b0, b1]), CN, M["ind_b"]))
        d.append(_rects([P((u0 - s * 0.0045, vt))], 0.003, 0.0045, CR, CU, M["ind_b"]))


# ============================================================================================== the centre stack
def _knob(origin, R):
    """A radio knob pointing along +Y: a dark body with a rolled shoulder, a chrome cap."""
    s = R / 0.019
    body = _round_profile([(0.0192 * s, 0.0), (0.0192 * s, 0.0150 * s), (0.0, 0.0150 * s)], [0, 0.0042 * s, 0], 4)
    cap = _round_profile([(0.0120 * s, 0.0138 * s), (0.0120 * s, 0.0172 * s), (0.0, 0.0172 * s)], [0, 0.0030 * s, 0], 4)
    return [_lathe_at(body, origin, EY, seg=32, mat=M["trim_dark"]), _lathe_at(cap, origin, EY, seg=28, mat=M["chrome"])]


def _stack_face(b):
    nrm, up, rt = EY, EZ, -EX                                         # the deck faces +Y; a viewer sees -X on the right
    fr = (rt, up, nrm)
    deck = lambda x, z, w=0.0: np.array([x, DY + w, z])               # noqa: E731

    # faceplate: a panel 12 mm thick, its face on y = DECK_FACE_Y, with a rolled rim
    plate = S.rounded_panel(S.rrect(0.250, 0.270, 0.026, 5), 0.012, r_edge=0.0045, n=3, mat=M["deck_face"])
    b.forms.append(_place(plate, deck(0.0, 0.800), fr, -0.012))

    # air vent bar above
    b.forms.append(_place(_vent(0.196, 0.034, 4, 26.0, M["trim_dark"], M["trim_dark"], depth=0.008, f=0.0060),
                          deck(0.0, 0.9075), fr, -0.0035))

    # cassette slot: a dark opening (decal) in a thin bright rim (outer 0.121 x 0.029, inner 0.108 x 0.016)
    b.decals.append(_rects([deck(0.0, SLOT_Z, DECK_LIFT)], 0.108, 0.016, rt, up, M["underbody"]))
    rim = S.rounded_frame(S.rrect(0.121, 0.029, 0.011, 4), S.rrect(0.108, 0.016, 0.0065, 4), 0.0075, r_out=0.0036,
                          r_in=0.0021, mat=M["chrome"])
    b.forms.append(_place(rim, deck(0.0, SLOT_Z), fr, -0.0045))

    # radio window: bezel, glass, ghost frequency digits (4 digits and a dot)
    wc = deck(0.0, WIN_Z)
    bez = S.rounded_frame(S.rrect(0.128, 0.046, 0.012, 4), S.rrect(0.110, 0.028, 0.006, 4), 0.0070, r_out=0.0050,
                          r_in=0.0030, mat=M["trim_dark"])
    b.forms.append(_place(bez, wc, fr, -0.0045))
    b.decals.append(S.quad_in_plane(wc + nrm * DECK_LIFT, nrm, up, (0.110, 0.028), mat=M["display_glass"]))
    dw, dh, t = 0.0105, 0.019, 0.0026
    A, B = [], []
    for c in (-0.0305, -0.012, 0.0065, 0.0295):
        a, bb = _digit_nodes(c, 0.0, dw, dh, t)
        A.append(a)
        B.append(bb)
    A, B = np.concatenate(A), np.concatenate(B)
    o = wc + nrm * (2 * DECK_LIFT)
    b.decals.append(_hexsegs(_to3(A, o, fr), _to3(B, o, fr), t, 0.0008, nrm, M["vfd_ghost"]))
    b.decals.append(_rects([o + rt * 0.018 + up * (-dh / 2 + 0.0022)], 0.0030, 0.0030, rt, up, M["vfd_ghost"]))

    # two knobs flanking the slot
    for x in (0.088, -0.088):
        b.forms += _knob(deck(x, SLOT_Z, -0.0006), 0.019)

    # six preset buttons
    btn = S.rounded_panel(S.rrect(0.028, 0.0135, 0.0052, 3), 0.0075, r_edge=0.0030, n=3, dome=0.0006, mat=M["chrome"])
    for j in range(6):
        b.forms.append(_place(btn, deck(-0.085 + 0.034 * j, 0.745), fr, -0.0012))

    # climate: two dials and two sliders
    dial = _round_profile([(0.0135, 0.0), (0.0135, 0.0105), (0.0, 0.0105)], [0, 0.0030, 0], 4)
    cap = _round_profile([(0.0082, 0.0098), (0.0082, 0.0128), (0.0, 0.0128)], [0, 0.0022, 0], 4)
    for x in (0.093, -0.093):
        b.forms.append(_lathe_at(dial, deck(x, 0.696, -0.0006), nrm, seg=28, mat=M["trim_dark"]))
        b.forms.append(_lathe_at(cap, deck(x, 0.696, -0.0006), nrm, seg=24, mat=M["chrome"]))
    tab = S.rounded_panel(S.rrect(0.011, 0.0125, 0.0045, 3), 0.0075, r_edge=0.0030, n=2, mat=M["chrome"])
    for z, x in ((0.704, 0.012), (0.686, -0.020)):
        b.decals.append(_rects([deck(0.0, z, DECK_LIFT)], 0.084, 0.0035, rt, up, M["underbody"]))
        b.forms.append(_place(tab, deck(x, z), fr, -0.0012))


# ======================================================================================== the cassette and the wheel
def _cassette_mesh():
    """Compact cassette in its own frame: origin at the centre of the label (on the label face), +X along the long side,
    +Y the way it is pushed in (the leading edge at y = -0.032), +Z the label side."""
    top = -0.0006                                                   # the shell's label face, just under the label paper
    X, Y = L.CASSETTE[0] / 2, L.CASSETTE[1] / 2
    pts = np.array([(-X, -Y), (-0.021, -Y), (-0.018, -Y + 0.0065), (0.018, -Y + 0.0065), (0.021, -Y), (X, -Y), (X, Y), (-X, Y)])
    outline = S.fillet(pts, [0.004, 0.0015, 0.0015, 0.0015, 0.0015, 0.004, 0.004, 0.004], n=3, closed=True)
    shell = _pillow(outline, L.CASSETTE[2], 0.0026, n=3, dome=(0.0, 0.0), mat=M["cassette"])
    parts = [shell.moved((0.0, 0.0, top - L.CASSETTE[2]))]
    r, u = -EX, -EY                                                  # the label's right and up (up = toward the leading edge)
    parts.append(S.quad_in_plane((0.0, 0.0, 0.0), EZ, u, (0.062, 0.034), mat=M["label"]))
    for y, mat in ((0.005, "stripe_a"), (0.0095, "stripe_b"), (0.014, "stripe_c")):
        parts.append(_rects([(0.0, y, 0.0003)], 0.062, 0.004, r, u, M[mat]))
    # the hub window on the leading half, with two hubs
    win = np.array([(x, -0.0245 + yy, top + 0.0003) for x, yy in S.rrect(0.060, 0.0085, 0.003, 2)])
    parts.append(_poly(win, EZ, M["trim_dark"]))
    hub = _round_profile([(0.0039, top + 0.0004), (0.0039, top + 0.0011), (0.0, top + 0.0011)], [0, 0.0006, 0], 2)
    for x in (-0.0155, 0.0155):
        parts.append(S.lathe(hub, seg=14, mat=M["label"]).moved((x, -0.0245, 0.0)))
    # the pressure pad in the leading notch
    pad = _pillow(S.rrect(0.036, 0.0060, 0.0022, 3), 0.0060, 0.0016, n=2, mat=M["chrome"])
    parts.append(pad.rot(rx=90.0).moved((0.0, -Y + 0.0065 + 0.0006, top - 0.0060)))
    return S.merge(parts)


def _wheel_mesh():
    """The 2-spoke wheel in its own frame: origin at the rim centre, the ring in XY, +Z toward the driver, spokes along
    X drooping 12 degrees (local +Y is down), the hub dished 26 mm away from the driver."""
    R, T = L.WHEEL_R, L.WHEEL_TUBE
    steer, chrome = M["steering"], M["chrome"]
    parts = [S.tube(S.torus_path(R, n=96), T, sides=24, closed=True, mat=steer).weld(1e-9)]
    droop = math.radians(12.0)
    n = 12
    t = np.linspace(0.0, 1.0, n)
    sm = t * t * (3 - 2 * t)
    sec = S.rrect(1.0, 1.0, 0.30, 3)
    width = 0.040 + 0.034 * np.clip((t - 0.62) / 0.38, 0.0, 1.0) ** 2           # flares into the rim
    thick = 0.024 - 0.008 * sm
    sc = np.stack([width, thick], 1)
    for sgn in (1.0, -1.0):
        rr = 0.030 + (R - 0.004 - 0.030) * t
        path = np.stack([sgn * rr * math.cos(droop), rr * math.sin(droop), -0.030 * (1 - sm)], 1)
        parts.append(S.sweep(path, sec, scale=sc, up=(0.0, 0.0, 1.0), mat=steer))
    # the horn pad: a squared pad with a rolled rim and a soft crown, a collar behind it
    thick_pad, dome = 0.040, 0.005
    pad = S.rounded_panel(S.superellipse(0.118, 0.088, 3.6, 56), thick_pad, r_edge=0.010, n=4, dome=dome, rings=3, mat=steer)
    z_face = 0.010
    parts.append(pad.moved((0.0, 0.0, z_face - thick_pad)))
    collar = _round_profile([(0.046, -0.050), (0.046, -0.012), (0.034, -0.002), (0.0, -0.002)], [0, 0.010, 0.0, 0], 4)
    parts.append(S.lathe(collar, seg=40, mat=steer))
    # a thin chrome bezel round the pad and two accent lines across its crown
    ring = S.rounded_frame(S.superellipse(0.111, 0.081, 3.6, 56), S.superellipse(0.095, 0.065, 3.6, 56), 0.0035,
                           r_out=0.0014, r_in=0.0012, mat=chrome)
    parts.append(ring.moved((0.0, 0.0, z_face + 0.0006 - 0.0035)))
    a, bb, p = 0.049, 0.034, 3.6
    for y in (-0.0115, 0.0115):
        x = np.linspace(-0.040, 0.040, 21)
        rho = (np.abs(x / a) ** p + abs(y / bb) ** p) ** (1.0 / p)
        z = z_face + dome * np.clip(1 - rho ** 2, 0.0, 1.0) + 0.0005
        parts.append(S.ribbon(np.stack([x, np.full(21, y), z], 1), 0.0025, EZ, mat=chrome))
    return S.merge(parts)


# ======================================================================================== column, pedals, mirror
def _column_s_face():
    """Distance s back along the column axis from the wheel centre at which the axis enters the band face."""
    s = (WHEEL_C[1] - YB) / WHEEL_AXIS[1]
    return float(s)


def _column(b):
    """Column shroud (a tapered lathe from behind the hub into the dash, with a round boot where it enters the band), the
    turn-signal stalk and the ignition ring. Built about the column axis (local -Z of the wheel) and moved into the car."""
    dark, chrome = M["trim_dark"], M["chrome"]
    sf = _column_s_face()
    prof = _round_profile([(0.034, 0.030), (0.037, 0.080), (0.046, sf - 0.050), (0.052, sf - 0.030), (0.088, sf + 0.004),
                           (0.088, sf + 0.060)], [0, 0, 0, 0.012, 0.004, 0], 4)
    parts = [S.lathe(prof, seg=40, mat=dark).apply(S.rotation_between((0, 0, 1), -WHEEL_AXIS), WHEEL_C)]
    # the turn-signal stalk to the car's left: a tapered lever with a chrome sleeve and a rounded end
    R = R_WHEEL
    d = _unit(R @ np.array([1.0, 0.42, 0.10]))
    base = WHEEL_C + R @ np.array([0.034, 0.003, -0.095])
    shaft = _round_profile([(0.0078, -0.010), (0.0068, 0.055), (0.0068, 0.062)], [0, 0, 0], 2)
    sleeve = _round_profile([(0.0080, 0.062), (0.0080, 0.098), (0.0, 0.098)], [0, 0.0045, 0], 4)
    parts.append(_lathe_at(shaft, base, d, seg=14, mat=dark))
    parts.append(_lathe_at(sleeve, base, d, seg=16, mat=chrome))
    # the ignition lock on the column's right (-X): a chrome ring round a dark barrel
    ax = WHEEL_C - WHEEL_AXIS * 0.125
    out = -EX
    ring = _round_profile([(0.0190, -0.004), (0.0190, 0.0030), (0.0155, 0.0052), (0.0100, 0.0052), (0.0100, 0.0)],
                          [0, 0.0016, 0.0016, 0, 0], 3)
    parts.append(_lathe_at(ring, ax + out * 0.037, out, seg=24, mat=chrome))
    parts.append(_lathe_at([(0.0100, 0.0016), (0.0100, 0.0036), (0.0, 0.0036)], ax + out * 0.037, out, seg=20, mat=M["underbody"]))
    b.forms += parts


def _pedals(b):
    """Brake (under the driver's right foot, in front of the seat), accelerator to its right (-X), both hanging on thin arms
    from the dash's underside, and the footrest on the driver's left."""
    dark, rub, chrome = M["trim_dark"], M["rubber"], M["chrome"]
    sx = L.SEAT_X

    def pedal(x, w, h, normal, centre, pivot, ribs):
        fr = _frame(normal, (0.0, -normal[2], normal[1]))
        c = np.array([x, *centre])
        outline = S.rrect(w, h, 0.012, 3)
        b.forms.append(_place(_pillow(outline, 0.016, 0.0055, n=2, mat=rub), c, fr, -0.016))
        for v in np.linspace(-h / 2 + 0.011, h / 2 - 0.011, ribs):
            rib = S.rounded_panel(S.stadium(w - 0.016, 0.0044, 4), 0.0030, r_edge=0.0013, n=2, mat=chrome)
            b.forms.append(_place(rib, c + fr[1] * v, fr, -0.0004))
        top = np.array([x, *pivot])
        b.forms.append(S.tube(np.array([top, c - fr[2] * 0.014]), np.array([0.0085, 0.0070]), sides=12, mat=dark))

    pedal(sx, 0.100, 0.062, (0.0, 0.80, 0.60), (YC + 0.075, 0.402), (YC + 0.040, 0.560), 5)
    pedal(sx - 0.12, 0.045, 0.110, (0.0, 0.85, 0.527), (YC + 0.085, 0.412), (YC + 0.040, 0.570), 6)
    # footrest: a ribbed pad on the cockpit's front wall (which leans ~10 degrees to the driver near the floor); a deep flange
    # holds it wherever the wall stands to within a centimetre
    fr = _frame((0.0, 0.986, 0.168), (0.0, -0.168, 0.986))
    c = np.array([sx + 0.20, YC + 0.016, 0.360])
    b.forms.append(_place(_pillow(S.rrect(0.150, 0.110, 0.030, 4), 0.030, 0.0030, n=2, mat=dark), c, fr, -0.026))
    b.forms.append(_place(_pillow(S.rrect(0.126, 0.086, 0.020, 4), 0.012, 0.0045, n=2, mat=dark), c, fr, -0.004))
    for v in np.linspace(-0.030, 0.030, 5):
        rib = S.rounded_panel(S.stadium(0.108, 0.0046, 4), 0.0030, r_edge=0.0013, n=2, mat=rub)
        b.forms.append(_place(rib, c + fr[1] * v, fr, 0.0075))


def _mirror(b):
    dark = M["trim_dark"]
    # rear-view mirror: a pillow housing on a stalk, the glass a flat stadium on its face; yawed toward the driver, tipped down
    c = np.array(L.MIRROR_C, float)
    rot = lambda m: m.rot(-3.0, 0.0, -14.0, about=c)               # noqa: E731
    depth = 0.032
    house = _pillow(S.stadium(0.245, 0.070, 8), depth, 0.012, n=3, dome=(0.014, 0.0), mat=dark)    # face (glass side) flat
    b.forms.append(rot(_place(house, c, (-EX, EZ, EY), -depth)))
    b.decals.append(rot(S.quad_in_plane(c + EY * 0.0006, EY, EZ, (0.212, 0.040), mat=M["mirror_glass"])))
    # the stalk: a ball joint in the housing's back, a short tapered arm up to a button on the header
    gn3 = np.array([0.0, _GN[0], _GN[1]])
    mount = np.array([0.0, *(_G0 + (GLASS_LEN - L.WS_FRAME) * _GD + 0.004 * _GN)])      # on the header, under its frame
    root = rot(Mesh(np.array([[0.0, -0.030, 0.026]]) + c)).V[0]
    b.forms.append(S.tube(np.array([root, (root + mount) / 2 + np.array([0.0, 0.004, 0.0]), mount]),
                          np.array([0.0062, 0.0055, 0.0050]), sides=14, mat=dark))
    ball = S.lathe(_round_profile([(0.0, -0.011), (0.0105, -0.005), (0.0105, 0.005), (0.0, 0.011)],
                                  [0, 0.004, 0.004, 0], 3), seg=18, mat=dark)
    b.forms.append(ball.moved(root))
    btn = S.lathe(_round_profile([(0.0150, 0.0), (0.0150, 0.0040), (0.0, 0.0040)], [0, 0.003, 0], 3), seg=20, mat=dark)
    b.forms.append(btn.apply(S.rotation_between(EZ, gn3), mount - gn3 * 0.003))


# ============================================================================================================ sun visors
# Two padded pads parallel to the glass, a centimetre inside it, hanging under the header, each from a chrome pivot rod along
# its top edge; the two rods lie on ONE line (VISOR_PIVOT, along X). They are a part of their own: the object `<car>_visors`,
# whose origin is on that line and whose rest pose (rotation 0) is the visors down against the glass. The builder turns it
# about its local X by VISOR_FLIP_DEG * (1 - visors): 0 = flipped up, level and pointing back from the header (a real visor
# under a folded top), 1 = down on the glass.
V_TOP = GLASS_LEN - L.WS_FRAME - 0.008                              # the pads' top edge, along the glass from its foot
V_CENTRE = V_TOP - 0.080                                            # the pads' centre
V_ROD = V_CENTRE + 0.082                                            # the rods: 2 mm above the pads' top edge
V_STANDOFF, V_THICK = 0.010, 0.022                                  # a centimetre inside the glass, 22 mm thick
VISOR_PIVOT = tuple(float(v) for v in _G0 + V_ROD * _GD + (V_STANDOFF + 0.5 * V_THICK) * _GN)    # (y, z) of the rods' axis
VISOR_ORIGIN = (0.0, *VISOR_PIVOT)                                   # the visors part's origin, on that axis


def _visors_mesh():
    """Both visors (pads, stitching, pivot rods, clips) in the part's own frame: the car's frame with the origin on the rods'
    axis, so the part turns about its local X."""
    fr = _frame(np.array([0.0, _GN[0], _GN[1]]), np.array([0.0, _GD[0], _GD[1]]))
    parts = []
    for x in (0.36, -0.36):
        centre = np.array([x, *(_G0 + V_CENTRE * _GD)])
        outline = S.rrect(0.380, 0.160, 0.040, 5)
        parts.append(_place(_pillow(outline, V_THICK, 0.0095, n=3, dome=(0.0, 0.003), mat=M["vinyl"]), centre, fr, V_STANDOFF))
        parts.append(S.ribbon(_to3(S.rrect(0.340, 0.120, 0.030, 4), centre, fr, 0.0330), 0.0014, fr[2], mat=M["seam"],
                              closed=True))
        # the pivot rod along the top edge, and a clip at the inner end
        top = centre + fr[1] * 0.082 + fr[2] * (V_STANDOFF + 0.5 * V_THICK)
        parts.append(S.tube(np.array([top - fr[0] * 0.20, top + fr[0] * 0.20]), 0.0036, sides=10, mat=M["chrome"]))
        inner = 1.0 if x > 0 else -1.0                                # toward the centre line: +right (-X) for the +X visor
        clip = _pillow(S.rrect(0.030, 0.022, 0.008, 3), 0.014, 0.0045, n=2, mat=M["trim_dark"])
        parts.append(_place(clip, centre + fr[0] * inner * 0.185 + fr[1] * 0.076, fr, 0.012))
    return S.merge(parts).moved(-np.array(VISOR_ORIGIN))


# ========================================================================================================= the module
def static_shells():
    return {"dash": S.smooth(dash_cage(), levels=2), "binnacle": S.smooth(hood_cage(), levels=2),
            "stack": S.smooth(stack_cage(), levels=2)}


def static_parts():
    b = _Bag()
    _pad_details(b)
    _wood(b)
    _glove(b)
    _end_vents(b)
    _bezel(b)
    _cluster_display(b)
    _stack_face(b)
    _column(b)
    _pedals(b)
    _mirror(b)
    return {"interior": S.merge(b.forms), "decals": S.merge(b.decals)}


def dynamic_parts():
    return {"steering_wheel": S.Part(_wheel_mesh(), tuple(float(x) for x in WHEEL_C), WHEEL_ROT),
            "cassette": S.Part(_cassette_mesh(), (0.0, float(L.TAPE_IN_Y), float(L.DECK_Z)), (0.0, 0.0, 0.0)),
            "visors": S.Part(_visors_mesh(), VISOR_ORIGIN, (0.0, 0.0, 0.0))}


# ===================================================================================================== card exports
def _surface(center, normal, up, size):
    return {"center": _lst(center), "normal": _lst(normal), "up": _lst(up), "size": [float(size[0]), float(size[1])]}


def _cluster_surface(uv, size):
    return _surface(CC + uv[0] * CR + uv[1] * CU + TEXT_LIFT * CN, CN, CU, size)


SURFACES = {
    "cluster": _cluster_surface((0.0, 0.0), (CW, CH)),
    "speed": _cluster_surface(SPEED_UV, SPEED_SIZE),
    "bars": _cluster_surface(BARS_UV, (0.236, 0.030)),
    "radio": _surface((0.0, DY + 3 * DECK_LIFT, WIN_Z), (0, 1, 0), (0, 0, 1), (0.110, 0.028)),
    "cassette_label": _surface((0.0, L.TAPE_IN_Y, L.DECK_Z + 0.0008), (0, 0, 1), (0, -1, 0), (0.062, 0.034)),
}

# the speed readout's digit grid, in the `speed` surface's own frame (u to the viewer's right from its centre, v up): three
# seven-segment cells side by side, cell k centred at u = (k - 1) * pitch; each digit is DIGIT_W wide, so its right edge is
# (pitch - DIGIT_W) / 2 inside the cell's right edge (a right-aligned numeral ends there). The ghost decimal dots sit at
# the lower right of cells 0 and 1.
SPEED_GRID = {"columns": 3, "pitch": round(SPEED_SIZE[0] / 3, 5), "digit_width": DIGIT_W, "digit_height": DIGIT_H,
              "stroke": DIGIT_T, "dots": 2}
GRIDS = {"speed": SPEED_GRID}                                        # surfaces that document an extra layout (the card)

ANCHORS = {
    "slot": {"point": _lst(L.SLOT_C), "dir": [0.0, -1.0, 0.0]},
    "mirror": {"point": _lst(L.MIRROR_C)},
}

LOOKS = {"dash": _lst(L.BINNACLE_C), "cassette": _lst(L.SLOT_C)}

_bar_u = BARS_UV[0] - BARS_SIZE[0] / 2 + (BARS_SIZE[0] - (BAR_CELLS - 1) * BAR_GAP) / BAR_CELLS / 2
BAR_GRAPH = {                                    # cell 0 is the leftmost as the driver sees it (car x largest)
    "cells": BAR_CELLS,
    "first": _lst(CC + _bar_u * CR + BARS_UV[1] * CU + LIFT * CN),
    "last": _lst(CC - _bar_u * CR + BARS_UV[1] * CU + LIFT * CN),
    "axis": _lst(CR),
}
