"""Seats, floor and the interior side details of the 1980s convertible: upholstered, pleated, rolled and piped forms, no boxes.

    static_shells()   {"seat_L", "seat_R", "bench", "console", "floor", "door_L", "door_R", "quarter_L", "quarter_R"}: creased
                      cages (`core.shell.smooth`: level 2 for the seats, 1 for the rest; the builder bakes one level more) the builder turns
                      into Subdivision Surface objects `<prop>_<key>`
    static_parts()    {"interior", "decals"}: the small plain parts (chrome headrest posts, recline knobs, seat rails, piping beads,
                      the shifter's gate, lever and knob, the parking brake, wood panels, ash tray, the door's strap, switches,
                      courtesy lamp and speaker, lock pins, floor mats) and the graphic layers (the PRNDL squares, the pinstripe
                      round the door's wood)
    SEAT              the seat numbers the card and the poses use (cushion top, the back's plane, the headrest's box)
    ANCHORS, LOOKS    the cabin's handles and pulls (shifter, handbrake, console lid, armrests, door pulls, lock pins)

Construction. A `pad` is a closed tube along y whose rings are half sections (a pleated insert with a border, a bolster, a wall
and an underside, mirrored); the rings sit at the pleat crowns and troughs, the trough rings are creased over the insert's width
only (a partial-ring crease: the seam stops at the piping), the insert's border is creased along the pad, and a rolled end is a
few rings that shrink along a quarter ellipse (plan corners and rolled edges come from the shrink, the Subdivision Surface does
the rounding). Cushions, backs, headrests, the bench, the console and its lid, the floor carpet with the tunnel, the wheel-house
humps, the door and quarter panels and their pleated inserts are pads, each in its own frame; the seat's back and headrest are
built upright and tipped by the recline. Details that must lie on the smooth surface get it from exact maths, not from Blender: the
piping beads on the Catmull-Clark limit of the cage (`limit_grid`), the carpet and the door panels on the analytic limit curve of
the body's own cockpit section (`inner`), so they follow the body when its section changes.

Frames and numbers. Car coordinates; the driver's seat is built about x = 0 (its outer side is +x) and moved to SEAT_X, the
passenger's is its mirror image (the recline knob is on the outer side of each). The front seat is: cushion top CUSHION_TOP under
the hip joint (a crown of the pleats), pleats every 8.5 cm, bolsters 3.6 cm over the crowns; the back's crown plane passes through
BACK_BASE_Y at the cushion top and is 10 cm behind the hip joint at hip height, tipped BACK_RECLINE_DEG; the headrest stands
HEAD_STANDOFF back along the back's normal and stays inside its collider box (SEAT["headrest_center"], ["headrest_size"]).
Roles: upholstery `vinyl` (pleated inserts, bolster tops, door pleats, door-top rolls) and `vinyl_dark` (walls, door panels), plastic
`trim_dark`, bright metal `chrome`, wood `wood`, carpet `carpet`, lit squares `vfd_lit`."""
import math
from functools import lru_cache

import numpy as np

from ....core import shell as S
from . import convertible_body as BODY
from . import convertible_layout as L
from .convertible_layout import M

# ------------------------------------------------------------------------------------------------------ numbers
GROUP = "interior"                             # the builder's group of the plain parts (the graphic layers go to "decals")
SX = L.SEAT_X
HW = L.CUSHION_W / 2.0                         # half width of a front cushion and of its backrest
CY0, CY1 = L.CUSHION_Y
CZT = L.CUSHION_TOP
REC = L.BACK_RECLINE_DEG
SIN_R, COS_R = math.sin(math.radians(REC)), math.cos(math.radians(REC))
BACK_Y0 = L.BACK_BASE_Y                        # y of the back's front plane (the crowns of the pleats) at cushion-top height
BACK_GAP = BACK_Y0 + (L.HIP_Z - CZT) * math.tan(math.radians(REC)) - L.HIP_Y     # how far it passes behind the hip joint
HEAD_SIZE = (0.23, 0.10, 0.165)                # pillow headrest (w, depth, h) = its collider box, which the mesh stays inside
HEAD_STANDOFF = 0.03                           # the pillow stands this far back (along the back's normal, away from the sitter)
HEAD_V, HEAD_W = 0.716, 0.04 + HEAD_STANDOFF   # centre along the back's axis and behind its front plane
HEAD_CROWN = 0.010                             # the pillow's front bulges this far (its body is that much shallower)

PITCH = 0.085                                  # pleat pitch of the front seats
INSERT = 0.15                                  # half width of the pleated insert
BACK_AXIS = np.array([0.0, SIN_R, COS_R])      # up the backrest
BACK_NORM = np.array([0.0, -COS_R, SIN_R])     # out of its front face (toward the sitter)
R_BACK = np.stack([np.array([1.0, 0.0, 0.0]), BACK_AXIS, BACK_NORM], 1)      # local (u, v, n) -> car


# ------------------------------------------------------------------------------------------------------ helpers
def _ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# nominal half section of an upholstered pad, from the top centre round the +x side to the bottom centre. Top points k = 0..6:
# the pleated insert (0..3, the border at 3), the bolster slope (4), the bolster crest (5) and its roll-over (6); then the wall
# and the underside (7..12).
_BOLSTER = np.array([0.00, 0.00, 0.08, 0.25, 0.65, 1.00, 0.72])            # share of the bolster height at each top point
_PLEAT = np.array([1.00, 1.00, 0.90, 0.55, 0.15, 0.00, 0.00])             # share of the pleat dip
#   x / hw, depth below the top as a fraction of the thickness   (k = 7..12)
_SIDE = np.array([(1.000, 0.04), (1.000, 0.34), (0.985, 0.66), (0.900, 0.92), (0.560, 1.00), (0.000, 1.00)])
K = 13                                            # points of a half section (every section function returns K of them)


def sec_upholstery(hw, zt, thick, bolster=0.0, pleat=0.0, f=1.0, edge=0.10, dish=None):
    """(K, 2) points (x >= 0, z) of the half section of an upholstered pad: top surface at zt (+ bolster raise + pleat dip + the
    function dish(x) if given), `thick` deep, the bolsters `edge` wide, shrunk by f about its middle (the rolled ends of a pad)."""
    xb = hw - edge
    xs = np.array([0.0, 0.4 * xb, 0.77 * xb, xb, hw - 0.06, hw - 0.028, hw - 0.009])
    z = zt + bolster * _BOLSTER + pleat * _PLEAT
    if dish is not None:
        z = z + dish(xs)
    top = np.stack([xs, z], 1)
    side = np.stack([hw * _SIDE[:, 0], zt - thick * _SIDE[:, 1] + 0.1 * bolster * (_SIDE[:, 1] < 0.1)], 1)
    P = np.concatenate([top, side])
    if f != 1.0:
        zc = zt - 0.5 * thick
        P = np.stack([P[:, 0] * f, zc + (P[:, 1] - zc) * f], 1)
    return P


def sec_round(hw, zt, thick, r=0.03, f=1.0, slope=0.0, rb=0.012, pleat=0.0):
    """(13, 2) half section of a plastic body: a flat top to hw - r, a quarter round of radius r, a wall that flares out by
    `slope` toward the bottom, a small bottom round rb; `pleat` dips the top points (a pleated panel)."""
    a = np.radians([90.0, 60.0, 30.0, 0.0])
    zb = zt - thick
    h = thick - r - rb
    P = np.concatenate([[(0.0, zt), (0.5 * (hw - r), zt)],
                        np.stack([hw - r + r * np.cos(a), zt - r + r * np.sin(a)], 1),
                        [(hw + 0.3 * slope, zt - r - 0.25 * h), (hw + 0.7 * slope, zt - r - 0.6 * h), (hw + slope, zb + rb),
                         (hw + slope - 0.35 * rb, zb + 0.1 * rb), (hw + slope - rb, zb), (0.5 * (hw + slope - rb), zb), (0.0, zb)]])
    if pleat:
        P[:6, 1] += pleat * np.array([1.0, 1.0, 0.8, 0.4, 0.12, 0.0])
    if f != 1.0:
        zc = zt - 0.5 * thick
        P = np.stack([P[:, 0] * f, zc + (P[:, 1] - zc) * f], 1)
    return P


def sec_slab(hw, zt, thick, r=0.010, rb=0.005, f=1.0, pleat=0.0):
    """(9, 2) half section of a thin pleated panel (a door insert): flat top, rounded edge r, a short wall, flat back."""
    a = np.radians([90.0, 45.0, 0.0])
    zb = zt - thick
    P = np.concatenate([[(0.0, zt), (0.5 * (hw - r), zt)],
                        np.stack([hw - r + r * np.cos(a), zt - r + r * np.sin(a)], 1),
                        [(hw, zt - r - 0.5 * max(thick - r - rb, 0.0)), (hw - 0.35 * rb, zb + 0.1 * rb), (hw - rb, zb), (0.0, zb)]])
    if pleat:
        P[:5, 1] += pleat * np.array([1.0, 1.0, 0.7, 0.25, 0.0])
    if f != 1.0:
        zc = zt - 0.5 * thick
        P = np.stack([P[:, 0] * f, zc + (P[:, 1] - zc) * f], 1)
    return P


def _full_ring(P):
    """(2 (K - 1), 2): the half section, then its mirror image back to the top centre (the centre points once)."""
    return np.concatenate([P, P[-2:0:-1] * (-1.0, 1.0)])


DIMS = {}                                           # id(pad mesh) -> (mesh, rings, points per ring): the ring layout of a cage


def ring_dims(mesh):
    """(rings, points per ring) of a cage built by pad() (its ring vertices come first, then the caps')."""
    return DIMS[id(mesh)][1:]


def _xf(mesh, R, t):
    """mesh.apply(R, t) that keeps the pad's ring layout (see DIMS)."""
    out = mesh.apply(R, t)
    if id(mesh) in DIMS:
        DIMS[id(out)] = (out,) + DIMS[id(mesh)][1:]
    return out


def _col(k, side, m):
    """Column of half-section point k on the +x (side +1) or -x side of a full ring of m points."""
    return k if side > 0 or k in (0, m // 2) else m - k


def pad(stations, section, mat_pale, mat_dark, cap_bulge=(0.0, 0.0), insert_k=3, pale_k=6, border_k=3,
        trough_crease=0.6, border_crease=0.45, cap_mat=(None, None)):
    """A pad along y: `stations` = [dict(y=, f=, trough=bool, **args of `section`)] one ring each (the half section
    section(**args, f=f) mirrored). Returns the cage Mesh (closed, outward normals): the trough rings creased over the insert
    (points 0..insert_k), the insert border (point border_k) creased along the pad, the top (points 0..pale_k) `mat_pale` and
    the wall and underside `mat_dark`; caps take cap_mat or the pale / dark role."""
    rings = []
    for st in stations:
        args = {k: v for k, v in st.items() if k not in ("y", "trough")}
        R = _full_ring(section(**args))
        rings.append(np.stack([R[:, 0], np.full(len(R), st["y"]), R[:, 1]], 1))
    rings = np.array(rings)
    ns, m = rings.shape[:2]
    cols = np.zeros(m)
    if border_k is not None:
        cols[[_col(border_k, +1, m), _col(border_k, -1, m)]] = border_crease
    mesh = S.cage(rings, closed=True, caps=("fan", "fan"), cap_axes=((0, -1, 0), (0, 1, 0)), cap_bulge=cap_bulge, cap_rings=2,
                  crease_cols=cols, mat=mat_dark)
    pairs, w = [], []
    for i, st in enumerate(stations):
        if not st.get("trough"):
            continue
        for side in (+1, -1):
            for k in range(insert_k):
                pairs.append((i * m + _col(k, side, m), i * m + _col(k + 1, side, m)))
                w.append(trough_crease)
    if pairs:
        mesh.crease(np.array(pairs), np.array(w))
    pale_cols = {_col(k, +1, m) for k in range(pale_k + 1)} | {_col(k, -1, m) for k in range(1, pale_k + 1)}
    sel = np.array([(j in pale_cols) and (((j + 1) % m) in pale_cols) for j in range(m)])
    qm = np.full((ns - 1, m), mat_dark, np.int32)
    qm[:, sel] = mat_pale
    nbody = (ns - 1) * m
    mesh.Qm[:nbody] = qm.ravel()
    ncap = m * 2
    c0, c1 = cap_mat[0] if cap_mat[0] is not None else mat_pale, cap_mat[1] if cap_mat[1] is not None else mat_dark
    mesh.Qm[nbody:nbody + ncap] = c0
    mesh.Qm[nbody + ncap:nbody + 2 * ncap] = c1
    mesh.Tm[:m] = c0
    mesh.Tm[m:2 * m] = c1
    DIMS[id(mesh)] = (mesh, ns, m)
    return mesh


# ------------------------------------------------------------------------------------------------------ front seat
HIP_Y = L.HIP_Y
BACK_H = L.BACK_H


LIFT = 0.0062                                  # the Subdivision Surface pulls in from the crowns: raise the cage so the surface is CUSHION_TOP


def _dish(y):
    """Height of the cushion's crowns on the centre line: lowest (= CUSHION_TOP) under the hip, rising a little toward the knee
    and more toward the back."""
    y = np.asarray(y, float)
    return CZT + LIFT + 0.006 * _ss((-0.04 - y) / 0.10) + 0.012 * _ss((y - 0.22) / 0.20)


def _end_scale(s, r):
    """Scale of a rounded end at distance s from its tip: a quarter ellipse of radius r (>= 0.38 so the ring stays a ring)."""
    s = np.asarray(s, float)
    return np.where(s >= r, 1.0, np.maximum(np.sqrt(np.maximum(1.0 - ((r - s) / r) ** 2, 0.0)), 0.38))


@lru_cache(maxsize=None)
def cushion_cage():
    h, thick, R = 0.017, 0.125, 0.075
    st = []

    def add(y, pleat, trough=False):
        st.append(dict(y=y, hw=HW, zt=float(_dish(y)), thick=thick, bolster=0.036 * (0.6 + 0.4 * float(_ss((y - CY0) / 0.25))),
                       pleat=pleat, f=float(_end_scale(y - CY0, R)), trough=trough))

    for s in (0.0, 0.005, 0.014, 0.030):                          # the rolled front (waterfall)
        add(CY0 + s, 0.0)
    for k in range(-3, 3):                                         # troughs between the crowns, the crown of band 0 on the hip
        add(HIP_Y + PITCH * (k + 0.5), -h, True)
        if k < 2:
            add(HIP_Y + PITCH * (k + 1), 0.0)
    add(CY1, -0.4 * h)
    return pad(st, sec_upholstery, M["vinyl"], M["vinyl_dark"], cap_bulge=(0.012, 0.0))


@lru_cache(maxsize=None)
def back_cage():
    h, R = 0.017, 0.05
    thick = lambda v: 0.20 + (0.13 - 0.20) * float(np.clip(v / BACK_H, 0.0, 1.0))
    zt = lambda v: 0.013 * float(np.exp(-((v - 0.25) / 0.10) ** 2))                    # lumbar bulge
    wing = lambda v: 0.046 * (0.55 + 0.45 * float(_ss((v - 0.05) / 0.15))) * (1.0 - 0.4 * float(_ss((v - 0.35) / 0.2)))
    st = []

    def add(v, pleat, trough=False):
        st.append(dict(y=v, hw=HW, zt=zt(v), thick=thick(v), bolster=wing(v), pleat=pleat, f=float(_end_scale(BACK_H - v, R)),
                       trough=trough))

    add(-0.05, -0.5 * h)
    for k in range(-1, 6):
        add(0.06 + PITCH * (k + 0.5), -h, True)
        if k < 5:
            add(0.06 + PITCH * (k + 1), 0.0)
    for s in (0.05, 0.026, 0.012, 0.004, 0.0):
        add(BACK_H - s, -0.3 * h * (s > 0.03))
    m = pad(st, sec_upholstery, M["vinyl"], M["vinyl_dark"], cap_bulge=(0.0, 0.010))
    return _xf(m, R_BACK, (0.0, BACK_Y0, CZT))


@lru_cache(maxsize=None)
def headrest_cage():
    """The pillow: a pad along the back's axis (stations up the headrest), a face that bulges HEAD_CROWN in the middle; the
    box HEAD_SIZE (the hair collider) just contains it."""
    hh = 0.5 * HEAD_SIZE[2]
    face0 = -HEAD_W + 0.5 * HEAD_SIZE[1]                     # the crown of the face, in the back's normal direction
    back = -HEAD_W - 0.5 * HEAD_SIZE[1]
    vs = [-hh + s for s in (0.0, 0.006, 0.020, 0.042, 0.070)] + [0.0] + [hh - s for s in (0.070, 0.042, 0.020, 0.006, 0.0)]
    st = []
    for v in vs:
        zt = face0 - HEAD_CROWN * (v / hh) ** 2
        st.append(dict(y=v, hw=0.5 * HEAD_SIZE[0], zt=zt, thick=zt - back, r=0.040, slope=0.0, rb=0.030,
                       f=float(_end_scale(v + hh, 0.050)) * float(_end_scale(hh - v, 0.050))))
    m = pad(st, sec_round, M["vinyl"], M["vinyl"], insert_k=0, border_k=None, pale_k=12)
    return _xf(m, R_BACK, np.array([0.0, BACK_Y0, CZT]) + HEAD_V * BACK_AXIS)


@lru_cache(maxsize=None)
def pan_cage():
    """The dark plastic pan under the cushion."""
    st = [dict(y=y, hw=0.205, zt=0.352, thick=0.074, r=0.03, slope=0.0,
               f=float(_end_scale(y + 0.07, 0.035)) * float(_end_scale(0.40 - y, 0.03)))
          for y in (-0.07, -0.04, 0.05, 0.20, 0.36, 0.40)]
    return pad(st, sec_round, M["trim_dark"], M["trim_dark"], insert_k=0, border_k=None)


@lru_cache(maxsize=None)
def shield_panel():
    """The plastic cover on the outer side of the cushion, under the bolster: a plate on the wall with a rolled rim."""
    outline = S.fillet(np.array([(-0.03, 0.335), (0.37, 0.335), (0.37, 0.404), (0.20, 0.418), (-0.03, 0.404)]),
                       [0.0, 0.02, 0.02, 0.02, 0.02], n=2, closed=True)
    m = S.rounded_panel(outline, 0.016, r_edge=0.007, n=2, dome=0.003, mat=M["trim_dark"])
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])          # local (x, y, z) -> (y, z, x): the face looks +x
    return m.apply(R, (HW - 0.009, 0.0, 0.0))


def front_seat_cage():
    """The whole upholstered seat as one cage mesh in seat-local x (the driver's seat; the outer side is +x)."""
    return S.merge([cushion_cage(), back_cage(), headrest_cage(), pan_cage(), shield_panel()])


def _on_axis(m, direction, pos):
    return m.apply(S.rotation_between((0.0, 0.0, 1.0), direction), pos)


def seat_parts():
    """Chrome posts, collars, recline knob, rails of the driver's seat (seat-local x)."""
    out = []
    for u in (-0.055, 0.055):
        a = np.array([0.0, BACK_Y0, CZT]) + 0.50 * BACK_AXIS - HEAD_W * BACK_NORM + (u, 0, 0)
        b = np.array([0.0, BACK_Y0, CZT]) + (HEAD_V + 0.02) * BACK_AXIS - HEAD_W * BACK_NORM + (u, 0, 0)
        out.append(S.tube(np.array([a, b]), 0.0095, sides=14, mat=M["chrome"], up=(1.0, 0.0, 0.0)))
        collar = S.lathe([(0.0145, 0.0), (0.0145, 0.006), (0.0125, 0.0105), (0.0095, 0.012), (0.0, 0.012)], seg=16, mat=M["trim_dark"])
        out.append(_on_axis(collar, BACK_AXIS, np.array([0.0, BACK_Y0, CZT]) + (BACK_H - 0.006) * BACK_AXIS - HEAD_W * BACK_NORM + (u, 0, 0)))
    knob = S.lathe([(0.0, 0.0), (0.028, 0.0), (0.031, 0.003), (0.031, 0.009), (0.027, 0.0145), (0.019, 0.0172), (0.0, 0.0172)],
                   seg=24, mat=M["chrome"])
    out.append(_on_axis(knob, (1.0, 0.0, 0.0), (HW + 0.0075, 0.31, 0.395)))
    for s in (-1, 1):                                               # seat rails under the pan
        y = np.linspace(-0.12, 0.42, 14)
        path = np.stack([np.full(14, s * 0.13), y, np.full(14, 0.262)], 1)
        taper = np.minimum(1.0, np.minimum(y - y[0], y[-1] - y) / 0.03 + 0.25)
        out.append(S.sweep(path, S.rrect(0.05, 0.05, 0.014, 3), scale=taper[:, None] * [1.0, 1.0], up=(0, 0, 1), mat=M["trim_dark"]))
    return out


# ------------------------------------------------------------------------------------------------------ limit surface
def limit_grid(cage, ns, m):
    """Positions on the Catmull-Clark limit surface of the cage's ring vertices (ns, m, 3): the exact limit stencil of a regular
    vertex, (16 v + 4 (edge neighbours) + (diagonal neighbours)) / 36, valid away from creases and cap poles; the two end rings
    follow their neighbour's displacement. A Subdivision Surface pulls in from its control polygon, so a bead laid on the
    polygon would hover (or hide): the limit sits where the surface really is."""
    G = cage.V[:ns * m].reshape(ns, m, 3)
    up, dn = np.roll(G, -1, axis=1), np.roll(G, 1, axis=1)
    lim = G.copy()
    for i in range(1, ns - 1):
        a, b = G[i - 1], G[i + 1]
        lim[i] = (16 * G[i] + 4 * (up[i] + dn[i] + a + b) + np.roll(a, -1, 0) + np.roll(a, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 0)) / 36.0
    lim[0] = G[0] + (lim[1] - G[1])
    lim[-1] = G[-1] + (lim[-2] - G[-2])
    return lim


def piping(cage, ns, m, k0, k1, side, r, mat, taper=0.03, sink=0.4):
    """A bead of radius r along the seam between half-section points k0 and k1 of a pad's rings, on the limit surface, sunk `sink`
    of its radius, tapering to nothing over the last `taper` metres of each end. Returns a Mesh (a tube)."""
    lim = limit_grid(cage, ns, m)
    ca, cb = _col(k0, side, m), _col(k1, side, m)
    A, B = lim[:, ca], lim[:, cb]
    P = 0.5 * (A + B)
    ctr = lim.mean(1)
    nxt = np.gradient(P, axis=0)
    nrm = np.cross(B - A, nxt)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
    nrm *= np.where(np.einsum("ij,ij->i", nrm, P - ctr) < 0, -1.0, 1.0)[:, None]
    C = P - nrm * sink * r
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    k = np.minimum(1.0, np.minimum(s, s[-1] - s) / taper) ** 0.6 * 0.9 + 0.1
    k[[0, -1]] = 0.25
    return S.tube(C, r * k, sides=8, mat=mat)


# ------------------------------------------------------------------------------------------------------ the rear bench
BY0, BY1 = L.REAR_SEAT_Y
BZ = L.REAR_SEAT_TOP
BHW = L.REAR_SEAT_W / 2.0
B_REC = L.REAR_BACK_RECLINE_DEG
B_AXIS = np.array([0.0, math.sin(math.radians(B_REC)), math.cos(math.radians(B_REC))])
B_NORM = np.array([0.0, -math.cos(math.radians(B_REC)), math.sin(math.radians(B_REC))])
R_BENCH = np.stack([np.array([1.0, 0.0, 0.0]), B_AXIS, B_NORM], 1)
B_BACK_Y = BY1 - 0.10                                  # where the back's front plane meets the cushion top
B_BACK_TOP = L.Z_DECK - 0.04                           # the back ends under the deck lid's line
B_BACK_H = (B_BACK_TOP - BZ) / math.cos(math.radians(B_REC))


def _bench_dish(x):
    """Two shallow dishes (the seating places) with a low ridge between them."""
    x = np.asarray(x, float)
    return 0.010 * np.exp(-(x / 0.10) ** 2) - 0.012 * np.exp(-((x - 0.30) / 0.12) ** 2)


@lru_cache(maxsize=None)
def bench_cushion_cage():
    h, thick, pitch = 0.013, 0.12, 0.08
    st = []

    def add(y, pleat, trough=False):
        st.append(dict(y=y, hw=BHW, zt=BZ, thick=thick, bolster=0.022, pleat=pleat, f=float(_end_scale(y - BY0, 0.05)), edge=0.09,
                       dish=_bench_dish, trough=trough))

    for s in (0.0, 0.005, 0.014, 0.030):
        add(BY0 + s, 0.0)
    for k in range(5):
        add(BY0 + 0.072 + pitch * k, -h, True)
        add(BY0 + 0.072 + pitch * (k + 0.5), 0.0)
    add(BY1, -0.4 * h)
    return pad(st, sec_upholstery, M["vinyl"], M["vinyl_dark"], cap_bulge=(0.010, 0.0))


@lru_cache(maxsize=None)
def bench_back_cage():
    h, R, pitch = 0.013, 0.04, 0.08
    thick = lambda v: 0.100 + (0.070 - 0.100) * float(np.clip(v / B_BACK_H, 0.0, 1.0))
    st = []

    def add(v, pleat, trough=False):
        st.append(dict(y=v, hw=BHW, zt=0.0, thick=thick(v), bolster=0.022, pleat=pleat, f=float(_end_scale(B_BACK_H - v, R)),
                       edge=0.09, trough=trough))

    add(-0.05, -0.5 * h)
    for k in range(4):
        add(0.045 + pitch * k, -h, True)
        add(0.045 + pitch * (k + 0.5), 0.0)
    for s in (0.04, 0.022, 0.010, 0.003, 0.0):
        add(B_BACK_H - s, -0.3 * h * (s > 0.03))
    m = pad(st, sec_upholstery, M["vinyl"], M["vinyl_dark"], cap_bulge=(0.0, 0.008))
    return _xf(m, R_BENCH, (0.0, B_BACK_Y, BZ))


# ------------------------------------------------------------------------------------------------------ the console
CX = L.CONSOLE_X
CYF, CYR = L.CONSOLE_Y
CZ = L.CONSOLE_Z
LID_Y = (0.29, 0.645)
LID_TOP = 0.64


@lru_cache(maxsize=None)
def console_cage():
    thick = CZ - 0.26
    st = []

    def add(y):
        f = float(_end_scale(y - CYF, 0.016)) * float(_end_scale(CYR - y, 0.040))
        st.append(dict(y=y, hw=CX, zt=CZ, thick=thick, r=0.04, slope=0.035, rb=0.014, f=f))

    for s in (0.0, 0.003, 0.008, 0.016):
        add(CYF + s)
    for y in (-0.15, 0.0, 0.15, 0.30, 0.45):
        add(y)
    for s in (0.040, 0.020, 0.008, 0.002, 0.0):
        add(CYR - s)
    return pad(st, sec_round, M["trim_dark"], M["trim_dark"], insert_k=0, border_k=None, pale_k=6)


@lru_cache(maxsize=None)
def lid_cage():
    thick, R = 0.09, 0.045
    st = []
    y0, y1 = LID_Y
    for y in (y0, y0 + 0.004, y0 + 0.012, y0 + 0.026, y0 + 0.05, 0.5 * (y0 + y1), y1 - 0.05, y1 - 0.026, y1 - 0.012, y1 - 0.004, y1):
        f = float(_end_scale(y - y0, R)) * float(_end_scale(y1 - y, R))
        st.append(dict(y=y, hw=0.135, zt=LID_TOP, thick=thick, bolster=0.010, f=f, edge=0.05))
    return pad(st, sec_upholstery, M["vinyl"], M["vinyl_dark"], insert_k=0, border_k=3, border_crease=0.35)


# ------------------------------------------------------------------------------------------------------ floor, tunnel, humps
CARPET_T = 0.0085                                  # the carpet's pile on the body's floor: its top is FLOOR_Z (the feet stand on it)


def _carpet_section():
    """Half section of the carpet sheet: the tunnel (rounded, 0.31 high), then the body's floor offset into the cabin by the
    pile, up the wall (and the floor fillet) to the panel's lip, an underside sunk into the body."""
    top = [(0.000, 0.312), (0.075, 0.312), (0.118, 0.305), (0.148, 0.284), (0.165, 0.256)]
    C = inner()[0][::-1]                                                        # from the centre outward and up
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    i0 = int(np.searchsorted(np.maximum.accumulate(C[:, 0]), 0.19))
    i1 = int(np.searchsorted(np.maximum.accumulate(C[:, 1]), Z_CARPET_UP))
    ss = s[i0] + (s[i1] - s[i0]) * np.linspace(0.0, 1.0, 10) ** 0.9
    P = np.stack([np.interp(ss, s, C[:, 0]), np.interp(ss, s, C[:, 1])], 1)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], 1)                                        # into the cabin (up on the floor, -x on the wall)
    offs = P + N * CARPET_T
    lip = P[-1] + N[-1] * 0.5 * CARPET_T + T[-1] * 0.0035
    under = [P[-1] - N[-1] * 0.003, P[len(P) // 2] - N[len(P) // 2] * 0.006, (0.0, float(floor_z(0.0)) - 0.006)]
    return np.array(top + [tuple(p) for p in offs] + [tuple(lip)] + [tuple(p) for p in under])


WELL_R_IN, WELL_R_OUT = 0.38, 0.40                 # the rear wheel well: the body cuts its opening with ARCH_R (0.36) about the axle
WELL_X = (0.60, 0.80)                              # the well's inboard end wall (its face) and its open end, inside the body's wall
WELL_T = 0.02                                      # thickness of the end wall
Y_CARPET_END = L.AXLE_R - WELL_R_OUT - 0.012       # the carpet stops where the well's outside meets the floor


@lru_cache(maxsize=None)
def carpet_cage():
    """The floor carpet with the transmission tunnel, one sheet from the toe board to the rear wheel wells."""
    ys = (L.Y_COWL - 0.02, -0.30, 0.20, 0.66, Y_CARPET_END)
    st = [dict(y=y) for y in ys]
    mesh = pad(st, _carpet_section, M["carpet"], M["carpet"], insert_k=0, border_k=None, pale_k=12)
    return mesh


def wheel_well():
    """The rear wheel well on the +x side, as the cabin sees it: a cup about the axle (a surface of revolution, 2 cm thick,
    outer radius WELL_R_OUT, inner WELL_R_IN) with an end wall at x = 0.60. The body's opening is cut with r = ARCH_R from
    x = 0.58, so nothing of the trim may stand inside r = WELL_R_IN: seen through the opening it is a clean dark well."""
    r0, r1 = WELL_R_IN, WELL_R_OUT
    x0, x1 = WELL_X
    rc, rc2 = 0.02, 0.015
    u = np.linspace(0.0, 0.5 * math.pi, 5)
    prof = [(0.0, x0), (0.10, x0), (0.20, x0), (0.30, x0)] + [(r1 - rc + rc * math.sin(t), x0 + rc - rc * math.cos(t)) for t in u]
    prof += [(r1, x1), (r0, x1), (r0, x0 + WELL_T + rc2)]
    prof += [(r0 - rc2 + rc2 * math.cos(t), x0 + WELL_T + rc2 - rc2 * math.sin(t)) for t in u[1:]]
    prof += [(0.30, x0 + WELL_T), (0.0, x0 + WELL_T)]
    m = S.lathe(prof, seg=96, mat=M["underbody"]).rot(0.0, 90.0, 0.0).moved((0.0, L.AXLE_R, L.WHEEL_Z))

    def outside(c, n):                                         # the cabin sees the end wall's face and the cylinder's outside
        dy, dz = c[:, 1] - L.AXLE_R, c[:, 2] - L.WHEEL_Z
        rho = np.hypot(dy, dz)
        radial = (n[:, 1] * dy + n[:, 2] * dz) / np.maximum(rho, 1e-9)
        return (n[:, 0] < -0.35) | ((radial > 0.35) & (rho > r1 - 0.012))

    m.assign(M["carpet"], lambda c, n: outside(c, n) & (c[:, 2] < 0.46))
    m.assign(M["vinyl_dark"], lambda c, n: outside(c, n) & (c[:, 2] >= 0.46))
    return _clip(m, y_max=Y_CABIN_REAR + 0.06, z_min=BODY.Z_FLOOR - 0.03)     # what the cabin sees: the rest is inside the body


def _clip(m, y_max, z_min):
    """The faces of a mesh that lie in front of y_max and above z_min (every corner; an open patch, the vertices compacted)."""
    ok = (m.V[:, 1] <= y_max) & (m.V[:, 2] >= z_min)
    kq, kt = ok[m.Q].all(1), (ok[m.T].all(1) if len(m.T) else np.zeros(0, bool))
    used = np.unique(np.concatenate([m.Q[kq].ravel(), m.T[kt].ravel()]))
    remap = np.full(len(m.V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return S.Mesh(m.V[used], remap[m.Q[kq]], remap[m.T[kt]], m.Qm[kq], m.Tm[kt])


def well_top(y):
    """Height of the well's outside (the trim's lower edge follows it) at y; -1 where the well is not."""
    dy = L.AXLE_R - y
    return L.WHEEL_Z + math.sqrt(WELL_R_OUT ** 2 - dy * dy) if abs(dy) < WELL_R_OUT else -1.0


@lru_cache(maxsize=None)
def bench_base_cage():
    """The dark plastic base under the rear cushion, down to the floor."""
    y0, y1 = L.REAR_SEAT_Y[0] + 0.02, L.REAR_SEAT_Y[1] - 0.02
    zt = L.REAR_SEAT_TOP - 0.12 + 0.03
    st = [dict(y=y, hw=0.56, zt=zt, thick=zt - (BODY.Z_FLOOR - 0.004), r=0.03, slope=0.0, rb=0.012,
               f=float(_end_scale(y - y0, 0.03)) * float(_end_scale(y1 - y, 0.03))) for y in (y0, y0 + 0.01, y0 + 0.03, 0.5 * (y0 + y1), y1 - 0.03, y1 - 0.01, y1)]
    return pad(st, sec_round, M["trim_dark"], M["trim_dark"], insert_k=0, border_k=None)


# ------------------------------------------------------------------------------------------------------ door panels (+X side)
XW = L.X_WALL_IN                                   # inner face of the body's wall above the bend
DOOR_Y = (L.Y_COWL - 0.03, L.Y_DOOR_REAR - 0.006)  # the door panel; its front end is hidden in the cowl
Y_CABIN_REAR = 1.10                                # the cockpit's rear wall (the body's COCKPIT stations end there)
QTR_Y = (L.Y_DOOR_REAR + 0.006, Y_CABIN_REAR)      # the rear quarter trim


def _bspline_span(p0, p1, p2, p3, u):
    u = np.asarray(u, float)[:, None]
    return (((1 - u) ** 3) * p0 + (3 * u ** 3 - 6 * u ** 2 + 4) * p1 + (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) * p2 + (u ** 3) * p3) / 6.0


def _inner_curve(n=24):
    """(m, 2) points (x, z) on the Subdivision Surface of the body's inner wall and floor, from the door-top cap's inner lip
    (e3) down the wall and across the floor to the centre line. The loft is regular there, so the limit surface is the uniform
    cubic B-spline of the section's control points: exact (it matches the evaluated body within a millimetre) and needs no
    Blender. Read from the body's own section, so it follows the body when that changes."""
    pts = BODY.open_section(BODY.HW, BODY.Z_SILL, L.Z_BELT, L.Z_SHOULDER)
    names = [p.name for p in pts]
    xz = np.array([(p.x, p.z) for p in pts])
    ring = np.concatenate([xz, xz[-2:0:-1] * (-1.0, 1.0)])               # ... e3 t1 t2 .. tc .. t2' t1' ...
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    last = len(xz) - 1
    out = [_bspline_span(ring[i - 1], ring[i], ring[i + 1], ring[i + 2], u) for i in range(names.index("e3"), last)]
    out.append(((ring[last - 1] + 4.0 * ring[last] + ring[last + 1]) / 6.0)[None, :])           # the limit point of tc
    return np.concatenate(out)


@lru_cache(maxsize=None)
def inner():
    """The body's inner wall and floor as (x, z) from the cap's lip to the centre line, plus the monotone tables the lookups use
    (computed on first use, so that importing this module never depends on the body's section)."""
    C = _inner_curve()
    return C, np.maximum.accumulate(C[::-1, 1]), C[::-1, 0], np.maximum.accumulate(C[::-1, 0]), C[::-1, 1]


def wall_x(z):
    """x of the body's inner wall at height z (clamped to its ends; on the floor fillet it is the fillet's x)."""
    _, zs, xs, _, _ = inner()
    return np.interp(z, zs, xs)


def floor_z(x):
    """Height of the body's floor (and the start of its fillet) at |x|."""
    _, _, _, xf, zf = inner()
    return np.interp(np.abs(x), xf, zf)


Z_ROLL = L.Z_BELT - 0.012                          # top of the rolled panel edge: it tucks under the door-top cap's inner lip
#   z,      d (stand-off from the wall),  crease along y,  armrest weight
Z_CARPET_UP = 0.395                                # the carpet sheet climbs the wall to here; the panel starts under its lip
_DOOR = np.array([
    (Z_CARPET_UP - 0.005, 0.015, 0.50, 0.0),
    (0.404, 0.026, 0.60, 0.0), (0.432, 0.027, 0.00, 0.0),                                  # kick strip
    (0.475, 0.022, 0.00, 0.0), (0.510, 0.026, 0.00, 0.5), (0.545, 0.031, 0.00, 1.0),      # lower panel, the armrest band
    (0.580, 0.031, 0.00, 1.0), (0.603, 0.025, 0.40, 0.0),
    (0.650, 0.024, 0.00, 0.0), (0.730, 0.024, 0.00, 0.0), (0.800, 0.026, 0.00, 0.0),      # behind the insert and the wood
    (0.850, 0.028, 0.00, 0.0), (0.872, 0.030, 0.00, 0.0),                                   # the rolled top
    (Z_ROLL - 0.014, 0.028, 0.00, 0.0), (Z_ROLL - 0.005, 0.016, 0.00, 0.0), (Z_ROLL, 0.004, 0.00, 0.0)])
_DOOR_BACK = np.array([(0.80, -0.005), (0.55, -0.005), (Z_CARPET_UP - 0.007, -0.005)])           # buried in the wall
N_DOOR = len(_DOOR) + len(_DOOR_BACK)
_ARM_AMOUNT = 0.022


def _bump(y, a, b, r):
    return float(_ss((y - a) / r) * _ss((b - y) / r))


def _door_ring(y, y0, y1, arm, roll, zlow=None):
    """Closed ring (N_DOOR, 3) of the panel at y: x = wall_x(z) - d. zlow(y): a higher lower edge (the quarter trim stays above
    the wheel well): the panel's heights are squeezed into the space between it and the rolled top."""
    zf, zb = _DOOR[:, 0], _DOOR_BACK[:, 0]
    if zlow is not None:
        z0, z1 = _DOOR_BACK[-1, 0], Z_ROLL
        k = (z1 - zlow(y)) / (z1 - z0)
        zf, zb = z1 - (z1 - zf) * k, z1 - (z1 - zb) * k
    s = 1.0
    if roll[0]:
        s *= float(np.clip(_end_scale(y - y0, roll[0]), 0.25, 1.0))
    if roll[1]:
        s *= float(np.clip(_end_scale(y1 - y, roll[1]), 0.25, 1.0))
    d = _DOOR[:, 1] * s + _ARM_AMOUNT * _DOOR[:, 3] * (_bump(y, arm[0], arm[1], 0.07) if arm else 0.0) * s
    P = np.concatenate([np.stack([wall_x(zf) - d, np.full(len(d), y), zf], 1),
                        np.stack([wall_x(zb) - _DOOR_BACK[:, 1], np.full(len(_DOOR_BACK), y), zb], 1)])
    return P


def _panel_cage(y0, y1, arm, ys_mid, roll=(0.0, 0.02), zlow=None):
    ends0 = [y0 + s for s in (0.0, 0.004, 0.012, 0.024)] if roll[0] else [y0]
    ends1 = [y1 - s for s in (0.024, 0.012, 0.004, 0.0)] if roll[1] else [y1]
    ys = sorted(set([round(v, 6) for v in ends0 + list(ys_mid) + ends1]))
    rings = np.array([_door_ring(y, y0, y1, arm, roll, zlow) for y in ys])
    cols = np.concatenate([_DOOR[:, 2], np.zeros(len(_DOOR_BACK))])
    mesh = S.cage(rings, closed=True, caps=("fan", "fan"), cap_axes=((0, -1, 0), (0, 1, 0)), cap_rings=2, crease_cols=cols,
                  mat=M["vinyl_dark"])
    ns, m = rings.shape[:2]
    colmat = np.full(m, M["vinyl_dark"], np.int32)
    colmat[len(_DOOR) - 4:len(_DOOR) - 1] = M["vinyl"]          # the rolled top is the pale upholstery
    mesh.Qm[:(ns - 1) * m] = np.tile(colmat, ns - 1)
    return mesh


@lru_cache(maxsize=None)
def door_cage():
    arm = (-0.45, 0.36)
    mid = list(np.linspace(-0.62, 0.40, 12)) + [arm[0], arm[1], arm[0] + 0.07, arm[1] - 0.07]
    return _panel_cage(DOOR_Y[0], DOOR_Y[1], arm, mid, roll=(0.0, 0.02))


def quarter_zlow(y):
    """The quarter trim's lower edge: the door panel's, and above the wheel well's outside (1.5 cm under its top) behind it."""
    return max(Z_CARPET_UP - 0.007, well_top(y) - 0.015)


@lru_cache(maxsize=None)
def quarter_cage():
    arm = (QTR_Y[0] + 0.05, QTR_Y[1] - 0.12)
    mid = list(np.linspace(QTR_Y[0] + 0.05, 0.88, 3)) + list(np.linspace(0.92, QTR_Y[1] - 0.02, 8)) + [arm[0], arm[1], arm[0] + 0.07, arm[1] - 0.07]
    return _panel_cage(QTR_Y[0], QTR_Y[1], arm, mid, roll=(0.02, 0.0), zlow=quarter_zlow)


R_INSERT = np.stack([np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])], 1)    # (x_l, y_l, z_l) -> (up, y, in)
R_FACE = np.stack([np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])], 1)      # a panel on the wall: (x_l, y_l, z_l) -> (y, up, in)


def _insert_cage(y0, y1, z0, z1, pitch=0.06, h=0.009, q=0.2):
    """A pleated panel (vertical pleats: flat crowns, narrow creased seams) standing 8 mm proud of the door base."""
    hw = 0.5 * (z1 - z0)
    n = int(round((y1 - y0) / pitch))
    pitch = (y1 - y0) / n
    st = []

    def add(y, pleat, trough, f=1.0):
        st.append(dict(y=y, hw=hw, zt=0.0, thick=0.02, pleat=pleat, f=f, trough=trough))

    add(y0, -h, True, 0.80)
    add(y0 + 0.006, -h, True, 0.93)
    for j in range(n):
        if j:
            add(y0 + pitch * j, -h, True)
        add(y0 + pitch * (j + q), 0.0, False)
        add(y0 + pitch * (j + 1 - q), 0.0, False)
    add(y1 - 0.006, -h, True, 0.93)
    add(y1, -h, True, 0.80)
    m = pad(st, sec_slab, M["vinyl"], M["vinyl_dark"], insert_k=3, pale_k=4, border_k=None, trough_crease=0.8)
    return m.apply(R_INSERT, (XW - 0.032, 0.0, 0.5 * (z0 + z1)))


@lru_cache(maxsize=None)
def door_insert_cage():
    return _insert_cage(-0.42, DOOR_Y[1] - 0.10, 0.585, 0.718)


@lru_cache(maxsize=None)
def quarter_insert_cage():
    return _insert_cage(0.74, 0.98, 0.585, 0.718)


def _face(mesh, x_back, y, z):
    """Stand a part built with its back at z_l = 0, facing +z_l, on the +x wall facing the cabin (-x), centred at (y, z)."""
    return mesh.apply(R_FACE, (x_back, y, z))


def _wall_x(y, z, arm=(-0.45, 0.36)):
    """x of the door panel's cabin-side surface at (y, z) (the cage polygon: flat enough where the parts stand)."""
    ring = _door_ring(y, DOOR_Y[0], DOOR_Y[1], arm, (0.0, 0.02))
    n = len(_DOOR)
    return float(np.interp(z, ring[:n, 2], ring[:n, 0]))


def _strap_section():
    """A ribbed band (u across, v out of the panel), counter-clockwise."""
    w, t = 0.036, 0.011
    ribs = 4
    u = np.linspace(w / 2, -w / 2, 4 * ribs + 1)
    top = np.stack([u, t / 2 + 0.0016 * np.cos(2 * np.pi * ribs * (u + w / 2) / w)], 1)
    bottom = np.stack([np.linspace(-w / 2, w / 2, 5), np.full(5, -t / 2)], 1)
    P = np.concatenate([bottom, top[1:-1]])
    return S.fillet(P, 0.0, closed=True) if False else P


def door_parts():
    """(interior meshes, decal meshes) of the +x door: wood strip, pull strap, switches, courtesy lens, speaker."""
    P, D = [], []
    # --- wood strip with its pinstripe
    y0, y1, zc, h = -0.50, DOOR_Y[1] - 0.08, 0.750, 0.050
    wood = S.rounded_panel(S.rrect(y1 - y0, h, 0.012, 3), 0.006, r_edge=0.003, n=2, uv=True, mat=M["wood"])
    xb = XW - 0.021
    P.append(_face(wood, xb, 0.5 * (y0 + y1), zc))
    ring = S.rrect(y1 - y0 - 0.012, h - 0.012, 0.009, 3)
    path = np.stack([np.full(len(ring), xb - 0.0064), ring[:, 0] + 0.5 * (y0 + y1), ring[:, 1] + zc], 1)
    D.append(S.ribbon(path, 0.0013, (-1.0, 0.0, 0.0), closed=True, mat=M["chrome"]))
    # --- the pull strap: a ribbed band that loops out of the panel between two chrome plates
    ya, yb, zs = -0.22, 0.30, 0.722
    t = np.linspace(0.0, 1.0, 24)
    ys = ya + (yb - ya) * t
    off = 0.030 + 0.030 * np.sin(np.pi * t) ** 0.7
    path = np.stack([XW - off, ys, np.full(len(t), zs)], 1)
    taper = np.clip(np.minimum(t, 1 - t) / 0.08, 0.25, 1.0)
    strap = S.sweep(path, _strap_section(), scale=taper[:, None] * [1.0, 1.0], up=(-1.0, 0.0, 0.0), mat=M["vinyl"])
    P.append(strap if strap.volume() >= 0 else strap.flip())
    for yy in (ya - 0.004, yb + 0.004):
        plate = S.rounded_panel(S.rrect(0.05, 0.052, 0.014, 3), 0.007, r_edge=0.003, n=2, dome=0.0012, mat=M["chrome"])
        P.append(_face(plate, XW - 0.026, yy, zs))
    # --- the window switches on the armrest band: a chrome plate with four dark rockers
    ysw, zsw = -0.36, 0.553
    xs = _wall_x(ysw, zsw)
    frame = S.rounded_frame(S.rrect(0.098, 0.058, 0.012, 3), S.rrect(0.086, 0.046, 0.007, 3), 0.006, r_out=0.0025, r_in=0.0015,
                            floor=0.0045, mat=M["chrome"])
    P.append(_face(frame, xs + 0.004, ysw, zsw))
    for iy, iz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        key = S.rounded_panel(S.superellipse(0.033, 0.018, 3.2, 24), 0.005, r_edge=0.0022, n=2, dome=0.0012, mat=M["trim_dark"])
        P.append(_face(key, xs - 0.0016, ysw + iy * 0.0195, zsw + iz * 0.0125))
    # --- courtesy lamp lens low on the door
    yl, zl = 0.10, 0.452
    xl = _wall_x(yl, zl)
    fr = S.rounded_frame(S.rrect(0.13, 0.034, 0.014, 3), S.rrect(0.116, 0.021, 0.009, 3), 0.006, r_out=0.0025, r_in=0.0015,
                         floor=0.0045, mat=M["chrome"])
    P.append(_face(fr, xl + 0.004, yl, zl))
    lens = S.rounded_panel(S.rrect(0.112, 0.017, 0.007, 3), 0.0022, r_edge=0.001, n=2, dome=0.0006, mat=M["plate"])
    P.append(_face(lens, xl - 0.0007, yl, zl))
    # --- speaker grille
    ysp, zsp = 0.30, 0.452
    xsp = _wall_x(ysp, zsp)
    prof = [(0.048, 0.0), (0.048, 0.005), (0.0455, 0.0062), (0.043, 0.0062), (0.041, 0.0022)]
    for rr in (0.032, 0.0215, 0.011):
        prof += [(rr + 0.0028, 0.0022), (rr + 0.0028, 0.0055), (rr - 0.0028, 0.0055), (rr - 0.0028, 0.0022)]
    prof += [(0.0075, 0.0022), (0.0, 0.0085)]
    sp = S.lathe(prof, seg=28, mat=M["trim_dark"], closed_ends=True)
    sp.assign(M["underbody"], lambda c, n: (c[:, 2] < 0.0028) & (n[:, 2] > 0.9))
    P.append(sp.apply(S.rotation_between((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)), (xsp + 0.0008, ysp, zsp)))
    return P, D


def lock_pin():
    """The door lock pin, standing on the door top at the sill."""
    pin = S.lathe([(0.0, 0.0), (0.0105, 0.0), (0.0105, 0.0022), (0.0034, 0.0034), (0.0034, 0.030), (0.0072, 0.0325), (0.0074, 0.040),
                   (0.0052, 0.0435), (0.0, 0.0435)], seg=14, mat=M["chrome"])
    return pin.moved((L.SILL_X, 0.40, L.Z_BELT - 0.001))


# ------------------------------------------------------------------------------------------------------ console parts
LEVER = np.array([(0.0, -0.02, CZ - 0.005), (0.0, -0.018, 0.615), (0.0, -0.010, 0.660), (0.0, 0.002, 0.700), (0.0, 0.010, 0.722)])
LEVER_DIR = (LEVER[-1] - LEVER[-2]) / np.linalg.norm(LEVER[-1] - LEVER[-2])
KNOB_C = LEVER[-1] + LEVER_DIR * 0.020                          # the widest ring of the knob
BRAKE_BASE = np.array([0.0, 0.235, CZ + 0.006])
BRAKE_DIR = np.array([0.0, math.sin(math.radians(20.0)), math.cos(math.radians(20.0))])        # leaning back 20 degrees
BRAKE_GRIP_C = BRAKE_BASE + BRAKE_DIR * 0.085


def _flat(mesh, x, y, z):
    """A panel / frame built with its back at z_l = 0 facing +z_l, laid on a horizontal surface: (x_l, y_l, z_l) -> (x, y, z)."""
    return mesh.moved((x, y, z))


def console_parts():
    P, D = [], []
    zt = CZ
    # the shifter gate: a chrome bezel round a dark pocket (the console top), a lever slot and the PRNDL window
    yg = -0.075
    P.append(_flat(S.rounded_frame(S.rrect(0.112, 0.222, 0.026, 4), S.rrect(0.094, 0.204, 0.017, 4), 0.008, r_out=0.0035,
                                   r_in=0.0025, mat=M["chrome"]), 0.0, yg, zt - 0.001))
    P.append(_flat(S.rounded_frame(S.rrect(0.026, 0.132, 0.013, 3), S.rrect(0.013, 0.119, 0.0065, 3), 0.004, r_out=0.0015,
                                   r_in=0.001, floor=0.0032, mat=M["underbody"]), 0.0, -0.03, zt - 0.0005))
    P.append(_flat(S.rounded_panel(S.rrect(0.076, 0.022, 0.006, 3), 0.003, r_edge=0.001, n=2, mat=M["display_glass"]), 0.0, -0.155, zt))
    for i in range(-2, 3):
        D.append(_flat(S.rounded_panel(S.rrect(0.0085, 0.0085, 0.002, 2), 0.0012, r_edge=0.0005, n=1, mat=M["vfd_lit"]),
                       i * 0.0135, -0.155, zt + 0.0031))
    # the lever and its knob
    P.append(S.tube(LEVER, 0.0065, sides=12, mat=M["chrome"], up=(1.0, 0.0, 0.0)))
    bulb = S.lathe([(0.0, -0.024), (0.011, -0.022), (0.019, -0.012), (0.0225, 0.0), (0.0205, 0.013), (0.0135, 0.0225), (0.0, 0.0258)],
                   seg=24, mat=M["trim_dark"])
    ring = S.lathe([(0.0, -0.0275), (0.0155, -0.0275), (0.0175, -0.0255), (0.0175, -0.0215), (0.0165, -0.0205), (0.0, -0.0205)], seg=24,
                   mat=M["chrome"])
    P += [_on_axis(bulb, LEVER_DIR, KNOB_C), _on_axis(ring, LEVER_DIR, KNOB_C)]
    # the wood panel behind the gate, with the ash tray; the parking brake on its rear edge
    P.append(_flat(S.rounded_panel(S.rrect(0.205, 0.234, 0.022, 4), 0.004, r_edge=0.002, n=2, uv=True, mat=M["wood"]), 0.0, 0.172, zt - 0.0008))
    P.append(_flat(S.rounded_frame(S.rrect(0.13, 0.085, 0.016, 4), S.rrect(0.112, 0.067, 0.010, 4), 0.006, r_out=0.0025, r_in=0.0015,
                                   floor=0.0045, mat=M["chrome"]), 0.0, 0.105, zt + 0.003))
    P.append(_flat(S.rounded_panel(S.superellipse(0.066, 0.052, 3.0, 28), 0.012, r_edge=0.0045, n=2, dome=0.003, mat=M["trim_dark"]),
                   0.0, 0.235, zt + 0.003))
    P.append(S.tube(np.array([BRAKE_BASE, BRAKE_BASE + BRAKE_DIR * 0.06]), 0.006, sides=12, mat=M["chrome"], up=(1.0, 0.0, 0.0)))
    grip = S.lathe([(0.0, 0.0), (0.0115, 0.0), (0.0135, 0.006), (0.0140, 0.012), (0.0140, 0.060), (0.0120, 0.069), (0.0, 0.0715)], seg=20,
                   mat=M["rubber"])
    P.append(_on_axis(grip, BRAKE_DIR, BRAKE_BASE + BRAKE_DIR * 0.05))
    return P, D


def rounded_slab(outline, t, r_top, r_bot, n=2, mat=0):
    """A thin plate whose top and bottom edges are both rolled (a mat, a trim): the outline (m, 2) pulled up to t."""
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)
    prof = [(r_bot * (1.0 - math.sin(u)), r_bot * (1.0 - math.cos(u))) for u in a]
    prof += [(0.0, t - r_top)] if t - r_top > r_bot + 1e-6 else []
    prof += [(r_top * (1.0 - math.cos(u)), t - r_top * (1.0 - math.sin(u))) for u in a[1:]]
    R = S.offset_rings(outline, prof)
    return S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_rings=1, mat=mat)


def mats_and_trim():
    """Floor mats (4 mm, rolled edges) on the carpet, with the driver's rubber heel pad."""
    P = []
    plan = S.rrect(0.36, 0.44, 0.05, 4)
    for s in (-1, 1):
        P.append(rounded_slab(plan, 0.0045, 0.002, 0.0015, mat=M["trim_dark"]).moved((s * 0.45, -0.33, L.FLOOR_Z - 0.0025)))
    P.append(rounded_slab(S.rrect(0.11, 0.13, 0.03, 3), 0.0035, 0.0017, 0.001, mat=M["rubber"]).moved((0.37, -0.44, L.FLOOR_Z + 0.0015)))
    return P


# ------------------------------------------------------------------------------------------------------ assembly
def static_shells():
    """The cages. The seats are subdivided one level more than the rest (the builder bakes every shell one level above `levels`):
    their pleats and bolsters fill the frame in the close-ups; the dense rings of the other cages are already round at level 2."""
    seat = front_seat_cage().moved((SX, 0.0, 0.0))
    door = S.merge([door_cage(), door_insert_cage()])
    qtr = S.merge([quarter_cage(), quarter_insert_cage()])
    return {
        "seat_L": S.smooth(seat, levels=2),
        "seat_R": S.smooth(seat.mirrored_x(), levels=2),
        "bench": S.smooth(S.merge([bench_cushion_cage(), bench_back_cage(), bench_base_cage()]), levels=1),
        "console": S.smooth(S.merge([console_cage(), lid_cage()]), levels=1),
        "floor": S.smooth(carpet_cage(), levels=1),
        "door_L": S.smooth(door, levels=1),
        "door_R": S.smooth(door.mirrored_x(), levels=1),
        "quarter_L": S.smooth(qtr, levels=1),
        "quarter_R": S.smooth(qtr.mirrored_x(), levels=1),
    }


def static_parts():
    interior, decals = [], []
    # --- the driver's seat (the passenger's is its mirror image)
    left = [p.moved((SX, 0.0, 0.0)) for p in seat_parts()]
    for cage in (cushion_cage(), back_cage()):
        ns, m = ring_dims(cage)
        c = cage.moved((SX, 0.0, 0.0))
        for side in (+1, -1):
            left.append(piping(c, ns, m, 6, 7, side, 0.0035, M["vinyl"]))
    interior += left + [p.mirrored_x() for p in left]
    # --- the rear bench
    for cage in (bench_cushion_cage(), bench_back_cage()):
        ns, m = ring_dims(cage)
        for side in (+1, -1):
            interior.append(piping(cage, ns, m, 6, 7, side, 0.0032, M["vinyl"]))
    # --- the console
    P, D = console_parts()
    interior += P
    decals += D
    # --- the doors
    P, D = door_parts()
    P.append(lock_pin())
    interior += P + [p.mirrored_x() for p in P]
    decals += D + [p.mirrored_x() for p in D]
    interior += mats_and_trim()
    well = wheel_well()
    interior += [well, well.mirrored_x()]
    return {GROUP: S.merge(interior), "decals": S.merge(decals)}


SEAT = {
    "cushion_top": CZT,                          # the seat surface under the hip centre (the dish's lowest crown)
    "back_base_yz": (BACK_Y0, CZT),              # where the back's front plane (the crowns of the pleats) meets the cushion top
    "back_axis_yz": (SIN_R, COS_R),              # unit vector up the back (y, z)
    "back_front_y_at_hip": BACK_Y0 + (L.HIP_Z - CZT) * math.tan(math.radians(REC)),
    "back_gap_at_hip": BACK_GAP,                 # how far behind the hip joint that plane passes at hip height
    "headrest_center": tuple(float(v) for v in (np.array([0.0, BACK_Y0, CZT]) + HEAD_V * BACK_AXIS - HEAD_W * BACK_NORM)[1:]),
    "headrest_size": HEAD_SIZE,
}


def _t(v):
    return tuple(float(c) for c in v)


_ARM_X = XW - (0.031 + _ARM_AMOUNT)
_PULL_X = XW - 0.0605
ANCHORS = {                                      # {name: {"point": ..., "dir": ...}} (L = +X, the driver's side)
    "shifter": {"point": _t(KNOB_C), "dir": _t(LEVER_DIR)},                          # centre of the knob, the lever's direction
    "handbrake": {"point": _t(BRAKE_GRIP_C), "dir": _t(BRAKE_DIR)},                  # centre of the grip
    "console_lid": {"point": (0.0, 0.5 * sum(LID_Y), LID_TOP + 0.004)},
    "armrest_L": {"point": (_ARM_X, 0.10, 0.583), "dir": (0.0, 1.0, 0.0)},
    "armrest_R": {"point": (-_ARM_X, 0.10, 0.583), "dir": (0.0, 1.0, 0.0)},
    "door_pull_L": {"point": (_PULL_X, 0.04, 0.722), "dir": (0.0, 1.0, 0.0)},
    "door_pull_R": {"point": (-_PULL_X, 0.04, 0.722), "dir": (0.0, 1.0, 0.0)},
    "lock_pin_L": {"point": (L.SILL_X, 0.40, L.Z_BELT + 0.0435)},
    "lock_pin_R": {"point": (-L.SILL_X, 0.40, L.Z_BELT + 0.0435)},
}

LOOKS = {"shifter": _t(KNOB_C)}
