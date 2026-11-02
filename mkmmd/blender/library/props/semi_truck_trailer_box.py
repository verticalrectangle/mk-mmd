"""The van's body (pure numpy, no bpy): composite plate walls with riveted joint strips, the crowned aluminium roof, the
extruded top and bottom rails, the rounded front corner caps and the rear frame (corner posts, header, sill).

The frame members are sections pulled along the van (semi_truck_trailer_kit): the top rail runs round the whole van, its
front part the nose cap (a quarter round rolling the front wall into the roof), its rear part the top of the header; the
bottom rail runs round the front as the front sill; the corner caps stand between them on the same plan corner. Every
member stands T out of the plates it laps, so the plates' outer faces lie exactly at x = +-TRAILER_HALF_W (the sides)
and the front at TRAILER_Y[0] is the members' face. The sides' outer faces inside SIDE_PANEL carry mural_left /
mural_right, the doors' (semi_truck_trailer_doors) mural_back."""
import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L
from . import semi_truck_trailer_kit as K

M = L.M
Y0, Y1 = L.TRAILER_Y
HW = L.TRAILER_HALF_W
ZT = L.TRAILER_TOP_Z
T = 0.004                         # how far a frame member stands out of the plate it laps
XF = HW + T                       # the frame's outer face on the sides
WALL = 0.0127                     # composite plate (1/2 in)
PLATE_R = 0.006                   # the plates' rolled edges
PLATE_Z = (1.30, 4.083)           # the plates' bottom and top edges, hidden in the rails' lips
RP = 0.085                        # plan radius of the front corners (the caps' outer face)
RREAR = 0.02                      # plan radius of the rear corners (posts, the top rail)
RTOP = 0.055                      # the top rail's round (the nose cap at the front)
CAP_LIP = (0.020, 0.015)          # the corner caps' laps on the front wall and on the side
POST = (1.165, 0.13)              # the rear posts: inner face x (the door opening), depth along the side
OPENING_Z = (1.340, 3.955)        # the door opening: top of the sill, bottom of the header band
ROOF_Z, CROWN = 4.0975, 0.011     # the roof sheet's top at its edges, and its crown on the centre line
ROOF_IN = 0.062                   # the roof sheet's edge lies this far inside the frame's face, under the top rail's lip
JOINTS = 12                       # vertical joint strips on each side (13 plates of about 4 ft)
JOINT_W, JOINT_H = 0.044, 0.0025  # a joint strip's width and how far it stands out of the plates
FRONT_JOINTS = (-0.40, 0.40)      # x of the front wall's joint strips
RIVET_PITCH = 0.15


def _pts(*parts):
    """A polyline (m, 2) from points and arrays of points, consecutive duplicates dropped."""
    P = np.concatenate([np.atleast_2d(np.asarray(p, float)) for p in parts])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9])
    P = P[keep]
    if np.linalg.norm(P[0] - P[-1]) < 1e-9:
        P = P[:-1]
    return P


# ===================================================================================================================
# sections
# ===================================================================================================================
def top_rail_section():
    """(s, z): s the depth inward from the frame's face. A lip laps the plate (its edge on the plate at the rail's foot),
    the face rolls over the quarter round RTOP onto the roof, a lip holds the roof sheet's edge; behind, the extrusion
    curls over the plate's top edge."""
    z0 = L.TRAILER_TOP_RAIL_Z[0]
    return _pts((T, z0), (0.0016, z0 + 0.0016), (0.0, z0 + 0.006), (0.0, ZT - RTOP),
                K.arc_pts((RTOP, ZT - RTOP), RTOP, 180.0, 90.0, 9)[1:-1], (RTOP, ZT), (0.077, ZT),
                K.arc_pts((0.077, ZT - 0.003), 0.003, 90.0, 0.0, 2)[1:], (0.080, ROOF_Z + 0.0012),
                (0.062, 4.091), (0.040, 4.0865), (0.018, PLATE_Z[1] + 0.001), (T, PLATE_Z[1] + 0.001))


def bottom_rail_section():
    """(s, z): the bottom rail (and the front sill): a tall face T proud of the plate, its top lip lapping the plate, a
    rounded foot, the bottom flange rising inward (the crossmembers sit on it), the inner side curving up under the
    plate."""
    z0, z1 = L.TRAILER_RAIL_Z
    return _pts((T, z1), (0.0016, z1 - 0.0016), (0.0, z1 - 0.006), (0.0, z0 + 0.012),
                K.arc_pts((0.012, z0 + 0.012), 0.012, 180.0, 270.0, 4)[1:], (0.070, z0 + 0.016),
                K.arc_pts((0.072, z0 + 0.024), 0.008, 285.0, 400.0, 4)[1:], (0.066, 1.262), (0.050, 1.284),
                (0.030, 1.296), (0.017, PLATE_Z[0]), (T, PLATE_Z[0]))


def header_section():
    """(y, z) of the rear header band between the posts: its rear face flush with the doors, a stop lip behind the
    doors' top edge; its peaked top runs up into the top rail's rear part."""
    z0 = OPENING_Z[1]
    return K.rounded([(Y1, 4.050), (Y1, z0), (Y1 - 0.046, z0), (Y1 - 0.048, z0 - 0.020), (Y1 - 0.075, z0 - 0.020),
                      (Y1 - 0.112, z0 + 0.030), (Y1 - 0.095, 4.040), (Y1 - 0.045, 4.075)],
                     [0.0, 0.006, 0.003, 0.003, 0.006, 0.02, 0.008, 0.006], n=3)


def sill_section():
    """(y, z) of the rear sill under the doors: the rear face that carries the tail lamps and the tape, its top level
    with the floor, the underside rising forward to the last crossmember."""
    z0, z1 = L.TRAILER_RAIL_Z[0], OPENING_Z[0]
    return K.rounded([(Y1, z1), (Y1, z0), (Y1 - 0.04, z0), (Y1 - 0.20, 1.245), (Y1 - 0.25, 1.262),
                      (Y1 - 0.05, 1.3055), (Y1 - 0.042, L.TRAILER_FLOOR_Z), (Y1 - 0.040, z1)],
                     [0.004, 0.012, 0.02, 0.03, 0.01, 0.004, 0.002, 0.002], n=3)


def corner_cap_outline(sx=1.0):
    """Plan outline (x, y) of a front corner cap (the +X one, mirrored by sx): the quarter round RP lapping the front
    wall by CAP_LIP[0] and the side by CAP_LIP[1], a sheet T thick."""
    cx, cy = XF - RP, Y0 + RP
    fl, sl = CAP_LIP
    out = _pts((cx - fl, Y0 + T), (cx - fl + 0.0016, Y0 + 0.0016), (cx - fl + 0.006, Y0),
               K.arc_pts((cx, cy), RP, 270.0, 360.0, 10), (XF, cy + sl - 0.006), (XF - 0.0016, cy + sl - 0.0016),
               (HW, cy + sl), (HW, cy), K.arc_pts((cx, cy), RP - T, 0.0, -90.0, 10)[1:-1], (cx, Y0 + T))
    return out * [sx, 1.0]


def rear_post_outline(sx=1.0):
    """Plan outline (x, y) of a rear corner post (the +X one, mirrored by sx): the side flange lapping the side plate,
    the rounded rear corner, the face the hinges sit on, the jamb the door closes in and the stop behind it."""
    xi, d = POST
    pts = _pts((HW, Y1 - d), (XF - 0.0016, Y1 - d + 0.0016), (XF, Y1 - d + 0.006), (XF, Y1 - RREAR),
               K.arc_pts((XF - RREAR, Y1 - RREAR), RREAR, 0.0, 90.0, 5)[1:-1], (XF - RREAR, Y1),
               K.arc_pts((xi + 0.008, Y1 - 0.008), 0.008, 90.0, 180.0, 3)[1:], (xi, Y1 - 0.044),
               K.arc_pts((xi - 0.004, Y1 - 0.044), 0.004, 0.0, -90.0, 2)[1:], (xi - 0.040, Y1 - 0.048),
               K.arc_pts((xi - 0.040, Y1 - 0.056), 0.008, 90.0, 180.0, 3)[1:], (xi - 0.048, Y1 - 0.075),
               (xi - 0.030, Y1 - 0.100), (1.215, Y1 - d + 0.006), (HW - WALL, Y1 - d))
    return pts * [sx, 1.0]


def joint_outline(c, axis_out):
    """Section (along, out) of a joint strip centred at `c` along the wall, `axis_out` the wall face's coordinate: a
    shallow crowned strip whose edges dip just under the plate's face."""
    w, h = JOINT_W, JOINT_H
    u = np.array([-0.5, -0.47, -0.36, -0.18, 0.0, 0.18, 0.36, 0.47, 0.5]) * w
    v = np.array([-0.0005, 0.0010, 0.0019, 0.0024, h, 0.0024, 0.0019, 0.0010, -0.0005])
    top = np.stack([c + u, axis_out + v], 1)
    back = np.array([[c + 0.5 * w, axis_out - 0.0012], [c - 0.5 * w, axis_out - 0.0012]])
    return np.concatenate([top, back])


# ===================================================================================================================
# parts
# ===================================================================================================================
def side_plate(sx):
    """The composite wall of one side (sx +1: +X): one plate whose outer face is cut at SIDE_PANEL, the faces inside it
    in the mural role."""
    y0, y1, z0, z1 = L.SIDE_PANEL
    xo = sx * HW
    lo, hi = (min(xo, xo - sx * WALL), Y0 + RP + 0.004, PLATE_Z[0]), (max(xo, xo - sx * WALL), Y1 - 0.08, PLATE_Z[1])
    m = K.slab(lo, hi, PLATE_R, lines=((), (y0, y1), (z0, z1)), mat=M["trailer_skin"])
    return K.assign_face(m, M["mural_left" if sx > 0 else "mural_right"], 0, sx, xo, ((y0, y1), (z0, z1)))


def front_plate():
    half = XF - RP - CAP_LIP[0] + 0.010
    return K.slab((-half, Y0 + T, PLATE_Z[0]), (half, Y0 + T + WALL, PLATE_Z[1]), PLATE_R, mat=M["trailer_skin"])


def roof():
    """The aluminium roof sheet: a 1.5 mm slab, crowned CROWN over its width, its edges under the top rail's lips."""
    hx = XF - ROOF_IN
    xs = np.linspace(-hx, hx, 13)[1:-1]
    m = K.slab((-hx, Y0 + ROOF_IN, ROOF_Z - 0.0015), (hx, Y1 - ROOF_IN, ROOF_Z), 0.0007, lines=(xs, (), ()),
               mat=M["trailer_metal"])
    V = m.V.copy()
    V[:, 2] += CROWN * (1.0 - np.clip(np.abs(V[:, 0]) / hx, 0.0, 1.0) ** 2)
    m.V = V
    return m


def top_rail():
    corners = [(XF, Y1), (XF, Y0), (-XF, Y0), (-XF, Y1)]
    return K.plan_sweep(corners, [RREAR, RP, RP, RREAR], top_rail_section(), closed=True, n=8, mat=M["trailer_metal"])


def bottom_rail():
    end = Y1 - 0.05                                     # the ends stop inside the rear posts
    corners = [(XF, end), (XF, Y0), (-XF, Y0), (-XF, end)]
    return K.plan_sweep(corners, [0.0, RP, RP, 0.0], bottom_rail_section(), closed=False, n=8, mat=M["trailer_metal"])


def corner_caps():
    z0, z1 = L.TRAILER_RAIL_Z[1] - 0.006, L.TRAILER_TOP_RAIL_Z[0] + 0.006
    return [K.prism(corner_cap_outline(sx), "z", z0, z1, mat=M["trailer_metal"]) for sx in (1.0, -1.0)]


def rear_frame():
    """The two corner posts (from the sill's foot to inside the top rail), the header band and the sill between them."""
    parts = [K.prism(rear_post_outline(sx), "z", L.TRAILER_RAIL_Z[0], L.TRAILER_TOP_RAIL_Z[0] + 0.006,
                     mat=M["trailer_metal"]) for sx in (1.0, -1.0)]
    xe = POST[0] + 0.012                                # the ends run into the posts
    parts.append(K.prism(header_section(), "x", -xe, xe, mat=M["trailer_metal"]))
    parts.append(K.prism(sill_section(), "x", -xe, xe, mat=M["trailer_metal"]))
    return parts


def joint_ys():
    """y of the side walls' joint strips: SIDE_PANEL split into JOINTS + 1 equal plates."""
    y0, y1 = L.SIDE_PANEL[:2]
    return y0 + (y1 - y0) * np.arange(1, JOINTS + 1) / (JOINTS + 1)


def joint_strips():
    z0, z1 = L.TRAILER_RAIL_Z[1] - 0.004, L.TRAILER_TOP_RAIL_Z[0] + 0.004
    out = []
    for sx in (1.0, -1.0):
        for y in joint_ys():
            sec = joint_outline(y, HW)                     # (y, x) on the +X wall
            sec = sec[:, ::-1] * [sx, 1.0]                 # (x, y)
            out.append(K.prism(sec, "z", z0, z1, mat=M["trailer_metal"]))
    for x in FRONT_JOINTS:                                 # the front wall: (x, y) with the face toward -Y
        sec = joint_outline(x, 0.0)
        sec = np.stack([sec[:, 0], Y0 + T - sec[:, 1]], 1)
        out.append(K.prism(sec, "z", z0, z1, mat=M["trailer_metal"]))
    return out


def body_rivets():
    """Rivet heads: two columns down every joint strip, a row through each rail's lip into the plates, a column on the
    corner caps' and the posts' side laps and on the front wall's joints."""
    mat = M["trailer_metal"]
    zs = np.arange(L.TRAILER_RAIL_Z[1] + 0.06, L.TRAILER_TOP_RAIL_Z[0] - 0.03, RIVET_PITCH)
    out = []
    lip_bot = L.TRAILER_RAIL_Z[1] - 0.010
    lip_top = L.TRAILER_TOP_RAIL_Z[0] + 0.010
    ys = np.arange(Y0 + RP + 0.10, Y1 - POST[1] - 0.02, RIVET_PITCH)
    for sx in (1.0, -1.0):
        n = (sx, 0.0, 0.0)
        for y in joint_ys():
            for dy in (-0.011, 0.011):
                P = np.stack([np.full_like(zs, sx * (HW + 0.0021)), np.full_like(zs, y + dy), zs], 1)
                out.append(K.rivets(P, n, (0.0, 0.0, 1.0), r=0.0042, h=0.0028, mat=mat))
        for z in (lip_bot, lip_top):                       # through the rails' lips
            P = np.stack([np.full_like(ys, sx * (XF - 0.0004)), ys, np.full_like(ys, z)], 1)
            out.append(K.rivets(P, n, (0.0, 0.0, 1.0), r=0.0045, h=0.003, mat=mat))
        for y in (Y0 + RP + 0.008, Y1 - POST[1] + 0.012):   # the caps' and the posts' laps on the side
            P = np.stack([np.full_like(zs, sx * (XF - 0.0004)), np.full_like(zs, y), zs], 1)
            out.append(K.rivets(P, n, (0.0, 0.0, 1.0), r=0.0045, h=0.003, mat=mat))
        # the cap's lap on the front wall
        P = np.stack([np.full_like(zs, sx * (XF - RP - 0.010)), np.full_like(zs, Y0 + 0.0004), zs], 1)
        out.append(K.rivets(P, (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), r=0.0045, h=0.003, mat=mat))
    for x in FRONT_JOINTS:
        for dx in (-0.011, 0.011):
            P = np.stack([np.full_like(zs, x + dx), np.full_like(zs, Y0 + T - 0.0021), zs], 1)
            out.append(K.rivets(P, (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), r=0.0042, h=0.0028, mat=mat))
    return out


def walls():
    """{"box": the plates and the roof, "frame": rails, caps, the rear frame, joint strips and rivets}."""
    box = [side_plate(1.0), side_plate(-1.0), front_plate(), roof()]
    frame = [top_rail(), bottom_rail()] + corner_caps() + rear_frame() + joint_strips() + body_rivets()
    return {"box": S.merge(box), "frame": S.merge(frame)}
