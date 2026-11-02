"""The van's two rear swing doors (pure numpy, no bpy): composite leaves whose outer faces inside BACK_PANEL carry
mural_back, four strap hinges per door on the corner posts, two lock rods per door (guides, cams in keepers on the header
and the sill, a handle low on each rod with its retainer) and the rubber seals round each leaf.

+X is the van's left: a viewer behind the van sees the +X leaf on the left. Every piece of a door is built for the +X
leaf and mirrored for the other."""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L
from . import semi_truck_trailer_box as BOX
from . import semi_truck_trailer_kit as K

M = L.M
Y1 = L.TRAILER_Y[1]
DOOR_T = 0.040                    # leaf thickness: its outer face at Y1, its inner face DOOR_T inside
GAP_C, GAP_E = 0.006, 0.005       # half the gap between the leaves, the gap to the jambs, header and sill
LEAF_R = 0.012                    # the leaves' rolled edges
LEAF_X = (GAP_C, BOX.POST[0] - GAP_E)
LEAF_Z = (BOX.OPENING_Z[0] + GAP_E, BOX.OPENING_Z[1] - GAP_E)
HINGE_Z = tuple(float(z) for z in np.linspace(LEAF_Z[0] + 0.27, LEAF_Z[1] - 0.24, 4))
HINGE_AXIS = (BOX.POST[0] + 0.008, Y1 + 0.013)   # (x, y) of the +X door's hinge pins
ROD_X = (0.21, 0.70)              # the +X leaf's lock rods
ROD_R, ROD_Y = 0.0135, Y1 + 0.034  # 1-1/16 in rods, their axis off the door face
ROD_Z = (1.292, 3.993)            # the rods' ends: the cams sit in keepers on the sill and on the header band
GUIDE_Z = (1.95, 2.75, 3.50)
HANDLE_Z, HANDLE_LEN = 1.47, 0.31  # the handles lie low on the door, pointing to the hinge side


def _mx(m, sx):
    return m if sx > 0 else m.mirrored_x()


def leaf(sx):
    """A door leaf: a slab with rolled edges, its outer face cut at BACK_PANEL; the faces inside it in mural_back."""
    hw, z0, z1 = L.BACK_PANEL
    m = K.slab((LEAF_X[0], Y1 - DOOR_T, LEAF_Z[0]), (LEAF_X[1], Y1, LEAF_Z[1]), LEAF_R, lines=((hw,), (), (z0, z1)),
               mat=M["trailer_skin"])
    K.assign_face(m, M["mural_back"], 1, 1.0, Y1, ((0.0, hw), (z0, z1)))
    return _mx(m, sx)


def _plate(outline, t, origin, normal, up=(0.0, 0.0, 1.0), mat=None, r_edge=0.002):
    m = S.rounded_panel(outline, t, r_edge=r_edge, n=2, rings=1, mat=M["trailer_metal"] if mat is None else mat)
    return S.place(m, origin, normal, up)


def hinge(z):
    """One strap hinge of the +X door at height z: the butt on the post, the knuckle round the pin, the tapered strap
    lying on the door, bolt heads on both leaves."""
    xh, yh = HINGE_AXIS
    out = []
    back = (0.0, 1.0, 0.0)
    butt = S.rrect(0.090, 0.074, 0.010, 2)
    out.append(_plate(butt, 0.008, (xh + 0.052, Y1 - 0.0008, z), back))
    # S.place on a face looking +Y maps the outline's +x to the world's -X: the strap runs from the pin inward
    strap = np.array([[0.0, -0.033], [0.30, -0.022], [0.31, 0.0], [0.30, 0.022], [0.0, 0.033]])
    strap = S.fillet(strap, [0.004, 0.012, 0.012, 0.012, 0.004], n=2, closed=True)
    out.append(_plate(strap, 0.008, (xh - 0.002, Y1 - 0.0008, z), back))
    path = np.array([[xh, yh, z - 0.040], [xh, yh, z + 0.040]])
    out.append(K.tube(path, 0.014, sides=10, ends=0.004, mat=M["trailer_metal"]))
    pin = np.array([[xh, yh, z - 0.047], [xh, yh, z + 0.047]])
    out.append(K.tube(pin, 0.0065, sides=6, ends=0.002, mat=M["trailer_metal"]))
    bolts = [(xh + 0.030, z - 0.020), (xh + 0.030, z + 0.020), (xh + 0.075, z - 0.020), (xh + 0.075, z + 0.020),
             (xh - 0.06, z), (xh - 0.14, z), (xh - 0.23, z)]
    P = np.array([[x, Y1 + 0.0072, zz] for x, zz in bolts])
    out.append(K.rivets(P, back, (0.0, 0.0, 1.0), r=0.0065, h=0.004, mat=M["trailer_metal"]))
    return out


def _staple(cx, cy, z, r_in, reach, height, thick, foot):
    """A bent strap: from a foot flat on the door (y = Y1) out round a rod centred (cx, cy) and back: the rods' guides
    and the handles' retainer. Its section is `height` tall (z) and `thick` thick."""
    a = np.linspace(math.pi, 0.0, 6)
    rr = r_in + 0.5 * thick
    arc = np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1)
    pts = np.concatenate([[[cx - rr - foot, Y1 + 0.5 * thick], [cx - rr, Y1 + 0.5 * thick]], arc[1:-1] if reach else arc,
                          [[cx + rr, Y1 + 0.5 * thick], [cx + rr + foot, Y1 + 0.5 * thick]]])
    pts = S.fillet(pts, [0.0, 0.006, *([0.0] * (len(pts) - 4)), 0.006, 0.0], n=1)
    path = np.column_stack([pts, np.full(len(pts), z)])
    return K.bar(path, S.rrect(thick, height, 0.4 * thick, 1), up=(0.0, 0.0, 1.0), mat=M["trailer_metal"])


def _cam(x, z, up):
    """The cam on a rod's end (a teardrop plate turned into the keeper) and the keeper: a back plate on the frame with a
    hood the cam's nose hooks behind. `up` +1 at the top end, -1 at the bottom."""
    out = []
    t = 0.014
    zc = z - up * 0.010
    blob = np.concatenate([K.arc_pts((0.0, 0.0), 0.024, 90.0, 270.0, 6), K.arc_pts((0.050, -0.012), 0.011, 270.0, 450.0, 4)])
    cam = S.rounded_panel(blob, t, r_edge=0.003, n=1, rings=1, mat=M["trailer_metal"])
    out.append(cam.moved((x, ROD_Y, zc - 0.5 * t)))
    kz = zc
    plate = S.rrect(0.11, 0.050, 0.010, 2)
    out.append(_plate(plate, 0.006, (x + 0.035, Y1 - 0.0008, kz), (0.0, 1.0, 0.0)))
    out.append(_staple(x + 0.048, ROD_Y - 0.010, kz, 0.016, True, 0.032, 0.006, 0.012))
    P = np.array([[x - 0.008, Y1 + 0.0055, kz], [x + 0.078, Y1 + 0.0055, kz]])
    out.append(K.rivets(P, (0.0, 1.0, 0.0), r=0.006, h=0.0035, mat=M["trailer_metal"]))
    return out


def lock_rod(x):
    """A lock rod of the +X door at x: the rod, three guides, the cams and keepers at both ends, the handle on its hub
    pointing to the hinge side and the retainer that holds it."""
    out = [K.tube(np.array([[x, ROD_Y, ROD_Z[0]], [x, ROD_Y, ROD_Z[1]]]), ROD_R, sides=10, ends=0.004,
                  mat=M["trailer_metal"])]
    for z in GUIDE_Z:
        out.append(_staple(x, ROD_Y, z, ROD_R + 0.001, True, 0.030, 0.005, 0.022))
    out += _cam(x, ROD_Z[1], +1) + _cam(x, ROD_Z[0], -1)
    hub = S.lathe([(0.0, -0.034), (0.019, -0.034), (0.023, -0.028), (0.023, 0.028), (0.019, 0.034), (0.0, 0.034)], seg=10,
                  mat=M["trailer_metal"])
    out.append(hub.moved((x, ROD_Y, HANDLE_Z)))
    # the handle: a flat bar from the hub to the hinge side, cranked in toward the door, a rounded grip at its end
    L_ = HANDLE_LEN
    path = np.array([[x + 0.018, ROD_Y, HANDLE_Z], [x + 0.10, ROD_Y, HANDLE_Z], [x + 0.17, ROD_Y - 0.010, HANDLE_Z],
                     [x + L_ - 0.06, ROD_Y - 0.012, HANDLE_Z], [x + L_, ROD_Y - 0.012, HANDLE_Z]])
    out.append(K.bar(path, S.rrect(0.010, 0.030, 0.004, 1), up=(0.0, 0.0, 1.0), ends=0.004, mat=M["trailer_metal"]))
    grip = np.array([[x + L_ - 0.07, ROD_Y - 0.012, HANDLE_Z], [x + L_ + 0.012, ROD_Y - 0.012, HANDLE_Z]])
    out.append(K.tube(grip, 0.013, sides=8, ends=0.008, aspect=1.15, mat=M["trailer_metal"]))
    out.append(_retainer(x + L_ - 0.10))
    return out


def _retainer(x):
    """The catch under a handle's end: a strap bent out from the door and up in front of the bar, so a closed handle
    rests in it."""
    zb = HANDLE_Z - 0.026
    pts = S.fillet(np.array([[Y1 + 0.003, zb], [ROD_Y + 0.003, zb], [ROD_Y + 0.004, HANDLE_Z - 0.004]]), [0.0, 0.006, 0.0],
                   n=3)
    path = np.column_stack([np.full(len(pts), x), pts])
    return K.bar(path, S.rrect(0.005, 0.036, 0.002, 2), up=(1.0, 0.0, 0.0), mat=M["trailer_metal"])


def seal(sx):
    """The rubber seal round a leaf, in the gap between it and the jambs, the header, the sill and the other leaf."""
    x0, x1 = LEAF_X[0] - 0.0035, LEAF_X[1] + 0.0035
    z0, z1 = LEAF_Z[0] - 0.0035, LEAF_Z[1] + 0.0035
    loop = S.fillet(np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]]), 0.02, n=3, closed=True)
    path = np.column_stack([loop[:, 0], np.full(len(loop), Y1 - 0.009), loop[:, 1]])
    t = 2.0 * math.pi * np.arange(6) / 6
    sec = np.stack([0.0058 * np.cos(t), 0.0058 * np.sin(t)], 1)
    rings = K.sweep_rings(np.vstack([path, path[:1]]), sec, up=(0.0, 1.0, 0.0))[:-1]
    m = K.ring_skin(rings, loop=True, mat=M["rubber"])
    return _mx(m, sx)


def doors():
    """All of both doors as one list of meshes (the leaves, the hardware, the seals)."""
    out = []
    for sx in (1.0, -1.0):
        parts = []
        for z in HINGE_Z:
            parts += hinge(z)
        for x in ROD_X:
            parts += lock_rod(x)
        out += [leaf(sx), seal(sx)] + [_mx(p, sx) for p in parts]
    return out


def handle_points():
    """{anchor name: grip point} of the door handles (the anchors a hand can take): `trailer_handle_<side><k>`."""
    out = {}
    for side, sx in (("L", 1.0), ("R", -1.0)):
        for k, x in enumerate(ROD_X):
            out[f"trailer_handle_{side}{k + 1}"] = (sx * (x + HANDLE_LEN - 0.03), ROD_Y - 0.012, HANDLE_Z)
    return out
