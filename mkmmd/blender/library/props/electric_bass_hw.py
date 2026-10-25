"""The bass's hardware (pure numpy, no bpy): the split pickup's two covers, the string-through bridge with its four saddles, the
two knobs and the jack on the control plate, the plug, strap buttons, the back's neck plate and string ferrules, the plectrum
and the cable's path. Everything stands on the pickguard (y = -GUARD_T), the control plate (y = -PLATE_T) or the body's face
(y = 0) at the positions of `electric_bass_layout`, modelled from lathes, sweeps and rolled panels.
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_bass_body as B
from . import electric_bass_layout as L
from . import electric_guitar_geo as G

M = L.MATS
OUT = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)
Y_G = -L.GUARD_T                                   # the pickguard's top
Y_P = -L.PLATE_T                                   # the control plate's top


def lathe(pts, r=0.0005, seg=24, mat=0, n=3):
    return S.lathe(S.fillet(np.array(pts, float), r, n), seg=seg, mat=mat)


def put(mesh, x, y, z, normal=OUT, up=UP):
    return S.place(mesh, (x, y, z), normal, up)


def dome_screw(r=0.0026, h=0.0016, mat=0):
    return lathe([(0.0, 0.0), (r, 0.0), (r, 0.45 * h), (0.75 * r, 0.85 * h), (0.0, h)], 0.0004, 16, mat, 2)


# ===================================================================================================================
# the split pickup
# ===================================================================================================================
def pickups_mesh():
    """The split single coil: two rolled stadium covers of black plastic (role `knobs`), each with a height screw beyond either
    end (role `hardware`)."""
    cover = S.rounded_panel(S.stadium(L.PICKUP_L, L.PICKUP_W, 8), L.PICKUP_H, r_edge=0.0026, n=3, dome=0.0006, mat=M["knobs"])
    screw = dome_screw(0.0024, 0.0015, M["hardware"])
    out = []
    for spec in L.PICKUPS.values():
        cx, cz = spec["center"]
        out.append(put(cover, cx, Y_G + 0.0001, cz))
        for sgn in (-1.0, 1.0):
            out.append(put(screw, cx + sgn * L.PICKUP_SCREW, Y_G + 0.0001, cz))
    return S.merge(out)


# ===================================================================================================================
# the bridge
# ===================================================================================================================
def bridge_outline():
    """The plate's outline (m, 2) in its own x-y frame (x across, y = the prop's z): a trapezoid, wide at the neck end, with
    large rounded corners there and small ones at the tail."""
    b = L.BRIDGE
    (x0, x1), (x2, x3) = b["top_x"], b["bottom_x"]
    P = np.array([[x2, b["bottom"]], [x3, b["bottom"]], [x1, b["top"]], [x0, b["top"]]])
    return G.ccw(S.fillet(P, [b["r_bottom"], b["r_bottom"], b["r_top"], b["r_top"]], n=6, closed=True))


def bridge_mesh():
    """The string-through bridge (hardware): the rolled plate, its rolled tail across the end where the strings go through,
    four saddles on height screws with their intonation screws running forward to the plate's flange, and four mounting
    screws."""
    b, hw = L.BRIDGE, M["hardware"]
    plate = S.rounded_panel(bridge_outline(), b["t"] + 0.0003, r_edge=0.0012, n=3, dome=0.0004, mat=hw)
    out = [S.place(plate, (0.0, 0.0003, 0.0), OUT, UP)]
    t = L.TAIL
    half = np.array([(math.cos(a) * t["r"], math.sin(a) * t["r"]) for a in np.linspace(0.0, math.pi, 13)])
    sec = G.ccw(np.vstack([half, [[-t["r"], -0.0004], [t["r"], -0.0004]]]))   # a half-round standing on the plate
    n = 15
    xs = np.linspace(t["x"][0] + 0.002, t["x"][1] - 0.002, n)
    path = np.stack([xs, np.full(n, -b["t"]), np.full(n, t["z"])], 1)
    taper = np.minimum(1.0, np.sin(np.linspace(0.0, math.pi, n)) * 5.0) * 0.85 + 0.15     # its ends roll down to the plate
    tail = S.sweep(path, sec, scale=np.stack([np.ones(n), taper], 1), up=OUT, mat=hw)
    out.append(tail if tail.volume() > 0 else tail.flip())
    y_plate = -b["t"]
    sec = S.rrect(L.SADDLE["w"], L.SADDLE["top"], 0.0018, 4)
    post = lathe([(0.0, 0.0), (0.0012, 0.0), (0.0012, 1.0), (0.0, 1.0)], 0.0001, 10, hw, 1)       # unit height, scaled below
    head = dome_screw(0.0022, 0.0014, hw)
    for i in range(4):
        _nut, sad = (np.array(p) for p in L.string_ends(i))
        top = sad[1] + L.STRING_R[i]                                     # the string lies on the saddle's crest
        zc = sad[2] - 0.0040
        c = np.array([sad[0], top + 0.5 * L.SADDLE["top"], zc])
        path = np.stack([c - np.array([0, 0, 0.5 * L.SADDLE["l"]]), c + np.array([0, 0, 0.5 * L.SADDLE["l"]])])
        blk = S.sweep(path, sec, up=OUT, mat=hw)
        out.append(blk if blk.volume() > 0 else blk.flip())
        h = y_plate - (top + L.SADDLE["top"])
        for dx in (-0.0030, 0.0030):
            p = post.apply(np.diag([1.0, 1.0, abs(h)]), (0, 0, 0))
            out.append(S.place(p, (sad[0] + dx, y_plate, zc), OUT, UP))
        ys = top + 0.5 * L.SADDLE["top"] + 0.0006                        # the intonation screw, forward to the flange
        scr = np.array([[sad[0], ys, zc + 0.5 * L.SADDLE["l"] - 0.001], [sad[0], ys, L.FLANGE_Z - 0.002]])
        out.append(S.tube(scr, 0.0010, sides=8, up=OUT, mat=hw))
        out.append(S.place(lathe([(0.0, 0.0), (0.0024, 0.0), (0.0024, 0.0016), (0.0016, 0.0024), (0.0, 0.0025)], 0.0003, 16,
                                 hw, 2), (sad[0], ys, L.FLANGE_Z - 0.0025), (0.0, 0.0, 1.0), OUT))
    for x, z in L.BRIDGE_SCREWS:
        out.append(put(head, x, y_plate + 0.0001, z))
    return S.merge(out)


# ===================================================================================================================
# the controls on the plate
# ===================================================================================================================
def knob_mesh():
    """A black skirted knob (role `knobs`) with a bright metal cap (role `hardware`), modelled standing on z = 0."""
    body = lathe([(0.0, 0.0), (0.0098, 0.0), (0.0098, 0.0028), (0.0090, 0.0036), (0.0082, 0.0060), (0.0080, 0.0130),
                  (0.0076, 0.0148), (0.0, 0.0150)], 0.0007, 36, M["knobs"], 3)
    cap = lathe([(0.0, 0.0146), (0.0072, 0.0146), (0.0070, 0.0160), (0.0050, 0.0170), (0.0, 0.0172)], 0.0006, 32,
                M["hardware"], 3)
    return S.merge([body, cap])


def knobs_mesh():
    k = knob_mesh()
    return S.merge([put(k, x, Y_P, z) for x, z in L.KNOBS])


def plate_parts_mesh():
    """What stands on the control plate besides the knobs: its three screws and the jack's nut (hardware)."""
    screw = dome_screw(0.0023, 0.0014, M["hardware"])
    out = [put(screw, x, Y_P + 0.0001, z) for x, z in L.PLATE_SCREWS]
    mouth, d, _ = L.jack_geometry()
    nut = lathe([(0.0, 0.0), (0.0068, 0.0), (0.0068, 0.0026), (0.0058, 0.0032), (0.0044, 0.0034), (0.0, 0.0034)], 0.0005, 24,
                M["hardware"], 3)
    cx, cz = L.JACK["center"]
    out.append(put(nut, cx, Y_P + 0.0001, cz))
    return S.merge(out)


def plug_mesh():
    """The cable's 1/4 inch plug standing straight out of the jack: a metal body and a dark boot. Its end, where the cord
    leaves, is the card's `jack` anchor."""
    mouth, d, _ = L.jack_geometry()
    n = L.JACK["plug_len"]
    body = lathe([(0.0, 0.0), (0.0050, 0.0), (0.0050, 0.0040), (0.0061, 0.0046), (0.0061, 0.0400), (0.0056, 0.0412),
                  (0.0, 0.0414)], 0.0006, 28, M["hardware"], 3)
    boot = lathe([(0.0, 0.0405), (0.0054, 0.0405), (0.0050, 0.0440), (0.0040, 0.0500), (0.0036, n - 0.0004), (0.0, n)], 0.0006, 20,
                 M["dark"], 3)
    return S.merge([S.place(body, mouth, d, UP), S.place(boot, mouth, d, UP)])


def decals_mesh():
    """Graphic layers (tagged exempt by the builder), role `dark`: the four holes the strings go down through behind the
    saddles, discs a fifth of a millimetre proud of the plate."""
    hole = lathe([(0.0, 0.0), (0.0024, 0.0), (0.0024, 0.0002), (0.0, 0.0002)], 0.00005, 14, M["dark"], 1)
    return S.merge([put(hole, x, -L.BRIDGE["t"] + 0.00005, L.STRING_HOLE_Z) for x in L.SADDLE_X])


def guard_screws_mesh():
    screw = dome_screw(0.0024, 0.0015, M["hardware"])
    return S.merge([put(screw, x, float(B.face(x, z)) - L.GUARD_T + 0.0001, z) for x, z in B.guard_screw_points()])


# ===================================================================================================================
# strap buttons
# ===================================================================================================================
def button_mesh():
    return lathe([(0.0, 0.0), (0.0075, 0.0), (0.0075, 0.0012), (0.0050, 0.0014), (0.0034, 0.0016), (0.0034, 0.0086),
                  (0.0062, 0.0092), (0.0092, 0.0102), (0.0098, 0.0120), (0.0090, 0.0142), (0.0060, 0.0154), (0.0, 0.0158)],
                 0.0005, 28, M["hardware"], 3)


def buttons_mesh():
    b = button_mesh()
    out = []
    for spec in (L.STRAP_TOP, L.STRAP_BOTTOM):
        d = np.array(spec["dir"], float)
        d /= np.linalg.norm(d)
        out.append(S.place(b, np.array(spec["point"], float) - d * 0.0003, d, (0.0, 1.0, 0.0)))
    return S.merge(out)


# ===================================================================================================================
# the back
# ===================================================================================================================
NECK_PLATE_Z = 0.2780                        # the neck plate's centre on the back, under the neck's heel


def back_mesh():
    """What you see turning the bass over: the neck plate with its four screws and the four string ferrules behind the bridge
    (role `hardware`), sunk into the back so the belly cut leaves no gap."""
    hw = M["hardware"]
    back = (0.0, 1.0, 0.0)
    yb = float(B.back(0.0, NECK_PLATE_Z)) - 0.0012
    plate = S.rounded_panel(S.rrect(0.0500, 0.0640, 0.0050, 5), 0.0034, r_edge=0.0010, n=3, dome=0.0003, mat=hw)
    screw = dome_screw(0.0027, 0.0017, hw)
    out = [put(plate, 0.0, yb, NECK_PLATE_Z, back)]
    for sx in (-0.0165, 0.0165):
        for sz in (-0.0235, 0.0235):
            out.append(put(screw, sx, yb + 0.0033, NECK_PLATE_Z + sz, back))
    ferrule = lathe([(0.0, 0.0), (0.0045, 0.0), (0.0045, 0.0008), (0.0034, 0.0014), (0.0022, 0.0016), (0.0022, 0.0004),
                     (0.0, 0.0004)], 0.0002, 18, hw, 2)
    for x in L.SADDLE_X:
        yf = float(B.back(x, L.STRING_HOLE_Z)) - 0.0004
        out.append(put(ferrule, x, yf, L.STRING_HOLE_Z, back))
    return S.merge(out)


# ===================================================================================================================
# the plectrum
# ===================================================================================================================
def pick_outline():
    """The 351 shape's outline (m, 2) in the pinch frame: a rounded triangle, tip at x = tip, base at x = tip - length, centred
    across (y), `width` wide at the base corners."""
    p = L.PICK
    L0, W0 = p["length"], p["width"]
    for _ in range(6):
        P = np.array([[0.0, 0.0], [-L0, 0.5 * W0], [-L0, -0.5 * W0]])
        o = S.fillet(P, [0.0034, 0.0075, 0.0075], n=10, closed=True)
        L0 *= p["length"] / (o[:, 0].max() - o[:, 0].min())
        W0 *= p["width"] / (o[:, 1].max() - o[:, 1].min())
    o[:, 0] += p["tip"] - o[:, 0].max()
    return G.ccw(o)


def pick_mesh():
    """The plectrum in the pinch frame of the grip solver: origin midway between the pads (the pick's mid-plane), +x toward the
    tip (at x = tip), y across its width, z across its thickness. Role `pick`."""
    t = L.PICK["thickness"]
    m = S.rounded_panel(pick_outline(), t, r_edge=0.0004, n=2, mat=M["pick"])
    return m.moved((0.0, 0.0, -0.5 * t))


def pick_rest():
    """(location, 3x3 rotation) of the pick object where it lies on the body's face between the bridge and the guard: local x
    toward +z, the face (local z) out of the bass's face (-y)."""
    x, z = L.PICK_REST["at"]
    R = np.array([[0.0, -1.0, 0.0],                  # columns: the images of the pick's x, y, z axes in the bass's axes
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])
    return np.array([x, float(B.face(x, z)) - 0.5 * L.PICK["thickness"] - 0.0001, z]), R


# ===================================================================================================================
# the cable
# ===================================================================================================================
def cable_path(n=22):
    """Points (m, 3) of the default cord: from the plug's end along the plug, in a soft curve down to the floor under the tail
    (z = FLOOR_Z), then a short way along the floor."""
    _, d, J = L.jack_geometry()
    P0 = J
    P3 = np.array([J[0] + 0.020, J[1] - 0.030, L.FLOOR_Z])
    P1 = P0 + d * 0.040
    P2 = P3 + np.array([0.0, 0.0, 0.060])
    t = np.linspace(0.0, 1.0, n)[:, None]
    bez = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    floor = np.array([[P3[0] + 0.02 * k, P3[1] - 0.012 * k, L.FLOOR_Z] for k in range(1, 6)])
    return np.concatenate([bez, floor])
