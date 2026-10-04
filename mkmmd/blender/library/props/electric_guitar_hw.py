"""The guitar's hardware (pure numpy, no bpy): pickups, bridge with its saddles, knobs, the five-way switch, the jack plate and
plug, strap buttons, the plectrum and the cable's path. Everything stands on the pickguard (y = -GUARD_T) or the body's
face (y = 0) at the positions of `electric_guitar_layout`, modelled from lathes, sweeps and rolled panels.
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_guitar_body as B
from . import electric_guitar_geo as G
from . import electric_guitar_layout as L

M = L.MATS
OUT = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)
Y_G = -L.GUARD_T                                   # the pickguard's top


def lathe(pts, r=0.0005, seg=24, mat=0, n=3):
    return S.lathe(S.fillet(np.array(pts, float), r, n), seg=seg, mat=mat)


def put(mesh, x, y, z, normal=OUT, up=UP):
    return S.place(mesh, (x, y, z), normal, up)


def dome_screw(r=0.0026, h=0.0016, mat=0):
    return lathe([(0.0, 0.0), (r, 0.0), (r, 0.45 * h), (0.75 * r, 0.85 * h), (0.0, h)], 0.0004, 16, mat, 2)


# ===================================================================================================================
# pickups
# ===================================================================================================================
def pickups_mesh():
    """Three single coils: a rolled stadium cover (role `knobs`), six pole pieces (role `hardware`), two height screws each."""
    cover = S.rounded_panel(S.stadium(L.PICKUP_L, L.PICKUP_W, 8), L.PICKUP_H, r_edge=0.0022, n=3, dome=0.0005, mat=M["knobs"])
    pole = lathe([(0.0, 0.0), (L.POLE_R, 0.0), (L.POLE_R, 0.55 * L.POLE_H), (0.8 * L.POLE_R, 0.9 * L.POLE_H), (0.0, L.POLE_H)],
                 0.0004, 16, M["hardware"], 2)
    screw = dome_screw(mat=M["hardware"])
    out = []
    for spec in L.PICKUPS.values():
        s = math.radians(spec["slant"])
        u = np.array([math.cos(s), 0.0, -math.sin(s)])                 # along the cover: bass end toward the neck
        up = np.array([math.sin(s), 0.0, math.cos(s)])
        c = np.array([0.0, Y_G, spec["z"]])
        out.append(S.place(cover, c, OUT, up))
        for k in range(6):
            p = c + u * ((k - 2.5) * L.POLE_PITCH)
            out.append(put(pole, p[0], Y_G - L.PICKUP_H - 0.0001, p[2]))
        for sgn in (-1.0, 1.0):
            p = c + u * (sgn * L.PICKUP_SCREW_X)
            out.append(put(screw, p[0], Y_G + 0.0001, p[2]))
    return S.merge(out)


# ===================================================================================================================
# bridge
# ===================================================================================================================
def bridge_mesh():
    """The tremolo plate (a rolled panel on the face), six saddles on height screws, the plate's six front screws. Hardware."""
    b = L.BRIDGE
    hw = M["hardware"]
    w, d = b["size"]
    plate = S.rounded_panel(S.rrect(w, d, b["corner"], 5), b["t"] + 0.0003, r_edge=0.0013, n=3, dome=0.0004, mat=hw)
    cx, cz = b["center"]
    out = [put(plate, cx, 0.0003, cz)]
    y_plate = -b["t"]
    sec = S.rrect(L.SADDLE["w"], L.SADDLE["top"], 0.0016, 4)
    post = lathe([(0.0, 0.0), (0.0011, 0.0), (0.0011, 1.0), (0.0, 1.0)], 0.0001, 10, hw, 1)       # unit height, scaled below
    head = dome_screw(0.0028, 0.0016, hw)
    for i in range(6):
        nut, sad = (np.array(p) for p in L.string_ends(i))
        top = sad[1] + L.STRING_R[i]                                     # the string lies on the saddle's crest
        zc = sad[2] - 0.0020
        c = np.array([sad[0], top + 0.5 * L.SADDLE["top"], zc])
        path = np.stack([c - np.array([0, 0, 0.5 * L.SADDLE["l"]]), c + np.array([0, 0, 0.5 * L.SADDLE["l"]])])
        blk = S.sweep(path, sec, up=OUT, mat=hw)
        out.append(blk if blk.volume() > 0 else blk.flip())
        h = y_plate - (top + L.SADDLE["top"])
        for dx in (-0.0028, 0.0028):
            p = post.apply(np.diag([1.0, 1.0, abs(h)]), (0, 0, 0))
            out.append(S.place(p, (sad[0] + dx, y_plate, zc + 0.0010), OUT, UP))
        out.append(put(head, (i - 2.5) * 0.0130, y_plate + 0.0001, cz + 0.0166))
    return S.merge(out)


# ===================================================================================================================
# controls
# ===================================================================================================================
def knob_mesh():
    return lathe([(0.0, 0.0), (0.0099, 0.0), (0.0099, 0.0022), (0.0094, 0.0030), (0.0090, 0.0058), (0.0087, 0.0105),
                  (0.0089, 0.0138), (0.0084, 0.0158), (0.0070, 0.0166), (0.0040, 0.0168), (0.0, 0.0168)], 0.0007, 36, M["knobs"], 3)


def knobs_mesh():
    k = knob_mesh()
    return S.merge([put(k, x, Y_G, z) for x, z in L.KNOBS])


def slot_dir():
    a = math.radians(L.SWITCH["angle"])
    return np.array([math.cos(a), 0.0, math.sin(a)])


def switch_mesh():
    """The five-way switch's lever and tip: the lever at the middle of its slot. Roles `hardware` (stem) and `knobs` (tip)."""
    cx, cz = L.SWITCH["center"]
    stem = S.tube(np.array([[cx, Y_G + 0.0002, cz], [cx, Y_G - 0.0070, cz]]), 0.0012, sides=10, up=OUT, mat=M["hardware"])
    tip = lathe([(0.0, 0.0), (0.0034, 0.0), (0.0034, 0.0075), (0.0028, 0.0092), (0.0, 0.0096)], 0.0007, 16, M["knobs"], 3)
    out = [stem, put(tip, cx, Y_G - 0.0066, cz)]
    screw = dome_screw(0.0022, 0.0014, M["hardware"])
    for x, z in ((0.0814, 0.0771), (0.1029, 0.0800)):
        out.append(put(screw, x, Y_G + 0.0001, z))
    return S.merge(out)


def decals_mesh():
    """Graphic layers (tagged exempt by the builder), role `dark`: the switch's slot in the pickguard, a thin ribbon, and the six
    holes the strings go down through behind the saddles, discs a fifth of a millimetre proud of the plate."""
    l, w = L.SWITCH["slot"]
    p = S.rounded_panel(S.stadium(w, l, 6), 0.0002, r_edge=0.0001, n=1, mat=M["dark"])
    cx, cz = L.SWITCH["center"]
    out = [S.place(p, (cx, Y_G - 0.00005, cz), OUT, slot_dir())]
    hole = lathe([(0.0, 0.0), (0.0021, 0.0), (0.0021, 0.0002), (0.0, 0.0002)], 0.00005, 14, M["dark"], 1)
    for x in L.SADDLE_X:
        out.append(put(hole, x, -L.BRIDGE["t"] + 0.00005, L.BRIDGE_HOLE_Z))
    return S.merge(out)


def guard_screws_mesh():
    screw = dome_screw(0.0024, 0.0015, M["hardware"])
    return S.merge([put(screw, x, B.face(x, z) - L.GUARD_T + 0.0001, z) for x, z in L.GUARD_SCREWS])


# ===================================================================================================================
# jack and plug
# ===================================================================================================================
def jack_plate_mesh():
    """The output jack's boat plate: an oval of rolled chrome on the face, turned along its long axis, with the socket's nut
    at its upper end. Role `hardware`."""
    j = L.JACK
    a = math.radians(j["angle"])
    s_hat = np.array([-math.sin(a), 0.0, math.cos(a)])              # across the plate; its long axis is then along (cos a, 0, sin a)
    plate = S.rounded_panel(S.superellipse(j["plate"][0], j["plate"][1], 2.3, 44), j["t"] + 0.0004, r_edge=0.0016, n=3,
                            dome=0.0010, mat=M["hardware"])
    cx, cz = j["center"]
    mouth, d, _ = L.jack_geometry()
    nut = lathe([(0.0, 0.0), (0.0075, 0.0), (0.0075, 0.0030), (0.0066, 0.0038), (0.0050, 0.0040), (0.0, 0.0040)], 0.0006, 24,
                M["hardware"], 3)
    return S.merge([put(plate, cx, 0.0004, cz, OUT, s_hat), S.place(nut, mouth, d, UP)])


def plug_mesh():
    """The cable's 1/4 inch plug standing in the jack, along the way out of the socket: a metal body and a dark boot. Its
    end, where the cord leaves, is the card's `jack` anchor."""
    mouth, d, _ = L.jack_geometry()
    n = L.JACK["plug_len"]
    body = lathe([(0.0, 0.0), (0.0050, 0.0), (0.0050, 0.0040), (0.0061, 0.0046), (0.0061, 0.0400), (0.0056, 0.0412),
                  (0.0, 0.0414)], 0.0006, 28, M["hardware"], 3)
    boot = lathe([(0.0, 0.0405), (0.0054, 0.0405), (0.0050, 0.0440), (0.0040, 0.0500), (0.0036, n - 0.0004), (0.0, n)], 0.0006, 20,
                 M["dark"], 3)
    return S.merge([S.place(body, mouth, d, UP), S.place(boot, mouth, d, UP)])


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
def back_mesh():
    """What you see turning the guitar over: the neck plate with its four screws (role `hardware`) and the tremolo cavity's cover
    with its six (the cover is the guard's colour), both sunk 1.2 mm into the back so the belly cut leaves no gap."""
    hw = M["hardware"]
    yb = L.BODY_T - 0.0012
    plate = S.rounded_panel(S.rrect(0.0480, 0.0580, 0.0045, 5), 0.0034, r_edge=0.0010, n=3, dome=0.0003, mat=hw)
    cover = S.rounded_panel(S.rrect(0.0780, 0.1250, 0.0070, 6), 0.0032, r_edge=0.0008, n=2, dome=0.0002, mat=M["pickguard"])
    screw = dome_screw(0.0027, 0.0017, hw)
    back = (0.0, 1.0, 0.0)
    zn, zc = L.NECK_PLATE_Z, L.COVER_Z
    out = [put(plate, 0.0, yb, zn, back), put(cover, 0.0, yb, zc, back)]
    y_face = L.BODY_T + 0.0022
    for sx in (-0.0160, 0.0160):
        for sz in (-0.0215, 0.0215):
            out.append(put(screw, sx, y_face - 0.0001, zn + sz, back))
    for sx in (-0.0335, 0.0335):
        for dz in (-0.0535, 0.0, 0.0535):
            out.append(put(screw, sx, y_face - 0.0003, zc + dz, back))
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
    m = S.rounded_panel(pick_outline(), t, r_edge=0.00035, n=2, mat=M["pick"])
    return m.moved((0.0, 0.0, -0.5 * t))


def pick_rest():
    """(location, 3x3 rotation) of the pick object where it lies on the pickguard near the strum zone: local x toward +z, the
    face (local z) out of the guitar's face (-y)."""
    x, z = L.PICK_REST["at"]
    R = np.array([[0.0, -1.0, 0.0],                  # columns: the images of the pick's x, y, z axes in the guitar's axes
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])
    return np.array([x, Y_G - 0.5 * L.PICK["thickness"] - 0.0001, z]), R


# ===================================================================================================================
# the cable
# ===================================================================================================================
def cable_path(n=22):
    """Points (m, 3) of the default cord: from the plug's end along the plug, in a soft curve down to the floor under the tail
    (z = FLOOR_Z), then a short way along the floor."""
    _, d, J = L.jack_geometry()
    P0 = J
    P3 = np.array([J[0] + 0.035, J[1] - 0.018, L.FLOOR_Z])
    P1 = P0 + d * 0.040
    P2 = P3 + np.array([0.0, 0.0, 0.045])
    t = np.linspace(0.0, 1.0, n)[:, None]
    bez = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    floor = np.array([[P3[0] + 0.02 * k, P3[1] - 0.012 * k, L.FLOOR_Z] for k in range(1, 6)])
    return np.concatenate([bez, floor])
