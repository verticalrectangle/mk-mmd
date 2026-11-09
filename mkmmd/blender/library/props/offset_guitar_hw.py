"""The offset guitar's hardware (pure numpy, no bpy): two wide pickups, the rocking bridge with its barrel saddles, the
floating tremolo (plate, string anchor, lock button, collet) and its arm, the lead circuit's knobs, toggle and jack, the
rhythm circuit's slide switch and thumb wheels, the screws, the strap buttons, the neck plate, the plug, the plectrum's
rest and the cable's path. Everything stands on the pickguard (y = -GUARD_T) or the body's face (y = 0) at the positions
of `offset_guitar_layout`, modelled from lathes, tubes and rolled panels (the electric guitar's helpers).

The tremolo ARM is an object of its own (`arm_mesh`, in the arm's frame: its origin the bend above the collet, +Z along the
bar to the tip): the builder stands it at `layout.arm_geometry` and makes it aim at an empty, so a hand can carry its tip.
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_guitar_hw as EH
from . import electric_guitar_neck as N
from . import offset_guitar_body as B
from . import offset_guitar_layout as L

M = L.MATS
OUT, UP = EH.OUT, EH.UP
Y_G = -L.GUARD_T                                   # the pickguard's top
Y_T = -L.TREM["t"]                                 # the tremolo plate's top
lathe, put, dome_screw = EH.lathe, EH.put, EH.dome_screw


# ===================================================================================================================
# pickups
# ===================================================================================================================
def pickups_mesh():
    """The two pickups: a flat plastic cover with round corners standing on the guard, six pole pieces flush in its top
    and a screw near each corner. Roles `knobs` (the covers) and `hardware`."""
    cover = S.rounded_panel(S.rrect(L.PICKUP_L, L.PICKUP_W, L.PICKUP_R, 5), L.PICKUP_H + 0.0002, r_edge=0.0016, n=3,
                            dome=0.0003, mat=M["knobs"])
    pole = lathe([(0.0, 0.0), (L.POLE_R, 0.0), (L.POLE_R, L.POLE_H), (0.7 * L.POLE_R, L.POLE_H + 0.0003),
                  (0.0, L.POLE_H + 0.0003)], 0.0002, 14, M["hardware"], 2)
    screw = dome_screw(0.0019, 0.0012, M["hardware"])
    y_top = Y_G - L.PICKUP_H
    out = []
    for spec in L.PICKUPS.values():
        z = spec["z"]
        out.append(put(cover, 0.0, Y_G + 0.0002, z))
        out += [put(pole, (i - 2.5) * L.POLE_PITCH, y_top + 0.0003, z) for i in range(6)]
        out += [put(screw, dx, y_top + 0.0002, z + dz) for dx, dz in L.PICKUP_SCREWS]
    return S.merge(out)


# ===================================================================================================================
# the bridge
# ===================================================================================================================
def saddle_axis(i):
    """The middle of string i's barrel saddle: the string lies on its crest at the card's bridge point."""
    _nut, sad = L.string_ends(i)
    return np.array([sad[0], sad[1] + L.STRING_R[i] + L.SADDLE["r"], sad[2]])


def bridge_mesh():
    """The rocking bridge: a bar in two cups on the guard, and on it six barrel saddles across the strings, each on two
    height screws and grooved where its string lies. Role `hardware`."""
    hw, b, s = M["hardware"], L.BRIDGE, L.SADDLE
    w, d = b["size"]
    y_bar = max(saddle_axis(i)[1] for i in range(6)) + s["r"] + 0.0003          # the bar's top: under the lowest barrel
    bar = S.rounded_panel(S.rrect(w, d, 0.0042, 5), b["t"], r_edge=0.0010, n=3, dome=0.0002, mat=hw)
    out = [put(bar, 0.0, y_bar + b["t"], b["z"])]
    cup = lathe([(0.0, 0.0), (b["cup_r"], 0.0), (b["cup_r"], 0.0012), (b["cup_r"] - 0.0012, 0.0019), (0.0, 0.0019)],
                0.0004, 20, hw, 2)
    out += [put(cup, sx * b["posts"], Y_G + 0.0001, b["z"]) for sx in (-1.0, 1.0)]
    r, ln = s["r"], s["l"]
    barrel = lathe([(0.0, 0.0), (r, 0.0), (r, 0.40 * ln), (r - 0.0007, 0.50 * ln), (r, 0.60 * ln), (r, ln), (0.0, ln)],
                   0.0004, 18, hw, 2)
    post = lathe([(0.0, 0.0), (0.0010, 0.0), (0.0010, 1.0), (0.0, 1.0)], 0.0001, 8, hw, 1)       # unit height, scaled below
    for i in range(6):
        c = saddle_axis(i)
        out.append(S.place(barrel, (c[0] - 0.5 * ln, c[1], c[2]), (1.0, 0.0, 0.0), UP))
        h = y_bar - (c[1] + 0.6 * r)                                    # the height screws: from inside the barrel to the bar
        if h > 0.0002:
            for dx in (-0.0030, 0.0030):
                p = post.apply(np.diag([1.0, 1.0, h + 0.0004]), (0, 0, 0))
                out.append(S.place(p, (c[0] + dx, y_bar + 0.0002, c[2]), OUT, UP))
    return S.merge(out)


# ===================================================================================================================
# the tremolo
# ===================================================================================================================
def trem_outline():
    """The tremolo plate's outline (m, 2) of (x, z) about its middle, counter-clockwise: square with round corners at
    the tail, a low arch toward the bridge."""
    t = L.TREM
    (z0, z1), w = t["z"], t["w"]
    zc = z1 - t["arch"]
    sides = S.fillet(np.array([(-0.5 * w, zc), (-0.5 * w, z0), (0.5 * w, z0), (0.5 * w, zc)]),
                     [0.0, t["corner"], t["corner"], 0.0], 4)
    a = np.linspace(0.0, math.pi, 28)[1:-1]
    arch = np.column_stack([0.5 * w * np.cos(a), zc + t["arch"] * np.sin(a)])
    return np.concatenate([sides, arch]) - np.array([0.0, 0.5 * (z0 + z1)])


def bend_point():
    """Where the arm's bar turns over, above the collet (the arm object's origin)."""
    return L.arm_geometry()[0]


def tremolo_mesh():
    """The floating tremolo as it shows on the face: the chrome plate, the rocking plate inside it, the bar the strings
    are anchored to, the lock button, the collet with the stub of the arm's bar standing in it up to its bend (the arm
    itself turns about that bend: `arm_mesh`), and the plate's screws. Role `hardware`."""
    hw, t = M["hardware"], L.TREM
    zm = 0.5 * sum(t["z"])
    out = [put(S.rounded_panel(trem_outline(), t["t"] + 0.0003, r_edge=0.0009, n=3, dome=0.0005, mat=hw), 0.0, 0.0003, zm)]
    rocker = S.rounded_panel(S.rrect(0.0600, 0.0440, 0.0040, 4), 0.0012, r_edge=0.0005, n=2, dome=0.0002, mat=hw)
    out.append(put(rocker, 0.0, Y_T + 0.0001, -0.1010))
    a = L.ANCHOR
    bar = S.rounded_panel(S.rrect(a["size"][0], a["size"][1], 0.0022, 4), a["h"], r_edge=0.0011, n=3, dome=0.0002, mat=hw)
    out.append(put(bar, 0.0, Y_T + 0.0001, a["z"]))
    k = L.LOCK
    lock = lathe([(0.0, 0.0), (k["r"], 0.0), (k["r"], 0.7 * k["h"]), (0.8 * k["r"], k["h"]), (0.0, k["h"])], 0.0005, 20, hw, 2)
    out.append(put(lock, k["at"][0], Y_T + 0.0001, k["at"][1]))
    c = L.COLLET
    collet = lathe([(0.0, 0.0), (c["r"] + 0.0012, 0.0), (c["r"] + 0.0012, 0.0012), (c["r"], 0.0018), (c["r"], c["h"]),
                    (c["r"] - 0.0010, c["h"] + 0.0004), (0.0, c["h"] + 0.0004)], 0.0004, 20, hw, 2)
    out.append(put(collet, c["at"][0], Y_T + 0.0001, c["at"][1]))
    bend = bend_point()
    stub = S.tube(np.array([[bend[0], Y_T - c["h"] + 0.001, bend[2]], bend]), L.ARM_R, sides=10, up=UP, mat=hw)
    ball = lathe([(0.0, 0.0), (0.0019, 0.0007), (0.0031, 0.0031), (0.0019, 0.0055), (0.0, 0.0062)], 0.0009, 14, hw, 3)
    out += [stub, put(ball, bend[0], bend[1] + 0.0031, bend[2])]
    screw = dome_screw(0.0023, 0.0013, hw)
    out += [put(screw, x, Y_T + 0.0001, z) for x, z in L.TREM_SCREWS]
    return S.merge(out)


def arm_mesh(arm=None):
    """The tremolo arm in its own frame (origin the bend, +Z to the tip's end, +Y out of the face): a steel bar bowed a
    little off the face and its plastic tip. `arm`: `layout.ARM`'s keys. Roles `hardware` and `knobs`."""
    a = dict(L.ARM, **(arm or {}))
    tip = L.ARM_TIP
    run = a["length"] - tip["l"] + 0.004                                   # the bar ends inside the tip
    s = np.linspace(0.0, 1.0, 13)
    path = np.column_stack([np.zeros_like(s), 0.0035 * np.sin(math.pi * s), run * s])
    bar = S.tube(path, L.ARM_R, sides=10, up=(1.0, 0.0, 0.0), mat=M["hardware"])
    r, ln = tip["r"], tip["l"]
    cap = lathe([(0.0, 0.0), (0.72 * r, 0.0), (r, 0.20 * ln), (0.96 * r, 0.72 * ln), (0.70 * r, 0.93 * ln), (0.0, ln)],
                0.0012, 16, M["knobs"], 3)
    return S.merge([bar, cap.moved((0.0, 0.0, a["length"] - ln))])


# ===================================================================================================================
# controls
# ===================================================================================================================
def knobs_mesh():
    k = EH.knob_mesh()
    return S.merge([put(k, x, Y_G, z) for x, z in L.KNOBS])


def roller_axis_y():
    """y of a thumb wheel's axle: under the guard, so the wheel's rim stands `ROLLER['out']` out of its slot."""
    return Y_G - L.ROLLER["out"] + L.ROLLER["r"]


def switch_mesh():
    """The three-way toggle (a nut on the guard, a lever leaning out of it with a plastic tip), the rhythm circuit's slide
    switch (a nub in a small chrome frame) and its two thumb wheels (their rims in slots of the guard). Roles `hardware`,
    `knobs` and `dark`."""
    hw = M["hardware"]
    tx, tz = L.TOGGLE["at"]
    lean = np.array(L.TOGGLE["lean"], float) + np.array([0.0, -1.0, 0.0])
    lean /= np.linalg.norm(lean)
    nut = lathe([(0.0, 0.0), (0.0068, 0.0), (0.0068, 0.0018), (0.0052, 0.0026), (0.0034, 0.0040), (0.0, 0.0040)], 0.0005, 20, hw, 2)
    shaft = lathe([(0.0, 0.0), (0.0019, 0.0), (0.0015, 0.0092), (0.0, 0.0092)], 0.0003, 12, hw, 2)
    tip = lathe([(0.0, 0.0082), (0.0026, 0.0086), (0.0030, 0.0140), (0.0024, 0.0166), (0.0, 0.0170)], 0.0008, 14, M["knobs"], 3)
    base = np.array([tx, Y_G - 0.0034, tz])
    out = [put(nut, tx, Y_G, tz), S.place(shaft, base, lean, UP), S.place(tip, base, lean, UP)]
    sx, sz = L.SLIDE["at"]
    frame = S.rounded_panel(S.rrect(0.0210, 0.0100, 0.0018, 3), 0.0009, r_edge=0.0004, n=2, mat=hw)
    out.append(put(frame, sx, Y_G + 0.0001, sz))
    out.append(S.rounded_box((0.0046, 0.0030, 0.0034), (sx - 0.0028, Y_G - 0.0021, sz), r=0.0008, k=2, mat=M["dark"]))
    r, w = L.ROLLER["r"], L.ROLLER["w"]
    wheel = lathe([(0.0, 0.0), (r - 0.0006, 0.0), (r, 0.0008), (r, w - 0.0008), (r - 0.0006, w), (0.0, w)], 0.0004, 28,
                  M["dark"], 2)
    out += [S.place(wheel, (x - 0.5 * w, roller_axis_y(), z), (1.0, 0.0, 0.0), UP) for x, z in L.ROLLERS]
    return S.merge(out)


def decals_mesh():
    """Graphic layers a hair above the guard: the dark slots of the slide switch and of the two thumb wheels (round
    their rims). Role `dark`."""
    out = []
    w, ln = L.SLIDE["slot"]
    out.append(put(S.rounded_panel(S.rrect(w, ln, 0.0012, 3), 0.0002, r_edge=0.0001, n=1, mat=M["dark"]),
                   L.SLIDE["at"][0], Y_G - 0.00095, L.SLIDE["at"][1]))
    w, ln = L.ROLLER["slot"]
    slot = S.rounded_panel(S.rrect(w, ln, 0.0010, 3), 0.0002, r_edge=0.0001, n=1, mat=M["dark"])
    out += [put(slot, x, Y_G - 0.00005, z) for x, z in L.ROLLERS]
    return S.merge(out)


def guard_screws_mesh():
    screw = dome_screw(0.0024, 0.0015, M["hardware"])
    return S.merge([put(screw, x, B.face(x, z) - L.GUARD_T + 0.0001, z) for x, z in L.GUARD_SCREWS])


# ===================================================================================================================
# jack and plug
# ===================================================================================================================
def jack_mesh():
    """The output jack's nut on the guard. Role `hardware`."""
    x, z = L.JACK["at"]
    nut = lathe([(0.0, 0.0), (0.0082, 0.0), (0.0082, 0.0016), (0.0062, 0.0023), (0.0048, 0.0026), (0.0, 0.0026)], 0.0005, 24,
                M["hardware"], 2)
    return put(nut, x, Y_G, z)


def plug_mesh():
    """The cable's 1/4 inch plug standing in the jack, along the way out of the socket: a metal body and a dark boot. Its
    end, where the cord leaves, is the card's `jack` anchor."""
    mouth, d, _ = L.jack_geometry()
    n = L.JACK["plug_len"]
    body = lathe([(0.0, 0.0), (0.0050, 0.0), (0.0050, 0.0040), (0.0061, 0.0046), (0.0061, 0.0400), (0.0056, 0.0412),
                  (0.0, 0.0414)], 0.0006, 28, M["hardware"], 3)
    boot = lathe([(0.0, 0.0405), (0.0054, 0.0405), (0.0050, 0.0440), (0.0040, 0.0500), (0.0036, n - 0.0004), (0.0, n)],
                 0.0006, 20, M["dark"], 3)
    return S.merge([S.place(body, mouth, d, UP), S.place(boot, mouth, d, UP)])


# ===================================================================================================================
# strap buttons, the back
# ===================================================================================================================
def buttons_mesh():
    b = EH.button_mesh()
    out = []
    for spec in (L.STRAP_TOP, L.STRAP_BOTTOM):
        d = np.array(spec["dir"], float)
        d /= np.linalg.norm(d)
        out.append(S.place(b, np.array(spec["point"], float) - d * 0.0003, d, (0.0, 1.0, 0.0)))
    return S.merge(out)


def back_mesh():
    """What you see turning the guitar over: the neck plate with its four screws, sunk 1.2 mm into the back. Role
    `hardware`."""
    hw = M["hardware"]
    yb = L.BODY_T - 0.0012
    back = (0.0, 1.0, 0.0)
    zn = L.NECK_PLATE_Z
    out = [put(S.rounded_panel(S.rrect(0.0480, 0.0580, 0.0045, 5), 0.0034, r_edge=0.0010, n=3, dome=0.0003, mat=hw),
               0.0, yb, zn, back)]
    screw = dome_screw(0.0027, 0.0017, hw)
    out += [put(screw, sx, L.BODY_T + 0.0021, zn + sz, back) for sx in (-0.0160, 0.0160) for sz in (-0.0215, 0.0215)]
    return S.merge(out)


# ===================================================================================================================
# the strings, the plectrum, the cable
# ===================================================================================================================
def string_meshes():
    """The six strings, the low E first, each its own thin tube (role `strings`): wound round its post, over the nut,
    EXACTLY along the card's nut and bridge points (a ring at every wire and where a finger holds it down:
    `electric_guitar_neck.playable`), over its saddle and on down to the tremolo's anchor bar."""
    a = L.ANCHOR
    wires = [L.fret_z(n) for n in range(L.N_FRETS + 1)]
    out = []
    for i in range(6):
        nut, sad = (np.array(p) for p in L.string_ends(i))
        back = np.array([nut[0], nut[1], N.Z_NUT_BACK])
        wraps = N.post_wraps(i, N.Y_FACE - 0.0062)
        y = Y_T - 0.55 * a["h"]
        tail = np.array([[sad[0], y, a["z"] + 0.5 * a["size"][1] + 0.0004], [sad[0], y, a["z"]]])
        path = np.concatenate([wraps[::-1], [back], N.playable(nut, sad, wires), tail])
        out.append(S.tube(path, L.STRING_R[i], sides=6, up=OUT, mat=M["strings"]))
    return out


def pick_rest():
    """(location, 3x3 rotation) of the pick object where it lies on the pickguard beside the strum zone: local x toward
    +z, the face (local z) out of the guitar's face (-y)."""
    x, z = L.PICK_REST["at"]
    R = np.array([[0.0, -1.0, 0.0],                  # columns: the images of the pick's x, y, z axes in the guitar's axes
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])
    return np.array([x, Y_G - 0.5 * L.PICK["thickness"] - 0.0001, z]), R


def cable_path(n=22):
    """Points (m, 3) of the default cord: from the plug's end along the plug, in a soft curve down to the floor under the
    tail (z = FLOOR_Z), then a short way along the floor."""
    _, d, J = L.jack_geometry()
    P0 = J
    P3 = np.array([J[0] + 0.035, J[1] - 0.018, L.FLOOR_Z])
    P1 = P0 + d * 0.040
    P2 = P3 + np.array([0.0, 0.0, 0.045])
    t = np.linspace(0.0, 1.0, n)[:, None]
    bez = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    floor = np.array([[P3[0] + 0.02 * k, P3[1] - 0.012 * k, L.FLOOR_Z] for k in range(1, 6)])
    return np.concatenate([bez, floor])
