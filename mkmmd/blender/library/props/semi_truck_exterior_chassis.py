"""The truck's hardware along the frame (bpy-free): the two 7 in chrome stacks (an elbow from the rail, the long
vertical run in the notch between the cab and the sleeper, a top bent back 30 degrees with an open, rolled mouth, a
perforated heat shield low on the stack with clamp bands, brackets to the body), the polished fuel tanks (dished heads
with crimped rims, two stainless straps on rubber liners, cradle brackets from the rails, a filler cap), the step boxes
under the doors (a formed polished face, diamond-plate treads, end plates, a lower step on hangers), the diamond-plate
deck behind the sleeper, the coiled air and electric lines from a bulkhead on the sleeper's back over a hose tender to
the van's glad hands and socket, and the tail lamps on a bar at the frame's end. Every function returns lists of meshes
in the truck's frame; `body` is a kit.Body."""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_exterior_kit as K
from . import semi_truck_layout as L

M = L.M

STACK = dict(bottom_z=0.90, bend_r=0.21, top_bend_r=0.42, top_deg=30.0, tip=0.08, wall=0.0035, sides=32,
             rail_x=L.FRAME_X + 0.004, depth=0.10)
SHIELD = dict(z=(1.30, 2.42), r=0.106, t=0.006, sides=32, bands=(1.345, 1.88, 2.375))
STACK_BRACKETS = (2.62, 3.12)                       # heights of the brackets that tie each stack to the body
TANK_STRAPS = (0.27, 1.13)                          # straps along the tank from its front end
TANK_CAP = (0.20, 45.0)                             # the filler: along the tank, degrees from the top toward outboard
STEP = dict(x_out=1.12, z=(0.62, 0.88), lower_z=0.48, lower_out=0.065, t=0.003)
LINES = dict(z_bulk=1.80, xs=(0.065, 0.0, -0.065), tender=(0.0, -1.98), hold_z=1.90, coil_r=0.027, pitch=0.034,
             hose_r=0.0058)
GLAD_X = 0.15                                       # the van's glad hands: x +-0.15, z 1.55 (TruckTrailer's front panel)
COUPLE_Z = 1.55
TAIL = dict(y=L.FRAME_Y[1] + 0.025, z=0.95, half=0.62, lamps=(0.53,))


# ===================================================================================================================
# stacks
# ===================================================================================================================
def stack_path(sx, z_vertical):
    """The +-X stack's centre line: from the rail's web out to the elbow, up the notch to `z_vertical`, bent back
    STACK['top_deg'] and on to its mouth."""
    st = STACK
    x, y = sx * L.STACK_X, L.STACK_Y
    a = math.radians(st["top_deg"])
    lead = st["top_bend_r"] * math.tan(0.5 * a) + st["tip"]
    pts = [(sx * st["rail_x"], y, st["bottom_z"]), (x, y, st["bottom_z"]), (x, y, z_vertical),
           (x, y + lead * math.sin(a), z_vertical + lead * math.cos(a))]
    P = K.fillet3(pts, [0.0, st["bend_r"], st["top_bend_r"], 0.0], n=10)
    return K.densify(P, 0.5)


def stack_pipe(sx):
    """The chrome pipe of one stack (its mouth open, dark inside), its top at STACK_TOP_Z."""
    st = STACK

    def build(zv):
        return K.tube(stack_path(sx, zv), L.STACK_R, sides=st["sides"], mat=M["chrome"], ends=("flat", "open"),
                      wall=st["wall"], depth=st["depth"], dark=M["seam"], up=(0.0, 1.0, 0.0))

    zv = L.STACK_TOP_Z - 0.25
    m = build(zv)
    zv += L.STACK_TOP_Z - m.V[:, 2].max()
    return build(zv)


def shield(sx):
    """The perforated heat shield: a sleeve of 6 mm sheet round the stack, rolled at both ends, its outer face in the
    `shield` material with uv in metres (u up the stack, v round it from the seam, which faces the body)."""
    sh = SHIELD
    z0, z1 = sh["z"]
    ro, e = sh["r"], 0.5 * sh["t"]
    ri = ro - 2 * e
    sec = [(ro, z0 + e), (ro, z0 + e + 0.004), (ro, z1 - e - 0.004), (ro, z1 - e)]
    sec += [(ro - e + e * math.cos(t), z1 - e + e * math.sin(t)) for t in np.radians((45.0, 90.0, 135.0))]
    sec += [(ri, z1 - e), (ri, z0 + e)]
    sec += [(ro - e + e * math.cos(t), z0 + e + e * math.sin(t)) for t in np.radians((225.0, 270.0, 315.0))]
    n = sh["sides"]
    a = 2.0 * math.pi * np.arange(n) / n
    inward = np.array([-sx, 0.0])
    side = np.array([0.0, 1.0])
    dirs = np.cos(a)[:, None] * inward + np.sin(a)[:, None] * side * sx
    cx, cy = sx * L.STACK_X, L.STACK_Y
    rings = np.stack([np.column_stack([cx + r * dirs[:, 0], cy + r * dirs[:, 1], np.full(n, z)]) for r, z in sec])
    m = S.cage(rings, closed=True, loop=True, mat=M["chrome"])
    m = K.pos(m)
    ang = np.repeat(a[None, :], len(sec), 0).reshape(-1)
    m.UV = np.column_stack([m.V[:, 2], ang * ro])
    # the outer wall's quads (rings 0..3) take the shield material, but for the seam's column (j = n - 1)
    k = len(sec)
    q = np.arange(3)[:, None] * n + np.arange(n - 1)[None, :]
    m.Qm[q.ravel()] = M["shield"]
    return m


def band(c, r, z, w=0.030, t=0.006, mat=None, sides=32):
    """A clamp band round a vertical cylinder of radius r at height z (a rolled flat strap)."""
    a = 2.0 * math.pi * np.arange(sides) / sides
    path = np.column_stack([c[0] + (r + 0.5 * t) * np.cos(a), c[1] + (r + 0.5 * t) * np.sin(a), np.full(sides, z)])
    return K.ring_sweep(path, S.rrect(t, w, 0.45 * t, 1), up=(0.0, 0.0, 1.0), mat=M["chrome"] if mat is None else mat)


def stack_brackets(body, sx):
    """Bands round the pipe tied to the cab's or the sleeper's nearest corner by a short bar and a foot plate."""
    out = []
    c = np.array([sx * L.STACK_X, L.STACK_Y])
    for z in STACK_BRACKETS:
        out.append(band(c, L.STACK_R, z))
        best = None
        for key, dy in (("cab", -0.55), ("sleeper", 0.55), ("sleeper", 0.0)):
            d = K.unit(np.array([-sx, dy, 0.0]))
            hit = body.try_hit(key, (c[0], c[1], z), d)
            if hit is not None:
                dist = np.linalg.norm(hit[0][:2] - c)
                if dist < 0.40 and (best is None or dist < best[0]):
                    best = (dist, hit[0], hit[1], d)
        if best is None:
            continue
        _, p, n, d = best
        a = np.array([c[0], c[1], z]) + d * (L.STACK_R + 0.004)
        bar = S.sweep(np.array([a, p - n * 0.006]), S.rrect(0.010, 0.032, 0.003, 2), up=(0.0, 0.0, 1.0), mat=M["chrome"])
        out.append(K.pos(bar))
        foot = S.rounded_panel(S.rrect(0.05, 0.07, 0.012, 3), 0.008, r_edge=0.003, n=2, mat=M["chrome"])
        out.append(S.place(foot, p - n * 0.004, n, (0.0, 0.0, 1.0)))
    return out


def stacks(body):
    out = []
    for sx in (1.0, -1.0):
        out.append(stack_pipe(sx))
        out.append(shield(sx))
        c = np.array([sx * L.STACK_X, L.STACK_Y])
        out += [band(c, SHIELD["r"], z, w=0.026, t=0.004) for z in SHIELD["bands"]]
        out += stack_brackets(body, sx)
        # the flange where the elbow meets the rail
        x = sx * (STACK["rail_x"] + 0.012)
        out.append(K.lathe_on(K.disc_profile(L.STACK_R + 0.03, 0.016, 0.004), (x, L.STACK_Y, STACK["bottom_z"]),
                              (sx, 0.0, 0.0), seg=24, mat=M["chassis"]))
    return out


# ===================================================================================================================
# fuel tanks
# ===================================================================================================================
def tank_profile():
    """(r, s) of the tank along its axis from the front head's centre: a dished head, the crimped rim, the shell, the
    rim and dished head at the back."""
    R, Lt = L.TANK_R, L.TANK_Y[1] - L.TANK_Y[0]
    head = [(0.0, -0.026), (0.10, -0.024), (0.19, -0.017), (0.25, -0.010), (0.29, -0.002), (0.312, 0.006), (0.322, 0.013)]
    rim = [(R + 0.000, 0.018), (R + 0.007, 0.022), (R + 0.009, 0.028), (R + 0.007, 0.034), (R + 0.001, 0.038), (R, 0.044)]
    front = head + rim
    back = [(r, Lt - s) for r, s in front[::-1]]
    return np.array(front + [(R, 0.30 * Lt), (R, 0.70 * Lt)] + back)


def tank(sx):
    """One fuel tank: polished, two straps on rubber liners, two cradle brackets from the rail and the filler."""
    R, Lt = L.TANK_R, L.TANK_Y[1] - L.TANK_Y[0]
    x, y0, z = sx * L.TANK_X, L.TANK_Y[0], L.TANK_Z
    body = K.pos(S.place(S.lathe(tank_profile(), seg=48, mat=M["alloy"]), (x, y0, z), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    alloy, chrome, dark = [body], [], []
    n = 40
    a = 2.0 * math.pi * np.arange(n) / n
    for s in TANK_STRAPS:
        y = y0 + s
        liner = np.array([(-0.0015, -0.030), (0.0015, -0.030), (0.0015, 0.030), (-0.0015, 0.030)])
        for r, sec, mat in ((R + 0.0015, liner, M["rubber"]), (R + 0.0045, S.rrect(0.004, 0.050, 0.0018, 1), M["chrome"])):
            path = np.column_stack([x + r * np.cos(a), np.full(n, y), z + r * np.sin(a)])
            (chrome if mat == M["chrome"] else dark).append(K.ring_sweep(path, sec, up=(0.0, 1.0, 0.0), mat=mat))
        # the cradle: a bent flat bar from the rail's web round under the tank
        rc = R + 0.0065 + 0.006
        ang = np.radians(np.linspace(196.0, 300.0, 14))
        arc = np.column_stack([x + sx * rc * np.cos(ang), np.full(len(ang), y), z + rc * np.sin(ang)])
        leg_top = np.array([sx * (L.FRAME_X + 0.007), y, L.FRAME_Z[1] - 0.06])
        leg_bot = np.array([sx * (L.FRAME_X + 0.007), y, arc[0][2] + 0.04])
        path = K.fillet3(np.concatenate([[leg_top, leg_bot], arc]), [0.0, 0.03] + [0.0] * len(arc), n=4)
        cradle = S.sweep(path, S.rrect(0.012, 0.060, 0.005, 1), up=(0.0, 1.0, 0.0), mat=M["chassis"])
        dark.append(K.pos(cradle))
    s, deg = TANK_CAP
    d = np.array([sx * math.sin(math.radians(deg)), 0.0, math.cos(math.radians(deg))])
    p = np.array([x, y0 + s, z]) + d * (R - 0.004)
    chrome.append(K.lathe_on(K.disc_profile(0.042, 0.034, 0.004), p, d, seg=24, mat=M["chrome"]))
    cap = [(0.0, 0.026), (0.040, 0.026), (0.054, 0.030), (0.058, 0.040), (0.056, 0.050), (0.046, 0.056), (0.0, 0.058)]
    chrome.append(K.lathe_on(cap, p, d, seg=24, mat=M["chrome"]))
    return alloy, chrome, dark


# ===================================================================================================================
# step boxes
# ===================================================================================================================
def _face_section(st):
    """The step box's formed face in (u = -x, v = z): its top return, the rolled top edge, the gently crowned face,
    the rolled bottom edge and the bottom return (outer surface, then made a sheet)."""
    xo, (zb, zt) = st["x_out"], st["z"]
    r = 0.020
    top = [(xo - 0.045, zt - 0.003), (xo - 0.030, zt)]
    top += [(xo - r + r * math.cos(t), zt - r + r * math.sin(t)) for t in np.radians(np.linspace(90.0, 0.0, 6))]
    zs = np.linspace(zt - r, zb + r, 7)[1:-1]
    mid, hm = 0.5 * (zt + zb), 0.5 * (zt - zb) - r
    face = [(xo + 0.004 * (1.0 - ((zz - mid) / hm) ** 2), zz) for zz in zs]
    bot = [(xo - r + r * math.cos(t), zb + r + r * math.sin(t)) for t in np.radians(np.linspace(0.0, -90.0, 6))]
    bot += [(xo - 0.030, zb), (xo - 0.045, zb + 0.003)]
    P = np.array(top + face + bot)
    sec = K.sheet_section(P, -st["t"])
    return np.column_stack([-sec[:, 0], sec[:, 1]])


def step_box(sx):
    """One step box under the door: the polished face (a formed sheet), the diamond-plate tread on top, two end plates,
    the bottom plate, and the lower step on two hangers. Each piece is its own formed sheet."""
    st = STEP
    y0, y1 = L.STEP_Y
    xo, (zb, zt) = st["x_out"], st["z"]
    xi = L.FRAME_X + 0.004
    alloy, tread, dark = [], [], []
    path = np.array([[0.0, y0, 0.0], [0.0, y1, 0.0]])
    alloy.append(K.sweep_solid(path, _face_section(st), up=(0.0, 0.0, 1.0), mat=M["alloy"]))
    top = S.rounded_panel(S.rrect(xo - 0.012 - xi, (y1 - y0) - 0.010, 0.02, 4), 0.008, r_edge=0.003, n=2, mat=M["tread_plate"])
    tread.append(top.moved((0.5 * (xo - 0.012 + xi), 0.5 * (y0 + y1), zt - 0.011)))
    end = S.rounded_panel(S.fillet(np.array([[xi, zb + 0.004], [xo - 0.006, zb + 0.004], [xo - 0.006, zt - 0.006],
                                             [xi, zt - 0.006]]), [0.0, 0.03, 0.03, 0.0], n=5, closed=True),
                          0.005, r_edge=0.002, n=2, mat=M["alloy"])
    for yy, dirn in ((y0, -1.0), (y1, 1.0)):
        # the outline is drawn in (x, z): turn it so its face looks along -y / +y
        e = S.Mesh(np.column_stack([end.V[:, 0], yy + dirn * (end.V[:, 2] - 0.0025), end.V[:, 1]]), end.Q, end.T, end.Qm, end.Tm)
        alloy.append(K.pos(e))
    bottom = S.rounded_panel(S.rrect(xo - 0.02 - xi, (y1 - y0) - 0.012, 0.02, 4), 0.005, r_edge=0.002, n=2, mat=M["underbody"])
    dark.append(bottom.moved((0.5 * (xo - 0.02 + xi), 0.5 * (y0 + y1), zb + 0.002)))
    # the lower step: a tread plate with a round alloy nosing, on two L hangers
    xs0, xs1 = xo - 0.20, xo + st["lower_out"]
    ys0, ys1 = y0 + 0.035, y1 - 0.035
    zs = st["lower_z"]
    low = S.rounded_panel(S.rrect(xs1 - xs0, ys1 - ys0, 0.02, 4), 0.008, r_edge=0.003, n=2, mat=M["tread_plate"])
    tread.append(low.moved((0.5 * (xs0 + xs1), 0.5 * (ys0 + ys1), zs - 0.008)))
    alloy.append(K.tube([(xs1 - 0.004, ys0 + 0.004, zs - 0.006), (xs1 - 0.004, ys1 - 0.004, zs - 0.006)], 0.011, sides=12,
                        mat=M["alloy"], ends=("round", "round")))
    for yy in (ys0 + 0.05, ys1 - 0.05):
        p = K.fillet3([(xs0 + 0.03, yy, zb + 0.01), (xs0 + 0.03, yy, zs - 0.018), (xs1 - 0.03, yy, zs - 0.018)],
                      [0.0, 0.03, 0.0], n=6)
        hanger = S.sweep(p, S.rrect(0.040, 0.010, 0.003, 3), up=(0.0, 1.0, 0.0), mat=M["alloy"])
        alloy.append(K.pos(hanger))
    meshes = [alloy, tread, dark]
    if sx < 0:
        meshes = [[m.mirrored_x() for m in grp] for grp in meshes]
    return meshes


# ===================================================================================================================
# deck and lines
# ===================================================================================================================
def deck():
    """The diamond-plate deck on the rails behind the sleeper."""
    y0, y1 = L.DECK_Y
    w = 2 * (L.FRAME_X + 0.03)
    plate = S.rounded_panel(S.rrect(w, y1 - y0, 0.03, 4), 0.006, r_edge=0.0025, n=2, mat=M["tread_plate"])
    return [plate.moved((0.0, 0.5 * (y0 + y1), L.FRAME_Z[1] + 0.001))]


def _sag(a, b, sag, n=24):
    t = np.linspace(0.0, 1.0, n)
    P = a[None, :] + (b - a)[None, :] * t[:, None]
    P[:, 2] -= sag * 4.0 * t * (1.0 - t)
    return P


def glad_hand(p, colour):
    """A glad hand whose seal face is at p looking +Y: the rubber seal, the cast body, its paddle (the line's colour)
    and the hose nipple at its back. Returns meshes."""
    f = np.array([0.0, 1.0, 0.0])
    out = [K.lathe_on(K.disc_profile(0.021, 0.007, 0.002), p - f * 0.007, f, seg=14, mat=M["rubber"])]
    body = [(0.0, -0.030), (0.020, -0.030), (0.028, -0.024), (0.030, -0.012), (0.028, -0.004), (0.024, 0.0), (0.0, 0.0)]
    out.append(K.lathe_on(body, p - f * 0.006, f, seg=14, mat=M["alloy"]))
    paddle = S.rounded_panel(S.stadium(0.030, 0.080, 4), 0.010, r_edge=0.004, n=2, mat=colour)
    out.append(S.place(paddle, p - f * 0.026 + np.array([0.0, 0.0, -0.036]), f, (0.0, 0.0, 1.0)))
    out.append(K.tube([p - f * 0.034, p - f * 0.060], 0.0075, sides=10, mat=M["alloy"], ends=("flat", "flat")))
    return out


def lines(body):
    """The bulkhead on the sleeper's back, the hose tender on the deck and the three lines: red (emergency) and blue
    (service) coiled air lines to the glad hands, the black electric coil to the 7-way plug in the van's socket."""
    ln = LINES
    zb = ln["z_bulk"]
    hit = body.try_hit("sleeper", (0.0, L.SLEEPER_Y[1] + 1.0, zb), (0.0, -1.0, 0.0))
    yb = hit[0][1] if hit is not None else L.SLEEPER_Y[1]
    f = np.array([0.0, 1.0, 0.0])
    dark, chrome, red, blue, alloy = [], [], [], [], []
    plate = S.rounded_panel(S.rrect(0.24, 0.10, 0.02, 4), 0.022, r_edge=0.006, n=3, mat=M["chassis"])
    dark.append(S.place(plate, (0.0, yb - 0.004, zb), f, (0.0, 0.0, 1.0)))
    # the tender: a chrome pole on a spring on the deck, a cross bar the lines lie over
    tx, ty = ln["tender"]
    zd = L.FRAME_Z[1] + 0.007
    chrome.append(K.lathe_on(K.disc_profile(0.045, 0.012, 0.003), (tx, ty, zd - 0.002), (0.0, 0.0, 1.0), seg=18,
                             mat=M["chrome"]))
    sp = K.coil(np.array([(tx, ty, zd + 0.010), (tx, ty, zd + 0.15)]), 0.022, 0.020, per_turn=8)
    chrome.append(K.tube(np.concatenate([[sp[0] - np.array([0.0, 0.0, 0.006])], sp]), 0.0035, sides=6, mat=M["chrome"],
                         ends=("round", "round")))
    zh = ln["hold_z"]
    chrome.append(K.tube([(tx, ty, zd + 0.005), (tx, ty, zh - 0.012)], 0.0125, sides=12, mat=M["chrome"], ends=("flat", "round")))
    chrome.append(K.tube(K.fillet3([(tx - 0.12, ty, zh + 0.03), (tx - 0.12, ty, zh), (tx + 0.12, ty, zh), (tx + 0.12, ty, zh + 0.03)],
                                   [0.0, 0.02, 0.02, 0.0], n=5), 0.0085, sides=10, mat=M["chrome"], ends=("round", "round")))
    y_gh = L.TRAILER_Y[0] - 0.06
    ends = {0.065: ("red", np.array([GLAD_X, y_gh, COUPLE_Z])), -0.065: ("blue", np.array([-GLAD_X, y_gh, COUPLE_Z])),
            0.0: ("electric", np.array([0.0, L.TRAILER_Y[0] - 0.07, COUPLE_Z]))}
    for xb in ln["xs"]:
        kind, tip = ends[xb]
        mat = {"red": M["hose_red"], "blue": M["hose_blue"], "electric": M["rubber"]}[kind]
        chrome.append(K.lathe_on(K.disc_profile(0.013, 0.030, 0.003), (xb, yb - 0.004, zb), f, seg=14, mat=M["chrome"]))
        start = np.array([xb, yb + 0.022, zb])
        over = np.array([xb, ty, zh + 0.0085 + ln["hose_r"] + 0.002])
        back = tip - f * (0.060 if kind != "electric" else 0.075)
        lead_end = back - f * 0.10
        lead = K.fillet3([start, start + f * 0.06, over + np.array([0.0, -0.06, 0.0]), over, over + f * 0.05],
                         [0.0, 0.04, 0.05, 0.0, 0.0], n=6)
        target = (red if kind == "red" else blue if kind == "blue" else dark)
        target.append(K.tube(lead, ln["hose_r"] + 0.001, sides=8, mat=mat, ends=("flat", "round")))
        axis = _sag(over + f * 0.05, lead_end, 0.10)
        cl = K.coil(axis, ln["coil_r"], ln["pitch"], per_turn=7, phase=1.3 * xb)
        cl = np.concatenate([[over + f * 0.03], cl, [lead_end + f * 0.02, back]])
        target.append(K.tube(cl, ln["hose_r"], sides=6, mat=mat, ends=("round", "flat")))
        if kind == "electric":
            plug = [(0.0, -0.080), (0.016, -0.080), (0.024, -0.072), (0.026, -0.050), (0.026, -0.012), (0.024, -0.004),
                    (0.018, 0.0), (0.0, 0.0)]
            dark.append(K.lathe_on(plug, tip, f, seg=18, mat=M["rubber"]))
            chrome.append(K.lathe_on(K.disc_profile(0.0275, 0.010, 0.002), tip - f * 0.022, f, seg=18, mat=M["chrome"]))
        else:
            alloy += glad_hand(tip, mat)
    return dark + chrome + red + blue + alloy


# ===================================================================================================================
# tail lamps
# ===================================================================================================================
def tail():
    """The lamp bar across the frame's end (satin black, on two stubs from the rails) and a round red tail lamp in a
    chrome ring at each end. Returns (bar parts, lamp parts)."""
    t = TAIL
    y, z, h = t["y"], t["z"], t["half"]
    bar = [K.pos(S.sweep(np.array([(-h, y, z), (h, y, z)]), S.rrect(0.028, 0.065, 0.010, 4), up=(0.0, 0.0, 1.0),
                         mat=M["chassis"]))]
    for sx in (1.0, -1.0):
        bar.append(K.tube([(sx * (L.FRAME_X - 0.03), L.FRAME_Y[1] - 0.03, z), (sx * (L.FRAME_X - 0.03), y, z)], 0.018,
                          sides=12, mat=M["chassis"], ends=("flat", "flat")))
    lamps = []
    f = np.array([0.0, 1.0, 0.0])
    for x in t["lamps"]:
        for sx in (1.0, -1.0):
            p = np.array([sx * x, y + 0.014, z])
            lamps.append(K.lathe_on([(0.0, 0.0), (0.058, 0.0), (0.060, 0.008), (0.057, 0.016), (0.050, 0.018), (0.0, 0.018)],
                                    p - f * 0.004, f, seg=28, mat=M["chrome"]))
            lamps.append(K.lathe_on([(0.0, 0.0), (0.049, 0.0), (0.049, 0.010), (0.040, 0.020), (0.022, 0.025), (0.0, 0.026)],
                                    p, f, seg=28, mat=M["lamp_red"]))
    return bar, lamps


# ===================================================================================================================
def parts(body):
    """{group: [Mesh]} of the hardware along the frame."""
    tanks, steps = [], []
    for sx in (1.0, -1.0):
        a, b, c = tank(sx)
        tanks += a + b + c
        for grp in step_box(sx):
            steps += grp
    bar, tail_lamps = tail()
    return {"ext_stacks": stacks(body), "ext_tanks": tanks, "ext_steps": steps, "ext_deck": deck() + lines(body) + bar,
            "ext_lamps": tail_lamps}
