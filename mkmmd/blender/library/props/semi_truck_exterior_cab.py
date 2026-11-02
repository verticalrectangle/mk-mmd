"""The truck's hardware round the cab (bpy-free): the chrome air cleaners on the cowl's sides with their elbow ducts,
the West Coast mirrors (a chrome loop of tube from the door's front edge holding the head, a convex spot mirror below
it, a CB antenna on the upper arm), three air horns on the roof, the five amber clearance lamps on the visor and five
on the sleeper's roof, the amber markers along the cab's and the sleeper's rockers, the wipers parked at the foot of
the windshield, the grab handles and the door handles. Every function returns lists of meshes in the truck's frame;
`body` is a kit.Body (TruckBody's shells or the stand-in)."""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_exterior_kit as K
from . import semi_truck_layout as L

M = L.M

COWL_SIDE_X = 0.70                                  # the cowl's side face (TruckBody's cowl: |x| <= 0.70)
BANDS = (0.19, 0.37, 0.55)                          # the air cleaners' raised bands, above their bottoms
LID = 0.11                                          # the lid's skirt: this far down from the top
HEAD = dict(w=0.21, h=0.46, depth=0.065, toe=8.0)   # West Coast head (L.MIRROR its centre); toe: turned in, degrees
ROD_GAP = 0.022                                     # from the head's inner edge to the rod
ARM_ROOTS = ((-4.86, 2.60), (-5.04, 1.82))          # (y, z) where the upper and lower arms meet the cab
SPOT = dict(c=(1.385, -5.115, 1.835), r=0.078, aim=(0.45, 1.0, -0.30))
WHIP = dict(at=0.62, spring=0.08, length=1.05, r=(0.0055, 0.0028))
HORNS = dict(x=(-0.30, -0.375, -0.45), y=-4.72, length=(0.36, 0.46, 0.56), bell=(0.042, 0.048, 0.055), lift=0.075)
VISOR_LAMPS = dict(x=(-0.60, -0.30, 0.0, 0.30, 0.60), y=-5.18)
SLEEPER_LAMPS = dict(x=(-0.60, -0.30, 0.0, 0.30, 0.60), y=L.SLEEPER_Y[0] + 0.30)
ROCKER_LAMPS = dict(z=1.165, cab=(-4.95, -4.15), sleeper=(-3.48, -2.92, -2.36))
WIPER = dict(x=(0.09, 0.65), up=0.065, pivot=(0.68, -0.07), lift=0.012)
GRAB = dict(y=-5.022, z=(1.36, 1.72), off=0.052, r=0.0155)
DOOR_HANDLE = dict(y=-4.21, z=1.34)


# ===================================================================================================================
# air cleaners
# ===================================================================================================================
def air_cleaner_profile():
    """(r, z) of the canister from the bottom's centre up the wall (three raised bands) to the lid's skirt and its
    domed top (z absolute)."""
    x, y, z0, z1, R = L.AIR_CLEANER
    P = [(0.0, z0), (0.09, z0 + 0.002), (0.15, z0 + 0.006), (R - 0.013, z0 + 0.013), (R - 0.003, z0 + 0.026),
         (R, z0 + 0.042)]
    for h in BANDS:
        zb = z0 + h
        P += [(R, zb - 0.018), (R + 0.0045, zb - 0.009), (R + 0.006, zb), (R + 0.0045, zb + 0.009), (R, zb + 0.018)]
    zl = z1 - LID
    P += [(R, zl - 0.010), (R + 0.004, zl - 0.006), (R + 0.010, zl - 0.004), (R + 0.0135, zl + 0.001),
          (R + 0.014, zl + 0.012), (R + 0.014, z1 - 0.040), (R + 0.011, z1 - 0.020), (R + 0.002, z1 - 0.008),
          (R - 0.030, z1 - 0.002), (R - 0.09, z1), (0.0, z1 + 0.004)]
    return np.array(P)


def air_cleaner(body, sx):
    """One canister (chrome) with an elbow duct (rubber, a chrome clamp) from its bottom into the cowl and two
    short brackets to the cowl's side face."""
    x, y, z0, z1, R = L.AIR_CLEANER
    c = np.array([sx * x, y, 0.0])
    can = K.pos(S.lathe(air_cleaner_profile(), seg=32, mat=M["chrome"]).moved(c))
    chrome, rubber = [can], []
    # the outlet: down out of the bottom, turning inboard into the cowl
    zd = z0 - 0.105
    xin = sx * (COWL_SIDE_X - 0.10)
    path = K.fillet3([(sx * x, y, z0 + 0.06), (sx * x, y, zd), (xin, y, zd)], [0.0, 0.085, 0.0], n=10)
    rubber.append(K.tube(path, 0.064, sides=16, mat=M["rubber"], ends=("flat", "flat")))
    chrome.append(K.tube([(sx * x, y, z0 - 0.012), (sx * x, y, z0 - 0.050)], 0.070, sides=20, mat=M["chrome"],
                         ends=("flat", "flat")))
    for zz in (z0 + 0.25, z0 + 0.62):
        hit = body.try_hit("cab", (sx * x, y, zz), (-sx, 0.0, 0.0))
        xe = hit[0][0] - sx * 0.006 if hit is not None and abs(hit[0][0]) > COWL_SIDE_X - 0.08 else sx * (COWL_SIDE_X - 0.006)
        chrome.append(K.tube([(sx * (x - R + 0.02), y, zz), (xe, y, zz)], 0.016, sides=12, mat=M["chrome"],
                             ends=("flat", "flat")))
    return chrome, rubber


# ===================================================================================================================
# mirrors
# ===================================================================================================================
def _root(body, y, z, sx):
    """The cab's side surface at (y, z): (point, outward normal)."""
    hit = body.try_hit("cab", (sx * 2.0, y, z), (-sx, 0.0, 0.0))
    if hit is None:
        return np.array([sx * L.CAB_HALF_W, y, z]), np.array([sx, 0.0, 0.0])
    return hit


def mirror(body, sx):
    """The West Coast mirror of one side: the head (a chrome pillow, the glass on its back, toed in toward the cab), the
    loop of 25 mm tube from the A-pillar out, down past the head and back to the door, two clamps, chrome feet, the
    convex spot mirror below and the CB antenna on the upper arm. Returns (chrome, glass, whip) meshes."""
    hd = HEAD
    mx, my, mz = L.MIRROR
    c = np.array([sx * mx, my, mz])
    toe = math.radians(hd["toe"])
    face = np.array([-sx * math.sin(toe), math.cos(toe), 0.0])            # the glass looks back, a little inboard
    up = np.array([0.0, 0.0, 1.0])
    shell = S.rounded_panel(S.rrect(hd["w"], hd["h"], 0.045, 6), hd["depth"], r_edge=0.022, n=4, dome=0.012,
                            r_back=0.006, mat=M["chrome"])
    chrome = [S.place(shell, c + face * 0.5 * hd["depth"], -face, up)]
    glass = S.rounded_panel(S.rrect(hd["w"] - 0.022, hd["h"] - 0.022, 0.036, 6), 0.003, r_edge=0.0012, n=2,
                            mat=M["mirror_glass"])
    glass_m = [S.place(glass, c + face * (0.5 * hd["depth"] - 0.001), face, up)]
    # the loop: A-pillar root -> out -> down the rod beside the head -> in to the door
    side = np.array([sx * math.cos(toe), math.sin(toe), 0.0])             # across the head, outboard
    rod_x = c - side * (0.5 * hd["w"] + ROD_GAP)
    top = rod_x + up * (0.5 * hd["h"] + 0.07)
    bot = rod_x - up * (0.5 * hd["h"] + 0.05)
    (yu, zu), (yl, zl) = ARM_ROOTS
    pu, nu = _root(body, yu, zu, sx)
    pl, nl = _root(body, yl, zl, sx)
    loop = K.fillet3([pu - nu * 0.012, pu + nu * 0.06, top, bot, pl + nl * 0.06, pl - nl * 0.012],
                     [0.0, 0.05, 0.07, 0.06, 0.05, 0.0], n=6)
    chrome.append(K.tube(loop, 0.0125, sides=12, mat=M["chrome"], ends=("flat", "flat")))
    for p, n in ((pu, nu), (pl, nl)):
        chrome.append(K.lathe_on(K.disc_profile(0.030, 0.014, 0.004), p - n * 0.006, n, seg=16, mat=M["chrome"]))
    for dz in (-0.16, 0.16):                                                # clamps from the rod to the head
        a = rod_x + up * dz
        b = c - side * (0.5 * hd["w"] - 0.012) + up * dz
        chrome.append(K.tube([a, b], 0.009, sides=10, mat=M["chrome"], ends=("round", "round")))
    # the spot mirror on a stalk off the rod's foot
    sp = SPOT
    sc = np.array([sx * sp["c"][0], sp["c"][1], sp["c"][2]])
    aim = K.unit(np.array([sx * sp["aim"][0], sp["aim"][1], sp["aim"][2]]))
    r = sp["r"]
    cup = [(0.0, -0.040), (0.45 * r, -0.038), (0.82 * r, -0.028), (r, -0.012), (r + 0.002, -0.002), (r - 0.002, 0.003),
           (r - 0.006, 0.002), (0.0, 0.002)]
    chrome.append(K.lathe_on(cup, sc, aim, seg=24, mat=M["chrome"]))
    glass_m.append(K.lathe_on([(0.0, 0.0), (r - 0.006, 0.0), (r - 0.007, 0.004), (0.6 * r, 0.011), (0.0, 0.014)],
                              sc, aim, seg=24, mat=M["mirror_glass"]))
    s0 = rod_x - up * (0.5 * hd["h"] - 0.02)
    chrome.append(K.tube(K.fillet3([s0, s0 - up * 0.05 + side * 0.02, sc - aim * 0.034], [0.0, 0.03, 0.0], n=6), 0.008,
                         sides=10, mat=M["chrome"], ends=("round", "flat")))
    # the CB antenna on the upper arm: a clamp, a chrome spring, a black whip
    s = K.arclength(loop)
    s_top = s[int(np.argmin(np.linalg.norm(loop - top, axis=1)))]
    base = np.array([np.interp(WHIP["at"] * s_top, s, loop[:, k]) for k in range(3)])
    chrome.append(K.lathe_on(K.disc_profile(0.019, 0.032, 0.004), base - up * 0.016, up, seg=14, mat=M["chrome"]))
    sp_path = np.array([base + up * 0.016, base + up * (0.016 + WHIP["spring"])])
    spring = K.coil(sp_path, 0.010, 0.016, per_turn=6)
    chrome.append(K.tube(np.concatenate([[base + up * 0.012], spring, [spring[-1] + up * 0.008]]), 0.0026, sides=6,
                         mat=M["chrome"], ends=("round", "round")))
    w0 = base + up * (0.016 + WHIP["spring"])
    whip_path = np.array([w0 + up * t for t in np.linspace(0.0, WHIP["length"], 6)])
    rad = np.linspace(WHIP["r"][0], WHIP["r"][1], len(whip_path))
    whip = [K.tube(whip_path, rad, sides=8, mat=M["rubber"], ends=("flat", "round"))]
    return chrome, glass_m, whip


# ===================================================================================================================
# roof: horns and clearance lamps
# ===================================================================================================================
def horn_profile(length, bell, r0=0.011):
    """(r, z) of a trumpet from its closed back (z = 0) along the throat flaring to the bell at z = length, round its
    rolled lip and back down the inside of the bell to a dark throat."""
    s = np.linspace(0.0, 1.0, 12)
    r = r0 + (bell - r0) * s ** 3.2
    out = [(0.0, -0.008), (0.6 * r0, -0.007), (r0, -0.002)]
    out += [(float(rr), float(length * ss)) for rr, ss in zip(r, s)][1:]
    out += [(bell + 0.0025, length + 0.002), (bell + 0.001, length + 0.005), (bell - 0.003, length + 0.004)]
    inner = [(float(max(rr - 0.003, 0.004)), float(length * ss)) for rr, ss in zip(r, s) if ss > 0.55][::-1]
    out += inner + [(0.0, length * 0.55)]
    return np.array(out)


def horns(body):
    """Three chrome trumpets of 0.36, 0.46 and 0.56 m side by side on the roof's passenger side, their bells forward,
    on a valve block and two satin black saddles."""
    h = HORNS
    xs = np.array(h["x"])
    y_bell = h["y"]
    xm = float(xs.mean())
    hit = body.try_hit("cab", (xm, y_bell + 0.25, 4.0), (0.0, 0.0, -1.0))
    roof = hit[0][2] if hit is not None else L.CAB_ROOF_Z
    chrome, dark = [], []
    for x, ln, bell in zip(xs, h["length"], h["bell"]):
        zc = roof + h["lift"] + 0.5 * (bell - 0.042)
        m = S.lathe(horn_profile(ln, bell), seg=18, mat=M["chrome"])
        m.assign(M["seam"], lambda cc, nn, ln=ln: (cc[:, 2] < ln * 0.80) & (np.hypot(cc[:, 0], cc[:, 1]) < 0.05)
                 & (cc[:, 2] > ln * 0.5) & (nn[:, 2] > 0.2))
        chrome.append(K.pos(S.place(m, (x, y_bell + ln, zc), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))))
    y_back = y_bell + max(h["length"]) + 0.02
    block = K.tube([(xs.min() - 0.03, y_back - 0.30, roof + h["lift"]), (xs.max() + 0.03, y_back - 0.30, roof + h["lift"])],
                   0.024, sides=14, mat=M["chrome"], ends=("round", "round"), up=(0.0, 1.0, 0.0))
    chrome.append(block)
    for yy in (y_bell + 0.22, y_back - 0.08):
        hit = body.try_hit("cab", (xm, yy, 4.0), (0.0, 0.0, -1.0))
        z = hit[0][2] if hit is not None else L.CAB_ROOF_Z
        saddle = S.rounded_panel(S.rrect(0.24, 0.05, 0.02, 4), h["lift"] - 0.02 + roof - z + 0.012, r_edge=0.006, n=2,
                                 mat=M["chassis"])
        dark.append(S.place(saddle, (xm, yy, z - 0.012), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    # the air line from the valve block down into the roof
    xa = xs.max() + 0.04
    dark.append(K.tube(K.fillet3([(xa - 0.02, y_back - 0.30, roof + h["lift"]), (xa + 0.03, y_back - 0.30, roof + h["lift"]),
                                  (xa + 0.05, y_back - 0.20, roof - 0.02)], [0.0, 0.03, 0.0], n=6), 0.006, sides=8,
                       mat=M["rubber"], ends=("round", "flat")))
    return chrome, dark


def torpedo_lamp(p, n, fwd=(0.0, -1.0, 0.0), post=0.035, length=0.085, r=0.022):
    """An amber torpedo lamp on a chrome post standing on a surface at p (normal n): the post, a chrome boot and the
    amber lens pointing `fwd`. Returns (chrome, lens)."""
    n, f = K.unit(n), K.unit(fwd)
    c = p + n * (post + r)
    chrome = [K.tube([p - n * 0.008, c - n * 0.6 * r], 0.009, sides=8, mat=M["chrome"], ends=("flat", "round"))]
    boot = [(0.0, -0.030), (0.55 * r, -0.029), (0.9 * r, -0.022), (r + 0.002, -0.010), (r + 0.002, 0.0), (0.0, 0.0)]
    chrome.append(K.lathe_on(boot, c, f, seg=14, mat=M["chrome"]))
    zs = np.linspace(0.0, 1.0, 6)
    lens = [(0.0, -0.003)] + [(float(r * math.sqrt(max(1.0 - t * t, 0.0)) * 0.98 + 0.0005), float(-0.002 + t * (length - 0.03)))
                              for t in zs[:-1]] + [(0.0, length - 0.03)]
    return chrome, [K.lathe_on(lens, c, f, seg=14, mat=M["lamp_amber"])]


def roof_lamps(body):
    """The five amber lamps on the visor and five on the sleeper roof's front, pointing forward."""
    chrome, lens = [], []
    for key, spec in (("cab", VISOR_LAMPS), ("sleeper", SLEEPER_LAMPS)):
        for x in spec["x"]:
            hit = body.try_hit(key, (x, spec["y"], 5.0), (0.0, 0.0, -1.0))
            if hit is None:
                continue
            p, n = hit
            a, b = torpedo_lamp(p, (0.0, 0.0, 1.0))
            chrome += a
            lens += b
    return chrome, lens


def rocker_lamps(body):
    """Small amber domes in chrome rings along the cab's and the sleeper's rockers, both sides."""
    chrome, lens = [], []
    z = ROCKER_LAMPS["z"]
    for key, ys in (("cab", ROCKER_LAMPS["cab"]), ("sleeper", ROCKER_LAMPS["sleeper"])):
        for sx in (1.0, -1.0):
            for y in ys:
                hit = body.try_hit(key, (sx * 2.0, y, z), (-sx, 0.0, 0.0))
                if hit is None:
                    continue
                p, n = hit
                chrome.append(K.lathe_on([(0.0, 0.0), (0.026, 0.0), (0.027, 0.006), (0.024, 0.011), (0.0, 0.011)],
                                         p - n * 0.004, n, seg=12, mat=M["chrome"]))
                lens.append(K.lathe_on([(0.0, 0.0), (0.019, 0.0), (0.018, 0.008), (0.012, 0.016), (0.0, 0.019)],
                                       p + n * 0.004, n, seg=12, mat=M["lamp_amber"]))
    return chrome, lens


# ===================================================================================================================
# wipers and handles
# ===================================================================================================================
def _glass_frame():
    """The windshield's plane: a point on its foot line, the direction up the glass and its outward normal."""
    (y0, z0), (y1, z1) = L.WS_BASE, L.WS_TOP
    d = K.unit(np.array([0.0, y1 - y0, z1 - z0]))
    n = np.array([0.0, -d[2], d[1]])
    return np.array([0.0, y0, z0]), d, n


def wipers(body):
    """Two wipers parked along the foot of the panes: a rubber blade on a black frame, the arm from a chrome pivot cap
    on the cowl."""
    o, d, n = _glass_frame()
    w = WIPER
    dark, chrome = [], []
    for sx in (1.0, -1.0):
        x0, x1 = w["x"]
        a = o + np.array([sx * x0, 0.0, 0.0]) + d * w["up"] + n * 0.006
        b = o + np.array([sx * x1, 0.0, 0.0]) + d * (w["up"] + 0.03) + n * 0.006
        blade = np.array([a + (b - a) * t for t in np.linspace(0.0, 1.0, 8)])
        dark.append(K.tube(blade, 0.0042, sides=8, mat=M["rubber"], ends=("round", "round"), up=n))
        frame = blade + n * 0.010
        dark.append(K.tube(frame, 0.0055, sides=8, mat=M["chassis"], ends=("round", "round"), up=n))
        px, pd = w["pivot"]
        pv = o + np.array([sx * px, 0.0, 0.0]) + d * pd
        hit = body.try_hit("cab", pv + np.array([0.0, 0.0, 1.0]), (0.0, 0.0, -1.0))
        pivot = hit[0] if hit is not None else pv
        mid = 0.5 * (a + b) + n * 0.022
        arm = K.fillet3([pivot + np.array([0.0, 0.0, 0.02]), pivot + np.array([0.0, 0.0, 0.035]) + d * 0.02, mid],
                        [0.0, 0.03, 0.0], n=6)
        dark.append(K.tube(arm, 0.006, sides=8, mat=M["chassis"], ends=("round", "round")))
        chrome.append(K.lathe_on(K.disc_profile(0.020, 0.030, 0.006), pivot - np.array([0.0, 0.0, 0.006]), (0.0, 0.0, 1.0),
                                 seg=18, mat=M["chrome"]))
    return chrome, dark


def grab_handles(body):
    """A vertical chrome grab handle on each door just behind its front edge, on two stand-offs."""
    g = GRAB
    out = []
    for sx in (1.0, -1.0):
        (p0, n0), (p1, n1) = _root(body, g["y"], g["z"][0], sx), _root(body, g["y"], g["z"][1], sx)
        path = K.fillet3([p0 - n0 * 0.010, p0 + n0 * g["off"], p1 + n1 * g["off"], p1 - n1 * 0.010], [0.0, 0.035, 0.035, 0.0],
                         n=8)
        out.append(K.tube(path, g["r"], sides=14, mat=M["chrome"], ends=("flat", "flat")))
        for p, n in ((p0, n0), (p1, n1)):
            out.append(K.lathe_on(K.disc_profile(0.027, 0.012, 0.004), p - n * 0.005, n, seg=18, mat=M["chrome"]))
    return out


def door_handles(body):
    """A chrome pull on each door: an escutcheon plate and a bow handle over it."""
    out = []
    for sx in (1.0, -1.0):
        p, n = _root(body, DOOR_HANDLE["y"], DOOR_HANDLE["z"], sx)
        fwd = np.array([0.0, -1.0, 0.0])
        plate = S.rounded_panel(S.rrect(0.22, 0.060, 0.022, 5), 0.007, r_edge=0.003, n=2, mat=M["chrome"])
        out.append(S.place(plate, p - n * 0.003, n, (0.0, 0.0, 1.0)))
        a, b = p + fwd * 0.085 + n * 0.002, p - fwd * 0.085 + n * 0.002
        path = K.fillet3([a, a + n * 0.030 - fwd * 0.02, b + n * 0.030 + fwd * 0.02, b], [0.0, 0.02, 0.02, 0.0], n=6)
        out.append(K.tube(path, 0.0085, sides=10, mat=M["chrome"], ends=("flat", "flat")))
    return out


# ===================================================================================================================
def parts(body):
    """{group: [Mesh]} of the hardware round the cab."""
    air, lamps, mirrors, roof, trim = [], [], [], [], []
    for sx in (1.0, -1.0):
        c, r = air_cleaner(body, sx)
        air += c + r
        a, b, w = mirror(body, sx)
        mirrors += a + b + w
    a, b = horns(body)
    roof += a + b
    a, b = roof_lamps(body)
    lamps += a + b
    a, b = rocker_lamps(body)
    lamps += a + b
    a, b = wipers(body)
    trim += a + b + grab_handles(body) + door_handles(body)
    return {"ext_air": air, "ext_lamps": lamps, "ext_mirrors": mirrors, "ext_horns": roof, "ext_trim": trim}
