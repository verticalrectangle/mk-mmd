"""The truck's nose hardware (bpy-free): the chrome wraparound bumper on its frame brackets with two tow hooks, the
headlamp pods on the fender fronts (a chrome bucket and bezel round two rectangular lenses) with the amber turn signals
at their outer ends, the amber markers on the fenders' sides and the round convex mirrors on stalks on the fender
crowns. Each function returns {group: [Mesh]} in the truck's frame; `body` is a kit.Body (TruckBody's shells or the
stand-in)."""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_exterior_kit as K
from . import semi_truck_layout as L

M = L.M

# the bumper: a formed chrome sheet (face, rolled top and bottom edges, short returns) swept along its plan line
BUMPER = dict(t=0.004, roll=0.042, ret=(0.026, 0.022), crown=0.008, wrap_deg=35.0, corner_r=0.13, end_taper=0.16,
              end_scale=0.30)
BRACKET_X = (L.FRAME_X + 0.001, L.FRAME_X + 0.015)  # the bumper brackets' plates on the rails' outer webs
HOOK_X = 0.33                                       # tow hooks under the bumper, inboard of the brackets
# the headlamp: a chrome bezel (w x h) round two 4 x 6 in lenses on a short bucket that sinks into the fender's face
LAMP = dict(w=0.400, h=0.150, r=0.034, open_w=0.364, open_h=0.118, rim=0.020, pocket=0.013, lens=(0.165, 0.104),
            lens_gap=0.014, sink=0.012)
SIGNAL = dict(dx=0.262, r=0.052, depth=0.05)        # the amber turn signal's cup: centre this far outboard of the lamp's
FENDER_MARKER = (-6.83, 1.12)                       # (y, z) of the amber marker on each fender's side
HOOD_MIRROR = dict(foot=(1.02, -6.55), head=(1.17, -6.47), rise=0.27, r=0.068)


# ===================================================================================================================
# bumper
# ===================================================================================================================
def bumper_plan(n_arc=10):
    """The plan line of the bumper's outer face from the -X end round the face to the +X end: (k, 2) points (x, y)
    and the arc length at each. The face is straight at y = Y_BUMPER + crown (the crown's peak at Y_BUMPER); the ends
    bend back by `wrap_deg` round a `corner_r` radius and end inside x = +-BUMPER_HALF_W, the tapered section's back
    corner at BUMPER_Y[1] (the back of the end wraps)."""
    b = BUMPER
    yf = L.Y_BUMPER + b["crown"]
    a = math.radians(b["wrap_deg"])
    reach = b["end_scale"] * (b["roll"] + b["ret"][0])            # how far the tapered section reaches behind its line
    depth = L.BUMPER_Y[1] - 0.001 - reach * math.cos(a) - yf
    xe = L.BUMPER_HALF_W - 0.004
    xc = xe - depth / math.tan(a)
    pts = np.array([[-xe, yf + depth], [-xc, yf], [xc, yf], [xe, yf + depth]])
    plan = S.fillet(pts, [0.0, b["corner_r"], b["corner_r"], 0.0], n=n_arc)
    plan = K.densify(np.column_stack([plan, np.zeros(len(plan))]), 0.30)[:, :2]
    return plan, K.arclength(plan)


def bumper_section():
    """The bumper's section in (u back, v up): the outer face from the top return's tip over the top roll, down the
    crowned face, round the bottom roll to the bottom return's tip, made a sheet of thickness t."""
    b = BUMPER
    z0, z1 = L.BUMPER_Z
    r, (rt, rb), c = b["roll"], b["ret"], b["crown"]
    top = [(r + rt, z1 - 0.010), (r + 0.5 * rt, z1 - 0.003)]
    top += [(r + r * math.cos(t), z1 - r + r * math.sin(t)) for t in np.radians(np.linspace(90.0, 180.0, 7))]
    vs = np.linspace(z1 - r, z0 + r, 9)[1:-1]
    hm, mid = 0.5 * (z1 - z0) - r, 0.5 * (z0 + z1)
    face = [(-c * (1.0 - ((v - mid) / hm) ** 2), v) for v in vs]
    bot = [(r + r * math.cos(t), z0 + r + r * math.sin(t)) for t in np.radians(np.linspace(180.0, 270.0, 7))]
    bot += [(r + 0.5 * rb, z0 + 0.003), (r + rb, z0 + 0.009)]
    return K.sheet_section(np.array(top + face + bot), b["t"])


def bumper():
    """The chrome bumper: its sheet section swept along the plan line (z = 0: the section's v is the height), the
    returns tapering into the cut ends."""
    b = BUMPER
    plan, s = bumper_plan()
    path = np.column_stack([plan, np.zeros(len(plan))])
    # extra rings near the ends for the taper
    total = s[-1]
    t_end = np.concatenate([np.linspace(0.0, b["end_taper"], 5), np.linspace(total - b["end_taper"], total, 5)])
    s2 = np.unique(np.round(np.concatenate([s, t_end]), 9))
    path = np.stack([np.interp(s2, s, path[:, k]) for k in range(3)], 1)
    k = np.clip(np.minimum(s2, total - s2) / b["end_taper"], 0.0, 1.0)
    k = k * k * (3.0 - 2.0 * k)
    scale = np.stack([b["end_scale"] + (1.0 - b["end_scale"]) * k, np.ones(len(k))], 1)
    return K.sweep_solid(path, bumper_section(), scale=scale, up=K.UP, mat=M["chrome"])


def bumper_brackets():
    """Two plates from the bumper's back up to the rails' front ends (satin black), mostly hidden behind the bumper."""
    yb = L.Y_BUMPER + BUMPER["roll"] + 0.004
    y_rail = L.FRAME_Y[0] + 0.16
    bottom = [(yb, 0.50), (yb + 0.08, 0.53), (y_rail, L.FRAME_Z[0] + 0.03)]
    top = [(yb, 0.80), (yb + 0.10, 0.84), (y_rail, L.FRAME_Z[1] - 0.04)]
    out = []
    for sx in (1.0, -1.0):
        m = S.extrude_profile(bottom, top, BRACKET_X[0], BRACKET_X[1], r0=0.004, r1=0.004, k0=2, k1=2,
                              flush=(False, False), mat=M["chassis"])
        out.append(K.pos(m) if sx > 0 else m.mirrored_x())
    return out


def tow_hooks():
    """Two chrome J hooks of 32 mm round bar on short plates under the bumper, their mouths up."""
    out = []
    y0 = L.Y_BUMPER + 0.11
    for sx in (1.0, -1.0):
        x = sx * HOOK_X
        path = K.fillet3([(x, y0 + 0.02, 0.47), (x, y0 - 0.02, 0.31), (x, y0 - 0.11, 0.30), (x, y0 - 0.13, 0.37)],
                         [0.0, 0.05, 0.04, 0.0], n=8)
        out.append(K.tube(path, 0.016, sides=14, mat=M["chrome"], ends=("round", "round"), up=(1.0, 0.0, 0.0)))
        plate = S.rounded_panel(S.rrect(0.09, 0.12, 0.02, 4), 0.012, r_edge=0.004, n=2, mat=M["chassis"])
        out.append(S.place(plate, (x, y0 + 0.03, 0.47), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
    hanger = []
    for sx in (1.0, -1.0):
        x = sx * HOOK_X
        hanger.append(K.tube([(x, y0 + 0.045, 0.47), (x, y0 + 0.045, 0.62)], 0.022, sides=12, mat=M["chassis"],
                             ends=("flat", "flat")))
    return out + hanger


# ===================================================================================================================
# headlamps
# ===================================================================================================================
def _lamp_frame(body, sx):
    """The +-X headlamp's frame: the lens centre (on L.HEADLAMP), the facing (-Y) and where the fender's face is
    behind it."""
    x, y, z = L.HEADLAMP
    c = np.array([sx * x, y, z])
    hit = body.try_hit("hood", (sx * x, y - 0.4, z), (0.0, 1.0, 0.0))
    face_y = hit[0][1] if hit is not None else y + 0.072
    return c, face_y


def headlamp(body, sx):
    """One headlamp pod: the chrome bucket from the fender's face out to the bezel, the bezel's pocket, a centre post
    and the two lenses (lens_head), their fronts in the plane of L.HEADLAMP."""
    lp = LAMP
    c, face_y = _lamp_frame(body, sx)
    back = c[1] + lp["rim"]                                 # the bezel's back plane (its rim is the lens plane)
    # the bucket: rounded-rectangle rings from the bezel's back to inside the fender, tapering a little
    rings = []
    for yy, k in ((back - 0.002, 1.0), (0.5 * (back + face_y), 0.975), (face_y, 0.94), (face_y + lp["sink"], 0.92)):
        o = S.rrect(lp["w"] * k, lp["h"] * k, lp["r"] * k, 5)
        rings.append(np.column_stack([c[0] + o[:, 0], np.full(len(o), yy), c[2] + o[:, 1]]))
    bucket = K.pos(S.cage(np.stack(rings), closed=True, caps=("fan", "fan"), cap_axes=((0, -1, 0), (0, 1, 0)), cap_rings=1,
                          mat=M["chrome"]))
    bez = S.rounded_frame(S.rrect(lp["w"], lp["h"], lp["r"], 5), S.rrect(lp["open_w"], lp["open_h"], 0.012, 4), lp["rim"],
                          r_out=0.007, r_in=0.003, n=3, floor=lp["pocket"], mat=M["chrome"], draft=0.004)
    floor_z = lp["rim"] - lp["pocket"]
    bez.assign(M["grille_back"], lambda cc, nn: (nn[:, 2] > 0.9) & (cc[:, 2] < floor_z + 0.0005))
    parts = [bez]
    lw, lh = lp["lens"]
    for dx in (-0.5, 0.5):
        lens = S.rounded_panel(S.rrect(lw, lh, 0.012, 4), 0.008, r_edge=0.003, n=2, dome=0.005, mat=M["lens_head"])
        parts.append(lens.moved((dx * (lw + lp["lens_gap"]), 0.0, floor_z - 0.001)))
    post = S.rounded_panel(S.stadium(0.012, lp["open_h"] + 0.004, 4), 0.012, r_edge=0.003, n=2, mat=M["chrome"])
    parts.append(post.moved((0.0, 0.0, floor_z - 0.001)))
    front = S.place(S.merge(parts), (c[0], back, c[2]), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    return [bucket, front]


def turn_signal(body, sx):
    """The amber turn signal at the pod's outer end: a chrome cup on a stalk into the fender, its domed lens facing
    forward in the lamps' plane."""
    sg = SIGNAL
    c, face_y = _lamp_frame(body, sx)
    x = c[0] + sx * sg["dx"]
    r, d = sg["r"], sg["depth"]
    cup = [(0.0, -d), (0.55 * r, -d + 0.002), (0.85 * r, -d + 0.010), (r, -0.5 * d), (r, -0.006), (r - 0.002, -0.001),
           (r - 0.006, 0.0), (0.0, 0.0)]
    m = K.lathe_on(cup, (x, c[1] + 0.008, c[2]), (0.0, -1.0, 0.0), seg=28, mat=M["chrome"])
    lens = [(0.0, -0.004), (r - 0.007, -0.004), (r - 0.0065, 0.002), (r - 0.015, 0.010), (0.55 * r, 0.014), (0.0, 0.016)]
    ln = K.lathe_on(lens, (x, c[1] + 0.008, c[2]), (0.0, -1.0, 0.0), seg=28, mat=M["lamp_amber"])
    hit = body.try_hit("hood", (x, c[1] - 0.4, c[2]), (0.0, 1.0, 0.0))
    fy = hit[0][1] if hit is not None else face_y
    stalk = K.tube([(x, c[1] + 0.008 + d - 0.01, c[2]), (x, max(fy + 0.015, c[1] + d + 0.02), c[2])], 0.014, sides=12,
                   mat=M["chrome"], ends=("flat", "flat"))
    return [m, stalk], [ln]


def fender_marker(body, sx):
    """An amber marker on the fender's side: a stadium lens on a chrome base, put on the fender's surface."""
    y, z = FENDER_MARKER
    hit = body.try_hit("hood", (sx * 2.0, y, z), (-sx, 0.0, 0.0))
    if hit is None:
        p, n = np.array([sx * L.FENDER_HALF_W, y, z]), np.array([sx, 0.0, 0.0])
    else:
        p, n = hit
    up = np.array([0.0, 0.0, 1.0])
    base = S.rounded_panel(S.stadium(0.095, 0.042, 6), 0.010, r_edge=0.004, n=2, mat=M["chrome"])
    lens = S.rounded_panel(S.stadium(0.078, 0.028, 6), 0.010, r_edge=0.004, n=2, dome=0.004, mat=M["lamp_amber"])
    return [S.place(base, p - n * 0.006, n, up)], [S.place(lens, p + n * 0.002, n, up)]


# ===================================================================================================================
# the fender mirrors
# ===================================================================================================================
def hood_mirror(body, sx):
    """A round convex mirror on a bent chrome stalk from the fender's crown, turned toward the driver's eye."""
    hm = HOOD_MIRROR
    fx, fy = hm["foot"]
    hit = body.try_hit("hood", (sx * fx, fy, 3.0), (0.0, 0.0, -1.0))
    p0, n0 = hit if hit is not None else (np.array([sx * fx, fy, L.FENDER_TOP_Z]), np.array([0.0, 0.0, 1.0]))
    hx, hy = hm["head"]
    head = np.array([sx * hx, hy, p0[2] + hm["rise"]])
    eye = np.array([sx * L.SEAT_X, L.HIP[1], L.HIP[2] + 0.75])
    look = K.unit(eye - head)
    road = K.unit(np.array([sx * 0.6, -1.0, -0.35]))      # what it shows: the road beside the front bumper
    nrm = K.unit(look + road)
    back = head - nrm * 0.035
    path = K.fillet3([p0 - n0 * 0.01, p0 + n0 * 0.10, back + np.array([0.0, 0.03, -0.05]), back], [0.0, 0.05, 0.03, 0.0], n=6)
    stalk = K.tube(path, 0.009, sides=10, mat=M["chrome"], ends=("flat", "round"))
    foot = K.lathe_on(K.disc_profile(0.030, 0.014, 0.004), p0 - n0 * 0.006, n0, seg=20, mat=M["chrome"])
    r = hm["r"]
    shell = [(0.0, -0.032), (0.45 * r, -0.030), (0.80 * r, -0.022), (r, -0.010), (r + 0.002, -0.002), (r - 0.002, 0.003),
             (r - 0.006, 0.002), (0.0, 0.002)]
    cup = K.lathe_on(shell, head, nrm, seg=28, mat=M["chrome"])
    glass = [(0.0, 0.0), (r - 0.006, 0.0), (r - 0.007, 0.004), (0.6 * r, 0.010), (0.0, 0.013)]
    gl = K.lathe_on(glass, head, nrm, seg=28, mat=M["mirror_glass"])
    return [stalk, foot, cup, gl]


# ===================================================================================================================
def parts(body):
    """{group: [Mesh]} of the nose."""
    chrome = [bumper()] + bumper_brackets() + tow_hooks()
    lamps, lenses = [], []
    for sx in (1.0, -1.0):
        lamps += headlamp(body, sx)
        a, b = turn_signal(body, sx)
        lamps += a
        lenses += b
        a, b = fender_marker(body, sx)
        lamps += a
        lenses += b
    mirrors = hood_mirror(body, 1.0) + hood_mirror(body, -1.0)
    return {"ext_bumper": chrome, "ext_lamps": lamps + lenses, "ext_mirrors": mirrors}
