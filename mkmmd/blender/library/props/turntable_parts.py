"""Meshes of the turntable (pure numpy, no bpy: tests build and measure them): lathes, sweeps and rounded panels from
`turntable_layout`'s numbers, never boxes (docs/modelling.md). `parts()` returns {key: Part(mesh, origin, rot)}: each part's
mesh in its own frame, the frame's origin in the prop's frame and its base rotation (Blender XYZ euler, degrees). The
builder makes one object `<name>_<key>` per part:

  plinth, feet, controls, arm_base    static: the top plate with its arm well, the base; the feet; the knobs, buttons,
                                      lamps and slider; the arm's base, rest, cue post and anti-skate knob
  platter                             turns about the spindle (frame: the spindle's axis on the plate's top face)
  mat, record, label, adapter         ride the platter (the same frame; the label is a graphic layer)
  yoke                                the gimbal's outer frame: swings with the arm (frame: on the axle, x along the arm)
  arm                                 swings and lifts (the same frame): tube, counterweight, headshell, cartridge, stylus
  cue_bar, cue_lever                  the cue's lift bar (rises) and its lever (tips up)"""
import math

import numpy as np

from ....core import shell as S
from . import turntable_layout as L

M = L.MATS
SEG = 256                       # segments round the big lathes (platter, mat, record)


def _lathe(profile, mat, seg=48, fillets=None, n=3, closed=False, at=(0.0, 0.0, 0.0)):
    """A lathe about +Z of a (r, z) profile (filleted first when `fillets` is given), moved to `at`. closed: the profile
    is a loop (an annulus: a record, a ring), joined back to its first point and welded."""
    P = np.asarray(profile, float)
    if fillets is not None:
        P = S.fillet(P, fillets, n, closed=closed)
    if closed:
        P = np.vstack([P, P[:1]])
    m = S.lathe(P, seg=seg, mat=mat)
    if closed:
        m = m.weld(1e-9)
    return m.moved(at)


def _assign(m, mat, fn):
    """Material `mat` for the faces whose centre (k, 3) passes fn; returns the mesh."""
    m.assign(mat, lambda c, nrm: fn(c))
    return m


def _knurl(m, z0, z1, r_min, ribs, depth, centre=(0.0, 0.0)):
    """Vertical ribs pressed into a lathe between heights z0 and z1 (vertices at least r_min from the axis): the radius
    goes in and out by `depth` (a share) `ribs` times round."""
    V = m.V.copy()
    d = V[:, :2] - np.asarray(centre, float)
    r = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0])
    sel = (V[:, 2] > z0) & (V[:, 2] < z1) & (r > r_min)
    k = 1.0 - depth * 0.5 * (1.0 - np.cos(ribs * a[sel]))
    V[sel, :2] = np.asarray(centre, float) + d[sel] * k[:, None]
    return S.Mesh(V, m.Q, m.T, m.Qm, m.Tm, m.UV, Ce=m.Ce, Cw=m.Cw)


def _panel(outline, t, mat, at, r_edge=0.001, dome=0.0, r_back=0.0, normal=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), n=3):
    """A rounded panel (outline in its own x right / y up frame, `t` thick) put on a surface at `at`."""
    m = S.rounded_panel(outline, t, r_edge=r_edge, n=n, dome=dome, r_back=r_back, mat=mat)
    return S.place(m, at, normal, up)


def _frame(outer, inner, h, mat, at, floor=None, r_out=0.0008, r_in=0.0006):
    m = S.rounded_frame(outer, inner, h, r_out=r_out, r_in=r_in, n=2, floor=floor, mat=mat)
    return m.moved(at)


def _circle(r, n=48):
    return S.ellipse(r, r, n)


# ===================================================================================================================
# plinth
# ===================================================================================================================
def _outline_hit(P, c, a):
    """Where the ray from c at angle a leaves the convex closed polygon P (m, 2)."""
    d = np.array([math.cos(a), math.sin(a)])
    A, B = P, np.roll(P, -1, axis=0)
    E = B - A
    den = E[:, 0] * d[1] - E[:, 1] * d[0]
    ok = np.abs(den) > 1e-15
    w = A - c
    t = np.where(ok, (E[:, 0] * w[:, 1] - E[:, 1] * w[:, 0]) / np.where(ok, den, 1.0), -1.0)   # along the ray
    u = np.where(ok, (d[0] * w[:, 1] - d[1] * w[:, 0]) / np.where(ok, den, 1.0), -1.0)         # along the edge
    hit = ok & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
    return c + d * t[hit].min()


def plate_mesh():
    """The aluminium top plate, BASE_TOP .. Z_PLATE, with both edges rolled and the arm well: a pocket WELL_R round the
    pivot, WELL_DEPTH deep with a cove at its floor (dark). Every ring is sampled on the same rays from the pivot (the
    rolled edge's rings on the inset outlines, so their points stay in step), and the top face runs along those rays from
    the edge straight in to the well's lip."""
    T = L.Z_PLATE - L.BASE_TOP
    outline = S.rrect(L.W, L.D, L.CORNER_R, 8)
    c = np.array(L.PIVOT)
    ang = np.sort(np.concatenate([np.arctan2(outline[:, 1] - c[1], outline[:, 0] - c[0]),
                                  np.linspace(-math.pi, math.pi, 160, endpoint=False)]))
    ang = ang[np.concatenate([[True], np.diff(ang) > 2e-3])]
    rad = np.stack([np.cos(ang), np.sin(ang)], 1)
    rb, rt, ri, cove = 0.0015, L.PLATE_EDGE_R, 0.0012, 0.0018

    def out_ring(d, z):                                  # the outline inset by d (its corners' radius shrinks by d)
        inset = S.rrect(L.W - 2 * d, L.D - 2 * d, L.CORNER_R - d, 8)
        return np.column_stack([np.array([_outline_hit(inset, c, a) for a in ang]), np.full(len(ang), z)])

    def well_ring(r, z):
        return np.column_stack([c + r * rad, np.full(len(rad), z)])

    R = [out_ring(rb, 0.0)]
    R += [out_ring(rb * (1.0 - math.sin(t)), rb * (1.0 - math.cos(t))) for t in np.linspace(0, math.pi / 2, 4)[1:]]
    R += [out_ring(rt * (1.0 - math.cos(t)), T - rt + rt * math.sin(t)) for t in np.linspace(0, math.pi / 2, 5)]
    R += [well_ring(L.WELL_R + ri - ri * math.sin(t), T - ri + ri * math.cos(t)) for t in np.linspace(0, math.pi / 2, 4)]
    zf = T - L.WELL_DEPTH
    R += [well_ring(L.WELL_R - cove + cove * math.cos(t), zf + cove - cove * math.sin(t)) for t in np.linspace(0, math.pi / 2, 4)]
    m = S.cage(np.stack(R), closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_rings=2, mat=M["plate"])
    m = m.moved((0.0, 0.0, L.BASE_TOP))
    return _assign(m, M["dark"], lambda C: (np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]) < L.WELL_R + 1e-4)
                   & (C[:, 2] < L.Z_PLATE - 0.0008))


def base_mesh():
    """The dark base under the plate, a little inset, its bottom edge rolled."""
    i = L.BASE_INSET
    o = S.rrect(L.W - 2 * i, L.D - 2 * i, L.CORNER_R - i, 8)
    return S.rounded_panel(o, L.BASE_TOP - L.FOOT_H + 0.0004, r_edge=0.0008, n=2, r_back=0.006, mat=M["body"]).moved(
        (0.0, 0.0, L.FOOT_H))


def feet_mesh():
    """Four big insulator feet: knurled rubber with a bright ring near the top."""
    prof = [(0.0, 0.0), (0.0300, 0.0), (L.FOOT_R, 0.0045), (L.FOOT_R, 0.0195), (0.0328, 0.0205), (0.0346, 0.0210),
            (0.0346, 0.0242), (0.0318, 0.0250), (0.0300, L.FOOT_H), (0.0, L.FOOT_H)]
    fil = [0.0, 0.003, 0.002, 0.0012, 0.0004, 0.0005, 0.0005, 0.0004, 0.0008, 0.0]
    one = _lathe(prof, M["rubber"], seg=160, fillets=fil, n=3)
    one = _knurl(one, 0.0055, 0.0185, L.FOOT_R - 0.0015, 40, 0.03)
    one = _assign(one, M["metal"], lambda C: (C[:, 2] > 0.0204) & (C[:, 2] < 0.0248))
    return S.merge([one.moved((x, y, 0.0)) for x, y in L.FEET])


# ===================================================================================================================
# controls
# ===================================================================================================================
def controls_mesh():
    """Power knob with the strobe lamp at its foot, start / stop, the 33 and 45 buttons with their lamps, the pop-up target
    light, the pitch slider and the empty 45-adapter well."""
    z = L.Z_PLATE
    parts = []
    # power knob: a bright body under a dark cap
    px, py = L.POWER
    body = [(0.0, 0.0), (L.POWER_R, 0.0), (L.POWER_R, 0.0140), (L.POWER_R - 0.0006, 0.0148), (0.0, 0.0148)]
    parts.append(_lathe(body, M["metal"], fillets=[0, 0.0006, 0.0008, 0.0003, 0], at=(px, py, z)))
    cap = [(0.0, 0.0146), (L.POWER_R - 0.0008, 0.0146), (L.POWER_R - 0.0008, L.POWER_H - 0.0015), (0.0, L.POWER_H)]
    parts.append(_lathe(cap, M["body"], fillets=[0, 0.0003, 0.0022, 0], at=(px, py, z)))
    parts.append(_frame(_circle(L.POWER_R + 0.0022), _circle(L.POWER_R + 0.0004), 0.0012, M["body"], (px, py, z)))
    # the strobe lamp: a small lens at the knob's foot, toward the platter
    to = np.array(L.SPINDLE) - np.array(L.POWER)
    to = to / np.linalg.norm(to)
    lens_at = np.array(L.POWER) + to * (L.POWER_R + 0.0062)
    up = (float(to[0]), float(to[1]), 0.0)
    parts.append(_panel(S.superellipse(0.0090, 0.0050, 3.0, 32), 0.0018, M["lamp"], (lens_at[0], lens_at[1], z),
                        r_edge=0.0006, dome=0.0004, up=up))
    # start / stop and the speed buttons, each in a dark bezel
    sx, sy = L.START
    w, h = L.START_SIZE
    parts.append(_frame(S.rrect(w + 0.004, h + 0.004, 0.004, 4), S.rrect(w + 0.0006, h + 0.0006, 0.0032, 4), 0.0012,
                        M["body"], (sx, sy, z), floor=0.0010))
    parts.append(_panel(S.rrect(w, h, 0.003, 4), 0.0050, M["plate"], (sx, sy, z), r_edge=0.0016, dome=0.0004))
    sw, sh = L.SPEED_SIZE
    for key, (bx, by) in L.SPEED.items():
        parts.append(_frame(S.rrect(sw + 0.003, sh + 0.003, 0.0022, 4), S.rrect(sw + 0.0005, sh + 0.0005, 0.0016, 4),
                            0.0010, M["body"], (bx, by, z), floor=0.0008))
        parts.append(_panel(S.rrect(sw, sh, 0.0015, 4), 0.0032, M["plate"], (bx, by, z), r_edge=0.0010))
        parts.append(_panel(S.stadium(0.0070, 0.0016, 4), 0.0004, M["led" + key], (bx, by - 0.0018, z + 0.0031),
                            r_edge=0.0002, n=2))
    # the pop-up target light, its lamp toward the platter
    tx, ty = L.TARGET
    parts.append(_frame(_circle(L.TARGET_R + 0.0022), _circle(L.TARGET_R + 0.0003), 0.0010, M["body"], (tx, ty, z)))
    post = [(0.0, 0.0), (L.TARGET_R, 0.0), (L.TARGET_R, L.TARGET_H - 0.003), (0.0, L.TARGET_H)]
    parts.append(_lathe(post, M["metal"], fillets=[0, 0.0003, 0.0035, 0], at=(tx, ty, z)))
    to = np.array(L.SPINDLE) - np.array(L.TARGET)
    to = to / np.linalg.norm(to)
    win_at = (tx + to[0] * (L.TARGET_R - 0.0003), ty + to[1] * (L.TARGET_R - 0.0003), z + L.TARGET_H - 0.0062)
    parts.append(_panel(S.superellipse(0.0060, 0.0042, 3.0, 24), 0.0008, M["lamp"], win_at, r_edge=0.0003,
                        normal=(float(to[0]), float(to[1]), 0.0), up=(0.0, 0.0, 1.0), n=2))
    # pitch slider: a bright scale plate round a dark slot, and the knob at 0 %
    y0, y1 = L.PITCH["y"]
    ln = y1 - y0
    cy = 0.5 * (y0 + y1)
    plate = S.rounded_frame(S.rrect(0.030, ln + 0.022, 0.003, 4), S.rrect(0.0055, ln, 0.0026, 4), 0.0010, r_out=0.0005,
                            r_in=0.0004, n=2, floor=0.0055, mat=M["metal"])
    plate = _assign(plate, M["dark"], lambda C: C[:, 2] < 0.0006)
    parts.append(S.place(plate, (L.PITCH["x"], cy, z), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    for k in range(-8, 9, 2):                                           # the scale's ticks (no figures), 0 % the longest
        w = 0.0060 if k == 0 else 0.0042 if k % 4 == 0 else 0.0030
        tick = _panel(S.stadium(w, 0.0008, 2), 0.0002, M["dark"], (L.PITCH["x"] - 0.0068 - 0.5 * w,
                      L.PITCH["knob"] + k * (0.5 * ln - 0.006) / 8.0, z + 0.0009), r_edge=0.0001, n=2)
        parts.append(tick)
    knob = S.rounded_panel(S.rrect(0.0165, 0.0250, 0.0030, 4), 0.0110, r_edge=0.0028, n=3, dome=0.0018, mat=M["metal"])
    parts.append(knob.moved((L.PITCH["x"], L.PITCH["knob"], z + 0.0008)))
    ridge = S.sweep(np.array([(L.PITCH["x"] - 0.0062, L.PITCH["knob"], z + 0.0131),
                              (L.PITCH["x"] + 0.0062, L.PITCH["knob"], z + 0.0131)]), S.rrect(0.0022, 0.0016, 0.0007, 2),
                    up=(0.0, 0.0, 1.0), mat=M["body"])
    parts.append(ridge)
    # the adapter's well (empty: the adapter is on the record)
    ax, ay = L.ADAPTER_WELL
    parts.append(_frame(_circle(0.0238, 64), _circle(0.0198, 64), 0.0016, M["body"], (ax, ay, z)))     # a rubber ring
    parts.append(_panel(_circle(0.0202, 64), 0.0004, M["dark"], (ax, ay, z - 0.0002), r_edge=0.0002, n=2))  # its floor
    # the dust cover's hinge sockets on the back edge
    hw, hd, hh = L.HINGE_SIZE
    for hx in L.HINGES:
        block = S.rounded_panel(S.superellipse(hw, hd, 3.2, 64), hh, r_edge=0.0040, n=3, dome=0.0012, mat=M["body"])
        parts.append(block.moved((hx, 0.5 * L.D - 0.5 * hd - 0.0015, z - 0.0004)))
        slot = _panel(S.stadium(0.55 * hw, 0.0036, 4), 0.0008, M["dark"], (hx, 0.5 * L.D - 0.5 * hd - 0.0015,
                      z + hh + 0.0002), r_edge=0.0003, n=2)
        parts.append(slot)
    return S.merge(parts)


# ===================================================================================================================
# the tonearm
# ===================================================================================================================
U_H = np.array([math.cos(math.radians(-L.OFFSET_DEG)), math.sin(math.radians(-L.OFFSET_DEG)), 0.0])  # headshell axis
V_H = np.array([-U_H[1], U_H[0], 0.0])                                    # across it, away from the spindle
COLLAR = np.array([L.ARM_L, 0.0, 0.0]) - 0.052 * U_H                      # the headshell's back (52 mm to the stylus)
COLLAR_L = 0.010
TUBE_END = COLLAR - COLLAR_L * U_H


def tube_path(n=48):
    """The S-tube's axis in the arm's frame (z = 0): from the gimbal out, bowed away from the spindle, into the collar
    along the headshell's axis."""
    P0, t0 = np.array([0.013, 0.0, 0.0]), np.array([math.cos(math.radians(4.0)), math.sin(math.radians(4.0)), 0.0])
    P3 = TUBE_END
    P1, P2 = P0 + 0.075 * t0, P3 - 0.055 * U_H
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3


def tube_point(s):
    """The tube's axis at x = s in the arm's frame (on its first, forward part)."""
    P = tube_path(200)
    i = int(np.argmin(np.abs(P[:, 0] - s)))
    return P[i]


def _along(m, axis, at):
    """A lathe made about +Z laid along the unit vector `axis` (its +Z onto axis), moved to `at`."""
    return m.apply(S.rotation_between((0.0, 0.0, 1.0), axis), at)


def arm_mesh():
    """The arm in its frame (origin on the gimbal's axle, x along the arm line, z up): the gimbal's housing, the rear stub
    and counterweight, the chrome S-tube, the collar, the headshell with its finger lift, the cartridge and the stylus,
    whose tip is at STYLUS_LOCAL."""
    parts = []
    housing = [(0.0, -0.016), (0.0105, -0.016), (0.0105, 0.016), (0.0, 0.016)]
    parts.append(_along(_lathe(housing, M["metal"], seg=32, fillets=[0, 0.003, 0.003, 0]), (1.0, 0.0, 0.0), (0, 0, 0)))
    stub = np.array([[-0.016, 0.0, 0.0], [-0.090, 0.0, 0.0]])
    parts.append(S.tube(np.linspace(stub[0], stub[1], 8), 0.0052, sides=20, mat=M["metal"]))
    cw = [(0.0, 0.0), (0.0185, 0.0), (0.0195, 0.0070), (0.0195, 0.0380), (0.0150, 0.0410), (0.0, 0.0410)]
    weight = _along(_lathe(cw, M["body"], seg=48, fillets=[0, 0.0015, 0.001, 0.004, 0.002, 0]), (-1.0, 0.0, 0.0),
                    (-0.042, 0.0, 0.0))
    parts.append(_assign(weight, M["metal"], lambda C: C[:, 0] > -0.0495))
    parts.append(S.tube(tube_path(), L.R_TUBE, sides=20, mat=M["metal"]))
    collar = [(0.0, 0.0), (0.0060, 0.0), (0.0068, 0.0012), (0.0068, 0.0046), (0.0060, 0.0058), (0.0060, COLLAR_L),
              (0.0, COLLAR_L)]
    col = _lathe(collar, M["metal"], seg=32, fillets=[0, 0.0004, 0.0003, 0.0003, 0.0003, 0.0008, 0])
    col = _knurl(col, 0.0014, 0.0044, 0.0063, 24, 0.06)
    parts.append(_along(col, U_H, TUBE_END))
    # the headshell: a plate along U_H under the axis, wider at the front
    hs = np.array([(-0.0068, 0.0), (0.0068, 0.0), (0.0090, 0.040), (0.0094, 0.058), (0.0060, 0.0625),
                   (-0.0060, 0.0625), (-0.0094, 0.058), (-0.0090, 0.040)])
    hs = S.fillet(hs, [0.002, 0.002, 0.004, 0.003, 0.002, 0.002, 0.003, 0.004], 3, closed=True)
    shell = S.rounded_panel(hs, 0.0030, r_edge=0.0010, n=2, r_back=0.0008, mat=M["metal"])
    R = np.stack([-V_H, U_H, np.array([0.0, 0.0, 1.0])], axis=1)          # panel x -> across, y -> along, z -> up
    parts.append(shell.apply(R, COLLAR + np.array([0.0, 0.0, -0.0058])))
    # the finger lift: a strip curling out of the headshell's outer front corner
    t = np.linspace(0.0, 1.0, 16)
    lift = (COLLAR + U_H * (0.050 + 0.010 * t[:, None]) + V_H * (0.0085 + 0.022 * t[:, None])
            + np.array([0.0, 0.0, 1.0]) * (-0.0045 + 0.0085 * t[:, None] ** 1.6))
    taper = np.minimum(1.0, np.sin(np.linspace(0.12, math.pi - 0.12, 16)) * 3.0)
    parts.append(S.sweep(lift, S.rrect(0.0055, 0.0014, 0.0006, 2), scale=taper[:, None] * [1.0, 1.0],
                         up=(0.0, 0.0, 1.0), mat=M["metal"]))
    # the cartridge under the headshell's front, its front slanted back over the stylus
    side = np.array([(0.035, -0.0058), (0.0575, -0.0058), (0.0560, -0.0120), (0.0540, -0.0180), (0.035, -0.0180)])
    side = S.fillet(side, [0.0025, 0.0020, 0.004, 0.0020, 0.0025], 3, closed=True)
    body = S.rounded_panel(side, 0.0170, r_edge=0.0020, n=3, r_back=0.0020, dome=0.0010, mat=M["cartridge"])
    Rc = np.stack([U_H, np.array([0.0, 0.0, 1.0]), -V_H], axis=1)       # panel x -> along, y -> up, z -> across
    parts.append(body.apply(Rc, COLLAR + V_H * 0.0085))
    tip = np.array(L.STYLUS_LOCAL)
    base = COLLAR + U_H * 0.0540 + np.array([0.0, 0.0, -0.0172])
    cant = np.linspace(base, tip + np.array([0.0, 0.0, 0.0004]), 6)
    parts.append(S.tube(cant, np.linspace(0.0005, 0.0003, 6), sides=8, mat=M["metal"]))
    parts.append(_lathe([(0.0, 0.0), (0.0004, 0.0003), (0.00045, 0.0009), (0.0, 0.0012)], M["dark"], seg=12, at=tip))
    return S.merge(parts)


def yoke_mesh():
    """The gimbal's outer frame in the arm's frame: a U round the housing, the axle's caps, the stem down to the pillar."""
    path = np.array([(0.0, -0.0155, 0.0075), (0.0, -0.0155, -0.0110), (0.0, -0.0100, -0.0165), (0.0, 0.0100, -0.0165),
                     (0.0, 0.0155, -0.0110), (0.0, 0.0155, 0.0075)])
    yz = S.fillet(path[:, 1:], [0.0, 0.004, 0.004, 0.004, 0.004, 0.0], 4)
    P = np.column_stack([np.zeros(len(yz)), yz])
    parts = [S.sweep(P, S.rrect(0.0040, 0.0090, 0.0015, 3), up=(1.0, 0.0, 0.0), mat=M["metal"])]
    for sgn in (-1.0, 1.0):
        cap = _lathe([(0.0, 0.0), (0.0034, 0.0), (0.0034, 0.0030), (0.0, 0.0042)], M["metal"], seg=24,
                     fillets=[0, 0.0005, 0.0012, 0])
        parts.append(_along(cap, (0.0, sgn, 0.0), (0.0, sgn * 0.0172, 0.0)))
    stem = [(0.0, 0.0), (0.0068, 0.0), (0.0068, 0.0110), (0.0, 0.0110)]
    parts.append(_lathe(stem, M["metal"], seg=32, fillets=[0, 0.0006, 0.0010, 0], at=(0.0, 0.0, -0.0270)))
    return S.merge(parts)


def arm_base_mesh():
    """In the prop's frame: the knurled height ring and bright collar in the arm well, the pillar up to the gimbal, the
    anti-skate knob, the rest that holds the tube at REST_DEG, the cue's post and its lever's hinge."""
    px, py = L.PIVOT
    zf = L.Z_PLATE - L.WELL_DEPTH
    parts = []
    ring = [(0.0, 0.0), (0.0320, 0.0), (0.0320, L.WELL_DEPTH + 0.0045), (0.0300, L.WELL_DEPTH + 0.0052), (0.0, L.WELL_DEPTH + 0.0052)]
    ring = _lathe(ring, M["rubber"], seg=128, fillets=[0, 0.0005, 0.0010, 0.0004, 0])
    parts.append(_knurl(ring, 0.0015, L.WELL_DEPTH + 0.004, 0.031, 90, 0.025).moved((px, py, zf)))
    top = L.Z_PLATE + 0.0052
    collar = [(0.0, 0.0), (0.0285, 0.0), (0.0285, 0.0036), (0.0120, 0.0050), (0.0, 0.0050)]
    parts.append(_lathe(collar, M["metal"], seg=96, fillets=[0, 0.0004, 0.0012, 0.002, 0], at=(px, py, top)))
    pillar_top = L.Z_ARM - 0.0270
    pillar = [(0.0, 0.0), (0.0100, 0.0), (0.0090, pillar_top - top - 0.0050), (0.0, pillar_top - top - 0.0050)]
    parts.append(_lathe(pillar, M["metal"], seg=48, fillets=[0, 0.0015, 0.0008, 0], at=(px, py, top + 0.0050)))
    # anti-skate knob on the well's floor
    ax, ay = L.around(L.PIVOT, L.ANTISKATE["deg"], L.ANTISKATE["s"])
    r = L.ANTISKATE["r"]
    knob = [(0.0, 0.0), (r, 0.0), (r, L.WELL_DEPTH + 0.0060), (0.0, L.WELL_DEPTH + 0.0072)]
    knob = _lathe(knob, M["body"], seg=96, fillets=[0, 0.0004, 0.0016, 0])
    parts.append(_knurl(knob, 0.0010, L.WELL_DEPTH + 0.0050, r - 0.0008, 48, 0.04).moved((ax, ay, zf)))
    # the arm rest: a post with a padded fork under the tube where the parked arm lies
    rx, ry, rz = rest_point()
    under = rz - L.R_TUBE
    post = [(0.0, 0.0), (0.0040, 0.0), (0.0032, under - L.Z_PLATE - 0.0030), (0.0, under - L.Z_PLATE - 0.0030)]
    parts.append(_lathe(post, M["metal"], seg=24, fillets=[0, 0.0012, 0.0004, 0], at=(rx, ry, L.Z_PLATE)))
    pad = [(0.0, 0.0), (0.0058, 0.0), (0.0058, 0.0024), (0.0, 0.0024)]
    parts.append(_lathe(pad, M["body"], seg=32, fillets=[0, 0.0008, 0.0008, 0], at=(rx, ry, under - 0.0030)))
    across = np.array([math.cos(L.THETA_REST + math.pi / 2), math.sin(L.THETA_REST + math.pi / 2)])
    for sgn in (-1.0, 1.0):
        q = np.array([rx, ry]) + across * sgn * (L.R_TUBE + 0.0016)
        prong = [(0.0, 0.0), (0.0012, 0.0), (0.0012, 0.0026), (0.0, 0.0032)]
        parts.append(_lathe(prong, M["body"], seg=12, fillets=[0, 0.0002, 0.0008, 0], at=(q[0], q[1], under - 0.0010)))
    # the cue: its bar's post, and the lever's hinge block
    bx, by = bar_centre()
    zb = bar_rest_z()
    post = [(0.0, 0.0), (0.0030, 0.0), (0.0030, zb - zf - 0.0036), (0.0, zb - zf - 0.0036)]
    parts.append(_lathe(post, M["metal"], seg=20, fillets=[0, 0.0004, 0.0006, 0], at=(bx, by, zf)))
    hx, hy = L.around(L.PIVOT, L.CUE_LEVER["deg"], L.CUE_LEVER["s"])
    hinge = [(0.0, 0.0), (0.0052, 0.0), (0.0048, L.WELL_DEPTH + 0.0040), (0.0, L.WELL_DEPTH + 0.0046)]
    parts.append(_lathe(hinge, M["body"], seg=32, fillets=[0, 0.0006, 0.0015, 0], at=(hx, hy, zf)))
    return S.merge(parts)


def rest_point():
    """Where the parked tube's axis lies over the rest (prop frame)."""
    p = tube_point(L.REST_S)
    c, s = math.cos(L.THETA_REST), math.sin(L.THETA_REST)
    return (L.PIVOT[0] + c * p[0] - s * p[1], L.PIVOT[1] + s * p[0] + c * p[1], L.Z_ARM + p[2])


def _bar_span():
    """The cue bar's arc: its radius from the pivot and the angles (radians) it spans under the tube's swing."""
    p = tube_point(L.CUE_S)
    r, off = math.hypot(p[0], p[1]), math.atan2(p[1], p[0])
    return r, L.THETA_IN + off - 0.10, L.THETA_REST + off + 0.10


def bar_centre():
    r, a0, a1 = _bar_span()
    return L.around(L.PIVOT, math.degrees(0.5 * (a0 + a1)), r)


def bar_rest_z():
    """The top of the cue bar with the arm down: BAR_GAP under the level tube."""
    return L.Z_ARM - L.R_TUBE - L.BAR_GAP


def cue_bar_mesh():
    """The lift bar in its own frame (origin at its middle's top, prop axes): an arc under the tube's whole swing, on a
    plunger that slides in its post."""
    r, a0, a1 = _bar_span()
    a = np.linspace(a0, a1, 28)
    cx, cy = bar_centre()
    path = np.column_stack([L.PIVOT[0] + r * np.cos(a) - cx, L.PIVOT[1] + r * np.sin(a) - cy, np.full(len(a), -0.0016)])
    taper = np.minimum(1.0, np.sin(np.linspace(0.08, math.pi - 0.08, len(a))) * 4.0)
    bar = S.sweep(path, S.rrect(0.0060, 0.0032, 0.0012, 3), scale=taper[:, None] * [1.0, 1.0], up=(0.0, 0.0, 1.0),
                  mat=M["body"])
    plunger = _lathe([(0.0, 0.0), (0.0018, 0.0), (0.0018, 0.0072), (0.0, 0.0072)], M["metal"], seg=16,
                     fillets=[0, 0.0003, 0.0003, 0], at=(0.0, 0.0, -0.0102))
    return S.merge([bar, plunger])


def cue_lever_mesh():
    """The cue lever in its frame (origin on its hinge, x along the lever): a tapered bar with a round tip."""
    n = 14
    t = np.linspace(0.0, 1.0, n)
    path = np.column_stack([L.CUE_LEVER["len"] * t, np.zeros(n), 0.0016 * np.sin(math.pi * t)])
    w = np.interp(t, [0.0, 0.15, 0.85, 1.0], [0.55, 1.0, 0.8, 0.5])
    lever = S.sweep(path, S.rrect(0.0042, 0.0030, 0.0012, 3), scale=w[:, None] * [1.0, 1.0], up=(0.0, 0.0, 1.0),
                    mat=M["body"])
    tip = _lathe([(0.0, -0.0032), (0.0034, -0.0028), (0.0034, 0.0024), (0.0, 0.0032)], M["body"], seg=24,
                 fillets=[0, 0.0012, 0.0012, 0], at=(L.CUE_LEVER["len"], 0.0, 0.0))
    return S.merge([lever, tip])


def hinge_point():
    """The cue lever's hinge (prop frame) and the direction it points (degrees)."""
    hx, hy = L.around(L.PIVOT, L.CUE_LEVER["deg"], L.CUE_LEVER["s"])
    return (hx, hy, L.Z_PLATE + 0.0040), L.CUE_LEVER["deg"]


# ===================================================================================================================
# platter, mat, record, label, adapter (the platter's frame: the spindle's axis on the plate's top face)
# ===================================================================================================================
def platter_mesh():
    """The platter: a bright lower lip, the dark bevel the strobe dots are drawn on (material `rim`), a bright top lip
    and the flat top the mat lies on; the spindle."""
    top = L.PLATTER_TOP - L.Z_PLATE
    prof = [(0.0, 0.0015), (0.1640, 0.0015), (L.PLATTER_R, 0.0040), (L.PLATTER_R, 0.0058), (0.1650, 0.0068),
            (0.1540, 0.0168), (0.1534, 0.0182), (L.TOP_R, top), (0.0, top)]
    fil = [0.0, 0.0012, 0.0008, 0.0005, 0.0006, 0.0005, 0.0004, 0.0006, 0.0]
    p = _lathe(prof, M["metal"], seg=SEG, fillets=fil, n=3)
    p = _assign(p, M["rim"], lambda C: (C[:, 2] > 0.0063) & (C[:, 2] < 0.0172) & (np.hypot(C[:, 0], C[:, 1]) > 0.152))
    sp = [(0.0, top - 0.002), (L.SPINDLE_R, top - 0.002), (L.SPINDLE_R, L.SPINDLE_TOP - L.Z_PLATE - 0.0012),
          (0.0, L.SPINDLE_TOP - L.Z_PLATE)]
    spindle = _lathe(sp, M["metal"], seg=24, fillets=[0, 0.0, 0.0016, 0])
    return S.merge([p, spindle])


def mat_mesh():
    """The felt slipmat: a soft-edged disc with the spindle's hole."""
    z0 = L.PLATTER_TOP - L.Z_PLATE
    z1 = z0 + L.MAT_T
    loop = [(L.SPINDLE_R + 0.0010, z0), (L.MAT_R - 0.0004, z0), (L.MAT_R, z1 - 0.0010), (L.MAT_R - 0.0010, z1),
            (L.SPINDLE_R + 0.0016, z1), (L.SPINDLE_R + 0.0010, z1 - 0.0006)]
    return _lathe(loop, M["mat"], seg=SEG, fillets=[0.0002, 0.0004, 0.0008, 0.0008, 0.0004, 0.0002], n=2, closed=True)


def record_mesh():
    """The 7-inch single: a rounded bead at its rim, the recorded surface at Z_REC, the raised label area round the 38.2 mm
    centre hole (all `vinyl`: the grooves and the sheen are drawn by its material)."""
    z0 = L.Z_MAT - L.Z_PLATE
    loop = [(L.HOLE_R, z0), (L.REC_R - 0.0004, z0), (L.REC_R, z0 + 0.0006), (L.REC_R - 0.0003, z0 + L.BEAD_T),
            (L.REC_R - 0.0010, z0 + L.BEAD_T), (L.LEADIN_R + 0.0004, z0 + L.REC_T), (L.LABEL_AREA_R + 0.0010, z0 + L.REC_T),
            (L.LABEL_AREA_R - 0.0004, z0 + L.LABEL_AREA_T), (L.HOLE_R + 0.0004, z0 + L.LABEL_AREA_T),
            (L.HOLE_R, z0 + L.LABEL_AREA_T - 0.0004)]
    fil = [0.0002, 0.0003, 0.0003, 0.0002, 0.0006, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
    return _lathe(loop, M["vinyl"], seg=SEG, fillets=fil, n=2, closed=True)


def label_mesh():
    """The paper label: a flat ring on the label area (a graphic layer, its printed rings drawn by its material)."""
    z = L.Z_LABEL - L.Z_PLATE
    return _lathe([(L.LABEL_R, z), (0.5 * (L.LABEL_R + L.HOLE_R), z), (L.HOLE_R + 0.0005, z)], M["label"], seg=SEG)


def adapter_mesh():
    """The 45 adapter: a bright puck that fills the centre hole, its top rounded, the spindle through its bore."""
    z0 = L.Z_MAT - L.Z_PLATE
    z1 = L.ADAPTER_TOP - L.Z_PLATE
    loop = [(L.SPINDLE_R + 0.0002, z0), (L.ADAPTER_R - 0.0005, z0), (L.ADAPTER_R, z0 + 0.0005), (L.ADAPTER_R, z1 - 0.0010),
            (L.ADAPTER_R - 0.0024, z1), (L.SPINDLE_R + 0.0012, z1), (L.SPINDLE_R + 0.0002, z1 - 0.0008)]
    fil = [0.0001, 0.0003, 0.0003, 0.0028, 0.0020, 0.0004, 0.0001]
    return _lathe(loop, M["metal"], seg=96, fillets=fil, n=3, closed=True)


# ===================================================================================================================
# all of it
# ===================================================================================================================
GRAPHIC = ("label",)            # graphic layers (exempt from `form`)
SPIN = ("mat", "record", "label", "adapter")     # ride the platter


def parts():
    """{key: Part(mesh, origin, rot)}: see the module docstring."""
    platter_at = (L.SPINDLE[0], L.SPINDLE[1], L.Z_PLATE)
    arm_at = (L.PIVOT[0], L.PIVOT[1], L.Z_ARM)
    yaw = math.degrees(L.THETA_REST)
    bx, by = bar_centre()
    hinge, hdeg = hinge_point()
    out = {"plinth": S.Part(S.merge([plate_mesh(), base_mesh()]), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
           "feet": S.Part(feet_mesh(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
           "controls": S.Part(controls_mesh(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
           "arm_base": S.Part(arm_base_mesh(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
           "platter": S.Part(platter_mesh(), platter_at, (0.0, 0.0, 0.0)),
           "mat": S.Part(mat_mesh(), platter_at, (0.0, 0.0, 0.0)),
           "record": S.Part(record_mesh(), platter_at, (0.0, 0.0, 0.0)),
           "label": S.Part(label_mesh(), platter_at, (0.0, 0.0, 0.0)),
           "adapter": S.Part(adapter_mesh(), platter_at, (0.0, 0.0, 0.0)),
           "yoke": S.Part(yoke_mesh(), arm_at, (0.0, 0.0, yaw)),
           "arm": S.Part(arm_mesh(), arm_at, (0.0, 0.0, yaw)),
           "cue_bar": S.Part(cue_bar_mesh(), (bx, by, bar_rest_z()), (0.0, 0.0, 0.0)),
           "cue_lever": S.Part(cue_lever_mesh(), hinge, (0.0, 0.0, hdeg))}
    return out
