"""Meshes of the sandbox piano (pure numpy, no bpy: tests build and measure them): rounded panels, lathes, sweeps and ring
cages from `core.shell`, in the prop's frame unless a part says otherwise; numbers from sandbox_piano_layout.

  boards      eight boards (two stacked a side), every edge rounded: rounded panels stood on edge
  seats       the four triangular corner seats
  sand        one closed cage: the bumpy top (rings scaled in toward the middle, heights from `sand_height`), a lip and
              a skirt down into the boards, a rolled bottom
  case        the rim on the grand's plan (rolled top, a coved well whose floor is the soundboard, `inside`), the cheek
              blocks, the key bed and the key slip
  lid         the lid in its hinge's frame (the builder turns it LID_DEG about Y)
  inside      the gold plate (a swept ring and a bar) and the strings
  keys        fifteen white keys and ten black keys, each a rounded panel
  trim        three turned legs, the pedal lyre with its box and three pedals, the lid's stick, the hinges, the music desk
  bench       a domed cushion on a swept rail and four turned legs"""
import itertools
import math

import numpy as np

from ....core import shell as S
from . import sandbox_piano_layout as L

M = L.MATS


def _lathe(profile, mat, seg=48, fillets=None, n=4):
    P = np.asarray(profile, float)
    if fillets is not None:
        P = S.fillet(P, fillets, n)
    return S.lathe(P, seg=seg, mat=mat)


def _catmull(points, n=10):
    """A uniform Catmull-Rom curve through `points` (k, 3), its ends kept: n samples per span."""
    P = np.asarray(points, float)
    G = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(G) - 2):
        p0, p1, p2, p3 = G[i - 1], G[i], G[i + 1], G[i + 2]
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1:])
    return np.concatenate(out)


def _panel(outline, t, mat, origin, normal, up, r_edge, r_back, dome=0.0):
    """A rounded panel (outline in its own x right / y up frame, `t` thick, both faces' edges rolled) put on a surface."""
    m = S.rounded_panel(outline, t, r_edge=r_edge, n=3, dome=dome, r_back=r_back, mat=mat)
    return S.place(m, origin, normal, up)


# ===================================================================================================================
# the sandbox
# ===================================================================================================================
def boards_mesh():
    """Two stacked boards a side; the front and back ones run the full width, the side ones between them, their ends
    hidden in the front and back boards."""
    out = []
    for k in range(2):
        zc = (k + 0.5) * L.BOARD_H
        long = S.rrect(L.BOX_W, L.BOARD_H - 0.002, L.BOARD_CORNER, 4)
        side = S.rrect(L.INNER_D + 2 * 0.020, L.BOARD_H - 0.002, L.BOARD_CORNER, 4)
        for s in (-1, 1):
            out.append(_panel(long, L.BOARD_T, M["wood"], (0.0, s * L.INNER_D / 2, zc), (0.0, s, 0.0), (0, 0, 1),
                              L.BOARD_R, L.BOARD_R))
            out.append(_panel(side, L.BOARD_T, M["wood"], (s * L.INNER_W / 2, 0.0, zc), (s, 0.0, 0.0), (0, 0, 1),
                              L.BOARD_R, L.BOARD_R))
    return S.merge(out)


def seats_mesh():
    """A right triangle of a seat on each corner, over the boards, overhanging them a little, its corners rounded."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            o = np.array([sx * (L.BOX_W / 2 + L.SEAT_OVER), sy * (L.BOX_D / 2 + L.SEAT_OVER)])
            P = np.array([o, o - [sx * (L.SEAT_LEG + L.SEAT_OVER), 0.0], o - [0.0, sy * (L.SEAT_LEG + L.SEAT_OVER)]])
            if S.polygon_area(P) < 0:
                P = P[::-1]
            P = S.resample_loop(S.fillet(P, [0.05, 0.06, 0.06], 8, closed=True), 72)
            m = S.rounded_panel(P, L.SEAT_T, r_edge=0.013, n=3, r_back=0.008, mat=M["seats"])
            out.append(m.moved((0.0, 0.0, L.BOARD_TOP)))
    return S.merge(out)


def sand_mesh(n=192, rings=40):
    """The sand: a closed cage from a rolled bottom up a skirt inside the boards, over a lip onto the bumpy top, whose
    rings are the edge scaled in toward the middle (heights from `sand_height`), closed by a fan in the middle."""
    O = S.resample_loop(S.rrect(L.INNER_W + 2 * L.SAND_TUCK, L.INNER_D + 2 * L.SAND_TUCK, 0.03, 4), n)
    Mi = S.mitre_offsets(O)
    edge = L.sand_height(O[:, 0], O[:, 1])

    def ring(d, z):
        return np.column_stack([O + d * Mi, np.broadcast_to(z, (n,))])

    R = [ring(0.035, 0.006), ring(0.012, 0.014), ring(0.002, 0.030), ring(0.0, 0.060), ring(0.0, edge - 0.010)]
    span = min(L.INNER_W, L.INNER_D) / 2
    for s in np.linspace(1.0 - 0.016 / span, 0.012, rings):
        P = O * s
        R.append(np.column_stack([P, L.sand_height(P[:, 0], P[:, 1])]))
    return S.cage(np.stack(R), closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_rings=1,
                  mat=M["sand"])


# ===================================================================================================================
# the piano's case
# ===================================================================================================================
def _roll(r, a0, a1, cx, cz, n=4):
    """Points (d, h) on a quarter round (centre (cx, cz), radius r) from angle a0 to a1 (degrees; 0 = +d, 90 = +h)."""
    a = np.radians(np.linspace(a0, a1, n + 1))
    return [(cx + r * math.cos(t), cz + r * math.sin(t)) for t in a]


def case_profile():
    """(inward offset, height over RIM_Z0) of the rim's section, from the underside's edge round the outside, over the
    rolled top, down the inner wall and through the cove onto the soundboard."""
    H = L.RIM_Z1 - L.RIM_Z0
    r, ri = L.RIM_R, 0.012
    floor = H - L.WELL
    prof = _roll(r, 270, 180, r, r)                                         # the underside's rolled edge
    prof += _roll(r, 180, 90, r, H - r)                                     # up the outside, over the top
    prof += _roll(ri, 90, 0, L.RIM_T - ri, H - ri)                          # the inner edge, down the wall
    prof += _roll(L.COVE, 180, 270, L.RIM_T + L.COVE, floor + L.COVE)      # the cove onto the soundboard
    out = []
    for p in prof:
        if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out, floor


def case_mesh():
    """The rim and the soundboard as one closed cage on the plan; the cheek blocks, the key bed and the key slip."""
    prof, floor = case_profile()
    R = S.offset_rings(L.plan(), prof) + np.array([0.0, 0.0, L.RIM_Z0])
    rim = S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_rings=2, mat=M["piano"])
    rim.assign(M["inside"], lambda c, nrm: (c[:, 2] < L.RIM_Z0 + floor + 0.006) & (nrm[:, 2] > 0.3))
    parts = [rim]
    u1 = L.KEY_Y0 + 0.014 - L.CASE_FRONT                                     # the cheek's front, past the key slip
    ch = np.array([(-0.06, L.Z_BED0 - 0.004), (u1, L.Z_BED0 - 0.004), (u1, L.KEYS_Z + 0.014), (u1 - 0.06, L.KEYS_Z + 0.046),
                   (-0.06, L.KEYS_Z + 0.070)])
    ch = S.resample_loop(S.fillet(ch, [0.012, 0.03, 0.03, 0.06, 0.012], 8, closed=True), 64)
    x0 = L.KEYS_W / 2 + 0.002
    cheek = _panel(ch, L.CHEEK_W, M["piano"], (x0, L.CASE_FRONT, 0.0), (1.0, 0.0, 0.0), (0, 0, 1), 0.027, 0.027)
    parts += [cheek, cheek.mirrored_x()]
    d0, d1 = L.CASE_FRONT - 0.02, L.KEY_Y0 + 0.006
    bed = S.rounded_panel(S.rrect(L.KEYS_W + 0.02, d1 - d0, 0.02, 4), L.BED_T, r_edge=0.018, n=3, r_back=0.018,
                          mat=M["piano"])
    parts.append(bed.moved((0.0, 0.5 * (d0 + d1), L.Z_BED0)))
    top = L.KEYS_Z - 0.012
    path = np.stack([np.linspace(-x0 - 0.02, x0 + 0.02, 24), np.full(24, L.KEY_Y0 + 0.017), np.full(24, 0.0)], 1)
    path[:, 2] = 0.5 * (L.Z_BED0 + top)
    parts.append(S.sweep(path, S.rrect(0.026, top - L.Z_BED0, 0.011, 4), up=(0, 0, 1), mat=M["piano"]))
    return S.merge(parts)


def lid_mesh():
    """The lid in its hinge's frame (origin on the hinge line over the spine's top edge, axes the prop's): the plan grown
    by LID_OVER, LID_T thick, its edges rolled; closed it lies on the rim's top."""
    P = S.offset_rings(L.plan(), [(-L.LID_OVER, 0.0)])[0][:, :2] - [L.HINGE[0], 0.0]
    return S.rounded_panel(P, L.LID_T, r_edge=0.010, n=3, r_back=0.009, mat=M["lid"])


def inside_mesh():
    """The gold plate on the soundboard (a swept ring inside the rim and a bar across the front) and the strings over it."""
    _, floor = case_profile()
    zf = L.RIM_Z0 + floor
    inset = L.RIM_T + L.COVE + 0.050
    ring = S.offset_rings(L.plan(), [(inset, zf + 0.012)])[0]
    plate = S.sweep(ring, S.rrect(0.080, 0.026, 0.011, 4), closed=True, up=(0, 0, 1), mat=M["plate"]).weld(1e-9)
    xa, xb = -L.CASE_W / 2 + inset - 0.03, L.CASE_W / 2 - inset + 0.03
    yb = L.CASE_FRONT - inset - 0.03
    bar = S.sweep(np.stack([np.linspace(xa, xb, 16), np.full(16, yb), np.full(16, zf + 0.020)], 1),
                  S.rrect(0.060, 0.036, 0.014, 4), up=(0, 0, 1), mat=M["plate"])
    parts = [plate, bar]
    P = L.plan()
    for x in np.linspace(-0.33, 0.40, 15):
        y1 = L.plan_back(x, P) + inset + 0.02
        path = np.stack([np.full(2, x), [yb - 0.01, y1], np.full(2, zf + 0.040)], 1)
        parts.append(S.tube(path, 0.0032, sides=8, mat=M["plate"]))
    return S.merge(parts)


# ===================================================================================================================
# the keyboard
# ===================================================================================================================
def keys_mesh():
    """Fifteen white keys from the lowest C at +X, and the black keys on their backs, each a rounded panel."""
    out = []
    white = S.rounded_panel(S.rrect(L.KEY_W, L.KEY_L, L.KEY_R, 4), L.KEY_H, r_edge=0.012, n=3, r_back=0.009,
                            mat=M["keys_white"])
    for i in range(L.WHITE):
        out.append(white.moved((L.white_x(i), L.KEY_Y0 - L.KEY_L / 2, L.KEYS_Z - L.KEY_H)))
    black = S.rounded_panel(S.rrect(L.BLACK_W, L.BLACK_L, 0.012, 4), L.BLACK_H, r_edge=0.010, n=3, r_back=0.006,
                            dome=0.0015, mat=M["keys_black"])
    y = L.KEY_Y0 - L.KEY_L + 0.004 + L.BLACK_L / 2
    for _i, x in L.black_keys():
        out.append(black.moved((x, y, L.KEYS_Z + L.BLACK_RISE - L.BLACK_H - 0.0015)))
    return S.merge(out)


# ===================================================================================================================
# legs, lyre, pedals, stick, hinges, desk
# ===================================================================================================================
def leg_mesh(z0, z1, r=L.LEG_R, seg=40):
    """A chunky turned leg from a ball foot at z0 up to a collar at z1 (a lathe about its own axis)."""
    h = z1 - z0
    prof = [(0.0, z0), (0.62 * r, z0), (0.95 * r, z0 + 0.06 * h), (0.58 * r, z0 + 0.15 * h), (0.80 * r, z0 + 0.19 * h),
            (0.62 * r, z0 + 0.25 * h), (0.70 * r, z0 + 0.55 * h), (1.00 * r, z0 + 0.72 * h), (0.66 * r, z0 + 0.86 * h),
            (0.95 * r, z0 + 0.92 * h), (0.95 * r, z1), (0.0, z1)]
    fil = [0.0, 0.3 * r, 0.6 * r, 0.3 * r, 0.25 * r, 0.3 * r, 1.0 * r, 1.2 * r, 0.3 * r, 0.15 * r, 0.1 * r, 0.0]
    return _lathe(prof, M["trim"], seg=seg, fillets=fil)


def _bentside_x(y, grow=0.0):
    """The x of the plan's treble side (the bentside, on -X) at y; `grow` moves the plan outward first (the lid's)."""
    P = L.plan() if not grow else S.offset_rings(L.plan(), [(-grow, 0.0)])[0][:, :2]
    Q = np.vstack([P, P[:1]])
    xs = [x0 + (x1 - x0) * (y - y0) / (y1 - y0) for (x0, y0), (x1, y1) in itertools.pairwise(Q)
          if (y0 - y) * (y1 - y) <= 0 and y0 != y1]
    return min(xs)


STICK_IN = 0.05                     # the stick holds the lid this far in from its edge


def stick_ends():
    """(foot, head) of the lid's stick: its foot on the middle of the bentside's rim STICK_BACK behind the front, its head
    on the open lid's underside STICK_IN in from the lid's edge (the lid's edge swings in toward the spine as it opens,
    so the stick leans in, as on a grand)."""
    y = L.CASE_FRONT - L.STICK_BACK
    foot = np.array([_bentside_x(y) + 0.5 * L.RIM_T * 1.25, y, L.RIM_Z1])
    x0 = _bentside_x(y, L.LID_OVER) + STICK_IN - L.HINGE[0]              # under the closed lid, from the hinge line
    head = np.array([L.HINGE[0], y, L.HINGE[1]]) + L.lid_matrix() @ np.array([x0, 0.0, 0.0])
    return foot, head


def stick_base():
    """Where the lid's stick stands on the bentside's rim (x, y)."""
    foot, _ = stick_ends()
    return float(foot[0]), float(foot[1])


def trim_mesh():
    """The three legs, the pedal lyre (two curved posts, the box, three pedals), the lid's stick, the hinges and the music
    desk with its ledge."""
    parts = []
    for x, b in L.LEGS:
        parts.append(leg_mesh(L.Z_SAND - L.SINK, L.RIM_Z0 + 0.025).moved((x * L.CASE_W, L.CASE_FRONT - b, 0.0)))
    # the lyre
    zt, zb = L.RIM_Z0 + 0.015, L.PEDAL_Z - 0.010
    for s in (-1, 1):
        pts = [(s * 0.050, L.LYRE_Y, zt), (s * 0.086, L.LYRE_Y, zt - 0.10), (s * 0.104, L.LYRE_Y, 0.5 * (zt + zb)),
               (s * 0.084, L.LYRE_Y, zb + 0.10), (s * 0.070, L.LYRE_Y, zb)]
        parts.append(S.tube(_catmull(pts, 8), 0.016, sides=14, mat=M["trim"]))
    box = S.rounded_panel(S.stadium(0.30, 0.080, 6), 0.050, r_edge=0.020, n=3, r_back=0.020, mat=M["trim"])
    parts.append(box.moved((0.0, L.LYRE_Y, L.PEDAL_Z - 0.050)))
    for x in (-0.075, 0.0, 0.075):
        path = _catmull([(x, L.LYRE_Y + 0.010, L.PEDAL_Z - 0.022), (x, L.LYRE_Y + 0.075, L.PEDAL_Z - 0.016),
                         (x, L.LYRE_Y + 0.135, L.PEDAL_Z - 0.006)], 6)
        sc = np.linspace(0.8, 1.25, len(path))
        sc[-1] *= 0.6                                                        # the tip rounds off
        parts.append(S.sweep(path, S.rrect(0.034, 0.012, 0.0055, 3), scale=np.stack([sc, np.ones_like(sc)], 1),
                             up=(0, 0, 1), mat=M["plate"]))
    # the stick, leaning in from the bentside's rim up to the open lid's underside
    foot, head = stick_ends()
    axis = head - foot
    n = float(np.linalg.norm(axis))
    z0, z1 = -0.004, n + 0.004
    prof = [(0.0, z0), (0.020, z0), (0.020, z0 + 0.016), (L.STICK_R, z0 + 0.04), (0.010, z1 - 0.03), (0.015, z1 - 0.012),
            (0.015, z1), (0.0, z1)]
    stick = _lathe(prof, M["trim"], seg=24, fillets=[0, 0.004, 0.006, 0.05, 0.05, 0.004, 0.003, 0])
    parts.append(stick.apply(S.rotation_between((0.0, 0.0, 1.0), axis / n), foot))
    # the hinges along the spine
    hinge = _lathe([(0.0, -0.035), (0.013, -0.035), (0.013, 0.035), (0.0, 0.035)], M["plate"], seg=20,
                   fillets=[0, 0.004, 0.004, 0]).rot(rx=90.0)
    for b in (0.12, 0.50, 0.84):
        parts.append(hinge.moved((L.HINGE[0] - 0.006, L.CASE_FRONT - b, L.HINGE[1])))
    # the music desk, leaning back on the rim's front, and its ledge
    d = L.DESK
    t = math.radians(d["tilt"])
    up, nrm = np.array([0.0, -math.sin(t), math.cos(t)]), np.array([0.0, math.cos(t), math.sin(t)])
    foot = np.array([-0.03, L.CASE_FRONT - d["back"], L.RIM_Z1 + 0.010])
    parts.append(_panel(S.rrect(d["w"], d["h"], 0.035, 5), d["t"], M["trim"], foot + up * d["h"] / 2, nrm, up, 0.008,
                        0.008))
    ledge = foot + nrm * (d["t"] + 0.010) + up * 0.004
    path = np.stack([np.linspace(-d["w"] / 2, d["w"] / 2, 12) + foot[0], np.full(12, ledge[1]), np.full(12, ledge[2])], 1)
    parts.append(S.sweep(path, S.rrect(0.034, 0.018, 0.0075, 3), up=tuple(up), mat=M["trim"]))
    return S.merge(parts)


# ===================================================================================================================
# the bench
# ===================================================================================================================
def bench_mesh():
    """A domed cushion (its top at SEAT_Z) on a swept rail round its underside, on four turned legs into the sand."""
    dome = 0.008
    z_base = L.SEAT_Z - L.CUSHION_T - dome
    cushion = S.rounded_panel(S.rrect(L.BENCH_W, L.BENCH_D, 0.070, 6), L.CUSHION_T, r_edge=0.028, n=4, r_back=0.014,
                              dome=dome, mat=M["bench"]).moved((0.0, L.BENCH_Y, z_base))
    loop = S.resample_loop(S.rrect(L.BENCH_W - 0.07, L.BENCH_D - 0.07, 0.05, 6), 64)
    zr = z_base - 0.022
    rail = S.sweep(np.column_stack([loop[:, 0], loop[:, 1] + L.BENCH_Y, np.full(len(loop), zr)]),
                   S.superellipse(0.040, 0.052, 2.6, 16), closed=True, up=(0, 0, 1), mat=M["bench_wood"]).weld(1e-9)
    parts = [cushion, rail]
    bx, by = L.BENCH_LEGS
    leg = leg_mesh(L.Z_SAND - L.SINK, zr + 0.012, r=0.034, seg=32)
    leg.Qm[:] = M["bench_wood"]
    leg.Tm[:] = M["bench_wood"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(leg.moved((sx * bx, L.BENCH_Y + sy * by, 0.0)))
    return S.merge(parts)


# ===================================================================================================================
# all of it
# ===================================================================================================================
def parts():
    """{key: Part(mesh, origin, rot)}: everything in the prop's frame but the lid, whose frame is its hinge line, turned
    LID_DEG about Y."""
    zero = (0.0, 0.0, 0.0)
    return {"boards": S.Part(boards_mesh(), zero, zero), "seats": S.Part(seats_mesh(), zero, zero),
            "sand": S.Part(sand_mesh(), zero, zero), "case": S.Part(case_mesh(), zero, zero),
            "lid": S.Part(lid_mesh(), (L.HINGE[0], 0.0, L.HINGE[1]), (0.0, L.LID_DEG, 0.0)),
            "inside": S.Part(inside_mesh(), zero, zero), "keys": S.Part(keys_mesh(), zero, zero),
            "trim": S.Part(trim_mesh(), zero, zero), "bench": S.Part(bench_mesh(), zero, zero)}
