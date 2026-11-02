"""Wheels of the semi truck (pure numpy, no bpy): what turns at each end of an axle.

A wheel object is built in its own frame: origin on the axle's centre line at the hub (the tyre's centre for a steer
wheel, midway between the two tyres of a dual pair), spin axis local X, the outboard side toward +X. The -X wheels are
mirror images, so one spin angle about local +X rolls every wheel forward (-Y). Every tyre's lowest point is at the
ground (the origin stands at L.WHEEL_Z = L.TYRE_R); ring vertex 0 of every surface of revolution is at the bottom.

    steer   one 11R22.5 rib tyre on a polished 22.5 x 8.25 wheel mounted disc-out (its mounting face L.WHEEL_OFFSET outboard
            of the tyre's centre), a pointed chrome hub cap and ten chrome nut covers
    dual    two tyres L.DUAL_SPACING apart (L.tyre_centres) on two identical wheels: the inner one disc-out like a steer
            wheel, the outer one turned round, their discs clamped together at the pair's centre; a chrome hub cover and ten
            nut covers on the outer disc's mounting side, at the bottom of the dish

Tyres (`tyre`): a real section (crowned tread, square-ish shoulders, a sidewall bulging to L.TYRE_W with a low raised
ring above the rim line, the bead sunk into the flange); the tread is geometry: steer and trailer tyres have five ribs
between four zigzag grooves, drive tyres staggered deep transverse lugs open over the shoulders and a centre rib. The tread
band has twice the segments of the sidewalls (stitched with triangles) to keep the budget.

Wheel (`rim`): one forged piece: barrel, both flanges, and the disc whose outer band rolls from the flat mounting pad into
the flange and carries ten oval hand holes between the ten studs (the inner wheel of a dual pair, hidden behind the outer
one, has a plain band). The hub (flange, pilot, bearing barrel bored to L.SPINDLE_R for the static spindle or axle tube)
and the brake drum (L.DRUM_R) are iron and turn with the wheel.

Roles: tyre; alloy (the wheels); iron (hub, drum); chrome (caps, nut covers).
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L

M = L.M

# ---- budgets: segments round the tread (rib, lug; the sidewalls have half), the wheel, the hub and drum, caps, a nut cover
SEG_RIB, SEG_LUG = 64, 96
SEG_RIM, SEG_HUB, SEG_CAP, SEG_NUT = 40, 24, 24, 8

# ---- the wheel (rim frame: the tyre's centre at x = 0, the disc on the +x side)
DISC_X = (L.WHEEL_OFFSET, L.DUAL_SPACING / 2)        # the disc's mounting face and its front face, from the tyre centre
DISC_T = DISC_X[1] - DISC_X[0]
SEAT_R = L.RIM_R - 0.004                             # the rim's tyre-side surface (under the beads)
BARREL_R = SEAT_R - 0.010                            # inside of the barrel
FLANGE_X = L.RIM_W / 2                               # inner face of a flange (the bead seat's end)
LIP_X = FLANGE_X + 0.0127                            # outer face of a flange
LIP_R = L.RIM_FLANGE_R + 0.004                       # the flange's rolled lip
PAD_R = 0.205                                        # the flat mounting pad round the bolt circle
HOLES = 10                                           # hand holes, between the studs
HOLE_W, HOLE_H, HOLE_P = 0.068, 0.040, 2.6           # tangential width, height along the band, superellipse exponent
HOLE_BEVEL = 0.0035
# the band the hand holes go through: front (outer) and back (inner) surfaces as quadratic Beziers (r, x), s = 0 inner
BAND_F = ((0.224, DISC_X[1] - 0.001), (0.279, DISC_X[1] - 0.001), (0.293, 0.131))
BAND_B = ((0.212, DISC_X[0] - 0.003), (0.250, DISC_X[0] - 0.003), (0.262, 0.104))
BORE_R = L.SPINDLE_R + 0.001                         # the hub turns on the spindle

# ---- tyres
CROWN_FLAT = 0.030                                   # the tread is flat this far either side of the centre line
TREAD_X = 0.112                                      # half width of the tread
CROWN_DROP = 0.0055                                  # how far the tread's edge lies below the crown
RIB_GROOVES = {"steer": (-0.072, -0.031, 0.031, 0.072),             # five ribs between four zigzag grooves
               "trailer": (-0.058, 0.0, 0.058)}                     # four ribs between three
RIB_GROOVE = (0.011, 0.006, 0.0135, 0.003)           # top width, bottom width, depth, zigzag amplitude
LUG = dict(rib=0.022, end=0.030, depth=(0.012, 0.017), wall=0.10, notch=0.006)   # centre rib half width, lug end, depth
                                                                                  # (in, edge), wall share, shoulder notch
SIDEWALL = ((L.TYRE_R - 0.040, 0.1352), (0.470, 0.1380), (0.438, 0.1395), (0.405, L.TYRE_W / 2), (0.352, 0.1356),
            (0.345, 0.1372), (0.338, 0.1352), (0.322, 0.1292), (0.300, LIP_X - 0.0008))   # (r, x) shoulder to rim line


def revolve(profile, seg, mat, closed_ends=False, phase=0.0):
    """Surface of revolution about the wheel's X axis: `profile` = (radius, x) points, outside on the right."""
    return S.lathe(profile, seg, mat=mat, closed_ends=closed_ends, phase=phase).rot(ry=90.0)


def _ring(r, x, theta):
    """Points (n, 3) at radius r (scalar or per point) and axial x round the X axis at the angles theta: angle 0 is at the
    bottom (-Z), positive angles turn toward +Y (the way a lathe turned by `revolve` lays its rings)."""
    r = np.broadcast_to(np.asarray(r, float), theta.shape)
    return np.stack([np.broadcast_to(np.asarray(x, float), theta.shape), r * np.sin(theta), -r * np.cos(theta)], 1)


def ring_mesh(rings, mat, skip=(), closed=True):
    """Faces joining consecutive rings (each (n_i, 3), starting at angle 0 and turning the same way): equal counts give
    quads, counts in a whole ratio a fan of triangles; `skip` lists ring indices i whose strip to i + 1 is left open;
    closed: the last ring joins the first. Outward winding is the caller's (check the volume)."""
    off = np.cumsum([0] + [len(r) for r in rings])
    Q, T = [], []
    pairs = list(range(len(rings) - (0 if closed else 1)))
    for i in pairs:
        if i in skip:
            continue
        k = (i + 1) % len(rings)
        na, nb = len(rings[i]), len(rings[k])
        a, b = off[i] + np.arange(na), off[k] + np.arange(nb)
        if na == nb:
            Q.append(np.stack([a, np.roll(a, -1), np.roll(b, -1), b], 1))
            continue
        flip = na < nb
        if flip:
            a, b, na, nb = b, a, nb, na
        q = na // nb
        assert q * nb == na, "ring counts must be in a whole ratio"
        j = np.arange(na)
        near = (j + q // 2) // q % nb                           # the coarse vertex each fine edge fans to
        tri = [np.stack([a[j], a[(j + 1) % na], b[near]], 1)]
        jc = np.arange(nb)
        tri.append(np.stack([b[(jc + 1) % nb], b[jc], a[(q * jc + q // 2) % na]], 1))
        tri = np.concatenate(tri)
        T.append(tri[:, ::-1] if flip else tri)
    V = np.concatenate(rings)
    return S.Mesh(V, np.concatenate(Q) if Q else None, np.concatenate(T) if T else None, mat=mat)


def outward(m):
    return m if m.volume() > 0 else m.flip()


def revolve_loop(loop, seg, mat, phase=0.0):
    """A closed (radius, x) outline revolved into a closed solid (outward whichever way the loop runs)."""
    th = phase + 2 * math.pi * np.arange(seg) / seg
    return outward(ring_mesh([_ring(r, x, th) for r, x in loop], mat))


# ===================================================================================================================
# tyre
# ===================================================================================================================
def crown_r(x):
    """Radius of the tread surface at axial x: flat across the middle, falling CROWN_DROP to the tread's edge."""
    t = np.clip((np.abs(x) - CROWN_FLAT) / (TREAD_X - CROWN_FLAT), 0.0, None)
    return L.TYRE_R - CROWN_DROP * t ** 2


def _tread_rib(n, grooves):
    """(theta (n,), [(r (n,), x (n,)) per column from -x to +x]): ribs between zigzag grooves centred at `grooves` (each
    groove's four columns move together, neighbours zigzag in opposite phase)."""
    th = 2 * math.pi * np.arange(n) / n
    zig = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    wt, wb, depth, amp = RIB_GROOVE
    cols = [(np.full(n, crown_r(TREAD_X)), np.full(n, -TREAD_X))]
    for g, xc in enumerate(grooves):
        xg = xc + amp * zig * (1.0 if g % 2 == 0 else -1.0)
        for dx, down in ((-wt / 2, 0.0), (-wb / 2, depth), (wb / 2, depth), (wt / 2, 0.0)):
            x = xg + dx
            cols.append((crown_r(x) - down, x))
    cols.append((np.full(n, crown_r(TREAD_X)), np.full(n, TREAD_X)))
    return th, cols


def _tread_lug(n):
    """(theta, columns): staggered transverse lugs (n / 4 pitches, four stations each: a steep wall, the lug or the groove,
    a steep wall, the other), deeper toward the shoulder and open over it, either side of a continuous centre rib."""
    p = 2 * math.pi / (n // 4)
    w = LUG["wall"]
    th = (np.arange(n) // 4) * p + np.array([0.0, w, 0.5, 0.5 + w])[np.arange(n) % 4] * p
    st = np.arange(n) % 4
    groove = {+1: (st == 1) | (st == 2), -1: (st == 3) | (st == 0)}     # the +x lugs' grooves, the -x side's half a pitch on
    d_in, d_edge = LUG["depth"]
    cols = []
    for side in (-1, 1):
        xs = [LUG["rib"], LUG["end"], TREAD_X]
        depth = [0.0, d_in, d_edge]
        part = []
        for x, d in zip(xs, depth):
            part.append((crown_r(x) - d * groove[side], np.full(n, side * x)))
        cols += part[::-1] if side < 0 else part
    return th, cols


def tyre(kind="steer"):
    """One 11R22.5 about its own centre (x = 0): "steer" (five ribs), "trailer" (four ribs) or "drive" (lugs). Closed,
    outward; its lowest point (ring vertex 0) is at -L.TYRE_R."""
    lug = kind == "drive"
    n = SEG_LUG if lug else SEG_RIB
    th, cols = _tread_lug(n) if lug else _tread_rib(n, RIB_GROOVES[kind])
    ths = 2 * math.pi * np.arange(n // 2) / (n // 2)
    shoulder_r, shoulder_x = L.TYRE_R - 0.020, 0.1300                   # one ring round the shoulder, on the tread's count
    edge = crown_r(TREAD_X) - 1e-9
    sr_neg = np.where(cols[0][0] < edge, shoulder_r - LUG["notch"], shoulder_r)   # a lug groove runs out over the shoulder
    sr_pos = np.where(cols[-1][0] < edge, shoulder_r - LUG["notch"], shoulder_r)
    rings = [_ring(SEAT_R + 0.001, -FLANGE_X, ths)]
    rings += [_ring(r, -x, ths) for r, x in SIDEWALL[::-1]]
    rings.append(_ring(sr_neg, -shoulder_x, th))
    rings += [_ring(r, x, th) for r, x in cols]
    rings.append(_ring(sr_pos, shoulder_x, th))
    rings += [_ring(r, x, ths) for r, x in SIDEWALL]
    rings.append(_ring(SEAT_R + 0.001, FLANGE_X, ths))
    return outward(ring_mesh(rings, M["tyre"]))


# ===================================================================================================================
# wheel: rim, flanges and the disc with its hand holes (rim frame)
# ===================================================================================================================
def _bez(ctrl, s):
    a, c, b = (np.asarray(p, float) for p in ctrl)
    s = np.asarray(s, float)[..., None]
    return (1 - s) ** 2 * a + 2 * s * (1 - s) * c + s ** 2 * b


def _arc_table(ctrl, n=200):
    s = np.linspace(0.0, 1.0, n + 1)
    P = _bez(ctrl, s)
    return s, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def rim_profile():
    """The wheel's section as a closed (r, x) loop and the indices (front, back) of the two band strips: the loop runs
    from the bore over the mounting pad, the band's back, into the barrel, round the plain flange, along the seat, round
    the disc-side flange, down the band's front and across the face back to the bore."""
    x0, x1 = DISC_X
    lip = [(0.2965, LIP_X), (0.3010, LIP_X - 0.0036), (LIP_R - 0.0009, FLANGE_X + 0.0047), (0.2992, FLANGE_X + 0.0008)]
    loop = ([(L.PILOT_R + 0.002, x0), (PAD_R - 0.004, x0), BAND_B[0], BAND_B[2],               # pad, the band's back
             (0.2702, 0.086), (BARREL_R, 0.066), (BARREL_R, -0.082), (0.2752, -0.0995),      # into the barrel and out
             (0.2812, -0.1140), (0.2870, -LIP_X + 0.0001)]                                 # round onto the plain flange
            + [(r, -x) for r, x in lip] + [(SEAT_R, -FLANGE_X), (SEAT_R, FLANGE_X)]          # its lip, the seat
            + lip[::-1][:-1] + [(0.2975, 0.119), BAND_F[2], BAND_F[0]]                       # the disc-side lip, the band
            + [(PAD_R, x1), (L.PILOT_R + 0.002, x1), (L.PILOT_R, x1 - 0.002), (L.PILOT_R, x0 + 0.002)])   # face, bore
    return loop, len(loop) - 6, 2


def _band(holes, seg):
    """The disc's band between the rings BAND_F / BAND_B at s = 0 and 1: with holes, per sector a frame of rings from the
    sector's outline (s = 0 and 1 on the lathe's angles, s = 1/2 on the sector's sides) in to an oval hand hole (bevelled
    both sides, its wall from the front surface to the back), centred between two studs; without, plain lathe strips."""
    if not holes:                                               # in the loop's order: front s 1 -> 0, back s 0 -> 1
        th = 2 * math.pi * np.arange(seg) / seg
        rings = [_ring(*_bez(BAND_F, s), th) for s in (1.0, 0.5, 0.0)]
        rings_b = [_ring(*_bez(BAND_B, s), th) for s in (0.0, 0.5, 1.0)]
        return [ring_mesh(rings, M["alloy"], closed=False), ring_mesh(rings_b, M["alloy"], closed=False)]
    k = seg // HOLES
    assert k * HOLES == seg
    sec = 2 * math.pi / HOLES
    # sector outline in (u, s): u along the sector (-1/2 .. 1/2), s across the band; counter-clockwise
    u_lo = np.linspace(-0.5, 0.5, k + 1)
    U = np.concatenate([u_lo, [0.5], u_lo[::-1], [-0.5]])
    Sv = np.concatenate([np.zeros(k + 1), [0.5], np.ones(k + 1), [0.5]])
    s_tab, a_tab = _arc_table(BAND_F)
    arc = lambda s: np.interp(s, s_tab, a_tab)                  # noqa: E731
    sarc = lambda a: np.interp(a, a_tab, s_tab)                 # noqa: E731
    a_h = arc(0.5)
    r_of = lambda s: _bez(BAND_F, s)[..., 0]                    # noqa: E731
    # the outline in metric coordinates about the hole's centre: rays to every outline point
    mu, mv = U * sec * r_of(Sv), arc(Sv) - a_h
    phi = np.arctan2(mv, mu)
    rho = 1.0 / (np.abs(np.cos(phi) / (HOLE_W / 2)) ** HOLE_P + np.abs(np.sin(phi) / (HOLE_H / 2)) ** HOLE_P) ** (1 / HOLE_P)

    def to_us(scale_add):
        rr = rho + scale_add
        a = a_h + rr * np.sin(phi)
        s = sarc(a)
        return rr * np.cos(phi) / (sec * r_of(s)), s

    u_e, s_e = to_us(HOLE_BEVEL)                                # the bevel's outer edge on the faces
    u_h, s_h = to_us(0.0)                                       # the hole itself
    out = []
    for c in range(HOLES):
        tc = (c + 0.5) * sec

        def front(u, s):
            r, x = _bez(BAND_F, s).T
            return _ring(r, x, tc + u * sec)

        def back(u, s):
            r, x = _bez(BAND_B, s).T
            return _ring(r, x, tc + u * sec)

        Fh, Bh = front(u_h, s_h), back(u_h, s_h)
        wall = np.linalg.norm(Bh - Fh, axis=1, keepdims=True)
        f = np.clip(HOLE_BEVEL / wall, 0.0, 0.45)
        rings = [front(U, Sv), front(u_e, s_e), Fh + (Bh - Fh) * f, Bh + (Fh - Bh) * f, back(u_e, s_e), back(U, Sv)]
        out.append(ring_mesh(rings, M["alloy"], closed=False).flip())     # wound like the lathe it closes
    return out


def rim(holes=True, seg=SEG_RIM):
    """The polished 22.5 x 8.25 wheel in the rim frame: one closed solid (barrel, flanges, disc with or without the ten
    hand holes)."""
    loop, i_front, i_back = rim_profile()
    th = 2 * math.pi * np.arange(seg) / seg
    body = ring_mesh([_ring(r, x, th) for r, x in loop], M["alloy"], skip=(i_front, i_back))
    m = S.merge([body] + _band(holes, seg)).weld(1e-7)
    return outward(m)


# ===================================================================================================================
# hub, drum, cap, nuts (wheel frame)
# ===================================================================================================================
def hub(x_m, x_end, drum_w):
    """The hub (its flange behind the mounting face x_m, the pilot through the discs, the bearing nose out to x_end, the
    barrel inboard, bored to BORE_R) and the brake drum bolted to the flange's back, open toward the axle (iron)."""
    fl = 0.020
    hub_loop = S.fillet([(BORE_R, x_end), (BORE_R, x_m - 0.200), (0.100, x_m - 0.200), (L.HUB_R, x_m - 0.060),
                         (L.HUB_R, x_m - fl), (0.168, x_m - fl), (0.168, x_m), (L.PILOT_R - 0.0008, x_m),
                         (L.PILOT_R - 0.0008, x_m + 0.050), (0.088, x_m + 0.056), (0.088, x_end)],
                        [0, 0, 0, 0, 0, 0.003, 0.002, 0, 0.003, 0, 0], n=1, closed=True)
    xd = x_m - fl
    drum_loop = S.fillet([(L.HUB_R - 0.004, xd), (0.236, xd), (0.236, xd - drum_w + 0.026), (0.243, xd - drum_w + 0.020),
                          (0.243, xd - drum_w), (L.DRUM_R, xd - drum_w), (L.DRUM_R, xd - 0.016), (L.HUB_R - 0.004, xd - 0.016)],
                         [0, 0.010, 0, 0, 0.003, 0.002, 0, 0], n=1, closed=True)
    return S.merge([revolve_loop(hub_loop, SEG_HUB, M["iron"]), revolve_loop(drum_loop, SEG_HUB, M["iron"])])


def nuts(x_face):
    """Ten chrome nut covers on the bolt circle at the studs (angles k * 36 degrees from the bottom, between the hand
    holes), standing on the face at x_face toward +x."""
    prof = S.fillet([(0.0, -0.004), (0.0205, -0.004), (0.0205, 0.007), (0.0180, 0.010), (0.0180, 0.040), (0.0, 0.056)],
                    [0, 0, 0.002, 0, 0.012, 0], n=1)
    nut = revolve(prof, SEG_NUT, M["chrome"], phase=math.pi / SEG_NUT)
    out = []
    for k in range(L.LUGS):
        a = 2 * math.pi * k / L.LUGS
        out.append(nut.moved((x_face, L.BOLT_R * math.sin(a), -L.BOLT_R * math.cos(a))))
    return S.merge(out)


def steer_cap(x_face):
    """The steer wheel's pointed chrome hub cap over the hub's end: a short skirt, a bead, an ogive to a rounded point."""
    prof = S.fillet([(0.0, x_face - 0.002), (0.104, x_face - 0.002), (0.104, x_face + 0.020), (0.096, x_face + 0.027),
                     (0.084, x_face + 0.060), (0.052, x_face + 0.102), (0.012, x_face + 0.126), (0.0, x_face + 0.128)],
                    [0, 0, 0.004, 0.006, 0.03, 0.03, 0.008, 0], n=2)
    return revolve(prof, SEG_CAP, M["chrome"])


def hub_cover(x_face, depth):
    """A drive or trailer hub's chrome cover: a flanged top hat over the hub's end, its crown domed."""
    prof = S.fillet([(0.0, x_face - 0.002), (0.114, x_face - 0.002), (0.114, x_face + 0.006), (0.100, x_face + 0.011),
                     (0.096, x_face + depth - 0.014), (0.072, x_face + depth), (0.0, x_face + depth + 0.006)],
                    [0, 0, 0.003, 0.004, 0.016, 0.03, 0], n=2)
    return revolve(prof, SEG_CAP, M["chrome"])


# ===================================================================================================================
# assembled wheels (the +X side; the -X side is the mirror image)
# ===================================================================================================================
def steer_wheel():
    x_m, x_face = DISC_X
    return S.merge([tyre("steer"), rim(True), hub(x_m, x_face + 0.050, 0.170), steer_cap(x_face), nuts(x_face)])


def dual_hub_x():
    """(mounting face, the outer disc's nut face) of a dual pair, from the pair's centre."""
    h = L.DUAL_SPACING / 2
    return DISC_X[0] - h, h - DISC_X[0]


def dual_wheel(kind):
    """A dual pair (kind "drive" or "trailer") about the pair's centre."""
    h = L.DUAL_SPACING / 2
    x_m, x_face = dual_hub_x()
    t = tyre(kind)
    inner, outer = rim(False).moved((-h, 0.0, 0.0)), rim(True).mirrored_x().moved((h, 0.0, 0.0))
    cover = hub_cover(x_face, 0.100 if kind == "drive" else 0.072)
    return S.merge([t.moved((-h, 0.0, 0.0)), t.moved((h, 0.0, 0.0)), inner, outer, hub(x_m, x_face + 0.040, 0.215), cover,
                    nuts(x_face)])


def wheel_meshes():
    """{kind: mesh of the +X wheel} for "steer", "drive" and "trailer"."""
    return {"steer": steer_wheel(), "drive": dual_wheel("drive"), "trailer": dual_wheel("trailer")}


def dynamic_parts():
    """Every wheel of L.wheel_hubs(): {key: Part(mesh, origin, (0, 0, 0))}."""
    meshes = wheel_meshes()
    mirrored = {k: m.mirrored_x() for k, m in meshes.items()}
    kinds = {L.wheel_key(a, s): kind for a, _, kind, _ in L.axles() for s in ("L", "R")}
    out = {}
    for key, origin in L.wheel_hubs().items():
        m = meshes[kinds[key]] if origin[0] > 0 else mirrored[kinds[key]]
        out[key] = S.Part(m.copy(), tuple(float(c) for c in origin), (0.0, 0.0, 0.0))
    return out
