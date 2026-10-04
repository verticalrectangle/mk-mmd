"""convertible_trim: the small exterior parts of the 1980s convertible (all swept, lathed or lofted forms).

Rub strips and rocker mouldings that follow the side of the body, the lip mouldings round the wheel openings, flush door
pulls, the fender vents behind the front wheels, a fuel door, side marker lamps, aero side mirrors on stalks, wipers,
an antenna, the exhaust tip, and the thin dark shut lines of the hood and the deck lid (graphic layers: group "decals").
Pieces that sit on the body are put on its smooth (subdivided) surface with ray casts (`probes["body"]`).

Exports: static_parts(probes) {"trim": ..., "lens": ..., "decals": ...}; the geometry is bpy-free (without probes it uses the
cage polygon of the body).
"""
import math

import numpy as np

from ....core import shell as S
from . import convertible_body as BODY
from . import convertible_layout as LAY

M = LAY.M
UP = (0.0, 0.0, 1.0)
EMBED = 0.004


def _body_probe(probes):
    if probes is not None and "body" in probes:
        return probes["body"]
    return S.Probe.from_mesh(BODY.body_loft().mesh)


def _side(probe, sx, y, z):
    """Point and outward normal of the body side (x > 0 for sx = +1) at (y, z)."""
    h = probe.ray((sx * 3.0, y, z), (-sx, 0.0, 0.0))
    if h is None:
        raise ValueError(f"no body side at y={y}, z={z}")
    return h[1], h[2]


def _pos(m):
    return m if m.volume() > 0 else m.flip()


def _taper(n, cap):
    """Per-point scale for a strip of n path points: 1 along the middle, rounding off over the `cap` points at both ends."""
    k = np.minimum(np.arange(n), np.arange(n)[::-1]) / max(cap, 1)
    k = np.clip(k, 0.0, 1.0)
    k = k * k * (3.0 - 2.0 * k)
    return np.stack([0.35 + 0.65 * k, 0.55 + 0.45 * k], 1)


def strip_on_side(probe, sx, y0, y1, z, section, mat, lift=0.002, n=60, cap=5):
    """A strip along the side of the body between y0 and y1 at height z: `section` (u, v) swept along the surface, centred
    `lift` above it (the strip is buried and proud by half its depth), its ends rounded off."""
    ys = np.linspace(y0, y1, n)
    hits = [_side(probe, sx, y, z) for y in ys]
    path = np.array([h[0] + h[1] * lift for h in hits])
    return _pos(S.sweep(path, section, scale=_taper(n, cap), up=UP, caps=(True, True), mat=mat))


def side_strips(probe):
    rub = S.rrect(0.016, 0.040, 0.007, 3)
    rocker = S.rrect(0.012, 0.020, 0.005, 3)
    out = []
    for sx in (1.0, -1.0):
        for y0, y1 in ((-2.07, -1.74), (-0.90, 0.90), (1.74, 2.07)):
            out.append(strip_on_side(probe, sx, y0, y1, LAY.Z_RUB + 0.012, rub, M["rubber"], lift=0.002))
        out.append(strip_on_side(probe, sx, -0.88, 0.88, 0.292, rocker, M["chrome"], lift=0.001, cap=3))
    return out


def arch_lips(probe):
    """The moulding round every wheel opening: a rounded rubber strip following the arch, a hand outside its edge."""
    out = []
    sec = S.rrect(0.026, 0.016, 0.006, 3)                          # u: across the strip in the surface, v: out of it
    for sx in (1.0, -1.0):
        for ay in (LAY.AXLE_F, LAY.AXLE_R):
            ang = np.radians(np.linspace(-3.0, 183.0, 56))                  # over the top, down to the sill level each side
            R = LAY.ARCH_R + 0.034
            pts = []
            for a in ang:
                y, z = ay + R * math.cos(a), LAY.WHEEL_Z + R * math.sin(a)
                p, nn = _side(probe, sx, y, z)
                pts.append(p + nn * 0.003)
            path = np.array(pts)
            out.append(_pos(S.sweep(path, sec, scale=_taper(len(path), 4), up=(sx, 0.0, 0.0), caps=(True, True), mat=M["rubber"])))
    return out


def door_pulls(probe):
    out = []
    for sx in (1.0, -1.0):
        p, n = _side(probe, sx, 0.46, 0.765)
        pull = S.rounded_panel(S.stadium(0.125, 0.024, 6), 0.011, r_edge=0.004, n=3, dome=0.003, mat=M["chrome"])
        rec = S.rounded_panel(S.stadium(0.145, 0.040, 6), 0.004, r_edge=0.0015, n=2, mat=M["trim_dark"])
        m = S.merge([rec, pull.moved((0, 0, 0.003))])
        out.append(S.place(m, p - n * EMBED, n, UP))
    return out


def fender_vents(probe):
    """The louvred vent behind each front wheel: a short frame with a dark pocket and three slanted slats."""
    out = []
    for sx in (1.0, -1.0):
        p, n = _side(probe, sx, -0.84, 0.675)
        fr = S.rounded_frame(S.rrect(0.050, 0.130, 0.014, 3), S.rrect(0.036, 0.116, 0.008, 3), 0.010, r_out=0.0025, r_in=0.0015,
                             n=2, floor=0.007, mat=M["chrome"])
        fr.assign(M["rubber"], lambda c, nn: (nn[:, 2] > 0.9) & (c[:, 2] < 0.0035))
        parts = [fr]
        for k in range(3):
            z = -0.032 + 0.032 * k
            path = np.array([[-0.0, z - 0.012, 0.0056], [0.0, z + 0.012, 0.0056]])
            parts.append(S.tube(path, 0.0024, sides=8, caps=(True, True), up=(1.0, 0.0, 0.0), aspect=1.0, mat=M["chrome"]))
        out.append(S.place(S.merge(parts), p - n * EMBED, n, UP))
    return out


def fuel_door(probe):
    p, n = _side(probe, 1.0, 1.93, 0.775)
    flap = S.rounded_panel(S.rrect(0.092, 0.118, 0.022, 4), 0.007, r_edge=0.0025, n=3, mat=M["paint"])
    ring = S.rounded_frame(S.rrect(0.106, 0.132, 0.028, 4), S.rrect(0.094, 0.120, 0.023, 4), 0.006, r_out=0.0015, r_in=0.001, n=2,
                           mat=M["seam"])
    return [S.place(S.merge([flap, ring]), p - n * EMBED, n, UP)]


def hood_vents(probe):
    """Two louvred vent panels on the hood: a rolled frame round a dark pocket with five slanted blades."""
    out = []
    for sx in (1.0, -1.0):
        x, y = sx * 0.24, -1.30
        h = probe.ray((x, y, 2.0), (0.0, 0.0, -1.0))
        p, n = h[1], h[2]
        fr = S.rounded_frame(S.rrect(0.240, 0.085, 0.016, 3), S.rrect(0.214, 0.059, 0.008, 3), 0.012, r_out=0.004, r_in=0.0025,
                             n=3, floor=0.009, mat=M["paint"], draft=0.003)
        fr.assign(M["rubber"], lambda c, nn: (nn[:, 2] > 0.9) & (c[:, 2] < 0.0035))
        parts = [fr]
        for k in range(5):
            yy = -0.0225 + 0.01125 * k
            path = np.array([[-0.100, yy, 0.0062], [0.100, yy, 0.0062]])
            parts.append(S.tube(path, 0.0022, sides=8, caps=(True, True), up=(0.0, 1.0, 0.0), aspect=1.5, mat=M["trim_dark"]))
        out.append(S.place(S.merge(parts), p - n * EMBED, n, (0.0, 1.0, 0.0)))
    return out


def side_markers(probe):
    out = []
    for sx in (1.0, -1.0):
        p, n = _side(probe, sx, 2.06, 0.70)
        lens = S.rounded_panel(S.rrect(0.075, 0.030, 0.012, 3), 0.007, r_edge=0.002, n=2, dome=0.0025, mat=M["lamp_tail"])
        out.append(S.place(lens, p - n * EMBED, n, UP))
    return out


# ===================================================================================== mirrors, wipers, antenna, tailpipe
def mirror(sx):
    """An aero side mirror: a pod lofted from rounded sections, a glass on its rear face, a stalk to the door corner."""
    x0, x1 = 0.865, 0.995
    xs = np.array([x0 - 0.02, x0 + 0.01, 0.905, 0.945, 0.975, x1])
    sd = np.array([0.040, 0.060, 0.074, 0.068, 0.054, 0.026])          # depth (y)
    sh = np.array([0.060, 0.092, 0.108, 0.100, 0.082, 0.040])          # height (z)
    yc, zc = -0.640, 1.040
    secs = []
    for x, d, h in zip(xs, sd, sh):
        P = S.superellipse(d, h, 2.5, 24)                              # (y, z) CCW seen from +x
        secs.append(np.array([[x, yc + q[0], zc + q[1]] for q in P]))
    pod = _pos(S.skin(np.array(secs), caps=(True, True), closed=True, mat=M["paint"]))
    glass = S.rounded_panel(S.rrect(0.130, 0.082, 0.030, 4), 0.004, r_edge=0.0015, n=2, mat=M["mirror_glass"])
    glass = S.place(glass, np.array([0.925, yc + 0.0335, zc]), np.array([-0.16, 0.987, 0.0]), UP)
    stalk = S.tube(np.array([[0.80, -0.69, 0.915], [0.845, -0.665, 0.985], [0.89, -0.645, 1.025]]), np.array([0.013, 0.011, 0.010]),
                   sides=10, caps=(True, True), mat=M["rubber"])
    m = S.merge([pod, glass, stalk])
    return m if sx > 0 else m.mirrored_x()


PIPE_X, PIPE_Z, PIPE_Y = -0.50, 0.205, (1.95, 2.25)       # tail pipe: right side, the tip points out of the tail (+Y)
ANT_BASE = (0.765, -1.02)                                 # antenna base (x, y) on the left front fender, by the cowl
ANT_LEAN_DEG = 14.0                                       # leaning back (toward +Y)
ANT_TIP_Z = 1.68                                          # mast tip height
WIPER_X = (0.17, -0.34)                                   # pivots of the driver's / the passenger's wiper
WIPER_T = 0.034                                           # pivot distance above the cowl line, along the glass
GLASS_TOP = 0.010                                         # the glass surface, out of the plane through the cowl line


def tailpipe():
    """Chrome tip with a rolled lip and a dark bore, a rubber hanger band; the tip leaves the tail panel."""
    x, z, (y0, y1) = PIPE_X, PIPE_Z, PIPE_Y
    ln, ro, roll = y1 - y0, 0.030, 0.0025
    floor = ln - 0.040
    lip = S.arc((ro - roll, ln - roll), roll, 0.0, math.pi, 6)
    prof = np.vstack([[(0.0, 0.0), (ro, 0.0)], lip, [(ro - 2 * roll, floor), (0.0, floor)]])
    pipe = S.lathe(prof, 32, mat=M["chrome"]).rot(rx=-90.0).moved((x, y0, z))
    pipe.assign(M["underbody"], lambda c, n: (n[:, 1] > 0.99) & (c[:, 1] < y0 + floor + 0.002))
    band = S.lathe([(ro - 0.0004, -0.008), (ro + 0.0018, -0.008), (ro + 0.0018, 0.008), (ro - 0.0004, 0.008),
                    (ro - 0.0004, -0.008)], 24, mat=M["rubber"]).weld(1e-9).rot(rx=-90.0).moved((x, y0 + 0.095, z))
    return [pipe, band]


def antenna(probe):
    """Rubber base and a tapered, telescoping chrome mast leaning back, on the left front fender."""
    h = probe.ray((ANT_BASE[0], ANT_BASE[1], 2.0), (0.0, 0.0, -1.0))
    base = np.array([ANT_BASE[0], ANT_BASE[1], h[1][2] - 0.004])
    grom = S.lathe([(0.0, -0.002), (0.020, -0.002), (0.020, 0.004), (0.0165, 0.010), (0.0115, 0.017),
                    (0.0100, 0.021), (0.0, 0.021)], 24, mat=M["rubber"]).moved(base)
    z0 = 0.018                                                          # the ferrule starts inside the base
    lean = math.radians(ANT_LEAN_DEG)
    length = (ANT_TIP_Z - (base[2] + z0)) / math.cos(lean)              # axis length from the ferrule to the tip
    rb = 0.0055                                                         # tip ball
    tip = 2 * rb * 0.9 + 0.0004
    rest = length - tip - 0.032                                         # the telescoping sections after the ferrule
    secs = [(0.0090, 0.032)] + [(r, f * rest) for r, f in ((0.0055, 0.34), (0.0046, 0.28), (0.0038, 0.22), (0.0030, 0.16))]
    pts, hh = [(0.0, 0.0)], 0.0
    for r, ln in secs:                                                  # steps: doubled points keep the shoulders crisp
        pts += [(r, hh), (r, hh), (r, hh + ln), (r, hh + ln)]
        hh += ln
    t = np.radians(np.linspace(-60.0, 90.0, 7))
    pts += [(rb * math.cos(a), hh + rb * 0.866 + rb * math.sin(a)) for a in t]
    pts[-1] = (0.0, pts[-1][1])
    mast = S.lathe(pts, 16, mat=M["chrome"]).weld(1e-9).rot(rx=-ANT_LEAN_DEG).moved(base + (0.0, 0.0, z0))
    return [grom, mast]


def _wiper():
    """One parked wiper in the glass frame (x along the cowl, y up the glass, z out of it): the pivot is the origin and
    the blade points toward +x; arm and blade are `rubber`. Nothing rises more than 2 cm off the glass."""
    head = S.lathe([(0.0, 0.003), (0.0115, 0.003), (0.0115, 0.010), (0.0092, 0.0165), (0.0050, 0.0192), (0.0, 0.0198)], 20,
                   mat=M["rubber"])
    arm_path = np.array([[0.0, 0.0, 0.0160], [0.06, 0.0, 0.0156], [0.12, 0.0, 0.0148], [0.19, 0.0, 0.0138]])
    arm = S.sweep(arm_path, S.rrect(0.012, 0.0038, 0.0016, n=2), up=(0.0, 0.0, 1.0), mat=M["rubber"])
    lip, spine = 0.00175, 0.00675                                       # blade section: a thin squeegee under a spine
    sec = np.array([(-lip, 0.0), (lip, 0.0), (lip, 0.003), (spine, 0.003), (spine, 0.009), (-spine, 0.009), (-spine, 0.003),
                    (-lip, 0.003)])
    xs = np.linspace(0.04, 0.34, 9)
    blade_path = np.stack([xs, np.zeros(9), np.full(9, 0.0025)], 1)
    taper = np.array([0.55, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.55])
    blade = S.sweep(blade_path, sec, scale=taper, up=(0.0, 0.0, 1.0), mat=M["rubber"])
    clip = S.rounded_panel(S.rrect(0.026, 0.015, 0.005, 3), 0.0085, r_edge=0.003, n=3, mat=M["rubber"]).moved((0.19, 0.0, 0.0085))
    return S.merge([head, arm, blade, clip])


def wipers():
    w = _wiper().rot(rz=3.0)                                            # parked: the tip a little above the pivot
    y0, z0 = LAY.WS_BASE
    return [S.merge([w.moved((x, WIPER_T, GLASS_TOP)).rot(rx=LAY.WS_RAKE_DEG).moved((0.0, y0, z0)) for x in WIPER_X])]


# ===================================================================================== shut lines (graphic layers)
def hood_seams(probe, probes_boot=None):
    """The dark shut lines on top: the hood's two sides and its rear edge, the deck lid's two sides. Thin ribbons that lie
    on the surface (a graphic layer)."""
    out = []

    def ray_down(x, y, z0=2.0):
        h = probe.ray((x, y, z0), (0.0, 0.0, -1.0))
        return h[1], h[2]

    for sx in (1.0, -1.0):
        for (y0, y1, x) in ((-2.10, LAY.Y_HOOD_REAR, 0.700), (1.72, 2.145, 0.715)):
            ys = np.linspace(y0, y1, 50)
            hits = [ray_down(sx * x, y) for y in ys]
            out.append(S.ribbon(np.array([h[0] for h in hits]), 0.0035, np.array([h[1] for h in hits]), lift=0.0004, mat=M["seam"]))
    xs = np.linspace(-0.70, 0.70, 40)
    hits = [ray_down(x, LAY.Y_HOOD_REAR) for x in xs]
    out.append(S.ribbon(np.array([h[0] for h in hits]), 0.0035, np.array([h[1] for h in hits]), lift=0.0004, mat=M["seam"]))
    if probes_boot is not None:                                  # the piping seam round the padded boot, 5 cm in from its edge
        b = BODY.BOOT
        yc, hl, hw = 0.5 * (b["y0"] + b["y1"]), 0.5 * (b["y1"] - b["y0"]) - 0.05, b["hw"] - 0.05
        t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        pts = []
        for a in t:
            c, sn = np.cos(a), np.sin(a)
            x, y = hw * np.sign(c) * abs(c) ** 0.45, yc + hl * np.sign(sn) * abs(sn) ** 0.45               # a rounded rectangle
            h = probes_boot.ray((x, y, 2.0), (0.0, 0.0, -1.0))
            if h is not None:
                pts.append((h[1], h[2]))
        if len(pts) > 20:
            out.append(S.ribbon(np.array([q[0] for q in pts]), 0.004, np.array([q[1] for q in pts]), lift=0.0005, mat=M["seam"], closed=True))
    return out


def static_parts(probes=None):
    probe = _body_probe(probes)
    trim = []
    trim += side_strips(probe) + arch_lips(probe) + door_pulls(probe) + fender_vents(probe) + fuel_door(probe) + hood_vents(probe)
    trim += [mirror(1.0), mirror(-1.0)] + wipers() + antenna(probe) + tailpipe()
    lens = side_markers(probe)
    boot = probes["boot"] if probes is not None and "boot" in probes else None
    return {"trim": S.merge(trim), "lens": S.merge(lens), "decals": S.merge(hood_seams(probe, boot))}
