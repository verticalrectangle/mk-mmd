"""convertible_exterior: the nose and the tail of the 1980s convertible, every piece a lofted, swept or lathed form.

The nose: a framed grille of fine slats, two flush rectangular headlamps (a bezel, two clear lenses, a dark reflector), amber
wraparound corner lamps and a swept rubber bumper with rounded end caps. The tail: a full-width ribbed red lamp band in two
units either side of a dark centre panel, a swept rubber bumper, plates in chrome frames. Pieces that sit on the body are put
on its smooth (subdivided) surface with ray casts (`probes["body"]`, a `core.shell.Probe`), so nothing floats and nothing sinks.

Exports: static_parts(probes) {"trim": ..., "lens": ...}; the geometry is bpy-free (without probes it uses the cage polygon).
"""
import numpy as np

from ....core import shell as S
from . import convertible_layout as LAY
from . import convertible_body as BODY

M = LAY.M
UP = (0.0, 0.0, 1.0)

GRILLE = dict(x=0.0, z=0.665, w=0.64, h=0.19, vbars=21)
HEADLAMP = dict(x=0.515, z=0.672, w=0.40, h=0.112, lens_w=0.180, lens_h=0.080, gap=0.012)
CORNER = dict(x=0.80, z=0.672, w=0.085, h=0.078)
BUMPER = dict(h=0.20, d=0.14, r=0.032, zc=0.40, x_end=0.80, y_face=0.07)       # section height/depth, centre height
EMBED = 0.004                    # how far a part's back sinks into the body surface it sits on
BEZEL_H, POCKET = 0.018, 0.010   # a bezel stands 14 mm proud of the body; its pocket floor is 10 mm below the rim
TAIL = dict(z=0.735, h=0.120, x0=0.145, x1=0.790, ribs=14, gap=0.004)


def _body_probe(probes):
    if probes is not None and "body" in probes:
        return probes["body"]
    return S.Probe.from_mesh(BODY.body_loft().mesh)


def _on(probe, x, z, side=-1):
    """The body surface point and outward normal at (x, z) seen from the nose (side -1) or the tail (+1)."""
    h = probe.ray((x, side * 3.0, z), (0.0, -side, 0.0))
    if h is None:
        raise ValueError(f"no body surface at x={x}, z={z} seen from y={side * 3.0}")
    return h[1], h[2]


# ===================================================================================== bumpers
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bumper_path(front, z, off=0.0, n=120):
    """The plan outline of a car end as a path at height z (straight across, rounded at the corners, along the sides),
    `off` further out than the bumper's centre line. Returns (path (n, 3), arc length of each point, total length)."""
    b = BUMPER
    yc = 0.5 * LAY.LENGTH - b["y_face"] + off
    sgn = -1.0 if front else 1.0
    xe = b["x_end"] + off
    pts = np.array([[-xe, sgn * (yc - 0.34)], [-xe, sgn * yc], [xe, sgn * yc], [xe, sgn * (yc - 0.34)]])
    if not front:
        pts = pts[::-1]
    plan = S.fillet(pts, [0.0, 0.32, 0.32, 0.0], n=10)
    seg = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    P = np.stack([np.interp(t, s, plan[:, 0]), np.interp(t, s, plan[:, 1])], 1)
    return np.column_stack([P, np.full(n, z)]), t, s[-1]


def bumper(front=True):
    """The bumper: a rounded rubber section swept along the plan outline of the car end, tapering into the body at both
    ends (the integrated end caps), with a bright bar along its top edge."""
    b = BUMPER
    path, t, total = _bumper_path(front, b["zc"])
    k = _smoothstep(np.minimum(t, total - t) / 0.16)                 # the end-cap taper
    scale = np.stack([0.30 + 0.70 * k, 0.72 + 0.28 * k], 1)
    m = _pos(S.sweep(path, S.rrect(b["d"], b["h"], b["r"], 4), scale=scale, up=UP, caps=(True, True), mat=M["bumper"]))
    # a rubber rub strip round the face of the bumper, at its middle
    spath, st, stotal = _bumper_path(front, b["zc"] + 0.012, off=0.0705)
    sk = _smoothstep((np.minimum(st, stotal - st) - 0.02) / 0.14)
    strip = _pos(S.sweep(spath, S.rrect(0.012, 0.034, 0.005, 3), scale=np.stack([0.3 + 0.7 * sk, 0.6 + 0.4 * sk], 1), up=UP,
                         caps=(True, True), mat=M["rubber"]))
    m = S.merge([m, strip])
    bpath, bt, btotal = _bumper_path(front, b["zc"] + 0.5 * b["h"] - 0.020, off=0.0145)
    bk = _smoothstep((np.minimum(bt, btotal - bt) - 0.03) / 0.10)
    bar = _pos(S.sweep(bpath, S.rrect(0.018, 0.024, 0.008, 3), scale=np.stack([0.4 + 0.6 * bk, 0.4 + 0.6 * bk], 1), up=UP,
                       caps=(True, True), mat=M["chrome"]))
    # rubber guards: two upright pads on the face of the bumper
    guards = []
    face = 0.5 * LAY.LENGTH - 0.0                                       # the bumper face (the car's tip)
    for sx in (1.0, -1.0):
        g = S.rounded_panel(S.stadium(0.050, 0.120, 6), 0.040, r_edge=0.012, n=4, dome=0.006, mat=M["rubber"])
        guards.append(S.place(g, (sx * (0.60 if front else 0.62), (-1.0 if front else 1.0) * (face - 0.045), b["zc"] - 0.02),
                              (0.0, -1.0 if front else 1.0, 0.0), UP))
    return S.merge([m, bar] + guards)


def _pos(m):
    return m if m.volume() > 0 else m.flip()


def plate(front=True):
    """Blank licence plate in a chrome frame on the bumper face (centre and size from the layout's plate surfaces)."""
    c = np.array(LAY.PLATE_FRONT if front else LAY.PLATE_REAR, float)
    nrm = np.array([0.0, -1.0 if front else 1.0, 0.0])
    w, h = LAY.PLATE_SIZE
    pl = S.rounded_panel(S.rrect(w, h, 0.010, 3), 0.003, r_edge=0.001, n=2, mat=M["plate"])
    fr = S.rounded_frame(S.rrect(w + 0.020, h + 0.020, 0.016, 3), S.rrect(w - 0.008, h - 0.008, 0.008, 3), 0.007, r_out=0.002,
                         r_in=0.0015, n=2, mat=M["chrome"])
    # the plate's face is the card surface plane: panel back 3 mm behind it, the frame one mm proud of it
    pl = S.place(pl, c - nrm * 0.003, nrm, UP)
    fr = S.place(fr, c - nrm * 0.006, nrm, UP)
    return S.merge([pl, fr])


# ===================================================================================== nose
def grille(probe):
    """A chrome frame round a dark pocket with a row of fine chrome vertical slats and two horizontal dividers."""
    g = GRILLE
    p, n = _on(probe, g["x"], g["z"])
    w, h = g["w"], g["h"]
    frame = S.rounded_frame(S.rrect(w, h, 0.022, 4), S.rrect(w - 0.040, h - 0.040, 0.012, 4), 0.022, r_out=0.006, r_in=0.003,
                            n=3, floor=0.016, mat=M["chrome"], draft=0.005)
    frame.assign(M["rubber"], lambda c, nn: (nn[:, 2] > 0.9) & (c[:, 2] < 0.0075))              # the pocket's floor is dark
    parts = [frame]
    iw, ih = w - 0.040, h - 0.040
    zf = 0.0085
    for k in range(g["vbars"]):                                                                # vertical slats
        x = -iw / 2 + (k + 0.5) * iw / g["vbars"]
        path = np.array([[x, -ih / 2 + 0.004, zf], [x, ih / 2 - 0.004, zf]])
        parts.append(S.tube(path, 0.0030, sides=8, caps=(True, True), up=UP, aspect=2.0, mat=M["chrome"]))
    for y in (-ih / 6, ih / 6):                                                                # two dividers across
        path = np.array([[-iw / 2 + 0.004, y, zf + 0.0012], [iw / 2 - 0.004, y, zf + 0.0012]])
        parts.append(S.tube(path, 0.0034, sides=8, caps=(True, True), up=(0.0, 1.0, 0.0), aspect=1.3, mat=M["chrome"]))
    return S.place(S.merge(parts), p - n * EMBED, n, UP)


def headlamp(probe, sx):
    hl = HEADLAMP
    p, n = _on(probe, sx * hl["x"], hl["z"])
    w, h = hl["w"], hl["h"]
    bez = S.rounded_frame(S.rrect(w, h, 0.018, 4), S.rrect(w - 0.024, h - 0.024, 0.011, 4), BEZEL_H, r_out=0.006, r_in=0.002,
                          n=3, floor=POCKET, mat=M["chrome"], draft=0.005)
    zfloor = BEZEL_H - POCKET
    bez.assign(M["rubber"], lambda c, nn: (nn[:, 2] > 0.9) & (c[:, 2] < zfloor + 0.0005))
    lens = []
    for dx in (-0.5, 0.5):
        L = S.rounded_panel(S.rrect(hl["lens_w"], hl["lens_h"], 0.011, 4), 0.0025, r_edge=0.0012, n=2, dome=0.002, mat=M["lens_head"])
        lens.append(L.moved(((hl["lens_w"] + hl["gap"]) * dx, 0.0, zfloor + 0.001)))
    m = S.merge([bez] + lens)
    return S.place(m, p - n * EMBED, n, UP)


def corner_lamp(probe, sx):
    c = CORNER
    p, n = _on(probe, sx * c["x"], c["z"])
    lens = S.rounded_panel(S.rrect(c["w"], c["h"], 0.022, 4), 0.012, r_edge=0.003, n=3, dome=0.004, mat=M["lamp_amber"])
    ring = S.rounded_frame(S.rrect(c["w"] + 0.014, c["h"] + 0.014, 0.027, 4), S.rrect(c["w"] - 0.002, c["h"] - 0.002, 0.021, 4),
                           0.010, r_out=0.002, r_in=0.0015, n=2, mat=M["chrome"])
    return S.place(S.merge([lens, ring]), p - n * EMBED, n, UP)


# ===================================================================================== tail
def tail_lamps(probe):
    """Two ribbed red lamp units (each 14 domed lens ribs in a dark pocket) and a dark centre panel with reverse lamps."""
    t = TAIL
    parts, lens_parts = [], []
    for sx in (1.0, -1.0):
        xs = sx * np.linspace(t["x0"], t["x1"], t["ribs"] + 1)
        step = (t["x1"] - t["x0"]) / t["ribs"]
        xc = sx * 0.5 * (t["x0"] + t["x1"])
        p, n = _on(probe, xc, t["z"], side=+1)
        pocket = S.rounded_frame(S.rrect(t["x1"] - t["x0"] + 0.028, t["h"] + 0.028, 0.020, 4),
                                 S.rrect(t["x1"] - t["x0"] + 0.004, t["h"] + 0.004, 0.012, 4), BEZEL_H, r_out=0.006, r_in=0.002,
                                 n=3, floor=POCKET, mat=M["chrome"], draft=0.005)
        zfloor = BEZEL_H - POCKET
        pocket.assign(M["rubber"], lambda c, nn: (nn[:, 2] > 0.9) & (c[:, 2] < zfloor + 0.0005))
        parts.append(S.place(pocket, p - n * EMBED, n, UP))
        for k in range(t["ribs"]):
            x = 0.5 * (xs[k] + xs[k + 1])
            pk, nk = _on(probe, x, t["z"], side=+1)
            rib = S.rounded_panel(S.rrect(step - t["gap"], t["h"] - 0.004, 0.010, 3), 0.003, r_edge=0.0012, n=2, dome=0.004,
                                  mat=M["lamp_tail"])
            lens_parts.append(S.place(rib, pk + nk * (BEZEL_H - POCKET - EMBED + 0.001), nk, UP))
    # the centre panel with two small reverse lamps
    p, n = _on(probe, 0.0, t["z"], side=+1)
    cen = S.rounded_panel(S.rrect(2 * t["x0"] - 0.02, t["h"], 0.014, 4), 0.014, r_edge=0.003, n=3, mat=M["trim_dark"])
    parts.append(S.place(cen, p - n * EMBED, n, UP))
    for sx in (1.0, -1.0):
        pk, nk = _on(probe, sx * 0.075, t["z"], side=+1)
        rev = S.rounded_panel(S.rrect(0.05, 0.044, 0.012, 3), 0.006, r_edge=0.002, n=2, dome=0.003, mat=M["plate"])
        parts.append(S.place(rev, pk + nk * 0.008, nk, UP))
    return S.merge(parts), S.merge(lens_parts)


# ===================================================================================== exports
def static_parts(probes=None):
    probe = _body_probe(probes)
    trim = [bumper(True), bumper(False), plate(True), plate(False), grille(probe)]
    lens = []
    for sx in (1.0, -1.0):
        trim += [headlamp(probe, sx), corner_lamp(probe, sx)]
    h = probe.ray((0.0, 1.78, 2.0), (0.0, 0.0, -1.0))                       # the high-mounted brake lamp on the deck lid
    brake = S.rounded_panel(S.stadium(0.20, 0.030, 6), 0.012, r_edge=0.004, n=3, dome=0.004, mat=M["lamp_tail"])
    lens.append(S.place(brake, h[1] - h[2] * EMBED, h[2], (0.0, 1.0, 0.0)))
    tail_trim, tail_lens = tail_lamps(probe)
    trim.append(tail_trim)
    lens.append(tail_lens)
    return {"trim": S.merge(trim), "lens": S.merge(lens)}
