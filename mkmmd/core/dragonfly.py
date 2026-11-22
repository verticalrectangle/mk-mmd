"""A paper dragonfly on a wire, bpy-free (the library prop `dragonfly`, docs/design.md: the library table, Flights): its
meshes at unit span, the colour roles, the wings' whirr and its driver expressions, the iridescent film of its cellophane
wings, and the card. The flight that moves it and the wire's shape are mkmmd/core/flight.py.

Frame (the butterfly's): the origin the middle of the thorax, the head toward -Y, the back +Z, its left +X. Sizes are at
unit span (wing tip to wing tip, wings flat); a builder scales by `size` (metres: 0.45 by default, a toy theatre's rod
puppet beside a marionette 1.5 m tall).

Modelled in proportion on a top-view photograph of a chaser spread flat (`mk ref photos`: the body 0.62 of the span,
the forewing 0.47 long and 0.086 across, the hindwing 0.147 across at its base, the pterostigma a dark cell near each tip
on the leading edge), then made a toy: the head twice as big with two big googly eyes (white domes, big dark pupils
looking forward and out, a glint), a little smile, the abdomen a striped paper straw (stripes wound round it, ring
grooves at its nine joints), the wings cellophane in a paper frame with two veins.

Whirr. A wing is lifted `wing_angle(phase)` above flat: (1 - rest) (mid + amp / 2 cos(2 pi phase)) - rest droop degrees,
the hindwings `lag` of a cycle behind the forewings. `phase` advances `whirr` cycles a POSE (not a second): at 12 poses a
second (a stepped set) and at 60 alike the wings stand somewhere else on every pose, so a whirr reads at both (a beat a
second's worth of cycles at 12 a second would look still). `rest` (0 flying, 1 perched) lays them out flat, drooping."""
import functools
import math

import numpy as np

from . import butterfly as BF
from . import shell as S

DEFAULTS = {"size": 0.45, "whirr": 0.41, "amp": 70.0, "mid": 10.0, "lag": 0.5, "droop": 4.0, "step": 0.0,
            "step_from": 0.0, "wire": 0.002, "film": 0.8, "clear_wing": 0.42, "smile": True,
            "hand": (0.95, 0.15, -0.55)}
ROOT_PROPS = ("phase", "rest", "amp", "mid", "lag", "droop")     # what the wings' drivers read off the root
FUNCS = frozenset({"radians", "cos"})
TAU = 6.2831853
MATS = ("straw", "stripe", "thorax", "face", "eye", "pupil", "shine", "legs", "wing", "vein", "spot")
MI = {k: i for i, k in enumerate(MATS)}
ROLES = {"straw": "surface", "stripe": "love", "thorax": "love:text:0.22", "face": "rose", "eye": "paper",
         "pupil": "text", "shine": "paper", "legs": "text", "wing": "surface", "vein": "text",
         "spot": "love", "wire": "subtle"}
ROLE_DOCS = {"straw": "the straw's paper between its stripes", "stripe": "the stripes wound round the abdomen",
             "thorax": "the thorax", "face": "the face under the eyes", "eye": "the eyes' whites",
             "pupil": "the pupils and the smile", "shine": "the glint on each pupil", "legs": "the legs",
             "wing": "the cellophane's own tint", "vein": "the wings' paper frame and veins",
             "spot": "the pterostigma near each wing tip", "wire": "the steel wire"}
INKS = ("foam", "iris", "rose", "gold")            # the film's colours, in order round its bands

# ===================================================================================================================
# the body (unit span)
# ===================================================================================================================
EYE_R = 0.062
EYE_AT = (0.050, -0.130, 0.036)                    # the left eye's centre (+X); the right one mirrored
PUPIL = (0.60, 34.0)                               # a pupil's look (out, as a share of forward) and angular radius
THORAX = ((0.0, -0.076), (0.027, -0.072), (0.045, -0.053), (0.053, -0.019), (0.054, 0.015), (0.048, 0.045),
          (0.036, 0.064), (0.020, 0.073), (0.0, 0.077))      # (r, toward the head) of the thorax's lathe
THORAX_TALL = 1.12
ABDOMEN = (0.050, 0.497)                           # y from, to
_ABDOMEN_R = ((0.050, 0.025), (0.075, 0.024), (0.110, 0.0185), (0.150, 0.0205), (0.300, 0.021), (0.420, 0.0225),
              (0.462, 0.0195), (0.485, 0.0125), (0.497, 0.0045))
JOINTS = 9                                         # ring grooves between the abdomen's segments
PITCH = 0.11                                       # the straw's stripes wind once round every PITCH along it
BANDS = 4                                          # stripes round it, the two colours in turn
BELLY = (0.0, 0.012, -0.060)                       # where the wire meets the thorax, underneath
FEET_Z = -0.105                                    # the feet's lowest point (perched: what stands on the target)


def abdomen_radius(y):
    P = np.asarray(_ABDOMEN_R, float)
    r = np.interp(y, P[:, 0], P[:, 1])
    j = 0.09 + np.arange(JOINTS) * (0.465 - 0.09) / (JOINTS - 1)
    groove = np.exp(-(((np.asarray(y, float)[..., None] - j) / 0.0055) ** 2)).sum(-1)
    return r * (1.0 - 0.075 * np.minimum(groove, 1.0))


def abdomen(m=24, rings=170):
    """The straw: rings along +Y, each turned on by its distance along / PITCH (so a column of faces follows a helix)
    and the columns in BANDS bands of the two colours; capped (the front cap hides in the thorax)."""
    y = np.linspace(ABDOMEN[0], ABDOMEN[1], rings)
    r = abdomen_radius(y)
    a = 2.0 * math.pi * np.arange(m) / m
    tw = 2.0 * math.pi * (y - y[0]) / PITCH
    ang = a[None, :] + tw[:, None]
    sec = np.stack([r[:, None] * np.cos(ang), np.broadcast_to(y[:, None], ang.shape), -r[:, None] * np.sin(ang)], -1)
    mesh = S.skin(sec, caps=(True, True), closed=True)
    band = (np.arange(m) * BANDS // m) % 2
    mesh.Qm = np.where(band[np.arange(len(mesh.Q)) % m] == 1, MI["stripe"], MI["straw"]).astype(np.int32)
    mesh.Tm = np.full(len(mesh.T), MI["straw"], np.int32)
    return mesh


def _lathe_y(profile, seg, mat, at=(0.0, 0.0, 0.0), tall=1.0):
    """A lathe about the body's axis: the profile's (r, toward the head) turned, laid along -Y, its height scaled."""
    m = S.lathe(np.asarray(profile, float), seg=seg, mat=mat).rot(rx=90.0)
    m.V[:, 2] *= tall
    return m.moved(at)


def _sphere(r, seg, mat, n=14):
    phi = np.linspace(0.0, math.pi, n)
    return S.lathe(np.stack([r * np.sin(phi), -r * np.cos(phi)], 1), seg=seg, mat=mat)


def _cap(r_in, r_out, alpha_deg, mat, seg=28, n=8):
    """A closed thin cap of a sphere about +Z (a paper disc glued round an eye): from the axis out to `alpha_deg`
    between radii r_in and r_out."""
    al = math.radians(alpha_deg)
    phi = np.linspace(0.0, al, n)
    inner = [(r_in * math.sin(p), r_in * math.cos(p)) for p in phi]
    outer = [(r_out * math.sin(p), r_out * math.cos(p)) for p in phi[::-1]]
    return S.lathe(np.array(inner + outer), seg=seg, mat=mat)


def _on_sphere(mesh, centre, direction):
    return mesh.apply(S.rotation_between((0.0, 0.0, 1.0), direction), centre)


def pupil_dir(sx):
    """The way a pupil looks (unit): forward (-Y), out to its side and a little up."""
    d = np.array([PUPIL[0] * sx, -1.0, 0.12])
    return d / np.linalg.norm(d)


def _eye(sx):
    c = np.array([EYE_AT[0] * sx, EYE_AT[1], EYE_AT[2]])
    d = pupil_dir(sx)
    up = np.array([0.0, 0.0, 1.0]) - d * d[2]
    up /= np.linalg.norm(up)
    g = d * math.cos(math.radians(16.0)) + up * math.sin(math.radians(16.0)) - np.array([0.25 * sx, 0.0, 0.0])
    g /= np.linalg.norm(g)
    return [_sphere(EYE_R, 32, MI["eye"]).moved(c),
            _on_sphere(_cap(EYE_R - 0.002, EYE_R + 0.0022, PUPIL[1], MI["pupil"]), c, d),
            _on_sphere(_cap(EYE_R + 0.0012, EYE_R + 0.0042, 8.5, MI["shine"]), c, g)]


def _quad_path(p0, p1, p2, n=12):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _legs():
    """Three pairs folded under the thorax and forward, a basket (as a dragonfly flies with them)."""
    out = []
    for sx in (1.0, -1.0):
        for y0, reach in ((-0.040, 0.050), (-0.012, 0.042), (0.016, 0.034)):
            path = _quad_path((0.022 * sx, y0, -0.040), (0.068 * sx, y0 - 0.010, -0.072),
                              (0.040 * sx, y0 - reach, FEET_Z + 0.004))
            out.append(S.tube(path, np.linspace(0.0050, 0.0034, len(path)), sides=8, mat=MI["legs"]))
    return out


def _smile():
    a = np.radians(np.linspace(-58.0, 58.0, 14))
    path = np.stack([0.020 * np.sin(a), -0.1735 + 0.012 * (1.0 - np.cos(a)), -0.040 - 0.011 * np.cos(a)], 1)
    return S.tube(path, 0.0028, sides=8, mat=MI["pupil"])


def body_mesh(smile=True):
    """The body at unit span (closed parts): the straw, the thorax, the face, the eyes with their pupils and glints,
    the legs and the smile; materials by MATS."""
    parts = [abdomen(),
             _lathe_y(THORAX, 28, MI["thorax"], tall=THORAX_TALL),
             _lathe_y([(0.0, -0.046), (0.030, -0.040), (0.046, -0.016), (0.048, 0.010), (0.038, 0.034),
                       (0.020, 0.046), (0.0, 0.049)], 24, MI["face"], at=(0.0, -0.128, -0.022), tall=0.9)]
    for sx in (1.0, -1.0):
        parts += _eye(sx)
    parts += _legs()
    if smile:
        parts.append(_smile())
    return S.merge(parts)


# ===================================================================================================================
# the wings: outlines in the wing's own frame (x out from its hinge, y toward the tail), cellophane in a paper frame
# ===================================================================================================================
_FORE = ((0.000, -0.010), (0.060, -0.017), (0.200, -0.021), (0.340, -0.021), (0.415, -0.016), (0.452, -0.004),
         (0.462, 0.014), (0.450, 0.034), (0.410, 0.050), (0.300, 0.064), (0.180, 0.070), (0.080, 0.062),
         (0.022, 0.040), (0.000, 0.016))
_HIND = ((0.000, -0.012), (0.060, -0.019), (0.200, -0.023), (0.340, -0.021), (0.415, -0.014), (0.446, 0.002),
         (0.452, 0.020), (0.436, 0.040), (0.380, 0.058), (0.260, 0.078), (0.150, 0.098), (0.075, 0.120),
         (0.032, 0.124), (0.008, 0.098), (0.000, 0.050))
SWEEP = {"fore": -5.0, "hind": 6.0}                 # degrees each wing is turned back from square (fore: forward)
HINGES = {"fore": (0.040, -0.030, 0.052), "hind": (0.040, 0.036, 0.052)}     # the left wings' hinges
THICK = 0.003                                       # the cellophane (unit span: 1.35 mm at 0.45 m)
FRAME = (0.007, 0.0034)                             # the paper frame round it: width, height
STIGMA = (0.034, 0.010)                             # the pterostigma: length along the leading edge, width
CAMBER = 0.035


def _cr(P0, P1, P2, P3, t, alpha=0.5):
    """Centripetal Catmull-Rom between P1 and P2 at parameters t (m,) in 0..1 -> (m, 2) (Barry-Goldman)."""
    t1 = max(float(np.linalg.norm(P1 - P0)), 1e-12) ** alpha
    t2 = t1 + max(float(np.linalg.norm(P2 - P1)), 1e-12) ** alpha
    t3 = t2 + max(float(np.linalg.norm(P3 - P2)), 1e-12) ** alpha
    tt = (t1 + np.asarray(t, float) * (t2 - t1))[:, None]
    A1 = (t1 - tt) / t1 * P0 + tt / t1 * P1
    A2 = (t2 - tt) / (t2 - t1) * P1 + (tt - t1) / (t2 - t1) * P2
    A3 = (t3 - tt) / (t3 - t2) * P2 + (tt - t2) / (t3 - t2) * P3
    B1 = (t2 - tt) / t2 * A1 + tt / t2 * A2
    B2 = (t3 - tt) / (t3 - t1) * A2 + (tt - t1) / (t3 - t1) * A3
    return (t2 - tt) / (t2 - t1) * B1 + (tt - t1) / (t2 - t1) * B2


def _closed_spline(P, n):
    """n points on a closed centripetal Catmull-Rom through the control points P (k, 2), evenly in arc length."""
    P = np.asarray(P, float)
    k = len(P)
    u = np.linspace(0.0, 1.0, 24, endpoint=False)
    Q = np.concatenate([_cr(P[(i - 1) % k], P[i], P[(i + 1) % k], P[(i + 2) % k], u) for i in range(k)])
    Qc = np.vstack([Q, Q[:1]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Qc, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n, endpoint=False)
    return np.stack([np.interp(s, d, Qc[:, 0]), np.interp(s, d, Qc[:, 1])], 1)


def outline(wing, n=160):
    """A left wing's outline (n, 2), counter-clockwise seen from above, swept by SWEEP about its hinge."""
    P = _closed_spline(_FORE if wing == "fore" else _HIND, n)
    a = math.radians(SWEEP[wing])
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    P = P @ Rm.T
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1].copy()


def camber(x):
    return CAMBER * np.asarray(x, float) ** 2


def wing_uv(wing, xy):
    """uv in the film image of points xy of a wing (its own frame): the forewings the image's upper half, the hind its
    lower; u out along the span."""
    P = outline(wing)
    lo, hi = P.min(0), P.max(0)
    u = np.clip((xy[:, 0] - lo[0]) / (hi[0] - lo[0]), 0.0, 1.0)
    v = np.clip((xy[:, 1] - lo[1]) / (hi[1] - lo[1]), 0.0, 1.0)
    base = 0.5 if wing == "fore" else 0.0
    return np.stack([u, base + 0.5 * (1.0 - v) * 0.98 + 0.005], 1)


def _membrane(wing):
    """The cellophane: a closed membrane THICK thick with a rolled edge (the butterfly's), camber toward the tip."""
    P = outline(wing)
    h = 0.5 * THICK
    t = np.gradient(np.vstack([P[-1:], P, P[:1]]), axis=0)[1:-1]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)                     # inward for a counter-clockwise outline
    F, Tf = BF.fill(P + h * nrm, spacing=0.02)
    m = len(P)

    def lift(xy, dz):
        return np.column_stack([xy, camber(xy[:, 0]) + dz])

    rings = [lift(F[:m], h)]
    for r in BF.RIM:
        phi = math.asin(r)
        rings.append(lift(P + h * (1.0 - math.cos(phi)) * nrm, h * r))
    rings.append(lift(F[:m], -h))
    R = np.stack(rings)
    K = len(R)
    idx = np.arange(K * m).reshape(K, m)
    j, j2 = np.arange(m), (np.arange(m) + 1) % m
    Q = np.concatenate([np.stack([idx[k][j], idx[k + 1][j], idx[k + 1][j2], idx[k][j2]], 1) for k in range(K - 1)])
    inner = F[m:]
    base = K * m
    top = np.concatenate([idx[0], base + np.arange(len(inner))])
    bot = np.concatenate([idx[K - 1], base + len(inner) + np.arange(len(inner))])
    T = np.concatenate([top[Tf], bot[Tf][:, ::-1]])
    V = np.concatenate([R.reshape(-1, 3), lift(inner, h), lift(inner, -h)])
    return S.Mesh(V, Q, T, UV=wing_uv(wing, V[:, :2]), mat=MI["wing"])


def _strip(path2, width, height, mat, closed=False, taper=None):
    """A paper strip (a flattened tube) lying on the membrane along a 2D path."""
    P = np.asarray(path2, float)
    path = np.column_stack([P, camber(P[:, 0]) + 0.5 * THICK + 0.35 * height])
    sec = np.array([[-0.5, 0.0], [-0.35, 0.5], [0.35, 0.5], [0.5, 0.0], [0.35, -0.5], [-0.35, -0.5]])
    sc = np.array([width, height])
    scale = np.tile(sc, (len(P), 1)) if taper is None else np.asarray(taper, float)[:, None] * sc
    m = S.sweep(path, sec[::-1], scale=scale, up=(0.0, 0.0, 1.0), closed=closed, mat=mat)
    if closed:
        m = m.weld()                                     # the ring's last section is its first again
    return m if m.volume() >= 0 else m.flip()


def _veins(wing):
    P = outline(wing)
    lo, hi = P.min(0), P.max(0)
    out = []
    span = hi[0] - lo[0]
    for frac, reach in ((0.30, 0.72), (0.55, 0.55)):
        x = np.linspace(0.02, reach * span, 18)
        top = np.interp(x, *_edge(P, "lead"))
        bot = np.interp(x, *_edge(P, "trail"))
        y = top + frac * (bot - top)
        taper = np.minimum(1.0, np.linspace(1.0, 0.0, len(x)) * 3.0 + 0.05)
        out.append(_strip(np.stack([x, y], 1), 0.0042, 0.0024, MI["vein"], taper=taper))
    return out


def _edge(P, which):
    """(x, y) of a wing outline's leading (the least y at each x) or trailing edge, sampled along x."""
    xs = np.linspace(P[:, 0].min() + 1e-4, P[:, 0].max() - 1e-4, 120)
    ys = []
    for x in xs:
        a, b = P, np.roll(P, -1, axis=0)
        hit = (a[:, 0] - x) * (b[:, 0] - x) <= 0
        yy = [a[i, 1] + (x - a[i, 0]) * (b[i, 1] - a[i, 1]) / ((b[i, 0] - a[i, 0]) or 1e-12) for i in np.flatnonzero(hit)]
        ys.append(min(yy) if which == "lead" else max(yy))
    return xs, np.array(ys)


def _stigma(wing):
    P = outline(wing)
    xs, lead = _edge(P, "lead")
    x1 = P[:, 0].max() - 0.045
    x0 = x1 - STIGMA[0]
    y0 = float(np.interp(0.5 * (x0 + x1), xs, lead)) + 0.0045
    pad = S.rounded_panel(S.rrect(STIGMA[0], STIGMA[1], 0.0035, 4), 0.0026, r_edge=0.001, dome=0.0004)
    a = math.atan2(float(np.interp(x1, xs, lead) - np.interp(x0, xs, lead)), x1 - x0)
    pad = pad.apply(S.rot_matrix(0.0, 0.0, math.degrees(a)), (0.5 * (x0 + x1), y0 + 0.5 * STIGMA[1],
                                                               float(camber(0.5 * (x0 + x1))) + 0.5 * THICK))
    pad.Qm[:] = MI["spot"]
    pad.Tm[:] = MI["spot"]
    return pad


def wing_mesh(wing, side="L"):
    """One wing (`fore` or `hind`) of a side in its own frame: the hinge at the origin, its axis along Y, the wing out
    along +X (L) or -X (R): the cellophane, its paper frame, two veins and the pterostigma; uv in the film image."""
    if wing not in ("fore", "hind"):
        raise ValueError(f"dragonfly: a wing is 'fore' or 'hind', not {wing!r}")
    P = outline(wing)
    t = np.gradient(np.vstack([P[-1:], P, P[:1]]), axis=0)[1:-1]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    inward = np.stack([-t[:, 1], t[:, 0]], 1)
    frame = _strip(P + inward * 0.45 * FRAME[0], FRAME[0], FRAME[1], MI["vein"], closed=True)
    parts = [_membrane(wing), frame, _stigma(wing)] + _veins(wing)
    m = S.merge(parts)
    if m.UV is None or len(m.UV) != len(m.V):            # the frame and veins take the uv of where they lie
        uv = wing_uv(wing, m.V[:, :2])
        m.UV = uv
    if side == "R":
        m = m.mirrored_x()
    elif side != "L":
        raise ValueError(f"dragonfly: side is 'L' or 'R', not {side!r}")
    return m


def hinge(wing, side="L"):
    x, y, z = HINGES[wing]
    return (x if side == "L" else -x, y, z)


@functools.cache
def parts(smile=True):
    """Every mesh at unit span (cached: read-only): {"body": Mesh, "fore_L", "fore_R", "hind_L", "hind_R": Mesh}."""
    out = {"body": body_mesh(smile)}
    for wing in ("fore", "hind"):
        for side in ("L", "R"):
            out[f"{wing}_{side}"] = wing_mesh(wing, side)
    return out


# ===================================================================================================================
# whirr
# ===================================================================================================================
def wing_angle(phase, rest, amp, mid, droop, ops=np):
    """How far a wing is lifted above flat (radians): (1 - rest) (mid + amp / 2 cos(2 pi phase)) - rest droop."""
    return np.radians((1.0 - rest) * (mid + 0.5 * amp * ops.cos(TAU * phase)) - rest * droop)


def expressions():
    """The wings' drivers (Euler Y about their hinges, radians), Blender simple expressions on the root's ROOT_PROPS:
    {"fore_L", "fore_R", "hind_L", "hind_R"}; the left wings turn by -angle (a lift), the right by +angle."""
    out = {}
    for wing, ph in (("fore", "phase"), ("hind", "(phase + lag)")):
        a = f"radians((1 - rest) * (mid + 0.5 * amp * cos({TAU} * {ph})) - rest * droop)"
        out[f"{wing}_L"], out[f"{wing}_R"] = f"-{a}", a
    return out


def poses(frames, frame0, fps, step=0.0, step_from=0.0):
    """The pose index of every frame (n,): the frame itself (from frame0) on every frame, or with `step` poses a second
    the step of the grid from clip second `step_from` it is held at (the paper theatre's grid)."""
    f = np.asarray(frames, float)
    if step <= 0.0:
        return f - frame0
    period = fps / step
    origin = round(frame0 + step_from * fps)
    return np.floor((f - origin + 0.5) / period)


# ===================================================================================================================
# the film: the cellophane's faint iridescence, printed (what a flat look draws) and turning with the view (lit)
# ===================================================================================================================
FILMS = 3                                          # film pictures the wings cycle through, one a pose (a whirr shimmers)


def ink_ramp(s, inks, soft=1.0):
    """Colours (k, 3) at s (k,) along the closed band of `inks` (m, 3): each ink in turn, wrapping; `soft` the share of
    each ink's stretch it blends into the next over (1: all of it, smoothly; smaller: clear bands)."""
    inks = np.asarray(inks, float)
    m = len(inks)
    x = (np.asarray(s, float) % 1.0) * m
    i = np.floor(x).astype(int) % m
    f = np.clip((x - np.floor(x) - 0.5) / max(soft, 1e-6) + 0.5, 0.0, 1.0)
    f = f * f * (3.0 - 2.0 * f)
    return inks[i] * (1.0 - f)[:, None] + inks[(i + 1) % m] * f[:, None]


def film(inks, base, strength=DEFAULTS["film"], size=256, variant=0):
    """The film's picture, linear RGB (size, size, 3), row 0 the top: bands of the inks across each wing with soft
    edges (the forewings' tile on top, the hind's below, set off from each other) over the cellophane's `base`,
    `strength` the share of ink at the wing's tip (a little less at its root), and a pale glint across it, as cellophane
    catches the light. `variant` 0 .. FILMS - 1 moves the bands on by a share of their round: the wings show the next
    one on every pose, so their colours shift as they whirr, in a flat look too."""
    v, u = np.mgrid[0:size, 0:size] / (size - 1.0)
    out = np.empty((size, size, 3))
    paper = np.ones(3)
    for k, (lo, hi) in enumerate(((0.0, 0.5), (0.5, 1.0))):        # rows: the top half is the forewings'
        rows = (v >= lo) & (v <= hi) if k == 0 else (v > lo)
        vv = (v[rows] - lo) / (hi - lo)
        uu = u[rows]
        s = (1.1 * uu + 0.5 * vv + 0.06 * np.sin(6.0 * uu + 3.0 * vv) + (0.0 if k == 0 else 0.37)
             + variant / (FILMS * len(inks)) * 1.6)
        c = ink_ramp(s, inks, soft=0.45)
        w = strength * (0.75 + 0.25 * uu)
        col = np.asarray(base, float) * (1.0 - w[:, None]) + c * w[:, None]
        x = uu - 0.58 - 0.35 * (vv - 0.5) + 0.12 * variant
        glint = 0.5 * np.exp(-((((x + 0.5) % 1.0) - 0.5) / 0.05) ** 2)
        out[rows] = col * (1.0 - glint[:, None]) + paper * glint[:, None]
    return out


# ===================================================================================================================
# options and the card
# ===================================================================================================================
def _number(slots, key, lo, hi):
    v = slots.get(key, DEFAULTS[key])
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
        raise ValueError(f"dragonfly: {key} is a number from {lo:g} to {hi:g} (got {v!r})")
    return float(v)


def options(slots):
    """The prop's options from its `slots`, checked: {size, whirr, amp, mid, lag, droop, step, step_from, wire, film,
    clear_wing, smile, hand}. Refuses what it cannot build, saying what to change."""
    slots = dict(slots or {})
    O = {"size": _number(slots, "size", 0.01, 10.0), "whirr": _number(slots, "whirr", 0.0, 0.5),
         "amp": _number(slots, "amp", 0.0, 160.0), "mid": _number(slots, "mid", -60.0, 60.0),
         "lag": _number(slots, "lag", -1.0, 1.0), "droop": _number(slots, "droop", -30.0, 30.0),
         "step": _number(slots, "step", 0.0, 240.0), "step_from": _number(slots, "step_from", -1e4, 1e4),
         "wire": _number(slots, "wire", 0.0, 0.05), "film": _number(slots, "film", 0.0, 1.0),
         "clear_wing": _number(slots, "clear_wing", 0.0, 1.0)}
    smile = slots.get("smile", DEFAULTS["smile"])
    if not isinstance(smile, bool):
        raise ValueError(f"dragonfly: smile is true or false (got {smile!r})")
    O["smile"] = smile
    hand = slots.get("hand", DEFAULTS["hand"])
    if not isinstance(hand, (list, tuple)) or len(hand) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in hand):
        raise ValueError(f"dragonfly: hand is [x, y, z], the wire's far end in the prop's frame (got {hand!r})")
    O["hand"] = tuple(float(v) for v in hand)
    inks = slots.get("inks", list(INKS))
    if not isinstance(inks, (list, tuple)) or len(inks) < 2 or not all(isinstance(c, str) for c in inks):
        raise ValueError(f"dragonfly: inks is a list of two or more colours, the film's bands (got {inks!r})")
    O["inks"] = [str(c) for c in inks]
    return O


def use_points(name, size, hand):
    """The card's use points (the prop's frame, metres): `look` head, body, tail; `anchor` belly (where the wire goes
    in, pointing down), perch (under the feet) and hand (the wire's far end: it rides `<name>_hand`, which a flight
    keys in the world)."""
    s = float(size)
    return {"look": [{"name": "head", "point": [0.0, round(-0.16 * s, 5), round(0.03 * s, 5)]},
                     {"name": "body", "point": [0.0, 0.0, 0.0]},
                     {"name": "tail", "point": [0.0, round(ABDOMEN[1] * s, 5), 0.0]}],
            "anchor": [{"name": "belly", "point": [round(v * s, 5) for v in BELLY], "dir": [0.0, 0.0, -1.0]},
                       {"name": "perch", "point": [0.0, round(-0.02 * s, 5), round(FEET_Z * s, 5)],
                        "dir": [0.0, 0.0, -1.0]},
                       {"name": "hand", "point": [round(float(v), 5) for v in hand], "object": f"{name}_hand"}]}


def bounds(size, amp, mid):
    """{min, max} in the prop's frame: the body and the wing tips through the stroke."""
    lo, hi = parts()["body"].bbox()
    up = math.radians(min(85.0, mid + 0.5 * amp))
    down = math.radians(max(-85.0, mid - 0.5 * amp))
    tip = max(float(outline(w)[:, 0].max()) + HINGES[w][0] for w in ("fore", "hind"))
    zs = [HINGES["fore"][2] + tip * math.sin(up), HINGES["fore"][2] + tip * math.sin(down), float(lo[2]), float(hi[2])]
    out_lo = [-tip, min(float(lo[1]), -0.1), min(zs)]
    out_hi = [tip, max(float(hi[1]), 0.15), max(zs)]
    return {"min": [round(v * size, 5) for v in out_lo], "max": [round(v * size, 5) for v in out_hi]}


def objects(name):
    """The prop's objects: the body, each wing (`fore_L` ..) and its film variants (`variants`: the wing itself
    showing film 0, then `<wing>_1`, `<wing>_2`, one shown a pose), the wire and its hand."""
    wings = {f"{w}_{s}": f"{name}_wing_{w}_{s}" for w in ("fore", "hind") for s in ("L", "R")}
    return {"body": f"{name}_body", "wings": wings,
            "variants": {k: [n] + [f"{n}_{v}" for v in range(1, FILMS)] for k, n in wings.items()},
            "wire": f"{name}_wire", "hand": f"{name}_hand"}


def card(name, O, roles):
    """The prop's card: size, bounds, use points, the colour roles, and `dragonfly` (what the flight stages key: the
    objects, the wire's radius, points and belly, the step grid, the whirr) and `wire` (a flown prop with one has its
    wire drawn)."""
    b = bounds(O["size"], O["amp"], O["mid"])
    ob = objects(name)
    return {"size": [round(h - lo, 5) for lo, h in zip(b["min"], b["max"])], "origin": "center", "front": "-Y",
            "blocks": False, "bounds": b, "layers": [], "colliders": [], "slots": dict(roles),
            "use": use_points(name, O["size"], O["hand"]),
            "wire": {"object": ob["wire"], "hand": ob["hand"], "radius": O["wire"], "points": 32,
                     "belly": [round(v * O["size"], 5) for v in BELLY], "rest_hand": list(O["hand"])},
            "dragonfly": {"size": O["size"], "objects": ob, "step": O["step"], "step_from": O["step_from"],
                          "whirr": O["whirr"], "params": {k: O[k] for k in ("amp", "mid", "lag", "droop")},
                          "perch": [0.0, round(-0.02 * O["size"], 5), round(FEET_Z * O["size"], 5)]}}
