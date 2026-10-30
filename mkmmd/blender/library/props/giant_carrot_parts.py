"""Meshes of the giant carrot (pure numpy, no bpy: importable from tests; the numbers are giant_carrot_layout.py).

  body      ONE lathe of the modelled profile from the rounded point to the shoulder and over the domed crown (N_AROUND
            points round each ring), the rings of the root cut in by faint transverse rings (`grooves`: shallow grooves that
            fade out round one side) and root-hair dimples in two ranks (`dimples`); denser rings where they are, and a ring
            at every sample of the grip's profile, so the card's profile is the mesh's. Its material is the body's (skin,
            flesh and core by the radius over the profile's, giant_carrot_mats.py).
  stages    `stage(k)`: the body's vertices after bites 1 .. k: every root vertex below bite k's rim is laid on its face,
            the tip in the middle and each ring further out (rho = R(rim) * z / rim): the shape keys' targets.
  greens    FRONDS fronds out of the crown: a tapering stem (`tube`) whose lean grows toward its tip, PAIRS pairs of leaflets
            and one at its end; a leaflet is a thin blade with rolled edges (a stadium section LEAF_T thick skinned along
            a drooping midrib, its half-width a lobed outline), cupped and turned a little along its length.
  chunks    `chunk_mesh`: a lumpy, flattened ball, its side toward +X the skin (the body's material), the rest broken flesh.
"""
import math

import numpy as np

from ....core import shell as S
from . import giant_carrot_layout as LAY

N_AROUND = 96               # points round a ring of the body
RING_STEP = 0.0025          # arc length between rings of the root (finer at the rings and dimples)
DOME_RINGS = 16
GROOVE_SEED, DIMPLE_SEED, GREENS_SEED = 3, 4, 5

FRONDS = 7                  # stems out of the crown, leaning out in turn a little and a lot
STEM_R = (0.0034, 0.0011)   # a stem's radius at its base and at its tip
STEM_SIDES = 10
PAIRS = 4                   # leaflet pairs on a frond, and one at its end
LEAF_L = 0.085              # the lowest pair's length (each pair up is 12 % shorter; the end leaflet 0.65 of it)
LEAF_T = 0.0016             # a leaflet's thickness: its rolled edge is that round
LEAF_STATIONS = 28
W_HALF = 0.17               # a leaflet's greatest half-width, share of its length
LOBES = 3.5                 # lobes along each side of a leaflet
LOBE_FLOOR = 0.3            # the notches between lobes go in to this share of the outline
MIDRIB = 0.7                # the blade is this much thicker along its midrib
DROOP = 0.16                # a leaflet's tip hangs this share of its length below its line
CUP = 0.035                 # its edges lift this share of its length (a cupped blade)
CARROT, CUT, STEM, LEAF = (LAY.MATS.index(m) for m in ("carrot", "cut", "stem", "leaf"))


# ================================================================================================= body
def grooves():
    """The faint transverse rings: [(z, depth as a share of R(z), sigma (m), angle where deepest, reach)]; a groove is
    full depth at its angle and fades out where cos(theta - angle) < -reach."""
    rng = np.random.RandomState(GROOVE_SEED)
    out, z = [], 0.03
    while True:
        z += rng.uniform(0.018, 0.042)
        if z > LAY.L - 0.035:
            return out
        out.append((z, rng.uniform(0.012, 0.026), rng.uniform(0.0012, 0.002), rng.uniform(0.0, 2.0 * math.pi),
                    rng.uniform(0.15, 0.6)))


def dimples():
    """The root-hair dimples, in two ranks on opposite sides: [(z, angle, radius (m), depth (m))]."""
    rng = np.random.RandomState(DIMPLE_SEED)
    out = []
    for th in (0.7, 0.7 + math.pi):
        for zc in (0.11, 0.21, 0.33, 0.46):
            z = zc + rng.uniform(-0.02, 0.02)
            out.append((z, th + rng.uniform(-0.25, 0.25), rng.uniform(0.0024, 0.0034), min(0.0009, 0.03 * LAY.radius(z))))
    return out


def ring_heights():
    """Heights of the root's rings, 0 (the tip's apex) .. L: RING_STEP apart along the profile, finer round every groove
    and dimple, and one at every sample of the grip's profile."""
    zz = np.linspace(0.0, LAY.L, 6001)
    rr = LAY.radius(zz)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(zz), np.diff(rr)))])
    base = np.interp(np.linspace(0.0, s[-1], math.ceil(s[-1] / RING_STEP) + 1), s, zz)
    marks = [(g[0], g[2]) for g in grooves()] + [(d[0], 0.5 * d[2]) for d in dimples()]
    keep = np.ones(len(base), bool)
    extra = [[0.0, LAY.L], [LAY.HOLD_FROM + v[0] for v in LAY.hold_profile()]]
    for zc, sg in marks:
        keep &= np.abs(base - zc) > 2.6 * sg
        extra.append(zc + sg * np.array([-2.6, -1.7, -0.85, 0.0, 0.85, 1.7, 2.6]))
    z = np.unique(np.concatenate([base[keep]] + [np.asarray(e, float) for e in extra]))
    z = z[(z >= 0.0) & (z <= LAY.L)]
    fixed = set(np.round(np.asarray(extra[1], float), 9)) | {0.0, LAY.L}
    out = [z[0]]
    for v in z[1:]:
        if v - out[-1] > 0.0004:
            out.append(v)
        elif round(v, 9) in fixed:                     # a sample that must stay: it wins over its close neighbour
            out[-1] = v
    return np.array(out)


def cut_in(z, theta):
    """How far (m) the grooves and dimples cut into the profile at heights z and angles theta (broadcast)."""
    z, theta = np.asarray(z, float), np.asarray(theta, float)
    R = LAY.radius(z)
    d = np.zeros(np.broadcast(z, theta).shape)
    for zc, depth, sg, th, reach in grooves():
        w = np.clip((np.cos(theta - th) + reach) / (1.0 + reach), 0.0, 1.0) ** 0.8
        d = d + depth * R * w * np.exp(-0.5 * ((z - zc) / sg) ** 2)
    for zc, th, rad, depth in dimples():
        dth = np.angle(np.exp(1j * (theta - th)))
        q = np.hypot(R * dth, z - zc) / rad
        d = d + depth * np.where(q < 1.0, (1.0 - q * q) ** 2, 0.0)
    return d


_BODY = None


def body():
    """(mesh, Z0, TH): the body (a lathe of `ring_heights` and the crown's dome, the root's rings cut in by `cut_in`), and
    per vertex its ring's height (NaN off the root: the dome and its top) and its angle."""
    global _BODY
    if _BODY is None:
        z = ring_heights()
        phi = np.linspace(0.0, 0.5 * math.pi, DOME_RINGS + 2)[1:-1]
        dr, dz = LAY.dome(phi)
        prof = ([(0.0, 0.0)] + [(float(LAY.radius(v)), float(v)) for v in z[1:]]
                + list(zip(dr.tolist(), dz.tolist())) + [(0.0, LAY.L + LAY.DOME_H)])
        m = S.lathe(np.array(prof), seg=N_AROUND, mat=CARROT)
        n_root = len(z) - 1
        th_ring = 2.0 * math.pi * np.arange(N_AROUND) / N_AROUND
        Z0 = np.full(len(m.V), np.nan)
        TH = np.zeros(len(m.V))
        Z0[0] = 0.0
        ring = 1 + np.arange(n_root * N_AROUND)
        Z0[ring] = np.repeat(z[1:], N_AROUND)
        TH[ring] = np.tile(th_ring, n_root)
        dome_ids = np.arange(1 + n_root * N_AROUND, len(m.V) - 1)
        TH[dome_ids] = np.tile(th_ring, len(phi))
        R = LAY.radius(Z0[ring])
        k = (R - cut_in(Z0[ring], TH[ring])) / R
        m.V[ring, :2] *= k[:, None]
        _BODY = (m, Z0, TH)
    return _BODY


def stage(k):
    """The body's vertices (n, 3) after bites 1 .. k (0: as modelled): every root vertex below bite k's rim is laid on its
    face at rho = R(rim) * z / rim, the tip's apex in the middle, the ring under the rim at its edge."""
    m, Z0, TH = body()
    V = m.V.copy()
    if k <= 0:
        return V
    idx = np.nonzero(np.isfinite(Z0))[0]
    th = TH[idx]
    zb = LAY.rim(k, th)
    eat = Z0[idx] < zb
    idx, th, zb = idx[eat], th[eat], zb[eat]
    u = Z0[idx] / zb
    rho = LAY.radius(zb) * u
    V[idx] = np.stack([rho * np.cos(th), rho * np.sin(th), LAY.face(k, u, th)], 1)
    return V


def key_deltas():
    """[stage(k) - stage(k - 1) for k = 1 .. BITES_MAX]: what each non-additive shape key adds to the basis."""
    st = [stage(k) for k in range(LAY.BITES_MAX + 1)]
    return [st[k] - st[k - 1] for k in range(1, LAY.BITES_MAX + 1)]


# ================================================================================================= greens
def _frames(path, up):
    """(T, U, B) along a path: the tangent, B the `up` direction made perpendicular to it, U = B x T (so a section drawn
    counter-clockwise in (U, B) faces along T, as `core.shell.sweep` has it)."""
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.asarray(up, float) - T * (T @ np.asarray(up, float))[:, None]
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return T, np.cross(B, T), B


def _stadium(c, rb, n_cap=5, n_face=3):
    """(x, y) (n, 2 n_cap + 2 n_face) of stadium sections, counter-clockwise: half-length c of the flat faces, round ends
    of radius rb (both (n,))."""
    c, rb = np.asarray(c, float)[:, None], np.asarray(rb, float)[:, None]
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_cap)[None, :]
    f = 1.0 - 2.0 * np.arange(1, n_face + 1)[None, :] / (n_face + 1)
    x = np.concatenate([c + rb * np.cos(a), c * f, -c - rb * np.cos(a), -c * f], 1)
    flat = np.repeat(rb, n_face, axis=1)
    y = np.concatenate([rb * np.sin(a), flat, -rb * np.sin(a), -flat], 1)
    return x, y


def leaflet_half_width(u):
    """A leaflet's half-width at u (0 its base .. 1 its tip) as a share of its length: an envelope round in the middle,
    LOBES round lobes along it with notches in to LOBE_FLOOR between them."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    env = np.sin(math.pi * u ** 0.8) ** 0.6
    lob = LOBE_FLOOR + (1.0 - LOBE_FLOOR) * np.abs(np.sin(math.pi * LOBES * u)) ** 0.5
    return W_HALF * env * lob


def leaflet(base, d, up, length, twist=0.0):
    """One leaflet from `base` (on its stem's axis) along the unit direction d, its upper face toward `up`: a closed blade."""
    u = np.linspace(0.0, 1.0, LEAF_STATIONS)
    path = (np.asarray(base, float) + length * u[:, None] * np.asarray(d, float)
            + length * DROOP * (u * u)[:, None] * np.array([0.0, 0.0, -1.0]))
    _, U, B = _frames(path, up)
    w = np.maximum(length * leaflet_half_width(u), 0.5 * LEAF_T + 0.0003)
    rb = np.minimum(0.5 * LEAF_T, w - 0.0003)
    x, y = _stadium(w - rb, rb)
    y = y * (1.0 + MIDRIB * np.exp(-(x / (0.3 * (w - rb)[:, None] + 1e-5)) ** 2))       # thicker along the midrib
    y = y + CUP * length * (x / (W_HALF * length)) ** 2
    tw = math.radians(twist) * u[:, None]
    x, y = x * np.cos(tw) - y * np.sin(tw), x * np.sin(tw) + y * np.cos(tw)
    P = path[:, None, :] + x[..., None] * U[:, None, :] + y[..., None] * B[:, None, :]
    return S.skin(P, caps=(True, True), mat=LEAF)


def fronds():
    """[{az, lean0, lean1 (deg), top (z of the stem's tip), base (3,)}] of the fronds out of the crown."""
    rng = np.random.RandomState(GREENS_SEED)
    out = []
    for f in range(FRONDS):
        az = 2.0 * math.pi * f / FRONDS + 0.3 + rng.uniform(-0.25, 0.25)
        rb = 0.005 + 0.004 * rng.rand()
        outer = f % 2 == 0
        out.append({"az": az, "lean0": rng.uniform(6.0, 13.0), "lean1": (50.0 if outer else 26.0) + rng.uniform(-6.0, 6.0),
                    "top": LAY.L + LAY.GREENS_H * ((0.76 if outer else 0.9) + 0.1 * rng.rand()) - 0.02,
                    "base": np.array([rb * math.cos(az), rb * math.sin(az), LAY.L + LAY.DOME_H - 0.012]),
                    "twist": rng.uniform(-20.0, 20.0)})
    return out


def frond(fr):
    """One frond: its stem (a tapering tube whose lean grows toward its tip) and its leaflets."""
    n = 48
    s = np.linspace(0.0, 1.0, n)
    lam = np.radians(fr["lean0"] + (fr["lean1"] - fr["lean0"]) * s ** 1.5)
    az = fr["az"]
    dirs = np.stack([np.sin(lam) * math.cos(az), np.sin(lam) * math.sin(az), np.cos(lam)], 1)
    unit = np.concatenate([[[0.0, 0.0, 0.0]], np.cumsum(0.5 * (dirs[1:] + dirs[:-1]) / (n - 1), axis=0)])
    path = fr["base"] + unit * (fr["top"] - fr["base"][2]) / unit[-1, 2]
    parts = [S.tube(path, STEM_R[0] + (STEM_R[1] - STEM_R[0]) * s ** 0.8, sides=STEM_SIDES, mat=STEM)]
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    side = np.array([-math.sin(az), math.cos(az), 0.0])
    for q in range(PAIRS):
        for sgn in (1.0, -1.0):
            sp = 0.42 + 0.13 * q + 0.018 * sgn
            i = sp * (n - 1)
            i0 = min(int(i), n - 2)
            f = i - i0
            p = (1.0 - f) * path[i0] + f * path[i0 + 1]
            t = (1.0 - f) * T[i0] + f * T[i0 + 1]
            t /= np.linalg.norm(t)
            up = np.cross(t, side)
            up /= np.linalg.norm(up)
            beta = math.radians(58.0 - 6.0 * q)
            d = math.cos(beta) * t + math.sin(beta) * sgn * side
            parts.append(leaflet(p, d / np.linalg.norm(d), up, LEAF_L * (1.0 - 0.12 * q), fr["twist"] * sgn))
    up = np.cross(T[-1], side)
    parts.append(leaflet(path[-1] - 0.004 * T[-1], T[-1], up / np.linalg.norm(up), 0.65 * LEAF_L, fr["twist"]))
    return S.merge(parts)


def greens_mesh():
    """Every frond: the stems (material `stem`) and the leaflets (`leaf`)."""
    return S.merge([frond(fr) for fr in fronds()])


# ================================================================================================= chunks
def chunk_mesh(size, seed):
    """A bitten-off piece about `size` across: a lumpy, flattened ball broken off along two or three planes (flat facets of
    broken flesh, `cut`), its rounded side toward +X the skin (`carrot`)."""
    rng = np.random.RandomState(seed)
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 10)
    m = S.lathe(np.stack([np.cos(a), np.sin(a)], 1), seg=14, mat=CUT)
    D = m.V / np.linalg.norm(m.V, axis=1, keepdims=True)
    f = np.ones(len(D))
    for _ in range(4):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        f = f + rng.uniform(0.06, 0.14) * np.cos(rng.uniform(2.0, 4.0) * (D @ ax) + rng.uniform(0.0, 2.0 * math.pi))
    V = D * f[:, None] * (0.5 * size) * np.array([1.0, 0.8, 0.62])
    for _ in range(rng.randint(2, 4)):                       # the breaks: everything past a plane laid on it
        n = rng.normal(size=3) - np.array([1.6, 0.0, 0.0])    # away from the skin side
        n /= np.linalg.norm(n)
        d = rng.uniform(0.18, 0.3) * size
        over = V @ n - d
        V = V - np.maximum(over, 0.0)[:, None] * n
    m.V = V
    return m.assign(CARROT, lambda c, nrm: nrm[:, 0] > 0.45)


# ================================================================================================= size
def size():
    """[x, y, z] extent of the carrot as modelled (the body and the greens; the chunks are inside it until they fly)."""
    V = np.concatenate([body()[0].V, greens_mesh().V])
    return (V.max(0) - V.min(0)).tolist()
