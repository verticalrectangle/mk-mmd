"""Layout of the carrot plush (`card = "library:carrot_plush"`; pure Python + numpy, no bpy: importable from tests): a
soft toy carrot, its meshes, the colour roles and the card.

Frame (metres): the origin at the middle of the body on its axis, +Z from the tip (Z_TIP) to the top (Z_TOP: where the
leaves come out, BODY_L = 0.34 from the tip), the face toward -Y. The body is a plump rounded cone with a soft rounded tip
and a round top (`radius(z)`), sewn from GORES fabric gores: a seam valley runs down between two gores (`gores(theta)`),
puffy in between; the face sits on the front gore and a seam runs down the back. Three plump felt leaves (`LEAVES`, about
0.12 m above the top) come out of it in the plane of the face, each a stuffed lens with a stitched midrib (a groove on both
faces, the stitches a dashed thread in it). The embroidered face (`face`, `cheeks`) is thin graphic layers laid on the
body: two oval eyes with a highlight, a small smile, pink cheeks.

Colour roles (`slots`: a palette slot or a hex): body (true carrot orange), leaf (green), face (the dark thread: Dawn
`text`), shine (the eyes' highlights: `surface`), cheek (`love`), stitch (the midribs' thread: `base`).

Card: `use.anchor` `hold` (the middle of its back, where an arm holds it to a side or a chest; `dir` the way its face looks,
-Y) and `use.look` `face` (between the eyes)."""
import math
import re

import numpy as np

from ....core import palette as PAL
from ....core import shell as S

BODY_L = 0.34               # tip -> top
Z_TIP, Z_TOP = -0.5 * BODY_L, 0.5 * BODY_L
R_MAX = 0.072               # the widest radius (a plump toy, 0.144 across)
Z_WIDE = 0.075              # where it is widest
TIP_Q = 1.8                 # the lower part: r = R_MAX sqrt(1 - x ** TIP_Q), x from the widest (0) to the tip (1)
DOME_P = 2.3                # the top: a superellipse quadrant from the widest up to Z_TOP
GORES = 5
SEAM = 0.07                 # a seam valley's depth, share of the radius
SEAM_P = 3.0                # how narrow it is
N_AROUND = 24 * GORES
RING_STEP = 0.003
# leaves (about 0.12 above the top): (tilt in the face's plane (deg, + toward +X), lean back (deg, + toward +Y), length,
# half-width)
LEAVES = ((0.0, 8.0, 0.14, 0.025), (-36.0, 2.0, 0.115, 0.022), (36.0, 2.0, 0.115, 0.022))
LEAF_HALF_T = 0.0085        # a leaf's greatest half-thickness (stuffed)
LEAF_P = 2.4                # its section: a superellipse, a little fuller than a lens
MIDRIB_DIP = 0.45           # the midrib's groove takes this share of the thickness
LEAF_STATIONS, LEAF_POINTS = 30, 28
STITCH = (0.005, 0.0006, 0.008)    # a stitch's length, radius, and the pitch along the midrib
# the face, on the front gore (theta = -90 deg), positions (s across the surface (+ toward +X), z) in metres
EYE = ((0.027, 0.05), (0.0085, 0.012))          # an eye's centre (mirrored) and semi-axes
SHINE = ((-0.0028, 0.0052), (0.0027, 0.0033))   # a highlight's offset from its eye's centre and semi-axes
SMILE = ((0.0, 0.034), 0.015, (-130.0, -50.0), 0.0015)     # its arc's centre, radius, angles (deg) and thread radius
CHEEK = ((0.037, 0.026), (0.0095, 0.0065))      # a cheek's centre (mirrored) and semi-axes
LIFT = 0.0003               # the layers float this far off the body
EYE_T, SHINE_T, CHEEK_T = 0.0011, 0.0005, 0.0005    # their thickness
FRONT = -0.5 * math.pi      # the face's angle round the axis

ROLES = {"body": "#ee7a24", "leaf": "#4e9a3a", "face": "text", "shine": "surface", "cheek": "love", "stitch": "base"}
MATS = tuple(ROLES)         # material index -> role
SLOTS = tuple(PAL.PALETTES["rose-pine-dawn"])
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ================================================================================================= the body
def radius(z):
    """The body's radius at height z (before the gores' seams), Z_TIP <= z <= Z_TOP."""
    z = np.asarray(z, float)
    x = np.clip((Z_WIDE - z) / (Z_WIDE - Z_TIP), 0.0, 1.0)
    low = R_MAX * np.sqrt(np.clip(1.0 - x ** TIP_Q, 0.0, 1.0))
    q = np.clip((z - Z_WIDE) / (Z_TOP - Z_WIDE), 0.0, 1.0)
    top = R_MAX * np.clip(1.0 - q ** DOME_P, 0.0, 1.0) ** (1.0 / DOME_P)
    return np.where(z < Z_WIDE, low, top)


def profile():
    """(r, z) of the body's outline from the tip to the top, RING_STEP apart along it (the lathe's profile)."""
    a = np.linspace(0.0, 0.5 * math.pi, 400)
    lo_z = Z_TIP + (Z_WIDE - Z_TIP) * (1.0 - np.cos(a))                 # dense at the tip, where the radius turns fast
    lo = np.stack([radius(lo_z), lo_z], 1)
    hi = np.stack([R_MAX * np.cos(a) ** (2.0 / DOME_P), Z_WIDE + (Z_TOP - Z_WIDE) * np.sin(a) ** (2.0 / DOME_P)], 1)
    P = np.concatenate([lo, hi[1:]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], math.ceil(s[-1] / RING_STEP) + 1)
    out = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)
    out[0], out[-1] = (0.0, Z_TIP), (0.0, Z_TOP)
    return out


def gores(theta):
    """The share of the radius kept at angle theta: 1 in the middle of a gore, 1 - SEAM in a seam's valley; the seams at
    FRONT + pi / GORES + 2 pi k / GORES (one down the back), so the face's gore is centred on the front."""
    s = np.abs(np.sin(0.5 * GORES * (np.asarray(theta, float) - FRONT) + 0.5 * math.pi))
    return 1.0 - SEAM * (1.0 - s) ** SEAM_P


def body_mesh():
    """The body (material `body`): a lathe of `profile`, every ring pulled in to the gores' seams."""
    m = S.lathe(profile(), seg=N_AROUND, mat=MATS.index("body"))
    th = np.arctan2(m.V[:, 1], m.V[:, 0])
    m.V[:, :2] *= gores(th)[:, None]
    return m


def surface(theta, z):
    """(points (n, 3), unit outward normals (n, 3)) of the body at angles theta and heights z (arrays)."""
    theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))

    def at(t, h):
        r = radius(h) * gores(t)
        return np.stack([r * np.cos(t), r * np.sin(t), h], -1)

    e = 1e-5
    P = at(theta, z)
    n = np.cross(at(theta + e, z) - at(theta - e, z), at(theta, z + e) - at(theta, z - e))
    return P, n / np.linalg.norm(n, axis=-1, keepdims=True)


def on_body(s, z):
    """The body's point and normal at (s across the front gore, + toward +X, z): theta = FRONT + s / r."""
    s, z = np.asarray(s, float), np.asarray(z, float)
    return surface(FRONT + s / (radius(z) * gores(FRONT)), z)


# ================================================================================================= the face
def _layer(outline, centre, thick, lift, mat):
    """A thin closed layer (a `rounded_panel` of `outline` (m, 2) metres, `thick` thick) laid on the body at `centre` (s, z):
    every point moved onto the surface and along its normal by lift + its height in the panel."""
    p = S.rounded_panel(outline, thick, r_edge=0.45 * thick, n=2, r_back=0.3 * thick, mat=mat)
    P, N = on_body(centre[0] + p.V[:, 0], centre[1] + p.V[:, 1])
    p.V = P + N * (lift + p.V[:, 2:3])
    return p


def face_mesh():
    """The embroidered face: two oval eyes (material `face`) with a highlight each (`shine`) and the smile (`face`)."""
    (ex, ez), (ea, eb) = EYE
    (hx, hz), (ha, hb) = SHINE
    parts = []
    for sgn in (1.0, -1.0):
        parts.append(_layer(S.ellipse(ea, eb, 40), (sgn * ex, ez), EYE_T, LIFT, MATS.index("face")))
        parts.append(_layer(S.ellipse(ha, hb, 24), (sgn * ex + hx, ez + hz), SHINE_T, LIFT + EYE_T,
                            MATS.index("shine")))
    (cx, cz), r, (a0, a1), tr = SMILE
    a = np.radians(np.linspace(a0, a1, 24))
    P, N = on_body(cx + r * np.cos(a), cz + r * np.sin(a))
    path = P + N * (LIFT + 0.6 * tr)
    taper = np.minimum(1.0, 0.35 + 0.65 * np.sin(np.linspace(0.0, math.pi, 24)) * 3.0)
    parts.append(S.tube(path, tr * taper, sides=10, up=tuple(N[12]), aspect=0.6, mat=MATS.index("face")))
    return S.merge(parts)


def cheeks_mesh():
    """The two pink cheeks (material `cheek`): thin felt ovals."""
    (cx, cz), (ca, cb) = CHEEK
    return S.merge([_layer(S.ellipse(ca, cb, 36), (sgn * cx, cz), CHEEK_T, LIFT, MATS.index("cheek"))
                    for sgn in (1.0, -1.0)])


# ================================================================================================= the leaves
def _leaf_frame(tilt, lean, length):
    """(path (n, 3), T, U, B) of a leaf's midrib: from inside the top, out along its tilt and lean, bending out a little; U
    across the leaf, B its front face's normal (toward -Y)."""
    u = 0.5 - 0.5 * np.cos(math.pi * np.linspace(0.0, 1.0, LEAF_STATIONS))      # closer together at both ends
    t, b = math.radians(tilt), math.radians(lean)
    d0 = np.array([math.sin(t) * math.cos(b), math.sin(b), math.cos(t) * math.cos(b)])
    out = np.array([math.sin(t), 0.0, 0.0]) * 0.18 + np.array([0.0, 0.25 * math.sin(b), 0.0])
    path = (np.array([0.0, 0.0, Z_TOP - 0.016]) + length * u[:, None] * d0
            + length * (u * u)[:, None] * out[None, :])
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.array([0.0, -1.0, 0.0]) - T * (T @ np.array([0.0, -1.0, 0.0]))[:, None]
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return u, path, T, np.cross(B, T), B


def leaf_width(u):
    """A leaf's half-width as a share of its greatest: narrow where it is sewn in, round and full, a rounded tip."""
    return np.sin(math.pi * (0.04 + 0.96 * np.clip(u, 0.0, 1.0) ** 0.85)) ** 0.5


def leaf_mesh(tilt, lean, length, half_w):
    """One stuffed felt leaf (material `leaf`): superellipse sections, plump in the middle, the midrib's groove on both
    faces."""
    u, path, _, U, B = _leaf_frame(tilt, lean, length)
    w = np.maximum(half_w * leaf_width(u), 0.0015)
    h = np.maximum(LEAF_HALF_T * np.sin(math.pi * (0.04 + 0.96 * u) ** 0.8) ** 0.6, 0.001)
    a = 2.0 * math.pi * np.arange(LEAF_POINTS) / LEAF_POINTS
    c, s = np.cos(a), np.sin(a)
    x = np.sign(c) * np.abs(c) ** (2.0 / LEAF_P)
    y = np.sign(s) * np.abs(s) ** (2.0 / LEAF_P) * (1.0 - MIDRIB_DIP * np.exp(-(x / 0.13) ** 2))
    P = path[:, None, :] + (w[:, None] * x)[..., None] * U[:, None, :] + (h[:, None] * y)[..., None] * B[:, None, :]
    return S.skin(P, caps=(True, True), mat=MATS.index("leaf"))


def leaves_mesh():
    return S.merge([leaf_mesh(*lv) for lv in LEAVES])


def stitches_mesh():
    """The midribs' stitches (material `stitch`): short thread dashes along the groove of both faces of every leaf."""
    parts = []
    for tilt, lean, length, _ in LEAVES:
        u, path, _, _, B = _leaf_frame(tilt, lean, length)
        h = np.maximum(LEAF_HALF_T * np.sin(math.pi * (0.04 + 0.96 * u) ** 0.8) ** 0.6, 0.001)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        ln, rad, pitch = STITCH
        for side in (1.0, -1.0):
            depth = side * (h * (1.0 - MIDRIB_DIP) + 0.35 * rad)
            for s0 in np.arange(0.18 * s[-1], 0.84 * s[-1], pitch):
                q = np.linspace(s0, s0 + ln, 6)
                p = np.stack([np.interp(q, s, path[:, k] + depth * B[:, k]) for k in range(3)], 1)
                parts.append(S.tube(p, rad * np.array([0.6, 0.95, 1.0, 1.0, 0.95, 0.6]), sides=6,
                                    up=tuple(B[len(B) // 2] * side), mat=MATS.index("stitch")))
    return S.merge(parts)


# ================================================================================================= options and card
def options(slots):
    """{role: slot or hex} from the `slots` of the [[prop]] (keys this prop does not know are ignored), checked."""
    slots = dict(slots or {})
    roles = {}
    for role, default in ROLES.items():
        v = slots.get(role, default)
        if not isinstance(v, str) or not (v in SLOTS or HEX.match(v)):
            raise ValueError(f"carrot_plush: {role} is a palette slot ({', '.join(SLOTS)}) or a hex \"#rrggbb\", got {v!r}")
        roles[role] = v
    return roles


def card(name, roles, size):
    """The prop card: `use.anchor` hold (the middle of its back, `dir` the way the face looks) and `use.look` face."""
    back, _ = surface(np.array([0.5 * math.pi]), np.array([0.0]))
    look, _ = on_body(np.array([0.0]), np.array([EYE[0][1]]))
    r = lambda v: [round(float(c), 5) for c in v]                                    # noqa: E731
    return {"size": [round(float(v), 4) for v in size], "origin": "center", "front": "-Y", "slots": dict(roles),
            "use": {"anchor": [{"name": "hold", "point": r(back[0]), "dir": [0.0, -1.0, 0.0]}],
                    "look": [{"name": "face", "point": r(look[0])}]},
            "colliders": [], "params": {},
            "objects": [f"{name}_body", f"{name}_leaves", f"{name}_face", f"{name}_cheeks", f"{name}_stitches"]}
