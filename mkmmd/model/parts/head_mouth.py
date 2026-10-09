"""The mouth: a slit in the skin (closed at rest: the upper and lower lip margins are separate vertices at the same
place), folded into the face as a notch (seen from the side the upper lip ends in an edge over the mouth line and the
lower lip comes out under it), the mouth interior behind it (a bag that follows the lips), upper teeth with a small fang,
a tongue, the lip lines, and the mouth morphs as deformation fields of the lips.

Head-local metres (see head_shape). The slit is centred on x = 0 at height `z`; the upper lip is above it, the lower below."""
import numpy as np

DEFAULTS = dict(
    z=0.0010,                      # height of the closed mouth line over the head bone
    half_width=0.0120,             # half length of the closed mouth
    smile=0.0020,                  # corners lifted over the middle of the closed line
    block_margin=0.0035,           # grid margin beyond the corners
    ring_t=(0.0, 0.42, 0.78, 1.0),  # rings 0..3 between the grid block and the slit
    fold=0.0040,                   # the slit lies this far back at the middle (1 - (x / half_width)^2)^2 of it sideways
    fold_share=(0.0, 0.05, 0.40, 1.0),  # the share of the fold rings 0..3 take: the lips curl into the mouth line
    line_width=0.0009,            # width of the lip line strips
)


def fold_depth(cfg, x):
    """How far back the closed mouth line lies at x (the notch: deepest in the middle, gone at the corners)."""
    s = np.asarray(x, float) / cfg["half_width"]
    return cfg["fold"] * np.clip(1.0 - s * s, 0.0, None) ** 2


def slit_z(cfg, x):
    s = np.asarray(x, float) / cfg["half_width"]
    return cfg["z"] + cfg["smile"] * np.minimum(s * s, 2.0)


def add_mouth(sb, cfg=None):
    """Cut the mouth slit into the SkinBuilder. The block is 2 rows tall; ring 3 is the closed slit: its lower-lip
    vertices (bottom row), right corner, upper-lip vertices (top row, right to left) and left corner. Returns a dict with the
    ring ids, the index sets and the parameters."""
    from .head_skin import lift

    c = dict(DEFAULTS)
    c.update(cfg or {})
    g = sb.g
    hw = c["half_width"]
    nf = int(np.sum((g.theta > 0) & (g.theta < np.radians(60.0))))
    cols = range(-nf, nf + 1)
    # block: two rows around the mouth (the midline row nearest the mouth point on the face: the midline's height is not
    # monotonic below the chin, where it comes back up the throat), columns covering the slit plus a margin
    target = lift(sb.shape, np.zeros(1), np.array([c["z"]]))[0]
    jm = int(np.argmin([np.linalg.norm(sb.V[g.A[j, 0]] - target) for j in range(g.nA)]))
    xs0 = np.array([sb.V[g.A[jm, k % g.n]][0] for k in cols])
    c1 = int(cols[np.min(np.where(xs0 >= hw + c["block_margin"])[0])])
    c0 = -c1
    j0, j1 = jm - 1, jm + 1
    ring0 = sb.carve(c0, c1, j0, j1)
    w = c1 - c0
    n = len(ring0)                                   # 2 (w + 1) + 2
    P0 = np.array([sb.V[i] for i in ring0])
    # parameters of the slit points: bottom row m -> s from the left corner, spaced like the block's columns (so the ring
    # lines from the block to the slit never cross where the grid is finer in the middle); the top row the same
    xb = P0[:w + 1, 0]
    U = float(np.abs(xb).max()) + float(np.abs(np.diff(xb[[0, 1]])[0]))
    s_low = 0.5 + 0.5 * xb / U
    s_up = s_low[::-1]
    s_ring = np.concatenate([s_low, [1.0], s_up, [0.0]])
    tx = -hw + s_ring * 2 * hw
    tz = slit_z(c, tx)
    rings = [ring0]
    for k in (1, 2, 3):
        t = c["ring_t"][k]
        x = P0[:, 0] + t * (tx - P0[:, 0])
        z = P0[:, 2] + t * (tz - P0[:, 2])
        P = lift(sb.shape, x, z)
        P[:, 1] += c["fold_share"][k] * fold_depth(c, x)            # the lips curl in to the folded mouth line
        rings.append(sb.add_verts(P))
    for k in range(3):
        sb.strip(rings[k], rings[k + 1], sb.SKIN)
    return dict(rings=rings, n=n, w=w, block=(c0, c1, j0, j1), cfg=c,
                lower=np.arange(0, w + 1), right=w + 1, upper=np.arange(w + 2, 2 * w + 3), left=2 * w + 3,
                x=tx, z=tz)


# -------------------------------------------------------------------------------------------------- the interior
INTERIOR = dict(
    depth=(0.0, 0.0075, 0.0160, 0.0225),            # y behind the lips of the front ring, ring B, ring C and the cap
    a=(0.0, 0.0140, 0.0112, 0.0), b=(0.0, 0.0100, 0.0078, 0.0),    # half width / half height of those rings
    follow=(1.0, 0.60, 0.22, 0.0),                    # how much of the lips' motion each ring takes
)


def cavity(m, V, cfg=None, shape=None, behind=0.0025):
    """Interior bag: rings A (on the slit, behind the lips), B, C and a cap. `m`: the dict of add_mouth with the final skin
    vertex ids in m["ids"] (ring 3, in ring order) and V the skin vertices; with `shape`, rings B, C and the cap stay at least
    `behind` behind the skin in front of them (a chin that recedes fast under the mouth would otherwise show the bag through
    it). Returns (verts, faces, uv, follow, ring_of_vertex), where `follow[i]` is the vertex's share of the lip motion and
    `m["ring3_row"][i]` the ring-3 column it follows."""
    from .head_skin import lift

    def keep_behind(P):
        if shape is not None:             # the line from the front ends inside, at the mouth's depth: under the chin it
            yb = float(np.mean(V[m["ids3"]][:, 1])) + 0.03            # would go on through the throat into the neck
            P[:, 1] = np.maximum(P[:, 1], lift(shape, P[:, 0], P[:, 2], y_back=yb)[:, 1] + behind)
        return P
    c = dict(INTERIOR)
    c.update(cfg or {})
    ring3 = V[m["ids3"]]
    n = m["n"]
    w = m["w"]
    z_m = m["cfg"]["z"]
    # angle of each ring-3 column on the loop: bottom row from pi to 2pi (left to right), right corner at 0, top row back
    th = np.empty(n)
    th[m["lower"]] = np.pi + np.pi * (np.arange(w + 1) + 1.0) / (w + 2)
    th[m["right"]] = 2 * np.pi
    th[m["upper"]] = np.pi * (np.arange(w + 1) + 1.0) / (w + 2)            # top row, right to left: from 0+ to pi-
    th[m["left"]] = np.pi
    verts = [ring3 + np.array([0.0, c["depth"][0] + 0.0010, 0.0])]
    follow = [np.full(n, c["follow"][0])]
    cols = [np.arange(n)]
    y0 = float(np.mean(ring3[:, 1]))
    uv = [np.stack([th / (2 * np.pi), np.full(n, 0.0)], -1)]
    for k in (1, 2):
        a, b = c["a"][k], c["b"][k]
        P = keep_behind(np.stack([a * np.cos(th), np.full(n, y0 + c["depth"][k]), z_m + b * np.sin(th)], -1))
        verts.append(P)
        follow.append(np.full(n, c["follow"][k]))
        cols.append(np.arange(n))
        uv.append(np.stack([th / (2 * np.pi), np.full(n, c["depth"][k] / c["depth"][3])], -1))
    cap = keep_behind(np.array([[0.0, y0 + c["depth"][3], z_m]]))
    verts.append(cap)
    follow.append(np.zeros(1))
    cols.append(np.zeros(1, int))
    uv.append(np.array([[0.5, 1.0]]))
    V_all = np.vstack(verts)
    F = []
    for k in range(3 - 1):
        a0, b0 = k * n, (k + 1) * n
        for i in range(n):
            i1 = (i + 1) % n
            F.append((a0 + i, a0 + i1, b0 + i1, b0 + i))
    base = 2 * n
    cap_i = 3 * n
    for i in range(n):
        i1 = (i + 1) % n
        F.append((base + i, base + i1, cap_i))
    return V_all, F, np.vstack(uv), np.concatenate(follow), np.concatenate(cols)


# ------------------------------------------------------------------------------------------------ teeth and tongue
TEETH = dict(half=0.0108, n=25, height=0.0029, fang=0.0017, fang_x=0.0066, fang_w=0.0008, back=0.0036)
TONGUE = dict(centre=(0.0, 0.0105, -0.0042), semi=(0.0075, 0.0072), tilt_deg=14.0, rings=3, spokes=12)


def teeth_geometry(A, m, cfg=None):
    c = dict(TEETH)
    c.update(cfg or {})
    up_idx = np.concatenate([[m["left"]], m["upper"][::-1], [m["right"]]])          # left corner -> right corner
    xs_u = A[up_idx, 0]
    o = np.argsort(xs_u)
    xs = np.linspace(-c["half"], c["half"], c["n"])
    top = np.stack([np.interp(xs, xs_u[o], A[up_idx, k][o]) for k in range(3)], -1)
    s = xs / c["half"]
    h = c["height"] * (0.58 + 0.42 * np.sqrt(np.clip(1.0 - s * s, 0.0, 1.0)))
    for sx in (-1.0, 1.0):
        h = h + c["fang"] * np.exp(-(((xs - sx * c["fang_x"]) / c["fang_w"]) ** 2))
    bot = top + np.stack([np.zeros_like(xs), np.full_like(xs, c["back"]), -h], -1)
    V = np.empty((2 * len(xs), 3))
    V[0::2], V[1::2] = top, bot
    F = []
    for k in range(len(xs) - 1):
        F.append((2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2))
    return V, F, xs


def tongue_geometry(A, m, cfg=None):
    c = dict(TONGUE)
    c.update(cfg or {})
    y0 = float(np.mean(A[:, 1]))
    cx, cy, cz = c["centre"]
    a, b = c["semi"]
    ph = np.radians(c["tilt_deg"])
    tdir = np.array([0.0, np.cos(ph), np.sin(ph)])
    ctr = np.array([cx, y0 + cy, A[:, 2].mean() + cz])
    P = [ctr]
    for k in range(1, c["rings"] + 1):
        r = k / c["rings"]
        for q in range(c["spokes"]):
            t = 2 * np.pi * q / c["spokes"]
            u, v = r * a * np.cos(t), r * b * np.sin(t)
            lift = 0.0006 * (1 - r * r)
            P.append(ctr + np.array([u, 0.0, 0.0]) + v * tdir + lift * np.array([0.0, -np.sin(ph), np.cos(ph)]))
    P = np.array(P)
    ns = c["spokes"]
    ring = lambda k: 1 + (k - 1) * ns
    F = [(0, ring(1) + q, ring(1) + (q + 1) % ns) for q in range(ns)]
    for k in range(1, c["rings"]):
        for q in range(ns):
            q1 = (q + 1) % ns
            F.append((ring(k) + q, ring(k + 1) + q, ring(k + 1) + q1, ring(k) + q1))
    # outward = up and forward (-y): flip if needed
    n = np.cross(P[F[0][1]] - P[F[0][0]], P[F[0][2]] - P[F[0][0]])
    want = np.array([0.0, -np.sin(ph), np.cos(ph)])
    if n @ want < 0:
        F = [tuple(reversed(f)) for f in F]
    return P, F


# ---------------------------------------------------------------------------------------------------- lip lines
def lip_line_edges(cfg, n=22, width=None, nudge=0.00018):
    """Edges (lo, hi) in the front plane (n, 2) of the upper and lower lip line strips along the closed slit; pointed at
    both corners. Returns {"up": (lo, hi), "low": (lo, hi)}."""
    c = dict(DEFAULTS)
    c.update(cfg or {})
    width = c["line_width"] if width is None else width
    hw = c["half_width"] * 0.985
    x = np.linspace(-hw, hw, n)
    z = slit_z(c, x)
    t = np.linspace(0.0, 1.0, n)
    s_ = 2 * t - 1
    taper = (0.55 + 0.95 * np.abs(s_) ** 2.5) * np.sin(np.pi * t) ** 0.35      # heavier at the corners, pointed ends
    up_lo = np.stack([x, z + nudge], -1)
    up_hi = np.stack([x, z + nudge + width * taper], -1)
    lo_hi = np.stack([x, z - nudge], -1)
    lo_lo = np.stack([x, z - nudge - width * taper], -1)
    return {"up": (up_lo, up_hi), "low": (lo_lo, lo_hi)}


# ------------------------------------------------------------------------------------------------- the lip field
def _sm(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lip_field(P, side, cfg, up=0.0, dn=0.0, p=0.9, wx=0.0, cl=0.0, nar=0.0, pro=0.0, wave=0.0, drop=0.0, R=0.014):
    """Skin offsets (n, 3) for a mouth shape. `P` rest positions, `side` (n,): +1 upper-lip margin vertices, -1 lower-lip
    margin vertices, 0 elsewhere. up / dn: how far the upper lip rises and the lower lip falls at the middle; p: the
    opening's profile exponent (smaller = rounder); wx: the corners move out by this much; cl: the corners rise; nar: the
    mouth narrows (fraction pulled in at the corners); pro: protrusion forward; wave: the closed line takes an omega
    shape (amplitude); drop: the whole lower face goes down."""
    c = dict(DEFAULTS)
    c.update(cfg or {})
    hw, zm = c["half_width"], c["z"]
    x, z = P[:, 0], P[:, 2]
    d = z - slit_z(c, x)
    d = np.where(side > 0, np.maximum(d, 1e-9), np.where(side < 0, np.minimum(d, -1e-9), d))
    s = x / hw
    env = np.clip(1.0 - s * s, 0.0, 1.0) ** p
    gu = _sm(1.0 - d / R)
    gl = _sm(1.0 + d / R)
    dz = np.where(d >= 0, up * env * gu, -dn * env * gl)
    if wave:
        a = np.abs(s)
        corner = 1.5 * np.minimum(a, 1.2) ** 4 * _sm((2.4 - a) / 1.2)       # the corners rise; fades out over the cheek
        f = 1.0 * np.exp(-(s / 0.2) ** 2) - 1.7 * np.exp(-((a - 0.55) / 0.26) ** 2) + corner
        dz = dz + wave * f * _sm(1.0 - np.abs(d) / R)
    r = np.hypot(np.abs(x) - hw, z - zm)        # to the nearer corner (np.sign(0) would put a "corner" on the midline)
    h = _sm(1.0 - r / 0.013)
    dx = np.sign(x) * wx * h
    dz = dz + cl * h
    k = _sm(1.0 - np.hypot(x, z - zm) / 0.022)
    dx = dx - x * nar * k
    dy = -pro * _sm(1.0 - np.hypot(x, z - zm) / 0.016)
    out = np.stack([dx, dy, dz], -1)
    box = (np.abs(x) < 0.05) & (np.abs(z - zm) < 0.035)
    out[~box] = 0.0
    return out


# name -> kwargs of lip_field (metres)
SHAPES = {
    "a": dict(up=0.0045, dn=0.0105, p=0.75, nar=0.28, cl=0.0004),
    "i": dict(up=0.0015, dn=0.0040, p=1.15, wx=0.0050, cl=0.0010),
    "u": dict(up=0.0025, dn=0.0040, p=0.6, nar=0.55, pro=0.0030),
    "e": dict(up=0.0025, dn=0.0065, p=0.9, wx=0.0020, cl=0.0004),
    "o": dict(up=0.0035, dn=0.0080, p=0.6, nar=0.36, pro=0.0020),
    "omega": dict(wave=0.0014),
    "mouth_smile": dict(cl=0.0028, wx=0.0008),
    "mouth_down": dict(cl=-0.0024),
    "grin": dict(up=0.0008, dn=0.0022, p=1.0, wx=0.0025, cl=0.0022),
}
