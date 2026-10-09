"""The hand of the procedural body: one closed surface (palm, four fingers, the thumb and the webs between them) drawn
as a coarse Catmull-Clark cage and subdivided twice (`mkmmd.model.subdiv`: the surface Blender's Subdivision modifier
makes), welded to the forearm at a seam `SEAM` metres before the wrist joint, plus a nail plate on every fingertip.

The hand has its own proportions, `DESIGN` (overridden key by key from `[body.hand]`): the finger joints the skeleton
takes are made from them (`joints`), so the spec's landmarks only place the wrist and the elbow (the forearm's
direction). Hand frame of the left hand (the right hand is its mirror): `a` along the hand (the forearm's direction
turned `bend` degrees towards the floor), `r` across it towards the little finger, `n` out of the palm. Design positions
are (along, across, out of the palm) from the wrist joint, in metres; the shape scales with `length`.

Every cage vertex has a TARGET, a point on the skin as designed; `fit` moves the cage until the subdivided surface
passes through all of them, so the numbers here set the skin itself, not a control polygon around it.

A hand drawn by someone else can stand in for the design: `[body.hand] mesh` names an asset that body_hand_mesh places
at the wrist instead (only `length`, `bend` and `nail` apply to it).

Cage layout (left hand, rings listed from the wrist outwards):
    W0       8 points  the seam: every fourth point of the forearm's last 32-point ring (two levels make it 32 again)
    W1       8 points  the wrist
    P1..P4  16 points  the palm: 8 on the back (d0 at the thumb side .. d7 at the little finger side), 8 on the
                       front (v7 .. v0); W1 -> P1 are pentagons (two palm edges per wrist edge); the side face
                       d0-v0 between P1 and P2 is where the thumb leaves the palm
    F0      16 points  the finger roots: finger f owns (d2f, d2f+1, v2f+1, v2f); a web quad joins neighbours
    fingers  4 points  per ring (back-thumb side, back-little side, front-little side, front-thumb side), eight rings
                       from the root to the tip, closed by a fan of four triangles on the fingertip
    thumb    4 points  per ring, the same way, from the palm's side face to its tip

UVs: the palm tile takes u round the 16-point rings (k / 16) and v along the hand; a finger's (or the thumb's) tile u
round its 4-point rings (k / 4) and v = s / length along its chain. `hand_shells` publishes, in the hand shell's
`info`: the frame, the chains, every vertex's share of each part (`member`), its coordinate along each chain
(`chain_s`), the along coordinate, and per tile the u of the back and of the front of the finger (for colour marks).
"""
import copy

import numpy as np

from ..subdiv import subdivide
from .body_geom import Path, Shell, pchip, smoothstep, surface_point, unit, vertex_normals

FINGERS = ("index", "middle", "ring", "little")
PARTS = ("palm",) + FINGERS + ("thumb",)
SEAM = -0.014                     # the forearm ends this far (m, along the hand) before the wrist joint
WRIST_RING = -0.003               # the cage's wrist ring (W1), along the hand from the wrist joint
LEVELS = 2                        # Catmull-Clark levels on the cage
CORNER = 0.73                     # a 4-point ring's targets sit at this share of the half width / depth: the
                                  # subdivided ring then reaches the full width and depth between them
REF_LENGTH = 0.146                # the hand length the palm's shape numbers below are drawn for (they scale with it)

# the default hand: small and soft, fingers that sit together and taper to rounded tips with a gentle resting curl;
# lengths in metres, angles in degrees
DESIGN = dict(
    length=0.140,                 # wrist joint -> middle fingertip, fingers straight
    palm=0.470,                   # share of `length` from the wrist joint to the middle finger's knuckle
    bend=6.0,                     # the hand turns down this much from the forearm's direction at rest
    mesh="",                      # a hand mesh asset (.npz, see body_hand_mesh) to use instead of this design
    knuckles=dict(                # finger root joint: along (from the middle knuckle), across, out of the palm
        index=(-0.0025, -0.0218, 0.0030), middle=(0.0, -0.0056, 0.0), ring=(-0.0025, 0.0104, 0.0015),
        little=(-0.0085, 0.0252, 0.0040)),
    fingers=dict(index=0.905, middle=1.0, ring=0.935, little=0.77),   # root joint -> tip, share of the middle finger
    width=dict(index=0.0140, middle=0.0144, ring=0.0136, little=0.0120, thumb=0.0145),   # at the finger's root
    taper=0.60,                   # fingertip width / root width
    depth=(0.76, 0.72),           # thickness / width at the root and at the tip: flat-ish fingers, not sausages
    palm_depth=0.0215,            # back to front through the middle of the palm
    split=(0.44, 0.29, 0.27),     # proximal, middle, distal share of a finger's length
    splay=dict(index=1.0, middle=0.0, ring=-1.0, little=-2.5),       # + towards the little finger: they lean together
    curl=dict(index=(6.0, 10.0, 7.0), middle=(7.0, 11.0, 7.0), ring=(8.0, 12.0, 8.0), little=(10.0, 14.0, 9.0)),
    thumb=dict(root=(0.019, -0.020, 0.007),        # its first joint (親指０) in the hand frame
               length=0.82,                         # root joint -> tip, share of the middle finger
               split=(0.40, 0.31, 0.29),
               yaw=(17.0, 5.0, -3.0),               # each segment turns this far from `a` away from the fingers
               pitch=(19.0, 0.0, -3.0)),            # and this far towards the palm side: it rests against the index
    nail=dict(start=0.40, end=0.91,                 # along the last phalanx (share of its length)
              width=0.64,                           # half width / the finger's half width
              lift=0.0003),                         # off the skin (m)
)


MESH_KEYS = ("mesh", "length", "bend", "nail")      # what a mesh hand takes from [body.hand]


def resolve(cfg=None):
    """DESIGN with the `[body.hand]` overrides (a table value updates the table key by key). Unknown keys are errors, and
    so are design keys next to a `mesh` (they shape only the designed hand)."""
    D = copy.deepcopy(DESIGN)
    for k, v in (cfg or {}).items():
        if k not in D:
            raise ValueError(f"[body.hand] has no key {k!r}; keys: {', '.join(DESIGN)}")
        if isinstance(D[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"[body.hand] {k} must be a table")
            bad = [x for x in v if x not in D[k]]
            if bad:
                raise ValueError(f"[body.hand.{k}] has no key {bad[0]!r}; keys: {', '.join(D[k])}")
            D[k].update(v)
        else:
            D[k] = v
    if D["mesh"]:
        extra = sorted(k for k in (cfg or {}) if k not in MESH_KEYS)
        if extra:
            raise ValueError(f"[body.hand] {', '.join(extra)}: a mesh hand takes only {', '.join(MESH_KEYS)}")
    return D


def frame(land, D):
    """(a, r, n) of the left hand: `a` the forearm's direction (elbow -> wrist) turned `bend` degrees towards the floor,
    `r` across the hand towards the little finger (backwards at rest), `n` out of the palm."""
    f = unit(np.asarray(land["wrist.L"], float) - np.asarray(land["elbow.L"], float))
    down = np.array([0.0, 0.0, -1.0])
    d = unit(down - (down @ f) * f)
    b = np.radians(float(D["bend"]))
    a = unit(np.cos(b) * f + np.sin(b) * d)
    back = np.array([0.0, 1.0, 0.0])
    r = unit(back - (back @ a) * a)
    n = -np.cross(a, r)
    return a, r, n


class _Hand:
    """The design in model space: frame, joints and chains (joint points root..tip) of the left hand."""

    def __init__(self, land, D):
        self.D = D
        self.sc = float(D["length"]) / REF_LENGTH
        self.W = np.asarray(land["wrist.L"], float)
        self.a, self.r, self.n = frame(land, D)
        self.L_mid = float(D["length"]) * (1.0 - float(D["palm"]))
        self.A_mid = float(D["length"]) * float(D["palm"])
        self.joints = {}
        self.chain = {}
        for f in FINGERS:
            dA, R, N = D["knuckles"][f]
            p = self.P(self.A_mid + dA, R, N)
            total = self.L_mid * float(D["fingers"][f])
            sp = np.radians(float(D["splay"][f]))
            base = np.cos(sp) * self.a + np.sin(sp) * self.r
            pts, cum = [p.copy()], 0.0
            for share, cu in zip(D["split"], D["curl"][f]):
                cum += np.radians(float(cu))
                p = p + (np.cos(cum) * base + np.sin(cum) * self.n) * total * float(share)
                pts.append(p.copy())
            self.chain[f] = np.array(pts)
            for i, nm in enumerate((f"{f}1", f"{f}2", f"{f}3", f"{f}_tip")):
                self.joints[nm + ".L"] = pts[i]
        T = D["thumb"]
        p = self.P(*T["root"])
        total = self.L_mid * float(T["length"])
        pts = [p.copy()]
        for share, yaw, pitch in zip(T["split"], T["yaw"], T["pitch"]):
            y, q = np.radians(float(yaw)), np.radians(float(pitch))
            d = np.cos(q) * (np.cos(y) * self.a - np.sin(y) * self.r) + np.sin(q) * self.n
            p = p + d * total * float(share)
            pts.append(p.copy())
        self.chain["thumb"] = np.array(pts)
        for i, nm in enumerate(("thumb0", "thumb1", "thumb2", "thumb_tip")):
            self.joints[nm + ".L"] = pts[i]

    def P(self, A, R, N):
        return self.W + float(A) * self.a + float(R) * self.r + float(N) * self.n

    def local(self, p):
        """(along, across, out of the palm) of model-space points p (..., 3)."""
        q = np.asarray(p, float) - self.W
        return np.stack([q @ self.a, q @ self.r, q @ self.n], -1)


def joints(land, D):
    """The finger joints of the left hand ({semantic name.L: (3,)}: thumb0..2, index1..3, ..., and the tips)."""
    return dict(_Hand(land, D).joints)


def seam_size(D):
    """Points of the hand's seam, which the forearm's last ring must have too: the cage's 8, doubled by every level."""
    return 8 * 2 ** LEVELS


def chain_coordinate(pts, P):
    """Arclength of the closest point of the joint polyline `pts` (root..tip) to each point of P (n, 3); before the root
    it runs on backwards along the first segment (negative), beyond the tip forwards."""
    seg = np.diff(pts, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    d = seg / ln[:, None]
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    best = np.full(len(P), np.inf)
    out = np.zeros(len(P))
    last = len(ln) - 1
    for j in range(len(ln)):
        t = (P - pts[j]) @ d[j]
        t = np.clip(t, -np.inf if j == 0 else 0.0, np.inf if j == last else ln[j])
        dist = np.linalg.norm(P - (pts[j] + t[:, None] * d[j]), axis=1)
        better = dist < best
        best = np.where(better, dist, best)
        out = np.where(better, cum[j] + t, out)
    return out


# ---------------------------------------------------------------- the cage
def _across_out(t, r, n):
    """Unit directions across the hand (towards the little finger, `r`) and out of the palm (`n`), perpendicular to the
    tangent t."""
    eu = unit(r - (t @ r) * t)
    ev = unit(n - (t @ n) * t - (n @ eu) * eu)
    return eu, ev


def _ring4(c, eu, ev, half_w, back, front):
    """The 4 targets of a finger ring: back-thumb side, back-little side, front-little side, front-thumb side."""
    x, yb, yf = CORNER * half_w, CORNER * back, CORNER * front
    return np.array([c - x * eu - yb * ev, c + x * eu - yb * ev, c + x * eu + yf * ev, c - x * eu + yf * ev])


def _rounded(x, p):
    """Height share of a superellipse |x|^p + |y|^p = 1 at x (the palm's back or front at a column)."""
    return np.clip(1.0 - np.abs(np.clip(x, -0.999, 0.999)) ** p, 0.0, 1.0) ** (1.0 / p)


class Cage:
    """Cage vertices (targets), faces, the part of every vertex and face, and per-part (u, v) of the vertices."""

    def __init__(self):
        self.T = []                   # targets
        self.part = []                # part index per vertex
        self.uv = []                  # {part name: (u, v)} per vertex
        self.faces = []
        self.fpart = []               # part index per face
        self.rings = {}               # name -> vertex indices
        self.meta = {}                # part -> u of its back / front, chain length, path (fingers and thumb)

    def add(self, name, pts, part, uvs):
        i0 = len(self.T)
        for p, uv in zip(pts, uvs):
            self.T.append(np.asarray(p, float))
            self.part.append(PARTS.index(part))
            self.uv.append(dict(uv))
        idx = np.arange(i0, i0 + len(pts))
        self.rings[name] = idx
        return idx

    def face(self, vs, part):
        self.faces.append([int(v) for v in vs])
        self.fpart.append(PARTS.index(part))

    def band(self, a, b, part):
        """Quads between two rings of equal size listed in the same direction (a proximal)."""
        m = len(a)
        for k in range(m):
            self.face([a[k], a[(k + 1) % m], b[(k + 1) % m], b[k]], part)

    def corner_uv(self):
        """Per-corner UVs (tile-local, 0..1) in face order: each face takes its part's (u, v) of every corner; a face
        across the u seam unwraps it (u 0 -> 1)."""
        out = []
        for f, p in zip(self.faces, self.fpart):
            uv = np.array([self.uv[v][PARTS[p]] for v in f], float)
            if uv[:, 0].max() - uv[:, 0].min() > 0.5:
                uv[uv[:, 0] < 0.25, 0] += 1.0
            out.append(uv)
        return np.concatenate(out, 0)


def _palm_rows(h, D, wrist):
    """Palm sections P1..P4: (along per column (8,), across per back column (8,), across per front column (8,), radial
    edge, ulnar edge, back depth, front depth, squareness). `wrist` = (radial edge, ulnar edge) of the wrist at W1. P4 is
    the knuckle line: its columns sit at the finger edges and follow the knuckles' arch. Near the wrist the front's
    second column (v1) stands further in, so the thumb's socket (v1-v0 between P1 and P2) is wide."""
    sc = h.sc
    kn = D["knuckles"]
    c = {f: float(kn[f][1]) for f in FINGERS}
    w = {f: float(D["width"][f]) for f in FINGERS}
    rad4 = c["index"] - 0.5 * w["index"] - 0.0008 * sc
    uln4 = c["little"] + 0.5 * w["little"] + 0.0010 * sc
    cols4 = np.array([rad4, c["index"] + 0.40 * w["index"], c["middle"] - 0.40 * w["middle"],
                      c["middle"] + 0.40 * w["middle"], c["ring"] - 0.40 * w["ring"], c["ring"] + 0.40 * w["ring"],
                      c["little"] - 0.40 * w["little"], uln4])
    colA4 = np.array([h.A_mid + float(kn[f][0]) - 0.003 * sc for f in FINGERS for _ in (0, 1)])
    depth = float(D["palm_depth"])
    rows = []
    A1 = 0.010 * sc
    for i, (A, thick) in enumerate(((A1, 1.17), (0.024 * sc, 1.07), (0.040 * sc, 1.0))):
        rad = PALM_FLARE[0][i] * rad4 + (1 - PALM_FLARE[0][i]) * wrist[0]
        uln = PALM_FLARE[1][i] * uln4 + (1 - PALM_FLARE[1][i]) * wrist[1]
        t = smoothstep((A - A1) / (colA4.min() - A1))
        even = np.linspace(rad, uln, 8)
        even[0], even[-1] = rad + 0.035 * (uln - rad), uln - 0.035 * (uln - rad)
        cols = (1 - t) * even + t * cols4
        cols[-1] = even[-1]                            # the little finger side's edge follows PALM_FLARE alone
        front = cols.copy()
        if i in (SOCKET_RING, SOCKET_RING + 1):        # the rings round the thumb's socket
            front[1] += SOCKET_WIDEN * (uln - rad)
        rows.append((np.full(8, A), cols, front, rad, uln, 0.49 * depth * thick, 0.51 * depth * thick,
                     PALM_ROUND[0] + (PALM_ROUND[1] - PALM_ROUND[0]) * t))
    rows.append((colA4, cols4, cols4.copy(), rad4, uln4, 0.48 * depth, 0.50 * depth, PALM_ROUND[1]))
    return rows


# how far the palm has widened from the wrist to the knuckle line at P1, P2, P3 (a share of the way from the wrist's
# edge to the knuckle line's): on the thumb side (the ball of the thumb) early, on the little finger side gradually, so
# that edge runs in a slender, nearly straight line from the wrist and swells only at the little finger's knuckle
PALM_FLARE = ((0.81, 0.925, 0.99), (0.36, 0.78, 1.06))
# soft bumps on the palm (along, across, radius, height; metres at REF_LENGTH, + out of the skin): the ball of the thumb,
# the heel on the little finger side, the hollow of the palm, the web between the thumb and the index
PALM_BUMPS = ((0.027, -0.021, 0.016, 0.0042), (0.027, 0.023, 0.014, 0.0022), (0.044, 0.002, 0.012, -0.0015),
              (0.052, -0.024, 0.014, 0.0045))
PALM_ROUND = (2.2, 2.4)           # superellipse exponent of the palm's sections at the wrist end and at the knuckles
                                  # (2 = an ellipse: a domed back of the hand, soft edges)
SOCKET, SOCKET_RING = 14, 1      # the thumb leaves the palm through the face of columns v1 .. v0 between P2 and P3:
                                  # its metacarpal stays in the palm's flesh, the first web starts at its knuckle
SOCKET_WIDEN = 0.12               # v1 moves this share of the palm's width inwards on the rings round the socket
WEB_BACK, WEB_FRONT = 0.34, 0.50  # a finger's root ring (the webs) along its first phalanx: back side, palm side
ROOT_FULL = (1.10, 1.04)          # the root ring and the next one are drawn this much wider: the webs pull them in


def _angle(arm, p):
    """Angle of points p (k, 3) round the forearm's axis at its seam, in the tube's convention (0 = the ring's first
    point, growing with the ring index)."""
    info = arm.info
    s = info["s_seam"]
    c = info["path"].point(np.array([s]))[0]
    t = unit(info["path"].tangent(np.array([s]))[0])
    ref = np.asarray(info["ref"], float)
    ef = unit(ref - (t @ ref) * t)
    eb = np.cross(t, ef)
    q = np.asarray(p, float) - c
    return np.mod(np.arctan2(q @ eb, q @ ef), 2 * np.pi)


def _seam_points(arm, P1):
    """The cage's 8 seam targets (every fourth point of the arm's last ring) and its 8 wrist targets: the seam points
    follow the turning sense of the palm rings and start where the pairing with P1's even points twists least; the
    wrist ring sits halfway (in angle) between the seam and P1."""
    info = arm.info
    M = info["M"]
    ring = arm.rings[-1]
    want = _angle(arm, P1[0::2])                       # the angles the 8 seam points should pair with
    sense = np.sign(np.sin(np.diff(np.unwrap(want))).sum())
    best = None
    for c in range(4):
        for m0 in range(8):
            k = np.array([(c + 4 * (m0 + int(sense) * j)) % M for j in range(8)])
            th = 2 * np.pi * k / M
            err = np.abs(np.angle(np.exp(1j * (th - want)))).sum()
            if best is None or err < best[0]:
                best = (err, k, th)
    _, k, th = best
    half = th + 0.5 * np.angle(np.exp(1j * (want - th)))
    s1 = info["s_w"] + WRIST_RING
    w1, _ = surface_point(info["path"], np.full(8, s1), info["tab"], info["ref"], half)
    return ring[k], w1


def cage(land, D, arm):
    """The cage of the left hand on the forearm `arm` (body_mesh.arm_shell). Returns (Cage, _Hand)."""
    h = _Hand(land, D)
    sc = h.sc
    C = Cage()
    info = arm.info
    th = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    wr, _ = surface_point(info["path"], np.full(64, info["s_w"] + WRIST_RING), info["tab"], info["ref"], th)
    wl = h.local(wr)
    rows = _palm_rows(h, D, (float(wl[:, 1].min()), float(wl[:, 1].max())))
    A_seam = float(h.local(arm.rings[-1])[:, 0].mean())
    A_end = max(float(h.local(h.chain[f][0])[0]) for f in FINGERS) + 0.012 * sc
    v_of = lambda A: float(np.clip((A - A_seam) / (A_end - A_seam), 0.0, 1.0))
    # ---- palm rings P1..P4 first (the seam is paired with P1): 16 targets each, d0..d7 over the back, v7..v0 over
    # the front
    Pt = []
    for i, (A, cols, front_cols, rad, uln, back, front, p) in enumerate(rows):
        Rc, Wh = 0.5 * (rad + uln), 0.5 * (uln - rad)
        nb = -back * _rounded((cols - Rc) / Wh, p)
        nf = front * _rounded((front_cols - Rc) / Wh, p)
        if i == 3:                                     # soft pads under the knuckles on the palm side
            nf = nf + 0.0006 * sc
        for bA, bR, rb, hgt in PALM_BUMPS:
            g = (((A - bA * sc) / (rb * sc)) ** 2 + ((front_cols - bR * sc) / (rb * sc)) ** 2)
            nf = nf + hgt * sc * np.exp(-g)
        pts = [h.P(A[k], cols[k], nb[k]) for k in range(8)] + [h.P(A[k], front_cols[k], nf[k]) for k in range(8)][::-1]
        Pt.append((np.array(pts), list(A) + list(A[::-1])))
    w0, w1 = _seam_points(arm, Pt[0][0])
    W0 = C.add("W0", w0, "palm", [{"palm": (j / 8, v_of(float(h.local(w0[j])[0])))} for j in range(8)])
    W1 = C.add("W1", w1, "palm", [{"palm": (j / 8, v_of(float(h.local(w1[j])[0])))} for j in range(8)])
    C.band(W0, W1, "palm")
    P = [C.add(f"P{i + 1}", pts, "palm", [{"palm": (k / 16, v_of(Av[k]))} for k in range(16)])
         for i, (pts, Av) in enumerate(Pt)]
    for j in range(8):                                 # W1 -> P1: two palm edges per wrist edge
        C.face([W1[j], W1[(j + 1) % 8], P[0][(2 * j + 2) % 16], P[0][2 * j + 1], P[0][2 * j]], "palm")
    for i in range(3):                                 # P1 -> P2 -> P3 -> P4, the thumb's socket left open
        for k in range(16):
            if (i, k) != (SOCKET_RING, SOCKET):
                C.face([P[i][k], P[i][(k + 1) % 16], P[i + 1][(k + 1) % 16], P[i + 1][k]], "palm")
    s0, s1 = P[SOCKET_RING], P[SOCKET_RING + 1]
    socket = [s0[SOCKET], s0[(SOCKET + 1) % 16], s1[(SOCKET + 1) % 16], s1[SOCKET]]
    F0 = np.zeros(16, int)
    for fi, f in enumerate(FINGERS):
        C.meta[f] = _finger(C, h, D, f, fi, F0, v_of)
    C.band(P[3], F0, "palm")
    for fi in range(3):                                # webs between neighbouring fingers
        C.face([F0[2 * fi + 1], F0[2 * fi + 2], F0[13 - 2 * fi], F0[14 - 2 * fi]], "palm")
    C.meta["thumb"] = _thumb(C, h, D, socket)
    return C, h


def _width(D, f, q):
    w0 = float(D["width"][f])
    return w0 * pchip([0.0, 0.18, 0.44, 0.73, 1.0], [1.0, 0.985, 0.90, 0.82, float(D["taper"])])(np.clip(q, 0, 1))


def _depth(D, q):
    d0, d1 = D["depth"]
    return float(d0) + (float(d1) - float(d0)) * float(np.clip(q, 0.0, 1.0))


def _finger(C, h, D, f, fi, F0, v_of):
    """Rings of finger f (the root ring is the palm's F0 slots 2f, 2f+1, 14-2f, 15-2f) and its tip fan. Returns the
    finger's u of its back and front (tile-local) and its chain length."""
    pts = h.chain[f]
    L1, L2, L3 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(L1 + L2 + L3)
    path = Path(pts, blend=0.004)
    s_d, s_v = WEB_BACK * L1, WEB_FRONT * L1          # the root ring (the webs): its back points, its front points
    stations = [0.70 * L1, L1 - 0.0028 * h.sc, L1 + 0.0028 * h.sc, L1 + 0.5 * L2, L1 + L2 - 0.0024 * h.sc,
                L1 + L2 + 0.0024 * h.sc, L1 + L2 + 0.45 * L3, L - 0.20 * L3]

    def corners(s, which, full=1.0):
        q = s / L
        c = path.point(np.array([s]))[0]
        eu, ev = _across_out(path.tangent(np.array([s]))[0], h.r, h.n)
        w = _width(D, f, q) * full
        th = w * _depth(D, q)
        return _ring4(c, eu, ev, 0.5 * w, 0.46 * th, 0.54 * th)[list(which)]

    r0 = np.concatenate([corners(s_d, (0, 1), ROOT_FULL[0]), corners(s_v, (2, 3), ROOT_FULL[0])])
    slots = (2 * fi, 2 * fi + 1, 14 - 2 * fi, 15 - 2 * fi)
    sv = (s_d, s_d, s_v, s_v)
    root = C.add(f"{f}0", r0, f, [{f: (k / 4, sv[k] / L), "palm": (slots[k] / 16, v_of(float(h.local(r0[k])[0])))}
                                  for k in range(4)])
    F0[list(slots)] = root
    prev = root
    for i, s in enumerate(stations):
        full = ROOT_FULL[1] if i == 0 else 1.0
        cur = C.add(f"{f}{i + 1}", corners(s, range(4), full), f, [{f: (k / 4, s / L)} for k in range(4)])
        C.band(prev, cur, f)
        prev = cur
    w = _width(D, f, 1.0)
    eu, ev = _across_out(path.tangent(np.array([L]))[0], h.r, h.n)
    tip = pts[-1] - 0.10 * w * _depth(D, 1.0) * ev    # the fingertip, a little towards the nail
    ti = C.add(f"{f}_tip", [tip], f, [{f: (0.5, 1.0)}])[0]
    for k in range(4):
        C.face([prev[k], prev[(k + 1) % 4], ti], f)
    return dict(back_u=0.125, front_u=0.625, length=L, path=path)


def _thumb(C, h, D, socket):
    """Rings of the thumb from the palm's side face (`socket`, 4 cage vertices in the face's order) and its tip fan."""
    pts = h.chain["thumb"]
    L1, L2, L3 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(L1 + L2 + L3)
    path = Path(pts, blend=0.005)
    w0 = float(D["width"]["thumb"])
    sc = h.sc
    # (s, width, thickness): the root at its knuckle (the metacarpal stays in the palm's flesh), then the phalanges
    table = [(L1 - 0.003 * sc, 1.12 * w0, 0.90 * w0),
             (L1 + 0.003 * sc, 1.02 * w0, 0.78 * w0), (L1 + 0.5 * L2, 0.98 * w0, 0.74 * w0),
             (L1 + L2 - 0.0025 * sc, 0.92 * w0, 0.68 * w0), (L1 + L2 + 0.0025 * sc, 0.90 * w0, 0.65 * w0),
             (L1 + L2 + 0.45 * L3, 0.86 * w0, 0.62 * w0), (L - 0.20 * L3, 0.66 * w0, 0.48 * w0)]
    pad = unit(h.r + h.n)                              # the thumb's pad faces the palm and the fingers

    def ring(s, w, th):
        c = path.point(np.array([s]))[0]
        t = unit(path.tangent(np.array([s]))[0])
        ev = unit(pad - (pad @ t) * t)
        eu = np.cross(ev, t)
        return _ring4(c, eu, ev, 0.5 * w, 0.46 * th, 0.54 * th)

    rings = [ring(*row) for row in table]
    # turn the thumb's rings the way the socket turns about the thumb's direction, and pair each socket point with
    # the nearest ring point
    T0 = np.array([C.T[i] for i in socket])
    d = unit(pts[1] - pts[0])
    sense = lambda R: np.sign(np.cross(R[1] - R[0], R[2] - R[1]) @ d)
    order = np.arange(4)
    if sense(rings[0]) != sense(T0):
        order = order[::-1]
    shift = min(range(4), key=lambda k: np.linalg.norm(T0 - rings[0][np.roll(order, -k)], axis=1).sum())
    order = np.roll(order, -shift)
    for i, v in enumerate(socket):
        C.uv[v]["thumb"] = (i / 4, 0.0)
    prev = list(socket)
    for i, (row, R) in enumerate(zip(table, rings)):
        s = row[0]
        cur = C.add(f"thumb{i + 1}", R[order], "thumb", [{"thumb": (k / 4, s / L)} for k in range(4)])
        C.band(prev, cur, "thumb")
        prev = cur
    t = unit(path.tangent(np.array([L]))[0])
    ev = unit(pad - (pad @ t) * t)
    ti = C.add("thumb_tip", [pts[-1] - 0.06 * w0 * ev], "thumb", [{"thumb": (0.5, 1.0)}])[0]
    for k in range(4):
        C.face([prev[k], prev[(k + 1) % 4], ti], "thumb")
    pos = {int(o): i for i, o in enumerate(order)}     # ring slot of each corner (0 back-thumb .. 3 front-thumb)
    u_mid = lambda a, b: (((pos[a] + pos[b]) / 2.0 if abs(pos[a] - pos[b]) == 1 else 3.5) / 4.0) % 1.0
    return dict(back_u=u_mid(0, 1), front_u=u_mid(2, 3), length=L, path=path, pad=pad)


# ---------------------------------------------------------------- subdivision, welding
def orient(C):
    """Flip every face if the cage is inside out (the back of the palm must face away from the palm)."""
    T = np.array(C.T)
    i = int(C.rings["P3"][3])                          # a point on the back of the palm
    centre = T[C.rings["P3"]].mean(0)
    f = next(f for f in C.faces if i in f)
    k = f.index(i)
    nrm = np.cross(T[f[(k + 1) % len(f)]] - T[i], T[f[k - 1]] - T[i])
    if nrm @ (T[i] - centre) < 0:
        C.faces = [g[::-1] for g in C.faces]


def fit(C, iters=10):
    """Subdivide the cage LEVELS times, the cage solved so that the surface passes through every target: returns
    (Subdivided with the fitted positions, cage positions)."""
    T = np.array(C.T)
    sd = subdivide(T, C.faces, LEVELS, uv=C.corner_uv())
    S0 = sd.S[:len(T)]
    X = T.copy()
    for _ in range(iters):
        X = X + (T - S0 @ X)
    sd.verts = np.ascontiguousarray(sd.S @ X)
    return sd, X


def _boundary_loop(faces, n):
    """Vertices of the open boundary (edges used by one face) of a quad mesh."""
    from collections import Counter
    cnt = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            cnt[(min(a, b), max(a, b))] += 1
    return np.unique([v for e, c in cnt.items() if c == 1 for v in e])


def _raycast(o, d, V, tris):
    """First hits of rays o + t d (k, 3) on triangles V[tris] (m, 3): (t (k,), triangle (k,), barycentric (k, 3));
    t = inf where a ray misses."""
    A, B, Cc = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    e1, e2 = B - A, Cc - A
    p = np.cross(d[:, None, :], e2[None])
    det = np.einsum("kmj,mj->km", p, e1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o[:, None, :] - A[None]
    u = np.einsum("kmj,kmj->km", s, p) * inv
    q = np.cross(s, e1[None])
    v = np.einsum("kj,kmj->km", d, q) * inv
    t = np.einsum("mj,kmj->km", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    t = np.where(hit, t, np.inf)
    j = np.argmin(t, axis=1)
    tt = t[np.arange(len(o)), j]
    uu, vv = u[np.arange(len(o)), j], v[np.arange(len(o)), j]
    return tt, j, np.stack([1 - uu - vv, uu, vv], 1)


NAIL_ROWS = np.array([0.0, 0.04, 0.10, 0.20, 0.35, 0.55, 0.72, 0.85, 0.94, 1.0])
NAIL_SHAPE = np.array([0.50, 0.74, 0.88, 0.96, 1.0, 1.0, 0.95, 0.84, 0.64, 0.34])     # rounded cuticle, almond tip


def nail_plate(part, path, s3, L3, axes, width, V, N, tri, nd, behind):
    """A nail plate on the back of a fingertip: rows from the cuticle to the free edge along the last phalanx (arclength
    s3 .. s3 + L3 of `path`, from nd `start` to nd `end` of it), each a row of points cast onto the skin triangles `tri`
    from the back of the finger and lifted off it (nd `lift`); an almond outline (rounded cuticle, a softly pointed free
    edge). axes(tangent) -> (across the finger, towards its pad); width(s) is the finger's width at s; the rays start
    `behind` metres off the chain. Returns (verts, faces), the faces turned away from the finger."""
    cols = np.linspace(-1.0, 1.0, 7)
    rows = []
    for t, g in zip(NAIL_ROWS, NAIL_SHAPE):
        s = s3 + L3 * (float(nd["start"]) + (float(nd["end"]) - float(nd["start"])) * t)
        c = path.point(np.array([s]))[0]
        eu, ev = axes(unit(path.tangent(np.array([s]))[0]))
        half = 0.5 * width(s) * float(nd["width"]) * g
        o = c[None] + cols[:, None] * half * eu[None] - behind * ev[None]
        tt, j, b = _raycast(o, np.repeat(ev[None], len(o), 0), V, tri)
        if not np.isfinite(tt).all():
            raise ValueError(f"the {part} nail does not land on the fingertip")
        hit = o + tt[:, None] * ev[None]
        nrm = unit((N[tri[j]] * b[:, :, None]).sum(1))
        rows.append(hit + nrm * float(nd["lift"]))
    P = np.array(rows).reshape(-1, 3)
    nu = len(cols)
    faces = [[r * nu + c, r * nu + c + 1, (r + 1) * nu + c + 1, (r + 1) * nu + c]
             for r in range(len(NAIL_ROWS) - 1) for c in range(nu - 1)]
    f0 = faces[len(faces) // 2]
    nrm = np.cross(P[f0[1]] - P[f0[0]], P[f0[3]] - P[f0[0]])
    if nrm @ -axes(unit(path.tangent(np.array([s3 + 0.5 * L3]))[0]))[1] < 0:
        faces = [f[::-1] for f in faces]
    return P, faces


def _nail(h, D, part, meta, V, N, F, fpart):
    """The design's nail plate on `part`: nail_plate on that part's skin near its tip, the design's finger widths."""
    pts = h.chain[part]
    L3 = float(np.linalg.norm(pts[-1] - pts[-2]))
    tri = []
    for f, p in zip(F, fpart):
        if p == PARTS.index(part):
            tri += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
    tri = np.array(tri)
    tri = tri[np.linalg.norm(V[tri].mean(1) - pts[-1], axis=1) < 1.3 * L3]
    if part == "thumb":
        pad = meta["pad"]

        def axes(tg):
            ev = unit(pad - (pad @ tg) * tg)
            return np.cross(ev, tg), ev

        def width(s):
            return float(D["width"]["thumb"]) * 0.92
    else:
        def axes(tg):
            return _across_out(tg, h.r, h.n)

        def width(s):
            return _width(D, part, s / meta["length"])
    return nail_plate(part, meta["path"], meta["length"] - L3, L3, axes, width, V, N, tri, D["nail"], 0.03 * h.sc)


def hand_shells(shape, arm, atlas, nails=True):
    """(hand_L Shell, [nail_<part>_L Shells]) for the left hand on the forearm `arm`: the hand's seam vertices are
    welded to the arm's last ring (the ring moves onto the subdivided seam, sub-millimetre)."""
    D = shape.hand
    C, h = cage(shape.land, D, arm)
    orient(C)
    sd, _ = fit(C)
    V, F = sd.verts, sd.faces
    n = len(V)
    # weld the seam: each boundary vertex onto the arm ring point it lies on
    bnd = _boundary_loop(F, n)
    ring_idx = arm.ring_index[-1]
    M = len(ring_idx)
    if len(bnd) != M:
        raise ValueError(f"the hand's seam has {len(bnd)} vertices, the arm's last ring {M}")
    ring = arm.verts[ring_idx]
    pair = np.argmin(np.linalg.norm(V[bnd][:, None] - ring[None], axis=2), axis=1)
    if len(set(pair.tolist())) != M:
        raise ValueError("the hand's seam does not pair one to one with the arm's last ring")
    gap = float(np.linalg.norm(V[bnd] - ring[pair], axis=1).max())
    if gap > 0.0015:
        raise ValueError(f"the hand's seam is {gap * 1000:.2f} mm off the arm's last ring")
    arm.verts[ring_idx[pair]] = V[bnd]
    arm.rings = arm.verts[arm.ring_index]
    # per-vertex data: share of each part, coordinate along each chain, UVs in the atlas
    onehot = np.zeros((len(C.T), len(PARTS)))
    onehot[np.arange(len(C.T)), C.part] = 1.0
    member = np.asarray(sd.L @ onehot)
    chain_s = np.stack([chain_coordinate(h.chain[p], V) for p in PARTS[1:]], 1)
    fpart = np.asarray(C.fpart)[sd.face_parent]
    uv = sd.uv.reshape(-1, 4, 2)
    uvs = []
    for q, p in zip(uv, fpart):
        u0, v0, u1, v1 = atlas[PARTS[p]]
        uvs.append(np.stack([u0 + (u1 - u0) * np.clip(q[:, 0], 0, 1), v0 + (v1 - v0) * np.clip(q[:, 1], 0, 1)], 1))
    sh = Shell("hand_L")
    sh.verts = V
    sh.faces = [list(f) for f in F]
    sh.uv = uvs
    sh.s = h.local(V)[:, 0]
    sh.theta = np.full(n, np.nan)
    sh.ring = np.full(n, -1)
    sh.face_mat = [0] * len(F)
    sh.ext = {int(b): ("arm_L", int(ring_idx[p])) for b, p in zip(bnd, pair)}
    lengths = {p: np.linalg.norm(np.diff(h.chain[p], axis=0), axis=1) for p in PARTS[1:]}
    sh.info = dict(kind="hand", wrist=h.W, frame=(h.a, h.r, h.n), chains=h.chain, lengths=lengths, member=member,
                   chain_s=chain_s, along=sh.s.copy(), fpart=fpart, a_mid=h.A_mid, la=arm.info["la"], lf=arm.info["lf"],
                   tiles={p: dict(back_u=C.meta[p]["back_u"], front_u=C.meta[p]["front_u"], length=C.meta[p]["length"])
                          for p in PARTS[1:]},
                   palm_u={f: ((2 * i + 0.5) / 16, (14.5 - 2 * i) / 16) for i, f in enumerate(FINGERS)},
                   palm_v=dict(seam=float(h.local(arm.rings[-1])[:, 0].mean()),
                               end=max(float(h.local(h.chain[f][0])[0]) for f in FINGERS) + 0.012 * h.sc))
    out = []
    if nails:
        N = vertex_normals(V, F)
        for p in PARTS[1:]:
            P, faces = _nail(h, D, p, C.meta[p], V, N, F, fpart)
            ns = Shell(f"nail_{p}_L")
            ns.verts = P
            ns.faces = faces
            ns.uv = [np.full((4, 2), 0.5) for _ in faces]
            ns.s = np.zeros(len(P))
            ns.theta = np.full(len(P), np.nan)
            ns.ring = np.full(len(P), -1)
            ns.face_mat = [1] * len(faces)
            ns.info = dict(kind="nail", part=p, hand="hand_L")
            out.append(ns)
    return sh, out
