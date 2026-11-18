"""Glasses (part `glasses`, numpy only): small round wire glasses worn on the nose, rigid on the head bone.

Two round lenses `size` across outside their rims, each rim a ring of round wire `wire` thick; between them a short
bridge arched over the nose; from each rim's outer side a thin temple of the same wire running back
`[glasses.temples] length` along the side of the head (into the side hair, which hides its end).

Where they sit (`[glasses.place]`): `height` puts the rims' tops on the eye opening, as a share of its height (0 its
lowest point, the lower lash line: she looks over them; 1 its highest; above 1, over the upper lashes, the whole eye
inside the lens); `x` sets the lenses' centres that far either side of her middle (0: on the pupils); `tilt` leans the
lenses' tops forward (degrees about her left-right axis, as worn glasses lean) and `wrap` turns each lens toward her
side; `gap` is the least distance from any of it to the head's meshes (skin, lashes, brows, the eyes' and mouth's
pieces) at rest, in every vertex morph of the face and in every eye morph with every brow morph: the glasses are moved
forward until it holds and the bridge (`[glasses.bridge]`: its ends `at` a share of the rim's radius above the lenses'
centres, rising `arch` in its middle) bows forward over the nose just as far as it needs. Each temple
(`[glasses.temples]`: from `at` above the centres, a hinge `end` out of the rim) runs `length` along the head toward
its back, rising `rise` degrees: straight back where it can, along the head where the head is in the way (level, so it
never climbs over the brow), pushed out level wherever it would come nearer than `gap` to the skin or to any other
piece of the head in any of those states (the lashes' and the brows' outer ends), and ending early where the ear
begins; so it follows the head into the side hair instead of splaying away from it. The quick fit measures to the
head's vertices; the triangles are then measured exactly in every state from every point of the wire, and what they
find closer is added to the gap of the lenses and the bridge, or of the temples, until it holds.

The lenses are clear glass, `[colors.glasses] lens` at `alpha` (0: nothing drawn over the eyes, which stay as vivid
as without them; a renderer that hashes transparency draws a faint tint as grain), with one crisp glint each
(`[glasses.glint]`: `at` [degrees round the lens from its outer side toward its top, share of its radius], a band
`width` metres wide and `length` metres long with round ends, at its `alpha`, `[colors.glasses] glint`), at the upper
outer edge of both. The wire is `[colors.glasses] frame`, lit `frame_light` along its top; its outline a darker shade of
it. Materials 眼鏡 (wire) and 眼鏡レンズ (lenses: alpha-blended, after the wire); their flags cast no shadow (the
wire's would lie as a line on her cheeks; mkr casts every material's shadow, so in its pictures only a face skin that
takes none loses it) and the lenses take none. `info` holds the measures: the lenses' size and centres, the wire, the
rims' tops on the eye opening, the smallest gap to the head and where (piece and morphs) and each piece's, the bridge
over the nose, the temples' length along the head, how far behind and outside their hinges they end, their ends, the
glint; landmarks `glasses.L`, `glasses.R` (the lenses' centres) and `glasses_bridge`."""
import colorsys

import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Material, Part
from ..skin import closest_on_mesh
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_wear as WR

DEFAULTS = {"size": 0.040, "wire": 0.0014, "alpha": 0.12,
            "place": {"height": 0.0, "x": 0.0, "tilt": 8.0, "wrap": 6.0, "gap": 0.002},
            "bridge": {"at": 0.0, "arch": 0.003},
            "temples": {"length": 0.050, "at": 0.7, "end": 0.002, "rise": 5.0},
            "glint": {"at": [45.0, 0.84], "length": 0.008, "width": 0.0015, "alpha": 1.0},
            "bone": "頭", "segments": 72, "sides": 8}
COLORS = {"frame": "#3b2a2a", "frame_light": "#7a5f58", "lens": "#f4fbff", "glint": "#ffffff"}
MATS = ("眼鏡", "眼鏡レンズ")
WIRE, LENS = 0, 1
BOX = 0.03                      # metres round the glasses inside which the head's vertices and triangles are looked at
INF = 1e9


# ---------------------------------------------------------------- the head the glasses keep off
class Face:
    """The head part's meshes in their rest shape and in their vertex morphs. `states` lists the morph sets looked at
    (none, each morph alone, each eye morph with each brow morph); `points(state)` the vertices in one of them; `mesh`
    the name of the mesh each vertex belongs to."""

    def __init__(self, head):
        V, T, names, offs, off = [], [], [], {}, 0
        for m in head.meshes:
            v = np.asarray(m.verts, float)
            if not len(v) or not len(m.faces):
                continue
            V.append(v)
            T.append(G.tri_arrays(list(m.faces))[0] + off)
            names += [m.name] * len(v)
            for k, d in m.morphs.items():
                offs.setdefault(k, []).append((off, np.asarray(d, float)))
            off += len(v)
        self.V, self.T, self.mesh = np.vstack(V), np.vstack(T), np.array(names)
        self.off = {}
        for k, pieces in offs.items():
            D = np.zeros_like(self.V)
            for o, d in pieces:
                D[o:o + len(d)] = d
            if np.abs(D).max() > 0.0:
                self.off[k] = D
        panel = {mo.name: mo.panel for mo in head.morphs}
        eye = [k for k in sorted(self.off) if panel.get(k) == "eye"]
        brow = [k for k in sorted(self.off) if panel.get(k) in ("brow", "eyebrow")]
        self.states = [()] + [(k,) for k in sorted(self.off)] + [(e, b) for e in eye for b in brow]

    def points(self, state):
        P = self.V.copy()
        for k in state:
            P += self.off[k]
        return P

    def cloud(self, lo, hi, tol=5e-5, skip=()):
        """Every vertex inside the box [lo, hi] in every state, stacked, the ones within `tol` of another merged (what
        the quick fit measures to); the vertices of the meshes named in `skip` left out."""
        keep = ~np.isin(self.mesh, list(skip))
        out = []
        for s in self.states:
            P = self.points(s)[keep]
            out.append(P[((P >= lo) & (P <= hi)).all(1)])
        P = np.vstack(out)
        return np.unique(np.round(P / tol), axis=0) * tol if len(P) else P


# ---------------------------------------------------------------- distances to the glasses' pieces
def disc_distance(P, c, n, R):
    """Distance (len(P),) of points to the flat disc of radius R at c with unit normal n."""
    q = P - c
    h = q @ n
    rho = np.linalg.norm(q - h[:, None] * n, axis=1)
    return np.hypot(np.maximum(rho - R, 0.0), h)


def ring_distance(P, c, n, R, r):
    """Distance of points to the surface of a ring of round wire: centreline a circle of radius R at c about the
    unit normal n, the wire r thick in radius."""
    q = P - c
    h = q @ n
    rho = np.linalg.norm(q - h[:, None] * n, axis=1)
    return np.hypot(rho - R, h) - r


def path_distance(P, path, r, chunk=2048):
    """Distance of points to the surface of a tube of radius r round the polyline `path` (k, 3)."""
    if not len(P):
        return np.full(0, INF)
    A, B = path[:-1], path[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        t = np.clip(np.einsum("pkj,kj->pk", Q[:, None, :] - A[None], AB) / L2[None], 0.0, 1.0)
        near = A[None] + t[..., None] * AB[None]
        out[i:i + chunk] = np.linalg.norm(Q[:, None, :] - near, axis=2).min(1)
    return out - r


def least(d):
    return float(d.min()) if len(d) else INF


# ---------------------------------------------------------------- the frame
def lens_frame(side, tilt, wrap):
    """(normal out of the face, up, toward her left) of the lens on `side` (+1 her left, -1 her right): the top
    leans `tilt` degrees forward, the lens turns `wrap` degrees toward her side."""
    t, w = np.radians(tilt), np.radians(wrap)
    n = np.array([side * np.sin(w) * np.cos(t), -np.cos(w) * np.cos(t), -np.sin(t)])
    up = G.unit(np.array([0.0, 0.0, 1.0]) - n[2] * n)
    return n, up, np.cross(up, n)


class Lens:
    """One lens and its rim: centre `c`, frame (n, up, ex), the rim's centreline radius R and the wire's radius r."""

    def __init__(self, side, c, frame, R, r):
        self.side, self.c, (self.n, self.up, self.ex), self.R, self.r = side, np.asarray(c, float), frame, R, r

    def at(self, phi, rad=None):
        """Points of the rim's centreline (or of a circle of radius `rad`) at angles `phi` (from ex toward up)."""
        rad = self.R if rad is None else rad
        phi = np.asarray(phi, float)[..., None]
        return self.c + rad * (np.cos(phi) * self.ex + np.sin(phi) * self.up)

    def moved(self, d):
        return Lens(self.side, self.c + d, (self.n, self.up, self.ex), self.R, self.r)

    def rim(self, m):
        return self.at(2.0 * np.pi * np.arange(m + 1) / m)

    def samples(self, rings=4, m=24):
        """Points over the lens's disc (what the exact check measures from)."""
        phi = 2.0 * np.pi * np.arange(m) / m
        return np.vstack([self.c[None]] + [self.at(phi, self.R * k / rings) for k in range(1, rings + 1)])

    def distance(self, P):
        return np.minimum(disc_distance(P, self.c, self.n, self.R), ring_distance(P, self.c, self.n, self.R, self.r))


def bridge_base(left, right, cfg, k=25):
    """The bridge's centreline before it bows forward: from the right lens's rim to the left's, `at` up their inner
    sides, rising `arch` in its middle; and the bow's profile along it (0 at the rims, 1 in the middle)."""
    a = float(np.arcsin(np.clip(float(cfg["at"]), -0.95, 0.95)))
    pr, pl = right.at(a), left.at(np.pi - a)
    t = np.linspace(0.0, 1.0, k)
    base = pr[None] + (pl - pr)[None] * t[:, None] + np.outer(np.sin(np.pi * t), [0.0, 0.0, float(cfg["arch"])])
    return base, np.sin(np.pi * t) ** 2


def level(v):
    """The level part of a vector, of unit length (zero when it has none)."""
    v = np.array([v[0], v[1], 0.0])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class Hull:
    """The head's triangles inside the box [lo, hi] in every state (or in `states`), each distinct one once: what
    the head is anywhere it goes there, for exact distances. `meshes` keeps only those meshes, `skip` leaves some out."""

    def __init__(self, face, lo, hi, meshes=None, skip=(), states=None):
        use = np.ones(len(face.V), bool) if meshes is None else np.isin(face.mesh, list(meshes))
        use &= ~np.isin(face.mesh, list(skip))
        T = face.T[use[face.T].all(1)]
        X = [np.zeros((0, 3, 3))]
        for s in face.states if states is None else states:
            Y = face.points(s)[T]
            X.append(Y[((Y.max(1) >= lo) & (Y.min(1) <= hi)).all(1)])
        X = (np.unique(np.round(np.concatenate(X).reshape(-1, 9) / 1e-7), axis=0) * 1e-7).reshape(-1, 3, 3)
        self.V, self.T = X.reshape(-1, 3), np.arange(3 * len(X)).reshape(-1, 3)
        self.c = X.mean(1)
        self.rad = np.linalg.norm(X - self.c[:, None], axis=2).max(1) if len(X) else np.zeros(0)

    def closest(self, q, reach=INF):
        """(distance, nearest point) of point q to the hull; INF (and q) when nothing of it is within `reach`."""
        q = np.asarray(q, float).reshape(3)
        near = np.flatnonzero(np.linalg.norm(self.c - q, axis=1) - self.rad <= reach)
        if not len(near):
            return INF, q
        p = closest_on_mesh(q[None], self.V, self.T[near], method="brute")[0][0]
        return float(np.linalg.norm(q - p)), p


def keep_off(q, head, hull, off, tries=8):
    """Point q pushed out level, along the outward normal of `head` (outfit_fit.Skin, a closed surface) nearest it,
    until it is `off` from `head` and from `hull` (Hull: the head's pieces in every state)."""
    d, _, tri, bary = head.closest(q[None])
    n = head.outward(tri, bary)[0]
    u = level(n)
    if d[0] < off:
        q = q + u * (off - d[0]) / max(float(np.hypot(n[0], n[1])), 0.3)
    for _ in range(tries):
        dh, near = hull.closest(q, off)
        if dh >= off - 1e-6:
            break
        cos = float((q - near) @ u) / max(dh, 1e-9)    # how directly a push along u leaves it
        q = q + u * (off - dh) / max(cos, 0.3)
    return q


def temple_path(lens, cfg, head, off, hull, ears, step=0.001):
    """A temple's centreline and the index where its run back starts: from the hinge on the rim's outer side (`at` a
    share of the radius above the centre) `end` out of the rim, then `length` along the head toward its back, rising
    `rise` degrees. Each step goes the most backward level way (turned toward her side in steps of 5 degrees) that keeps
    it `off` from `hull` (Hull: every piece of the head in every state, the skin at rest among them, and the lashes'
    and the brows' outer ends), so it follows the head round to its side without climbing over the brow, and is pushed
    out level where no way does (from `head`, outfit_fit.Skin, the closed skin, too); it ends early where it would come
    nearer than `off` to `ears` (Hull)."""
    a = float(np.arcsin(np.clip(float(cfg["at"]), -0.95, 0.95)))
    h = lens.at(a if lens.side > 0 else np.pi - a)
    corner = h + float(cfg["end"]) * G.unit(h - lens.c)
    climb = np.array([0.0, 0.0, np.tan(np.radians(float(cfg["rise"])))])
    n0 = max(int(float(cfg["end"]) / (0.5 * step)), 2)
    P = [p for p in np.linspace(h, corner, n0, endpoint=False)] + [corner]
    turns = np.radians(np.arange(0.0, 90.1, 5.0)) * lens.side
    k = 0
    for _ in range(max(int(float(cfg["length"]) / step), 2)):
        for k in range(max(k - 3, 0), len(turns)):      # from straight back toward her side: the first way that is clear
            q = P[-1] + step * (np.array([np.sin(turns[k]), np.cos(turns[k]), 0.0]) + climb)
            if hull.closest(q, off)[0] >= off - 1e-6:
                break
        else:
            q = keep_off(q, head, hull, off)
        if ears.closest(q, off)[0] < off:
            break                                       # the ear begins here
        P.append(q)
    P = np.array(P)
    for _ in range(8):                                  # round the bend over a few millimetres, then off the head again
        P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    P[n0:] = [keep_off(p, head, hull, off) for p in P[n0:]]
    return P, n0


def fit(face, eyes, skull, c, extra=(0.0, 0.0)):
    """The glasses where [glasses.place] puts them, `gap` off the head's vertices in every state, the lenses and the
    bridge `extra`[0] more, the temples `extra`[1] more: (lenses, bridge, temples, the eye opening's heights, the bridge
    before its bow)."""
    pc, r_out, r = c["place"], 0.5 * float(c["size"]), 0.5 * float(c["wire"])
    gap = float(pc["gap"]) + extra[0]
    open_ = [np.asarray(eyes[s]["opening"], float) for s in ("L", "R")]
    lo = float(np.mean([o[:, 1].min() for o in open_]))
    hi = float(np.mean([o[:, 1].max() for o in open_]))
    xc = float(pc["x"]) or float(np.mean([abs(float(np.asarray(eyes[s]["pupil"], float)[0])) for s in ("L", "R")]))
    top = lo + float(pc["height"]) * (hi - lo)
    lenses = []
    for side in (1.0, -1.0):
        fr = lens_frame(side, float(pc["tilt"]), float(pc["wrap"]))
        lenses.append(Lens(side, [side * xc, 0.0, top - (r_out - r) * fr[1][2] - r], fr, r_out - r, r))
    span = np.vstack([L.rim(16) for L in lenses])
    cloud = face.cloud(np.array([span[:, 0].min() - BOX, -INF, span[:, 2].min() - BOX]),
                       np.array([span[:, 0].max() + BOX, INF, span[:, 2].max() + BOX]))

    # the lenses: from in front of everything, straight back toward the face until the gap would not hold
    back = np.array([0.0, 1.0, 0.0])
    y0 = float(cloud[:, 1].min()) - gap - 2.0 * r - 0.002 if len(cloud) else -0.2
    lenses = [L.moved(np.array([0.0, y0, 0.0])) for L in lenses]

    def keeps(s):
        return min(least(L.moved(s * back).distance(cloud)) for L in lenses) >= gap
    s = search(keeps, 0.0, 0.002, 0.1)
    lenses = [L.moved(s * back) for L in lenses]
    left, right = lenses

    # the bridge: bowed forward over the nose as far as the gap asks
    fwd = G.unit((left.n + right.n) * [1.0, 1.0, 0.0])
    base, bow = bridge_base(left, right, c["bridge"])
    near_b = face.cloud(base.min(0) - BOX, base.max(0) + BOX)
    s = search(lambda s: least(path_distance(near_b, base - s * np.outer(bow, fwd), r)) >= gap, -0.03, 0.002, 0.0)
    bridge = base - s * np.outer(bow, fwd)

    # the temples: back along the side of the head, `gap` off the skin and off every other piece of the head in every
    # state (the lashes' and the brows' outer ends lie in their way), ending where the ears begin; the head looked at
    # in a band round the hinges' height (a temple never goes down, and rises `rise`)
    tc = c["temples"]
    reach = float(tc["length"]) + BOX
    a = float(np.arcsin(np.clip(float(tc["at"]), -0.95, 0.95)))
    hz = [float(L.at(a)[2]) for L in lenses]
    sp = np.vstack([L.rim(16) for L in lenses])
    lo_t = np.array([sp[:, 0].min() - reach, sp[:, 1].min() - BOX, min(hz) - 0.012])
    hi_t = np.array([sp[:, 0].max() + reach, sp[:, 1].max() + reach,
                     max(hz) + reach * np.tan(np.radians(max(float(tc["rise"]), 0.0))) + 0.012])
    hull = Hull(face, lo_t, hi_t, skip=("ears",))
    ears = Hull(face, lo_t, hi_t, meshes=("ears",), states=[()])
    X = skull.verts[skull.tris]
    band = skull.where(tri_mask=((X.max(1) >= lo_t - BOX) & (X.min(1) <= hi_t + BOX)).all(1))
    off = float(pc["gap"]) + extra[1] + r
    temples = [temple_path(lens, tc, band, off, hull, ears) for lens in lenses]
    return lenses, bridge, temples, dict(low=lo, high=hi, top=top), base


def search(ok, start, step, stop, tol=2e-6):
    """The last value from `start` toward `stop` (stepping `step`, then halving down to `tol`) for which `ok` holds
    (`ok(start)` is taken to hold)."""
    a = start
    while a + step <= stop and ok(a + step):
        a += step
    b = min(a + step, stop)
    while b - a > tol:
        m = 0.5 * (a + b)
        a, b = (m, b) if ok(m) else (a, m)
    return a


def exact_gap(face, pieces):
    """The least distance from each of the glasses' pieces ({name: (points, radius)}: centrelines of wire, or points
    over a lens with radius 0) to the head's triangles in every state: {name: (distance, state)}."""
    pts = np.vstack([p for p, _ in pieces.values()])
    lo, hi = pts.min(0) - BOX, pts.max(0) + BOX
    best = {name: (INF, ()) for name in pieces}
    for s in face.states:
        V = face.points(s)
        near = ((V[face.T] >= lo) & (V[face.T] <= hi)).all(2).any(1)
        if not near.any():
            continue
        d = np.linalg.norm(pts - closest_on_mesh(pts, V, face.T[near])[0], axis=1)
        k = 0
        for name, (p, rad) in pieces.items():
            dd = float(d[k:k + len(p)].min()) - rad
            k += len(p)
            if dd < best[name][0]:
                best[name] = (dd, s)
    return best


# ---------------------------------------------------------------- meshes
def wire_uv(patch):
    """Texture coordinates of the wire: u by how far up its surface faces (its top lit), one row."""
    n = G.vertex_normals(patch.v, patch.f)
    u = 0.5 + 0.5 * n[:, 2]
    patch.uv = np.array([[u[i], 0.5] for f in patch.f for i in f])
    return patch


def rim_patch(lens, segments, sides):
    """The rim: a ring of round wire round the lens."""
    phi = 2.0 * np.pi * np.arange(segments + 1) / segments
    psi = 2.0 * np.pi * np.arange(sides) / sides
    radial = np.cos(phi)[:, None] * lens.ex + np.sin(phi)[:, None] * lens.up
    centre = lens.c + lens.R * radial
    rings = centre[:, None, :] + lens.r * (np.cos(psi)[None, :, None] * radial[:, None, :]
                                           + np.sin(psi)[None, :, None] * lens.n[None, None, :])
    p = G.loft(rings, closed=True, mat=WIRE, tag="rim")
    return wire_uv(G.orient_outward(p, np.repeat(centre, sides, 0)))


def lens_patch(lens, rings=6, m=48):
    """The lens: a flat disc out to the wire's middle, facing out of the face; texture coordinates across it (u toward
    her left, v up)."""
    phi = 2.0 * np.pi * np.arange(m) / m
    loc = np.vstack([[0.0, 0.0]] + [np.stack([np.cos(phi), np.sin(phi)], 1) * (k / rings) for k in range(1, rings + 1)])
    V = lens.c + lens.R * (loc[:, :1] * lens.ex + loc[:, 1:] * lens.up)
    f = [(0, 1 + j, 1 + (j + 1) % m) for j in range(m)]
    for k in range(rings - 1):
        a, b = 1 + k * m, 1 + (k + 1) * m
        f += [(a + j, b + j, b + (j + 1) % m, a + (j + 1) % m) for j in range(m)]
    uv = 0.5 + 0.5 * loc
    if lens.side < 0:                                   # mirrored on her right lens: each glint at its own outer side
        uv[:, 0] = 1.0 - uv[:, 0]
    p = G.Patch(V, f, np.array([uv[i] for face in f for i in face]), None, LENS, "lens")
    if (G.face_normals(p.v, p.f) @ lens.n).sum() < 0.0:
        G.flip_patch(p)
    return p


# ---------------------------------------------------------------- pictures and materials
def wire_picture(col, w=64, h=4):
    """The wire's picture across u (how far up the surface faces): the frame colour, its light along the top."""
    u = (np.arange(w) + 0.5) / w
    t = G.smoothstep((u - 0.72) / 0.2)
    rgb = col["frame"][None] + (col["frame_light"] - col["frame"])[None] * t[:, None]
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[None]
    out[..., 3] = 255
    return out


def lens_picture(col, alpha, gc, R, size=512):
    """The lens's picture (row 0 its top) across its disc, `R` metres in radius: the lens colour at `alpha` (0: clear
    glass) and one crisp glint (`gc`): a band `width` metres wide along the circle at its place (`at`: degrees round
    from the lens's outer side toward its top, share of the radius), `length` metres long with round ends, the glint
    colour at its `alpha`, its edge one texel soft."""
    px = 2.0 * R / size
    x = ((np.arange(size) + 0.5) / size * 2.0 - 1.0) * R
    X, Y = np.meshgrid(x, -x)
    a, rr = np.radians(float(gc["at"][0])), float(gc["at"][1]) * R
    w, length = 0.5 * float(gc["width"]), float(gc["length"])
    half = max(0.5 * length - w, 0.0) / max(rr, 1e-6)    # half the angle of its middle; each round end adds w
    rho = np.hypot(X, Y)
    dth = np.angle(np.exp(1j * (np.arctan2(Y, X) - a)))
    ends = np.hypot(X[None] - rr * np.cos(a + np.array([-half, half]))[:, None, None],
                    Y[None] - rr * np.sin(a + np.array([-half, half]))[:, None, None]).min(0)
    dist = np.where(np.abs(dth) <= half, np.abs(rho - rr), ends) - w
    g = np.clip(0.5 - dist / px, 0.0, 1.0)
    A = np.clip(alpha + (float(gc["alpha"]) - alpha) * g, 0.0, 1.0) * (rho <= R)
    rgb = col["lens"][None, None] + (col["glint"] - col["lens"])[None, None] * g[..., None]
    out = np.empty((size, size, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = np.rint(A * 255.0).astype(np.uint8)
    return out


def ink(rgb, k=0.7):
    """The wire's outline: its own hue, darker."""
    h, s, v = colorsys.rgb_to_hsv(*(float(x) for x in rgb))
    return colorsys.hsv_to_rgb(h, min(1.0, s * 1.1), v * k)


def materials(ctx, col, c, R):
    """The wire's material and the lenses' (`R` the lens's radius in metres). Neither casts a shadow: the wire's
    would lie as a line on her cheeks."""
    wire = WR.material(ctx, MATS[WIRE], "glasses", col["frame"], wire_picture(col), shadow=(0.78, 0.72, 0.72),
                       edge_size=0.25, specular=(0.25, 0.22, 0.2), shininess=30.0, double_sided=False)
    wire.edge_color = tuple(float(x) for x in ink(col["frame"])) + (1.0,)
    wire.drop_shadow = wire.self_shadow_map = False
    lens = Material(MATS[LENS], name_en="glasses_lens", diffuse=(1.0, 1.0, 1.0, 1.0), ambient=(0.6, 0.6, 0.6),
                    texture=ctx.save_png("lens", lens_picture(col, float(c["alpha"]), c["glint"], R)),
                    double_sided=True, edge=False, drop_shadow=False, self_shadow_map=False, self_shadow=False,
                    alpha_blend=True)
    return [wire, lens]


# ---------------------------------------------------------------- the builder
@builder("glasses", needs=("body", "head"))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    col = WR.colours(ctx, "glasses", COLORS)
    head = ctx.need("head")
    eyes = head.info.get("eyes") or {}
    if not all(s in eyes and "opening" in eyes[s] and "pupil" in eyes[s] for s in ("L", "R")):
        raise ValueError("glasses: the head part publishes no eye openings and pupils (info['eyes'][L|R]['opening'], "
                         "['pupil'])")
    face = Face(head)
    sk = head.info["skin"]
    skin = [m for m in head.meshes if m.name == "face" and len(m.faces)]          # the imported face's own skin
    surface = OF.Skin.from_meshes(skin, "head") if skin else OF.Skin(np.asarray(sk["verts"], float),
                                                                     np.asarray(sk["faces"], int).reshape(-1, 3))
    r, gap = 0.5 * float(c["wire"]), float(c["place"]["gap"])
    extra = [0.0, 0.0]                                  # the lenses' and the bridge's, the temples'
    for _ in range(4):                                  # the quick fit's vertices, then the triangles exactly
        lenses, bridge, temples, eye, base = fit(face, eyes, surface, c, extra)
        pieces = {"rim.L": (lenses[0].rim(int(c["segments"])), r), "rim.R": (lenses[1].rim(int(c["segments"])), r),
                  "lens.L": (lenses[0].samples(), 0.0), "lens.R": (lenses[1].samples(), 0.0), "bridge": (bridge, r),
                  "temple.L": (temples[0][0], r), "temple.R": (temples[1][0], r)}
        found = exact_gap(face, pieces)
        short = [gap - min(d for k, (d, _) in found.items() if k.startswith("temple") == t) for t in (False, True)]
        if max(short) <= 5e-5:
            break
        extra = [e + s + 5e-5 if s > 5e-5 else e for e, s in zip(extra, short)]
    worst = min(found, key=lambda k: found[k][0])
    least_gap, state = found[worst]
    if least_gap < gap - 2e-4:
        ctx.log(f"WARNING glasses: {worst} comes {least_gap * 1000:.2f} mm near the head in "
                f"{'+'.join(state) or 'rest'} (gap {gap * 1000:.1f} mm)")

    bone, seg, sides = str(c["bone"]), int(c["segments"]), int(c["sides"])
    soup = G.Soup("glasses", list(MATS))
    wire = [rim_patch(L, seg, sides) for L in lenses] + G.tube(bridge, r, n=sides, caps=(False, False), mat=WIRE)
    for P, _ in temples:
        wire += G.tube(P, r, n=sides, caps=(False, True), mat=WIRE)
    for p in wire:
        wire_uv(p).w = {bone: np.ones(len(p.v))}
        soup.add(p, WIRE)
    for L in lenses:
        p = lens_patch(L)
        p.w = {bone: np.ones(len(p.v))}
        soup.add(p, LENS)

    left, right = lenses
    tops = [float(L.rim(seg)[:, 2].max() + r) for L in lenses]
    mid = bridge[len(bridge) // 2]
    info = {"size": 2.0 * (left.R + r), "wire": 2.0 * r, "centres": {"L": left.c, "R": right.c},
            "apart": float(left.c[0] - right.c[0]),
            "eye": {**eye, "rims_top": float(np.mean(tops)),
                    "rims_top_share": float((np.mean(tops) - eye["low"]) / (eye["high"] - eye["low"]))},
            "gap": {"least": float(least_gap), "piece": worst, "morphs": list(state), "extra": list(extra),
                    "pieces": {k: float(d) for k, (d, _) in found.items()}},
            "bridge": {"middle": mid, "bow": float(np.linalg.norm(mid - base[len(base) // 2])),
                       "over_nose": least(path_distance(face.cloud(mid - 0.03, mid + 0.03), bridge, r))},
            "temples": {"length": float(np.linalg.norm(np.diff(temples[0][0][temples[0][1]:], axis=0), axis=1).sum()),
                        "back": [float(P[-1, 1] - P[k, 1]) for P, k in temples],
                        "out": [float(abs(P[-1, 0]) - abs(P[k, 0])) for P, k in temples],
                        "ends": {"L": temples[0][0][-1], "R": temples[1][0][-1]}},
            "glint": {"length": float(c["glint"]["length"]), "width": float(c["glint"]["width"])},
            "landmarks": {"glasses.L": left.c, "glasses.R": right.c, "glasses_bridge": mid}}
    info["bridge"]["length"] = float(np.linalg.norm(bridge[-1] - bridge[0]))
    t = info["temples"]
    ctx.log(f"lenses {info['size'] * 1000:.1f} mm across, wire {info['wire'] * 1000:.1f} mm, centres "
            f"{info['apart'] * 1000:.1f} mm apart at z {left.c[2]:.4f}, y {left.c[1]:.4f}; rims' tops at z "
            f"{info['eye']['rims_top']:.4f} ({info['eye']['rims_top_share']:.2f} of the eye opening); least gap "
            f"{least_gap * 1000:.2f} mm ({worst}, {'+'.join(state) or 'rest'}); bridge "
            f"{info['bridge']['length'] * 1000:.0f} mm, {info['bridge']['over_nose'] * 1000:.1f} mm over the nose; "
            f"temples {t['length'] * 1000:.0f} mm along the head, ending {t['back'][0] * 1000:.0f} mm behind and "
            f"{t['out'][0] * 1000:.0f} mm outside their hinges")
    return Part("glasses", meshes=[soup.mesh()], materials=materials(ctx, col, c, left.R), info=info)
