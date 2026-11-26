"""A folded paper plane, bpy-free (the library prop `paper_plane`, docs/design.md: the library table): the classic dart
folded from one sheet of construction paper, made big enough to ride, and its motion (bank, pitch, the wings' flutter,
a landing's dip), held on a step grid as the paper theatre's pieces are.

The sheet is W x L (W = `sheet` x L, the length L = `size`; u across it from its middle, v along it from the nose end)
and folded as a child folds a dart: (1) a crease down the middle; (2) the two top corners to the middle (those little
triangles end up hidden inside the nose and are left out here); (3) the new slanted edges to the middle again, along F2
(u = v tan 22.5 degrees): each flap (R2) lies on the sheet with its free edge, the sheet's side edge, now a diagonal from
the middle crease out to the wing's tip; (4) in half along the middle, the flaps inside; (5) each side down along G, a
line from the nose to `keel` x L under the top edge at the tail: the paper under G is the keel, the paper over it the
wing, folded out level. The wing is two plies where the flap lies on it (its whole front, a triangle from the nose to the
tip) and its leading edge is the fold F2, the paper turned right back over itself: a rounded bead two plies thick.

Every other fold is a crease: the flat facets are slabs of paper PLY x L thick whose ends meet on the plane that bisects
their fold, as a picture frame's mitre does, and every slab is the convex solid its planes bound (`polyhedron`), so the
plies stack, the keel's two walls fuse into the bottom crease and everything runs to a point at the nose without a gap or
a crossing. The edges are rolled as the paper theatre's paper is (a bevel ROLL of the ply into each face) and the
outsides of the folds rounded off (CREASE_ROLL), so its big flat faces never end in a hard rim (the `form` check: 0.03).
Folds are straight and the sheet's edges are its factory edges: nothing is scissor-cut but the doodles.

Frame: origin on the spine at half the length, in the plane of the wings' undersides (the wings level); +Z up, front -Y
(the nose at y = -L/2), +X the plane's left. The keel is a narrow V (its walls lean KEEL_V degrees in, meeting on the
bottom crease at x = 0), so the wings' roots run out a little toward the tail and a slit opens along the spine.

Faces: the sheet's outside is `sheet` on the wings' undersides and on the flaps that show on the wings' tops, and `keel`
on the keel's outside (the same face of the paper, a little darker: a separate piece of paper to the cut-out look); its
inside `inner` (the wing tops behind the flaps, the keel's inside); its edges `edge`. Doodles: a cut-paper star glued on
each wing (top and underneath) and on each side of the keel, a cut-paper stripe along each wing's leading edge, and pen
speed lines behind the top stars.

Motion (clip seconds; everything keyed on `step` / `step_from` as mkmmd.core.paper_theatre.key_times does): `bank`
rolls it about its long axis (+ its left wing down), `pitch` lifts its nose (+), both on the ROOT (rotation mode YXZ:
the bank first, then the pitch, then the heading, `rotation_euler` index 2, which `[[prop]] yaw` or a `[[key]]` sets), so
a figure seated on it (whose root the pose stage parents to the prop's root) banks and pitches with it; `land` dips the
root (its `delta_location` z, so `at` and any `[[key]]` on `location` still fly it) and flexes the wings; `flutter`
flexes each wing about its root crease."""
import functools
import math
from itertools import combinations, pairwise

import numpy as np

from . import paper_theatre as PT
from . import shell as S

# colour roles (docs: ROLE_DOCS); materials are named <prop>_<role>
ROLES = {"sheet": "foam:surface:0.3", "inner": "foam:surface:0.62", "keel": "foam:pine:0.22", "edge": "foam:surface:0.82",
         "star": "love", "stripe": "gold", "ink": "text"}
ROLE_DOCS = {"sheet": "the paper's outside: the wings' undersides and the flaps on the wings' tops",
             "inner": "the paper's inside: the wings' tops behind the flaps, the keel's inside",
             "keel": "the keel's outside (the same face as sheet, a little darker)",
             "edge": "the paper's edges: the leading edges (folded, two plies thick), the tail, the tips, the flaps' edges",
             "star": "the cut-paper stars on the wings and the keel", "stripe": "the cut-paper stripes along the leading edges",
             "ink": "the pen speed lines behind the stars"}
MATS = tuple(ROLES)
MI = {k: i for i, k in enumerate(MATS)}

PLY = 1.0 / 250.0                # the paper's thickness over the sheet's length (6 mm at 1.5 m: a toy's thick paper)
GAP = 0.15                       # the air between two plies, over the paper's thickness
ROLL = 0.35                      # a cut edge rolls this share of its ply into each face (as the paper theatre's paper)
CREASE_ROLL = 0.5                # the outside of a fold is rounded off this share of the ply
KEEL_V = 11.0                    # degrees each keel wall leans in from the vertical, toward the bottom crease (past 10:
#                                  a wing over an upright wall is a box corner to the form check)
T22 = math.tan(math.radians(22.5))
REACH = 0.24                     # m: the leading edge this far ahead of the rider's hips, square to it (her knees past it)
HIP = 0.085                      # m: her hip joints over the paper (a figure 1.5 m tall)
FOOT = (0.38, 0.09, 0.45)        # m: her ankles this far ahead of her hips, this far to either side; her floor this far down
BEAD_ARC = 8                     # segments round the leading edge's fold
FLUTTER_HZ = 2.4                 # the flutter's main rate (a second, faster one rides on it)
LAND = {"hz": 1.6, "decay": 0.3, "flex": 5.0, "flex_hz": 3.0, "flex_decay": 0.35}   # a landing's dip and the wings' kick
DEFAULTS = {"size": 1.5, "sheet": 0.773, "keel": 0.11, "seat": None, "side": "L", "bank": 0.0, "pitch": 0.0,
            "flutter": 2.0, "land": (), "sink": 0.05, "step": 0.0, "step_from": 0.0, "seed": 1, "doodles": True}
NAME = "paper_plane"


# ===================================================================================================================
# options
# ===================================================================================================================
def _num(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
        raise ValueError(f"{NAME}: {what} is a number from {lo:g} to {hi:g} (got {v!r})")
    return float(v)


def _keys(v, what, lo, hi):
    """A number (held all along) or keys [[t, value], ...] in time order -> ((t,), (value,)) tuples."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (0.0,), (_num(v, what, lo, hi),)
    if not isinstance(v, (list, tuple)) or not v or not all(isinstance(k, (list, tuple)) and len(k) == 2 for k in v):
        raise ValueError(f"{NAME}: {what} is a number or keys [[t, value], ...] in clip seconds (got {v!r})")
    ts = tuple(_num(k[0], f"{what}'s time", -1e4, 1e4) for k in v)
    if any(b <= a for a, b in pairwise(ts)):
        raise ValueError(f"{NAME}: {what}'s keys are in time order, one a time (got {list(ts)})")
    return ts, tuple(_num(k[1], what, lo, hi) for k in v)


def options(slots):
    """The prop's options from its `slots`, checked: {size, sheet, keel, seat (a share of the length from the nose, or
    None: where the leading edge is REACH ahead of her), side, bank, pitch, flutter (each ((t,), (value,))), land (sorted
    clip seconds), sink, step, step_from, seed, doodles}. Refuses what it cannot build, saying what to change."""
    slots = dict(slots or {})
    O = {k: _num(slots.get(k, DEFAULTS[k]), k, lo, hi) for k, lo, hi in (
        ("size", 0.2, 20.0), ("sheet", 0.6, 0.8), ("keel", 0.05, 0.2), ("sink", 0.0, 2.0), ("step", 0.0, 240.0),
        ("step_from", -1e4, 1e4))}
    seat = slots.get("seat", DEFAULTS["seat"])
    O["seat"] = None if seat is None else _num(seat, "seat", 0.25, 0.8)
    side = slots.get("side", DEFAULTS["side"])
    if side not in ("L", "R"):
        raise ValueError(f"{NAME}: side is \"L\" (her legs over the left wing, her right side to the nose) or \"R\" "
                         f"(got {side!r})")
    O["side"] = side
    O["bank"] = _keys(slots.get("bank", DEFAULTS["bank"]), "bank", -90.0, 90.0)
    O["pitch"] = _keys(slots.get("pitch", DEFAULTS["pitch"]), "pitch", -60.0, 60.0)
    O["flutter"] = _keys(slots.get("flutter", DEFAULTS["flutter"]), "flutter", 0.0, 20.0)
    land = slots.get("land", DEFAULTS["land"])
    land = [land] if isinstance(land, (int, float)) and not isinstance(land, bool) else land
    if not isinstance(land, (list, tuple)):
        raise ValueError(f"{NAME}: land is a clip second or a list of them (got {land!r})")
    O["land"] = sorted(_num(v, "land", -1e4, 1e4) for v in land)
    seed = slots.get("seed", DEFAULTS["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 10 ** 6:
        raise ValueError(f"{NAME}: seed is a whole number (got {seed!r})")
    O["seed"] = seed
    doodles = slots.get("doodles", DEFAULTS["doodles"])
    if not isinstance(doodles, bool):
        raise ValueError(f"{NAME}: doodles is true or false (got {doodles!r})")
    O["doodles"] = doodles
    return O


# ===================================================================================================================
# the folded sheet
# ===================================================================================================================
class Fold:
    """The folded sheet's numbers and its maps, for the left side (sheet u >= 0, the plane's +X; the right side is its
    mirror image in x). (a, b) are sheet coordinates turned to G: b along it from the nose, a square to it (+ toward the
    wing, - down the keel). A ply's offset h runs from the sheet's outside (h = 0) toward its inside."""

    def __init__(self, size=DEFAULTS["size"], sheet=DEFAULTS["sheet"], keel=DEFAULTS["keel"]):
        L = self.L = float(size)
        H = self.H = 0.5 * float(sheet) * L
        s = self.s = keel / math.sqrt(1.0 - keel * keel)          # G: u = s v; the keel's depth at the tail = keel x L
        self.c = math.sqrt(1.0 + s * s)
        self.t = PLY * L
        self.g = GAP * self.t
        chi = math.radians(KEEL_V)
        psi = self.psi = math.atan(s * math.sin(chi))              # the wing roots' turn out: the walls meet at x = 0
        self.G3 = np.array([math.sin(psi), math.cos(psi), 0.0])
        self.W3 = np.array([math.cos(psi), -math.sin(psi), 0.0])
        self.D3 = -math.cos(chi) * np.array([0.0, 0.0, 1.0]) - math.sin(chi) * self.W3
        self.IN = np.cross(self.G3, self.D3)                       # the wall's inside: toward the other wall
        self.M = (self.W3 - self.D3) / np.linalg.norm(self.W3 - self.D3)   # the root crease's mitre, toward the wing
        self.N3 = np.array([0.0, -0.5 * L, 0.0])
        if H / T22 >= L or s * L >= H:
            raise ValueError(f"{NAME}: sheet {sheet:g} and keel {keel:g} fold no dart (the slanted edge must end before "
                             f"the tail and the keel be narrower than the wing)")
        r2 = math.sqrt(2.0) * H
        lam = r2 * s / (H - s * (H / T22 - r2))                     # G across the flap's free edge C2 -> TC
        self.uv = {"N": (0.0, 0.0), "TC": (H, H / T22), "TT": (H, L), "GT": (s * L, L), "BT": (0.0, L),
                   "C2": (0.0, r2), "PX": (lam * H, r2 + lam * (H / T22 - r2))}

    def ab(self, uv):
        u, v = np.asarray(uv, float).T
        return np.stack([(u - self.s * v) / self.c, (self.s * u + v) / self.c], -1)

    def P(self, name):
        """A named point's (a, b)."""
        return self.ab([self.uv[name]])[0]

    def wing(self, a, b, h=0.0):
        """Points of the wing (a >= 0), lying level: (n, 3)."""
        a, b, h = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(h, float))
        return self.N3 + b[..., None] * self.G3 + a[..., None] * self.W3 + h[..., None] * np.array([0.0, 0.0, 1.0])

    def wall(self, a, b, h=0.0):
        """Points of the keel's wall (a <= 0), its inside toward the other wall: (n, 3)."""
        a, b, h = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(h, float))
        return self.N3 + b[..., None] * self.G3 - a[..., None] * self.D3 + h[..., None] * self.IN

    def le_dir(self):
        """The leading edge's direction (nose to tip) and its outward square in plan, left side."""
        d = self.wing(*self.P("TC")) - self.N3
        d /= np.linalg.norm(d)
        return d, np.array([d[1], -d[0], 0.0])

    def depth(self, b):
        """How deep the keel hangs under the spine at b (to its bottom crease, along the wall)."""
        return self.s * np.asarray(b, float)


# ===================================================================================================================
# a convex slab from its planes
# ===================================================================================================================
def polyhedron(planes, tol=1e-7):
    """The convex solid {x: n . x <= d for every (n, d, role) in `planes`} as a closed Mesh (outward faces, each face
    its plane's material `role`; a plane given twice counts once), or None when it is empty or flat. Faces of more than
    four corners are fanned from their middle, so a corner that lies on an edge keeps the mesh closed."""
    Nn, D, roles = [], [], []
    for n, d, role in planes:
        k = float(np.linalg.norm(n))
        n, d = np.asarray(n, float) / k, float(d) / k
        if not any(np.allclose(n, m, atol=1e-12) and abs(d - e) < 1e-12 for m, e in zip(Nn, D)):
            Nn.append(n)
            D.append(d)
            roles.append(role)
    Nn, D = np.array(Nn), np.array(D)
    V = []
    for i, j, k in combinations(range(len(D)), 3):
        A = Nn[[i, j, k]]
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        x = np.linalg.solve(A, D[[i, j, k]])
        if np.all(Nn @ x <= D + 1e-9) and not any(np.linalg.norm(x - v) < tol for v in V):
            V.append(x)
    if len(V) < 4:
        return None
    V = np.array(V)
    if np.linalg.matrix_rank(V[1:] - V[0], tol=tol) < 3:
        return None
    verts, Q, T, Qm, Tm = [V], [], [], [], []
    nv = len(V)
    for n, d, role in zip(Nn, D, roles):
        on = np.nonzero(np.abs(V @ n - d) < tol)[0]
        if len(on) < 3:
            continue
        c = V[on].mean(0)
        e1 = V[on[0]] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ring = on[np.argsort(np.arctan2((V[on] - c) @ e2, (V[on] - c) @ e1))]   # counter-clockwise seen from outside
        P = V[ring] - c
        if 0.5 * np.linalg.norm(np.cross(P, np.roll(P, -1, 0)).sum(0)) < tol * tol:
            continue                                                # a plane that only touches an edge or a corner
        if len(ring) == 3:
            T.append(ring)
            Tm.append(role)
        elif len(ring) == 4:
            Q.append(ring)
            Qm.append(role)
        else:
            verts.append(c[None])
            T += [(nv, ring[q], ring[(q + 1) % len(ring)]) for q in range(len(ring))]
            Tm += [role] * len(ring)
            nv += 1
    return S.Mesh(np.concatenate(verts), np.array(Q, np.int64).reshape(-1, 4), np.array(T, np.int64).reshape(-1, 3),
                  np.array(Qm, np.int32), np.array(Tm, np.int32))


def _bevel(n1, d1, n2, d2, c):
    """The plane (n, d) that cuts the corner where the faces n1 . x = d1 and n2 . x = d2 meet, `c` back along each."""
    n = n1 + n2
    k = float(np.linalg.norm(n))
    half = 0.5 * math.acos(float(np.clip(n1 @ n2, -1.0, 1.0)))
    return n / k, (d1 + d2) / k - c * math.sin(half)


def _slab(F, kind, poly, h0, h1, edges, faces):
    """One ply of one facet, left side: `kind` "wing" or "wall", its outline `poly` (k, 2) in (a, b), between the
    offsets h0 and h1; `edges[i]` how the side from corner i to i + 1 ends: "cut" (square to the paper, its corners
    bevelled ROLL of the ply into both faces, as the theatre's paper rolls), "fold" (the same, under the leading edge's
    bead), "crease" (the root crease's mitre, the fold's outside bevelled CREASE_ROLL of the ply by one plane shared with
    the facet across it) or "bottom" (the bottom crease: the left side's x = 0 bound is its mitre, its outside rounded
    by a flat along it and a bevel); `faces` the roles of (the h0 face, the h1 face, the cut sides, the hidden ones)."""
    out_role, in_role, cut_role, hidden = (MI[r] for r in faces)
    to3 = F.wing if kind == "wing" else F.wall
    nrm = np.array([0.0, 0.0, 1.0]) if kind == "wing" else F.IN
    z = np.array([0.0, 0.0, 1.0])
    t = h1 - h0
    P3 = to3(poly[:, 0], poly[:, 1], np.full(len(poly), h0))
    cen = P3.mean(0)
    base = float(nrm @ P3[0])
    lo, hi = (-nrm, -base), (nrm, base + t)
    planes = [(*lo, out_role), (*hi, in_role), (np.array([-1.0, 0.0, 0.0]), 0.0, hidden)]   # the left side: x >= 0
    for i, how in enumerate(edges):
        p, q = P3[i], P3[(i + 1) % len(P3)]
        if how == "crease":
            n, p = F.M.copy(), F.N3
        elif how == "bottom":
            n = np.cross(np.array([1.0, 0.0, 0.0]), q - p)                     # square to the crease, in x = 0
        else:
            n = np.cross(q - p, nrm)
        n = n / np.linalg.norm(n)
        if n @ (cen - p) > 0:
            n = -n
        d = float(n @ p)
        if how == "bottom":
            flat = (n, d - CREASE_ROLL * t)
            planes += [(*flat, out_role), (*_bevel(*lo, *flat, CREASE_ROLL * t), out_role)]
            continue
        planes.append((n, d, cut_role if how in ("cut", "fold") else hidden))
        if how in ("cut", "fold"):
            planes += [(*_bevel(*lo, n, d, ROLL * t), cut_role), (*_bevel(*hi, n, d, ROLL * t), cut_role)]
        elif how == "crease":                                    # the fold's outside: the wing's top and the keel's inside
            planes.append((*_bevel(z, float(z @ F.N3) + h1, F.IN, float(F.IN @ F.N3) + h1, CREASE_ROLL * t), in_role))
    return polyhedron(planes)


def _bead(F):
    """The leading edge: the paper turned back over itself along F2, from the nose to the tip, a half ring two plies
    thick round the line between the plies (its round outside the paper's `edge`, as every edge of the sheet is; its inside
    within; the ends capped), left side."""
    t, g = F.t, F.g
    a0, b0 = F.P("N")
    a1, b1 = F.P("TC")
    n = max(math.ceil(math.hypot(a1 - a0, b1 - b0) / (4.0 * t)), 8)
    w = np.linspace(0.0, 1.0, n + 1)
    C = F.wing(a0 + w * (a1 - a0), b0 + w * (b1 - b0), np.full(n + 1, t + 0.5 * g))
    _d, o = F.le_dir()
    z = np.array([0.0, 0.0, 1.0])
    th = np.linspace(-0.5 * math.pi, 0.5 * math.pi, BEAD_ARC + 1)
    ro, ri = t + 0.5 * g, 0.5 * g
    ring = np.concatenate([ro * (np.cos(th)[:, None] * o + np.sin(th)[:, None] * z),
                           ri * (np.cos(th[::-1])[:, None] * o + np.sin(th[::-1])[:, None] * z)])
    m = S.skin(C[:, None, :] + ring[None], caps=(True, True), closed=True)
    m.Qm = np.where(np.arange(len(m.Q)) % len(ring) < BEAD_ARC, MI["edge"], MI["inner"]).astype(np.int32)
    m.Tm = np.full(len(m.T), MI["edge"], np.int32)
    return m if m.volume() > 0 else m.flip()


def _flat_piece(F, outline, at, frame, h, thick, role, r_edge):
    """A piece of paper glued flat on the plane: `outline` (k, 2) about its middle in the piece's own (x, y), its middle
    at `at` (a, b) of the facet `frame` ("top": a wing's top, "under": its underside, "keel": the keel's outside), lying
    `h` off the plane of that face, `thick` thick."""
    m = S.rounded_panel(np.asarray(outline, float), thick, r_edge=r_edge, n=2, mat=role)
    a, b = at
    if frame == "keel":
        R = np.column_stack([F.G3, -F.D3, -F.IN])                  # x toward the tail, y up the wall, z out of it
        o = F.wall(a, b) - F.IN * h
    elif frame == "under":
        R = np.column_stack([F.W3, -F.G3, [0.0, 0.0, -1.0]])        # x out, y toward the nose, z down
        o = F.wing(a, b, -h)
    else:
        R = np.column_stack([-F.W3, -F.G3, [0.0, 0.0, 1.0]])        # y toward the nose, z up
        o = F.wing(a, b, h)
    return m.apply(R, o)


def _doodles(F, seed):
    """The doodles, left side: {"wing": the cut-paper stars on the wing's top (on its inside, between the flap's free edge
    and the root, halfway back from where the edge leaves the root) and underneath, three pen speed lines trailing the top
    one to near the trailing edge, the cut-paper stripe along the leading edge (on the flap); "keel": a little star on the
    keel's outside, near the tail}. Each piece a closed mesh with its role."""
    rng = np.random.default_rng([int(seed), 7])
    t = F.t
    td, lift = 0.5 * t, 0.0003
    a_tc, b_tc = F.P("TC")
    a_px, b_px = F.P("PX")
    a_tt, b_tt = F.P("TT")
    b_gt = F.P("GT")[1]

    def diag(b):                                                              # the flap's free edge across the top
        return a_px + (a_tc - a_px) * np.clip((b - b_px) / (b_tc - b_px), 0.0, 1.0)

    def trailing(a):                                                          # the trailing edge, GT -> TT
        return b_gt + (b_tt - b_gt) * a / a_tt
    b_c = b_px + 0.47 * (b_gt - b_px)
    c = np.array([0.5 * float(diag(b_c)), b_c])
    r = 0.55 * c[0]
    wing = [_flat_piece(F, PT.star_outline(r, rng), c, "top", t + lift, td, MI["star"], 0.35 * td),
            _flat_piece(F, PT.star_outline(0.85 * r, rng), c + [0.0, 0.1 * r], "under", lift, td, MI["star"],
                        0.35 * td)]
    for k, dx in enumerate((-0.45, 0.0, 0.45)):                               # speed lines trailing the star
        a = c[0] + dx * r
        y0 = c[1] + 1.2 * r + (0.0 if k == 1 else 0.25 * r)
        ys = np.linspace(y0, trailing(a) - (0.5 if k == 1 else 0.9) * r, 16)
        xs = a + 0.03 * r * np.sin(np.linspace(0.0, 2.0 * math.pi, 16) + rng.uniform(0.0, 3.0))
        w, h = 0.12 * r, 0.3 * t                                              # a pen line: round-edged, flat on the paper
        wing.append(S.tube(F.wing(xs, ys, np.full(16, t + lift + 0.5 * h)), 0.5 * w, sides=10, aspect=h / w,
                           mat=MI["ink"]))
    kap = a_tc / b_tc                                                         # the stripe along the leading edge, on
    cl = math.cos(math.atan(kap))                                             # the flap: its inner side clear of the
    i0, i1 = 2.6 * t + 0.008 * F.L, 2.6 * t + 0.032 * F.L                     # flap's free edge
    bs = np.linspace(0.5 * b_tc, b_tc, 400)
    fits = bs[kap * bs - i1 / cl >= diag(bs) + 0.01 * F.L]
    b1, b2 = 0.5 * b_tc, float(fits.max()) if len(fits) else 0.9 * b_tc
    corners = np.array([(kap * b1 - i1 / cl, b1), (kap * b2 - i1 / cl, b2), (kap * b2 - i0 / cl, b2),
                        (kap * b1 - i0 / cl, b1)]) + rng.uniform(-0.0015, 0.0015, (4, 2)) * F.L
    mid = corners.mean(0)
    m = S.rounded_panel(PT.densify(PT.ccw(corners - mid), 0.01), td, r_edge=0.35 * td, n=2, mat=MI["stripe"])
    wing.append(m.apply(np.column_stack([F.W3, F.G3, [0.0, 0.0, 1.0]]), F.wing(mid[0], mid[1], 2.0 * t + F.g + lift)))
    bk = 0.8 * F.L
    dk = float(F.depth(bk))
    keel = [_flat_piece(F, PT.star_outline(0.3 * dk, rng), (-0.48 * dk, bk), "keel", lift, td, MI["star"], 0.35 * td)]
    return {"wing": wing, "keel": keel}


def _left(F, doodles=True, seed=1):
    """The left side's meshes in the prop's frame: (the keel's wall and the flap's part inside it, the wing with its
    flap, its leading edge and its doodles)."""
    t, g = F.t, F.g
    pN, pTC, pTT, pGT, pBT, pC2, pPX = (F.P(k) for k in ("N", "TC", "TT", "GT", "BT", "C2", "PX"))
    flap = ("inner", "sheet", "edge", "inner")
    wing = [_slab(F, "wing", np.array([pN, pTC, pTT, pGT]), 0.0, t, ("fold", "cut", "cut", "crease"),
                  ("sheet", "inner", "edge", "inner")),
            _slab(F, "wing", np.array([pN, pTC, pPX]), t + g, 2.0 * t + g, ("fold", "cut", "crease"), flap),
            _bead(F)]
    body = [_slab(F, "wall", np.array([pN, pGT, pBT]), 0.0, t, ("crease", "cut", "bottom"),
                  ("keel", "inner", "edge", "inner")),
            _slab(F, "wall", np.array([pN, pPX, pC2]), t + g, 2.0 * t + g, ("crease", "cut", "cut"), flap)]
    if doodles:
        d = _doodles(F, seed)
        wing += d["wing"]
        body += d["keel"]
    return S.merge([m for m in body if m is not None]), S.merge([m for m in wing if m is not None])


MIRROR = np.diag([-1.0, 1.0, 1.0])


@functools.lru_cache(maxsize=8)
def parts(size=DEFAULTS["size"], sheet=DEFAULTS["sheet"], keel=DEFAULTS["keel"], doodles=True, seed=1):
    """The meshes in the prop's frame, at rest (cached: read-only): {"body": the keel (both walls and what is inside
    them, the keel stars), "wing_L", "wing_R": each wing with its flap, its leading edge and its doodles}."""
    F = Fold(size, sheet, keel)
    body, wing = _left(F, doodles, seed)
    return {"body": S.merge([body, body.apply(MIRROR)]), "wing_L": wing, "wing_R": wing.apply(MIRROR)}


def hinge(F, side):
    """A wing's hinge: (origin (3,), yaw (radians about Z)) of its object, whose local Y runs along the root crease
    from the nose; the flutter turns it about that Y."""
    return F.N3.copy(), (-F.psi if side == "L" else F.psi)


def tip(F, side):
    """The wing tip (the outermost point of the wing: the leading edge's end), between its plies, prop frame."""
    p = F.wing(*F.P("TC"), F.t + 0.5 * F.g)
    return p if side == "L" else MIRROR @ p


def top_z(F, a, b):
    """The paper's top over the wing at (a, b), left side: on the flap where it lies (two plies), else one ply."""
    tri = np.array([F.P("N"), F.P("TC"), F.P("PX")])
    on = PT.inside(np.array([[a, b]]), tri)[0]
    return 2.0 * F.t + F.g if on else F.t


# ===================================================================================================================
# where she sits
# ===================================================================================================================
def seat_b(F, seat=None):
    """b of the seat on the spine: `seat` x the length, or (None) where the leading edge is REACH ahead of her, square to
    it, kept within 0.3 .. 0.7 of the length (the edge runs from the nose, so a spine point b along is b G3y dx from it)."""
    if seat is not None:
        return float(seat) * F.L
    d, _o = F.le_dir()
    return float(np.clip(REACH / (F.G3[1] * d[0]), 0.3 * F.L, 0.7 * F.L))


def seat(F, O):
    """The rider's use points (prop frame): {b, hip, facing, floor_z, seat_z, feet {L, R}}. She sits on the spine at
    seat_b, side-saddle: both legs over the `side` wing, facing square to its leading edge (side L: toward +X and a little
    forward, her right side toward the nose), her knees past the edge and her shins hanging below it."""
    b = seat_b(F, O["seat"])
    z_top = top_z(F, 0.0, b)
    sx = 1.0 if O["side"] == "L" else -1.0
    _d, o = F.le_dir()
    facing = np.array([sx * o[0], o[1], 0.0])
    left = np.array([-facing[1], facing[0], 0.0])                        # her left
    hip = np.array([0.0, F.N3[1] + b * F.G3[1], z_top + HIP])
    floor = hip[2] - FOOT[2]
    feet = {k: hip + facing * FOOT[0] + left * (FOOT[1] if k == "L" else -FOOT[1]) for k in ("L", "R")}
    for foot in feet.values():
        foot[2] = floor
    return {"b": b, "hip": hip, "facing": facing, "floor_z": floor, "seat_z": z_top, "feet": feet}


# ===================================================================================================================
# the motion
# ===================================================================================================================
def _curve(keys, t):
    """Keys ((t,), (v,)) at clip seconds t: monotone cubic through them, held before the first and after the last."""
    ts, vs = keys
    t = np.asarray(t, float)
    if len(ts) == 1:
        return np.full(len(t), vs[0])
    return S.pchip(np.array(ts), np.array(vs)[:, None], np.clip(t, ts[0], ts[-1]))[:, 0]


def motion(O, t):
    """The plane's motion at clip seconds t (n,), as the builder keys it: {"bank": radians, the root's rotation_euler[1]
    (+ its left wing down); "pitch": radians nose up (the root's rotation_euler[0] is minus it); "dip": m, the root's
    delta_location z (a landing: from 0 ON its time, down `sink` x 0.6 and back); "flex_L", "flex_R": radians each wing
    is flexed up about its root crease (the flutter and a landing's kick, tips down first)}."""
    t = np.asarray(t, float)
    amp = np.radians(_curve(O["flutter"], t))
    w = 2.0 * math.pi * FLUTTER_HZ
    flex = {"L": amp * (0.65 * np.sin(w * t) + 0.35 * np.sin(1.73 * w * t + 1.1)),
            "R": amp * (0.65 * np.sin(w * t + 2.3) + 0.35 * np.sin(1.73 * w * t + 0.4))}
    dip = np.zeros(len(t))
    for tl in O["land"]:
        tau = np.maximum(t - tl, 0.0)
        on = (t >= tl).astype(float)
        dip += -O["sink"] * np.exp(-tau / LAND["decay"]) * np.sin(2.0 * math.pi * LAND["hz"] * tau) * on
        kick = -math.radians(LAND["flex"]) * np.exp(-tau / LAND["flex_decay"]) * np.sin(
            2.0 * math.pi * LAND["flex_hz"] * tau) * on
        flex["L"] = flex["L"] + kick
        flex["R"] = flex["R"] + kick
    return {"bank": np.radians(_curve(O["bank"], t)), "pitch": np.radians(_curve(O["pitch"], t)), "dip": dip,
            "flex_L": flex["L"], "flex_R": flex["R"]}


def moves(O):
    """Which channels the builder keys: {"turn": bank or pitch is ever off level, "dip": something lands, "flex": the
    wings flutter or a landing kicks them}."""
    return {"turn": any(v != 0.0 for v in O["bank"][1] + O["pitch"][1]), "dip": bool(O["land"]),
            "flex": any(v > 0.0 for v in O["flutter"][1]) or bool(O["land"])}


def euler_yxz(rx, ry, rz):
    """Blender's Euler of rotation mode YXZ as a matrix: the Y turn first, then X, then Z (Rz Rx Ry)."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Rx @ Ry


def root_matrix(bank, pitch, yaw=0.0, dip=0.0, at=(0.0, 0.0, 0.0)):
    """The root's 4 x 4 matrix in its parent's frame for a bank, a pitch and a heading (radians) and a dip (m) at `at`:
    rotation_euler (-pitch, bank, yaw) in mode YXZ, delta_location (0, 0, dip)."""
    M = np.eye(4)
    M[:3, :3] = euler_yxz(-pitch, bank, yaw)
    M[:3, 3] = np.asarray(at, float) + [0.0, 0.0, dip]
    return M


def tips(O, t, yaw=0.0, at=(0.0, 0.0, 0.0)):
    """Where the wing tips are (n, 2, 3: left, right) in the root's parent frame at clip seconds t: the flutter, the
    bank, the pitch and the dip (what `<name>_tip_L` and `_R` ride)."""
    F = Fold(O["size"], O["sheet"], O["keel"])
    m = motion(O, t)
    out = np.empty((len(m["dip"]), 2, 3))
    for k, side in enumerate("LR"):
        o, yw = hinge(F, side)
        local = np.array([[math.cos(-yw), -math.sin(-yw), 0.0], [math.sin(-yw), math.cos(-yw), 0.0],
                          [0.0, 0.0, 1.0]]) @ (tip(F, side) - o)            # the tip in its wing's frame
        sgn = -1.0 if side == "L" else 1.0                                  # its rotation_euler[1]: + flexes R up
        for i in range(len(out)):
            W = euler_yxz(0.0, sgn * m[f"flex_{side}"][i], 0.0)
            Rw = np.array([[math.cos(yw), -math.sin(yw), 0.0], [math.sin(yw), math.cos(yw), 0.0], [0.0, 0.0, 1.0]])
            p = o + Rw @ (W @ local)
            M = root_matrix(m["bank"][i], m["pitch"][i], yaw, m["dip"][i], at)
            out[i, k] = M[:3, :3] @ p + M[:3, 3]
    return out


# ===================================================================================================================
# the card
# ===================================================================================================================
def objects(name):
    return {"body": f"{name}_body", "wing_L": f"{name}_wing_L", "wing_R": f"{name}_wing_R", "tip_L": f"{name}_tip_L",
            "tip_R": f"{name}_tip_R", "col_wing_L": f"{name}_col_wing_L", "col_wing_R": f"{name}_col_wing_R",
            "col_spine": f"{name}_col_spine"}


# the rider's colliders, for her hair and skirt in the sim: a wing's box is a band `band` x the length wide in from its
# leading edge (where her hair falls off the wing), the spine's `spine` x the length either side of the spine (her seat,
# the strip by the roots the bands leave toward the tail); `thick` m deep under the paper's top (deep, so a skirt pressed
# down under her as she sits is pushed back up, not out through the bottom of a thin box), edges rounded `rnd` m, their
# top faces' corners `margin` m inside the plane's outline in plan (a box past the paper's edge holds hair in the air)
COL = {"band": 0.2, "spine": 0.08, "thick": 0.1, "rnd": 0.008, "margin": 0.01}


def outline(F):
    """The plane's outline in plan (prop frame x, y, counter-clockwise): the nose, the left wing's tip and trailing
    corner, the tail either side of the spine, the right wing's."""
    left = np.array([F.wing(*F.P(k))[:2] for k in ("N", "TC", "TT", "GT")])
    return PT.ccw(np.vstack([left, left[:0:-1] * [-1.0, 1.0]]))


def _scan(lo, hi, fits, step=0.002):
    """The first value from `lo` toward `hi` (either way) where `fits` holds."""
    n = max(int(abs(hi - lo) / step), 1)
    for v in np.linspace(lo, hi, n + 1):
        if fits(float(v)):
            return float(v)
    raise ValueError(f"{NAME}: no collider fits the paper (sheet and keel too narrow)")


def colliders(F):
    """The rider's colliders: {"wing_L", "wing_R", "spine": (centre (3,), axes (3, 3) its x y z as columns, size (3,))}
    in the prop's frame, thin boxes whose top faces lie on the paper's top where the flap doubles it (her seat's
    height): each wing's a band whose outer side is its leading edge, from as near the tip as its corners stay `margin`
    inside the plane's outline forward to where its inner corner would cross the other wing's leading edge; the spine's
    from where the leading edges are wide enough to the tail."""
    poly = outline(F)
    m, w, half, th = COL["margin"], COL["band"] * F.L, COL["spine"] * F.L, COL["thick"]
    top = 2.0 * F.t + F.g
    tc = F.P("TC")
    u = tc / np.linalg.norm(tc)                                            # along the leading edge, in (a, b)
    n_in = np.array([-u[1], u[0]])                                         # square to it, toward the root

    def plan(s, d):
        a, b = s * u + d * n_in
        return F.wing(a, b)[:2]

    def band_fits(s):                                                      # the end corners at s, `margin` beyond them
        return bool(PT.inside(np.array([plan(s, m), plan(s, w + m)]), poly).all())
    s1 = _scan(float(np.linalg.norm(tc)), 0.0, band_fits) - m
    s0 = _scan(0.0, s1, band_fits) + m
    a, b = 0.5 * (s0 + s1) * u + 0.5 * w * n_in
    R = np.column_stack([-(n_in[0] * F.W3 + n_in[1] * F.G3), u[0] * F.W3 + u[1] * F.G3, [0.0, 0.0, 1.0]])
    wing = (F.wing(a, b, top - 0.5 * th), R, np.array([w, s1 - s0, th]))

    def spine_fits(y):
        return bool(PT.inside(np.array([[half + m, y], [-half - m, y]]), poly).all())
    y1 = _scan(float(poly[:, 1].max()), F.N3[1], spine_fits) - m
    y0 = _scan(F.N3[1], y1, spine_fits) + m
    spine = (np.array([0.0, 0.5 * (y0 + y1), top - 0.5 * th]), np.eye(3), np.array([2.0 * half, y1 - y0, th]))
    return {"wing_L": wing, "wing_R": (MIRROR @ wing[0], MIRROR @ wing[1] @ MIRROR, wing[2]), "spine": spine}


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, O, roles):
    """The prop's card: size, bounds, use points (sit and feet `seat`; anchor and look `tip_L`, `tip_R` riding the
    wing tips' empties; anchor `nose`; look `plane`), the rider's colliders (`colliders`: the wings' and the spine's
    boxes), the colour roles and `plane` (what it is and what is keyed)."""
    F = Fold(O["size"], O["sheet"], O["keel"])
    P = parts(O["size"], O["sheet"], O["keel"], O["doodles"], O["seed"])
    V = np.concatenate([m.V for m in P.values()])
    lo, hi = V.min(0), V.max(0)
    st = seat(F, O)
    ob = objects(name)
    tL, tR = tip(F, "L"), tip(F, "R")
    nose = F.N3 + np.array([0.0, 0.0, F.t + 0.5 * F.g])
    _d, o = F.le_dir()
    sit = {"name": "seat", "hip": _r(st["hip"]), "facing": _r(st["facing"]), "floor_z": round(float(st["floor_z"]), 5),
           "seat_z": round(float(st["seat_z"]), 5), "pelvis_deg": 8.0, "back_deg": 0.0}
    use = {"sit": [sit],
           "feet": [{"name": "seat", "L": _r(st["feet"]["L"]), "R": _r(st["feet"]["R"]),
                     "floor_z": round(float(st["floor_z"]), 5)}],
           "anchor": [{"name": "tip_L", "point": _r(tL), "dir": _r(o), "object": ob["tip_L"]},
                      {"name": "tip_R", "point": _r(tR), "dir": _r(MIRROR @ o), "object": ob["tip_R"]},
                      {"name": "nose", "point": _r(nose), "dir": [0.0, -1.0, 0.0]}],
           "look": [{"name": "plane", "point": [0.0, 0.0, 0.0]},
                    {"name": "tip_L", "point": _r(tL), "object": ob["tip_L"]},
                    {"name": "tip_R", "point": _r(tR), "object": ob["tip_R"]}]}
    b = {"min": _r(lo), "max": _r(hi)}
    return {"size": _r(hi - lo), "origin": "center", "front": "-Y", "blocks": False, "bounds": b, "layers": [],
            "colliders": [{"type": "box", "object": ob[f"col_{k}"], "rnd": COL["rnd"], "tag": name}
                          for k in ("wing_L", "wing_R", "spine")], "slots": dict(roles), "use": use,
            "plane": {"size": O["size"], "span": round(float(hi[0] - lo[0]), 5), "ply": round(F.t, 5),
                      "keel_depth": round(float(-lo[2]), 5), "seat": round(st["b"] / F.L, 4), "side": O["side"],
                      "step": O["step"], "step_from": O["step_from"], "land": O["land"], "objects": ob}}
