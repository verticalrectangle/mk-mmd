"""The head skin mesh: a (theta, phi) grid of the implicit head shape with holes for the eyes and the mouth.

Grid: columns go around the vertical axis (theta, 0 = forward, +x positive), rows up the head (phi, elevation seen from
the skull centre); the lowest row is the neck ring (a horizontal ellipse, so it welds to the body's neck), the highest
is a pole. Columns halve in three bands towards the crown so the quads stay about square. Everything is HEAD-LOCAL
(see head_shape)."""
import numpy as np

from .head_shape import directions

# -------------------------------------------------------------------------------------------------- spacing helpers


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def spaced(f, n_steps, lo, hi, table=40001):
    """n_steps+1 values from lo to hi whose local spacing follows f(x) (spacing proportional to f)."""
    x = np.linspace(lo, hi, table)
    inv = 1.0 / np.maximum(f(x), 1e-9)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(x))])
    c /= c[-1]
    return np.interp(np.linspace(0.0, 1.0, n_steps + 1), c, x)


def column_angles(n, face_deg, back_deg, face_to=38.0, ramp=85.0):
    """n azimuths (even, symmetric, 0 and pi included), ascending 0..pi then -pi..: spacing face_deg in front growing
    to back_deg behind."""
    half = n // 2
    dens = lambda t: np.radians(face_deg) + (np.radians(back_deg) - np.radians(face_deg)) * \
        smooth((np.abs(t) - np.radians(face_to)) / np.radians(ramp))
    th = spaced(dens, half, 0.0, np.pi)
    return np.concatenate([th, -th[-2:0:-1]])


def row_elevations(shape, ring_dir0, phi_end, spacing):
    """Elevations (rad) of the grid rows above the neck ring, from the front profile: spacing(phi) metres apart along
    the profile, starting one step above the ring row (whose elevation is ring_dir0's)."""
    phis = np.radians(np.arange(-79.0, 90.0, 0.02))
    prof = shape.surface(directions(0.0 * phis, phis))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    s0 = float(np.interp(ring_dir0, phis, s))
    out = [ring_dir0]
    cur = s0
    while True:
        ph = float(np.interp(cur, s, phis))
        cur += float(spacing(ph))
        ph = float(np.interp(cur, s, phis))
        if ph >= phi_end:
            break
        out.append(ph)
    return np.array(out)


class Grid:
    """Vertices and faces of the closed head surface (no holes yet) plus the index maps the later steps use.

    `rows`: list of (vertex ids, column stride) from the neck ring up; stride 1 rows have all `n` columns (the finest
    layer, index `A`), stride 2 and 4 rows every 2nd / 4th. `phi`: elevation of every row (the ring row's is its
    value at theta = 0)."""

    UPPER = ((2, 66.0), (2, 72.5), (4, 80.0))             # (stride, elevation in degrees) rows above the finest layer

    def __init__(self, shape, cfg=None):
        c = dict(n_cols=96, face_deg=1.9, back_deg=8.0, ring_z=-0.045, phi_end_a=58.0,
                 spacing_face=0.0029, spacing_neck=0.0036, spacing_top=0.0062)
        c.update(cfg or {})
        self.cfg = c
        self.shape = shape
        self.neck = shape.p["neck"]
        self.theta = column_angles(c["n_cols"], c["face_deg"], c["back_deg"])
        N = self.n = len(self.theta)
        ring_dirs, ring_pts = self.ring(self.theta)
        phi_ring0 = float(np.arcsin(ring_dirs[0, 2]))
        spacing = lambda ph: (c["spacing_face"]
                              + (c["spacing_neck"] - c["spacing_face"]) * smooth((np.radians(-38.0) - ph) / np.radians(12.0))
                              + (c["spacing_top"] - c["spacing_face"]) * smooth((ph - np.radians(30.0)) / np.radians(26.0)))
        phi = list(row_elevations(shape, phi_ring0, np.radians(c["phi_end_a"]), spacing))
        self.nA = len(phi)
        V = []
        rows = []
        for j in range(self.nA):
            if j == 0:
                pts = ring_pts
            else:
                pts = shape.surface(directions(self.theta, np.full(N, phi[j])))
            rows.append((len(V) + np.arange(N), 1))
            V.extend(pts)
        for stride, deg in self.UPPER:
            cols = np.arange(0, N, stride)
            pts = shape.surface(directions(self.theta[cols], np.full(len(cols), np.radians(deg))))
            rows.append((len(V) + np.arange(len(cols)), stride))
            V.extend(pts)
            phi.append(np.radians(deg))
        self.pole = len(V)
        V.append(shape.surface(np.array([[0.0, 0.0, 1.0]]))[0])
        self.V = np.array(V)
        self.rows = rows
        self.phi = np.array(phi)
        self.A = np.array([r[0] for r in rows[:self.nA]])          # (nA, n) vertex ids of the finest layer

    def ring(self, theta):
        nk = self.neck
        a, b = nk["rx"], nk["ry"]
        rho = 1.0 / np.sqrt((np.sin(theta) / a) ** 2 + (np.cos(theta) / b) ** 2)
        p = np.stack([rho * np.sin(theta), nk["c"][1] - rho * np.cos(theta), np.full_like(theta, self.cfg["ring_z"])], -1)
        v = p - self.shape.centre
        return v / np.linalg.norm(v, axis=-1, keepdims=True), p

    def band_faces(self, lo, hi):
        """Faces between row `lo` and the row above it (indices into self.rows), CCW from outside."""
        (a, sa), (b, sb) = self.rows[lo], self.rows[hi]
        F = []
        if sa == sb:
            m = len(a)
            for i in range(m):
                i1 = (i + 1) % m
                F.append((a[i], a[i1], b[i1], b[i]))
        else:
            assert sb == 2 * sa
            m, Nl = len(b), len(a)
            for q in range(m):
                a0, a1, a2 = a[(2 * q) % Nl], a[(2 * q + 1) % Nl], a[(2 * q + 2) % Nl]
                u0, u1 = b[q], b[(q + 1) % m]
                F.append((a0, a1, u0))
                F.append((a1, a2, u1, u0))
        return F

    def fan_faces(self):
        top = self.rows[-1][0]
        m = len(top)
        return [(top[q], top[(q + 1) % m], self.pole) for q in range(m)]


# ------------------------------------------------------------------------------------------------- holes and rings


def lift(shape, x, z, y_hi=0.2, iters=40):
    """Points of the head surface seen from the front at (x, z): y of the first crossing walking back from y = -y_hi
    (vectorised; bisection, the shape is a graph over the front plane in the face region)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    lo = np.full(x.shape, -y_hi)
    hi = np.full(x.shape, float(shape.centre[1]))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = shape.phi(np.stack([x, mid, z], -1)) < 0.0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    return np.stack([x, 0.5 * (lo + hi), z], -1)


def rect_ring(ids, c0, c1, j0, j1):
    """Boundary vertex ids of the grid block of columns c0..c1 (may be negative: taken mod n) and rows j0..j1, CCW seen
    from the front: along the bottom row left to right, up the right column, along the top row right to left, down the
    left column (corners once)."""
    n = ids.shape[1]
    ring = [ids[j0, c % n] for c in range(c0, c1 + 1)]
    ring += [ids[j, c1 % n] for j in range(j0 + 1, j1 + 1)]
    ring += [ids[j1, c % n] for c in range(c1 - 1, c0 - 1, -1)]
    ring += [ids[j, c0 % n] for j in range(j1 - 1, j0, -1)]
    return np.array(ring)


class SkinBuilder:
    """Collects the vertices and faces of the skin: the plain grid minus the hole blocks, plus the ring zones."""

    SKIN, RIM = 0, 1                                  # face groups

    def __init__(self, shape, grid):
        self.shape = shape
        self.g = grid
        self.V = [v for v in grid.V]
        self.faces = []
        self.group = []
        self.cut = set()                              # removed (row, column) cells of the finest layer
        self.dead = set()                             # removed vertex ids
        self.holes = {}

    # ---- grid blocks
    def xz(self, ids):
        P = np.asarray([self.V[i] for i in np.atleast_1d(ids)])
        return P[:, 0], P[:, 2]

    def find_block(self, xmin, xmax, zmin, zmax, margins, cols):
        """Smallest block of finest-layer lines containing the box grown by margins (left, right, bottom, top) in the
        front plane. `cols`: candidate unwrapped column indices (ascending x)."""
        g = self.g
        ml, mr, mb, mt = margins
        cols = np.asarray(cols)
        zc = 0.5 * (zmin + zmax)
        xs_row = lambda j: np.array([self.V[g.A[j, c % g.n]][0] for c in cols])
        zs_col = lambda c: np.array([self.V[g.A[j, c % g.n]][2] for j in range(g.nA)])
        # mid row: the row whose z is closest to the box centre at the box's centre column
        xc = 0.5 * (xmin + xmax)
        c_mid = cols[np.argmin(np.abs(xs_row(g.nA // 2) - xc))]
        j_mid = int(np.argmin(np.abs(zs_col(c_mid) - zc)))
        xs = xs_row(j_mid)
        c0 = cols[np.max(np.where(xs <= xmin - ml)[0])]
        c1 = cols[np.min(np.where(xs >= xmax + mr)[0])]
        c_mid = (c0 + c1) // 2
        zs = zs_col(c_mid)
        j0 = int(np.max(np.where(zs <= zmin - mb)[0]))
        j1 = int(np.min(np.where(zs >= zmax + mt)[0]))
        return int(c0), int(c1), j0, j1

    def carve(self, c0, c1, j0, j1):
        n = self.g.n
        for j in range(j0, j1):
            for c in range(c0, c1):
                self.cut.add((j, c % n))
        for j in range(j0 + 1, j1):
            for c in range(c0 + 1, c1):
                self.dead.add(int(self.g.A[j, c % n]))
        return rect_ring(self.g.A, c0, c1, j0, j1)

    def build_grid_faces(self):
        g = self.g
        N = g.n
        A = g.A
        for j in range(g.nA - 1):
            for i in range(N):
                if (j, i) in self.cut:
                    continue
                i1 = (i + 1) % N
                self.add_face((A[j, i], A[j, i1], A[j + 1, i1], A[j + 1, i]), self.SKIN)
        for r in range(g.nA - 1, len(g.rows) - 1):
            for f in g.band_faces(r, r + 1):
                self.add_face(f, self.SKIN)
        for f in g.fan_faces():
            self.add_face(f, self.SKIN)

    def add_face(self, f, group):
        self.faces.append(tuple(int(i) for i in f))
        self.group.append(group)

    def add_verts(self, P):
        base = len(self.V)
        self.V.extend(np.asarray(P, float))
        return base + np.arange(len(P))

    def strip(self, a, b, group, cyclic=True):
        """Quads between two equally long loops a and b (CCW seen from the front, b inside a)."""
        n = len(a)
        for m in range(n if cyclic else n - 1):
            m1 = (m + 1) % n
            self.add_face((a[m], a[m1], b[m1], b[m]), group)

    def finish(self):
        """Compact the vertex list (drop the removed and unused vertices); returns V, faces, groups, remap."""
        used = np.zeros(len(self.V), bool)
        for f in self.faces:
            used[list(f)] = True
        keep = np.where(used)[0]
        remap = -np.ones(len(self.V), int)
        remap[keep] = np.arange(len(keep))
        V = np.asarray(self.V)[keep]
        F = [tuple(int(remap[i]) for i in f) for f in self.faces]
        return V, F, np.array(self.group, int), remap
