"""Camera clip ranges fitted to what a camera sees, bpy-free (docs/design.md: Shots: Clip ranges).

A render tells two surfaces on one line of sight apart only down to a depth step that grows as the square of their
distance over the camera's clip_start: from a 2 cm start, about 0.05 mm at 4 m. MMD models keep their drawn expressions
(star and heart eyes, a blush, tears) a few hundredths of a millimetre behind the skin until a morph brings them out,
so a camera that starts at 2 cm sees them all through the face from a couple of metres away. A shot camera starts
instead, on each frame, at NEAR_SHARE of the depth of the nearest geometry inside its view (`near`): as late as it can
without cutting anything off.

Points are in the camera's axes (x right, y up, looking down -z); a point's depth is -z. The view (`View`) is the
camera's pyramid of sight widened by MARGIN on each side, so what is just outside the frame counts too. `least_depth`
searches coarse to fine: each object's box, then the boxes of its chunks of about CHUNK nearby triangles (`Piece`),
then those triangles, nearest first, until no box left can hold anything nearer than what was found. Each step is
exact: the least depth of a convex piece inside the view is at a corner of the piece clipped to the view (`tri_depth`),
so nothing inside the view is ever nearer than where the camera starts."""
import numpy as np

NEAR_SHARE = 0.5          # the camera starts at this share of the depth of the nearest geometry it sees ...
NEAR_MIN = 0.001          # m: ... never nearer (geometry touching the lens)
NEAR_NONE = 1.0           # m: where it sees no geometry at all
MARGIN = 1.1              # the view is widened by this on each side
CHUNK = 256               # triangles in a chunk of an object
BATCH = 2048              # triangles clipped at a time
TOL = 1e-9                # m: a point this far outside a side of the view is on it
CORNERS = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)     # of a box, from lo (0) to hi (1)
BOX_TRIS = np.array([[0, 2, 6], [0, 6, 4], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4],   # its faces, two triangles each
                     [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6]])


class View:
    """The pyramid of sight of a perspective camera, widened by `margin`: the tangents (x / depth, y / depth) of its
    left, right, bottom and top sides as `xl`, `xr`, `yb`, `yt`; `planes` (4, 3) their inward normals (n . p >= 0
    inside: all four pass through the camera, and together they hold only points in front of it); `rays` (4, 3) its
    edges, from the camera along two sides each, with z = -1 (so a ray's parameter is the depth). `lens` and `sensor` in
    mm (the sensor across the side Blender fits it to: the longer one with fit "AUTO"), the picture `size` (w, h) in
    pixels, `shift` Blender's lens shift in shares of that side."""

    def __init__(self, lens, sensor, size, shift=(0.0, 0.0), fit="AUTO", margin=MARGIN):
        w, h = float(size[0]), float(size[1])
        side = max(w, h) if fit == "AUTO" else w if fit == "HORIZONTAL" else h
        k = float(sensor) / float(lens) / side                    # tangent per pixel
        cx, cy = float(shift[0]) * side, float(shift[1]) * side
        self.xl, self.xr = (cx - margin * w / 2) * k, (cx + margin * w / 2) * k
        self.yb, self.yt = (cy - margin * h / 2) * k, (cy + margin * h / 2) * k
        self.planes = np.array([[1.0, 0.0, self.xl], [-1.0, 0.0, -self.xr], [0.0, 1.0, self.yb],
                                [0.0, -1.0, -self.yt]])
        self.rays = np.array([[x, y, -1.0] for x in (self.xl, self.xr) for y in (self.yb, self.yt)])


def tri_depth(T, view):
    """The least depth inside `view` of each triangle of T (n, 3, 3) in the camera's axes, inf where none of it is
    inside: the least depth of the triangle clipped to the view is at a corner of the clipped polygon, which is a corner
    of the triangle inside the view, a point where an edge crosses a side inside the other sides, or a point where an
    edge of the view passes through the triangle (only triangles that cross a side are clipped)."""
    T = np.asarray(T, float).reshape(-1, 3, 3)
    if not len(T):
        return np.zeros(0)
    N = view.planes
    S = T @ N.T                                                   # (n, 3, 4): >= 0 inside each side
    inside = S >= -TOL
    corner_in = inside.all(-1)                                    # (n, 3)
    best = np.where(corner_in, -T[..., 2], np.inf).min(-1)
    cut = ~corner_in.all(-1) & inside.any(1).all(-1)              # crosses a side, and not wholly outside one
    if not cut.any():
        return best
    T, S = T[cut], S[cut]
    A, B, SA, SB = T, T[:, [1, 2, 0]], S, S[:, [1, 2, 0]]
    cross = SA * SB < 0                                           # (m, 3, 4): that edge crosses that side
    t = np.where(cross, SA / np.where(cross, SA - SB, 1.0), 0.0)
    P = A[:, :, None, :] + t[..., None] * (B - A)[:, :, None, :]  # (m, 3, 4, 3)
    ok = cross & ((P @ N.T) >= -TOL).all(-1)
    d = np.where(ok, -P[..., 2], np.inf).min((1, 2))
    e1, e2, tv = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], -T[:, 0]  # Moller-Trumbore from the camera, along each ray
    R = view.rays
    pv = np.cross(R[None, :, :], e2[:, None, :])                  # (m, 4, 3)
    det = (e1[:, None, :] * pv).sum(-1)                           # (m, 4)
    qv = np.cross(tv, e1)                                         # (m, 3)
    live = np.abs(det) > 1e-15
    inv = 1.0 / np.where(live, det, 1.0)
    u = (tv[:, None, :] * pv).sum(-1) * inv
    v = (R[None, :, :] * qv[:, None, :]).sum(-1) * inv
    s = (e2 * qv).sum(-1)[:, None] * inv
    hit = live & (u >= -TOL) & (v >= -TOL) & (u + v <= 1.0 + TOL) & (s > 0)
    best[cut] = np.minimum(best[cut], np.minimum(d, np.where(hit, s, np.inf).min(-1)))
    return best


def box_depth(lo, hi, M, view):
    """The least depth inside `view` of boxes [lo, hi] ((k, 3) each, in their own frames), M (k, 4, 4) or one (4, 4)
    for all: their frames to the camera's axes. 0 where the camera is inside a box (so it starts at the nearest), inf
    where a box is outside the view."""
    lo, hi = np.asarray(lo, float).reshape(-1, 3), np.asarray(hi, float).reshape(-1, 3)
    k = len(lo)
    if not k:
        return np.zeros(0)
    M = np.broadcast_to(np.asarray(M, float), (k, 4, 4))
    C = lo[:, None, :] + CORNERS[None] * (hi - lo)[:, None, :]   # (k, 8, 3) in their frames
    Cc = np.einsum("kij,kcj->kci", M[:, :3, :3], C) + M[:, None, :3, 3]
    d = tri_depth(Cc[:, BOX_TRIS].reshape(-1, 3, 3), view).reshape(k, 12).min(1)
    o = np.linalg.inv(M)[:, :3, 3]                               # the camera in each box's frame
    inside = ((o >= lo - TOL) & (o <= hi + TOL)).all(1)
    return np.where(inside, 0.0, d)


def chunks(centres, size=CHUNK):
    """Triangle indices grouped in chunks of at most `size` nearby ones: their `centres` (n, 3) split at the median of
    their longest extent until each part is small enough."""
    centres = np.asarray(centres, float).reshape(-1, 3)
    out, todo = [], [np.arange(len(centres))]
    while todo:
        idx = todo.pop()
        if len(idx) <= size:
            if len(idx):
                out.append(idx)
            continue
        c = centres[idx]
        ax = int(np.argmax(c.max(0) - c.min(0)))
        idx = idx[np.argsort(c[:, ax], kind="stable")]
        todo += [idx[:len(idx) // 2], idx[len(idx) // 2:]]
    return out


class Piece:
    """An object's triangles F (m, 3) (indices into its vertices) in chunks of nearby ones, made from the vertices it
    is first seen with; `boxes(V)` the chunks' boxes (lo, hi) for vertex positions V (n, 3) on any frame, so a deforming
    object keeps its chunks while it moves."""

    def __init__(self, V, F, size=CHUNK):
        V, self.F = np.asarray(V, float).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3)
        self.chunks = chunks(V[self.F].mean(1), size) if len(self.F) else []
        verts = [np.unique(self.F[c].ravel()) for c in self.chunks]
        self._at = np.concatenate(verts) if verts else np.zeros(0, np.int64)
        self._starts = np.cumsum([0] + [len(v) for v in verts[:-1]]).astype(np.int64)
        self._for = self._boxes = None

    def boxes(self, V):
        """The chunks' boxes (lo, hi) for vertex positions V (the last V's are kept: the cameras of one frame share them)."""
        if self._for is V:
            return self._boxes
        if not self.chunks:
            got = np.zeros((0, 3)), np.zeros((0, 3))
        else:
            P = np.asarray(V, float).reshape(-1, 3)[self._at]
            got = np.minimum.reduceat(P, self._starts, 0), np.maximum.reduceat(P, self._starts, 0)
        self._for, self._boxes = V, got
        return got


def corner_bounds(lo, hi, M, view):
    """Lower bounds of the depth inside `view` of boxes [lo, hi] ((k, 3), M (k, 4, 4) or one (4, 4) as in
    `box_depth`): inf for a box wholly outside one side, else the depth of its nearest corner (0 behind the camera). A
    box's least depth is at a corner, so nothing of it inside the view is nearer."""
    lo, hi = np.asarray(lo, float).reshape(-1, 3), np.asarray(hi, float).reshape(-1, 3)
    M = np.broadcast_to(np.asarray(M, float), (len(lo), 4, 4))
    C = lo[:, None, :] + CORNERS[None] * (hi - lo)[:, None, :]
    Cc = np.einsum("kij,kcj->kci", M[:, :3, :3], C) + M[:, None, :3, 3]
    out = ((Cc @ view.planes.T) < -TOL).all(1).any(-1)
    return np.where(out, np.inf, np.maximum(-Cc[..., 2].max(1), 0.0))


def least_depth(items, view):
    """The least depth (m) of the geometry of `items` inside `view`, inf when there is none. Each item is (box, M,
    piece): box = (lo, hi) the object's box in its own frame, M (4, 4) that frame to the camera's axes, piece() ->
    (Piece, V) its triangles and their vertices when they are wanted (a callable, so objects whose boxes are farther
    than what was found are never read), or None for an object that only has its box (taken as solid). Objects and
    their chunks are taken nearest corner first, the chunks' triangles BATCH at a time, until none left can be nearer."""
    if not items:
        return np.inf
    lo = np.array([b[0] for b, _, _ in items], float)
    hi = np.array([b[1] for b, _, _ in items], float)
    M = np.array([m for _, m, _ in items], float)
    b0 = corner_bounds(lo, hi, M, view)
    best = np.inf
    for i in np.argsort(b0, kind="stable"):
        if b0[i] >= best:
            break
        got = items[i][2]()
        if got is None:
            best = min(best, float(box_depth(lo[i], hi[i], M[i], view)[0]))
            continue
        piece, V = got
        clo, chi = piece.boxes(V)
        b1 = corner_bounds(clo, chi, M[i], view)
        order = [j for j in np.argsort(b1, kind="stable") if b1[j] < best]
        R, t = M[i, :3, :3], M[i, :3, 3]
        V = np.asarray(V, float).reshape(-1, 3)
        k = 0
        while k < len(order) and b1[order[k]] < best:
            take, n = [], 0
            while k < len(order) and b1[order[k]] < best and (not take or n < BATCH):
                take.append(piece.chunks[order[k]])
                n += len(take[-1])
                k += 1
            T = V[piece.F[np.concatenate(take)]] @ R.T + t
            best = min(best, float(tri_depth(T, view).min()))
    return best


def near(depth):
    """The clip_start of a camera whose nearest visible geometry is `depth` m away (inf: it sees none)."""
    if not np.isfinite(depth):
        return NEAR_NONE
    return max(NEAR_MIN, NEAR_SHARE * float(depth))
