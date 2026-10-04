"""Geometry of the desk's drawer pedestal (pure numpy through the shell toolkit, no bpy: importable from tests). The
pedestal of `bedroom_desk` is a moulded unit, not a box:

    carcass   an open tube (hollow, closed at the top by the slab) whose plan is a rounded rectangle with 70 mm (front) and
              100 mm (back) corner radii (so its side walls are two flat runs between big curved corners), rolled over at
              the foot into a recessed kick plate. Its surface is `offset_rings` of the plan outline at the heights of a
              profile, skinned into a cage.
    fronts    three drawer fronts that wrap round the corners: each is a thin rounded shell, built flat in (s, v, w) with s =
              arc length along the plan outline from the middle of the front, v = height, w = thickness, then bent onto the
              outline (position = outline(s) + normal(s) * (w + gap)) and crowned by 3 mm up its height.
    joints    dark reveal strips on the carcass in the 4 mm gaps between the fronts, bent the same way.

Frame of the desk: x to the right, y toward the wall, z up; the pedestal stands at the -X end under the slab. Everything here
is in that frame (metres)."""
import math

import numpy as np

from ....core import shell as CS

# ---------------------------------------------------------------- measures
PED_X = (-0.592, -0.28)            # outer extents with the drawer fronts on (the collider's box)
PED_Y = (-0.28, 0.285)
FRONT_T = 0.010                    # thickness of a drawer front = how far it stands proud of the carcass
GAP = 0.0004                       # clearance between the fronts and the carcass
CROWN = 0.003                      # crown of a front up its height (m at its middle row, on the flat of the front only)
CARC_X = (PED_X[0] + FRONT_T + GAP, PED_X[1] - FRONT_T - GAP)      # the carcass inside the fronts
CARC_Y = (PED_Y[0] + FRONT_T + GAP + CROWN, PED_Y[1])
PLAN_R = (0.07, 0.10)              # plan corner radii of the carcass: front corners, back corners
Z_TOP = 0.7099                     # top of the carcass (just under the slab)
ROLL_R = 0.014                     # the foot rolls inward over this radius (convex) ...
COVE_R = 0.008                     # ... and meets the recessed kick plate in a cove of this radius
KICK_Z = 0.077                     # height of the underside of the roll (the kick plate's overhang)
WALL_Z = KICK_Z + ROLL_R           # where the vertical wall begins
WRAP = 0.035                       # how far a front runs on along the side walls past the corner arcs
SEAT = 0.0008                      # a reveal strip stands this far off the carcass


def drawer_rows(top=0.695, heights=(0.165, 0.19, 0.22), joint=0.004):
    """(z bottom, z top) of the drawer fronts, top to bottom, with 4 mm joints."""
    rows, z = [], top
    for h in heights:
        rows.append((z - h, z))
        z -= h + joint
    return rows


ROWS = drawer_rows()
JOINTS = [(ROWS[i + 1][1], ROWS[i][0]) for i in range(len(ROWS) - 1)]       # (z bottom, z top) of the gaps


# ---------------------------------------------------------------- the plan outline as a path
def outline(n=16, r=PLAN_R):
    """Closed counter-clockwise plan outline (m, 2) of the carcass, the first point in the middle of the front. The front
    corners (radius r[0]) stay within half of the front, so their arcs are exact."""
    (x0, x1), (y0, y1) = CARC_X, CARC_Y
    assert r[0] <= (x1 - x0) / 4 + 1e-9, "front corner radius: the arc may use at most half of half the front"
    pts = [((x0 + x1) / 2, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return CS.fillet(pts, [0.0, r[0], r[1], r[1], r[0]], n=n, closed=True)


class Path:
    """A closed counter-clockwise polygon (m, 2) as a path by arc length s from its first point: `at(s)` gives the position
    and the unit outward normal (smoothly interpolated between the vertex normals), s wrapping round."""

    def __init__(self, P):
        self.P = P = np.asarray(P, float)
        E = np.roll(P, -1, axis=0) - P
        self.seg = np.linalg.norm(E, axis=1)
        T = E / self.seg[:, None]
        self.start = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.total = float(self.start[-1])
        nrm = np.stack([T[:, 1], -T[:, 0]], 1)
        vn = np.roll(nrm, 1, axis=0) + nrm
        self.N = vn / np.linalg.norm(vn, axis=1, keepdims=True)

    def at(self, s):
        s = np.mod(np.asarray(s, float), self.total)
        m = len(self.P)
        i = np.clip(np.searchsorted(self.start, s, side="right") - 1, 0, m - 1)
        t = ((s - self.start[i]) / self.seg[i])[:, None]
        j = (i + 1) % m
        pos = self.P[i] * (1.0 - t) + self.P[j] * t
        n = self.N[i] * (1.0 - t) + self.N[j] * t
        return pos, n / np.linalg.norm(n, axis=1, keepdims=True)

    def arc_to(self, point_fn):
        """Arc length from the first point to the first vertex (going counter-clockwise) where point_fn(P) is True."""
        hit = np.flatnonzero(point_fn(self.P))
        return float(self.start[hit[0]])


def half_wrap(path, extra=WRAP):
    """Half length of a drawer front along the outline: the straight front, the corner arc and `extra` on the side."""
    x1 = CARC_X[1]
    s_arc_end = path.arc_to(lambda P: P[:, 0] > x1 - 1e-6)            # first vertex on the right wall = end of the arc
    return s_arc_end + extra


# ---------------------------------------------------------------- the carcass
def carcass(mat_wall=0, mat_kick=1):
    """The carcass tube: open at the foot and the top, kick plate and roll in role `mat_kick`, walls in `mat_wall`."""
    P = outline()
    a = np.linspace(0.0, 0.5 * math.pi, 5)[1:]
    cove = [(ROLL_R + COVE_R * math.sin(t), KICK_Z - COVE_R + COVE_R * math.cos(t)) for t in a[::-1]]
    roll = [(ROLL_R * (1.0 - math.cos(t)), WALL_Z - ROLL_R * math.sin(t)) for t in list(a[::-1][1:]) + [0.0]]
    # from the floor up: the kick wall, the cove, the underside of the overhang, the convex roll, the wall
    profile = [(ROLL_R + COVE_R, 0.0)] + cove + [(ROLL_R, KICK_Z)] + roll + [(0.0, Z_TOP)]
    rings = CS.offset_rings(P, profile)
    m = CS.cage(rings, closed=True, caps=(None, None), mat=mat_wall)
    m.assign(mat_kick, lambda c, n: c[:, 2] < WALL_Z - 1e-6)
    return m


# ---------------------------------------------------------------- fronts and joints
def _bend(mesh, path, zc, gap, crown=0.0, h=1.0, t=1.0):
    """Bend a flat part built in (s, v, w) = (x, y, z) onto the outline: position = outline(s) + normal(s) * (gap + w) at
    height zc + v. The crown lifts the outer face by up to `crown` at the middle row (v = 0), nothing at the top and bottom
    edges, and fades out round the corners (it is there only where the surface faces the front, never on the sides)."""
    V = mesh.V
    s, v, w = V[:, 0], V[:, 1], V[:, 2]
    pos, n = path.at(s)
    if crown:
        f = np.clip((-n[:, 1] - 0.35) / 0.65, 0.0, 1.0)
        w = w + crown * (f * f * (3.0 - 2.0 * f)) * np.clip(1.0 - (2.0 * v / h) ** 2, 0.0, 1.0) * np.clip(w / t, 0.0, 1.0)
    out = np.column_stack([pos + n * (gap + w)[:, None], zc + v])
    m = mesh.copy()
    m.V = out
    return m


def front(path, z0, z1, half, mat=0, div=0.014):
    """One drawer front between heights z0 and z1, running `half` either side of the middle of the front."""
    h, t = z1 - z0, FRONT_T
    box = CS.rounded_box((2.0 * half, h, t), center=(0.0, 0.0, t / 2), r=0.0035, k=2, div=div, mat=mat)
    return _bend(box, path, (z0 + z1) / 2, GAP, CROWN, h, t)


def joint(path, z0, z1, half, mat=0, div=0.012):
    """The dark reveal strip of the gap between two fronts (z0..z1), lying on the carcass inside the gap."""
    h, t = (z1 - z0) + 0.003, 0.0015
    box = CS.rounded_box((2.0 * half - 0.02, h, t), center=(0.0, 0.0, t / 2), r=0.0006, k=1, div=div, mat=mat)
    return _bend(box, path, (z0 + z1) / 2, SEAT, 0.0, h, t)


def fronts(mats=(0, 1, 2)):
    """The three drawer fronts merged into one mesh, front i in role mats[i]."""
    path = Path(outline())
    half = half_wrap(path)
    return CS.merge([front(path, z0, z1, half, mat=mats[i]) for i, (z0, z1) in enumerate(ROWS)])


def joints(mat=0):
    path = Path(outline())
    half = half_wrap(path)
    return CS.merge([joint(path, z0, z1, half, mat=mat) for z0, z1 in JOINTS])


def pull_centre():
    """x of the middle of the front and y of the outer face of a front at its middle row (the crown included): PED_Y[0]."""
    return (CARC_X[0] + CARC_X[1]) / 2, CARC_Y[0] - GAP - FRONT_T - CROWN
