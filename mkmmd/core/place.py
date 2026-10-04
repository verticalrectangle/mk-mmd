"""Placement maths (docs/design.md: Placement): where a prop's footprint lands on a surface. numpy only, so the props stage
(Blender's Python) and the tests (no Blender) share one implementation.

Vocabulary.
  Surface    a flat face props can stand or hang on, with a right-handed frame (u, v, n) in the world: a horizontal
             `floor` (a rest plane of a prop or a set's floor; n up, rectangle or disc), a vertical `wall` (n points
             into the room, u to the right as seen from the room, v up) or a 1-D `edge` (u along a segment). Surface
             coordinates (u, v) are metres from the surface's centre.
  Footprint  a prop's local bounds (lo, hi), its `front` axis and flags. On a floor its plan rectangle (x by y) is what
             lands on the surface and the bounds' lowest point rests on it; on a wall its width and height (x by z)
             land on the surface, the prop faces out of the wall with its back (bounds' +Y side) on the wall plane (a
             prop whose card origin is a wall origin has its ORIGIN on the plane instead).
  Prism      what an already placed prop occupies: an oriented plan rectangle and a height interval, one per height
             LAYER of the prop (a chair is wide at the seat and only a backrest above it, a lamp a small base under a
             long shade), so a mug can stand under a lamp's shade and a lamp on a desk clears the chair tucked under it.
             A new prop is blocked by another when a layer's plan rectangle comes closer than `clear` while their
             height intervals overlap (resting on top of something is not overlapping); a `flat` prop (a rug) never
             blocks, and an `avoid` entry blocks whatever the heights.
  Rule       what the project asked for: `at` (an explicit (u, v), "center", Near(...) or Region(...)), `yaw` or
             `facing`, `align` to surface edges, `clear` (m), `avoid`, `seed`.

Everything is deterministic: candidates come in a fixed order and random choices use numpy's default_rng(seed).
`PlaceError` says what could not be satisfied, in metres and millimetres, and which prop is in the way.

Yaw is a turn about +Z (radians internally, degrees at the rule level) of the prop's own frame: yaw 0 leaves its local
-Y axis (the usual front) pointing along the surface's -v direction (-Y of the owner), +90 deg turns it toward +u. Yaws
in a rule are relative to the surface's frame, so a prop placed on a rotated desk follows it."""
import math
from dataclasses import dataclass, field

import numpy as np

FRONTS = {"-Y": (0.0, -1.0), "+Y": (0.0, 1.0), "-X": (-1.0, 0.0), "+X": (1.0, 0.0)}
SIDES = {"front": (0.0, -1.0), "back": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
         "bottom": (0.0, -1.0), "top": (0.0, 1.0)}        # outward in (u, v); bottom / top are front / back on a wall
WALL_NAMES = {"front": "bottom", "back": "top"}
HORIZONTAL = 0.7              # |n_z| at or above this: a floor-like surface; below: a wall (which must be vertical)
FLAT_EPS = 2.0e-3             # height overlap (m) that still counts as one thing resting on the other
TOL = 1.0e-9
RING_STEP = 15.0              # degrees between the bearings `near` tries


class PlaceError(ValueError):
    """The rule cannot be satisfied (the build stage reports it as a BuildError)."""


# ------------------------------------------------------------------------------------------------------ rectangles
def wrap(a):
    """An angle (rad) in (-pi, pi]."""
    return math.pi - (math.pi - a) % (2.0 * math.pi)


def heading(yaw):
    """Plan direction a local -Y axis points along after turning by `yaw` (rad)."""
    return np.array([math.sin(yaw), -math.cos(yaw)])


def yaw_for(direction, front="-Y"):
    """Yaw (rad) that turns the prop's `front` axis toward the plan direction `direction` (x, y)."""
    if front not in FRONTS:
        raise PlaceError(f"front {front!r} is not one of {sorted(FRONTS)}")
    fx, fy = FRONTS[front]
    return wrap(math.atan2(direction[1], direction[0]) - math.atan2(fy, fx))


@dataclass(frozen=True)
class Rect:
    """An oriented rectangle in a plane: centre, half sizes along its own x and y axes, turned by `yaw` (rad)."""
    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float = 0.0

    def axes(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c, s]), np.array([-s, c])

    def corners(self):
        ax, ay = self.axes()
        c = np.array([self.cx, self.cy])
        return np.array([c - ax * self.hx - ay * self.hy, c + ax * self.hx - ay * self.hy,
                         c + ax * self.hx + ay * self.hy, c - ax * self.hx + ay * self.hy])

    @property
    def centre(self):
        return np.array([self.cx, self.cy])


def _point_segment(p, a, b):
    """Distance from the points p (n, 2) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def gap(a, b):
    """Signed distance between two rectangles: the Euclidean gap when they are apart (0 touching), minus the depth of
    the overlap (the shortest move that separates them) when they intersect."""
    ca, cb = a.corners(), b.corners()
    depth = math.inf
    for ax in (*a.axes(), *b.axes()):
        pa, pb = ca @ ax, cb @ ax
        depth = min(depth, float(min(pa.max() - pb.min(), pb.max() - pa.min())))
    if depth > 0.0:
        return -float(depth)
    best = math.inf
    for p, q in ((ca, cb), (cb, ca)):
        for i in range(4):
            best = min(best, float(_point_segment(p, q[i], q[(i + 1) % 4]).min()))
    return best


def gap_to_point(rect, p):
    """Signed distance from the rectangle to the point p (negative inside: minus the depth to the nearest side)."""
    ax, ay = rect.axes()
    d = np.asarray(p, float) - rect.centre
    dx, dy = abs(float(d @ ax)) - rect.hx, abs(float(d @ ay)) - rect.hy
    if dx <= 0.0 and dy <= 0.0:
        return max(dx, dy)
    return math.hypot(max(dx, 0.0), max(dy, 0.0))


@dataclass(frozen=True)
class Prism:
    """What a placed prop occupies: plan rectangle and height interval (z0, z1) in the world."""
    name: str
    rect: Rect
    z0: float
    z1: float
    flat: bool = False


def conflict(new, other, clear, forced=False):
    """The gap (m) to `other` when it blocks `new` (closer than `clear` and, unless `forced`, sharing heights), else
    None. A flat prop (on either side) never blocks unless forced; heights that only touch (one rests on the other) do
    not count."""
    g = gap(new.rect, other.rect)
    if g >= clear - TOL:
        return None
    if not forced:
        if other.flat or new.flat:
            return None
        if min(new.z1, other.z1) - max(new.z0, other.z0) <= FLAT_EPS:
            return None
    return g


def prism_of(name, lo, hi, world, flat=False):
    """The prism of a prop with local bounds lo..hi and world matrix `world` (4x4). A turn about Z is exact; a tilted
    prop gets the axis-aligned hull of its bounds."""
    M = np.asarray(world, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    R, t = M[:3, :3], M[:3, 3]
    sx, sy = float(np.linalg.norm(R[:, 0])), float(np.linalg.norm(R[:, 1]))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = corners @ R.T + t
    upright = abs(R[2, 0]) + abs(R[2, 1]) + abs(R[0, 2]) + abs(R[1, 2]) < 1e-6 * max(sx, 1.0)
    if upright:
        yaw = math.atan2(R[1, 0], R[0, 0])
        c = t[:2] + R[:2, :2] @ ((lo[:2] + hi[:2]) / 2.0)
        rect = Rect(float(c[0]), float(c[1]), float((hi[0] - lo[0]) / 2 * sx), float((hi[1] - lo[1]) / 2 * sy), yaw)
    else:
        a, b = w[:, :2].min(0), w[:, :2].max(0)
        rect = Rect(float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2), float((b[0] - a[0]) / 2),
                    float((b[1] - a[1]) / 2), 0.0)
    return Prism(name, rect, float(w[:, 2].min()), float(w[:, 2].max()), flat)


def prisms_of(name, lo, hi, world, flat=False, layers=()):
    """The prisms of a placed prop with local bounds lo..hi, height `layers` ((z0, z1, x0, y0, x1, y1) in its frame) and
    world matrix `world`: one per layer when it stands upright, else the single prism of its bounds."""
    M = np.asarray(world, float)
    R, t = M[:3, :3], M[:3, 3]
    upright = abs(R[2, 0]) + abs(R[2, 1]) + abs(R[0, 2]) + abs(R[1, 2]) < 1e-6 * max(float(np.linalg.norm(R[:, 0])), 1.0)
    if not layers or not upright:
        return [prism_of(name, lo, hi, M, flat)]
    yaw = math.atan2(R[1, 0], R[0, 0])
    sx, sy, sz = (float(np.linalg.norm(R[:, i])) for i in range(3))
    out = []
    for z0, z1, x0, y0, x1, y1 in layers:
        c = t[:2] + R[:2, :2] @ np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        out.append(Prism(name, Rect(float(c[0]), float(c[1]), float((x1 - x0) / 2 * sx), float((y1 - y0) / 2 * sy), yaw),
                         float(t[2] + z0 * sz), float(t[2] + z1 * sz), flat))
    return out


def layers_of_points(points, step=0.05, max_layers=10, merge=0.005):
    """Height layers of a point cloud (n, 3) in a prop's frame: [(z0, z1, x0, y0, x1, y1), ...]. The z range is cut into
    slices of about `step` metres (at most `max_layers`) and every slice gets the plan bounding box of its points; empty
    slices are skipped and neighbours whose boxes differ by less than `merge` are joined. Sample the surfaces densely
    (mkmmd.core.propcard.sample_points) so a post that spans several slices is in each of them."""
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(P):
        return []
    z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
    n = int(np.clip(math.ceil((z1 - z0) / step), 1, max_layers))
    edges = np.linspace(z0, z1, n + 1)
    idx = np.clip(np.searchsorted(edges, P[:, 2], side="right") - 1, 0, n - 1)
    out = []
    for i in range(n):
        q = P[idx == i]
        if not len(q):
            continue
        box = (*q[:, :2].min(0), *q[:, :2].max(0))
        if out and all(abs(a - b) < merge for a, b in zip(out[-1][2:], box)):
            lo_ = tuple(min(a, b) for a, b in zip(out[-1][2:4], box[:2]))
            hi_ = tuple(max(a, b) for a, b in zip(out[-1][4:6], box[2:]))
            out[-1] = (out[-1][0], float(edges[i + 1]), *lo_, *hi_)
        else:
            out.append((float(edges[i]), float(edges[i + 1]), *(float(v) for v in box)))
    out[-1] = (*out[-1][:1], z1, *out[-1][2:])
    return out


# ------------------------------------------------------------------------------------------------------ surfaces
@dataclass(frozen=True)
class Surface:
    """A face props land on: world `origin` (its centre), orthonormal right-handed frame (u, v, n) as 3-vectors,
    `half` = (hu, hv) half extents ((R, 0) for a disc, (half length, 0) for an edge)."""
    name: str
    kind: str                     # "floor" | "wall" | "edge"
    origin: tuple
    u: tuple
    v: tuple
    n: tuple
    half: tuple
    shape: str = "rect"           # "rect" | "disc"

    @property
    def O(self):
        return np.asarray(self.origin, float)

    @property
    def U(self):
        return np.asarray(self.u, float)

    @property
    def V(self):
        return np.asarray(self.v, float)

    @property
    def N(self):
        return np.asarray(self.n, float)

    @property
    def angle(self):
        """Turn of the surface's u axis about Z (rad)."""
        return math.atan2(self.u[1], self.u[0])


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise PlaceError("a direction of zero length")
    return v / n


def surface_from_entry(name, entry, world, bounds=None):
    """A Surface from a card entry and its owner's world matrix (4x4, owner -> world): a `use.rest` plane
    ({center, normal, size [w, d] | radius, u?}) or edge ({type: edge, a, b, normal}), or a `use.surface` panel
    ({center, normal, up, size [w, h]}). `bounds` = (lo, hi) of the owner, used for a plane without size."""
    M = np.asarray(world, float)
    R, t = M[:3, :3], M[:3, 3]
    R = R / np.maximum(np.linalg.norm(R, axis=0), 1e-12)

    def pt(p):
        return M[:3, :3] @ np.asarray(p, float) + t

    if entry.get("type") == "edge":
        a, b = pt(entry["a"]), pt(entry["b"])
        if float(np.linalg.norm(b - a)) < 1e-9:
            raise PlaceError(f"{name}: an edge of zero length")
        n = _unit(R @ np.asarray(entry.get("normal", (0, 0, 1)), float))
        u = _unit(b - a)
        v = _unit(np.cross(n, u))
        return Surface(name, "edge", tuple((a + b) / 2), tuple(u), tuple(v), tuple(n),
                       (float(np.linalg.norm(b - a)) / 2.0, 0.0))
    n = _unit(R @ np.asarray(entry.get("normal", (0, 0, 1)), float))
    center = pt(entry.get("center", (0.0, 0.0, 0.0)))
    if abs(n[2]) >= HORIZONTAL:
        if n[2] < 0:
            raise PlaceError(f"{name}: faces down (a ceiling): props stand on floors and hang on walls")
        if n[2] < 0.999:
            raise PlaceError(f"{name}: tilted by {math.degrees(math.acos(min(n[2], 1.0))):.0f} deg: placement "
                             f"needs a horizontal plane or a vertical wall")
        u = R @ np.asarray(entry.get("u", (1.0, 0.0, 0.0)), float)
        u = _unit(u - (u @ n) * n)
        v = np.cross(n, u)
        if entry.get("radius") is not None:
            r = float(entry["radius"])
            return Surface(name, "floor", tuple(center), tuple(u), tuple(v), tuple(n), (r, 0.0), "disc")
        if entry.get("size") is not None:
            w, d = (float(x) for x in entry["size"][:2])
            return Surface(name, "floor", tuple(center), tuple(u), tuple(v), tuple(n), (w / 2.0, d / 2.0))
        if bounds is None:
            raise PlaceError(f"{name}: the plane has neither a `size` nor a `radius` (give one in the card or "
                             f"card_extra)")
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        c = pt((lo + hi) / 2.0)
        c[2] = center[2]
        return Surface(name, "floor", tuple(c), tuple(u), tuple(v), tuple(n),
                       (float(hi[0] - lo[0]) / 2.0, float(hi[1] - lo[1]) / 2.0))
    if abs(n[2]) > 0.05:
        raise PlaceError(f"{name}: tilted by {math.degrees(math.asin(abs(n[2]))):.0f} deg from vertical: placement "
                         f"needs a horizontal plane or a vertical wall")
    n = _unit(np.array([n[0], n[1], 0.0]))
    v = np.array([0.0, 0.0, 1.0])
    u = np.cross(v, n)
    if entry.get("size") is None:
        raise PlaceError(f"{name}: a wall needs a `size` [w, h]")
    w, h = (float(x) for x in entry["size"][:2])
    return Surface(name, "wall", tuple(center), tuple(u), tuple(v), tuple(n), (w / 2.0, h / 2.0))


# ------------------------------------------------------------------------------------------------------ footprints
@dataclass(frozen=True)
class Footprint:
    """A prop's local bounds and how it stands: `front` (the axis a person faces), `flat` (others may overlap it: a rug),
    `wall_origin` (its origin lies on the wall plane when hung)."""
    name: str
    lo: tuple
    hi: tuple
    front: str = "-Y"
    flat: bool = False
    wall_origin: bool = False
    layers: tuple = ()            # height layers (z0, z1, x0, y0, x1, y1); none = the whole bounds

    @property
    def size(self):
        return tuple(float(h - l) for l, h in zip(self.lo, self.hi))

    @property
    def centre(self):
        return tuple(float((h + l) / 2.0) for l, h in zip(self.lo, self.hi))

    @property
    def height(self):
        return float(self.hi[2] - self.lo[2])


def bounds_from_size(size, origin="floor_center"):
    """(lo, hi) a card's `size` implies when nothing was measured: `floor_center` (centre of the footprint, bottom at
    z = 0; the default), `center` (centred on the origin) or `wall_*` (hung on a wall: centred across, the body in front
    of the wall plane y = 0, i.e. toward -Y)."""
    sx, sy, sz = (float(v) for v in size)
    if str(origin).startswith("wall"):
        return (-sx / 2, -sy, -sz / 2), (sx / 2, 0.0, sz / 2)
    if origin == "center":
        return (-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2)
    return (-sx / 2, -sy / 2, 0.0), (sx / 2, sy / 2, sz)


@dataclass
class Pose:
    """A footprint put on a surface: the prop's origin and yaw in the world, its prism and its corners in (u, v)."""
    origin: np.ndarray
    yaw: float
    prism: Prism
    corners: np.ndarray           # (4, 2) footprint corners in the surface's (u, v) plane
    uv: tuple                     # the footprint's centre in (u, v)
    layers: list = field(default_factory=list)       # the prism of each height layer (what blocks others)


def _layer_prisms(fp, origin, yaw, R2):
    """World prisms of the footprint's height layers for a prop at `origin` turned by `yaw` (R2 its 2x2 rotation)."""
    lo, hi = fp.lo, fp.hi
    layers = fp.layers or ((lo[2], hi[2], lo[0], lo[1], hi[0], hi[1]),)
    out = []
    for z0, z1, x0, y0, x1, y1 in layers:
        c = origin[:2] + R2 @ np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        out.append(Prism(fp.name, Rect(float(c[0]), float(c[1]), float((x1 - x0) / 2.0), float((y1 - y0) / 2.0), yaw),
                         float(origin[2] + z0), float(origin[2] + z1), fp.flat))
    return out


def pose_on(surface, fp, cu, cv, yaw):
    """The pose of `fp` with its footprint centred at (cu, cv) on `surface`, turned by the world yaw `yaw` (rad); on a
    wall the yaw is ignored and the prop faces out of it."""
    lo, hi = np.asarray(fp.lo, float), np.asarray(fp.hi, float)
    cl = (lo[:2] + hi[:2]) / 2.0
    half = (hi[:2] - lo[:2]) / 2.0
    O, U, V, N = surface.O, surface.U, surface.V, surface.N
    if surface.kind == "wall":
        yaw = yaw_for(N[:2], fp.front)
    c, s = math.cos(yaw), math.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])
    if surface.kind != "wall":
        C = O + U * cu + V * cv
        origin = np.array([*(C[:2] - R2 @ cl), O[2] - lo[2]])
        rect = Rect(float(C[0]), float(C[1]), float(half[0]), float(half[1]), yaw)
        corners = np.array([[(p - O[:2]) @ U[:2], (p - O[:2]) @ V[:2]] for p in rect.corners()])
        return Pose(origin, yaw, Prism(fp.name, rect, float(O[2]), float(O[2] + hi[2] - lo[2]), fp.flat), corners,
                    (float(cu), float(cv)), _layer_prisms(fp, origin, yaw, R2))
    # a wall: put the origin on the plane at the surface centre, measure the box, then slide it to (cu, cv)
    box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w0 = np.concatenate([box[:, :2] @ R2.T, box[:, 2:]], axis=1) + O
    pu, pv, pn = (w0 - O) @ U, (w0 - O) @ V, (w0 - O) @ N
    mu, mv = (pu.min() + pu.max()) / 2.0, (pv.min() + pv.max()) / 2.0
    shift = (cu - mu) * U + (cv - mv) * V + (0.0 if fp.wall_origin else -float(pn.min())) * N
    origin = O + shift
    C = origin[:2] + R2 @ cl
    rect = Rect(float(C[0]), float(C[1]), float(half[0]), float(half[1]), yaw)
    hu, hv = (pu.max() - pu.min()) / 2.0, (pv.max() - pv.min()) / 2.0
    corners = np.array([[cu - hu, cv - hv], [cu + hu, cv - hv], [cu + hu, cv + hv], [cu - hu, cv + hv]])
    return Pose(origin, yaw, Prism(fp.name, rect, float(origin[2] + lo[2]), float(origin[2] + hi[2]), fp.flat),
                corners, (float(cu), float(cv)), _layer_prisms(fp, origin, yaw, R2))


def _side_margins(surface, pose):
    """How far inside each side of the surface the footprint is (m; negative = sticking out): front, back, left,
    right for a rectangle, `round` for a disc."""
    c = pose.corners
    hu, hv = surface.half
    if surface.shape == "disc":
        return {"round": float(hu - np.linalg.norm(c, axis=1).max())}
    if surface.kind == "edge":
        out = {"along": float(hu - abs(pose.uv[0]))}
        if abs(pose.uv[1]) > 1e-9:
            out["across"] = float(-abs(pose.uv[1]))
        return out
    return {"front": float(c[:, 1].min() + hv), "back": float(hv - c[:, 1].max()),
            "left": float(c[:, 0].min() + hu), "right": float(hu - c[:, 0].max())}


def edge_margin(surface, pose):
    """The smallest margin between the footprint and the surface's boundary (m; negative = sticking out)."""
    return min(_side_margins(surface, pose).values())


# ------------------------------------------------------------------------------------------------------ rules
@dataclass(frozen=True)
class Near:
    """Put the prop `distance` (m between footprints, negative = overlapping that much) from `target`: a Rect (a prop's
    plan footprint, with its yaw as `target_yaw`) or a point (x, y). `bearing` (deg) is where around it, measured from
    the target's front (0 = in front of it, +90 toward its left (+X for a prop facing -Y))."""
    target: object
    distance: float = 0.0
    bearing: float = 0.0
    target_name: str = ""
    target_yaw: float = 0.0


@dataclass(frozen=True)
class Region:
    """Anywhere with the footprint's centre inside the (u, v) rectangle lo..hi (seeded)."""
    lo: tuple
    hi: tuple


@dataclass
class Rule:
    at: object = "center"         # (u, v) | "center" | Near | Region
    yaw: float = 0.0              # deg, relative to the surface (used when `facing` is None)
    facing: object = None         # None | ("point", (x, y)) a world plan point | ("yaw", deg)
    align: tuple = ()             # surface edges to push against: "front" "back" "left" "right"
    clear: float = 0.02           # gap (m) to other props and to the surface's edge
    avoid: tuple = ()             # names of props to keep clear of whatever their heights
    host: str = ""                # the prop that owns the surface: it never blocks what stands on it
    seed: int = 0
    samples: int = 400            # random candidates tried for a Region


@dataclass
class Placement:
    """A solved placement and what it came to."""
    name: str
    pose: Pose
    surface: Surface
    clearance: dict = field(default_factory=dict)       # {"edge": m, "prop": (name, m) | None}
    tries: int = 1

    @property
    def yaw_deg(self):
        return math.degrees(self.pose.yaw)

    def report(self):
        """JSON-able digest for the stage output: world position of the origin, yaw, (u, v), clearances in mm."""
        near = self.clearance.get("prop")
        edge = self.clearance.get("edge")
        return {"on": self.surface.name, "position": [round(float(x), 4) for x in self.pose.origin],
                "yaw_deg": round(self.yaw_deg, 2), "uv": [round(x, 4) for x in self.pose.uv],
                "clearance_mm": {"edge": None if edge is None or math.isinf(edge) else round(edge * 1000.0, 1),
                                 "prop": None if near is None else {"name": near[0], "mm": round(near[1] * 1000.0, 1)}}}


def _yaw_of(surface, fp, rule, cu, cv):
    """World yaw (rad) for a footprint centred at (cu, cv) under `rule`."""
    if surface.kind == "wall":
        return yaw_for(surface.N[:2], fp.front)
    if rule.facing is not None and rule.facing[0] == "point":
        C = surface.O + surface.U * cu + surface.V * cv
        d = np.asarray(rule.facing[1], float) - C[:2]
        if float(np.linalg.norm(d)) < 1e-6:
            raise PlaceError(f"{fp.name}: cannot face {tuple(round(float(x), 3) for x in rule.facing[1])}, it is where "
                             f"the prop stands")
        return yaw_for(d, fp.front)
    deg = rule.facing[1] if rule.facing is not None else rule.yaw
    return wrap(surface.angle + math.radians(float(deg)))


def _need(surface, rule):
    """The margin (m) the footprint must keep inside the surface: `clear`, none along an edge."""
    return 0.0 if surface.kind == "edge" else rule.clear


def _inside_by(surface, fp, rule, cu, cv):
    """True when the footprint centred at (cu, cv) keeps its margin inside the surface."""
    pose = pose_on(surface, fp, cu, cv, _yaw_of(surface, fp, rule, cu, cv))
    return edge_margin(surface, pose) >= _need(surface, rule) - TOL


def _align(surface, fp, rule, cu, cv):
    """Push the footprint against the surface edges `rule.align` names, `clear` inside them: the coordinate along an
    aligned axis is replaced, the other one kept."""
    if not rule.align:
        return cu, cv
    axes = {}
    for side in rule.align:
        if side not in SIDES:
            raise PlaceError(f"{fp.name}: align {side!r} is not one of {sorted(SIDES)}")
        ax = 0 if SIDES[side][0] else 1
        if ax in axes:
            raise PlaceError(f"{fp.name}: align {side!r} and {axes[ax]!r} both claim the same axis")
        axes[ax] = side
    if surface.kind == "edge":
        raise PlaceError(f"{fp.name}: align: the edge {surface.name!r} has no sides")
    if surface.shape == "disc" and len(axes) > 1:
        raise PlaceError(f"{fp.name}: align: a round surface can be aligned to one side only")
    for ax, side in sorted(axes.items()):
        d = SIDES[side]
        base = (0.0 if d[0] else cu, 0.0 if d[1] else cv)
        if not _inside_by(surface, fp, rule, *base):
            cu, cv = base                 # the footprint does not fit even here; the checks that follow say why
            continue
        lo, hi = 0.0, 2.0 * max(surface.half) + 1.0 + max(fp.size[:2])
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _inside_by(surface, fp, rule, base[0] + d[0] * mid, base[1] + d[1] * mid):
                lo = mid
            else:
                hi = mid
        cu, cv = base[0] + d[0] * lo, base[1] + d[1] * lo
    return cu, cv


def _problems(surface, pose, others, rule, exempt):
    """([(kind, name, text)], clearance) of a pose: sticking out past an edge, blocked by other props. `exempt` names
    props that may be overlapped (a near-target whose gap the rule controls) unless `avoid` lists them."""
    out = []
    margins = _side_margins(surface, pose)
    worst = min(margins, key=margins.get)
    need = _need(surface, rule)
    if margins[worst] < need - TOL:
        short = (need - margins[worst]) * 1000.0
        if surface.shape == "disc":
            text = f"sticks out of the round {surface.name!r} by {short:.0f} mm (clear {rule.clear * 1000:.0f} mm)"
        elif surface.kind == "edge":
            text = (f"is off the edge {surface.name!r}" if worst == "across"
                    else f"sticks out past the end of the edge {surface.name!r}")
        else:
            side = WALL_NAMES.get(worst, worst) if surface.kind == "wall" else worst
            text = (f"is {short:.0f} mm too close to the {side} edge of {surface.name!r} "
                    f"(clear {rule.clear * 1000:.0f} mm)")
        out.append(("edge", worst, text))
    mine = pose.layers or [pose.prism]
    hits, nearest = {}, None
    for o in others:
        forced = o.name in rule.avoid
        if o.name in exempt and not forced:
            continue
        for n in mine:
            eligible = forced or not (o.flat or n.flat or min(n.z1, o.z1) - max(n.z0, o.z0) <= FLAT_EPS)
            if not eligible:
                continue
            bound = math.hypot(n.rect.cx - o.rect.cx, n.rect.cy - o.rect.cy) \
                - math.hypot(n.rect.hx, n.rect.hy) - math.hypot(o.rect.hx, o.rect.hy)
            if bound >= max(rule.clear, nearest[1] if nearest else math.inf) + 1e-9:
                continue                                     # too far apart to block or to be the nearest
            g = gap(n.rect, o.rect)
            if nearest is None or g < nearest[1]:
                nearest = (o.name, g)
            if g < rule.clear - TOL and (o.name not in hits or g < hits[o.name]):
                hits[o.name] = g
    for name, g in hits.items():
        text = (f"overlaps {name!r} by {-g * 1000:.0f} mm" if g < 0
                else f"is only {g * 1000:.0f} mm from {name!r}") + f" (clear {rule.clear * 1000:.0f} mm)"
        out.append(("overlap" if g < 0 else "close", name, text))
    return out, {"edge": min(margins.values()), "prop": nearest}


def _ring(near, rule, surface, fp, rng):
    """Candidate centres (cu, cv) around the near-target at the asked distance, nearest the asked bearing first."""
    t = near.target
    centre = t.centre if isinstance(t, Rect) else np.asarray(t, float)
    deltas = [0.0]
    for k in range(1, int(180 / RING_STEP)):
        pair = [k * RING_STEP, -k * RING_STEP]
        deltas += pair if rng.random() < 0.5 else pair[::-1]
    deltas.append(180.0)
    reach = abs(near.distance) + max(fp.size[:2]) + (max(t.hx, t.hy) if isinstance(t, Rect) else 0.0)
    for delta in deltas:
        d = heading(math.radians(near.bearing + delta) + near.target_yaw)

        def at_radius(r, d=d):
            C = centre + d * r
            cu, cv = float((C - surface.O[:2]) @ surface.U[:2]), float((C - surface.O[:2]) @ surface.V[:2])
            pose = pose_on(surface, fp, cu, cv, _yaw_of(surface, fp, rule, cu, cv))
            g = gap(pose.prism.rect, t) if isinstance(t, Rect) else gap_to_point(pose.prism.rect, t)
            return g, cu, cv

        hi = 1.0 + 2.0 * reach
        if at_radius(hi)[0] < near.distance:
            continue
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if at_radius(mid)[0] >= near.distance:
                hi = mid
            else:
                lo = mid
        yield at_radius(hi)[1:]


def _candidates(surface, fp, rule, rng):
    at = rule.at
    if isinstance(at, Near):
        return _ring(at, rule, surface, fp, rng)
    if isinstance(at, Region):
        (u0, v0), (u1, v1) = at.lo, at.hi
        return ((float(rng.uniform(u0, u1)), float(rng.uniform(v0, v1))) for _ in range(int(rule.samples)))
    if isinstance(at, str):
        if at == "center":
            return iter([(0.0, 0.0)])
    elif isinstance(at, (tuple, list)) and len(at) in (1, 2) and all(isinstance(x, (int, float)) for x in at):
        return iter([(float(at[0]), float(at[1]) if len(at) == 2 else 0.0)])
    raise PlaceError(f"{fp.name}: `at` must be [u, v], \"center\", near:<target> or a [[u0, v0], [u1, v1]] region, "
                     f"got {at!r}")


def place(surface, fp, others, rule, name=None):
    """Solve one rule for footprint `fp` on `surface` among the prisms `others`. Returns a Placement, or raises
    PlaceError saying why not (the numbers, and the props in the way)."""
    rng = np.random.default_rng(int(rule.seed))
    exempt = {rule.at.target_name} if isinstance(rule.at, Near) and rule.at.target_name else set()
    if rule.host:
        exempt.add(rule.host)
    if surface.kind == "wall" and (rule.facing is not None or rule.yaw):
        raise PlaceError(f"{fp.name}: a prop on a wall faces out of it: drop `facing` and `yaw`")
    tries, reasons, best = 0, {}, None
    for cu, cv in _candidates(surface, fp, rule, rng):
        tries += 1
        if surface.kind == "edge":
            cv = 0.0
        cu, cv = _align(surface, fp, rule, cu, cv)
        pose = pose_on(surface, fp, cu, cv, _yaw_of(surface, fp, rule, cu, cv))
        probs, clearance = _problems(surface, pose, others, rule, exempt)
        if not probs:
            return Placement(name or fp.name, pose, surface, clearance, tries)
        for kind, who, _text in probs:
            reasons[(kind, who)] = reasons.get((kind, who), 0) + 1
        if best is None or len(probs) < len(best[1]):
            best = ((cu, cv), [p[2] for p in probs])
    raise PlaceError(_explain(surface, fp, tries, reasons, best))


def _why(reasons):
    """The commonest reasons candidates failed, as text: 'overlap 'desk' x12; front edge x3'."""
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:4]
    return "; ".join(f"{kind} {who!r} x{n}" if kind != "edge" else f"{who} edge x{n}" for (kind, who), n in top)


def _explain(surface, fp, tries, reasons, best):
    head = f"{fp.name}: no place on {surface.name!r} for its {fp.size[0]:.3f} x {fp.size[1]:.3f} m footprint"
    if best is None:
        return f"{head}: nothing to try (the ring around the near target is empty or the region has no samples)"
    (cu, cv), probs = best
    if tries == 1:
        return f"{head} at u = {cu:.3f}, v = {cv:.3f}: " + "; ".join(probs)
    why = _why(reasons)
    return f"{head}: {tries} candidates tried ({why}). Closest u = {cu:.3f}, v = {cv:.3f}: " + "; ".join(probs)


# ------------------------------------------------------------------------------------------------------ scatter
def scatter(surface, fps, others, *, region=None, min_dist=0.0, yaw=(0.0, 0.0), clear=0.02, avoid=(), seed=0,
            tries=300, names=None, host=""):
    """Place the footprints `fps` (one per item, in order) at random on `surface`, one after the other: the footprint
    centre uniform in `region` ((u0, v0), (u1, v1); default: the whole surface), the yaw uniform in `yaw` (deg, relative
    to the surface; ignored on a wall), clear of `others` (the surface's `host` prop excepted) and of each other by at
    least `max(min_dist, clear)`. Returns the Placements. Raises PlaceError when an item finds no spot in `tries` candidates."""
    rng = np.random.default_rng(int(seed))
    hu, hv = surface.half
    if surface.shape == "disc":
        hv = hu
    reg = ((-hu, -hv), (hu, hv)) if region is None else tuple(tuple(float(x) for x in p) for p in region)
    if surface.kind == "edge":
        reg = ((reg[0][0], 0.0), (reg[1][0], 0.0))
    rule = Rule(clear=clear, avoid=tuple(avoid))
    out, others = [], list(others)
    for i, fp in enumerate(fps):
        label = names[i] if names else fp.name
        placed, reasons = None, {}
        for k in range(int(tries)):
            cu, cv = float(rng.uniform(reg[0][0], reg[1][0])), float(rng.uniform(reg[0][1], reg[1][1]))
            psi = float(rng.uniform(yaw[0], yaw[1])) if yaw[1] > yaw[0] else float(yaw[0])
            pose = pose_on(surface, fp, cu, cv, wrap(surface.angle + math.radians(psi)))
            probs, clearance = _problems(surface, pose, others, rule, {host} if host else set())
            for m in out:
                g = gap(pose.prism.rect, m.pose.prism.rect)
                if g < min_dist - TOL and not any(p[1] == m.name for p in probs):
                    probs.append(("close", m.name, f"is only {g * 1000:.0f} mm from {m.name!r} "
                                                   f"(min_dist {min_dist * 1000:.0f} mm)"))
            if not probs:
                prism = Prism(label, pose.prism.rect, pose.prism.z0, pose.prism.z1, fp.flat)
                layers = [Prism(label, q.rect, q.z0, q.z1, q.flat) for q in pose.layers]
                placed = Placement(label, Pose(pose.origin, pose.yaw, prism, pose.corners, pose.uv, layers), surface,
                                   clearance, k + 1)
                break
            kind, who, _text = probs[0]
            reasons[(kind, who)] = reasons.get((kind, who), 0) + 1
        if placed is None:
            why = _why(reasons)
            raise PlaceError(f"scatter: placed {len(out)} of {len(fps)}; {label} ({fp.size[0]:.3f} x {fp.size[1]:.3f} m) "
                             f"found no spot in {tries} tries on {surface.name!r} ({why})")
        out.append(placed)
        others.extend(placed.pose.layers or [placed.pose.prism])
    return out
