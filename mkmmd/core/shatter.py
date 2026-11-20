"""The shatter, bpy-free and numpy-free (docs/design.md: Transitions and inserts): `[[transition]] kind = "shatter"`. The
outgoing picture breaks like a thin sheet of plastic: cracks run out from an impact point over the first frames, then the
shards come away, each tumbling about its own axis in the camera's perspective toward the lens and down under gravity,
and all of them have left the frame by the cut. This module checks it and works out the shards and where each one is on
every window frame; mkmmd.shatter draws them.

    center = [0.5, 0.45]    the impact: [x, y] frame fractions from the top left, or {point = "<mk q expression>"}, a
                            point of the scene seen through the outgoing shot's camera on the window's first frame
    material = "plastic"    "plastic" (chunky shards, a pale back, a bright edge, glints), "paper" (a softer, slower,
                            fluttering fall, no glints) or "glass" (long splinters, the picture mirrored on the back,
                            the strongest glints)
    count = 80              about how many shards, 12 .. 400
    seed = 1                another break
    crack = 0.05            seconds the cracks take to run out before anything moves (at least one frame)
    shadow = 0.4            how dark the shadows the shards cast on the incoming picture, 0..1

The shards are the cells of the frame grown from seeded points (a Voronoi partition clipped to the frame): rings of points
round the impact, each ring `growth` times the last one's radius and its points between seeded spokes, so the cracks run
out radially and the shards are small near the impact and large far from it. Each shard keeps its piece of the picture
(the picture goes on playing on it). It comes away later the farther it lies from the impact, and from then on moves on
its own clock s = 0..1 (1 on the window's last frame): out from the impact at a speed set so that by s = 1 it has left the
frame (in perspective, turned as it is then), down by gravity, up toward the lens and turned about its own axis."""
import math
import random

from . import sheet3d as S3

KEYS = {"at", "kind", "dur", "center", "material", "count", "seed", "crack", "shadow"}
DUR = 0.5
MATERIALS = {
    # growth: each ring's radius over the last one's; jitter: how far points stray (share of their spacing); gravity:
    # the fall by s = 1 (short sides); spin: turns by s = 1; lens: how near the lens the nearest shards come (share of
    # the camera distance); glint: the specular highlight's strength; thick: the sheet's thickness (short sides);
    # flutter: the sideways sway of a falling leaf (short sides); delay: the share of the fly time the farthest shard waits
    "plastic": {"growth": 1.55, "jitter": 0.35, "gravity": 0.9, "spin": 1.1, "lens": 0.55, "glint": 0.45,
                "thick": 0.005, "flutter": 0.0, "delay": 0.55},
    "paper": {"growth": 1.45, "jitter": 0.6, "gravity": 0.5, "spin": 0.8, "lens": 0.35, "glint": 0.0,
              "thick": 0.002, "flutter": 0.06, "delay": 0.6},
    "glass": {"growth": 1.75, "jitter": 0.18, "gravity": 1.0, "spin": 1.3, "lens": 0.6, "glint": 0.7,
              "thick": 0.007, "flutter": 0.0, "delay": 0.5},
}
DEFAULTS = {"center": [0.5, 0.45], "material": "plastic", "count": 80, "seed": 1, "crack": 0.05, "shadow": 0.4}
CRUSH = 0.035                   # the radius of the first ring round the impact (short sides)


def normalize(spec, out, palette, what):
    """A shatter's own keys with their defaults laid into `out` (the transition's common keys), colours resolved."""
    from . import transition as TR
    from .iris import _lightest
    c = spec.get("center", DEFAULTS["center"])
    if isinstance(c, dict):
        if set(c) != {"point"} or not isinstance(c["point"], str) or not c["point"].strip():
            raise TR.TransitionError(f"{what}: center = {c!r}: expected {{point = \"<expression>\"}} or [x, y]")
        out["center"] = {"mode": "point", "expr": c["point"]}
    else:
        out["center"] = {"mode": "frame", "at": TR._fractions(c, f"{what}: center")}
    m = spec.get("material", DEFAULTS["material"])
    if m not in MATERIALS:
        raise TR.TransitionError(f"{what}: material = {m!r}, expected one of {tuple(MATERIALS)}")
    out["material"] = m
    for k, lo, hi in (("count", 12, 400), ("seed", -2 ** 31, 2 ** 31 - 1)):
        v = spec.get(k, DEFAULTS[k])
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise TR.TransitionError(f"{what}: {k} = {v!r}: expected a whole number in {lo} .. {hi}")
        out[k] = int(v)
    for k, lo, hi in (("crack", 0.0, 1.0), ("shadow", 0.0, 1.0)):
        v = spec.get(k, DEFAULTS[k])
        try:
            if isinstance(v, bool):
                raise TypeError
            out[k] = float(v)
        except (TypeError, ValueError):
            raise TR.TransitionError(f"{what}: {k} = {v!r}: expected a number") from None
        if not lo <= out[k] <= hi:
            raise TR.TransitionError(f"{what}: {k} = {out[k]}: expected {lo} .. {hi}")
    out["back"] = TR._colour(_lightest(palette), palette, f"{what}: back")
    out["ink"] = TR._colour(S3.darkest(palette), palette, f"{what}: shadow")
    return out


def crack_frames(tr, fps):
    """The window frames the cracks take to run out (at least one, and two short of the window at most)."""
    return max(1, min(int(round(tr["crack"] * fps)), tr["n"] - 2))


# ------------------------------------------------------------------------------------------------ the cells
def seeds(tr, size, impact):
    """[(x, y)]: the points the shards grow from (px), for a frame `size` = (w, h) and the impact at `impact` (px)."""
    mat = MATERIALS[tr["material"]]
    rng = random.Random(f"shatter:{tr['seed']}")
    w, h = size
    S = float(min(size))
    px, py = impact
    reach = max(math.hypot(x - px, y - py) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)) + 0.25 * S
    radii, r = [], CRUSH * S
    while r < reach:
        radii.append(r)
        r *= mat["growth"]
    radii.append(r)
    spokes = max(5, int(round((tr["count"] - 4) / len(radii))))
    base = rng.uniform(0.0, 2.0 * math.pi)
    turn = [base + 2.0 * math.pi * (i + rng.uniform(-0.3, 0.3)) / spokes for i in range(spokes)]
    out = [(px + rng.uniform(-0.4, 0.4) * CRUSH * S, py + rng.uniform(-0.4, 0.4) * CRUSH * S) for _ in range(4)]
    for j, r in enumerate(radii):
        step = 2.0 * math.pi / spokes
        for i in range(spokes):
            a = turn[i] + 0.5 * step + mat["jitter"] * step * rng.uniform(-0.5, 0.5)
            rr = r * (1.0 + mat["jitter"] * (mat["growth"] - 1.0) * rng.uniform(-0.5, 0.5))
            out.append((px + rr * math.cos(a), py + rr * math.sin(a)))
    return out


def _clip(poly, labels, a, b, c, label):
    """The part of the polygon `poly` (its edges labelled: edge k runs from vertex k to k + 1) where a x + b y <= c, the
    new edge along the line labelled `label`."""
    out, lab = [], []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        dp, dq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if dp <= 0.0:                                # kept: its edge goes on along the old line (or leaves it)
            out.append(p)
            lab.append(labels[i])
        if (dp <= 0.0) != (dq <= 0.0):               # crossing: leaving, the new edge runs along the clip line
            t = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
            lab.append(label if dp <= 0.0 else labels[i])
    return out, lab


def cells(tr, size, impact):
    """[{"seed": (x, y), "poly": [(x, y), ...] listed clockwise as seen (x right, y down: `area` positive),
    "labels": [neighbour index of edge k, -1 on the frame's border], "index"}]: the shards, the Voronoi cells of
    `seeds` clipped to the frame (px, the frame's outer edge at -0.5 and w - 0.5). Together they cover the frame once."""
    w, h = size
    pts = seeds(tr, size, impact)
    frame = [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]
    out = []
    for i, (sx, sy) in enumerate(pts):
        order = sorted((j for j in range(len(pts)) if j != i),
                       key=lambda j: (pts[j][0] - sx) ** 2 + (pts[j][1] - sy) ** 2)
        poly, lab = list(frame), [-1, -1, -1, -1]
        for j in order:
            qx, qy = pts[j]
            d2 = (qx - sx) ** 2 + (qy - sy) ** 2
            if d2 <= 1e-12:
                continue
            far = max((x - sx) ** 2 + (y - sy) ** 2 for x, y in poly)
            if d2 / 4.0 > far:                       # every later point is farther: its bisector misses the cell
                break
            a, b = qx - sx, qy - sy                  # keep the side nearer to i: a x + b y <= the bisector's value
            c = (qx * qx + qy * qy - sx * sx - sy * sy) / 2.0
            poly, lab = _clip(poly, lab, a, b, c, j)
            if len(poly) < 3:
                break
        if len(poly) >= 3 and abs(area(poly)) > 1e-6:
            out.append({"index": i, "seed": (sx, sy), "poly": poly, "labels": lab})
    ids = {c["index"]: k for k, c in enumerate(out)}
    for c in out:                                    # neighbours by their place in the list
        c["labels"] = [ids.get(j, -1) if j >= 0 else -1 for j in c["labels"]]
    for k, c in enumerate(out):
        c["index"] = k
    return out


def area(poly):
    """The signed area of a polygon (px²; positive when listed clockwise to the eye, x right and y down)."""
    s = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def centroid(poly):
    a = area(poly)
    cx = cy = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        f = x0 * y1 - x1 * y0
        cx += (x0 + x1) * f
        cy += (y0 + y1) * f
    return cx / (6.0 * a), cy / (6.0 * a)


def cracks(shards):
    """[(i, j, (x0, y0), (x1, y1))]: the crack between shards i < j, once each."""
    out = []
    for c in shards:
        poly = c["poly"]
        for k, j in enumerate(c["labels"]):
            if j > c["index"]:
                out.append((c["index"], j, poly[k], poly[(k + 1) % len(poly)]))
    return out


# ------------------------------------------------------------------------------------------------ the flight
def _rot(axis, ang):
    """The rotation matrix (rows) turning `ang` radians about the unit `axis` (Rodrigues)."""
    x, y, z = axis
    c, s, t = math.cos(ang), math.sin(ang), 1.0 - math.cos(ang)
    return ((t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c))


def flights(tr, size, impact, shards):
    """Each shard's flight: [{"delay" (the share of the fly phase before it moves), "g" (its centroid), "v" (px it is
    thrown by s = 1), "fall" (px it falls by s = 1), "z" (px toward the lens by s = 1), "axis", "turn" (radians by s =
    1), "flutter" (px of sideways sway), "near" (1 at the impact, 0 at the farthest corner: how much it is thrown rather
    than let fall)}], each thrown just hard enough to have left the frame by s = 1."""
    mat = MATERIALS[tr["material"]]
    rng = random.Random(f"shatter-flight:{tr['seed']}")
    w, h = size
    S = float(min(size))
    D = S3.distance(size)
    px, py = impact
    far = max(math.hypot(x - px, y - py) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5))
    out = []
    for c in shards:
        g = centroid(c["poly"])
        dx, dy = g[0] - px, g[1] - py
        d = math.hypot(dx, dy)
        near = 1.0 - min(d / far, 1.0)
        a = math.atan2(dy, dx) if d > 1e-6 else rng.uniform(0.0, 2.0 * math.pi)
        a += rng.uniform(-0.35, 0.35)
        u = (math.cos(a), math.sin(a))
        tilt = rng.uniform(0.0, 2.0 * math.pi)
        axis_z = rng.uniform(-0.25, 0.25)
        k = math.sqrt(1.0 - axis_z * axis_z)
        f = {"delay": mat["delay"] * min((1.0 - near) ** 0.8 + rng.uniform(-0.06, 0.06), 1.0),
             "g": g, "fall": mat["gravity"] * S * rng.uniform(0.8, 1.2),
             "z": mat["lens"] * D * (0.25 + 0.75 * near ** 1.5) * rng.uniform(0.7, 1.1),
             "axis": (k * math.cos(tilt), k * math.sin(tilt), axis_z),
             "turn": mat["spin"] * 2.0 * math.pi * rng.uniform(0.6, 1.3) * rng.choice((-1.0, 1.0)),
             "flutter": mat["flutter"] * S * rng.uniform(0.6, 1.4), "phase": rng.uniform(0.0, 2.0 * math.pi),
             "near": near}
        f["delay"] = max(0.0, f["delay"])
        speed = S * (0.25 + 0.6 * near)
        for _ in range(40):
            f["v"] = (u[0] * speed, u[1] * speed)
            if _gone(c["poly"], f, size):
                break
            speed *= 1.2
        out.append(f)
    return out


def at(f, s):
    """(dx, dy, z, rotation) of a flight at its own time s = 0..1."""
    s = min(max(s, 0.0), 1.0)
    e = s ** (2.0 - f["near"])                      # thrown at once by the impact, or falling away from rest
    sway = f["flutter"] * math.sin(f["phase"] + 3.0 * math.pi * s) * s
    return (f["v"][0] * e + sway, f["v"][1] * e + f["fall"] * s * s, f["z"] * s ** 1.2, _rot(f["axis"], f["turn"] * e))


def homography(f, s, size):
    """The 3x3 matrix (rows) that takes a point of the outgoing picture (px) on the shard of flight `f` to where it is
    seen at its time s (homogeneous: divide by the third row)."""
    dx, dy, z, R = at(f, s)
    gx, gy = f["g"]
    D = S3.distance(size)
    cx, cy = S3.centre(size)
    # X = R [x - gx, y - gy, 0] + [gx + dx, gy + dy, z]: affine in (x, y, 1)
    A = [[R[r][0], R[r][1], -R[r][0] * gx - R[r][1] * gy + (gx + dx, gy + dy, z)[r]] for r in range(3)]
    P = ((D, 0.0, -cx, 0.0), (0.0, D, -cy, 0.0), (0.0, 0.0, -1.0, D))
    A4 = A + [[0.0, 0.0, 1.0]]
    return [[sum(P[r][k] * A4[k][q] for k in range(4)) for q in range(3)] for r in range(3)]


def apply(H, x, y):
    X = H[0][0] * x + H[0][1] * y + H[0][2]
    Y = H[1][0] * x + H[1][1] * y + H[1][2]
    W = H[2][0] * x + H[2][1] * y + H[2][2]
    return X / W, Y / W


def _gone(poly, f, size):
    """Whether the shard `poly` is wholly outside the frame at the end of its flight."""
    w, h = size
    H = homography(f, 1.0, size)
    pts = [apply(H, x, y) for x, y in poly]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) < -1.0 or min(xs) > w or max(ys) < -1.0 or min(ys) > h


def clock(tr, f, k, fps):
    """A shard's own time s at window frame `k`: 0 through the cracks and until its delay, 1 on the window's last frame."""
    nc = crack_frames(tr, fps)
    if k < nc:
        return 0.0
    tau = (k - nc + 1) / (tr["n"] - nc)
    return min(max((tau - f["delay"]) / (1.0 - f["delay"]), 0.0), 1.0)
