"""Pure maths (no bpy, importable from tests) behind bedroom_furniture.py: the rake of the desk chair's backrest, outlines
of rounded rectangles, bent-tube centre lines and evenly spaced stitch marks.

Chair frame. The backrest collider of `cafe_chair` (a 0.38 x 0.030 x 0.46 box) is the reference every chair with the same
`use.sit` follows. Its centre is BACK_C, its axes are the chair axes rotated by -BACK_TILT about X, so its local +Z leans
back (toward +Y) and its local -Y face looks at the sitter. `pad_point(x, yp, zp)` maps a point of that frame (the
"pad frame": x across, yp through the cushion, zp up the cushion) into the chair frame.

Outlines. `rrect` is the outline of a rounded rectangle; shrinking it by `inset` keeps the point count, so rings of
different insets join into a loft without a remap. `round_path` turns corner points into a dense centre line with circular
fillets (a tube sweep follows it); `dash_marks` spaces stitch marks evenly along a closed outline."""
import math

SEAT_Z = 0.45                                                   # top of the seat
BACK_TILT = math.atan2(0.244 - 0.170, 0.853 - 0.450)            # rake of the backrest (rad), 10.4 deg


def back_y(z):
    """Backrest plane of the chair: y as a function of height (the plane through the collider's centre line)."""
    return 0.170 + (z - SEAT_Z) * math.tan(BACK_TILT)


BACK_C = (0.0, back_y(0.66), 0.66)                              # centre of the backrest collider box


def pad_point(x, yp, zp):
    """Pad frame (origin BACK_C, axes rotated -BACK_TILT about X) -> chair frame."""
    c, s = math.cos(BACK_TILT), math.sin(BACK_TILT)
    return (x, BACK_C[1] + yp * c + zp * s, BACK_C[2] - yp * s + zp * c)


def pad_dir(dx, dyp, dzp):
    """Direction in the pad frame -> chair frame."""
    c, s = math.cos(BACK_TILT), math.sin(BACK_TILT)
    return (dx, dyp * c + dzp * s, -dyp * s + dzp * c)


# ================================================================= outlines
def rrect(w, h, r, n=6, inset=0.0):
    """Counter-clockwise outline (list of (x, y), 4 * (n + 1) points) of a w x h rectangle centred on the origin with
    corner radius r, shrunk by `inset` on every side. The radius shrinks with the inset (never below 0.1 mm), so every
    inset of the same (w, h, r, n) has the same point count and point k of one ring faces point k of the next."""
    hw, hh = w / 2.0 - inset, h / 2.0 - inset
    if hw <= 0 or hh <= 0:
        raise ValueError(f"rrect {w} x {h} inset {inset}: nothing left")
    rr = max(min(r - inset, hw, hh), 1e-4)
    out = []
    for cx, cy, a0 in ((hw - rr, hh - rr, 0.0), (rr - hw, hh - rr, 90.0), (rr - hw, rr - hh, 180.0),
                       (hw - rr, rr - hh, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            out.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return out


def perimeter(pts, closed=True):
    m = len(pts)
    return sum(math.dist(pts[i], pts[(i + 1) % m]) for i in range(m if closed else m - 1))


def dash_marks(pts, pitch, closed=True):
    """Evenly spaced marks along a polyline of 2D or 3D points: [(point, unit tangent)]. The count is the whole number
    of `pitch` steps that fits the length, so the spacing is `pitch` rounded to close the loop; the first mark sits half a
    step along the first segment."""
    m = len(pts)
    segs = [(pts[i], pts[(i + 1) % m]) for i in range(m if closed else m - 1)]
    lens = [math.dist(a, b) for a, b in segs]
    total = sum(lens)
    count = max(int(round(total / pitch)), 1)
    step = total / count
    out, k, acc = [], 0, 0.0
    for (a, b), ln in zip(segs, lens):
        if ln < 1e-12:
            continue
        t = tuple((bb - aa) / ln for aa, bb in zip(a, b))
        while k < count and (k + 0.5) * step <= acc + ln + 1e-12:
            s = (k + 0.5) * step - acc
            out.append((tuple(aa + tt * s for aa, tt in zip(a, t)), t))
            k += 1
        acc += ln
    return out


# ================================================================= bent tubes
def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _mul(a, s):
    return tuple(x * s for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _len(a)
    if n < 1e-12:
        raise ValueError("round_path: two corner points coincide")
    return _mul(a, 1.0 / n)


def _rotate(v, axis, ang):
    """Rodrigues: v rotated about the unit `axis` by `ang`."""
    c, s = math.cos(ang), math.sin(ang)
    return _add(_add(_mul(v, c), _mul(_cross(axis, v), s)), _mul(axis, _dot(axis, v) * (1.0 - c)))


def round_path(pts, radii, arc_deg=6.0, step=0.05, closed=False):
    """Dense centre line through the corner points `pts` (3-tuples) with a circular fillet of radius radii[i] (a number or
    one per point) at every corner: the interior corners of an open path, all corners of a closed one. Straight runs are
    cut into pieces of at most `step`, arcs get a point per `arc_deg`. A fillet that would not fit between its
    neighbours is shortened to half of the shorter adjacent run (so two fillets of one run never overlap). A closed path
    does not repeat its first point. Returns a list of 3-tuples without repeated points."""
    P = [tuple(float(c) for c in p) for p in pts]
    n = len(P)
    if n < 2:
        raise ValueError("round_path needs at least two points")
    rad = list(radii) if isinstance(radii, (list, tuple)) else [float(radii)] * n
    corners = []                                     # per point: None or (A, B, centre, axis, turn angle)
    for i in range(n):
        if not closed and i in (0, n - 1):
            corners.append(None)
            continue
        p0, p1, p2 = P[(i - 1) % n], P[i], P[(i + 1) % n]
        d1, d2 = _unit(_sub(p1, p0)), _unit(_sub(p2, p1))
        cosang = max(-1.0, min(1.0, _dot(d1, d2)))
        turn = math.acos(cosang)
        r = float(rad[i])
        if turn < 1e-6 or r <= 0.0:
            corners.append(None)
            continue
        if turn > math.pi - 1e-6:
            raise ValueError(f"round_path: corner {i} doubles back")
        t = r * math.tan(turn / 2.0)
        tmax = 0.5 * min(_len(_sub(p1, p0)), _len(_sub(p2, p1)))
        if t > tmax:
            r, t = tmax / math.tan(turn / 2.0), tmax
        axis = _unit(_cross(d1, d2))
        a, b = _sub(p1, _mul(d1, t)), _add(p1, _mul(d2, t))
        centre = _add(p1, _mul(_unit(_sub(d2, d1)), r / math.cos(turn / 2.0)))
        corners.append((a, b, centre, axis, turn))

    out = []

    def emit(p):
        if not out or _len(_sub(p, out[-1])) > 1e-9:
            out.append(p)

    def between(a, b):                              # points strictly between a and b, at most `step` apart
        m = max(int(math.ceil(_len(_sub(b, a)) / step - 1e-9)), 1)
        for k in range(1, m):
            emit(_add(a, _mul(_sub(b, a), k / m)))

    entry = [c[0] if c else P[i] for i, c in enumerate(corners)]       # where the path reaches / leaves corner i
    leave = [c[1] if c else P[i] for i, c in enumerate(corners)]
    for i in range(n):
        c = corners[i]
        if c is None:
            emit(P[i])
        else:
            a, b, centre, axis, turn = c
            m = max(int(math.ceil(math.degrees(turn) / arc_deg - 1e-9)), 1)
            va = _sub(a, centre)
            for k in range(m):
                emit(_add(centre, _rotate(va, axis, turn * k / m)))
            emit(b)
        if closed:
            between(leave[i], entry[(i + 1) % n])
        elif i < n - 1:
            between(leave[i], entry[i + 1])
    if closed and len(out) > 1 and _len(_sub(out[0], out[-1])) < 1e-9:
        out.pop()
    return out
