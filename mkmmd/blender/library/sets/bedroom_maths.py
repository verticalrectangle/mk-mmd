"""Pure maths of the bedroom_80s set (no bpy, importable from tests): the spec with every default filled in and checked,
the wall frames and watertight wall slabs (with a window opening or a door pocket), the parquet layout, the venetian
blind's slat levels and the card.

Frames. The room is the box [x0, x1] x [y0, y1] x [0, H] (set frame, floor z = 0). Every wall has a frame (u, v, w):
u runs along the wall in WORLD coordinates (x for the back and front walls, y for the left and right ones), v is z, w
is the depth into the wall, measured outwards from the interior plane. `wall_mesh` returns the slab in world space."""
import math

import numpy as np

BUILD_KEYS = {"name", "kind", "at", "yaw"}                  # keys of the build stage itself
WALLS = ("back", "left", "right", "front")                  # back = the window wall (+Y), right = the door wall (+X)
OPENABLE = WALLS + ("ceiling", "floor")

# animatable parameters: name -> (default, min, max, description); custom properties on the set root
PARAMS = {
    "blinds": (0.5, 0.0, 1.0, "venetian blind: 0 = fully lowered over the window, 1 = fully raised (piled at the top)"),
    "slat_angle": (20.0, -85.0, 85.0, "slat tilt in degrees: 0 flat; positive lifts the outer (street-side) edge"),
    "window_glow": (1.0, 0.0, 3.0, "strength of the light that comes in through the window (1 = as built)"),
    "city_glow": (1.0, 0.0, 3.0, "brightness of the city's lit windows, glow and beacons (1 = as built)"),
    "neon": (0.7, 0.0, 1.0, "the neon strip's brightness (0 = off)"),
}
KEYS = BUILD_KEYS | {"size", "wall", "open", "seed", "render", "window", "door", "floor", "pendant", "neon_wall",
                     "fill", "city", "near", "sky", "colors"} | set(PARAMS)

WINDOW = {"x": 0.4, "width": 1.3, "height": 1.2, "sill": 0.95, "cols": 2, "rows": 2}
DOOR = {"y": -1.0, "width": 0.9, "height": 2.05}

# depth coordinates w (m, outwards from the interior wall plane; negative = in the room)
CASING_W, CASING_T = 0.075, 0.018        # architrave board: width, how far it stands off the wall
FRAME_W0, FRAME_W1 = 0.09, 0.15          # window frame (lining): front and back faces; the glass sits half-way
SILL_OUT = 0.06                          # the sill board's reach into the room
BLIND_W = -0.035                         # centre plane of the blind's slats (in front of the casing)
POCKET = 0.10                            # depth of the door pocket in the wall
LEAF_W0, LEAF_W1 = 0.04, 0.08            # door leaf: front and back faces
RAIL_DROP = 0.23                         # the picture rail's reference height is this far below the ceiling
SKIRT_H = 0.12

BLIND_OVER = 0.10                        # blind width = window width + this
HEAD_OVER = 0.07                         # headrail bottom above the opening's top
HEAD_H = 0.042                           # headrail height
SLAT_C = 0.025                           # slat chord (m)
PITCH_MIN = 0.0016                       # pitch of slats stacked on each other


# ---------------------------------------------------------------------------------------------------- the spec


def _table(spec, key, defaults, who):
    """A sub-table merged over its defaults; unknown keys are errors."""
    over = spec.get(key, {})
    if not isinstance(over, dict):
        raise ValueError(f"{who}: {key} must be a table, not {over!r}")
    bad = sorted(set(over) - set(defaults))
    if bad:
        raise ValueError(f"{who}: unknown keys {bad} in {key} (have {sorted(defaults)})")
    return {**defaults, **over}


def _option(spec, key, who):
    """A true / false / table option -> None (off) or a dict of overrides (empty for true)."""
    v = spec.get(key, True)
    if v is False or v is None:
        return None
    if v is True:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"{who}: {key} must be true, false or a table, not {v!r}")
    return {k: x for k, x in v.items() if k != "on"} if v.get("on", True) else None


def parse(name, spec):
    """The spec with every default filled in and checked -> dict (see the builder's docstring for the keys)."""
    who = f"bedroom_80s {name!r}"
    bad = sorted(set(spec) - KEYS)
    if bad:
        raise ValueError(f"{who}: unknown keys {bad} (have {sorted(KEYS - BUILD_KEYS)})")
    try:
        W, D, H = (float(v) for v in spec.get("size", (3.6, 3.2, 2.55)))
    except (TypeError, ValueError):
        raise ValueError(f"{who}: size must be [width, depth, height], not {spec.get('size')!r}")
    if min(W, D) < 2.0 or not 2.2 <= H <= 4.0:
        raise ValueError(f"{who}: size must be [width >= 2, depth >= 2, height 2.2 .. 4], not {[W, D, H]}")
    T = float(spec.get("wall", 0.2))
    if not 0.16 <= T <= 0.5:
        raise ValueError(f"{who}: wall must be 0.16 .. 0.5 m, not {T}")
    shut = spec.get("open", [])
    shut = [shut] if isinstance(shut, str) else [str(v) for v in shut]
    if any(v not in OPENABLE for v in shut):
        raise ValueError(f"{who}: open must list some of {list(OPENABLE)}, not {shut}")
    win = _table(spec, "window", WINDOW, who)
    win = {k: (int(v) if k in ("cols", "rows") else float(v)) for k, v in win.items()}
    win.update(x0=win["x"] - win["width"] / 2, x1=win["x"] + win["width"] / 2, z0=win["sill"],
               z1=win["sill"] + win["height"])
    if win["cols"] < 1 or win["rows"] < 1 or win["cols"] > 4 or win["rows"] > 3:
        raise ValueError(f"{who}: window cols 1 .. 4 and rows 1 .. 3, not {win['cols']} x {win['rows']}")
    if win["width"] < 0.4 or win["height"] < 0.4 or win["x0"] < -W / 2 + 0.15 - 1e-9 or win["x1"] > W / 2 - 0.15 + 1e-9:
        raise ValueError(f"{who}: the window (x {win['x0']:.2f} .. {win['x1']:.2f}) must fit the back wall with 0.15 m "
                         f"to spare and be at least 0.4 m big")
    if win["z0"] < 0.3 - 1e-9 or win["z1"] > H - 0.38 + 1e-9:
        raise ValueError(f"{who}: the window (z {win['z0']:.2f} .. {win['z1']:.2f}) must keep 0.3 m to the floor and "
                         f"0.38 m to the ceiling (headrail and picture rail)")
    door = _table(spec, "door", DOOR, who)
    door = {k: float(v) for k, v in door.items()}
    door.update(y0=door["y"] - door["width"] / 2, y1=door["y"] + door["width"] / 2, z1=door["height"])
    if (door["width"] < 0.5 or door["y0"] < -D / 2 + 0.15 - 1e-9 or door["y1"] > D / 2 - 0.15 + 1e-9 or
            door["height"] > H - 0.35):
        raise ValueError(f"{who}: the door (y {door['y0']:.2f} .. {door['y1']:.2f}, {door['height']} high) must fit "
                         f"the right wall with 0.15 m to spare")
    floor = spec.get("floor", "herringbone")
    if floor not in ("herringbone", "boards"):
        raise ValueError(f"{who}: floor must be 'herringbone' or 'boards', not {floor!r}")
    neon = spec.get("neon", True)
    if neon is True:
        neon = PARAMS["neon"][0]
    elif neon is False or neon is None:
        neon = None
    else:
        neon = float(neon)
    neon_wall = spec.get("neon_wall", "left")
    if neon_wall not in WALLS:
        raise ValueError(f"{who}: neon_wall must be one of {list(WALLS)}, not {neon_wall!r}")
    params = {}
    for k, (default, lo, hi, _) in PARAMS.items():
        v = (0.0 if neon is None else neon) if k == "neon" else float(spec.get(k, default))
        if not lo <= v <= hi:
            raise ValueError(f"{who}: {k} must be in [{lo}, {hi}], not {v}")
        params[k] = v
    city = _option(spec, "city", who)
    near = _option(spec, "near", who) if "near" in spec else (None if city is None else {})
    cfg = {"name": name, "size": (W, D, H), "x0": -W / 2, "x1": W / 2, "y0": -D / 2, "y1": D / 2, "H": H, "T": T,
           "rail_z": H - RAIL_DROP, "open": shut, "seed": int(spec.get("seed", 80)),
           "render": bool(spec.get("render", True)),
           "window": win, "door": door, "floor": floor, "pendant": bool(spec.get("pendant", True)), "neon": neon,
           "neon_wall": neon_wall, "fill": float(spec.get("fill", 1.0)), "params": params,
           "city": city, "near": near, "sky": _option(spec, "sky", who),
           "colors": dict(spec.get("colors") or {})}
    return cfg


# ---------------------------------------------------------------------------------------------------- walls


def wall_frame(cfg, wall):
    """Frame of a wall: {"origin": a point of the interior plane at u = 0, "U": unit vector along the wall, "O": unit
    vector into the wall, "n": the interior normal, "u0", "u1": the interior's extent in u}."""
    x0, x1, y0, y1 = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"]
    if wall == "back":
        o, U, O, ext = np.array([0.0, y1, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), (x0, x1)
    elif wall == "front":
        o, U, O, ext = np.array([0.0, y0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), (x0, x1)
    elif wall == "left":
        o, U, O, ext = np.array([x0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), (y0, y1)
    elif wall == "right":
        o, U, O, ext = np.array([x1, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), (y0, y1)
    else:
        raise ValueError(f"no wall {wall!r} (have {list(WALLS)})")
    return {"origin": o, "U": U, "O": O, "n": -O, "u0": ext[0], "u1": ext[1]}


def to_world(fr, u, v, w):
    """World point(s) of frame coordinates (broadcasts)."""
    u, v, w = (np.asarray(a, float) for a in (u, v, w))
    return fr["origin"] + u[..., None] * fr["U"] + v[..., None] * np.array([0.0, 0.0, 1.0]) + w[..., None] * fr["O"]


def _breaks(*vals):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > 1e-9:
            out.append(v)
    return out


def slab(rect, t, interior, hole=None, pocket=None):
    """A wall slab in frame coordinates: the rectangle rect = (u0, u1, v0, v1), thickness t along w, with a rectangular
    `hole` = (hu0, hu1, hv0, hv1) through it (or a pocket of depth `pocket` in the interior face). `interior` = (u0, u1,
    v0, v1) is the part of the interior plane that is inside the room. Returns (V (n, 3) in (u, v, w), faces [(4
    vertex indices)], normals (m, 3) the outward normal wanted for each face, tags): tags are "room" (the interior
    face inside the room), "outer" (everything else on the outside), "reveal" (the sides of the hole or pocket) and
    "pocket" (the back of a pocket). Every edge is shared by two faces (a closed surface)."""
    u0, u1, v0, v1 = rect
    iu0, iu1, iv0, iv1 = interior
    ub = _breaks(u0, u1, iu0, iu1, *(hole[:2] if hole else ()))
    vb = _breaks(v0, v1, iv0, iv1, *(hole[2:] if hole else ()))
    ub = [u for u in ub if u0 - 1e-9 <= u <= u1 + 1e-9]
    vb = [v for v in vb if v0 - 1e-9 <= v <= v1 + 1e-9]
    verts, index, faces, normals, tags = [], {}, [], [], []

    def vid(u, v, w):
        k = (round(u, 6), round(v, 6), round(w, 6))
        if k not in index:
            index[k] = len(verts)
            verts.append((u, v, w))
        return index[k]

    def quad(pts, n, tag):
        faces.append([vid(*p) for p in pts])
        normals.append(n)
        tags.append(tag)

    def in_hole(ua, ub_, va, vb_):
        return (hole is not None and ua >= hole[0] - 1e-9 and ub_ <= hole[1] + 1e-9 and
                va >= hole[2] - 1e-9 and vb_ <= hole[3] + 1e-9)

    for i in range(len(ub) - 1):
        for j in range(len(vb) - 1):
            a, b, c, d = ub[i], ub[i + 1], vb[j], vb[j + 1]
            inside_room = a >= iu0 - 1e-9 and b <= iu1 + 1e-9 and c >= iv0 - 1e-9 and d <= iv1 + 1e-9
            hole_cell = in_hole(a, b, c, d)
            if not hole_cell:
                quad([(a, c, 0), (b, c, 0), (b, d, 0), (a, d, 0)], (0, 0, -1), "room" if inside_room else "outer")
            if not hole_cell or pocket is not None:                  # a pocket is closed on the outside
                quad([(a, c, t), (b, c, t), (b, d, t), (a, d, t)], (0, 0, 1), "outer")
    for i in range(len(ub) - 1):                                    # bottom and top sides
        quad([(ub[i], v0, 0), (ub[i + 1], v0, 0), (ub[i + 1], v0, t), (ub[i], v0, t)], (0, -1, 0), "outer")
        quad([(ub[i], v1, 0), (ub[i + 1], v1, 0), (ub[i + 1], v1, t), (ub[i], v1, t)], (0, 1, 0), "outer")
    for j in range(len(vb) - 1):                                    # left and right sides
        quad([(u0, vb[j], 0), (u0, vb[j + 1], 0), (u0, vb[j + 1], t), (u0, vb[j], t)], (-1, 0, 0), "outer")
        quad([(u1, vb[j], 0), (u1, vb[j + 1], 0), (u1, vb[j + 1], t), (u1, vb[j], t)], (1, 0, 0), "outer")
    if hole is not None:
        hu0, hu1, hv0, hv1 = hole
        depth = t if pocket is None else pocket
        us = [u for u in ub if hu0 - 1e-9 <= u <= hu1 + 1e-9]
        vs = [v for v in vb if hv0 - 1e-9 <= v <= hv1 + 1e-9]
        for i in range(len(us) - 1):                                # the sill and the head of the opening
            quad([(us[i], hv0, 0), (us[i + 1], hv0, 0), (us[i + 1], hv0, depth), (us[i], hv0, depth)], (0, 1, 0),
                 "reveal")
            quad([(us[i], hv1, 0), (us[i + 1], hv1, 0), (us[i + 1], hv1, depth), (us[i], hv1, depth)], (0, -1, 0),
                 "reveal")
        for j in range(len(vs) - 1):                                # the jambs
            quad([(hu0, vs[j], 0), (hu0, vs[j + 1], 0), (hu0, vs[j + 1], depth), (hu0, vs[j], depth)], (1, 0, 0),
                 "reveal")
            quad([(hu1, vs[j], 0), (hu1, vs[j + 1], 0), (hu1, vs[j + 1], depth), (hu1, vs[j], depth)], (-1, 0, 0),
                 "reveal")
        if pocket is not None:
            for i in range(len(us) - 1):
                for j in range(len(vs) - 1):
                    quad([(us[i], vs[j], depth), (us[i + 1], vs[j], depth), (us[i + 1], vs[j + 1], depth),
                          (us[i], vs[j + 1], depth)], (0, 0, -1), "pocket")
    return np.array(verts, float), faces, np.array(normals, float), tags


def orient(V, faces, normals):
    """Faces with their winding turned (when needed) so the geometric normal agrees with the wanted one."""
    out = []
    for f, n in zip(faces, normals):
        p = V[f]
        g = np.cross(p[2] - p[0], p[3] - p[1])
        out.append(list(f) if np.dot(g, n) >= 0 else list(reversed(f)))
    return out


def wall_mesh(cfg, wall):
    """The slab of one wall in set coordinates: (V (n, 3), faces, tags). The back wall has the window opening (through
    the wall), the right wall the door pocket; the slab spans from below the floor to above the ceiling and reaches
    into the neighbouring walls (the left and right slabs by T, the back and front ones by T / 2), so a corner of the
    room is two interior planes crossing and no slab has an end face, a chamfer or a gap there."""
    fr = wall_frame(cfg, wall)
    T, H = cfg["T"], cfg["H"]
    u0, u1 = fr["u0"], fr["u1"]
    ext = T if wall in ("left", "right") else T / 2          # the back and front slabs end inside the side walls
    rect = (u0 - ext, u1 + ext, -T, H + T)
    hole = pocket = None
    if wall == "back":
        w = cfg["window"]
        hole = (w["x0"], w["x1"], w["z0"], w["z1"])
    elif wall == "right":
        d = cfg["door"]
        hole, pocket = (d["y0"], d["y1"], 0.0, d["z1"]), POCKET
    V, faces, normals, tags = slab(rect, T, (u0, u1, 0.0, H), hole, pocket)
    # frame -> world (w is measured outwards); the wanted normals map the same way
    Wd = to_world(fr, V[:, 0], V[:, 1], V[:, 2])
    nw = normals[:, 0:1] * fr["U"] + normals[:, 1:2] * np.array([0.0, 0.0, 1.0]) + normals[:, 2:3] * fr["O"]
    return Wd, orient(Wd, faces, nw), tags


# ---------------------------------------------------------------------------------------------------- the floor


def clip_rect(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman: a convex polygon [(x, y)] clipped to the rectangle."""
    def clip(pts, inside, cut):
        out = []
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            pi, qi = inside(p), inside(q)
            if pi:
                out.append(p)
            if pi != qi:
                out.append(cut(p, q))
        return out
    pts = [tuple(p) for p in poly]
    for inside, cut in (
            (lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))),
            (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1]) * (x1 - p[0]) / (q[0] - p[0]))),
            (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)),
            (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0]) * (y1 - p[1]) / (q[1] - p[1]), y1))):
        if not pts:
            break
        pts = clip(pts, inside, cut)
    return pts


def inset_polygon(poly, d):
    """A convex polygon [(x, y)] (counter-clockwise) shrunk by `d` all round, or None where it would collapse (a sliver
    narrower than 2 d). Duplicate neighbouring points are ignored."""
    pts = [tuple(p) for i, p in enumerate(poly)
           if i == 0 or np.hypot(p[0] - poly[i - 1][0], p[1] - poly[i - 1][1]) > 1e-9]
    if len(pts) > 1 and np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 1e-9:
        pts.pop()
    P = np.asarray(pts, float)
    n = len(P)
    if n < 3:
        return None
    E = np.roll(P, -1, axis=0) - P
    N = np.stack([-E[:, 1], E[:, 0]], 1) / np.linalg.norm(E, axis=1, keepdims=True)       # inward for a CCW polygon
    out = np.array([P[i] + d * (N[i - 1] + N[i]) / (1.0 + float(N[i - 1] @ N[i])) for i in range(n)])
    E2 = np.roll(out, -1, axis=0) - out
    if np.any(np.einsum("ij,ij->i", E, E2) <= 0.0):                                       # an edge turned round
        return None
    return [(float(x), float(y)) for x, y in out]


def poly_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def parquet(x0, x1, y0, y1, kind="herringbone", w=0.07, k=4, gap=0.0012, seed=1):
    """Floor planks over the rectangle: a list of {"poly": [(x, y)], "c": (x, y) centre, "a": (x, y) unit vector along
    the plank, "L": length, "w": width, "r": (r1, r2) two random numbers 0..1}, clipped to the rectangle and shrunk
    by `gap` all round (the groove).

    herringbone: planks w x (k w), square ended, in two directions at 45 degrees to the walls. In the plank frame
    (p, q) (cells of w, planks along p or q) the horizontal plank n of band b has its lower-left cell at (n + 2 k b, n)
    and the vertical plank at (n + 2 k - 1 + 2 k b, n): every cell of the plane is covered once. boards: planks
    `w` x 0.9 .. 1.6 m along y, butt joints staggered."""
    rng = np.random.default_rng(int(seed))
    planks = []

    def add(poly_world, axis, L, ww):
        poly = clip_rect(poly_world, x0, x1, y0, y1)
        if poly_area(poly) < 2.0e-5:
            return
        c = np.mean(np.asarray(poly_world), axis=0)                  # the plank's own centre, not the clipped one's
        planks.append({"poly": poly, "c": (float(c[0]), float(c[1])), "a": axis, "L": L, "w": ww,
                       "r": (float(rng.random()), float(rng.random()))})

    if kind == "boards":
        x = x0
        while x < x1 - 1e-9:
            xe = min(x + w, x1)
            y = y0 - float(rng.uniform(0.0, 1.2))
            while y < y1 - 1e-9:
                ln = float(rng.uniform(0.9, 1.6))
                a, b = max(y, y0), min(y + ln, y1)
                if b - a > 0.03:
                    g = gap
                    poly = [(x + g, a + g), (xe - g, a + g), (xe - g, b - g), (x + g, b - g)]
                    add(poly, (0.0, 1.0), ln, xe - x)
                y += ln
            x = xe
        return planks
    L = k * w
    s = math.sqrt(0.5)
    # world (x, y) = ((p - q) s, (p + q) s)  <=>  p = (x + y) s, q = (y - x) s
    corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    pp, qq = (corners[:, 0] + corners[:, 1]) * s, (corners[:, 1] - corners[:, 0]) * s
    n0, n1 = int(math.floor(qq.min() / w)) - k - 2, int(math.ceil(qq.max() / w)) + 2
    i0, i1 = int(math.floor(pp.min() / w)) - 2 * k - 2, int(math.ceil(pp.max() / w)) + 2
    for n in range(n0, n1 + 1):
        b0 = int(math.floor((i0 - n) / (2 * k))) - 1
        b1 = int(math.ceil((i1 - n) / (2 * k))) + 1
        for b in range(b0, b1 + 1):
            for horiz in (True, False):
                i = n + 2 * k * b + (0 if horiz else 2 * k - 1)
                p0, q0 = i * w + gap, n * w + gap
                dp, dq = (L, w) if horiz else (w, L)
                hp, hq = p0 + dp - 2 * gap, q0 + dq - 2 * gap
                rect = [(p0, q0), (hp, q0), (hp, hq), (p0, hq)]
                poly = [((p - q) * s, (p + q) * s) for p, q in rect]
                axis = (s, s) if horiz else (-s, s)
                add(poly, axis, L, w)
    return planks


# ---------------------------------------------------------------------------------------------------- mouldings


def signed_area(poly):
    """Signed area of a closed polygon [(a, b)] (positive when counter-clockwise)."""
    p = np.asarray(poly, float)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def fillet_polygon(poly, radii, seg=3):
    """Round the corners of a closed polygon [(a, b)]: `radii` is one value or one per vertex (0 keeps a corner sharp).
    A corner becomes an arc of seg + 1 points tangent to both edges; the radius is cut back where an edge is too short
    (each corner may use at most half of each of its edges, so neighbouring arcs never overlap)."""
    P = np.asarray(poly, float)
    n = len(P)
    r = np.broadcast_to(np.asarray(radii, float), (n,))
    out = []
    for i in range(n):
        c, p, q = P[i], P[i - 1], P[(i + 1) % n]
        u, v = p - c, q - c
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        theta = math.acos(float(np.clip((u @ v) / max(lu * lv, 1e-18), -1.0, 1.0))) if lu * lv > 0 else math.pi
        if r[i] <= 0.0 or theta > math.pi - 1e-6 or theta < 1e-6:
            out.append((float(c[0]), float(c[1])))
            continue
        u, v = u / lu, v / lv
        t = min(r[i] / math.tan(theta / 2), 0.5 * lu, 0.5 * lv)       # distance from the corner to the tangent points
        rr = t * math.tan(theta / 2)
        bis = (u + v) / np.linalg.norm(u + v)
        centre = c + bis * (rr / math.sin(theta / 2))
        s0, s1 = c + u * t - centre, c + v * t - centre
        a0, a1 = math.atan2(s0[1], s0[0]), math.atan2(s1[1], s1[0])
        da = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi            # the short way round
        for k in range(seg + 1):
            a = a0 + da * k / seg
            out.append((float(centre[0] + rr * math.cos(a)), float(centre[1] + rr * math.sin(a))))
    return out


def wall_runs(cfg, gap=None):
    """Polylines [(x, y)] along the interior planes of the walls that are not open, counter-clockwise round the room
    (the room on their left), each with whether it is a closed loop: [(points, closed)]. `gap` = (y0, y1) leaves that
    stretch of the right wall out (the door). Consecutive walls form one run, so a sweep along it is mitred at the
    corners."""
    x0, x1, y0, y1 = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"]
    back, left, front = ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)), ((x0, y0), (x1, y0))
    if gap is None:
        order = [("right", ((x1, y0), (x1, y1))), ("back", back), ("left", left), ("front", front)]
    else:                                    # the loop is cut at the door: it starts after the gap and ends before it
        order = [("right", ((x1, gap[1]), (x1, y1))), ("back", back), ("left", left), ("front", front),
                 ("right", ((x1, y0), (x1, gap[0])))]
    runs = []
    for wall, (a, b) in order:
        if wall in cfg["open"]:
            continue
        if runs and runs[-1][-1] == a:
            runs[-1].append(b)
        else:
            runs.append([a, b])
    if gap is None and len(runs) > 1 and runs[-1][-1] == runs[0][0]:     # a run that wraps round the start
        runs[0] = runs.pop()[:-1] + runs[0]
    if gap is None and len(runs) == 1 and runs[0][0] == runs[0][-1]:
        return [(runs[0][:-1], True)]
    return [(r, False) for r in runs]


def sweep(points, profile, closed=False):
    """Sweep a profile along a horizontal polyline. `points` [(x, y)] run counter-clockwise round the room (the room on
    their left); `profile` [(d, z)] is a closed polygon in (distance from the path towards the room, height), either
    way round. Every vertex of the polyline gets a ring of the profile in its mitre plane, so runs join at corners
    without seams; an open run is closed with a cap at each end. Returns (V (n, 3), faces [[vertex index]], rings
    (number of rings)) with every face wound counter-clockwise seen from outside (positive volume)."""
    P = np.asarray(points, float)
    m = len(P)
    prof = [tuple(q) for q in profile]
    if signed_area(prof) < 0:
        prof = prof[::-1]
    k = len(prof)
    seg = P[(np.arange(m) + 1) % m] - P if closed else P[1:] - P[:-1]
    t = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)                        # the left normal of every segment
    ring_dir = []
    for i in range(m):
        if closed:
            a, b = nrm[i - 1], nrm[i]
        elif i == 0:
            a = b = nrm[0]
        elif i == m - 1:
            a = b = nrm[-1]
        else:
            a, b = nrm[i - 1], nrm[i]
        ring_dir.append((a + b) / (1.0 + float(a @ b)))           # the mitre: d along each wall's normal
    V = np.array([(P[i][0] + ring_dir[i][0] * d, P[i][1] + ring_dir[i][1] * d, z)
                  for i in range(m) for d, z in prof], float)
    faces = []
    spans = m if closed else m - 1
    for i in range(spans):
        i2 = (i + 1) % m
        for j in range(k):
            j2 = (j + 1) % k
            faces.append([i * k + j, i * k + j2, i2 * k + j2, i2 * k + j])
    if not closed:
        faces.append(list(range(k))[::-1])
        faces.append([(m - 1) * k + j for j in range(k)])
    return V, faces, m


# profiles (distance from the wall plane towards the room, height); the first 4 mm of the depth is buried in the wall
BURY = 0.004
SKIRT_PROFILE = [(-BURY, 0.0), (0.016, 0.0), (0.016, 0.098), (0.0185, 0.103), (0.0215, 0.111), (0.0215, 0.123),
                 (0.017, 0.134), (-BURY, 0.134)]
SKIRT_FILLET = [0.0015, 0.0015, 0.0025, 0.004, 0.004, 0.0055, 0.0045, 0.0015]
RAIL_PROFILE = [(-BURY, -0.035), (0.016, -0.035), (0.016, -0.012), (0.0215, -0.005), (0.028, 0.004), (0.028, 0.022),
                (0.022, 0.030), (-BURY, 0.030)]
RAIL_FILLET = [0.0015, 0.003, 0.003, 0.004, 0.004, 0.004, 0.006, 0.0015]
HEAD_BACK, HEAD_FRONT = 0.020, 0.058    # the headrail's section spans these distances from the wall plane (d)


def head_profile(seg=10):
    """The headrail's section in (d, z), z from its bottom: a bullnose, flat at the back, top and bottom, rolled in
    front over a half ellipse (semi-axes 0.024 x 0.021) so that no flat face looks into the room."""
    h = HEAD_H / 2.0
    ad = HEAD_FRONT - HEAD_BACK - 0.014               # the ellipse's depth: it starts 14 mm from the back
    c = HEAD_FRONT - ad
    pts = [(HEAD_BACK, 0.0), (c, 0.0)]
    for k in range(1, seg):
        a = -math.pi / 2 + math.pi * k / seg
        pts.append((c + ad * math.cos(a), h + h * math.sin(a)))
    pts += [(c, HEAD_H), (HEAD_BACK, HEAD_H)]
    return pts


HEAD_PROFILE = head_profile()
HEAD_FILLET = [0.002, 0.0, *([0.0] * 9), 0.0, 0.002]     # the back corners only; the rest is already round


def moulding(profile, fillet, seg=3):
    """A moulding's section: the profile with its corners rounded."""
    return fillet_polygon(profile, fillet, seg)


def sill_profile():
    """The window sill board's section in (d, z): d = -w, so the nosing reaches 0.06 m into the room, the back 0.15 m
    behind the wall face; 35 mm thick with a rounded nosing."""
    return fillet_polygon([(-FRAME_W1, -0.035), (SILL_OUT, -0.035), (SILL_OUT, 0.0), (-FRAME_W1, 0.0)],
                          [0.003, 0.005, 0.014, 0.003], 4)


# ---------------------------------------------------------------------------------------------------- the blind


def blind_layout(cfg):
    """Constants of the venetian blind (set frame; z values relative to the bottom of the headrail `top`)."""
    w = cfg["window"]
    top = w["z1"] + HEAD_OVER
    n = max(8, int(round((top - w["z0"]) / 0.0225)))
    rail_t = 0.012
    lowered_rail = w["z0"] + rail_t + 0.002                   # top of the bottom rail resting on the sill
    first = 0.012                                             # the first slat hangs this far below the headrail
    pitch = (top - first - PITCH_MIN - lowered_rail) / (n - 1)
    return {"n": n, "top": top, "pitch": pitch, "pitch_min": PITCH_MIN, "first": first, "rail_t": rail_t,
            "width": w["width"] + BLIND_OVER, "x": w["x"], "chord": SLAT_C}


def blind_levels(lay, raise_):
    """z (relative to the bottom of the headrail, <= 0) of every slat from the top (i = 0) and of the top of the bottom
    rail, for the blind raised by `raise_` (0 lowered .. 1 stacked at the top). The slats hang from the ladder at the
    full pitch; the bottom rail rises and the slats pile on it at the minimum pitch, so slat i sits at the higher of
    its hanging place and its place in the pile: z_i = max(hang_i, rail + (n - i) pitch_min)."""
    n, pf, pm, first = lay["n"], lay["pitch"], lay["pitch_min"], lay["first"]
    i = np.arange(n, dtype=float)
    hang = -first - i * pf
    z_low = -first - (n - 1) * pf - pm                         # rail when lowered: the last slat just rests on it
    z_high = -first - n * pm                                   # rail when raised: the whole pile hangs from the head
    rail = z_low + float(raise_) * (z_high - z_low)
    return np.maximum(hang, rail + (n - i) * pm), rail


def cord_length(raise_):
    """Hanging length of the lift cord below the headrail: the cord is pulled out as the blind goes up."""
    return 0.45 + 0.7 * float(raise_)


# ---------------------------------------------------------------------------------------------------- the card


def _round(v, n=4):
    """Floats rounded to `n` places, through nested dicts and lists."""
    if isinstance(v, float):
        return round(v, n) + 0.0
    if isinstance(v, dict):
        return {k: _round(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x, n) for x in v]
    return v


def card(cfg, lights, params, city_point):
    """The set's card (docs: sets/__init__ and the builder's docstring)."""
    x0, x1, y0, y1, H = cfg["x0"], cfg["x1"], cfg["y0"], cfg["y1"], cfg["H"]
    W, D = x1 - x0, y1 - y0
    w, d = cfg["window"], cfg["door"]
    zc = H / 2
    glass_y = y1 + (FRAME_W0 + FRAME_W1) / 2
    lay = blind_layout(cfg)
    head_top = lay["top"] + HEAD_H + 0.003
    up = [0.0, 0.0, 1.0]
    surfaces = [
        {"name": "wall_back", "center": [0.0, y1, zc], "normal": [0.0, -1.0, 0.0], "up": up, "size": [W, H]},
        {"name": "wall_left", "center": [x0, 0.0, zc], "normal": [1.0, 0.0, 0.0], "up": up, "size": [D, H]},
        {"name": "wall_right", "center": [x1, 0.0, zc], "normal": [-1.0, 0.0, 0.0], "up": up, "size": [D, H]},
        {"name": "wall_front", "center": [0.0, y0, zc], "normal": [0.0, 1.0, 0.0], "up": up, "size": [W, H]},
        {"name": "window", "center": [w["x"], glass_y, (w["z0"] + w["z1"]) / 2], "normal": [0.0, -1.0, 0.0],
         "up": up, "size": [w["width"], w["height"]]},
    ]
    wc = CASING_W
    obstacles = [
        {"name": "window", "min": [w["x0"] - wc, y1 - 0.10, w["z0"] - 0.06], "max": [w["x1"] + wc, y1, head_top]},
        {"name": "door", "min": [x1 - 0.10, d["y0"] - wc, 0.0], "max": [x1, d["y1"] + wc, d["z1"] + wc]},
    ]
    return _round({
        "kind": "bedroom_80s", "paths": {},
        "use": {"rest": [{"name": "floor", "type": "plane", "center": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                          "size": [W, D]}],
                "surface": surfaces,
                "look": [{"name": "window", "point": [w["x"], y1, (w["z0"] + w["z1"]) / 2]},
                         {"name": "city", "point": [float(v) for v in city_point]},
                         {"name": "desk_zone", "point": [w["x"], y1 - 0.6, 0.9]},
                         {"name": "bed_zone", "point": [x0 + 0.5, y1 - 1.0, 0.6]},
                         {"name": "room", "point": [0.0, 0.0, 1.2]}],
                "obstacles": obstacles},
        "obstacles": obstacles,                          # also at the top level, where the placement stage looks
        "colliders": [{"type": "floor", "z": 0.0, "tag": cfg["name"]}],
        "lights": list(lights),
        "room": {"min": [x0, y0, 0.0], "max": [x1, y1, H]},
        "params": dict(params),
    })
