"""Pure geometry of the bedroom decor props (no bpy: importable from tests).

Poster frame: origin at the sheet's centre on the wall plane (the wall is y = 0), +Z up, +X to the right as seen from
the room, the sheet faces -Y. The sheet's surface is `y = -sheet_gap(x, z)`: the gap to the wall is `GAP_PIN` at a
push-pin (the paper lies nearly on the wall there) and grows with the bow of the paper between the fixings, so that the
sheet's mean plane stands about 3 mm off the wall."""
import math
import random

GAP_PIN = 0.0012                 # sheet to wall at a fixing (m)
PAPER_T = 0.0002                 # paper thickness (m)


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def bow_amp(w, h):
    """Amplitude of the paper's bow (m): 4.2 mm on a 0.5 m sheet, between 1.5 and 7 mm."""
    return min(0.007, max(0.0015, 0.0042 * min(w, h) / 0.5))


def pin_inset(w, h):
    return min(0.022, 0.06 * min(w, h))


def bow(x, z, w, h, mount="pins"):
    """Paper bow toward the viewer (m, never negative: the paper never goes into the wall) at the sheet point (x, z).
    `pins`: four corner pins, the bow vanishes at the pins; `tape`: held along its top edge, the bottom edge curls out;
    `none`: a free sheet."""
    A = bow_amp(w, h)
    if mount == "pins":
        ins = pin_inset(w, h)
        u = max(-1.0, min(1.0, x / (w / 2 - ins)))
        v = max(-1.0, min(1.0, z / (h / 2 - ins)))
        return A * ((0.8 + 0.3 * u - 0.2 * v) * (1 - u * u) * (1 - v * v)
                    + 0.25 * (u * u * (1 - v * v) + v * v * (1 - u * u)))
    u = max(-1.0, min(1.0, x / (w / 2)))
    v = max(-1.0, min(1.0, z / (h / 2)))
    if mount == "tape":
        k = (1.0 - v) / 2.0
        return A * k ** 1.6 * (0.9 + 0.5 * (1 - u * u) + 0.3 * u)
    return A * ((0.5 + 0.2 * u - 0.15 * v) * (1 - u * u) * (1 - v * v) + 0.2 * (u * u * (1 - v * v) + v * v * (1 - u * u)))


def sheet_gap(x, z, w, h, mount="pins"):
    """Distance (m) of the paper's front face from the wall at (x, z)."""
    return GAP_PIN + bow(x, z, w, h, mount)


def sheet_grid(w, h, step=0.02, cap=72):
    """Quads along x and z of the sheet mesh."""
    return max(8, min(cap, round(w / step))), max(8, min(cap, round(h / step)))


def corner_points(w, h, mount="pins"):
    """The sheet's corners as seen from the room (TL, TR, BR, BL) on its front face."""
    hw, hh = w / 2, h / 2
    return {tag: (sx * hw, -sheet_gap(sx * hw, sz * hh, w, h, mount), sz * hh)
            for tag, sx, sz in (("TL", -1, 1), ("TR", 1, 1), ("BR", 1, -1), ("BL", -1, -1))}


def pin_points(w, h, torn=(), mount="pins"):
    """Push-pin positions (x, z) at the four corners, inset; none at a torn corner."""
    ins = pin_inset(w, h)
    hw, hh = w / 2 - ins, h / 2 - ins
    return [(sx * hw, sz * hh) for tag, sx, sz in (("TL", -1, 1), ("TR", 1, 1), ("BR", 1, -1), ("BL", -1, -1))
            if tag not in torn]


def tape_specs(w, h, name_seed, torn=()):
    """Two strips of tape across the top edge near the top corners: [(x of the crossing, angle in degrees)]."""
    rng = random.Random(name_seed)
    ins = min(0.05, 0.12 * min(w, h))
    out = []
    for tag, sx in (("TL", -1), ("TR", 1)):
        if tag in torn:
            continue
        out.append((sx * (w / 2 - ins) + rng.uniform(-0.004, 0.004), sx * rng.uniform(6.0, 13.0)))
    return out


def tape_strip(x0, ang_deg, w, h, mount, width=0.017, lp=0.014, lw=0.032, nc=8, nr=24):
    """A strip of tape across the sheet's top edge at x0, turned `ang_deg` (anticlockwise as seen from the room) about
    the crossing point: `lp` over the paper, `lw` out over the wall, `width` wide. It follows the bowed paper (0.4 mm in
    front) and ramps to the wall (0.5 mm in front of it) over 7 mm beyond the edge. Returns (verts, quads, uvs):
    verts [(x, y, z)] relative to the crossing point's (x0, -gap, h/2), quads [(i, j, k, l)], uvs per vert (u across the
    width, v along the length, metres)."""
    th = math.radians(ang_deg)
    e = (math.cos(th), math.sin(th))
    d = (-math.sin(th), math.cos(th))
    z_edge = h / 2
    org = (x0, -sheet_gap(x0, z_edge, w, h, mount) - 0.0004, z_edge)
    verts, uvs, idx = [], [], {}
    for i in range(nc + 1):
        a = -width / 2 + width * i / nc
        for j in range(nr + 1):
            s = -lp + (lp + lw) * j / nr
            x = x0 + a * e[0] + s * d[0]
            z = z_edge + a * e[1] + s * d[1]
            sd = max(abs(x) - w / 2, abs(z) - h / 2)                   # distance outside the sheet (box approx.)
            yp = -sheet_gap(x, z, w, h, mount) - 0.0004
            y = yp + (-0.0005 - yp) * smooth(sd / 0.007)
            idx[(i, j)] = len(verts)
            verts.append((x - org[0], y - org[1], z - org[2]))
            uvs.append((a + width / 2, s + lp))
    quads = [(idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]) for i in range(nc) for j in range(nr)]
    return verts, quads, uvs, org


# ================================================================= alarm clock
# Frame: origin on the table under the centre of the footprint, +Z up, front (the display) toward -Y.
CLOCK_W, CLOCK_D = 0.16, 0.07                       # footprint (X, Y)
CLOCK_PLINTH = (0.002, 0.014)                       # dark base: z range (feet below it)
CLOCK_FRONT_BOTTOM, CLOCK_FRONT_TOP = (-0.0325, 0.014), (-0.019, 0.063)      # (y, z) of the sloped front face
CLOCK_BACK_TOP = (0.035, 0.068)
PLATE_W, PLATE_H = 0.132, 0.042                     # the display plate on the face
DIGIT_W, DIGIT_H, SEG_T, SEG_GAP, SLANT = 0.0205, 0.030, 0.0036, 0.0008, 6.0

DIGIT_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
                  "7": "abc", "8": "abcdefg", "9": "abcdfg", " ": ""}


def parse_time(text):
    """'02:47' / '2:47' -> four display characters (a one-digit hour leaves the first digit blank). ValueError if the
    text is not a 12/24-hour clock time."""
    s = str(text).strip()
    hh, sep, mm = s.partition(":")
    if sep != ":" or not hh.isdigit() or not mm.isdigit() or len(hh) not in (1, 2) or len(mm) != 2:
        raise ValueError(f"time {text!r}: expected H:MM or HH:MM")
    if int(hh) > 23 or int(mm) > 59:
        raise ValueError(f"time {text!r} is not a time of day")
    return tuple(hh.rjust(2) + mm)


def face_frame():
    """The clock's sloped front face: (tilt from vertical in radians, centre (y, z), length along the face, out normal
    (nx, ny, nz), up vector along the face (ux, uy, uz))."""
    (y0, z0), (y1, z1) = CLOCK_FRONT_BOTTOM, CLOCK_FRONT_TOP
    phi = math.atan2(y1 - y0, z1 - z0)
    return (phi, ((y0 + y1) / 2, (z0 + z1) / 2), math.hypot(y1 - y0, z1 - z0),
            (0.0, -math.cos(phi), math.sin(phi)), (0.0, math.sin(phi), math.cos(phi)))


def top_z(y):
    """Height of the clock's top surface at depth y (it slopes up toward the back)."""
    (y0, z0), (y1, z1) = CLOCK_FRONT_TOP, CLOCK_BACK_TOP
    return z0 + (z1 - z0) * (y - y0) / (y1 - y0)


def display_cells(gap=0.0045, colon_w=0.010, colon_gap=0.004):
    """x of the four digit centres and of the colon centre on the display plate: [d1, d2, colon, d3, d4] centred on
    the plate (a layout of two digit pairs around the colon)."""
    total = 4 * DIGIT_W + 2 * gap + colon_w + 2 * colon_gap
    x = -total / 2
    d1 = x + DIGIT_W / 2
    d2 = d1 + DIGIT_W + gap
    colon = d2 + DIGIT_W / 2 + colon_gap + colon_w / 2
    d3 = colon + colon_w / 2 + colon_gap + DIGIT_W / 2
    d4 = d3 + DIGIT_W + gap
    return {"digits": (d1, d2, d3, d4), "colon": colon, "width": total}


def _hex_bar(cx, cy, length, t, vertical):
    """Elongated hexagon (pointed ends) centred at (cx, cy), `length` tip to tip, `t` thick, counter-clockwise."""
    a, b = length / 2, t / 2
    if vertical:
        pts = [(0.0, -a), (b, -a + b), (b, a - b), (0.0, a), (-b, a - b), (-b, -a + b)]
    else:
        pts = [(-a, 0.0), (-a + b, -b), (a - b, -b), (a, 0.0), (a - b, b), (-a + b, b)]
    return [(cx + x, cy + y) for x, y in pts]


def segment_polygons(w=DIGIT_W, h=DIGIT_H, t=SEG_T, gap=SEG_GAP, slant=SLANT):
    """The seven segments {a..g: [(x, y)] hexagon} of one digit centred at the origin of its cell, sheared by `slant`
    degrees (italic). a top, b top right, c bottom right, d bottom, e bottom left, f top left, g middle."""
    lh, lv = w - t - gap, h / 2 - t - gap
    raw = {"a": (0.0, h / 2 - t / 2, lh, False), "g": (0.0, 0.0, lh, False), "d": (0.0, -h / 2 + t / 2, lh, False),
           "f": (-w / 2 + t / 2, h / 4, lv + t * 0.5, True), "b": (w / 2 - t / 2, h / 4, lv + t * 0.5, True),
           "e": (-w / 2 + t / 2, -h / 4, lv + t * 0.5, True), "c": (w / 2 - t / 2, -h / 4, lv + t * 0.5, True)}
    k = math.tan(math.radians(slant))
    return {s: [(x + y * k, y) for x, y in _hex_bar(cx, cy, ln, t, v)] for s, (cx, cy, ln, v) in raw.items()}


def poly_area(pts):
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


# ================================================================= nightstand
# Frame: origin on the floor under the centre, +Z up, the drawer faces -Y.
NS_W, NS_D, NS_H = 0.40, 0.35, 0.55                 # top slab footprint and height
NS_TOP_T = 0.025
NS_BODY_Z = (0.405, 0.525)                          # the drawer unit
NS_BODY_W, NS_BODY_D = 0.372, 0.322
NS_SHELF_Z, NS_SHELF_T = 0.20, 0.018                # lower shelf board centre height and thickness
LEG_R = 0.0105


def leg_points(sx, sy):
    """Axis of the leg at the corner (sx, sy = +-1): from inside the drawer unit down to the floor, splayed outward."""
    return (sx * 0.165, sy * 0.140, 0.43), (sx * 0.178, sy * 0.152, 0.0125)


def leg_at(sx, sy, z):
    """(x, y) of that leg's axis at height z."""
    (x0, y0, z0), (x1, y1, z1) = leg_points(sx, sy)
    t = (z0 - z) / (z0 - z1)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


# ================================================================= wall shelf
# Frame: origin at the centre of the board's top-back edge on the wall plane (y = 0), +Z up, the shelf sticks out toward
# -Y; the board's top is z = 0.
SH_W, SH_D, SH_T = 0.80, 0.20, 0.030
BOOK_SLOTS = ("iris", "love", "gold", "pine", "foam", "rose", "overlay", "muted", "hl_high")
TAPE_SLOTS = ("love", "foam", "gold", "iris")
TAPE_W, TAPE_D, TAPE_H = 0.110, 0.070, 0.015
ORN_X = 0.345


def shelf_layout(seed=0):
    """Deterministic contents of the shelf: a row of books at the left end (the last one leaning on its neighbour), a
    stack of cassettes, a small totem at the right end and the free stretch of board between. Returns
    {books: [{x, w, h, d, slot, lean_deg, bands}], tapes: [{x, y, yaw, slot}], ornament_x, free: (x0, x1), books_x: (x0, x1),
    tapes_x: (x0, x1)}; x of a book is its left side, books stand on z = 0 with their spines toward -Y."""
    rng = random.Random(seed * 7919 + 11)
    books, x = [], -SH_W / 2 + 0.030
    for i in range(9):
        w = rng.uniform(0.016, 0.038)
        h = rng.uniform(0.17, 0.27) if i < 8 else rng.uniform(0.15, 0.19)
        books.append({"x": x, "w": w, "h": h, "d": rng.uniform(0.130, 0.165), "slot": BOOK_SLOTS[(i * 4 + rng.randrange(3)) % len(BOOK_SLOTS)],
                      "lean_deg": 0.0, "bands": rng.choice((1, 2, 2, 3))})
        x += w + 0.0012
    last = books[-1]
    last["x"] += 0.040                                           # leave a gap at the foot, the top rests on the neighbour
    last["lean_deg"] = math.degrees(math.atan2(0.040, 0.92 * last["h"]))
    books_x = (books[0]["x"], max(b["x"] + b["w"] for b in books))
    tx = books_x[1] + 0.075
    tapes = [{"x": tx + rng.uniform(-0.004, 0.004), "y": -0.105 + rng.uniform(-0.006, 0.006), "yaw": rng.uniform(-4.0, 4.0),
              "slot": TAPE_SLOTS[i % len(TAPE_SLOTS)]} for i in range(4)]
    tapes_x = (tx - TAPE_W / 2 - 0.004, tx + TAPE_W / 2 + 0.004)
    free = (tapes_x[1] + 0.025, ORN_X - 0.045)
    return {"books": books, "tapes": tapes, "ornament_x": ORN_X, "free": free, "books_x": books_x, "tapes_x": tapes_x}


# ================================================================= rounded outlines, books and cassettes
CASSETTE = (0.1004, 0.0635, 0.0125)                 # a compact cassette lying flat: x, y (spine at -y), z (as cassette_tape)
CASSETTE_CORNER = 0.0028                            # plan corner radius of its shell
BOOK_COVER_T = 0.0022                               # thickness of a book's cover boards
BOOK_OVER = 0.0025                                  # the covers overhang the page block by this much (head, tail, fore-edge)


def rounded_poly(pts, radii, n=6):
    """Polygon `pts` [(u, v)] with corner i rounded by radii[i] (a float for all; 0 keeps a corner sharp). A radius that
    does not fit between its neighbours' tangent points is reduced. Returns the outline, n segments per arc."""
    out, m = [], len(pts)
    for i, (px, py) in enumerate(pts):
        r = radii[i] if hasattr(radii, "__len__") else radii
        ax, ay = pts[i - 1]
        bx, by = pts[(i + 1) % m]
        ux, uy, vx, vy = ax - px, ay - py, bx - px, by - py
        lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
        ux, uy, vx, vy = ux / lu, uy / lu, vx / lv, vy / lv
        ang = math.acos(max(-1.0, min(1.0, ux * vx + uy * vy)))             # interior angle at the corner
        if r <= 0.0 or ang < 1e-6 or abs(ang - math.pi) < 1e-6:
            out.append((px, py))
            continue
        t = min(r / math.tan(ang / 2), 0.45 * min(lu, lv))
        r = t * math.tan(ang / 2)
        bxs, bys = ux + vx, uy + vy
        lb = math.hypot(bxs, bys)
        cx, cy = px + bxs / lb * r / math.sin(ang / 2), py + bys / lb * r / math.sin(ang / 2)
        a1 = math.atan2(py + uy * t - cy, px + ux * t - cx)
        a2 = math.atan2(py + vy * t - cy, px + vx * t - cx)
        da = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi                # the short way round
        out += [(cx + r * math.cos(a1 + da * k / n), cy + r * math.sin(a1 + da * k / n)) for k in range(n + 1)]
    return out


def rounded_rect(hw, hh, r, n=5):
    """Counter-clockwise outline of a rectangle (half sizes hw, hh) with corner radius r."""
    return rounded_poly([(hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)], r, n)


def book_sag(w):
    """How far the rounded spine bulges beyond the ends of the covers (m)."""
    return min(0.0065, 0.24 * w)


def _spine_circle(w, d):
    s = book_sag(w)
    R = (w * w / 4.0 + s * s) / (2.0 * s)
    return R, w / 2.0, -d + R                                                # radius, centre x, centre y


def spine_point(w, d, x, off=0.0):
    """Point of a book's spine curve at x across its thickness (0..w), pushed outward by `off`: (x, y). The book stands
    with its fore-edge at y = 0 and the spine toward -y (the outermost point is y = -d)."""
    R, cx, cy = _spine_circle(w, d)
    dx = x - cx
    s = math.sqrt(max(R * R - dx * dx, 0.0))
    return x + dx / R * off, cy - s - s / R * off


def book_case_outline(w, d, tc=BOOK_COVER_T, n=10):
    """Counter-clockwise top view of the book's case (front board, round spine, back board): a U-shaped outline with
    open fore-edge; boards `tc` thick, the spine a circular arc."""
    R, cx, cy = _spine_circle(w, d)
    outer = [spine_point(w, d, w * k / n) for k in range(n + 1)]
    R2 = R - tc
    inner = []
    for k in range(n + 1):
        x = (w - tc) + (2.0 * tc - w) * k / n
        inner.append((x, cy - math.sqrt(max(R2 * R2 - (x - cx) ** 2, 0.0))))
    return [(0.0, 0.0)] + outer + [(w, 0.0), (w - tc, 0.0)] + inner + [(tc, 0.0)]


def book_block_outline(w, d, tc=BOOK_COVER_T, over=BOOK_OVER, gap=0.0003, n=8):
    """Counter-clockwise top view of the page block inside the case: a rounded back that follows the spine, a fore-edge
    inset by `over` that bows out 0.8 mm in the middle."""
    R, cx, cy = _spine_circle(w, d)
    R3 = R - tc - gap
    xl, xr = tc + gap, w - tc - gap
    arc = []
    for k in range(n + 1):
        x = xl + (xr - xl) * k / n
        arc.append((x, cy - math.sqrt(max(R3 * R3 - (x - cx) ** 2, 0.0))))
    half = (xr - xl) / 2.0
    bow = [(xr + (xl - xr) * k / n, -over + 0.0008 * (1.0 - ((xr + (xl - xr) * k / n - cx) / half) ** 2))
           for k in range(n + 1)]
    return [(xl, -over)] + arc + bow[:-1]


# ================================================================= clock case profile
CLOCK_SHELL_RADII = (0.003, 0.009, 0.011, 0.0055)   # fillets of the case's side profile: front-bottom, back-bottom, back-top, front-top
CLOCK_END_R = 0.010                                 # the case's ends are rolled over by this much


def clock_profile(seg=8):
    """The case's side profile [(y, z)], counter-clockwise, with its four fillets."""
    f0, f1, b1 = CLOCK_FRONT_BOTTOM, CLOCK_FRONT_TOP, CLOCK_BACK_TOP
    return rounded_poly([f0, (b1[0], f0[1]), b1, f1], CLOCK_SHELL_RADII, seg)
