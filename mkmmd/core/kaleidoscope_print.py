"""The printed paper of a cardboard kaleidoscope toy (library:kaleidoscope): the gift-wrap print round the tube, where it
has rubbed off, and the view through the eyepiece, as images.

numpy only (it runs inside Blender 4.2's Python: numpy 1.24, no PIL, cv2 or scipy); pure functions, no file I/O.

The paper (`paper_print`, `paper_wear`) is laid out in millimetres, so a layout is the same at every resolution: columns x
run round the tube (size_mm[0], the circumference; x = 0 and x = W meet at the seam, where the print is simply cut), rows y
along it from the eyepiece end (row 0) to the far end. It is printed like offset gift wrap: each ink a plate of coverage
0..1, every motif knocked out of the plates under it (with a hair of trap), the inks multiplying over the paper's white (a
dark ground is a flood with the motifs knocked out of it, so a dark palette still reads), one plate off register by
MISREGISTER mm, the ink film's density drifting over the sheet and the paper's tooth starving it in specks.

Patterns (`PATTERNS`; their colour roles and default palette slots in `ROLES`, ground first):
- roses: cabbage roses (rings of cupped petals round a spiral heart; the main ink, highlights in the petal ink, depth in
  the main ink printed twice) with leaves, three- and four-leaf clovers on stems and tiny accents (dots, sparkles,
  sprigs), scattered all over.
- memphis: 80s Memphis: squiggles, zigzags, confetti triangles, dots, rings, half discs, grid and dot patches, some
  shapes with an offset outline.
- marbled: combed marbling: veins and combed stripes of the inks over the ground, domain-warped into flowing waves.

`paper_wear` is where the print has rubbed off (0 intact .. 1 bare paper): the turned rims at both ends, the seam and a
few scuffs and scratches. `view` is what the eyepiece shows: beads and glass bits folded by three mirrors (a 30-60-90
triangle) into a six-fold star, inside a circle with a soft dark edge."""
import math

import numpy as np

from . import palette as PAL

PATTERNS = ("roses", "memphis", "marbled")
ROLES = {                                    # pattern -> {role: default palette slot}, the ground first
    "roses": {"ground": "base", "rose": "love", "petal": "rose", "leaf": "pine", "clover": "foam", "accent": "gold"},
    "memphis": {"ground": "base", "line": "text", "love": "love", "gold": "gold", "pine": "pine", "iris": "iris"},
    "marbled": {"ground": "overlay", "love": "love", "rose": "rose", "iris": "iris", "pine": "pine", "gold": "gold"},
}
SIZE_MM = (154.0, 204.0)                     # default sheet: round the tube x along it, millimetres
PX_PER_MM = 10.0
TRAP = 0.05                                  # mm a motif's ink spreads under the plates it knocks out
MISREGISTER = 0.15                           # mm one plate is off register
PAPER = (1.0, 0.993, 0.976)                  # the paper stock under the inks (sRGB)
LIGHT = math.radians(-125.0)                 # the drawn light comes from the top left (image rows run down)
F32 = np.float32
TAU = 2.0 * math.pi


# ===================================================================================================================
# colours
# ===================================================================================================================
def _roles(pattern):
    if pattern not in ROLES:
        raise ValueError(f"kaleidoscope print: no pattern {pattern!r} (have {list(PATTERNS)})")
    return ROLES[pattern]


def _hex_rgb(hexstr, what):
    h = str(hexstr).strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"kaleidoscope print: {what} = {hexstr!r} is not a '#rrggbb' colour")
    return PAL.srgb("#" + h)


def colors(pattern, palette, overrides=None):
    """{role: (r, g, b)} sRGB floats 0..1 of `pattern`'s colour roles (ground first): each role's default slot of
    `palette` ({slot: "#rrggbb"}), or what `overrides` ({role: slot name or "#rrggbb"}) gives it. An unknown pattern,
    role or slot is a ValueError."""
    roles = _roles(pattern)
    over = dict(overrides or {})
    bad = sorted(set(over) - set(roles))
    if bad:
        raise ValueError(f"kaleidoscope print: {pattern} has no colour role {bad} (roles {list(roles)})")
    out = {}
    for role, slot in roles.items():
        spec = str(over.get(role, slot)).strip()
        if spec.startswith("#"):
            out[role] = _hex_rgb(spec, role)
        elif spec in palette:
            out[role] = _hex_rgb(palette[spec], f"{role} (slot {spec})")
        else:
            raise ValueError(f"kaleidoscope print: {role} = {spec!r} is neither a palette slot ({sorted(palette)}) "
                             f"nor a '#rrggbb' colour")
    return out


def _inks(pattern, cols):
    """The role colours `paper_print` was given, checked: {role: (r, g, b)}."""
    roles = _roles(pattern)
    missing = [r for r in roles if r not in cols]
    if missing:
        raise ValueError(f"kaleidoscope print: {pattern} needs colours for {missing} (roles {list(roles)})")
    out = {}
    for r in roles:
        c = tuple(float(v) for v in cols[r])
        if len(c) != 3:
            raise ValueError(f"kaleidoscope print: colour {r} = {cols[r]!r} is not (r, g, b)")
        out[r] = tuple(min(max(v, 0.0), 1.0) for v in c)
    return out


def _lum(c):
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


# ===================================================================================================================
# the sheet: sizes, random streams, noise, plates
# ===================================================================================================================
def _sheet_size(size_mm, px_per_mm):
    """(w_mm, h_mm, ppm, W, H) of a sheet, checked."""
    try:
        w_mm, h_mm = (float(v) for v in size_mm)
        ppm = float(px_per_mm)
    except (TypeError, ValueError):
        raise ValueError(f"kaleidoscope print: size_mm {size_mm!r} / px_per_mm {px_per_mm!r} are not numbers") from None
    W, H = int(round(w_mm * ppm)), int(round(h_mm * ppm))
    if not (w_mm > 0 and h_mm > 0 and ppm > 0) or W < 1 or H < 1:
        raise ValueError(f"kaleidoscope print: a sheet of {size_mm!r} mm at {px_per_mm!r} px/mm has no pixels")
    return w_mm, h_mm, ppm, W, H


def _rng(seed, *key):
    """A random stream of its own for every purpose (`key`), so that what one draws never shifts another."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF] + [int(k) for k in key])


def _knots(n, ppm, cell, m):
    """(n, m) weights that interpolate knots spaced `cell` mm (smoothstep) at the centres of n pixels."""
    u = (np.arange(n, dtype=np.float64) + 0.5) / (ppm * cell)
    i = np.floor(u).astype(np.intp)
    f = u - i
    f = f * f * (3.0 - 2.0 * f)
    w = np.zeros((n, m), F32)
    rows = np.arange(n)
    w[rows, i] = 1.0 - f
    w[rows, i + 1] = f
    return w


def _smooth(rng, h_mm, w_mm, ppm, H, W, cell, cell_x=None):
    """(H, W) float32 value noise in -1..1 with knots every `cell` mm down and `cell_x` mm across (default `cell`):
    the same field at every resolution (knots in millimetres; two matrix products)."""
    cx = cell if cell_x is None else cell_x
    gh, gw = int(h_mm / cell) + 3, int(w_mm / cx) + 3
    g = rng.uniform(-1.0, 1.0, (gh, gw)).astype(F32)
    return (_knots(H, ppm, cell, gh) @ g) @ _knots(W, ppm, cx, gw).T


def _edge(d):
    """Coverage 0..1 from a signed distance in pixels (positive inside): a one-pixel anti-aliased edge."""
    return np.clip(d + F32(0.5), F32(0.0), F32(1.0))


class _Sheet:
    """The ink plates of one print (float32 (H, W) coverage each), drawn motif by motif in painter's order; the first
    plate is the ground, a flood every motif is knocked out of."""

    def __init__(self, w_mm, h_mm, ppm, W, H, plates):
        self.w_mm, self.h_mm, self.ppm, self.W, self.H = w_mm, h_mm, ppm, W, H
        self.a = {p: np.zeros((H, W), F32) for p in plates}
        self.a[plates[0]].fill(1.0)
        self.trap = F32(TRAP * ppm)

    def box(self, cx, cy, rx, ry=None):
        """(slices, X (1, w), Y (h, 1)) of the pixels within rx, ry mm of (cx, cy) on the sheet, X and Y in mm from
        (cx, cy); None when the box misses the sheet."""
        ry = rx if ry is None else ry
        p = self.ppm
        x0, x1 = max(int(math.floor((cx - rx) * p)), 0), min(int(math.ceil((cx + rx) * p)), self.W)
        y0, y1 = max(int(math.floor((cy - ry) * p)), 0), min(int(math.ceil((cy + ry) * p)), self.H)
        if x0 >= x1 or y0 >= y1:
            return None
        X = (np.arange(x0, x1, dtype=F32) + F32(0.5)) * F32(1.0 / p) - F32(cx)
        Y = (np.arange(y0, y1, dtype=F32) + F32(0.5)) * F32(1.0 / p) - F32(cy)
        return (slice(y0, y1), slice(x0, x1)), X[None, :], Y[:, None]

    def lay(self, sl, plate, d):
        """Print `plate` where the signed distance `d` (pixels, positive inside) says, knocking every other plate out
        a trap's width inside the same outline."""
        keep = F32(1.0) - _edge(d - self.trap)
        for name, a in self.a.items():
            if name != plate:
                a[sl] *= keep
        self.over(sl, plate, _edge(d + self.trap))

    def knock(self, sl, plate, cov, under):
        """Print coverage `cov` of `plate`, knocking every other plate out under coverage `under`."""
        keep = F32(1.0) - under
        for name, a in self.a.items():
            if name != plate:
                a[sl] *= keep
        self.over(sl, plate, cov)

    def over(self, sl, plate, cov):
        """Overprint coverage `cov` of `plate` (nothing knocked out)."""
        a = self.a[plate][sl]
        a += cov * (F32(1.0) - a)


def _shifted(a, dx, dy):
    """Plate `a` moved by (dx, dy) pixels (bilinear; no ink comes in at the sheet's edges)."""
    H, W = a.shape
    out = np.zeros_like(a)
    ix, iy = int(math.floor(dx)), int(math.floor(dy))
    fx, fy = dx - ix, dy - iy
    for ox, oy, w in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)), (ix, iy + 1, (1 - fx) * fy),
                      (ix + 1, iy + 1, fx * fy)):
        if w <= 0.0 or abs(ox) >= W or abs(oy) >= H:
            continue
        dst = (slice(max(oy, 0), H + min(oy, 0)), slice(max(ox, 0), W + min(ox, 0)))
        src = (slice(max(-oy, 0), H + min(-oy, 0)), slice(max(-ox, 0), W + min(-ox, 0)))
        out[dst] += F32(w) * a[src]
    return out


def _compose(S, inks, seed, key, offset_plate):
    """The printed sheet, (H, W, 3) uint8 sRGB: the paper's white with its tooth, every plate's ink multiplied over it at
    the plate's coverage times its film density; `offset_plate` printed MISREGISTER mm off register."""
    H, W, ppm = S.H, S.W, S.ppm
    rd = _rng(seed, key, 91)
    ang = rd.uniform(0.0, TAU)
    off = MISREGISTER * ppm
    plates = dict(S.a)
    if offset_plate is not None:
        plates[offset_plate] = _shifted(plates[offset_plate], off * math.cos(ang), off * math.sin(ang))
    # the ink film: one smooth field (knots in mm) wide enough to give every plate its own window of it
    pad = 24.0
    film = _smooth(rd, S.h_mm + pad, S.w_mm + pad, ppm, H + int(round(pad * ppm)), W + int(round(pad * ppm)), 6.0)
    film += F32(0.6) * _smooth(rd, S.h_mm + pad, S.w_mm + pad, ppm, H + int(round(pad * ppm)),
                               W + int(round(pad * ppm)), 1.6)
    # the paper's tooth: per-pixel grain, softened over a few pixels
    rt = _rng(seed, key, 92)
    g = rt.random((H, W), dtype=F32)
    t = g.copy()
    t[1:] += g[:-1]
    t[:-1] += g[1:]
    tooth = t.copy()
    tooth[:, 1:] += t[:, :-1]
    tooth[:, :-1] += t[:, 1:]
    tooth *= F32(1.0 / 9.0)
    tooth -= F32(0.5)
    tooth *= F32(1.0 / 0.096)                                          # ~unit spread
    starve = np.clip((tooth - F32(2.0)) * F32(0.35), F32(0.0), F32(0.3))    # specks on the tooth's peaks
    starve = F32(1.0) - starve
    chans = [F32(PAPER[c]) * (F32(1.0) + F32(0.005) * tooth) for c in range(3)]
    e = np.empty((H, W), F32)
    tmp = np.empty((H, W), F32)
    for i, (name, a) in enumerate(plates.items()):
        oy, ox = int(round((2.0 + 3.7 * i) % pad * ppm)), int(round((1.0 + 5.3 * i) % pad * ppm))
        amp = 0.015 if i == 0 else 0.04                                # a flood's film is even, a motif's less so
        np.multiply(film[oy:oy + H, ox:ox + W], F32(amp), out=e)
        e += F32(1.0 - amp)
        e *= a
        e *= starve
        for c in range(3):
            k = 1.0 - inks[name][c]
            if k <= 0.0:
                continue
            np.multiply(e, F32(-k), out=tmp)
            tmp += F32(1.0)
            chans[c] *= tmp
    out = np.empty((H, W, 3), np.uint8)
    for c in range(3):
        v = chans[c]
        np.clip(v, F32(0.0), F32(1.0), out=v)
        v *= F32(255.0)
        v += F32(0.5)
        out[..., c] = v.astype(np.uint8)
    return out


# ===================================================================================================================
# layout: discs kept apart
# ===================================================================================================================
class _Discs:
    """Occupied discs (x, y, r) in mm: a motif goes where it keeps `gap` from every one."""

    def __init__(self, cap=4096):
        self.d = np.zeros((cap, 3))
        self.n = 0

    def free(self, x, y, r, gap):
        if self.n == 0:
            return True
        d = self.d[:self.n]
        return bool((np.hypot(d[:, 0] - x, d[:, 1] - y) - d[:, 2]).min() >= r + gap)

    def add(self, x, y, r):
        if self.n == len(self.d):
            self.d = np.concatenate([self.d, np.zeros_like(self.d)])
        self.d[self.n] = (x, y, r)
        self.n += 1


def _rot(X, Y, ang):
    """Local coordinates (u along `ang`, v across) of the box's X (1, w), Y (h, 1): (h, w) arrays."""
    c, s = F32(math.cos(ang)), F32(math.sin(ang))
    return X * c + Y * s, Y * c - X * s


def _seg_d(u, v, x0, y0, x1, y1):
    """Distance (same units) from points (u, v) to the segment (x0, y0)-(x1, y1)."""
    ex, ey = x1 - x0, y1 - y0
    L2 = ex * ex + ey * ey
    if L2 <= 0.0:
        return np.hypot(u - F32(x0), v - F32(y0))
    t = np.clip(((u - F32(x0)) * F32(ex) + (v - F32(y0)) * F32(ey)) * F32(1.0 / L2), F32(0.0), F32(1.0))
    return np.hypot(u - F32(x0) - t * F32(ex), v - F32(y0) - t * F32(ey))


def _arc_d(X, Y, x0, y0, ang, length, curv):
    """Distance (mm) from the box's points to an arc starting at (x0, y0) heading `ang`, `length` mm long, bending
    with curvature `curv` (1/mm; positive turns toward +v)."""
    if abs(curv) * length < 1e-3:
        return _seg_d(X + F32(0.0) * Y, Y + F32(0.0) * X, x0, y0, x0 + length * math.cos(ang),
                      y0 + length * math.sin(ang))
    rho = 1.0 / curv
    cx, cy = x0 - rho * math.sin(ang), y0 + rho * math.cos(ang)
    a0 = math.atan2(y0 - cy, x0 - cx)
    sweep = length * curv
    x1, y1 = cx + abs(rho) * math.cos(a0 + sweep), cy + abs(rho) * math.sin(a0 + sweep)
    dx, dy = X - F32(cx), Y - F32(cy)
    rr = np.hypot(dx, dy)
    rel = np.arctan2(dy, dx) - F32(a0)
    rel = (rel + F32(math.pi)) % F32(TAU) - F32(math.pi)
    lo, hi = min(0.0, sweep), max(0.0, sweep)
    on = (rel >= F32(lo)) & (rel <= F32(hi))
    ends = np.minimum(np.hypot(X - F32(x0), Y - F32(y0)), np.hypot(X - F32(x1), Y - F32(y1)))
    return np.where(on, np.abs(rr - F32(abs(rho))), ends)


# ===================================================================================================================
# roses
# ===================================================================================================================
def _rose_params(rng, x, y, R):
    rings = []
    for j, rho in enumerate((1.0, 0.8, 0.6, 0.42)):
        n = int(rng.integers(5, 8)) if j == 0 else int(rng.integers(4, 7)) if j < 3 else int(rng.integers(3, 5))
        rings.append((n, float(rng.uniform(0.0, TAU)), rng.uniform(0.93, 1.05, n).astype(F32), rho,
                      float(rng.uniform(0.09, 0.15))))
    return {"x": x, "y": y, "R": R, "rot": float(rng.uniform(0.0, TAU)), "squash": float(rng.uniform(0.84, 1.0)),
            "rings": rings, "pitch": float(rng.uniform(0.07, 0.09)), "heart": float(rng.uniform(0.21, 0.26))}


def _draw_rose(S, m):
    R = m["R"]
    hit = S.box(m["x"], m["y"], 1.08 * R)
    if hit is None:
        return
    sl, X, Y = hit
    ppm = F32(S.ppm)
    k = F32(R * S.ppm)                                                 # pixels per unit of radius
    u, v = _rot(X, Y, m["rot"])
    v = v * F32(1.0 / m["squash"])
    r = np.sqrt(u * u + v * v) * F32(1.0 / R)
    th = np.arctan2(v, u)
    lit = F32(0.5) + F32(0.5) * np.cos(th + F32(m["rot"] - LIGHT))
    rings = []
    for n, ph, amp, rho, depth in m["rings"]:
        t = (th - F32(ph)) * F32(n / TAU)
        fl = np.floor(t)
        f = t - fl
        b = np.sqrt(np.clip(F32(4.0) * f * (F32(1.0) - f), F32(0.0), F32(1.0)))   # 0 at the joins, 1 mid-petal
        Rj = amp[fl.astype(np.intp) % n] * (F32(rho * (1.0 - depth)) + F32(rho * depth) * b)
        rings.append((Rj, b, f, n, _edge((Rj - r) * k)))
    nr = len(rings)
    S.lay(sl, "rose", (rings[0][0] - r) * k)
    # depth: crescents under every inner ring, creases at the petals' joins, the heart
    shade = np.zeros_like(r)
    for j in range(nr):
        Rj, b, f, n, ins = rings[j]
        above = (F32(1.0) - rings[j + 1][4]) if j + 1 < nr else None
        if j > 0:
            w = F32(0.03) + F32(0.08) * b * (F32(0.35) + F32(0.65) * (F32(1.0) - lit))
            np.maximum(shade, _edge((Rj + w - r) * k) * (F32(1.0) - ins), out=shade)
        dj = np.minimum(f, F32(1.0) - f) * r * F32(TAU / n * R)       # mm from the nearest join
        cw = F32(0.17) * np.clip(F32(1.0) - (Rj - r) * F32(1.0 / 0.17), F32(0.0), F32(1.0))
        crease = _edge((cw - dj) * ppm) * ins
        if above is not None:
            crease *= above
        np.maximum(shade, crease, out=shade)
    heart = m["heart"]
    np.maximum(shade, _edge((F32(heart) - r) * k), out=shade)
    S.over(sl, "shade", shade)
    # highlights in the petal ink, knocked out of the main ink and the shade
    hl = np.zeros_like(r)
    for j in range(nr):
        Rj, b, f, n, ins = rings[j]
        h = F32(0.09) * b * b * (F32(0.2) + F32(0.8) * lit)
        band = ins * (F32(1.0) - _edge((Rj - h - r) * k))
        if j + 1 < nr:
            band *= F32(1.0) - rings[j + 1][4]
        np.maximum(hl, band, out=hl)
    p = r * F32(1.0 / m["pitch"]) - th * F32(1.0 / TAU)
    fp = p - np.floor(p)
    ds = np.minimum(fp, F32(1.0) - fp) * F32(m["pitch"] * R)          # mm to the spiral's nearest arm
    spiral = _edge((F32(0.13) - ds) * ppm) * _edge((F32(heart * 0.9) - r) * k) * _edge((r - F32(0.04)) * k)
    np.maximum(hl, spiral, out=hl)
    S.knock(sl, "petal", hl, hl)


def _leaf_params(rng, x, y, ang, L):
    return {"x": x, "y": y, "ang": ang, "L": L, "wid": float(rng.uniform(0.44, 0.56)) * L,
            "bend": float(rng.uniform(-0.12, 0.12)), "teeth": int(rng.integers(7, 11)),
            "veins": int(rng.integers(5, 7))}


def _draw_leaf(S, m, plate="leaf"):
    L, ang = m["L"], m["ang"]
    mx, my = m["x"] + 0.5 * L * math.cos(ang), m["y"] + 0.5 * L * math.sin(ang)
    hit = S.box(mx, my, 0.5 * L + 0.5)
    if hit is None:
        return
    sl, X, Y = hit
    ppm = F32(S.ppm)
    u, v = _rot(X, Y, ang)
    u = u + F32(0.5 * L)
    t = np.clip(u * F32(1.0 / L), F32(0.0), F32(1.0))
    v = v - F32(m["bend"] * L) * F32(4.0) * t * (F32(1.0) - t)        # the midrib bends
    shape = np.sqrt(t) * (F32(1.0) - t) ** F32(1.1) * F32(2.6)          # rounded base, pointed tip
    saw = t * F32(m["teeth"])
    saw = (saw - np.floor(saw)) ** F32(2.0)
    hw = F32(0.5 * m["wid"]) * shape * (F32(1.0) - F32(0.07) * saw * (t > F32(0.12)))
    inside = (hw - np.abs(v)) * ppm
    inside = np.where((u > F32(0.0)) & (u < F32(L)), inside, F32(-4.0))
    # veins knocked out of the ink: the midrib and slanted side veins
    av = np.abs(v)
    mid = _edge((F32(0.11) * (F32(1.0) - F32(0.6) * t) - av) * ppm) * (t < F32(0.93))
    sp = L / (m["veins"] + 1.0)
    s = (u - av * F32(1.2)) * F32(1.0 / sp)
    fs = s - np.floor(s)
    dv = np.minimum(fs, F32(1.0) - fs) * F32(sp / math.sqrt(1.0 + 1.2 ** 2))
    side = _edge((F32(0.075) - dv) * ppm) * (av < F32(0.8) * hw) * (t > F32(0.1)) * (t < F32(0.85))
    vein = np.maximum(mid, side)
    trap = S.trap
    S.knock(sl, plate, _edge(inside + trap) * (F32(1.0) - vein), _edge(inside - trap))


def _clover_params(rng, x, y, R, four):
    n = 4 if four else 3
    return {"x": x, "y": y, "R": R, "n": n, "rot": float(rng.uniform(0.0, TAU)),
            "scale": rng.uniform(0.9, 1.06, n).astype(F32), "stem_ang": float(rng.uniform(0.0, TAU)),
            "stem_len": float(rng.uniform(0.9, 1.5)) * R, "stem_curv": float(rng.uniform(-0.12, 0.12))}


def _draw_clover(S, m):
    R, n = m["R"], m["n"]
    ext = R + m["stem_len"]
    hit = S.box(m["x"], m["y"], ext)
    if hit is None:
        return
    sl, X, Y = hit
    ppm = F32(S.ppm)
    stem = _arc_d(X, Y, 0.0, 0.0, m["stem_ang"], m["stem_len"], m["stem_curv"] / max(R, 1e-6))
    S.lay(sl, "clover", (F32(0.3) - stem) * ppm)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X) - F32(m["rot"])
    t = th * F32(n / TAU)
    i = np.floor(t + F32(0.5))
    a = (t - i) * F32(TAU / n)                                         # angle from the nearest leaflet's axis
    Ls = F32(R) * m["scale"][i.astype(np.intp) % n]                   # that leaflet's length, mm
    u = r * np.cos(a) / Ls                                             # leaflet units: u out along it, v across
    v = np.abs(r * np.sin(a)) / Ls
    lobe = np.hypot(u - F32(0.6), v - F32(0.2)) - F32(0.3)            # two round lobes, a notch between them
    side = (v * F32(0.53) - (u - F32(0.02)) * F32(0.42)) * F32(1.0 / 0.676)
    wedge = np.maximum(side, u - F32(0.6))                             # narrowing to the clover's middle
    d = np.minimum(np.minimum(lobe, wedge), np.hypot(u, v) - F32(0.09)) * Ls
    inside = -d * ppm
    chev = np.abs(u - F32(0.4) - F32(0.55) * v) * F32(1.0 / math.hypot(1.0, 0.55)) * Ls
    mark = _edge((F32(0.11) - chev) * ppm) * (v < F32(0.3)) * _edge((-d - F32(0.25)) * ppm)
    trap = S.trap
    S.knock(sl, "clover", _edge(inside + trap) * (F32(1.0) - mark), _edge(inside - trap))


def _accent_params(rng, x, y, size):
    kind = ("dot", "dots", "star", "star", "sprig")[int(rng.integers(0, 5))]
    return {"x": x, "y": y, "size": size, "kind": kind, "rot": float(rng.uniform(0.0, TAU)),
            "curv": float(rng.uniform(-0.25, 0.25))}


def _rhombus_d(u, v, a, b):
    """Signed distance to a rhombus with half-diagonals a (along u) and b (along v)."""
    pu, pv = np.abs(u), np.abs(v)
    h = np.clip(((a - 2.0 * pu) * a - (b - 2.0 * pv) * b) * F32(1.0 / (a * a + b * b)), F32(-1.0), F32(1.0))
    d = np.hypot(pu - F32(0.5 * a) * (F32(1.0) - h), pv - F32(0.5 * b) * (F32(1.0) + h))
    return d * np.sign(pu * F32(b) + pv * F32(a) - F32(a * b))


def _accent_d(X, Y, m):
    """Signed distance (mm, negative inside) of an accent motif."""
    size, kind = m["size"], m["kind"]
    u, v = _rot(X, Y, m["rot"])
    if kind == "dot":
        return np.hypot(u, v) - F32(0.28 * size)
    if kind == "dots":
        d = None
        for a in (0.0, TAU / 3, 2 * TAU / 3):
            di = np.hypot(u - F32(0.3 * size * math.cos(a)), v - F32(0.3 * size * math.sin(a))) - F32(0.17 * size)
            d = di if d is None else np.minimum(d, di)
        return d
    if kind == "star":
        return np.minimum(_rhombus_d(u, v, 0.5 * size, 0.12 * size), _rhombus_d(v, u, 0.36 * size, 0.1 * size))
    stem = _arc_d(u, v, -0.45 * size, 0.0, 0.0, 0.9 * size, m["curv"] / size)
    d = stem - F32(0.05 * size)
    for f, side in ((0.25, 1.0), (0.5, -1.0), (0.75, 1.0)):
        cx, cy = -0.45 * size + f * 0.9 * size, side * 0.17 * size
        d = np.minimum(d, np.hypot(u - F32(cx), v - F32(cy)) - F32(0.1 * size))
    return np.minimum(d, np.hypot(u - F32(0.47 * size), v) - F32(0.12 * size))


def _draw_accent(S, m, plate="accent"):
    hit = S.box(m["x"], m["y"], 0.6 * m["size"] + 0.3)
    if hit is None:
        return
    sl, X, Y = hit
    S.lay(sl, plate, -_accent_d(X, Y, m) * F32(S.ppm))


def _roses(S, rng):
    w, h = S.w_mm, S.h_mm
    occ = _Discs()
    roses, leaves = [], []
    for _ in range(700):
        R = float(rng.uniform(8.0, 13.0))
        x, y = float(rng.uniform(-8.0, w + 8.0)), float(rng.uniform(-8.0, h + 8.0))
        rose = _rose_params(rng, x, y, R)
        a0 = float(rng.uniform(0.0, TAU))
        cnt = int(rng.integers(2, 4))
        mine, reach = [], R
        for j in range(cnt):
            a = a0 + j * TAU / cnt + float(rng.uniform(-0.45, 0.45))
            L = float(rng.uniform(0.95, 1.25)) * R + 1.5
            b = 0.45 * R
            mine.append(_leaf_params(rng, x + b * math.cos(a), y + b * math.sin(a), a + float(rng.uniform(-0.3, 0.3)), L))
            reach = max(reach, b + L)
        if occ.free(x, y, 0.5 * (R + reach), 7.0):
            occ.add(x, y, 0.5 * (R + reach))
            roses.append(rose)
            leaves += mine
    clovers = []
    for _ in range(1600):
        R = 0.5 * float(rng.uniform(8.0, 13.0))
        x, y = float(rng.uniform(-4.0, w + 4.0)), float(rng.uniform(-4.0, h + 4.0))
        m = _clover_params(rng, x, y, R, rng.uniform() < 0.18)
        if len(clovers) < w * h / 700.0 and occ.free(x, y, R + 0.35 * m["stem_len"], 3.0):
            occ.add(x, y, R + 0.35 * m["stem_len"])
            clovers.append(m)
    accents = []
    for _ in range(4000):
        size = float(rng.uniform(1.6, 3.6))
        x, y = float(rng.uniform(-1.0, w + 1.0)), float(rng.uniform(-1.0, h + 1.0))
        m = _accent_params(rng, x, y, size)
        if occ.free(x, y, 0.55 * size, 2.2):
            occ.add(x, y, 0.55 * size)
            accents.append(m)
    for m in leaves:
        _draw_leaf(S, m)
    for m in roses:
        _draw_rose(S, m)
    for m in clovers:
        _draw_clover(S, m)
    for m in accents:
        _draw_accent(S, m)


# ===================================================================================================================
# memphis
# ===================================================================================================================
MEMPHIS_INKS = ("love", "gold", "pine", "iris")
MEMPHIS_KINDS = (("squiggle", 5), ("zigzag", 3), ("triangle", 5), ("dot", 4), ("ring", 2), ("half", 3), ("grid", 2),
                 ("dotgrid", 2), ("pill", 2))


def _memphis_params(rng, x, y):
    names = [k for k, _ in MEMPHIS_KINDS]
    wts = np.array([w for _, w in MEMPHIS_KINDS], float)
    kind = names[int(rng.choice(len(names), p=wts / wts.sum()))]
    ink = MEMPHIS_INKS[int(rng.integers(0, 4))]
    m = {"x": x, "y": y, "kind": kind, "ink": ink, "rot": float(rng.uniform(0.0, TAU)),
         "outline": bool(rng.uniform() < 0.35), "oa": float(rng.uniform(0.0, TAU))}
    if kind == "squiggle":
        m.update(L=float(rng.uniform(16.0, 26.0)), A=float(rng.uniform(1.4, 2.4)), lam=float(rng.uniform(5.5, 8.0)),
                 t=float(rng.uniform(1.3, 2.0)), ph=float(rng.uniform(0.0, TAU)))
        m["ink"] = (MEMPHIS_INKS + ("line",))[int(rng.integers(0, 5))]
        m["r"] = 0.5 * m["L"] + m["A"]
    elif kind == "zigzag":
        m.update(L=float(rng.uniform(13.0, 22.0)), A=float(rng.uniform(1.0, 1.8)), lam=float(rng.uniform(3.0, 4.6)),
                 t=float(rng.uniform(0.6, 1.0)), ph=float(rng.uniform(0.0, 1.0)))
        m["ink"] = ("line", "line", "love", "pine", "iris")[int(rng.integers(0, 5))]
        m["r"] = 0.5 * m["L"] + m["A"]
    elif kind == "triangle":
        s = float(rng.uniform(4.0, 8.0))
        a = np.arange(3) * (TAU / 3) + rng.uniform(-0.3, 0.3, 3)     # about even: confetti, never a sliver
        rr = rng.uniform(0.8, 1.0, 3) * s * 0.6
        m.update(pts=[(float(rr[i] * math.cos(a[i])), float(rr[i] * math.sin(a[i]))) for i in range(3)], r=0.6 * s)
    elif kind == "dot":
        m.update(rad=float(rng.uniform(0.9, 2.2)))
        m["r"] = m["rad"]
    elif kind == "ring":
        m.update(rad=float(rng.uniform(2.2, 3.8)), t=float(rng.uniform(0.6, 0.9)))
        m["r"] = m["rad"] + m["t"]
    elif kind == "half":
        m.update(rad=float(rng.uniform(3.5, 6.0)))
        m["r"] = m["rad"]
    elif kind in ("grid", "dotgrid"):
        m.update(side=float(rng.uniform(7.0, 11.0)), sp=float(rng.uniform(1.3, 1.8)))
        m["ink"] = "line"
        m["r"] = 0.72 * m["side"]
    else:
        m.update(L=float(rng.uniform(6.0, 10.0)), rad=float(rng.uniform(1.1, 1.7)))
        m["r"] = 0.5 * m["L"] + m["rad"]
    return m


def _memphis_d(X, Y, m, dx=0.0, dy=0.0):
    """Signed distance (mm) of a Memphis shape, moved by (dx, dy) mm."""
    u, v = _rot(X - F32(dx), Y - F32(dy), m["rot"])
    kind = m["kind"]
    if kind == "squiggle":
        L, A, w, ph = m["L"], m["A"], TAU / m["lam"], m["ph"]
        lo, hi = F32(-0.5 * L), F32(0.5 * L)
        t = np.clip(u, lo, hi)
        for _ in range(3):                                             # Gauss-Newton: the wave's nearest point
            arg = t * F32(w) + F32(ph)
            dy = F32(A * w) * np.cos(arg)
            t = np.clip(t - ((t - u) + (F32(A) * np.sin(arg) - v) * dy) / (F32(1.0) + dy * dy), lo, hi)
        return np.hypot(t - u, F32(A) * np.sin(t * F32(w) + F32(ph)) - v) - F32(0.5 * m["t"])
    if kind == "zigzag":
        L, A, lam = m["L"], m["A"], m["lam"]
        q = u * F32(1.0 / lam) + F32(m["ph"])
        tri = F32(A) * (F32(4.0) * np.abs(q - np.floor(q) - F32(0.5)) - F32(1.0))
        body = np.abs(v - tri) * F32(1.0 / math.sqrt(1.0 + (4.0 * A / lam) ** 2))
        d = np.where(np.abs(u) <= F32(0.5 * L), body, F32(1e3))
        return d - F32(0.5 * m["t"])
    if kind == "triangle":
        P = m["pts"]
        d = None
        for i in range(3):
            (x0, y0), (x1, y1) = P[i], P[(i + 1) % 3]
            nx, ny = y1 - y0, -(x1 - x0)
            ln = math.hypot(nx, ny)
            di = ((u - F32(x0)) * F32(nx / ln) + (v - F32(y0)) * F32(ny / ln))
            d = di if d is None else np.maximum(d, di)
        return d
    if kind == "dot":
        return np.hypot(u, v) - F32(m["rad"])
    if kind == "ring":
        return np.abs(np.hypot(u, v) - F32(m["rad"])) - F32(0.5 * m["t"])
    if kind == "half":
        return np.maximum(np.hypot(u, v) - F32(m["rad"]), -v)
    if kind == "grid":
        sp, half = m["sp"], 0.5 * m["side"]
        gu = u * F32(1.0 / sp)
        gv = v * F32(1.0 / sp)
        lu = np.abs(gu - np.floor(gu) - F32(0.5)) * F32(sp)
        lv = np.abs(gv - np.floor(gv) - F32(0.5)) * F32(sp)
        lines = np.minimum(lu, lv) - F32(0.16)
        return np.maximum(lines, np.maximum(np.abs(u), np.abs(v)) - F32(half))
    if kind == "dotgrid":
        sp, half = m["sp"], 0.5 * m["side"]
        gu = u * F32(1.0 / sp)
        gv = v * F32(1.0 / sp)
        du = (gu - np.floor(gu) - F32(0.5)) * F32(sp)
        dv = (gv - np.floor(gv) - F32(0.5)) * F32(sp)
        return np.maximum(np.hypot(du, dv) - F32(0.3), np.maximum(np.abs(u), np.abs(v)) - F32(half))
    L = m["L"]
    return _seg_d(u, v, -0.5 * L, 0.0, 0.5 * L, 0.0) - F32(m["rad"])


def _draw_memphis(S, m):
    pad = 1.2 if m["outline"] else 0.3
    hit = S.box(m["x"], m["y"], m["r"] + pad)
    if hit is None:
        return
    sl, X, Y = hit
    ppm = F32(S.ppm)
    S.lay(sl, m["ink"], -_memphis_d(X, Y, m) * ppm)
    if m["outline"] and m["kind"] in ("triangle", "dot", "half", "pill") and m["ink"] != "line":
        o = 0.75
        d = _memphis_d(X, Y, m, o * math.cos(m["oa"]), o * math.sin(m["oa"]))
        S.lay(sl, "line", (F32(0.22) - np.abs(d)) * ppm)


def _memphis(S, rng):
    w, h = S.w_mm, S.h_mm
    occ = _Discs()
    items = []
    for _ in range(5000):
        x, y = float(rng.uniform(-6.0, w + 6.0)), float(rng.uniform(-6.0, h + 6.0))
        m = _memphis_params(rng, x, y)
        if occ.free(x, y, 0.8 * m["r"], 2.5):
            occ.add(x, y, 0.8 * m["r"])
            items.append(m)
    for m in items:
        _draw_memphis(S, m)


# ===================================================================================================================
# marbled
# ===================================================================================================================
MARBLE_INKS = ("love", "rose", "iris", "pine", "gold")
MARBLE_PERIOD = 90.0                         # mm of stripe coordinate the band sequence repeats over
MARBLE_STEP = 0.5                            # mm between samples of the warp


def _perlin_tables(rng):
    perm = rng.permutation(256)
    perm = np.concatenate([perm, perm]).astype(np.intp)
    a = rng.uniform(0.0, TAU, 256)
    return perm, np.cos(a).astype(F32), np.sin(a).astype(F32)


def _perlin(x, y, tab):
    """Gradient noise, about -1..1, at (x, y) in lattice units (float32 arrays)."""
    perm, gx, gy = tab
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    xi, yi = x0.astype(np.intp) & 255, y0.astype(np.intp) & 255
    u = fx * fx * fx * (fx * (fx * F32(6.0) - F32(15.0)) + F32(10.0))
    v = fy * fy * fy * (fy * (fy * F32(6.0) - F32(15.0)) + F32(10.0))
    pa, pb = perm[xi], perm[xi + 1]
    h00, h10, h01, h11 = perm[pa + yi], perm[pb + yi], perm[pa + yi + 1], perm[pb + yi + 1]
    n00 = gx[h00] * fx + gy[h00] * fy
    n10 = gx[h10] * (fx - F32(1.0)) + gy[h10] * fy
    n01 = gx[h01] * fx + gy[h01] * (fy - F32(1.0))
    n11 = gx[h11] * (fx - F32(1.0)) + gy[h11] * (fy - F32(1.0))
    a = n00 + (n10 - n00) * u
    b = n01 + (n11 - n01) * u
    return (a + (b - a) * v) * F32(1.4)


def _fbm(x, y, tab, octaves=3):
    tot, amp, f = None, 1.0, 1.0
    for o in range(octaves):
        n = _perlin(x * F32(f) + F32(17.3 * o), y * F32(f) + F32(5.1 * o), tab) * F32(amp)
        tot = n if tot is None else tot + n
        amp *= 0.5
        f *= 2.03
    return tot


def _marble_bands(rng):
    """The band sequence along the stripe coordinate: (edges (n + 1,) mm, ink index per band; -1 is the ground). The inks
    come from a shuffled bag, so every one of them runs through the sheet about as often as the others."""
    edges, ids, bag = [0.0], [], []

    def put(width, which):
        ids.append(which)
        edges.append(edges[-1] + width)

    def ink():
        if not bag:
            bag.extend(int(i) for i in rng.permutation(len(MARBLE_INKS)))
        return bag.pop()

    while edges[-1] < MARBLE_PERIOD:
        put(float(rng.uniform(1.2, 6.5)), -1)
        roll = rng.uniform()
        if roll < 0.5:
            put(float(rng.uniform(0.3, 1.4)), ink())
        elif roll < 0.8:
            a, b = ink(), ink()
            for j in range(int(rng.integers(3, 7))):
                put(float(rng.uniform(0.18, 0.4)), a if j % 2 == 0 else b)
                put(float(rng.uniform(0.25, 0.6)), -1)
        else:
            edge, fill = ink(), ink()
            put(float(rng.uniform(0.25, 0.5)), edge)
            put(float(rng.uniform(1.5, 3.5)), fill)
            put(float(rng.uniform(0.2, 0.4)), edge)
    e = np.array(edges)
    e *= MARBLE_PERIOD / e[-1]
    return e, np.array(ids, np.intp)


def _upsample(a, n, ppm, step):
    """Rows of `a` (samples every `step` mm from 0) at the centres of n pixels (linear)."""
    f = (np.arange(n, dtype=np.float64) + 0.5) / (ppm * step)
    i = np.minimum(np.floor(f).astype(np.intp), a.shape[0] - 2)
    w = (f - i).astype(F32)
    shape = (n,) + (1,) * (a.ndim - 1)
    return a[i] * (F32(1.0) - w).reshape(shape) + a[i + 1] * w.reshape(shape)


def _marbled(S, rng):
    w_mm, h_mm, ppm, W, H = S.w_mm, S.h_mm, S.ppm, S.W, S.H
    tab = _perlin_tables(rng)
    gx = np.arange(int(w_mm / MARBLE_STEP) + 3, dtype=F32) * F32(MARBLE_STEP)
    gy = np.arange(int(h_mm / MARBLE_STEP) + 3, dtype=F32) * F32(MARBLE_STEP)
    X, Y = np.meshgrid(gx, gy)
    sc1, sc2 = 1.0 / 52.0, 1.0 / 34.0
    qx = X + F32(14.0) * _fbm(X * F32(sc1), Y * F32(sc1), tab, 2)
    qy = Y + F32(14.0) * _fbm(X * F32(sc1) + F32(31.7), Y * F32(sc1) + F32(8.3), tab, 2)
    s = qy + F32(7.0) * _fbm(qx * F32(sc2) + F32(3.1), qy * F32(sc2) + F32(71.9), tab, 1)
    s += F32(1.6) * np.sin(qx * F32(TAU / 26.0))
    sy, sx = np.gradient(s, MARBLE_STEP)
    # up to the sheet's pixels: rows, then columns
    s_r, sx_r, sy_r = (_upsample(a, H, ppm, MARBLE_STEP) for a in (s, sx, sy))
    s_f, sx_f, sy_f = (_upsample(a.T, W, ppm, MARBLE_STEP).T for a in (s_r, sx_r, sy_r))
    # the comb: tines along the tube pull the bands up and down across it (and a finer one)
    xs = (np.arange(W, dtype=np.float64) + 0.5) / ppm
    c1, c2 = float(rng.uniform(7.0, 9.0)), float(rng.uniform(1.8, 2.4))
    comb = (0.9 * np.sin(TAU * xs / c1) + 0.15 * np.sin(TAU * xs / c2 + 1.0)).astype(F32)
    dcomb = (0.9 * TAU / c1 * np.cos(TAU * xs / c1) + 0.15 * TAU / c2 * np.cos(TAU * xs / c2 + 1.0)).astype(F32)
    s_f += sy_f * comb[None, :]
    sx_f += sy_f * dcomb[None, :]
    g = F32(ppm) / np.maximum(np.hypot(sx_f, sy_f), F32(1e-3))       # pixels per mm of stripe coordinate
    edges, ids = _marble_bands(rng)
    sm = np.mod(s_f, F32(MARBLE_PERIOD))
    b = np.clip(np.searchsorted(edges.astype(F32), sm, side="right") - 1, 0, len(ids) - 1)
    lo, hi = edges.astype(F32)[b], edges.astype(F32)[b + 1]
    dlo, dhi = (sm - lo) * g, (hi - sm) * g
    cov = np.clip(dlo + F32(0.5), F32(0.0), F32(1.0)) + np.clip(dhi + F32(0.5), F32(0.0), F32(1.0)) - F32(1.0)
    nb = np.where(dlo < dhi, ids[(b - 1) % len(ids)], ids[(b + 1) % len(ids)])
    me = ids[b]
    rest = F32(1.0) - cov
    for i, name in enumerate(MARBLE_INKS):
        a = S.a[name]
        a[...] = np.where(me == i, cov, F32(0.0)) + np.where(nb == i, rest, F32(0.0))
    S.a["ground"][...] = np.where(me < 0, cov, F32(0.0)) + np.where(nb < 0, rest, F32(0.0))


# ===================================================================================================================
# the print
# ===================================================================================================================
_PLATES = {                                  # pattern -> ([plate], {plate: role whose colour it prints}, off-register plate)
    "roses": (("ground", "leaf", "rose", "shade", "petal", "clover", "accent"), {"shade": "rose"}, "petal"),
    "memphis": (("ground", "love", "gold", "pine", "iris", "line"), {}, "line"),
    "marbled": (("ground",) + MARBLE_INKS, {}, "gold"),
}
_DRAW = {"roses": _roses, "memphis": _memphis, "marbled": _marbled}


def paper_print(pattern, colors, seed=1, size_mm=SIZE_MM, px_per_mm=PX_PER_MM):
    """The printed paper, (H, W, 3) uint8 sRGB: W = round(size_mm[0] * px_per_mm) columns round the tube (x = 0 and W
    meet at the seam), H = round(size_mm[1] * px_per_mm) rows along it from the eyepiece end. `colors` = colors(pattern,
    ...). The layout is in millimetres (the same motifs in the same places at every px_per_mm) and fixed by `seed`."""
    cols = _inks(pattern, colors)
    w_mm, h_mm, ppm, W, H = _sheet_size(size_mm, px_per_mm)
    plates, alias, offset = _PLATES[pattern]
    key = PATTERNS.index(pattern) + 1
    S = _Sheet(w_mm, h_mm, ppm, W, H, plates)
    _DRAW[pattern](S, _rng(seed, key, 1))
    inks = {p: cols[alias.get(p, p)] for p in plates}
    return _compose(S, inks, seed, key, offset)


# ===================================================================================================================
# wear
# ===================================================================================================================
def _smoothstep(e0, e1, x):
    t = np.clip((x - F32(e0)) * F32(1.0 / (e1 - e0)), F32(0.0), F32(1.0))
    return t * t * (F32(3.0) - F32(2.0) * t)


def paper_wear(seed=1, size_mm=SIZE_MM, px_per_mm=PX_PER_MM):
    """Where the print has rubbed off, (H, W) float32: 0 the print intact, 1 rubbed through to bare paper. Strongest in
    a ragged, fibrous band a few mm deep at both ends (the turned rims) and along the seam (columns near 0 and W - 1),
    with a few seeded scuffs and fine scratches elsewhere; the same in millimetres at every px_per_mm."""
    w_mm, h_mm, ppm, W, H = _sheet_size(size_mm, px_per_mm)
    rng = _rng(seed, 7, 1)
    out = np.zeros((H, W), F32)
    xs = (np.arange(W, dtype=F32) + F32(0.5)) * F32(1.0 / ppm)
    ys = (np.arange(H, dtype=F32) + F32(0.5)) * F32(1.0 / ppm)
    # the rims: a band whose depth wanders along the rim, torn into fibres that run round it
    deep = 7.0
    nb = min(int(math.ceil(deep * ppm)), H)
    for end in (0, 1):
        depth = (F32(2.2) + F32(0.9) * _smooth(rng, 1.0, w_mm, ppm, 1, W, 1.0, 9.0)[0]
                 + F32(0.45) * _smooth(rng, 1.0, w_mm, ppm, 1, W, 1.0, 2.2)[0])
        fib = _smooth(rng, deep, w_mm, ppm, nb, W, 0.18, 2.6)
        d = ((np.arange(nb, dtype=F32) + F32(0.5)) * F32(1.0 / ppm))[:, None]
        band = F32(1.0) - _smoothstep(-0.9, 0.9, d - depth[None, :] + F32(0.9) * fib)
        band *= F32(0.82) + F32(0.18) * np.clip(F32(1.0) - d * F32(1.0 / 1.2), F32(0.0), F32(1.0))
        rows = slice(0, nb) if end == 0 else slice(H - nb, H)
        out[rows] = np.maximum(out[rows], band if end == 0 else band[::-1])
    # the seam: the overlapping edge lifts and rubs
    deep_s = 4.0
    ns = min(int(math.ceil(deep_s * ppm)), W)
    for end in (0, 1):
        depth = (F32(1.1) + F32(0.5) * _smooth(rng, h_mm, 1.0, ppm, H, 1, 8.0, 1.0)[:, 0]
                 + F32(0.25) * _smooth(rng, h_mm, 1.0, ppm, H, 1, 1.7, 1.0)[:, 0])
        fib = _smooth(rng, h_mm, deep_s, ppm, H, ns, 2.4, 0.16)
        d = ((np.arange(ns, dtype=F32) + F32(0.5)) * F32(1.0 / ppm))[None, :]
        band = F32(1.0) - _smoothstep(-0.7, 0.7, d - depth[:, None] + F32(0.6) * fib)
        band *= F32(0.75)
        cols = slice(0, ns) if end == 0 else slice(W - ns, W)
        out[:, cols] = np.maximum(out[:, cols], band if end == 0 else band[:, ::-1])
    # scuffs: soft rubbed patches, patchy inside
    tab = _perlin_tables(rng)
    n_scuff = int(round(w_mm * h_mm / 2600.0))
    for _ in range(n_scuff):
        cx, cy = float(rng.uniform(0.0, w_mm)), float(rng.uniform(0.0, h_mm))
        a, b = float(rng.uniform(2.5, 8.0)), float(rng.uniform(0.8, 2.5))
        ang = float(rng.uniform(-0.5, 0.5)) + (0.0 if rng.uniform() < 0.7 else 0.5 * math.pi)
        amp = float(rng.uniform(0.25, 0.6))
        x0, x1 = max(int((cx - a) * ppm), 0), min(int(math.ceil((cx + a) * ppm)), W)
        y0, y1 = max(int((cy - a) * ppm), 0), min(int(math.ceil((cy + a) * ppm)), H)
        if x0 >= x1 or y0 >= y1:
            continue
        X, Y = xs[None, x0:x1] - F32(cx), ys[y0:y1, None] - F32(cy)
        u, v = _rot(X, Y, ang)
        e = (u * F32(1.0 / a)) ** 2 + (v * F32(1.0 / b)) ** 2
        patch = F32(1.0) - _smoothstep(0.35, 1.0, e)
        grain = _perlin(u * F32(0.9) + F32(cx), v * F32(4.0) + F32(cy), tab)
        patch *= np.clip(F32(0.55) + F32(0.9) * grain, F32(0.0), F32(1.0)) * F32(amp)
        np.maximum(out[y0:y1, x0:x1], patch, out=out[y0:y1, x0:x1])
    # scratches: fine, slightly bent lines
    n_scr = int(round(w_mm * h_mm / 1100.0))
    for _ in range(n_scr):
        cx, cy = float(rng.uniform(0.0, w_mm)), float(rng.uniform(0.0, h_mm))
        L = float(rng.uniform(4.0, 22.0))
        ang = float(rng.uniform(0.0, TAU))
        curv = float(rng.uniform(-0.04, 0.04))
        wid = float(rng.uniform(0.04, 0.12))
        amp = float(rng.uniform(0.3, 0.75))
        sx, sy = cx - 0.5 * L * math.cos(ang), cy - 0.5 * L * math.sin(ang)
        r = 0.5 * L + 1.0
        x0, x1 = max(int((cx - r) * ppm), 0), min(int(math.ceil((cx + r) * ppm)), W)
        y0, y1 = max(int((cy - r) * ppm), 0), min(int(math.ceil((cy + r) * ppm)), H)
        if x0 >= x1 or y0 >= y1:
            continue
        X, Y = xs[None, x0:x1], ys[y0:y1, None]
        d = _arc_d(X, Y, sx, sy, ang, L, curv)
        t = np.clip(np.hypot(X - F32(cx), Y - F32(cy)) * F32(2.0 / L), F32(0.0), F32(1.0))
        line = _edge((F32(0.5 * wid) - d) * F32(ppm)) * (F32(1.0) - t * t) * F32(amp)
        np.maximum(out[y0:y1, x0:x1], line, out=out[y0:y1, x0:x1])
    return np.clip(out, F32(0.0), F32(1.0))


# ===================================================================================================================
# the view through the eyepiece
# ===================================================================================================================
SQ3 = math.sqrt(3.0)
VIEW_RADIUS = 0.47                           # the circle of view, share of the image's side
VIEW_CELLS = 2.3                             # hexagon apothems from the centre to the circle
VIEW_DIM = 0.84                              # light kept per ring of reflected cells
VIEW_EDGE = 0.2                              # soft dark edge, share of the circle's radius


def _view_colours(cols):
    items = [(k, tuple(min(max(float(x), 0.0), 1.0) for x in v)) for k, v in dict(cols).items()]
    if len(items) < 2 or any(len(c) != 3 for _, c in items):
        raise ValueError("kaleidoscope view: needs a ground and at least one ink, each (r, g, b)")
    ground = dict(items).get("ground", items[0][1])
    inks = [c for k, c in items if k != "ground"] if "ground" in dict(items) else [c for _, c in items[1:]]
    dark = min((c for _, c in items), key=_lum)
    L = _lum(dark)
    if L > 0.012:                                                      # bring it down to near-black, keeping its hue
        lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in dark]
        lin = [v * 0.012 / L for v in lin]
        dark = tuple(12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in lin)
    return ground, inks, dark


def _fold(x, y):
    """Fold the plane (units: the hexagons' apothem; lattice points 2 apart) into the 30-60-90 triangle (0, 0), (1, 0),
    (1, 1/sqrt 3) by the reflections of three mirrors; also the ring of hexagons each point lies in."""
    r = y * F32(1.0 / SQ3)
    q = (x - r) * F32(0.5)
    s = -q - r
    rq, rr, rs = np.rint(q), np.rint(r), np.rint(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    lx = x - (F32(2.0) * rq + rr)
    ly = y - F32(SQ3) * rr
    ring = ((np.abs(rq) + np.abs(rr) + np.abs(rq + rr)) * F32(0.5)).astype(np.intp)
    ang = np.mod(np.arctan2(ly, lx), F32(math.pi / 3))
    ang = np.minimum(ang, F32(math.pi / 3) - ang)
    rad = np.hypot(lx, ly)
    return rad * np.cos(ang), rad * np.sin(ang), ring


def _view_cell(rng, ground, inks, res):
    """The object cell seen through the mirrors' triangle: (ny, nx, 3) float32 over x 0..1, y 0..1/sqrt 3 (apothems)
    at `res` px per apothem: glass bits (colour filters: an overlap mixes) and beads (lit spheres) over the ground."""
    nx, ny = int(res) + 3, int(res / SQ3) + 3
    C = np.empty((ny, nx, 3), F32)
    C[...] = np.asarray(ground, F32)
    G = np.zeros((ny, nx), F32)
    xs = np.arange(nx, dtype=F32) * F32(1.0 / res)
    ys = np.arange(ny, dtype=F32) * F32(1.0 / res)
    inks = [np.asarray(c, F32) for c in inks]

    def box(cx, cy, r):
        x0, x1 = max(int((cx - r) * res), 0), min(int(math.ceil((cx + r) * res)) + 1, nx)
        y0, y1 = max(int((cy - r) * res), 0), min(int(math.ceil((cy + r) * res)) + 1, ny)
        if x0 >= x1 or y0 >= y1:
            return None
        return (slice(y0, y1), slice(x0, x1)), xs[None, x0:x1] - F32(cx), ys[y0:y1, None] - F32(cy)

    def glass(cx, cy, size, col, alpha):
        hit = box(cx, cy, size)
        if hit is None:
            return
        sl, X, Y = hit
        n = int(rng.integers(3, 7))
        a = np.sort(rng.uniform(0.0, TAU, n))
        rr = rng.uniform(0.55, 1.0, n) * size
        P = [(float(rr[i] * math.cos(a[i])), float(rr[i] * math.sin(a[i]))) for i in range(n)]
        d = None
        for i in range(n):
            (x0, y0), (x1, y1) = P[i], P[(i + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            ln = math.hypot(ex, ey)
            if ln < 1e-9:
                continue
            di = ((X - F32(x0)) * F32(ey / ln) - (Y - F32(y0)) * F32(ex / ln))
            d = di if d is None else np.maximum(d, di)
        cov = _edge(-d * F32(res)) * F32(alpha)
        c, g = C[sl], G[sl]
        target = col * (F32(1.0) - g[..., None] + g[..., None] * c)
        c += (target - c) * cov[..., None]
        g += cov - g * cov

    def bead(cx, cy, rad, col, hole=0.0):
        hit = box(cx, cy, rad)
        if hit is None:
            return
        sl, X, Y = hit
        rr = np.hypot(X, Y) * F32(1.0 / rad)
        cov = _edge((F32(1.0) - rr) * F32(rad * res))
        if hole > 0:
            cov *= _edge((rr - F32(hole)) * F32(rad * res))
        z = np.sqrt(np.clip(F32(1.0) - rr * rr, F32(0.0), F32(1.0)))
        shade = (F32(0.55) + F32(0.45) * z)[..., None] * col
        spot = np.exp(-((X + F32(0.35 * rad)) ** 2 + (Y + F32(0.4 * rad)) ** 2) * F32(1.0 / (0.09 * rad * rad)))
        shade = shade + (F32(1.0) - shade) * (F32(0.85) * spot)[..., None]
        c = C[sl]
        c += (shade - c) * cov[..., None]

    order = rng.permutation(len(inks) * 3)
    for i in order:                                                    # glass: big bits first
        glass(float(rng.uniform(-0.05, 1.05)), float(rng.uniform(-0.05, 0.62)), float(rng.uniform(0.1, 0.3)),
              inks[i % len(inks)], float(rng.uniform(0.75, 0.92)))
    for i in range(int(rng.integers(8, 13))):
        col = inks[int(rng.integers(0, len(inks)))]
        bead(float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 0.58)), float(rng.uniform(0.035, 0.08)), col,
             hole=0.42 if rng.uniform() < 0.35 else 0.0)
    return C


def view(colors, seed=1, px=512):
    """What one sees through the eyepiece, (px, px, 3) uint8 sRGB: beads and glass bits in the colours given (the
    roles' inks over their ground; `colors` as colors() returns it) folded by three mirrors into a six-fold star (a
    60-degree turn and a mirror map it onto itself), inside a circle with a soft dark edge; outside it the darkest
    colour given, brought down to near-black."""
    px = int(px)
    if px < 8:
        raise ValueError(f"kaleidoscope view: px = {px} is too small")
    ground, inks, dark = _view_colours(colors)
    rng = _rng(seed, 50, 1)
    radius = VIEW_RADIUS * px
    apo = radius / VIEW_CELLS                                          # pixels per apothem
    res = 2.0 * apo
    cell = _view_cell(rng, ground, inks, res)
    c = np.arange(px, dtype=F32) + F32(0.5 - 0.5 * px)
    X, Y = np.broadcast_arrays(c[None, :] * F32(1.0 / apo), c[:, None] * F32(1.0 / apo))
    fx, fy, ring = _fold(X, Y)
    sx, sy = fx * F32(res), fy * F32(res)
    x0 = np.clip(np.floor(sx).astype(np.intp), 0, cell.shape[1] - 2)
    y0 = np.clip(np.floor(sy).astype(np.intp), 0, cell.shape[0] - 2)
    wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
    img = ((cell[y0, x0] * (F32(1.0) - wx) + cell[y0, x0 + 1] * wx) * (F32(1.0) - wy)
           + (cell[y0 + 1, x0] * (F32(1.0) - wx) + cell[y0 + 1, x0 + 1] * wx) * wy)
    img *= (F32(VIEW_DIM) ** ring.astype(F32))[..., None]
    rr = np.hypot(X, Y) * F32(apo / radius)
    lit = F32(1.0) - _smoothstep(1.0 - VIEW_EDGE, 1.0, rr)
    lit = lit * (F32(0.92) + F32(0.08) * np.clip(F32(1.0) - rr, F32(0.0), F32(1.0)))
    dark = np.asarray(dark, F32)
    img = dark + (img - dark) * lit[..., None]
    return (np.clip(img, F32(0.0), F32(1.0)) * F32(255.0) + F32(0.5)).astype(np.uint8)
