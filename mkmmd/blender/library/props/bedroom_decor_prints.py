"""The printed designs of `poster_80s` (pure Python, no bpy): each style is a function of the printed area's aspect that
returns a `Print` (a colour expression and a gloss expression over the design space of `bedroom_decor_shapes`).
Colours are palette-slot recipes, so the designs follow the project's palette."""
import math
from dataclasses import dataclass

from . import bedroom_decor_field as F
from .bedroom_decor_field import (X, Y, add, c, cadd, cmix, cramp, div, fabs, fmax, fract, mul, over, ramp, remap, sin,
                                  sub, union)
from .bedroom_decor_shapes import (P0, SOFT, BlockRow, D_box, D_disc, D_tri, at, band, bands, box_d, checker, cov, disc,
                                   dots, grad_y, inv, len2, line, profile, rays, repeat_y, ridge, ring, sparkle,
                                   sprinkle, stripes, wave, zigzag)


@dataclass
class Print:
    color: object                   # colour expression
    gloss: object = 0.5             # how glossy the ink is, 0 (matt) .. 1 (varnished): scalar expression or number


STYLES = {}


def style(fn):
    STYLES[fn.__name__] = fn
    return fn


# Colourways: `variant` n re-assigns the accent slots of the designs (a permutation of slot names applied to every colour
# recipe of the design); 0 keeps the designs as drawn.
VARIANTS = (
    {},
    {"love": "pine", "pine": "love", "gold": "foam", "foam": "gold", "rose": "iris", "iris": "rose"},   # warm <-> cool
    {"love": "gold", "gold": "foam", "foam": "iris", "iris": "love", "rose": "pine", "pine": "rose"},   # rotated
)
_PERM = {}


def col(k=1.0, hue=0.0, chroma=1.0, fixed=False, **w):
    """`bedroom_decor_field.col` with the slots of the current colourway (`fixed`: the slots as written)."""
    out = {}
    for s, v in w.items():
        s = s if fixed else _PERM.get(s, s)
        out[s] = out.get(s, 0.0) + v
    return F.col(k=k, hue=hue, chroma=chroma, **out)


def build(name, aspect, seed=0, variant=0):
    """The design `name` for a printed area `aspect` wide (x height), colourway `variant`."""
    global _PERM
    if name not in STYLES:
        raise KeyError(f"poster style {name!r} (have {sorted(STYLES)})")
    _PERM = VARIANTS[int(variant) % len(VARIANTS)]
    try:
        return STYLES[name](float(aspect), seed, variant)
    finally:
        _PERM = {}


PAPER = col(paper=.94, gold=.06)                    # warm paper white
PAPER_WORN = col(paper=.90, gold=.10)               # a creased, rubbed edge
FOLDS = ("none", "cross", "thirds")


def fold_lines(sheet_h, folds):
    """Crease positions (xs, ys) in design units: `cross` = one vertical and one horizontal fold in the middle,
    `thirds` = two horizontal letter folds."""
    if folds == "cross":
        return [0.0], [0.0]
    if folds == "thirds":
        return [], [sheet_h / 6.0, -sheet_h / 6.0]
    return [], []


def finish(pr, a, sheet_a, sheet_h, folds="cross", torn=()):
    """The sheet's surface from a `Print`: the print on warm paper (a margin is paper), worn along the fold lines, torn
    corners, and the surface channels. `a` = the printed area's aspect, `sheet_a`, `sheet_h` = the sheet's width and
    height in printed heights. Returns expressions {albedo, rough, spec, height, alpha}."""
    ink = cov(box_d(a / 2, 0.5))
    xs, ys = fold_lines(sheet_h, folds)
    creases = [remap(fabs(sub(X, x0)), 0.0022, 0.0, 0.0, 1.0, smooth=True) for x0 in xs] + \
              [remap(fabs(sub(Y, y0)), 0.0022, 0.0, 0.0, 1.0, smooth=True) for y0 in ys]
    fold = union(*creases) if creases else None
    albedo = cmix(PAPER, pr.color, ink)
    rough = remap(mul(ink, pr.gloss), 0.0, 1.0, 0.86, 0.52)
    spec = remap(mul(ink, pr.gloss), 0.0, 1.0, 0.24, 0.50)
    height = c(0.0)
    if fold is not None:
        albedo = cmix(albedo, PAPER_WORN, mul(fold, 0.09))
        rough = add(rough, mul(fold, 0.08))
        height = sub(0.0, fold)
    alpha = c(1.0)
    hw, hh = sheet_a / 2, sheet_h / 2
    for tag in torn:
        sx, sy = (-1.0 if tag[1] == "L" else 1.0), (1.0 if tag[0] == "T" else -1.0)
        r = 0.30 * min(hw, hh)                                          # leg of the missing triangle
        jag = add(mul(add(sin(mul(X, 170.0)), sin(add(mul(Y, 130.0), 0.7))), 0.0026), mul(sin(add(mul(X, 37.0 * sy), mul(Y, 29.0 * sx))), 0.005))
        d = add(sub(add(mul(X, sx), mul(Y, sy)), hw + hh - r), jag)
        alpha = mul(alpha, cov(d))
        albedo = cmix(albedo, PAPER, remap(d, -0.0075, -0.0020, 0.0, 1.0))     # bare paper fibres along the tear
    return {"albedo": albedo, "rough": rough, "spec": spec, "height": height, "alpha": alpha, "ink": ink}


def frame(img, a, inset, w, colour):
    """A thin frame line `inset` inside the print's edge."""
    return over(img, colour, cov(sub(fabs(box_d(a * 0.5 - inset, 0.5 - inset)), w * 0.5)))


CHROME = [(0.00, col(text=1.0)), (0.24, col(text=.55, foam=.45)), (0.46, col(pine=.55, iris=.25, overlay=.20)),
          (0.50, col(overlay=.70, base=.30)), (0.515, col(gold=.75, text=.25)), (0.74, col(love=.60, gold=.40)),
          (1.00, col(iris=.65, love=.35))]


def chrome_title(img, row, stops=CHROME, shadow=None, outline=None, ow=0.0022, offs=(0.006, -0.006)):
    """A BlockRow in chrome: an optional drop shadow and outline, then the vertical gradient `stops`."""
    if shadow is not None:
        img = over(img, shadow, row.mask(at(P0, offs[0], offs[1]), ow, windows=False))
    if outline is not None:
        img = over(img, outline, row.mask(P0, ow, windows=False))
    return over(img, cramp(row.fill_t(), stops), row.mask())


def mountains(img, a, yh, seed, hi, far, near, ridge_col=None):
    """A faceted ridge standing on the horizon `yh` (a valley in the middle): facets that face the middle of the
    print are `near` coloured (lit), the others `far`; optional glowing ridge line."""
    n = max(8, int(round(22 * a)))
    pts = ridge(seed, n, 0.0, hi, valley=0.5, width=0.26, sink=0.05)
    height = profile(X, -a / 2, a / 2, [(t, yh + h) for t, h in pts])
    stops = []
    for i in range(n):
        (t0, h0), (t1, h1) = pts[i], pts[i + 1]
        stops.append((t0, 1.0 if (h1 - h0) * ((t0 + t1) / 2 - 0.5) > 0 else 0.0))
    stops.append((1.0, stops[-1][1]))
    lit = ramp(remap(X, -a / 2, a / 2, 0.0, 1.0), stops, "CONSTANT")
    img = over(img, cmix(far, near, lit), cov(sub(Y, height)))
    if ridge_col is not None:
        img = over(img, ridge_col, cov(sub(fabs(sub(Y, height)), 0.0022)))
    return img


@style
def sunset_grid(a, seed=0, variant=0):
    """Synthwave: a striped sun setting behind faceted mountains over a perspective neon grid, chrome title."""
    yh = -0.10                                              # horizon
    R = min(0.23, 0.36 * a)                                 # sun radius
    ysun = yh - 0.02 + R
    img = cramp(grad_y(yh, 0.5), [
        (0.00, col(gold=.62, rose=.28, love=.10)),
        (0.045, col(love=.80, gold=.20)),
        (0.16, col(love=.46, iris=.30, overlay=.24)),
        (0.40, col(iris=.24, overlay=.58, love=.08, pine=.10)),
        (1.00, col(overlay=.84, pine=.09, iris=.07)),
    ])
    stars = sprinkle(P0, 0.055, 0.0042, keep=0.38, seed=3)
    img = over(img, col(text=1.0), mul(stars, remap(Y, yh + 0.20, yh + 0.42, 0.0, 0.9)))
    ps = at(P0, 0.0, ysun)
    img = cadd(img, col(gold=.5, love=.5, k=.5), mul(remap(len2(*ps), R * 0.9, R * 2.0, 1.0, 0.0, smooth=True), 0.45))
    n = 8                                                   # the sun's gaps: thicker toward the bottom
    gaps = bands(Y, yh - 0.03, ysun, [(ysun - R * (0.10 + 0.82 * (k / (n - 1)) ** 0.9) - R * (0.02 + 0.085 * (k / (n - 1)) ** 1.1) / 2,
                                       ysun - R * (0.10 + 0.82 * (k / (n - 1)) ** 0.9) + R * (0.02 + 0.085 * (k / (n - 1)) ** 1.1) / 2)
                                      for k in range(n)])
    sun_fill = cramp(grad_y(ysun + R * 0.95, yh + 0.01), [(0.0, col(gold=.78, text=.22, k=1.05)), (0.5, col(gold=.5, love=.5)),
                                                          (1.0, col(love=.92, iris=.08))])
    img = over(img, sun_fill, mul(disc(R, ps), inv(gaps)))
    img = mountains(img, a, yh, 11 + seed, 0.15, col(overlay=.72, iris=.14, love=.14), col(iris=.28, love=.34, overlay=.38))
    img = mountains(img, a, yh, 23 + seed, 0.085, col(overlay=.90, base=.10), col(overlay=.62, iris=.18, love=.20),
                    col(foam=.7, iris=.3))
    t = sub(yh, Y)                                          # depth below the horizon
    img = over(img, cramp(remap(t, 0.0, yh + 0.5, 0.0, 1.0), [
        (0.0, col(love=.60, iris=.25, overlay=.15)), (0.14, col(iris=.18, overlay=.72, love=.10)),
        (1.0, col(overlay=.92, iris=.08))]), cov(sub(Y, yh)))
    c0 = 0.62                                               # horizontal lines at equal steps of depth c0 / t
    dh = mul(fabs(sub(fract(add(div(c0, t), 0.5)), 0.5)), mul(t, t) / c0)
    kx = 5.5                                                # radial lines: x / t in steps of 1 / kx
    dv = mul(fabs(sub(fract(add(mul(div(X, t), kx), 0.5)), 0.5)), t / kx)
    fade = mul(remap(t, 0.006, 0.07, 0.0, 1.0), cov(sub(Y, yh)))
    hw_h, hw_v = add(0.0012, mul(t, 0.004)), add(0.0010, mul(t, 0.0035))
    halo = union(remap(div(dh, hw_h), 7.0, 0.0, 0.0, 1.0, smooth=True), remap(div(dv, hw_v), 7.0, 0.0, 0.0, 1.0, smooth=True))
    img = cadd(img, col(iris=.6, love=.4, k=.8), mul(mul(halo, fade), 0.34))
    img = over(img, col(foam=.85, text=.15), mul(union(cov(sub(dh, hw_h)), cov(sub(dv, hw_v))), fade))
    img = over(img, col(gold=.5, text=.5), cov(sub(fabs(sub(Y, yh)), 0.0028)))
    row = BlockRow(0.0, 0.405, 0.05, [0.06, 0.046, 0.075, 0.05, 0.062, 0.042, 0.07], gap=0.011, seed=5 + seed)
    img = chrome_title(img, row, outline=col(base=.6, overlay=.4), shadow=col(love=1.0))
    img = frame(img, a, 0.024, 0.0030, col(gold=.8, text=.2))
    return Print(img, 0.5)



def stick(img, d, x, y, deg, fill, outline=None, ow=0.0035, shadow=None, off=(0.008, -0.008)):
    """A sticker: the signed-distance shape `d` placed at (x, y) turned `deg`, with an optional drop shadow and
    outline, filled with the colour expression `fill` (build patterned fills on the same frame, `at(P0, x, y, deg)`)."""
    p = at(P0, x, y, deg)
    if shadow is not None:
        img = over(img, shadow, cov(d(at(P0, x + off[0], y + off[1], deg))))
    if outline is not None:
        img = over(img, outline, cov(sub(d(p), ow)))
    return over(img, fill, cov(d(p)))


MEMPHIS = (   # field, and the accents in order of weight
    (col(foam=.55, pine=.25, paper=.20), col(love=1.0), col(gold=1.0), col(iris=1.0), col(rose=.7, paper=.3)),
    (col(rose=.62, paper=.28, iris=.10), col(pine=1.0), col(gold=1.0), col(iris=1.0), col(foam=.7, paper=.3)),
)


@style
def memphis(a, seed=0, variant=0):
    """Memphis: a pale field, bold flat shapes with pattern fills and drop shadows, squiggles and confetti."""
    field, A, B, C, D = MEMPHIS[variant % len(MEMPHIS)]
    INK, WHITE, SH = col(ink=.85, pine=.15), col(paper=1.0), col(ink=.8, pine=.2)
    hw = a / 2
    u = min(a, 1.0) / 0.714                                       # sizes follow the narrower side
    img = field
    img = over(img, col(pine=.55, ink=.45), mul(sprinkle(P0, 0.05, 0.0048, keep=0.55, seed=1, jitter=0.28, grow=0.7), 0.5))
    img = over(img, D, sprinkle(P0, 0.064, 0.0072, keep=0.42, seed=2, shape="ring"))
    img = over(img, A, sprinkle(P0, 0.074, 0.0085, keep=0.4, seed=4, shape="plus"))
    img = over(img, B, sprinkle(P0, 0.081, 0.0085, keep=0.35, seed=7, shape="diamond"))
    # arcs rising from the bottom edge
    for r, cc in ((0.215, INK), (0.17, A), (0.125, B), (0.08, WHITE), (0.035, C)):
        img = over(img, cc, ring(r * u, 0.04 * u, at(P0, 0.15 * hw, -0.5)))
    # squiggle and zigzag strokes
    img = over(img, INK, mul(wave(at(P0, 0.0, -0.215), 0.10 * u, 0.022 * u, 0.014 * u), band(X, -0.95 * hw, 0.2 * hw)))
    img = over(img, B, mul(zigzag(at(P0, 0.0, 0.445), 0.075 * u, 0.016 * u, 0.013 * u), band(X, -0.55 * hw, 0.95 * hw)))
    # stickers
    x, y = -0.46 * hw, 0.29                                       # disc with zigzag lines
    img = stick(img, D_disc(0.125 * u), x, y, 0.0,
                cmix(A, WHITE, zigzag(repeat_y(at(P0, x, y), 0.04 * u), 0.06 * u, 0.009 * u, 0.0075 * u)),
                outline=INK, shadow=SH)
    x, y = 0.42 * hw, 0.31                                        # rounded rectangle with hex dots
    img = stick(img, D_box(0.105 * u, 0.14 * u, 0.025 * u), x, y, -12.0,
                cmix(C, WHITE, dots(at(P0, x, y, -12.0), 0.034 * u, 0.0075 * u, stagger=True)), outline=INK, shadow=SH)
    x, y = -0.16 * hw, 0.03                                       # diagonally striped bar
    img = stick(img, D_box(0.20 * u, 0.032 * u), x, y, 7.0,
                cmix(INK, WHITE, stripes(add(X, Y), 0.036 * u, 0.5)), outline=INK, shadow=SH)
    x, y = 0.36 * hw, -0.06                                       # triangle with dots
    img = stick(img, D_tri(0.30 * u, 12.0), x, y, 0.0,
                cmix(B, A, dots(at(P0, x, y, 12.0), 0.04 * u, 0.008 * u)), outline=INK, shadow=SH)
    x, y = -0.5 * hw, -0.35                                       # checker square
    img = stick(img, D_box(0.085 * u, 0.085 * u), x, y, -8.0,
                cmix(INK, WHITE, checker(at(P0, x, y, -8.0), 0.034 * u)), outline=INK, shadow=SH)
    x, y = 0.55 * hw, -0.37                                       # grid of dots on a block
    img = stick(img, D_box(0.1 * u, 0.1 * u, 0.02 * u), x, y, 9.0,
                cmix(D, INK, dots(at(P0, x, y, 9.0), 0.042 * u, 0.009 * u)), outline=INK, shadow=SH)
    return Print(img, 0.6)


@style
def trio(a, seed=0, variant=0):
    """Trio: a circle, a triangle and a square mixing light over concentric rings, a chrome logotype and small print."""
    hw = a / 2
    u = min(a, 1.0) / 0.714
    yc = 0.15
    img = cramp(grad_y(0.5, -0.5), [(0.0, col(overlay=.66, pine=.20, iris=.14, fixed=True)),
                                    (0.55, col(overlay=.64, iris=.26, pine=.10, fixed=True)),
                                    (1.0, col(overlay=.62, iris=.18, love=.20, fixed=True))])
    pc = at(P0, 0.0, yc)
    img = cadd(img, col(iris=.6, love=.4, k=.55), mul(remap(len2(*pc), 0.40 * u, 0.04 * u, 0.0, 1.0, smooth=True), 0.55))
    img = over(img, col(iris=.8, text=.2), mul(stripes(len2(*pc), 0.036 * u, 0.06), mul(remap(len2(*pc), 0.5, 0.2, 0.0, 1.0), 0.42)))
    shapes = [(D_disc(0.15 * u), -0.075 * u, yc - 0.03 * u, 0.0, col(love=1.0)),
              (D_tri(0.40 * u), 0.075 * u, yc - 0.025 * u, 0.0, col(gold=1.0)),
              (D_box(0.09 * u, 0.09 * u, 0.012 * u), 0.0, yc + 0.075 * u, 45.0, col(foam=.8, pine=.2))]
    for d, x, y, deg, cc in shapes:
        img = cadd(img, cc, mul(cov(d(at(P0, x, y, deg))), 0.55))
    for d, x, y, deg, cc in shapes:
        img = over(img, col(text=1.0), mul(cov(sub(fabs(d(at(P0, x, y, deg))), 0.0016)), 0.75))
    for x, y, s in ((-0.62 * hw, 0.38, 0.032), (0.7 * hw, 0.05, 0.022), (0.5 * hw, 0.44, 0.016), (-0.72 * hw, -0.02, 0.014)):
        img = over(img, col(text=1.0), sparkle(s, at(P0, x, y)))
    row = BlockRow(0.0, -0.25, 0.092, [0.085, 0.07, 0.11, 0.075, 0.095, 0.07], gap=0.014, slant=14.0, seed=9 + seed, windows=0.2)
    img = chrome_title(img, row, outline=col(ink=.7, overlay=.3), shadow=col(love=1.0), ow=0.003, offs=(0.008, -0.008))
    row2 = BlockRow(0.0, -0.345, 0.016, [0.07, 0.04, 0.09, 0.05, 0.03, 0.08, 0.06, 0.04], gap=0.012, slant=14.0, seed=2 + seed,
                    cut=0.0, windows=0.0)
    img = over(img, col(gold=.8, text=.2), row2.mask())
    row3 = BlockRow(0.0, -0.41, 0.010, [0.03, 0.05, 0.02, 0.06, 0.03, 0.04, 0.05, 0.02, 0.04, 0.06, 0.03, 0.05, 0.04], gap=0.009,
                    slant=0.0, seed=4 + seed, cut=0.0, windows=0.0)
    img = over(img, col(subtle=.8, text=.2), row3.mask())
    img = frame(img, a, 0.024, 0.0026, col(iris=.7, text=.3))
    return Print(img, 0.55)


# Roofline of the low wedge coupe in car units (x along the car, nose at +0.5; height above the ground, length 1).
CAR_TOP = [(-0.500, 0.036), (-0.497, 0.150), (-0.468, 0.172), (-0.400, 0.183), (-0.315, 0.190), (-0.200, 0.200),
           (-0.110, 0.216), (-0.030, 0.238), (0.040, 0.250), (0.100, 0.246), (0.170, 0.224), (0.250, 0.200),
           (0.285, 0.186), (0.400, 0.140), (0.500, 0.088)]
CAR_WHEELS = (-0.315, 0.285)
TYRE_R, ARCH_R = 0.075, 0.089


def draw_car(img, x0, yg, S):
    """The coupe in side view (nose right) standing on the ground line y = yg, S long: dark body with a lit roof edge
    and character line, tinted glass, wheels, head and tail light."""
    p = at(P0, x0, yg, 0.0, S)
    px, py = p
    cv = lambda d: cov(d, SOFT / S)                              # distances in car units
    top = ramp(remap(px, -0.5, 0.5, 0.0, 1.0), [(x + 0.5, h) for x, h in CAR_TOP])
    arches = union(*[cv(sub(len2(*at(p, wx, TYRE_R)), ARCH_R)) for wx in CAR_WHEELS])
    body = mul(cv(fmax(sub(0.036, py), sub(py, top), sub(fabs(px), 0.5))), inv(arches))
    paint = cramp(remap(py, 0.036, 0.25, 0.0, 1.0), [(0.0, col(ink=.8, overlay=.2)), (0.5, col(overlay=.85, iris=.15)),
                                                      (1.0, col(overlay=.5, iris=.5))])
    img = over(img, col(ink=.7, overlay=.3), mul(disc(S * 0.5, at(P0, x0, yg - 0.004, 0.0, (0.62, 0.05))), 0.55))   # shadow
    img = over(img, paint, body)
    # roof edge light and a character line that dives toward the nose
    img = over(img, col(gold=.6, rose=.4), mul(mul(body, cv(sub(fabs(sub(py, sub(top, 0.006))), 0.0035))), 0.95))
    crease = ramp(remap(px, -0.5, 0.5, 0.0, 1.0), [(0.0, 0.125), (0.5, 0.112), (1.0, 0.078)])
    img = over(img, col(rose=.7, iris=.3), mul(mul(body, cv(sub(fabs(sub(py, crease)), 0.0022))), 0.6))
    # glass: under the roof line between a sloping C pillar and the windshield, with a B pillar and a reflection
    glass_d = fmax(sub(0.165, py), sub(py, sub(top, 0.014)), sub(sub(mul(sub(py, 0.165), 1.75), 0.13), px), sub(px, 0.225))
    glass = mul(cv(glass_d), inv(cv(sub(fabs(sub(px, 0.055)), 0.008))))
    tint = cramp(remap(py, 0.165, 0.24, 0.0, 1.0), [(0.0, col(pine=.5, overlay=.5)), (1.0, col(overlay=.85, pine=.15))])
    img = over(img, tint, glass)
    img = over(img, col(foam=.7, text=.3), mul(mul(glass, stripes(sub(px, mul(py, 0.9)), 0.13, 0.16, 0.5)), 0.55))
    # door line, wheels
    img = over(img, col(ink=.9, overlay=.1), mul(mul(body, cv(sub(fabs(sub(px, 0.17)), 0.0022))), band(py, 0.045, 0.165)))
    for wx in CAR_WHEELS:
        w = at(p, wx, TYRE_R)
        r = len2(*w)
        img = over(img, col(ink=.9, base=.1), cv(sub(r, TYRE_R)))
        img = over(img, col(subtle=.6, text=.4), cv(sub(r, 0.047)))
        img = over(img, col(overlay=.9, ink=.1), cv(sub(r, 0.040)))
        img = over(img, col(subtle=.5, text=.5), cv(sub(r, 0.016)))
        img = over(img, col(ink=.9, base=.1), cv(sub(r, 0.007)))
    # lights with a little glow
    for wx, wy, cc, rr in ((0.468, 0.108, col(gold=.5, text=.5), 0.011), (-0.492, 0.125, col(love=.85, text=.15), 0.014)):
        img = cadd(img, cc, mul(remap(len2(*at(p, wx, wy)), 0.075, 0.0, 0.0, 1.0, smooth=True), 0.55))
        img = over(img, cc, cv(sub(len2(*at(p, wx, wy)), rr)))
    return img


@style
def car(a, seed=0, variant=0):
    """A low wedge coupe on a road against banded sunset stripes and a pale sun, chrome title."""
    hw = a / 2
    yh = -0.16                                                    # horizon
    starts = (0.0, 0.060, 0.125, 0.195, 0.270, 0.350, 0.435, 0.520, 0.620)          # stripe starts above the horizon
    bands_c = [col(gold=.78, text=.22), col(gold=.45, rose=.55), col(rose=.72, love=.28), col(love=.88, iris=.12),
               col(love=.42, iris=.58), col(iris=.68, overlay=.32), col(iris=.32, pine=.28, overlay=.40),
               col(pine=.42, overlay=.58), col(overlay=.78, pine=.22)]
    img = cramp(remap(Y, yh, 0.5, 0.0, 1.0), [(s / (0.5 - yh), cc) for s, cc in zip(starts, bands_c)], interp="CONSTANT")
    sun_c = at(P0, 0.10 * hw, yh + 0.10)
    img = over(img, cmix(img, col(paper=1.0), 0.50), disc(0.24, sun_c))
    # road: dark, with the sky's colours mirrored faintly, a lane line
    road = cramp(remap(sub(yh, Y), 0.0, yh + 0.5, 0.0, 1.0), [(0.0, col(overlay=.9, iris=.1)), (1.0, col(ink=.6, overlay=.4))])
    refl = cramp(remap(sub(yh, Y), 0.0, 0.50, 0.0, 1.0), [(0.0, col(love=.7, gold=.3, k=.9)), (0.35, col(iris=.7, love=.3, k=.6)),
                                                          (1.0, col(overlay=1.0))])
    img = over(img, cmix(road, refl, 0.34), cov(sub(Y, yh)))
    img = over(img, col(gold=.6, text=.4), mul(cov(sub(fabs(sub(Y, yh)), 0.0022)), 0.9))
    for yy, ww in ((-0.405, 0.012), (-0.465, 0.016)):
        img = over(img, col(iris=.6, text=.4), mul(stripes(X, 0.16, 0.45, 0.0), band(Y, yy - ww / 2, yy + ww / 2)))
    S = min(0.54 + 0.3 * max(0.0, a - 1.0), 0.76 * a)
    x0, yg = -0.03 * hw, -0.33
    for k, (dy, ln) in enumerate(((0.05, 0.22), (0.095, 0.30), (0.14, 0.17))):             # speed lines behind the tail
        xe = x0 - S * 0.54
        img = over(img, col(text=.7, rose=.3), mul(line((xe - ln, yg + dy), (xe, yg + dy), 0.0045), 0.55))
    img = draw_car(img, x0, yg, S)
    row = BlockRow(0.0, 0.43, 0.05, [0.07, 0.05, 0.085, 0.06, 0.07, 0.05], gap=0.012, slant=12.0, seed=21 + seed, windows=0.2)
    img = chrome_title(img, row, outline=col(ink=.7, overlay=.3), shadow=col(love=.9, iris=.1))
    img = frame(img, a, 0.024, 0.0030, col(text=.8, gold=.2))
    return Print(img, 0.55)


def palm(img, base, crown, bend, scale, fronds, colour):
    """A palm silhouette: a tapering curved trunk from `base` (x, y) to `crown` (x, y) bulging by `bend` sideways, and
    leaf fronds [(angle deg, length, droop)] at the crown, `scale` times the usual size."""
    (bx, by), (cx, cy) = base, crown
    t = remap(Y, by, cy, 0.0, 1.0)
    xt = add(add(bx, mul(t, cx - bx)), mul(mul(t, sub(1.0, t)), bend))
    wt = sub(0.020 * scale, mul(t, 0.010 * scale))
    img = over(img, colour, mul(cov(sub(fabs(sub(X, xt)), wt)), band(Y, by - 0.05, cy + 0.004)))
    for deg, length, droop in fronds:
        lx, ly = at(P0, cx, cy, deg)
        L = length * scale
        side = -1.0 if math.cos(math.radians(deg)) < 0 else 1.0
        yc = mul(mul(lx, lx), -side * droop / scale)
        u = mul(lx, 1.0 / L)
        half = mul(mul(u, sub(1.0, u)), 4.0 * 0.0185 * scale)
        d = fmax(sub(fabs(sub(ly, yc)), half), sub(0.0, lx), sub(lx, L))
        img = over(img, colour, cov(d))
    return img


PALM_FRONDS = [(172, 0.21, 1.7), (150, 0.23, 1.5), (128, 0.21, 1.1), (106, 0.17, 0.7), (84, 0.15, 0.4), (64, 0.17, 0.7),
               (44, 0.21, 1.1), (24, 0.23, 1.5), (4, 0.21, 1.7), (198, 0.15, 2.2), (-22, 0.15, 2.2)]


@style
def sunburst(a, seed=0, variant=0):
    """A banded sun on the horizon with rays, glints on the sea and palms in silhouette."""
    hw = a / 2
    yh = -0.10
    pc = at(P0, 0.0, yh)
    r = len2(*pc)
    rayc = cmix(col(gold=.58, rose=.42), col(rose=.55, love=.45), rays(pc, 30, 0.5, 0.0))
    img = cmix(rayc, col(love=.35, iris=.65), remap(r, 0.18, 0.80, 0.0, 0.85, smooth=True))
    img = cadd(img, col(gold=.6, text=.4, k=.7), mul(remap(r, 0.40, 0.04, 0.0, 1.0, smooth=True), 0.5))
    for rr, cc in ((0.215, col(love=.92, iris=.08)), (0.175, col(rose=.65, love=.35)), (0.135, col(gold=.62, rose=.38)),
                   (0.095, col(gold=.55, text=.45)), (0.055, col(paper=1.0))):
        img = over(img, cc, disc(rr, pc))
    # sea with the sun's glints
    sea = cramp(remap(sub(yh, Y), 0.0, yh + 0.5, 0.0, 1.0), [(0.0, col(rose=.5, love=.3, iris=.2)), (0.12, col(iris=.5, pine=.5)),
                                                              (0.5, col(pine=.5, overlay=.5)), (1.0, col(overlay=.85, pine=.15))])
    img = over(img, sea, cov(sub(Y, yh)))
    for k in range(9):
        yk = yh - 0.018 - 0.011 * k - 0.0025 * k * k
        wk = 0.20 * (1.0 - k / 10.0) ** 0.7
        img = over(img, col(gold=.6, text=.4), mul(cov(D_box(wk, 0.0028 + 0.0004 * k, 0.002)(at(P0, 0.0, yk))), 0.9))
    img = over(img, col(gold=.8, text=.2), mul(cov(sub(fabs(sub(Y, yh)), 0.0018)), 0.8))
    ink = col(overlay=.9, ink=.1)
    img = palm(img, (-0.33 * hw - 0.02, -0.5), (-0.20 * hw - 0.03, 0.12), -0.05, 1.0, PALM_FRONDS, ink)
    img = palm(img, (0.62 * hw, -0.5), (0.52 * hw, -0.02), 0.04, 0.66, PALM_FRONDS[:7], ink)
    img = frame(img, a, 0.024, 0.0030, col(text=.8, gold=.2))
    return Print(img, 0.55)


# ================================================================= surface patterns of the furniture (object metres)
def drawer_panel(hw, hh):
    """Memphis drawer front for a panel 2 hw x 2 hh m about the origin of the object's (x, z): a pink field, an iris
    triangle, a gold zigzag band, foam dots and a few chips. Colour expression over (X, Y) = the panel's (x, z)."""
    INK = col(ink=.85, pine=.15)
    img = col(love=.74, rose=.26)
    img = over(img, col(paper=1.0), mul(sprinkle(P0, 0.034, 0.0034, keep=0.5, seed=5, jitter=0.28, grow=0.6), 0.9))
    img = over(img, col(gold=.92, rose=.08), mul(zigzag(at(P0, 0.01 * hw, -0.003), 0.052, 0.0125, 0.0105),
                                               band(X, -0.30 * hw, 0.50 * hw)))
    img = over(img, INK, mul(zigzag(at(P0, 0.01 * hw, 0.025), 0.052, 0.0125, 0.0042), band(X, -0.30 * hw, 0.50 * hw)))
    img = stick(img, D_tri(0.082, 0.0), -0.64 * hw, -0.002, 0.0, col(iris=1.0), outline=INK, ow=0.0025,
                shadow=INK, off=(0.0035, -0.0035))
    m = cov(D_box(0.17 * hw, 0.74 * hh, 0.007)(at(P0, 0.72 * hw, 0.0)))
    img = over(img, col(foam=.85, paper=.15), m)
    img = over(img, INK, mul(dots(at(P0, 0.72 * hw, 0.0), 0.0165, 0.0046, 0.0, True), m))
    return img


def laminate():
    """Pale terrazzo laminate for a table top (object metres x, y): chips of four sizes and colours."""
    img = col(paper=.80, rose=.13, iris=.07)
    for cell, r, cc, seed, shape in ((0.021, 0.0040, col(pine=.9, ink=.1), 1, "disc"), (0.033, 0.0050, col(love=1.0), 2, "disc"),
                                     (0.028, 0.0034, col(gold=1.0), 3, "diamond"), (0.043, 0.0060, col(iris=1.0), 4, "disc"),
                                     (0.019, 0.0030, col(ink=1.0), 6, "disc")):
        img = over(img, cc, sprinkle(P0, cell, r, keep=0.42, seed=seed, shape=shape, jitter=0.28, grow=0.8))
    return img
