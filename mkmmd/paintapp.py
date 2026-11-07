"""The toy paint app on the finished pictures (docs/design.md: Frame effects, Toy paint app): the picture work of the
`[[fx]]` kind `paintapp` that mkmmd.core.paintapp checks, times and lays out. PaintAppFx is the part of
mkmmd.framefx.FrameFx that runs it.

All of the app's art is drawn here in code from the palette (no picture of any real program in it): the chrome (the
border round the canvas, its outline and the shade it casts in, the toolbar and the palette strip as raised panels), the
buttons (a face over a lip, pressed down when picked or clicked) with their icons (a pencil, a crayon, a rubber stamp, a
paint can pouring, a two-tone eraser block, a curling undo arrow), the swatches, the cursor (a gloved hand pointing or a
pencil) and the sparkles. Vector parts are painted SUPER times larger and shrunk, each union wearing a thick ink
outline. The chrome is drawn once per output and kept; the buttons and swatches as tiles per state.

The paint goes on in the order it landed: a fill floods the canvas as it stands then (the frame as it reached the
effect, the fills and stamps before it on it), found again on every frame from its point at a fraction of the frame's
size; a stamp is a sticker (mkmmd.mandala: its die-cut edge, its shadow, its googly eyes through mkmmd.eyes). An eraser
holds the frame before its cut as it reached the effect, the paint of that frame on it (PictureFx._reached: kept as
mk post goes through the frames, worked out again for mk look), and takes it off the incoming frames: rubbed out along
its strokes, or cut into confetti that lift off and fly, each a tile of the held picture (its back a flat colour) laid
by an affine map. Images are float RGB in display space (0..1), (h, w, 3); a coverage is (h, w) in 0..1."""
import math
from collections import OrderedDict

import cv2
import numpy as np

from . import eyes as EYD
from . import mandala as MDF
from .core import framefx as FX
from .core import mandala as MD
from .core import paintapp as PA
from .core import tween as TW

SUPER = 4                        # the art is painted this many times larger and shrunk
SHIFT = 4                        # fixed-point bits of the art's points (sub-pixel edges)
TILES = 96                       # sprite tiles kept from frame to frame
FLOOD_SIDE = 540.0               # px on the short side a fill's region is found at
FLOOD_BLUR = 1.0                 # px there: the picture smoothed first, so noise does not break a region
FLOOD_CLOSE = 5                  # px there: specks left inside a region are closed
FRONT = 8.0                      # px at 1080: the soft edge of a fill's spreading front
BLOB = 0.14                      # how far a fill's front wanders in and out of its circle, shares of its radius
OUTLINE = 6.0                    # px at 1080: the ink line round the buttons, the panels, the swatches, the icons
CURSOR_OUTLINE = 6.0             # px at 1080: round the cursor
SHADOW = (5.0, 9.0, 6.0, 0.28)   # the cursor's and the panels' shadow: offset (dx, dy) and blur px at 1080, opacity
CANVAS_SHADE = (7.0, 0.22)       # the shade the border casts on the canvas: how far in (px at 1080), how dark at its edge
LIP = 0.11                       # a button's lip, a share of its side: how far it goes down when pressed
HELD = 0.45                      # how far the button of the tool in hand sits pressed down
ICON = 0.7                       # an icon's size on its button's face
SQUASH = (0.1, 0.18)             # the cursor pressed fully: this much wider and this much shorter, about its tip
RUB = 0.6                        # how far the cursor is pressed down while it rubs with an eraser
RING = (0.02, 0.06, 4.0)         # a click's ring: from and to (radii, short sides), its line (px at 1080)


def _ff():
    from . import framefx as FF          # framefx imports this module for PaintAppFx: its helpers are there by now
    return FF


def _mix(a, b, k):
    return tuple(float(x + (y - x) * k) for x, y in zip(a, b))


# ================================================================================================= vector art
def _rrect(x0, y0, x1, y1, r, n=8):
    """A rounded rectangle's outline (corners of radius `r`)."""
    r = max(min(r, 0.5 * (x1 - x0), 0.5 * (y1 - y0)), 0.0)
    out = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90.0), (x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0)):
        a = np.radians(a0 + np.linspace(0.0, 90.0, n))
        out.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
    return np.concatenate(out)


def _capsule(a, b, r, n=12):
    """A stadium round the segment a..b, `r` from it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ang = math.atan2(d[1], d[0])
    t1 = ang - math.pi / 2.0 + np.linspace(0.0, math.pi, n)
    t2 = ang + math.pi / 2.0 + np.linspace(0.0, math.pi, n)
    return np.concatenate([b + r * np.stack([np.cos(t1), np.sin(t1)], 1), a + r * np.stack([np.cos(t2), np.sin(t2)], 1)])


def _ell(cx, cy, rx, ry, n=40):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def draw(parts, scale, outline=0.0, ink=(0.0, 0.0, 0.0)):
    """Vector `parts` [(colour, kind, (n, 2) points, width)] in units, `scale` px a unit, as a tile: (rgb premultiplied
    (h, w, 3), coverage (h, w), origin (x, y): where the unit point (0, 0) falls, px). `kind` "fill" (a polygon), "line"
    (a polyline `width` units wide, round ends) or "loop" (a closed one). The parts are painted in order SUPER times
    larger and shrunk, and their union wears an `outline` px wide in `ink` (0: none)."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for _, kind, p, wd in parts:
        p = np.asarray(p, float)
        half = 0.0 if kind == "fill" else 0.5 * wd
        lo, hi = np.minimum(lo, p.min(0) - half), np.maximum(hi, p.max(0) + half)
    m = (outline + 2.0) / scale
    lo, hi = lo - m, hi + m
    W, H = int(math.ceil((hi[0] - lo[0]) * scale)), int(math.ceil((hi[1] - lo[1]) * scale))
    k = scale * SUPER
    rgb, a = np.zeros((H * SUPER, W * SUPER, 3), np.float32), np.zeros((H * SUPER, W * SUPER), np.float32)
    for colour, kind, p, wd in parts:
        q = np.rint((np.asarray(p, float) - lo) * k * (1 << SHIFT)).astype(np.int32)
        cov = np.zeros(a.shape, np.uint8)
        if kind == "fill":
            cv2.fillPoly(cov, [q], 255, cv2.LINE_AA, SHIFT)
        else:
            cv2.polylines(cov, [q], kind == "loop", 255, max(int(round(wd * k)), 1), cv2.LINE_AA, SHIFT)
        c = cov.astype(np.float32) / np.float32(255.0)
        rgb = rgb * (1.0 - c[..., None]) + np.asarray(colour, np.float32) * c[..., None]
        a = a + (1.0 - a) * c
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    a = np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    if outline > 0.0:
        ring = _ff()._grow(a, outline)
        rgb = rgb + np.asarray(ink, np.float32) * (ring * (1.0 - a))[..., None]
        a = a + ring - a * ring
    return (np.minimum(np.clip(rgb, 0.0, 1.0), a[..., None]), a,       # a pixel's middle: its samples' mean
            ((-lo[0] * k + 0.5) / SUPER - 0.5, (-lo[1] * k + 0.5) / SUPER - 0.5))


def _along(a, hw):
    """A point maker along an axis at `a` degrees counter-clockwise on screen: P(s, k), `s` along it, `k` across it in
    half widths `hw`."""
    d = np.array([math.cos(math.radians(a)), -math.sin(math.radians(a))])
    n = np.array([-d[1], d[0]])
    return lambda s, k=0.0: s * d + k * hw * n


def _pencil(art, ink, angle=45.0):
    """A yellow pencil along `angle`, its point toward the lower left (shares of a unit box)."""
    P = _along(angle, 0.12)
    gold = art["gold"]
    light, dark = _mix(gold, art["base"], 0.45), _mix(gold, ink, 0.18)
    wood, metal = _mix(gold, art["base"], 0.72), art["hl_high"]

    def quad(s0, s1, k0, k1):
        return [P(s0, k0), P(s1, k0), P(s1, k1), P(s0, k1)]
    end = [P(0.38, -1.0), P(0.44, -1.0)] + [P(0.44 + 0.07 * math.sin(a), -math.cos(a)) for a in
                                            np.linspace(0.0, math.pi, 9)[1:-1]] + [P(0.44, 1.0), P(0.38, 1.0)]
    return [(gold, "fill", quad(-0.2, 0.3, -1.0, 1.0), 0.0),
            (light, "fill", quad(-0.2, 0.3, -1.0, -0.33), 0.0),
            (dark, "fill", quad(-0.2, 0.3, 0.33, 1.0), 0.0),
            (wood, "fill", [P(-0.2, -1.0), P(-0.47, 0.0), P(-0.2, 1.0)], 0.0),
            (ink, "fill", [P(-0.38, -0.34), P(-0.47, 0.0), P(-0.38, 0.34)], 0.0),
            (art["rose"], "fill", end, 0.0),
            (metal, "fill", quad(0.3, 0.38, -1.05, 1.05), 0.0),
            (ink, "line", [P(0.325, -1.0), P(0.325, 1.0)], 0.014),
            (ink, "line", [P(0.355, -1.0), P(0.355, 1.0)], 0.014),
            (ink, "line", [P(-0.2, -0.33), P(0.3, -0.33)], 0.01),
            (ink, "line", [P(-0.2, 0.33), P(0.3, 0.33)], 0.01)]


def _crayon(art, ink, angle=45.0):
    """A wax crayon in its paper wrapper along `angle`, its tip toward the lower left."""
    P = _along(angle, 0.135)
    love = art["love"]
    paper, deep = _mix(love, art["base"], 0.62), _mix(love, ink, 0.15)

    def quad(s0, s1, k0=-1.0, k1=1.0):
        return [P(s0, k0), P(s1, k0), P(s1, k1), P(s0, k1)]
    tip = [P(-0.28, -1.0), P(-0.41, -0.42), P(-0.47, -0.22), P(-0.47, 0.22), P(-0.41, 0.42), P(-0.28, 1.0)]
    return [(love, "fill", quad(-0.28, 0.45), 0.0),
            (deep, "fill", tip, 0.0),
            (paper, "fill", quad(-0.16, 0.33), 0.0),
            (love, "fill", quad(-0.12, -0.07), 0.0),
            (love, "fill", quad(0.24, 0.29), 0.0),
            (_mix(paper, art["base"], 0.5), "line", [P(-0.12, -0.55), P(0.29, -0.55)], 0.03),
            (ink, "line", [P(-0.16, -1.0), P(-0.16, 1.0)], 0.012),
            (ink, "line", [P(0.33, -1.0), P(0.33, 1.0)], 0.012)]


def _stamp(art, ink):
    """A rubber stamp standing up: a round knob, a neck, a wooden block, the red rubber under it."""
    iris = art["iris"]
    wood = _mix(art["gold"], art["base"], 0.5)
    return [(_mix(iris, ink, 0.25), "fill", [(-0.075, -0.2), (0.075, -0.2), (0.1, 0.03), (-0.1, 0.03)], 0.0),
            (iris, "fill", _ell(0.0, -0.29, 0.17, 0.14), 0.0),
            (_mix(iris, art["base"], 0.55), "fill", _ell(-0.06, -0.33, 0.055, 0.035), 0.0),
            (wood, "fill", _rrect(-0.36, 0.0, 0.36, 0.24, 0.05), 0.0),
            (_mix(wood, art["base"], 0.5), "line", [(-0.28, 0.06), (0.22, 0.06)], 0.03),
            (art["love"], "fill", _rrect(-0.33, 0.22, 0.33, 0.34, 0.02), 0.0)]


def _bucket(art, ink):
    """A paint can tipped to the left, paint pouring from its lip into a drop."""
    foam = art["foam"]
    metal, label = art["hl_high"], _mix(foam, art["base"], 0.45)
    can = [(metal, "fill", _rrect(-0.24, -0.1, 0.24, 0.3, 0.05), 0.0),
           (label, "fill", [(-0.225, 0.02), (0.225, 0.02), (0.21, 0.17), (-0.21, 0.17)], 0.0),
           (metal, "fill", _ell(0.0, -0.1, 0.24, 0.075), 0.0),
           (foam, "fill", _ell(0.0, -0.095, 0.2, 0.052), 0.0),
           (ink, "line", np.stack([0.26 * np.cos(np.linspace(math.pi, 2 * math.pi, 16)),
                                   -0.1 + 0.3 * np.sin(np.linspace(math.pi, 2 * math.pi, 16))], 1), 0.03)]
    turn = lambda p: MD._turned(np.asarray(p, float), 32.0) + (0.1, -0.02)            # noqa: E731
    parts = [(c, k, turn(p), w) for c, k, p, w in can]
    lip = turn([(-0.24, -0.1)])[0]
    pour = MD._smooth([lip + (0.02, -0.02), lip + (-0.05, 0.06), (-0.33, 0.26), (-0.37, 0.4), (-0.27, 0.43),
                       (-0.25, 0.3), lip + (0.06, 0.07)], 8)
    return parts + [(foam, "fill", pour, 0.0),
                    (_mix(foam, art["base"], 0.6), "fill", _ell(-0.32, 0.36, 0.025, 0.035), 0.0)]


def _eraser(art, ink):
    """A two-tone eraser block in three-quarter view, tipped a little."""
    rose, foam = art["rose"], art["foam"]
    faces = [(rose, [(-0.42, -0.02), (0.0, -0.02), (0.0, 0.22), (-0.42, 0.22)]),
             (_mix(rose, art["base"], 0.45), [(-0.42, -0.02), (-0.3, -0.2), (0.12, -0.2), (0.0, -0.02)]),
             (foam, [(0.0, -0.02), (0.28, -0.02), (0.28, 0.22), (0.0, 0.22)]),
             (_mix(foam, art["base"], 0.45), [(0.0, -0.02), (0.12, -0.2), (0.4, -0.2), (0.28, -0.02)]),
             (_mix(foam, ink, 0.22), [(0.28, -0.02), (0.4, -0.2), (0.4, 0.04), (0.28, 0.22)])]
    parts = [(c, "fill", MD._turned(p, 12.0), 0.0) for c, p in faces]
    return parts + [(ink, "line", MD._turned([(0.0, -0.02), (0.0, 0.22)], 12.0), 0.012),
                    (ink, "line", MD._turned([(-0.42, -0.02), (0.28, -0.02), (0.4, -0.2)], 12.0), 0.012)]


def _undo(art, ink):
    """A thick arrow curling back over the top to the left, its head pointing down."""
    pine = art["pine"]
    c, r0, r1 = (0.04, 0.08), 0.17, 0.31
    th = np.radians(np.linspace(-35.0, 175.0, 40))
    ring = np.concatenate([np.stack([c[0] + r1 * np.cos(th), c[1] - r1 * np.sin(th)], 1),
                           np.stack([c[0] + r0 * np.cos(th[::-1]), c[1] - r0 * np.sin(th[::-1])], 1)])
    e = math.radians(175.0)
    rad, tan = np.array([math.cos(e), -math.sin(e)]), np.array([-math.sin(e), -math.cos(e)])
    base = np.asarray(c) + 0.5 * (r0 + r1) * rad
    head = [base + 0.27 * rad, base + 0.21 * tan, base - 0.27 * rad]
    shine = np.stack([c[0] + 0.27 * np.cos(th[8:30]), c[1] - 0.27 * np.sin(th[8:30])], 1)
    return [(pine, "fill", ring, 0.0), (pine, "fill", head, 0.0),
            (_mix(pine, art["base"], 0.45), "line", shine, 0.035)]


ICONS = {"pencil": _pencil, "crayon": _crayon, "stamp": _stamp, "bucket": _bucket, "eraser": _eraser, "undo": _undo}
ACCENT = {"pencil": "gold", "crayon": "love", "stamp": "iris", "bucket": "foam", "eraser": "rose", "undo": "pine"}


def _hand(art, ink):
    """A white glove pointing up and a little left, its fingertip at (0, 0), about a unit tall; a rose cuff."""
    glove = art["surface"]
    parts = [(glove, "fill", _capsule((-0.07, 0.52), (-0.22, 0.38), 0.07), 0.0),
             (glove, "fill", _rrect(-0.12, 0.34, 0.33, 0.8, 0.13), 0.0),
             (glove, "fill", _capsule((0.0, 0.075), (0.0, 0.5), 0.075), 0.0)]
    for y, x1 in ((0.42, 0.29), (0.535, 0.33), (0.65, 0.31)):
        parts += [(glove, "fill", _capsule((0.1, y), (x1, y), 0.058), 0.0),
                  (ink, "loop", _capsule((0.1, y), (x1, y), 0.058), 0.018)]
    parts += [(ink, "line", [(-0.075, 0.36), (-0.075, 0.5)], 0.016),
              (art["rose"], "fill", _rrect(-0.13, 0.79, 0.34, 0.96, 0.05), 0.0),
              (_mix(art["rose"], art["base"], 0.45), "line", [(-0.07, 0.84), (0.28, 0.84)], 0.03)]
    return [(c, k, MD._turned(np.asarray(p, float), 18.0), w) for c, k, p, w in parts]


def _pencil_cursor(art, ink):
    """The pencil, its point at (0, 0) toward the lower left, about a unit long."""
    parts = _pencil(art, ink)
    tip = np.array([math.cos(math.radians(45.0)), -math.sin(math.radians(45.0))]) * -0.47
    return [(c, k, (np.asarray(p, float) - tip) * 1.35, w * 1.35) for c, k, p, w in parts]


def _star(r=1.0, inner=0.3):
    """A four-pointed twinkle about (0, 0), points `r` out."""
    a = np.radians(np.arange(8) * 45.0)
    rad = np.where(np.arange(8) % 2 == 0, r, r * inner)
    return np.stack([rad * np.cos(a), -rad * np.sin(a)], 1)


# ================================================================================================= coverage of shapes
def _grid(box):
    x0, y0, x1, y1 = box
    return np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))


def rrect_sdf(xs, ys, box, r):
    """Signed distance (px; negative inside) of pixels (xs, ys) from a rounded rectangle `box` (x0, y0, x1, y1: its
    edges) with corners of radius `r`."""
    x0, y0, x1, y1 = box
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    r = min(r, hx, hy)
    qx, qy = np.abs(xs - cx) - (hx - r), np.abs(ys - cy) - (hy - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - r).astype(
        np.float32)


def _cov(d):
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def poly_cover(shape, pts, line=0.0):
    """The coverage (uint8, 0..255, OpenCV's antialiased fill at SHIFT bits) of polygon `pts` (px) on a picture of `shape`
    (h, w), with the coverage of an outline `line` px out from it (None: `line` 0): (x0, y0, outline, fill), the two
    covering the same box whose top left is pixel (x0, y0); None when the box misses the picture."""
    h, w = shape
    pad = line + 2.0
    x0, y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, w), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    q = np.rint((pts - (x0, y0)) * (1 << SHIFT)).astype(np.int32)
    outline = None
    if line > 0.0:
        outline = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(outline, [q], True, 255, max(int(round(2.0 * line)), 1), cv2.LINE_AA, SHIFT)
    fill = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(fill, [q], 255, cv2.LINE_AA, SHIFT)
    return x0, y0, outline, fill


def _between(a, b, mask):
    """A new picture: `a` where `mask` (uint8 coverage, 0..255) is 0, `b` where it is 255, mixed between."""
    if not mask.any():
        return np.array(a, np.float32, copy=True)
    out = np.subtract(b, a, dtype=np.float32)
    out *= (mask.astype(np.float32) / np.float32(255.0))[..., None]
    out += a
    return out


def _over(rgb, a, colour, cov):
    """Premultiplied `rgb`, `a` (in place) with `colour` laid over them where `cov` covers."""
    rgb *= (1.0 - cov)[..., None]
    rgb += np.asarray(colour, np.float32) * cov[..., None]
    a *= 1.0 - cov
    a += cov


def _edges(box):
    """A box's pixel edges (x0, y0, x1, y1) for pixel middles at whole numbers: its sides half a pixel out."""
    return box[0] - 0.5, box[1] - 0.5, box[2] - 0.5, box[3] - 0.5


def chrome(fx, lay):
    """The app's still chrome over a frame (its layout `lay`): (rgb premultiplied (h, w, 3), coverage (h, w), strips):
    the border round the canvas, the shade it casts in and the ink line along the canvas's edge, the toolbar and the
    palette strip as raised panels with their shadows (the buttons and the swatches are laid on each frame, by state);
    `strips` [(y0, y1, x0, x1)] the parts of the frame that hold all of it (the canvas inside its shade holds none)."""
    w, h = (int(round(v)) for v in lay["size"])
    k = lay["unit"]
    xs, ys = _grid((0, 0, w, h))
    rgb, a = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
    ink = fx["ink"]
    d = rrect_sdf(xs, ys, _edges(lay["canvas"]), lay["corner"])
    depth, dark = max(CANVAS_SHADE[0] * k, 1e-3), CANVAS_SHADE[1]
    shade = dark * np.exp(np.minimum(d, 0.0) / depth)
    _over(rgb, a, ink, np.where((d < 0.0) & (shade >= 1.0 / 1024.0), shade, 0.0).astype(np.float32))
    _over(rgb, a, fx["border"], _cov(-d))
    _over(rgb, a, ink, np.clip(_cov(-d) - _cov(lay["outline"] - d), 0.0, 1.0))
    ox, oy, blur, alpha = SHADOW
    for box in (lay["toolbar"], lay["strip"]):
        bx = _edges(box)
        r = 0.28 * min(bx[2] - bx[0], bx[3] - bx[1])
        sd = _cov(rrect_sdf(xs - ox * k, ys - oy * k, bx, r) - OUTLINE * k)
        _over(rgb, a, ink, (alpha * cv2.GaussianBlur(sd, (0, 0), max(blur * k, 0.5))).astype(np.float32))
        pd = rrect_sdf(xs, ys, bx, r)
        _over(rgb, a, ink, _cov(pd - OUTLINE * k))
        _over(rgb, a, fx["panel"], _cov(pd))
    m = max(depth * math.log(dark * 1024.0) + 2.0, lay["corner"] + 1.0)   # inside this the shade is under 1/1024
    x0, y0, x1, y1 = lay["canvas"]
    x0, y0 = int(math.ceil(x0 + m)), int(math.ceil(y0 + m))
    x1, y1 = int(math.floor(x1 - m)), int(math.floor(y1 - m))
    if x1 <= x0 or y1 <= y0 or a[y0:y1, x0:x1].any():
        return rgb, a, [(0, h, 0, w)]
    return rgb, a, [(0, y0, 0, w), (y1, h, 0, w), (y0, y1, 0, x0), (y0, y1, x1, w)]


def _cell(box, room):
    """A whole-pixel box round `box` (pixel middles at whole numbers), `room` px more each way: (x0, y0, x1, y1)."""
    return (int(math.floor(box[0] - room)), int(math.floor(box[1] - room)), int(math.ceil(box[2] + room)) + 1,
            int(math.ceil(box[3] + room)) + 1)


def button_tile(fx, lay, tool, box, active, down, icons):
    """An opaque tile of the panel round `tool`'s button (its `box`, px): (cell, rgb). The button: a face over a lip, in
    the panel's colour or, `active` (the tool in hand), a pastel of its accent; `down` 0..1 how far it is pressed onto
    its lip; the icon on its face (`icons`: a cache of drawn icons)."""
    k = lay["unit"]
    cell = _cell(box, 0.4 * PA.UI["spacing"] * k)
    xs, ys = _grid(cell)
    rgb = np.empty(xs.shape + (3,), np.float32)
    rgb[:] = fx["panel"]
    a = np.ones(xs.shape, np.float32)
    x0, y0, x1, y1 = _edges(box)
    side = x1 - x0
    lip, r = LIP * side, 0.22 * side
    face = _mix(fx["art"][ACCENT[tool]], fx["art"]["base"], 0.55) if active else fx["art"]["surface"]
    edge = _mix(face, fx["ink"], 0.35)
    shift = lip * max(min(down, 1.0), -0.5)
    under = rrect_sdf(xs, ys, (x0, y0 + lip, x1, y1), r)
    top = rrect_sdf(xs, ys, (x0, y0 + shift, x1, y1 - lip + shift), r)
    both = np.minimum(under, top)
    _over(rgb, a, fx["ink"], _cov(both - OUTLINE * k))
    _over(rgb, a, edge, _cov(under))
    _over(rgb, a, fx["ink"], _cov(top - 0.6 * OUTLINE * k))
    _over(rgb, a, face, _cov(top))
    size = ICON * side
    key = (tool, round(size, 2))
    if key not in icons:
        icons[key] = draw(ICONS[tool](fx["art"], fx["ink"]), size, 0.75 * OUTLINE * k, fx["ink"])
    irgb, ia, origin = icons[key]
    cx, cy = 0.5 * (x0 + x1) - cell[0], 0.5 * (y0 + y1 - lip) + shift - cell[1]
    M = np.float32([[1.0, 0.0, cx - origin[0]], [0.0, 1.0, cy - origin[1]]])
    size_ = (cell[2] - cell[0], cell[3] - cell[1])
    gi = cv2.warpAffine(np.dstack([irgb, ia]), M, size_, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
    rgb = rgb * (1.0 - gi[..., 3:4]) + gi[..., :3]
    return cell, rgb.astype(np.float32)


def swatch_tile(fx, lay, i, picked):
    """Swatch `i` as a sticker over the strip: (cell, rgb premultiplied, coverage): a blob of its colour with a gloss and
    an ink line, and when `picked` larger and ringed in white (it may reach over its neighbours: it is laid last)."""
    k = lay["unit"]
    cx, cy, r = lay["swatches"][i]
    R = r * (1.14 if picked else 1.0)
    ring = 0.5 * PA.UI["swatch_gap"] * k
    reach = R + ring + OUTLINE * k + 2.0
    cell = _cell((cx, cy, cx, cy), reach)
    xs, ys = _grid(cell)
    rgb, a = np.zeros(xs.shape + (3,), np.float32), np.zeros(xs.shape, np.float32)
    d = np.hypot(xs - cx, ys - cy) - R
    if picked:
        _over(rgb, a, fx["ink"], _cov(d - ring - OUTLINE * k))
        _over(rgb, a, (1.0, 1.0, 1.0), _cov(d - ring))
    _over(rgb, a, fx["ink"], _cov(d - OUTLINE * k))
    colour = fx["swatches"][i]
    _over(rgb, a, colour, _cov(d))
    g = np.hypot((xs - (cx - 0.35 * R)) / 0.32, (ys - (cy - 0.4 * R)) / 0.2) - R
    _over(rgb, a, _mix(colour, (1.0, 1.0, 1.0), 0.75), 0.8 * _cov(g * 0.25))
    return cell, rgb, a


# ================================================================================================= fills
def flood(img, seed, tolerance):
    """The colour region of `img` round pixel `seed` (x, y), found on the picture shrunk to FLOOD_SIDE on its short side
    and smoothed a little: every pixel joined to the seed through pixels within `tolerance` (0..1 on each channel) of its
    colour, specks inside it closed. (coverage there (h', w') in 0..1, the seed there (x, y) px, the shrink) or None when
    the seed is off the picture."""
    h, w = img.shape[:2]
    f = min(1.0, FLOOD_SIDE / float(min(w, h)))
    sw, sh = max(int(round(w * f)), 1), max(int(round(h * f)), 1)
    sx, sy = (seed[0] + 0.5) * sw / w - 0.5, (seed[1] + 0.5) * sh / h - 0.5
    ix, iy = int(round(sx)), int(round(sy))
    if not (0 <= ix < sw and 0 <= iy < sh):
        return None
    small = cv2.resize(np.ascontiguousarray(img, np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), FLOOD_BLUR)
    u8 = (np.clip(small, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    mask = np.zeros((sh + 2, sw + 2), np.uint8)
    tol = (float(tolerance) * 255.0,) * 3
    cv2.floodFill(u8, mask, (ix, iy), (0, 0, 0), tol, tol,
                  4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
    region = cv2.morphologyEx(mask[1:-1, 1:-1], cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FLOOD_CLOSE, FLOOD_CLOSE)))
    return region.astype(np.float32) / np.float32(255.0), (sx, sy), sw / float(w)


def spread(region, seed, u, soft, phase):
    """`region` as far as a fill's front has spread `u` (0..1) of the way out from `seed` (x, y) to the region's
    farthest pixel: a front `soft` px wide wandering BLOB in and out of its circle (its waves' `phase`, seeded) and
    settling round as it reaches the end; the whole region at 1."""
    if u >= 1.0:
        return region
    ys, xs = np.nonzero(region > 0.02)
    if not len(xs) or u <= 0.0:
        return np.zeros_like(region)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    gx, gy = _grid((x0, y0, x1, y1))
    dx, dy = gx - np.float32(seed[0]), gy - np.float32(seed[1])
    dist = np.hypot(dx, dy)
    reach = float(dist[region[y0:y1, x0:x1] > 0.02].max()) + soft
    th = np.arctan2(dy, dx)
    wave = sum(np.sin(m * th + p) / m for m, p in zip((3, 4, 5), phase)) / 1.2
    r = u * reach * (1.0 + BLOB * (1.0 - u) * wave)
    out = np.zeros_like(region)
    out[y0:y1, x0:x1] = region[y0:y1, x0:x1] * np.clip((r - dist) / soft + 0.5, 0.0, 1.0)
    return out


def fill_cover(img, seed, tolerance, u, px, phase):
    """The coverage (h, w) of a fill of the colour region of `img` round pixel `seed`, spread `u` of the way out from it
    (flood, spread: worked out at FLOOD_SIDE, its edges soft as they are grown back to the picture's size; FRONT px at
    1080 the front, `px` a pixel at 1080); None when the seed is off the picture."""
    got = flood(img, seed, tolerance)
    if got is None:
        return None
    region, at, f = got
    region = spread(region, at, u, max(FRONT * px * f, 1.0), phase)
    h, w = img.shape[:2]
    if region.shape != (h, w):
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(region, 0.0, 1.0)


# ================================================================================================= over a cut
class PaintAppFx:
    """The paint app's part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, entry_at,
    _t, _mix, _reached, _seen, _point_of, _hits and _sprite_of, which `_mandala_init` sets). `_paintapp_init` is called
    once its effects are known: their hits are picked, their history worked out and checked flash-safe, their sprites
    read."""

    def _paintapp_init(self):
        self._pa = {}
        self._pa_tiles = OrderedDict()           # sprite tiles kept from frame to frame
        self._pa_held = {}                       # (entry, eraser) -> the picture it holds, its paint on it
        self._pa_pieces = {}                     # (entry, eraser) -> its confetti's tiles
        for fx in self.fxs:
            if fx["kind"] != "paintapp":
                continue
            times = PA.tool_times(fx, self._hits)
            hist = PA.history(fx, times)
            PA.flash_check(fx, hist)
            clicks, tools, picked = PA.clicks(fx, hist)
            st = {"times": times, "hist": hist, "clicks": clicks, "tools": tools, "picked": picked,
                  "lay": PA.layout(fx, self.size), "inks": None, "chrome": None, "buttons": {}, "icons": {},
                  "paths": {}, "files": [], "eye_hits": ()}
            if fx["stamps"] is not None:
                st["files"] = self._sprite_of(fx["stamps"]["refs"])
                for f in st["files"] if self._preload else ():
                    MDF._sprite(str(f))
                face = fx["stamps"]["face"]
                if face is not None:
                    st["eye_hits"] = EYD.hit_times(face, times["stamps"], self._hits)
            self._pa[fx["index"]] = st

    # ---------------------------------------------------------------- the frame
    def _paintapp(self, fx, frame, img):
        st = self._pa[fx["index"]]
        inks = (fx["border"], fx["panel"], fx["ink"], fx["art"], fx["swatches"])
        if st["inks"] != inks:                   # the chrome drawn again in the colours it has now (a breathing palette)
            st.update(inks=inks, chrome=None, buttons={}, icons={})
        t = self._t(frame)
        for e in fx["eraser"]:                   # mk post keeps the frame an eraser will hold, as it reached the effect
            if frame == e["frame"] - 1:
                self._reached(fx, frame, frame, img)
        out = self._pa_paint(fx, st, frame, t, np.array(img, np.float32, copy=True))
        e = PA.eraser_at(fx, frame)
        for key in [k for k in self._pa_held if k[0] == fx["index"] and (e is None or k[1] != e["at"])]:
            self._pa_held.pop(key)
            self._pa_pieces.pop(key, None)
        if e is not None:
            out = self._pa_erase(fx, st, e, frame, out, self._pa_hold(fx, st, e, frame, img))
        if fx["ui"]:
            out = self._pa_chrome(fx, st, t, out)
        if fx["cursor"] is not None:
            self._pa_cursor(fx, st, frame, t, out)
        return self._mix(fx, frame, img, np.clip(out, 0.0, 1.0).astype(np.float32))

    def _pa_hold(self, fx, st, e, frame, img):
        """The picture eraser `e` holds: the frame before it as it reached the effect, with the paint of that frame (in
        that frame's colours)."""
        key = (fx["index"], e["at"])
        if key not in self._pa_held:
            g = e["frame"] - 1
            pic = self._reached(fx, g, frame, img)
            pic = img if pic is None else pic
            self._seen.get(fx["index"], {}).pop(g, None)
            self._pa_held[key] = self._pa_paint(self.entry_at(fx, g), st, g, (g - self.frame0) / self.fps,
                                                np.array(pic, np.float32, copy=True))
        return self._pa_held[key]

    # ---------------------------------------------------------------- paint
    def _pa_seed(self, fx, j, frame):
        """Fill j's point on `frame`, px: its `at` entry, a scene point projected through the shot in the cut."""
        p = fx["bucket"]["at"][j % len(fx["bucket"]["at"])]
        if p["mode"] == "frame":
            return p["at"][0] * self.size[0] - 0.5, p["at"][1] * self.size[1] - 0.5
        return self._point_of(f"{FX.point_key(fx)}b{j % len(fx['bucket']['at'])}", frame)[0]

    def _pa_paint(self, fx, st, frame, t, canvas):
        """`canvas` (in place, and returned) with the stamps and fills on it at clip second `t` (on `frame`, where a
        fill's scene point is read), laid in the order they landed: a fill floods the canvas as it stands then."""
        for it in st["hist"]["items"]:
            if it["kind"] == "fill":
                s = PA.fill_state(fx, it, t)
                if s is not None and s[0] > 0.0 and s[1] > 0.0:
                    self._pa_fill(fx, it["j"], frame, s, canvas)
            else:
                s = PA.stamp_state(fx, it, t)
                if s is not None and s[0] > 1e-3 and s[2] > 0.0:
                    self._pa_stamp(fx, st, it, t, s, canvas)
        return canvas

    def _pa_fill(self, fx, j, frame, state, canvas):
        rng = np.random.default_rng([fx["seed"], fx["index"], 13, j])
        cov = fill_cover(canvas, self._pa_seed(fx, j, frame), fx["bucket"]["tolerance"], state[0], self.px,
                         rng.uniform(0.0, 2.0 * math.pi, 3))
        if cov is None:
            return
        cov *= np.float32(state[1])
        canvas *= (1.0 - cov)[..., None]
        canvas += np.asarray(PA.fill_colour(fx, j), np.float32) * cov[..., None]

    def _pa_tile(self, fx, st, j):
        """Stamp j's sticker tile: its sprite at the stamp's size, with its edge and face (kept from frame to frame)."""
        sp = fx["stamps"]
        path = str(st["files"][PA.stamp_pick(fx, j, len(st["files"]))])
        size = sp["size"] * min(self.size)
        ew = 0.0 if sp["edge"] is None else sp["edge"]["width"] * self.px
        face = sp["face"]
        key = (path, round(size, 2), round(ew, 2), None if face is None else (face["eyes"], face["mouth"], face["size"]))
        if key in self._pa_tiles:
            self._pa_tiles.move_to_end(key)
        else:
            self._pa_tiles[key] = MDF.sprite_tile(path, size, ew, face)
            while len(self._pa_tiles) > TILES:
                self._pa_tiles.popitem(last=False)
        return self._pa_tiles[key]

    def _pa_at(self, fx, st, it, age):
        """Where stamp `it` is `age` s after it lands, px: its spot, scrolled on since."""
        sp, short = fx["stamps"], min(self.size)
        x, y = PA.stamp_spot(fx, st["lay"], it["j"])
        return x + sp["scroll"][0] * short * age, y + sp["scroll"][1] * short * age

    def _pa_map(self, fx, st, it, tile, t):
        """The affine map that lays stamp `it` at clip second `t` (its pop, its turn, its scroll); at its landing pose,
        all but nothing, before it lands."""
        s = PA.stamp_state(fx, it, max(t, it["t"]))
        scale, turn, age = (1e-3, PA.SETTLE, 0.0) if s is None else (max(s[0], 1e-3), s[1], s[3])
        return MDF.placed(tile, self._pa_at(fx, st, it, age), PA.stamp_angle(fx, it["j"]) + turn, scale)

    def _pa_pose(self, fx, st, it, t, state):
        """(tile, map, pupils, shadow) of stamp `it` at clip second `t` (`state` its stamp_state): its sticker tile, the
        affine map that lays it, its pupils' offsets (None: at rest or no eyes) and its drop shadow (mkmmd.mandala.lay's
        argument: its alpha faded with the stamp; None for none)."""
        sp = fx["stamps"]
        tile = self._pa_tile(fx, st, it["j"])
        M = self._pa_map(fx, st, it, tile, t)
        pupils = None
        face = sp["face"]
        if face is not None and face["eyes"]:
            x0, y0, x1, y1 = st["lay"]["canvas"]
            maps = lambda ts: np.stack([self._pa_map(fx, st, it, tile, s) for s in ts])     # noqa: E731
            pupils = EYD.pupils(face, tile, t, st["eye_hits"], (fx["index"], it["j"]), maps,
                                EYD.look_at(face, self.size, (0.5 * (x0 + x1) - 0.5, 0.5 * (y0 + y1) - 0.5)))
        sh = sp["shadow"]
        shadow = None if sh is None else (sh["color"], (sh["offset"][0] * self.size[1], sh["offset"][1] * self.size[1]),
                                          sh["blur"] * self.px, sh["alpha"] * state[2])
        return tile, M, pupils, shadow

    def _pa_stamp(self, fx, st, it, t, state, canvas):
        sp = fx["stamps"]
        tile, M, pupils, shadow = self._pa_pose(fx, st, it, t, state)
        if state[2] < 1.0:
            k = np.float32(state[2])
            tile = tile._replace(rgb=tile.rgb * k, a=tile.a * k, ring=None if tile.ring is None else tile.ring * k)
        MDF.lay(canvas, tile, M, None if sp["edge"] is None else sp["edge"]["color"], shadow, pupils=pupils)
        self._pa_sparkles(fx, it["j"], state[3], self._pa_at(fx, st, it, state[3]), canvas)

    def _pa_sparks(self, fx, j, age, centre):
        """The sparkles round stamp j (its middle at `centre` px) `age` s after it lands: [(points px, accent colour,
        alpha)], those big enough to show."""
        size = fx["stamps"]["size"] * min(self.size)
        out = []
        for dx, dy, s, ang, k, alpha in PA.sparkles(fx, j, age):
            r = s * size
            if r < 0.5 or alpha <= 0.0:
                continue
            pts = MD._turned(_star(r), ang) + (centre[0] + dx * size, centre[1] + dy * size)
            out.append((pts, fx["art"][PA.ACCENTS[k]], alpha))
        return out

    def _pa_sparkles(self, fx, j, age, centre, canvas):
        """The sparkles round stamp j (its middle at `centre` px) `age` s after it lands."""
        for pts, colour, alpha in self._pa_sparks(fx, j, age, centre):
            self._pa_poly(canvas, pts, colour, alpha, fx["ink"], max(1.5 * self.px, 0.8))

    @staticmethod
    def _pa_poly(canvas, pts, colour, alpha, ink=None, line=0.0):
        """`canvas` (in place) with the polygon `pts` (px) filled in `colour` at `alpha`, with an `ink` line round it."""
        got = poly_cover(canvas.shape[:2], pts, line)
        if got is None:
            return
        x0, y0, outline, fill = got
        region = canvas[y0:y0 + fill.shape[0], x0:x0 + fill.shape[1]]
        if ink is not None and outline is not None:
            c = (outline.astype(np.float32) / np.float32(255.0) * np.float32(alpha))[..., None]
            region *= 1.0 - c
            region += np.asarray(ink, np.float32) * c
        c = (fill.astype(np.float32) / np.float32(255.0) * np.float32(alpha))[..., None]
        region *= 1.0 - c
        region += np.asarray(colour, np.float32) * c

    # ---------------------------------------------------------------- the eraser
    def _pa_path(self, st, e):
        width = e["width"] * min(self.size)
        if width not in st["paths"]:
            st["paths"][width] = PA.wipe_path(self.size, width)
        return st["paths"][width], width

    def _pa_erase(self, fx, st, e, frame, live, held):
        """The incoming frame `live` (its paint on it) with what is left of the `held` picture over it, `u` of the way
        through eraser `e`."""
        u = (frame - e["frame"]) / float(e["frames"])
        if e["mode"] == "wipe":
            path, width = self._pa_path(st, e)
            w, h = self.size
            mask = np.zeros((h, w), np.uint8)
            q = np.rint(PA.upto(path, u) * (1 << SHIFT)).astype(np.int32)
            cv2.polylines(mask, [q], False, 255, max(int(round(width)), 1), cv2.LINE_AA, SHIFT)
            return _between(held, live, mask)
        return self._pa_confetti(fx, e, u, live, held)

    def _pa_confetti(self, fx, e, u, live, held):
        key = (fx["index"], e["at"])
        w, h = self.size
        short = float(min(w, h))
        if key not in self._pa_pieces:
            polys = PA.confetti(self.size, e["pieces"], fx["seed"], (fx["index"], int(round(e["at"] * 1000))))
            centre = (e["center"][0] * w, e["center"][1] * h)
            fls = PA.flights(polys, centre, self.size, e["dur"], fx["seed"], (fx["index"], int(round(e["at"] * 1000))))
            tiles = []
            for p, fl in zip(polys, fls):
                x0, y0 = int(math.floor(p[:, 0].min())) - 1, int(math.floor(p[:, 1].min())) - 1
                x1, y1 = int(math.ceil(p[:, 0].max())) + 1, int(math.ceil(p[:, 1].max())) + 1
                cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(cov, [np.rint((p - (x0 + 0.5, y0 + 0.5)) * (1 << SHIFT)).astype(np.int32)], 255,
                             cv2.LINE_AA, SHIFT)
                a = cov.astype(np.float32) / np.float32(255.0)
                pic = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
                X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
                pic[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = held[Y0:Y1, X0:X1]
                tiles.append({"rgba": np.dstack([pic * a[..., None], a]), "c": (fl["c"][0] - x0 - 0.5,
                                                                                 fl["c"][1] - y0 - 0.5)})
            self._pa_pieces[key] = (polys, fls, tiles)
        polys, fls, tiles = self._pa_pieces[key]
        s_now = u * e["dur"]
        gone = np.zeros((h, w), np.uint8)
        left = [i for i, fl in enumerate(fls) if fl["t0"] <= s_now]
        if left:
            cv2.fillPoly(gone, [np.rint((polys[i] - 0.5) * (1 << SHIFT)).astype(np.int32) for i in left], 255,
                         cv2.LINE_AA, SHIFT)
        out = _between(held, live, gone)
        laid, reach = [], None
        shade = np.zeros((h, w), np.float32)
        for i in sorted(left, key=lambda i: fls[i]["t0"]):
            pose = PA.flight_pose(fls[i], s_now - fls[i]["t0"], u, short, e["dur"])
            if pose is None:
                continue
            (x, y), ang, sx, sy, back = pose
            tl = tiles[i]
            r = math.radians(ang)
            c, s = math.cos(r), math.sin(r)
            A = np.array([[c * sx, s * sy], [-s * sx, c * sy]])
            M = np.hstack([A, (np.array([x - 0.5, y - 0.5]) - A @ np.asarray(tl["c"]))[:, None]])
            hh, ww = tl["rgba"].shape[:2]
            corners = np.array([[-0.5, -0.5], [ww - 0.5, -0.5], [-0.5, hh - 0.5], [ww - 0.5, hh - 0.5]])
            pts = corners @ M[:, :2].T + M[:, 2]
            bx0, by0 = max(int(math.floor(pts[:, 0].min())), 0), max(int(math.floor(pts[:, 1].min())), 0)
            bx1, by1 = min(int(math.ceil(pts[:, 0].max())) + 1, w), min(int(math.ceil(pts[:, 1].max())) + 1, h)
            if bx1 <= bx0 or by1 <= by0:
                continue
            Mt = M.copy()
            Mt[:, 2] -= (bx0, by0)
            got = cv2.warpAffine(tl["rgba"], Mt.astype(np.float32), (bx1 - bx0, by1 - by0),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = np.clip(got[..., 3], 0.0, 1.0)
            if back:
                col = np.asarray(fx["art"][PA.ACCENTS[fls[i]["back"]]], np.float32) * np.float32(0.8 + 0.2 * abs(sx))
                rgb = col * a[..., None]
            else:
                rgb = got[..., :3]
            laid.append(((bx0, by0, bx1, by1), rgb, a))
            shade[by0:by1, bx0:bx1] = np.maximum(shade[by0:by1, bx0:bx1], a)
            reach = (bx0, by0, bx1, by1) if reach is None else (min(reach[0], bx0), min(reach[1], by0),
                                                                 max(reach[2], bx1), max(reach[3], by1))
        if laid:                                 # their shadow, worked out where they are, then the pieces
            k = self.px
            pad = int(math.ceil(15.0 * k + 3.0 * 5.0 * k)) + 2
            X0, Y0 = max(reach[0] - pad, 0), max(reach[1] - pad, 0)
            X1, Y1 = min(reach[2] + pad, w), min(reach[3] + pad, h)
            sh = MDF.soft(_ff().plate_shift(shade[Y0:Y1, X0:X1], 6.0 * k, 10.0 * k), max(5.0 * k, 0.5))
            sh = (sh * np.float32(0.3))[..., None]
            region = out[Y0:Y1, X0:X1]
            region -= (region - np.asarray(fx["ink"], np.float32)) * sh
            for (bx0, by0, bx1, by1), rgb, a in laid:
                region = out[by0:by1, bx0:bx1]
                region *= (1.0 - a)[..., None]
                region += rgb
        return out

    # ---------------------------------------------------------------- the chrome
    def _pa_chrome(self, fx, st, t, out):
        """`out` with the chrome over it: the still part, then each button and swatch as it stands at `t` (the tool in
        hand picked, a button pressed by the cursor's click, the colour picked)."""
        lay = st["lay"]
        if st["chrome"] is None:
            st["chrome"] = chrome(fx, lay)
        rgb, a, strips = st["chrome"]
        for y0, y1, x0, x1 in strips:                                   # the canvas inside them is left as it is
            region = out[y0:y1, x0:x1]
            region *= (1.0 - a[y0:y1, x0:x1])[..., None]
            region += rgb[y0:y1, x0:x1]
        tool = PA.in_force(st["tools"], t)
        for name, box in lay["buttons"]:
            hits = [c["t"] for c in st["clicks"]
                    if c["on"] == ("tool", name) or (name == "undo" and c["on"][0] == "undo")]
            active = name == tool
            down = round(10.0 * (PA.press(hits, t) + (HELD if active else 0.0))) / 10.0
            key = (name, active, down)
            if key not in st["buttons"]:
                st["buttons"][key] = button_tile(fx, lay, name, box, active, down, st["icons"])
            self._pa_put(out, *st["buttons"][key])
        picked = PA.in_force(st["picked"], t)
        for i in sorted(range(len(lay["swatches"])), key=lambda i: i == picked):
            key = ("swatch", i, i == picked)
            if key not in st["buttons"]:
                st["buttons"][key] = swatch_tile(fx, lay, i, i == picked)
            self._pa_put(out, *st["buttons"][key])
        return out

    @staticmethod
    def _pa_put(out, cell, rgb, a=None):
        """`out` (in place) with a tile put at `cell`: opaque, or laid over it by its coverage `a` (premultiplied)."""
        h, w = out.shape[:2]
        x0, y0, x1, y1 = cell
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if X1 <= X0 or Y1 <= Y0:
            return
        src = rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        if a is None:
            out[Y0:Y1, X0:X1] = src
        else:
            region = out[Y0:Y1, X0:X1]
            region *= (1.0 - a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0])[..., None]
            region += src

    # ---------------------------------------------------------------- the cursor
    def _pa_where(self, fx, st, on, frame):
        """Where a click `on` is, px, on `frame`; None where the cursor stays put (an undo without the chrome)."""
        lay, what = st["lay"], on[0]
        if what == "stamp":
            return PA.stamp_spot(fx, lay, on[1])
        if what == "fill":
            return self._pa_seed(fx, on[1], frame)
        if what == "erase":
            e = fx["eraser"][on[1]]
            if e["mode"] == "wipe":
                return tuple(self._pa_path(st, e)[0][0])
            return e["center"][0] * self.size[0] - 0.5, e["center"][1] * self.size[1] - 0.5
        if what == "undo":
            return PA.button_of(lay, "undo") if fx["ui"] else None
        if what == "tool":
            return PA.button_of(lay, on[1])
        if what == "swatch":
            return lay["swatches"][on[1]][:2]
        k = fx["cursor"]["keys"][on[1]]
        return k["at"][0] * self.size[0] - 0.5, k["at"][1] * self.size[1] - 0.5

    def _pa_cursor(self, fx, st, frame, t, out):
        """The cursor (in place) on `out`: where it glides between its stops (along an eraser's strokes while it rubs),
        squashed as it clicks, its clicks' rings spreading, an eraser block under it while it rubs."""
        cur, lay = fx["cursor"], st["lay"]
        stops, taps, wipe = [(fx["from"], lay["home"])], [], None
        for c in st["clicks"]:
            p = self._pa_where(fx, st, c["on"], frame)
            if p is None:
                continue
            stops.append((c["t"], p))
            if c["on"][0] != "key" or cur["keys"][c["on"][1]]["click"]:
                taps.append((c["t"], p))
            if c["on"][0] == "erase" and fx["eraser"][c["on"][1]]["mode"] == "wipe":
                e = fx["eraser"][c["on"][1]]
                path, width = self._pa_path(st, e)
                stops.append((c["t"] + e["dur"], tuple(path[-1])))
                if e["at"] <= t < e["at"] + e["dur"]:
                    wipe = (e, path, width)
        stops.sort(key=lambda s: s[0])
        if wipe is not None:
            e, path, width = wipe
            at = PA.along(path, (t - e["at"]) / e["dur"])
        else:
            at = PA.glide(stops, t, cur["travel"])
        k, short = self.px, float(min(self.size))
        for tc, p in taps:                                              # each click's ring, where it clicked
            for u in PA.ripples([tc], t):
                r = (RING[0] + (RING[1] - RING[0]) * TW.ease(u, "out")) * short
                self._pa_ring(out, p, r, RING[2] * k, fx["ink"], (1.0 - u) ** 2)
        shadow = (fx["ink"], (SHADOW[0] * k, SHADOW[1] * k), SHADOW[2] * k, SHADOW[3])
        if wipe is not None:                                            # the block it rubs with, under the hand
            key = ("eraser", round(0.8 * wipe[2], 2))
            if key not in st["icons"]:
                rgb, a, origin = draw(_eraser(fx["art"], fx["ink"]), 0.8 * wipe[2], OUTLINE * k, fx["ink"])
                st["icons"][key] = MDF.tile(rgb, a, origin, 0.0)
            tile = st["icons"][key]
            MDF.lay(out, tile, MDF.placed(tile, at, 0.0, 1.0), shadow=shadow)
        key = ("cursor", cur["kind"], round(cur["size"] * short, 2))
        if key not in st["icons"]:
            parts = _hand(fx["art"], fx["ink"]) if cur["kind"] == "hand" else _pencil_cursor(fx["art"], fx["ink"])
            rgb, a, origin = draw(parts, cur["size"] * short, CURSOR_OUTLINE * k, fx["ink"])
            st["icons"][key] = MDF.tile(rgb, a, origin, 0.0)
        tile = st["icons"][key]
        p = PA.press([tc for tc, _ in taps], t) if wipe is None else RUB
        sx, sy = 1.0 + SQUASH[0] * p, 1.0 - SQUASH[1] * p
        A = np.array([[sx, 0.0], [0.0, sy]])
        M = np.hstack([A, (np.asarray(at, float) - A @ np.asarray(tile.anchor, float))[:, None]])
        lift = 1.0 - 0.6 * max(p, 0.0)                                 # pressed down, its shadow comes in under it
        MDF.lay(out, tile, M, shadow=(shadow[0], (shadow[1][0] * lift, shadow[1][1] * lift), shadow[2], shadow[3]))

    @staticmethod
    def _pa_ring(out, c, r, width, colour, alpha):
        h, w = out.shape[:2]
        reach = r + width + 2.0
        x0, y0 = max(int(math.floor(c[0] - reach)), 0), max(int(math.floor(c[1] - reach)), 0)
        x1, y1 = min(int(math.ceil(c[0] + reach)) + 1, w), min(int(math.ceil(c[1] + reach)) + 1, h)
        if x1 <= x0 or y1 <= y0:
            return
        xs, ys = _grid((x0, y0, x1, y1))
        d = np.abs(np.hypot(xs - c[0], ys - c[1]) - r) - 0.5 * width
        cov = (_cov(d) * np.float32(alpha))[..., None]
        region = out[y0:y1, x0:x1]
        region *= 1.0 - cov
        region += np.asarray(colour, np.float32) * cov
