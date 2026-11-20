"""Sticker sheets on the finished pictures (docs/design.md: Frame effects, Sticker sheets): the picture work of the `[[fx]]`
kinds `sheet` and `fling` that mkmmd.core.sheet checks, times and lays out. SheetFx is the part of mkmmd.framefx.FrameFx
that runs them.

A sheet is drawn back to front: the paper (made once per output, a little larger than the frame, and carried by the
desk's drift on every frame), the hollow outlines (each made once into a tile), the doodles (the pen kits' tiles,
mkmmd.mandala.shape_tile), the dust of a slap, then the live figure sticker over its rim and shadow, or, while it is
peeled, its sheet rolled back by mkmmd.peel.peel_sheet (the peel transition's roll) over the outline it leaves. Every
part of it lies on the desk: the drift moves them all as one, so the camera seems to hang over it. Images are float RGB
in display space (0..1), (h, w, 3); a coverage is (h, w) in 0..1."""
import math

import cv2
import numpy as np

from . import mandala as MDF
from . import peel as PL
from .core import framefx as FX
from .core import mandala as MD
from .core import peel as PE
from .core import sheet as ST

WHITE = (1.0, 1.0, 1.0)
LIFT_OFFSET = 3.0        # a lifted sticker's shadow falls this many times farther at lift 1 ...
LIFT_BLUR = 1.5          # ... this many times softer ...
LIFT_FADE = 0.35         # ... and this much fainter
DASH = (9.0, 6.0)        # a hollow's cut line: dash and gap, px at 1080
DASH_WIDTH = 1.7         # its width, px at 1080
DASH_INSET = 3.0         # how far inside the outline it runs, px at 1080
DASH_ALPHA = 0.42
DUST_LIFE = 0.45         # seconds a slap's dust flies
DUST_TAU = 0.11          # its slowing down: it travels speed x tau x (1 - e^(-age / tau))
FLING_SAMPLES = 16       # at most this many samples over a frame's shutter for a sticker moving fast
SIZE_STEP = 1.25         # a thrown sticker's tile is cut at sizes this far apart and shrunk to the size it shows at
FLING_SHUTTER = 0.5      # a thrown sticker's streak: the share of a frame's time it is smeared over
FLING_RIM_AT = 0.1       # a thrown sticker's rim is its `edge` width where it is this many frame heights long


def _ff():
    from . import framefx as FF          # framefx imports this module for SheetFx: its helpers are there by now
    return FF


# ================================================================================================= paper
def _field(rng, shape, sigma):
    """Smooth noise of unit spread over `shape` (h, w), features about `sigma` px: drawn at a coarser grid and blurred."""
    h, w = shape
    f = max(1, int(sigma / 3.0))
    n = rng.standard_normal((h // f + 3, w // f + 3)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(sigma / f, 0.5))
    if f > 1:
        n = cv2.resize(n, (n.shape[1] * f, n.shape[0] * f), interpolation=cv2.INTER_CUBIC)
    n = n[:h, :w]
    return (n - n.mean()) / (n.std() + 1e-6)


def paper_texture(size, paper, seed, pad):
    """The sheet's paper for a frame `size` (w, h) with `pad` px more on every side: `color` mottled toward `light` and
    `shade` (clouds of a few sizes), its fibres (short curling strands, darker and lighter), its tooth (a fine relief lit
    from the upper left and a grain finer still: a dither, so the paper never bands) and a grid of `ink` dots or rules
    `spacing` frame heights apart, all `scale` times their size. Float RGB (h + 2 pad, w + 2 pad, 3)."""
    W, H = int(size[0]) + 2 * pad, int(size[1]) + 2 * pad
    px = min(size) / 1080.0
    k = paper["scale"]
    rng = np.random.default_rng([int(seed), 17])
    base, light, shade, ink = (np.asarray(paper[c], np.float32) for c in ("color", "light", "shade", "ink"))
    m = 0.62 * _field(rng, (H, W), 0.16 * size[1] * k) + 0.45 * _field(rng, (H, W), 0.045 * size[1] * k) + \
        0.22 * _field(rng, (H, W), 0.012 * size[1] * k)
    m = np.clip(m * np.float32(0.42 * paper["mottle"]), -1.0, 1.0)[..., None]
    out = base + (light - base) * np.maximum(m, 0.0) + (shade - base) * np.maximum(-m, 0.0)
    if paper["fibres"] > 0:
        dark, lite = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
        n = int(paper["fibres"] * 1100 * W * H / (1080.0 * 1920.0) / (k * k))
        for i in range(n):
            canvas = dark if i % 3 else lite
            p = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            a, step = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(2.5, 7.0) * px * k
            pts = [p.copy()]
            for _ in range(6):
                a += rng.normal(0.0, 0.35)
                p = p + step * np.array([math.cos(a), math.sin(a)])
                pts.append(p.copy())
            q = np.rint(np.array(pts) * 16.0).astype(np.int32)
            cv2.polylines(canvas, [q], False, int(rng.uniform(90, 255)), 1, cv2.LINE_AA, 4)
        soft = max(0.45 * px * k, 0.35)
        fd = cv2.GaussianBlur(dark.astype(np.float32) / 255.0, (0, 0), soft)[..., None] * np.float32(paper["fibres"])
        fl = cv2.GaussianBlur(lite.astype(np.float32) / 255.0, (0, 0), soft)[..., None] * np.float32(paper["fibres"])
        out = out + (shade - out) * (0.5 * fd) + (np.asarray(WHITE, np.float32) - out) * (0.45 * fl)
    if paper["tooth"] > 0:
        relief = _field(rng, (H, W), 1.8 * px * k)
        lit = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3) * -0.6 + cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3) * -0.8
        lit /= lit.std() + 1e-6
        grain = _field(rng, (H, W), 0.8 * px)
        out = out * (1.0 + np.float32(paper["tooth"]) * (0.0045 * lit + 0.0035 * grain))[..., None]
    if paper["grid"] is not None:
        g = np.zeros((H, W), np.uint8)
        s = paper["spacing"] * size[1] * k
        if paper["grid"] == "dots":
            r = max(1.6 * px * k, 0.8)
            for y in np.arange(s / 2.0, H, s):
                for x in np.arange(s / 2.0, W, s):
                    cv2.circle(g, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)), 255, -1, cv2.LINE_AA, 4)
            alpha = 0.16
        else:
            th = max(int(round(1.2 * px)), 1)
            for y in np.arange(s / 2.0, H, s):
                cv2.line(g, (0, int(round(y * 16))), ((W - 1) * 16, int(round(y * 16))), 255, th, cv2.LINE_AA, 4)
            if paper["grid"] == "squares":
                for x in np.arange(s / 2.0, W, s):
                    cv2.line(g, (int(round(x * 16)), 0), (int(round(x * 16)), (H - 1) * 16), 255, th, cv2.LINE_AA, 4)
            alpha = 0.11
        a = (g.astype(np.float32) / 255.0 * np.float32(alpha))[..., None]
        out = out + (ink - out) * a
    return np.ascontiguousarray(out, np.float32)


def lamp(size, amount):
    """The desk lamp's falloff over a frame `size`, a multiplier (h, w, 1): 1 near the upper left, `amount` darker in the
    farthest corner (fixed to the frame: the lamp does not drift with the camera)."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xs / max(w, h) - 0.22 * w / max(w, h), ys / max(w, h) - 0.12 * h / max(w, h))
    r /= r.max()
    return (1.0 - np.float32(amount) * r ** 1.6)[..., None]


# ================================================================================================= helpers
def _affine(M):
    return np.asarray(M, np.float64).reshape(2, 3)


def _apply(M, p):
    M = _affine(M)
    return (float(M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]), float(M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]))


def _scaled(c, sx, sy):
    """The map that scales by (sx, sy) about pixel `c`."""
    return [[sx, 0.0, c[0] - sx * c[0]], [0.0, sy, c[1] - sy * c[1]]]


def _shift(x, y):
    return [[1.0, 0.0, x], [0.0, 1.0, y]]


def lift_shadow(shadow, lift):
    """A sticker's shadow (colour, (dx, dy) px, blur px, alpha) as it is `lift` off the sheet (0 lying on it, 1 a
    sticker's height in a hand): farther, softer, fainter."""
    if shadow is None:
        return None
    c, (dx, dy), blur, a = shadow
    k = 1.0 + LIFT_OFFSET * lift
    return c, (dx * k, dy * k), blur * (1.0 + LIFT_BLUR * lift), a * (1.0 - LIFT_FADE * min(lift, 1.0))


def _box(a, pad=0):
    x, y, w, h = cv2.boundingRect((np.asarray(a) > 1.0 / 512.0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    H, W = a.shape
    return max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)


def stamp(t, M, size, edge, shadow, back=None):
    """What a sticker tile `t` laid by `M` puts down in a picture of `size`: (box (x0, y0, x1, y1), premultiplied rgb,
    coverage) of it, its edge and its shadow, as mkmmd.mandala.lay draws them (None when it falls outside); `back`, a
    colour: its back shows instead of its face, the paper of the sticker shaded by `back`."""
    FF = _ff()
    got = MDF.lay_box(t.a.shape, M, shadow, size)
    if got is None:
        return None
    (x0, y0, x1, y1), Mt = got
    sz = (x1 - x0, y1 - y0)
    if back is not None:
        cov = t.a if t.ring is None else np.maximum(t.ring, t.a)
        w = FF._warp(cov, Mt, sz)
        a = np.clip(w, 0.0, 1.0)
        rgb = np.asarray(back, np.float32) * a[..., None]
    else:
        got = FF._warp(np.dstack([t.rgb, t.a]), Mt, sz)
        a, rgb = np.clip(got[..., 3], 0.0, 1.0), got[..., :3]
        if t.ring is not None and edge is not None:
            ring = np.clip(FF._warp(t.ring, Mt, sz), 0.0, 1.0)
            rgb = rgb + np.asarray(edge, np.float32) * (ring * (1.0 - a))[..., None]
            a = a + ring - a * ring
    if shadow is not None and shadow[3] > 0:
        k = np.float32(shadow[3]) * MDF.soft(FF.plate_shift(a, shadow[1][0], shadow[1][1]), shadow[2])
        rgb = rgb + np.asarray(shadow[0], np.float32) * (k * (1.0 - a))[..., None]
        a = a + k - a * k
    return (x0, y0, x1, y1), rgb, a


def put(out, box, rgb, a):
    """`out` (in place) with premultiplied `rgb` and coverage `a` laid over it in `box`."""
    x0, y0, x1, y1 = box
    region = out[y0:y1, x0:x1]
    region *= (1.0 - a)[..., None]
    region += rgb
    return out


def _sticker_map(at, angle, sx, sy, anchor):
    """A tile's map: its `anchor` to pixel `at`, turned `angle` degrees counter-clockwise, then squashed (sx, sy) on
    screen about that point (a landing's squash stays upright on the screen)."""
    r = math.radians(angle)
    c, s = math.cos(r), math.sin(r)
    A = np.array([[sx * c, sx * s], [-sy * s, sy * c]])
    return np.hstack([A, (np.asarray(at, float) - A @ np.asarray(anchor, float))[:, None]])


def dashes(cov, inset, dash, gap, width):
    """A dashed line `inset` px inside the outline of coverage `cov` (its outer outline, where it is over half), dashes
    `dash` px long `gap` apart, `width` px wide, antialiased: a coverage of `cov`'s shape."""
    inside = (cov >= 0.5).astype(np.uint8)
    if inset > 0:
        d = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        inside = (d > inset).astype(np.uint8)
    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(cov.shape, np.uint8)
    th = max(int(round(width)), 1)
    for c in contours:
        p = c[:, 0, :].astype(np.float64)
        if len(p) < 12:
            continue
        k = 5                                    # a running mean round the loop: the pixel stairs smoothed away
        p = sum(np.roll(p, s, 0) for s in range(-k, k + 1)) / (2 * k + 1)
        p = np.vstack([p, p[:1]])
        seg = np.hypot(*np.diff(p, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        period = dash + gap
        on = (s % period) < dash
        run = []
        for i in range(len(p)):
            if on[i]:
                run.append(p[i])
            elif run:
                if len(run) > 1:
                    cv2.polylines(out, [np.rint(np.array(run) * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, 4)
                run = []
        if len(run) > 1:
            cv2.polylines(out, [np.rint(np.array(run) * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, 4)
    return out.astype(np.float32) / 255.0


def hollow_layer(cov, edge, shadow, sheen, dash, px, ink, light):
    """The outline a sticker of coverage `cov` (frame px, cropped) leaves when it is peeled off: its die-cut rim (`edge`
    (colour, width px)) with nothing inside but the paper, the rim's soft shadow on the paper round it and inside it, a
    faint gloss where the glue was (`sheen` 0..1, in `light`) and, with `dash`, a dashed cut line in `ink` just inside
    the outline. (premultiplied rgb, coverage), the crop's size."""
    FF = _ff()
    width = 0.0 if edge is None else edge[1]
    ring = FF._grow(cov, width) if width > 0 else cov
    rim = np.clip(ring - cov, 0.0, 1.0)
    h, w = cov.shape
    rgb, a = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)

    def over(colour, k):
        nonlocal rgb, a
        rgb = rgb * (1.0 - k)[..., None] + np.asarray(colour, np.float32) * k[..., None]
        a = a + k - a * k
    if shadow is not None and shadow[3] > 0:
        k = np.float32(shadow[3]) * MDF.soft(FF.plate_shift(rim, shadow[1][0], shadow[1][1]), shadow[2])
        over(shadow[0], k * (1.0 - rim))
    if sheen > 0:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        period = max(0.9 * max(h, w), 8.0)
        band = 0.5 + 0.5 * np.cos(2.0 * math.pi * (xs * 0.8 + ys * 0.6) / period)
        gloss = band ** 3
        over(light, np.float32(sheen) * cov * (0.18 + 0.5 * gloss))
    if dash:
        d = dashes(cov, DASH_INSET * px, DASH[0] * px, DASH[1] * px, DASH_WIDTH * px)
        over(ink, np.float32(DASH_ALPHA) * d * cov)
    if edge is not None:
        over(edge[0], rim)
    return np.ascontiguousarray(rgb, np.float32), a


def tape_layer(size, places, tapes, seed, shadow):
    """The washi tapes of a sheet as one layer the frame's size: (premultiplied rgb, coverage, (0, 0)). Each strip
    (mkmmd.core.sheet.tape_places) is its colour lightened, patterned in the colour itself (`stripes` across it on the
    bias, `dots`, `gingham`), its ends torn, `alpha` translucent (the paper shows through), a soft shadow under it."""
    FF = _ff()
    W, H = int(size[0]), int(size[1])
    rgb, a = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
    px = min(size) / 1080.0
    for k, tp in enumerate(places):
        rng = np.random.default_rng([int(seed), int(tapes["seed"]), k, 31])
        L, Wd = tp["length"] * H, tp["width"] * H
        r = math.radians(tp["angle"])
        u, v = np.array([math.cos(r), -math.sin(r)]), np.array([math.sin(r), math.cos(r)])   # along, across (y down)
        c = np.array([tp["at"][0] * W - 0.5, tp["at"][1] * H - 0.5])
        n = 7
        end = [(-L / 2 + rng.uniform(-0.05, 0.05) * Wd, -Wd / 2 + Wd * i / (n - 1)) for i in range(n)]
        far = [(L / 2 + rng.uniform(-0.05, 0.05) * Wd, Wd / 2 - Wd * i / (n - 1)) for i in range(n)]
        torn = [(s + (0.06 * Wd if i % 2 else -0.06 * Wd), q) for i, (s, q) in enumerate(end)] + \
            [(s + (0.06 * Wd if i % 2 else -0.06 * Wd), q) for i, (s, q) in enumerate(far)]
        poly = np.array([c + s * u + q * v for s, q in torn])
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.rint(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        m = mask.astype(np.float32) / 255.0
        b = _box(m, 2)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        su = (xs - c[0]) * u[0] + (ys - c[1]) * u[1]
        sv = (xs - c[0]) * v[0] + (ys - c[1]) * v[1]
        p = tp["pattern"]
        step = Wd / 3.0
        if p == "stripes":
            ink = (((su + sv) / step) % 1.0 < 0.45).astype(np.float32)
        elif p == "dots":
            gu, gv = (su / step + 0.5) % 1.0 - 0.5, (sv / step + 0.5 * np.floor(su / step + 0.5)) % 1.0 - 0.5
            ink = np.clip((0.22 - np.hypot(gu, gv)) * step + 0.5, 0.0, 1.0)
        else:
            ink = 0.5 * ((su / step) % 1.0 < 0.5) + 0.5 * ((sv / step) % 1.0 < 0.5)
        col = np.asarray(tapes["colors"][tp["color"]], np.float32)
        light = col + (1.0 - col) * 0.55
        face = light + (col - light) * ink[..., None]
        mm = m[y0:y1, x0:x1] * np.float32(tapes["alpha"])
        if shadow is not None and shadow[3] > 0:
            sh = 0.6 * shadow[3] * MDF.soft(FF.plate_shift(m[y0:y1, x0:x1], 0.5 * shadow[1][0], 0.5 * shadow[1][1]),
                                            max(shadow[2] * 0.7, 1.0 * px))
            k = (sh * (1.0 - mm)).astype(np.float32)
            rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1.0 - k)[..., None] + np.asarray(shadow[0], np.float32) * k[..., None]
            a[y0:y1, x0:x1] = a[y0:y1, x0:x1] + k - a[y0:y1, x0:x1] * k
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1.0 - mm)[..., None] + face * mm[..., None]
        a[y0:y1, x0:x1] = a[y0:y1, x0:x1] + mm - a[y0:y1, x0:x1] * mm
    if not a.any():
        return None
    return rgb, a, (0, 0)


def scribble(out, lines, at, size, angle, ink, width, jitter):
    """Pen `lines` (mkmmd.core.sheet.scribble_lines) drawn on `out` in place: their box `size` px across about pixel
    `at`, turned `angle` degrees, each point moved by `jitter` ((n, 2) px for all the points in order: the hand's
    wobble), in `ink`, `width` px, antialiased."""
    r = math.radians(angle)
    R = np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])
    pts, i = [], 0
    for ln in lines:
        q = (np.asarray(ln) @ R.T) * size + at + jitter[i:i + len(ln)]
        i += len(ln)
        pts.append(q)
    allp = np.concatenate(pts)
    pad = int(width) + 3
    x0, y0 = int(max(allp[:, 0].min() - pad, 0)), int(max(allp[:, 1].min() - pad, 0))
    x1, y1 = int(min(allp[:, 0].max() + pad, out.shape[1])), int(min(allp[:, 1].max() + pad, out.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return out
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    th = max(int(round(width)), 1)
    for q in pts:
        cv2.polylines(m, [np.rint((q - (x0, y0)) * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, 4)
    k = (m.astype(np.float32) / 255.0 * np.float32(0.9))[..., None]
    region = out[y0:y1, x0:x1]
    region += (np.asarray(ink, np.float32) - region) * k
    return out


# ================================================================================================= over a cut
class SheetFx:
    """The sheets' and flings' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, _t, _at,
    _figure, _shot_frame, _pose_map, _placing, _pivot, _hit_times and _hits). `_sheet_init` is called once."""

    def _sheet_init(self):
        self._sheets = {}            # a sheet's index -> what it keeps from frame to frame on this output
        self._sheet_tiles = {}       # (shape, colours, size px, edge px) -> a doodle's tile
        self._flings = {}            # a fling's index -> its stickers

    # ---------------------------------------------------------------- what a sheet keeps
    def _sheet_state(self, fx):
        st = self._sheets.get(fx["index"])
        if st is not None:
            return st
        W, H = self.size
        st = {"paper": None, "lamp": None, "pad": 0, "own": None, "box": None, "centre": None, "hollows": [],
              "doodles": None}
        if fx["paper"] is not None:
            d = fx["drift"] or {"amount": 0.0, "turn": 0.0}
            st["pad"] = int(math.ceil(d["amount"] * H * 1.3 + math.radians(d["turn"]) * math.hypot(W, H) / 2.0)) + 4
            st["paper"] = paper_texture(self.size, fx["paper"], fx["seed"], st["pad"])
            st["lamp"] = lamp(self.size, fx["paper"]["lamp"]) if fx["paper"]["lamp"] > 0 else None
        if fx["objects"] is not None:
            m0 = self._figure(fx, fx["f0"])
            box = _box(m0) if m0 is not None else None
            if box is None:
                raise FX.FxError(f"fx {fx['index']} (sheet): the figure is not in the frame on the span's first frame "
                                 f"({fx['f0']}), which places it")
            st["own"] = ((box[0] + box[2]) / 2.0 - 0.5, (box[1] + box[3]) / 2.0 - 0.5)   # its middle, its own frame
            M0 = self._pose_map(fx, fx["f0"])
            cov0 = m0 if M0 is None else _ff()._warp(m0, M0, self.size)
            e = self._sheet_edge(fx)
            b = _box(cov0, int(math.ceil(e[1])) if e else 0)
            st["box"] = None if b is None else (b[0] / W, b[1] / H, b[2] / W, b[3] / H)
            st["taken"] = _ff()._grow(cov0, e[1]) if e is not None and e[1] > 0 else cov0
            st["centre"] = _apply(M0, st["own"]) if M0 is not None else st["own"]
            st["hollows"] = [self._hollow_tile(fx, h) for h in fx["hollows"]]
            st["blanks"] = [self._blank_of(fx, h) for h in fx["hollows"] if h["peel"] is not None]
        self._sheets[fx["index"]] = st
        st["doodles"] = self._doodle_plan(fx, st)
        return st

    def _sheet_edge(self, fx):
        e = fx["edge"]
        return None if e is None else (e["color"], e["width"] * self.px)

    def _sheet_shadow(self, fx):
        s, h = fx["shadow"], self.size[1]
        return None if s is None else (s["color"], (s["offset"][0] * h, s["offset"][1] * h), s["blur"] * self.px,
                                       s["alpha"])

    def desk(self, fx, t):
        """The 2 x 3 map of the desk's drift at clip second `t` (the sheet's px to the frame's): turned about the frame's
        middle and moved, as mkmmd.core.sheet.drift_at says."""
        dx, dy, turn = ST.drift_at(fx["drift"], t, fx["seed"])
        W, H = self.size
        r = math.radians(turn)
        c, s = math.cos(r), math.sin(r)
        C = ((W - 1) / 2.0, (H - 1) / 2.0)
        return [[c, s, C[0] + dx * H - (c * C[0] + s * C[1])], [-s, c, C[1] + dy * H - (-s * C[0] + c * C[1])]]

    def _figure_map(self, fx, g, st):
        """The map that lays the figure of frame `g` on the sheet: set in and moved (the collage's pose map), then its
        slap's lift and squash about its middle. (map, the slap's state)."""
        M = self._pose_map(fx, g)
        M = _shift(0.0, 0.0) if M is None else M
        sl = ST.slap_at(fx["slap"], (g - self.frame0) / self.fps)
        if sl["scale"] != 1.0 or sl["sx"] != 1.0 or sl["sy"] != 1.0:
            c = _apply(M, st["own"])
            M = FX.compose(_scaled(c, sl["scale"] * sl["sx"], sl["scale"] * sl["sy"]), M)
        return M, sl

    def _hollow_tile(self, fx, h=None):
        """A hollow outline's layer, made once: (premultiplied rgb, coverage, its corner (x, y) on the sheet). `h` a
        hollow of `hollows` (the figure at its `t`, set at its own `at` and `height` or where the sticker was then, turned
        and mirrored), or None for the one the peel leaves (the figure as it lay when the peel began)."""
        FF = _ff()
        W, H = self.size
        if h is None:
            g = FX.frame_of(fx["peel"]["t"], self.fps, self.frame0)
            M, _ = self._figure_map(fx, g, self._sheets[fx["index"]])
            sheen, dash = ST.HOLLOW["sheen"], ST.HOLLOW["dash"]
        else:
            g = FX.frame_of(h["t"], self.fps, self.frame0)
            M = self._hollow_map(fx, h, g)
            sheen, dash = h["sheen"], h["dash"]
        m = self._figure(fx, g)
        if m is None:
            raise FileNotFoundError(f"fx {fx['index']} (sheet): no figure at frame {g} for its outline")
        cov = FF._warp(m, M, self.size)
        e = self._sheet_edge(fx)
        sh = self._sheet_shadow(fx)
        reach = (0 if e is None else e[1]) + (0 if sh is None else math.hypot(*sh[1]) + 3.0 * sh[2])
        b = _box(cov, int(math.ceil(reach)) + 2)
        if b is None:
            return None
        x0, y0, x1, y1 = b
        light = fx["paper"]["light"] if fx["paper"] is not None else WHITE
        ink = fx["paper"]["ink"] if fx["paper"] is not None else (0.34, 0.32, 0.47)
        rgb, a = hollow_layer(np.ascontiguousarray(cov[y0:y1, x0:x1]), e, sh, sheen, dash, self.px, ink, light)
        return rgb, a, (x0, y0)

    def _hollow_map(self, fx, h, g):
        """The map that lays the figure of frame `g` as hollow `h`: where the sticker lay then, or by its own `at` and
        `height` about its own pivot; turned `turn` and mirrored about that point."""
        W, H = self.size
        m = self._figure(fx, g)
        if m is None or not (m >= 0.5).any():
            raise FX.FxError(f"fx {fx['index']} (sheet): hollow at {h['t']:g} s: the figure is not in the frame there")
        if h["at"] is None and h["height"] is None:
            M = self._pose_map(fx, g)
            M = _shift(0.0, 0.0) if M is None else M
            p = _apply(M, _ff().centroid(m))
        else:
            ys, xs = np.nonzero(m >= 0.5)
            pv = fx["pivot"]
            if not isinstance(pv, str):
                a = (pv[0] * W - 0.5, pv[1] * H - 0.5)
            elif pv == "center":
                a = _ff().centroid(m)
            else:
                a = ((xs.min() + xs.max()) / 2.0, ys.max() + 0.5)
            if h["height"] is not None:
                s = h["height"] * H / float(ys.max() - ys.min() + 1)
            else:
                P = self._placing(fx)[0]
                s = 1.0 if P is None else math.hypot(P[0][0], P[1][0])
            if h["at"] is not None:
                p = (h["at"][0] * W - 0.5, h["at"][1] * H - 0.5)
            else:
                M = self._pose_map(fx, g)
                p = a if M is None else _apply(M, a)
            M = FX.placement(a, p, s)
        if h["turn"] or h["mirror"]:
            r = math.radians(h["turn"])
            c, s_ = math.cos(r), math.sin(r)
            f = -1.0 if h["mirror"] else 1.0
            A = np.array([[c * f, s_], [-s_ * f, c]])
            T = np.hstack([A, (np.asarray(p) - A @ np.asarray(p))[:, None]])
            M = FX.compose(T.tolist(), M)
        return M

    def _blank_of(self, fx, h):
        """What the blank sticker of hollow `h` (its own `peel`) keeps: (its peel, the figure's coverage at the hollow's
        second, the map that lays it in the hollow)."""
        g = FX.frame_of(h["t"], self.fps, self.frame0)
        return h["peel"], self._figure(fx, g), self._hollow_map(fx, h, g)

    def _blank(self, fx, blank, t, out, D, shadow):
        """`out` with a hollow's blank sticker at clip second `t`: the figure's shape in the peel's flat colour (gone
        BLANK_FADE of the way to the paper's light over its `fade`) in the sheet's rim and shadow, lying in its hollow
        until its peel and rolled off by it as the figure's sticker is; gone after."""
        p, m, M = blank
        if ST.peeled(p, t):
            return out
        FF = _ff()
        cov = np.clip(FF._warp(m, FX.compose(D, M), self.size), 0.0, 1.0)
        colour = np.asarray(p["color"], np.float32)
        if p["fade"] is not None:
            k = float(np.clip((t - p["fade"][0]) / (p["fade"][1] - p["fade"][0]), 0.0, 1.0))
            light = np.asarray(fx["paper"]["light"] if fx["paper"] is not None else WHITE, np.float32)
            colour = colour + (light - colour) * np.float32(ST.BLANK_FADE * k)
        e = self._sheet_edge(fx)
        ring = FF._grow(cov, e[1]) if e is not None and e[1] > 0 else cov
        face = colour * cov[..., None] + (np.asarray(e[0], np.float32) * (ring - cov)[..., None] if e is not None else 0.0)
        u = ST.peel_progress(p, t)
        if u is not None:
            return self._peeled(out, face, ring, u, shadow, p["corner"], p["radius"] * min(self.size), p["back"])
        if shadow is not None and shadow[3] > 0:
            out = FF._cast(out, ring, shadow)
        return face + out * (1.0 - ring[..., None])

    def _doodle_plan(self, fx, st):
        """A sheet's furniture laid out for this output, clear of the figure, the hollows and the `clear` boxes: the
        doodles' draws, places, ripple shares, peel times, the hops' times and where a jump leaves from; the
        scribbles' places (clear of the doodles too) and the tapes' places at the sheet's edges."""
        dd, sc, tp = fx["doodles"], fx["scribbles"], fx["tapes"]
        if (dd is None or dd["count"] == 0) and sc is None and tp is None:
            return None
        W, H = self.size
        taken = np.zeros((H, W), np.float32) if st.get("taken") is None else np.array(st["taken"])
        e = self._sheet_edge(fx)
        for h in fx["hollows"]:                              # a hollow's whole shape, its rim in: nothing is laid in it,
            g = FX.frame_of(h["t"], self.fps, self.frame0)   # nor on the blank sticker that may lie there (the hollow's
            m = self._figure(fx, g)                          # own layer is too faint inside its rim to say where it is)
            if m is not None:
                cov = _ff()._warp(m, self._hollow_map(fx, h, g), self.size)
                taken = np.maximum(taken, (_ff()._grow(cov, e[1]) if e is not None and e[1] > 0 else cov) >= 0.5)
        far = cv2.distanceTransform((taken < 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        def away(p):
            xs = np.clip(np.rint(p[:, 0] * H - 0.5).astype(int), 0, W - 1)
            ys = np.clip(np.rint(p[:, 1] * H - 0.5).astype(int), 0, H - 1)
            return far[ys, xs] / H
        centre = (0.5, 0.5) if st["centre"] is None else ((st["centre"][0] + 0.5) / W, (st["centre"][1] + 0.5) / H)
        out = {"deal": [], "places": [], "wave": [], "hops": [], "origin": None, "peel": {}, "scribbles": [],
               "tapes": []}
        clear = list(dd["clear"]) if dd is not None else []
        if dd is not None and dd["count"] > 0:
            deal = ST.deal(dd, fx["seed"])
            e = self._sheet_edge(fx)
            ew = 0.0 if e is None else e[1] / H
            radii = [ST.REACH * dd["size"] * d["size"] + ew for d in deal]
            places = ST.scatter(radii, W / H, dd["area"], clear, dd["gap"], fx["seed"] * 1009 + dd["seed"], away)
            hops = []
            hp = dd["hop"]
            if hp is not None and hp["beats"]:
                hits = self._hits or {}
                if "beats" not in hits:
                    raise FX.FxError(f"fx {fx['index']} (sheet): hop beats = true needs the timeline's beats "
                                     f"(audio/timeline.json or [audio] timeline)")
                hops = ST.beat_times(hits["beats"], hits.get("downbeats") or [], hp["every"])
            elif hp is not None:
                hops = FX.hit_times({**hp, "index": fx["index"], "kind": "sheet", "from": fx["from"] - 2.0,
                                     "to": fx["to"] + 1.0}, self._hits)
            out.update(deal=deal, places=places, wave=ST.ripple(places, centre), hops=hops,
                       origin=None if dd["jump"] is None else
                       centre if dd["jump"]["from"] == "figure" else tuple(dd["jump"]["from"]),
                       peel=ST.peel_order(dd, places, centre, fx["seed"]))
            clear += [(p[0] - r * H / W, p[1] - r, p[0] + r * H / W, p[1] + r)
                      for p, r in zip(places, radii) if p is not None]
        if sc is not None and sc["count"] > 0:
            area = dd["area"] if dd is not None else ST.DOODLES["area"]
            radii = [0.6 * sc["size"]] * sc["count"]
            got = ST.scatter(radii, W / H, area, clear, 0.004, fx["seed"] * 7919 + sc["seed"], away)
            rng = np.random.default_rng([int(fx["seed"]), int(sc["seed"]), 29])
            out["scribbles"] = [{"at": p, "shape": sc["shapes"][j % len(sc["shapes"])],
                                 "ink": sc["inks"][int(rng.integers(len(sc["inks"])))],
                                 "angle": float(rng.uniform(-25, 25)), "size": sc["size"] * float(rng.uniform(0.75, 1.3))}
                                for j, p in enumerate(got) if p is not None]
        if tp is not None and tp["count"] > 0:
            out["tapes"] = ST.tape_places(tp, W / H, fx["seed"], away)
        return out

    def _doodle_tile(self, fx, d, size_px):
        dd = fx["doodles"]
        main = tuple(FX.pick(ST.colours(dd, d["shape"]), d["color"]))
        accent, alt, _ = dd["own"][d["shape"]]
        accent = alt if tuple(accent) == main else accent
        e = self._sheet_edge(fx)
        ew = 0.0 if e is None else e[1]
        key = (d["shape"], main, tuple(accent), tuple(dd["ink"]), int(round(size_px)), round(ew, 2))
        t = self._sheet_tiles.get(key)
        if t is None:
            pen = 0.0 if d["shape"] in MD.PENNED else MD.PEN
            t = self._sheet_tiles[key] = MDF.shape_tile(d["shape"], main, tuple(accent), tuple(dd["ink"]),
                                                        int(round(size_px)), ew, pen)
        return t

    # ---------------------------------------------------------------- the sheet
    def _sheet(self, fx, frame, img):
        FF = _ff()
        W, H = self.size
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0:                                          # switched off (an output that does not want it)
            return img
        st = self._sheet_state(fx)
        D = self.desk(fx, t)
        if st["paper"] is not None:
            pad = st["pad"]
            out = cv2.warpAffine(st["paper"], np.float32(FX.compose(D, _shift(-pad, -pad))), (W, H),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if st["lamp"] is not None:
                out *= st["lamp"]
        else:
            out = np.array(img, np.float32)
        shadow = self._sheet_shadow(fx)
        plan = st["doodles"]
        if plan is not None and plan["scribbles"]:
            self._scribbles(fx, t, out, D, plan["scribbles"])
        for got in st["hollows"]:
            if got is not None:
                self._lay_layer(out, got, D)
        if plan is not None and plan["tapes"]:
            if "tape_layer" not in st:
                st["tape_layer"] = tape_layer(self.size, plan["tapes"], fx["tapes"], fx["seed"], shadow)
            if st["tape_layer"] is not None:
                self._lay_layer(out, st["tape_layer"], D)
        for blank in st.get("blanks", ()):
            out = self._blank(fx, blank, t, out, D, shadow)
        peel = fx["peel"] if fx["objects"] is not None else None
        u = ST.peel_progress(peel, t)
        if u is not None:
            if "peel_hollow" not in st:
                st["peel_hollow"] = self._hollow_tile(fx)
            if st["peel_hollow"] is not None:
                self._lay_layer(out, st["peel_hollow"], D)
        if st["doodles"] is not None:
            self._doodles(fx, t, out, D, st, shadow)
        if fx["objects"] is not None and fx["slap"]:
            self._dust(fx, t, out, D, st)
        if fx["objects"] is not None and not ST.peeled(peel, t):
            m = self._figure(fx, frame)
            if m is not None:
                pic = self._shot_frame(fx, frame) if fx.get("shot") else img
                M, sl = self._figure_map(fx, frame, st)
                Mt = FX.compose(D, M)
                cov = np.clip(FF._warp(m, Mt, self.size), 0.0, 1.0)
                rgb = FF._warp(pic * m[..., None], Mt, self.size)
                e = self._sheet_edge(fx)
                ring = FF._grow(cov, e[1]) if e is not None and e[1] > 0 else cov
                face = rgb + (np.asarray(e[0], np.float32) * (ring - cov)[..., None] if e is not None else 0.0)
                if u is None:
                    sh = lift_shadow(shadow, sl["lift"])
                    if sh is not None and sh[3] > 0:
                        out = FF._cast(out, ring, sh)
                    out = face + out * (1.0 - ring[..., None])
                else:
                    p = fx["peel"]
                    out = self._peeled(out, face, ring, u, shadow, p["corner"], p["radius"] * min(self.size), p["back"])
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _lay_layer(self, out, got, D):
        rgb, a, (x0, y0) = got
        MDF.lay(out, MDF.Tile(rgb, a, None, (0.0, 0.0)), FX.compose(D, _shift(x0, y0)))

    def _scribbles(self, fx, t, out, D, items):
        """The pen marks on the paper, redrawn `fps` times a second by a hand that wobbles (they boil), carried by the
        desk's drift."""
        sc = fx["scribbles"]
        W, H = self.size
        step = int(math.floor(t * sc["fps"])) if sc["fps"] > 0 else 0
        for j, it in enumerate(items):
            lines = ST.scribble_lines(it["shape"])
            n = sum(len(ln) for ln in lines)
            rng = np.random.default_rng([int(fx["seed"]), int(sc["seed"]), j, step])
            jitter = rng.normal(0.0, sc["wobble"] * self.px, (n, 2))
            jitter = cv2.GaussianBlur(jitter.astype(np.float32).reshape(-1, 1, 2), (1, 5), 0).reshape(-1, 2) \
                if n >= 5 else jitter
            at = _apply(D, (it["at"][0] * W - 0.5, it["at"][1] * H - 0.5))
            scribble(out, lines, np.asarray(at), it["size"] * H, it["angle"] + _turn_of(D), it["ink"],
                     sc["width"] * self.px, jitter)

    def _doodles(self, fx, t, out, D, st, shadow):
        """The doodles at clip second `t`, the highest lifted last; one moving fast (a jump) is drawn at samples over a
        frame's shutter, so it streaks."""
        dd, plan = fx["doodles"], st["doodles"]
        W, H = self.size
        e = self._sheet_edge(fx)
        dt = 1.0 / self.fps

        def state(j, d, at, tau):
            return ST.doodle_at(tau, d, at, plan["wave"][j], dd["jump"], plan["hops"], dd["hop"], plan["peel"].get(j),
                                dd["peel"]["dur"] if dd["peel"] else 0.0, plan["origin"])
        drawn = []
        for j, (d, at) in enumerate(zip(plan["deal"], plan["places"])):
            if at is not None:
                s = state(j, d, at, t)
                if s["shown"]:
                    drawn.append((s["lift"], j, d, at, s))
        for lift, j, d, at, s in sorted(drawn, key=lambda x: (x[0], x[1])):
            size = dd["size"] * d["size"] * H
            tile = self._doodle_tile(fx, d, size)

            def placed(s):
                at_px = _apply(D, (s["at"][0] * W - 0.5, s["at"][1] * H - 0.5))
                return _sticker_map(at_px, s["angle"] + _turn_of(D), s["scale"] * s["sx"], s["scale"] * s["sy"],
                                    tile.anchor)
            if s["peel"] is not None:
                self._peel_doodle(fx, out, tile, placed(s), d, s["peel"], size, shadow)
                continue
            a, b = state(j, d, at, t - 0.5 * dt), state(j, d, at, t + 0.5 * dt)
            move = math.hypot((b["at"][0] - a["at"][0]) * W, (b["at"][1] - a["at"][1]) * H) + \
                math.radians(abs(b["angle"] - a["angle"])) * size / 2.0
            n = int(min(max(math.ceil(move / (4.0 * self.px)), 1), FLING_SAMPLES))
            if n == 1 or not (a["shown"] and b["shown"]):
                MDF.lay(out, tile, placed(s), None if e is None else e[0], lift_shadow(shadow, lift))
                continue
            acc = None
            for i in range(n):
                si = state(j, d, at, t + ((i + 0.5) / n - 0.5) * dt)
                got = stamp(tile, placed(si), self.size, None if e is None else e[0], lift_shadow(shadow, si["lift"]))
                if got is not None:
                    acc = _accumulate(acc, *got)
            if acc is not None:
                box, rgb, al = acc
                put(out, box, rgb / n, al / n)

    def _dust(self, fx, t, out, D, st):
        """The paper dust a slap throws up: specks from round the sticker's outline flying out and settling."""
        W, H = self.size
        for k, sl in enumerate(fx["slap"]):
            age = t - sl["t"]
            if sl["dust"] <= 0 or not 0.0 <= age < DUST_LIFE or st.get("taken") is None:
                continue
            if "rim" not in st:                       # the sticker's outline as it lies, where the dust flies from
                cs, _ = cv2.findContours((st["taken"] >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pts = np.concatenate([c[:, 0, :] for c in cs]).astype(np.float64) if cs else np.zeros((0, 2))
                st["rim"] = pts[(pts[:, 0] > 0) & (pts[:, 0] < W - 1) & (pts[:, 1] > 0) & (pts[:, 1] < H - 1)]
            if not len(st["rim"]):
                continue
            rng = np.random.default_rng([int(fx["seed"]), k, 23])
            n = sl["dust"]
            p0 = st["rim"][rng.integers(len(st["rim"]), size=n)]
            out_of = p0 - np.asarray(st["centre"])
            a = np.arctan2(out_of[:, 1], out_of[:, 0]) + rng.normal(0, 0.45, n)
            dirs = np.stack([np.cos(a), np.sin(a)], 1)
            speed = rng.uniform(0.15, 0.9, n) * H
            travel = (speed * DUST_TAU * (1.0 - math.exp(-age / DUST_TAU)))[:, None]
            pos = p0 + dirs * travel
            rad = rng.uniform(1.8, 6.0, n) * self.px
            kinds = rng.random(n)
            fade = (1.0 - age / DUST_LIFE) ** 1.5
            shade_px = 1.6 * self.px
            paper = fx["paper"] or {"shade": (0.95, 0.91, 0.88), "ink": (0.34, 0.32, 0.47)}
            ink = np.asarray(paper["ink"], np.float32)
            dusk = tuple(float(c) for c in np.asarray(paper["shade"], np.float32) * 0.6 + ink * 0.4)
            for layer in (0, 1):                   # each speck's little shadow first, then the specks: paper bits
                canvas = np.zeros((H, W, 4), np.uint8)
                for i in range(n):
                    x, y = _apply(D, pos[i])
                    if not (-10 <= x < W + 10 and -10 <= y < H + 10):
                        continue
                    if layer == 0:
                        col, alpha = tuple(ink), 0.3 * fade
                        x, y = x + shade_px, y + 1.5 * shade_px
                    else:
                        col = WHITE if kinds[i] < 0.55 else dusk if kinds[i] < 0.9 else tuple(ink)
                        alpha = fade * (1.0 if kinds[i] < 0.9 else 0.5)
                    cv2.circle(canvas, (int(round(x * 16)), int(round(y * 16))), int(round(rad[i] * 16)),
                               tuple(int(round(c * 255)) for c in col) + (int(round(alpha * 255)),), -1, cv2.LINE_AA,
                               4)
                c = canvas.astype(np.float32) / 255.0
                a_ = c[..., 3:4]
                out *= 1.0 - a_
                out += c[..., :3] * a_

    # ---------------------------------------------------------------- peels
    def _peeled(self, out, face, ring, u, shadow, corner, R, back, own=False):
        """`out` with a sticker (premultiplied `face`, coverage `ring`, rim in) peeled from its `corner`, `u` of its time
        through, rolled over a roll `R` px across (mkmmd.peel.sheet, the peel transition's; the fold along
        mkmmd.core.sheet.peel_roll, from the sticker's own point nearest that corner, grip_of: it is seen to lift from
        the first moment): the part still stuck, the shadow of what is lifted (in the sheet's shadow colour)
        and the lifted part, its paper `back` up once it lies back over itself. The figure leaves by the frame's far
        side; with `own` (a doodle) the sticker only lifts clear of its own place, flips up and fades over the last
        third, and the work is kept to the box round it."""
        n = PE.normal({"from": corner, "angle": 0.0})
        b = _box(ring)
        if b is None:
            return out
        along = [x * n[0] + y * n[1] for x in (b[0] - 0.5, b[2] - 0.5) for y in (b[1] - 0.5, b[3] - 0.5)]
        grip = ST.grip_of(ring[b[1]:b[3], b[0]:b[2]], corner)
        c0 = max(along) if grip is None else ST.peel_start((grip[0] + b[0], grip[1] + b[1]), n)
        W, H = self.size
        if own:
            lo, reach = min(along), int(2.5 * max(b[2] - b[0], b[3] - b[1])) + int(4 * R)
            box = (max(b[1] - reach, 0), min(b[3] + reach, H), max(b[0] - reach, 0), min(b[2] + reach, W))
        else:
            lo = min(x * n[0] + y * n[1] for x in (-0.5, W - 0.5) for y in (-0.5, H - 0.5))
            box = None
        c, f = ST.peel_roll(n, c0, lo, R, u)
        straight = face / np.maximum(ring, 1e-4)[..., None]
        sh = PL.sheet(straight, ring, back, n, c, R, f, box)
        ink, k = ((0.0, 0.0, 0.0), 0.0) if shadow is None else (shadow[0], min(1.5 * shadow[3], 1.0))
        res = sh.over(out, ink, k)
        if own and u > 0.65:
            s = min((u - 0.65) / 0.35, 1.0)
            res = res + (out - res) * np.float32(s * s * (3.0 - 2.0 * s))
        return res

    def sticker_grip(self, ride, frame):
        """Where the grip of a sticker that is peeled off a sheet is seen at `frame`, in frame fractions, for a cutout
        whose `at` rides it (`ride` = {"sheet": the sheet's index, "hollow": its hollow's place in `hollows`, or None for
        the figure's own peel}): the sticker's own point nearest the pinched corner as it lay when the peel began
        (mkmmd.core.sheet.grip_of), carried by the desk's drift and lifted by the peel (peel_grip)."""
        fx = next(f for f in self.fxs if f["index"] == ride["sheet"])
        st = self._sheet_state(fx)
        key = ("grip", ride["hollow"])
        if key not in st:
            if ride["hollow"] is None:
                peel = fx["peel"]
                g = FX.frame_of(peel["t"], self.fps, self.frame0)
                m, M = self._figure(fx, g), self._figure_map(fx, g, st)[0]
            else:
                peel, m, M = self._blank_of(fx, fx["hollows"][ride["hollow"]])
            if m is None:
                raise FileNotFoundError(f"fx {fx['index']} (sheet): no figure at its peel's frame for a hand to take")
            cov = _ff()._warp(m, M, self.size)
            e = self._sheet_edge(fx)
            grip = ST.grip_of(_ff()._grow(cov, e[1]) if e is not None and e[1] > 0 else cov, peel["corner"])
            if grip is None:
                raise FX.FxError(f"fx {fx['index']} (sheet): the sticker is not in the frame when its peel begins")
            st[key] = (peel, grip)
        peel, grip = st[key]
        t = self._t(frame)
        return ST.peel_grip(peel, t, _apply(self.desk(fx, t), grip), self.size)["p"]

    def _peel_back(self, fx):
        if fx["peel"] is not None:
            return fx["peel"]["back"]
        return fx["paper"]["light"] if fx["paper"] is not None else WHITE

    def _peel_doodle(self, fx, out, tile, M, d, u, size, shadow):
        """A doodle peeled off, `u` of its time through: laid alone on a clear frame and rolled back as the figure's
        sticker is, from the corner away from the nearest edge of the frame (it lifts toward that edge), clear of its
        place and gone."""
        e = self._sheet_edge(fx)
        got = stamp(tile, M, self.size, None if e is None else e[0], None)
        if got is None:
            return
        (x0, y0, x1, y1), rgb, a = got
        W, H = self.size
        face, ring = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
        face[y0:y1, x0:x1], ring[y0:y1, x0:x1] = rgb, a
        dp = fx["doodles"]["peel"]
        corner = dp["corner"]
        if corner is None:                       # it slides toward -n: pinch the side away from the nearest edge
            cx, cy = (x0 + x1) / 2.0 / W, (y0 + y1) / 2.0 / H
            horiz = "right" if cx < 0.5 else "left"
            vert = "bottom" if cy < 0.5 else "top"
            corner = f"{vert}-{horiz}"
        out[...] = self._peeled(out, face, ring, u, shadow, corner, dp["radius"] * size, self._peel_back(fx), True)

    # ---------------------------------------------------------------- fling
    def _fling(self, fx, frame, img):
        """Stickers thrown at the lens: the farthest first, each streaked over half a frame's shutter (short streaks in
        its own colours), its rim in proportion to it, its white back for the frame or two it flips over."""
        t = self._t(frame)
        W, H = self.size
        c = self._at(fx, frame)[0]
        items = self._flings.get(fx["index"])
        if items is None:
            aim = None if fx["aim"] is None else ((fx["aim"][0] * W - 0.5 - c[0]) / H, (fx["aim"][1] * H - 0.5 - c[1]) / H)
            items = self._flings[fx["index"]] = ST.fling_items(fx, self._hit_times(fx), aim)
        live = []
        for it in items:
            s = ST.fling_at(fx, it, t)
            if s is not None:
                live.append((s["depth"], it, s))
        if not live:
            return img
        out = np.array(img, np.float32)
        e = None if fx["edge"] is None else (fx["edge"]["color"], fx["edge"]["width"] * self.px)
        s0 = fx["shadow"]
        dt = FLING_SHUTTER / self.fps
        for _, it, s in sorted(live, key=lambda x: -x[0]):
            a = ST.fling_at(fx, it, t - 0.5 * dt) or s
            b = ST.fling_at(fx, it, t + 0.5 * dt) or s
            move = math.hypot((b["off"][0] - a["off"][0]) * H, (b["off"][1] - a["off"][1]) * H)
            n = int(min(max(math.ceil(move / (4.0 * self.px)), 1), FLING_SAMPLES))
            size = s["scale"] * H
            q = SIZE_STEP ** math.ceil(math.log(max(size, 2.0)) / math.log(SIZE_STEP))
            ew = 0.0 if e is None else e[1] * q / (FLING_RIM_AT * H)
            d = {"shape": it["shape"], "color": it["color"]}
            tile = self._fling_tile(fx, d, q, ew)
            acc = None
            for i in range(n):
                tau = t + ((i + 0.5) / n - 0.5) * dt if n > 1 else t
                si = ST.fling_at(fx, it, tau) or s
                at = (c[0] + si["off"][0] * H, c[1] + si["off"][1] * H)
                k = si["scale"] * H / q
                face = si["face"]
                M = _sticker_map(at, si["angle"], k * max(abs(face), 0.04), k, tile.anchor)
                m = si["scale"] / fx["size"]
                sh = None if s0 is None else (s0["color"], (s0["offset"][0] * H * m, s0["offset"][1] * H * m),
                                              s0["blur"] * self.px * m, s0["alpha"])
                shade = 0.82 + 0.18 * abs(face)
                back = None if face >= 0 else tuple(np.asarray(e[0] if e else WHITE) * shade)
                got = stamp(tile, M, self.size, None if e is None else e[0], sh, back)
                if got is None:
                    continue
                box, rgb, al = got
                if face >= 0 and shade < 1.0:
                    rgb = rgb * np.float32(shade)
                acc = _accumulate(acc, box, rgb, al)
            if acc is not None:
                box, rgb, al = acc
                put(out, box, rgb / n, al / n)
        st = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        return out if st >= 1.0 else (img * (1.0 - st) + out * st).astype(np.float32)

    def _fling_tile(self, fx, d, size_px, edge_px):
        main = tuple(FX.pick(ST.colours(fx, d["shape"]), d["color"]))
        accent, alt, _ = fx["own"][d["shape"]]
        accent = alt if tuple(accent) == main else accent
        key = ("fling", d["shape"], main, tuple(accent), tuple(fx["ink"]), int(round(size_px)), round(edge_px, 2))
        t = self._sheet_tiles.get(key)
        if t is None:
            pen = 0.0 if d["shape"] in MD.PENNED else MD.PEN
            t = self._sheet_tiles[key] = MDF.shape_tile(d["shape"], main, tuple(accent), tuple(fx["ink"]),
                                                        int(round(size_px)), edge_px, pen)
        return t


def _turn_of(M):
    """The turn (degrees counter-clockwise) of a 2 x 3 map."""
    M = _affine(M)
    return math.degrees(math.atan2(M[0, 1], M[0, 0]))


def _accumulate(acc, box, rgb, a):
    """Premultiplied samples summed over the box that holds them all."""
    if acc is None:
        return box, rgb.copy(), a.copy()
    (X0, Y0, X1, Y1), R, A = acc
    x0, y0, x1, y1 = box
    U = (min(X0, x0), min(Y0, y0), max(X1, x1), max(Y1, y1))
    if U != (X0, Y0, X1, Y1):
        R2 = np.zeros((U[3] - U[1], U[2] - U[0], 3), np.float32)
        A2 = np.zeros((U[3] - U[1], U[2] - U[0]), np.float32)
        R2[Y0 - U[1]:Y1 - U[1], X0 - U[0]:X1 - U[0]] = R
        A2[Y0 - U[1]:Y1 - U[1], X0 - U[0]:X1 - U[0]] = A
        R, A = R2, A2
    R[y0 - U[1]:y1 - U[1], x0 - U[0]:x1 - U[0]] += rgb
    A[y0 - U[1]:y1 - U[1], x0 - U[0]:x1 - U[0]] += a
    return U, R, A
