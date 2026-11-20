"""Craft type, the drawing (docs/design.md: Text, Craft type): the lyric lines `mk post` lays on the picture as things made
by hand, each unit one sprite drawn once at rest and carried onto every frame by one map (mkmmd.core.crafttype decides
which units, when, where and how they move). The crafts:

- emboss: one strip of glossy label tape in a pop colour, the capitals raised in pale plastic with the stress halo
  embossing leaves round each letter, the ends cut serrated;
- masking: a torn strip of cream masking tape (fibrous ends, a little translucent) with the line written on it by a pen
  that runs along it;
- sticker: the line as one die-cut sticker, heavy round letters, each word its own colour, one white rim round the union
  of the letters (their counters filled);
- cutpaper: a paper label taped at its corners, its letters cut from coloured papers with scissors (straight cuts), each
  a little off;
- brush: letters painted in one stroke along the line, dry-brushed where the paint runs out.

Every unit wears the sheet's soft drop shadow, which grows and softens while the unit is in the air. Images are float
RGB in display space (0..1), (h, w, 3); a sprite is premultiplied RGBA. Lyric text reaches only the font rasteriser."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import crafttype as CT
from .core import palette as PAL
from .core.typeset import TextError

SS = 2                     # sprites are drawn this many times larger and boxed down
REF_CAP = 48.0             # px: the cap height the units are measured at for the layout
WHITE = np.float32([1.0, 1.0, 1.0])


# ================================================================================================= small helpers
@lru_cache(maxsize=32)
def _font(path, em):
    return ImageFont.truetype(path, int(em))


@lru_cache(maxsize=32)
def _cap_ratio(path):
    """A font's cap height (the ink of an H) per em."""
    f = _font(path, 400)
    l, t, r, b = f.getbbox("H", anchor="ls")
    return max(b - t, 1) / 400.0


def _noise(shape, cell, seed, octaves=1):
    """Smooth value noise in [-1, 1] of `shape` (h, w), features about `cell` px across, `octaves` halving each time."""
    h, w = shape
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for _ in range(max(octaves, 1)):
        gh, gw = max(int(h / max(cell, 1.0)) + 3, 3), max(int(w / max(cell, 1.0)) + 3, 3)
        g = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        cell *= 0.5
    return out / np.float32(tot)


def _streaks(shape, length, width, angle, seed):
    """Noise stretched along `angle` degrees: fibres, brush hairs, crepe. [-1, 1]."""
    h, w = shape
    d = int(math.ceil(math.hypot(h, w))) + 4
    rng = np.random.default_rng(seed)
    gh, gw = max(int(d / max(width, 1.0)), 3), max(int(d / max(length, 1.0)), 3)
    g = cv2.resize(rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32), (d, d), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), angle, 1.0)
    M[0, 2] -= (d - w) / 2.0
    M[1, 2] -= (d - h) / 2.0
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _blur(a, sigma):
    return a if sigma < 0.3 else cv2.GaussianBlur(a, (0, 0), float(sigma))


def _fill_holes(mask):
    """A coverage with every hole inside it filled (a die cut follows the outer contour only)."""
    m8 = (mask >= 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full = np.zeros_like(m8)
    cv2.drawContours(full, cs, -1, 1, -1)
    return np.maximum(mask, full.astype(np.float32))


def _grow(a, r):
    """Coverage `a` grown by `r` px, antialiased, rounded."""
    if r <= 0:
        return a
    inside = (a >= 0.5).astype(np.uint8)
    if not inside.any():
        return a
    d = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.maximum(np.clip(np.float32(r + 0.5) - d, 0.0, 1.0), a)


def _over(dst, rgb, a):
    """Premultiplied RGBA `dst` (h, w, 4), in place, with colour `rgb` ((3,) or (h, w, 3)) laid over where `a` covers."""
    a = np.clip(a, 0.0, 1.0)[..., None]
    dst[..., :3] = dst[..., :3] * (1.0 - a) + np.asarray(rgb, np.float32) * a
    dst[..., 3:] = dst[..., 3:] * (1.0 - a) + a
    return dst


def _mix(a, b, t):
    return np.asarray(a, np.float32) * (1.0 - t) + np.asarray(b, np.float32) * t


def _shade(rgb, k):
    """`rgb` darkened (k < 0) or lightened (k > 0) by |k| (toward black is toward the palette's ink: multiplied)."""
    rgb = np.asarray(rgb, np.float32)
    if np.ndim(k):
        k = np.asarray(k, np.float32)[..., None]
    return np.where(np.asarray(k) >= 0, rgb + (1.0 - rgb) * k, rgb * (1.0 + k)).astype(np.float32)


def _poly_mask(shape, pts, ss=4):
    """Antialiased coverage of the polygon `pts` (px) in an image of `shape`."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    P = np.round(np.asarray(pts, np.float64) * (1 << ss)).astype(np.int32)
    cv2.fillPoly(m, [P], 255, cv2.LINE_AA, ss)
    return m.astype(np.float32) / 255.0


# ================================================================================================= setting the letters
class Glyph:
    """One letter set on a line: its coverage `mask` (float32) with its top left at (x, y) px from the line's origin
    (the first letter's pen position on the baseline), the word `word` and letter `i` it is."""
    __slots__ = ("mask", "x", "y", "word", "i", "ch")

    def __init__(self, mask, x, y, word, i, ch):
        self.mask, self.x, self.y, self.word, self.i, self.ch = mask, x, y, word, i, ch


def set_words(words, path, cap, tracking=0.0, gap=0.32):
    """The letters of `words` (strings) set on one line in the font at `path` with capitals `cap` px tall: [Glyph],
    the line's advance (px) and the word spans [(x0, x1)]. `tracking` and `gap` (between words) are in caps."""
    em = max(int(round(cap / _cap_ratio(path))), 4)
    font = _font(path, em)
    out, spans, x = [], [], 0.0
    for j, word in enumerate(words):
        x_start = x
        for i, ch in enumerate(word):
            if ch.isspace():
                continue
            pre = font.getlength(word[:i]) + i * tracking * cap
            l, t, r, b = font.getbbox(ch, anchor="ls")
            if r <= l or b <= t:
                continue
            pad = 2
            im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
            ImageDraw.Draw(im).text((pad - l, pad - t), ch, font=font, fill=255, anchor="ls")
            out.append(Glyph(np.asarray(im, np.float32) / 255.0, x + pre + l - pad, t - pad, j, i, ch))
        x += font.getlength(word) + max(len(word) - 1, 0) * tracking * cap
        spans.append((x_start, x))
        x += gap * cap
    return out, x - gap * cap, spans


def ink_box(glyphs):
    """(x0, y0, x1, y1): the ink of the letters, line coordinates."""
    xs0 = min(g.x for g in glyphs)
    ys0 = min(g.y for g in glyphs)
    xs1 = max(g.x + g.mask.shape[1] for g in glyphs)
    ys1 = max(g.y + g.mask.shape[0] for g in glyphs)
    return xs0, ys0, xs1, ys1


def paste(canvas, mask, x, y, op=np.maximum):
    """`canvas` (in place) with `mask` combined by `op` at its top left (x, y) (rounded), cut at the edges."""
    x, y = int(round(x)), int(round(y))
    h, w = mask.shape[:2]
    H, W = canvas.shape[:2]
    ax, ay, bx, by = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if bx > ax and by > ay:
        canvas[ay:by, ax:bx] = op(canvas[ay:by, ax:bx], mask[ay - y:by - y, ax - x:bx - x])
    return canvas


def turned(mask, deg):
    """A glyph's coverage turned `deg` degrees counter-clockwise about its middle, on a canvas that holds it all, and the
    (dx, dy) its top left moved by."""
    if abs(deg) < 1e-3:
        return mask, 0.0, 0.0
    h, w = mask.shape
    d = int(math.ceil(math.hypot(h, w))) + 2
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    M[0, 2] += (d - w) / 2.0
    M[1, 2] += (d - h) / 2.0
    return cv2.warpAffine(mask, M, (d, d), flags=cv2.INTER_LINEAR), -(d - w) / 2.0, -(d - h) / 2.0


# ================================================================================================= a unit at rest
class Sprite:
    """A unit drawn at rest: `body` (premultiplied RGBA, (h, w, 4)) is what is there from its landing, `ink` (the same,
    or None) what a pen or a brush lays on it (`order`: (h, w), when along the stroke each pixel of the ink is laid;
    `marks`: the order by which each word is all laid, in reading order), `back` the colour of its back (what a peel
    shows), `anchor` (x, y) px its centre, `box` its size (w, h) px as the layout counts it."""

    def __init__(self, body, ink=None, order=None, back=(1.0, 1.0, 1.0), box=None, marks=None):
        self.body, self.ink, self.order, self.marks = body, ink, order, marks
        self.back = np.asarray(back, np.float32)
        h, w = body.shape[:2]
        self.anchor = (w / 2.0, h / 2.0)
        self.box = box if box is not None else (w, h)
        self._rest = None

    def at(self, reveal):
        """The sprite as it shows with its ink laid as far as `reveal` (i + f: words 0..i-1 laid and f of word i, as
        mkmmd.core.crafttype.pen_progress gives it) (premultiplied RGBA)."""
        n = len(self.marks) if self.marks else 1
        if self.ink is None or reveal >= n:
            if self._rest is None:
                self._rest = self.body if self.ink is None else _stack(self.body, self.ink)
            return self._rest
        if reveal <= 0.0:
            return self.body
        i, f = int(reveal), reveal - int(reveal)
        lo = self.marks[i - 1] if i > 0 else float(self.order.min())
        thr = lo + f * (self.marks[i] - lo)
        m = np.clip((thr - self.order) / SOFT + 0.5, 0.0, 1.0)
        return _stack(self.body, self.ink * m[..., None])


SOFT = 0.025               # the width of the pen's or the brush's front, in the order's units
MARK = 0.02                # how far past a word's last pixel its mark lies (inside the front's soft edge)


def _stack(under, over):
    """Premultiplied `over` laid on premultiplied `under`."""
    return over + under * (1.0 - over[..., 3:4])


def _down(a):
    """A layer drawn SS times larger, boxed down."""
    if SS == 1:
        return a
    h, w = a.shape[:2]
    return cv2.resize(a, (max(w // SS, 1), max(h // SS, 1)), interpolation=cv2.INTER_AREA)


def build(p, u, cap, font, px=1.0):
    """Unit `u` of plan `p` drawn at rest with its capitals `cap` px tall (output pixels) in the font file `font`, on an
    output of `px` pixels per pixel at 1080 (what a rim's width is counted in): a Sprite. Its words are set in `rows`
    rows of about equal width."""
    c = p["craft"]
    if isinstance(c["color"], list):                             # a colour per unit, cycled
        c = {**c, "color": c["color"][u["k"] % len(c["color"])]}
    texts = [w.text for w in u["words"]]
    f = _font(font, 200)
    split = CT.split_rows([f.getlength(t) for t in texts], u["rows"], f.getlength(" ") * 1.5)
    rows = [[(j, texts[j]) for j in r] for r in split]
    return KINDS[c["kind"]](c, rows, cap * SS, font, u["seed"], px * SS)


def measure(p, u, font, cap_px, px=1.0):
    """Unit `u`'s size at rest at a cap height of 1 (its own, before its `scale`): (w, h) in caps, as the layout stacks
    it, for capitals of about `cap_px` output pixels (what a rim in pixels is measured against)."""
    s = build(p, u, REF_CAP, font, px * REF_CAP / max(cap_px, 1.0))
    return s.box[0] / REF_CAP, s.box[1] / REF_CAP


def set_rows(rows, path, cap, tracking, gap, pitch, align="center", stagger=0.0):
    """The letters of `rows` ([[(word number, string)]]) set row under row, `pitch` caps apart, each row set against
    `align` and shifted `stagger` caps (alternating sides): [Glyph] (their `word` the word's number), the word spans
    [(x0, x1, row)] and the rows' advances."""
    lines = []
    for row in rows:
        gl, adv, sp = set_words([t for _, t in row], path, cap, tracking, gap)
        for g in gl:
            g.word = row[g.word][0]
        lines.append((gl, adv, sp))
    W = max(adv for _, adv, _ in lines)
    out, spans = [], []
    for r, (gl, adv, sp) in enumerate(lines):
        dx = {"left": 0.0, "center": 0.5 * (W - adv), "right": W - adv}[align]
        if len(lines) > 1:
            dx += stagger * cap * (1 if r % 2 else -1)
        dy = r * pitch * cap
        for g in gl:
            g.x += dx
            g.y += dy
        out += gl
        spans += [(a + dx, b + dx, r) for a, b in sp]
    return out, spans, [adv for _, adv, _ in lines]


def _line_mask(glyphs, w, h, ox, oy):
    m = np.zeros((int(math.ceil(h)), int(math.ceil(w))), np.float32)
    for g in glyphs:
        paste(m, g.mask, g.x + ox, g.y + oy)
    return m


def _stack_strips(parts, cap, gap=0.22, stagger=0.35):
    """Strips made one per row (each {body, ink, order, box (w, h), margin} with the box at `margin` in its canvas)
    stacked into one unit, `gap` caps apart, shifted `stagger` caps to alternate sides; the ink's order runs row after
    row. A Sprite's (body, ink, order, box)."""
    if len(parts) == 1:
        q = parts[0]
        return q["body"], q["ink"], q["order"], q["box"], q.get("marks")
    n = len(parts)
    m = max(q["margin"] for q in parts)
    shifts = [stagger * cap * (1 if r % 2 else -1) for r in range(n)]
    W = max(q["box"][0] + 2 * abs(s) for q, s in zip(parts, shifts))
    H = sum(q["box"][1] for q in parts) + gap * cap * (n - 1)
    hh, ww = int(math.ceil(H + 2 * m)), int(math.ceil(W + 2 * m))
    body = np.zeros((hh, ww, 4), np.float32)
    ink = np.zeros((hh, ww, 4), np.float32) if parts[0]["ink"] is not None else None
    order = np.full((hh, ww), 2.0, np.float32) if ink is not None else None
    y = m
    for r, (q, s) in enumerate(zip(parts, shifts)):
        x = m + 0.5 * (W - q["box"][0]) + s
        X, Y = int(round(x - q["margin"])), int(round(y - q["margin"]))
        h, w = q["body"].shape[:2]
        ax, ay, bx, by = max(X, 0), max(Y, 0), min(X + w, ww), min(Y + h, hh)
        src = (slice(ay - Y, by - Y), slice(ax - X, bx - X))
        body[ay:by, ax:bx] = _stack(body[ay:by, ax:bx], q["body"][src])
        if ink is not None:
            ink[ay:by, ax:bx] = _stack(ink[ay:by, ax:bx], q["ink"][src])
            here = q["ink"][src][..., 3] > 0.0
            order[ay:by, ax:bx] = np.where(here, (r + q["order"][src]) / n, order[ay:by, ax:bx])
        y += q["box"][1] + gap * cap
    marks = None if ink is None else [(r + m) / n for r, q in enumerate(parts) for m in q["marks"]]
    return body, ink, order, (W, H), marks


# ------------------------------------------------------------------------------------------------- emboss
def _emboss_strip(c, row, cap, font, seed):
    """One strip of label tape with the row's capitals embossed in it: a part for _stack_strips."""
    glyphs, adv, _ = set_words([t for _, t in row], font, cap, tracking=c["tracking"], gap=0.6)
    x0, y0, x1, y1 = ink_box(glyphs)
    tape_h, tooth, pad = 1.8 * cap, 0.15 * cap, 0.6 * cap
    W, H = (x1 - x0) + 2 * pad + 2 * tooth, tape_h
    margin = 0.25 * cap
    hh, ww = int(math.ceil(H + 2 * margin)), int(math.ceil(W + 2 * margin))
    letters = _line_mask(glyphs, ww, hh, margin + tooth + pad - x0, margin + 0.5 * tape_h + 0.5 * cap)
    L, R, T = margin, margin + W, margin
    n = max(int(round(H / (0.2 * cap))), 4)
    left = [(L + (tooth if k % 2 == 0 else 0.0), T + H * k / n) for k in range(n + 1)]       # pinked ends
    right = [(R - (tooth if k % 2 == 0 else 0.0), T + H * k / n) for k in range(n + 1)]
    strip = _poly_mask((hh, ww), right + left[::-1])
    yy = (np.arange(hh, dtype=np.float32)[:, None] - T) / H
    base = np.asarray(c["color"], np.float32)
    # glossy plastic: nearly flat colour, a sheen along the top, the slit edges a shade deeper, a faint grain
    k = 0.03 - 0.07 * np.clip(yy, 0.0, 1.0) + 0.015 * _noise((hh, ww), 1.2 * cap, seed, 2)
    edge = np.clip(1.0 - np.minimum(np.abs(yy), np.abs(yy - 1.0)) * H / (0.06 * cap), 0, 1)
    col = _shade(np.broadcast_to(base, (hh, ww, 3)), np.broadcast_to(k - 0.12 * edge, (hh, ww)))
    sheen = 0.32 * np.exp(-((yy - 0.16) / 0.05) ** 2) + 0.10 * np.exp(-((yy - 0.3) / 0.12) ** 2)
    sheen = sheen * (0.8 + 0.2 * _noise((hh, ww), 3.0 * cap, seed + 2))
    col = col + (1.0 - col) * np.clip(sheen, 0, 1)[..., None]
    # the letters pushed up from behind: the plastic stretches pale round them (the stress halo) and white in them
    halo = np.clip(_blur(_grow(letters, 0.06 * cap), 0.07 * cap) * 1.4, 0, 1) * c["halo"]
    col = col + (_mix(base, WHITE, 0.5) - col) * halo[..., None]
    face = _mix(c["ink"], base, 0.10) + 0.025 * _noise((hh, ww), 0.4 * cap, seed + 4)[..., None]
    col = col + (face - col) * letters[..., None]
    sig = 0.022 * cap
    gy, gx = np.gradient(_blur(letters, sig))
    light = np.clip(-(gx * -0.55 + gy * -0.83) * sig * 4.0, -1.0, 1.0)              # lit from the top left
    near = np.clip(_grow(letters, 0.04 * cap), 0, 1)
    col = col + (1.0 - col) * (np.clip(light, 0, 1) * near * 0.55)[..., None]
    col = col + (_shade(base, -0.35) - col) * (np.clip(-light, 0, 1) * near * 0.45)[..., None]
    body = np.zeros((hh, ww, 4), np.float32)
    _over(body, np.clip(col, 0, 1), strip)
    return {"body": body, "ink": None, "order": None, "box": (W, H), "margin": margin}


def _emboss(c, rows, cap, font, seed, px):
    parts = [_emboss_strip(c, row, cap, font, seed + 17 * r) for r, row in enumerate(rows)]
    body, _, _, box, _ = _stack_strips(parts, cap)
    return Sprite(_down(body), back=_mix(c["color"], WHITE, 0.35), box=(box[0] / SS, box[1] / SS))


# ------------------------------------------------------------------------------------------------- masking tape
def _torn_edge(n, depth, seed):
    """A torn end's profile: n samples (0..depth px inward), rough at two scales."""
    rng = np.random.default_rng(seed)
    a = cv2.resize(rng.uniform(0, 1, (max(n // 18, 3), 1)).astype(np.float32), (1, n),
                   interpolation=cv2.INTER_CUBIC)[:, 0]
    b = rng.uniform(0, 1, n).astype(np.float32)
    b = np.convolve(b, np.ones(3) / 3, mode="same")
    return depth * np.clip(0.65 * a + 0.35 * b, 0, 1)


def _tape_strip(shape, L, R, T, B, depth, seed, slant=0.0):
    """Coverage of a strip of tape from x L to R and y T to B with both ends torn (fibres reaching past them)."""
    hh, ww = shape
    rows = np.arange(hh)
    inside = (rows >= T) & (rows < B)
    n = int(B - T) + 1
    lprof = _torn_edge(n, depth, seed)
    rprof = _torn_edge(n, depth, seed + 7)
    xs = np.arange(ww, dtype=np.float32)[None, :]
    k = np.clip(rows - int(T), 0, n - 1)
    sl = slant * (rows - (T + B) / 2.0)
    li = L + lprof[k] + sl
    ri = R - rprof[k] - sl
    m = np.clip(xs - li[:, None] + 0.5, 0, 1) * np.clip(ri[:, None] - xs + 0.5, 0, 1)
    yv = np.clip(rows[:, None] - T + 0.5, 0, 1) * np.clip(B - rows[:, None] + 0.5, 0, 1)
    m = m * yv * inside[:, None]
    rng = np.random.default_rng(seed + 11)                    # fibres: thin hairs past each torn end
    fib = np.zeros((hh, ww), np.uint8)
    for side, prof in ((-1, lprof), (1, rprof)):
        for _ in range(max(int(n / 5), 6)):
            r = int(rng.integers(0, n))
            y = T + r
            x = (L + prof[r] + slant * (y - (T + B) / 2.0)) if side < 0 else (R - prof[r] - slant * (y - (T + B) / 2.0))
            ln = rng.uniform(0.3, 1.0) * depth * 1.1
            ang = rng.uniform(-0.6, 0.6)
            x2, y2 = x + side * ln * math.cos(ang), y + ln * math.sin(ang)
            cv2.line(fib, (int(x * 4), int(y * 4)), (int(x2 * 4), int(y2 * 4)), 255,
                     max(int(rng.uniform(0.6, 1.4) * 4), 1), cv2.LINE_AA, 2)
    fib = fib.astype(np.float32) / 255.0 * 0.55
    return np.maximum(m, fib * (np.abs(rows[:, None] - (T + B) / 2) < (B - T) / 2))


def _masking_strip(c, row, cap, font, seed):
    """One torn strip of masking tape with the row written on it: a part for _stack_strips."""
    glyphs, adv, spans = set_words([t for _, t in row], font, cap, tracking=c["tracking"], gap=0.38)
    x0, y0, x1, y1 = ink_box(glyphs)
    pad, depth = 0.7 * cap, 0.32 * cap
    top, bot = min(y0, -1.05 * cap) - 0.3 * cap, max(y1, 0.3 * cap) + 0.28 * cap   # room for ascenders, descenders
    W, H = (x1 - x0) + 2 * pad + 2 * depth, bot - top
    margin = 0.5 * cap
    hh, ww = int(math.ceil(H + 2 * margin)), int(math.ceil(W + 2 * margin))
    ox, oy = margin + depth + pad - x0, margin - top
    L, R, T, B = margin, margin + W, margin, margin + H
    strip = _tape_strip((hh, ww), L, R, T, B, depth, seed, slant=0.05)
    base = np.asarray(c["color"], np.float32)
    crepe = 0.5 * _streaks((hh, ww), 0.2 * cap, 0.035 * cap, 82.0, seed + 3) + \
        0.5 * _streaks((hh, ww), 0.15 * cap, 0.03 * cap, 97.0, seed + 4)
    yy = (np.arange(hh, dtype=np.float32)[:, None] - T) / H
    k = 0.04 * crepe + 0.035 * _noise((hh, ww), 0.8 * cap, seed + 5, 2) - 0.05 * (np.abs(yy - 0.5) > 0.46)
    col = _shade(np.broadcast_to(base, (hh, ww, 3)), np.broadcast_to(k, (hh, ww)))
    alpha = c["alpha"] * (0.94 + 0.06 * _noise((hh, ww), 0.4 * cap, seed + 6))
    body = np.zeros((hh, ww, 4), np.float32)
    _over(body, np.clip(col, 0, 1), strip * alpha)
    letters = _line_mask(glyphs, ww, hh, ox, oy)
    wet = 0.92 + 0.08 * _noise((hh, ww), 0.25 * cap, seed + 8)              # the pen lays it a little unevenly
    ink = np.zeros((hh, ww, 4), np.float32)
    _over(ink, np.asarray(c["ink"], np.float32), np.clip(_blur(letters, 0.012 * cap) * wet, 0, 1))
    xs = np.arange(ww, dtype=np.float32)[None, :]                             # the pen's way: along the row, each
    order = (xs - (ox + x0)) / max(x1 - x0, 1.0) + 0.025 * np.clip(yy, 0, 1)  # letter top to bottom a little
    marks = [(b - x0) / max(x1 - x0, 1.0) + 0.025 + MARK for _, b in spans]
    return {"body": body, "ink": ink, "order": np.ascontiguousarray(order, np.float32), "box": (W, H),
            "margin": margin, "marks": marks}


def _masking(c, rows, cap, font, seed, px):
    parts = [_masking_strip(c, row, cap, font, seed + 17 * r) for r, row in enumerate(rows)]
    body, ink, order, box, marks = _stack_strips(parts, cap, gap=0.12)
    return Sprite(_down(body), ink=_down(ink), order=_down(order), back=_mix(c["color"], WHITE, 0.25),
                  box=(box[0] / SS, box[1] / SS), marks=marks)


# ------------------------------------------------------------------------------------------------- letter sticker
def _sticker(c, rows, cap, font, seed, px):
    glyphs, _, _ = set_rows(rows, font, cap, c["tracking"], 0.3, 1.18, stagger=0.25)
    rim = c["rim"]["width"] * px if c["rim"] else 0.0
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = ink_box(glyphs)
    margin = rim + 0.3 * cap
    hh, ww = int(math.ceil((y1 - y0) + 2 * margin + 0.2 * cap)), int(math.ceil((x1 - x0) + 2 * margin))
    ox, oy = margin - x0, margin - y0 + 0.1 * cap
    cols = c["colors"]
    start = int(rng.integers(0, len(cols)))
    letters = np.zeros((hh, ww), np.float32)
    col = np.zeros((hh, ww, 3), np.float32)
    for g in glyphs:                                   # each word its own colour, each letter a little bounced
        m, dx, dy = turned(g.mask, float(rng.uniform(-3.0, 3.0)))
        one = np.zeros((hh, ww), np.float32)
        paste(one, m, g.x + ox + dx, g.y + oy + dy + float(rng.uniform(-0.035, 0.035)) * cap)
        rgb = np.asarray(cols[(start + g.word) % len(cols)], np.float32)
        col = col * (1.0 - one[..., None]) + rgb * one[..., None]
        letters = np.maximum(letters, one)
    col = np.where((letters > 0)[..., None], col / np.maximum(letters, 1e-6)[..., None], col)
    # vinyl: a soft light along each letter's upper inside, a deeper lower edge
    sig = 0.05 * cap
    gy, gx = np.gradient(_blur(letters, sig))
    light = np.clip(-(gx * -0.45 + gy * -0.9) * sig * 3.0, -1.0, 1.0) * (letters > 0.5)
    col = col + (1.0 - col) * (0.22 * np.clip(light, 0, 1))[..., None]
    col = col * (1.0 - 0.22 * np.clip(-light, 0, 1))[..., None]
    body = np.zeros((hh, ww, 4), np.float32)
    if c["rim"]:
        cut = _fill_holes(_grow(letters, rim))
        r2 = max(int(round(0.9 * rim)), 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))
        cut = np.maximum(cut, cv2.morphologyEx((cut >= 0.5).astype(np.uint8), cv2.MORPH_CLOSE, kern).astype(np.float32))
        cut = _fill_holes(_blur(cut, 0.6 * SS))
        inner = _blur(cut, 0.035 * cap)                 # the cut edge turns away from the light a little
        rc = np.asarray(c["rim"]["color"], np.float32) * (1.0 - 0.07 * np.clip(1.0 - inner, 0, 1) ** 0.5)[..., None]
        _over(body, rc, cut)
    _over(body, np.clip(col, 0, 1), letters)
    ys, xs = np.nonzero(body[..., 3] > 0.02)
    box = (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) if len(xs) else (ww, hh)
    return Sprite(_down(body), back=(1.0, 0.995, 0.985), box=(box[0] / SS, box[1] / SS))


# ------------------------------------------------------------------------------------------------- cut paper
def _scissor_cut(mask, tol):
    """A glyph's coverage as scissors would cut it: its outline (and its holes) as straight cuts at most `tol` px off."""
    m8 = (mask >= 0.5).astype(np.uint8)
    cs, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    k = 4
    cut = np.zeros((mask.shape[0] * k, mask.shape[1] * k), np.uint8)
    for ci, cnt in enumerate(cs):
        ap = cv2.approxPolyDP(cnt.astype(np.float32), tol, True)
        if len(ap) < 3:
            continue
        hole = hier is not None and hier[0][ci][3] >= 0
        cv2.fillPoly(cut, [np.round((ap + 0.5) * k * 4).astype(np.int32)], 0 if hole else 255, cv2.LINE_AA, 2)
    return cv2.resize(cut, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _cutpaper(c, rows, cap, font, seed, px):
    glyphs, _, _ = set_rows(rows, font, cap, c["tracking"], 0.45, 1.35)
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = ink_box(glyphs)
    pad_x, pad_y = 0.48 * cap, 0.4 * cap
    W, H = (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y
    margin = 0.6 * cap
    hh, ww = int(math.ceil(H + 2 * margin)), int(math.ceil(W + 2 * margin))
    L, T, R, B = margin, margin, margin + W, margin + H
    j = 0.05 * cap                                     # the label cut by hand: no edge quite square
    corners = [(L + rng.uniform(-j, j), T + rng.uniform(-j, j)), (R + rng.uniform(-j, j), T + rng.uniform(-j, j)),
               (R + rng.uniform(-j, j), B + rng.uniform(-j, j)), (L + rng.uniform(-j, j), B + rng.uniform(-j, j))]
    dog = seed % 3 == 0                                # one label in three: a dog-eared bottom right corner
    paper = np.asarray(c["color"], np.float32)
    outline = list(corners)
    if dog:
        e = 0.6 * cap
        C = np.asarray(corners[2])
        A = C + (np.asarray(corners[3]) - C) / np.linalg.norm(np.asarray(corners[3]) - C) * e
        Bp = C + (np.asarray(corners[1]) - C) / np.linalg.norm(np.asarray(corners[1]) - C) * e
        outline = [corners[0], corners[1], tuple(Bp), tuple(A), corners[3]]
    label = _poly_mask((hh, ww), outline)
    tex = 0.018 * _noise((hh, ww), 0.5 * cap, seed + 1, 3) + 0.012 * _noise((hh, ww), 0.03 * cap, seed + 2) + \
        0.008 * _streaks((hh, ww), 0.12 * cap, 0.006 * cap, float(rng.uniform(0, 180)), seed + 3)   # paper fibres
    body = np.zeros((hh, ww, 4), np.float32)
    _over(body, _shade(np.broadcast_to(paper, (hh, ww, 3)), np.broadcast_to(tex - 0.01, (hh, ww))), label)
    ox, oy = margin + pad_x - x0, margin + pad_y - y0
    cols = c["colors"]
    shadow = np.asarray(c["ink"], np.float32)
    last = -1
    for n, g in enumerate(glyphs):                     # each letter cut from a paper of its own, a little off
        pick = int(rng.integers(0, len(cols)))
        if pick == last:
            pick = (pick + 1) % len(cols)
        last = pick
        m, dx, dy = turned(_scissor_cut(g.mask, 0.05 * cap), float(rng.uniform(-c["jitter"], c["jitter"])))
        one = np.zeros((hh, ww), np.float32)
        paste(one, m, g.x + ox + dx, g.y + oy + dy + float(rng.uniform(-0.05, 0.05)) * cap)
        sh = _blur(cv2.warpAffine(one, np.float32([[1, 0, 0.025 * cap], [0, 1, 0.04 * cap]]), (ww, hh)), 0.02 * cap)
        _over(body, shadow, 0.3 * sh * (1.0 - one))
        rgb = np.asarray(cols[pick], np.float32)
        t = 0.035 * _noise((hh, ww), 0.3 * cap, seed + 10 + n) + 0.015 * _noise((hh, ww), 0.04 * cap, seed + 40 + n)
        _over(body, _shade(np.broadcast_to(rgb, (hh, ww, 3)), np.broadcast_to(t, (hh, ww))), one)
    if dog:                                            # the folded corner: its back, paler, shaded toward the fold
        d = Bp - A
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        Cf = C - 2.0 * float(nrm @ (C - A)) * nrm
        sh = _blur(_poly_mask((hh, ww), [tuple(A), tuple(Bp), tuple(Cf + 0.1 * cap)]), 0.06 * cap)
        _over(body, np.asarray(c["ink"], np.float32), 0.4 * sh * label)
        flap = _poly_mask((hh, ww), [tuple(A), tuple(Bp), tuple(Cf)])
        X, Y = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
        far = np.clip(((X - A[0]) * nrm[0] + (Y - A[1]) * nrm[1]) / max(float(np.linalg.norm(Cf - A) * 0.7), 1), -1, 1)
        _over(body, _shade(np.broadcast_to(paper, (hh, ww, 3)),                # its back, deeper toward the fold
                           np.broadcast_to(-0.16 * (1.0 - np.abs(far)) - 0.03, (hh, ww))), flap)
    tape = np.asarray(c["tape"], np.float32)           # bits of masking tape over its corners, no two alike
    for i, ((cx, cy), (sx, sy)) in enumerate(zip(corners, ((-1, -1), (1, -1), (1, 1), (-1, 1)))):
        if dog and i == 2:
            continue
        tl, tw = rng.uniform(0.6, 1.25) * cap, rng.uniform(0.28, 0.42) * cap
        ang = math.radians(-45.0 * sx * sy + rng.uniform(-28, 28))
        ux, uy = math.cos(ang), math.sin(ang)
        mx, my = cx - sx * rng.uniform(0.02, 0.16) * cap, cy - sy * rng.uniform(0.02, 0.16) * cap
        pts = [(mx - ux * tl / 2 - uy * tw / 2, my - uy * tl / 2 + ux * tw / 2),
               (mx + ux * tl / 2 - uy * tw / 2, my + uy * tl / 2 + ux * tw / 2),
               (mx + ux * tl / 2 + uy * tw / 2, my + uy * tl / 2 - ux * tw / 2),
               (mx - ux * tl / 2 + uy * tw / 2, my - uy * tl / 2 - ux * tw / 2)]
        tm = _poly_mask((hh, ww), pts)
        tcol = _shade(np.broadcast_to(tape, (hh, ww, 3)),
                      np.broadcast_to(0.03 * _streaks((hh, ww), 0.15 * cap, 0.03 * cap, 90 + math.degrees(ang),
                                                      seed + 31 + i), (hh, ww)))
        _over(body, tcol, 0.78 * tm)
    return Sprite(_down(body), back=_mix(paper, WHITE, 0.3), box=(W / SS, H / SS))


# ------------------------------------------------------------------------------------------------- brush
def _along(noise, tx, ty, length, steps=12):
    """`noise` smeared along the field (tx, ty) (unit vectors) over `length` px each way: streaks that follow it (a
    line integral convolution)."""
    h, w = noise.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc, n = noise.copy(), 1
    for sgn in (1.0, -1.0):
        px, py = X.copy(), Y.copy()
        for _ in range(steps):
            dx = cv2.remap(tx, px, py, cv2.INTER_LINEAR)
            dy = cv2.remap(ty, px, py, cv2.INTER_LINEAR)
            px += np.float32(sgn * length / steps) * dx
            py += np.float32(sgn * length / steps) * dy
            acc += cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            n += 1
    return acc / np.float32(n)


def _brush(c, rows, cap, font, seed, px):
    glyphs, spans, _ = set_rows(rows, font, cap, c["tracking"], 0.3, 1.25, align="left")
    x0, y0, x1, y1 = ink_box(glyphs)
    rim = c["rim"]["width"] * px if c["rim"] else 0.0
    margin = 0.3 * cap + rim
    hh, ww = int(math.ceil((y1 - y0) + 2 * margin)), int(math.ceil((x1 - x0) + 2 * margin))
    ox, oy = margin - x0, margin - y0
    letters = _grow(_line_mask(glyphs, ww, hh, ox, oy), 0.035 * cap)         # a loaded brush: a fuller stroke
    letters = _blur(letters, 0.008 * cap)
    # the strokes' direction: the structure tensor over a stroke's width (along the stroke even at its round ends,
    # where the distance to the edge would turn round it)
    gy, gx = np.gradient(_blur(letters, 0.02 * cap))
    sig = 0.12 * cap
    jxx, jxy, jyy = _blur(gx * gx, sig), _blur(gx * gy, sig), _blur(gy * gy, sig)
    th = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)                               # the edges' normal
    tx, ty = (-np.sin(th)).astype(np.float32), np.cos(th).astype(np.float32)
    rng = np.random.default_rng(seed + 2)
    fine = _blur(rng.uniform(-1, 1, (hh, ww)).astype(np.float32), 0.012 * cap)   # a bristle's width
    hair = _along(fine, tx, ty, 0.45 * cap, steps=max(int(0.45 * cap / 1.5), 8))  # streaks along the strokes
    hair = hair / np.float32(hair.std() + 1e-6) * 0.5
    xs = np.arange(ww, dtype=np.float32)[None, :]
    run = np.zeros((hh, ww), np.float32)
    for a, b, r in spans:
        a, b = a + ox, b + ox
        u = np.clip((xs - a) / max(b - a, 1.0), 0, 1) * ((xs >= a - 0.2 * cap) & (xs <= b + 0.2 * cap))
        run = np.maximum(run, np.broadcast_to(u, (hh, ww)))
    dry = c["dry"] * (0.05 + 0.95 * run ** 2.5) * (0.75 + 0.5 * _noise((hh, ww), 0.7 * cap, seed + 4))
    gaps = np.clip((hair - (0.9 - 1.7 * dry)) * 3.0, 0, 1)                    # open where the paint runs out
    paint = letters * (1.0 - 0.95 * gaps)
    base = np.asarray(c["color"], np.float32)
    pool = np.clip(_blur(letters, 0.03 * cap) - _blur(letters, 0.1 * cap), 0, 1)    # paint gathers at the edges
    start = 1.0 - np.clip(run * 5.0, 0, 1)                                    # and where each stroke of it starts
    k = 0.07 * hair * (0.3 + dry) - 0.08 * pool - 0.18 * start ** 2
    ink = np.zeros((hh, ww, 4), np.float32)
    if c["rim"]:                                              # a paint pen's pale edge, laid with the paint: the
        edge = _blur(_fill_holes(_grow(letters, rim)), 0.6 * SS)   # letters read over any picture, their counters closed
        _over(ink, np.broadcast_to(np.asarray(c["rim"]["color"], np.float32), (hh, ww, 3)), edge)
    _over(ink, np.clip(_shade(np.broadcast_to(base, (hh, ww, 3)), np.broadcast_to(k, (hh, ww))), 0, 1), paint)
    body = np.zeros((hh, ww, 4), np.float32)
    nrow = max(r for _, _, r in spans) + 1                    # the brush's way: along each row, row after row
    row_of = np.clip(((np.arange(hh, dtype=np.float32)[:, None] - oy) / max(1.25 * cap, 1.0) + 0.75).astype(np.int32),
                     0, nrow - 1)
    order = (row_of + np.clip((xs - (ox + x0)) / max(x1 - x0, 1.0), 0, 1)) / nrow
    marks = [(r + min(max((b - x0) / max(x1 - x0, 1.0), 0.0), 1.0)) / nrow + MARK for _, b, r in spans]
    return Sprite(_down(body), ink=_down(ink), order=_down(np.ascontiguousarray(order, np.float32)),
                  back=_shade(base, -0.12), box=((x1 - x0) / SS, (y1 - y0) / SS), marks=marks)  # paint, from behind


KINDS = {"emboss": _emboss, "masking": _masking, "sticker": _sticker, "cutpaper": _cutpaper, "brush": _brush}


# ================================================================================================= peeling
CORNER_AT = {"top_left": (0.0, 0.0), "top_right": (1.0, 0.0), "bottom_left": (0.0, 1.0), "bottom_right": (1.0, 1.0)}


def peeled(sprite_rgba, fold, corner, back, roll=0.18):
    """A sprite (premultiplied RGBA) with its `corner` peeled back `fold` of the way along its diagonal: the lifted part
    rolls over a cylinder (its radius `roll` of the sprite's shorter side) whose axis lies on the line it has come to,
    its face turning away into the roll and its back (`back`, shaded along the roll) coming over the top and lying
    back over the rest, which the roll shades softly."""
    h, w = sprite_rgba.shape[:2]
    a = sprite_rgba[..., 3]
    ys, xs = np.nonzero(a > 0.02)
    if len(xs) == 0 or fold <= 0:
        return sprite_rgba
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    fx, fy = CORNER_AT[corner]
    C = np.array([bx0 + fx * (bx1 - bx0), by0 + fy * (by1 - by0)], np.float64)
    O = np.array([bx0 + (1 - fx) * (bx1 - bx0), by0 + (1 - fy) * (by1 - by0)], np.float64)
    Ld = float(np.linalg.norm(O - C))
    n = (C - O) / max(Ld, 1e-9)                                # toward the corner
    r = max(roll * min(bx1 - bx0, by1 - by0), 3.0) * min(1.0, 0.4 + fold * 3.0)
    d0 = float(n @ (C - n * fold * Ld))                        # the roll's axis: n.x = d0
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    t = (n[0] * X + n[1] * Y - d0).astype(np.float32)          # each pixel's place past the axis

    def from_sheet(s):                                         # the sprite at the flat sheet's place s past the axis
        mx, my = X + (s - t) * np.float32(n[0]), Y + (s - t) * np.float32(n[1])
        return cv2.remap(sprite_rgba, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = sprite_rgba * np.clip(0.5 - t, 0, 1)[..., None]     # the part still down
    on = (t >= 0) & (t < r)
    tt = np.clip(t, 0.0, r * 0.9999) / r
    phi1 = np.arcsin(tt)                                       # the face turning into the roll
    face = from_sheet(r * phi1) * on[..., None]
    face[..., :3] *= (1.0 - 0.55 * (phi1 / (np.pi / 2)) ** 2)[..., None]
    phi2 = np.pi - phi1                                        # the back coming over the top of the roll
    top = from_sheet(r * phi2)[..., 3] * on
    flat = from_sheet(np.pi * r - t)[..., 3] * (t < 0)          # and lying back over the rest
    lifted = np.clip(face[..., 3] + top + flat, 0, 1)
    sh = _blur(cv2.warpAffine(lifted, np.float32([[1, 0, 0.25 * r], [0, 1, 0.45 * r]]), (w, h)), 0.6 * r)
    out[..., :3] *= (1.0 - 0.45 * sh)[..., None]               # the soft shadow of the roll on what is beneath
    out = _stack(out, face)
    rolled = np.where(t >= 0, 0.7 + 0.3 * np.sin(phi2),      # lit at the top of the roll, deeper round it
                      0.9 + 0.1 * np.clip(-t / (2.0 * r), 0, 1))
    col = np.clip(np.asarray(back, np.float32) * rolled[..., None].astype(np.float32), 0, 1)
    _over(out, col, np.clip(top + flat, 0, 1))
    return out


def wiped(sprite_rgba, share):
    """Paint smeared off sideways by a thumb, `share` of the way across: behind the thumb a thin streak of it, at the
    thumb the paint dragged along, ahead of it the letters as they were."""
    h, w = sprite_rgba.shape[:2]
    ys, xs = np.nonzero(sprite_rgba[..., 3] > 0.02)
    if len(xs) == 0 or share <= 0:
        return sprite_rgba
    x0, x1 = xs.min(), xs.max() + 1
    L = max(int(0.3 * (x1 - x0)), 3)
    front = x0 + share * (x1 - x0 + L)
    X = np.arange(w, dtype=np.float32)[None, :]
    ahead = np.clip((X - front) / 3.0 + 0.5, 0, 1)
    smear = cv2.warpAffine(cv2.blur(sprite_rgba, (L, 3)), np.float32([[1, 0, 0.5 * L], [0, 1, 0]]), (w, h))
    band = np.clip(1.0 - (front - X) / L, 0, 1) * (1.0 - ahead)
    rows = np.random.default_rng(int(x1)).uniform(0.0, 0.25, (h, 1)).astype(np.float32)   # what the thumb leaves
    return sprite_rgba * ahead[..., None] + smear * (0.9 * band + rows * (1.0 - band) * (1.0 - ahead))[..., None]


# ================================================================================================= onto the frame
def place(rgba, M, shadow, lift, size, px):
    """A sprite carried onto the frame by the affine map `M` (sprite pixels to frame pixels) with its drop shadow:
    (premultiplied RGBA patch, x0, y0) of the box it covers, or None off the frame. `shadow` the craft's, lifted
    `lift` (0..1) off the paper (farther out and softer); `px` the output's pixels per px at 1080."""
    H, W = size[1], size[0]
    h, w = rgba.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ np.asarray(M, np.float64).T
    reach = 2.0
    if shadow is not None:
        off = np.asarray(shadow["offset"], np.float64) * H * (1.0 + 2.5 * lift)
        blur = shadow["blur"] * px * (1.0 + 1.5 * lift)
        reach += abs(off).max() + 3.0 * blur
    x0 = max(int(math.floor(corners[:, 0].min() - reach)), 0)
    y0 = max(int(math.floor(corners[:, 1].min() - reach)), 0)
    x1 = min(int(math.ceil(corners[:, 0].max() + reach)), W)
    y1 = min(int(math.ceil(corners[:, 1].max() + reach)), H)
    if x1 <= x0 or y1 <= y0:
        return None
    Mp = np.asarray(M, np.float64).copy()
    Mp[0, 2] -= x0
    Mp[1, 2] -= y0
    pw, ph = x1 - x0, y1 - y0
    scale = math.sqrt(abs(Mp[0, 0] * Mp[1, 1] - Mp[0, 1] * Mp[1, 0]))
    flags = cv2.INTER_AREA if scale < 0.7 else cv2.INTER_LINEAR
    if scale < 0.7:                                            # shrinking: box the sprite down first
        k = max(scale, 0.05)
        small = cv2.resize(rgba, (max(int(w * k), 1), max(int(h * k), 1)), interpolation=cv2.INTER_AREA)
        Mp = Mp @ np.array([[1 / k, 0, 0], [0, 1 / k, 0], [0, 0, 1]])
        rgba, flags = small, cv2.INTER_LINEAR
    out = cv2.warpAffine(rgba, Mp.astype(np.float32), (pw, ph), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
    if shadow is not None and shadow["alpha"] > 0:
        a = out[..., 3]
        sh = cv2.warpAffine(a, np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (pw, ph), flags=cv2.INTER_LINEAR)
        sh = _blur(sh, blur) * np.float32(shadow["alpha"] * (1.0 - 0.35 * lift))
        out = out + np.dstack([np.broadcast_to(np.asarray(shadow["color"], np.float32), (ph, pw, 3)) * sh[..., None],
                               sh[..., None]]) * (1.0 - a)[..., None]
    return out, x0, y0


def lay(img, patches):
    """The picture `img` (float RGB) with the patches (premultiplied RGBA, x0, y0) laid over it in order (a new array)."""
    out = np.array(img, np.float32, copy=True)
    for patch, x0, y0 in patches:
        ph, pw = patch.shape[:2]
        d = out[y0:y0 + ph, x0:x0 + pw]
        d *= 1.0 - patch[..., 3:4]
        d += patch[..., :3]
    return out


def as_layer(layer, patches, size):
    """A straight RGBA layer (h, w, 4) with the patches laid over `layer` (straight RGBA or None): the screen type and
    the craft type above it as one layer, for mk post's screen step and mkr-post."""
    w, h = size
    prem = np.zeros((h, w, 4), np.float32)
    if layer is not None:
        L = np.asarray(layer, np.float32)
        prem[..., :3] = L[..., :3] * L[..., 3:4]
        prem[..., 3] = L[..., 3]
    for patch, x0, y0 in patches:
        ph, pw = patch.shape[:2]
        d = prem[y0:y0 + ph, x0:x0 + pw]
        d *= 1.0 - patch[..., 3:4]
        d += patch
    a = prem[..., 3:4]
    prem[..., :3] = np.where(a > 1e-6, prem[..., :3] / np.maximum(a, 1e-6), 0.0)
    return prem


def write_layer(path, layer):
    """A straight RGBA layer written as an 8-bit PNG, as mk render writes a screen layer."""
    u8 = (np.clip(layer, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGBA2BGRA)):
        raise OSError(f"{path}: cannot write")


# ================================================================================================= a project's crafts
class Crafts:
    """A project's craft type as one output shows it: `groups(frame)` lays the units on screen on a frame, each where
    its entry's `under` puts it in the chain of the frame effects: a slot -1 (before every `[[fx]]`), the index of the
    `[[fx]]` it goes just before (`under = "<name>"`, `fx_names` maps the names), or None (over them). Plans are made
    and units measured up front; sprites are drawn when first needed (each worker process has its own)."""

    def __init__(self, entries, size, fps, frame0, palette, font_of, timeline_of, fx_names=None):
        self.size, self.fps, self.frame0 = (int(size[0]), int(size[1])), float(fps), int(frame0)
        self.px = min(self.size) / 1080.0
        self.plans = []
        for name, spec in entries:
            lyr = spec.get("lyrics") or {}
            if not lyr.get("timeline"):
                raise TextError(f"text {name!r}: lyrics.timeline is required")
            p = CT.plan(spec, timeline_of(lyr["timeline"]), palette, fps, self.size)
            if not p["font"]:
                raise TextError(f"text {name!r}: craft type needs a `font` (a font slug of the asset registry or a file)")
            under = p["under"]
            if isinstance(under, str):
                if under not in (fx_names or {}):
                    raise TextError(f"text {name!r}: under = {under!r}: no [[fx]] has that name (named: "
                                    f"{sorted(fx_names or {})})")
                p["slot"] = fx_names[under]
            else:
                p["slot"] = -1 if under else None
            p["font_path"] = font_of(p["font"])
            guess = (p["size"] or 0.06) * self.size[1]
            CT.layout(p, [measure(p, u, p["font_path"], guess, self.px) for u in p["units"]])
            self.plans.append(p)
        self._sprites = {}

    @classmethod
    def of(cls, proj, output, size, font_of, fx_names=None):
        """The project's craft type for `output` (size (w, h)), or None when it has none. TextError says what is wrong."""
        import json
        outs = [o.get("name") for o in proj.data.get("output", []) or []] or [output]
        found = CT.entries(proj.data.get("text"), outs)
        if not found:
            return None
        look = proj.data.get("look", {})
        pal = PAL.get(look.get("palette", "rose-pine-moon"), look.get("slots"))
        cache = {}

        def timeline_of(rel):
            if rel not in cache:
                path = (proj.root / rel) if not str(rel).startswith("~") else __import__("pathlib").Path(rel).expanduser()
                cache[rel] = json.loads(path.read_text())
            return cache[rel]
        return cls([(n, by[output]) for n, by in found], size, proj.fps, proj.frame0, pal, font_of, timeline_of,
                   fx_names)

    def active(self, frame):
        f = int(frame) - self.frame0
        return any(CT.on_screen(p, f) for p in self.plans)

    def _sprite(self, p, u):
        key = (p["name"], u["k"])
        if key not in self._sprites:
            self._sprites[key] = build(p, u, p["cap"] * u["scale"] * self.size[1], p["font_path"], self.px)
        return self._sprites[key]

    def groups(self, frame):
        """{slot: patches} of the units laid on Blender frame `frame`, by where they go (see the class)."""
        out = {}
        for slot in {p["slot"] for p in self.plans}:
            got = self.patches(frame, slot)
            if got:
                out[slot] = got
        return out

    def patches(self, frame, slot):
        """[(premultiplied RGBA patch, x0, y0)] of the units of the entries in `slot` (True: -1, False: None) laid on
        Blender frame `frame`, in the order of the entries and their units."""
        slot = -1 if slot is True else None if slot is False else slot
        f = int(frame) - self.frame0
        out = []
        for p in self.plans:
            if p["slot"] != slot:
                continue
            for u in CT.on_screen(p, f):
                st = CT.state(p, u, f, self.fps)
                if st is None or st["alpha"] <= 0.0:
                    continue
                sp = self._sprite(p, u)
                rgba = sp.at(st["reveal"])
                if st["peel"] is not None:
                    rgba = peeled(rgba, st["peel"]["fold"], st["peel"]["corner"], sp.back)
                if st["wipe"] is not None:
                    rgba = wiped(rgba, st["wipe"])
                if st["alpha"] < 1.0:
                    rgba = rgba * np.float32(st["alpha"])
                M = np.asarray(CT.matrix(st, self.size), np.float64)
                ax, ay = sp.anchor
                M[:, 2] -= M[:, :2] @ np.array([ax, ay])
                got = place(rgba, M, p["craft"]["shadow"], st["lift"], self.size, self.px)
                if got is not None:
                    out.append(got)
        return out
