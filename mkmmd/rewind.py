"""Rewinds on the finished pictures (docs/design.md: Frame effects, Rewinds): the picture work of `kind = "rewind"`, which
mkmmd.core.rewind checks, times and lays out. On each frame of its span the picture is the cut's own (before the time
map's first segment) or the cut's frame at the second the scrub shows, with the effects before the rewind in the table
applied to it as they are there (no type: mk post lays type on after the frame effects); a segment's start tears it as a
tape shuttling does (mkmmd.framefx `split_shifts`' torn bands over a ghost of the pose before); then the scrub bar is
laid on it, each piece drawn once at the output's size as craft type draws its units (mkmmd.crafttype: a torn strip of
masking tape, glossy label tape with raised marks, a die-cut sticker, a paper tag; SS times larger, boxed down) and
carried onto the frame by one map with its soft drop shadow (`crafttype.place`). Images are float RGB in display space
(0..1), (h, w, 3); a sprite is premultiplied RGBA."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import crafttype as CR
from .core import rewind as RW

SS = CR.SS                       # pieces are drawn this many times larger and boxed down
PAPER = (1.0, 0.995, 0.985)      # a die-cut rim's white paper (craft type's sticker back)
TEMPLATE = "8:88.8"              # the counter's widest text: its tag is cut to fit it


def _ff():
    from . import framefx as FF          # framefx imports this module for RewindFx: its helpers are there by now
    return FF


# ================================================================================================= the pieces at rest
def bar_sprites(length, height, colour, played, ink, marks, seed, px):
    """The scrub bar at rest, a strip of masking tape `length` px between its timeline's ends (the tape runs on a little
    past them to its torn ends) and `height` px wide: (plain, painted or None, (x, y) sprite px of the timeline's
    middle), premultiplied RGBA of one size; `painted` is the tape coloured `played` (the part before the playhead).
    `marks` [(0..1 along the timeline, long)] are pen ticks in `ink` up from its lower edge, a long one twice as tall."""
    L, Hh = float(length) * SS, float(height) * SS
    cap = Hh / 1.9                                         # the masking craft's tape is about 1.9 caps wide
    depth, pad, margin = 0.32 * cap, 0.35 * Hh, 0.5 * Hh
    ww, hh = int(math.ceil(L + 2 * (pad + depth + margin))), int(math.ceil(Hh + 2 * margin))
    x0 = margin + depth + pad                              # the timeline's 0
    left, right, top, bot = margin, ww - margin, margin, margin + Hh
    strip = CR._tape_strip((hh, ww), left, right, top, bot, depth, seed, slant=0.05)
    crepe = 0.5 * CR._streaks((hh, ww), 0.2 * cap, 0.035 * cap, 82.0, seed + 3) + \
        0.5 * CR._streaks((hh, ww), 0.15 * cap, 0.03 * cap, 97.0, seed + 4)
    yy = (np.arange(hh, dtype=np.float32)[:, None] - top) / Hh
    grain = 0.035 * CR._noise((hh, ww), 0.8 * cap, seed + 5, 2)
    edge = np.clip(1.0 - np.minimum(yy, 1.0 - yy) / 0.14, 0.0, 1.0) * (np.abs(yy - 0.5) <= 0.5)   # its long edges darker
    k = 0.04 * crepe + grain - 0.22 * edge ** 1.5                                                  # (a light ground too)
    alpha = 0.92 * (0.95 + 0.05 * CR._noise((hh, ww), 0.4 * cap, seed + 6))
    tick = np.zeros((hh, ww), np.uint8)                    # the pen's ticks, a little wobbly (a hand drew them)
    rng = np.random.default_rng(seed + 9)
    w = max(int(round(2.6 * px * SS)), 1)                 # px (cv2's `shift` scales the points, not the thickness)
    for u, long in marks:
        x = x0 + u * L
        y1 = bot - 0.1 * Hh
        y2 = bot - (0.68 if long else 0.4) * Hh
        dx = rng.uniform(-0.6, 0.6) * SS
        cv2.line(tick, (int(round(x * 4)), int(round(y1 * 4))), (int(round((x + dx) * 4)), int(round(y2 * 4))), 255,
                 w, cv2.LINE_AA, 2)
    tick = tick.astype(np.float32) / 255.0 * 0.95 * (0.92 + 0.08 * CR._noise((hh, ww), 0.3 * cap, seed + 8))
    out = []
    for c in (colour, played):
        if c is False:
            out.append(None)
            continue
        col = CR._shade(np.broadcast_to(np.asarray(c, np.float32), (hh, ww, 3)), np.broadcast_to(k, (hh, ww)))
        body = np.zeros((hh, ww, 4), np.float32)
        CR._over(body, np.clip(col, 0, 1), strip * alpha)
        CR._over(body, np.asarray(ink, np.float32), tick * strip)
        out.append(CR._down(body))
    return out[0], out[1], ((x0 + 0.5 * L) / SS, (top + 0.5 * Hh) / SS)


def head_sprite(d, colour, rim, seed):
    """The playhead: a round sticker `d` px across in `colour`, glossy and lit from the upper left, in a die-cut white
    rim `rim` px wide: premultiplied RGBA, its middle the sprite's middle."""
    D, R = float(d) * SS, float(rim) * SS
    n = int(math.ceil(D + 2.0 * R + 6 * SS))
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - c, yy - c)
    disc = np.clip(0.5 * D - r + 0.5, 0.0, 1.0)
    outer = np.clip(0.5 * D + R - r + 0.5, 0.0, 1.0)
    nx, ny = (xx - c) / (0.5 * D), (yy - c) / (0.5 * D)
    k = -0.16 * np.clip(0.55 * nx + 0.83 * ny, -1.0, 1.0) + 0.02 * CR._noise((n, n), 0.3 * D, seed)
    col = CR._shade(np.broadcast_to(np.asarray(colour, np.float32), (n, n, 3)), k)
    gloss = 0.55 * np.exp(-((nx + 0.3) ** 2 / 0.05 + (ny + 0.42) ** 2 / 0.02))     # a highlight on its upper left
    col = col + (1.0 - col) * np.clip(gloss, 0, 1)[..., None]
    paper = CR._shade(np.broadcast_to(np.asarray(PAPER, np.float32), (n, n, 3)),
                      -0.05 * np.clip(r - (0.5 * D + R - 1.5 * SS), 0, None) / (1.5 * SS))   # the cut edge
    body = np.zeros((n, n, 4), np.float32)
    CR._over(body, np.clip(paper, 0, 1), outer)
    CR._over(body, np.clip(col, 0, 1), disc)
    return CR._down(body)


def _embossed(mask, colour, ink, seed, unit):
    """Glossy label tape in `colour` with the shapes `mask` (coverage, the strip's size) pushed up from behind in `ink`,
    lit from the upper left, as craft type's emboss lays its capitals (`unit` the shapes' height, px): (h, w, 3)."""
    hh, ww = mask.shape
    yy = np.linspace(0.0, 1.0, hh, dtype=np.float32)[:, None] * np.ones((1, ww), np.float32)
    base = np.asarray(colour, np.float32)
    k = 0.03 - 0.07 * yy + 0.015 * CR._noise((hh, ww), 1.2 * unit, seed, 2)
    edge = np.clip(1.0 - np.minimum(yy, 1.0 - yy) * hh / (0.06 * unit), 0, 1)
    col = CR._shade(np.broadcast_to(base, (hh, ww, 3)), k - 0.12 * edge)
    sheen = 0.32 * np.exp(-((yy - 0.16) / 0.05) ** 2) + 0.10 * np.exp(-((yy - 0.3) / 0.12) ** 2)
    col = col + (1.0 - col) * np.clip(sheen * (0.8 + 0.2 * CR._noise((hh, ww), 3.0 * unit, seed + 2)), 0, 1)[..., None]
    halo = np.clip(CR._blur(CR._grow(mask, 0.06 * unit), 0.07 * unit) * 1.4, 0, 1) * 0.55
    col = col + (CR._mix(base, CR.WHITE, 0.5) - col) * halo[..., None]
    face = CR._mix(ink, base, 0.10) + 0.025 * CR._noise((hh, ww), 0.4 * unit, seed + 4)[..., None]
    col = col + (face - col) * mask[..., None]
    sig = 0.022 * unit
    gy, gx = np.gradient(CR._blur(mask, sig))
    light = np.clip(-(gx * -0.55 + gy * -0.83) * sig * 4.0, -1.0, 1.0)
    near = np.clip(CR._grow(mask, 0.04 * unit), 0, 1)
    col = col + (1.0 - col) * (np.clip(light, 0, 1) * near * 0.55)[..., None]
    col = col + (CR._shade(base, -0.35) - col) * (np.clip(-light, 0, 1) * near * 0.45)[..., None]
    return np.clip(col, 0, 1)


def mark_sprite(h, colour, ink, seed):
    """The rewind mark: a piece of label tape `h` px wide, pinked at both ends, two triangles pointing left raised in it
    side by side (the double arrow): premultiplied RGBA, its middle the sprite's middle."""
    H = float(h) * SS
    tri_h, tri_w, tooth = 0.5 * H, 0.42 * H, 0.08 * H
    W = 2.0 * tri_w + 2.0 * 0.42 * H + 2.0 * tooth
    margin = 0.2 * H
    hh, ww = int(math.ceil(H + 2 * margin)), int(math.ceil(W + 2 * margin))
    L, R, T = margin, margin + W, margin
    n = max(int(round(H / (0.11 * H))), 4)
    left = [(L + (tooth if k % 2 == 0 else 0.0), T + H * k / n) for k in range(n + 1)]
    right = [(R - (tooth if k % 2 == 0 else 0.0), T + H * k / n) for k in range(n + 1)]
    strip = CR._poly_mask((hh, ww), right + left[::-1])
    cx, cy = margin + 0.5 * W, T + 0.5 * H
    arrows = np.zeros((hh, ww), np.float32)
    for x in (cx - tri_w, cx):                             # ◀◀: each triangle's point on its left
        arrows = np.maximum(arrows, CR._poly_mask((hh, ww), [(x, cy), (x + tri_w, cy - 0.5 * tri_h),
                                                             (x + tri_w, cy + 0.5 * tri_h)]))
    col = _embossed(CR._blur(arrows, 0.6 * SS), colour, ink, seed, tri_h)
    body = np.zeros((hh, ww, 4), np.float32)
    CR._over(body, col, strip)
    return CR._down(body)


def counter_sprite(text, cap, colour, ink, font, seed):
    """The counter: `text` in the font file `font`, its digits `cap` px tall, in `ink` on a paper tag in `colour` cut to
    fit TEMPLATE (round corners, its cut edge a shade darker, a faint grain): premultiplied RGBA, its middle the sprite's
    middle."""
    C = float(cap) * SS
    f = ImageFont.truetype(font, 400)
    l, t, r, b = f.getbbox("8", anchor="ls")
    em = int(round(400.0 * C / max(b - t, 1)))
    f = ImageFont.truetype(font, max(em, 4))
    x0, y0, x1, y1 = f.getbbox(TEMPLATE, anchor="ls")
    pad_x, pad_y = 0.55 * C, 0.42 * C
    W, H = (x1 - x0) + 2 * pad_x, C + 2 * pad_y
    margin = 0.2 * C
    hh, ww = int(math.ceil(H + 2 * margin)), int(math.ceil(W + 2 * margin))
    tag = np.zeros((hh, ww), np.uint8)
    rad = 0.28 * H
    cv2.rectangle(tag, (int(margin + rad), int(margin)), (int(margin + W - rad), int(margin + H)), 255, -1)
    cv2.rectangle(tag, (int(margin), int(margin + rad)), (int(margin + W), int(margin + H - rad)), 255, -1)
    for cx in (margin + rad, margin + W - rad):
        for cy in (margin + rad, margin + H - rad):
            cv2.circle(tag, (int(round(cx)), int(round(cy))), int(round(rad)), 255, -1, cv2.LINE_AA)
    tag = CR._blur(tag.astype(np.float32) / 255.0, 0.6 * SS)
    img = Image.new("L", (ww, hh), 0)
    tx1 = f.getbbox(text, anchor="ls")[2]
    ImageDraw.Draw(img).text((margin + W - pad_x - tx1, margin + pad_y + C), text, font=f, fill=255, anchor="ls")
    digits = np.asarray(img, np.float32) / 255.0
    paper = CR._shade(np.broadcast_to(np.asarray(colour, np.float32), (hh, ww, 3)),
                      0.03 * CR._noise((hh, ww), 0.5 * C, seed, 2))
    edge = np.clip((1.0 - CR._blur(tag, 1.2 * SS)) * tag * 2.5, 0, 1)      # the cut edge, a shade darker
    paper = CR._shade(paper, -0.12 * edge)
    body = np.zeros((hh, ww, 4), np.float32)
    CR._over(body, np.clip(paper, 0, 1), tag)
    CR._over(body, np.asarray(ink, np.float32), digits * tag * 0.92)
    return CR._down(body)


def _about(M, sprite, at):
    """The 2 x 3 map carrying a sprite's own px to the frame's: `M` (a piece's px about its middle to the frame's)
    after moving the sprite's `at` px (its middle by default) to that middle."""
    h, w = sprite.shape[:2]
    ax, ay = ((w - 1) / 2.0, (h - 1) / 2.0) if at is None else at
    return [[M[0][0], M[0][1], M[0][2] - M[0][0] * ax - M[0][1] * ay],
            [M[1][0], M[1][1], M[1][2] - M[1][0] * ax - M[1][1] * ay]]


def _scaled(M, s, about):
    """`M` scaled `s` times about the frame pixel `about` (a piece pressed flat)."""
    return [[M[r][0] * s, M[r][1] * s, (M[r][2] - about[r]) * s + about[r]] for r in (0, 1)]


# ================================================================================================= over a cut
class RewindFx:
    """The rewinds' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fxs, entries, _get, _apply,
    entry_at and _font_of). `_rewind_init` is called once the effects and the fonts are known."""

    def _rewind_init(self):
        """The counters' fonts found now (a missing one is an FX.FxError before any frame is drawn), the memos of the
        pieces drawn (by their colours: a breathing palette draws them again) and of the pictures scrubbed back to."""
        self._rewind_fonts, self._rewind_art, self._rewind_pics = {}, {}, {}
        for fx in self.fxs:
            if fx["kind"] in RW.KINDS and fx["counter"] is not None:
                self._rewind_fonts[fx["index"]] = self._font_of(fx["counter"]["font"])

    def rewind_grip(self, index, frame):
        """The frame pixel of the playhead of fx `index` (a rewind) on `frame`, where a cutout riding it puts its
        anchor (mkmmd.core.rewind.grip)."""
        fx = next(e for e in self.fxs if e["index"] == index)
        return RW.grip(fx, frame, self.size)

    def _scrubbed(self, fx, g, frame, img):
        """The picture of the cut's frame `g` as it reached the rewind: `img` on `frame` itself, else the cut's frame `g`
        with the effects before the rewind in the table applied again as `g` draws them (PictureFx._reached's way),
        kept while it is shown (the pose held, the ghost after it)."""
        if g == frame:
            return img
        memo = self._rewind_pics.setdefault(fx["index"], {})
        if g not in memo:
            pic = self._get(g)
            if pic is None:
                raise FileNotFoundError(f"fx {fx['index']} (rewind): the frame it scrubs back to, {g}, is not there")
            for e in self.entries(g):
                if e["index"] >= fx["index"]:
                    break
                pic = self._apply(e, g, pic)
            memo[g] = pic
        return memo[g]

    def _art(self, fx, part, key, make):
        """A piece of fx's bar drawn once (`make()`), kept by `key` (its size and colours)."""
        memo = self._rewind_art.setdefault((fx["index"], part), {})
        if key not in memo:
            if len(memo) > 32:                       # a breath on its way: only the latest colours are kept
                memo.clear()
            memo[key] = make()
        return memo[key]

    def _rewind(self, fx, frame, img):
        """The picture scrubbed back to (or the cut's own before the first segment), torn at the jumps, with the bar,
        the playhead, the mark and the counter laid on it."""
        sec, g, seg = RW.shown(fx, frame)
        pic = img
        if seg >= 0:
            ghost, alpha = RW.ghost_at(fx, frame)
            need = {g} | ({ghost} if ghost is not None else set())
            memo = self._rewind_pics.setdefault(fx["index"], {})
            for k in [k for k in memo if k not in need]:
                del memo[k]
            pic = self._scrubbed(fx, g, frame, img)
            lvl = RW.tear_at(fx, frame)
            if lvl > 0.0:
                pic = self._torn(fx, frame, pic, ghost, alpha, lvl, seg, img)
        return self._bar(fx, frame, pic, sec)

    def _torn(self, fx, frame, pic, ghost, alpha, lvl, seg, img):
        """`pic` torn as a tape shuttling is, `lvl` hard (0..1): the ghost of the picture before the jump over it at
        `alpha` (just after a jump), then split's torn bands (`bands`, `share` of them torn up to `shift` frame widths x
        lvl, the red and blue planes `rgb` px at 1080 x lvl apart), breaking another way every `hold` s from the
        segment's start."""
        tr, (w, h) = fx["tear"], self.size
        out = pic
        if ghost is not None and alpha > 0:
            a = np.float32(alpha)
            out = (pic * (1.0 - a) + self._scrubbed(fx, ghost, frame, img) * a).astype(np.float32)
        s = fx["map"][seg]
        step = int(math.floor((frame - s["f"]) / (tr["hold"] * fx["frame_rate"]) + 1e-6))
        rng = np.random.default_rng([fx["seed"], fx["index"], seg, step])
        FF = _ff()
        bands = FF.split_shifts(h, rng, tr["bands"], tr["shift"] * w * lvl, tr["share"], tr["rgb"] * self.px * lvl)
        return FF.split_rolled(out, bands)

    def _bar(self, fx, frame, img, sec):
        """`img` with the scrub bar on it: the tape painted up to the playhead, the playhead's sticker (pressed flat
        under the finger from `press`), the mark and the counter, each with its drop shadow; as it is before the bar
        comes down."""
        got = RW.bar_map(fx, frame, self.size)
        if got is None:
            return img
        M, lift = got
        w, h = self.size
        b, hd = fx["bar"], fx["head"]
        length, height = b["length"] * w, b["height"] * h
        marks = [(u / fx["duration"], long) for u, long in RW.ticks(fx["duration"], b["ticks"])]
        plain, painted, mid = self._art(fx, "bar", (w, h, tuple(b["color"]), b["played"] if b["played"] is False else
                                                    tuple(b["played"]), tuple(b["ink"])),
                                        lambda: bar_sprites(length, height, b["color"], b["played"], b["ink"], marks,
                                                            fx["seed"], self.px))
        u = min(max(RW.placed(fx, frame) / fx["duration"], 0.0), 1.0)      # the counter shows `sec`, the second shown
        sprite = plain
        if painted is not None:                       # the part before the playhead painted
            xs = np.arange(plain.shape[1], dtype=np.float32) - (mid[0] + (u - 0.5) * length)
            m = np.clip(0.5 - xs, 0.0, 1.0)[None, :, None]
            sprite = plain * (1.0 - m) + painted * m
        shadow = fx["shadow"]
        patches = []
        got = CR.place(sprite, _about(M, sprite, mid), shadow, lift, self.size, self.px)
        if got is not None:
            patches.append(got)
        ph = RW.playhead(fx, frame, self.size)[0]
        d = hd["size"] * h
        head = self._art(fx, "head", (w, h, tuple(hd["color"])),
                         lambda: head_sprite(d, hd["color"], hd["rim"] * self.px, fx["seed"]))
        scale, flat = RW.pressed(fx, frame)
        Mh = _scaled([[M[0][0], M[0][1], ph[0]], [M[1][0], M[1][1], ph[1]]], scale, ph)
        sh = None if shadow is None else {**shadow, "offset": [o * (1.0 - 0.65 * flat) for o in shadow["offset"]],
                                          "alpha": shadow["alpha"] * (1.0 - 0.3 * flat)}
        got = CR.place(head, _about(Mh, head, None), sh, lift, self.size, self.px)
        if got is not None:
            patches.append(got)
        for part, sign, make in (("mark", -1.0, self._mark_art), ("counter", 1.0, self._counter_art)):
            if fx[part] is None:
                continue
            pm = RW.piece_map(fx, part, frame, self.size, sign)
            if pm is None:
                continue
            sprite = make(fx, sec)
            got = CR.place(sprite, _about(pm[0], sprite, None), shadow, pm[1], self.size, self.px)
            if got is not None:
                patches.append(got)
        return CR.lay(img, patches)

    def _mark_art(self, fx, sec):
        mk, h = fx["mark"], self.size[1]
        return self._art(fx, "mark", (self.size, tuple(mk["color"]), tuple(mk["ink"])),
                         lambda: mark_sprite(mk["size"] * h, mk["color"], mk["ink"], fx["seed"]))

    def _counter_art(self, fx, sec):
        c, h = fx["counter"], self.size[1]
        text = RW.counter_text(sec)
        return self._art(fx, "counter", (self.size, tuple(c["color"]), tuple(c["ink"]), text),
                         lambda: counter_sprite(text, c["size"] * h, c["color"], c["ink"],
                                                self._rewind_fonts[fx["index"]], fx["seed"]))
