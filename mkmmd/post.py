"""Post: the grade applied to rendered frames before encoding (docs/design.md: Post). Images are float RGB in
display space (0..1, sRGB as rendered through the view transform).

[post] keys (all optional)
  streaks = {strength = 0.35, threshold = 0.88, length = 0.15, core = 0.02, tint = "foam", mix = 0.75, cap = 1.0}
                                      horizontal lens streaks (an anamorphic lens's flare) from the brightest points of
                                      the picture, laid on before the screen type (Streaks)
  contrast = 1.10, pivot = 0.60       midtone contrast around a pivot
  saturation = 1.06
  keep = {cast = "<name>", materials = ["瞳"], saturation = 1.0}   those materials of a cast member keep their colour
                                      (their own saturation) where they are seen: a black-and-white picture with her eyes
                                      in colour (mkmmd.core.keep; the mask `keep/<frame>.png` mk render draws)
  split = {shadows = "iris", highlights = "gold", amount = 0.06}   split toning toward palette slots
  floor = <darkest>, toe = 0.12       the darkest colour allowed (default: the palette's darkest slot, `base` in the dark
                                      palettes, `text` in Dawn): luminance below floor + toe is lifted smoothly into that
                                      band and tinted toward the floor colour, so nothing in the frame is black
  halation = {strength = 0.22, radius = 0.012, threshold = 0.72, tint = "rose"}   glow around highlights (radius as a
                                      fraction of the frame height), tinted like film halation
  vignette = 0.12                     corner darkening (fraction)
  grain = {amount = 0.012, size = 1.5, seed = 7}

With `[look] breathe` (Breath) every colour post takes from the palette is resolved with both palettes and mixed by
each frame's mix: the cut plan's entries (BR.Pair), the grade's and the streaks' colours (Mixed)."""
import copy
import math

import cv2
import numpy as np
from PIL import Image

from .core import breathe as BR
from .core.palette import darkest, resolve, srgb

Y_W = np.array([0.2126, 0.7152, 0.0722], np.float32)

STREAKS = {"strength": 0.35, "threshold": 0.88, "length": 0.15, "core": 0.02, "tint": "foam", "mix": 0.75, "cap": 1.0}
STREAK_SCALE = (4, 2)               # the streaks are made at a quarter of the width and half of the height


class Streaks:
    """Horizontal lens streaks, the flare of an anamorphic lens: luminance above `threshold` (eased in over the rest of
    the range) is smeared sideways, a tight core of Gaussian width `core` and a long tail of width `length` (fractions of
    the frame's longer side, so every output streaks alike), each normalised to its peak so a lone lamp draws a thin line
    about as bright as itself; the sum is capped softly at `cap` (a big source saturates instead of flooding the frame),
    tinted `mix` of the way toward the palette slot `tint` and screened over the picture at `strength`."""

    COLOURS = ("tint",)             # what it takes from the palette (`_colour`; Mixed mixes them)

    def __init__(self, spec, palette, size):
        bad = sorted(set(spec) - set(STREAKS))
        if bad:
            raise ValueError(f"[post] streaks: unknown key{'s' if len(bad) > 1 else ''} {', '.join(bad)} "
                             f"(have {', '.join(STREAKS)})")
        s = {**STREAKS, **spec}
        self.w, self.h = size
        self.small = (max(1, self.w // STREAK_SCALE[0]), max(1, self.h // STREAK_SCALE[1]))
        long_px = max(self.w, self.h) / STREAK_SCALE[0]
        self.sigmas = (float(s["core"]) * long_px, float(s["length"]) * long_px)
        self.strength, self.thr, self.mix, self.cap = (float(s[k]) for k in ("strength", "threshold", "mix", "cap"))
        if not (0.0 <= self.thr < 1.0 and self.cap > 0.0 and min(self.sigmas) > 0.0):
            raise ValueError("[post] streaks: threshold in [0, 1), cap > 0, length and core > 0")
        self._slot = s["tint"]
        self._colour(palette)

    def _colour(self, palette):
        self.tint = np.array(srgb(resolve(self._slot, palette)), np.float32)

    @staticmethod
    def _hblur(img, sigma):
        """A horizontal Gaussian of `sigma` px (three box passes), scaled so that a lone pixel keeps its value at the
        centre."""
        w = 2 * int(round(math.sqrt(4.0 * sigma * sigma + 1.0) / 2.0)) + 1
        box = np.full(w, 1.0 / w)
        peak = float(np.convolve(np.convolve(box, box), box).max())
        out = img
        for _ in range(3):
            out = cv2.blur(out, (w, 1), borderType=cv2.BORDER_CONSTANT)
        return out / peak

    def __call__(self, img):
        img = img.astype(np.float32)
        y = img @ Y_W
        k = np.clip((y - self.thr) / (1.0 - self.thr), 0.0, 1.0) ** 2
        if not k.any():
            return img
        small = cv2.resize(k[..., None] * img, self.small, interpolation=cv2.INTER_AREA)
        raw = 0.6 * self._hblur(small, self.sigmas[0]) + 0.4 * self._hblur(small, self.sigmas[1])
        st = self.cap * (1.0 - np.exp(-np.maximum(raw, 0.0) / self.cap))
        st = cv2.resize(st, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        st = self.mix * (st @ Y_W)[..., None] * self.tint + (1.0 - self.mix) * st
        return 1.0 - (1.0 - img) * (1.0 - np.clip(self.strength * st, 0.0, 1.0))
Y_W = np.array([0.2126, 0.7152, 0.0722], np.float32)


class Grade:
    COLOURS = ("floor", "split", "hal")     # what it takes from the palette (`_colour`; Mixed mixes them)

    def __init__(self, spec, palette, size):
        self.spec = spec or {}
        self.w, self.h = size
        g = self.spec.get("grain", {})
        self.grain_amt = float(g.get("amount", 0.012))
        gs = max(1.0, float(g.get("size", 1.5)))
        rng = np.random.default_rng(int(g.get("seed", 7)))
        gw, gh = max(2, int(self.w / gs)), max(2, int(self.h / gs))
        self.grain = [cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (self.w, self.h),
                                 interpolation=cv2.INTER_LINEAR) for _ in range(8)]
        v = float(self.spec.get("vignette", 0.12))
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        r = np.hypot((xx - self.w / 2) / (self.w / 2), (yy - self.h / 2) / (self.h / 2)) / np.sqrt(2)
        self.vig = (1.0 - v * np.clip(r, 0, 1) ** 2.2)[..., None].astype(np.float32)
        self.toe = float(self.spec.get("toe", 0.12))
        self.keep_sat = float((self.spec.get("keep") or {}).get("saturation", 1.0))
        self._colour(palette)

    def _colour(self, palette):
        """The floor (the palette's darkest slot by default), the split toning's colours and the halation's tint, from
        `palette`."""
        fl = self.spec["floor"] if "floor" in self.spec else darkest(palette)
        self.floor = np.array(srgb(resolve(fl, palette)), np.float32) if fl else None
        sp = self.spec.get("split")
        self.split = None
        if sp:
            self.split = (np.array(srgb(resolve(sp.get("shadows", "iris"), palette)), np.float32),
                          np.array(srgb(resolve(sp.get("highlights", "gold"), palette)), np.float32),
                          float(sp.get("amount", 0.06)))
        hl = self.spec.get("halation")
        self.hal = None
        if hl:
            self.hal = (float(hl.get("strength", 0.22)), max(1.0, float(hl.get("radius", 0.012)) * self.h),
                        float(hl.get("threshold", 0.72)),
                        np.array(srgb(resolve(hl.get("tint", "rose"), palette)), np.float32))

    def __call__(self, img, i=0, keep=None):
        """The graded frame `img` (float RGB, display space), `i` its index (the grain cycles); `keep` (h, w) in 0..1, the
        mask of `[post] keep`, gives its share of the picture the kept saturation instead of the grade's."""
        s = self.spec
        img = img.astype(np.float32)
        if self.hal:
            strength, rad, thr, tint = self.hal
            y = img @ Y_W
            bright = np.clip((y - thr) / max(1e-3, 1 - thr), 0, 1)[..., None] * img
            small = cv2.resize(bright, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            glow = cv2.GaussianBlur(small, (0, 0), rad / 4)
            glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            gl = (glow @ Y_W)[..., None] * tint * 0.6 + glow * 0.4
            img = 1 - (1 - img) * (1 - strength * gl)            # screen
        y0 = (img @ Y_W)[..., None]
        sat = float(s.get("saturation", 1.06))
        if keep is not None:
            sat = (sat + (self.keep_sat - sat) * np.clip(keep, 0.0, 1.0))[..., None]
        img = y0 + (img - y0) * sat
        pv = float(s.get("pivot", 0.60))
        img = pv + (img - pv) * float(s.get("contrast", 1.10))
        if self.split:
            sh, hi, amt = self.split
            y = np.clip(img @ Y_W, 0, 1)[..., None]
            img = img + amt * ((1 - y) * (sh - 0.5) + y * (hi - 0.5))
        img = np.clip(img, 0, 1) * self.vig
        if self.floor is not None:
            yf = float(self.floor @ Y_W)
            lim = yf + self.toe
            y = img @ Y_W
            u = np.clip(y / lim, 0, 1)
            f = u * u * (2 - u)
            ny = np.where(y < lim, yf + (lim - yf) * f, y)
            scale = ny / np.maximum(y, 1e-4)
            w = ((1 - u) ** 2)[..., None]
            img = img * scale[..., None] * (1 - w) + (self.floor + (img - y[..., None]) * 0.3) * w
        img = img + (self.grain[i % len(self.grain)] * self.grain_amt)[..., None]
        lo = float(self.floor @ Y_W) * 0.98 if self.floor is not None else 0.0
        return np.clip(img, lo, 1.0)


class Mixed:
    """A Grade or Streaks `a` and the same resolved with the palette `other` too: `at(m)` the one whose colours (its
    COLOURS) lie `m` of the way from a's to the other palette's, display-space floats mixed as `"a:b:t"` mixes (BR.Pair):
    `a` itself at 0, the other palette's at 1, else a copy of `a` that shares the rest (its grain, its vignette)."""

    def __init__(self, a, other):
        self.a, self.b = a, copy.copy(a)
        self.b._colour(other)
        self.pair = BR.Pair({k: getattr(a, k) for k in a.COLOURS}, {k: getattr(self.b, k) for k in a.COLOURS},
                            type(a).__name__)

    def at(self, m):
        if self.pair.static or m <= 0.0:
            return self.a
        if m >= 1.0:
            return self.b
        out = copy.copy(self.a)
        out.__dict__.update(self.pair.at(m))
        return out


def _named(kind, e):
    """How the plan's errors name entry `e` of `kind`."""
    if kind == "transition":
        return f"transition {e['index']} ({e['kind']} at {e['at']} s)"
    if kind == "fx":
        return f"fx {e['index']} ({e['kind']})"
    return f"{kind} {e['index']} ({e['from']}-{e['to']} s)"


class Breath:
    """`[look] breathe` in `mk post` and `mk look` (docs/design.md: Post; mkmmd.core.breathe): every colour they take
    from the palette is resolved with the project's palette (mix 0) and with the other one (mix 1), and on each frame
    the two are mixed by that frame's mix (`mix`, held before the first key and after the last). `palettes` is (the
    project's, the other: BR.palettes), `plan` the cut's plan made with the first and `other` the same plan made with
    the second (mkmmd.core.transition.plan; None without one): each entry of theirs makes a BR.Pair (`pairs`, by
    ("transition" | "insert" | "glitch" | "fx", index)), and one that is not the same thing with both palettes is a
    BreatheError that names it."""

    KINDS = (("transitions", "transition"), ("inserts", "insert"), ("glitches", "glitch"), ("fx", "fx"))

    def __init__(self, spec, fps, frame0, palettes, plan=None, other=None):
        self.spec, self.fps, self.frame0, self.palettes = spec, float(fps), int(frame0), tuple(palettes)
        self.pairs = {}
        for key, kind in self.KINDS:
            mine, theirs = (plan or {}).get(key) or [], (other or {}).get(key) or []
            if len(mine) != len(theirs):
                raise BR.BreatheError(f"{len(mine)} {key} with one palette, {len(theirs)} with the other")
            for a, b in zip(mine, theirs):
                self.pairs[kind, a["index"]] = BR.Pair(a, b, _named(kind, a))

    def mix(self, frame):
        """The mix on Blender frame `frame`: 0 the project's palette, 1 the other."""
        return BR.frame_mix(self.spec, self.fps, self.frame0, frame)

    def at(self, kind, entry, frame):
        """The plan's entry `entry` of `kind` as frame `frame` draws it: its colours mixed by the frame's mix (BR.Pair:
        copied only where they differ; `entry` itself where nothing of it breathes)."""
        p = self.pairs.get((kind, entry["index"]))
        return entry if p is None or p.static else p.at(self.mix(frame))


def read(path, size=None):
    """A rendered frame as float RGB (PIL: Blender's PNG metadata makes libpng warn through OpenCV)."""
    try:
        im = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def read_rgba(path, size=None):
    """A screen layer (straight RGBA PNG) as float (h, w, 4), or None when the file is missing or unreadable; cut (not
    resampled) to `size` when it is a pixel or two larger."""
    try:
        im = Image.open(path).convert("RGBA")
    except (OSError, ValueError):
        return None
    if size and im.size != tuple(size):
        w, h = size
        im = im.crop((0, 0, w, h)) if im.size[0] - w in (0, 1) and im.size[1] - h in (0, 1) else im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def luma_min(img):
    return float((img @ Y_W).min())
