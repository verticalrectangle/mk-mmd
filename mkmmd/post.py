"""Post: the grade applied to rendered frames before encoding (docs/design.md: Post). Images are float RGB in
display space (0..1, sRGB as rendered through the view transform).

[post] keys (all optional)
  contrast = 1.10, pivot = 0.60       midtone contrast around a pivot
  saturation = 1.06
  split = {shadows = "iris", highlights = "gold", amount = 0.06}   split toning toward palette slots
  floor = "base", toe = 0.12          the darkest colour allowed: luminance below floor + toe is lifted smoothly into
                                      that band and tinted toward the floor colour, so nothing in the frame is black
  halation = {strength = 0.22, radius = 0.012, threshold = 0.72, tint = "rose"}   glow around highlights (radius as a
                                      fraction of the frame height), tinted like film halation
  vignette = 0.12                     corner darkening (fraction)
  grain = {amount = 0.012, size = 1.5, seed = 7}"""
import cv2
import numpy as np
from PIL import Image

from .core.palette import resolve, srgb

Y_W = np.array([0.2126, 0.7152, 0.0722], np.float32)


class Grade:
    def __init__(self, spec, palette, size):
        self.spec = spec or {}
        self.pal = palette
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
        fl = self.spec.get("floor", "base")
        self.floor = np.array(srgb(resolve(fl, palette)), np.float32) if fl else None
        self.toe = float(self.spec.get("toe", 0.12))
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

    def __call__(self, img, i=0):
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
        img = y0 + (img - y0) * float(s.get("saturation", 1.06))
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
