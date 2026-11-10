"""Haze on the finished pictures (docs/design.md: Frame effects, Haze): the picture work of the `[[fx]]` kinds `mist`,
`fog` and `bokeh` that mkmmd.core.haze checks and lays out. HazeFx is the part of mkmmd.framefx.FrameFx that draws them:
a mist's bloom (the highlights' light spread round them) and its diffusion (the whole picture a little soft), a fog's
sheets of drifting noise in its colours, thicker toward the ground, and a bokeh's discs of light. A `mask` keeps a fog
and a bokeh behind the figure, but for a veil of the fog and the largest of the lights. Every frame is a pure function
of the entry and the clip time: the same picture wherever a render starts. Images are float RGB in display space
(0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from . import fields as FL
from .core import framefx as FX
from .core import haze as HZ

WORK = 270.0                    # px on the short side at which a fog is worked out, then resized up (it has no detail)
BLUR_PX = 4.0                   # a wide blur is made at the size where its sigma is this many px
GLOW = 2.0                      # a bloom spreads what a highlight has over the threshold this many times as strong
BLOOM = (0.6, 0.4, 2.5)         # ... this much of it at the radius, this much at 2.5 times it
DIFFUSE = 0.6                   # a mist's `diffuse` 1 mixes this much of the soft picture in
FOG_CUT = (0.3, 1.9)            # a fog sheet's noise less 0.3, times 1.9: wisps with clear air between them
FOG_SHEET = (0.75, 0.55)        # how much of the picture the nearest sheet covers where it is thickest, and each further
FOG_TURN = 25.0                 # degrees each further sheet drifts off the fog's way
FOG_SLOW = 0.35                 # ... and this much slower than the one before it
FOG_MORPH = 0.15                # noise slices a second a sheet changes through as it drifts
DISC = 0.5                      # a bokeh light's disc at full level: this much of its colour
RIM = 0.18                      # the brighter rim of a disc is this share of its radius wide
SWELL = 0.25                    # on a hit at `pulse` 1 a light is this much larger


def blur(img, sigma):
    """`img` blurred by a Gaussian of `sigma` px: made small first where the blur is wide (as smooth, and quick)."""
    h, w = img.shape[:2]
    k = sigma / BLUR_PX
    if k <= 1.0:
        return cv2.GaussianBlur(img, (0, 0), max(float(sigma), 0.3), borderType=cv2.BORDER_REPLICATE)
    small = cv2.resize(img, (max(1, int(round(w / k))), max(1, int(round(h / k)))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), BLUR_PX, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def bloom(img, radius, threshold, knee):
    """(h, w, 3) float32: the light of the picture's highlights spread round them. What a pixel's brightest channel has
    over `threshold` (coming in softly from `knee` under it to `knee` over it: only what is over, so a bright ground
    does not wash the picture out), in the pixel's own colour and GLOW times as strong, is blurred at `radius` px and,
    fainter, at 2.5 times it."""
    top = img.max(-1)
    x = top - np.float32(threshold)
    over = np.maximum(x, 0.0)
    if knee > 0.0:
        over = np.where(np.abs(x) < knee, (x + np.float32(knee)) ** 2 / np.float32(4.0 * knee), over)
    lit = (img * (np.float32(GLOW) * over / np.maximum(top, 1e-4))[..., None]).astype(np.float32)
    near, far, wide = BLOOM
    return np.float32(near) * blur(lit, radius) + np.float32(far) * blur(lit, wide * radius)


def mist(img, fx, s):
    """`img` through a mist entry at strength `s`: its diffusion mixed in, then its bloom screened or added on."""
    short = min(img.shape[0], img.shape[1])
    radius = fx["radius"] * short
    glow = bloom(img, radius, fx["threshold"], fx["knee"]) * np.float32(s)
    if fx["tint"] is not None:
        glow = glow * np.asarray(fx["tint"], np.float32)
    base = img
    if fx["diffuse"] > 0.0:
        base = img + (blur(img, 0.5 * radius) - img) * np.float32(DIFFUSE * fx["diffuse"] * min(s, 1.0))
    if fx["blend"] == "screen":
        out = 1.0 - (1.0 - base) * (1.0 - np.clip(glow, 0.0, 1.0))
    else:
        out = base + glow
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fog(size, t, fx):
    """(thick (h, w) float32 0..1, colour (h, w, 3) float32): a fog entry over a picture of `size` (w, h) at clip second
    `t`. `layers` sheets of noise (`detail` octaves each, its patches `scale` short sides across) drift at `speed` short
    sides a second along `angle` (degrees counter-clockwise from the right), each further sheet slower and a little off
    that way; with `floor` the fog thins toward the top of the frame. Its colour runs through `colors` from where it is
    thin to where it is thick."""
    w, h = int(size[0]), int(size[1])
    short = min(w, h)
    k = max(1.0, short / WORK)
    ws, hs = max(2, int(round(w / k))), max(2, int(round(h / k)))
    cell = fx["scale"] * short / k
    X = np.broadcast_to((np.arange(ws, dtype=np.float32) / np.float32(cell))[None, :], (hs, ws))
    Y = np.broadcast_to((np.arange(hs, dtype=np.float32) / np.float32(cell))[:, None], (hs, ws))
    clear = np.ones((hs, ws), np.float32)
    for layer in range(int(fx["layers"])):
        way = math.radians(fx["angle"] + FOG_TURN * layer)
        run = fx["speed"] / fx["scale"] * (1.0 - FOG_SLOW) ** layer * t          # lattice cells the sheet has drifted
        ox, oy = math.cos(way) * run, -math.sin(way) * run                      # y runs down the picture
        n, amp, total = np.zeros((hs, ws), np.float32), 1.0, 0.0
        for octave in range(int(fx["detail"])):
            f = float(2 ** octave)
            n += np.float32(amp) * FL.noise((X - np.float32(ox)) * np.float32(f) + np.float32(17.3 * layer),
                                            (Y - np.float32(oy)) * np.float32(f) + np.float32(9.1 * octave),
                                            FOG_MORPH * t * (1.0 + 0.5 * octave) + 5.0 * layer,
                                            int(fx["seed"]) + 31 * layer + 7 * octave)
            total += amp
            amp *= 0.5
        sheet = np.clip((n / np.float32(total) - np.float32(FOG_CUT[0])) * np.float32(FOG_CUT[1]), 0.0, 1.0)
        clear *= 1.0 - sheet * np.float32(FOG_SHEET[0] if layer == 0 else FOG_SHEET[1])
    thick = 1.0 - clear
    if fx["floor"] > 0.0:
        down = HZ.smooth(np.arange(hs, dtype=np.float32) / np.float32(hs - 1))
        thick = thick * (np.float32(1.0 - fx["floor"]) + np.float32(fx["floor"]) * down)[:, None]
    cols = np.asarray(fx["colors"], np.float32)
    at = np.clip(thick, 0.0, 1.0) * np.float32(len(cols) - 1)
    lo = np.minimum(at.astype(int), max(len(cols) - 2, 0))
    hi = np.minimum(lo + 1, len(cols) - 1)
    colour = cols[lo] + (cols[hi] - cols[lo]) * (at - lo)[..., None]
    thick = cv2.resize(np.ascontiguousarray(thick), (w, h), interpolation=cv2.INTER_LINEAR)
    colour = cv2.resize(np.ascontiguousarray(colour), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(thick, 0.0, 1.0).astype(np.float32), colour.astype(np.float32)


def discs(size, lights, colours, soft, rim, pick=None, swell=1.0):
    """(h, w, 3) float32: the light of a bokeh's `lights` (mkmmd.core.haze.lights) over a picture of `size` (w, h), those
    `pick` (bool, one per light) marks, or all. Each is a disc of its colour at DISC of its level, its edge soft over
    `soft` of its radius, a little brighter (`rim`) just inside the edge, as a lens draws a light out of focus; where
    discs overlap their light adds. `swell` scales every radius."""
    w, h = int(size[0]), int(size[1])
    short = float(min(w, h))
    out = np.zeros((h, w, 3), np.float32)
    cols = np.asarray(colours, np.float32)
    for i in range(len(lights["r"])):
        if pick is not None and not pick[i]:
            continue
        r = float(lights["r"][i]) * short * swell
        cx, cy = float(lights["at"][i][0]) * short - 0.5, float(lights["at"][i][1]) * short - 0.5
        edge = max(soft * r, 1.0)
        x0, x1 = max(int(math.floor(cx - r - edge)), 0), min(int(math.ceil(cx + r + edge)) + 1, w)
        y0, y1 = max(int(math.floor(cy - r - edge)), 0), min(int(math.ceil(cy + r + edge)) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        d = np.hypot(np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(cx),
                     np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(cy))
        disc = HZ.smooth((np.float32(r + 0.5 * edge) - d) / np.float32(edge))
        ring = np.exp(-((d - np.float32(r - 0.5 * edge)) / np.float32(max(RIM * r, 1.0))) ** 2)
        a = np.float32(DISC * float(lights["level"][i])) * disc * (1.0 + np.float32(rim) * ring)
        out[y0:y1, x0:x1] += a[..., None] * cols[int(lights["color"][i])]
    return out


class HazeFx:
    """The haze's part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, frame0, _t, _hit_times and
    _masked)."""

    def _mist(self, fx, frame, img):
        s = max(float(FX.strength(fx, self._t(frame))), 0.0)
        if s <= 0.0:
            return img
        out = mist(img, fx, s)
        region = self._masked(fx, frame)
        return out if region is None else (img + (out - img) * region[..., None]).astype(np.float32)

    def _fog(self, fx, frame, img):
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0 or fx["density"] <= 0.0:
            return img
        thick, colour = fog(self.size, t, fx)
        a = thick * np.float32(fx["density"] * s)
        region = self._masked(fx, frame)
        if region is not None:                                   # behind her, but for a veil of it
            a = a * (region + (1.0 - region) * np.float32(fx["veil"]))
        a = a[..., None]
        if fx["blend"] == "screen":
            return np.clip(1.0 - (1.0 - img) * (1.0 - colour * a), 0.0, 1.0).astype(np.float32)
        return np.clip(img + (colour - img) * a, 0.0, 1.0).astype(np.float32)

    def bokeh_at(self, fx, frame):
        """(lights, swell): a bokeh entry's lights on `frame` (mkmmd.core.haze.lights, their level lifted on a hit) and
        how much larger than themselves the hits before it draw them."""
        t = self._t(frame)
        w, h = self.size
        lights = HZ.lights(fx, t, w / h)
        hits = [(FX.frame_of(x, self.fps, self.frame0) - self.frame0) / self.fps for x in self._hit_times(fx)]
        p = fx["pulse"] * HZ.pulse(t, hits, fx["decay"])
        lights["level"] = lights["level"] * (1.0 + p)
        return lights, 1.0 + SWELL * p

    def _bokeh(self, fx, frame, img):
        s = max(float(FX.strength(fx, self._t(frame))), 0.0)
        if s <= 0.0 or fx["count"] <= 0:
            return img
        lights, swell = self.bokeh_at(fx, frame)
        region = self._masked(fx, frame)
        if region is None:
            lit = discs(self.size, lights, fx["colors"], fx["soft"], fx["rim"], None, swell)
        else:                                                    # behind her, but for the largest of them
            lit = discs(self.size, lights, fx["colors"], fx["soft"], fx["rim"], ~lights["front"], swell) * region[..., None] \
                + discs(self.size, lights, fx["colors"], fx["soft"], fx["rim"], lights["front"], swell)
        lit = lit * np.float32(s)
        if fx["blend"] == "screen":
            out = 1.0 - (1.0 - img) * (1.0 - np.clip(lit, 0.0, 1.0))
        else:
            out = img + lit
        return np.clip(out, 0.0, 1.0).astype(np.float32)
