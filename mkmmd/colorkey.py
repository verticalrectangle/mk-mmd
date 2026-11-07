"""Colour keys pushed too far on the finished pictures (docs/design.md: Frame effects, Keys pushed too far): the picture
work of the `key` kind that mkmmd.core.colorkey checks and times. Images are float RGB in display space (0..1),
(h, w, 3). The key is worked out per pixel: how far its colour lies from the key colour (`distance`), a matte for each
of the red, green and blue planes at its own tolerance, crushed into steps (`matte`), the mattes of the frames before
kept as a fading trail (FrameFx), and what shows through laid in plane by plane (`lay`)."""
from collections import OrderedDict

import numpy as np

from .core import colorkey as CK
from .core import framefx as FX

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)       # BT.709
CB, CR = 1.8556, 1.5748                                       # BT.709's Cb and Cr scales
TRAIL = 16                      # trail mattes kept (uint8, 6 MB a frame at 1080p): a burn's frames, for every key


def _ff():
    from . import framefx as FF          # framefx imports this module for KeyFx: its helpers are there by now
    return FF


def distance(img, color, space):
    """(h, w) float32: how far each pixel's colour lies from `color`: in RGB (the Euclidean distance over sqrt 3, 0..1),
    in the chroma plane alone (BT.709 Cb Cr: brightness left out, a chroma keyer's) or in brightness alone (BT.709
    luma)."""
    img = np.asarray(img, np.float32)
    c = np.asarray(color, np.float32)
    if space == "rgb":
        d = img - c
        return np.sqrt(np.einsum("hwc,hwc->hw", d, d) * np.float32(1.0 / 3.0))
    y = img @ LUMA
    yc = np.float32(c @ LUMA)
    if space == "luma":
        return np.abs(y - yc)
    cb = (img[..., 2] - y) * np.float32(1.0 / CB) - np.float32((c[2] - yc) / CB)
    cr = (img[..., 0] - y) * np.float32(1.0 / CR) - np.float32((c[0] - yc) / CR)
    return np.sqrt(cb * cb + cr * cr)


def matte(d, tol, soft, split, bands):
    """(h, w, 3) float32: how much of each plane is keyed out, 1 within `tol` (a number, or (h, w): each pixel's own)
    + its `split` of the key colour, falling to 0 over `soft` further (a hard step without it), crushed into `bands`
    equal steps (0: smooth)."""
    t = np.asarray(tol, np.float32)[..., None] + np.asarray(split, np.float32)
    d = np.asarray(d, np.float32)[..., None]
    if soft > 0:
        a = np.clip((t + np.float32(soft) - d) * np.float32(1.0 / soft), 0.0, 1.0)
    else:
        a = (d <= t).astype(np.float32)
    if bands:
        a = np.round(a * np.float32(bands)) * np.float32(1.0 / bands)
    return a.astype(np.float32)


def lay(img, back, a, gain, spill=None, d=None, tol=0.0, soft=0.0, work=None):
    """`img` with what shows through (`back`: (h, w, 3), or a colour) laid in by each plane's matte `a`, its planes
    pushed by `gain` first; with a `spill` ({color, amount, reach}) what is left near the key colour (`d`, the
    distance, under `tol` (a number or each pixel's) + `soft` + reach) stained toward its colour; all of it only where
    `work` ((h, w) 0..1, a mask) says. Clipped to 0..1."""
    img = np.asarray(img, np.float32)
    b = np.asarray(back, np.float32) * np.asarray(gain, np.float32)
    out = img + (b - img) * a
    if spill is not None:
        s = np.clip((np.asarray(tol, np.float32) + np.float32(soft + spill["reach"]) - d) *
                    np.float32(1.0 / spill["reach"]), 0.0, 1.0)
        s *= np.float32(spill["amount"]) * (1.0 - a.max(axis=-1))
        out += (np.asarray(spill["color"], np.float32) - out) * s[..., None]
    if work is not None:
        out = img + (out - img) * np.asarray(work, np.float32)[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ================================================================================================= over a cut
class KeyFx:
    """The keys' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, frame0, _t, _get, _hit_times,
    _masked, _mix, _stem_of and _footage_of)."""

    def _key_init(self):
        self._key_levels = {}             # (index, stem, timeline, smooth, range) -> the stem's loudness per frame
        self._key_trail = OrderedDict()   # (index, frame, how it keys) -> a cut frame's mattes, uint8, for its trail

    def key_trail(self, fx, g):
        """The mattes of the cut's frame `g` keyed as `fx` keys at `g`'s own tolerance (uint8, 0..255; None where the
        cut has no frame), kept for the frames after it whose `burn` reads them: a trail never depends on the order the
        frames come in."""
        memo = (fx["index"], g, tuple(fx["color"]), fx["space"], fx["soft"], tuple(fx["split"]), fx["bands"],
                fx["spread"])
        if memo in self._key_trail:
            self._key_trail.move_to_end(memo)
            return self._key_trail[memo]
        pic = self._get(g) if g >= self.frame0 else None
        m = None if pic is None else np.round(matte(distance(pic, fx["color"], fx["space"]), self.key_tolerance(fx, g) -
                                                    self.key_spread(fx, g), fx["soft"], fx["split"], fx["bands"]) *
                                              255.0).astype(np.uint8)
        self._key_trail[memo] = m
        while len(self._key_trail) > TRAIL:
            self._key_trail.popitem(last=False)
        return m

    def key_level(self, fx, frame):
        """0..1, the loudness of a key's `follow` stem on `frame` (mkmmd.core.colorkey.levels over its span; a frame
        before the span reads its first)."""
        f = fx["follow"]
        memo = (fx["index"], f["stem"], f["timeline"], f["smooth"], f["range"])
        if memo not in self._key_levels:
            if self._stem_of is None:
                raise FileNotFoundError(f"fx {fx['index']} (key): no audio to follow")
            x, sr, pre = self._stem_of(f["timeline"], f["stem"])
            self._key_levels[memo] = CK.levels(x, sr, pre, self._t(fx["f0"]), self._t(fx["f1"]), self.fps,
                                               f["smooth"], f["range"])
        lv = self._key_levels[memo]
        return float(lv[min(max(frame - fx["f0"], 0), len(lv) - 1)])

    def key_tolerance(self, fx, frame):
        """A key's tolerance on `frame`: its keys, its `follow` stem's loudness and its pump on the hits."""
        level = self.key_level(fx, frame) if fx["follow"] is not None else 0.0
        times = self._hit_times(fx) if fx["pump"] else ()
        return CK.tolerance(fx, self._t(frame), level, times)

    def key_spread(self, fx, frame):
        """(h, w) float32, the tolerance a key with a `spread` loses at each pixel on `frame`: `spread` for each frame
        height of its distance from the `center` there (read on the span's first frame before the span); 0 without."""
        if not fx["spread"]:
            return 0.0
        (cx, cy), _ = self._at(fx, max(frame, fx["f0"]))
        w, h = self.size
        ys = (np.arange(h, dtype=np.float32) - np.float32(cy))[:, None]
        xs = (np.arange(w, dtype=np.float32) - np.float32(cx))[None, :]
        return np.sqrt(xs * xs + ys * ys) * np.float32(fx["spread"] / h)

    def key_back(self, fx, frame, img):
        """What shows through a key on `frame`: its clip's picture there fitted to cover the frame, its flat colour, or
        the cut's frame `delay` frames before (the clip's first frame stands in before the clip)."""
        if fx["source"] == "back":
            return np.asarray(fx["back"], np.float32)
        if fx["source"] == "delay":
            pic = self._get(max(frame - fx["delay"], self.frame0))
            return img if pic is None else pic
        if self._footage_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (key): no footage to read")
        clip = _ff().clip_of(fx, self._footage_of)
        return clip.picture((frame - fx["f0"]) / self.fps, self.size, fx["in"], fx["rate"], fx["loop"], "cover",
                            fx["focus"], fx["mirror"])

    def key_matte(self, fx, frame, img):
        """((h, w, 3) mattes, (h, w) distance, tolerance) of a key on `frame`: the picture's own matte, and with a
        `burn` the mattes of the cut's frames before it (keyed at their own tolerance) kept as a trail, frame j back
        weighing `fade` ** j, the strongest counting."""
        tol = self.key_tolerance(fx, frame) - self.key_spread(fx, frame)
        d = distance(img, fx["color"], fx["space"])
        a = matte(d, tol, fx["soft"], fx["split"], fx["bands"])
        for j in range(1, fx["burn"] + 1):
            m = self.key_trail(fx, frame - j)
            if m is not None:
                np.maximum(a, m.astype(np.float32) * np.float32(fx["fade"] ** j / 255.0), out=a)
        return a, d, tol

    def _key(self, fx, frame, img):
        if FX.strength(fx, self._t(frame)) <= 0.0:
            return img
        a, d, tol = self.key_matte(fx, frame, img)
        out = lay(img, self.key_back(fx, frame, img), a, fx["gain"], fx["spill"], d, tol, fx["soft"],
                  self._masked(fx, frame))
        return self._mix(fx, frame, img, out)
