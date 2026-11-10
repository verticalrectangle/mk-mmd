"""Insets on the finished pictures (docs/design.md: Frame effects, Insets): the picture work of the `inset` kind that
mkmmd.core.inset checks and lays out. Images are float RGB in display space (0..1), (h, w, 3). The clip is read at its
own size and seen through the view's window (`backdrop`); the picture is shrunk by area to its place and set there to
the part of a pixel (`lay`), so it neither shimmers in a small screen nor jumps as the view moves."""
import cv2
import numpy as np

from . import footage as FT
from .core import inset as IN


def _ff():
    from . import framefx as FF          # framefx imports this module for InsetFx: its helpers are there by now
    return FF


def backdrop(rgb, cover, win, size):
    """A clip's frame `rgb` (uint8 RGB as decoded; its part `cover` (x, y, w, h) fills a picture of `size` (w, h) at
    view 0) as the window `win` (mkmmd.core.inset.window, px of that picture) shows it: float RGB of `size`."""
    (x0, y0, cw, ch), (W, H) = cover, size
    ax, ay = (win[2] - win[0]) / W * cw / W, (win[3] - win[1]) / H * ch / H      # clip px a picture px, across and down
    M = np.float32([[ax, 0.0, x0 + win[0] * cw / W + 0.5 * ax - 0.5], [0.0, ay, y0 + win[1] * ch / H + 0.5 * ay - 0.5]])
    how = cv2.INTER_CUBIC if ax < 1.0 else cv2.INTER_LINEAR
    out = cv2.warpAffine(rgb, M, (W, H), flags=how | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32) / np.float32(255.0)


def lay(out, pic, box):
    """`pic` laid into the box (x0, y0, x1, y1) px of `out`, in place: resized (by area where it shrinks) to the box's
    size in whole pixels, then set on the box itself to the part of a pixel, its edge antialiased."""
    x0, y0, x1, y1 = (float(v) for v in box)
    H, W = out.shape[:2]
    w, h = x1 - x0, y1 - y0
    iw, ih = max(int(round(w)), 1), max(int(round(h)), 1)
    ph, pw = pic.shape[:2]
    small = pic if (iw, ih) == (pw, ph) else cv2.resize(
        pic, (iw, ih), interpolation=cv2.INTER_AREA if iw * ih < pw * ph else cv2.INTER_LINEAR)
    kx, ky = w / iw, h / ih
    M = np.float32([[kx, 0.0, x0 + 0.5 * kx - 0.5], [0.0, ky, y0 + 0.5 * ky - 0.5]])
    got = cv2.warpAffine(np.dstack([small, np.ones((ih, iw), np.float32)]), M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out *= np.float32(1.0) - got[..., 3:]
    out += got[..., :3]
    return out


class InsetFx:
    """The inset's part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, fps, _t and _footage_of)."""

    def inset_at(self, fx, frame):
        """(clip, seconds into the span, view, the picture's box at view 0 px, the clip's part that covers the frame)
        of an inset on `frame`."""
        if self._footage_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (inset): no footage to read")
        clip = _ff().clip_of(fx, self._footage_of)
        w, h = clip.size
        W, H = self.size
        cover = FT.fit_box((w, h), self.size, "cover", fx["focus"], fx["mirror"])[0]
        a0, b0, a1, b1 = fx["into"]
        if fx["mirror"]:
            a0, a1 = 1.0 - a1, 1.0 - a0
        box = ((a0 * w - cover[0]) * W / cover[2], (b0 * h - cover[1]) * H / cover[3],
               (a1 * w - cover[0]) * W / cover[2], (b1 * h - cover[1]) * H / cover[3])
        return clip, (frame - fx["f0"]) / self.fps, IN.view(fx, self._t(frame)), box, cover

    def _inset(self, fx, frame, img):
        if IN.view(fx, self._t(frame)) >= 1.0:                 # the picture alone: exactly as it came
            return img
        clip, t, v, box, cover = self.inset_at(fx, frame)
        if v <= 0.0:                                            # the clip's whole frame, fitted as footage is
            out = np.array(clip.picture(t, self.size, fx["in"], fx["rate"], fx["loop"], "cover", fx["focus"],
                                        fx["mirror"]), np.float32)
            win = (0.0, 0.0, float(self.size[0]), float(self.size[1]))
        else:
            rgb = clip.frame(clip.index(t, fx["in"], fx["rate"], fx["loop"]))
            win = IN.window(box, self.size, v)
            out = backdrop(rgb[:, ::-1] if fx["mirror"] else rgb, cover, win, self.size)
        return lay(out, np.asarray(img, np.float32), IN.place(box, win, self.size))
