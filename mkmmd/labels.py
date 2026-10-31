"""Editorial labels on the finished pictures (docs/design.md: Frame effects, Labels): the picture work of the `[[fx]]`
kind `label` that mkmmd.core.labels checks and times. LabelFx is the part of mkmmd.framefx.FrameFx that draws it: each
label's text typed out letter by letter in a small face of the asset registry (tracked out, on a box if it has one),
then its dotted leader line drawn out to the point it names and a dot set there. Images are float RGB in display space
(0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from . import picturefx as PFX
from .core import framefx as FX
from .core import labels as LB


def _ink(out, cov, colour, x0, y0, alpha=1.0):
    """`out` (in place) with `colour` laid where `cov` (float 0..1) covers, its top left at pixel (x0, y0)."""
    h, w = cov.shape
    H, W = out.shape[:2]
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if bx <= ax or by <= ay:
        return
    a = (np.float32(alpha) * cov[ay - y0:by - y0, ax - x0:bx - x0])[..., None]
    out[ay:by, ax:bx] = out[ay:by, ax:bx] * (1.0 - a) + np.asarray(colour, np.float32) * a


def set_text(text, font, em, tracking):
    """(coverage (h, w) float32, baseline row, ink start x, width) of a line of `text` set letter by letter, `tracking`
    px added after each letter (labels are tracked out)."""
    parts, x = [], 0.0
    for ch in text:
        fill, _, base, x0, adv = PFX._line(ch, font, em, 0)
        parts.append((fill, base, x0, x))
        x += adv + tracking
    width = max(x - tracking, 1.0)
    hh = max(p[0].shape[0] for p in parts)
    base = max(p[1] for p in parts)
    W = int(math.ceil(width)) + 8
    cov = np.zeros((hh + 4, W), np.float32)
    for fill, b, x0, at in parts:
        dx, dy = int(round(at)) + 4 - x0, base - b
        h, w = fill.shape
        xs, ys = max(dx, 0), max(dy, 0)
        xe, ye = min(dx + w, W), min(dy + h, cov.shape[0])
        if xe > xs and ye > ys:
            cov[ys:ye, xs:xe] = np.maximum(cov[ys:ye, xs:xe], fill[ys - dy:ye - dy, xs - dx:xe - dx])
    return cov, base, 4, width


class LabelFx:
    """The labels' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t, _point_of
    and the fonts the picture effects found)."""

    def _label_init(self):
        """The labels' fonts, found now: a missing one is an FX.FxError before any frame is drawn."""
        self._lb_fonts = {fx["index"]: self._font_of(fx["font"]) for fx in self.fxs if fx["kind"] == "label"}

    def _label(self, fx, frame, img):
        out = np.array(img, np.float32, copy=True)
        for cov, colour, x0, y0, alpha in self._label_inks(fx, frame):
            _ink(out, cov, colour, x0, y0, alpha)
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _label_inks(self, fx, frame):
        """What a label entry lays on `frame`, in order: [(coverage (h, w) float32 0..1, colour, x0, y0 (its top left,
        px), alpha)]: each label's box, its text typed so far, its leader."""
        t = self._t(frame)
        w, h = self.size
        font = self._lb_fonts[fx["index"]]
        inks = []
        for j, lb in enumerate(fx["labels"]):
            n = LB.shown(lb, t, fx["type"])
            if n == 0:
                continue
            if lb["at"]["mode"] == "frame":
                p = (lb["at"]["at"][0] * w - 0.5, lb["at"]["at"][1] * h - 0.5)
            else:
                p = self._point_of(LB.point_key(fx, j), frame)[0]
            q, side = LB.layout(lb, p, self.size, h)
            cap = lb["size"] * h
            em = PFX._em(font, cap)
            track = fx["tracking"] * self.px
            width = set_text(lb["text"], font, em, track)[3]                   # the whole label: its place never moves
            cov, base, ix, _ = set_text(lb["text"][:n], font, em, track)
            left = q[0] if side == "left" else q[0] - width
            edge = LB.EDGE * h                                                 # kept inside the frame, whatever the output
            left = min(max(left, edge), w - edge - width)
            q = (left if side == "left" else left + width, min(max(q[1], edge + cap), h - edge))
            x0, y0 = int(round(left - ix)), int(round(q[1] - base))
            if fx["box"]:
                b = fx["box"]
                pad = b["pad"] * self.px
                bx0, by0 = int(round(left - pad)), int(round(q[1] - cap - pad))
                bx1, by1 = int(round(left + width + pad)), int(round(q[1] + pad + 0.25 * cap))
                if bx1 > bx0 and by1 > by0:
                    inks.append((np.ones((by1 - by0, bx1 - bx0), np.float32), b["color"], bx0, by0, b["alpha"]))
            inks.append((cov, lb["color"], x0, y0, 1.0))
            if fx["leader"]:
                inks += self._leader(fx, lb, p, q, side, cap, t)
        return inks

    def _leader(self, fx, lb, p, q, side, cap, t):
        """The dotted leader from the label's side (`q` is where its text starts, or ends with side "right") to its
        point `p`, drawn out as far as `LB.drawn` says, the dot at the point once it gets there: [(coverage, colour, x0,
        y0, alpha)], none before it starts."""
        u = LB.drawn(lb, t, fx["type"], fx["draw"])
        if u <= 0:
            return []
        ld = fx["leader"]
        gap0 = 0.6 * cap
        a = np.array([q[0] - gap0 if side == "left" else q[0] + gap0, q[1] - 0.45 * cap])
        b = np.asarray(p, np.float64)
        d = b - a
        L = float(np.linalg.norm(d))
        end = ld["end"] * self.px
        stop = max(L - end - 2.0 * self.px, 0.0)
        r = 0.5 * ld["dot"] * self.px
        pitch = max(ld["dot"] + ld["gap"], 1.0) * self.px
        n = int(stop * u / pitch) + 1 if stop > 0 else 0
        pts = [a + d / max(L, 1e-9) * (k * pitch) for k in range(n)]
        pts_end = [b] if u >= 1.0 and end > 0 else []
        if not pts + pts_end:
            return []
        P = np.asarray(pts + pts_end)
        x0, y0 = np.floor(P.min(0) - max(r, end) - 3).astype(int)
        x1, y1 = np.ceil(P.max(0) + max(r, end) + 3).astype(int)
        cov = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
        for c in pts:
            cv2.circle(cov, (int(round((c[0] - x0) * 16)), int(round((c[1] - y0) * 16))), max(int(round(r * 16)), 1),
                       255, -1, cv2.LINE_AA, 4)
        for c in pts_end:
            cv2.circle(cov, (int(round((c[0] - x0) * 16)), int(round((c[1] - y0) * 16))), max(int(round(end * 16)), 1),
                       255, -1, cv2.LINE_AA, 4)
        return [(cov.astype(np.float32) / 255.0, ld["color"], int(x0), int(y0), 1.0)]
