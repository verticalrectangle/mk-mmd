"""Festival lasers on the finished pictures (docs/design.md: Frame effects, Lasers): the picture work of the `[[fx]]` kind
`lasers` that mkmmd.core.lasers checks and times. LaserFx is the part of mkmmd.framefx.FrameFx that draws it: each
fan's beams drawn as antialiased lines from its origin (in their colours, fading along their length), a glow round
them, a white-hot middle where they are brightest and the light the haze scatters round them (a slow smoke of value
noise), all of it light added to the picture (or screened) and kept off her or on her by a mask. Images are float RGB in
display space (0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from . import fields as FL
from .core import framefx as FX
from .core import lasers as LZ

SHIFT = 4                       # fixed-point bits of a beam's ends (sub-pixel lines)
PAD = 2                         # px of canvas round the frame a beam is drawn on (its edge then cut off)
GLOW_SIGMA = 8.0                # px at 1080 on the short side: the glow round a beam
HAZE_SIGMA = 30.0               # px at 1080: how far the haze scatters a beam's light
HAZE_SCALE = 0.25               # the haze is worked out at this share of the frame's size, then resized up
HAZE_MAX = 0.6                  # the haze's light runs softly up to this (where a fan's beams meet it would pile up)
HAZE_CELLS = 4.0                # the smoke's noise cells across the frame's short side
HAZE_DRIFT = 0.12               # noise cells a second the smoke drifts (and slices it changes through)
HAZE_FLOOR = 0.2                # the thinnest the smoke gets: its light times 0.2 .. 1
HOT = 0.15                      # how white a beam's middle runs where it is brightest
BURN = 0.6                      # an overexposed beam runs white by this much for each time over full its light is


def clip_segment(p, q, box):
    """The part of the segment `p` .. `q` (px) inside `box` (x0, y0, x1, y1), or None (Liang-Barsky)."""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    u0, u1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0.0:
            if qk < 0.0:
                return None
            continue
        r = qk / pk
        if pk < 0.0:
            u0 = max(u0, r)
        else:
            u1 = min(u1, r)
    if u0 > u1:
        return None
    return (p[0] + u0 * dx, p[1] + u0 * dy), (p[0] + u1 * dx, p[1] + u1 * dy)


def beam_mask(size, origin, angles, length, thick, margin=0.0):
    """uint8 (h, w): beams from `origin` (px, anywhere: off the frame too) at `angles` (degrees counter-clockwise on
    screen from the right), `length` px long and `thick` px wide, antialiased; where two cross the brighter counts.
    Each is clipped to the frame grown by `margin` px and drawn on a canvas PAD px larger all round (OpenCV dims a line
    where it clips one at the canvas's own edge)."""
    w, h = int(size[0]), int(size[1])
    pad = PAD + int(thick)
    m = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    box = (-margin, -margin, w - 1.0 + margin, h - 1.0 + margin)
    k = float(1 << SHIFT)
    for a in angles:
        r = math.radians(a)
        q = (origin[0] + length * math.cos(r), origin[1] - length * math.sin(r))
        seg = clip_segment((float(origin[0]), float(origin[1])), q, box)
        if seg is None:
            continue
        (ax, ay), (bx, by) = seg
        cv2.line(m, (int(round((ax + pad) * k)), int(round((ay + pad) * k))),
                 (int(round((bx + pad) * k)), int(round((by + pad) * k))), 255, int(thick), cv2.LINE_AA, SHIFT)
    return np.ascontiguousarray(m[pad:pad + h, pad:pad + w])


def smoke(size, t, seed):
    """(h, w) float32 HAZE_FLOOR .. 1 over a picture of `size` (w, h): the haze's density, value noise of HAZE_CELLS cells
    across the short side (its contrast stretched: thick and thin patches), drifting and changing slowly with the clip
    second `t`."""
    w, h = int(size[0]), int(size[1])
    cell = min(w, h) / HAZE_CELLS
    X = (np.arange(w, dtype=np.float32) / np.float32(cell) + np.float32(HAZE_DRIFT * t))[None, :]
    Y = (np.arange(h, dtype=np.float32) / np.float32(cell) - np.float32(0.5 * HAZE_DRIFT * t))[:, None]
    n = FL.noise(np.broadcast_to(X, (h, w)), np.broadcast_to(Y, (h, w)), HAZE_DRIFT * t, seed)
    n = np.clip((n - np.float32(0.25)) * np.float32(2.0), 0.0, 1.0)
    return np.float32(HAZE_FLOOR) + np.float32(1.0 - HAZE_FLOOR) * n


def light(fx, size, px, fans, t):
    """(h, w, 3) float32: the light of a lasers entry's `fans` [(origin px, [(angle, colour index)])] over a picture of
    `size` (w, h) (`px` its px per px at 1080 on the short side) at clip second `t`, at full brightness: each beam's line
    in its colour, fading by `fade` along its `length`, a white-hot middle, the glow round it (`glow`) and the haze's
    scatter (`haze`)."""
    w, h = int(size[0]), int(size[1])
    C = np.asarray(fx["colors"], np.float32)
    thick = max(1, int(round(fx["width"] * px)))
    length = fx["length"] * h
    sg = GLOW_SIGMA * px
    out = np.zeros((h, w, 3), np.float32)
    hot = np.zeros((h, w), np.float32)
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    for origin, beams in fans:
        groups = {}
        for a, ci in beams:
            groups.setdefault(int(ci) % len(C), []).append(a)
        fade = None
        if fx["fade"] > 0:
            d = np.hypot(xs[None, :] - np.float32(origin[0]), ys[:, None] - np.float32(origin[1]))
            fade = np.float32(1.0) - np.float32(fx["fade"]) * np.clip(d / np.float32(length), 0.0, 1.0)
        for ci, angles in groups.items():
            m = beam_mask(size, origin, angles, length, thick, 3.0 * sg).astype(np.float32) * np.float32(1.0 / 255.0)
            if fade is not None:
                m *= fade
            out += m[..., None] * C[ci]
            np.maximum(hot, m, out=hot)
    if not hot.any():
        return out
    total = out + np.float32(HOT) * hot[..., None]
    if fx["glow"] > 0:
        norm = math.sqrt(2.0 * math.pi) * sg / thick             # a thin line blurred keeps its peak: glow at the core
        total += cv2.GaussianBlur(out, (0, 0), sg) * np.float32(fx["glow"] * norm)
    if fx["haze"] > 0:
        sw, sh = max(int(round(w * HAZE_SCALE)), 1), max(int(round(h * HAZE_SCALE)), 1)
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        shz = HAZE_SIGMA * px * HAZE_SCALE
        norm = math.sqrt(2.0 * math.pi) * HAZE_SIGMA * px / thick      # the scatter's peak at a beam: `haze`
        small = cv2.GaussianBlur(small, (0, 0), shz) * (np.float32(fx["haze"] * norm) * smoke((sw, sh), t,
                                                                                               fx["seed"]))[..., None]
        small = np.float32(HAZE_MAX) * (np.float32(1.0) - np.exp(-small / np.float32(HAZE_MAX)))
        total += cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return total


def burn(lit):
    """`lit` (h, w, 3) float32 light with its brightest channel held to 1 and its hue kept, what it had past that turning
    it white (BURN): an overexposed beam's middle runs white while its edges and its glow keep their colour."""
    mx = lit.max(-1, keepdims=True)
    if float(mx.max()) <= 1.0:
        return lit
    k = np.maximum(mx, np.float32(1.0))
    col = lit / k
    return col + (np.float32(1.0) - col) * np.clip((k - np.float32(1.0)) * np.float32(BURN), 0.0, 1.0)


class LaserFx:
    """The lasers' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, _t, _point_of,
    _hit_times and _masked)."""

    def lasers_at(self, fx, frame):
        """(state, [(origin px, [(angle, colour index)])]) of a lasers entry on `frame`: its fans as mkmmd.core.lasers
        state and fan make them, on its hits paced to three a second (each acting from its frame on), each fan from its
        origin there."""
        t = self._t(frame)
        hits = LZ.paced([(FX.frame_of(h, self.fps, self.frame0) - self.frame0) / self.fps for h in self._hit_times(fx)])
        st = LZ.state(fx, t, hits)
        w, h = self.size
        fans = []
        for j, o in enumerate(fx["origin"]):
            if o["mode"] == "frame":
                p = (o["at"][0] * w - 0.5, o["at"][1] * h - 0.5)
            else:
                p = self._point_of(LZ.point_key(fx, j), frame)[0]
            fans.append((p, LZ.fan(fx, j, st)))
        return st, fans

    def _lasers(self, fx, frame, img):
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0:
            return img
        st, fans = self.lasers_at(fx, frame)
        lit = burn(light(fx, self.size, self.px, fans, t) * np.float32(s * st[3]))
        region = self._masked(fx, frame)
        if region is not None:
            lit *= region[..., None]
        if fx["blend"] == "screen":
            out = 1.0 - (1.0 - img) * (1.0 - lit)
        else:
            out = img + lit
        return np.clip(out, 0.0, 1.0).astype(np.float32)
