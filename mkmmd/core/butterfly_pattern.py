"""Wing patterns, bpy-free (the library props `butterfly` and `butterflies`): every style's upper side and underside drawn
as masks in the style's atlas (mkmmd.core.butterfly.atlas: the forewing's tiles left, the hindwing's right, the upper
side on top, the underside below), which the material turns into colours from the colour roles. Four channels, laid
over the ground colour in this order:

    sheen    where the iridescence lies (the sheen colour at a glancing view; the material's `sheen` scales it)
    accent   coloured spots: eyespot rings, lunules, red spots
    ink      the dark pattern: veins, borders, stripes, eye rings, the dusky base
    spot     the light pattern: edge dots, fringes, pale spots, pupils

Painting a channel clears the ones above it there (an orange spot in a black border shows orange). Everything is drawn
from the wing's own anatomy, not from a picture: the venation (a discal cell near the middle of the wing, veins fanning
from it and from the base to the margin), bands along the edges by the distance to the outline, dots along the edges by
the arc length, eyespots at points on a curve inside the margin. Deterministic: no randomness at all.

    masks(style, tile=512)  ->  (2 tile, 2 tile, 4) float32 in 0..1, rows from the top (the head's side) down
    cached_masks(style, tile, folder)   the same, kept on disk under a hash of this maths"""
import hashlib
import math
import os
import sys

import numpy as np

from . import butterfly as BF

CHANNELS = ("ink", "spot", "accent", "sheen")            # the atlas' R, G, B, A
STACK = ("sheen", "accent", "ink", "spot")              # bottom to top over the ground
BAND = 0.085                                            # distances to the outline are exact this far from it (span)
TILE = 512

# per style and wing: the landmark of the tip, the visible outer edge (first, last landmark) and the hind edge's start
ANCHORS = {
    "swallowtail": {"fore": {"apex": 7, "outer": (7, 11), "inner": 11},
                    "hind": {"apex": 11, "outer": (5, 19), "inner": 19, "tail": (9, 15), "anal": 18}},
    "monarch": {"fore": {"apex": 8, "outer": (8, 14), "inner": 14},
                "hind": {"apex": 6, "outer": (5, 13), "inner": 13}},
    "morpho": {"fore": {"apex": 8, "outer": (8, 13), "inner": 13},
               "hind": {"apex": 6, "outer": (5, 13), "inner": 13}},
    "blue": {"fore": {"apex": 6, "outer": (6, 11), "inner": 11},
             "hind": {"apex": 6, "outer": (4, 12), "inner": 12}},
}


def _cov(d, px):
    """Anti-aliased coverage of the inside of an edge at signed distance d (positive inside), pixel size px."""
    return np.clip(d / px + 0.5, 0.0, 1.0)


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bezier(p0, p1, p2, n=24):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


class Tile:
    """One wing's canvas: pixel centres X, Y in the body's frame (unit span), the outline, the inside, the distance to the
    outline `d` (exact up to BAND) with the arc parameter `s` of the nearest outline point (0 the root's front corner, 1 the
    rear corner along the costa, the tip and the outer and hind edges; 2 on the straight root edge; -1 far from the
    edge), and the four channels."""

    def __init__(self, style, wing, tile=TILE):
        self.style, self.wing, self.n = style, wing, tile
        x0, y0, size = BF.atlas(style)[wing]
        self.px = size / tile
        c = (np.arange(tile) + 0.5) * self.px
        self.X, self.Y = np.meshgrid(x0 + c, y0 + c)
        self.x0, self.y0 = x0, y0
        self.P = BF.outline(style, wing, 240)
        self.L = BF.landmarks(style)[wing]
        self.ls = BF.landmark_s(style, wing)
        self.A = ANCHORS[style][wing]
        self.B = 0.5 * (self.L[0] + self.L[-1])                      # the wing's base
        self.ch = {k: np.zeros((tile, tile), np.float32) for k in CHANNELS}
        self._edge()
        self.inside = self._fill(self.P)
        self.sd = np.where(self.inside, self.d, -self.d)          # signed: positive inside the wing
        self.veins = []                                           # [(polyline, width)], set by `venation`

    # ---- grid helpers
    def _window(self, lo, hi):
        """Row and column slices of the pixels whose centres lie in the box [lo, hi]."""
        n = self.n
        j0 = int(np.clip(math.floor((lo[0] - self.x0) / self.px - 0.5), 0, n))
        j1 = int(np.clip(math.ceil((hi[0] - self.x0) / self.px + 0.5), 0, n))
        i0 = int(np.clip(math.floor((lo[1] - self.y0) / self.px - 0.5), 0, n))
        i1 = int(np.clip(math.ceil((hi[1] - self.y0) / self.px + 0.5), 0, n))
        return slice(i0, i1), slice(j0, j1)

    def _edge(self):
        n = self.n
        self.d = np.full((n, n), BAND, np.float64)
        self.s = np.full((n, n), -1.0)
        P = self.P
        m = len(P)
        sv = np.concatenate([np.linspace(0.0, 1.0, m - 1), [2.0]])          # the root edge's middle: s = 2
        for k in range(m):
            a, b = P[k], P[(k + 1) % m]
            sa, sb = sv[k], sv[(k + 1) % m]
            if k >= m - 2:
                sa = sb = 2.0                                                # the straight root edge
            rs, cs = self._window(np.minimum(a, b) - BAND, np.maximum(a, b) + BAND)
            if rs.start >= rs.stop or cs.start >= cs.stop:
                continue
            X, Y = self.X[rs, cs], self.Y[rs, cs]
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
            dd = np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1])
            cur = self.d[rs, cs]
            better = dd < cur
            cur[better] = dd[better]
            ss = self.s[rs, cs]
            ss[better] = (sa + t * (sb - sa))[better]

    def _fill(self, poly):
        """Inside the closed polygon (pixel centres; even-odd, by scanlines)."""
        poly = np.asarray(poly, float)
        a, b = poly, np.roll(poly, -1, axis=0)
        out = np.zeros((self.n, self.n), bool)
        xs = self.X[0]
        for i, y in enumerate(self.Y[:, 0]):
            hit = (a[:, 1] <= y) != (b[:, 1] <= y)
            if not hit.any():
                continue
            aa, bb = a[hit], b[hit]
            x = aa[:, 0] + (y - aa[:, 1]) * (bb[:, 0] - aa[:, 0]) / (bb[:, 1] - aa[:, 1])
            out[i] = (np.searchsorted(np.sort(x), xs) % 2) == 1
        return out

    # ---- anatomy
    def s_of(self, k):
        """The arc parameter of landmark k."""
        return float(self.ls[k])

    def at_s(self, s):
        """The outline point at arc parameter s (0..1)."""
        m = len(self.P) - 1
        f = np.clip(np.asarray(s, float), 0.0, 1.0) * (m - 1)
        i = np.minimum(np.floor(f).astype(int), m - 2)
        t = (f - i)[..., None]
        return self.P[i] * (1 - t) + self.P[i + 1] * t

    def inward(self, s):
        """The unit inward normal of the outline at arc parameter s."""
        e = 1e-3
        t = self.at_s(np.asarray(s) + e) - self.at_s(np.asarray(s) - e)
        t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
        return np.stack([-t[..., 1], t[..., 0]], -1)

    def inset(self, s, depth):
        """The point `depth` in from the outline at arc parameter s."""
        return self.at_s(s) + depth * self.inward(s)

    # ---- painting
    def paint(self, ch, cov, a=1.0, where=None):
        """Lay `cov` (coverage, full array or (array, (rows, cols)) of a window) at strength `a` into channel `ch` and clear
        the channels above it in STACK there."""
        if isinstance(cov, tuple):
            cov, (rs, cs) = cov
        else:
            rs, cs = slice(None), slice(None)
        cov = np.asarray(cov, np.float32) * a
        if where is not None:
            cov = cov * where[rs, cs]
        dst = self.ch[ch][rs, cs]
        np.maximum(dst, cov, out=dst)
        for above in STACK[STACK.index(ch) + 1:]:
            self.ch[above][rs, cs] *= 1.0 - cov

    def disc(self, c, r, soft=0.0):
        """(coverage, window) of a disc; `soft` widens its edge to a smooth fall-off that wide."""
        c = np.asarray(c, float)
        rs, cs = self._window(c - r - soft - 2 * self.px, c + r + soft + 2 * self.px)
        d = r - np.hypot(self.X[rs, cs] - c[0], self.Y[rs, cs] - c[1])
        cov = _smooth(-0.5 * soft, 0.5 * soft, d) if soft > 0 else _cov(d, self.px)
        return cov, (rs, cs)

    def ellipse(self, c, rx, ry, angle=0.0, soft=0.0):
        """(coverage, window) of an ellipse turned `angle` radians (an approximate signed distance)."""
        c = np.asarray(c, float)
        R = max(rx, ry) + soft + 2 * self.px
        rs, cs = self._window(c - R, c + R)
        dx, dy = self.X[rs, cs] - c[0], self.Y[rs, cs] - c[1]
        ca, sa = math.cos(angle), math.sin(angle)
        u, v = ca * dx + sa * dy, -sa * dx + ca * dy
        q = np.hypot(u / rx, v / ry)
        d = (1.0 - q) * min(rx, ry)
        cov = _smooth(-0.5 * soft, 0.5 * soft, d) if soft > 0 else _cov(d, self.px)
        return cov, (rs, cs)

    def line(self, pts, w0, w1=None):
        """(coverage, window) of a polyline drawn `w0` wide at its start, tapering to `w1` at its end."""
        pts = np.asarray(pts, float)
        w1 = w0 if w1 is None else w1
        ws = np.linspace(w0, w1, len(pts))
        pad = max(w0, w1) + 2 * self.px
        rs, cs = self._window(pts.min(0) - pad, pts.max(0) + pad)
        best = np.full((rs.stop - rs.start, cs.stop - cs.start), -1e9)
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            r2, c2 = self._window(np.minimum(a, b) - pad, np.maximum(a, b) + pad)
            if r2.start >= r2.stop or c2.start >= c2.stop:
                continue
            X, Y = self.X[r2, c2], self.Y[r2, c2]
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
            d = 0.5 * (ws[k] + t * (ws[k + 1] - ws[k])) - np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1])
            sub = best[r2.start - rs.start:r2.stop - rs.start, c2.start - cs.start:c2.stop - cs.start]
            np.maximum(sub, d, out=sub)
        return _cov(best, self.px), (rs, cs)

    def edge_band(self, w, s0=0.0, s1=1.0, fade=0.03, soft=None):
        """Coverage of the band within `w` of the outline (w may be a function of s) between arc parameters s0 and s1, its
        ends fading over `fade` of the arc, its inner edge soft over `soft` (default a pixel)."""
        s = self.s
        width = w(s) if callable(w) else w
        along = _smooth(s0 - fade, s0 + fade, s) * (1.0 - _smooth(s1 - fade, s1 + fade, s))
        along = np.where((s >= 0) & (s <= 1), along, 0.0)
        inner = _smooth(-0.5 * soft, 0.5 * soft, width - self.d) if soft else _cov(width - self.d, self.px)
        return inner * along

    def edge_line(self, depth, w, s0, s1, waves=0, amp=0.0, fade=0.02):
        """Coverage of a line `w` wide running `depth` in from the outline from s0 to s1, waving `waves` times by `amp`."""
        s = self.s
        along = _smooth(s0 - fade, s0 + fade, s) * (1.0 - _smooth(s1 - fade, s1 + fade, s))
        along = np.where((s >= 0) & (s <= 1), along, 0.0)
        dd = depth + amp * np.sin(2.0 * math.pi * waves * (s - s0) / max(s1 - s0, 1e-9))
        return _cov(0.5 * w - np.abs(self.d - dd), self.px) * along

    def edge_dots(self, depth, r, spacing, s0, s1, phase=0.5, squash=1.0):
        """Coverage of dots `depth` in from the outline every `spacing` (span units) along it from s0 to s1; `squash` < 1
        flattens them along the edge into dashes (> 1: into crescents' long axis)."""
        length = self._arc_length()
        s = self.s
        ok = (s >= s0) & (s <= s1)
        u = (s - s0) * length / spacing + phase
        along = (u - np.floor(u) - 0.5) * spacing
        across = self.d - depth
        q = np.hypot(along * squash, across)
        return np.where(ok, _cov(r - q, self.px), 0.0)

    def _arc_length(self):
        P = self.P[:-1]
        return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

    # ---- venation
    def venation(self, n_veins, cell=(0.5, 0.2), costa_share=0.3, w=0.004, taper=0.5, anal=1):
        """The veins as polylines (kept on the tile for later use) and the discal cell's outline: a cell from near the
        base toward the middle of the outer edge (`cell` = (length as a share of the base-to-edge distance, half width as a
        share of its length)), `n_veins` veins fanning from the cell's front, end and rear to the edge (the first
        `costa_share` of them ending on the costa before the tip), a vein along the costa and `anal` along the hind
        edge."""
        A = self.A
        s_out0, s_out1 = self.s_of(A["outer"][0]), self.s_of(A["outer"][1])
        s_tip = self.s_of(A["apex"])
        mid = self.at_s(0.5 * (s_out0 + s_out1))
        B = self.B + np.array([0.012, 0.0])
        u = mid - B
        dist = float(np.linalg.norm(u))
        u /= dist
        v = np.array([u[1], -u[0]])                                    # toward the costa (the head's side)
        if float((self.at_s(0.5 * s_tip) - B) @ v) < 0:
            v = -v
        Lc, W = cell[0] * dist, cell[1] * cell[0] * dist
        tau = np.linspace(0.0, 1.0, 18)[:, None]
        front = B + u * (Lc * tau) + v * (W * np.sin(0.5 * math.pi * tau) ** 0.9)
        rear = B + u * (0.9 * Lc * tau) - v * (0.8 * W * np.sin(0.5 * math.pi * tau) ** 0.9)
        close = _bezier(front[-1], B + u * (0.86 * Lc) + 0.1 * W * v, rear[-1], 10)
        self.cell = np.vstack([front, close[1:-1], rear[::-1]])
        # the vein ends: some on the costa before the tip, the rest along the tip and outer edge
        n_c = round(n_veins * costa_share)
        ends = list(np.linspace(0.62 * s_tip, s_tip - 0.25 * (s_tip - 0.62 * s_tip), n_c)) if n_c else []
        ends += list(np.linspace(s_tip, s_out1 - 0.06 * (s_out1 - s_out0), n_veins - n_c))
        # their starts along the cell's boundary: the front edge's outer half, the closing vein, the rear's outer part
        bound = np.vstack([front[9:], close[1:-1], rear[::-1][:9]])
        bs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(bound, axis=0), axis=1))])
        starts = [np.array([np.interp(q, bs, bound[:, 0]), np.interp(q, bs, bound[:, 1])])
                  for q in np.linspace(0.0, bs[-1], len(ends))]
        veins = []
        for p0, se in zip(starts, ends):
            p2 = self.at_s(se) - 0.004 * self.inward(se)                # just past the edge: no gap at the margin
            out = p0 - B
            out /= max(float(np.linalg.norm(out)), 1e-9)
            p1 = p0 + out * 0.42 * float(np.linalg.norm(p2 - p0))
            veins.append(_bezier(p0, p1, p2, 20))
        # the costal vein, the base of the cell, the anal veins
        sc = np.linspace(0.03, 0.55 * s_tip, 16)
        veins.append(np.array([self.inset(q, 0.014) for q in sc]))
        s_in = self.s_of(A["inner"])
        for k in range(anal):
            q = np.linspace(s_in + 0.12 * (1 - s_in), 0.985, 14)
            veins.append(np.array([self.inset(x, 0.022 + 0.035 * k) for x in q])[::-1])
        self.veins = [(v_, w * (1.25 if i >= len(ends) else 1.0)) for i, v_ in enumerate(veins)]
        self.taper = taper
        return self

    def paint_veins(self, ch="ink", a=1.0, scale=1.0, cell=True):
        for pts, w in self.veins:
            self.paint(ch, self.line(pts, w * scale, w * scale * self.taper), a)
        if cell:
            ring = np.vstack([self.cell, self.cell[:1]])
            w = self.veins[0][1] * scale if self.veins else 0.003
            self.paint(ch, self.line(ring, w, w), a)

    def vein_distance(self, reach=0.04):
        """Distance from every pixel to the nearest vein (or the cell's outline), up to `reach`."""
        d = np.full((self.n, self.n), reach)
        for pts, _w in self.veins + [(np.vstack([self.cell, self.cell[:1]]), 0.0)]:
            for k in range(len(pts) - 1):
                a, b = pts[k], pts[k + 1]
                rs, cs = self._window(np.minimum(a, b) - reach, np.maximum(a, b) + reach)
                if rs.start >= rs.stop or cs.start >= cs.stop:
                    continue
                X, Y = self.X[rs, cs], self.Y[rs, cs]
                ab = b - a
                t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
                cur = d[rs, cs]
                np.minimum(cur, np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]), out=cur)
        return d

    def base_dusk(self, reach, a=0.3):
        """The dusky scales round the wing's base: ink fading out over `reach`."""
        r = np.hypot(self.X - self.B[0], self.Y - self.B[1])
        self.paint("ink", (1.0 - _smooth(0.0, reach, r)) * a)

    def eye(self, c, r, rings=("accent", "ink"), pupil=0.3, ratio=0.62, soft=0.0):
        """An eyespot: a ring of the first channel round a disc of the second, a pupil (spot) off-centre toward the
        light."""
        self.paint(rings[0], self.disc(c, r, soft))
        self.paint(rings[1], self.disc(c, r * ratio, soft * 0.5))
        if pupil > 0:
            self.paint("spot", self.disc(np.asarray(c) + np.array([-0.18, -0.22]) * r * ratio, r * ratio * pupil))

    def array(self):
        return np.stack([self.ch[k] for k in CHANNELS], -1)

    def clear(self):
        """Empty channels and no veins: the same wing, ready for its other side."""
        for v in self.ch.values():
            v[:] = 0.0
        self.veins = []
        return self


# =====================================================================================================================
# the styles: (upper side, underside) painters per wing
# =====================================================================================================================
def _monarch(t, under):
    fore = t.wing == "fore"
    A = t.A
    t.venation(10 if fore else 8, cell=(0.5, 0.2) if fore else (0.48, 0.24), w=0.0062 if fore else 0.0058,
               costa_share=0.3 if fore else 0.0, anal=1 if fore else 2)
    s0, s1 = t.s_of(A["outer"][0]), t.s_of(A["outer"][1])
    if fore:
        tip = t.at_s(t.s_of(A["apex"]))
        if not under:                                              # the black tip with its band of pale spots
            t.paint("ink", t.disc(tip + np.array([-0.05, 0.03]), 0.135, soft=0.03))
        else:
            t.paint("accent", t.disc(tip + np.array([-0.04, 0.02]), 0.11, soft=0.05), 0.55)
        t.paint("ink", t.edge_band(lambda s: 0.010 + 0.006 * (s < 0.5), 0.0, t.s_of(A["apex"]), fade=0.02))
    t.paint("ink", t.edge_band(0.034 if fore else 0.03, s0 - 0.04, s1 + 0.02, fade=0.02))
    t.paint_veins(scale=1.0)
    if under:                                                     # pale cells between the veins of the underside
        dv = t.vein_distance()
        t.paint("spot", _smooth(0.012, 0.03, dv) * t.inside, 0.18)
    t.paint("spot", t.edge_dots(0.009, 0.0042, 0.0165, s0, s1 - 0.01))
    t.paint("spot", t.edge_dots(0.0235, 0.0036, 0.0165, s0 + 0.01, s1 - 0.03, phase=0.0))
    if fore:                                                      # the subapical spots: pale orange and white
        tip_s = t.s_of(A["apex"])
        for k, (q, depth, r, ch) in enumerate(((0.80, 0.055, 0.0095, "accent"), (0.84, 0.065, 0.010, "accent"),
                                               (0.88, 0.075, 0.0095, "accent"), (0.93, 0.06, 0.0075, "spot"),
                                               (0.97, 0.05, 0.0065, "spot"), (1.0, 0.07, 0.006, "spot"))):
            c = t.inset(q * tip_s, depth)
            t.paint(ch, t.ellipse(c, r * 1.25, r, angle=0.4))
        t.paint("spot", t.disc(t.inset(0.62 * tip_s, 0.03), 0.006))
        t.paint("spot", t.disc(t.inset(0.66 * tip_s, 0.042), 0.0055))
    t.base_dusk(0.06, 0.35)


def _morpho(t, under):
    fore = t.wing == "fore"
    A = t.A
    s0, s1 = t.s_of(A["outer"][0]), t.s_of(A["outer"][1])
    tip_s = t.s_of(A["apex"])
    t.venation(10 if fore else 8, cell=(0.48, 0.18) if fore else (0.45, 0.22), w=0.0026,
               costa_share=0.3 if fore else 0.0, anal=1 if fore else 2)
    if not under:
        t.paint("sheen", np.ones((t.n, t.n), np.float32))
        if fore:                                                  # the broad black costa and tip with white spots
            t.paint("ink", t.edge_band(lambda s: 0.02 + 0.075 * _smooth(0.25, 0.95, s / tip_s), 0.0, tip_s + 0.02,
                                       fade=0.02, soft=0.008))
            t.paint("ink", t.edge_band(lambda s: 0.095 - 0.06 * _smooth(s0, s1, s), tip_s - 0.02, s1 + 0.02,
                                       fade=0.02, soft=0.01))
            for k, q in enumerate(np.linspace(0.62, 1.0, 6)):
                t.paint("spot", t.disc(t.inset(q * tip_s, 0.055 - 0.012 * (k % 2)), 0.0068 - 0.0004 * k))
            for q in np.linspace(tip_s + 0.01, s1 - 0.02, 6):
                t.paint("spot", t.disc(t.inset(q, 0.03), 0.0052))
        else:
            t.paint("ink", t.edge_band(lambda s: 0.055 + 0.015 * _smooth(s0, s1, s), s0 - 0.05, s1 + 0.02, fade=0.02,
                                       soft=0.01))
            t.paint("ink", t.edge_band(0.03, s1, 1.0, fade=0.02, soft=0.008))
            t.paint("accent", t.edge_dots(0.03, 0.0055, 0.034, s0 + 0.03, s1 - 0.02))
        t.paint_veins(scale=1.0, a=0.85)
        t.paint("spot", t.edge_dots(0.0035, 0.003, 0.02, s0, s1))
        t.base_dusk(0.08, 0.3)
        return
    # the underside: brown, eyespots in a row, pale wavy lines, white edge dots
    t.paint_veins(scale=0.8, a=0.5)
    for depth, waves in ((0.042, 9), (0.074, 7)):                  # two pale wavy lines, each shadowed inside
        t.paint("spot", t.edge_line(depth, 0.006, s0 - 0.03, s1 + 0.01, waves, 0.006), 0.55)
        t.paint("ink", t.edge_line(depth + 0.007, 0.006, s0 - 0.03, s1 + 0.01, waves, 0.006), 0.3)
    n_eye = 3 if fore else 4
    lo, hi = (0.25, 0.85) if fore else (0.12, 0.8)
    for k, q in enumerate(np.linspace(lo, hi, n_eye)):
        s = s0 + q * (s1 - s0)
        r = (0.028 if fore else 0.03) * (1.0 - 0.15 * abs(q - 0.5))
        t.eye(t.inset(s, 0.105 if fore else 0.1), r, ratio=0.66, pupil=0.32)
    t.paint("ink", t.edge_band(0.012, s0, s1, fade=0.02), 0.5)
    t.paint("spot", t.edge_dots(0.0035, 0.0032, 0.02, s0, s1))
    t.base_dusk(0.07, 0.25)


def _swallowtail(t, under):
    fore = t.wing == "fore"
    A = t.A
    s0, s1 = t.s_of(A["outer"][0]), t.s_of(A["outer"][1])
    tip_s = t.s_of(A["apex"])
    t.venation(10 if fore else 8, cell=(0.5, 0.18) if fore else (0.5, 0.2), w=0.0034,
               costa_share=0.3 if fore else 0.0, anal=1)
    B = t.B
    if fore:                                                      # four tiger stripes from the costa, the last ones short
        for q, reach, w in ((0.1, 1.0, 0.026), (0.27, 1.0, 0.03), (0.46, 0.62, 0.034), (0.66, 0.4, 0.03)):
            p0 = t.inset(q * tip_s, -0.01)
            back = t.at_s(t.s_of(A["inner"]) + (1.0 - t.s_of(A["inner"])) * (0.05 + 0.85 * (1 - q)))
            p2 = p0 + (back - p0) * reach
            t.paint("ink", t.line(_bezier(p0, 0.5 * (p0 + p2) + np.array([0.012, 0.0]), p2, 16), w, w * 0.55))
        t.paint("ink", t.edge_band(0.012, 0.0, tip_s, fade=0.02))
        t.paint("ink", t.edge_band(0.05, tip_s - 0.05, s1 + 0.02, fade=0.03, soft=0.008))
        t.paint("ink", t.edge_band(0.016, s1, 1.0, fade=0.03), 0.8)
        t.paint("spot", t.edge_dots(0.03, 0.0062, 0.026, tip_s, s1 - 0.01, squash=0.8))
        t.paint("spot", t.edge_dots(0.009, 0.0036, 0.026, tip_s - 0.01, s1, phase=0.0))
    else:
        tail0, tail1 = t.s_of(A["tail"][0]), t.s_of(A["tail"][1])
        t.paint("ink", t.edge_band(0.066, s0 - 0.04, s1 + 0.01, fade=0.02, soft=0.01))
        t.paint("ink", t.edge_band(0.05, tail0 - 0.01, tail1 + 0.01, fade=0.005))     # the tail is black
        t.paint("ink", t.edge_band(0.036, s1 - 0.01, 0.99, fade=0.02, soft=0.006))    # the stripe along the body
        t.paint("sheen", t.edge_dots(0.036, 0.0105, 0.03, s0 + 0.02, tail0 - 0.01, squash=0.75), 0.9)
        t.paint("sheen", t.edge_dots(0.036, 0.0105, 0.03, tail1 + 0.01, s1 - 0.03, squash=0.75), 0.9)
        t.paint("spot", t.edge_dots(0.0085, 0.0075, 0.03, s0 + 0.01, tail0, squash=1.6))
        t.paint("spot", t.edge_dots(0.0085, 0.0075, 0.03, tail1, s1 - 0.02, squash=1.6))
        anal = t.at_s(t.s_of(A["anal"]))
        c = anal + 0.03 * t.inward(t.s_of(A["anal"]))
        t.paint("accent", t.disc(c, 0.016))
        t.paint("ink", t.disc(c + np.array([0.0, 0.006]), 0.0075))
        if under:
            t.paint("accent", t.edge_dots(0.05, 0.0085, 0.03, s0 + 0.03, tail0 - 0.02, squash=1.3), 0.85)
    t.paint_veins(scale=1.0)
    t.base_dusk(0.05, 0.28 if not under else 0.15)
    del B


def _blue(t, under):
    fore = t.wing == "fore"
    A = t.A
    s0, s1 = t.s_of(A["outer"][0]), t.s_of(A["outer"][1])
    tip_s = t.s_of(A["apex"])
    t.venation(9 if fore else 8, cell=(0.45, 0.2), w=0.0018, costa_share=0.3 if fore else 0.0,
               anal=1 if fore else 2)
    if not under:
        t.paint("sheen", np.ones((t.n, t.n), np.float32), 0.85)
        t.paint_veins(scale=0.9, a=0.45, cell=False)
        t.paint("ink", t.edge_band(0.011, s0 - 0.08, s1 + 0.05, fade=0.03))
        t.paint("spot", t.edge_band(0.0048, s0 - 0.12, s1 + 0.06, fade=0.03))
        t.base_dusk(0.07, 0.35)
        return
    t.paint("sheen", t.disc(t.B, 0.1, soft=0.08), 0.8)            # the blue flush at the base
    t.paint_veins(scale=0.8, a=0.18, cell=False)
    n_dots = 6 if fore else 8
    for k, q in enumerate(np.linspace(0.06, 0.94, n_dots)):        # the curved row of black dots ringed in white
        s = (tip_s - 0.04) + q * (s1 - tip_s + 0.02) if fore else s0 + q * (s1 - s0)
        c = t.inset(s, 0.075 + 0.012 * math.sin(math.pi * q))
        r = 0.0095 if fore else 0.0085
        t.paint("spot", t.disc(c, r + 0.0045))
        t.paint("ink", t.disc(c, r))
    cell_end = t.cell[len(t.cell) // 2]
    t.paint("spot", t.ellipse(cell_end, 0.012, 0.007, angle=1.2))
    t.paint("ink", t.ellipse(cell_end, 0.0085, 0.0038, angle=1.2))
    for c in ([0.08, 0.02], [0.12, -0.01]) if fore else ([0.07, 0.07], [0.1, 0.03], [0.06, 0.12]):
        p = t.B + np.array(c)
        t.paint("spot", t.disc(p, 0.011))
        t.paint("ink", t.disc(p, 0.0068))
    lo = s0 + (0.02 if not fore else 0.35 * (s1 - s0))
    t.paint("accent", t.edge_dots(0.03, 0.0095, 0.033, lo, s1 - 0.02, squash=0.7), 0.95 if not fore else 0.6)
    t.paint("ink", t.edge_dots(0.0135, 0.0045, 0.033, lo, s1 - 0.02), 0.85)
    t.paint("spot", t.edge_dots(0.045, 0.0075, 0.033, lo, s1 - 0.02, squash=0.6), 0.7)
    t.paint("ink", t.edge_band(0.0085, s0 - 0.08, s1 + 0.05, fade=0.03), 0.8)
    t.paint("spot", t.edge_band(0.0045, s0 - 0.12, s1 + 0.06, fade=0.03))


PAINTERS = {"monarch": _monarch, "morpho": _morpho, "swallowtail": _swallowtail, "blue": _blue}


def masks(style, tile=TILE):
    """The style's atlas: (2 tile, 2 tile, 4) float32, channels CHANNELS, rows from the head's side down: the upper side's
    forewing and hindwing on top, the underside's below."""
    if style not in PAINTERS:
        raise ValueError(f"butterfly: no style {style!r} (have {', '.join(BF.STYLES)})")
    out = np.zeros((2 * tile, 2 * tile, 4), np.float32)
    for col, wing in enumerate(("fore", "hind")):
        t = Tile(style, wing, tile)
        for row, under in enumerate((False, True)):
            PAINTERS[style](t.clear(), under)
            out[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = t.array()
    return np.clip(out, 0.0, 1.0)


def masks_key(style, tile):
    """A key of what `masks(style, tile)` draws: the style, the tile and the source of this maths and of the outlines."""
    key = hashlib.sha256()
    for mod in (BF, sys.modules[__name__]):
        with open(mod.__file__, "rb") as fh:
            key.update(fh.read())
    key.update(f"{style}/{tile}".encode())
    return key.hexdigest()[:16]


def cached_masks(style, tile, folder):
    """`masks`, kept in `folder` as .npy under `masks_key` (a change to the outlines or the painters draws them again)."""
    path = os.path.join(folder, f"{style}_{tile}_{masks_key(style, tile)}.npy")
    if os.path.isfile(path):
        try:
            M = np.load(path)
            if M.shape == (2 * tile, 2 * tile, 4):
                return M
        except (OSError, ValueError):
            pass
    M = masks(style, tile)
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, M)
    os.replace(tmp, path)
    return M


def preview(style, tile=TILE, slots=None, way=0):
    """The atlas coloured as the material colours it, flat (no light, no sheen angle): (2 tile, 2 tile, 3) sRGB 0..1, for
    looking at a pattern without Blender."""
    M = masks(style, tile)
    R = BF.roles(style, slots, way)
    col = {k: np.array(BF.colour(R[k], slots)) for k in ("ground", "under", "ink", "spot", "accent", "sheen")}
    out = np.zeros(M.shape[:2] + (3,))
    out[:tile] = col["ground"]
    out[tile:] = col["under"]
    for ch in STACK:
        a = M[..., CHANNELS.index(ch)][..., None]
        c = col["sheen" if ch == "sheen" else ch]
        if ch == "sheen":
            a = a * 0.35
        out = out * (1 - a) + c * a
    return np.where(out <= 0.0031308, out * 12.92, 1.055 * np.power(np.clip(out, 0, 1), 1 / 2.4) - 0.055)
