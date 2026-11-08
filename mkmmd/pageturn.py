"""The page turn drawn (docs/design.md: Transitions and inserts), the post layer (numpy, OpenCV): the outgoing frame as a
page that curls over a travelling roll onto the picture under it. mkmmd.core.pageturn checks it and says where the
curl's line is on each window frame; `Turn` works out where the page lies on one of them, once, and turns the picture and
what goes with it. A turn at a cut is a cut effect: mkmmd.cutfx hands each window frame here with the incoming shot's
plate (`frame`). The loop's wrap is not: mk post and mk look turn each frame of its window as finished (its frame
effects and screen type on) over frame 0 as finished, held still (`wrap`), and the grade runs on the result; the screen
type laid over the graded picture and `[post] keep`'s mask turn with the page (`Turn.layer`, `Turn.mask`).

Every pixel is worked out exactly from the roll (mkmmd.core.pageturn): the page's front (the outgoing picture) flat and
then up the inside of the roll, shaded darker the deeper in it is; over it the page's back, round the top of the roll and
laid turned over beyond it: the paper colour with the picture showing faintly through it (the same point of the page
seen from behind, so mirrored), a fine paper grain, lit as a cylinder from the upper left (brightest on the roll's top,
dim on its far side, a soft sheen along it); under both the incoming picture. The page's edges are antialiased over a
pixel (widened where the roll is seen at a slant), the roll's far side too. The lifted page (the roll and the turned-over
part) casts a soft shadow to the lower right on what is under it: the incoming picture and the flat page.

Only the strip the page's geometry touches is worked out pixel by pixel, in slabs cut to it (`_strip`): from the
turned-over part's far edge to past the roll; on either side of it the frame is the flat page or the incoming picture."""
import math
from functools import lru_cache

import cv2
import numpy as np

from . import matte as M
from .core import pageturn as PG

LIGHT = np.float32([-0.45, -0.55, 0.70]) / np.float32(math.sqrt(0.45 ** 2 + 0.55 ** 2 + 0.70 ** 2))   # x right, y down
HALF = (LIGHT + np.float32([0.0, 0.0, 1.0])) / np.float32(np.linalg.norm(LIGHT + np.float32([0.0, 0.0, 1.0])))
AMBIENT = 0.45                  # the light a face turned from the lamp still gets (a flat page gets all of it: 1)
INSIDE = 0.3                    # the inside of the roll this much darker where it is deepest
SHEEN = (24.0, 0.22)            # the roll's sheen: its exponent and strength (none on the flat page)
GRAIN = 0.035                   # the paper's grain on the back, +-
SHADOW = (0.45, 0.3)            # the shadow's blur and its offset away from the light, in roll radii
SHADOW_STEP = 4                 # the shadow is worked out at a quarter of the frame's size (it is soft)
SLABS = 24                      # the strip is worked on in this many slabs, each cut to the part of it they hold


def frame(tr, frame, main, incoming):
    """The outgoing frame `main` (float RGB) turned as a page at `frame` of the transition `tr`'s window, over the
    incoming picture `incoming` (the frame's size). `main` itself before the page has lifted anywhere in the frame."""
    return Turn(tr, frame, (main.shape[1], main.shape[0])).picture(main, incoming)


def wrap(tr, frame, page, under):
    """The loop's wrap on `frame` of its window (`tr`: the page turn at the clip's end, mkmmd.core.transition.wrap_of).
    `page` is the outgoing frame and `under` frame 0, each as finished before the grade, as (picture, layer, mask,
    negative): the picture (float RGB) with its cut effects, frame effects, lens streaks and the screen type graded with
    it on it, the screen type still to be laid over the graded picture (straight RGBA), `[post] keep`'s mask and the
    negative layer still to invert the graded picture with (straight RGBA, its alpha the coverage), any of the last three
    None for none. Returns the frame's own four: the picture turned as a page over frame 0's, held still, and the layers
    and the mask turned with it."""
    t = Turn(tr, frame, (page[0].shape[1], page[0].shape[0]))
    return (t.picture(page[0], under[0]), t.layer(page[1], under[1]), t.mask(page[2], under[2]),
            t.layer(page[3], under[3]))


class Turn:
    """Where the page of the transition `tr` lies on `frame` of its window over pictures of `size` (w, h): the strip the
    roll works on, cut into parts, each with the point of the page its pixels see (its front flat or up the inside of the
    roll, its back round the roll's top or turned over flat beyond it) and how much of them the page covers there, and
    the shadow the lifted page casts. `lifted` is False before the page has lifted anywhere: the frame is all page."""

    def __init__(self, tr, frame, size):
        w, h = int(size[0]), int(size[1])
        self.tr, self.size = tr, (w, h)
        n = PG.normal(tr)
        R = PG.roll(tr, (w, h))
        c = PG.line(tr, frame - tr["first"], (w, h))
        c0, _ = PG.travel(tr, (w, h))
        self.lifted = c < c0
        if not self.lifted:
            return
        lo, hi = -(max(c0 - c - math.pi * R, 0.0) + 2.0), R + 2.0       # the strip: the turned-over part's far edge .. the roll
        self.parts = [_part(box, n, c, R, (w, h)) for box in _strip(n, c, lo, hi, w, h)]
        caster = np.zeros((h, w), np.float32)
        for p in self.parts:
            y0, y1, x0, x1 = p["box"]
            caster[y0:y1, x0:x1] = np.maximum(p["cov_up"], np.where(p["lifted"], p["cov_lo"], 0.0))
        self.keep = (1.0 - np.float32(tr["shadow"]) * _shadow(caster, R))[..., None]    # (h, w, 1): the light left
        xs, ys = M.grid(h, w)
        self.beyond = (xs * np.float32(n[0]) + ys * np.float32(n[1]) - np.float32(c) > np.float32(hi))[..., None]

    def picture(self, main, incoming):
        """The outgoing picture `main` (float RGB) turned as the page over `incoming`, the picture under it."""
        if not self.lifted:
            return main
        w, h = self.size
        main = np.ascontiguousarray(main, np.float32)
        incoming = np.ascontiguousarray(incoming, np.float32)
        out = np.where(self.beyond, incoming, main) * self.keep                   # past the roll / the flat page
        paper, k = np.asarray(self.tr["back"], np.float32), np.float32(self.tr["through"])
        for p in self.parts:
            y0, y1, x0, x1 = p["box"]
            # the front, flat or up the inside of the roll (dimmer the deeper in it is)
            front = cv2.remap(main, p["lx"], p["ly"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            front *= p["b_lo"][..., None]
            # the back: the paper, the picture faintly through it, its grain, lit as the roll's outside (flat: as it is)
            seen = cv2.remap(main, p["ux"], p["uy"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            grain = cv2.remap(_grain(h, w), p["ux"], p["uy"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            back = paper * (1.0 - k + k * seen) * (1.0 + grain)[..., None]
            back = np.clip(back * p["b_up"][..., None] + p["sheen"][..., None], 0.0, 1.0)
            s = self.keep[y0:y1, x0:x1]
            o = incoming[y0:y1, x0:x1] * s
            o += (front * s - o) * p["cov_lo"][..., None]
            o += (back - o) * p["cov_up"][..., None]
            out[y0:y1, x0:x1] = o
        return out

    def layer(self, main, incoming):
        """What is laid over the picture after the grade (screen type: straight RGBA, float (h, w, 4)) turned with the
        page: the outgoing frame's `main` printed on its front, moving with it and shaded as it is, the incoming one's on
        what lies under it, nothing on its back (its paper hides both); under the shadow darker, not fainter. Either may
        be None for none; None when both are."""
        if main is None and incoming is None:
            return None
        if not self.lifted:
            return main
        w, h = self.size
        main, incoming = (np.zeros((h, w, 4), np.float32) if x is None else np.ascontiguousarray(x, np.float32)
                          for x in (main, incoming))
        lit = np.float32([1.0, 1.0, 1.0, 0.0])
        out = np.where(self.beyond, incoming, main) * (1.0 + (self.keep - 1.0) * lit)     # as it is, the colour shaded
        pm, pi = _premultiplied(main), _premultiplied(incoming)
        for p in self.parts:
            y0, y1, x0, x1 = p["box"]
            o = self._printed(p, pm, pi, 3)
            a = o[..., 3:]
            out[y0:y1, x0:x1] = np.concatenate([np.where(a > 1e-6, o[..., :3] / np.maximum(a, np.float32(1e-6)), 0.0), a],
                                               axis=2)
        return np.clip(out, 0.0, 1.0)

    def mask(self, main, incoming):
        """A mask of the picture (`[post] keep`'s: (h, w), 0..1) turned with the page: the outgoing frame's `main` on its
        front, moving with it, the incoming one's on what lies under it, none on its back. Either may be None for none
        (all 0); None when both are."""
        if main is None and incoming is None:
            return None
        if not self.lifted:
            return main
        w, h = self.size
        main, incoming = (np.zeros((h, w, 1), np.float32) if m is None else
                          np.ascontiguousarray(m, np.float32).reshape(h, w, 1) for m in (main, incoming))
        out = np.where(self.beyond, incoming, main)[..., 0]
        for p in self.parts:
            y0, y1, x0, x1 = p["box"]
            out[y0:y1, x0:x1] = self._printed(p, main, incoming, 0)[..., 0]
        return out

    def _printed(self, p, main, incoming, lit):
        """Over part `p`'s box: `main` printed on the page's front, moving with it, and `incoming` on what lies under it
        ((h, w, c) each: premultiplied colour and alpha, or a mask), nothing on the page's back (its paper is the
        picture's). The first `lit` channels are shaded as the picture is (up the inside of the roll, under the shadow),
        the others (an alpha, a mask) are not."""
        y0, y1, x0, x1 = p["box"]
        c = main.shape[2]
        on = np.float32(np.arange(c) < lit)
        s = 1.0 + (self.keep[y0:y1, x0:x1] - 1.0) * on
        front = cv2.remap(main, p["lx"], p["ly"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        front = front.reshape(y1 - y0, x1 - x0, c) * (1.0 + (p["b_lo"][..., None] - 1.0) * on)
        o = incoming[y0:y1, x0:x1] * s
        o += (front * s - o) * p["cov_lo"][..., None]
        return o * (1.0 - p["cov_up"][..., None])


def _premultiplied(layer):
    """A straight RGBA layer (h, w, 4) with its colour multiplied by its alpha: what blends without fringes."""
    return np.ascontiguousarray(np.concatenate([layer[..., :3] * layer[..., 3:], layer[..., 3:]], axis=2))


def _part(box, n, c, R, size):
    """Where the page lies over `box` = (y0, y1, x0, x1) of a frame of `size` (w, h): {box, lifted (where the roll lifts
    it), lx, ly (the point of the page its front shows there: flat, or up the inside of the roll), cov_lo (how much of
    the pixel that covers), b_lo (the front's light: dimmer the deeper in the roll), ux, uy, cov_up, b_up (the same of
    its back, round the roll's top or turned over flat beyond it, lit as the roll's outside), sheen (the roll's)}."""
    y0, y1, x0, x1 = box
    w, h = size
    nx, ny = (np.float32(v) for v in n)
    xs, ys = M.grid(h, w)
    xs, ys = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
    xp = xs * nx + ys * ny                                                    # px along the normal
    yp = ys * nx - xs * ny                                                    # and along the line
    d = xp - np.float32(c)
    lifted = d > 0.0
    th = np.arcsin(np.clip(d / np.float32(R), 0.0, 1.0))                     # how far up the roll the inside seen here is
    # where on the page (along the normal) what is seen comes from: its front inside the roll (flat before the line),
    # its back on the roll's top or turned over flat beyond it
    lo = np.where(lifted, np.float32(c) + np.float32(R) * th, xp)
    up = np.where(lifted, np.float32(c) + np.float32(R) * (np.float32(math.pi) - th), np.float32(2.0 * c + math.pi * R) - xp)
    stretch = np.where(lifted, 1.0 / np.maximum(np.cos(th), 0.05), 1.0).astype(np.float32)
    far = np.clip(np.float32(R + 0.5) - d, 0.0, 1.0)                          # the roll's far side, seen edge on
    lx, ly = lo * nx - yp * ny, lo * ny + yp * nx
    ux, uy = up * nx - yp * ny, up * ny + yp * nx
    sn, cs = np.sin(th), np.cos(th)
    side = np.float32(n[0] * LIGHT[0] + n[1] * LIGHT[1])
    nh = np.maximum(sn * np.float32(n[0] * HALF[0] + n[1] * HALF[1]) + cs * HALF[2], 0.0)
    return {"box": box, "lifted": lifted, "lx": lx, "ly": ly, "ux": ux, "uy": uy,
            "cov_lo": np.where(lifted, _on_page(lx, ly, w, h, stretch) * far, 1.0).astype(np.float32),
            "cov_up": (_on_page(ux, uy, w, h, stretch) * far).astype(np.float32),
            "b_lo": (AMBIENT + (1.0 - AMBIENT) * np.maximum(cs * LIGHT[2] - sn * side, 0.0) / LIGHT[2]) * (1.0 - INSIDE * sn),
            "b_up": AMBIENT + (1.0 - AMBIENT) * np.maximum(cs * LIGHT[2] + sn * side, 0.0) / LIGHT[2],
            "sheen": np.float32(SHEEN[1]) * np.maximum(nh ** SHEEN[0] - np.float32(HALF[2] ** SHEEN[0]), 0.0)}


def _strip(n, c, lo, hi, w, h, slabs=SLABS):
    """[(y0, y1, x0, x1)]: parts of a frame w x h px holding every pixel whose place along the normal `n` less `c` lies in
    `lo` .. `hi`: the frame cut into slabs across the strip's run (rows for a strip nearer upright, else columns), each
    cut to where the strip crosses it, a pixel to spare."""
    nx, ny = n
    out = []
    rows = abs(nx) >= abs(ny)
    a, b = (nx, ny) if rows else (ny, nx)                                   # along the slab, across it
    length, extent = (h, w) if rows else (w, h)
    edges = np.linspace(0, length, min(slabs, length) + 1).round().astype(int).tolist()
    for s0, s1 in zip(edges[:-1], edges[1:]):
        if s1 <= s0:
            continue
        ends = [(c + dd - s * b) / a for dd in (lo, hi) for s in (s0 - 1.0, s1)]
        e0, e1 = max(0, int(math.floor(min(ends))) - 1), min(extent, int(math.ceil(max(ends))) + 2)
        if e1 > e0:
            out.append((s0, s1, e0, e1) if rows else (e0, e1, s0, s1))
    return out


def _on_page(x, y, w, h, stretch):
    """How much of a pixel the page covers when what it sees is the page's point (x, y) (index coordinates of the frame
    it was): 1 well inside, 0 off it, antialiased over the pixel at its edges (`stretch` page px to a frame px there)."""
    inside = np.minimum(np.minimum(x + 0.5, np.float32(w - 0.5) - x), np.minimum(y + 0.5, np.float32(h - 0.5) - y))
    return np.clip(inside / stretch + 0.5, 0.0, 1.0)


@lru_cache(maxsize=4)
def _grain(h, w, seed=3):
    """The paper's grain over a page of h x w px: fine noise, +-GRAIN."""
    g = np.random.default_rng(seed).standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    g = cv2.addWeighted(g, 0.7, cv2.GaussianBlur(g, (0, 0), 3.0), 2.0, 0.0)
    return np.ascontiguousarray(g / np.float32(max(float(np.abs(g).max()), 1e-6)) * np.float32(GRAIN))


def _shadow(caster, R):
    """The soft shadow of the coverage `caster` (h, w): blurred SHADOW[0] roll radii and moved SHADOW[1] of them away
    from the light, worked out at 1 / SHADOW_STEP of the size; 0 .. 1."""
    h, w = caster.shape
    if float(caster.max()) <= 0.0:
        return np.zeros_like(caster)
    s = SHADOW_STEP
    small = cv2.resize(caster, ((w + s - 1) // s, (h + s - 1) // s), interpolation=cv2.INTER_AREA)
    away = -LIGHT[:2] / np.float32(np.linalg.norm(LIGHT[:2]))
    off = SHADOW[1] * R / s
    small = cv2.warpAffine(small, np.float32([[1, 0, away[0] * off], [0, 1, away[1] * off]]), small.shape[::-1],
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    small = cv2.GaussianBlur(small, (0, 0), max(SHADOW[0] * R / s, 0.5))
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
