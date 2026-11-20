"""The peel drawn (docs/design.md: Transitions and inserts), the post layer (numpy, OpenCV): the outgoing picture as a
sticker pulled off the incoming one. mkmmd.core.peel checks it and says where the fold is on each window frame; `sheet`
works out, pixel by pixel, which points of a sticker each pixel sees (flat, or on any turn of the roll it is rolled up
into, in the camera's perspective: mkmmd.core.sheet3d) and `frame` lays the result on the incoming picture with the
shadow of what is lifted.

`sheet` is not tied to the whole frame: it peels any sticker given as a picture and its coverage in their flat place
(a figure sticker, a doodle), so everything peeled in a video peels alike. The roll's shape is the same all along the
fold, so it is worked out once a frame in 1D: its cross-section (mkmmd.core.peel.profile) seen in perspective, cut into
the runs that go one way across the screen, each matched to a fine grid of places along the normal. A pixel then
takes from every run over it, front to back (each turn's top over the turns inside it, those over the bottoms), until
it is covered: the picture where its printed face is seen (up the near side of the roll, darker as it turns away),
the paper back elsewhere (lit as a cylinder from the upper left, lightest on the crest with a soft sheen along it,
darker where it turns down to meet the sheet), antialiased at the roll's silhouettes. `Sheet.shadow` is the soft
shadow of the lifted parts, offset and blurred by their height (the light is the page turn's), and the roll's contact
shadow on what lies under it, either side of the fold."""
import math
from functools import lru_cache

import cv2
import numpy as np

from . import matte as M
from . import pageturn as PT
from .core import peel as PL
from .core import sheet3d as S3

AMBIENT = 0.5                   # the light a face turned from the lamp still gets (a flat sheet gets all of it: 1)
PAPER = 0.9                     # and the paper back's (white paper reads white even turned from the lamp)
INSIDE = 0.4                    # the picture up the inside of the roll this much darker where it stands on end
CREST = 0.3                     # the paper back this much darker where it turns down to meet the sheet than on its crest
SHEEN = (10.0, 0.12)            # the sheen along the roll's crest: its exponent and strength
GRAIN = 0.025                   # the paper's grain on the back, +-
CONTACT = 0.9                   # the roll's contact shadow right under it (a share of the sky hidden)
STEP = 0.25                     # the grid (px along the normal) the roll's runs are matched on
SHADOW_STEP = 4                 # the shadow is worked out at a quarter of the frame's size
SHADOW_BANDS = (0.0, 0.025, 0.08, 0.22, 10.0)   # heights (short sides) of the lifted parts' bands, each its own shadow
SHADOW_BLUR = (0.004, 0.22)     # a shadow's blur: short sides plus this share of the height that casts it
HALO = (0.02, 0.3)              # the halo round a lifted sheet: its blur (short sides) and darkness


def frame(tr, frame, main, incoming):
    """The outgoing frame `main` (float RGB) peeled at `frame` of the transition `tr`'s window off `incoming`."""
    h, w = main.shape[:2]
    k = frame - tr["first"]
    front, cover = _sticker(tr, main)
    sh = sheet(front, cover, tr["back"], PL.normal(tr), PL.line(tr, k, (w, h)), PL.roll(tr, (w, h)), PL.phi(tr, k))
    return sh.over(np.ascontiguousarray(incoming, np.float32), tr["ink"], tr["shadow"])


def _sticker(tr, main):
    """The outgoing picture as a sticker: (front, coverage), with its die-cut rim and rounded corners when it has an
    `edge`."""
    h, w = main.shape[:2]
    main = np.ascontiguousarray(main, np.float32)
    if not tr.get("edge"):
        return main, np.ones((h, w), np.float32)
    e = tr["edge"]
    px = min(w, h) / 1080.0
    d = _rounded(w, h, e["round"] * min(w, h))
    rim = np.clip(e["width"] * px + 0.5 + d, 0.0, 1.0)              # d < 0 inside: the band within `width` of the edge
    front = main + (np.asarray(e["color"], np.float32) - main) * rim[..., None]
    return front, np.clip(0.5 - d, 0.0, 1.0)


@lru_cache(maxsize=4)
def _rounded(w, h, r):
    """The signed distance (px, negative inside) to a w x h frame's edge with its corners rounded by r px."""
    xs, ys = M.grid(h, w)
    qx = np.abs(xs - np.float32((w - 1) / 2.0)) - np.float32(w / 2.0 - r)
    qy = np.abs(ys - np.float32((h - 1) / 2.0)) - np.float32(h / 2.0 - r)
    out = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - np.float32(r))
    out.setflags(write=False)
    return out


class Sheet:
    """A sticker peeled to one fold: `flat` (premultiplied RGB, alpha) of what is still stuck, `up` (premultiplied RGB,
    alpha) of what is lifted and `height` (px toward the lens it casts its shadow from where `up` covers: no higher than
    the roll's axis). (h, w) each. `foot` = (n, c, r0): the fold's normal and line and the roll's radius there, for its
    contact shadow, and `nook` (h, w): how much of that shadow falls on `up` where it shows the printed face rising from
    the fold (None while nothing is lifted)."""

    def __init__(self, flat, flat_a, up, up_a, height, foot=None, nook=None, parts=None):
        self.flat, self.flat_a, self.up, self.up_a, self.height = flat, flat_a, up, up_a, height
        self.foot, self.nook = foot, nook
        self.parts = parts                  # [(y0, y1, x0, x1)] holding all that is lifted (None: anywhere)

    def shadow(self):
        """(h, w) 0..1: the soft shadow the lifted parts cast on what is under them and the roll's contact shadow
        (`over` lays it on)."""
        s = shadow(self.up_a, self.height)
        if self.foot is not None:
            s = 1.0 - (1.0 - s) * (1.0 - contact(self.up_a, *self.foot))
        return s

    def over(self, under, ink, strength):
        """The sheet laid on `under` (float RGB): the stuck part, the shadow of the lifted parts (`strength` of the way
        toward the picture multiplied by `ink`: on a light picture all but toward `ink` itself, and a dark one is
        darkened too, never lifted), the lifted parts (the picture rising from the fold in the same contact shadow as
        the sheet beside it)."""
        out = under * (1.0 - self.flat_a[..., None]) + self.flat
        h, w = self.up_a.shape
        parts = [(0, h, 0, w)] if self.parts is None else self.parts
        if not parts:
            return out
        dark = np.float32(strength) * (1.0 - np.asarray(ink, np.float32))
        if strength > 0.0:
            s = self.shadow()
            for y0, y1, x0, x1 in _spans(s):
                out[y0:y1, x0:x1] *= 1.0 - s[y0:y1, x0:x1, None] * dark
        for y0, y1, x0, x1 in parts:
            up = self.up[y0:y1, x0:x1]
            if strength > 0.0 and self.nook is not None:
                up = up * (1.0 - self.nook[y0:y1, x0:x1, None] * dark)
            o = out[y0:y1, x0:x1]
            o *= 1.0 - self.up_a[y0:y1, x0:x1, None]
            o += up
        return out


def _spans(s, rows=64, least=1e-3):
    """[(y0, y1, x0, x1)]: bands of `rows` rows of `s` (h, w), each cut to the columns where it is over `least`."""
    h = s.shape[0]
    out = []
    for y0 in range(0, h, rows):
        y1 = min(y0 + rows, h)
        cols = np.flatnonzero((s[y0:y1] > least).any(axis=0))
        if cols.size:
            out.append((y0, y1, int(cols[0]), int(cols[-1]) + 1))
    return out


def sheet(front, cover, back, n, c, R, f, box=None):
    """A sticker peeled at a fold (mkmmd.core.peel's geometry): `front` (h, w, 3) its picture in its flat place, `cover`
    (h, w) its coverage there (its rim included), `back` the RGB of its paper, `n` the unit normal from its stuck part
    toward the corner pulled up, `c` the fold line (px along n), `R` the roll's radius (px) and `f` how far round the
    roll (radians) the sheet goes at that radius before it winds in on itself. `box` = (y0, y1, x0, x1) is where the
    sticker lies (default the whole frame): the sheet is lifted up to the farthest corner of the box round what it
    covers there (its pinched corner), and the roll is worked out wherever it is seen in the frame, over the strip it
    covers. Returns a Sheet the frame's size."""
    h, w = cover.shape
    Y0, Y1, X0, X1 = box or (0, h, 0, w)
    front = np.ascontiguousarray(front, np.float32)
    cover = np.ascontiguousarray(cover, np.float32)
    nx, ny = np.float32(n[0]), np.float32(n[1])
    gx, gy = M.grid(h, w)
    D = float(S3.distance((w, h)))
    cx, cy = S3.centre((w, h))
    Cn, Cl = cx * n[0] + cy * n[1], cy * n[0] - cx * n[1]
    out = Sheet(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w, 3), np.float32),
                np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), parts=[])

    def coords(bx):
        y0, y1, x0, x1 = bx
        xs, ys = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
        return xs * nx + ys * ny, ys * nx - xs * ny                  # the pixel along the normal and along the fold

    sn, _ = coords((Y0, Y1, X0, X1))                                 # the part still stuck
    out.flat_a[Y0:Y1, X0:X1] = cover[Y0:Y1, X0:X1] * np.clip(np.float32(c) - sn + 0.5, 0.0, 1.0)
    out.flat[Y0:Y1, X0:X1] = front[Y0:Y1, X0:X1] * out.flat_a[Y0:Y1, X0:X1, None]

    far = _far(cover, (Y0, Y1, X0, X1), n)
    if far is None or far <= c:
        return out
    end = far - float(c)                                             # how much of it is lifted
    us, xs, zs, ths = (np.asarray(v, np.float64) for v in PL.profile(R, f, end))
    xs += c
    P = Cn + (xs - Cn) * D / (D - zs)                                # the roll's cross-section on the screen
    r0 = PL.spiral(R, f, end)[0]
    out.foot, out.nook = (tuple(n), float(c), r0), np.zeros((h, w), np.float32)
    g0 = math.floor(float(P.min())) - 2.0
    G = int(math.ceil((float(P.max()) + 2.0 - g0) / STEP)) + 1
    grid = g0 + STEP * np.arange(G)
    runs = [_run(grid, P[i:j + 1], us[i:j + 1], xs[i:j + 1], zs[i:j + 1], ths[i:j + 1], Cn, D, n)
            for i, j in _runs(P)]
    tops = sorted((r for r in runs if r[0]), key=lambda r: r[1])    # front to back: the turns' tops outside in,
    bottoms = sorted((r for r in runs if not r[0]), key=lambda r: -r[1])     # then their bottoms inside out
    order = [r[2:] for r in tops + bottoms]

    m = D / (D - float(zs.max()))                                    # where the sticker's box is seen, lifted that high
    sx = [cx + (x - cx) * m for x in (X0 - 0.5, X1 - 0.5)] + [X0 - 0.5, X1 - 0.5]
    sy = [cy + (y - cy) * m for y in (Y0 - 0.5, Y1 - 0.5)] + [Y0 - 0.5, Y1 - 0.5]
    vb = (max(int(math.floor(min(sy))) - 1, 0), min(int(math.ceil(max(sy))) + 2, h),
          max(int(math.floor(min(sx))) - 1, 0), min(int(math.ceil(max(sx))) + 2, w))
    back = np.asarray(back, np.float32)
    grain = _grain(h, w)
    cf, Clf, Df = np.float32(c), np.float32(Cl), np.float32(D)
    for bx in PT._strip(n, float(c), g0 - float(c) - 1.0, g0 + STEP * G - float(c) + 1.0, w, h):
        y0, y1, x0, x1 = max(bx[0], vb[0]), min(bx[1], vb[1]), max(bx[2], vb[2]), min(bx[3], vb[3])
        if y1 <= y0 or x1 <= x0:
            continue
        sn, sl = coords((y0, y1, x0, x1))
        gi = np.clip(((sn - np.float32(g0)) * np.float32(1.0 / STEP) + 0.5).astype(np.int32), 0, G - 1).ravel()
        sl = sl.ravel()
        T = np.ones(gi.size, np.float32)                             # how much still shows through
        rgb = np.zeros((gi.size, 3), np.float32)
        zsum = np.zeros(gi.size, np.float32)
        nook = np.zeros(gi.size, np.float32)
        for aa, uu, zz, face, light, sheen in order:
            a1 = aa[gi]
            idx = np.flatnonzero((a1 > 0.0) & (T > 1e-3))
            if idx.size == 0:
                continue
            g = gi[idx]
            z = zz[g]
            pl = Clf + (sl[idx] - Clf) * (1.0 - z / Df)              # the pixel's place along the fold, on the sheet
            un = cf + uu[g]
            mx, my = un * nx - pl * ny, un * ny + pl * nx
            a = _pick(cover, mx, my, 0.0) * a1[idx]
            col = np.empty((idx.size, 3), np.float32)
            fr = face[g]
            if fr.any():                                             # the printed face, up the inside of the roll
                col[fr] = _pick(front, mx[fr], my[fr], (0.0, 0.0, 0.0)) * light[g[fr]][:, None]
                low = np.zeros(idx.size, np.float32)                 # in the contact shadow near the fold
                low[fr] = np.float32(CONTACT) / (1.0 + (uu[g[fr]] / np.float32(r0)) ** 2)
                nook[idx] += low * T[idx] * a
            bk = ~fr
            if bk.any():                                             # the paper back
                gb = g[bk]
                gr = _pick(grain, mx[bk], my[bk], 0.0)
                col[bk] = np.clip(back * (light[gb] * (1.0 + gr))[:, None] + sheen[gb][:, None], 0.0, 1.0)
            wt = T[idx] * a
            rgb[idx] += col * wt[:, None]
            zsum[idx] += z * wt
            T[idx] *= 1.0 - a
        cov = 1.0 - T
        shape = (y1 - y0, x1 - x0)
        out.up[y0:y1, x0:x1] = rgb.reshape(shape + (3,))
        out.up_a[y0:y1, x0:x1] = cov.reshape(shape)
        zc = np.minimum(zsum / np.maximum(cov, 1e-6), np.float32(r0))   # the shadow is cast from no higher than the
        out.height[y0:y1, x0:x1] = zc.reshape(shape)                   # roll's axis: a tube lies on the frame
        out.nook[y0:y1, x0:x1] = (nook / np.maximum(cov, 1e-6)).reshape(shape)
        out.parts.append((y0, y1, x0, x1))
    return out


def _far(cover, box, n):
    """How far along `n` (px) the farthest corner of the box round what `cover` covers in `box` lies, or None."""
    Y0, Y1, X0, X1 = box
    sub = cover[Y0:Y1, X0:X1] > 0.0
    rows, cols = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
    if rows.size == 0:
        return None
    return max(x * n[0] + y * n[1] for x in (X0 + cols[0] - 0.5, X0 + cols[-1] + 0.5)
               for y in (Y0 + rows[0] - 0.5, Y0 + rows[-1] + 0.5))


def _runs(P):
    """[(i, j)]: the runs of samples over which the roll's place on the screen `P` goes one way (from silhouette to
    silhouette, the fold and the free end), each sharing its ends with its neighbours."""
    up = np.diff(P) >= 0.0
    ends = [0] + (np.flatnonzero(up[1:] != up[:-1]) + 1).tolist() + [len(P) - 1]
    return [(i, j) for i, j in zip(ends[:-1], ends[1:]) if j > i]


def _run(grid, p, u, x, z, th, Cn, D, n):
    """One run of the roll matched to the grid of places along the normal: (whether it is a turn's top, where it
    starts on the sheet, then per grid place: its coverage there (antialiased at its ends), the sheet's point u, its
    height, whether its printed face is seen, its light, its sheen)."""
    if p[-1] < p[0]:
        p, u, x, z, th = p[::-1], u[::-1], x[::-1], z[::-1], th[::-1]
    lo, hi = float(p[0]), float(p[-1])
    a_lo = np.clip(grid - lo + 0.5, 0.0, 1.0) if u[0] > 0.0 else (grid >= lo - 0.5).astype(np.float64)
    a_hi = np.clip(hi - grid + 0.5, 0.0, 1.0) if u[-1] > 0.0 else (grid <= hi + 0.5).astype(np.float64)
    aa = np.minimum(a_lo, a_hi)                                     # antialiased at its ends, but where it leaves the
                                                                    # fold: it goes on from the flat sheet there
    ug, xg, zg, tg = (np.interp(grid, p, v) for v in (u, x, z, th))
    s, c = np.sin(tg), np.cos(tg)
    L = (S3.LIGHT[0] * n[0] + S3.LIGHT[1] * n[1], S3.LIGHT[2])     # the lamp and the sheen's half way in (n, height)
    Hh = (S3.HALF[0] * n[0] + S3.HALF[1] * n[1], S3.HALF[2])

    def lit(nn, nz, ambient):                                       # never brighter than the flat sheet
        return ambient + (1.0 - ambient) * np.minimum(np.maximum(nn * L[0] + nz * L[1], 0.0) / L[1], 1.0)

    face = (-s) * (Cn - xg) + c * (D - zg) > 0.0                   # the printed face (the inside of the turn) is seen
    inside = lit(-s, c, AMBIENT) * (1.0 - INSIDE * np.clip(tg / (0.5 * math.pi), 0.0, 1.0) ** 2)
    paper = lit(s, -c, PAPER) * (1.0 - CREST * (1.0 - np.clip(-c, 0.0, 1.0)) ** 2)
    sheen = np.where(face, 0.0, np.maximum(s * Hh[0] - c * Hh[1], 0.0) ** SHEEN[0] * SHEEN[1])
    f32 = np.float32
    return (float(np.mean(np.cos(th))) < 0.0, float(u.min()), aa.astype(f32), ug.astype(f32), zg.astype(f32), face,
            np.where(face, inside, paper).astype(f32), sheen.astype(f32))


def _pick(img, mx, my, border):
    """`img` sampled (bilinear) at the points (mx, my), 1D: (N,) or (N, channels)."""
    N = mx.size
    cols = 1024
    rows = (N + cols - 1) // cols
    pad = rows * cols - N
    mx = np.pad(mx.astype(np.float32), (0, pad)).reshape(rows, cols)
    my = np.pad(my.astype(np.float32), (0, pad)).reshape(rows, cols)
    got = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return got.reshape((rows * cols,) + img.shape[2:])[:N]


@lru_cache(maxsize=4)
def _grain(h, w, seed=5):
    """The paper's grain over a sheet of h x w px: fine noise, +-GRAIN."""
    g = np.random.default_rng(seed).standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    g = cv2.addWeighted(g, 0.7, cv2.GaussianBlur(g, (0, 0), 3.0), 2.0, 0.0)
    g = np.ascontiguousarray(g / np.float32(max(float(np.abs(g).max()), 1e-6)) * np.float32(GRAIN))
    g.setflags(write=False)
    return g


def contact(alpha, n, c, r0):
    """(h, w) 0..1: the contact shadow of a roll of radius `r0` lying along the fold line `c` (px along `n`) on what is
    under it: the share of the sky a cylinder on the frame hides d px off the line, r0^2 / (r0^2 + d^2), times CONTACT,
    where the roll lies along the line (its coverage `alpha`, blurred), worked out at 1 / SHADOW_STEP of the size."""
    h, w = alpha.shape
    st = SHADOW_STEP
    small = ((w + st - 1) // st, (h + st - 1) // st)
    a = cv2.resize(alpha, small, interpolation=cv2.INTER_AREA)
    a = np.clip(2.0 * cv2.GaussianBlur(a, (0, 0), max(0.5 * r0 / st, 0.5)), 0.0, 1.0)
    xs, ys = M.grid(small[1], small[0])
    off = np.float32((st - 1) / 2.0)
    d = ((xs * st + off) * np.float32(n[0]) + (ys * st + off) * np.float32(n[1]) - np.float32(c)) / np.float32(r0)
    k = np.float32(CONTACT) * a / (1.0 + d * d)
    return cv2.resize(k, (w, h), interpolation=cv2.INTER_LINEAR)


def shadow(alpha, height):
    """(h, w) 0..1: the soft shadow cast by what covers `alpha` (h, w) at `height` (h, w, px toward the lens): each band
    of heights moved away from the lamp by its height (mkmmd.core.sheet3d.SHADOW_RUN) and blurred the more the higher it
    is, and a faint halo round all of it that is well off the frame (the room's light is not one lamp), worked out at
    1 / SHADOW_STEP of the size."""
    h, w = alpha.shape
    S = float(min(h, w))
    st = SHADOW_STEP
    small = ((w + st - 1) // st, (h + st - 1) // st)
    a = cv2.resize(alpha, small, interpolation=cv2.INTER_AREA)
    z = cv2.resize(height * alpha, small, interpolation=cv2.INTER_AREA)
    zs = z / np.maximum(a, 1e-6)
    keep = np.ones_like(a)
    for lo, hi in zip(SHADOW_BANDS[:-1], SHADOW_BANDS[1:]):
        band = a * ((zs >= lo * S) & (zs < hi * S))
        if float(band.max()) <= 0.0:
            continue
        zm = float((band * zs).sum() / band.sum())
        ox, oy = S3.SHADOW_RUN[0] * zm / st, S3.SHADOW_RUN[1] * zm / st
        moved = cv2.warpAffine(band, np.float32([[1, 0, ox], [0, 1, oy]]), small, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sigma = max((SHADOW_BLUR[0] * S + SHADOW_BLUR[1] * zm) / st, 0.5)
        keep *= 1.0 - np.clip(cv2.GaussianBlur(moved, (0, 0), sigma), 0.0, 1.0)
    lifted = a * (zs > SHADOW_BANDS[1] * S)                    # and a faint halo all round what is well off the frame
    if float(lifted.max()) > 0.0:
        keep *= 1.0 - HALO[1] * np.clip(cv2.GaussianBlur(lifted, (0, 0), max(HALO[0] * S / st, 0.5)), 0.0, 1.0)
    return np.clip(cv2.resize(1.0 - keep, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
