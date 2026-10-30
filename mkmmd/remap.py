"""The remapping frame effects on the finished pictures (docs/design.md: Frame effects, Remaps): the picture work of the
`[[fx]]` kinds kaleido, tunnel and melt that mkmmd.core.remap checks, times and maps. RemapFx is the
part of mkmmd.framefx.FrameFx that runs them. Images are float RGB in display space (0..1), (h, w, 3); a figure is its
coverage, (h, w) in 0..1 over the same picture; `px` is a frame's pixels per pixel at 1080 on its short side.

Every remap samples the picture bilinearly with its edge mirrored (a kaleidoscope's wedge or a melt reaching past the
frame sees the frame reflected, never black), from a smaller copy of it where the remap shrinks it (no shimmer in fine
detail as a tunnel falls or a kaleidoscope zooms out)."""
import math

import cv2
import numpy as np

from .core import framefx as FX
from .core import remap as RM

HOLE_MARGIN = 1.5                # px at 1080: a figure's place grown past its antialiased outline before it is filled
HOLE_CUT = 1.0 / 255.0           # coverage above this is the figure's place (what the fill replaces)
FADE = 0.25                      # the share of a run over which the ground mirrored in from its two sides crossfades
SHRINK = 1.25                    # a remap that shrinks the picture more than this samples a smaller copy of it


# ================================================================================================= sampling
class Pyramid:
    """A picture and its copies halved again and again (cv2.pyrDown: blurred, then every second pixel), made as they
    are asked for: level j's pixel (x, y) is the picture's (x * 2 ** j, y * 2 ** j)."""

    def __init__(self, pic):
        self.levels = [np.ascontiguousarray(pic, np.float32)]

    def __getitem__(self, j):
        while len(self.levels) <= j and min(self.levels[-1].shape[:2]) > 2:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        return self.levels[min(j, len(self.levels) - 1)]


def lods(shrink):
    """[(pyramid level, weight)] to sample a picture shrunk `shrink` times from: the level itself up to SHRINK, then
    the two levels round log2(shrink), blended so that a zoom glides from one to the next."""
    if shrink <= SHRINK:
        return [(0, 1.0)]
    lod = math.log2(shrink / SHRINK)
    j = int(math.floor(lod))
    f = lod - j
    return [(j, 1.0 - f), (j + 1, f)] if f > 1e-3 else [(j, 1.0)]


def _remap(pic, sx, sy):
    out = cv2.remap(pic, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out if out.ndim == 3 else out[..., None]


def sample(pyr, sx, sy, shrink):
    """The picture of `pyr` (a Pyramid) at (sx, sy) (float32 px of the picture), bilinear, its edge mirrored, from the
    copies a remap that shrinks it `shrink` times calls for."""
    out = None
    for j, wt in lods(shrink):
        k = np.float32(0.5 ** j)
        got = _remap(pyr[j], sx * k, sy * k) if j else _remap(pyr[0], sx, sy)
        out = got * np.float32(wt) if out is None else out + got * np.float32(wt)
    return out


def warp(pyr, M, m, size, border=cv2.BORDER_REFLECT):
    """The picture of `pyr` carried by the affine map `M` (2 x 3: each pixel of the result -> the picture's px) into a
    picture of `size` (w, h), drawn `m` times as large (from the smaller copies when m is under 1)."""
    M = np.asarray(M, np.float64)
    out = None
    for j, wt in lods(1.0 / m):
        got = cv2.warpAffine(pyr[j], (M * 0.5 ** j).astype(np.float32), (int(size[0]), int(size[1])),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)
        got = got if got.ndim == 3 else got[..., None]
        out = got * np.float32(wt) if out is None else out + got * np.float32(wt)
    return out


def half_map(size, coords_of):
    """coords_of(xs, ys) -> (sx, sy) worked out on a grid of half the frame's resolution and carried up to its own
    (bilinear): for a map smooth but for its creases (a two-mirror kaleidoscope's), four times less work. Each coarse
    point stands where cv2.resize reads it, so a linear map comes back exact."""
    w, h = size
    hw, hh = (w + 1) // 2, (h + 1) // 2
    gx, gy = RM.grid((hw, hh))
    sx, sy = coords_of((gx + np.float32(0.5)) * np.float32(w / hw) - np.float32(0.5),
                       (gy + np.float32(0.5)) * np.float32(h / hh) - np.float32(0.5))
    return (cv2.resize(np.ascontiguousarray(sx, np.float32), (w, h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(np.ascontiguousarray(sy, np.float32), (w, h), interpolation=cv2.INTER_LINEAR))


# ================================================================================================= the kaleidoscope
def kaleido(pic, coords, shrink, seams=None):
    """`pic` ((h, w, c)) through a kaleidoscope's map: coords (sx, sy) from mkmmd.core.remap.kaleido_coords, shrinking
    the picture `shrink` times; `seams` (index, weight, (sx, sy) across) blends the pixels by the jump where plain
    rotational copies meet with the picture across it (antialiased)."""
    pyr = Pyramid(pic)
    out = sample(pyr, coords[0], coords[1], shrink)
    if seams is not None:
        idx, wt, (ox, oy) = seams
        other = sample(pyr, ox[None, :], oy[None, :], shrink)[0]
        flat = out.reshape(-1, out.shape[2])
        flat[idx] = flat[idx] * wt[:, None] + other * (1.0 - wt[:, None])
    return out


def seam_blend(xs, ys, fx, centre, source, zoom, radius, turn, spin):
    """(index, weight, (sx, sy)) of the pixels within a pixel of a seam between plain rotational copies: the share of
    each that its own copy covers and where the copy across the seam samples it."""
    d, side = RM.kaleido_seams(xs, ys, centre, fx["segments"], fx["source"]["angle"], spin)
    near = np.flatnonzero(d.ravel() < 1.0)
    if near.size == 0:
        return None
    px, py, sd = xs.ravel()[near], ys.ravel()[near], side.ravel()[near]
    wedge = 360.0 / fx["segments"]
    across = [RM.kaleido_coords(px, py, centre, fx["segments"], False, fx["source"]["angle"], turn + s * wedge, spin,
                                source, zoom, radius) for s in (1.0, -1.0)]
    ox = np.where(sd > 0, across[0][0], across[1][0]).astype(np.float32)
    oy = np.where(sd > 0, across[0][1], across[1][1]).astype(np.float32)
    return near, np.clip(0.5 + d.ravel()[near], 0.0, 1.0).astype(np.float32), (ox, oy)


# ================================================================================================= the tunnel
def level_cover(box, A, m, size, width=0.0):
    """(coverage, line) (bh, bw) over the screen box (x0, y0, x1, y1) of a tunnel level whose screen -> frame map is `A`
    (drawn `m` times as large): how much of each pixel the level's frame covers, antialiased on screen, and how much of
    it a frame line `width` px wide (in the frame's own px, along its inside edge) covers."""
    x0, y0, x1, y1 = box
    w, h = size
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    qx = np.float32(A[0, 0]) * X + np.float32(A[0, 1]) * Y + np.float32(A[0, 2])
    qy = np.float32(A[1, 0]) * X + np.float32(A[1, 1]) * Y + np.float32(A[1, 2])
    d = np.minimum(np.minimum(qx + 0.5, np.float32(w - 0.5) - qx), np.minimum(qy + 0.5, np.float32(h - 0.5) - qy))
    cov = np.clip(d * np.float32(m) + 0.5, 0.0, 1.0)
    line = None if width <= 0 else np.clip((np.float32(width) - d) * np.float32(m) + 0.5, 0.0, 1.0)
    return cov, line


def tunnel(pic, levels, centre, colour_of=None, width=0.0, nest=True):
    """`pic` ((h, w, c)) as a Droste tunnel: `levels` from mkmmd.core.remap.tunnel_levels (k, m, theta, alpha), the
    first covering the frame, each next laid over it antialiased; with `colour_of` (k -> colour) a frame line `width`
    px wide (the frame's own px) round each, inside its edge, drawn as part of its picture (a level's line shrinks with
    it). With `nest` false `pic` is a cut-out (4 channels: colour premultiplied by coverage, and the coverage) and
    `levels` mkmmd.core.remap.tunnel_queue's: whole copies of it laid far to near, nothing outside the frame, each
    copy's frame line round it (of coverage 1)."""
    h, w = pic.shape[:2]
    pyr = Pyramid(pic)
    lined = colour_of is not None and width > 0
    out = None if nest else np.zeros(pic.shape, np.float32)
    for k, m, theta, alpha in levels:
        A = RM.level_map(centre, m, theta)
        box = (0, 0, w, h) if out is None else RM.level_box((w, h), centre, m, theta)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        Ab = np.hstack([A[:, :2], (A[:, :2] @ np.array([x0, y0], float) + A[:, 2])[:, None]])
        got = warp(pyr, Ab, m, (x1 - x0, y1 - y0), cv2.BORDER_REFLECT if nest else cv2.BORDER_CONSTANT)
        cov = line = None
        if out is not None or lined:
            cov, line = level_cover(box, A, m, (w, h), width if lined else 0.0)
        if line is not None:
            col = np.asarray(list(colour_of(k)) + [1.0] * (got.shape[2] - 3), np.float32)
            got = got + (col - got) * (line * cov)[..., None]
        if out is None:
            out = got
        elif nest:
            out[y0:y1, x0:x1] += (got - out[y0:y1, x0:x1]) * (cov * np.float32(alpha))[..., None]
        else:
            got *= np.float32(alpha)
            out[y0:y1, x0:x1] = got + out[y0:y1, x0:x1] * (1.0 - got[..., 3:4])
    return pic if out is None else out


# ================================================================================================= the figure's place
def pushpull(img, known):
    """`img` (h, w, c) with its colours where `known` ((h, w) 0..1) is low spread smoothly in from where it is high:
    the known colours (premultiplied) and their weights halved down a pyramid, then each level's gaps filled from the
    one below on the way back up."""
    P, A = [np.ascontiguousarray(img * known[..., None], np.float32)], [np.ascontiguousarray(known, np.float32)]
    while min(A[-1].shape) > 2:
        P.append(cv2.pyrDown(P[-1]))
        A.append(cv2.pyrDown(A[-1]))
    a = A[-1][..., None]
    col = np.where(a > 1e-4, P[-1] / np.maximum(a, 1e-4), np.float32(0.5))
    for j in range(len(P) - 2, -1, -1):
        up = cv2.resize(col, (P[j].shape[1], P[j].shape[0]), interpolation=cv2.INTER_LINEAR)
        up = up if up.ndim == 3 else up[..., None]
        col = P[j] + (1.0 - A[j][..., None]) * up
    return col.astype(np.float32)


def fill(img, hole):
    """`img` with the place `hole` ((h, w) 0..1) covers filled from the ground round it, mixed in by `hole`: each row's
    run of it takes the ground on either side mirrored in (the ground's own texture goes on), the two crossfaded where
    they meet; what the mirrors cannot reach (a run at the frame's edge, a mirror landing in another run) is the
    ground's colour spread in from round it (pushpull)."""
    H = hole > HOLE_CUT
    rows = np.flatnonzero(H.any(1))
    if rows.size == 0:
        return img
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    Hr = H[r0:r1]
    w = Hr.shape[1]
    xs = np.arange(w, dtype=np.int32)[None, :]
    left = np.maximum.accumulate(np.where(Hr, np.int32(-1), xs), axis=1)              # the last ground pixel at or
    right = np.minimum.accumulate(np.where(Hr, np.int32(w), xs)[:, ::-1], axis=1)[:, ::-1]   # before x, the first after
    at = np.flatnonzero(Hr)                                                 # the pixels of the runs, row by row
    x, row = (at % w).astype(np.int32), (at // w).astype(np.int32) * np.int32(w)
    lo, hi = left.ravel()[at], right.ravel()[at]
    src_l, src_r = 2 * lo + 1 - x, 2 * hi - 1 - x                           # mirrored about the run's two ends
    flat_h = Hr.ravel()
    ok_l = (lo >= 0) & (src_l >= 0) & ~flat_h[row + np.clip(src_l, 0, w - 1)]
    ok_r = (hi < w) & (src_r < w) & ~flat_h[row + np.clip(src_r, 0, w - 1)]
    dl, dr = (x - lo).astype(np.float32), (hi - x).astype(np.float32)
    u = np.clip(0.5 + (dr - dl) / np.maximum(np.float32(2.0 * FADE) * (dl + dr), np.float32(8.0)), 0.0, 1.0)
    wl = np.where(ok_l & ok_r, u * u * (3.0 - 2.0 * u), ok_l.astype(np.float32))[:, None]
    sub = np.ascontiguousarray(img[r0:r1], np.float32)
    flat = sub.reshape(-1, sub.shape[2])
    filled = flat[row + np.clip(src_l, 0, w - 1)] * wl + flat[row + np.clip(src_r, 0, w - 1)] * (1.0 - wl)
    lost = ~(ok_l | ok_r)
    if lost.any():                     # spread in from the whole frame: a run may fill whole rows of the band
        spread = pushpull(np.ascontiguousarray(img, np.float32), (~H).astype(np.float32))[r0:r1]
        filled[lost] = spread.reshape(-1, sub.shape[2])[at[lost]]
    a = hole[r0:r1].ravel()[at][:, None]
    out = np.array(img, np.float32, copy=True)
    band = out[r0:r1].reshape(-1, out.shape[2])
    band[at] = flat[at] * (1.0 - a) + filled * a
    return out


def shade(cov, shadow):
    """(h, w) how much of each pixel the drop shadow of a coverage `cov` darkens, as a collage casts it
    (mkmmd.framefx._cast): `cov` moved (dx, dy) px, its edge repeated, blurred `blur` px (a Gaussian's sigma), times
    `alpha`; shadow = (colour, (dx, dy), blur, alpha). A wide blur is worked out at a quarter of the size: the same
    shadow for a sixteenth of the work."""
    _, (dx, dy), blur, alpha = shadow
    h, w = cov.shape
    q = 4 if blur >= 6.0 and min(w, h) >= 64 else 1
    small = cov if q == 1 else cv2.resize(np.ascontiguousarray(cov, np.float32), (w // q, h // q),
                                          interpolation=cv2.INTER_AREA)
    k = small.shape[1] / w
    sh = cv2.warpAffine(np.ascontiguousarray(small, np.float32), np.float32([[1, 0, dx * k], [0, 1, dy * k]]),
                        (small.shape[1], small.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if blur * k > 0.3:
        sh = cv2.GaussianBlur(sh, (0, 0), blur * k)
    if q > 1:
        sh = cv2.resize(sh, (w, h), interpolation=cv2.INTER_LINEAR)
    return sh * np.float32(alpha)


def sticker(dst, prem, cov, edge=None, shadow=None):
    """`dst` (changed in place and returned) with a figure laid on it as a collage lays its sticker
    (mkmmd.framefx._paste): its colour `prem` (premultiplied by its coverage `cov`) over an `edge` (colour, width px)
    grown round it, the two casting a `shadow` (colour, (dx, dy) px, blur px, alpha). Only the box the figure, its
    edge and its shadow reach is worked on."""
    from . import framefx as FF
    x, y, bw, bh = cv2.boundingRect((cov > HOLE_CUT).astype(np.uint8))
    if bw == 0 or bh == 0:
        return dst
    grow = 2.0 + (edge[1] if edge is not None else 0.0)
    reach = grow + (math.hypot(*shadow[1]) + 3.0 * shadow[2] if shadow is not None else 0.0)
    h, w = cov.shape
    x0, y0 = max(int(x - reach), 0), max(int(y - reach), 0)
    x1, y1 = min(int(math.ceil(x + bw + reach)), w), min(int(math.ceil(y + bh + reach)), h)
    d, c = dst[y0:y1, x0:x1], cov[y0:y1, x0:x1]
    ring = None if edge is None or edge[1] <= 0 else FF._grow(c, edge[1])
    if shadow is not None and shadow[3] > 0:
        a = shade(c if ring is None else ring, shadow)[..., None]
        d += (np.asarray(shadow[0], np.float32) - d) * a
    if ring is not None:
        d += (np.asarray(edge[0], np.float32) - d) * ring[..., None]
    d *= 1.0 - c[..., None]
    d += prem[y0:y1, x0:x1]
    return dst


# ================================================================================================= over a cut
class RemapFx:
    """The remapping kinds' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, _t, _at, _hit_times,
    _figure, _edge, _shadow and _mix)."""

    def _remapped(self, fx, frame, img, remap):
        """`img` through `remap` (a picture -> the picture remapped; it takes 3 and 4 channels) as the effect's `mask`
        says: the whole picture; the ground with the figure laid back on top as rendered (`ground`); or the figure alone
        remapped (its colour premultiplied and its coverage, together) over the ground or a flat `back` (`figure`).
        Under a mask the figure's place in the ground is filled from the ground round it first (and the place of an
        `edge` and a `shadow` the effect draws), and the figure wears the edge and the shadow as a collage's sticker
        does."""
        if fx["mask"] is None:
            return remap(img)
        from . import framefx as FF          # framefx imports this module: it is all there by the time this runs
        m = self._figure(fx, frame)
        if m is None or not (m > HOLE_CUT).any():
            return remap(img) if fx["mask"] == "ground" else img
        edge, shadow = self._edge(fx), self._shadow(fx)
        prem = img * m[..., None]
        if fx["mask"] == "figure" and fx["back"] is not None:
            ground = np.empty_like(img, np.float32)
            ground[:] = np.asarray(fx["back"], np.float32)
        else:
            reach = HOLE_MARGIN * self.px + (edge[1] if edge else 0.0)
            if shadow is not None:
                reach += math.hypot(*shadow[1]) + 2.5 * shadow[2]
            ground = fill(img, FF._grow(m, reach))
        if fx["mask"] == "ground":
            return sticker(remap(ground), prem, m, edge, shadow)
        four = np.empty(img.shape[:2] + (4,), np.float32)
        four[..., :3], four[..., 3] = prem, m
        both = remap(four)
        return sticker(ground, both[..., :3], np.clip(both[..., 3], 0.0, 1.0), edge, shadow)

    def _kaleido(self, fx, frame, img):
        """The frame folded into wedges about the centre, each the source wedge of the picture (mirrored in turn),
        turned, stepped, spun and zoomed as the entry says, mixed in by the strength."""
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0:
            return img
        w, h = self.size
        times = self._hit_times(fx)
        c = self._at(fx, frame)[0]
        src = fx["source"]
        sc = c if src["center"] is None else (src["center"][0] * w - 0.5, src["center"][1] * h - 0.5)
        zoom, turn, spin = RM.kaleido_zoom(fx, t, times), RM.kaleido_turn(fx, t, times), RM.kaleido_spin(fx, t)
        radius = None if src["radius"] is None else src["radius"] * min(w, h)

        def coords_of(xs, ys):
            return RM.kaleido_coords(xs, ys, c, fx["segments"], fx["mirror"], src["angle"], turn, spin, sc, zoom,
                                     radius)
        if fx["mirror"]:              # a two-mirror kaleidoscope's map is smooth but for its creases: half the work
            coords, seams = half_map(self.size, coords_of), None
        else:                         # plain copies jump at their seams: the map at full size, the seams blended
            xs, ys = RM.grid(self.size)
            coords, seams = coords_of(xs, ys), seam_blend(xs, ys, fx, c, sc, zoom, radius, turn, spin)
        out = self._remapped(fx, frame, img, lambda pic: kaleido(pic, coords, 1.0 / zoom, seams))
        return self._mix(fx, frame, img, out)

    def _tunnel(self, fx, frame, img):
        """The frame nested in itself level after level about the centre, falling inward (a figure mask: copies of
        the figure running out of the vanishing point), mixed in by the strength."""
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0:
            return img
        phase = RM.tunnel_phase(fx, t, self._hit_times(fx))
        c = self._at(fx, frame)[0]
        args = (self.size, c, fx["scale"], fx["twist"], phase, fx["depth"])
        line = fx["frame"]
        colour_of = None if line is None else (lambda k: RM.frame_colour(line["color"], k, phase))
        width = 0.0 if line is None else line["width"] * self.px

        def remap(pic):
            if pic.shape[2] == 4:              # the figure alone: a queue of whole copies
                return tunnel(pic, RM.tunnel_queue(*args), c, colour_of, width, nest=False)
            return tunnel(pic, RM.tunnel_levels(*args), c, colour_of, width)
        return self._mix(fx, frame, img, self._remapped(fx, frame, img, remap))

    def _melt(self, fx, frame, img):
        """The picture displaced by the mode's field at this frame; the strength scales the amount (a melt at half
        strength is half a melt, not a mix). A ripple's swell on a hit runs out from the centre."""
        t = self._t(frame)
        times = self._hit_times(fx)
        unit = min(self.size)

        def level(age):
            return RM.melt_level(fx, fx["from"] + age, times, lambda u: FX.strength(fx, u))
        dx, dy = RM.melt_field(fx["mode"], self.size, self._at(fx, frame)[0], level, fx["scale"], fx["speed"],
                               t - fx["from"], unit, (fx["seed"], fx["index"]),
                               front=RM.FRONT * unit if RM.fires(fx) else None)
        if max(float(np.abs(dx).max()), float(np.abs(dy).max())) < 0.01:
            return img
        xs, ys = RM.grid(self.size)
        sx, sy = xs + dx, ys + dy
        return self._remapped(fx, frame, img, lambda pic: _remap(np.ascontiguousarray(pic, np.float32), sx, sy))
