"""Echoes of a sheet's live sticker on the finished pictures (docs/design.md: Frame effects, Mandalas and trails): the
picture work of a `trails` entry that follows a sheet (mkmmd.core.echoes). Its copies are the sticker as the sheet laid
it on frames before (mkmmd.zap.ZapFx.live_sticker: kept as the sheet laid it, or the sheet run again on a frame out of
order), her picture washed toward each copy's ink by `tint`: `delay` frames apart where she was (fanned by `step`), a
`train` dropped along the path she took, or wound out on a `spiral` round her. Images are float RGB in display space
(0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from . import mandala as MDF
from .core import echoes as EC
from .core import framefx as FX
from .core import mandala as MD


def _ff():
    from . import framefx as FF          # framefx imports mandala, which imports this module: its helpers are there now
    return FF


def middle(got):
    """The frame px of the coverage-weighted middle of a sticker ((box, face, coverage)), or None."""
    (x0, y0, _, _), _, a = got
    m = cv2.moments(np.ascontiguousarray(a, np.float32))
    return None if m["m00"] < 1e-6 else (x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"])


def height(got):
    """A sticker's height px: its box's."""
    return float(got[0][3] - got[0][1])


def laid(out, got, colour, alpha, tint, how, M=None):
    """`out` (in place) with a copy of a sticker `got` ((box, premultiplied face, coverage)) in `colour`, its own picture
    mixed in by 1 - `tint`, at `alpha`, blended `how` (`over`, `screen`, `add`), carried by the 2 x 3 map `M` (frame px
    to frame px; None: where it lies)."""
    (x0, y0, x1, y1), face, cov = got
    H, W = out.shape[:2]
    if M is None:
        rx0, ry0, rx1, ry1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if rx1 <= rx0 or ry1 <= ry0:
            return out
        a = cov[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0]
        f = face[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0]
    else:
        A = np.asarray(M, np.float64)
        corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float) @ A[:, :2].T + A[:, 2]
        rx0, ry0 = max(int(math.floor(corners[:, 0].min())) - 1, 0), max(int(math.floor(corners[:, 1].min())) - 1, 0)
        rx1, ry1 = min(int(math.ceil(corners[:, 0].max())) + 2, W), min(int(math.ceil(corners[:, 1].max())) + 2, H)
        if rx1 <= rx0 or ry1 <= ry0:
            return out
        B = A.copy()                                     # the crop's px -> the region's px
        B[:, 2] = A[:, 2] + A[:, :2] @ np.array([x0, y0], float) - np.array([rx0, ry0], float)
        Mf, shrink = _ff().warp_spec(B, cov.shape)
        src = np.ascontiguousarray(np.dstack([face, cov]), np.float32)
        if shrink is not None:
            src = cv2.resize(src, shrink, interpolation=cv2.INTER_AREA)
        w = cv2.warpAffine(src, Mf, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        f, a = w[..., :3], np.clip(w[..., 3], 0.0, 1.0)
    col = np.asarray(colour, np.float32)
    if tint < 1.0:
        own = f / np.maximum(a, 1e-4)[..., None]
        col = col + (own - col) * np.float32(1.0 - tint)
    region = out[ry0:ry1, rx0:rx1]
    out[ry0:ry1, rx0:rx1] = MDF.blend(region, col, (a * np.float32(alpha))[..., None], how)
    return out


def scaled(got, k, dx=0.0, dy=0.0):
    """The 2 x 3 map that scales a sticker `k` times about its own middle and moves it `dx`, `dy` px (None: neither)."""
    if k == 1.0 and not dx and not dy:
        return None
    own = middle(got)
    if own is None:
        return None
    return [[k, 0.0, own[0] * (1.0 - k) + dx], [0.0, k, own[1] * (1.0 - k) + dy]]


def sheet_of(ffx, fx):
    """This output's entry of the sheet a trail follows."""
    return next(s for s in ffx.fxs if s["index"] == fx["sheet_index"])


def train_drops(ffx, fx, sheet, frame):
    """A train's drops up to the frame before `frame` ([(index along the path, frame)], mkmmd.core.echoes.travel's
    rule), followed on from the frame before when that was the last asked (the stickers there are kept), else along
    the whole path from the trail's first frame again."""
    st = ffx._echo_paths.get(fx["index"])
    if st is None or st["upto"] > frame - 1:
        st = ffx._echo_paths[fx["index"]] = {"upto": fx["f0"] - 1, "s": 0.0, "last": None, "drops": []}
    for g in range(st["upto"] + 1, frame):
        got = ffx.live_sticker(sheet, g)
        c = None if got is None else middle(got)
        st["s"], st["last"], k = EC.travel(st["s"], st["last"], c, None if got is None else height(got),
                                           fx["train"]["spacing"])
        if k is not None:
            st["drops"].append((k, g))
        st["upto"] = g
    return st["drops"]


def draw(ffx, fx, frame, img):
    """Trail `fx` of a sheet's sticker on `frame` over `img`, the copies the oldest (or farthest) first: a `train`'s
    (mkmmd.core.echoes.train_copies: one ink after another along the path), a `spiral`'s (round her middle now, copy j
    her sticker j x `delay` frames before, its ink the j-th) or `delay` frames apart (mkmmd.core.mandala.trail_copies,
    `step` x j further, `(1 - shrink)` ** j as large, with `still` none of a sticker that has barely moved since). The
    sticker of `frame` is laid back over them when the sheet comes before the trail in the table (it has laid it
    already); after it, the sheet lays it over them."""
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return img
    sheet = sheet_of(ffx, fx)
    out = np.array(img, np.float32, copy=True)
    inks = fx["colors"]
    if fx["train"] is not None:
        for k, g, alpha, scale in EC.train_copies(fx, frame, train_drops(ffx, fx, sheet, frame), ffx.fps):
            got = ffx.live_sticker(sheet, g)
            if got is not None:
                laid(out, got, inks[k % len(inks)], alpha * s, fx["tint"], fx["blend"], scaled(got, scale))
    elif fx["spiral"] is not None:
        before = sheet["index"] > fx["index"]               # the trail before its sheet: her middle a frame ago, kept
        now = ffx.live_sticker(sheet, frame - 1 if before and sheet["f0"] < frame else frame)
        here = None if now is None else middle(now)
        if here is not None:
            for j in range(fx["count"], 0, -1):
                got = ffx.live_sticker(sheet, frame - j * fx["delay"])
                own = None if got is None else middle(got)
                if own is None:
                    continue
                M, shown = EC.spiral_map(fx["spiral"], j, fx["count"], t, fx["from"], fx["to"], own, here, height(now))
                if shown > 0.0:
                    laid(out, got, inks[(j - 1) % len(inks)], shown * s, fx["tint"], fx["blend"], M)
    else:
        unit = float(min(ffx.size))
        now = ffx.live_sticker(sheet, frame) if fx["still"] > 0 else None
        here = None if now is None else middle(now)
        still = fx["still"] * unit
        for j, g, alpha, u in MD.trail_copies(fx, frame, t, ffx._hit_times(fx), ffx.fps, ffx.frame0):
            got = ffx.live_sticker(sheet, g)
            if got is None:
                continue
            a = alpha * s
            if still > 0 and here is not None:              # gone where the sticker has barely moved since (a hover)
                was = middle(got)
                moved = 0.0 if was is None else math.hypot(was[0] - here[0], was[1] - here[1])
                a *= min(max(moved / still - 1.0, 0.0), 1.0)
            if a > 0.0:
                laid(out, got, MD.ramp(inks, u), a, fx["tint"], fx["blend"],
                     scaled(got, (1.0 - fx["shrink"]) ** j, fx["step"][0] * unit * j, fx["step"][1] * unit * j))
    if sheet["index"] < fx["index"]:                        # the sheet laid her already: she lies over her trail
        got = ffx.live_sticker(sheet, frame)
        if got is not None:
            (x0, y0, x1, y1), f, c = got
            out[y0:y1, x0:x1] = f + out[y0:y1, x0:x1] * (1.0 - c[..., None])
    return out
