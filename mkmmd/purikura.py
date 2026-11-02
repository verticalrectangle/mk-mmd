"""The purikura look of a photo booth (docs/design.md: Picture effects: pixel sort, film, camcorder, doodles, photo
booth): the picture work of what mkmmd.core.purikura checks and lays out, drawn into the strip mkmmd.picturefx.booth
makes: each capture through the beauty filter in its pastel frame, the paper's polka dots, the stamps as stickers and the
date or word the gel pen writes. Images are float RGB in display space (0..1), (h, w, 3); `px` is a frame's pixels per
pixel at 1080 on its short side."""
import math
from functools import lru_cache

import cv2
import numpy as np

from . import eralook as ERF
from . import picturefx as PF
from .core import hand as HD
from .core import picturefx as PC
from .core import purikura as PK

CORNER = 0.07                    # a photo's corners are rounded this share of its short side
STITCH = 1.5                     # the dots along a frame's middle come this many frame widths apart
STITCH_SIZE = 0.2                # and are this share of its width across (their radius)
DOT_GAP = 4.2                    # the paper's polka dots: this many of their radii apart
DOT_ALPHA = 0.75                 # how strongly they show
SLOT = 0.9                       # an empty slot: the paper this much as light
SMOOTH = 0.35                    # the beauty filter's soft focus blurs the picture this share of `soft` toward its blur
DIFFUSE = 4.0                    # px at 1080: the soft focus's blur


# ================================================================================================= coverages
@lru_cache(maxsize=64)
def rounded(w, h, r, inset=0.0):
    """(h, w) float32 0..1: a rectangle of w x h px `inset` px in from its edges, its corners rounded `r` px (less the
    inset), antialiased."""
    x = np.arange(w, dtype=np.float32)[None, :] + np.float32(0.5 - w / 2.0)
    y = np.arange(h, dtype=np.float32)[:, None] + np.float32(0.5 - h / 2.0)
    hx, hy = w / 2.0 - inset, h / 2.0 - inset
    rr = max(min(r - inset, hx, hy), 0.0)
    qx, qy = np.abs(x) - np.float32(hx - rr), np.abs(y) - np.float32(hy - rr)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - np.float32(rr)
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=8)
def polka(w, h, r):
    """(h, w) float32 0..1: polka dots `r` px in radius over a strip w x h px, DOT_GAP radii apart, every other row
    shifted half a step."""
    step = DOT_GAP * r
    y = np.arange(h, dtype=np.float32)[:, None] + np.float32(0.5)
    x = np.arange(w, dtype=np.float32)[None, :] + np.float32(0.5) + (np.floor(y / step) % 2) * np.float32(step / 2.0)
    dx, dy = x % np.float32(step) - np.float32(step / 2.0), y % np.float32(step) - np.float32(step / 2.0)
    return np.clip(np.float32(r + 0.5) - np.sqrt(dx * dx + dy * dy), 0.0, 1.0).astype(np.float32)


def _over(img, pic, cover, x0, y0):
    """`img` (in place) with `pic` laid by its coverage `cover` (h, w), its top left at pixel (x0, y0), cut at the edge."""
    H, W = img.shape[:2]
    h, w = cover.shape
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if bx <= ax or by <= ay:
        return
    a = cover[ay - y0:by - y0, ax - x0:bx - x0, None]
    img[ay:by, ax:bx] = img[ay:by, ax:bx] * (1.0 - a) + pic[ay - y0:by - y0, ax - x0:bx - x0] * a


# ================================================================================================= the strip
def paper(strip, fx, px):
    """`strip` (in place) with the paper's polka dots in the palette's light colour, `dots` px at 1080 across."""
    pk = fx["purikura"]
    r = 0.5 * pk["dots"] * px
    if r < 0.5:
        return strip
    m = polka(strip.shape[1], strip.shape[0], round(r, 2))
    PF._lay(strip, m, 0, 0, pk["light"], DOT_ALPHA)
    return strip


def slot(strip, at, size, paper_):
    """`strip` (in place) with an empty photo slot of `size` (w, h) px at `at` (its top left): the paper a little
    darker, rounded as the photos are."""
    w, h = size
    PF._lay(strip, rounded(w, h, round(CORNER * min(w, h), 2)), at[0], at[1], np.asarray(paper_, np.float32) * SLOT)
    return strip


def beauty(pic, pk, px):
    """A capture through purikura's beauty filter: lifted toward its lights (`lift`, its key no further than PK.KEY: a
    light capture keeps its picture), soft-focused (`soft`: its lights spread into the darks round them, skin smoothed)
    and its whites blooming in `glow` (`bloom`)."""
    x = np.clip(np.asarray(pic, np.float32), 0.0, 1.0)
    if pk["lift"] > 0:
        x = 1.0 - (1.0 - x) ** np.float32(PK.lift_power(pk["lift"], float((x @ PF.LUMA).mean())))
    if pk["soft"] > 0:
        b = cv2.GaussianBlur(x, (0, 0), max(DIFFUSE * px, 0.5))
        s = np.float32(pk["soft"])
        x = x + (np.maximum(x, b) - x) * s
        x = x + (b - x) * (s * np.float32(SMOOTH))
    if pk["bloom"] > 0:
        x = ERF._screen(x, np.float32(pk["bloom"]) * ERF.glow(x, px) * np.asarray(pk["glow"], np.float32))
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def framed(pic, pk, j, px):
    """(card, coverage): photo `j` (h, w, 3) in its frame: a band `frame` px at 1080 wide over its edge in its pastel
    (the frames in turn), a row of dots in the palette's light colour along the band's middle, the corners rounded."""
    h, w = pic.shape[:2]
    r = round(CORNER * min(w, h), 2)
    card = np.array(pic, np.float32, copy=True)
    fw = pk["frame"] * px
    if fw >= 0.5:
        band = (1.0 - rounded(w, h, r, round(fw, 2)))[..., None]
        card = card * (1.0 - band) + np.asarray(pk["frames"][j % len(pk["frames"])], np.float32) * band
        m = np.zeros((h, w), np.uint8)
        e, gap, rad = fw / 2.0, max(STITCH * fw, 3.0), max(STITCH_SIZE * fw, 0.6)
        for x0, y0, x1, y1 in ((r, e, w - r, e), (r, h - e, w - r, h - e), (e, r, e, h - r), (w - e, r, w - e, h - r)):
            n = max(int(math.floor(math.hypot(x1 - x0, y1 - y0) / gap)), 1)
            for u in np.linspace(0.0, 1.0, n + 1):
                c = PF._q((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
                cv2.circle(m, (int(c[0]), int(c[1])), int(round(rad * 16)), 255, -1, cv2.LINE_AA, 4)
        PF._lay(card, m.astype(np.float32) / 255.0, 0, 0, pk["light"])
    return card, rounded(w, h, r)


def _drop(strip, card, cover, cell, s, lift, px):
    """`strip` (in place) with a framed photo (`card` and its coverage) dropping into `cell` ((x, y) of its top left):
    `s` times its size about the cell's middle, its shadow `lift` (0 .. 1) below it, as a plain booth's card drops."""
    ph, pw = cover.shape
    if abs(s - 1.0) > 1e-3:
        size = (max(int(round(pw * s)), 1), max(int(round(ph * s)), 1))
        how = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        card, cover = cv2.resize(card, size, interpolation=how), cv2.resize(cover, size, interpolation=how)
    ch, cw = cover.shape
    x0, y0 = int(round(cell[0] + pw / 2.0 - cw / 2.0)), int(round(cell[1] + ph / 2.0 - ch / 2.0))
    if lift > 0.01:
        off, blur = int(round(lift * 0.08 * ph)), max(6.0 * px, 0.5)
        pad = int(math.ceil(3 * blur))
        sh = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
        sh[pad:pad + ch, pad:pad + cw] = cover
        PF._lay(strip, cv2.GaussianBlur(sh, (0, 0), blur), x0 - pad + off, y0 - pad + off, (0.0, 0.0, 0.0),
                0.35 * min(lift * 2.0, 1.0))
    _over(strip, card, cover, x0, y0)


# ================================================================================================= stamps and the pen
@lru_cache(maxsize=64)
def parts(shape):
    """How a stamp draws each stroke of doodle `shape` (PK.stamp_parts)."""
    return tuple(PK.stamp_parts(PC.shape(shape)))


def stamp(img, strokes, kinds, colour, tint, width, outline):
    """`img` (in place) with a doodle's `strokes` ([(n, 2) px]) as a sticker (`kinds`: each stroke's part, `parts`): its
    closed outlines filled in `colour`, its open strokes drawn in it, the holes in its fills and the lines across them
    in `tint`, all of it on a white rim `outline` px wide, as the doodle pen's rim."""
    h, w = img.shape[:2]
    keep = [(p, k) for p, k in zip(strokes, kinds) if len(p) >= 2]
    if not keep:
        return img
    pts = np.concatenate([p for p, _ in keep])
    pad = width / 2.0 + outline + 3.0
    x0, y0 = max(int(math.floor(pts[:, 0].min() - pad)), 0), max(int(math.floor(pts[:, 1].min() - pad)), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max() + pad)) + 1, w), min(int(math.ceil(pts[:, 1].max() + pad)) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return img
    q = {k: [PF._q(p - (x0, y0)) for p, kk in keep if kk == k] for k in ("fill", "hole", "line", "mark")}
    rim, body, light = (np.zeros((y1 - y0, x1 - x0), np.uint8) for _ in range(3))
    if outline > 0:
        cv2.polylines(rim, [PF._q(p - (x0, y0)) for p, _ in keep], False, 255, max(int(round(width + 2.0 * outline)), 1),
                      cv2.LINE_AA, 4)
    if q["fill"]:
        cv2.fillPoly(body, q["fill"], 255, cv2.LINE_AA, 4)
    if q["fill"] or q["line"]:
        cv2.polylines(body, q["fill"] + q["line"], False, 255, max(int(round(width)), 1), cv2.LINE_AA, 4)
    if q["hole"]:
        cv2.fillPoly(light, q["hole"], 255, cv2.LINE_AA, 4)
    if q["hole"] or q["mark"]:
        cv2.polylines(light, q["hole"] + q["mark"], False, 255, max(int(round(0.6 * width)), 1), cv2.LINE_AA, 4)
    reg = img[y0:y1, x0:x1]
    a, b, c = (m.astype(np.float32)[..., None] / 255.0 for m in (rim, body, light))
    reg = reg * (1.0 - a) + a
    reg = reg * (1.0 - b) + np.asarray(colour, np.float32) * b
    img[y0:y1, x0:x1] = reg * (1.0 - c) + np.asarray(tint, np.float32) * c
    return img


def write(img, w, origin, t, rng):
    """`img` (in place) with what the pen has written of `w` (PK.writing) by clip second `t` on the photo whose top
    left is pixel `origin`: the hand's strokes boiled by `rng` (one boil step's draw), cut where the pen has got to,
    drawn in its colour over a white rim as the booth's doodles are."""
    u = (t - w["t"]) / w["dur"]
    if u <= 0:
        return img
    unit = HD.strokes(w["text"], w["seed"])
    pen = PK.PEN
    boiled = PC.boil(unit, rng, pen["wobble"], pen["turn"], pen["grow"])
    done = PC.turned(HD.written(boiled, min(u, 1.0), HD.lengths(unit)), w["angle"])
    at = np.array([origin[0] + w["at"][0], origin[1] + w["at"][1]])
    return PF.draw_strokes(img, [at + p * w["cap"] for p in done], w["colour"], max(pen["width"] * w["cap"], 1.0),
                           pen["outline"] * w["cap"])


def photo(strip, fx, j, pic, cell, t, t_capture, px):
    """`strip` (in place) with capture `j` of a purikura booth (`pic`, taken at clip second `t_capture`) at clip second
    `t`: through the beauty filter in its frame, dropping into `cell` ((x, y) of its top left), its stamps popping in
    once it has landed (boiling as the booth's doodles boil) and the pen writing on it."""
    pk = fx["purikura"]
    s, lift = PC.drop(t - t_capture, fx["drop"])
    _drop(strip, *framed(beauty(pic, pk, px), pk, j, px), cell, s, lift, px)
    size = (pic.shape[1], pic.shape[0])
    pen = PC.BOOTH_PEN
    b = max(int(math.floor((t - fx["from"]) * pen["fps"] + 1e-9)), 0)
    for i, d in enumerate(PK.stamps(fx, j, size, t_capture)):
        rng = np.random.default_rng([fx["seed"], fx["index"], 33, j, i, b])
        strokes, k = PF.doodle_strokes(d["shape"], (cell[0] + d["at"][0], cell[1] + d["at"][1]), d["size"], d["angle"],
                                       t - d["t"], rng, pen["wobble"] * px)
        stamp(strip, strokes, parts(d["shape"]), d["colour"], PK.mix(d["colour"], pk["light"], PK.TINT),
              max(pen["width"] * px * min(k, 1.0), 1.0), pen["outline"] * px * min(k, 1.0))
    w = PK.writing(fx, j, size, t_capture)
    if w is not None:
        write(strip, w, cell, t, np.random.default_rng([fx["seed"], fx["index"], 34, j, b]))
    return strip
