"""Zaps on the finished pictures (docs/design.md: Frame effects, Zaps): the picture work of the `[[fx]]` kind `zap` that
mkmmd.core.zap checks and times. ZapFx is the part of mkmmd.framefx.FrameFx that runs it. Its own pass draws the ghosts
on the picture as it reached the zap (put the zap before its sheets in the table: the ghosts lie under her); every sheet
lays its live sticker through ZapFx._lay_live, which around a zap catches the picture a ghost is made of, tears the
sticker and streaks it along the line it zapped. Images are float RGB in display space (0..1), (h, w, 3); a coverage is
(h, w) in 0..1; a sticker is its premultiplied face (its die-cut rim in it) and its coverage, the frame's size."""
import math

import cv2
import numpy as np

from .core import echoes as EC
from .core import framefx as FX
from .core import zap as ZP

TONE = 0.35            # how far a sliding slice's face goes toward its ink at a full tear (her colours coming apart)
INK = 0.9              # the inks' opacity at a full tear (they come up as the square root of the amount)
SCALE = 0.25           # the picture under a ghost is followed at this share of the frame's size (optical flow)
MARGIN = 0.12          # a ghost's layer reaches this share of the frame's height past where it lies (its drag)
CAP = 0.35             # a sticker taller than this share of the frame's height tears as one this tall
EDGE_IN = 3            # px into the sticker a streak's row takes its colour from, past its trailing edge
DRAG = 0.06            # the farthest a ghost's block is dragged in one frame, a share of the frame's height
LIGHT = 0.25           # how far a streak's rows go toward white: light rows read on any ground


def _ff():
    from . import framefx as FF          # framefx imports this module for ZapFx: its helpers are there by now
    return FF


def box_of(a, pad=0):
    """(x0, y0, x1, y1) px of what coverage `a` covers (over 1/512), `pad` px more, inside it; None for nothing."""
    x, y, w, h = cv2.boundingRect((np.asarray(a) > 1.0 / 512.0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    H, W = a.shape[:2]
    return max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)


# ================================================================================================= the tear
def torn(face, ring, cuts, amount, cap=None):
    """A sticker (premultiplied `face`, coverage `ring`) cut into `cuts` (mkmmd.core.zap.slices': top and bottom shares
    of its box's height, a slide in its heights) and each slice moved sideways `amount` x its slide, in whole pixels (a
    digital tear): ((x0, y0) the frame px of the result's corner, its face, its coverage, the px a slide is a share of:
    the box's height, or `cap` px when the box is taller: a large sticker tears as one that tall), or None when the
    sticker covers nothing."""
    b = box_of(ring)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    hh = y1 - y0
    unit = float(hh if cap is None else min(hh, cap))
    moves = [int(round(amount * s * unit)) for _, _, s in cuts]
    reach = max([abs(m) for m in moves] + [0])
    cx0, cx1 = x0 - reach, x1 + reach
    f = np.zeros((hh, cx1 - cx0, 3), np.float32)
    a = np.zeros((hh, cx1 - cx0), np.float32)
    for (top, bottom, _), m in zip(cuts, moves):
        r0, r1 = int(round(top * hh)), int(round(bottom * hh))
        if r1 > r0:
            f[r0:r1, x0 + m - cx0:x1 + m - cx0] = face[y0 + r0:y0 + r1, x0:x1]
            a[r0:r1, x0 + m - cx0:x1 + m - cx0] = ring[y0 + r0:y0 + r1, x0:x1]
    return (cx0, y0), f, a, unit


def tone(f, a, cuts, amount, inks):
    """The torn face `f` (coverage `a`) with each sliding slice's colours gone TONE x `amount` toward the ink of the way
    it slides (the first ink left, the second right): her colours coming apart into the inks. In place."""
    hh = f.shape[0]
    for top, bottom, s in cuts:
        if abs(s) <= 1e-9:
            continue
        r0, r1 = int(round(top * hh)), int(round(bottom * hh))
        k = np.float32(TONE * amount * min(abs(s) / max(abs(c[2]) for c in cuts), 1.0))
        ink = np.asarray(inks[0] if s < 0 else inks[1], np.float32)
        f[r0:r1] += (ink * a[r0:r1, :, None] - f[r0:r1]) * k
    return f


def lay(out, corner, f, a, shadow=None, inks=None, split=0):
    """`out` (in place) with a torn sticker (face `f`, coverage `a`, its corner at frame px `corner`) laid over it: its
    drop `shadow` (colour, (dx, dy) px, blur px, alpha) cast by the torn shape, two copies of that shape in the `inks`
    `split` px to the left and to the right under it (each at INK), then its face. Parts off the picture are left out."""
    FF = _ff()
    H, W = out.shape[:2]
    reach = split + (0 if shadow is None else int(math.ceil(abs(shadow[1][0]) + abs(shadow[1][1]) + 3.0 * shadow[2])))
    x0, y0 = corner
    h, w = a.shape
    rx0, ry0, rx1, ry1 = max(x0 - reach, 0), max(y0 - reach, 0), min(x0 + w + reach, W), min(y0 + h + reach, H)
    if rx1 <= rx0 or ry1 <= ry0:
        return out
    cov = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
    rgb = np.zeros((ry1 - ry0, rx1 - rx0, 3), np.float32)
    sx0, sy0 = max(rx0 - x0, 0), max(ry0 - y0, 0)
    sx1, sy1 = min(rx1 - x0, w), min(ry1 - y0, h)
    if sx1 > sx0 and sy1 > sy0:
        cov[y0 + sy0 - ry0:y0 + sy1 - ry0, x0 + sx0 - rx0:x0 + sx1 - rx0] = a[sy0:sy1, sx0:sx1]
        rgb[y0 + sy0 - ry0:y0 + sy1 - ry0, x0 + sx0 - rx0:x0 + sx1 - rx0] = f[sy0:sy1, sx0:sx1]
    region = out[ry0:ry1, rx0:rx1]
    if shadow is not None and shadow[3] > 0:
        region = FF._cast(region, cov, shadow)
    if inks is not None and split > 0:
        for ink, dx in ((inks[0], -split), (inks[1], split)):
            k = (FF.plate_shift(cov, dx, 0.0) * np.float32(INK))[..., None]
            region = region * (1.0 - k) + np.asarray(ink, np.float32) * k
    out[ry0:ry1, rx0:rx1] = rgb + region * (1.0 - cov[..., None])
    return out


# ================================================================================================= the streak
def streak(out, face, ring, frm, reach, alpha, rows, width, tint, inks, rng, back=0.0):
    """`out` (in place) with thin rows of a sticker's pixels (premultiplied `face`, coverage `ring`, the frame's size)
    stretched back along the line from frame px `frm` (where it was) to its middle: across the line its rows are cut
    in runs `width` px high (from half to one and a half times that), `rows` of them drawn, each in the colour of the
    sticker just inside its trailing edge (the side it came from) mixed `tint` toward an ink (the two in turn), from
    `reach` of the way back to where it was (a little more or less, row by row) up to that edge, coming up from nothing
    there, at `alpha` (each row its own share of it). Drawn under the sticker: lay the sticker after it."""
    b = box_of(ring)
    if b is None or alpha <= 0 or reach <= 0:
        return out
    m = cv2.moments(np.ascontiguousarray(ring[b[1]:b[3], b[0]:b[2]], np.float32))
    if m["m00"] < 1.0:
        return out
    c1 = np.array([b[0] + m["m10"] / m["m00"], b[1] + m["m01"] / m["m00"]])
    d = c1 - np.asarray(frm, float)
    L = float(np.hypot(*d))
    if L < 4.0:
        return out
    u, n = d / L, np.array([-d[1], d[0]]) / L
    R = int(math.ceil(math.hypot(b[2] - b[0], b[3] - b[1]) / 2.0)) + 2
    Lc, Hc = int(math.ceil(L)) + 2 * R, 2 * R
    A = np.array([[u[0], u[1]], [n[0], n[1]]])
    M = np.hstack([A, (np.array([L + R, R]) - A @ c1)[:, None]]).astype(np.float32)     # frame px -> along, across
    fc = cv2.warpAffine(np.ascontiguousarray(face, np.float32), M, (Lc, Hc), flags=cv2.INTER_LINEAR)
    ac = cv2.warpAffine(np.ascontiguousarray(ring, np.float32), M, (Lc, Hc), flags=cv2.INTER_LINEAR)
    on = ac > 0.5
    has = on.any(1)
    edge = np.argmax(on, 1)
    rgb = np.zeros((Hc, Lc, 3), np.float32)
    a = np.zeros((Hc, Lc), np.float32)
    cols = np.arange(Lc, dtype=np.float32)
    j, flip = 0, 0
    while j < Hc:
        run = max(1, int(round(width * rng.uniform(0.5, 1.5))))
        keep, share, late = rng.random() < rows, rng.uniform(0.55, 1.0), rng.uniform(-0.08, 0.08)
        for r in range(j, min(j + run, Hc)):
            if not (keep and has[r]):
                continue
            e = int(edge[r])
            k = min(e + EDGE_IN, Lc - 1)
            cov = float(ac[r, k])
            if cov < 0.5:
                continue
            colour = fc[r, k] / cov
            colour = colour + (np.asarray(inks[flip % 2], np.float32) - colour) * np.float32(tint)
            colour = colour + (1.0 - colour) * np.float32(LIGHT)                 # a light streak reads on any ground
            tail = (L + R) - (reach + late) * (L + back)
            if tail >= e:
                continue
            w = np.clip((cols - tail) / max(e - tail, 1.0), 0.0, 1.0) * (cols < e)
            row_a = (w ** 0.3) * np.float32(alpha * share)
            a[r] = row_a
            rgb[r] = colour[None, :] * row_a[:, None]
        if keep:
            flip += 1
        j += run
    if not a.any():
        return out
    H, W = out.shape[:2]
    corners = np.array([[0, 0], [Lc, 0], [0, Hc], [Lc, Hc]], float)
    Ainv = np.linalg.inv(A)
    pts = (corners - np.array([L + R, R])) @ Ainv.T + c1
    x0, y0 = max(int(math.floor(pts[:, 0].min())), 0), max(int(math.floor(pts[:, 1].min())), 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max())) + 1, W), min(int(math.ceil(pts[:, 1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return out
    Mb = M.astype(np.float64).copy()
    Mb[:, 2] += Mb[:, :2] @ np.array([x0, y0], float)                    # the box's px -> along, across
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    back = cv2.warpAffine(np.dstack([rgb, a]), Mb.astype(np.float32), (x1 - x0, y1 - y0), flags=flags)
    region = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = back[..., :3] + region * (1.0 - np.clip(back[..., 3:], 0.0, 1.0))
    return out


# ================================================================================================= the ghost
def ghost_start(caught, cuts, tear, ink, tint, size, margin):
    """A ghost's layer on its zap's frame from the sticker `caught` on the frame before ((face, coverage) the frame's
    size): torn `tear` of a full tear by `cuts` and frozen so, its colours gone `tint` toward `ink`, in a canvas
    reaching `margin` px past it on every side (inside a picture of `size`): ((x0, y0) its corner, premultiplied rgb,
    coverage), or None."""
    got = torn(caught[0], caught[1], cuts, tear, CAP * size[1])
    if got is None:
        return None
    (x0, y0), f, a, _ = got
    f = f + (np.asarray(ink, np.float32) * a[..., None] - f) * np.float32(tint)
    W, H = size
    cx0, cy0 = max(x0 - margin, 0), max(y0 - margin, 0)
    cx1, cy1 = min(x0 + a.shape[1] + margin, W), min(y0 + a.shape[0] + margin, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    rgb = np.zeros((cy1 - cy0, cx1 - cx0, 3), np.float32)
    cov = np.zeros((cy1 - cy0, cx1 - cx0), np.float32)
    sx0, sy0 = max(cx0 - x0, 0), max(cy0 - y0, 0)
    sx1, sy1 = min(cx1 - x0, a.shape[1]), min(cy1 - y0, a.shape[0])
    rgb[y0 + sy0 - cy0:y0 + sy1 - cy0, x0 + sx0 - cx0:x0 + sx1 - cx0] = f[sy0:sy1, sx0:sx1]
    cov[y0 + sy0 - cy0:y0 + sy1 - cy0, x0 + sx0 - cx0:x0 + sx1 - cx0] = a[sy0:sy1, sx0:sx1]
    return (cx0, cy0), rgb, cov


def grey(img):
    """The picture as the ghost's drag follows it: its luma at SCALE of its size, 8 bits."""
    h, w = img.shape[:2]
    small = cv2.resize(np.ascontiguousarray(img, np.float32), (max(8, int(round(w * SCALE))),
                                                                max(8, int(round(h * SCALE)))),
                       interpolation=cv2.INTER_AREA)
    return (np.clip(small @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def drag(layer, prev, cur, block):
    """A ghost's layer ((x0, y0), rgb, coverage) carried one frame along the motion of the picture under it, from
    `prev` to `cur` (grey()'s): Farneback optical flow averaged over `block` px squares of the frame (no block moving
    more than DRAG of the frame's height a frame: a cut under it smears it, it does not fling it), each pixel taking
    what stood its block's move behind it (a codec's motion vectors, as a mosh moves the picture)."""
    (x0, y0), rgb, cov = layer
    h, w = cov.shape
    sh, sw = cur.shape
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    b = max(1, int(round(block * SCALE)))
    gh, gw = -(-sh // b), -(-sw // b)
    pad = np.pad(flow, ((0, gh * b - sh), (0, gw * b - sw), (0, 0)), mode="edge")
    blocks = pad.reshape(gh, b, gw, b, 2).mean((1, 3)) / np.float32(SCALE)          # px of the frame a frame
    cap = np.float32(DRAG * sh / SCALE)
    blocks = np.clip(blocks, -cap, cap)
    full = b / SCALE
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bi = np.clip(((ys + y0) / full).astype(np.int64), 0, gh - 1)
    bj = np.clip(((xs + x0) / full).astype(np.int64), 0, gw - 1)
    mv = blocks[bi, bj]
    rgba = np.dstack([rgb, cov])
    out = cv2.remap(rgba, xs - mv[..., 0], ys - mv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return (x0, y0), np.ascontiguousarray(out[..., :3]), np.ascontiguousarray(out[..., 3])


def blocks_laid(out, layer, size, block, alive, alpha, offsets=None):
    """`out` (in place) with a ghost's layer ((x0, y0), rgb, coverage) laid over it block by block, the frame cut into
    `block` px squares: a block with `alive` False (an array over the frame's blocks) is left out, the others laid at
    `alpha` times their own opacity (an array over the blocks, or a number), each moved by its `offsets` ((rows, cols,
    2) whole px; None: none)."""
    (x0, y0), rgb, cov = layer
    H, W = out.shape[:2]
    h, w = cov.shape
    by0, by1 = int(y0 // block), int(math.ceil((y0 + h) / block))
    bx0, bx1 = int(x0 // block), int(math.ceil((x0 + w) / block))
    for by in range(by0, by1):
        ya, yb = max(int(round(by * block)), y0), min(int(round((by + 1) * block)), y0 + h)
        if yb <= ya:
            continue
        for bx in range(bx0, bx1):
            if not alive[by, bx]:
                continue
            k = float(alpha if np.isscalar(alpha) else alpha[by, bx])
            if k <= 0.0:
                continue
            xa, xb = max(int(round(bx * block)), x0), min(int(round((bx + 1) * block)), x0 + w)
            if xb <= xa:
                continue
            c = cov[ya - y0:yb - y0, xa - x0:xb - x0]
            if not c.any():
                continue
            dx, dy = (0, 0) if offsets is None else (int(offsets[by, bx, 0]), int(offsets[by, bx, 1]))
            ta, tb, sa, sb = ya + dy, yb + dy, xa + dx, xb + dx
            ca, cb, da, db = max(ta, 0), min(tb, H), max(sa, 0), min(sb, W)
            if cb <= ca or db <= da:
                continue
            src_c = c[ca - ta:cb - ta, da - sa:db - sa] * np.float32(k)
            src_f = rgb[ya - y0 + ca - ta:ya - y0 + cb - ta, xa - x0 + da - sa:xa - x0 + db - sa] * np.float32(k)
            region = out[ca:cb, da:db]
            out[ca:cb, da:db] = src_f + region * (1.0 - src_c[..., None])
    return out


# ================================================================================================= over a cut
class ZapFx:
    """The zaps' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t, _get,
    _apply, _reached and _seen), and the live sticker's: every sheet lays its live sticker through _lay_live. A zap
    draws nothing in its own place in the table: _lay_live draws the ghosts and the streaks of the zaps that name the
    sheet under the sticker (over a table's own clip) and tears it; it keeps the stickers a sheet lays for the trails
    that follow it (live_sticker, mkmmd.echoes). `_zap_init` is called once its effects are known."""

    def _zap_init(self):
        self._zaps_of = {}           # a sheet's index -> the zaps (this output's entries) that name it
        self._zap_entries = {fx["index"]: fx for fx in self.fxs}
        self._live_reach = {}        # a sheet's index -> how many frames back the trails that follow it reach
        for fx in self.fxs:
            if fx["kind"] in ZP.KINDS:
                for i, _, _ in fx["laid"]:
                    self._zaps_of.setdefault(i, []).append(fx)
            if fx["kind"] == "trails" and fx.get("sheet_index") is not None:
                i = fx["sheet_index"]
                self._live_reach[i] = max(self._live_reach.get(i, 0), EC.reach(fx, self.fps))
        self._live_kept = {}         # a sheet's index -> {frame: the sticker it laid there, 8 bits}, its last frames
        self._echo_paths = {}        # a train's index -> how far along her path it has followed her (mkmmd.echoes)
        self._zap_caught = {}        # (zap index, zap j) -> the sticker its sheet laid on the frame before the zap
        self._zap_layers = {}        # (zap index, zap j) -> (frame, its ghost's layer dragged to that frame)
        self._zap_grounds = {}       # zap index -> (frame, grey() of what lay under its ghosts there)
        self._zap_deaths = {}        # (zap index, zap j) -> its blocks' deaths and drifts
        self._zap_quiet = None       # a sheet run again on another frame: what to keep from it ({what, frame, ..})

    def _zap(self, fx, frame, img):
        """A zap draws in the sheets it names (_lay_live): its own place in the table leaves the picture as it is."""
        return img

    # ---------------------------------------------------------------- her sticker, laid by its sheet
    def _lay_live(self, fx, frame, out, face, ring, shadow):
        """A sheet's live sticker (premultiplied `face`, coverage `ring`, the frame's size) laid on `out` under its drop
        `shadow` (colour, (dx, dy) px, blur px, alpha): as it is, or with the zaps that name the sheet: their ghosts on
        `out` first (where it is the last of their sheets that lays her on the frame), the sticker caught on the frame
        before a zap it leaves, streaked along the line it zapped over the frames after one it arrives from, and torn
        over a zap's lead and snap. Every sheet lays its sticker so (mkmmd.sheet, mkmmd.table)."""
        FF = _ff()
        zaps = [z for z in self._zaps_of.get(fx["index"], ()) if z["f0"] <= frame < z["f1"]]
        q = self._zap_quiet
        if q is not None:                                    # run again for her sticker or for what lies under her
            if q["frame"] == frame and q["what"] == "sticker" and q["sheet"] == fx["index"]:
                q["got"] = _caught(face, ring)
            for z in zaps:
                if q["frame"] == frame and q["what"] == "ground" and q["zap"] == z["index"] and \
                        ZP.drawer(z, frame) == fx["index"]:
                    q["got"] = grey(out)
            zaps = []
        elif fx["index"] in self._live_reach:
            self._live_keep(fx["index"], frame, _caught(face, ring))
        tears = []
        if zaps:
            out = np.array(out, np.float32)                  # its own: a table keeps the ground it was handed
        for z_fx in zaps:
            s = float(np.clip(FX.strength(z_fx, self._t(frame)), 0.0, 1.0))
            if ZP.drawer(z_fx, frame) == fx["index"]:
                live = [(j, z, ZP.ghost_age(z, frame)) for j, z in enumerate(z_fx["zaps"])]
                live = [(j, z, a) for j, z, a in live if a is not None]
                if live:
                    now = grey(out)
                    for j, z, age in live:
                        layer = self._zap_layer(z_fx, j, z, frame, now)
                        if layer is not None and s > 0.0:
                            self._zap_ghost(z_fx, j, z, frame, age, layer, s, out)
                    self._zap_grounds[z_fx["index"]] = (frame, now)
            for j, z in enumerate(z_fx["zaps"]):
                if not z["fail"] and frame == z["frame"] - 1 and z["before"] == fx["index"]:
                    self._zap_caught[(z_fx["index"], j)] = _caught(face, ring)
                k = frame - z["frame"]
                if not z["fail"] and z["after"] == fx["index"] and 0 <= k < z_fx["streak"]["frames"] and s > 0.0:
                    caught = self._zap_catch(z_fx, j)
                    if caught is not None:
                        reach, alpha = ZP.streak_reach(z_fx, k)
                        st = z_fx["streak"]
                        rng = np.random.default_rng([z_fx["seed"], z_fx["index"], j, frame, 17])
                        streak(out, face, ring, _middle(caught), reach, alpha * s, st["rows"], st["width"] * self.px,
                               st["tint"], z_fx["tear"]["inks"], rng, 0.5 * caught[2].shape[0])
            for j, side, amount in ZP.tear_at(z_fx, frame):
                z = z_fx["zaps"][j]
                if (side == "before" and z["before"] == fx["index"]) or (side == "after" and z["after"] == fx["index"]):
                    tears.append(((ZP.fail_amount(z_fx, z, frame) if z["fail"] else amount) * s, z_fx, j))
        if tears:
            amount, z_fx, j = max(tears, key=lambda x: x[0])
            if amount > 0.0:
                cuts = ZP.slices(z_fx, j, frame)
                got = torn(face, ring, cuts, amount, CAP * self.size[1])
                if got is not None:
                    corner, f, a, hh = got
                    tone(f, a, cuts, amount, z_fx["tear"]["inks"])
                    split = int(round(amount * z_fx["tear"]["split"] * hh))
                    return lay(out, corner, f, a, shadow, z_fx["tear"]["inks"], split)
        if shadow is not None and shadow[3] > 0:
            out = FF._cast(out, ring, shadow)
        return face + out * (1.0 - ring[..., None])

    def _sticker_again(self, sheet, g):
        """The sticker `sheet` lays on frame `g` ((box, face, coverage) in it, or None), the sheet run again on it
        quietly (every sticker laid as it is, no zap drawn) over the cut's frame there: its figure comes from its own
        plate, or from that frame as it was rendered."""
        pic = self._get(g)
        q = {"what": "sticker", "sheet": sheet["index"], "frame": g, "got": None}
        prev, self._zap_quiet = self._zap_quiet, q
        try:
            self._apply(sheet, g, np.zeros((self.size[1], self.size[0], 3), np.float32) if pic is None else pic)
        finally:
            self._zap_quiet = prev
        return q["got"]

    def _zap_catch(self, fx, j):
        """The sticker zap j of `fx` leaves ((box, face, coverage) on the frame before the zap), as its sheet laid it
        there: kept from that frame, or that sheet run again on it (frames out of order)."""
        key = (fx["index"], j)
        if key not in self._zap_caught:
            got = self._sticker_again(self._zap_entries[fx["zaps"][j]["before"]], fx["zaps"][j]["frame"] - 1)
            if got is not None:
                self._zap_caught[key] = got
        return self._zap_caught.get(key)

    def live_sticker(self, sheet, g):
        """The live sticker `sheet` (an entry of this output) lays on frame `g`: (its box (x0, y0, x1, y1), its
        premultiplied face and its coverage in it, kept to 8 bits), or None where it lays none: as it laid it there
        (kept for the trails that follow it), else the sheet run again on that frame."""
        kept = self._live_kept.setdefault(sheet["index"], {})
        if g not in kept:
            if not sheet["f0"] <= g < sheet["f1"] or sheet.get("objects") is None:
                return None
            self._live_keep(sheet["index"], g, self._sticker_again(sheet, g))
        got = kept.get(g)
        if got is None:
            return None
        box, f, a = got
        return box, f.astype(np.float32) / np.float32(255.0), a.astype(np.float32) / np.float32(255.0)

    def _live_keep(self, i, g, got):
        """Sheet i's sticker on frame `g` kept (8 bits) for the trails that follow it, as far back as they reach."""
        kept = self._live_kept.setdefault(i, {})
        kept[g] = None if got is None else (got[0], np.clip(got[1] * 255.0 + 0.5, 0, 255).astype(np.uint8),
                                            np.clip(got[2] * 255.0 + 0.5, 0, 255).astype(np.uint8))
        reach = self._live_reach.get(i, 0) + 1
        for old in [f for f in kept if abs(f - g) > reach]:
            del kept[old]

    def _zap_ground(self, fx, g):
        """grey() of what lies under zap `fx`'s ghosts on frame `g` (frames out of order): the sheet that draws them
        there run again on the cut's frame `g` as it reached that sheet; None where the cut has no frame `g`."""
        sheet = self._zap_entries[ZP.drawer(fx, g)]
        q = {"what": "ground", "zap": fx["index"], "frame": g, "got": None}
        prev, self._zap_quiet = self._zap_quiet, q          # quiet already while the effects before it are applied
        try:
            pic = self._reached(sheet, g, None, None)
            if pic is not None:
                self._apply(sheet, g, pic)
        finally:
            self._zap_quiet = prev
            self._seen.get(sheet["index"], {}).pop(g, None)
        return q["got"]

    # ---------------------------------------------------------------- the ghosts
    def _zap_layer(self, fx, j, z, frame, now):
        """Ghost j's layer dragged to `frame` (`now`: grey() of what lies under it there): one frame on from the one
        before when that was the last drawn, else from its zap's frame again, what lay under it on each frame between
        worked out again."""
        key = (fx["index"], j)
        got = self._zap_layers.get(key)
        if got is not None and got[0] == frame:
            return got[1]
        block = fx["ghost"]["block"] * self.px
        prev = self._zap_grounds.get(fx["index"])
        if got is not None and got[0] == frame - 1 and prev is not None and prev[0] == frame - 1:
            layer = None if got[1] is None else drag(got[1], prev[1], now, block)
        else:
            caught = self._zap_catch(fx, j)
            layer = None if caught is None else self._zap_start(fx, j, caught)
            last = None
            for g in range(z["frame"], frame):
                cur = self._zap_ground(fx, g)
                if layer is not None and last is not None and cur is not None:
                    layer = drag(layer, last, cur, block)
                last = cur
            if layer is not None and last is not None:
                layer = drag(layer, last, now, block)
        self._zap_layers[key] = (frame, layer)
        return layer

    def _zap_start(self, fx, j, caught):
        """Ghost j's layer on its zap's frame (ghost_start) from the sticker its sheet laid on the frame before."""
        (x0, y0, x1, y1), f, a = caught
        W, H = self.size
        face = np.zeros((H, W, 3), np.float32)
        ring = np.zeros((H, W), np.float32)
        face[y0:y1, x0:x1], ring[y0:y1, x0:x1] = f, a
        g = fx["ghost"]
        return ghost_start((face, ring), ZP.slices(fx, j), g["tear"], g["ink"], g["tint"], self.size,
                           int(round(MARGIN * H)))

    def _zap_ghost(self, fx, j, z, frame, age, layer, s, out):
        """Ghost j on `frame` (`age` frames after its zap) laid on `out` in place: its blocks that are still there, each
        wandering off on a drift of its own, jittering now and then, flying apart from a touch, at the ghost's opacity
        times the strength `s`."""
        W, H = self.size
        g = fx["ghost"]
        block = g["block"] * self.px
        gh, gw = int(math.ceil(H / block)) + 1, int(math.ceil(W / block)) + 1
        key = (fx["index"], j)
        if key not in self._zap_deaths:
            self._zap_deaths[key] = (ZP.block_deaths(fx, j, gh * gw).reshape(gh, gw), ZP.block_drift(fx, j, (gh, gw)))
        deaths, drift = self._zap_deaths[key]
        big = max(self._zap_catch(fx, j)[2].shape[0] / (CAP * H), 1.0)     # a large sticker's ghost: lighter, sooner gone
        u = min(age / max(z["ghost"], 1) * big ** 0.5, 1.0)
        alive = deaths > u
        rng = np.random.default_rng([fx["seed"], fx["index"], j, frame, 19])
        jit = np.rint(drift * (age / self.fps) * H).astype(np.float32)
        hop = rng.random((gh, gw)) < 0.06 + 0.3 * u
        jit[..., 0] += np.where(hop, np.rint(rng.uniform(-1.0, 1.0, (gh, gw)) * g["jitter"] * self.px), 0.0)
        a = np.full((gh, gw), ZP.ghost_alpha(fx, z, age) * s / big ** 0.75, np.float32)
        if z["touch_frame"] is not None and frame >= z["touch_frame"]:
            p = self._zap_touch(fx, j, z)
            cy, cx = (np.mgrid[0:gh, 0:gw].astype(np.float32) + 0.5) * np.float32(block)
            dx, dy = cx - p[0], cy - p[1]
            spin = np.random.default_rng([fx["seed"], fx["index"], j, 23]).uniform(-0.45, 0.45, (gh, gw))
            ang = np.arctan2(dy, dx) + spin
            speed = np.random.default_rng([fx["seed"], fx["index"], j, 29]).uniform(0.6, 1.4, (gh, gw))
            fly, keep = ZP.burst_at(fx, z, frame, self.fps, np.hypot(dx, dy) / H)
            jit[..., 0] += np.rint(np.cos(ang) * fly * speed * H)
            jit[..., 1] += np.rint(np.sin(ang) * fly * speed * H)
            a = a * keep.astype(np.float32)
        blocks_laid(out, layer, self.size, block, alive, a, jit)

    def _zap_touch(self, fx, j, z):
        """The frame px a ghost's touch bursts it from: its `at`, or the middle of the sticker it was made of."""
        at = z["touch"]["at"]
        if at is not None:
            return at[0] * self.size[0] - 0.5, at[1] * self.size[1] - 0.5
        return _middle(self._zap_catch(fx, j))


def _caught(face, ring):
    """What a zap keeps of a sticker (premultiplied `face`, coverage `ring`, the frame's size): (its box, face and
    coverage in it), or None when it covers nothing."""
    b = box_of(ring)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    return b, face[y0:y1, x0:x1].copy(), ring[y0:y1, x0:x1].copy()


def _middle(caught):
    """The frame px of the coverage-weighted middle of a caught sticker ((box, face, coverage))."""
    (x0, y0, _, _), _, a = caught
    m = cv2.moments(np.ascontiguousarray(a, np.float32))
    if m["m00"] < 1e-6:
        return x0 + a.shape[1] / 2.0, y0 + a.shape[0] / 2.0
    return x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"]
