"""Skies on the finished pictures (docs/design.md: Frame effects, Skies): the picture work of `[[fx]] kind = "sky"`, which
mkmmd.core.sky checks and times, and of a table's `smear`. SkyFx is the part of mkmmd.framefx.FrameFx that draws them.

A sky's segment is drawn at the output's own size: each layer's clip through its view and the sky's grade by warmth
(mkmmd.table's `carried` and `painted`, as a table draws its clip, with the layer's own `lean` and the stretch of its own
clip), or the clear sky's paper, each in its box of the frame. Through a smear the held picture is carried from frame to
frame along the new picture's motion (Farneback optical flow on pictures SMALL of the output's size, averaged over the
smear's blocks, `gain` times, moved by whole pixels so the blocks stay sharp) or a made-up flow
(mkmmd.framefx.mosh_flow), and the blocks of the new picture break through in an order seeded once (patches of
smooth noise and single blocks); a block that has broken is the new picture, carried on with the rest. The colours bleed
at the output only: the held picture's chroma (its colour less its luma) drifts toward the new one's, and a new block
keeps a little of the held one's chroma under it, both gone when the last block is through. A melt streaks the columns
of the picture (`melt_light`, `melt_all`), each column its own seeded share of the length. Images are float RGB in
display space (0..1), (h, w, 3)."""
import math

import cv2
import numpy as np

from . import fields as FL
from . import table as TBF
from .core import framefx as FX
from .core import sky as SY
from .core import table as TB

SMALL = 0.25             # a smear's motion is worked out on pictures this share of the output's size
LUMA = np.float32([0.2126, 0.7152, 0.0722])
GRAIN = 0.6 / 255.0      # a fine grain over the paper: a dither, so its gradient never bands
MOTTLE_CELL = 0.45       # frame heights: the size of the paper's soft clouds of light and shade
MOTTLE_DEPTH = 0.035     # how far the paper's clouds move its colour toward white and toward its shade, at mottle 1
TOOTH = 0.012            # the paper's tooth: how much its fine relief lights and shades it, at tooth 1
TOOTH_SIZE = 0.9         # px at 1080: the grain of that relief
BAND = (2.0, 14.0)       # px at 1080: the widths of the bands of columns a melt streaks alike
BAND_SHARE = (0.25, 1.0)  # each band's share of the melt's length
STRETCH = 4.0            # a light melt draws the column under its edge this many times longer
BLEED_HELD = 0.6         # how far a held block's chroma has drifted toward the new picture's when the last breaks,
BLEED_NEW = 0.5          # and how much of the held chroma a new block keeps on the frame it breaks, at bleed 1
KEEP_REACH = 0.02        # frame heights: a layer that keeps the dark keeps what is darker than the sky round it this
KEEP_RANGE = 0.12        # small (a bird), wholly once its luma is this much under that sky's


def _ff():
    from . import framefx as FF          # framefx imports this module for SkyFx: its helpers are there by now
    return FF


# ================================================================================================= pure parts
def paper(size, p, unit=None):
    """The clear sky (`p`: mkmmd.core.sky's paper, its colours resolved) as a picture of `size` (w, h): `top` at the top
    of the frame going smoothly to `bottom` at its foot, in soft clouds of light and shade `mottle` deep (MOTTLE_CELL
    frame heights across, `unit` px a frame height: the frame's own by default), the paper's `tooth` (a fine relief lit
    from the upper left) and a grain finer than a pixel."""
    W, H = int(size[0]), int(size[1])
    unit = float(unit or H)
    y = (np.arange(H, dtype=np.float32) + 0.5) / np.float32(H)
    k = (y * y * (3.0 - 2.0 * y))[:, None, None]
    top, bottom = np.asarray(p["top"], np.float32), np.asarray(p["bottom"], np.float32)
    out = np.broadcast_to(top + (bottom - top) * k, (H, W, 3)).astype(np.float32)
    if p["mottle"] > 0:
        gw, gh = max(int(math.ceil(W / 8.0)), 2), max(int(math.ceil(H / 8.0)), 2)
        cell = MOTTLE_CELL * unit / 8.0
        X, Y = np.meshgrid(np.arange(gw, dtype=np.float32) / np.float32(cell),
                           np.arange(gh, dtype=np.float32) / np.float32(cell))
        n = FL.noise(X, Y, 0.0, int(p["seed"])) - 0.5 + 0.5 * (FL.noise(X * 2.3, Y * 2.3, 0.0, int(p["seed"]) + 7) - 0.5)
        n = cv2.resize(n.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
        d = np.float32(MOTTLE_DEPTH * p["mottle"])
        out = out + np.where(n > 0, (1.0 - out) * n, out * n) * d * 2.0
    rng = np.random.default_rng([int(p["seed"]), 61])
    if p["tooth"] > 0:
        r = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), max(TOOTH_SIZE * unit / 1080.0, 0.5))
        lit = (np.roll(r, 1, 0) + np.roll(r, 1, 1) - 2.0 * r) / np.float32(max(float(r.std()), 1e-6))
        out = out * (1.0 + np.float32(TOOTH * p["tooth"]) * lit[..., None])
    return (out + rng.standard_normal((H, W, 1)).astype(np.float32) * np.float32(GRAIN)).astype(np.float32)


def jump_flow(jump, dur, fps, seed):
    """The made-up flow of a smear's `jump` ([dx, dy] frame heights, x right, y down: where she went at the zap): a drift
    that carries the held picture that far, that way, over the smear's `dur` (mkmmd.framefx.mosh_flow's keys)."""
    dx, dy = float(jump[0]), float(jump[1])
    return {"kind": "drift", "angle": math.degrees(math.atan2(-dy, dx)), "steps": 1, "jitter": 0.25, "stuck": 0.3,
            "amount": math.hypot(dx, dy) * 1080.0 / max(dur * fps, 1.0), "seed": int(seed)}


def area_mask(size, area, feather):
    """(h, w) float32 0..1: a layer's or a melt's box (`area`, frame fractions; None: the whole frame) with a smooth
    ramp `feather` px wide across its edges (not along the frame's own)."""
    W, H = size
    if area is None:
        return None
    from .picturefx import rect_mask
    return rect_mask((W, H), tuple(area), feather)


def chroma(img):
    """A picture's colour less its luma (each pixel's RGB minus its Rec. 709 luma)."""
    return img - (img @ LUMA)[..., None]


def kept_dark(pic, reach):
    """(h, w) float32 0..1: how much of a layer that keeps the dark (a flock against its sky) is kept, by how much darker
    each pixel is than its sky round it (the luma's largest over `reach` px, softened): wholly KEEP_RANGE under it."""
    L = np.ascontiguousarray(pic @ LUMA)
    k = max(int(round(reach)) | 1, 3)
    sky = cv2.GaussianBlur(cv2.dilate(L, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))), (0, 0), k / 2.0)
    return np.clip((sky - L) / np.float32(KEEP_RANGE), 0.0, 1.0)


def bands(width, seed, px):
    """(width,) float32: each column's share of a melt's length, alike over bands of BAND px at 1080 (seeded), so the
    melt runs in streaks; the same for every frame."""
    rng = np.random.default_rng([int(seed), 67])
    out = np.empty(width, np.float32)
    x = 0
    while x < width:
        n = max(int(round(rng.uniform(*BAND) * px)), 1)
        out[x:x + n] = rng.uniform(*BAND_SHARE)
        x += n
    return cv2.GaussianBlur(out[None, :], (0, 0), max(0.6 * px, 0.3))[0] if width > 2 else out


def melt_light(img, dir_, level, length, share):
    """`img` with what is darker than `level` (luma, 0..1) run into the light beside it along each column: `dir_` up (a
    tower's top stretched up into the sky above it) or down; each light pixel within `length` x `share` (px, a share for
    each column) of the nearest dark one below it (above it, down) shows that one's column read back from it STRETCH
    times slower, fading out to the streak's end."""
    h, w = img.shape[:2]
    L = img @ LUMA
    dark = L <= np.float32(level)
    rows = np.arange(h, dtype=np.int64)[:, None]
    reach = np.maximum(np.float32(length) * share, 1e-3)[None, :]
    if dir_ == "up":
        edge = np.minimum.accumulate(np.where(dark, rows, h)[::-1], axis=0)[::-1]           # the nearest dark row below
        d = (edge - rows).astype(np.float32)
        src = edge + d / np.float32(STRETCH)
        lit = (~dark) & (edge < h)
    else:
        edge = np.maximum.accumulate(np.where(dark, rows, -1), axis=0)                       # the nearest dark row above
        d = (rows - edge).astype(np.float32)
        src = edge - d / np.float32(STRETCH)
        lit = (~dark) & (edge >= 0)
    a = np.where(lit, np.clip(1.0 - d / reach, 0.0, 1.0), 0.0).astype(np.float32)
    a = a * a
    srcr = np.clip(np.rint(src), 0, h - 1).astype(np.int64)
    pulled = img[srcr, np.arange(w)[None, :]]
    return img + (pulled - img) * a[..., None]


def melt_all(img, length, share):
    """`img` smeared along each column: each pixel the mean of its column over `length` x `share` px about it (a share
    for each column), a motion streak whose strands are as long as their columns' shares."""
    h, w = img.shape[:2]
    half = np.maximum(np.rint(0.5 * np.float32(length) * share), 0).astype(np.int64)[None, :]
    if not half.any():
        return img
    S = np.zeros((h + 1, w, 3), np.float32)
    np.cumsum(img, axis=0, out=S[1:])
    rows = np.arange(h, dtype=np.int64)[:, None]
    lo, hi = np.clip(rows - half, 0, h - 1), np.clip(rows + half, 0, h - 1) + 1
    cols = np.arange(w)[None, :]
    return (S[hi, cols] - S[lo, cols]) / (hi - lo).astype(np.float32)[..., None]


def flow_moves(prev, cur, block, gain):
    """The motion from picture `prev` to `cur` (the same small size) as whole-pixel moves of blocks `block` small px
    across: Farneback optical flow on their luma, averaged over each block, `gain` times. ((gh, gw, 2) float32, x right,
    y down, small px)."""
    def grey(p):
        return (np.clip(p @ LUMA, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(grey(prev), grey(cur), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = f.shape[:2]
    gw, gh = max(int(math.ceil(w / block)), 1), max(int(math.ceil(h / block)), 1)
    return cv2.resize(f, (gw, gh), interpolation=cv2.INTER_AREA) * np.float32(gain)


def carry(held, moves, scale):
    """`held` (full size) moved one step along block `moves` ((gh, gw, 2), small px: times `scale` full px, rounded to
    whole pixels so each block moves sharp): each pixel takes what stood its block's move behind it; past the frame's
    edge its edge repeats."""
    h, w = held.shape[:2]
    gh, gw = moves.shape[:2]
    full = np.rint(moves * np.float32(scale))
    yi = np.minimum((np.arange(h) * gh) // h, gh - 1)
    xi = np.minimum((np.arange(w) * gw) // w, gw - 1)
    m = full[yi[:, None], xi[None, :]]
    if not m.any():
        return held
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(held, np.float32), xs - m[..., 0], ys - m[..., 1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REPLICATE)


def break_order(size, block, patch, seed):
    """(gh, gw) float32 in 0 .. 1: when each block of `block` px of a picture of `size` breaks through a smear (it shows
    the new picture once the smear's share passes it): `patch` of it smooth noise (patches break together), the rest a
    block's own chance, ranked so a share u breaks u of the blocks."""
    W, H = size
    gw, gh = max(int(math.ceil(W / block)), 1), max(int(math.ceil(H / block)), 1)
    rng = np.random.default_rng([int(seed), 71])
    cells = rng.random((max(gh // 6, 1) + 2, max(gw // 6, 1) + 2)).astype(np.float32)
    smooth = cv2.resize(cells, (gw, gh), interpolation=cv2.INTER_CUBIC)
    score = np.float32(patch) * smooth + np.float32(1.0 - patch) * rng.random((gh, gw)).astype(np.float32)
    order = np.empty(gh * gw, np.float32)
    order[np.argsort(score, axis=None, kind="stable")] = np.arange(gh * gw, dtype=np.float32) / (gh * gw)
    return order.reshape(gh, gw)


def broken(order, u, size):
    """(h, w) bool: the pixels of a picture of `size` whose blocks have broken through at share `u` (break_order's)."""
    W, H = size
    gh, gw = order.shape
    yi = np.minimum((np.arange(H) * gh) // H, gh - 1)
    xi = np.minimum((np.arange(W) * gw) // W, gw - 1)
    return (order < np.float32(u))[yi[:, None], xi[None, :]]


def bled(held, carried, new, mask, u, bleed):
    """The smear's picture: `held` where no block has broken (`mask` False), its chroma drifting toward `new`'s with the
    share `u`; `new` where it has, keeping some of the chroma `carried` (the held picture moved there) had while `u` is
    young. `bleed` 0 leaves both their own colours."""
    if bleed <= 0.0:
        return np.where(mask[..., None], new, held)
    cn = chroma(new)
    a = np.float32(BLEED_HELD * bleed * u)
    b = np.float32(BLEED_NEW * bleed * (1.0 - u))
    out_held = held + (cn - chroma(held)) * a
    out_new = new + (chroma(carried) - cn) * b
    return np.where(mask[..., None], out_new, out_held).astype(np.float32)


# ================================================================================================= over a cut
class SkyFx:
    """The skies' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t and
    _footage_of, and TableFx's _table_state, table_view and _ground). `_sky_init` is called once. Everything is drawn
    at the output's own size."""

    def _sky_init(self):
        self._sky_layers = {}        # (entry index, segment, layer) -> (its clip, its grade's luts, its grade)
        self._sky_papers = {}        # (entry index, segment, layer) -> the paper picture
        self._sky_held = {}          # a smear's key -> (frame, the held picture, the new one's small picture, the output)
        self._sky_orders = {}        # a smear's key -> (its break order, its made-up flow's block moves or None)
        self._sky_bands = {}         # (entry index, melt) -> its columns' shares

    # ---------------------------------------------------------------- pictures
    def _sky_layer(self, fx, i, j):
        """(the clip of layer `j` of segment `i` of sky `fx`, its grade's luts on the clip's own stretch, its grade)."""
        key = (fx["index"], i, j)
        got = self._sky_layers.get(key)
        if got is not None:
            return got
        if self._footage_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (sky): no footage to read")
        layer = fx["clips"][i]["layers"][j]
        from . import footage as FT
        try:
            clip = self._footage_of(layer["clip"])
            clip.index(0.0, layer["in"], layer["rate"], layer["loop"])
        except FT.FootageError as e:
            raise FT.FootageError(f"fx {fx['index']} (sky) clips[{i}]: {e}") from None
        g = {**fx["grade"], "lean": fx["grade"]["lean"] if layer["lean"] is None else layer["lean"],
             "stretch": fx["grade"]["stretch"] if layer["stretch"] is None else layer["stretch"]}
        stretch = TBF.clip_stretch(clip, g, layer["in"] if layer["at"] is None else layer["at"], layer["loop"])
        got = self._sky_layers[key] = (clip, TBF.luts(g, stretch), g)
        return got

    def sky_segment(self, fx, i, t):
        """Segment `i` of sky `fx` at clip second `t`, its layers composed in order (each melted by its own melts, then
        laid over the ones before it in its box; one that keeps the dark, by how much darker than its sky it is, and
        nothing past its clip's edges); None for a table's segment (the table draws it)."""
        seg = fx["clips"][i]
        if seg["table"] is not None:
            return None
        out = None
        for j, layer in enumerate(seg["layers"]):
            if layer["paper"] is not None:
                pic, cover = self._sky_paper(fx, i, j).copy(), None
            else:
                pic, cover = self._sky_clip(fx, i, j, t)
            for k, m, s in SY.melts_at(layer, t):
                pic = self._sky_melt((fx["index"], i, j, k), m, s, pic)
            m = None if out is None else area_mask(self.size, layer["area"], layer["feather"] * self.size[1])
            if out is not None and layer["keep"] == "dark":
                a = kept_dark(pic, KEEP_REACH * self.size[1]) * cover
                m = a if m is None else m * a
            out = pic if m is None else out + (pic - out) * m[..., None]
        return out

    def _sky_paper(self, fx, i, j):
        key = (fx["index"], i, j)
        got = self._sky_papers.get(key)
        if got is None:
            got = self._sky_papers[key] = paper(self.size, fx["clips"][i]["layers"][j]["paper"])
        return got

    def _sky_clip(self, fx, i, j, t):
        """(layer `j` of segment `i` at clip second `t`: its clip's frame there through its view, graded; for a layer
        that keeps the dark, how much of each pixel its clip covers (past its edges nothing), else None)."""
        seg, layer = fx["clips"][i], fx["clips"][i]["layers"][j]
        clip, tables, g = self._sky_layer(fx, i, j)
        rgb = clip.frame(clip.index(t - seg["t"], layer["in"], layer["rate"], layer["loop"]))
        D = TB.view_map(layer["view"], t, self.size, clip.size, 0.5 / self.fps)
        out, _ = TBF.painted(TBF.carried(rgb, D, self.size, clip.size, g["soften"] * clip.size[1], True), g, tables)
        if layer["keep"] is None:
            return out, None
        k = 8.0                                                         # the clip's area, worked on an eighth of it
        small = np.ones((max(int(clip.size[1] / k), 1), max(int(clip.size[0] / k), 1)), np.float32)
        Ds = np.asarray(D, np.float64) @ np.array([[k, 0.0, 0.5 * k - 0.5], [0.0, k, 0.5 * k - 0.5], [0.0, 0.0, 1.0]])
        return out, cv2.warpAffine(small, Ds, self.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _picture_of(self, name, frame, asked):
        """The picture the entry named `name` draws on `frame`: a sky's, or a table's ground (its clip through its view
        and grade, never what lies on it)."""
        e = SY.named(self.fxs, name)
        if e is None:
            raise FX.FxError(f"fx {asked['index']} ({asked['kind']}): smear from {name!r}: no sky or table (a sheet with a "
                             f"`clip`) of that name")
        if e["kind"] in SY.KINDS:
            return self.sky_picture(e, frame)
        return self._ground(e, self._table_state(e), self._t(frame), self.table_view(e, frame)[0])[0]

    def sky_picture(self, fx, frame):
        """What sky `fx` draws on `frame`: its segment there, smeared in from the picture before through its smear's
        window, melted; on a table's segment, that table's ground."""
        t = self._t(frame)
        i = SY.segment_at(fx, t)
        seg = fx["clips"][i]
        if seg["table"] is not None:
            return self._picture_of(seg["table"], frame, fx)
        sm = seg["smear"]
        if sm is None or SY.smear_share(sm, t, seg["t"]) is None:
            out = self.sky_segment(fx, i, t)
        else:
            old = (lambda g: self._picture_of(sm["from"], g, fx)) if sm["from"] is not None else \
                (lambda g: self.sky_picture_of_segment(fx, i - 1, g))
            out = self._smeared(("sky", fx["index"], i), sm, seg["t"], frame, old,
                                lambda g: self.sky_segment(fx, i, self._t(g)))
        for k, m, s in SY.melts_at(fx, t):
            out = self._sky_melt((fx["index"], k), m, s, out)
        return out

    def sky_picture_of_segment(self, fx, i, frame):
        """Segment `i` of sky `fx` as the sky drew it on `frame` (a table's: that table's ground), its melts on it."""
        seg = fx["clips"][i]
        t = self._t(frame)
        out = self._picture_of(seg["table"], frame, fx) if seg["table"] is not None else self.sky_segment(fx, i, t)
        for k, m, s in SY.melts_at(fx, t):
            out = self._sky_melt((fx["index"], k), m, s, out)
        return out

    # ---------------------------------------------------------------- the smear
    def _smeared(self, key, sm, t0, frame, old, new):
        """The smear `sm` (key `key`) that starts at clip second `t0`, on `frame`: the picture `old(g)` draws on the
        frame before `t0`'s held and carried along `new`'s motion (or the made-up flow) frame by frame from `t0`'s, the
        blocks of `new(g)` breaking through by its order. One step on from the frame before when that was the last
        asked, the same picture again for the same frame, else worked out again from the start."""
        f0 = FX.frame_of(t0, self.fps, self.frame0)
        W, H = self.size
        small = (max(int(round(W * SMALL)), 8), max(int(round(H * SMALL)), 8))
        block = sm["block"] * self.px
        if key not in self._sky_orders:
            flow = sm["flow"] if sm["jump"] is None else jump_flow(sm["jump"], sm["dur"], self.fps, sm["seed"])
            made = None if flow is None else (*_ff().mosh_flow(flow, self.size, block, self.px, sm["seed"]), flow["steps"])
            self._sky_orders[key] = (break_order(self.size, block, sm["patch"], sm["seed"]), made)
        order, made = self._sky_orders[key]
        state = self._sky_held.get(key)
        if state is not None and state[0] == frame:
            return state[3].copy()
        if state is not None and state[0] == frame - 1:
            held, prev, start = state[1], state[2], frame
        else:
            held = old(f0 - 1)
            prev = None if made is not None else cv2.resize(new(f0 - 1), small, interpolation=cv2.INTER_AREA)
            start = f0
        for g in range(start, frame + 1):
            pic = new(g)
            if made is not None:
                carried = _ff().mosh_carry(held, made[0], made[1], made[2])
            else:
                cur = cv2.resize(pic, small, interpolation=cv2.INTER_AREA)
                carried = carry(held, flow_moves(prev, cur, max(block * SMALL, 1.0), sm["gain"]), W / small[0])
                prev = cur
            u = SY.smear_share(sm, self._t(g), t0)
            u = 1.0 if u is None else u
            mask = broken(order, u, self.size)
            held = np.where(mask[..., None], pic, carried).astype(np.float32)
        out = bled(held, carried, pic, mask, u, sm["bleed"])
        self._sky_held[key] = (frame, held, prev, out)
        return out.copy()

    def table_smear(self, fx, frame, ground):
        """A table's ground on `frame` through its `smear` (from a named sky or table), or `ground` as it is outside its
        window."""
        sm = fx.get("smear")
        if sm is None or SY.smear_share(sm, self._t(frame), fx["from"]) is None:
            return ground
        st = self._table_state(fx)

        def own(g):
            return ground if g == frame else self._ground(fx, st, self._t(g), self.table_view(fx, g)[0])[0]
        return self._smeared(("table", fx["index"]), sm, fx["from"], frame,
                             lambda g: self._picture_of(sm["from"], g, fx), own)

    # ---------------------------------------------------------------- melts
    def _sky_melt(self, key, m, s, img):
        """`img` melted by melt `m` (`key`: the sky's index and the melt's place, a layer's or the sky's own) at
        strength `s`, inside its box. A streak of every pixel (`run = "all"`) mixes colours along each column, so it is
        graded again through the sky's ramps by the luma and warmth it ends with (as strongly as it streaks): it keeps
        Dawn's clean colours instead of their mud."""
        H, W = img.shape[:2]
        share = self._sky_bands.get(key)
        if share is None:
            share = self._sky_bands[key] = bands(W, m["seed"] * 1009 + sum(key), self.px)
        length = m["length"] * H * s
        if m["run"] == "light":
            out = melt_light(img, m["dir"], m["level"], length, share)
        else:
            out = melt_all(img, length, share)
            g = next(f for f in self.fxs if f["index"] == key[0])["grade"]
            tables = self._sky_bands.get(("grade", key[0]))
            if tables is None:
                tables = self._sky_bands[("grade", key[0])] = TBF.luts(g, (0.0, 255.0))
            clean = TBF.warmed((out @ LUMA) * np.float32(255.0), (out[..., 0] - out[..., 2]) * np.float32(255.0), tables,
                               0.0, g["warmth"])
            out = out + (clean - out) * np.float32(min(s, 1.0))
        mask = area_mask((W, H), m["area"], m["feather"] * H)
        return out if mask is None else (img + (out - img) * mask[..., None]).astype(np.float32)

    # ---------------------------------------------------------------- the frame
    def _sky(self, fx, frame, img):
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0 or fx["clips"][SY.segment_at(fx, t)]["table"] is not None:
            return img
        out = self.sky_picture(fx, frame)
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)
