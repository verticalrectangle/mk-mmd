"""Tables on the finished pictures (docs/design.md: Frame effects, Sticker sheets: the table): the picture work of a
sheet with a `clip`, which mkmmd.core.table checks and times. TableFx is the part of mkmmd.framefx.FrameFx that draws it.

A table is drawn through its view on every frame, back to front and every part at the output's own size: the clip (its
luma softened by the grade's `soften`, carried into the frame first, then through the grade's map, or inside the cup's
rim through the coffee's own, so the coffee seen thirty times closer is smooth and never bands; a grain finer than a
pixel over it), the drift rings and the pads on the coffee (cut by the cup's rim, an
ellipse worked out on the frame's own pixels, with a thin line of light along it), the drops, her sticker on her pad (her
plate's figure carried onto the table by its placement or its `track`, dipping with the pad, its die-cut rim and shadow
in the frame's px) and the doodles. A pad and a doodle are cut at the size they show at, in steps SIZE_STEP apart, and
kept a while. Through a punch the picture under the table (the cut's own frame, as it reached the table) shows in the
cup, scaled with it. Images are float RGB in display space (0..1), (h, w, 3); a coverage is (h, w) in 0..1."""
import math
from collections import OrderedDict

import cv2
import numpy as np

from . import fields as FL
from .core import framefx as FX
from .core import sheet as ST
from .core import table as TB

LUT = 4096               # steps of a table's gradient map over the luma 0..1
SHRINK = 0.7             # the clip's luma is shrunk by area first where the view shows it smaller than this
SOFTEN = 3.0             # and softened where it shows it larger than this (its blocks and grain never show blown up)
GRAIN = 0.6 / 255.0      # a fine grain over the graded clip: a dither, so a smooth coffee never bands
PAD_STEP = 1.25          # a pad's tile is cut at sizes this far apart and shrunk to the size it shows at
PAD_SUPER = 3            # and drawn that many times larger first (its outline antialiased)
PAD_TILES = 48           # pad tiles kept
PAD_SHADOW = 0.35        # a pad's shadow on the coffee: alpha; its offset and blur are the sheet's shadow's halves
WORK = 1080              # px: the table's height in the picture its layout is worked out on
_STRETCH = {}            # (clip path, frame, low, high) -> the luma at those percentiles of that frame


def _ff():
    from . import framefx as FF          # framefx imports this module for TableFx: its helpers are there by now
    return FF


def _sheet():
    from . import sheet as STF           # sheet imports this module's neighbours only: its helpers are there by now
    return STF


# ================================================================================================= pure parts
def luma8(rgb):
    """The luma of uint8 RGB as uint8 (OpenCV's: 0.299, 0.587, 0.114)."""
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def stretch_of(rgb, lo, hi):
    """(luma at the `lo` and `hi` percentiles) of a uint8 RGB frame, 0..255 (at least one code value apart)."""
    a, b = np.percentile(luma8(rgb), [lo, hi])
    return float(a), float(max(b, a + 1.0))


def spread(stretch, n):
    """The lumas (0..1) `n` colours of a map lie at when it is spread evenly over a `stretch` (the luma at its low and
    high percentiles, 0..255): the first at the low one, the last at the high one."""
    return np.linspace(stretch[0] / 255.0, stretch[1] / 255.0, n).tolist()


def ramp(colors, lumas, n=LUT):
    """(n, 3) float32: a gradient map over the luma 0..1, each of `colors` (dark to light) at its luma of `lumas`, the
    first below the first luma and the last above the last."""
    c = np.asarray(colors, np.float32)
    u = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(u, lumas, c[:, k]) for k in range(3)], 1).astype(np.float32)


def graded(L, table):
    """A luma picture `L` (float32, 0..255) through the gradient map `table` (ramp): float32 RGB."""
    k = np.float32((len(table) - 1) / 255.0)
    return table[np.clip(np.rint(L * k), 0, len(table) - 1).astype(np.int32)]


def ellipse_distance(size, M, centre, radii, box=None):
    """The signed distance (frame px, negative inside) of each pixel of a picture of `size` (or of its `box` (x0, y0, x1,
    y1)) from an ellipse of the table (`centre`, `radii` in table px) seen through the 2 x 3 map `M` (table px -> frame
    px): its implicit value over its gradient, a first-order distance, exact on the outline."""
    W, H = size
    x0, y0, x1, y1 = box or (0, 0, W, H)
    A = np.asarray(M, np.float64)
    inv = cv2.invertAffineTransform(A)
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    X = inv[0, 0] * xs[None, :] + inv[0, 1] * ys[:, None] + inv[0, 2]
    Y = inv[1, 0] * xs[None, :] + inv[1, 1] * ys[:, None] + inv[1, 2]
    u = (X - np.float32(centre[0])) / np.float32(radii[0])
    v = (Y - np.float32(centre[1])) / np.float32(radii[1])
    f = u * u + v * v - 1.0
    g = 2.0 * np.sqrt((u / np.float32(radii[0])) ** 2 + (v / np.float32(radii[1])) ** 2) + np.float32(1e-9)
    return (f / g) * np.float32(TB.zoom_of(A))


def footprint(D, size, table, pad):
    """The table px (x0, y0, x1, y1) a frame of `size` shows through the 2 x 3 map `D` (table px -> frame px), `pad` px
    more on every side, kept inside the table of `table` (w, h) px (at least a pixel)."""
    inv = cv2.invertAffineTransform(np.asarray(D, np.float64))
    W, H = size
    c = inv @ np.array([[0, W, 0, W], [0, 0, H, H], [1, 1, 1, 1]], float)
    x0, y0 = max(int(math.floor(c[0].min() - pad)), 0), max(int(math.floor(c[1].min() - pad)), 0)
    x1, y1 = min(int(math.ceil(c[0].max() + pad)) + 1, table[0]), min(int(math.ceil(c[1].max() + pad)) + 1, table[1])
    if x1 <= x0:
        x0, x1 = (table[0] - 1, table[0]) if x0 >= table[0] else (0, 1)
    if y1 <= y0:
        y0, y1 = (table[1] - 1, table[1]) if y0 >= table[1] else (0, 1)
    return x0, y0, x1, y1


def pad_outline(notch, seed, n=180):
    """A lily pad's outline about (0, 0), radius 1 (x right, y down): a disc a little irregular (seeded) with a wedge
    `notch` degrees wide cut in to just short of its middle, the wedge pointing right (angle 0)."""
    rng = np.random.default_rng([int(seed), 41])
    half = math.radians(max(notch, 1.0)) / 2.0
    a = np.linspace(half, 2.0 * math.pi - half, n)
    ph = rng.uniform(0.0, 2.0 * math.pi, 3)
    r = 1.0 + 0.018 * np.sin(3.0 * a + ph[0]) + 0.012 * np.sin(5.0 * a + ph[1]) + 0.008 * np.sin(9.0 * a + ph[2])
    rim = np.stack([r * np.cos(a), -r * np.sin(a)], 1)
    apex = np.array([[0.07, 0.0]])
    return np.concatenate([apex, rim])


def pad_tile(pad, size, light_from=(-0.6, -0.8)):
    """A cut-paper lily pad (`pad`: its colours, notch, veins, seed) as a tile (premultiplied RGB, coverage, its middle
    px) `size` px across: its paper (the colour mottled toward `light` in soft clouds, a fine tooth: the paper's own
    texture at the size it shows at, so it is as sharp seen close as far), faint veins curving out from the notch's foot
    and fading toward the rim, the cut edge a pale line along the outline (the paper's core), a touch lighter toward the
    lamp. A small tile is drawn PAD_SUPER times larger first, a large one as it is."""
    k = PAD_SUPER if size < 300 else 2 if size < 700 else 1
    n = int(math.ceil(size)) + 4
    big = n * k
    rad = size / 2.0 * k
    c = big / 2.0
    out_pts = pad_outline(pad["notch"], pad["seed"]) * rad + c
    cov = np.zeros((big, big), np.uint8)
    cv2.fillPoly(cov, [np.rint(out_pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    m = cov.astype(np.float32) / 255.0
    rng = np.random.default_rng([int(pad["seed"]), 43])
    col, light = np.asarray(pad["color"], np.float32), np.asarray(pad["light"], np.float32)
    g = max(min(big, 256), 8)                                       # the clouds, worked out coarse and resized up
    q = (np.arange(g, dtype=np.float32) + 0.5) * np.float32(big / g)
    w1 = (q - c) / np.float32(rad)
    u, v = np.broadcast_to(w1[None, :], (g, g)), np.broadcast_to(w1[:, None], (g, g))
    mottle = np.zeros((g, g), np.float32)
    for f, w in ((2.5, 0.55), (6.0, 0.3), (15.0, 0.15)):
        if rad / f >= 2.0:
            mottle += np.float32(w) * (FL.noise(np.ascontiguousarray(u * f), np.ascontiguousarray(v * f), 0.0,
                                                int(pad["seed"]) * 7 + int(f)) - 0.5)
    lamp = np.clip(-(u * light_from[0] + v * light_from[1]) * 0.5, -1.0, 1.0)
    shade = cv2.resize(np.clip(0.34 + 0.9 * mottle + 0.12 * lamp, 0.0, 1.0).astype(np.float32), (big, big),
                       interpolation=cv2.INTER_CUBIC)
    tooth = cv2.GaussianBlur(rng.standard_normal((big, big)).astype(np.float32), (0, 0), 0.6 * k)
    shade = shade + np.float32(0.05) * tooth
    rgb = col + (light - col) * (shade * np.float32(0.55))[..., None]
    if pad["veins"]:
        veins = np.zeros((big, big), np.uint8)
        a0 = math.radians(pad["notch"]) / 2.0 + 0.3
        th = max(int(round(0.007 * rad)), 1)
        for a in np.linspace(a0, 2.0 * math.pi - a0, 8):
            a = a + rng.uniform(-0.12, 0.12)
            bend = rng.uniform(-0.3, 0.3)
            s = np.linspace(0.1, 0.86, 32)
            pts = np.stack([s * np.cos(a + bend * s * s), -s * np.sin(a + bend * s * s)], 1) * rad + c
            cv2.polylines(veins, [np.rint(pts * 16).astype(np.int32)], False, 255, th, cv2.LINE_AA, 4)
        rr = np.hypot(*np.meshgrid((np.arange(big, dtype=np.float32) - c) / np.float32(rad),
                                   (np.arange(big, dtype=np.float32) - c) / np.float32(rad)))
        vk = (veins.astype(np.float32) / 255.0 * np.float32(0.32) * np.clip(1.15 - rr, 0.0, 1.0))[..., None]
        rgb = rgb + (light + (1.0 - light) * np.float32(0.3) - rgb) * vk
    edge = np.zeros((big, big), np.uint8)
    cv2.polylines(edge, [np.rint(out_pts * 16).astype(np.int32)], True, 255, max(int(round(1.4 * k)), 1), cv2.LINE_AA, 4)
    ek = (edge.astype(np.float32) / 255.0 * np.float32(0.55))[..., None]
    rgb = rgb + (light + (1.0 - light) * np.float32(0.6) - rgb) * ek
    rgb = np.ascontiguousarray(rgb * m[..., None], np.float32)
    if k > 1:
        rgb = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA)
        m = np.clip(cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    return rgb, m, (n / 2.0 - 0.5, n / 2.0 - 0.5)


def lay_tile(out, rgb, a, M, shadow=None):
    """`out` (in place) with a premultiplied tile (`rgb`, coverage `a`) laid by the 2 x 3 map `M` (its px -> the
    picture's), under it its drop `shadow` (colour, (dx, dy) px, blur px, alpha); only the box it reaches is worked."""
    H, W = out.shape[:2]
    h, w = a.shape
    corners = np.asarray(M, float) @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], float)
    reach = 2.0
    if shadow is not None:
        reach += math.hypot(*shadow[1]) + 3.0 * shadow[2]
    x0, y0 = max(int(math.floor(corners[0].min() - reach)), 0), max(int(math.floor(corners[1].min() - reach)), 0)
    x1, y1 = min(int(math.ceil(corners[0].max() + reach)) + 1, W), min(int(math.ceil(corners[1].max() + reach)) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return out
    Mb = np.asarray(M, np.float64).copy()
    Mb[:, 2] -= (x0, y0)
    sz = (x1 - x0, y1 - y0)
    ca = np.clip(cv2.warpAffine(a, Mb, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), 0.0, 1.0)
    crgb = cv2.warpAffine(rgb, Mb, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    region = out[y0:y1, x0:x1]
    if shadow is not None and shadow[3] > 0:
        sh = _ff().plate_shift(ca, shadow[1][0], shadow[1][1])
        if shadow[2] > 0.3:
            sh = cv2.GaussianBlur(sh, (0, 0), shadow[2])
        k = (np.float32(shadow[3]) * sh * (1.0 - ca))[..., None]
        region *= 1.0 - k
        region += np.asarray(shadow[0], np.float32) * k
    region *= (1.0 - ca)[..., None]
    region += crgb
    return out


# ================================================================================================= over a cut
class TableFx:
    """The tables' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, _t, _figure,
    _shot_frame, _point_of, _footage_of, _hits, _pivot and _tall, and SheetFx's doodles, sticker edge and shadow).
    `_table_init` is called once."""

    def _table_init(self):
        self._tables = {}                        # a table's index -> what it keeps from frame to frame on this output
        self._pad_tiles = OrderedDict()          # (pad, size px) -> its tile

    # ---------------------------------------------------------------- what a table keeps
    def _table_state(self, fx):
        st = self._tables.get(fx["index"])
        if st is not None:
            return st
        if self._footage_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (sheet): no footage to read for its table")
        clip = _ff().clip_of(fx, self._footage_of)
        st = {"clip": clip, "table": tuple(clip.size)}
        g, cup = fx["grade"], fx["cup"]
        own = None if cup is None or cup["map"] is None else (cup["map"], cup["lumas"])
        stretch = None
        if g["lumas"] is None or (own is not None and own[1] is None):         # a map spread over the measured stretch
            stretch = self._stretch(fx, clip)
        st["lut"] = ramp(g["map"], g["lumas"] or spread(stretch, len(g["map"])))
        st["cup_lut"] = None if own is None else ramp(own[0], own[1] or spread(stretch, len(own[0])))
        st["places"] = [TB.touch_place(k, fx["pads"]) for k in fx["touch"]]
        Wt, Ht = st["table"]
        st["starts"] = TB.ring_starts(fx["touch"], fx["pads"], Wt / Ht)
        r = WORK / float(Ht)
        st["work"] = (max(int(round(Wt * r)), 1), WORK)
        st["taken"] = np.zeros(st["work"][::-1], np.float32)
        st["centre"] = st["places"][0] if st["places"] else (0.5, 0.5)
        self._tables[fx["index"]] = st
        if fx["objects"] is not None:
            m0 = self._figure(fx, fx["f0"])
            if m0 is None or not (m0 >= 0.5).any():
                raise FX.FxError(f"fx {fx['index']} (sheet): the figure is not in its frame on the span's first frame "
                                 f"({fx['f0']}), which lays the table out round it")
            M = FX.compose([[r, 0.0, 0.5 * r - 0.5], [0.0, r, 0.5 * r - 0.5]], self._figure_table(fx, st, fx["f0"]))
            st["taken"] = np.clip(_ff()._warp(m0, M, st["work"]), 0.0, 1.0)
            if (st["taken"] >= 0.5).any():
                cx, cy = _ff().centroid(st["taken"])
                st["centre"] = ((cx + 0.5) / st["work"][0], (cy + 0.5) / st["work"][1])
        st["doodles"] = None if fx["doodles"] is None else self._table_plan(fx, st, fx["doodles"], True)
        if fx["drops"] is not None:
            st["drops_dd"], st["drops"] = self._drops_of(fx, st)
        return st

    def _stretch(self, fx, clip):
        """The luma at the grade's percentiles, on the clip's frame at `grade.at` (its `in` by default), once."""
        g = fx["grade"]
        i = clip.index(0.0, fx["in"] if g["at"] is None else g["at"], 0.0, fx["loop"])
        key = (str(clip.path), i, g["stretch"][0], g["stretch"][1])
        if key not in _STRETCH:
            _STRETCH[key] = stretch_of(clip.frame(i), *g["stretch"])
        return _STRETCH[key]

    def _figure_table(self, fx, st, g):
        """The 2 x 3 map (the figure's plate px -> table px) on frame `g`: by its `track` (its world point seen through the
        plate there) or by `at` and `height` about its pivot (found on the span's first frame), then moved by its keys."""
        Wt, Ht = st["table"]
        if fx["track"] is not None:
            p, m = self._point_of(TB.track_key(fx), g)
            if m is None or m <= 0:
                raise FX.FxError(f"fx {fx['index']} (sheet): track: its point has no size on frame {g}: it lies behind "
                                 f"the plate's camera or at it")
            M = TB.track_map(fx["track"], p, m, self.size[1], st["table"]).tolist()
            to = (fx["track"]["at"][0] * Wt - 0.5, fx["track"]["at"][1] * Ht - 0.5)
        else:
            if fx["at"] is None or fx["height"] is None:
                raise FX.FxError(f"fx {fx['index']} (sheet): a table's figure lies where `at` and `height` (table "
                                 f"fractions and heights) or a `track` put it")
            a = self._pivot(fx)
            to = (fx["at"][0] * Wt - 0.5, fx["at"][1] * Ht - 0.5)
            M = FX.placement(a, to, fx["height"] * Ht / self._tall(fx))
        if fx["keys"]:
            mv = FX.move_at(fx["keys"], self._t(g))
            if mv != FX.MOVE:
                M = FX.compose(FX.move_matrix(mv, to, Ht), M)
        return M

    def _table_plan(self, fx, st, dd, doodles):
        """A doodle group laid out on the table, as a sheet lays its doodles on the frame (SheetFx._doodle_plan): clear of
        her sticker as it lies on the span's first frame and, for the doodles, of the cup; the drops on the coffee, clear
        of the pads."""
        W, H = st["work"]
        taken = st["taken"]
        if fx["cup"] is not None:
            c, r = fx["cup"]["center"], fx["cup"]["radius"]
            ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
            e = ((xs + 0.5) / H - c[0] * W / H) ** 2 / r[0] ** 2 + ((ys + 0.5) / H - c[1]) ** 2 / r[1] ** 2
            taken = np.maximum(taken, (e <= 1.12 if doodles else e >= 0.9).astype(np.float32))
        if not doodles and fx["pads"]:
            pads = np.zeros((H, W), np.uint8)
            for p in fx["pads"]:
                cv2.circle(pads, (int(round(p["at"][0] * W - 0.5)), int(round(p["at"][1] * H - 0.5))),
                           int(math.ceil(0.5 * p["size"] * H)), 1, -1)
            taken = np.maximum(taken, pads.astype(np.float32))
        far = cv2.distanceTransform((taken < 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        def away(p):
            xs = np.clip(np.rint(p[:, 0] * H - 0.5).astype(int), 0, W - 1)
            ys = np.clip(np.rint(p[:, 1] * H - 0.5).astype(int), 0, H - 1)
            return far[ys, xs] / H
        seed = fx["seed"] if doodles else fx["seed"] * 31 + 7
        deal = ST.deal(dd, seed)
        radii = [ST.REACH * dd["size"] * d["size"] for d in deal]
        area = dd["area"]
        if not doodles and not dd["own_area"]:
            x, y = st["places"][0]
            k = dd["reach"]
            area = [x - k * H / W, y - k, x + k * H / W, y + k]
        places = ST.scatter(radii, W / H, area, dd["clear"], dd["gap"], seed * 1009 + dd["seed"], away)
        hops = []
        hp = dd["hop"]
        if hp is not None and hp["beats"]:
            hits = self._hits or {}
            if "beats" not in hits:
                raise FX.FxError(f"fx {fx['index']} (sheet): hop beats = true needs the timeline's beats "
                                 f"(audio/timeline.json or [audio] timeline)")
            hops = ST.beat_times(hits["beats"], hits.get("downbeats") or [], hp["every"])
        elif hp is not None:
            hops = FX.hit_times({**hp, "index": fx["index"], "kind": "sheet", "from": fx["from"] - 2.0,
                                 "to": fx["to"] + 1.0}, self._hits)
        centre = st["centre"] if doodles else st["places"][0]
        origin = None
        if dd["jump"] is not None:
            origin = centre if dd["jump"]["from"] in ("figure", None) else tuple(dd["jump"]["from"])
        return {"deal": deal, "places": places, "wave": ST.ripple(places, centre), "hops": hops, "origin": origin,
                "peel": ST.peel_order(dd, places, centre, seed), "slaps": ST.slap_order(dd, places), "scribbles": [],
                "tapes": []}

    def _drops_of(self, fx, st):
        """A table's drops as a doodle group whose jump is the first touch's (on its second) unless it gave its own, and
        their layout: thrown from the touch (a jump from "figure", the default), each leaves from the rim of the pad the
        touch lands on toward where it lands (TB.drop_origins)."""
        dd = dict(fx["drops"])
        if dd["jump"]["t"] is None:
            dd["jump"] = {**dd["jump"], "t": fx["touch"][0]["t"]}
        plan = self._table_plan(fx, st, dd, False)
        if dd["jump"]["from"] in ("figure", None):
            Wt, Ht = st["table"]
            plan["origins"] = TB.drop_origins(plan["places"], st["places"][0], st["starts"][0], Wt / Ht)
        return dd, plan

    # ---------------------------------------------------------------- the view
    def table_view(self, fx, frame):
        """(the 2 x 3 map table px -> frame px on `frame`, the punch on (punch, u) or None, the scale of the picture
        shown in the cup through it or None): the table's view, or a punch's way into or out of the cup, and the desk's
        drift over either (fading to nothing at a punch's `t`, where the frame is the cut's own)."""
        st = self._table_state(fx)
        t = self._t(frame)
        hit = TB.punch_at(fx["punch"], t)
        k, fade = None, 1.0
        if hit is None:
            M = TB.view_map(fx["view"], t, self.size, st["table"], 0.5 / self.fps)
        else:
            p, u = hit
            a, b = TB.window(p)
            rest = TB.view_map(fx["view"], b if p["out"] else a, self.size, st["table"], 0.5 / self.fps)
            M, k = TB.punch_map(p, u, rest, self.size, st["table"], fx["cup"])
            fade = u if p["out"] else 1.0 - u
        dx, dy, turn = ST.drift_at(fx["drift"], t, fx["seed"])
        if fade <= 0.0 or (dx == 0.0 and dy == 0.0 and turn == 0.0):
            return np.asarray(M, float), hit, k
        W, H = self.size
        r = math.radians(turn * fade)
        c, s = math.cos(r), math.sin(r)
        C = TB.middle(self.size)
        Dr = [[c, s, C[0] + dx * fade * H - (c * C[0] + s * C[1])], [-s, c, C[1] + dy * fade * H - (-s * C[0] + c * C[1])]]
        return np.asarray(FX.compose(Dr, np.asarray(M).tolist()), float), hit, k

    def table_point(self, sheet, xy, frame):
        """(frame px, frame heights per table height, turn degrees) of the table point `xy` (fractions) of the table
        `sheet` (its index) on `frame`: where a cutout riding the table stands, how much larger it is seen, how far
        turned."""
        fx = next(f for f in self.fxs if f["index"] == sheet)
        st = self._table_state(fx)
        D, _, _ = self.table_view(fx, frame)
        Wt, Ht = st["table"]
        p = D @ np.array([xy[0] * Wt - 0.5, xy[1] * Ht - 0.5, 1.0])
        return (float(p[0]), float(p[1])), TB.zoom_of(D) * Ht / self.size[1], TB.turn_of(D)

    # ---------------------------------------------------------------- the frame
    def _table(self, fx, frame, img):
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        if s <= 0.0 or TB.gone(fx["punch"], t):
            return img
        st = self._table_state(fx)
        D, hit, k = self.table_view(fx, frame)
        cup = dist = None
        if fx["cup"] is not None:
            cup, dist = self._cup(fx, st, D)
        cut = fx["inside"] is not None and hit is None          # the cut's own frame lies in the cup
        whole = cup is not None and float(dist.max()) < -0.5    # the coffee covers the frame: no table to see round it
        ground = L = None
        if not (cut and whole):
            ground, L = self._ground(fx, st, t, D)
        out = ground
        if cup is not None:
            if cut:
                coffee = self._cut_in_cup(fx, st, frame, D, img)
            else:
                coffee = ground.copy() if st["cup_lut"] is None else graded(L, st["cup_lut"]) + st["grain"]
            fronts = TB.ring_fronts(fx["touch"], fx["rings"], t, st["places"], st["starts"])
            if fronts:
                self._rings(fx, st, D, coffee, fronts)
            self._pads(fx, st, t, D, coffee, fronts)
            if cut and fx["inside"]["shadow"] is not None:
                self._inner_shadow(fx, st, D, coffee)
            out = coffee if whole else coffee * cup[..., None] + ground * (1.0 - cup)[..., None]   # exact where covered
        shadow = self._sheet_shadow(fx)
        if fx["drops"] is not None:
            self._doodles(fx, t, out, D, st, shadow, st["drops_dd"], st["drops"], st["table"])
        if fx["objects"] is not None:
            out = self._table_figure(fx, st, frame, t, D, out, shadow, img)
        if hit is not None and cup is not None:
            out = self._inside(fx, hit, k, D, st, img, out, cup)
        line = None if cup is None else fx["cup"]["line"]
        if line is not None and line["width"] > 0 and float(dist.max()) > -(line["width"] * self.px + 1.0):
            w = line["width"] * self.px
            a = np.clip(0.5 - (np.abs(dist + w / 2.0) - w / 2.0), 0.0, 1.0) * np.float32(line["alpha"])
            out = out + (np.asarray(line["color"], np.float32) - out) * a[..., None]
        if fx["doodles"] is not None:
            self._doodles(fx, t, out, D, st, shadow, fx["doodles"], st["doodles"], st["table"])
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    def _cut_in_cup(self, fx, st, frame, D, img):
        """The cut's own frame `img` as the coffee: carried onto the table by `inside` (its world point, seen through the
        cut's camera on this frame, on the table's `at`, a metre there `metre` table heights), upright in the frame
        whatever the view's turn, and into the frame through the view `D`. When the cut's lens follows the view (the
        map's scale 1) it is laid on whole pixels, the frame untouched at its own size; otherwise resampled (shrunk by
        area first where it is seen smaller)."""
        p, m = self._point_of(TB.inside_key(fx), frame)
        if m is None or m <= 0:
            raise FX.FxError(f"fx {fx['index']} (sheet): inside: its point has no size on frame {frame}: it lies "
                             f"behind the cut's camera or at it")
        W, H = self.size
        Mi = TB.track_map(fx["inside"], p, m, H, st["table"], -math.radians(TB.turn_of(D)))
        A = np.asarray(FX.compose(np.asarray(D).tolist(), Mi.tolist()), np.float64)
        pic = np.ascontiguousarray(img, np.float32)
        if np.abs(A[:, :2] - np.eye(2)).max() < 1e-4:                   # the frame at its own size: whole pixels
            dx, dy = (int(round(v)) for v in A[:, 2])
            if dx == 0 and dy == 0:
                return pic.copy()
            return cv2.warpAffine(pic, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)
        z = TB.zoom_of(A)
        if z < SHRINK:
            f = z / SHRINK
            sw, sh = max(int(round(W * f)), 1), max(int(round(H * f)), 1)
            fx_, fy_ = sw / W, sh / H
            pic = cv2.resize(pic, (sw, sh), interpolation=cv2.INTER_AREA)
            A = A @ np.array([[1.0 / fx_, 0.0, 0.5 / fx_ - 0.5], [0.0, 1.0 / fy_, 0.5 / fy_ - 0.5], [0.0, 0.0, 1.0]])
        return cv2.warpAffine(pic, A, (W, H), flags=cv2.INTER_CUBIC if z > 1.0 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    def _inner_shadow(self, fx, st, D, coffee):
        """`coffee` (in place) under the soft shadow the cup's wall casts inside it: dark beyond the rim's ellipse moved
        `offset` (table heights, the clip's light: a crescent along the rim on the lamp's side), fading over `width`."""
        sh = fx["inside"]["shadow"]
        Wt, Ht = st["table"]
        c = fx["cup"]["center"]
        q = (c[0] * Wt - 0.5 + sh["offset"][0] * Ht, c[1] * Ht - 0.5 + sh["offset"][1] * Ht)
        r = (fx["cup"]["radius"][0] * Ht, fx["cup"]["radius"][1] * Ht)
        d = ellipse_distance(self.size, D, q, r)
        w = np.float32(max(sh["width"] * Ht * TB.zoom_of(D), 0.5))
        u = np.clip((d + w) / (2.0 * w), 0.0, 1.0)
        a = u * u * (3.0 - 2.0 * u) * np.float32(sh["alpha"])
        if a.any():
            coffee += (np.asarray(sh["color"], np.float32) - coffee) * a[..., None]

    def _ground(self, fx, st, t, D):
        """(the clip under the table at clip second `t` as the frame shows it through `D`, its luma there): its luma (the
        part the frame shows, shrunk by area first where the view is far, softened by the grade's `soften` and where the
        view is close) carried into the frame, past the clip's edges reflected, then through the grade's map, with a
        grain finer than a pixel."""
        clip = st["clip"]
        rgb = clip.frame(clip.index(t - fx["from"], fx["in"], fx["rate"], fx["loop"]))
        z = TB.zoom_of(D)
        soft = fx["grade"]["soften"] * st["table"][1]                   # clip px
        pad = 4.0 / min(z, 1.0) + 3.0 + 3.0 * soft
        x0, y0, x1, y1 = footprint(D, self.size, st["table"], pad)
        L = luma8(rgb[y0:y1, x0:x1]).astype(np.float32)
        M = np.asarray(D, np.float64).copy()
        M[:, 2] += M[:, :2] @ np.array([x0, y0], float)             # the crop's px -> frame px
        blur = 0.0
        if z < SHRINK:
            f = z / SHRINK
            sw, sh = max(int(round(L.shape[1] * f)), 1), max(int(round(L.shape[0] * f)), 1)
            fx_, fy_ = sw / L.shape[1], sh / L.shape[0]
            L = cv2.resize(L, (sw, sh), interpolation=cv2.INTER_AREA)
            M = M @ np.array([[1.0 / fx_, 0.0, 0.5 / fx_ - 0.5], [0.0, 1.0 / fy_, 0.5 / fy_ - 0.5], [0.0, 0.0, 1.0]])
            soft *= fy_
        elif z > SOFTEN:
            blur = min(0.25 * z / SOFTEN, 1.2)
        blur = math.hypot(blur, soft)
        if blur > 0.25:
            L = cv2.GaussianBlur(L, (0, 0), blur)
        W, H = self.size
        L = cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_CUBIC if z > 1.0 else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        out = graded(L, st["lut"])
        if "grain" not in st:
            rng = np.random.default_rng([fx["index"], 59])
            st["grain"] = (rng.standard_normal((H, W)).astype(np.float32) * np.float32(GRAIN))[..., None]
        return out + st["grain"], L

    def _cup(self, fx, st, D):
        """(coverage of the coffee in the frame, signed distance from its rim in frame px)."""
        Wt, Ht = st["table"]
        c = fx["cup"]["center"]
        q = (c[0] * Wt - 0.5, c[1] * Ht - 0.5)
        r = (fx["cup"]["radius"][0] * Ht, fx["cup"]["radius"][1] * Ht)
        dist = ellipse_distance(self.size, D, q, r)
        return np.clip(0.5 - dist, 0.0, 1.0), dist

    def _rings(self, fx, st, D, coffee, fronts):
        """The drift rings (mkmmd.fields.drift, standing still) on the coffee, shown in each front's band: from the front
        (a soft edge) back toward the touch over `band` rings, fading as the front fades."""
        rg = fx["rings"]
        Wt, Ht = st["table"]
        z = TB.zoom_of(D)
        unit = z * Ht
        W, H = self.size
        by = {}
        for c, r, a in fronts:
            by.setdefault(tuple(c), []).append((r, a))
        spec = {"colors": rg["colors"], "layout": "rings", "motif": rg["motif"], "size": rg["size"], "angle": 0.0}
        for c, fr in by.items():
            p = D @ np.array([c[0] * Wt - 0.5, c[1] * Ht - 0.5, 1.0])
            R = max(r for r, _ in fr) * unit + 2.0
            x0, y0 = max(int(math.floor(p[0] - R)), 0), max(int(math.floor(p[1] - R)), 0)
            x1, y1 = min(int(math.ceil(p[0] + R)) + 1, W), min(int(math.ceil(p[1] + R)) + 1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = np.arange(x0, x1, dtype=np.float32) - np.float32(p[0])
            ys = np.arange(y0, y1, dtype=np.float32) - np.float32(p[1])
            rho = np.hypot(xs[None, :], ys[:, None]) / np.float32(unit)             # table heights from the touch
            band = rg["band"] * rg["size"]
            m = np.zeros(rho.shape, np.float32)
            for r, a in fr:
                edge = np.clip((np.float32(r) - rho) * np.float32(unit) / np.float32(max(1.5, 0.12 * band * unit)) + 0.5,
                               0.0, 1.0)                                             # the front: a soft rim
                tail = np.clip(1.0 - (np.float32(r) - rho) / np.float32(band), 0.0, 1.0)  # fading back to the touch
                m = np.maximum(m, np.float32(a) * edge * tail * tail)
            if not m.any():
                continue
            pic = FL.drift(spec, (x1 - x0, y1 - y0), 0.0, (p[0] - x0, p[1] - y0), unit)
            k = (m * np.float32(rg["opacity"]))[..., None]
            region = coffee[y0:y1, x0:x1]
            region += (pic - region) * k

    def _pads(self, fx, st, t, D, coffee, fronts):
        """The pads on the coffee, each dipping under its touches and bobbing as the ring fronts pass under it, cut at
        the size it shows at (a step up, shrunk to it) and laid with its shadow."""
        Wt, Ht = st["table"]
        z = TB.zoom_of(D)
        turn = TB.turn_of(D)
        sh = self._sheet_shadow(fx)
        aspect = Wt / Ht
        for i, pad in enumerate(fx["pads"]):
            dip = TB.dip_at(fx["touch"], fx["pads"], i, t, aspect) + TB.pad_bob(pad, fronts, fx["rings"], aspect)
            size = pad["size"] * Ht * z * (1.0 - dip)
            p = D @ np.array([pad["at"][0] * Wt - 0.5, pad["at"][1] * Ht - 0.5, 1.0])
            if size < 1.0 or not _sheet()._on_frame(p, size, self.size):
                continue
            q = PAD_STEP ** math.ceil(math.log(max(size, 2.0)) / math.log(PAD_STEP))
            key = (fx["index"], i, int(q), tuple(pad["color"]), tuple(pad["light"]))
            tile = self._pad_tiles.get(key)
            if tile is None:
                tile = self._pad_tiles[key] = pad_tile(pad, q)
                while len(self._pad_tiles) > PAD_TILES:
                    self._pad_tiles.popitem(last=False)
            else:
                self._pad_tiles.move_to_end(key)
            rgb, a, mid = tile
            k = size / q
            M = _sheet()._sticker_map(p[:2], pad["turn"] + turn, k, k, mid)
            shadow = None if sh is None else (sh[0], (0.5 * sh[1][0], 0.5 * sh[1][1]), max(0.5 * sh[2], 1.0),
                                              PAD_SHADOW)
            lay_tile(coffee, rgb, a, M, shadow)

    def _table_figure(self, fx, st, frame, t, D, out, shadow, img):
        """Her live sticker on the table: her plate's figure carried onto it (_figure_table), dipping with the first pad
        about its middle, through the view; its die-cut rim at the frame's own px and its shadow on what lies under it,
        farther and softer as she is lifted."""
        FF = _ff()
        m = self._figure(fx, frame)
        if m is None:
            return out
        pic = self._shot_frame(fx, frame) if fx.get("shot") else img
        M = self._figure_table(fx, st, frame)
        if fx["pads"]:
            Wt, Ht = st["table"]
            pad = fx["pads"][0]
            fronts = TB.ring_fronts(fx["touch"], fx["rings"], t, st["places"], st["starts"])
            dip = TB.dip_at(fx["touch"], fx["pads"], 0, t, Wt / Ht) + TB.pad_bob(pad, fronts, fx["rings"], Wt / Ht)
            c = (pad["at"][0] * Wt - 0.5, pad["at"][1] * Ht - 0.5)
            M = FX.compose([[1.0 - dip, 0.0, c[0] * dip], [0.0, 1.0 - dip, c[1] * dip]], M)
        Mt = FX.compose(np.asarray(D).tolist(), M)
        cov = np.clip(FF._warp(m, Mt, self.size), 0.0, 1.0)
        if not cov.any():
            return out
        rgb = FF._warp(pic * m[..., None], Mt, self.size)
        e = self._sheet_edge(fx)
        ring = FF._grow(cov, e[1]) if e is not None and e[1] > 0 else cov
        face = rgb + (np.asarray(e[0], np.float32) * (ring - cov)[..., None] if e is not None else 0.0)
        sh = _sheet().lift_shadow(shadow, TB.lift_at(fx, t))
        if sh is not None and sh[3] > 0:
            out = FF._cast(out, ring, sh)
        return face + out * (1.0 - ring[..., None])

    def _inside(self, fx, hit, k, D, st, img, out, cup):
        """`out` with the picture under the table (`img`, the cut's own frame) in the cup through a punch, dissolving in
        (or out) over the punch's `fade`: centred on the cup's middle as the frame shows it, as large as fills the
        coffee (its shorter side the coffee's shorter diameter) and never larger than its own size, which it reaches by
        the punch's `t` (`k`: the zoom's share of the last one), where the coffee covers the frame."""
        p, u = hit
        a = TB.inside_alpha(p, u)
        if a <= 0.0:
            return out
        Wt, Ht = st["table"]
        c = fx["cup"]["center"]
        q = D @ np.array([c[0] * Wt - 0.5, c[1] * Ht - 0.5, 1.0])
        W, H = self.size
        s = min(1.0, max(k, 2.0 * TB.zoom_of(D) * min(fx["cup"]["radius"]) * Ht / min(W, H)))
        C = TB.middle(self.size)
        A = [[s, 0.0, q[0] - s * C[0]], [0.0, s, q[1] - s * C[1]]]
        inside = cv2.warpAffine(np.ascontiguousarray(img, np.float32), np.float32(A), (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        w = (cup * np.float32(a))[..., None]
        return out + (inside - out) * w
