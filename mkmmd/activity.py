"""The activity book on the finished pictures (docs/design.md: Frame effects, Activity book): the picture work of the
`[[fx]]` kind `activity` that mkmmd.core.activity checks and times. ActivityFx is the part of mkmmd.framefx.FrameFx that
draws it: the page (cream paper with its fibres, faint rules and a margin, mkmmd.core.crayon), the frame drawn on it, and
the exercises printed on the page that a crayon does on the hits: a dotted line traced, numbered dots joined, regions
coloured by number. Printed things (dots, dotted lines, numbers, region outlines) are crisp ink; what the crayon does
is wax (mkmmd.core.crayon.stroke and scribble), drawn again `fps` times a second by a hand that wobbles `boil` px.

Outlines are worked out once per item and size (a sprite's or a drawn shape's coverage, its contour started at the top
and run clockwise), her silhouette's on every frame (her figure layer grown by `offset`). Images are float RGB in display
space (0..1), (h, w, 3)."""
import math
from collections import OrderedDict

import cv2
import numpy as np

from . import eyes as EYD
from . import mandala as MDF
from . import picturefx as PFX
from .core import activity as AC
from .core import crayon as CR
from .core import framefx as FX

SHAPE_MAIN = {"star": "gold", "heart": "love", "clover": "pine", "flower": "rose", "butterfly": "iris"}
DOT_INK = 0.95           # how dark a printed dot or number is: printed ink, nearly solid
NUMBER_CAP = 0.016       # a printed number's capitals, frame heights
NUMBER_GAP = 9.0         # px at 1080 between a dot and its number
LINE = 1.6               # px at 1080: a colour-by-number's printed region lines
OUTLINE = 2.4            # px at 1080: its outline
LEGEND_DOT = 0.018       # frame heights: a legend swatch's radius
REVEAL = 0.4             # s a joined picture takes to appear
TIP = (0.085, 0.017)     # frame heights: a crayon's length and width
CACHE = 32               # outlines and colour-by-number pages kept


def _disc(cov, c, r):
    """`cov` (uint8, in place) with a filled antialiased disc at (x, y) px, radius r px."""
    cv2.circle(cov, (int(round(c[0] * 16)), int(round(c[1] * 16))), max(int(round(r * 16)), 1), 255, -1, cv2.LINE_AA, 4)


def _poly(cov, pts, closed, width):
    """`cov` (uint8, in place) with an antialiased polyline through `pts` ((n, 2) px), `width` px thick."""
    q = np.rint(np.asarray(pts, np.float64) * 16.0).astype(np.int32)
    cv2.polylines(cov, [q], closed, 255, max(int(round(width)), 1), cv2.LINE_AA, 4)


def _ink(out, cov, colour, alpha=DOT_INK, at=(0, 0)):
    """`out` (in place) with `colour` laid where `cov` (float 0..1, or uint8) covers, its top left at pixel `at`."""
    c = cov.astype(np.float32) / (255.0 if cov.dtype == np.uint8 else 1.0)
    x0, y0 = at
    h, w = c.shape
    H, W = out.shape[:2]
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if bx <= ax or by <= ay:
        return
    a = (np.float32(alpha) * c[ay - y0:by - y0, ax - x0:bx - x0])[..., None]
    out[ay:by, ax:bx] = out[ay:by, ax:bx] * (1.0 - a) + np.asarray(colour, np.float32) * a


def _box(pts, margin, size):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    x0, y0 = np.floor(p.min(0) - margin).astype(int)
    x1, y1 = np.ceil(p.max(0) + margin).astype(int) + 1
    return max(x0, 0), max(y0, 0), min(x1, size[0]), min(y1, size[1])


def contour(a, level=0.5):
    """The outline of the largest part of a coverage `a` (h, w) as (n, 2) px points, started at its top and running
    clockwise; None when nothing is covered."""
    m = (np.asarray(a) >= level).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        return None
    return np.asarray(AC.start_top([tuple(p) for p in c]))


def regions(rgb, a, k, min_share=0.006, most=AC.MAX_REGIONS, seed=1):
    """A picture (premultiplied `rgb`, coverage `a`) cut into regions by its colours for a colour by number:
    (labels (h, w) int, 0 outside and 1.. a region, [{number, colour, at (x, y) px, area}] per region): its covered
    pixels in `k` colours (k-means in Lab), specks smoothed away, each connected run of a colour a region, the small ones
    (under `min_share` of the picture) given to their nearest neighbour, at most `most` regions; a region's number is
    its colour's (1 the most common), its colour the picture's mean over it, `at` the point deepest inside it."""
    inside = a >= 0.5
    if not inside.any():
        return np.zeros(a.shape, np.int32), []
    col = np.clip(rgb / np.maximum(a, 1e-6)[..., None], 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(col, cv2.COLOR_RGB2Lab)
    data = lab[inside].reshape(-1, 3).astype(np.float32)
    k = int(min(k, len(data)))
    cv2.setRNGSeed(int(seed))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, idx, _ = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(idx.ravel(), minlength=k)
    rank = np.empty(k, np.int32)
    rank[np.argsort(-counts, kind="stable")] = np.arange(k)              # number 1: the most common colour
    cl = np.zeros(a.shape, np.uint8)
    cl[inside] = rank[idx.ravel()] + 1
    cl = cv2.medianBlur(cl, 5) * inside                                    # specks smoothed away, the outline kept
    area = inside.sum()
    big = np.zeros(a.shape, np.int32)
    found = []
    for c in range(1, k + 1):
        n, comp, stats, _ = cv2.connectedComponentsWithStats((cl == c).astype(np.uint8), 8)
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] >= min_share * area:
                found.append((int(stats[j, cv2.CC_STAT_AREA]), c, comp == j))
    found.sort(key=lambda f: -f[0])
    for r, (_, c, m) in enumerate(found[:most], 1):
        big[m] = r
    if not found:
        big[inside] = 1
        found = [(int(area), 1, inside)]
    hole = inside & (big == 0)
    if hole.any():                                                         # the specks and the rest: nearest region
        _, near = cv2.distanceTransformWithLabels((big == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(big > 0)
        owner = np.zeros(int(near.max()) + 1, np.int32)
        owner[near[ys, xs]] = big[ys, xs]
        big[hole] = owner[near[hole]]
    out = []
    for r, (_, c, _) in enumerate(found[:most], 1):
        m = big == r
        if not m.any():
            continue
        d = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        out.append({"label": r, "number": c, "colour": [float(v) for v in col[m].mean(0)], "at": (float(x), float(y)),
                    "area": int(m.sum()), "room": float(d[y, x])})
    return big, out


def wedges(a, centre, count, colours):
    """A drawn shape (coverage `a`) cut into `count` wedges about `centre` (x, y) px for a colour by number: (labels,
    regions) as `regions` gives them, wedge j coloured colours[j % len] with that colour's number."""
    h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ang = (np.degrees(np.arctan2(xs - centre[0], -(ys - centre[1]))) + 360.0) % 360.0     # clockwise from the top
    inside = a >= 0.5
    lab = np.where(inside, (ang * count / 360.0).astype(np.int32) % count + 1, 0)
    out = []
    for j in range(count):
        m = lab == j + 1
        if not m.any():
            continue
        d = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        c = j % len(colours)
        out.append({"label": j + 1, "number": c + 1, "colour": list(colours[c]), "at": (float(x), float(y)),
                    "area": int(m.sum()), "room": float(d[y, x])})
    return lab, out


def crayon_tip(out, tip, direction, colour, h):
    """`out` (in place) with a crayon held to the page, its point at `tip` (x, y) px, leaning back along `direction`
    (a unit vector the line runs, its body up and behind the point as a right hand holds it): the wax cone, the body
    and its paper wrapper, `TIP` frame heights long and wide."""
    L, W = TIP[0] * h, TIP[1] * h
    d = np.asarray(direction, np.float64)
    d = d / max(np.linalg.norm(d), 1e-9)
    back = np.array([0.55, -0.83]) - 0.45 * d                    # up and to the right, a little behind the stroke
    back /= max(np.linalg.norm(back), 1e-9)
    side = np.array([-back[1], back[0]])
    p = np.asarray(tip, np.float64)
    cone = 0.16 * L
    body0, wrap0, wrap1, end = p + back * cone, p + back * 0.38 * L, p + back * 0.86 * L, p + back * L
    darker = tuple(float(c) * 0.72 for c in colour)
    paper = (0.96, 0.94, 0.9)
    parts = [([p, body0 + side * W * 0.5, body0 - side * W * 0.5], darker),
             ([body0 + side * W * 0.5, end + side * W * 0.5, end - side * W * 0.5, body0 - side * W * 0.5], colour),
             ([wrap0 + side * W * 0.52, wrap1 + side * W * 0.52, wrap1 - side * W * 0.52, wrap0 - side * W * 0.52],
              paper)]
    x0, y0, x1, y1 = _box([p, end + side * W, end - side * W, body0 + side * W, body0 - side * W], 3, (out.shape[1],
                                                                                                    out.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return
    for pts, c in parts:
        cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
        q = np.rint((np.asarray(pts) - (x0, y0)) * 16.0).astype(np.int32)
        cv2.fillPoly(cov, [q], 255, cv2.LINE_AA, 4)
        _ink(out, cov, c, 1.0, (x0, y0))
    band = np.zeros((y1 - y0, x1 - x0), np.uint8)               # the wrapper's two printed lines
    for u in (0.5, 0.74):
        a = p + back * u * L
        _poly(band, [a + side * W * 0.5 - (x0, y0), a - side * W * 0.5 - (x0, y0)], False, max(0.06 * W, 1.0))
    _ink(out, band, darker, 0.9, (x0, y0))


class ActivityFx:
    """The activity pages' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t,
    _hit_times, _figure, _hits, and the fonts and sprites the picture effects and the mandalas found)."""

    def _activity_init(self):
        """The pages' fonts and sprites, found now: a missing one is an FX.FxError before any frame is drawn."""
        self._ac_cache, self._ac_fonts, self._ac_sprites, self._ac_pages = OrderedDict(), {}, {}, {}
        for fx in self.fxs:
            if fx["kind"] != "activity":
                continue
            if any(it["kind"] in ("dots", "paint") for it in fx["items"]):
                self._ac_fonts[fx["index"]] = self._font_of(fx["font"])
            for it in fx["items"]:
                if it["sprites"] is not None:
                    files = self._sprite_of(it["sprites"])
                    self._ac_sprites[fx["index"], it["index"]] = str(files[it["pick"] % len(files)])

    # ---------------------------------------------------------------- the page
    def _page(self, fx):
        """The empty page at this output's size: the paper and its fibres, the rules and the margin (kept, and drawn
        again when their colours move: a breathing palette)."""
        key, inks = (fx["index"], self.size), (fx["paper"], fx["rule"], fx["margin"])
        if self._ac_pages.get(key, (None,))[0] != inks:
            w, h = self.size
            px = self.px
            page = np.asarray(fx["paper"], np.float32)[None, None, :] * CR.fibres((h, w), px, fx["seed"], fx["grain"])[
                ..., None]
            if fx["rule"]:
                r = fx["rule"]
                cov = np.zeros((h, w), np.uint8)
                pitch = r["pitch"] * h
                y = pitch
                while y < h:
                    _poly(cov, [(-4.0, y), (w + 4.0, y)], False, r["width"] * px)
                    y += pitch
                _ink(page, cov, r["color"], r["alpha"])
            if fx["margin"]:
                m = fx["margin"]
                cov = np.zeros((h, w), np.uint8)
                x = m["at"] * w
                _poly(cov, [(x, -4.0), (x, h + 4.0)], False, m["width"] * px)
                _ink(page, cov, m["color"], m["alpha"])
            self._ac_pages[key] = (inks, np.clip(page, 0.0, 1.0).astype(np.float32))
        return self._ac_pages[key][1]

    def _activity(self, fx, frame, img):
        t = self._t(frame)
        page = self._page(fx)
        if fx["picture"] == "multiply":
            key = np.maximum(np.asarray(fx["key"], np.float32), 1e-3)
            out = np.clip(page * np.minimum(img / key, 1.0), 0.0, 1.0).astype(np.float32)
        else:
            out = page.copy()
        times = self._hit_times(fx)
        tooth = CR.tooth((self.size[1], self.size[0]), self.px, fx["seed"])
        for it in fx["items"]:
            if t >= it["t"]:
                getattr(self, "_ac_" + it["kind"])(fx, it, frame, t, out, times, tooth)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)

    # ---------------------------------------------------------------- outlines
    def _kept(self, key, make):
        if key in self._ac_cache:
            self._ac_cache.move_to_end(key)
            return self._ac_cache[key]
        v = self._ac_cache[key] = make()
        while len(self._ac_cache) > CACHE:
            self._ac_cache.popitem(last=False)
        return v

    def _tile(self, fx, it, size, colour=None):
        """(key, tile): the item's picture as a sticker tile (mkmmd.mandala.Tile) `size` px on its longer side, its
        sprite (googly eyes painted in) or its drawn shape in `colour` (default the item's), and the key it is kept by."""
        if it["sprites"] is not None:
            path = self._ac_sprites[fx["index"], it["index"]]
            key = ("sprite", path, round(size, 2), fx["index"], it["index"])
            return key, self._kept(key, lambda: MDF.sprite_tile(path, size, 0.0, face=it["face"]))
        name = it["shape"]
        main = tuple(colour if colour is not None else it["color"])
        key = (name, round(size, 2), main)
        if name == "circle":
            def make():
                n = int(math.ceil(size)) + 6
                cov = np.zeros((n, n), np.uint8)
                _disc(cov, ((n - 1) / 2.0, (n - 1) / 2.0), size / 2.0)
                a = cov.astype(np.float32) / 255.0
                return MDF.tile(np.asarray(main, np.float32) * a[..., None], a, ((n - 1) / 2.0, (n - 1) / 2.0), 0.0)
            return key, self._kept(key, make)
        return key, self._kept(key, lambda: MDF.shape_tile(name, main, main, (0.15, 0.13, 0.2), size, 0.0))

    def _placement(self, it):
        """The affine map (tile px to the frame's) of an item: its tile's anchor at `at`, turned `angle` degrees."""
        w, h = self.size
        return lambda tile: MDF.placed(tile, (it["at"][0] * w - 0.5, it["at"][1] * h - 0.5), it["angle"], 1.0)

    def _outline(self, fx, it, frame):
        """(points (n, 2) px of the frame, closed): the item's outline as the frame shows it."""
        w, h = self.size
        if it["points"] is not None:
            return np.asarray([(x * w - 0.5, y * h - 0.5) for x, y in it["points"]], np.float64), it["closed"]
        if it["shape"] == AC.FIGURE:
            m = self._figure(fx, frame)
            if m is None or float(m.max()) < 0.5:
                return None, True
            off = it["offset"] * self.px
            if off > 0:
                d = cv2.distanceTransform((m < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
                m = (d <= off).astype(np.float32)
            c = contour(m)
            return c, True
        key, tile = self._tile(fx, it, it["size"] * h)
        c = self._kept(("outline",) + key, lambda: contour(tile.a))
        if c is None:
            return None, True
        M = self._placement(it)(tile)
        return c @ M[:, :2].T + M[:, 2], True

    # ---------------------------------------------------------------- the crayon
    def _crayon(self, fx, out, path, closed, width, colour, tooth, key, frame):
        """`out` (in place) with a crayon line along `path` ((n, 2) px), `width` px at 1080, wobbling as the hand that
        draws it again `fps` times a second."""
        if len(path) < 2:
            return
        px, t = self.px, self._t(frame)
        step = AC.boil_step(fx, t)
        rng = np.random.default_rng([fx["seed"], fx["index"], *key, step % (1 << 32)])
        p = np.asarray(path, np.float64)
        wob = fx["boil"] * px
        if wob > 0 and len(p) > 2:
            k = max(int(len(p) / 8), 2)
            at = np.linspace(0, len(p) - 1, k)
            r = rng.uniform(-wob, wob, (k, 2))
            p = p + np.stack([np.interp(np.arange(len(p)), at, r[:, 0]), np.interp(np.arange(len(p)), at, r[:, 1])], 1)
        wpx = width * px
        x0, y0, x1, y1 = _box(p, wpx + 6, self.size)
        if x1 <= x0 or y1 <= y0:
            return
        cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
        _poly(cov, p - (x0, y0), closed, wpx)
        a = CR.stroke(cov > 127, wpx, px, rng, tooth[y0:y1, x0:x1], 0.8)
        _ink(out, a, colour, 0.92, (x0, y0))

    def _dotted(self, out, path, closed, gap, radius, colour):
        """`out` (in place) with printed dots every `gap` px along `path`."""
        p = np.asarray(path, np.float64)
        seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0), axis=1)
        n = max(int(seg.sum() / max(gap, 1.0)), 2)
        pts = AC.resample([tuple(q) for q in p], n, closed)
        x0, y0, x1, y1 = _box(pts, radius + 3, self.size)
        if x1 <= x0 or y1 <= y0:
            return
        cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for q in pts:
            _disc(cov, (q[0] - x0, q[1] - y0), radius)
        _ink(out, cov, colour, DOT_INK, (x0, y0))

    def _number(self, fx, out, text, at, colour, cap=NUMBER_CAP):
        """`out` (in place) with a small printed number centred at `at` (x, y) px."""
        font = self._ac_fonts[fx["index"]]
        em = PFX._em(font, cap * self.size[1])
        fill, _, base, x0, adv = PFX._line(text, font, em, 0)
        ys = np.nonzero(fill.max(1) > 0.05)[0]
        mid = 0.5 * (ys.min() + ys.max()) if len(ys) else base
        _ink(out, fill, colour, DOT_INK, (int(round(at[0] - x0 - adv / 2.0)), int(round(at[1] - mid))))

    # ---------------------------------------------------------------- the exercises
    def _ac_trace(self, fx, it, frame, t, out, times, tooth):
        path, closed = self._outline(fx, it, frame)
        if path is None or len(path) < 2:
            return
        self._dotted(out, path, closed, it["dash"] * self.px, 1.9 * self.px, fx["ink"])
        n = AC.steps(it, times)
        share = AC.progress(it, t, times, fx["snap"]) / n
        if share <= 0:
            return
        done = AC.head([tuple(q) for q in path], share, closed)
        self._crayon(fx, out, done, False, it["width"], it["color"], tooth, (it["index"], 1), frame)
        last = max([h for h in AC.item_hits(it, times) if h <= t], default=None)
        if it["tip"] and last is not None and share < 1.0 + 1e-9 and t - last < fx["snap"] + 0.35 and len(done) > 1:
            d = np.subtract(done[-1], done[-2])
            self._tip(out, done[-1], d, it["color"])

    def _tip(self, out, at, direction, colour):
        crayon_tip(out, at, direction, colour, self.size[1])

    def _ac_dots(self, fx, it, frame, t, out, times, tooth):
        path, _ = self._outline(fx, it, frame)
        if path is None:
            return
        pts = AC.resample([tuple(q) for q in path], it["count"], True)
        c = np.mean(pts, axis=0)
        n = len(pts)
        per = AC.per_step(it, n, times)
        joined = min(AC.progress(it, t, times, fx["snap"]) * per, float(n))
        k = int(math.floor(joined + 1e-9))
        if joined > 0:
            line = pts[:k + 1] + ([pts[0]] if k >= n else [])
            if k < n:
                a, b = pts[k], pts[(k + 1) % n]
                u = joined - k
                line = pts[:k + 1] + ([(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)] if u > 0 else [])
            self._crayon(fx, out, line, False, it["width"], it["color"], tooth, (it["index"], 2), frame)
        if it["reveal"] and joined >= n - 1e-9:
            hits = AC.item_hits(it, times)
            need = int(math.ceil(n / float(per)))
            closed_at = (hits[need - 1] if need - 1 < len(hits) else hits[-1]) + fx["snap"]
            u = min(max((t - closed_at) / REVEAL, 0.0), 1.0)
            if u > 0:
                self._reveal(fx, it, out, u * u * (3.0 - 2.0 * u), t, times, tooth, frame)
        r = it["dot"] * self.px
        x0, y0, x1, y1 = _box(pts, r + 3, self.size)
        cov = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
        for q in pts:
            _disc(cov, (q[0] - x0, q[1] - y0), r)
        _ink(out, cov, fx["ink"], DOT_INK, (x0, y0))
        if it["numbers"]:
            for j, q in enumerate(pts):
                v = np.subtract(q, c)
                v = v / max(np.linalg.norm(v), 1e-9)
                gap = r + NUMBER_GAP * self.px + 0.5 * NUMBER_CAP * self.size[1]
                self._number(fx, out, str(j + 1), (q[0] + v[0] * gap, q[1] + v[1] * gap), fx["ink"])

    def _reveal(self, fx, it, out, u, t, times, tooth, frame):
        """The picture the joined dots outline appearing in it (`u` 0..1): the sprite, its pupils swinging, or the
        drawn shape coloured in with the crayon."""
        h = self.size[1]
        _, tile = self._tile(fx, it, it["size"] * h)
        M = self._placement(it)(tile)
        if it["sprites"] is not None:
            pupils = EYD.pupils(it["face"], tile, t, EYD.hit_times(it["face"], times, self._hits),
                                (fx["index"], it["index"])) if it["face"] else None
            laid = MDF.lay(out.copy(), tile, M, pupils=pupils)
            out[:] = out * (1.0 - u) + laid * u
            return
        th, tw = tile.a.shape
        corners = np.array([[0.0, 0.0], [tw, 0.0], [0.0, th], [tw, th]]) @ M[:, :2].T + M[:, 2]
        x0, y0, x1, y1 = _box(corners, 2, self.size)
        if x1 <= x0 or y1 <= y0:
            return
        Mt = M.copy()
        Mt[:, 2] -= (x0, y0)
        cov = MDF._ff()._warp(tile.a, Mt, (x1 - x0, y1 - y0))
        rng = np.random.default_rng([fx["seed"], fx["index"], it["index"], AC.boil_step(fx, t) % (1 << 32), 5])
        a = CR.scribble(cov >= 0.5, self.px, rng, tooth[y0:y1, x0:x1], 35.0, 0.85) * np.float32(u)
        _ink(out, a, it["color"], 0.92, (x0, y0))

    def _paint_page(self, fx, it):
        """A colour by number's printed page in the item's tile: (tile, labels, regions, print coverage (h, w))."""
        h = self.size[1]
        key, tile = self._tile(fx, it, it["size"] * h, colour=(1.0, 1.0, 1.0))

        def make():
            if it["sprites"] is not None:
                lab, regs = regions(tile.rgb, tile.a, it["colors"], seed=fx["seed"])
            else:
                lab, regs = wedges(tile.a, tile.anchor, it["count"], it["colors"])
            edge = np.zeros(lab.shape, bool)
            edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            edge[1:] |= lab[1:] != lab[:-1]
            inner = cv2.dilate((edge & (lab > 0)).astype(np.uint8), np.ones((2, 2), np.uint8)).astype(np.float32)
            inner = cv2.GaussianBlur(inner, (0, 0), max(0.35 * LINE * self.px, 0.3))
            outer = np.zeros(lab.shape, np.uint8)
            c = contour(tile.a)
            if c is not None:
                _poly(outer, c, True, OUTLINE * self.px)
            ink = np.maximum(np.clip(inner * 1.6, 0.0, 1.0), outer.astype(np.float32) / 255.0)
            return lab, regs, ink
        lab, regs, ink = self._kept(("paint",) + key, make)
        return tile, lab, regs, ink

    def _ac_paint(self, fx, it, frame, t, out, times, tooth):
        tile, lab, regs, ink = self._paint_page(fx, it)
        if not regs:
            return
        M = self._placement(it)(tile)
        FF = MDF._ff()
        order = sorted(regs, key=lambda r: (r["number"], -r["area"]))
        per = AC.per_step(it, len(order), times)
        filled = min(AC.progress(it, t, times, fx["snap"]) * per, float(len(order)))
        step = AC.boil_step(fx, t)
        if filled > 0:
            wax = np.zeros(lab.shape + (3,), np.float32)
            cover = np.zeros(lab.shape, np.float32)
            th, tw = lab.shape
            grain = CR.tooth((th, tw), self.px, fx["seed"] + 17 * (it["index"] + 1))
            for j, r in enumerate(order):
                share = min(max(filled - j, 0.0), 1.0)
                if share <= 0:
                    break
                rng = np.random.default_rng([fx["seed"], fx["index"], it["index"], r["label"], step % (1 << 32)])
                m = lab == r["label"]
                a = CR.scribble(m, self.px, rng, grain, 20.0 + 140.0 * rng.random(), 0.55 + 0.4 * share)
                a = a * np.float32(min(share * 1.5, 1.0)) * np.float32(0.93)
                wax = wax * (1.0 - a[..., None]) + np.asarray(r["colour"], np.float32) * a[..., None]
                cover = cover + (1.0 - cover) * a
            FFw = FF._warp(np.dstack([wax, cover]), M, self.size)
            a = np.clip(FFw[..., 3], 0.0, 1.0)
            out[:] = out * (1.0 - a[..., None]) + FFw[..., :3]
        printed = np.clip(FF._warp(ink, M, self.size), 0.0, 1.0)
        out[:] = out * (1.0 - DOT_INK * printed[..., None]) + np.asarray(fx["ink"], np.float32) * (DOT_INK * printed[
            ..., None])
        for r in regs:
            q = M[:, :2] @ np.asarray(r["at"]) + M[:, 2]
            if r["room"] >= 0.45 * NUMBER_CAP * self.size[1]:
                self._number(fx, out, str(r["number"]), q, fx["ink"], NUMBER_CAP * 0.9)
        if it["legend"]:
            self._legend(fx, it, out, M, tile, regs)

    def _legend(self, fx, it, out, M, tile, regs):
        """The key under a colour by number: a crayon swatch per number, its number beside it."""
        h = self.size[1]
        nums = sorted({r["number"]: r["colour"] for r in regs}.items())
        r = LEGEND_DOT * h
        step = 4.2 * r
        th = tile.a.shape[0]
        base = M[:, :2] @ np.asarray((tile.anchor[0], th + 2.5 * r)) + M[:, 2]
        x = base[0] - step * (len(nums) - 1) / 2.0
        for j, (num, col) in enumerate(nums):
            c = (x + j * step, base[1])
            x0, y0, x1, y1 = _box([c], r + 3, self.size)
            if x1 <= x0 or y1 <= y0:
                continue
            cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
            _disc(cov, (c[0] - x0, c[1] - y0), r)
            _ink(out, cov, col, 0.95, (x0, y0))
            ring = np.zeros_like(cov)
            cv2.circle(ring, (int(round((c[0] - x0) * 16)), int(round((c[1] - y0) * 16))), int(round(r * 16)), 255,
                       max(int(round(1.2 * self.px)), 1), cv2.LINE_AA, 4)
            _ink(out, ring, fx["ink"], DOT_INK, (x0, y0))
            self._number(fx, out, str(num), (c[0] + 1.75 * r, c[1]), fx["ink"])
