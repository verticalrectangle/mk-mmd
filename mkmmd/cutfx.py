"""Cut effects composited over the cut's frames, before the grade (docs/design.md: Shots: Transitions and inserts,
Glitches): the post layer (numpy, OpenCV, Pillow). `mk post` and `mk look` hand each frame of the cut to `CutFx.frame`;
frames outside every window come back untouched. The layers an effect needs were rendered next to the cut's frames
(`mk render`, mkmmd.blender.transition); the plan (mkmmd.core.transition) says which.

    expand / collapse   the figure of a silhouette shot (`matte/`, its coverage as a distance field, mkmmd.matte) is turned
                        and scaled about its centre; inside it plays the other shot, outside is the silhouette shot's frame
                        without its figure (`back/`); a rim of `edge` colour runs round it. The scale rises until the figure
                        fills the frame at the last frame of an expand / the first of a collapse, whatever its size.
    slash               a band sweeps across; behind it the incoming shot, ahead of it the outgoing one, the switch under it.
    whip                a whip pan: the outgoing frame is flung off the frame and the incoming shot follows it in, side by
                        side, both smeared along their way by the shutter.
    slice               the outgoing frame cut in two along a line, the halves sliding apart across it and off the frame by
                        the cut, the incoming shot in the widening gap, an `edge` along each half's cut side.
    spin                the cut's own frame (the outgoing shot, from the cut the incoming one) turned about a point faster
                        and faster up to the cut and slowing to rest after it, smeared along its turn by the shutter.
    page                the outgoing frame turned over as a picture book's page onto the incoming shot, its paper back
                        showing (mkmmd.pageturn). The loop's wrap, a page turn at the clip's end, is not one of these:
                        mk post and mk look turn the finished frames over frame 0 (mkmmd.pageturn.wrap).
    insert              a thought bubble over the host frame holds the other shot at reduced size (picture in picture),
                        popping in and out, or growing until its picture is the whole frame; or a split screen, the host
                        and the other shot (or a video, mkmmd.footage) side by side, sliding in, trading sides, sliding out;
                        or an iris, a round window from a scene point onto the other shot or a video (mkmmd.iris), with
                        a record, a plate or chrome rings round it (mkmmd.portal).
    glitch              the figure (the frame against its `bare/` layer, the same frame without the glitch's objects)
                        breaks up in jumping slices and colour ghosts and drops out until the bare frame is left
                        (mkmmd.core.glitch)."""
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import iris as IRIS
from . import matte as M
from . import pageturn as PAGE
from . import post as P
from .core import glitch as GL
from .core import iris as IR
from .core import transition as TR
from .core import tween as TW

COVER_MARGIN = 1.02                      # the scale that fills the frame, with a little to spare
DOT_RADII = (0.12, 0.19, 0.28)           # the trailing circles' radii, as shares of the bubble's half-height
DOT_FIT = (0.085, 0.13, 0.18)            # and at most these shares of the trail's length (head edge to cloud rim)
DOT_AT = (0.12, 0.40, 0.75)              # where they sit along it, as shares measured from the head edge
SMEAR_STEP = 1.5                         # px between the turned copies a spin's smear averages, at the frame's far corner
MAX_TAPS = 2048                          # and at most this many copies


def read_point(frames_dir, key, frame, size):
    """(pixel coordinates, frame heights per metre at its depth) of a projected point, point/<key>/<frame>.json in an
    output's frame folder; a FileNotFoundError when it is missing, unreadable or from before its scale was stored."""
    rel = TR.point_rel(key, frame)
    try:
        data = json.loads((Path(frames_dir) / rel).read_text(encoding="utf-8"))
        return M.frac_to_px(data["p"], size), float(data["m"])
    except (OSError, ValueError, KeyError):
        raise FileNotFoundError(f"{Path(frames_dir) / rel}: layer missing, unreadable or from before its scale was stored "
                                f"(mk render makes it)") from None


def read_figure(frames_dir, key, frame, size):
    """The coverage (h, w) float32 in 0..1 of a collage's figure, figure/<key>/<frame>.png in an output's frame folder,
    at `size` (width, height); a FileNotFoundError when it is missing, empty or unreadable."""
    path = Path(frames_dir) / TR.figure_rel(key, frame)
    try:
        im = Image.open(path).convert("L")
        if im.size != tuple(size):
            im = im.resize(tuple(size), Image.BILINEAR)
        return np.asarray(im, np.float32) / 255.0
    except (OSError, ValueError):
        raise FileNotFoundError(f"{path}: layer missing or unreadable (mk render makes it)") from None


def _along(pic, at, length, across):
    """The picture `pic` moved along the axis (x when `across`, else y) so that its first pixel lands at `at`, rounded to a
    whole pixel (a picture at rest stays sharp), `length` pixels long, its edge pixels repeated beyond it."""
    n = pic.shape[1] if across else pic.shape[0]
    idx = np.clip(np.arange(length) - int(round(at)), 0, n - 1)
    return pic[:, idx] if across else pic[idx]


class CutFx:
    """The effects of a plan over one output's frames. `frames_dir` holds the cut's frames and the layers, `size` is
    (width, height) of the frames being graded, `output` the aspect's name (for an insert's per-aspect geometry),
    `footage_of` a clip -> mkmmd.footage.Clip opener (mkmmd.footage.source) for a split's video half or an iris's video,
    `sprite_of` a sprite finder (mkmmd.mandala.sprite_source) for a plate portal's sprites and `hits_of` a callable
    giving the [audio] hits file ({lane: [...]}, or None) for their googly eyes' lanes."""

    def __init__(self, plan, frames_dir, size, output=None, footage_of=None, sprite_of=None, hits_of=None):
        self.plan, self.dir, self.size, self.output = plan, Path(frames_dir), (int(size[0]), int(size[1])), output
        self.footage_of, self.sprite_of, self.hits_of = footage_of, sprite_of, hits_of
        self.px = min(self.size) / 1080.0                                  # widths are given at 1080 on the short side
        self.at = {}
        for tr in plan["transitions"]:
            if tr.get("still") is not None:                               # the loop's wrap turns the finished frames
                continue
            for f in range(tr["first"], tr["last"] + 1):
                self.at[f] = ("transition", tr)
        for ins in plan["inserts"]:
            for f in range(ins["f0"], ins["f1"]):
                self.at[f] = ("insert", ins)
        for g in plan["glitches"]:
            for f in range(g["f0"], g["f1"]):
                self.at[f] = ("glitch", g)
        self._top, self._grown, self._centre, self._clips = {}, {}, {}, {}
        self._sprites, self._tiles, self._hits = {}, {}, {}
        self.notes = []

    # ---------------------------------------------------------------- what is there
    def active(self, frame):
        return frame in self.at

    def missing(self, frames):
        """The layer files the given frames' effects need that are not on disk, are empty, or (a point) predate its scale."""
        out = []
        for f, items in TR.demands(self.plan, frames).items():
            for item in items:
                for rel in TR.rel_paths(item, f):
                    p = self.dir / rel
                    if not p.exists() or p.stat().st_size == 0 or (item["kind"] == "point" and TR.stale_point(p)):
                        out.append(rel)
        return out

    def check(self, frames):
        """Settle what depends on the pictures before anything is encoded (a TransitionError when one cannot be): the scale
        at which each expand / collapse that touches `frames` fills the frame, the point a slice through the figure cuts
        through, that a split's or an iris's video can be read over its window and that a plate portal's sprites are
        there."""
        want = set(frames)
        for tr in self.plan["transitions"]:
            if not any(f in want for f in range(tr["first"], tr["last"] + 1)):
                continue
            if tr["kind"] in ("expand", "collapse"):
                self._top_scale(tr)
            elif tr["kind"] == "slice":
                self._slice_centre(tr)
        for ins in self.plan["inserts"]:
            if not any(f in want for f in range(ins["f0"], ins["f1"])):
                continue
            if ins.get("clip"):
                self._video(ins)
            if _plate_sprites(ins):
                self._sprite_files(ins)
                self._eye_hits(ins)

    def frame(self, frame, main):
        """The cut's frame `main` (float RGB) with the effect of `frame`'s window composited on it."""
        hit = self.at.get(frame)
        if hit is None:
            return main
        kind, spec = hit
        if kind == "insert":
            draw = {"split": self._split, "iris": self._iris}.get(spec["shape"], self._insert)
            return draw(spec, frame, main)
        if kind == "glitch":
            return GL.frame(main, self._rgb(TR.bare_rel(f"g{spec['index']}", frame)), frame - spec["f0"],
                            spec["f1"] - spec["f0"], seed=spec["seed"], shift=spec["shift"], split=spec["split"] * self.px,
                            colors=spec["colors"])
        if spec["kind"] == "slash":
            return self._slash(spec, frame, main)
        if spec["kind"] == "whip":
            return self._whip(spec, frame, main)
        if spec["kind"] == "slice":
            return self._slice(spec, frame, main)
        if spec["kind"] == "spin":
            return self._spin(spec, frame, main)
        if spec["kind"] == "page":
            return self._page(spec, frame, main)
        return self._matte(spec, frame, main)

    # ---------------------------------------------------------------- layers
    def _rgb(self, rel):
        img = P.read(self.dir / rel, self.size)
        if img is None:
            raise FileNotFoundError(f"{self.dir / rel}: layer missing or unreadable (mk render makes it)")
        return img

    def _alpha(self, rel):
        """A coverage layer at its own size (a matte is drawn finer than the frame: see Looks.matte)."""
        try:
            im = Image.open(self.dir / rel).convert("L")
        except (OSError, ValueError):
            raise FileNotFoundError(f"{self.dir / rel}: layer missing or unreadable (mk render makes it)") from None
        return np.asarray(im, np.float32) / 255.0

    def _point(self, key, frame):
        return read_point(self.dir, key, frame, self.size)

    def _plate(self, shot, frame, main):
        """What `shot` shows at `frame`: the cut's own frame when it is the shot in the cut there, else its plate."""
        return main if TR.in_force(self.plan, shot, frame) else self._rgb(TR.plate_rel(shot, frame))

    # ---------------------------------------------------------------- expand, collapse
    def _figure(self, tr, frame):
        """(distance field, centre in frame pixels) of the matte shot's figure at `frame`; centre None when the frame holds no
        figure. The centre is the figure's centroid, a point of the frame or a projected point, moved into the figure when
        it falls outside (or, for the centroid, is shallower than a third of the figure's deepest part)."""
        alpha = self._alpha(TR.matte_rel(tr["matte"], frame))
        S = M.signed_distance(alpha)
        k = alpha.shape[1] / self.size[0]                                    # map pixels per frame pixel
        cen = tr["center"]
        if cen["mode"] == "subject":
            p = M.inside_point(S, M.centroid(alpha), max(1.0, 0.35 * float(-S.min())))
        else:
            at = M.frac_to_px(cen["at"], self.size) if cen["mode"] == "frame" else self._point(f"t{tr['index']}", frame)[0]
            p = M.inside_point(S, ((at[0] + 0.5) * k - 0.5, (at[1] + 0.5) * k - 0.5), 0.5 * k)
        c = None if p is None else ((p[0] + 0.5) / k - 0.5, (p[1] + 0.5) / k - 0.5)
        return M.AlphaField(S, k), c

    def _top_scale(self, tr):
        """The scale the figure reaches where it must fill the frame: the spec's number, or, when that is too small for
        this figure or the spec says "fill", the one that covers the frame."""
        if tr["index"] not in self._top:
            frame = TR.covering_frame(tr)
            field, c = self._figure(tr, frame)
            if c is None:
                raise TR.TransitionError(f"transition {tr['index']} ({tr['kind']} at {tr['at']} s): shot {tr['matte']!r} "
                                         f"shows no figure at frame {frame}, so it cannot fill the frame")
            need = M.cover_scale(field, self.size, c, c, tr["turn"][1])
            if not np.isfinite(need):
                raise TR.TransitionError(f"transition {tr['index']} ({tr['kind']} at {tr['at']} s): the centre "
                                         f"{tuple(round(v) for v in c)} of frame {frame} is not inside the figure of "
                                         f"{tr['matte']!r}: it cannot fill the frame (centre = \"subject\" always can)")
            spec_hi = tr["scale"][1]
            top = need * COVER_MARGIN if spec_hi is None else max(spec_hi, need * COVER_MARGIN)
            self._top[tr["index"]] = top
            self.notes.append({"transition": tr["index"], "scale": [tr["scale"][0], round(top, 3)],
                               "raised": spec_hi is not None and top > spec_hi, "needed": round(need, 3)})
        return self._top[tr["index"]]

    def _matte(self, tr, frame, main):
        k = frame - tr["first"]
        field, c = self._figure(tr, frame)
        back = self._rgb(TR.back_rel(tr["matte"], frame))
        plate = self._plate(tr["plate"], frame, main)
        if c is None:                                                        # no figure in this frame: nothing to show it in
            return back
        scale, turn = TR.pose(tr, k, self._top_scale(tr))
        d = M.placed(field, self.size, c, c, scale, turn)
        out = back
        if tr["edge"]:
            out = M.paint(out, tr["edge"]["color"], M.inside_band(d, tr["edge"]["width"] * self.px))
        return M.paint(out, plate, M.coverage(d))

    # ---------------------------------------------------------------- slash
    def _slash(self, tr, frame, main):
        w, h = self.size
        incoming = self._plate(tr["plate"], frame, main)
        diag = float(np.hypot(w, h))
        a = np.radians(tr["angle"])
        nx, ny = float(np.cos(a)), float(np.sin(a))                          # across the band: the way it sweeps
        if tr["dir"] == "left":
            nx, ny = -nx, -ny
        xs, ys = M.grid(h, w)
        s = (xs - (w - 1) / 2.0) * nx + (ys - (h - 1) / 2.0) * ny            # px along the sweep, 0 at the frame's centre
        sec = tr["second"]
        extra = (sec["width"] + abs(sec["offset"])) * diag if sec else 0.0   # the thin band must clear the frame too
        reach = (abs(nx) * w + abs(ny) * h) / 2.0 + tr["width"] * diag / 2.0 + extra
        pos = TW.lerp(-reach, reach, TR.slash_progress(tr, frame - tr["first"]))
        out = M.paint(main, incoming, M.coverage(s - pos))                   # behind the band: the incoming shot
        if sec:
            half = (tr["width"] + sec["width"]) * diag / 2.0 + abs(sec["offset"]) * diag
            mid = pos + (-half if sec["offset"] < 0 else half)               # the thin band's centre, a gap clear of the main one
            out = M.paint(out, sec["color"], M.coverage(np.abs(s - mid) - sec["width"] * diag / 2.0))
        return M.paint(out, tr["color"], M.coverage(np.abs(s - pos) - tr["width"] * diag / 2.0))

    # ---------------------------------------------------------------- whip
    def _whip(self, tr, frame, main):
        """The outgoing frame and the incoming shot side by side, flung along the whip's way by its eased progress and
        smeared over the way each frame's shutter (`blur` frames) covers."""
        w, h = self.size
        incoming = self._plate(tr["plate"], frame, main)
        k = frame - tr["first"]
        across = tr["dir"] in ("left", "right")
        span = w if across else h
        u = TR.whip_progress(tr, k)
        speed = (TR.whip_progress(tr, k + 0.5) - TR.whip_progress(tr, k - 0.5)) * span      # px per frame
        first, second = (main, incoming) if tr["dir"] in ("left", "up") else (incoming, main)
        strip = np.concatenate([first, second], axis=1 if across else 0)                   # in the order they pass
        blur = int(round(abs(speed) * tr["blur"]))
        if blur > 1:
            strip = cv2.blur(strip, (blur, 1) if across else (1, blur), borderType=cv2.BORDER_REPLICATE)
        start = u * span if tr["dir"] in ("left", "up") else (1.0 - u) * span                # where the frame looks in
        M = np.float32([[1, 0, -start], [0, 1, 0]]) if across else np.float32([[1, 0, 0], [0, 1, -start]])
        return cv2.warpAffine(strip, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---------------------------------------------------------------- slice
    def _slice_centre(self, tr):
        """The point a slice's cut runs through, in frame pixels: a point of the frame, or the centroid of the outgoing flat
        shot's figure on the window's first frame (a TransitionError when that frame shows none)."""
        cen = tr["center"]
        if cen["mode"] == "frame":
            return M.frac_to_px(cen["at"], self.size)
        if tr["index"] not in self._centre:
            alpha = self._alpha(TR.matte_rel(tr["matte"], tr["first"]))
            if float(alpha.max()) <= M.EPS:
                raise TR.TransitionError(f"transition {tr['index']} (slice at {tr['at']} s): shot {tr['matte']!r} shows no "
                                         f"figure at frame {tr['first']}, so center = \"figure\" has nothing to cut through")
            k = alpha.shape[1] / self.size[0]                                # map pixels per frame pixel
            x, y = M.centroid(alpha)
            self._centre[tr["index"]] = ((x + 0.5) / k - 0.5, (y + 0.5) / k - 0.5)
        return self._centre[tr["index"]]

    def _slice(self, tr, frame, main):
        """The outgoing frame cut in two along a line through the centre, the halves sliding apart across it (each its own
        way, off the frame by the cut), the incoming shot in the widening gap, an `edge` along each half's cut side."""
        w, h = self.size
        incoming = self._plate(tr["plate"], frame, main)
        c = self._slice_centre(tr)
        nx, ny = TR.slice_normal(tr["angle"])
        xs, ys = M.grid(h, w)
        s = (xs - c[0]) * nx + (ys - c[1]) * ny                              # px across the cut, + on the normal's side
        p = TR.slice_progress(tr, frame - tr["first"])
        ew = tr["edge"]["width"] * self.px if tr["edge"] else 0.0
        out = incoming
        for sign, reach in zip((1.0, -1.0), TR.slice_reach(self.size, c, (nx, ny))):
            dx, dy = sign * p * reach * nx, sign * p * reach * ny
            half = cv2.warpAffine(main, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            inside = sign * s - p * reach                                    # px into the moved half from its cut side
            sx, sy = xs - dx, ys - dy                                        # where each pixel of it was in the frame
            room = np.minimum(np.minimum(sx + 0.5, w - 0.5 - sx), np.minimum(sy + 0.5, h - 0.5 - sy))
            if ew > 0.0:
                half = M.paint(half, tr["edge"]["color"], np.clip(ew + 0.5 - inside, 0.0, 1.0))
            out = M.paint(out, half, np.clip(inside + 0.5, 0.0, 1.0) * np.clip(room + 0.5, 0.0, 1.0))
        return out

    # ---------------------------------------------------------------- spin
    def _spin(self, tr, frame, main):
        """The cut's own frame (the outgoing shot before the cut, the incoming one from it) turned about the centre and
        scaled, smeared along its turn over the shutter's sweep: the mean of copies turned at even steps across it."""
        w, h = self.size
        st = TR.spin_state(tr, frame)
        c = M.frac_to_px(tr["center"]["at"], self.size)
        if tr["back"] == "mirror":
            border, value = cv2.BORDER_REFLECT_101, (0.0, 0.0, 0.0)
        else:
            border, value = cv2.BORDER_CONSTANT, tuple(float(v) for v in tr["back"])
        lo, hi = st["smear"]
        radius = max(math.hypot(x - c[0], y - c[1]) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5))
        sweep = math.radians(hi - lo) * radius                               # px the far corner moves with the shutter open
        taps = 1 if sweep <= SMEAR_STEP else min(MAX_TAPS, 2 ** math.ceil(math.log2(sweep / SMEAR_STEP)))
        if taps == 1:
            m = cv2.getRotationMatrix2D(c, -(st["turn"] + (lo + hi) / 2.0), st["scale"])   # cv2 turns counter-clockwise
            return cv2.warpAffine(main, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)
        # a canvas round the frame that holds the disc its pixels turn on, so the copies bring in what turns into the frame
        r = math.ceil(radius) + 2
        pl, pt = max(0, r - math.floor(c[0])), max(0, r - math.floor(c[1]))
        pr, pb = max(0, r - math.floor(w - 1 - c[0])), max(0, r - math.floor(h - 1 - c[1]))
        big = (w + pl + pr, h + pt + pb)
        m = cv2.getRotationMatrix2D(c, -(st["turn"] + lo + (hi - lo) / taps / 2.0), st["scale"])
        m[0, 2] += pl
        m[1, 2] += pt
        acc = cv2.warpAffine(main, m, big, flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)
        centre, turn = (c[0] + pl, c[1] + pt), (hi - lo) / 2.0
        while taps > 1:                      # each pass averages the copies with themselves turned on: 2^k evenly spread
            turned = cv2.warpAffine(acc, cv2.getRotationMatrix2D(centre, -turn, 1.0), big, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            acc = cv2.addWeighted(acc, 0.5, turned, 0.5, 0.0)
            turn /= 2.0
            taps //= 2
        return np.ascontiguousarray(acc[pt:pt + h, pl:pl + w])

    # ---------------------------------------------------------------- page
    def _page(self, tr, frame, main):
        """The outgoing frame turned over as a page (mkmmd.pageturn) onto the incoming shot's plate."""
        return PAGE.frame(tr, frame, main, self._plate(tr["plate"], frame, main))

    # ---------------------------------------------------------------- insert
    def _dots(self, start, centre, a, b):
        """[(x, y, radius)] of the three circles of the trail between `start` (the edge of the head, where the trail begins)
        and the bubble (centre, half-axes a, b), nearest the head first; [] when there is no room between them."""
        dx, dy = start[0] - centre[0], start[1] - centre[1]
        dist = float(np.hypot(dx, dy))
        if dist < 1e-3:
            return []
        ux, uy = dx / dist, dy / dist
        edge = 1.0 / float(np.hypot(ux / a, uy / b))                         # from the centre to the bubble's rim, toward it
        gap = dist - edge
        if gap < 3.0:
            return []
        out = []
        for share, base, cap in zip(DOT_AT, DOT_RADII, DOT_FIT):
            r = min(base * b, cap * gap)
            along = dist - share * gap                                       # from the bubble's centre
            out.append((centre[0] + ux * along, centre[1] + uy * along, r))
        return out

    def _grow(self, ins, cloud, geo):
        """The scale at which the bubble, turned as `expand.turn` leaves it and sitting in the frame's middle, fills the frame."""
        key = (ins["index"], geo["size"], geo["ratio"])
        if key not in self._grown:
            w, h = self.size
            need = M.cover_scale(cloud, self.size, (0.0, 0.0), ((w - 1) / 2.0, (h - 1) / 2.0), ins["expand"]["turn"])
            self._grown[key] = need * COVER_MARGIN
        return self._grown[key]

    def _insert(self, ins, frame, main):
        w, h = self.size
        geo = TR.insert_for(ins, self.output)
        st = TR.insert_state(ins, frame)
        ring, col = ins["outline"]["width"] * self.px, ins["outline"]["color"]
        anchor, per_metre = self._point(f"i{ins['index']}", frame)
        b = geo["size"] * h / 2.0
        a = b * geo["ratio"]
        pad = ring + 0.015 * h
        bx = anchor[0] + geo["offset"][0] * h
        by = anchor[1] + geo["offset"][1] * h
        bx = (w - 1) / 2.0 if a + pad > (w - 1) / 2.0 else min(max(bx, a + pad), w - 1 - a - pad)
        by = (h - 1) / 2.0 if b + pad > (h - 1) / 2.0 else min(max(by, b + pad), h - 1 - b - pad)
        out = main
        dx, dy = bx - anchor[0], by - anchor[1]                              # the trail begins where the head (a sphere of
        dist = float(np.hypot(dx, dy))                                       # `radius` metres about the anchor) ends,
        reach = min(ins["radius"] * per_metre * h, dist)                     # on the side of the bubble
        start = (anchor[0] + dx / dist * reach, anchor[1] + dy / dist * reach) if dist > 1e-3 else anchor
        for (x, y, r), s in zip(self._dots(start, (bx, by), a, b), st["dots"]):
            if s > 0.01:
                out = M.paint(out, col, M.coverage(M.disc(self.size, (x, y), r * s)))
        cloud = M.CloudField(a, b)
        if st["phase"] == "expand":
            p = st["p"]
            centre = (TW.lerp(bx, (w - 1) / 2.0, p), TW.lerp(by, (h - 1) / 2.0, p))
            scale = TW.geometric(1.0, self._grow(ins, cloud, geo), p)
            turn = ins["expand"]["turn"] * p
            k = TW.geometric(max(2.0 * a / w, 2.0 * b / h) * 1.04, 1.0, p)
        else:
            centre, scale, turn = (bx, by), st["bubble"], 0.0
            k = max(2.0 * a / w, 2.0 * b / h) * 1.04 * scale
        if scale < 0.01:
            return out
        reach = cloud.radius() * scale + ring + 4.0
        d = M.placed(cloud, self.size, (0.0, 0.0), centre, scale, turn,
                     crop=(centre[0] - reach, centre[1] - reach, centre[0] + reach, centre[1] + reach))
        out = M.paint(out, col, M.inside_band(d, ring))
        picture = self._plate(ins["shot"], frame, main)
        if abs(k - 1.0) > 1e-4 or abs(centre[0] - (w - 1) / 2.0) > 1e-3 or abs(centre[1] - (h - 1) / 2.0) > 1e-3:
            picture = M.warp_scaled(picture, centre, k)
        return M.paint(out, picture, M.coverage(d))

    # ---------------------------------------------------------------- split
    def _video(self, ins):
        """The mkmmd.footage.Clip of an insert's video (a split's or an iris's), opened once; a TransitionError when it
        cannot be read or its span starts past its end without `loop`."""
        if ins["index"] not in self._clips:
            c = ins["clip"]
            what = f"insert {ins['index']} ({ins['shape']} {ins['from']}-{ins['to']} s): clip {c['src']!r}"
            if self.footage_of is None:
                raise TR.TransitionError(f"{what}: no video source was given to the cut effects (footage_of)")
            try:
                clip = self.footage_of(c["src"])
                clip.index(0.0, start=c["in"], rate=c["rate"], loop=c["loop"])
            except FileNotFoundError as e:                                   # mkmmd.footage.FootageError
                raise TR.TransitionError(f"{what}: {e}") from None
            self._clips[ins["index"]] = clip
        return self._clips[ins["index"]]

    def _split(self, ins, frame, main):
        """The frame split in two: the host (the cut's frame) and the inner half (the inner shot's plate or the video), each
        cropped about its focus in its panel. The inner half slides in from its side, the halves trade places on each swap
        (the row running on round like a carousel) and it slides back out; a divider runs along each seam."""
        w, h = self.size
        geo = TR.split_for(ins, self.output, self.size)
        across = geo["side"] in ("left", "right")                            # panels side by side: the axis is x
        length, axis = (w, 0) if across else (h, 1)
        panels = TR.split_panels(TR.split_state(ins, frame), geo["share"], length, geo["side"])
        c = ins["clip"]
        if c is not None:                                                    # the video fits the inner half at rest
            rest = int(math.ceil(geo["share"] * length))
            try:
                inner = self._video(ins).picture((frame - ins["f0"]) / self.plan["fps"], (rest, h) if across else (w, rest),
                                                 start=c["in"], rate=c["rate"], loop=c["loop"], fit=c["fit"],
                                                 focus=tuple(geo["focus"]["inner"]), mirror=c["mirror"], back=tuple(c["back"]))
            except FileNotFoundError as e:
                raise TR.TransitionError(f"insert {ins['index']} (split): clip {c['src']!r}: {e}") from None
        else:
            inner = self._plate(ins["shot"], frame, main)
        edges = np.arange(length, dtype=np.float32)                          # pixel i spans [i, i + 1) along the axis
        out = np.zeros_like(main)
        for role, start, size in panels:
            cov = np.clip(np.minimum(edges + 1.0, start + size) - np.maximum(edges, start), 0.0, 1.0)
            if not cov.any():
                continue
            if role == "host":
                pic, at = main, TR.panel_offset(start, size, geo["focus"]["host"][axis], length)
            elif c is not None:
                pic, at = inner, start
            else:
                pic, at = inner, TR.panel_offset(start, size, geo["focus"]["inner"][axis], length)
            out += _along(pic, at, length, across) * (cov[None, :, None] if across else cov[:, None, None])
        div = ins["divider"]
        if div:
            half = div["width"] * self.px / 2.0
            for s in TR.split_dividers(panels, length):
                band = np.clip(np.minimum(edges + 1.0, s + half) - np.maximum(edges, s - half), 0.0, 1.0)
                out = M.paint(out, div["color"], np.broadcast_to(band[None, :] if across else band[:, None], (h, w)))
        return out

    # ---------------------------------------------------------------- iris
    def _iris(self, ins, frame, main):
        """A round window from the projected anchor onto the inner shot's plate or the video, fitted to the frame
        (mkmmd.iris draws it, mkmmd.core.iris says where)."""
        anchor, per_metre = self._point(f"i{ins['index']}", frame)
        c = ins["clip"]
        if c is None:
            inner = self._plate(ins["shot"], frame, main)
        else:
            try:
                inner = self._video(ins).picture((frame - ins["f0"]) / self.plan["fps"], self.size, start=c["in"],
                                                 rate=c["rate"], loop=c["loop"], fit=c["fit"], focus=(0.5, 0.5),
                                                 mirror=c["mirror"], back=tuple(c["back"]))
            except FileNotFoundError as e:
                raise TR.TransitionError(f"insert {ins['index']} (iris): clip {c['src']!r}: {e}") from None
        if ins["ring"] is None:
            return IRIS.frame(ins, frame, main, inner, anchor, per_metre, self.output)
        fps = self.plan["fps"]
        extra = {}
        if _plate_sprites(ins):
            extra = {"sprites": self._sprite_files(ins), "tiles": self._tiles.setdefault(ins["index"], OrderedDict()),
                     "eyes": self._eyes(ins, frame, anchor, per_metre)}
        return IRIS.frame(ins, frame, main, inner, anchor, per_metre, self.output, t=(frame - ins["f0"]) / fps, **extra)

    def _sprite_files(self, ins):
        """The files of a plate portal's sprites, found once; a TransitionError naming the insert when one finds nothing."""
        if ins["index"] not in self._sprites:
            from . import mandala as MDF
            from .core import framefx as FX
            find = self.sprite_of or MDF.sprite_source()
            try:
                files = find(ins["ring"]["sprites"])
            except FX.FxError as e:
                raise TR.TransitionError(f"insert {ins['index']} (iris {ins['from']}-{ins['to']} s): ring: {e}") from None
            self._sprites[ins["index"]] = [str(f) for f in files]
        return self._sprites[ins["index"]]

    def _eye_hits(self, ins):
        """The clip seconds that kick a plate portal's sprites' googly eyes (mkmmd.eyes.hit_times: their own `times` or
        `lane`; an iris has no hits of its own); [] without eyes."""
        face = ins["ring"]["face"]
        if face is None:
            return []
        if ins["index"] not in self._hits:
            from . import eyes as EYD
            from .core import framefx as FX
            try:
                hits = self.hits_of() if self.hits_of is not None and face.get("lane") is not None else None
                self._hits[ins["index"]] = EYD.hit_times(face, None, hits)
            except (FX.FxError, FileNotFoundError) as e:
                raise TR.TransitionError(f"insert {ins['index']} (iris {ins['from']}-{ins['to']} s): ring: {e}") from None
        return self._hits[ins["index"]]

    def _eyes(self, ins, frame, anchor, per_metre):
        """What a plate portal's googly eyes swing by at `frame`: the clip second, the hits and where the window was at
        other clip seconds (its centre held at this frame's anchor), with the seconds into the window then; None
        without eyes."""
        if ins["ring"]["face"] is None:
            return None
        fps, f0 = self.plan["fps"], self.plan["frame0"]
        size, output = self.size, self.output

        def circle_at(ts):
            out = []
            for s in np.atleast_1d(ts):
                f = f0 + float(s) * fps
                c = IR.circle(ins, IR.state(ins, f), output, size, anchor, per_metre)
                out.append((c["centre"], c["radius"], (f - ins["f0"]) / fps))
            return out
        return {"t": (frame - f0) / fps, "hits": self._eye_hits(ins), "circle_at": circle_at}


def _plate_sprites(ins):
    """Whether an insert is an iris whose plate ring has sprites round it."""
    ring = ins.get("ring")
    return ring is not None and ring["kind"] == "plate" and ring["sprites"] is not None
