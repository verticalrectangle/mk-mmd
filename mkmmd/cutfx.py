"""Cut effects composited over the cut's frames, before the grade (docs/design.md: Shots: Transitions and inserts,
Glitches): the post layer (numpy, OpenCV, Pillow). `mk post` and `mk look` hand each frame of the cut to `CutFx.frame`;
frames outside every window come back untouched. The layers an effect needs were rendered next to the cut's frames
(`mk render`, mkmmd.blender.transition); the plan (mkmmd.core.transition) says which.

    expand / collapse   the figure of a silhouette shot (`matte/`, its coverage as a distance field, mkmmd.matte) is turned
                        and scaled about its centre; inside it plays the other shot, outside is the silhouette shot's frame
                        without its figure (`back/`); a rim of `edge` colour runs round it. The scale rises until the figure
                        fills the frame at the last frame of an expand / the first of a collapse, whatever its size.
    slash               a band sweeps across; behind it the incoming shot, ahead of it the outgoing one, the switch under it.
    insert              a thought bubble over the host frame holds the other shot at reduced size (picture in picture),
                        popping in and out, or growing until its picture is the whole frame.
    glitch              the figure (the frame against its `bare/` layer, the same frame without the glitch's objects)
                        breaks up in jumping slices and colour ghosts and drops out until the bare frame is left
                        (mkmmd.core.glitch)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import matte as M
from . import post as P
from .core import glitch as GL
from .core import transition as TR
from .core import tween as TW

COVER_MARGIN = 1.02                      # the scale that fills the frame, with a little to spare
DOT_RADII = (0.12, 0.19, 0.28)           # the trailing circles' radii, as shares of the bubble's half-height
DOT_FIT = (0.085, 0.13, 0.18)            # and at most these shares of the trail's length (head edge to cloud rim)
DOT_AT = (0.12, 0.40, 0.75)              # where they sit along it, as shares measured from the head edge


class CutFx:
    """The effects of a plan over one output's frames. `frames_dir` holds the cut's frames and the layers, `size` is
    (width, height) of the frames being graded, `output` the aspect's name (for an insert's per-aspect geometry)."""

    def __init__(self, plan, frames_dir, size, output=None):
        self.plan, self.dir, self.size, self.output = plan, Path(frames_dir), (int(size[0]), int(size[1])), output
        self.px = min(self.size) / 1080.0                                  # widths are given at 1080 on the short side
        self.at = {}
        for tr in plan["transitions"]:
            for f in range(tr["first"], tr["last"] + 1):
                self.at[f] = ("transition", tr)
        for ins in plan["inserts"]:
            for f in range(ins["f0"], ins["f1"]):
                self.at[f] = ("insert", ins)
        for g in plan["glitches"]:
            for f in range(g["f0"], g["f1"]):
                self.at[f] = ("glitch", g)
        self._top, self._grown = {}, {}
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
        """Settle what depends on the pictures before anything is encoded: the scale at which each expand / collapse that
        touches `frames` fills the frame (a TransitionError when one cannot)."""
        want = set(frames)
        for tr in self.plan["transitions"]:
            if tr["kind"] != "slash" and any(f in want for f in range(tr["first"], tr["last"] + 1)):
                self._top_scale(tr)

    def frame(self, frame, main):
        """The cut's frame `main` (float RGB) with the effect of `frame`'s window composited on it."""
        hit = self.at.get(frame)
        if hit is None:
            return main
        kind, spec = hit
        if kind == "insert":
            return self._insert(spec, frame, main)
        if kind == "glitch":
            return GL.frame(main, self._rgb(TR.bare_rel(f"g{spec['index']}", frame)), frame - spec["f0"],
                            spec["f1"] - spec["f0"], seed=spec["seed"], shift=spec["shift"], split=spec["split"] * self.px,
                            colors=spec["colors"])
        if spec["kind"] == "slash":
            return self._slash(spec, frame, main)
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
        """(index coordinates, frame heights per metre at its depth) of a projected point, point/<key>/<frame>.json."""
        rel = TR.point_rel(key, frame)
        try:
            data = json.loads((self.dir / rel).read_text(encoding="utf-8"))
            return M.frac_to_px(data["p"], self.size), float(data["m"])
        except (OSError, ValueError, KeyError):
            raise FileNotFoundError(f"{self.dir / rel}: layer missing, unreadable or from before its scale was stored "
                                    f"(mk render makes it)") from None

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
