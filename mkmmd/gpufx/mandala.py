"""mandala on the GPU (mkr-post fx/mandala.rs): the rings laid out on the frame (mkmmd.core.mandala.copies, as
MandalaFx._mandala lays them), each sticker's box and map (mkmmd.mandala.placed, lay_box), the back, the tiles of sprites
(their googly eyes' whites and smiles painted on) and drawn shapes (MandalaFx._md_tile, written once to files), and the
pupils and glints of each copy with eyes (MandalaFx._md_pupils, as mkmmd.eyes.pupil_inks covers them, laid after its
sticker) worked out here; the figure's own tiles (cut from the figure layer and the picture as it reaches the op) and
every sticker's pixels drawn by mkr-post. A mandala whose figure is moved by keys, `at` or `height`, or a silkscreen
fill stays on the CPU."""
import math
from collections import namedtuple

import numpy as np

from .. import eyes as EYD
from .. import framefx as FF
from .. import mandala as MDF
from ..core import fields as CF
from ..core import framefx as FX
from ..core import mandala as MD
from . import fields as GFL
from . import figure as GF

Anchored = namedtuple("Anchored", "anchor")     # what mkmmd.mandala.placed reads of a tile


def back(ffx, fx, frame, k):
    """A collage's back entry `k` on `frame` as mkr-post's Back: {"color": rgb}, "picture" or {"pattern": PatternSpec}
    (FieldFx._back, the pattern painted over the whole frame)."""
    b = FX.pick(fx["back"], k)
    if b is None:
        return "picture"
    if not isinstance(b, dict):
        return {"color": GF.colour(b)}
    q = CF.phase(b, ffx._t(frame), fx["from"], ffx._on(fx, first=False))
    return {"pattern": GFL.painter((b, q), *ffx.size)}


def figure_tile(m, s, fill, ew, centre=None):
    """(mkr-post's Tile::Figure, the tile's shape (h, w), its anchor) of MDF.figure_tile(pic, m, s, fill, ew, centre);
    None when nothing is covered."""
    geo = MDF.figure_geometry(m, s, 0, centre)
    if geo is None:
        return None
    (X0, Y0, X1, Y1), (nw, nh), (sx, sy), c = geo
    pad = int(math.ceil(ew)) + 2
    anchor = ((c[0] - X0 + 0.5) * sx - 0.5 + pad, (c[1] - Y0 + 0.5) * sy - 0.5 + pad)
    tile = {"kind": "figure", "crop": [int(X0), int(Y0), int(X1), int(Y1)], "size": [int(nw), int(nh)],
            "shrink": bool(s < 1.0), "fill": GF.colour(fill), "pad": pad, "edge": float(ew)}
    return tile, (nh + 2 * pad, nw + 2 * pad), anchor


def ops(ffx, fx, frame):
    if ffx.paths is None or ffx.paths.scratch is None:
        return None
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    times = ffx._hit_times(fx)
    k = FX.step(fx, frame, times, ffx.fps, ffx.frame0)
    m, layer = None, None
    if fx["objects"] is not None:
        if ffx._pose_map(fx, frame) is not None:
            return None
        layer = ffx.paths.figure(FX.figure_key(fx), frame)
        if not ffx.paths.fits(layer):
            return None
        m = ffx._figure(fx, frame)
    centre = ffx._at(fx, frame)[0]
    unit = float(min(ffx.size))
    edge, shadow = ffx._edge(fx), ffx._shadow(fx)
    ew = 0.0 if edge is None else edge[1]
    schedule = ffx._md_schedule(fx)
    tiles, index, stickers = [], {}, []
    eyes = ffx._md_pupils(fx, t, times, schedule, centre, unit, edge)

    def tile(key, make):
        """(index, (tile, shape, anchor)) of the tile `key`, made once a frame; None for a figure covering nothing."""
        if key not in index:
            got = make()
            index[key] = None if got is None else len(tiles)
            if got is not None:
                tiles.append(got)
        i = index[key]
        return None if i is None else (i, tiles[i])

    def lay(i, shape, anchor, at, angle, grow, mirror, face=None, pupils=None):
        M = MDF.placed(Anchored(anchor), at, angle, grow, mirror)
        got = MDF.lay_box(shape, M, shadow, ffx.size)
        if got is not None:
            (x0, y0, x1, y1), Mt = got
            inks = EYD.pupil_inks((ffx.size[1], ffx.size[0]), face, M, pupils)
            stickers.append({"tile": i, "box": [x0, y0, x1, y1], "warp": GF.warp(Mt, shape),
                             "inks": GF.ink_list(inks, ffx.paths.scratch,
                                                 f"mandala-{fx['index']}-{frame}-{len(stickers)}")})
    for cp in MD.copies(fx, t, times, ffx.fps, ffx.frame0, schedule):
        r = fx["rings"][cp["ring"]]
        grow = cp["scale"] / r["scale"] * cp["pop"]
        if grow < MDF.MIN_SCALE:
            continue
        src, j = r["source"], cp["j"]
        face = pupils = None
        if src["kind"] == "figure":
            if m is None:
                continue
            fill = ffx._fill(fx, k + j, r["fill"])
            if isinstance(fill, dict):
                return None
            sc = r["scale"] * unit / ffx._md_height(fx)
            got = tile(("figure", r["scale"], repr(fill)), lambda: figure_tile(m, sc, fill, ew))
            if got is None:
                continue
        else:
            tl = ffx._md_tile(fx, cp, r, k, None, None, unit, edge, None)
            size = r["scale"] * unit
            if src["kind"] == "sprites":
                files = ffx._md_sprites[fx["index"], cp["ring"]]
                own = src.get("face")
                looks = None if own is None else (own["eyes"], own["mouth"], own["size"], own["look"] == "viewer")
                key = ("sprite", str(files[MD.sprite_pick(fx, cp["ring"], j, len(files))]), round(size, 2), ew, looks)
                face, pupils = tl.face, eyes.get((cp["ring"], j))
            else:
                main, accent, ink = MD.shape_colours(src, j, k)
                key = ("shape", MD.shape_of(src, j), tuple(main), tuple(accent), tuple(ink), round(size, 2), ew)
            got = tile(key, lambda: (GF.tile_files(ffx, tl, key), tl.a.shape, tl.anchor))
        i, (_, shape, anchor) = got
        lay(i, shape, anchor, (centre[0] + cp["x"] * unit, centre[1] + cp["y"] * unit), cp["angle"], grow,
            cp["mirror"], face, pupils)
    c = None if not fx["center_figure"] or m is None else FF.centroid(m)
    if c is not None:                                    # the figure itself, as the frame shows it, over the rings
        fill = ffx._fill(fx, k)
        if isinstance(fill, dict):
            return None
        got = tile(("centre", repr(fill)), lambda: figure_tile(m, 1.0, fill, ew, c))
        if got is not None:
            i, (_, shape, anchor) = got
            lay(i, shape, anchor, c, 0.0, 1.0, False)
    return [{"op": "mandala", "back": back(ffx, fx, frame, k), "figure": None if layer is None else str(layer),
             "tiles": [tl for tl, _, _ in tiles], "stickers": stickers,
             "edge": None if edge is None else GF.colour(edge[0]), "shadow": GF.shadow(ffx, fx), "strength": s}]
