"""paintapp on the GPU (mkr-post fx/mandala.rs, the op the mandala's stickers are laid by): the stamps of a paint app that is
only stamps (`ui = false`, `cursor = false`, no `bucket`, `eraser` or `undo`: its chrome, cursor, fills, erasers and
undo stay on the CPU). Each stamp on the canvas on the frame is a sticker laid in the order it landed
(mkmmd.paintapp.PaintAppFx._pa_pose: its sprite tile, the affine map of its pop, turn and scroll, its pupils, its drop
shadow), at the opacity it fades to; its sparkles' coverages (PaintAppFx._pa_sparks, drawn by OpenCV as the CPU path
draws them) are laid over it as inks. The tiles are written once to files, as the mandala's are."""
import numpy as np

from .. import eyes as EYD
from .. import mandala as MDF
from .. import paintapp as PAF
from ..core import framefx as FX
from ..core import paintapp as PA
from . import figure as GF


def stamps_only(fx):
    """Whether entry `fx` is a paint app of stamps and nothing else (what mkr-post draws)."""
    return (fx["stamps"] is not None and not fx["ui"] and fx["cursor"] is None and fx["bucket"] is None
            and not fx["eraser"] and fx["undo"] is None)


def tile_of(ffx, fx, st, j, tile):
    """Stamp j's tile as mkr-post's Tile::File (written once to the scratch folder; kept for the FrameFx's run)."""
    sp = fx["stamps"]
    path = str(st["files"][PA.stamp_pick(fx, j, len(st["files"]))])
    face = sp["face"]
    key = ("stamp", path, round(sp["size"] * min(ffx.size), 2),
           round(0.0 if sp["edge"] is None else sp["edge"]["width"] * ffx.px, 2),
           None if face is None else (face["eyes"], face["mouth"], face["size"]))
    memo = ffx.__dict__.setdefault("_gpu_stamps", {})
    if key not in memo:
        memo[key] = GF.tile_files(ffx, tile, key)
    return key, memo[key]


def sparkle_inks(ffx, fx, sparks):
    """The sparkles `sparks` (PaintAppFx._pa_sparks) as coverages laid in order, as PaintAppFx._pa_poly lays each: its
    ink line, then its fill ([(coverage, colour, x0, y0, alpha)], GF.ink_list's argument)."""
    line = max(1.5 * ffx.px, 0.8)
    out = []
    for pts, colour, alpha in sparks:
        got = PAF.poly_cover((ffx.size[1], ffx.size[0]), pts, line)
        if got is None:
            continue
        x0, y0, outline, fill = got
        for cov, c in ((outline, fx["ink"]), (fill, colour)):
            out.append((cov.astype(np.float32) / np.float32(255.0), c, x0, y0, alpha))
    return out


def ops(ffx, fx, frame):
    if not stamps_only(fx) or ffx.paths is None or ffx.paths.scratch is None:
        return None
    t = ffx._t(frame)
    s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
    if s <= 0.0:
        return []
    st, sp = ffx._pa[fx["index"]], fx["stamps"]
    size = ffx.size
    sh = sp["shadow"]
    shadow = None if sh is None else {"color": GF.colour(sh["color"]),
                                      "offset": [float(sh["offset"][0] * size[1]), float(sh["offset"][1] * size[1])],
                                      "blur": float(sh["blur"] * ffx.px), "alpha": float(sh["alpha"])}
    tiles, index, stickers = [], {}, []
    for it in st["hist"]["items"]:
        state = PA.stamp_state(fx, it, t)
        if state is None or state[0] <= 1e-3 or state[2] <= 0.0:
            continue
        tile, M, pupils, shade = ffx._pa_pose(fx, st, it, t, state)
        sparks = ffx._pa_sparks(fx, it["j"], state[3], ffx._pa_at(fx, st, it, state[3]))
        got = MDF.lay_box(tile.a.shape, M, shade, size)
        inks = [] if got is None else EYD.pupil_inks((size[1], size[0]), tile.face, M, pupils)
        inks += sparkle_inks(ffx, fx, sparks)
        if got is None and not inks:
            continue
        sticker = {"tile": None, "opacity": float(state[2]),
                   "inks": GF.ink_list(inks, ffx.paths.scratch, f"paintapp-{fx['index']}-{frame}-{len(stickers)}")}
        if got is not None:
            key, made = tile_of(ffx, fx, st, it["j"], tile)
            if key not in index:
                index[key] = len(tiles)
                tiles.append(made)
            (x0, y0, x1, y1), Mt = got
            sticker.update(tile=index[key], box=[x0, y0, x1, y1], warp=GF.warp(Mt, tile.a.shape))
        stickers.append(sticker)
    if not stickers:
        return []
    return [{"op": "mandala", "back": "picture", "figure": None, "tiles": tiles, "stickers": stickers,
             "edge": None if sp["edge"] is None else GF.colour(sp["edge"]["color"]), "shadow": shadow, "strength": s}]
