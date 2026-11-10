"""mosh on the GPU (mkr-post fx/mosh.rs). Along the cut's own motion: the motion between its frames worked out here as
FrameFx._moshed works it (mkmmd.framefx.mosh_blocks: optical flow at a quarter size, averaged over blocks), each
frame's blocks written once to the scratch folder, and every step since the span's start handed over with the picture
it starts from; mkr-post keeps the held picture from frame to frame and takes only this frame's step when it holds the
frame before. With a made-up `flow`: its block moves (FrameFx._flow) written once, the picture carried along them
frame after frame from the span's first frame as it reaches the mosh. A frame the cut has not rendered, or a flowing
mosh whose frame before went another way, is drawn on the CPU."""
import os
from pathlib import Path

import numpy as np

from .. import framefx as FF
from .. import post as P
from ..core import framefx as FX

_MAPS = {}                  # (frame size, b, flow size, blocks) -> (cols, rows)


def _state(ffx):
    """This process's notes on the moshes of `ffx`: "blocks" {(entry index, frame g or "flow"): what was written for
    it}, "flowed" {flowing entry index: the last frame handed over to mkr-post}."""
    got = getattr(ffx, "_gpu_mosh", None)
    if got is None:
        got = ffx._gpu_mosh = {"blocks": {}, "flowed": {}}
    return got


def _maps(size, b, small, gw, gh):
    """(cols, rows): the block each column and row of a frame of `size` reads, as mkmmd.framefx.mosh_field's nearest
    resizes read them."""
    key = (size, b, small, gw, gh)
    if key not in _MAPS:
        idx = np.ascontiguousarray(np.stack(np.meshgrid(np.arange(gw, dtype=np.float32),
                                                        np.arange(gh, dtype=np.float32)), -1))
        field = FF.mosh_field(idx, b, small, size) * np.float32(FF.MOSH_SCALE)
        _MAPS[key] = (field[0, :, 0].astype(np.int64).tolist(), field[:, 0, 1].astype(np.int64).tolist())
    return _MAPS[key]


def _step(ffx, fx, g, paths):
    """(file, b, flow size, blocks shape) of the cut's motion from frame g - 1 to g, worked out once in this process."""
    blocks_of, key = _state(ffx)["blocks"], (fx["index"], g)
    if key not in blocks_of:
        size = ffx.size
        blocks, b, small = FF.mosh_blocks(P.read(paths.frame(g - 1), size), P.read(paths.frame(g), size),
                                          fx["block"] * ffx.px)
        path = Path(paths.scratch) / f"{os.getpid()}-mosh{fx['index']}-{g}.f32"
        blocks.tofile(path)
        blocks_of[key] = (str(path), b, small, blocks.shape[:2])
    return blocks_of[key]


def ops(ffx, fx, frame):
    paths = ffx.paths
    if paths is None or paths.scratch is None:
        return None
    op = _flowing(ffx, fx, frame) if fx["flow"] is not None else _cut_motion(ffx, fx, frame)
    if op is None:
        return None
    op.update(op="mosh", id=fx["index"], frame=frame, s=float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0)))
    if fx["mask"] is not None:
        fig = paths.figure(FX.figure_key(fx), frame)
        if not paths.fits(fig):
            return None
        op["mask"] = {"figure": str(fig), "inside": fx["mask"] == "figure"}
    if fx["flow"] is not None:
        _state(ffx)["flowed"][fx["index"]] = frame
    return [op]


def dropped(ffx, frame):
    """The ops worked out for `frame` were not sent (mkmmd.postframes.Output.plan): a flowing mosh handed over there
    goes on from there no more (the next frame is drawn on the CPU)."""
    flowed = _state(ffx)["flowed"]
    for i in [i for i, g in flowed.items() if g == frame]:
        del flowed[i]


def _flowing(ffx, fx, frame):
    """A mosh with a made-up flow: on the span's first frame the picture as it reaches it, carried the flow's `steps`
    along its block moves (FrameFx._flow, mosh_carry), and as many more on each frame after, from what mkr-post held of
    the frame before as this process handed it over (a chunk of frames only starts on the span's first: FrameFx.safe);
    None (the CPU) when that frame went another way."""
    f0 = fx["f0"]
    st = _state(ffx)
    if frame != f0 and st["flowed"].get(fx["index"]) != frame - 1:
        return None
    moves, b = ffx._flow(fx)
    key = (fx["index"], "flow")
    if key not in st["blocks"]:
        path = Path(ffx.paths.scratch) / f"{os.getpid()}-moshflow{fx['index']}.f32"
        moves.astype(np.float32).tofile(path)
        w, h = ffx.size
        st["blocks"][key] = (str(path), FF._block_of(w, b).tolist(), FF._block_of(h, b).tolist())
    path, cols, rows = st["blocks"][key]
    gh, gw = moves.shape[:2]
    return {"flow": {"moves": path, "fresh": frame == f0, "carries": int(fx["flow"]["steps"])}, "steps": [],
            "gw": gw, "gh": gh, "cols": cols, "rows": rows}


def _cut_motion(ffx, fx, frame):
    """A mosh along the cut's own motion: the picture it starts from and every step since the span's start. None (the
    CPU) when that picture, the frame before the span as it reached the mosh at its own time, is no file of the cut:
    an effect before the mosh in the table acts on it."""
    paths = ffx.paths

    def cut(g):                             # the cut's frame g, as FrameFx._get reads it
        return paths.frame(g) if g >= ffx.frame0 else None
    if cut(frame) is None:                  # its motion would run to the picture as it reaches the mosh
        return None
    f0 = fx["f0"]
    start = cut(f0 - 1) or cut(f0)
    if any(e["index"] < fx["index"] for e in ffx.entries(f0 - 1 if cut(f0 - 1) is not None else f0)):
        return None
    held = start is not None or frame == f0
    for g in (f0 - 1, f0, frame):
        if cut(g) is not None and not paths.fits(cut(g)):
            return None
    steps, shape = [], None
    for g in range(f0 + 1, frame + 1):
        if held and cut(g - 1) is not None and cut(g) is not None:
            if not paths.fits(cut(g - 1)):
                return None
            path, b, small, shape = _step(ffx, fx, g, paths)
            steps.append([g, path])
    w, h = ffx.size
    if shape is None:                       # no step: the blocks' layout is not needed, any will do
        b, small, shape = 1, (w, h), (1, 1)
    gh, gw = shape
    cols, rows = _maps((w, h), b, small, gw, gh)
    return {"start": None if start is None else str(start), "steps": steps, "gw": gw, "gh": gh, "cols": cols,
            "rows": rows}
