"""footage on the GPU (mkr-post fx/footage.rs): the clip's frame decoded and fitted as the CPU path does it
(mkmmd.footage: OpenCV's decode, its resize by area or bicubic) and handed over as rgb24 in the scratch folder; the figure
laid over it as a sticker (mkmmd.framefx.FrameFx._footage, `_paste`) by mkr-post; the frame, the fit, the back, the fill
and the figure's map worked out here as the CPU path works them out. `ops` names the clip's frame and `hand` decodes it,
in the process that hands the frames to mkr-post (a worker that plans a run of frames would seek the clip afresh for
each); `Kept` holds what the first output to show a clip frame fitted for the run's later outputs, so a run decodes
each clip frame once, not once an output. A pattern round a contained clip and a silkscreen fill stay on the CPU."""
import itertools
import os
import queue
import shutil
import threading
from pathlib import Path

import numpy as np

from .. import footage as FT
from .. import framefx as FF
from ..core import framefx as FX
from . import figure as GF

_HANDED = itertools.count()                             # this process's clip frames written for mkr-post
KEEP_AHEAD = 4                  # decoded frames waiting to be fitted for the later outputs (a 4K frame is 25 MB)
KEEP_SHARE = 0.25               # of the free space on the kept frames' disk, what they may take
SPAN = ("clip", "in", "rate", "loop", "f0", "f1")       # an entry shows the same clip frames on the same frames


def ops(ffx, fx, frame):
    if ffx._footage_of is None or ffx.paths is None or ffx.paths.scratch is None:
        return None
    s = float(np.clip(FX.strength(fx, ffx._t(frame)), 0.0, 1.0))
    if s <= 0.0:                                         # mixed in by nothing: the picture as it is
        return []
    k = FX.step(fx, frame, ffx._hit_times(fx), ffx.fps, ffx.frame0)
    back, fill = FX.pick(fx["back"], k), FX.pick(fx["fill"], k)
    if fill == FX.SILK or (fx["fit"] == "contain" and isinstance(back, dict)):
        return None
    figure = GF.cutout(ffx, fx, frame, picture=fill is None)
    if figure is None:
        return None
    clip = FF.clip_of(fx, ffx._footage_of)
    i = clip.index((frame - fx["f0"]) / ffx.fps, fx["in"], fx["rate"], fx["loop"])
    _, _, at, _ = FT.fit_box(clip.size, ffx.size, fx["fit"], fx["focus"], fx["mirror"])
    return [{"op": "footage", "clip": {"name": fx["clip"], "index": int(i), "fit": fx["fit"],
                                       "focus": [float(v) for v in fx["focus"]], "mirror": bool(fx["mirror"]),
                                       "entry": fx["index"]},
             "at": [int(at[0]), int(at[1])], "back": GF.colour(back) if fx["fit"] == "contain" else None,
             "figure": figure, "fill": GF.colour(fill), "edge": GF.edge(ffx, fx), "shadow": GF.shadow(ffx, fx),
             "strength": s}]


def _key(name, index, size, fit, focus, mirror):
    """What a fitted clip frame is known by: the clip, its frame, the picture's size and the fit."""
    return name, int(index), int(size[0]), int(size[1]), fit, float(focus[0]), float(focus[1]), bool(mirror)


def hand(ffx, op, folder, footage_of=None, kept=None):
    """The clip's frame `op` names decoded and fitted (FT.fitted) into an rgb24 file in `folder`, `op`'s clip now that
    file and its size, as mkr-post reads it; the clip from `footage_of` (mkmmd.footage.source; default the FrameFx's).
    With `kept` (Kept) a frame an output before this one fitted for it is taken from there instead (a file mkr-post
    leaves where it is), and a frame decoded here is offered to it for the outputs after this one."""
    c = op["clip"]
    got = None if kept is None else kept.get(_key(c["name"], c["index"], ffx.size, c["fit"], c["focus"], c["mirror"]))
    if got is not None:
        op["clip"] = {"path": got[0], "size": got[1], "keep": True}
        return
    clip = (footage_of or ffx._footage_of)(c["name"])
    rgb = clip.frame(c["index"])
    small, _ = FT.fitted(rgb, ffx.size, c["fit"], c["focus"], c["mirror"])
    path = Path(folder) / f"{os.getpid()}-clip{next(_HANDED)}.rgb"
    np.ascontiguousarray(small, np.uint8).tofile(path)
    op["clip"] = {"path": str(path), "size": [int(small.shape[1]), int(small.shape[0])]}
    if kept is not None:
        kept.offer(c, rgb)


class Kept:
    """The clip frames a run's later outputs show, fitted for them by a thread of its own as the first output to show
    each decodes it (`hand`), so a run decodes a clip frame once, not once an output: rgb24 files in `folder` (on disk:
    some 10 MB a frame for two later outputs, KEEP_SHARE of the disk's free space at most), removed by `close`.
    `outputs` lists, in the order the run finishes them, (name, size, {entry index: footage entry as that output draws
    it}), `size` a callable giving the output's (w, h) once its frames say (None before: a followed render that has not
    drawn it yet). A later output's entry takes the frames of the same entry before it where it shows the same clip on
    the same frames (SPAN); a frame not kept (another span, no room left) is decoded by the output that shows it, the
    same bytes either way (each fitted as the output's own sender would fit it, in this process)."""

    def __init__(self, folder, outputs):
        self.folder = Path(folder)
        self.outputs = list(outputs)
        self.sizes = {}
        self.now = None                                   # the output finished now (its index in `outputs`)
        self.files = {}                                   # _key -> (path, [w, h])
        self.room = KEEP_SHARE * shutil.disk_usage(self.folder).free
        self.used = 0
        self.full = False                                 # the disk said no: nothing more is kept
        self.error = None                                 # what stopped the thread otherwise (raised by offer)
        self.lock = threading.Lock()
        self.todo = queue.Queue(KEEP_AHEAD)
        self.names = itertools.count()
        self.thread = threading.Thread(target=self._run, name="mkpost-keep", daemon=True)
        self.thread.start()

    def begin(self, name):
        """Output `name` is finished from now on: what was offered for it is written first."""
        self.todo.join()
        self.now = next(i for i, o in enumerate(self.outputs) if o[0] == name)

    def get(self, key):
        """(path, [w, h]) of the fitted frame known by `key` (_key), or None."""
        with self.lock:
            return self.files.get(key)

    def _size(self, i):
        if i not in self.sizes:
            got = self.outputs[i][1]()
            if got is None:
                return None
            self.sizes[i] = (int(got[0]), int(got[1]))
        return self.sizes[i]

    def offer(self, clip, rgb):
        """A frame decoded for the output finished now (`rgb`, the clip frame a footage op of it names: `clip` with its
        name, index and entry), to be fitted for the outputs after it that show it."""
        if self.error is not None:
            raise self.error
        if self.full or self.now is None:
            return
        mine = self.outputs[self.now][2].get(clip.get("entry"))
        if mine is None:
            return
        keys = []
        for i in range(self.now + 1, len(self.outputs)):
            theirs = self.outputs[i][2].get(clip["entry"])
            size = self._size(i) if theirs is not None and all(theirs[k] == mine[k] for k in SPAN) else None
            if size is None:
                continue
            key = _key(clip["name"], clip["index"], size, theirs["fit"], theirs["focus"], theirs["mirror"])
            with self.lock:
                if key in self.files or self.used + size[0] * size[1] * 3 > self.room:
                    continue
                self.files[key] = None                    # on its way (an output's sender asks once it is written)
                self.used += size[0] * size[1] * 3
            keys.append(key)
        if keys:
            self.todo.put((rgb, keys))

    def _run(self):
        while True:
            item = self.todo.get()
            try:
                if item is None:
                    return
                rgb, keys = item
                for key in keys:
                    if self.full or self.error is not None:
                        break
                    _, _, w, h, fit, fx, fy, mirror = key
                    small, _ = FT.fitted(rgb, (w, h), fit, (fx, fy), mirror)
                    path = self.folder / f"{next(self.names)}.rgb"
                    try:
                        np.ascontiguousarray(small, np.uint8).tofile(path)
                    except OSError:                       # no room after all: the later outputs decode their own
                        path.unlink(missing_ok=True)
                        self.full = True
                        break
                    with self.lock:
                        self.files[key] = (str(path), [int(small.shape[1]), int(small.shape[0])])
            except Exception as e:                        # noqa: BLE001 (raised by offer, on the sender's thread)
                self.error = e
            finally:
                self.todo.task_done()

    def close(self):
        """Stop the thread and remove the kept frames."""
        self.todo.put(None)
        self.thread.join()
        shutil.rmtree(self.folder, ignore_errors=True)
