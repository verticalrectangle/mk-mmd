"""The frame effect kinds mkr-post draws (docs/design.md: Post), one module per kind: `ops(ffx, fx, frame)`, the ops of
entry `fx` of a mkmmd.framefx.FrameFx `ffx` on Blender frame `frame` (a list of JSON-ready dicts, empty when it leaves
the picture as it is), or None when mkr-post cannot draw it there (mkmmd.postgpu then leaves it and everything before
it to the CPU). Each works its parameters out with the FrameFx methods the CPU path draws with, and its ops are the
`Op` variants of mkr/mkr-post/src/fx/<kind>.rs; a FileNotFoundError when an input is missing, as the CPU path raises.
The files an op names come from `ffx.paths` (Paths; None: nothing is drawn on the GPU).

A kind is one line in KINDS."""
from pathlib import Path

from PIL import Image

from ..core import transition as TR
from . import (camcorder, doodle, echo, film, flood, footage, grid, halftone, invert, kaleido, key, label, lens, liquid,
               mandala, melt, mosh, pixelsort, push, scope, shapes, shock, slab, slitscan, spiral, split, stack, tint,
               trails, tunnel, zoom)

KINDS = {"zoom": zoom, "footage": footage, "echo": echo, "slitscan": slitscan, "lens": lens, "shock": shock,
         "push": push, "slab": slab, "split": split, "melt": melt, "flood": flood, "spiral": spiral, "kaleido": kaleido,
         "liquid": liquid, "tunnel": tunnel, "mandala": mandala, "trails": trails, "scope": scope, "grid": grid,
         "stack": stack, "pixelsort": pixelsort, "doodle": doodle, "mosh": mosh, "film": film, "tint": tint,
         "halftone": halftone, "invert": invert, "label": label, "camcorder": camcorder, "shapes": shapes, "key": key}


class Paths:
    """Where the pictures a FrameFx reads are on disk, for mkr-post: the cut's frames `files` ({Blender frame: PNG}),
    and the layers mk render drew beside them in `src`, all of `size` (w, h); `plan` the cut's plan (the shot in the cut
    on each frame, for a collage's own `shot`); `scratch` the folder files written for mkr-post go in (sticker tiles;
    mkmmd.postframes.Output sets it, None: none)."""

    def __init__(self, files, src, size, plan=None, scratch=None):
        self.files, self.src, self.size, self.plan = dict(files), Path(src), (int(size[0]), int(size[1])), plan
        self.scratch = scratch
        self._fits = {}

    def frame(self, g):
        """The cut's frame `g` (as FrameFx's frame_of reads it), or None when it is not there."""
        return self.files.get(g)

    def figure(self, key, g):
        """The figure layer of `key` on frame `g` (as FrameFx's figure_of reads it); a FileNotFoundError when it is
        missing, as mkmmd.cutfx.read_figure raises."""
        p = self.src / TR.figure_rel(key, g)
        if not p.is_file():
            raise FileNotFoundError(f"{p}: layer missing or unreadable (mk render makes it)")
        return p

    def plate(self, shot, g):
        """What `shot` shows at frame `g` (as FrameFx's plate_of reads it: the cut's frame where it is the shot in the
        cut, else its plate beside the frames); a FileNotFoundError when it is not there, as FrameFx._shot_frame
        raises."""
        p = (self.files.get(g) if TR.in_force(self.plan, shot, g) else self.src / TR.plate_rel(shot, g))
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f"shot {shot!r} at frame {g}: its plate is missing (mk render draws it)")
        return Path(p)

    def fits(self, path):
        """Whether mkr-post reads the PNG `path` as the CPU path does: an 8-bit PNG of `size` (the CPU resamples one of
        another size); checked once per folder (mk render draws a layer's frames alike)."""
        d = Path(path).parent
        if d not in self._fits:
            try:
                with Image.open(path) as im:
                    self._fits[d] = (im.format == "PNG" and im.size == self.size
                                     and im.mode in ("RGB", "RGBA", "L", "LA", "P"))
            except (OSError, ValueError):
                return False
        return self._fits[d]
