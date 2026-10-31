"""mkr-post against the CPU path on synthetic pictures, for the tests of the frame effects mkr-post draws (mkmmd.gpufx):
a plain mkr-post (no screen type, no grade) draws an entry's ops on a picture, the CPU path (FrameFx._apply) the entry
itself, both quantised as mk post quantises. The frames and figure layers an effect reads are 8-bit PNGs in a folder,
laid out as mk render leaves them, read by both. Tests using it skip without the binary or a GPU.

    gpu = GPU(size, tmp_path)                  a plain mkr-post for pictures of `size` (w, h); close() at the end
    ffx = frame_fx(fxs, size, tmp_path, frames={g: picture}, figures={(key, g): coverage}, fps=30, frame0=1, ...)
    worst, over = gpu.check(ffx, fx, frame, img)   max level difference and pixels off by more than one
"""
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import cutfx as CF
from mkmmd import framefx as FF
from mkmmd import gpufx as GFX
from mkmmd import post as P
from mkmmd import postgpu as PG
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

_PROBE = {}


def usable():
    """Why mkr-post cannot run here (no binary, no adapter), or None."""
    if "why" not in _PROBE:
        exe = PG.binary()
        if exe is None:
            _PROBE["why"] = "no mkr-post binary (cargo build --release in mkr/)"
        else:
            got = subprocess.run([exe, "info"], capture_output=True, text=True)
            _PROBE["why"] = None if got.returncode == 0 else f"mkr-post has no GPU: {got.stderr.strip()[-200:]}"
    return _PROBE["why"]


def save(path, pic):
    """`pic` (h, w, 3 float or h, w coverage) as an 8-bit PNG, as mk render writes frames and figure layers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = P.to_u8(pic)
    Image.fromarray(np.dstack([a] * 3) if a.ndim == 2 else a, "RGB").save(path)
    return path


def frame_fx(fxs, size, folder, frames=None, figures=None, fps=30, frame0=1, **kw):
    """A FrameFx over the plan entries `fxs` whose frames ({Blender frame: picture}) and figure layers ({(key, frame):
    coverage}) are saved in `folder` and read back from there by both paths; what the ops write for mkr-post goes in
    `folder`/scratch."""
    folder = Path(folder)
    files = {g: save(folder / f"{g:05d}.png", pic) for g, pic in (frames or {}).items()}
    for (key, g), cov in (figures or {}).items():
        save(folder / TR.figure_rel(key, g), cov)
    scratch = folder / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return FF.FrameFx(fxs, size, fps=fps, frame0=frame0,
                      frame_of=lambda g: P.read(files[g], size) if g in files else None,
                      point_of=kw.pop("point_of", None),
                      figure_of=lambda key, g: CF.read_figure(folder, key, g, size),
                      paths=GFX.Paths(files, folder, size, scratch=scratch), **kw)


class GPU:
    """A plain mkr-post (Server(plain=True)) for pictures of `size`, its scratch files in `folder`."""

    def __init__(self, size, folder):
        why = usable()
        if why:
            pytest.skip(why)
        self.size = (int(size[0]), int(size[1]))
        self.folder = Path(folder) / "mkr-post"
        self.folder.mkdir(parents=True, exist_ok=True)
        out = types.SimpleNamespace(size=self.size, screen="graded",
                                    grade=P.Grade({}, PAL.get("rose-pine-moon"), self.size))
        self.server = PG.Server(out, self.folder, plain=True)
        self.n = 0

    def draw(self, img, ops):
        """`img` (h, w, 3 float32) with `ops` drawn on it: rgb24 bytes."""
        self.n += 1
        raw = self.folder / f"in{self.n}.f32"
        np.ascontiguousarray(img, np.float32).tofile(raw)
        self.server.send({"k": self.n, "raw": str(raw), "ops": json.loads(json.dumps(ops))})
        return self.server.receive()[2]

    def check(self, ffx, fx, frame, img):
        """(max level difference, pixels off by more than one) between the CPU path and mkr-post drawing entry `fx` of
        `ffx` on `frame` over `img`; the ops must be there (mkr-post draws it)."""
        ops = GFX.KINDS[fx["kind"]].ops(ffx, fx, frame)
        assert ops is not None, f"{fx['kind']}: left to the CPU"
        got = self.draw(img, PG.hand(ffx, ops))
        want = P.to_u8(ffx._apply(fx, frame, img)).tobytes()
        d = np.abs(np.frombuffer(got, np.uint8).astype(np.int16) - np.frombuffer(want, np.uint8).astype(np.int16))
        return int(d.max()), int((d > 1).sum())

    def close(self):
        self.server.close()
