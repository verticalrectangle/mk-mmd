"""mk post on the GPU (docs/design.md: Post): mkr-post (the Rust crate mkr/mkr-post) finishes a frame from where the
Python effects leave it: the frame effects it draws (mkmmd.gpufx.KINDS), the screen type, the keep mask, the grade and
the bytes the encoder takes, each kernel following the numpy / OpenCV code it replaces (within a level of it). This
module finds the binary, writes an output's setup, turns an effect entry into its ops and runs `mkr-post serve` for one
output.

    binary()                    the executable: $MK_MKR_POST, else <repo>/mkr/target/release/mkr-post, or None
    refused(post)               why mkr-post cannot finish any output of a project with this [post] table, or None
    unusable(out)               why mkr-post cannot finish the frames of `out` (a mkmmd.postframes.Output), or None
    ops(ffx, fx, frame)         the ops of entry `fx` of a mkmmd.framefx.FrameFx on `frame`, None when mkr-post cannot
                                draw it; a FileNotFoundError when its inputs are missing (the chain is left out)
    hand(ffx, ops, folder, footage_of, kept)   those ops made ready to send by the process that sends them (a clip's
                                frame decoded, or taken from what an output before fitted for this one)
    Server(out, folder, sink=None)   `mkr-post serve` for `out`: send(job), receive() -> (k, darkest luma, rgb24 bytes);
                                with a sink (the encoder's stdin) mkr-post writes the bytes there, receive() gives None
    grade(g)                    a mkmmd.post.Grade as mkr-post's Grade: the setup's, or a frame's own

A job (one JSON line) names the cut's frame (or a float RGB picture a worker wrote, `raw`), its screen layer, its negative
layer (`negative = true` screen type: the picture is inverted under its coverage after the screen layer is laid) and keep
mask, and the ops still to draw, or carries a frame this side finished (`finished`: its bytes in a file and its darkest
luma); with a breathing palette (`[look] breathe`) it carries its own `grade`, the setup's with that frame's colours.
mkr-post answers each with its clip index, the darkest luma and the frame's bytes, in the order the jobs came."""
import hashlib
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from . import gpufx as GFX
from . import post as P

THREADS = 4                 # threads of mkr-post decoding the frames' PNGs


def binary():
    """The mkr-post executable, or None when there is none."""
    env = os.environ.get("MK_MKR_POST")
    if env:
        return env if Path(env).is_file() else None
    p = Path(__file__).resolve().parent.parent / "mkr" / "target" / "release" / "mkr-post"
    return str(p) if p.is_file() else None


def _png_ok(path, size, layer=False):
    """Whether mkr-post reads the PNG at `path` as mk post does: 8 bits, the output's size (a layer a pixel larger)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            ok_mode = im.mode in ("RGB", "RGBA", "L", "LA", "P") and im.format == "PNG"
    except (OSError, ValueError):
        return False
    if layer:
        return ok_mode and 0 <= w - size[0] <= 1 and 0 <= h - size[1] <= 1
    return ok_mode and (w, h) == tuple(size)


def refused(post):
    """Why mkr-post cannot finish any output of a project whose `[post]` table is `post`: no binary, or a grade part it
    does not draw yet; None when it can."""
    if binary() is None:
        return "no mkr-post binary (cargo build --release in mkr/, or MK_MKR_POST)"
    if post.get("halation"):
        return "[post] halation is drawn on the CPU"
    if post.get("streaks"):
        return "[post] streaks are drawn on the CPU"
    return None


def unusable(out, files=True):
    """Why mkr-post cannot finish the frames of `out`: what `refused` says of its project, or (with `files`) a file it
    would read otherwise than mk post (another size or bit depth); None when it can. Following a render the files are
    not there yet: mkr-post itself refuses one of another size when it comes."""
    why = refused(out.grade.spec)
    if why is not None or not files:
        return why
    from .core import keep as KP
    from .core import screentype as SRT
    for f, p in out.files.items():
        if not _png_ok(p, out.size):
            return f"{p}: not an 8-bit PNG of {out.size[0]}x{out.size[1]}"
        for lp in (out.src / SRT.layer_rel(f), out.src / SRT.negative_rel(f)):
            if lp.exists() and not _png_ok(lp, out.size, layer=True):
                return f"{lp}: not an 8-bit PNG of the frame's size"
        kp = out.src / KP.rel(f)
        if out.keep and kp.exists() and not _png_ok(kp, out.size):
            return f"{kp}: not an 8-bit PNG of {out.size[0]}x{out.size[1]}"
    return None


def ops(ffx, fx, frame):
    """The ops of entry `fx` on `frame` (a list, maybe empty when it leaves the picture as it is), or None when mkr-post
    cannot draw it: a kind it lacks (mkmmd.gpufx.KINDS) or what its kind's module leaves to the CPU. Its parameters come
    from the same FrameFx methods the CPU path uses. An entry limited to a `rect` is bracketed by the generic `rect` ops
    (FrameFx._apply's `img + (out - img) * rect_mask`) unless its kind's module says RECT: it draws the rect itself."""
    kind = GFX.KINDS.get(fx["kind"])
    if kind is None:
        return None
    got = kind.ops(ffx, fx, frame)
    if got and fx.get("rect") is not None and not getattr(kind, "RECT", False):
        return [{"op": "rect", "phase": "begin"}, *got, rect_end(ffx, fx)]
    return got


def rect_end(ffx, fx):
    """The `rect` op that keeps the ops of entry `fx` inside its rect: the ramps of mkmmd.picturefx.rect_ramps over the
    whole frame (rows then columns, float32) in a file of the scratch folder named by what they are made of, written once
    and kept (several workers write the same bytes: the file appears whole)."""
    from . import picturefx as PFX
    rect, feather = tuple(float(v) for v in fx["rect"]), float(fx["feather"] * ffx.px)
    rows, cols = PFX.rect_ramps(ffx.size, rect, feather)
    data = np.concatenate([rows, cols]).astype(np.float32).tobytes()
    path = Path(ffx.paths.scratch) / f"rect-{hashlib.sha1(data).hexdigest()[:16]}.f32"
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return {"op": "rect", "phase": "end", "ramps": str(path), "count": len(data) // 4}


def dropped(ffx, frame):
    """The ops worked out for `frame` were not sent (a layer missing on the way): what a kind's module noted it had
    mkr-post keep from them (a film's print) is forgotten."""
    for kind in GFX.KINDS.values():
        forget = getattr(kind, "dropped", None)
        if forget is not None:
            forget(ffx, frame)


def hand(ffx, ops, folder=None, footage_of=None, kept=None):
    """`ops` (from ops, for a frame of `ffx`) made ready for mkr-post in place, by the process that sends them, the files
    they need written into `folder` (default `ffx.paths.scratch`): a footage op's (or a key's) clip frame decoded and
    fitted (mkmmd.gpufx.footage.hand: one decoder per clip, read in order) from `footage_of` (mkmmd.footage.source;
    default the FrameFx's own clips), or taken from `kept` (mkmmd.gpufx.footage.Kept: fitted for this output by the
    run's output before that decoded it); `ops`."""
    for op in ops:
        if op["op"] in ("footage", "key") and isinstance(op.get("clip"), dict) and "name" in op["clip"]:
            GFX.footage.hand(ffx, op, folder or ffx.paths.scratch, footage_of, kept)
    return ops


def first(fx, ops):
    """Whether entry `fx`'s `ops` (from ops) must be the first mkr-post draws on their frame, the entries before them
    drawn here (mkmmd.postframes.Output.plan): a kind whose module says FIRST (a pixel sort) is chaotic in its input,
    so its GPU draw matches the CPU's only on the very picture the CPU path would sort (mkmmd.gpufx.pixelsort)."""
    return bool(ops) and getattr(GFX.KINDS.get(fx["kind"]), "FIRST", False)


def grade(g):
    """Grade `g` (mkmmd.post.Grade) as mkr-post's Grade: the setup's, for every frame, or with a breathing palette a
    frame's own (its colours that frame's: mkmmd.postframes.Output.graded)."""
    s = g.spec
    sat = float(s.get("saturation", 1.06))
    out = {"saturation": sat, "keep_delta": g.keep_sat - sat, "pivot": float(s.get("pivot", 0.60)),
           "contrast": float(s.get("contrast", 1.10)), "split": None, "floor": None}
    if g.split:
        sh, hi, amt = g.split
        out["split"] = {"shadows": [float(v) for v in sh - 0.5], "highlights": [float(v) for v in hi - 0.5],
                        "amount": amt}
    if g.floor is not None:
        yf = float(g.floor @ P.Y_W)
        lim = yf + g.toe
        out["floor"] = {"rgb": [float(v) for v in g.floor], "yf": yf, "lim": lim, "lim_yf": lim - yf, "lo": yf * 0.98}
    return out


def setup(out, folder, plain=False, floats=False):
    """Write the setup of `out` into `folder`: setup.json, the vignette and the grain as numpy made them; its path. With
    `plain` mkr-post lays no screen type and grades nothing: the picture the ops leave, quantised, or with `floats` as
    float32 RGBA (kernel tests)."""
    g, (w, h) = out.grade, out.size
    folder = Path(folder)
    np.ascontiguousarray(g.vig[..., 0], np.float32).tofile(folder / "vignette.f32")
    np.ascontiguousarray(np.stack(g.grain), np.float32).tofile(folder / "grain.f32")
    data = {"size": [w, h], "screen": out.screen,
            "lut": [float(v) for v in np.arange(256).astype(np.float32) / 255.0],
            "vignette": str(folder / "vignette.f32"),
            "grain": {"file": str(folder / "grain.f32"), "count": len(g.grain), "amount": g.grain_amt},
            "grade": grade(g), "plain": bool(plain or floats), "floats": bool(floats)}
    path = folder / "setup.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class Server:
    """`mkr-post serve` for one output (`out`, a mkmmd.postframes.Output), its setup and scratch files in `folder`:
    send(job) hands it a frame, receive() the next frame it finished, in the order they were sent: (k, darkest luma,
    rgb24 bytes). With a `sink` (a binary file with a descriptor: the encoder's stdin) mkr-post writes the frames' bytes
    there itself and receive() gives (k, darkest luma, None). A RuntimeError carries its last words when it stops.
    `plain` and `floats` as setup's (a frame's data then w * h * 16 bytes)."""

    def __init__(self, out, folder, plain=False, floats=False, sink=None):
        self.size = out.size
        self.floats = floats
        self.folder = Path(folder)
        self.sink = sink
        self.log = open(self.folder / "mkr-post.log", "w+b")
        cmd = [binary(), "serve", "--setup", str(setup(out, folder, plain, floats)), "--threads", str(THREADS)]
        if sink is None:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
            self.records = self.proc.stdout
        else:                                            # the frames to the encoder, (k, luma) on a pipe of their own
            r, w = os.pipe()
            try:
                self.proc = subprocess.Popen([*cmd, "--stats", str(w)], stdin=subprocess.PIPE, stdout=sink,
                                             stderr=self.log, pass_fds=(w,))
            except BaseException:
                os.close(r)
                raise
            finally:
                os.close(w)
            self.records = os.fdopen(r, "rb")
        self.results = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        n = 0 if self.sink is not None else self.size[0] * self.size[1] * (16 if self.floats else 3)
        src = self.records
        while True:
            head = src.read(8)
            if len(head) < 8:
                self.results.put(None)
                return
            k, dark = struct.unpack("<If", head)
            data = src.read(n) if n else None
            if n and len(data) < n:
                self.results.put(None)
                return
            self.results.put((k, dark, data))

    def _failed(self):
        self.proc.poll()
        self.log.flush()
        self.log.seek(0)
        words = self.log.read().decode("utf-8", "replace").strip().splitlines()[-5:]
        return RuntimeError(f"mkr-post stopped (exit {self.proc.returncode}): {' / '.join(words) or 'no message'}")

    def send(self, job):
        try:
            self.proc.stdin.write((json.dumps(job) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._failed() from None

    def receive(self):
        got = self.results.get()
        if got is None:
            self.proc.wait()
            raise self._failed()
        return got

    def end(self):
        """No more jobs: mkr-post finishes the frames it has and stops."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def close(self):
        """Stop it (a clean end of input, or killed when it is still busy) and remove its scratch files."""
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.records.close()
        self.log.close()
        shutil.rmtree(self.folder, ignore_errors=True)
