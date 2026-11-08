"""Video clips as pictures (docs/design.md: Frame effects, Footage): a clip named by a file or an asset-registry slug, its frames
decoded with OpenCV (FFmpeg) to display-space RGB, the last few kept, and fitted to a picture of any size: `cover` (filled,
cropped about a focus point) or `contain` (whole, on a colour), mirrored or not. The footage frame effect lays a figure
over them (mkmmd.framefx); anything else that shows a clip (a split screen's video half) reads it through this module.

    source(root)                 callable(clip) -> Clip, each clip opened once (what FrameFx takes as `footage_of`)
    resolve(clip, root)          the clip's path: a file (project-relative, absolute or ~/...) or a registry slug
    Clip(path)                   .path .fps .count .size (w, h) .duration
      .index(t, start, rate, loop)    the clip frame shown t s into a span that starts `start` s into the clip
      .frame(i)                       frame i decoded, RGB uint8 (h, w, 3)
      .picture(t, size, start, rate, loop, fit, focus, mirror, back)   float32 RGB 0..1 (h, w, 3) at size (w, h)
    fit(rgb, size, fit, focus, mirror, back)   one decoded frame fitted to a size
    fitted(rgb, size, fit, focus, mirror)      the part of it that shows, resized (uint8), and where it lands
    fit_box(src, size, fit, focus, mirror)     where: the part that shows, the size it takes, where it lands, how

A FootageError (a FileNotFoundError) says what is wrong: a missing or unreadable clip, a span that starts past its end."""
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".mts", ".m2ts", ".ogv")
FITS = ("cover", "contain")
FRAMES = 3                      # decoded frames kept per clip (a 4K frame is 25 MB)
PICTURES = 2                    # fitted pictures kept per clip (one shown over several frames of a slower clip)
SKIP = 120                      # frames read on rather than seeking (a seek decodes on from the key frame before)
OPEN = 2                        # clips a source always keeps open, the last asked (a 4K decoder and its frames: some 300 MB)
OPEN_BYTES = 600 << 20          # and as many more as fit this (a clip counts w * h * 36 bytes: 4K 300 MB, 1080p 75 MB):
                                # a project that shows four clips frame after frame must not reopen two every frame
DECODE_THREADS = 4              # FFmpeg's threads per clip: its default, one a core, spends twice the CPU on a frame
                                # (the frames are the same bytes whatever the count)


def _capture(path):
    """OpenCV's FFmpeg reader of the clip at `path`, decoding on DECODE_THREADS threads."""
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, DECODE_THREADS])


class FootageError(FileNotFoundError):
    """A clip that is missing or unreadable, or a span that starts past its end without `loop`."""


def is_file(clip):
    """Whether `clip` names a file (a path separator, `~` or a video extension) rather than an asset-registry slug."""
    s = str(clip)
    return any(c in s for c in "/\\~") or Path(s).suffix.lower() in VIDEO_EXT


def resolve(clip, root=None):
    """The path of `clip`: a file (project-relative to `root`, absolute or ~/...) or the slug of an asset-registry entry
    whose path is a video (`mk ref footage` registers Pexels clips so). A FootageError when there is no such file."""
    s = str(clip).strip()
    if not s:
        raise FootageError("clip: empty (a video file or an asset slug)")
    if is_file(s):
        p = Path(s).expanduser()
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
        if not p.is_file():
            raise FootageError(f"clip {s!r}: {p} does not exist")
        return p
    from .assets import AssetError, Registry          # the registry only for a slug (a file needs none)
    reg = Registry()
    try:
        e = reg.get(s)
    except AssetError:
        videos = sorted(x["slug"] for x in reg.entries if Path(x.get("path") or "").suffix.lower() in VIDEO_EXT)
        raise FootageError(f"clip {s!r}: not a video file and not an asset in {reg.path} (the videos there: "
                           f"{', '.join(videos) or 'none'}; `mk ref footage ID` downloads and registers a Pexels clip)"
                           ) from None
    p = Path(e.get("path") or "").expanduser()
    if not p.is_absolute():
        p = reg.root / p
    if p.suffix.lower() not in VIDEO_EXT:
        raise FootageError(f"clip {s!r}: the asset's path {p} is not a video ({' '.join(VIDEO_EXT)})")
    if not p.is_file():
        raise FootageError(f"clip {s!r}: the asset's file {p} does not exist")
    return p


def source(root=None):
    """callable(clip) -> Clip for a project at `root`: each clip resolved (`resolve`) and made once, the OPEN asked last
    kept open and, behind them, the ones asked before while they fit OPEN_BYTES together (the others let go of their
    decoders and frames until they are asked again; reopening a clip seeks, decoding on from a key frame, some hundred
    milliseconds of a frame)."""
    opened = OrderedDict()

    def get(clip):
        if clip not in opened:
            opened[clip] = Clip(resolve(clip, root))
        opened.move_to_end(clip)
        held = 0
        for n, c in enumerate(reversed(list(opened.values()))):
            held += c.size[0] * c.size[1] * 36
            if n >= OPEN and held > OPEN_BYTES:
                c.close()
        return opened[clip]
    return get


def fit_box(src, size, fit="cover", focus=(0.5, 0.5), mirror=False):
    """How a frame of `src` (w, h) px meets a picture of `size` (w, h) (`fit()`): ((x, y, w, h) the part of the frame that
    shows, in the frame as shown (mirrored first with `mirror`), the size (w, h) it is resized to, where in the picture
    its top left lands (x, y), how it is resized: "area" (shrunk) or "cubic"; one of its own size is left as it is)."""
    if fit not in FITS:
        raise ValueError(f"fit {fit!r}: one of {', '.join(FITS)}")
    W, H = int(size[0]), int(size[1])
    w, h = int(src[0]), int(src[1])
    fx, fy = float(focus[0]), float(focus[1])
    if mirror:
        fx = 1.0 - fx
    if fit == "cover":                          # the part that shows, in the frame's own pixels, then resized
        s = max(W / w, H / h)
        cw, ch = min(max(int(round(W / s)), 1), w), min(max(int(round(H / s)), 1), h)
        x0 = int(round(min(max(fx * w - cw / 2.0, 0.0), w - cw)))
        y0 = int(round(min(max(fy * h - ch / 2.0, 0.0), h - ch)))
        return (x0, y0, cw, ch), (W, H), (0, 0), "area" if s < 1.0 else "cubic"
    s = min(W / w, H / h)
    sw, sh = min(max(int(round(w * s)), 1), W), min(max(int(round(h * s)), 1), H)
    return (0, 0, w, h), (sw, sh), ((W - sw) // 2, (H - sh) // 2), "area" if s < 1.0 else "cubic"


def fitted(rgb, size, fit="cover", focus=(0.5, 0.5), mirror=False):
    """A decoded frame (uint8 RGB (h, w, 3)) resized as `fit()` lays it in a picture of `size` (w, h): (the part that
    shows, uint8 RGB (sh, sw, 3), resized by OpenCV; where its top left lands in the picture (x, y))."""
    (x0, y0, cw, ch), (sw, sh), at, how = fit_box((rgb.shape[1], rgb.shape[0]), size, fit, focus, mirror)
    if mirror:
        rgb = rgb[:, ::-1]
    part = np.ascontiguousarray(rgb[y0:y0 + ch, x0:x0 + cw])
    return cv2.resize(part, (sw, sh), interpolation=cv2.INTER_AREA if how == "area" else cv2.INTER_CUBIC), at


def fit(rgb, size, fit="cover", focus=(0.5, 0.5), mirror=False, back=(0.0, 0.0, 0.0)):
    """A decoded frame (uint8 RGB (h, w, 3)) fitted to `size` (w, h) as float32 RGB 0..1: `cover` fills it, scaled to the
    larger of the two ratios and cropped with the video's point `focus` ([x, y], 0..1 of the video as the file has it) as
    near the middle as the crop allows; `contain` shows it whole, scaled to the smaller ratio, centred on `back`.
    `mirror` flips it left to right (the focus point flips with it). `fit_box` works out where, `fitted` resizes."""
    small, (px, py) = fitted(rgb, size, fit, focus, mirror)
    if fit == "cover":
        return small.astype(np.float32) / np.float32(255.0)
    W, H = int(size[0]), int(size[1])
    sh, sw = small.shape[:2]
    out = np.empty((H, W, 3), np.float32)
    out[:] = np.asarray(back, np.float32)
    out[py:py + sh, px:px + sw] = small.astype(np.float32) / np.float32(255.0)
    return out


class Clip:
    """One video file read frame by frame: `fps`, `count` (frames), `size` (w, h) and `duration` (s) from its container;
    a frame decoded once is kept until a few others have been. `close` lets go of the decoder and the frames until the
    next frame is asked."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FootageError(f"{self.path} does not exist")
        cap = _capture(self.path)
        if not cap.isOpened():
            raise FootageError(f"{self.path}: not a video OpenCV can read")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.fps <= 0 or self.count <= 0 or min(self.size) <= 0:
            cap.release()
            raise FootageError(f"{self.path}: no frame rate, frame count or size in it ({self.fps:g} fps, {self.count} "
                               f"frames, {self.size[0]}x{self.size[1]})")
        self._cap, self._next = cap, 0                  # the frame the next read() returns
        self._frames, self._pictures = OrderedDict(), OrderedDict()

    @property
    def duration(self):
        return self.count / self.fps

    def close(self):
        """Let go of the decoder and of the decoded frames and pictures (a clip not shown for a while)."""
        if self._cap is not None:
            self._cap.release()
        self._cap, self._next = None, 0
        self._frames.clear()
        self._pictures.clear()

    def index(self, t, start=0.0, rate=1.0, loop=False):
        """The clip's frame (0 .. count - 1) on screen `t` seconds into a span that starts `start` seconds into the clip
        and plays at `rate` times its speed (0: a still): past its end the last frame holds or, with `loop`, the clip
        starts over. A FootageError when the span starts past the end without `loop`."""
        if not loop and int(math.floor(start * self.fps + 1e-6)) >= self.count:
            raise FootageError(f"{self.path.name}: in = {start:g} s is past its end ({self.duration:.2f} s, {self.count} "
                               f"frames at {self.fps:g} fps): an earlier `in`, or `loop = true`")
        i = max(int(math.floor((start + rate * t) * self.fps + 1e-6)), 0)
        return i % self.count if loop else min(i, self.count - 1)

    def frame(self, i):
        """Frame `i` decoded, RGB uint8 (h, w, 3). Reading on from the frame read last is fast; anything else seeks."""
        i = int(i)
        if not 0 <= i < self.count:
            raise IndexError(f"{self.path.name}: frame {i} is outside 0..{self.count - 1}")
        got = self._frames.get(i)
        if got is not None:
            self._frames.move_to_end(i)
            return got
        if self._cap is None:                           # closed: opened again, at its first frame
            self._cap = _capture(self.path)
            if not self._cap.isOpened():
                raise FootageError(f"{self.path}: not a video OpenCV can read")
        if not self._next <= i <= self._next + SKIP:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self._next = i
        while self._next < i:
            if not self._cap.grab():
                raise FootageError(f"{self.path.name}: frame {self._next} of {self.count} could not be read")
            self._next += 1
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise FootageError(f"{self.path.name}: frame {i} of {self.count} could not be decoded")
        self._next = i + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._frames[i] = rgb
        while len(self._frames) > FRAMES:
            self._frames.popitem(last=False)
        return rgb

    def picture(self, t, size, start=0.0, rate=1.0, loop=False, fit="cover", focus=(0.5, 0.5), mirror=False,
                back=(0.0, 0.0, 0.0)):
        """The clip's picture `t` seconds into a span (`index`) fitted to `size` (`fit()`): float32 RGB 0..1 (h, w, 3), the
        same array again while the same frame is asked at the same size (do not write into it)."""
        i = self.index(t, start, rate, loop)
        key = (i, int(size[0]), int(size[1]), fit, float(focus[0]), float(focus[1]), bool(mirror),
               tuple(float(c) for c in back))
        got = self._pictures.get(key)
        if got is None:
            got = self._pictures[key] = _fit(self.frame(i), size, fit, focus, mirror, back)
            while len(self._pictures) > PICTURES:
                self._pictures.popitem(last=False)
        else:
            self._pictures.move_to_end(key)
        return got


_fit = fit                      # Clip.picture's `fit` argument hides the function inside it
