"""Frame effects on the finished pictures (docs/design.md: Shots, Frame effects): the picture work of the `[[fx]]` entries
that mkmmd.core.framefx checks and times. `mk post` hands each frame of the cut to FrameFx.frame after the cut's own
effects (mkmmd.cutfx) and before the screen type and the grade; `mk look` does the same for the frames it draws. Images are
float RGB in display space (0..1), (h, w, 3)."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .core import framefx as FX
from .core import lanes as LN

GLOW_SIGMA = 6.0                 # px at 1080 on the short side: the scope trace's glow
LINE_SHIFT = 4                   # fixed-point bits of the trace's points (sub-pixel lines)
CACHE = 64                       # decoded frames kept for echoes and slitscans
SLAB_SAMPLES = 8                 # positions a moving slab is drawn at over its shutter (motion blur)
SHOCK_RIM = 0.6                  # the lit front of a shock ring, as a share of its width
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def scale_about(img, c, z):
    """The picture zoomed `z` times about pixel `c` (x, y; z > 1 is in), the frame's edge reflected where it runs out."""
    if abs(z - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = np.array([[1.0 / z, 0.0, c[0] * (1.0 - 1.0 / z)], [0.0, 1.0 / z, c[1] * (1.0 - 1.0 / z)]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def shock(img, c, waves, width, amount, colour, glow):
    """`img` bent by rings running out from pixel `c`: each wave (radius px, weight 0..1) pushes the picture out just
    ahead of its front and pulls it in just behind (a derivative of a Gaussian across `width` px, `amount` px at its
    peak), and `glow` lights its front in `colour`."""
    if not waves:
        return img
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = xs - np.float32(c[0]), ys - np.float32(c[1])
    r = np.sqrt(dx * dx + dy * dy) + np.float32(1e-3)
    disp, rim = np.zeros_like(r), np.zeros_like(r)
    for R, s in waves:
        u = (r - np.float32(R)) / np.float32(width)
        disp += np.float32(amount * s * 2.33) * u * np.exp(-u * u)       # 2.33: the curve's peak is 1 (at u = 0.707)
        rim += np.float32(s) * np.exp(-(u / SHOCK_RIM) ** 2)
    out = cv2.remap(img, xs - disp * dx / r, ys - disp * dy / r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if glow > 0:
        out = np.clip(out + np.float32(glow) * rim[..., None] * np.asarray(colour, np.float32), 0.0, 1.0)
    return out


def slab_cover(size, centres, w, h, slant, angle):
    """Coverage (rows, cols) 0..1 of a parallelogram `w` x `h` px whose sides lean `slant` degrees (to the right going up,
    as italics do) and which is turned `angle` degrees counter-clockwise on screen, averaged over the pixel `centres`
    [(x, y), ...] it stands at in turn (one: still; several along its way: blurred by its motion)."""
    W, H = int(size[0]), int(size[1])
    out = np.zeros((H, W), np.float32)
    a, t = np.radians(angle), np.tan(np.radians(slant))
    ca, sa, cs = np.float32(np.cos(a)), np.float32(np.sin(a)), np.float32(np.cos(np.radians(slant)))
    reach = 0.5 * (w + abs(h * t)) + 0.5 * h + 2.0                       # the slab's half-extent, any way, plus the edge
    for cx, cy in centres:
        x0, x1 = max(int(np.floor(cx - reach)), 0), min(int(np.ceil(cx + reach)) + 1, W)
        y0, y1 = max(int(np.floor(cy - reach)), 0), min(int(np.ceil(cy + reach)) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - np.float32(cx),
                           np.float32(cy) - np.arange(y0, y1, dtype=np.float32))   # y up
        u, v = X * ca + Y * sa, -X * sa + Y * ca
        d = np.maximum(np.abs(v) - np.float32(h / 2.0), (np.abs(u - v * np.float32(t)) - np.float32(w / 2.0)) * cs)
        out[y0:y1, x0:x1] += np.clip(0.5 - d, 0.0, 1.0)
    return out / max(len(centres), 1)


def gradient_map(img, dark, colour, light):
    """Each pixel by its brightness through dark (black) .. `colour` (mid grey) .. light (white)."""
    L = np.clip(img @ LUMA, 0.0, 1.0)[..., None]
    lo = np.asarray(dark, np.float32) + (np.asarray(colour, np.float32) - np.asarray(dark, np.float32)) * (2.0 * L)
    hi = np.asarray(colour, np.float32) + (np.asarray(light, np.float32) - np.asarray(colour, np.float32)) * (2.0 * L - 1.0)
    return np.where(L < 0.5, lo, hi).astype(np.float32)


# ================================================================================================= the effects, pure
def lens(img, c, theta, shadow=0.0):
    """A point mass at pixel `c` (x, y) with Einstein radius `theta` (px): a pixel at distance r from c shows the picture
    at r - theta^2 / r along the same line (inside the ring that is the far side, folded in; on the ring, c itself),
    nothing from outside the frame; `shadow` x theta about c is black."""
    h, w = img.shape[:2]
    if theta <= 0:
        return img
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - np.float32(c[0]), yy - np.float32(c[1])
    r2 = np.maximum(dx * dx + dy * dy, np.float32(1e-6))
    k = 1.0 - np.float32(theta * theta) / r2
    out = cv2.remap(np.ascontiguousarray(img, np.float32), np.float32(c[0]) + dx * k, np.float32(c[1]) + dy * k,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
    if shadow > 0:
        out[r2 < (shadow * theta) ** 2] = 0.0
    return out


def echo(img, copies, blend="lighten", tint=None):
    """`img` with `copies` [(picture, alpha), ...] laid on it: `lighten` keeps the brighter of the two (a light figure
    on a dark ground leaves light echoes), `darken` the darker (a dark figure on a light ground). A `tint` (rgb) colours
    the echoes where they differ from the ground: a copy of the ground itself changes nothing."""
    out = np.array(img, np.float32, copy=True)
    t = None if tint is None else np.asarray(tint, np.float32)
    for pic, a in copies:
        if a <= 0:
            continue
        if blend == "lighten":
            np.maximum(out, a * pic * (1.0 if t is None else t), out=out)
        else:
            np.minimum(out, 1.0 - a * (1.0 - pic) * (1.0 if t is None else 1.0 - t), out=out)
    return out


def slit_ages(n, depth, reverse):
    """The age (frames back) of each of `n` columns or rows: 0 at the present edge, `depth` at the far one."""
    u = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    a = np.rint(depth * (u if reverse else 1.0 - u)).astype(int)
    return a


def slitscan(img, older, now="right", depth=24):
    """Each column (`now` left / right) or row (top / bottom) of the picture from a frame of its own age: `older(k)` is
    the frame k back (k >= 1; None when the clip has none so old: the oldest there is stands in)."""
    h, w = img.shape[:2]
    across = now in ("left", "right")
    ages = slit_ages(w if across else h, depth, reverse=now in ("left", "top"))
    out = np.array(img, np.float32, copy=True)
    last = img
    for k in range(1, int(ages.max()) + 1 if len(ages) else 1):
        pic = older(k)
        last = last if pic is None else pic
        sel = ages == k
        if sel.any():
            if across:
                out[:, sel] = last[:, sel]
            else:
                out[sel] = last[sel]
    return out


def zoom(log_zmax, u, ease):
    """(zoom factor, share of the way in log zoom) at progress u (0..1)."""
    if ease == "linear":
        z = 1.0 + (np.exp(log_zmax) - 1.0) * u
        return z, (np.log(z) / log_zmax if log_zmax > 0 else 1.0)
    g = u * u * (3.0 - 2.0 * u) if ease == "smooth" else u
    return float(np.exp(log_zmax * g)), g


def push(img, inner, c, r0, u, ease="log"):
    """The picture zooming into a disc of radius r0 (px) about `c` that shows `inner`: at u = 0 the picture as it is, at
    u = 1 the disc, its centre glided to the frame's middle, covers the frame and `inner` is shown at its own size."""
    h, w = img.shape[:2]
    m = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    R = float(np.hypot(w, h)) / 2.0 + 1.0
    z, g = zoom(np.log(max(R / max(r0, 1e-6), 1.0)), float(u), ease)
    p = np.asarray(c, float) + (m - np.asarray(c, float)) * g                       # the disc's centre on screen
    r = r0 * z
    s = r / R                                                                        # the inner frame's scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - np.float32(p[0]), yy - np.float32(p[1])
    outer = cv2.remap(np.ascontiguousarray(img, np.float32), np.float32(c[0]) + dx / np.float32(z),
                      np.float32(c[1]) + dy / np.float32(z), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inside = cv2.remap(np.ascontiguousarray(inner, np.float32), np.float32(m[0]) + dx / np.float32(s),
                       np.float32(m[1]) + dy / np.float32(s), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cov = np.clip(r + 0.5 - np.sqrt(dx * dx + dy * dy), 0.0, 1.0)[..., None]        # a one-pixel soft edge
    return outer * (1.0 - cov) + inside * cov


def scope_points(x, i, n, lag, gain):
    """(n, 2) points in -1..1 of an XY trace of the mono signal `x` over samples i - n .. i: the signal against itself
    `lag` samples later, scaled by `gain`."""
    a = max(0, i - n - lag)
    seg = np.asarray(x[a:i], np.float32)
    if len(seg) <= lag + 1:
        return np.zeros((0, 2), np.float32)
    return np.clip(np.stack([seg[:len(seg) - lag], seg[lag:]], 1) * gain, -1.0, 1.0) if lag else \
        np.clip(np.stack([np.linspace(-1, 1, len(seg)), seg], 1) * [1.0, gain], -1.0, 1.0)


def draw_trace(img, pts, centre, radius, colour, width, glow, opacity=1.0, px=1.0):
    """`img` with the trace `pts` (n, 2, -1..1; y up) drawn as a line of `width` px about `centre` (px) at `radius` px,
    added in `colour` with a glow of share `glow`."""
    h, w = img.shape[:2]
    if len(pts) < 2 or opacity <= 0:
        return img
    q = np.stack([centre[0] + pts[:, 0] * radius, centre[1] - pts[:, 1] * radius], 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.rint(q * (1 << LINE_SHIFT)).astype(np.int32)], False, 255, max(1, int(round(width))),
                  cv2.LINE_AA, shift=LINE_SHIFT)
    a = mask.astype(np.float32) / 255.0
    if glow > 0:
        a = a + glow * cv2.GaussianBlur(a, (0, 0), GLOW_SIGMA * px) * 3.0
    return np.clip(img + opacity * a[..., None] * np.asarray(colour, np.float32), 0.0, 1.0)


def stem_source(root, cache_dir):
    """callable(timeline, stem) -> (mono float32 samples, sample rate, pre) for a project at `root`: the stems of the
    window `timeline` (project-relative) was analysed on, from the stem cache `cache_dir` only (mkmmd.timeline.hits
    .cached_stems), each loaded once; the timeline's audio file, like every path of a project, is project-relative.
    A FileNotFoundError says what to run when they are not there."""
    import json
    import subprocess
    from pathlib import Path

    from .timeline import hits as H
    from .timeline import signal as S
    loaded = {}

    def get(timeline, stem):
        if timeline not in loaded:
            p = Path(root) / Path(timeline).expanduser()
            try:
                tl = json.loads(p.read_text(encoding="utf-8"))
                au = dict(tl.get("audio") or {})
                if au.get("file"):
                    f = Path(au["file"]).expanduser()
                    au["file"] = str(f if f.is_absolute() else Path(root) / f)
                st, pre = H.cached_stems({**tl, "audio": au}, str(cache_dir))
            except (OSError, ValueError, subprocess.CalledProcessError) as e:     # HitsError is a ValueError
                raise FileNotFoundError(f"scope: {p}: {e}") from None
            loaded[timeline] = ({k: np.asarray(v, np.float32).mean(0) for k, v in st.items()}, pre)
        st, pre = loaded[timeline]
        return st[stem], S.SR, pre
    return get


def project_hits(root, data):
    """The [audio] hits file of a project (mk.toml `data`, at `root`) as {lane: [...]}, or None when it names none; a
    zoom or a shock on a lane is then an error that says to write it."""
    rel = (data.get("audio") or {}).get("hits")
    if not rel:
        return None
    p = Path(rel).expanduser()
    try:
        return LN.load(p if p.is_absolute() else Path(root) / p)
    except LN.LaneError as e:
        raise FileNotFoundError(str(e)) from None


# ================================================================================================= over a cut
class FrameFx:
    """The frame effects of a plan's `fx` over one output's frames.

    size      (width, height) of the pictures
    frame_of  callable(Blender frame) -> picture or None: the cut's frame there, for echoes, slitscans and pushes
    point_of  callable(key, frame) -> ((x, y) px, frame heights per metre at its depth): a projected scene point
    stem_of   callable(timeline, stem name) -> (mono float32 samples, sample rate, song seconds before clip time 0 in
              them), for scopes (None: a scope is an error); `stem_source` makes one for a project
    hits      the project's [audio] hits file ({lane: [...]}, mkmmd.core.lanes), for zooms and shocks on a lane"""

    def __init__(self, fxs, size, *, fps, frame0, frame_of, point_of, stem_of=None, hits=None):
        self.fxs, self.size, self.fps, self.frame0 = fxs, (int(size[0]), int(size[1])), float(fps), int(frame0)
        self._frame_of, self._point_of, self._stem_of, self._hits = frame_of, point_of, stem_of, hits
        self.px = min(self.size) / 1080.0
        self._cache = OrderedDict()
        self._gain, self._times = {}, {}

    def active(self, frame):
        return bool(FX.at(self.fxs, frame))

    def frame(self, frame, img):
        """`img` (the cut's frame `frame`, its cut effects composited) with every effect on that frame, in table order."""
        for fx in FX.at(self.fxs, frame):
            img = getattr(self, "_" + fx["kind"])(fx, frame, img)
        return img

    # ---------------------------------------------------------------- inputs
    def _get(self, f):
        if f in self._cache:
            self._cache.move_to_end(f)
            return self._cache[f]
        pic = self._frame_of(f) if f >= self.frame0 else None
        self._cache[f] = pic
        while len(self._cache) > CACHE:
            self._cache.popitem(last=False)
        return pic

    def _at(self, fx, frame):
        """(centre px, frame heights per metre at its depth or None) of an effect's `center` on `frame`."""
        w, h = self.size
        if fx["center"]["mode"] == "frame":
            x, y = fx["center"]["at"]
            return (x * w - 0.5, y * h - 0.5), None
        return self._point_of(FX.point_key(fx), frame)

    def _centre(self, fx, frame):
        """(centre px, radius px) of a lens or push on `frame`."""
        c, m = self._at(fx, frame)
        h = self.size[1]
        if "radius_m" in fx:
            return c, fx["radius_m"] * m * h
        return c, fx["radius"] * h

    def _hit_times(self, fx):
        if fx["index"] not in self._times:
            self._times[fx["index"]] = FX.hit_times(fx, self._hits)
        return self._times[fx["index"]]

    def _t(self, frame):
        return (frame - self.frame0) / self.fps

    # ---------------------------------------------------------------- kinds
    def _lens(self, fx, frame, img):
        c, theta = self._centre(fx, frame)
        return lens(img, c, theta * np.sqrt(max(FX.strength(fx, self._t(frame)), 0.0)), fx["shadow"])

    def _echo(self, fx, frame, img):
        a0 = fx["fade"] * FX.strength(fx, self._t(frame))
        copies = []
        for j, f in enumerate(FX.sources(fx, frame, self.frame0)):
            pic = self._get(f)
            if pic is not None:
                copies.append((pic, a0 * fx["fade"] ** j))
        return echo(img, copies, fx["blend"], fx["tint"])

    def _slitscan(self, fx, frame, img):
        depth = int(round(fx["depth"] * max(FX.strength(fx, self._t(frame)), 0.0)))
        if depth < 1:
            return img
        return slitscan(img, lambda k: self._get(frame - k), fx["now"], depth)

    def _push(self, fx, frame, img):
        c, r0 = self._centre(fx, frame)
        inner = self._get(FX.sources(fx, frame, self.frame0)[0])
        if inner is None:
            raise FileNotFoundError(f"fx {fx['index']} (push): the frame inside, {fx['inside_frame']}, is not there")
        return push(img, inner, c, r0, FX.progress(fx, frame), fx["ease"])

    def _scope(self, fx, frame, img):
        if self._stem_of is None:
            raise FileNotFoundError(f"fx {fx['index']} (scope): no audio to draw")
        x, sr, pre = self._stem_of(fx["timeline"], fx["stem"])
        if fx["index"] not in self._gain:                       # one gain over the effect's span: the trace keeps its size
            a, b = int((pre + fx["from"]) * sr), int((pre + fx["to"]) * sr)
            peak = float(np.percentile(np.abs(x[max(a, 0):max(b, a + 1)]), 99.5)) if b > a else 0.0
            self._gain[fx["index"]] = fx["gain"] / max(peak, 1e-6)
        i = int(round((pre + self._t(frame)) * sr))
        pts = scope_points(x, i, int(fx["window"] * sr), int(round(fx["lag"] * sr)), self._gain[fx["index"]])
        w, h = self.size
        return draw_trace(img, pts, (fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5), fx["size"] * h, fx["color"],
                          fx["width"] * self.px, fx["glow"], FX.strength(fx, self._t(frame)), self.px)

    def _zoom(self, fx, frame, img):
        t = self._t(frame)
        k = float(FX.punch(t, self._hit_times(fx), fx["attack"], fx["decay"])) * FX.strength(fx, t)
        return scale_about(img, self._at(fx, frame)[0], 1.0 + fx["amount"] * k)

    def _shock(self, fx, frame, img):
        t, h = self._t(frame), self.size[1]
        s = FX.strength(fx, t)
        waves = [(fx["speed"] * (t - th) * h, s * (1.0 - (t - th) / fx["life"]) ** 2)
                 for th in self._hit_times(fx) if 0.0 <= t - th < fx["life"]]
        return shock(img, self._at(fx, frame)[0], waves, fx["width"] * h, fx["amount"] * h, fx["color"], fx["glow"])

    def _slab(self, fx, frame, img):
        w, h = self.size
        sw, sh = fx["size"][0] * h, fx["size"][1] * h
        C = np.array([fx["at"][0] * w - 0.5, fx["at"][1] * h - 0.5])
        v = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}[fx["dir"]]
        rb = 0.5 * float(np.hypot(sw + abs(sh * np.tan(np.radians(fx["slant"]))), sh))   # its reach any way
        ahead = (w - C[0]) if v[0] > 0 else C[0] if v[0] < 0 else (h - C[1]) if v[1] > 0 else C[1]
        behind = (w if v[0] else h) - ahead
        t = self._t(frame)
        n = SLAB_SAMPLES if fx["shutter"] > 0 else 1
        times = [t + fx["shutter"] / self.fps * ((j + 0.5) / n - 0.5) for j in range(n)] if n > 1 else [t]
        centres = []
        for tj in times:
            u = FX.slab_offset(fx, tj)
            d = u * ((behind + rb) if u < 0 else (ahead + rb))
            centres.append((C[0] + v[0] * d, C[1] + v[1] * d))
        cov = slab_cover(self.size, centres, sw, sh, fx["slant"], fx["angle"])
        a = (fx["alpha"] * FX.strength(fx, t) * cov)[..., None]
        return (img * (1.0 - a) + np.asarray(fx["color"], np.float32) * a).astype(np.float32)

    def _tint(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        if s <= 0.0:
            return img
        return (img * (1.0 - s) + gradient_map(img, fx["dark"], fx["color"], fx["light"]) * s).astype(np.float32)
