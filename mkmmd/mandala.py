"""Mandalas and trails on the finished pictures (docs/design.md: Frame effects, Mandalas and trails): the picture work of
the `[[fx]]` kinds `mandala` and `trails` that mkmmd.core.mandala checks, times and lays out. MandalaFx is the part of
mkmmd.framefx.FrameFx that runs them; `sprite_source` finds a mandala's sprites (registry slugs and tags, files, globs)
and `sprite_slugs` the registry entries among them for the credits.

Every copy of a mandala is a sticker drawn once into a tile at its ring's size (its picture premultiplied, its coverage,
its die-cut edge grown round it) and laid onto the picture by an affine map, its drop shadow cast where it lands: tiles of
sprites and drawn shapes are kept from frame to frame, the figure's are cut again on each frame. Images are float RGB in
display space (0..1), (h, w, 3); a coverage is (h, w) in 0..1."""
import glob
import math
from collections import OrderedDict, namedtuple
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import eyes as EYD
from .core import eyes as EY
from .core import framefx as FX
from .core import mandala as MD

SPRITE_EXT = (".png", ".webp", ".tif", ".tiff", ".jpg", ".jpeg")
GLOB = "*?["
SUPER = 3                        # a drawn shape is painted this many times larger and shrunk: smooth edges, fine lines
TILES = 256                      # sticker tiles kept from frame to frame
LAYOUTS = 1024                   # a mandala's layouts at the googly eyes' samples, kept for the frames after
DARKEN, LIGHTEN = 0.35, 0.55     # a shape's `dark` and `light`: its colour this far toward the ink, toward white
WHITE = (1.0, 1.0, 1.0)
MIN_SCALE = 0.01                 # a copy laid smaller than this (popping in) is left out

# a sticker tile: its picture premultiplied (h, w, 3), coverage, edge (or None), anchor (x, y) px, and its face (the
# geometry of its googly eyes and smile in its px, mkmmd.eyes.geometry; None: none)
Tile = namedtuple("Tile", "rgb a ring anchor face", defaults=(None,))


def _ff():
    from . import framefx as FF          # framefx imports this module for MandalaFx: its helpers are there by now
    return FF


# ================================================================================================= sprites
def _is_path(ref):
    return any(c in ref for c in "/\\~" + GLOB) or Path(ref).suffix.lower() in SPRITE_EXT


def _image(p):
    return Path(p).suffix.lower() in SPRITE_EXT


def _resolve(refs, root, reg, strict, what="sprites"):
    """[(path, slug or None)] of a sprite source's `refs` (MD._refs): a slug is a registry entry whose path is a picture,
    {tag} every picture in the registry with all those tags (by slug), a file or a glob (project-relative to `root`,
    absolute or ~/...) the pictures there, each once in order. `reg()` gives the registry (mkmmd.assets.Registry), read
    when first needed. Strict: what finds nothing is an FX.FxError; else it is left out. Each picture's registry
    `extra.eyes` (its googly eyes' anchors) is noted for its tiles (mkmmd.eyes.remember)."""
    out, seen = [], set()

    def add(p, slug, e=None):
        p = str(Path(p).resolve())
        EYD.remember(p, ((e or {}).get("extra") or {}).get("eyes"))
        if p not in seen:
            seen.add(p)
            out.append((p, slug))

    def entry_path(r, e):
        p = Path(e["path"]).expanduser()
        return p if p.is_absolute() else r.root / p

    def fail(msg):
        if strict:
            raise FX.FxError(f"{what}: {msg}")
    for ref in refs:
        if isinstance(ref, dict):
            r = reg()
            hits = sorted((e for e in r.entries if set(ref["tag"]) <= set(e.get("tags") or []) and e.get("path")
                           and _image(e["path"])), key=lambda e: e["slug"])
            if not hits:
                fail(f"no picture in the asset registry ({r.path}) has the tags {ref['tag']}")
            for e in hits:
                p = entry_path(r, e)
                if p.is_file():
                    add(p, e["slug"], e)
                else:
                    fail(f"sprite {e['slug']!r}: {p} does not exist")
            continue
        if _is_path(ref):
            p = Path(ref).expanduser()
            p = p if p.is_absolute() or root is None else Path(root) / p
            if any(c in ref for c in GLOB):
                files = sorted(f for f in glob.glob(str(p)) if _image(f) and Path(f).is_file())
                if not files:
                    fail(f"{ref!r}: no pictures ({', '.join(SPRITE_EXT)}) match {p}")
            elif p.is_file():
                files = [str(p)]
            else:
                files = []
                fail(f"{ref!r}: {p} is not a file")
            for f in files:
                e = reg().find_path(f)
                add(f, e["slug"] if e else None, e)
            continue
        r = reg()
        e = next((x for x in r.entries if x.get("slug") == ref), None)
        if e is None or not e.get("path") or not _image(e["path"]):
            fail(f"{ref!r}: not a picture in the asset registry ({r.path}) and not a file "
                 f"(a sprite is a slug, a file, a glob or {{tag = ...}})")
            continue
        p = entry_path(r, e)
        if p.is_file():
            add(p, ref, e)
        else:
            fail(f"sprite {ref!r}: {p} does not exist")
    return out


def _registry(registry):
    held = [registry]

    def reg():
        if held[0] is None:
            from .assets import Registry
            held[0] = Registry()
        return held[0]
    return reg


def sprite_source(root=None, registry=None):
    """callable(refs) -> [picture files] of a mandala ring's sprites (its normalised `sprites`): registry slugs and tags
    (`registry`, an mkmmd.assets.Registry; the configured one by default, read when first needed), files and globs
    (project-relative to `root`, absolute or ~/...). An FX.FxError says what finds nothing."""
    reg = _registry(registry)

    def get(refs):
        return [p for p, _ in _resolve(refs, root, reg, True)]
    return get


def sprite_refs(data):
    """The sprite references of a project (mk.toml `data`): every `sprites` value at any depth of its `[[fx]]` and
    `[[insert]]` entries (a mandala ring's source, a stack's band, a paint app's stamps, an iris's plate ring, their
    aspect.<output> tables too), each once, as they are written."""
    out = []

    def add(items):
        for r in [items] if isinstance(items, (str, dict)) else items if isinstance(items, list) else []:
            if isinstance(r, dict) and isinstance(r.get("tag"), (str, list)):
                r = {"tag": [r["tag"]] if isinstance(r["tag"], str) else list(r["tag"])}
            elif isinstance(r, str) and r.strip():
                r = r.strip()
            else:
                continue
            if r not in out:
                out.append(r)

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "sprites":
                    add(x)
                else:
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for key in ("fx", "insert"):
        walk(data.get(key) or [])
    return out


def sprite_slugs(data, root, registry=None):
    """The registry entries a project's sprites show (sprite_refs; mk.toml `data`, the project at `root`), each once:
    the slugs and tags they name, and the entries whose path is a file or a glob's match; what finds nothing is left to
    mk post and mk look to report."""
    refs = sprite_refs(data)
    if not refs:
        return []
    return list(dict.fromkeys(s for _, s in _resolve(refs, root, _registry(registry), False) if s))


@lru_cache(maxsize=48)
def _sprite(path):
    """A sprite's picture as (h, w, 4) uint8 RGBA, trimmed to what its alpha covers (opaque without one); an FX.FxError
    when it cannot be read or is empty."""
    try:
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA"))
    except (OSError, ValueError) as e:
        raise FX.FxError(f"sprite {path}: {e}") from None
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    if not len(xs):
        raise FX.FxError(f"sprite {path}: nothing in it (its alpha is clear everywhere)")
    return np.ascontiguousarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


# ================================================================================================= stickers
def _resized(x, size, shrink):
    return cv2.resize(np.ascontiguousarray(x, np.float32), size,
                      interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def tile(rgb, a, anchor, edge, face=None):
    """A sticker tile of a premultiplied picture `rgb` and its coverage `a` (anchor (x, y) px, where it is laid and
    turned): padded for an `edge` px wide (FF._grow: a die-cut border, rounded where it bends in; 0 none). `face`: the
    geometry of its googly eyes and smile in the picture's px (mkmmd.eyes.paint), moved with the padding."""
    pad = int(math.ceil(edge)) + 2
    rgb = cv2.copyMakeBorder(np.ascontiguousarray(rgb, np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = cv2.copyMakeBorder(np.ascontiguousarray(a, np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ring = _ff()._grow(a, edge) if edge > 0 else None
    return Tile(rgb, a, ring, (anchor[0] + pad, anchor[1] + pad), None if face is None else EYD.shifted(face, pad, pad))


def sprite_tile(path, size, edge, face=None, tint=None):
    """Sprite `path` as a sticker tile, its longer side `size` px, anchored at its box's middle; its own pixels mixed
    toward `tint` ((rgb, amount 0..1): a band's colour; None: as they are). With a `face` (mkmmd.core.eyes.face) its
    googly eyes' whites and its smile are painted on where its anchors say (its registry entry's extra.eyes, else found
    from its alpha; mkmmd.eyes.anchors_for), the tile grown to hold eyes that stick out of it; the pupils are drawn on
    each frame where it is laid (lay, mkmmd.eyes.pupils)."""
    rgba = _sprite(str(path))
    h, w = rgba.shape[:2]
    s = size / float(max(h, w))
    nw, nh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    f = rgba.astype(np.float32) / np.float32(255.0)
    a = f[..., 3]
    rgb = f[..., :3] * a[..., None]
    if (nw, nh) != (w, h):
        rgb, a = _resized(rgb, (nw, nh), s < 1.0), _resized(a, (nw, nh), s < 1.0)
        a = np.clip(a, 0.0, 1.0)
        rgb = np.minimum(np.clip(rgb, 0.0, 1.0), a[..., None])
    if tint is not None:
        colour, k = tint
        rgb = rgb + (np.asarray(colour, np.float32) * a[..., None] - rgb) * np.float32(k)
    anchor, geo = ((nw - 1) / 2.0, (nh - 1) / 2.0), None
    if face is not None:
        rgb, a, anchor, geo = EYD.paint(rgb, a, anchor, face, EYD.anchors_for(str(path), f[..., 3]))
    return tile(rgb, a, anchor, edge, geo)


def _mix(a, b, k):
    return tuple(float(x + (y - x) * k) for x, y in zip(a, b))


@lru_cache(maxsize=16)
def _parts(name):
    return MD.shape_parts(name)


def shape_tile(name, main, accent, ink, size, edge):
    """Sticker shape `name` (mkmmd.core.mandala.shape_parts) as a tile, its box's longer side `size` px, anchored at the
    shape's middle: its parts painted in order in `main`, `accent` and `ink` and the shades of the first two (`dark`
    toward the ink, `light` toward white), at SUPER times the size and shrunk."""
    parts = _parts(name)
    lo, hi = MD.shape_box(parts)
    k = size * SUPER
    m = 2.0 * SUPER                                      # canvas px round the shape
    W, H = int(math.ceil((hi[0] - lo[0]) * k + 2 * m)), int(math.ceil((hi[1] - lo[1]) * k + 2 * m))
    W, H = W + (-W) % SUPER, H + (-H) % SUPER
    colours = {"main": main, "dark": _mix(main, ink, DARKEN), "light": _mix(main, WHITE, LIGHTEN), "accent": accent,
               "accent_dark": _mix(accent, ink, DARKEN), "ink": ink}
    rgb, a = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
    for role, kind, p, width in parts:
        q = np.rint(((np.asarray(p, float) - lo) * k + m) * 16.0).astype(np.int32)
        cov = np.zeros((H, W), np.uint8)
        if kind == "fill":
            cv2.fillPoly(cov, [q], 255, cv2.LINE_AA, 4)
        else:
            cv2.polylines(cov, [q], False, 255, max(int(round(width * k)), 1), cv2.LINE_AA, 4)
        c = (cov.astype(np.float32) / np.float32(255.0))
        rgb = rgb * (1.0 - c[..., None]) + np.asarray(colours[role], np.float32) * c[..., None]
        a = a + (1.0 - a) * c
    w, h = W // SUPER, H // SUPER
    rgb, a = _resized(rgb, (w, h), True), np.clip(_resized(a, (w, h), True), 0.0, 1.0)
    anchor = ((-lo[0] * k + m + 0.5) / SUPER - 0.5, (-lo[1] * k + m + 0.5) / SUPER - 0.5)
    return tile(rgb, a, anchor, edge)


def figure_geometry(m, s, grow=0, centre=None):
    """Where figure_tile cuts the figure (coverage `m` of a frame) out and how large it makes it: ((X0, Y0, X1, Y1) the
    part cut, its box with `grow` px more round it, (nw, nh) that part `s` times as large, (sx, sy) the scale it took,
    the anchor's pixel of the frame: `centre`, or the centroid); None when nothing is covered."""
    FF = _ff()
    x, y, w, h = cv2.boundingRect((np.asarray(m) > 1.0 / 512.0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    c = FF.centroid(m) if centre is None else centre
    if c is None:
        return None
    H, W = m.shape
    X0, Y0, X1, Y1 = max(x - grow, 0), max(y - grow, 0), min(x + w + grow, W), min(y + h + grow, H)
    cw, ch = X1 - X0, Y1 - Y0
    nw, nh = max(int(round(cw * s)), 1), max(int(round(ch * s)), 1)
    return (X0, Y0, X1, Y1), (nw, nh), (nw / float(cw), nh / float(ch)), c


def figure_tile(pic, m, s, fill, edge, centre=None):
    """The figure (picture `pic`, coverage `m` of a frame) as a sticker tile `s` times its size, drawn `fill` (None: its
    picture; a colour; a silkscreen {inks, offset px, soft px, line px}), anchored at `centre` (px of the frame; default
    its centroid); None when nothing is covered."""
    FF = _ff()
    silk = isinstance(fill, dict)
    grow = 0 if not silk else int(math.ceil(3.0 * (fill["soft"] + fill["line"]))) + 1   # their blurs reach this far
    geo = figure_geometry(m, s, grow, centre)
    if geo is None:
        return None
    (X0, Y0, X1, Y1), (nw, nh), (sx, sy), c = geo
    mc = np.ascontiguousarray(m[Y0:Y1, X0:X1], np.float32)
    if silk:
        key, steps = FF.silk_bands(pic[Y0:Y1, X0:X1], mc, len(fill["inks"]), fill["soft"], fill["line"])
        planes = [mc, key] + list(steps)
    elif fill is None:
        planes = [mc, np.ascontiguousarray(pic[Y0:Y1, X0:X1] * mc[..., None], np.float32)]
    else:
        planes = [mc]
    if (nw, nh) != (X1 - X0, Y1 - Y0):
        planes = [_resized(p, (nw, nh), s < 1.0) for p in planes]
    cov = np.clip(planes[0], 0.0, 1.0)
    if silk:
        off = fill["offset"]
        pad = int(math.ceil(math.hypot(*off))) + 1
        border = lambda p: cv2.copyMakeBorder(p, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)   # noqa: E731
        cov, key, steps = border(cov), border(planes[1]), [border(p) for p in planes[2:]]
        rgb = FF.silk_print(np.zeros(cov.shape + (3,), np.float32), cov, key, steps, fill["inks"], off)
        k = FF.plate_shift(key, off[0], off[1]) if off[0] or off[1] else key
        a = cov + k - cov * k
        X0, Y0 = X0 - pad / sx, Y0 - pad / sy
    elif fill is None:
        rgb, a = np.minimum(np.clip(planes[1], 0.0, 1.0), cov[..., None]), cov
    else:
        rgb, a = np.asarray(fill, np.float32) * cov[..., None], cov
    anchor = ((c[0] - X0 + 0.5) * sx - 0.5, (c[1] - Y0 + 0.5) * sy - 0.5)
    return tile(rgb, a, anchor, edge)


def placed(t, at, angle, scale, mirror=False):
    """The affine map (2 x 3, a tile's px to the picture's) that lays tile `t` with its anchor at pixel `at`, turned
    `angle` degrees counter-clockwise on screen, `scale` times its size, flipped left to right first when `mirror`."""
    r = math.radians(angle)
    c, s = math.cos(r) * scale, math.sin(r) * scale
    f = -1.0 if mirror else 1.0
    A = np.array([[c * f, s], [-s * f, c]])
    return np.hstack([A, (np.asarray(at, float) - A @ np.asarray(t.anchor, float))[:, None]])


def lay_box(shape, M, shadow, size):
    """Where `lay` draws a tile of `shape` ((h, w)) laid by the affine map `M` in a picture of `size` (w, h): ((x0, y0,
    x1, y1) the box it covers, its drop `shadow` too; the map from the tile's px to the box's), None for nothing."""
    h, w = shape
    M = np.asarray(M, np.float64)
    corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [-0.5, h - 0.5], [w - 0.5, h - 0.5]])
    pts = corners @ M[:, :2].T + M[:, 2]
    lo, hi = pts.min(0), pts.max(0)
    if shadow is not None and shadow[3] > 0:
        reach = 3.0 * shadow[2]
        lo = np.minimum(lo, lo + shadow[1]) - reach
        hi = np.maximum(hi, hi + shadow[1]) + reach
    W, H = size
    x0, y0 = max(int(math.floor(lo[0])), 0), max(int(math.floor(lo[1])), 0)
    x1, y1 = min(int(math.ceil(hi[0])) + 1, W), min(int(math.ceil(hi[1])) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    Mt = M.copy()
    Mt[:, 2] -= (x0, y0)
    return (x0, y0, x1, y1), Mt


def lay(out, t, M, edge=None, shadow=None, pupils=None):
    """`out` (in place, and returned) with sticker tile `t` laid by the affine map `M` (placed()): its drop `shadow`
    (colour, (dx, dy) px, blur px, alpha) cast first where it lands, then its `edge` (colour, or None) and its picture,
    all within the box it covers (lay_box); then, for a tile with googly eyes, its pupils at the offsets `pupils`
    (mkmmd.eyes.pupils; None: at rest) and the glints over them (mkmmd.eyes.draw_pupils)."""
    FF = _ff()
    got = lay_box(t.a.shape, M, shadow, (out.shape[1], out.shape[0]))
    if got is None:
        return out
    (x0, y0, x1, y1), Mt = got
    edged = t.ring is not None and edge is not None
    size = (x1 - x0, y1 - y0)
    got = FF._warp(np.dstack([t.rgb, t.a]), Mt, size)              # OpenCV shrinks by area up to 4 channels at once
    ring = np.clip(FF._warp(t.ring, Mt, size), 0.0, 1.0) if edged else None
    a = np.clip(got[..., 3], 0.0, 1.0)
    rgb = got[..., :3]
    if edged:                                        # the edge under the picture (it holds the picture's coverage)
        rgb = rgb + np.asarray(edge, np.float32) * (ring * (1.0 - a))[..., None]
        a = a + ring - a * ring
    if shadow is not None and shadow[3] > 0:         # the shadow under both, cast from what they cover
        k = np.float32(shadow[3]) * soft(FF.plate_shift(a, shadow[1][0], shadow[1][1]), shadow[2])
        rgb = rgb + np.asarray(shadow[0], np.float32) * (k * (1.0 - a))[..., None]
        a = a + k - a * k
    region = out[y0:y1, x0:x1]
    region *= (1.0 - a)[..., None]
    region += rgb
    if t.face is not None:
        EYD.draw_pupils(out, t.face, M, pupils)
    return out


def soft(x, sigma):
    """A coverage blurred `sigma` px (a Gaussian's), a wide blur worked out at half the size (a shadow is soft)."""
    if sigma <= 0.3:
        return x
    h, w = x.shape
    if sigma < 4.0 or min(h, w) < 16:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    half = cv2.resize(x, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(half, (0, 0), sigma / 2.0), (w, h), interpolation=cv2.INTER_LINEAR)


# ================================================================================================= trails
def blend(dst, colour, a, how):
    """`dst` with `colour` (rgb, or a picture of dst's shape) laid on it at opacity `a` ((h, w, 1)): `over`, `screen` or
    `add` (clipped)."""
    c = np.asarray(colour, np.float32)
    if how == "over":
        return dst + (c - dst) * a
    if how == "screen":
        return 1.0 - (1.0 - dst) * (1.0 - c * a)
    return np.minimum(dst + c * a, 1.0)


def _box(m, pad=1):
    """(x0, y0, x1, y1) px holding what coverage `m` covers, `pad` px more; None for nothing."""
    x, y, w, h = cv2.boundingRect((m > 1.0 / 512.0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    H, W = m.shape
    return max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)


def _shifted(x, dx, dy, border):
    """A picture or a coverage moved `dx`, `dy` px (sub-pixel), the frame's edge as `border` says."""
    h, w = x.shape[:2]
    return cv2.warpAffine(np.ascontiguousarray(x, np.float32), np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)


# ================================================================================================= over a cut
class MandalaFx:
    """The mandalas' and trails' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs,
    _t, _get, _at, _figure, _posed, _fill, _edge, _shadow, _hit_times, _mix, _hits and the collages' backs, _back).
    `_mandala_init` is called once its effects are known."""

    def _mandala_init(self, sprite_of):
        """The mandalas' sprites found now and read (FrameFx `preload`): a missing or unreadable one is an FX.FxError
        before any frame is drawn."""
        self._sprite_of = sprite_of or sprite_source()
        self._md_sprites, self._md_tiles, self._md_heights, self._md_grows = {}, OrderedDict(), {}, {}
        self._md_layouts = OrderedDict()                     # (entry, clip second) -> its copies there, for the eyes
        for fx in self.fxs:
            if fx["kind"] != "mandala":
                continue
            for i, r in enumerate(fx["rings"]):
                if r["source"]["kind"] == "sprites":
                    files = self._sprite_of(r["source"]["refs"])
                    for f in files if self._preload else ():
                        _sprite(str(f))
                    self._md_sprites[fx["index"], i] = files

    # ---------------------------------------------------------------- mandala
    def _mandala(self, fx, frame, img):
        t = self._t(frame)
        times = self._hit_times(fx)
        k = FX.step(fx, frame, times, self.fps, self.frame0)
        out = _ff()._flat(self.size, self._back(fx, frame, k, img))     # a colour, a pattern, or a copy of the picture
        pic = m = None
        if fx["objects"] is not None:
            pic, m = self._posed(fx, frame, img, self._figure(fx, frame))
        centre = self._at(fx, frame)[0]
        unit = float(min(self.size))
        edge, shadow = self._edge(fx), self._shadow(fx)
        schedule = self._md_schedule(fx)
        figs = {}                                            # this frame's tiles of the figure
        copies = MD.copies(fx, t, times, self.fps, self.frame0, schedule)
        eyes = self._md_pupils(fx, t, times, schedule, centre, unit, edge)
        for cp in copies:
            r = fx["rings"][cp["ring"]]
            grow = cp["scale"] / r["scale"] * cp["pop"]
            if grow < MIN_SCALE:
                continue
            tl = self._md_tile(fx, cp, r, k, pic, m, unit, edge, figs)
            if tl is None:
                continue
            at = (centre[0] + cp["x"] * unit, centre[1] + cp["y"] * unit)
            lay(out, tl, placed(tl, at, cp["angle"], grow, cp["mirror"]), None if edge is None else edge[0], shadow,
                eyes.get((cp["ring"], cp["j"])))
        c = None if not fx["center_figure"] or m is None else _ff().centroid(m)
        tl = None if c is None else figure_tile(pic, m, 1.0, self._fill(fx, k), 0.0 if edge is None else edge[1], c)
        if tl is not None:                                   # the figure itself, as the frame shows it, over the rings
            lay(out, tl, placed(tl, c, 0.0, 1.0), None if edge is None else edge[0], shadow)
        return self._mix(fx, frame, img, out)

    def _md_tile(self, fx, cp, r, k, pic, m, unit, edge, figs):
        """The tile of a mandala's copy: the figure at its ring's scale in the copy's fill (cut on this frame), or a
        sprite or a shape at its ring's size (kept from frame to frame)."""
        src, j, ew = r["source"], cp["j"], 0.0 if edge is None else edge[1]
        if src["kind"] == "figure":
            if m is None:
                return None
            fill = self._fill(fx, k + j, r["fill"])
            key = (r["scale"], repr(fill))
            if key not in figs:
                figs[key] = figure_tile(pic, m, r["scale"] * unit / self._md_height(fx), fill, ew)
            return figs[key]
        size = r["scale"] * unit
        if src["kind"] == "sprites":
            files = self._md_sprites[fx["index"], cp["ring"]]
            face = src.get("face")
            looks = None if face is None else (face["eyes"], face["mouth"], face["size"], face["look"] == "viewer")
            key = ("sprite", str(files[MD.sprite_pick(fx, cp["ring"], j, len(files))]), round(size, 2), ew, looks)
            make = lambda: sprite_tile(key[1], size, ew, face)                    # noqa: E731
        else:
            main, accent, ink = MD.shape_colours(src, j, k)
            key = ("shape", MD.shape_of(src, j), tuple(main), tuple(accent), tuple(ink), round(size, 2), ew)
            make = lambda: shape_tile(key[1], key[2], key[3], key[4], size, ew)   # noqa: E731
        got = self._md_tiles.get(key)
        if got is None:
            got = self._md_tiles[key] = make()
            while len(self._md_tiles) > TILES:
                self._md_tiles.popitem(last=False)
        else:
            self._md_tiles.move_to_end(key)
        return got

    def _md_layout(self, fx, s, times, schedule):
        """{(ring, j): (x, y, angle, grow, mirror)} of a mandala's copies at clip second `s` (MD.copies; `grow` how many
        times its ring's size a copy is laid), kept for the frames after it."""
        key = (fx["index"], round(s * 1e6))
        got = self._md_layouts.get(key)
        if got is None:
            got = self._md_layouts[key] = {
                (cp["ring"], cp["j"]): (cp["x"], cp["y"], cp["angle"],
                                        cp["scale"] / fx["rings"][cp["ring"]]["scale"] * cp["pop"], cp["mirror"])
                for cp in MD.copies(fx, s, times, self.fps, self.frame0, schedule)}
            while len(self._md_layouts) > LAYOUTS:
                self._md_layouts.popitem(last=False)
        else:
            self._md_layouts.move_to_end(key)
        return got

    def _md_pupils(self, fx, t, times, schedule, centre, unit, edge):
        """{(ring, j): pupil offsets} at clip second `t` of the copies of a mandala's rings of sprites with googly eyes
        (mkmmd.core.eyes.swing): every eye's centre followed over the swing's window through the rings' layout at its
        samples, kicked on the face's own hits or the mandala's, seeded by the entry, the ring and the copy; they look
        where the face says (`center`: the mandala's centre on this frame)."""
        out = {}
        for i, r in enumerate(fx["rings"]):
            face = r["source"].get("face")
            if r["source"]["kind"] != "sprites" or face is None or not face["eyes"]:
                continue
            ts = EY.window(face, t)
            lays = [self._md_layout(fx, s, times, schedule) for s in ts]
            n, count = len(ts), r["count"]
            X, Y, A, G, F = (np.array([[lay_[(i, j)][q] for j in range(count)] for lay_ in lays], np.float64)
                             for q in range(5))
            cent, free = np.zeros((n, count, 2, 2)), np.zeros((n, count, 2))
            ax, ay = centre[0] + X * unit, centre[1] + Y * unit
            cs, sn, fl = np.cos(np.radians(A)) * G, np.sin(np.radians(A)) * G, np.where(F > 0.5, -1.0, 1.0)
            for j in range(count):                        # as placed() lays the copy's tile at each sample
                tl = self._md_tile(fx, {"ring": i, "j": j}, r, 0, None, None, unit, edge, None)
                ex, ey = (np.asarray(tl.face["eyes"], np.float64) - np.asarray(tl.anchor, np.float64)).T
                cent[:, j, :, 0] = ax[:, j, None] + cs[:, j, None] * fl[:, j, None] * ex + sn[:, j, None] * ey
                cent[:, j, :, 1] = ay[:, j, None] - sn[:, j, None] * fl[:, j, None] * ex + cs[:, j, None] * ey
                free[:, j, :] = tl.face["free"] * np.maximum(G[:, j], 0.3)[:, None]
            keys = [[fx["index"], i, j, e] for j in range(count) for e in range(2)]
            got = EY.swing(face, ts, cent.reshape(n, 2 * count, 2), free.reshape(n, 2 * count),
                           EYD.hit_times(face, times, self._hits), keys, EYD.look_at(face, self.size, centre))
            out.update(((i, j), got[2 * j:2 * j + 2]) for j in range(count))
        return out

    def _md_height(self, fx):
        """The height in px of a mandala's figure (its box) on the span's first frame: a ring's `scale` of short sides is
        a copy's height, the copies the same size relative to the rings in every output."""
        got = self._md_heights.get(fx["index"])
        if got is None:
            m = self._figure(fx, fx["f0"])
            box = None if m is None else _box(m, 0)
            if box is None:
                raise FX.FxError(f"fx {fx['index']} (mandala): the figure is not in the frame on the span's first frame "
                                 f"({fx['f0']}): a ring of it sizes its copies by its height there")
            got = self._md_heights[fx["index"]] = float(box[3] - box[1])
        return got

    def _md_schedule(self, fx):
        """A mandala's grow schedule (mkmmd.core.mandala.grow_schedule) on its grow's times or lane; None without one."""
        if fx["grow"] is None:
            return None
        got = self._md_grows.get(fx["index"])
        if got is None:
            g = fx["grow"]
            times = g["times"] if g.get("times") is not None else FX.hit_times(
                {**g, "kind": "mandala grow", "index": fx["index"], "from": fx["from"], "to": fx["to"]}, self._hits)
            got = self._md_grows[fx["index"]] = MD.grow_schedule(fx, times)
        return got

    # ---------------------------------------------------------------- trails
    def trails_at(self, fx, frame):
        """(strength, [(frame g it shows, opacity times the strength, ramp colour float32, (dx, dy) px)]) of a trail's
        copies on `frame`, the oldest first."""
        t = self._t(frame)
        s = float(np.clip(FX.strength(fx, t), 0.0, 1.0))
        unit = float(min(self.size))
        return s, [(g, alpha * s, np.asarray(MD.ramp(fx["colors"], u), np.float32),
                    (fx["step"][0] * unit * j, fx["step"][1] * unit * j))
                   for j, g, alpha, u in MD.trail_copies(fx, frame, t, self._hit_times(fx), self.fps, self.frame0)]

    def _trails(self, fx, frame, img):
        """The copies laid oldest first, each in its ramp colour (mixed with its own picture by 1 - `tint`) at its
        opacity times the strength, then the figure of this frame back over them as it is."""
        s, copies = self.trails_at(fx, frame)
        if s <= 0.0:
            return img
        out = np.array(img, np.float32, copy=True)
        for g, opacity, colour, (dx, dy) in copies:
            mg = self._figure(fx, g)
            if mg is None:
                continue
            pg = self._get(g) if fx["tint"] < 1.0 else None
            if dx or dy:
                mg = _shifted(mg, dx, dy, cv2.BORDER_CONSTANT)
                pg = None if pg is None else _shifted(pg, dx, dy, cv2.BORDER_REPLICATE)
            box = _box(mg)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            if pg is not None:
                colour = pg[y0:y1, x0:x1] + (colour - pg[y0:y1, x0:x1]) * np.float32(fx["tint"])
            a = (np.float32(opacity) * mg[y0:y1, x0:x1])[..., None]
            out[y0:y1, x0:x1] = blend(out[y0:y1, x0:x1], colour, a, fx["blend"])
        m = self._figure(fx, frame)
        box = None if m is None else _box(m)
        if box is not None:
            x0, y0, x1, y1 = box
            k = m[y0:y1, x0:x1, None]
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - k) + img[y0:y1, x0:x1] * k
        return out
