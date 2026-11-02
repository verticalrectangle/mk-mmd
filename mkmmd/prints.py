"""Photographs as prints on the finished pictures (docs/design.md: Frame effects, Prints): the picture work of the
`[[fx]]` kind `prints` that mkmmd.core.prints checks, times and lays out. PrintsFx is the part of mkmmd.framefx.FrameFx
that draws it; `photo_refs` and `photo_slugs` find the registry entries of the photographs a project's prints show, for
the credits; `stereo_halves` finds the other half of a stereograph in the registry (an entry's `extra.stereo`).

A print is drawn once into a tile at its size: its card (cream stock, a faint grain, its edges a little aged), the
photograph whole on it (toned when the entry says), a hairline where the photograph meets the card, and its tape strips
across two corners. The tile is laid onto the picture by an affine map, its drop shadow cast from the card alone where it
lands, larger and softer while the print is in the air; pins are pushed in when it lands. The photograph's own pixels are
only resized: never cut, never retouched. Images are float RGB in display space (0..1), (h, w, 3); a coverage is (h, w)
in 0..1."""
import math
from collections import OrderedDict, namedtuple
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import eralook as ERF
from . import mandala as MDF
from .core import framefx as FX
from .core import prints as PR

TILES = 48                       # print tiles kept from frame to frame
SUPER = 4                        # a pin's head is painted this many times larger and shrunk
HAIRLINE = 0.3                   # how much the hairline round a photograph darkens the card
GRAIN = (0.018, 0.008)           # the card's grain: its slow blotches and its fine tooth, shares of its colour
AGED = (0.07, 0.035)             # how much the card's edges darken, and over what share of its shorter side
CARD_CORNER = 0.012              # a print's card's corners, rounded: a share of its shorter side
STEREO_CORNER = 0.05             # a stereocard's
TAPE_JAG = 0.06                  # how ragged a tape strip's torn ends are, a share of its width
LIFT_SHADOW = (3.0, 4.0, 1.2)    # a print in the air: its shadow's offset and blur grow by these times its height
#                                  (lift x h), its opacity falls by 1 / (1 + the third)
PIN_SHADOW = (0.22, 0.35, 0.3, 0.5)   # a pin's shadow: (dx, dy) and blur in its head's diameters, opacity

# a print's tile: its picture premultiplied (h, w, 3), its coverage (card, photograph and tape), what casts its shadow (the
# card alone), the card's middle (x, y) px (where it is laid and turned), its pins (x, y) px, and its photograph's box
# (x, y, w, h) px on it
class Print(namedtuple("Print", "rgba cast anchor pins photo")):
    """A print's tile: its picture premultiplied and its coverage in one (h, w, 4) float32 array (`rgb`, `a`), what casts
    its shadow (the card alone), the card's middle, its pins and its photograph's box (above)."""
    __slots__ = ()

    @property
    def rgb(self):
        return self.rgba[..., :3]

    @property
    def a(self):
        return self.rgba[..., 3]


def _ff():
    from . import framefx as FF          # framefx imports this module for PrintsFx: its helpers are there by now
    return FF


# ================================================================================================= photographs
@lru_cache(maxsize=4)
def _photo(path):
    """A photograph's pixels as (h, w, 3) uint8 RGB (a grey one made RGB); an FX.FxError when it cannot be read."""
    try:
        with Image.open(path) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB")))
    except (OSError, ValueError) as e:
        raise FX.FxError(f"photo {path}: {e}") from None


def photo_at(path, w, h, tone=None, toning=1.0):
    """Photograph `path` resized to `w` x `h` px (by area when it shrinks), float RGB; with a `tone` (colours, dark to
    light) its luma taken through them, mixed in by `toning`."""
    src = _photo(str(path))
    shrink = w < src.shape[1] or h < src.shape[0]
    pic = cv2.resize(src, (int(w), int(h)), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    pic = np.clip(pic.astype(np.float32) / np.float32(255.0), 0.0, 1.0)
    if tone is None or toning <= 0.0:
        return pic
    mapped = ERF.gradient_map(pic, ERF.lut(tone))
    return (pic + (mapped - pic) * np.float32(toning)).astype(np.float32)


def stereo_halves(path, registry=None):
    """(left, right) halves of the stereograph `path` is one half of: its registry entry's `extra.stereo` names the
    other (`{"right": file}` for a left half, `{"left": file}` for a right one; relative to the registry's folder or
    absolute); without one, (path, path): the half twice. `registry` an mkmmd.assets.Registry (the configured one by
    default)."""
    if registry is None:
        from .assets import Registry
        registry = Registry()
    e = registry.find_path(path)
    st = ((e or {}).get("extra") or {}).get("stereo") or {}
    for side in ("right", "left"):
        other = st.get(side)
        if isinstance(other, str) and other.strip():
            p = Path(other).expanduser()
            p = p if p.is_absolute() else registry.root / p
            if not p.is_file():
                raise FX.FxError(f"photo {path}: the other half of its stereograph, {p}, does not exist")
            return (str(path), str(p)) if side == "right" else (str(p), str(path))
    return str(path), str(path)


# ================================================================================================= coverages
def _rounded(w, h, r, top_only=False, ox=0.0, oy=0.0, shape=None):
    """Coverage of a rectangle `w` x `h` px with its top left at (ox, oy), its corners (the top two only with
    `top_only`) rounded by `r` px, in a picture of `shape` (h, w; default the rectangle's own), antialiased."""
    H, W = shape if shape is not None else (int(math.ceil(h)), int(math.ceil(w)))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = xx + 0.5 - ox, yy + 0.5 - oy                           # pixel centres in the rectangle's own frame
    r = float(min(r, w / 2.0, h / 2.0))
    dx = np.maximum(np.abs(x - w / 2.0) - (w / 2.0 - r), 0.0)
    dy = np.maximum(np.abs(y - h / 2.0) - (h / 2.0 - r), 0.0)
    if top_only:
        dy = np.where(y > h / 2.0, 0.0, dy)
    corner = np.hypot(dx, dy) - r                                 # outside a rounded corner: its distance
    edge = np.maximum(np.abs(x - w / 2.0) - w / 2.0, np.abs(y - h / 2.0) - h / 2.0)
    d = np.where((dx > 0) & (dy > 0), corner, edge)
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def _strip(shape, centre, angle, width, length, jag, rng):
    """Coverage of a tape strip `width` x `length` px about `centre`, its length turned `angle` degrees counter-clockwise
    (y down), its two ends torn (ragged by `jag` px, drawn by `rng`), in a picture of `shape` (h, w), antialiased."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    x, y = xx + 0.5 - centre[0], yy + 0.5 - centre[1]
    u, v = x * c - y * s, x * s + y * c                         # along the strip, across it
    n = 9
    tear = [np.interp(v, np.linspace(-width / 2.0, width / 2.0, n), rng.uniform(-jag, jag, n)) for _ in range(2)]
    du = np.where(u >= 0.0, u - (length / 2.0 + tear[0]), -u - (length / 2.0 + tear[1]))
    dv = np.abs(v) - width / 2.0
    return np.clip(0.5 - np.maximum(du, dv), 0.0, 1.0).astype(np.float32), v / max(width / 2.0, 1e-6)


def _grain(shape, rng):
    """The card's grain over a picture of `shape` (h, w): slow blotches and a fine tooth about 0, shares of GRAIN."""
    h, w = shape
    slow = rng.normal(0.0, 1.0, (max(h // 24, 2), max(w // 24, 2))).astype(np.float32)
    slow = cv2.resize(slow, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    return GRAIN[0] * slow + GRAIN[1] * fine


def _aged(cov, size):
    """How much the card darkens toward its edges: AGED[0] at the edge, gone AGED[1] of its shorter side in."""
    reach = max(AGED[1] * size, 1.0)
    inside = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    return AGED[0] * np.exp(-inside / reach)


def _hairline(cov, colour_dark):
    """Where a hairline runs round a window of coverage `cov`: a ring a pixel wide just outside it."""
    grown = cv2.dilate(cov, np.ones((3, 3), np.float32))
    return np.clip(grown - cov, 0.0, 1.0) * colour_dark


# ================================================================================================= tiles
def _card_paint(cov, colour, size, rng):
    """The card's colour over its coverage `cov`, its grain and aged edges in: (h, w, 3) not premultiplied."""
    tone = 1.0 + _grain(cov.shape, rng) - _aged(cov, size)
    return np.clip(np.asarray(colour, np.float32)[None, None, :] * tone[..., None], 0.0, 1.0)


def _tape(rgb, a, cast_pad, strips, tape, k, rng):
    """`rgb` (premultiplied) and `a` (in place) with the tape strips laid over them, translucent, each a little lighter
    along its middle as matte tape catches the light."""
    for (cx, cy), angle, sw, sl in strips:
        cov, across = _strip(a.shape, (cx + cast_pad, cy + cast_pad), angle, sw, sl, TAPE_JAG * sw, rng)
        colour = np.asarray(PR.pick(tape["colors"], k), np.float32)
        lit = np.clip(colour[None, None, :] * (1.0 + 0.05 * (1.0 - np.abs(across)))[..., None], 0.0, 1.0)
        ta = (np.float32(tape["alpha"]) * cov)[..., None]
        rgb *= 1.0 - ta
        rgb += lit * ta
        a[...] = a + ta[..., 0] * (1.0 - a)


def print_tile(path, size, card, tone=None, toning=1.0, tape=None, corners=0, k=0, seed=0):
    """A mounted print as a tile: photograph `path` whole on its card, the card's longer side `size` px (its border and
    foot as `card` says), toned by `tone` (None: as it is), with `tape` strips (None: none) across the corners
    `corners`, print k's tape colour; one pin at the top middle. `seed` makes its card's grain."""
    pw0, ph0 = _photo(str(path)).shape[1], _photo(str(path)).shape[0]
    cw, ch, (bx, by) = PR.card_layout(pw0, ph0, card["border"], card["foot"])
    s = size / max(cw, ch)
    pw, ph = max(int(round(pw0 * s)), 1), max(int(round(ph0 * s)), 1)
    cw, ch, (bx, by) = PR.card_layout(pw, ph, card["border"], card["foot"])
    strips = [] if tape is None else PR.tape_strips(cw, ch, corners, tape["width"], tape["length"])
    pad = int(math.ceil(max([2.0] + [sl / 2.0 + sw for _, _, sw, sl in strips])))
    H, W = int(math.ceil(ch)) + 2 * pad, int(math.ceil(cw)) + 2 * pad
    rng = np.random.default_rng([seed, pw, ph, 1])
    cov = _rounded(cw, ch, CARD_CORNER * min(cw, ch), ox=pad, oy=pad, shape=(H, W))
    paint = _card_paint(cov, card["color"], min(cw, ch), rng)
    x0, y0 = int(round(pad + bx)), int(round(pad + by))
    win = np.zeros((H, W), np.float32)
    win[y0:y0 + ph, x0:x0 + pw] = 1.0
    paint *= 1.0 - _hairline(win, HAIRLINE)[..., None]
    paint[y0:y0 + ph, x0:x0 + pw] = photo_at(path, pw, ph, tone, toning)
    rgb = paint * cov[..., None]
    a = cov.copy()
    if strips:
        _tape(rgb, a, pad, strips, tape, k, rng)
    return Print(np.dstack([rgb, a]).astype(np.float32), cov, (pad + cw / 2.0 - 0.5, pad + ch / 2.0 - 0.5),
                 [(pad + cw / 2.0 - 0.5, y0 - 0.5)], (x0, y0, pw, ph))


def stereo_tile(halves, size, card, tone=None, toning=1.0, tape=None, corners=0, k=0, seed=0):
    """A stereocard as a tile: the two `halves` (files, left and right) side by side in windows with arched tops (their
    top corners rounded by `card` arch of their width) on a card `size` px wide and half as tall, its corners rounded;
    toned and taped as print_tile's; a pin above each window."""
    aspects = [_photo(str(p)).shape[1] / _photo(str(p)).shape[0] for p in halves]
    cw = float(size)
    ch, wins = PR.stereo_layout(cw, aspects)
    strips = [] if tape is None else PR.tape_strips(cw, ch, corners, tape["width"], tape["length"])
    pad = int(math.ceil(max([2.0] + [sl / 2.0 + sw for _, _, sw, sl in strips])))
    H, W = int(math.ceil(ch)) + 2 * pad, int(math.ceil(cw)) + 2 * pad
    rng = np.random.default_rng([seed, int(round(size)), 2])
    cov = _rounded(cw, ch, STEREO_CORNER * ch, ox=pad, oy=pad, shape=(H, W))
    paint = _card_paint(cov, card["color"], ch, rng)
    pins = []
    for p, (x, y, w, h) in zip(halves, wins):
        x0, y0 = int(round(pad + x)), int(round(pad + y))
        pw, ph = max(int(round(w)), 1), max(int(round(h)), 1)
        win = _rounded(pw, ph, card["arch"] * pw, top_only=True)
        full = np.zeros((H, W), np.float32)
        full[y0:y0 + ph, x0:x0 + pw] = win
        paint *= 1.0 - _hairline(full, HAIRLINE)[..., None]
        part = paint[y0:y0 + ph, x0:x0 + pw]
        part[...] = part * (1.0 - win[..., None]) + photo_at(p, pw, ph, tone, toning) * win[..., None]
        pins.append((x0 + pw / 2.0 - 0.5, pad + y * 0.5 - 0.5))
    rgb = paint * cov[..., None]
    a = cov.copy()
    if strips:
        _tape(rgb, a, pad, strips, tape, k, rng)
    x, y, w, h = wins[0]
    return Print(np.dstack([rgb, a]).astype(np.float32), cov, (pad + cw / 2.0 - 0.5, pad + ch / 2.0 - 0.5), pins,
                 (int(round(pad + x)), int(round(pad + y)), int(round(w)), int(round(h))))


@lru_cache(maxsize=32)
def pin_tile(colour, size):
    """A pushpin's head seen from above as a tile ((premultiplied rgb, coverage, its middle (x, y) px)): a disc `size`
    px across in `colour`, lit from the top left (a lighter crown and a glint, a darker rim)."""
    n = max(int(math.ceil(size)) + 2, 3)
    big = n * SUPER
    yy, xx = (np.mgrid[0:big, 0:big].astype(np.float32) + 0.5) / SUPER
    c = n / 2.0
    r = size / 2.0
    d = np.hypot(xx - c, yy - c)
    cov = np.clip((r - d) * SUPER + 0.5, 0.0, 1.0)
    col = np.asarray(colour, np.float32)
    lx, ly = (xx - c) / r, (yy - c) / r
    light = np.clip(1.0 - np.hypot(lx + 0.35, ly + 0.35), 0.0, 1.0) ** 1.5
    rim = np.clip(d / r, 0.0, 1.0) ** 4
    rgb = col * (1.0 - 0.35 * rim)[..., None] + (1.0 - col) * (0.45 * light)[..., None]
    glint = np.clip((0.16 * r - np.hypot(xx - (c - 0.36 * r), yy - (c - 0.36 * r))) * SUPER + 0.5, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.85 * glint)[..., None] + 0.85 * glint[..., None]
    small = lambda x: cv2.resize(np.ascontiguousarray(x, np.float32), (n, n), interpolation=cv2.INTER_AREA)  # noqa: E731
    a = np.clip(small(cov), 0.0, 1.0)
    return np.clip(small(rgb * cov[..., None]), 0.0, 1.0), a, ((n - 1) / 2.0, (n - 1) / 2.0)


# ================================================================================================= laying
def lay_print(out, t, M, shadow):
    """`out` (in place, and returned) with print tile `t` laid by the affine map `M` (mkmmd.mandala.placed): its drop
    `shadow` (colour, (dx, dy) px, blur px, alpha; None: none) cast from its card where it lands, the print over it."""
    FF = _ff()
    got = MDF.lay_box(t.cast.shape, M, shadow, (out.shape[1], out.shape[0]))
    if got is None:
        return out
    (x0, y0, x1, y1), Mt = got
    size = (x1 - x0, y1 - y0)
    region = out[y0:y1, x0:x1]
    laid = FF._warp(t.rgba, Mt, size)
    keep = 1.0 - np.clip(laid[..., 3], 0.0, 1.0)
    if shadow is not None and shadow[3] > 0:
        cast = np.clip(FF._warp(t.cast, Mt, size), 0.0, 1.0)
        k = np.float32(shadow[3]) * MDF.soft(FF.plate_shift(cast, shadow[1][0], shadow[1][1]), shadow[2])
        k *= keep                                    # the shadow under the print, the picture under both
        region *= (keep - k)[..., None]
        region += np.asarray(shadow[0], np.float32) * k[..., None]
    else:
        region *= keep[..., None]
    region += laid[..., :3]
    return out


def lay_pin(out, colour, size, at, scale, shadow=None):
    """`out` (in place) with a pushpin's head `size` px across in `colour` at pixel `at`, `scale` times its size (pushed
    in from larger), over its own small shadow in the colour `shadow` (None: none)."""
    rgb, a, anchor = pin_tile(tuple(float(c) for c in colour), float(size))
    t = Print(np.dstack([rgb, a]).astype(np.float32), a, anchor, [], None)
    dx, dy, blur, alpha = PIN_SHADOW
    cast = None if shadow is None else (shadow, (dx * size * scale, dy * size * scale), blur * size, alpha)
    M = np.array([[scale, 0.0, at[0] - scale * anchor[0]], [0.0, scale, at[1] - scale * anchor[1]]])
    return lay_print(out, t, M, cast)


# ================================================================================================= credits
def photo_refs(data):
    """The photographs of a project's prints (mk.toml `data`): every `photos` and `base` photo of its `[[fx]] kind =
    "prints"` entries (their aspect.<output> tables too), each once, as they are written (a tag as a list)."""
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

    def entry(e):
        if not isinstance(e, dict):
            return
        add(e.get("photos"))
        if isinstance(e.get("base"), dict):
            add(e["base"].get("photo"))
    fxs = data.get("fx") or []
    for e in [fxs] if isinstance(fxs, dict) else fxs:
        if isinstance(e, dict) and e.get("kind") == "prints":
            entry(e)
            for over in (e.get("aspect") or {}).values() if isinstance(e.get("aspect"), dict) else ():
                entry(over)
    return out


def photo_slugs(data, root, registry=None):
    """The registry entries of the photographs a project's prints show (photo_refs; mk.toml `data`, the project at
    `root`), each once: the slugs and tags they name and the entries whose path is a file or a glob's match; what finds
    nothing is left to mk post and mk look to report."""
    refs = photo_refs(data)
    if not refs:
        return []
    return list(dict.fromkeys(s for _, s in MDF._resolve(refs, root, MDF._registry(registry), False, "photos") if s))


# ================================================================================================= over a cut
# what a prints entry lays on a frame, in order: print k (-1 the base print), its tile, the affine map that lays it, its
# height (lift x h: its shadow grows with it) and its pins' scale (None: no pins, or not yet in)
Laid = namedtuple("Laid", "k tile M h pins")


class PrintsFx:
    """The prints' part of mkmmd.framefx.FrameFx (a mixin: it uses FrameFx's size, px, fps, frame0, fxs, _t, _at,
    _hit_times, _preload and the sprite finder the mandalas set up, _sprite_of). `_prints_init` is called once its
    effects are known, after the mandalas'."""

    def _prints_init(self):
        """The prints' photographs found now (and read, with FrameFx `preload`), a stereocard's other halves too (the
        asset registry's `extra.stereo`): a slug, a tag, a file or a glob that finds nothing, or a picture that cannot
        be read, is an FX.FxError before any frame is drawn."""
        self._pr_photos, self._pr_bases, self._pr_halves, self._pr_tiles = {}, {}, {}, OrderedDict()
        registry = []
        for fx in self.fxs:
            if fx["kind"] != "prints":
                continue
            what = f"fx {fx['index']} (prints)"
            try:
                files = [str(f) for f in self._sprite_of(fx["photos"])]
                base = None if fx["base"] is None else str(self._sprite_of([fx["base"]["photo"]])[0])
            except FX.FxError as e:
                raise FX.FxError(f"{what} photos: {e}") from None
            self._pr_photos[fx["index"]], self._pr_bases[fx["index"]] = files, base
            if fx["style"] == "stereo":
                if not registry:
                    from .assets import Registry
                    registry.append(Registry())
                for f in files:
                    if f not in self._pr_halves:
                        self._pr_halves[f] = stereo_halves(f, registry[0])
            if self._preload:
                for f in files + ([base] if base else []) + [h for f in files for h in self._pr_halves.get(f, ())]:
                    _photo(f)

    def _pr_tile(self, fx, path, size, k, corners, base=False):
        """The tile of print k (photograph `path` at `size` px, its tape across `corners`), or of the base print, kept
        from frame to frame by everything that draws it (its colours too: a breathing palette changes them)."""
        stereo = fx["style"] == "stereo" and not base
        tape = fx["tape"] if fx["fix"] == "tape" and not base else None
        tone = fx["base"]["tone"] if base else fx["tone"]
        key = (path, stereo, int(round(size)), tuple(fx["card"]["color"]), fx["card"]["border"], fx["card"]["foot"],
               fx["card"]["arch"], None if tone is None else tuple(map(tuple, tone)), fx["toning"],
               None if tape is None else (tuple(PR.pick(tape["colors"], k)), tape["alpha"], tape["width"],
                                          tape["length"], corners), fx["seed"])
        got = self._pr_tiles.get(key)
        if got is None:
            args = (fx["card"], tone, fx["toning"], tape, corners, k, fx["seed"])
            got = stereo_tile(self._pr_halves[path], size, *args) if stereo else print_tile(path, size, *args)
            self._pr_tiles[key] = got
            while len(self._pr_tiles) > TILES:
                self._pr_tiles.popitem(last=False)
        else:
            self._pr_tiles.move_to_end(key)
        return got

    def _pr_shadow(self, fx, h):
        """The drop shadow of a print `h` (lift x height) in the air: (colour, (dx, dy) px, blur px, alpha), its offset
        and blur larger and its opacity lower the higher it is; None without a shadow."""
        s = fx["shadow"]
        if s is None:
            return None
        H = self.size[1]
        grow, soften, fade = LIFT_SHADOW
        return (s["color"], (s["offset"][0] * H * (1.0 + grow * h), s["offset"][1] * H * (1.0 + grow * h)),
                s["blur"] * self.px * (1.0 + soften * h), s["alpha"] / (1.0 + fade * h))

    def prints_at(self, fx, frame):
        """What a prints entry lays on `frame`, in order ([Laid]): a pin's base print first, then every print whose
        fall has begun in the order they were dealt, each where its mode puts its place, slid and spun by what is left
        of its fall (mkmmd.core.prints.fall), as large as its height above the page makes it."""
        t = self._t(frame)
        unit = float(min(self.size))
        w, H = self.size
        files = self._pr_photos[fx["index"]]
        lands = PR.landings(fx, self._hit_times(fx) if fx.get("lane") is not None or fx.get("times") else None)
        out = []
        if fx["mode"] == "pin":
            b = fx["base"]
            tile = self._pr_tile(fx, self._pr_bases[fx["index"]], b["size"] * unit, -1, 0, base=True)
            f = PR.fall(fx, t, -math.inf if b["t"] is None else b["t"])
            if f is None:
                return out
            M0 = MDF.placed(tile, (b["at"][0] * w - 0.5, b["at"][1] * H - 0.5), b["angle"] + 0.25 * fx["spin"] * f[1],
                            1.0 + fx["lift"] * f[0])
            out.append(Laid(-1, tile, M0, fx["lift"] * f[0], None))
            x0, y0, pw, ph = tile.photo
            hw, hh = PR.half_extent(fx)
            places = [_apply(M0, (x0 + u * pw - 0.5, y0 + v * ph - 0.5))
                      for u, v in PR.pin_places(fx, pw / float(ph), (hw * unit / pw, hh * unit / ph))]
        elif fx["mode"] == "scatter":
            places = [(x * unit - 0.5, y * unit - 0.5) for x, y in PR.scatter_places(fx, w / float(H))]
        else:
            c = self._at(fx, frame)[0]
            places = [(c[0] + dx * unit, c[1] + dy * unit) for dx, dy in
                      (PR.ring_offset(fx, j, t) for j in range(fx["count"]))]
        turn = fx["base"]["angle"] if fx["mode"] == "pin" else 0.0
        for k in PR.shown(fx, t, lands):
            f = PR.fall(fx, t, lands[k])
            if f is None:
                continue
            h, r = f
            tilt, spin, (ox, oy), corners = PR.jitter(fx, k)
            at = places[k % fx["count"]]
            tile = self._pr_tile(fx, files[PR.photo_pick(fx, k, len(files))], fx["size"] * unit, k, corners)
            M = MDF.placed(tile, (at[0] + ox * unit * r, at[1] + oy * unit * r), turn + tilt + spin * r,
                           1.0 + fx["lift"] * h)
            out.append(Laid(k, tile, M, fx["lift"] * h, PR.pin_push(t, lands[k]) if fx["fix"] == "pins" else None))
        return out

    def _prints(self, fx, frame, img):
        s = float(np.clip(FX.strength(fx, self._t(frame)), 0.0, 1.0))
        if s <= 0.0:
            return img
        laid = self.prints_at(fx, frame)
        if not laid:
            return img
        out = np.array(img, np.float32, copy=True)
        pin = fx["pins"]["size"] * float(min(self.size))
        for p in laid:
            lay_print(out, p.tile, p.M, self._pr_shadow(fx, p.h))
            for q in p.tile.pins if p.pins is not None else ():
                lay_pin(out, PR.pick(fx["pins"]["colors"], p.k), pin, _apply(p.M, q), p.pins,
                        None if fx["shadow"] is None else fx["shadow"]["color"])
        return out if s >= 1.0 else (img * (1.0 - s) + out * s).astype(np.float32)


def _apply(M, p):
    """Pixel `p` carried by the affine map `M` (2 x 3)."""
    return (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2], M[1][0] * p[0] + M[1][1] * p[1] + M[1][2])
