"""Colour bars (docs/design.md: Era looks): a TV's test colours as pictures, numpy only (it runs inside Blender too).

`texture(w, h, palette, style)` is the face of a colour-bar card: the bars as SMPTE lays them out in true colours
("true"), the same layout in the palette's colours ("dawn": its accents brightest first, as true bars run), or the
palette's bars over a row of the true ones ("both"). `back(w, h, palette)` is a card's back. The `testcard` and `cards`
frame effects (mkmmd.core.eralook) draw with them and the `library:colorbar_card` prop lays them on its faces. A palette
is {slot: "#hex"} (mkmmd.core.palette; None: rose-pine-dawn); a picture is (h, w, 3) float32 display-space RGB 0..1,
row 0 its top."""
import numpy as np

from . import palette as PAL

STYLES = ("dawn", "true", "both")
TRUE = ((0.75, 0.75, 0.75), (0.75, 0.75, 0.0), (0.0, 0.75, 0.75), (0.0, 0.75, 0.0), (0.75, 0.0, 0.75),
        (0.75, 0.0, 0.0), (0.0, 0.0, 0.75))           # 75 % bars: grey, yellow, cyan, green, magenta, red, blue
MINUS_I, PLUS_Q = (0.0, 0.129, 0.298), (0.196, 0.0, 0.416)   # the I and Q chroma patches of SMPTE's bottom row
SETUP = 0.075                                         # black at 7.5 IRE; the pluge's bars 4 below it and 4 above
ACCENTS = ("gold", "rose", "foam", "iris", "love", "pine")
NEUTRALS = ("base", "surface", "overlay", "muted", "subtle", "text", "hl_low", "hl_med", "hl_high")
BOTTOM = (1.25, 1.25, 1.25, 1.25, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0)   # the bottom row's patches, in bar widths
SPLIT = {"true": (0.67, 0.75), "dawn": (0.67, 0.75), "both": (0.5, 0.75)}   # where the middle and bottom rows start


def _palette(palette):
    return PAL.get("rose-pine-dawn") if palette is None else palette


def luminance(rgb):
    """Relative luminance of a display-space colour (sRGB decoded, Rec. 709 weights)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def colours(palette=None, style="dawn"):
    """The seven bars of the top row, brightest first: the palette's lightest neutral and its accents by luminance
    ("dawn"), or the true 75 % bars ("true")."""
    if style == "true":
        return [tuple(c) for c in TRUE]
    pal = _palette(palette)
    light = max((PAL.srgb(pal[s]) for s in NEUTRALS if s in pal), key=luminance)
    accents = sorted((PAL.srgb(pal[s]) for s in ACCENTS if s in pal), key=luminance, reverse=True)
    return [light] + accents


def ends(palette=None):
    """(lightest, darkest) neutral of the palette: its white and its black."""
    pal = _palette(palette)
    ns = [PAL.srgb(pal[s]) for s in NEUTRALS if s in pal]
    return max(ns, key=luminance), min(ns, key=luminance)


def _row(out, y0, y1, cols, widths=None):
    """Fill rows y0..y1 of `out` with vertical bars of `cols`, as wide as `widths` say (equal without), edges on
    whole pixels."""
    w = out.shape[1]
    widths = [1.0] * len(cols) if widths is None else list(widths)
    edges = np.rint(np.concatenate([[0.0], np.cumsum(widths)]) / sum(widths) * w).astype(int)
    for c, a, b in zip(cols, edges[:-1], edges[1:]):
        out[y0:y1, a:b] = c


def _mix(a, b, t):
    return tuple(x * (1.0 - t) + y * t for x, y in zip(a, b))


def texture(width, height, palette=None, style="both"):
    """The face of a colour-bar card, (height, width, 3) float32: seven bars brightest first over two thirds of it,
    under them a strip of the castellation (every other bar of the top row backwards, black between: SMPTE's) and at
    the foot the chroma patches, a white and the pluge (black and a bar just under and just over it). "dawn" in the
    palette's colours, "true" in true ones, "both" the palette's bars over a row of the true ones and the true foot."""
    if style not in STYLES:
        raise ValueError(f"colour bars style {style!r} is none of {', '.join(STYLES)}")
    w, h = int(width), int(height)
    if w < 7 or h < 4:
        raise ValueError(f"colour bars of {w} x {h} px: at least 7 x 4")
    out = np.zeros((h, w, 3), np.float32)
    mid, foot = (int(round(f * h)) for f in SPLIT[style])
    white, black = ends(palette)
    top = colours(palette, "true" if style == "true" else "dawn")
    _row(out, 0, mid, top)
    if style == "both":
        _row(out, mid, foot, colours(palette, "true"))
    else:
        castle = [top[6], black, top[4], black, top[2], black, top[0]] if style == "dawn" else \
            [TRUE[6], (SETUP,) * 3, TRUE[4], (SETUP,) * 3, TRUE[2], (SETUP,) * 3, TRUE[0]]
        _row(out, mid, foot, castle)
    if style == "dawn":
        pal = _palette(palette)
        deep = _mix(PAL.srgb(pal["pine"]), black, 0.45) if "pine" in pal else MINUS_I
        violet = _mix(PAL.srgb(pal["iris"]), black, 0.45) if "iris" in pal else PLUS_Q
        lo, hi = _mix(black, white, -0.04 / 0.9), _mix(black, white, 0.04)
        bottom = [deep, white, violet, black, tuple(max(c, 0.0) for c in lo), black, hi, black]
    else:
        bottom = [MINUS_I, (1.0, 1.0, 1.0), PLUS_Q, (SETUP,) * 3, (SETUP - 0.04,) * 3, (SETUP,) * 3,
                  (SETUP + 0.04,) * 3, (SETUP,) * 3]
    _row(out, foot, h, bottom, BOTTOM)
    return out


def _ring(xx, yy, cx, cy, r, half):
    """Coverage 0..1 of a ring of radius r about (cx, cy), `half` px either side of it, antialiased."""
    d = np.abs(np.hypot(xx - cx, yy - cy) - r)
    return np.clip(half - d + 0.5, 0.0, 1.0)


def _lines(n, period, half, offset=0.0):
    """Coverage 0..1 along a side of `n` px of lines every `period` px, `half` px either side of each, antialiased."""
    x = np.arange(n, dtype=np.float32) + 0.5 - offset
    d = np.abs((x + period / 2.0) % period - period / 2.0)
    return np.clip(half - d + 0.5, 0.0, 1.0)


def back(width, height, palette=None):
    """The back of a colour-bar card, (height, width, 3) float32: the palette's lightest neutral ruled in a fine grid
    of squares (nine down), a ring in the middle and the top row's accents along its foot."""
    w, h = int(width), int(height)
    if w < 7 or h < 4:
        raise ValueError(f"a card back of {w} x {h} px: at least 7 x 4")
    pal = _palette(palette)
    white, black = ends(palette)
    rule = PAL.srgb(pal["hl_med"]) if "hl_med" in pal else _mix(white, black, 0.15)
    out = np.empty((h, w, 3), np.float32)
    out[:] = white
    period, half = h / 9.0, max(h / 600.0, 0.5)
    gx, gy = _lines(w, period, half, (w / 2.0) % period), _lines(h, period, half, (h / 2.0) % period)
    grid = np.maximum(gx[None, :], gy[:, None])[..., None]
    out = out * (1.0 - grid) + np.asarray(rule, np.float32) * grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    ring = _ring(xx, yy, w / 2.0, h / 2.0, 0.3 * h, max(h / 90.0, 0.75))[..., None]
    out = out * (1.0 - ring) + np.asarray(_mix(black, white, 0.25), np.float32) * ring
    foot = max(int(round(h / 14.0)), 1)
    _row(out, h - foot, h, colours(palette, "dawn")[1:])
    return np.clip(out, 0.0, 1.0).astype(np.float32)
