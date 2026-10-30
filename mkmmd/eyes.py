"""Googly eyes drawn on sprites (docs/design.md: Frame effects, Mandalas and trails, Googly eyes): the picture work of
mkmmd.core.eyes. A sprite tile with a face (mkmmd.mandala.sprite_tile(..., face=)) has its eyes' whites (a dark rim, a
white dome shading grey toward its lower right) and its smile painted in, and carries the face's geometry in the tile's
px (`Tile.face`); the pupils, black discs swinging inside the whites, and the dome's glint over them are drawn on every
frame where the tile is laid (`draw_pupils`, which mkmmd.mandala.lay calls) or into a copy of the tile (`with_pupils`).
`pupils` works out where they are at a clip second for one copy of a sprite, from how it moves and the hits.

Every edge is antialiased analytically (a pixel's coverage is how far inside the shape its middle is, clipped to one
pixel), so eyes a few pixels across still read. Images are float RGB in display space (0..1); a tile's colour is
premultiplied by its coverage."""
import math

import numpy as np

from .core import eyes as EY
from .core import framefx as FX
from .core import lanes as LN

INK = (0.055, 0.05, 0.065)          # the pupils and the smile
RIM_INK = (0.12, 0.11, 0.13)        # the line round an eye
WHITE = (1.0, 1.0, 1.0)
SHADE = (0.78, 0.79, 0.85)          # the dome's grey toward its lower right
RIM = 0.1                           # the rim line's width, shares of the eye's radius
PUPIL = 0.52                        # a pupil's radius, shares of the white inside the rim
MARGIN = 0.03                       # room kept between a pupil and the rim, shares of the eye's radius
GLINT = ((-0.34, -0.4), (0.19, 0.12), -38.0, 0.9)   # the dome's highlight: middle and radii (eye radii), angle, opacity
MOUTH = {"half": 0.62, "sag": 0.34, "line": 0.17}  # the smile: half its width, how far its middle dips, its line (eye radii)
MIN_LINE = 1.1                      # px: the thinnest line drawn

_ANCHORS = {}                       # a sprite's file (resolved) -> its registry entry's extra.eyes (None: auto)
_FACES = {}                         # a sprite's file -> its anchors, worked out once


# ================================================================================================= anchors
def remember(path, extra_eyes):
    """Note a registry sprite's `extra.eyes` (or None) for its file (mkmmd.mandala's sprite lookup calls this for every
    entry it resolves), so its tile's face sits where its entry says."""
    p = str(path)
    if _ANCHORS.get(p, 0) != extra_eyes:
        _ANCHORS[p] = extra_eyes
        _FACES.pop(p, None)


def anchors_for(path, alpha):
    """The face anchors of the sprite at `path` (mkmmd.core.eyes.anchors: its registry entry's extra.eyes over the auto
    face of its coverage `alpha`, (h, w) trimmed to it), worked out once per file."""
    p = str(path)
    got = _FACES.get(p)
    if got is None:
        got = _FACES[p] = EY.anchors(_ANCHORS.get(p), alpha, f"sprite {p}")
    return got


# ================================================================================================= coverage
def _box(shape, cx, cy, reach):
    h, w = shape
    x0, y0 = max(int(math.floor(cx - reach)), 0), max(int(math.floor(cy - reach)), 0)
    x1, y1 = min(int(math.ceil(cx + reach)) + 1, w), min(int(math.ceil(cy + reach)) + 1, h)
    return x0, y0, x1, y1


def _grid(x0, y0, x1, y1):
    return np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))


def disc(shape, cx, cy, r):
    """((x0, y0, x1, y1) box, coverage there) of a disc of radius `r` px about (cx, cy) (pixel middles at whole
    numbers); None when it misses the picture."""
    b = _box(shape, cx, cy, r + 1.0)
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    x, y = _grid(*b)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return b, np.clip(r + 0.5 - d, 0.0, 1.0)


def ellipse(shape, cx, cy, rx, ry, angle):
    """As `disc`, an ellipse of radii `rx`, `ry` turned `angle` degrees counter-clockwise on screen."""
    b = _box(shape, cx, cy, max(rx, ry) + 1.0)
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    x, y = _grid(*b)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    u, v = (x - cx) * c - (y - cy) * s, (x - cx) * s + (y - cy) * c
    q = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    return b, np.clip((1.0 - q) * min(rx, ry) + 0.5, 0.0, 1.0)


def stroke(shape, pts, width):
    """As `disc`, a line `width` px wide along the polyline `pts` ((n, 2) px), its ends round."""
    pts = np.asarray(pts, np.float32)
    lo, hi = pts.min(0), pts.max(0)
    half = 0.5 * width
    c = 0.5 * (lo + hi)
    b = _box(shape, c[0], c[1], 0.5 * float(np.max(hi - lo)) + half + 1.0)
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    x, y = _grid(*b)
    q = np.stack([x.ravel(), y.ravel()], 1)
    a, d = pts[:-1], pts[1:] - pts[:-1]
    t = np.clip(((q[:, None, :] - a[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-9)[None], 0.0, 1.0)
    dist = np.sqrt((((a[None] + t[..., None] * d[None]) - q[:, None, :]) ** 2).sum(-1)).min(1)
    return b, np.clip(half + 0.5 - dist, 0.0, 1.0).reshape(x.shape)


def _over(rgb, a, got, colour, alpha=1.0):
    """`rgb` (premultiplied) and coverage `a` (or None: an opaque picture) with `colour` laid over them where `got`
    (disc() and co.) covers, at `alpha`; in place."""
    if got is None:
        return
    (x0, y0, x1, y1), c = got
    c = (c * np.float32(alpha)).astype(np.float32)
    col = np.asarray(colour, np.float32)
    if col.ndim == 3:
        col = col[y0:y1, x0:x1]
    region = rgb[y0:y1, x0:x1]
    region *= (1.0 - c)[..., None]
    region += col * c[..., None]
    if a is not None:
        ar = a[y0:y1, x0:x1]
        ar *= 1.0 - c
        ar += c


# ================================================================================================= the face on a tile
def geometry(face, anchors, w, h):
    """A face's geometry in px of a sprite drawn `w` x `h` px (its box), from its `anchors` (fractions) and the face's
    `size`: {eyes (the two centres or None), R (an eye's radius), inner (inside its rim), pupil, free (how far a pupil's
    centre may stray), mouth (the smile's middle or None), rest (where the pupils rest unless told)}."""
    R = anchors["r"] * max(w, h) * face["size"]
    inner = R * (1.0 - RIM)
    pupil = PUPIL * inner
    eyes = [[x * w - 0.5, y * h - 0.5] for x, y in anchors["at"]] if face["eyes"] else None
    mo = anchors.get("mouth") if face["mouth"] else None
    if mo is not None and eyes is None:
        mo = [mo[0] * w - 0.5, mo[1] * h - 0.5]
    elif mo is not None:                 # the smile keeps its place under the eyes when they grow
        ex, ey = (anchors["at"][0][0] + anchors["at"][1][0]) * 0.5 * w - 0.5, anchors["at"][0][1] * h - 0.5
        base = anchors["r"] * max(w, h)
        mo = [ex + (mo[0] * w - 0.5 - ex), ey + (mo[1] * h - 0.5 - ey) * (R / base if base > 0 else 1.0)]
    return {"eyes": eyes, "R": R, "inner": inner, "pupil": pupil, "free": max(inner - pupil - MARGIN * R, 0.0),
            "mouth": mo, "rest": [0.0, 0.0] if face["look"] == "viewer" else [0.0, EY.SAG]}


def smile(geo):
    """The smile's line ((n, 2) px) and its width: a doodled arc whose middle is the mouth's anchor, its ends `sag`
    higher, the right one a touch higher still."""
    R, (mx, my) = geo["R"], geo["mouth"]
    half, sag = MOUTH["half"] * R, MOUTH["sag"] * R
    u = np.linspace(0.0, 1.0, 24)[:, None]
    p0, p2 = np.array([mx - half, my - sag]), np.array([mx + half, my - 1.15 * sag])
    c = np.array([mx - 0.05 * half, my + 1.075 * sag])         # a quadratic's middle: half the control, a quarter each end
    return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * c + u ** 2 * p2, max(MOUTH["line"] * R, MIN_LINE)


def _reach(geo):
    """How far the face reaches round its parts: (x0, y0, x1, y1) px (None: nothing to draw)."""
    boxes = []
    if geo["eyes"] is not None:
        boxes += [(x - geo["R"], y - geo["R"], x + geo["R"], y + geo["R"]) for x, y in geo["eyes"]]
    if geo["mouth"] is not None:
        pts, wd = smile(geo)
        boxes.append((pts[:, 0].min() - wd, pts[:, 1].min() - wd, pts[:, 0].max() + wd, pts[:, 1].max() + wd))
    if not boxes:
        return None
    b = np.asarray(boxes)
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


def paint(rgb, a, anchor, face, anchors):
    """A sprite's picture (`rgb` premultiplied (h, w, 3), coverage `a`, its `anchor` (x, y) px) with `face`'s whites and
    smile painted on where `anchors` say, grown to hold them where they stick out: (rgb, a, anchor, geometry), the
    geometry (`geometry`) in the grown picture's px."""
    h, w = a.shape
    geo = geometry(face, anchors, w, h)
    reach = _reach(geo)
    if reach is None:
        return rgb, a, anchor, geo
    left, top = max(int(math.ceil(-reach[0])) + 1, 0), max(int(math.ceil(-reach[1])) + 1, 0)
    right, bottom = max(int(math.ceil(reach[2] - (w - 1))) + 1, 0), max(int(math.ceil(reach[3] - (h - 1))) + 1, 0)
    if left or top or right or bottom:
        rgb = np.pad(rgb, ((top, bottom), (left, right), (0, 0)))
        a = np.pad(a, ((top, bottom), (left, right)))
        anchor = (anchor[0] + left, anchor[1] + top)
        geo = shifted(geo, left, top)
    rgb, a = np.array(rgb, np.float32), np.array(a, np.float32)
    shape = a.shape
    if geo["eyes"] is not None:
        for x, y in geo["eyes"]:
            _over(rgb, a, disc(shape, x, y, geo["R"]), RIM_INK)
            got = disc(shape, x, y, geo["inner"])
            if got is not None:
                (x0, y0, x1, y1), _ = got
                gx, gy = _grid(x0, y0, x1, y1)
                d = np.sqrt((gx - (x - 0.3 * geo["inner"])) ** 2 + (gy - (y - 0.35 * geo["inner"])) ** 2)
                k = np.clip((d / max(geo["inner"], 1e-6) - 0.55) / 0.75, 0.0, 1.0) ** 2
                col = np.asarray(WHITE, np.float32) + (np.asarray(SHADE, np.float32) - np.asarray(WHITE, np.float32)) \
                    * k[..., None].astype(np.float32)
                full = np.zeros(shape + (3,), np.float32)
                full[y0:y1, x0:x1] = col
                _over(rgb, a, got, full)
    if geo["mouth"] is not None:
        pts, wd = smile(geo)
        _over(rgb, a, stroke(shape, pts, wd), INK)
    return rgb, a, anchor, geo


def shifted(geo, dx, dy):
    """A face geometry moved `dx`, `dy` px (its picture padded)."""
    out = dict(geo)
    if geo["eyes"] is not None:
        out["eyes"] = [[x + dx, y + dy] for x, y in geo["eyes"]]
    if geo["mouth"] is not None:
        out["mouth"] = [geo["mouth"][0] + dx, geo["mouth"][1] + dy]
    return out


# ================================================================================================= the pupils
def _pupil_layers(shape, centre, s, geo, off):
    """[(coverage, colour, opacity)] of one eye's pupil at offset `off` (free radii, screen axes) and the glint over it,
    the eye's middle at `centre` px, the tile laid `s` times its size."""
    cx, cy = centre
    px, py = cx + off[0] * geo["free"] * s, cy + off[1] * geo["free"] * s
    (gx, gy), (rx, ry), ang, alpha = GLINT
    R = geo["inner"] * s
    return [(disc(shape, px, py, geo["pupil"] * s), INK, 1.0),
            (ellipse(shape, cx + gx * R, cy + gy * R, max(rx * R, 0.6), max(ry * R, 0.45), ang), WHITE, alpha)]


def draw_pupils(out, geo, M, offsets=None):
    """`out` (an opaque picture, in place, and returned) with the pupils and glints of a tile with face `geo` laid by
    the affine map `M` (mkmmd.mandala.placed): `offsets` (2, 2) free radii, screen axes (`pupils`; None: at rest)."""
    if geo is None or geo["eyes"] is None:
        return out
    M = np.asarray(M, np.float64)
    s = math.sqrt(abs(float(np.linalg.det(M[:, :2]))))
    offs = np.tile(geo["rest"], (2, 1)) if offsets is None else np.asarray(offsets, np.float64)
    for e, (x, y) in enumerate(geo["eyes"]):
        c = M[:, :2] @ np.array([x, y]) + M[:, 2]
        for got, colour, alpha in _pupil_layers(out.shape[:2], c, s, geo, offs[e]):
            _over(out, None, got, colour, alpha)
    return out


def with_pupils(tile, offsets=None):
    """A copy of sticker `tile` (mkmmd.mandala.Tile) with its pupils and glints drawn in at `offsets` ((2, 2) free radii,
    the tile's axes; None: at rest); the tile itself when it has no eyes."""
    geo = getattr(tile, "face", None)
    if geo is None or geo["eyes"] is None:
        return tile
    rgb = np.array(tile.rgb, np.float32, copy=True)
    offs = np.tile(geo["rest"], (2, 1)) if offsets is None else np.asarray(offsets, np.float64)
    for e, c in enumerate(geo["eyes"]):
        for got, colour, alpha in _pupil_layers(rgb.shape[:2], c, 1.0, geo, offs[e]):
            _over(rgb, None, got, colour, alpha)
    return tile._replace(rgb=rgb)


def hit_times(face, own, hits):
    """The clip seconds that kick a face's pupils: its own `times`, its own `lane` picked from `hits` (the [audio] hits
    file, {lane: [...]}), else `own` (the hits of what shows it). An FX.FxError when its lane cannot be read."""
    if face.get("times") is not None:
        return list(face["times"])
    if face.get("lane") is None:
        return list(own or [])
    if hits is None:
        raise FX.FxError(f"eyes: lane {face['lane']!r} needs the [audio] hits file (`mk timeline hits` writes it)")
    try:
        return LN.pick(hits, face["lane"], face["min_db"], face["every"])
    except LN.LaneError as e:
        raise FX.FxError(f"eyes: {e}") from None


def look_at(face, size, centre=None):
    """Where a face's pupils look in px of a picture of `size` (w, h): None (they fall, the default), "viewer" (straight
    out) or (x, y): its [x, y] frame fractions, or for "center" `centre` (what shows it: a mandala's centre; default the
    picture's middle)."""
    look = face["look"]
    if look == "down":
        return None
    if look == "viewer":
        return "viewer"
    w, h = size
    if look == "center":
        return tuple(centre) if centre is not None else (0.5 * w - 0.5, 0.5 * h - 0.5)
    return (look[0] * w - 0.5, look[1] * h - 0.5)


def pupils(face, tile, t, hits=(), key=(), maps=None, look=None):
    """The pupil offsets ((2, 2) free radii, screen axes) of one copy of a sprite tile with eyes at clip second `t`, to
    lay it with (mkmmd.mandala.lay(..., pupils=)): `maps(ts)` gives its affine maps ((n, 2, 3), tile px to the picture's,
    as mkmmd.mandala.placed makes them) at the clip seconds `ts` (None: it stands still), `hits` the clip seconds that
    kick them (`hit_times`), `key` whole numbers naming the copy (its kicks are seeded by them and the face's `seed`),
    `look` where they look (`look_at`: None, "viewer" or (x, y) px of the picture, or of the tile without `maps`). None
    for a tile without eyes."""
    geo = getattr(tile, "face", None)
    if geo is None or geo["eyes"] is None or face is None or not face["eyes"]:
        return None
    ts = EY.window(face, t)
    E = np.asarray(geo["eyes"], np.float64)
    if maps is None:
        centres = np.broadcast_to(E, (len(ts), 2, 2))
        free = np.full((len(ts), 2), geo["free"])
    else:
        Ms = np.asarray(maps(ts), np.float64).reshape(len(ts), 2, 3)
        centres = np.einsum("nij,ej->nei", Ms[:, :, :2], E) + Ms[:, None, :, 2]
        s = np.sqrt(np.abs(np.linalg.det(Ms[:, :, :2])))
        free = geo["free"] * np.maximum(s, 0.3 * max(float(s.max()), 1e-6))[:, None] * np.ones((1, 2))
    keys = [list(key) + [0], list(key) + [1]]
    return EY.swing(face, ts, centres, free, hits, keys, look)
