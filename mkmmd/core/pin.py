"""Pinned stickers, bpy-free (docs/design.md: Pinned stickers): a table's live figure (her sticker from a plate) pinned
to a point of the clip under it, a seat that the clip's own camera moves, with contact shadows on the seat and the pieces
of the clip that stand in front of her. mkmmd.core.table checks a table's `pin` with `normalize`; the maths is here
(numpy, tested), mkmmd/pin.py draws it and `python -m mkmmd.pintrack` measures a seat's track in a clip.

    track   the seat's motion in the clip: rows [video second, x, y] or [video second, x, y, scale, turn] (x, y fractions
            of the clip's frame as the file has it, 0, 0 its top left; scale and turn the size and the turn, degrees
            counter-clockwise on screen, of the picture about that point against a fixed frame of the clip), straight
            between rows. The figure is placed as on the span's first frame (`at` and `height` about its pivot, or its
            `track`, as any table's figure) and on every frame carried by the similarity that takes the point and the
            picture about it from the clip frame shown on that first frame to the clip frame shown now (`carry`): she
            moves, grows and turns with her seat and steps with the clip's own frames, never between them
    shadow  contact shadows: soft ellipses on the seat under her (`at` table fractions and `size` [w, h] table heights as
            on the span's first frame, `turn` degrees), the clip mixed toward `color` by `alpha` (a dark colour shades it,
            a light one lifts a seat too dark for a shadow; a number or keys over clip seconds: dark from a landing on),
            fading out from the middle over `soft` of the radius, cut to an `on` outline (the seat's surface: no shadow on
            the sky past its edge), carried with the seat, under her sticker
    front   the pieces of the clip in front of her: outlines (table fractions as on the span's first frame) inside which
            the clip as the table drew it is laid again over her and her rim, the outline's edge feathered `soft` px at
            1080; a `key` takes only the pixels whose luma lies in its range (a dark wire on a light sky), so a thin
            thing needs no tight outline
"""
import math

import numpy as np

KEYS = ("pin",)
SHADOW = {"at": None, "size": None, "turn": 0.0, "alpha": 0.45, "color": "text", "soft": 1.0, "on": None}
FRONT = {"outline": None, "key": None, "soft": 1.0}
KEY = {"luma": None, "soft": 0.04}
EPS = 1e-6             # seconds: a clip frame's second on a track's end row is inside it


def _table(FX, v, defaults, what, name, required=()):
    known = sorted(set(defaults) | set(required))
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table ({{{', '.join(known)}}})")
    extra = sorted(set(v) - set(known))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {known})")
    lack = [k for k in required if v.get(k) is None]
    if lack:
        raise FX.FxError(f"{what}: {name} needs {', '.join(f'`{k}`' for k in lack)}")
    return {**defaults, **v}


def _list(v):
    return [] if v is None else [v] if isinstance(v, dict) else list(v)


# ================================================================================================= the entry
def normalize(FX, v, spec, palette, what):
    """A table's `pin` checked and filled in (None without one): `{track, shadow, front}`, `track` required and its
    figure's `objects` given (the pin carries the figure sticker)."""
    if v is None:
        return None
    p = _table(FX, v, {"shadow": None, "front": None}, what, "pin", required=("track",))
    if spec.get("objects") is None:
        raise FX.FxError(f"{what}: pin carries the figure sticker: give its `objects`")
    w = f"{what} pin"
    return {"track": track(FX, p["track"], f"{w} track"),
            "shadow": [_shadow(FX, s, palette, f"{w} shadow[{i}]") for i, s in enumerate(_list(p["shadow"]))],
            "front": [_front(FX, f, f"{w} front[{i}]") for i, f in enumerate(_list(p["front"]))]}


def track(FX, v, what):
    """A seat's track: rows [video second, x, y] or [video second, x, y, scale, turn], two or more, all of one length,
    their seconds rising; the short rows read as scale 1, turn 0. Returned as five-column rows."""
    if not (isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(r, (list, tuple)) for r in v)):
        raise FX.FxError(f"{what} = {v!r}: [[video second, x, y, scale, turn], ...] (or [[video second, x, y], ...]), "
                         f"two rows or more")
    n = len(v[0])
    if n not in (3, 5) or any(len(r) != n for r in v):
        raise FX.FxError(f"{what}: every row is [video second, x, y] or every row [video second, x, y, scale, turn]")
    rows = [[FX._num(c, what) for c in r] + ([1.0, 0.0] if n == 3 else []) for r in v]
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise FX.FxError(f"{what}: a track's video seconds rise from row to row")
    if any(r[3] <= 0.0 for r in rows):
        raise FX.FxError(f"{what}: a row's scale is positive (the picture's size against the track's own frame)")
    return rows


def _outline(FX, v, what):
    if not (isinstance(v, (list, tuple)) and len(v) >= 3):
        raise FX.FxError(f"{what} = {v!r}: an outline is [[x, y], ...], three points or more (table fractions)")
    return [FX._pair(q, what) for q in v]


def _shadow(FX, v, palette, what):
    s = _table(FX, v, SHADOW, what, "shadow", required=("at", "size"))
    size = FX._pair(s["size"], f"{what} size")
    if min(size) <= 0.0:
        raise FX.FxError(f"{what}: size = {s['size']!r}: [width, height] table heights, both positive")
    alpha = FX._keys(s["alpha"], f"{what} alpha")
    if any(not 0.0 <= a <= 1.0 for _, a in alpha):
        raise FX.FxError(f"{what}: alpha = {s['alpha']!r}: 0..1, a number or [[t, alpha], ...]")
    return {"at": FX._pair(s["at"], f"{what} at"), "size": size, "turn": FX._num(s["turn"], f"{what} turn"),
            "alpha": alpha, "color": FX._colour(s["color"], palette, f"{what} color"),
            "soft": FX._num(s["soft"], f"{what} soft", 0.05, 1.0),
            "on": None if s["on"] is None else _outline(FX, s["on"], f"{what} on")}


def _front(FX, v, what):
    f = _table(FX, v, FRONT, what, "front", required=("outline",))
    key = None
    if f["key"] is not None:
        k = _table(FX, f["key"], KEY, f"{what}", "key", required=("luma",))
        lo, hi = FX._pair(k["luma"], f"{what} key luma")
        if not 0.0 <= lo < hi <= 1.0:
            raise FX.FxError(f"{what}: key luma = {k['luma']!r}: [low, high], 0 <= low < high <= 1")
        key = {"luma": [lo, hi], "soft": FX._num(k["soft"], f"{what} key soft", 0.0, 0.5)}
    return {"outline": _outline(FX, f["outline"], f"{what} outline"), "key": key,
            "soft": FX._num(f["soft"], f"{what} soft", 0.0, 40.0)}


# ================================================================================================= the maths
def row_at(rows, s):
    """(x, y, scale, turn) of a track at video second `s`: straight between rows, the end rows' beyond them."""
    r = np.asarray(rows, float)
    return tuple(float(np.interp(s, r[:, 0], r[:, k])) for k in range(1, 5))


def covers(rows, s0, s1):
    """Whether a track's rows reach over the video seconds `s0` .. `s1` (the clip frames a span shows)."""
    return rows[0][0] <= s0 + EPS and rows[-1][0] >= s1 - EPS


def similarity(p, z, a, q):
    """The 2 x 3 map that puts px `q` at px `p`, zoomed `z` and turned `a` radians counter-clockwise on screen about it
    (mkmmd.core.table.similarity's)."""
    c, s = math.cos(a), math.sin(a)
    A = z * np.array([[c, s], [-s, c]])
    return np.hstack([A, (np.asarray(p, float) - A @ np.asarray(q, float))[:, None]])


def carry(rows, s0, s, table):
    """The 2 x 3 map (table px -> table px, a clip of `table` (w, h) px) that carries what lies on the seat from the clip
    frame at video second `s0` to the one at `s`: the tracked point there to the point here, the picture about it
    scaled and turned as the rows' scale and turn changed between them."""
    W, H = table
    x0, y0, k0, a0 = row_at(rows, s0)
    x1, y1, k1, a1 = row_at(rows, s)
    return similarity((x1 * W - 0.5, y1 * H - 0.5), k1 / k0, math.radians(a1 - a0), (x0 * W - 0.5, y0 * H - 0.5))


def alpha_at(keys, t):
    """A contact shadow's alpha at clip second `t`: linear between its keys, held outside them."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def outline_px(points, table):
    """An outline's table fractions as table px (n, 2)."""
    W, H = table
    q = np.asarray(points, float)
    return np.stack([q[:, 0] * W - 0.5, q[:, 1] * H - 0.5], 1)


def through(M, pts):
    """Points (n, 2) carried by the 2 x 3 map `M`."""
    M = np.asarray(M, float)
    return pts @ M[:, :2].T + M[:, 2]


def falloff(d, soft):
    """A contact shadow's darkness (0..1) at `d`, its ellipse's own radius (0 its middle, 1 its rim): whole inside
    1 - soft, then easing out to nothing at the rim."""
    u = np.clip((1.0 - d) / soft, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)
