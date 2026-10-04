"""Ink: handwriting as a ribbon mesh in which every vertex knows when the nib passed it (docs/design.md: Text, Ink).

A stroke is one pen-down polyline: points in page millimetres (from the top-left corner as read: x right, y down) and
the clip time (seconds, never decreasing) at which the nib is on each point. `ribbon` turns all strokes into one mesh in
the page's surface frame (x right, y up, z out of the paper): per stroke a strip of quads of one width riding the
polyline, `lift` above the paper, with a square cap half a width long at both ends, and the write time of the
cross-section as the value `tw` of its two vertices. A shader that shows a fragment once the scene clock has passed its
`tw` shows the ink exactly behind the nib: attributes interpolate along every quad, so the front moves smoothly between
points and between frames, and a nib that crosses 3-10 mm per frame still leaves legible letters.

The strokes of a project come from its own reference data (`strokes` file: the writing schedule) or, coarsely, from a
pen track's `down` runs (`from_track`). numpy only: runs in Blender, the CLI and the tests."""
import json
from typing import NamedTuple

import numpy as np

WIDTH = 0.00042                 # ribbon width, m: a fine fountain-pen nib
LIFT = 0.00018                  # m above the surface (no z-fighting with the paper)
MITER = 2.0                     # longest miter at a bend, in half widths: sharper bends get thinner, never spiky
DOT = 0.1                       # a one-point stroke is a segment this many widths long (+ the caps: a dot)
DOT_S = 1e-3                    # ... written in this many seconds
TOL = 0.002                     # points closer than this many widths to their predecessor are one point


class InkError(ValueError):
    """A strokes file or option that cannot make ink."""


class Stroke(NamedTuple):
    t: np.ndarray               # (n,) clip seconds, non-decreasing
    p: np.ndarray               # (n, 2) page millimetres


class Ribbon(NamedTuple):
    verts: np.ndarray           # (V, 3) metres in the surface frame (x right, y up, z out); two per cross-section
    faces: np.ndarray           # (F, 4) vertex indices, counter-clockwise seen from +z
    tw: np.ndarray              # (V,) seconds: when the nib is on the vertex's cross-section
    strokes: int
    points: int                 # polyline points kept (repeated points merged, dots given a second one)


# ------------------------------------------------------------------------------------------------------ strokes
def parse(doc, where="strokes"):
    """A strokes document `{"unit": "mm", "strokes": [{"t": [...], "p": [[x, y], ...]}, ...]}` -> [Stroke].
    `unit` may be left out (millimetres); strokes keep their order, the rest is checked: one [x, y] per time, finite
    numbers, times that do not decrease."""
    if not isinstance(doc, dict) or not isinstance(doc.get("strokes"), list):
        raise InkError(f"{where}: needs a `strokes` list ({{\"unit\": \"mm\", \"strokes\": [{{\"t\": [...], \"p\": "
                       f"[[x, y], ...]}}]}})")
    unit = doc.get("unit", "mm")
    if unit != "mm":
        raise InkError(f"{where}: unit {unit!r}: strokes are in page millimetres (\"unit\": \"mm\")")
    out = []
    for k, s in enumerate(doc["strokes"]):
        if not isinstance(s, dict) or "t" not in s or "p" not in s:
            raise InkError(f"{where}: stroke {k}: needs `t` (times) and `p` (points)")
        try:
            t, p = np.asarray(s["t"], float), np.asarray(s["p"], float)
        except (TypeError, ValueError):
            raise InkError(f"{where}: stroke {k}: `t` and `p` must be numbers") from None
        if t.ndim != 1 or len(t) == 0:
            raise InkError(f"{where}: stroke {k}: needs at least one time and point")
        if p.shape != (len(t), 2):
            raise InkError(f"{where}: stroke {k}: {len(t)} times but `p` has shape {list(p.shape)}: one [x, y] per time")
        if not (np.isfinite(t).all() and np.isfinite(p).all()):
            raise InkError(f"{where}: stroke {k}: times and points must be finite numbers")
        if np.any(np.diff(t) < 0):
            raise InkError(f"{where}: stroke {k}: times must not decrease along a stroke")
        out.append(Stroke(t, p))
    if not out:
        raise InkError(f"{where}: no strokes")
    return out


def page_size(doc, where="strokes"):
    """(w, h) in millimetres of the page the strokes were made for (`"page": [w, h]`), or None when not given."""
    page = doc.get("page") if isinstance(doc, dict) else None
    if page is None:
        return None
    try:
        w, h = (float(v) for v in page)
    except (TypeError, ValueError):
        raise InkError(f"{where}: `page` must be [width, height] in millimetres") from None
    if not (w > 0 and h > 0):
        raise InkError(f"{where}: `page` must be positive [width, height] in millimetres")
    return w, h


def load(path):
    """(strokes, page_mm or None) of a strokes file."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except OSError as e:
        raise InkError(f"{path}: {e.strerror or e}") from None
    except ValueError as e:
        raise InkError(f"{path}: not valid JSON ({e})") from None
    return parse(doc, str(path)), page_size(doc, str(path))


def from_track(t, xy_mm, down):
    """Coarse strokes from a pen track sampled once per frame: one stroke per run of samples with `down` set, through
    the nib positions `xy_mm` (n, 2, page millimetres) at the times `t` (n,). Letters will not be legible."""
    t, xy, down = np.asarray(t, float), np.asarray(xy_mm, float), np.asarray(down).astype(bool)
    if xy.shape != (len(t), 2) or down.shape != (len(t),):
        raise InkError(f"a pen track needs one time, [x, y] and down flag per sample "
                       f"(got {len(t)} times, positions {list(xy.shape)}, {down.shape[0] if down.ndim else 0} flags)")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise InkError("a pen track's times must increase")
    edge = np.diff(np.r_[False, down, False].astype(np.int8))
    return [Stroke(t[a:b], xy[a:b]) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def window(strokes, t0=None, t1=None):
    """The part of the ink written between t0 and t1 (clip seconds; None = unbounded): strokes outside are dropped, a
    stroke that crosses a bound is cut exactly there (the cut point interpolated between its neighbours)."""
    t0 = -np.inf if t0 is None else float(t0)
    t1 = np.inf if t1 is None else float(t1)
    if t1 < t0:
        raise InkError(f"the ink window ends ({t1:g} s) before it starts ({t0:g} s)")
    out = []
    for s in strokes:
        if s.t[-1] < t0 or s.t[0] > t1:
            continue
        lo, hi = int(np.searchsorted(s.t, t0, "left")), int(np.searchsorted(s.t, t1, "right"))
        t, p = s.t[lo:hi], s.p[lo:hi]
        if lo > 0 and (len(t) == 0 or t[0] > t0):                   # started before t0: begin exactly at t0
            f = (t0 - s.t[lo - 1]) / (s.t[lo] - s.t[lo - 1])
            t, p = np.r_[t0, t], np.vstack([s.p[lo - 1] + f * (s.p[lo] - s.p[lo - 1]), p])
        if hi < len(s.t) and t[-1] < t1:                            # runs past t1: end exactly at t1
            f = (t1 - s.t[hi - 1]) / (s.t[hi] - s.t[hi - 1])
            t, p = np.r_[t, t1], np.vstack([p, s.p[hi - 1] + f * (s.p[hi] - s.p[hi - 1])])
        if len(t):
            out.append(Stroke(t, p))
    return out


def front(strokes, t):
    """Page millimetres where the nib is at time t (the end of the ink being written), or None between strokes."""
    for s in strokes:
        if s.t[0] <= t <= s.t[-1]:
            return np.array([np.interp(t, s.t, s.p[:, 0]), np.interp(t, s.t, s.p[:, 1])])
    return None


def length_mm(strokes):
    """Total length of the ink, millimetres."""
    return float(sum(np.hypot(*np.diff(s.p, axis=0).T).sum() for s in strokes))


# ------------------------------------------------------------------------------------------------------ frames
def to_surface(p_mm, size):
    """Page millimetres (from the top-left corner as read, y down) -> metres in a surface's frame (origin at the middle
    of the panel, x right, y up); `size` = (w, h) of the panel in metres. cafe_page's page_to_local() for 160 x 220."""
    p = np.asarray(p_mm, float) * 1e-3
    return np.stack([p[..., 0] - 0.5 * size[0], 0.5 * size[1] - p[..., 1]], -1)


def to_page(xy, size):
    """The inverse of to_surface: metres in the surface frame -> page millimetres."""
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] + 0.5 * size[0]) * 1e3, (0.5 * size[1] - xy[..., 1]) * 1e3], -1)


# ------------------------------------------------------------------------------------------------------ ribbon
def ribbon(strokes, size, width=WIDTH, lift=LIFT, miter=MITER):
    """All strokes as one ribbon mesh in the surface frame (see the module docstring); `size` = (w, h) m of the panel
    the page millimetres map onto. Repeated points merge (a pen resting in place), a one-point stroke becomes a dot,
    the strip is offset by half a width along the bisector normal at every point, lengthened at bends so it keeps its
    width (up to `miter` half widths), and each end gets a cap half a width long. Vertices alternate left, right of the
    direction of writing; both of a cross-section carry its time."""
    strokes = list(strokes)
    S = len(strokes)
    if S == 0:
        return Ribbon(np.zeros((0, 3)), np.zeros((0, 4), np.int32), np.zeros(0), 0, 0)
    if not (width > 0 and miter >= 1.0):
        raise InkError("ink width must be positive and the miter at least 1")
    hw = 0.5 * width
    sid = np.repeat(np.arange(S), [len(s.t) for s in strokes])
    p = to_surface(np.concatenate([s.p for s in strokes]), size)
    t = np.concatenate([s.t for s in strokes]).astype(float)

    keep = np.r_[True, (sid[1:] != sid[:-1]) | (np.hypot(*(p[1:] - p[:-1]).T) > TOL * width)]
    p, t, sid = p[keep], t[keep], sid[keep]
    counts = np.bincount(sid, minlength=S)
    dots = np.flatnonzero(counts == 1)
    if len(dots):                                                   # a dot: a second point a tenth of a width on
        at = (np.cumsum(counts) - counts)[dots] + 1
        p = np.insert(p, at, p[at - 1] + [DOT * width, 0.0], axis=0)
        t = np.insert(t, at, t[at - 1] + DOT_S)
        sid = np.insert(sid, at, dots)
        counts[dots] = 2
    start = np.r_[0, np.cumsum(counts)]
    N = len(p)

    # direction of writing into and out of every point; the ends have only one
    seg = p[1:] - p[:-1]
    u = seg / np.hypot(seg[:, 0], seg[:, 1])[:, None].clip(1e-30)
    inner = sid[1:] == sid[:-1]
    d_in, d_out = np.zeros((N, 2)), np.zeros((N, 2))
    d_in[1:], d_out[:-1] = u, u
    has_in, has_out = np.r_[False, inner], np.r_[inner, False]
    d_in[~has_in] = d_out[~has_in]
    d_out[~has_out] = d_in[~has_out]
    bis = d_in + d_out
    bl = np.hypot(bis[:, 0], bis[:, 1])
    turned = bl < 1e-6                                              # the nib doubles back: stay square to the way in
    d = np.where(turned[:, None], d_in, bis / bl[:, None].clip(1e-30))
    cos_half = (d * d_out).sum(1)
    scale = np.where(turned, 1.0, np.minimum(1.0 / cos_half.clip(1e-9), miter))
    off = np.stack([-d[:, 1], d[:, 0]], 1) * (hw * scale)[:, None]  # to the left of the direction of writing

    # rows: [cap, points..., cap] per stroke; row of point i = i + 2 * stroke + 1
    R = N + 2 * S
    pos, offs, tw = np.empty((R, 2)), np.empty((R, 2)), np.empty(R)
    row = np.arange(N) + 2 * sid + 1
    pos[row], offs[row], tw[row] = p, off, t
    first, last = start[:-1], start[1:] - 1
    cap_a, cap_b = first + 2 * np.arange(S), start[1:] + 2 * np.arange(S) + 1
    pos[cap_a], offs[cap_a], tw[cap_a] = p[first] - d[first] * hw, off[first], t[first]
    pos[cap_b], offs[cap_b], tw[cap_b] = p[last] + d[last] * hw, off[last], t[last]

    verts = np.empty((2 * R, 3))
    verts[0::2, :2], verts[1::2, :2], verts[:, 2] = pos + offs, pos - offs, lift
    r = np.arange(R - 1)
    r = r[~np.isin(r, cap_b)]                                       # no quad from a stroke's last row to the next one
    faces = np.stack([2 * r + 1, 2 * r + 3, 2 * r + 2, 2 * r], 1).astype(np.int32)
    return Ribbon(verts, faces, np.repeat(tw, 2), S, N)
