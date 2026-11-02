"""Numerals drawn in code, no font file (docs/design.md: Props, `mile_post`; docs/modelling.md: Worked example: the mile post).

A monoline skeleton per digit 0-9 in the proportions of the US highway alphabets (FHWA Standard Alphabets, Series E 2000:
one stroke weight, 0.1725 of the height; the numerals 0.81 of it wide, the 1 0.30, the 4 0.94, the 0 0.84), and the meshes
made from it. Pure numpy (no bpy): it runs in Blender and in the tests, and a result depends on its arguments alone.

A numeral's frame is x right, y up, its ink box from (0, 0) to (width, height); a skeleton is its strokes' centre lines,
which keep half a stroke inside that box. Each digit is drawn once in the unit box of its centre lines (u, v in 0..1:
straight segments, elliptical arcs, cubic Beziers and superellipse loops); a stroke is tangent-continuous and a corner starts
a new stroke, so the round caps of two strokes make a round join. The unit box is mapped onto the ink box of the width and
stroke asked: `condense` (1: Series E) narrows a numeral without thinning its stroke, as the condensed gothics of
19th-century castings, and the 1 keeps its width.

    strokes(d, height, condense, stroke)         [(points (k, 2), closed)] of one digit, its ink box from the origin
    layout(text, height, condense, ...)          the strokes of every digit in a row (or a column), centred on the origin
    ribbons(placed, width, lift)                 flat strips with round caps facing +z: paint, sign sheeting (a graphic layer)
    relief(placed, width, height, sink)          raised strokes of a rounded section with round ends: a casting, an emboss

The meshes are in the surface's frame (x right, y up, z out of it): `core.shell.place` puts them on a plate."""
import math

import numpy as np

from . import shell as S

DIGITS = "0123456789"
STROKE = 0.1725                 # Series E: the stroke is 0.1725 of the numeral height
WIDTH = {"0": 0.84, "1": 0.30, "2": 0.81, "3": 0.81, "4": 0.94, "5": 0.81, "6": 0.81, "7": 0.81, "8": 0.81, "9": 0.81}
CELL = 0.81                     # the width a row gives every numeral (a narrower one is centred in it, the 4 overhangs it)
GAP = 0.15                      # between two cells of a row, heights (Series E's side bearings are 0.07 .. 0.10 each)
STEP = 0.012                    # sample spacing along a stroke, heights
_DENSE = 400                    # samples per unit of length while a curve is drawn in the unit box (then resampled)


# ================================================================================================= the unit-box drawing
def _n(length):
    return max(2, int(math.ceil(length * _DENSE)))


def _line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0.0, 1.0, _n(np.linalg.norm(b - a)) + 1)[:, None]
    return a + (b - a) * t


def _arc(c, r, a0, a1):
    """An elliptical arc about c with radii r = (ru, rv) from angle a0 to a1 (degrees; a1 < a0 runs clockwise)."""
    n = _n(abs(math.radians(a1 - a0)) * max(r))
    t = np.radians(np.linspace(a0, a1, n + 1))
    return np.stack([c[0] + r[0] * np.cos(t), c[1] + r[1] * np.sin(t)], 1)


def _bez(p0, p1, p2, p3):
    P = np.array([p0, p1, p2, p3], float)
    n = _n(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])


def _loop(c, r, p):
    """A closed superellipse |u'|^p + |v'|^p = 1 about c with radii r, counter-clockwise from its right side."""
    n = _n(2.0 * math.pi * max(r))
    t = 2.0 * math.pi * np.arange(n) / n
    cs, sn = np.cos(t), np.sin(t)
    return np.stack([c[0] + r[0] * np.sign(cs) * np.abs(cs) ** (2.0 / p),
                     c[1] + r[1] * np.sign(sn) * np.abs(sn) ** (2.0 / p)], 1)


def _path(*segs):
    """One stroke from segments that meet end to start (the joint points are kept once)."""
    out = [segs[0]]
    for s in segs[1:]:
        out.append(s[1:] if np.allclose(s[0], out[-1][-1], atol=1e-9) else s)
    return np.concatenate(out)


def _rot180(strokes):
    return [(1.0 - P, closed) for P, closed in strokes]


def _skeletons():
    """{digit: [(points (k, 2) in the unit box of its centre lines, closed)]}: the drawing of every numeral."""
    sk = {}
    sk["0"] = [(_loop((0.5, 0.5), (0.5, 0.5), 2.6), True)]
    sk["1"] = [(_line((1.0, 0.0), (1.0, 1.0)), False),
               (_path(_bez((1.0, 1.0), (0.72, 0.90), (0.30, 0.80), (0.0, 0.78))), False)]
    top2 = _arc((0.5, 0.69), (0.5, 0.31), 162.0, -28.0)
    a = math.radians(-28.0)
    t0 = np.array([0.5 * math.sin(a), -0.31 * math.cos(a)])           # the arc's way on (clockwise) where it ends
    t0 /= np.linalg.norm(t0)
    sk["2"] = [(_path(top2, _bez(top2[-1], top2[-1] + 0.36 * t0, (0.0, 0.34), (0.0, 0.0))), False),
               (_line((0.0, 0.0), (1.0, 0.0)), False)]
    sk["3"] = [(_path(_arc((0.48, 0.775), (0.45, 0.225), 158.0, -90.0), _line((0.48, 0.55), (0.30, 0.55))), False),
               (_arc((0.49, 0.275), (0.51, 0.275), 90.0, -158.0), False)]
    sk["4"] = [(_line((0.78, 0.0), (0.78, 1.0)), False), (_line((0.78, 1.0), (0.0, 0.28)), False),
               (_line((0.0, 0.28), (1.0, 0.28)), False)]
    bowl5 = _arc((0.48, 0.33), (0.52, 0.33), 90.0, -155.0)
    sk["5"] = [(_line((0.97, 1.0), (0.10, 1.0)), False), (_line((0.10, 1.0), (0.05, 0.57)), False),
               (_path(_bez((0.05, 0.57), (0.17, 0.64), (0.32, 0.66), (0.48, 0.66)), bowl5), False)]
    sk["6"] = [(_loop((0.5, 0.31), (0.5, 0.31), 2.3), True),
               (_bez((0.84, 1.0), (0.36, 1.0), (0.0, 0.82), (0.0, 0.31)), False)]
    sk["7"] = [(_line((0.0, 1.0), (1.0, 1.0)), False), (_bez((1.0, 1.0), (0.74, 0.70), (0.36, 0.34), (0.32, 0.0)), False)]
    sk["8"] = [(_loop((0.5, 0.775), (0.42, 0.225), 2.3), True), (_loop((0.5, 0.29), (0.5, 0.29), 2.3), True)]
    sk["9"] = _rot180(sk["6"])
    return sk


SKELETONS = _skeletons()


# ================================================================================================= one numeral
def width(d, condense=1.0, stroke=STROKE):
    """The ink width of digit d over its height: Series E's, narrowed by `condense` (the stroke kept; never below the 1's)."""
    w = WIDTH[d]
    return max(stroke + (w - stroke) * float(condense), min(w, WIDTH["1"]))


def _resample(P, step, closed):
    if closed:
        n = max(8, int(math.ceil(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum() / step)))
        return S.resample_loop(P, n)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)


def strokes(d, height=1.0, condense=1.0, stroke=STROKE, step=STEP):
    """[(points (k, 2), closed)] of digit `d` (a character "0" .. "9") `height` tall: its ink box from (0, 0) to
    (width(d) * height, height), the strokes' centre lines half a stroke inside it, sampled every `step` heights."""
    if d not in SKELETONS:
        raise ValueError(f"numerals: {d!r} is not a digit 0-9")
    w = width(d, condense, stroke)
    lo = np.array([0.5 * stroke, 0.5 * stroke])
    span = np.array([w - stroke, 1.0 - stroke])
    return [(_resample((lo + P * span) * height, step * height, closed), closed) for P, closed in SKELETONS[d]]


# ================================================================================================= a number
def check(text):
    """`text` as a string of digits, or a ValueError."""
    s = str(text)
    if not s or any(c not in DIGITS for c in s):
        raise ValueError(f"numerals: {text!r} is not a string of digits 0-9")
    return s


def cell(condense=1.0, stroke=STROKE):
    """The width a row gives each numeral, over the height."""
    return stroke + (CELL - stroke) * float(condense)


def extent(n, height=1.0, condense=1.0, stroke=STROKE, gap=GAP, column=False):
    """(w, h) of the box `n` numerals take in a row (side by side, `gap` heights between cells) or a column (one under
    the other, `gap` heights apart)."""
    c = cell(condense, stroke)
    if column:
        return c * height, (n + (n - 1) * gap) * height
    return (n * c + (n - 1) * gap) * height, height


def layout(text, height=1.0, condense=1.0, stroke=STROKE, gap=GAP, column=False, step=STEP):
    """Every stroke of the numerals of `text`, each numeral centred in its cell, the whole box (`extent`) centred on the
    origin: [(points (k, 2), closed)]. A column reads top to bottom."""
    s = check(text)
    W, H = extent(len(s), height, condense, stroke, gap, column)
    c = cell(condense, stroke) * height
    out = []
    for i, d in enumerate(s):
        w = width(d, condense, stroke) * height
        if column:
            x0, y0 = -0.5 * w, 0.5 * H - height - i * (1.0 + gap) * height
        else:
            x0, y0 = -0.5 * W + i * (c + gap * height) + 0.5 * (c - w), -0.5 * height
        out += [(P + (x0, y0), closed) for P, closed in strokes(d, height, condense, stroke, step)]
    return out


# ================================================================================================= meshes
def _tangents(P, closed):
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.gradient(P, axis=0)
    return T / np.maximum(np.linalg.norm(T, axis=1), 1e-12)[:, None]


def _cap_fan(c, t, r, seg, z):
    """A half disc of radius r about c (2D) bulging along t, flat at height z: (V, T)."""
    a0 = math.atan2(t[1], t[0])
    ang = a0 + np.linspace(-0.5 * math.pi, 0.5 * math.pi, seg + 1)
    rim = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.full(seg + 1, z)], 1)
    V = np.vstack([[c[0], c[1], z], rim])
    T = np.stack([np.zeros(seg, np.int64), np.arange(1, seg + 1), np.arange(2, seg + 2)], 1)
    return V, T


def section(width, height, sink=0.0, p=2.6, n=12):
    """The raised stroke's section, (u, v) counter-clockwise: a flat foot `sink` under the surface (none when 0), the
    sides rising into a rounded top (a half superellipse of exponent p) `height` over it."""
    t = np.linspace(0.0, math.pi, n + 1)
    c, s = np.cos(t), np.sin(t)
    top = np.stack([0.5 * width * np.sign(c) * np.abs(c) ** (2.0 / p), height * np.abs(s) ** (2.0 / p)], 1)
    if sink <= 0.0:
        return top
    return np.vstack([[0.5 * width, -sink], top, [-0.5 * width, -sink]])


def ribbons(placed, width, lift=0.0, cap_seg=8, mat=0):
    """Flat strips `width` wide along the strokes, round caps on the open ends, `lift` over z = 0 and facing +z: one
    Mesh (a graphic layer: overlapping strokes of one colour, no thickness)."""
    out = []
    for P, closed in placed:
        P3 = np.column_stack([P, np.full(len(P), lift)])
        out.append(S.ribbon(P3, width, (0.0, 0.0, 1.0), closed=closed, mat=mat))
        if not closed:
            T = _tangents(P, False)
            for c, t in ((P[0], -T[0]), (P[-1], T[-1])):
                V, F = _cap_fan(c, t, 0.5 * width, cap_seg, lift)
                cap = S.Mesh(V, None, F, mat=mat)
                nq, nt = cap.face_vectors()
                out.append(cap if nt[:, 2].sum() >= 0 else cap.flip())
    return S.merge(out)


def relief(placed, width, height, sink=0.0, end_seg=6, mat=0):
    """Raised strokes standing `height` out of the surface z = 0 (their feet `sink` under it): the rounded `section`
    swept along each stroke, an open stroke running on half a stroke past its ends while its section narrows on a
    quarter circle (a round end in plan). One Mesh of closed solids."""
    sec = section(width, height, sink)
    r = 0.5 * width
    out = []
    for P, closed in placed:
        P3 = np.column_stack([P, np.zeros(len(P))])
        if closed:
            m = S.sweep(P3, sec, up=(0.0, 0.0, 1.0), closed=True, mat=mat).weld()     # a ring: weld its seam shut
        else:
            T = _tangents(P, False)
            d = r * np.sin(np.linspace(0.0, 0.5 * math.pi, end_seg + 1)[1:])
            k = np.sqrt(np.maximum(1.0 - (d / r) ** 2, 0.0))
            k = np.maximum(k, 0.18)
            head = P3[0] - np.column_stack([np.outer(d[::-1], T[0]), np.zeros(end_seg)])
            tail = P3[-1] + np.column_stack([np.outer(d, T[-1]), np.zeros(end_seg)])
            path = np.vstack([head, P3, tail])
            su = np.concatenate([k[::-1], np.ones(len(P3)), k])
            m = S.sweep(path, sec, scale=np.column_stack([su, np.ones(len(su))]), up=(0.0, 0.0, 1.0), mat=mat)
        out.append(m if m.volume() >= 0 else m.flip())
    return S.merge(out)
