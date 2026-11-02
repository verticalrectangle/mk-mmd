"""The gel-pen hand, bpy-free (docs/design.md: Picture effects): letters as the strokes a pen writes them in, for an
effect that writes a short text by hand over the picture (a purikura booth's dates and words), drawn with the doodle pen of
mkmmd.picturefx and boiling with it.

A glyph is a few polylines in a box one cap tall: x right from its left edge, y down from the cap line, the baseline at 1,
a lowercase letter's top at X_HEIGHT and the descenders to about 1.36; each stroke is one pen-down line, in the order a
hand writes it. `strokes(text)` lays a text out in cap heights about its middle, each letter a little turned and lifted
as a hand leaves it (seeded); `written` cuts the strokes where the pen has got to."""
import math
from functools import lru_cache

import numpy as np

X_HEIGHT = 0.42           # a lowercase letter's top, from the cap line (caps)
SPACING = 0.2             # between two letters (caps)
SPACE = 0.4               # a space's width (caps)
STEP = 0.03               # a curve's points come every this many caps
JITTER = {"turn": 4.0, "lift": 0.035, "grow": 0.04}   # a hand's letters: degrees, caps, a share of their size


class HandError(ValueError):
    """A text the hand cannot write."""


# ================================================================================================= drawing a glyph
def _arc(cx, cy, rx, ry, a0, a1):
    """Points on an ellipse from angle a0 to a1 (degrees: 0 to the right, 90 down, so a rising angle runs clockwise on
    screen and a falling one counter-clockwise)."""
    span = math.radians(abs(a1 - a0)) * max(rx, ry)
    a = np.radians(np.linspace(a0, a1, max(int(math.ceil(span / STEP)), 2) + 1))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def _oval(cx, cy, rx, ry, start=-120.0):
    """A closed ellipse as a pen draws it: from `start` round once and a little on past it."""
    return _arc(cx, cy, rx, ry, start, start + 380.0)


def _bezier(*ctrl):
    """A quadratic (three points) or cubic (four) Bezier curve from the first point to the last."""
    p = np.asarray(ctrl, float)
    length = float(np.hypot(*np.diff(p, axis=0).T).sum())
    u = np.linspace(0.0, 1.0, max(int(math.ceil(length / STEP)), 2) + 1)[:, None]
    if len(p) == 3:
        return (1 - u) ** 2 * p[0] + 2 * (1 - u) * u * p[1] + u ** 2 * p[2]
    return (1 - u) ** 3 * p[0] + 3 * (1 - u) ** 2 * u * p[1] + 3 * (1 - u) * u ** 2 * p[2] + u ** 3 * p[3]


def _path(*parts):
    """One stroke from its parts in order: points (x, y), runs of points (an _arc), and curves from the point reached so
    far, ("q", control, end) or ("c", control, control, end)."""
    out = []
    for p in parts:
        if isinstance(p, tuple) and p and isinstance(p[0], str):
            a = _bezier(out[-1][-1], *p[1:])
        else:
            a = np.atleast_2d(np.asarray(p, float))
        if out and np.allclose(out[-1][-1], a[0]):
            a = a[1:]
        if len(a):
            out.append(a)
    return np.concatenate(out)


def _dot(x, y):
    return _oval(x, y, 0.035, 0.035)


def _lines(*pts):
    return np.asarray(pts, float)


def _heart():
    t = np.linspace(0.0, 2.0 * np.pi, 72)
    x, y = 16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    p = np.stack([x / 33.0, (y - 2.5) / 33.0], 1) * 0.95 + (0.47, 0.52)
    return np.concatenate([p, p[1:4]])


def _star():
    a = -np.pi / 2 + np.arange(11) * np.pi / 5
    r = np.where(np.arange(11) % 2 == 0, 0.5, 0.21)
    p = np.stack([0.47 + 0.95 * r * np.cos(a), 0.54 + 0.95 * r * np.sin(a)], 1)
    return np.concatenate([p, p[1:2] * 0.25 + p[:1] * 0.75])


def _upper():
    P = _lines
    return {
        "A": (0.66, [P((0, 1), (0.33, 0), (0.66, 1)), P((0.15, 0.6), (0.51, 0.6))]),
        "B": (0.6, [P((0, 0), (0, 1)), _path((0, 0), _arc(0.28, 0.24, 0.25, 0.24, -90, 90), (0.02, 0.48),
                                              _arc(0.31, 0.74, 0.29, 0.26, -90, 90), (0, 1))]),
        "C": (0.68, [_arc(0.36, 0.5, 0.34, 0.5, -42, -318)]),
        "D": (0.64, [P((0, 0), (0, 1)), _path((0, 0), _arc(0.16, 0.5, 0.46, 0.5, -90, 90), (0, 1))]),
        "E": (0.54, [P((0.52, 0), (0, 0), (0, 1), (0.54, 1)), P((0, 0.5), (0.44, 0.5))]),
        "F": (0.5, [P((0.5, 0), (0, 0), (0, 1)), P((0, 0.48), (0.42, 0.48))]),
        "G": (0.7, [_path(_arc(0.36, 0.5, 0.34, 0.5, -42, -340), (0.69, 0.52), (0.42, 0.52))]),
        "H": (0.6, [P((0, 0), (0, 1)), P((0.6, 0), (0.6, 1)), P((0, 0.5), (0.6, 0.5))]),
        "I": (0.3, [P((0, 0), (0.3, 0)), P((0.15, 0), (0.15, 1)), P((0, 1), (0.3, 1))]),
        "J": (0.48, [_path((0.46, 0), (0.46, 0.72), _arc(0.24, 0.72, 0.22, 0.28, 0, 180))]),
        "K": (0.58, [P((0, 0), (0, 1)), P((0.56, 0), (0.02, 0.6)), P((0.2, 0.42), (0.6, 1))]),
        "L": (0.5, [P((0, 0), (0, 1), (0.5, 1))]),
        "M": (0.76, [P((0, 1), (0.05, 0), (0.38, 0.64), (0.71, 0), (0.76, 1))]),
        "N": (0.62, [P((0, 1), (0, 0), (0.62, 1), (0.62, 0))]),
        "O": (0.76, [_oval(0.38, 0.5, 0.38, 0.5)]),
        "P": (0.56, [_path((0, 1), (0, 0), _arc(0.26, 0.27, 0.28, 0.27, -90, 90), (0, 0.54))]),
        "Q": (0.78, [_oval(0.38, 0.5, 0.38, 0.5), P((0.46, 0.7), (0.8, 1.05))]),
        "R": (0.6, [_path((0, 1), (0, 0), _arc(0.26, 0.27, 0.28, 0.27, -90, 90), (0, 0.54)), P((0.24, 0.54), (0.6, 1))]),
        "S": (0.56, [_path(_arc(0.28, 0.25, 0.26, 0.25, -25, -270), _arc(0.28, 0.75, 0.28, 0.25, -90, 155))]),
        "T": (0.62, [P((0, 0), (0.62, 0)), P((0.31, 0), (0.31, 1))]),
        "U": (0.62, [_path((0, 0), _arc(0.31, 0.66, 0.31, 0.34, 180, 0), (0.62, 0))]),
        "V": (0.64, [P((0, 0), (0.32, 1), (0.64, 0))]),
        "W": (0.86, [P((0, 0), (0.2, 1), (0.43, 0.32), (0.66, 1), (0.86, 0))]),
        "X": (0.6, [P((0, 0), (0.6, 1)), P((0.6, 0), (0, 1))]),
        "Y": (0.62, [P((0, 0), (0.31, 0.5), (0.62, 0)), P((0.31, 0.5), (0.31, 1))]),
        "Z": (0.6, [P((0.02, 0), (0.6, 0), (0, 1), (0.62, 1))]),
    }


def _lower():
    P, x = _lines, X_HEIGHT
    return {
        "a": (0.5, [_oval(0.24, 0.71, 0.24, 0.29, -30), P((0.48, x), (0.48, 1))]),
        "b": (0.5, [P((0, 0), (0, 1)), _oval(0.25, 0.71, 0.25, 0.29, 150)]),
        "c": (0.46, [_arc(0.24, 0.71, 0.23, 0.29, -40, -320)]),
        "d": (0.5, [_oval(0.24, 0.71, 0.24, 0.29, -30), P((0.48, 0), (0.48, 1))]),
        "e": (0.5, [_path((0.02, 0.72), (0.47, 0.72), _arc(0.25, 0.71, 0.23, 0.29, 0, -322))]),
        "f": (0.36, [_path((0.38, 0.04), ("q", (0.12, -0.06), (0.12, 0.3)), (0.12, 1)), P((0, 0.44), (0.32, 0.44))]),
        "g": (0.5, [_oval(0.24, 0.7, 0.24, 0.28, -30),
                    _path((0.48, x), (0.48, 1.14), ("q", (0.48, 1.38), (0.24, 1.38)), ("q", (0.08, 1.38), (0.02, 1.28)))]),
        "h": (0.48, [P((0, 0), (0, 1)), _path(_arc(0.23, 0.66, 0.23, 0.24, 180, 360), (0.46, 1))]),
        "i": (0.1, [P((0.04, 0.44), (0.04, 1)), _dot(0.04, 0.22)]),
        "j": (0.26, [_path((0.2, 0.44), (0.2, 1.16), ("q", (0.2, 1.38), (0.0, 1.32))), _dot(0.2, 0.22)]),
        "k": (0.44, [P((0, 0), (0, 1)), P((0.42, 0.44), (0.02, 0.78)), P((0.16, 0.66), (0.44, 1))]),
        "l": (0.18, [_path((0.04, 0), (0.04, 0.86), ("q", (0.04, 1.0), (0.18, 0.98)))]),
        "m": (0.7, [P((0, x), (0, 1)), _path(_arc(0.175, 0.62, 0.175, 0.2, 180, 360), (0.35, 1)),
                    _path(_arc(0.525, 0.62, 0.175, 0.2, 180, 360), (0.7, 1))]),
        "n": (0.46, [P((0, x), (0, 1)), _path(_arc(0.23, 0.64, 0.23, 0.22, 180, 360), (0.46, 1))]),
        "o": (0.5, [_oval(0.25, 0.71, 0.25, 0.29)]),
        "p": (0.5, [P((0, x), (0, 1.36)), _oval(0.25, 0.71, 0.25, 0.29, 150)]),
        "q": (0.5, [_oval(0.24, 0.71, 0.24, 0.29, -30), P((0.48, x), (0.48, 1.36), (0.58, 1.28))]),
        "r": (0.36, [P((0, x), (0, 1)), _path((0, 0.66), ("q", (0.08, x), (0.36, 0.44)))]),
        "s": (0.42, [_path(_arc(0.21, 0.565, 0.19, 0.145, -25, -270), _arc(0.21, 0.855, 0.21, 0.145, -90, 155))]),
        "t": (0.36, [_path((0.14, 0.1), (0.14, 0.86), ("q", (0.14, 1.0), (0.32, 0.98))), P((0, 0.44), (0.34, 0.44))]),
        "u": (0.46, [_path((0, x), _arc(0.23, 0.76, 0.23, 0.24, 180, 0), (0.46, x)), P((0.46, x), (0.46, 1))]),
        "v": (0.46, [P((0, x), (0.23, 1), (0.46, x))]),
        "w": (0.66, [P((0, x), (0.16, 1), (0.33, 0.6), (0.5, 1), (0.66, x))]),
        "x": (0.44, [P((0, x), (0.44, 1)), P((0.44, x), (0, 1))]),
        "y": (0.46, [P((0, x), (0.23, 0.94)), P((0.46, x), (0.12, 1.36))]),
        "z": (0.44, [P((0.02, x), (0.44, x), (0, 1), (0.46, 1))]),
    }


def _figures():
    P = _lines
    return {
        "0": (0.56, [_oval(0.28, 0.5, 0.28, 0.5)]),
        "1": (0.36, [P((0.04, 0.2), (0.26, 0), (0.26, 1))]),
        "2": (0.56, [_path(_arc(0.27, 0.27, 0.26, 0.27, -165, 10), ("q", (0.42, 0.6), (0, 1)), (0.58, 1))]),
        "3": (0.56, [_path(_arc(0.26, 0.25, 0.25, 0.25, -160, 90), _arc(0.26, 0.74, 0.29, 0.26, -90, 160))]),
        "4": (0.6, [P((0.44, 1), (0.44, 0), (0, 0.68), (0.6, 0.68))]),
        "5": (0.56, [_path((0.52, 0), (0.08, 0), (0.04, 0.46), _arc(0.27, 0.7, 0.29, 0.3, -128, 155))]),
        "6": (0.56, [_path((0.48, 0.03), ("q", (0.02, 0.18), (0.01, 0.68)), _arc(0.28, 0.7, 0.27, 0.3, 180, 540))]),
        "7": (0.56, [P((0, 0), (0.56, 0), (0.18, 1))]),
        "8": (0.56, [_oval(0.28, 0.25, 0.22, 0.25, 90), _oval(0.28, 0.74, 0.28, 0.26, -90)]),
        "9": (0.56, [_path(_arc(0.28, 0.3, 0.27, 0.3, 0, -360), ("q", (0.56, 0.95), (0.12, 1.0)))]),
        ".": (0.1, [_dot(0.05, 0.95)]),
        ",": (0.14, [P((0.08, 0.9), (0.02, 1.14))]),
        ":": (0.1, [_dot(0.05, 0.42), _dot(0.05, 0.95)]),
        "/": (0.42, [P((0.42, 0), (0, 1))]),
        "-": (0.38, [P((0, 0.56), (0.38, 0.56))]),
        "!": (0.1, [P((0.05, 0), (0.05, 0.68)), _dot(0.05, 0.95)]),
        "?": (0.48, [_path(_arc(0.24, 0.25, 0.24, 0.25, -160, 60), ("q", (0.24, 0.52), (0.24, 0.7))), _dot(0.24, 0.95)]),
        "'": (0.1, [P((0.05, 0), (0.02, 0.26))]),
        "+": (0.56, [P((0.28, 0.28), (0.28, 0.84)), P((0, 0.56), (0.56, 0.56))]),
        "~": (0.56, [np.stack([np.linspace(0.0, 0.56, 24), 0.56 - 0.09 * np.sin(np.linspace(0.0, 2.0 * np.pi, 24))], 1)]),
        "\u2661": (0.95, [_heart()]),                                          # a heart
        "\u2606": (0.95, [_star()]),                                           # a star
        "\u266a": (0.62, [_oval(0.17, 0.85, 0.17, 0.13, -60),                 # a note
                          _path((0.34, 0.85), (0.34, 0.0), ("q", (0.38, 0.22), (0.62, 0.32)))]),
    }


SAME = {"\u2665": "\u2661", "\u2764": "\u2661", "\u2605": "\u2606", "\u266b": "\u266a"}   # filled ones: the pen outlines


@lru_cache(maxsize=1)
def glyphs():
    """{character: (advance, [strokes (n, 2)])}: every letter the hand writes, a space as an advance without strokes."""
    out = {**_upper(), **_lower(), **_figures(), " ": (SPACE, [])}
    out.update({k: out[v] for k, v in SAME.items()})
    return out


def alphabet():
    """The characters the hand writes, as one string."""
    return "".join(sorted(glyphs()))


def check(text, what="text"):
    """`text` if the hand writes every character of it, else a HandError naming those it does not."""
    missing = sorted(set(text) - set(glyphs()))
    if missing:
        raise HandError(f"{what}: the pen does not write {''.join(missing)!r} (it writes A-Z, a-z, 0-9, space and "
                        f"{''.join(c for c in alphabet() if not c.isalnum() and c != ' ')!r})")
    return text


# ================================================================================================= a text
def width(text):
    """A text's width in caps, its letters side by side SPACING apart."""
    g = glyphs()
    return sum(g[c][0] for c in text) + SPACING * max(len(text) - 1, 0)


def strokes(text, seed=0):
    """The strokes of `text` in writing order, [(n, 2)] in caps about the middle of its line (the cap line at -0.5, the
    baseline at 0.5), each letter a little turned, lifted and grown as a hand leaves it (JITTER, seeded by `seed`)."""
    check(text)
    g = glyphs()
    rng = np.random.default_rng([int(seed), 77])
    w = width(text)
    out, x = [], 0.0
    for c in text:
        adv, parts = g[c]
        a = math.radians(float(rng.uniform(-1.0, 1.0)) * JITTER["turn"])
        lift = float(rng.uniform(-1.0, 1.0)) * JITTER["lift"]
        k = 1.0 + float(rng.uniform(-1.0, 1.0)) * JITTER["grow"]
        rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) * k
        mid = np.array([adv / 2.0, 0.5])
        for p in parts:
            out.append((p - mid) @ rot.T + mid + (x - w / 2.0, lift - 0.5))
        x += adv + SPACING
    return out


def lengths(strokes_):
    """Each stroke's length (its own units)."""
    return [float(np.hypot(*np.diff(np.asarray(p, float), axis=0).T).sum()) if len(p) > 1 else 0.0 for p in strokes_]


def written(strokes_, share, lengths_=None):
    """The strokes as far as the pen has written them when `share` (0..1) of their whole length is down: those before
    whole, the one it is on cut there (each cut at its own share of its length, so a boiled copy of the strokes is cut
    where the strokes are), the rest not yet. `lengths_` the strokes' lengths (`lengths`, of the strokes as laid out)."""
    if lengths_ is None:
        lengths_ = lengths(strokes_)
    total = float(sum(lengths_))
    if share >= 1.0 or total <= 0.0:
        return [np.asarray(p, float) for p in strokes_]
    left, out = max(float(share), 0.0) * total, []
    for p, n in zip(strokes_, lengths_):
        if left <= 0.0:
            break
        p = np.asarray(p, float)
        if left >= n:
            out.append(p)
            left -= n
            continue
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        at = left / n * s[-1]
        k = int(np.searchsorted(s, at, side="right"))
        tip = p[k - 1] + (p[k] - p[k - 1]) * ((at - s[k - 1]) / max(s[k] - s[k - 1], 1e-12))
        out.append(np.concatenate([p[:k], tip[None]]))
        break
    return out
