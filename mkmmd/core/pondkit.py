"""The pond and café sticker kits (docs/design.md: Mandalas and trails), drawn with the pen's ink line round their parts
as the other kits in mkmmd.core.mandala are; mandala registers them (SHAPES, PENNED, their colours) and holds the
helpers they draw with. Each function gives its shape's parts, [(role, "fill" or "line", (n, 2) points, line width)]
about (0, 0), x right and y down, in the order they are painted (mandala.shape_parts scales the box to 1).

    pond   a water lily in bloom, a lily pad, a lily's bud, a reed (a cattail), a dragonfly and a water drop
    café   a croissant, a latte on its saucer seen from above, a sugar cube, an orange slice and a coffee bean
"""
import math

import numpy as np


def _cubic(p0, c0, c1, p1, n=16):
    """A cubic Bezier from p0 to p1 pulled toward c0 and then c1."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    p0, c0, c1, p1 = (np.asarray(p, float) for p in (p0, c0, c1, p1))
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c0 + 3 * (1 - u) * u ** 2 * c1 + u ** 3 * p1


def _pointed(length, half, base=0.0, at=0.45, n=14):
    """A pointed petal or leaf up from (0, -base) to its tip (0, -length), `half` wide each side at `at` of the way."""
    y0, y1 = -base, -length
    ym = y0 + (y1 - y0) * at
    left = _cubic([0.0, y0], [-1.3 * half, ym + 0.25 * (y0 - ym)], [-0.9 * half, ym + 0.55 * (y1 - ym)], [0.0, y1], n)
    return np.concatenate([left, left[-2:0:-1] * (-1.0, 1.0)])


def _teardrop(r, tip, n=48):
    """A drop: a disc `r` wide about (0, 0) drawn out to a point `tip` above its middle (y = -tip)."""
    a = math.asin(r / tip)                                    # where the sides leave the disc, tangent to it
    ts = np.linspace(-a, math.pi + a, n)                      # round the bottom, from the right side to the left
    disc = r * np.stack([np.cos(ts), np.sin(ts)], 1)
    side = _cubic(disc[-1], disc[-1] + (0.0, -0.35 * tip), (0.0, -0.75 * tip), (0.0, -tip), 12)
    return np.concatenate([disc, side[1:], side[-2:0:-1] * (-1.0, 1.0)])


def _arc(cx, cy, rx, ry, a0, a1, n=14):
    """Points along an ellipse's rim from a0 to a1 degrees (clockwise on screen from 3 o'clock)."""
    ts = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(ts), cy + ry * np.sin(ts)], 1)


def _blade(p0, c, p1, half, n=24):
    """A leaf blade along the curve from p0 to p1 pulled toward c, `half` wide each side at its foot and narrowing to
    a point at p1; and the curve (its midrib)."""
    from .mandala import _curve
    spine = _curve(p0, c, p1, n)
    d = np.gradient(spine, axis=0)
    side = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1)[:, None]
    w = (half * (1.0 - np.linspace(0.0, 1.0, n)) ** 0.6)[:, None]
    return np.concatenate([spine + side * w, (spine - side * w)[-2::-1]]), spine


# ================================================================================================= the pond
def lily():
    """A water lily in bloom, seen from the side and above: its outer row of pointed petals fanned out behind and lying
    out toward us, a paler inner row standing round its gold heart."""
    from .mandala import _drawn, _ellipse, _tilted, _turned
    foot = np.array([0.0, 0.18])

    def petal(a, length, half, squash=1.0):                  # from the foot, leaning `a` degrees right of upright
        return _turned(_pointed(length, half, 0.0, at=0.45), -a) * (1.0, squash) + foot
    parts = []
    for a, length in ((-64.0, 0.44), (64.0, 0.44), (-36.0, 0.5), (36.0, 0.5)):      # the outer row, behind
        parts += _drawn("main", petal(a, length, 0.1), 0.018)
    for a in (-32.0, 32.0):                                                           # the inner row's sides
        parts += _drawn("light", petal(a, 0.42, 0.075), 0.016)
    parts += _drawn("accent", _ellipse(0.0, 0.01, 0.15, 0.1, n=40), 0.014)            # the heart, its stamens
    parts += [("accent_dark", "fill", _ellipse(x, -0.04 + 2.5 * x * x, 0.016, 0.016, n=10), 0.0)
              for x in np.linspace(-0.1, 0.1, 7)]
    parts += _drawn("light", petal(0.0, 0.42, 0.066) + (0.0, 0.02), 0.016)            # the inner row's middle
    for a, length in ((-98.0, 0.42), (98.0, 0.42), (-132.0, 0.34), (132.0, 0.34), (180.0, 0.3)):  # lying toward us
        parts += _drawn("main", petal(a, length, 0.1, 0.6), 0.018)
    return _tilted(parts, 4.0)


def lilypad():
    """A round pad, its notch cut in to the middle, its veins out from there, a shine on its rim and dew on it."""
    from .mandala import _curve, _drawn, _ellipse
    a0, half = math.radians(58.0), math.radians(9.0)          # the notch: its middle (down and right), half its gap
    ts = np.linspace(a0 + half, a0 + 2.0 * math.pi - half, 120)
    rim = np.stack([0.5 * np.cos(ts), 0.46 * np.sin(ts)], 1)
    parts = _drawn("main", np.concatenate([[[0.03 * math.cos(a0), 0.03 * math.sin(a0)]], rim]))
    for k in range(1, 7):                                    # veins round from the notch, bowed a little
        a = a0 + 2.0 * math.pi * k / 7.0
        u, v = np.array([math.cos(a), 0.92 * math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        parts.append(("light", "line", _curve(0.09 * u, 0.23 * u + 0.03 * v, 0.37 * u, n=10), 0.024))
    parts.append(("light", "line", _arc(0.0, 0.0, 0.42, 0.385, 195.0, 245.0), 0.04))
    for x, y, r in ((-0.15, 0.17, 0.06), (0.05, -0.24, 0.04)):  # dew on the pad, a glint in each
        parts += _drawn("light", _ellipse(x, y, r, r * 0.92, n=24), 0.014)
        parts.append(("paper", "fill", _ellipse(x - 0.35 * r, y - 0.35 * r, 0.3 * r, 0.3 * r, n=12), 0.0))
    return parts


def bud():
    """A water lily's bud, closed: two petals wrapped round a paler one whose tip peeks out, the sepals cupping them,
    a short stalk."""
    from .mandala import _curve, _drawn, _tilted
    stalk = _curve([0.0, 0.15], [0.02, 0.32], [0.08, 0.42])
    parts = [("ink", "line", stalk, 0.075), ("accent", "line", stalk, 0.035)]
    parts += _drawn("light", _pointed(0.7, 0.18, 0.0, at=0.4) + (0.0, 0.22))
    for s in (-1.0, 1.0):                                    # the side petals, the right one over
        outer = _cubic([0.0, 0.22], [0.3 * s, 0.18], [0.2 * s, -0.2], [0.03 * s, -0.33], 18)
        inner = _cubic([0.03 * s, -0.33], [-0.04 * s, -0.2], [-0.07 * s, 0.05], [0.0, 0.22], 16)
        parts += _drawn("main", np.concatenate([outer, inner[1:-1]]))
    for s in (-1.0, 1.0):                                    # the sepals, cupping its foot
        out = _cubic([0.0, 0.25], [0.2 * s, 0.25], [0.21 * s, 0.12], [0.16 * s, 0.04], 14)
        back = _cubic([0.16 * s, 0.04], [0.11 * s, 0.1], [0.05 * s, 0.15], [-0.02 * s, 0.2], 12)
        parts += _drawn("accent", np.concatenate([out, back[1:-1]]), 0.018)
    parts.append(("light", "line", _arc(0.02, -0.04, 0.17, 0.2, 200.0, 245.0, 12), 0.03))
    return _tilted(parts, -14.0)


def reed():
    """A cattail: a velvety head on its stalk, the spike above it, a long leaf arching beside it and a short one."""
    from .mandala import _box, _curve, _drawn, _tilted
    foot = 0.38                                              # where the stalk and the leaves come up
    stalk = _curve([0.03, foot], [-0.02, 0.05], [0.0, -0.3])
    parts = [("ink", "line", stalk, 0.07), ("accent", "line", stalk, 0.034)]
    for p0, c, tip, w in (((-0.01, foot), (-0.06, 0.12), (-0.24, -0.02), 0.045),       # the leaves, the long one over
                          ((0.04, foot), (0.1, 0.0), (0.32, -0.3), 0.055)):
        blade, spine = _blade(p0, c, tip, w)
        parts += _drawn("accent", blade, 0.018) + [("accent_dark", "line", spine[2:-4], 0.014)]
    parts.append(("ink", "line", np.array([[0.0, -0.36], [0.01, -0.5]]), 0.03))       # the spike
    parts += _drawn("main", _box(-0.1, -0.38, 0.1, 0.07, 0.1))
    parts += [("light", "line", np.array([[-0.05, -0.29], [-0.05, -0.03]]), 0.035),
              ("dark", "line", np.array([[0.055, -0.22], [0.055, 0.0]]), 0.025)]
    return _tilted(parts, -10.0)


def dragonfly():
    """A dragonfly from above: four veined wings, a long slender body in segments, big eyes."""
    from .mandala import _box, _drawn, _ellipse, _mirrored, _tilted
    wings = []
    for cx, cy, rx, ry, a in ((0.26, -0.3, 0.24, 0.065, -6.0), (0.24, -0.19, 0.22, 0.075, 12.0)):   # fore, hind
        m = np.array([cx, cy])
        u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])  # along the wing; v across, to its back
        v = np.array([-u[1], u[0]])
        wings += _drawn("light", _ellipse(cx, cy, rx, ry, angle=a, n=48), 0.016)
        wings += [("paper", "line", np.array([m - 0.75 * rx * u - 0.25 * ry * v, m + 0.7 * rx * u - 0.25 * ry * v]),
                   0.022),                                   # a vein along its front, the spot near its tip
                  ("main", "fill", _ellipse(*(m + 0.74 * rx * u - 0.42 * ry * v), 0.034, 0.02, angle=a, n=14), 0.0)]
    parts = _mirrored(wings)
    parts += _drawn("main", _box(-0.036, -0.22, 0.036, 0.47, 0.034))            # the tail
    parts += [("dark", "line", np.array([[-0.034, y], [0.034, y]]), 0.012) for y in np.linspace(-0.08, 0.38, 6)]
    parts += _drawn("main", _ellipse(0.0, -0.25, 0.066, 0.09, n=32))           # the thorax
    for s in (-1.0, 1.0):                                    # the eyes, a glint in each
        parts += _drawn("accent", _ellipse(0.068 * s, -0.375, 0.076, 0.076, n=32), 0.016)
        parts.append(("paper", "fill", _ellipse(0.068 * s - 0.024, -0.398, 0.024, 0.024, n=12), 0.0))
    return _tilted(parts, -28.0)


def drop():
    """A water drop, a shine and a glint on it, a little one beside it."""
    from .mandala import _drawn, _ellipse
    parts = _drawn("accent", _teardrop(0.075, 0.17) + (0.32, -0.32), 0.016)
    parts.append(("paper", "fill", _ellipse(0.3, -0.3, 0.014, 0.022, angle=20.0, n=12), 0.0))
    parts += _drawn("main", _teardrop(0.31, 0.62) + (0.0, 0.17))
    parts += [("paper", "fill", _ellipse(-0.13, 0.12, 0.05, 0.11, angle=18.0, n=24), 0.0),
              ("paper", "fill", _ellipse(-0.14, -0.06, 0.026, 0.026, n=14), 0.0),
              ("light", "line", _arc(0.0, 0.17, 0.24, 0.24, 15.0, 75.0), 0.03)]
    return parts


# ================================================================================================= the café
def croissant():
    """A croissant from above: a fat crescent rolled up in five, the middle roll on top, its horns tapering to points
    as they curl in."""
    from .mandala import _drawn, _tilted
    rx, ry, cy, hmax, tmax = 0.48, 0.4, 0.38, 0.25, 104.0     # the curl (an ellipse about (0, cy)), its thickness

    def thick(t):                                            # half its thickness t degrees round from the top
        return hmax * np.clip(1.0 - (np.abs(t) / tmax) ** 2, 0.0, 1.0) ** 0.85 + 0.004

    def at(t, r):                                            # t degrees round the curl, r out from it (- in)
        a = np.radians(t)
        return np.stack([(rx + r) * np.sin(a), cy - (ry + r) * np.cos(a)], -1)

    v = np.linspace(0.0, 1.0, 12)[1:-1]

    def cap(t0, s):                                          # a roll's end at t0, bulging round it (s its side)
        h0 = float(thick(np.array([t0]))[0])
        return at(t0 + s * math.degrees(0.4 * h0 / rx) * np.sin(np.pi * v), s * h0 * (1.0 - 2.0 * v))

    def roll(ta, tb):                                        # the roll from ta to tb degrees, a little puffed
        ts = np.linspace(ta, tb, 24)
        h = thick(ts) * (1.0 + 0.06 * np.sin(np.pi * (ts - ta) / (tb - ta)))
        return np.concatenate([at(ts, h), cap(tb, 1.0), at(ts[::-1], -h[::-1]), cap(ta, -1.0)])

    def shine(ta, tb, k=0.03):
        tm, d = 0.5 * (ta + tb), tb - ta
        return ("light", "line", at(np.array([tm - 0.25 * d, tm + 0.15 * d]), 0.55 * thick(np.array([tm]))[0]), k)
    parts = []
    edges = (13.0, 38.0, 62.0, tmax)
    for k in range(len(edges) - 1, 0, -1):                   # the horns first, in to the middle
        for s in (-1.0, 1.0):
            ta, tb = sorted((edges[k - 1] * s, edges[k] * s))
            parts += _drawn("main", roll(ta, tb)) + [shine(ta, tb)]
    parts += _drawn("main", roll(-edges[0], edges[0])) + [shine(-2.0 * edges[0], 1.2 * edges[0], 0.035)]
    return _tilted(parts, 6.0)


def cup():
    """A latte from above: a cup on its saucer, its handle out to the side, a heart poured in the foam."""
    from .mandala import _box, _closed, _drawn, _ellipse, _heart, _tilted
    parts = _drawn("light", _ellipse(0.0, 0.0, 0.44, 0.44, n=72))                              # the saucer
    parts.append(("paper", "line", _closed(_ellipse(0.0, 0.0, 0.375, 0.375, n=72)), 0.022))    # its well
    parts += _drawn("main", _box(0.22, -0.065, 0.47, 0.065, 0.06))                              # the handle
    parts += [("light", "line", np.array([[0.33, 0.0], [0.42, 0.0]]), 0.03)]
    parts += _drawn("main", _ellipse(0.0, 0.0, 0.31, 0.31, n=64))                               # the cup
    parts += _drawn("accent_dark", _ellipse(0.0, 0.0, 0.245, 0.245, n=64), 0.016)               # the coffee
    parts.append(("paper", "fill", _heart() * 0.27, 0.0))
    return _tilted(parts, 14.0)


def sugar():
    """A sugar cube: its three faces (the top and one side white, the other in the colour's pale tint), crystals
    spilt by it, a sparkle over it."""
    from .mandala import _box, _drawn, _ellipse, _rounded, _tilted, _turned, _twinkle
    top = np.array([[0.0, -0.4], [0.35, -0.2], [0.0, 0.0], [-0.35, -0.2]])
    left = np.array([[-0.35, -0.2], [0.0, 0.0], [0.0, 0.42], [-0.35, 0.22]])
    right = np.array([[0.0, 0.0], [0.35, -0.2], [0.35, 0.22], [0.0, 0.42]])
    parts = []
    for x, y, k, a in ((-0.36, 0.38, 0.045, 20.0), (-0.47, 0.3, 0.032, -15.0)):
        parts += _drawn("paper", _turned(_box(-k, -k, k, k, 0.3 * k), a) + (x, y), 0.012)
    for role, face in (("paper", top), ("paper", left), ("light", right)):
        parts += _drawn(role, _rounded(face, 0.045))
    parts += _drawn("accent", _twinkle(0.37, -0.42, 0.14), 0.016)
    return _tilted(parts + _drawn("main", _ellipse(0.47, -0.19, 0.032, 0.032, n=14), 0.012), -6.0)


def orange():
    """An orange slice: its rind, the pith inside it, the segments round the middle and a mint leaf laid on it."""
    from .mandala import _drawn, _ellipse, _smooth, _turned
    parts = _drawn("main", _ellipse(0.0, 0.0, 0.5, 0.5, n=96))
    parts.append(("paper", "fill", _ellipse(0.0, 0.0, 0.43, 0.43, n=96), 0.0))
    k = 10
    for j in range(k):
        a0, a1 = 2.0 * math.pi * j / k, 2.0 * math.pi * (j + 1) / k
        am, g = (a0 + a1) / 2.0, 0.07
        pts = [(0.07, am), (0.22, a0 + g), (0.38, a0 + g * 0.6), (0.4, am), (0.38, a1 - g * 0.6), (0.22, a1 - g)]
        seg = _smooth([[r * math.sin(a), -r * math.cos(a)] for r, a in pts], n=6)
        parts += [("main", "fill", seg, 0.0),
                  ("light", "line", np.array([[0.2 * math.sin(am), -0.2 * math.cos(am)],
                                              [0.32 * math.sin(am), -0.32 * math.cos(am)]]), 0.022)]
    parts += _drawn("accent", _turned(_pointed(0.26, 0.075, 0.0, at=0.45), -35.0) + (0.3, -0.32), 0.018)
    return parts + [("accent_dark", "line", _turned(np.array([[0.0, -0.02], [0.0, -0.2]]), -35.0) + (0.3, -0.32),
                     0.012)]


def bean():
    """A coffee bean: an oval with its S crease down the middle."""
    from .mandala import _drawn, _ellipse, _tilted
    parts = _drawn("main", _ellipse(0.0, 0.0, 0.3, 0.45, n=64))
    ys = np.linspace(-0.37, 0.37, 30)
    parts += [("ink", "line", np.stack([0.06 * np.sin(ys / 0.37 * math.pi), ys], 1), 0.04),
              ("light", "line", _arc(0.0, 0.0, 0.22, 0.36, 200.0, 250.0, 12), 0.03)]
    return _tilted(parts, -32.0)
