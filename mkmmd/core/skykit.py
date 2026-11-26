"""The sky sticker kit (docs/design.md: Mandalas and trails), drawn with the pen's ink line round its parts as the other
kits in mkmmd.core.mandala are; mandala registers it (SHAPES, PENNED, its colours) and holds the helpers it draws with.
Each function gives its shape's parts, [(role, "fill" or "line", (n, 2) points, line width)] about (0, 0), x right and
y down, in the order they are painted (mandala.shape_parts scales the box to 1).

    sky    a paper plane, a bird perched on a wire, a bird flying at us with its wings up, a cumulus cloud, a sun,
           a kite, a balloon, a shooting star and a feather
"""
import math

import numpy as np

from .pondkit import _arc, _cubic


def _fan(root, angles, lengths, half, role, width):
    """Feathers fanned from `root`: one a pointed oval per angle (degrees out from straight up, clockwise), drawn in
    `role` with the pen's line `width`, the first under the next."""
    from .mandala import _drawn, _ellipse, _turned
    parts = []
    for a, length in zip(angles, lengths):
        parts += _drawn(role, _turned(_ellipse(0.0, -length / 2.0, half, length / 2.0, n=36), -a) + root, width)
    return parts


def _centred(parts):
    """The parts moved so that their box's middle (lines' widths in) is (0, 0): what a ring turns its copy about."""
    from .mandala import shape_box
    lo, hi = shape_box(parts)
    c = (lo + hi) / 2.0
    return [(role, kind, np.asarray(p, float) - c, w) for role, kind, p, w in parts]


def plane():
    """A folded paper dart in three-quarter view, flying up and to the right away from us: its far wing, the keel under
    it, the near wing with a stripe across it, the folds creased; a dashed swoop behind it."""
    from .mandala import _curve, _drawn, _tilted
    e, yaw, dih = math.radians(40.0), math.radians(35.0), math.radians(15.0)   # seen from above, the nose turned away
    span, deep = 0.52, 0.3                                   # a wing's span, the keel's depth at the tail

    def seen(p):                                             # the dart's frame (x to the nose, y left, z up) on screen
        p = np.asarray(p, float)
        x = p[:, 0] * math.cos(yaw) - p[:, 1] * math.sin(yaw)
        y = p[:, 0] * math.sin(yaw) + p[:, 1] * math.cos(yaw)
        return np.stack([x, -(y * math.sin(e) + p[:, 2] * math.cos(e))], 1)

    def edge(x, s):                                          # wing s's leading edge, x along the dart (tail 0, nose 1)
        return (x, s * span * math.cos(dih) * (1.0 - x), span * math.sin(dih) * (1.0 - x))
    nose, tail = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    back = seen([tail])[0]
    swoop = _curve(back + (-0.03, 0.05), back + (-0.22, 0.28), back + (-0.48, 0.16), n=31)
    parts = [("ink", "line", swoop[i:i + 6], 0.024) for i in (5, 14, 23)]
    parts += _drawn("light", seen([nose, tail, edge(0.0, 1.0)]))                    # the far wing, the keel: behind
    parts.append(("dark", "line", seen([(0.85, 0.0, 0.0), edge(0.06, 0.5)]), 0.014))
    parts += _drawn("dark", seen([nose, tail, (0.08, 0.0, -deep)]))
    wing = _drawn("main", seen([nose, tail, edge(0.0, -1.0)]))
    stripe = seen([(0.1, 0.0, 0.0), edge(0.1, -1.0), edge(0.2, -1.0), (0.2, 0.0, 0.0)])
    parts += wing[:1] + [("accent", "fill", stripe, 0.0)] + wing[1:]
    parts.append(("dark", "line", seen([(0.85, 0.0, 0.0), edge(0.06, -0.5)]), 0.014))
    return _centred(_tilted(parts, 15.0))


def bird():
    """A round little bird perched on a wire, from the side: its tail fanned up behind, its belly pale, a wing folded
    on its side, a beak, an eye with a glint, a tuft on its head; its legs down to the wire, its toes round it."""
    from .mandala import _curve, _drawn, _ellipse, _rounded
    wire = _curve([-0.5, 0.37], [0.0, 0.42], [0.5, 0.35], n=24)
    parts = [("ink", "line", wire, 0.04)]
    parts += _fan(np.array([-0.2, 0.04]), (-58.0, -76.0, -94.0), (0.3, 0.33, 0.27), 0.055, "dark", 0.018)
    for x in (-0.04, 0.09):                                   # the legs, the toes round the wire
        leg = np.array([[x, 0.2], [x + 0.01, 0.37]])
        toes = _curve([x - 0.045, 0.41], [x - 0.005, 0.33], [x + 0.06, 0.39], n=8)
        parts += [("ink", "line", leg, 0.046), ("accent", "line", leg, 0.02),
                  ("ink", "line", toes, 0.04), ("accent", "line", toes[1:-1], 0.016)]
    tuft = [_curve([0.02, -0.25], [0.0, -0.36], [0.07, -0.37], n=8),
            _curve([0.06, -0.25], [0.09, -0.33], [0.14, -0.31], n=8)]
    parts += [("ink", "line", t, 0.022) for t in tuft]
    parts += _drawn("main", _ellipse(0.0, 0.0, 0.29, 0.26, n=72))
    parts += [("light", "fill", _ellipse(0.1, 0.1, 0.16, 0.12, angle=-25.0, n=48), 0.0)]
    beak = _rounded(np.array([[0.26, -0.11], [0.4, -0.05], [0.27, 0.0]]), 0.02)
    parts += _drawn("accent", beak, 0.016)
    top = _cubic([0.05, -0.03], [-0.04, -0.12], [-0.22, -0.06], [-0.34, 0.1], 18)
    under = _cubic([-0.34, 0.1], [-0.2, 0.16], [-0.04, 0.14], [0.05, -0.03], 16)
    wing = np.concatenate([top, under[1:-1]])
    parts += _drawn("dark", wing, 0.018)
    parts += [("light", "line", _cubic([-0.02, 0.07], [-0.1, 0.1], [-0.18, 0.1], [-0.25, 0.07], 10), 0.016)]
    parts += [("ink", "fill", _ellipse(0.155, -0.1, 0.033, 0.038, n=20), 0.0),
              ("paper", "fill", _ellipse(0.145, -0.113, 0.012, 0.012, n=10), 0.0),
              ("light", "fill", _ellipse(0.2, -0.02, 0.04, 0.025, n=16), 0.0),
              ("light", "line", _arc(0.0, 0.0, 0.22, 0.19, 205.0, 250.0, 12), 0.03)]
    return parts


def bird_up():
    """A little bird flying at us, its wings raised in a V, each a fan of feathers; its tail fanned under it, its feet
    tucked, its eyes, beak, cheeks and pale belly."""
    from .mandala import _drawn, _ellipse, _mirrored, _rounded
    wings = _fan(np.array([0.11, 0.06]), (64.0, 47.0, 31.0, 16.0), (0.3, 0.4, 0.44, 0.38), 0.066, "main", 0.018)
    feet = []
    for s in (-1.0, 1.0):
        foot = np.array([[0.05 * s, 0.26], [0.07 * s, 0.33]])
        feet += [("ink", "line", foot, 0.042), ("accent", "line", foot, 0.018)]
    parts = feet + _fan(np.array([0.0, 0.2]), (160.0, 180.0, 200.0), (0.16, 0.18, 0.16), 0.04, "dark", 0.016)
    parts += _mirrored(wings)
    parts += _drawn("main", _ellipse(0.0, 0.09, 0.17, 0.2, n=64))
    parts += [("light", "fill", _ellipse(0.0, 0.16, 0.11, 0.1, n=40), 0.0)]
    for s in (-1.0, 1.0):
        parts += [("ink", "fill", _ellipse(0.065 * s, 0.0, 0.026, 0.03, n=16), 0.0),
                  ("paper", "fill", _ellipse(0.065 * s - 0.008, -0.01, 0.009, 0.009, n=10), 0.0),
                  ("light", "fill", _ellipse(0.105 * s, 0.05, 0.032, 0.02, n=14), 0.0)]
    beak = _rounded(np.array([[-0.035, 0.035], [0.035, 0.035], [0.0, 0.1]]), 0.012)
    return parts + _drawn("accent", beak, 0.014)


def _billow(bumps, origin, foot, n=240):
    """The outline of round bumps [(x, y, rx, ry)] cut flat along y = foot: from `origin` (inside every bump's reach)
    each ray out to where it leaves the last bump it is in, or meets the foot."""
    o = np.asarray(origin, float)
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    u = np.stack([np.cos(a), np.sin(a)], 1)
    reach = np.zeros(n)
    for cx, cy, rx, ry in bumps:
        q, v = (o - (cx, cy)) / (rx, ry), u / (rx, ry)
        A, B, C = (v * v).sum(1), 2.0 * (q * v).sum(1), float(q @ q) - 1.0
        disc = B * B - 4.0 * A * C
        reach = np.maximum(reach, np.where(disc >= 0.0, (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A), 0.0))
    down = u[:, 1] > 1e-6
    reach[down] = np.minimum(reach[down], (foot - o[1]) / u[down, 1])
    return o + u * reach[:, None]


def cumulus():
    """A fluffy cumulus heaped up in puffs on a flat foot, no face: a shine on its top puff, the others' tops lit."""
    from .mandala import _drawn, _ellipse
    puffs = ((-0.38, 0.14, 0.12), (-0.24, 0.0, 0.17), (-0.03, -0.17, 0.22), (0.17, -0.21, 0.15), (0.29, -0.02, 0.17),
             (0.41, 0.14, 0.11))
    foot = 0.28
    parts = _drawn("main", _billow([(x, y, r, r) for x, y, r in puffs] + [(0.01, 0.18, 0.47, 0.12)], (0.0, 0.1), foot))
    parts += [("light", "line", _arc(-0.03, -0.17, 0.16, 0.16, 200.0, 255.0), 0.034),
              ("paper", "fill", _ellipse(-0.13, -0.3, 0.022, 0.022, n=12), 0.0)]
    parts += [("light", "line", _arc(x, y, 0.7 * r, 0.7 * r, a, a + 40.0, 10), 0.026)
              for (x, y, r), a in zip(puffs[:2] + puffs[3:], (200.0, 200.0, 250.0, 250.0, 250.0))]
    parts.append(("light", "line", np.array([[-0.26, foot - 0.055], [0.26, foot - 0.055]]), 0.026))
    return _centred(parts)


def sun():
    """A round sun, its rays round it long and short in turn, a shine on it and a glint."""
    from .mandala import _box, _drawn, _ellipse, _turned
    parts = []
    for j in range(12):
        ray = _box(-0.036, -(0.5 if j % 2 == 0 else 0.44), 0.036, -0.35, 0.036)
        parts += _drawn("accent", _turned(ray, 30.0 * j), 0.018)
    parts += _drawn("main", _ellipse(0.0, 0.0, 0.3, 0.3, n=80))
    parts += [("light", "line", _arc(0.0, 0.0, 0.21, 0.21, 200.0, 250.0), 0.036),
              ("paper", "fill", _ellipse(-0.09, -0.17, 0.024, 0.024, n=12), 0.0)]
    return parts


def _bow(c, a, k):
    """A bow tied on a string at c, turned `a` degrees, its loops k out each side."""
    from .mandala import _drawn, _ellipse, _rounded, _turned
    loop = _rounded(np.array([[0.0, 0.0], [k, -0.65 * k], [k, 0.65 * k]]), 0.3 * k)
    parts = []
    for s in (-1.0, 1.0):
        parts += _drawn("accent", _turned(loop * (s, 1.0), a) + c, 0.014)
    return parts + _drawn("accent", _ellipse(c[0], c[1], 0.28 * k, 0.28 * k, n=14), 0.012)


def kite():
    """A diamond kite: four panels in two colours, its cross-spars, its tail swinging down with bows on it."""
    from .mandala import _closed, _tilted
    top, left, right, foot = np.array([[0.0, -0.52], [-0.33, -0.2], [0.33, -0.2], [0.0, 0.32]])
    cross = np.array([0.0, -0.2])
    tail = _cubic(foot, [0.06, 0.48], [-0.22, 0.46], [-0.24, 0.66], 31)
    parts = [("ink", "line", tail, 0.024)]
    for i in (10, 22):
        d = tail[i + 1] - tail[i - 1]
        parts += _bow(tail[i], math.degrees(math.atan2(d[0], d[1])), 0.085)
    for role, tri in (("main", (top, left, cross)), ("light", (top, cross, right)), ("light", (left, foot, cross)),
                      ("main", (cross, foot, right))):
        parts.append((role, "fill", np.array(tri), 0.0))
    parts += [("dark", "line", np.array([top + (0.0, 0.04), foot - (0.0, 0.04)]), 0.02),
              ("dark", "line", np.array([left + (0.04, 0.0), right - (0.04, 0.0)]), 0.02),
              ("ink", "line", _closed(np.array([top, left, foot, right])), 0.022),
              ("light", "line", np.array([top + (-0.07, 0.12), top + (-0.17, 0.27)]), 0.03)]
    return _tilted(parts, -18.0)


def balloon():
    """A round toy balloon, its knot under it, a shine on it and its string waving down."""
    from .mandala import _drawn, _ellipse, _rounded, _tilted
    t = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    y = -np.cos(t)
    body = np.stack([0.3 * np.sin(t) * (1.0 - 0.12 * np.clip(y, 0.0, 1.0)), -0.16 + 0.34 * y], 1)
    string = _cubic([0.0, 0.24], [0.12, 0.36], [-0.12, 0.46], [0.03, 0.62], 32)
    parts = [("ink", "line", string, 0.022)]
    parts += _drawn("main", _rounded(np.array([[0.0, 0.15], [0.06, 0.26], [-0.06, 0.26]]), 0.018), 0.018)
    parts += _drawn("main", body)
    parts += [("paper", "fill", _ellipse(-0.13, -0.27, 0.045, 0.09, angle=-28.0, n=24), 0.0),
              ("paper", "fill", _ellipse(-0.17, -0.12, 0.018, 0.018, n=12), 0.0),
              ("light", "line", _arc(0.0, -0.16, 0.24, 0.27, 20.0, 70.0), 0.03)]
    return _tilted(parts, -8.0)


def shooting_star():
    """A puffy five-pointed star shooting up and to the right, three streaks trailing behind it, a spark by it."""
    from .mandala import _curve, _drawn, _ellipse, _rounded, _turned, _twinkle
    head = np.array([0.2, -0.2])
    back, side = np.array([-0.82, 0.57]), np.array([0.57, 0.82])     # behind its flight; across it, down
    parts = []
    for off, length, role in ((-0.11, 0.5, "light"), (0.0, 0.74, "accent"), (0.11, 0.48, "main")):
        p0 = head + side * off
        spine = _curve(p0, p0 + back * 0.5 * length + side * 0.06, p0 + back * length, n=20)
        parts += [("ink", "line", spine, 0.076), (role, "line", spine, 0.04)]
    a = np.radians(np.arange(10) * 36.0)
    rad = np.where(np.arange(10) % 2 == 0, 0.28, 0.14)
    star = _rounded(np.stack([rad * np.sin(a), -rad * np.cos(a)], 1), 0.032)
    parts += _drawn("main", _turned(star, -14.0) + head)
    parts += [("light", "fill", _ellipse(head[0] - 0.05, head[1] - 0.06, 0.025, 0.055, angle=-30.0, n=20), 0.0)]
    return _centred(parts + _drawn("main", _twinkle(-0.22, -0.3, 0.085), 0.014))


def feather():
    """A feather: its vane round the shaft, wider on one side and split twice, its barbs drawn; the quill below and
    wisps of down at the vane's foot."""
    from .mandala import _closed, _curve, _smooth, _tilted
    vane = _smooth([[0.0, 0.3], [0.13, 0.2], [0.19, -0.02], [0.16, -0.26], [0.08, -0.43], [0.0, -0.49],
                    [-0.07, -0.42], [-0.12, -0.25], [-0.13, 0.0], [-0.08, 0.2]], n=12)

    def shaft_x(y):                                           # the shaft leans a little to the right as it goes up
        return 0.02 * (0.3 - y) / 0.8
    for i in (24, 34, 92):                                    # splits in the vane, two on its right and one on its
        y = vane[i][1]                                        # left, cut in toward the shaft and down
        vane[i] = vane[i] + (np.array([shaft_x(y), y + 0.07]) - vane[i]) * 0.55
    parts = [("main", "fill", vane, 0.0)]
    half = vane[vane[:, 0] < shaft_x(vane[:, 1])]            # the left of the vane, paler
    parts.append(("light", "fill", np.concatenate([half, [[shaft_x(0.3), 0.3]]]), 0.0))
    for y in np.linspace(0.16, -0.3, 6):                      # the barbs, out and up from the shaft
        for s, reach in ((1.0, 0.13), (-1.0, 0.09)):
            k = 1.0 - ((y + 0.1) / 0.42) ** 2
            parts.append(("dark", "line", np.array([[shaft_x(y), y], [shaft_x(y) + s * reach * k, y - 0.09 * k]]),
                          0.012))
    parts.append(("ink", "line", _closed(vane), 0.022))
    for s in (-1.0, 1.0):                                     # wisps of down at its foot, curling up
        for y0, c, tip in ((0.27, (0.06, 0.31), (0.11, 0.29)), (0.31, (0.05, 0.37), (0.1, 0.36))):
            wisp = _curve([shaft_x(y0) + 0.01 * s, y0], (c[0] * s, c[1]), (tip[0] * s, tip[1]), n=10)
            parts += [("ink", "line", wisp, 0.032), ("light", "line", wisp[2:-1], 0.012)]
    shaft = _cubic([0.0, 0.5], [0.0, 0.3], [0.01, -0.1], [shaft_x(-0.44), -0.44], 24)
    parts += [("ink", "line", shaft, 0.04), ("paper", "line", shaft[2:-2], 0.015)]
    return _tilted(parts, -32.0)
