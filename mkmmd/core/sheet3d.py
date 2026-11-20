"""What the physical cuts share, bpy-free (docs/design.md: Transitions and inserts): the outgoing picture as a sheet that
comes off the frame plane (`peel`, `shatter`, `tear`; mkmmd.core.peel, mkmmd.core.shatter, mkmmd.core.tear). Pixel
coordinates are index coordinates of the frame (x right, y down, pixel (i, j) centred on x = j, y = i); a height z is in
px toward the lens. The camera looks straight down the frame's middle from `distance` px: a point (x, y) at height z is
seen at C + ((x, y) - C) * D / (D - z), so a lifted piece grows as it comes toward the lens. The lamp is the page turn's,
at the upper left in front: shadows fall to the lower right, offset by the height of what casts them."""
import math

LENS = 2.5                       # the camera's distance from the frame plane, in the frame's long sides
LIGHT = (-0.45, -0.55, 0.70)    # toward the lamp (x right, y down, z toward the lens), unit below
_n = math.sqrt(sum(v * v for v in LIGHT))
LIGHT = tuple(v / _n for v in LIGHT)
HALF = tuple(v / math.sqrt(2.0 * (1.0 + LIGHT[2])) for v in (LIGHT[0], LIGHT[1], LIGHT[2] + 1.0))   # light <-> eye
SHADOW_RUN = (-LIGHT[0] / LIGHT[2], -LIGHT[1] / LIGHT[2])   # how far a shadow falls per px of height (x, y)


def distance(size):
    """D: the camera's distance from the frame plane in px for a frame `size` = (w, h)."""
    return LENS * max(size)


def centre(size):
    """C: the point of the frame the camera looks at (its middle), index coordinates."""
    return ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)


def project(x, y, z, size):
    """Where the point (x, y) at height z is seen (px), and the scale it is seen at there (D / (D - z))."""
    D = distance(size)
    cx, cy = centre(size)
    m = D / (D - z)
    return cx + (x - cx) * m, cy + (y - cy) * m, m


def hermite(u, p0, p1, m0, m1):
    """The cubic from p0 (slope m0) at u = 0 to p1 (slope m1) at u = 1."""
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1)


HAND = ((0.0, 0.0, 0.5), (0.53, 0.06, 0.12), (1.0, 0.1, 0.55))   # the pinch's keys (share of it, p, dp/du at 0.34)


def hand(u, pinch=0.2):
    """A hand's pull, 0..1 at u = 0..1: the pinch (the first `pinch` of the time) lifts the corner at once and holds it
    under tension (10 % of the way by its end, through the keys HAND), then the pull speeds up into a flick (at the end
    2.4 times the pull's mean speed). A cubic between keys with the slopes given: monotone and smooth."""
    u = 0.0 if u <= 0.0 else 1.0 if u >= 1.0 else float(u)
    keys = [(a * pinch, p, m * 0.34 / pinch) for a, p, m in HAND] + [(1.0, 1.0, 2.4 * 0.9 / (1.0 - pinch))]
    for (u0, p0, m0), (u1, p1, m1) in zip(keys[:-1], keys[1:]):
        if u <= u1:
            du = u1 - u0
            return hermite((u - u0) / du, p0, p1, m0 * du, m1 * du)
    return 1.0


def darkest(palette):
    """The palette's darkest colour by luminance (hex): Dawn's `text`, the ink every shadow is made of."""
    from . import palette as PAL

    def luminance(hexstr):
        r, g, b = PAL.linear(hexstr)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return min(palette.values(), key=luminance)
