"""Layout of the golden spike (`card = "library:golden_spike"`; pure Python + numpy, no bpy: importable from tests): the
option, its meshes, the colour roles and the card.

Frame (metres): the origin at the TIP, the spike standing along +Z to the top of its head at z = `length`; front -Y. So
`at` a tie's top less how deep it is driven stands it in the tie. The ceremonial last spike of 1869 in proportion (5 5/8
in long, the shank 17/32 in square; read off a photograph of the spike at its museum: the head about 1.8 times the
shank's width and 4.5 mm high over a 7.5 mm flare), a cut spike's shape: a square shank with rounded arrises
(superellipse rings, SHANK_P) tapering a little toward the point, the chisel point (its two Y faces running to an edge
EDGE_T thick across X over POINT_H), the shank widening under the head in a concave taper, and the head, its edges
rolled and its top crowned. ONE cage of rings from the tip to the crown. Round the shank's middle an engraved band
(BAND_H tall, at BAND_AT of the shank) between two raised beads; the engraving itself (an ornament of interlaced waves
and dots: no lettering, no copy of the real spike's inscriptions) is the material's, cut in by a bump and darkened, so
it reads as engraving at any distance.

Option (`slots` of the prop): `length` (m, LENGTH by default, LENGTH_RANGE): the shank grows or shrinks, the section, the
point and the head keep their size and the band its share of the shank; `[[prop]] scale` scales the whole spike.

Colour roles (`slots`: a palette slot, `paper` / `ink`, a hex or a mix `a:b:t`): metal (the gold), engraving (the cut
grooves).

Card: `use.grip` `hold` (type "hold": the shank a fist may hold, from the point to the flare, `axis` +Z toward the
head, `radius` the section's (its perimeter's) radius), `use.anchor` `tip` (the origin, `dir` -Z: the way it is driven),
`use.look` `head` (the crown) and `band`."""
import math

import numpy as np

from ....core import shell as S

LENGTH = 0.1429                     # 5 5/8 in: the 1869 last spike's length
LENGTH_RANGE = (0.09, 0.40)         # `length` the shank can be stretched or shrunk to (m)
SHANK = (0.0128, 0.0135)            # the shank's width just over the point and under the flare (17/32 in)
SHANK_P = 4.5                       # the superellipse exponent of its section (flat faces, rounded arrises)
POINT_H = 0.028                     # the chisel point's length
EDGE_T = 0.0018                     # the chisel's edge, its thickness (along Y) at the tip
TIP_ROUND = 0.0006                  # the edge is rounded over this height (the cap's dome)
FLARE_H = 0.0075                    # the flare under the head: the shank widening in a concave taper
FLARE_TO = (0.0205, 0.0150)         # ... out to this width (X) and depth (Y) at the head's underside
FLARE_Q = 1.8                       # its profile: the share u ** FLARE_Q of the widening at the share u of its height
HEAD = (0.0250, 0.0195, 0.0045)     # the head: width (X), depth (Y), height
HEAD_P = 5.0                        # its outline's superellipse exponent (a rounded rectangle)
HEAD_EDGE = 0.0014                  # the radius its bottom and top edges are rolled to
CROWN = 0.0007                      # the top's dome
BAND_H, BAND_AT = 0.028, 0.56       # the engraved band's height and its middle's share of the shank (point to flare)
BEAD_R = 0.00065                    # the beads that bound it: their radius (half sunk into the shank)
RING = 48                           # points round a ring
ROLES = {"metal": "gold", "engraving": "gold:ink:0.5"}
MATS = ("gold",)                    # material index -> material (the engraving is drawn in it)
NDIGITS = 5


class LayoutError(ValueError):
    """A bad option of the golden spike."""


def _r(x):
    return round(float(x), NDIGITS)


def options(slots=None):
    """{"length": m} from the `slots` of the [[prop]] (the palette's slots and the colour roles are not options): a bad
    value is a LayoutError saying what to change."""
    v = (slots or {}).get("length", LENGTH)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not LENGTH_RANGE[0] <= v <= LENGTH_RANGE[1]:
        raise LayoutError(f"golden_spike: length is the spike's length in metres, {LENGTH_RANGE[0]:g} .. "
                          f"{LENGTH_RANGE[1]:g} (got {v!r})")
    return {"length": float(v)}


def shank_top(length=LENGTH):
    """z where the shank meets the flare."""
    return length - HEAD[2] - FLARE_H


def band(length=LENGTH):
    """(z0, z1) of the engraved band."""
    mid = POINT_H + BAND_AT * (shank_top(length) - POINT_H)
    return mid - BAND_H / 2, mid + BAND_H / 2


def shank_w(z, length=LENGTH):
    """The shank's width at heights z (the point keeps it across X)."""
    u = np.clip((np.asarray(z, float) - POINT_H) / (shank_top(length) - POINT_H), 0.0, 1.0)
    return SHANK[0] + (SHANK[1] - SHANK[0]) * u


def section(z, length=LENGTH):
    """(width along X, depth along Y, superellipse exponent) of the spike's ring at heights z (n,), from the tip up to
    the head's underside (the flare's top)."""
    z = np.asarray(z, float)
    w = shank_w(z, length)
    d = np.where(z < POINT_H, EDGE_T + (w - EDGE_T) * np.clip(z / POINT_H, 0.0, 1.0), w)
    p = np.full(z.shape, SHANK_P)
    top = shank_top(length)
    u = np.clip((z - top) / FLARE_H, 0.0, 1.0)
    f = u ** FLARE_Q                                                     # leaving the shank smoothly, a taper
    w = np.where(z > top, w + (FLARE_TO[0] - w) * f, w)
    d = np.where(z > top, d + (FLARE_TO[1] - d) * f, d)
    p = np.where(z > top, SHANK_P + (HEAD_P - SHANK_P) * u, p)
    return w, d, p


def section_at(z, length=LENGTH):
    """`section` at one height, as floats."""
    return tuple(float(v[0]) for v in section(np.array([float(z)]), length))


def _ring(w, d, p, z):
    o = S.superellipse(w, d, p, RING)
    return np.stack([o[:, 0], o[:, 1], np.full(RING, z)], 1)


def stations(length=LENGTH):
    """Heights of the rings from the tip to the flare's top: dense over the point and the flare, sparse along the shank,
    with a ring at each edge of the band."""
    top = shank_top(length)
    b0, b1 = band(length)
    z = np.concatenate([np.linspace(TIP_ROUND, POINT_H, 9), np.linspace(POINT_H, top, 12)[1:], [b0, b1],
                        np.linspace(top, top + FLARE_H, 9)[1:]])
    return np.unique(np.round(z, 9))


def spike_mesh(length=LENGTH):
    """The spike (material gold): ONE closed cage of rings, the chisel edge's rounded cap at the origin, the shank, the
    flare, the head's underside, its rolled edges and its crowned top at z = length: a smooth pillow over the rim
    (CROWN (1 - (x / a)^2) (1 - (y / b)^2) on the cap's inner rings; a dome that follows the rounded rectangle's own
    rings would crease along its diagonals)."""
    rings = []
    for z in stations(length):
        w, d, p = section_at(z, length)
        rings.append(_ring(w, d, p, z))
    hw, hd, hh = HEAD
    z0 = length - hh                                                     # the head's underside
    for a in np.linspace(0.5 * math.pi, 0.0, 5):                         # the bottom edge, rolled: underside to side
        inset = HEAD_EDGE * (1.0 - math.cos(a))
        rings.append(_ring(hw - 2 * inset, hd - 2 * inset, HEAD_P, z0 + HEAD_EDGE * (1.0 - math.sin(a))))
    zt = length - CROWN                                                  # the top's rim plane
    for a in np.linspace(0.0, 0.5 * math.pi, 5):                         # the top edge: side to top
        inset = HEAD_EDGE * (1.0 - math.cos(a))
        rings.append(_ring(hw - 2 * inset, hd - 2 * inset, HEAD_P, zt - HEAD_EDGE + HEAD_EDGE * math.sin(a)))
    m = S.cage(np.array(rings), closed=True, caps=("fan", "fan"), cap_bulge=(TIP_ROUND, CROWN), cap_rings=6,
               mat=MATS.index("gold"))
    top = m.V[:, 2] > zt + 1e-9                                          # the top cap's inner rings and its middle
    u, v = m.V[top, 0] / (0.5 * hw - HEAD_EDGE), m.V[top, 1] / (0.5 * hd - HEAD_EDGE)
    m.V[top, 2] = zt + CROWN * np.clip(1.0 - u * u, 0.0, 1.0) * np.clip(1.0 - v * v, 0.0, 1.0)
    return m


def bead_mesh(length=LENGTH):
    """The two beads bounding the band (material gold): tubes round the shank's section at the band's edges, their
    middles on its surface."""
    out = []
    for z in band(length):
        w, d, p = section_at(z, length)
        path = _ring(w, d, p, z)
        m = S.tube(path, BEAD_R, sides=10, closed=True, mat=MATS.index("gold")).weld()
        out.append(m if m.volume() > 0 else m.flip())
    return S.merge(out)


def meshes(length=LENGTH):
    """{part: Mesh} of the spike's objects (in the prop's frame)."""
    return {"spike": spike_mesh(length), "beads": bead_mesh(length)}


def perimeter(w, d, p, n=720):
    """The perimeter of a superellipse section."""
    o = S.superellipse(w, d, p, n)
    return float(np.linalg.norm(o - np.roll(o, -1, axis=0), axis=1).sum())


def hold_grip(length=LENGTH):
    """The `use.grip` entry `hold` (type "hold"): the shank from the point to the flare, its axis toward the head,
    `radius` the radius of a circle as long round as the section at the span's middle."""
    z0, z1 = POINT_H, shank_top(length)
    mid = 0.5 * (z0 + z1)
    w, d, p = section_at(mid, length)
    return {"name": "hold", "type": "hold", "center": [0.0, 0.0, _r(mid)], "axis": [0.0, 0.0, 1.0],
            "length": _r(z1 - z0), "radius": _r(perimeter(w, d, p) / (2.0 * math.pi))}


def card(name, roles, length=LENGTH):
    """The prop card (docs/design.md: Prop card). `roles` {role: spec}."""
    b0, b1 = band(length)
    use = {"grip": [hold_grip(length)],
           "anchor": [{"name": "tip", "point": [0.0, 0.0, 0.0], "dir": [0.0, 0.0, -1.0]}],
           "look": [{"name": "head", "point": [0.0, 0.0, _r(length)]},
                    {"name": "band", "point": [0.0, 0.0, _r(0.5 * (b0 + b1))]}]}
    return {"size": [_r(HEAD[0]), _r(HEAD[1]), _r(length)], "origin": "tip", "front": "-Y", "slots": dict(roles),
            "use": use, "colliders": [], "objects": [f"{name}_spike", f"{name}_beads"]}
