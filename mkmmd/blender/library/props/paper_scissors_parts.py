"""Meshes of the paper scissors (`card = "library:paper_scissors"`; the drawing, the turn and the card are
mkmmd.core.paper_scissors): numpy only, no bpy, so tests import it. A helper module: it registers no builder.

Every card is cut with the paper theatre's scissors (paper_theatre_parts.cut: straight snips that turn a few degrees, at
this prop's size) and made a closed plate with rolled edges (electric_guitar_geo.slab; a handle with its finger hole by
paper_theatre_parts.holed_slab). A piece is its blade's grey card, its handle's coloured card glued on the blade card's
outer face, and a pencil line along the cutting edge on both faces of the blade (a band 0.1 mm thick). The pin is a
brass paper fastener at the scissors' size: a domed head on the front piece's handle, the shank through the four cards,
two legs spread along the back piece's handle (it is that piece's child, so they turn with it)."""
import math

import numpy as np

from ....core import paper_scissors as PS
from ....core import shell as S
from ..sets import paper_theatre_parts as P
from . import electric_guitar_geo as GEO

SNIP = (0.010, 0.00025, 0.0015)          # the scissors on card this small: snip (m), wobble, ring step
SNIP_HOLE = (0.005, 0.0002, 0.0012)      # round a finger hole
SPACING = 0.0025                         # the faces' fill
PENCIL_T = (0.00003, 0.0001)             # a pencil line's lift off its face and its thickness


def _flat(y):
    return lambda x, z: np.full(np.shape(x), float(y))


def plate(outline, holes, faces, mat, seed):
    """A card plate (closed) in the X-Z plane between y = faces[0] (its front) and faces[1] (its back): the outline and
    its holes cut by scissors, the edges rolled 0.35 of the card's thickness."""
    y0, y1 = faces
    ring = P.cut(outline, seed, SNIP)
    hs = [P.cut(h, seed + 1 + k, SNIP_HOLE, hole=True) for k, h in enumerate(holes)]
    r = 0.35 * (y1 - y0)
    if hs:
        return P.holed_slab(ring, hs, _flat(y0), _flat(y1), r, SPACING, mat)
    return GEO.slab(ring, _flat(y0), _flat(y1), r=r, n=2, wall=1, spacing=SPACING, mat=mat)


def pencil(side, faces):
    """The pencil lines along a blade's cutting edge, one on each of its faces."""
    inset, z0, z1, w = PS.PENCIL
    x = (1.0 if side == "a" else -1.0) * (inset - PS.EDGE)
    z = np.linspace(z0, z1, 24)
    lift, thick = PENCIL_T
    out = []
    for y, n in ((faces[0], -1.0), (faces[1], 1.0)):
        m = P.band(np.stack([np.full_like(z, x), np.full_like(z, y), z], 1), (0.0, n, 0.0), w, thick, "pencil", lift)
        m.Qm[:] = PS.MI[f"pencil_{side}"]
        m.Tm[:] = PS.MI[f"pencil_{side}"]
        out.append(m)
    return out


def piece(side):
    """Piece `a` or `b` in its own frame (the prop's frame shut): blade card, handle card, pencil lines."""
    O, lay = PS.outlines(), PS.layers()
    seed = P.seed_of("paper_scissors", side)
    blade = plate(O[f"blade_{side}"], [], lay[f"blade_{side}"], PS.MI[f"blade_{side}"], seed)
    handle = plate(O[f"handle_{side}"], [O[f"hole_{side}"]], lay[f"handle_{side}"], PS.MI[f"loop_{side}"], seed + 10)
    return S.merge([blade, handle] + pencil(side, lay[f"blade_{side}"]))


def pin():
    """The split pin in the back piece's frame: its head domed out of the front piece's handle (toward -Y) over the pin,
    the shank through to the back piece's handle and the two legs folded flat on it along +-Z."""
    R, H, rs = PS.PIN["head"], PS.PIN["dome"], PS.PIN["shank"]
    L, w, t = PS.PIN["leg"]
    lay = PS.layers()
    y_head, y_back = lay["handle_a"][0], lay["handle_b"][1]
    depth = y_back - y_head
    dome = [(R * math.cos(a), 0.0004 + (H - 0.0004) * math.sin(a)) for a in np.radians(np.linspace(0.0, 90.0, 9))]
    prof = [(0.0, -depth), (rs, -depth), (rs, -0.00012), (R - 0.0006, -0.00012), (R - 0.00012, 0.00008)] + dome
    head = S.lathe(np.array(prof), seg=32, mat=PS.MI["brad"]).rot(rx=90.0)          # its axis +z -> -y (to the front)
    legs = [S.rounded_panel(S.rrect(w, L, 0.4 * w, 3), t, r_edge=0.00012, r_back=0.0001, mat=PS.MI["brad"])
            .rot(rx=-90.0).moved((0.0, depth - 0.0001, s * (0.5 * L + rs))) for s in (1.0, -1.0)]
    return S.merge([head] + legs).moved((0.0, y_head, 0.0))


def parts():
    """{"blade_a", "blade_b": each piece in its own frame; "pin": in the back piece's frame}."""
    return {"blade_a": piece("a"), "blade_b": piece("b"), "pin": pin()}
