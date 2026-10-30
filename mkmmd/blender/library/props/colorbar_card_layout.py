"""Layout of the colour-bar card (`library:colorbar_card`; pure Python + numpy, no bpy: importable from tests): its mesh,
the pictures' pixels, the options, the colour roles and the card. The pictures are EraLooks' (mkmmd.core.colorbars).

Frame (metres): origin at the card's centre, the bars face -Y: the card lies in the x-z plane, `size` = [w, h] (x across,
z up; default 0.60 x 0.45, 4:3) and `thick` along y (0.012). Its outline is a rounded TV screen (a superellipse, P), its
edges rolled on both faces (R_EDGE), the face domed DOME out. Materials: the face (the bars: `colorbars.texture` at FACE_PX,
an sRGB byte image), the back (`colorbars.back`) and the edge (cream paper). The pictures cover the card's box (uv 0..1 over
w x h, the face read from the front, the back from behind); the card's corners trim them as a TV's mask does.

Colour role (`slots`: a palette slot, a hex or a 'slot:weight,...' blend; `chalk` the lighter of the palette's base and
text): edge (chalk:5,gold:1, a cream). The bars take the palette's slots from `slots` (rose-pine-dawn's where none are
given). Options: size ([0.60, 0.45]), thick (0.012), style ("both"; "dawn", "true": colorbars.STYLES)."""
import math

import numpy as np

from ....core import colorbars as CB
from ....core import palette as PAL
from ....core import shell as S
from . import convertible_palette as CP

SIZE = (0.60, 0.45)                 # w, h (m)
THICK = 0.012
P = 5.0                             # the outline's superellipse exponent: a rounded TV screen
N_OUT = 96                          # points round the outline
R_EDGE = 0.004                      # the rolled edges, front and back
DOME = 0.0015                       # the face bulges out this much at its centre
FACE_PX = (1024, 768)               # the pictures' pixels (w, h)
STYLE = "both"
SLOTS = tuple(PAL.PALETTES["rose-pine-dawn"])       # the 15 Rose Pine slots the bars take
MATS = ("face", "back", "edge")                     # material index -> what it is
ROLES = {"edge": "chalk:5,gold:1"}                  # cream paper


class Pal(CP.Pal):
    """The card's colour role over the palette; `chalk` is the lighter of the palette's base and text."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}

    def slot_lin(self, name):
        if name == "chalk":
            a, b = (PAL.linear(self.palette[k]) for k in ("base", "text"))
            lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]      # noqa: E731
            return a if lum(a) > lum(b) else b
        return super().slot_lin(name)


def options(slots):
    """(size (w, h), thick, style) from the prop's slots, checked (the style by colorbars when the bars are drawn)."""
    slots = dict(slots or {})
    size = slots.get("size", list(SIZE))
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in size)):
        raise ValueError(f"colorbar_card: size is [width, height] in metres, got {size!r}")
    w, h = (float(v) for v in size)
    if not (0.05 <= w <= 10.0 and 0.05 <= h <= 10.0):
        raise ValueError(f"colorbar_card: size is 0.05 .. 10 m each way, got [{w:g}, {h:g}]")
    t = slots.get("thick", THICK)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        raise ValueError(f"colorbar_card: thick is metres, got {t!r}")
    t = float(t)
    if not 0.002 <= t <= 0.25 * min(w, h):
        raise ValueError(f"colorbar_card: thick is 0.002 m .. a quarter of the card's smaller side "
                         f"({0.25 * min(w, h):g}), got {t:g}")
    return (w, h), t, str(slots.get("style", STYLE))


def palette(slots):
    """The palette the bars are drawn in: {slot: '#hex'} of the 15 Rose Pine slots from the prop's slots (a slot naming
    another is followed; rose-pine-dawn's for the slots not given), or None (rose-pine-dawn) when none is given."""
    slots = dict(slots or {})
    if not any(k in slots for k in SLOTS):
        return None
    dawn = PAL.PALETTES["rose-pine-dawn"]
    out = {}
    for k in SLOTS:
        v, seen = str(slots.get(k, dawn[k])), {k}
        while not v.startswith("#"):
            if v not in SLOTS or v in seen:
                raise ValueError(f"colorbar_card: palette slot {k} = {slots.get(k)!r}: give a '#rrggbb' hex or another slot")
            seen.add(v)
            v = str(slots.get(v, dawn[v]))
        if len(v) != 7 or any(c not in "0123456789abcdefABCDEF" for c in v[1:]):
            raise ValueError(f"colorbar_card: palette slot {k} = {v!r} is not a '#rrggbb' hex")
        out[k] = v
    return out


def outline(size):
    """The card's outline (N_OUT, 2) of (x, z), counter-clockwise: a rounded TV screen filling w x h."""
    return S.superellipse(size[0], size[1], P, N_OUT)


def mesh(size, thick):
    """The card: a plate cut to `outline`, its face (material `face`) toward -Y at y = -thick / 2 (domed DOME further),
    its back (`back`) at +thick / 2, the rolled rim between (`edge`). uv per vertex: the face's over w x h as seen from the
    front (u right, v up), the back's as seen from behind."""
    w, h = size
    m = S.rounded_panel(outline(size), thick, r_edge=R_EDGE, n=3, dome=DOME, rings=4, r_back=R_EDGE)
    m = m.rot(rx=90.0).moved((0.0, 0.5 * thick, 0.0))          # the panel's face (+z) -> -y, its back at y = thick / 2
    u = (m.V[:, 0] + 0.5 * w) / w
    v = (m.V[:, 2] + 0.5 * h) / h
    m.UV = np.stack([np.where(m.V[:, 1] > 0.0, 1.0 - u, u), v], 1)
    m.Qm[:] = MATS.index("edge")
    m.Tm[:] = MATS.index("edge")
    m.assign(MATS.index("face"), lambda c, n: n[:, 1] < -0.7)
    m.assign(MATS.index("back"), lambda c, n: n[:, 1] > 0.7)
    return m


def pictures(size, pal, style):
    """(face, back): EraLooks' pictures at FACE_PX, (h, w, 3) float32 display RGB 0..1, row 0 the top."""
    w, h = FACE_PX
    return CB.texture(w, h, palette=pal, style=style), CB.back(w, h, palette=pal)


def pixels(rgb):
    """The flat RGBA float32 buffer of a Blender image (rows bottom-up, alpha 1) from a picture whose row 0 is its top."""
    h, w = rgb.shape[:2]
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = rgb[::-1]
    return out.ravel()


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def card(name, size, thick, style, roles=None):
    """The prop card (docs/design.md: Prop card): `use.surface` `face` (the printed picture's rectangle on the face) and
    `use.look` `centre`."""
    w, h = size
    face = [0.0, -0.5 * thick, 0.0]
    return {"size": _r([w, thick, h], 4), "origin": "center", "front": "-Y", "style": style, "slots": dict(roles or {}),
            "use": {"surface": [{"name": "face", "center": face, "normal": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0],
                                 "size": _r([w, h], 4)}],
                    "look": [{"name": "centre", "point": face}]},
            "colliders": [], "params": {}, "objects": [f"{name}_card"]}
