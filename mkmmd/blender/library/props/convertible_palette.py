"""Colour roles of the 1980s convertible (pure Python, no bpy: importable from tests).

A role is a name the car's materials colour by (`body`, `rubber`, `interior`, ...). Its colour is a palette slot name, a
'#rrggbb' hex or a blend 'slot:weight,slot:weight' mixed in linear light. A project recolours the car with
`[[prop]] slots = {body = "iris"}` (a role name) or by overriding palette slots (`{love = "#b4637a"}`); the roles'
defaults are slot names, so every palette (rose-pine-moon, rose-pine, rose-pine-dawn) recolours the car consistently.
"""
from ....core import palette as PAL

MOON = PAL.PALETTES["rose-pine-moon"]

# role -> default colour spec
ROLES = {
    "body": "love",                         # paint above the two-tone line
    "lower": "muted",                       # paint below it (the two-tone)
    "stripe": "gold",                       # the pinstripe, cassette stripes
    "trim": "text",                         # bright metal: bumper caps, window frame, mirrors, handles
    "rubber": "base",                       # black rubber: the darkest slot, never pure black
    "interior": "gold:5,rose:3,overlay:2",  # upholstery: a warm tan
    "display": "foam",                      # the digital cluster and radio displays
    "headlamp": "text",                     # headlamp light
    "taillamp": "love",                     # tail lamps
    "glass": "foam",                        # glass tint
    "wheel": "subtle",                      # alloy wheels
    "boot": "overlay:3,base:2",             # padded top boot
}


def parse(part):
    """'slot' -> ('slot', 1.0); 'slot:3' or '#rrggbb:3' -> (name, 3.0)."""
    name, sep, w = part.strip().rpartition(":")
    if sep:
        try:
            return name, float(w)
        except ValueError:
            pass
    return part.strip(), 1.0


class Pal:
    """Role colours from the palette slots (and per-prop overrides), as linear RGB."""

    def __init__(self, slots=None):
        slots = dict(slots or {})
        self.palette = {k: slots.get(k, v) for k, v in MOON.items()}
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}

    def slot_lin(self, name):
        if name.startswith("#"):
            return PAL.linear(name)
        if name not in self.palette:
            raise KeyError(f"colour {name!r} is neither a hex value nor a palette slot ({sorted(self.palette)})")
        return PAL.linear(self.palette[name])

    def lin(self, spec):
        """Linear RGB of a role name, a slot, a hex or a 'slot:weight,slot:weight' blend (weights are normalised)."""
        if spec in self.specs:
            spec = self.specs[spec]
        rgb, tot = [0.0, 0.0, 0.0], 0.0
        for part in spec.split(","):
            name, w = parse(part)
            c = self.slot_lin(name)
            for i in range(3):
                rgb[i] += c[i] * w
            tot += w
        return tuple(v / tot for v in rgb)

    def mix(self, *pairs, k=1.0):
        """Linear blend of colours (spec strings or linear tuples), each bare or as a (colour, weight) pair, times k."""
        rgb, tot = [0.0, 0.0, 0.0], 0.0
        for item in pairs:
            c, w = (item, 1.0) if isinstance(item, str) or len(item) == 3 else item      # a colour, or (colour, weight)
            c = self.lin(c) if isinstance(c, str) else c
            for i in range(3):
                rgb[i] += c[i] * w
            tot += w
        return tuple(min(max(v / tot * k, 0.0), 1.0) for v in rgb)

    def roles(self):
        """{role: colour spec} as the card's `slots` (what a project can override)."""
        return dict(self.specs)


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)
