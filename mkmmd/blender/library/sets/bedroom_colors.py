"""Palette-slot colours of the bedroom set (pure Python, no bpy: importable from tests).

Every colour is a blend of palette slots mixed in linear light (props/cafe_colors.py: `Colors.blend`, `Colors.light`);
`spec["colors"]` overrides single roles with a slot name or a "#rrggbb" value. The set is designed for a dark palette
(rose-pine-moon): walls, floor and lights are dusty and low in value, the 80s comes from the pastel trim, the teal door
and the pink and violet accents."""
from ....core.palette import linear
from ..props.cafe_colors import Colors, mix


def roles(C, over=None):
    """{role: linear RGBA} for the set; light tints are normalised to their brightest channel (hue only)."""
    b, L = C.blend, C.light
    paper = b(overlay=.52, muted=.28, iris=.20)
    out = {
        # walls and ceiling
        "paper": paper,                                              # wallpaper: the wide stripe
        "paper_light": b(muted=.46, overlay=.32, iris=.22),          # the narrow lighter stripe
        "paper_pin": b(rose=.32, muted=.68),                         # the pin stripe between them
        "paper_motif": b(iris=.4, muted=.6),                         # the small diamonds in the light stripe
        "frieze": mix(b(overlay=.7, muted=.3), b(iris=1), 0.12),     # plain band above the picture rail
        "ceiling": b(subtle=.42, muted=.38, text=.20, k=.64),
        "plaster": b(muted=.5, overlay=.3, iris=.2),                 # reveals and anything else painted wall colour
        "outside": b(base=1),                                        # the faces nobody sees
        # paint and wood
        "trim": b(text=.58, subtle=.30, gold=.08, rose=.04),         # skirting, casings, window frame
        "trim_dark": b(subtle=.55, muted=.35, overlay=.10),          # the shadow side of mouldings
        "door": b(foam=.30, pine=.30, text=.20, overlay=.20),        # a pastel teal door
        "door_dark": b(pine=.5, overlay=.5),
        # floor
        "floor_a": b(gold=.14, rose=.18, overlay=.48, base=.20),
        "floor_b": b(gold=.22, rose=.20, overlay=.42, base=.16),
        "floor_groove": b(base=.6, overlay=.3, gold=.1, k=.55),
        # window, blind
        "glass": L(text=.55, foam=.35, iris=.10),                    # a faint cool tint, nearly clear
        "slat": b(text=.50, subtle=.34, iris=.16),
        "string": b(text=.5, subtle=.5),
        "cord": b(subtle=.6, rose=.2, text=.2),
        "metal": b(gold=.30, subtle=.50, text=.20),
        # fixtures
        "plate": b(text=.62, gold=.14, subtle=.24),
        "rocker": b(text=.5, subtle=.3, gold=.2),
        "insert": b(overlay=.7, muted=.3),
        "bulb": b(text=.7, gold=.3),
        "neon": b(love=.82, rose=.18),
        "neon_tube": b(text=.6, love=.3, rose=.1),
        # light tints (hue and saturation only; the strength is set by the light)
        "key": L(iris=.42, foam=.38, rose=.20),
        "glow": L(iris=.40, foam=.42, rose=.18),
        "fill": L(iris=.38, pine=.62),
        "neon_light": L(love=.85, iris=.15),
        "world": b(iris=.3, pine=.2, base=.5),
    }
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"bedroom_80s colors: no role {k!r} (have {sorted(out)})")
        out[k] = (*linear(v), 1.0) if str(v).startswith("#") else C.slot(v)
    return out


__all__ = ["Colors", "roles"]
