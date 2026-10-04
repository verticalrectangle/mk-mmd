"""Palettes (docs/design.md: Palettes): named colour slots shared by sets, props, lights and the grade.

A project picks one with [look] palette = "rose-pine-moon" and may override slots with [look.slots]. Builders colour
by slot name (base, surface, overlay, muted, subtle, text, love, gold, rose, pine, foam, iris, hl_low, hl_med,
hl_high) so a scene can be re-coloured by switching palettes."""

PALETTES = {
    "rose-pine-moon": {"base": "#232136", "surface": "#2a273f", "overlay": "#393552", "muted": "#6e6a86",
                       "subtle": "#908caa", "text": "#e0def4", "love": "#eb6f92", "gold": "#f6c177",
                       "rose": "#ea9a97", "pine": "#3e8fb0", "foam": "#9ccfd8", "iris": "#c4a7e7",
                       "hl_low": "#2a283e", "hl_med": "#44415a", "hl_high": "#56526e"},
    "rose-pine": {"base": "#191724", "surface": "#1f1d2e", "overlay": "#26233a", "muted": "#6e6a86",
                  "subtle": "#908caa", "text": "#e0def4", "love": "#eb6f92", "gold": "#f6c177", "rose": "#ebbcba",
                  "pine": "#31748f", "foam": "#9ccfd8", "iris": "#c4a7e7", "hl_low": "#21202e", "hl_med": "#403d52",
                  "hl_high": "#524f67"},
    "rose-pine-dawn": {"base": "#faf4ed", "surface": "#fffaf3", "overlay": "#f2e9e1", "muted": "#9893a5",
                       "subtle": "#797593", "text": "#575279", "love": "#b4637a", "gold": "#ea9d34",
                       "rose": "#d7827e", "pine": "#286983", "foam": "#56949f", "iris": "#907aa9",
                       "hl_low": "#f4ede8", "hl_med": "#dfdad9", "hl_high": "#cecacd"},
}


def get(name="rose-pine-moon", overrides=None):
    if name not in PALETTES:
        raise KeyError(f"palette {name!r} (have {sorted(PALETTES)})")
    p = dict(PALETTES[name])
    p.update(overrides or {})
    return p


def srgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def linear(hexstr):
    """'#rrggbb' -> linear-light RGB (what Blender colour inputs take)."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb(hexstr))


def mix(a, b, t):
    """Mix two hex colours in sRGB, returns hex."""
    ca, cb = srgb(a), srgb(b)
    return "#" + "".join(f"{round((x * (1 - t) + y * t) * 255):02x}" for x, y in zip(ca, cb))


def resolve(colour, palette):
    """A slot name or a hex string -> hex string."""
    if colour.startswith("#"):
        return colour
    if colour not in palette:
        raise KeyError(f"colour {colour!r} is neither a hex value nor a palette slot ({sorted(palette)})")
    return palette[colour]
