"""Palette-slot colours of the cafe props and sets (pure Python, no bpy: importable from tests).

Every colour is a blend of palette slots mixed in linear light: `Colors.blend(gold=.6, base=.4)` (weights are normalised;
`k` scales the brightness, `hue` (deg) and `chroma` tilt it in OKLab, `a` is the alpha), `Colors.slot("pine")` is one
slot, `Colors.light(...)` a light tint (a blend normalised to its brightest channel). `Colors.hex(spec)` resolves a colour
spec as the rest of mk does (core.shotstyle.resolve_colour): a slot, a `#hex` or a mix `a:b:t` of two such; `paper` and
`ink` name the palette's lightest and darkest neutral slot (white paint, black iron on any palette)."""
import math

from ....core import palette as PAL

DAWN = PAL.PALETTES["rose-pine-dawn"]
NEUTRAL = ("base", "surface", "overlay", "text")     # the slots `paper` (the lightest) and `ink` (the darkest) pick from


def _to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (max(v, 0.0) ** (1.0 / 3.0) for v in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _from_oklab(L, a, b):
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


class Colors:
    """Palette slots -> linear RGBA colours for Blender colour sockets."""

    def __init__(self, palette=None):
        self.p = dict(DAWN)
        for k, v in (palette or {}).items():
            if k in DAWN:
                self.p[k] = v
        self.used = set()

    def slot(self, name, a=1.0):
        if name not in self.p:
            raise KeyError(f"palette slot {name!r} (have {sorted(self.p)})")
        self.used.add(name)
        return (*PAL.linear(self.p[name]), a)

    def blend(self, k=1.0, hue=0.0, chroma=1.0, a=1.0, **w):
        """Weighted mix of slots in linear light (weights are normalised), scaled by `k`, optionally rotated by `hue`
        degrees and scaled by `chroma` in OKLab. -> linear RGBA."""
        tot = sum(w.values()) or 1.0
        rgb = [0.0, 0.0, 0.0]
        for name, wt in w.items():
            c = self.slot(name)
            for i in range(3):
                rgb[i] += c[i] * wt / tot
        if hue or chroma != 1.0:
            L, aa, bb = _to_oklab(*rgb)
            ch, h = math.hypot(aa, bb) * chroma, math.atan2(bb, aa) + math.radians(hue)
            rgb = list(_from_oklab(L, ch * math.cos(h), ch * math.sin(h)))
        return tuple(min(max(v * k, 0.0), 1.0) for v in rgb) + (a,)

    def light(self, k=1.0, **w):
        """Tint of a light: the blend of slots normalised so its brightest channel is 1 (only the hue and saturation
        of the mix matter, so the light keeps its strength whatever the palette's lightness). `k` scales it. -> RGBA."""
        r, g, b, _ = self.blend(**w)
        m = max(r, g, b) or 1.0
        return (r / m * k, g / m * k, b / m * k, 1.0)

    def resolved(self):
        """{slot: '#rrggbb'} of the slots used so far (the card's `slots`)."""
        return {n: self.p[n] for n in sorted(self.used)}

    def neutral(self, which):
        """The palette slot that `paper` (the lightest of NEUTRAL) or `ink` (the darkest) names in this palette: text and
        base in the dark Rose Pines, surface and text in Dawn."""
        def lum(name):
            r, g, b = PAL.linear(self.p[name])
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (max if which == "paper" else min)(NEUTRAL, key=lum)

    def hex(self, spec):
        """'#rrggbb' of a colour spec: a palette slot, `paper` / `ink` (Colors.neutral), a '#hex' or a mix 'a:b:t' of two
        such, the second weighing t (0..1), mixed in sRGB as core.palette.mix (and core.shotstyle.resolve_colour). The
        slots it names count as used."""
        parts = str(spec).split(":")
        if len(parts) == 3:
            try:
                t = float(parts[2])
            except ValueError:
                raise ValueError(f"colour {spec!r}: bad mix weight") from None
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"colour {spec!r}: the mix weight must be in 0..1")
            return PAL.mix(self.hex(parts[0]), self.hex(parts[1]), t)
        if len(parts) != 1 or not parts[0]:
            raise ValueError(f"colour {spec!r}: expected a palette slot, 'paper', 'ink', '#hex' or 'a:b:t'")
        name = parts[0]
        if name.startswith("#"):
            return name
        if name in ("paper", "ink"):
            name = self.neutral(name)
        self.slot(name)
        return self.p[name]


def mix(c1, c2, t):
    """Linear blend of two linear RGBA tuples."""
    return tuple(a * (1 - t) + b * t for a, b in zip(c1, c2))


def shade(c, k):
    """Scale the brightness of a linear RGBA tuple (alpha kept)."""
    return (*(min(max(v * k, 0.0), 1.0) for v in c[:3]), c[3] if len(c) > 3 else 1.0)
