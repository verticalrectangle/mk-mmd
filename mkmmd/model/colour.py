"""Colour maths the part builders share (numpy and colorsys only: no Pillow, Python 3.11 / numpy 1.24 safe).

A part's tuned colours often go with one palette: Rin's crown sheen with her red hair, the green-tinted shadow of her
green frills. When a character changes the palette, these follow it:

  follow(c, now, was)        a colour that goes with `was` moved the way `now` differs from it: hue turned by theirs,
                             saturation and value scaled by theirs
  follow_tint(m, now, was)   a multiplier or tint (a toon shadow step, an ambient, a diffuse tint, an outline colour)
                             made for `was` adapted to `now`: hue turned, saturation only ever scaled down (a vivid
                             colour keeps the tuned depth of tint, a grey one gets a neutral tint), value kept (shadows and
                             outlines stay as dark)

Both give the tuned value back exactly when `now` equals `was`, so a palette left as it was builds bit-identical.
Colours are rgb triples of sRGB values in 0..1."""
import colorsys

import numpy as np


def _hsv(c):
    return colorsys.rgb_to_hsv(*(float(x) for x in c))


def follow(c, now, was):
    """Colour `c` (that goes with colour `was`) for colour `now`: see the module docstring."""
    if np.array_equal(np.asarray(now, float), np.asarray(was, float)):
        return np.array(c, float)
    h0, s0, v0 = _hsv(was)
    h1, s1, v1 = _hsv(now)
    h, s, v = _hsv(c)
    return np.clip(np.array(colorsys.hsv_to_rgb((h + h1 - h0) % 1.0, min(s * s1 / max(s0, 1e-6), 1.0),
                                                min(v * v1 / max(v0, 1e-6), 1.0))), 0.0, 1.0)


def follow_tint(m, now, was):
    """Multiplier or tint `m` (made for colour `was`) for colour `now`: see the module docstring."""
    if np.array_equal(np.asarray(now, float), np.asarray(was, float)):
        return np.array(m, float)
    h0, s0, _ = _hsv(was)
    h1, s1, _ = _hsv(now)
    h, s, v = _hsv(m)
    return np.clip(np.array(colorsys.hsv_to_rgb((h + h1 - h0) % 1.0, s * min(s1 / max(s0, 1e-6), 1.0), v)), 0.0, 1.0)
