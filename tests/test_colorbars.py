"""Colour bars (mkmmd.core.colorbars): a card's face in true, palette and both colours, row 0 its top, and its back."""
import numpy as np
import pytest

from mkmmd.core import colorbars as CB
from mkmmd.core import palette as PAL

DAWN = PAL.get("rose-pine-dawn")


def top_row(img):
    """The seven colours across the top row, sampled in the middle of each bar."""
    w = img.shape[1]
    return [tuple(float(c) for c in img[2, int((k + 0.5) * w / 7)]) for k in range(7)]


@pytest.mark.parametrize("style", CB.STYLES)
def test_a_face_is_a_display_picture_of_its_size(style):
    img = CB.texture(64, 48, DAWN, style)
    assert img.shape == (48, 64, 3) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_true_bars_are_seventy_five_percent_and_run_from_grey_to_blue():
    assert np.allclose(top_row(CB.texture(140, 60, DAWN, "true")), CB.TRUE, atol=1e-6)


def test_palette_bars_are_its_accents_brightest_first_after_its_white():
    got = top_row(CB.texture(140, 60, DAWN, "dawn"))
    assert np.allclose(got[0], PAL.srgb(DAWN["surface"]), atol=1e-6)        # Dawn's lightest neutral
    lum = [CB.luminance(c) for c in got]
    assert lum == sorted(lum, reverse=True)
    accents = sorted((PAL.srgb(DAWN[s]) for s in CB.ACCENTS), key=CB.luminance, reverse=True)
    assert np.allclose(got[1:], accents, atol=1e-6)


def test_both_puts_the_palette_over_the_true_bars_and_the_pluge_at_the_foot():
    img = CB.texture(140, 100, DAWN, "both")
    mid = [img[60, int((k + 0.5) * 20)] for k in range(7)]
    assert np.allclose(top_row(img)[0], PAL.srgb(DAWN["surface"]), atol=1e-6)
    assert np.allclose(mid, CB.TRUE, atol=1e-6)
    assert np.allclose(img[-1, 12], CB.MINUS_I)                               # the foot: -I under the first bars


def test_another_palette_colours_the_bars_its_own_way():
    moon = PAL.get("rose-pine-moon")
    got = top_row(CB.texture(140, 60, moon, "dawn"))
    assert np.allclose(got[0], PAL.srgb(moon["text"]), atol=1e-6)           # the moon's white is its text
    assert got != top_row(CB.texture(140, 60, DAWN, "dawn"))


def test_bad_styles_and_sizes_say_so():
    with pytest.raises(ValueError, match="none of dawn, true, both"):
        CB.texture(64, 48, DAWN, "smpte")
    with pytest.raises(ValueError, match="at least 7 x 4"):
        CB.texture(5, 48, DAWN)


def test_the_back_is_the_palettes_white_ruled_with_a_ring():
    img = CB.back(120, 90, None)                                              # None: Dawn
    assert img.shape == (90, 120, 3)
    white = np.asarray(PAL.srgb(DAWN["surface"]), np.float32)
    share = float(np.mean(np.all(np.abs(img - white) < 1e-3, axis=2)))
    assert share > 0.5                                                        # mostly paper, the rules and ring thin
    assert not np.allclose(img[45, 60 + int(0.3 * 90)], white, atol=0.05)    # the ring crosses its middle row
