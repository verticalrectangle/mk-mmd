"""The pond and café sticker kits (mkmmd.core.pondkit, registered in mkmmd.core.mandala): known to the mandala and the
sheet, drawn with the pen's ink line round them in their own palette colours, and importable on their own."""
import subprocess
import sys

import cv2
import numpy as np
import pytest

from mkmmd import mandala as MDF
from mkmmd.core import framefx as FX
from mkmmd.core import mandala as MD
from mkmmd.core import palette as PAL

DAWN = PAL.get("rose-pine-dawn")
KITS = ("lily", "lilypad", "bud", "reed", "dragonfly", "drop", "croissant", "cup", "sugar", "orange", "bean")
RED, BLUE, GREEN = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)


@pytest.mark.parametrize("name", KITS)
def test_a_kit_sticker_is_registered_and_drawn_round_with_the_pen(name):
    assert name in MD.SHAPES and name in MD.PENNED
    assert len(MD.SHAPE_COLORS[name]) >= 2                    # the sheet's stand-in for an accent is the second
    assert set(MD.SHAPE_COLORS[name]) | {MD.SHAPE_ACCENT[name]} <= set(DAWN)   # palette slots, never hex
    lo, hi = MD.shape_box(MD.shape_parts(name))
    assert max(hi - lo) == pytest.approx(1.0)
    t = MDF.shape_tile(name, RED, BLUE, GREEN, 70, 0.0, 0.0)  # as a sheet draws a doodle of a pen kit: no extra pen
    assert t.a.mean() > 0.2                                   # a sticker, not a scribble
    rgb = t.rgb / np.maximum(t.a, 1e-6)[..., None]
    solid = (t.a > 0.5).astype(np.uint8)
    rim = (solid > 0) & (cv2.erode(solid, np.ones((3, 3), np.uint8)) == 0)
    inked = (rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] > rgb[..., 2])
    assert inked[rim].mean() > 0.85                           # its edge is the ink line all round
    own = (rgb[..., 0] > 0.9) & (rgb[..., 2] < 0.7) & (t.a > 0.99)   # its colour or its light tint, not white paper
    assert own.sum() > 40


def test_a_sheet_deals_the_kits_as_doodles_in_their_own_colours():
    fx = FX.normalize({"kind": "sheet", "from": 0.0, "to": 1.0, "doodles": {"shapes": list(KITS)}}, DAWN)
    dd = fx["doodles"]
    assert dd["shapes"] == list(KITS)
    for name in KITS:
        accent, alt, own = dd["own"][name]
        assert accent == pytest.approx(PAL.srgb(DAWN[MD.SHAPE_ACCENT[name]]), abs=1e-5)
        assert [tuple(c) for c in own] == [pytest.approx(PAL.srgb(DAWN[s]), abs=1e-5) for s in MD.SHAPE_COLORS[name]]


@pytest.mark.parametrize("first", ["mkmmd.core.pondkit", "mkmmd.core.mandala"])
def test_the_kits_import_whichever_module_comes_first(first):
    code = f"import {first}; from mkmmd.core import mandala as MD; print(len(MD.shape_parts('lily')))"
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr
