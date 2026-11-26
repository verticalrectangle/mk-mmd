"""The sky sticker kit (mkmmd.core.skykit, registered in mkmmd.core.mandala): new names beside the old shapes, each
sticker drawn with the pen's ink line round it about its middle, its accent apart from every colour a sheet deals it
in, and the kit importable on its own."""
import subprocess
import sys

import cv2
import numpy as np
import pytest

from mkmmd import mandala as MDF
from mkmmd.core import framefx as FX
from mkmmd.core import mandala as MD
from mkmmd.core import palette as PAL
from mkmmd.core import skykit as SK

DAWN = PAL.get("rose-pine-dawn")
SKY = ("plane", "bird", "bird_up", "cumulus", "sun", "kite", "balloon", "shooting_star", "feather")
RED, BLUE, GREEN = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)


def test_the_sky_kit_takes_new_names_and_leaves_the_old_shapes_as_they_were():
    assert len(set(MD.SHAPES)) == len(MD.SHAPES)                              # no name given twice
    assert set(MD._SHAPE_PARTS) == set(MD.SHAPES) == set(MD.SHAPE_COLORS) == set(MD.SHAPE_ACCENT)
    assert set(SKY) <= set(MD.PENNED)                                         # drawn with the pen already
    assert all(MD._SHAPE_PARTS[n] is getattr(SK, n) for n in SKY)
    assert MD._SHAPE_PARTS["cloud"] is MD._cloud and MD._SHAPE_PARTS["star"] is MD._star


@pytest.mark.parametrize("name", SKY)
def test_a_sky_sticker_is_drawn_round_with_the_pen_about_its_middle(name):
    parts = MD.shape_parts(name)
    assert {role for role, _, _, _ in parts} <= set(MD.ROLES)
    assert all(len(p) >= 2 and np.isfinite(np.asarray(p, float)).all() for _, _, p, _ in parts)
    lo, hi = MD.shape_box(parts)
    assert max(hi - lo) == pytest.approx(1.0)
    assert np.abs((lo + hi) / 2.0).max() < 0.08                              # a ring turns it about its middle
    t = MDF.shape_tile(name, RED, BLUE, GREEN, 70, 0.0, 0.0)                 # as a sheet draws a pen kit's doodle
    assert t.a.mean() > 0.2                                                   # a sticker, not a scribble
    rgb = t.rgb / np.maximum(t.a, 1e-6)[..., None]
    solid = (t.a > 0.5).astype(np.uint8)
    rim = (solid > 0) & (cv2.erode(solid, np.ones((3, 3), np.uint8)) == 0)
    inked = (rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] > rgb[..., 2])
    assert inked[rim].mean() > 0.85                                           # its edge is the ink line all round
    assert ((rgb[..., 0] > 0.9) & (rgb[..., 2] < 0.7) & (t.a > 0.99)).sum() > 40   # its own colour shows


def test_a_sheet_deals_the_sky_kit_with_an_accent_apart_from_each_of_its_colours():
    fx = FX.normalize({"kind": "sheet", "from": 0.0, "to": 1.0, "doodles": {"shapes": list(SKY)}}, DAWN)
    for name in SKY:
        accent, alt, own = fx["doodles"]["own"][name]
        assert [tuple(c) for c in own] == [pytest.approx(PAL.srgb(DAWN[s]), abs=1e-5) for s in MD.SHAPE_COLORS[name]]
        for main in own:                       # a sticker in its accent's colour wears the second colour for it
            used = alt if tuple(accent) == tuple(main) else accent
            assert tuple(used) != tuple(main)  # the beak, the rays, the stripe never sink into the body


@pytest.mark.parametrize("first", ["mkmmd.core.skykit", "mkmmd.core.mandala"])
def test_the_sky_kit_imports_whichever_module_comes_first(first):
    code = f"import {first}; from mkmmd.core import mandala as MD; print(len(MD.shape_parts('plane')))"
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr
