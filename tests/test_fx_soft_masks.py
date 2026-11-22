"""A trail kept off a figure (`[[fx]] kind = "trails"` with `mask = {ground = ...}`) and a liquid light show's soft oil
edge (`soft`) and colour blend (mkmmd.core.mandala, mkmmd.mandala, mkmmd.core.fields, mkmmd.fields)."""
import cv2
import numpy as np
import pytest

from mkmmd import fields as FLD
from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 48


def plan(*fxs):
    return TR.plan({"project": {"duration": 2.0}, "output": [{"name": "o", "size": [W, H]}],
                    "shot": [{"name": "a", "from": 0.0, "to": 2.0}], "fx": list(fxs)}, FPS, F0, DAWN)


def test_a_trail_masked_off_her_leaves_her_pixels_untouched():
    pl = plan({"kind": "trails", "from": 0.0, "to": 2.0, "objects": "dfly", "count": 3, "delay": 2,
               "mask": {"ground": "her_*"}})
    fx = pl["fx"][0]
    bug, her = FX.figure_key({"objects": ["dfly"]}), FX.figure_key({"objects": ["her_*"]})
    hers = np.zeros((H, W), np.float32)
    hers[:, 20:34] = 1.0                                         # she stands across the middle, where it flew past

    def figure_of(key, g):
        m = np.zeros((H, W), np.float32)
        if key == her:
            return hers
        x = 6 + 3 * (g - F0)                                     # the dragonfly darting right, 3 px a frame
        m[10:18, x:x + 6] = 1.0
        return m
    pic = np.full((H, W, 3), 0.5, np.float32)
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: pic, point_of=None,
                     figure_of=figure_of)
    f = F0 + 10                                                   # the dragonfly at x 36: its copies at 30, 24, 18
    got = run.frame(f, pic)
    assert np.array_equal(got[:, 20:34], pic[:, 20:34])           # nothing of the trail on her
    assert np.abs(got[10:18, 18:20] - 0.5).max() > 0.05           # the oldest copy shows beside her
    assert {it["key"] for it in TR.demands(pl)[f] if it["kind"] == "figure"} == {bug, her}   # her layer drawn too
    calm = plan({"kind": "trails", "from": 0.0, "to": 2.0, "objects": "dfly", "count": 3, "delay": 2, "still": 0.2})

    def hovering(key, g):                                         # it drifts half a px a frame: a hover
        m = np.zeros((H, W), np.float32)
        m[10:18, 30 + (g - F0) // 2:36 + (g - F0) // 2] = 1.0
        return m
    run = FF.FrameFx(calm["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: pic, point_of=None,
                     figure_of=hovering)
    assert np.array_equal(run.frame(f, pic), pic)                 # `still`: a hover leaves no after-images


def test_a_soft_liquid_edge_at_0_is_today_s_and_at_1_has_no_rim():
    base = {"kind": "liquid", "from": 0.0, "to": 2.0, "inks": ["rose"], "light": "rose", "cells": 0.0, "bubbles": 0}
    today = FX.normalize(base, DAWN)
    soft0, soft1 = (FX.normalize({**base, "soft": s}, DAWN) for s in (0.0, 1.0))
    del today["soft"]                                             # an entry planned before the key: today's code
    memo = FF.FrameFx([], (96, 72), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None)._memo
    a = FLD.liquid(today, (96, 72), 0.7, (48.0, 36.0), 0.0, memo)
    assert np.array_equal(FLD.liquid(soft0, (96, 72), 0.7, (48.0, 36.0), 0.0, memo), a)
    b = FLD.liquid(soft1, (96, 72), 0.7, (48.0, 36.0), 0.0, memo)
    sharp = [float(np.abs(cv2.Laplacian(np.ascontiguousarray(p @ FLD.LUMA), cv2.CV_32F)).max()) for p in (a, b)]
    assert sharp[1] < 0.35 * sharp[0]                             # no meniscus, no lit rim: no sharp line left
    rose = np.asarray(today["light"], np.float32)
    img = np.full((4, 4, 3), [0.34, 0.32, 0.47], np.float32)
    tinted = FLD.blend(img, np.broadcast_to(rose, img.shape), 1.0, "color")
    assert tinted @ FLD.LUMA == pytest.approx(img @ FLD.LUMA, abs=1e-5) and not np.allclose(tinted, img)
