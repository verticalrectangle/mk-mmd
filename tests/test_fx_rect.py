"""The region an [[fx]] entry is limited to (`rect`, `feather`, per output through aspect.<output>): its errors, the mask
(antialiased hard edges, a feathered ramp that stays sharp along the frame's own edges), the frame outside it left as it
was, and the picture effects taking it as their frame (a camcorder's stamp in the corner of its half)."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
OUTPUTS = [{"name": "16x9", "size": [320, 180]}, {"name": "9x16", "size": [180, 320]}]


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": OUTPUTS, "shot": [{"name": "a", "from": 0.0, "to": 4.0}],
            "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


@pytest.mark.parametrize("spec, frag", [
    ({"rect": [0, 0, 0.5]}, "expected [x0, y0, x1, y1]"),
    ({"rect": [0.5, 0, 0.5, 1]}, "x1 must be right of x0"),
    ({"rect": [0, 0, 1.2, 1]}, "expected 0..1"),
    ({"feather": 4}, "feather softens the edge of a `rect`"),
    ({"rect": [0, 0, 0.5, 1], "feather": -1}, "feather = -1"),
])
def test_bad_rects_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag.replace("[", r"\[")):
        FX.normalize({"kind": "tint", "from": 0, "to": 1, **spec}, DAWN)


def test_every_kind_takes_a_rect_and_false_is_the_whole_frame():
    assert all({"rect", "feather"} <= FX.KEYS[k] for k in FX.KINDS)
    fx = FX.normalize({"kind": "tint", "from": 0, "to": 1, "rect": False}, DAWN)
    assert fx["rect"] is None and fx["feather"] == 0.0


def test_the_mask_is_the_rect_antialiased_or_feathered():
    hard = PF.rect_mask((100, 50), (0.1, 0.2, 0.555, 1.0), 0.0)
    assert hard.sum() == pytest.approx(100 * 0.455 * 50 * 0.8, abs=1e-3)       # exactly its area
    assert hard[25, 55] == pytest.approx(0.5) and hard[25, 54] == 1.0 and hard[25, 56] == 0.0
    soft = PF.rect_mask((100, 50), (0.0, 0.0, 0.5, 1.0), 10.0)
    row = soft[25]
    assert np.all(np.diff(row) <= 1e-7)                                       # falling across the edge
    assert row[:44].min() == 1.0 and row[56:].max() == 0.0 and 0.0 < row[50] < 1.0   # a ramp 10 px wide at x = 50
    assert soft[0, 0] == 1.0 and soft[-1, 0] == 1.0                            # no fade along the frame's edges


def test_outside_the_rect_the_frame_is_untouched():
    pl = plan({"kind": "tint", "from": 0, "to": 2, "color": "love", "rect": [0.25, 0.0, 0.75, 1.0]})
    img = np.random.default_rng(1).random((180, 320, 3)).astype(np.float32)
    full = FF.FrameFx(plan({"kind": "tint", "from": 0, "to": 2, "color": "love"})["fx"], (320, 180), fps=FPS, frame0=F0,
                      frame_of=lambda g: None, point_of=None).frame(F0 + 5, img)
    out = FF.FrameFx(pl["fx"], (320, 180), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None).frame(F0 + 5, img)
    assert np.array_equal(out[:, :80], img[:, :80]) and np.array_equal(out[:, 240:], img[:, 240:])
    assert np.allclose(out[:, 80:240], full[:, 80:240], atol=1e-6)


def test_each_output_takes_its_own_rect():
    pl = plan({"kind": "tint", "from": 0, "to": 2, "color": "love", "rect": [0.0, 0.0, 0.5, 1.0],
               "aspect": {"9x16": {"rect": [0.0, 0.0, 1.0, 0.5]}}})
    wide, tall = (FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output=o)
                  for o, (w, h) in (("16x9", (320, 180)), ("9x16", (180, 320))))
    a = wide.frame(F0 + 3, np.full((180, 320, 3), 0.5, np.float32))
    b = tall.frame(F0 + 3, np.full((320, 180, 3), 0.5, np.float32))
    assert np.allclose(a[:, 200:], 0.5) and not np.allclose(a[:, :120], 0.5)   # left half in 16x9
    assert np.allclose(b[200:], 0.5) and not np.allclose(b[:120], 0.5)         # top half in 9x16


def test_a_camcorder_takes_its_half_as_its_frame():
    from test_fx_camcorder import system_font
    font = system_font()
    pl = plan({"kind": "camcorder", "from": 0, "to": 2, "time": "8:41 PM", "font": font, "color": "#FFFFFF",
               "indicator": None, "tracking": 0.0, "noise": 0.0, "rect": [0.0, 0.0, 0.5, 1.0]})
    w, h = 640, 360
    img = np.full((h, w, 3), 0.3, np.float32)
    out = FF.FrameFx(pl["fx"], (w, h), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None).frame(F0 + 3, img)
    ys, xs = np.nonzero(np.all(out > 0.999, axis=2))
    assert len(xs) > 50 and xs.max() < w / 2 and xs.max() > w / 2 - 60          # the bottom right of the left half
    assert np.array_equal(out[:, w // 2 + 2:], img[:, w // 2 + 2:])
