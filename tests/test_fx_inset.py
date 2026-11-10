"""`inset`: the picture set into a part of a clip's frame, a view between the clip and the picture (docs/design.md:
Frame effects, Insets)."""
import cv2
import numpy as np
import pytest

from mkmmd import footage as FT
from mkmmd import framefx as FF
from mkmmd import postgpu as PG
from mkmmd.core import inset as IN
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 160, 90
INTO = [0.5, 0.2, 0.75, 0.6]                    # the screen in the clip: x 80..120, y 18..54 of the frame it covers
RED = np.zeros((H, W, 3), np.float32)
RED[..., 0] = 1.0


def inset(**kw):
    return {"kind": "inset", "from": 0.0, "to": 2.0, "clip": "desk.mp4", "into": INTO, **kw}


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


@pytest.fixture
def desk(tmp_path):
    """What opens desk.mp4: a clip of flat green frames twice the frame's size."""
    vw = cv2.VideoWriter(str(tmp_path / "desk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (2 * W, 2 * H))
    assert vw.isOpened()
    for _ in range(6):
        vw.write(np.full((2 * H, 2 * W, 3), (0, 200, 0), np.uint8))
    vw.release()
    return FT.source(tmp_path)


def runner(pl, footage):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, footage_of=footage)


@pytest.mark.parametrize("spec, frag", [
    ({"kind": "inset", "from": 0.0, "to": 1.0, "clip": "desk.mp4"}, "`into` is required"),
    (inset(into=[0.6, 0.2, 0.5, 0.6]), "x1 must be right of x0"),
    (inset(view=[[0.0, 0.0], [1.0, 1.5]]), "view runs from 0"),
    (inset(strength=0.5), "unknown keys ['strength']"),
    ({"kind": "inset", "from": 0.0, "to": 1.0, "into": INTO}, "`clip` is required"),
])
def test_bad_insets_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError) as e:
        plan(spec)
    assert frag in str(e.value)


def test_the_window_runs_from_the_whole_frame_to_the_box_about_one_fixed_point():
    box = (80.0, 18.0, 120.0, 54.0)
    assert IN.window(box, (W, H), 0.0) == pytest.approx((0.0, 0.0, W, H))
    assert IN.window(box, (W, H), 1.0) == pytest.approx(box)
    x0, y0, x1, y1 = IN.window(box, (W, H), 0.5)
    assert x1 - x0 == pytest.approx(np.sqrt(W * 40.0)) and y1 - y0 == pytest.approx(np.sqrt(H * 36.0))   # a log scale
    px = 80.0 * W / (W - 40.0)                  # the point as far across the frame as across the box: it never moves
    assert (px - x0) / (x1 - x0) == pytest.approx(px / W)
    assert IN.place(box, (x0, y0, x1, y1), (W, H))[0] == pytest.approx((80.0 - x0) * W / (x1 - x0))


def test_the_picture_sits_in_its_part_of_the_clip_and_alone_at_view_one(desk):
    def red(view):
        out = runner(plan(inset(view=view)), desk).frame(F0 + 3, RED)
        return out, out[..., 0] > 0.9
    out, r = red(0.0)
    assert r[20:52, 82:118].all() and not r[:16].any() and not r[:, :78].any() and not r[:, 122:].any()   # in the box
    assert out[5, 5] == pytest.approx([0.0, 200 / 255, 0.0], abs=0.03)                 # the clip round it
    assert red(0.5)[1].sum() > 2 * r.sum()                                           # the view comes in on it
    same = runner(plan(inset(view=1.0)), desk).frame(F0 + 3, RED)
    assert same is RED                                                               # the picture itself, untouched


def test_the_gpu_leaves_an_inset_to_the_cpu(desk):
    r = runner(plan(inset()), desk)
    assert PG.ops(r, r.fxs[0], F0 + 3) is None
