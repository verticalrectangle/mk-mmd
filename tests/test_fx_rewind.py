"""Rewinds ([[fx]] kind rewind: mkmmd.core.rewind and mkmmd.rewind): a finger scrubbing the video back along a paper bar,
the picture the cut's own at the seconds of its time map, its playhead where the second shown is."""
import re

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import rewind as RW
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 90, 160
MAP = [{"t": 2.2, "from": 1.5, "to": 1.0}, {"t": 2.8, "from": 0.6, "to": 0.2}]


def rewind(**spec):
    return {"kind": "rewind", "from": 2.0, "to": 3.5, "map": MAP, "fps": 10, "counter": False, **spec}


def plan(*fxs, **extra):
    return TR.plan({"project": {"duration": 4.0}, "output": [{"name": "9x16", "size": [W, H]}],
                    "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fxs), **extra}, FPS, F0, DAWN)


def frame(t):
    return FX.frame_of(t, FPS, F0)


def flat(g):
    """The cut's frame `g`: flat grey, its level its frame number, so the frame a picture comes from reads off it."""
    return np.full((H, W, 3), g / 255.0, np.float32)


def test_the_time_map_lands_each_segment_on_its_seconds_in_poses_held_from_its_start():
    tint = {"kind": "tint", "from": 0.0, "to": 1.6, "color": "#FF0000", "strength": 0.5}
    pl = plan(tint, rewind(tear=False))
    fx = pl["fx"][1]
    shows = {f: RW.shown(fx, f)[1] for f in range(fx["f0"], fx["f1"])}
    assert shows[frame(2.1)] == frame(2.1)                          # before the first segment: the cut's own frame
    a, b = frame(2.2), frame(2.8)
    assert shows[a] == frame(1.5) and shows[b - 1] == frame(1.0)    # a segment runs `from` down to `to` ...
    assert shows[b] == frame(0.6) and shows[fx["f1"] - 1] == frame(0.2)   # ... and the next jumps to its own `from`
    assert [shows[f] for f in range(a, a + 3)] == [frame(1.5)] * 3 and shows[a + 3] < frame(1.5)   # 10 poses a second
    assert all(shows[f] >= shows[f + 1] for f in range(a, fx["f1"] - 1) if f + 1 != b)              # it runs back
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=flat, point_of=None, output="9x16")
    f = a + 4
    got = run.frame(f, flat(f))
    source = run.frame(shows[f], flat(shows[f]))                    # that second's picture, its tint on it there
    assert np.allclose(got[:20], source[:20], atol=1e-6)            # above the bar: the picture scrubbed back to
    assert not np.allclose(source[:20], flat(shows[f])[:20], atol=0.05)
    assert TR.fx_frames(pl, [f]) == {shows[f]: [{"kind": "plate", "shot": "a"}]}    # mk look renders what it reads


def test_the_playhead_drags_steadily_left_with_a_small_snap_at_each_jump_and_a_finger_riding_it_keeps_its_tip_on_it():
    finger = {"kind": "cutout", "from": 2.0, "to": 3.5, "clip": "hand.mp4", "at": {"rewind": 2.0}}
    pl = plan(rewind(bar={"at": [0.1, 0.7], "length": 0.8, "tilt": 0.0}), finger)
    fx, cut = pl["fx"]
    assert cut["at"]["tr"] == {"rewind": 0}
    secs = {}
    for f in range(frame(2.5), fx["f1"]):                           # the bar landed at 2.08 and has stopped ringing
        (x, y), secs[f] = RW.playhead(fx, f, (W, H))
        if f >= frame(2.8):                                         # the bar's landing has rung out
            assert (x, y) == pytest.approx(((0.1 + 0.8 * secs[f] / 4.0) * W - 0.5, 0.7 * H - 0.5), abs=0.02)
    a, b, end = frame(2.2), frame(2.8), fx["f1"] - 1
    steps = [secs[f] - secs[f + 1] for f in range(frame(2.5), end)]
    assert min(steps) >= -1e-9 and sum(s > 0 for s in steps) >= 9          # left on every pose (10 a second)
    assert RW.placed(fx, a - 1) == pytest.approx((a - 1 - F0) / FPS) and secs[end] == pytest.approx(0.2)
    jump, whole = secs[b - 1] - secs[b], RW.shown(fx, b - 1)[0] - RW.shown(fx, b)[0]
    usual = sorted(s for s in steps if s > 0)[len([s for s in steps if s > 0]) // 2]
    assert usual < jump < whole                                     # a small snap where the picture jumps
    assert RW.shown(fx, b)[0] == pytest.approx(0.6)                 # while the counter shows the second shown
    whole_jumps = plan(rewind(snap=1.0))["fx"][0]
    assert all(RW.placed(whole_jumps, f) == RW.shown(whole_jumps, f)[0] for f in range(fx["f0"], fx["f1"]))
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=flat, point_of=None, output="9x16")
    f = frame(2.9)
    assert run._cutout_at(cut, f) == pytest.approx(RW.playhead(fx, f, (W, H))[0])
    with pytest.raises(TR.TransitionError, match="comes after it in the table"):
        plan(finger, rewind())


@pytest.mark.parametrize("spec, extra, frag", [
    (rewind(map=[{"t": 2.2, "from": 2.4, "to": 1.0}]), {}, "a rewind shows seconds that came before"),
    (rewind(map=[{"t": 2.2 + 0.25 * k, "from": 1.0, "to": 0.5} for k in range(4)]), {}, "at most 3 a second"),
    (rewind(land=2.5), {}, "from (2) <= land (2.5"),
    (rewind(), {"insert": [{"from": 0.4, "to": 0.8, "shape": "split", "clip": {"src": "x.mp4"}, "in": 0.0,
                            "out": 0.0}]}, "choose seconds outside the window"),
])
def test_bad_rewinds_say_what_is_wrong(spec, extra, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        plan(spec, **extra)
