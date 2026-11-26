"""A sheet's sticker's echoes (mkmmd.core.echoes: a train dropped along her path, a spiral; mkmmd.echoes: drawn under
her)."""
import numpy as np

from mkmmd import framefx as FF
from mkmmd.core import echoes as EC
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 160, 320
SKY = (0.55, 0.62, 0.7)


def test_a_train_drops_a_copy_each_spacing_of_her_travel_and_none_while_she_stands_still():
    def path(speed, n=30):
        return [(g, (80.0, 300.0 - speed * g), 60.0) for g in range(n)]     # 60 px tall, rising `speed` px a frame
    fast, slow, still = (EC.dropped(path(v), 1.0 / 6.0) for v in (20.0, 2.0, 0.0))
    assert len(fast) == 29 and len(slow) == 5 and still == []              # one a frame at most; 10 px a copy: 1 in 5
    fx = {"train": {"spacing": 1.0 / 6.0, "life": 0.5}, "count": 10, "fade": 0.8, "shrink": 0.1}
    shown = EC.train_copies(fx, 30, EC.dropped(path(2.0, 30), 1.0 / 6.0), FPS)
    assert [g for _, g, _, _ in shown] == [20, 25]                         # slow: two, those under half a second old
    assert shown[-1][2] > shown[0][2] and shown[-1][3] > shown[0][3]       # the newest the most opaque and the largest


def figure(g):
    m = np.zeros((H, W), np.float32)
    m[100:220, 60:100] = 1.0
    return m


def test_a_train_lies_behind_her_as_separate_copies_and_never_over_her():
    her = {"kind": "sheet", "name": "her", "from": 0.0, "to": 2.0, "objects": "R*", "shot": "plate", "paper": False,
           "drift": False, "edge": False, "shadow": False, "at": [0.5, 1.0], "height": 0.2,
           "keys": [{"t": 0.0, "dy": 0.0}, {"t": 0.7, "dy": -0.8, "ease": "linear"}]}      # bottom to top in 0.7 s
    trail = {"kind": "trails", "from": 0.0, "to": 2.0, "sheet": "her", "train": {}, "count": 10, "fade": 0.8,
             "tint": 0.5, "colors": ["foam", "iris", "rose", "gold"], "cycle": 0.0, "shrink": 0.1}

    def frames(fx):
        data = {"project": {"duration": 3.0}, "output": [{"name": "o", "size": [W, H]}],
                "shot": [{"name": "a", "from": 0.0, "to": 3.0}, {"name": "plate", "plate": True}], "fx": fx}
        pl = TR.plan(data, FPS, F0, DAWN)
        return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: np.full((H, W, 3), SKY, np.float32),
                          point_of=lambda key, g: ((W / 2, H / 2), None), figure_of=lambda key, g: figure(g),
                          plate_of=lambda shot, g: np.full((H, W, 3), (0.9, 0.3, 0.4), np.float32))
    with_trail, plain = frames([trail, her]), frames([her])
    pics = {g: with_trail.frame(g, np.full((H, W, 3), SKY, np.float32)) for g in range(F0, F0 + 16)}
    g = F0 + 15
    bare = plain.frame(g, np.full((H, W, 3), SKY, np.float32))
    her_now = np.abs(bare - np.asarray(SKY, np.float32)).max(2) > 0.02
    solid = np.abs(bare - np.asarray((0.9, 0.3, 0.4), np.float32)).max(2) < 0.01
    assert solid.sum() > 500 and np.array_equal(pics[g][solid], bare[solid])     # nothing of it over her
    rows = (np.abs(pics[g] - bare).max(2) > 0.05).any(1)                   # where the copies are, row by row
    below = np.nonzero(rows)[0]
    assert below.size and below.min() > np.nonzero(her_now.any(1))[0].min()   # behind her: under where she is now
    runs = np.diff(np.concatenate([[0], rows.astype(int), [0]]))
    assert (runs == 1).sum() >= 1 and below.max() > np.nonzero(her_now.any(1))[0].max() + 20   # a train, not a fringe
