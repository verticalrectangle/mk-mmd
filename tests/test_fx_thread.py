"""Threads (mkmmd.core.thread: the entry checked, the shape; mkmmd.thread: a thread drawn from a sheet's live sticker)."""
import math

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import thread as TH
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 240, 320
MAGENTA = (1.0, 0.0, 1.0)


def wave(amp, length=0.3, speed=1.0):
    return {"amp": [[0.0, amp]], "length": length, "speed": speed}


def test_a_still_tie_with_no_wave_is_a_straight_thread_from_the_tie_to_its_end():
    p = TH.shape(lambda tau: (100.0, 200.0), (130.0, -60.0), 1.0, wave(0.0), 0.25, H)
    assert p[0] == pytest.approx((100.0, 200.0)) and p[-1] == pytest.approx((130.0, -60.0))
    d = (p - p[0]) @ np.array([260.0, 30.0]) / math.hypot(30.0, 260.0)     # across the line from the tie to the end
    assert np.abs(d).max() < 1e-6


def test_the_wave_runs_out_from_the_tie_and_grows_from_nothing_there():
    def across(t):
        p = TH.shape(lambda tau: (120.0, 300.0), (120.0, -200.0), t, wave(0.03), 0.0, H)
        return p[:, 0] - 120.0, (300.0 - p[:, 1]) / H                        # px across, frame heights out
    y, s = across(1.0)
    crest = s[np.argmax(np.where(s < 0.45, y, -1e9))]                       # the first full crest out from the tie
    y2, _ = across(1.05)
    near = np.abs(s - crest) < 0.1
    later = s[near][np.argmax(y2[near])]
    assert crest < later == pytest.approx(crest + 0.05 * 0.3, abs=0.01)    # 1 crest a second, 0.3 apart: out 0.015


def test_the_far_part_follows_the_tie_late():
    def tie(tau):
        return 100.0 + 200.0 * tau, 300.0                                    # the tie moving right
    p = TH.shape(tie, (150.0, -100.0), 1.0, wave(0.0), 0.5, H)
    now = TH.shape(lambda tau: tie(1.0), (150.0, -100.0), 1.0, wave(0.0), 0.0, H)
    mid = len(p) // 2
    assert p[0] == pytest.approx(now[0]) and p[mid][0] < now[mid][0] - 20.0  # behind where a taut one would be


def test_a_thread_needs_its_sheet_and_tie_and_takes_wave_keys():
    base = {"kind": "thread", "from": 0.0, "to": 2.0, "sheet": "her", "at": [0.5, 0.1]}
    fx = FX.normalize({**base, "wave": {"amp": [[0.0, 0.03], [1.0, 0.0]]}}, DAWN)
    assert TH.amp_at(fx["wave"], 0.5) == pytest.approx(0.015) and TH.amp_at(fx["wave"], 2.0) == 0.0
    assert fx["color"] == FX._colour("surface", DAWN, "x") and fx["shadow"]["color"] == FX._colour("text", DAWN, "x")
    for k in ("sheet", "at"):
        with pytest.raises(FX.FxError, match=k):
            FX.normalize({key: v for key, v in base.items() if key != k}, DAWN)
    with pytest.raises(FX.FxError, match="amp"):
        FX.normalize({**base, "wave": {"amp": -0.1}}, DAWN)


def figure(g):
    m = np.zeros((H, W), np.float32)
    m[60:160, 50:110] = 1.0
    return m


def frames(fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    pl = TR.plan(data, FPS, F0, DAWN)
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: np.full((H, W, 3), MAGENTA, np.float32),
                      point_of=lambda key, g: ((W / 2, H / 2), None), figure_of=lambda key, g: figure(g))


def test_a_thread_is_tied_where_the_sheet_lays_its_sticker_and_runs_out_to_its_end():
    keys = [{"t": 1.0}, {"t": 2.0, "dx": 0.2, "turn": 30.0, "ease": "linear"}]
    her = {"kind": "sheet", "name": "her", "from": 1.0, "to": 3.0, "objects": "Rin*", "paper": False, "drift": False,
           "edge": False, "shadow": False, "at": [0.4, 0.6], "height": 0.4, "pivot": "center", "keys": keys}
    thread = {"kind": "thread", "from": 1.0, "to": 3.0, "sheet": "her", "at": [80.5 / W, 60.5 / H], "end": [0.5, -0.3],
              "width": 6.0, "shadow": False, "wave": {"amp": 0.0}, "lag": 0.0}
    ffx = frames([her, thread])
    cream_rgb = np.asarray(FX._colour("surface", DAWN, "x"), np.float32)
    for t in (1.0, 2.0):
        pic = ffx.frame(F0 + int(round(t * FPS)), np.full((H, W, 3), MAGENTA, np.float32))
        cream = np.abs(pic - cream_rgb).max(2) < 0.05
        # the figure's top middle (80, 60) where the sheet lays it: its centroid (79.5, 109.5) at `at`, 1.28 times as
        # large (0.4 of the frame over its box 100 px tall), then keyed: turned about it and moved right
        s, a = 1.28, math.radians(30.0 * (t - 1.0))
        dx, dy = 80.0 - 79.5, 60.0 - 109.5
        tie = np.array([0.4 * W - 0.5 + 0.2 * H * (t - 1.0) + s * (math.cos(a) * dx + math.sin(a) * dy),
                        0.6 * H - 0.5 + s * (-math.sin(a) * dx + math.cos(a) * dy)])
        end = np.array([0.5 * W - 0.5, -0.3 * H - 0.5])
        for k in (0.05, 0.3, 0.6):                                           # cream along the line out from the tie
            x, y = tie + (end - tie) * k
            if y >= 0:
                assert cream[int(round(y)), int(round(x))], (t, k)
        assert not cream[int(round(tie[1])) + 12, int(round(tie[0]))]       # nothing below the tie: it starts there
        assert not cream[:, : int(min(tie[0], end[0])) - 8].any()            # nor off to the side
    bad = frames([her, {**thread, "sheet": "nobody"}])
    with pytest.raises(FX.FxError, match="no sheet is named"):
        bad.frame(F0 + 45, np.full((H, W, 3), MAGENTA, np.float32))


def test_a_thread_tied_to_a_world_point_rides_it_as_the_sheets_plate_sees_it_reading_it_over_its_lag():
    def crown(g):                                            # her crown in the plate: sinking 2 px a frame (a crouch)
        return 80.0, 60.0 + 2.0 * (g - F0 - 30)
    read = []

    def point(key, g):
        read.append(g)
        return crown(g), 0.3
    her = {"kind": "sheet", "name": "her", "from": 1.0, "to": 3.0, "objects": "Rin*", "shot": "plate", "paper": False,
           "drift": False, "edge": False, "shadow": False}
    thread = {"kind": "thread", "from": 1.0, "to": 3.0, "sheet": "her", "at": {"point": "bone('head').tail"},
              "end": [0.5, -0.3], "width": 6.0, "shadow": False, "wave": {"amp": 0.0}, "lag": 0.25}
    data = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "plate", "plate": True}], "fx": [her, thread]}
    pl = TR.plan(data, FPS, F0, DAWN)
    back = math.ceil(0.25 * FPS)                             # the frames its lag looks back over: 8
    got = [(f, it["shot"]) for f, items in TR.demands(pl).items() for it in items if it.get("key") == "x1a"]
    assert got == [(f, "plate") for f in range(F0 + 30 - back, F0 + 90)]     # through her plate, its span and lag
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: np.full((H, W, 3), MAGENTA, np.float32),
                     point_of=point, figure_of=lambda key, g: figure(g),
                     plate_of=lambda shot, g: np.full((H, W, 3), 0.5, np.float32))
    cream_rgb = np.asarray(FX._colour("surface", DAWN, "x"), np.float32)
    end = (0.5 * W - 0.5, -0.3 * H - 0.5)
    for g in (F0 + 30, F0 + 60):
        read.clear()
        pic = ffx.frame(g, np.full((H, W, 3), MAGENTA, np.float32))
        cream = np.abs(pic - cream_rgb).max(2) < 0.05
        assert min(read) == g - back and max(read) == g                     # where it was over its lag, and now
        line = TH.shape(lambda tau: crown(F0 + tau * FPS), end, (g - F0) / FPS, pl["fx"][1]["wave"], 0.25, H)
        for k in (8, 48, 96):                                                # cream along it, out from her crown ...
            x, y = line[k]
            if y >= 0:
                assert cream[int(round(y)), int(round(x))], (g, k)
        x, y = crown(g)
        assert not cream[int(round(y)) + 12, int(round(x))]                  # ... which sank with her: it starts there
    with pytest.raises(TR.TransitionError, match="no sheet of that `name`"):
        TR.plan({**data, "fx": [her, {**thread, "sheet": "nobody"}]}, FPS, F0, DAWN)
