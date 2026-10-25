"""Photo booth ([[fx]] kind "booth", mkmmd.core.picturefx and mkmmd.picturefx): its keys, the layout per output (a
column in a tall frame, a row in a wide one, a block in a square; `layout` and aspect.<output> override it), a strip that
fits the frame tilted, the captures (the first `count` hits, the frames it reads, on a lane only with the hits file), the
flash and the drop, and each cell showing the frame captured there, in mk post's order or out of order."""
import math

import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import picturefx as PC
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
OUTPUTS = [{"name": "16x9", "size": [192, 108]}, {"name": "1x1", "size": [108, 108]}, {"name": "9x16", "size": [108, 192]}]


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": OUTPUTS, "shot": [{"name": "a", "from": 0.0, "to": 4.0}],
            "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def booth(**spec):
    return {"kind": "booth", "from": 1.0, "to": 3.0, "times": [1.2, 1.6, 2.0, 2.4, 2.8], "doodles": 0, **spec}


@pytest.mark.parametrize("spec, frag", [
    ({"times": None}, "give `times`"),
    ({"layout": [1, 3]}, "with room for 4 photos"),
    ({"layout": [2.5, 2]}, "whole numbers"),
    ({"count": 0}, "expected 1..12"),
    ({"tilt": 60}, "expected -45..45"),
    ({"caption": "x" * 41}, "short text"),
    ({"back": "nope"}, "back"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    s = {k: v for k, v in booth(**spec).items() if v is not None}
    with pytest.raises(FX.FxError, match=frag):
        FX.normalize(s, DAWN)


@pytest.mark.parametrize("size, count, lay", [((1080, 1920), 4, (1, 4)), ((1920, 1080), 4, (4, 1)),
                                              ((1080, 1080), 4, (2, 2)), ((1080, 1080), 3, (2, 2)),
                                              ((1080, 1080), 6, (3, 2))])
def test_the_layout_follows_the_frame(size, count, lay):
    fx = FX.normalize(booth(count=count), DAWN)
    assert PC.booth_layout(fx, size) == lay
    assert PC.booth_layout(FX.normalize(booth(layout=[2, 3]), DAWN), size) == (2, 3)


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1080), (1080, 1920)])
@pytest.mark.parametrize("tilt", [0.0, -6.0])
def test_the_strip_fits_the_frame_turned(size, tilt):
    fx = FX.normalize(booth(tilt=tilt, caption="hi"), DAWN)
    geo = PC.booth_strip(fx, size, min(size) / 1080)
    sw, sh = geo["strip"]
    a = math.radians(abs(tilt))
    bw, bh = sw * math.cos(a) + sh * math.sin(a), sw * math.sin(a) + sh * math.cos(a)
    assert bw <= fx["size"] * size[0] + 1e-6 and bh <= fx["size"] * size[1] + 1e-6
    assert max(bw / size[0], bh / size[1]) == pytest.approx(fx["size"], rel=1e-6)    # as large as fits
    pw, ph = geo["photo"]
    assert pw / ph == pytest.approx(fx["ratio"])
    for x, y in geo["cells"]:
        assert x >= geo["border"] - 1e-6 and y >= geo["border"] - 1e-6
        assert x + pw <= sw - geo["border"] + 1e-6 and y + ph <= sh - geo["foot"] - geo["border"] + 1e-6
    assert len(geo["cells"]) == 4


def test_captures_are_the_first_hits_inside_the_span():
    fx = plan(booth(times=[0.5, 1.2, 1.21, 1.6, 2.0, 2.4, 2.8]))["fx"][0]
    caps = PC.captures(fx, fx["times"], FPS, F0)
    assert [t for t, _ in caps] == [1.2, 1.6, 2.0, 2.4]                 # 1.21 lands on 1.2's frame: one capture
    assert [g for _, g in caps] == [FX.frame_of(t, FPS, F0) for t in (1.2, 1.6, 2.0, 2.4)]


def test_it_reads_its_captures_so_far():
    pl = plan(booth())
    fx = pl["fx"][0]
    g1, g2 = FX.frame_of(1.2, FPS, F0), FX.frame_of(1.6, FPS, F0)
    assert FX.sources(fx, g2 + 2, F0) == [g1, g2]
    assert TR.fx_frames(pl, [g2 + 2]) == {g1: [{"kind": "plate", "shot": "a"}], g2: [{"kind": "plate", "shot": "a"}]}
    lane = plan({k: v for k, v in booth(lane="snare").items() if k != "times"})
    hits = {"snare": [{"t": 1.2, "db": 0.0}, {"t": 1.6, "db": -9.0}]}
    assert TR.fx_frames(lane, [g2 + 2]) == {}                         # a lane needs the hits file to know them
    assert list(TR.fx_frames(lane, [g2 + 2], hits)) == [g1, g2]
    assert FX.holds(fx)


def test_flash_and_drop():
    assert PC.flash(0.0, 0.15) == 1.0 and PC.flash(0.15, 0.15) == 0.0 and PC.flash(-0.01, 0.15) == 0.0
    assert 0.0 < PC.flash(0.05, 0.15) < 1.0
    s0, h0 = PC.drop(0.0, 0.3)
    s1, h1 = PC.drop(0.3, 0.3)
    assert s0 == pytest.approx(1.3) and h0 == pytest.approx(1.0) and (s1, h1) == (1.0, 0.0)
    scales = [PC.drop(a, 0.3)[0] for a in np.linspace(0, 0.21, 8)]
    assert all(b < a for a, b in zip(scales, scales[1:]))             # falling to the paper


def clip_frames(size):
    """A clip whose frame g is a flat colour of its own: the cell of a capture shows which frame it took."""
    w, h = size
    return {F0 + k: np.full((h, w, 3), [(k % 7) / 7.0, (k % 11) / 11.0, 0.5], np.float32) for k in range(130)}


@pytest.mark.parametrize("output", ["16x9", "1x1", "9x16"])
def test_each_cell_shows_its_capture(output):
    pl = plan(booth(flash=0.1, drop=0.2, tilt=0.0, back="love"))
    size = tuple(next(o["size"] for o in OUTPUTS if o["name"] == output))
    clip = clip_frames(size)
    ffx = FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=clip.get, point_of=None, output=output)
    fx = ffx.fxs[0]
    caps = PC.captures(fx, fx["times"], FPS, F0)
    out = None
    for f in range(fx["f0"], caps[2][1] + 12):                        # mk post's order
        out = ffx.frame(f, clip[f])
        if f == caps[1][1]:
            assert np.allclose(out, 1.0)                              # the flash on a capture
    geo = PC.booth_strip(fx, size, min(size) / 1080)
    sw, sh = geo["strip"]
    ox, oy = size[0] / 2 - sw / 2, size[1] / 2 - sh / 2
    pw, ph = geo["photo"]
    for j, (_, g) in enumerate(caps):
        x, y = geo["cells"][j]
        px = out[int(oy + y + ph / 2), int(ox + x + pw / 2)]
        if j <= 2:
            assert np.allclose(px, clip[g][0, 0], atol=0.02)          # the frame captured there
        else:
            assert np.allclose(px, np.asarray(fx["paper"]) * 0.86, atol=0.02)   # not yet: a blank slot
    look = FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=clip.get, point_of=None, output=output)
    f = caps[2][1] + 11
    assert np.allclose(look.frame(f, clip[f]), out, atol=1e-5)        # asked out of order, as mk look asks


def test_an_aspect_table_lays_one_output_out_its_own_way():
    pl = plan({**booth(), "aspect": {"9x16": {"layout": [2, 2]}}})
    fx = pl["fx"][0]
    assert PC.booth_layout(FX.for_output(fx, "9x16"), (1080, 1920)) == (2, 2)
    assert PC.booth_layout(FX.for_output(fx, "16x9"), (1920, 1080)) == (4, 1)


def test_doodles_sit_round_the_edges_of_a_photo():
    fx = FX.normalize(booth(doodles=3, seed=2), DAWN)
    fx["index"] = 0
    for cell in range(4):
        ds = PC.booth_doodles(fx, cell, (100.0, 125.0), 1.2)
        assert len(ds) == 3 and all(d["shape"] in PC.STICKERS for d in ds)
        for d in ds:
            u, v = d["at"][0] / 100.0 - 0.5, d["at"][1] / 125.0 - 0.5
            assert 0.3 < math.hypot(u, v) < 0.45                       # away from the face in the middle
            assert d["t"] > 1.2 + fx["drop"]
    assert PC.booth_doodles(fx, 1, (100.0, 125.0), 1.2) == PC.booth_doodles(fx, 1, (100.0, 125.0), 1.2)


def test_photos_are_cut_to_the_cell_about_the_focus():
    pic = np.zeros((100, 200, 3), np.float32)
    pic[:, 150:] = 1.0                                                # the right quarter white
    left = PF.cover(pic, (40, 50), (0.1, 0.5))
    right = PF.cover(pic, (40, 50), (0.9, 0.5))                       # held against the right edge
    assert left.shape == (50, 40, 3) and left.max() == 0.0
    assert right.mean() == pytest.approx(50 / 80, abs=0.02)           # the crop's last 80 px: 50 of them white
