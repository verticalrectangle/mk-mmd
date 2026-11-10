"""Cell division ([[fx]] kind "mitosis", mkmmd.core.mitosis and mkmmd.mitosis): what the keys refuse, divisions paced by
their own length and capped, the cells' layout (two for one, pulled apart by the figure's own size, kept in the frame),
the picture (a copy of the figure in each cell, her place closed from the ground, a membrane round them), the same frame
in any order, and the GPU leaving the kind to the CPU."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import mitosis as MTF
from mkmmd import postgpu as PG
from mkmmd.core import framefx as FX
from mkmmd.core import mitosis as MT
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 160, 90


def mitosis(**kw):
    return {"kind": "mitosis", "from": 0.0, "to": 3.5, "objects": "Reisen*", **kw}


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def runner(pl, figure):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=lambda key, g: figure)


def scene():
    """A white figure 20 px wide and 30 tall in the middle of a dark ground, and its coverage."""
    m = np.zeros((H, W), np.float32)
    m[30:60, 70:90] = 1.0
    img = np.full((H, W, 3), 0.1, np.float32)
    img[30:60, 70:90] = 1.0
    return img, m


@pytest.mark.parametrize("spec, frag", [
    ({"kind": "mitosis", "from": 0.0, "to": 3.0}, "the figure that divides"),
    (mitosis(max=9), "max = 9: expected 0..6"),
    (mitosis(split=0.0), "split = 0.0: expected 0.1..10"),
    (mitosis(times=[1.0], lane="kick"), "one of them"),
    (mitosis(cells=4), "unknown keys ['cells']"),
    (mitosis(shot=""), "expected the name of a [[shot]]"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError) as e:
        plan(spec)
    assert frag in str(e.value)


def test_divisions_wait_for_one_another_and_stop_at_max():
    assert MT.divisions([1.0, 1.2, 1.7, 2.0, 5.0], 0.6, 3) == [1.0, 1.7, 5.0]     # 1.2 and 2.0 fall in a division
    assert MT.divisions([1.0, 2.0, 3.0], 0.5, 2) == [1.0, 2.0]
    fx = plan(mitosis(times=[1.0, 2.0], split=0.5))["fx"][0]
    assert MT.state(fx, 0.5, fx["times"])[:2] == (0, 0.0)
    gen, phase, since = MT.state(fx, 1.25, fx["times"])
    assert (gen, phase) == (0, pytest.approx(0.5)) and since == np.inf
    gen, phase, since = MT.state(fx, 2.8, fx["times"])
    assert (gen, phase, since) == (2, 0.0, pytest.approx(0.3))


def test_cells_double_part_by_the_figures_own_size_and_stay_in_the_frame():
    fx = plan(mitosis(tilt=0.0, gap=0.1, wobble=0.0))["fx"][0]
    box = (0.2, 0.4)
    at, scale, angle, stretch = MT.cells(fx, 0, 0.0, np.inf, box)
    assert len(at) == 1 and scale[0] == 1.0 and stretch[0] == 0.0
    at, scale, angle, stretch = MT.cells(fx, 1, 0.0, np.inf, box)
    assert len(at) == 2 and scale == pytest.approx(MT.SHRINK)
    assert abs(at[0, 0] - at[1, 0]) == pytest.approx(box[0] * MT.SHRINK * 1.1) and at[:, 1] == pytest.approx(0.0)
    mid = MT.cells(fx, 0, 0.5, np.inf, box)
    assert len(mid[0]) == 2 and mid[3][0] == pytest.approx(MT.PULL) and abs(mid[0][0, 0]) < abs(at[0, 0])   # pulled long
    at, scale, angle, stretch = MT.cells(fx, 4, 0.0, np.inf, box)
    assert len(at) == 16 and len({(round(x, 6), round(y, 6)) for x, y in at}) == 16
    k, shift = MT.fit(at, scale, stretch, box, (0.3, 0.5), (16 / 9, 1.0), 0.9)
    lo = np.array([0.3, 0.5]) + k * (at - 0.5 * np.array(box) * scale[:, None]).min(0) + shift
    hi = np.array([0.3, 0.5]) + k * (at + 0.5 * np.array(box) * scale[:, None]).max(0) + shift
    assert (lo >= 0.05 * np.array([16 / 9, 1.0]) - 1e-9).all() and (hi <= 0.95 * np.array([16 / 9, 1.0]) + 1e-9).all()
    assert MT.fit(*MT.cells(fx, 0, 0.0, np.inf, box)[:2], np.zeros(1), box, (0.9, 0.5), (16 / 9, 1.0), 0.9) == (1.0, pytest.approx([0, 0]))


def test_a_figure_divides_into_two_copies_and_her_place_closes():
    img, m = scene()
    spec = mitosis(times=[1.0], split=0.5, tilt=0.0, gap=0.5, alpha=0.0, back="picture", wobble=0.0, membrane="love")
    r = runner(plan(spec), m)
    before = r.frame(F0 + 15, img)
    assert np.allclose(before[30:60, 70:90], 1.0)                # one cell: she is where she was
    assert not np.allclose(before[26, 70:90], 0.1)               # with a membrane round her
    after = r.frame(F0 + 60, img)
    assert after[45, 80].max() < 0.3                             # her place is ground again
    cols = np.nonzero(after[45].min(-1) > 0.9)[0]                # and two copies stand to either side of it
    assert cols.min() < 75 and cols.max() > 85 and not ((cols > 76) & (cols < 84)).any()
    fresh = runner(plan(spec), m)
    assert np.array_equal(fresh.frame(F0 + 60, img), after) and np.array_equal(fresh.frame(F0 + 15, img), before)
    assert np.array_equal(runner(plan(dict(spec, strength=0.0)), m).frame(F0 + 60, img), img)
    onto = plan(dict(spec, back="#204060"))
    flat = runner(onto, m).frame(F0 + 60, img)
    assert np.allclose(flat[2, 2], onto["fx"][0]["back"]) and np.allclose(flat[45, 80], onto["fx"][0]["back"], atol=0.02)
    assert MTF.ground(img, m)[45, 80].max() < 0.2                # the ground closes from what is round her


def test_a_figure_from_a_shot_of_its_own_is_set_into_a_picture_she_is_not_in():
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}, {"name": "far", "plate": True}],
            "fx": [mitosis(shot="far", at=[0.25, 0.9], height=0.5, alpha=0.0, wobble=0.0)]}
    pl = TR.plan(data, FPS, F0, MOON)
    img, m = scene()                              # the cut's frame, and where the shot `far` has her
    green = np.zeros((H, W, 3), np.float32)
    green[..., 1] = 1.0                           # what `far` shows
    r = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                   figure_of=lambda key, g: m, plate_of=lambda shot, g: green)
    out = r.frame(F0 + 15, img)
    assert np.allclose(out[40:78, 28:52], [0.0, 1.0, 0.0], atol=0.02)      # feet at (0.25, 0.9), half the frame tall
    assert np.array_equal(out[28:62, 68:92], img[28:62, 68:92])            # the cut's frame is left as it is


def test_a_later_frame_reads_the_figure_on_the_spans_first_frame():
    pl = plan(mitosis())
    read = TR.fx_frames(pl, [F0 + 60])[F0]                                        # her box there sizes the cells
    assert [it["objects"] for it in read if it["kind"] == "figure"] == [["Reisen*"]]
    assert TR.fx_frames(pl, [F0]) == {}


def test_the_gpu_leaves_mitosis_to_the_cpu():
    img, m = scene()
    r = runner(plan(mitosis()), m)
    assert PG.ops(r, r.fxs[0], F0 + 3) is None
