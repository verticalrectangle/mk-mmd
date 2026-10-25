"""Pixel sort ([[fx]] kind "pixelsort", mkmmd.core.picturefx and mkmmd.picturefx): its keys, the sort of runs (only
eligible pixels move, never across a gap or a row, rising or falling), firing on hits and dying away, the figure mask and
the layer it asks for."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd import picturefx as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 48, 20


def plan(*fx):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def noise(seed=1):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def runner(pl, figure=None):
    return FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=(lambda key, g: figure) if figure is not None else None)


@pytest.mark.parametrize("spec, frag", [
    ({"by": "value"}, "none of luma"),
    ({"lo": 0.6, "hi": 0.4}, "hi must be above lo"),
    ({"lo": 0.5, "hi": 0.5}, "hi must be above lo"),
    ({"hi": 1.5}, "expected 0..1"),
    ({"axis": "z"}, "none of x, y"),
    ({"reverse": 1}, "reverse is true or false"),
    ({"mask": "figure"}, "needs the figure's `objects`"),
    ({"objects": "Reisen*"}, "give both or neither"),
    ({"mask": "body", "objects": "Reisen*"}, "none of figure, ground"),
    ({"min_db": -2}, "give the `lane`"),
    ({"times": [0.5], "lane": "snare"}, "one of them"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag.replace("[", r"\[").replace("(", r"\(")):
        FX.normalize({"kind": "pixelsort", "from": 0, "to": 1, **spec}, DAWN)


def test_runs_sort_inside_themselves_only():
    key = np.array([[0.9, 0.5, 0.1, 0.0, 0.8, 0.2, 0.6],
                    [0.3, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4]], np.float32)
    ok = np.array([[1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1]], bool)
    img = np.dstack([key, np.arange(14, dtype=np.float32).reshape(2, 7), np.zeros_like(key)])
    out = PF.sort_runs(img, key, ok)
    # row 0: two runs split by the gap (column 3 stays), each rising; row 1: one run, rising, not mixed with row 0
    assert out[0, :, 0].tolist() == pytest.approx([0.1, 0.5, 0.9, 0.0, 0.2, 0.6, 0.8])
    assert out[1, :, 0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert out[0, 3, 1] == 3.0                                    # the gap pixel did not move
    assert set(out[0, :, 1].tolist()) == set(range(7))            # a row keeps its own pixels
    down = PF.sort_runs(img, key, ok, reverse=True)
    assert down[0, :3, 0].tolist() == pytest.approx([0.9, 0.5, 0.1])
    assert down[1, :, 0].tolist() == pytest.approx([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def test_columns_sort_as_transposed_rows():
    img = noise(3)
    key = PF.sort_key(img, "luma")
    ok = (key > 0.3) & (key < 0.8)
    by_y = PF.sort_runs(img, key, ok, axis="y")
    by_x_t = PF.sort_runs(img.transpose(1, 0, 2), key.T, ok.T, axis="x").transpose(1, 0, 2)
    assert np.array_equal(by_y, by_x_t)
    col = by_y[:, 5]
    k = PF.sort_key(col[:, None, :], "luma")[:, 0]
    runs = np.split(np.arange(H), np.flatnonzero(np.diff(ok[:, 5].astype(int))) + 1)
    for r in runs:
        if ok[r[0], 5]:
            assert np.all(np.diff(k[r]) >= -1e-6)                 # each eligible run rises down the column


def test_keys_hue_and_saturation():
    img = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]], np.float32)
    hue, sat = PF.sort_key(img, "hue"), PF.sort_key(img, "sat")
    assert hue[0, :3] == pytest.approx([0.0, 1 / 3, 2 / 3], abs=1e-3)
    assert sat[0].tolist() == pytest.approx([1, 1, 1, 0])


def test_it_fires_on_hits_and_dies_away():
    pl = plan({"kind": "pixelsort", "from": 0.0, "to": 2.0, "times": [0.5], "decay": 0.1, "lo": 0.0, "hi": 1.0})
    fx = pl["fx"][0]
    img = noise()
    full = PF.sort_runs(img, PF.sort_key(img, "luma"), np.ones((H, W), bool))
    ffx = runner(pl)
    hit = FX.frame_of(0.5, FPS, F0)
    assert np.array_equal(ffx.frame(hit - 1, img), img)                       # nothing before the hit
    assert np.allclose(ffx.frame(hit, img), full)                             # full on it
    k = np.exp(-(3 / FPS) / 0.1)                                              # 3 frames on
    assert np.allclose(ffx.frame(hit + 3, img), img * (1 - k) + full * k, atol=1e-5)
    assert FX.uses_figure(fx) is False


def test_steady_without_hits_mixes_by_strength():
    pl = plan({"kind": "pixelsort", "from": 0.0, "to": 2.0, "strength": 0.25, "lo": 0.0, "hi": 1.0, "axis": "y"})
    img = noise(2)
    full = PF.sort_runs(img, PF.sort_key(img, "luma"), np.ones((H, W), bool), axis="y")
    assert np.allclose(runner(pl).frame(F0 + 10, img), img * 0.75 + full * 0.25, atol=1e-5)


@pytest.mark.parametrize("mask", ["figure", "ground"])
def test_the_figure_mask_keeps_the_sort_inside_or_outside_her(mask):
    pl = plan({"kind": "pixelsort", "from": 0.0, "to": 2.0, "lo": 0.0, "hi": 1.0, "mask": mask, "objects": "Reisen*"})
    fig = np.zeros((H, W), np.float32)
    fig[:, 10:30] = 1.0
    img = noise(4)
    out = runner(pl, fig).frame(F0 + 3, img)
    still = fig < 0.5 if mask == "figure" else fig > 0.5
    assert np.array_equal(out[still], img[still])
    assert not np.array_equal(out[~still], img[~still])
    dem = TR.demands(pl)
    assert {"kind": "figure", "key": FX.figure_key(pl["fx"][0]), "objects": ["Reisen*"], "shot": "a"} in dem[F0 + 3]
    assert FX.uses_figure(pl["fx"][0])


def test_unmasked_sort_asks_for_no_layer():
    pl = plan({"kind": "pixelsort", "from": 0.0, "to": 1.0})
    assert TR.demands(pl) == {}
