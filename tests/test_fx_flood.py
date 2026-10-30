"""A flood ([[fx]] kind "flood", mkmmd.core.fields and mkmmd.fields): its keys, its drops' timing (out over `grow`, full
for `hold`, back over `recede` or faded over `fade`), the colour's coverage over time (growing and never shrinking while
it grows, the whole frame at `grow`, shrinking as it recedes, gone after), a blotchy front that is not a circle, its
colours read drop by drop, a scene point as its centre, and a ground mask that keeps her in front."""
import re

import numpy as np
import pytest

from mkmmd import fields as FL
from mkmmd import framefx as FF
from mkmmd.core import fields as CF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 160, 90
LOVE = [round(c, 6) for c in PAL.srgb(DAWN["love"])]


def plan(*fx, size=(W, H)):
    data = {"project": {"duration": 6.0}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 6.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def runner(pl, figure=None, points=None, size=(W, H)):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None,
                      point_of=lambda key, g: (points or {})[key],
                      figure_of=(lambda key, g: figure) if figure is not None else None)


def flood(**kw):
    return {"kind": "flood", "from": 0.0, "to": 5.0, "times": [1.0], **kw}


@pytest.mark.parametrize("spec, frag", [
    ({"kind": "flood", "from": 0.0, "to": 2.0}, "give `times` (clip seconds) or `lane`"),
    (flood(grow=0), "grow = 0: the seconds the colour takes to fill the frame"),
    (flood(hold=-1), "hold = -1: expected at least 0"),
    (flood(color="nope"), "colour 'nope'"),
    (flood(soft=-2), "soft = -2: expected 0..400"),
    (flood(wobble=2), "wobble = 2: expected 0..1"),
    (flood(mask="ground"), "needs the figure's `objects`"),
    (flood(blend="screen"), "unknown keys ['blend']"),
    (flood(center=[0.5]), "expected [x, y]"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, DAWN)


def test_a_drop_grows_holds_and_recedes_or_fades():
    fx = FX.normalize(flood(grow=0.8, hold=0.4, recede=0.6), DAWN)
    share = [CF.flood_drops(fx, 1.0 + k * 0.05, [1.0])[0][1] for k in range(1, 17)]  # 1.05 .. 1.8: growing
    assert all(b > a for a, b in zip(share, share[1:]))
    assert CF.flood_drops(fx, 1.0, [1.0]) == []                                       # nothing yet on the drop itself
    assert share[-1] == pytest.approx(1.0)
    assert CF.flood_drops(fx, 2.1, [1.0]) == [(0, 1.0, 1.0)]                          # holding
    back = [CF.flood_drops(fx, 2.2 + k * 0.1, [1.0])[0][1] for k in range(5)]         # receding
    assert all(b < a for a, b in zip(back, back[1:]))
    assert CF.flood_drops(fx, 2.81, [1.0]) == [] and CF.flood_drops(fx, 0.99, [1.0]) == []
    faded = FX.normalize(flood(grow=0.5, hold=0.0, recede=0.0, fade=0.5), DAWN)
    assert CF.flood_drops(faded, 1.75, [1.0]) == [(0, 1.0, pytest.approx(0.5))]
    assert CF.flood_drops(faded, 2.01, [1.0]) == []
    stays = FX.normalize(flood(recede=0.0), DAWN)
    assert CF.flood_drops(stays, 4.9, [1.0]) == [(0, 1.0, 1.0)]


def _coverage(r, frame):
    """How much of each px the flood covers on `frame`: its colour laid over black and white, compared."""
    black = r.frame(frame, np.zeros((H, W, 3), np.float32))
    white = r.frame(frame, np.ones((H, W, 3), np.float32))
    return 1.0 - (white - black).mean(-1)


def test_the_colour_covers_more_and_more_then_all_then_less():
    pl = plan(flood(center=[0.3, 0.6], grow=0.8, hold=0.4, recede=0.6))
    r = runner(pl)
    grow = [_coverage(r, F0 + 30 + k) for k in range(25)]                 # 1.0 .. 1.8 s
    for a, b in zip(grow, grow[1:]):
        assert np.all(b >= a - 1e-5)                                       # never shrinks anywhere while it grows
    means = [float(c.mean()) for c in grow]
    assert means[0] < 0.02 and means[8] < means[16] < means[24]
    assert float(grow[-1].min()) > 0.999                                   # the whole frame at `grow`
    assert float(_coverage(r, F0 + 65).min()) > 0.999                      # and while it holds
    back = [float(_coverage(r, F0 + 67 + 3 * k).mean()) for k in range(6)]  # receding
    assert all(b < a for a, b in zip(back, back[1:]))
    assert float(_coverage(r, F0 + 86).max()) < 1e-6                       # gone


def test_the_front_is_blotchy_not_a_circle_and_is_its_colour():
    pl = plan(flood(color="love", grow=1.0))
    r = runner(pl)
    cov = _coverage(r, F0 + 36)                                            # 0.2 s in: a blob about the middle
    ys, xs = np.nonzero(cov > 0.5)
    d = np.hypot(xs - (W / 2 - 0.5), ys - (H / 2 - 0.5))
    edge = np.hypot(*np.nonzero((cov > 0.4) & (cov < 0.6))[::-1] - np.array([[W / 2 - 0.5], [H / 2 - 0.5]]))
    assert d.size > 20 and edge.std() / edge.mean() > 0.08                 # the front's distance varies round it
    full = r.frame(F0 + 70, np.zeros((H, W, 3), np.float32))
    assert np.allclose(full.reshape(-1, 3).mean(0), LOVE, atol=0.05)       # filled with its colour, a little uneven
    assert full[..., 0].std() > 0.002


def test_each_drop_takes_the_next_colour_over_the_last():
    pl = plan(flood(times=[0.5, 2.0], color=["love", "#2244ff"], grow=0.5, hold=0.5, recede=0.0))
    r = runner(pl)
    first = r.frame(F0 + 40, np.zeros((H, W, 3), np.float32))             # 1.33 s: the first, full
    second = r.frame(F0 + 80, np.zeros((H, W, 3), np.float32))            # 2.67 s: the second over it, full
    assert np.allclose(first.reshape(-1, 3).mean(0), LOVE, atol=0.05)
    assert second[..., 2].mean() > 0.8 and second[..., 0].mean() < 0.3


def test_a_scene_point_centres_it_and_asks_for_its_layer():
    pl = plan(flood(center={"point": "bone('head').head"}))
    key = FX.point_key(pl["fx"][0])
    assert any(it["kind"] == "point" for items in TR.demands(pl).values() for it in items)
    r = runner(pl, points={key: ((60.0, 40.0), None)})
    cov = _coverage(r, F0 + 33)
    ys, xs = np.nonzero(cov > 0.5)
    assert xs.size > 10 and abs(xs.mean() - 60.0) < 8 and abs(ys.mean() - 40.0) < 8


def test_a_ground_mask_keeps_her_in_front():
    m = np.zeros((H, W), np.float32)
    m[10:80, 70:90] = 1.0
    img = np.random.default_rng(5).random((H, W, 3)).astype(np.float32)
    out = runner(plan(flood(mask="ground", objects="Reisen*")), figure=m).frame(F0 + 70, img)
    assert np.array_equal(out[10:80, 70:90], img[10:80, 70:90])
    assert np.allclose(out[:, :60].reshape(-1, 3).mean(0), LOVE, atol=0.05)


def test_a_drop_before_the_span_is_already_on_its_way_at_its_start():
    pl = plan(flood(**{"from": 1.0, "to": 3.0}, times=[0.6], grow=0.2, hold=0.4, recede=0.6))
    r = runner(pl)
    assert float(_coverage(r, F0 + 30).min()) > 0.999                      # 1.0 s, the span's first frame: full
    back = [float(_coverage(r, F0 + 37 + 3 * k).mean()) for k in range(5)]  # 1.23 .. 1.63 s: drawing back
    assert all(b < a for a, b in zip(back, back[1:]))
    assert float(_coverage(r, F0 + 60).max()) < 1e-6                       # gone by 2.0 s


def test_every_output_fills_at_grow():
    fx = dict(FX.normalize(flood(), DAWN), index=0, f0=F0, f1=F0 + 150)
    for size in ((192, 108), (108, 192), (100, 100)):
        _, cov = FL.flood(fx, size, (size[0] * 0.8, size[1] * 0.2), [(0, 1.0, 1.0)], FL._Memo())
        assert float(cov.min()) > 0.999
