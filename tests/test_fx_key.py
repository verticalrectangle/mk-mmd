"""Colour keys pushed too far (mkmmd.core.colorkey: the entries checked, the tolerance over time, a stem's loudness;
mkmmd.colorkey: what shows through, each plane at its own tolerance, crushed steps, the trail of the frames before, the
figure kept clean) on synthetic frames."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import colorkey as CK
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 36
CREAM = (0.98, 0.96, 0.93)
PLUM = (0.34, 0.32, 0.47)
GREEN = (0.1, 0.6, 0.3)


def entries(*specs):
    proj = {"project": {"duration": 4.0}, "output": [{"name": "o", "size": [W, H]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(specs)}
    return TR.plan(proj, FPS, F0, DAWN)["fx"]


def key(**kw):
    return {"kind": "key", "from": 0.0, "to": 4.0, "color": list(CREAM), "back": list(GREEN), "soft": 0.0, **kw}


def runner(fxs, frames=None, figure=None, stem=None, hits=None):
    return FF.FrameFx(fxs, (W, H), fps=FPS, frame0=F0, frame_of=lambda g: (frames or {}).get(g),
                      point_of=None, figure_of=figure, stem_of=stem, hits=hits)


def flat(colour):
    return np.full((H, W, 3), colour, np.float32)


def halves():
    """Cream on the left, plum on the right."""
    img = flat(CREAM)
    img[:, W // 2:] = PLUM
    return img


def test_the_key_colour_shows_what_lies_behind_and_the_rest_stays_exactly():
    fx = entries(key(tolerance=0.05, gain=[1.0, 1.0, 1.0]))
    out = runner(fx).frame(F0 + 3, halves())
    assert np.allclose(out[:, : W // 2], GREEN, atol=1e-6)
    assert np.array_equal(out[:, W // 2:], halves()[:, W // 2:])


def test_each_plane_keys_at_its_own_tolerance_so_the_edge_comes_apart_in_fringes():
    near = np.array(CREAM, np.float32) * 0.9                      # 0.06 or so from the key colour
    d = float(np.sqrt(np.sum((near - CREAM) ** 2) / 3.0))
    fx = entries(key(tolerance=d, split=0.02))                    # red keys at d + 0.02, blue at d - 0.02
    out = runner(fx).frame(F0 + 3, flat(near))[0, 0]
    assert out[0] == pytest.approx(GREEN[0]) and out[1] == pytest.approx(GREEN[1])  # red and green keyed
    assert out[2] == pytest.approx(near[2])                       # blue kept: a fringe of its own colour


def test_bands_crush_the_soft_edge_of_the_key_into_hard_steps():
    from mkmmd import colorkey as CKF
    d = np.linspace(0.0, 0.4, 41, dtype=np.float32)[None, :]
    a = CKF.matte(d, 0.1, 0.2, [0.0, 0.0, 0.0], 2)[..., 0]
    assert set(np.round(a.ravel(), 6)) == {0.0, 0.5, 1.0}
    smooth = CKF.matte(d, 0.1, 0.2, [0.0, 0.0, 0.0], 0)[..., 0]
    assert len(set(np.round(smooth.ravel(), 6))) > 10


def test_a_burn_keeps_the_key_of_the_frames_before_fading_with_their_age():
    before = flat(PLUM)
    before[:, :8] = CREAM                                         # the key colour was on the left two frames ago
    now = flat(PLUM)
    fx = entries(key(tolerance=0.05, burn=2, fade=0.5))
    f = F0 + 10
    out = runner(fx, frames={f - 1: flat(PLUM), f - 2: before}).frame(f, now)
    expect = np.array(PLUM) + (np.array(GREEN) - np.array(PLUM)) * 0.25   # two frames back: fade ** 2
    assert np.allclose(out[:, :8], expect, atol=2.5 / 255)
    assert np.array_equal(out[:, 8:], now[:, 8:])


def test_a_ground_mask_leaves_the_figure_exactly_as_it_was_though_it_wears_the_key_colour():
    cov = np.zeros((H, W), np.float32)
    cov[10:20, 10:20] = 1.0
    fx = entries(key(tolerance=0.05, mask="ground", objects="Rin*"))
    img = flat(CREAM)
    out = runner(fx, figure=lambda k, g: cov).frame(F0 + 3, img)
    assert np.array_equal(out[10:20, 10:20], img[10:20, 10:20])
    assert np.allclose(out[0, 0], GREEN, atol=1e-6)


def test_the_key_widens_with_a_stem_s_loudness_and_on_the_hits():
    sr, n = 1000, 4000                                            # a stem silent for 2 s, then steadily loud
    x = np.zeros(n, np.float32)
    x[2000:] = 0.5
    fx = entries(key(tolerance=0.1, follow={"stem": "vocals", "amount": 0.3, "smooth": 0.0}))[0]
    r = runner([fx], stem=lambda tl, stem: (x, sr, 0.0))
    assert r.key_tolerance(fx, F0 + 15) == pytest.approx(0.1)     # 0.5 s: silent
    assert r.key_tolerance(fx, F0 + 90) == pytest.approx(0.4)     # 3 s: loud
    pumped = entries(key(tolerance=0.1, pump=0.2, decay=0.2, times=[1.0]))[0]
    p = runner([pumped])
    assert p.key_tolerance(pumped, F0 + 27) == pytest.approx(0.1)  # 0.9 s, before the hit
    assert p.key_tolerance(pumped, F0 + 31) > 0.25                 # just after it


def test_levels_run_from_a_stem_s_quiet_to_its_loud_passages():
    x = np.concatenate([np.zeros(3000), np.ones(3000)]).astype(np.float32)
    lv = CK.levels(x, 1000, 0.0, 0.0, 6.0, 10, 0.0, 24.0)
    assert lv.shape == (60,) and lv[:25].max() == 0.0 and lv[35:].min() == 1.0


def test_with_a_spread_the_key_opens_from_its_centre_as_far_as_the_tolerance_reaches():
    fx = entries(key(tolerance=0.3, spread=1.0, center=[0.25, 0.5]))
    out = runner(fx).frame(F0 + 3, flat(CREAM))
    cx, cy = 0.25 * W - 0.5, 0.5 * H - 0.5
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - cx, yy - cy) / H                             # frame heights from the centre
    keyed = np.all(np.abs(out - GREEN) < 1e-6, axis=-1)
    assert keyed[r < 0.28].all() and not keyed[r > 0.32].any()    # a disc of 0.3 / spread frame heights


@pytest.mark.parametrize("bad, says", [
    ({"back": None}, "give one of"),
    ({"clip": "x.mp4"}, "not clip and back"),
    ({"pump": 0.2}, "give `times` or a `lane`"),
    ({"times": [1.0]}, "give the `pump`"),
    ({"rate": 2.0}, "are a clip's"),
    ({"follow": {"stem": "kazoo"}}, "follow stem"),
    ({"center": [0.5, 0.5]}, "give the spread"),
])
def test_a_key_that_cannot_be_drawn_is_refused_with_a_reason(bad, says):
    with pytest.raises(TR.TransitionError, match=says):
        entries(key(**bad))
